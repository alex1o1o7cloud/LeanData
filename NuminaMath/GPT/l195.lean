import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Complex.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Trigonometry
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics.Factorial
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.Init.Data.Nat.Lemmas
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Cardinal
import Mathlib.Tactic
import Real
import data.real.basic

namespace ram_krish_task_completion_l195_195135

/-!
  Given:
  1. Ram's efficiency (R) is half of Krish's efficiency (K).
  2. Ram can complete the task alone in 24 days.

  To Prove:
  Ram and Krish will complete the task together in 8 days.
-/

theorem ram_krish_task_completion {R K : ℝ} (hR : R = 1 / 2 * K)
  (hRAMalone : R ≠ 0) (hRAMtime : 24 * R = 1) :
  1 / (R + K) = 8 := by
  sorry

end ram_krish_task_completion_l195_195135


namespace sum_complex_binom_l195_195691

noncomputable def x := (2 * complex.i) / (1 - complex.i)

theorem sum_complex_binom :
  ∑ k in finset.range 2016, (nat.choose 2016 (k + 1)) * (x ^ (k + 1)) = 0 :=
  sorry

end sum_complex_binom_l195_195691


namespace area_of_triangle_ABC_l195_195367

theorem area_of_triangle_ABC :
  ∀ (A B C M P : Type) [Field A] [Field B] [Field C] [Field M] [Field P],
  (is_median AM ∧ is_median BP) ∧
  (angle APB = angle BMA) ∧
  (cos_angle ACB = 0.8) ∧
  (length BP = 1) →
  (area_triangle A B C = 2 / 3) :=
by 
  intros A B C M P _ _ _ _ _
  assume h
  sorry

end area_of_triangle_ABC_l195_195367


namespace total_cost_of_pencils_l195_195434

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end total_cost_of_pencils_l195_195434


namespace triangle_determinant_zero_l195_195400

theorem triangle_determinant_zero (P Q R : ℝ) (h : P + Q + R = π) :
  det ![
    [(Real.cos P)^2, Real.tan P, 1],
    [(Real.cos Q)^2, Real.tan Q, 1],
    [(Real.cos R)^2, Real.tan R, 1]
  ] = 0 :=
by sorry

end triangle_determinant_zero_l195_195400


namespace find_lambda_find_T_l195_195319

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2

-- Define the function F
def F (x : ℝ) (λ : ℝ) : ℝ := (g x)^2 - λ * (f x)

-- Problem Statement 1
theorem find_lambda (λ : ℝ) (x : ℝ) (h1 : x ∈ Set.Ici (1 / 8)) (h2 : F x λ = -16) : λ = -32 ∨ λ = 8 :=
sorry

-- Define the inequality and the function h for Problem Statement 2
def ineq (T : ℝ) (x : ℝ) : Prop := 2^(3 - g (Real.sqrt x)) - 2^(f (x^2)) ≤ Real.log T
def h (x : ℝ) : ℝ := -x^2 + x

-- Problem Statement 2
theorem find_T (T : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / 8) 2 → ¬(ineq T x)) ↔ (T > 0 ∧ T < 1 / Real.exp 2) :=
sorry

end find_lambda_find_T_l195_195319


namespace boys_at_reunion_l195_195564

theorem boys_at_reunion (n : ℕ) (h : n * (n - 1) = 56) : n = 8 :=
sorry

end boys_at_reunion_l195_195564


namespace pentagonal_number_formula_l195_195960

def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n + 1)) / 2

theorem pentagonal_number_formula (n : ℕ) :
  pentagonal_number n = (n * (3 * n + 1)) / 2 :=
by
  sorry

end pentagonal_number_formula_l195_195960


namespace flight_duration_l195_195377

theorem flight_duration :
  ∀ (h m : ℕ),
  3 * 60 + 42 = 15 * 60 + 57 →
  0 < m ∧ m < 60 →
  h + m = 18 :=
by
  intros h m h_def hm_bound
  sorry

end flight_duration_l195_195377


namespace min_max_values_of_f_l195_195455

noncomputable def f (x : ℝ) : ℝ := -2 * x + 1

theorem min_max_values_of_f :
  let min_val := -3
  let max_val := 5
  (∀ x ∈ set.Icc (-2 : ℝ) 2, min_val ≤ f x) ∧ (∀ x ∈ set.Icc (-2 : ℝ) 2, f x ≤ max_val)
:= by
  -- Placeholder for the proof
  sorry

end min_max_values_of_f_l195_195455


namespace linda_savings_fraction_l195_195009

theorem linda_savings_fraction (savings tv_cost : ℝ) (h1 : savings = 960) (h2 : tv_cost = 240) : (savings - tv_cost) / savings = 3 / 4 :=
by
  intros
  sorry

end linda_savings_fraction_l195_195009


namespace fC_is_increasing_l195_195593

-- Define the functions A, B, C, and D
def fA (x : ℝ) := -2 * x + 1
def fB (x : ℝ) := -2 / x
def fC (x : ℝ) := 2 * x
def fD (x : ℝ) := x^2

-- Define what it means for a function to be increasing on ℝ
def is_increasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

-- Prove that fC is the increasing function on ℝ
theorem fC_is_increasing : is_increasing fC := sorry

end fC_is_increasing_l195_195593


namespace number_of_solutions_is_6_l195_195996

theorem number_of_solutions_is_6 :
  (∃ (a : finset (ℤ × ℤ)), a.card = 6 ∧ ∀ (x, y) ∈ a, x^4 + y^2 = 4y) :=
sorry

end number_of_solutions_is_6_l195_195996


namespace Q_eval_at_1_l195_195261

noncomputable def Q (x : ℚ) : ℚ := x^4 - 4*x^2 + 16 

theorem Q_eval_at_1 : Q 1 = 13 := 
by 
  unfold Q
  calc 
    1^4 - 4*1^2 + 16 = 1 - 4 + 16 : by ring
    ... = 13 : by norm_num

end Q_eval_at_1_l195_195261


namespace parallelogram_area_l195_195066

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (hb : b = 15) (hh : h = 2 * b) (hA : A = b * h) : A = 450 := 
by
  rw [hb, hh] at hA
  rw [hA]
  sorry

end parallelogram_area_l195_195066


namespace derived_income_is_correct_l195_195850

variable (market_value : ℝ) (investment : ℝ) (brokerage_fee : ℝ) (interest_rate : ℝ)

def effective_market_value (market_value : ℝ) (brokerage_fee : ℝ) : ℝ :=
  market_value - (market_value * brokerage_fee / 100)

def face_value (investment : ℝ) (effective_market_value : ℝ) : ℝ :=
  (investment / effective_market_value) * 100

def income (face_value : ℝ) (interest_rate : ℝ) : ℝ :=
  (face_value * interest_rate) / 100

theorem derived_income_is_correct :
  market_value = 96.97222222222223 →
  investment = 7000 →
  brokerage_fee = 0.25 →
  interest_rate = 10.5 →
  income (face_value investment (effective_market_value market_value brokerage_fee)) interest_rate = 759.50 := 
by
  intros h1 h2 h3 h4
  sorry

end derived_income_is_correct_l195_195850


namespace triangle_max_area_l195_195859

-- Define the angles and sides of triangle ABC
variables {A B C : ℝ}
variables {a b c : ℝ}
variable (ABC : Triangle a b c A B C)

-- Given conditions
variable (h1 : a = 2) -- side a is 2
variable (h2 : (a - b + c) / c = b / (a + b - c)) -- given equation

-- Prove the maximum area is √3
theorem triangle_max_area (ABC : Triangle a b c A B C) (h1 : a = 2) (h2 : (a - b + c) / c = b / (a + b - c)) :
  ∃ (S : ℝ), S = sqrt 3 :=
sorry

end triangle_max_area_l195_195859


namespace parabola_y_relation_l195_195779

theorem parabola_y_relation (m : ℝ) (y1 y2 y3 : ℝ) :
  (y1 = 3 * (-2 + 1)^2 + 4 * m) →
  (y2 = 3 * (1 + 1)^2 + 4 * m) →
  (y3 = 3 * (2 + 1)^2 + 4 * m) →
  y1 < y2 ∧ y2 < y3 :=
by {
  intros h1 h2 h3,
  sorry
}

end parabola_y_relation_l195_195779


namespace solve_for_x_l195_195832

theorem solve_for_x : 
  ∃ x : ℝ, (x^2 + 6 * x + 8 = -(x + 2) * (x + 6)) ∧ (x = -2 ∨ x = -5) :=
sorry

end solve_for_x_l195_195832


namespace tangent_product_independence_l195_195586

theorem tangent_product_independence
  (circle : Type) [NormedAddCommGroup circle]
  (M N : circle) -- MN is the diameter of the circle
  (tangent : circle → Prop)
  (tangent_M : tangent M) -- Tangent to the circle at point M
  (A B : circle)
  (chord_AB : Prop) -- AB is a chord parallel to MN
  (parallel : Prop) (h_parallel : parallel) -- AB is parallel to MN
  (P Q : circle)
  (NA NB : circle → Prop) -- Lines NA and NB
  (NA_A : NA A) (NB_B : NB B)
  (intersection_P : tangent M ∧ NA A → P)
  (intersection_Q : tangent M ∧ NB B → Q)
  : (dist M P * dist M Q = dist M N ^ 2) :=
sorry

end tangent_product_independence_l195_195586


namespace problem_statement_l195_195725

theorem problem_statement (a b c d : ℤ) (h1 : a - b = -3) (h2 : c + d = 2) : (b + c) - (a - d) = 5 :=
by
  -- Proof steps skipped.
  sorry

end problem_statement_l195_195725


namespace belle_rawhide_bones_per_evening_l195_195607

theorem belle_rawhide_bones_per_evening 
  (cost_rawhide_bone : ℝ)
  (cost_dog_biscuit : ℝ)
  (num_dog_biscuits_per_evening : ℕ)
  (total_weekly_cost : ℝ)
  (days_per_week : ℕ)
  (rawhide_bones_per_evening : ℕ)
  (h1 : cost_rawhide_bone = 1)
  (h2 : cost_dog_biscuit = 0.25)
  (h3 : num_dog_biscuits_per_evening = 4)
  (h4 : total_weekly_cost = 21)
  (h5 : days_per_week = 7)
  (h6 : rawhide_bones_per_evening * cost_rawhide_bone * (days_per_week : ℝ) = total_weekly_cost - num_dog_biscuits_per_evening * cost_dog_biscuit * (days_per_week : ℝ)) :
  rawhide_bones_per_evening = 2 := 
sorry

end belle_rawhide_bones_per_evening_l195_195607


namespace tetrahedron_volume_l195_195194

theorem tetrahedron_volume (r : ℝ) : 
  ∃ V : ℝ, V = (r^3 / 3) * (2 * real.sqrt 3 + real.sqrt 2)^3 :=
sorry

end tetrahedron_volume_l195_195194


namespace writing_numbers_finite_l195_195037

theorem writing_numbers_finite :
  ∀ (a : ℕ → ℕ), (∀ n, a n > 0) →
  (∀ n, ∃ k : ℕ → ℕ, a (n + 1) ∉ set.range (λ k, ∑ i in finset.range n, (k i) * (a i))) →
  ∃ N, ∀ n ≥ N, false :=
by sorry

end writing_numbers_finite_l195_195037


namespace inverse_proposition_false_l195_195454

-- Definitions for the conditions
def congruent (A B C D E F: ℝ) : Prop := 
  A = D ∧ B = E ∧ C = F

def angles_equal (α β γ δ ε ζ: ℝ) : Prop := 
  α = δ ∧ β = ε ∧ γ = ζ

def original_proposition (A B C D E F α β γ : ℝ) : Prop :=
  congruent A B C D E F → angles_equal α β γ A B C

-- The inverse proposition
def inverse_proposition (α β γ δ ε ζ A B C D E F : ℝ) : Prop :=
  angles_equal α β γ δ ε ζ → congruent A B C D E F

-- The main theorem: the inverse proposition is false
theorem inverse_proposition_false (α β γ δ ε ζ A B C D E F : ℝ) :
  ¬(inverse_proposition α β γ δ ε ζ A B C D E F) := by
  sorry

end inverse_proposition_false_l195_195454


namespace student_correct_answers_l195_195587

theorem student_correct_answers 
  (total_questions : ℕ)
  (points_correct : ℤ)
  (points_incorrect : ℤ)
  (student_attempted_all : total_questions = 26)
  (total_score : ℤ)
  (student_score_zero : total_score = 0) :
  ∃ x : ℕ, 
  x ≤ total_questions ∧
  8 * (x : ℤ) - 5 * ((total_questions - x : ℕ) : ℤ) = total_score ∧
  x = 10 :=
by {
  use 10,
  split,
  { exact (nat.le_refl 26) },
  split,
  { simp,
    norm_num },
  { refl }
}

end student_correct_answers_l195_195587


namespace larger_number_is_A_l195_195138

def HCF : ℕ := 120
def factor1 : ℕ := 13
def factor2 : ℕ := 17
def factor3 : ℕ := 23
def LCM : ℕ := HCF * factor1 * factor2 * factor3
def A : ℕ := HCF * (factor1 * factor2)
def B : ℕ := HCF * factor3

theorem larger_number_is_A : A > B :=
by {
  have hA : A = 120 * (13 * 17) := rfl,
  have hB : B = 120 * 23 := rfl,
  have hA_val : A = 26520,
  { calc A = 120 * (13 * 17) : rfl
       ... = 120 * 221 : by norm_num
       ... = 26520 : by norm_num },
  have hB_val : B = 2760,
  { calc B = 120 * 23 : rfl
       ... = 2760 : by norm_num },
  show 26520 > 2760,
  from nat.lt_of_sub_pos (by norm_num)
}

end larger_number_is_A_l195_195138


namespace major_axis_length_of_ellipse_l195_195184

theorem major_axis_length_of_ellipse
  (tangent_x_axis : ∀ (x : ℝ), (x, 0) ∉ ellipse)
  (tangent_y_axis : ∀ (y : ℝ), (0, y) ∉ ellipse)
  (foci_ellipse : ∃ (f1 f2 : ℝ × ℝ), f1 = (3, -4 + 2*Real.sqrt 2) ∧ f2 = (3, -4 - 2*Real.sqrt 2)) :
  major_axis_length ellipse = 8 :=
sorry

end major_axis_length_of_ellipse_l195_195184


namespace num_ways_to_plus_at_top_l195_195743

def cell (val : Bool) : Int :=
  if val then 1 else -1

-- Function to get the top cell value based on the bottom row
def pyramid_top (a b c d e : Bool) : Int :=
  let row2 := [a != b, b != c, c != d, d != e]
  let row3 := [(row2[0] = row2[1]), (row2[1] = row2[2]), (row2[2] = row2[3])]
  let row4 := [(row3[0] = row3[1]), (row3[1] = row3[2])]
  if row4[0] = row4[1] then 1 else -1

theorem num_ways_to_plus_at_top :
  (∃ s : Finset (Bool × Bool × Bool × Bool × Bool),
    (∀ (a b c d e : Bool), 
       (a, b, c, d, e) ∈ s → (pyramid_top a b c d e = 1)) ∧ 
    s.card = 11) :=
  sorry

end num_ways_to_plus_at_top_l195_195743


namespace solve_for_x_l195_195430

theorem solve_for_x (x : ℝ) :
  (1 / 4)^(2 * x + 8) = (16)^(2 * x + 5) ↔ x = -3 :=
by sorry

end solve_for_x_l195_195430


namespace f_value_5pi_over_3_l195_195976

noncomputable def f : ℝ → ℝ :=
sorry -- We need to define the function according to the given conditions but we skip this for now.

lemma f_property_1 (x : ℝ) : f (-x) = -f x :=
sorry -- as per the condition f(-x) = -f(x)

lemma f_property_2 (x : ℝ) : f (x + π/2) = f (x - π/2) :=
sorry -- as per the condition f(x + π/2) = f(x - π/2)

lemma f_property_3 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π/2) : f x = Real.sin x :=
sorry -- as per the condition f(x) = sin(x) for x in [0, π/2]

theorem f_value_5pi_over_3 : f (5 * π / 3) = - (Real.sin (π / 3)) :=
by
  -- The exact proof steps are omitted; this involves properties of periodicity and the odd function nature
  sorry

end f_value_5pi_over_3_l195_195976


namespace number_of_ways_to_choose_roles_l195_195352

-- Define the problem setup
def friends := Fin 6
def cooks (maria : Fin 1) := {f : Fin 6 | f ≠ maria}
def cleaners (cooks : Fin 6 → Prop) := {f : Fin 6 | ¬cooks f}

-- The number of ways to select one additional cook from the remaining friends
def chooseSecondCook : ℕ := Nat.choose 5 1  -- 5 ways

-- The number of ways to select two cleaners from the remaining friends
def chooseCleaners : ℕ := Nat.choose 4 2  -- 6 ways

-- The final number of ways to choose roles
theorem number_of_ways_to_choose_roles (maria : Fin 1) : 
  let total_ways : ℕ := chooseSecondCook * chooseCleaners
  total_ways = 30 := sorry

end number_of_ways_to_choose_roles_l195_195352


namespace compare_abc_l195_195295

def a : ℝ := 0.6 ^ 0.4
def b : ℝ := 0.4 ^ 0.6
def c : ℝ := 0.4 ^ 0.4

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l195_195295


namespace no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l195_195765

theorem no_triangle_sum_of_any_two_angles_lt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) :=
by
  sorry

theorem no_triangle_sum_of_any_two_angles_gt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120) :=
by
  sorry

end no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l195_195765


namespace james_bike_ride_l195_195372

theorem james_bike_ride :
  ∃ x t : ℝ, 
    let first_hour_distance := x,
        second_hour_distance := 12,
        third_hour_distance := 1.25 * second_hour_distance,
        total_distance := first_hour_distance + second_hour_distance + third_hour_distance in
    1.20 * first_hour_distance = second_hour_distance ∧
    third_hour_distance = 1.25 * second_hour_distance ∧
    total_distance = 37 ∧
    t = 3 := 
by
  sorry

end james_bike_ride_l195_195372


namespace wheel_travel_distance_l195_195943

theorem wheel_travel_distance (r : ℝ) (n : ℕ) (h1 : r = 2) (h2 : n = 3) : 
  let circumference := 2 * Real.pi * r in
  let distance := circumference * n in
  distance = 12 * Real.pi := 
by 
  sorry

end wheel_travel_distance_l195_195943


namespace least_product_of_distinct_primes_over_30_l195_195505

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l195_195505


namespace expected_difference_tea_and_coffee_l195_195946

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop := 
  n = 4 ∨ n = 6

theorem expected_difference_tea_and_coffee : 
  let days_in_year := 365
  let probability_tea := 4 /  7
  let probability_coffee := 2 / 7
  let expected_tea_days := probability_tea * days_in_year
  let expected_coffee_days := probability_coffee * days_in_year
  let expected_difference := expected_tea_days - expected_coffee_days
  (expected_difference : ℝ) ≈ 104 :=
by
  -- Proving the expected difference is approximately 104
  sorry

end expected_difference_tea_and_coffee_l195_195946


namespace initially_calculated_average_height_l195_195839

/-- Suppose the average height of 20 students was initially calculated incorrectly. Later, it was found that one student's height 
was incorrectly recorded as 151 cm instead of 136 cm. Given the actual average height of the students is 174.25 cm, prove that the 
initially calculated average height was 173.5 cm. -/
theorem initially_calculated_average_height
  (initial_avg actual_avg : ℝ)
  (num_students : ℕ)
  (incorrect_height correct_height : ℝ)
  (h_avg : actual_avg = 174.25)
  (h_students : num_students = 20)
  (h_incorrect : incorrect_height = 151)
  (h_correct : correct_height = 136)
  (h_total_actual : num_students * actual_avg = num_students * initial_avg + incorrect_height - correct_height) :
  initial_avg = 173.5 :=
by
  sorry

end initially_calculated_average_height_l195_195839


namespace find_m_l195_195569

noncomputable def g : ℤ → ℤ
| n := if n % 2 = 1 then n + 5 else n / 3

theorem find_m : ∃ m : ℤ, odd m ∧ g (g (g m)) = 14 ∧ m = 121 := by
  sorry

end find_m_l195_195569


namespace trivia_team_l195_195866

theorem trivia_team (total_students groups students_per_group students_not_picked : ℕ) (h1 : total_students = 65)
  (h2 : groups = 8) (h3 : students_per_group = 6) (h4 : students_not_picked = total_students - groups * students_per_group) :
  students_not_picked = 17 :=
sorry

end trivia_team_l195_195866


namespace picture_distance_l195_195110

theorem picture_distance 
  (wall_width : ℝ) (pic_width : ℝ) (space_between_pics : ℝ) 
  (num_pics : ℕ) (is_centered : Bool) 
  (h_wall : wall_width = 30) 
  (h_pic : pic_width = 4) 
  (h_space : space_between_pics = 1) 
  (h_num_pics : num_pics = 2) 
  (h_centered : is_centered):
  let total_pics_width := num_pics * pic_width + (num_pics - 1) * space_between_pics
  let x := (wall_width - total_pics_width) / 2
  is_centered = true -> x = 10.5 := 
by {
  simp [wall_width, pic_width, space_between_pics, num_pics, is_centered, h_wall, h_pic, h_space, h_num_pics, h_centered],
  sorry
}

end picture_distance_l195_195110


namespace equation_of_line_AB_minimum_AF_BF_l195_195703

-- Given parabola and point P (-2, 2)
def parabola (x y : ℝ) : Prop := x^2 = 4 * y
def point_P := (-2 : ℝ, 2 : ℝ)

-- Conditions for points A and B and the line passing through them
variables (A B : ℝ × ℝ) (l : ℝ → ℝ) 

-- Line passes through point P and is such that P is the midpoint of A and B
def line_through_P_and_midpoint : Prop := 
  (l -2 = 2) ∧ (A.1 + B.1) / 2 = -2 ∧ (A.2 + B.2) / 2 = 2

-- Proving the equation of line AB when P is the midpoint of AB
theorem equation_of_line_AB (hA : parabola A.1 A.2) (hB : parabola B.1 B.2) (h_line : line_through_P_and_midpoint A B l) : 
  ∃ a b c, (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, (parabola x y → a * x + b * y + c = 0) :=
sorry

-- Proving the minimum value of |AF| • |BF|
noncomputable def focus_F : ℝ × ℝ := (0, 1)

def distance (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem minimum_AF_BF (hA : parabola A.1 A.2) (hB : parabola B.1 B.2) (h_line : line_through_P_and_midpoint A B l) : 
  ∃ k : ℝ, k = -3 / 4 → |distance (A, focus_F)| * |distance (B, focus_F)| = 9 / 2 :=
sorry

end equation_of_line_AB_minimum_AF_BF_l195_195703


namespace vehicles_assembled_eq_floor_div_2_l195_195353

-- Define the conditions
def is_vehicle_assembled_every_2_hours : Prop := 
  ∀ t, t ∈ [0, 2] → y = 0

def production_duration : Prop := 
  ∀ t, t ∈ [0, 8]

-- Define the goal to prove
def correct_function_expression : Prop := 
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 8) → y = ⌊x / 2⌋

-- The statement of the theorem
theorem vehicles_assembled_eq_floor_div_2 (x : ℝ) (h1 : is_vehicle_assembled_every_2_hours) (h2 : production_duration) : correct_function_expression := 
  sorry

end vehicles_assembled_eq_floor_div_2_l195_195353


namespace area_of_square_l195_195583

theorem area_of_square : 
  (∃ x₁ x₂ : ℝ, (3 = -x₁^2 + 2*x₁ + 4) ∧ (3 = -x₂^2 + 2*x₂ + 4) ∧ (y - 3 = (λ x, -x^2 + 2*x + 4)) ∧ (x₁ - x₂ = 2*sqrt(2)) ∧ (x₂ - x₁ = 2*sqrt(2))) → 
  ∃ a : ℝ, a = 2 * sqrt(2) ∧ a^2 = 8 :=
by
  sorry

end area_of_square_l195_195583


namespace balloon_permutations_l195_195234

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l195_195234


namespace find_magnitude_b_l195_195709

open Real

-- Definitions based on the given conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (t : ℝ) : ℝ × ℝ := (2, t)

-- Condition that a • b = 0
def dot_product_zero (t : ℝ) : Prop := (fst vector_a) * (fst (vector_b t)) + (snd vector_a) * (snd (vector_b t)) = 0

-- The magnitude of vector b
def magnitude_b (t : ℝ) : ℝ := sqrt ((fst (vector_b t))^2 + (snd (vector_b t))^2)

-- The main theorem using the conditions
theorem find_magnitude_b (t : ℝ) (h : dot_product_zero t) : magnitude_b t = sqrt 5 :=
by
  sorry

end find_magnitude_b_l195_195709


namespace sum_of_7_more_likely_than_sum_of_8_l195_195476

noncomputable def probability_sum_equals_seven : ℚ := 6 / 36
noncomputable def probability_sum_equals_eight : ℚ := 5 / 36

theorem sum_of_7_more_likely_than_sum_of_8 :
  probability_sum_equals_seven > probability_sum_equals_eight :=
by 
  sorry

end sum_of_7_more_likely_than_sum_of_8_l195_195476


namespace rotation_180_produces_result_l195_195464

def v : ℝ × ℝ × ℝ := (2, 2, 1)

def rotated_180_deg (u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-u.1, -u.2, -u.3)

theorem rotation_180_produces_result :
  rotated_180_deg v = (-2, -2, -1) :=
by
  sorry

end rotation_180_produces_result_l195_195464


namespace coefficient_of_x3_in_expansion_of_2_plus_x_5_l195_195841

theorem coefficient_of_x3_in_expansion_of_2_plus_x_5 :
  ( ∑ r in finset.range 6, nat.choose 5 r * 2 ^ (5 - r) * (x : ℤ) ^ r).coeff 3 = 40 := by
sorry

end coefficient_of_x3_in_expansion_of_2_plus_x_5_l195_195841


namespace cats_asleep_l195_195090

theorem cats_asleep
  (total_cats : ℕ) (awake_cats : ℕ) (asleep_cats : ℕ) 
  (h1 : total_cats = 98) (h2 : awake_cats = 6) 
  (h3 : asleep_cats = total_cats - awake_cats) : asleep_cats = 92 :=
by
  rw [h1, h2] at h3
  exact h3.symm

end cats_asleep_l195_195090


namespace find_a_l195_195306

-- Define the center of the circle
def center_of_circle : ℝ × ℝ := (3, 1)

-- Define the line equation
def line (a : ℝ) : ℝ → ℝ → Prop := λ x y, x + a * y - 1 = 0

-- Define the distance from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ := 
  (abs (p.1 + a * p.2 - 1)) / (sqrt (1 + a^2))

-- Given the circle x^2 + y^2 - 6x - 2y + 3 = 0 and the distance from its center 
-- to the line x + ay - 1 = 0 is 1, prove that a = -3/4
theorem find_a (h : distance_point_to_line center_of_circle a = 1) : a = -3/4 := 
by {
  sorry
}

end find_a_l195_195306


namespace painting_price_increase_l195_195457

theorem painting_price_increase 
( original_price : ℝ ) 
( final_price : ℝ )
( a x : ℝ )
( h1 : final_price = 1.0625 * original_price )
( h2 : x / 100 = 0.25 ) :
x = 25 :=
by
  have h : 1 + x / 100 = 1.25,
  { rw [←div_eq_mul_one_div, h2], simp },
  sorry

end painting_price_increase_l195_195457


namespace nada_house_size_l195_195424

variable (N : ℕ) -- N represents the size of Nada's house

theorem nada_house_size :
  (1000 = 2 * N + 100) → (N = 450) :=
by
  intro h
  sorry

end nada_house_size_l195_195424


namespace trajectory_of_point_l195_195575

theorem trajectory_of_point (x y k : ℝ) (hx : x ≠ 0) (hk : k ≠ 0) (h : |y| / |x| = k) : y = k * x ∨ y = -k * x :=
by
  sorry

end trajectory_of_point_l195_195575


namespace infinite_solutions_sum_of_squares_l195_195823

def is_sum_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

theorem infinite_solutions_sum_of_squares :
  ∃∞ (n : ℤ), (is_sum_of_squares n ∧ is_sum_of_squares (n + 1) ∧ is_sum_of_squares (n + 2)) := 
by
  sorry

end infinite_solutions_sum_of_squares_l195_195823


namespace solve_for_x_l195_195429

theorem solve_for_x (x : ℝ) (h : 7 - 2 * x = -3) : x = 5 := by
  sorry

end solve_for_x_l195_195429


namespace campers_morning_count_l195_195432

theorem campers_morning_count (afternoon_count : ℕ) (additional_morning : ℕ) (h1 : afternoon_count = 39) (h2 : additional_morning = 5) :
  afternoon_count + additional_morning = 44 :=
by
  sorry

end campers_morning_count_l195_195432


namespace ways_to_append_digit_divisible_by_3_l195_195469

-- Define a function that takes a digit and checks if it can make the number divisible by 3
def is_divisible_by_3 (n : ℕ) (d : ℕ) : Bool :=
  (n * 10 + d) % 3 == 0

-- Theorem stating that there are 4 ways to append a digit to make the number divisible by 3
theorem ways_to_append_digit_divisible_by_3 
  (n : ℕ) 
  (divisible_by_9_conditions : (n * 10 + 0) % 9 = 0 ∧ (n * 10 + 9) % 9 = 0) : 
  ∃ (ds : Finset ℕ), ds.card = 4 ∧ ∀ d ∈ ds, is_divisible_by_3 n d :=
  sorry

end ways_to_append_digit_divisible_by_3_l195_195469


namespace six_coins_value_not_90_l195_195056

theorem six_coins_value_not_90 :
  ¬∃ (n d h : ℕ), n + d + h = 6 ∧ 5 * n + 10 * d + 50 * h = 90 :=
begin
  -- Problem statement asserts that there is no combination of nickels (n), dimes (d) and half-dollars (h)
  -- such that the total number of coins is 6 and their combined value is 90 cents.
  sorry
end

end six_coins_value_not_90_l195_195056


namespace sum_of_consecutive_odds_exactly_3_ways_l195_195333

theorem sum_of_consecutive_odds_exactly_3_ways:
  {N : ℕ} → (N < 100) → (∃ m k : ℕ, N = k * (2 * m + k)) → (count (λ N, ∃ a_1 a_2 a_3 : ℕ, 
  ∃ (sum1 sum2 sum3 : finset ℕ), 
  (N = finset.sum sum1 id) ∧ (∀ x ∈ sum1, x ∈ {odd n : ℕ | n % 2 = 1 ∧ 0 < n}) ∧
  (N = finset.sum sum2 id) ∧ (∀ x ∈ sum2, x ∈ {odd n : ℕ | n % 2 = 1 ∧ 0 < n}) ∧
  (N = finset.sum sum3 id) ∧ (∀ x ∈ sum3, x ∈ {odd n : ℕ | n % 2 = 1 ∧ 0 < n})) = 5 :=
begin
  sorry
end

end sum_of_consecutive_odds_exactly_3_ways_l195_195333


namespace minimum_value_l195_195662

noncomputable def min_value_b_plus_4_over_a (a : ℝ) (b : ℝ) :=
  b + 4 / a

theorem minimum_value (a : ℝ) (b : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x : ℝ, x > 0 → (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  min_value_b_plus_4_over_a a b = 2 * Real.sqrt 5 :=
sorry

end minimum_value_l195_195662


namespace cards_problem_l195_195879

-- Define the conditions and goal
theorem cards_problem 
    (L R : ℕ) 
    (h1 : L + 6 = 3 * (R - 6))
    (h2 : R + 2 = 2 * (L - 2)) : 
    L = 66 := 
by 
  -- proof goes here
  sorry

end cards_problem_l195_195879


namespace real_solutions_of_equation_l195_195998

noncomputable def sqrt : ℝ → ℝ := Real.sqrt

theorem real_solutions_of_equation :
  ∀ x : ℝ, 4 * x^2 - 40 * Real.floor x + 51 = 0 ↔ 
    x = sqrt (29) / 2 ∨ x = sqrt (189) / 2 ∨ x = sqrt (229) / 2 ∨ x = sqrt (269) / 2 :=
by
  sorry

end real_solutions_of_equation_l195_195998


namespace sequence_divisibility_l195_195082

def a : ℕ → ℕ
| 0 := 0 -- base case for indexing convenience
| 1 := 1
| 2 := 2
| (n+2) := a n * (a (n+1) + 1)

theorem sequence_divisibility (n : ℕ) (h : n ≥ 100) : ∃ k : ℕ, a (a n) = k * (a n ^ n) := 
by {
  sorry,
}

end sequence_divisibility_l195_195082


namespace num_ways_to_choose_numbers_l195_195732

noncomputable def count_valid_triplets : ℕ :=
  let choices : Finset (ℕ × ℕ × ℕ) := 
    (Finset.range 15).product (Finset.range 15).product (Finset.range 15)
  let valid_triplets := choices.filter (λ t, 
    let (a1, t2) := t in
    let (a2, a3) := t2 in
    1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 14 ∧ 
    a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3)
  valid_triplets.card

theorem num_ways_to_choose_numbers : count_valid_triplets = 120 :=
by
  sorry

end num_ways_to_choose_numbers_l195_195732


namespace sum_of_divisors_exclusive_l195_195384

-- Define the necessary conditions and state the given problem
variables (p : ℕ) (a : ℕ → ℕ)
hypothesis (hp : Nat.Prime p) (p_odd : p % 2 = 1)
hypothesis (h_a : ∀ k, 1 ≤ k ∧ k ≤ p - 1 → ∃ m, m < k ∧ k < p ∧ a k = (Nat.divisors (k * p + 1)).count (λ d, k < d ∧ d < p))

-- State the equivalent proof problem
theorem sum_of_divisors_exclusive (p_odd : Nat.Odd p) : (Finset.sum (Finset.range (p - 1)) (λ k, a k)) = p - 2 := 
sorry

end sum_of_divisors_exclusive_l195_195384


namespace unique_positive_integer_solution_l195_195890

theorem unique_positive_integer_solution (n p : ℕ) (x y : ℕ) :
  (x + p * y = n ∧ x + y = p^2 ∧ x > 0 ∧ y > 0) ↔ 
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ, n ≠ p^k ∧ ∃! t : ℕ × ℕ, (t.1 + p * t.2 = n ∧ t.1 + t.2 = p^2 ∧ t.1 > 0 ∧ t.2 > 0)) :=
by
  sorry

end unique_positive_integer_solution_l195_195890


namespace least_prime_product_l195_195487
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l195_195487


namespace motorcycle_toll_l195_195658

theorem motorcycle_toll (toll_car : ℝ) (num_car_trips : ℕ) (num_bike_trips : ℕ) (mpg : ℕ) 
  (trip_distance : ℝ) (gas_price : ℝ) (weekly_cost : ℝ) :
  toll_car = 12.50 →
  num_car_trips = 3 →
  num_bike_trips = 2 →
  mpg = 35 →
  trip_distance = 14 →
  gas_price = 3.75 →
  weekly_cost = 118 →
  (let total_toll_cost_car := num_car_trips * toll_car,
       total_gas_cost := ((num_car_trips + num_bike_trips) * (2 * trip_distance) / mpg) * gas_price,
       total_toll_cost_bike := num_bike_trips * M in
   weekly_cost = total_toll_cost_car + total_toll_cost_bike + total_gas_cost) →
  (M = 32.75) :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  let total_toll_cost_car := num_car_trips * toll_car,
  let total_gas_cost := ((num_car_trips + num_bike_trips) * (2 * trip_distance) / mpg) * gas_price,
  let total_toll_cost_bike := num_bike_trips * M,
  have hweekly_cost : weekly_cost = total_toll_cost_car + total_toll_cost_bike + total_gas_cost := sorry,
  have htotal_toll_cost_car := h2 ▸ h1 ▸ (3 * 12.5 : ℝ),
  have htotal_gas_cost := (3 + 2) * (2 * trip_distance) / mpg * gas_price,
  calc
    total_toll_cost_car + total_toll_cost_bike + total_gas_cost = 118 : by assumption
    ... = 52.50 + total_toll_cost_bike + 15 : by linarith 
    ... = 67.50 + total_toll_cost_bike : by linarith
    ... = 67.50 + 2 * M : by simp
  exact sorry -- actual proof step where you show M = 32.75
end

end motorcycle_toll_l195_195658


namespace find_angle_A_find_perimeter_l195_195366

variable (a b c A B C : ℝ)

def vector_m : ℝ × ℝ := (a, b)
def vector_n : ℝ × ℝ := (Real.sin B, Real.sqrt 3 * Real.cos A)
def triangle_ABC := a = Real.sqrt 7 ∧ (1 / 2) * (b * c * Real.sin A) = (Real.sqrt 3) / 2

theorem find_angle_A (h1 : a * Real.sin B + (Real.sqrt 3) * b * Real.cos A = 0) :
  A = 2 * Real.pi / 3 := sorry

theorem find_perimeter (h1 : a = Real.sqrt 7) (h2 : (1 / 2) * b * c * Real.sin A = (Real.sqrt 3) / 2)
  (h3 : A = 2 * Real.pi / 3) :
  a + b + c = Real.sqrt 7 + 3 := sorry

end find_angle_A_find_perimeter_l195_195366


namespace max_ray_obtuse_angle_l195_195667

theorem max_ray_obtuse_angle (n : ℕ) (rays : Fin n → ℝ × ℝ × ℝ) 
  (h : ∀ i j : Fin n, i ≠ j → (rays i).dot (rays j) < 0) : n ≤ 4 :=
sorry

end max_ray_obtuse_angle_l195_195667


namespace tan_double_angle_l195_195453

theorem tan_double_angle 
  (x : ℝ) 
  (y : ℝ) 
  (tan_alpha : ℝ) 
  (h₁ : x ≠ 0) 
  (h₂ : x = -2) 
  (h₃ : y = 1) 
  (h₄ : tan_alpha = y / x) : 
  (2 * tan_alpha) / (1 - tan_alpha^2) = -4 / 3 :=
begin
  sorry
end

end tan_double_angle_l195_195453


namespace find_k_satisfying_norm_l195_195642

open real

theorem find_k_satisfying_norm (k : ℝ) :
  (∃ k : ℝ, 
    ∥k • (3, 4) + (-2, 5)∥ = 5 * sqrt 2 ∧ 
    (k = 13 / 25 ∨ k = -41 / 25)) :=
begin
  use [13 / 25, -41 / 25],
  split,
  { have h1 : ∃ k : ℝ, ∥k • (3, 4) + (-2, 5)∥ = 5 * sqrt 2, sorry,
    exact h1
  },
  { left,
    norm_num },
  { right,
    norm_num }
end

end find_k_satisfying_norm_l195_195642


namespace class_sizes_l195_195614

theorem class_sizes (size_class_b : ℕ) (h1 : 20 = size_class_b) (h2 : ∀ size_class_a, size_class_a = 2 * size_class_b → ∀ size_class_c, size_class_a = size_class_c / 3 → size_class_c = 120) :
  ∃ size_class_c : ℕ, size_class_c = 120 :=
by
  use 120
  sorry

end class_sizes_l195_195614


namespace limit_seq_a_l195_195836

noncomputable def seq_a : ℕ → ℝ
| 0       := 1
| (n + 1) := seq_a n + Real.exp (-seq_a n)

def limit_difference (a : ℕ → ℝ) : Prop :=
  Tendsto (fun n => a n - Real.log (n + 1)) atTop (𝓝 0)

theorem limit_seq_a : limit_difference seq_a :=
sorry

end limit_seq_a_l195_195836


namespace integral_partial_fraction_decomposition_l195_195546

theorem integral_partial_fraction_decomposition :
  ∃ C, ∫ (f := (λ x : ℝ, (2 * x^3 + 4 * x^2 + 2 * x - 1) / ((x + 1)^2 * (x^2 + 2 * x + 2)))) dx
      = λ x : ℝ, (1 / (x + 1) + log (abs (x^2 + 2 * x + 2)) - arctan (x + 1) + C) := 
sorry

end integral_partial_fraction_decomposition_l195_195546


namespace y_intercept_of_line_b_l195_195015

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l195_195015


namespace probability_limit_of_hypergeometric_distribution_l195_195936

theorem probability_limit_of_hypergeometric_distribution
  (M M1 M2 n n1 n2 : ℕ)
  (p : ℝ)
  (hM : M1 + M2 = M)
  (hn : n1 + n2 = n)
  (hM1 : ∃ M → ∞, M1 / M → p)
  (hM_inf : M → ∞) :
  (∃ M1 → ∞, (M1 / M → p) →
  (∃ M2 → ∞, (M2 / M → 1 - p) →
  (lim (λ M, P(B_{n1, n2})) = binom n n1 * p ^ n1 * (1 - p) ^ n2))) :=
sorry

end probability_limit_of_hypergeometric_distribution_l195_195936


namespace range_of_f_l195_195080

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * (Real.cos x) ^ 2

theorem range_of_f : set.range f = set.Icc (-Real.sqrt 3 / 8 * 3) (Real.sqrt 3 / 8 * 3) := sorry

end range_of_f_l195_195080


namespace polynomial_integer_values_l195_195050

noncomputable def p (x : ℤ) : ℚ :=
  (1 / 630) * x^9 - (1 / 21) * x^7 + (13 / 20) * x^5 - (82 / 63) * x^3 + (32 / 35) * x

theorem polynomial_integer_values (x : ℤ) : p x ∈ ℤ := 
  sorry

end polynomial_integer_values_l195_195050


namespace count_valid_integers_l195_195334

-- Definition for a positive integer whose digits do not include 0 and the sum of digits is 6
def validInteger (n : ℕ) : Prop :=
  (∀ d : ℕ, (d ∈ digitList n) → (1 ≤ d ∧ d ≤ 9)) ∧
  (digitList n).sum = 6

-- Statement of the math proof problem
theorem count_valid_integers : 
  (Finset.filter validInteger (Finset.range 1000000)).card = 32 :=
by
  sorry

end count_valid_integers_l195_195334


namespace solve_quadratic_eq_l195_195830

theorem solve_quadratic_eq (x : ℂ) : 
  x^2 + 6 * x + 8 = -(x + 2) * (x + 6) ↔ (x = -3 + complex.I ∨ x = -3 - complex.I) := 
by
  sorry 

end solve_quadratic_eq_l195_195830


namespace isosceles_triangle_base_angles_equal_l195_195531

theorem isosceles_triangle_base_angles_equal (ABC : Type) [metric_space ABC] 
  (A B C : ABC) : let is_isosceles (A B C : ABC) := dist A B = dist A C in
    is_isosceles A B C → angle B A C = angle C A B :=
by
  sorry

end isosceles_triangle_base_angles_equal_l195_195531


namespace constant_term_in_binomial_expansion_l195_195731

theorem constant_term_in_binomial_expansion (n : ℕ) (h_sum : (∑ k in finset.range (n+1), nat.choose n k) = 64) :
  let x := (1 : ℝ) in
  let expansion_term (r : ℕ) := (nat.choose n r) * x^(n - 2 * r) in
  ∑ r in finset.range (n+1), if n = 2 * r then expansion_term r else 0 = 20 :=
by
  sorry

end constant_term_in_binomial_expansion_l195_195731


namespace balloon_permutations_l195_195226

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l195_195226


namespace find_z_l195_195718

theorem find_z (z : ℂ) (hz : conj z * (1 + complex.I) = 1 - complex.I) : 
  z = complex.I :=
by 
  sorry

end find_z_l195_195718


namespace least_prime_product_l195_195486
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l195_195486


namespace areas_equal_l195_195166

noncomputable def area (P Q R S : Point) := 1 / 2 * abs (P.x * (Q.y - S.y) + Q.x * (S.y - P.y) + S.x * (P.y - Q.y))

variables 
  (A B C D O M N K : Point)
  (circle O : Circle O)
  (in_circle_ABCD : (circle O : Set Point) = Set.univ)
  (intersect_diagonals : ∃ M, is_intersection (Diagonal A C) (Diagonal B D))
  (intersect_circum_ABM_AD_N : ∃ N, Intersection (Circumcircle A B M) (Line A D) = {N})
  (intersect_circum_ABM_BC_K : ∃ K, Intersection (Circumcircle A B M) (Line B C) = {K})

open_locale real_InnerProductSpace

theorem areas_equal : area N O M D = area K O M C :=
sorry

end areas_equal_l195_195166


namespace quadratic_always_two_real_roots_quadratic_root_less_than_one_l195_195268

theorem quadratic_always_two_real_roots (k : ℝ) : 
  let a := 1
  let b := -(k + 3)
  let c := 2 * k + 2
  let Δ := b * b - 4 * a * c
  in Δ ≥ 0 :=
by {
  let a := 1,
  let b := -(k + 3),
  let c := 2 * k + 2,
  let Δ := b * b - 4 * a * c,
  exact (Δ ≥ 0)
  sorry
}

theorem quadratic_root_less_than_one (k : ℝ) (h : k + 1 < 1) :
  k < 0 :=
by {
  exact h.trans (by norm_num),
  sorry
}

end quadratic_always_two_real_roots_quadratic_root_less_than_one_l195_195268


namespace find_j_value_l195_195843

variable {R : Type*} [LinearOrderedField R]

-- Definitions based on conditions
def polynomial_has_four_distinct_real_roots_in_arithmetic_progression
(p : Polynomial R) : Prop :=
∃ a d : R, p.roots.toFinset = {a, a + d, a + 2*d, a + 3*d} ∧
a ≠ a + d ∧ a ≠ a + 2*d ∧ a ≠ a + 3*d ∧ a + d ≠ a + 2*d ∧
a + d ≠ a + 3*d ∧ a + 2*d ≠ a + 3*d

-- The main theorem statement
theorem find_j_value (k : R) 
  (h : polynomial_has_four_distinct_real_roots_in_arithmetic_progression 
  (Polynomial.X^4 + Polynomial.C j * Polynomial.X^2 + Polynomial.C k * Polynomial.X + Polynomial.C 900)) :
  j = -900 :=
sorry

end find_j_value_l195_195843


namespace arcsin_sqrt2_over_2_l195_195616

noncomputable def sin_π_over_4 : ℝ := Real.sin (Real.pi / 4)

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
by 
  have h : sin_π_over_4 = Real.sqrt 2 / 2,
  { sorry },
  rw [←Real.arcsin_sin (Real.pi / 4)],
  exact h

end arcsin_sqrt2_over_2_l195_195616


namespace find_radius_of_cone_l195_195472

variables (R : ℝ) (r : ℝ := real.sqrt 24)

theorem find_radius_of_cone (h : R = r * (7 + 4 * real.sqrt 3 + 2 * real.sqrt 6)) :
  R = 7 + 4 * real.sqrt 3 + 2 * real.sqrt 6 :=
by
  sorry

end find_radius_of_cone_l195_195472


namespace average_wage_correct_l195_195559

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_amount_paid_per_day : ℕ := 
  (male_workers * male_wage) + (female_workers * female_wage) + (child_workers * child_wage)

def total_number_of_workers : ℕ := 
  male_workers + female_workers + child_workers

def average_wage_per_day : ℕ := 
  total_amount_paid_per_day / total_number_of_workers

theorem average_wage_correct : 
  average_wage_per_day = 21 := by 
  sorry

end average_wage_correct_l195_195559


namespace sum_of_c_for_g_eq_c_has_4_solutions_l195_195451

noncomputable def g : ℝ → ℝ :=
  λ x, ((x - 4) * (x - 2) * (x + 2) * (x + 4)) / 48 - 2

theorem sum_of_c_for_g_eq_c_has_4_solutions :
  (∑ c in {c : ℤ | ∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = c ∧ g x₂ = c ∧ g x₃ = c ∧ g x₄ = c ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄}, c) = -3 :=
sorry

end sum_of_c_for_g_eq_c_has_4_solutions_l195_195451


namespace sum_product_eq_six_l195_195835

theorem sum_product_eq_six :
  (∑ (a b : ℕ), if a^2 + 2 = b! then a * b else 0) = 6 :=
sorry

end sum_product_eq_six_l195_195835


namespace max_value_of_e_n_l195_195787

def b (n : ℕ) : ℕ := (8^n - 1) / 7
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_of_e_n : ∀ n : ℕ, e n = 1 := 
by
  sorry

end max_value_of_e_n_l195_195787


namespace sum_sequence_l195_195277

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 0     := 1
| (n+1) := @klass Boekest Hutches offendent someg pingreciety bagg isley druric Ferni∃

theorem sum_sequence (S_2023 : ℕ) : 
  (sequence 0) = 1 →
  (∀ (n : ℕ), (sequence n) * (sequence (n+1)) = 2 n) →
  S_2023 = ∑ (k : ℕ) in range (2023), (sequence k) :=
  S_2023 = 2 ^ 1013 - 3 :=
begin
  intros,
  sorry
end

end sum_sequence_l195_195277


namespace probability_of_2_pow_x_lt_2_is_1_over_4_l195_195674

noncomputable def probability_of_2_pow_x_lt_2 (x : ℝ) : ℝ :=
if x ∈ set.Ioo 0 4 then (if x ∈ set.Ioo 0 1 then (1/4) else 0) else 0

theorem probability_of_2_pow_x_lt_2_is_1_over_4 :
  ∀ x, x ∈ set.Ioo 0 4 → probability_of_2_pow_x_lt_2 x = 1/4 :=
begin
  intros,
  simp [probability_of_2_pow_x_lt_2],
  split_ifs,
  { sorry }, -- Here, the actual detailed proof would be written
  { sorry }  -- where we argue that if x ∈ (0, 1) then indeed the probability is 1/4.
end

end probability_of_2_pow_x_lt_2_is_1_over_4_l195_195674


namespace number_of_zeros_g_l195_195299

variable (f : ℝ → ℝ)
variable (hf_cont : continuous f)
variable (hf_diff : differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, x * (deriv f x) + f x > 0)

theorem number_of_zeros_g (hg : ∀ x : ℝ, x > 0 → x * f x + 1 = 0 → false) : 
    ∀ x : ℝ , x > 0 → ¬ (x * f x + 1 = 0) :=
by
  sorry

end number_of_zeros_g_l195_195299


namespace sum_of_magnitudes_l195_195780

-- Given conditions and definitions
def parabola (x y : ℝ) := y^2 = 4 * x
def is_focus_of_parabola (F : ℝ × ℝ) := F = (1, 0)
def lies_on_parabola (P : ℝ × ℝ) := parabola P.1 P.2
def FA (F A : ℝ × ℝ) := (A.1 - F.1, A.2 - F.2)
def magnitudes_sum_zero (F A B C : ℝ × ℝ) := 
  (FA F A).1 + (FA F A).2 + (FA F B).1 + (FA F B).2 + (FA F C).1 + (FA F C).2 = 0

-- Proof we're looking to state
theorem sum_of_magnitudes (A B C F : ℝ × ℝ) 
  (hA : lies_on_parabola A) (hB : lies_on_parabola B) (hC : lies_on_parabola C) 
  (hF : is_focus_of_parabola F) (hSum : magnitudes_sum_zero F A B C) :
  ( |  FA F A| + | FA F B | + |FA F C | ) = 6 := sorry

end sum_of_magnitudes_l195_195780


namespace sequence_is_arithmetic_find_T_n_l195_195782

-- Given sequence conditions
def a (n : ℕ) : ℕ := sorry
def S (n : ℕ) : ℕ := n * a n - 2 * n * (n - 1)

-- Problem statement part (I): Prove sequence is arithmetic
theorem sequence_is_arithmetic (a b : ℕ) : 
  ∀ n, a 1 = 1 ∧ S n = n * a n - 2 * n * (n - 1) → a (n + 1) - a n = 4 :=
sorry

-- Problem statement part (II): Find sum T_n
def T (n : ℕ) : ℕ := 
  ∑ i in range n, 1 / (a i * a (i + 1))

theorem find_T_n (n : ℕ) :
  T n = 1 / 4 - 1 / (16 * n + 4) :=
sorry

end sequence_is_arithmetic_find_T_n_l195_195782


namespace find_m_value_l195_195855

def polynomial (m : ℤ) : ℤ[X] := X^3 - X^2 - (m^2 + m) * X + 2 * m^2 + 4 * m + 2

theorem find_m_value (m : ℤ) (hf : polynomial m = X^3 - X^2 - (m^2 + m) * X + 2 * m^2 + 4 * m + 2)
  (h1: polynomial m.eval 4 = 0) (h2: ∀ z ∈ polynomial m.roots, z ∈ ℤ) :
  m = 5 :=
sorry

end find_m_value_l195_195855


namespace bacteria_exceeds_200_in_four_days_l195_195741

theorem bacteria_exceeds_200_in_four_days (n : ℕ) : 
  (5 * 3^n > 200) → (n ≥ 4) :=
begin
  sorry
end

end bacteria_exceeds_200_in_four_days_l195_195741


namespace correct_differentiation_operations_l195_195528

def f_A (x : ℝ) : ℝ := sin (2 * x - 1)
def f_B (x : ℝ) : ℝ := exp (-0.05 * x + 1)
def f_C (x : ℝ) : ℝ := x / exp(x)
def f_D (x : ℝ) : ℝ := x * log x

lemma correct_A : deriv f_A = λ x, 2 * cos (2 * x - 1) := sorry
lemma incorrect_B : deriv f_B ≠ λ x, exp (-0.05 * x + 1) := sorry
lemma incorrect_C : deriv f_C ≠ λ x, (1 + x) / exp(x) := sorry
lemma correct_D : deriv f_D = λ x, log x + 1 := sorry

theorem correct_differentiation_operations : 
  (deriv f_A = λ x, 2 * cos (2 * x - 1)) ∧ 
  (deriv f_D = λ x, log x + 1) := 
  by 
  exact ⟨correct_A, correct_D⟩

end correct_differentiation_operations_l195_195528


namespace part_a_part_b_l195_195402

variables (a b : Real) (sqrt5 : Real)
axiom sqrt5_property : sqrt5^2 = 5

theorem part_a (h : a^2 - 5 * b^2 = 1) : (a - b * sqrt5)^2 - 5 * (b * sqrt5)^2 = 1 :=
by {
  rw [sq_sub, ←mul_assoc b b sqrt5, sqrt5_property],
  rw [mul_comm, ←add_assoc, ←pow_two, ←neg_mul_eq_mul_neg, ←neg_mul_eq_neg_mul],
  assumption,
}

theorem part_b (h : a^2 - 5 * b^2 = 1) : 1 / (a + b * sqrt5) = a - b * sqrt5 :=
by {
  have h_neg : (a - b * sqrt5) * (a + b * sqrt5) = 1,
  { rw [mul_comm, mul_add, mul_sub_left_distrib, mul_sub_right_distrib, ←neg_mul_eq_mul_neg],
    rw [sqrt5_property, mul_comm, ←neg_mul_eq_neg_mul_symm (b * b), ←neg_mul_eq_neg_mul_symm (5 * b)],
    rw [←mul_assoc, mul_comm, ←mul_assoc, ←pow_two, h, sub_add_cancel],
  },
  exact mul_inv_cancel h_neg,
}

#check part_a
#check part_b

end part_a_part_b_l195_195402


namespace total_marbles_l195_195102

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l195_195102


namespace rain_probability_l195_195266

theorem rain_probability (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.3)
  (h_saturday : p_saturday = 0.6)
  (h_sunday : p_sunday = 0.4)
  (independent : true) : 
  1 - ((1 - p_friday) * (1 - p_saturday) * (1 - p_sunday)) = 0.832 :=
by
  rw [h_friday, h_saturday, h_sunday]
  norm_num [independent]
  sorry

end rain_probability_l195_195266


namespace least_product_of_primes_gt_30_l195_195497

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l195_195497


namespace composition_equals_everywhere_l195_195143

variable {n : Type} [Real n]

variable (f g : n → n)

def f_poly : Polynomial n → n := λ f, f.coeffs.map toReal

theorem composition_equals_everywhere
  (hg : ∀ x : n, f(g(x)) = g(f(x)))
  (hnr : ∀ x : n, f(x) ≠ g(x)) :
  ∀ x : n, f(f(x)) ≠ g(g(x)) := 
sorry

end composition_equals_everywhere_l195_195143


namespace total_number_of_socks_l195_195370

def number_of_pairs (pairs : ℕ) := 2 * pairs

theorem total_number_of_socks :
  let red_pairs := 20 in
  let black_pairs := red_pairs / 2 in
  let white_pairs := 2 * (red_pairs + black_pairs) in
  let total_pairs := red_pairs + black_pairs + white_pairs in
  let green_pairs := Int.floor (Real.sqrt ↑total_pairs) in
  number_of_pairs red_pairs +
  number_of_pairs black_pairs +
  number_of_pairs white_pairs +
  number_of_pairs green_pairs = 198 := by
{
  -- Proof steps would go here
  sorry
}

end total_number_of_socks_l195_195370


namespace minimum_phi_tranlation_even_function_l195_195474

theorem minimum_phi_tranlation_even_function :
  ∀ (φ : ℝ), φ > 0 → 
  (∀ x : ℝ, 2 * sin (2 * (x + φ) + π / 3) = 2 * sin (-(2 * (x + φ) + π / 3))) → 
  φ = π / 12 :=
by
  assume φ hφ heven
  -- proof to be completed
  sorry

end minimum_phi_tranlation_even_function_l195_195474


namespace line_b_y_intercept_l195_195011

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l195_195011


namespace area_of_room_l195_195770

def length : ℝ := 12
def width : ℝ := 8

theorem area_of_room : length * width = 96 :=
by sorry

end area_of_room_l195_195770


namespace heartbeats_during_race_l195_195183

theorem heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : 
  heart_rate = 180 → pace = 3 → distance = 50 → (pace * distance * heart_rate = 27000) :=
by
  intros h_hr h_p h_d
  rw [h_hr, h_p, h_d]
  exact eq.refl 27000

end heartbeats_during_race_l195_195183


namespace desired_profit_is_eight_percent_l195_195171

def cp : ℝ := 100
def mp : ℝ := cp * 1.3
def discount_percentage : ℝ := 0.1692307692307692
def sp : ℝ := mp * (1 - discount_percentage)

def desired_profit_percentage : ℝ := ((sp - cp) / cp) * 100

theorem desired_profit_is_eight_percent : desired_profit_percentage = 8 := by
  sorry

end desired_profit_is_eight_percent_l195_195171


namespace flowers_brought_at_dawn_l195_195567

theorem flowers_brought_at_dawn (F : ℕ) 
  (h1 : (3 / 5) * F = 180)
  (h2 :  (2 / 5) * F + (F - (3 / 5) * F) = 180) : 
  F = 300 := 
by
  sorry

end flowers_brought_at_dawn_l195_195567


namespace inclination_angle_m_l195_195729

noncomputable def line1 : ℝ × ℝ × ℝ := (1, -√3, 1)
noncomputable def line2 : ℝ × ℝ × ℝ := (1, -√3, 3)

noncomputable def is_perpendicular (A B C D : ℝ) : Prop :=
  A * B + C * D = 0

theorem inclination_angle_m :
  let m := 120 in
  ∀ A B C D : ℝ,
  (line1 = (A, B, C)) ∧ (line2 = (A, B, D)) ∧ (m = 120) → 
  is_perpendicular B A (-√1) (√3) :=
by
  sorry

end inclination_angle_m_l195_195729


namespace value_of_a_b_c_l195_195125

theorem value_of_a_b_c 
    (a b c : Int)
    (h1 : ∀ x : Int, x^2 + 10*x + 21 = (x + a) * (x + b))
    (h2 : ∀ x : Int, x^2 + 3*x - 88 = (x + b) * (x - c))
    :
    a + b + c = 18 := 
sorry

end value_of_a_b_c_l195_195125


namespace find_sum_of_A_and_B_l195_195916

theorem find_sum_of_A_and_B :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ B = A - 2 ∧ A = 5 + 3 ∧ A + B = 14 :=
by
  sorry

end find_sum_of_A_and_B_l195_195916


namespace sum_of_coeffs_l195_195390

noncomputable def k : ℝ :=
  ∫ x in (0:ℝ)..π, (sin x - cos x)

def polynomial_expansion (k : ℝ) : Fin 9 → ℝ :=
  fun i => (∑ a, ((1 - k*x)^8).coeffs)

theorem sum_of_coeffs : (polynomial_expansion k) 1 + (polynomial_expansion k) 2 + (polynomial_expansion k) 3 +
  (polynomial_expansion k) 4 + (polynomial_expansion k) 5 + (polynomial_expansion k) 6 +
  (polynomial_expansion k) 7 + (polynomial_expansion k) 8 = 0 :=
sorry

end sum_of_coeffs_l195_195390


namespace smallest_divisible_by_3_and_4_is_12_l195_195601

theorem smallest_divisible_by_3_and_4_is_12 
  (n : ℕ) 
  (h1 : ∃ k1 : ℕ, n = 3 * k1) 
  (h2 : ∃ k2 : ℕ, n = 4 * k2) 
  : n ≥ 12 := sorry

end smallest_divisible_by_3_and_4_is_12_l195_195601


namespace inradius_circumradius_of_right_triangle_l195_195274

theorem inradius_circumradius_of_right_triangle (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) 
(h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) :
  let A := 0.5 * a * b,
  s := (a + b + c) / 2,
  r := A / s,
  R := c / 2 in
  r = 3 ∧ R = 7.5 := by
  sorry

end inradius_circumradius_of_right_triangle_l195_195274


namespace factor_polynomial_l195_195793

theorem factor_polynomial (n : ℕ) (hn : 2 ≤ n) 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℤ, n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧ 
  a = (-(2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n)))) ^ (2 * n / (2 * n - 1)) ∧ 
  b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n))) ^ (2 / (2 * n - 1)) := sorry

end factor_polynomial_l195_195793


namespace total_amount_shared_l195_195178

theorem total_amount_shared
  (A B C : ℕ)
  (h_ratio : A / 2 = B / 3 ∧ B / 3 = C / 8)
  (h_Ben_share : B = 30) : A + B + C = 130 :=
by
  -- Add placeholder for the proof.
  sorry

end total_amount_shared_l195_195178


namespace cameron_total_questions_l195_195964

def usual_questions : Nat := 2

def group_a_questions : Nat := 
  let q1 := 2 * 1 -- 2 people who asked a single question each
  let q2 := 3 * usual_questions -- 3 people who asked two questions as usual
  let q3 := 1 * 5 -- 1 person who asked 5 questions
  q1 + q2 + q3

def group_b_questions : Nat :=
  let q1 := 1 * 0 -- 1 person asked no questions
  let q2 := 6 * 3 -- 6 people asked 3 questions each
  let q3 := 4 * usual_questions -- 4 people asked the usual number of questions
  q1 + q2 + q3

def group_c_questions : Nat :=
  let q1 := 1 * (usual_questions * 3) -- 1 person asked three times as many questions as usual
  let q2 := 1 * 1 -- 1 person asked only one question
  let q3 := 2 * 0 -- 2 members asked no questions
  let q4 := 4 * usual_questions -- The remaining tourists asked the usual 2 questions each
  q1 + q2 + q3 + q4

def group_d_questions : Nat :=
  let q1 := 1 * (usual_questions * 4) -- 1 individual asked four times as many questions as normal
  let q2 := 1 * 0 -- 1 person asked no questions at all
  let q3 := 3 * usual_questions -- The remaining tourists asked the usual number of questions
  q1 + q2 + q3

def group_e_questions : Nat :=
  let q1 := 3 * (usual_questions * 2) -- 3 people asked double the average number of questions
  let q2 := 2 * 0 -- 2 people asked none
  let q3 := 1 * 5 -- 1 tourist asked 5 questions
  let q4 := 3 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3 + q4

def group_f_questions : Nat :=
  let q1 := 2 * 3 -- 2 individuals asked three questions each
  let q2 := 1 * 0 -- 1 person asked no questions
  let q3 := 4 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3

def total_questions : Nat :=
  group_a_questions + group_b_questions + group_c_questions + group_d_questions + group_e_questions + group_f_questions

theorem cameron_total_questions : total_questions = 105 := by
  sorry

end cameron_total_questions_l195_195964


namespace sector_max_angle_l195_195677

variables (r l : ℝ)

theorem sector_max_angle (h : 2 * r + l = 40) : (l / r) = 2 :=
sorry

end sector_max_angle_l195_195677


namespace maximize_x6_y3_l195_195391

theorem maximize_x6_y3 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 40) :
  (x, y) = (24, 16) → (∀ (a b : ℝ), (0 < a ∧ 0 < b ∧ a + b = 40) → a^6 * b^3 ≤ 24^6 * 16^3) :=
-- here we define the conditions in the goal state
-- then we use condition result to perform the actual proof that there is a maximum value
begin
  sorry
end

end maximize_x6_y3_l195_195391


namespace constant_sum_of_distances_l195_195755

open Real

theorem constant_sum_of_distances (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
    (ellipse_condition : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∀ A B : ℝ × ℝ, A.2 > 0 ∧ B.2 > 0)
    (foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0)))
    (points_AB : ∃ (A B : ℝ × ℝ), A.2 > 0 ∧ B.2 > 0 ∧ (A.1 - c)^2 / a^2 + A.2^2 / b^2 = 1 ∧ (B.1 - -c)^2 / a^2 + B.2^2 / b^2 = 1)
    (AF1_parallel_BF2 : ∀ (A B : ℝ × ℝ), (A.1 - -c) * (B.2 - 0) - (A.2 - 0) * (B.1 - c) = 0)
    (intersection_P: ∀ (A B : ℝ × ℝ), ∃ P : ℝ × ℝ, ((A.1 - c) * (B.2 - 0) = (A.2 - 0) * (P.1 - c)) ∧ ((B.1 - -c) * (A.2 - 0) = (B.2 - 0) * (P.1 - -c))) :
    ∃ k : ℝ, ∀ (P : ℝ × ℝ), dist P (foci.fst) + dist P (foci.snd) = k := 
sorry

end constant_sum_of_distances_l195_195755


namespace correct_options_l195_195283

variable (x : Fin 6 → ℝ)

def median_of_4 (a b c d : ℝ) : ℝ := (b + c) / 2

def median_of_6 (a b c d e f : ℝ) : ℝ := (c + d) / 2

theorem correct_options (x : Fin 6 → ℝ)
  (h1 : x 0 = min (min (x 0) (x 1)) (min (min (x 2) (x 3)) (min (x 4) (x 5)))) 
  (h6 : x 5 = max (max (x 0) (x 1)) (max (max (x 2) (x 3)) (max (x 4) (x 5)))) :
  (median_of_4 (x 1) (x 2) (x 3) (x 4) = median_of_6 (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) ∧
  (range (x 1) (x 2) (x 3) (x 4) ≤ range (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) :=
sorry

end correct_options_l195_195283


namespace no_valid_five_digit_number_l195_195727

theorem no_valid_five_digit_number (n : ℕ) (digits : fin 5 → ℕ) :
  (∀ i, digits i < 10) ∧ (digits 0 > 0) ∧ (∑ i, digits i = 20) →
  ¬(n % 9 = 0 ∧ (∑ i, digits i) % 4 = 0) :=
by
  sorry

end no_valid_five_digit_number_l195_195727


namespace gcd_54_180_l195_195521

theorem gcd_54_180 : Nat.gcd 54 180 = 18 := by
  sorry

end gcd_54_180_l195_195521


namespace intersection_points_l195_195884

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 12 * x - 5
def parabola2 (x : ℝ) : ℝ := x ^ 2 - 2 * x + 3

theorem intersection_points :
  { p : ℝ × ℝ | p.snd = parabola1 p.fst ∧ p.snd = parabola2 p.fst } =
  { (1, -14), (4, -5) } :=
by
  sorry

end intersection_points_l195_195884


namespace particle_speed_at_time_t_l195_195162

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + t + 1, 6 * t + 2)

theorem particle_speed_at_time_t (t : ℝ) :
  let dx := (position t).1
  let dy := (position t).2
  let vx := 6 * t + 1
  let vy := 6
  let speed := Real.sqrt (vx^2 + vy^2)
  speed = Real.sqrt (36 * t^2 + 12 * t + 37) :=
by
  sorry

end particle_speed_at_time_t_l195_195162


namespace least_product_of_primes_over_30_l195_195480

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l195_195480


namespace miniature_tank_height_l195_195638

-- Given conditions
def actual_tank_height : ℝ := 50
def actual_tank_volume : ℝ := 200000
def model_tank_volume : ℝ := 0.2

-- Theorem: Calculate the height of the miniature water tank
theorem miniature_tank_height :
  (model_tank_volume / actual_tank_volume) ^ (1/3 : ℝ) * actual_tank_height = 0.5 :=
by
  sorry

end miniature_tank_height_l195_195638


namespace limit_of_sequence_l195_195192

open_locale classical

noncomputable def geom_series_sum (n : ℕ) : ℝ :=
  (1 - 3^(n + 1)) / (1 - 3)

theorem limit_of_sequence :
  tendsto (λ n : ℕ, (geom_series_sum n) / (3^n + 2^n)) at_top (𝓝 (3 / 2)) :=
sorry

end limit_of_sequence_l195_195192


namespace imaginary_part_z_l195_195308

theorem imaginary_part_z : 
  ∀ (z : ℂ), z = (5 - I) / (1 - I) → z.im = 2 := 
by
  sorry

end imaginary_part_z_l195_195308


namespace least_product_of_distinct_primes_over_30_l195_195503

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l195_195503


namespace correct_options_l195_195282

variable (x : Fin 6 → ℝ)

def median_of_4 (a b c d : ℝ) : ℝ := (b + c) / 2

def median_of_6 (a b c d e f : ℝ) : ℝ := (c + d) / 2

theorem correct_options (x : Fin 6 → ℝ)
  (h1 : x 0 = min (min (x 0) (x 1)) (min (min (x 2) (x 3)) (min (x 4) (x 5)))) 
  (h6 : x 5 = max (max (x 0) (x 1)) (max (max (x 2) (x 3)) (max (x 4) (x 5)))) :
  (median_of_4 (x 1) (x 2) (x 3) (x 4) = median_of_6 (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) ∧
  (range (x 1) (x 2) (x 3) (x 4) ≤ range (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) :=
sorry

end correct_options_l195_195282


namespace line_b_y_intercept_l195_195012

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l195_195012


namespace paco_cookie_problem_l195_195043

theorem paco_cookie_problem (x : ℕ) (hx : x + 9 = 18) : x = 9 :=
by sorry

end paco_cookie_problem_l195_195043


namespace max_downward_closed_sum_l195_195670

def downward_closed_family (n : ℕ) (D : finset (finset (fin n))) : Prop :=
∀ {A B : finset (fin n)}, A ∈ D → B ⊆ A → B ∈ D

def sum_expression (n : ℕ) (D : finset (finset (fin n))) : ℤ :=
∑ A in D, (-1) ^ A.card

theorem max_downward_closed_sum (n : ℕ) (D : finset (finset (fin n))) (h : downward_closed_family n D) :
  sum_expression n D ≤ ∑ i in finset.range (2 * (n / 4)), (-1 : ℤ) ^ i * nat.choose n i :=
sorry

end max_downward_closed_sum_l195_195670


namespace cos2a_over_sina_plus_pi_4_l195_195300

theorem cos2a_over_sina_plus_pi_4 (α : ℝ) (h0 : α ∈ Ioo 0 (π / 2)) (h1 : sin α - cos α = 1 / 2) : (cos (2 * α) / sin (α + π / 4)) = - (sqrt 2 / 2) :=
by
  sorry

end cos2a_over_sina_plus_pi_4_l195_195300


namespace items_per_charge_is_five_l195_195191

-- Define the number of dog treats, chew toys, rawhide bones, and credit cards as constants.
def num_dog_treats := 8
def num_chew_toys := 2
def num_rawhide_bones := 10
def num_credit_cards := 4

-- Define the total number of items.
def total_items := num_dog_treats + num_chew_toys + num_rawhide_bones

-- Prove that the number of items per credit card charge is 5.
theorem items_per_charge_is_five :
  (total_items / num_credit_cards) = 5 :=
by
  -- Proof goes here (we use sorry to skip the actual proof)
  sorry

end items_per_charge_is_five_l195_195191


namespace original_number_l195_195117

theorem original_number (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 :=
sorry

end original_number_l195_195117


namespace tan_sum_min_value_l195_195745

variable (A B C a b c : ℝ)

theorem tan_sum_min_value
  (h1 : 0 < A) (h2 : A < π / 2)
  (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 0 < C) (h6 : C < π / 2)
  (h7 : a = b^2 + c^2 - 2 * b * c * Math.cos A)
  (h8 : b^2 + c^2 = 4 * b * c * Math.sin (A + π / 6))
  : tan A + tan B + tan C = 48 := by
  sorry

end tan_sum_min_value_l195_195745


namespace hit_target_at_least_once_l195_195514

variables (P : Set Bool → ℝ)
variables (A B : Set Bool)

def prob_A := P A = 0.6
def prob_B := P B = 0.5

theorem hit_target_at_least_once : P (A ∪ B) = 0.8 :=
by
  have complement_A : P (Set.univ \ A) = 0.4 := sorry
  have complement_B : P (Set.univ \ B) = 0.5 := sorry
  have neither_hit : P ((Set.univ \ A) ∩ (Set.univ \ B)) = 0.2 := sorry
  have target_at_least_once : P (A ∪ B) = 1 - 0.2 := sorry
  exact target_at_least_once

end hit_target_at_least_once_l195_195514


namespace smallest_mn_sum_l195_195077

theorem smallest_mn_sum {n m : ℕ} (h1 : n > m) (h2 : 1978 ^ n % 1000 = 1978 ^ m % 1000) (h3 : m ≥ 1) : m + n = 106 := 
sorry

end smallest_mn_sum_l195_195077


namespace identity_proof_l195_195518

def KnightOrLiar := Prop

variable (A B C : KnightOrLiar)
variable (S_A S_B : KnightOrLiar)

-- A says: "We are all liars."
axiom A_statement : S_A ↔ (A ∧ B ∧ C → false)

-- B says: "Exactly one of us is a liar."
axiom B_statement : S_B ↔ ((A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C))

theorem identity_proof : (A = false) ∧ (C = true) ∧ (¬(B = true ∨ B = false)) :=
by
  sorry

end identity_proof_l195_195518


namespace smallest_positive_period_and_range_l195_195694

noncomputable def f (ω λ x : ℝ) : ℝ :=
  (sin (ω * x))^2 + 2 * sqrt 3 * sin (ω * x) * cos (ω * x) - (cos (ω * x))^2 + λ

theorem smallest_positive_period_and_range (ω : ℝ) (h_ω : 0 < ω ∧ ω < 2) 
  (λ : ℝ) (h_symmetric : ∀ x : ℝ, f ω λ (π/3 - x) = f ω λ (π/3 + x)) :
  (∃ T : ℝ, T = π ∧ ∀ x : ℝ, f ω λ (x + T) = f ω λ x) ∧
  (∃ λ : ℝ, f ω λ (π/6) = 0 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ π/2 → f ω λ x ∈ [-2, 1])) :=
begin
  sorry
end

end smallest_positive_period_and_range_l195_195694


namespace find_number_l195_195347

theorem find_number (x : ℝ) (h : (((18 + x) / 3 + 10) / 5 = 4)) : x = 12 :=
by
  sorry

end find_number_l195_195347


namespace sum_of_probabilities_l195_195989

theorem sum_of_probabilities (red_boxes blue_boxes : ℕ) (stacks : ℕ) (boxes_per_stack : ℕ)
  (h_red_boxes : red_boxes = 8) (h_blue_boxes : blue_boxes = 8) 
  (h_stacks : stacks = 4) (h_boxes_per_stack : boxes_per_stack = 4) :
  let m := 128
  let n := 715
  m + n = 843 :=
by
  have : red_boxes = 8 := h_red_boxes
  have : blue_boxes = 8 := h_blue_boxes
  have : stacks = 4 := h_stacks
  have : boxes_per_stack = 4 := h_boxes_per_stack
  let m := 128
  let n := 715
  have h_mn_rel_prime : Nat.gcd m n = 1 := by sorry -- Proof of gcd is 1
  exact (Nat.add m n).symm

end sum_of_probabilities_l195_195989


namespace BC_length_is_16_l195_195815

noncomputable def length_of_BC (α : Real) (sin_α : Real) (r : Real) : Real :=
  2 * r * Real.cos α

theorem BC_length_is_16 (α : Real) (sin_α : sin_α = Real.sqrt 45 / 7) (r : r = 28)
    (hypo_α : Real.sin α = Real.sqrt 45 / 7) : length_of_BC α sin_α r = 16 := by
  sorry

end BC_length_is_16_l195_195815


namespace number_of_pairs_l195_195328

theorem number_of_pairs (x y : ℤ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000) :
  (x^2 + y^2) % 7 = 0 → (∃ n : ℕ, n = 20164) :=
by {
  sorry
}

end number_of_pairs_l195_195328


namespace polygon_properties_l195_195168

theorem polygon_properties (sum_angles : ℝ) (one_angle : ℝ) (n : ℕ) :
  sum_angles = 3420 ∧ one_angle = 160 ∧ (180 * (n - 2) = sum_angles) →
  n = 21 ∧ (∀ remaining_angle, (remaining_angle ∈ (range n).erase 160) → remaining_angle = 163) :=
by
  sorry

end polygon_properties_l195_195168


namespace jessies_weight_after_first_week_l195_195374

-- Definitions from the conditions
def initial_weight : ℕ := 92
def first_week_weight_loss : ℕ := 56

-- The theorem statement
theorem jessies_weight_after_first_week : initial_weight - first_week_weight_loss = 36 := by
  -- Skip the proof
  sorry

end jessies_weight_after_first_week_l195_195374


namespace rectangle_area_l195_195644

-- Define the width and length of the rectangle
def w : ℚ := 20 / 3
def l : ℚ := 2 * w

-- Define the perimeter constraint
def perimeter_condition : Prop := 2 * (l + w) = 40

-- Define the area of the rectangle
def area : ℚ := l * w

-- The theorem to prove
theorem rectangle_area : perimeter_condition → area = 800 / 9 :=
by
  intro h
  have hw : w = 20 / 3 := rfl
  have hl : l = 2 * w := rfl
  have hp : 2 * (l + w) = 40 := h
  sorry

end rectangle_area_l195_195644


namespace maximize_total_profit_l195_195949

noncomputable def p (t : ℝ) (a : ℝ) : ℝ := a * t^3 + 21 * t
noncomputable def g (t : ℝ) (a b : ℝ) : ℝ := -2 * a * (t - b)^2

theorem maximize_total_profit :
  let a := (-1 / 60 : ℝ),
      b := (110 : ℝ),
      f (x : ℝ) := p x a + g (200 - x) a b
  in ∀ (x : ℝ), 10 ≤ x ∧ x ≤ 190 →
      f x = -1/60 * (x^3 - 2*x^2 - 900*x - 16200) ∧
      (∃ max_x : ℝ, max_x = 18 ∧ f max_x = 453.6) ↔ x = 18 ∧ f 18 = 453.6 :=
begin
  intros a b f h x hx,
  sorry
end

end maximize_total_profit_l195_195949


namespace similar_triangle_longest_side_length_l195_195848

theorem similar_triangle_longest_side_length
  (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) (p : ℝ) (h₄ : p = 150) :
  ∃ x : ℝ, 13 * x = 65 ∧ (5 * x + 12 * x + 13 * x = p) := 
by
  use 5
  split
  · sorry   -- This part will prove 13 * x = 65
  · sorry   -- This part will prove 5 * x + 12 * x + 13 * x = 150

end similar_triangle_longest_side_length_l195_195848


namespace count_correct_propositions_l195_195181

theorem count_correct_propositions 
  (a b c : ℝ^3) -- assuming the vectors are in R^3
  (h1 : a • b = 0)
  (h2 : ∥a∥ = ∥b∥)
  (h3 : a • b = a • c)
  (h4a : ∃ k, b = k • a)
  (h4b : ∃ k, c = k • b) :
  finset.filter (λ i, 
    match i with
      | 1 => a • b = 0 → a = 0 ∨ b = 0
      | 2 => ∥a∥ = ∥b∥ → (a + b) • (a - b) = 0
      | 3 => a • b = a • c → b = c
      | 4 => (a • b = 0 ∨ b = k • a) ∧ (b • c = 0 ∨ c = k' • b) → (a • c = 0 ∨ c = k'' • a)
    end)
    (finset.range 4) |>.card = 1 := 
sorry

end count_correct_propositions_l195_195181


namespace largest_root_of_equation_l195_195250

theorem largest_root_of_equation :
  ∃ (x : ℝ), x ∈ Ioo (1/4) 2 ∧ 
              (| (Real.sin (2 * Real.pi * x) - Real.cos (Real.pi * x)) | = 
               | (| Real.sin (2 * Real.pi * x) | - | Real.cos (Real.pi * x) | )) ∧
              ∀ (y : ℝ), y ∈ Ioo (1/4) 2 ∧ 
                         (| (Real.sin (2 * Real.pi * y) - Real.cos (Real.pi * y)) | = 
                          | (| Real.sin (2 * Real.pi * y) | - | Real.cos (Real.pi * y) | )) → y ≤ x := 
begin
  existsi (3/2 : ℝ),
  sorry
end

end largest_root_of_equation_l195_195250


namespace card_pair_probability_l195_195923

theorem card_pair_probability :
  ∃ m n : ℕ, 
    0 < m ∧ 0 < n ∧ 
    (m.gcd n = 1) ∧ 
    (m = 73) ∧ 
    (n = 1225) ∧ 
    (m + n = 1298) := by
{
  sorry
}

end card_pair_probability_l195_195923


namespace smallest_clock_equiv_to_square_greater_than_10_l195_195811

def clock_equiv (h k : ℕ) : Prop :=
  (h % 12) = (k % 12)

theorem smallest_clock_equiv_to_square_greater_than_10 : ∃ h > 10, clock_equiv h (h * h) ∧ ∀ h' > 10, clock_equiv h' (h' * h') → h ≤ h' :=
by
  sorry

end smallest_clock_equiv_to_square_greater_than_10_l195_195811


namespace sum_first_11_terms_l195_195752

variable (a : ℕ → ℚ) -- define a sequence of rational numbers, assuming n is in natural numbers
variables (d : ℚ) (a_1 : ℚ) -- common difference d and first term a_1

-- Define the arithmetic sequence: a_n = a_1 + (n - 1) * d
def arithmetic_sequence (n : ℕ) : ℚ := a_1 + (n - 1) * d

-- Define S_n as the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℚ := (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given condition: a_4 + a_8 = 16
axiom condition : arithmetic_sequence a 4 + arithmetic_sequence a 8 = 16

-- The goal is to prove S_11 = 88
theorem sum_first_11_terms : S a 11 = 88 :=
by 
  sorry

end sum_first_11_terms_l195_195752


namespace quadratic_distinct_positive_roots_l195_195622

theorem quadratic_distinct_positive_roots (a : ℝ) : 
  9 * (a - 2) > 0 → 
  a > 0 → 
  a^2 - 9 * a + 18 > 0 → 
  a ≠ 11 → 
  (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a) := 
by 
  intros h1 h2 h3 h4
  sorry

end quadratic_distinct_positive_roots_l195_195622


namespace appropriate_sampling_method_most_appropriate_sampling_method_l195_195579

/-- A school has different groups of students and a sample needs to be drawn. -/
def student_groups :=
  { elementary_students : Nat // elementary_students = 125 } ×
  { junior_high_students : Nat // junior_high_students = 280 } ×
  { high_school_students : Nat // high_school_students = 95 } 

/-- A sample size of 100 students needs to be drawn from the groups. -/
def sample_size (n : Nat) : Prop := n = 100

/-- Define the stratified sampling method over the student groups. -/
def stratified_sampling := 
  ∀ (S : student_groups), 
    sample_size 100 →
    ∃ (method : String), method = "Stratified sampling"

/-- Prove the most suitable sampling method is stratified sampling. -/
theorem appropriate_sampling_method :
  stratified_sampling :=
by
  intros,
  exact ⟨_, rfl⟩

theorem most_appropriate_sampling_method (S : student_groups) (h : sample_size 100) :
  stratified_sampling :=
by
  exact appropriate_sampling_method

end appropriate_sampling_method_most_appropriate_sampling_method_l195_195579


namespace susie_remaining_money_l195_195837

noncomputable def calculate_remaining_money : Float :=
  let weekday_hours := 4.0
  let weekday_rate := 12.0
  let weekdays := 5.0
  let weekend_hours := 2.5
  let weekend_rate := 15.0
  let weekends := 2.0
  let total_weekday_earnings := weekday_hours * weekday_rate * weekdays
  let total_weekend_earnings := weekend_hours * weekend_rate * weekends
  let total_earnings := total_weekday_earnings + total_weekend_earnings
  let spent_makeup := 3 / 8 * total_earnings
  let remaining_after_makeup := total_earnings - spent_makeup
  let spent_skincare := 2 / 5 * remaining_after_makeup
  let remaining_after_skincare := remaining_after_makeup - spent_skincare
  let spent_cellphone := 1 / 6 * remaining_after_skincare
  let final_remaining := remaining_after_skincare - spent_cellphone
  final_remaining

theorem susie_remaining_money : calculate_remaining_money = 98.4375 := by
  sorry

end susie_remaining_money_l195_195837


namespace area_of_region_R_l195_195517

variables (A B C D E : Point)
variables (AD : Segment) (AB BE : Triangle)
variables (distance : Point → Segment → Real)

-- Define the vertices of the unit square
def unit_square (A B C D : Point) :=
  distance A B = 1 ∧ distance B C = 1 ∧ distance C D = 1 ∧ distance D A = 1

-- Define the right triangle condition with E inside the square
def right_triangle_abe (A B E : Point) :=
  angle A B E = 90 ∧ E ∈ interior (unit_square A B C D)

-- Define the strip region
def strip_region (P : Point) (AD : Segment) :=
  distance P AD > 1/4 ∧ distance P AD < 1/2

-- Define the region R as per the conditions
def region_R (P : Point) :=
  P ∈ unit_square A B C D ∧ ¬P ∈ triangle A B E ∧ strip_region P AD

-- Statement for the area of region R
theorem area_of_region_R :
  area (region_R A B C D E AD) = 7 / 32 :=
sorry

end area_of_region_R_l195_195517


namespace percentage_difference_l195_195904

theorem percentage_difference : (70 / 100 : ℝ) * 100 - (60 / 100 : ℝ) * 80 = 22 := by
  sorry

end percentage_difference_l195_195904


namespace median_eq_range_le_l195_195284

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end median_eq_range_le_l195_195284


namespace angles_of_MNP_fixed_l195_195654

theorem angles_of_MNP_fixed (A B C M N P : EuclideanGeometry.Point) 
  (cond : AM - BC = BN - AC = CP - AB) : 
  ∃ α β γ : ℝ, 
    α = 90 - ∠A / 2 ∧ 
    β = 90 - ∠B / 2 ∧ 
    γ = 90 - ∠C / 2 ∧ 
    ∠MNP = α ∧ 
    ∠PNM = β ∧ 
    ∠NMP = γ := 
sorry

end angles_of_MNP_fixed_l195_195654


namespace darwin_spending_fraction_l195_195626

theorem darwin_spending_fraction {x : ℝ} (h1 : 600 - 600 * x - (1 / 4) * (600 - 600 * x) = 300) :
  x = 1 / 3 :=
sorry

end darwin_spending_fraction_l195_195626


namespace calculate_total_loss_l195_195804

def original_cost_paintings (num_paintings : ℕ) (cost_per_painting : ℕ) : ℕ :=
  num_paintings * cost_per_painting

def original_cost_wooden_toys (num_toys : ℕ) (cost_per_toy : ℕ) : ℕ :=
  num_toys * cost_per_toy

def total_original_cost (cost_paintings : ℕ) (cost_toys : ℕ) : ℕ :=
  cost_paintings + cost_toys

def selling_price_painting (original_price : ℕ) (discount_percent : ℕ) : ℕ :=
  original_price - (original_price * discount_percent / 100)

def selling_price_toy (original_price : ℕ) (discount_percent : ℕ) : ℕ :=
  original_price - (original_price * discount_percent / 100)

def total_selling_price (num_paintings : ℕ) (selling_price_painting : ℕ) (num_toys : ℕ) (selling_price_toy : ℕ) : ℕ :=
  (num_paintings * selling_price_painting) + (num_toys * selling_price_toy)

def total_loss (original_cost : ℕ) (selling_cost : ℕ) : ℕ :=
  original_cost - selling_cost

theorem calculate_total_loss :
  let num_paintings := 10
  let cost_per_painting := 40
  let num_toys := 8
  let cost_per_toy := 20
  let discount_percent_painting := 10
  let discount_percent_toy := 15

  original_cost_paintings num_paintings cost_per_painting + original_cost_wooden_toys num_toys cost_per_toy 
  = 560 →
  
  total_original_cost 400 160 = 560 →

  selling_price_painting cost_per_painting discount_percent_painting = 36 →
  selling_price_toy cost_per_toy discount_percent_toy = 17 →
  
  total_selling_price num_paintings 36 num_toys 17 
  = 496 →
  
  total_loss 560 496 = 64 :=
by
  intros
  simp_all
  sorry

end calculate_total_loss_l195_195804


namespace polar_coordinates_of_point_l195_195200

theorem polar_coordinates_of_point : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (-1:ℝ, Real.sqrt 3) = (r * Real.cos θ, r * Real.sin θ) ∧ r = 2 ∧ θ = 2 * Real.pi / 3 :=
by
  exists 2, 2 * Real.pi / 3
  sorry

end polar_coordinates_of_point_l195_195200


namespace yellow_shirts_per_pack_l195_195024

theorem yellow_shirts_per_pack :
  ∃ Y : ℕ,
    (3 * 5) + 3 * Y = 21 ∧ Y = 2 :=
by
  use 2
  sorry

end yellow_shirts_per_pack_l195_195024


namespace smallest_of_three_consecutive_even_numbers_l195_195861

def sum_of_three_consecutive_even_numbers (n : ℕ) : Prop :=
  n + (n + 2) + (n + 4) = 162

theorem smallest_of_three_consecutive_even_numbers (n : ℕ) (h : sum_of_three_consecutive_even_numbers n) : n = 52 :=
by
  sorry

end smallest_of_three_consecutive_even_numbers_l195_195861


namespace jerry_more_votes_l195_195373

-- Definitions based on conditions
def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375
def john_votes : ℕ := total_votes - jerry_votes

-- Theorem to prove the number of more votes Jerry received than John Pavich
theorem jerry_more_votes : jerry_votes - john_votes = 20196 :=
by
  -- Definitions and proof can be filled out here as required.
  sorry

end jerry_more_votes_l195_195373


namespace asteroid_fragments_total_is_70_l195_195113

noncomputable def total_fragments (X : ℕ) :=
  (X / 5) + 26 + (((X - (X / 5) - 26) / (X / 7)) * (X / 7)) = X

theorem asteroid_fragments_total_is_70 :
  ∃ (X : ℕ), total_fragments X ∧ X = 70 :=
by
  use 70
  unfold total_fragments
  norm_num
  sorry

end asteroid_fragments_total_is_70_l195_195113


namespace two_lines_parallel_to_same_line_l195_195530

theorem two_lines_parallel_to_same_line (l m n : Line) : 
  (l ∥ m) → (m ∥ n) → (l ∥ n) :=
by
  intro h_lm h_mn
  -- Proof goes here
  sorry

end two_lines_parallel_to_same_line_l195_195530


namespace balloon_permutations_l195_195219

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l195_195219


namespace min_value_of_sum_l195_195298

theorem min_value_of_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1 / x + 1 / y + 1 / z = 1) :
  x + 4 * y + 9 * z ≥ 36 ∧ (x + 4 * y + 9 * z = 36 ↔ x = 6 ∧ y = 3 ∧ z = 2) := 
sorry

end min_value_of_sum_l195_195298


namespace arrangement_methods_l195_195065

-- Define the set of reporters
def male_reporters : set ℕ := {1, 2, 3} -- C, D, E
def female_reporters : set ℕ := {4, 5} -- A, B

-- Define the set of tasks
def tasks : set ℕ := {1, 2, 3, 4} -- "carrying equipment", "interviewing objects", "writing scripts", "compiling and editing"

-- Constraint: "A and B do not participate in carrying equipment"
def carrying_equipment : ℕ := 1
def non_carrying_tasks : set ℕ := {2, 3, 4}

-- Prove that the number of different arrangement methods is 126
theorem arrangement_methods (n : ℕ) (male_reporters female_reporters tasks : set ℕ)
  (hc : carrying_equipment ∈ tasks)
  (hf : ∀ (f ∈ female_reporters), f ≠ carrying_equipment)
  : n = 126 :=
sorry

end arrangement_methods_l195_195065


namespace total_spending_l195_195990

-- Define the condition of spending for each day
def friday_spending : ℝ := 20
def saturday_spending : ℝ := 2 * friday_spending
def sunday_spending : ℝ := 3 * friday_spending

-- Define the statement to be proven
theorem total_spending : friday_spending + saturday_spending + sunday_spending = 120 :=
by
  -- Provide conditions and calculations here (if needed)
  sorry

end total_spending_l195_195990


namespace daily_sales_volume_selling_price_for_profit_l195_195801

noncomputable def cost_price : ℝ := 40
noncomputable def initial_selling_price : ℝ := 60
noncomputable def initial_sales_volume : ℝ := 20
noncomputable def price_decrease_per_increase : ℝ := 5
noncomputable def volume_increase_per_decrease : ℝ := 10

theorem daily_sales_volume (p : ℝ) (v : ℝ) :
  v = initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease :=
sorry

theorem selling_price_for_profit (p : ℝ) (profit : ℝ) :
  profit = (p - cost_price) * (initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease) → p = 54 :=
sorry

end daily_sales_volume_selling_price_for_profit_l195_195801


namespace min_coins_needed_l195_195197

variable (d n : ℕ)

theorem min_coins_needed (D N : ℕ) : 40 + 2.5 + 0.10 * d + 0.05 * n ≥ 45.50 → d + n ≥ 30 := by
  intros h
  -- since we need to perform rounding and dealing with mixed notation, let's assume the problem simplified to basic linear arithmetic
  calc 
    (0.1 : ℚ) * D + (0.05 : ℚ) * N ≥ (3 : ℚ) : 
      -- assume the hypothesis h gives us this simplified form
    2 * D + N ≥ (60 : ℕ) : sorry
    -- this concludes that d + n ≥ 30. a more refined calculation would be yet considered
    D + N ≥ 30 : sorry

end min_coins_needed_l195_195197


namespace general_form_of_pattern_l195_195412

theorem general_form_of_pattern (n : ℤ) : 
  ∑ p in [(2 / (2 - 4)) + (6 / (6 - 4)), 
           (5 / (5 - 4)) + (3 / (3 - 4)), 
           (7 / (7 - 4)) + (1 / (1 - 4)), 
           (10 / (10 - 4)) + ((-2) / (-2 - 4))], p = 
  2 :=
begin
  sorry
end

end general_form_of_pattern_l195_195412


namespace balloon_permutations_l195_195235

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l195_195235


namespace small_mold_radius_l195_195571

theorem small_mold_radius (r : ℝ) (n : ℝ) (s : ℝ) :
    r = 2 ∧ n = 8 ∧ (1 / 2) * (2 / 3) * Real.pi * r^3 = (8 * (2 / 3) * Real.pi * s^3) → s = 1 :=
by
  sorry

end small_mold_radius_l195_195571


namespace greatest_integer_sum_of_integers_l195_195520

-- Definition of the quadratic function
def quadratic_expr (n : ℤ) : ℤ := n^2 - 15 * n + 56

-- The greatest integer n such that quadratic_expr n ≤ 0
theorem greatest_integer (n : ℤ) (h : quadratic_expr n ≤ 0) : n ≤ 8 := 
  sorry

-- All integers that satisfy the quadratic inequality
theorem sum_of_integers (sum_n : ℤ) (h : ∀ n : ℤ, 7 ≤ n ∧ n ≤ 8 → quadratic_expr n ≤ 0) 
  (sum_eq : sum_n = 7 + 8) : sum_n = 15 :=
  sorry

end greatest_integer_sum_of_integers_l195_195520


namespace probability_sqrt3_le_abs_v_plus_w_l195_195001

theorem probability_sqrt3_le_abs_v_plus_w :
  (∃ v w : ℂ, v ≠ w ∧ v^2023 = 1 ∧ w^2023 = 1 ∧ (filter (λ v : ℂ, sqrt 3 ≤ abs (v + w)) (set.univ : set ℂ)).card / 2022 = 337 / 1011) :=
sorry

end probability_sqrt3_le_abs_v_plus_w_l195_195001


namespace area_trapezoid_AFGE_l195_195838

theorem area_trapezoid_AFGE
  (ABCD_area : ℝ)
  (F_on_BC : Prop)
  (D_midpoint_EG : Prop)
  (h_ABCD_area : ABCD_area = 2011) :
  ∃ AFGE_area : ℝ, AFGE_area = 2011 :=
by
  -- Giving hypotheses
  let ABCD_area := 2011 in
  let F_on_BC := True in
  let D_midpoint_EG := True in
  exists.intro 2011 sorry

end area_trapezoid_AFGE_l195_195838


namespace solution_set_range_ineq_l195_195254

theorem solution_set_range_ineq (m : ℝ) :
  ∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0 ↔ (-5: ℝ)⁻¹ < m ∧ m ≤ 3 :=
by
  sorry

end solution_set_range_ineq_l195_195254


namespace max_area_triangle_l195_195683

-- Define basic entities and conditions
variables {V : Type*} [InnerProductSpace ℝ V]
variables {A B C O : V}
variable {t : ℝ}

-- Given conditions
def O_is_circumcenter (A B C O : V) : Prop :=
IsCircumcenter O A B C  -- O is the circumcenter of triangle ABC

def vector_equation (A B C O : V) (t : ℝ) : Prop :=
(O - C) = t • (A - C) + ((1 / 2) - (3 * t / 4)) • (B - C)  -- Given vector equation

def length_AB (A B : V) : Prop :=
‖A - B‖ = 3  -- Given length of side AB

-- The theorem to prove
theorem max_area_triangle (hO : O_is_circumcenter A B C O)
                       (hvec : vector_equation A B C O t)
                       (hab : length_AB A B) :
  ∃ max_area, max_area = 9 :=
begin
  sorry  -- Proof to be completed
end

end max_area_triangle_l195_195683


namespace term_2157_is_153_l195_195449

def digit_cubes_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d, d^3) |>.sum

noncomputable def sequence : ℕ → ℕ
| 0     := 2157
| (n+1) := digit_cubes_sum (sequence n)

theorem term_2157_is_153 : sequence 2156 = 153 := 
  sorry

end term_2157_is_153_l195_195449


namespace cupcakes_left_correct_l195_195201

-- Definitions based on conditions
def total_cupcakes : ℕ := 10 * 12 + 1 * 12 / 2
def total_students : ℕ := 48
def absent_students : ℕ := 6 
def field_trip_students : ℕ := 8
def teachers : ℕ := 2
def teachers_aids : ℕ := 2

-- Function to calculate the number of present people
def total_present_people : ℕ :=
  total_students - absent_students - field_trip_students + teachers + teachers_aids

-- Function to calculate the cupcakes left
def cupcakes_left : ℕ := total_cupcakes - total_present_people

-- The theorem to prove
theorem cupcakes_left_correct : cupcakes_left = 85 := 
by
  -- This is where the proof would go
  sorry

end cupcakes_left_correct_l195_195201


namespace find_complex_z_l195_195717

theorem find_complex_z (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
sorry

end find_complex_z_l195_195717


namespace y_intercept_of_line_b_l195_195018

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l195_195018


namespace least_value_a_l195_195134

theorem least_value_a (a : ℤ) (h : 240 ∣ a^3) : a = 60 :=
begin
  sorry
end

end least_value_a_l195_195134


namespace least_product_of_distinct_primes_greater_than_30_l195_195507

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l195_195507


namespace length_of_bullet_train_l195_195560

theorem length_of_bullet_train
    (speed_train : ℝ) (speed_man : ℝ) (time : ℝ)
    (v_train : speed_train = 50)
    (v_man : speed_man = 4)
    (t_pass : time = 8) :
    let relative_speed_kmph := speed_train + speed_man,
        relative_speed_mps := (relative_speed_kmph * 1000) / 3600,
        length_train := relative_speed_mps * time
    in length_train = 120 :=
by
    intros,
    sorry

end length_of_bullet_train_l195_195560


namespace total_marbles_l195_195109

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l195_195109


namespace diff_of_squares_635_615_l195_195127

theorem diff_of_squares_635_615 : 635^2 - 615^2 = 25000 :=
by
  sorry

end diff_of_squares_635_615_l195_195127


namespace area_of_triangle_ABC_l195_195761

noncomputable def triangle_area (A B C : ℝ) : ℝ :=
  0.5 * B * C * (real.sqrt (1 - (real.cos A)^2))

theorem area_of_triangle_ABC :
  ∀ (B C : ℝ), 
  cos B = 1/2 ∧ 
  cos C = 1/7 ∧ 
  let medianLength := ℝ.sqrt 21 in 
  ∃ area : ℝ, area = 10 * ℝ.sqrt 3 :=
by
  sorry

end area_of_triangle_ABC_l195_195761


namespace initial_number_of_friends_is_six_l195_195925

theorem initial_number_of_friends_is_six
  (car_cost : ℕ)
  (car_wash_earnings : ℕ)
  (F : ℕ)
  (additional_cost_when_one_friend_leaves : ℕ)
  (h1 : car_cost = 1700)
  (h2 : car_wash_earnings = 500)
  (remaining_cost := car_cost - car_wash_earnings)
  (cost_per_friend_before := remaining_cost / F)
  (cost_per_friend_after := remaining_cost / (F - 1))
  (h3 : additional_cost_when_one_friend_leaves = 40)
  (h4 : cost_per_friend_after = cost_per_friend_before + additional_cost_when_one_friend_leaves) :
  F = 6 :=
by
  sorry

end initial_number_of_friends_is_six_l195_195925


namespace taxi_ride_distance_l195_195920

theorem taxi_ride_distance
  (initial_charge : ℝ) (additional_charge : ℝ) 
  (total_charge : ℝ) (initial_increment : ℝ) (distance_increment : ℝ)
  (initial_charge_eq : initial_charge = 2.10) 
  (additional_charge_eq : additional_charge = 0.40) 
  (total_charge_eq : total_charge = 17.70) 
  (initial_increment_eq : initial_increment = 1/5) 
  (distance_increment_eq : distance_increment = 1/5) : 
  (distance : ℝ) = 8 :=
by sorry

end taxi_ride_distance_l195_195920


namespace Luke_spends_per_week_l195_195022

theorem Luke_spends_per_week (mowing : ℕ) (weed_eating : ℕ) (weeks : ℕ) (total_money : ℕ) :
  mowing = 9 → weed_eating = 18 → total_money = mowing + weed_eating → weeks = 9 → (total_money / weeks) = 3 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h4]
  simp [h3]
  sorry

end Luke_spends_per_week_l195_195022


namespace median_eq_range_le_l195_195285

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end median_eq_range_le_l195_195285


namespace find_day_equal_kangaroos_l195_195379

-- Definitions to encapsulate the initial conditions and the purchasing patterns

def Kameron (t : ℕ) : ℕ := 100

def Bert (t : ℕ) : ℕ :=
  let seq := λ t, (t * (t + 1)) / 2 + 20
  20 + (t * (t + 1)) / 2

def Christina (t : ℕ) : ℕ :=
  45 + if t < 1 then 3 * t
       else if t < 4 then 3 * t + 3 * (t - 1)
       else (1 + t / 3) * 6 * (t / 3) + 9 * (t - (t / 3))

def David (t : ℕ) : ℕ :=
  let add_kangaroos (d : ℕ) := 5 - (d - 1) / 2
  10 + ∑ i in (range (t + 1)), add_kangaroos i

-- Main theorem statement 
theorem find_day_equal_kangaroos : ∃ t : ℕ, Bert t = 100 ∧ Christina t = 100 ∧ David t = 100 :=
begin
  sorry -- Proof to be filled in
end

end find_day_equal_kangaroos_l195_195379


namespace proof_problem_l195_195131

/- Define relevant concepts -/
def is_factor (a b : Nat) := ∃ k, b = a * k
def is_divisor := is_factor

/- Given conditions with their translations -/
def condition_A : Prop := is_factor 5 35
def condition_B : Prop := is_divisor 21 252 ∧ ¬ is_divisor 21 48
def condition_C : Prop := ¬ (is_divisor 15 90 ∨ is_divisor 15 74)
def condition_D : Prop := is_divisor 18 36 ∧ ¬ is_divisor 18 72
def condition_E : Prop := is_factor 9 180

/- The main proof problem statement -/
theorem proof_problem : condition_A ∧ condition_B ∧ ¬ condition_C ∧ ¬ condition_D ∧ condition_E :=
by
  sorry

end proof_problem_l195_195131


namespace arrange_students_l195_195179

theorem arrange_students (students : Fin 7 → Prop) : 
  ∃ arrangements : ℕ, arrangements = 140 :=
by
  -- Define selection of 6 out of 7
  let selection_ways := Nat.choose 7 6
  -- Define arrangement of 6 into two groups of 3 each
  let arrangement_ways := (Nat.choose 6 3) * (Nat.choose 3 3)
  -- Calculate total arrangements by multiplying the two values
  let total_arrangements := selection_ways * arrangement_ways
  use total_arrangements
  simp [selection_ways, arrangement_ways, total_arrangements]
  exact rfl

end arrange_students_l195_195179


namespace john_number_is_55_l195_195771

-- Define the problem and the condition
def johns_number (y : ℕ) : Prop :=
  ∃ c d e : ℕ,
  let result := 2 * y + 13 in
  let reversed_digits := (result % 10) * 100 + (((result % 100) / 10) * 10) + (result / 100) in
  (result < 1000) ∧ (321 <= reversed_digits) ∧ (reversed_digits <= 325)

-- The theorem to prove
theorem john_number_is_55 : johns_number 55 :=
by
  intro y,
  use 55,
  existsi [1, 2, 3],  -- Corresponding digits c = 1, d = 2, e = 3 for result 123
  simp only [johns_number],
  split,
  {
    -- Prove result < 1000
    exact nat.lt_add_of_pos_right (by norm_num),
  },
  split,
  {
    -- Prove 321 <= reversed_digits
    exact nat.le_refl 321,
  },
  {
    -- Prove reversed_digits <= 325
    exact nat.le_refl 325,
  },
  -- Left the proof with "sorry" for completeness as per guideline
  sorry

end john_number_is_55_l195_195771


namespace greatest_possible_x_l195_195074

theorem greatest_possible_x : ∀ x : ℕ, (lcm x (lcm 15 (lcm 18 21)) = 630) → x ≤ 630 :=
by
  intro x hx
  sorry

example : greatest_possible_x 630 (by simp [lcm, hx]) :=
by rfl

end greatest_possible_x_l195_195074


namespace basis_groups_l195_195594

-- Definition of vectors in group ①
def e1_group1 : ℝ × ℝ := (-1, 2)
def e2_group1 : ℝ × ℝ := (5, 7)

-- Definition of vectors in group ②
def e1_group2 : ℝ × ℝ := (3, 5)
def e2_group2 : ℝ × ℝ := (6, 10)

-- Definition of vectors in group ③
def e1_group3 : ℝ × ℝ := (2, 3)
def e2_group3 : ℝ × ℝ := (1/2, -3/4)

-- Definition of the cross product of 2D vectors (used to determine collinearity)
def cross_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.2 - v1.2 * v2.1

-- Theorem stating which groups can form a basis
theorem basis_groups :
  (cross_product e1_group1 e2_group1 ≠ 0) ∧
  (cross_product e1_group2 e2_group2 = 0) ∧
  (cross_product e1_group3 e2_group3 ≠ 0) :=
by sorry

end basis_groups_l195_195594


namespace mother_l195_195733

def problem_conditions (D M : ℤ) : Prop :=
  (2 * D + M = 70) ∧ (D + 2 * M = 95)

theorem mother's_age_is_40 (D M : ℤ) (h : problem_conditions D M) : M = 40 :=
by sorry

end mother_l195_195733


namespace adjacent_students_permutations_l195_195088

open Nat

/--
Given 6 students standing in a row, with students labeled A, B, and 4 others, 
prove that the number of permutations in which A and B are adjacent is 240.
-/
theorem adjacent_students_permutations (students : Fin 6 → ℕ) (A B : Fin 6) :
  A ≠ B → 
  ∃ n : ℕ, n = 240 ∧
  (count_adjacent_perms students A B) = n := 
sorry

-- A function to count the permutations where A and B are adjacent
noncomputable def count_adjacent_perms (students : Fin 6 → ℕ) (A B : Fin 6) : ℕ := 
  ((6 - 1)! * 2!)


end adjacent_students_permutations_l195_195088


namespace half_vector_mn_l195_195322

open ComplexConjugate

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def scalar_mult (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (a * v.1, a * v.2)

theorem half_vector_mn :
  let OM := (3, -2)
  let ON := (-5, -1)
  let MN := vec_sub ON OM
  let half_MN := scalar_mult (1/2) MN
  half_MN = (-4, 1/2) := by
  -- This will be replaced by the complete proof.
  simp only [OM, ON, MN, half_MN, vec_sub, scalar_mult, Prod.mk.injEq]
  norm_num
  sorry

end half_vector_mn_l195_195322


namespace edward_candy_purchase_l195_195144

theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) 
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := 
by 
  sorry

end edward_candy_purchase_l195_195144


namespace root_of_quad_eqn_l195_195713

theorem root_of_quad_eqn {b c : ℝ} (h : (1 + complex.I * real.sqrt 2) * (1 - complex.I * real.sqrt 2) = c ∧ 
                          (1 + complex.I * real.sqrt 2) + (1 - complex.I * real.sqrt 2) = -b ) :
  b = -2 ∧ c = 3 :=
by {
  sorry
}

end root_of_quad_eqn_l195_195713


namespace problem_proof_l195_195294

-- Define necessary conditions and questions
noncomputable def a_property (a : Real) := a > 1
noncomputable def max_val (a : Real) := Real.log 6 / Real.log a
noncomputable def min_val (a : Real) := Real.log 3 / Real.log a
noncomputable def difference_is_one (a : Real) := max_val a - min_val a = 1

-- Define equivalent proof problem
theorem problem_proof (a : Real) (h : a_property a) (h_diff : difference_is_one a) : 
  a = 2 ∧ 3 ^ (Real.log 6 / Real.log 2) = 6 ^ (Real.log 3 / Real.log 2) :=
sorry

end problem_proof_l195_195294


namespace QF_value_l195_195704

-- Define the parabola C
def parabola (x y : ℝ) := y^2 = 8 * x

-- Define focus F of the parabola C
def focusF := (0, 2 : ℝ)

-- Define equation of the line PF (given by the slope sqrt(3) and the y-intercept adjustment)
def linePF (x : ℝ) := real.sqrt 3 * (x - 2)

-- Define the condition that PF = 3 QF
def vector_condition (PF QF : ℝ) := PF = 3 * QF

-- Distance condition to compute |QF|
def distance_condition (d : ℝ) := |QF| = d

-- Prove the desired |QF| value
theorem QF_value : ∀ (QF : ℝ), parabola (2/3) (linePF (2/3)) ∧ vector_condition F QF → |QF| = 8 / 3 :=
by
  sorry

end QF_value_l195_195704


namespace acute_triangle_sum_l195_195309

-- Define the function f
def f (x : ℝ) : ℝ :=
  (1 + 2 * x - x^2) / ((1 + x) * (1 + x^2))

-- Define the problem statement as a theorem
theorem acute_triangle_sum (α β γ : ℝ)
  (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (sum_angles : α + β + γ = π)
  (ha : α < π/2) (hb : β < π/2) (hc : γ < π/2) :
  f (real.tan α) + f (real.tan β) + f (real.tan γ) +
  f (real.cot α) + f (real.cot β) + f (real.cot γ) = 3 :=
sorry

end acute_triangle_sum_l195_195309


namespace balloon_permutations_l195_195225

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l195_195225


namespace angles_not_equal_or_sum_180_l195_195764

-- Definition of a cube and angles within it
structure Cube :=
  (A B C D A' B' C' D' : Type)

-- Define the conditions of perpendicular lines and angles
variable (c : Cube)

def perpendicular_lines : Prop :=
  ⟦c.A⟧ ⟋ ⟦c.BC'⟧ ∧ ⟦c.BC⟧ ⟋ ⟦c.C'C⟧

def angle_ABC := 90
def angle_BC'C := 45

-- The statement to be proved is the given statement is false
theorem angles_not_equal_or_sum_180 :
  ¬ ((angle_ABC = angle_BC'C) ∨ (angle_ABC + angle_BC'C = 180)) :=
by
  sorry

end angles_not_equal_or_sum_180_l195_195764


namespace combination_of_seven_choose_three_l195_195425

theorem combination_of_seven_choose_three : nat.choose 7 3 = 35 :=
by {
  sorry
}

end combination_of_seven_choose_three_l195_195425


namespace periodic_odd_function_l195_195682

theorem periodic_odd_function (f : ℝ → ℝ) (period : ℝ) (h_periodic : ∀ x, f (x + period) = f x) (h_odd : ∀ x, f (-x) = -f x) (h_value : f (-3) = 1) (α : ℝ) (h_tan : Real.tan α = 2) :
  f (20 * Real.sin α * Real.cos α) = -1 := 
sorry

end periodic_odd_function_l195_195682


namespace find_pairs_l195_195997

-- Define predicative statements for the conditions
def is_integer (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = n

def condition1 (m n : ℕ) : Prop := 
  (n^2 + 1) % (2 * m) = 0

def condition2 (m n : ℕ) : Prop := 
  is_integer (Real.sqrt (2^(n-1) + m + 4))

-- The goal is to find the pairs of positive integers
theorem find_pairs (m n : ℕ) (h1: condition1 m n) (h2: condition2 m n) : 
  (m = 61 ∧ n = 11) :=
sorry

end find_pairs_l195_195997


namespace log_sum_tan_eq_zero_l195_195239

theorem log_sum_tan_eq_zero :
  (∑ n in finset.range 44, real.log10 ((real.tan (real.pi / 180 * (n + 1))) ^ 2)) = 0 :=
by
  sorry

end log_sum_tan_eq_zero_l195_195239


namespace number_of_schools_l195_195242

theorem number_of_schools (n : ℕ) (a : ℕ) :
  (70 < 3 * n) ∧ (3 * n < 120) ∧ (a = (3 * n + 1) / 2) ∧ (42 < a) ∧ (a < 57) ∧ (a ∈ ℕ) 
  → n = 33 := 
by 
  sorry

end number_of_schools_l195_195242


namespace mark_profit_l195_195027

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l195_195027


namespace min_tiles_needed_to_cover_rectangle_l195_195167

theorem min_tiles_needed_to_cover_rectangle : 
  let tile_length := 2
  let tile_width := 3
  let region_length := 3 * 12
  let region_width := 6 * 12
  let room_area := region_length * region_width
  let tile_area := tile_length * tile_width
  room_area / tile_area = 432 :=
by
  let tile_length := 2
  let tile_width := 3
  let region_length := 3 * 12
  let region_width := 6 * 12
  let room_area := region_length * region_width
  let tile_area := tile_length * tile_width
  -- The key calculation explicitly listed
  have h1 : room_area = 36 * 72 := rfl
  have h2 : tile_area = 2 * 3 := rfl
  have h3 : room_area = 2592 := by simp [h1]
  have h4 : tile_area = 6 := by simp [h2]
  have h5 : room_area / tile_area = 2592 / 6 := by simp [h3, h4]
  have h6 : 2592 / 6 = 432 := by norm_num
  show 2592 / 6 = 432 from h6

end min_tiles_needed_to_cover_rectangle_l195_195167


namespace minimum_inscribed_quadrilateral_area_l195_195888

-- Define the problem's conditions.
variables (A B C D : Type) [EuclideanGeometry A B C D]

-- Main theorem statement.
theorem minimum_inscribed_quadrilateral_area (quad : unit_area_quadrilateral A B C D) :
  ∃ (inscribed_quad : quadrilateral A B C D),
  (∀ (tri : triangle A B C), area tri = area (triangle A B D)) →
  area inscribed_quad ≥ 1 / 2 := 
sorry

end minimum_inscribed_quadrilateral_area_l195_195888


namespace inclination_angle_range_l195_195321

theorem inclination_angle_range (k : ℝ) (α : ℝ) (h1 : -1 ≤ k) (h2 : k < 1)
  (h3 : k = Real.tan α) (h4 : 0 ≤ α) (h5 : α < 180) :
  (0 ≤ α ∧ α < 45) ∨ (135 ≤ α ∧ α < 180) :=
sorry

end inclination_angle_range_l195_195321


namespace least_product_of_primes_over_30_l195_195477

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l195_195477


namespace inequality_of_sequence_l195_195265

-- We define the notion of an arithmetic sequence and its sum
def arith_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * a1 + (n * (n-1) / 2) * d

-- Define the sum term S_n for n
def S (n : ℕ) (a1 d : ℝ) : ℝ :=
  arith_seq a1 d n

theorem inequality_of_sequence {a1 d : ℝ} (h5 : S 5 a1 d < S 6 a1 d) (h67 : S 6 a1 d = S 7 a1 d) (h78 : S 7 a1 d > S 8 a1 d) : S 9 a1 d > S 8 a1 d :=
begin
  sorry
end

end inequality_of_sequence_l195_195265


namespace inequality_solution_set_l195_195696

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

theorem inequality_solution_set :
  { x : ℝ | (x+1) * f x > 2 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end inequality_solution_set_l195_195696


namespace distance_of_given_parallel_lines_l195_195686

def parallel_lines (a b c a' b' c' : ℝ) : Prop := (a' * b - a * b') = 0

def distance_between_parallel_lines (a b c a' b' c' : ℝ) [parallel_lines a b c a' b' c'] : ℝ :=
  |c - c'| / (Real.sqrt (a ^ 2 + b ^ 2))

theorem distance_of_given_parallel_lines :
  parallel_lines 3 m (-3) 6 4 1 → distance_between_parallel_lines 3 m (-3) 6 4 1 = 7 * Real.sqrt 13 / 26 :=
by
  intro h
  sorry

end distance_of_given_parallel_lines_l195_195686


namespace probability_black_ball_l195_195762

variable (total_balls : ℕ) (red_balls : ℕ) (white_ball_probability : ℚ)

theorem probability_black_ball :
  total_balls = 100 →
  red_balls = 45 →
  white_ball_probability = 0.23 →
  (total_balls - red_balls - (white_ball_probability * total_balls).toNat) / total_balls = 0.32 := 
by
  intros ht hr hp
  sorry

end probability_black_ball_l195_195762


namespace defective_bolt_probability_l195_195955

noncomputable def machine1_prob : ℝ := 0.30
noncomputable def machine2_prob : ℝ := 0.25
noncomputable def machine3_prob : ℝ := 0.45

noncomputable def defect_prob_machine1 : ℝ := 0.02
noncomputable def defect_prob_machine2 : ℝ := 0.01
noncomputable def defect_prob_machine3 : ℝ := 0.03

noncomputable def total_defect_prob : ℝ :=
  machine1_prob * defect_prob_machine1 +
  machine2_prob * defect_prob_machine2 +
  machine3_prob * defect_prob_machine3

theorem defective_bolt_probability : total_defect_prob = 0.022 := by
  sorry

end defective_bolt_probability_l195_195955


namespace semicircle_divides_equilateral_triangle_l195_195441
noncomputable def semicircle_divides_triangle (T : Triangle) (C : Circle) : Prop :=
  equilateral T ∧ T.base = C.diameter →
  (∃ p q r : Point, p ∈ C ∧ q ∈ C ∧ r ∈ C ∧ 
   divides_into_n_parts T.side p q && 
   divides_into_n_parts T.side q r &&
   divides_into_n_parts T.side r p && 
   divides_into_m_parts T.base_p q)

theorem semicircle_divides_equilateral_triangle (T : Triangle) (C : Circle) :
  semicircle_divides_triangle T C = 
    (one_side T.base into 3 equal parts ∧ 
     other_two_sides_divided T.sides into 2 equal parts)
:= by 
   sorry

end semicircle_divides_equilateral_triangle_l195_195441


namespace solve_for_x_l195_195526

theorem solve_for_x (x : ℝ) : (1 + 2*x + 3*x^2) / (3 + 2*x + x^2) = 3 → x = -2 :=
by
  intro h
  sorry

end solve_for_x_l195_195526


namespace sum_of_remainders_of_four_digit_number_with_consecutive_digits_l195_195156

open Finset

theorem sum_of_remainders_of_four_digit_number_with_consecutive_digits : 
  let possible_remainders := {a | a ∈ range 7}.map (λ a, (a + 2) % 11)
  in possible_remainders.sum = 35 := by
{
  sorry
}

end sum_of_remainders_of_four_digit_number_with_consecutive_digits_l195_195156


namespace y_intercept_of_line_b_l195_195019

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l195_195019


namespace smallest_solution_eq_l195_195258

noncomputable def smallest_solution := 4 - Real.sqrt 3

theorem smallest_solution_eq (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 3 / (x - 4)) → x = smallest_solution :=
sorry

end smallest_solution_eq_l195_195258


namespace product_of_solutions_l195_195650

theorem product_of_solutions (p : ℕ) (prime_p : Nat.Prime p) (h : ∃ (n : ℕ), n^2 - 31 * n + 240 = p) :
  (∀ (n : ℕ), n^2 - 31 * n + 240 = p → (n = 14 ∨ n = 17) ) →
  ∏ (n : ℕ) in {n | n = 14 ∨ n = 17}, n = 238 :=
by
  sorry

end product_of_solutions_l195_195650


namespace half_angle_in_quadrant_l195_195292

theorem half_angle_in_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 / 2) * Real.pi) :
  (π / 2 < α / 2 ∧ α / 2 < π) ∨ (3 * π / 2 < α / 2 ∧ α / 2 < 2 * π) :=
sorry

end half_angle_in_quadrant_l195_195292


namespace total_cost_in_dollars_l195_195436

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end total_cost_in_dollars_l195_195436


namespace max_count_greater_than_20_l195_195085

theorem max_count_greater_than_20 (a b c d e f g : ℤ) 
  (h1 : a + b + c + d + e + f + g = 10)
  (h2 : ∃ x y, (x < -5) ∧ (y < -5) ∧ (x ≠ y) ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f ∨ x = g) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f ∨ y = g))
  (h3 : ∀ i, i ∈ {a, b, c, d, e, f, g} → i ≥ 21 → i = a ∨ i = b ∨ i = c ∨ i = d ∨ i = e ∨ i = f ∨ i = g) :
  ∃ k, k ≤ 4 ∧ (∀ s t u v w x y, s + t + u + v + w + x + y = 10 → (∃ x y, (x < -5) ∧ (y < -5) ∧ x ≠ y ∧ (x = s ∨ x = t ∨ x = u ∨ x = v ∨ x = w ∨ x = x ∨ x = y) ∧ (y = s ∨ y = t ∨ y = u ∨ y = v ∨ y = w ∨ y = x ∨ y = y)) 
  → (∀ i, i ∈ {s, t, u, v, w, x, y} → i ≥ 21 → i = s ∨ i = t ∨ i = u ∨ i = v ∨ i = w ∨ i = x ∨ i = y) → {i | i ∈ {s, t, u, v, w, x, y} ∧ i ≥ 21}.card ≤ 4))
: sorry

end max_count_greater_than_20_l195_195085


namespace max_possible_value_of_e_n_l195_195785

noncomputable def b (n : ℕ) : ℚ := (8^n - 1) / 7

def e (n : ℕ) : ℕ := Int.gcd (b n).numerator (b (n + 1)).numerator

theorem max_possible_value_of_e_n : ∀ n : ℕ, e n = 1 :=
by sorry

end max_possible_value_of_e_n_l195_195785


namespace tetrahedron_from_triangle_is_logical_reasoning_l195_195368

def is_logical_reasoning (reasoning : String) : Prop := 
  reasoning = "inductive" ∨ reasoning = "analogical"

def analogical_reasoning : String := "analogical"

theorem tetrahedron_from_triangle_is_logical_reasoning :
  is_logical_reasoning analogical_reasoning := 
by
  left
  rfl

end tetrahedron_from_triangle_is_logical_reasoning_l195_195368


namespace delores_initial_money_l195_195202

def computer_price : ℕ := 400
def printer_price : ℕ := 40
def headphones_price : ℕ := 60
def discount_percentage : ℕ := 10
def left_money : ℕ := 10

theorem delores_initial_money :
  ∃ initial_money : ℕ,
    initial_money = printer_price + headphones_price + (computer_price - (discount_percentage * computer_price / 100)) + left_money :=
  sorry

end delores_initial_money_l195_195202


namespace find_rectangle_width_l195_195335

variable (length_square : ℕ) (length_rectangle : ℕ) (width_rectangle : ℕ)

-- Given conditions
def square_side_length := 700
def rectangle_length := 400
def square_perimeter := 4 * square_side_length
def rectangle_perimeter := square_perimeter / 2
def rectangle_perimeter_eq := 2 * length_rectangle + 2 * width_rectangle

-- Statement to prove
theorem find_rectangle_width :
  (square_perimeter = 2800) →
  (rectangle_perimeter = 1400) →
  (length_rectangle = 400) →
  (rectangle_perimeter_eq = 1400) →
  (width_rectangle = 300) :=
by
  intros
  sorry

end find_rectangle_width_l195_195335


namespace area_bisecting_lines_l195_195763

/-- Given a triangle ABC with incenter D, the distance from D to side AB is 
0.32 times the altitude from vertex C. The distance from D to side AC is 
0.36 times the altitude from vertex B. The problem is to determine all the 
lines through point D which divide the area of triangle ABC into two equal parts. -/
theorem area_bisecting_lines 
  (A B C D : Point) 
  (h1 : inside_triangle D A B C)
  (h2 : distance_to_line D A B = 0.32 * altitude_to_side C A B)
  (h3 : distance_to_line D A C = 0.36 * altitude_to_side B A C) :
  area_bisecting_lines D A B C = 
  { (9 / 16, 8 / 9), (1, 1 / 2) } :=
sorry

end area_bisecting_lines_l195_195763


namespace vertical_line_slope_does_not_exist_l195_195687

def Point := (ℝ × ℝ)

def line_through (A B : Point) : Prop :=
  ∃ m b, ∀ x y, (y = m * x + b) ↔ (x, y) ∈ {A, B}

def is_vertical (A B : Point) : Prop :=
  ∀ x₁ y₁ x₂ y₂, A = (x₁, y₁) → B = (x₂, y₂) → x₁ ≠ x₂ → false

theorem vertical_line_slope_does_not_exist :
  ∀ y : ℝ, ∃! l slope, (∃ A B : Point, A = (5, -3) ∧ B = (5, y) ∧ line_through A B ∧ is_vertical A B) →
  slope = none :=
by
  sorry

end vertical_line_slope_does_not_exist_l195_195687


namespace cos_shift_left_l195_195872

theorem cos_shift_left : 
  ∀ x : ℝ, 
    cos (2 * x + π / 3) = cos (2 * (x + π / 6)) :=
by
  sorry

end cos_shift_left_l195_195872


namespace enclosed_area_l195_195999

theorem enclosed_area (f g : ℝ → ℝ) (h1 : ∀ x, f x = 2 - x^2) (h2 : ∀ x, g x = x) :
  ∫ x in -2..1, (f x - g x) = 9 / 2 :=
by 
  -- Definitions of the functions
  have h1_def : ∀ x, f x = 2 - x^2 := h1,
  have h2_def : ∀ x, g x = x := h2,
  sorry

end enclosed_area_l195_195999


namespace least_product_of_primes_gt_30_l195_195500

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l195_195500


namespace smallest_of_a_b_c_l195_195337

def a : ℝ := (-2)^2
def b : ℝ := (-2)^0
def c : ℝ := (-2)^(-2)

theorem smallest_of_a_b_c : c < b ∧ b < a :=
by
  rw [a, b, c]
  simp
  have h1 : (-2)^2 = 4 := by norm_num
  have h2 : (-2)^0 = 1 := by norm_num
  have h3 : (-2)^(-2) = 1 / ((-2)^2) := by norm_num
  rw [h1, h2, h3]
  have h4 : 1 / 4 < 1 := by norm_num
  have h5 : 1 < 4 := by norm_num
  exact ⟨h4, h5⟩

end smallest_of_a_b_c_l195_195337


namespace card_probability_is_correct_l195_195095

noncomputable def total_cards := 52
noncomputable def probability_of_event : ℚ := 17 / 11050

def event_probability : ℚ :=
  let p_first_5_3_suits      := (3/total_cards) * (12/(total_cards-1)) * (4/(total_cards-2))
  let p_first_5_diamond   := (3/total_cards) * (1/(total_cards-1)) * (3/(total_cards-2))
  let p_first_5diamond_3suits  := (1/total_cards) * (12/(total_cards-1)) * (4/(total_cards-2))
  let p_first_5diamond_3diamond := (1/total_cards) * (1/(total_cards-1)) * (3/(total_cards-2))
  
  p_first_5_3_suits + p_first_5_diamond + p_first_5diamond_3suits + p_first_5diamond_3diamond

theorem card_probability_is_correct : event_probability = probability_of_event := by
  sorry

end card_probability_is_correct_l195_195095


namespace fair_haired_women_percentage_l195_195913

-- Let E be the total number of employees
variables (E : ℕ)

-- 32% of employees are women with fair hair
def women_with_fair_hair := 0.32 * E

-- 80% of employees have fair hair
def fair_haired_employees := 0.80 * E

-- Prove that the percentage of fair-haired employees who are women is 40%
theorem fair_haired_women_percentage : 
  (women_with_fair_hair E / fair_haired_employees E) * 100 = 40 :=
by 
  sorry

end fair_haired_women_percentage_l195_195913


namespace cost_of_double_burger_l195_195613

-- Definitions based on conditions
def total_cost : ℝ := 64.50
def total_burgers : ℕ := 50
def single_burger_cost : ℝ := 1.00
def double_burgers : ℕ := 29

-- Proof goal
theorem cost_of_double_burger : (total_cost - single_burger_cost * (total_burgers - double_burgers)) / double_burgers = 1.50 :=
by
  sorry

end cost_of_double_burger_l195_195613


namespace candle_height_after_half_time_l195_195919

-- Define the parameters
def initial_height : ℕ := 100
def first_cm_time : ℕ := 15
def time_increment : ℕ := 15
def total_cm : ℕ := 100

-- Arithmetic sequence sum function
def arithmetic_sum (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- The total time to burn the candle completely
def total_time : ℕ :=
  arithmetic_sum total_cm first_cm_time time_increment

-- Half of the total burning time
def half_total_time : ℕ := total_time / 2

-- Find the height of the candle after half the burning time
theorem candle_height_after_half_time :
  initial_height - 70 = 30 := by
  sorry

end candle_height_after_half_time_l195_195919


namespace sam_walked_distance_when_meeting_l195_195657

variable (D_s D_f : ℝ)
variable (t : ℝ)

theorem sam_walked_distance_when_meeting
  (h1 : 55 = D_f + D_s)
  (h2 : D_f = 6 * t)
  (h3 : D_s = 5 * t) :
  D_s = 25 :=
by 
  -- This is where the proof would go
  sorry

end sam_walked_distance_when_meeting_l195_195657


namespace arithmetic_sum_l195_195612

theorem arithmetic_sum (a₁ an n : ℕ) (h₁ : a₁ = 5) (h₂ : an = 32) (h₃ : n = 10) :
  (n * (a₁ + an)) / 2 = 185 :=
by
  sorry

end arithmetic_sum_l195_195612


namespace line_intersects_y_axis_at_point_l195_195985

theorem line_intersects_y_axis_at_point :
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 0 → y1 = 8 → x2 = 6 → y2 = -4 → 
  ∃ y : ℝ, (x = 0 → y = y1) :=
by 
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  use y1
  split
  . exact hx1
  . exact hy1

end line_intersects_y_axis_at_point_l195_195985


namespace complex_abs_inequality_l195_195383

noncomputable def complex_abs (z : ℂ) : ℝ := complex.abs z

theorem complex_abs_inequality (a b c : ℂ)
    (h1 : ab + ac - bc ≠ 0)
    (h2 : ba + bc - ac ≠ 0)
    (h3 : ca + cb - ab ≠ 0) :
    complex_abs ( a * a / (a * b + a * c - b * c)) + complex_abs ( b * b / (b * a + b * c - a * c)) + complex_abs ( c * c / (c * a + c * b - a * b)) ≥ 3 / 2 :=
by
  sorry

end complex_abs_inequality_l195_195383


namespace sin_square_range_l195_195659

def range_sin_square_values (α β : ℝ) : Prop :=
  3 * (Real.sin α) ^ 2 - 2 * Real.sin α + 2 * (Real.sin β) ^ 2 = 0

theorem sin_square_range (α β : ℝ) (h : range_sin_square_values α β) :
  0 ≤ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ∧ 
  (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ≤ 4 / 9 :=
sorry

end sin_square_range_l195_195659


namespace negation_proof_l195_195075

def negation_of_p (a : ℝ) : Prop :=
  ¬ (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 3 → x^2 - a ≥ 0) ↔ ∃ x : ℝ, x ∈ set.Icc (-1 : ℝ) 3 ∧ x^2 - a < 0

theorem negation_proof (a : ℝ) : negation_of_p a :=
by sorry

end negation_proof_l195_195075


namespace rectangle_length_width_ratio_l195_195675

theorem rectangle_length_width_ratio (A d : ℝ) (hA : 0 < A) (hd : 0 < d) 
  (hxy : ∃ (x y : ℝ), x > y ∧ x * y = A ∧ x^2 + y^2 = d^2) :
  ∃ (r : ℝ), r = sqrt ((d^2 + sqrt (d^4 - 4 * A^2)) / (d^2 - sqrt (d^4 - 4 * A^2))) :=
by
  sorry

end rectangle_length_width_ratio_l195_195675


namespace quadratic_root_product_l195_195396

theorem quadratic_root_product (a b : ℝ) (m p r : ℝ)
  (h1 : a * b = 3)
  (h2 : ∀ x, x^2 - mx + 3 = 0 → x = a ∨ x = b)
  (h3 : ∀ x, x^2 - px + r = 0 → x = a + 2 / b ∨ x = b + 2 / a) :
  r = 25 / 3 := by
  sorry

end quadratic_root_product_l195_195396


namespace sugar_percentage_after_addition_l195_195915

-- Variables for the initial conditions
def initial_volume : ℝ := 340
def initial_water_percent : ℝ := 0.64
def initial_kola_percent : ℝ := 0.09
def initial_sugar_percent : ℝ := 1 - initial_water_percent - initial_kola_percent

-- Variables for addition
def added_sugar : ℝ := 3.2
def added_water : ℝ := 8
def added_kola : ℝ := 6.8

-- Calculate initial amounts
def initial_water : ℝ := initial_water_percent * initial_volume
def initial_kola : ℝ := initial_kola_percent * initial_volume
def initial_sugar : ℝ := initial_sugar_percent * initial_volume

-- Calculate new amounts
def new_sugar : ℝ := initial_sugar + added_sugar
def new_water : ℝ := initial_water + added_water
def new_kola : ℝ := initial_kola + added_kola

-- Calculate new total volume
def new_total_volume : ℝ := initial_volume + added_sugar + added_water + added_kola

-- Calculate the new percentage of sugar
def new_sugar_percent : ℝ := (new_sugar / new_total_volume) * 100

-- Proof statement
theorem sugar_percentage_after_addition : new_sugar_percent ≈ 26.54 :=
  by
    sorry

end sugar_percentage_after_addition_l195_195915


namespace part1_part2_l195_195002

-- Definitions for part 1
def prop_p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def prop_q (x : ℝ) : Prop := (x - 3) / (x + 2) < 0

-- Definitions for part 2
def neg_prop_q (x : ℝ) : Prop := ¬((x - 3) / (x + 2) < 0)
def neg_prop_p (a x : ℝ) : Prop := ¬(x^2 - 4*a*x + 3*a^2 < 0)

-- Proof problems
theorem part1 (a : ℝ) (x : ℝ) (h : a = 1) (hpq : prop_p a x ∧ prop_q x) : 1 < x ∧ x < 3 := 
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x, neg_prop_q x → neg_prop_p a x) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end part1_part2_l195_195002


namespace count_special_numbers_l195_195712
-- Import necessary library

-- Define the mathematical proof problem
theorem count_special_numbers (n : ℕ) (h : n ≥ 5) : 
  (number_of_n_digit_numbers_including_1_2_3_4_5_without_0_9 n) = 8^n - 5 * 7^n + 10 * 6^n - 10 * 5^n + 5 * 4^n - 3^n :=
by
  sorry

-- Mock definition to use in the theorem statement
-- This would need to be correctly defined to accurately compute the number of such special numbers
def number_of_n_digit_numbers_including_1_2_3_4_5_without_0_9 (n : ℕ) : ℕ :=
  sorry

end count_special_numbers_l195_195712


namespace stickers_on_last_page_l195_195945

def total_stickers (books pages stickers_per_page : ℕ) : ℕ :=
  books * pages * stickers_per_page

def stickers_in_full_pages (full_books pages_full new_stickers_per_page : ℕ) : ℕ :=
  full_books * pages_full * new_stickers_per_page

def stickers_in_partial_pages (pages_partial new_stickers_per_page : ℕ) : ℕ :=
  pages_partial * new_stickers_per_page

def remaining_stickers (total full_stickers : ℕ) : ℕ :=
  total - full_stickers

theorem stickers_on_last_page :
  let total = total_stickers 10 30 5 in
  let full_stickers = stickers_in_full_pages 6 30 8 in
  let rem_stickers = remaining_stickers total full_stickers in
  let full_seventh_book_stickers = stickers_in_partial_pages 7 8 in
  let last_page_stickers = remaining_stickers rem_stickers full_seventh_book_stickers in
  last_page_stickers = 4 :=
by
  let total := 10 * 30 * 5
  let full_stickers := 6 * 30 * 8
  let rem_stickers := 1500 - full_stickers
  let full_seventh_book_stickers := 7 * 8
  let last_page_stickers := rem_stickers - full_seventh_book_stickers
  have : total = 1500 := rfl
  have : full_stickers = 1440 := rfl
  have : rem_stickers = 60 := rfl
  have : full_seventh_book_stickers = 56 := rfl
  have : last_page_stickers = 4 := rfl
  exact rfl

end stickers_on_last_page_l195_195945


namespace mark_profit_l195_195028

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l195_195028


namespace total_marbles_l195_195101

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l195_195101


namespace probability_team_B_wins_third_game_l195_195064

theorem probability_team_B_wins_third_game :
  ∀ (A B : ℕ → Prop),
    (∀ n, A n ∨ B n) ∧ -- Each game is won by either A or B
    (∀ n, A n ↔ ¬ B n) ∧ -- No ties, outcomes are independent
    (A 0) ∧ -- Team A wins the first game
    (B 1) ∧ -- Team B wins the second game
    (∃ n1 n2 n3, A n1 ∧ A n2 ∧ A n3 ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3) -- Team A wins three games
    → (∃ S, ((A 0) ∧ (B 1) ∧ (B 2)) ↔ (S = 1/3)) := sorry

end probability_team_B_wins_third_game_l195_195064


namespace coloring_schemes_l195_195634

noncomputable def a : ℕ → ℤ
| 2 := 20
| 3 := 60
| (n + 4) := 3 * a (n + 3) + 4 * a (n + 2)

theorem coloring_schemes (n : ℕ) (hn : n ≥ 4) :
  a n = 4^n + 4 * (-1)^n := sorry

end coloring_schemes_l195_195634


namespace part_a_part_b_l195_195147

noncomputable def smallest_multiple_of_9_with_digits_0_and_1 : ℕ :=
  111111111

theorem part_a :
  ∀ n : ℕ, (∀d ∈ digits 10 n, d = 0 ∨ d = 1) → (∃ k, n = 9 * k) → n ≥ smallest_multiple_of_9_with_digits_0_and_1 :=
sorry

noncomputable def smallest_multiple_of_9_with_digits_1_and_2 : ℕ :=
  12222

theorem part_b :
  ∀ n : ℕ, (∀d ∈ digits 10 n, d = 1 ∨ d = 2) → (∃ k, n = 9 * k) → n ≥ smallest_multiple_of_9_with_digits_1_and_2 :=
sorry

end part_a_part_b_l195_195147


namespace ellipse_eccentricity_l195_195153

theorem ellipse_eccentricity (a b : ℝ) (h : a = 3 * b) :
  let c := Real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end ellipse_eccentricity_l195_195153


namespace union_M_N_intersection_M_complement_N_l195_195706

open Set

variable (U : Set ℝ) (M N : Set ℝ)

-- Define the universal set
def is_universal_set (U : Set ℝ) : Prop :=
  U = univ

-- Define the set M
def is_set_M (M : Set ℝ) : Prop :=
  M = {x | ∃ y, y = (x - 2).sqrt}  -- or equivalently x ≥ 2

-- Define the set N
def is_set_N (N : Set ℝ) : Prop :=
  N = {x | x < 1 ∨ x > 3}

-- Define the complement of N in U
def complement_set_N (U N : Set ℝ) : Set ℝ :=
  U \ N

-- Prove M ∪ N = {x | x < 1 ∨ x ≥ 2}
theorem union_M_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∪ N = {x | x < 1 ∨ x ≥ 2} :=
  sorry

-- Prove M ∩ (complement of N in U) = {x | 2 ≤ x ≤ 3}
theorem intersection_M_complement_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∩ (complement_set_N U N) = {x | 2 ≤ x ∧ x ≤ 3} :=
  sorry

end union_M_N_intersection_M_complement_N_l195_195706


namespace total_spent_snacks_l195_195053

-- Define the costs and discounts
def cost_pizza : ℕ := 10
def boxes_robert_orders : ℕ := 5
def pizza_discount : ℝ := 0.15
def cost_soft_drink : ℝ := 1.50
def soft_drinks_robert : ℕ := 10
def cost_hamburger : ℕ := 3
def hamburgers_teddy_orders : ℕ := 6
def hamburger_discount : ℝ := 0.10
def soft_drinks_teddy : ℕ := 10

-- Calculate total costs
def total_cost_robert : ℝ := 
  let cost_pizza_total := (boxes_robert_orders * cost_pizza) * (1 - pizza_discount)
  let cost_soft_drinks_total := soft_drinks_robert * cost_soft_drink
  cost_pizza_total + cost_soft_drinks_total

def total_cost_teddy : ℝ :=
  let cost_hamburger_total := (hamburgers_teddy_orders * cost_hamburger) * (1 - hamburger_discount)
  let cost_soft_drinks_total := soft_drinks_teddy * cost_soft_drink
  cost_hamburger_total + cost_soft_drinks_total

-- The final theorem to prove the total spending
theorem total_spent_snacks : 
  total_cost_robert + total_cost_teddy = 88.70 := by
  sorry

end total_spent_snacks_l195_195053


namespace linear_function_not_passing_second_quadrant_l195_195891

noncomputable def probability_not_passing_second_quadrant : ℚ :=
  let ks := {-2, -1, 1, 2, 3}
  let bs := {-2, -1, 1, 2, 3}
  let total_pairs := 5 * 4 -- Total pairs (5 choices for k, 4 remaining choices for b)
  let favorable_pairs := 3 * 2 -- Positive k (3 choices) and Negative b (2 choices)
  favorable_pairs / total_pairs

theorem linear_function_not_passing_second_quadrant :
  probability_not_passing_second_quadrant = 3 / 10 :=
sorry

end linear_function_not_passing_second_quadrant_l195_195891


namespace least_product_of_distinct_primes_greater_than_30_l195_195508

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l195_195508


namespace appropriate_colorings_count_l195_195671

-- Conditions
variables (n : ℕ) (S : Finset (ℝ × ℝ)) 

-- Assertions based on the conditions
def points_conditions (n : ℕ) (S : Finset (ℝ × ℝ)) : Prop :=
  n ≥ 3 ∧ S.card = n ∧ ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ S → p2 ∈ S → p3 ∈ S → 
    (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) → ¬ (collinear ℝ {p1, p2, p3}) ∧
    ∀ (p1 p2 p3 p4 : ℝ × ℝ), p1 ∈ S → p2 ∈ S → p3 ∈ S → p4 ∈ S → 
        (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4) → 
        ¬ (concyclic ℝ {p1, p2, p3, p4})

-- Theorem statement in Lean
theorem appropriate_colorings_count : 
  points_conditions n S → 
  (∑ k in Finset.range (4), Nat.choose n k) = (Nat.choose n 3 + Nat.choose n 2 + Nat.choose n 1 + Nat.choose n 0) :=
by
  sorry

end appropriate_colorings_count_l195_195671


namespace vector_equality_sufficient_not_necessary_l195_195724

theorem vector_equality_sufficient_not_necessary (a b : ℝ^2) :
  (a = b → |a| = |b|) ∧ (¬(a = b) ∧ |a| = |b|) :=
by
  sorry

end vector_equality_sufficient_not_necessary_l195_195724


namespace total_number_of_marbles_is_1050_l195_195105

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l195_195105


namespace least_product_of_distinct_primes_greater_than_30_l195_195509

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l195_195509


namespace contrapositive_eq_inverse_l195_195079

variable (p q : Prop)

theorem contrapositive_eq_inverse (h1 : p → q) :
  (¬ p → ¬ q) ↔ (q → p) := by
  sorry

end contrapositive_eq_inverse_l195_195079


namespace functional_equation_solution_l195_195203

-- Define the conditions and theorem
theorem functional_equation_solution {f : ℝ → ℝ} (hf : continuous_on f (set.Ioo (-1:ℝ) (1:ℝ)))
  (h_eq : ∀ x y, x ∈ set.Ioo (-1:ℝ) 1 → y ∈ set.Ioo (-1:ℝ) 1 → x + y ∈ set.Ioo (-1:ℝ) 1 →
            f (x + y) = (f x + f y) / (1 - f x * f y)) :
  ∃ a, |a| ≤ π / 2 ∧ ∀ x, x ∈ set.Ioo (-1:ℝ) 1 → f x = Real.tan (a * x) := 
sorry

end functional_equation_solution_l195_195203


namespace isosceles_triangle_sides_l195_195174

variables {x y : ℝ}

theorem isosceles_triangle_sides (x y : ℝ) (h1 : x + y + y = 20) (h2 : y = 2 * x) 
  (ineq1 : x + y > y) (ineq2 : x + y > x) (ineq3 : y + y > x) :
  (x = 4) ∧ (y = 8) :=
by
  have h : 5 * x = 20 :=
    calc
    x + y + y = 20 : h1
    _ = x + 2 * x + 2 * x : by rw h2
    _ = 5 * x           : by ring
  have seq : x = 4 := by linarith
  have y_val : y = 8 := by { rw [h2, seq], ring }
  exact ⟨seq, y_val⟩

end isosceles_triangle_sides_l195_195174


namespace calculate_value_l195_195193

def a : ℤ := 3 * 4 * 5
def b : ℚ := 1/3 + 1/4 + 1/5

theorem calculate_value :
  (a : ℚ) * b = 47 := by
sorry

end calculate_value_l195_195193


namespace line_b_y_intercept_l195_195010

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l195_195010


namespace conditional_probability_of_plant_growth_l195_195302

namespace PlantGrowth

def P (event : Type) := ℝ

variable (A B : Type)
variable [P_A : P A = 0.4] [P_B : P B = 0.8]

-- The conditional probability P(A | B)
def P_cond (A B : Type) [P_A : P A] [P_B : P B] [P_A_and_B : P (A ∩ B) = P A] : P (A | B) :=
  (P A) / (P B)

theorem conditional_probability_of_plant_growth :
  P_cond A B = 0.5 :=
by
  sorry

end PlantGrowth

end conditional_probability_of_plant_growth_l195_195302


namespace num_perfect_square_factors_of_360_l195_195332

theorem num_perfect_square_factors_of_360 : 
  (∃ f : ℕ, prime_factors f = {2, 3, 5} ∧ positive_factors f = 360 ∧ perfect_square f 4) :=
  sorry

end num_perfect_square_factors_of_360_l195_195332


namespace average_price_of_pencil_l195_195555

-- Define the given conditions
variables (num_pens num_pencils total_cost pens_avg_price : ℕ)
variables (pens_cost pencils_cost : ℕ)

-- Define the average price of a pencil which we want to prove
def pencils_avg_price := 2

-- State the mathematically equivalent proof problem
theorem average_price_of_pencil : 
  30 * 20 + (75 * pencils_avg_price) = 750 :=
by 
  have pens_cost : 30 * 20 = 600 := by sorry
  have pencils_cost : 75 * pencils_avg_price = 150 := by sorry
  calc
    30 * 20 + 75 * pencils_avg_price = pens_cost + pencils_cost := by sorry
    ... = 600 + 150 := by sorry
    ... = 750 := by sorry

end average_price_of_pencil_l195_195555


namespace linear_function_no_second_quadrant_l195_195730

theorem linear_function_no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (y : ℝ) → y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)) ↔ k ≥ 3 :=
sorry

end linear_function_no_second_quadrant_l195_195730


namespace perimeter_of_A₀OC₀_eq_AC_l195_195790

open EuclideanGeometry

variables {A B C H A₀ C₀ O : Point}

-- Define conditions
def is_acute_angled_triangle (A B C : Point) : Prop := sorry

def is_orthocenter (H A B C : Point) : Prop := sorry

def circumcenter (O A B C : Point) : Prop := sorry

def perpendicular_bisector_intersect (B H A B A₀ : Point) (B H C B C₀ : Point) : Prop := sorry

def perimeter_of_triangle_eq (A₀ O C₀ A C : Point) : Prop :=
  triangle.perimeter A₀ O C₀ = segment_length A C

-- Lean statement
theorem perimeter_of_A₀OC₀_eq_AC
  {A B C H A₀ C₀ O : Point} :
  is_acute_angled_triangle A B C →
  is_orthocenter H A B C →
  circumcenter O A B C →
  perpendicular_bisector_intersect B H A B A₀ B H C B C₀ →
  perimeter_of_triangle_eq A₀ O C₀ A C :=
begin
  sorry
end

end perimeter_of_A₀OC₀_eq_AC_l195_195790


namespace trapezoid_tangential_problem_l195_195777

theorem trapezoid_tangential_problem
    (A B C D N K L : Type*) -- declaring points as types
    [trapezoid : IsTrapezoid A B C D]
    [tangential : IsTangential A B C D]
    (h_parallel : Parallel AD BC)
    (h_eq : |AB| = |CD|)
    (incircle_touch : IncircleTouchCD N)
    (meet_again_AK : MeetAgain AN K)
    (meet_again_BL : MeetAgain BN L) :
    ∃ (x y : ℝ), |AN| / |AK| + |BN| / |BL| = 10 :=
begin
  sorry
end

end trapezoid_tangential_problem_l195_195777


namespace area_of_triangle_l195_195736

-- Definitions based on the given conditions
variables (A : ℝ) (b c : ℝ)
def cos2A := Real.cos (2 * A)
def sinA := Real.sin A
def area_triangle (a b c : ℝ) (A : ℝ) := (1 / 2) * b * c * Real.sin A

-- Conditions
variable (h1 : cos2A = sinA)
variable (h2 : b * c = 2)

-- The proof goal
theorem area_of_triangle (h1 : cos2A = sinA) (h2 : b * c = 2) : area_triangle a b c A = 1 / 2 := by
  sorry

end area_of_triangle_l195_195736


namespace find_z_find_theta_l195_195289

open Complex Real

noncomputable def z1 (θ : ℝ) : ℂ := (sin θ)^2 + I
noncomputable def z2 (θ : ℝ) : ℂ := - (cos θ)^2 + I * (cos (2 * θ))

def z (θ : ℝ) : ℂ := z2 θ - z1 θ

-- Part (1) of the problem
theorem find_z (θ : ℝ) (h_θ : 0 < θ ∧ θ < π) : 
  z θ = -1 + (cos (2 * θ) - 1) * I :=
sorry

-- Part (2) of the problem
theorem find_theta (θ : ℝ) (h_θ : 0 < θ ∧ θ < π) (h_line : (cos (2 * θ) - 1) = -(1 / 2)) : 
  θ = π / 6 ∨ θ = 5 * π / 6 :=
sorry

end find_z_find_theta_l195_195289


namespace distance_focus_directrix_l195_195071

theorem distance_focus_directrix (y x p : ℝ) (h : y^2 = 4 * x) (hp : 2 * p = 4) : p = 2 :=
by sorry

end distance_focus_directrix_l195_195071


namespace intersection_S_T_l195_195798

def S : set ℝ := { x | (x-2)*(x-3) ≥ 0 }
def T : set ℝ := { x | x > 0 }

theorem intersection_S_T : S ∩ T = { x | (x > 0 ∧ x ≤ 2) ∨ x ≥ 3 } :=
by
  sorry

end intersection_S_T_l195_195798


namespace complex_conjugate_identity_l195_195722

theorem complex_conjugate_identity (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
by sorry

end complex_conjugate_identity_l195_195722


namespace problem_1_problem_2_l195_195408

-- Problem 1
theorem problem_1 (x : ℝ) : (|x-1| + |2x-1| > 3 - 4 * x) → (x > 3 / 5) :=
sorry

-- Problem 2
theorem problem_2 (m : ℝ) : 
  (∀ x : ℝ, (|x-1| + |2x-1| + |1-x| ≥ 6 * m^2 - 5 * m)) → 
  (-1 / 6 ≤ m ∧ m ≤ 1) :=
sorry

end problem_1_problem_2_l195_195408


namespace symmetric_line_eq_l195_195448

noncomputable def point_symmetric (p : ℝ × ℝ) (q : ℝ × ℝ) : ℝ × ℝ :=
(2 * q.1 - p.1, 2 * q.2 - p.2)

def line (a b c : ℝ) : set (ℝ × ℝ) :=
{p | a * p.1 + b * p.2 + c = 0}

theorem symmetric_line_eq :
  let l1 := line 2 3 (-6),
      p := (1 : ℝ, -1 : ℝ),
      l2 := line 2 3 8 in
  symmetric_line l1 p = l2 := sorry

end symmetric_line_eq_l195_195448


namespace circumcircle_ABK_passes_through_orthocenter_AXY_l195_195357

-- Define the points and geometric configurations
variables (A B C D K P X Y : Type)
variables (circumcircle : ∀ {α : Type}, (α → α → α → α → Prop))
variables [has_eucl_ops α] [has_circularity α]
variables (h_sq : square A B C D)
variables (h_K : on_extension DA K)
variables (h_BK_AB : dist B K = dist A B)
variables (h_P_on_AB : on_line_segment A B P)
variables (h_bisect_PC : perpendicular_bisector C P X Y)

theorem circumcircle_ABK_passes_through_orthocenter_AXY :
  passes_through (circumcircle A B K) (orthocenter A X Y) :=
sorry

end circumcircle_ABK_passes_through_orthocenter_AXY_l195_195357


namespace ticket_price_divisor_l195_195572

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

def GCD (a b : ℕ) := Nat.gcd a b

theorem ticket_price_divisor :
  let total7 := 70
  let total8 := 98
  let y := 4
  is_divisor (GCD total7 total8) y :=
by
  sorry

end ticket_price_divisor_l195_195572


namespace magnitude_of_power_l195_195252

-- Given conditions
def z : ℂ := 3 + 2 * Complex.I
def n : ℕ := 6

-- Mathematical statement to prove
theorem magnitude_of_power :
  Complex.abs (z ^ n) = 2197 :=
by
  sorry

end magnitude_of_power_l195_195252


namespace sum_of_elements_l195_195393

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℕ}

-- Condition 1: The elements of A are non-negative integers (implicit from the type of A)
-- Condition 2: For any a_{ij} = 0, the sum of the i-th row and the j-th column is not less than n
def condition (A : Matrix (Fin n) (Fin n) ℕ) (i j : Fin n) : Prop :=
  if A i j = 0 then (∑ k, A i k) + (∑ k, A k j) ≥ n else True

theorem sum_of_elements (h : ∀ i j, condition A i j) : (∑ i j, A i j) ≥ n * n / 2 :=
sorry

end sum_of_elements_l195_195393


namespace median_eq_range_le_l195_195286

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end median_eq_range_le_l195_195286


namespace kwi_wins_race_l195_195513

-- Define the conditions of the problem
def race_distance_meters : ℕ := 20
def race_distance_decimeters : ℕ := race_distance_meters * 10
def kwa_jump_length : ℕ := 6
def kwi_jump_length : ℕ := 4
def kwa_jump_time : ℕ := 3
def kwi_jump_time : ℕ := 2

-- Prove that under these conditions, Kvi wins the race
theorem kwi_wins_race : 
  (let kwa_total_jumps := (2 * race_distance_decimeters) / kwa_jump_length,
       kwi_total_jumps := (2 * race_distance_decimeters) / kwi_jump_length in
    (kwa_total_jumps * kwa_jump_time > kwi_total_jumps * kwi_jump_time)) :=
by {
  sorry
}

end kwi_wins_race_l195_195513


namespace evaluate_expression_l195_195241

variables {x y z : ℝ}

def P : ℝ := x + y + z
def Q : ℝ := x - y - z

theorem evaluate_expression :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2 * y * z - z^2) / (x * y + x * z) := sorry

end evaluate_expression_l195_195241


namespace min_additional_matchsticks_needed_l195_195772

-- Define the number of matchsticks in a 3x7 grid
def matchsticks_in_3x7_grid : Nat := 4 * 7 + 3 * 8

-- Define the number of matchsticks in a 5x5 grid
def matchsticks_in_5x5_grid : Nat := 6 * 5 + 6 * 5

-- Define the minimum number of additional matchsticks required
def additional_matchsticks (matchsticks_in_3x7_grid matchsticks_in_5x5_grid : Nat) : Nat :=
  matchsticks_in_5x5_grid - matchsticks_in_3x7_grid

theorem min_additional_matchsticks_needed :
  additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid = 8 :=
by 
  unfold additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid
  sorry

end min_additional_matchsticks_needed_l195_195772


namespace trey_task_time_l195_195414

theorem trey_task_time :
  let clean_house_tasks := 7
      shower_tasks := 1
      dinner_tasks := 4
      total_time_minutes := 2 * 60
      total_tasks := clean_house_tasks + shower_tasks + dinner_tasks
  in total_time_minutes / total_tasks = 10 := by
  -- sorry is used to indicate missing proof as per instructions
  sorry

end trey_task_time_l195_195414


namespace mean_equivalence_l195_195851

theorem mean_equivalence {x : ℚ} :
  (8 + 15 + 21) / 3 = (18 + x) / 2 → x = 34 / 3 :=
by
  sorry

end mean_equivalence_l195_195851


namespace smallest_yellow_marbles_l195_195802

-- Definitions for given conditions
def total_marbles (n : ℕ): Prop := n > 0
def blue_marbles (n : ℕ) : ℕ := n / 4
def red_marbles (n : ℕ) : ℕ := n / 6
def green_marbles : ℕ := 7
def yellow_marbles (n : ℕ) : ℕ := n - (blue_marbles n + red_marbles n + green_marbles)

-- Lean statement that verifies the smallest number of yellow marbles is 0
theorem smallest_yellow_marbles (n : ℕ) (h : total_marbles n) : yellow_marbles n = 0 :=
  sorry

end smallest_yellow_marbles_l195_195802


namespace probability_real_complex_product_l195_195883

theorem probability_real_complex_product :
  let outcomes : Finset (ℕ × ℕ) := { (m, n) | m ∈ Finset.range 1 7 ∧ n ∈ Finset.range 1 7 }.to_finset in
  let real_condition : (ℕ × ℕ) → Prop := λ (p : ℕ × ℕ), let (m, n) := p in (m^2 = n^2) in
  (outcomes.filter real_condition).card / outcomes.card = 1 / 6 :=
by
  sorry

end probability_real_complex_product_l195_195883


namespace foreign_objects_total_sum_l195_195187

-- define the conditions
def dog_burrs : Nat := 12
def dog_ticks := 6 * dog_burrs
def dog_fleas := 3 * dog_ticks

def cat_burrs := 2 * dog_burrs
def cat_ticks := dog_ticks / 3
def cat_fleas := 4 * cat_ticks

-- calculate the total foreign objects
def total_dog := dog_burrs + dog_ticks + dog_fleas
def total_cat := cat_burrs + cat_ticks + cat_fleas

def total_objects := total_dog + total_cat

-- state the theorem
theorem foreign_objects_total_sum : total_objects = 444 := by
  sorry

end foreign_objects_total_sum_l195_195187


namespace profit_share_of_B_l195_195944

variable {capital_A capital_B capital_C : ℚ}
variable {total_profit : ℚ}

-- Definitions based on given conditions
def investment_ratio_A := capital_A / 2000
def investment_ratio_B := capital_B / 2000
def investment_ratio_C := capital_C / 2000

def profit_share_A (P : ℚ) := (investment_ratio_A / (investment_ratio_A + investment_ratio_B + investment_ratio_C)) * P
def profit_share_B (P : ℚ) := (investment_ratio_B / (investment_ratio_A + investment_ratio_B + investment_ratio_C)) * P
def profit_share_C (P : ℚ) := (investment_ratio_C / (investment_ratio_A + investment_ratio_B + investment_ratio_C)) * P

-- Given condition of profit shares difference
axiom profit_difference_condition : profit_share_C total_profit - profit_share_A total_profit = 500

-- Definition of capitals
def capital_A := 6000
def capital_B := 8000
def capital_C := 10000

-- Theorem statement to prove
theorem profit_share_of_B : profit_share_B total_profit = 1000 := by
  sorry

end profit_share_of_B_l195_195944


namespace count_valid_committees_l195_195957

-- Definitions for the conditions of the problem
def departments : List String := ["Mathematics", "Statistics", "Computer Science"]

def professors : String → List (String × String)
| "Mathematics" => [("M1", "Male"), ("M2", "Male"), ("M3", "Male"), ("F1", "Female"), ("F2", "Female"), ("F3", "Female")]
| "Statistics" => [("M1", "Male"), ("M2", "Male"), ("M3", "Male"), ("F1", "Female"), ("F2", "Female"), ("F3", "Female")]
| "Computer Science" => [("M1", "Male"), ("M2", "Male"), ("M3", "Male"), ("F1", "Female"), ("F2", "Female"), ("F3", "Female")]
| _ => []

def valid_committee (committee : List (String × String × String)) : Prop :=
  committee.length = 6 ∧
  committee.count_map (λ ⟨_, gender, _⟩ => gender) = [("Male", 3), ("Female", 3)] ∧
  committee.count_map (λ ⟨_, _, department⟩ => department) = [("Mathematics", 1), ("Statistics", 1), ("Computer Science", 1)] ∧
  ∀ dept, committee.count_map (λ ⟨_, _, department⟩ => department.declaration) dept ≤ 2

-- Main theorem statement
theorem count_valid_committees (S : List (String × String × String)) :
    (∃ (committee : List (String × String × String)), valid_committee committee ∧ list.length (filter valid_committee S) = 1215) := 
sorry

end count_valid_committees_l195_195957


namespace parabola_vertex_sum_l195_195442

theorem parabola_vertex_sum (p q r : ℝ) (h1 : ∀ x : ℝ, x = p * (x - 3)^2 + 2 → y) (h2 : p * (1 - 3)^2 + 2 = 6) :
  p + q + r = 6 :=
sorry

end parabola_vertex_sum_l195_195442


namespace birthday_cakes_verified_l195_195416

structure Person := 
  (candles : ℕ)
  (cakeSurfaceArea : ℝ)

def peter : Person := 
  { candles := 10, cakeSurfaceArea := 8 * 8 }

def rupert : Person := 
  { candles := 10 * 3.5, cakeSurfaceArea := 16 * 10 }

def mary : Person := 
  { candles := (10 * 3.5) - 5, cakeSurfaceArea := Real.pi * (6 * 6) }

def totalCandles : ℕ := 
  peter.candles + rupert.candles + mary.candles

def verifiedCandles : Prop :=
  totalCandles = 75

def verifiedSurfaceArea : Prop :=
  peter.cakeSurfaceArea = 64 ∧ 
  rupert.cakeSurfaceArea = 160 ∧
  mary.cakeSurfaceArea ≈ 113.097

theorem birthday_cakes_verified : verifiedCandles ∧ verifiedSurfaceArea :=
  by sorry

end birthday_cakes_verified_l195_195416


namespace repeating_decimal_fraction_product_l195_195892

theorem repeating_decimal_fraction_product :
  let x := (0.0012 : ℝ) in
  let num_denom_product := 13332 in
  (real.to_rat x).num * (real.to_rat x).denom = num_denom_product :=
by
  -- Mathematic proof steps go here
  sorry

end repeating_decimal_fraction_product_l195_195892


namespace monotonic_intervals_range_of_values_l195_195699

-- Part (1): Monotonic intervals of the function
theorem monotonic_intervals (a : ℝ) (h_a : a = 0) :
  (∀ x, 0 < x ∧ x < 1 → (1 + Real.log x) / x > 0) ∧ (∀ x, 1 < x → (1 + Real.log x) / x < 0) :=
by
  sorry

-- Part (2): Range of values for \(a\)
theorem range_of_values (a : ℝ) (h_f : ∀ x, 0 < x → (1 + Real.log x) / x - a ≤ 0) : 
  1 ≤ a :=
by
  sorry

end monotonic_intervals_range_of_values_l195_195699


namespace part1_parity_part2_range_l195_195701

-- Definitions
def f (a : ℝ) (x : ℝ) : ℝ := log a ((2 - x) / (2 + x))

-- Conditions
variables (a : ℝ) (h : a > 0) (h_ne : a ≠ 1)

-- Part 1: Prove parity of f
theorem part1_parity : ∀ x : ℝ, x ∈ Set.Ioo (-2 : ℝ) 2 → f a (-x) = -f a x := by
  sorry

-- Part 2: Prove the range of real number m
theorem part2_range (x m : ℝ) : (f a x = log a (x - m)) → m ∈ Set.Iio 2 := by
  sorry

end part1_parity_part2_range_l195_195701


namespace tileable_contains_domino_l195_195047

theorem tileable_contains_domino {m n a b : ℕ} (h_m : m ≥ a) (h_n : n ≥ b) :
  (∀ (x : ℕ) (y : ℕ), x + a ≤ m → y + b ≤ n → ∃ (p : ℕ) (q : ℕ), p = x ∧ q = y) :=
sorry

end tileable_contains_domino_l195_195047


namespace count_integers_200_300_l195_195326

theorem count_integers_200_300 :
  let is_valid (n : ℕ) := n ≥ 200 ∧ n < 300 ∧
                         let d1 := n / 100 in
                         let d2 := (n / 10) % 10 in
                         let d3 := n % 10 in
                         d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
                         d1 < d2 ∧ d2 < d3 in
  (finset.range 100).filter (λ n, is_valid (n + 200)).card = 18 :=
by
  sorry

end count_integers_200_300_l195_195326


namespace balloon_arrangements_correct_l195_195216

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l195_195216


namespace median_equality_of_sorted_quartet_range_subsets_l195_195279

variable {α : Type*} [LinearOrder α] [Add α] [Div α]
variable (x1 x2 x3 x4 x5 x6 : α)

theorem median_equality_of_sorted_quartet :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x3 + x4) / 2 = (x3 + x4) / 2 :=
sorry

theorem range_subsets :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_of_sorted_quartet_range_subsets_l195_195279


namespace sum_of_squares_of_digits_1000th_non_perfect_power_l195_195165

-- Define a perfect power
def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b ≥ 2 ∧ n = a ^ b

-- Define the sequence of non-perfect power integers
def non_perfect_powers : ℕ → ℕ
| 0        := 1
| (n + 1)  := let m := non_perfect_powers n + 1 in
              if is_perfect_power m then non_perfect_powers (n + 1) + 1 else m

-- Define a function to get the sum of the squares of the digits of a number
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d, d ^ 2) |>.sum

-- Define the problem statement that needs to be proved
theorem sum_of_squares_of_digits_1000th_non_perfect_power : 
  sum_of_squares_of_digits (non_perfect_powers 1000) = 21 :=
sorry

end sum_of_squares_of_digits_1000th_non_perfect_power_l195_195165


namespace number_of_storks_joined_l195_195912

variable (B : ℕ) (S : ℕ)

theorem number_of_storks_joined (hB : B = 3) (h : S = B + 2 + 1) : S = 6 :=
by
  rw [hB] at h
  calc
    S = 3 + 2 + 1 : by rw [h]
    _ = 6 : by norm_num

end number_of_storks_joined_l195_195912


namespace base7_to_base9_conversion_l195_195146

theorem base7_to_base9_conversion :
  (536 : ℕ)₇ = (332 : ℕ)₉ :=
by
  /-
  Outline:
  1. Convert 536 in base 7 to a decimal (base 10) number.
  2. Convert the decimal number to base 9 and prove it equals 332 in base 9.
  -/
  sorry

end base7_to_base9_conversion_l195_195146


namespace value_of_expression_l195_195005

noncomputable def integer_part (x : ℝ) : ℕ := ⌊x⌋
noncomputable def decimal_part (x : ℝ) : ℝ := x - ↑(⌊x⌋)

theorem value_of_expression :
  let a := integer_part (6 - Real.sqrt 10)
  let b := decimal_part (6 - Real.sqrt 10)
  (2 * a + Real.sqrt 10) * b = 6 :=
by
  let a := integer_part (6 - Real.sqrt 10)
  let b := decimal_part (6 - Real.sqrt 10)
  sorry

end value_of_expression_l195_195005


namespace motorcyclist_initial_speed_l195_195574

variable (initial_speed distance : ℕ)
variable (t1_hours stop_time_hours remaining_distance remaining_time_hours total_time_return duration_to_B)

def time_to_A (x : ℕ) := 120 / x
def time_return (x : ℕ) := 1 + (1 / 6) + (120 - x) / (x + 6)

theorem motorcyclist_initial_speed
  (H_dist : distance = 120)
  (H_stop : stop_time_hours = 1 / 6)
  (H_same_time : total_time_return = duration_to_B)
  (H_duration_to_B : duration_to_B = 120 / initial_speed)
  : initial_speed = 48 := 
begin
  sorry,
end

end motorcyclist_initial_speed_l195_195574


namespace f_increasing_interval_l195_195853

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 2 * sin x * cos x + 3 * cos x ^ 2

theorem f_increasing_interval :
  ∀ x ∈ Icc 0 (π / 8), 
  ∀ y ∈ Icc 0 (π / 8), x ≤ y → f x ≤ f y :=
sorry

end f_increasing_interval_l195_195853


namespace option_a_incorrect_l195_195130

theorem option_a_incorrect (a b : Int) : a = -3 → b = 10 → a + b ≠ abs (b - a) := 
by
  intros ha hb
  rw [ha, hb]
  simp
  sorry

end option_a_incorrect_l195_195130


namespace prove_true_prop_l195_195291

def p : Prop := ∀ (x : ℝ), x^2 ≥ 0
def q : Prop := ∀ (y : ℝ), y > 0 → log y < 0

theorem prove_true_prop
  (hp : p)
  (hq : ¬q) :
  (¬p) ∨ (¬q) :=
by {
  sorry
}

end prove_true_prop_l195_195291


namespace part1_min_value_a_1_part2_extreme_points_l195_195314

-- Define the function f for the given problem
def f (a x : ℝ) : ℝ := (x + 2) / Real.exp x + a * x - 2

-- Problem 1: Prove that for a = 1, the minimum value of f(x) on the interval [0, ∞) is 0
theorem part1_min_value_a_1 : ∀ x ≥ 0, f 1 x ≥ 0 :=
sorry

-- Problem 2: If f has two extreme points x1 and x2 on ℝ, prove that e^x2 - e^x1 > 2/a - 2
theorem part2_extreme_points (a : ℝ) (ha : 0 < a ∧ a < 1) (x1 x2 : ℝ) (hx1_lt : -1 < x1 ∧ x1 < 0) (hx2_gt : 0 < x2)
  (hx1_x2 : x1 < x2) (h_fx1_fx2_zero : (∃ x1 x2 : ℝ, f a x1 = 0 ∧ f a x2 = 0)) :
  Real.exp x2 - Real.exp x1 > 2 / a - 2 :=
sorry

end part1_min_value_a_1_part2_extreme_points_l195_195314


namespace least_product_of_distinct_primes_gt_30_l195_195490

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l195_195490


namespace loci_of_common_tangents_l195_195040

noncomputable def circle (O : Type) (R : ℝ) :=
  {p : point | dist O p = R}

theorem loci_of_common_tangents
  (O O' : point)
  (R r : ℝ)
  (s := circle O R)
  (s' := circle O' r)
  (l : line)
  (h1 : O ∈ l)
  (h2 : O' ∈ l)
  (P Q : point)
  (h3 : P ∈ s)
  (h4 : Q ∈ s)
  (h5 : P ≠ Q) :
  ∀ (M : point),
  (∃ t : tangent, tangent_to t M s) →
  (∃ t' : tangent, tangent_to t' M s') →
  (M ∈ (tangent_to s P ∪ tangent_to s Q) - {P, Q}) :=
sorry

end loci_of_common_tangents_l195_195040


namespace Cary_final_salary_l195_195967

def initial_salary : ℝ := 10
def raise_percentage : ℝ := 0.20
def cut_percentage : ℝ := 0.75

theorem Cary_final_salary :
  let raise_amount := raise_percentage * initial_salary in
  let new_salary_after_raise := initial_salary + raise_amount in
  let final_salary := cut_percentage * new_salary_after_raise in
  final_salary = 9 := by
  sorry

end Cary_final_salary_l195_195967


namespace number_of_rectangles_in_5x5_grid_l195_195609

theorem number_of_rectangles_in_5x5_grid : 
  let n : ℕ := 5
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l195_195609


namespace find_z_l195_195719

theorem find_z (z : ℂ) (hz : conj z * (1 + complex.I) = 1 - complex.I) : 
  z = complex.I :=
by 
  sorry

end find_z_l195_195719


namespace arc_sum_eq_l195_195566

-- Definitions based on conditions
variable (ABCD : CyclicQuadrilateral)
variable (K1 K2 L1 L2 M1 M2 N1 N2 : Point)
variable (omega : Circle)
variable (a b c d : ℝ)
variable (h_on_omega : (K1 = ω ∧ K2 = ω ∧ L1 = ω ∧ L2 = ω ∧ M1 = ω ∧ M2 = ω ∧ N1 = ω ∧ N2 = ω))
variable (arc_lengths : (a = length(arc ω N2 K1) ∧ b = length(arc ω K2 L1) ∧ c = length(arc ω L2 M1) ∧ d = length(arc ω M2 N1)))
variable (not_containing : (not_containing(arc ω a K2) ∧ not_containing(arc ω b L2) ∧ not_containing(arc ω c M2) ∧ not_containing(arc ω d N2)))

-- Theorem statement
theorem arc_sum_eq (a b c d : ℝ) (h : AB = ω) (h_arc : arc_lengths a b c d) (h_not_containing : not_containing a b c d) : a + c = b + d :=
sorry

end arc_sum_eq_l195_195566


namespace balloon_permutations_l195_195210

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l195_195210


namespace ratio_of_A_to_B_l195_195615

theorem ratio_of_A_to_B (total_weight compound_A_weight compound_B_weight : ℝ)
  (h1 : total_weight = 108)
  (h2 : compound_B_weight = 90)
  (h3 : compound_A_weight = total_weight - compound_B_weight) :
  compound_A_weight / compound_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_A_to_B_l195_195615


namespace assembly_line_arrangement_l195_195597

def total_ways_to_arrange : Nat :=
  let factorial (n : Nat) : Nat :=
    if n = 0 then 1 else n * factorial (n-1)
  factorial 5

theorem assembly_line_arrangement : total_ways_to_arrange = 120 := by
  sorry

end assembly_line_arrangement_l195_195597


namespace least_product_of_distinct_primes_gt_30_l195_195489

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l195_195489


namespace domain_of_problem1_domain_of_problem2_l195_195248

-- Definition for problem 1
def problem1_domain (x : ℝ) :=
  x + 3 ≥ 0 ∧ x + 2 ≠ 0

-- Definition for problem 2
def problem2_domain (x : ℝ) :=
  x > 0 ∧ log 3 x ≥ 0

-- Theorem for problem 1
theorem domain_of_problem1 (x : ℝ) : problem1_domain x ↔ x ≥ -3 ∧ x ≠ -2 :=
by sorry

-- Theorem for problem 2
theorem domain_of_problem2 (x : ℝ) : problem2_domain x ↔ x ≥ 1 :=
by sorry

end domain_of_problem1_domain_of_problem2_l195_195248


namespace sqrt_simplification_l195_195338

theorem sqrt_simplification (m : ℝ) (h : m < 1) : real.sqrt (m^2 - 2*m + 1) = 1 - m := 
by sorry

end sqrt_simplification_l195_195338


namespace product_of_elements_in_A_is_zero_l195_195796

def is_solution (x : ℤ) : Prop := x^2 + x - 2 ≤ 0
def A : Set ℤ := { x | is_solution x }

theorem product_of_elements_in_A_is_zero :
  (∏ x in A.toFinset, x) = 0 :=
sorry

end product_of_elements_in_A_is_zero_l195_195796


namespace find_c_l195_195849

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 6) (h3 : ((6 - c) / c) = 4 / 9) : c = 54 / 13 :=
sorry

end find_c_l195_195849


namespace line_circle_separate_if_point_inside_circle_l195_195933

variables {a b r : ℝ}

/-- If the point (a, b) is inside the circle x^2 + y^2 = r^2, then the line ax + by = r^2 and 
    the circle x^2 + y^2 = r^2 are separate. -/
theorem line_circle_separate_if_point_inside_circle
  (h1 : a^2 + b^2 < r^2) :
  ¬ ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ ax + by = r^2 := 
sorry

end line_circle_separate_if_point_inside_circle_l195_195933


namespace weight_of_new_person_l195_195137

theorem weight_of_new_person 
  (avg_weight_increase : ℝ) 
  (num_people : ℕ) 
  (weight_replaced_person : ℝ) 
  (total_weight_increase : ℝ := avg_weight_increase * ↑num_people):
  avg_weight_increase = 2.5 → 
  num_people = 8 → 
  weight_replaced_person = 66 → 
  ∃ W : ℝ, W = weight_replaced_person + total_weight_increase :=
by
  intros h_avg h_num h_weight
  use (weight_replaced_person + total_weight_increase)
  have h_total : total_weight_increase = 20, by 
  { rw [←h_avg, ←h_num], 
    norm_num }
  rw h_total
  rw [h_weight]
  norm_num
  sorry

end weight_of_new_person_l195_195137


namespace stratified_sampling_l195_195580

theorem stratified_sampling (total_students : ℕ) (num_freshmen : ℕ)
                            (freshmen_sample : ℕ) (sample_size : ℕ)
                            (h1 : total_students = 1500)
                            (h2 : num_freshmen = 400)
                            (h3 : freshmen_sample = 12)
                            (h4 : (freshmen_sample : ℚ) / num_freshmen = sample_size / total_students) :
  sample_size = 45 :=
  by
  -- There would be some steps to prove this, but they are omitted.
  sorry

end stratified_sampling_l195_195580


namespace blueberries_per_blue_box_l195_195544

theorem blueberries_per_blue_box
  (B S : ℕ)
  (h1 : S - B = 30)
  (h2 : (D : ℕ) + 100 = (D + S) - (B - B)) :
  B = 70 :=
  by
    have h3 : 100 = S := by
      linarith [h2]
    have h4 : S = 100 := by
      exact eq.symm h3
    have h5 : 100 - B = 30 := by
      rw [h4] at h1
      exact h1
    have h6 : B = 70 := by
      linarith [h5]
    exact h6

end blueberries_per_blue_box_l195_195544


namespace log_condition_implies_ratio_is_one_l195_195680

theorem log_condition_implies_ratio_is_one
  (a b : ℝ)
  (h1 : log a b + log b a = (5 : ℝ) / 2)
  (h2 : 1 < b)
  (h3 : b < a) :
  (a + b^4) / (a^2 + b^2) = 1 :=
by sorry

end log_condition_implies_ratio_is_one_l195_195680


namespace smallest_positive_period_intervals_of_monotonic_increase_max_value_in_interval_l195_195695

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sin (2 * x + Real.pi / 2)

theorem smallest_positive_period : 
  ∃ T > 0, (∀ x ∈ ℝ, f (x + T) = f x) ∧ (T = Real.pi) :=
  sorry

theorem intervals_of_monotonic_increase (k : ℤ) : 
  ∀ x ∈ [k * Real.pi - (3 * Real.pi / 8), k * Real.pi + (Real.pi / 8)], f' x > 0 :=
  sorry

theorem max_value_in_interval : 
  ∃ x ∈ [0, Real.pi / 3], f x = sqrt 2 :=
  sorry

end smallest_positive_period_intervals_of_monotonic_increase_max_value_in_interval_l195_195695


namespace mr_callen_total_loss_l195_195806

noncomputable def total_loss : ℕ :=
  let n_paintings := 10
  let cost_painting := 40
  let n_wooden_toys := 8
  let cost_wooden_toy := 20
  let reduction_painting := 0.10
  let reduction_wooden_toy := 0.15

  let loss_per_painting := cost_painting * reduction_painting
  let total_loss_paintings := n_paintings * loss_per_painting

  let loss_per_wooden_toy := cost_wooden_toy * reduction_wooden_toy
  let total_loss_wooden_toys := n_wooden_toys * loss_per_wooden_toy

  ((total_loss_paintings + total_loss_wooden_toys).toNat)

theorem mr_callen_total_loss : total_loss = 64 := by
  sorry

end mr_callen_total_loss_l195_195806


namespace tan_neg_7pi_over_6_l195_195994

-- Definitions for given conditions
def radians_to_degrees (r : ℝ) : ℝ := 180 * r / real.pi
def tangent_period (θ : ℝ) : ℝ := θ + 180
def tan_neg_30_deg : ℝ := - (1 / real.sqrt 3)

-- The main statement of the problem to be proved
theorem tan_neg_7pi_over_6 : real.tan (-7 * real.pi / 6) = - (1 / real.sqrt 3) := 
by
  sorry

end tan_neg_7pi_over_6_l195_195994


namespace train_speed_l195_195538

-- Definitions to capture the conditions
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 300
def time_to_cross_bridge : ℝ := 36

-- The speed of the train calculated according to the condition
def total_distance : ℝ := length_of_train + length_of_bridge

theorem train_speed : total_distance / time_to_cross_bridge = 11.11 :=
by
  sorry

end train_speed_l195_195538


namespace sequence_bounds_l195_195937

theorem sequence_bounds (n : ℕ) (hpos : 0 < n) :
  ∃ (a : ℕ → ℝ), (a 0 = 1/2) ∧
  (∀ k < n, a (k + 1) = a k + (1/n) * (a k)^2) ∧
  (1 - 1 / n < a n ∧ a n < 1) :=
sorry

end sequence_bounds_l195_195937


namespace stratified_sampling_probability_l195_195351

theorem stratified_sampling_probability :
  ∀ (total_primary total_middle total_university total_selected : ℕ) 
    (primary_selected middle_selected university_selected : ℕ) 
    (total_outcomes favorable_outcomes : ℕ),
    total_primary = 21 →
    total_middle = 14 →
    total_university = 7 →
    total_selected = 6 →
    primary_selected = 3 →
    middle_selected = 2 →
    university_selected = 1 →
    total_outcomes = 15 →
    favorable_outcomes = 3 →
    (favorable_outcomes / total_outcomes : ℚ) = 1 / 5 :=
begin
  intros,
  sorry
end

end stratified_sampling_probability_l195_195351


namespace graph_symmetry_l195_195452

-- Define the function and properties
def f : ℝ → ℝ := sorry

-- Conditions
def t (x : ℝ) : ℝ := f(x + 2)
def P := (-1 : ℝ, 3 : ℝ)
def is_symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

-- Given conditions
def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

theorem graph_symmetry 
  (h1 : passes_through t P) 
  (h2 : is_symmetric_about_origin f) : 
  passes_through f (-1, -3) :=
sorry

end graph_symmetry_l195_195452


namespace astros_win_seven_games_probability_l195_195359

theorem astros_win_seven_games_probability :
  let win_series := 4
  let dodgers_win_prob := (3 : ℚ) / 4
  let astros_win_game := (1 : ℚ) / 4
  let binom := nat.choose
  let tied_games_probability := binom 6 3 * (astros_win_game) ^ 3 * (dodgers_win_prob) ^ 3
  let astros_win_seventh_game := astros_win_game
  tied_games_probability * astros_win_seventh_game = 135 / 4096 :=
by
  let win_series := 4
  let dodgers_win_prob := (3 : ℚ) / 4
  let astros_win_game := (1 : ℚ) / 4
  let binom := nat.choose
  let tied_games_probability := binom 6 3 * (astros_win_game) ^ 3 * (dodgers_win_prob) ^ 3
  let astros_win_seventh_game := astros_win_game
  show _ = _
  sorry

end astros_win_seven_games_probability_l195_195359


namespace find_complex_z_l195_195716

theorem find_complex_z (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
sorry

end find_complex_z_l195_195716


namespace sandy_painting_area_l195_195422

theorem sandy_painting_area :
  let wall_height := 10
  let wall_length := 15
  let painting_height := 3
  let painting_length := 5
  let wall_area := wall_height * wall_length
  let painting_area := painting_height * painting_length
  let area_to_paint := wall_area - painting_area
  area_to_paint = 135 := 
by 
  sorry

end sandy_painting_area_l195_195422


namespace unique_geometric_progression_12_a_b_ab_l195_195984

noncomputable def geometric_progression_12_a_b_ab : Prop :=
  ∃ (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3

theorem unique_geometric_progression_12_a_b_ab :
  ∃! (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3 :=
by
  sorry

end unique_geometric_progression_12_a_b_ab_l195_195984


namespace assembly_line_arrangement_l195_195596

open Nat

theorem assembly_line_arrangement :
  let A := "add axles"
  let W := "add wheels"
  let I := "install windshield"
  let P := "install instrument panel"
  let S := "install steering wheel"
  let C := "install interior seating"
  let tasks := [A, W, I, P, S, C]
  (∃ (order : List String) (H : Multiset Nodup order), -- There exists an order of tasks
    Multiset.card (Multiset.filter (λ t, t = A ∨ t = W) (Multiset.of_list order)) = 2 -- Axles and Wheels both are in order.
    ∧ List.index_of A order < List.index_of W order -- Axles are before Wheels.
  ) →
  (fact 5 = 120) -- The number of valid arrangements is 120
 := by {
  sorry 
}

end assembly_line_arrangement_l195_195596


namespace prob_two_red_balls_consecutively_without_replacement_l195_195350

def numOfRedBalls : ℕ := 3
def totalNumOfBalls : ℕ := 8

theorem prob_two_red_balls_consecutively_without_replacement :
  (numOfRedBalls / totalNumOfBalls) * ((numOfRedBalls - 1) / (totalNumOfBalls - 1)) = 3 / 28 :=
by
  sorry

end prob_two_red_balls_consecutively_without_replacement_l195_195350


namespace median_equality_of_sorted_quartet_range_subsets_l195_195278

variable {α : Type*} [LinearOrder α] [Add α] [Div α]
variable (x1 x2 x3 x4 x5 x6 : α)

theorem median_equality_of_sorted_quartet :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x3 + x4) / 2 = (x3 + x4) / 2 :=
sorry

theorem range_subsets :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_of_sorted_quartet_range_subsets_l195_195278


namespace exists_starting_position_l195_195072

theorem exists_starting_position (n : ℕ) (a b : Fin n → ℝ) (h : ∑ i, a i = ∑ i, b i) :
  ∃ (s : Fin n), 
    ∀ k: Fin n, let i := (s + k) % n in
    ∑ j in Finset.range ((s + k + 1) % n), (a (s + j) % n - b (s + j) % n) ≥ 0 := 
sorry

end exists_starting_position_l195_195072


namespace maximum_value_of_vectors_l195_195293

open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 3))

def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1

def given_conditions (a b c : EuclideanSpace ℝ (Fin 3)) : Prop :=
  unit_vector a ∧ unit_vector b ∧ ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖ ∧ ‖c‖ = 2

theorem maximum_value_of_vectors
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖)
  (hc : ‖c‖ = 2) :
  ‖a + b - c‖ ≤ sqrt 2 + 2 := 
by
  sorry

end maximum_value_of_vectors_l195_195293


namespace equilateral_triangle_incircle_l195_195382

/-- Let 𝛾 be the incircle of an equilateral triangle ABC of side length 2 units.
(a) Show that for all points P on 𝛾, PA² + PB² + PC² = 5.
(b) Show that for all points P on 𝛾, it is possible to construct a triangle of sides 
PA, PB, PC and whose area is √3/4 units. -/
theorem equilateral_triangle_incircle (A B C P : ℝ) (a : ℝ) (h_a : a = 2) :
  (∀ P ∈ 𝛾, PA² + PB² + PC² = 5) ∧
  (∀ P ∈ 𝛾, area_of_triangle PA PB PC = sqrt 3 / 4) :=
sorry

end equilateral_triangle_incircle_l195_195382


namespace right_triangle_area_l195_195676

variable (a b : ℝ)
variable (h1 : a^2 + b^2 = 100)
variable (h2 : a + b = 14)

theorem right_triangle_area : 
  (1 / 2) * a * b = 24 :=
by 
sor

end right_triangle_area_l195_195676


namespace blanch_initial_slices_l195_195958

theorem blanch_initial_slices :
  let breakfast := 4
  let lunch := 2
  let snack := 2
  let dinner := 5
  let slices_left := 2
  let total_eaten := breakfast + lunch + snack + dinner
  in total_eaten + slices_left = 15 :=
by
  let breakfast := 4
  let lunch := 2
  let snack := 2
  let dinner := 5
  let slices_left := 2
  let total_eaten := breakfast + lunch + snack + dinner
  show total_eaten + slices_left = 15
  sorry

end blanch_initial_slices_l195_195958


namespace exactly_one_line_meeting_conditions_l195_195272

noncomputable def line_meet_conditions (a : ℝ) : Prop :=
∃ l : ℝ → ℝ, (∀ x, l x = x - a) ∧ (a ≠ 0) ∧ (dist (1, -1) (1, -1 + a) = sqrt 2)

theorem exactly_one_line_meeting_conditions :
  ∃! a : ℝ, line_meet_conditions a := by
  sorry

end exactly_one_line_meeting_conditions_l195_195272


namespace minimum_value_x_plus_3y_plus_6z_l195_195392

theorem minimum_value_x_plus_3y_plus_6z 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y * z = 18) : 
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end minimum_value_x_plus_3y_plus_6z_l195_195392


namespace num_correct_propositions_l195_195909

-- Definitions for the conditions

variable (l : Type) [LinearOrder l] -- l represents a line
variable (α β : Type) [LinearOrder α] [LinearOrder β] -- α and β represent planes

variable (perpendicular : l → α → Prop) -- l is perpendicular to α
variable (parallel : l → β → Prop) -- l is parallel to β
variable (perpendicular_planes : α → β → Prop) -- α is perpendicular to β

-- Hypotheses based on the conditions given in the problem
variable (l_perp_alpha : perpendicular l α)
variable (l_para_beta : parallel l β)
variable (alpha_perp_beta : perpendicular_planes α β)

-- Mathematically equivalent proof problem statement
theorem num_correct_propositions :
  (cond1 : (perpendicular l α) → (parallel l β) → (perpendicular_planes α β)) ∧
  (cond2 : (perpendicular l α) → (perpendicular_planes α β) → (parallel l β)) ∧
  ¬(cond3 : (parallel l β) → (perpendicular_planes α β) → (perpendicular l α)) →
  count_correct_propositions = 2 :=
sorry

end num_correct_propositions_l195_195909


namespace sum_A_B_equals_twelve_l195_195091

def is_single_digit (n : ℕ) : Prop := n < 10

def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits_multiple_of_72 (A B : ℕ) : Prop :=
  let num := A * 100000 + 4 * 10000 + 4 * 1000 + 6 * 100 + 1 * 10 + B
  num % 72 = 0

theorem sum_A_B_equals_twelve {A B : ℕ} (hA : is_single_digit A) (hB : is_single_digit B) 
    (h : sum_of_digits_multiple_of_72 A B) : 
  A + B = 12 :=
begin
  -- proof goes here, but not needed for this task
  sorry
end

end sum_A_B_equals_twelve_l195_195091


namespace infinite_n_frac_divisors_integer_l195_195977

def num_divisors (n : ℕ) : ℕ :=
  (factors n).map (λ p, (p.2 + 1)).prod

theorem infinite_n_frac_divisors_integer :
  ∃ᶠ n in at_top, ∃ k : ℕ, n = k * num_divisors n := sorry

end infinite_n_frac_divisors_integer_l195_195977


namespace positive_value_of_a_l195_195911

-- Definitions based on the problem's conditions
variable (X : ℝ → Prop)
variable (normal_dist : X ∼ Normal 1 σ)
variable (a : ℝ)

-- The proposition that needs to be proven
theorem positive_value_of_a (h : P(X ≤ a ^ 2 - 1) = P(X > a - 3)) : a = 2 := by
  sorry

end positive_value_of_a_l195_195911


namespace max_additional_plates_l195_195653

def letters1 := {'B', 'F', 'K', 'S', 'Z'}
def letters2 := {'E', 'U', 'Y'}
def letters3 := {'G', 'J', 'Q', 'V'}

def initial_plates : ℕ := 5 * 3 * 4 -- 60 initial plates

-- The problem is to prove the largest number of additional plates is 40
theorem max_additional_plates : 
  (letters1.size + 2) * letters2.size * letters3.size - initial_plates ≤ 40 ∧
  letters1.size * (letters2.size + 2) * letters3.size - initial_plates ≤ 40 ∧
  letters1.size * letters2.size * (letters3.size + 2) - initial_plates ≤ 40 ∧
  (letters1.size + 1) * (letters2.size + 1) * letters3.size - initial_plates ≤ 40 ∧
  (letters1.size + 1) * letters2.size * (letters3.size + 1) - initial_plates ≤ 40 ∧
  letters1.size * (letters2.size + 1) * (letters3.size + 1) - initial_plates ≤ 40 := 
sorry

end max_additional_plates_l195_195653


namespace amount_left_in_wallet_l195_195470

theorem amount_left_in_wallet
  (initial_amount : ℝ)
  (spent_amount : ℝ)
  (h_initial : initial_amount = 94)
  (h_spent : spent_amount = 16) :
  initial_amount - spent_amount = 78 :=
by
  sorry

end amount_left_in_wallet_l195_195470


namespace smallest_number_to_add_l195_195906

theorem smallest_number_to_add (x : ℕ) : (8261955 + x) % 11 = 0 ↔ x = 2 := 
begin
  sorry
end

end smallest_number_to_add_l195_195906


namespace towns_actual_distance_l195_195446

/-- The distance on the map between two towns is 18 inches. -/
def map_distance : ℝ := 18

/-- The scale factor of the map is 0.2 inches = 4 miles. -/
def scale_map_inch_to_miles : ℝ := 4 / 0.2

/-- The actual distance between the towns. -/
def actual_distance := map_distance * scale_map_inch_to_miles

/-- Prove that the actual distance between two towns is 360 miles.
given that the distance on the map is 18 inches and the scale is 0.2 inches = 4 miles. -/
theorem towns_actual_distance : actual_distance = 360 :=
by
  sorry

end towns_actual_distance_l195_195446


namespace y_intercept_of_line_b_l195_195014

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l195_195014


namespace sum_of_solutions_correct_l195_195860

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in {x : ℝ | -2 ≤ x ∧ x ≤ 4 ∧ 2 * (x - 1) * (Real.sin (Real.pi * x)) + 1 = 0}, x

theorem sum_of_solutions_correct :
  sum_of_solutions = 8 :=
by
  sorry

end sum_of_solutions_correct_l195_195860


namespace exactly_one_line_l195_195747

theorem exactly_one_line (a b : ℕ) (ha : even a) (hb : odd b) (h : ∃ (a b : ℕ), even a ∧ odd b ∧ 
  (∃ (a b : ℕ), even a ∧ odd b ∧ (6 * b + 5 * a = a * b)) ∧ 
  (∀ (a b : ℕ), even a ∧ odd b → (6 * b + 5 * a = a * b) → (a, b) = (36, 6))):
  ∃! (a b : ℕ), even a ∧ odd b ∧ (6 * b + 5 * a = a * b) :=
sorry

end exactly_one_line_l195_195747


namespace a_2015_eq_l195_195275

noncomputable def a : ℕ+ → ℚ
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 1 / 2
| ⟨n+3, h⟩ := 2 / ((1 / a ⟨n+1, Nat.succ_pos _⟩) + (1 / a ⟨n+2, lt_trans (Nat.succ_pos _) h⟩))

theorem a_2015_eq : a ⟨2015, by decide⟩ = 1 / 2015 :=
sorry

end a_2015_eq_l195_195275


namespace number_of_x_intercepts_l195_195204

def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem number_of_x_intercepts : ∃! (x : ℝ), ∃ (y : ℝ), parabola y = x ∧ y = 0 :=
by
  sorry

end number_of_x_intercepts_l195_195204


namespace y_intercept_of_line_b_l195_195021

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l195_195021


namespace decreasing_function_a_l195_195310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4 * a * x + 2 else log a x

theorem decreasing_function_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by
  sorry

end decreasing_function_a_l195_195310


namespace sum_odd_divisors_120_l195_195524

theorem sum_odd_divisors_120 : 
  let n := 120 
  let odd_divisors := {d : Nat | d ∣ n ∧ d % 2 = 1}
  Finset.sum (Finset.filter (λ x => x ∈ odd_divisors) (Finset.range (n + 1))) = 24 := 
by
  sorry

end sum_odd_divisors_120_l195_195524


namespace mean_median_mode_equal_l195_195237

def data := [20, 30, 40, 50, 50, 60, 70, 80]

noncomputable def mean (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

def mode (lst : List ℕ) : ℕ :=
  lst.foldl (λ x y -> if (lst.count x) > (lst.count y) then x else y) (lst.head!)

noncomputable def median (lst : List ℕ) : ℚ :=
  let sorted := lst.qsort (· < ·)
  if lst.length % 2 = 0 then
    (sorted.get! (lst.length / 2 - 1) + sorted.get! (lst.length / 2)) / 2
  else
    sorted.get! (lst.length / 2)

theorem mean_median_mode_equal :
  mean data = 50 ∧ median data = 50 ∧ mode data = 50 := sorry

end mean_median_mode_equal_l195_195237


namespace mark_profit_from_selling_magic_card_l195_195034

theorem mark_profit_from_selling_magic_card : 
    ∀ (purchase_price new_value profit : ℕ), 
        purchase_price = 100 ∧ 
        new_value = 3 * purchase_price ∧ 
        profit = new_value - purchase_price 
    → 
        profit = 200 := 
by 
  intros purchase_price new_value profit h,
  cases h with hp1 h,
  cases h with hv1 hp2,
  rw hp1 at hv1,
  rw hp1 at hp2,
  rw hv1 at hp2,
  rw hp2,
  rw hp1,
  norm_num,
  exact eq.refl 200

end mark_profit_from_selling_magic_card_l195_195034


namespace strawberries_weight_before_l195_195025

variables (M D E B : ℝ)

noncomputable def total_weight_before (M D E : ℝ) := M + D - E

theorem strawberries_weight_before :
  ∀ (M D E : ℝ), M = 36 ∧ D = 16 ∧ E = 30 → total_weight_before M D E = 22 :=
by
  intros M D E h
  simp [total_weight_before, h]
  sorry

end strawberries_weight_before_l195_195025


namespace maximum_small_circles_l195_195044

-- Definitions for small circle radius, large circle radius, and the maximum number n.
def smallCircleRadius : ℝ := 1
def largeCircleRadius : ℝ := 11

-- Function to check if small circles can be placed without overlapping
def canPlaceCircles (n : ℕ) : Prop := n * 2 < 2 * Real.pi * (largeCircleRadius - smallCircleRadius)

theorem maximum_small_circles : ∀ n : ℕ, canPlaceCircles n → n ≤ 31 := by
  sorry

end maximum_small_circles_l195_195044


namespace log_sum_correct_l195_195124

noncomputable def log_sum : ℝ := 
  Real.log 8 / Real.log 10 + 
  3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 2 / Real.log 10 +
  2 * Real.log 5 / Real.log 10 +
  5 * Real.log 25 / Real.log 10

theorem log_sum_correct : abs (log_sum - 12.301) < 0.001 :=
by sorry

end log_sum_correct_l195_195124


namespace sum_integral_c_values_l195_195629

theorem sum_integral_c_values :
  (∑ c in (Finset.filter (λ c, Int.sqrt (25 + 4 * c) * Int.sqrt (25 + 4 * c) = (25 + 4 * c))
                          (Finset.Icc (-6) 18)), c) = 10 :=
by
  sorry

end sum_integral_c_values_l195_195629


namespace least_prime_product_l195_195488
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l195_195488


namespace pentagon_area_proof_l195_195981

noncomputable def area_of_pentagon : ℕ :=
  let side1 := 18
  let side2 := 25
  let side3 := 30
  let side4 := 28
  let side5 := 25
  -- Assuming the total area calculated from problem's conditions
  950

theorem pentagon_area_proof : area_of_pentagon = 950 := by
  sorry

end pentagon_area_proof_l195_195981


namespace average_fixed_points_l195_195930

noncomputable def expected_fixed_points (n : ℕ) : ℕ := 1

theorem average_fixed_points (n : ℕ) :
  (∑ σ in (Finset.univ : Finset (Equiv.Perm (Fin n))), (Finset.filter (λ i, σ i = i) (Finset.univ : Finset (Fin n))).card) / n! = expected_fixed_points n := by
  sorry

end average_fixed_points_l195_195930


namespace percentage_of_chess_in_swimming_l195_195865

noncomputable def total_students : ℕ := 2000
noncomputable def chess_percentage : ℝ := 0.10
noncomputable def swimming_students : ℕ := 100

def chess_students : ℕ := (total_students : ℝ * chess_percentage).to_nat

theorem percentage_of_chess_in_swimming :
  (swimming_students : ℝ / chess_students * 100) = 50 :=
by
  sorry

end percentage_of_chess_in_swimming_l195_195865


namespace sum_of_fractions_l195_195611

theorem sum_of_fractions:
  (7 / 12) + (11 / 15) = 79 / 60 :=
by
  sorry

end sum_of_fractions_l195_195611


namespace part1_part2_l195_195311

-- Definition of the given function
def f (x : ℝ) : ℝ := 2 * real.cos x * real.sin (x + real.pi / 6) + 1

-- Part 1: Prove the smallest positive period and the monotonically increasing intervals
theorem part1 : 
  (∀ x : ℝ, f (x + real.pi) = f x) ∧ -- Smallest positive period
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ real.pi → 
    ((0 ≤ x ∧ x ≤ real.pi / 6) ∨ (real.pi * 2 / 3 ≤ x ∧ x ≤ real.pi)) → 
    f x ≤ f (x + 1e-6)) := -- Monotonically increasing intervals
by
  sorry

-- Part 2: Prove the range of the function in the given interval
theorem part2 :
  ∀ x : ℝ, (-real.pi / 6) ≤ x ∧ x ≤ real.pi / 3 → (1 ≤ f x ∧ f x ≤ 5 / 2) :=
by
  sorry

end part1_part2_l195_195311


namespace find_x_l195_195545

theorem find_x : ∃ (x : ℝ), x > 0 ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_x_l195_195545


namespace alice_has_ball_after_three_turns_l195_195591

def probability_Alice_has_ball (turns: ℕ) : ℚ :=
  match turns with
  | 0 => 1 -- Alice starts with the ball
  | _ => sorry -- We would typically calculate this by recursion or another approach.

theorem alice_has_ball_after_three_turns :
  probability_Alice_has_ball 3 = 11 / 27 :=
by
  sorry

end alice_has_ball_after_three_turns_l195_195591


namespace least_product_of_distinct_primes_greater_than_30_l195_195511

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l195_195511


namespace problem_statement_l195_195000

theorem problem_statement
  (n : ℕ) (a b c d : ℤ)
  (h1 : n > 0)
  (h2 : n ∣ a + b + c + d)
  (h3 : n ∣ a^2 + b^2 + c^2 + d^2) :
  n ∣ a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d :=
sorry

end problem_statement_l195_195000


namespace john_bathroom_uses_during_movie_and_intermissions_l195_195376

-- Define the conditions
def uses_bathroom_interval := 50   -- John uses the bathroom every 50 minutes
def walking_time := 5              -- It takes him an additional 5 minutes to walk to and from the bathroom
def movie_length := 150            -- The movie length in minutes (2.5 hours)
def intermission_length := 15      -- Each intermission length in minutes
def intermission_count := 2        -- The number of intermissions

-- Derived condition
def effective_interval := uses_bathroom_interval + walking_time

-- Total movie time including intermissions
def total_movie_time := movie_length + (intermission_length * intermission_count)

-- Define the theorem to be proved
theorem john_bathroom_uses_during_movie_and_intermissions : 
  ∃ n : ℕ, n = 3 + 2 ∧ total_movie_time = 180 ∧ effective_interval = 55 :=
by
  sorry

end john_bathroom_uses_during_movie_and_intermissions_l195_195376


namespace exists_polynomial_pairwise_powers_of_two_l195_195822

-- Define the problem statement
theorem exists_polynomial_pairwise_powers_of_two (n : ℕ) (hn : n > 0) : 
  ∃ P : ℤ[X], 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ k : ℕ, P.eval (i : ℤ) = 2^k) ∧ 
    (∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j →
      ∃ ki kj : ℕ, P.eval (i : ℤ) = 2^ki ∧ P.eval (j : ℤ) = 2^kj ∧ ki ≠ kj) :=
by 
  sorry

end exists_polynomial_pairwise_powers_of_two_l195_195822


namespace profit_calculation_l195_195030

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l195_195030


namespace area_triangle_ABN_l195_195603

-- Variables representing points in the plane
variables (A B C P Q N : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q] [metric_space N]
-- Distances from points
variables (BC AC : Type*) [has_scalar ℝ BC] [has_scalar ℝ AC]
-- Areas and intersection point
variables (area_ABC area_ABN : ℝ)

-- Conditions
axiom CP_eq_third_BC : P = C + (1 / 3 : ℝ) • BC
axiom CQ_eq_quarter_AC : Q = C + (1 / 4 : ℝ) • AC
axiom area_triangle_ABC : area_ABC = 12
axiom N_intersection_AP_BQ : N = (span ℝ (A + (1 / 3 : ℝ) • BC)) ∩ (span ℝ (B + (1 / 4 : ℝ) • AC))

theorem area_triangle_ABN : area_ABN = 72 / 11 :=
sorry

end area_triangle_ABN_l195_195603


namespace balloon_permutations_l195_195222

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l195_195222


namespace original_population_l195_195903

theorem original_population (X : ℕ) (h1 : 0.10 * X = x1) (h2 : 0.25 * (X - x1) = x2) (h3 : X - x1 - x2 = 5130) : X = 7600 :=
by
  sorry

end original_population_l195_195903


namespace inequalities_hold_l195_195062

noncomputable def digit_assignment := 
  ('T', 9), ('R', 6), ('A', 4), ('N', 0), ('S', 1), ('P', 7), ('O', 5), ('B', 2), ('K', 3)

theorem inequalities_hold:
  let T := 9
  let R := 6
  let A := 4
  let N := 0
  let S := 1
  let P := 7
  let O := 5
  let B := 2
  let K := 3
  T > R ∧ R > A ∧ A > N ∧ N < S ∧ S < P ∧ P < O ∧ O < R ∧ R < T ∧
  T > R ∧ R > O ∧ O < A ∧ A > B ∧ B < K ∧ K < A :=
by
  sorry

end inequalities_hold_l195_195062


namespace choose_18_4_eq_3060_l195_195610

/-- The number of ways to select 4 members from a group of 18 people (without regard to order). -/
theorem choose_18_4_eq_3060 : Nat.choose 18 4 = 3060 := 
by
  sorry

end choose_18_4_eq_3060_l195_195610


namespace matrix_inverse_matrix_eigenvalues_and_eigenvectors_l195_195320

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![2, 0], ![1, 1]]

theorem matrix_inverse :
  inverse M = ![![1/2, 0], ![-1/2, 1]] :=
sorry

theorem matrix_eigenvalues_and_eigenvectors :
  ∃ (λ1 λ2 : ℚ) (v1 v2 : Vector (Fin 2) ℚ),
  (λ1 = 1 ∧ v1 = ![0, 1]) ∧
  (λ2 = 2 ∧ v2 = ![1, 1]) ∧
  isEigenpair M λ1 v1 ∧
  isEigenpair M λ2 v2 :=
sorry

end matrix_inverse_matrix_eigenvalues_and_eigenvectors_l195_195320


namespace anna_discontinued_coaching_on_2nd_august_l195_195600

theorem anna_discontinued_coaching_on_2nd_august
  (coaching_days : ℕ) (non_leap_year : ℕ) (first_day : ℕ) 
  (days_in_january : ℕ) (days_in_february : ℕ) (days_in_march : ℕ) 
  (days_in_april : ℕ) (days_in_may : ℕ) (days_in_june : ℕ) 
  (days_in_july : ℕ) (days_in_august : ℕ)
  (not_leap_year : non_leap_year = 365)
  (first_day_of_year : first_day = 1)
  (january_days : days_in_january = 31)
  (february_days : days_in_february = 28)
  (march_days : days_in_march = 31)
  (april_days : days_in_april = 30)
  (may_days : days_in_may = 31)
  (june_days : days_in_june = 30)
  (july_days : days_in_july = 31)
  (august_days : days_in_august = 31)
  (total_coaching_days : coaching_days = 245) :
  ∃ day, day = 2 ∧ month = "August" := 
sorry

end anna_discontinued_coaching_on_2nd_august_l195_195600


namespace min_value_fraction_l195_195387

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end min_value_fraction_l195_195387


namespace nth_pattern_eq_l195_195810

theorem nth_pattern_eq (n : ℕ) : 
  (∏ i in (finset.range n).map (λ i, n+i+1), i) = 
  2^n * (∏ i in (finset.range n).map (λ k, 2*k+1), i) :=
sorry

end nth_pattern_eq_l195_195810


namespace arithmetic_sequence_sum_l195_195355

noncomputable def sum_of_first_n_terms (n : ℕ) : ℝ :=
  - (3 / 4) * (n ^ 2) + (7 / 4) * n

theorem arithmetic_sequence_sum (d : ℝ) (n : ℕ) (h_nonzero : d ≠ 0) 
  (h_geometric : (1 + 4 * d) ^ 2 = (1 + 2 * d) * (1 + 9 * d)) :
  let a : ℕ → ℝ := λ n, 1 + (n - 1) * d in
  (∑ i in range (n + 1), a i) = sum_of_first_n_terms n :=
by sorry

end arithmetic_sequence_sum_l195_195355


namespace area_AM_E_l195_195825

noncomputable def point (A B : ℝ × ℝ) (ratio : ℝ) : ℝ × ℝ :=
  ( (ratio * fst A + snd A) / (1 + ratio), snd A)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ( (fst P + fst Q) / 2, (snd P + snd Q) / 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((fst P - fst Q)^2 + (snd P - snd Q)^2)

def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (fst A * (snd B - snd C) + fst B * (snd C - snd A) + fst C * (snd A - snd B))

theorem area_AM_E (A B C E M : ℝ × ℝ) (h_AB : distance A B = 10) (h_BC : distance B C = 8) (h_am : M = midpoint A C) (h_Epos : E = point A B (3 / 2)) (h_perpendicular : (distance E M)^2 = (distance A M)^2 - (distance A E)^2) :
  area_triangle A M E = 3 * real.sqrt 5 :=
sorry

end area_AM_E_l195_195825


namespace problem_part1_problem_part2_l195_195342

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end problem_part1_problem_part2_l195_195342


namespace green_block_weight_l195_195774

theorem green_block_weight
    (y : ℝ)
    (g : ℝ)
    (h1 : y = 0.6)
    (h2 : y = g + 0.2) :
    g = 0.4 :=
by
  sorry

end green_block_weight_l195_195774


namespace balloon_permutations_l195_195206

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l195_195206


namespace find_longer_side_length_l195_195934

noncomputable def length_of_longer_side (width height distance_between_poles num_poles : ℝ) : ℝ :=
  let perimeter := (num_poles - 1) * distance_between_poles in
  let equation := perimeter = 2 * (width + height) in
  let height_value := (perimeter / 2) - width in
  height_value

theorem find_longer_side_length (width : ℝ) (distance_between_poles : ℝ) (num_poles : ℝ) 
  (perimeter : ℝ) (height : ℝ) : length_of_longer_side width height distance_between_poles num_poles = 47.5 :=
by
  sorry

end find_longer_side_length_l195_195934


namespace angle_equality_proof_l195_195190

noncomputable def proof_problem : Prop :=
  ∀ (A B C D E F: Type)
    (h1 : AcuteAngledTriangle A B C)
    (h2 : Circumcircle A B C O)
    (h3 : Diameter AD O)
    (h4 : PerpendicularToLine BC (LineThroughPoints B C) (ExtensionPointsOf CA BA E F)),
  ∠ADF = ∠BED

-- To avoid cluttering with proof steps, we use sorry here
theorem angle_equality_proof : proof_problem := sorry

end angle_equality_proof_l195_195190


namespace inscribed_circle_radius_l195_195119

variable (AB AC BC : ℝ) (r : ℝ)

theorem inscribed_circle_radius 
  (h1 : AB = 9) 
  (h2 : AC = 9) 
  (h3 : BC = 8) : r = (4 * Real.sqrt 65) / 13 := 
sorry

end inscribed_circle_radius_l195_195119


namespace additional_charge_l195_195152

variable (charge_first : ℝ) -- The charge for the first 1/5 of a mile
variable (total_charge : ℝ) -- Total charge for an 8-mile ride
variable (distance : ℝ) -- Total distance of the ride

theorem additional_charge 
  (h1 : charge_first = 3.50) 
  (h2 : total_charge = 19.1) 
  (h3 : distance = 8) :
  ∃ x : ℝ, x = 0.40 :=
  sorry

end additional_charge_l195_195152


namespace circle_intersection_l195_195753

theorem circle_intersection (z : ℂ) (k : ℝ) :
  (∣z - 2∣ = 3 * ∣z + 2∣) → (∃ k, ∣z∣ = k) → (k = 1 ∨ k = 4) :=
by
  intros h1 h2
  sorry  -- Proof to be completed

end circle_intersection_l195_195753


namespace anna_bob_numbers_not_equal_l195_195953

theorem anna_bob_numbers_not_equal (A : ℕ) (B : ℕ) 
    (hA : ∃ (n : ℕ), A = ∑ i in range (20), to_digits 10 (n + i))
    (hB : ∃ (m : ℕ), B = ∑ i in range (21), to_digits 10 (m + i)) : 
    A ≠ B :=
by
  sorry

end anna_bob_numbers_not_equal_l195_195953


namespace balloon_permutations_l195_195218

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l195_195218


namespace james_painted_area_l195_195767

-- Define the dimensions of the wall and windows
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 6

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length
def total_window_area : ℕ := window1_area + window2_area
def painted_area : ℕ := wall_area - total_window_area

theorem james_painted_area : painted_area = 123 :=
by
  -- The proof is omitted
  sorry

end james_painted_area_l195_195767


namespace factorize_expression_l195_195993

theorem factorize_expression (y a : ℝ) : 
  3 * y * a ^ 2 - 6 * y * a + 3 * y = 3 * y * (a - 1) ^ 2 :=
by
  sorry

end factorize_expression_l195_195993


namespace line_b_y_intercept_l195_195013

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l195_195013


namespace distinct_color_cube_construction_l195_195155

/--
  Prove that the number of distinct ways to construct a 2x2x2 cube using 3 red unit cubes, 
  3 blue unit cubes, and 2 green unit cubes, where distinctiveness is considered under rotational symmetry, is 26.
-/
theorem distinct_color_cube_construction : 
  ∃ (n : ℕ), n = 26 ∧ ∀ (r b g : ℕ), r = 3 → b = 3 → g = 2 → 
  -- Calculation of distinct configurations under rotation
  -- Assuming the group G of rotational symmetries fixing configurations
  let total_configurations := (fact 8 / (fact 3 * fact 3 * fact 2)) in
  let face_rotations := 3 * 0 in
  let edge_rotations := 6 * 6 in
  let total_fixed_points := total_configurations + edge_rotations + face_rotations in
  (total_fixed_points / 24) = 26 :=
begin
  -- We skip the proof details and directly assert the existence
  use 26,
  split,
  { refl, },
  { intros r b g hr hb hg,
    let total_configurations := fact 8 / (fact 3 * fact 3 * fact 2),
    let face_rotations := 3 * 0,
    let edge_rotations := 6 * 6,
    let total_fixed_points := total_configurations + edge_rotations + face_rotations,
    have : total_fixed_points / 24 = 26, by sorry,
    assumption
  }
end

end distinct_color_cube_construction_l195_195155


namespace gain_percent_is_correct_l195_195900

theorem gain_percent_is_correct :
  let CP : ℝ := 450
  let SP : ℝ := 520
  let gain : ℝ := SP - CP
  let gain_percent : ℝ := (gain / CP) * 100
  gain_percent = 15.56 :=
by
  sorry

end gain_percent_is_correct_l195_195900


namespace impossible_partition_of_integers_l195_195196

theorem impossible_partition_of_integers :
  ¬ ∃ (S1 S2 S3 : set ℤ), ∀ n : ℤ, 
    ((n ∈ S1 ∧ n-50 ∈ S2 ∧ n+1987 ∈ S3) ∨ 
     (n ∈ S1 ∧ n-50 ∈ S3 ∧ n+1987 ∈ S2) ∨ 
     (n ∈ S2 ∧ n-50 ∈ S1 ∧ n+1987 ∈ S3) ∨
     (n ∈ S2 ∧ n-50 ∈ S3 ∧ n+1987 ∈ S1) ∨
     (n ∈ S3 ∧ n-50 ∈ S1 ∧ n+1987 ∈ S2) ∨
     (n ∈ S3 ∧ n-50 ∈ S2 ∧ n+1987 ∈ S1)) := 
by {
  sorry
}

end impossible_partition_of_integers_l195_195196


namespace Sm_7_equals_2_l195_195795

variable (x : ℝ)

def Sm (m : ℕ) : ℝ := x^m + 1/(x^m)

theorem Sm_7_equals_2 (h : x + 1/x = 2) : Sm x 7 = 2 :=
sorry

end Sm_7_equals_2_l195_195795


namespace point_distance_from_original_position_l195_195940

theorem point_distance_from_original_position :
  ∀ (s : ℝ), s^2 = 12 → 
  ∀ (x : ℝ), ∃ A : ℝ, 1 / 2 * x^2 = 6 → A^2 = 24 ∧ A = 2 * real.sqrt 6 :=
begin
  assume s,
  assume hs : s^2 = 12,
  assume x,
  assume hx : 1 / 2 * x^2 = 6,
  existsi (x : ℝ),
  split,
  {
    -- half the are of the square is 6
    calc (1 / 2 * x^2) = 6   : by exact hx 
              ...        x^2 = 12,
    sorry
  },
  {
    
    sorry
  },
end

end point_distance_from_original_position_l195_195940


namespace smallest_x_for_perfect_cube_l195_195907

theorem smallest_x_for_perfect_cube :
  ∃ (x : ℕ) (h : x > 0), x = 36 ∧ (∃ (k : ℕ), 1152 * x = k ^ 3) := by
  sorry

end smallest_x_for_perfect_cube_l195_195907


namespace transform_A_to_A_plus_one_l195_195533

-- Define the operations
def add_nine (n : ℕ) : ℕ := n + 9

def erase_one (n : ℕ) : ℕ :=
  let str := n.toString;
  let str_ef := str.filter (λ c, c ≠ '1');
  if str_ef.front = some '0' then str_ef.tail.get.toNat else str_ef.toNat
  
theorem transform_A_to_A_plus_one (A : ℕ) : ∃ n, transform n A = A + 1 := 
begin
  sorry
end

end transform_A_to_A_plus_one_l195_195533


namespace angle_AMD_is_45_degrees_l195_195437

-- Define the square ABCD with side length 5
variable (A B C D M: Point)
variable (side_length : ℝ)
variable (h_square : square A B C D ∧ side_length = 5)

-- Define the points and condition: Point M is on side AB and ∠AMD = ∠CMD
variable (h_M_on_AB : M ∈ segment A B)
variable (h_angles_equal : angle A M D = angle C M D)
variable (h_correct_answer : angle A M D = 45)

-- Statement to prove
theorem angle_AMD_is_45_degrees : angle A M D = 45 := by
  sorry

end angle_AMD_is_45_degrees_l195_195437


namespace arithmetic_sequence_sum_l195_195126

theorem arithmetic_sequence_sum (x y : ℕ) (a : ℕ → ℕ) 
  (h1 : a 0 = 3)
  (h2 : a 1 = 7)
  (h3 : a 2 = 11)
  (h4 : a (a.length - 1) = 35)
  (h5 : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) 
  : x + y = 58 := 
sorry

end arithmetic_sequence_sum_l195_195126


namespace card_probability_is_correct_l195_195094

noncomputable def total_cards := 52
noncomputable def probability_of_event : ℚ := 17 / 11050

def event_probability : ℚ :=
  let p_first_5_3_suits      := (3/total_cards) * (12/(total_cards-1)) * (4/(total_cards-2))
  let p_first_5_diamond   := (3/total_cards) * (1/(total_cards-1)) * (3/(total_cards-2))
  let p_first_5diamond_3suits  := (1/total_cards) * (12/(total_cards-1)) * (4/(total_cards-2))
  let p_first_5diamond_3diamond := (1/total_cards) * (1/(total_cards-1)) * (3/(total_cards-2))
  
  p_first_5_3_suits + p_first_5_diamond + p_first_5diamond_3suits + p_first_5diamond_3diamond

theorem card_probability_is_correct : event_probability = probability_of_event := by
  sorry

end card_probability_is_correct_l195_195094


namespace retreat_center_probability_each_guest_gets_one_of_each_l195_195578

/-- Suppose there are four guests. Each guest should receive one nut roll, 
one cheese roll, one fruit roll, and one veggie roll. Once wrapped, 
the rolls became indistinguishable and rolls are picked randomly for each guest. 
The probability that each guest receives one roll of each type is \( \frac{1}{5320} \). -/
theorem retreat_center_probability_each_guest_gets_one_of_each :
  let p : ℚ := 1 / 5320 in
  p = (4 / 16) * (4 / 15) * (4 / 14) * (4 / 13) * (3 / 12) * (3 / 11) * (3 / 10) * (3 / 9) *
      (2 / 8) * (2 / 7) * (2 / 6) * (2 / 5) * 1 :=
sorry

end retreat_center_probability_each_guest_gets_one_of_each_l195_195578


namespace probability_zero_after_2017_days_l195_195023

-- Define the people involved
inductive Person
| Lunasa | Merlin | Lyrica
deriving DecidableEq, Inhabited

open Person

-- Define the initial state with each person having their own distinct hat
def initial_state : Person → Person
| Lunasa => Lunasa
| Merlin => Merlin
| Lyrica => Lyrica

-- Define a function that represents switching hats between two people
def switch_hats (p1 p2 : Person) (state : Person → Person) : Person → Person :=
  λ p => if p = p1 then state p2 else if p = p2 then state p1 else state p

-- Define a function to represent the state after n days (iterations)
def iter_switch_hats (n : ℕ) : Person → Person :=
  sorry -- This would involve implementing the iterative random switching

-- Proposition: The probability that after 2017 days, every person has their own hat back is 0
theorem probability_zero_after_2017_days :
  iter_switch_hats 2017 = initial_state → false :=
by
  sorry

end probability_zero_after_2017_days_l195_195023


namespace sum_inverse_r_le_sum_inverse_a_l195_195385

theorem sum_inverse_r_le_sum_inverse_a (a : ℕ → ℕ) (r : ℕ → ℕ)
  (h1 : r 1 = 2)
  (h2 : ∀ n, r (n + 1) = ∏ i in Finset.range n, r (i + 1) + 1)
  (h3 : ∀ i, 0 < a i)
  (h4 : ∑ k in Finset.range n, 1 / (a k) < 1)
  : ∑ k in Finset.range n, 1 / (a k) ≤ ∑ k in Finset.range n, 1 / (r k) := 
sorry

end sum_inverse_r_le_sum_inverse_a_l195_195385


namespace tangent_line_to_circle_l195_195982

theorem tangent_line_to_circle : 
  ∀ (ρ θ : ℝ), (ρ = 4 * Real.sin θ) → (∃ ρ θ : ℝ, ρ * Real.cos θ = 2) :=
by
  sorry

end tangent_line_to_circle_l195_195982


namespace mean_distance_correct_mn_sum_l195_195829

section mean_distance

-- General setup for a 4x4 grid
-- In this setup, we can calculate the mean distance between any two distinct dots.

def grid_size : ℕ := 4

-- Total number of distinct pairs in the 4x4 grid
def total_pairs : ℕ := (grid_size * grid_size * (grid_size * grid_size - 1)) / 2 -- 16 choose 2

-- Defining the count of pairs for each distance from 1 to 6 based on given conditions
def count_pairs_for_distance (dist : ℕ) : ℕ :=
  match dist with
  | 1 => 24
  | 2 => 36
  | 3 => 36
  | 4 => 36
  | 5 => 36
  | 6 => 36
  | _ => 0

-- Summing the total weighted distances
def total_distance : ℕ := List.sum (List.map (λ dist, dist * count_pairs_for_distance dist) [1, 2, 3, 4, 5, 6])

-- Mean distance calculation
def mean_distance : ℚ := total_distance / total_pairs

-- Simplified mean distance fraction
def mean_distance_frac : ℚ := 31 / 5

-- Proof statement
theorem mean_distance_correct : mean_distance = mean_distance_frac := by
  sorry

-- Final statement to prove m + n
theorem mn_sum : 31 + 5 = 36 := by
  rfl

end mean_distance

end mean_distance_correct_mn_sum_l195_195829


namespace train_speed_faster_l195_195112

-- The Lean statement of the problem
theorem train_speed_faster (Vs : ℝ) (L : ℝ) (T : ℝ) (Vf : ℝ) :
  Vs = 36 ∧ L = 340 ∧ T = 17 ∧ (Vf - Vs) * (5 / 18) = L / T → Vf = 108 :=
by 
  intros 
  sorry

end train_speed_faster_l195_195112


namespace coefficient_sum_of_squares_is_23456_l195_195726

theorem coefficient_sum_of_squares_is_23456 
  (p q r s t u : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := 
by
  sorry

end coefficient_sum_of_squares_is_23456_l195_195726


namespace probability_of_drawing_odd_ball_l195_195356

theorem probability_of_drawing_odd_ball (balls : Finset ℕ) (h : balls = {1, 2, 3}) :
  (Finset.card (balls.filter (λ n, n % 2 = 1)) : ℚ) / (Finset.card balls) = 2/3 :=
by
  -- Skipping the proof as per the instructions
  sorry

end probability_of_drawing_odd_ball_l195_195356


namespace problem_statement_l195_195339

variable (a b c d x : ℕ)

theorem problem_statement
  (h1 : a + b = x)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : a + d = 6) :
  x = 12 :=
by
  sorry

end problem_statement_l195_195339


namespace base_polygon_is_regular_l195_195440

-- Defining the problem conditions
variable (n : ℕ) (P : Type)

-- Assume n is the number of sides and is odd
def odd_sided_polygon : Prop := n % 2 = 1

-- Assume edges of the base polygon are of equal length
def equal_length_edges (positions : Fin n → P) : Prop :=
∀ i j : Fin n, (positions i = positions j) → (positions i ≠ positions j)

-- Assume angles between adjacent faces are equal
def equal_adjacent_angles (adj_face_angle : Fin n → ℝ) : Prop :=
∀ i j : Fin n, adj_face_angle i = adj_face_angle j

-- State the theorem
theorem base_polygon_is_regular
  (n : ℕ) (positions : Fin n → P) (adj_face_angle : Fin n → ℝ)
  (h1 : odd_sided_polygon n) (h2 : equal_length_edges positions) (h3 : equal_adjacent_angles adj_face_angle) :
  ∃ r : ℝ, ∀ i j : Fin n, ∥positions i - r∥ = ∥positions j - r∥ ∧ adj_face_angle i = adj_face_angle j :=
sorry

end base_polygon_is_regular_l195_195440


namespace least_product_of_distinct_primes_gt_30_l195_195494

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l195_195494


namespace binom_inequality_l195_195419

theorem binom_inequality {n : ℕ} (hn : n > 0) {x : ℝ} (hx : x > 0) :
  (∑ k in range (n + 1), if even k then (nat.choose (2 * n) k / (x + k) : ℝ) else 0)
  > (∑ k in range (n + 1), if odd k then (nat.choose (2 * n) k / (x + k) : ℝ) else 0) := 
sorry

end binom_inequality_l195_195419


namespace volume_of_T_is_12_l195_195624

noncomputable def volume_of_T : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| <= 1.5 ∧ |p.1| + |p.3| <= 1.5 ∧ |p.2| + |p.3| <= 1 }
  calc_volume region sorry

theorem volume_of_T_is_12 : volume_of_T = 12 := by
  sorry

end volume_of_T_is_12_l195_195624


namespace total_pairs_sold_l195_195599

theorem total_pairs_sold (H S : ℕ) 
    (soft_lens_cost hard_lens_cost : ℕ)
    (total_sales : ℕ)
    (h1 : soft_lens_cost = 150)
    (h2 : hard_lens_cost = 85)
    (h3 : S = H + 5)
    (h4 : soft_lens_cost * S + hard_lens_cost * H = total_sales)
    (h5 : total_sales = 1455) :
    H + S = 11 := 
  sorry

end total_pairs_sold_l195_195599


namespace calculate_total_loss_l195_195805

def original_cost_paintings (num_paintings : ℕ) (cost_per_painting : ℕ) : ℕ :=
  num_paintings * cost_per_painting

def original_cost_wooden_toys (num_toys : ℕ) (cost_per_toy : ℕ) : ℕ :=
  num_toys * cost_per_toy

def total_original_cost (cost_paintings : ℕ) (cost_toys : ℕ) : ℕ :=
  cost_paintings + cost_toys

def selling_price_painting (original_price : ℕ) (discount_percent : ℕ) : ℕ :=
  original_price - (original_price * discount_percent / 100)

def selling_price_toy (original_price : ℕ) (discount_percent : ℕ) : ℕ :=
  original_price - (original_price * discount_percent / 100)

def total_selling_price (num_paintings : ℕ) (selling_price_painting : ℕ) (num_toys : ℕ) (selling_price_toy : ℕ) : ℕ :=
  (num_paintings * selling_price_painting) + (num_toys * selling_price_toy)

def total_loss (original_cost : ℕ) (selling_cost : ℕ) : ℕ :=
  original_cost - selling_cost

theorem calculate_total_loss :
  let num_paintings := 10
  let cost_per_painting := 40
  let num_toys := 8
  let cost_per_toy := 20
  let discount_percent_painting := 10
  let discount_percent_toy := 15

  original_cost_paintings num_paintings cost_per_painting + original_cost_wooden_toys num_toys cost_per_toy 
  = 560 →
  
  total_original_cost 400 160 = 560 →

  selling_price_painting cost_per_painting discount_percent_painting = 36 →
  selling_price_toy cost_per_toy discount_percent_toy = 17 →
  
  total_selling_price num_paintings 36 num_toys 17 
  = 496 →
  
  total_loss 560 496 = 64 :=
by
  intros
  simp_all
  sorry

end calculate_total_loss_l195_195805


namespace copies_per_person_l195_195423

-- Definitions derived from the conditions
def pages_per_contract : ℕ := 20
def total_pages_copied : ℕ := 360
def number_of_people : ℕ := 9

-- Theorem stating the result based on the conditions
theorem copies_per_person : (total_pages_copied / pages_per_contract) / number_of_people = 2 := by
  sorry

end copies_per_person_l195_195423


namespace trigonometric_inequality_l195_195398

theorem trigonometric_inequality (x : ℝ) (hx1 : 0 < x) (hx2 : x < real.pi / 2) :
  real.cos x ^ 2 * real.cot x + real.sin x ^ 2 * real.tan x ≥ 1 := 
sorry

end trigonometric_inequality_l195_195398


namespace least_product_of_primes_gt_30_l195_195498

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l195_195498


namespace least_product_of_primes_gt_30_l195_195495

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l195_195495


namespace triangle_right_angle_BC_AB_area_150_incircle_BL_length_l195_195880

theorem triangle_right_angle_BC_AB_area_150_incircle_BL_length
  (A B C M L : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq M] [decidable_eq L]
  (angle_A_90 : ∠A = 90)
  (BC_25 : segment_length B C = 25)
  (area_ABC_150 : area (triangle A B C) = 150)
  (AB_greater_AC : length (segment A B) > length (segment A C))
  (tangent_M_AC : is_tangent M (circle_incenter (triangle A B C)))
  (BM_intersects_L : intersection (line B M) (circle_incenter (triangle A B C)) = L) :
  length (segment B L) = 45 * sqrt 17 / 17 :=
by sorry

end triangle_right_angle_BC_AB_area_150_incircle_BL_length_l195_195880


namespace sandra_stickers_l195_195827

theorem sandra_stickers :
  ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ (N % 11 = 1) ∧ N = 166 :=
by {
  sorry
}

end sandra_stickers_l195_195827


namespace travel_time_K_l195_195550

/-
Given that:
1. K's speed is x miles per hour.
2. M's speed is x - 1 miles per hour.
3. K takes 1 hour less than M to travel 60 miles (i.e., 60/x hours).
Prove that K's time to travel 60 miles is 6 hours.
-/
theorem travel_time_K (x : ℝ)
  (h1 : x > 0)
  (h2 : x ≠ 1)
  (h3 : 60 / (x - 1) - 60 / x = 1) :
  60 / x = 6 :=
sorry

end travel_time_K_l195_195550


namespace total_viewing_time_l195_195605

theorem total_viewing_time :
  let original_times := [4, 6, 7, 5, 9]
  let new_species_times := [3, 7, 8, 10]
  let total_breaks := 8
  let break_time_per_animal := 2
  let total_time := (original_times.sum + new_species_times.sum) + (total_breaks * break_time_per_animal)
  total_time = 75 :=
by
  sorry

end total_viewing_time_l195_195605


namespace locus_of_centers_l195_195840

-- Definitions for the given circles
def C1 : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 1}
def C2 : set (ℝ × ℝ) := {p | (p.1 - 2) ^ 2 + p.2 ^ 2 = 16}

-- The locus of centers (a, b)
def locus (a b : ℝ) : Prop := 84 * a ^ 2 + 100 * b ^ 2 - 168 * a - 441 = 0

-- Proof goal: given that (a, b) satisfies the conditions of tangency to C1 and C2, it must lie on the locus
theorem locus_of_centers (a b r : ℝ) :
  {p : ℝ × ℝ | (p.1 = a ∧ p.2 = b) ∧ 
               (∀ p ∈ C1, ∃ r, (a - p.1) ^ 2 + (b - p.2) ^ 2 = (r + 1) ^ 2) ∧
               (∀ p ∈ C2, ∃ r, (p.1 - a) ^ 2 + (p.2 - b) ^ 2 = (4 - r) ^ 2) } -> locus a b :=
by
  sorry

end locus_of_centers_l195_195840


namespace balloon_permutations_l195_195223

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l195_195223


namespace least_prime_product_l195_195484
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l195_195484


namespace sum_of_digits_t_is_9_l195_195410

noncomputable def lily_jumps (n : ℕ) : ℕ := (n + 2) / 3  -- Number of jumps if Lily leaps 3 steps a time
noncomputable def felix_jumps (n : ℕ) : ℕ := (n + 3) / 4  -- Number of jumps if Felix jumps 4 steps a time

theorem sum_of_digits_t_is_9 
  (h : ∀ (n : ℕ), lily_jumps n = felix_jumps n + 15) :
  let t := ∑ n in (finset.range 1000), if lily_jumps n - felix_jumps n = 15 then ∑ digits n in (finset.range 10), digits n else 0 in
  t.digits.sum = 9 :=
by
  sorry

end sum_of_digits_t_is_9_l195_195410


namespace Zoe_house_number_l195_195776

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits (n : ℕ) : ℕ := (Nat.digitSum n)
def has_digit_7 (n : ℕ) : Prop := 7 ∈ Nat.digits 10 n

theorem Zoe_house_number : ∃ n : ℕ, is_two_digit n ∧ (n.prime ∨ (sum_of_digits n).even ∨ n % 9 = 0 ∨ has_digit_7 n) ∧
  (Nat.countb (λ cond : Prop, cond) 
    [n.prime, (sum_of_digits n).even, n % 9 = 0, has_digit_7 n] = 3) ∧ n = 72 :=
by
  sorry

end Zoe_house_number_l195_195776


namespace distance_they_both_run_l195_195918

theorem distance_they_both_run
  (D : ℝ)
  (A_time : D / 28 = A_speed)
  (B_time : D / 32 = B_speed)
  (A_beats_B : A_speed * 28 = B_speed * 28 + 16) :
  D = 128 := 
sorry

end distance_they_both_run_l195_195918


namespace ratio_a_to_c_l195_195345

variable (a b c : ℚ)

theorem ratio_a_to_c (h1 : a / b = 7 / 3) (h2 : b / c = 1 / 5) : a / c = 7 / 15 := 
sorry

end ratio_a_to_c_l195_195345


namespace linear_function_m_l195_195910

theorem linear_function_m (m : ℤ) (x : ℤ) (h : (m - 1) * x^abs m + 3 = (m - 1) * x + 3) : m = -1 := by
  sorry

end linear_function_m_l195_195910


namespace inequality_positive_reals_l195_195045

open Real

variable (x y : ℝ)

theorem inequality_positive_reals (hx : 0 < x) (hy : 0 < y) : x^2 + (8 / (x * y)) + y^2 ≥ 8 :=
by
  sorry

end inequality_positive_reals_l195_195045


namespace sum_first_7_terms_l195_195361

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Given the condition
theorem sum_first_7_terms {a d : ℝ} : 
  (∃ a2 a3 a7 : ℝ, a2 = arithmetic_seq a d 2 ∧ a3 = arithmetic_seq a d 3 ∧ a7 = arithmetic_seq a d 7 ∧ a2 + a3 + a7 = 12) →
  (a2 : ℝ) → (a3 : ℝ) → (a4 : ℝ) → (a5 : ℝ) → (a6 : ℝ) → (a7 : ℝ),
    a4 = arithmetic_seq a d 4 →
    a2 = arithmetic_seq a d 2 →
    a3 = arithmetic_seq a d 3 →
    a5 = arithmetic_seq a d 5 →
    a1 = arithmetic_seq a d 1 →
    a6 = arithmetic_seq a d 6 →
    a7 = arithmetic_seq a d 7 →
  a2 + a3 + a4 + a5 + a6 + a7 / 7 = 28 :=
begin
  sorry
end

end sum_first_7_terms_l195_195361


namespace cafeteria_can_make_7_pies_l195_195549

theorem cafeteria_can_make_7_pies (initial_apples handed_out apples_per_pie : ℕ)
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  ((initial_apples - handed_out) / apples_per_pie) = 7 := 
by
  sorry

end cafeteria_can_make_7_pies_l195_195549


namespace parallel_lines_l195_195707

theorem parallel_lines (a : ℝ) : 
  let line1 := (λ x : ℝ, (a-1) * x - 2),
      line2 := (λ (x y : ℝ), 3 * x + (a+3) * y - 1 = 0)
  in (∀ x : ℝ, ∃ y : ℝ, line2 x y → line1 x = y) → (a = 0 ∨ a = -2) :=
by 
  intros,
  sorry

end parallel_lines_l195_195707


namespace max_gcd_consecutive_terms_l195_195522

def b (n : ℕ) : ℕ := n! + n^2

theorem max_gcd_consecutive_terms : ∃ k, k = 2 ∧ ∀ n, Nat.gcd (b n) (b (n + 1)) ≤ k := by
  sorry

end max_gcd_consecutive_terms_l195_195522


namespace least_product_of_primes_over_30_l195_195479

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l195_195479


namespace perimeter_of_triangle_lines_l195_195100

theorem perimeter_of_triangle_lines (PQ QR PR : ℝ) (mP mQ mR : ℝ)
  (hPQ : PQ = 160) (hQR : QR = 300) (hPR : PR = 240)
  (hmP : mP = 75) (hmQ : mQ = 60) (hmR : mR = 20) :
  mP + mQ + mR = 155 :=
by
  rw [hmP, hmQ, hmR]
  exact rfl
  sorry

end perimeter_of_triangle_lines_l195_195100


namespace card_sequence_probability_l195_195093

-- Defining the mathematical concepts
def deck := {cards : List (ℕ × String) // cards.length = 52}
def is_five (card : ℕ × String) := card.1 = 5
def is_diamond (card : ℕ × String) := card.2 = "diamonds"
def is_three (card : ℕ × String) := card.1 = 3

-- Defining the probability of a specific sequence of cards
def sequence_probability (seq : List (ℕ × String)) : ℚ :=
  (seq.head?.is_some ∧ is_five (seq.head?.getD (0, "")) ∧
   seq.tail?.head?.is_some ∧ is_diamond (seq.tail?.head?.getD (0, "")) ∧
   seq.tail?.tail?.head?.is_some ∧ is_three (seq.tail?.tail?.head?.getD (0, ""))) →
  1 / 663

-- The main theorem stating that the probability is as calculated
theorem card_sequence_probability (d : deck) (cards_drawn : List (ℕ × String)) :
  cards_drawn.length = 3 →
  sequence_probability cards_drawn :=
by
  intro h_len
  sorry -- Proof goes here

end card_sequence_probability_l195_195093


namespace cosine_of_sum_l195_195708

theorem cosine_of_sum (x : ℝ)
  (h : (sqrt 3 * sin (x / 4)) * (cos (x / 4)) + 1 * (cos (x / 4))^2 = 1) :
  cos (x + π / 3) = 1 / 2 :=
by
  sorry

end cosine_of_sum_l195_195708


namespace smallest_n_l195_195974

def f (n : ℕ) : ℕ :=
  let ⟨k, l⟩ := n.div2; -- Retrieving k and l such that n = 2^k * (2l + 1)
  k^2 + k + 1

def S (n : ℕ) : ℕ :=
  (List.range n).sum (λ i, f (i + 1))

theorem smallest_n (n : ℕ) (h : S n ≥ 123456) : ∃ n, S n ≥ 123456 :=
sorry

end smallest_n_l195_195974


namespace irrational_between_three_and_four_l195_195534

theorem irrational_between_three_and_four : ∃ (x : ℝ), irrational x ∧ 3 < x ∧ x < 4 :=
by
  use Real.pi
  split
  . sorry   -- Proof that π is irrational
  split
  . linarith [Real.pi_gt_three]  -- Proof that 3 < π
  . linarith [Real.pi_lt_four]   -- Proof that π < 4

end irrational_between_three_and_four_l195_195534


namespace first_point_x_coord_l195_195757

variables (m n : ℝ)

theorem first_point_x_coord (h1 : m = 2 * n + 5) (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 :=
by 
  sorry

end first_point_x_coord_l195_195757


namespace least_prime_product_l195_195483
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l195_195483


namespace simplify_expression_l195_195828

theorem simplify_expression (x y : ℝ) : (5 - 4 * y) - (6 + 5 * y - 2 * x) = -1 - 9 * y + 2 * x := by
  sorry

end simplify_expression_l195_195828


namespace part1_min_value_a_1_part2_extreme_points_l195_195315

-- Define the function f for the given problem
def f (a x : ℝ) : ℝ := (x + 2) / Real.exp x + a * x - 2

-- Problem 1: Prove that for a = 1, the minimum value of f(x) on the interval [0, ∞) is 0
theorem part1_min_value_a_1 : ∀ x ≥ 0, f 1 x ≥ 0 :=
sorry

-- Problem 2: If f has two extreme points x1 and x2 on ℝ, prove that e^x2 - e^x1 > 2/a - 2
theorem part2_extreme_points (a : ℝ) (ha : 0 < a ∧ a < 1) (x1 x2 : ℝ) (hx1_lt : -1 < x1 ∧ x1 < 0) (hx2_gt : 0 < x2)
  (hx1_x2 : x1 < x2) (h_fx1_fx2_zero : (∃ x1 x2 : ℝ, f a x1 = 0 ∧ f a x2 = 0)) :
  Real.exp x2 - Real.exp x1 > 2 / a - 2 :=
sorry

end part1_min_value_a_1_part2_extreme_points_l195_195315


namespace max_elements_in_set_l195_195547

theorem max_elements_in_set (S : Finset ℕ) :
  (∀ a ∈ S, a > 0 ∧ a ≤ 100) →
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd a c = 1 ∧ Nat.gcd b c = 1) →
  (∀ a b ∈ S, a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ Nat.gcd a d > 1 ∧ Nat.gcd b d > 1) →
  S.card ≤ 72 :=
by sorry

end max_elements_in_set_l195_195547


namespace proof_trajectory_equation_proof_area_of_triangle_l195_195750

noncomputable def trajectory_equation : Prop :=
∀ (P Q: ℝ × ℝ) (L : ℝ → ℝ),
  Q = (1, 2) →
  (P.1 ≠ 0 ∧ P.2 ≠ 0 ∧ P.2 ≠ 2) →
  (∃ k_OP k_OQ k_PQ : ℝ, k_OP = P.2 / P.1 ∧ k_OQ = 2 ∧ k_PQ = (P.2 - 2) / (P.1 - 1) ∧
    1 / k_OP + 1 / k_OQ = 1 / k_PQ) →
  P.2^2 = 4 * P.1

noncomputable def area_of_triangle : Prop :=
∀ (A B F : ℝ × ℝ) (O: ℝ × ℝ)
  (L : ℝ → ℝ),
  O = (0, 0) →
  F = (1, 0) →
  (∀ x, L x = Math.sqrt 3 * (x - 1)) →
  (∀ x1 x2 y1 y2 : ℝ, 
    (y1^2 = 4 * x1 ∧ L x1 = y1) ∧ 
    (y2^2 = 4 * x2 ∧ L x2 = y2) →
    A = (x1, y1) ∧ B = (x2, y2)) →
  ∃ (S : ℝ), S = 1 / 2 * 1 * Math.sqrt ((A.2 + B.2)^2 / 3 + 16) ∧ S = 4 * Math.sqrt 3 / 3

-- Proof of the properties (not required as per the instruction)
theorem proof_trajectory_equation : trajectory_equation := by
  sorry

theorem proof_area_of_triangle : area_of_triangle := by
  sorry

end proof_trajectory_equation_proof_area_of_triangle_l195_195750


namespace part1_min_value_function_part2_prove_inequality_l195_195312

-- Lean translation for Part (1)
theorem part1_min_value_function (a : ℝ) (x : ℝ) (h_a : a = 1) (h_x : 0 ≤ x) :
  let f (x : ℝ) := (x + 2)/Real.exp x + x - 2 in
  f 0 = 0 := sorry

-- Lean translation for Part (2)
theorem part2_prove_inequality (a x1 x2 : ℝ) (h_extreme : f'(x1) = 0 ∧ f'(x2) = 0) (h_order : x1 < x2) (h_a : 0 < a ∧ a < 1) :
  let f' (x : ℝ) := a - (x + 1)/Real.exp x in
  Real.exp x2 - Real.exp x1 > (2/a) - 2 := sorry

end part1_min_value_function_part2_prove_inequality_l195_195312


namespace original_number_is_144_l195_195175

theorem original_number_is_144 (A B C : ℕ) (A_digit : A < 10) (B_digit : B < 10) (C_digit : C < 10)
  (h1 : 100 * A + 10 * B + B = 144)
  (h2 : A * B * B = 10 * A + C)
  (h3 : (10 * A + C) % 10 = C) : 100 * A + 10 * B + B = 144 := 
sorry

end original_number_is_144_l195_195175


namespace smallest_hiding_number_l195_195008

/-- Define the concept of "hides" -/
def hides (A B : ℕ) : Prop :=
  ∃ (remove : ℕ → ℕ), remove A = B

/-- The smallest natural number that hides all numbers from 2000 to 2021 is 20012013456789 -/
theorem smallest_hiding_number : hides 20012013456789 2000 ∧ hides 20012013456789 2001 ∧ hides 20012013456789 2002 ∧
    hides 20012013456789 2003 ∧ hides 20012013456789 2004 ∧ hides 20012013456789 2005 ∧ hides 20012013456789 2006 ∧
    hides 20012013456789 2007 ∧ hides 20012013456789 2008 ∧ hides 20012013456789 2009 ∧ hides 20012013456789 2010 ∧
    hides 20012013456789 2011 ∧ hides 20012013456789 2012 ∧ hides 20012013456789 2013 ∧ hides 20012013456789 2014 ∧
    hides 20012013456789 2015 ∧ hides 20012013456789 2016 ∧ hides 20012013456789 2017 ∧ hides 20012013456789 2018 ∧
    hides 20012013456789 2019 ∧ hides 20012013456789 2020 ∧ hides 20012013456789 2021 :=
by
  sorry

end smallest_hiding_number_l195_195008


namespace prime_cube_difference_l195_195059

theorem prime_cube_difference (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (eqn : p^3 - q^3 = 11 * r) : 
  (p = 13 ∧ q = 2 ∧ r = 199) :=
sorry

end prime_cube_difference_l195_195059


namespace total_number_of_marbles_is_1050_l195_195106

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l195_195106


namespace trajectory_of_C_l195_195970

theorem trajectory_of_C (x y : ℝ) : 
  (∃ θ : ℝ, (x, y) = (3 * Real.cos θ + Real.sqrt 5, 2 * Real.sin θ)) →
  \frac{(x - Real.sqrt 5)^2}{9} + \frac{y^2}{4} = 1 := sorry

end trajectory_of_C_l195_195970


namespace sqrt_product_l195_195962

theorem sqrt_product (a b : ℝ) (ha : a = 20) (hb : b = 1/5) : Real.sqrt a * Real.sqrt b = 2 := 
by
  sorry

end sqrt_product_l195_195962


namespace balloon_permutations_l195_195208

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l195_195208


namespace Dan_gave_Sara_limes_l195_195625

theorem Dan_gave_Sara_limes : 
  ∀ (original_limes now_limes given_limes : ℕ),
  original_limes = 9 →
  now_limes = 5 →
  given_limes = original_limes - now_limes →
  given_limes = 4 :=
by
  intros original_limes now_limes given_limes h1 h2 h3
  sorry

end Dan_gave_Sara_limes_l195_195625


namespace mr_bird_on_time_exactly_l195_195036

noncomputable def required_speed (t d : ℝ) : ℝ :=
  let t_late := t + 5 / 60
  let t_early := t - 5 / 60
  let d_late := 30 * t_late
  let d_early := 50 * t_early
  d_late / t

theorem mr_bird_on_time_exactly :
  ∃ t d : ℝ, (d = 30 * (t + 5 / 60)) ∧ (d = 50 * (t - 5 / 60)) ∧ required_speed t d = 37.5 :=
begin
  sorry
end

end mr_bird_on_time_exactly_l195_195036


namespace line_m_equation_l195_195632

-- Definitions of the given conditions
variable {Q Q' Q'' : ℝ × ℝ} {m : ℝ → ℝ → Prop}
def point_Q : ℝ × ℝ := (3, -2)
def point_Q'' : ℝ × ℝ := (-2, 3)
def line_ell : ℝ → ℝ → Prop := fun x y ↦ 2 * x - 5 * y = 0
def line_m (m : ℝ → ℝ → Prop) : Prop := ∀ x y, m x y ↦ 5 * x + 2 * y = 0

-- The main theorem statement
theorem line_m_equation :
  (∃ Q Q' Q'', Q = (3, -2) ∧ Q'' = (-2, 3) ∧ Q' = reflection m Q) → 
  (∃ ell : ℝ → ℝ → Prop, ell = line_ell ∧ ∀ Q', reflection ell Q' = Q'') → 
  (∃ m : ℝ → ℝ → Prop, ∀ x y, m x y ↦ 5 * x + 2 * y = 0) := sorry

end line_m_equation_l195_195632


namespace count_three_digit_numbers_l195_195323

def digits : Finset ℕ := {2, 3, 4, 5, 5, 6}

def three_digit_numbers (s : Finset ℕ) : Finset (ℕ × ℕ × ℕ) :=
  (s.product s).product s |>.filter (λ t, t.1.1 != t.1.2 ∧ t.1.2 != t.2 ∧ t.1.1 != t.2) ++
  ([(5, 5, 2), (5, 5, 3), (5, 5, 4), (5, 5, 6), (5, 2, 5), (5, 3, 5), (5, 4, 5), (5, 6, 5),
   (2, 5, 5), (3, 5, 5), (4, 5, 5), (6, 5, 5)] : Finset (ℕ × ℕ × ℕ))

theorem count_three_digit_numbers : 
  (three_digit_numbers digits).card = 72 := 
sorry

end count_three_digit_numbers_l195_195323


namespace pentagon_area_ratio_l195_195475

variable {A B C D E : Type} [regular_pentagon A B C D E]

noncomputable def internal_angle_pentagon : ℝ := 108 * (π / 180)

axiom area_ratio :
  ∃ (AC AE : line_segment), 
    originates_from A AC ∧ 
    originates_from A AE ∧ 
    intersects C AC ∧
    intersects E AE ∧ 
    ratio_of_areas ACE ABC ADE = cos (36 * (π / 180))

theorem pentagon_area_ratio :
  ∃ (ABC ACE ADE : triangle), 
    create_from_pentagon A B C D E ABC ∧ 
    create_from_pentagon A C E D B ACE ∧ 
    create_from_pentagon A D E B C ADE ∧ 
    ratio_of_areas ACE ABC ADE = cos (36 * (π / 180)) :=
  by
    intro A B C D E ABC ACE ADE
    cases area_ratio with AC h
    sorry

end pentagon_area_ratio_l195_195475


namespace triangle_median_XZ_YZ_l195_195365

theorem triangle_median_XZ_YZ {X Y Z M: Type} 
  (XY : ℝ) (XM : ℝ) (MY MZ : ℝ) (XZ_YZ_squared : ℝ):
  XY = 10 → 
  XM = 6 → 
  MY = 5 → 
  MZ = 5 → 
  XZ_YZ_squared = 122 :=
by
  assume XY_eq XY_val : XY = 10,
  assume XM_eq XM_val : XM = 6,
  assume MY_eq MY_val : MY = 5,
  assume MZ_eq MZ_val : MZ = 5,
  sorry

end triangle_median_XZ_YZ_l195_195365


namespace find_f_neg_5_l195_195450

noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ (Set.Ioo 0 3) then 2^x 
  else if x ∈ (fun y => -3 < y ∧ y < 0) then -2^(-x)
  else if x ∈ (fun y => 3 < y ∧ y < ∞) then 2^(x-3)
  else if x ∈ (fun y => -∞ < y ∧ y < -3) then -2^((-(x + 3)):ℝ)
  else 0  -- kind of handling outside conditions roughly, not provided exact behavior

axiom odd_property : ∀ x : ℝ, f (-x) = -f x
axiom symmetry_property : ∀ x : ℝ, 0 < x ∧ x < 3 → f (3 + x) = f (3 - x)
axiom condition_property : ∀ x : ℝ, 0 < x ∧ x < 3 → f x = 2^x

theorem find_f_neg_5 : f (-5) = -2 := by
  sorry

end find_f_neg_5_l195_195450


namespace area_increase_is_50_l195_195576

def length := 13
def width := 10
def length_new := length + 2
def width_new := width + 2
def area_original := length * width
def area_new := length_new * width_new
def area_increase := area_new - area_original

theorem area_increase_is_50 : area_increase = 50 :=
by
  -- Here we will include the steps to prove the theorem if required
  sorry

end area_increase_is_50_l195_195576


namespace integral_sqrt_4_minus_x_sq_eq_pi_l195_195684

variable {a : ℝ} (h : a = 2)

theorem integral_sqrt_4_minus_x_sq_eq_pi (ha : a = 2) :
    ∫ x in 0..a, sqrt (4 - x ^ 2) = π :=
by
  rw ha
  calc
    ∫ x in 0..2, sqrt (4 - x ^ 2) = π
  sorry

end integral_sqrt_4_minus_x_sq_eq_pi_l195_195684


namespace sum_of_coordinates_of_B_is_zero_l195_195358

structure Point where
  x : Int
  y : Int

def translation_to_right (P : Point) (n : Int) : Point :=
  { x := P.x + n, y := P.y }

def translation_down (P : Point) (n : Int) : Point :=
  { x := P.x, y := P.y - n }

def A : Point := { x := -1, y := 2 }

def B : Point := translation_down (translation_to_right A 1) 2

theorem sum_of_coordinates_of_B_is_zero :
  B.x + B.y = 0 := by
  sorry

end sum_of_coordinates_of_B_is_zero_l195_195358


namespace factorial_expression_simplification_l195_195608

theorem factorial_expression_simplification : (3 * (Nat.factorial 5) + 15 * (Nat.factorial 4)) / (Nat.factorial 6) = 1 := by
  sorry

end factorial_expression_simplification_l195_195608


namespace max_side_length_l195_195553

theorem max_side_length (a b c : ℕ) (h : a + b + c = 30) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_order : a ≤ b ∧ b ≤ c) (h_triangle_ineq : a + b > c) : c ≤ 14 := 
sorry

end max_side_length_l195_195553


namespace f_bijective_l195_195887

-- Define the operation ⊕ on {0, 1}
def op (a b : ℕ) : ℕ :=
  match a, b with
  | 0, 0 => 0
  | 0, 1 => 1
  | 1, 0 => 1
  | 1, 1 => 0
  | _, _ => 0 -- Since {0, 1} is the domain, other cases are not necessary

-- Define the binary representation
def bin_op (a b : ℕ) : ℕ := 
  let rec aux a b c i := match a, b with
    | 0, 0 => c
    | _, _ => aux (a / 2) (b / 2) ((c + (op (a % 2) (b % 2)) * 2^i)) (i + 1)
  aux a b 0 0

-- Define f(n)
def f (n : ℕ) : ℕ :=
  bin_op n (n / 2)

-- Prove f is a bijection on ℕ
theorem f_bijective : Function.bijective f :=
by {
  -- Proof needs to be filled in
  sorry
}

end f_bijective_l195_195887


namespace sum_of_squares_inequality_l195_195413

theorem sum_of_squares_inequality (n : ℕ) (hn : 2 ≤ n) :
  (1 : ℝ) + ∑ k in finset.range (n-1) + 1, (1 / (k + 2) ^ 2 : ℝ) < (2 * n - 1) / n :=
sorry

end sum_of_squares_inequality_l195_195413


namespace evaluate_expression_l195_195263

variable {x y : ℝ}

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y ^ 2) :
  (x - 1 / x ^ 2) * (y + 2 / y) = 2 * x ^ (5 / 2) - 1 / x := 
by
  sorry

end evaluate_expression_l195_195263


namespace least_prime_product_l195_195485
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l195_195485


namespace min_value_fraction_l195_195386

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end min_value_fraction_l195_195386


namespace no_positive_integer_solutions_for_f_l195_195669

def f (x : ℕ) : ℕ := x^2 + x

theorem no_positive_integer_solutions_for_f :
  ∀ (a b : ℕ), 0 < a → 0 < b → 4 * f a ≠ f b :=
begin
  -- proof goes here
  intros a b ha hb,
  sorry
end

end no_positive_integer_solutions_for_f_l195_195669


namespace area_of_DEF_l195_195756

noncomputable def tetrahedron_A := (0, 0, 0)
noncomputable def tetrahedron_B := (7, 0, 0)
noncomputable def tetrahedron_C := (7/2, 7 * Real.sqrt(3) / 2, 0)
noncomputable def tetrahedron_D := (7/2, 7 * Real.sqrt(3) / 6, 7 * Real.sqrt(6) / 3)
noncomputable def E := (3, 0, 0)
noncomputable def F := (2, 2 * Real.sqrt(3), 0)

theorem area_of_DEF :
  (let D := tetrahedron_D;
       H := (5/2, Real.sqrt(3), 0);
       DH := Real.sqrt((7/2 - 5/2)^2 + (7 * Real.sqrt(3)/6 - Real.sqrt(3))^2 + (7 * Real.sqrt(6)/3)^2))
  ∃ D E F : (ℝ × ℝ × ℝ), 
  D = tetrahedron_D ∧ E = E ∧ F = F ∧
  Real.sqrt((2 - 3) ^ 2 + (2 * Real.sqrt(3) - 0) ^ 2 + (0 - 0) ^ 2) = 4 ∧
  (1 / 2) * 4 * DH = 2 * Real.sqrt(33) := by
  sorry

end area_of_DEF_l195_195756


namespace repeating_decimal_to_fraction_l195_195519

noncomputable def x : ℚ := 0.6 + 41 / 990  

theorem repeating_decimal_to_fraction (h : x = 0.6 + 41 / 990) : x = 127 / 198 :=
by sorry

end repeating_decimal_to_fraction_l195_195519


namespace comic_stack_permutations_l195_195808

theorem comic_stack_permutations :
  let spiderman_factorial := Nat.factorial 7
      archie_factorial := Nat.factorial 4
      garfield_factorial := Nat.factorial 5
      batman_factorial := Nat.factorial 3
      group_permutations := Nat.factorial 3
  in (spiderman_factorial * archie_factorial * garfield_factorial * batman_factorial) * group_permutations = 55_085_760 :=
by
  let spiderman_factorial := Nat.factorial 7
  let archie_factorial := Nat.factorial 4
  let garfield_factorial := Nat.factorial 5
  let batman_factorial := Nat.factorial 3
  let group_permutations := Nat.factorial 3
  have h1 : spiderman_factorial = 5040 := sorry
  have h2 : archie_factorial = 24 := sorry
  have h3 : garfield_factorial = 120 := sorry
  have h4 : batman_factorial = 6 := sorry
  have h5 : group_permutations = 6 := sorry
  calc (spiderman_factorial * archie_factorial * garfield_factorial * batman_factorial) * group_permutations
      = (5040 * 24 * 120 * 6) * 6 : by rw [h1, h2, h3, h4, h5]
  ... = 55_085_760 : sorry

end comic_stack_permutations_l195_195808


namespace sequence_1990_l195_195536

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 4 else S (n-1) + 2*n + 1
where
  S (m : ℕ) : ℕ :=
    if m = 0 then 0 else ∑ i in (range m).map(nat.succ), sequence i

theorem sequence_1990 :
  sequence 1990 = 11 * 2^1988 - 2 :=
sorry

end sequence_1990_l195_195536


namespace compute_expression_l195_195620

theorem compute_expression : 10 * (3 / 27) * 36 = 40 := 
by 
  sorry

end compute_expression_l195_195620


namespace least_product_of_distinct_primes_greater_than_30_l195_195510

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l195_195510


namespace g_800_eq_768_l195_195845

noncomputable def g : ℕ → ℕ := sorry

axiom g_condition1 (n : ℕ) : g (g n) = 2 * n
axiom g_condition2 (n : ℕ) : g (4 * n + 3) = 4 * n + 1

theorem g_800_eq_768 : g 800 = 768 := by
  sorry

end g_800_eq_768_l195_195845


namespace sum_of_cubes_six_l195_195039

-- Define the generalized pattern as a condition
def sum_of_cubes (n : ℕ) : ℕ := (finset.range (n + 1)).sum (λ k, k^3)

def sum_of_first_n (n : ℕ) : ℕ := (finset.range (n + 1)).sum id

-- State the theorem to prove the correctness of the pattern for the specific case
theorem sum_of_cubes_six : sum_of_cubes 6 = (sum_of_first_n 6)^2 := 
by 
  sorry

end sum_of_cubes_six_l195_195039


namespace middle_digit_base_7_of_reversed_base_9_l195_195929

noncomputable def middle_digit_of_number_base_7 (N : ℕ) : ℕ :=
  let x := (N / 81) % 9  -- Extract the first digit in base-9
  let y := (N / 9) % 9   -- Extract the middle digit in base-9
  let z := N % 9         -- Extract the last digit in base-9
  -- Given condition: 81x + 9y + z = 49z + 7y + x
  let eq1 := 81 * x + 9 * y + z
  let eq2 := 49 * z + 7 * y + x
  let condition := eq1 = eq2 ∧ 0 ≤ y ∧ y < 7 -- y is a digit in base-7
  if condition then y else sorry

theorem middle_digit_base_7_of_reversed_base_9 (N : ℕ) :
  (∃ (x y z : ℕ), x < 9 ∧ y < 9 ∧ z < 9 ∧
  N = 81 * x + 9 * y + z ∧ N = 49 * z + 7 * y + x) → middle_digit_of_number_base_7 N = 0 :=
  by sorry

end middle_digit_base_7_of_reversed_base_9_l195_195929


namespace problem1_problem2_l195_195678

open Real

-- Definitions based on the problem statement
def is_ellipse (a b : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 3 + y^2 = 1)

def is_focus_right (c : ℝ) : Prop :=
  ∀ d : ℝ, abs ((c + 2 * sqrt 2) / sqrt 2) = 3 ↔ c = sqrt 2

-- Problem 1: Equation of the ellipse
theorem problem1 :
  ∃ (a b : ℝ),
    is_ellipse a b ∧ b = 1 ∧ (a^2 = 3) :=
sorry

-- Definitions for problem 2
def is_line_through_point (k : ℝ) : Prop :=
  ∀ x y : ℝ, (y = k * x + 3 / 2) ∧ k² = 2/3

-- Problem 2: Equation of the line, given the point and equal distances
theorem problem2 :
  ∃ k : ℝ,
    is_line_through_point k ∧ (y = k * x + 3 / 2) ∨ (y = -k * x + 3 / 2) :=
sorry

end problem1_problem2_l195_195678


namespace compare_squares_l195_195665

theorem compare_squares (a b : ℝ) : a^2 + b^2 ≥ ab + a + b - 1 :=
by
  sorry

end compare_squares_l195_195665


namespace least_product_of_primes_gt_30_l195_195496

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l195_195496


namespace fraction_of_yellow_balls_l195_195348

theorem fraction_of_yellow_balls
  (total_balls : ℕ)
  (fraction_green : ℚ)
  (fraction_blue : ℚ)
  (number_blue : ℕ)
  (number_white : ℕ)
  (total_balls_eq : total_balls = number_blue * (1 / fraction_blue))
  (fraction_green_eq : fraction_green = 1 / 4)
  (fraction_blue_eq : fraction_blue = 1 / 8)
  (number_white_eq : number_white = 26)
  (number_blue_eq : number_blue = 6) :
  (total_balls - (total_balls * fraction_green + number_blue + number_white)) / total_balls = 1 / 12 :=
by
  sorry

end fraction_of_yellow_balls_l195_195348


namespace area_of_triangle_ABE_is_correct_l195_195739

-- Define the dimensions and angles of the triangle
def AB : ℝ := 3
def AC : ℝ := 4
def BC : ℝ := 5

-- Define the side length of the square BCDE
def side_of_square : ℝ := BC

-- Define the conditions that triangle ABC is a right triangle and BCDE is a square
axiom right_triangle_ABC : ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a = AB ∧ b = AC ∧ c = BC
axiom square_BCDE : ∃ d e : ℝ, d = e ∧ d = BC ∧ e = BC

-- Define the problem of finding the area of triangle ABE
noncomputable def area_triangle_ABE : ℝ :=
  let height := AB * (AB / BC) in
  (1 / 2) * side_of_square * height

-- Theorem to prove that the area of triangle ABE is 9/2 square centimeters
theorem area_of_triangle_ABE_is_correct : area_triangle_ABE = 9 / 2 := by
  unfold area_triangle_ABE
  sorry

end area_of_triangle_ABE_is_correct_l195_195739


namespace domain_of_f_l195_195646

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6

def f (x : ℝ) : ℝ := (x^3 - 3 * x^2 + 6 * x - 8) / q x

theorem domain_of_f :
  {x : ℝ | q x ≠ 0} = {x : ℝ | x ≠ 2 ∧ x ≠ 3} :=
by
  sorry

end domain_of_f_l195_195646


namespace probability_no_more_than_five_girls_between_first_last_boys_l195_195159

theorem probability_no_more_than_five_girls_between_first_last_boys :
    let n := 20
    let g := 11
    let b := 9
    let total_combinations := Nat.choose n b
    let satisfactory_arrangements := Nat.choose 14 9 + 6 * Nat.choose 13 8
    let probability := satisfactory_arrangements.toReal / total_combinations.toReal
in abs (probability - 0.058) < 0.001 := sorry

end probability_no_more_than_five_girls_between_first_last_boys_l195_195159


namespace profit_percentage_correct_l195_195939

def cost_price (C : ℝ) : ℝ := C

def profit_percentage (P : ℝ) : ℝ := P

def selling_price (C P : ℝ) := C * (1 + P / 100)

def selling_price_with_theft (C P : ℝ) := 0.4 * C * (1 + P / 100)

def loss (C : ℝ) := 0.56 * C

def stolen_goods_cost (C : ℝ) := 0.6 * C

theorem profit_percentage_correct (C : ℝ) (hC : C ≠ 0) :
  (selling_price_with_theft C 10 = C - loss C) →
  10 = profit_percentage 10 :=
by
  sorry

end profit_percentage_correct_l195_195939


namespace least_product_of_distinct_primes_gt_30_l195_195492

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l195_195492


namespace find_x_l195_195854

def operation_eur (x y : ℕ) : ℕ := 3 * x * y

theorem find_x (y x : ℕ) (h1 : y = 3) (h2 : operation_eur y (operation_eur x 5) = 540) : x = 4 :=
by
  sorry

end find_x_l195_195854


namespace quadratic_no_real_roots_l195_195461

theorem quadratic_no_real_roots :
  ∀ x : ℝ, ¬(x^2 - 2 * x + 3 = 0) :=
by
  sorry

end quadratic_no_real_roots_l195_195461


namespace card_sequence_probability_l195_195092

-- Defining the mathematical concepts
def deck := {cards : List (ℕ × String) // cards.length = 52}
def is_five (card : ℕ × String) := card.1 = 5
def is_diamond (card : ℕ × String) := card.2 = "diamonds"
def is_three (card : ℕ × String) := card.1 = 3

-- Defining the probability of a specific sequence of cards
def sequence_probability (seq : List (ℕ × String)) : ℚ :=
  (seq.head?.is_some ∧ is_five (seq.head?.getD (0, "")) ∧
   seq.tail?.head?.is_some ∧ is_diamond (seq.tail?.head?.getD (0, "")) ∧
   seq.tail?.tail?.head?.is_some ∧ is_three (seq.tail?.tail?.head?.getD (0, ""))) →
  1 / 663

-- The main theorem stating that the probability is as calculated
theorem card_sequence_probability (d : deck) (cards_drawn : List (ℕ × String)) :
  cards_drawn.length = 3 →
  sequence_probability cards_drawn :=
by
  intro h_len
  sorry -- Proof goes here

end card_sequence_probability_l195_195092


namespace mittens_per_box_l195_195052

theorem mittens_per_box (boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) (h_boxes : boxes = 7) (h_scarves : scarves_per_box = 3) (h_total : total_clothing = 49) : 
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  total_mittens / boxes = 4 :=
by
  sorry

end mittens_per_box_l195_195052


namespace find_number_l195_195652

theorem find_number (x : ℤ) (h : 5 + x * 5 = 15) : x = 2 :=
by
  sorry

end find_number_l195_195652


namespace triangle_area_l195_195979

theorem triangle_area (h b : ℝ) (Hhb : h < b) :
  let P := (0, b)
  let B := (b, 0)
  let D := (h, h)
  let PD := b - h
  let DB := b - h
  1 / 2 * PD * DB = 1 / 2 * (b - h) ^ 2 := by 
  sorry

end triangle_area_l195_195979


namespace no_bounded_sequences_exist_l195_195636

theorem no_bounded_sequences_exist :
  ¬ ∃ (a b : ℕ → ℝ),
    (∀ n, ∃ M, ∀ m, |a m| ≤ M ∧ ∀ k, |b k| ≤ M) ∧
    (∀ (n m : ℕ), n > 0 → m > n → (|a m - a n| > 1 / Real.sqrt n ∨ |b m - b n| > 1 / Real.sqrt n)) := 
sorry

end no_bounded_sequences_exist_l195_195636


namespace find_value_of_f_f_neg1_l195_195073

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -2 / x else 3 + Real.log x / Real.log 2

theorem find_value_of_f_f_neg1 :
  f (f (-1)) = 4 := by
  -- proof omitted
  sorry

end find_value_of_f_f_neg1_l195_195073


namespace average_speed_l195_195846

def initial_odometer_reading : ℕ := 20
def final_odometer_reading : ℕ := 200
def travel_duration : ℕ := 6

theorem average_speed :
  (final_odometer_reading - initial_odometer_reading) / travel_duration = 30 := by
  sorry

end average_speed_l195_195846


namespace numWaysToReplaceStars_2016xxxxx02x_l195_195754

def numWaysToReplaceStars (allowedDigits : List ℕ) (totalDigits requireDivBy3 requiredLastDigitDivBy5 : ℕ) := 
  sorry

theorem numWaysToReplaceStars_2016xxxxx02x : 
  numWaysToReplaceStars [0, 2, 4, 5, 7, 9] 11 3 15 = 864 :=
sorry

end numWaysToReplaceStars_2016xxxxx02x_l195_195754


namespace min_value_expression_l195_195405

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (c : ℝ), c = 216 ∧
    ∀ (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
      ( (a^2 + 3*a + 2) * (b^2 + 3*b + 2) * (c^2 + 3*c + 2) / (a * b * c) ) ≥ 216 := 
sorry

end min_value_expression_l195_195405


namespace tom_total_payment_l195_195876

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end tom_total_payment_l195_195876


namespace tangent_line_circle_intersect_incenter_excenter_l195_195735

theorem tangent_line_circle_intersect_incenter_excenter
  (A B C D E F I Iₐ l : Type)
  [Real A B C D E F I Iₐ l]
  (triangle_ABC : Triangle A B C)
  (h1 : AB > AC)
  (circumcircle_ABC : Circle containing ⟨A, B, C⟩)
  (tangent_l : Line A circumcircle_ABC)
  (circle_A_AC : Circle.centered_at A (AC))
  (intersect_AB : circle_A_AC.intersect AB = D)
  (intersect_line_l : circle_A_AC.intersect_line l = (E, F))
  (incenter : Incenter A B C I)
  (excenter : Excenter A B C Iₐ) :
  passes_through_line D E I ∧ passes_through_line D F Iₐ := sorry

end tangent_line_circle_intersect_incenter_excenter_l195_195735


namespace triangle_inequality_l195_195897

-- Define the lengths of the existing sticks
def a := 4
def b := 7

-- Define the list of potential third sticks
def potential_sticks := [3, 6, 11, 12]

-- Define the triangle inequality conditions
def valid_length (c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Prove that the valid length satisfying these conditions is 6
theorem triangle_inequality : ∃ c ∈ potential_sticks, valid_length c ∧ c = 6 :=
by
  sorry

end triangle_inequality_l195_195897


namespace rectangular_prism_faces_l195_195577

theorem rectangular_prism_faces (n : ℕ) (h1 : ∀ z : ℕ, z > 0 → z^3 = 2 * n^3) 
  (h2 : n > 0) :
  (∃ f : ℕ, f = (1 / 6 : ℚ) * (6 * 2 * n^3) ∧ 
    f = 10 * n^2) ↔ n = 5 := by
sorry

end rectangular_prism_faces_l195_195577


namespace max_reachable_vertices_two_edges_l195_195740

variable (G : Type) [Graph G]
variable [Finite G]
variable [DecidableRel (@Graph.edge G)]
variable (v : Vertex G)

-- Definitions corresponding to the conditions
def directed_graph_condition (G : Type) [Graph G] : Prop :=
  ∀ (u v : Vertex G), u ≠ v → ∃ (d : DirectedEdge), d.source = u ∧ d.target = v

def reachable_in_two_edges (v : Vertex G) (k : ℕ) : Prop :=
  ∀ (v : Vertex G), ∃ (reachable_vertices : Finset (Vertex G)), 
  reachable_vertices.card = k ∧ 
  ∀ (u : Vertex G), u ∈ reachable_vertices ↔ ∃ (w : Vertex G), Graph.edge G v w ∧ Graph.edge G w u

-- The statement of the problem
theorem max_reachable_vertices_two_edges 
  (G : Type) [Graph G] [Finite G] [DecidableRel (@Graph.edge G)]
  (condition : directed_graph_condition G)
  (verts_2013 : Finset (Vertex G)) 
  (hverts : verts_2013.card = 2013) :
  reachable_in_two_edges v 2012 :=
sorry

end max_reachable_vertices_two_edges_l195_195740


namespace transformed_expression_l195_195734

def new_add (a b : ℕ) : ℕ := a * b
def new_sub (a b : ℕ) : ℕ := a + b
def new_mul (a b : ℕ) : ℕ := a / b
def new_div (a b : ℕ) : ℕ := a - b

theorem transformed_expression :
  new_sub 6 (new_add 9 (new_mul 8 (new_div 3 25))) = 5 :=
begin
  -- Calculate transformed expression
  sorry -- proof is omitted
end

end transformed_expression_l195_195734


namespace mark_profit_l195_195026

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l195_195026


namespace g_interval_l195_195791

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem g_interval (a b c : ℝ) (ha : 0 < a) (hb: 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
sorry

end g_interval_l195_195791


namespace exists_infinitely_many_coprime_pairs_l195_195428

theorem exists_infinitely_many_coprime_pairs (n : ℕ) : 
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ Nat.coprime a b ∧ (a = 2 * n - 1) ∧ (b = 2 * n + 1) ∧ (4 * n ∣ a ^ b + b ^ a) :=
by sorry

end exists_infinitely_many_coprime_pairs_l195_195428


namespace age_of_youngest_child_l195_195541

/-- Given that the sum of ages of 5 children born at 3-year intervals is 70, prove the age of the youngest child is 8. -/
theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 := 
  sorry

end age_of_youngest_child_l195_195541


namespace Q_difference_l195_195264

noncomputable def Q (x : ℤ) : ℤ :=
  ∑ k in finset.range 10000, (x / (k + 1))

theorem Q_difference :
  Q(2023) - Q(2022) = 6 := 
by {
  sorry
}

end Q_difference_l195_195264


namespace cos_squared_sum_l195_195963

theorem cos_squared_sum : 
  (∑ k in finset.range 45, cos (k * real.pi / 180) ^ 2 + cos ((90 - k) * real.pi / 180) ^ 2) = 44.5 := 
sorry

end cos_squared_sum_l195_195963


namespace mark_profit_from_selling_magic_card_l195_195032

theorem mark_profit_from_selling_magic_card : 
    ∀ (purchase_price new_value profit : ℕ), 
        purchase_price = 100 ∧ 
        new_value = 3 * purchase_price ∧ 
        profit = new_value - purchase_price 
    → 
        profit = 200 := 
by 
  intros purchase_price new_value profit h,
  cases h with hp1 h,
  cases h with hv1 hp2,
  rw hp1 at hv1,
  rw hp1 at hp2,
  rw hv1 at hp2,
  rw hp2,
  rw hp1,
  norm_num,
  exact eq.refl 200

end mark_profit_from_selling_magic_card_l195_195032


namespace least_product_of_primes_over_30_l195_195478

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l195_195478


namespace count_increasing_digits_between_200_and_300_l195_195325

def is_increasing_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 3 ∧ digits.nodup ∧ digits = digits.sorted

def count_integers_increasing_order (a b : ℕ) : ℕ :=
  (a+1).upto b |>.filter (λ n, is_increasing_digits n) |>.length

theorem count_increasing_digits_between_200_and_300 :
  count_integers_increasing_order 200 300 = 21 :=
sorry

end count_increasing_digits_between_200_and_300_l195_195325


namespace Balloonist_Arrangements_l195_195236

theorem Balloonist_Arrangements :
  ∑ i in "BALLOONIST".toFinset, ("BALLOONIST".count i).factorial ∣ (10.factorial / (2.factorial * 2.factorial)) = 907200 :=
by
  sorry

end Balloonist_Arrangements_l195_195236


namespace graphs_are_symmetric_about_origin_l195_195666

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log a x

-- Conditions
variables {a b : ℝ}
variable (x : ℝ)
hypothesis h1 : log a + log b = 0
hypothesis h2 : a ≠ 1
hypothesis h3 : b ≠ 1

-- A function to check symmetry about the origin
def symmetric_about_origin (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -g(x)

theorem graphs_are_symmetric_about_origin
    (h1 : log a + log b = 0)
    (h2 : a ≠ 1)
    (h3 : b ≠ 1) :
    symmetric_about_origin (f a) (g a) :=
sorry

end graphs_are_symmetric_about_origin_l195_195666


namespace range_of_x_l195_195006

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (x : α) : α 

def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing_on_positive (f : α → α) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

def f1_zero (f : α → α) : Prop :=
  f 1 = 0

theorem range_of_x (f : α → α) : 
  is_odd_function f → is_increasing_on_positive f → f1_zero f →
  ∀ x, (f x - f (-x)) / x < 0 ↔ x ∈ Ioo (-1 : α) 0 ∪ Ioo 0 1 :=
by
  intros
  first assume (is_odd_function f)
  assume (is_increasing_on_positive f)
  assume (f1_zero f)
  sorry

end range_of_x_l195_195006


namespace algebraic_expression_value_l195_195336

theorem algebraic_expression_value (x : ℝ) :
  let a := 2003 * x + 2001
  let b := 2003 * x + 2002
  let c := 2003 * x + 2003
  a^2 + b^2 + c^2 - a * b - a * c - b * c = 3 :=
by
  sorry

end algebraic_expression_value_l195_195336


namespace least_product_of_distinct_primes_over_30_l195_195504

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l195_195504


namespace edward_money_l195_195988

theorem edward_money (X : ℝ) (H1 : X - 130 - 0.25 * (X - 130) = 270) : X = 490 :=
by
  sorry

end edward_money_l195_195988


namespace max_possible_value_of_e_n_l195_195784

noncomputable def b (n : ℕ) : ℚ := (8^n - 1) / 7

def e (n : ℕ) : ℕ := Int.gcd (b n).numerator (b (n + 1)).numerator

theorem max_possible_value_of_e_n : ∀ n : ℕ, e n = 1 :=
by sorry

end max_possible_value_of_e_n_l195_195784


namespace sum_last_two_digits_9_pow_23_plus_11_pow_23_l195_195894

theorem sum_last_two_digits_9_pow_23_plus_11_pow_23 :
  (9^23 + 11^23) % 100 = 60 :=
by
  sorry

end sum_last_two_digits_9_pow_23_plus_11_pow_23_l195_195894


namespace smallest_n_satisfying_condition_l195_195188

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (n > 1) ∧ (∀ i : ℕ, i ≥ 1 → i < n → (∃ k : ℕ, i + (i+1) = k^2)) ∧ n = 8 :=
sorry

end smallest_n_satisfying_condition_l195_195188


namespace find_c_l195_195271

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_c (c : ℝ) (h1 : f 1 = 1) (h2 : ∀ x y : ℝ, f (x + y) = f x + f y + 8 * x * y - c) (h3 : f 7 = 163) :
  c = 2 / 3 :=
sorry

end find_c_l195_195271


namespace oil_level_drop_l195_195568

-- Let V_truck be the volume of the truck's tank
def V_truck (r_truck h_truck : ℝ) : ℝ := π * r_truck^2 * h_truck

-- Let V_drop be the volume of oil removed from the stationary tank
def V_drop (r_stationary h_drop : ℝ) : ℝ := π * r_stationary^2 * h_drop

-- Given radius and height of truck's tank
def r_truck : ℝ := 4
def h_truck : ℝ := 10

-- Given radius of the stationary tank
def r_stationary : ℝ := 100

-- Volume of the truck's tank
def volume_truck := V_truck r_truck h_truck

-- Proof that, given the calculated volume drop and radius, the oil level drops by 0.016 feet
theorem oil_level_drop : 
  (V_drop r_stationary h_drop = volume_truck) → h_drop = 0.016 := by
  sorry

end oil_level_drop_l195_195568


namespace partition_integer_sets_l195_195628

theorem partition_integer_sets (a b d n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (∃ (A B : set ℕ),  0 ∈ A ∧ 0 ∈ B ∧ (∀ k ∈ A, ∃ m ∈ B, a * k = b * m) ∧ (∀ k ∈ B, ∃ m ∈ A, b * k = a * m) ∧ 
  A ∪ B = set.univ ∧ A ∩ B = ∅) ↔ (∃ d n: ℕ, a = d ∧ b = n * d) ∨ (∃ d n: ℕ, b = d ∧ a = n * d) :=
by sorry

end partition_integer_sets_l195_195628


namespace balloon_permutations_l195_195220

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l195_195220


namespace number_of_factors_n_l195_195983

-- Defining the value of n with its prime factorization
def n : ℕ := 2^5 * 3^9 * 5^5

-- Theorem stating the number of natural-number factors of n
theorem number_of_factors_n : 
  (Nat.divisors n).card = 360 := by
  -- Proof is omitted
  sorry

end number_of_factors_n_l195_195983


namespace square_garden_area_l195_195961

/-- Calculate the area of a square garden where one side length is given by 
    the simplified expression of (2222 - 2121)^2 / 196 -/
theorem square_garden_area : 
  let side_length := (2222 - 2121)^2 / 196 in
  side_length^2 = 2401 :=
  by
    let side_length := (2222 - 2121)^2 / 196
    show side_length^2 = 2401
    sorry

end square_garden_area_l195_195961


namespace total_cost_price_l195_195938
noncomputable theory

def CP1 : ℝ := 600 / 1.25
def CP2 : ℝ := 800 / 0.80
def CP3 : ℝ := 1000 / 1.30
def TCP : ℝ := CP1 + CP2 + CP3

theorem total_cost_price :
  TCP = 2249.23 := 
by sorry

end total_cost_price_l195_195938


namespace Zoe_wrote_GRE_in_June_l195_195899

-- Define the months and their respective indexes
inductive Month
| January | February | March | April | May | June | July | August | September | October | November | December

open Month

-- Define a function to compute the month from a starting month and a number of months to add
def add_months (start: Month) (n: ℕ) : Month :=
  match start, n with
  | January, n   => Month.recOn n January
  | February, n  => Month.recOn (n+1) January
  | March, n     => Month.recOn (n+2) January
  | April, n     => Month.recOn (n+3) January
  | May, n       => Month.recOn (n+4) January
  | June, n      => Month.recOn (n+5) January
  | July, n      => Month.recOn (n+6) January
  | August, n    => Month.recOn (n+7) January
  | September, n => Month.recOn (n+8) January
  | October, n   => Month.recOn (n+9) January
  | November, n  => Month.recOn (n+10) January
  | December, n  => Month.recOn (n+11) January

theorem Zoe_wrote_GRE_in_June : add_months April 2 = June :=
sorry

end Zoe_wrote_GRE_in_June_l195_195899


namespace least_product_of_distinct_primes_over_30_l195_195506

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l195_195506


namespace employee_payment_l195_195139

theorem employee_payment (X Y : ℝ) 
  (h1 : X + Y = 880) 
  (h2 : X = 1.2 * Y) : Y = 400 := by
  sorry

end employee_payment_l195_195139


namespace binomial_expansion_constant_term_l195_195068

theorem binomial_expansion_constant_term : 
  let T (r : ℕ) := (nat.choose 6 r) * (2^(6-r)) * (x^(6-3*r)) 
  in 6 - 3 * 2 = 0 → (2 ^ 4) * (nat.choose 6 2) = 240 :=
by
  intros
  sorry

end binomial_expansion_constant_term_l195_195068


namespace product_ab_ac_l195_195354

theorem product_ab_ac (a b k : ℝ) (h : ∀ (A B C P Q X Y : Type),
  acute_triangle A B C →
  perpendicular C A B P →
  perpendicular B A C Q →
  on_circumcircle A B C X Y PQ →
  distance X P = 12 →
  distance P Q = 20 →
  distance Q Y = 13 →
  AB = b / k →
  AC = a / k →
  k = (ab) / (a^2 + 396) →
  k = (ab) / (b^2 + 416) →
  400 = (a^2 + b^2 - 2abk) →
  (AB * AC = 500 * Real.sqrt 15)) : 
  AB * AC = 500 * Real.sqrt 15 ∧ m + n = 515 :=
by
  sorry

end product_ab_ac_l195_195354


namespace no_solutions_l195_195060

theorem no_solutions (x : ℝ) (h : x ≠ 0) : 4 * Real.sin x - 3 * Real.cos x ≠ 5 + 1 / |x| := 
by
  sorry

end no_solutions_l195_195060


namespace eval_expression_l195_195991

def x : ℤ := 18 / 3 * 7^2 - 80 + 4 * 7

theorem eval_expression : -x = -242 := by
  sorry

end eval_expression_l195_195991


namespace sequence_equals_identity_l195_195460

theorem sequence_equals_identity (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j) : 
  ∀ i : ℕ, a i = i := 
by 
  sorry

end sequence_equals_identity_l195_195460


namespace math_problem_l195_195969

/-- The proof problem: Calculate -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11. -/
theorem math_problem : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 :=
by
  sorry

end math_problem_l195_195969


namespace problem_I_problem_II_problem_III_l195_195700

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * (log x - x)
noncomputable def g (x : ℝ) (b : ℝ) := - (1/2) * x^2 + (1 - b) * x

theorem problem_I {b : ℝ} (h : Deriv f 1 = 1) : ∃ a : ℝ, a = -(1/2) := sorry

theorem problem_II (a b : ℝ) :
  (b ≤ -4 ∧ (exists x : ℝ, x = -(b + sqrt (b^2 + 4 * b)) / 2 ∨ x = -(b - sqrt (b^2 + 4 * b)) / 2)) ∨
  ((-4 ≤ b ∧ b ≤ 0) ∧ (∀ x, x ∉ ℝ)) ∨
  (b > 0 ∧ ∃ x, x = -(b - sqrt (b^2 + 4 * b)) / 2) := sorry

theorem problem_III (m : ℝ) : ∀ b ∈ set.Ioi 1,
  (∃ x1 x2 ∈ set.Icc 1 b, f x1 0 b - f x2 0 b - 1 > g x1 b - g x2 b + m) ↔ m ≤ -1 := sorry

end problem_I_problem_II_problem_III_l195_195700


namespace cone_volume_not_product_base_height_l195_195465

noncomputable def cone_volume (S h : ℝ) := (1/3) * S * h

theorem cone_volume_not_product_base_height (S h : ℝ) :
  cone_volume S h ≠ S * h :=
by sorry

end cone_volume_not_product_base_height_l195_195465


namespace find_prime_p_l195_195641

-- setup for the problem
def is_prime (p : ℕ) : Prop := Nat.Prime p

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).length

theorem find_prime_p
  (p : ℕ)
  (hp : is_prime p)
  (hdiv : num_divisors (3^p + 4^p + 5^p + 9^p - 98) ≤ 6) :
  p = 2 ∨ p = 3 :=
sorry

end find_prime_p_l195_195641


namespace weight_of_each_dumbbell_l195_195818

-- Definitions based on conditions
def initial_dumbbells : Nat := 4
def added_dumbbells : Nat := 2
def total_dumbbells : Nat := initial_dumbbells + added_dumbbells -- 6
def total_weight : Nat := 120

-- Theorem statement
theorem weight_of_each_dumbbell (h : total_dumbbells = 6) (w : total_weight = 120) :
  total_weight / total_dumbbells = 20 :=
by
  -- Proof is to be written here
  sorry

end weight_of_each_dumbbell_l195_195818


namespace tom_total_payment_l195_195875

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end tom_total_payment_l195_195875


namespace points_P_Q_R_C_on_same_circle_l195_195287

open EuclideanGeometry

variables (A B C A₁ B₁ P S Q R: Point)

-- Assuming the necessary conditions
variable (h_acute : is_acute_triangle A B C)
variable (h_AC_lt_BC : distance A C < distance B C)
variable (h_circle : circle A B)
variable (h_A₁ : on_circle h_circle A₁ ∧ lies_on_segment A₁ C A)
variable (h_B₁ : on_circle h_circle B₁ ∧ lies_on_segment B₁ C B)
variable (h_circumcircle_ABC_A₁B₁C : circumcircle A B C ∩ circumcircle A₁ B₁ C = {P})
variable (h_intersection : intersection (segment A B₁) (segment B A₁) = S)
variable (h_symmetric_Q : symmetric_point S (line C A) = Q)
variable (h_symmetric_R : symmetric_point S (line C B) = R)

theorem points_P_Q_R_C_on_same_circle :
  on_circle (circumcircle P Q R) C :=
sorry

end points_P_Q_R_C_on_same_circle_l195_195287


namespace positive_difference_between_jo_and_laura_sums_l195_195375

noncomputable def jo_sum : ℕ := (200 * 201) / 2

noncomputable def nearest_multiple_of_five (n : ℕ) : ℕ :=
  let r := n % 5
  if r < 3 then n - r else n + 5 - r

noncomputable def laura_sum : ℕ := 
  (Finset.range 200).sum (λ n, nearest_multiple_of_five (n + 1))

theorem positive_difference_between_jo_and_laura_sums :
  |jo_sum - laura_sum| = 0 :=
  sorry

end positive_difference_between_jo_and_laura_sums_l195_195375


namespace distance_james_rode_l195_195371

def speed : ℝ := 80.0
def time : ℝ := 16.0
def distance : ℝ := speed * time

theorem distance_james_rode :
  distance = 1280.0 :=
by
  -- to show the theorem is sane
  sorry

end distance_james_rode_l195_195371


namespace sequence_of_cards_l195_195812

theorem sequence_of_cards (card1 card2 card3 : String) 
(Cond1 : card1 = "A")
(Cond2 : card3 = "K") 
(Cond3 : card1 = "H")
(Cond4 : card3 = "D")
(Cond5 : card2 = "K" ∧ card2 = "H" ∧ card2 = "K" ∧ card2 = "H") :
(card1, card2, card3) = ("H", "H", "D") :=
begin
  sorry
end

end sequence_of_cards_l195_195812


namespace cylinder_volume_l195_195649

theorem cylinder_volume :
  let s := 20 in
  let h := s * Real.sqrt 2 in
  let r := s / 2 in
  let V := Real.pi * r^2 * h in
  V = 2000 * Real.sqrt 2 * Real.pi := by
  let s := 20
  let h := s * Real.sqrt 2
  let r := s / 2
  let V := Real.pi * r^2 * h
  sorry

end cylinder_volume_l195_195649


namespace product_of_terms_eq_72_l195_195403

theorem product_of_terms_eq_72
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 12) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 72 :=
by
  sorry

end product_of_terms_eq_72_l195_195403


namespace initial_cards_l195_195421

/-- Given the conditions of the problem, prove that Robie had 627 cards in the beginning. -/
theorem initial_cards (cards_per_box : ℕ) (loose_cards : ℕ) (boxes_given : ℕ) (boxes_returned : ℕ) (current_boxes : ℕ) (cards_bought : ℕ) (cards_traded : ℕ) :
  cards_per_box = 30 →
  loose_cards = 18 →
  boxes_given = 8 →
  boxes_returned = 2 →
  current_boxes = 15 →
  cards_bought = 21 →
  cards_traded = 12 →
  ((current_boxes - boxes_returned + boxes_given) * cards_per_box + loose_cards - cards_bought = 627) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  have h8 : current_boxes - boxes_returned + boxes_given = 21 := by rw [h5, h4, h3]; norm_num
  have h9 : (current_boxes - boxes_returned + boxes_given) * cards_per_box = 630 := by rw [h8, h1]; norm_num
  have h10 : (current_boxes - boxes_returned + boxes_given) * cards_per_box + loose_cards = 648 := by rw [h9, h2]; norm_num
  have h11 : (current_boxes - boxes_returned + boxes_given) * cards_per_box + loose_cards - cards_bought = 627 := by rw [h10, h6]; norm_num
  exact h11

end initial_cards_l195_195421


namespace man_is_older_by_24_l195_195573

-- Define the conditions as per the given problem
def present_age_son : ℕ := 22
def present_age_man (M : ℕ) : Prop := M + 2 = 2 * (present_age_son + 2)

-- State the problem: Prove that the man is 24 years older than his son
theorem man_is_older_by_24 (M : ℕ) (h : present_age_man M) : M - present_age_son = 24 := 
sorry

end man_is_older_by_24_l195_195573


namespace total_cost_in_dollars_l195_195435

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end total_cost_in_dollars_l195_195435


namespace existence_of_nm_l195_195049

theorem existence_of_nm (m : ℕ) (hm : 0 < m) :
  ∃ n_m : ℕ, 0 < n_m ∧ ∀ n : ℕ, n ≥ n_m → ∃ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) ∧ (∑ i in Finset.range n, 1 / (a i)^m = 1) :=
sorry

end existence_of_nm_l195_195049


namespace finitely_many_squares_in_seq_l195_195679

open Nat

theorem finitely_many_squares_in_seq (a : ℤ) (p : ℕ) (h1 : p.Prime) (h2 : p ∣ a) (h3 : p ≡ 3 [MOD 8] ∨ p ≡ -3 [MOD 8]) :
    ∀ (a_n : ℕ → ℤ), (∀ n, a_n n = 2 ^ n + a) → 
    ∃ (N : ℕ), ∀ n > N, ¬ ∃ x : ℤ, a_n n = x^2 :=
sorry

end finitely_many_squares_in_seq_l195_195679


namespace company_initial_bureaus_l195_195922

theorem company_initial_bureaus (B : ℕ) (offices : ℕ) (extra_bureaus : ℕ) 
  (h1 : offices = 14) 
  (h2 : extra_bureaus = 10) 
  (h3 : (B + extra_bureaus) % offices = 0) : 
  B = 8 := 
by
  sorry

end company_initial_bureaus_l195_195922


namespace euclidean_division_P1_by_D1_euclidean_division_P2_by_D2_l195_195149

-- Definitions for the first problem
def P1 (X : ℝ) : ℝ := X^4 + 3 * X^2 + 2 * X + 1
def D1 (X : ℝ) : ℝ := X^2 - 2
def Q1 (X : ℝ) : ℝ := X^2 + 5
def R1 (X : ℝ) : ℝ := 2 * X + 11

-- Definitions for the second problem
def P2 (X : ℝ) (n : ℕ) : ℝ := X^n - X + 1
def D2 (X : ℝ) : ℝ := X^2 - 1
def R2 (X : ℝ) (n : ℕ) : ℝ := -((1 + (-1)^(n : ℤ)) / 2) * X + (3 + (-1)^(n : ℤ)) / 2

-- First proof statement
theorem euclidean_division_P1_by_D1 :
  ∀ X : ℝ, P1(X) = D1(X) * Q1(X) + R1(X) :=
by
  sorry

-- Second proof statement
theorem euclidean_division_P2_by_D2 :
  ∀ (X : ℝ) (n : ℕ), n > 0 → P2(X, n) = D2(X) * (P2(X, n) / D2(X)) + R2(X, n) :=
by
  sorry

end euclidean_division_P1_by_D1_euclidean_division_P2_by_D2_l195_195149


namespace polygon_interior_angle_144_proof_l195_195182

-- Definitions based on the conditions in the problem statement
def interior_angle (n : ℕ) : ℝ := 144
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- The problem statement as a Lean 4 theorem to prove n = 10
theorem polygon_interior_angle_144_proof : ∃ n : ℕ, interior_angle n = 144 ∧ sum_of_interior_angles n = n * 144 → n = 10 := by
  sorry

end polygon_interior_angle_144_proof_l195_195182


namespace least_product_of_primes_over_30_l195_195482

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l195_195482


namespace find_z_l195_195720

theorem find_z (z : ℂ) (hz : conj z * (1 + complex.I) = 1 - complex.I) : 
  z = complex.I :=
by 
  sorry

end find_z_l195_195720


namespace collinear_implies_coplanar_l195_195180

-- Define the concept of collinear points in space
def collinear (p1 p2 p3 : Point) : Prop := ∃ l : Line, p1 ∈ l ∧ p2 ∈ l ∧ p3 ∈ l

-- Define the concept of coplanar points in space
def coplanar (s : Set Point) : Prop := ∃ p : Plane, ∀ x ∈ s, x ∈ p

-- State the problem conditions and conclusion in Lean statement
theorem collinear_implies_coplanar (p1 p2 p3 p4 : Point) :
  collinear p1 p2 p3 → coplanar {p1, p2, p3, p4} ∧ 
  ¬ (coplanar {p1, p2, p3, p4} → collinear p1 p2 p3) :=
sorry

end collinear_implies_coplanar_l195_195180


namespace total_interest_rate_l195_195948

theorem total_interest_rate (I_total I_11: ℝ) (r_9 r_11: ℝ) (h1: I_total = 100000) (h2: I_11 = 12499.999999999998) (h3: I_11 < I_total):
  r_9 = 0.09 →
  r_11 = 0.11 →
  ( ((I_total - I_11) * r_9 + I_11 * r_11) / I_total * 100 = 9.25 ) :=
by
  sorry

end total_interest_rate_l195_195948


namespace greatest_elements_in_T_l195_195170

-- Conditions
def T : Set ℕ := sorry -- Define the set T satisfying the conditions
def has_integer_arithmetic_mean (T : Set ℕ) (y : ℕ) : Prop :=
  let T' := T.erase y
  (∑ i in T', i) % (T'.card) = 0

axiom prop1 (y : ℕ) (hy : y ∈ T) : has_integer_arithmetic_mean T y
axiom prop2 : 2 ∈ T
axiom prop3 : 1024 = Finset.max' T sorry

-- Target proving statement
theorem greatest_elements_in_T : T.card ≤ 15 := sorry

end greatest_elements_in_T_l195_195170


namespace find_complex_z_l195_195715

theorem find_complex_z (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
sorry

end find_complex_z_l195_195715


namespace prod_of_two_reds_is_red_l195_195947

-- Definitions for the problem conditions
def Color := {red, blue} -- The colors available

variables (is_red : ℕ → Prop) -- Predicate: is_red n means n is red
variables (is_blue : ℕ → Prop) -- Predicate: is_blue n means n is blue

-- Condition: all numbers are either red or blue
axiom all_colored : ∀ n : ℕ, is_red n ∨ is_blue n

-- Condition: sum of different colors is blue
axiom sum_diff_colors_blue : ∀ {a b : ℕ}, (is_red a ∧ is_blue b) ∨ (is_blue a ∧ is_red b) → is_blue (a + b)

-- Condition: product of different colors is red
axiom prod_diff_colors_red : ∀ {a b : ℕ}, (is_red a ∧ is_blue b) ∨ (is_blue a ∧ is_red b) → is_red (a * b)

-- Theorem: product of two red numbers is red
theorem prod_of_two_reds_is_red (p1 p2 : ℕ) (h1 : is_red p1) (h2 : is_red p2) : is_red (p1 * p2) :=
sorry

end prod_of_two_reds_is_red_l195_195947


namespace arithmetic_sequence_l195_195462

-- Define the arithmetic sequence and its sum properties
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
def SeqSum_conditions (n : ℕ) : Prop := 
  S 5 = 25 ∧ a 5 = 9

-- Claim
def SeqSum_claim (n : ℕ) : Prop :=
  S 8 = 64

-- Proof statement
theorem arithmetic_sequence (n : ℕ) (h : SeqSum_conditions n) : SeqSum_claim n :=
by sorry

end arithmetic_sequence_l195_195462


namespace least_product_of_distinct_primes_over_30_l195_195502

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l195_195502


namespace table_sale_price_percentage_l195_195157

theorem table_sale_price_percentage (W : ℝ) : 
  let S := 1.4 * W
  let P := 0.65 * S
  P = 0.91 * W :=
by
  sorry

end table_sale_price_percentage_l195_195157


namespace algae_difference_l195_195035

-- Define the original number of algae plants.
def original_algae := 809

-- Define the current number of algae plants.
def current_algae := 3263

-- Statement to prove: The difference between the current number of algae plants and the original number of algae plants is 2454.
theorem algae_difference : current_algae - original_algae = 2454 := by
  sorry

end algae_difference_l195_195035


namespace white_square_area_l195_195378

-- Define variables based on the conditions
def edge_length : ℝ := 15
def total_blue_paint : ℝ := 500
def faces_of_cube : ℝ := 6
def area_of_one_face : ℝ := edge_length ^ 2
def total_surface_area : ℝ := faces_of_cube * area_of_one_face
def blue_area_per_face : ℝ := total_blue_paint / faces_of_cube
def white_area_per_face : ℝ := area_of_one_face - blue_area_per_face

-- Statement to prove the area of one of the white squares on the face of the cube
theorem white_square_area : white_area_per_face = (425 / 3) :=
  by
    sorry

end white_square_area_l195_195378


namespace model_to_real_water_tower_volume_l195_195160

theorem model_to_real_water_tower_volume (h_real_tower : ℝ) (h_model : ℝ) (h_conversion : ℝ) 
  (volume_scale : ℝ) :
  h_real_tower = 90 →
  h_model = 0.5 →
  h_conversion = 1728 →
  volume_scale = (h_real_tower / h_model) ^ 3 →
  (volume_scale / h_conversion) = 3375 :=
by
  intros h_real_tower_val h_model_val h_conversion_val volume_scale_val
  rw [h_real_tower_val, h_model_val, h_conversion_val, volume_scale_val]
  sorry

end model_to_real_water_tower_volume_l195_195160


namespace library_growth_rate_l195_195871

theorem library_growth_rate (C_2022 C_2024: ℝ) (h₁ : C_2022 = 100000) (h₂ : C_2024 = 144000) :
  ∃ x : ℝ, (1 + x) ^ 2 = C_2024 / C_2022 ∧ x = 0.2 := 
by {
  sorry
}

end library_growth_rate_l195_195871


namespace infinite_series_sum_l195_195198

theorem infinite_series_sum :
  (∑' n : ℕ, n > 0 → (2 * n^3 + n^2 - n + 1) / (n^5 - n^4 + n^3 - n^2 + n)) = 1 :=
begin
  sorry
end

end infinite_series_sum_l195_195198


namespace minimum_box_height_l195_195411

theorem minimum_box_height (x : ℝ) (h₁ : ∀ y, y ≥ 0 → y^2 ≥ 12 → x ≥ y)
    : 2*x = 4*real.sqrt 3 :=
  sorry

end minimum_box_height_l195_195411


namespace min_triangles_cover_large_triangle_l195_195118

/-- 
A proof problem equivalent to finding the minimum number of equilateral 
triangles of side length 1 unit required to cover an equilateral triangle 
with side length 15 units.
-/
theorem min_triangles_cover_large_triangle (side_small side_large : ℕ) (h1 : side_small = 1) (h2 : side_large = 15) :
  let area_ratio := (side_small / side_large : ℚ)^2,
      num_triangles := 1 / area_ratio in
  num_triangles = 225 :=
by 
  sorry

end min_triangles_cover_large_triangle_l195_195118


namespace bird_counts_remaining_l195_195466

theorem bird_counts_remaining
  (peregrine_falcons pigeons crows sparrows : ℕ)
  (chicks_per_pigeon chicks_per_crow chicks_per_sparrow : ℕ)
  (peregrines_eat_pigeons_percent peregrines_eat_crows_percent peregrines_eat_sparrows_percent : ℝ)
  (initial_peregrine_falcons : peregrine_falcons = 12)
  (initial_pigeons : pigeons = 80)
  (initial_crows : crows = 25)
  (initial_sparrows : sparrows = 15)
  (chicks_per_pigeon_cond : chicks_per_pigeon = 8)
  (chicks_per_crow_cond : chicks_per_crow = 5)
  (chicks_per_sparrow_cond : chicks_per_sparrow = 3)
  (peregrines_eat_pigeons_percent_cond : peregrines_eat_pigeons_percent = 0.4)
  (peregrines_eat_crows_percent_cond : peregrines_eat_crows_percent = 0.25)
  (peregrines_eat_sparrows_percent_cond : peregrines_eat_sparrows_percent = 0.1)
  : 
  (peregrine_falcons = 12) ∧
  (pigeons = 48) ∧
  (crows = 19) ∧
  (sparrows = 14) :=
by
  sorry

end bird_counts_remaining_l195_195466


namespace team_A_min_bid_at_4_valid_a_range_l195_195565

-- Define the conditions
def existing_wall_height : ℝ := 3
def base_area : ℝ := 12
def front_wall_cost_per_sqm : ℝ := 400
def side_wall_cost_per_sqm : ℝ := 150
def total_other_costs : ℝ := 7200
def length_range := set.Icc (2 : ℝ) (6 : ℝ)

-- Define team A's cost function
def team_A_cost (x : ℝ) : ℝ :=
  have front_wall_length := base_area / x
  900 * (16 / x + x) + total_other_costs

-- Define team B's cost function
def team_B_cost (x a : ℝ) : ℝ :=
  (900 * a * (1 + x)) / x

-- Part 1: Prove that the minimum cost occurs when x = 4
theorem team_A_min_bid_at_4 : 
  (∀ x ∈ length_range, team_A_cost x ≥ team_A_cost 4) :=
by
  sorry

-- Part 2: Prove the range for a
theorem valid_a_range (a : ℝ) : 
  (∀ x ∈ length_range, team_A_cost x > team_B_cost x a) → 0 < a ∧ a < 12 :=
by
  sorry

end team_A_min_bid_at_4_valid_a_range_l195_195565


namespace percentage_relationship_l195_195540

variable {x y z : ℝ}

theorem percentage_relationship (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z :=
by
  sorry

end percentage_relationship_l195_195540


namespace whole_number_fraction_l195_195129

theorem whole_number_fraction (h₁: 8 / 6 ≠ 1)
    (h₂: 9 / 5 ≠ 1)
    (h₃: 10 / 4 ≠ 2)
    (h₄: 11 / 3 ≠ 3)
    (h₅: 12 / 2 = 6) : 
    (∃ x, 12 / 2 = x ∧ (forall y, y ∈ [8 / 6, 9 / 5, 10 / 4, 11 / 3] → y ≠ x)) :=
begin
  use 6,
  split,
  { exact h₅, },
  { intros y hy,
    rcases hy with ⟨rfl | rfl | rfl | rfl⟩,
    any_goals { contradiction, },
  }
end

end whole_number_fraction_l195_195129


namespace length_of_CD_l195_195647

noncomputable def volume_of_3d_region (r : ℝ) (l : ℝ) : ℝ :=
  2 * (2 / 3) * Math.pi * r^3 + Math.pi * r^2 * l

theorem length_of_CD (r : ℝ := 4) (volume : ℝ := 384 * Math.pi) :
  Exists (fun l : ℝ => volume_of_3d_region r l = volume) := by
  use 18.67
  sorry

end length_of_CD_l195_195647


namespace angle_bisector_slope_l195_195438

theorem angle_bisector_slope 
  (m1 m2 : ℝ) 
  (h1 : m1 = 2) 
  (h2 : m2 = 4) 
  : (∀ m : ℝ, y = m * x) → m = (sqrt 21 - 6) / 7 :=
by
  sorry

end angle_bisector_slope_l195_195438


namespace time_to_ride_down_when_standing_l195_195766

variable (d b s : ℝ) -- Define the variables for distance, Brian's speed, and escalator's speed

-- Conditions given in the problem
axiom h1 : d = 80 * b
axiom h2 : d = 30 * (b + s)

-- Proven that
theorem time_to_ride_down_when_standing : s = (5 * b) / 3 → (d / s) = 48 :=
by
  intros h3
  sorry

end time_to_ride_down_when_standing_l195_195766


namespace principal_amount_l195_195901

-- Define the principal amount problem with given conditions
theorem principal_amount (R : ℝ) (T : ℝ) (SI : ℝ) (P : ℝ) :
  R = 12 → T = 3 → SI = 5400 → SI = (P * R * T) / 100 → P = 15000 :=
by
  intros hR hT hSI hFormula
  rw [hR, hT] at hFormula
  have h : 5400 = (P * 36) / 100 := hFormula
  calc
    5400 = (P * 36) / 100          : h
    5400 * 100 = P * 36            : by linarith
    540000 = P * 36                : by linarith
    P = 540000 / 36                : by linarith
    P = 15000                      : by norm_num

end principal_amount_l195_195901


namespace smallest_prime_linear_pair_l195_195881

def is_prime (n : ℕ) : Prop := ¬(∃ k > 1, k < n ∧ k ∣ n)

theorem smallest_prime_linear_pair :
  ∃ a b : ℕ, is_prime a ∧ is_prime b ∧ a + b = 180 ∧ a > b ∧ b = 7 := 
by
  sorry

end smallest_prime_linear_pair_l195_195881


namespace smallest_n_for_10n_minus_1_mod_37_l195_195121

theorem smallest_n_for_10n_minus_1_mod_37 : ∃ n : ℕ, 10^n - 1 % 37 = 0 ∧ ∀ m : ℕ, m < n → 10^m - 1 % 37 ≠ 0 :=
begin
  use 3,
  sorry
end

end smallest_n_for_10n_minus_1_mod_37_l195_195121


namespace second_pipe_filling_time_l195_195468

theorem second_pipe_filling_time :
  ∀ (T : ℝ), (1 / 60 * 37 + 1 / T * 27 = 1) → T = 1620 / 23 := 
begin
  sorry
end

end second_pipe_filling_time_l195_195468


namespace optionA_is_quadratic_optionB_is_not_quadratic_optionC_is_not_quadratic_optionD_is_not_quadratic_only_optionA_is_quadratic_l195_195529

-- Define the given options as propositions
def optionA (x : ℝ) := 3 * x^2 - 5 * x = 6
def optionB (x : ℝ) := 1 / x - 2 = 0
def optionC (x : ℝ) := 6 * x + 1 = 0
def optionD (x y : ℝ) := 2 * x^2 + y^2 = 0

-- Define the property of being a quadratic equation in terms of a single variable x
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c, (∀ x, eq x = (a * x^2 + b * x + c = 0)) ∧ a ≠ 0

-- State the theorem
theorem optionA_is_quadratic : is_quadratic optionA :=
sorry

-- State that other options are not quadratic equations in terms of x
theorem optionB_is_not_quadratic : ¬is_quadratic optionB :=
sorry

theorem optionC_is_not_quadratic : ¬is_quadratic optionC :=
sorry

theorem optionD_is_not_quadratic : ¬is_quadratic optionD :=
sorry

-- The main conclusion that only option A is a quadratic equation in terms of x
theorem only_optionA_is_quadratic (x y : ℝ) :
  is_quadratic optionA ∧ 
  ¬is_quadratic optionB ∧ 
  ¬is_quadratic optionC ∧ 
  ¬is_quadratic (optionD x y) :=
⟨optionA_is_quadratic, optionB_is_not_quadratic, optionC_is_not_quadratic, optionD_is_not_quadratic⟩

end optionA_is_quadratic_optionB_is_not_quadratic_optionC_is_not_quadratic_optionD_is_not_quadratic_only_optionA_is_quadratic_l195_195529


namespace part1_min_value_function_part2_prove_inequality_l195_195313

-- Lean translation for Part (1)
theorem part1_min_value_function (a : ℝ) (x : ℝ) (h_a : a = 1) (h_x : 0 ≤ x) :
  let f (x : ℝ) := (x + 2)/Real.exp x + x - 2 in
  f 0 = 0 := sorry

-- Lean translation for Part (2)
theorem part2_prove_inequality (a x1 x2 : ℝ) (h_extreme : f'(x1) = 0 ∧ f'(x2) = 0) (h_order : x1 < x2) (h_a : 0 < a ∧ a < 1) :
  let f' (x : ℝ) := a - (x + 1)/Real.exp x in
  Real.exp x2 - Real.exp x1 > (2/a) - 2 := sorry

end part1_min_value_function_part2_prove_inequality_l195_195313


namespace jason_reroll_prob_l195_195768

open real

/-- Given four fair standard six-sided dice, Jason can reroll (a) all four dice or 
(b) keep two and reroll the other two. He wins if the sum of the numbers on the four dice is 9.
Jason chooses the optimal reroll strategy to maximize his winning chances. Prove that the probability
that Jason chooses to reroll exactly two of the dice is 1/18. -/

theorem jason_reroll_prob : 
  let dice_distribution := list.range' 1 6 in
  let dice_combinations := {ns : list ℕ // ns.length = 4 ∧ ∀ n ∈ ns, n ∈ dice_distribution} in
  let win_probability (ns : list ℕ) := if ns.sum = 9 then (1 : ℝ)/(dice_combinations.card) else 0 in
  let optimal_strategy_probability :=
    max (∑ ns in dice_combinations, win_probability ns)
        (∑ (pair1 pair2 : ℕ) in dice_combinations, 
          (if (pair1 + pair2) ∈ dice_distribution then 1 else 0) / (6 * 6)) in
  (optimal_strategy_probability = 1/18) :=
sorry

end jason_reroll_prob_l195_195768


namespace dice_sum_probability_l195_195096

theorem dice_sum_probability :
  (∑ x in ({1, 2, 3, 4, 5, 6} : Finset ℕ), x) > 0 →
  (∑ x in ({1, 2, 3, 4, 5, 6} : Finset ℕ), x) > 0 →
  (∑ x in ({1, 2, 3, 4, 5, 6} : Finset ℕ), x) > 0 →
  ((∑ (a : ℕ) in ({1, 2, 3, 4, 5, 6} : Finset ℕ), a) =
  (∑ b in ({1, 2, 3, 4, 5, 6} : Finset ℕ), b) →
  (∑ c in ({1, 2, 3, 4, 5, 6} : Finset ℕ), c) →
  (((Finset.filter (λ (t : ℕ × ℕ × ℕ), t.fst + t.snd.fst + t.snd.snd = 15)
   ({ (x, y, z) | x ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), y ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), z ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) } : Finset (ℕ × ℕ × ℕ))).card / 216 = 1 / 24))) :=
sorry

end dice_sum_probability_l195_195096


namespace decreasing_number_4312_max_decreasing_number_divisible_by_9_l195_195341

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end decreasing_number_4312_max_decreasing_number_divisible_by_9_l195_195341


namespace balloon_permutations_l195_195233

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l195_195233


namespace spinner_divisible_by_5_probability_l195_195987

theorem spinner_divisible_by_5_probability :
  let outcomes := {5, 1, 2},
      total_numbers := outcomes.product(outcomes).product(outcomes),
      divisible_by_5 := {x | (x.2.2 = 5)},
      total_divisible_by_5 := total_numbers.filter (λ x, x ∈ divisible_by_5) in
  total_divisible_by_5.card / total_numbers.card = (1/3 : ℚ) :=
by
  -- Definitions for conditions
  let outcomes := {5, 1, 2}
  let total_numbers := outcomes × outcomes × outcomes
  let divisible_by_5 := {x | (x.2.2 = 5)}
  let total_divisible_by_5 := total_numbers.filter (λ x, x ∈ divisible_by_5)

  -- Main theorem statement
  have : total_divisible_by_5.card / total_numbers.card = (1/3 : ℚ),
  sorry

end spinner_divisible_by_5_probability_l195_195987


namespace commutative_dot_product_distributive_dot_product_l195_195959

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Statement 1: Commutative Law for Scalar Product
theorem commutative_dot_product :
  a ⬝ b = b ⬝ a :=
by sorry

-- Statement 2: Distributive Law for Scalar Product
theorem distributive_dot_product :
  (a + b) ⬝ c = (a ⬝ c) + (b ⬝ c) :=
by sorry

end commutative_dot_product_distributive_dot_product_l195_195959


namespace positional_relationship_l195_195273

theorem positional_relationship 
  (m n : ℝ) 
  (h_points_on_ellipse : (m^2 / 4) + (n^2 / 3) = 1)
  (h_relation : n^2 = 3 - (3/4) * m^2) : 
  (∃ x y : ℝ, (x^2 + y^2 = 1/3) ∧ (m * x + n * y + 1 = 0)) ∨ 
  (∀ x y : ℝ, (x^2 + y^2 = 1/3) → (m * x + n * y + 1 ≠ 0)) :=
sorry

end positional_relationship_l195_195273


namespace cost_increase_per_scrap_rate_l195_195858

theorem cost_increase_per_scrap_rate (x : ℝ) :
  ∀ x Δx, y = 56 + 8 * x → Δx = 1 → y + Δy = 56 + 8 * (x + Δx) → Δy = 8 :=
by
  sorry

end cost_increase_per_scrap_rate_l195_195858


namespace probability_colors_match_l195_195409

section ProbabilityJellyBeans

structure JellyBeans where
  green : ℕ
  blue : ℕ
  red : ℕ

def total_jellybeans (jb : JellyBeans) : ℕ :=
  jb.green + jb.blue + jb.red

-- Define the situation using structures
def lila_jellybeans : JellyBeans := { green := 1, blue := 1, red := 1 }
def max_jellybeans : JellyBeans := { green := 2, blue := 1, red := 3 }

-- Define probabilities
noncomputable def probability (count : ℕ) (total : ℕ) : ℚ :=
  if total = 0 then 0 else (count : ℚ) / (total : ℚ)

-- Main theorem
theorem probability_colors_match :
  probability lila_jellybeans.green (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.green (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.blue (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.blue (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.red (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.red (total_jellybeans max_jellybeans) = 1 / 3 :=
by sorry

end ProbabilityJellyBeans

end probability_colors_match_l195_195409


namespace tangent_line_slope_l195_195304

noncomputable def f : ℝ → ℝ := sorry

-- Hypothesis: The tangent line to the function y = f(x) at the point (1, f(1)) is y = 1/2 * x + 3
def tangent_at_M := ∀ x : ℝ, f(x) = 1 / 2 * x + 3

-- Proof goal: Prove that f(1) + f'(1) = 4
theorem tangent_line_slope (h_tangent : tangent_at_M 1) : f 1 + deriv f 1 = 4 := sorry

end tangent_line_slope_l195_195304


namespace count_extraordinary_numbers_l195_195150

def isExtraordinary (d : Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10) : Prop :=
  d.1 = d.4 ∧ d.2 = d.5 ∧ d.3 = d.6 ∧ d.4 = d.7 ∧ d.5 = d.8 ∧ d.6 = d.9

theorem count_extraordinary_numbers : 
  let count := (Fin 10) × (Fin 10) × (Fin 10) in
  (∀ d : count, isExtraordinary d) →
  ∃ n : ℕ, n = 1000 :=
sorry

end count_extraordinary_numbers_l195_195150


namespace solution_l195_195245

theorem solution (x : ℝ) (h₁ : 0 < x) (h₂ : x * sqrt (20 - x) + sqrt (20 * x - x^3) ≥ 20) : x = 20 := 
sorry

end solution_l195_195245


namespace ratio_of_friday_to_thursday_l195_195161

theorem ratio_of_friday_to_thursday
  (wednesday_copies : ℕ)
  (total_copies : ℕ)
  (ratio : ℚ)
  (h1 : wednesday_copies = 15)
  (h2 : total_copies = 69)
  (h3 : ratio = 1 / 5) :
  (total_copies - wednesday_copies - 3 * wednesday_copies) / (3 * wednesday_copies) = ratio :=
by
  -- proof goes here
  sorry

end ratio_of_friday_to_thursday_l195_195161


namespace ratio_of_averages_is_one_l195_195173

variable {incorrectAverage : ℝ} {numberOfScores : ℕ} {correctScore : ℝ} {incorrectScore : ℝ}
variable (correctedSum : ℝ) (correctedAverage : ℝ) (newSum : ℝ) (newAverage : ℝ)

def initialSum (incorrectAverage * numberOfScores : ℝ) : ℝ :=
  incorrectAverage * numberOfScores

def correctedSum (initialSum : ℝ) : ℝ :=
  initialSum - incorrectScore + correctScore

def correctedAverage (correctedSum : ℝ) : ℝ :=
  correctedSum / numberOfScores

def newSum (correctedSum : ℝ) (correctedAverage : ℝ) : ℝ :=
  correctedSum + correctedAverage

def newAverage (newSum : ℝ) : ℝ :=
  newSum / (numberOfScores + 1)

theorem ratio_of_averages_is_one : 
  let invalidSum := initialSum 77 50
  let validSum := correctedSum invalidSum
  let validAverage := correctedAverage validSum
  let sumIncludingAverage := newSum validSum validAverage
  let averageIncludingPlaceholder := newAverage sumIncludingAverage
  (averageIncludingPlaceholder / validAverage) = 1 :=
by
  -- Skipping proof. Insert detailed proof steps here.
  sorry

end ratio_of_averages_is_one_l195_195173


namespace prime_factors_of_expression_l195_195260

theorem prime_factors_of_expression : ∀ (x : ℕ), ∃ (num : ℕ), (4 ^ 11) * (num ^ 7) * (11 ^ 2) = x → 
  nat.num_prime_divisors x = 31 → num = 7 :=
by
  sorry

end prime_factors_of_expression_l195_195260


namespace balloon_permutations_l195_195209

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l195_195209


namespace pedestrian_travel_time_l195_195163

noncomputable def travel_time (d : ℝ) (x y : ℝ) : ℝ :=
  d / x

theorem pedestrian_travel_time
  (d : ℝ)
  (x y : ℝ)
  (h1 : d = 1)
  (h2 : 3 * x = 1 - x - y)
  (h3 : (1 / 2) * (x + y) = 1 - x - y)
  : travel_time d x y = 9 := 
sorry

end pedestrian_travel_time_l195_195163


namespace total_number_of_marbles_is_1050_l195_195104

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l195_195104


namespace max_planes_15_points_l195_195668

-- Define the total number of points
def total_points : ℕ := 15

-- Define the number of collinear points
def collinear_points : ℕ := 5

-- Compute the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of planes formed by any 3 out of 15 points
def total_planes : ℕ := binom total_points 3

-- Number of degenerate planes formed by the collinear points
def degenerate_planes : ℕ := binom collinear_points 3

-- Maximum number of unique planes
def max_unique_planes : ℕ := total_planes - degenerate_planes

-- Lean theorem statement
theorem max_planes_15_points : max_unique_planes = 445 :=
by
  sorry

end max_planes_15_points_l195_195668


namespace min_max_f_l195_195908

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 3 - 3) * (Real.log (3 * x) / Real.log 3)

theorem min_max_f :
  ∀ x ∈ Icc (1 / 27 : ℝ) 9, f x = ((Real.log x / Real.log 3) ^ 2 - 2 * (Real.log x / Real.log 3) - 3)
  ∧ ∃ x ∈ Icc (1 / 27 : ℝ) 9, f x = -4
  ∧ ∃ x ∈ Icc (1 / 27 : ℝ) 9, f x = 12 := 
sorry

end min_max_f_l195_195908


namespace binomial_theorem_l195_195046

theorem binomial_theorem (x y : ℝ) (n : ℕ) :
  (x + y)^n = ∑ k in Finset.range (n + 1), (Nat.choose n k : ℝ) * x^k * y^(n - k) :=
by
  sorry

end binomial_theorem_l195_195046


namespace solve_eqn_l195_195431

theorem solve_eqn (x : ℝ) (h1 : x + real.sqrt (3 * x - 2) = 6) : x = (15 - real.sqrt 73) / 2 :=
sorry

end solve_eqn_l195_195431


namespace unique_root_iff_l195_195083

def has_unique_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), (a * y^2 + 2 * y - 1 = 0 ↔ y = x)

theorem unique_root_iff (a : ℝ) : has_unique_solution a ↔ (a = 0 ∨ a = 1) := 
sorry

end unique_root_iff_l195_195083


namespace remainder_when_divided_by_x_minus_3_l195_195255

def p (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7

theorem remainder_when_divided_by_x_minus_3 : p 3 = 52 := 
by
  -- proof here
  sorry

end remainder_when_divided_by_x_minus_3_l195_195255


namespace hyperbola_eccentricity_l195_195702

variables {a b c e : ℝ}

-- Condition: a > 0 and b > 0
axiom h_a_pos : a > 0
axiom h_b_pos : b > 0

-- Condition: Hyperbola equation
axiom hyperbola_cond : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1

-- Condition: Asymptote equations imply b = 2a
axiom asymptote_cond : b = 2 * a

-- Define semi-conjugate axis
def semi_conjugate_axis (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

-- Define eccentricity
def eccentricity (c a : ℝ) : ℝ := c / a

-- Statement asserting the eccentricity of the hyperbola is sqrt(5)
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = sqrt 5 ∧
  (let c := semi_conjugate_axis a b in
  eccentricity c a = e) :=
sorry


end hyperbola_eccentricity_l195_195702


namespace time_to_pass_bridge_correct_l195_195176

variables (length_train length_bridge speed_train_kmph : ℕ)
variable (speed_conversion_factor : ℚ)

-- Given Conditions
def length_train : ℕ := 360
def length_bridge : ℕ := 140
def speed_train_kmph : ℕ := 72
def speed_conversion_factor : ℚ := (1000 : ℚ) / 3600

-- Definitions
def total_distance := length_train + length_bridge
def speed_train_mps := speed_train_kmph * speed_conversion_factor
def time_to_pass_bridge := total_distance / speed_train_mps

-- Prove Time to pass the bridge
theorem time_to_pass_bridge_correct : time_to_pass_bridge = 25 := 
by
  -- mathematical proof steps here
  sorry

end time_to_pass_bridge_correct_l195_195176


namespace smallest_k_exists_l195_195257

theorem smallest_k_exists :
  ∃ k : ℕ, (∀ n > 2, a_n = a_{n-1} + a_{n-2}) ∧ (a_9 = k) ∧ 
           (more_than_one_sequence_pos_int a) ∧
           (nondecreasing a) ∧ 
           (k = 748) :=
begin
  sorry
end

end smallest_k_exists_l195_195257


namespace balloon_arrangements_correct_l195_195213

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l195_195213


namespace accounting_balance_l195_195773

theorem accounting_balance :
  ∀ (q x p : ℂ), 3 * q - x = 15000 ∧ q = 7 ∧ x = 7 + 75*complex.I → p = 5005 + 25*complex.I :=
by
  intros q x p h
  sorry

end accounting_balance_l195_195773


namespace count_ordered_pairs_l195_195205

theorem count_ordered_pairs :
  {p : ℤ × ℤ | p.1 * p.2 ≥ 0 ∧ p.1 ^ 3 + p.2 ^ 3 + 25 * p.1 * p.2 = 15 ^ 3}.to_finset.card = 17 := 
by {
  sorry
}

end count_ordered_pairs_l195_195205


namespace length_of_AB_area_of_ΔABF1_l195_195692

theorem length_of_AB (A B : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3)) →
  |((x1 - x2)^2 + (y1 - y2)^2)^(1/2)| = (8 / 3) * (2)^(1/2) :=
by sorry

theorem area_of_ΔABF1 (A B F1 : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (F1 = (0, -2)) ∧ ((y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3))) →
  (1/2) * (((x1 - x2)^2 + (y1 - y2)^2)^(1/2)) * (|(-2-2)/((2)^(1/2))|) = 16 / 3 :=
by sorry

end length_of_AB_area_of_ΔABF1_l195_195692


namespace keys_per_lock_l195_195097

-- Define the given conditions
def num_complexes := 2
def apartments_per_complex := 12
def total_keys := 72

-- Calculate the total number of apartments
def total_apartments := num_complexes * apartments_per_complex

-- The theorem statement to prove
theorem keys_per_lock : total_keys / total_apartments = 3 := 
by
  sorry

end keys_per_lock_l195_195097


namespace find_x_l195_195004

variable (x : ℤ)
def A : Set ℤ := {x^2, x + 1, -3}
def B : Set ℤ := {x - 5, 2 * x - 1, x^2 + 1}

theorem find_x (h : A x ∩ B x = {-3}) : x = -1 :=
sorry

end find_x_l195_195004


namespace perimeter_is_22_l195_195078

-- Definitions for the dimensions of the rectangle
def length : ℕ := 6 -- length in cm
def width : ℕ := 5 -- width in cm

-- Calculate the perimeter using the given dimensions
def perimeter (l w : ℕ) := 2 * (l + w)

-- Theorem statement: proving the perimeter of the rectangle is 22 cm
theorem perimeter_is_22 : perimeter length width = 22 :=
by {
  -- Calculation statements with evidence from problem
  have h1 : length + width = 11 := rfl,
  have h2 : 2 * (length + width) = 22 := by rw [h1]; rfl,
  -- Concluding the proof
  exact h2,
}

end perimeter_is_22_l195_195078


namespace assembly_line_arrangement_l195_195598

def total_ways_to_arrange : Nat :=
  let factorial (n : Nat) : Nat :=
    if n = 0 then 1 else n * factorial (n-1)
  factorial 5

theorem assembly_line_arrangement : total_ways_to_arrange = 120 := by
  sorry

end assembly_line_arrangement_l195_195598


namespace function_local_minimum_has_b_in_interval_l195_195728

theorem function_local_minimum_has_b_in_interval :
  (∃ x ∈ Ioo (0 : ℝ) 1, has_local_min f x) → (b ∈ Ioo (0 : ℝ) 1) :=
by
  let f (x : ℝ) : ℝ := x^3 - 3 * b * x + b
  sorry

end function_local_minimum_has_b_in_interval_l195_195728


namespace negation_proof_l195_195456

theorem negation_proof :
  (¬ ∃ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∀ x : ℝ, x > 1 ∧ x^2 ≤ 4) :=
by
  sorry

end negation_proof_l195_195456


namespace invertible_interval_l195_195623

def f (x : ℝ) := 3 * x^2 - 6 * x - 9

theorem invertible_interval (x : ℝ) (h : x = 2) : 
  ∃ I, I = set.Ici 1 ∧ x ∈ I ∧ function.injective (f ∘ (λ y, y)) :=
sorry

end invertible_interval_l195_195623


namespace djibo_age_sum_years_ago_l195_195238

theorem djibo_age_sum_years_ago (x : ℕ) (h₁: 17 - x + 28 - x = 35) : x = 5 :=
by
  -- proof is omitted as per instructions
  sorry

end djibo_age_sum_years_ago_l195_195238


namespace remainder_proof_l195_195928

theorem remainder_proof (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := 
by 
  sorry

end remainder_proof_l195_195928


namespace incorrect_transformation_when_c_zero_l195_195896

theorem incorrect_transformation_when_c_zero {a b c : ℝ} (h : a * c = b * c) (hc : c = 0) : a ≠ b :=
by
  sorry

end incorrect_transformation_when_c_zero_l195_195896


namespace functional_eq_solution_l195_195246

theorem functional_eq_solution {f : ℝ → ℝ} (h : ∀ x y : ℝ, f(x)^2 - f(y)^2 = f(x + y) * f(x - y))
  (h_diff : ∀ x : ℝ, differentiable ℝ f) :
  (∃ A B : ℝ, f = λ x, A * x + B) ∨ (∃ A B k : ℝ, f = λ x, A * exp(k * x) + B * exp(-k * x)) :=
sorry

end functional_eq_solution_l195_195246


namespace volume_increase_factor_l195_195895

   variable (π : ℝ) (r h : ℝ)

   def original_volume : ℝ := π * r^2 * h

   def new_height : ℝ := 3 * h

   def new_radius : ℝ := 2.5 * r

   def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

   theorem volume_increase_factor :
     new_volume π r h = 18.75 * original_volume π r h := 
   by
     sorry
   
end volume_increase_factor_l195_195895


namespace sample_weights_of_students_l195_195873

theorem sample_weights_of_students :
  (∃ (students : Fin 100 → ℝ), students.sample == "the weights of 100 students") := 
by
  sorry

end sample_weights_of_students_l195_195873


namespace y_intercept_of_line_b_l195_195017

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l195_195017


namespace andrew_kept_correct_l195_195951

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end andrew_kept_correct_l195_195951


namespace find_lambda_l195_195290

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (λ μ : ℝ)
variable (h_non_coplanar : a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b)
variables (ha : ∥a∥ = 3) (hb : ∥b∥ = 2)
variables (h_sum : λ + μ = 1)
let c := λ • a + μ • b in
variables (h_dot_eq : (c ⬝ b) / ∥b∥ = (c ⬝ a) / ∥a∥)

theorem find_lambda (h_non_coplanar : a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b) (ha : ∥a∥ = 3) (hb : ∥b∥ = 2) (h_sum : λ + μ = 1) (h_dot_eq : (λ • a + μ • b) ⬝ b / ∥b∥ = (λ • a + μ • b) ⬝ a / ∥a∥) :
  λ = 2 / 5 :=
sorry

end find_lambda_l195_195290


namespace mass_percentage_of_Cl_in_compound_l195_195648

theorem mass_percentage_of_Cl_in_compound (mass_percentage_Cl : ℝ) (h : mass_percentage_Cl = 92.11) : mass_percentage_Cl = 92.11 :=
sorry

end mass_percentage_of_Cl_in_compound_l195_195648


namespace unique_a_exists_for_prime_p_l195_195978

theorem unique_a_exists_for_prime_p (p : ℕ) [Fact p.Prime] :
  (∃! (a : ℕ), a ∈ Finset.range (p + 1) ∧ (a^3 - 3*a + 1) % p = 0) ↔ p = 3 := by
  sorry

end unique_a_exists_for_prime_p_l195_195978


namespace y_intercept_of_line_b_l195_195016

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l195_195016


namespace find_a_and_theta_find_max_min_g_l195_195318

noncomputable def f (x a θ : ℝ) : ℝ := (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

-- Provided conditions
variable (a : ℝ)
variable (θ : ℝ)
variable (is_odd : ∀ x, f x a θ = -f (-x) a θ)
variable (f_pi_over_4 : f ((Real.pi) / 4) a θ = 0)
variable (theta_in_range : 0 < θ ∧ θ < Real.pi)

-- To Prove
theorem find_a_and_theta :
  a = -1 ∧ θ = (Real.pi / 2) :=
sorry

-- Define g(x) and its domain
noncomputable def g (x : ℝ) : ℝ := f x (-1) (Real.pi / 2) + f (x + (Real.pi / 3)) (-1) (Real.pi / 2)

-- Provided domain condition
variable (x_in_domain : 0 ≤ x ∧ x ≤ (Real.pi / 4))

-- To Prove maximum and minimum value of g(x)
theorem find_max_min_g :
  (∀ x, x ∈ Set.Icc (0 : ℝ) (Real.pi / 4) → -((Real.sqrt 3) / 2) ≤ g x ∧ g x ≤ (Real.sqrt 3) / 2)
  ∧ ∃ x_min, g x_min = -((Real.sqrt 3) / 2) ∧ x_min = (Real.pi / 8)
  ∧ ∃ x_max, g x_max = ((Real.sqrt 3) / 2) ∧ x_max = (Real.pi / 4) :=
sorry

end find_a_and_theta_find_max_min_g_l195_195318


namespace range_of_a_l195_195288

-- Define the condition p
def p (x : ℝ) : Prop := (2 * x^2 - 3 * x + 1) ≤ 0

-- Define the condition q
def q (x a : ℝ) : Prop := (x^2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0

-- Lean statement for the problem
theorem range_of_a (a : ℝ) : (¬ (∃ x, p x) → ¬ (∃ x, q x a)) → ((0 : ℝ) ≤ a ∧ a ≤ (1 / 2 : ℝ)) :=
by 
  sorry

end range_of_a_l195_195288


namespace part1_part2_l195_195737

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Condition 1 in Lean
axiom triangle_ABC : a * sin C = sqrt 3 * c * cos A

-- First proof problem: Prove A = 60°
theorem part1 (h1 : 0 < A ∧ A < 180) : A = 60 :=
by
  -- Insert required steps or reasoning here
  sorry

-- Second proof problem: Prove given conditions a and c
theorem part2 (h2 : a = 5 * sqrt 3) (h3 : c = 4) : cos (2 * C - A) = (17 + 12 * sqrt 7) / 50 :=
by
  -- Insert required steps or reasoning here
  sorry

end part1_part2_l195_195737


namespace num_supermarkets_us_l195_195089

noncomputable def num_supermarkets_total : ℕ := 84

noncomputable def us_canada_relationship (C : ℕ) : Prop := C + (C + 10) = num_supermarkets_total

theorem num_supermarkets_us (C : ℕ) (h : us_canada_relationship C) : C + 10 = 47 :=
sorry

end num_supermarkets_us_l195_195089


namespace solve_determinant_l195_195664

theorem solve_determinant (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | det !![[x + a, 2*x, 2*x], [2*x, x + a, 2*x], [2*x, 2*x, x + a]] = 0} = {-a, a/3} :=
by
  sorry

end solve_determinant_l195_195664


namespace probability_of_winning_exactly_once_l195_195882

-- Define the probability of player A winning a match
def prob_win_A (p : ℝ) : Prop := (1 - p) ^ 3 = 1 - 63 / 64

-- Define the binomial probability for exactly one win in three matches
def binomial_prob (p : ℝ) : ℝ := 3 * p * (1 - p) ^ 2

theorem probability_of_winning_exactly_once (p : ℝ) (h : prob_win_A p) : binomial_prob p = 9 / 64 :=
sorry

end probability_of_winning_exactly_once_l195_195882


namespace find_value_l195_195783

open Classical

variables (a b c : ℝ)

-- Assume a, b, c are roots of the polynomial x^3 - 24x^2 + 50x - 42
def is_root (x : ℝ) : Prop := x^3 - 24*x^2 + 50*x - 42 = 0

-- Vieta's formulas for the given polynomial
axiom h1 : is_root a
axiom h2 : is_root b
axiom h3 : is_root c
axiom h4 : a + b + c = 24
axiom h5 : a * b + b * c + c * a = 50
axiom h6 : a * b * c = 42

-- We want to prove the given expression equals 476/43
theorem find_value : 
  (a/(1/a + b*c) + b/(1/b + c*a) + c/(1/c + a*b) = 476/43) :=
sorry

end find_value_l195_195783


namespace value_of_B_l195_195870

theorem value_of_B (B : ℝ) : 3 * B ^ 2 + 3 * B + 2 = 29 ↔ (B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2) :=
by sorry

end value_of_B_l195_195870


namespace total_cost_of_pencils_l195_195433

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end total_cost_of_pencils_l195_195433


namespace profit_percentage_is_40_l195_195931

-- Defining the cost price
def cost_price : ℝ := 500

-- Defining the selling price
def selling_price : ℝ := 700

-- Calculating the profit
def profit : ℝ := selling_price - cost_price

-- Calculating the profit percentage
def profit_percentage : ℝ := (profit / cost_price) * 100

-- Proving that the profit percentage is 40%
theorem profit_percentage_is_40 : profit_percentage = 40 := 
by 
  -- We skip the proof here
  sorry

end profit_percentage_is_40_l195_195931


namespace PQ_bisects_AC_l195_195189

-- We define all points and midpoints as per the given conditions
variables {A B C D M N P Q: Type}
variable [geometry E]

-- Midpoints definitions
def is_midpoint (M : E) (A B : E) : Prop := dist M A = dist M B

-- Given conditions
axioms
  (M_midpoint : is_midpoint M A B)
  (C_on_AB : C ∈ open_segment A B)
  (N_midpoint : is_midpoint N C D)
  (P_midpoint : is_midpoint P B D)
  (Q_midpoint : is_midpoint Q M N)

-- The main theorem we need to prove
theorem PQ_bisects_AC : is_midpoint (line_through Q P ∩ segment A C) A C :=
sorry

end PQ_bisects_AC_l195_195189


namespace suitcase_lock_combinations_l195_195942

def is_valid_setting (digits : List ℕ) : Prop :=
  digits.length = 4 ∧
  (digits.nodup) ∧
  (∀ x ∈ digits, x ≥ 0 ∧ x ≤ 9) ∧
  (digits.sum ≤ 20)

theorem suitcase_lock_combinations : 
  {digits : List ℕ // is_valid_setting digits}.card = 1980 :=
sorry

end suitcase_lock_combinations_l195_195942


namespace assembly_line_arrangement_l195_195595

open Nat

theorem assembly_line_arrangement :
  let A := "add axles"
  let W := "add wheels"
  let I := "install windshield"
  let P := "install instrument panel"
  let S := "install steering wheel"
  let C := "install interior seating"
  let tasks := [A, W, I, P, S, C]
  (∃ (order : List String) (H : Multiset Nodup order), -- There exists an order of tasks
    Multiset.card (Multiset.filter (λ t, t = A ∨ t = W) (Multiset.of_list order)) = 2 -- Axles and Wheels both are in order.
    ∧ List.index_of A order < List.index_of W order -- Axles are before Wheels.
  ) →
  (fact 5 = 120) -- The number of valid arrangements is 120
 := by {
  sorry 
}

end assembly_line_arrangement_l195_195595


namespace exists_100_digit_number_divisible_by_sum_of_digits_l195_195640

-- Definitions
def is_100_digit_number (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

-- Main theorem statement
theorem exists_100_digit_number_divisible_by_sum_of_digits :
  ∃ n : ℕ, is_100_digit_number n ∧ no_zero_digits n ∧ is_divisible_by_sum_of_digits n :=
sorry

end exists_100_digit_number_divisible_by_sum_of_digits_l195_195640


namespace cost_of_four_pencils_and_three_pens_l195_195069

variable {p q : ℝ}

theorem cost_of_four_pencils_and_three_pens (h1 : 3 * p + 2 * q = 4.30) (h2 : 2 * p + 3 * q = 4.05) : 4 * p + 3 * q = 5.97 := by
  sorry

end cost_of_four_pencils_and_three_pens_l195_195069


namespace intersection_of_medians_coincide_l195_195852

variable {A1 A2 A3 A4 A5 A6 : Point}

def midpoint (P Q : Point) : Point := sorry

def midpoint_hex (P1 P2 P3 P4 P5 P6 : Point) :=
  (midpoint P1 P2, midpoint P2 P3, midpoint P3 P4, midpoint P4 P5, midpoint P5 P6, midpoint P6 P1)

theorem intersection_of_medians_coincide :
  let (B1, B2, B3, B4, B5, B6) := midpoint_hex A1 A2 A3 A4 A5 A6
  let G1 := centroid (B1, B3, B5)
  let G2 := centroid (B2, B4, B6)
  G1 = G2 := sorry

end intersection_of_medians_coincide_l195_195852


namespace toy_store_bears_shelves_l195_195589

theorem toy_store_bears_shelves (initial_stock shipment bears_per_shelf total_bears number_of_shelves : ℕ)
  (h1 : initial_stock = 17)
  (h2 : shipment = 10)
  (h3 : bears_per_shelf = 9)
  (h4 : total_bears = initial_stock + shipment)
  (h5 : number_of_shelves = total_bears / bears_per_shelf) :
  number_of_shelves = 3 :=
by
  sorry

end toy_store_bears_shelves_l195_195589


namespace bart_pages_bought_l195_195606

theorem bart_pages_bought (total_money : ℝ) (price_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_money = 10) (h2 : price_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_money / price_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_pages_bought_l195_195606


namespace rachel_biology_homework_pages_l195_195824

-- Declare the known quantities
def math_pages : ℕ := 8
def total_math_biology_pages : ℕ := 11

-- Define biology_pages
def biology_pages : ℕ := total_math_biology_pages - math_pages

-- Assert the main theorem
theorem rachel_biology_homework_pages : biology_pages = 3 :=
by 
  -- Proof is omitted as instructed
  sorry

end rachel_biology_homework_pages_l195_195824


namespace parallelogram_area_l195_195439

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (base_condition : b = 8) (altitude_condition : h = 2 * b) : 
  A = 128 :=
by 
  sorry

end parallelogram_area_l195_195439


namespace isosceles_triangles_with_perimeter_100_l195_195076

open Nat

theorem isosceles_triangles_with_perimeter_100 :
  ∃ n : ℕ, n = 24 ∧ (∀ a b c : ℕ, a = b ∧ a + b + c = 100 ∧ 25 < a ∧ a < 50 → True) :=
begin
  sorry
end

end isosceles_triangles_with_perimeter_100_l195_195076


namespace average_marks_l195_195954

theorem average_marks {n : ℕ} (h1 : 5 * 74 + 104 = n * 79) : n = 6 :=
by
  sorry

end average_marks_l195_195954


namespace solve_for_a_l195_195697

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem solve_for_a : 
  (∃ a : ℝ, f (f 0 a) a = 4 * a) → (a = 2) :=
by
  sorry

end solve_for_a_l195_195697


namespace unique_solution_eqn_l195_195656

theorem unique_solution_eqn (a : ℝ) :
  (∃! x : ℝ, 3^(x^2 + 6 * a * x + 9 * a^2) = a * x^2 + 6 * a^2 * x + 9 * a^3 + a^2 - 4 * a + 4) ↔ (a = 1) :=
by
  sorry

end unique_solution_eqn_l195_195656


namespace cosine_of_angle_through_point_l195_195305

theorem cosine_of_angle_through_point :
  ∀ (α : ℝ) (P : ℝ × ℝ),
  P = (-3, 4) →
  cos α = -3 / 5 :=
by
  sorry

end cosine_of_angle_through_point_l195_195305


namespace number_of_valid_6x6_arrays_l195_195711

theorem number_of_valid_6x6_arrays : 
  ∃ A : Array (Array Int), 
    (∀ i, ∑ j in Finset.range 6, A[i][j] = 0) ∧
    (∀ j, ∑ i in Finset.range 6, A[i][j] = 0) ∧
    (∀ i in List.range 3, ∑ j in Finset.range 6, (A[2*i][j] + A[2*i+1][j]) = 0) ∧
    (∀ j in List.range 3, ∑ i in Finset.range 6, (A[i][2*j] + A[i][2*j+1]) = 0) ∧
    (finset.card { A | ⋆ (* conditions defined above *) } = 8000) :=
sorry

end number_of_valid_6x6_arrays_l195_195711


namespace points_Q_are_concyclic_l195_195401

-- Assume we have a cyclic 100-gon
variable (P : Fin 100 → ℝ × ℝ)

-- Assume P_i = P_{i+100} for all i
def cyclic_100_gon (P : Fin 100 → ℝ × ℝ) : Prop :=
  ∀ i : Fin 100, P i = P (⟨i + 100 % 100, sorry⟩)

-- Define Q_i as the intersection point of diagonals
def Q (i : Fin 100) : ℝ × ℝ := sorry

-- Assume there exists a point 'p' such that PP_i is perpendicular to P_(i-1)P_(i+1)
def exists_perpendicular_point (P : Fin 100 → ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ i : Fin 100, sorry -- define the perpendicular condition properly using inner_product

-- Main theorem
theorem points_Q_are_concyclic (P : Fin 100 → ℝ × ℝ) (p : ℝ × ℝ)
  (h_cyclic : cyclic_100_gon P)
  (h_perp : exists_perpendicular_point P p) :
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ i : Fin 100, dist (Q P i) c = r :=
sorry

end points_Q_are_concyclic_l195_195401


namespace solution_set_for_log_inequality_l195_195681

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem solution_set_for_log_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∃ x : ℝ, f x = a^(Real.log (x^2 - 2*x + 3) / Real.log 10)): 
  {x : ℝ | log_a a (x^2 - 5*x + 7) > 0} = {x : ℝ | 2 < x ∧ x < 3} := 
by
  sorry

end solution_set_for_log_inequality_l195_195681


namespace solve_for_x_l195_195833

theorem solve_for_x : 
  ∃ x : ℝ, (x^2 + 6 * x + 8 = -(x + 2) * (x + 6)) ∧ (x = -2 ∨ x = -5) :=
sorry

end solve_for_x_l195_195833


namespace log_sub_eval_l195_195240

theorem log_sub_eval : 
  (∀ a, 2 ^ a = 16 → a = log 2 16) → 
  (∀ b, 2 ^ b = (1 / 4) → b = log 2 (1 / 4)) → 
  log 2 16 - log 2 (1 / 4) = 6 :=
by
  intros h1 h2
  exact sorry

end log_sub_eval_l195_195240


namespace solve_abs_equation_l195_195058

theorem solve_abs_equation (y : ℝ) (h : |y - 8| + 3 * y = 12) : y = 2 :=
sorry

end solve_abs_equation_l195_195058


namespace unhappy_redheads_ratio_l195_195399

theorem unhappy_redheads_ratio (x y z : ℕ) (h1 : 40 * x = 60 * y) (h2 : z = x + (2 / 5) * y) :
  let t := x / 3 in
  let k := t / 5 in
  let unhappy_y := (2 / 5) * y in
  (unhappy_y / z : ℚ) = 4 / 19 :=
by
  sorry

end unhappy_redheads_ratio_l195_195399


namespace center_of_circle_l195_195627

theorem center_of_circle (x y : ℝ) : (x - 3)^2 + (y + 4)^2 = 25 ↔ x^2 + y^2 - 6x + 8y = 0 :=
by
    intro h
    -- completing the square to prove equivalence
    suffices : (x - 3)^2 + (y + 4)^2 = x^2 + y^2 - 6x + 8y + 25
    exact this.trans (by ring) ▸ by_ring
    intro h
    -- reverting from standard form to expanded form
    suffices : x^2 + y^2 - 6x + 8y = (x - 3)^2 + (y + 4)^2 - 25
    exact this.symm.trans (by ring) ▸ by_ring

end center_of_circle_l195_195627


namespace range_of_x_for_f_l195_195842

theorem range_of_x_for_f (f : ℝ → ℝ) 
  (H_even : ∀ x, f x = f (-x)) 
  (H_decreasing : ∀ x y, x < y ∧ y < 0 → f x > f y) 
  (H_condition : f (-1) < f ((x : ℝ)^2)) : 
  x ∈ set.Ioo (-∞) (-1 : ℝ) ∪ set.Ioo 1 (∞) :=
by
  sorry

end range_of_x_for_f_l195_195842


namespace range_of_a_l195_195844

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 2*a*x else (2*a - 1)*x - 3*a + 6

theorem range_of_a :
  { a : ℝ // ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f a x1 - f a x2) > 0 } = { a | 1 ≤ a ∧ a ≤ 2 } :=
sorry

end range_of_a_l195_195844


namespace balloon_permutations_l195_195221

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l195_195221


namespace ratio_wealth_citizen_XY_l195_195972

noncomputable def wealth_ratio_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : ℝ :=
  let pop_X := 0.4 * P
  let wealth_X_before_tax := 0.5 * W
  let tax_X := 0.1 * wealth_X_before_tax
  let wealth_X_after_tax := wealth_X_before_tax - tax_X
  let wealth_per_citizen_X := wealth_X_after_tax / pop_X

  let pop_Y := 0.3 * P
  let wealth_Y := 0.6 * W
  let wealth_per_citizen_Y := wealth_Y / pop_Y

  wealth_per_citizen_X / wealth_per_citizen_Y

theorem ratio_wealth_citizen_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : 
  wealth_ratio_XY P W h1 h2 = 9 / 16 := 
by
  sorry

end ratio_wealth_citizen_XY_l195_195972


namespace al_average_speed_l195_195590

variables (downhill_speed : ℕ) (downhill_time : ℕ)
variables (uphill_speed : ℕ) (uphill_time : ℕ)
variables (total_distance : ℕ) (total_time : ℕ) (average_speed : ℕ)

-- Let conditions be:
def condition1 := downhill_speed = 20
def condition2 := downhill_time = 2
def condition3 := uphill_speed = 4
def condition4 := uphill_time = 6
def condition5 := total_distance = (downhill_speed * downhill_time) + (uphill_speed * uphill_time)
def condition6 := total_time = downhill_time + uphill_time
def condition7 := average_speed = total_distance / total_time

-- Prove that the average speed is 8 miles per hour given the conditions
theorem al_average_speed :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 → average_speed = 8 :=
by
  sorry

end al_average_speed_l195_195590


namespace num_intersection_points_l195_195971

-- Define the equations of the lines as conditions
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- The theorem to prove the number of intersection points
theorem num_intersection_points :
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨ (line2 p.1 p.2 ∧ line3 p.1 p.2) :=
sorry

end num_intersection_points_l195_195971


namespace running_time_is_five_thirds_l195_195140

-- Define the conditions
def walking_speed : ℝ := 5 -- km/h
def walking_time : ℝ := 5 -- hours

-- Define the running speed
def running_speed : ℝ := 15 -- km/h

-- Define the distance covered while walking
def distance_covered : ℝ := walking_speed * walking_time

-- Define the time to cover the same distance while running
def running_time : ℝ := distance_covered / running_speed

-- State the theorem
theorem running_time_is_five_thirds :
  running_time = 5 / 3 := by
  sorry

end running_time_is_five_thirds_l195_195140


namespace solve_mod_equation_l195_195249

theorem solve_mod_equation (x : ℤ) (h : 10 * x + 3 ≡ 7 [ZMOD 18]) : x ≡ 4 [ZMOD 9] :=
sorry

end solve_mod_equation_l195_195249


namespace tax_free_amount_is_600_l195_195588

variable (X : ℝ) -- X is the tax-free amount

-- Given conditions
variable (total_value : ℝ := 1720)
variable (tax_paid : ℝ := 89.6)
variable (tax_rate : ℝ := 0.08)

-- Proof problem
theorem tax_free_amount_is_600
  (h1 : 0.08 * (total_value - X) = tax_paid) :
  X = 600 :=
by
  sorry

end tax_free_amount_is_600_l195_195588


namespace balloon_permutations_l195_195232

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l195_195232


namespace problem_1_problem_2_problem_3_l195_195672

noncomputable def a_n (n : ℕ) (hn : n > 0) : ℝ := n

noncomputable def S_n (n : ℕ) (hn : n > 0) : ℝ := (1 / 2) * (a_n n hn) ^ 2 + (1 / 2) * (a_n n hn)

theorem problem_1 (n : ℕ) (hn : n > 0) : a_n n hn = n := 
sorry

noncomputable def b_n (n : ℕ) (hn : n > 0) : ℝ := (2 * n - 1) * 2^n

noncomputable def T_n (n : ℕ) (hn : n > 0) : ℝ := 6 + (2 * n - 3) * 2^(n + 1)

theorem problem_2 (n : ℕ) (hn : n > 0) : T_n n hn = 6 + (2 * n - 3) * 2^(n + 1) :=
sorry

noncomputable def c_n (n : ℕ) (hn : n > 0) : ℝ := (4 * n - 6) / (T_n n hn - 6) - 1 / (a_n n hn * a_n (n + 1) (by norm_num))

theorem problem_3 (a : ℝ) : (∀ (n : ℕ) (hn : n > 0), ∃ (x0 : ℝ) (h0 : x0 ∈ (-1/2, 1/2)), 
  ∑ i in range n, c_n i (by norm_num) ≤ f x0 - a) → a ≤ 19 / 80 :=
sorry

end problem_1_problem_2_problem_3_l195_195672


namespace find_B_l195_195177

-- Define the conditions for the polynomial.
def polynomial (z : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), 
  (z = a * b * c * d * e * f) ∧ 
  (a + b + c + d + e + f = 12) ∧ 
  (1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e ∧ 1 ≤ f)

-- Define the polynomial's roots being positive integers resulting in desired output.
theorem find_B 
  (roots : list ℤ)
  (h₁ : ∀ r ∈ roots, r > 0) 
  (h₂ : ∀ s : ℤ, polynomial s)
  (h₃ : roots.sum = 12) :
  let B := - (roots.map (λ (r : ℤ), r * r * r).sum)
  in B = -122 :=
by
  sorry

end find_B_l195_195177


namespace original_silk_proof_l195_195360

def original_amount_of_raw_silk (dried_silk_loss_pounds : ℚ) (dried_silk_pounds : ℚ) (original_silk_pounds : ℚ) : ℚ :=
  let loss := dried_silk_loss_pounds
  let new_weight := original_silk_pounds - loss
  let proportion := dried_silk_pounds * original_silk_pounds = 30 * new_weight
  proportion

theorem original_silk_proof :
  original_amount_of_raw_silk (3 + 12/16) 12 (96/7) := 
by
  sorry

end original_silk_proof_l195_195360


namespace coefficient_x8_in_expansion_l195_195645

noncomputable def binomial_expansion_coefficient (a b : ℝ) (n k : ℕ) : ℝ :=
  (nat.choose n k) * (a ^ (n - k)) * (b ^ k)

theorem coefficient_x8_in_expansion : 
  let a := (x : ℝ) ^ 3 / 3
  let b := -3 / (x : ℝ) ^ 2 
  ∀(x : ℝ), coefficient_x8_in_expansion (a - b) 9 = 0 :=
by
  sorry

end coefficient_x8_in_expansion_l195_195645


namespace sequence_formula_l195_195276

theorem sequence_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) ∧ (∀ n, 2 * (S n) = a (n + 1)) ∧ (∀ n, S n = ∑ k in range (n + 1), a k) →
  ∀ n, a (n + 1) = 2 * 3 ^ (n - 1) :=
by
  sorry

end sequence_formula_l195_195276


namespace tom_total_payment_l195_195878

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end tom_total_payment_l195_195878


namespace triple_conditions_l195_195142

theorem triple_conditions (a m n : ℕ) (a_positive : a > 0) (m_positive : m > 0) (n_positive : n > 0) :
  (a ^ m + 1) ∣ (a + 1) ^ n ↔ 
  ((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
begin
  sorry
end

end triple_conditions_l195_195142


namespace intersection_of_A_and_B_l195_195551

def A : Set ℤ := {-1, 0, 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {0, -1} :=
by sorry

end intersection_of_A_and_B_l195_195551


namespace power_subtraction_l195_195087

theorem power_subtraction : 2^4 - 2^3 = 2^3 := by
  sorry

end power_subtraction_l195_195087


namespace binary_arithmetic_l195_195256

theorem binary_arithmetic :
    let a := 0b1011101
    let b := 0b1101
    let c := 0b101010
    let d := 0b110
    ((a + b) * c) / d = 0b1110111100 :=
by
  sorry

end binary_arithmetic_l195_195256


namespace present_value_l195_195905

theorem present_value (BD TD PV : ℝ) (hBD : BD = 42) (hTD : TD = 36)
  (h : BD = TD + (TD^2 / PV)) : PV = 216 :=
sorry

end present_value_l195_195905


namespace least_product_of_distinct_primes_over_30_l195_195501

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l195_195501


namespace max_subset_count_l195_195710

/-- 
Given vectors x₁, x₂, ..., xₙ such that ∥xᵢ∥ ≥ 1 for all i,
and a vector x, the maximum number of vectors x_A = ∑ᵢ ∈ A xᵢ
for which ∥x_A - x∥ < 1 / 2 is binom n (⌊n / 2⌋).
-/
theorem max_subset_count (n : ℕ) (x : ℝ) (x_i : ℕ → ℝ)
  (h : ∀ i, ∥x_i i∥ ≥ 1) : 
  ∃ A : finset (fin n), ∥A.sum x_i - x∥ < 1 / 2
  → finset.card A = nat.choose n (nat.floor (n / 2)) :=
sorry

end max_subset_count_l195_195710


namespace complex_conjugate_identity_l195_195721

theorem complex_conjugate_identity (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
by sorry

end complex_conjugate_identity_l195_195721


namespace arcsin_sqrt2_div_2_eq_pi_div_4_l195_195618

theorem arcsin_sqrt2_div_2_eq_pi_div_4 : Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
by
  -- proof goes here
  sorry

end arcsin_sqrt2_div_2_eq_pi_div_4_l195_195618


namespace particular_integral_solve_diff_eq_l195_195148

-- Problem 1
theorem particular_integral (y : ℝ → ℝ) (y' : ℝ → ℝ → ℝ) (C x : ℝ)
  (h_diff_eq : ∀ x, y x * (y' x) + (Real.cot x) * (y' x) = 0)
  (h_initial_cond : y (Real.pi / 3) = -1) :
  y = (λ x, -2 * Real.cos x) :=
by
  sorry

-- Problem 2
theorem solve_diff_eq (s s_t : ℝ → ℝ) (C t : ℝ)
  (h_diff_eq : ∀ t, s t = (s_t t) * (Real.cos t) ^ 2 * Real.log (s t))
  (h_initial_cond : s Real.pi = 1) :
  ∀ t, (Real.log (s t)) ^ 2 = 2 * Real.tan t :=
by
  sorry

end particular_integral_solve_diff_eq_l195_195148


namespace equation1_solutions_equation2_solutions_equation3_solutions_l195_195061

noncomputable def solution1 : set ℝ := { x | x^2 - 9 = 0 }

theorem equation1_solutions :
  solution1 = {3, -3} :=
by
  sorry

noncomputable def solution2 : set ℝ := { x | x * (x - 2) = 5 * (2 - x) }

theorem equation2_solutions :
  solution2 = {2, -5} :=
by
  sorry

noncomputable def solution3 : set ℝ := { x | x^2 - 4x - 2 = 0 }

theorem equation3_solutions :
  solution3 = {2 + real.sqrt 6, 2 - real.sqrt 6} :=
by
  sorry

end equation1_solutions_equation2_solutions_equation3_solutions_l195_195061


namespace unique_g_l195_195781

noncomputable def T : Set ℝ := {x | x ≠ 0}
noncomputable def g (x : T) : ℝ := sorry

axiom g_1 : g 1 = 2
axiom g_2 : ∀ x y : ℝ, (x ∈ T) → (y ∈ T) → (x + y ∈ T) → g (⟨1 / (x + y), sorry⟩) = g (⟨1 / x, sorry⟩) + g (⟨1 / y, sorry⟩) + 1
axiom g_3 : ∀ x y : ℝ, (x ∈ T) → (y ∈ T) → (x + y ∈ T) → 2 * (x + y) * g ⟨x + y, sorry⟩ = x * y * g ⟨x, sorry⟩ * g ⟨y, sorry⟩

theorem unique_g : ∃! (f : T → ℝ), forall x, g x = f x := sorry

end unique_g_l195_195781


namespace fifth_patient_cure_rate_l195_195563

-- Define the cure rate for any patient
def cure_rate (h : ℕ → Prop) (n : ℕ) : Prop := h n = (1 / 5)

-- Condition: The probability of any given patient being cured is 1/5
axiom cure_rate_constant : ∀ (n : ℕ), cure_rate (λ n, 1 / 5) n

-- Condition: The outcomes of previous patients' treatments are independent events
axiom independent_events : ∀ (n m : ℕ), m ≠ n → (cure_rate (λ n, 1 / 5) n ∧ cure_rate (λ m, 1 / 5) m)

-- Task: Prove that the cure rate for the fifth patient is 1/5
theorem fifth_patient_cure_rate : cure_rate (λ n, 1 / 5) 5 :=
by 
sor-- Lean expects a proof here, but we'll skip it with 'sorry'

end fifth_patient_cure_rate_l195_195563


namespace decreasing_intervals_max_value_on_interval_l195_195316

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 2 + 2 * sqrt 3 * sin (x + π / 4) * cos (x - π / 4) - cos x ^ 2 - sqrt 3

theorem decreasing_intervals (k : ℤ) :
  ∀ x, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 → ∀ x1 x2, x1 ≤ x2 → f x1 ≥ f x2 :=
sorry

theorem max_value_on_interval :
  ∃ x, -π / 12 ≤ x ∧ x ≤ 25 * π / 36 ∧ f x = 2 :=
sorry

end decreasing_intervals_max_value_on_interval_l195_195316


namespace balloon_permutations_l195_195211

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l195_195211


namespace triangle_altitude_length_l195_195744

theorem triangle_altitude_length (h : ℝ) : 
  let a₁ a₂ a₃ := (1 : ℝ), (5 : ℝ), (6 : ℝ)
  let s₁ s₂ s₃ := 15 * π / 180, 75 * π / 180, π / 2
  angles_sum : a₁ + a₂ + a₃ = 12 
  (longest_side : ℝ) := 12
  (hypotenuse : ℝ) := 12
  h = hypotenuse * sin (s₁) * cos(s₁)

  (h = 3) := 
by
  sorry

end triangle_altitude_length_l195_195744


namespace monotonicity_f_l195_195980

def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

theorem monotonicity_f : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 0 → f x1 < f x2 := by
  sorry

end monotonicity_f_l195_195980


namespace last_three_digits_of_product_l195_195471

theorem last_three_digits_of_product (A B C : ℕ) (hA : A > 1000) (hB : B > 1000) (hC : C > 1000) :
  (A % 10 = (B + C) % 10) ∧ (B % 10 = (A + C) % 10) ∧ (C % 10 = (A + B) % 10) →
  ∃ d ∈ ({000, 250, 500, 750} : set ℕ), (A * B * C) % 1000 = d :=
begin
  sorry
end

end last_three_digits_of_product_l195_195471


namespace inequality_always_holds_l195_195404

noncomputable def range_for_inequality (k : ℝ) : Prop :=
  0 < k ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)

theorem inequality_always_holds (x y k : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y = k) :
  (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2 ↔ range_for_inequality k :=
sorry

end inequality_always_holds_l195_195404


namespace solve_for_x_l195_195834

theorem solve_for_x : 
    ∃ x : ℚ, 6 * (3 * x - 1) + 7 = -3 * (2 - 5 * x) - 4 ∧ x = -11 / 3 :=
begin
  sorry,
end

end solve_for_x_l195_195834


namespace average_speed_approx_l195_195637

def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

def next_highest_palindrome (n : ℕ) : ℕ :=
  Nat.find (λ m, m > n ∧ is_palindrome m)

theorem average_speed_approx 
  (initial : ℕ := 12321) 
  (final : ℕ := 12421)
  (time : ℕ := 3)
  (hp_initial : is_palindrome initial)
  (hp_final : final = next_highest_palindrome initial) :
  (float.of_nat (final - initial)) / (float.of_nat time) ≈ 33.33 :=
by
  sorry

end average_speed_approx_l195_195637


namespace divide_square_sides_l195_195369

noncomputable def is_rect_outline_possible (prime1 prime2 prime3 prime4 : ℕ) : Prop :=
  ∀ (AB BC CD DA : list ℚ),
      AB.length = 100 ∧ BC.length = 100 ∧ CD.length = 100 ∧ DA.length = 100 →
      (∀ M1 M2 : list ℚ, M1 ⊆ AB ∧ M2 ⊆ BC ∧ M1 ≠ M2 → ∑ M1 ≠ ∑ M2) →
      (∀ M1 M2 : list ℚ, M1 ⊆ BC ∧ M2 ⊆ CD ∧ M1 ≠ M2 → ∑ M1 ≠ ∑ M2) →
      (∀ M1 M2 : list ℚ, M1 ⊆ CD ∧ M2 ⊆ DA ∧ M1 ≠ M2 → ∑ M1 ≠ ∑ M2) →
      (∀ M1 M2 : list ℚ, M1 ⊆ DA ∧ M2 ⊆ AB ∧ M1 ≠ M2 → ∑ M1 ≠ ∑ M2) →
      (∀ (rect : list (list ℚ)), rect.length = 4 → (rect = [AB, BC, CD, DA] ∨ rect = [DA, CD, BC, AB]))

theorem divide_square_sides : ∃ prime1 prime2 prime3 prime4 : ℕ, is_rect_outline_possible prime1 prime2 prime3 prime4 :=
begin
  sorry
end

end divide_square_sides_l195_195369


namespace tax_rate_correct_l195_195582

noncomputable def tax_rate (total_payroll : ℕ) (tax_free_payroll : ℕ) (tax_paid : ℕ) : ℚ :=
  if total_payroll > tax_free_payroll 
  then (tax_paid : ℚ) / (total_payroll - tax_free_payroll) * 100
  else 0

theorem tax_rate_correct :
  tax_rate 400000 200000 400 = 0.2 :=
by
  sorry

end tax_rate_correct_l195_195582


namespace sin_arithmetic_mean_inequality_l195_195397

variable {n : ℕ} (a : Fin n → ℝ)

def mean (a : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, a i) / n

theorem sin_arithmetic_mean_inequality 
  (h : ∀ i : Fin n, 0 < a i ∧ a i < π) :
  let μ := mean a in
  ∏ i, (Real.sin (a i)) / (a i) ≤ (Real.sin μ / μ) ^ n :=
by
  sorry

end sin_arithmetic_mean_inequality_l195_195397


namespace odd_n_divides_3n_plus_1_is_1_l195_195244

theorem odd_n_divides_3n_plus_1_is_1 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) (h3 : n ∣ 3^n + 1) : n = 1 :=
sorry

end odd_n_divides_3n_plus_1_is_1_l195_195244


namespace general_formula_sequence_sum_first_n_terms_l195_195552

-- Define the axioms or conditions of the arithmetic sequence
axiom a3_eq_7 : ∃ a1 d : ℝ, a1 + 2 * d = 7
axiom a5_plus_a7_eq_26 : ∃ a1 d : ℝ, (a1 + 4 * d) + (a1 + 6 * d) = 26

-- State the theorem for the general formula of the arithmetic sequence
theorem general_formula_sequence (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, a1 + (n - 1) * d = 2 * n + 1 :=
sorry

-- State the theorem for the sum of the first n terms of the arithmetic sequence
theorem sum_first_n_terms (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, n * (a1 + (n - 1) * d + a1) / 2 = (n^2 + 2 * n) :=
sorry

end general_formula_sequence_sum_first_n_terms_l195_195552


namespace extra_apples_correct_l195_195459

def num_red_apples : ℕ := 6
def num_green_apples : ℕ := 15
def num_students : ℕ := 5
def num_apples_ordered : ℕ := num_red_apples + num_green_apples
def num_apples_taken : ℕ := num_students
def num_extra_apples : ℕ := num_apples_ordered - num_apples_taken

theorem extra_apples_correct : num_extra_apples = 16 := by
  sorry

end extra_apples_correct_l195_195459


namespace length_of_CE_l195_195760

theorem length_of_CE 
  (A B C D E F : Type)
  (AB DEF : Type)
  (DE CE CB : ℝ)
  (h_AB : AB = (8 : ℝ))
  (h_parallel : DEF ∥ AB)
  (h_intersect_AC : DEF ∩ AC = D)
  (h_intersect_BC : DEF ∩ BC = E)
  (h_bisect : ∀ x, extension AE x ∧ bisects x FEC → ∀ y, bisects y FEC)
  (h_DE : DE = (5 : ℝ))
  (h_similar : ∀ x, similar ∆CDE ∆CAB)
  (h_CE_DE_AB_ratio : CE / (CB : ℝ) = DE / AB)
  (h_CB_equation : CB = CE + 8) :
  CE = (40 / 3 : ℝ) :=
sorry

end length_of_CE_l195_195760


namespace trigonometric_identity_quadratic_solution_l195_195145

theorem trigonometric_identity :
  cos (30 * Real.pi / 180) * tan (60 * Real.pi / 180) - 2 * sin (45 * Real.pi / 180) 
  = (3 / 2 : ℝ) - Real.sqrt 2 :=
by sorry

theorem quadratic_solution (x : ℝ) :
  (3 * x^2 + 2 * x - 1 = 0) ↔ (x = 1/3 ∨ x = -1) :=
by sorry

end trigonometric_identity_quadratic_solution_l195_195145


namespace solve_for_k_l195_195689

noncomputable def k_value (k : ℝ) : Prop := 
  let a : ℝ × ℝ × ℝ := (1, 1, 2)
  let b : ℝ × ℝ × ℝ := (-1, k, 3)
  let dot_product : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  dot_product = 0 → k = -5

-- Full definition for the proof statement
theorem solve_for_k : k_value (-5) :=
  sorry

end solve_for_k_l195_195689


namespace subset_implies_a_ge_2_l195_195003

theorem subset_implies_a_ge_2 (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → x ≤ a) → a ≥ 2 :=
by sorry

end subset_implies_a_ge_2_l195_195003


namespace power_of_two_with_many_nines_exists_l195_195048

theorem power_of_two_with_many_nines_exists (k : ℕ) (h : 1 < k) : 
  ∃ n : ℕ, 
    let last_k_digits := nat.digits 10 (2 ^ n) in
    let num_nines := (last_k_digits.take k).count 9 in
    num_nines ≥ k / 2 :=
by sorry

end power_of_two_with_many_nines_exists_l195_195048


namespace num_perfect_square_factors_of_360_l195_195330

theorem num_perfect_square_factors_of_360 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d : ℕ, d ∣ 360 → (∀ p e, p^e ∣ d → (p = 2 ∨ p = 3 ∨ p = 5) ∧ e % 2 = 0) :=
by
  sorry

end num_perfect_square_factors_of_360_l195_195330


namespace distance_between_vertices_l195_195789

-- Definitions of the vertices from the conditions
def vertex_1 : ℝ × ℝ := (2, 1)
def vertex_2 : ℝ × ℝ := (-3, 4)

-- Definition of the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The theorem we need to prove
theorem distance_between_vertices : distance vertex_1 vertex_2 = real.sqrt 34 :=
by
  -- The proof steps go here
  sorry

end distance_between_vertices_l195_195789


namespace mike_spent_on_speakers_l195_195803

theorem mike_spent_on_speakers :
  let S_total : ℝ := 224.87
  let S_tires : ℝ := 106.33
  let S_speakers := S_total - S_tires
  S_speakers = 118.54 :=
by
  let S_total : ℝ := 224.87
  let S_tires : ℝ := 106.33
  have S_speakers := S_total - S_tires
  show S_speakers = 118.54 from sorry

end mike_spent_on_speakers_l195_195803


namespace average_speed_proof_l195_195902

def distance1 : ℝ := 225
def time1 : ℝ := 3.5
def distance2 : ℝ := 370
def time2 : ℝ := 5

def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2

def average_speed : ℝ := total_distance / total_time

theorem average_speed_proof : average_speed = 70 := by
  sorry

end average_speed_proof_l195_195902


namespace Cary_final_salary_l195_195968

def initial_salary : ℝ := 10
def raise_percentage : ℝ := 0.20
def cut_percentage : ℝ := 0.75

theorem Cary_final_salary :
  let raise_amount := raise_percentage * initial_salary in
  let new_salary_after_raise := initial_salary + raise_amount in
  let final_salary := cut_percentage * new_salary_after_raise in
  final_salary = 9 := by
  sorry

end Cary_final_salary_l195_195968


namespace solution_cos_eq_l195_195643

open Real

theorem solution_cos_eq (x : ℝ) :
  (cos x)^2 + (cos (2 * x))^2 + (cos (3 * x))^2 = 1 ↔
  (∃ k : ℤ, x = k * π / 2 + π / 4) ∨ (∃ k : ℤ, x = k * π / 3 + π / 6) :=
by sorry

end solution_cos_eq_l195_195643


namespace rewrite_expression_l195_195420

theorem rewrite_expression : ∀ x : ℝ, x^2 + 4 * x + 1 = (x + 2)^2 - 3 :=
by
  intros
  sorry

end rewrite_expression_l195_195420


namespace kenneth_fabric_amount_l195_195038

theorem kenneth_fabric_amount :
  ∃ K : ℤ, (∃ N : ℤ, N = 6 * K ∧ (K * 40 + 140000 = N * 40) ∧ K > 0) ∧ K = 700 :=
by
  sorry

end kenneth_fabric_amount_l195_195038


namespace andrew_kept_correct_l195_195950

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end andrew_kept_correct_l195_195950


namespace octagon_area_difference_l195_195445

-- Define the regular octagon properties and regions
def regular_octagon_side_length := 1
def central_square_area := 1
def right_angled_isosceles_triangle_area := 1 / 4

-- Gray region includes two rectangles and three right-angled isosceles triangles 
def grey_region_area := 2 * (area of rectangles not required for this proof) + 3 * right_angled_isosceles_triangle_area

-- Hatched region includes two rectangles and the central square
def hatched_region_area := 2 * (area of rectangles not required for this proof) + central_square_area

-- The problem statement to be proved
theorem octagon_area_difference:
  (hatched_region_area - grey_region_area) = (1 - 3 * (1 / 4)) := by
  sorry

end octagon_area_difference_l195_195445


namespace triangle_area_l195_195748

noncomputable def curve (x : ℝ) : ℝ := x * Real.log x

def tangent_line (x y : ℝ) (e : ℝ) : Prop :=
  y = 2 * (x - e) + e

theorem triangle_area :
  let e := Real.exp 1 in
  let area := (1 / 2) * e * (e / 2) in
  area = Real.exp 2 / 4 :=
by
  sorry

end triangle_area_l195_195748


namespace solve_distance_between_lines_l195_195742

def point := (ℝ × ℝ × ℝ)

def distance_between_lines (A B D A₁ B₁ C₁ : point) (AE BF : set point) : ℝ :=
  let E := (7, 0, 40) in
  let F := (14, 30, 40) in
  let d1 := (7, 0, 40) in
  let d2 := (0, 30, 40) in
  let c := (14, 0, 0) in
  let cross_product := (λ (x y : point), ((x.2.1 * y.2.2 - x.2.2 * y.2.1),
                                         (x.2.0 * y.2.2 - x.2.2 * y.2.0),
                                         (x.2.0 * y.2.1 - x.2.1 * y.2.0))) in
  let magnitude := (λ (x : point), real.sqrt (x.2.0 ^ 2 + x.2.1 ^ 2 + x.2.2 ^ 2)) in
  let dot_product := (λ (x y : point), (x.2.0 * y.2.0 + x.2.1 * y.2.1 + x.2.2 * y.2.2)) in
  let cross_d1_d2 := cross_product d1 d2 in
  let mag_cross_d1_d2 := magnitude cross_d1_d2 in
  let numerator := abs (dot_product c cross_d1_d2) in
  numerator / mag_cross_d1_d2

theorem solve_distance_between_lines :
  distance_between_lines (0,0,0) (14,0,0) (0,60,0) (0,0,40) (14,0,40) (14,60,40) 
                         ({(0,0,0), (7,0,40)}) ({(14,0,0), (14,30,40)}) = 13.44 := 
sorry

end solve_distance_between_lines_l195_195742


namespace empty_subset_of_A_l195_195172

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A :=
by
  sorry

end empty_subset_of_A_l195_195172


namespace chord_length_l195_195688

theorem chord_length (ρ θ : ℝ) (p : ℝ) : 
  (∀ θ, ρ = 6 * Real.cos θ) ∧ (θ = Real.pi / 4) → 
  ∃ l : ℝ, l = 3 * Real.sqrt 2 :=
by
  sorry

end chord_length_l195_195688


namespace arcsin_sqrt2_div_2_eq_pi_div_4_l195_195619

theorem arcsin_sqrt2_div_2_eq_pi_div_4 : Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
by
  -- proof goes here
  sorry

end arcsin_sqrt2_div_2_eq_pi_div_4_l195_195619


namespace green_shirt_pairs_l195_195956

theorem green_shirt_pairs (r g : ℕ) (p total_pairs red_pairs : ℕ) :
  r = 63 → g = 69 → p = 66 → red_pairs = 25 → (g - (r - red_pairs * 2)) / 2 = 28 :=
by
  intros hr hg hp hred_pairs
  sorry

end green_shirt_pairs_l195_195956


namespace andrew_purchased_11_kg_of_grapes_l195_195186

theorem andrew_purchased_11_kg_of_grapes
  (rate_grapes : ℕ)
  (kg_mangoes : ℕ)
  (rate_mangoes : ℕ)
  (total_paid : ℕ) 
  (H1 : rate_grapes = 98)
  (H2 : kg_mangoes = 7)
  (H3 : rate_mangoes = 50)
  (H4 : total_paid = 1428) :
  ∃ G : ℕ, 98 * G + 7 * 50 = 1428 ∧ G = 11 :=
by
  use 11
  simp [H1, H2, H3, H4]
  sorry

end andrew_purchased_11_kg_of_grapes_l195_195186


namespace range_of_values_l195_195975

noncomputable def increasing_function (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f(x) < f(y)

theorem range_of_values (f : ℝ → ℝ) (h_inc : increasing_function f) :
  {x : ℝ | f(x) < f(2 * x - 3)} = {x : ℝ | x > 3} :=
by
  sorry

end range_of_values_l195_195975


namespace midpoint_trajectory_is_circle_l195_195447

theorem midpoint_trajectory_is_circle
  (d : ℝ)
  (a b : ℝ)
  (fixed_length : a^2 + b^2 = d^2 - 4)
  (M : ℝ × ℝ × ℝ := (a / 2, b / 2, 0)) :
  ∃ r : ℝ, ∀ (x y : ℝ), (x, y, 0) ∈ set_of (λ p : ℝ × ℝ × ℝ, (p.1^2 + p.2^2 = (d^2 - 4) / 4)) :=
begin
  sorry
end

end midpoint_trajectory_is_circle_l195_195447


namespace find_m_n_calculate_expression_l195_195660

-- Define the polynomials A and B
def A (m x : ℝ) := 5 * x^2 - m * x + 1
def B (n x : ℝ) := 2 * x^2 - 2 * x - n

-- The conditions
variable (x : ℝ) (m n : ℝ)
def no_linear_or_constant_terms (m : ℝ) (n : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + (2 - m) * x + (1 + n) = 3 * x^2

-- The target theorem
theorem find_m_n 
  (h : no_linear_or_constant_terms m n) : 
  m = 2 ∧ n = -1 := sorry

-- Calculate the expression when m = 2 and n = -1
theorem calculate_expression
  (hm : m = 2)
  (hn : n = -1) : 
  m^2 + n^2 - 2 * m * n = 9 := sorry

end find_m_n_calculate_expression_l195_195660


namespace stock_z_shares_l195_195862

variables (v w x y z : ℕ) (v_init : v = 68) (w_init : w = 112) (x_init : x = 56) (y_init : y = 94)
variables (x_after : x = 36) (y_after : y = 117) (range_increase : 70 = 112 - 36 + 14)

theorem stock_z_shares : z = 47 :=
by
  unfold v w x y z at *
  rw [v_init, w_init, x_init, y_init, x_after, y_after, range_increase]
  sorry

end stock_z_shares_l195_195862


namespace sum_of_ages_l195_195769

variable (Matt : ℝ)

axiom Jed_age_condition : Matt + 10.4 + 9 = 23.8
axiom Emily_age_definition : 2.5 * Matt = 2.5 * Matt  -- Holds definition for Emily's age

theorem sum_of_ages :
    let Jed := Matt + 10.4 in
    let Emily := 2.5 * Matt in
    let sum := Matt + Jed + Emily in
    sum.round = 30 :=
by
    sorry

end sum_of_ages_l195_195769


namespace balloon_permutations_l195_195229

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l195_195229


namespace second_train_length_l195_195515

/-- Two trains running in opposite directions clear each other in 6.851865643851941 seconds.
    The first train is 111 meters long and its speed is 80 km/h.
    The second train is traveling at a speed of 65 km/h.
    Prove that the length of the second train is 165 meters. -/
theorem second_train_length
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (length1 : ℝ)
  (h_speed1 : speed1 = 80) (h_speed2 : speed2 = 65) 
  (h_time : time = 6.851865643851941) (h_length1 : length1 = 111) :
  let relative_speed := (speed1 + speed2) * 1000 / 3600 in
  let total_distance := relative_speed * time in
  total_distance - length1 = 165 :=
by
  -- Placeholder for proof
  sorry

end second_train_length_l195_195515


namespace find_n_l195_195269

theorem find_n (n : ℕ) (hnpos : 0 < n)
  (hsquare : ∃ k : ℕ, k^2 = n^4 + 2*n^3 + 5*n^2 + 12*n + 5) :
  n = 1 ∨ n = 2 := 
sorry

end find_n_l195_195269


namespace solve_abs_equation_l195_195057

theorem solve_abs_equation (y : ℝ) (h : |y - 8| + 3 * y = 12) : y = 2 :=
sorry

end solve_abs_equation_l195_195057


namespace pow_sixteen_l195_195714

theorem pow_sixteen (x : ℝ) (h : 2 ^ (4 * x) = 9) : 16 ^ (x + 1) = 144 :=
by
  -- The proof will be here
  sorry

end pow_sixteen_l195_195714


namespace min_value_y_l195_195661

theorem min_value_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 2) : 4 * a + b ≥ 8 :=
sorry

end min_value_y_l195_195661


namespace fraction_of_grid_covered_by_triangle_l195_195813

-- Definitions of the vertices, grid dimensions, and area calculation.
def D : (ℝ × ℝ) := (2, 1)
def E : (ℝ × ℝ) := (7, 1)
def F : (ℝ × ℝ) := (5, 5)
def width : ℝ := 8
def height : ℝ := 6

-- Function to compute the area of the triangle using the Shoelace Theorem.
def triangleArea (D E F : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := D
  let (x2, y2) := E
  let (x3, y3) := F
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Function to compute the area of the grid.
def gridArea (width height : ℝ) : ℝ := width * height

-- Function to compute the fraction of the grid covered by the triangle.
def fractionCovered (triangleArea gridArea : ℝ) : ℝ := triangleArea / gridArea

-- Proof statement showing the fraction of the grid covered by the triangle is 5/24.
theorem fraction_of_grid_covered_by_triangle : 
  fractionCovered (triangleArea D E F) (gridArea width height) = 5 / 24 :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l195_195813


namespace locus_midpoint_l195_195251

theorem locus_midpoint (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) :
  ∃ (x y : ℝ), (x = a / 2) ∧ (y = 1 - a / 2) ∧ (x + y = 1) ∧ (x ≠ 0) ∧ (x ≠ 1) :=
by {
  let x := a / 2,
  let y := 1 - a / 2,
  use [x, y],
  split,
  { refl },
  split,
  { refl },
  split,
  { calc 
      x + y = a / 2 + (1 - a / 2) : by rw [x, y]
          ... = a / 2 + 1 - a / 2 : by ring
          ... = 1 : by ring },
  split,
  { exact (λ h, h1 (div_eq_zero_iff.1 h)) },
  { exact (λ h, h2 (eq_div_of_mul_eq _ (by { ring, exact h })) sorry) }
}
sorrry

end locus_midpoint_l195_195251


namespace probability_of_heart_or_king_l195_195561

theorem probability_of_heart_or_king:
  let deck_size := 52
  let hearts := 13
  let kings := 4
  let non_heart_king_cards := deck_size - (hearts + kings - 1)
  let prob_first_not_heart_king := (deck_size - hearts - (kings - 1)) / deck_size
  let prob_second_not_heart_king := (non_heart_king_cards - 1) / (deck_size - 1)
  let prob_neither_heart_king := prob_first_not_heart_king * prob_second_not_heart_king
  let prob_at_least_one_heart_king := 1 - prob_neither_heart_king
  in prob_at_least_one_heart_king = 31 / 52 := by {
    sorry
  }

end probability_of_heart_or_king_l195_195561


namespace years_ago_three_times_anna_l195_195952

theorem years_ago_three_times_anna (x : ℕ) (h_anna_age : 54) (h_clara_age : 80) 
    (h_multiple : 80 - x = 3 * (54 - x)) : x = 41 :=
by
  sorry

end years_ago_three_times_anna_l195_195952


namespace problem_solution_l195_195525

theorem problem_solution :
  (3^1 + 4^2 - 2) ^ (-2) * 4 = 4 / 289 := 
by 
  sorry

end problem_solution_l195_195525


namespace nine_digit_number_l195_195995

theorem nine_digit_number (n : ℕ) : ∃ n, 
  (digits_unique n) ∧
  (non_zero_digits n) ∧
  (sqrt_form_ababc n) ∧
  (ab_eq_c_cubed n) : 
  \boxed{n = 743816529} := 
by
  sorry

-- Definitions
def digits_unique (n : ℕ) : Prop := 
  (n.toString.toList.nodup)

def non_zero_digits (n : ℕ) : Prop :=
  ∀ digit ∈ n.toString.toList, digit ≠ '0'

def sqrt_form_ababc (n : ℕ) : Prop :=
  let sqrt_n := (Int.sqrt n).toString
  let ab := sqrt_n.take 2
  ab = (sqrt_n.drop 3).init.take 2

def ab_eq_c_cubed (n : ℕ) : Prop :=
  let sqrt_n := (Int.sqrt n).toString
  let ab := sqrt_n.take 2
  let c := ab.toNat.get.sqrt
  ab.toNat.get = c^3

end nine_digit_number_l195_195995


namespace gasoline_cost_calculation_l195_195924

def tank_full_fraction_before_use : ℚ := 5 / 6
def tank_full_fraction_after_use : ℚ := 2 / 3
def gallons_used : ℚ := 18
def cost_per_gallon : ℚ := 4

theorem gasoline_cost_calculation :
  let x : ℚ := (tank_full_fraction_before_use - tank_full_fraction_after_use)⁻¹ * gallons_used in
  let cost : ℚ := gallons_used * cost_per_gallon in
  cost = 72 := by
  sorry

end gasoline_cost_calculation_l195_195924


namespace total_candies_l195_195758

-- Define variables and conditions
variables (x y z : ℕ)
axiom h1 : x = y / 2
axiom h2 : x + z = 24
axiom h3 : y + z = 34

-- The statement to be proved
theorem total_candies : x + y + z = 44 :=
by
  sorry

end total_candies_l195_195758


namespace product_of_3_point_6_and_0_point_25_l195_195253

theorem product_of_3_point_6_and_0_point_25 : 3.6 * 0.25 = 0.9 := 
by 
  sorry

end product_of_3_point_6_and_0_point_25_l195_195253


namespace average_marks_math_chem_l195_195863

theorem average_marks_math_chem (M P C : ℝ) (h1 : M + P = 60) (h2 : C = P + 20) : 
  (M + C) / 2 = 40 := 
by
  sorry

end average_marks_math_chem_l195_195863


namespace P_on_line_l_PA_PB_sum_l195_195749

-- Definitions of point, curve, and line
def Point : Type := {x : Real, y : Real}
def Curve (P : Point) : Prop := ∃ φ : Real, P.x = 2 * cos φ ∧ P.y = 2 * sin φ
def PolarLine (ρ θ : Real) : Prop := ρ * cos (θ + π / 3) = 3 / 2

-- Conditions
def P : Point := {x := 0, y := -sqrt 3}
def line_l (ρ θ : Real) : Prop := PolarLine ρ θ
def curve_C (P : Point) : Prop := Curve P

-- The first statement
theorem P_on_line_l : PolarLine (sqrt 3) (π / 3) :=
sorry

-- The second statement
theorem PA_PB_sum (A B : Point) (hA : A ∈ { P | Curve P })
  (hB : B ∈ { P | Curve P }) : 1 / dist P A + 1 / dist P B = sqrt 7 :=
sorry

end P_on_line_l_PA_PB_sum_l195_195749


namespace part1_part2_l195_195698

-- Define the function f
def f (ω x m : ℝ) : ℝ := (sqrt 3 / 2) * sin (ω * x) - (1 / 2) * cos (ω * x) - m

-- Define the conditions
def ω_pos (ω : ℝ) : Prop := ω > 0
def f_max_eq_5_f_min (m ω : ℝ) : Prop := 
    let f_max := 1 - m in
    let f_min := -1 - m in
    f_max = 5 * f_min
def f_zero_conditions (m ω : ℝ) (x1 x2 : ℝ) : Prop := 
    m = sqrt 2 / 2 ∧ 
    let x1_zero := (5 * π) / (12 * ω) in
    let x2_zero := (11 * π) / (12 * ω) in
    x2 - 2 * x1 = π / 36

-- Lean 4 statements only, no proofs
theorem part1 (ω m : ℝ) (hω_pos : ω_pos ω) (hf_max_eq_5_f_min : f_max_eq_5_f_min m ω) :
  m = -3/2 :=
sorry

theorem part2 (ω : ℝ) (x1 x2 : ℝ) (hf_zero_conditions : f_zero_conditions (sqrt 2 / 2) ω x1 x2) :
  ω = 3 :=
sorry

end part1_part2_l195_195698


namespace max_sin_plus_sin_mul_sin_plus_cos_mul_cos_l195_195592

noncomputable def sin_plus_sin_mul_sin_plus_cos_mul_cos_max (A B C : ℝ) (h : A + B + C = π) : ℝ :=
  sin A + sin B * sin C + cos B * cos C

theorem max_sin_plus_sin_mul_sin_plus_cos_mul_cos (A B C : ℝ) (h : A + B + C = π) :
  sin_plus_sin_mul_sin_plus_cos_mul_cos_max A B C h ≤ 2 :=
sorry

end max_sin_plus_sin_mul_sin_plus_cos_mul_cos_l195_195592


namespace sum_dihedral_angles_eq_360_l195_195814

variables {A B C O : Point}
variables {OA OB OC : Line}
variables (cylinder : Cylinder)

-- Definitions from conditions
def diametrically_opposite (A B : Point) : Prop :=
  A ≠ B ∧ distance A B = diameter_of cylinder.base

def not_in_plane (P Q R S : Point) : Prop :=
  ¬ (coplanar P Q R S)

def midpoint (M A B : Point) : Prop :=
  distance M A = distance M B ∧ A ≠ B

-- The theorem to prove
theorem sum_dihedral_angles_eq_360
  (h1 : diametrically_opposite A B)
  (h2 : not_in_plane A B O C)
  (h3 : midpoint O A B)
  (h4 : midpoint O (O + cylinder.axis / 2) (O - cylinder.axis / 2)) :
  sum_dihedral_angles A B C O = 360 :=
sorry

end sum_dihedral_angles_eq_360_l195_195814


namespace carnival_rent_l195_195562

-- Define the daily popcorn earnings
def daily_popcorn : ℝ := 50
-- Define the multiplier for cotton candy earnings
def multiplier : ℝ := 3
-- Define the number of operational days
def days : ℕ := 5
-- Define the cost of ingredients
def ingredients_cost : ℝ := 75
-- Define the net earnings after expenses
def net_earnings : ℝ := 895
-- Define the total earnings from selling popcorn for all days
def total_popcorn_earnings : ℝ := daily_popcorn * days
-- Define the total earnings from selling cotton candy for all days
def total_cottoncandy_earnings : ℝ := (daily_popcorn * multiplier) * days
-- Define the total earnings before expenses
def total_earnings : ℝ := total_popcorn_earnings + total_cottoncandy_earnings
-- Define the amount remaining after paying the rent (which includes net earnings and ingredient cost)
def remaining_after_rent : ℝ := net_earnings + ingredients_cost
-- Define the rent
def rent : ℝ := total_earnings - remaining_after_rent

theorem carnival_rent : rent = 30 := by
  sorry

end carnival_rent_l195_195562


namespace balloon_permutations_l195_195230

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l195_195230


namespace first_machine_time_l195_195816

def machine_times (T₁ T₂ T₃ : ℝ) : Prop :=
  (1 / T₁ + 1 / T₂ = 1 / T₃)

theorem first_machine_time
  (T₂ : ℝ)
  (T₃ : ℝ) 
  (h₁ : T₂ = 8) 
  (h₂ : T₃ = 4.235294117647059) : 
  ∃ T₁ : ℝ, machine_times T₁ T₂ T₃ ∧ T₁ ≈ 9 := 
by {
  sorry
}

end first_machine_time_l195_195816


namespace volume_of_fuel_A_l195_195185

variables (V_A V_B : ℝ)

def condition1 := V_A + V_B = 212
def condition2 := 0.12 * V_A + 0.16 * V_B = 30

theorem volume_of_fuel_A :
  condition1 V_A V_B → condition2 V_A V_B → V_A = 98 :=
by
  intros h1 h2
  sorry

end volume_of_fuel_A_l195_195185


namespace num_perfect_square_factors_of_360_l195_195329

theorem num_perfect_square_factors_of_360 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d : ℕ, d ∣ 360 → (∀ p e, p^e ∣ d → (p = 2 ∨ p = 3 ∨ p = 5) ∧ e % 2 = 0) :=
by
  sorry

end num_perfect_square_factors_of_360_l195_195329


namespace total_transportation_cost_l195_195926

def weights_in_grams : List ℕ := [300, 450, 600]
def cost_per_kg : ℕ := 15000

def convert_to_kg (w : ℕ) : ℚ :=
  w / 1000

def calculate_cost (weight_in_kg : ℚ) (cost_per_kg : ℕ) : ℚ :=
  weight_in_kg * cost_per_kg

def total_cost (weights_in_grams : List ℕ) (cost_per_kg : ℕ) : ℚ :=
  weights_in_grams.map (λ w => calculate_cost (convert_to_kg w) cost_per_kg) |>.sum

theorem total_transportation_cost :
  total_cost weights_in_grams cost_per_kg = 20250 := by
  sorry

end total_transportation_cost_l195_195926


namespace sequence_sum_square_l195_195199

-- Definition of the sum of the symmetric sequence.
def sequence_sum (n : ℕ) : ℕ :=
  (List.range' 1 (n+1)).sum + (List.range' 1 n).sum

-- The conjecture that the sum of the sequence equals n^2.
theorem sequence_sum_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end sequence_sum_square_l195_195199


namespace total_marbles_l195_195107

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l195_195107


namespace sin_graph_shift_l195_195098

theorem sin_graph_shift :
  ∀ x : ℝ, sin (2 * (x - π / 6)) = sin (2 * x - π / 3) :=
by
  intro x
  sorry

end sin_graph_shift_l195_195098


namespace balloon_permutations_l195_195227

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l195_195227


namespace alternately_seat_circular_table_l195_195558

-- Definitions based on conditions
def is_democrat (person : ℕ) : Prop := person <= 6
def is_republican (person : ℕ) : Prop := person > 6

def alternates (seating : List ℕ) : Prop :=
  ∀ i : ℕ, i < seating.length → 
    if i % 2 = 0 then is_democrat (seating.nth_le i (by linarith))
    else is_republican (seating.nth_le i (by linarith))

def circular_equiv (seating1 seating2 : List ℕ) : Prop :=
  ∃ k, cycle_permutation k seating1 = seating2

def num_democrats (seating : List ℕ) : Prop := 
  seating.countp is_democrat = 6

def num_republicans (seating : List ℕ) : Prop := 
  seating.countp is_republican = 6

-- Statement of the theorem
theorem alternately_seat_circular_table :
  ∃ (seatings : Finset (List ℕ)), 
    (∀ s ∈ seatings, seating.length = 12 ∧ alternates s ∧ num_democrats s ∧ num_republicans s) ∧ 
    seating.card = 86400 :=
sorry

end alternately_seat_circular_table_l195_195558


namespace FarmB_chickens_count_l195_195243

def FarmA :=
  {chickens : ℝ,
   ducks : ℝ,
   total := chickens + ducks = 625}

def FarmB :=
  {chickens : ℝ,
   ducks : ℝ,
   total := chickens + ducks = 748}

noncomputable def chicken_proportion_A_to_B (C_A C_B : ℝ) : Prop :=
  C_B = 1.24 * C_A

noncomputable def duck_proportion_B_to_A (D_A D_B : ℝ) : Prop :=
  D_A = 0.85 * D_B

theorem FarmB_chickens_count 
  (C_A C_B : ℝ) 
  (D_A D_B : ℝ) 
  (hA : FarmA)
  (hB : FarmB)
  (hc : chicken_proportion_A_to_B C_A C_B)
  (hd : duck_proportion_B_to_A D_A D_B) :
  C_B = 248 :=
sorry

end FarmB_chickens_count_l195_195243


namespace ratio_of_wealth_l195_195973

theorem ratio_of_wealth (W P : ℝ) 
  (h1 : 0 < P) (h2 : 0 < W) 
  (pop_X : ℝ := 0.4 * P) 
  (wealth_X : ℝ := 0.6 * W) 
  (top50_pop_X : ℝ := 0.5 * pop_X) 
  (top50_wealth_X : ℝ := 0.8 * wealth_X) 
  (pop_Y : ℝ := 0.2 * P) 
  (wealth_Y : ℝ := 0.3 * W) 
  (avg_wealth_top50_X : ℝ := top50_wealth_X / top50_pop_X) 
  (avg_wealth_Y : ℝ := wealth_Y / pop_Y) : 
  avg_wealth_top50_X / avg_wealth_Y = 1.6 := 
by sorry

end ratio_of_wealth_l195_195973


namespace eggs_in_box_l195_195346

-- Given conditions as definitions in Lean 4
def initial_eggs : ℕ := 7
def additional_whole_eggs : ℕ := 3

-- The proof statement
theorem eggs_in_box : initial_eggs + additional_whole_eggs = 10 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end eggs_in_box_l195_195346


namespace correct_options_l195_195281

variable (x : Fin 6 → ℝ)

def median_of_4 (a b c d : ℝ) : ℝ := (b + c) / 2

def median_of_6 (a b c d e f : ℝ) : ℝ := (c + d) / 2

theorem correct_options (x : Fin 6 → ℝ)
  (h1 : x 0 = min (min (x 0) (x 1)) (min (min (x 2) (x 3)) (min (x 4) (x 5)))) 
  (h6 : x 5 = max (max (x 0) (x 1)) (max (max (x 2) (x 3)) (max (x 4) (x 5)))) :
  (median_of_4 (x 1) (x 2) (x 3) (x 4) = median_of_6 (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) ∧
  (range (x 1) (x 2) (x 3) (x 4) ≤ range (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) :=
sorry

end correct_options_l195_195281


namespace range_of_a_dot_product_l195_195800

variable {ℝ : Type*} [Real]

variables (a b : EuclideanSpace ℝ)
-- Given Conditions
axiom h1 : ∥a + b∥ = 3
axiom h2 : ∥a - b∥ = 2

-- Question to be proved
theorem range_of_a_dot_product (a b : EuclideanSpace ℝ)
  (h1 : ∥a + b∥ = 3) (h2 : ∥a - b∥ = 2) :
  ∃ (A B : ℝ), A = 2/5 ∧ B = 2 ∧ 
  ∀ k : ℝ, k = (∥a∥) / (a ⬝ b) → k ∈ set.Icc A B :=
sorry

end range_of_a_dot_product_l195_195800


namespace balloon_permutations_l195_195228

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l195_195228


namespace profit_calculation_l195_195031

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l195_195031


namespace volume_of_tetrahedron_is_8_l195_195705

-- Define the tetrahedron and its properties
constant Tetrahedron : Type
constant givenThreeViews : Tetrahedron → Prop

-- The given condition in the problem
constant given_tetrahedron : Tetrahedron
axiom three_views_condition : givenThreeViews given_tetrahedron

-- The statement to prove
theorem volume_of_tetrahedron_is_8 : givenThreeViews given_tetrahedron → volume given_tetrahedron = 8 :=
by
  sorry

end volume_of_tetrahedron_is_8_l195_195705


namespace sum_of_coordinates_of_reflected_midpoint_l195_195417

theorem sum_of_coordinates_of_reflected_midpoint :
  let P := (3 : ℤ, 2 : ℤ),
      R := (17 : ℤ, 18 : ℤ),
      M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2),
      reflected_P := (-P.1, P.2),
      reflected_R := (-R.1, R.2),
      reflected_M := ((reflected_P.1 + reflected_R.1) / 2, (reflected_P.2 + reflected_R.2) / 2)
  in reflected_M.1 + reflected_M.2 = 0 := sorry

end sum_of_coordinates_of_reflected_midpoint_l195_195417


namespace find_function_expression_find_tangent_line_l195_195317

noncomputable def f (x : ℝ) := a * x^3 + b * x^2 + c * x

theorem find_function_expression (a b c : ℝ) (h1 : 3 * a + 2 * b + c = 0) (h2 : 3 * a - 2 * b + c = 0) (h3 : c = -3) :
  f = λ x, x^3 - 3 * x :=
by {
  sorry
}

noncomputable def tangent_line (x0 : ℝ) : ℝ → ℝ := λ x, (3 * x0^2 - 3) * (x - x0) + (x0^3 - 3 * x0)

theorem find_tangent_line (hA : ∃ x0, tangent_line x0 2 = 2) :
  (∃ x0, tangent_line x0 = λ x, 2) ∨ (∃ x0, tangent_line x0 = λ x, 9 * x - 16) :=
by {
  sorry
}

end find_function_expression_find_tangent_line_l195_195317


namespace cone_height_l195_195154

theorem cone_height (r : ℝ) (n : ℕ) (h : ℝ) :
  r = 8 → n = 4 → h = 2 * Real.sqrt 15 →
  let base_circumference := (2 * Real.pi * r) / n
  let base_radius := base_circumference / (2 * Real.pi)
  ∃ h : ℝ, h = Real.sqrt (r^2 - base_radius^2) := 
by
  intros hr hn hh
  have base_circ := (2 * Real.pi * r) / n
  have base_rad := base_circ / (2 * Real.pi)
  use Real.sqrt (r^2 - base_rad^2)
  split
  all_goals { sorry }

end cone_height_l195_195154


namespace count_increasing_digits_between_200_and_300_l195_195324

def is_increasing_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 3 ∧ digits.nodup ∧ digits = digits.sorted

def count_integers_increasing_order (a b : ℕ) : ℕ :=
  (a+1).upto b |>.filter (λ n, is_increasing_digits n) |>.length

theorem count_increasing_digits_between_200_and_300 :
  count_integers_increasing_order 200 300 = 21 :=
sorry

end count_increasing_digits_between_200_and_300_l195_195324


namespace factor_expression_l195_195639

theorem factor_expression (x y : ℝ) :
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) :=
by
  sorry

end factor_expression_l195_195639


namespace part_a_part_b_l195_195914

-- Part (a)
theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (a - b) / (1 + a * b) ∧ (a - b) / (1 + a * b) ≤ 1 := sorry

-- Part (b)
theorem part_b (x y z u : ℝ) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (b - a) / (1 + a * b) ∧ (b - a) / (1 + a * b) ≤ 1 := sorry

end part_a_part_b_l195_195914


namespace stations_distance_l195_195467

theorem stations_distance {α : ℝ} (hα : α < 1) :
  ∃ (n : ℕ), ∃ (positions : Finset (Fin n → Fin n)), Finset.card positions ≥ 100^n ∧ ∀ (f g ∈ positions), f ≠ g → ∀ i, ∥f i - g i∥ ≥ n * α :=
by sorry

end stations_distance_l195_195467


namespace travel_time_specific_l195_195070

-- Define the initial time to travel from A to B
def travel_time (distance v : ℕ) : ℕ :=
  distance / v

-- Define the time that includes speed increment 'a'
def travel_time_with_increment (distance v a : ℕ) : ℕ :=
  distance / (v + a)

-- Define the time difference after the speed increment
def time_difference (distance v a : ℕ) : ℕ :=
  travel_time distance v - travel_time_with_increment distance v a

-- Prove the specific cases when a = 10 and v = 40    
theorem travel_time_specific : 
  travel_time 100 40 = 2.5 ∧ time_difference 100 40 10 = 0.5 :=
by
  simp [travel_time, travel_time_with_increment, time_difference]
  exact sorry

end travel_time_specific_l195_195070


namespace balloon_permutations_l195_195207

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l195_195207


namespace original_vertices_in_terms_of_extended_points_l195_195673

variables {V : Type*} [inner_product_space ℝ V]

def quadrilateral_relation (E F G H E' F' G' H' : V) : Prop :=
  F = 1/3 • E + 2/3 • E' ∧
  G = 1/2 • F + 1/2 • F' ∧
  H = 1/3 • G + 2/3 • G' ∧
  E = 1/2 • H + 1/2 • H'

theorem original_vertices_in_terms_of_extended_points
  {E F G H E' F' G' H' : V}
  (h : quadrilateral_relation E F G H E' F' G' H') :
  E = 1/35 • E' + 7/70 • F' + 14/35 • G' + 28/35 • H' :=
sorry

end original_vertices_in_terms_of_extended_points_l195_195673


namespace sam_mary_total_balloons_l195_195826

def Sam_initial_balloons : ℝ := 6.0
def Sam_gives : ℝ := 5.0
def Sam_remaining_balloons : ℝ := Sam_initial_balloons - Sam_gives

def Mary_balloons : ℝ := 7.0

def total_balloons : ℝ := Sam_remaining_balloons + Mary_balloons

theorem sam_mary_total_balloons : total_balloons = 8.0 :=
by
  sorry

end sam_mary_total_balloons_l195_195826


namespace find_k_l195_195364

variable (m n p k : ℝ)

-- Conditions
def cond1 : Prop := m = 2 * n + 5
def cond2 : Prop := p = 3 * m - 4
def cond3 : Prop := m + 4 = 2 * (n + k) + 5
def cond4 : Prop := p + 3 = 3 * (m + 4) - 4

theorem find_k (h1 : cond1 m n)
               (h2 : cond2 m p)
               (h3 : cond3 m n k)
               (h4 : cond4 m p) :
               k = 2 :=
  sorry

end find_k_l195_195364


namespace supplement_complement_57_degrees_l195_195116

theorem supplement_complement_57_degrees :
  let θ := 57 in
  let complement := 90 - θ in
  let supplement := 180 - complement in
  supplement = 147 :=
by
  -- This hypothesis and the result, embodied in a Lean theorem statement.
  sorry

end supplement_complement_57_degrees_l195_195116


namespace max_n_for_powers_of_primes_l195_195267

noncomputable def q_seq : ℕ → ℕ → ℕ
| 0, q0 := q0
| (n + 1), q0 := (q_seq n q0 - 1)^3 + 3

def is_power_of_prime (p : ℕ) : ℕ → Prop
| k := ∃ n : ℕ, k = p^n

theorem max_n_for_powers_of_primes (q0 : ℕ) (h0 : q0 > 0) :
  (∀ n, (∀ i ≤ n, ∃ p : ℕ, prime p ∧ is_power_of_prime p (q_seq i q0)) → n ≤ 2) :=
by sorry

end max_n_for_powers_of_primes_l195_195267


namespace repeating_decimal_sum_l195_195992

theorem repeating_decimal_sum :
  (0.12121212 + 0.003003003 + 0.0000500005 : ℚ) = 124215 / 999999 :=
by 
  have h1 : (0.12121212 : ℚ) = (0.12 + 0.0012) := sorry
  have h2 : (0.003003003 : ℚ) = (0.003 + 0.000003) := sorry
  have h3 : (0.0000500005 : ℚ) = (0.00005 + 0.0000000005) := sorry
  sorry


end repeating_decimal_sum_l195_195992


namespace distance_from_focus_to_asymptote_l195_195303

theorem distance_from_focus_to_asymptote (b : ℝ) (h_b_pos : 0 < b) 
    (asymptote_condition : ∀ x y, y = 2 * x ↔ (x^2 - (y^2 / b^2) = 1)) : 
    let c := real.sqrt (1 + b^2) in
    let focus_x := c in
    let focus_y := 0 in
    let A := 2 in
    let B := -1 in
    let C := 0 in
    ∃ d : ℝ, d = 2 :=
sorry

end distance_from_focus_to_asymptote_l195_195303


namespace imaginary_part_of_z_l195_195307

noncomputable def z : ℂ := (1 - complex.i) / (1 + complex.i)

theorem imaginary_part_of_z :
  complex.abs_im z = -1 := 
sorry -- Proof to be provided separately

end imaginary_part_of_z_l195_195307


namespace find_number_l195_195041

theorem find_number : ∃ n : ℕ, n = (15 * 6) + 5 := 
by sorry

end find_number_l195_195041


namespace least_product_of_distinct_primes_gt_30_l195_195493

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l195_195493


namespace sum_mean_median_mode_l195_195122

def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

noncomputable def median (l : List ℝ) : ℝ := 
  let sl := l.qsort (≺)
  if sl.length % 2 = 0 then
    (sl.get ⟨(sl.length / 2) - 1, sorry⟩ + sl.get ⟨sl.length / 2, sorry⟩) / 2
  else
    sl.get ⟨sl.length / 2, sorry⟩

def mode (l : List ℝ) : ℝ :=
  l.foldl (λ m p => if l.count p > l.count m then p else m) 0

theorem sum_mean_median_mode : 
  mean [2, 3, 0, 3, 1, 4, 0, 3] + median [2, 3, 0, 3, 1, 4, 0, 3] + mode [2, 3, 0, 3, 1, 4, 0, 3] = 7.5 := 
by
  sorry

end sum_mean_median_mode_l195_195122


namespace necessary_but_not_sufficient_l195_195663

theorem necessary_but_not_sufficient (a : ℝ) : (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) := sorry

end necessary_but_not_sufficient_l195_195663


namespace binomial_divisible_by_prime_l195_195427

-- Define the conditions: p is prime and 0 < k < p
variables (p k : ℕ)
variable (hp : Nat.Prime p)
variable (hk : 0 < k ∧ k < p)

-- State that the binomial coefficient \(\binom{p}{k}\) is divisible by \( p \)
theorem binomial_divisible_by_prime
  (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_l195_195427


namespace probability_line_intersects_circle_l195_195473

theorem probability_line_intersects_circle :
  (let favorable_pairs : ℕ := 15 in
   let total_pairs : ℕ := 36 in
   let probability := (favorable_pairs : ℝ) / (total_pairs : ℝ) in
   probability = (5 / 12)) :=
by
  -- Probability computation logic
  let favorable_pairs := 15
  let total_pairs := 36
  let probability := (favorable_pairs : ℝ) / (total_pairs : ℝ)
  exact Eq.refl (5 / 12) sorry

end probability_line_intersects_circle_l195_195473


namespace find_current_l195_195007

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end find_current_l195_195007


namespace pechkin_speed_l195_195821

-- Define the problem conditions

def pechkin_overtake_km : ℝ := 4.5 -- Pechkin is overtaken every 4.5 kilometers
def bus_opposite_time_min : ℕ := 9 -- A bus traveling in the opposite direction passes every 9 minutes
def bus_interval_min : ℕ := 12 -- The interval between bus movements in both directions

-- Convert bus opposite time and interval to hours
def bus_opposite_time_hr : ℝ := (bus_opposite_time_min : ℝ) / 60
def bus_interval_hr : ℝ := (bus_interval_min : ℝ) / 60

-- Prove the speed of Pechkin
theorem pechkin_speed (x y : ℝ) (h1 : 4.5 / x = 12 / 60 + 4.5 / y)
                      (h2 : (3 / 20) * (x + y) = y / 5) :
    x = 15 :=
sorry

end pechkin_speed_l195_195821


namespace decreasing_number_4312_max_decreasing_number_divisible_by_9_l195_195340

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end decreasing_number_4312_max_decreasing_number_divisible_by_9_l195_195340


namespace principal_equivalence_l195_195585

-- Define the conditions
def SI : ℝ := 4020.75
def R : ℝ := 9
def T : ℝ := 5

-- Define the principal calculation
noncomputable def P := SI / (R * T / 100)

-- Prove that the principal P equals 8935
theorem principal_equivalence : P = 8935 := by
  sorry

end principal_equivalence_l195_195585


namespace num_perfect_square_factors_of_360_l195_195331

theorem num_perfect_square_factors_of_360 : 
  (∃ f : ℕ, prime_factors f = {2, 3, 5} ∧ positive_factors f = 360 ∧ perfect_square f 4) :=
  sorry

end num_perfect_square_factors_of_360_l195_195331


namespace find_longer_subsegment_of_YZ_l195_195084

-- Define the problem conditions in Lean 4
variables {k : ℝ} (XY YZ XZ YE EZ : ℝ)
variable (triangle_inequality : ∀ a b c : ℝ, a + b > c ∧ b + c > a ∧ c + a > b)

-- State the conditions and the answer
noncomputable def problem_statement (YZ : ℝ) (ratioXY : ℝ) (ratioXZ : ℝ) (k : ℝ) :=
  XY = 3 * k ∧ YZ = 4 * k ∧ XZ = 5 * k ∧ YZ = 15 ∧ YE = (3/5) * EZ ∧ YE + EZ = 15

noncomputable def correct_answer (EZ : ℝ) :=
  EZ = 75 / 8

-- The theorem statement
theorem find_longer_subsegment_of_YZ :
  ∃ (EZ : ℝ), problem_statement 15 3 5 (15 / 4) ∧ correct_answer EZ :=
by {
  sorry
}

end find_longer_subsegment_of_YZ_l195_195084


namespace geometric_series_sum_l195_195259

theorem geometric_series_sum :
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 6 in
  a * (1 - r^n) / (1 - r) = 1953 / 7812 :=
by
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 6
  have h1 : (a * (1 - r^n) / (1 - r)) = (1953 / 7812) := sorry
  exact h1

end geometric_series_sum_l195_195259


namespace rectangle_dimensions_exist_l195_195869

theorem rectangle_dimensions_exist :
  ∃ (a b c d : ℕ), (a * b + c * d = 81) ∧ (2 * (a + b) = 2 * 2 * (c + d) ∨ 2 * (c + d) = 2 * 2 * (a + b)) :=
by sorry

end rectangle_dimensions_exist_l195_195869


namespace closed_fishing_season_purpose_sustainable_l195_195458

-- Defining the options for the purpose of the closed fishing season
inductive FishingPurpose
| sustainable_development : FishingPurpose
| inspect_fishing_vessels : FishingPurpose
| prevent_red_tides : FishingPurpose
| zoning_management : FishingPurpose

-- Defining rational utilization of resources involving fishing seasons
def rational_utilization (closed_fishing_season: Bool) : FishingPurpose := 
  if closed_fishing_season then FishingPurpose.sustainable_development 
  else FishingPurpose.inspect_fishing_vessels -- fallback for contradiction; shouldn't be used

-- The theorem we want to prove
theorem closed_fishing_season_purpose_sustainable :
  rational_utilization true = FishingPurpose.sustainable_development :=
sorry

end closed_fishing_season_purpose_sustainable_l195_195458


namespace exists_consecutive_2022_with_22_primes_l195_195635

def P (n : ℕ) : ℕ := (Finset.range 2022).count (λ k, Nat.prime (n + k))

theorem exists_consecutive_2022_with_22_primes : 
  ∃ n : ℕ, P n = 22 :=
by
  have upper_bound : P 1 > 22 :=
    sorry -- Proven in the problem statement, needs detailed proof otherwise

  have lower_bound : P (2023! + 2) = 0 :=
    sorry -- Proven in the problem statement, needs detailed proof otherwise

  have step_change : ∀ n : ℕ, (P (n + 1)).abs (P n) ≤ 1 :=
    sorry -- Based on the step nature proven in the problem statement

  -- Provided these observations, we can state that there exists an n such that P n = 22
  exact sorry

end exists_consecutive_2022_with_22_primes_l195_195635


namespace balloon_arrangements_correct_l195_195217

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l195_195217


namespace no_similar_triangle_in_process_l195_195935

theorem no_similar_triangle_in_process (a b c : ℝ) (h₁ : 0 < a) (h₂ : a < b) (h₃ : b < c)
  (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : a + c > b) :
  ¬∃ n : ℕ, ∃ (a' b' c' : ℝ), (a' / a = b' / b) ∧ (b' / b = c' / c) ∧
  a' = (a + b - c) ∧ b' = (b + c - a) ∧ c' = (a + c - b)
  ∧ (a' + b' > c') ∧ (b' + c' > a') ∧ (a' + c' > b') := 
begin
  sorry
end

end no_similar_triangle_in_process_l195_195935


namespace part_I_part_II_part_III_l195_195363

noncomputable theory
open Classical

section
variables {a : ℕ → ℝ}
variables {b c : ℕ → ℝ}

-- Definitions based on the given conditions
def a_seq (n : ℕ) : ℕ → ℝ := -- monotonic increasing
  if h : n = 0 then 2 else sorry

axiom a1 : a 1 = 2
axiom a_monotonic : ∀ n : ℕ, a (n + 1) ≥ a n
axiom a_condition : ∀ n : ℕ, (n + 1) * a n ≥ n * a (2 * n)

def b (n : ℕ) : ℝ :=
  (List.ofFn (λ i, 1 + 1 / (2 ^ (i + 1))) (n + 1)).foldl (λ x y, x * y) 1

def c (n : ℕ) : ℝ :=
  6 * (1 - 1 / (2 ^ n))

-- Theorem 1: Prove the range of a_2
theorem part_I : (2 : ℝ) < a 2 ∧ a 2 ≤ 4 := sorry

-- Theorem 2: Prove that the sequence cannot be a geometric progression
theorem part_II : ¬∃ q > 1, ∀ n : ℕ, a n = 2 * q^n := sorry

-- Theorem 3: Prove that for any n >= 1, (b_n - c_n) / (a_n - 12) ≥ 0
theorem part_III : ∀ n : ℕ, 0 < n → 0 ≤ (b n - c n) / (a n - 12) := sorry
end

end part_I_part_II_part_III_l195_195363


namespace prism_division_l195_195532

-- Define a prism and the division by a parallel plane
def isPrism (P : Type) : Prop := sorry

-- Conditions that need to be proven
/-
We will assume the necessary definitions for "prism" and division by a plane.
The definition should capture the properties that allow us to claim that a prism
can be divided into two prisms by a plane parallel to its base.
-/

-- The main theorem we need to prove
theorem prism_division (P : Type) (h : isPrism P) :
  ∃ P1 P2 : Type, isPrism P1 ∧ isPrism P2 ∧ (P1 ∪ P2 = P ∧ P1 ∩ P2 = ∅) :=
sorry

end prism_division_l195_195532


namespace cos_gamma_prime_l195_195394

theorem cos_gamma_prime (
  (Q : ℝ × ℝ × ℝ)
  (hQ_pos : 0 < Q.1 ∧ 0 < Q.2 ∧ 0 < Q.3)
  (α' β' γ' : ℝ)
  (hcos_α' : real.cos α' = (2 : ℝ) / 5)
  (hcos_β' : real.cos β' = (1 : ℝ) / 4)
  (OQ_sq : (Q.1 ^ 2 + Q.2 ^ 2 + Q.3 ^ 2) = 53)
) : real.cos γ' = (real.sqrt 42.14) / (real.sqrt 53) :=
sorry

end cos_gamma_prime_l195_195394


namespace reciprocal_of_common_fraction_form_l195_195893

-- Definitions based on conditions
def repeating_decimal_x : ℚ := 0.565656565656... -- This is .\overline{56}

-- Proof statement
theorem reciprocal_of_common_fraction_form :
  let x := repeating_decimal_x
  in  (1 / x = 99 / 56) :=
by
  sorry

end reciprocal_of_common_fraction_form_l195_195893


namespace tiling_impossible_l195_195539

theorem tiling_impossible (T2 T14 : ℕ) :
  let S_before := 2 * T2
  let S_after := 2 * (T2 - 1) + 1 
  S_after ≠ S_before :=
sorry

end tiling_impossible_l195_195539


namespace problem_proof_l195_195195

def mixed_to_improper (a b c : ℚ) : ℚ := a + b / c

noncomputable def evaluate_expression : ℚ :=
  100 - (mixed_to_improper 3 1 8) / (mixed_to_improper 2 1 12 - 5 / 8) * (8 / 5 + mixed_to_improper 2 2 3)

theorem problem_proof : evaluate_expression = 636 / 7 := 
  sorry

end problem_proof_l195_195195


namespace first_player_wins_optimal_play_l195_195885

def is_proper_divisor (d n : ℕ) : Prop :=
  d > 0 ∧ d < n ∧ n % d = 0

def next_number (n : ℕ) : Set ℕ :=
  { m | ∃ d, is_proper_divisor d n ∧ m = n + d }

def losing_number (n : ℕ) : Prop :=
  ∀ m ∈ next_number n, m > 19891989

noncomputable def game_state := ℕ

def moves (n : game_state) : game_state :=
  n + 1  -- This function captures adding the smallest proper divisor 1

theorem first_player_wins_optimal_play :
  ∀ (initial_state : game_state) (player_turn : ℕ),
  initial_state = 2 →
  (∀ n, losing_number n → n > 19891989) →
  (∀ n, losing_number n → player_turn % 2 = 1) →
  player_turn % 2 = 0 →
  false :=
by
  intros
  sorry

end first_player_wins_optimal_play_l195_195885


namespace pizza_topping_count_l195_195164

theorem pizza_topping_count:
  let toppings := 6 in
  (toppings + (toppings.choose 2) + (toppings.choose 3) = 41) :=
by
  let toppings := 6
  have ht1: toppings = 6 := rfl
  have ht2: (toppings.choose 2) = 15 := by sorry
  have ht3: (toppings.choose 3) = 20 := by sorry
  have ha1: (6 : ℕ) = 6 := rfl
  have ha2: 15 = 15 := rfl
  have ha3: 20 = 20 := rfl
  calc
    (6 + 15 + 20) = 41 : rfl
      ... = 41 : rfl

end pizza_topping_count_l195_195164


namespace initial_number_of_students_l195_195067

-- Definitions
def avg_initial_weight := 15
def avg_new_weight := 14.4
def new_student_weight := 3

-- Theorem statement
theorem initial_number_of_students :
  ∃ n : ℕ, ∃ W : ℝ, 
    W = n * avg_initial_weight ∧ 
    W + new_student_weight = (n + 1) * avg_new_weight ∧ 
    n = 19 :=
sorry

end initial_number_of_students_l195_195067


namespace cary_wage_after_two_years_l195_195965

theorem cary_wage_after_two_years (initial_wage raise_percentage cut_percentage : ℝ) (wage_after_first_year wage_after_second_year : ℝ) :
  initial_wage = 10 ∧ raise_percentage = 0.2 ∧ cut_percentage = 0.75 ∧ 
  wage_after_first_year = initial_wage * (1 + raise_percentage) ∧
  wage_after_second_year = wage_after_first_year * cut_percentage → 
  wage_after_second_year = 9 :=
by
  sorry

end cary_wage_after_two_years_l195_195965


namespace least_product_of_primes_gt_30_l195_195499

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l195_195499


namespace range_of_ab_eq_l195_195270

theorem range_of_ab_eq (-4b^2 + b) : Set.Iic (1 / 8) := sorry

end range_of_ab_eq_l195_195270


namespace problem_statement_proof_l195_195751

noncomputable def polar_to_cartesian_l := ∀ (ρ θ : ℝ),
  ρ * Math.cos (θ + Real.pi / 4) = Real.sqrt 2 / 2 → ρ * Math.cos θ - ρ * Math.sin θ = 1

noncomputable def parametric_to_ordinary_C := ∀ (θ : ℝ),
  (let x := 5 + Math.cos θ in let y := Math.sin θ in (x - 5) ^ 2 + y ^ 2 = 1)

noncomputable def minimal_perimeter := 
  ∀ (A B : ℝ × ℝ) (P : ℝ × ℝ),
    A = (4, 0) ∧ B = (6, 0) ∧ (P.1 - P.2 - 1 = 0) → 
    let perimeter := (Real.sqrt ((1 - 6)^2 + (3 - 0)^2)) + 2 in
    perimeter = Real.sqrt 34 + 2

theorem problem_statement_proof :
  (polar_to_cartesian_l ∧ parametric_to_ordinary_C) ∧ minimal_perimeter :=
begin
  sorry
end

end problem_statement_proof_l195_195751


namespace number_of_distinct_c_l195_195388

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem number_of_distinct_c : 
  {c : ℝ | f (f (f (f c))) = 5}.to_finset.card = 8 :=
sorry

end number_of_distinct_c_l195_195388


namespace concurrent_lines_l195_195407

open EuclideanGeometry

theorem concurrent_lines
    (A B C D X Y O M N : Point)
    (h_collinear : Collinear [A, B, C, D])
    (h_circles : ∃ Γ1 Γ2 : Circle, (Diameter Γ1 (Segment A C)) ∧ (Diameter Γ2 (Segment B D))
                   ∧ (Intersection Γ1 Γ2 = [X, Y]))
    (h_O_on_XY : O ∈ Segment X Y)
    (h_intersections : (Betw (O, C, M) ∧ (M ∈ Γ1))
                       ∧ (Betw (O, B, N) ∧ (N ∈ Γ2))) :
    ConcurrentLines (Line A M) (Line X Y) (Line N D) :=
begin
  sorry
end

end concurrent_lines_l195_195407


namespace solve_quadratic_eq_l195_195831

theorem solve_quadratic_eq (x : ℂ) : 
  x^2 + 6 * x + 8 = -(x + 2) * (x + 6) ↔ (x = -3 + complex.I ∨ x = -3 - complex.I) := 
by
  sorry 

end solve_quadratic_eq_l195_195831


namespace profit_calculation_l195_195029

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l195_195029


namespace chicken_wings_cost_final_cost_of_chicken_wings_l195_195381

variable (cost_of_chicken_wings : ℕ)

-- Conditions
def lee_money : ℕ := 10
def friend_money : ℕ := 8
def chicken_salad_cost : ℕ := 4
def sodas_cost : ℕ := 2
def tax : ℕ := 3
def change_received : ℕ := 3

-- Question: Prove the cost of chicken wings
theorem chicken_wings_cost (total_money := lee_money + friend_money)
                          (total_spent := total_money - change_received)
                          (total_meal_cost := cost_of_chicken_wings + chicken_salad_cost + sodas_cost + tax) :
  total_meal_cost = total_spent :=
by
  sorry

-- Final statement involves proving the final value
theorem final_cost_of_chicken_wings (h : total_meal_cost 6) : cost_of_chicken_wings = 6 :=
by
  sorry

end chicken_wings_cost_final_cost_of_chicken_wings_l195_195381


namespace machine_present_value_l195_195927

theorem machine_present_value
  (r : ℝ)  -- the depletion rate
  (t : ℝ)  -- the time in years
  (V_t : ℝ)  -- the value of the machine after time t
  (V_0 : ℝ)  -- the present value of the machine
  (h1 : r = 0.10)  -- condition for depletion rate
  (h2 : t = 2)  -- condition for time
  (h3 : V_t = 729)  -- condition for machine's value after time t
  (h4 : V_t = V_0 * (1 - r) ^ t)  -- exponential decay formula
  : V_0 = 900 :=
sorry

end machine_present_value_l195_195927


namespace complex_conjugate_identity_l195_195723

theorem complex_conjugate_identity (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
by sorry

end complex_conjugate_identity_l195_195723


namespace Tn_ge_n_l195_195633

noncomputable section 

def T (a b : List ℤ) (n : ℕ) : ℤ :=
  (∑ i in Finset.range n, ∑ j in Finset.range n, |a[i] - b[j]|) - 
  (∑ i in Finset.range (n - 1), ∑ j in Finset.Ico(i + 1) n, (|a[j] - a[i]| + |b[j] - b[i]|))

theorem Tn_ge_n (n : ℕ) (a b : ℕ → ℤ) (h_length_a : a.length = n) (h_length_b : b.length = n) :
  T a b n ≥ n := by
  sorry

end Tn_ge_n_l195_195633


namespace find_27th_number_l195_195136

variable (nums : List ℝ)
variable (first_15 : List ℝ)
variable (next_12 : List ℝ)
variable (last_13 : List ℝ)

-- Conditions:
variable (h1 : first_15.length = 15)
variable (h2 : next_12.length = 12)
variable (h3 : last_13.length = 13)
variable (h_total_40 : (first_15 ++ next_12 ++ last_13).length = 40)
variable (h_avg_40 : (first_15 ++ next_12 ++ last_13).sum / 40 = 55.8)
variable (h_avg_first_15 : first_15.sum / 15 = 53.2)
variable (h_avg_next_12 : next_12.sum / 12 = 52.1)
variable (h_avg_last_13 : last_13.sum / 13 = 60.5)

-- Statement:
theorem find_27th_number :
  (nums = first_15 ++ next_12 ++ last_13) → 
  (h_avg_40) → 
  (h_avg_first_15) → 
  (h_avg_next_12) → 
  (h_avg_last_13) → 
  nums.nthLe 26 sorry = 52.1 :=
by sorry

end find_27th_number_l195_195136


namespace history_teachers_count_l195_195581

theorem history_teachers_count (E G H : ℕ) (hE : E = 9) (hG : G = 6) (hMin : 11 ≤ E + G - (nat.div G 2)) : 
  (∃ H : ℕ, H = (E - nat.div G 2) + (nat.sub 11 (E + G - 2 * (nat.div G 2)))) :=
sorry

end history_teachers_count_l195_195581


namespace hiking_duration_l195_195114

theorem hiking_duration (violet_consumption_per_hour : ℕ) (dog_consumption_per_hour : ℕ) (total_water : ℕ) :
  violet_consumption_per_hour = 800 →
  dog_consumption_per_hour = 400 →
  total_water = 4800 →
  total_water / (violet_consumption_per_hour + dog_consumption_per_hour) = 4 :=
by
  intros h1 h2 h3
  rw [total_water, violet_consumption_per_hour, dog_consumption_per_hour, h1, h2, h3]
  exact sorry

end hiking_duration_l195_195114


namespace train_cross_time_approx_30_seconds_l195_195557

noncomputable def length_of_train := 100 -- in meters
noncomputable def length_of_bridge := 300 -- in meters
noncomputable def speed_of_train_kmh := 48 -- in km/h

noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600) -- in m/s
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge -- in meters

theorem train_cross_time_approx_30_seconds :
  (total_distance / speed_of_train_ms) ≈ 30 :=
sorry

end train_cross_time_approx_30_seconds_l195_195557


namespace bisect_pentagon_l195_195794

noncomputable def segments_bisect (A B C D P : Point) : Prop :=
  let M := midpoint C D
  collinear A P M

theorem bisect_pentagon (A B C D E P : Point) (h1 : angle_eq A B C A C D A D E) 
  (h2 : angle_eq B A C C A D D A E) (hP : intersection_point P B D C E) :
  segments_bisect A C D P :=
begin
  sorry
end

end bisect_pentagon_l195_195794


namespace ratio_of_roots_l195_195631

theorem ratio_of_roots (c : ℝ) :
  (∃ (x1 x2 : ℝ), 5 * x1^2 - 2 * x1 + c = 0 ∧ 5 * x2^2 - 2 * x2 + c = 0 ∧ x1 / x2 = -3 / 5) → c = -3 :=
by
  sorry

end ratio_of_roots_l195_195631


namespace possible_values_of_f2001_l195_195081

noncomputable def f : ℕ → ℝ := sorry

theorem possible_values_of_f2001 (f : ℕ → ℝ)
    (H : ∀ a b : ℕ, a > 1 → b > 1 → ∀ d : ℕ, d = Nat.gcd a b → 
           f (a * b) = f d * (f (a / d) + f (b / d))) :
    f 2001 = 0 ∨ f 2001 = 1/2 :=
sorry

end possible_values_of_f2001_l195_195081


namespace exists_divisible_pair_l195_195042

theorem exists_divisible_pair (s : Finset ℕ) (h : s.card = 11) (h_bound : ∀ n ∈ s, n ≤ 20) :
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end exists_divisible_pair_l195_195042


namespace total_marbles_l195_195108

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l195_195108


namespace range_of_m_l195_195406

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0 
def neg_q_sufficient_for_neg_p (m : ℝ) : Prop :=
  ∀ x : ℝ, p x → q x m

theorem range_of_m (m : ℝ) : neg_q_sufficient_for_neg_p m → m ≥ 9 :=
by
  sorry

end range_of_m_l195_195406


namespace arcsin_sqrt2_over_2_l195_195617

noncomputable def sin_π_over_4 : ℝ := Real.sin (Real.pi / 4)

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
by 
  have h : sin_π_over_4 = Real.sqrt 2 / 2,
  { sorry },
  rw [←Real.arcsin_sin (Real.pi / 4)],
  exact h

end arcsin_sqrt2_over_2_l195_195617


namespace sin_greater_of_angle_greater_acute_triangle_sin_cos_inequality_l195_195759

variables {A B C : ℝ} {a b c : ℝ}

-- First part: A > B implies sin A > sin B
theorem sin_greater_of_angle_greater (h1 : A > B) : sin A > sin B :=
by sorry

-- Second part: For an acute triangle, sin A + sin B > cos A + cos B
theorem acute_triangle_sin_cos_inequality (h2 : A + B + C = π) 
  (h3 : A < π/2) (h4 : B < π/2) (h5 : C < π/2) : sin A + sin B > cos A + cos B :=
by sorry

end sin_greater_of_angle_greater_acute_triangle_sin_cos_inequality_l195_195759


namespace inscribed_circle_radius_square_l195_195921

theorem inscribed_circle_radius_square (ER RF GS SH : ℝ) (r : ℝ) 
  (hER : ER = 23) (hRF : RF = 34) (hGS : GS = 42) (hSH : SH = 28)
  (h_tangent : ∀ t, t = r * r * (70 * t - 87953)) :
  r^2 = 87953 / 70 :=
by
  sorry

end inscribed_circle_radius_square_l195_195921


namespace ratio_of_AD_and_BC_l195_195362

-- Define the conditions of the problem
variables (s : ℝ) (A B C D : ℝ × ℝ)

-- Conditions for the triangles
def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB2 := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  let AC2 := (A.1 - C.1)^2 + (A.2 - C.2)^2 in
  let BC2 := (B.1 - C.1)^2 + (B.2 - C.2)^2 in
  AB2 = AC2 ∧ BC2 = AB2 + AC2

def is_equilateral_triangle (B C D : ℝ × ℝ) (s : ℝ) : Prop :=
  let BC2 := (B.1 - C.1)^2 + (B.2 - C.2)^2 in
  let CD2 := (C.1 - D.1)^2 + (C.2 - D.2)^2 in
  let DB2 := (D.1 - B.1)^2 + (D.2 - B.2)^2 in
  BC2 = s^2 ∧ CD2 = s^2 ∧ DB2 = s^2

-- Statement to prove AD / BC = √14 / 4
theorem ratio_of_AD_and_BC (h_iso : is_isosceles_right_triangle A B C)
  (h_equi : is_equilateral_triangle B C D s) :
  let AB2 := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  let AD := sqrt (AB2 + (s^2 * 3 / 4)) in
  let BC := sqrt AB2 * sqrt 2 in
  AD / BC = sqrt 14 / 4 :=
begin
  sorry
end

end ratio_of_AD_and_BC_l195_195362


namespace solve_for_a_l195_195693

theorem solve_for_a : 
  let a := 2
  in (binom 4 2 * 4 - binom 4 1 * 2 * binom 5 1 * a + binom 5 2 * a^2 = -16) → a = 2 :=
by 
  intros
  let a := 2
  sorry

end solve_for_a_l195_195693


namespace one_room_cheaper_by_l195_195817

-- Define the initial prices of the apartments
variables (a b : ℝ)

-- Define the increase rates and the new prices
def new_price_one_room := 1.21 * a
def new_price_two_room := 1.11 * b
def new_total_price := 1.15 * (a + b)

-- The main theorem encapsulating the problem
theorem one_room_cheaper_by : a + b ≠ 0 → 1.21 * a + 1.11 * b = 1.15 * (a + b) → b / a = 1.5 :=
by
  intro h_non_zero h_prices
  -- we assume the main theorem is true to structure the goal state
  sorry

end one_room_cheaper_by_l195_195817


namespace verify_yearly_interest_l195_195415

structure Investment :=
(total_investment : ℕ)
(amount_invested_10 : ℕ)
(interest_rate_10 : ℝ)
(interest_rate_8 : ℝ)

def yearly_interest (inv : Investment) : ℝ :=
  let interest_10 := inv.amount_invested_10 * inv.interest_rate_10
  let amount_invested_8 := inv.total_investment - inv.amount_invested_10
  let interest_8 := amount_invested_8 * inv.interest_rate_8
  interest_10 + interest_8

theorem verify_yearly_interest :
  let inv := Investment.mk 3000 800 0.10 0.08 in
  yearly_interest inv = 256 :=
by 
  -- proof omitted
  sorry

end verify_yearly_interest_l195_195415


namespace probability_of_B_l195_195063

variable (Ω : Type) [ProbabilitySpace Ω]
variable (A B C : Event Ω)
variable (hP_A : ℙ[A] = 0.4)
variable (hP_A_and_B : ℙ[A ∩ B] = 0.25)
variable (hP_A_or_B : ℙ[A ∪ B] = 0.8)
variable (hP_A_and_C : ℙ[A ∩ C] = 0.1)
variable (hP_B_and_C : ℙ[B ∩ C] = 0.15)
variable (hP_A_and_B_and_C : ℙ[A ∩ B ∩ C] = 0.05)

theorem probability_of_B :
  ℙ[B] = 0.65 :=
sorry

end probability_of_B_l195_195063


namespace balloon_permutations_l195_195224

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l195_195224


namespace cary_wage_after_two_years_l195_195966

theorem cary_wage_after_two_years (initial_wage raise_percentage cut_percentage : ℝ) (wage_after_first_year wage_after_second_year : ℝ) :
  initial_wage = 10 ∧ raise_percentage = 0.2 ∧ cut_percentage = 0.75 ∧ 
  wage_after_first_year = initial_wage * (1 + raise_percentage) ∧
  wage_after_second_year = wage_after_first_year * cut_percentage → 
  wage_after_second_year = 9 :=
by
  sorry

end cary_wage_after_two_years_l195_195966


namespace time_to_cross_opposite_directions_l195_195543

def length_of_train (L : ℝ) : Prop :=
  ∃ v1 v2 t1, v1 = 60 ∧ v2 = 40 ∧ t1 = 60 ∧
  L = (v1 - v2) * 1000 * t1 / 3600  / 2

theorem time_to_cross_opposite_directions (L : ℝ) (v1 v2 t1 t2 : ℝ) 
(h1 : v1 = 60) (h2 : v2 = 40) (h3 : t1 = 60) (h4 : t2 = 12)
(hL : length_of_train L) :
  let relative_speed := (v1 + v2) * 1000 / 3600 in
  t2 = (2 * L) / relative_speed :=
by sorry

end time_to_cross_opposite_directions_l195_195543


namespace distance_center_to_line_l195_195941

theorem distance_center_to_line (a b c d x : ℝ) (e : ℝ → ℝ → Prop) :
  (∀ (A B C D : ℝ → ℝ → Prop), 
    A (0,0) ∧ B (1,0) ∧ C (1,1) ∧ D (0,1) ∧ 
    (e (0, 0) ∧ ¬ e (1,0) ∧ ¬ e (1,1) ∧ ¬ e (0,1))) → 
  (a = dist (0, 0) e ∧ b = dist (1, 0) e ∧ c = dist (1, 1) e ∧ d = dist (0, 1) e) →
  (a * c = b * d) →
  (x = dist (1/2, 1/2) e) →
  x = 1 / 2 :=
by
  sorry

end distance_center_to_line_l195_195941


namespace array_sum_ge_half_n_squared_l195_195621

theorem array_sum_ge_half_n_squared (n : ℕ) 
  (a : ℕ → ℕ → ℕ)
  (h : ∀ i j, a i j = 0 → (∑ k, a i k) + (∑ k, a k j) ≥ n)  : 
  (∑ i j, a i j) ≥ n * n / 2 := 
sorry

end array_sum_ge_half_n_squared_l195_195621


namespace harmonic_integral_polynomial_integer_image_sum_identity_equivalence_conditions_quadratic_real_solution_count_l195_195554

def harmonic_number (n k : ℤ) : ℤ :=
  -- Definition of H_n(x) as provided
  (1 / n.factorial : ℤ) * ((k - 0) * (k - 1) * ⋯ * (k - (n - 1)))

theorem harmonic_integral (n k : ℤ) : harmonic_number n k ∈ ℤ := sorry

def polynomial_solution_condition (P : ℂ[X]) : Prop :=
  ∀ k : ℕ, P.eval k ∈ ℤ

theorem polynomial_integer_image (P : ℂ[X]) : polynomial_solution_condition P :=
  -- Equivalent proof steps leading to the integer combinations of H_i(x)
  sorry

theorem sum_identity (j k : ℕ) : 
  (∑ i in j..k, (-1)^(k - i) * (nat.choose k i) * (nat.choose i j)) = 
  if j = k then 1 else 0 := sorry

def polynomial_condition (u : ℕ → ℤ) (P : ℝ[X]) : Prop :=
  ∀ j : ℕ, u j = P.eval j

def summation_condition (u : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≥ n + 1 → (∑ j in 0..i, (-1)^(i - j) * (nat.choose i j) * u j) = 0

theorem equivalence_conditions (u : ℕ → ℤ) :
  (∃ P : ℝ[X], polynomial_condition u P) ↔ (∃ n : ℕ, summation_condition u n) := sorry

theorem quadratic_real_solution_count (b c : ℝ) :
  ∃ (n : ℕ), (n = 0 ∧ b^2 - 4*c < 0) ∨ (n = 1 ∧ b^2 - 4*c = 0) ∨ (n = 2 ∧ b^2 - 4*c > 0) := sorry

end harmonic_integral_polynomial_integer_image_sum_identity_equivalence_conditions_quadratic_real_solution_count_l195_195554


namespace sale_in_third_month_l195_195158

def sales_in_months (m1 m2 m3 m4 m5 m6 : Int) : Prop :=
  m1 = 5124 ∧
  m2 = 5366 ∧
  m4 = 6124 ∧
  m6 = 4579 ∧
  (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 5400

theorem sale_in_third_month (m5 : Int) :
  (∃ m3 : Int, sales_in_months 5124 5366 m3 6124 m5 4579 → m3 = 11207) :=
sorry

end sale_in_third_month_l195_195158


namespace passengers_landed_on_time_l195_195775

theorem passengers_landed_on_time (total_passengers : ℕ) (late_passengers : ℕ)
  (h1 : total_passengers = 14720) (h2 : late_passengers = 213) :
  total_passengers - late_passengers = 14507 :=
by {
  rw [h1, h2],
  norm_num,
}

end passengers_landed_on_time_l195_195775


namespace sum_of_binomial_expansion_l195_195690

theorem sum_of_binomial_expansion (n : ℕ) (hn : 0 < n) :
  (∑ k in finset.range n, 4 ^ (k + 1)) = (4 * (4 ^ n - 1)) / 3 :=
by sorry

end sum_of_binomial_expansion_l195_195690


namespace find_number_of_people_l195_195570

def number_of_people (total_shoes : Nat) (shoes_per_person : Nat) : Nat :=
  total_shoes / shoes_per_person

theorem find_number_of_people :
  number_of_people 20 2 = 10 := 
by
  sorry

end find_number_of_people_l195_195570


namespace sum_powers_zero_iff_odd_l195_195778

theorem sum_powers_zero_iff_odd (a : Fin 8 → ℝ) : 
  (∀ n, (∃ x ∈ (Set.Icc (0 : ℝ) 8), a x ≠ 0) → c_n = ∑ k, a k ^ n) → 
  (∀ n, (c_n = 0 → (∃ m, n = 2 * m + 1))) := 
sorry

end sum_powers_zero_iff_odd_l195_195778


namespace proof_problem_l195_195389

noncomputable def f : ℝ → ℝ
| x if x % 4 == 0                   := f(0)
| x if x % 4 == 1 ∧ x % 2 == 0      := f(1)
| x if x % 4 == 2 ∧ x % 2 == 0      := f(2)
| x if x % 4 == -1 ∨ x % 4 == 3     := -f(1)
| x if x % 4 == -2 ∨ x % 4 == 2 ∧ x % 2 != 0 := -f(2)
| x := f(x % 4)

lemma f_values : f(0) = 0 ∧ f(1) = 1 ∧ f(2) = 2 ∧ (∀ x, f(-x) = -f(x)) ∧ ∀ x, f(x + 4) = f(x) := 
by sorry

theorem proof_problem :
  f(2015) + f(2016) + f(2017) + f(2018) = 2 :=
by
  have f_is_odd : ∀ x, f(x) = -f(-x) := by sorry
  have periodicity : ∀ x, f(x + 4) = f(x) := by sorry
  have f_0 := (f_values).1
  have f_1 := (f_values).2.1
  have f_2 := (f_values).2.2.1
  have f_3 := (f_values).2.2.2.1
  calc 
    f(2015) + f(2016) + f(2017) + f(2018) 
        = -f(1) + f(0) + f(1) + f(2) : by sorry
    ... = -1 + 0 + 1 + 2 : by sorry
    ... = 2 : by sorry

end proof_problem_l195_195389


namespace max_value_of_e_n_l195_195786

def b (n : ℕ) : ℕ := (8^n - 1) / 7
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_of_e_n : ∀ n : ℕ, e n = 1 := 
by
  sorry

end max_value_of_e_n_l195_195786


namespace sum_of_two_digit_integers_squares_ending_in_25_l195_195523

theorem sum_of_two_digit_integers_squares_ending_in_25 : 
  let valid_n := {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5} in
  (∑ n in valid_n, n) = 495 :=
by 
  sorry

end sum_of_two_digit_integers_squares_ending_in_25_l195_195523


namespace six_digit_palindromes_count_l195_195630

theorem six_digit_palindromes_count : 
  let a_choices := 9 in
  let b_choices := 10 in
  let c_choices := 10 in
  let d_choices := 10 in
  a_choices * b_choices * c_choices * d_choices = 9000 := 
by {
  sorry -- Proof to be filled in later
}

end six_digit_palindromes_count_l195_195630


namespace unique_triangle_areas_l195_195820

-- Define the distances in Lean
def AB : ℝ := 1
def BC : ℝ := 2
def CD : ℝ := 3
def EF : ℝ := 1
def FG : ℝ := 2

-- Define the base lengths that can be formed from the given points
def base_lengths : set ℝ := {1, 2, 3, 5, 6}

-- Define the proof statement that there are 5 unique areas
theorem unique_triangle_areas : base_lengths.card = 5 :=
by sorry

end unique_triangle_areas_l195_195820


namespace sin_identity_cos_identity_l195_195051

-- Part (a)
theorem sin_identity (α β γ : ℝ) :
  sin α + sin β + sin γ - sin (α + β + γ) =
  4 * sin ((α + β) / 2) * sin ((β + γ) / 2) * sin ((α + γ) / 2) :=
sorry

-- Part (b)
theorem cos_identity (α β γ : ℝ) :
  cos α + cos β + cos γ + cos (α + β + γ) =
  4 * cos ((α + β) / 2) * cos ((β + γ) / 2) * cos ((α + γ) / 2) :=
sorry

end sin_identity_cos_identity_l195_195051


namespace analytical_expression_of_f_f_monotonically_increasing_on_interval_range_of_a_l195_195685

noncomputable def f (a b : ℝ) (x : ℝ) := (ax + b) / (x^2 + 4)

theorem analytical_expression_of_f:
  (f (1:ℝ) 0) = λ x, x / (x^2 + 4) := 
sorry

theorem f_monotonically_increasing_on_interval :
  ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 -> (x1 / (x1^2 + 4)) < (x2 / (x2^2 + 4)) :=
sorry

theorem range_of_a (a : ℝ) :
  (λ (a:ℝ), (a+1) / ((a+1)^2 + 4) + (1-2a) / ((1-2a)^2 + 4) > 0) -> (-1/2:ℝ) < a ∧ a < 1 :=
sorry

end analytical_expression_of_f_f_monotonically_increasing_on_interval_range_of_a_l195_195685


namespace fraction_female_participants_l195_195819

theorem fraction_female_participants
  (males_last_year : ℕ)
  (females_last_year : ℕ)
  (males_this_year : ℕ)
  (females_this_year : ℕ)
  (total_last_year : ℕ)
  (total_this_year : ℕ) :
  males_last_year = 30 →
  males_this_year = 33 →
  females_this_year = 1.25 * females_last_year →
  total_last_year = males_last_year + females_last_year →
  total_this_year = 1.15 * total_last_year →
  (females_this_year : ℚ) / (males_this_year + females_this_year) = 25 / 69 :=
by
  intros
  simp_all
  sorry

end fraction_female_participants_l195_195819


namespace least_product_of_distinct_primes_gt_30_l195_195491

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l195_195491


namespace complex_example_true_l195_195055

section
  variables (a b c d e f g h i j k l : ℂ)

  def complex_example : Prop :=
    (2 - 5 * complex.I) - (-3 + 7 * complex.I) - 4 * (-1 + 2 * complex.I) = 9 - 20 * complex.I

  theorem complex_example_true : complex_example :=
  by
    sorry
end

end complex_example_true_l195_195055


namespace cooks_number_l195_195604

variable (C W : ℕ)

theorem cooks_number (h1 : 10 * C = 3 * W) (h2 : 14 * C = 3 * (W + 12)) : C = 9 :=
by
  sorry

end cooks_number_l195_195604


namespace sum_is_multiple_of_31_l195_195516

theorem sum_is_multiple_of_31 (n : ℕ) (h : n > 0) :
  (∑ i in finset.range (5 * n), (2 : ℕ)^i) % 31 = 0 :=
sorry

end sum_is_multiple_of_31_l195_195516


namespace quadrilaterals_not_necessarily_congruent_l195_195746

variables {P Q R S P1 Q1 R1 S1 : Type*} [IsPreorder P Q] [IsPreorder P1 Q1]

structure Quadrilateral (A B C D : Type*) :=
  (angleA : angle)
  (angleB : angle)
  (angleC : angle)
  (angleD : angle)
  (sideAB : ℝ)
  (sideAC : ℝ)
  (sideBD : ℝ)
  
variables (ABCD A1B1C1D1 : Quadrilateral P Q R S)

axiom equal_angles : (ABCD.angleA = A1B1C1D1.angleA) ∧
                     (ABCD.angleB = A1B1C1D1.angleB) ∧
                     (ABCD.angleC = A1B1C1D1.angleC) ∧
                     (ABCD.angleD = A1B1C1D1.angleD)

axiom equal_side_lengths : (ABCD.sideAB = A1B1C1D1.sideAB) ∧
                           (ABCD.sideAC = A1B1C1D1.sideAC) ∧
                           (ABCD.sideBD = A1B1C1D1.sideBD)

theorem quadrilaterals_not_necessarily_congruent :
  ¬ (ABCD = A1B1C1D1) :=
sorry

end quadrilaterals_not_necessarily_congruent_l195_195746


namespace hexagon_area_is_20_sqrt_3_l195_195444

noncomputable def area_of_hexagon : ℝ :=
  let side_length_large := 11
  let side_length_small_1 := 1
  let side_length_small_2 := 2
  let side_length_small_6 := 6

  let area_equilateral (side : ℝ) : ℝ :=
    (real.sqrt 3 / 4) * side ^ 2

  let area_large := area_equilateral side_length_large
  let area_small_1 := area_equilateral side_length_small_1
  let area_small_2 := area_equilateral side_length_small_2
  let area_small_6 := area_equilateral side_length_small_6

  area_large - (area_small_1 + area_small_2 + area_small_6)

theorem hexagon_area_is_20_sqrt_3 :
  area_of_hexagon = 20 * real.sqrt 3 :=
by
  -- Here we would provide the proof of our theorem, currently omitted.
  sorry

end hexagon_area_is_20_sqrt_3_l195_195444


namespace divisibility_by_5_l195_195115

theorem divisibility_by_5 (B : ℕ) (hB : B < 10) : (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := 
by
  sorry

end divisibility_by_5_l195_195115


namespace range_of_a_l195_195788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1) → f a x1 ≥ g x2) →
  a ≥ -2 :=
sorry

end range_of_a_l195_195788


namespace xixi_cards_l195_195898

variable (x y : ℕ)

def condition1 := x + 3 = 3 * (y - 3)
def condition2 := y + 4 = 4 * (x - 4)
def condition3 := x + 5 = 5 * (y - 5)
def exactly_one_incorrect := 
  (condition1 ∧ ¬condition2 ∧ condition3) ∨
  (¬condition1 ∧ condition2 ∧ condition3) ∨
  (condition1 ∧ condition2 ∧ ¬condition3)

theorem xixi_cards : 
  exactly_one_incorrect → x = 15 :=
sorry

end xixi_cards_l195_195898


namespace ratio_S9_S3_l195_195799

-- Define the geometric sequence and its sum
def geom_sequence_sum (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

-- Define the conditions from the problem
variables (a q : ℝ) (h : q ≠ 1)
variables (S3 S6 : ℝ)
hypothesis (h₁ : geom_sequence_sum a q 6 / geom_sequence_sum a q 3 = 1 / 2)

-- Prove the desired ratio
theorem ratio_S9_S3 : geom_sequence_sum a q 9 / geom_sequence_sum a q 3 = 3 / 4 := by
  sorry

end ratio_S9_S3_l195_195799


namespace ratio_QP_l195_195847

noncomputable def P : ℚ := 11 / 6
noncomputable def Q : ℚ := 5 / 2

theorem ratio_QP : Q / P = 15 / 11 := by 
  sorry

end ratio_QP_l195_195847


namespace tom_total_payment_l195_195877

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end tom_total_payment_l195_195877


namespace total_weight_of_snacks_l195_195380

-- Definitions for conditions
def weight_peanuts := 0.1
def weight_raisins := 0.4
def weight_almonds := 0.3

-- Theorem statement
theorem total_weight_of_snacks : weight_peanuts + weight_raisins + weight_almonds = 0.8 := by
  sorry

end total_weight_of_snacks_l195_195380


namespace xiaoding_distance_l195_195463

--- Define the distances for Xiaoding, Xiaowang, Xiaocheng, and Xiaozhang
variables (d_xd dx dw_1 dc_2 dz_3 : ℕ)

--- Assume the following conditions
def condition1 : Prop := dx + d_xd + dc_2 + dz_3 = 705
def condition2 : Prop := dw_1 = 4 * d_xd
def condition3 : Prop := dc_2 = (dw_1 / 2) + 20
def condition4 : Prop := dz_3 = 2 * dc_2 - 15

--- The main theorem
theorem xiaoding_distance :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 → d_xd = 60 :=
by
  sorry

end xiaoding_distance_l195_195463


namespace balloon_arrangements_correct_l195_195212

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l195_195212


namespace shifted_parabola_vertex_l195_195864

-- Given the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 + 2 * x

-- Shifts
def shifted_left (x : ℝ) : ℝ := x - 1
def shifted_up (y : ℝ) : ℝ := y + 2

-- Prove the new vertex coordinates after the shifts
theorem shifted_parabola_vertex :
  let (vx, vy) := (-1, -1) in
  let new_vx := shifted_left vx in
  let new_vy := shifted_up vy in
  (new_vx, new_vy) = (-2, 1) :=
by
  sorry

end shifted_parabola_vertex_l195_195864


namespace coefficient_x4_l195_195296

noncomputable def a : ℝ :=
  4 * ∫ x in 0..(Real.pi/2), Real.cos (2 * x + Real.pi / 6)

theorem coefficient_x4 :
  a = 4 * ∫ x in 0..(Real.pi/2), Real.cos (2 * x + Real.pi / 6) →
  (x : ℝ) -> (∑ r in Finset.range 6, binomial 5 r * (x^2)^(5-r) * (-a/x)^r = ((40 : ℝ) * x^4) + ....) :=
by
    sorry

end coefficient_x4_l195_195296


namespace balloon_permutations_l195_195231

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l195_195231


namespace units_digit_7_pow_451_l195_195123

theorem units_digit_7_pow_451 : (7^451 % 10) = 3 := by
  sorry

end units_digit_7_pow_451_l195_195123


namespace external_diagonal_impossible_l195_195128

theorem external_diagonal_impossible (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) :
  ¬ (a = 8 ∧ b = 15 ∧ c = 18) := 
by {
  -- Definitions of the side lengths of the right rectangular prism
  let x := 8
  let y := 15
  let z := 18
  have h₂ : x^2 + y^2 = 289 := by norm_num,
  have h₃ : z^2 = 324 := by norm_num,
  -- Check the condition
  have : x^2 + y^2 < z^2 := by linarith,
  -- Definition indicating that the lengths cannot be external diagonals
  exact this,
  sorry
}

end external_diagonal_impossible_l195_195128


namespace count_integers_200_300_l195_195327

theorem count_integers_200_300 :
  let is_valid (n : ℕ) := n ≥ 200 ∧ n < 300 ∧
                         let d1 := n / 100 in
                         let d2 := (n / 10) % 10 in
                         let d3 := n % 10 in
                         d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
                         d1 < d2 ∧ d2 < d3 in
  (finset.range 100).filter (λ n, is_valid (n + 200)).card = 18 :=
by
  sorry

end count_integers_200_300_l195_195327


namespace find_principal_amount_l195_195537

theorem find_principal_amount 
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ)
  (hA : A = 3087) (hr : r = 0.05) (hn : n = 1) (ht : t = 2)
  (hcomp : A = P * (1 + r / n)^(n * t)) :
  P = 2800 := 
  by sorry

end find_principal_amount_l195_195537


namespace boxes_needed_to_win_trip_l195_195874

theorem boxes_needed_to_win_trip : 
  ∀ (total_bars boxes_bars : ℕ), total_bars = 849 → boxes_bars = 5 → (total_bars + boxes_bars - 1) / boxes_bars = 170 :=
by
  intros total_bars boxes_bars h1 h2
  rw [h1, h2]
  norm_num
  sorry

end boxes_needed_to_win_trip_l195_195874


namespace find_number_l195_195556

theorem find_number (x : ℕ) (h : (40 * 30 + (x + 8) * 3) / 5 = 1212) : x = 1612 :=
by {
  sorry,
}

end find_number_l195_195556


namespace curve_is_line_segment_l195_195443

noncomputable def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = Real.cos θ ^ 2 ∧ p.2 = Real.sin θ ^ 2}

theorem curve_is_line_segment :
  (∀ p ∈ parametric_curve, p.1 + p.2 = 1 ∧ p.1 ∈ Set.Icc 0 1) :=
by
  sorry

end curve_is_line_segment_l195_195443


namespace least_product_of_primes_over_30_l195_195481

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l195_195481


namespace find_k_l195_195297

variables (e1 e2 : Vector ℝ) (k : ℝ)

-- Two non-zero non-collinear vectors
noncomputable def non_zero_non_collinear (e1 e2 : Vector ℝ) : Prop :=
  e1 ≠ 0 ∧ e2 ≠ 0 ∧ ¬collinear e1 e2

-- Definition of vectors a and b
def vector_a (e1 e2 : Vector ℝ) : Vector ℝ := 2 • e1 - e2
def vector_b (e1 e2 : Vector ℝ) (k : ℝ) : Vector ℝ := k • e1 + e2

-- Condition that a and b are collinear
def collinear_vectors (a b : Vector ℝ) : Prop := ∃ λ : ℝ, a = λ • b

-- The main statement to prove
theorem find_k (e1 e2 : Vector ℝ) (k : ℝ) :
  non_zero_non_collinear e1 e2 →
  collinear_vectors (vector_a e1 e2) (vector_b e1 e2 k) →
  k = -2 :=
by
  sorry

end find_k_l195_195297


namespace smallest_x_for_M_squared_l195_195651

theorem smallest_x_for_M_squared (M x : ℤ) (h1 : 540 = 2^2 * 3^3 * 5) (h2 : 540 * x = M^2) (h3 : x > 0) : x = 15 :=
sorry

end smallest_x_for_M_squared_l195_195651


namespace sum_of_squares_leq_m_plus_rsq_l195_195857

theorem sum_of_squares_leq_m_plus_rsq (x : ℕ → ℝ) (n : ℕ) (m : ℤ) (r : ℝ) 
  (hx_interval : ∀ i, i < n → 0 < x i ∧ x i < 1)
  (hx_sum : ∑ i in finset.range n, x i = m + r)
  (hr_bound : 0 ≤ r ∧ r < 1) :
  ∑ i in finset.range n, (x i)^2 ≤ m + r^2 :=
by
  sorry

end sum_of_squares_leq_m_plus_rsq_l195_195857


namespace g_neg_one_l195_195792

def g (d e f x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

theorem g_neg_one {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end g_neg_one_l195_195792


namespace problem_part1_problem_part2_l195_195343

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end problem_part1_problem_part2_l195_195343


namespace flower_position_after_50_beats_l195_195426

-- Define the number of students
def num_students : Nat := 7

-- Define the initial position of the flower
def initial_position : Nat := 1

-- Define the number of drum beats
def drum_beats : Nat := 50

-- Theorem stating that after 50 drum beats, the flower will be with the 2nd student
theorem flower_position_after_50_beats : 
  (initial_position + (drum_beats % num_students)) % num_students = 2 := by
  -- Start the proof (this part usually would contain the actual proof logic)
  sorry

end flower_position_after_50_beats_l195_195426


namespace solution_set_inequality_l195_195797

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

theorem solution_set_inequality (a x : ℝ) (h : Set.Ioo (-1 : ℝ) (2 : ℝ) = {x | |f a x| < 6}) : 
    {x | f a x ≤ 1} = {x | x ≥ 1 / 4} :=
sorry

end solution_set_inequality_l195_195797


namespace translate_parabola_l195_195099

theorem translate_parabola :
  (∀ x : ℝ, y = 2 * x^2 + 1) →
  (∀ x : ℝ, y' = y.subst (λ x, x + 1) - 3) →
  (∀ x : ℝ, y' = 2 * (x + 1)^2 - 2) :=
begin
  intros,
  rw H,
  sorry
end

end translate_parabola_l195_195099


namespace only_cats_count_l195_195867

/-
There are 69 people that own pets. Let:
- D = 15 people own only dogs.
- C be the number of people who own only cats.
- CD = 5 people own only cats and dogs.
- CDS = 3 people own cats, dogs, and snakes.
- NS = 39 people own snakes (cats or dogs incl.).

We are asked to prove that the number of people who own only cats is 10.
-/

theorem only_cats_count :
  ∃ C : ℕ, 
  let D := 15,
      CD := 5,
      CDS := 3,
      NS := 39,
      TotalOwners := 69 in
  (C + D + CD + CDS + (NS - CDS) = TotalOwners) →
  C = 10 :=
by
  sorry

end only_cats_count_l195_195867


namespace smallest_t_value_at_70_degrees_l195_195738

theorem smallest_t_value_at_70_degrees :
  ∃ t : ℝ, (-t^2 + 10 * t + 60 = 70) ∧ t = 5 + real.sqrt 35 := 
sorry

end smallest_t_value_at_70_degrees_l195_195738


namespace least_possible_value_l195_195889

noncomputable def least_value_expression (x : ℝ) : ℝ :=
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024

theorem least_possible_value : ∃ x : ℝ, least_value_expression x = 2023 :=
  sorry

end least_possible_value_l195_195889


namespace differential_at_zero_l195_195247

noncomputable def y (x : ℝ) : ℝ := Real.exp x * (x^2 + 3)

theorem differential_at_zero :
  let dy := (λ x dx, (deriv y x) * dx)
  dy 0 = 3 * dx :=
by
  sorry

end differential_at_zero_l195_195247


namespace problem_solution_l195_195169

def num_volunteers : ℕ := 5
def num_elderly : ℕ := 2

def total_arrangements : ℕ :=
  let perm_ends :=(num_volunteers - 2) * (num_volunteers - 1) -- Choosing 2 out of 5 volunteers to occupy the ends
  let entities := (num_volunteers - 2) + 1   -- Remaining volunteers and 1 entity consisting of the elderly
  let perm_entities := entities!   -- Total permutations of the entities
  let perm_within_entity := num_elderly! -- Permutations of the elderly people within the entity
  perm_ends * perm_entities * perm_within_entity

theorem problem_solution : total_arrangements = 960 := by
sorry

end problem_solution_l195_195169


namespace simon_change_l195_195054

theorem simon_change (price_pansy price_hydrangea price_petunia : ℝ)
  (num_pansies num_hydrangeas num_petunias : ℕ)
  (discount_percentage : ℝ)
  (payment : ℝ) :
  price_pansy = 2.5 →
  price_hydrangea = 12.5 →
  price_petunia = 1.0 →
  num_pansies = 5 →
  num_hydrangeas = 1 →
  num_petunias = 5 →
  discount_percentage = 0.10 →
  payment = 50.0 →
  let total_cost := num_pansies * price_pansy + num_hydrangeas * price_hydrangea + num_petunias * price_petunia in
  let total_cost_after_discount := total_cost * (1 - discount_percentage) in
  let change := payment - total_cost_after_discount in
  change = 23.0 :=
begin
  intros,
  sorry
end

end simon_change_l195_195054


namespace sequences_exists_l195_195655

def sequence_a (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a 1 * a 2 * ... * a n = 2 ^ b n

def sequence_b (b : ℕ → ℝ) : Prop :=
  ∃ a, a 1 = 2 ∧ (∀ n, n > 0 → a n = 2 ^ n ∧ b n = n * (n + 1) / 2)

def sequence_c (c : ℕ → ℝ) : Prop :=
  ∃ b : ℕ → ℝ, c = λ n, 2 * b n / n ^ 2

theorem sequences_exists (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (m n : ℕ) :
  sequence_a a ∧ sequence_b b ∧ sequence_c c ∧ (m ≠ n) →
  a 1 = 2 ∧ b 3 - b 2 = 3 ∧ (m > 0 ∧ n > 0 ∧ m ≠ n) →
  (∃ a_3 : ℝ, a 3 = 8) ∧ (∀ n, b n = n * (n + 1) / 2) ∧ 
  (1 + 1 / (2 : ℝ)), (1 + 1 / (m : ℝ)), (1 + 1 / (n : ℝ)) form an arithmetic sequence :=
by 
  sorry

end sequences_exists_l195_195655


namespace non_self_intersecting_pentagon_one_side_l195_195418

-- Define what it means to be a non-self-intersecting pentagon
variable (A B C D E : Type) [Affine A] 

structure non_self_intersecting (pentagon : set A) : Prop :=
(is_convex : convex pentagon)
(no_self_intersection : ∀ {p1 p2 : set A}, p1 ≠ p2 → ¬(p1 ∩ p2 ≠ ∅))

-- Define the theorem to be proved
theorem non_self_intersecting_pentagon_one_side (pentagon : set A)
  [non_self_intersecting pentagon] :
  ∃ (s : line_seg pentagon), 
    ∀ (v : A), v ∈ pentagon.vertices → 
    (v ∉ pentagon.interior_side s ∨ v ∉ pentagon.exterior_side s) :=
sorry

end non_self_intersecting_pentagon_one_side_l195_195418


namespace Y_lies_on_circumcircle_of_ABC_l195_195932

-- Defining the key entities and conditions as variables and assumptions
variables {A B C X P Q Y : Point}
variables {triangle : Triangle A B C}

-- Assuming the isosceles property of the triangle
axiom isosceles_ABC : triangle.is_isosceles A B C (eq AB AC)

-- Assuming X is a point on BC
axiom X_on_BC : X ∈ line_segment B C

-- Assuming P and Q are points on AB and AC respectively, and APXQ is a parallelogram
axiom P_on_AB : P ∈ line_segment A B
axiom Q_on_AC : Q ∈ line_segment A C
axiom APXQ_parallelogram : parallelogram A P X Q

-- Assuming Y is the point symmetrical to X with respect to line PQ
axiom Y_symmetrical : symmetrical_point X PQ Y

-- The theorem to prove
theorem Y_lies_on_circumcircle_of_ABC :
  lies_on_circumcircle Y (triangle ABC) :=
by sorry

end Y_lies_on_circumcircle_of_ABC_l195_195932


namespace find_number_divided_by_6_l195_195527

theorem find_number_divided_by_6 (x : ℤ) (h : (x + 17) / 5 = 25) : x / 6 = 18 :=
by
  sorry

end find_number_divided_by_6_l195_195527


namespace number_problem_l195_195344

theorem number_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 34) / 10 = 2 := by
  sorry

end number_problem_l195_195344


namespace distance_inequality_l195_195548

-- Definitions for the problem conditions
variables {A B C D M : Type*} 
variables {a b c : ℝ}
variables (AD BD CD DM : ℝ) 
variables (S : ℝ)
variables (X : Type*)

-- The edges AD, BD, and CD are mutually perpendicular with lengths a, b, and c
def edges_perpendicular (AD BD CD : ℝ) (a b c : ℝ) : Prop :=
  AD = a ∧ BD = b ∧ CD = c

-- S represents the sum of distances from vertices A, B, and C to the line DM
def sum_of_distances (A B C D M : Type*) (DM : ℝ) (S : ℝ) : Prop :=
  S = dist A DM + dist B DM + dist C DM

-- The theorem statement
theorem distance_inequality 
  (h_edges : edges_perpendicular AD BD CD a b c) 
  (h_sum : sum_of_distances A B C D M DM S) :
  S ≤ sqrt (2 * (a^2 + b^2 + c^2)) :=
sorry

end distance_inequality_l195_195548


namespace mark_profit_from_selling_magic_card_l195_195033

theorem mark_profit_from_selling_magic_card : 
    ∀ (purchase_price new_value profit : ℕ), 
        purchase_price = 100 ∧ 
        new_value = 3 * purchase_price ∧ 
        profit = new_value - purchase_price 
    → 
        profit = 200 := 
by 
  intros purchase_price new_value profit h,
  cases h with hp1 h,
  cases h with hv1 hp2,
  rw hp1 at hv1,
  rw hp1 at hp2,
  rw hv1 at hp2,
  rw hp2,
  rw hp1,
  norm_num,
  exact eq.refl 200

end mark_profit_from_selling_magic_card_l195_195033


namespace mr_callen_total_loss_l195_195807

noncomputable def total_loss : ℕ :=
  let n_paintings := 10
  let cost_painting := 40
  let n_wooden_toys := 8
  let cost_wooden_toy := 20
  let reduction_painting := 0.10
  let reduction_wooden_toy := 0.15

  let loss_per_painting := cost_painting * reduction_painting
  let total_loss_paintings := n_paintings * loss_per_painting

  let loss_per_wooden_toy := cost_wooden_toy * reduction_wooden_toy
  let total_loss_wooden_toys := n_wooden_toys * loss_per_wooden_toy

  ((total_loss_paintings + total_loss_wooden_toys).toNat)

theorem mr_callen_total_loss : total_loss = 64 := by
  sorry

end mr_callen_total_loss_l195_195807


namespace median_equality_of_sorted_quartet_range_subsets_l195_195280

variable {α : Type*} [LinearOrder α] [Add α] [Div α]
variable (x1 x2 x3 x4 x5 x6 : α)

theorem median_equality_of_sorted_quartet :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x3 + x4) / 2 = (x3 + x4) / 2 :=
sorry

theorem range_subsets :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_of_sorted_quartet_range_subsets_l195_195280


namespace number_of_n_l195_195395

def T_k (k : ℕ) : ℕ := List.prod (List.take k [3, 5, 7, 11, 13, 17, 19, 23, 29, 31])

theorem number_of_n (k : ℕ) : 
  let t_k := T_k k in
  (finset.card {n : ℕ | ∃ m : ℕ, m^2 = n^2 + t_k * n} = (3^k - 1) / 2) :=
by
  sorry

end number_of_n_l195_195395


namespace cousin_typing_time_l195_195809

theorem cousin_typing_time (speed_ratio : ℕ) (my_time_hours : ℕ) (minutes_per_hour : ℕ) (my_time_minutes : ℕ) :
  speed_ratio = 4 →
  my_time_hours = 3 →
  minutes_per_hour = 60 →
  my_time_minutes = my_time_hours * minutes_per_hour →
  ∃ (cousin_time : ℕ), cousin_time = my_time_minutes / speed_ratio := by
  sorry

end cousin_typing_time_l195_195809


namespace triangle_equilateral_of_equal_angle_ratios_l195_195856

theorem triangle_equilateral_of_equal_angle_ratios
  (a b c : ℝ)
  (h₁ : a + b + c = 180)
  (h₂ : a = b)
  (h₃ : b = c) :
  a = 60 ∧ b = 60 ∧ c = 60 :=
by
  sorry

end triangle_equilateral_of_equal_angle_ratios_l195_195856


namespace ratio_of_area_of_square_to_circle_l195_195584

theorem ratio_of_area_of_square_to_circle (r : ℝ) (s : ℝ) 
  (h1 : s = r * ℝ.sqrt 2) : 
  (s * s) / (π * r * r) = 2 / π :=
by
  sorry

end ratio_of_area_of_square_to_circle_l195_195584


namespace triangle_area_is_48_l195_195886

noncomputable def triangle_area (a b median : ℝ) : ℝ := 
  let d := (1 / 2) * b -- The base length of the corresponding sub-triangle ABD
  let h := (a^2 - d^2).sqrt -- The height of the sub-triangle ABD calculated using Pythagorean theorem
  in (1 / 2) * b * h

theorem triangle_area_is_48 (A B : ℝ) (M : ℝ) (hA : A = 10) (hB : B = 12) (hM : M = 5) :
  triangle_area A B M = 48 :=
by
  rw [hA, hB, hM]
  norm_num
  sorry -- proof to be completed

end triangle_area_is_48_l195_195886


namespace y_intercept_of_line_b_l195_195020

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l195_195020


namespace gittes_age_is_25_l195_195535

-- Define the known year and birthday conditions
def year : ℕ := 2003
def birthday_month_day : ℤ × ℤ := (4, 22)

-- Define a function to sum the digits of a year
def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000 in
  let d2 := (n % 1000) / 100 in
  let d3 := (n % 100) / 10 in
  let d4 := n % 10 in
  d1 + d2 + d3 + d4

-- Define a function to calculate age
def age (birth_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - birth_year

-- Gittes was born in a year such that her age equals the sum of its digits
def gittes_birth_year : ℕ :=
  let possible_year := 1000 in
  1978 -- determined from the problem

-- Final proof statement that Gittes is 25 years old
theorem gittes_age_is_25 : age gittes_birth_year year = 25 :=
by
  have h1 : gottes_birth_year = 1978 := rfl
  calc 
    age gottes_birth_year year
        = year - gottes_birth_year : rfl
    ... = 25 : rfl
    -- More detailed proof steps verifying the birth year determined can be added here
    sorry

end gittes_age_is_25_l195_195535


namespace box_height_l195_195151

theorem box_height (volume length width : ℝ) (h : ℝ) (h_volume : volume = 315) (h_length : length = 7) (h_width : width = 9) :
  h = 5 :=
by
  -- Proof would go here
  sorry

end box_height_l195_195151


namespace calculator_press_count_l195_195917

theorem calculator_press_count : 
  ∃ n : ℕ, n ≥ 4 ∧ (2 ^ (2 ^ n)) > 500 := 
by
  sorry

end calculator_press_count_l195_195917


namespace balloon_arrangements_correct_l195_195215

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l195_195215


namespace balloon_arrangements_correct_l195_195214

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l195_195214


namespace sum_of_perimeters_l195_195086

theorem sum_of_perimeters (A1 A2 : ℝ) (h1 : A1 + A2 = 145) (h2 : A1 - A2 = 25) :
  4 * Real.sqrt 85 + 4 * Real.sqrt 60 = 4 * Real.sqrt A1 + 4 * Real.sqrt A2 :=
by
  sorry

end sum_of_perimeters_l195_195086


namespace time_to_complete_job_l195_195133

-- Define the conditions
variables {A B : ℕ} -- Efficiencies of A and B

-- Assume B's efficiency is 100 units, and A is 130 units.
def efficiency_A : ℕ := 130
def efficiency_B : ℕ := 100

-- Given: A can complete the job in 23 days
def days_A : ℕ := 23

-- Compute total work W. Since A can complete the job in 23 days and its efficiency is 130 units/day:
def total_work : ℕ := efficiency_A * days_A

-- Combined efficiency of A and B
def combined_efficiency : ℕ := efficiency_A + efficiency_B

-- Determine the time taken by A and B working together
def time_A_B_together : ℕ := total_work / combined_efficiency

-- Prove that the time A and B working together is 13 days
theorem time_to_complete_job : time_A_B_together = 13 :=
by
  sorry -- Proof is omitted as per instructions

end time_to_complete_job_l195_195133


namespace not_prime_712_fact_plus_one_l195_195986

theorem not_prime_712_fact_plus_one :
  ¬(Nat.Prime (712! + 1)) :=
by
  -- We will skip the proof steps here
  sorry

end not_prime_712_fact_plus_one_l195_195986


namespace find_denomination_of_bills_produced_by_press_F_l195_195111

noncomputable def denomination_of_bills_produced_by_press_F := 81

axiom press_F_rate_per_minute : ℕ := 1000
axiom press_T_denomination : ℕ := 20
axiom press_T_rate_per_minute : ℕ := 200
axiom press_F_produces_50_dollars_more_in_3_seconds : 
  3 * F * (press_F_rate_per_minute / 60) = 3 * press_T_denomination * (press_T_rate_per_minute / 60) + 50

theorem find_denomination_of_bills_produced_by_press_F : 
  denomination_of_bills_produced_by_press_F = 81 := by
  sorry

end find_denomination_of_bills_produced_by_press_F_l195_195111


namespace least_product_of_distinct_primes_greater_than_30_l195_195512

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l195_195512


namespace repriced_books_l195_195349

def total_books : ℤ := 1452
def initial_price : ℤ := 42
def new_price : ℤ := 45

def total_money := total_books * initial_price

def money_left_over := total_money % new_price

theorem repriced_books (total_books = 1452) (initial_price = 42) (new_price = 45) : money_left_over = 9 := by
  sorry

end repriced_books_l195_195349


namespace problem_solution_l195_195602

noncomputable def circle_area (O : Type) [MetricSpace O] (A B C D : O)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : Diameter ∂ (B, C))
  (h3 : Midpoint D arc AC)
  (hAB : dist A B = 2)
  (hAD : dist A D = 2 * Real.sqrt 3) : Real :=
by
  -- Proof goes here.
  sorry

theorem problem_solution (O : Type) [MetricSpace O] (A B C D : O)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : Diameter ∂ (B, C))
  (h3 : Midpoint D arc AC)
  (hAB : dist A B = 2)
  (hAD : dist A D = 2 * Real.sqrt 3) :
  circle_area O A B C D h1 h2 h3 hAB hAD = 9 * Real.pi := 
by
  -- Proof goes here.
  sorry

end problem_solution_l195_195602


namespace total_trees_after_planting_l195_195868

def current_trees : ℕ := 7
def trees_planted_today : ℕ := 5
def trees_planted_tomorrow : ℕ := 4

theorem total_trees_after_planting : 
  current_trees + trees_planted_today + trees_planted_tomorrow = 16 :=
by
  sorry

end total_trees_after_planting_l195_195868


namespace candy_from_sister_l195_195262

variable (total_neighbors : Nat) (pieces_per_day : Nat) (days : Nat) (total_pieces : Nat)
variable (pieces_per_day_eq : pieces_per_day = 9)
variable (days_eq : days = 9)
variable (total_neighbors_eq : total_neighbors = 66)
variable (total_pieces_eq : total_pieces = 81)

theorem candy_from_sister : 
  total_pieces = total_neighbors + 15 :=
by
  sorry

end candy_from_sister_l195_195262


namespace fraction_of_a_eq_1_fifth_of_b_l195_195132

theorem fraction_of_a_eq_1_fifth_of_b (a b : ℝ) (x : ℝ) 
  (h1 : a + b = 100) 
  (h2 : (1/5) * b = 12)
  (h3 : b = 60) : x = 3/10 := by
  sorry

end fraction_of_a_eq_1_fifth_of_b_l195_195132


namespace geometric_sequence_seventh_term_l195_195120

theorem geometric_sequence_seventh_term :
  let a := 6
  let r := -2
  (a * r^(7 - 1)) = 384 := 
by
  sorry

end geometric_sequence_seventh_term_l195_195120


namespace find_n_l195_195542

-- Definitions for the conditions
def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Given conditions
variables (n : ℕ)
axiom lcm_n_12 : lcm n 12 = 54
axiom gcd_n_12 : gcd n 12 = 8

-- Proof statement
theorem find_n : n = 36 :=
by
  have h1 : 54 * 8 = n * 12 := by sorry
  have h2 : 432 = n * 12 := by sorry
  have h3 : n = 36 := by sorry
  exact h3

end find_n_l195_195542


namespace cubic_root_is_neg_one_l195_195301

theorem cubic_root_is_neg_one (a b c d : ℝ) (k : ℝ) 
  (h1 : b = a - k) 
  (h2 : c = a - 2 * k)
  (h3 : d = a - 3 * k)
  (h4 : a ≥ b)
  (h5 : b ≥ c)
  (h6 : c ≥ d)
  (h7 : d ≥ 0)
  (h8 : has_two_distinct_real_roots : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (a * r1^3 + b * r1^2 + c * r1 + d = 0) ∧ (a * r2^3 + b * r2^2 + c * r2 + d = 0) ∧ (∀ r : ℝ, a * r^3 + b * r^2 + c * r + d = 0 → r = r1 ∨ r = r2)) :
  ∃ r : ℝ, r = -1 :=
sorry

end cubic_root_is_neg_one_l195_195301


namespace total_marbles_l195_195103

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l195_195103


namespace normal_price_of_article_l195_195141

theorem normal_price_of_article 
  (final_price : ℝ)
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (P : ℝ)
  (h : final_price = 108) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20)
  (h_eq : (1 - discount1) * (1 - discount2) * P = final_price) :
  P = 150 := by
  sorry

end normal_price_of_article_l195_195141
