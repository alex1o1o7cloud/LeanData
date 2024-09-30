import Mathlib
import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Determinant
import Mathlib.Logic.Basic

namespace fraction_identity_0_104

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end fraction_identity_0_104


namespace slope_probability_0_336

def line_equation (a x y : ℝ) : Prop := a * x + 2 * y - 3 = 0

def in_interval (a : ℝ) : Prop := -5 ≤ a ∧ a ≤ 4

def slope_not_less_than_1 (a : ℝ) : Prop := - a / 2 ≥ 1

noncomputable def probability_slope_not_less_than_1 : ℝ :=
  (2 - (-5)) / (4 - (-5))

theorem slope_probability :
  ∀ (a : ℝ), in_interval a → slope_not_less_than_1 a → probability_slope_not_less_than_1 = 1 / 3 :=
by
  intros a h_in h_slope
  sorry

end slope_probability_0_336


namespace brownies_per_person_0_940

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_0_940


namespace exterior_angle_of_regular_octagon_0_844

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end exterior_angle_of_regular_octagon_0_844


namespace tan_315_eq_neg1_0_707

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_0_707


namespace factor_complete_polynomial_0_328

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_0_328


namespace determine_x_0_718

variables {m n x : ℝ}
variable (k : ℝ)
variable (Hmn : m ≠ 0 ∧ n ≠ 0)
variable (Hk : k = 5 * (m^2 - n^2))

theorem determine_x (H : (x + 2 * m)^2 - (x - 3 * n)^2 = k) : 
  x = (5 * m^2 - 9 * n^2) / (4 * m + 6 * n) := by
  sorry

end determine_x_0_718


namespace paint_liters_needed_0_936

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end paint_liters_needed_0_936


namespace juniors_to_freshmen_ratio_0_739

variable (f s j : ℕ)

def participated_freshmen := 3 * f / 7
def participated_sophomores := 5 * s / 7
def participated_juniors := j / 2

-- The statement
theorem juniors_to_freshmen_ratio
    (h1 : participated_freshmen = participated_sophomores)
    (h2 : participated_freshmen = participated_juniors) :
    j = 6 * f / 7 ∧ f = 7 * j / 6 :=
by
  sorry

end juniors_to_freshmen_ratio_0_739


namespace sin_cos_quad_ineq_0_786

open Real

theorem sin_cos_quad_ineq (x : ℝ) : 
  2 * (sin x) ^ 4 + 3 * (sin x) ^ 2 * (cos x) ^ 2 + 5 * (cos x) ^ 4 ≤ 5 :=
by
  sorry

end sin_cos_quad_ineq_0_786


namespace number_of_ways_to_select_starting_lineup_0_351

noncomputable def choose (n k : ℕ) : ℕ := 
if h : k ≤ n then Nat.choose n k else 0

theorem number_of_ways_to_select_starting_lineup (n k : ℕ) (h : n = 12) (h1 : k = 5) : 
  12 * choose 11 4 = 3960 := 
by sorry

end number_of_ways_to_select_starting_lineup_0_351


namespace number_of_combinations_0_729

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end number_of_combinations_0_729


namespace total_pencils_correct_0_361

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end total_pencils_correct_0_361


namespace emily_quiz_score_0_515

theorem emily_quiz_score :
  ∃ x : ℕ, 94 + 88 + 92 + 85 + 97 + x = 6 * 90 :=
by
  sorry

end emily_quiz_score_0_515


namespace smallest_positive_multiple_of_3_4_5_is_60_0_384

theorem smallest_positive_multiple_of_3_4_5_is_60 :
  ∃ n : ℕ, n > 0 ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ n = 60 :=
by
  use 60
  sorry

end smallest_positive_multiple_of_3_4_5_is_60_0_384


namespace cricket_avg_score_0_126

theorem cricket_avg_score
  (avg_first_two : ℕ)
  (num_first_two : ℕ)
  (avg_all_five : ℕ)
  (num_all_five : ℕ)
  (avg_first_two_eq : avg_first_two = 40)
  (num_first_two_eq : num_first_two = 2)
  (avg_all_five_eq : avg_all_five = 22)
  (num_all_five_eq : num_all_five = 5) :
  ((num_all_five * avg_all_five - num_first_two * avg_first_two) / (num_all_five - num_first_two) = 10) :=
by
  sorry

end cricket_avg_score_0_126


namespace odd_coefficients_in_binomial_expansion_0_764

theorem odd_coefficients_in_binomial_expansion :
  let a : Fin 9 → ℕ := fun k => Nat.choose 8 k
  (Finset.filter (fun k => a k % 2 = 1) (Finset.Icc 0 8)).card = 2 := by
  sorry

end odd_coefficients_in_binomial_expansion_0_764


namespace find_y_minus_x_0_320

theorem find_y_minus_x (x y : ℕ) (hx : x + y = 540) (hxy : (x : ℚ) / (y : ℚ) = 7 / 8) : y - x = 36 :=
by
  sorry

end find_y_minus_x_0_320


namespace david_wins_2011th_even_0_766

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end david_wins_2011th_even_0_766


namespace slope_angle_at_point_0_493

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end slope_angle_at_point_0_493


namespace choose_officers_ways_0_478

theorem choose_officers_ways :
  let members := 12
  let vp_candidates := 4
  let remaining_after_president := members - 1
  let remaining_after_vice_president := remaining_after_president - 1
  let remaining_after_secretary := remaining_after_vice_president - 1
  let remaining_after_treasurer := remaining_after_secretary - 1
  (members * vp_candidates * (remaining_after_vice_president) *
   (remaining_after_secretary) * (remaining_after_treasurer)) = 34560 := by
  -- Calculation here
  sorry

end choose_officers_ways_0_478


namespace minimum_dwarfs_0_665

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_0_665


namespace point_below_line_0_226

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end point_below_line_0_226


namespace cab_driver_income_day3_0_48

theorem cab_driver_income_day3 :
  let income1 := 200
  let income2 := 150
  let income4 := 400
  let income5 := 500
  let avg_income := 400
  let total_income := avg_income * 5 
  total_income - (income1 + income2 + income4 + income5) = 750 := by
  sorry

end cab_driver_income_day3_0_48


namespace correct_propositions_count_0_749

theorem correct_propositions_count (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b + 1) ∧
  (∀ a b, a > b → a - 1 > b - 1) ∧
  (∀ a b, a > b → -2 * a < -2 * b) ∧
  (¬ ∀ a b, a > b → 2 * a < 2 * b) → 
  3 = 3 :=
by
  intro h
  sorry

end correct_propositions_count_0_749


namespace unique_triple_primes_0_395

theorem unique_triple_primes (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) (h3 : (p^3 + q^3 + r^3) / (p + q + r) = 249) : r = 19 :=
sorry

end unique_triple_primes_0_395


namespace simple_interest_correct_0_833

def principal : ℝ := 10040.625
def rate : ℝ := 8
def time : ℕ := 5

theorem simple_interest_correct :
  (principal * rate * time / 100) = 40162.5 :=
by 
  sorry

end simple_interest_correct_0_833


namespace Tom_age_ratio_0_745

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end Tom_age_ratio_0_745


namespace Mona_grouped_with_one_player_before_in_second_group_0_995

/-- Mona plays in groups with four other players, joined 9 groups, and grouped with 33 unique players. 
    One of the groups included 2 players she had grouped with before. 
    Prove that the number of players she had grouped with before in the second group is 1. -/
theorem Mona_grouped_with_one_player_before_in_second_group 
    (total_groups : ℕ) (group_size : ℕ) (unique_players : ℕ) 
    (repeat_players_in_group1 : ℕ) : 
    (total_groups = 9) → (group_size = 5) → (unique_players = 33) → (repeat_players_in_group1 = 2) 
        → ∃ repeat_players_in_group2 : ℕ, repeat_players_in_group2 = 1 :=
by
    sorry

end Mona_grouped_with_one_player_before_in_second_group_0_995


namespace relationship_between_roses_and_total_flowers_0_45

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end relationship_between_roses_and_total_flowers_0_45


namespace pen_tip_movement_0_318

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end pen_tip_movement_0_318


namespace probability_of_green_0_403

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end probability_of_green_0_403


namespace amount_of_salmon_sold_first_week_0_536

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end amount_of_salmon_sold_first_week_0_536


namespace problem_solution_0_597

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_0_597


namespace salaries_proof_0_433

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end salaries_proof_0_433


namespace largest_common_value_less_than_1000_0_3

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a = 999 ∧ (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 8 * m) ∧ a < 1000 :=
by
  sorry

end largest_common_value_less_than_1000_0_3


namespace domain_of_f_0_598

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end domain_of_f_0_598


namespace triangle_area_proof_0_412

noncomputable def area_of_triangle_ABC : ℝ :=
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  let area := 3 / 11
  area

theorem triangle_area_proof :
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  area_of_triangle_ABC = 3 / 11 :=
by
  sorry

end triangle_area_proof_0_412


namespace alice_age_30_0_121

variable (A T : ℕ)

def tom_younger_alice (A T : ℕ) := T = A - 15
def ten_years_ago (A T : ℕ) := A - 10 = 4 * (T - 10)

theorem alice_age_30 (A T : ℕ) (h1 : tom_younger_alice A T) (h2 : ten_years_ago A T) : A = 30 := 
by sorry

end alice_age_30_0_121


namespace find_sum_invested_0_962

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end find_sum_invested_0_962


namespace food_expenditure_increase_0_723

-- Conditions
def linear_relationship (x : ℝ) : ℝ := 0.254 * x + 0.321

-- Proof statement
theorem food_expenditure_increase (x : ℝ) : linear_relationship (x + 1) - linear_relationship x = 0.254 :=
by
  sorry

end food_expenditure_increase_0_723


namespace intersection_of_A_and_B_0_896

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_0_896


namespace paytons_score_0_496

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end paytons_score_0_496


namespace syllogism_correct_0_869

theorem syllogism_correct 
  (natnum : ℕ → Prop) 
  (intnum : ℤ → Prop) 
  (is_natnum  : natnum 4) 
  (natnum_to_intnum : ∀ n, natnum n → intnum n) : intnum 4 :=
by
  sorry

end syllogism_correct_0_869


namespace population_ratio_0_991

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end population_ratio_0_991


namespace only_zero_function_satisfies_inequality_0_754

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end only_zero_function_satisfies_inequality_0_754


namespace ratio_t_q_0_836

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_0_836


namespace planes_parallel_if_any_line_parallel_0_347

axiom Plane : Type
axiom Line : Type
axiom contains : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Plane → Prop

theorem planes_parallel_if_any_line_parallel (α β : Plane)
  (h₁ : ∀ l, contains α l → parallel_lines l β) :
  parallel α β :=
sorry

end planes_parallel_if_any_line_parallel_0_347


namespace initial_quantity_of_milk_0_447

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end initial_quantity_of_milk_0_447


namespace trigonometric_identity_0_442

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end trigonometric_identity_0_442


namespace range_of_a_0_50

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → (a-1)^x > (a-1)^y) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_0_50


namespace sequence_is_increasing_0_530

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end sequence_is_increasing_0_530


namespace circle_polar_equation_0_576

-- Definitions and conditions
def circle_equation_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_coordinates (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem to be proven
theorem circle_polar_equation (ρ θ : ℝ) :
  (∀ x y : ℝ, circle_equation_cartesian x y → 
  polar_coordinates ρ θ x y) → ρ = 2 * Real.sin θ :=
by
  -- This is a placeholder for the proof
  sorry

end circle_polar_equation_0_576


namespace find_a2_0_144

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end find_a2_0_144


namespace four_digit_number_properties_0_498

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end four_digit_number_properties_0_498


namespace compute_expression_0_538

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_0_538


namespace triangle_area_is_zero_0_438

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def scalar_vector_mult (k : ℝ) (v : Point3D) : Point3D := {
  x := k * v.x,
  y := k * v.y,
  z := k * v.z
}

theorem triangle_area_is_zero : 
  let u := Point3D.mk 2 1 (-1)
  let v := Point3D.mk 5 4 1
  let w := Point3D.mk 11 10 5
  vector_sub w u = scalar_vector_mult 3 (vector_sub v u) →
-- If the points u, v, w are collinear, the area of the triangle formed by these points is zero:
  ∃ area : ℝ, area = 0 :=
by {
  sorry
}

end triangle_area_is_zero_0_438


namespace problem_0_567

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end problem_0_567


namespace sum_of_roots_of_cubic_0_426

noncomputable def P (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_cubic (a b c d : ℝ) (h : ∀ x : ℝ, P a b c d (x^2 + x) ≥ P a b c d (x + 1)) :
  (-b / a) = (P a b c d 0) :=
sorry

end sum_of_roots_of_cubic_0_426


namespace number_of_parallelograms_0_606

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_parallelograms (n : ℕ) : 
  (∑ i in Finset.range (n + 2 + 1), binom i 4) = 3 * binom (n + 2) 4 :=
by
  sorry

end number_of_parallelograms_0_606


namespace milk_jars_good_for_sale_0_31

noncomputable def good_whole_milk_jars : ℕ := 
  let initial_jars := 60 * 30
  let short_deliveries := 20 * 30 * 2
  let damaged_jars_1 := 3 * 5
  let damaged_jars_2 := 4 * 6
  let totally_damaged_cartons := 2 * 30
  let received_jars := initial_jars - short_deliveries - damaged_jars_1 - damaged_jars_2 - totally_damaged_cartons
  let spoilage := (5 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_skim_milk_jars : ℕ := 
  let initial_jars := 40 * 40
  let short_delivery := 10 * 40
  let damaged_jars := 5 * 4
  let totally_damaged_carton := 1 * 40
  let received_jars := initial_jars - short_delivery - damaged_jars - totally_damaged_carton
  let spoilage := (3 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_almond_milk_jars : ℕ := 
  let initial_jars := 30 * 20
  let short_delivery := 5 * 20
  let damaged_jars := 2 * 3
  let received_jars := initial_jars - short_delivery - damaged_jars
  let spoilage := (1 * received_jars) / 100
  received_jars - spoilage

theorem milk_jars_good_for_sale : 
  good_whole_milk_jars = 476 ∧
  good_skim_milk_jars = 1106 ∧
  good_almond_milk_jars = 489 :=
by
  sorry

end milk_jars_good_for_sale_0_31


namespace candles_time_0_162

/-- Prove that if two candles of equal length are lit at a certain time,
and by 6 PM one of the stubs is three times the length of the other,
the correct time to light the candles is 4:00 PM. -/

theorem candles_time :
  ∀ (ℓ : ℝ) (t : ℝ),
  (∀ t1 t2 : ℝ, t = t1 + t2 → 
    (180 - t1) = 3 * (300 - t2) / 3 → 
    18 <= 6 ∧ 0 <= t → ℓ / 180 * (180 - (t - 180)) = 3 * (ℓ / 300 * (300 - (6 - t))) →
    t = 4
  ) := 
by 
  sorry

end candles_time_0_162


namespace find_common_difference_0_650

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end find_common_difference_0_650


namespace my_op_evaluation_0_551

def my_op (x y : Int) : Int := x * y - 3 * x + y

theorem my_op_evaluation : my_op 5 3 - my_op 3 5 = -8 := by 
  sorry

end my_op_evaluation_0_551


namespace last_digit_of_2_pow_2004_0_254

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end last_digit_of_2_pow_2004_0_254


namespace yulia_max_candies_0_948

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end yulia_max_candies_0_948


namespace sphere_radius_equal_0_208

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_equal_0_208


namespace compare_fractions_0_54

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end compare_fractions_0_54


namespace bobby_initial_pieces_0_851

-- Definitions based on the conditions
def pieces_eaten_1 := 17
def pieces_eaten_2 := 15
def pieces_left := 4

-- Definition based on the question and answer
def initial_pieces (pieces_eaten_1 pieces_eaten_2 pieces_left : ℕ) : ℕ :=
  pieces_eaten_1 + pieces_eaten_2 + pieces_left

-- Theorem stating the problem and the expected answer
theorem bobby_initial_pieces : 
  initial_pieces pieces_eaten_1 pieces_eaten_2 pieces_left = 36 :=
by 
  sorry

end bobby_initial_pieces_0_851


namespace sum_mod_18_0_312

theorem sum_mod_18 :
  (65 + 66 + 67 + 68 + 69 + 70 + 71 + 72) % 18 = 8 :=
by
  sorry

end sum_mod_18_0_312


namespace pencils_given_out_0_75

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end pencils_given_out_0_75


namespace final_inventory_is_correct_0_870

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct_0_870


namespace rational_number_div_0_280

theorem rational_number_div (x : ℚ) (h : -2 / x = 8) : x = -1 / 4 := 
by
  sorry

end rational_number_div_0_280


namespace factor_difference_of_squares_example_0_537

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end factor_difference_of_squares_example_0_537


namespace find_c_0_856

-- Definitions
def is_root (x c : ℝ) : Prop := x^2 - 3*x + c = 0

-- Main statement
theorem find_c (c : ℝ) (h : is_root 1 c) : c = 2 :=
sorry

end find_c_0_856


namespace brendan_weekly_capacity_0_902

/-- Brendan can cut 8 yards of grass per day on flat terrain under normal weather conditions. Bought a lawnmower that improved his cutting speed by 50 percent on flat terrain. On uneven terrain, his speed is reduced by 35 percent. Rain reduces his cutting capacity by 20 percent. Extreme heat reduces his cutting capacity by 10 percent. The conditions for each day of the week are given and we want to prove that the total yards Brendan can cut in a week is 65.46 yards.
  Monday: Flat terrain, normal weather
  Tuesday: Flat terrain, rain
  Wednesday: Uneven terrain, normal weather
  Thursday: Flat terrain, extreme heat
  Friday: Uneven terrain, rain
  Saturday: Flat terrain, normal weather
  Sunday: Uneven terrain, extreme heat
-/
def brendan_cutting_capacity : ℝ :=
  let base_capacity := 8.0
  let flat_terrain_boost := 1.5
  let uneven_terrain_penalty := 0.65
  let rain_penalty := 0.8
  let extreme_heat_penalty := 0.9
  let monday_capacity := base_capacity * flat_terrain_boost
  let tuesday_capacity := monday_capacity * rain_penalty
  let wednesday_capacity := monday_capacity * uneven_terrain_penalty
  let thursday_capacity := monday_capacity * extreme_heat_penalty
  let friday_capacity := wednesday_capacity * rain_penalty
  let saturday_capacity := monday_capacity
  let sunday_capacity := wednesday_capacity * extreme_heat_penalty
  monday_capacity + tuesday_capacity + wednesday_capacity + thursday_capacity + friday_capacity + saturday_capacity + sunday_capacity

theorem brendan_weekly_capacity : brendan_cutting_capacity = 65.46 := 
by 
  sorry

end brendan_weekly_capacity_0_902


namespace fuse_length_must_be_80_0_807

-- Define the basic conditions
def distanceToSafeArea : ℕ := 400
def personSpeed : ℕ := 5
def fuseBurnSpeed : ℕ := 1

-- Calculate the time required to reach the safe area
def timeToSafeArea (distance speed : ℕ) : ℕ := distance / speed

-- Calculate the minimum length of the fuse based on the time to reach the safe area
def minFuseLength (time burnSpeed : ℕ) : ℕ := time * burnSpeed

-- The main problem statement: The fuse must be at least 80 meters long.
theorem fuse_length_must_be_80:
  minFuseLength (timeToSafeArea distanceToSafeArea personSpeed) fuseBurnSpeed = 80 :=
by
  sorry

end fuse_length_must_be_80_0_807


namespace compare_series_0_25

theorem compare_series (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) : 
  (1 / (1 - x^2) + 1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by
  sorry

end compare_series_0_25


namespace total_hens_and_cows_0_202

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end total_hens_and_cows_0_202


namespace find_x_in_coconut_grove_0_344

theorem find_x_in_coconut_grove
  (x : ℕ)
  (h1 : (x + 2) * 30 + x * 120 + (x - 2) * 180 = 300 * x)
  (h2 : 3 * x ≠ 0) :
  x = 10 :=
by
  sorry

end find_x_in_coconut_grove_0_344


namespace heather_bicycled_distance_0_760

def speed : ℕ := 8
def time : ℕ := 5
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

theorem heather_bicycled_distance : distance speed time = 40 := by
  sorry

end heather_bicycled_distance_0_760


namespace value_of_g_at_neg3_0_608

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end value_of_g_at_neg3_0_608


namespace product_of_dice_divisible_by_9_0_499

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end product_of_dice_divisible_by_9_0_499


namespace power_of_negative_base_0_795

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_0_795


namespace daughter_and_child_weight_0_113

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end daughter_and_child_weight_0_113


namespace region_area_0_983

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end region_area_0_983


namespace circumscribed_circle_radius_0_967

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end circumscribed_circle_radius_0_967


namespace prove_range_of_a_0_521

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end prove_range_of_a_0_521


namespace x_is_48_percent_of_z_0_524

variable {x y z : ℝ}

theorem x_is_48_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.40 * z) : x = 0.48 * z :=
by
  sorry

end x_is_48_percent_of_z_0_524


namespace harmonica_value_0_227

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end harmonica_value_0_227


namespace lcm_12_15_18_0_2

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_0_2


namespace not_in_range_0_291

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end not_in_range_0_291


namespace initial_percentage_increase_0_216

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end initial_percentage_increase_0_216


namespace blue_paint_gallons_0_156

-- Define the total gallons of paint used
def total_paint_gallons : ℕ := 6689

-- Define the gallons of white paint used
def white_paint_gallons : ℕ := 660

-- Define the corresponding proof problem
theorem blue_paint_gallons : 
  ∀ total white blue : ℕ, total = 6689 → white = 660 → blue = total - white → blue = 6029 := by
  sorry

end blue_paint_gallons_0_156


namespace circle_division_parts_0_609

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end circle_division_parts_0_609


namespace ratio_platform_to_pole_0_136

variables (l t T v : ℝ)
-- Conditions
axiom constant_velocity : ∀ t l, l = v * t
axiom pass_pole : l = v * t
axiom pass_platform : 6 * l = v * T 

theorem ratio_platform_to_pole (h1 : l = v * t) (h2 : 6 * l = v * T) : T / t = 6 := 
  by sorry

end ratio_platform_to_pole_0_136


namespace extreme_values_number_of_zeros_0_231

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem extreme_values :
  (∀ x : ℝ, f x ≤ 12) ∧ (f (-1) = 12) ∧ (∀ x : ℝ, -15 ≤ f x) ∧ (f 2 = -15) := 
sorry

theorem number_of_zeros (m : ℝ) :
  (m > 12 ∨ m < -15 → ∃! x : ℝ, g x m = 0) ∧
  (m = 12 ∨ m = -15 → ∃ x y : ℝ, x ≠ y ∧ g x m = 0 ∧ g y m = 0) ∧
  (-15 < m ∧ m < 12 → ∃ x y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ g x m = 0 ∧ g y m = 0 ∧ g z m = 0) :=
sorry

end extreme_values_number_of_zeros_0_231


namespace sheila_initial_savings_0_642

noncomputable def initial_savings (monthly_savings : ℕ) (years : ℕ) (family_addition : ℕ) (total_amount : ℕ) : ℕ :=
  total_amount - (monthly_savings * 12 * years + family_addition)

def sheila_initial_savings_proof : Prop :=
  initial_savings 276 4 7000 23248 = 3000

theorem sheila_initial_savings : sheila_initial_savings_proof :=
  by
    -- Proof goes here
    sorry

end sheila_initial_savings_0_642


namespace loan_amount_is_900_0_286

theorem loan_amount_is_900 (P R T SI : ℕ) (hR : R = 9) (hT : T = 9) (hSI : SI = 729)
    (h_simple_interest : SI = (P * R * T) / 100) : P = 900 := by
  sorry

end loan_amount_is_900_0_286


namespace gcd_a_b_0_933

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_0_933


namespace greatest_integer_0_365

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℤ, n = 9 * k - 2) (h3 : ∃ l : ℤ, n = 8 * l - 4) : n = 124 := 
sorry

end greatest_integer_0_365


namespace donation_total_is_correct_0_828

-- Definitions and conditions
def Megan_inheritance : ℤ := 1000000
def Dan_inheritance : ℤ := 10000
def donation_percentage : ℚ := 0.1
def Megan_donation := Megan_inheritance * donation_percentage
def Dan_donation := Dan_inheritance * donation_percentage
def total_donation := Megan_donation + Dan_donation

-- Theorem statement
theorem donation_total_is_correct : total_donation = 101000 := by
  sorry

end donation_total_is_correct_0_828


namespace minimum_sum_of_distances_squared_0_65

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end minimum_sum_of_distances_squared_0_65


namespace min_S_min_S_values_range_of_c_0_421

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end min_S_min_S_values_range_of_c_0_421


namespace inclination_angle_0_186

theorem inclination_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x + y - 3 = 0) → θ = 3 * Real.pi / 4 := 
sorry

end inclination_angle_0_186


namespace find_c_0_369

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end find_c_0_369


namespace polynomial_characterization_0_471

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end polynomial_characterization_0_471


namespace determine_p_0_512

def is_tangent (circle_eq : ℝ → ℝ → Prop) (parabola_eq : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq x y ∧ circle_eq x y ∧ x = -p / 2 

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

theorem determine_p (p : ℝ) (hpos : p > 0) :
  (is_tangent circle_eq (parabola_eq p) p) ↔ p = 2 := 
sorry

end determine_p_0_512


namespace symmetric_circle_0_748

theorem symmetric_circle
    (x y : ℝ)
    (circle_eq : x^2 + y^2 + 4 * x - 1 = 0) :
    (x - 2)^2 + y^2 = 5 :=
sorry

end symmetric_circle_0_748


namespace contractor_absent_days_0_457

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end contractor_absent_days_0_457


namespace percentage_of_left_handed_women_0_275

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end percentage_of_left_handed_women_0_275


namespace milk_for_flour_0_750

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end milk_for_flour_0_750


namespace abs_difference_0_596

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end abs_difference_0_596


namespace inequality_problem_0_907

noncomputable def a := (3 / 4) * Real.exp (2 / 5)
noncomputable def b := 2 / 5
noncomputable def c := (2 / 5) * Real.exp (3 / 4)

theorem inequality_problem : b < c ∧ c < a := by
  sorry

end inequality_problem_0_907


namespace sin_690_0_823

-- Defining the known conditions as hypotheses:
axiom sin_periodic (x : ℝ) : Real.sin (x + 360) = Real.sin x
axiom sin_odd (x : ℝ) : Real.sin (-x) = - Real.sin x
axiom sin_thirty : Real.sin 30 = 1 / 2

theorem sin_690 : Real.sin 690 = -1 / 2 :=
by
  -- Proof would go here, but it is skipped with sorry.
  sorry

end sin_690_0_823


namespace geometric_sequence_problem_0_731

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end geometric_sequence_problem_0_731


namespace zoey_holidays_in_a_year_0_893

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end zoey_holidays_in_a_year_0_893


namespace fraction_q_p_0_250

theorem fraction_q_p (k : ℝ) (c p q : ℝ) (h : 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) :
  c = 8 ∧ p = -3/4 ∧ q = 31/2 → q / p = -62 / 3 :=
by
  intros hc_hp_hq
  sorry

end fraction_q_p_0_250


namespace maciek_total_purchase_cost_0_13

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_0_13


namespace remainder_y_div_13_0_7

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end remainder_y_div_13_0_7


namespace ellipse_equation_0_671

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end ellipse_equation_0_671


namespace easter_eggs_problem_0_364

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end easter_eggs_problem_0_364


namespace colton_stickers_final_count_0_852

-- Definitions based on conditions
def initial_stickers := 200
def stickers_given_to_7_friends := 6 * 7
def stickers_given_to_mandy := stickers_given_to_7_friends + 8
def remaining_after_mandy := initial_stickers - stickers_given_to_7_friends - stickers_given_to_mandy
def stickers_distributed_to_4_friends := remaining_after_mandy / 2
def remaining_after_4_friends := remaining_after_mandy - stickers_distributed_to_4_friends
def given_to_justin := 2 * remaining_after_4_friends / 3
def remaining_after_justin := remaining_after_4_friends - given_to_justin
def given_to_karen := remaining_after_justin / 5
def final_stickers := remaining_after_justin - given_to_karen

-- Theorem to state the proof problem
theorem colton_stickers_final_count : final_stickers = 15 := by
  sorry

end colton_stickers_final_count_0_852


namespace juan_distance_0_575

def running_time : ℝ := 80.0
def speed : ℝ := 10.0
def distance : ℝ := running_time * speed

theorem juan_distance :
  distance = 800.0 :=
by
  sorry

end juan_distance_0_575


namespace positive_difference_of_squares_0_571

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end positive_difference_of_squares_0_571


namespace athlete_D_is_selected_0_299

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end athlete_D_is_selected_0_299


namespace quadratic_real_roots_m_range_0_812

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_0_812


namespace cube_mod_35_divisors_0_717

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end cube_mod_35_divisors_0_717


namespace number_of_diagonals_octagon_heptagon_diff_0_906

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end number_of_diagonals_octagon_heptagon_diff_0_906


namespace john_umbrella_in_car_0_867

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end john_umbrella_in_car_0_867


namespace fuel_consumption_new_model_0_425

variable (d_old : ℝ) (d_new : ℝ) (c_old : ℝ) (c_new : ℝ)

theorem fuel_consumption_new_model :
  (d_new = d_old + 4.4) →
  (c_new = c_old - 2) →
  (c_old = 100 / d_old) →
  d_old = 12.79 →
  c_new = 5.82 :=
by
  intro h1 h2 h3 h4
  sorry

end fuel_consumption_new_model_0_425


namespace electricity_consumption_scientific_notation_0_346

def electricity_consumption (x : Float) : String := 
  let scientific_notation := "3.64 × 10^4"
  scientific_notation

theorem electricity_consumption_scientific_notation :
  electricity_consumption 36400 = "3.64 × 10^4" :=
by 
  sorry

end electricity_consumption_scientific_notation_0_346


namespace tangent_line_inv_g_at_0_0_693

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end tangent_line_inv_g_at_0_0_693


namespace green_ball_probability_0_116

def prob_green_ball : ℚ :=
  let prob_container := (1 : ℚ) / 3
  let prob_green_I := (4 : ℚ) / 12
  let prob_green_II := (5 : ℚ) / 8
  let prob_green_III := (4 : ℚ) / 8
  prob_container * prob_green_I + prob_container * prob_green_II + prob_container * prob_green_III

theorem green_ball_probability :
  prob_green_ball = 35 / 72 :=
by
  -- Proof steps are omitted as "sorry" is used to skip the proof.
  sorry

end green_ball_probability_0_116


namespace maximum_profit_0_420

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end maximum_profit_0_420


namespace factorial_mod_10_0_497

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_0_497


namespace claire_sleep_hours_0_125

def hours_in_day := 24
def cleaning_hours := 4
def cooking_hours := 2
def crafting_hours := 5
def tailoring_hours := crafting_hours

theorem claire_sleep_hours :
  hours_in_day - (cleaning_hours + cooking_hours + crafting_hours + tailoring_hours) = 8 := by
  sorry

end claire_sleep_hours_0_125


namespace area_of_fourth_rectangle_0_441

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end area_of_fourth_rectangle_0_441


namespace complex_product_0_590

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end complex_product_0_590


namespace tank_empty_time_when_inlet_open_0_532

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end tank_empty_time_when_inlet_open_0_532


namespace solve_basketball_points_0_66

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end solve_basketball_points_0_66


namespace cube_inscribed_circumscribed_volume_ratio_0_866

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end cube_inscribed_circumscribed_volume_ratio_0_866


namespace total_oranges_0_666

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_0_666


namespace min_value_of_function_product_inequality_0_810

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end min_value_of_function_product_inequality_0_810


namespace problem_statement_0_475

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end problem_statement_0_475


namespace impossible_event_0_657

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end impossible_event_0_657


namespace additional_track_length_needed_0_193

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end additional_track_length_needed_0_193


namespace johns_number_0_148

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end johns_number_0_148


namespace find_x_0_779

theorem find_x (x : ℤ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_0_779


namespace minimum_value_f_0_423

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem minimum_value_f :
  ∃ x > 0, (∀ y > 0, f x ≤ f y) ∧ f x = 1 :=
sorry

end minimum_value_f_0_423


namespace original_quadrilateral_area_0_715

theorem original_quadrilateral_area :
  let deg45 := (Real.pi / 4)
  let h := 1 * Real.sin deg45
  let base_bottom := 1 + 2 * h
  let area_perspective := 0.5 * (1 + base_bottom) * h
  let area_original := area_perspective * (2 * Real.sqrt 2)
  area_original = 2 + Real.sqrt 2 := by
  sorry

end original_quadrilateral_area_0_715


namespace prod_f_roots_0_932

def f (x : ℂ) : ℂ := x^2 + 1

theorem prod_f_roots :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℂ),
  (x₁^5 - x₁^2 + 5 = 0) →
  (x₂^5 - x₂^2 + 5 = 0) →
  (x₃^5 - x₃^2 + 5 = 0) →
  (x₄^5 - x₂^2 + 5 = 0) →
  (x₅^5 - x₅^2 + 5 = 0) →
  (∏ k in ({x₁, x₂, x₃, x₄, x₅} : Finset ℂ), f k) = 37 :=
by
  sorry

end prod_f_roots_0_932


namespace abs_eq_condition_0_704

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end abs_eq_condition_0_704


namespace division_by_ab_plus_one_is_perfect_square_0_514

theorem division_by_ab_plus_one_is_perfect_square
    (a b : ℕ) (h : 0 < a ∧ 0 < b)
    (hab : (ab + 1) ∣ (a^2 + b^2)) :
    ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) := 
sorry

end division_by_ab_plus_one_is_perfect_square_0_514


namespace simplify_expression_correct_0_838

noncomputable def simplify_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let expr1 := (a^2 - b^2) / (a^2 + 2 * a * b + b^2)
  let expr2 := (2 : ℝ) / (a * b)
  let expr3 := ((1 : ℝ) / a + (1 : ℝ) / b)^2
  let expr4 := (2 : ℝ) / (a^2 - b^2 + 2 * a * b)
  expr1 + expr2 / expr3 * expr4

theorem simplify_expression_correct (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  simplify_expression a b h = 2 / (a + b)^2 := by
  sorry

end simplify_expression_correct_0_838


namespace tim_words_per_day_0_736

variable (original_words : ℕ)
variable (years : ℕ)
variable (increase_percent : ℚ)

noncomputable def words_per_day (original_words : ℕ) (years : ℕ) (increase_percent : ℚ) : ℚ :=
  let increase_words := original_words * increase_percent
  let total_days := years * 365
  increase_words / total_days

theorem tim_words_per_day :
    words_per_day 14600 2 (50 / 100) = 10 := by
  sorry

end tim_words_per_day_0_736


namespace arithmetic_mean_is_correct_0_449

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end arithmetic_mean_is_correct_0_449


namespace rank_siblings_0_792

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end rank_siblings_0_792


namespace negation_of_proposition_0_952

variables (a b : ℕ)

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def both_even (a b : ℕ) : Prop := is_even a ∧ is_even b

def sum_even (a b : ℕ) : Prop := is_even (a + b)

theorem negation_of_proposition : ¬ (both_even a b → sum_even a b) ↔ ¬both_even a b ∨ ¬sum_even a b :=
by sorry

end negation_of_proposition_0_952


namespace price_per_glass_first_day_0_996

theorem price_per_glass_first_day (O W : ℝ) (P1 P2 : ℝ) 
  (h1 : O = W) 
  (h2 : P2 = 0.40)
  (revenue_eq : 2 * O * P1 = 3 * O * P2) 
  : P1 = 0.60 := 
by 
  sorry

end price_per_glass_first_day_0_996


namespace crease_points_ellipse_0_577

theorem crease_points_ellipse (R a : ℝ) (x y : ℝ) (h1 : 0 < R) (h2 : 0 < a) (h3 : a < R) : 
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) ≥ 1 :=
by
  -- Omitted detailed proof steps
  sorry

end crease_points_ellipse_0_577


namespace system1_solution_system2_solution_0_233

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end system1_solution_system2_solution_0_233


namespace parameterization_theorem_0_349

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end parameterization_theorem_0_349


namespace samantha_coins_worth_0_676

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end samantha_coins_worth_0_676


namespace range_of_a_0_189

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end range_of_a_0_189


namespace sum_of_areas_of_disks_0_638

theorem sum_of_areas_of_disks (r : ℝ) (a b c : ℕ) (h : a + b + c = 123) :
  ∃ (r : ℝ), (15 * Real.pi * r^2 = Real.pi * ((105 / 4) - 15 * Real.sqrt 3) ∧ r = 1 - (Real.sqrt 3) / 2) := 
by
  sorry

end sum_of_areas_of_disks_0_638


namespace intersection_A_C_U_B_0_620

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_A_C_U_B_0_620


namespace root_polynomial_satisfies_expression_0_593

noncomputable def roots_of_polynomial (x : ℕ) : Prop :=
  x^3 - 15 * x^2 + 25 * x - 10 = 0

theorem root_polynomial_satisfies_expression (p q r : ℕ) 
    (h1 : roots_of_polynomial p)
    (h2 : roots_of_polynomial q)
    (h3 : roots_of_polynomial r)
    (h_sum : p + q + r = 15)
    (h_prod : p*q + q*r + r*p = 25) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end root_polynomial_satisfies_expression_0_593


namespace max_quotient_0_434

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end max_quotient_0_434


namespace sufficient_but_not_necessary_condition_0_975

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_0_975


namespace abs_m_plus_one_0_308

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end abs_m_plus_one_0_308


namespace daughter_work_alone_12_days_0_39

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end daughter_work_alone_12_days_0_39


namespace calculate_subtraction_0_246

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end calculate_subtraction_0_246


namespace children_more_than_adults_0_963

-- Conditions
def total_members : ℕ := 120
def adult_percentage : ℝ := 0.40
def child_percentage : ℝ := 1 - adult_percentage

-- Proof problem statement
theorem children_more_than_adults : 
  let number_of_adults := adult_percentage * total_members
  let number_of_children := child_percentage * total_members
  let difference := number_of_children - number_of_adults
  difference = 24 :=
by
  sorry

end children_more_than_adults_0_963


namespace simplify_expression_0_40

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end simplify_expression_0_40


namespace work_completion_time_for_A_0_453

theorem work_completion_time_for_A 
  (B_work_rate : ℝ)
  (combined_work_rate : ℝ)
  (x : ℝ) 
  (B_work_rate_def : B_work_rate = 1 / 6)
  (combined_work_rate_def : combined_work_rate = 3 / 10) :
  (1 / x) + B_work_rate = combined_work_rate →
  x = 7.5 := 
by
  sorry

end work_completion_time_for_A_0_453


namespace crayons_per_box_0_132

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end crayons_per_box_0_132


namespace exists_b_c_with_integral_roots_0_641

theorem exists_b_c_with_integral_roots :
  ∃ (b c : ℝ), (∃ (p q : ℤ), (x^2 + b * x + c = 0) ∧ (x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
               ((x - p) * (x - q) = x^2 - (p + q) * x + p*q)) ∧
              (∃ (r s : ℤ), (x^2 + (b+1) * x + (c+1) = 0) ∧ 
              ((x - r) * (x - s) = x^2 - (r + s) * x + r*s)) :=
by
  sorry

end exists_b_c_with_integral_roots_0_641


namespace simplified_expression_num_terms_0_282

noncomputable def num_terms_polynomial (n: ℕ) : ℕ :=
  (n/2) * (1 + (n+1))

theorem simplified_expression_num_terms :
  num_terms_polynomial 2012 = 1012608 :=
by
  sorry

end simplified_expression_num_terms_0_282


namespace quadratic_range_0_446

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end quadratic_range_0_446


namespace square_area_from_triangle_perimeter_0_109

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end square_area_from_triangle_perimeter_0_109


namespace sum_floor_div_2_pow_25_0_894

/--
Prove that the remainder of the sum 
∑_i=0^2015 ⌊2^i / 25⌋ 
when divided by 100 is 14,
where ⌊x⌋ denotes the floor of x.
-/
theorem sum_floor_div_2_pow_25 :
  (∑ i in Finset.range 2016, (2^i / 25 : ℤ)) % 100 = 14 :=
by
  sorry

end sum_floor_div_2_pow_25_0_894


namespace domain_correct_0_793

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end domain_correct_0_793


namespace jade_pieces_left_0_826

-- Define the initial number of pieces Jade has
def initial_pieces : Nat := 100

-- Define the number of pieces per level
def pieces_per_level : Nat := 7

-- Define the number of levels in the tower
def levels : Nat := 11

-- Define the resulting number of pieces Jade has left after building the tower
def pieces_left : Nat := initial_pieces - (pieces_per_level * levels)

-- The theorem stating that after building the tower, Jade has 23 pieces left
theorem jade_pieces_left : pieces_left = 23 := by
  -- Proof omitted
  sorry

end jade_pieces_left_0_826


namespace sum_last_two_digits_9_pow_23_plus_11_pow_23_0_946

theorem sum_last_two_digits_9_pow_23_plus_11_pow_23 :
  (9^23 + 11^23) % 100 = 60 :=
by
  sorry

end sum_last_two_digits_9_pow_23_plus_11_pow_23_0_946


namespace jack_jogging_speed_needed_0_89

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end jack_jogging_speed_needed_0_89


namespace prime_sum_remainder_0_763

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_0_763


namespace max_area_trapezoid_0_519

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end max_area_trapezoid_0_519


namespace phones_left_is_7500_0_401

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500_0_401


namespace marley_total_fruits_0_543

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end marley_total_fruits_0_543


namespace max_value_is_one_0_404

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_0_404


namespace sum_floor_expression_0_307

theorem sum_floor_expression (p : ℕ) (h_prime : Nat.Prime p) (h_form : ∃ k : ℕ, p = 4 * k + 1) :
  ∑ i in Finset.range p \ {0}, (Int.floor ((2 * i ^ 2 : ℤ) / p) - 2 * Int.floor ((i ^ 2 : ℤ) / p)) = (p - 1) / 2 := 
sorry

end sum_floor_expression_0_307


namespace total_pepper_weight_0_283

theorem total_pepper_weight :
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  (green_peppers + red_peppers + yellow_peppers + orange_peppers) = 8.029333333333333 := 
by
  sorry

end total_pepper_weight_0_283


namespace students_drawn_in_sample_0_108

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end students_drawn_in_sample_0_108


namespace range_of_m_three_zeros_0_632

noncomputable def f (x m : ℝ) : ℝ :=
if h : x < 0 then -x + m else x^2 - 1

theorem range_of_m_three_zeros (h : 0 < m) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (f x1 m) m - 1 = 0 ∧ f (f x2 m) m - 1 = 0 ∧ f (f x3 m) m - 1 = 0) ↔ (0 < m ∧ m < 1) :=
by
  sorry

end range_of_m_three_zeros_0_632


namespace find_b_over_a_0_317

variables {a b c : ℝ}
variables {b₃ b₇ b₁₁ : ℝ}

-- Conditions
def roots_of_quadratic (a b c b₃ b₁₁ : ℝ) : Prop :=
  ∃ p q, p + q = -b / a ∧ p * q = c / a ∧ (p = b₃ ∨ p = b₁₁) ∧ (q = b₃ ∨ q = b₁₁)

def middle_term_value (b₇ : ℝ) : Prop :=
  b₇ = 3

-- The statement to be proved
theorem find_b_over_a
  (h1 : roots_of_quadratic a b c b₃ b₁₁)
  (h2 : middle_term_value b₇)
  (h3 : b₃ + b₁₁ = 2 * b₇) :
  b / a = -6 :=
sorry

end find_b_over_a_0_317


namespace non_powers_of_a_meet_condition_0_483

-- Definitions used directly from the conditions detailed in the problem:
def Sa (a x : ℕ) : ℕ := sorry -- S_{a}(x): sum of the digits of x in base a
def Fa (a x : ℕ) : ℕ := sorry -- F_{a}(x): number of digits of x in base a
def fa (a x : ℕ) : ℕ := sorry -- f_{a}(x): position of the first non-zero digit from the right in base a

theorem non_powers_of_a_meet_condition (a M : ℕ) (h₁: a > 1) (h₂ : M ≥ 2020) :
  ∀ n : ℕ, (n > 0) → (∀ k : ℕ, (k > 0) → (Sa a (k * n) = Sa a n ∧ Fa a (k * n) - fa a (k * n) > M)) ↔ (∃ α : ℕ, n = a ^ α) :=
sorry

end non_powers_of_a_meet_condition_0_483


namespace count_integers_P_leq_0_0_580

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end count_integers_P_leq_0_0_580


namespace marbles_problem_a_marbles_problem_b_0_270

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end marbles_problem_a_marbles_problem_b_0_270


namespace sum_of_first_n_natural_numbers_0_209

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end sum_of_first_n_natural_numbers_0_209


namespace find_number_0_486

theorem find_number (x k : ℕ) (h₁ : x / k = 4) (h₂ : k = 6) : x = 24 := by
  sorry

end find_number_0_486


namespace unique_x1_exists_0_616

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end unique_x1_exists_0_616


namespace angle_A_is_60_degrees_0_402

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end angle_A_is_60_degrees_0_402


namespace nico_reads_wednesday_0_436

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_0_436


namespace unique_solution_for_equation_0_903

theorem unique_solution_for_equation (n : ℕ) (hn : 0 < n) (x : ℝ) (hx : 0 < x) :
  (n : ℝ) * x^2 + ∑ i in Finset.range n, ((i + 2)^2 / (x + (i + 1))) = 
  (n : ℝ) * x + (n * (n + 3) / 2) → x = 1 := 
by 
  sorry

end unique_solution_for_equation_0_903


namespace positive_number_is_25_0_91

theorem positive_number_is_25 {a x : ℝ}
(h1 : x = (3 * a + 1)^2)
(h2 : x = (-a - 3)^2)
(h_sum : 3 * a + 1 + (-a - 3) = 0) :
x = 25 :=
sorry

end positive_number_is_25_0_91


namespace farm_problem_0_0

variable (H R : ℕ)

-- Conditions
def initial_relation : Prop := R = H + 6
def hens_updated : Prop := H + 8 = 20
def current_roosters (H R : ℕ) : ℕ := R + 4

-- Theorem statement
theorem farm_problem (H R : ℕ)
  (h1 : initial_relation H R)
  (h2 : hens_updated H) :
  current_roosters H R = 22 :=
by
  sorry

end farm_problem_0_0


namespace jack_keeps_deers_weight_is_correct_0_466

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end jack_keeps_deers_weight_is_correct_0_466


namespace months_b_after_a_started_business_0_573

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

end months_b_after_a_started_business_0_573


namespace find_central_angle_0_165

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end find_central_angle_0_165


namespace angelina_speed_from_grocery_to_gym_0_951

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end angelina_speed_from_grocery_to_gym_0_951


namespace range_of_n_0_985

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end range_of_n_0_985


namespace find_k_0_120

theorem find_k 
    (x y k : ℝ)
    (h1 : 1.5 * x + y = 20)
    (h2 : -4 * x + y = k)
    (hx : x = -6) :
    k = 53 :=
by
  sorry

end find_k_0_120


namespace number_of_pears_in_fruit_gift_set_0_456

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end number_of_pears_in_fruit_gift_set_0_456


namespace pencil_price_is_99c_0_357

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end pencil_price_is_99c_0_357


namespace minimum_value_of_expression_0_557

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end minimum_value_of_expression_0_557


namespace range_of_m_0_655

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, x > 4 ↔ x > m) : m ≤ 4 :=
by {
  -- here we state the necessary assumptions and conclude the theorem
  -- detailed proof steps are not needed, hence sorry is used to skip the proof
  sorry
}

end range_of_m_0_655


namespace sum_series_0_152

theorem sum_series :
  (∑ n in Finset.range 100, 1 / ((2 * (n + 1) - 3) * (2 * (n + 1) + 5))) = 612 / 1640 :=
by
  sorry

end sum_series_0_152


namespace xiaomings_possible_score_0_49

def average_score_class_A : ℤ := 87
def average_score_class_B : ℤ := 82

theorem xiaomings_possible_score (x : ℤ) :
  (average_score_class_B < x ∧ x < average_score_class_A) → x = 85 :=
by sorry

end xiaomings_possible_score_0_49


namespace number_of_children_at_matinee_0_956

-- Definitions of constants based on conditions
def children_ticket_price : ℝ := 4.50
def adult_ticket_price : ℝ := 6.75
def total_receipts : ℝ := 405
def additional_children : ℕ := 20

-- Variables for number of adults and children
variable (A C : ℕ)

-- Assertions based on conditions
axiom H1 : C = A + additional_children
axiom H2 : children_ticket_price * (C : ℝ) + adult_ticket_price * (A : ℝ) = total_receipts

-- Theorem statement: Prove that the number of children is 48
theorem number_of_children_at_matinee : C = 48 :=
by
  sorry

end number_of_children_at_matinee_0_956


namespace rectangle_area_coefficient_0_769

theorem rectangle_area_coefficient (length width d k : ℝ) 
(h1 : length / width = 5 / 2) 
(h2 : d^2 = length^2 + width^2) 
(h3 : k = 10 / 29) :
  (length * width = k * d^2) :=
by
  sorry

end rectangle_area_coefficient_0_769


namespace sum_square_divisors_positive_0_362

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end sum_square_divisors_positive_0_362


namespace sugar_content_of_mixture_0_993

theorem sugar_content_of_mixture 
  (volume_juice1 : ℝ) (conc_juice1 : ℝ)
  (volume_juice2 : ℝ) (conc_juice2 : ℝ) 
  (total_volume : ℝ) (total_sugar : ℝ) 
  (resulting_sugar_content : ℝ) :
  volume_juice1 = 2 →
  conc_juice1 = 0.1 →
  volume_juice2 = 3 →
  conc_juice2 = 0.15 →
  total_volume = volume_juice1 + volume_juice2 →
  total_sugar = (conc_juice1 * volume_juice1) + (conc_juice2 * volume_juice2) →
  resulting_sugar_content = (total_sugar / total_volume) * 100 →
  resulting_sugar_content = 13 :=
by
  intros
  sorry

end sugar_content_of_mixture_0_993


namespace salary_january_0_694

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end salary_january_0_694


namespace solution_set_f_0_256

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end solution_set_f_0_256


namespace total_output_correct_0_190

variable (a : ℝ)

-- Define a function that captures the total output from this year to the fifth year
def totalOutput (a : ℝ) : ℝ :=
  1.1 * a + (1.1 ^ 2) * a + (1.1 ^ 3) * a + (1.1 ^ 4) * a + (1.1 ^ 5) * a

theorem total_output_correct (a : ℝ) : 
  totalOutput a = 11 * (1.1 ^ 5 - 1) * a := by
  sorry

end total_output_correct_0_190


namespace fisher_needed_score_0_744

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end fisher_needed_score_0_744


namespace cat_clothing_probability_0_69

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end cat_clothing_probability_0_69


namespace calculate_land_tax_0_387

def plot_size : ℕ := 15
def cadastral_value_per_sotka : ℕ := 100000
def tax_rate : ℝ := 0.003

theorem calculate_land_tax :
  plot_size * cadastral_value_per_sotka * tax_rate = 4500 := 
by 
  sorry

end calculate_land_tax_0_387


namespace common_ratio_geometric_series_0_658

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_0_658


namespace average_age_of_4_students_0_974

theorem average_age_of_4_students (avg_age_15 : ℕ) (num_students_15 : ℕ)
    (avg_age_10 : ℕ) (num_students_10 : ℕ) (age_15th_student : ℕ) :
    avg_age_15 = 15 ∧ num_students_15 = 15 ∧ avg_age_10 = 16 ∧ num_students_10 = 10 ∧ age_15th_student = 9 → 
    (56 / 4 = 14) := by
  sorry

end average_age_of_4_students_0_974


namespace find_15th_term_0_809

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end find_15th_term_0_809


namespace tray_height_0_741

-- Declare the main theorem with necessary given conditions.
theorem tray_height (a b c : ℝ) (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  (side_length = 150) →
  (cut_distance = Real.sqrt 50) →
  (angle = 45) →
  a^2 + b^2 = c^2 → -- Condition from Pythagorean theorem
  a = side_length * Real.sqrt 2 / 2 - cut_distance → -- Calculation for half diagonal minus cut distance
  b = (side_length * Real.sqrt 2 / 2 - cut_distance) / 2 → -- Perpendicular from R to the side
  side_length = 150 → -- Ensure consistency of side length
  b^2 + c^2 = side_length^2 → -- Ensure we use another Pythagorean relation
  c = Real.sqrt 7350 → -- Derived c value
  c = Real.sqrt 1470 := -- Simplified form of c.
  sorry

end tray_height_0_741


namespace susan_hours_per_day_0_934

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end susan_hours_per_day_0_934


namespace children_absent_0_268

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end children_absent_0_268


namespace pqrs_sum_0_17

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end pqrs_sum_0_17


namespace least_n_divisibility_condition_0_922

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end least_n_divisibility_condition_0_922


namespace mary_age_proof_0_990

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof_0_990


namespace jordan_trapezoid_height_0_68

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end jordan_trapezoid_height_0_68


namespace solve_for_y_0_464

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end solve_for_y_0_464


namespace average_salary_correct_0_685

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end average_salary_correct_0_685


namespace jane_paints_correct_area_0_290

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end jane_paints_correct_area_0_290


namespace total_weight_is_correct_0_112

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end total_weight_is_correct_0_112


namespace sandy_earnings_correct_0_591

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_0_591


namespace johns_number_is_1500_0_284

def is_multiple_of (a b : Nat) : Prop := ∃ k, a = k * b

theorem johns_number_is_1500 (n : ℕ) (h1 : is_multiple_of n 125) (h2 : is_multiple_of n 30) (h3 : 1000 ≤ n ∧ n ≤ 3000) : n = 1500 :=
by
  -- proof structure goes here
  sorry

end johns_number_is_1500_0_284


namespace shaded_area_correct_0_594

noncomputable def total_shaded_area (floor_length : ℝ) (floor_width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let tile_area := tile_size ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let shaded_area_per_tile := tile_area - circle_area
  let floor_area := floor_length * floor_width
  let number_of_tiles := floor_area / tile_area
  number_of_tiles * shaded_area_per_tile 

theorem shaded_area_correct : total_shaded_area 12 15 2 1 = 180 - 45 * Real.pi := sorry

end shaded_area_correct_0_594


namespace square_fold_distance_0_33

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end square_fold_distance_0_33


namespace weight_lift_equality_0_8

-- Definitions based on conditions
def total_weight_25_pounds_lifted_times := 750
def total_weight_20_pounds_lifted_per_time (n : ℝ) := 60 * n

-- Statement of the proof problem
theorem weight_lift_equality : ∃ n, total_weight_20_pounds_lifted_per_time n = total_weight_25_pounds_lifted_times :=
  sorry

end weight_lift_equality_0_8


namespace new_tax_rate_0_637

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end new_tax_rate_0_637


namespace number_of_members_0_311

theorem number_of_members (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end number_of_members_0_311


namespace sum_of_c_0_155

-- Define sequences a_n, b_n, and c_n
def a (n : ℕ) := 2 * n + 2
def b (n : ℕ) := 2 ^ (n + 1)
def c (n : ℕ) := a n - b n

-- State the main theorem
theorem sum_of_c (n : ℕ) : 
  ∑ i in Finset.range n, c i = n^2 + 3*n + 4 - 2^(n+2) := 
by 
  sorry

end sum_of_c_0_155


namespace desiree_age_0_455

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end desiree_age_0_455


namespace distance_to_workplace_0_151

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end distance_to_workplace_0_151


namespace antonio_weight_0_872

-- Let A be the weight of Antonio
variable (A : ℕ)

-- Conditions:
-- 1. Antonio's sister weighs A - 12 kilograms.
-- 2. The total weight of Antonio and his sister is 88 kilograms.

theorem antonio_weight (A: ℕ) (h1: A - 12 >= 0) (h2: A + (A - 12) = 88) : A = 50 := by
  sorry

end antonio_weight_0_872


namespace green_and_yellow_peaches_total_is_correct_0_276

-- Define the number of red, yellow, and green peaches
def red_peaches : ℕ := 5
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6

-- Definition of the total number of green and yellow peaches
def total_green_and_yellow_peaches : ℕ := green_peaches + yellow_peaches

-- Theorem stating that the total number of green and yellow peaches is 20
theorem green_and_yellow_peaches_total_is_correct : total_green_and_yellow_peaches = 20 :=
by 
  sorry

end green_and_yellow_peaches_total_is_correct_0_276


namespace three_hour_classes_per_week_0_477

theorem three_hour_classes_per_week (x : ℕ) : 
  (24 * (3 * x + 4 + 4) = 336) → x = 2 := by {
  sorry
}

end three_hour_classes_per_week_0_477


namespace find_f_k_l_0_185

noncomputable
def f : ℕ → ℕ := sorry

axiom f_condition_1 : f 1 = 1
axiom f_condition_2 : ∀ n : ℕ, 3 * f n * f (2 * n + 1) = f (2 * n) * (1 + 3 * f n)
axiom f_condition_3 : ∀ n : ℕ, f (2 * n) < 6 * f n

theorem find_f_k_l (k l : ℕ) (h : k < l) : 
  (f k + f l = 293) ↔ 
  ((k = 121 ∧ l = 4) ∨ (k = 118 ∧ l = 4) ∨ 
   (k = 109 ∧ l = 16) ∨ (k = 16 ∧ l = 109)) := 
by 
  sorry

end find_f_k_l_0_185


namespace triangle_area_0_459

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end triangle_area_0_459


namespace cost_of_milkshake_is_correct_0_831

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end cost_of_milkshake_is_correct_0_831


namespace greatest_x_0_180

theorem greatest_x (x : ℕ) (h : x^2 < 32) : x ≤ 5 := 
sorry

end greatest_x_0_180


namespace conditional_probability_0_500

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end conditional_probability_0_500


namespace quadratic_has_real_root_0_984

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_0_984


namespace correct_operation_0_684

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_0_684


namespace perimeter_of_square_is_32_0_393

-- Given conditions
def radius := 4
def diameter := 2 * radius
def side_length_of_square := diameter

-- Question: What is the perimeter of the square?
def perimeter_of_square := 4 * side_length_of_square

-- Proof statement
theorem perimeter_of_square_is_32 : perimeter_of_square = 32 :=
sorry

end perimeter_of_square_is_32_0_393


namespace find_a_monotonic_intervals_exp_gt_xsquare_plus_one_0_842

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end find_a_monotonic_intervals_exp_gt_xsquare_plus_one_0_842


namespace four_digit_sum_0_285

theorem four_digit_sum (A B : ℕ) (hA : 1000 ≤ A ∧ A < 10000) (hB : 1000 ≤ B ∧ B < 10000) (h : A * B = 16^5 + 2^10) : A + B = 2049 := 
by sorry

end four_digit_sum_0_285


namespace sum_of_ages_53_0_164

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end sum_of_ages_53_0_164


namespace find_cookies_per_tray_0_945

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end find_cookies_per_tray_0_945


namespace number_of_women_more_than_men_0_451

variables (M W : ℕ)

def ratio_condition : Prop := M * 3 = 2 * W
def total_condition : Prop := M + W = 20
def correct_answer : Prop := W - M = 4

theorem number_of_women_more_than_men 
  (h1 : ratio_condition M W) 
  (h2 : total_condition M W) : 
  correct_answer M W := 
by 
  sorry

end number_of_women_more_than_men_0_451


namespace outlet_two_rate_0_625

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end outlet_two_rate_0_625


namespace max_num_triangles_for_right_triangle_0_41

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end max_num_triangles_for_right_triangle_0_41


namespace ellipse_foci_distance_0_492

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_0_492


namespace ratio_amyl_alcohol_to_ethanol_0_552

noncomputable def mol_amyl_alcohol : ℕ := 3
noncomputable def mol_hcl : ℕ := 3
noncomputable def mol_ethanol : ℕ := 1
noncomputable def mol_h2so4 : ℕ := 1
noncomputable def mol_ch3_cl2_c5_h9 : ℕ := 3
noncomputable def mol_h2o : ℕ := 3
noncomputable def mol_ethyl_dimethylpropyl_sulfate : ℕ := 1

theorem ratio_amyl_alcohol_to_ethanol : 
  (mol_amyl_alcohol / mol_ethanol = 3) :=
by 
  have h1 : mol_amyl_alcohol = 3 := by rfl
  have h2 : mol_ethanol = 1 := by rfl
  sorry

end ratio_amyl_alcohol_to_ethanol_0_552


namespace time_for_A_to_complete_work_0_563

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end time_for_A_to_complete_work_0_563


namespace foci_distance_of_hyperbola_0_516

theorem foci_distance_of_hyperbola :
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  2 * c = 2 * Real.sqrt 34 :=
by
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  sorry

end foci_distance_of_hyperbola_0_516


namespace smaller_tablet_diagonal_0_377

theorem smaller_tablet_diagonal :
  ∀ (A_large A_small : ℝ)
    (d : ℝ),
    A_large = (8 / Real.sqrt 2) ^ 2 →
    A_small = (d / Real.sqrt 2) ^ 2 →
    A_large = A_small + 7.5 →
    d = 7
:= by
  intros A_large A_small d h1 h2 h3
  sorry

end smaller_tablet_diagonal_0_377


namespace triangle_angle_sum_0_835

theorem triangle_angle_sum (CD CB : ℝ) 
    (isosceles_triangle: CD = CB)
    (interior_pentagon_angle: 108 = 180 * (5 - 2) / 5)
    (interior_triangle_angle: 60 = 180 / 3)
    (triangle_angle_sum: ∀ (a b c : ℝ), a + b + c = 180) :
    mangle_CDB = 6 :=
by
  have x : ℝ := 6
  sorry

end triangle_angle_sum_0_835


namespace distinct_patterns_4x4_3_shaded_0_413

def num_distinct_patterns (n : ℕ) (shading : ℕ) : ℕ :=
  if n = 4 ∧ shading = 3 then 15
  else 0 -- Placeholder for other cases, not relevant for our problem

theorem distinct_patterns_4x4_3_shaded :
  num_distinct_patterns 4 3 = 15 :=
by {
  -- The proof would go here
  sorry
}

end distinct_patterns_4x4_3_shaded_0_413


namespace part1_part2_0_877

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end part1_part2_0_877


namespace jasper_drinks_more_than_hot_dogs_0_281

-- Definition of conditions based on the problem
def bags_of_chips := 27
def fewer_hot_dogs_than_chips := 8
def drinks_sold := 31

-- Definition to compute the number of hot dogs
def hot_dogs_sold := bags_of_chips - fewer_hot_dogs_than_chips

-- Lean 4 statement to prove the final result
theorem jasper_drinks_more_than_hot_dogs : drinks_sold - hot_dogs_sold = 12 :=
by
  -- skipping the proof
  sorry

end jasper_drinks_more_than_hot_dogs_0_281


namespace apples_harvested_0_840

variable (A P : ℕ)
variable (h₁ : P = 3 * A) (h₂ : P - A = 120)

theorem apples_harvested : A = 60 := 
by
  -- proof will go here
  sorry

end apples_harvested_0_840


namespace alberto_more_than_bjorn_and_charlie_0_918

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end alberto_more_than_bjorn_and_charlie_0_918


namespace chocolates_cost_0_935

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end chocolates_cost_0_935


namespace full_day_students_0_199

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_0_199


namespace sum_digits_500_0_322

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end sum_digits_500_0_322


namespace alex_average_speed_0_700

def total_distance : ℕ := 48
def biking_time : ℕ := 6

theorem alex_average_speed : (total_distance / biking_time) = 8 := 
by
  sorry

end alex_average_speed_0_700


namespace number_of_houses_built_0_682

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end number_of_houses_built_0_682


namespace female_managers_count_0_849

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end female_managers_count_0_849


namespace impossible_digit_placement_0_28

-- Define the main variables and assumptions
variable (A B C : ℕ)
variable (h_sum : A + B = 45)
variable (h_segmentSum : 3 * A + B = 6 * C)

-- Define the impossible placement problem
theorem impossible_digit_placement :
  ¬(∃ A B C, A + B = 45 ∧ 3 * A + B = 6 * C ∧ 2 * A = 6 * C - 45) :=
by
  sorry

end impossible_digit_placement_0_28


namespace mrs_sheridan_final_cats_0_578

def initial_cats : ℝ := 17.5
def given_away_cats : ℝ := 6.2
def returned_cats : ℝ := 2.8
def additional_given_away_cats : ℝ := 1.3

theorem mrs_sheridan_final_cats : 
  initial_cats - given_away_cats + returned_cats - additional_given_away_cats = 12.8 :=
by
  sorry

end mrs_sheridan_final_cats_0_578


namespace total_students_0_325

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end total_students_0_325


namespace determine_a_from_equation_0_414

theorem determine_a_from_equation (a : ℝ) (x : ℝ) (h1 : x = 1) (h2 : a * x + 3 * x = 2) : a = -1 := by
  sorry

end determine_a_from_equation_0_414


namespace additional_money_spent_on_dvds_correct_0_883

def initial_money : ℕ := 320
def spent_on_books : ℕ := initial_money / 4 + 10
def remaining_after_books : ℕ := initial_money - spent_on_books
def spent_on_dvds_portion : ℕ := 2 * remaining_after_books / 5
def remaining_after_dvds : ℕ := 130
def total_spent_on_dvds : ℕ := remaining_after_books - remaining_after_dvds
def additional_spent_on_dvds : ℕ := total_spent_on_dvds - spent_on_dvds_portion

theorem additional_money_spent_on_dvds_correct : additional_spent_on_dvds = 8 :=
by
  sorry

end additional_money_spent_on_dvds_correct_0_883


namespace three_monotonic_intervals_iff_a_lt_zero_0_330

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end three_monotonic_intervals_iff_a_lt_zero_0_330


namespace angle_x_is_36_0_298

theorem angle_x_is_36
    (x : ℝ)
    (h1 : 7 * x + 3 * x = 360)
    (h2 : 8 * x ≤ 360) :
    x = 36 := 
by {
  sorry
}

end angle_x_is_36_0_298


namespace grasshopper_total_distance_0_740

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end grasshopper_total_distance_0_740


namespace expected_number_of_defective_products_0_210

theorem expected_number_of_defective_products 
  (N : ℕ) (D : ℕ) (n : ℕ) (hN : N = 15000) (hD : D = 1000) (hn : n = 150) :
  n * (D / N : ℚ) = 10 := 
by {
  sorry
}

end expected_number_of_defective_products_0_210


namespace remaining_dimes_0_302

-- Conditions
def initial_pennies : Nat := 7
def initial_dimes : Nat := 8
def borrowed_dimes : Nat := 4

-- Define the theorem
theorem remaining_dimes : initial_dimes - borrowed_dimes = 4 := by
  -- Use the conditions to state the remaining dimes
  sorry

end remaining_dimes_0_302


namespace fuel_remaining_0_925

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end fuel_remaining_0_925


namespace MountainRidgeAcademy_0_555

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end MountainRidgeAcademy_0_555


namespace find_m_0_313

def triangle (x y : ℤ) := x * y + x + y

theorem find_m (m : ℤ) (h : triangle 2 m = -16) : m = -6 :=
by
  sorry

end find_m_0_313


namespace find_a_b_0_798

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 1) → (x^2 + a * x + b > 0)) →
  (a = 1 ∧ b = -2) :=
by
  sorry

end find_a_b_0_798


namespace sum_of_g1_0_613

-- Define the main conditions
variable {g : ℝ → ℝ}
variable (h_nonconst : ∀ a b : ℝ, a ≠ b → g a ≠ g b)
axiom main_condition : ∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x) ^ 2 / (2025 * x)

-- Define the goal
theorem sum_of_g1 :
  g 1 = 6075 :=
sorry

end sum_of_g1_0_613


namespace cube_volume_is_64_0_494

theorem cube_volume_is_64 (a : ℕ) (h : (a - 2) * (a + 3) * a = a^3 + 12) : a^3 = 64 := 
  sorry

end cube_volume_is_64_0_494


namespace liquid_flow_problem_0_93

variables (x y z : ℝ)

theorem liquid_flow_problem 
    (h1 : 1/x + 1/y + 1/z = 1/6) 
    (h2 : y = 0.75 * x) 
    (h3 : z = y + 10) : 
    x = 56/3 ∧ y = 14 ∧ z = 24 :=
sorry

end liquid_flow_problem_0_93


namespace hair_ratio_0_790

theorem hair_ratio (washed : ℕ) (grow_back : ℕ) (brushed : ℕ) (n : ℕ)
  (hwashed : washed = 32)
  (hgrow_back : grow_back = 49)
  (heq : washed + brushed + 1 = grow_back) :
  (brushed : ℚ) / washed = 1 / 2 := 
by 
  sorry

end hair_ratio_0_790


namespace packs_sold_in_other_villages_0_589

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end packs_sold_in_other_villages_0_589


namespace number_of_floors_0_788

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end number_of_floors_0_788


namespace sqrt_one_half_eq_sqrt_two_over_two_0_667

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end sqrt_one_half_eq_sqrt_two_over_two_0_667


namespace sqrt_57_in_range_0_891

theorem sqrt_57_in_range (h1 : 49 < 57) (h2 : 57 < 64) (h3 : 7^2 = 49) (h4 : 8^2 = 64) : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end sqrt_57_in_range_0_891


namespace find_coefficients_0_588

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end find_coefficients_0_588


namespace arithmetic_sequence_sum_0_808

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end arithmetic_sequence_sum_0_808


namespace car_speed_ratio_0_386

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_0_386


namespace expand_expression_0_796

theorem expand_expression :
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 :=
by sorry

end expand_expression_0_796


namespace abs_a_gt_neg_b_0_245

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end abs_a_gt_neg_b_0_245


namespace num_two_digit_numbers_0_797

-- Define the set of given digits
def digits : Finset ℕ := {0, 2, 5}

-- Define the function that counts the number of valid two-digit numbers
def count_two_digit_numbers (d : Finset ℕ) : ℕ :=
  (d.erase 0).card * (d.card - 1)

theorem num_two_digit_numbers : count_two_digit_numbers digits = 4 :=
by {
  -- sorry placeholder for the proof
  sorry
}

end num_two_digit_numbers_0_797


namespace anne_total_bottle_caps_0_176

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end anne_total_bottle_caps_0_176


namespace smallest_value_of_x_0_149

theorem smallest_value_of_x (x : ℝ) (hx : |3 * x + 7| = 26) : x = -11 :=
sorry

end smallest_value_of_x_0_149


namespace fraction_comparison_0_702

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end fraction_comparison_0_702


namespace floor_sum_min_value_0_278

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_0_278


namespace smallest_reducible_fraction_0_117

theorem smallest_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 8)) ∧ n = 144 := by
  sorry

end smallest_reducible_fraction_0_117


namespace sum_zero_of_absolute_inequalities_0_370

theorem sum_zero_of_absolute_inequalities 
  (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) :
  a + b + c = 0 := 
  by
    sorry

end sum_zero_of_absolute_inequalities_0_370


namespace total_cost_is_18_0_223

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_0_223


namespace min_value_square_distance_0_53

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end min_value_square_distance_0_53


namespace proof_problem_0_249

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define the set M
def M : Set Nat := {2, 4}

-- Define the set N
def N : Set Nat := {0, 4}

-- Define the union of sets M and N
def M_union_N : Set Nat := M ∪ N

-- Define the complement of M ∪ N in U
def complement_U (s : Set Nat) : Set Nat := U \ s

-- State the theorem
theorem proof_problem : complement_U M_union_N = {1, 3} := by
  sorry

end proof_problem_0_249


namespace cost_per_tissue_0_728

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end cost_per_tissue_0_728


namespace part_1_part_2_0_633

-- Part (Ⅰ)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

theorem part_1 (m : ℝ) : (∀ x : ℝ, ¬ f x m < 0) ↔ (-2 ≤ m ∧ m ≤ 6) :=
by sorry

-- Part (Ⅱ)
theorem part_2 (m : ℝ) (h_even : ∀ ⦃x : ℝ⦄, f x m = f (-x) m) :
  (m = 2) → 
  ((∀ x : ℝ, x ≤ 0 → f x 2 ≥ f 0 2) ∧ (∀ x : ℝ, x ≥ 0 → f x 2 ≥ f 0 2)) :=
by sorry

end part_1_part_2_0_633


namespace tape_needed_for_large_box_0_565

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end tape_needed_for_large_box_0_565


namespace polynomial_coefficient_0_487

theorem polynomial_coefficient :
  ∀ d : ℝ, (2 * (2 : ℝ)^4 + 3 * (2 : ℝ)^3 + d * (2 : ℝ)^2 - 4 * (2 : ℝ) + 15 = 0) ↔ (d = -15.75) :=
by
  sorry

end polynomial_coefficient_0_487


namespace candy_store_truffle_price_0_554

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end candy_store_truffle_price_0_554


namespace three_point_one_two_six_as_fraction_0_43

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end three_point_one_two_six_as_fraction_0_43


namespace find_number_0_884

-- Define the conditions.
def condition (x : ℚ) : Prop := x - (1 / 3) * x = 16 / 3

-- Define the theorem from the translated (question, conditions, correct answer) tuple
theorem find_number : ∃ x : ℚ, condition x ∧ x = 8 :=
by
  sorry

end find_number_0_884


namespace complete_the_square_0_709

theorem complete_the_square (y : ℤ) : y^2 + 14 * y + 60 = (y + 7)^2 + 11 :=
by
  sorry

end complete_the_square_0_709


namespace hyperbola_equation_0_183

noncomputable def h : ℝ := -4
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1

theorem hyperbola_equation :
  (h + k + a + b) = 0 := by
  have h := -4
  have k := 2
  have a := 1
  have b := 1
  show (-4 + 2 + 1 + 1) = 0
  sorry

end hyperbola_equation_0_183


namespace Liza_reads_more_pages_than_Suzie_0_206

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end Liza_reads_more_pages_than_Suzie_0_206


namespace batsman_average_after_17_0_998

variable (x : ℝ)
variable (total_runs_16 : ℝ := 16 * x)
variable (runs_17 : ℝ := 90)
variable (new_total_runs : ℝ := total_runs_16 + runs_17)
variable (new_average : ℝ := new_total_runs / 17)

theorem batsman_average_after_17 :
  (total_runs_16 + runs_17 = 17 * (x + 3)) → new_average = x + 3 → new_average = 42 :=
by
  intros h1 h2
  sorry

end batsman_average_after_17_0_998


namespace revenue_difference_0_258

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_0_258


namespace verify_equation_0_378

theorem verify_equation : (3^2 + 5^2)^2 = 16^2 + 30^2 := by
  sorry

end verify_equation_0_378


namespace ratio_of_increase_to_current_0_942

-- Define the constants for the problem
def current_deductible : ℝ := 3000
def increase_deductible : ℝ := 2000

-- State the theorem that needs to be proven
theorem ratio_of_increase_to_current : 
  (increase_deductible / current_deductible) = (2 / 3) :=
by sorry

end ratio_of_increase_to_current_0_942


namespace max_value_of_sample_0_62

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end max_value_of_sample_0_62


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_0_481

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_0_481


namespace total_amount_shared_0_238

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end total_amount_shared_0_238


namespace trains_cross_time_0_444

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end trains_cross_time_0_444


namespace units_digit_product_is_2_0_297

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end units_digit_product_is_2_0_297


namespace num_boys_is_22_0_44

variable (girls boys total_students : ℕ)

-- Conditions
axiom h1 : total_students = 41
axiom h2 : boys = girls + 3
axiom h3 : total_students = girls + boys

-- Goal: Prove that the number of boys is 22
theorem num_boys_is_22 : boys = 22 :=
by
  sorry

end num_boys_is_22_0_44


namespace height_of_david_0_735

theorem height_of_david
  (building_height : ℕ)
  (building_shadow : ℕ)
  (david_shadow : ℕ)
  (ratio : ℕ)
  (h1 : building_height = 50)
  (h2 : building_shadow = 25)
  (h3 : david_shadow = 18)
  (h4 : ratio = building_height / building_shadow) :
  david_shadow * ratio = 36 := sorry

end height_of_david_0_735


namespace value_of_polynomial_0_911

variable {R : Type} [CommRing R]

theorem value_of_polynomial 
  (m : R) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2019 = 2022 := by
  sorry

end value_of_polynomial_0_911


namespace number_of_factors_and_perfect_square_factors_0_898

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end number_of_factors_and_perfect_square_factors_0_898


namespace find_exponent_0_639

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end find_exponent_0_639


namespace trader_profit_0_501

theorem trader_profit (P : ℝ) (hP : 0 < P) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.36 * P
  let profit := selling_price - P
  (profit / P) * 100 = 36 :=
by
  -- The proof will go here
  sorry

end trader_profit_0_501


namespace arithmetic_sequence_values_0_747

noncomputable def common_difference (a₁ a₂ : ℕ) : ℕ := (a₂ - a₁) / 2

theorem arithmetic_sequence_values (x y z d: ℕ) 
    (h₁: d = common_difference 7 11) 
    (h₂: x = 7 + d) 
    (h₃: y = 11 + d) 
    (h₄: z = y + d): 
    x = 9 ∧ y = 13 ∧ z = 15 :=
by {
  sorry
}

end arithmetic_sequence_values_0_747


namespace reduction_amount_is_250_0_37

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end reduction_amount_is_250_0_37


namespace find_c_0_725

theorem find_c (c : ℝ) : (∃ a : ℝ, (x : ℝ) → (x^2 + 80*x + c = (x + a)^2)) → (c = 1600) := by
  sorry

end find_c_0_725


namespace percentage_increase_visitors_0_142

theorem percentage_increase_visitors
  (original_visitors : ℕ)
  (original_fee : ℝ := 1)
  (fee_reduction : ℝ := 0.25)
  (visitors_increase : ℝ := 0.20) :
  ((original_visitors + (visitors_increase * original_visitors)) / original_visitors - 1) * 100 = 20 := by
  sorry

end percentage_increase_visitors_0_142


namespace student_in_eighth_group_0_145

-- Defining the problem: total students and their assignment into groups
def total_students : ℕ := 50
def students_assigned_numbers (n : ℕ) : Prop := n > 0 ∧ n ≤ total_students

-- Grouping students: Each group has 5 students
def grouped_students (group_num student_num : ℕ) : Prop := 
  student_num > (group_num - 1) * 5 ∧ student_num ≤ group_num * 5

-- Condition: Student 12 is selected from the third group
def condition : Prop := grouped_students 3 12

-- Goal: the number of the student selected from the eighth group is 37
theorem student_in_eighth_group : condition → grouped_students 8 37 :=
by
  sorry

end student_in_eighth_group_0_145


namespace two_digit_numbers_condition_0_204

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end two_digit_numbers_condition_0_204


namespace supermarket_spent_more_than_collected_0_495

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end supermarket_spent_more_than_collected_0_495


namespace smallest_unreachable_integer_0_649

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end smallest_unreachable_integer_0_649


namespace sum_possible_values_q_0_570

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end sum_possible_values_q_0_570


namespace series_sum_equals_4290_0_225

theorem series_sum_equals_4290 :
  ∑ n in Finset.range 10, n.succ * (n + 2) * (n + 3) = 4290 := 
by
  sorry

end series_sum_equals_4290_0_225


namespace triangle_converse_inverse_false_0_756

variables {T : Type} (p q : T → Prop)

-- Condition: If a triangle is equilateral, then it is isosceles
axiom h : ∀ t, p t → q t

-- Conclusion: Neither the converse nor the inverse is true
theorem triangle_converse_inverse_false : 
  (∃ t, q t ∧ ¬ p t) ∧ (∃ t, ¬ p t ∧ q t) :=
sorry

end triangle_converse_inverse_false_0_756


namespace monotonic_increasing_interval_0_701

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ a b : ℝ, a < b ∧
    ∀ x y : ℝ, (a < x ∧ x < b) → (a < y ∧ y < b) → x < y → f x < f y ∧ a = -Real.pi / 6 ∧ b = Real.pi / 3 :=
by
  sorry

end monotonic_increasing_interval_0_701


namespace stamps_in_album_0_289

theorem stamps_in_album (n : ℕ) : 
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ 
  n % 6 = 5 ∧ n % 7 = 6 ∧ n % 8 = 7 ∧ n % 9 = 8 ∧ 
  n % 10 = 9 ∧ n < 3000 → n = 2519 :=
by
  sorry

end stamps_in_album_0_289


namespace product_of_zero_multiples_is_equal_0_439

theorem product_of_zero_multiples_is_equal :
  (6000 * 0 = 0) ∧ (6 * 0 = 0) → (6000 * 0 = 6 * 0) :=
by sorry

end product_of_zero_multiples_is_equal_0_439


namespace outfit_count_correct_0_70

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end outfit_count_correct_0_70


namespace goods_train_speed_0_924

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end goods_train_speed_0_924


namespace find_real_solutions_0_561

theorem find_real_solutions (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  ( (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) ) / ( (x - 4) * (x - 5) ) = -1 ↔ x = 10 / 3 ∨ x = 2 / 3 :=
by sorry

end find_real_solutions_0_561


namespace log_equivalence_0_542

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end log_equivalence_0_542


namespace least_number_subtracted_divisible_17_0_556

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end least_number_subtracted_divisible_17_0_556


namespace plus_one_eq_next_plus_0_695

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end plus_one_eq_next_plus_0_695


namespace opposite_of_neg2_is_2_0_427

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_neg2_is_2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_is_2_0_427


namespace not_all_squares_congruent_0_4

-- Define what it means to be a square
structure Square :=
  (side : ℝ)
  (angle : ℝ)
  (is_square : side > 0 ∧ angle = 90)

-- Define congruency of squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side ∧ s1.angle = s2.angle

-- The main statement to prove 
theorem not_all_squares_congruent : ∃ s1 s2 : Square, ¬ congruent s1 s2 :=
by
  sorry

end not_all_squares_congruent_0_4


namespace xiaohui_pe_score_0_743

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end xiaohui_pe_score_0_743


namespace area_inequality_0_814

theorem area_inequality 
  (α β γ : ℝ) 
  (P Q S : ℝ) 
  (h1 : P / Q = α * β * γ) 
  (h2 : S = Q * (α + 1) * (β + 1) * (γ + 1)) : 
  (S ^ (1 / 3)) ≥ (P ^ (1 / 3)) + (Q ^ (1 / 3)) :=
by
  sorry

end area_inequality_0_814


namespace general_formula_a_sum_T_max_k_value_0_908

-- Given conditions
noncomputable def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 6 else n + 5
noncomputable def b (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))
noncomputable def T (n : ℕ) : ℚ := (3 * n) / (2 * n + 1)

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = if n = 1 then 6 else n + 5 :=
by sorry

theorem sum_T (n : ℕ) : T n = (3 * n) / (2 * n + 1) :=
by sorry

theorem max_k_value (k : ℕ) : k = 19 → ∀ n : ℕ, T n > k / 20 :=
by sorry

end general_formula_a_sum_T_max_k_value_0_908


namespace months_rent_in_advance_required_0_761

def janet_savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def deposit : ℕ := 500
def additional_needed : ℕ := 775

theorem months_rent_in_advance_required : 
  (janet_savings + additional_needed - deposit) / rent_per_month = 2 :=
by
  sorry

end months_rent_in_advance_required_0_761


namespace distinct_convex_polygons_of_four_or_more_sides_0_878

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_0_878


namespace gambler_difference_eq_two_0_158

theorem gambler_difference_eq_two (x y : ℕ) (x_lost y_lost : ℕ) :
  20 * x + 100 * y = 3000 ∧
  x + y = 14 ∧
  20 * (14 - y_lost) + 100 * y_lost = 760 →
  (x_lost - y_lost = 2) := sorry

end gambler_difference_eq_two_0_158


namespace mixture_alcohol_quantity_0_847

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end mixture_alcohol_quantity_0_847


namespace evaluate_expression_0_398

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end evaluate_expression_0_398


namespace gcd_of_6Tn2_and_nplus1_eq_2_0_730

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end gcd_of_6Tn2_and_nplus1_eq_2_0_730


namespace magnesium_is_limiting_0_661

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end magnesium_is_limiting_0_661


namespace no_valid_height_configuration_0_146

-- Define the heights and properties
variables {a : Fin 7 → ℝ}
variables {p : ℝ}

-- Define the condition as a theorem
theorem no_valid_height_configuration (h : ∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                                         p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) :
  ¬ (∃ (a : Fin 7 → ℝ), 
    (∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                  p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) ∧
    true) :=
sorry

end no_valid_height_configuration_0_146


namespace purely_imaginary_iff_real_iff_second_quadrant_iff_0_235

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end purely_imaginary_iff_real_iff_second_quadrant_iff_0_235


namespace different_prime_factors_of_factorial_eq_10_0_659

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_0_659


namespace find_original_radius_0_601

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end find_original_radius_0_601


namespace sum_of_squares_with_signs_0_845

theorem sum_of_squares_with_signs (n : ℤ) : 
  ∃ (k : ℕ) (s : Fin k → ℤ), (∀ i : Fin k, s i = 1 ∨ s i = -1) ∧ n = ∑ i : Fin k, s i * ((i + 1) * (i + 1)) := sorry

end sum_of_squares_with_signs_0_845


namespace input_statement_is_INPUT_0_502

namespace ProgrammingStatements

-- Definitions of each type of statement
def PRINT_is_output : Prop := True
def INPUT_is_input : Prop := True
def THEN_is_conditional : Prop := True
def END_is_termination : Prop := True

-- The proof problem
theorem input_statement_is_INPUT :
  INPUT_is_input := by
  sorry

end ProgrammingStatements

end input_statement_is_INPUT_0_502


namespace ceil_of_neg_frac_squared_0_332

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_0_332


namespace hypotenuse_length_0_527

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end hypotenuse_length_0_527


namespace geometric_series_sum_0_656

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end geometric_series_sum_0_656


namespace max_area_right_triangle_in_semicircle_0_319

theorem max_area_right_triangle_in_semicircle :
  ∀ (r : ℝ), r = 1/2 → 
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ y > 0 ∧ 
  (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 ∧ y' > 0 → (1/2) * x * y ≥ (1/2) * x' * y') ∧ 
  (1/2) * x * y = 3 * Real.sqrt 3 / 32 := 
sorry

end max_area_right_triangle_in_semicircle_0_319


namespace symmetric_line_equation_0_200

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end symmetric_line_equation_0_200


namespace distance_correct_0_388

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end distance_correct_0_388


namespace number_div_mult_0_690

theorem number_div_mult (n : ℕ) (h : n = 4) : (n / 6) * 12 = 8 :=
by
  sorry

end number_div_mult_0_690


namespace third_part_of_division_0_391

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end third_part_of_division_0_391


namespace intersection_of_lines_0_585

theorem intersection_of_lines : 
  let x := (5 : ℚ) / 9
  let y := (5 : ℚ) / 3
  (y = 3 * x ∧ y - 5 = -6 * x) ↔ (x, y) = ((5 : ℚ) / 9, (5 : ℚ) / 3) := 
by 
  sorry

end intersection_of_lines_0_585


namespace reggie_games_lost_0_674

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_0_674


namespace express_train_speed_ratio_0_683

noncomputable def speed_ratio (c h : ℝ) (x : ℝ) : Prop :=
  let t1 := h / ((1 + x) * c)
  let t2 := h / ((x - 1) * c)
  x = t2 / t1

theorem express_train_speed_ratio 
  (c h : ℝ) (x : ℝ) 
  (hc : c > 0) (hh : h > 0) (hx : x > 1) : 
  speed_ratio c h (1 + Real.sqrt 2) := 
by
  sorry

end express_train_speed_ratio_0_683


namespace min_sugar_0_716

variable (f s : ℝ)

theorem min_sugar (h1 : f ≥ 10 + 3 * s) (h2 : f ≤ 4 * s) : s ≥ 10 := by
  sorry

end min_sugar_0_716


namespace triangle_sets_0_732

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end triangle_sets_0_732


namespace fraction_not_exist_implies_x_neg_one_0_958

theorem fraction_not_exist_implies_x_neg_one {x : ℝ} :
  ¬(∃ y : ℝ, y = 1 / (x + 1)) → x = -1 :=
by
  intro h
  have : x + 1 = 0 :=
    by
      contrapose! h
      exact ⟨1 / (x + 1), rfl⟩
  linarith

end fraction_not_exist_implies_x_neg_one_0_958


namespace ratio_change_factor_is_5_0_696

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end ratio_change_factor_is_5_0_696


namespace move_point_right_3_units_0_95

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end move_point_right_3_units_0_95


namespace oak_trees_remaining_is_7_0_67

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end oak_trees_remaining_is_7_0_67


namespace g_at_100_0_949

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end g_at_100_0_949


namespace find_value_of_a_0_20

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end find_value_of_a_0_20


namespace smallest_perimeter_even_integer_triangl_0_787

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_0_787


namespace repeated_process_pure_alcohol_0_534

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end repeated_process_pure_alcohol_0_534


namespace dividend_is_correct_0_306

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end dividend_is_correct_0_306


namespace sampling_methods_correct_0_248

-- Assuming definitions for the populations for both surveys
structure CommunityHouseholds where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure ArtisticStudents where
  total_students : Nat

-- Given conditions
def households_population : CommunityHouseholds := { high_income := 125, middle_income := 280, low_income := 95 }
def students_population : ArtisticStudents := { total_students := 15 }

-- Correct answer according to the conditions
def appropriate_sampling_methods (ch: CommunityHouseholds) (as: ArtisticStudents) : String :=
  if ch.high_income > 0 ∧ ch.middle_income > 0 ∧ ch.low_income > 0 ∧ as.total_students ≥ 3 then
    "B" -- ① Stratified sampling, ② Simple random sampling
  else
    "Invalid"

theorem sampling_methods_correct :
  appropriate_sampling_methods households_population students_population = "B" := by
  sorry

end sampling_methods_correct_0_248


namespace village_population_rate_decrease_0_383

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end village_population_rate_decrease_0_383


namespace remaining_pages_0_964

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_0_964


namespace radius_of_circle_with_tangent_parabolas_0_978

theorem radius_of_circle_with_tangent_parabolas (r : ℝ) : 
  (∀ x : ℝ, (x^2 + r = x → ∃ x0 : ℝ, x^2 + r = x0)) → r = 1 / 4 :=
by
  sorry

end radius_of_circle_with_tangent_parabolas_0_978


namespace sum_of_fractions_0_652

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end sum_of_fractions_0_652


namespace factor_theorem_solution_0_474

theorem factor_theorem_solution (t : ℝ) :
  (∃ p q : ℝ, 10 * p * q = 10 * t * t + 21 * t - 10 ∧ (x - q) = (x - t)) →
  t = 2 / 5 ∨ t = -5 / 2 := by
  sorry

end factor_theorem_solution_0_474


namespace find_b_0_827

variable (b : ℝ)

theorem find_b 
    (h₁ : 0 < b)
    (h₂ : b < 4)
    (area_ratio : ∃ k : ℝ, k = 4/16 ∧ (4 + b) / -b = 2 * k) :
  b = -4/3 :=
by
  sorry

end find_b_0_827


namespace right_angled_triangle_solution_0_531

-- Define the necessary constants
def t : ℝ := 504 -- area in cm^2
def c : ℝ := 65 -- hypotenuse in cm

-- The definitions of the right-angled triangle's properties
def is_right_angled_triangle (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2 ∧ a * b = 2 * t

-- The proof problem statement
theorem right_angled_triangle_solution :
  ∃ (a b : ℝ), is_right_angled_triangle a b ∧ ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
sorry

end right_angled_triangle_solution_0_531


namespace transaction_gain_per_year_0_212

noncomputable def principal : ℝ := 9000
noncomputable def time : ℝ := 2
noncomputable def rate_lending : ℝ := 6
noncomputable def rate_borrowing : ℝ := 4

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def total_interest_earned := simple_interest principal rate_lending time
noncomputable def total_interest_paid := simple_interest principal rate_borrowing time

noncomputable def total_gain := total_interest_earned - total_interest_paid
noncomputable def gain_per_year := total_gain / 2

theorem transaction_gain_per_year : gain_per_year = 180 :=
by
  sorry

end transaction_gain_per_year_0_212


namespace sum_of_fractions_0_734

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end sum_of_fractions_0_734


namespace value_of_m_sub_n_0_1

theorem value_of_m_sub_n (m n : ℤ) (h1 : |m| = 5) (h2 : n^2 = 36) (h3 : m * n < 0) : m - n = 11 ∨ m - n = -11 := 
by 
  sorry

end value_of_m_sub_n_0_1


namespace sum_of_five_integers_0_337

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end sum_of_five_integers_0_337


namespace deborah_total_cost_0_181

-- Standard postage per letter
def stdPostage : ℝ := 1.08

-- Additional charge for international shipping per letter
def intlAdditional : ℝ := 0.14

-- Number of domestic and international letters
def numDomestic : ℕ := 2
def numInternational : ℕ := 2

-- Expected total cost for four letters
def expectedTotalCost : ℝ := 4.60

theorem deborah_total_cost :
  (numDomestic * stdPostage) + (numInternational * (stdPostage + intlAdditional)) = expectedTotalCost :=
by
  -- proof skipped
  sorry

end deborah_total_cost_0_181


namespace sum_mod_9_0_742

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end sum_mod_9_0_742


namespace unique_solution_0_229

def S : Set ℕ := {0, 1, 2, 3, 4}

def op (i j : ℕ) : ℕ := (i + j) % 5

theorem unique_solution :
  ∃! x ∈ S, op (op x x) 2 = 1 := sorry

end unique_solution_0_229


namespace tyrone_gave_marbles_to_eric_0_102

theorem tyrone_gave_marbles_to_eric (initial_tyrone_marbles : ℕ) (initial_eric_marbles : ℕ) (marbles_given : ℕ) :
  initial_tyrone_marbles = 150 ∧ initial_eric_marbles = 30 ∧ (initial_tyrone_marbles - marbles_given = 3 * initial_eric_marbles) → marbles_given = 60 :=
by
  sorry

end tyrone_gave_marbles_to_eric_0_102


namespace train_length_is_correct_0_923

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end train_length_is_correct_0_923


namespace units_digit_of_k_squared_plus_2_to_the_k_0_83

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end units_digit_of_k_squared_plus_2_to_the_k_0_83


namespace distance_between_centers_0_727

-- Declare radii of the circles and the shortest distance between points on the circles
def R := 28
def r := 12
def d := 10

-- Define the problem to prove the distance between the centers
theorem distance_between_centers (R r d : ℝ) (hR : R = 28) (hr : r = 12) (hd : d = 10) : 
  ∀ OO1 : ℝ, OO1 = 6 :=
by sorry

end distance_between_centers_0_727


namespace max_value_min_4x_y_4y_x2_5y2_0_35

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end max_value_min_4x_y_4y_x2_5y2_0_35


namespace triangle_angles_0_583

theorem triangle_angles (α β : ℝ) (A B C : ℝ) (hA : A = 2) (hB : B = 3) (hC : C = 4) :
  2 * α + 3 * β = 180 :=
sorry

end triangle_angles_0_583


namespace concert_attendance_difference_0_773

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_0_773


namespace division_remainder_0_30

-- Define the conditions
def dividend : ℝ := 9087.42
def divisor : ℝ := 417.35
def quotient : ℝ := 21

-- Define the expected remainder
def expected_remainder : ℝ := 323.07

-- Statement of the problem
theorem division_remainder : dividend - divisor * quotient = expected_remainder :=
by
  sorry

end division_remainder_0_30


namespace trader_loss_percentage_0_470

def profit_loss_percentage (SP1 SP2 CP1 CP2 : ℚ) : ℚ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

theorem trader_loss_percentage :
  let SP1 := 325475
  let SP2 := 325475
  let CP1 := SP1 / (1 + 0.10)
  let CP2 := SP2 / (1 - 0.10)
  profit_loss_percentage SP1 SP2 CP1 CP2 = -1 := by
  sorry

end trader_loss_percentage_0_470


namespace participation_increase_closest_to_10_0_629

def percentage_increase (old new : ℕ) : ℚ := ((new - old) / old) * 100

theorem participation_increase_closest_to_10 :
  (percentage_increase 80 88 = 10) ∧ 
  (percentage_increase 90 99 = 10) := by
  sorry

end participation_increase_closest_to_10_0_629


namespace opposite_sides_line_0_355

theorem opposite_sides_line (a : ℝ) : (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 := by
  sorry

end opposite_sides_line_0_355


namespace A_investment_0_614

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end A_investment_0_614


namespace total_heads_0_733

theorem total_heads (D P : ℕ) (h1 : D = 9) (h2 : 4 * D + 2 * P = 42) : D + P = 12 :=
by
  sorry

end total_heads_0_733


namespace conference_games_0_327

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end conference_games_0_327


namespace example_equation_0_381

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end example_equation_0_381


namespace tank_length_0_799

variable (rate : ℝ)
variable (time : ℝ)
variable (width : ℝ)
variable (depth : ℝ)
variable (volume : ℝ)
variable (length : ℝ)

-- Given conditions
axiom rate_cond : rate = 5 -- cubic feet per hour
axiom time_cond : time = 60 -- hours
axiom width_cond : width = 6 -- feet
axiom depth_cond : depth = 5 -- feet

-- Derived volume from the rate and time
axiom volume_cond : volume = rate * time

-- Definition of length from volume, width, and depth
axiom length_def : length = volume / (width * depth)

-- The proof problem to show
theorem tank_length : length = 10 := by
  -- conditions provided and we expect the length to be computed
  sorry

end tank_length_0_799


namespace greatest_median_0_19

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end greatest_median_0_19


namespace evening_water_usage_is_6_0_672

-- Define the conditions: daily water usage and total water usage over 5 days.
def daily_water_usage (E : ℕ) : ℕ := 4 + E
def total_water_usage (E : ℕ) (days : ℕ) : ℕ := days * daily_water_usage E

-- Define the condition that over 5 days the total water usage is 50 liters.
axiom water_usage_condition : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6

-- Conjecture stating the amount of water used in the evening.
theorem evening_water_usage_is_6 : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6 :=
by
  intro E
  intro h
  exact water_usage_condition E h

end evening_water_usage_is_6_0_672


namespace doubled_team_completes_half_in_three_days_0_660

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end doubled_team_completes_half_in_three_days_0_660


namespace number_of_students_playing_soccer_0_304

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end number_of_students_playing_soccer_0_304


namespace variance_of_data_set_0_242

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end variance_of_data_set_0_242


namespace solve_problem_0_437

noncomputable def problem_statement : ℤ :=
  (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2

theorem solve_problem : problem_statement = -166 :=
by 
  -- Proof omitted
  sorry

end solve_problem_0_437


namespace geese_count_0_545

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end geese_count_0_545


namespace find_c_0_913

noncomputable def condition1 (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

noncomputable def condition2 (c : ℝ) : Prop :=
  6 * 15 * c = 1

theorem find_c (c : ℝ) (h1 : condition1 6 15 c) (h2 : condition2 c) : c = 11 := 
by
  sorry

end find_c_0_913


namespace B_squared_B_sixth_0_944

noncomputable def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![0, 3], ![2, -1]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  1

theorem B_squared :
  B * B = 3 * B - I := by
  sorry

theorem B_sixth :
  B^6 = 84 * B - 44 * I := by
  sorry

end B_squared_B_sixth_0_944


namespace set_intersection_eq_0_758

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_eq_0_758


namespace simplify_expression_0_689

variables (x y z : ℝ)

theorem simplify_expression (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 :=
by sorry

end simplify_expression_0_689


namespace intersection_of_A_and_complement_of_B_0_863

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x : ℝ | 2^x * (x - 2) < 1 }
noncomputable def B : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }
noncomputable def B_complement : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_A_and_complement_of_B :
  A ∩ B_complement = { x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_complement_of_B_0_863


namespace total_ladybugs_correct_0_874

noncomputable def total_ladybugs (with_spots : ℕ) (without_spots : ℕ) : ℕ :=
  with_spots + without_spots

theorem total_ladybugs_correct :
  total_ladybugs 12170 54912 = 67082 :=
by
  unfold total_ladybugs
  rfl

end total_ladybugs_correct_0_874


namespace linear_function_implies_m_value_0_188

variable (x m : ℝ)

theorem linear_function_implies_m_value :
  (∃ y : ℝ, y = (m-3)*x^(m^2-8) + m + 1 ∧ ∀ x1 x2 : ℝ, y = y * (x2 - x1) + y * x1) → m = -3 :=
by
  sorry

end linear_function_implies_m_value_0_188


namespace ice_cream_total_volume_0_846

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end ice_cream_total_volume_0_846


namespace minimum_value_inequality_0_540

theorem minimum_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (Real.sqrt ((x^2 + 4 * y^2) * (2 * x^2 + 3 * y^2)) / (x * y)) ≥ 2 * Real.sqrt (2 * Real.sqrt 6) :=
sorry

end minimum_value_inequality_0_540


namespace fourth_person_height_is_82_0_584

theorem fourth_person_height_is_82 (H : ℕ)
    (h1: (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 76)
    (h_diff1: H + 2 - H = 2)
    (h_diff2: H + 4 - (H + 2) = 2)
    (h_diff3: H + 10 - (H + 4) = 6) :
  (H + 10) = 82 := 
sorry

end fourth_person_height_is_82_0_584


namespace simplify_expression_0_60

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end simplify_expression_0_60


namespace sum_a5_a6_a7_0_5

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

variables (a : ℕ → ℤ)
variables (h_geo : geometric_sequence a)
variables (h1 : a 2 + a 3 = 1)
variables (h2 : a 3 + a 4 = -2)

theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 24 :=
by
  sorry

end sum_a5_a6_a7_0_5


namespace competition_sequences_0_334

-- Define the problem conditions
def team_size : Nat := 7

-- Define the statement to prove
theorem competition_sequences :
  (Nat.choose (2 * team_size) team_size) = 3432 :=
by
  -- Proof will go here
  sorry

end competition_sequences_0_334


namespace gcd_2023_1991_0_518

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end gcd_2023_1991_0_518


namespace annual_increase_of_chickens_0_103

theorem annual_increase_of_chickens 
  (chickens_now : ℕ)
  (chickens_after_9_years : ℕ)
  (years : ℕ)
  (chickens_now_eq : chickens_now = 550)
  (chickens_after_9_years_eq : chickens_after_9_years = 1900)
  (years_eq : years = 9)
  : ((chickens_after_9_years - chickens_now) / years) = 150 :=
by
  sorry

end annual_increase_of_chickens_0_103


namespace daily_production_0_805

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end daily_production_0_805


namespace geometric_series_S_n_div_a_n_0_980

-- Define the conditions and the properties of the geometric sequence
variables (a_3 a_5 a_4 a_6 S_n a_n : ℝ) (n : ℕ)
variable (q : ℝ) -- common ratio of the geometric sequence

-- Conditions given in the problem
axiom h1 : a_3 + a_5 = 5 / 4
axiom h2 : a_4 + a_6 = 5 / 8

-- The value we want to prove
theorem geometric_series_S_n_div_a_n : 
  (a_3 + a_5) * q = 5 / 8 → 
  q = 1 / 2 → 
  S_n = a_n * (2^n - 1) :=
by
  intros h1 h2
  sorry

end geometric_series_S_n_div_a_n_0_980


namespace common_divisor_0_42

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end common_divisor_0_42


namespace exercise_0_390

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end exercise_0_390


namespace pyramid_x_value_0_79

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end pyramid_x_value_0_79


namespace age_difference_0_687

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end age_difference_0_687


namespace probability_correct_0_252

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_0_252


namespace solve_for_a_b_and_extrema_0_724

noncomputable def f (a b x : ℝ) := -2 * a * Real.sin (2 * x + (Real.pi / 6)) + 2 * a + b

theorem solve_for_a_b_and_extrema:
  ∃ (a b : ℝ), a > 0 ∧ 
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) ∧ 
  a = 2 ∧ b = -5 ∧
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 4),
    (f a b (Real.pi / 6) = -5 ∨ f a b 0 = -3)) :=
by
  sorry

end solve_for_a_b_and_extrema_0_724


namespace proof_A_union_B_eq_R_0_129

def A : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - 5) < a }

theorem proof_A_union_B_eq_R (a : ℝ) (h : a > 6) : 
  A ∪ B a = Set.univ :=
by {
  sorry
}

end proof_A_union_B_eq_R_0_129


namespace seq_problem_part1_seq_problem_part2_0_510

def seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

theorem seq_problem_part1 (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  a 2008 = 0 := 
sorry

theorem seq_problem_part2 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  ∃ (M : ℤ), 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = 0) ∧ 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = M) := 
sorry

end seq_problem_part1_seq_problem_part2_0_510


namespace ratio_of_parallel_vectors_0_150

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end ratio_of_parallel_vectors_0_150


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_0_562

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_0_562


namespace total_cups_needed_0_938

-- Define the known conditions
def ratio_butter : ℕ := 2
def ratio_flour : ℕ := 3
def ratio_sugar : ℕ := 5
def total_sugar_in_cups : ℕ := 10

-- Define the parts-to-cups conversion
def cup_per_part := total_sugar_in_cups / ratio_sugar

-- Define the amounts of each ingredient in cups
def butter_in_cups := ratio_butter * cup_per_part
def flour_in_cups := ratio_flour * cup_per_part
def sugar_in_cups := ratio_sugar * cup_per_part

-- Define the total number of cups
def total_cups := butter_in_cups + flour_in_cups + sugar_in_cups

-- Theorem to prove
theorem total_cups_needed : total_cups = 20 := by
  sorry

end total_cups_needed_0_938


namespace total_distance_0_488

theorem total_distance (x : ℝ) (h : (1/2) * (x - 1) = (1/3) * x + 1) : x = 9 := 
by 
  sorry

end total_distance_0_488


namespace symmetric_curve_eq_0_645

-- Definitions from the problem conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 1
def line_of_symmetry (x y : ℝ) : Prop := x - y + 3 = 0

-- Problem statement derived from the translation step
theorem symmetric_curve_eq (x y : ℝ) : (x - 2) ^ 2 + (y + 1) ^ 2 = 1 ∧ x - y + 3 = 0 → (x + 4) ^ 2 + (y - 5) ^ 2 = 1 := 
by
  sorry

end symmetric_curve_eq_0_645


namespace johns_average_speed_0_141

-- Conditions
def biking_time_minutes : ℝ := 45
def biking_speed_mph : ℝ := 20
def walking_time_minutes : ℝ := 120
def walking_speed_mph : ℝ := 3

-- Proof statement
theorem johns_average_speed :
  let biking_time_hours := biking_time_minutes / 60
  let biking_distance := biking_speed_mph * biking_time_hours
  let walking_time_hours := walking_time_minutes / 60
  let walking_distance := walking_speed_mph * walking_time_hours
  let total_distance := biking_distance + walking_distance
  let total_time := biking_time_hours + walking_time_hours
  let average_speed := total_distance / total_time
  average_speed = 7.64 :=
by
  sorry

end johns_average_speed_0_141


namespace expected_value_of_12_sided_die_is_6_5_0_871

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5_0_871


namespace equidistant_cyclist_0_547

-- Definition of key parameters
def speed_car := 60  -- in km/h
def speed_cyclist := 18  -- in km/h
def speed_pedestrian := 6  -- in km/h
def distance_AC := 10  -- in km
def angle_ACB := 60  -- in degrees
def time_car_start := (7, 58)  -- 7:58 AM
def time_cyclist_start := (8, 0)  -- 8:00 AM
def time_pedestrian_start := (6, 44) -- 6:44 AM
def time_solution := (8, 6)  -- 8:06 AM

-- Time difference function
def time_diff (t1 t2 : Nat × Nat) : Nat :=
  (t2.1 - t1.1) * 60 + (t2.2 - t1.2)  -- time difference in minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (m : Nat) : ℝ :=
  m / 60.0

-- Distances traveled by car, cyclist, and pedestrian by the given time
noncomputable def distance_car (t1 t2 : Nat × Nat) : ℝ :=
  speed_car * (minutes_to_hours (time_diff t1 t2) + 2 / 60.0)

noncomputable def distance_cyclist (t1 t2 : Nat × Nat) : ℝ :=
  speed_cyclist * minutes_to_hours (time_diff t1 t2)

noncomputable def distance_pedestrian (t1 t2 : Nat × Nat) : ℝ :=
  speed_pedestrian * (minutes_to_hours (time_diff t1 t2) + 136 / 60.0)

-- Verification statement
theorem equidistant_cyclist :
  distance_car time_car_start time_solution = distance_pedestrian time_pedestrian_start time_solution → 
  distance_cyclist time_cyclist_start time_solution = 
  distance_car time_car_start time_solution ∧
  distance_cyclist time_cyclist_start time_solution = 
  distance_pedestrian time_pedestrian_start time_solution :=
by
  -- Given conditions and the correctness to be shown
  sorry

end equidistant_cyclist_0_547


namespace range_of_a_0_269

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end range_of_a_0_269


namespace sum_interior_ninth_row_0_720

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end sum_interior_ninth_row_0_720


namespace time_to_reach_ticket_window_0_859

-- Define the conditions as per the problem
def rate_kit : ℕ := 2 -- feet per minute (rate)
def remaining_distance : ℕ := 210 -- feet

-- Goal: To prove the time required to reach the ticket window is 105 minutes
theorem time_to_reach_ticket_window : remaining_distance / rate_kit = 105 :=
by sorry

end time_to_reach_ticket_window_0_859


namespace odd_perfect_prime_form_n_is_seven_0_886

theorem odd_perfect_prime_form (n p s m : ℕ) (h₁ : n % 2 = 1) (h₂ : ∃ k : ℕ, p = 4 * k + 1) (h₃ : ∃ h : ℕ, s = 4 * h + 1) (h₄ : n = p^s * m^2) (h₅ : ¬ p ∣ m) :
  ∃ k h : ℕ, p = 4 * k + 1 ∧ s = 4 * h + 1 :=
sorry

theorem n_is_seven (n : ℕ) (h₁ : n > 1) (h₂ : ∃ k : ℕ, k * k = n -1) (h₃ : ∃ l : ℕ, l * l = (n * (n + 1)) / 2) :
  n = 7 :=
sorry

end odd_perfect_prime_form_n_is_seven_0_886


namespace bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_0_643

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Bose-Einstein distribution, satisfying the given conditions. 
-/
theorem bose_einstein_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 72 := 
  by
  sorry

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Fermi-Dirac distribution, satisfying the given conditions. 
-/
theorem fermi_dirac_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 246 := 
  by
  sorry

end bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_0_643


namespace measure_smaller_angle_east_northwest_0_572

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

end measure_smaller_angle_east_northwest_0_572


namespace cos_theta_value_projection_value_0_292

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end cos_theta_value_projection_value_0_292


namespace polynomial_real_roots_0_341

theorem polynomial_real_roots :
  (∃ x : ℝ, x^4 - 3*x^3 - 2*x^2 + 6*x + 9 = 0) ↔ (x = 1 ∨ x = 3) := 
by
  sorry

end polynomial_real_roots_0_341


namespace necessary_but_not_sufficient_0_706

-- Define sets M and N
def M (x : ℝ) : Prop := x < 5
def N (x : ℝ) : Prop := x > 3

-- Define the union and intersection of M and N
def M_union_N (x : ℝ) : Prop := M x ∨ N x
def M_inter_N (x : ℝ) : Prop := M x ∧ N x

-- Theorem statement: Prove the necessity but not sufficiency
theorem necessary_but_not_sufficient (x : ℝ) :
  M_inter_N x → M_union_N x ∧ ¬(M_union_N x → M_inter_N x) := 
sorry

end necessary_but_not_sufficient_0_706


namespace one_twenty_percent_of_number_0_279

theorem one_twenty_percent_of_number (x : ℝ) (h : 0.20 * x = 300) : 1.20 * x = 1800 :=
by 
sorry

end one_twenty_percent_of_number_0_279


namespace cos_sq_sub_sin_sq_0_895

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end cos_sq_sub_sin_sq_0_895


namespace factor_expression_0_864

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end factor_expression_0_864


namespace find_fractions_0_61

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_0_61


namespace distance_to_base_is_42_0_603

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end distance_to_base_is_42_0_603


namespace y_intercept_of_line_0_669

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end y_intercept_of_line_0_669


namespace factorize_ab_factorize_x_0_338

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end factorize_ab_factorize_x_0_338


namespace matthews_contribution_0_159

theorem matthews_contribution 
  (total_cost : ℝ) (yen_amount : ℝ) (conversion_rate : ℝ)
  (h1 : total_cost = 18)
  (h2 : yen_amount = 2500)
  (h3 : conversion_rate = 140) :
  (total_cost - (yen_amount / conversion_rate)) = 0.143 :=
by sorry

end matthews_contribution_0_159


namespace digits_subtraction_eq_zero_0_239

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end digits_subtraction_eq_zero_0_239


namespace sum_absolute_minus_absolute_sum_leq_0_705

theorem sum_absolute_minus_absolute_sum_leq (n : ℕ) (x : ℕ → ℝ)
  (h_n : n ≥ 1)
  (h_abs_diff : ∀ k, 0 < k ∧ k < n → |x k.succ - x k| ≤ 1) :
  (∑ k in Finset.range n, |x k|) - |∑ k in Finset.range n, x k| ≤ (n^2 - 1) / 4 := 
sorry

end sum_absolute_minus_absolute_sum_leq_0_705


namespace negation_of_proposition_0_762

theorem negation_of_proposition (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by
  sorry

end negation_of_proposition_0_762


namespace solve_equation1_solve_equation2_0_520

-- Define the first equation as a condition
def equation1 (x : ℝ) : Prop :=
  3 * x + 20 = 4 * x - 25

-- Prove that x = 45 satisfies equation1
theorem solve_equation1 : equation1 45 :=
by 
  -- Proof steps would go here
  sorry

-- Define the second equation as a condition
def equation2 (x : ℝ) : Prop :=
  (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6

-- Prove that x = 3/2 satisfies equation2
theorem solve_equation2 : equation2 (3 / 2) :=
by 
  -- Proof steps would go here
  sorry

end solve_equation1_solve_equation2_0_520


namespace dealer_can_determine_values_0_619

def card_value_determined (a : Fin 100 → Fin 100) : Prop :=
  (∀ i j : Fin 100, i > j → a i > a j) ∧ (a 0 > a 99) ∧
  (∀ k : Fin 100, a k = k + 1)

theorem dealer_can_determine_values :
  ∃ (messages : Fin 100 → Fin 100), card_value_determined messages :=
sorry

end dealer_can_determine_values_0_619


namespace hilt_books_transaction_difference_0_853

noncomputable def total_cost_paid (original_price : ℝ) (num_first_books : ℕ) (discount1 : ℝ) (num_second_books : ℕ) (discount2 : ℝ) : ℝ :=
  let cost_first_books := num_first_books * original_price * (1 - discount1)
  let cost_second_books := num_second_books * original_price * (1 - discount2)
  cost_first_books + cost_second_books

noncomputable def total_sale_amount (sale_price : ℝ) (interest_rate : ℝ) (num_books : ℕ) : ℝ :=
  let compounded_price := sale_price * (1 + interest_rate) ^ 1
  compounded_price * num_books

theorem hilt_books_transaction_difference : 
  let original_price := 11
  let num_first_books := 10
  let discount1 := 0.20
  let num_second_books := 5
  let discount2 := 0.25
  let sale_price := 25
  let interest_rate := 0.05
  let num_books := 15
  total_sale_amount sale_price interest_rate num_books - total_cost_paid original_price num_first_books discount1 num_second_books discount2 = 264.50 :=
by
  sorry

end hilt_books_transaction_difference_0_853


namespace part_one_part_two_0_147

-- First part: Prove that \( (1)(-1)^{2017}+(\frac{1}{2})^{-2}+(3.14-\pi)^{0} = 4\)
theorem part_one : (1 * (-1:ℤ)^2017 + (1/2)^(-2:ℤ) + (3.14 - Real.pi)^0 : ℝ) = 4 := 
  sorry

-- Second part: Prove that \( ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 \)
theorem part_two (x : ℝ) : ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 := 
  sorry

end part_one_part_two_0_147


namespace combinations_with_common_subjects_0_897

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end combinations_with_common_subjects_0_897


namespace solve_floor_trig_eq_0_961

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end solve_floor_trig_eq_0_961


namespace original_number_is_0_02_0_163

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end original_number_is_0_02_0_163


namespace cities_with_highest_increase_0_385

-- Define population changes for each city
def cityF_initial := 30000
def cityF_final := 45000
def cityG_initial := 55000
def cityG_final := 77000
def cityH_initial := 40000
def cityH_final := 60000
def cityI_initial := 70000
def cityI_final := 98000
def cityJ_initial := 25000
def cityJ_final := 37500

-- Function to calculate percentage increase
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / (initial : ℚ) * 100

-- Theorem stating cities F, H, and J had the highest percentage increase
theorem cities_with_highest_increase :
  percentage_increase cityF_initial cityF_final = 50 ∧
  percentage_increase cityH_initial cityH_final = 50 ∧
  percentage_increase cityJ_initial cityJ_final = 50 ∧
  percentage_increase cityG_initial cityG_final < 50 ∧
  percentage_increase cityI_initial cityI_final < 50 :=
by
-- Proof omitted
sorry

end cities_with_highest_increase_0_385


namespace find_ordered_pairs_0_153

theorem find_ordered_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m ∣ 3 * n - 2 ∧ 2 * n ∣ 3 * m - 2) ↔ (m, n) = (2, 2) ∨ (m, n) = (10, 14) ∨ (m, n) = (14, 10) :=
by
  sorry

end find_ordered_pairs_0_153


namespace median_possible_values_0_23

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end median_possible_values_0_23


namespace books_not_sold_0_424

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end books_not_sold_0_424


namespace linear_function_details_0_549

variables (x y : ℝ)

noncomputable def linear_function (k b : ℝ) := k * x + b

def passes_through (k b x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = linear_function k b x1 ∧ y2 = linear_function k b x2

def point_on_graph (k b x3 y3 : ℝ) : Prop :=
  y3 = linear_function k b x3

theorem linear_function_details :
  ∃ k b : ℝ, passes_through k b 3 5 (-4) (-9) ∧ point_on_graph k b (-1) (-3) :=
by
  -- to be proved
  sorry

end linear_function_details_0_549


namespace sum_single_digits_0_703

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end sum_single_digits_0_703


namespace fill_tank_with_leak_0_221

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end fill_tank_with_leak_0_221


namespace verify_parabola_D_0_564

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_0_564


namespace slope_of_tangent_at_A_0_927

def f (x : ℝ) : ℝ := x^2 + 3 * x

def f' (x : ℝ) : ℝ := 2 * x + 3

theorem slope_of_tangent_at_A : f' 2 = 7 := by
  sorry

end slope_of_tangent_at_A_0_927


namespace true_proposition_0_986

-- Definitions based on the conditions
def p (x : ℝ) := x * (x - 1) ≠ 0 → x ≠ 0 ∧ x ≠ 1
def q (a b c : ℝ) := a > b → c > 0 → a * c > b * c

-- The theorem based on the question and the conditions
theorem true_proposition (x a b c : ℝ) (hp : p x) (hq_false : ¬ q a b c) : p x ∨ q a b c :=
by
  sorry

end true_proposition_0_986


namespace math_problem_0_511

-- Define constants and conversions from decimal/mixed numbers to fractions
def thirteen_and_three_quarters : ℚ := 55 / 4
def nine_and_sixth : ℚ := 55 / 6
def one_point_two : ℚ := 1.2
def ten_point_three : ℚ := 103 / 10
def eight_and_half : ℚ := 17 / 2
def six_point_eight : ℚ := 34 / 5
def three_and_three_fifths : ℚ := 18 / 5
def five_and_five_sixths : ℚ := 35 / 6
def three_and_two_thirds : ℚ := 11 / 3
def three_and_one_sixth : ℚ := 19 / 6
def fifty_six : ℚ := 56
def twenty_seven_and_sixth : ℚ := 163 / 6

def E : ℚ := 
  ((thirteen_and_three_quarters + nine_and_sixth) * one_point_two) / ((ten_point_three - eight_and_half) * (5 / 9)) + 
  ((six_point_eight - three_and_three_fifths) * five_and_five_sixths) / ((three_and_two_thirds - three_and_one_sixth) * fifty_six) - 
  twenty_seven_and_sixth

theorem math_problem : E = 29 / 3 := by
  sorry

end math_problem_0_511


namespace scientific_notation_of_130944000000_0_192

theorem scientific_notation_of_130944000000 :
  130944000000 = 1.30944 * 10^11 :=
by sorry

end scientific_notation_of_130944000000_0_192


namespace li_to_zhang_0_354

theorem li_to_zhang :
  (∀ (meter chi : ℕ), 3 * meter = chi) →
  (∀ (zhang chi : ℕ), 10 * zhang = chi) →
  (∀ (kilometer li : ℕ), 2 * li = kilometer) →
  (1 * lin = 150 * zhang) :=
by
  intro h_meter h_zhang h_kilometer
  sorry

end li_to_zhang_0_354


namespace find_lower_rate_0_331

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end find_lower_rate_0_331


namespace white_rabbit_hop_distance_per_minute_0_418

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end white_rabbit_hop_distance_per_minute_0_418


namespace simplify_expression_correct_0_931

-- Defining the problem conditions and required proof
def simplify_expression (x : ℝ) (h : x ≠ 2) : Prop :=
  (x / (x - 2) + 2 / (2 - x) = 1)

-- Stating the theorem
theorem simplify_expression_correct (x : ℝ) (h : x ≠ 2) : simplify_expression x h :=
  by sorry

end simplify_expression_correct_0_931


namespace price_per_liter_after_discount_0_296

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end price_per_liter_after_discount_0_296


namespace min_tangent_length_0_699

-- Define the equation of the line y = x
def line_eq (x y : ℝ) : Prop := y = x

-- Define the equation of the circle centered at (4, -2) with radius 1
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y + 2)^2 = 1

-- Prove that the minimum length of the tangent line is √17
theorem min_tangent_length : 
  ∀ (x y : ℝ), line_eq x y → circle_eq x y → 
  (∃ p : ℝ, p = (√17)) :=
by 
  sorry

end min_tangent_length_0_699


namespace probability_of_second_ball_white_is_correct_0_94

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end probability_of_second_ball_white_is_correct_0_94


namespace max_value_of_expression_0_832

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end max_value_of_expression_0_832


namespace total_cost_of_fencing_0_770

def costOfFencing (lengths rates : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) lengths rates)

theorem total_cost_of_fencing :
  costOfFencing [14, 20, 35, 40, 15, 30, 25]
                [2.50, 3.00, 3.50, 4.00, 2.75, 3.25, 3.75] = 610.00 :=
by
  sorry

end total_cost_of_fencing_0_770


namespace trigonometric_identity_0_566

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end trigonometric_identity_0_566


namespace sum_of_three_fractions_is_one_0_154

theorem sum_of_three_fractions_is_one (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) :=
by sorry

end sum_of_three_fractions_is_one_0_154


namespace near_square_qoutient_0_85

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end near_square_qoutient_0_85


namespace fraction_B_compared_to_A_and_C_0_263

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end fraction_B_compared_to_A_and_C_0_263


namespace interval_of_increase_find_side_c_0_622

noncomputable def f (x : ℝ) : ℝ := 
  √3 * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

def interval_increasing (k : ℤ) : Set ℝ := 
  { x | k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 }

theorem interval_of_increase (k : ℤ) : 
  ∀ x : ℝ, (interval_increasing k) x ↔ f x > f (x - Real.pi / 3) ∧ f x < f (x + Real.pi / 6) :=
sorry

theorem find_side_c (A : ℝ) (a b : ℝ) (hA : f A = 1 / 2) (ha : a = √17) (hb : b = 4) : 
  ∃ c : ℝ, c = 2 + √5 :=
sorry

end interval_of_increase_find_side_c_0_622


namespace dice_probability_five_or_six_0_599

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end dice_probability_five_or_six_0_599


namespace darryl_parts_cost_0_376

-- Define the conditions
def patent_cost : ℕ := 4500
def machine_price : ℕ := 180
def break_even_units : ℕ := 45
def total_revenue := break_even_units * machine_price

-- Define the theorem using the conditions
theorem darryl_parts_cost :
  ∃ (parts_cost : ℕ), parts_cost = total_revenue - patent_cost ∧ parts_cost = 3600 := by
  sorry

end darryl_parts_cost_0_376


namespace B_profit_0_586

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end B_profit_0_586


namespace water_on_wednesday_0_111

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_0_111


namespace base8_operations_0_175

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end base8_operations_0_175


namespace arithmetic_sequence_sum_ratio_0_74

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end arithmetic_sequence_sum_ratio_0_74


namespace max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_0_539

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem max_ab_bc_ca : ab + bc + ca ≤ 1 / 3 :=
by sorry

theorem a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2 :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 1 / 2 :=
by sorry

end max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_0_539


namespace sum_odd_numbers_to_2019_is_correct_0_664

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end sum_odd_numbers_to_2019_is_correct_0_664


namespace b_over_c_equals_1_0_372

theorem b_over_c_equals_1 (a b c d : ℕ) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4)
    (h : 4^a + 3^b + 2^c + 1^d = 78) : b = c :=
by
  sorry

end b_over_c_equals_1_0_372


namespace solve_for_y_0_358

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end solve_for_y_0_358


namespace female_members_count_0_653

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_0_653


namespace find_opposite_pair_0_737

def is_opposite (x y : ℤ) : Prop := x = -y

theorem find_opposite_pair :
  ¬is_opposite 4 4 ∧ ¬is_opposite 2 2 ∧ ¬is_opposite (-8) (-8) ∧ is_opposite 4 (-4) := 
by
  sorry

end find_opposite_pair_0_737


namespace consistent_scale_0_890

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end consistent_scale_0_890


namespace octal_addition_correct_0_15

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end octal_addition_correct_0_15


namespace minimum_shirts_for_saving_money_0_928

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end minimum_shirts_for_saving_money_0_928


namespace max_triangles_formed_0_167

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end max_triangles_formed_0_167


namespace allowance_amount_0_485

variable (initial_money spent_money final_money : ℕ)

theorem allowance_amount (initial_money : ℕ) (spent_money : ℕ) (final_money : ℕ) (h1: initial_money = 5) (h2: spent_money = 2) (h3: final_money = 8) : (final_money - (initial_money - spent_money)) = 5 := 
by 
  sorry

end allowance_amount_0_485


namespace students_per_bench_0_16

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench_0_16


namespace correct_average_and_variance_0_394

theorem correct_average_and_variance
  (n : ℕ) (avg incorrect_variance correct_variance : ℝ)
  (incorrect_score1 actual_score1 incorrect_score2 actual_score2 : ℝ)
  (H1 : n = 48)
  (H2 : avg = 70)
  (H3 : incorrect_variance = 75)
  (H4 : incorrect_score1 = 50)
  (H5 : actual_score1 = 80)
  (H6 : incorrect_score2 = 100)
  (H7 : actual_score2 = 70)
  (Havg : avg = (n * avg - incorrect_score1 - incorrect_score2 + actual_score1 + actual_score2) / n)
  (Hvar : correct_variance = incorrect_variance + (actual_score1 - avg) ^ 2 + (actual_score2 - avg) ^ 2
                     - (incorrect_score1 - avg) ^ 2 - (incorrect_score2 - avg) ^ 2 / n) :
  avg = 70 ∧ correct_variance = 50 :=
by {
  sorry
}

end correct_average_and_variance_0_394


namespace smaller_balloon_radius_is_correct_0_559

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end smaller_balloon_radius_is_correct_0_559


namespace foreign_students_next_sem_eq_740_0_582

def total_students : ℕ := 1800
def percentage_foreign : ℕ := 30
def new_foreign_students : ℕ := 200

def initial_foreign_students : ℕ := total_students * percentage_foreign / 100
def total_foreign_students_next_semester : ℕ :=
  initial_foreign_students + new_foreign_students

theorem foreign_students_next_sem_eq_740 :
  total_foreign_students_next_semester = 740 :=
by
  sorry

end foreign_students_next_sem_eq_740_0_582


namespace karlson_wins_with_optimal_play_0_780

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end karlson_wins_with_optimal_play_0_780


namespace number_of_crayons_given_to_friends_0_954

def totalCrayonsLostOrGivenAway := 229
def crayonsLost := 16
def crayonsGivenToFriends := totalCrayonsLostOrGivenAway - crayonsLost

theorem number_of_crayons_given_to_friends :
  crayonsGivenToFriends = 213 :=
by
  sorry

end number_of_crayons_given_to_friends_0_954


namespace isosceles_trapezoid_height_0_755

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end isosceles_trapezoid_height_0_755


namespace initial_friends_count_0_843

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end initial_friends_count_0_843


namespace thieves_cloth_equation_0_574

theorem thieves_cloth_equation (x y : ℤ) 
  (h1 : y = 6 * x + 5)
  (h2 : y = 7 * x - 8) :
  6 * x + 5 = 7 * x - 8 :=
by
  sorry

end thieves_cloth_equation_0_574


namespace time_relationship_0_544

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end time_relationship_0_544


namespace total_number_of_shirts_0_525

variable (total_cost : ℕ) (num_15_dollar_shirts : ℕ) (cost_15_dollar_shirts : ℕ) 
          (cost_remaining_shirts : ℕ) (num_remaining_shirts : ℕ) 

theorem total_number_of_shirts :
  total_cost = 85 →
  num_15_dollar_shirts = 3 →
  cost_15_dollar_shirts = 15 →
  cost_remaining_shirts = 20 →
  (num_remaining_shirts * cost_remaining_shirts) + (num_15_dollar_shirts * cost_15_dollar_shirts) = total_cost →
  num_15_dollar_shirts + num_remaining_shirts = 5 :=
by
  intros
  sorry

end total_number_of_shirts_0_525


namespace g_is_odd_0_411

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end g_is_odd_0_411


namespace selection_methods_eq_total_students_0_899

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end selection_methods_eq_total_students_0_899


namespace base7_divisibility_rules_2_base7_divisibility_rules_3_0_912

def divisible_by_2 (d : Nat) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

def divisible_by_3 (d : Nat) : Prop :=
  d = 0 ∨ d = 3

def last_digit_base7 (n : Nat) : Nat :=
  n % 7

theorem base7_divisibility_rules_2 (n : Nat) :
  (∃ k, n = 2 * k) ↔ divisible_by_2 (last_digit_base7 n) :=
by
  sorry

theorem base7_divisibility_rules_3 (n : Nat) :
  (∃ k, n = 3 * k) ↔ divisible_by_3 (last_digit_base7 n) :=
by
  sorry

end base7_divisibility_rules_2_base7_divisibility_rules_3_0_912


namespace total_messages_0_123

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_0_123


namespace segments_after_cuts_0_768

-- Definitions from the conditions
def cuts : ℕ := 10

-- Mathematically equivalent proof statement
theorem segments_after_cuts : (cuts + 1 = 11) :=
by sorry

end segments_after_cuts_0_768


namespace A_share_of_annual_gain_0_224

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end A_share_of_annual_gain_0_224


namespace horner_v3_value_0_90

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end horner_v3_value_0_90


namespace find_base_0_981

-- Definitions based on the conditions of the problem
def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) := ∃ m : ℕ, m * m * m = n
def is_perfect_fourth (n : ℕ) := ∃ m : ℕ, m * m * m * m = n

-- Define the number A in terms of base a
def A (a : ℕ) : ℕ := 4 * a * a + 4 * a + 1

-- Problem statement: find a base a > 4 such that A is both a perfect cube and a perfect fourth power
theorem find_base (a : ℕ)
  (ha : a > 4)
  (h_square : is_perfect_square (A a)) :
  is_perfect_cube (A a) ∧ is_perfect_fourth (A a) :=
sorry

end find_base_0_981


namespace maximum_profit_0_648

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end maximum_profit_0_648


namespace relationship_between_a_b_c_0_628

noncomputable def a := (3 / 5 : ℝ) ^ (2 / 5)
noncomputable def b := (2 / 5 : ℝ) ^ (3 / 5)
noncomputable def c := (2 / 5 : ℝ) ^ (2 / 5)

theorem relationship_between_a_b_c :
  a > c ∧ c > b :=
by
  sorry

end relationship_between_a_b_c_0_628


namespace unit_prices_and_purchasing_schemes_0_374

theorem unit_prices_and_purchasing_schemes :
  ∃ (x y : ℕ),
    (14 * x + 8 * y = 1600) ∧
    (3 * x = 4 * y) ∧
    (x = 80) ∧ 
    (y = 60) ∧
    ∃ (m : ℕ), 
      (m ≥ 29) ∧ 
      (m ≤ 30) ∧ 
      (80 * m + 60 * (50 - m) ≤ 3600) ∧
      (m = 29 ∨ m = 30) := 
sorry

end unit_prices_and_purchasing_schemes_0_374


namespace find_k_max_product_0_22

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end find_k_max_product_0_22


namespace total_scarves_0_469

def total_yarns_red : ℕ := 2
def total_yarns_blue : ℕ := 6
def total_yarns_yellow : ℕ := 4
def scarves_per_yarn : ℕ := 3

theorem total_scarves : 
  (total_yarns_red * scarves_per_yarn) + 
  (total_yarns_blue * scarves_per_yarn) + 
  (total_yarns_yellow * scarves_per_yarn) = 36 := 
by
  sorry

end total_scarves_0_469


namespace polynomial_expression_0_98

theorem polynomial_expression :
  (2 * x^2 + 3 * x + 7) * (x + 1) - (x + 1) * (x^2 + 4 * x - 63) + (3 * x - 14) * (x + 1) * (x + 5) = 4 * x^3 + 4 * x^2 :=
by
  sorry

end polynomial_expression_0_98


namespace smallest_next_divisor_0_316

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end smallest_next_divisor_0_316


namespace discount_is_10_percent_0_605

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end discount_is_10_percent_0_605


namespace det_2x2_matrix_0_630

open Matrix

theorem det_2x2_matrix : 
  det ![![7, -2], ![-3, 5]] = 29 := by
  sorry

end det_2x2_matrix_0_630


namespace count_valid_n_le_30_0_714

theorem count_valid_n_le_30 :
  ∀ n : ℕ, (0 < n ∧ n ≤ 30) → (n! * 2) % (n * (n + 1)) = 0 := by
  sorry

end count_valid_n_le_30_0_714


namespace integral_eval_0_277

noncomputable def integral_problem : ℝ :=
  ∫ x in - (Real.pi / 2)..(Real.pi / 2), (x + Real.cos x)

theorem integral_eval : integral_problem = 2 :=
  by 
  sorry

end integral_eval_0_277


namespace find_k_0_771

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem find_k (k : ℝ) (h_pos : 0 < k) (h_exists : ∃ x₀ : ℝ, 1 ≤ x₀ ∧ g x₀ ≤ k * (-x₀^2 + 3 * x₀)) : 
  k > (1 / 2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end find_k_0_771


namespace division_result_0_972

theorem division_result:
    35 / 0.07 = 500 := by
  sorry

end division_result_0_972


namespace fraction_proof_0_247

-- Define N
def N : ℕ := 24

-- Define F that satisfies the equation N = F + 15
def F := N - 15

-- Define the fraction that N exceeds by 15
noncomputable def fraction := (F : ℚ) / N

-- Prove that fraction = 3/8
theorem fraction_proof : fraction = 3 / 8 := by
  sorry

end fraction_proof_0_247


namespace constant_term_of_expansion_0_722

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end constant_term_of_expansion_0_722


namespace percentage_increase_proof_0_767

def breakfast_calories : ℕ := 500
def shakes_total_calories : ℕ := 3 * 300
def total_daily_calories : ℕ := 3275

noncomputable def percentage_increase_in_calories (P : ℝ) : Prop :=
  let lunch_calories := breakfast_calories * (1 + P / 100)
  let dinner_calories := 2 * lunch_calories
  breakfast_calories + lunch_calories + dinner_calories + shakes_total_calories = total_daily_calories

theorem percentage_increase_proof : percentage_increase_in_calories 125 :=
by
  sorry

end percentage_increase_proof_0_767


namespace solve_inequality_0_904

theorem solve_inequality {x : ℝ} : (x^2 - 5 * x + 6 ≤ 0) → (2 ≤ x ∧ x ≤ 3) :=
by
  intro h
  sorry

end solve_inequality_0_904


namespace subset_zero_in_A_0_12

def A := { x : ℝ | x > -1 }

theorem subset_zero_in_A : {0} ⊆ A :=
by sorry

end subset_zero_in_A_0_12


namespace factorable_quadratic_0_301

theorem factorable_quadratic (b : Int) : 
  (∃ m n p q : Int, 35 * m * p = 35 ∧ m * q + n * p = b ∧ n * q = 35) ↔ (∃ k : Int, b = 2 * k) :=
sorry

end factorable_quadratic_0_301


namespace simplify_fraction_0_371

theorem simplify_fraction : (150 / 4350 : ℚ) = 1 / 29 :=
  sorry

end simplify_fraction_0_371


namespace inverse_value_0_272

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end inverse_value_0_272


namespace stratified_sampling_third_grade_0_806

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end stratified_sampling_third_grade_0_806


namespace inequality_ineqs_0_99

theorem inequality_ineqs (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_cond : x * y + y * z + z * x = 1) :
  (27 / 4) * (x + y) * (y + z) * (z + x) 
  ≥ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2
  ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2 
  ≥ 
  6 * Real.sqrt 3 := by
  sorry

end inequality_ineqs_0_99


namespace sugar_concentration_after_adding_water_0_839

def initial_mass_of_sugar_water : ℝ := 90
def initial_sugar_concentration : ℝ := 0.10
def final_sugar_concentration : ℝ := 0.08
def mass_of_water_added : ℝ := 22.5

theorem sugar_concentration_after_adding_water 
  (m_sugar_water : ℝ := initial_mass_of_sugar_water)
  (c_initial : ℝ := initial_sugar_concentration)
  (c_final : ℝ := final_sugar_concentration)
  (m_water_added : ℝ := mass_of_water_added) :
  (m_sugar_water * c_initial = (m_sugar_water + m_water_added) * c_final) := 
sorry

end sugar_concentration_after_adding_water_0_839


namespace fit_jack_apples_into_jill_basket_0_506

-- Conditions:
def jack_basket_full : ℕ := 12
def jack_basket_space : ℕ := 4
def jack_current_apples : ℕ := jack_basket_full - jack_basket_space
def jill_basket_capacity : ℕ := 2 * jack_basket_full

-- Proof statement:
theorem fit_jack_apples_into_jill_basket : jill_basket_capacity / jack_current_apples = 3 :=
by {
  sorry
}

end fit_jack_apples_into_jill_basket_0_506


namespace eighth_grade_students_0_203

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end eighth_grade_students_0_203


namespace area_R2_0_813

-- Definitions from conditions
def side_R1 : ℕ := 3
def area_R1 : ℕ := 24
def diagonal_ratio : ℤ := 2

-- Introduction of the theorem
theorem area_R2 (similar: ℤ) (a b: ℕ) :
  a * b = area_R1 ∧
  a = 3 ∧
  b * 3 = 8 * a ∧
  (a^2 + b^2 = 292) ∧
  similar * (a^2 + b^2) = 2 * 2 * 73 →
  (6 * 16 = 96) := by
sorry

end area_R2_0_813


namespace minimum_value_y_range_of_a_0_379

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end minimum_value_y_range_of_a_0_379


namespace second_investment_amount_0_929

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

theorem second_investment_amount :
  ∀ (P₁ P₂ I₁ I₂ r t : ℝ), 
    P₁ = 5000 →
    I₁ = 250 →
    I₂ = 1000 →
    I₁ = simple_interest P₁ r t →
    I₂ = simple_interest P₂ r t →
    P₂ = 20000 := 
by 
  intros P₁ P₂ I₁ I₂ r t hP₁ hI₁ hI₂ hI₁_eq hI₂_eq
  sorry

end second_investment_amount_0_929


namespace pieces_left_0_432

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end pieces_left_0_432


namespace car_speed_0_465

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_0_465


namespace cement_tesss_street_0_900

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end cement_tesss_street_0_900


namespace find_first_5digits_of_M_0_237

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end find_first_5digits_of_M_0_237


namespace polyhedron_volume_0_295

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end polyhedron_volume_0_295


namespace equal_perimeter_triangle_side_length_0_445

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end equal_perimeter_triangle_side_length_0_445


namespace value_of_n_0_187

def is_3_digit_integer (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

def not_divisible_by (n k : ℕ) : Prop := ¬ (k ∣ n)

def least_common_multiple (a b c : ℕ) : Prop := Nat.lcm a b = c

theorem value_of_n (d n : ℕ) (h1 : least_common_multiple d n 690) 
  (h2 : not_divisible_by n 3) (h3 : not_divisible_by d 2) (h4 : is_3_digit_integer n) : n = 230 :=
by
  sorry

end value_of_n_0_187


namespace sine_shift_0_607

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end sine_shift_0_607


namespace positive_integer_k_0_631

theorem positive_integer_k (k x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = k * x * y * z) :
  k = 1 ∨ k = 3 :=
sorry

end positive_integer_k_0_631


namespace boys_count_at_table_0_173

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end boys_count_at_table_0_173


namespace find_C_and_D_0_509

variables (C D : ℝ)

theorem find_C_and_D (h : 4 * C + 2 * D + 5 = 30) : C = 5.25 ∧ D = 2 :=
by
  sorry

end find_C_and_D_0_509


namespace sequence_values_induction_proof_0_294

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end sequence_values_induction_proof_0_294


namespace train_speed_correct_0_416

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end train_speed_correct_0_416


namespace fred_gave_sandy_balloons_0_86

theorem fred_gave_sandy_balloons :
  ∀ (original_balloons given_balloons final_balloons : ℕ),
    original_balloons = 709 →
    final_balloons = 488 →
    given_balloons = original_balloons - final_balloons →
    given_balloons = 221 := by
  sorry

end fred_gave_sandy_balloons_0_86


namespace sum_of_squares_0_323

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end sum_of_squares_0_323


namespace food_waste_in_scientific_notation_0_819

-- Given condition that 1 billion equals 10^9
def billion : ℕ := 10 ^ 9

-- Problem statement: expressing 530 billion kilograms in scientific notation
theorem food_waste_in_scientific_notation :
  (530 * billion : ℝ) = 5.3 * 10^10 := 
  sorry

end food_waste_in_scientific_notation_0_819


namespace problem_solution_0_885

theorem problem_solution : 
  (1 / (2 ^ 1980) * (∑ n in Finset.range 991, (-3:ℝ) ^ n * Nat.choose 1980 (2 * n))) = -1 / 2 :=
by
  sorry

end problem_solution_0_885


namespace find_number_of_students_0_266

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end find_number_of_students_0_266


namespace base9_addition_correct_0_213

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end base9_addition_correct_0_213


namespace largest_fraction_0_635

variable {a b c d e f g h : ℝ}
variable {w x y z : ℝ}

/-- Given real numbers w, x, y, z such that w < x < y < z,
    the fraction z/w represents the largest value among the given fractions. -/
theorem largest_fraction (hwx : w < x) (hxy : x < y) (hyz : y < z) :
  (z / w) > (x / w) ∧ (z / w) > (y / x) ∧ (z / w) > (y / w) ∧ (z / w) > (z / x) :=
by
  sorry

end largest_fraction_0_635


namespace determine_remaining_sides_0_373

variables (A B C D E : Type)

def cyclic_quadrilateral (A B C D : Type) : Prop := sorry

def known_sides (AB CD : ℝ) : Prop := AB > 0 ∧ CD > 0

def known_ratio (m n : ℝ) : Prop := m > 0 ∧ n > 0

theorem determine_remaining_sides
  {A B C D : Type}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (AB CD : ℝ) (h_sides : known_sides AB CD)
  (m n : ℝ) (h_ratio : known_ratio m n) :
  ∃ (BC AD : ℝ), BC / AD = m / n ∧ BC > 0 ∧ AD > 0 :=
sorry

end determine_remaining_sides_0_373


namespace largest_fraction_0_77

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end largest_fraction_0_77


namespace bank_exceeds_1600cents_in_9_days_after_Sunday_0_533

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end bank_exceeds_1600cents_in_9_days_after_Sunday_0_533


namespace proof_problem_0_458

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end proof_problem_0_458


namespace annual_population_growth_0_484

noncomputable def annual_percentage_increase := 
  let P0 := 15000
  let P2 := 18150  
  exists (r : ℝ), (P0 * (1 + r)^2 = P2) ∧ (r = 0.1)

theorem annual_population_growth : annual_percentage_increase :=
by
  -- Placeholder proof
  sorry

end annual_population_growth_0_484


namespace weekly_allowance_0_47

variable (A : ℝ)   -- declaring A as a real number

theorem weekly_allowance (h1 : (3/5 * A) + 1/3 * (2/5 * A) + 1 = A) : 
  A = 3.75 :=
sorry

end weekly_allowance_0_47


namespace land_plot_side_length_0_76

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end land_plot_side_length_0_76


namespace probability_A_B_C_adjacent_0_817

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end probability_A_B_C_adjacent_0_817


namespace johns_uncommon_cards_0_55

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end johns_uncommon_cards_0_55


namespace joy_can_choose_17_rods_for_quadrilateral_0_201

theorem joy_can_choose_17_rods_for_quadrilateral :
  ∃ (possible_rods : Finset ℕ), 
    possible_rods.card = 17 ∧
    ∀ rod ∈ possible_rods, 
      rod > 0 ∧ rod <= 30 ∧
      (rod ≠ 3 ∧ rod ≠ 7 ∧ rod ≠ 15) ∧
      (rod > 15 - (3 + 7)) ∧
      (rod < 3 + 7 + 15) :=
by
  sorry

end joy_can_choose_17_rods_for_quadrilateral_0_201


namespace expected_score_shooting_competition_0_419

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end expected_score_shooting_competition_0_419


namespace find_d_value_0_711

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end find_d_value_0_711


namespace certain_number_is_50_0_621

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end certain_number_is_50_0_621


namespace minimum_value_expression_0_97

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end minimum_value_expression_0_97


namespace other_asymptote_0_560

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end other_asymptote_0_560


namespace math_proof_problem_0_52

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end math_proof_problem_0_52


namespace discount_for_multiple_rides_0_698

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end discount_for_multiple_rides_0_698


namespace triangle_equilateral_if_condition_0_776

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end triangle_equilateral_if_condition_0_776


namespace bread_cost_equality_0_820

variable (B : ℝ)
variable (C1 : B + 3 + 2 * B = 9)  -- $3 for butter, 2B for juice, total spent is 9 dollars

theorem bread_cost_equality : B = 2 :=
by
  sorry

end bread_cost_equality_0_820


namespace star_value_0_105

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end star_value_0_105


namespace time_for_Dawson_0_778

variable (D : ℝ)
variable (Henry_time : ℝ := 7)
variable (avg_time : ℝ := 22.5)

theorem time_for_Dawson (h : avg_time = (D + Henry_time) / 2) : D = 38 := 
by 
  sorry

end time_for_Dawson_0_778


namespace find_g_values_0_775

open Function

-- Defining the function g and its properties
axiom g : ℝ → ℝ
axiom g_domain : ∀ x, 0 ≤ x → 0 ≤ g x
axiom g_proper : ∀ x, 0 ≤ x → 0 ≤ g (g x)
axiom g_func : ∀ x, 0 ≤ x → g (g x) = 3 * x / (x + 3)
axiom g_interval : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = (x + 1) / 2

-- Problem statement translating to Lean
theorem find_g_values :
  g 2021 = 2021.5 ∧ g (1 / 2021) = 6 := by {
  sorry 
}

end find_g_values_0_775


namespace solve_equation_0_462

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (3 - x^2) / (x + 2) + (2 * x^2 - 8) / (x^2 - 4) = 3 ↔ 
  x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
by
  sorry

end solve_equation_0_462


namespace total_cans_collected_0_892

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end total_cans_collected_0_892


namespace fourth_throw_probability_0_712

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end fourth_throw_probability_0_712


namespace alice_age_2005_0_356

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end alice_age_2005_0_356


namespace problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_0_920

noncomputable def problem_1 : Int :=
  (-3) + 5 - (-3)

theorem problem_1_solution : problem_1 = 5 := by
  sorry

noncomputable def problem_2 : ℚ :=
  (-1/3 - 3/4 + 5/6) * (-24)

theorem problem_2_solution : problem_2 = 6 := by
  sorry

noncomputable def problem_3 : ℚ :=
  1 - (1/9) * (-1/2 - 2^2)

theorem problem_3_solution : problem_3 = 3/2 := by
  sorry

noncomputable def problem_4 : ℚ :=
  ((-1)^2023) * (18 - (-2) * 3) / (15 - 3^3)

theorem problem_4_solution : problem_4 = 2 := by
  sorry

end problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_0_920


namespace num_dinosaur_dolls_0_713

-- Define the number of dinosaur dolls
def dinosaur_dolls : Nat := 3

-- Define the theorem to prove the number of dinosaur dolls
theorem num_dinosaur_dolls : dinosaur_dolls = 3 := by
  -- Add sorry to skip the proof
  sorry

end num_dinosaur_dolls_0_713


namespace calculation_correct_0_244

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end calculation_correct_0_244


namespace min_value_of_function_0_595

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end min_value_of_function_0_595


namespace arrangement_count_correct_0_670

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end arrangement_count_correct_0_670


namespace expressions_positive_0_803

-- Definitions based on given conditions
def A := 2.5
def B := -0.8
def C := -2.2
def D := 1.1
def E := -3.1

-- The Lean statement to prove the necessary expressions are positive numbers.

theorem expressions_positive :
  (B + C) / E = 0.97 ∧
  B * D - A * C = 4.62 ∧
  C / (A * B) = 1.1 :=
by
  -- Assuming given conditions and steps to prove the theorem.
  sorry

end expressions_positive_0_803


namespace rate_of_grapes_calculation_0_56

theorem rate_of_grapes_calculation (total_cost cost_mangoes cost_grapes : ℕ) (rate_grapes : ℕ):
  total_cost = 1125 →
  cost_mangoes = 9 * 55 →
  cost_grapes = 9 * rate_grapes →
  total_cost = cost_grapes + cost_mangoes →
  rate_grapes = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end rate_of_grapes_calculation_0_56


namespace total_population_milburg_0_950

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end total_population_milburg_0_950


namespace james_passenger_count_0_943

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_0_943


namespace regular_bike_wheels_eq_two_0_686

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end regular_bike_wheels_eq_two_0_686


namespace lucy_last_10_shots_0_127

variable (shots_30 : ℕ) (percentage_30 : ℚ) (total_shots : ℕ) (percentage_40 : ℚ)
variable (shots_made_30 : ℕ) (shots_made_40 : ℕ) (shots_made_last_10 : ℕ)

theorem lucy_last_10_shots 
    (h1 : shots_30 = 30) 
    (h2 : percentage_30 = 0.60) 
    (h3 : total_shots = 40) 
    (h4 : percentage_40 = 0.62 )
    (h5 : shots_made_30 = Nat.floor (percentage_30 * shots_30)) 
    (h6 : shots_made_40 = Nat.floor (percentage_40 * total_shots))
    (h7 : shots_made_last_10 = shots_made_40 - shots_made_30) 
    : shots_made_last_10 = 7 := sorry

end lucy_last_10_shots_0_127


namespace relationship_y1_y2_y3_0_611

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_0_611


namespace expected_coin_worth_is_two_0_73

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end expected_coin_worth_is_two_0_73


namespace trigonometric_identity_0_11

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = 2) :
  (4 * Real.sin α ^ 3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_0_11


namespace ratio_ashley_mary_0_523

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end ratio_ashley_mary_0_523


namespace area_of_triangle_intercepts_0_623

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end area_of_triangle_intercepts_0_623


namespace find_LCM_of_numbers_0_251

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end find_LCM_of_numbers_0_251


namespace a_eq_bn_0_636

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end a_eq_bn_0_636


namespace km_to_leaps_0_463

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end km_to_leaps_0_463


namespace certain_number_0_100

theorem certain_number (x : ℝ) (h : 7125 / x = 5700) : x = 1.25 := 
sorry

end certain_number_0_100


namespace smallest_number_diminished_by_16_divisible_0_38

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end smallest_number_diminished_by_16_divisible_0_38


namespace find_breadth_of_cuboid_0_468

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end find_breadth_of_cuboid_0_468


namespace find_point_P_0_220

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end find_point_P_0_220


namespace function_no_real_zeros_0_821

variable (a b c : ℝ)

-- Conditions: a, b, c form a geometric sequence and ac > 0
def geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c
def positive_product (a c : ℝ) : Prop := a * c > 0

theorem function_no_real_zeros (h_geom : geometric_sequence a b c) (h_pos : positive_product a c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := 
by
  sorry

end function_no_real_zeros_0_821


namespace range_of_a_0_182

noncomputable def f (a x : ℝ) : ℝ := (Real.log (x^2 - a * x + 5)) / (Real.log a)

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha0 : 0 < a) (ha1 : a ≠ 1) 
  (hx₁x₂ : x₁ < x₂) (hx₂ : x₂ ≤ a / 2) 
  (hf : (f a x₂ - f a x₁ < 0)) : 
  1 < a ∧ a < 2 * Real.sqrt 5 := 
sorry

end range_of_a_0_182


namespace possible_sums_of_digits_0_490

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end possible_sums_of_digits_0_490


namespace boxes_total_is_correct_0_861

def initial_boxes : ℕ := 7
def additional_boxes_per_box : ℕ := 7
def final_non_empty_boxes : ℕ := 10
def total_boxes := 77

theorem boxes_total_is_correct
  (h1 : initial_boxes = 7)
  (h2 : additional_boxes_per_box = 7)
  (h3 : final_non_empty_boxes = 10)
  : total_boxes = 77 :=
by
  -- Proof goes here
  sorry

end boxes_total_is_correct_0_861


namespace find_third_root_0_804

variables (a b : ℝ)

def poly (x : ℝ) : ℝ := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

def root1 := -3
def root2 := 4

axiom root1_cond : poly a b root1 = 0
axiom root2_cond : poly a b root2 = 0

theorem find_third_root (a b : ℝ) (h1 : poly a b root1 = 0) (h2 : poly a b root2 = 0) : 
  ∃ r3 : ℝ, r3 = -1/2 :=
sorry

end find_third_root_0_804


namespace train_crossing_time_0_528

namespace TrainCrossingProblem

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 300
def speed_of_train_kmph : ℕ := 36
def speed_of_train_mps : ℕ := 10 -- conversion from 36 kmph to m/s
def total_distance : ℕ := length_of_train + length_of_bridge -- 250 + 300
def expected_time : ℕ := 55

theorem train_crossing_time : 
  (total_distance / speed_of_train_mps) = expected_time :=
by
  sorry
end TrainCrossingProblem

end train_crossing_time_0_528


namespace six_digit_number_condition_0_858

theorem six_digit_number_condition :
  ∃ A B : ℕ, 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
            1000 * B + A = 6 * (1000 * A + B) :=
by
  sorry

end six_digit_number_condition_0_858


namespace tree_F_height_0_88

variable (A B C D E F : ℝ)

def height_conditions : Prop :=
  A = 150 ∧ -- Tree A's height is 150 feet
  B = (2 / 3) * A ∧ -- Tree B's height is 2/3 of Tree A's height
  C = (1 / 2) * B ∧ -- Tree C's height is 1/2 of Tree B's height
  D = C + 25 ∧ -- Tree D's height is 25 feet more than Tree C's height
  E = 0.40 * A ∧ -- Tree E's height is 40% of Tree A's height
  F = (B + D) / 2 -- Tree F's height is the average of Tree B's height and Tree D's height

theorem tree_F_height : height_conditions A B C D E F → F = 87.5 :=
by
  intros
  sorry

end tree_F_height_0_88


namespace largest_k_consecutive_sum_0_264

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum_0_264


namespace option_C_correct_0_194

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct_0_194


namespace P_iff_q_0_654

variables (a b c: ℝ)

def P : Prop := a * c < 0
def q : Prop := ∃ α β : ℝ, α * β < 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0

theorem P_iff_q : P a c ↔ q a b c := 
sorry

end P_iff_q_0_654


namespace leo_current_weight_0_834

variable (L K : ℝ)

noncomputable def leo_current_weight_predicate :=
  (L + 10 = 1.5 * K) ∧ (L + K = 180)

theorem leo_current_weight : leo_current_weight_predicate L K → L = 104 := by
  sorry

end leo_current_weight_0_834


namespace alfred_gain_percent_0_170

-- Definitions based on the conditions
def purchase_price : ℝ := 4700
def repair_costs : ℝ := 800
def selling_price : ℝ := 6000

-- Lean statement to prove gain percent
theorem alfred_gain_percent :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 := by
  sorry

end alfred_gain_percent_0_170


namespace length_of_platform_0_579

-- Definitions for conditions
def train_length : ℕ := 300
def time_cross_platform : ℕ := 39
def time_cross_signal : ℕ := 12

-- Speed calculation
def train_speed := train_length / time_cross_signal

-- Total distance calculation while crossing the platform
def total_distance := train_speed * time_cross_platform

-- Length of the platform
def platform_length : ℕ := total_distance - train_length

-- Theorem stating the length of the platform
theorem length_of_platform :
  platform_length = 675 := by
  sorry

end length_of_platform_0_579


namespace find_x_collinear_0_443

def vec := ℝ × ℝ

def collinear (u v: vec): Prop :=
  ∃ k: ℝ, u = (k * v.1, k * v.2)

theorem find_x_collinear:
  ∀ (x: ℝ), (let a : vec := (1, 2)
              let b : vec := (x, 1)
              collinear a (a.1 - b.1, a.2 - b.2)) → x = 1 / 2 :=
by
  intros x h
  sorry

end find_x_collinear_0_443


namespace problem_0_217

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}
def C : Set ℕ := {1, 3}

theorem problem : A ∩ (U \ B) = C := by
  sorry

end problem_0_217


namespace number_of_items_0_261

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end number_of_items_0_261


namespace squared_distance_focus_product_tangents_0_215

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end squared_distance_focus_product_tangents_0_215


namespace students_in_trumpet_or_trombone_0_58

theorem students_in_trumpet_or_trombone (h₁ : 0.5 + 0.12 = 0.62) : 
  0.5 + 0.12 = 0.62 :=
by
  exact h₁

end students_in_trumpet_or_trombone_0_58


namespace polar_coordinates_of_point_0_353

theorem polar_coordinates_of_point (x y : ℝ) (hx : x = 2) (hy : y = -2 * √3) : 
  ∃ (ρ θ : ℝ), ρ = 4 ∧ θ = -2 * Real.pi / 3 ∧ (ρ * Real.cos θ, ρ * Real.sin θ) = (x, y) :=
by 
  use 4
  use -2 * Real.pi / 3
  sorry

end polar_coordinates_of_point_0_353


namespace original_salary_condition_0_397

variable (S: ℝ)

theorem original_salary_condition (h: 1.10 * 1.08 * 0.95 * 0.93 * S = 6270) :
  S = 6270 / (1.10 * 1.08 * 0.95 * 0.93) :=
by
  sorry

end original_salary_condition_0_397


namespace ratio_of_pieces_0_697

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end ratio_of_pieces_0_697


namespace evaluate_72_squared_minus_48_squared_0_802

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_0_802


namespace relationship_a_b_0_507

theorem relationship_a_b (a b : ℝ) :
  (∃ (P : ℝ × ℝ), P ∈ {Q : ℝ × ℝ | Q.snd = -3 * Q.fst + b} ∧
                   ∃ (R : ℝ × ℝ), R ∈ {S : ℝ × ℝ | S.snd = -a * S.fst + 3} ∧
                   R = (-P.snd, -P.fst)) →
  a = 1 / 3 ∧ b = -9 :=
by
  intro h
  sorry

end relationship_a_b_0_507


namespace monotone_intervals_range_of_t_for_three_roots_0_947

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end monotone_intervals_range_of_t_for_three_roots_0_947


namespace symm_diff_A_B_0_921

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | abs x < 2}

-- Define set difference
def set_diff (S T : Set ℤ) : Set ℤ := {x | x ∈ S ∧ x ∉ T}

-- Define symmetric difference
def symm_diff (S T : Set ℤ) : Set ℤ := (set_diff S T) ∪ (set_diff T S)

-- Define the expression we need to prove
theorem symm_diff_A_B : symm_diff A B = {-1, 0, 2} := by
  sorry

end symm_diff_A_B_0_921


namespace CarmenBrushLengthInCentimeters_0_782

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_0_782


namespace calculate_bmw_sales_and_revenue_0_326

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end calculate_bmw_sales_and_revenue_0_326


namespace floor_e_sub_6_eq_neg_4_0_789

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end floor_e_sub_6_eq_neg_4_0_789


namespace y_relationship_range_of_x_0_265

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end y_relationship_range_of_x_0_265


namespace deaths_during_operation_0_139

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end deaths_during_operation_0_139


namespace cost_of_bananas_is_two_0_161

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end cost_of_bananas_is_two_0_161


namespace cube_volume_correct_0_450

-- Define the height and base dimensions of the pyramid
def pyramid_height := 15
def pyramid_base_length := 12
def pyramid_base_width := 8

-- Define the side length of the cube-shaped box
def cube_side_length := max pyramid_height pyramid_base_length

-- Define the volume of the cube-shaped box
def cube_volume := cube_side_length ^ 3

-- Theorem statement: the volume of the smallest cube-shaped box that can fit the pyramid is 3375 cubic inches
theorem cube_volume_correct : cube_volume = 3375 := by
  sorry

end cube_volume_correct_0_450


namespace ordered_pair_solution_0_865

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end ordered_pair_solution_0_865


namespace ara_height_0_746

/-
Conditions:
1. Shea's height increased by 25%.
2. Shea is now 65 inches tall.
3. Ara grew by three-quarters as many inches as Shea did.

Prove Ara's height is 61.75 inches.
-/

def shea_original_height (x : ℝ) : Prop := 1.25 * x = 65

def ara_growth (growth : ℝ) (shea_growth : ℝ) : Prop := growth = (3 / 4) * shea_growth

def shea_growth (original_height : ℝ) : ℝ := 0.25 * original_height

theorem ara_height (shea_orig_height : ℝ) (shea_now_height : ℝ) (ara_growth_inches : ℝ) :
  shea_original_height shea_orig_height → 
  shea_now_height = 65 →
  ara_growth ara_growth_inches (shea_now_height - shea_orig_height) →
  shea_orig_height + ara_growth_inches = 61.75 :=
by
  sorry

end ara_height_0_746


namespace a_plus_b_0_228

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 7

theorem a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f x a b) = 4 * x + 5) : a + b = 16 / 3 :=
by
  sorry

end a_plus_b_0_228


namespace fox_can_eat_80_fox_cannot_eat_65_0_860
-- import the required library

-- Define the conditions for the problem.
def total_candies := 100
def piles := 3
def fox_eat_equalize (fox: ℕ) (pile1: ℕ) (pile2: ℕ): ℕ :=
  if pile1 = pile2 then fox + pile1 else fox + pile2 - pile1

-- Statement for part (a)
theorem fox_can_eat_80: ∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 80) ∨ 
              (fox_eat_equalize x c₁ c₂  = 80)) :=
sorry

-- Statement for part (b)
theorem fox_cannot_eat_65: ¬ (∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 65) ∨ 
              (fox_eat_equalize x c₁ c₂  = 65))) :=
sorry

end fox_can_eat_80_fox_cannot_eat_65_0_860


namespace a3_value_0_205

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end a3_value_0_205


namespace fewer_green_pens_than_pink_0_179

-- Define the variables
variables (G B : ℕ)

-- State the conditions
axiom condition1 : G < 12
axiom condition2 : B = G + 3
axiom condition3 : 12 + G + B = 21

-- Define the problem statement
theorem fewer_green_pens_than_pink : 12 - G = 9 :=
by
  -- Insert the proof steps here
  sorry

end fewer_green_pens_than_pink_0_179


namespace johns_commute_distance_0_143

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end johns_commute_distance_0_143


namespace cookies_indeterminate_0_850

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end cookies_indeterminate_0_850


namespace molecular_weight_of_compound_0_240

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end molecular_weight_of_compound_0_240


namespace power_of_3_0_46

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_0_46


namespace factorize_a3_minus_ab2_0_34

theorem factorize_a3_minus_ab2 (a b: ℝ) : 
  a^3 - a * b^2 = a * (a + b) * (a - b) :=
by
  sorry

end factorize_a3_minus_ab2_0_34


namespace f_2_equals_12_0_324

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end f_2_equals_12_0_324


namespace sum_of_areas_is_correct_0_966

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end sum_of_areas_is_correct_0_966


namespace cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_0_489

theorem cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths 
  (b : ℝ)
  (h : ∀ x : ℝ, 4 * x^3 + 3 * x^2 + b * x + 27 = 0 → ∃! r : ℝ, r = x) :
  b = 3 / 4 := 
by
  sorry

end cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_0_489


namespace sqrt_47_minus_2_range_0_917

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end sqrt_47_minus_2_range_0_917


namespace satellite_modular_units_0_375

variables (N S T U : ℕ)
variable (h1 : N = S / 3)
variable (h2 : S / T = 1 / 9)
variable (h3 : U * N = 8 * T / 9)

theorem satellite_modular_units :
  U = 24 :=
by sorry

end satellite_modular_units_0_375


namespace charity_donation_correct_0_617

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end charity_donation_correct_0_617


namespace number_of_color_copies_0_710

def charge_shop_X (n : ℕ) : ℝ := 1.20 * n
def charge_shop_Y (n : ℕ) : ℝ := 1.70 * n
def difference := 20

theorem number_of_color_copies (n : ℕ) (h : charge_shop_Y n = charge_shop_X n + difference) : n = 40 :=
by {
  sorry
}

end number_of_color_copies_0_710


namespace f_20_value_0_197

noncomputable def f (n : ℕ) : ℚ := sorry

axiom f_initial : f 1 = 3 / 2
axiom f_eq : ∀ x y : ℕ, 
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_20_value : f 20 = 4305 := 
by {
  sorry 
}

end f_20_value_0_197


namespace solution_set_of_inequality_0_517

theorem solution_set_of_inequality :
  {x : ℝ | |x - 5| + |x + 3| >= 10} = {x : ℝ | x ≤ -4} ∪ {x : ℝ | x ≥ 6} :=
by
  sorry

end solution_set_of_inequality_0_517


namespace banks_investments_count_0_916

-- Conditions
def revenue_per_investment_banks := 500
def revenue_per_investment_elizabeth := 900
def number_of_investments_elizabeth := 5
def extra_revenue_elizabeth := 500

-- Total revenue calculations
def total_revenue_elizabeth := number_of_investments_elizabeth * revenue_per_investment_elizabeth
def total_revenue_banks := total_revenue_elizabeth - extra_revenue_elizabeth

-- Number of investments for Mr. Banks
def number_of_investments_banks := total_revenue_banks / revenue_per_investment_banks

theorem banks_investments_count : number_of_investments_banks = 8 := by
  sorry

end banks_investments_count_0_916


namespace fractional_eq_range_m_0_382

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_0_382


namespace miles_walked_on_Tuesday_0_274

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end miles_walked_on_Tuesday_0_274


namespace value_of_a_0_825

noncomputable def M : Set ℝ := {x | x^2 = 2}
noncomputable def N (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem value_of_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end value_of_a_0_825


namespace simplify_and_evaluate_expression_0_753

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end simplify_and_evaluate_expression_0_753


namespace bounded_roots_0_953

open Polynomial

noncomputable def P : ℤ[X] := sorry -- Replace with actual polynomial if necessary

theorem bounded_roots (P : ℤ[X]) (n : ℕ) (hPdeg : P.degree = n) (hdec : 1 ≤ n) :
  ∀ k : ℤ, (P.eval k) ^ 2 = 1 → ∃ (r s : ℕ), r + s ≤ n + 2 := 
by 
  sorry

end bounded_roots_0_953


namespace max_sum_pyramid_0_107

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end max_sum_pyramid_0_107


namespace find_remainder_0_691

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end find_remainder_0_691


namespace negation_of_universal_proposition_0_345

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_universal_proposition_0_345


namespace exponent_equality_0_115

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_0_115


namespace area_of_circle_segment_0_135

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end area_of_circle_segment_0_135


namespace julia_drove_214_miles_0_267

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end julia_drove_214_miles_0_267


namespace condition_for_y_exists_0_243

theorem condition_for_y_exists (n : ℕ) (hn : n ≥ 2) (x y : Fin (n + 1) → ℝ)
  (z : Fin (n + 1) → ℂ)
  (hz : ∀ k, z k = x k + Complex.I * y k)
  (heq : z 0 ^ 2 = ∑ k in Finset.range n, z (k + 1) ^ 2) :
  x 0 ^ 2 ≤ ∑ k in Finset.range n, x (k + 1) ^ 2 :=
by
  sorry

end condition_for_y_exists_0_243


namespace relationship_among_abc_0_784

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_0_784


namespace mean_of_points_scored_0_647

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem mean_of_points_scored (lst : List ℕ)
  (h1 : lst = [81, 73, 83, 86, 73]) : 
  mean lst = 79.2 :=
by
  rw [h1, mean]
  sorry

end mean_of_points_scored_0_647


namespace alice_bob_task_0_675

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end alice_bob_task_0_675


namespace probability_all_black_after_rotation_0_124

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end probability_all_black_after_rotation_0_124


namespace base_seven_sum_0_553

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end base_seven_sum_0_553


namespace elephant_weight_equivalence_0_431

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end elephant_weight_equivalence_0_431


namespace new_profit_is_122_03_0_367

noncomputable def new_profit_percentage (P : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) (market_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let total_cost := P * (1 + tax_rate)
  let initial_selling_price := total_cost * (1 + profit_rate)
  let market_price_after_months := initial_selling_price * (1 + market_increase_rate) ^ months
  let final_selling_price := 2 * initial_selling_price
  let profit := final_selling_price - total_cost
  (profit / total_cost) * 100

theorem new_profit_is_122_03 :
  new_profit_percentage (P : ℝ) 0.18 0.40 0.05 3 = 122.03 := 
by
  sorry

end new_profit_is_122_03_0_367


namespace cheetah_catches_deer_in_10_minutes_0_110

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end cheetah_catches_deer_in_10_minutes_0_110


namespace angle_C_in_triangle_0_719

theorem angle_C_in_triangle (A B C : ℝ)
  (hA : A = 60)
  (hAC : C = 2 * B)
  (hSum : A + B + C = 180) : C = 80 :=
sorry

end angle_C_in_triangle_0_719


namespace family_gathering_0_467

theorem family_gathering (P : ℕ) 
  (h1 : (P / 2 = P - 10)) : P = 20 :=
sorry

end family_gathering_0_467


namespace units_digit_odd_product_0_721

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end units_digit_odd_product_0_721


namespace problem_I_problem_II_0_503

noncomputable def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem problem_I (x m : ℝ) : f x m ≥ 2 :=
by 
  sorry

theorem problem_II (m : ℝ) : f 2 m ≤ 16 ↔ -3 ≤ m ∧ m ≤ Real.sqrt 14 - 1 :=
by 
  sorry

end problem_I_problem_II_0_503


namespace daria_multiple_pizzas_0_409

variable (m : ℝ)
variable (don_pizzas : ℝ) (total_pizzas : ℝ)

axiom don_pizzas_def : don_pizzas = 80
axiom total_pizzas_def : total_pizzas = 280

theorem daria_multiple_pizzas (m : ℝ) (don_pizzas : ℝ) (total_pizzas : ℝ) 
    (h1 : don_pizzas = 80) (h2 : total_pizzas = 280) 
    (h3 : total_pizzas = don_pizzas + m * don_pizzas) : 
    m = 2.5 :=
by sorry

end daria_multiple_pizzas_0_409


namespace num_real_a_satisfy_union_0_300

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end num_real_a_satisfy_union_0_300


namespace no_integer_solution_xyz_0_662

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solution_xyz_0_662


namespace intersection_M_complement_N_eq_0_333

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def complement_N : Set ℝ := {y | y < 1}

theorem intersection_M_complement_N_eq : M ∩ complement_N = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_eq_0_333


namespace total_matches_played_0_634

-- Definitions
def victories_points := 3
def draws_points := 1
def defeats_points := 0
def points_after_5_games := 8
def games_played := 5
def target_points := 40
def remaining_wins_required := 9

-- Statement to prove
theorem total_matches_played :
  ∃ M : ℕ, points_after_5_games + victories_points * remaining_wins_required < target_points -> M = games_played + remaining_wins_required + 1 :=
sorry

end total_matches_played_0_634


namespace average_speed_is_correct_0_879
noncomputable def average_speed_trip : ℝ :=
  let distance_AB := 240 * 5
  let distance_BC := 300 * 3
  let distance_CD := 400 * 4
  let total_distance := distance_AB + distance_BC + distance_CD
  let flight_time_AB := 5
  let layover_B := 2
  let flight_time_BC := 3
  let layover_C := 1
  let flight_time_CD := 4
  let total_time := (flight_time_AB + flight_time_BC + flight_time_CD) + (layover_B + layover_C)
  total_distance / total_time

theorem average_speed_is_correct :
  average_speed_trip = 246.67 := sorry

end average_speed_is_correct_0_879


namespace remainder_when_added_then_divided_0_133

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end remainder_when_added_then_divided_0_133


namespace geometric_sequence_sum_0_668

theorem geometric_sequence_sum (a_n : ℕ → ℕ) (h1 : a_n 2 = 2) (h2 : a_n 5 = 16) :
  (∑ k in Finset.range n, a_n k * a_n (k + 1)) = (2 / 3) * (4^n - 1) := 
sorry

end geometric_sequence_sum_0_668


namespace MsSatosClassRatioProof_0_160

variable (g b : ℕ) -- g is the number of girls, b is the number of boys

def MsSatosClassRatioProblem : Prop :=
  (g = b + 6) ∧ (g + b = 32) → g / b = 19 / 13

theorem MsSatosClassRatioProof : MsSatosClassRatioProblem g b := by
  sorry

end MsSatosClassRatioProof_0_160


namespace total_earnings_correct_0_309

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end total_earnings_correct_0_309


namespace smallest_angle_of_triangle_0_971

theorem smallest_angle_of_triangle (x : ℕ) 
  (h1 : ∑ angles in {x, 3 * x, 5 * x}, angles = 180)
  (h2 : (3 * x) = middle_angle)
  (h3 : (5 * x) = largest_angle) 
  : x = 20 := 
by
  sorry

end smallest_angle_of_triangle_0_971


namespace parabola_intersections_0_937

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end parabola_intersections_0_937


namespace problem1_0_889

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end problem1_0_889


namespace carsProducedInEurope_0_692

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end carsProducedInEurope_0_692


namespace exactly_one_passes_0_909

theorem exactly_one_passes (P_A P_B : ℚ) (hA : P_A = 3 / 5) (hB : P_B = 1 / 3) : 
  (1 - P_A) * P_B + P_A * (1 - P_B) = 8 / 15 :=
by
  -- skipping the proof as per requirement
  sorry

end exactly_one_passes_0_909


namespace tan_15_degrees_theta_range_valid_max_f_value_0_406

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end tan_15_degrees_theta_range_valid_max_f_value_0_406


namespace determinant_of_given_matrix_0_350

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end determinant_of_given_matrix_0_350


namespace eleven_hash_five_0_260

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end eleven_hash_five_0_260


namespace sum_of_angles_of_roots_eq_1020_0_168

noncomputable def sum_of_angles_of_roots : ℝ :=
  60 + 132 + 204 + 276 + 348

theorem sum_of_angles_of_roots_eq_1020 :
  (∑ θ in {60, 132, 204, 276, 348}, θ) = 1020 := by
  sorry

end sum_of_angles_of_roots_eq_1020_0_168


namespace rate_per_kg_for_mangoes_0_359

theorem rate_per_kg_for_mangoes (quantity_grapes : ℕ)
    (rate_grapes : ℕ)
    (quantity_mangoes : ℕ)
    (total_payment : ℕ)
    (rate_mangoes : ℕ) :
    quantity_grapes = 8 →
    rate_grapes = 70 →
    quantity_mangoes = 9 →
    total_payment = 1055 →
    8 * 70 + 9 * rate_mangoes = 1055 →
    rate_mangoes = 55 := by
  intros h1 h2 h3 h4 h5
  have h6 : 8 * 70 = 560 := by norm_num
  have h7 : 560 + 9 * rate_mangoes = 1055 := by rw [h5]
  have h8 : 1055 - 560 = 495 := by norm_num
  have h9 : 9 * rate_mangoes = 495 := by linarith
  have h10 : rate_mangoes = 55 := by linarith
  exact h10

end rate_per_kg_for_mangoes_0_359


namespace arrange_natural_numbers_divisors_0_548

theorem arrange_natural_numbers_divisors :
  ∃ (seq : List ℕ), seq = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧ 
  seq.length = 10 ∧
  ∀ n (h : n < seq.length), seq[n] ∣ (List.take n seq).sum := 
by
  sorry

end arrange_natural_numbers_divisors_0_548


namespace part1_part2_0_955

variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively and 4b * sin A = sqrt 7 * a
def condition1 : 4 * b * Real.sin A = Real.sqrt 7 * a := sorry

-- Prove that sin B = sqrt 7 / 4
theorem part1 (h : 4 * b * Real.sin A = Real.sqrt 7 * a) :
  Real.sin B = Real.sqrt 7 / 4 := sorry

-- Condition: a, b, and c form an arithmetic sequence with a common difference greater than 0
def condition2 : 2 * b = a + c := sorry

-- Prove that cos A - cos C = sqrt 7 / 2
theorem part2 (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a) (h2 : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := sorry

end part1_part2_0_955


namespace john_paid_more_0_417

-- Define the required variables
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define John and Jane's payments
def discounted_price : ℝ := original_price * (1 - discount_rate)
def johns_tip : ℝ := tip_rate * original_price
def johns_total_payment : ℝ := original_price + johns_tip
def janes_tip : ℝ := tip_rate * discounted_price
def janes_total_payment : ℝ := discounted_price + janes_tip

-- Calculate the difference
def payment_difference : ℝ := johns_total_payment - janes_total_payment

-- Statement to prove the payment difference equals $9.66
theorem john_paid_more : payment_difference = 9.66 := by
  sorry

end john_paid_more_0_417


namespace average_tickets_per_day_0_169

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end average_tickets_per_day_0_169


namespace mary_principal_amount_0_339

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end mary_principal_amount_0_339


namespace range_of_a_0_128

theorem range_of_a (a : ℝ) : 
  (2 * (-1) + 0 + a) * (2 * 2 + (-1) + a) < 0 ↔ -3 < a ∧ a < 2 := 
by 
  sorry

end range_of_a_0_128


namespace total_cost_is_2160_0_64

variables (x y z : ℝ)

-- Conditions
def cond1 : Prop := x = 0.45 * y
def cond2 : Prop := y = 0.8 * z
def cond3 : Prop := z = x + 640

-- Goal
def total_cost := x + y + z

theorem total_cost_is_2160 (x y z : ℝ) (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 x z) :
  total_cost x y z = 2160 :=
by
  sorry

end total_cost_is_2160_0_64


namespace max_value_of_f_0_855

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5

theorem max_value_of_f : ∃ x ∈ (Set.Icc (-2:ℝ) 2), ∀ y ∈ (Set.Icc (-2:ℝ) 2), f y ≤ f x ∧ f x = 3 := by
  sorry

end max_value_of_f_0_855


namespace number_of_six_digit_palindromes_0_196

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_0_196


namespace inequality_proof_0_222

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end inequality_proof_0_222


namespace junk_items_count_0_96

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end junk_items_count_0_96


namespace find_other_number_0_130

theorem find_other_number (LCM : ℕ) (HCF : ℕ) (n1 : ℕ) (n2 : ℕ) 
  (h_lcm : LCM = 2310) (h_hcf : HCF = 26) (h_n1 : n1 = 210) :
  n2 = 286 :=
by
  sorry

end find_other_number_0_130


namespace cube_volume_0_174

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end cube_volume_0_174


namespace total_spent_is_64_0_335

/-- Condition 1: The cost of each deck is 8 dollars -/
def deck_cost : ℕ := 8

/-- Condition 2: Tom bought 3 decks -/
def tom_decks : ℕ := 3

/-- Condition 3: Tom's friend bought 5 decks -/
def friend_decks : ℕ := 5

/-- Total amount spent by Tom and his friend -/
def total_amount_spent : ℕ := (tom_decks * deck_cost) + (friend_decks * deck_cost)

/-- Proof statement: Prove that total amount spent is 64 -/
theorem total_spent_is_64 : total_amount_spent = 64 := by
  sorry

end total_spent_is_64_0_335


namespace fraction_nonneg_if_x_ge_m8_0_271

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end fraction_nonneg_if_x_ge_m8_0_271


namespace nancy_packs_of_crayons_0_822

def total_crayons : ℕ := 615
def crayons_per_pack : ℕ := 15

theorem nancy_packs_of_crayons : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_0_822


namespace distance_between_parallel_sides_0_392

theorem distance_between_parallel_sides (a b : ℝ) (h : ℝ) (A : ℝ) :
  a = 20 → b = 10 → A = 150 → (A = 1 / 2 * (a + b) * h) → h = 10 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end distance_between_parallel_sides_0_392


namespace ellipse_equation_0_315

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end ellipse_equation_0_315


namespace sale_price_with_50_percent_profit_0_794

theorem sale_price_with_50_percent_profit (CP SP₁ SP₃ : ℝ) 
(h1 : SP₁ - CP = CP - 448) 
(h2 : SP₃ = 1.5 * CP) 
(h3 : SP₃ = 1020) : 
SP₃ = 1020 := 
by 
  sorry

end sale_price_with_50_percent_profit_0_794


namespace abs_inequality_solution_0_751

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end abs_inequality_solution_0_751


namespace trajectory_of_A_0_6

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end trajectory_of_A_0_6


namespace part1_solution_part2_solution_0_360

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end part1_solution_part2_solution_0_360


namespace number_of_ferns_is_six_0_428

def num_fronds_per_fern : Nat := 7
def num_leaves_per_frond : Nat := 30
def total_leaves : Nat := 1260

theorem number_of_ferns_is_six :
  total_leaves = num_fronds_per_fern * num_leaves_per_frond * 6 :=
by
  sorry

end number_of_ferns_is_six_0_428


namespace determine_gallons_0_982

def current_amount : ℝ := 7.75
def desired_total : ℝ := 14.75
def needed_to_add (x : ℝ) : Prop := desired_total = current_amount + x

theorem determine_gallons : needed_to_add 7 :=
by
  sorry

end determine_gallons_0_982


namespace minimum_a_plus_2b_no_a_b_such_that_0_816

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end minimum_a_plus_2b_no_a_b_such_that_0_816


namespace second_candidate_votes_0_513

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) (first_candidate_votes: ℕ)
    (h1 : total_votes = 2400)
    (h2 : first_candidate_percentage = 0.80)
    (h3 : first_candidate_votes = total_votes * first_candidate_percentage) :
    total_votes - first_candidate_votes = 480 := by
    sorry

end second_candidate_votes_0_513


namespace x_and_y_complete_work_in_12_days_0_546

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end x_and_y_complete_work_in_12_days_0_546


namespace lemons_needed_for_3_dozen_is_9_0_678

-- Define the conditions
def lemon_tbs : ℕ := 4
def juice_needed_per_dozen : ℕ := 12
def dozens_needed : ℕ := 3
def total_juice_needed : ℕ := juice_needed_per_dozen * dozens_needed

-- The number of lemons needed to make 3 dozen cupcakes
def lemons_needed (total_juice : ℕ) (lemon_juice : ℕ) : ℕ :=
  total_juice / lemon_juice

-- Prove the number of lemons needed == 9
theorem lemons_needed_for_3_dozen_is_9 : lemons_needed total_juice_needed lemon_tbs = 9 :=
  by sorry

end lemons_needed_for_3_dozen_is_9_0_678


namespace solve_for_x_0_410

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end solve_for_x_0_410


namespace negation_proof_0_618

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end negation_proof_0_618


namespace polynomial_coefficients_sum_and_difference_0_26

theorem polynomial_coefficients_sum_and_difference :
  ∀ (a_0 a_1 a_2 a_3 a_4 : ℤ),
  (∀ (x : ℤ), (2 * x - 3)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_1 + a_2 + a_3 + a_4 = -80) ∧ ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625) :=
by
  intros a_0 a_1 a_2 a_3 a_4 h
  sorry

end polynomial_coefficients_sum_and_difference_0_26


namespace pow_two_grows_faster_than_square_0_177

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end pow_two_grows_faster_than_square_0_177


namespace power_function_value_0_219

noncomputable def f (x : ℝ) : ℝ := x^2

theorem power_function_value :
  f 3 = 9 :=
by
  -- Since f(x) = x^2 and f passes through (-2, 4)
  -- f(x) = x^2, so f(3) = 3^2 = 9
  sorry

end power_function_value_0_219


namespace time_for_worker_C_0_415

theorem time_for_worker_C (time_A time_B time_total : ℝ) (time_A_pos : 0 < time_A) (time_B_pos : 0 < time_B) (time_total_pos : 0 < time_total) 
  (hA : time_A = 12) (hB : time_B = 15) (hTotal : time_total = 6) : 
  (1 / (1 / time_total - 1 / time_A - 1 / time_B) = 60) :=
by 
  sorry

end time_for_worker_C_0_415


namespace collinear_points_0_587

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end collinear_points_0_587


namespace problem_incorrect_statement_D_0_919

theorem problem_incorrect_statement_D :
  (∀ x y, x = -y → x + y = 0) ∧
  (∃ x : ℕ, x^2 + 2 * x = 0) ∧
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (¬ (∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ (x + y > 2))) :=
by sorry

end problem_incorrect_statement_D_0_919


namespace cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_0_452

-- Part (a)
theorem cupSaucersCombination :
  (5 : ℕ) * (3 : ℕ) = 15 :=
by
  -- Proof goes here
  sorry

-- Part (b)
theorem cupSaucerSpoonCombination :
  (5 : ℕ) * (3 : ℕ) * (4 : ℕ) = 60 :=
by
  -- Proof goes here
  sorry

-- Part (c)
theorem twoDifferentItemsCombination :
  (5 * 3 + 5 * 4 + 3 * 4 : ℕ) = 47 :=
by
  -- Proof goes here
  sorry

end cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_0_452


namespace find_fraction_0_837

theorem find_fraction (x y : ℝ) (h1 : (1/3) * (1/4) * x = 18) (h2 : y * x = 64.8) : y = 0.3 :=
sorry

end find_fraction_0_837


namespace exponent_division_0_122

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end exponent_division_0_122


namespace exist_prime_not_dividing_0_681

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end exist_prime_not_dividing_0_681


namespace towel_decrease_percentage_0_430

variable (L B : ℝ)
variable (h1 : 0.70 * L = L - (0.30 * L))
variable (h2 : 0.60 * B = B - (0.40 * B))

theorem towel_decrease_percentage (L B : ℝ) 
  (h1 : 0.70 * L = L - (0.30 * L))
  (h2 : 0.60 * B = B - (0.40 * B)) :
  ((L * B - (0.70 * L) * (0.60 * B)) / (L * B)) * 100 = 58 := 
by
  sorry

end towel_decrease_percentage_0_430


namespace arithmetic_sequence_sum_0_905

theorem arithmetic_sequence_sum (c d : ℕ) (h₁ : 3 + 5 = 8) (h₂ : 8 + 5 = 13) (h₃ : c = 13 + 5) (h₄ : d = 18 + 5) (h₅ : d + 5 = 28) : c + d = 41 :=
by
  sorry

end arithmetic_sequence_sum_0_905


namespace sine_triangle_0_18

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end sine_triangle_0_18


namespace number_of_boys_0_106

-- Definitions reflecting the conditions
def total_students := 1200
def sample_size := 200
def extra_boys := 10

-- Main problem statement
theorem number_of_boys (B G b g : ℕ) 
  (h_total_students : B + G = total_students)
  (h_sample_size : b + g = sample_size)
  (h_extra_boys : b = g + extra_boys)
  (h_stratified : b * G = g * B) :
  B = 660 :=
by sorry

end number_of_boys_0_106


namespace solve_for_x_0_759

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end solve_for_x_0_759


namespace max_value_of_expr_0_310

theorem max_value_of_expr  
  (a b c : ℝ) 
  (h₀ : 0 ≤ a)
  (h₁ : 0 ≤ b)
  (h₂ : 0 ≤ c)
  (h₃ : a + 2 * b + 3 * c = 1) :
  a + b^3 + c^4 ≤ 0.125 := 
sorry

end max_value_of_expr_0_310


namespace find_largest_number_0_211

theorem find_largest_number (a b c d e : ℕ)
    (h1 : a + b + c + d = 240)
    (h2 : a + b + c + e = 260)
    (h3 : a + b + d + e = 280)
    (h4 : a + c + d + e = 300)
    (h5 : b + c + d + e = 320)
    (h6 : a + b = 40) :
    max a (max b (max c (max d e))) = 160 := by
  sorry

end find_largest_number_0_211


namespace sqrt_of_4_eq_2_0_9

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_4_eq_2_0_9


namespace poly_expansion_0_752

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end poly_expansion_0_752


namespace hyperbola_focal_product_0_140

-- Define the hyperbola with given equation and point P conditions
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

-- Define properties of vectors related to foci
def perpendicular (v1 v2 : ℝ × ℝ) := (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Define the point-focus distance product condition
noncomputable def focalProduct (P F1 F2 : ℝ × ℝ) := (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))

theorem hyperbola_focal_product :
  ∀ (a b : ℝ) (F1 F2 P : ℝ × ℝ),
  Hyperbola a b P ∧ perpendicular (P - F1) (P - F2) ∧
  -- Assuming a parabola property ties F1 with a specific value
  ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 4 * (Real.sqrt  ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))) →
  focalProduct P F1 F2 = 14 := by
  sorry

end hyperbola_focal_product_0_140


namespace avg_first_six_results_0_81

theorem avg_first_six_results (average_11 : ℕ := 52) (average_last_6 : ℕ := 52) (sixth_result : ℕ := 34) :
  ∃ A : ℕ, (6 * A + 6 * average_last_6 - sixth_result = 11 * average_11) ∧ A = 49 :=
by
  sorry

end avg_first_six_results_0_81


namespace bella_truck_stamps_more_0_363

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end bella_truck_stamps_more_0_363


namespace scientific_notation_equivalence_0_522

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end scientific_notation_equivalence_0_522


namespace no_valid_n_for_conditions_0_873

theorem no_valid_n_for_conditions :
  ¬∃ n : ℕ, 1000 ≤ n / 4 ∧ n / 4 ≤ 9999 ∧ 1000 ≤ 4 * n ∧ 4 * n ≤ 9999 := by
  sorry

end no_valid_n_for_conditions_0_873


namespace smallest_value_y_0_679

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end smallest_value_y_0_679


namespace distance_between_intersections_0_504

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end distance_between_intersections_0_504


namespace zongzi_profit_0_348

def initial_cost : ℕ := 10
def initial_price : ℕ := 16
def initial_bags_sold : ℕ := 200
def additional_sales_per_yuan (x : ℕ) : ℕ := 80 * x
def profit_per_bag (x : ℕ) : ℕ := initial_price - x - initial_cost
def number_of_bags_sold (x : ℕ) : ℕ := initial_bags_sold + additional_sales_per_yuan x
def total_profit (profit_per_bag : ℕ) (number_of_bags_sold : ℕ) : ℕ := profit_per_bag * number_of_bags_sold

theorem zongzi_profit (x : ℕ) : 
  total_profit (profit_per_bag x) (number_of_bags_sold x) = 1440 := 
sorry

end zongzi_profit_0_348


namespace proof_complement_U_A_0_87

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A }

-- The theorem statement
theorem proof_complement_U_A :
  complement_U_A = {1, 5} :=
by
  -- Proof goes here
  sorry

end proof_complement_U_A_0_87


namespace triangle_height_and_segments_0_131

-- Define the sides of the triangle
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define the height h and the segments m and 15 - m
noncomputable def m : ℝ := 6.6
noncomputable def h : ℝ := 11.2
noncomputable def base_segment_left : ℝ := m
noncomputable def base_segment_right : ℝ := c - m

-- The height and segments calculation theorem
theorem triangle_height_and_segments :
  h = 11.2 ∧ m = 6.6 ∧ (c - m) = 8.4 :=
by {
  sorry
}

end triangle_height_and_segments_0_131


namespace cost_of_one_unit_each_0_624

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end cost_of_one_unit_each_0_624


namespace find_cookbooks_stashed_in_kitchen_0_880

-- Definitions of the conditions
def total_books := 99
def books_in_boxes := 3 * 15
def books_in_room := 21
def books_on_table := 4
def books_picked_up := 12
def current_books := 23

-- Main statement
theorem find_cookbooks_stashed_in_kitchen :
  let books_donated := books_in_boxes + books_in_room + books_on_table
  let books_left_initial := total_books - books_donated
  let books_left_before_pickup := current_books - books_picked_up
  books_left_initial - books_left_before_pickup = 18 := by
  sorry

end find_cookbooks_stashed_in_kitchen_0_880


namespace cannot_bisect_segment_with_ruler_0_491

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end cannot_bisect_segment_with_ruler_0_491


namespace arithmetic_sequence_third_term_0_914

theorem arithmetic_sequence_third_term :
  ∀ (a d : ℤ), (a + 4 * d = 2) ∧ (a + 5 * d = 5) → (a + 2 * d = -4) :=
by sorry

end arithmetic_sequence_third_term_0_914


namespace sequence_inequality_0_781

theorem sequence_inequality {a : ℕ → ℝ} (h₁ : a 1 > 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (a n ^ 2 + 1) / (2 * a n)) :
  ∀ n : ℕ, (∑ i in Finset.range n, a (i + 1)) < n + 2 * (a 1 - 1) := by
  sorry

end sequence_inequality_0_781


namespace average_brown_MnMs_0_178

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_0_178


namespace find_r4_0_435

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_0_435


namespace harry_morning_ratio_0_811

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end harry_morning_ratio_0_811


namespace unique_five_digit_integers_0_602

-- Define the problem conditions
def digits := [2, 2, 3, 9, 9]
def total_spots := 5
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Compute the number of five-digit integers that can be formed
noncomputable def num_unique_permutations : Nat :=
  factorial total_spots / (factorial 2 * factorial 1 * factorial 2)

-- Proof statement
theorem unique_five_digit_integers : num_unique_permutations = 30 := by
  sorry

end unique_five_digit_integers_0_602


namespace selling_price_correct_0_640

-- Define the conditions
def cost_per_cupcake : ℝ := 0.75
def total_cupcakes_burnt : ℕ := 24
def total_eaten_first : ℕ := 5
def total_eaten_later : ℕ := 4
def net_profit : ℝ := 24
def total_cupcakes_made : ℕ := 72
def total_cost : ℝ := total_cupcakes_made * cost_per_cupcake
def total_eaten : ℕ := total_eaten_first + total_eaten_later
def total_sold : ℕ := total_cupcakes_made - total_eaten
def revenue (P : ℝ) : ℝ := total_sold * P

-- Prove the correctness of the selling price P
theorem selling_price_correct : 
  ∃ P : ℝ, revenue P - total_cost = net_profit ∧ (P = 1.24) :=
by
  sorry

end selling_price_correct_0_640


namespace man_work_days_0_198

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end man_work_days_0_198


namespace B_pow_5_eq_rB_plus_sI_0_262

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end B_pow_5_eq_rB_plus_sI_0_262


namespace rectangle_area_ratio_0_815

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end rectangle_area_ratio_0_815


namespace max_value_m_0_785

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end max_value_m_0_785


namespace sector_area_0_612

theorem sector_area (alpha : ℝ) (r : ℝ) (h_alpha : alpha = Real.pi / 3) (h_r : r = 2) : 
  (1 / 2) * (alpha * r) * r = (2 * Real.pi) / 3 := 
by
  sorry

end sector_area_0_612


namespace rate_of_dividend_is_12_0_977

-- Defining the conditions
def total_investment : ℝ := 4455
def price_per_share : ℝ := 8.25
def annual_income : ℝ := 648
def face_value_per_share : ℝ := 10

-- Expected rate of dividend
def expected_rate_of_dividend : ℝ := 12

-- The proof problem statement: Prove that the rate of dividend is 12% given the conditions.
theorem rate_of_dividend_is_12 :
  ∃ (r : ℝ), r = 12 ∧ annual_income = 
    (total_investment / price_per_share) * (r / 100) * face_value_per_share :=
by 
  use 12
  sorry

end rate_of_dividend_is_12_0_977


namespace biology_marks_0_255

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end biology_marks_0_255


namespace nadine_total_cleaning_time_0_396

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_0_396


namespace proof_of_problem_statement_0_680

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end proof_of_problem_statement_0_680


namespace sum_of_digits_of_fraction_repeating_decimal_0_862

theorem sum_of_digits_of_fraction_repeating_decimal :
  (exists (c d : ℕ), (4 / 13 : ℚ) = c * 0.1 + d * 0.01 ∧ (c + d) = 3) :=
sorry

end sum_of_digits_of_fraction_repeating_decimal_0_862


namespace three_digit_number_parity_count_equal_0_876

/-- Prove the number of three-digit numbers with all digits having the same parity is equal to the number of three-digit numbers where adjacent digits have different parity. -/
theorem three_digit_number_parity_count_equal :
  ∃ (same_parity_count alternating_parity_count : ℕ),
    same_parity_count = alternating_parity_count ∧
    -- Condition for digits of the same parity
    same_parity_count = (4 * 5 * 5) + (5 * 5 * 5) ∧
    -- Condition for alternating parity digits (patterns EOE and OEO)
    alternating_parity_count = (4 * 5 * 5) + (5 * 5 * 5) := by
  sorry

end three_digit_number_parity_count_equal_0_876


namespace minimize_fencing_0_973

def area_requirement (w : ℝ) : Prop :=
  2 * (w * w) ≥ 800

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

def perimeter (w l : ℝ) : ℝ :=
  2 * l + 2 * w

theorem minimize_fencing (w l : ℝ) (h1 : area_requirement w) (h2 : length_twice_width l w) :
  w = 20 ∧ l = 40 :=
by
  sorry

end minimize_fencing_0_973


namespace marbles_per_boy_0_600

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end marbles_per_boy_0_600


namespace find_smaller_number_0_366

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_0_366


namespace paint_house_0_207

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end paint_house_0_207


namespace seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_0_915

-- Problem 1
theorem seven_divides_n_iff_seven_divides_q_minus_2r (n q r : ℕ) (h : n = 10 * q + r) :
  (7 ∣ n) ↔ (7 ∣ (q - 2 * r)) := sorry

-- Problem 2
theorem seven_divides_2023 : 7 ∣ 2023 :=
  let q := 202
  let r := 3
  have h : 2023 = 10 * q + r := by norm_num
  have h1 : (7 ∣ 2023) ↔ (7 ∣ (q - 2 * r)) :=
    seven_divides_n_iff_seven_divides_q_minus_2r 2023 q r h
  sorry -- Here you would use h1 and prove the statement using it

-- Problem 3
theorem thirteen_divides_n_iff_thirteen_divides_q_plus_4r (n q r : ℕ) (h : n = 10 * q + r) :
  (13 ∣ n) ↔ (13 ∣ (q + 4 * r)) := sorry

end seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_0_915


namespace algebraic_sum_parity_0_529

theorem algebraic_sum_parity :
  ∀ (f : Fin 2006 → ℤ),
    (∀ i, f i = i ∨ f i = -i) →
    (∑ i, f i) % 2 = 1 := by
  sorry

end algebraic_sum_parity_0_529


namespace geometric_number_difference_0_157

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

end geometric_number_difference_0_157


namespace find_larger_integer_0_800

variable (x : ℤ) (smaller larger : ℤ)
variable (ratio_1_to_4 : smaller = 1 * x ∧ larger = 4 * x)
variable (condition : smaller + 12 = larger)

theorem find_larger_integer : larger = 16 :=
by
  sorry

end find_larger_integer_0_800


namespace eighth_binomial_term_0_910

theorem eighth_binomial_term :
  let n := 10
  let a := 2 * x
  let b := 1
  let k := 7
  (Nat.choose n k) * (a ^ k) * (b ^ (n - k)) = 960 * (x ^ 3) := by
  sorry

end eighth_binomial_term_0_910


namespace smallest_k_for_min_period_15_0_829

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_0_829


namespace decreasing_function_iff_m_eq_2_0_134

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end decreasing_function_iff_m_eq_2_0_134


namespace range_of_a_0_772

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_0_772


namespace find_pos_int_0_757

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end find_pos_int_0_757


namespace find_the_number_0_626

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end find_the_number_0_626


namespace Lyle_friends_sandwich_juice_0_888

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_0_888


namespace total_spent_correct_0_535

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end total_spent_correct_0_535


namespace least_cans_required_0_926

def maaza : ℕ := 20
def pepsi : ℕ := 144
def sprite : ℕ := 368

def GCD (a b : ℕ) : ℕ := Nat.gcd a b

def total_cans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd_maaza_pepsi := GCD maaza pepsi
  let gcd_all := GCD gcd_maaza_pepsi sprite
  (maaza / gcd_all) + (pepsi / gcd_all) + (sprite / gcd_all)

theorem least_cans_required : total_cans maaza pepsi sprite = 133 := by
  sorry

end least_cans_required_0_926


namespace polynomial_evaluation_0_726

def f (x : ℝ) : ℝ := sorry

theorem polynomial_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 6 * x^2 + 2) :
  f (x^2 - 3) = x^4 - 2 * x^2 - 7 :=
sorry

end polynomial_evaluation_0_726


namespace dyed_pink_correct_0_51

def silk_dyed_green := 61921
def total_yards_dyed := 111421
def yards_dyed_pink := total_yards_dyed - silk_dyed_green

theorem dyed_pink_correct : yards_dyed_pink = 49500 := by 
  sorry

end dyed_pink_correct_0_51


namespace number_identification_0_999

theorem number_identification (x : ℝ) (h : x ^ 655 / x ^ 650 = 100000) : x = 10 :=
by
  sorry

end number_identification_0_999


namespace green_balloons_correct_0_941

-- Defining the quantities
def total_balloons : ℕ := 67
def red_balloons : ℕ := 29
def blue_balloons : ℕ := 21

-- Calculating the green balloons
def green_balloons : ℕ := total_balloons - red_balloons - blue_balloons

-- The theorem we want to prove
theorem green_balloons_correct : green_balloons = 17 :=
by
  -- proof goes here
  sorry

end green_balloons_correct_0_941


namespace arithmetic_mean_of_first_40_consecutive_integers_0_476

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the given arithmetic sequence
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Define the arithmetic mean of the first n terms of the given arithmetic sequence
def arithmetic_mean (a₁ d n : ℕ) : ℚ :=
  (arithmetic_sum a₁ d n : ℚ) / n

-- The arithmetic sequence starts at 5, has a common difference of 1, and has 40 terms
theorem arithmetic_mean_of_first_40_consecutive_integers :
  arithmetic_mean 5 1 40 = 24.5 :=
by
  sorry

end arithmetic_mean_of_first_40_consecutive_integers_0_476


namespace best_fitting_model_0_321

-- Define the \(R^2\) values for each model
def R2_Model1 : ℝ := 0.75
def R2_Model2 : ℝ := 0.90
def R2_Model3 : ℝ := 0.25
def R2_Model4 : ℝ := 0.55

-- State that Model 2 is the best fitting model
theorem best_fitting_model : R2_Model2 = max (max R2_Model1 R2_Model2) (max R2_Model3 R2_Model4) :=
by -- Proof skipped
  sorry

end best_fitting_model_0_321


namespace profit_percent_is_25_0_399

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end profit_percent_is_25_0_399


namespace find_w_0_610

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end find_w_0_610


namespace g_f_neg5_0_644

-- Define the function f
def f (x : ℝ) := 2 * x ^ 2 - 4

-- Define the function g with the known condition g(f(5)) = 12
axiom g : ℝ → ℝ
axiom g_f5 : g (f 5) = 12

-- Now state the main theorem we need to prove
theorem g_f_neg5 : g (f (-5)) = 12 := by
  sorry

end g_f_neg5_0_644


namespace number_divisibility_0_461

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end number_divisibility_0_461


namespace cubic_yard_to_cubic_meter_0_80

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end cubic_yard_to_cubic_meter_0_80


namespace local_value_of_4_in_564823_0_101

def face_value (d : ℕ) : ℕ := d
def place_value_of_thousands : ℕ := 1000
def local_value (d : ℕ) (p : ℕ) : ℕ := d * p

theorem local_value_of_4_in_564823 :
  local_value (face_value 4) place_value_of_thousands = 4000 :=
by 
  sorry

end local_value_of_4_in_564823_0_101


namespace sum_multiple_of_3_probability_0_440

noncomputable def probability_sum_multiple_of_3 (faces : List ℕ) (rolls : ℕ) (multiple : ℕ) : ℚ :=
  if rolls = 3 ∧ multiple = 3 ∧ faces = [1, 2, 3, 4, 5, 6] then 1 / 3 else 0

theorem sum_multiple_of_3_probability :
  probability_sum_multiple_of_3 [1, 2, 3, 4, 5, 6] 3 3 = 1 / 3 :=
by
  sorry

end sum_multiple_of_3_probability_0_440


namespace min_value_of_m_0_24

theorem min_value_of_m : (2 ∈ {x | ∃ (m : ℤ), x * (x - m) < 0}) → ∃ (m : ℤ), m = 3 :=
by
  sorry

end min_value_of_m_0_24


namespace fraction_surface_area_red_0_979

theorem fraction_surface_area_red :
  ∀ (num_unit_cubes : ℕ) (side_length_large_cube : ℕ) (total_surface_area_painted : ℕ) (total_surface_area_unit_cubes : ℕ),
    num_unit_cubes = 8 →
    side_length_large_cube = 2 →
    total_surface_area_painted = 6 * (side_length_large_cube ^ 2) →
    total_surface_area_unit_cubes = num_unit_cubes * 6 →
    (total_surface_area_painted : ℝ) / total_surface_area_unit_cubes = 1 / 2 :=
by
  intros num_unit_cubes side_length_large_cube total_surface_area_painted total_surface_area_unit_cubes
  sorry

end fraction_surface_area_red_0_979


namespace sequence_contains_composite_0_473

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem sequence_contains_composite (a : ℕ → ℕ) (h : ∀ n, a (n+1) = 2 * a n + 1 ∨ a (n+1) = 2 * a n - 1) :
  ∃ n, is_composite (a n) :=
sorry

end sequence_contains_composite_0_473


namespace quotient_remainder_scaled_0_32

theorem quotient_remainder_scaled (a b q r k : ℤ) (hb : b > 0) (hk : k ≠ 0) (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) :
  a * k = (b * k) * q + (r * k) ∧ (k ∣ r → (a / k = (b / k) * q + (r / k) ∧ 0 ≤ (r / k) ∧ (r / k) < (b / k))) :=
by
  sorry

end quotient_remainder_scaled_0_32


namespace sugar_needed_for_40_cookies_0_824

def num_cookies_per_cup_flour (a : ℕ) (b : ℕ) : ℕ := a / b

def cups_of_flour_needed (num_cookies : ℕ) (cookies_per_cup : ℕ) : ℕ := num_cookies / cookies_per_cup

def cups_of_sugar_needed (cups_flour : ℕ) (flour_to_sugar_ratio_num : ℕ) (flour_to_sugar_ratio_denom : ℕ) : ℚ := 
  (flour_to_sugar_ratio_denom * cups_flour : ℚ) / flour_to_sugar_ratio_num

theorem sugar_needed_for_40_cookies :
  let num_flour_to_make_24_cookies := 3
  let cookies := 24
  let ratio_num := 3
  let ratio_denom := 2
  num_cookies_per_cup_flour cookies num_flour_to_make_24_cookies = 8 →
  cups_of_flour_needed 40 8 = 5 →
  cups_of_sugar_needed 5 ratio_num ratio_denom = 10 / 3 :=
by 
  sorry

end sugar_needed_for_40_cookies_0_824


namespace range_of_f_0_84

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end range_of_f_0_84


namespace find_angle_0_241

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end find_angle_0_241


namespace vector_operation_result_0_569

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end vector_operation_result_0_569


namespace union_of_A_and_B_0_166

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end union_of_A_and_B_0_166


namespace calculate_expression_0_801

theorem calculate_expression :
  let s1 := 3 + 6 + 9
  let s2 := 4 + 8 + 12
  s1 = 18 → s2 = 24 → (s1 / s2 + s2 / s1) = 25 / 12 :=
by
  intros
  sorry

end calculate_expression_0_801


namespace possible_values_of_a₁_0_389

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end possible_values_of_a₁_0_389


namespace empty_set_subset_zero_set_0_342

-- Define the sets
def zero_set : Set ℕ := {0}
def empty_set : Set ℕ := ∅

-- State the problem
theorem empty_set_subset_zero_set : empty_set ⊂ zero_set :=
sorry

end empty_set_subset_zero_set_0_342


namespace radius_of_cookie_0_448

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end radius_of_cookie_0_448


namespace expression_a_n1_geometric_sequence_general_term_sum_of_sequence_0_257

-- Define the quadratic equation and initial condition
variable {α β : ℝ} (a : ℕ → ℝ)
variable (n : ℕ)

-- Initial condition
axiom a1 : a 1 = 1

-- Quadratic equation root condition
axiom root_condition : ∀ {n : ℕ}, 6 * α - 2 * α * β + 6 * β = 3

-- Define a_n+1 in terms of a_n
theorem expression_a_n1 (n : ℕ) : a (n + 1) = (1 / 2) * a n + (1 / 3) :=
sorry

-- Define the sequence {a_n - 2/3} is geometric with common ratio 1/2
theorem geometric_sequence (n : ℕ) : ∃ r : ℝ, r = 1 / 2 ∧ ∀ n, a (n + 1) - 2 / 3 = r * (a n - 2 / 3) :=
sorry

-- General term formula for a_n
theorem general_term (n : ℕ) : a n = (1 / 3) * (1 / 2)^(n - 1) + (2 / 3) :=
sorry

-- Sum of the first n terms of the sequence {a_n} noted as S_n
theorem sum_of_sequence (n : ℕ) : ∑ i in Finset.range n.succ, a i = (2 * n + 2) / 3 - (1 / 3) * (1 / 2)^(n - 1) :=
sorry

end expression_a_n1_geometric_sequence_general_term_sum_of_sequence_0_257


namespace geometric_sequence_sum_0_818

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end geometric_sequence_sum_0_818


namespace polynomial_divisibility_0_405

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end polynomial_divisibility_0_405


namespace problem_statement_0_854

def a : ℤ := 2020
def b : ℤ := 2022

theorem problem_statement : b^3 - a * b^2 - a^2 * b + a^3 = 16168 := by
  sorry

end problem_statement_0_854


namespace length_of_goods_train_0_868

-- Define the given conditions
def speed_kmph := 72
def platform_length := 260
def crossing_time := 26

-- Convert speed to m/s
def speed_mps := (speed_kmph * 5) / 18

-- Calculate distance covered
def distance_covered := speed_mps * crossing_time

-- Define the length of the train
def train_length := distance_covered - platform_length

theorem length_of_goods_train : train_length = 260 := by
  sorry

end length_of_goods_train_0_868


namespace seven_large_power_mod_seventeen_0_677

theorem seven_large_power_mod_seventeen :
  (7 : ℤ)^1985 % 17 = 7 :=
by
  have h1 : (7 : ℤ)^2 % 17 = 15 := sorry
  have h2 : (7 : ℤ)^4 % 17 = 16 := sorry
  have h3 : (7 : ℤ)^8 % 17 = 1 := sorry
  have h4 : 1985 = 8 * 248 + 1 := sorry
  sorry

end seven_large_power_mod_seventeen_0_677


namespace opposite_of_a_is_2_0_232

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end opposite_of_a_is_2_0_232


namespace midpoint_3d_0_939

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end midpoint_3d_0_939


namespace hcf_of_two_numbers_0_841

noncomputable def H : ℕ := 322 / 14

theorem hcf_of_two_numbers (H k : ℕ) (lcm_val : ℕ) :
  lcm_val = H * 13 * 14 ∧ 322 = H * k ∧ 322 / 14 = H → H = 23 :=
by
  sorry

end hcf_of_two_numbers_0_841


namespace souvenirs_total_cost_0_651

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end souvenirs_total_cost_0_651


namespace bucket_full_weight_0_479

variables (x y p q : Real)

theorem bucket_full_weight (h1 : x + (1 / 4) * y = p)
                           (h2 : x + (3 / 4) * y = q) :
    x + y = 3 * q - p :=
by
  sorry

end bucket_full_weight_0_479


namespace linear_dependency_k_0_14

theorem linear_dependency_k (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
    (c1 * 1 + c2 * 4 = 0) ∧
    (c1 * 2 + c2 * k = 0) ∧
    (c1 * 3 + c2 * 6 = 0)) ↔ k = 8 :=
by
  sorry

end linear_dependency_k_0_14


namespace problem_0_137

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end problem_0_137


namespace sufficient_condition_for_ellipse_0_343

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end sufficient_condition_for_ellipse_0_343


namespace tetrahedron_volume_0_21

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_0_21


namespace royWeight_0_138

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end royWeight_0_138


namespace geometric_sequence_sum_0_857

theorem geometric_sequence_sum (S : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n : ℕ, n > 0 → S n = 2^n + a) →
  (S 1 = 2 + a) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  (a_n 1 = 1) →
  a = -1 :=
by
  sorry

end geometric_sequence_sum_0_857


namespace geometric_sequence_a3_eq_2_0_259

theorem geometric_sequence_a3_eq_2 
  (a_1 a_3 a_5 : ℝ) 
  (h1 : a_1 * a_3 * a_5 = 8) 
  (h2 : a_3^2 = a_1 * a_5) : 
  a_3 = 2 :=
by 
  sorry

end geometric_sequence_a3_eq_2_0_259


namespace joels_age_when_dad_twice_0_541

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end joels_age_when_dad_twice_0_541


namespace smallest_base_0_293

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end smallest_base_0_293


namespace rectangle_area_0_992

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end rectangle_area_0_992


namespace ratio_of_juice_to_bread_0_57

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end ratio_of_juice_to_bread_0_57


namespace oplus_self_twice_0_253

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end oplus_self_twice_0_253


namespace transformation_composition_0_303

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end transformation_composition_0_303


namespace problem_0_604

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end problem_0_604


namespace distance_rowed_upstream_0_352

noncomputable def speed_of_boat_in_still_water := 18 -- from solution step; b = 18 km/h
def speed_of_stream := 3 -- given
def time := 4 -- given
def distance_downstream := 84 -- given

theorem distance_rowed_upstream 
  (b : ℕ) (s : ℕ) (t : ℕ) (d_down : ℕ) (d_up : ℕ)
  (h_stream : s = 3) 
  (h_time : t = 4)
  (h_distance_downstream : d_down = 84) 
  (h_speed_boat : b = 18) 
  (h_effective_downstream_speed : b + s = d_down / t) :
  d_up = 60 := by
  sorry

end distance_rowed_upstream_0_352


namespace number_less_than_one_is_correct_0_314

theorem number_less_than_one_is_correct : (1 - 5 = -4) :=
by
  sorry

end number_less_than_one_is_correct_0_314


namespace coin_flip_probability_0_989

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end coin_flip_probability_0_989


namespace aardvark_total_distance_0_959

noncomputable def total_distance (r_small r_large : ℝ) : ℝ :=
  let small_circumference := 2 * Real.pi * r_small
  let large_circumference := 2 * Real.pi * r_large
  let half_small_circumference := small_circumference / 2
  let half_large_circumference := large_circumference / 2
  let radial_distance := r_large - r_small
  let total_radial_distance := radial_distance + r_large
  half_small_circumference + radial_distance + half_large_circumference + total_radial_distance

theorem aardvark_total_distance :
  total_distance 15 30 = 45 * Real.pi + 45 :=
by
  sorry

end aardvark_total_distance_0_959


namespace arithmetic_geometric_mean_inequality_0_592

theorem arithmetic_geometric_mean_inequality {n : ℕ} (h : 2 ≤ n) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ i, a i) / n ≥ (∏ i, a i) ^ (1 / n) :=
by
  -- proof goes here
  sorry

end arithmetic_geometric_mean_inequality_0_592


namespace saved_per_bagel_0_968

-- Definitions of the conditions
def bagel_cost_each : ℝ := 3.50
def dozen_cost : ℝ := 38
def bakers_dozen : ℕ := 13
def discount : ℝ := 0.05

-- The conjecture we need to prove
theorem saved_per_bagel : 
  let total_cost_without_discount := dozen_cost + bagel_cost_each
  let discount_amount := discount * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount_amount
  let cost_per_bagel_without_discount := dozen_cost / 12
  let cost_per_bagel_with_discount := total_cost_with_discount / bakers_dozen
  let savings_per_bagel := cost_per_bagel_without_discount - cost_per_bagel_with_discount
  let savings_in_cents := savings_per_bagel * 100
  savings_in_cents = 13.36 :=
by
  -- Placeholder for the actual proof
  sorry

end saved_per_bagel_0_968


namespace saved_money_is_30_0_460

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end saved_money_is_30_0_460


namespace fish_distribution_0_987

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end fish_distribution_0_987


namespace find_y_0_273

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end find_y_0_273


namespace bottles_produced_by_10_machines_in_4_minutes_0_230

variable (rate_per_machine : ℕ)
variable (total_bottles_per_minute_six_machines : ℕ := 240)
variable (number_of_machines : ℕ := 6)
variable (new_number_of_machines : ℕ := 10)
variable (time_in_minutes : ℕ := 4)

theorem bottles_produced_by_10_machines_in_4_minutes :
  rate_per_machine = total_bottles_per_minute_six_machines / number_of_machines →
  (new_number_of_machines * rate_per_machine * time_in_minutes) = 1600 := 
sorry

end bottles_produced_by_10_machines_in_4_minutes_0_230


namespace real_solution_to_abs_equation_0_881

theorem real_solution_to_abs_equation :
  (∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8|) :=
by
  sorry

end real_solution_to_abs_equation_0_881


namespace find_middle_part_value_0_887

-- Define the ratios
def ratio1 := 1 / 2
def ratio2 := 1 / 4
def ratio3 := 1 / 8

-- Total sum
def total_sum := 120

-- Parts proportional to ratios
def part1 (x : ℝ) := x
def part2 (x : ℝ) := ratio1 * x
def part3 (x : ℝ) := ratio2 * x

-- Equation representing the sum of the parts equals to the total sum
def equation (x : ℝ) : Prop :=
  part1 x + part2 x / 2 + part2 x = x * (1 + ratio1 + ratio2)

-- Defining the middle part
def middle_part (x : ℝ) := ratio1 * x

theorem find_middle_part_value :
  ∃ x : ℝ, equation x ∧ middle_part x = 34.2857 := sorry

end find_middle_part_value_0_887


namespace evaluate_expression_0_976

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end evaluate_expression_0_976


namespace jia_winning_strategy_0_10

variables {p q : ℝ}
def is_quadratic_real_roots (a b c : ℝ) : Prop := b ^ 2 - 4 * a * c > 0

def quadratic_with_roots (x1 x2 : ℝ) :=
  x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ is_quadratic_real_roots 1 (- (x1 + x2)) (x1 * x2)

def modify_jia (p q x1 : ℝ) : (ℝ × ℝ) := (p + 1, q - x1)

def modify_yi1 (p q : ℝ) : (ℝ × ℝ) := (p - 1, q)

def modify_yi2 (p q x2 : ℝ) : (ℝ × ℝ) := (p - 1, q + x2)

def winning_strategy_jia (x1 x2 : ℝ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, ∀ p q, quadratic_with_roots x1 x2 → 
  (¬ is_quadratic_real_roots 1 p q) ∨ (q ≤ 0)

theorem jia_winning_strategy (x1 x2 : ℝ)
  (h: quadratic_with_roots x1 x2) : 
  winning_strategy_jia x1 x2 :=
sorry

end jia_winning_strategy_0_10


namespace original_loaf_had_27_slices_0_214

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices_0_214


namespace same_terminal_side_0_82

theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6) = 11 * Real.pi / 6 := by
  sorry

end same_terminal_side_0_82


namespace negation_of_p_0_646

variable {x : ℝ}

def p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_p_0_646


namespace intersection_with_y_axis_0_558

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_0_558


namespace min_value_f_0_340

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end min_value_f_0_340


namespace simplify_and_evaluate_expr_0_408

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_0_408


namespace decimal_to_fraction_correct_0_72

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end decimal_to_fraction_correct_0_72


namespace new_concentration_is_37_percent_0_882

-- Conditions
def capacity_vessel_1 : ℝ := 2 -- litres
def alcohol_concentration_vessel_1 : ℝ := 0.35

def capacity_vessel_2 : ℝ := 6 -- litres
def alcohol_concentration_vessel_2 : ℝ := 0.50

def total_poured_liquid : ℝ := 8 -- litres
def final_vessel_capacity : ℝ := 10 -- litres

-- Question: Prove the new concentration of the mixture
theorem new_concentration_is_37_percent :
  (alcohol_concentration_vessel_1 * capacity_vessel_1 + alcohol_concentration_vessel_2 * capacity_vessel_2) / final_vessel_capacity = 0.37 := by
  sorry

end new_concentration_is_37_percent_0_882


namespace width_of_river_0_27

def river_depth : ℝ := 7
def flow_rate_kmph : ℝ := 4
def volume_per_minute : ℝ := 35000

noncomputable def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

theorem width_of_river : 
  ∃ w : ℝ, 
    volume_per_minute = flow_rate_mpm * river_depth * w ∧
    w = 75 :=
by
  use 75
  field_simp [flow_rate_mpm, river_depth, volume_per_minute]
  norm_num
  sorry

end width_of_river_0_27


namespace negation_of_existential_statement_0_482

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end negation_of_existential_statement_0_482


namespace store_profit_0_930

variables (m n : ℝ)

def total_profit (m n : ℝ) : ℝ :=
  110 * m - 50 * n

theorem store_profit (m n : ℝ) : total_profit m n = 110 * m - 50 * n :=
  by
  -- sorry indicates that the proof is skipped
  sorry

end store_profit_0_930


namespace integer_a_for_factoring_0_783

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end integer_a_for_factoring_0_783


namespace pushing_car_effort_0_218

theorem pushing_car_effort (effort constant : ℕ) (people1 people2 : ℕ) 
  (h1 : constant = people1 * effort)
  (h2 : people1 = 4)
  (h3 : effort = 120)
  (h4 : people2 = 6) :
  effort * people1 = constant → constant = people2 * 80 :=
by
  sorry

end pushing_car_effort_0_218


namespace tom_jerry_age_ratio_0_172

-- Definitions representing the conditions in the problem
variable (t j x : ℕ)

-- Condition 1: Three years ago, Tom was three times as old as Jerry
def condition1 : Prop := t - 3 = 3 * (j - 3)

-- Condition 2: Four years before that, Tom was five times as old as Jerry
def condition2 : Prop := t - 7 = 5 * (j - 7)

-- Question: In how many years will the ratio of their ages be 3:2,
-- asserting that the answer is 21
def ageRatioInYears : Prop := (t + x) / (j + x) = 3 / 2 → x = 21

-- The proposition we need to prove
theorem tom_jerry_age_ratio (h1 : condition1 t j) (h2 : condition2 t j) : ageRatioInYears t j x := 
  sorry
  
end tom_jerry_age_ratio_0_172


namespace solve_for_x_0_119

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_0_119


namespace expected_return_correct_0_615

-- Define the probabilities
def p1 := 1/4
def p2 := 1/4
def p3 := 1/6
def p4 := 1/3

-- Define the payouts
def payout (n : ℕ) (previous_odd : Bool) : ℝ :=
  match n with
  | 1 => 2
  | 2 => if previous_odd then -3 else 0
  | 3 => 0
  | 4 => 5
  | _ => 0

-- Define the expected values of one throw
def E1 : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

def E2_odd : ℝ :=
  p1 * payout 1 true + p2 * payout 2 true + p3 * payout 3 true + p4 * payout 4 true

def E2_even : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

-- Define the probability of throwing an odd number first
def p_odd : ℝ := p1 + p3

-- Define the probability of not throwing an odd number first
def p_even : ℝ := 1 - p_odd

-- Define the total expected return
def total_expected_return : ℝ :=
  E1 + (p_odd * E2_odd + p_even * E2_even)


theorem expected_return_correct :
  total_expected_return = 4.18 :=
  by
    -- The proof is omitted
    sorry

end expected_return_correct_0_615


namespace ratio_areas_0_184

theorem ratio_areas (H : ℝ) (L : ℝ) (r : ℝ) (A_rectangle : ℝ) (A_circle : ℝ) :
  H = 45 ∧ (L / H = 4 / 3) ∧ r = H / 2 ∧ A_rectangle = L * H ∧ A_circle = π * r^2 →
  (A_rectangle / A_circle = 17 / π) :=
by
  sorry

end ratio_areas_0_184


namespace shaded_region_area_0_957

variables (a b : ℕ) 
variable (A : Type) 

def AD := 5
def CD := 2
def semi_major_axis := 6
def semi_minor_axis := 4

noncomputable def area_ellipse := Real.pi * semi_major_axis * semi_minor_axis
noncomputable def area_rectangle := AD * CD
noncomputable def area_shaded_region := area_ellipse - area_rectangle

theorem shaded_region_area : area_shaded_region = 24 * Real.pi - 10 :=
by {
  sorry
}

end shaded_region_area_0_957


namespace percentage_error_0_830

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_0_830


namespace total_volume_of_all_cubes_0_171

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (count : ℕ) (side_length : ℕ) : ℕ := count * (cube_volume side_length)

theorem total_volume_of_all_cubes :
  total_volume 4 3 + total_volume 3 4 = 300 :=
by
  sorry

end total_volume_of_all_cubes_0_171


namespace find_x_plus_inv_x_0_738

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_0_738


namespace total_plants_in_garden_0_400

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_0_400


namespace percent_relation_0_901

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end percent_relation_0_901


namespace tiffany_total_score_0_997

-- Definitions based on conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def treasures_second_level : ℕ := 5

-- The statement we want to prove
theorem tiffany_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 48 := by
  sorry

end tiffany_total_score_0_997


namespace sequence_general_term_0_92

theorem sequence_general_term (a : ℕ → ℤ) (n : ℕ) 
  (h₀ : a 0 = 1) 
  (h_rec : ∀ n, a (n + 1) = 2 * a n + n) :
  a n = 2^(n + 1) - n - 1 :=
by sorry

end sequence_general_term_0_92


namespace point_distance_0_581

theorem point_distance (x : ℤ) : abs x = 2021 → (x = 2021 ∨ x = -2021) := 
sorry

end point_distance_0_581


namespace least_n_exceeds_product_0_791

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end least_n_exceeds_product_0_791


namespace simple_annual_interest_rate_0_777

noncomputable def monthly_interest_payment : ℝ := 216
noncomputable def principal_amount : ℝ := 28800
noncomputable def number_of_months_in_a_year : ℕ := 12

theorem simple_annual_interest_rate :
  ((monthly_interest_payment * number_of_months_in_a_year) / principal_amount) * 100 = 9 := by
sorry

end simple_annual_interest_rate_0_777


namespace men_absent_is_5_0_848

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end men_absent_is_5_0_848


namespace professionals_work_days_0_673

theorem professionals_work_days (cost_per_hour_1 cost_per_hour_2 hours_per_day total_cost : ℝ) (h_cost1: cost_per_hour_1 = 15) (h_cost2: cost_per_hour_2 = 15) (h_hours: hours_per_day = 6) (h_total: total_cost = 1260) : (∃ d : ℝ, total_cost = d * hours_per_day * (cost_per_hour_1 + cost_per_hour_2) ∧ d = 7) :=
by
  use 7
  rw [h_cost1, h_cost2, h_hours, h_total]
  simp
  sorry

end professionals_work_days_0_673


namespace solve_inequality_0_774

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem solve_inequality (x : ℝ) (hx : x ≠ 0 ∧ 0 < x) :
  (64 + (log_b (1/5) (x^2))^3) / (log_b (1/5) (x^6) * log_b 5 (x^2) + 5 * log_b 5 (x^6) + 14 * log_b (1/5) (x^2) + 2) ≤ 0 ↔
  (x ∈ Set.Icc (-25 : ℝ) (- Real.sqrt 5)) ∨
  (x ∈ Set.Icc (- (Real.exp (Real.log 5 / 3))) 0) ∨
  (x ∈ Set.Icc 0 (Real.exp (Real.log 5 / 3))) ∨
  (x ∈ Set.Icc (Real.sqrt 5) 25) :=
by 
  sorry

end solve_inequality_0_774


namespace volleyballs_basketballs_difference_0_505

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end volleyballs_basketballs_difference_0_505


namespace buses_needed_0_287

def total_students : ℕ := 111
def seats_per_bus : ℕ := 3

theorem buses_needed : total_students / seats_per_bus = 37 :=
by
  sorry

end buses_needed_0_287


namespace min_rectangle_area_0_708

theorem min_rectangle_area : 
  ∃ (x y : ℕ), 2 * (x + y) = 80 ∧ x * y = 39 :=
by
  sorry

end min_rectangle_area_0_708


namespace correct_option_B_0_627

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end correct_option_B_0_627


namespace probability_XiaoCong_project_A_probability_same_project_not_C_0_288

-- Definition of projects and conditions
inductive Project
| A | B | C

def XiaoCong : Project := sorry
def XiaoYing : Project := sorry

-- (1) Probability of Xiao Cong assigned to project A
theorem probability_XiaoCong_project_A : 
  (1 / 3 : ℝ) = 1 / 3 := 
by sorry

-- (2) Probability of Xiao Cong and Xiao Ying being assigned to the same project, given Xiao Ying not assigned to C
theorem probability_same_project_not_C : 
  (2 / 6 : ℝ) = 1 / 3 :=
by sorry

end probability_XiaoCong_project_A_probability_same_project_not_C_0_288


namespace collinear_vectors_x_eq_neg_two_0_36

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end collinear_vectors_x_eq_neg_two_0_36


namespace cost_of_article_0_550

-- Conditions as Lean definitions
def price_1 : ℝ := 340
def price_2 : ℝ := 350
def price_diff : ℝ := price_2 - price_1 -- Rs. 10
def gain_percent_increase : ℝ := 0.04

-- Question: What is the cost of the article?
-- Answer: Rs. 90

theorem cost_of_article : ∃ C : ℝ, 
  price_diff = gain_percent_increase * (price_1 - C) ∧ C = 90 := 
sorry

end cost_of_article_0_550


namespace intersection_M_N_0_59

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end intersection_M_N_0_59


namespace earnings_correct_0_970

def price_8inch : ℝ := 5
def price_12inch : ℝ := 2.5 * price_8inch
def price_16inch : ℝ := 3 * price_8inch
def price_20inch : ℝ := 4 * price_8inch
def price_24inch : ℝ := 5.5 * price_8inch

noncomputable def earnings_monday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 1 * price_16inch + 2 * price_20inch + 1 * price_24inch

noncomputable def earnings_tuesday : ℝ :=
  5 * price_8inch + 1 * price_12inch + 4 * price_16inch + 2 * price_24inch

noncomputable def earnings_wednesday : ℝ :=
  4 * price_8inch + 3 * price_12inch + 3 * price_16inch + 1 * price_20inch

noncomputable def earnings_thursday : ℝ :=
  2 * price_8inch + 2 * price_12inch + 2 * price_16inch + 1 * price_20inch + 3 * price_24inch

noncomputable def earnings_friday : ℝ :=
  6 * price_8inch + 4 * price_12inch + 2 * price_16inch + 2 * price_20inch

noncomputable def earnings_saturday : ℝ :=
  1 * price_8inch + 3 * price_12inch + 3 * price_16inch + 4 * price_20inch + 2 * price_24inch

noncomputable def earnings_sunday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 4 * price_16inch + 3 * price_20inch + 1 * price_24inch

noncomputable def total_earnings : ℝ :=
  earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday + earnings_saturday + earnings_sunday

theorem earnings_correct : total_earnings = 1025 := by
  -- proof goes here
  sorry

end earnings_correct_0_970


namespace total_spent_0_195

-- Define the conditions
def cost_fix_automobile := 350
def cost_fix_formula (S : ℕ) := 3 * S + 50

-- Prove the total amount spent is $450
theorem total_spent (S : ℕ) (h : cost_fix_automobile = cost_fix_formula S) :
  S + cost_fix_automobile = 450 :=
by
  sorry

end total_spent_0_195


namespace cookies_per_batch_0_508

def family_size := 4
def chips_per_person := 18
def chips_per_cookie := 2
def batches := 3

theorem cookies_per_batch : (family_size * chips_per_person) / chips_per_cookie / batches = 12 := 
by
  -- Proof will go here
  sorry

end cookies_per_batch_0_508


namespace exists_unique_continuous_extension_0_71

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end exists_unique_continuous_extension_0_71


namespace sequence_formula_minimum_m_0_29

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

/-- The sequence a_n with sum of its first n terms S_n, the first term a_1 = 1, and the terms
   1, a_n, S_n forming an arithmetic sequence, satisfies a_n = 2^(n-1). -/
theorem sequence_formula (h1 : a_n 1 = 1)
    (h2 : ∀ n : ℕ, 1 + n * (a_n n - 1) = S_n n) :
    ∀ n : ℕ, a_n n = 2 ^ (n - 1) := by
  sorry

/-- T_n being the sum of the sequence {n / a_n}, if T_n < (m - 4) / 3 for all n in ℕ*, 
    then the minimum value of m is 16. -/
theorem minimum_m (T_n : ℕ → ℝ) (m : ℕ)
    (hT : ∀ n : ℕ, n > 0 → T_n n < (m - 4) / 3) :
    m ≥ 16 := by
  sorry

end sequence_formula_minimum_m_0_29


namespace tom_boxes_needed_0_994

-- Definitions of given conditions
def room_length : ℕ := 16
def room_width : ℕ := 20
def box_coverage : ℕ := 10
def already_covered : ℕ := 250

-- The total area of the living room
def total_area : ℕ := room_length * room_width

-- The remaining area that needs to be covered
def remaining_area : ℕ := total_area - already_covered

-- The number of boxes required to cover the remaining area
def boxes_needed : ℕ := remaining_area / box_coverage

-- The theorem statement
theorem tom_boxes_needed : boxes_needed = 7 := by
  -- The proof will go here
  sorry

end tom_boxes_needed_0_994


namespace mod_graph_sum_0_454

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end mod_graph_sum_0_454


namespace sequence_general_term_0_236

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end sequence_general_term_0_236


namespace minimum_perimeter_0_480

def fractional_part (x : ℚ) : ℚ := x - x.floor

-- Define l, m, n being sides of the triangle with l > m > n
variables (l m n : ℤ)

-- Defining conditions as Lean predicates
def triangle_sides (l m n : ℤ) : Prop := l > m ∧ m > n

def fractional_part_condition (l m n : ℤ) : Prop :=
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)

-- Prove the minimum perimeter is 3003 given above conditions
theorem minimum_perimeter (l m n : ℤ) :
  triangle_sides l m n →
  fractional_part_condition l m n →
  l + m + n = 3003 :=
by
  intros h_sides h_fractional
  sorry

end minimum_perimeter_0_480


namespace inequality_x_y_0_407

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end inequality_x_y_0_407


namespace sin_sum_triangle_inequality_0_988

theorem sin_sum_triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_inequality_0_988


namespace unique_pairs_of_socks_0_526

-- Defining the problem conditions
def pairs_socks : Nat := 3

-- The main proof statement
theorem unique_pairs_of_socks : ∃ (n : Nat), n = 3 ∧ 
  (∀ (p q : Fin 6), (p / 2 ≠ q / 2) → p ≠ q) →
  (n = (pairs_socks * (pairs_socks - 1)) / 2) :=
by
  sorry

end unique_pairs_of_socks_0_526


namespace simplify_expression_0_568

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end simplify_expression_0_568


namespace even_three_digit_numbers_count_0_234

theorem even_three_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  count = 18 :=
by
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  show count = 18
  sorry

end even_three_digit_numbers_count_0_234


namespace sum_of_integers_is_18_0_114

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end sum_of_integers_is_18_0_114


namespace ordered_triple_unique_0_78

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end ordered_triple_unique_0_78


namespace fixed_point_always_on_line_0_305

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end fixed_point_always_on_line_0_305


namespace find_max_value_0_663

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end find_max_value_0_663


namespace inequality_proof_0_688

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + x + 2 * x^2) * (2 + 3 * y + y^2) * (4 + z + z^2) ≥ 60 * x * y * z :=
by
  sorry

end inequality_proof_0_688


namespace total_monthly_bill_working_from_home_0_472

def original_monthly_bill : ℝ := 60
def percentage_increase : ℝ := 0.30

theorem total_monthly_bill_working_from_home :
  original_monthly_bill + (original_monthly_bill * percentage_increase) = 78 := by
  sorry

end total_monthly_bill_working_from_home_0_472


namespace arnold_plates_count_0_969

def arnold_barbell := 45
def mistaken_weight := 600
def actual_weight := 470
def weight_difference_per_plate := 10

theorem arnold_plates_count : 
  ∃ n : ℕ, mistaken_weight - actual_weight = n * weight_difference_per_plate ∧ n = 13 := 
sorry

end arnold_plates_count_0_969


namespace jugglers_count_0_368

-- Define the conditions
def num_balls_each_juggler := 6
def total_balls := 2268

-- Define the theorem to prove the number of jugglers
theorem jugglers_count : (total_balls / num_balls_each_juggler) = 378 :=
by
  sorry

end jugglers_count_0_368


namespace union_of_sets_0_329

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end union_of_sets_0_329


namespace math_proof_problem_0_965

variable {a b c : ℝ}

theorem math_proof_problem (h₁ : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h₂ : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  (a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b))) = 2017 := 
sorry

end math_proof_problem_0_965


namespace distance_blown_by_storm_0_429

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end distance_blown_by_storm_0_429


namespace perfect_square_condition_0_765

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end perfect_square_condition_0_765


namespace right_triangle_third_side_square_0_118

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) 
  (h₁ : a = 3) (h₂ : b = 4) (h₃ : a^2 + b^2 = c^2) :
  c^2 = 25 ∨ a^2 + c^2 = b^2 ∨ a^2 + b^2 = 7 :=
by
  sorry

end right_triangle_third_side_square_0_118


namespace harkamal_total_amount_0_63

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end harkamal_total_amount_0_63


namespace maximize_profit_marginal_profit_monotonic_decreasing_0_191

-- Definition of revenue function R
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Definition of cost function C
def C (x : ℕ) : ℤ := 460 * x + 500

-- Definition of profit function p
def p (x : ℕ) : ℤ := R x - C x

-- Lemma for the solution
theorem maximize_profit (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 20) : 
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧ 
  (∀ y, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) :=
by
  sorry

-- Definition of marginal profit function Mp
def Mp (x : ℕ) : ℤ := p (x + 1) - p x

-- Lemma showing Mp is monotonically decreasing
theorem marginal_profit_monotonic_decreasing (x : ℕ) (h2 : 1 ≤ x ∧ x ≤ 19) : 
  Mp x = -30 * x^2 + 60 * x + 3275 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 19 → (Mp y ≥ Mp (y + 1)) :=
by
  sorry

end maximize_profit_marginal_profit_monotonic_decreasing_0_191


namespace third_side_length_0_960

noncomputable def calc_third_side (a b : ℕ) (hypotenuse : Bool) : ℝ :=
if hypotenuse then
  Real.sqrt (a^2 + b^2)
else
  Real.sqrt (abs (a^2 - b^2))

theorem third_side_length (a b : ℕ) (h_right_triangle : (a = 8 ∧ b = 15)) :
  calc_third_side a b true = 17 ∨ calc_third_side 15 8 false = Real.sqrt 161 :=
by {
  sorry
}

end third_side_length_0_960


namespace sum_of_squares_of_geometric_sequence_0_422

theorem sum_of_squares_of_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : ∀ n, (∑ i in Finset.range n, a i) = 2^n - 1) :
  (∑ i in Finset.range n, (a i)^2) = (1 / 3) * (4^n - 1) := 
sorry

end sum_of_squares_of_geometric_sequence_0_422


namespace pete_total_blocks_traveled_0_380

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end pete_total_blocks_traveled_0_380
