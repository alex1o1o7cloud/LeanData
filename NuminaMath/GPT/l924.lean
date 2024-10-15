import Mathlib

namespace NUMINAMATH_GPT_volunteer_selection_count_l924_92464

open Nat

theorem volunteer_selection_count :
  let boys : ℕ := 5
  let girls : ℕ := 2
  let total_ways := choose girls 1 * choose boys 2 + choose girls 2 * choose boys 1
  total_ways = 25 :=
by
  sorry

end NUMINAMATH_GPT_volunteer_selection_count_l924_92464


namespace NUMINAMATH_GPT_team_overall_progress_is_89_l924_92460

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

def overall_progress (changes : List Int) : Int :=
  changes.sum

theorem team_overall_progress_is_89 :
  overall_progress yard_changes = 89 :=
by
  sorry

end NUMINAMATH_GPT_team_overall_progress_is_89_l924_92460


namespace NUMINAMATH_GPT_janice_total_hours_worked_l924_92436

-- Declare the conditions as definitions
def hourly_rate_first_40_hours : ℝ := 10
def hourly_rate_overtime : ℝ := 15
def first_40_hours : ℕ := 40
def total_pay : ℝ := 700

-- Define the main theorem
theorem janice_total_hours_worked (H : ℕ) (O : ℕ) : 
  H = first_40_hours + O ∧ (hourly_rate_first_40_hours * first_40_hours + hourly_rate_overtime * O = total_pay) → H = 60 :=
by
  sorry

end NUMINAMATH_GPT_janice_total_hours_worked_l924_92436


namespace NUMINAMATH_GPT_exists_divisible_by_2021_l924_92451

def concat_numbers (n m : ℕ) : ℕ :=
  -- function to concatenate numbers from n to m
  sorry

theorem exists_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concat_numbers n m :=
by
  sorry

end NUMINAMATH_GPT_exists_divisible_by_2021_l924_92451


namespace NUMINAMATH_GPT_mr_brown_no_calls_in_2020_l924_92454

noncomputable def number_of_days_with_no_calls (total_days : ℕ) (calls_niece1 : ℕ) (calls_niece2 : ℕ) (calls_niece3 : ℕ) : ℕ := 
  let calls_2 := total_days / calls_niece1
  let calls_3 := total_days / calls_niece2
  let calls_4 := total_days / calls_niece3
  let calls_6 := total_days / (Nat.lcm calls_niece1 calls_niece2)
  let calls_12_ := total_days / (Nat.lcm calls_niece1 (Nat.lcm calls_niece2 calls_niece3))
  total_days - (calls_2 + calls_3 + calls_4 - calls_6 - calls_4 - (total_days / calls_niece2 / 4) + calls_12_)

theorem mr_brown_no_calls_in_2020 : number_of_days_with_no_calls 365 2 3 4 = 122 := 
  by 
    -- Proof steps would go here
    sorry

end NUMINAMATH_GPT_mr_brown_no_calls_in_2020_l924_92454


namespace NUMINAMATH_GPT_petya_purchase_cost_l924_92421

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end NUMINAMATH_GPT_petya_purchase_cost_l924_92421


namespace NUMINAMATH_GPT_total_cars_parked_l924_92483

theorem total_cars_parked
  (area_a : ℕ) (util_a : ℕ)
  (area_b : ℕ) (util_b : ℕ)
  (area_c : ℕ) (util_c : ℕ)
  (area_d : ℕ) (util_d : ℕ)
  (space_per_car : ℕ) 
  (ha: area_a = 400 * 500)
  (hu_a: util_a = 80)
  (hb: area_b = 600 * 700)
  (hu_b: util_b = 75)
  (hc: area_c = 500 * 800)
  (hu_c: util_c = 65)
  (hd: area_d = 300 * 900)
  (hu_d: util_d = 70)
  (h_sp: space_per_car = 10) :
  (util_a * area_a / 100 / space_per_car + 
   util_b * area_b / 100 / space_per_car + 
   util_c * area_c / 100 / space_per_car + 
   util_d * area_d / 100 / space_per_car) = 92400 :=
by sorry

end NUMINAMATH_GPT_total_cars_parked_l924_92483


namespace NUMINAMATH_GPT_minimum_fraction_l924_92430

theorem minimum_fraction (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m + 2 * n = 8) : 2 / m + 1 / n = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_fraction_l924_92430


namespace NUMINAMATH_GPT_range_of_a_l924_92494

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3) ↔ (-1 < a ∧ a < 2) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l924_92494


namespace NUMINAMATH_GPT_amount_paid_l924_92405

def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def change_received : ℕ := 11

theorem amount_paid (h_cost : ℕ := hamburger_cost) (o_cost : ℕ := onion_rings_cost) (s_cost : ℕ := smoothie_cost) (change : ℕ := change_received) :
  h_cost + o_cost + s_cost + change = 20 := by
  sorry

end NUMINAMATH_GPT_amount_paid_l924_92405


namespace NUMINAMATH_GPT_negation_p_equiv_l924_92497

noncomputable def negation_of_proposition_p : Prop :=
∀ m : ℝ, ¬ ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem negation_p_equiv (p : Prop) (h : p = ∃ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0) :
  ¬ p ↔ negation_of_proposition_p :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_p_equiv_l924_92497


namespace NUMINAMATH_GPT_solve_for_N_l924_92480

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_N_l924_92480


namespace NUMINAMATH_GPT_general_term_l924_92489

noncomputable def S : ℕ → ℤ
| n => 3 * n ^ 2 - 2 * n + 1

def a : ℕ → ℤ
| 0 => 2  -- Since sequences often start at n=1 and MATLAB indexing starts at 0.
| 1 => 2
| (n+2) => 6 * (n + 2) - 5

theorem general_term (n : ℕ) : 
  a n = if n = 1 then 2 else 6 * n - 5 :=
by sorry

end NUMINAMATH_GPT_general_term_l924_92489


namespace NUMINAMATH_GPT_expected_red_light_l924_92428

variables (n : ℕ) (p : ℝ)
def binomial_distribution : Type := sorry

noncomputable def expected_value (n : ℕ) (p : ℝ) : ℝ :=
n * p

theorem expected_red_light :
  expected_value 3 0.4 = 1.2 :=
by
  simp [expected_value]
  sorry

end NUMINAMATH_GPT_expected_red_light_l924_92428


namespace NUMINAMATH_GPT_number_of_allowed_pairs_l924_92419

theorem number_of_allowed_pairs (total_books : ℕ) (prohibited_books : ℕ) : ℕ :=
  let total_pairs := (total_books * (total_books - 1)) / 2
  let prohibited_pairs := (prohibited_books * (prohibited_books - 1)) / 2
  total_pairs - prohibited_pairs

example : number_of_allowed_pairs 15 3 = 102 :=
by
  sorry

end NUMINAMATH_GPT_number_of_allowed_pairs_l924_92419


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l924_92459

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1))
  (h2 : S 3 = 6)
  (h3 : a 3 = 3) :
  S 2023 / 2023 = 1012 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l924_92459


namespace NUMINAMATH_GPT_total_weekly_airflow_l924_92446

-- Definitions from conditions
def fanA_airflow : ℝ := 10  -- liters per second
def fanA_time_per_day : ℝ := 10 * 60  -- converted to seconds (10 minutes * 60 seconds/minute)

def fanB_airflow : ℝ := 15  -- liters per second
def fanB_time_per_day : ℝ := 20 * 60  -- converted to seconds (20 minutes * 60 seconds/minute)

def fanC_airflow : ℝ := 25  -- liters per second
def fanC_time_per_day : ℝ := 30 * 60  -- converted to seconds (30 minutes * 60 seconds/minute)

def days_in_week : ℝ := 7

-- Theorem statement to be proven
theorem total_weekly_airflow : fanA_airflow * fanA_time_per_day * days_in_week +
                               fanB_airflow * fanB_time_per_day * days_in_week +
                               fanC_airflow * fanC_time_per_day * days_in_week = 483000 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_total_weekly_airflow_l924_92446


namespace NUMINAMATH_GPT_div_by_1963_iff_odd_l924_92427

-- Define the given condition and statement
theorem div_by_1963_iff_odd (n : ℕ) :
  (1963 ∣ (82^n + 454 * 69^n)) ↔ (n % 2 = 1) :=
sorry

end NUMINAMATH_GPT_div_by_1963_iff_odd_l924_92427


namespace NUMINAMATH_GPT_min_value_of_quadratic_l924_92409

theorem min_value_of_quadratic (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) : 
  ∃ (x : ℝ), (∃ (y_min : ℝ), y_min = -( (abs (a - b)/2)^2 ) 
  ∧ ∀ (x : ℝ), (x - a)*(x - b) ≥ y_min) :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l924_92409


namespace NUMINAMATH_GPT_non_integer_interior_angle_count_l924_92403

theorem non_integer_interior_angle_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n < 10 ∧ ¬(∃ k : ℕ, 180 * (n - 2) = n * k) :=
by sorry

end NUMINAMATH_GPT_non_integer_interior_angle_count_l924_92403


namespace NUMINAMATH_GPT_find_satisfying_pairs_l924_92426

theorem find_satisfying_pairs (n p : ℕ) (prime_p : Nat.Prime p) :
  n ≤ 2 * p ∧ (p - 1)^n + 1 ≡ 0 [MOD n^2] →
  (n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end NUMINAMATH_GPT_find_satisfying_pairs_l924_92426


namespace NUMINAMATH_GPT_problem_statement_l924_92435

variable {f : ℝ → ℝ}
variable {a : ℝ}

def odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem problem_statement
  (h_odd : odd_function f)
  (h_periodic : periodic_function f 3)
  (h_f1 : f 1 < 1)
  (h_f2 : f 2 = a) :
  -1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l924_92435


namespace NUMINAMATH_GPT_simplify_cube_root_21952000_l924_92423

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem simplify_cube_root_21952000 : 
  cube_root 21952000 = 280 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_cube_root_21952000_l924_92423


namespace NUMINAMATH_GPT_total_pencils_crayons_l924_92470

theorem total_pencils_crayons (r : ℕ) (p : ℕ) (c : ℕ) 
  (hp : p = 31) (hc : c = 27) (hr : r = 11) : 
  r * p + r * c = 638 := 
  by
  sorry

end NUMINAMATH_GPT_total_pencils_crayons_l924_92470


namespace NUMINAMATH_GPT_fraction_of_sum_l924_92481

theorem fraction_of_sum (P : ℝ) (R : ℝ) (T : ℝ) (H_R : R = 8.333333333333337) (H_T : T = 2) : 
  let SI := (P * R * T) / 100
  let A := P + SI
  A / P = 1.1666666666666667 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_sum_l924_92481


namespace NUMINAMATH_GPT_line_parallel_through_point_l924_92467

theorem line_parallel_through_point (P : ℝ × ℝ) (a b c : ℝ) (ha : a = 3) (hb : b = -4) (hc : c = 6) (hP : P = (4, -1)) :
  ∃ d : ℝ, (d = -16) ∧ (∀ x y : ℝ, a * x + b * y + d = 0 ↔ 3 * x - 4 * y - 16 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_parallel_through_point_l924_92467


namespace NUMINAMATH_GPT_snowball_total_distance_l924_92491

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem snowball_total_distance :
  total_distance 6 5 25 = 1650 := by
  sorry

end NUMINAMATH_GPT_snowball_total_distance_l924_92491


namespace NUMINAMATH_GPT_ratio_is_one_to_five_l924_92479

def ratio_of_minutes_to_hour (twelve_minutes : ℕ) (one_hour : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd twelve_minutes one_hour
  (twelve_minutes / gcd, one_hour / gcd)

theorem ratio_is_one_to_five : ratio_of_minutes_to_hour 12 60 = (1, 5) := 
by 
  sorry

end NUMINAMATH_GPT_ratio_is_one_to_five_l924_92479


namespace NUMINAMATH_GPT_expression_value_l924_92498

   theorem expression_value :
     (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := 
   by
     sorry
   
end NUMINAMATH_GPT_expression_value_l924_92498


namespace NUMINAMATH_GPT_total_selling_price_correct_l924_92495

def meters_sold : ℕ := 85
def cost_price_per_meter : ℕ := 80
def profit_per_meter : ℕ := 25

def selling_price_per_meter : ℕ :=
  cost_price_per_meter + profit_per_meter

def total_selling_price : ℕ :=
  selling_price_per_meter * meters_sold

theorem total_selling_price_correct :
  total_selling_price = 8925 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l924_92495


namespace NUMINAMATH_GPT_walter_age_1999_l924_92401

variable (w g : ℕ) -- represents Walter's age (w) and his grandmother's age (g) in 1994
variable (birth_sum : ℕ) (w_age_1994 : ℕ) (g_age_1994 : ℕ)

axiom h1 : g = 2 * w
axiom h2 : (1994 - w) + (1994 - g) = 3838

theorem walter_age_1999 (w g : ℕ) (h1 : g = 2 * w) (h2 : (1994 - w) + (1994 - g) = 3838) : w + 5 = 55 :=
by
  sorry

end NUMINAMATH_GPT_walter_age_1999_l924_92401


namespace NUMINAMATH_GPT_mike_taller_than_mark_l924_92468

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end NUMINAMATH_GPT_mike_taller_than_mark_l924_92468


namespace NUMINAMATH_GPT_inequality_proof_l924_92411

open Real

noncomputable def f (t x : ℝ) : ℝ := t * x - (t - 1) * log x - t

theorem inequality_proof (t x : ℝ) (h_t : t ≤ 0) (h_x : x > 1) : 
  f t x < exp (x - 1) - 1 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l924_92411


namespace NUMINAMATH_GPT_general_formula_a_n_sum_T_n_l924_92410

-- Definitions of the sequences
def a (n : ℕ) : ℕ := 4 + (n - 1) * 1
def S (n : ℕ) : ℕ := n / 2 * (2 * 4 + (n - 1) * 1)
def b (n : ℕ) : ℕ := 2 ^ (a n - 3)
def T (n : ℕ) : ℕ := 2 * (2 ^ n - 1)

-- Given conditions
axiom a4_eq_7 : a 4 = 7
axiom S2_eq_9 : S 2 = 9

-- Theorems to prove
theorem general_formula_a_n : ∀ n, a n = n + 3 := 
by sorry

theorem sum_T_n : ∀ n, T n = 2 ^ (n + 1) - 2 := 
by sorry

end NUMINAMATH_GPT_general_formula_a_n_sum_T_n_l924_92410


namespace NUMINAMATH_GPT_f_increasing_f_odd_function_l924_92463

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : ∀ (x1 x2 : ℝ), x1 < x2 → f a x1 < f a x2 :=
by
  sorry

theorem f_odd_function (a : ℝ) : f a 0 = 0 → (a = 1) :=
by
  sorry

end NUMINAMATH_GPT_f_increasing_f_odd_function_l924_92463


namespace NUMINAMATH_GPT_find_a_l924_92414
open Real

theorem find_a (a : ℝ) (k : ℤ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x1^2 + y1^2 = 10 * (x1 * cos a + y1 * sin a) ∧
     x2^2 + y2^2 = 10 * (x2 * sin (3 * a) + y2 * cos (3 * a)) ∧
     (x2 - x1)^2 + (y2 - y1)^2 = 64)) ↔
  (∃ k : ℤ, a = π / 8 + k * π / 2) :=
sorry

end NUMINAMATH_GPT_find_a_l924_92414


namespace NUMINAMATH_GPT_paul_lost_crayons_l924_92453

theorem paul_lost_crayons :
  let total := 229
  let given_away := 213
  let lost := total - given_away
  lost = 16 :=
by
  sorry

end NUMINAMATH_GPT_paul_lost_crayons_l924_92453


namespace NUMINAMATH_GPT_part1_part2_l924_92412

variable (m : ℝ)

def p (m : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x - 2 ≥ m^2 - 3 * m
def q (m : ℝ) : Prop := ∃ x0 : ℝ, -1 ≤ x0 ∧ x0 ≤ 1 ∧ m ≤ x0

theorem part1 (h : p m) : 1 ≤ m ∧ m ≤ 2 := sorry

theorem part2 (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m < 1) ∨ (1 < m ∧ m ≤ 2) := sorry

end NUMINAMATH_GPT_part1_part2_l924_92412


namespace NUMINAMATH_GPT_combined_score_of_three_students_left_l924_92487

variable (T S : ℕ) (avg16 avg13 : ℝ) (N16 N13 : ℕ)

theorem combined_score_of_three_students_left (h_avg16 : avg16 = 62.5) 
  (h_avg13 : avg13 = 62.0) (h_N16 : N16 = 16) (h_N13 : N13 = 13) 
  (h_total16 : T = avg16 * N16) (h_total13 : T - S = avg13 * N13) :
  S = 194 :=
by
  sorry

end NUMINAMATH_GPT_combined_score_of_three_students_left_l924_92487


namespace NUMINAMATH_GPT_find_x_l924_92488

theorem find_x (n x : ℚ) (h1 : 3 * n + x = 6 * n - 10) (h2 : n = 25 / 3) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l924_92488


namespace NUMINAMATH_GPT_fixed_point_PQ_passes_l924_92417

theorem fixed_point_PQ_passes (P Q : ℝ × ℝ) (x1 x2 : ℝ)
  (hP : P = (x1, x1^2))
  (hQ : Q = (x2, x2^2))
  (hC1 : x1 ≠ 0)
  (hC2 : x2 ≠ 0)
  (hSlopes : (x2 / x2^2 * (2 * x1)) = -2) :
  ∃ D : ℝ × ℝ, D = (0, 1) ∧
    ∀ (x y : ℝ), (y = x1^2 + (x1 - (1 / x1)) * (x - x1)) → ((x, y) = P ∨ (x, y) = Q) := sorry

end NUMINAMATH_GPT_fixed_point_PQ_passes_l924_92417


namespace NUMINAMATH_GPT_find_sum_of_x_and_y_l924_92408

theorem find_sum_of_x_and_y (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end NUMINAMATH_GPT_find_sum_of_x_and_y_l924_92408


namespace NUMINAMATH_GPT_negation_of_p_l924_92420

def proposition_p (n : ℕ) : Prop := 3^n ≥ n + 1

theorem negation_of_p : (∃ n0 : ℕ, 3^n0 < n0^2 + 1) :=
  by sorry

end NUMINAMATH_GPT_negation_of_p_l924_92420


namespace NUMINAMATH_GPT_value_at_2_l924_92444

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

theorem value_at_2 : f 2 = 0 := by
  sorry

end NUMINAMATH_GPT_value_at_2_l924_92444


namespace NUMINAMATH_GPT_find_unit_prices_l924_92402

variable (x : ℝ)

def typeB_unit_price (priceB : ℝ) : Prop :=
  priceB = 15

def typeA_unit_price (priceA : ℝ) : Prop :=
  priceA = 40

def budget_condition : Prop :=
  900 / x = 3 * (800 / (x + 25))

theorem find_unit_prices (h : budget_condition x) :
  typeB_unit_price x ∧ typeA_unit_price (x + 25) :=
sorry

end NUMINAMATH_GPT_find_unit_prices_l924_92402


namespace NUMINAMATH_GPT_stockholm_to_malmo_road_distance_l924_92438

-- Define constants based on the conditions
def map_distance_cm : ℕ := 120
def scale_factor : ℕ := 10
def road_distance_multiplier : ℚ := 1.15

-- Define the real distances based on the conditions
def straight_line_distance_km : ℕ :=
  map_distance_cm * scale_factor

def road_distance_km : ℚ :=
  straight_line_distance_km * road_distance_multiplier

-- Assert the final statement
theorem stockholm_to_malmo_road_distance :
  road_distance_km = 1380 := 
sorry

end NUMINAMATH_GPT_stockholm_to_malmo_road_distance_l924_92438


namespace NUMINAMATH_GPT_find_m_values_l924_92450

theorem find_m_values (m : ℕ) : (m - 3) ^ m = 1 ↔ m = 0 ∨ m = 2 ∨ m = 4 := sorry

end NUMINAMATH_GPT_find_m_values_l924_92450


namespace NUMINAMATH_GPT_total_apples_picked_l924_92425

def Mike_apples : ℕ := 7
def Nancy_apples : ℕ := 3
def Keith_apples : ℕ := 6
def Jennifer_apples : ℕ := 5
def Tom_apples : ℕ := 8
def Stacy_apples : ℕ := 4

theorem total_apples_picked : 
  Mike_apples + Nancy_apples + Keith_apples + Jennifer_apples + Tom_apples + Stacy_apples = 33 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_picked_l924_92425


namespace NUMINAMATH_GPT_parabola_vertex_eq_l924_92469

theorem parabola_vertex_eq : 
  ∃ (x y : ℝ), y = -3 * x^2 + 6 * x + 1 ∧ (x = 1) ∧ (y = 4) := 
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_eq_l924_92469


namespace NUMINAMATH_GPT_min_value_shift_l924_92406

noncomputable def f (x : ℝ) (c : ℝ) := x^2 + 4 * x + 5 - c

theorem min_value_shift (c : ℝ) (h : ∀ x : ℝ, f x c ≥ 2) :
  ∀ x : ℝ, f (x - 2009) c ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_value_shift_l924_92406


namespace NUMINAMATH_GPT_find_integer_a_l924_92496

-- Definitions based on the conditions
def in_ratio (x y z : ℕ) := ∃ k : ℕ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k
def satisfies_equation (z : ℕ) (a : ℕ) := z = 30 * a - 15

-- The proof problem statement
theorem find_integer_a (x y z : ℕ) (a : ℕ) :
  in_ratio x y z →
  satisfies_equation z a →
  (∃ a : ℕ, a = 4) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_integer_a_l924_92496


namespace NUMINAMATH_GPT_smaller_solution_l924_92440

theorem smaller_solution (x : ℝ) (h : x^2 + 9 * x - 22 = 0) : x = -11 :=
sorry

end NUMINAMATH_GPT_smaller_solution_l924_92440


namespace NUMINAMATH_GPT_hexadecagon_area_l924_92404

theorem hexadecagon_area (r : ℝ) : 
  let θ := (360 / 16 : ℝ)
  let A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180)
  let total_area := 16 * A_triangle
  3 * r^2 = total_area :=
by
  sorry

end NUMINAMATH_GPT_hexadecagon_area_l924_92404


namespace NUMINAMATH_GPT_coffee_table_price_l924_92457

theorem coffee_table_price :
  let sofa := 1250
  let armchairs := 2 * 425
  let rug := 350
  let bookshelf := 200
  let subtotal_without_coffee_table := sofa + armchairs + rug + bookshelf
  let C := 429.24
  let total_before_discount_and_tax := subtotal_without_coffee_table + C
  let discounted_total := total_before_discount_and_tax * 0.90
  let final_invoice_amount := discounted_total * 1.06
  final_invoice_amount = 2937.60 :=
by
  sorry

end NUMINAMATH_GPT_coffee_table_price_l924_92457


namespace NUMINAMATH_GPT_least_homeowners_l924_92441

theorem least_homeowners (M W : ℕ) (total_members : M + W = 150)
  (men_homeowners : ∃ n : ℕ, n = 10 * M / 100) 
  (women_homeowners : ∃ n : ℕ, n = 20 * W / 100) : 
  ∃ homeowners : ℕ, homeowners = 16 := 
sorry

end NUMINAMATH_GPT_least_homeowners_l924_92441


namespace NUMINAMATH_GPT_fraction_of_donations_l924_92400

def max_donation_amount : ℝ := 1200
def total_money_raised : ℝ := 3750000
def donations_from_500_people : ℝ := 500 * max_donation_amount
def fraction_of_money_raised : ℝ := 0.4 * total_money_raised
def num_donors : ℝ := 1500

theorem fraction_of_donations (f : ℝ) :
  donations_from_500_people + num_donors * f * max_donation_amount = fraction_of_money_raised → f = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_donations_l924_92400


namespace NUMINAMATH_GPT_min_value_geometric_sequence_l924_92407

theorem min_value_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h : 0 < q ∧ 0 < a 0) 
  (H : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) 
  (h_geom : ∀ n, a (n+1) = a n * q) : 
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_geometric_sequence_l924_92407


namespace NUMINAMATH_GPT_elizabeth_bananas_eaten_l924_92429

theorem elizabeth_bananas_eaten (initial_bananas remaining_bananas eaten_bananas : ℕ) 
    (h1 : initial_bananas = 12) 
    (h2 : remaining_bananas = 8) 
    (h3 : eaten_bananas = initial_bananas - remaining_bananas) :
    eaten_bananas = 4 := 
sorry

end NUMINAMATH_GPT_elizabeth_bananas_eaten_l924_92429


namespace NUMINAMATH_GPT_product_identity_l924_92448

variable (x y : ℝ)

theorem product_identity :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end NUMINAMATH_GPT_product_identity_l924_92448


namespace NUMINAMATH_GPT_miles_per_hour_l924_92439

theorem miles_per_hour (total_distance : ℕ) (total_hours : ℕ) (h1 : total_distance = 81) (h2 : total_hours = 3) :
  total_distance / total_hours = 27 :=
by
  sorry

end NUMINAMATH_GPT_miles_per_hour_l924_92439


namespace NUMINAMATH_GPT_percentage_greater_l924_92499

theorem percentage_greater (x : ℝ) (h1 : x = 96) (h2 : x > 80) : ((x - 80) / 80) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_greater_l924_92499


namespace NUMINAMATH_GPT_min_bounces_for_height_less_than_two_l924_92476

theorem min_bounces_for_height_less_than_two : 
  ∃ (k : ℕ), (20 * (3 / 4 : ℝ)^k < 2 ∧ ∀ n < k, ¬(20 * (3 / 4 : ℝ)^n < 2)) :=
sorry

end NUMINAMATH_GPT_min_bounces_for_height_less_than_two_l924_92476


namespace NUMINAMATH_GPT_corvette_trip_average_rate_l924_92478

theorem corvette_trip_average_rate (total_distance : ℕ) (first_half_distance : ℕ)
  (first_half_rate : ℕ) (second_half_time_multiplier : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_rate = 80 →
  second_half_time_multiplier = 3 →
  total_time = (first_half_distance / first_half_rate) + (second_half_time_multiplier * (first_half_distance / first_half_rate)) →
  (total_distance / total_time) = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_corvette_trip_average_rate_l924_92478


namespace NUMINAMATH_GPT_simplify_fraction_l924_92416

theorem simplify_fraction :
  (1 / (3 / (Real.sqrt 5 + 2) + 4 / (Real.sqrt 7 - 2))) = (3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10)) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l924_92416


namespace NUMINAMATH_GPT_equation_solutions_35_implies_n_26_l924_92461

theorem equation_solutions_35_implies_n_26 (n : ℕ) (h3x3y2z_eq_n : ∃ (s : Finset (ℕ × ℕ × ℕ)), (∀ t ∈ s, ∃ (x y z : ℕ), 
  t = (x, y, z) ∧ 3 * x + 3 * y + 2 * z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ s.card = 35) : n = 26 := 
sorry

end NUMINAMATH_GPT_equation_solutions_35_implies_n_26_l924_92461


namespace NUMINAMATH_GPT_fabulous_integers_l924_92482

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (a^n - a) % n = 0

theorem fabulous_integers (n : ℕ) : is_fabulous n ↔ ¬(∃ k : ℕ, n = 2^k ∧ k ≥ 1) := 
sorry

end NUMINAMATH_GPT_fabulous_integers_l924_92482


namespace NUMINAMATH_GPT_Jason_reroll_probability_optimal_l924_92473

/-- Represents the action of rerolling dice to achieve a sum of 9 when
    the player optimizes their strategy. The probability 
    that the player chooses to reroll exactly two dice.
 -/
noncomputable def probability_reroll_two_dice : ℚ :=
  13 / 72

/-- Prove that the probability Jason chooses to reroll exactly two
    dice to achieve a sum of 9, given the optimal strategy, is 13/72.
 -/
theorem Jason_reroll_probability_optimal :
  probability_reroll_two_dice = 13 / 72 :=
sorry

end NUMINAMATH_GPT_Jason_reroll_probability_optimal_l924_92473


namespace NUMINAMATH_GPT_function_correct_max_min_values_l924_92484

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

@[simp]
theorem function_correct : (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧ 
                           (f (3 * Real.pi / 8) = 0) ∧ 
                           (f (Real.pi / 8) = 2) :=
by
  sorry

theorem max_min_values : (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = -2) ∧ 
                         (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = 2) :=
by
  sorry

end NUMINAMATH_GPT_function_correct_max_min_values_l924_92484


namespace NUMINAMATH_GPT_inequality_solution_l924_92477

theorem inequality_solution (x : ℝ) :
  (0 < x ∧ x ≤ 5 / 6 ∨ 2 < x) ↔ 
  ((2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l924_92477


namespace NUMINAMATH_GPT_det_E_eq_25_l924_92415

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end NUMINAMATH_GPT_det_E_eq_25_l924_92415


namespace NUMINAMATH_GPT_required_connections_l924_92492

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_GPT_required_connections_l924_92492


namespace NUMINAMATH_GPT_students_taking_either_geometry_or_history_but_not_both_l924_92475

theorem students_taking_either_geometry_or_history_but_not_both
    (students_in_both : ℕ)
    (students_in_geometry : ℕ)
    (students_only_in_history : ℕ)
    (students_in_both_cond : students_in_both = 15)
    (students_in_geometry_cond : students_in_geometry = 35)
    (students_only_in_history_cond : students_only_in_history = 18) :
    (students_in_geometry - students_in_both + students_only_in_history = 38) :=
by
  sorry

end NUMINAMATH_GPT_students_taking_either_geometry_or_history_but_not_both_l924_92475


namespace NUMINAMATH_GPT_find_first_term_l924_92486

def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem find_first_term (a r : ℝ) (h1 : r = 2/3) (h2 : geom_seq a r 3 = 18) (h3 : geom_seq a r 4 = 12) : a = 40.5 := 
by sorry

end NUMINAMATH_GPT_find_first_term_l924_92486


namespace NUMINAMATH_GPT_mom_younger_than_grandmom_l924_92418

def cara_age : ℕ := 40
def cara_younger_mom : ℕ := 20
def grandmom_age : ℕ := 75

def mom_age : ℕ := cara_age + cara_younger_mom
def age_difference : ℕ := grandmom_age - mom_age

theorem mom_younger_than_grandmom : age_difference = 15 := by
  sorry

end NUMINAMATH_GPT_mom_younger_than_grandmom_l924_92418


namespace NUMINAMATH_GPT_find_number_l924_92445

theorem find_number :
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  x = 24998465 :=
by
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  sorry

end NUMINAMATH_GPT_find_number_l924_92445


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_absolute_sum_first_19_terms_l924_92452

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) :
  ∀ n : ℕ, a n = 28 - 3 * n := 
sorry

theorem absolute_sum_first_19_terms (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) (an_eq : ∀ n : ℕ, a n = 28 - 3 * n) :
  |a 1| + |a 3| + |a 5| + |a 7| + |a 9| + |a 11| + |a 13| + |a 15| + |a 17| + |a 19| = 150 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_absolute_sum_first_19_terms_l924_92452


namespace NUMINAMATH_GPT_percentage_equivalence_l924_92472

theorem percentage_equivalence (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 :=
sorry

end NUMINAMATH_GPT_percentage_equivalence_l924_92472


namespace NUMINAMATH_GPT_find_b_plus_c_l924_92431

theorem find_b_plus_c (a b c d : ℝ) 
    (h₁ : a + d = 6) 
    (h₂ : a * b + a * c + b * d + c * d = 40) : 
    b + c = 20 / 3 := 
sorry

end NUMINAMATH_GPT_find_b_plus_c_l924_92431


namespace NUMINAMATH_GPT_network_connections_l924_92485

theorem network_connections (n m : ℕ) (hn : n = 30) (hm : m = 5) 
(h_total_conn : (n * 4) / 2 = 60) : 
60 + m = 65 :=
by
  sorry

end NUMINAMATH_GPT_network_connections_l924_92485


namespace NUMINAMATH_GPT_simplify_expression_l924_92466

theorem simplify_expression :
  (625: ℝ)^(1/4) * (256: ℝ)^(1/3) = 20 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l924_92466


namespace NUMINAMATH_GPT_ratio_thursday_to_wednesday_l924_92490

variables (T : ℕ)

def time_studied_wednesday : ℕ := 2
def time_studied_thursday : ℕ := T
def time_studied_friday : ℕ := T / 2
def time_studied_weekend : ℕ := 2 + T + T / 2
def total_time_studied : ℕ := 22

theorem ratio_thursday_to_wednesday (h : 
  time_studied_wednesday + time_studied_thursday + time_studied_friday + time_studied_weekend = total_time_studied
) : (T : ℚ) / time_studied_wednesday = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_thursday_to_wednesday_l924_92490


namespace NUMINAMATH_GPT_count_squares_below_graph_l924_92465

theorem count_squares_below_graph (x y: ℕ) (h_eq : 12 * x + 180 * y = 2160) (h_first_quadrant : x ≥ 0 ∧ y ≥ 0) :
  let total_squares := 180 * 12
  let diagonal_squares := 191
  let below_squares := total_squares - diagonal_squares
  below_squares = 1969 :=
by
  sorry

end NUMINAMATH_GPT_count_squares_below_graph_l924_92465


namespace NUMINAMATH_GPT_range_of_b_if_solution_set_contains_1_2_3_l924_92413

theorem range_of_b_if_solution_set_contains_1_2_3 
  (b : ℝ)
  (h : ∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) :
  5 < b ∧ b < 7 :=
sorry

end NUMINAMATH_GPT_range_of_b_if_solution_set_contains_1_2_3_l924_92413


namespace NUMINAMATH_GPT_min_vertical_segment_length_l924_92442

noncomputable def f₁ (x : ℝ) : ℝ := |x|
noncomputable def f₂ (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem min_vertical_segment_length :
  ∃ m : ℝ, m = 3 ∧
            ∀ x : ℝ, abs (f₁ x - f₂ x) ≥ m :=
sorry

end NUMINAMATH_GPT_min_vertical_segment_length_l924_92442


namespace NUMINAMATH_GPT_pencils_pens_total_l924_92434

theorem pencils_pens_total (x : ℕ) (h1 : 4 * x + 1 = 7 * (5 * x - 1)) : 4 * x + 5 * x = 45 :=
by
  sorry

end NUMINAMATH_GPT_pencils_pens_total_l924_92434


namespace NUMINAMATH_GPT_domain_of_f_x_minus_1_l924_92424

theorem domain_of_f_x_minus_1 (f : ℝ → ℝ) (h : ∀ x, x^2 + 1 ∈ Set.Icc 1 10 → x ∈ Set.Icc (-3 : ℝ) 2) :
  Set.Icc 2 (11 : ℝ) ⊆ {x : ℝ | x - 1 ∈ Set.Icc 1 10} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_x_minus_1_l924_92424


namespace NUMINAMATH_GPT_shorter_piece_length_l924_92462

theorem shorter_piece_length (L : ℝ) (k : ℝ) (shorter_piece : ℝ) : 
  L = 28 ∧ k = 2.00001 / 5 ∧ L = shorter_piece + k * shorter_piece → 
  shorter_piece = 20 :=
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l924_92462


namespace NUMINAMATH_GPT_rowing_upstream_speed_l924_92493

theorem rowing_upstream_speed (Vm Vdown : ℝ) (H1 : Vm = 20) (H2 : Vdown = 33) :
  ∃ Vup Vs : ℝ, Vup = Vm - Vs ∧ Vs = Vdown - Vm ∧ Vup = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_rowing_upstream_speed_l924_92493


namespace NUMINAMATH_GPT_pencils_to_sell_l924_92437

/--
A store owner bought 1500 pencils at $0.10 each. 
Each pencil is sold for $0.25. 
He wants to make a profit of exactly $100. 
Prove that he must sell 1000 pencils to achieve this profit.
-/
theorem pencils_to_sell (total_pencils : ℕ) (cost_per_pencil : ℝ) (selling_price_per_pencil : ℝ) (desired_profit : ℝ)
  (h1 : total_pencils = 1500)
  (h2 : cost_per_pencil = 0.10)
  (h3 : selling_price_per_pencil = 0.25)
  (h4 : desired_profit = 100) :
  total_pencils * cost_per_pencil + desired_profit = 1000 * selling_price_per_pencil :=
by
  -- Since Lean code requires some proof content, we put sorry to skip it.
  sorry

end NUMINAMATH_GPT_pencils_to_sell_l924_92437


namespace NUMINAMATH_GPT_max_integer_is_twelve_l924_92432

theorem max_integer_is_twelve
  (a b c d e : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : (a + b + c + d + e) / 5 = 9)
  (h6 : ((a - 9)^2 + (b - 9)^2 + (c - 9)^2 + (d - 9)^2 + (e - 9)^2) / 5 = 4) :
  e = 12 := sorry

end NUMINAMATH_GPT_max_integer_is_twelve_l924_92432


namespace NUMINAMATH_GPT_minimum_value_a_plus_2b_l924_92447

theorem minimum_value_a_plus_2b {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) : a + 2 * b = 9 :=
by sorry

end NUMINAMATH_GPT_minimum_value_a_plus_2b_l924_92447


namespace NUMINAMATH_GPT_find_values_l924_92455

theorem find_values (a b : ℝ) 
  (h1 : a + b = 10)
  (h2 : a - b = 4) 
  (h3 : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ ab = 21 := 
by 
  sorry

end NUMINAMATH_GPT_find_values_l924_92455


namespace NUMINAMATH_GPT_evaluateExpression_at_1_l924_92422

noncomputable def evaluateExpression (x : ℝ) : ℝ :=
  (x^2 - 3 * x - 10) / (x - 5)

theorem evaluateExpression_at_1 : evaluateExpression 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluateExpression_at_1_l924_92422


namespace NUMINAMATH_GPT_Mrs_Hilt_bought_two_cones_l924_92456

def ice_cream_cone_cost : ℕ := 99
def total_spent : ℕ := 198

theorem Mrs_Hilt_bought_two_cones : total_spent / ice_cream_cone_cost = 2 :=
by
  sorry

end NUMINAMATH_GPT_Mrs_Hilt_bought_two_cones_l924_92456


namespace NUMINAMATH_GPT_distinct_arrangements_of_PHONE_l924_92449

-- Condition: The word PHONE consists of 5 distinct letters
def distinctLetters := 5

-- Theorem: The number of distinct arrangements of the letters in the word PHONE
theorem distinct_arrangements_of_PHONE : Nat.factorial distinctLetters = 120 := sorry

end NUMINAMATH_GPT_distinct_arrangements_of_PHONE_l924_92449


namespace NUMINAMATH_GPT_magician_identifies_card_l924_92443

def Grid : Type := Fin 6 → Fin 6 → Nat

def choose_card (g : Grid) (c : Fin 6) (r : Fin 6) : Nat := g r c

def rearrange_columns_to_rows (s : List Nat) : Grid :=
  λ r c => s.get! (r.val * 6 + c.val)

theorem magician_identifies_card (g : Grid) (c1 : Fin 6) (r2 : Fin 6) :
  ∃ (card : Nat), (choose_card g c1 r2 = card) :=
  sorry

end NUMINAMATH_GPT_magician_identifies_card_l924_92443


namespace NUMINAMATH_GPT_tan_at_max_value_l924_92471

theorem tan_at_max_value : 
  ∃ x₀, (∀ x, 3 * Real.sin x₀ - 4 * Real.cos x₀ ≥ 3 * Real.sin x - 4 * Real.cos x) → Real.tan x₀ = 3/4 := 
sorry

end NUMINAMATH_GPT_tan_at_max_value_l924_92471


namespace NUMINAMATH_GPT_possible_values_x_l924_92474

variable (a b x : ℕ)

theorem possible_values_x (h1 : a + b = 20)
                          (h2 : a * x + b * 3 = 109) :
    x = 10 ∨ x = 52 :=
sorry

end NUMINAMATH_GPT_possible_values_x_l924_92474


namespace NUMINAMATH_GPT_average_temperature_correct_l924_92433

theorem average_temperature_correct (W T : ℝ) :
  (38 + W + T) / 3 = 32 →
  44 = 44 →
  38 = 38 →
  (W + T + 44) / 3 = 34 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_average_temperature_correct_l924_92433


namespace NUMINAMATH_GPT_exactly_one_solves_problem_l924_92458

theorem exactly_one_solves_problem (pA pB pC : ℝ) (hA : pA = 1 / 2) (hB : pB = 1 / 3) (hC : pC = 1 / 4) :
  (pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC) = 11 / 24 :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_solves_problem_l924_92458
