import Mathlib

namespace NUMINAMATH_GPT_shirts_before_buying_l436_43636

-- Define the conditions
variable (new_shirts : ℕ)
variable (total_shirts : ℕ)

-- Define the statement where we need to prove the number of shirts Sarah had before buying the new ones
theorem shirts_before_buying (h₁ : new_shirts = 8) (h₂ : total_shirts = 17) : total_shirts - new_shirts = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_shirts_before_buying_l436_43636


namespace NUMINAMATH_GPT_prob_next_black_ball_l436_43605

theorem prob_next_black_ball
  (total_balls : ℕ := 100) 
  (black_balls : Fin 101) 
  (next_black_ball_probability : ℚ := 2 / 3) :
  black_balls.val ≤ total_balls →
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (p : ℚ) / q = next_black_ball_probability ∧ p + q = 5 :=
by
  intros h
  use 2, 3
  repeat { sorry }

end NUMINAMATH_GPT_prob_next_black_ball_l436_43605


namespace NUMINAMATH_GPT_direct_proportion_function_l436_43693

theorem direct_proportion_function (m : ℝ) 
  (h1 : m + 1 ≠ 0) 
  (h2 : m^2 - 1 = 0) : 
  m = 1 :=
sorry

end NUMINAMATH_GPT_direct_proportion_function_l436_43693


namespace NUMINAMATH_GPT_min_side_length_l436_43642

def table_diagonal (w h : ℕ) : ℕ :=
  Nat.sqrt (w * w + h * h)

theorem min_side_length (w h : ℕ) (S : ℕ) (dw : w = 9) (dh : h = 12) (dS : S = 15) :
  S >= table_diagonal w h :=
by
  sorry

end NUMINAMATH_GPT_min_side_length_l436_43642


namespace NUMINAMATH_GPT_batsman_average_increase_l436_43653

-- Definitions to capture the initial conditions
def runs_scored_in_17th_inning : ℕ := 74
def average_after_17_innings : ℕ := 26

-- Statement to prove the increment in average is 3 runs per inning
theorem batsman_average_increase (A : ℕ) (initial_avg : ℕ)
  (h_initial_runs : 16 * initial_avg + 74 = 17 * 26) :
  26 - initial_avg = 3 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l436_43653


namespace NUMINAMATH_GPT_rhombus_perimeter_l436_43660

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l436_43660


namespace NUMINAMATH_GPT_simplify_expression_l436_43689

variable {a b c : ℝ}

-- Assuming the conditions specified in the problem
def valid_conditions (a b c : ℝ) : Prop := (1 - a * b ≠ 0) ∧ (1 + c * a ≠ 0)

theorem simplify_expression (h : valid_conditions a b c) :
  (a + b) / (1 - a * b) + (c - a) / (1 + c * a) / 
  (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) = 
  (b + c) / (1 - b * c) := 
sorry

end NUMINAMATH_GPT_simplify_expression_l436_43689


namespace NUMINAMATH_GPT_least_positive_n_for_reducible_fraction_l436_43613

theorem least_positive_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (6 * n + 7)) ∧ n = 126 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_n_for_reducible_fraction_l436_43613


namespace NUMINAMATH_GPT_incorrect_proposition3_l436_43687

open Real

-- Definitions from the problem
def prop1 (x : ℝ) := 2 * sin (2 * x - π / 3) = 2
def prop2 (x y : ℝ) := tan x + tan (π - x) = 0
def prop3 (x1 x2 : ℝ) (k : ℤ) := x1 - x2 = (k : ℝ) * π → k % 2 = 1
def prop4 (x : ℝ) := cos x ^ 2 + sin x >= -1

-- Incorrect proposition proof
theorem incorrect_proposition3 (x1 x2 : ℝ) (k : ℤ) :
  sin (2 * x1 - π / 4) = 0 →
  sin (2 * x2 - π / 4) = 0 →
  x1 - x2 ≠ (k : ℝ) * π := sorry

end NUMINAMATH_GPT_incorrect_proposition3_l436_43687


namespace NUMINAMATH_GPT_circle_center_and_radius_l436_43627

-- Define the given conditions
variable (a : ℝ) (h : a^2 = a + 2 ∧ a ≠ 0)

-- Define the equation
noncomputable def circle_equation (x y : ℝ) : ℝ := a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a

-- Lean definition to represent the problem
theorem circle_center_and_radius :
  (∃a : ℝ, a ≠ 0 ∧ a^2 = a + 2 ∧
    (∃x y : ℝ, circle_equation a x y = 0) ∧
    ((a = -1) → ((∃x y : ℝ, (x + 2)^2 + (y + 4)^2 = 25) ∧
                 (center_x = -2) ∧ (center_y = -4) ∧ (radius = 5)))) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_l436_43627


namespace NUMINAMATH_GPT_maximize_x2y5_l436_43622

theorem maximize_x2y5 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 50) : 
  x = 100 / 7 ∧ y = 250 / 7 :=
sorry

end NUMINAMATH_GPT_maximize_x2y5_l436_43622


namespace NUMINAMATH_GPT_square_area_side4_l436_43602

theorem square_area_side4
  (s : ℕ)
  (A : ℕ)
  (P : ℕ)
  (h_s : s = 4)
  (h_A : A = s * s)
  (h_P : P = 4 * s)
  (h_eqn : (A + s) - P = 4) : A = 16 := sorry

end NUMINAMATH_GPT_square_area_side4_l436_43602


namespace NUMINAMATH_GPT_inequality_proof_l436_43645

theorem inequality_proof (a b c : ℝ) (k : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : k ≥ 1) : 
  (a^(k + 1) / b^k + b^(k + 1) / c^k + c^(k + 1) / a^k) ≥ (a^k / b^(k - 1) + b^k / c^(k - 1) + c^k / a^(k - 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l436_43645


namespace NUMINAMATH_GPT_nth_equation_l436_43640

theorem nth_equation (n : ℕ) : 
  1 - (1 / ((n + 1)^2)) = (n / (n + 1)) * ((n + 2) / (n + 1)) :=
by sorry

end NUMINAMATH_GPT_nth_equation_l436_43640


namespace NUMINAMATH_GPT_perpendicular_lines_l436_43609

theorem perpendicular_lines :
  ∃ y x : ℝ, (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 12) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l436_43609


namespace NUMINAMATH_GPT_factorize_expression_l436_43676

theorem factorize_expression (a x : ℝ) : a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l436_43676


namespace NUMINAMATH_GPT_probability_of_triangle_l436_43634

/-- There are 12 figures in total: 4 squares, 5 triangles, and 3 rectangles.
    Prove that the probability of choosing a triangle is 5/12. -/
theorem probability_of_triangle (total_figures : ℕ) (num_squares : ℕ) (num_triangles : ℕ) (num_rectangles : ℕ)
  (h1 : total_figures = 12)
  (h2 : num_squares = 4)
  (h3 : num_triangles = 5)
  (h4 : num_rectangles = 3) :
  num_triangles / total_figures = 5 / 12 :=
sorry

end NUMINAMATH_GPT_probability_of_triangle_l436_43634


namespace NUMINAMATH_GPT_percentage_women_no_french_speak_spanish_german_l436_43646

variable (total_workforce : Nat)
variable (men_percentage women_percentage : ℕ)
variable (men_only_french men_only_spanish men_only_german : ℕ)
variable (men_both_french_spanish men_both_french_german men_both_spanish_german : ℕ)
variable (men_all_three_languages women_only_french women_only_spanish : ℕ)
variable (women_only_german women_both_french_spanish women_both_french_german : ℕ)
variable (women_both_spanish_german women_all_three_languages : ℕ)

-- Conditions
axiom h1 : men_percentage = 60
axiom h2 : women_percentage = 40
axiom h3 : women_only_french = 30
axiom h4 : women_only_spanish = 25
axiom h5 : women_only_german = 20
axiom h6 : women_both_french_spanish = 10
axiom h7 : women_both_french_german = 5
axiom h8 : women_both_spanish_german = 5
axiom h9 : women_all_three_languages = 5

theorem percentage_women_no_french_speak_spanish_german:
  women_only_spanish + women_only_german + women_both_spanish_german = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_women_no_french_speak_spanish_german_l436_43646


namespace NUMINAMATH_GPT_A_knit_time_l436_43637

def rate_A (x : ℕ) : ℚ := 1 / x
def rate_B : ℚ := 1 / 6

def combined_rate_two_pairs_in_4_days (x : ℕ) : Prop :=
  rate_A x + rate_B = 1 / 2

theorem A_knit_time : ∃ x : ℕ, combined_rate_two_pairs_in_4_days x ∧ x = 3 :=
by
  existsi 3
  -- (Formal proof would go here)
  sorry

end NUMINAMATH_GPT_A_knit_time_l436_43637


namespace NUMINAMATH_GPT_height_of_parabolic_arch_l436_43658

theorem height_of_parabolic_arch (a : ℝ) (x : ℝ) (k : ℝ) (h : ℝ) (s : ℝ) :
  k = 20 →
  s = 30 →
  a = - 4 / 45 →
  x = 3 →
  k = h →
  y = a * x^2 + k →
  h = 20 → 
  y = 19.2 :=
by
  -- Given the conditions, we'll prove using provided Lean constructs
  sorry

end NUMINAMATH_GPT_height_of_parabolic_arch_l436_43658


namespace NUMINAMATH_GPT_values_of_a_l436_43694

theorem values_of_a (a : ℝ) : 
  ∃a1 a2 : ℝ, 
  (∀ x y : ℝ, (y = 3 * x + a) ∧ (y = x^3 + 3 * a^2) → (x = 0) → (y = 3 * a^2)) →
  ((a = 0) ∨ (a = 1/3)) ∧ 
  ((a1 = 0) ∨ (a1 = 1/3)) ∧
  ((a2 = 0) ∨ (a2 = 1/3)) ∧ 
  (a ≠ a1 ∨ a ≠ a2) ∧ 
  (∃ n : ℤ, n = 2) :=
by sorry

end NUMINAMATH_GPT_values_of_a_l436_43694


namespace NUMINAMATH_GPT_marked_price_l436_43690

theorem marked_price (P : ℝ)
  (h₁ : 20 / 100 = 0.20)
  (h₂ : 15 / 100 = 0.15)
  (h₃ : 5 / 100 = 0.05)
  (h₄ : 7752 = 0.80 * 0.85 * 0.95 * P)
  : P = 11998.76 := by
  sorry

end NUMINAMATH_GPT_marked_price_l436_43690


namespace NUMINAMATH_GPT_right_triangle_smaller_angle_l436_43629

theorem right_triangle_smaller_angle (x : ℝ) (h_right_triangle : 0 < x ∧ x < 90)
  (h_double_angle : ∃ y : ℝ, y = 2 * x)
  (h_angle_sum : x + 2 * x = 90) :
  x = 30 :=
  sorry

end NUMINAMATH_GPT_right_triangle_smaller_angle_l436_43629


namespace NUMINAMATH_GPT_geometric_seq_ad_eq_2_l436_43656

open Real

def geometric_sequence (a b c d : ℝ) : Prop :=
∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r 

def is_max_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
f x = y ∧ ∀ z : ℝ, z ≠ x → f x ≥ f z

theorem geometric_seq_ad_eq_2 (a b c d : ℝ) :
  geometric_sequence a b c d →
  is_max_point (λ x => 3 * x - x ^ 3) b c →
  a * d = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_ad_eq_2_l436_43656


namespace NUMINAMATH_GPT_engineers_percentage_calculation_l436_43698

noncomputable def percentageEngineers (num_marketers num_engineers num_managers total_salary: ℝ) : ℝ := 
  let num_employees := num_marketers + num_engineers + num_managers 
  if num_employees = 0 then 0 else num_engineers / num_employees * 100

theorem engineers_percentage_calculation : 
  let marketers_percentage := 0.7 
  let engineers_salary := 80000
  let average_salary := 80000
  let marketers_salary_total := 50000 * marketers_percentage 
  let managers_total_percent := 1 - marketers_percentage - x / 100
  let managers_salary := 370000 * managers_total_percent 
  marketers_salary_total + engineers_salary * x / 100 + managers_salary = average_salary -> 
  x = 22.76 
:= 
sorry

end NUMINAMATH_GPT_engineers_percentage_calculation_l436_43698


namespace NUMINAMATH_GPT_prime_gt_five_condition_l436_43682

theorem prime_gt_five_condition (p : ℕ) [Fact (Nat.Prime p)] (h : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < p - a^2 ∧ p - a^2 < p - b^2 ∧ (p - a^2) ∣ (p - b)^2 := 
sorry

end NUMINAMATH_GPT_prime_gt_five_condition_l436_43682


namespace NUMINAMATH_GPT_electrical_bill_undetermined_l436_43631

theorem electrical_bill_undetermined
    (gas_bill : ℝ)
    (gas_paid_fraction : ℝ)
    (additional_gas_payment : ℝ)
    (water_bill : ℝ)
    (water_paid_fraction : ℝ)
    (internet_bill : ℝ)
    (internet_payments : ℝ)
    (payment_amounts: ℝ)
    (total_remaining : ℝ) :
    gas_bill = 40 →
    gas_paid_fraction = 3 / 4 →
    additional_gas_payment = 5 →
    water_bill = 40 →
    water_paid_fraction = 1 / 2 →
    internet_bill = 25 →
    internet_payments = 4 * 5 →
    total_remaining = 30 →
    (∃ electricity_bill : ℝ, true) -> 
    false := by
  intro gas_bill_eq gas_paid_fraction_eq additional_gas_payment_eq
  intro water_bill_eq water_paid_fraction_eq
  intro internet_bill_eq internet_payments_eq 
  intro total_remaining_eq 
  intro exists_electricity_bill 
  sorry -- Proof that the electricity bill cannot be determined

end NUMINAMATH_GPT_electrical_bill_undetermined_l436_43631


namespace NUMINAMATH_GPT_largest_of_numbers_l436_43670

theorem largest_of_numbers (a b c d : ℝ) (hₐ : a = 0) (h_b : b = -1) (h_c : c = -2) (h_d : d = Real.sqrt 3) :
  d = Real.sqrt 3 ∧ d > a ∧ d > b ∧ d > c :=
by
  -- Using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_largest_of_numbers_l436_43670


namespace NUMINAMATH_GPT_intersection_lines_l436_43624

theorem intersection_lines (c d : ℝ) :
    (∃ x y, x = (1/3) * y + c ∧ y = (1/3) * x + d ∧ x = 3 ∧ y = -1) →
    c + d = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_lines_l436_43624


namespace NUMINAMATH_GPT_general_formula_l436_43677

open Nat

def a (n : ℕ) : ℚ :=
  if n = 0 then 7/6 else 0 -- Recurrence initialization with dummy else condition

-- Defining the recurrence relation as a function
lemma recurrence_relation {n : ℕ} (h : n > 0) : 
    a n = (1 / 2) * a (n - 1) + (1 / 3) := 
sorry

-- Proof of the general formula
theorem general_formula (n : ℕ) : a n = (1 / (2^n : ℚ)) + (2 / 3) :=
sorry

end NUMINAMATH_GPT_general_formula_l436_43677


namespace NUMINAMATH_GPT_third_quadrant_condition_l436_43647

-- Define the conditions for the third quadrant
def in_third_quadrant (p: ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- Translate the problem statement to a Lean theorem
theorem third_quadrant_condition (a b : ℝ) (h1 : a + b < 0) (h2 : a * b > 0) : in_third_quadrant (a, b) :=
sorry

end NUMINAMATH_GPT_third_quadrant_condition_l436_43647


namespace NUMINAMATH_GPT_group_interval_eq_l436_43630

noncomputable def group_interval (a b m h : ℝ) : ℝ := abs (a - b)

theorem group_interval_eq (a b m h : ℝ) 
  (h1 : h = m / abs (a - b)) :
  abs (a - b) = m / h := 
by 
  sorry

end NUMINAMATH_GPT_group_interval_eq_l436_43630


namespace NUMINAMATH_GPT_f_2015_value_l436_43628

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_odd : odd_function f
axiom f_periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom f_definition_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 3^x - 1

theorem f_2015_value : f 2015 = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_2015_value_l436_43628


namespace NUMINAMATH_GPT_gwendolyn_reading_time_l436_43672

/--
Gwendolyn can read 200 sentences in 1 hour. 
Each paragraph has 10 sentences. 
There are 20 paragraphs per page. 
The book has 50 pages. 
--/
theorem gwendolyn_reading_time : 
  let sentences_per_hour := 200
  let sentences_per_paragraph := 10
  let paragraphs_per_page := 20
  let pages := 50
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  (total_sentences / sentences_per_hour) = 50 := 
by
  let sentences_per_hour : ℕ := 200
  let sentences_per_paragraph : ℕ := 10
  let paragraphs_per_page : ℕ := 20
  let pages : ℕ := 50
  let sentences_per_page : ℕ := sentences_per_paragraph * paragraphs_per_page
  let total_sentences : ℕ := sentences_per_page * pages
  have h : (total_sentences / sentences_per_hour) = 50 := by sorry
  exact h

end NUMINAMATH_GPT_gwendolyn_reading_time_l436_43672


namespace NUMINAMATH_GPT_mother_nickels_eq_two_l436_43643

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def total_nickels : ℕ := 18

theorem mother_nickels_eq_two : (total_nickels = initial_nickels + dad_nickels + 2) :=
by
  sorry

end NUMINAMATH_GPT_mother_nickels_eq_two_l436_43643


namespace NUMINAMATH_GPT_count_integers_congruent_mod_l436_43675

theorem count_integers_congruent_mod (n : ℕ) (h₁ : n < 1200) (h₂ : n ≡ 3 [MOD 7]) : 
  ∃ (m : ℕ), (m = 171) :=
by
  sorry

end NUMINAMATH_GPT_count_integers_congruent_mod_l436_43675


namespace NUMINAMATH_GPT_min_fraction_l436_43681

theorem min_fraction (x A C : ℝ) (hx : x > 0) (hA : A = x^2 + 1/x^2) (hC : C = x + 1/x) :
  ∃ m, m = 2 * Real.sqrt 2 ∧ ∀ B, B > 0 → x^2 + 1/x^2 = B → x + 1/x = C → B / C ≥ m :=
by
  sorry

end NUMINAMATH_GPT_min_fraction_l436_43681


namespace NUMINAMATH_GPT_valid_combinations_count_l436_43601

theorem valid_combinations_count : 
  let wrapping_paper_count := 10
  let ribbon_count := 3
  let gift_card_count := 5
  let invalid_combinations := 1 -- red ribbon with birthday card
  let total_combinations := wrapping_paper_count * ribbon_count * gift_card_count
  total_combinations - invalid_combinations = 149 := 
by 
  sorry

end NUMINAMATH_GPT_valid_combinations_count_l436_43601


namespace NUMINAMATH_GPT_sum_a_b_eq_neg2_l436_43659

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : f a + f b = 20) : a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_eq_neg2_l436_43659


namespace NUMINAMATH_GPT_total_cost_of_tires_and_battery_l436_43621

theorem total_cost_of_tires_and_battery :
  (4 * 42 + 56 = 224) := 
  by
    sorry

end NUMINAMATH_GPT_total_cost_of_tires_and_battery_l436_43621


namespace NUMINAMATH_GPT_max_value_x_plus_2y_l436_43626

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x * y = 4) :
  x + 2 * y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_x_plus_2y_l436_43626


namespace NUMINAMATH_GPT_randy_initial_money_l436_43692

theorem randy_initial_money (X : ℕ) (h : X + 200 - 1200 = 2000) : X = 3000 :=
by {
  sorry
}

end NUMINAMATH_GPT_randy_initial_money_l436_43692


namespace NUMINAMATH_GPT_x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l436_43620

theorem x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1 (x : ℝ) : (x > 1 → |x| > 1) ∧ (¬(x > 1 ↔ |x| > 1)) :=
by
  sorry

end NUMINAMATH_GPT_x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l436_43620


namespace NUMINAMATH_GPT_inequality_max_k_l436_43625

theorem inequality_max_k (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2 * d)^5) ≥ 174960 * a * b * c * d^3 :=
sorry

end NUMINAMATH_GPT_inequality_max_k_l436_43625


namespace NUMINAMATH_GPT_division_remainder_l436_43639

/-- The remainder when 3572 is divided by 49 is 44. -/
theorem division_remainder :
  3572 % 49 = 44 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l436_43639


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l436_43614

theorem quadratic_inequality_solution (a : ℝ) :
  ((0 ≤ a ∧ a < 3) → ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) :=
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l436_43614


namespace NUMINAMATH_GPT_totalUniqueStudents_l436_43632

-- Define the club memberships and overlap
variable (mathClub scienceClub artClub overlap : ℕ)

-- Conditions based on the problem
def mathClubSize : Prop := mathClub = 15
def scienceClubSize : Prop := scienceClub = 10
def artClubSize : Prop := artClub = 12
def overlapSize : Prop := overlap = 5

-- Main statement to prove
theorem totalUniqueStudents : 
  mathClubSize mathClub → 
  scienceClubSize scienceClub →
  artClubSize artClub →
  overlapSize overlap →
  mathClub + scienceClub + artClub - overlap = 32 := by
  intros
  sorry

end NUMINAMATH_GPT_totalUniqueStudents_l436_43632


namespace NUMINAMATH_GPT_two_digit_number_conditions_l436_43623

-- Definitions for two-digit number and its conditions
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The proof problem statement in Lean 4
theorem two_digit_number_conditions (N : ℕ) (c d : ℕ) :
  is_two_digit_number N ∧ N = 10 * c + d ∧ N' = N + 7 ∧ 
  N = 6 * sum_of_digits (N + 7) →
  N = 24 ∨ N = 78 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_conditions_l436_43623


namespace NUMINAMATH_GPT_rabbit_time_2_miles_l436_43680

def rabbit_travel_time (distance : ℕ) (rate : ℕ) : ℕ :=
  (distance * 60) / rate

theorem rabbit_time_2_miles : rabbit_travel_time 2 5 = 24 := by
  sorry

end NUMINAMATH_GPT_rabbit_time_2_miles_l436_43680


namespace NUMINAMATH_GPT_T_five_three_l436_43686

def T (a b : ℤ) : ℤ := 4 * a + 6 * b + 2

theorem T_five_three : T 5 3 = 40 := by
  sorry

end NUMINAMATH_GPT_T_five_three_l436_43686


namespace NUMINAMATH_GPT_roger_left_money_correct_l436_43655

noncomputable def roger_left_money (P : ℝ) (q : ℝ) (E : ℝ) (r1 : ℝ) (C : ℝ) (r2 : ℝ) : ℝ :=
  let feb_expense := q * P
  let after_feb := P - feb_expense
  let mar_expense := E * r1
  let after_mar := after_feb - mar_expense
  let mom_gift := C * r2
  after_mar + mom_gift

theorem roger_left_money_correct :
  roger_left_money 45 0.35 20 1.2 46 0.8 = 42.05 :=
by
  sorry

end NUMINAMATH_GPT_roger_left_money_correct_l436_43655


namespace NUMINAMATH_GPT_fuel_consumption_l436_43697

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end NUMINAMATH_GPT_fuel_consumption_l436_43697


namespace NUMINAMATH_GPT_simplify_expr_l436_43685

theorem simplify_expr (a : ℝ) : 2 * a * (3 * a ^ 2 - 4 * a + 3) - 3 * a ^ 2 * (2 * a - 4) = 4 * a ^ 2 + 6 * a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l436_43685


namespace NUMINAMATH_GPT_chess_match_probability_l436_43666

theorem chess_match_probability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (3 * p^3 * (1 - p) ≤ 6 * p^3 * (1 - p)^2) → (p ≤ 1/2) :=
by
  sorry

end NUMINAMATH_GPT_chess_match_probability_l436_43666


namespace NUMINAMATH_GPT_leopards_arrangement_l436_43683

theorem leopards_arrangement :
  let total_leopards := 9
  let ends_leopards := 2
  let middle_leopard := 1
  let remaining_leopards := total_leopards - ends_leopards - middle_leopard
  (2 * 1 * (Nat.factorial remaining_leopards) = 1440) := by
  sorry

end NUMINAMATH_GPT_leopards_arrangement_l436_43683


namespace NUMINAMATH_GPT_second_hand_distance_l436_43671

theorem second_hand_distance (r : ℝ) (t : ℝ) (π : ℝ) (hand_length_6cm : r = 6) (time_15_min : t = 15) : 
  ∃ d : ℝ, d = 180 * π :=
by
  sorry

end NUMINAMATH_GPT_second_hand_distance_l436_43671


namespace NUMINAMATH_GPT_plates_used_l436_43615

theorem plates_used (P : ℕ) (h : 3 * 2 * P + 4 * 8 = 38) : P = 1 := by
  sorry

end NUMINAMATH_GPT_plates_used_l436_43615


namespace NUMINAMATH_GPT_picture_size_l436_43611

theorem picture_size (total_pics_A : ℕ) (size_A : ℕ) (total_pics_B : ℕ) (C : ℕ)
  (hA : total_pics_A * size_A = C) (hB : total_pics_B = 3000) : 
  (C / total_pics_B = 8) :=
by
  sorry

end NUMINAMATH_GPT_picture_size_l436_43611


namespace NUMINAMATH_GPT_bicycle_cost_price_l436_43678

variable (CP_A SP_B SP_C : ℝ)

theorem bicycle_cost_price 
  (h1 : SP_B = CP_A * 1.20) 
  (h2 : SP_C = SP_B * 1.25) 
  (h3 : SP_C = 225) :
  CP_A = 150 := 
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_price_l436_43678


namespace NUMINAMATH_GPT_other_root_of_quadratic_l436_43673

theorem other_root_of_quadratic 
  (a b c: ℝ) 
  (h : a * (b - c - d) * (1:ℝ)^2 + b * (c - a + d) * (1:ℝ) + c * (a - b - d) = 0) : 
  ∃ k: ℝ, k = c * (a - b - d) / (a * (b - c - d)) :=
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l436_43673


namespace NUMINAMATH_GPT_eight_b_plus_one_composite_l436_43699

theorem eight_b_plus_one_composite (a b : ℕ) (h₀ : a > b)
  (h₁ : a - b = 5 * b^2 - 4 * a^2) : ∃ (n m : ℕ), 1 < n ∧ 1 < m ∧ (8 * b + 1) = n * m :=
by
  sorry

end NUMINAMATH_GPT_eight_b_plus_one_composite_l436_43699


namespace NUMINAMATH_GPT_katherine_age_l436_43652

-- Define a Lean statement equivalent to the given problem
theorem katherine_age (K M : ℕ) (h1 : M = K - 3) (h2 : M = 21) : K = 24 := sorry

end NUMINAMATH_GPT_katherine_age_l436_43652


namespace NUMINAMATH_GPT_lock_settings_are_5040_l436_43603

def num_unique_settings_for_lock : ℕ := 10 * 9 * 8 * 7

theorem lock_settings_are_5040 : num_unique_settings_for_lock = 5040 :=
by
  sorry

end NUMINAMATH_GPT_lock_settings_are_5040_l436_43603


namespace NUMINAMATH_GPT_max_n_perfect_cube_l436_43679

-- Definition for sum of squares
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Definition for sum of squares from (n+1) to 2n
def sum_of_squares_segment (n : ℕ) : ℕ :=
  2 * n * (2 * n + 1) * (4 * n + 1) / 6 - n * (n + 1) * (2 * n + 1) / 6

-- Definition for the product of the sums
def product_of_sums (n : ℕ) : ℕ :=
  (sum_of_squares n) * (sum_of_squares_segment n)

-- Predicate for perfect cube
def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y ^ 3 = x

-- The main theorem to be proved
theorem max_n_perfect_cube : ∃ (n : ℕ), n ≤ 2050 ∧ is_perfect_cube (product_of_sums n) ∧ ∀ m : ℕ, (m ≤ 2050 ∧ is_perfect_cube (product_of_sums m)) → m ≤ 2016 := 
sorry

end NUMINAMATH_GPT_max_n_perfect_cube_l436_43679


namespace NUMINAMATH_GPT_cost_per_steak_knife_l436_43674

theorem cost_per_steak_knife :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ),
  sets = 2 →
  knives_per_set = 4 →
  cost_per_set = 80 →
  (cost_per_set * sets) / (sets * knives_per_set) = 20 :=
by
  intros sets knives_per_set cost_per_set sets_eq knives_per_set_eq cost_per_set_eq
  rw [sets_eq, knives_per_set_eq, cost_per_set_eq]
  sorry

end NUMINAMATH_GPT_cost_per_steak_knife_l436_43674


namespace NUMINAMATH_GPT_cos_theta_plus_pi_over_3_l436_43664

theorem cos_theta_plus_pi_over_3 {θ : ℝ} (h : Real.sin (θ / 2 + π / 6) = 2 / 3) :
  Real.cos (θ + π / 3) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_theta_plus_pi_over_3_l436_43664


namespace NUMINAMATH_GPT_product_xyz_l436_43612

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) (h3 : x + 1 / z = 3) : x * y * z = 2 := 
by sorry

end NUMINAMATH_GPT_product_xyz_l436_43612


namespace NUMINAMATH_GPT_dozens_of_golf_balls_l436_43641

theorem dozens_of_golf_balls (total_balls : ℕ) (dozen_size : ℕ) (h1 : total_balls = 156) (h2 : dozen_size = 12) : total_balls / dozen_size = 13 :=
by
  have h_total : total_balls = 156 := h1
  have h_size : dozen_size = 12 := h2
  sorry

end NUMINAMATH_GPT_dozens_of_golf_balls_l436_43641


namespace NUMINAMATH_GPT_angle_B_in_triangle_l436_43657

/-- In triangle ABC, if BC = √3, AC = √2, and ∠A = π/3,
then ∠B = π/4. -/
theorem angle_B_in_triangle
  (BC AC : ℝ) (A B : ℝ)
  (hBC : BC = Real.sqrt 3)
  (hAC : AC = Real.sqrt 2)
  (hA : A = Real.pi / 3) :
  B = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_angle_B_in_triangle_l436_43657


namespace NUMINAMATH_GPT_toll_booth_ratio_l436_43644

theorem toll_booth_ratio (total_cars : ℕ) (monday_cars tuesday_cars friday_cars saturday_cars sunday_cars : ℕ)
  (x : ℕ) (h1 : total_cars = 450) (h2 : monday_cars = 50) (h3 : tuesday_cars = 50) (h4 : friday_cars = 50)
  (h5 : saturday_cars = 50) (h6 : sunday_cars = 50) (h7 : monday_cars + tuesday_cars + x + x + friday_cars + saturday_cars + sunday_cars = total_cars) :
  x = 100 ∧ x / monday_cars = 2 :=
by
  sorry

end NUMINAMATH_GPT_toll_booth_ratio_l436_43644


namespace NUMINAMATH_GPT_find_volume_of_pyramid_l436_43669

noncomputable def volume_of_pyramid
  (a : ℝ) (α : ℝ)
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) : ℝ :=
  (a ^ 3 * abs (Real.cos α)) / 3

--and the theorem to prove the statement
theorem find_volume_of_pyramid
  (a α : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) :
  volume_of_pyramid a α h1 h2 h3 = (a ^ 3 * abs (Real.cos α)) / 3 :=
sorry

end NUMINAMATH_GPT_find_volume_of_pyramid_l436_43669


namespace NUMINAMATH_GPT_john_paid_8000_l436_43667

-- Define the variables according to the conditions
def upfront_fee : ℕ := 1000
def hourly_rate : ℕ := 100
def court_hours : ℕ := 50
def prep_hours : ℕ := 2 * court_hours
def total_hours : ℕ := court_hours + prep_hours
def total_fee : ℕ := upfront_fee + total_hours * hourly_rate
def john_share : ℕ := total_fee / 2

-- Prove that John's share is $8,000
theorem john_paid_8000 : john_share = 8000 :=
by sorry

end NUMINAMATH_GPT_john_paid_8000_l436_43667


namespace NUMINAMATH_GPT_problem_D_l436_43695

theorem problem_D (a b c : ℝ) (h : |a^2 + b + c| + |a + b^2 - c| ≤ 1) : a^2 + b^2 + c^2 < 100 := 
sorry

end NUMINAMATH_GPT_problem_D_l436_43695


namespace NUMINAMATH_GPT_fewest_printers_l436_43618

theorem fewest_printers (x y : ℕ) (h : 8 * x = 7 * y) : x + y = 15 :=
sorry

end NUMINAMATH_GPT_fewest_printers_l436_43618


namespace NUMINAMATH_GPT_arithmetic_seq_a7_value_l436_43650

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ): Prop := 
  ∀ n : ℕ, a (n+1) = a n + d

theorem arithmetic_seq_a7_value
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 4 = 4)
  (h3 : a 3 + a 8 = 5) :
  a 7 = 1 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_value_l436_43650


namespace NUMINAMATH_GPT_quadratic_root_l436_43607

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end NUMINAMATH_GPT_quadratic_root_l436_43607


namespace NUMINAMATH_GPT_maximum_sum_of_O_and_square_l436_43651

theorem maximum_sum_of_O_and_square 
(O square : ℕ) (h1 : (O > 0) ∧ (square > 0)) 
(h2 : (O : ℚ) / 11 < (7 : ℚ) / (square))
(h3 : (7 : ℚ) / (square) < (4 : ℚ) / 5) : 
O + square = 18 :=
sorry

end NUMINAMATH_GPT_maximum_sum_of_O_and_square_l436_43651


namespace NUMINAMATH_GPT_nancy_balloons_l436_43668

variable (MaryBalloons : ℝ) (NancyBalloons : ℝ)

theorem nancy_balloons (h1 : NancyBalloons = 4 * MaryBalloons) (h2 : MaryBalloons = 1.75) : 
  NancyBalloons = 7 := 
by 
  sorry

end NUMINAMATH_GPT_nancy_balloons_l436_43668


namespace NUMINAMATH_GPT_spending_together_l436_43691

def sandwich_cost := 2
def hamburger_cost := 2
def hotdog_cost := 1
def juice_cost := 2
def selene_sandwiches := 3
def selene_juices := 1
def tanya_hamburgers := 2
def tanya_juices := 2

def selene_spending : ℕ := (selene_sandwiches * sandwich_cost) + (selene_juices * juice_cost)
def tanya_spending : ℕ := (tanya_hamburgers * hamburger_cost) + (tanya_juices * juice_cost)
def total_spending : ℕ := selene_spending + tanya_spending

theorem spending_together : total_spending = 16 :=
by
  sorry

end NUMINAMATH_GPT_spending_together_l436_43691


namespace NUMINAMATH_GPT_part_I_part_II_l436_43617

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_I (x : ℝ) : (f x 3) ≥ 1 ↔ (0 ≤ x ∧ x ≤ 4 / 3) :=
by sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x a - |2 * x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l436_43617


namespace NUMINAMATH_GPT_div_condition_positive_integers_l436_43696

theorem div_condition_positive_integers 
  (a b d : ℕ) 
  (h1 : a + b ≡ 0 [MOD d]) 
  (h2 : a * b ≡ 0 [MOD d^2]) 
  (h3 : 0 < a) 
  (h4 : 0 < b) 
  (h5 : 0 < d) : 
  d ∣ a ∧ d ∣ b :=
sorry

end NUMINAMATH_GPT_div_condition_positive_integers_l436_43696


namespace NUMINAMATH_GPT_math_problem_l436_43635

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l436_43635


namespace NUMINAMATH_GPT_weekly_goal_cans_l436_43610

theorem weekly_goal_cans : (20 +  (20 * 1.5) + (20 * 2) + (20 * 2.5) + (20 * 3)) = 200 := by
  sorry

end NUMINAMATH_GPT_weekly_goal_cans_l436_43610


namespace NUMINAMATH_GPT_solve_for_x_l436_43684

theorem solve_for_x (x : ℝ) (h : (6 * x ^ 2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1) : x = -18 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l436_43684


namespace NUMINAMATH_GPT_value_of_a_plus_d_l436_43654

theorem value_of_a_plus_d 
  (a b c d : ℤ)
  (h1 : a + b = 12) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) 
  : a + d = 9 := 
  sorry

end NUMINAMATH_GPT_value_of_a_plus_d_l436_43654


namespace NUMINAMATH_GPT_probability_two_students_same_school_l436_43633

/-- Definition of the problem conditions -/
def total_students : ℕ := 3
def total_schools : ℕ := 4
def total_basic_events : ℕ := total_schools ^ total_students
def favorable_events : ℕ := 36

/-- Theorem stating the probability of exactly two students choosing the same school -/
theorem probability_two_students_same_school : 
  favorable_events / (total_schools ^ total_students) = 9 / 16 := 
  sorry

end NUMINAMATH_GPT_probability_two_students_same_school_l436_43633


namespace NUMINAMATH_GPT_find_marks_in_chemistry_l436_43608

theorem find_marks_in_chemistry
  (marks_english : ℕ)
  (marks_math : ℕ)
  (marks_physics : ℕ)
  (marks_biology : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (marks_english_eq : marks_english = 86)
  (marks_math_eq : marks_math = 85)
  (marks_physics_eq : marks_physics = 92)
  (marks_biology_eq : marks_biology = 95)
  (average_marks_eq : average_marks = 89)
  (num_subjects_eq : num_subjects = 5) : 
  ∃ marks_chemistry : ℕ, marks_chemistry = 87 :=
by
  sorry

end NUMINAMATH_GPT_find_marks_in_chemistry_l436_43608


namespace NUMINAMATH_GPT_arithmetic_sqrt_9_l436_43638

def arithmetic_sqrt (x : ℕ) : ℕ :=
  if h : 0 ≤ x then Nat.sqrt x else 0

theorem arithmetic_sqrt_9 : arithmetic_sqrt 9 = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sqrt_9_l436_43638


namespace NUMINAMATH_GPT_compute_product_l436_43600

variable (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop := x^3 - 3 * x * y^2 = 2010
def condition2 (x y : ℝ) : Prop := y^3 - 3 * x^2 * y = 2000

theorem compute_product (h1 : condition1 x1 y1) (h2 : condition2 x1 y1)
    (h3 : condition1 x2 y2) (h4 : condition2 x2 y2)
    (h5 : condition1 x3 y3) (h6 : condition2 x3 y3) :
    (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 100 := 
    sorry

end NUMINAMATH_GPT_compute_product_l436_43600


namespace NUMINAMATH_GPT_cost_of_natural_seedless_raisins_l436_43604

theorem cost_of_natural_seedless_raisins
  (cost_golden: ℝ) (n_golden: ℕ) (n_natural: ℕ) (cost_mixture: ℝ) (cost_per_natural: ℝ) :
  cost_golden = 2.55 ∧ n_golden = 20 ∧ n_natural = 20 ∧ cost_mixture = 3
  → cost_per_natural = 3.45 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_natural_seedless_raisins_l436_43604


namespace NUMINAMATH_GPT_unit_vector_norm_diff_l436_43661

noncomputable def sqrt42_sqrt3_div_2 : ℝ := (Real.sqrt 42 * Real.sqrt 3) / 2
noncomputable def sqrt17_div_sqrt2 : ℝ := (Real.sqrt 17) / Real.sqrt 2

theorem unit_vector_norm_diff {x1 y1 z1 x2 y2 z2 : ℝ}
  (h1 : x1^2 + y1^2 + z1^2 = 1)
  (h2 : 3*x1 + y1 + 2*z1 = sqrt42_sqrt3_div_2)
  (h3 : 2*x1 + 2*y1 + 3*z1 = sqrt17_div_sqrt2)
  (h4 : x2^2 + y2^2 + z2^2 = 1)
  (h5 : 3*x2 + y2 + 2*z2 = sqrt42_sqrt3_div_2)
  (h6 : 2*x2 + 2*y2 + 3*z2 = sqrt17_div_sqrt2)
  (h_distinct : (x1, y1, z1) ≠ (x2, y2, z2)) :
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_unit_vector_norm_diff_l436_43661


namespace NUMINAMATH_GPT_lindsay_dolls_l436_43619

theorem lindsay_dolls (B B_b B_k : ℕ) 
  (h1 : B_b = 4 * B)
  (h2 : B_k = 4 * B - 2)
  (h3 : B_b + B_k = B + 26) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_lindsay_dolls_l436_43619


namespace NUMINAMATH_GPT_fill_tank_in_18_minutes_l436_43649

-- Define the conditions
def rate_pipe_A := 1 / 9  -- tanks per minute
def rate_pipe_B := - (1 / 18) -- tanks per minute (negative because it's emptying)

-- Define the net rate of both pipes working together
def net_rate := rate_pipe_A + rate_pipe_B

-- Define the time to fill the tank when both pipes are working
def time_to_fill_tank := 1 / net_rate

theorem fill_tank_in_18_minutes : time_to_fill_tank = 18 := 
    by
    -- Sorry to skip the actual proof
    sorry

end NUMINAMATH_GPT_fill_tank_in_18_minutes_l436_43649


namespace NUMINAMATH_GPT_proof_problem_l436_43662

theorem proof_problem
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2009)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2009)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2009) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l436_43662


namespace NUMINAMATH_GPT_MoneyDivision_l436_43665

theorem MoneyDivision (w x y z : ℝ)
  (hw : y = 0.5 * w)
  (hx : x = 0.7 * w)
  (hz : z = 0.3 * w)
  (hy : y = 90) :
  w + x + y + z = 450 := by
  sorry

end NUMINAMATH_GPT_MoneyDivision_l436_43665


namespace NUMINAMATH_GPT_sqrt_D_irrational_l436_43616

open Real

theorem sqrt_D_irrational (a : ℤ) (D : ℝ) (hD : D = a^2 + (a + 2)^2 + (a^2 + (a + 2))^2) : ¬ ∃ m : ℤ, D = m^2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_D_irrational_l436_43616


namespace NUMINAMATH_GPT_minimum_value_l436_43663

noncomputable def min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z))

theorem minimum_value : ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9 / 2 :=
by
  intro x y z hx hy hz
  sorry

end NUMINAMATH_GPT_minimum_value_l436_43663


namespace NUMINAMATH_GPT_lcm_10_to_30_l436_43606

def list_of_ints := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

def lcm_of_list (l : List Nat) : Nat :=
  l.foldr Nat.lcm 1

theorem lcm_10_to_30 : lcm_of_list list_of_ints = 232792560 :=
  sorry

end NUMINAMATH_GPT_lcm_10_to_30_l436_43606


namespace NUMINAMATH_GPT_downstream_speed_l436_43648

noncomputable def upstream_speed : ℝ := 5
noncomputable def still_water_speed : ℝ := 15

theorem downstream_speed:
  ∃ (Vd : ℝ), Vd = 25 ∧ (still_water_speed = (upstream_speed + Vd) / 2) := 
sorry

end NUMINAMATH_GPT_downstream_speed_l436_43648


namespace NUMINAMATH_GPT_find_a1_l436_43688

-- Given an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Arithmetic sequence is monotonically increasing
def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- First condition: sum of first three terms
def sum_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 12

-- Second condition: product of first three terms
def product_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 * a 1 * a 2 = 48

-- Proving that a_1 = 2 given the conditions
theorem find_a1 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : is_monotonically_increasing a)
  (h3 : sum_first_three_terms a) (h4 : product_first_three_terms a) : a 0 = 2 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_find_a1_l436_43688
