import Mathlib

namespace prove_inequality_l166_166320

noncomputable def inequality_holds (x y : ℝ) : Prop :=
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2)

theorem prove_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : inequality_holds x y :=
  sorry

end prove_inequality_l166_166320


namespace gym_hours_tuesday_equals_friday_l166_166060

-- Definitions
def weekly_gym_hours : ℝ := 5
def monday_hours : ℝ := 1.5
def wednesday_hours : ℝ := 1.5
def friday_hours : ℝ := 1
def total_weekly_hours : ℝ := weekly_gym_hours - (monday_hours + wednesday_hours + friday_hours)

-- Theorem statement
theorem gym_hours_tuesday_equals_friday : 
  total_weekly_hours = friday_hours :=
by
  sorry

end gym_hours_tuesday_equals_friday_l166_166060


namespace halfway_between_frac_l166_166282

theorem halfway_between_frac : (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end halfway_between_frac_l166_166282


namespace prob_2_out_of_5_exactly_A_and_B_l166_166641

noncomputable def probability_exactly_A_and_B_selected (students : List String) : ℚ :=
  if students = ["A", "B", "C", "D", "E"] then 1 / 10 else 0

theorem prob_2_out_of_5_exactly_A_and_B :
  probability_exactly_A_and_B_selected ["A", "B", "C", "D", "E"] = 1 / 10 :=
by 
  sorry

end prob_2_out_of_5_exactly_A_and_B_l166_166641


namespace find_x_for_prime_square_l166_166388

theorem find_x_for_prime_square (x p : ℤ) (hp : Prime p) (h : 2 * x^2 - x - 36 = p^2) : x = 13 ∧ p = 17 :=
by
  sorry

end find_x_for_prime_square_l166_166388


namespace pumpkin_weight_difference_l166_166732

theorem pumpkin_weight_difference (Brad: ℕ) (Jessica: ℕ) (Betty: ℕ) 
    (h1 : Brad = 54) 
    (h2 : Jessica = Brad / 2) 
    (h3 : Betty = Jessica * 4) 
    : (Betty - Jessica) = 81 := 
by
  sorry

end pumpkin_weight_difference_l166_166732


namespace rectangle_length_l166_166884

theorem rectangle_length (P B L : ℕ) (h1 : P = 800) (h2 : B = 300) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l166_166884


namespace mike_spending_l166_166126

noncomputable def marbles_cost : ℝ := 9.05
noncomputable def football_cost : ℝ := 4.95
noncomputable def baseball_cost : ℝ := 6.52

noncomputable def toy_car_original_cost : ℝ := 6.50
noncomputable def toy_car_discount : ℝ := 0.20
noncomputable def toy_car_discounted_cost : ℝ := toy_car_original_cost * (1 - toy_car_discount)

noncomputable def puzzle_cost : ℝ := 3.25
noncomputable def puzzle_total_cost : ℝ := puzzle_cost -- 'buy one get one free' condition

noncomputable def action_figure_original_cost : ℝ := 15.00
noncomputable def action_figure_discounted_cost : ℝ := 10.50

noncomputable def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + toy_car_discounted_cost + puzzle_total_cost + action_figure_discounted_cost

theorem mike_spending : total_cost = 39.47 := by
  sorry

end mike_spending_l166_166126


namespace sufficient_but_not_necessary_l166_166167

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬(∀ y : ℝ, (x < -1 ∨ y > 1) → (y < -1)) :=
by
  -- This means we would prove that if x < -1, then x < -1 ∨ x > 1 holds (sufficient),
  -- and show that there is a case (x > 1) where x < -1 is not necessary for x < -1 ∨ x > 1. 
  sorry

end sufficient_but_not_necessary_l166_166167


namespace intersection_is_singleton_l166_166973

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_is_singleton : M ∩ N = {1} :=
by sorry

end intersection_is_singleton_l166_166973


namespace fraction_multiplier_l166_166160

theorem fraction_multiplier (x y : ℝ) :
  (3 * x * 3 * y) / (3 * x + 3 * y) = 3 * (x * y) / (x + y) :=
by
  sorry

end fraction_multiplier_l166_166160


namespace average_of_last_three_numbers_l166_166163

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l166_166163


namespace monthly_earnings_l166_166915

theorem monthly_earnings (savings_per_month : ℤ) (total_needed : ℤ) (total_earned : ℤ)
  (H1 : savings_per_month = 500)
  (H2 : total_needed = 45000)
  (H3 : total_earned = 360000) :
  total_earned / (total_needed / savings_per_month) = 4000 := by
  sorry

end monthly_earnings_l166_166915


namespace bob_and_bill_same_class_probability_l166_166048

-- Definitions based on the conditions mentioned in the original problem
def total_people : ℕ := 32
def allowed_per_class : ℕ := 30
def number_chosen : ℕ := 2
def number_of_classes : ℕ := 2
def bob_and_bill_pair : ℕ := 1

-- Binomial coefficient calculation (32 choose 2)
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k
def total_ways := binomial_coefficient total_people number_chosen

-- Probability that Bob and Bill are chosen
def probability_chosen : ℚ := bob_and_bill_pair / total_ways

-- Probability that Bob and Bill are placed in the same class
def probability_same_class : ℚ := 1 / number_of_classes

-- Total combined probability
def combined_probability : ℚ := probability_chosen * probability_same_class

-- Statement of the theorem
theorem bob_and_bill_same_class_probability :
  combined_probability = 1 / 992 := 
sorry

end bob_and_bill_same_class_probability_l166_166048


namespace volume_of_cone_l166_166299

theorem volume_of_cone (d : ℝ) (h : ℝ) (r : ℝ) : 
  d = 10 ∧ h = 0.6 * d ∧ r = d / 2 → (1 / 3) * π * r^2 * h = 50 * π :=
by
  intro h1
  rcases h1 with ⟨h_d, h_h, h_r⟩
  sorry

end volume_of_cone_l166_166299


namespace quadratic_inequality_iff_abs_a_le_two_l166_166293

-- Definitions from the condition
variable (a : ℝ)
def quadratic_expr (x : ℝ) : ℝ := x^2 + a * x + 1

-- Statement of the problem as a Lean 4 statement
theorem quadratic_inequality_iff_abs_a_le_two :
  (∀ x : ℝ, quadratic_expr a x ≥ 0) ↔ (|a| ≤ 2) := sorry

end quadratic_inequality_iff_abs_a_le_two_l166_166293


namespace range_of_a_l166_166603

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 - 3 * a * x + 9 ≤ 0) → a ∈ Set.Ico 0 4 := by
  sorry

end range_of_a_l166_166603


namespace original_number_l166_166607

theorem original_number (x : ℕ) (h : ∃ k, 14 * x = 112 * k) : x = 8 :=
sorry

end original_number_l166_166607


namespace jelly_bean_ratio_l166_166357

-- Define the number of jelly beans each person has
def napoleon_jelly_beans : ℕ := 17
def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4
def mikey_jelly_beans : ℕ := 19

-- Define the sum of jelly beans of Napoleon and Sedrich
def sum_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

-- Define the ratio of the sum of Napoleon and Sedrich's jelly beans to Mikey's jelly beans
def ratio : ℚ := sum_jelly_beans / mikey_jelly_beans

-- Prove that the ratio is 2
theorem jelly_bean_ratio : ratio = 2 := by
  -- We skip the proof steps since the focus here is on the correct statement
  sorry

end jelly_bean_ratio_l166_166357


namespace find_m_value_l166_166242

theorem find_m_value (x y m : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x - 2 * y = m)
  (h3 : 2 * x - 3 * y = 1) : 
  m = 0 := 
sorry

end find_m_value_l166_166242


namespace arthur_speed_l166_166016

/-- Suppose Arthur drives to David's house and aims to arrive exactly on time. 
If he drives at 60 km/h, he arrives 5 minutes late. 
If he drives at 90 km/h, he arrives 5 minutes early. 
We want to find the speed n in km/h at which he arrives exactly on time. -/
theorem arthur_speed (n : ℕ) :
  (∀ t, 1 * (t + 5) = (3 / 2) * (t - 5)) → 
  (60 : ℝ) = 1 →
  (90 : ℝ) = (3 / 2) → 
  n = 72 := by
sorry

end arthur_speed_l166_166016


namespace largest_integer_among_four_l166_166344

theorem largest_integer_among_four 
  (x y z w : ℤ)
  (h1 : x + y + z = 234)
  (h2 : x + y + w = 255)
  (h3 : x + z + w = 271)
  (h4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := 
by
  -- This is a placeholder for the actual proof
  sorry

end largest_integer_among_four_l166_166344


namespace find_integers_l166_166639

theorem find_integers (x y : ℕ) (d : ℕ) (x1 y1 : ℕ) 
  (hx1 : x = d * x1) (hy1 : y = d * y1)
  (hgcd : Nat.gcd x y = d)
  (hcoprime : Nat.gcd x1 y1 = 1)
  (h1 : x1 + y1 = 18)
  (h2 : d * x1 * y1 = 975) : 
  ∃ (x y : ℕ), (Nat.gcd x y > 0) ∧ (x / Nat.gcd x y + y / Nat.gcd x y = 18) ∧ (Nat.lcm x y = 975) :=
sorry

end find_integers_l166_166639


namespace jenna_bill_eel_ratio_l166_166019

theorem jenna_bill_eel_ratio:
  ∀ (B : ℕ), (B + 16 = 64) → (16 / B = 1 / 3) :=
by
  intros B h
  sorry

end jenna_bill_eel_ratio_l166_166019


namespace ramesh_transport_cost_l166_166394

-- Definitions for conditions
def labelled_price (P : ℝ) : Prop := P = 13500 / 0.80
def selling_price (P : ℝ) : Prop := P * 1.10 = 18975
def transport_cost (T : ℝ) (extra_amount : ℝ) (installation_cost : ℝ) : Prop := T = extra_amount - installation_cost

-- The theorem statement to be proved
theorem ramesh_transport_cost (P T extra_amount installation_cost: ℝ) 
  (h1 : labelled_price P) 
  (h2 : selling_price P) 
  (h3 : extra_amount = 18975 - P)
  (h4 : installation_cost = 250) : 
  transport_cost T extra_amount installation_cost :=
by
  sorry

end ramesh_transport_cost_l166_166394


namespace max_bus_capacity_l166_166744

-- Definitions and conditions
def left_side_regular_seats := 12
def left_side_priority_seats := 3
def right_side_regular_seats := 9
def right_side_priority_seats := 2
def right_side_wheelchair_space := 1
def regular_seat_capacity := 3
def priority_seat_capacity := 2
def back_row_seat_capacity := 7
def standing_capacity := 14

-- Definition of total bus capacity
def total_bus_capacity : ℕ :=
  (left_side_regular_seats * regular_seat_capacity) + 
  (left_side_priority_seats * priority_seat_capacity) + 
  (right_side_regular_seats * regular_seat_capacity) + 
  (right_side_priority_seats * priority_seat_capacity) + 
  back_row_seat_capacity + 
  standing_capacity

-- Theorem to prove
theorem max_bus_capacity : total_bus_capacity = 94 := by
  -- skipping the proof
  sorry

end max_bus_capacity_l166_166744


namespace count_even_thousands_digit_palindromes_l166_166633

-- Define the set of valid digits
def valid_A : Finset ℕ := {2, 4, 6, 8}
def valid_B : Finset ℕ := Finset.range 10

-- Define the condition of a four-digit palindrome ABBA where A is even and non-zero
def is_valid_palindrome (a b : ℕ) : Prop :=
  a ∈ valid_A ∧ b ∈ valid_B

-- The proof problem: Prove that the total number of valid palindromes ABBA is 40
theorem count_even_thousands_digit_palindromes :
  (valid_A.card) * (valid_B.card) = 40 :=
by
  -- Skipping the proof itself
  sorry

end count_even_thousands_digit_palindromes_l166_166633


namespace find_x_l166_166037

theorem find_x (p q x : ℚ) (h1 : p / q = 4 / 5)
    (h2 : 4 / 7 + x / (2 * q + p) = 1) : x = 12 := 
by
  sorry

end find_x_l166_166037


namespace lara_gives_betty_l166_166092

variables (X Y : ℝ)

-- Conditions
-- Lara has spent X dollars
-- Betty has spent Y dollars
-- Y is greater than X
theorem lara_gives_betty (h : Y > X) : (Y - X) / 2 = (X + Y) / 2 - X :=
by
  sorry

end lara_gives_betty_l166_166092


namespace fraction_not_integer_l166_166042

theorem fraction_not_integer (a b : ℕ) (h : a ≠ b) (parity: (a % 2 = b % 2)) 
(h_pos_a : 0 < a) (h_pos_b : 0 < b) : ¬ ∃ k : ℕ, (a! + b!) = k * 2^a := 
by sorry

end fraction_not_integer_l166_166042


namespace arithmetic_progression_ratio_l166_166214

variable {α : Type*} [LinearOrder α] [Field α]

theorem arithmetic_progression_ratio (a d : α) (h : 15 * a + 105 * d = 3 * (8 * a + 28 * d)) : a / d = 7 / 3 := 
by sorry

end arithmetic_progression_ratio_l166_166214


namespace ruiz_original_salary_l166_166075

theorem ruiz_original_salary (S : ℝ) (h : 1.06 * S = 530) : S = 500 :=
by {
  -- Proof goes here
  sorry
}

end ruiz_original_salary_l166_166075


namespace no_fraternity_member_is_club_member_thm_l166_166867

-- Definitions from the conditions
variable (Person : Type)
variable (Club : Person → Prop)
variable (Honest : Person → Prop)
variable (Student : Person → Prop)
variable (Fraternity : Person → Prop)

-- Hypotheses from the problem statements
axiom all_club_members_honest (p : Person) : Club p → Honest p
axiom some_students_not_honest : ∃ p : Person, Student p ∧ ¬ Honest p
axiom no_fraternity_member_is_club_member (p : Person) : Fraternity p → ¬ Club p

-- The theorem to be proven
theorem no_fraternity_member_is_club_member_thm : 
  ∀ p : Person, Fraternity p → ¬ Club p := 
by 
  sorry

end no_fraternity_member_is_club_member_thm_l166_166867


namespace quadratic_vertex_coordinates_l166_166510

theorem quadratic_vertex_coordinates : ∀ x : ℝ,
  (∃ y : ℝ, y = 2 * x^2 - 4 * x + 5) →
  (1, 3) = (1, 3) :=
by
  intro x
  intro h
  sorry

end quadratic_vertex_coordinates_l166_166510


namespace original_price_of_shirts_l166_166839

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end original_price_of_shirts_l166_166839


namespace find_real_solutions_l166_166554

theorem find_real_solutions (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  ( (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) ) / ( (x - 4) * (x - 5) ) = -1 ↔ x = 10 / 3 ∨ x = 2 / 3 :=
by sorry

end find_real_solutions_l166_166554


namespace problem_1_problem_2_l166_166280

noncomputable section

variables {A B C : ℝ} {a b c : ℝ}

-- Condition definitions
def triangle_conditions (A B : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Problem 1: Prove that a = 2 * sqrt(3)
theorem problem_1 {A B C a : ℝ} (h : triangle_conditions A B a b c) : a = 2 * Real.sqrt 3 := sorry

-- Problem 2: Prove the value of cos(2A + π/6)
theorem problem_2 {A B C a : ℝ} (h : triangle_conditions A B a b c) : 
  Real.cos (2 * A + Real.pi / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := sorry

end problem_1_problem_2_l166_166280


namespace rooster_ratio_l166_166851

theorem rooster_ratio (R H : ℕ) 
  (h1 : R + H = 80)
  (h2 : R + (1 / 4) * H = 35) :
  R / 80 = 1 / 4 :=
  sorry

end rooster_ratio_l166_166851


namespace eval_f_at_3_l166_166027

def f (x : ℝ) : ℝ := 3 * x + 1

theorem eval_f_at_3 : f 3 = 10 :=
by
  -- computation of f at x = 3
  sorry

end eval_f_at_3_l166_166027


namespace find_number_l166_166183

theorem find_number :
  let f_add (a b : ℝ) : ℝ := a * b
  let f_sub (a b : ℝ) : ℝ := a + b
  let f_mul (a b : ℝ) : ℝ := a / b
  let f_div (a b : ℝ) : ℝ := a - b
  (f_div 9 8) * (f_mul 7 some_number) + (f_sub some_number 10) = 13.285714285714286 :=
  let some_number := 5
  sorry

end find_number_l166_166183


namespace dividend_calculation_l166_166926

theorem dividend_calculation :
  let divisor := 17
  let quotient := 9
  let remainder := 6
  let dividend := 159
  (divisor * quotient) + remainder = dividend :=
by
  sorry

end dividend_calculation_l166_166926


namespace eggs_in_box_l166_166692

-- Given conditions as definitions in Lean 4
def initial_eggs : ℕ := 7
def additional_whole_eggs : ℕ := 3

-- The proof statement
theorem eggs_in_box : initial_eggs + additional_whole_eggs = 10 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end eggs_in_box_l166_166692


namespace find_J_l166_166034

-- Define the problem conditions
def eq1 : Nat := 32
def eq2 : Nat := 4

-- Define the target equation form
def target_eq (J : Nat) : Prop := (eq1^3) * (eq2^3) = 2^J

theorem find_J : ∃ J : Nat, target_eq J ∧ J = 21 :=
by
  -- Rest of the proof goes here
  sorry

end find_J_l166_166034


namespace pow_sum_ge_mul_l166_166148

theorem pow_sum_ge_mul (m n : ℕ) : 2^(m + n - 2) ≥ m * n := 
sorry

end pow_sum_ge_mul_l166_166148


namespace eve_distance_ran_more_l166_166791

variable (ran walked : ℝ)

def eve_distance_difference (ran walked : ℝ) : ℝ :=
  ran - walked

theorem eve_distance_ran_more :
  eve_distance_difference 0.7 0.6 = 0.1 :=
by
  sorry

end eve_distance_ran_more_l166_166791


namespace certain_number_is_1_l166_166416

theorem certain_number_is_1 (z : ℕ) (hz : z % 4 = 0) :
  ∃ n : ℕ, (z * (6 + z) + n) % 2 = 1 ∧ n = 1 :=
by
  sorry

end certain_number_is_1_l166_166416


namespace gcd_is_13_eval_at_neg1_l166_166405

-- Define the GCD problem
def gcd_117_182 : ℕ := gcd 117 182

-- Define the polynomial evaluation problem
def f (x : ℝ) : ℝ := 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

-- Formalize the statements to be proved
theorem gcd_is_13 : gcd_117_182 = 13 := 
by sorry

theorem eval_at_neg1 : f (-1) = 12 := 
by sorry

end gcd_is_13_eval_at_neg1_l166_166405


namespace find_first_remainder_l166_166108

theorem find_first_remainder (N : ℕ) (R₁ R₂ : ℕ) (h1 : N = 184) (h2 : N % 15 = R₂) (h3 : R₂ = 4) : 
  N % 13 = 2 :=
by
  sorry

end find_first_remainder_l166_166108


namespace probability_of_selecting_one_defective_l166_166722

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l166_166722


namespace hyperbola_slope_reciprocals_l166_166954

theorem hyperbola_slope_reciprocals (P : ℝ × ℝ) (t : ℝ) :
  (P.1 = t ∧ P.2 = - (8 / 9) * t ∧ t ≠ 0 ∧  
    ∃ k1 k2: ℝ, k1 = - (8 * t) / (9 * (t + 3)) ∧ k2 = - (8 * t) / (9 * (t - 3)) ∧
    (1 / k1) + (1 / k2) = -9 / 4) ∧
    ((P = (9/5, -(8/5)) ∨ P = (-(9/5), 8/5)) →
        ∃ kOA kOB kOC kOD : ℝ, (kOA + kOB + kOC + kOD = 0)) := 
sorry

end hyperbola_slope_reciprocals_l166_166954


namespace milk_students_l166_166309

theorem milk_students (T : ℕ) (h1 : (1 / 4) * T = 80) : (3 / 4) * T = 240 := by
  sorry

end milk_students_l166_166309


namespace tan_double_beta_alpha_value_l166_166195

open Real

-- Conditions
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def beta_in_interval (β : ℝ) : Prop := π / 2 < β ∧ β < π
def cos_beta (β : ℝ) : Prop := cos β = -1 / 3
def sin_alpha_plus_beta (α β : ℝ) : Prop := sin (α + β) = (4 - sqrt 2) / 6

-- Proof problem 1: Prove that tan 2β = 4√2 / 7 given the conditions
theorem tan_double_beta (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  tan (2 * β) = (4 * sqrt 2) / 7 :=
by sorry

-- Proof problem 2: Prove that α = π / 4 given the conditions
theorem alpha_value (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  α = π / 4 :=
by sorry

end tan_double_beta_alpha_value_l166_166195


namespace magnitude_of_power_l166_166547

noncomputable def z : ℂ := 4 + 2 * Real.sqrt 2 * Complex.I

theorem magnitude_of_power :
  Complex.abs (z ^ 4) = 576 := by
  sorry

end magnitude_of_power_l166_166547


namespace initial_tree_height_l166_166856

-- Definition of the problem conditions as Lean definitions.
def quadruple (x : ℕ) : ℕ := 4 * x

-- Given conditions of the problem
def final_height : ℕ := 256
def height_increase_each_year (initial_height : ℕ) : Prop :=
  quadruple (quadruple (quadruple (quadruple initial_height))) = final_height

-- The proof statement that we need to prove
theorem initial_tree_height 
  (initial_height : ℕ)
  (h : height_increase_each_year initial_height)
  : initial_height = 1 := sorry

end initial_tree_height_l166_166856


namespace rhombus_longer_diagonal_l166_166878

theorem rhombus_longer_diagonal (a b d_1 : ℝ) (h_side : a = 60) (h_d1 : d_1 = 56) :
  ∃ d_2, d_2 = 106 := by
  sorry

end rhombus_longer_diagonal_l166_166878


namespace banana_cost_l166_166870

/-- If 4 bananas cost $20, then the cost of one banana is $5. -/
theorem banana_cost (total_cost num_bananas : ℕ) (cost_per_banana : ℕ) 
  (h : total_cost = 20 ∧ num_bananas = 4) : cost_per_banana = 5 := by
  sorry

end banana_cost_l166_166870


namespace polygon_diagonals_twice_sides_l166_166333

theorem polygon_diagonals_twice_sides
  (n : ℕ)
  (h : n * (n - 3) / 2 = 2 * n) :
  n = 7 :=
sorry

end polygon_diagonals_twice_sides_l166_166333


namespace sum_of_common_ratios_l166_166662

noncomputable def geometric_sequence (m x : ℝ) : ℝ × ℝ × ℝ := (m, m * x, m * x^2)

theorem sum_of_common_ratios
  (m x y : ℝ)
  (h1 : x ≠ y)
  (h2 : m ≠ 0)
  (h3 : ∃ c3 c2 d3 d2 : ℝ, geometric_sequence m x = (m, c2, c3) ∧ geometric_sequence m y = (m, d2, d3) ∧ c3 - d3 = 3 * (c2 - d2)) :
  x + y = 3 := by
  sorry

end sum_of_common_ratios_l166_166662


namespace ticket_queue_correct_l166_166659

-- Define the conditions
noncomputable def ticket_queue_count (m n : ℕ) (h : n ≥ m) : ℕ :=
  (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))

-- State the theorem
theorem ticket_queue_correct (m n : ℕ) (h : n ≥ m) :
  ticket_queue_count m n h = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by
  sorry

end ticket_queue_correct_l166_166659


namespace house_transaction_l166_166628

variable (initial_value : ℝ) (loss_rate : ℝ) (gain_rate : ℝ) (final_loss : ℝ)

theorem house_transaction
  (h_initial : initial_value = 12000)
  (h_loss : loss_rate = 0.15)
  (h_gain : gain_rate = 0.15)
  (h_final_loss : final_loss = 270) :
  let selling_price := initial_value * (1 - loss_rate)
  let buying_price := selling_price * (1 + gain_rate)
  (initial_value - buying_price) = final_loss :=
by
  simp only [h_initial, h_loss, h_gain, h_final_loss]
  sorry

end house_transaction_l166_166628


namespace total_flowering_bulbs_count_l166_166267

-- Definitions for the problem conditions
def crocus_cost : ℝ := 0.35
def daffodil_cost : ℝ := 0.65
def total_budget : ℝ := 29.15
def crocus_count : ℕ := 22

-- Theorem stating the total number of bulbs that can be bought
theorem total_flowering_bulbs_count : 
  ∃ daffodil_count : ℕ, (crocus_count + daffodil_count = 55) ∧ (total_budget = crocus_cost * crocus_count + daffodil_count * daffodil_cost) :=
  sorry

end total_flowering_bulbs_count_l166_166267


namespace combination_equality_l166_166489

theorem combination_equality : 
  Nat.choose 5 2 + Nat.choose 5 3 = 20 := 
by 
  sorry

end combination_equality_l166_166489


namespace mat_weavers_proof_l166_166116

def mat_weavers_rate
  (num_weavers_1 : ℕ) (num_mats_1 : ℕ) (num_days_1 : ℕ)
  (num_mats_2 : ℕ) (num_days_2 : ℕ) : ℕ :=
  let rate_per_weaver_per_day := num_mats_1 / (num_weavers_1 * num_days_1)
  let num_weavers_2 := num_mats_2 / (rate_per_weaver_per_day * num_days_2)
  num_weavers_2

theorem mat_weavers_proof :
  mat_weavers_rate 4 4 4 36 12 = 12 := by
  sorry

end mat_weavers_proof_l166_166116


namespace sequence_formula_l166_166773

open Nat

noncomputable def S : ℕ → ℤ
| n => n^2 - 2 * n + 2

noncomputable def a : ℕ → ℤ
| 0 => 1  -- note that in Lean, sequence indexing starts from 0
| (n+1) => 2*(n+1) - 3

theorem sequence_formula (n : ℕ) : 
  a n = if n = 0 then 1 else 2*n - 3 := by
  sorry

end sequence_formula_l166_166773


namespace sequence_sum_identity_l166_166636

theorem sequence_sum_identity 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ)
  (h1 : ∀ n, b_n n - a_n n = 2^n + 1)
  (h2 : ∀ n, S_n n + T_n n = 2^(n+1) + n^2 - 2) : 
  ∀ n, 2 * T_n n = n * (n - 1) :=
by sorry

end sequence_sum_identity_l166_166636


namespace ticket_price_increase_l166_166537

noncomputable def y (x : ℕ) : ℝ :=
  if x ≤ 100 then
    30 * x - 50 * Real.sqrt x - 500
  else
    30 * x - 50 * Real.sqrt x - 700

theorem ticket_price_increase (m : ℝ) : 
  m * 20 - 50 * Real.sqrt 20 - 500 ≥ 0 → m ≥ 37 := sorry

end ticket_price_increase_l166_166537


namespace evaporation_rate_l166_166413

theorem evaporation_rate (initial_water_volume : ℕ) (days : ℕ) (percentage_evaporated : ℕ) (evaporated_fraction : ℚ)
  (h1 : initial_water_volume = 10)
  (h2 : days = 50)
  (h3 : percentage_evaporated = 3)
  (h4 : evaporated_fraction = percentage_evaporated / 100) :
  (initial_water_volume * evaporated_fraction) / days = 0.06 :=
by
  -- Proof goes here
  sorry

end evaporation_rate_l166_166413


namespace num_members_in_league_l166_166370

-- Definitions based on conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def shorts_cost : ℕ := tshirt_cost
def total_cost_per_member : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)
def total_league_cost : ℕ := 4719

-- Theorem statement
theorem num_members_in_league : (total_league_cost / total_cost_per_member) = 74 :=
by
  sorry

end num_members_in_league_l166_166370


namespace log_suff_nec_l166_166311

theorem log_suff_nec (a b : ℝ) (ha : a > 0) (hb : b > 0) : ¬ ((a > b) ↔ (Real.log b / Real.log a < 1)) := 
sorry

end log_suff_nec_l166_166311


namespace find_x2_y2_l166_166845

theorem find_x2_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x^2 * y + x * y^2 = 1512) :
  x^2 + y^2 = 1136 ∨ x^2 + y^2 = 221 := by
  sorry

end find_x2_y2_l166_166845


namespace no_upper_bound_l166_166452

-- Given Conditions
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {M : ℝ}

-- Condition: widths and lengths of plates are 1 and a1, a2, a3, ..., respectively
axiom width_1 : ∀ n, (S n > 0)

-- Condition: a1 ≠ 1
axiom a1_neq_1 : a 1 ≠ 1

-- Condition: plates are similar but not congruent starting from the second
axiom similar_not_congruent : ∀ n > 1, (a (n+1) > a n)

-- Condition: S_n denotes the length covered after placing n plates
axiom Sn_length : ∀ n, S (n+1) = S n + a (n+1)

-- Condition: a_{n+1} = 1 / S_n
axiom an_reciprocal : ∀ n, a (n+1) = 1 / S n

-- The final goal: no such real number exists that S_n does not exceed
theorem no_upper_bound : ∀ M : ℝ, ∃ n : ℕ, S n > M := 
sorry

end no_upper_bound_l166_166452


namespace license_plate_combinations_l166_166257

theorem license_plate_combinations : 
  let letters := 26 
  let letters_and_digits := 36 
  let middle_character_choices := 2
  3 * letters * letters_and_digits * middle_character_choices = 1872 :=
by
  sorry

end license_plate_combinations_l166_166257


namespace find_f_0_plus_f_neg_1_l166_166672

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - x^2 else
if x < 0 then -(2^(-x) - (-x)^2) else 0

theorem find_f_0_plus_f_neg_1 : f 0 + f (-1) = -1 := by
  sorry

end find_f_0_plus_f_neg_1_l166_166672


namespace total_hatched_eggs_l166_166918

noncomputable def fertile_eggs (total_eggs : ℕ) (infertility_rate : ℝ) : ℝ :=
  total_eggs * (1 - infertility_rate)

noncomputable def hatching_eggs_after_calcification (fertile_eggs : ℝ) (calcification_rate : ℝ) : ℝ :=
  fertile_eggs * (1 - calcification_rate)

noncomputable def hatching_eggs_after_predator (hatching_eggs : ℝ) (predator_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - predator_rate)

noncomputable def hatching_eggs_after_temperature (hatching_eggs : ℝ) (temperature_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - temperature_rate)

open Nat

theorem total_hatched_eggs :
  let g1_total_eggs := 30
  let g2_total_eggs := 40
  let g1_infertility_rate := 0.20
  let g2_infertility_rate := 0.25
  let g1_calcification_rate := 1.0 / 3.0
  let g2_calcification_rate := 0.25
  let predator_rate := 0.10
  let temperature_rate := 0.05
  let g1_fertile := fertile_eggs g1_total_eggs g1_infertility_rate
  let g1_hatch_calcification := hatching_eggs_after_calcification g1_fertile g1_calcification_rate
  let g1_hatch_predator := hatching_eggs_after_predator g1_hatch_calcification predator_rate
  let g1_hatch_temp := hatching_eggs_after_temperature g1_hatch_predator temperature_rate
  let g2_fertile := fertile_eggs g2_total_eggs g2_infertility_rate
  let g2_hatch_calcification := hatching_eggs_after_calcification g2_fertile g2_calcification_rate
  let g2_hatch_predator := hatching_eggs_after_predator g2_hatch_calcification predator_rate
  let g2_hatch_temp := hatching_eggs_after_temperature g2_hatch_predator temperature_rate
  let total_hatched := g1_hatch_temp + g2_hatch_temp
  floor total_hatched = 32 :=
by
  sorry

end total_hatched_eggs_l166_166918


namespace find_constant_x_geom_prog_l166_166450

theorem find_constant_x_geom_prog (x : ℝ) :
  (30 + x) ^ 2 = (10 + x) * (90 + x) → x = 0 :=
by
  -- Proof omitted
  sorry

end find_constant_x_geom_prog_l166_166450


namespace value_of_m_l166_166686

theorem value_of_m (m x : ℝ) (h : x - 4 ≠ 0) (hx_pos : x > 0) 
  (eqn : m / (x - 4) - (1 - x) / (4 - x) = 0) : m = 3 := 
by
  sorry

end value_of_m_l166_166686


namespace negation_correct_l166_166530

-- Define the initial proposition
def initial_proposition : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 - 2 * x > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 - 2 * x ≤ 0

-- Statement of the theorem
theorem negation_correct :
  (¬ initial_proposition) = negated_proposition :=
by
  sorry

end negation_correct_l166_166530


namespace dave_fifth_store_car_count_l166_166159

theorem dave_fifth_store_car_count :
  let cars_first_store := 30
  let cars_second_store := 14
  let cars_third_store := 14
  let cars_fourth_store := 21
  let mean := 20.8
  let total_cars := mean * 5
  let total_cars_first_four := cars_first_store + cars_second_store + cars_third_store + cars_fourth_store
  total_cars - total_cars_first_four = 25 := by
sorry

end dave_fifth_store_car_count_l166_166159


namespace prism_diagonal_and_surface_area_l166_166194

/-- 
  A rectangular prism has dimensions of 12 inches, 16 inches, and 21 inches.
  Prove that the length of the diagonal is 29 inches, 
  and the total surface area of the prism is 1560 square inches.
-/
theorem prism_diagonal_and_surface_area :
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  d = 29 ∧ S = 1560 := by
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  sorry

end prism_diagonal_and_surface_area_l166_166194


namespace Jakes_weight_is_198_l166_166538

variable (Jake Kendra : ℕ)

-- Conditions
variable (h1 : Jake - 8 = 2 * Kendra)
variable (h2 : Jake + Kendra = 293)

theorem Jakes_weight_is_198 : Jake = 198 :=
by
  sorry

end Jakes_weight_is_198_l166_166538


namespace sums_have_same_remainder_l166_166360

theorem sums_have_same_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) : 
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i.val) % (2 * n) = (a j + j.val) % (2 * n)) := 
sorry

end sums_have_same_remainder_l166_166360


namespace quadratic_positive_range_l166_166472

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) ↔ ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) := 
by {
  sorry
}

end quadratic_positive_range_l166_166472


namespace range_of_t_l166_166498

def ellipse (x y t : ℝ) : Prop := (x^2) / 4 + (y^2) / t = 1

def distance_greater_than_one (x y t : ℝ) : Prop := 
  let a := if t > 4 then Real.sqrt t else 2
  let b := if t > 4 then 2 else Real.sqrt t
  let c := if t > 4 then Real.sqrt (t - 4) else Real.sqrt (4 - t)
  a - c > 1

theorem range_of_t (t : ℝ) : 
  (∀ x y, ellipse x y t → distance_greater_than_one x y t) ↔ 
  (3 < t ∧ t < 4) ∨ (4 < t ∧ t < 25 / 4) := 
sorry

end range_of_t_l166_166498


namespace distance_to_hospital_l166_166911

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end distance_to_hospital_l166_166911


namespace no_minimum_value_l166_166448

noncomputable def f (x : ℝ) : ℝ :=
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) - x)) *
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

theorem no_minimum_value : ¬ ∃ x, (0 < x ∧ x < 4.5) ∧ (∀ y, (0 < y ∧ y < 4.5) → f x ≤ f y) :=
sorry

end no_minimum_value_l166_166448


namespace ratio_of_kids_to_adult_meals_l166_166542

theorem ratio_of_kids_to_adult_meals (k a : ℕ) (h1 : k = 8) (h2 : k + a = 12) : k / a = 2 := 
by 
  sorry

end ratio_of_kids_to_adult_meals_l166_166542


namespace smallest_x_y_sum_l166_166513

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≠ y) (h_fraction : 1/x + 1/y = 1/20) : x + y = 90 :=
sorry

end smallest_x_y_sum_l166_166513


namespace value_three_in_range_of_g_l166_166374

theorem value_three_in_range_of_g (a : ℝ) : ∀ (a : ℝ), ∃ (x : ℝ), x^2 + a * x + 1 = 3 :=
by
  sorry

end value_three_in_range_of_g_l166_166374


namespace cubic_difference_l166_166658

theorem cubic_difference (x y : ℝ) (h1 : x + y = 15) (h2 : 2 * x + y = 20) : x^3 - y^3 = -875 := 
by
  sorry

end cubic_difference_l166_166658


namespace factor_polynomial_equiv_l166_166024

theorem factor_polynomial_equiv :
  (x^2 + 2 * x + 1) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 7 * x + 1) * (x^2 + 3 * x + 7) :=
by sorry

end factor_polynomial_equiv_l166_166024


namespace a_gt_b_l166_166714

theorem a_gt_b (n : ℕ) (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hn_ge_two : n ≥ 2)
  (ha_eq : a^n = a + 1) (hb_eq : b^(2*n) = b + 3 * a) : a > b :=
by
  sorry

end a_gt_b_l166_166714


namespace line_passes_through_fixed_point_l166_166069

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : x = 6) 
  (h3 : y = 4) : 
  a * x + b * y - 10 = 0 := 
by
  sorry

end line_passes_through_fixed_point_l166_166069


namespace find_value_of_a_l166_166495

noncomputable def value_of_a (a : ℝ) (hyp_asymptotes_tangent_circle : Prop) : Prop :=
  a = (Real.sqrt 3) / 3 → hyp_asymptotes_tangent_circle

theorem find_value_of_a (a : ℝ) (condition1 : 0 < a)
  (condition_hyperbola : ∀ x y, x^2 / a^2 - y^2 = 1)
  (condition_circle : ∀ x y, x^2 + y^2 - 4*y + 3 = 0)
  (hyp_asymptotes_tangent_circle : Prop) :
  value_of_a a hyp_asymptotes_tangent_circle := 
sorry

end find_value_of_a_l166_166495


namespace exists_An_Bn_l166_166181

theorem exists_An_Bn (n : ℕ) : ∃ (A_n B_n : ℕ), (3 - Real.sqrt 7) ^ n = A_n - B_n * Real.sqrt 7 := by
  sorry

end exists_An_Bn_l166_166181


namespace area_of_region_l166_166313

theorem area_of_region :
  ∫ y in (0:ℝ)..(1:ℝ), y ^ (2 / 3) = 3 / 5 :=
by
  sorry

end area_of_region_l166_166313


namespace cost_unit_pen_max_profit_and_quantity_l166_166676

noncomputable def cost_pen_A : ℝ := 5
noncomputable def cost_pen_B : ℝ := 10
noncomputable def profit_pen_A : ℝ := 2
noncomputable def profit_pen_B : ℝ := 3
noncomputable def spent_on_A : ℝ := 400
noncomputable def spent_on_B : ℝ := 800
noncomputable def total_pens : ℝ := 300

theorem cost_unit_pen : (spent_on_A / cost_pen_A) = (spent_on_B / (cost_pen_A + 5)) := by
  sorry

theorem max_profit_and_quantity
    (xa xb : ℝ)
    (h1 : xa ≥ 4 * xb)
    (h2 : xa + xb = total_pens)
    : ∃ (wa : ℝ), wa = 2 * xa + 3 * xb ∧ xa = 240 ∧ xb = 60 ∧ wa = 660 := by
  sorry

end cost_unit_pen_max_profit_and_quantity_l166_166676


namespace lemons_for_10_gallons_l166_166698

noncomputable def lemon_proportion : Prop :=
  ∃ x : ℝ, (36 / 48) = (x / 10) ∧ x = 7.5

theorem lemons_for_10_gallons : lemon_proportion :=
by
  sorry

end lemons_for_10_gallons_l166_166698


namespace equilateral_triangle_side_length_l166_166846

theorem equilateral_triangle_side_length (a : ℝ) (h : 3 * a = 18) : a = 6 :=
by
  sorry

end equilateral_triangle_side_length_l166_166846


namespace find_dividend_l166_166670

theorem find_dividend (D Q R dividend : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) (h4 : dividend = D * Q + R) :
  dividend = 5336 :=
by
  -- We will complete the proof using the provided conditions
  sorry

end find_dividend_l166_166670


namespace adam_spent_on_ferris_wheel_l166_166528

-- Define the conditions
def ticketsBought : Nat := 13
def ticketsLeft : Nat := 4
def costPerTicket : Nat := 9

-- Define the question and correct answer as a proof goal
theorem adam_spent_on_ferris_wheel : (ticketsBought - ticketsLeft) * costPerTicket = 81 := by
  sorry

end adam_spent_on_ferris_wheel_l166_166528


namespace reese_practice_hours_l166_166696

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l166_166696


namespace intersection_complement_l166_166240

open Set

variable (U A B : Set ℕ)

-- Definitions based on conditions given in the problem
def universal_set : Set ℕ := {1, 2, 3, 4, 5}
def set_A : Set ℕ := {2, 4}
def set_B : Set ℕ := {4, 5}

-- Proof statement
theorem intersection_complement :
  A = set_A → 
  B = set_B → 
  U = universal_set → 
  (A ∩ (U \ B)) = {2} := 
by
  intros hA hB hU
  sorry

end intersection_complement_l166_166240


namespace marvin_next_birthday_monday_l166_166321

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def day_of_week_after_leap_years (start_day : ℕ) (leap_years : ℕ) : ℕ :=
  (start_day + 2 * leap_years) % 7

def next_birthday_on_monday (year : ℕ) (start_day : ℕ) : ℕ :=
  let next_day := day_of_week_after_leap_years start_day ((year - 2012)/4)
  year + 4 * ((7 - next_day + 1) / 2)

theorem marvin_next_birthday_monday : next_birthday_on_monday 2012 3 = 2016 :=
by sorry

end marvin_next_birthday_monday_l166_166321


namespace area_of_region_l166_166611

theorem area_of_region (x y : ℝ) (h : x^2 + y^2 + 6 * x - 8 * y - 5 = 0) : 
  ∃ (r : ℝ), (π * r^2 = 30 * π) :=
by -- Starting the proof, skipping the detailed steps
sorry -- Proof placeholder

end area_of_region_l166_166611


namespace base8_to_base10_l166_166598

theorem base8_to_base10 {a b : ℕ} (h1 : 3 * 64 + 7 * 8 + 4 = 252) (h2 : 252 = a * 10 + b) :
  (a + b : ℝ) / 20 = 0.35 :=
sorry

end base8_to_base10_l166_166598


namespace social_media_phone_ratio_l166_166323

/-- 
Given that Jonathan spends 8 hours on his phone daily and 28 hours on social media in a week, 
prove that the ratio of the time spent on social media to the total time spent on his phone daily is \( 1 : 2 \).
-/
theorem social_media_phone_ratio (daily_phone_hours : ℕ) (weekly_social_media_hours : ℕ) 
  (h1 : daily_phone_hours = 8) (h2 : weekly_social_media_hours = 28) :
  (weekly_social_media_hours / 7) / daily_phone_hours = 1 / 2 := 
by
  sorry

end social_media_phone_ratio_l166_166323


namespace scout_troop_net_profit_l166_166356

theorem scout_troop_net_profit :
  ∃ (cost_per_bar selling_price_per_bar : ℝ),
    cost_per_bar = 1 / 3 ∧
    selling_price_per_bar = 0.6 ∧
    (1500 * selling_price_per_bar - (1500 * cost_per_bar + 50) = 350) :=
by {
  sorry
}

end scout_troop_net_profit_l166_166356


namespace max_points_on_poly_graph_l166_166125

theorem max_points_on_poly_graph (P : Polynomial ℤ) (h_deg : P.degree = 20):
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, 0 ≤ p.snd ∧ p.snd ≤ 10) ∧ S.card ≤ 20 ∧ 
  ∀ S' : Finset (ℤ × ℤ), (∀ p ∈ S', 0 ≤ p.snd ∧ p.snd ≤ 10) → S'.card ≤ 20 :=
by
  sorry

end max_points_on_poly_graph_l166_166125


namespace acute_triangle_l166_166591

theorem acute_triangle (r R : ℝ) (h : R < r * (Real.sqrt 2 + 1)) : 
  ∃ (α β γ : ℝ), α + β + γ = π ∧ (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ (α < π / 2) ∧ (β < π / 2) ∧ (γ < π / 2) := 
sorry

end acute_triangle_l166_166591


namespace sum_of_irreducible_fractions_is_integer_iff_same_denominator_l166_166994

theorem sum_of_irreducible_fractions_is_integer_iff_same_denominator
  (a b c d A : ℤ) (h_irred1 : Int.gcd a b = 1) (h_irred2 : Int.gcd c d = 1) (h_sum : (a : ℚ) / b + (c : ℚ) / d = A) :
  b = d := 
by
  sorry

end sum_of_irreducible_fractions_is_integer_iff_same_denominator_l166_166994


namespace inequality_condition_l166_166032

theorem inequality_condition (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 :=
sorry

end inequality_condition_l166_166032


namespace inequality_l166_166225

-- Given three distinct positive real numbers a, b, c
variables {a b c : ℝ}

-- Assume a, b, and c are distinct and positive
axiom distinct_positive (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) : 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- The inequality to be proven
theorem inequality (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) :
  (a / b) + (b / c) > (a / c) + (c / a) := 
sorry

end inequality_l166_166225


namespace sum_y_coordinates_of_other_vertices_l166_166190

theorem sum_y_coordinates_of_other_vertices (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 10)) (h2 : (x2, y2) = (-6, -6)) :
  (∃ y3 y4 : ℤ, (4 : ℤ) = y3 + y4) :=
by
  sorry

end sum_y_coordinates_of_other_vertices_l166_166190


namespace chairs_removal_correct_chairs_removal_l166_166090

theorem chairs_removal (initial_chairs : ℕ) (chairs_per_row : ℕ) (participants : ℕ) : ℕ :=
  let total_chairs := 169
  let per_row := 13
  let attendees := 95
  let needed_chairs := (attendees + per_row - 1) / per_row * per_row
  let chairs_to_remove := total_chairs - needed_chairs
  chairs_to_remove

theorem correct_chairs_removal : chairs_removal 169 13 95 = 65 :=
by
  sorry

end chairs_removal_correct_chairs_removal_l166_166090


namespace geom_prog_roots_a_eq_22_l166_166492

theorem geom_prog_roots_a_eq_22 (x1 x2 x3 a : ℝ) :
  (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) → 
  (∃ b q, (x1 = b ∧ x2 = b * q ∧ x3 = b * q^2) ∧ (x1 + x2 + x3 = 11) ∧ (x1 * x2 * x3 = 8) ∧ (x1*x2 + x2*x3 + x3*x1 = a)) → 
  a = 22 :=
sorry

end geom_prog_roots_a_eq_22_l166_166492


namespace terminating_decimals_l166_166795

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l166_166795


namespace find_cos_E_floor_l166_166912

theorem find_cos_E_floor (EF GH EH FG : ℝ) (E G : ℝ) 
  (h1 : EF = 200) 
  (h2 : GH = 200) 
  (h3 : EH ≠ FG) 
  (h4 : EF + GH + EH + FG = 800) 
  (h5 : E = G) : 
  (⌊1000 * Real.cos E⌋ = 1000) := 
by 
  sorry

end find_cos_E_floor_l166_166912


namespace avg_price_of_racket_l166_166109

theorem avg_price_of_racket (total_revenue : ℝ) (pairs_sold : ℝ) (h1 : total_revenue = 686) (h2 : pairs_sold = 70) : 
  total_revenue / pairs_sold = 9.8 := by
  sorry

end avg_price_of_racket_l166_166109


namespace initial_men_is_250_l166_166837

-- Define the given conditions
def provisions (initial_men remaining_men initial_days remaining_days : ℕ) : Prop :=
  initial_men * initial_days = remaining_men * remaining_days

-- Define the problem statement
theorem initial_men_is_250 (initial_days remaining_days : ℕ) (remaining_men_leaving : ℕ) :
  provisions initial_men (initial_men - remaining_men_leaving) initial_days remaining_days → initial_men = 250 :=
by
  intros h
  -- Requirement to solve the theorem.
  -- This is where the proof steps would go, but we put sorry to satisfy the statement requirement.
  sorry

end initial_men_is_250_l166_166837


namespace abs_twice_sub_pi_l166_166384

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l166_166384


namespace parametric_equation_solution_l166_166818

noncomputable def solve_parametric_equation (a b : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) : ℝ :=
  (5 / (a - 2 * b))

theorem parametric_equation_solution (a b x : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) 
  (h : (a * x - 3) / (b * x + 1) = 2) : 
  x = solve_parametric_equation a b ha2b ha3b :=
sorry

end parametric_equation_solution_l166_166818


namespace total_toys_is_correct_l166_166782

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end total_toys_is_correct_l166_166782


namespace determine_angle_range_l166_166668

variable (α : ℝ)

theorem determine_angle_range 
  (h1 : 0 < α) 
  (h2 : α < 2 * π) 
  (h_sin : Real.sin α < 0) 
  (h_cos : Real.cos α > 0) : 
  (3 * π / 2 < α ∧ α < 2 * π) := 
sorry

end determine_angle_range_l166_166668


namespace min_fencing_cost_l166_166002

theorem min_fencing_cost {A B C : ℕ} (h1 : A = 25) (h2 : B = 35) (h3 : C = 40)
  (h_ratio : ∃ (x : ℕ), 3 * x * 4 * x = 8748) : 
  ∃ (total_cost : ℝ), total_cost = 87.75 :=
by
  sorry

end min_fencing_cost_l166_166002


namespace bridge_length_l166_166860

-- Defining the problem based on the given conditions and proof goal
theorem bridge_length (L : ℝ) 
  (h1 : L / 4 + L / 3 + 120 = L) :
  L = 288 :=
sorry

end bridge_length_l166_166860


namespace painting_price_difference_l166_166459

theorem painting_price_difference :
  let previous_painting := 9000
  let recent_painting := 44000
  let five_times_more := 5 * previous_painting + previous_painting
  five_times_more - recent_painting = 10000 :=
by
  intros
  sorry

end painting_price_difference_l166_166459


namespace find_certain_number_l166_166805

-- Definitions of conditions from the problem
def greatest_number : ℕ := 10
def divided_1442_by_greatest_number_leaves_remainder := (1442 % greatest_number = 12)
def certain_number_mod_greatest_number (x : ℕ) := (x % greatest_number = 6)

-- Theorem statement
theorem find_certain_number (x : ℕ) (h1 : greatest_number = 10)
  (h2 : 1442 % greatest_number = 12)
  (h3 : certain_number_mod_greatest_number x) : x = 1446 :=
sorry

end find_certain_number_l166_166805


namespace x1_value_l166_166485

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + 2 * (x1 - x2)^2 + 2 * (x2 - x3)^2 + x3^2 = 1 / 2) : 
  x1 = 2 / 3 :=
sorry

end x1_value_l166_166485


namespace straight_flush_probability_l166_166401

open Classical

noncomputable def number_of_possible_hands : ℕ := Nat.choose 52 5

noncomputable def number_of_straight_flushes : ℕ := 40 

noncomputable def probability_of_straight_flush : ℚ := number_of_straight_flushes / number_of_possible_hands

theorem straight_flush_probability :
  probability_of_straight_flush = 1 / 64974 := by
  sorry

end straight_flush_probability_l166_166401


namespace total_pencils_proof_l166_166412

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end total_pencils_proof_l166_166412


namespace solve_equation_l166_166864

theorem solve_equation (x : ℝ) : (2*x - 1)^2 = 81 ↔ (x = 5 ∨ x = -4) :=
by
  sorry

end solve_equation_l166_166864


namespace solution_set_of_inequality_l166_166259

theorem solution_set_of_inequality (x : ℝ) : (1 / 2 < x ∧ x < 1) ↔ (x / (2 * x - 1) > 1) :=
by { sorry }

end solution_set_of_inequality_l166_166259


namespace determine_constants_l166_166618

theorem determine_constants :
  ∃ P Q R : ℚ, (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → (x^2 - 4 * x + 8) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧ 
  P = 1 / 3 ∧ Q = - 4 / 3 ∧ R = 2 :=
by
  -- Proof is left as a placeholder
  sorry

end determine_constants_l166_166618


namespace sum_of_numbers_l166_166920

theorem sum_of_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := 
by
  sorry

end sum_of_numbers_l166_166920


namespace smallest_lcm_of_4_digit_integers_with_gcd_5_l166_166467

-- Definition of the given integers k and l
def positive_4_digit_integers (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- The main theorem we want to prove
theorem smallest_lcm_of_4_digit_integers_with_gcd_5 :
  ∃ (k l : ℕ), positive_4_digit_integers k ∧ positive_4_digit_integers l ∧ gcd k l = 5 ∧ lcm k l = 201000 :=
by {
  sorry
}

end smallest_lcm_of_4_digit_integers_with_gcd_5_l166_166467


namespace find_k_l166_166707

theorem find_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (h : k * x - y = 3) : k = 2 := by
  sorry

end find_k_l166_166707


namespace prob_not_all_same_correct_l166_166184

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l166_166184


namespace cube_volume_l166_166838

theorem cube_volume (h : 12 * l = 72) : l^3 = 216 :=
sorry

end cube_volume_l166_166838


namespace al_original_portion_l166_166863

theorem al_original_portion (a b c : ℝ) (h1 : a + b + c = 1200) (h2 : 0.75 * a + 2 * b + 2 * c = 1800) : a = 480 :=
by
  sorry

end al_original_portion_l166_166863


namespace arcsin_sqrt_three_over_two_l166_166162

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l166_166162


namespace fixed_point_l166_166325

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x - 1)

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : f a (1/2) = 1 :=
by
  sorry

end fixed_point_l166_166325


namespace perimeter_of_square_l166_166927

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end perimeter_of_square_l166_166927


namespace books_arrangement_l166_166491

-- All conditions provided in Lean as necessary definitions
def num_arrangements (math_books english_books science_books : ℕ) : ℕ :=
  if math_books = 4 ∧ english_books = 6 ∧ science_books = 2 then
    let arrangements_groups := 2 * 3  -- Number of valid group placements
    let arrangements_math := Nat.factorial math_books
    let arrangements_english := Nat.factorial english_books
    let arrangements_science := Nat.factorial science_books
    arrangements_groups * arrangements_math * arrangements_english * arrangements_science
  else
    0

theorem books_arrangement : num_arrangements 4 6 2 = 207360 :=
by
  sorry

end books_arrangement_l166_166491


namespace no_grasshopper_at_fourth_vertex_l166_166200

-- Definitions based on given conditions
def is_vertex_of_square (x : ℝ) (y : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1)

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

def leapfrog_jump (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2)

-- Problem statement
theorem no_grasshopper_at_fourth_vertex (a b c : ℝ × ℝ) :
  is_vertex_of_square a.1 a.2 ∧ is_vertex_of_square b.1 b.2 ∧ is_vertex_of_square c.1 c.2 →
  ∃ d : ℝ × ℝ, is_vertex_of_square d.1 d.2 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c →
  ∀ (n : ℕ) (pos : ℕ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ), (pos 0 a b = leapfrog_jump a b) ∧
    (pos n a b = leapfrog_jump (pos (n-1) a b) (pos (n-1) b c)) →
    (pos n a b).1 ≠ (d.1) ∨ (pos n a b).2 ≠ (d.2) :=
sorry

end no_grasshopper_at_fourth_vertex_l166_166200


namespace tan_half_sum_of_angles_l166_166113

theorem tan_half_sum_of_angles (p q : ℝ) 
    (h1 : Real.cos p + Real.cos q = 3 / 5) 
    (h2 : Real.sin p + Real.sin q = 1 / 4) :
    Real.tan ((p + q) / 2) = 5 / 12 := by
  sorry

end tan_half_sum_of_angles_l166_166113


namespace min_value_expression_l166_166624

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((5 / c) - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := 
by
  sorry

end min_value_expression_l166_166624


namespace equidistant_xaxis_point_l166_166218

theorem equidistant_xaxis_point {x : ℝ} :
  (∃ x : ℝ, ∀ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (2, 5) →
    ∀ P : ℝ × ℝ, P = (x, 0) →
      (dist A P = dist B P) → x = 2) := sorry

end equidistant_xaxis_point_l166_166218


namespace coefficient_j_l166_166157

theorem coefficient_j (j k : ℝ) (p : Polynomial ℝ) (h : p = Polynomial.C 400 + Polynomial.X * Polynomial.C k + Polynomial.X^2 * Polynomial.C j + Polynomial.X^4) :
  (∃ a d : ℝ, (d ≠ 0) ∧ (0 > (4*a + 6*d)) ∧ (p.eval a = 0) ∧ (p.eval (a + d) = 0) ∧ (p.eval (a + 2*d) = 0) ∧ (p.eval (a + 3*d) = 0)) → 
  j = -40 :=
by
  sorry

end coefficient_j_l166_166157


namespace lcm_of_three_l166_166071

theorem lcm_of_three (A1 A2 A3 : ℕ) (D : ℕ)
  (hD : D = Nat.gcd (A1 * A2) (Nat.gcd (A2 * A3) (A3 * A1))) :
  Nat.lcm (Nat.lcm A1 A2) A3 = (A1 * A2 * A3) / D :=
sorry

end lcm_of_three_l166_166071


namespace simplify_neg_x_mul_3_minus_x_l166_166353

theorem simplify_neg_x_mul_3_minus_x (x : ℝ) : -x * (3 - x) = -3 * x + x^2 :=
by
  sorry

end simplify_neg_x_mul_3_minus_x_l166_166353


namespace find_other_outlet_rate_l166_166402

open Real

-- Definitions based on conditions
def V : ℝ := 20 * 1728   -- volume of the tank in cubic inches
def r1 : ℝ := 5          -- rate of inlet pipe in cubic inches/min
def r2 : ℝ := 8          -- rate of one outlet pipe in cubic inches/min
def t : ℝ := 2880        -- time in minutes required to empty the tank
 
-- Mathematically equivalent proof statement
theorem find_other_outlet_rate (x : ℝ) : 
  -- Given conditions
  V = 34560 →
  r1 = 5 →
  r2 = 8 →
  t = 2880 →
  -- Statement to prove
  V = (r2 + x - r1) * t → x = 9 :=
by
  intro hV hr1 hr2 ht hEq
  sorry

end find_other_outlet_rate_l166_166402


namespace tangential_circle_radius_l166_166623

theorem tangential_circle_radius (R r x : ℝ) (hR : R > r) (hx : x = 4 * R * r / (R + r)) :
  ∃ x, x = 4 * R * r / (R + r) := by
sorry

end tangential_circle_radius_l166_166623


namespace overall_gain_percent_l166_166545

theorem overall_gain_percent {initial_cost first_repair second_repair third_repair sell_price : ℝ} 
  (h1 : initial_cost = 800) 
  (h2 : first_repair = 150) 
  (h3 : second_repair = 75) 
  (h4 : third_repair = 225) 
  (h5 : sell_price = 1600) :
  (sell_price - (initial_cost + first_repair + second_repair + third_repair)) / 
  (initial_cost + first_repair + second_repair + third_repair) * 100 = 28 := 
by 
  sorry

end overall_gain_percent_l166_166545


namespace dividend_is_2160_l166_166709

theorem dividend_is_2160 (d q r : ℕ) (h₁ : d = 2016 + d) (h₂ : q = 15) (h₃ : r = 0) : d = 2160 :=
by
  sorry

end dividend_is_2160_l166_166709


namespace male_salmon_count_l166_166057

theorem male_salmon_count (total_count : ℕ) (female_count : ℕ) (male_count : ℕ) :
  total_count = 971639 →
  female_count = 259378 →
  male_count = (total_count - female_count) →
  male_count = 712261 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end male_salmon_count_l166_166057


namespace find_f_3_l166_166584

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) : f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_3 : f 3 = 3 / 2 := 
by
  sorry

end find_f_3_l166_166584


namespace ali_peter_fish_ratio_l166_166757

theorem ali_peter_fish_ratio (P J A : ℕ) (h1 : J = P + 1) (h2 : A = 12) (h3 : A + P + J = 25) : A / P = 2 :=
by
  -- Step-by-step simplifications will follow here in the actual proof.
  sorry

end ali_peter_fish_ratio_l166_166757


namespace a_and_b_work_together_l166_166066
noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem a_and_b_work_together (A_days B_days : ℕ) (hA : A_days = 32) (hB : B_days = 32) :
  (1 / work_rate A_days + 1 / work_rate B_days) = 16 := by
  sorry

end a_and_b_work_together_l166_166066


namespace intersection_M_N_l166_166261

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }
def N : Set ℝ := { x : ℝ | 1 < x }

-- State the problem in terms of Lean definitions and theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l166_166261


namespace smallest_t_circle_sin_l166_166327

theorem smallest_t_circle_sin (t : ℝ) (h0 : 0 ≤ t) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ k : ℤ, θ = (π/2 + 2 * π * k) ∨ θ = (3 * π / 2 + 2 * π * k)) : t = π :=
by {
  sorry
}

end smallest_t_circle_sin_l166_166327


namespace oil_to_water_ratio_in_bottle_D_l166_166718

noncomputable def bottle_oil_water_ratio (CA : ℝ) (CB : ℝ) (CC : ℝ) (CD : ℝ) : ℝ :=
  let oil_A := (1 / 2) * CA
  let water_A := (1 / 2) * CA
  let oil_B := (1 / 4) * CB
  let water_B := (1 / 4) * CB
  let total_water_B := CB - oil_B - water_B
  let oil_C := (1 / 3) * CC
  let water_C := 0.4 * CC
  let total_water_C := CC - oil_C - water_C
  let total_capacity_D := CD
  let total_oil_D := oil_A + oil_B + oil_C
  let total_water_D := water_A + total_water_B + water_C + total_water_C
  total_oil_D / total_water_D

theorem oil_to_water_ratio_in_bottle_D (CA : ℝ) :
  let CB := 2 * CA
  let CC := 3 * CA
  let CD := CA + CC
  bottle_oil_water_ratio CA CB CC CD = (2 / 3.7) :=
by 
  sorry

end oil_to_water_ratio_in_bottle_D_l166_166718


namespace lcm_hcf_product_l166_166665

theorem lcm_hcf_product (lcm hcf a b : ℕ) (hlcm : lcm = 2310) (hhcf : hcf = 30) (ha : a = 330) (eq : lcm * hcf = a * b) : b = 210 :=
by {
  sorry
}

end lcm_hcf_product_l166_166665


namespace expansion_correct_l166_166161

noncomputable def P (x y : ℝ) : ℝ := 2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9

noncomputable def M (x : ℝ) : ℝ := 3 * x^7

theorem expansion_correct (x y : ℝ) :
  (P x y) * (M x) = 6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 :=
by
  sorry

end expansion_correct_l166_166161


namespace each_persons_tip_l166_166981

theorem each_persons_tip
  (cost_julie cost_letitia cost_anton : ℕ)
  (H1 : cost_julie = 10)
  (H2 : cost_letitia = 20)
  (H3 : cost_anton = 30)
  (total_people : ℕ)
  (H4 : total_people = 3)
  (tip_percentage : ℝ)
  (H5 : tip_percentage = 0.20) :
  ∃ tip_per_person : ℝ, tip_per_person = 4 := 
by
  sorry

end each_persons_tip_l166_166981


namespace perp_bisector_of_AB_l166_166477

noncomputable def perpendicular_bisector_eq : Prop :=
  ∀ (x y : ℝ), (x - y + 1 = 0) ∧ (x^2 + y^2 = 1) → (x + y = 0)

-- The proof is omitted
theorem perp_bisector_of_AB : perpendicular_bisector_eq :=
sorry

end perp_bisector_of_AB_l166_166477


namespace mod_exp_value_l166_166316

theorem mod_exp_value (m : ℕ) (h1: 0 ≤ m) (h2: m < 9) (h3: 14^4 ≡ m [MOD 9]) : m = 5 :=
by
  sorry

end mod_exp_value_l166_166316


namespace Jasmine_shopping_time_l166_166501

-- Define the variables for the times in minutes
def T_start := 960  -- 4:00 pm in minutes (4*60)
def T_commute := 30
def T_dryClean := 10
def T_dog := 20
def T_cooking := 90
def T_dinner := 1140  -- 7:00 pm in minutes (19*60)

-- The calculated start time for cooking in minutes
def T_startCooking := T_dinner - T_cooking

-- The time Jasmine has between arriving home and starting cooking
def T_groceryShopping := T_startCooking - (T_start + T_commute + T_dryClean + T_dog)

theorem Jasmine_shopping_time :
  T_groceryShopping = 30 := by
  sorry

end Jasmine_shopping_time_l166_166501


namespace circle_intersection_range_l166_166397

noncomputable def circleIntersectionRange (r : ℝ) : Prop :=
  1 < r ∧ r < 11

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) :
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) ↔ circleIntersectionRange r :=
by
  sorry

end circle_intersection_range_l166_166397


namespace total_actions_135_l166_166247

theorem total_actions_135
  (y : ℕ) -- represents the total number of actions
  (h1 : y ≥ 10) -- since there are at least 10 initial comments
  (h2 : ∀ (likes dislikes : ℕ), likes + dislikes = y - 10) -- total votes exclude neutral comments
  (score_eq : ∀ (likes dislikes : ℕ), 70 * dislikes = 30 * likes)
  (score_50 : ∀ (likes dislikes : ℕ), 50 = likes - dislikes) :
  y = 135 :=
by {
  sorry
}

end total_actions_135_l166_166247


namespace power_multiplication_l166_166970

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end power_multiplication_l166_166970


namespace expand_expression_l166_166759

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 :=
  sorry

end expand_expression_l166_166759


namespace ellipse_problem_l166_166064

theorem ellipse_problem
  (F2 : ℝ) (a : ℝ) (A B : ℝ × ℝ)
  (on_ellipse_A : (A.1 ^ 2) / (a ^ 2) + (25 * (A.2 ^ 2)) / (9 * a ^ 2) = 1)
  (on_ellipse_B : (B.1 ^ 2) / (a ^ 2) + (25 * (B.2 ^ 2)) / (9 * a ^ 2) = 1)
  (focal_distance : |A.1 + F2| + |B.1 + F2| = 8 / 5 * a)
  (midpoint_to_directrix : |(A.1 + B.1) / 2 + 5 / 4 * a| = 3 / 2) :
  a = 1 → (∀ x y, (x^2 + (25 / 9) * y^2 = 1) ↔ ((x^2) / (a^2) + (25 * y^2) / (9 * a^2) = 1)) :=
by
  sorry

end ellipse_problem_l166_166064


namespace least_multiple_of_7_not_lucky_l166_166068

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem least_multiple_of_7_not_lucky : ∃ n, is_multiple_of_7 n ∧ ¬ is_lucky n ∧ n = 14 :=
by
  sorry

end least_multiple_of_7_not_lucky_l166_166068


namespace parabola_c_value_l166_166425

theorem parabola_c_value (b c : ℝ) 
  (h1 : 20 = 2*(-2)^2 + b*(-2) + c) 
  (h2 : 28 = 2*2^2 + b*2 + c) : 
  c = 16 :=
by
  sorry

end parabola_c_value_l166_166425


namespace find_A_l166_166555

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 10 * A + 3 + 610 + B = 695) : A = 8 :=
by {
  sorry
}

end find_A_l166_166555


namespace find_range_of_a_l166_166301

noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp x) - (Real.exp x) + 2 * x - (1 / 3) * x ^ 3

theorem find_range_of_a (a : ℝ) (h : f (3 * a ^ 2) + f (2 * a - 1) ≥ 0) : a ∈ Set.Icc (-1 : ℝ) (1 / 3) :=
sorry

end find_range_of_a_l166_166301


namespace perfect_squares_l166_166774

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l166_166774


namespace correct_response_percentage_l166_166105

def number_of_students : List ℕ := [300, 1100, 100, 600, 400]
def total_students : ℕ := number_of_students.sum
def correct_response_students : ℕ := number_of_students.maximum.getD 0

theorem correct_response_percentage :
  (correct_response_students * 100 / total_students) = 44 := by
  sorry

end correct_response_percentage_l166_166105


namespace three_digit_odds_factors_count_l166_166468

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l166_166468


namespace vector_addition_example_l166_166106

noncomputable def OA : ℝ × ℝ := (-2, 3)
noncomputable def AB : ℝ × ℝ := (-1, -4)
noncomputable def OB : ℝ × ℝ := (OA.1 + AB.1, OA.2 + AB.2)

theorem vector_addition_example :
  OB = (-3, -1) :=
by
  sorry

end vector_addition_example_l166_166106


namespace sinA_value_find_b_c_l166_166302

-- Define the conditions
def triangle (A B C : Type) (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

variable {A B C : Type} (a b c : ℝ)
variable {S_triangle_ABC : ℝ}
variable {cosB : ℝ}

-- Given conditions
axiom cosB_val : cosB = 3 / 5
axiom a_val : a = 2

-- Problem 1: Prove sinA = 2/5 given additional condition b = 4
axiom b_val : b = 4

theorem sinA_value (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_b : b = 4) : 
  ∃ sinA : ℝ, sinA = 2 / 5 :=
sorry

-- Problem 2: Prove b = sqrt(17) and c = 5 given the area
axiom area_val : S_triangle_ABC = 4

theorem find_b_c (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_area : S_triangle_ABC = 4) : 
  ∃ b c : ℝ, b = Real.sqrt 17 ∧ c = 5 :=
sorry

end sinA_value_find_b_c_l166_166302


namespace num_senior_in_sample_l166_166458

-- Definitions based on conditions
def total_students : ℕ := 2000
def senior_students : ℕ := 700
def sample_size : ℕ := 400

-- Theorem statement for the number of senior students in the sample
theorem num_senior_in_sample : 
  (senior_students * sample_size) / total_students = 140 :=
by 
  sorry

end num_senior_in_sample_l166_166458


namespace power_identity_l166_166138

-- Define the given definitions
def P (m : ℕ) : ℕ := 5 ^ m
def R (n : ℕ) : ℕ := 7 ^ n

-- The theorem to be proved
theorem power_identity (m n : ℕ) : 35 ^ (m + n) = (P m ^ n * R n ^ m) := 
by sorry

end power_identity_l166_166138


namespace recurring_decimals_sum_correct_l166_166802

noncomputable def recurring_decimals_sum : ℚ :=
  let x := (2:ℚ) / 3
  let y := (4:ℚ) / 9
  x + y

theorem recurring_decimals_sum_correct :
  recurring_decimals_sum = 10 / 9 := 
  sorry

end recurring_decimals_sum_correct_l166_166802


namespace smallest_d_for_inverse_g_l166_166172

def g (x : ℝ) := (x - 3)^2 - 8

theorem smallest_d_for_inverse_g : ∃ d : ℝ, (∀ x y : ℝ, x ≠ y → x ≥ d → y ≥ d → g x ≠ g y) ∧ ∀ d' : ℝ, d' < 3 → ∃ x y : ℝ, x ≠ y ∧ x ≥ d' ∧ y ≥ d' ∧ g x = g y :=
by
  sorry

end smallest_d_for_inverse_g_l166_166172


namespace machine_p_vs_machine_q_l166_166843

variable (MachineA_rate MachineQ_rate MachineP_rate : ℝ)
variable (Total_sprockets : ℝ := 550)
variable (Production_rate_A : ℝ := 5)
variable (Production_rate_Q : ℝ := MachineA_rate + 0.1 * MachineA_rate)
variable (Time_Q : ℝ := Total_sprockets / Production_rate_Q)
variable (Time_P : ℝ)
variable (Difference : ℝ)

noncomputable def production_times_difference (MachineA_rate MachineQ_rate MachineP_rate : ℝ) : ℝ :=
  let Production_rate_Q := MachineA_rate + 0.1 * MachineA_rate
  let Time_Q := Total_sprockets / Production_rate_Q
  let Difference := Time_P - Time_Q
  Difference

theorem machine_p_vs_machine_q : 
  Production_rate_A = 5 → 
  Total_sprockets = 550 →
  Production_rate_Q = 5.5 →
  Time_Q = 100 →
  MachineP_rate = MachineP_rate →
  Time_P = Time_P →
  Difference = (Time_P - Time_Q) :=
by
  intros
  sorry

end machine_p_vs_machine_q_l166_166843


namespace sufficient_not_necessary_condition_l166_166202

theorem sufficient_not_necessary_condition (x : ℝ) : x - 1 > 0 → (x > 2) ∧ (¬ (x - 1 > 0 → x > 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l166_166202


namespace power_division_correct_l166_166727

theorem power_division_correct :
  (∀ x : ℝ, x^4 / x = x^3) ∧ 
  ¬(∀ x : ℝ, 3 * x^2 * 4 * x^2 = 12 * x^2) ∧
  ¬(∀ x : ℝ, (x - 1) * (x - 1) = x^2 - 1) ∧
  ¬(∀ x : ℝ, (x^5)^2 = x^7) := 
by {
  -- Proof would go here
  sorry
}

end power_division_correct_l166_166727


namespace intersection_of_S_and_T_l166_166044

def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | 0 < x}

theorem intersection_of_S_and_T : S ∩ T = {x | 1 ≤ x} := by
  sorry

end intersection_of_S_and_T_l166_166044


namespace dice_sum_four_l166_166275

def possible_outcomes (x : Nat) : Set (Nat × Nat) :=
  { (d1, d2) | d1 + d2 = x ∧ 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 }

theorem dice_sum_four :
  possible_outcomes 4 = {(3, 1), (1, 3), (2, 2)} :=
by
  sorry -- We acknowledge that this outline is equivalent to the provided math problem.

end dice_sum_four_l166_166275


namespace largest_common_value_less_than_1000_l166_166443

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, 
    (∃ n : ℕ, a = 4 + 5 * n) ∧
    (∃ m : ℕ, a = 5 + 10 * m) ∧
    a % 4 = 1 ∧
    a < 1000 ∧
    (∀ b : ℕ, 
      (∃ n : ℕ, b = 4 + 5 * n) ∧
      (∃ m : ℕ, b = 5 + 10 * m) ∧
      b % 4 = 1 ∧
      b < 1000 → 
      b ≤ a) ∧ 
    a = 989 :=
by
  sorry

end largest_common_value_less_than_1000_l166_166443


namespace simplify_and_evaluate_l166_166220

theorem simplify_and_evaluate :
  let a := 1
  let b := 2
  (a - b) ^ 2 - a * (a - b) + (a + b) * (a - b) = -1 := by
  sorry

end simplify_and_evaluate_l166_166220


namespace president_savings_l166_166569

theorem president_savings (total_funds : ℕ) (friends_percentage : ℕ) (family_percentage : ℕ) 
  (friends_contradiction funds_left family_contribution fundraising_amount : ℕ) :
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  friends_contradiction = (total_funds * friends_percentage) / 100 →
  funds_left = total_funds - friends_contradiction →
  family_contribution = (funds_left * family_percentage) / 100 →
  fundraising_amount = funds_left - family_contribution →
  fundraising_amount = 4200 :=
by
  intros
  sorry

end president_savings_l166_166569


namespace find_a_l166_166129

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end find_a_l166_166129


namespace question1_question2_l166_166389

-- Define the conditions
def numTraditionalChinesePaintings : Nat := 5
def numOilPaintings : Nat := 2
def numWatercolorPaintings : Nat := 7

-- Define the number of ways to choose one painting from each category
def numWaysToChooseOnePaintingFromEachCategory : Nat :=
  numTraditionalChinesePaintings * numOilPaintings * numWatercolorPaintings

-- Define the number of ways to choose two paintings of different types
def numWaysToChooseTwoPaintingsOfDifferentTypes : Nat :=
  (numTraditionalChinesePaintings * numOilPaintings) +
  (numTraditionalChinesePaintings * numWatercolorPaintings) +
  (numOilPaintings * numWatercolorPaintings)

-- Theorems to prove the required results
theorem question1 : numWaysToChooseOnePaintingFromEachCategory = 70 := by
  sorry

theorem question2 : numWaysToChooseTwoPaintingsOfDifferentTypes = 59 := by
  sorry

end question1_question2_l166_166389


namespace find_b_l166_166484

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (317212435 * 101 - b) % 25 = 0 ∧ b = 13 := by
  sorry

end find_b_l166_166484


namespace chord_length_of_tangent_circle_l166_166352

theorem chord_length_of_tangent_circle
  (area_of_ring : ℝ)
  (diameter_large_circle : ℝ)
  (h1 : area_of_ring = (50 / 3) * Real.pi)
  (h2 : diameter_large_circle = 10) :
  ∃ (length_of_chord : ℝ), length_of_chord = (10 * Real.sqrt 6) / 3 := by
  sorry

end chord_length_of_tangent_circle_l166_166352


namespace dylan_speed_constant_l166_166385

theorem dylan_speed_constant (d t s : ℝ) (h1 : d = 1250) (h2 : t = 25) (h3 : s = d / t) : s = 50 := 
by 
  -- Proof steps will go here
  sorry

end dylan_speed_constant_l166_166385


namespace train_speed_in_kmh_l166_166804

def length_of_train : ℝ := 600
def length_of_overbridge : ℝ := 100
def time_to_cross_overbridge : ℝ := 70

theorem train_speed_in_kmh :
  (length_of_train + length_of_overbridge) / time_to_cross_overbridge * 3.6 = 36 := 
by 
  sorry

end train_speed_in_kmh_l166_166804


namespace sum_of_decimals_l166_166997

theorem sum_of_decimals : (0.305 : ℝ) + (0.089 : ℝ) + (0.007 : ℝ) = 0.401 := by
  sorry

end sum_of_decimals_l166_166997


namespace nancy_packs_of_crayons_l166_166682

theorem nancy_packs_of_crayons (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l166_166682


namespace negation_equivalence_l166_166432

theorem negation_equivalence (x : ℝ) : ¬(∀ x, x^2 - x + 2 ≥ 0) ↔ ∃ x, x^2 - x + 2 < 0 :=
sorry

end negation_equivalence_l166_166432


namespace horner_v2_value_l166_166963

def polynomial : ℤ → ℤ := fun x => 208 + 9 * x^2 + 6 * x^4 + x^6

def horner (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x
  let v2 := v1 * x + 6
  v2

theorem horner_v2_value (x : ℤ) : x = -4 → horner x = 22 :=
by
  intro h
  rw [h]
  rfl

end horner_v2_value_l166_166963


namespace true_prices_for_pie_and_mead_l166_166224

-- Definitions for true prices
variable (k m : ℕ)

-- Definitions for conditions
def honest_pravdoslav (k m : ℕ) : Prop :=
  4*k = 3*(m + 2) ∧ 4*(m+2) = 3*k + 14

theorem true_prices_for_pie_and_mead (k m : ℕ) (h : honest_pravdoslav k m) : k = 6 ∧ m = 6 := sorry

end true_prices_for_pie_and_mead_l166_166224


namespace rowing_speed_upstream_l166_166137

theorem rowing_speed_upstream (V_m V_down : ℝ) (h_Vm : V_m = 35) (h_Vdown : V_down = 40) : V_m - (V_down - V_m) = 30 :=
by
  sorry

end rowing_speed_upstream_l166_166137


namespace toothpicks_in_stage_200_l166_166051

def initial_toothpicks : ℕ := 6
def toothpicks_per_stage : ℕ := 5
def stage_number : ℕ := 200

theorem toothpicks_in_stage_200 :
  initial_toothpicks + (stage_number - 1) * toothpicks_per_stage = 1001 := by
  sorry

end toothpicks_in_stage_200_l166_166051


namespace sum_of_final_two_numbers_l166_166829

noncomputable def final_sum (X m n : ℚ) : ℚ :=
  3 * m + 3 * n - 14

theorem sum_of_final_two_numbers (X m n : ℚ) 
  (h1 : m + n = X) :
  final_sum X m n = 3 * X - 14 :=
  sorry

end sum_of_final_two_numbers_l166_166829


namespace max_value_sequence_l166_166256

theorem max_value_sequence (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a (n + 1) = (-1 : ℝ)^n * n - a n)
  (h2 : a 10 = a 1) :
  ∃ n, a n * a (n + 1) = 33 / 4 :=
sorry

end max_value_sequence_l166_166256


namespace f_2006_eq_1_l166_166730

noncomputable def f : ℤ → ℤ := sorry
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3 : ∀ x : ℤ, f (3 * (x + 1)) = f (3 * x + 1)
axiom f_at_1 : f 1 = -1

theorem f_2006_eq_1 : f 2006 = 1 := by
  sorry

end f_2006_eq_1_l166_166730


namespace distance_between_vertices_hyperbola_l166_166910

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l166_166910


namespace minimum_value_of_expression_l166_166600

theorem minimum_value_of_expression {x : ℝ} (hx : x > 0) : (2 / x + x / 2) ≥ 2 :=
by sorry

end minimum_value_of_expression_l166_166600


namespace tangent_circles_locus_l166_166512

theorem tangent_circles_locus :
  ∃ (a b : ℝ), ∀ (C1_center : ℝ × ℝ) (C2_center : ℝ × ℝ) (C1_radius : ℝ) (C2_radius : ℝ),
    C1_center = (0, 0) ∧ C2_center = (2, 0) ∧ C1_radius = 1 ∧ C2_radius = 3 ∧
    (∀ (r : ℝ), (a - 0)^2 + (b - 0)^2 = (r + C1_radius)^2 ∧ (a - 2)^2 + (b - 0)^2 = (C2_radius - r)^2) →
    84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 := sorry

end tangent_circles_locus_l166_166512


namespace find_y_l166_166015

theorem find_y (y: ℕ) (h1: y > 0) (h2: y ≤ 100)
  (h3: (43 + 69 + 87 + y + y) / 5 = 2 * y): 
  y = 25 :=
sorry

end find_y_l166_166015


namespace aku_invited_friends_l166_166367

def total_cookies (packages : ℕ) (cookies_per_package : ℕ) := packages * cookies_per_package

def total_children (total_cookies : ℕ) (cookies_per_child : ℕ) := total_cookies / cookies_per_child

def invited_friends (total_children : ℕ) := total_children - 1

theorem aku_invited_friends (packages cookies_per_package cookies_per_child : ℕ) (h1 : packages = 3) (h2 : cookies_per_package = 25) (h3 : cookies_per_child = 15) :
  invited_friends (total_children (total_cookies packages cookies_per_package) cookies_per_child) = 4 :=
by
  sorry

end aku_invited_friends_l166_166367


namespace trader_sells_cloth_l166_166339

variable (x : ℝ) (SP_total : ℝ := 6900) (profit_per_meter : ℝ := 20) (CP_per_meter : ℝ := 66.25)

theorem trader_sells_cloth : SP_total = x * (CP_per_meter + profit_per_meter) → x = 80 :=
by
  intro h
  -- Placeholder for actual proof
  sorry

end trader_sells_cloth_l166_166339


namespace num_students_taking_music_l166_166120

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_taking_art : ℕ := 20
def students_taking_both_music_and_art : ℕ := 10
def students_taking_neither_music_nor_art : ℕ := 450

-- Theorem statement to prove the number of students taking music
theorem num_students_taking_music :
  ∃ (M : ℕ), M = 40 ∧ 
  (total_students - students_taking_neither_music_nor_art = M + students_taking_art - students_taking_both_music_and_art) := 
by
  sorry

end num_students_taking_music_l166_166120


namespace largest_value_x_l166_166919

theorem largest_value_x (x a b c d : ℝ) (h_eq : 7 * x ^ 2 + 15 * x - 20 = 0) (h_form : x = (a + b * Real.sqrt c) / d) (ha : a = -15) (hb : b = 1) (hc : c = 785) (hd : d = 14) : (a * c * d) / b = -164850 := 
sorry

end largest_value_x_l166_166919


namespace dog_bones_l166_166666

theorem dog_bones (initial_bones found_bones : ℕ) (h₁ : initial_bones = 15) (h₂ : found_bones = 8) : initial_bones + found_bones = 23 := by
  sorry

end dog_bones_l166_166666


namespace blue_paint_cans_l166_166026

noncomputable def ratio_of_blue_to_green := 4 / 1
def total_cans := 50
def fraction_of_blue := 4 / (4 + 1)
def number_of_blue_cans := fraction_of_blue * total_cans

theorem blue_paint_cans : number_of_blue_cans = 40 := by
  sorry

end blue_paint_cans_l166_166026


namespace proof_problem_l166_166849

def sum_even_ints (n : ℕ) : ℕ := n * (n + 1)
def sum_odd_ints (n : ℕ) : ℕ := n^2
def sum_specific_primes : ℕ := [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].sum

theorem proof_problem : (sum_even_ints 100 - sum_odd_ints 100) + sum_specific_primes = 1063 :=
by
  sorry

end proof_problem_l166_166849


namespace gauss_algorithm_sum_l166_166505

def f (x : Nat) (m : Nat) : Rat := x / (3 * m + 6054)

theorem gauss_algorithm_sum (m : Nat) :
  (Finset.sum (Finset.range (m + 2017 + 1)) (λ x => f x m)) = (m + 2017) / 6 := by
sorry

end gauss_algorithm_sum_l166_166505


namespace find_number_of_girls_l166_166891

noncomputable def B (G : ℕ) : ℕ := (8 * G) / 5

theorem find_number_of_girls (B G : ℕ) (h_ratio : B = (8 * G) / 5) (h_total : B + G = 312) : G = 120 :=
by
  -- the proof would be done here
  sorry

end find_number_of_girls_l166_166891


namespace volume_displacement_square_l166_166289

-- Define the given conditions
def radius_cylinder := 5
def height_cylinder := 12
def side_length_cube := 10

theorem volume_displacement_square :
  let r := radius_cylinder
  let h := height_cylinder
  let s := side_length_cube
  let cube_diagonal := s * Real.sqrt 3
  let w := (125 * Real.sqrt 6) / 8
  w^2 = 1464.0625 :=
by
  sorry

end volume_displacement_square_l166_166289


namespace f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l166_166933

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 / x - 1 else 2 / (-x) - 1

-- Assertion 1: Value of f(-1)
theorem f_neg_one : f (-1) = 1 := 
sorry

-- Assertion 2: f(x) is a decreasing function on (0, +∞)
theorem f_decreasing_on_positive : ∀ a b : ℝ, 0 < b → b < a → f (a) < f (b) := 
sorry

-- Assertion 3: Expression of the function when x < 0
theorem f_expression_on_negative (x : ℝ) (hx : x < 0) : f x = 2 / (-x) - 1 := 
sorry

end f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l166_166933


namespace solve_exponent_equation_l166_166679

theorem solve_exponent_equation (x y z : ℕ) :
  7^x + 1 = 3^y + 5^z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end solve_exponent_equation_l166_166679


namespace how_many_times_l166_166980

theorem how_many_times (a b : ℝ) (h1 : a = 0.5) (h2 : b = 0.01) : a / b = 50 := 
by 
  sorry

end how_many_times_l166_166980


namespace solve_ineq_system_l166_166007

theorem solve_ineq_system (x : ℝ) :
  (x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x ≤ 1 :=
by sorry

end solve_ineq_system_l166_166007


namespace sum_of_x_and_y_l166_166276

theorem sum_of_x_and_y 
  (x y : ℝ)
  (h : ((x + 1) + (y-1)) / 2 = 10) : x + y = 20 :=
sorry

end sum_of_x_and_y_l166_166276


namespace cylinder_radius_l166_166713

theorem cylinder_radius
  (r h : ℝ) (S : ℝ) (h_cylinder : h = 8) (S_surface : S = 130 * Real.pi)
  (surface_area_eq : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 :=
by
  sorry

end cylinder_radius_l166_166713


namespace circumcenter_rational_l166_166626

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l166_166626


namespace two_x_plus_y_equals_7_l166_166806

noncomputable def proof_problem (x y A : ℝ) : ℝ :=
  if (2 * x + y = A ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) then A else 0

theorem two_x_plus_y_equals_7 (x y : ℝ) : 
  (2 * x + y = proof_problem x y 7) ↔
  (2 * x + y = 7 ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) :=
by sorry

end two_x_plus_y_equals_7_l166_166806


namespace arctan_sum_l166_166890

theorem arctan_sum (a b : ℝ) (h1 : a = 2 / 3) (h2 : (a + 1) * (b + 1) = 8 / 3) :
  Real.arctan a + Real.arctan b = Real.arctan (19 / 9) := by
  sorry

end arctan_sum_l166_166890


namespace path_length_of_dot_l166_166708

-- Define the edge length of the cube
def edge_length : ℝ := 3

-- Define the conditions of the problem
def cube_condition (l : ℝ) (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) : Prop :=
  l = edge_length ∧ rolling_without_slipping ∧ at_least_two_vertices_touching ∧ dot_at_one_corner ∧ returns_to_original_position

-- Define the theorem to be proven
theorem path_length_of_dot (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) :
  cube_condition edge_length rolling_without_slipping at_least_two_vertices_touching dot_at_one_corner returns_to_original_position →
  ∃ c : ℝ, c = 6 ∧ (c * Real.pi) = 6 * Real.pi :=
by
  intro h
  sorry

end path_length_of_dot_l166_166708


namespace relation_of_exponents_l166_166729

theorem relation_of_exponents
  (a b c d : ℝ)
  (x y p z : ℝ)
  (h1 : a^x = c)
  (h2 : b^p = c)
  (h3 : b^y = d)
  (h4 : a^z = d) :
  py = xz :=
sorry

end relation_of_exponents_l166_166729


namespace inequality_proof_l166_166749

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) +
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) +
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 := 
by
  sorry

end inequality_proof_l166_166749


namespace value_of_k_l166_166380

theorem value_of_k :
  (∀ x : ℝ, x ^ 2 - x - 2 > 0 → 2 * x ^ 2 + (5 + 2 * k) * x + 5 * k < 0 → x = -2) ↔ -3 ≤ k ∧ k < 2 :=
sorry

end value_of_k_l166_166380


namespace range_of_a_l166_166519

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by
  sorry

end range_of_a_l166_166519


namespace digit_B_divisible_by_9_l166_166480

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l166_166480


namespace uncovered_area_l166_166646

def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side : ℕ := 4

theorem uncovered_area (height width side : ℕ) (h : height = shoebox_height) (w : width = shoebox_width) (s : side = block_side) :
  (width * height) - (side * side) = 8 :=
by
  rw [h, w, s]
  -- Area of shoebox bottom = width * height
  -- Area of square block = side * side
  -- Uncovered area = (width * height) - (side * side)
  -- Therefore, (6 * 4) - (4 * 4) = 24 - 16 = 8
  sorry

end uncovered_area_l166_166646


namespace max_min_f_values_l166_166072

noncomputable def f (a b c d : ℝ) : ℝ := (Real.sqrt (5 * a + 9) + Real.sqrt (5 * b + 9) + Real.sqrt (5 * c + 9) + Real.sqrt (5 * d + 9))

theorem max_min_f_values (a b c d : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h₁ : a + b + c + d = 32) :
  (f a b c d ≤ 28) ∧ (f a b c d ≥ 22) := by
  sorry

end max_min_f_values_l166_166072


namespace find_y_l166_166168

theorem find_y (y: ℕ)
  (h1: ∃ (k : ℕ), y = 9 * k)
  (h2: y^2 > 225)
  (h3: y < 30)
: y = 18 ∨ y = 27 := 
sorry

end find_y_l166_166168


namespace solution_system_of_equations_l166_166579

theorem solution_system_of_equations : 
  ∃ (x y : ℝ), (2 * x - y = 3 ∧ x + y = 3) ∧ (x = 2 ∧ y = 1) := 
by
  sorry

end solution_system_of_equations_l166_166579


namespace computer_hardware_contract_prob_l166_166570

theorem computer_hardware_contract_prob :
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  ∃ P_H : ℝ, P_at_least_one = P_H + P_S - P_H_and_S ∧ P_H = 0.8 :=
by
  -- Let definitions and initial conditions
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  -- Solve for P(H)
  let P_H := 0.8
  -- Show the proof of the calculation
  sorry

end computer_hardware_contract_prob_l166_166570


namespace coefficient_sum_of_squares_is_23456_l166_166028

theorem coefficient_sum_of_squares_is_23456 
  (p q r s t u : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := 
by
  sorry

end coefficient_sum_of_squares_is_23456_l166_166028


namespace box_height_l166_166435

theorem box_height (volume length width : ℝ) (h : ℝ) (h_volume : volume = 315) (h_length : length = 7) (h_width : width = 9) :
  h = 5 :=
by
  -- Proof would go here
  sorry

end box_height_l166_166435


namespace find_k_eq_3_l166_166755

theorem find_k_eq_3 (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3) → k = 3 :=
by sorry

end find_k_eq_3_l166_166755


namespace first_discount_percentage_l166_166403

theorem first_discount_percentage
  (P : ℝ)
  (initial_price final_price : ℝ)
  (second_discount : ℕ)
  (h1 : initial_price = 200)
  (h2 : final_price = 144)
  (h3 : second_discount = 10)
  (h4 : final_price = (P - (second_discount / 100) * P)) :
  (∃ x : ℝ, P = initial_price - (x / 100) * initial_price ∧ x = 20) :=
sorry

end first_discount_percentage_l166_166403


namespace f_36_l166_166558

variable {R : Type*} [CommRing R]
variable (f : R → R) (p q : R)

-- Conditions
axiom f_mult_add : ∀ x y, f (x * y) = f x + f y
axiom f_2 : f 2 = p
axiom f_3 : f 3 = q

-- Statement to prove
theorem f_36 : f 36 = 2 * (p + q) :=
by
  sorry

end f_36_l166_166558


namespace total_concrete_weight_l166_166508

theorem total_concrete_weight (w1 w2 : ℝ) (c1 c2 : ℝ) (total_weight : ℝ)
  (h1 : w1 = 1125)
  (h2 : w2 = 1125)
  (h3 : c1 = 0.093)
  (h4 : c2 = 0.113)
  (h5 : (w1 * c1 + w2 * c2) / (w1 + w2) = 0.108) :
  total_weight = w1 + w2 :=
by
  sorry

end total_concrete_weight_l166_166508


namespace time_for_A_to_complete_race_l166_166701

open Real

theorem time_for_A_to_complete_race (V_A V_B : ℝ) (T_A : ℝ) :
  (V_B = 4) →
  (V_B = 960 / T_A) →
  T_A = 1000 / V_A →
  T_A = 240 := by
  sorry

end time_for_A_to_complete_race_l166_166701


namespace seating_arrangements_correct_l166_166102

-- Conditions
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def driver_choices : ℕ := 2

-- Function to calculate the number of arrangements
noncomputable def seating_arrangements (children : ℕ) (front_seats : ℕ) (back_seats : ℕ) (driver_choices : ℕ) : ℕ :=
  driver_choices * (children + 1) * (back_seats.factorial)

-- Problem Statement
theorem seating_arrangements_correct : 
  seating_arrangements num_children num_front_seats num_back_seats driver_choices = 48 :=
by
  -- Translate conditions to computation
  have h1: num_children = 3 := rfl
  have h2: num_front_seats = 2 := rfl
  have h3: num_back_seats = 3 := rfl
  have h4: driver_choices = 2 := rfl
  sorry

end seating_arrangements_correct_l166_166102


namespace magdalena_fraction_picked_l166_166381

noncomputable def fraction_picked_first_day
  (produced_apples: ℕ)
  (remaining_apples: ℕ)
  (fraction_picked: ℚ) : Prop :=
  ∃ (f : ℚ),
  produced_apples = 200 ∧
  remaining_apples = 20 ∧
  (f = fraction_picked) ∧
  (200 * f + 2 * 200 * f + (200 * f + 20)) = 200 - remaining_apples ∧
  fraction_picked = 1 / 5

theorem magdalena_fraction_picked :
  fraction_picked_first_day 200 20 (1 / 5) :=
sorry

end magdalena_fraction_picked_l166_166381


namespace arithmetic_sequence_ratio_q_l166_166400

theorem arithmetic_sequence_ratio_q :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ), 
    (0 < q) →
    (S 2 = 3 * a 2 + 2) →
    (S 4 = 3 * a 4 + 2) →
    (q = 3 / 2) :=
by
  sorry

end arithmetic_sequence_ratio_q_l166_166400


namespace alice_needs_to_add_stamps_l166_166655

variable (A B E P D : ℕ)
variable (h₁ : B = 4 * E)
variable (h₂ : E = 3 * P)
variable (h₃ : P = 2 * D)
variable (h₄ : D = A + 5)
variable (h₅ : A = 65)

theorem alice_needs_to_add_stamps : (1680 - A = 1615) :=
by
  sorry

end alice_needs_to_add_stamps_l166_166655


namespace sandy_age_l166_166541

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 :=
sorry

end sandy_age_l166_166541


namespace yeast_counting_procedure_l166_166346

def yeast_counting_conditions (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool) : Prop :=
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true

theorem yeast_counting_procedure :
  ∀ (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool),
  yeast_counting_conditions counting_method shake_test_tube_needed dilution_needed →
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true :=
by
  intros counting_method shake_test_tube_needed dilution_needed h_condition
  exact h_condition

end yeast_counting_procedure_l166_166346


namespace complement_union_l166_166131

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

theorem complement_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (hU : U = univ) 
(hA : A = {x : ℝ | 0 < x}) 
(hB : B = {x : ℝ | -3 < x ∧ x < 1}) : 
compl (A ∪ B) = {x : ℝ | x ≤ -3} :=
by
  sorry

end complement_union_l166_166131


namespace partnership_profit_l166_166725

noncomputable def total_profit
  (P : ℝ)
  (mary_investment : ℝ := 700)
  (harry_investment : ℝ := 300)
  (effort_share := P / 3 / 2)
  (remaining_share := 2 / 3 * P)
  (total_investment := mary_investment + harry_investment)
  (mary_share_remaining := (mary_investment / total_investment) * remaining_share)
  (harry_share_remaining := (harry_investment / total_investment) * remaining_share) : Prop :=
  (effort_share + mary_share_remaining) - (effort_share + harry_share_remaining) = 800

theorem partnership_profit : ∃ P : ℝ, total_profit P ∧ P = 3000 :=
  sorry

end partnership_profit_l166_166725


namespace f_0_eq_0_l166_166371

-- Define a function f with the given condition
def f (x : ℤ) : ℤ := if x = 0 then 0
                     else (x-1)^2 + 2*(x-1) + 1

-- State the theorem
theorem f_0_eq_0 : f 0 = 0 :=
by sorry

end f_0_eq_0_l166_166371


namespace diamond_3_7_l166_166605

def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2
def diamond (a b : ℕ) : ℕ := star a b - a * b

theorem diamond_3_7 : diamond 3 7 = 79 :=
by 
  sorry

end diamond_3_7_l166_166605


namespace balance_scale_cereal_l166_166336

def scales_are_balanced (left_pan : ℕ) (right_pan : ℕ) : Prop :=
  left_pan = right_pan

theorem balance_scale_cereal (inaccurate_scales : ℕ → ℕ → Prop)
  (cereal : ℕ)
  (correct_weight : ℕ) :
  (∀ left_pan right_pan, inaccurate_scales left_pan right_pan → left_pan = right_pan) →
  (cereal / 2 = 1) →
  true :=
  sorry

end balance_scale_cereal_l166_166336


namespace expected_number_of_digits_is_1_55_l166_166487

def probability_one_digit : ℚ := 9 / 20
def probability_two_digits : ℚ := 1 / 2
def probability_twenty : ℚ := 1 / 20
def expected_digits : ℚ := (1 * probability_one_digit) + (2 * probability_two_digits) + (2 * probability_twenty)

theorem expected_number_of_digits_is_1_55 :
  expected_digits = 1.55 :=
sorry

end expected_number_of_digits_is_1_55_l166_166487


namespace total_legs_correct_l166_166219

variable (a b : ℕ)

def total_legs (a b : ℕ) : ℕ := 2 * a + 4 * b

theorem total_legs_correct (a b : ℕ) : total_legs a b = 2 * a + 4 * b :=
by sorry

end total_legs_correct_l166_166219


namespace inequality_solution_set_min_value_of_x_plus_y_l166_166503

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem inequality_solution_set (a : ℝ) :
  (if a < 0 then (∀ x : ℝ, f a x > 0 ↔ (1/a < x ∧ x < 2))
   else if a = 0 then (∀ x : ℝ, f a x > 0 ↔ x < 2)
   else if 0 < a ∧ a < 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 2 ∨ 1/a < x))
   else if a = 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x ≠ 2))
   else if a > 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 1/a ∨ x > 2))
   else false) := 
sorry

theorem min_value_of_x_plus_y (a : ℝ) (h : 0 < a) (x y : ℝ) (hx : y ≥ f a (|x|)) :
  x + y ≥ -a - (1/a) := 
sorry

end inequality_solution_set_min_value_of_x_plus_y_l166_166503


namespace parametric_circle_section_l166_166481

theorem parametric_circle_section (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x = 4 - Real.cos θ ∧ y = 1 - Real.sin θ) ∧ (4 - x)^2 + (1 - y)^2 = 1 :=
sorry

end parametric_circle_section_l166_166481


namespace slower_plane_speed_l166_166559

-- Let's define the initial conditions and state the theorem in Lean 4
theorem slower_plane_speed 
    (x : ℕ) -- speed of the slower plane
    (h1 : x + 2*x = 900) : -- based on the total distance after 3 hours
    x = 300 :=
by
    -- Proof goes here
    sorry

end slower_plane_speed_l166_166559


namespace mod_remainder_7_10_20_3_20_l166_166657

theorem mod_remainder_7_10_20_3_20 : (7 * 10^20 + 3^20) % 9 = 7 := sorry

end mod_remainder_7_10_20_3_20_l166_166657


namespace sin_alpha_value_l166_166411

theorem sin_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < Real.pi)
  (h₂ : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

end sin_alpha_value_l166_166411


namespace perpendicular_lines_m_value_l166_166866

-- Define the first line
def line1 (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the second line
def line2 (x y : ℝ) (m : ℝ) : Prop := 6 * x - m * y - 3 = 0

-- Define the perpendicular condition for slopes of two lines
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Prove the value of m for perpendicular lines
theorem perpendicular_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, line1 x y → ∃ y', line2 x y' m) →
  (∀ x y : ℝ, ∃ x', line1 x y ∧ line2 x' y m) →
  perpendicular_slopes 3 (6 / m) →
  m = -18 :=
by
  sorry

end perpendicular_lines_m_value_l166_166866


namespace kangaroo_meetings_l166_166685

/-- 
Two kangaroos, A and B, start at point A and jump in specific sequences:
- Kangaroo A jumps in the sequence A, B, C, D, E, F, G, H, I, A, B, C, ... in a loop every 9 jumps.
- Kangaroo B jumps in the sequence A, B, D, E, G, H, A, B, D, ... in a loop every 6 jumps.
They start at point A together. Prove that they will land on the same point 226 times after 2017 jumps.
-/
theorem kangaroo_meetings (n : Nat) (ka : Fin 9 → Fin 9) (kb : Fin 6 → Fin 6)
  (hka : ∀ i, ka i = (i + 1) % 9) (hkb : ∀ i, kb i = (i + 1) % 6) :
  n = 2017 →
  -- Prove that the two kangaroos will meet 226 times after 2017 jumps
  ∃ k, k = 226 :=
by
  sorry

end kangaroo_meetings_l166_166685


namespace find_a_l166_166199

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

theorem find_a (a : ℝ) : f_prime a 1 = 2 → a = -3 := by
  intros h
  -- skipping the proof, as it is not required
  sorry

end find_a_l166_166199


namespace max_cells_intersected_10_radius_circle_l166_166271

noncomputable def max_cells_intersected_by_circle (radius : ℝ) (cell_size : ℝ) : ℕ :=
  if radius = 10 ∧ cell_size = 1 then 80 else 0

theorem max_cells_intersected_10_radius_circle :
  max_cells_intersected_by_circle 10 1 = 80 :=
sorry

end max_cells_intersected_10_radius_circle_l166_166271


namespace tan_theta_cos_sin_id_l166_166390

theorem tan_theta_cos_sin_id (θ : ℝ) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) =
  (17 * (Real.sqrt 10 + 1)) / 24 :=
by
  sorry

end tan_theta_cos_sin_id_l166_166390


namespace product_of_two_numbers_l166_166689

theorem product_of_two_numbers (x y : ℕ) 
  (h1 : y = 15 * x) 
  (h2 : x + y = 400) : 
  x * y = 9375 :=
by
  sorry

end product_of_two_numbers_l166_166689


namespace books_per_day_l166_166379

-- Define the condition: Mrs. Hilt reads 15 books in 3 days.
def reads_books_in_days (total_books : ℕ) (days : ℕ) : Prop :=
  total_books = 15 ∧ days = 3

-- Define the theorem to prove that Mrs. Hilt reads 5 books per day.
theorem books_per_day (total_books : ℕ) (days : ℕ) (h : reads_books_in_days total_books days) : total_books / days = 5 :=
by
  -- Stub proof
  sorry

end books_per_day_l166_166379


namespace jeremy_can_win_in_4_turns_l166_166088

noncomputable def game_winnable_in_4_turns (left right : ℕ) : Prop :=
∃ n1 n2 n3 n4 : ℕ,
  n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧
  (left + n1 + n2 + n3 + n4 = right * n1 * n2 * n3 * n4)

theorem jeremy_can_win_in_4_turns (left right : ℕ) (hleft : left = 17) (hright : right = 5) : game_winnable_in_4_turns left right :=
by
  rw [hleft, hright]
  sorry

end jeremy_can_win_in_4_turns_l166_166088


namespace unique_real_solution_k_l166_166156

-- Definitions corresponding to problem conditions:
def is_real_solution (a b k : ℤ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (∃ (x y : ℝ), x * x = a - 1 ∧ y * y = b - 1 ∧ x + y = Real.sqrt (a * b + k))

-- Theorem statement:
theorem unique_real_solution_k (k : ℤ) : (∀ a b : ℤ, is_real_solution a b k → (a = 2 ∧ b = 2)) ↔ k = 0 :=
sorry

end unique_real_solution_k_l166_166156


namespace polynomial_g_correct_l166_166995

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

end polynomial_g_correct_l166_166995


namespace problem_inequality_l166_166616

variable {n : ℕ}
variable (S_n : Finset (Fin n)) (f : Finset (Fin n) → ℝ)

axiom pos_f : ∀ A : Finset (Fin n), 0 < f A
axiom cond_f : ∀ (A : Finset (Fin n)) (x y : Fin n), x ≠ y → f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A

theorem problem_inequality (A B : Finset (Fin n)) : f A * f B ≤ f (A ∪ B) * f (A ∩ B) := sorry

end problem_inequality_l166_166616


namespace different_movies_count_l166_166855

theorem different_movies_count 
    (d_movies : ℕ) (h_movies : ℕ) (a_movies : ℕ) (b_movies : ℕ) (c_movies : ℕ) 
    (together_movies : ℕ) (dha_movies : ℕ) (bc_movies : ℕ) 
    (db_movies : ℕ) (ac_movies : ℕ)
    (H_d : d_movies = 20) (H_h : h_movies = 26) (H_a : a_movies = 35) 
    (H_b : b_movies = 29) (H_c : c_movies = 16)
    (H_together : together_movies = 5)
    (H_dha : dha_movies = 4) (H_bc : bc_movies = 3) 
    (H_db : db_movies = 2) (H_ac : ac_movies = 4) :
    d_movies + h_movies + a_movies + b_movies + c_movies 
    - 4 * together_movies - 3 * dha_movies - 2 * bc_movies - db_movies - 3 * ac_movies = 74 := by sorry

end different_movies_count_l166_166855


namespace eddie_age_l166_166746

theorem eddie_age (Becky_age Irene_age Eddie_age : ℕ)
  (h1 : Becky_age * 2 = Irene_age)
  (h2 : Irene_age = 46)
  (h3 : Eddie_age = 4 * Becky_age) :
  Eddie_age = 92 := by
  sorry

end eddie_age_l166_166746


namespace meter_to_skips_l166_166490

/-!
# Math Proof Problem
Suppose hops, skips and jumps are specific units of length. Given the following conditions:
1. \( b \) hops equals \( c \) skips.
2. \( d \) jumps equals \( e \) hops.
3. \( f \) jumps equals \( g \) meters.

Prove that one meter equals \( \frac{cef}{bdg} \) skips.
-/

theorem meter_to_skips (b c d e f g : ℝ) (h1 : b ≠ 0) (h2 : c ≠ 0) (h3 : d ≠ 0) (h4 : e ≠ 0) (h5 : f ≠ 0) (h6 : g ≠ 0) :
  (1 : ℝ) = (cef) / (bdg) :=
by
  -- skipping the proof
  sorry

end meter_to_skips_l166_166490


namespace trigonometric_identity_l166_166366

theorem trigonometric_identity : 
  Real.cos 6 * Real.cos 36 + Real.sin 6 * Real.cos 54 = Real.sqrt 3 / 2 :=
sorry

end trigonometric_identity_l166_166366


namespace minimum_employees_needed_l166_166780

def min_new_employees (water_pollution: ℕ) (air_pollution: ℕ) (both: ℕ) : ℕ :=
  119 + 34

theorem minimum_employees_needed : min_new_employees 98 89 34 = 153 := 
  by
  sorry

end minimum_employees_needed_l166_166780


namespace new_box_volume_eq_5_76_m3_l166_166726

-- Given conditions:
def original_width_cm := 80
def original_length_cm := 75
def original_height_cm := 120
def conversion_factor_cm3_to_m3 := 1000000

-- New dimensions after doubling
def new_width_cm := 2 * original_width_cm
def new_length_cm := 2 * original_length_cm
def new_height_cm := 2 * original_height_cm

-- Statement of the problem
theorem new_box_volume_eq_5_76_m3 :
  (new_width_cm * new_length_cm * new_height_cm : ℝ) / conversion_factor_cm3_to_m3 = 5.76 := 
  sorry

end new_box_volume_eq_5_76_m3_l166_166726


namespace f_difference_l166_166272

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom local_f : ∀ x : ℝ, -2 < x ∧ x < 0 → f x = 2^x

-- State the problem
theorem f_difference :
  f 2012 - f 2011 = -1 / 2 := sorry

end f_difference_l166_166272


namespace no_a_satisfy_quadratic_equation_l166_166305

theorem no_a_satisfy_quadratic_equation :
  ∀ (a : ℕ), (a > 0) ∧ (a ≤ 100) ∧
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ * x₂ = 2 * a^2 ∧ x₁ + x₂ = -(3*a + 1)) → false := by
  sorry

end no_a_satisfy_quadratic_equation_l166_166305


namespace find_x_in_plane_figure_l166_166428

theorem find_x_in_plane_figure (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 360) 
  (h3 : 2 * x + 160 = 360) : 
  x = 100 :=
by
  sorry

end find_x_in_plane_figure_l166_166428


namespace second_particle_catches_first_l166_166074

open Real

-- Define the distance functions for both particles
def distance_first (t : ℝ) : ℝ := 34 + 5 * t
def distance_second (t : ℝ) : ℝ := 0.25 * t^2 + 2.75 * t

-- The proof statement
theorem second_particle_catches_first : ∃ t : ℝ, distance_second t = distance_first t ∧ t = 17 :=
by
  have : distance_first 17 = 34 + 5 * 17 := by sorry
  have : distance_second 17 = 0.25 * 17^2 + 2.75 * 17 := by sorry
  sorry

end second_particle_catches_first_l166_166074


namespace measure_of_angle_C_l166_166842

theorem measure_of_angle_C (m l : ℝ) (angle_A angle_B angle_D angle_C : ℝ)
  (h_parallel : l = m)
  (h_angle_A : angle_A = 130)
  (h_angle_B : angle_B = 140)
  (h_angle_D : angle_D = 100) :
  angle_C = 90 :=
by
  sorry

end measure_of_angle_C_l166_166842


namespace pre_images_of_one_l166_166800

def f (x : ℝ) := x^3 - x + 1

theorem pre_images_of_one : {x : ℝ | f x = 1} = {-1, 0, 1} :=
by {
  sorry
}

end pre_images_of_one_l166_166800


namespace raft_travel_distance_l166_166990

theorem raft_travel_distance (v_b v_s t : ℝ) (h1 : t > 0) 
  (h2 : v_b + v_s = 90 / t) (h3 : v_b - v_s = 70 / t) : 
  v_s * t = 10 := by
  sorry

end raft_travel_distance_l166_166990


namespace find_north_speed_l166_166265

-- Define the variables and conditions
variables (v : ℝ)  -- the speed of the cyclist going towards the north
def south_speed : ℝ := 25  -- the speed of the cyclist going towards the south is 25 km/h
def time_taken : ℝ := 1.4285714285714286  -- time taken to be 50 km apart
def distance_apart : ℝ := 50  -- distance apart after given time

-- Define the hypothesis based on the conditions
def relative_speed (v : ℝ) : ℝ := v + south_speed
def distance_formula (v : ℝ) : Prop :=
  distance_apart = relative_speed v * time_taken

-- The statement to prove
theorem find_north_speed : distance_formula v → v = 10 :=
  sorry

end find_north_speed_l166_166265


namespace absolute_value_neg_2022_l166_166566

theorem absolute_value_neg_2022 : abs (-2022) = 2022 :=
by sorry

end absolute_value_neg_2022_l166_166566


namespace rhombus_side_length_l166_166955

theorem rhombus_side_length (area d1 d2 side : ℝ) (h_area : area = 24)
(h_d1 : d1 = 6) (h_other_diag : d2 * 6 = 48) (h_side : side = Real.sqrt (3^2 + 4^2)) :
  side = 5 :=
by
  -- This is where the proof would go
  sorry

end rhombus_side_length_l166_166955


namespace marble_distribution_l166_166101

-- Define the problem statement using conditions extracted above
theorem marble_distribution :
  ∃ (A B C D : ℕ), A + B + C + D = 28 ∧
  (A = 7 ∨ B = 7 ∨ C = 7 ∨ D = 7) ∧
  ((A = 7 → B + C + D = 21) ∧
   (B = 7 → A + C + D = 21) ∧
   (C = 7 → A + B + D = 21) ∧
   (D = 7 → A + B + C = 21)) :=
sorry

end marble_distribution_l166_166101


namespace iesha_total_books_l166_166063

theorem iesha_total_books (schoolBooks sportsBooks : ℕ) (h1 : schoolBooks = 19) (h2 : sportsBooks = 39) : schoolBooks + sportsBooks = 58 :=
by
  sorry

end iesha_total_books_l166_166063


namespace largest_crate_dimension_l166_166281

def largest_dimension_of_crate : ℝ := 10

theorem largest_crate_dimension (length width : ℝ) (r : ℝ) (h : ℝ) 
  (h_length : length = 5) (h_width : width = 8) (h_radius : r = 5) (h_height : h >= 10) :
  h = largest_dimension_of_crate :=
by 
  sorry

end largest_crate_dimension_l166_166281


namespace arithmetic_sequence_term_12_l166_166551

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_term_12 (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 :=
by
  -- The following line ensures the theorem compiles correctly.
  sorry

end arithmetic_sequence_term_12_l166_166551


namespace largest_divisor_of_expression_l166_166534

theorem largest_divisor_of_expression 
  (x : ℤ) (h_odd : x % 2 = 1) :
  384 ∣ (8*x + 4) * (8*x + 8) * (4*x + 2) :=
sorry

end largest_divisor_of_expression_l166_166534


namespace amusement_park_l166_166853

theorem amusement_park
  (A : ℕ)
  (adult_ticket_cost : ℕ := 22)
  (child_ticket_cost : ℕ := 7)
  (num_children : ℕ := 2)
  (total_cost : ℕ := 58)
  (cost_eq : adult_ticket_cost * A + child_ticket_cost * num_children = total_cost) :
  A = 2 :=
by {
  sorry
}

end amusement_park_l166_166853


namespace least_number_remainder_l166_166022

noncomputable def lcm_12_15_20_54 : ℕ := 540

theorem least_number_remainder :
  ∀ (n r : ℕ), (n = lcm_12_15_20_54 + r) → 
  (n % 12 = r) ∧ (n % 15 = r) ∧ (n % 20 = r) ∧ (n % 54 = r) → 
  r = 0 :=
by
  sorry

end least_number_remainder_l166_166022


namespace milk_savings_l166_166739

theorem milk_savings :
  let cost_for_two_packs : ℝ := 2.50
  let cost_per_pack_individual : ℝ := 1.30
  let num_packs_per_set := 2
  let num_sets := 10
  let cost_per_pack_set := cost_for_two_packs / num_packs_per_set
  let savings_per_pack := cost_per_pack_individual - cost_per_pack_set
  let total_packs := num_sets * num_packs_per_set
  let total_savings := savings_per_pack * total_packs
  total_savings = 1 :=
by
  sorry

end milk_savings_l166_166739


namespace skittles_total_l166_166149

-- Define the conditions
def skittles_per_friend : ℝ := 40.0
def number_of_friends : ℝ := 5.0

-- Define the target statement using the conditions
theorem skittles_total : (skittles_per_friend * number_of_friends = 200.0) :=
by 
  -- Using sorry to placeholder the proof
  sorry

end skittles_total_l166_166149


namespace correct_calculation_l166_166824

theorem correct_calculation (a : ℝ) : a^4 / a = a^3 :=
by {
  sorry
}

end correct_calculation_l166_166824


namespace smallest_b_for_factorization_l166_166522

theorem smallest_b_for_factorization : ∃ (p q : ℕ), p * q = 2007 ∧ p + q = 232 :=
by
  sorry

end smallest_b_for_factorization_l166_166522


namespace train_crossing_time_l166_166869

theorem train_crossing_time
  (length : ℝ) (speed : ℝ) (time : ℝ)
  (h1 : length = 100) (h2 : speed = 30.000000000000004) :
  time = length / speed :=
by
  sorry

end train_crossing_time_l166_166869


namespace seokgi_money_l166_166497

open Classical

variable (S Y : ℕ)

theorem seokgi_money (h1 : ∃ S, S + 2000 < S + Y + 2000)
                     (h2 : ∃ Y, Y + 1500 < S + Y + 1500)
                     (h3 : 3500 + (S + Y + 2000) = (S + Y) + 3500)
                     (boat_price1: ∀ S, S + 2000 = S + 2000)
                     (boat_price2: ∀ Y, Y + 1500 = Y + 1500) :
  S = 5000 :=
by sorry

end seokgi_money_l166_166497


namespace determine_rectangle_R_area_l166_166180

def side_length_large_square (s : ℕ) : Prop :=
  s = 4

def area_rectangle_R (s : ℕ) (area_R : ℕ) : Prop :=
  s * s - (1 * 4 + 1 * 1) = area_R

theorem determine_rectangle_R_area :
  ∃ (s : ℕ) (area_R : ℕ), side_length_large_square s ∧ area_rectangle_R s area_R :=
by {
  sorry
}

end determine_rectangle_R_area_l166_166180


namespace no_isosceles_triangle_exists_l166_166173

-- Define the grid size
def grid_size : ℕ := 5

-- Define points A and B such that AB is three units horizontally
structure Point where
  x : ℕ
  y : ℕ

-- Define specific points A and B
def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 2⟩

-- Define a function to check if a triangle is isosceles
def is_isosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p1.x - p3.x)^2 + (p1.y - p3.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2

-- Prove that there are no points C that make triangle ABC isosceles
theorem no_isosceles_triangle_exists :
  ¬ ∃ C : Point, C.x ≤ grid_size ∧ C.y ≤ grid_size ∧ is_isosceles A B C :=
by
  sorry

end no_isosceles_triangle_exists_l166_166173


namespace product_units_tens_not_divisible_by_5_l166_166003

-- Define the list of four-digit numbers
def numbers : List ℕ := [4750, 4760, 4775, 4785, 4790]

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function to extract the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Statement: The product of the units digit and the tens digit of the number
-- that is not divisible by 5 in the list is 0
theorem product_units_tens_not_divisible_by_5 : 
  ∃ n ∈ numbers, ¬divisible_by_5 n ∧ (units_digit n * tens_digit n = 0) :=
by sorry

end product_units_tens_not_divisible_by_5_l166_166003


namespace triangle_area_difference_l166_166565

theorem triangle_area_difference 
  (b h : ℝ)
  (hb : 0 < b)
  (hh : 0 < h)
  (A_base : ℝ) (A_height : ℝ)
  (hA_base: A_base = 1.20 * b)
  (hA_height: A_height = 0.80 * h)
  (A_area: ℝ) (B_area: ℝ)
  (hA_area: A_area = 0.5 * A_base * A_height)
  (hB_area: B_area = 0.5 * b * h) :
  (B_area - A_area) / B_area = 0.04 := 
by sorry

end triangle_area_difference_l166_166565


namespace inequality_holds_l166_166422

theorem inequality_holds (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by 
  sorry

end inequality_holds_l166_166422


namespace quadratic_has_two_real_roots_l166_166764

theorem quadratic_has_two_real_roots (a b c : ℝ) (h : a * c < 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * (x1^2) + b * x1 + c = 0 ∧ a * (x2^2) + b * x2 + c = 0) :=
by
  sorry

end quadratic_has_two_real_roots_l166_166764


namespace tom_boxes_needed_l166_166892

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

end tom_boxes_needed_l166_166892


namespace rainfall_march_l166_166972

variable (M A : ℝ)
variable (Hm : A = M - 0.35)
variable (Ha : A = 0.46)

theorem rainfall_march : M = 0.81 := by
  sorry

end rainfall_march_l166_166972


namespace find_constant_c_l166_166345

theorem find_constant_c (c : ℝ) :
  (∀ x y : ℝ, x + y = c ∧ y - (2 + 5) / 2 = x - (8 + 11) / 2) →
  (c = 13) :=
by
  sorry

end find_constant_c_l166_166345


namespace money_put_in_by_A_l166_166996

theorem money_put_in_by_A 
  (B_capital : ℕ := 25000)
  (total_profit : ℕ := 9600)
  (A_management_fee : ℕ := 10)
  (A_total_received : ℕ := 4200) 
  (A_puts_in : ℕ) :
  (A_management_fee * total_profit / 100 
    + (A_puts_in / (A_puts_in + B_capital)) * (total_profit - A_management_fee * total_profit / 100) = A_total_received)
  → A_puts_in = 15000 :=
  by
    sorry

end money_put_in_by_A_l166_166996


namespace boat_speed_is_20_l166_166937

-- Definitions based on conditions from the problem
def boat_speed_still_water (x : ℝ) : Prop := 
  let current_speed := 5
  let downstream_distance := 8.75
  let downstream_time := 21 / 60
  let downstream_speed := x + current_speed
  downstream_speed * downstream_time = downstream_distance

-- The theorem to prove
theorem boat_speed_is_20 : boat_speed_still_water 20 :=
by 
  unfold boat_speed_still_water
  sorry

end boat_speed_is_20_l166_166937


namespace jet_bar_sales_difference_l166_166049

variable (monday_sales : ℕ) (total_target : ℕ) (remaining_target : ℕ)
variable (sales_so_far : ℕ) (tuesday_sales : ℕ)
def JetBarsDifference : Prop :=
  monday_sales = 45 ∧ total_target = 90 ∧ remaining_target = 16 ∧
  sales_so_far = total_target - remaining_target ∧
  tuesday_sales = sales_so_far - monday_sales ∧
  (monday_sales - tuesday_sales = 16)

theorem jet_bar_sales_difference :
  JetBarsDifference 45 90 16 (90 - 16) (90 - 16 - 45) :=
by
  sorry

end jet_bar_sales_difference_l166_166049


namespace complement_of_beta_l166_166030

variable (α β : ℝ)
variable (compl : α + β = 180)
variable (alpha_greater_beta : α > β)

theorem complement_of_beta (h : α + β = 180) (h' : α > β) : 90 - β = (1 / 2) * (α - β) :=
by
  sorry

end complement_of_beta_l166_166030


namespace original_fraction_l166_166182

theorem original_fraction (x y : ℝ) (hxy : x / y = 5 / 7)
  (hx : 1.20 * x / (0.90 * y) = 20 / 21) : x / y = 5 / 7 :=
by {
  sorry
}

end original_fraction_l166_166182


namespace original_price_of_cycle_l166_166291

noncomputable def original_price_given_gain (SP : ℝ) (gain : ℝ) : ℝ :=
  SP / (1 + gain)

theorem original_price_of_cycle (SP : ℝ) (HSP : SP = 1350) (Hgain : gain = 0.5) : 
  original_price_given_gain SP gain = 900 := 
by
  sorry

end original_price_of_cycle_l166_166291


namespace range_of_a_l166_166132

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (1 / 4)^x + (1 / 2)^(x - 1) + a = 0) →
  (-3 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l166_166132


namespace base6_sum_eq_10_l166_166974

theorem base6_sum_eq_10 
  (A B C : ℕ) 
  (hA : 0 < A ∧ A < 6) 
  (hB : 0 < B ∧ B < 6) 
  (hC : 0 < C ∧ C < 6)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_add : A*36 + B*6 + C + B*6 + C = A*36 + C*6 + A) :
  A + B + C = 10 := 
by
  sorry

end base6_sum_eq_10_l166_166974


namespace discount_is_one_percent_l166_166969

/-
  Assuming the following:
  - market_price is the price of one pen in dollars.
  - num_pens is the number of pens bought.
  - cost_price is the total cost price paid by the retailer.
  - profit_percentage is the profit made by the retailer.
  We need to prove that the discount percentage is 1.
-/

noncomputable def discount_percentage
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (SP_per_pen : ℝ) : ℝ :=
  ((market_price - SP_per_pen) / market_price) * 100

theorem discount_is_one_percent
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (buying_condition : cost_price = (market_price * num_pens * (36 / 60)))
  (SP : ℝ)
  (selling_condition : SP = cost_price * (1 + profit_percentage / 100))
  (SP_per_pen : ℝ)
  (sp_per_pen_condition : SP_per_pen = SP / num_pens)
  (profit_condition : profit_percentage = 65) :
  discount_percentage market_price num_pens cost_price profit_percentage SP_per_pen = 1 := by
  sorry

end discount_is_one_percent_l166_166969


namespace Emma_investment_l166_166917

-- Define the necessary context and variables
variable (E : ℝ) -- Emma's investment
variable (B : ℝ := 500) -- Briana's investment which is a known constant
variable (ROI_Emma : ℝ := 0.30 * E) -- Emma's return on investment after 2 years
variable (ROI_Briana : ℝ := 0.20 * B) -- Briana's return on investment after 2 years
variable (ROI_difference : ℝ := ROI_Emma - ROI_Briana) -- The difference in their ROI

theorem Emma_investment :
  ROI_difference = 10 → E = 366.67 :=
by
  intros h
  sorry

end Emma_investment_l166_166917


namespace translation_of_point_l166_166960

variable (P : ℝ × ℝ) (xT yT : ℝ)

def translate_x (P : ℝ × ℝ) (xT : ℝ) : ℝ × ℝ :=
    (P.1 + xT, P.2)

def translate_y (P : ℝ × ℝ) (yT : ℝ) : ℝ × ℝ :=
    (P.1, P.2 + yT)

theorem translation_of_point : translate_y (translate_x (-5, 1) 2) (-4) = (-3, -3) :=
by
  sorry

end translation_of_point_l166_166960


namespace inequality_solution_l166_166914

-- Define the problem statement formally
theorem inequality_solution (x : ℝ)
  (h1 : 2 * x > x + 1)
  (h2 : 4 * x - 1 > 7) :
  x > 2 :=
sorry

end inequality_solution_l166_166914


namespace households_used_both_brands_l166_166811

theorem households_used_both_brands (X : ℕ) : 
  (80 + 60 + X + 3 * X = 260) → X = 30 :=
by
  sorry

end households_used_both_brands_l166_166811


namespace joe_spent_on_food_l166_166988

theorem joe_spent_on_food :
  ∀ (initial_savings flight hotel remaining food : ℝ),
    initial_savings = 6000 →
    flight = 1200 →
    hotel = 800 →
    remaining = 1000 →
    food = initial_savings - remaining - (flight + hotel) →
    food = 3000 :=
by
  intros initial_savings flight hotel remaining food h₁ h₂ h₃ h₄ h₅
  sorry

end joe_spent_on_food_l166_166988


namespace inequality_l166_166322

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) : 
  (1 / (8 * a^2 - 18 * a + 11)) + (1 / (8 * b^2 - 18 * b + 11)) + (1 / (8 * c^2 - 18 * c + 11)) ≤ 3 := 
sorry

end inequality_l166_166322


namespace describe_shape_cylinder_l166_166039

-- Define cylindrical coordinates
structure CylindricalCoordinates where
  r : ℝ -- radial distance
  θ : ℝ -- azimuthal angle
  z : ℝ -- height

-- Define the positive constant c
variable (c : ℝ) (hc : 0 < c)

-- The theorem statement
theorem describe_shape_cylinder (p : CylindricalCoordinates) (h : p.r = c) : 
  ∃ (p : CylindricalCoordinates), p.r = c :=
by
  sorry

end describe_shape_cylinder_l166_166039


namespace doug_age_l166_166269

theorem doug_age (Qaddama Jack Doug : ℕ) 
  (h1 : Qaddama = Jack + 6)
  (h2 : Jack = Doug - 3)
  (h3 : Qaddama = 19) : 
  Doug = 16 := 
by 
  sorry

end doug_age_l166_166269


namespace right_triangle_leg_lengths_l166_166363

theorem right_triangle_leg_lengths (a b c : ℕ) (h : a ^ 2 + b ^ 2 = c ^ 2) (h1: c = 17) (h2: a + (c - b) = 17) (h3: b + (c - a) = 17) : a = 8 ∧ b = 15 :=
by {
  sorry
}

end right_triangle_leg_lengths_l166_166363


namespace find_analytical_expression_of_f_l166_166627

-- Given conditions: f(1/x) = 1/(x+1)
def f (x : ℝ) : ℝ := sorry

-- Domain statement (optional for additional clarity):
def domain (x : ℝ) := x ≠ 0 ∧ x ≠ -1

-- Proof obligation: Prove that f(x) = x / (x + 1)
theorem find_analytical_expression_of_f :
  ∀ x : ℝ, domain x → f x = x / (x + 1) := sorry

end find_analytical_expression_of_f_l166_166627


namespace calculate_expression_l166_166212

noncomputable def expr1 : ℝ := (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3)
noncomputable def expr2 : ℝ := (2 * Real.sqrt 2 - 1) ^ 2
noncomputable def combined_expr : ℝ := expr1 + expr2

-- We need to prove the main statement
theorem calculate_expression : combined_expr = 8 - 4 * Real.sqrt 2 :=
by
  sorry

end calculate_expression_l166_166212


namespace parabola_vertex_l166_166903

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := (x - 2)^2 + 5

-- State the theorem to find the vertex
theorem parabola_vertex : ∃ h k : ℝ, ∀ x : ℝ, parabola_equation x = (x - h)^2 + k ∧ h = 2 ∧ k = 5 :=
by
  sorry

end parabola_vertex_l166_166903


namespace biff_break_even_hours_l166_166494

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l166_166494


namespace seed_mixture_percentage_l166_166985

theorem seed_mixture_percentage (x y : ℝ) 
  (hx : 0.4 * x + 0.25 * y = 30)
  (hxy : x + y = 100) :
  x / 100 = 0.3333 :=
by 
  sorry

end seed_mixture_percentage_l166_166985


namespace sin_gt_sub_cubed_l166_166001

theorem sin_gt_sub_cubed (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  Real.sin x > x - x^3 / 6 := 
by 
  sorry

end sin_gt_sub_cubed_l166_166001


namespace num_factors_of_2310_with_more_than_three_factors_l166_166743

theorem num_factors_of_2310_with_more_than_three_factors : 
  (∃ n : ℕ, n > 0 ∧ ∀ d : ℕ, d ∣ 2310 → (∀ f : ℕ, f ∣ d → f = 1 ∨ f = d ∨ f ∣ d) → 26 = n) := sorry

end num_factors_of_2310_with_more_than_three_factors_l166_166743


namespace max_sum_of_two_integers_l166_166754

theorem max_sum_of_two_integers (x : ℕ) (h : x + 2 * x < 100) : x + 2 * x = 99 :=
sorry

end max_sum_of_two_integers_l166_166754


namespace range_of_m_l166_166961

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) : -6 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l166_166961


namespace solution_l166_166862

noncomputable def triangle_perimeter (AB BC AC : ℕ) (lA lB lC : ℕ) : ℕ :=
  -- This represents the proof problem using the given conditions
  if (AB = 130) ∧ (BC = 240) ∧ (AC = 190)
     ∧ (lA = 65) ∧ (lB = 50) ∧ (lC = 20)
  then
    130  -- The correct answer
  else
    0    -- If the conditions are not met, return 0 

theorem solution :
  triangle_perimeter 130 240 190 65 50 20 = 130 :=
by
  -- This theorem states that with the given conditions, the perimeter of the triangle is 130
  sorry

end solution_l166_166862


namespace number_plus_273_l166_166546

theorem number_plus_273 (x : ℤ) (h : x - 477 = 273) : x + 273 = 1023 := by
  sorry

end number_plus_273_l166_166546


namespace arccos_half_eq_pi_div_three_l166_166815

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l166_166815


namespace solution_set_of_inequality_l166_166207

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 4 * x else (-(x^2 - 4 * x))

theorem solution_set_of_inequality :
  {x : ℝ | f (x - 2) < 5} = {x : ℝ | -3 < x ∧ x < 7} := by
  sorry

end solution_set_of_inequality_l166_166207


namespace tan_half_angle_second_quadrant_l166_166778

variables (θ : ℝ) (k : ℤ)
open Real

theorem tan_half_angle_second_quadrant (h : (π / 2) + 2 * k * π < θ ∧ θ < π + 2 * k * π) : 
  tan (θ / 2) > 1 := 
sorry

end tan_half_angle_second_quadrant_l166_166778


namespace product_N_l166_166822

theorem product_N (A D D1 A1 : ℤ) (N : ℤ) 
  (h1 : D = A - N)
  (h2 : D1 = D + 7)
  (h3 : A1 = A - 2)
  (h4 : |D1 - A1| = 8) : 
  N = 1 → N = 17 → N * 17 = 17 :=
by
  sorry

end product_N_l166_166822


namespace ratio_of_roses_l166_166766

theorem ratio_of_roses (total_flowers tulips carnations roses : ℕ) 
  (h1 : total_flowers = 40) 
  (h2 : tulips = 10) 
  (h3 : carnations = 14) 
  (h4 : roses = total_flowers - (tulips + carnations)) :
  roses / total_flowers = 2 / 5 :=
by
  sorry

end ratio_of_roses_l166_166766


namespace extra_apples_proof_l166_166141

def total_apples (red_apples : ℕ) (green_apples : ℕ) : ℕ :=
  red_apples + green_apples

def apples_taken_by_students (students : ℕ) : ℕ :=
  students

def extra_apples (total_apples : ℕ) (apples_taken : ℕ) : ℕ :=
  total_apples - apples_taken

theorem extra_apples_proof
  (red_apples : ℕ) (green_apples : ℕ) (students : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21) :
  extra_apples (total_apples red_apples green_apples) (apples_taken_by_students students) = 35 :=
by
  sorry

end extra_apples_proof_l166_166141


namespace factory_X_bulbs_percentage_l166_166139

theorem factory_X_bulbs_percentage (p : ℝ) (hx : 0.59 * p + 0.65 * (1 - p) = 0.62) : p = 0.5 :=
sorry

end factory_X_bulbs_percentage_l166_166139


namespace question_one_question_two_l166_166104

variable (b x : ℝ)
def f (x : ℝ) : ℝ := x^2 - b * x + 3

theorem question_one (h : f b 0 = f b 4) : ∃ x1 x2 : ℝ, f b x1 = 0 ∧ f b x2 = 0 ∧ (x1 = 3 ∧ x2 = 1) ∨ (x1 = 1 ∧ x2 = 3) := by 
  sorry

theorem question_two (h1 : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f b x1 = 0 ∧ f b x2 = 0) : b > 4 := by
  sorry

end question_one_question_two_l166_166104


namespace lcm_3_4_6_15_l166_166115

noncomputable def lcm_is_60 : ℕ := 60

theorem lcm_3_4_6_15 : lcm (lcm (lcm 3 4) 6) 15 = lcm_is_60 := 
by 
    sorry

end lcm_3_4_6_15_l166_166115


namespace value_of_N_l166_166258

theorem value_of_N (a b c N : ℚ) 
  (h1 : a + b + c = 120)
  (h2 : a + 8 = N)
  (h3 : 8 * b = N)
  (h4 : c / 8 = N) :
  N = 960 / 73 :=
by
  sorry

end value_of_N_l166_166258


namespace cameron_list_length_l166_166721

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l166_166721


namespace max_triangle_area_l166_166478

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

theorem max_triangle_area
  (x1 y1 x2 y2 : ℝ)
  (hA : parabola x1 y1)
  (hB : parabola x2 y2)
  (h_sum_y : y1 + y2 = 2)
  (h_neq : y1 ≠ y2) :
  ∃ area : ℝ, area = 121 / 12 :=
sorry

end max_triangle_area_l166_166478


namespace price_difference_l166_166617

theorem price_difference (P : ℝ) :
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  difference = 0.24 * P := by
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  sorry

end price_difference_l166_166617


namespace right_triangle_sides_unique_l166_166326

theorem right_triangle_sides_unique (a b c : ℕ) 
  (relatively_prime : Int.gcd (Int.gcd a b) c = 1) 
  (right_triangle : a ^ 2 + b ^ 2 = c ^ 2) 
  (increased_right_triangle : (a + 100) ^ 2 + (b + 100) ^ 2 = (c + 140) ^ 2) : 
  (a = 56 ∧ b = 33 ∧ c = 65) :=
by
  sorry 

end right_triangle_sides_unique_l166_166326


namespace fraction_division_problem_l166_166731

theorem fraction_division_problem :
  (-1/42 : ℚ) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 :=
by
  -- Skipping the proof step as per the instructions
  sorry

end fraction_division_problem_l166_166731


namespace monotonicity_intervals_max_m_value_l166_166127

noncomputable def f (x : ℝ) : ℝ :=  (3 / 2) * x^2 - 3 * Real.log x

theorem monotonicity_intervals :
  (∀ x > (1:ℝ), ∃ ε > (0:ℝ), ∀ y, x < y → y < x + ε → f x < f y)
  ∧ (∀ x, (0:ℝ) < x → x < (1:ℝ) → ∃ ε > (0:ℝ), ∀ y, x - ε < y → y < x → f y < f x) :=
by sorry

theorem max_m_value (m : ℤ) (h : ∀ x > (1:ℝ), f (x * Real.log x + 2 * x - 1) > f (↑m * (x - 1))) :
  m ≤ 4 :=
by sorry

end monotonicity_intervals_max_m_value_l166_166127


namespace value_of_D_l166_166674

variable (L E A D : ℤ)

-- given conditions
def LEAD := 41
def DEAL := 45
def ADDED := 53

-- condition that L = 15
axiom hL : L = 15

-- equations from the problem statement
def eq1 := L + E + A + D = 41
def eq2 := D + E + A + L = 45
def eq3 := A + 3 * D + E = 53

-- stating the problem as proving that D = 4 given the conditions
theorem value_of_D : D = 4 :=
by
  sorry

end value_of_D_l166_166674


namespace hyperbola_eccentricity_l166_166441

open Real

/-- Given the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 2 - x^2 / 8 = 1

/-- Prove the eccentricity of the given hyperbola -/
theorem hyperbola_eccentricity (x y : ℝ) (h : hyperbola_equation x y) : 
  ∃ e : ℝ, e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l166_166441


namespace order_of_three_numbers_l166_166089

theorem order_of_three_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 :=
by
  sorry

end order_of_three_numbers_l166_166089


namespace john_investment_in_bank_a_l166_166844

theorem john_investment_in_bank_a :
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1500 ∧
    x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1740.54 ∧
    x = 695 := sorry

end john_investment_in_bank_a_l166_166844


namespace lcm_36_100_eq_900_l166_166084

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l166_166084


namespace sum_first_five_arithmetic_l166_166775

theorem sum_first_five_arithmetic (a : ℕ → ℝ) (h₁ : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h₂ : a 1 = -1) (h₃ : a 3 = -5) :
  (a 0 + a 1 + a 2 + a 3 + a 4) = -15 :=
by
  sorry

end sum_first_five_arithmetic_l166_166775


namespace rectangle_area_l166_166365

theorem rectangle_area (x w : ℝ) (h₁ : 3 * w = 3 * w) (h₂ : x^2 = 9 * w^2 + w^2) : 
  (3 * w) * w = (3 / 10) * x^2 := 
by
  sorry

end rectangle_area_l166_166365


namespace max_number_of_circular_triples_l166_166999

theorem max_number_of_circular_triples (players : Finset ℕ) (game_results : ℕ → ℕ → Prop) (total_players : players.card = 14)
  (each_plays_13_others : ∀ (p : ℕ) (hp : p ∈ players), ∃ wins losses : Finset ℕ, wins.card = 6 ∧ losses.card = 7 ∧
    (∀ w ∈ wins, game_results p w) ∧ (∀ l ∈ losses, game_results l p)) :
  (∃ (circular_triples : Finset (Finset ℕ)), circular_triples.card = 112 ∧
    ∀ t ∈ circular_triples, t.card = 3 ∧
    (∀ x y z : ℕ, x ∈ t ∧ y ∈ t ∧ z ∈ t → game_results x y ∧ game_results y z ∧ game_results z x)) := 
sorry

end max_number_of_circular_triples_l166_166999


namespace calculate_expression_l166_166620

theorem calculate_expression :
  ((7 / 9) - (5 / 6) + (5 / 18)) * 18 = 4 :=
by
  -- proof to be filled in later.
  sorry

end calculate_expression_l166_166620


namespace correct_addition_l166_166203

-- Define the initial conditions and goal
theorem correct_addition (x : ℕ) : (x + 26 = 61) → (x + 62 = 97) :=
by
  intro h
  -- Proof steps would be provided here
  sorry

end correct_addition_l166_166203


namespace range_of_a_for_inequality_l166_166351

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end range_of_a_for_inequality_l166_166351


namespace dogs_not_doing_anything_l166_166177

def total_dogs : ℕ := 500
def dogs_running : ℕ := 18 * total_dogs / 100
def dogs_playing_with_toys : ℕ := (3 * total_dogs) / 20
def dogs_barking : ℕ := 7 * total_dogs / 100
def dogs_digging_holes : ℕ := total_dogs / 10
def dogs_competing : ℕ := 12
def dogs_sleeping : ℕ := (2 * total_dogs) / 25
def dogs_eating_treats : ℕ := total_dogs / 5

def dogs_doing_anything : ℕ := dogs_running + dogs_playing_with_toys + dogs_barking + dogs_digging_holes + dogs_competing + dogs_sleeping + dogs_eating_treats

theorem dogs_not_doing_anything : total_dogs - dogs_doing_anything = 98 :=
by
  -- proof steps would go here
  sorry

end dogs_not_doing_anything_l166_166177


namespace description_of_S_l166_166951

noncomputable def S := {p : ℝ × ℝ | (3 = (p.1 + 2) ∧ p.2 - 5 ≤ 3) ∨ 
                                      (3 = (p.2 - 5) ∧ p.1 + 2 ≤ 3) ∨ 
                                      (p.1 + 2 = p.2 - 5 ∧ 3 ≤ p.1 + 2 ∧ 3 ≤ p.2 - 5)}

theorem description_of_S :
  S = {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 ≤ 8) ∨ 
                    (p.2 = 8 ∧ p.1 ≤ 1) ∨ 
                    (p.2 = p.1 + 7 ∧ p.1 ≥ 1 ∧ p.2 ≥ 8)} :=
sorry

end description_of_S_l166_166951


namespace modulus_z_eq_sqrt_10_l166_166992

noncomputable def z : ℂ := (1 + 7 * Complex.I) / (2 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := sorry

end modulus_z_eq_sqrt_10_l166_166992


namespace right_triangle_legs_sum_l166_166769

theorem right_triangle_legs_sum (x : ℕ) (hx1 : x * x + (x + 1) * (x + 1) = 41 * 41) : x + (x + 1) = 59 :=
by sorry

end right_triangle_legs_sum_l166_166769


namespace geometric_sequence_sum_first_five_terms_l166_166813

theorem geometric_sequence_sum_first_five_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 30)
  (h_geom : ∀ n, a (n + 1) = a n * q) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
sorry

end geometric_sequence_sum_first_five_terms_l166_166813


namespace half_angle_quadrant_l166_166593

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + (3 * π / 2)) : 
  (∃ j : ℤ, j * π + (π / 2) < (α / 2) ∧ (α / 2) < j * π + (3 * π / 4)) :=
  by sorry

end half_angle_quadrant_l166_166593


namespace valid_n_values_l166_166763

theorem valid_n_values :
  {n : ℕ | ∀ a : ℕ, a^(n+1) ≡ a [MOD n]} = {1, 2, 6, 42, 1806} :=
sorry

end valid_n_values_l166_166763


namespace compute_j_in_polynomial_arithmetic_progression_l166_166005

theorem compute_j_in_polynomial_arithmetic_progression 
  (P : Polynomial ℝ)
  (roots : Fin 4 → ℝ)
  (hP : P = Polynomial.C 400 + Polynomial.X * (Polynomial.C k + Polynomial.X * (Polynomial.C j + Polynomial.X * (Polynomial.C 0 + Polynomial.X))))
  (arithmetic_progression : ∃ b d : ℝ, roots 0 = b ∧ roots 1 = b + d ∧ roots 2 = b + 2 * d ∧ roots 3 = b + 3 * d ∧ Polynomial.degree P = 4) :
  j = -200 :=
by
  sorry

end compute_j_in_polynomial_arithmetic_progression_l166_166005


namespace proof_problem_l166_166691

theorem proof_problem
  (a b c : ℂ)
  (h1 : ac / (a + b) + ba / (b + c) + cb / (c + a) = -4)
  (h2 : bc / (a + b) + ca / (b + c) + ab / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 7 := 
sorry

end proof_problem_l166_166691


namespace multiplicative_inverse_mod_l166_166278

-- We define our variables
def a := 154
def m := 257
def inv_a := 20

-- Our main theorem stating that inv_a is indeed the multiplicative inverse of a modulo m
theorem multiplicative_inverse_mod : (a * inv_a) % m = 1 := by
  sorry

end multiplicative_inverse_mod_l166_166278


namespace grain_remaining_l166_166368

def originalGrain : ℕ := 50870
def spilledGrain : ℕ := 49952
def remainingGrain : ℕ := 918

theorem grain_remaining : originalGrain - spilledGrain = remainingGrain := by
  -- calculations are omitted in the theorem statement
  sorry

end grain_remaining_l166_166368


namespace sum_of_adjacent_to_7_l166_166362

/-- Define the divisors of 245, excluding 1 -/
def divisors245 : Set ℕ := {5, 7, 35, 49, 245}

/-- Define the adjacency condition to ensure every pair of adjacent integers has a common factor greater than 1 -/
def adjacency_condition (a b : ℕ) : Prop := (a ≠ b) ∨ (Nat.gcd a b > 1)

/-- Prove the sum of the two integers adjacent to 7 in the given condition is 294. -/
theorem sum_of_adjacent_to_7 (d1 d2 : ℕ) (h1 : d1 ∈ divisors245) (h2 : d2 ∈ divisors245) 
    (adj1 : adjacency_condition 7 d1) (adj2 : adjacency_condition 7 d2) : 
    d1 + d2 = 294 := 
sorry

end sum_of_adjacent_to_7_l166_166362


namespace positive_integers_satisfy_inequality_l166_166454

theorem positive_integers_satisfy_inequality :
  ∀ (n : ℕ), 2 * n - 5 < 5 - 2 * n ↔ n = 1 ∨ n = 2 :=
by
  intro n
  sorry

end positive_integers_satisfy_inequality_l166_166454


namespace solution_set_of_inequality_l166_166688

theorem solution_set_of_inequality (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_l166_166688


namespace correct_incorrect_difference_l166_166029

variable (x : ℝ)

theorem correct_incorrect_difference : (x - 2152) - (x - 1264) = 888 := by
  sorry

end correct_incorrect_difference_l166_166029


namespace correct_equation_l166_166930

theorem correct_equation (x : ℕ) (h : x ≤ 26) :
    let a_parts := 2100
    let b_parts := 1200
    let total_workers := 26
    let a_rate := 30
    let b_rate := 20
    let type_a_time := (a_parts : ℚ) / (a_rate * x)
    let type_b_time := (b_parts : ℚ) / (b_rate * (total_workers - x))
    type_a_time = type_b_time :=
by
    sorry

end correct_equation_l166_166930


namespace correct_profit_equation_l166_166085

def total_rooms : ℕ := 50
def initial_price : ℕ := 180
def price_increase_step : ℕ := 10
def cost_per_occupied_room : ℕ := 20
def desired_profit : ℕ := 10890

theorem correct_profit_equation (x : ℕ) : 
  (x - cost_per_occupied_room : ℤ) * (total_rooms - (x - initial_price : ℤ) / price_increase_step) = desired_profit :=
by sorry

end correct_profit_equation_l166_166085


namespace blue_paint_quantity_l166_166213

-- Conditions
def paint_ratio (r b y w : ℕ) : Prop := r = 2 * w / 4 ∧ b = 3 * w / 4 ∧ y = 1 * w / 4 ∧ w = 4 * (r + b + y + w) / 10

-- Given
def quart_white_paint : ℕ := 16

-- Prove that Victor should use 12 quarts of blue paint
theorem blue_paint_quantity (r b y w : ℕ) (h : paint_ratio r b y w) (hw : w = quart_white_paint) : 
  b = 12 := by
  sorry

end blue_paint_quantity_l166_166213


namespace sum_x_y_z_eq_3_or_7_l166_166610

theorem sum_x_y_z_eq_3_or_7 (x y z : ℝ) (h1 : x + y / z = 2) (h2 : y + z / x = 2) (h3 : z + x / y = 2) : x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_x_y_z_eq_3_or_7_l166_166610


namespace rain_on_both_days_l166_166991

-- Define the events probabilities
variables (P_M P_T P_N P_MT : ℝ)

-- Define the initial conditions
axiom h1 : P_M = 0.6
axiom h2 : P_T = 0.55
axiom h3 : P_N = 0.25

-- Define the statement to prove
theorem rain_on_both_days : P_MT = 0.4 :=
by
  -- The proof is omitted for now
  sorry

end rain_on_both_days_l166_166991


namespace real_no_impure_l166_166516

theorem real_no_impure {x : ℝ} (h1 : x^2 - 1 = 0) (h2 : x^2 + 3 * x + 2 ≠ 0) : x = 1 :=
by
  sorry

end real_no_impure_l166_166516


namespace total_hours_worked_l166_166567

def hours_per_day : ℕ := 8 -- Frank worked 8 hours on each day
def number_of_days : ℕ := 4 -- First 4 days of the week

theorem total_hours_worked : hours_per_day * number_of_days = 32 := by
  sorry

end total_hours_worked_l166_166567


namespace rectangle_perimeter_l166_166723

variables (L W : ℕ)

-- conditions
def conditions : Prop :=
  L - 4 = W + 3 ∧
  (L - 4) * (W + 3) = L * W

-- prove the solution
theorem rectangle_perimeter (h : conditions L W) : 2 * L + 2 * W = 50 := sorry

end rectangle_perimeter_l166_166723


namespace closed_polygonal_chain_exists_l166_166631

theorem closed_polygonal_chain_exists (n m : ℕ) : 
  ((n % 2 = 1 ∨ m % 2 = 1) ↔ 
   ∃ (length : ℕ), length = (n + 1) * (m + 1) ∧ length % 2 = 0) :=
by sorry

end closed_polygonal_chain_exists_l166_166631


namespace abs_five_minus_e_l166_166206

noncomputable def e : ℝ := Real.exp 1

theorem abs_five_minus_e : |5 - e| = 5 - e := by
  sorry

end abs_five_minus_e_l166_166206


namespace pirates_total_distance_l166_166036

def adjusted_distance_1 (d: ℝ) : ℝ := d * 1.10
def adjusted_distance_2 (d: ℝ) : ℝ := d * 1.15
def adjusted_distance_3 (d: ℝ) : ℝ := d * 1.20
def adjusted_distance_4 (d: ℝ) : ℝ := d * 1.25

noncomputable def total_distance : ℝ := 
  let first_island := (adjusted_distance_1 10) + (adjusted_distance_1 15) + (adjusted_distance_1 20)
  let second_island := adjusted_distance_2 40
  let third_island := (adjusted_distance_3 25) + (adjusted_distance_3 20) + (adjusted_distance_3 25) + (adjusted_distance_3 20)
  let fourth_island := adjusted_distance_4 35
  first_island + second_island + third_island + fourth_island

theorem pirates_total_distance : total_distance = 247.25 := by
  sorry

end pirates_total_distance_l166_166036


namespace find_other_factor_l166_166121

theorem find_other_factor 
    (w : ℕ) 
    (hw_pos : w > 0) 
    (h_factor : ∃ (x y : ℕ), 936 * w = x * y ∧ (2 ^ 5 ∣ x) ∧ (3 ^ 3 ∣ x)) 
    (h_ww : w = 156) : 
    ∃ (other_factor : ℕ), 936 * w = 156 * other_factor ∧ other_factor = 72 := 
by 
    sorry

end find_other_factor_l166_166121


namespace cylinder_volume_l166_166634

theorem cylinder_volume (r h V: ℝ) (r_pos: r = 4) (lateral_area: 2 * 3.14 * r * h = 62.8) : 
    V = 125600 :=
by
  sorry

end cylinder_volume_l166_166634


namespace original_length_l166_166652

-- Definitions based on conditions
def length_sawed_off : ℝ := 0.33
def remaining_length : ℝ := 0.08

-- The problem statement translated to a Lean 4 theorem
theorem original_length (L : ℝ) (h1 : L = length_sawed_off + remaining_length) : 
  L = 0.41 :=
by
  sorry

end original_length_l166_166652


namespace sin_330_l166_166175

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l166_166175


namespace angle_between_vectors_l166_166359

noncomputable def vec_a : ℝ × ℝ := (-2 * Real.sqrt 3, 2)
noncomputable def vec_b : ℝ × ℝ := (1, - Real.sqrt 3)

-- Define magnitudes
noncomputable def mag_a : ℝ := Real.sqrt ((-2 * Real.sqrt 3) ^ 2 + 2^2)
noncomputable def mag_b : ℝ := Real.sqrt (1^2 + (- Real.sqrt 3) ^ 2)

-- Define the dot product
noncomputable def dot_product : ℝ := (-2 * Real.sqrt 3) * 1 + 2 * (- Real.sqrt 3)

-- Define cosine of the angle theta
-- We use mag_a and mag_b defined above
noncomputable def cos_theta : ℝ := dot_product / (mag_a * mag_b)

-- Define the angle theta, within the range [0, π]
noncomputable def theta : ℝ := Real.arccos cos_theta

-- The expected result is θ = 5π / 6
theorem angle_between_vectors : theta = (5 * Real.pi) / 6 :=
by
  sorry

end angle_between_vectors_l166_166359


namespace sufficient_not_necessary_condition_l166_166728

noncomputable def has_negative_root (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x < 0

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ x < 0) ↔ (a < 0) :=
sorry

end sufficient_not_necessary_condition_l166_166728


namespace volume_of_pure_water_added_l166_166083

theorem volume_of_pure_water_added 
  (V0 : ℝ) (P0 : ℝ) (Pf : ℝ) 
  (V0_eq : V0 = 50) 
  (P0_eq : P0 = 0.30) 
  (Pf_eq : Pf = 0.1875) : 
  ∃ V : ℝ, V = 30 ∧ (15 / (V0 + V)) = Pf := 
by
  sorry

end volume_of_pure_water_added_l166_166083


namespace fraction_value_l166_166690

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem
theorem fraction_value (h : 2 * x = -y) : (x * y) / (x^2 - y^2) = 2 / 3 :=
by
  sorry

end fraction_value_l166_166690


namespace rem_frac_l166_166114

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_frac : rem (5/7 : ℚ) (3/4 : ℚ) = (5/7 : ℚ) := 
by 
  sorry

end rem_frac_l166_166114


namespace customer_paid_l166_166761

def cost_price : ℝ := 7999.999999999999
def percentage_markup : ℝ := 0.10
def selling_price (cp : ℝ) (markup : ℝ) := cp + cp * markup

theorem customer_paid :
  selling_price cost_price percentage_markup = 8800 :=
by
  sorry

end customer_paid_l166_166761


namespace distance_halfway_along_orbit_l166_166740

-- Define the conditions
variables (perihelion aphelion : ℝ) (perihelion_dist : perihelion = 3) (aphelion_dist : aphelion = 15)

-- State the theorem
theorem distance_halfway_along_orbit : 
  ∃ d, d = (perihelion + aphelion) / 2 ∧ d = 9 :=
by
  sorry

end distance_halfway_along_orbit_l166_166740


namespace Katie_marble_count_l166_166369

theorem Katie_marble_count :
  ∀ (pink_marbles orange_marbles purple_marbles total_marbles : ℕ),
  pink_marbles = 13 →
  orange_marbles = pink_marbles - 9 →
  purple_marbles = 4 * orange_marbles →
  total_marbles = pink_marbles + orange_marbles + purple_marbles →
  total_marbles = 33 :=
by
  intros pink_marbles orange_marbles purple_marbles total_marbles
  intros hpink horange hpurple htotal
  sorry

end Katie_marble_count_l166_166369


namespace tina_brownies_per_meal_l166_166246

-- Define the given conditions
def total_brownies : ℕ := 24
def days : ℕ := 5
def meals_per_day : ℕ := 2
def brownies_by_husband_per_day : ℕ := 1
def total_brownies_shared_with_guests : ℕ := 4
def total_brownies_left : ℕ := 5

-- Conjecture: How many brownies did Tina have with each meal
theorem tina_brownies_per_meal :
  (total_brownies 
  - (brownies_by_husband_per_day * days) 
  - total_brownies_shared_with_guests 
  - total_brownies_left)
  / (days * meals_per_day) = 1 :=
by
  sorry

end tina_brownies_per_meal_l166_166246


namespace average_cost_of_testing_l166_166391

theorem average_cost_of_testing (total_machines : Nat) (faulty_machines : Nat) (cost_per_test : Nat) 
  (h_total : total_machines = 5) (h_faulty : faulty_machines = 2) (h_cost : cost_per_test = 1000) :
  (2000 * (2 / 5 * 1 / 4) + 3000 * (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3) + 
  4000 * (1 - (2 / 5 * 1 / 4) - (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3))) = 3500 :=
  by
  sorry

end average_cost_of_testing_l166_166391


namespace intersection_complement_N_l166_166675

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}
def C_U_M : Set ℕ := U \ M

theorem intersection_complement_N : (C_U_M ∩ N) = {4, 6} :=
by
  sorry

end intersection_complement_N_l166_166675


namespace fruit_prices_l166_166178

theorem fruit_prices :
  (∃ x y : ℝ, 60 * x + 40 * y = 1520 ∧ 30 * x + 50 * y = 1360 ∧ x = 12 ∧ y = 20) :=
sorry

end fruit_prices_l166_166178


namespace abc_sum_71_l166_166630

theorem abc_sum_71 (a b c : ℝ) (h₁ : ∀ x, (x ≤ -3 ∨ 23 ≤ x ∧ x < 27) ↔ ( (x - a) * (x - b) / (x - c) ≥ 0)) (h₂ : a < b) : 
  a + 2 * b + 3 * c = 71 :=
sorry

end abc_sum_71_l166_166630


namespace total_equipment_cost_l166_166424

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l166_166424


namespace find_a_plus_b_l166_166103

/-- Given the sets M = {x | |x-4| + |x-1| < 5} and N = {x | a < x < 6}, and M ∩ N = {2, b}, 
prove that a + b = 7. -/
theorem find_a_plus_b 
  (M : Set ℝ := { x | |x - 4| + |x - 1| < 5 }) 
  (N : Set ℝ := { x | a < x ∧ x < 6 }) 
  (a b : ℝ)
  (h_inter : M ∩ N = {2, b}) :
  a + b = 7 :=
sorry

end find_a_plus_b_l166_166103


namespace evaluate_f_difference_l166_166230

def f (x : ℝ) : ℝ := x^6 - 2 * x^4 + 7 * x

theorem evaluate_f_difference :
  f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_difference_l166_166230


namespace sum_of_ages_is_l166_166768

-- Define the ages of the triplets and twins
def age_triplet (x : ℕ) := x
def age_twin (x : ℕ) := x - 3

-- Define the total age sum
def total_age_sum (x : ℕ) := 3 * age_triplet x + 2 * age_twin x

-- State the theorem
theorem sum_of_ages_is (x : ℕ) (h : total_age_sum x = 89) : ∃ x : ℕ, total_age_sum x = 89 := 
sorry

end sum_of_ages_is_l166_166768


namespace find_m_l166_166164

open Set

variable (A B : Set ℝ) (m : ℝ)

theorem find_m (h : A = {-1, 2, 2 * m - 1}) (h2 : B = {2, m^2}) (h3 : B ⊆ A) : m = 1 := 
by
  sorry

end find_m_l166_166164


namespace students_per_bench_l166_166608

theorem students_per_bench (num_male num_benches : ℕ) (h₁ : num_male = 29) (h₂ : num_benches = 29) (h₃ : ∀ num_female, num_female = 4 * num_male) : 
  ((29 + 4 * 29) / 29) = 5 :=
by
  sorry

end students_per_bench_l166_166608


namespace S_5_value_l166_166532

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom a2a4 (h : geometric_sequence a) : a 1 * a 3 = 16
axiom S3 : S 3 = 7

theorem S_5_value 
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = a 0 * (1 - (a 1)^(n)) / (1 - a 1)) :
  S 5 = 31 :=
sorry

end S_5_value_l166_166532


namespace louis_age_l166_166916

variable (L J M : ℕ) -- L for Louis, J for Jerica, and M for Matilda

theorem louis_age : 
  (M = 35) ∧ (M = J + 7) ∧ (J = 2 * L) → L = 14 := 
by 
  intro h 
  sorry

end louis_age_l166_166916


namespace not_divisible_by_1980_divisible_by_1981_l166_166406

open Nat

theorem not_divisible_by_1980 (x : ℕ) : ¬ (2^100 * x - 1) % 1980 = 0 := by
sorry

theorem divisible_by_1981 : ∃ x : ℕ, (2^100 * x - 1) % 1981 = 0 := by
sorry

end not_divisible_by_1980_divisible_by_1981_l166_166406


namespace net_sag_calculation_l166_166023

open Real

noncomputable def sag_of_net (m1 m2 h1 h2 x1 : ℝ) : ℝ :=
  let g := 9.81
  let a := 28
  let b := -1.75
  let c := -50.75
  let D := b^2 - 4*a*c
  let sqrtD := sqrt D
  (1.75 + sqrtD) / (2 * a)

theorem net_sag_calculation :
  let m1 := 78.75
  let x1 := 1
  let h1 := 15
  let m2 := 45
  let h2 := 29
  sag_of_net m1 m2 h1 h2 x1 = 1.38 := 
by
  sorry

end net_sag_calculation_l166_166023


namespace percentage_error_square_area_l166_166343

theorem percentage_error_square_area (s : ℝ) (h : s > 0) :
  let s' := (1.02 * s)
  let actual_area := s^2
  let measured_area := s'^2
  let error_area := measured_area - actual_area
  let percentage_error := (error_area / actual_area) * 100
  percentage_error = 4.04 := 
sorry

end percentage_error_square_area_l166_166343


namespace plane_tiled_squares_triangles_percentage_l166_166439

theorem plane_tiled_squares_triangles_percentage :
    (percent_triangle_area : ℚ) = 625 / 10000 := sorry

end plane_tiled_squares_triangles_percentage_l166_166439


namespace cos_pi_plus_alpha_l166_166377

theorem cos_pi_plus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π + α) = - 1 / 3 :=
by
  sorry

end cos_pi_plus_alpha_l166_166377


namespace sophie_saves_money_l166_166429

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end sophie_saves_money_l166_166429


namespace select_terms_from_sequence_l166_166784

theorem select_terms_from_sequence (k : ℕ) (hk : k ≥ 3) :
  ∃ (terms : Fin k → ℚ), (∀ i j : Fin k, i < j → (terms j - terms i) = (j.val - i.val) / k!) ∧
  (∀ i : Fin k, terms i ∈ {x : ℚ | ∃ n : ℕ, x = 1 / (n : ℚ)}) :=
by
  sorry

end select_terms_from_sequence_l166_166784


namespace fraction_e_over_d_l166_166563

theorem fraction_e_over_d :
  ∃ (d e : ℝ), (∀ (x : ℝ), x^2 + 2600 * x + 2600 = (x + d)^2 + e) ∧ e / d = -1298 :=
by 
  sorry

end fraction_e_over_d_l166_166563


namespace solve_for_F_l166_166590

theorem solve_for_F (F C : ℝ) (h₁ : C = 4 / 7 * (F - 40)) (h₂ : C = 25) : F = 83.75 :=
sorry

end solve_for_F_l166_166590


namespace parabola_directrix_l166_166762

theorem parabola_directrix {x y : ℝ} (h : y^2 = 6 * x) : x = -3 / 2 := 
sorry

end parabola_directrix_l166_166762


namespace number_of_people_in_group_l166_166286

theorem number_of_people_in_group :
  ∀ (N : ℕ), (75 - 35) = 5 * N → N = 8 :=
by
  intros N h
  sorry

end number_of_people_in_group_l166_166286


namespace product_of_invertible_labels_l166_166209

def f1 (x : ℤ) : ℤ := x^3 - 2 * x
def f2 (x : ℤ) : ℤ := x - 2
def f3 (x : ℤ) : ℤ := 2 - x

theorem product_of_invertible_labels :
  (¬ ∃ inv : ℤ → ℤ, f1 (inv 0) = 0 ∧ ∀ x : ℤ, f1 (inv (f1 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f2 (inv 0) = 0 ∧ ∀ x : ℤ, f2 (inv (f2 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f3 (inv 0) = 0 ∧ ∀ x : ℤ, f3 (inv (f3 x)) = x) →
  (2 * 3 = 6) :=
by sorry

end product_of_invertible_labels_l166_166209


namespace prove_ab_leq_one_l166_166331

theorem prove_ab_leq_one (a b : ℝ) (h : (a + b + a) * (a + b + b) = 9) : ab ≤ 1 := 
by
  sorry

end prove_ab_leq_one_l166_166331


namespace circle_radius_l166_166956

theorem circle_radius (x y : ℝ) :
  y = (x - 2)^2 ∧ x - 3 = (y + 1)^2 →
  (∃ c d r : ℝ, (c, d) = (3/2, -1/2) ∧ r^2 = 25/4) :=
by
  sorry

end circle_radius_l166_166956


namespace ladder_alley_width_l166_166872

theorem ladder_alley_width (l : ℝ) (m : ℝ) (w : ℝ) (h : m = l / 2) :
  w = (l * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end ladder_alley_width_l166_166872


namespace choose_rectangles_l166_166086

theorem choose_rectangles (n : ℕ) (hn : n ≥ 2) :
  ∃ (chosen_rectangles : Finset (ℕ × ℕ)), 
    (chosen_rectangles.card = 2 * n ∧
     ∀ (r1 r2 : ℕ × ℕ), r1 ∈ chosen_rectangles → r2 ∈ chosen_rectangles →
      (r1.fst ≤ r2.fst ∧ r1.snd ≤ r2.snd) ∨ 
      (r2.fst ≤ r1.fst ∧ r2.snd ≤ r1.snd) ∨ 
      (r1.fst ≤ r2.snd ∧ r1.snd ≤ r2.fst) ∨ 
      (r2.fst ≤ r1.snd ∧ r2.snd <= r1.fst)) :=
sorry

end choose_rectangles_l166_166086


namespace coplanar_AD_eq_linear_combination_l166_166341

-- Define the points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨4, 1, 3⟩
def B : Point3D := ⟨2, 3, 1⟩
def C : Point3D := ⟨3, 7, -5⟩
def D : Point3D := ⟨11, -1, 3⟩

-- Define the vectors
def vector (P Q : Point3D) : Point3D := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def AB := vector A B
def AC := vector A C
def AD := vector A D

-- Coplanar definition: AD = λ AB + μ AC
theorem coplanar_AD_eq_linear_combination (lambda mu : ℝ) :
  AD = ⟨lambda * 2 + mu * (-1), lambda * (-2) + mu * 6, lambda * (-2) + mu * (-8)⟩ :=
sorry

end coplanar_AD_eq_linear_combination_l166_166341


namespace proof_problem_l166_166825

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end proof_problem_l166_166825


namespace inequality_2_inequality_4_l166_166446

variables (a b : ℝ)
variables (h₁ : 0 < a) (h₂ : 0 < b)

theorem inequality_2 (h₁ : 0 < a) (h₂ : 0 < b) : a > |a - b| - b :=
by
  sorry

theorem inequality_4 (h₁ : 0 < a) (h₂ : 0 < b) : ab + 2 / ab > 2 :=
by
  sorry

end inequality_2_inequality_4_l166_166446


namespace tangent_line_eq_l166_166079

def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

theorem tangent_line_eq (x y : ℝ) (h : (x, y) = (1, 10)) : 
  (7 * x - y + 3 = 0) :=
sorry

end tangent_line_eq_l166_166079


namespace derivative_of_f_l166_166461

noncomputable def f (x : ℝ) : ℝ :=
  (Nat.choose 4 0 : ℝ) - (Nat.choose 4 1 : ℝ) * x + (Nat.choose 4 2 : ℝ) * x^2 - (Nat.choose 4 3 : ℝ) * x^3 + (Nat.choose 4 4 : ℝ) * x^4

theorem derivative_of_f : 
  ∀ (x : ℝ), (deriv f x) = 4 * (-1 + x)^3 :=
by
  sorry

end derivative_of_f_l166_166461


namespace scholarship_total_l166_166953

-- Definitions of the money received by Wendy, Kelly, Nina, and Jason based on the given conditions
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000
def jason_scholarship : ℕ := (3 * kelly_scholarship) / 4

-- Total amount of scholarships
def total_scholarship : ℕ := wendy_scholarship + kelly_scholarship + nina_scholarship + jason_scholarship

-- The proof statement that needs to be proven
theorem scholarship_total : total_scholarship = 122000 := by
  -- Here we use 'sorry' to indicate that the proof is not provided.
  sorry

end scholarship_total_l166_166953


namespace correct_operation_l166_166877

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end correct_operation_l166_166877


namespace union_sets_eq_l166_166318

-- Definitions of the sets M and N according to the conditions.
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- The theorem we want to prove
theorem union_sets_eq :
  (M ∪ N) = Set.Icc 0 1 :=
by
  sorry

end union_sets_eq_l166_166318


namespace mary_walking_speed_l166_166483

-- Definitions based on the conditions:
def distance_sharon (t : ℝ) : ℝ := 6 * t
def distance_mary (x t : ℝ) : ℝ := x * t
def total_distance (x t : ℝ) : ℝ := distance_sharon t + distance_mary x t

-- Lean statement to prove that the speed x is 4 given the conditions
theorem mary_walking_speed (x : ℝ) (t : ℝ) (h1 : t = 0.3) (h2 : total_distance x t = 3) : x = 4 :=
by
  sorry

end mary_walking_speed_l166_166483


namespace total_money_correct_l166_166165

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end total_money_correct_l166_166165


namespace compound_proposition_truth_l166_166445

theorem compound_proposition_truth (p q : Prop) (h1 : ¬p ∨ ¬q = False) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end compound_proposition_truth_l166_166445


namespace planted_fraction_l166_166820

theorem planted_fraction (length width radius : ℝ) (h_field : length * width = 24)
  (h_circle : π * radius^2 = π) : (24 - π) / 24 = (24 - π) / 24 :=
by
  -- all proofs are skipped
  sorry

end planted_fraction_l166_166820


namespace algebraic_expression_value_l166_166499

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1) * (m - 1) + 2022 = 2024 :=
by
  sorry

end algebraic_expression_value_l166_166499


namespace trimino_tilings_greater_l166_166146

noncomputable def trimino_tilings (n : ℕ) : ℕ := sorry
noncomputable def domino_tilings (n : ℕ) : ℕ := sorry

theorem trimino_tilings_greater (n : ℕ) (h : n > 1) : trimino_tilings (3 * n) > domino_tilings (2 * n) :=
sorry

end trimino_tilings_greater_l166_166146


namespace sequence_general_term_l166_166255

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l166_166255


namespace cyclist_rate_l166_166251

theorem cyclist_rate 
  (rate_hiker : ℝ := 4)
  (wait_time_1 : ℝ := 5 / 60)
  (wait_time_2 : ℝ := 10.000000000000002 / 60)
  (hiker_distance : ℝ := rate_hiker * wait_time_2)
  (cyclist_distance : ℝ := hiker_distance)
  (cyclist_rate := cyclist_distance / wait_time_1) :
  cyclist_rate = 8 := by 
sorry

end cyclist_rate_l166_166251


namespace smaller_solution_quadratic_equation_l166_166376

theorem smaller_solution_quadratic_equation :
  (∀ x : ℝ, x^2 + 7 * x - 30 = 0 → x = -10 ∨ x = 3) → -10 = min (-10) 3 :=
by
  sorry

end smaller_solution_quadratic_equation_l166_166376


namespace slope_of_line_l166_166417

theorem slope_of_line : ∀ x y : ℝ, 3 * y + 2 * x = 6 * x - 9 → ∃ m b : ℝ, y = m * x + b ∧ m = -4 / 3 :=
by
  -- Sorry to skip proof
  sorry

end slope_of_line_l166_166417


namespace zero_of_f_l166_166936

noncomputable def f (x : ℝ) : ℝ := (|Real.log x - Real.log 2|) - (1 / 3) ^ x

theorem zero_of_f :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ (f x1 = 0) ∧ (f x2 = 0) ∧
  (1 < x1 ∧ x1 < 2) ∧ (2 < x2) := 
sorry

end zero_of_f_l166_166936


namespace sisters_work_together_days_l166_166521

-- Definitions based on conditions
def task_completion_rate_older_sister : ℚ := 1/10
def task_completion_rate_younger_sister : ℚ := 1/20
def work_done_by_older_sister_alone : ℚ := 4 * task_completion_rate_older_sister
def remaining_task_after_older_sister : ℚ := 1 - work_done_by_older_sister_alone
def combined_work_rate : ℚ := task_completion_rate_older_sister + task_completion_rate_younger_sister

-- Statement of the proof problem
theorem sisters_work_together_days : 
  (combined_work_rate * x = remaining_task_after_older_sister) → 
  (x = 4) :=
by
  sorry

end sisters_work_together_days_l166_166521


namespace min_value_expression_is_4_l166_166430

noncomputable def min_value_expression (x : ℝ) : ℝ :=
(3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1)

theorem min_value_expression_is_4 : ∃ x : ℝ, min_value_expression x = 4 :=
sorry

end min_value_expression_is_4_l166_166430


namespace probability_of_event_l166_166040

noncomputable def interval_probability : ℝ :=
  if 0 ≤ 1 ∧ 1 ≤ 1 then (1 - (1/3)) / (1 - 0) else 0

theorem probability_of_event :
  interval_probability = 2 / 3 :=
by
  rw [interval_probability]
  sorry

end probability_of_event_l166_166040


namespace frog_jump_distance_l166_166894

theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_jump : ℕ) (frog_jump : ℕ) :
  grasshopper_jump = 9 → extra_jump = 3 → frog_jump = grasshopper_jump + extra_jump → frog_jump = 12 :=
by
  intros h_grasshopper h_extra h_frog
  rw [h_grasshopper, h_extra] at h_frog
  exact h_frog

end frog_jump_distance_l166_166894


namespace line_segment_AB_length_l166_166588

noncomputable def length_AB (xA yA xB yB : ℝ) : ℝ :=
  Real.sqrt ((xA - xB)^2 + (yA - yB)^2)

theorem line_segment_AB_length :
  ∀ (xA yA xB yB : ℝ),
    (xA - yA = 0) →
    (xB + yB = 0) →
    (∃ k : ℝ, yA = k * (xA + 1) ∧ yB = k * (xB + 1)) →
    (-1 ≤ xA ∧ xA ≤ 0) →
    (xA + xB = 2 * k ∧ yA + yB = 2 * k) →
    length_AB xA yA xB yB = (4/3) * Real.sqrt 5 :=
by
  intros xA yA xB yB h1 h2 h3 h4 h5
  sorry

end line_segment_AB_length_l166_166588


namespace tan_C_in_triangle_l166_166298

theorem tan_C_in_triangle (A B C : ℝ) (hA : Real.tan A = 1 / 2) (hB : Real.cos B = 3 * Real.sqrt 10 / 10) :
  Real.tan C = -1 :=
sorry

end tan_C_in_triangle_l166_166298


namespace daily_serving_size_l166_166817

-- Definitions based on problem conditions
def days : ℕ := 180
def capsules_per_bottle : ℕ := 60
def bottles : ℕ := 6
def total_capsules : ℕ := bottles * capsules_per_bottle

-- Theorem statement to prove the daily serving size
theorem daily_serving_size :
  total_capsules / days = 2 := by
  sorry

end daily_serving_size_l166_166817


namespace right_triangle_medians_l166_166712

theorem right_triangle_medians
    (a b c d m : ℝ)
    (h1 : ∀(a b c d : ℝ), 2 * (c/d) = 3)
    (h2 : m = 4 * 3 ∨ m = (3/4)) :
    ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ (m₁ = 12 ∨ m₁ = 3/4) ∧ (m₂ = 12 ∨ m₂ = 3/4) :=
by 
  sorry

end right_triangle_medians_l166_166712


namespace workshop_personnel_l166_166142

-- Definitions for workshops with their corresponding production constraints
def workshopA_production (x : ℕ) : ℕ := 6 + 11 * (x - 1)
def workshopB_production (y : ℕ) : ℕ := 7 + 10 * (y - 1)

-- The main theorem to be proved
theorem workshop_personnel :
  ∃ (x y : ℕ), workshopA_production x = workshopB_production y ∧
               100 ≤ workshopA_production x ∧ workshopA_production x ≤ 200 ∧
               x = 12 ∧ y = 13 :=
by
  sorry

end workshop_personnel_l166_166142


namespace cos_a3_value_l166_166643

theorem cos_a3_value (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 1 + a 3 + a 5 = Real.pi) : 
  Real.cos (a 3) = 1/2 := 
by 
  sorry

end cos_a3_value_l166_166643


namespace intersection_of_sets_l166_166681

noncomputable def A : Set ℝ := { x | x^2 - 1 > 0 }
noncomputable def B : Set ℝ := { x | Real.log x / Real.log 2 > 0 }

theorem intersection_of_sets :
  A ∩ B = { x | x > 1 } :=
by {
  sorry
}

end intersection_of_sets_l166_166681


namespace find_speed_B_l166_166577

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l166_166577


namespace point_in_third_quadrant_l166_166978

theorem point_in_third_quadrant (x y : ℤ) (hx : x = -8) (hy : y = -3) : (x < 0) ∧ (y < 0) :=
by
  have hx_neg : x < 0 := by rw [hx]; norm_num
  have hy_neg : y < 0 := by rw [hy]; norm_num
  exact ⟨hx_neg, hy_neg⟩

end point_in_third_quadrant_l166_166978


namespace difference_of_two_numbers_l166_166140

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l166_166140


namespace estimate_pi_simulation_l166_166602

theorem estimate_pi_simulation :
  let side := 2
  let radius := 1
  let total_seeds := 1000
  let seeds_in_circle := 778
  (π : ℝ) * radius^2 / side^2 = (seeds_in_circle : ℝ) / total_seeds → π = 3.112 :=
by
  intros
  sorry

end estimate_pi_simulation_l166_166602


namespace current_time_is_208_l166_166056

def minute_hand_position (t : ℝ) : ℝ := 6 * t
def hour_hand_position (t : ℝ) : ℝ := 0.5 * t

theorem current_time_is_208 (t : ℝ) (h1 : 0 < t) (h2 : t < 60) 
  (h3 : minute_hand_position (t + 8) + 60 = hour_hand_position (t + 5)) : 
  t = 8 :=
by sorry

end current_time_is_208_l166_166056


namespace perfect_square_condition_l166_166373

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l166_166373


namespace value_of_y_l166_166193

theorem value_of_y (x y : ℤ) (h1 : 1.5 * (x : ℝ) = 0.25 * (y : ℝ)) (h2 : x = 24) : y = 144 :=
  sorry

end value_of_y_l166_166193


namespace marketing_survey_l166_166198

theorem marketing_survey
  (H_neither : Nat := 80)
  (H_only_A : Nat := 60)
  (H_ratio_Both_to_Only_B : Nat := 3)
  (H_both : Nat := 25) :
  H_neither + H_only_A + (H_ratio_Both_to_Only_B * H_both) + H_both = 240 := 
sorry

end marketing_survey_l166_166198


namespace measure_of_angle_A_l166_166292

theorem measure_of_angle_A (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := 
by 
  sorry

end measure_of_angle_A_l166_166292


namespace parametric_to_cartesian_l166_166013

theorem parametric_to_cartesian (θ : ℝ) (x y : ℝ) :
  (x = 1 + 2 * Real.cos θ) →
  (y = 2 * Real.sin θ) →
  (x - 1) ^ 2 + y ^ 2 = 4 :=
by 
  sorry

end parametric_to_cartesian_l166_166013


namespace simplify_expression_l166_166720

theorem simplify_expression (a b c x : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  ( ( (x + a)^4 ) / ( (a - b) * (a - c) ) 
  + ( (x + b)^4 ) / ( (b - a) * (b - c) ) 
  + ( (x + c)^4 ) / ( (c - a) * (c - b) ) ) = a + b + c + 4 * x := 
by
  sorry

end simplify_expression_l166_166720


namespace pies_left_l166_166786

theorem pies_left (pies_per_batch : ℕ) (batches : ℕ) (dropped : ℕ) (total_pies : ℕ) (pies_left : ℕ)
  (h1 : pies_per_batch = 5)
  (h2 : batches = 7)
  (h3 : dropped = 8)
  (h4 : total_pies = pies_per_batch * batches)
  (h5 : pies_left = total_pies - dropped) :
  pies_left = 27 := by
  sorry

end pies_left_l166_166786


namespace diameter_increase_l166_166533

theorem diameter_increase (D D' : ℝ) (h : π * (D' / 2) ^ 2 = 2.4336 * π * (D / 2) ^ 2) : D' / D = 1.56 :=
by
  -- Statement only, proof is omitted
  sorry

end diameter_increase_l166_166533


namespace roots_of_polynomial_in_range_l166_166223

theorem roots_of_polynomial_in_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 * x2 = m^2 - 2 ∧ (x1 + x2) = -(m - 1)) 
  -> 0 < m ∧ m < 1 :=
by
  sorry

end roots_of_polynomial_in_range_l166_166223


namespace find_largest_number_l166_166062

-- Define what it means for a sequence of 4 numbers to be an arithmetic progression with a given common difference d
def is_arithmetic_progression (a b c d : ℝ) (diff : ℝ) : Prop := (b - a = diff) ∧ (c - b = diff) ∧ (d - c = diff)

-- Define what it means for a sequence of 4 numbers to be a geometric progression
def is_geometric_progression (a b c d : ℝ) : Prop := b / a = c / b ∧ c / b = d / c

-- Given conditions for the sequence of 8 increasing real numbers
def conditions (a : ℕ → ℝ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧
  ∃ i j k, is_arithmetic_progression (a i) (a (i+1)) (a (i+2)) (a (i+3)) 4 ∧
            is_arithmetic_progression (a j) (a (j+1)) (a (j+2)) (a (j+3)) 36 ∧
            is_geometric_progression (a k) (a (k+1)) (a (k+2)) (a (k+3))

-- Prove that under these conditions, the largest number in the sequence is 126
theorem find_largest_number (a : ℕ → ℝ) : conditions a → a 7 = 126 :=
by
  sorry

end find_largest_number_l166_166062


namespace problem1_problem2_l166_166609

-- Define the propositions
def S (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

def p (m : ℝ) : Prop := 0 < m ∧ m < 2

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ 1 ≤ m := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hpq : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l166_166609


namespace cost_prices_max_profit_find_m_l166_166638

-- Part 1
theorem cost_prices (x y: ℕ) (h1 : 40 * x + 30 * y = 5000) (h2 : 10 * x + 50 * y = 3800) : 
  x = 80 ∧ y = 60 :=
sorry

-- Part 2
theorem max_profit (a: ℕ) (h1 : 70 ≤ a ∧ a ≤ 75) : 
  (20 * a + 6000) ≤ 7500 :=
sorry

-- Part 3
theorem find_m (m : ℝ) (h1 : 4 < m ∧ m < 8) (h2 : (20 - 5 * m) * 70 + 6000 = 5720) : 
  m = 4.8 :=
sorry

end cost_prices_max_profit_find_m_l166_166638


namespace group_members_count_l166_166767

theorem group_members_count (n: ℕ) (total_paise: ℕ) (condition1: total_paise = 3249) :
  (n * n = total_paise) → n = 57 :=
by
  sorry

end group_members_count_l166_166767


namespace remainder_sum_mod_l166_166409

theorem remainder_sum_mod (a b c d e : ℕ)
  (h₁ : a = 17145)
  (h₂ : b = 17146)
  (h₃ : c = 17147)
  (h₄ : d = 17148)
  (h₅ : e = 17149)
  : (a + b + c + d + e) % 10 = 5 := by
  sorry

end remainder_sum_mod_l166_166409


namespace polynomial_degree_le_one_l166_166462

theorem polynomial_degree_le_one {P : ℝ → ℝ} (h : ∀ x : ℝ, 2 * P x = P (x + 3) + P (x - 3)) :
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x + b :=
sorry

end polynomial_degree_le_one_l166_166462


namespace correct_amendment_statements_l166_166442

/-- The amendment includes the abuse of administrative power by administrative organs 
    to exclude or limit competition. -/
def abuse_of_power_in_amendment : Prop :=
  true

/-- The amendment includes illegal fundraising. -/
def illegal_fundraising_in_amendment : Prop :=
  true

/-- The amendment includes apportionment of expenses. -/
def apportionment_of_expenses_in_amendment : Prop :=
  true

/-- The amendment includes failure to pay minimum living allowances or social insurance benefits according to law. -/
def failure_to_pay_benefits_in_amendment : Prop :=
  true

/-- The amendment further standardizes the exercise of government power. -/
def standardizes_govt_power : Prop :=
  true

/-- The amendment better protects the legitimate rights and interests of citizens. -/
def protects_rights : Prop :=
  true

/-- The amendment expands the channels for citizens' democratic participation. -/
def expands_democratic_participation : Prop :=
  false

/-- The amendment expands the scope of government functions. -/
def expands_govt_functions : Prop :=
  false

/-- The correct answer to which set of statements is true about the amendment is {②, ③}.
    This is encoded as proving (standardizes_govt_power ∧ protects_rights) = true. -/
theorem correct_amendment_statements : (standardizes_govt_power ∧ protects_rights) ∧ 
                                      ¬(expands_democratic_participation ∧ expands_govt_functions) :=
by {
  sorry
}

end correct_amendment_statements_l166_166442


namespace average_disk_space_per_minute_l166_166221

theorem average_disk_space_per_minute 
  (days : ℕ := 15) 
  (disk_space : ℕ := 36000) 
  (minutes_per_day : ℕ := 1440) 
  (total_minutes := days * minutes_per_day) 
  (average_space_per_minute := disk_space / total_minutes) :
  average_space_per_minute = 2 :=
sorry

end average_disk_space_per_minute_l166_166221


namespace find_total_money_l166_166770

theorem find_total_money
  (d x T : ℝ)
  (h1 : d = 5 / 17)
  (h2 : x = 35)
  (h3 : d * T = x) :
  T = 119 :=
by sorry

end find_total_money_l166_166770


namespace labourer_monthly_income_l166_166948

-- Define the conditions
def total_expense_first_6_months : ℕ := 90 * 6
def total_expense_next_4_months : ℕ := 60 * 4
def debt_cleared_and_savings : ℕ := 30

-- Define the monthly income
def monthly_income : ℕ := 81

-- The statement to be proven
theorem labourer_monthly_income (I D : ℕ) (h1 : 6 * I + D = total_expense_first_6_months) 
                               (h2 : 4 * I - D = total_expense_next_4_months + debt_cleared_and_savings) :
  I = monthly_income :=
by {
  sorry
}

end labourer_monthly_income_l166_166948


namespace t_mobile_first_two_lines_cost_l166_166058

theorem t_mobile_first_two_lines_cost :
  ∃ T : ℝ,
  (T + 16 * 3) = (45 + 14 * 3 + 11) → T = 50 :=
by
  sorry

end t_mobile_first_two_lines_cost_l166_166058


namespace find_x_l166_166734

theorem find_x : ∃ x : ℕ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end find_x_l166_166734


namespace carla_water_requirement_l166_166987

theorem carla_water_requirement (h: ℕ) (p: ℕ) (c: ℕ) (gallons_per_pig: ℕ) (horse_factor: ℕ) 
  (num_pigs: ℕ) (num_horses: ℕ) (tank_water: ℕ): 
  num_pigs = 8 ∧ num_horses = 10 ∧ gallons_per_pig = 3 ∧ horse_factor = 2 ∧ tank_water = 30 →
  h = horse_factor * gallons_per_pig ∧ p = num_pigs * gallons_per_pig ∧ c = tank_water →
  h * num_horses + p + c = 114 :=
by
  intro h1 h2
  cases h1
  cases h2
  sorry

end carla_water_requirement_l166_166987


namespace eggs_left_in_jar_l166_166897

def eggs_after_removal (original removed : Nat) : Nat :=
  original - removed

theorem eggs_left_in_jar : eggs_after_removal 27 7 = 20 :=
by
  sorry

end eggs_left_in_jar_l166_166897


namespace petya_max_votes_difference_l166_166303

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l166_166303


namespace dima_can_find_heavy_ball_l166_166253

noncomputable def find_heavy_ball
  (balls : Fin 9) -- 9 balls, indexed from 0 to 8 representing the balls 1 to 9
  (heavy : Fin 9) -- One of the balls is heavier
  (weigh : Fin 9 → Fin 9 → Ordering) -- A function that compares two groups of balls and gives an Ordering: .lt, .eq, or .gt
  (predetermined_sets : List (Fin 9 × Fin 9)) -- A list of tuples representing balls on each side for the two weighings
  (valid_sets : predetermined_sets.length ≤ 2) : Prop := -- Not more than two weighings
  ∃ idx : Fin 9, idx = heavy -- Need to prove that we can always find the heavier ball

theorem dima_can_find_heavy_ball :
  ∀ (balls : Fin 9) (heavy : Fin 9)
    (weigh : Fin 9 → Fin 9 → Ordering)
    (predetermined_sets : List (Fin 9 × Fin 9))
    (valid_sets : predetermined_sets.length ≤ 2),
  find_heavy_ball balls heavy weigh predetermined_sets valid_sets :=
sorry -- Proof is omitted

end dima_can_find_heavy_ball_l166_166253


namespace nonneg_integer_solution_l166_166879

theorem nonneg_integer_solution (a b c : ℕ) (h : 5^a * 7^b + 4 = 3^c) : (a, b, c) = (1, 0, 2) := 
sorry

end nonneg_integer_solution_l166_166879


namespace unit_digit_calc_l166_166678

theorem unit_digit_calc : (8 * 19 * 1981 - 8^3) % 10 = 0 := by
  sorry

end unit_digit_calc_l166_166678


namespace translate_down_three_units_l166_166635

def original_function (x : ℝ) : ℝ := 3 * x + 2

def translated_function (x : ℝ) : ℝ := 3 * x - 1

theorem translate_down_three_units :
  ∀ x : ℝ, translated_function x = original_function x - 3 :=
by
  intro x
  simp [original_function, translated_function]
  sorry

end translate_down_three_units_l166_166635


namespace arccos_half_eq_pi_div_three_l166_166404

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l166_166404


namespace totalPearsPicked_l166_166152

-- Define the number of pears picked by each individual
def jasonPears : ℕ := 46
def keithPears : ℕ := 47
def mikePears : ℕ := 12

-- State the theorem to prove the total number of pears picked
theorem totalPearsPicked : jasonPears + keithPears + mikePears = 105 := 
by
  -- The proof is omitted
  sorry

end totalPearsPicked_l166_166152


namespace parabola_x_coordinate_l166_166715

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p, 0)

theorem parabola_x_coordinate
  (M : ℝ × ℝ)
  (h_parabola : (M.2)^2 = 4 * M.1)
  (h_distance : dist M (parabola_focus 2) = 3) :
  M.1 = 1 :=
by
  sorry

end parabola_x_coordinate_l166_166715


namespace box_dimensions_l166_166444

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  -- We assume the proof is correct based on given conditions
  sorry

end box_dimensions_l166_166444


namespace find_denominator_of_second_fraction_l166_166848

theorem find_denominator_of_second_fraction (y : ℝ) (h : y > 0) (x : ℝ) :
  (2 * y) / 5 + (3 * y) / x = 0.7 * y → x = 10 :=
by
  sorry

end find_denominator_of_second_fraction_l166_166848


namespace division_multiplication_result_l166_166699

theorem division_multiplication_result :
  (7.5 / 6) * 12 = 15 := by
  sorry

end division_multiplication_result_l166_166699


namespace factorize_polynomial_l166_166836

theorem factorize_polynomial (x y : ℝ) : 
  (x^2 - y^2 - 2 * x - 4 * y - 3) = (x + y + 1) * (x - y - 3) :=
  sorry

end factorize_polynomial_l166_166836


namespace magic_square_proof_l166_166654

theorem magic_square_proof
    (a b c d e S : ℕ)
    (h1 : 35 + e + 27 = S)
    (h2 : 30 + c + d = S)
    (h3 : a + 32 + b = S)
    (h4 : 35 + c + b = S)
    (h5 : a + c + 27 = S)
    (h6 : 35 + c + b = S)
    (h7 : 35 + c + 27 = S)
    (h8 : a + c + d = S) :
  d + e = 35 :=
  sorry

end magic_square_proof_l166_166654


namespace cost_price_equivalence_l166_166096

theorem cost_price_equivalence (list_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) :
  list_price = 132 → discount_rate = 0.1 → profit_rate = 0.1 → 
  (list_price * (1 - discount_rate)) = cost_price * (1 + profit_rate) →
  cost_price = 108 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_price_equivalence_l166_166096


namespace coprime_divisors_property_l166_166031

theorem coprime_divisors_property (n : ℕ) 
  (h : ∀ a b : ℕ, a ∣ n → b ∣ n → gcd a b = 1 → (a + b - 1) ∣ n) : 
  (∃ k : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n = p ^ k) ∨ (n = 12) :=
sorry

end coprime_divisors_property_l166_166031


namespace find_common_remainder_l166_166423

theorem find_common_remainder :
  ∃ (d : ℕ), 100 ≤ d ∧ d ≤ 999 ∧ (312837 % d = 96) ∧ (310650 % d = 96) :=
sorry

end find_common_remainder_l166_166423


namespace total_reams_of_paper_l166_166548

def reams_for_haley : ℕ := 2
def reams_for_sister : ℕ := 3

theorem total_reams_of_paper : reams_for_haley + reams_for_sister = 5 := by
  sorry

end total_reams_of_paper_l166_166548


namespace fit_small_boxes_l166_166410

def larger_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def small_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

theorem fit_small_boxes (L W H l w h : ℕ)
  (larger_box_dim : L = 12 ∧ W = 14 ∧ H = 16)
  (small_box_dim : l = 3 ∧ w = 7 ∧ h = 2)
  (min_boxes : larger_box_volume L W H / small_box_volume l w h = 64) :
  ∃ n, n ≥ 64 :=
by
  sorry

end fit_small_boxes_l166_166410


namespace eval_abs_a_plus_b_l166_166243

theorem eval_abs_a_plus_b (a b : ℤ) (x : ℤ) 
(h : (7 * x - a) ^ 2 = 49 * x ^ 2 - b * x + 9) : |a + b| = 45 :=
sorry

end eval_abs_a_plus_b_l166_166243


namespace find_m_plus_M_l166_166349

-- Given conditions
def cond1 (x y z : ℝ) := x + y + z = 4
def cond2 (x y z : ℝ) := x^2 + y^2 + z^2 = 6

-- Proof statement: The sum of the smallest and largest possible values of x is 8/3
theorem find_m_plus_M :
  ∀ (x y z : ℝ), cond1 x y z → cond2 x y z → (min (x : ℝ) (max x y) + max (x : ℝ) (min x y) = 8 / 3) :=
by
  sorry

end find_m_plus_M_l166_166349


namespace harkamal_mangoes_l166_166808

theorem harkamal_mangoes (m : ℕ) (h1: 8 * 70 = 560) (h2 : m * 50 + 560 = 1010) : m = 9 :=
by
  sorry

end harkamal_mangoes_l166_166808


namespace solve_for_x_l166_166153

theorem solve_for_x (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2 * y = 10) : x = 26 / 3 := by
  sorry

end solve_for_x_l166_166153


namespace perfect_square_trinomial_m_l166_166647

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2*(m-3)*x + 16) = (x + a)^2) ↔ (m = 7 ∨ m = -1) := 
sorry

end perfect_square_trinomial_m_l166_166647


namespace smallest_n_for_identity_matrix_l166_166274

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l166_166274


namespace simplify_expression_to_inverse_abc_l166_166122

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem simplify_expression_to_inverse_abc :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ca + 3)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹ + 3) = (1 : ℝ) / (abc) :=
by
  sorry

end simplify_expression_to_inverse_abc_l166_166122


namespace induction_proof_l166_166061

def f (n : ℕ) : ℕ := (List.range (2 * n - 1)).sum + n

theorem induction_proof (n : ℕ) (h : n > 0) : f (n + 1) - f n = 8 * n := by
  sorry

end induction_proof_l166_166061


namespace person_died_at_33_l166_166006

-- Define the conditions and constants
def start_age : ℕ := 25
def insurance_payment : ℕ := 10000
def premium : ℕ := 450
def loss : ℕ := 1000
def annual_interest_rate : ℝ := 0.05
def half_year_factor : ℝ := 1.025 -- half-yearly compounded interest factor

-- Calculate the number of premium periods (as an integer)
def n := 16 -- (derived from the calculations in the given solution)

-- Define the final age based on the number of premium periods
def final_age : ℕ := start_age + (n / 2)

-- The proof statement
theorem person_died_at_33 : final_age = 33 := by
  sorry

end person_died_at_33_l166_166006


namespace horner_method_evaluation_l166_166233

def f (x : ℝ) := 0.5 * x^5 + 4 * x^4 + 0 * x^3 - 3 * x^2 + x - 1

theorem horner_method_evaluation : f 3 = 1 :=
by
  -- Placeholder for the proof
  sorry

end horner_method_evaluation_l166_166233


namespace chord_eq_l166_166330

/-- 
If a chord of the ellipse x^2 / 36 + y^2 / 9 = 1 is bisected by the point (4,2),
then the equation of the line on which this chord lies is x + 2y - 8 = 0.
-/
theorem chord_eq {x y : ℝ} (H : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 / 36 + A.2 ^ 2 / 9 = 1) ∧ 
  (B.1 ^ 2 / 36 + B.2 ^ 2 / 9 = 1) ∧ 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4, 2)) :
  x + 2 * y = 8 :=
sorry

end chord_eq_l166_166330


namespace solve_equation_l166_166908

noncomputable def f (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs x - 8) - 4) - 2) - 1)

noncomputable def g (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs x - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1)

theorem solve_equation : ∀ (x : ℝ), f x = g x :=
by
  sorry -- The proof will be inserted here

end solve_equation_l166_166908


namespace number_of_square_tiles_l166_166913

-- A box contains a mix of triangular and square tiles.
-- There are 30 tiles in total with 100 edges altogether.
variable (x y : ℕ) -- where x is the number of triangular tiles and y is the number of square tiles, both must be natural numbers
-- Each triangular tile has 3 edges, and each square tile has 4 edges.

-- Define the conditions
def tile_condition_1 : Prop := x + y = 30
def tile_condition_2 : Prop := 3 * x + 4 * y = 100

-- The goal is to prove the number of square tiles y is 10.
theorem number_of_square_tiles : tile_condition_1 x y → tile_condition_2 x y → y = 10 :=
  by
    intros h1 h2
    sorry

end number_of_square_tiles_l166_166913


namespace day_crew_fraction_loaded_l166_166875

-- Let D be the number of boxes loaded by each worker on the day crew
-- Let W_d be the number of workers on the day crew
-- Let W_n be the number of workers on the night crew
-- Let B_d be the total number of boxes loaded by the day crew
-- Let B_n be the total number of boxes loaded by the night crew

variable (D W_d : ℕ) 
variable (B_d := D * W_d)
variable (W_n := (4 / 9 : ℚ) * W_d)
variable (B_n := (3 / 4 : ℚ) * D * W_n)
variable (total_boxes := B_d + B_n)

theorem day_crew_fraction_loaded : 
  (D * W_d) / (D * W_d + (3 / 4 : ℚ) * D * ((4 / 9 : ℚ) * W_d)) = (3 / 4 : ℚ) := sorry

end day_crew_fraction_loaded_l166_166875


namespace a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l166_166921

theorem a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b
  (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a ^ 2 < 0 → a < b) ∧ 
  (¬∀ a b : ℝ, a < b → (a - b) * a ^ 2 < 0) :=
sorry

end a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l166_166921


namespace solve_for_a_l166_166950

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end solve_for_a_l166_166950


namespace only_n_divides_2_to_n_minus_1_l166_166236

theorem only_n_divides_2_to_n_minus_1 (n : ℕ) (h1 : n > 0) : n ∣ (2^n - 1) ↔ n = 1 :=
by
  sorry

end only_n_divides_2_to_n_minus_1_l166_166236


namespace length_of_AP_l166_166810

noncomputable def square_side_length : ℝ := 8
noncomputable def rect_width : ℝ := 12
noncomputable def rect_height : ℝ := 8

axiom AD_perpendicular_WX : true
axiom shaded_area_half_WXYZ : true

theorem length_of_AP (AP : ℝ) (shaded_area : ℝ)
  (h1 : shaded_area = (rect_width * rect_height) / 2)
  (h2 : shaded_area = (square_side_length - AP) * square_side_length)
  : AP = 2 := by
  sorry

end length_of_AP_l166_166810


namespace total_pencils_bought_l166_166667

theorem total_pencils_bought (x y : ℕ) (y_pos : 0 < y) (initial_cost : y * (x + 10) = 5 * x) (later_cost : (4 * y) * (x + 10) = 20 * x) :
    x = 15 → (40 = x + x + 10) ∨ x = 40 → (90 = x + (x + 10)) :=
by
  sorry

end total_pencils_bought_l166_166667


namespace largest_of_three_l166_166798

theorem largest_of_three (a b c : ℝ) (h₁ : a = 43.23) (h₂ : b = 2/5) (h₃ : c = 21.23) :
  max (max a b) c = a :=
by
  sorry

end largest_of_three_l166_166798


namespace simplify_sqrt_450_l166_166284

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l166_166284


namespace sum_of_arithmetic_sequence_zero_l166_166117

noncomputable def arithmetic_sequence_sum (S : ℕ → ℤ) : Prop :=
S 20 = S 40

theorem sum_of_arithmetic_sequence_zero {S : ℕ → ℤ} (h : arithmetic_sequence_sum S) : 
  S 60 = 0 :=
sorry

end sum_of_arithmetic_sequence_zero_l166_166117


namespace cost_for_3300_pens_l166_166976

noncomputable def cost_per_pack (pack_cost : ℝ) (num_pens_per_pack : ℕ) : ℝ :=
  pack_cost / num_pens_per_pack

noncomputable def total_cost (cost_per_pen : ℝ) (num_pens : ℕ) : ℝ :=
  cost_per_pen * num_pens

theorem cost_for_3300_pens (pack_cost : ℝ) (num_pens_per_pack num_pens : ℕ) (h_pack_cost : pack_cost = 45) (h_num_pens_per_pack : num_pens_per_pack = 150) (h_num_pens : num_pens = 3300) :
  total_cost (cost_per_pack pack_cost num_pens_per_pack) num_pens = 990 :=
  by
    sorry

end cost_for_3300_pens_l166_166976


namespace simplify_expression_l166_166463

theorem simplify_expression :
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 :=
by
  sorry

end simplify_expression_l166_166463


namespace sum_of_a_and_b_l166_166794

noncomputable def f (x : Real) : Real := (1 + Real.sin (2 * x)) / 2
noncomputable def a : Real := f (Real.log 5)
noncomputable def b : Real := f (Real.log (1 / 5))

theorem sum_of_a_and_b : a + b = 1 := by
  -- proof to be provided
  sorry

end sum_of_a_and_b_l166_166794


namespace rational_b_if_rational_a_l166_166073

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end rational_b_if_rational_a_l166_166073


namespace Teresa_age_at_Michiko_birth_l166_166437

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end Teresa_age_at_Michiko_birth_l166_166437


namespace ellipse_intersects_x_axis_at_four_l166_166758

theorem ellipse_intersects_x_axis_at_four
    (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 0))
    (h2 : f2 = (4, 0))
    (h3 : ∃ P : ℝ × ℝ, P = (1, 0) ∧ (dist P f1 + dist P f2 = 4)) :
  ∃ Q : ℝ × ℝ, Q = (4, 0) ∧ (dist Q f1 + dist Q f2 = 4) :=
sorry

end ellipse_intersects_x_axis_at_four_l166_166758


namespace man_salary_l166_166998

variable (S : ℝ)

theorem man_salary (S : ℝ) (h1 : S - (1/3) * S - (1/4) * S - (1/5) * S = 1760) : S = 8123 := 
by 
  sorry

end man_salary_l166_166998


namespace no_solution_for_squares_l166_166457

theorem no_solution_for_squares (x y : ℤ) (hx : x > 0) (hy : y > 0) :
  ¬ ∃ k m : ℤ, x^2 + y + 2 = k^2 ∧ y^2 + 4 * x = m^2 :=
sorry

end no_solution_for_squares_l166_166457


namespace min_positive_period_f_max_value_f_decreasing_intervals_g_l166_166789

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem min_positive_period_f : 
  ∃ (p : ℝ), p > 0 ∧ (∀ x : ℝ, f (x + 2*Real.pi) = f x) :=
sorry

theorem max_value_f : 
  ∃ (M : ℝ), (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = 2 :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (-x)

theorem decreasing_intervals_g :
  ∀ (k : ℤ), ∀ x : ℝ, (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
  ∀ (h : x ≤ Real.pi * 2 * (↑k+1)), g x ≥ g (x + Real.pi) :=
sorry

end min_positive_period_f_max_value_f_decreasing_intervals_g_l166_166789


namespace A_subset_B_l166_166859

def inA (n : ℕ) : Prop := ∃ x y : ℕ, n = x^2 + 2 * y^2 ∧ x > y
def inB (n : ℕ) : Prop := ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ n = (a^3 + b^3 + c^3) / (a + b + c)

theorem A_subset_B : ∀ (n : ℕ), inA n → inB n := 
sorry

end A_subset_B_l166_166859


namespace greatest_integer_x_l166_166095

theorem greatest_integer_x (x : ℤ) : 
  (∀ x : ℤ, (8 / 11 : ℝ) > (x / 17) → x ≤ 12) ∧ (8 / 11 : ℝ) > (12 / 17) :=
sorry

end greatest_integer_x_l166_166095


namespace product_of_roots_l166_166882

theorem product_of_roots (a b c d : ℝ)
  (h1 : a = 16 ^ (1 / 5))
  (h2 : 16 = 2 ^ 4)
  (h3 : b = 64 ^ (1 / 6))
  (h4 : 64 = 2 ^ 6):
  a * b = 2 * (16 ^ (1 / 5)) := by
  sorry

end product_of_roots_l166_166882


namespace num_terms_arith_seq_l166_166854

theorem num_terms_arith_seq {a d t : ℕ} (h_a : a = 5) (h_d : d = 3) (h_t : t = 140) :
  ∃ n : ℕ, t = a + (n-1) * d ∧ n = 46 :=
by
  sorry

end num_terms_arith_seq_l166_166854


namespace remainder_17_pow_63_mod_7_l166_166210

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l166_166210


namespace molecular_weight_CaOH2_l166_166576

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_CaOH2 :
  (atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H = 74.10) := 
by 
  sorry

end molecular_weight_CaOH2_l166_166576


namespace no_valid_digit_replacement_l166_166742

theorem no_valid_digit_replacement :
  ¬ ∃ (A B C D E M X : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ M ∧ A ≠ X ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ M ∧ B ≠ X ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ M ∧ C ≠ X ∧
     D ≠ E ∧ D ≠ M ∧ D ≠ X ∧
     E ≠ M ∧ E ≠ X ∧
     M ≠ X ∧
     0 ≤ A ∧ A < 10 ∧
     0 ≤ B ∧ B < 10 ∧
     0 ≤ C ∧ C < 10 ∧
     0 ≤ D ∧ D < 10 ∧
     0 ≤ E ∧ E < 10 ∧
     0 ≤ M ∧ M < 10 ∧
     0 ≤ X ∧ X < 10 ∧
     A * B * C * D + 1 = C * E * M * X) :=
sorry

end no_valid_digit_replacement_l166_166742


namespace find_X_l166_166245

theorem find_X (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 :=
sorry

end find_X_l166_166245


namespace sin_double_angle_value_l166_166296

open Real

theorem sin_double_angle_value (x : ℝ) 
  (h1 : sin (x + π/3) * cos (x - π/6) + sin (x - π/6) * cos (x + π/3) = 5 / 13)
  (h2 : -π/3 ≤ x ∧ x ≤ π/6) :
  sin (2 * x) = (5 * sqrt 3 - 12) / 26 :=
by
  sorry

end sin_double_angle_value_l166_166296


namespace men_in_first_scenario_l166_166539

theorem men_in_first_scenario 
  (M : ℕ) 
  (daily_hours_first weekly_earning_first daily_hours_second weekly_earning_second : ℝ) 
  (number_of_men_second : ℕ)
  (days_per_week : ℕ := 7) 
  (h1 : M * daily_hours_first * days_per_week = weekly_earning_first)
  (h2 : number_of_men_second * daily_hours_second * days_per_week = weekly_earning_second) 
  (h1_value : daily_hours_first = 10) 
  (w1_value : weekly_earning_first = 1400) 
  (h2_value : daily_hours_second = 6) 
  (w2_value : weekly_earning_second = 1890)
  (second_scenario_men : number_of_men_second = 9) : 
  M = 4 :=
by
  sorry

end men_in_first_scenario_l166_166539


namespace bottle_caps_per_friend_l166_166335

-- The context where Catherine has 18 bottle caps
def bottle_caps : Nat := 18

-- Catherine distributes these bottle caps among 6 friends
def number_of_friends : Nat := 6

-- We need to prove that each friend gets 3 bottle caps
theorem bottle_caps_per_friend : bottle_caps / number_of_friends = 3 :=
by sorry

end bottle_caps_per_friend_l166_166335


namespace john_total_cost_l166_166741

-- The total cost John incurs to rent a car, buy gas, and drive 320 miles
def total_cost (rental_cost gas_cost_per_gallon cost_per_mile miles driven_gallons : ℝ): ℝ :=
  rental_cost + (gas_cost_per_gallon * driven_gallons) + (cost_per_mile * miles)

theorem john_total_cost :
  let rental_cost := 150
  let gallons := 8
  let gas_cost_per_gallon := 3.50
  let cost_per_mile := 0.50
  let miles := 320
  total_cost rental_cost gas_cost_per_gallon cost_per_mile miles gallons = 338 := 
by
  -- The detailed proof is skipped here
  sorry

end john_total_cost_l166_166741


namespace cubic_roots_arithmetic_progression_l166_166099

theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x : ℝ, x^3 + a * x^2 + b * x + c = 0) ∧ 
  (∀ x : ℝ, x^3 + a * x^2 + b * x + c = 0 → 
    (x = p - t ∨ x = p ∨ x = p + t) ∧ 
    (a ≠ 0)) ↔ 
  ((a * b / 3) - 2 * (a^3) / 27 - c = 0 ∧ (a^3 / 3) - b ≥ 0) := 
by sorry

end cubic_roots_arithmetic_progression_l166_166099


namespace dolphins_scored_15_l166_166796

theorem dolphins_scored_15 (s d : ℤ) 
  (h1 : s + d = 48) 
  (h2 : s - d = 18) : 
  d = 15 := 
sorry

end dolphins_scored_15_l166_166796


namespace book_pages_l166_166059

theorem book_pages (P : ℝ) (h1 : 2/3 * P = 1/3 * P + 20) : P = 60 :=
by
  sorry

end book_pages_l166_166059


namespace smallest_enclosing_sphere_radius_l166_166880

theorem smallest_enclosing_sphere_radius :
  let r := 2
  let d := 4 * Real.sqrt 3
  let total_diameter := d + 2*r
  let radius_enclosing_sphere := total_diameter / 2
  radius_enclosing_sphere = 2 + 2 * Real.sqrt 3 := by
  -- Define the radius of the smaller spheres
  let r : ℝ := 2
  -- Space diagonal of the cube which is 4√3 where 4 is the side length
  let d : ℝ := 4 * Real.sqrt 3
  -- Total diameter of the sphere containing the cube (space diagonal + 2 radius of one sphere)
  let total_diameter : ℝ := d + 2 * r
  -- Radius of the enclosing sphere
  let radius_enclosing_sphere : ℝ := total_diameter / 2
  -- We need to prove that this radius equals 2 + 2√3
  sorry

end smallest_enclosing_sphere_radius_l166_166880


namespace f_odd_f_inequality_solution_l166_166076

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd: 
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = - f x := 
by
  sorry

theorem f_inequality_solution:
  { x : ℝ // -1 < x ∧ x < 1 ∧ f x < -1 } = { x : ℝ // -1 < x ∧ x < -1/3 } := 
by 
  sorry

end f_odd_f_inequality_solution_l166_166076


namespace add_100ml_water_l166_166833

theorem add_100ml_water 
    (current_volume : ℕ) 
    (current_water_percentage : ℝ) 
    (desired_water_percentage : ℝ) 
    (current_water_volume : ℝ) 
    (x : ℝ) :
    current_volume = 300 →
    current_water_percentage = 0.60 →
    desired_water_percentage = 0.70 →
    current_water_volume = 0.60 * 300 →
    180 + x = 0.70 * (300 + x) →
    x = 100 := 
sorry

end add_100ml_water_l166_166833


namespace find_weekly_allowance_l166_166350

noncomputable def weekly_allowance (A : ℝ) : Prop :=
  let spent_at_arcade := (3/5) * A
  let remaining_after_arcade := A - spent_at_arcade
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  remaining_after_toy_store = 1.20

theorem find_weekly_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 4.50 := 
  sorry

end find_weekly_allowance_l166_166350


namespace find_missing_dimension_of_carton_l166_166266

-- Definition of given dimensions and conditions
def carton_length : ℕ := 25
def carton_width : ℕ := 48
def soap_length : ℕ := 8
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 300
def soap_volume : ℕ := soap_length * soap_width * soap_height
def total_carton_volume : ℕ := max_soap_boxes * soap_volume

-- The main statement to prove
theorem find_missing_dimension_of_carton (h : ℕ) (volume_eq : carton_length * carton_width * h = total_carton_volume) : h = 60 :=
sorry

end find_missing_dimension_of_carton_l166_166266


namespace part1_part2_l166_166925

open Real

theorem part1 (m : ℝ) (h : ∀ x : ℝ, abs (x - 2) + abs (x - 3) ≥ m) : m ≤ 1 := 
sorry

theorem part2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 = 1 / a + 1 / (2 * b) + 1 / (3 * c)) : a + 2 * b + 3 * c ≥ 9 := 
sorry

end part1_part2_l166_166925


namespace find_a_l166_166431

theorem find_a (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 20) (h2 : (56831742 - a) % 17 = 0) : a = 2 :=
by
  sorry

end find_a_l166_166431


namespace distance_to_bus_stand_l166_166909

theorem distance_to_bus_stand :
  ∀ D : ℝ, (D / 5 - 0.2 = D / 6 + 0.25) → D = 13.5 :=
by
  intros D h
  sorry

end distance_to_bus_stand_l166_166909


namespace ronaldo_current_age_l166_166055

noncomputable def roonie_age_one_year_ago (R L : ℕ) := 6 * L / 7
noncomputable def new_ratio (R L : ℕ) := (R + 5) * 8 = 7 * (L + 5)

theorem ronaldo_current_age (R L : ℕ) 
  (h1 : R = roonie_age_one_year_ago R L)
  (h2 : new_ratio R L) : L + 1 = 36 :=
by
  sorry

end ronaldo_current_age_l166_166055


namespace regression_correlation_relation_l166_166656

variable (b r : ℝ)

theorem regression_correlation_relation (h : b = 0) : r = 0 := 
sorry

end regression_correlation_relation_l166_166656


namespace seniors_selected_correct_l166_166962

-- Definitions based on the conditions problem
def total_freshmen : ℕ := 210
def total_sophomores : ℕ := 270
def total_seniors : ℕ := 300
def selected_freshmen : ℕ := 7

-- Problem statement to prove
theorem seniors_selected_correct : 
  (total_seniors / (total_freshmen / selected_freshmen)) = 10 := 
by 
  sorry

end seniors_selected_correct_l166_166962


namespace cost_sum_in_WD_l166_166123

def watch_cost_loss (W : ℝ) : ℝ := 0.9 * W
def watch_cost_gain (W : ℝ) : ℝ := 1.04 * W
def bracelet_cost_gain (B : ℝ) : ℝ := 1.08 * B
def bracelet_cost_reduced_gain (B : ℝ) : ℝ := 1.02 * B

theorem cost_sum_in_WD :
  ∃ W B : ℝ, 
    watch_cost_loss W + 196 = watch_cost_gain W ∧ 
    bracelet_cost_gain B - 100 = bracelet_cost_reduced_gain B ∧ 
    (W + B / 1.5 = 2511.11) :=
sorry

end cost_sum_in_WD_l166_166123


namespace ratio_surface_area_volume_l166_166155

theorem ratio_surface_area_volume (a b : ℕ) (h1 : a^3 = 6 * b^2) (h2 : 6 * a^2 = 6 * b) : 
  (6 * a^2) / (b^3) = 7776 :=
by
  sorry

end ratio_surface_area_volume_l166_166155


namespace area_of_intersection_is_zero_l166_166427

-- Define the circles
def circle1 (x y : ℝ) := x^2 + y^2 = 16
def circle2 (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Define the theorem to prove
theorem area_of_intersection_is_zero : 
  ∃ x1 y1 x2 y2 : ℝ,
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    x1 = x2 ∧ y1 = -y2 → 
    0 = 0 :=
by
  sorry -- proof goes here

end area_of_intersection_is_zero_l166_166427


namespace find_r_given_conditions_l166_166562

theorem find_r_given_conditions (p c r : ℝ) (h1 : p * r = 360) (h2 : 6 * c * r = 15) (h3 : r = 4) : r = 4 :=
by
  sorry

end find_r_given_conditions_l166_166562


namespace runners_speed_ratio_l166_166671

/-- Two runners, 20 miles apart, start at the same time, aiming to meet. 
    If they run in the same direction, they meet in 5 hours. 
    If they run towards each other, they meet in 1 hour.
    Prove that the ratio of the speed of the faster runner to the slower runner is 3/2. -/
theorem runners_speed_ratio (v1 v2 : ℝ) (h1 : v1 > v2)
  (h2 : 20 = 5 * (v1 - v2)) 
  (h3 : 20 = (v1 + v2)) : 
  v1 / v2 = 3 / 2 :=
sorry

end runners_speed_ratio_l166_166671


namespace add_decimal_l166_166486

theorem add_decimal (a b : ℝ) (h1 : a = 0.35) (h2 : b = 124.75) : a + b = 125.10 :=
by sorry

end add_decimal_l166_166486


namespace monotonic_decreasing_interval_l166_166733

noncomputable def xlnx (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval : 
  ∀ x, (0 < x) ∧ (x < 5) → (Real.log x + 1 < 0) ↔ (0 < x) ∧ (x < 1 / Real.exp 1) := 
by
  sorry

end monotonic_decreasing_interval_l166_166733


namespace four_digit_numbers_with_one_digit_as_average_l166_166923

noncomputable def count_valid_four_digit_numbers : Nat := 80

theorem four_digit_numbers_with_one_digit_as_average :
  ∃ n : Nat, n = count_valid_four_digit_numbers ∧ n = 80 := by
  use count_valid_four_digit_numbers
  constructor
  · rfl
  · rfl

end four_digit_numbers_with_one_digit_as_average_l166_166923


namespace min_dot_product_on_hyperbola_l166_166438

open Real

theorem min_dot_product_on_hyperbola :
  ∀ (P : ℝ × ℝ), (P.1 ≥ 1 ∧ P.1^2 - (P.2^2) / 3 = 1) →
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  ∃ m : ℝ, m = -2 ∧ PA1.1 * PF2.1 + PA1.2 * PF2.2 = m :=
by
  intros P h
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  use -2
  sorry

end min_dot_product_on_hyperbola_l166_166438


namespace dividend_is_176_l166_166753

theorem dividend_is_176 (divisor quotient remainder : ℕ) (h1 : divisor = 19) (h2 : quotient = 9) (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_is_176_l166_166753


namespace problem_nine_chapters_l166_166517

theorem problem_nine_chapters (x y : ℝ) :
  (x + (1 / 2) * y = 50) →
  (y + (2 / 3) * x = 50) →
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end problem_nine_chapters_l166_166517


namespace distinct_integers_no_perfect_square_product_l166_166717

theorem distinct_integers_no_perfect_square_product
  (k : ℕ) (hk : 0 < k) :
  ∀ a b : ℕ, k^2 < a ∧ a < (k+1)^2 → k^2 < b ∧ b < (k+1)^2 → a ≠ b → ¬∃ m : ℕ, a * b = m^2 :=
by sorry

end distinct_integers_no_perfect_square_product_l166_166717


namespace largest_b_value_l166_166943

open Real

structure Triangle :=
(side_a side_b side_c : ℝ)
(a_pos : 0 < side_a)
(b_pos : 0 < side_b)
(c_pos : 0 < side_c)
(tri_ineq_a : side_a + side_b > side_c)
(tri_ineq_b : side_b + side_c > side_a)
(tri_ineq_c : side_c + side_a > side_b)

noncomputable def inradius (T : Triangle) : ℝ :=
  let s := (T.side_a + T.side_b + T.side_c) / 2
  let A := sqrt (s * (s - T.side_a) * (s - T.side_b) * (s - T.side_c))
  A / s

noncomputable def circumradius (T : Triangle) : ℝ :=
  let A := sqrt (((T.side_a + T.side_b + T.side_c) / 2) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_a) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_b) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_c))
  (T.side_a * T.side_b * T.side_c) / (4 * A)

noncomputable def condition_met (T1 T2 : Triangle) : Prop :=
  (inradius T1 / circumradius T1) = (inradius T2 / circumradius T2)

theorem largest_b_value :
  let T1 := Triangle.mk 8 11 11 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  ∃ b > 0, ∃ T2 : Triangle, T2.side_a = b ∧ T2.side_b = 1 ∧ T2.side_c = 1 ∧ b = 14 / 11 ∧ condition_met T1 T2 :=
  sorry

end largest_b_value_l166_166943


namespace ratio_of_segments_l166_166050

theorem ratio_of_segments
  (x y z u v : ℝ)
  (h_triangle : x^2 + y^2 = z^2)
  (h_ratio_legs : 4 * x = 3 * y)
  (h_u : u = x^2 / z)
  (h_v : v = y^2 / z) :
  u / v = 9 / 16 := 
  sorry

end ratio_of_segments_l166_166050


namespace final_percentage_acid_l166_166823

theorem final_percentage_acid (initial_volume : ℝ) (initial_percentage : ℝ)
(removal_volume : ℝ) (final_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 12 → 
  initial_percentage = 0.40 → 
  removal_volume = 4 →
  final_volume = initial_volume - removal_volume →
  final_percentage = (initial_percentage * initial_volume) / final_volume * 100 →
  final_percentage = 60 := by
  intros h1 h2 h3 h4 h5
  sorry

end final_percentage_acid_l166_166823


namespace grasshoppers_total_l166_166010

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end grasshoppers_total_l166_166010


namespace value_of_t_l166_166653

def vec (x y : ℝ) := (x, y)

def p := vec 3 3
def q := vec (-1) 2
def r := vec 4 1

noncomputable def t := 3

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem value_of_t (t : ℝ) : (dot_product (vec (6 + 4 * t) (6 + t)) q = 0) ↔ t = 3 :=
by
  sorry

end value_of_t_l166_166653


namespace distance_behind_l166_166632

-- Given conditions
variables {A B E : ℝ} -- Speed of Anusha, Banu, and Esha
variables {Da Db De : ℝ} -- distances covered by Anusha, Banu, and Esha

axiom const_speeds : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)

-- The proof to be established
theorem distance_behind (h : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)) :
  100 - De = 19 :=
by sorry

end distance_behind_l166_166632


namespace quadratic_polynomial_value_l166_166285

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end quadratic_polynomial_value_l166_166285


namespace problem_solution_l166_166307

theorem problem_solution (a b c : ℤ)
  (h1 : ∀ x : ℤ, |x| ≠ |a|)
  (h2 : ∀ x : ℤ, x^2 ≠ b^2)
  (h3 : ∀ x : ℤ, x * c ≤ 1):
  a + b + c = 0 :=
by sorry

end problem_solution_l166_166307


namespace simple_interest_calculation_l166_166606

-- Define the known quantities
def principal : ℕ := 400
def rate_of_interest : ℕ := 15
def time : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Statement to be proved
theorem simple_interest_calculation :
  simple_interest principal rate_of_interest time = 60 :=
by
  -- This space is used for the proof, We assume the user will complete it
  sorry

end simple_interest_calculation_l166_166606


namespace lollipop_problem_l166_166873

def arithmetic_sequence_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem lollipop_problem
  (a : ℕ) (h1 : arithmetic_sequence_sum a 5 7 = 175) :
  (a + 15) = 25 :=
by
  sorry

end lollipop_problem_l166_166873


namespace dividend_is_correct_l166_166797

theorem dividend_is_correct :
  ∃ (R D Q V: ℕ), R = 6 ∧ D = 5 * Q ∧ D = 3 * R + 2 ∧ V = D * Q + R ∧ V = 86 :=
by
  sorry

end dividend_is_correct_l166_166797


namespace min_value_of_expression_l166_166169

noncomputable def problem_statement : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ ((1/x) + (1/y) + (1/z) = 9) ∧ (x^2 * y^3 * z^2 = 1/2268)

theorem min_value_of_expression :
  problem_statement := 
sorry

end min_value_of_expression_l166_166169


namespace bed_height_l166_166130

noncomputable def bed_length : ℝ := 8
noncomputable def bed_width : ℝ := 4
noncomputable def bags_of_soil : ℕ := 16
noncomputable def soil_per_bag : ℝ := 4
noncomputable def total_volume_of_soil : ℝ := bags_of_soil * soil_per_bag
noncomputable def number_of_beds : ℕ := 2
noncomputable def volume_per_bed : ℝ := total_volume_of_soil / number_of_beds

theorem bed_height :
  volume_per_bed / (bed_length * bed_width) = 1 :=
sorry

end bed_height_l166_166130


namespace planning_committee_ways_is_20_l166_166989

-- Define the number of students in the council
def num_students : ℕ := 6

-- Define the ways to choose a 3-person committee from num_students
def committee_ways (x : ℕ) : ℕ := Nat.choose x 3

-- Given condition: number of ways to choose the welcoming committee is 20
axiom welcoming_committee_condition : committee_ways num_students = 20

-- Statement to prove
theorem planning_committee_ways_is_20 : committee_ways num_students = 20 := by
  exact welcoming_committee_condition

end planning_committee_ways_is_20_l166_166989


namespace x_squared_minus_y_squared_l166_166680

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end x_squared_minus_y_squared_l166_166680


namespace triangle_equilateral_if_abs_eq_zero_l166_166931

theorem triangle_equilateral_if_abs_eq_zero (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_if_abs_eq_zero_l166_166931


namespace S21_sum_is_4641_l166_166801

-- Define the conditions and the sum of the nth group
def first_number_in_group (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

def last_number_in_group (n : ℕ) : ℕ :=
  first_number_in_group n + (n - 1)

def sum_of_group (n : ℕ) : ℕ :=
  n * (first_number_in_group n + last_number_in_group n) / 2

-- The theorem to prove
theorem S21_sum_is_4641 : sum_of_group 21 = 4641 := by
  sorry

end S21_sum_is_4641_l166_166801


namespace fraction_zero_iff_x_neg_one_l166_166154

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h₀ : x^2 - 1 = 0) (h₁ : x - 1 ≠ 0) :
  (x^2 - 1) / (x - 1) = 0 ↔ x = -1 :=
by
  sorry

end fraction_zero_iff_x_neg_one_l166_166154


namespace calc_val_l166_166821

theorem calc_val : 
  (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 :=
by 
  -- Calculation proof
  sorry

end calc_val_l166_166821


namespace a4_value_l166_166043

-- Definitions and helper theorems can go here
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- These are our conditions
axiom h1 : S 2 = a 1 + a 2
axiom h2 : a 2 = 3
axiom h3 : ∀ n, S (n + 1) = 2 * S n + 1

theorem a4_value : a 4 = 12 :=
sorry  -- proof to be filled in later

end a4_value_l166_166043


namespace triangle_area_example_l166_166447

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_example :
  triangle_area (-2, 3) (7, -1) (4, 6) = 25.5 :=
by
  -- Proof will be here
  sorry

end triangle_area_example_l166_166447


namespace meal_cost_per_person_l166_166781

/-
Problem Statement:
Prove that the cost per meal is $3 given the conditions:
- There are 2 adults and 5 children.
- The total bill is $21.
-/

theorem meal_cost_per_person (total_adults : ℕ) (total_children : ℕ) (total_bill : ℝ) 
(total_people : ℕ) (cost_per_meal : ℝ) : 
total_adults = 2 → total_children = 5 → total_bill = 21 → total_people = total_adults + total_children →
cost_per_meal = total_bill / total_people → 
cost_per_meal = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end meal_cost_per_person_l166_166781


namespace find_multiple_l166_166238

variable (P W : ℕ)
variable (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
variable (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2)

theorem find_multiple (P W : ℕ) (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
                      (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2) : m = 2 :=
by
  sorry

end find_multiple_l166_166238


namespace prove_f_10_l166_166294

variable (f : ℝ → ℝ)

-- Conditions from the problem
def condition : Prop := ∀ x : ℝ, f (3 ^ x) = x

-- Statement of the problem
theorem prove_f_10 (h : condition f) : f 10 = Real.log 10 / Real.log 3 :=
by
  sorry

end prove_f_10_l166_166294


namespace train_crossing_time_l166_166553

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

end train_crossing_time_l166_166553


namespace final_position_l166_166716

structure Position where
  base : ℝ × ℝ
  stem : ℝ × ℝ

def rotate180 (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectX (pos : Position) : Position :=
  { base := (pos.base.1, -pos.base.2),
    stem := (pos.stem.1, -pos.stem.2) }

def rotateHalfTurn (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectY (pos : Position) : Position :=
  { base := (-pos.base.1, pos.base.2),
    stem := (-pos.stem.1, pos.stem.2) }

theorem final_position : 
  let initial_pos := Position.mk (1, 0) (0, 1)
  let pos1 := rotate180 initial_pos
  let pos2 := reflectX pos1
  let pos3 := rotateHalfTurn pos2
  let final_pos := reflectY pos3
  final_pos = { base := (-1, 0), stem := (0, -1) } :=
by
  sorry

end final_position_l166_166716


namespace sign_of_c_l166_166226

theorem sign_of_c (a b c : ℝ) (h1 : (a * b / c) < 0) (h2 : (a * b) < 0) : c > 0 :=
sorry

end sign_of_c_l166_166226


namespace team_total_points_l166_166136

theorem team_total_points (three_points_goals: ℕ) (two_points_goals: ℕ) (half_of_total: ℕ) 
  (h1 : three_points_goals = 5) 
  (h2 : two_points_goals = 10) 
  (h3 : half_of_total = (3 * three_points_goals + 2 * two_points_goals) / 2) 
  : 2 * half_of_total = 70 := 
by 
  -- proof to be filled
  sorry

end team_total_points_l166_166136


namespace man_is_26_years_older_l166_166020

variable (S : ℕ) (M : ℕ)

-- conditions
def present_age_of_son : Prop := S = 24
def future_age_relation : Prop := M + 2 = 2 * (S + 2)

-- question transformed to a proof problem
theorem man_is_26_years_older
  (h1 : present_age_of_son S)
  (h2 : future_age_relation S M) : M - S = 26 := by
  sorry

end man_is_26_years_older_l166_166020


namespace area_of_trapezium_l166_166197

-- Definitions
def length_parallel_side_1 : ℝ := 4
def length_parallel_side_2 : ℝ := 5
def perpendicular_distance : ℝ := 6

-- Statement
theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side_1 + length_parallel_side_2) * perpendicular_distance = 27 :=
by
  sorry

end area_of_trapezium_l166_166197


namespace hydrogen_atoms_in_compound_l166_166179

theorem hydrogen_atoms_in_compound :
  ∀ (molecular_weight_of_compound atomic_weight_Al atomic_weight_O atomic_weight_H : ℕ)
    (num_Al num_O num_H : ℕ),
    molecular_weight_of_compound = 78 →
    atomic_weight_Al = 27 →
    atomic_weight_O = 16 →
    atomic_weight_H = 1 →
    num_Al = 1 →
    num_O = 3 →
    molecular_weight_of_compound = 
      (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H) →
    num_H = 3 := by
  intros
  sorry

end hydrogen_atoms_in_compound_l166_166179


namespace mark_siblings_l166_166975

theorem mark_siblings (total_eggs : ℕ) (eggs_per_person : ℕ) (persons_including_mark : ℕ) (h1 : total_eggs = 24) (h2 : eggs_per_person = 6) (h3 : persons_including_mark = total_eggs / eggs_per_person) : persons_including_mark - 1 = 3 :=
by 
  sorry

end mark_siblings_l166_166975


namespace koschei_coins_l166_166664

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l166_166664


namespace maria_money_left_l166_166834

def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def transportation_cost : ℕ := 80
def num_days : ℕ := 5
def avg_meal_cost_per_day : ℕ := 40
def tourist_tax_rate : ℚ := 0.10
def starting_amount : ℕ := 760

def total_meal_cost : ℕ := num_days * avg_meal_cost_per_day
def expenses_subject_to_tax := hotel_cost + transportation_cost
def tourist_tax := tourist_tax_rate * expenses_subject_to_tax
def total_expenses := ticket_cost + hotel_cost + transportation_cost + total_meal_cost + tourist_tax
def money_left := starting_amount - total_expenses

theorem maria_money_left : money_left = 7 := by
  sorry

end maria_money_left_l166_166834


namespace numerator_of_fraction_l166_166170

theorem numerator_of_fraction (y x : ℝ) (hy : y > 0) (h : (9 * y) / 20 + x / y = 0.75 * y) : x = 3 :=
sorry

end numerator_of_fraction_l166_166170


namespace area_inside_C_but_outside_A_and_B_l166_166474

def radius_A := 1
def radius_B := 1
def radius_C := 2
def tangency_AB := true
def tangency_AC_non_midpoint := true

theorem area_inside_C_but_outside_A_and_B :
  let areaC := π * (radius_C ^ 2)
  let areaA := π * (radius_A ^ 2)
  let areaB := π * (radius_B ^ 2)
  let overlapping_area := 2 * (π * (radius_A ^ 2) / 2) -- approximation
  areaC - overlapping_area = 3 * π - 2 :=
by
  sorry

end area_inside_C_but_outside_A_and_B_l166_166474


namespace evaluate_expression_l166_166852

theorem evaluate_expression (b : ℤ) (x : ℤ) (h : x = b + 9) : (x - b + 5 = 14) :=
by
  sorry

end evaluate_expression_l166_166852


namespace roots_of_unity_polynomial_l166_166793

theorem roots_of_unity_polynomial (c d : ℤ) (z : ℂ) (hz : z^3 = 1) :
  (z^3 + c * z + d = 0) → (z = 1) :=
sorry

end roots_of_unity_polynomial_l166_166793


namespace binomial_identity_l166_166840

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) :=
sorry

end binomial_identity_l166_166840


namespace day_of_18th_day_of_month_is_tuesday_l166_166549

theorem day_of_18th_day_of_month_is_tuesday
  (day_of_24th_is_monday : ℕ → ℕ)
  (mod_seven : ∀ n, n % 7 = n)
  (h24 : day_of_24th_is_monday 24 = 1) : day_of_24th_is_monday 18 = 2 :=
by
  sorry

end day_of_18th_day_of_month_is_tuesday_l166_166549


namespace correct_statement_l166_166372

theorem correct_statement (x : ℝ) : 
  (∃ y : ℝ, y ≠ 0 ∧ y * x = 1 → x = 1 ∨ x = -1 ∨ x = 0) → false ∧
  (∃ y : ℝ, -y = y → y = 0 ∨ y = 1) → false ∧
  (abs x = x → x ≥ 0) → (x ^ 2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end correct_statement_l166_166372


namespace misha_darts_score_l166_166858

theorem misha_darts_score (x : ℕ) 
  (h1 : x >= 24)
  (h2 : x * 3 <= 72) : 
  2 * x = 48 :=
by
  sorry

end misha_darts_score_l166_166858


namespace alice_acorns_purchase_l166_166451

variable (bob_payment : ℕ) (alice_payment_rate : ℕ) (price_per_acorn : ℕ)

-- Given conditions
def bob_paid : Prop := bob_payment = 6000
def alice_paid : Prop := alice_payment_rate = 9
def acorn_price : Prop := price_per_acorn = 15

-- Proof statement
theorem alice_acorns_purchase
  (h1 : bob_paid bob_payment)
  (h2 : alice_paid alice_payment_rate)
  (h3 : acorn_price price_per_acorn) :
  ∃ n : ℕ, n = (alice_payment_rate * bob_payment) / price_per_acorn ∧ n = 3600 := 
by
  sorry

end alice_acorns_purchase_l166_166451


namespace elapsed_time_l166_166009

theorem elapsed_time (x : ℕ) (h1 : 99 > 0) (h2 : (2 : ℚ) / (3 : ℚ) * x = (4 : ℚ) / (5 : ℚ) * (99 - x)) : x = 54 := by
  sorry

end elapsed_time_l166_166009


namespace complement_intersection_l166_166831

def U : Set ℝ := fun x => True
def A : Set ℝ := fun x => x < 0
def B : Set ℝ := fun x => x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2

theorem complement_intersection (hU : ∀ x : ℝ, U x) :
  ((compl A) ∩ B) = {0, 1, 2} :=
by {
  sorry
}

end complement_intersection_l166_166831


namespace unique_polynomial_P_l166_166041

noncomputable def P : ℝ → ℝ := sorry

axiom P_func_eq (x : ℝ) : P (x^2 + 1) = P x ^ 2 + 1
axiom P_zero : P 0 = 0

theorem unique_polynomial_P (x : ℝ) : P x = x :=
by
  sorry

end unique_polynomial_P_l166_166041


namespace arithmetic_sequence_sum_l166_166888

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n+1) = a n + d)
  (h1 : a 2 + a 3 = 1)
  (h2 : a 10 + a 11 = 9) :
  a 5 + a 6 = 4 :=
sorry

end arithmetic_sequence_sum_l166_166888


namespace probability_margo_pairing_l166_166065

-- Definition of the problem
def num_students : ℕ := 32
def num_pairings (n : ℕ) : ℕ := n - 1
def favorable_pairings : ℕ := 2

-- Theorem statement
theorem probability_margo_pairing :
  num_students = 32 →
  ∃ (p : ℚ), p = favorable_pairings / num_pairings num_students ∧ p = 2/31 :=
by
  intros h
  -- The proofs are omitted for brevity.
  sorry

end probability_margo_pairing_l166_166065


namespace find_number_l166_166703

theorem find_number (x : ℕ) (h : x + 8 = 500) : x = 492 :=
by sorry

end find_number_l166_166703


namespace geometric_sequence_term_l166_166460

noncomputable def b_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 1 => Real.sin x ^ 2
  | 2 => Real.sin x * Real.cos x
  | 3 => Real.cos x ^ 2 / Real.sin x
  | n + 4 => (Real.cos x / Real.sin x) ^ n * Real.cos x ^ 3 / Real.sin x ^ 2
  | _ => 0 -- Placeholder to cover all case

theorem geometric_sequence_term (x : ℝ) :
  ∃ n, b_n n x = Real.cos x + Real.sin x ∧ n = 7 := by
  sorry

end geometric_sequence_term_l166_166460


namespace earnings_difference_l166_166556

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l166_166556


namespace find_set_A_find_range_a_l166_166434

-- Define the universal set and the complement condition for A
def universal_set : Set ℝ := {x | true}
def complement_A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 > 0}

-- Define the set A
def set_A : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 2}

-- Define the set C
def set_C (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

-- Define the proof problem for part (1)
theorem find_set_A : { x | -1 / 2 ≤ x ∧ x ≤ 2 } = { x | ¬ (2 * x^2 - 3 * x - 2 > 0) } :=
by
  sorry

-- Define the proof problem for part (2)
theorem find_range_a (a : ℝ) (C_ne_empty : (set_C a).Nonempty) (sufficient_not_necessary : ∀ x, x ∈ set_C a → x ∈ set_A → x ∈ set_A) :
  a ∈ Set.Icc (-1/8 : ℝ) 0 ∪ Set.Icc 1 (4/3 : ℝ) :=
by
  sorry

end find_set_A_find_range_a_l166_166434


namespace ellipse_eccentricity_l166_166907

theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b : b > 0) (c : ℝ)
  (h_ellipse : (b^2 / c^2) = 3)
  (eccentricity_eq : ∀ (e : ℝ), e = c / a ↔ e = 1 / 2) : 
  ∃ e, e = (c / a) :=
by {
  sorry
}

end ellipse_eccentricity_l166_166907


namespace zeros_of_quadratic_l166_166935

theorem zeros_of_quadratic (a b : ℝ) (h : a + b = 0) : 
  ∀ x, (b * x^2 - a * x = 0) ↔ (x = 0 ∨ x = -1) :=
by
  intro x
  sorry

end zeros_of_quadratic_l166_166935


namespace percentage_increase_l166_166893

theorem percentage_increase (x : ℝ) (y : ℝ) (h1 : x = 114.4) (h2 : y = 88) : 
  ((x - y) / y) * 100 = 30 := 
by 
  sorry

end percentage_increase_l166_166893


namespace equal_sums_arithmetic_sequences_l166_166561

-- Define the arithmetic sequences and their sums
def s₁ (n : ℕ) : ℕ := n * (5 * n + 13) / 2
def s₂ (n : ℕ) : ℕ := n * (3 * n + 37) / 2

-- State the theorem: for given n != 0, prove s₁ n = s₂ n implies n = 12
theorem equal_sums_arithmetic_sequences (n : ℕ) (h : n ≠ 0) : 
  s₁ n = s₂ n → n = 12 :=
by
  sorry

end equal_sums_arithmetic_sequences_l166_166561


namespace sphere_surface_area_l166_166835

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) : ∃ A, A = 36 * Real.pi * (2 ^ (2 / 3)) := 
by
  sorry

end sphere_surface_area_l166_166835


namespace factorize_expression_l166_166317

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l166_166317


namespace product_price_interval_l166_166640

def is_too_high (price guess : ℕ) : Prop := guess > price
def is_too_low  (price guess : ℕ) : Prop := guess < price

theorem product_price_interval 
    (price : ℕ)
    (h1 : is_too_high price 2000)
    (h2 : is_too_low price 1000)
    (h3 : is_too_high price 1500)
    (h4 : is_too_low price 1250)
    (h5 : is_too_low price 1375) :
    1375 < price ∧ price < 1500 :=
    sorry

end product_price_interval_l166_166640


namespace rational_sum_of_cubes_l166_166622

theorem rational_sum_of_cubes (t : ℚ) : 
    ∃ (a b c : ℚ), t = (a^3 + b^3 + c^3) :=
by
  sorry

end rational_sum_of_cubes_l166_166622


namespace members_play_both_l166_166724

-- Define the conditions
variables (N B T neither : ℕ)
variables (B_union_T B_and_T : ℕ)

-- Assume the given conditions
axiom hN : N = 42
axiom hB : B = 20
axiom hT : T = 23
axiom hNeither : neither = 6
axiom hB_union_T : B_union_T = N - neither

-- State the problem: Prove that B_and_T = 7
theorem members_play_both (N B T neither B_union_T B_and_T : ℕ) 
  (hN : N = 42) 
  (hB : B = 20) 
  (hT : T = 23) 
  (hNeither : neither = 6) 
  (hB_union_T : B_union_T = N - neither) 
  (hInclusionExclusion : B_union_T = B + T - B_and_T) :
  B_and_T = 7 := sorry

end members_play_both_l166_166724


namespace candies_for_50_rubles_l166_166470

theorem candies_for_50_rubles : 
  ∀ (x : ℕ), (45 * x = 45) → (50 / x = 50) := 
by
  intros x h
  sorry

end candies_for_50_rubles_l166_166470


namespace part1_part2_l166_166436

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

theorem part1 {x : ℝ} (hx : x = 0) : 
    f x 0 = 1 :=
by
  sorry

theorem part2 {x k : ℝ} (hx : 0 ≤ x) (hxf : f x k ≥ 1) : 
    k ≤ 1 / 2 :=
by
  sorry

end part1_part2_l166_166436


namespace problem_1_problem_2_l166_166319

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x)
def g (x : ℝ) : ℝ := x^2

theorem problem_1 (a : ℝ) (ha : a ≠ 0) : 
  (∀ (x : ℝ), f a x = a * (x + Real.log x)) →
  deriv (f a) 1 = deriv g 1 → a = 1 := 
by 
  sorry

theorem problem_2 (a : ℝ) (ha : 0 < a) (hb : a < 1) (x1 x2 : ℝ) 
  (hx1 : 1 ≤ x1) (hx2 : x2 ≤ 2) (hx12 : x1 ≠ x2) : 
  |f a x1 - f a x2| < |g x1 - g x2| := 
by 
  sorry

end problem_1_problem_2_l166_166319


namespace sin_alpha_eq_63_over_65_l166_166967

open Real

variables {α β : ℝ}

theorem sin_alpha_eq_63_over_65
  (h1 : tan β = 4 / 3)
  (h2 : sin (α + β) = 5 / 13)
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  sin α = 63 / 65 := 
by
  sorry

end sin_alpha_eq_63_over_65_l166_166967


namespace abhinav_annual_salary_l166_166748

def RamMontlySalary : ℝ := 25600
def ShyamMontlySalary (A : ℝ) := 2 * A
def AbhinavAnnualSalary (A : ℝ) := 12 * A

theorem abhinav_annual_salary (A : ℝ) : 
  0.10 * RamMontlySalary = 0.08 * ShyamMontlySalary A → 
  AbhinavAnnualSalary A = 192000 :=
by
  sorry

end abhinav_annual_salary_l166_166748


namespace jessica_cut_roses_l166_166592

/-- There were 13 roses and 84 orchids in the vase. Jessica cut some more roses and 
orchids from her flower garden. There are now 91 orchids and 14 roses in the vase. 
How many roses did she cut? -/
theorem jessica_cut_roses :
  let initial_roses := 13
  let new_roses := 14
  ∃ cut_roses : ℕ, new_roses = initial_roses + cut_roses ∧ cut_roses = 1 :=
by
  sorry

end jessica_cut_roses_l166_166592


namespace system1_solution_system2_solution_l166_166809

-- System 1
theorem system1_solution (x y : ℝ) 
  (h1 : y = 2 * x - 3)
  (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := 
by
  sorry

-- System 2
theorem system2_solution (x y : ℝ) 
  (h1 : x + 2 * y = 3)
  (h2 : 2 * x - 4 * y = -10) : 
  x = -1 ∧ y = 2 := 
by
  sorry

end system1_solution_system2_solution_l166_166809


namespace fill_pool_with_B_only_l166_166453

theorem fill_pool_with_B_only
    (time_AB : ℝ)
    (R_AB : time_AB = 30)
    (time_A_B_then_B : ℝ)
    (R_A_B_then_B : (10 / 30 + (time_A_B_then_B - 10) / time_A_B_then_B) = 1)
    (only_B_time : ℝ)
    (R_B : only_B_time = 60) :
    only_B_time = 60 :=
by
    sorry

end fill_pool_with_B_only_l166_166453


namespace average_height_of_females_at_school_l166_166124

-- Define the known quantities and conditions
variable (total_avg_height male_avg_height female_avg_height : ℝ)
variable (male_count female_count : ℕ)

-- Given conditions
def conditions :=
  total_avg_height = 180 ∧ 
  male_avg_height = 185 ∧ 
  male_count = 2 * female_count ∧
  (male_count + female_count) * total_avg_height = male_count * male_avg_height + female_count * female_avg_height

-- The theorem we want to prove
theorem average_height_of_females_at_school (total_avg_height male_avg_height female_avg_height : ℝ)
    (male_count female_count : ℕ) (h : conditions total_avg_height male_avg_height female_avg_height male_count female_count) :
    female_avg_height = 170 :=
  sorry

end average_height_of_females_at_school_l166_166124


namespace problem_prove_divisibility_l166_166649

theorem problem_prove_divisibility (n : ℕ) : 11 ∣ (5^(2*n) + 3^(n+2) + 3^n) :=
sorry

end problem_prove_divisibility_l166_166649


namespace solve_fractional_equation_l166_166945

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x + 1 ≠ 0) :
  (1 / x = 2 / (x + 1)) → x = 1 := 
by
  sorry

end solve_fractional_equation_l166_166945


namespace probability_green_given_not_red_l166_166277

theorem probability_green_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let green_balls := 10
  let non_red_balls := total_balls - red_balls

  let probability_green_given_not_red := (green_balls : ℚ) / (non_red_balls : ℚ)

  probability_green_given_not_red = 2 / 3 :=
by
  sorry

end probability_green_given_not_red_l166_166277


namespace cost_of_apples_is_2_l166_166573

variable (A : ℝ)

def cost_of_apples (A : ℝ) : ℝ := 5 * A
def cost_of_sugar (A : ℝ) : ℝ := 3 * (A - 1)
def cost_of_walnuts : ℝ := 0.5 * 6
def total_cost (A : ℝ) : ℝ := cost_of_apples A + cost_of_sugar A + cost_of_walnuts

theorem cost_of_apples_is_2 (A : ℝ) (h : total_cost A = 16) : A = 2 := 
by 
  sorry

end cost_of_apples_is_2_l166_166573


namespace expected_ties_after_10_l166_166986

def binom: ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom n (k+1)

noncomputable def expected_ties : ℕ → ℝ 
| 0 => 0
| n+1 => expected_ties n + (binom (2*(n+1)) (n+1) / 2^(2*(n+1)))

theorem expected_ties_after_10 : expected_ties 5 = 1.707 := 
by 
  -- Placeholder for the actual proof
  sorry

end expected_ties_after_10_l166_166986


namespace cost_to_feed_turtles_l166_166263

-- Define the conditions
def ounces_per_half_pound : ℝ := 1 
def total_weight_turtles : ℝ := 30
def food_per_half_pound : ℝ := 0.5
def ounces_per_jar : ℝ := 15
def cost_per_jar : ℝ := 2

-- Define the statement to prove
theorem cost_to_feed_turtles : (total_weight_turtles / food_per_half_pound) / ounces_per_jar * cost_per_jar = 8 := by
  sorry

end cost_to_feed_turtles_l166_166263


namespace betty_sugar_l166_166300

theorem betty_sugar (f s : ℝ) (hf1 : f ≥ 8 + (3 / 4) * s) (hf2 : f ≤ 3 * s) : s ≥ 4 := 
sorry

end betty_sugar_l166_166300


namespace course_count_l166_166340

theorem course_count (n1 n2 : ℕ) (sum_x1 sum_x2 : ℕ) :
  (n1 = 6) →
  (sum_x1 = n1 * 100) →
  (sum_x2 = n2 * 50) →
  ((sum_x1 + sum_x2) / (n1 + n2) = 77) →
  n2 = 5 :=
by
  intros h1 h2 h3 h4
  sorry

end course_count_l166_166340


namespace relay_race_total_time_l166_166312

-- Definitions based on the problem conditions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25

-- Problem statement
theorem relay_race_total_time : 
  athlete1_time + athlete2_time + athlete3_time + athlete4_time = 200 := 
by 
  sorry

end relay_race_total_time_l166_166312


namespace leonard_younger_than_nina_by_4_l166_166946

variable (L N J : ℕ)

-- Conditions based on conditions from the problem
axiom h1 : L = 6
axiom h2 : N = 1 / 2 * J
axiom h3 : L + N + J = 36

-- Statement to prove
theorem leonard_younger_than_nina_by_4 : N - L = 4 :=
by 
  sorry

end leonard_younger_than_nina_by_4_l166_166946


namespace evaluate_given_condition_l166_166902

noncomputable def evaluate_expression (b : ℚ) : ℚ :=
  (7 * b^2 - 15 * b + 5) * (3 * b - 4)

theorem evaluate_given_condition (b : ℚ) (h : b = 4 / 3) : evaluate_expression b = 0 := by
  sorry

end evaluate_given_condition_l166_166902


namespace rectangle_length_l166_166185

theorem rectangle_length : 
  ∃ l b : ℝ, 
    (l = 2 * b) ∧ 
    (20 < l ∧ l < 50) ∧ 
    (10 < b ∧ b < 30) ∧ 
    ((l - 5) * (b + 5) = l * b + 75) ∧ 
    (l = 40) :=
sorry

end rectangle_length_l166_166185


namespace rectangular_garden_length_l166_166174

theorem rectangular_garden_length (L P B : ℕ) (h1 : P = 600) (h2 : B = 150) (h3 : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangular_garden_length_l166_166174


namespace bike_ride_ratio_l166_166297

theorem bike_ride_ratio (J : ℕ) (B : ℕ) (M : ℕ) (hB : B = 17) (hM : M = J + 10) (hTotal : B + J + M = 95) :
  J / B = 2 :=
by
  sorry

end bike_ride_ratio_l166_166297


namespace cos_2x_eq_cos_2y_l166_166772

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end cos_2x_eq_cos_2y_l166_166772


namespace angle_BCA_measure_l166_166827

theorem angle_BCA_measure
  (A B C : Type)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_BAC : ℝ)
  (h1 : angle_ABC = 90)
  (h2 : angle_BAC = 2 * angle_BCA) :
  angle_BCA = 30 :=
by
  sorry

end angle_BCA_measure_l166_166827


namespace problem_1_problem_2_l166_166466

theorem problem_1 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 3 / 4 := 
sorry

theorem problem_2 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a / (b + 2 * c + 3 * d) + b / (c + 2 * d + 3 * a) + c / (d + 2 * a + 3 * b) + d / (a + 2 * b + 3 * c)) ≥ 2 / 3 :=
sorry

end problem_1_problem_2_l166_166466


namespace percent_of_juniors_involved_in_sports_l166_166581

theorem percent_of_juniors_involved_in_sports
  (total_students : ℕ)
  (percent_juniors : ℝ)
  (juniors_in_sports : ℕ)
  (h1 : total_students = 500)
  (h2 : percent_juniors = 0.40)
  (h3 : juniors_in_sports = 140) :
  (juniors_in_sports : ℝ) / (total_students * percent_juniors) * 100 = 70 := 
by
  -- By conditions h1, h2, h3:
  sorry

end percent_of_juniors_involved_in_sports_l166_166581


namespace find_n_find_m_constant_term_find_m_max_coefficients_l166_166514

-- 1. Prove that if the sum of the binomial coefficients is 256, then n = 8.
theorem find_n (n : ℕ) (h : 2^n = 256) : n = 8 :=
by sorry

-- 2. Prove that if the constant term is 35/8, then m = ±1/2.
theorem find_m_constant_term (m : ℚ) (h : m^4 * (Nat.choose 8 4) = 35/8) : m = 1/2 ∨ m = -1/2 :=
by sorry

-- 3. Prove that if only the 6th and 7th terms have the maximum coefficients, then m = 2.
theorem find_m_max_coefficients (m : ℚ) (h1 : m ≠ 0) (h2 : m^5 * (Nat.choose 8 5) = m^6 * (Nat.choose 8 6)) : m = 2 :=
by sorry

end find_n_find_m_constant_term_find_m_max_coefficients_l166_166514


namespace total_money_is_twenty_l166_166586

-- Define Henry's initial money
def henry_initial_money : Nat := 5

-- Define the money Henry earned
def henry_earned_money : Nat := 2

-- Define Henry's total money
def henry_total_money : Nat := henry_initial_money + henry_earned_money

-- Define friend's money
def friend_money : Nat := 13

-- Define the total combined money
def total_combined_money : Nat := henry_total_money + friend_money

-- The main statement to prove
theorem total_money_is_twenty : total_combined_money = 20 := sorry

end total_money_is_twenty_l166_166586


namespace smallest_divisible_by_15_18_20_is_180_l166_166465

theorem smallest_divisible_by_15_18_20_is_180 :
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (20 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (20 ∣ m)) → n ≤ m ∧ n = 180 := by
  sorry

end smallest_divisible_by_15_18_20_is_180_l166_166465


namespace min_value_quadratic_l166_166361

theorem min_value_quadratic : 
  ∀ x : ℝ, (4 * x^2 - 12 * x + 9) ≥ 0 :=
by
  sorry

end min_value_quadratic_l166_166361


namespace evaluate_f_at_5_l166_166347

def f (x : ℝ) := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 524

theorem evaluate_f_at_5 : f 5 = 2176 :=
by
  sorry

end evaluate_f_at_5_l166_166347


namespace current_number_of_women_is_24_l166_166661

-- Define initial person counts based on the given ratio and an arbitrary factor x.
variables (x : ℕ)
def M_initial := 4 * x
def W_initial := 5 * x
def C_initial := 3 * x
def E_initial := 2 * x

-- Define the changes that happened to the room.
def men_after_entry := M_initial x + 2
def women_after_leaving := W_initial x - 3
def women_after_doubling := 2 * women_after_leaving x
def children_after_leaving := C_initial x - 5
def elderly_after_leaving := E_initial x - 3

-- Define the current counts after all changes.
def men_current := 14
def children_current := 7
def elderly_current := 6

-- Prove that the current number of women is 24.
theorem current_number_of_women_is_24 :
  men_after_entry x = men_current ∧
  children_after_leaving x = children_current ∧
  elderly_after_leaving x = elderly_current →
  women_after_doubling x = 24 :=
by
  sorry

end current_number_of_women_is_24_l166_166661


namespace probability_sum_18_l166_166819

def total_outcomes := 100

def successful_pairs := [(8, 10), (9, 9), (10, 8)]

def num_successful_outcomes := successful_pairs.length

theorem probability_sum_18 : (num_successful_outcomes / total_outcomes : ℚ) = 3 / 100 := 
by
  -- The actual proof should go here
  sorry

end probability_sum_18_l166_166819


namespace min_sum_a_b_l166_166295

theorem min_sum_a_b (a b : ℕ) (h1 : a ≠ b) (h2 : 0 < a ∧ 0 < b) (h3 : (1/a + 1/b) = 1/12) : a + b = 54 :=
sorry

end min_sum_a_b_l166_166295


namespace find_m_if_purely_imaginary_l166_166651

theorem find_m_if_purely_imaginary : ∀ m : ℝ, (m^2 - 5*m + 6 = 0) → (m = 2) :=
by 
  intro m
  intro h
  sorry

end find_m_if_purely_imaginary_l166_166651


namespace area_of_segment_l166_166151

theorem area_of_segment (R : ℝ) (hR : R > 0) (h_perimeter : 4 * R = 2 * R + 2 * R) :
  (1 - (1 / 2) * Real.sin 2) * R^2 = (fun R => (1 - (1 / 2) * Real.sin 2) * R^2) R :=
by
  sorry

end area_of_segment_l166_166151


namespace right_triangle_area_l166_166078

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l166_166078


namespace parabola_directrix_eq_l166_166308

theorem parabola_directrix_eq (p : ℝ) (h : y^2 = 2 * x ∧ p = 1) : x = -p / 2 := by
  sorry

end parabola_directrix_eq_l166_166308


namespace common_difference_of_consecutive_multiples_l166_166252

/-- The sides of a rectangular prism are consecutive multiples of a certain number n. The base area is 450.
    Prove that the common difference between the consecutive multiples is 15. -/
theorem common_difference_of_consecutive_multiples (n d : ℕ) (h₁ : n * (n + d) = 450) : d = 15 :=
sorry

end common_difference_of_consecutive_multiples_l166_166252


namespace phung_more_than_chiu_l166_166234

theorem phung_more_than_chiu
  (C P H : ℕ)
  (h1 : C = 56)
  (h2 : H = P + 5)
  (h3 : C + P + H = 205) :
  P - C = 16 :=
by
  sorry

end phung_more_than_chiu_l166_166234


namespace total_miles_driven_l166_166783

-- Conditions
def miles_darius : ℕ := 679
def miles_julia : ℕ := 998

-- Proof statement
theorem total_miles_driven : miles_darius + miles_julia = 1677 := 
by
  -- placeholder for the proof steps
  sorry

end total_miles_driven_l166_166783


namespace find_length_QR_l166_166906

-- Define the provided conditions as Lean definitions
variables (Q P R : ℝ) (h_cos : Real.cos Q = 0.3) (QP : ℝ) (h_QP : QP = 15)
  
-- State the theorem we need to prove
theorem find_length_QR (QR : ℝ) (h_triangle : QP / QR = Real.cos Q) : QR = 50 := sorry

end find_length_QR_l166_166906


namespace probability_factor_lt_10_l166_166119

theorem probability_factor_lt_10 (n : ℕ) (h : n = 90) :
  (∃ factors_lt_10 : ℕ, ∃ total_factors : ℕ,
    factors_lt_10 = 7 ∧ total_factors = 12 ∧ (factors_lt_10 / total_factors : ℚ) = 7 / 12) :=
by sorry

end probability_factor_lt_10_l166_166119


namespace monotonicity_f_parity_f_max_value_f_min_value_f_l166_166841

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

-- Monotonicity Proof
theorem monotonicity_f : ∀ {x1 x2 : ℝ}, 2 < x1 → 2 < x2 → x1 < x2 → f x1 > f x2 :=
sorry

-- Parity Proof
theorem parity_f : ∀ x : ℝ, f (-x) = -f x :=
sorry

-- Maximum Value Proof
theorem max_value_f : ∀ {x : ℝ}, x = -6 → f x = -3/16 :=
sorry

-- Minimum Value Proof
theorem min_value_f : ∀ {x : ℝ}, x = -3 → f x = -3/5 :=
sorry

end monotonicity_f_parity_f_max_value_f_min_value_f_l166_166841


namespace solve_quadratic_eq_l166_166536

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2024 * x ↔ x = 0 ∨ x = 2024 :=
by sorry

end solve_quadratic_eq_l166_166536


namespace percent_profit_l166_166847

variable (C S : ℝ)

theorem percent_profit (h : 72 * C = 60 * S) : ((S - C) / C) * 100 = 20 := by
  sorry

end percent_profit_l166_166847


namespace round_robin_teams_l166_166788

theorem round_robin_teams (x : ℕ) (h : x ≠ 0) :
  (x * (x - 1)) / 2 = 15 → ∃ n : ℕ, x = n :=
by
  sorry

end round_robin_teams_l166_166788


namespace sean_total_apples_l166_166939

-- Define initial apples
def initial_apples : Nat := 9

-- Define the number of apples Susan gives each day
def apples_per_day : Nat := 8

-- Define the number of days Susan gives apples
def number_of_days : Nat := 5

-- Calculate total apples given by Susan
def total_apples_given : Nat := apples_per_day * number_of_days

-- Define the final total apples
def total_apples : Nat := initial_apples + total_apples_given

-- Prove the number of total apples is 49
theorem sean_total_apples : total_apples = 49 := by
  sorry

end sean_total_apples_l166_166939


namespace point_on_y_axis_l166_166938

theorem point_on_y_axis (m : ℝ) (M : ℝ × ℝ) (hM : M = (m + 1, m + 3)) (h_on_y_axis : M.1 = 0) : M = (0, 2) :=
by
  -- Proof omitted
  sorry

end point_on_y_axis_l166_166938


namespace geometric_seq_a7_l166_166186

-- Definitions for the geometric sequence and conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ}
axiom a1 : a 1 = 2
axiom a3 : a 3 = 4
axiom geom_seq : geometric_sequence a

-- Statement to prove
theorem geometric_seq_a7 : a 7 = 16 :=
by
  -- proof will be filled in here
  sorry

end geometric_seq_a7_l166_166186


namespace unique_z_value_l166_166952

theorem unique_z_value (x y u z : ℕ) (hx : 0 < x)
    (hy : 0 < y) (hu : 0 < u) (hz : 0 < z)
    (h1 : 3 + x + 21 = y + 25 + z)
    (h2 : 3 + x + 21 = 15 + u + 4)
    (h3 : y + 25 + z = 15 + u + 4)
    (h4 : 3 + y + 15 = x + 25 + u)
    (h5 : 3 + y + 15 = 21 + z + 4)
    (h6 : x + 25 + u = 21 + z + 4):
    z = 20 :=
by
    sorry

end unique_z_value_l166_166952


namespace brick_length_proof_l166_166354

-- Definitions based on conditions
def courtyard_length_m : ℝ := 18
def courtyard_width_m : ℝ := 16
def brick_width_cm : ℝ := 10
def total_bricks : ℝ := 14400

-- Conversion factors
def sqm_to_sqcm (area_sqm : ℝ) : ℝ := area_sqm * 10000
def courtyard_area_cm2 : ℝ := sqm_to_sqcm (courtyard_length_m * courtyard_width_m)

-- The proof statement
theorem brick_length_proof :
  (∀ (L : ℝ), courtyard_area_cm2 = total_bricks * (L * brick_width_cm)) → 
  (∃ (L : ℝ), L = 20) :=
by
  intro h
  sorry

end brick_length_proof_l166_166354


namespace total_quarters_l166_166637

-- Definitions from conditions
def initial_quarters : ℕ := 49
def quarters_given_by_dad : ℕ := 25

-- Theorem to prove the total quarters is 74
theorem total_quarters : initial_quarters + quarters_given_by_dad = 74 :=
by sorry

end total_quarters_l166_166637


namespace geom_seq_prod_of_terms_l166_166543

theorem geom_seq_prod_of_terms (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end geom_seq_prod_of_terms_l166_166543


namespace trapezoid_side_length_l166_166395

theorem trapezoid_side_length (s : ℝ) (A : ℝ) (x : ℝ) (y : ℝ) :
  s = 1 ∧ A = 1 ∧ y = 1/2 ∧ (1/2) * ((x + y) * y) = 1/4 → x = 1/2 :=
by
  intro h
  rcases h with ⟨hs, hA, hy, harea⟩
  sorry

end trapezoid_side_length_l166_166395


namespace cubic_yard_to_cubic_meter_and_liters_l166_166329

theorem cubic_yard_to_cubic_meter_and_liters :
  (1 : ℝ) * (0.9144 : ℝ)^3 = 0.764554 ∧ 0.764554 * 1000 = 764.554 :=
by
  sorry

end cubic_yard_to_cubic_meter_and_liters_l166_166329


namespace theater_rows_25_l166_166244

theorem theater_rows_25 (n : ℕ) (x : ℕ) (k : ℕ) (h : n = 1000) (h1 : k > 16) (h2 : (2 * x + k) * (k + 1) = 2000) : (k + 1) = 25 :=
by
  -- The proof goes here, which we omit for the problem statement.
  sorry

end theater_rows_25_l166_166244


namespace line_equation_l166_166947

noncomputable def P (A B C x y : ℝ) := A * x + B * y + C

theorem line_equation {A B C x₁ y₁ x₂ y₂ : ℝ} (h1 : P A B C x₁ y₁ = 0) (h2 : P A B C x₂ y₂ ≠ 0) :
    ∀ (x y : ℝ), P A B C x y - P A B C x₁ y₁ - P A B C x₂ y₂ = 0 ↔ P A B 0 x y = -P A B 0 x₂ y₂ := by
  sorry

end line_equation_l166_166947


namespace unique_solution_f_eq_x_l166_166248

theorem unique_solution_f_eq_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + y + f y) = 2 * y + f x ^ 2) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_eq_x_l166_166248


namespace average_distance_run_l166_166135

theorem average_distance_run :
  let mickey_lap := 250
  let johnny_lap := 300
  let alex_lap := 275
  let lea_lap := 280
  let johnny_times := 8
  let lea_times := 5
  let mickey_times := johnny_times / 2
  let alex_times := mickey_times + 1 + 2 * lea_times
  let total_distance := johnny_times * johnny_lap + mickey_times * mickey_lap + lea_times * lea_lap + alex_times * alex_lap
  let number_of_participants := 4
  let avg_distance := total_distance / number_of_participants
  avg_distance = 2231.25 := by
  sorry

end average_distance_run_l166_166135


namespace find_c_l166_166750

theorem find_c (a b c : ℝ) (h1 : ∃ a, ∃ b, ∃ c, 
              ∀ y, (∀ x, (x = a * (y-1)^2 + 4) ↔ (x = -2 → y = 3)) ∧
              (∀ y, x = a * y^2 + b * y + c)) : c = 1 / 2 :=
sorry

end find_c_l166_166750


namespace packs_of_sugar_l166_166710

theorem packs_of_sugar (cost_apples_per_kg cost_walnuts_per_kg cost_apples total : ℝ) (weight_apples weight_walnuts : ℝ) (less_sugar_by_1 : ℝ) (packs : ℕ) :
  cost_apples_per_kg = 2 →
  cost_walnuts_per_kg = 6 →
  cost_apples = weight_apples * cost_apples_per_kg →
  weight_apples = 5 →
  weight_walnuts = 0.5 →
  less_sugar_by_1 = 1 →
  total = 16 →
  packs = (total - (weight_apples * cost_apples_per_kg + weight_walnuts * cost_walnuts_per_kg)) / (cost_apples_per_kg - less_sugar_by_1) →
  packs = 3 :=
by
  sorry

end packs_of_sugar_l166_166710


namespace no_real_roots_poly_l166_166217

theorem no_real_roots_poly (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := 
  sorry

end no_real_roots_poly_l166_166217


namespace power_mod_condition_l166_166054

-- Defining the main problem conditions
theorem power_mod_condition (n: ℕ) : 
  (7^2 ≡ 1 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k+1) ≡ 7 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k) ≡ 1 [MOD 12]) →
  7^135 ≡ 7 [MOD 12] :=
by
  intros h1 h2 h3
  sorry

end power_mod_condition_l166_166054


namespace cab_driver_income_day3_l166_166787

theorem cab_driver_income_day3 :
  let income1 := 200
  let income2 := 150
  let income4 := 400
  let income5 := 500
  let avg_income := 400
  let total_income := avg_income * 5 
  total_income - (income1 + income2 + income4 + income5) = 750 := by
  sorry

end cab_driver_income_day3_l166_166787


namespace field_area_proof_l166_166971

-- Define the length of the uncovered side
def L : ℕ := 20

-- Define the total amount of fencing used for the other three sides
def total_fence : ℕ := 26

-- Define the field area function
def field_area (length width : ℕ) : ℕ := length * width

-- Statement: Prove that the area of the field is 60 square feet
theorem field_area_proof : 
  ∃ W : ℕ, (2 * W + L = total_fence) ∧ (field_area L W = 60) :=
  sorry

end field_area_proof_l166_166971


namespace min_dot_product_of_vectors_at_fixed_point_l166_166881

noncomputable def point := ℝ × ℝ

def on_ellipse (x y : ℝ) : Prop := 
  (x^2) / 36 + (y^2) / 9 = 1

def dot_product (p q : point) : ℝ := 
  p.1 * q.1 + p.2 * q.2

def vector_magnitude_squared (p : point) : ℝ := 
  p.1^2 + p.2^2

def KM (M : point) : point := 
  (M.1 - 2, M.2)

def NM (N M : point) : point := 
  (M.1 - N.1, M.2 - N.2)

def fixed_point_K : point := 
  (2, 0)

theorem min_dot_product_of_vectors_at_fixed_point (M N : point) 
  (hM_on_ellipse : on_ellipse M.1 M.2) 
  (hN_on_ellipse : on_ellipse N.1 N.2) 
  (h_orthogonal : dot_product (KM M) (KM N) = 0) : 
  ∃ (α : ℝ), dot_product (KM M) (NM N M) = 23 / 3 :=
sorry

end min_dot_product_of_vectors_at_fixed_point_l166_166881


namespace min_value_of_M_l166_166677

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem min_value_of_M (M : ℝ) (hM : M = Real.sqrt 2) :
  ∀ (a b c : ℝ), a > M → b > M → c > M → a^2 + b^2 = c^2 → 
  (f a) + (f b) > f c ∧ (f a) + (f c) > f b ∧ (f b) + (f c) > f a :=
by
  sorry

end min_value_of_M_l166_166677


namespace total_time_for_seven_flights_l166_166328

theorem total_time_for_seven_flights :
  let a := 15
  let d := 8
  let n := 7
  let l := a + (n - 1) * d
  let S_n := n * (a + l) / 2
  S_n = 273 :=
by
  sorry

end total_time_for_seven_flights_l166_166328


namespace pet_store_dogs_count_l166_166229

def initial_dogs : ℕ := 2
def sunday_received_dogs : ℕ := 5
def sunday_sold_dogs : ℕ := 2
def monday_received_dogs : ℕ := 3
def monday_returned_dogs : ℕ := 1
def tuesday_received_dogs : ℕ := 4
def tuesday_sold_dogs : ℕ := 3

theorem pet_store_dogs_count :
  initial_dogs 
  + sunday_received_dogs - sunday_sold_dogs
  + monday_received_dogs + monday_returned_dogs
  + tuesday_received_dogs - tuesday_sold_dogs = 10 := 
sorry

end pet_store_dogs_count_l166_166229


namespace total_candy_count_l166_166235

def numberOfRedCandies : ℕ := 145
def numberOfBlueCandies : ℕ := 3264
def totalNumberOfCandies : ℕ := numberOfRedCandies + numberOfBlueCandies

theorem total_candy_count :
  totalNumberOfCandies = 3409 :=
by
  unfold totalNumberOfCandies
  unfold numberOfRedCandies
  unfold numberOfBlueCandies
  sorry

end total_candy_count_l166_166235


namespace vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l166_166144

/-- The test consists of 30 questions, each with two possible answers (one correct and one incorrect). 
    Vitya can proceed in such a way that he can guarantee to know all the correct answers no later than:
    (a) after the 29th attempt (and answer all questions correctly on the 30th attempt)
    (b) after the 24th attempt (and answer all questions correctly on the 25th attempt)
    - Vitya initially does not know any of the answers.
    - The test is always the same.
-/
def vitya_test (k : Nat) : Prop :=
  k = 30 ∧ (∀ (attempts : Fin 30 → Bool), attempts 30 = attempts 29 ∧ attempts 30)

theorem vitya_knows_answers_29_attempts :
  vitya_test 30 :=
by 
  sorry

theorem vitya_knows_answers_24_attempts :
  vitya_test 25 :=
by 
  sorry

end vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l166_166144


namespace min_value_f_l166_166100

noncomputable def f (x : Fin 5 → ℝ) : ℝ :=
  (x 0 + x 2) / (x 4 + 2 * x 1 + 3 * x 3) +
  (x 1 + x 3) / (x 0 + 2 * x 2 + 3 * x 4) +
  (x 2 + x 4) / (x 1 + 2 * x 3 + 3 * x 0) +
  (x 3 + x 0) / (x 2 + 2 * x 4 + 3 * x 1) +
  (x 4 + x 1) / (x 3 + 2 * x 0 + 3 * x 2)

def min_f (x : Fin 5 → ℝ) : Prop :=
  (∀ i, 0 < x i) → f x = 5 / 3

theorem min_value_f : ∀ x : Fin 5 → ℝ, min_f x :=
by
  intros
  sorry

end min_value_f_l166_166100


namespace calculate_expression_l166_166000

theorem calculate_expression :
  3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 :=
by
  sorry

end calculate_expression_l166_166000


namespace nonnegative_integer_solution_count_l166_166283

theorem nonnegative_integer_solution_count :
  ∃ n : ℕ, (∀ x : ℕ, x^2 + 6 * x = 0 → x = 0) ∧ n = 1 :=
by
  sorry

end nonnegative_integer_solution_count_l166_166283


namespace minimize_quadratic_expression_l166_166771

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end minimize_quadratic_expression_l166_166771


namespace triangle_BX_in_terms_of_sides_l166_166133

-- Define the triangle with angles and points
variables {A B C : ℝ}
variables {AB AC BC : ℝ}
variables (X Y : ℝ) (AZ : ℝ)

-- Add conditions as assumptions
variables (angle_A_bisector : 2 * A = (B + C)) -- AZ is the angle bisector of angle A
variables (angle_B_lt_C : B < C) -- angle B < angle C
variables (point_XY : X / AB = Y / AC ∧ X = Y) -- BX = CY and angles BZX = CZY

-- Define the statement to be proved
theorem triangle_BX_in_terms_of_sides :
    BX = CY →
    (AZ < 1 ∧ AZ > 0) →
    A + B + C = π → 
    BX = (BC * BC) / (AB + AC) :=
sorry

end triangle_BX_in_terms_of_sides_l166_166133


namespace how_many_more_red_balls_l166_166697

def r_packs : ℕ := 12
def y_packs : ℕ := 9
def r_balls_per_pack : ℕ := 24
def y_balls_per_pack : ℕ := 20

theorem how_many_more_red_balls :
  (r_packs * r_balls_per_pack) - (y_packs * y_balls_per_pack) = 108 :=
by
  sorry

end how_many_more_red_balls_l166_166697


namespace find_a_l166_166957

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 1 - x else a * x

theorem find_a (a : ℝ) : f (-1) a = f 1 a → a = 2 := by
  intro h
  sorry

end find_a_l166_166957


namespace wood_cost_l166_166663

theorem wood_cost (C : ℝ) (h1 : 20 * 15 = 300) (h2 : 300 - C = 200) : C = 100 :=
by
  -- The proof is to be filled here, but it is currently skipped with 'sorry'.
  sorry

end wood_cost_l166_166663


namespace neither_outstanding_nor_young_pioneers_is_15_l166_166857

-- Define the conditions
def total_students : ℕ := 87
def outstanding_students : ℕ := 58
def young_pioneers : ℕ := 63
def both_outstanding_and_young_pioneers : ℕ := 49

-- Define the function to calculate the number of students who are neither
def neither_outstanding_nor_young_pioneers
: ℕ :=
total_students - (outstanding_students - both_outstanding_and_young_pioneers) - (young_pioneers - both_outstanding_and_young_pioneers) - both_outstanding_and_young_pioneers

-- The theorem to prove
theorem neither_outstanding_nor_young_pioneers_is_15
: neither_outstanding_nor_young_pioneers = 15 :=
by
  sorry

end neither_outstanding_nor_young_pioneers_is_15_l166_166857


namespace solution_to_problem_l166_166887

-- Definitions of conditions
def condition_1 (x : ℝ) : Prop := 2 * x - 6 ≠ 0
def condition_2 (x : ℝ) : Prop := 5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10

-- Definition of solution set
def solution_set (x : ℝ) : Prop := 3 < x ∧ x < 60 / 19

-- The theorem to be proven
theorem solution_to_problem (x : ℝ) (h1 : condition_1 x) : condition_2 x ↔ solution_set x :=
by sorry

end solution_to_problem_l166_166887


namespace matrix_multiplication_problem_l166_166407

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

theorem matrix_multiplication_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = ![![5, 2], ![-2, 4]]) :
  B * A = ![![5, 2], ![-2, 4]] :=
sorry

end matrix_multiplication_problem_l166_166407


namespace minimum_value_l166_166025

variables (a b c d : ℝ)
-- Conditions
def condition1 := (b - 2 * a^2 + 3 * Real.log a)^2 = 0
def condition2 := (c - d - 3)^2 = 0

-- Theorem stating the goal
theorem minimum_value (h1 : condition1 a b) (h2 : condition2 c d) : 
  (a - c)^2 + (b - d)^2 = 8 :=
sorry

end minimum_value_l166_166025


namespace most_appropriate_survey_is_D_l166_166378

-- Define the various scenarios as Lean definitions
def survey_A := "Testing whether a certain brand of fresh milk meets food hygiene standards, using a census method."
def survey_B := "Security check before taking the subway, using a sampling survey method."
def survey_C := "Understanding the sleep time of middle school students in Jiangsu Province, using a census method."
def survey_D := "Understanding the way Nanjing residents commemorate the Qingming Festival, using a sampling survey method."

-- Define the type for specifying which survey method is the most appropriate
def appropriate_survey (survey : String) : Prop := 
  survey = survey_D

-- The theorem statement proving that the most appropriate survey is D
theorem most_appropriate_survey_is_D : appropriate_survey survey_D :=
by sorry

end most_appropriate_survey_is_D_l166_166378


namespace fifth_grade_total_students_l166_166816

-- Define the conditions given in the problem
def total_boys : ℕ := 350
def total_playing_soccer : ℕ := 250
def percentage_boys_playing_soccer : ℝ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- Define the total number of students
def total_students : ℕ := 500

-- Prove that the total number of students is 500
theorem fifth_grade_total_students 
  (H1 : total_boys = 350) 
  (H2 : total_playing_soccer = 250) 
  (H3 : percentage_boys_playing_soccer = 0.86) 
  (H4 : girls_not_playing_soccer = 115) :
  total_students = 500 := 
sorry

end fifth_grade_total_students_l166_166816


namespace probability_at_least_one_multiple_of_4_l166_166760

/-- Definition for the total number of integers in the range -/
def total_numbers : ℕ := 60

/-- Definition for the number of multiples of 4 within the range -/
def multiples_of_4 : ℕ := 15

/-- Probability that a single number chosen is not a multiple of 4 -/
def prob_not_multiple_of_4 : ℚ := (total_numbers - multiples_of_4) / total_numbers

/-- Probability that none of the three chosen numbers is a multiple of 4 -/
def prob_none_multiple_of_4 : ℚ := prob_not_multiple_of_4 ^ 3

/-- Given condition that Linda choose three times -/
axiom linda_chooses_thrice (x y z : ℕ) : 
1 ≤ x ∧ x ≤ 60 ∧ 
1 ≤ y ∧ y ≤ 60 ∧ 
1 ≤ z ∧ z ≤ 60

/-- Theorem stating the desired probability -/
theorem probability_at_least_one_multiple_of_4 : 
1 - prob_none_multiple_of_4 = 37 / 64 := by
  sorry

end probability_at_least_one_multiple_of_4_l166_166760


namespace area_of_rectangle_l166_166702

def length : ℝ := 0.5
def width : ℝ := 0.24

theorem area_of_rectangle :
  length * width = 0.12 :=
by
  sorry

end area_of_rectangle_l166_166702


namespace xyz_sum_sqrt14_l166_166476

theorem xyz_sum_sqrt14 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 1) (h2 : x + 2 * y + 3 * z = Real.sqrt 14) :
  x + y + z = (3 * Real.sqrt 14) / 7 :=
sorry

end xyz_sum_sqrt14_l166_166476


namespace ellipse_foci_coordinates_l166_166196

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, x^2 / 9 + y^2 / 5 = 1 → (x = 2 ∧ y = 0) ∨ (x = -2 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l166_166196


namespace length_of_first_platform_is_140_l166_166038

-- Definitions based on problem conditions
def train_length : ℝ := 190
def time_first_platform : ℝ := 15
def time_second_platform : ℝ := 20
def length_second_platform : ℝ := 250

-- Definition for the length of the first platform (what we're proving)
def length_first_platform (L : ℝ) : Prop :=
  (time_first_platform * (train_length + L) = time_second_platform * (train_length + length_second_platform))

-- Theorem: The length of the first platform is 140 meters
theorem length_of_first_platform_is_140 : length_first_platform 140 :=
  by sorry

end length_of_first_platform_is_140_l166_166038


namespace sequence_a2017_l166_166287

theorem sequence_a2017 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2017 = 2 :=
sorry

end sequence_a2017_l166_166287


namespace find_w_l166_166629

variable (p j t : ℝ) (w : ℝ)

-- Definitions based on conditions
def j_less_than_p : Prop := j = 0.75 * p
def j_less_than_t : Prop := j = 0.80 * t
def t_less_than_p : Prop := t = p * (1 - w / 100)

-- Objective: Prove that given these conditions, w = 6.25
theorem find_w (h1 : j_less_than_p p j) (h2 : j_less_than_t j t) (h3 : t_less_than_p t p w) : 
  w = 6.25 := 
by 
  sorry

end find_w_l166_166629


namespace lottery_sample_representativeness_l166_166081

theorem lottery_sample_representativeness (A B C D : Prop) :
  B :=
by
  sorry

end lottery_sample_representativeness_l166_166081


namespace tub_drain_time_l166_166650

theorem tub_drain_time (time_for_five_sevenths : ℝ)
  (time_for_five_sevenths_eq_four : time_for_five_sevenths = 4) :
  let rate := time_for_five_sevenths / (5 / 7)
  let time_for_two_sevenths := 2 * rate
  time_for_two_sevenths = 11.2 := by
  -- Definitions and initial conditions
  sorry

end tub_drain_time_l166_166650


namespace pq_combined_work_rate_10_days_l166_166529

/-- Conditions: 
1. wr_p = wr_qr, where wr_qr is the combined work rate of q and r
2. wr_r allows completing the work in 30 days
3. wr_q allows completing the work in 30 days

We need to prove that the combined work rate of p and q allows them to complete the work in 10 days.
-/
theorem pq_combined_work_rate_10_days
  (wr_p wr_q wr_r wr_qr : ℝ)
  (h1 : wr_p = wr_qr)
  (h2 : wr_r = 1/30)
  (h3 : wr_q = 1/30) :
  wr_p + wr_q = 1/10 := by
  sorry

end pq_combined_work_rate_10_days_l166_166529


namespace problem_1_exists_a_problem_2_values_of_a_l166_166098

open Set

-- Definitions for sets A, B, C
def A (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + 4 * a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- Lean statements for the two problems
theorem problem_1_exists_a : ∃ a : ℝ, A a ∩ B = A a ∪ B ∧ a = 1/2 := by
  sorry

theorem problem_2_values_of_a (a : ℝ) : 
  (A a ∩ B ≠ ∅ ∧ A a ∩ C = ∅) → 
  (A a = {-1} → a = -1) ∧ (∀ x, A a = {-1, x} → x ≠ 2 → False) := 
  by sorry

end problem_1_exists_a_problem_2_values_of_a_l166_166098


namespace geometric_series_sum_l166_166527

theorem geometric_series_sum (n : ℕ) : 
  let a₁ := 2
  let q := 2
  let S_n := a₁ * (1 - q^n) / (1 - q)
  S_n = 2 - 2^(n + 1) := 
by
  sorry

end geometric_series_sum_l166_166527


namespace number_of_integer_segments_l166_166338

theorem number_of_integer_segments (DE EF : ℝ) (H1 : DE = 24) (H2 : EF = 25) : 
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_integer_segments_l166_166338


namespace final_computation_l166_166070

noncomputable def N := (15 ^ 10 / 15 ^ 9) ^ 3 * 5 ^ 3

theorem final_computation : (N / 3 ^ 3) = 15625 := 
by 
  sorry

end final_computation_l166_166070


namespace joseph_investment_after_two_years_l166_166201

noncomputable def initial_investment : ℝ := 1000
noncomputable def monthly_addition : ℝ := 100
noncomputable def yearly_interest_rate : ℝ := 0.10
noncomputable def time_in_years : ℕ := 2

theorem joseph_investment_after_two_years :
  let first_year_total := initial_investment + 12 * monthly_addition
  let first_year_interest := first_year_total * yearly_interest_rate
  let end_of_first_year_total := first_year_total + first_year_interest
  let second_year_total := end_of_first_year_total + 12 * monthly_addition
  let second_year_interest := second_year_total * yearly_interest_rate
  let end_of_second_year_total := second_year_total + second_year_interest
  end_of_second_year_total = 3982 := 
by
  sorry

end joseph_investment_after_two_years_l166_166201


namespace cost_of_balls_max_basketball_count_l166_166187

-- Define the prices of basketball and soccer ball
variables (x y : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := 2 * x + 3 * y = 310
def condition2 : Prop := 5 * x + 2 * y = 500

-- Proving the cost of each basketball and soccer ball
theorem cost_of_balls (h1 : condition1 x y) (h2 : condition2 x y) : x = 80 ∧ y = 50 :=
sorry

-- Define the total number of balls and the inequality constraint
variable (m : ℕ)
def total_balls_condition : Prop := m + (60 - m) = 60
def cost_constraint : Prop := 80 * m + 50 * (60 - m) ≤ 4000

-- Proving the maximum number of basketballs
theorem max_basketball_count (hc : cost_constraint m) (ht : total_balls_condition m) : m ≤ 33 :=
sorry

end cost_of_balls_max_basketball_count_l166_166187


namespace students_not_enrolled_l166_166949

theorem students_not_enrolled (total_students : ℕ) (students_french : ℕ) (students_german : ℕ) (students_both : ℕ)
  (h1 : total_students = 94)
  (h2 : students_french = 41)
  (h3 : students_german = 22)
  (h4 : students_both = 9) : 
  ∃ (students_neither : ℕ), students_neither = 40 :=
by
  -- We would show the calculation here in a real proof 
  sorry

end students_not_enrolled_l166_166949


namespace lcm_12_18_l166_166348

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l166_166348


namespace college_girls_count_l166_166176

theorem college_girls_count 
  (B G : ℕ)
  (h1 : B / G = 8 / 5)
  (h2 : B + G = 455) : 
  G = 175 := 
sorry

end college_girls_count_l166_166176


namespace george_total_blocks_l166_166314

-- Definitions (conditions).
def large_boxes : ℕ := 5
def small_boxes_per_large_box : ℕ := 8
def blocks_per_small_box : ℕ := 9
def individual_blocks : ℕ := 6

-- Mathematical proof problem statement.
theorem george_total_blocks :
  (large_boxes * small_boxes_per_large_box * blocks_per_small_box + individual_blocks) = 366 :=
by
  -- Placeholder for proof.
  sorry

end george_total_blocks_l166_166314


namespace mrs_heine_dogs_treats_l166_166582

theorem mrs_heine_dogs_treats (heart_biscuits_per_dog puppy_boots_per_dog total_items : ℕ)
  (h_biscuits : heart_biscuits_per_dog = 5)
  (h_boots : puppy_boots_per_dog = 1)
  (total : total_items = 12) :
  (total_items / (heart_biscuits_per_dog + puppy_boots_per_dog)) = 2 :=
by
  sorry

end mrs_heine_dogs_treats_l166_166582


namespace triangle_properties_equivalence_l166_166574

-- Define the given properties for the two triangles
variables {A B C A' B' C' : Type}

-- Triangle side lengths and properties
def triangles_equal (b b' c c' : ℝ) : Prop :=
  (b = b') ∧ (c = c')

def equivalent_side_lengths (a a' b b' c c' : ℝ) : Prop :=
  a = a'

def equivalent_medians (ma ma' b b' c c' a a' : ℝ) : Prop :=
  ma = ma'

def equivalent_altitudes (ha ha' Δ Δ' a a' : ℝ) : Prop :=
  ha = ha'

def equivalent_angle_bisectors (ta ta' b b' c c' a a' : ℝ) : Prop :=
  ta = ta'

def equivalent_circumradii (R R' a a' b b' c c' : ℝ) : Prop :=
  R = R'

def equivalent_areas (Δ Δ' b b' c c' A A' : ℝ) : Prop :=
  Δ = Δ'

-- Main theorem statement
theorem triangle_properties_equivalence
  (b b' c c' a a' ma ma' ha ha' ta ta' R R' Δ Δ' : ℝ)
  (A A' : ℝ)
  (eq_b : b = b')
  (eq_c : c = c') :
  equivalent_side_lengths a a' b b' c c' ∧ 
  equivalent_medians ma ma' b b' c c' a a' ∧ 
  equivalent_altitudes ha ha' Δ Δ' a a' ∧ 
  equivalent_angle_bisectors ta ta' b b' c c' a a' ∧ 
  equivalent_circumradii R R' a a' b b' c c' ∧ 
  equivalent_areas Δ Δ' b b' c c' A A'
:= by
  sorry

end triangle_properties_equivalence_l166_166574


namespace system_of_equations_correct_l166_166396

def question_statement (x y : ℕ) : Prop :=
  x + y = 12 ∧ 6 * x = 3 * 4 * y

theorem system_of_equations_correct
  (x y : ℕ)
  (h1 : x + y = 12)
  (h2 : 6 * x = 3 * 4 * y)
: question_statement x y :=
by
  unfold question_statement
  exact ⟨h1, h2⟩

end system_of_equations_correct_l166_166396


namespace smallest_x_plus_y_l166_166250

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x^2 - 29 * y^2 = 1) : x + y = 11621 := 
sorry

end smallest_x_plus_y_l166_166250


namespace max_score_per_student_l166_166493

theorem max_score_per_student (score_tests : ℕ → ℕ) (avg_score_tests_lt_8 : ℕ) (combined_score_two_tests : ℕ) : (∀ i, 1 ≤ i ∧ i ≤ 8 → score_tests i ≤ 100) ∧ avg_score_tests_lt_8 = 70 ∧ combined_score_two_tests = 290 →
  ∃ max_score : ℕ, max_score = 145 := 
by
  sorry

end max_score_per_student_l166_166493


namespace arithmetic_seq_max_S_l166_166496

theorem arithmetic_seq_max_S {S : ℕ → ℝ} (h1 : S 2023 > 0) (h2 : S 2024 < 0) : S 1012 > S 1013 :=
sorry

end arithmetic_seq_max_S_l166_166496


namespace triangle_angles_l166_166928

variable (a b c t : ℝ)

def angle_alpha : ℝ := 43

def area_condition (α β : ℝ) : Prop :=
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β)

theorem triangle_angles (α β γ : ℝ) (hα : α = angle_alpha) (h_area : area_condition a b t α β) :
  α = 43 ∧ β = 17 ∧ γ = 120 := sorry

end triangle_angles_l166_166928


namespace cooking_ways_l166_166479

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem cooking_ways : comb 5 2 = 10 :=
  by
  sorry

end cooking_ways_l166_166479


namespace prob_of_target_hit_l166_166885

noncomputable def probability_target_hit : ℚ :=
  let pA := (1 : ℚ) / 2
  let pB := (1 : ℚ) / 3
  let pC := (1 : ℚ) / 4
  let pA' := 1 - pA
  let pB' := 1 - pB
  let pC' := 1 - pC
  let pNoneHit := pA' * pB' * pC'
  1 - pNoneHit

-- Statement to be proved
theorem prob_of_target_hit : probability_target_hit = 3 / 4 :=
  sorry

end prob_of_target_hit_l166_166885


namespace uv_divisible_by_3_l166_166158

theorem uv_divisible_by_3
  {u v : ℤ}
  (h : 9 ∣ (u^2 + u * v + v^2)) :
  3 ∣ u ∧ 3 ∣ v :=
sorry

end uv_divisible_by_3_l166_166158


namespace parallelogram_area_formula_l166_166966

noncomputable def parallelogram_area (ha hb : ℝ) (γ : ℝ) : ℝ := 
  ha * hb / Real.sin γ

theorem parallelogram_area_formula (ha hb γ : ℝ) (a b : ℝ) 
  (h₁ : Real.sin γ ≠ 0) :
  (parallelogram_area ha hb γ = ha * hb / Real.sin γ) := by
  sorry

end parallelogram_area_formula_l166_166966


namespace chameleons_changed_color_l166_166145

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l166_166145


namespace range_of_a_l166_166944

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7
  else 9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8/7 :=
by
  intros h
  -- Detailed proof would go here
  sorry

end range_of_a_l166_166944


namespace num_distinct_triangles_in_octahedron_l166_166504

theorem num_distinct_triangles_in_octahedron : ∃ n : ℕ, n = 48 ∧ ∀ (V : Finset (Fin 8)), 
  V.card = 3 → (∀ {a b c : Fin 8}, a ∈ V ∧ b ∈ V ∧ c ∈ V → 
  ¬((a = 0 ∧ b = 1 ∧ c = 2) ∨ (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 6 ∧ b = 7 ∧ c = 8)
  ∨ (a = 7 ∧ b = 0 ∧ c = 1) ∨ (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 6 ∧ c = 7))) :=
by sorry

end num_distinct_triangles_in_octahedron_l166_166504


namespace not_right_triangle_l166_166382

/-- In a triangle ABC, with angles A, B, C, the condition A = B = 2 * C does not form a right-angled triangle. -/
theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) (h3 : A + B + C = 180) : 
    ¬(A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end not_right_triangle_l166_166382


namespace least_people_to_complete_job_on_time_l166_166585

theorem least_people_to_complete_job_on_time
  (total_duration : ℕ)
  (initial_days : ℕ)
  (initial_people : ℕ)
  (initial_work_done : ℚ)
  (efficiency_multiplier : ℚ)
  (remaining_work_fraction : ℚ)
  (remaining_days : ℕ)
  (resulting_people : ℕ)
  (work_rate_doubled : ℕ → ℚ → ℚ)
  (final_resulting_people : ℚ)
  : initial_work_done = 1/4 →
    efficiency_multiplier = 2 →
    remaining_work_fraction = 3/4 →
    total_duration = 40 →
    initial_days = 10 →
    initial_people = 12 →
    remaining_days = 20 →
    work_rate_doubled 12 2 = 24 →
    final_resulting_people = (1/2) →
    resulting_people = 6 :=
sorry

end least_people_to_complete_job_on_time_l166_166585


namespace dresser_clothing_capacity_l166_166004

theorem dresser_clothing_capacity (pieces_per_drawer : ℕ) (number_of_drawers : ℕ) (total_pieces : ℕ) 
  (h1 : pieces_per_drawer = 5)
  (h2 : number_of_drawers = 8)
  (h3 : total_pieces = 40) :
  pieces_per_drawer * number_of_drawers = total_pieces :=
by {
  sorry
}

end dresser_clothing_capacity_l166_166004


namespace donation_amount_l166_166392

theorem donation_amount 
  (total_needed : ℕ) (bronze_amount : ℕ) (silver_amount : ℕ) (raised_so_far : ℕ)
  (bronze_families : ℕ) (silver_families : ℕ) (other_family_donation : ℕ)
  (final_push_needed : ℕ) 
  (h1 : total_needed = 750) 
  (h2 : bronze_amount = 25)
  (h3 : silver_amount = 50)
  (h4 : bronze_families = 10)
  (h5 : silver_families = 7)
  (h6 : raised_so_far = 600)
  (h7 : final_push_needed = 50)
  (h8 : raised_so_far = bronze_families * bronze_amount + silver_families * silver_amount)
  (h9 : total_needed - raised_so_far - other_family_donation = final_push_needed) : 
  other_family_donation = 100 :=
by
  sorry

end donation_amount_l166_166392


namespace remainder_T_2015_mod_10_l166_166968

-- Define the number of sequences with no more than two consecutive identical letters
noncomputable def T : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 6
| n + 1 => (T n + T (n - 1) + T (n - 2) + T (n - 3))  -- hypothetically following initial conditions pattern

theorem remainder_T_2015_mod_10 : T 2015 % 10 = 6 :=
by 
  sorry

end remainder_T_2015_mod_10_l166_166968


namespace nearest_integer_pow_l166_166669

noncomputable def nearest_integer_to_power : ℤ := 
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_pow : nearest_integer_to_power = 7414 := 
  by
    unfold nearest_integer_to_power
    sorry -- Proof skipped

end nearest_integer_pow_l166_166669


namespace unique_handshakes_count_l166_166934

-- Definitions from the conditions
def teams : Nat := 4
def players_per_team : Nat := 2
def total_players : Nat := teams * players_per_team

def handshakes_per_player : Nat := total_players - players_per_team

-- The Lean statement to prove the total number of unique handshakes
theorem unique_handshakes_count : (total_players * handshakes_per_player) / 2 = 24 := 
by
  -- Proof steps would go here
  sorry

end unique_handshakes_count_l166_166934


namespace sequence_term_306_l166_166419

theorem sequence_term_306 (a1 a2 : ℤ) (r : ℤ) (n : ℕ) (h1 : a1 = 7) (h2 : a2 = -7) (h3 : r = -1) (h4 : a2 = r * a1) : 
  ∃ a306 : ℤ, a306 = -7 ∧ a306 = a1 * r^305 :=
by
  use -7
  sorry

end sequence_term_306_l166_166419


namespace part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l166_166604

-- Part (1)
theorem part1_coordinates_on_x_axis (a : ℝ) (h : a + 5 = 0) : (2*a - 2, a + 5) = (-12, 0) :=
by sorry

-- Part (2)
theorem part2_coordinates_parallel_y_axis (a : ℝ) (h : 2*a - 2 = 4) : (2*a - 2, a + 5) = (4, 8) :=
by sorry

-- Part (3)
theorem part3_distances_equal_second_quadrant (a : ℝ) 
  (h1 : 2*a-2 < 0) (h2 : a+5 > 0) (h3 : abs (2*a - 2) = abs (a + 5)) : a^(2022 : ℕ) + 2022 = 2023 :=
by sorry

end part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l166_166604


namespace complex_exp_form_pow_four_l166_166745

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l166_166745


namespace solution_is_13_l166_166288

def marbles_in_jars : Prop :=
  let jar1 := (5, 3, 1)  -- (red, blue, green)
  let jar2 := (1, 5, 3)  -- (red, blue, green)
  let jar3 := (3, 1, 5)  -- (red, blue, green)
  let total_ways := 125 + 15 + 15 + 3 + 27 + 15
  let favorable_ways := 125
  let probability := favorable_ways / total_ways
  let simplified_probability := 5 / 8
  let m := 5
  let n := 8
  m + n = 13

theorem solution_is_13 : marbles_in_jars :=
by {
  sorry
}

end solution_is_13_l166_166288


namespace carmen_candle_usage_l166_166418

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end carmen_candle_usage_l166_166418


namespace correct_option_is_A_l166_166241

def a (n : ℕ) : ℤ :=
  match n with
  | 1 => -3
  | 2 => 7
  | _ => 0  -- This is just a placeholder for other values

def optionA (n : ℕ) : ℤ := (-1)^n * (4*n - 1)
def optionB (n : ℕ) : ℤ := (-1)^n * (4*n + 1)
def optionC (n : ℕ) : ℤ := 4*n - 7
def optionD (n : ℕ) : ℤ := (-1)^(n + 1) * (4*n - 1)

theorem correct_option_is_A :
  (a 1 = -3) ∧ (a 2 = 7) ∧
  (optionA 1 = -3 ∧ optionA 2 = 7) ∧
  ¬(optionB 1 = -3 ∧ optionB 2 = 7) ∧
  ¬(optionC 1 = -3 ∧ optionC 2 = 7) ∧
  ¬(optionD 1 = -3 ∧ optionD 2 = 7) :=
by
  sorry

end correct_option_is_A_l166_166241


namespace profit_increase_l166_166488

theorem profit_increase (x y : ℝ) (a : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (profit_eq : y - x = x * (a / 100))
  (new_profit_eq : y - 0.95 * x = 0.95 * x * (a / 100) + 0.95 * x * (15 / 100)) :
  a = 185 :=
by
  sorry

end profit_increase_l166_166488


namespace system_of_inequalities_solution_l166_166094

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l166_166094


namespace max_intersection_points_circles_lines_l166_166204

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l166_166204


namespace total_water_filled_jars_l166_166711

theorem total_water_filled_jars :
  ∃ x : ℕ, 
    16 * (1/4) + 12 * (1/2) + 8 * 1 + 4 * 2 + x * 3 = 56 ∧
    16 + 12 + 8 + 4 + x = 50 :=
by
  sorry

end total_water_filled_jars_l166_166711


namespace gdp_scientific_notation_l166_166150

theorem gdp_scientific_notation :
  (121 * 10^12 : ℝ) = 1.21 * 10^14 := by
  sorry

end gdp_scientific_notation_l166_166150


namespace machine_x_widgets_per_hour_l166_166433

-- Definitions of the variables and conditions
variable (Wx Wy Tx Ty: ℝ)
variable (h1: Tx = Ty + 60)
variable (h2: Wy = 1.20 * Wx)
variable (h3: Wx * Tx = 1080)
variable (h4: Wy * Ty = 1080)

-- Statement of the problem to prove
theorem machine_x_widgets_per_hour : Wx = 3 := by
  sorry

end machine_x_widgets_per_hour_l166_166433


namespace sphere_radius_l166_166826

theorem sphere_radius (r_A r_B : ℝ) (h₁ : r_A = 40) (h₂ : (4 * π * r_A^2) / (4 * π * r_B^2) = 16) : r_B = 20 :=
  sorry

end sphere_radius_l166_166826


namespace vector_equivalence_l166_166737

-- Define the vectors a and b
noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)

-- Define the operation 3a - b
noncomputable def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
  (3 * a.1 - b.1, 3 * a.2 - b.2)

-- State that for given vectors a and b, the result of the operation equals (4, 2)
theorem vector_equivalence : vector_operation vector_a vector_b = (4, 2) :=
  sorry

end vector_equivalence_l166_166737


namespace odd_function_value_l166_166812

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

-- Prove that f(-1/2) = -1/2 given the conditions
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = x) →
  f (-1/2) = -1/2 :=
by
  sorry

end odd_function_value_l166_166812


namespace compute_a_l166_166324

theorem compute_a 
  (a b : ℚ) 
  (h : ∃ (x : ℝ), x^3 + (a : ℝ) * x^2 + (b : ℝ) * x - 37 = 0 ∧ x = 2 - 3 * Real.sqrt 3) : 
  a = -55 / 23 :=
by 
  sorry

end compute_a_l166_166324


namespace hyperbola_smaller_focus_l166_166191

noncomputable def smaller_focus_coordinates : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 3
  let b := 7
  let c := Real.sqrt (a^2 + b^2)
  (h - c, k)

theorem hyperbola_smaller_focus :
  (smaller_focus_coordinates = (Real.sqrt 58 - 2.62, 20)) :=
by
  sorry

end hyperbola_smaller_focus_l166_166191


namespace jill_spent_50_percent_on_clothing_l166_166572

theorem jill_spent_50_percent_on_clothing (
  T : ℝ) (hT : T ≠ 0)
  (h : 0.05 * T * C + 0.10 * 0.30 * T = 0.055 * T):
  C = 0.5 :=
by
  sorry

end jill_spent_50_percent_on_clothing_l166_166572


namespace logical_impossibility_of_thoughts_l166_166469

variable (K Q : Prop)

/-- Assume that King and Queen are sane (sane is represented by them not believing they're insane) -/
def sane (p : Prop) : Prop :=
  ¬(p = true)

/-- Define the nested thoughts -/
def KingThinksQueenThinksKingThinksQueenOutOfMind (K Q : Prop) :=
  K ∧ Q ∧ K ∧ Q = ¬sane Q

/-- The main proposition -/
theorem logical_impossibility_of_thoughts (hK : sane K) (hQ : sane Q) : 
  ¬KingThinksQueenThinksKingThinksQueenOutOfMind K Q :=
by sorry

end logical_impossibility_of_thoughts_l166_166469


namespace reciprocal_relationship_l166_166440

theorem reciprocal_relationship (a b : ℝ) (h₁ : a = 2 - Real.sqrt 3) (h₂ : b = Real.sqrt 3 + 2) : 
  a * b = 1 :=
by
  rw [h₁, h₂]
  sorry

end reciprocal_relationship_l166_166440


namespace geometric_series_S6_value_l166_166342

theorem geometric_series_S6_value (S : ℕ → ℝ) (S3 : S 3 = 3) (S9_minus_S6 : S 9 - S 6 = 12) : 
  S 6 = 9 :=
by
  sorry

end geometric_series_S6_value_l166_166342


namespace more_cats_than_spinsters_l166_166807

theorem more_cats_than_spinsters :
  ∀ (S C : ℕ), (S = 18) → (2 * C = 9 * S) → (C - S = 63) :=
by
  intros S C hS hRatio
  sorry

end more_cats_than_spinsters_l166_166807


namespace calculate_inverse_y3_minus_y_l166_166426

theorem calculate_inverse_y3_minus_y
  (i : ℂ) (y : ℂ)
  (h_i : i = Complex.I)
  (h_y : y = (1 + i * Real.sqrt 3) / 2) :
  (1 / (y^3 - y)) = -1/2 + i * (Real.sqrt 3) / 6 :=
by
  sorry

end calculate_inverse_y3_minus_y_l166_166426


namespace sufficient_not_necessary_range_l166_166386

variable (x a : ℝ)

theorem sufficient_not_necessary_range (h1 : ∀ x, |x| < 1 → x < a) 
                                       (h2 : ¬(∀ x, x < a → |x| < 1)) :
  a ≥ 1 :=
sorry

end sufficient_not_necessary_range_l166_166386


namespace terminal_side_second_quadrant_l166_166143

theorem terminal_side_second_quadrant (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end terminal_side_second_quadrant_l166_166143


namespace average_selling_price_is_86_l166_166502

def selling_prices := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

def average (prices : List Nat) : Nat :=
  (prices.sum) / prices.length

theorem average_selling_price_is_86 :
  average selling_prices = 86 :=
by
  sorry

end average_selling_price_is_86_l166_166502


namespace pirate_ship_minimum_speed_l166_166932

noncomputable def minimum_speed (initial_distance : ℝ) (caravel_speed : ℝ) (caravel_direction : ℝ) : ℝ :=
  let caravel_velocity_x := -caravel_speed * Real.cos caravel_direction
  let caravel_velocity_y := -caravel_speed * Real.sin caravel_direction
  let t := initial_distance / (caravel_speed * (1 + Real.sqrt 3))
  let v_p := Real.sqrt ((initial_distance / t - caravel_velocity_x)^2 + (caravel_velocity_y)^2)
  v_p

theorem pirate_ship_minimum_speed : 
  minimum_speed 10 12 (Real.pi / 3) = 6 * Real.sqrt 6 :=
by
  sorry

end pirate_ship_minimum_speed_l166_166932


namespace molecular_weight_correct_l166_166790

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms
def num_N : ℕ := 2
def num_O : ℕ := 3

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 76.02

-- The theorem to prove
theorem molecular_weight_correct :
  (num_N * atomic_weight_N + num_O * atomic_weight_O) = expected_molecular_weight := 
by
  sorry

end molecular_weight_correct_l166_166790


namespace Jillian_had_200_friends_l166_166874

def oranges : ℕ := 80
def pieces_per_orange : ℕ := 10
def pieces_per_friend : ℕ := 4
def number_of_friends : ℕ := oranges * pieces_per_orange / pieces_per_friend

theorem Jillian_had_200_friends :
  number_of_friends = 200 :=
sorry

end Jillian_had_200_friends_l166_166874


namespace intersection_points_l166_166613

def f(x : ℝ) : ℝ := x^2 + 3*x + 2
def g(x : ℝ) : ℝ := 4*x^2 + 6*x + 2

theorem intersection_points : {p : ℝ × ℝ | ∃ x, f x = p.2 ∧ g x = p.2 ∧ p.1 = x} = { (0, 2), (-1, 0) } := 
by {
  sorry
}

end intersection_points_l166_166613


namespace prime_p_geq_5_div_24_l166_166500

theorem prime_p_geq_5_div_24 (p : ℕ) (hp : Nat.Prime p) (hp_geq_5 : p ≥ 5) : 24 ∣ (p^2 - 1) :=
sorry

end prime_p_geq_5_div_24_l166_166500


namespace M_intersection_P_l166_166621

namespace IntersectionProof

-- Defining the sets M and P with given conditions
def M : Set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}
def P : Set ℝ := {y | y ≥ 1}

-- The theorem that corresponds to the problem statement
theorem M_intersection_P : (M ∩ P) = {y | y ≥ 1} :=
sorry

end IntersectionProof

end M_intersection_P_l166_166621


namespace coffee_is_32_3_percent_decaf_l166_166421

def percent_decaf_coffee_stock (total_weight initial_weight : ℕ) (initial_A_rate initial_B_rate initial_C_rate additional_weight additional_A_rate additional_D_rate : ℚ) 
(initial_A_decaf initial_B_decaf initial_C_decaf additional_D_decaf : ℚ) : ℚ :=
  let initial_A_weight := initial_A_rate * initial_weight
  let initial_B_weight := initial_B_rate * initial_weight
  let initial_C_weight := initial_C_rate * initial_weight
  let additional_A_weight := additional_A_rate * additional_weight
  let additional_D_weight := additional_D_rate * additional_weight

  let initial_A_decaf_weight := initial_A_decaf * initial_A_weight
  let initial_B_decaf_weight := initial_B_decaf * initial_B_weight
  let initial_C_decaf_weight := initial_C_decaf * initial_C_weight
  let additional_A_decaf_weight := initial_A_decaf * additional_A_weight
  let additional_D_decaf_weight := additional_D_decaf * additional_D_weight

  let total_decaf_weight := initial_A_decaf_weight + initial_B_decaf_weight + initial_C_decaf_weight + additional_A_decaf_weight + additional_D_decaf_weight

  (total_decaf_weight / total_weight) * 100

theorem coffee_is_32_3_percent_decaf : 
  percent_decaf_coffee_stock 1000 800 (40/100) (35/100) (25/100) 200 (50/100) (50/100) (20/100) (30/100) (45/100) (65/100) = 32.3 := 
  by 
    sorry

end coffee_is_32_3_percent_decaf_l166_166421


namespace third_number_is_42_l166_166515

variable (x : ℕ)

def number1 : ℕ := 5 * x
def number2 : ℕ := 6 * x
def number3 : ℕ := 8 * x

theorem third_number_is_42 (h : number1 x + number3 x = number2 x + 49) : number2 x = 42 :=
by
  sorry

end third_number_is_42_l166_166515


namespace max_area_of_right_angled_isosceles_triangle_l166_166315

theorem max_area_of_right_angled_isosceles_triangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) :
  ∃ A : ℝ, A = 72 ∧ 
  (∀ (x : ℝ), x ≤ min a b → (1 / 2) * x^2 ≤ A) :=
by
  use 72
  sorry

end max_area_of_right_angled_isosceles_triangle_l166_166315


namespace ratio_of_doctors_to_lawyers_l166_166738

/--
Given the average age of a group consisting of doctors and lawyers is 47,
the average age of doctors is 45,
and the average age of lawyers is 55,
prove that the ratio of the number of doctors to the number of lawyers is 4:1.
-/
theorem ratio_of_doctors_to_lawyers
  (d l : ℕ) -- numbers of doctors and lawyers
  (avg_group_age : ℝ := 47)
  (avg_doctors_age : ℝ := 45)
  (avg_lawyers_age : ℝ := 55)
  (h : (45 * d + 55 * l) / (d + l) = 47) :
  d = 4 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l166_166738


namespace probability_black_pen_l166_166211

-- Define the total number of pens and the specific counts
def total_pens : ℕ := 5 + 6 + 7
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the probability calculation
def probability (total : ℕ) (count : ℕ) : ℚ := count / total

-- State the theorem
theorem probability_black_pen :
  probability total_pens black_pens = 1 / 3 :=
by sorry

end probability_black_pen_l166_166211


namespace num_divisible_by_both_digits_l166_166228

theorem num_divisible_by_both_digits : 
  ∃ n, n = 14 ∧ ∀ (d : ℕ), (d ≥ 10 ∧ d < 100) → 
      (∀ a b, (d = 10 * a + b) → d % a = 0 ∧ d % b = 0 → (a = b ∨ a * 2 = b ∨ a * 5 = b)) :=
sorry

end num_divisible_by_both_digits_l166_166228


namespace time_diff_is_6_l166_166876

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end time_diff_is_6_l166_166876


namespace positive_difference_of_two_numbers_l166_166597

theorem positive_difference_of_two_numbers
  (x y : ℝ)
  (h₁ : x + y = 10)
  (h₂ : x^2 - y^2 = 24) :
  |x - y| = 12 / 5 :=
sorry

end positive_difference_of_two_numbers_l166_166597


namespace contrapositive_example_l166_166850

theorem contrapositive_example (x : ℝ) :
  (x ^ 2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x ^ 2 ≥ 1) :=
sorry

end contrapositive_example_l166_166850


namespace total_rainfall_recorded_l166_166264

-- Define the conditions based on the rainfall amounts for each day
def rainfall_monday : ℝ := 0.16666666666666666
def rainfall_tuesday : ℝ := 0.4166666666666667
def rainfall_wednesday : ℝ := 0.08333333333333333

-- State the theorem: the total rainfall recorded over the three days is 0.6666666666666667 cm.
theorem total_rainfall_recorded :
  (rainfall_monday + rainfall_tuesday + rainfall_wednesday) = 0.6666666666666667 := by
  sorry

end total_rainfall_recorded_l166_166264


namespace monotonic_range_of_b_l166_166993

noncomputable def f (b x : ℝ) : ℝ := x^3 - b * x^2 + 3 * x - 5

theorem monotonic_range_of_b (b : ℝ) : (∀ x y: ℝ, (f b x) ≤ (f b y) → x ≤ y) ↔ -3 ≤ b ∧ b ≤ 3 :=
sorry

end monotonic_range_of_b_l166_166993


namespace company_profit_is_correct_l166_166832

structure CompanyInfo where
  num_employees : ℕ
  shirts_per_employee_per_day : ℕ
  hours_per_shift : ℕ
  wage_per_hour : ℕ
  bonus_per_shirt : ℕ
  price_per_shirt : ℕ
  nonemployee_expenses_per_day : ℕ

def daily_profit (info : CompanyInfo) : ℤ :=
  let total_shirts_per_day := info.num_employees * info.shirts_per_employee_per_day
  let total_revenue := total_shirts_per_day * info.price_per_shirt
  let daily_wage_per_employee := info.wage_per_hour * info.hours_per_shift
  let total_daily_wage := daily_wage_per_employee * info.num_employees
  let daily_bonus_per_employee := info.bonus_per_shirt * info.shirts_per_employee_per_day
  let total_daily_bonus := daily_bonus_per_employee * info.num_employees
  let total_labor_cost := total_daily_wage + total_daily_bonus
  let total_expenses := total_labor_cost + info.nonemployee_expenses_per_day
  total_revenue - total_expenses

theorem company_profit_is_correct (info : CompanyInfo) (h : 
  info.num_employees = 20 ∧
  info.shirts_per_employee_per_day = 20 ∧
  info.hours_per_shift = 8 ∧
  info.wage_per_hour = 12 ∧
  info.bonus_per_shirt = 5 ∧
  info.price_per_shirt = 35 ∧
  info.nonemployee_expenses_per_day = 1000
) : daily_profit info = 9080 := 
by
  sorry

end company_profit_is_correct_l166_166832


namespace ratio_of_ages_l166_166765

theorem ratio_of_ages (sandy_future_age : ℕ) (sandy_years_future : ℕ) (molly_current_age : ℕ)
  (h1 : sandy_future_age = 42) (h2 : sandy_years_future = 6) (h3 : molly_current_age = 27) :
  (sandy_future_age - sandy_years_future) / gcd (sandy_future_age - sandy_years_future) molly_current_age = 
    4 / 3 :=
by
  sorry

end ratio_of_ages_l166_166765


namespace equation_of_line_l166_166865

theorem equation_of_line (x y : ℝ) :
  (∃ (x1 y1 : ℝ), (x1 = 0) ∧ (y1= 2) ∧ (y - y1 = 2 * (x - x1))) → (y = 2 * x + 2) :=
by
  sorry

end equation_of_line_l166_166865


namespace vasya_kolya_difference_impossible_l166_166883

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end vasya_kolya_difference_impossible_l166_166883


namespace cubics_inequality_l166_166268

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

end cubics_inequality_l166_166268


namespace bread_cost_each_is_3_l166_166306

-- Define the given conditions
def initial_amount : ℕ := 86
def bread_quantity : ℕ := 3
def orange_juice_quantity : ℕ := 3
def orange_juice_cost_each : ℕ := 6
def remaining_amount : ℕ := 59

-- Define the variable for bread cost
variable (B : ℕ)

-- Lean 4 statement to prove the cost of each loaf of bread
theorem bread_cost_each_is_3 :
  initial_amount - remaining_amount = (bread_quantity * B + orange_juice_quantity * orange_juice_cost_each) →
  B = 3 :=
by
  sorry

end bread_cost_each_is_3_l166_166306


namespace find_a100_find_a1983_l166_166571

open Nat

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ k, a (a k) = 3 * k

theorem find_a100 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 100 = 181 := 
sorry

theorem find_a1983 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 1983 = 3762 := 
sorry

end find_a100_find_a1983_l166_166571


namespace solve_fraction_identity_l166_166262

theorem solve_fraction_identity (x : ℝ) (hx : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_identity_l166_166262


namespace difference_of_square_of_non_divisible_by_3_l166_166779

theorem difference_of_square_of_non_divisible_by_3 (n : ℕ) (h : ¬ (n % 3 = 0)) : (n^2 - 1) % 3 = 0 :=
sorry

end difference_of_square_of_non_divisible_by_3_l166_166779


namespace class_duration_l166_166035

theorem class_duration (h1 : 8 * 60 + 30 = 510) (h2 : 9 * 60 + 5 = 545) : (545 - 510 = 35) :=
by
  sorry

end class_duration_l166_166035


namespace road_length_l166_166310

theorem road_length (L : ℝ) (h1 : 300 = 200 + 100)
  (h2 : 50 * 100 = 2.5 / (L / 300))
  (h3 : 75 + 50 = 125)
  (h4 : (125 / 50) * (2.5 / 100) * 200 = L - 2.5) : L = 15 := 
by
  sorry

end road_length_l166_166310


namespace find_B_l166_166110

variable {U : Set ℕ}

def A : Set ℕ := {1, 3, 5, 7}
def complement_A : Set ℕ := {2, 4, 6}
def complement_B : Set ℕ := {1, 4, 6}
def B : Set ℕ := {2, 3, 5, 7}

theorem find_B
  (hU : U = A ∪ complement_A)
  (A_comp : ∀ x, x ∈ complement_A ↔ x ∉ A)
  (B_comp : ∀ x, x ∈ complement_B ↔ x ∉ B) :
  B = {2, 3, 5, 7} :=
sorry

end find_B_l166_166110


namespace number_of_moles_of_water_formed_l166_166011

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

end number_of_moles_of_water_formed_l166_166011


namespace angle_PQR_correct_l166_166355

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l166_166355


namespace crowdfunding_highest_level_backing_l166_166792

-- Definitions according to the conditions
def lowest_level_backing : ℕ := 50
def second_level_backing : ℕ := 10 * lowest_level_backing
def highest_level_backing : ℕ := 100 * lowest_level_backing
def total_raised : ℕ := (2 * highest_level_backing) + (3 * second_level_backing) + (10 * lowest_level_backing)

-- Statement of the problem
theorem crowdfunding_highest_level_backing (h: total_raised = 12000) :
  highest_level_backing = 5000 :=
sorry

end crowdfunding_highest_level_backing_l166_166792


namespace factorize_x_squared_minus_nine_l166_166694

theorem factorize_x_squared_minus_nine : ∀ (x : ℝ), x^2 - 9 = (x - 3) * (x + 3) :=
by
  intro x
  exact sorry

end factorize_x_squared_minus_nine_l166_166694


namespace passenger_waiting_time_probability_l166_166550

def bus_arrival_interval : ℕ := 5

def waiting_time_limit : ℕ := 3

/-- 
  Prove that for a bus arriving every 5 minutes,
  the probability that a passenger's waiting time 
  is no more than 3 minutes, given the passenger 
  arrives at a random time, is 3/5. 
--/
theorem passenger_waiting_time_probability 
  (bus_interval : ℕ) (time_limit : ℕ) 
  (random_arrival : ℝ) :
  bus_interval = 5 →
  time_limit = 3 →
  0 ≤ random_arrival ∧ random_arrival < bus_interval →
  (random_arrival ≤ time_limit) →
  (random_arrival / ↑bus_interval) = 3 / 5 :=
by
  sorry

end passenger_waiting_time_probability_l166_166550


namespace company_KW_price_percentage_l166_166147

theorem company_KW_price_percentage
  (A B : ℝ)
  (h1 : ∀ P: ℝ, P = 1.9 * A)
  (h2 : ∀ P: ℝ, P = 2 * B) :
  Price = 131.034 / 100 * (A + B) := 
by
  sorry

end company_KW_price_percentage_l166_166147


namespace probability_of_red_jelly_bean_l166_166552

-- Definitions based on conditions
def total_jelly_beans := 7 + 9 + 4 + 10
def red_jelly_beans := 7

-- Statement we want to prove
theorem probability_of_red_jelly_bean : (red_jelly_beans : ℚ) / total_jelly_beans = 7 / 30 :=
by
  -- Proof here
  sorry

end probability_of_red_jelly_bean_l166_166552


namespace smallest_part_of_division_l166_166922

theorem smallest_part_of_division (x : ℝ) (h : 2 * x + (1/2) * x + (1/4) * x = 105) : 
  (1/4) * x = 10.5 :=
sorry

end smallest_part_of_division_l166_166922


namespace intersection_correct_union_correct_l166_166648

variable (U A B : Set Nat)

def U_set : U = {1, 2, 3, 4, 5, 6} := by sorry
def A_set : A = {2, 4, 5} := by sorry
def B_set : B = {1, 2, 5} := by sorry

theorem intersection_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∩ B) = {2, 5} := by sorry

theorem union_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∪ (U \ B)) = {2, 3, 4, 5, 6} := by sorry

end intersection_correct_union_correct_l166_166648


namespace simplified_value_l166_166189

-- Define the given expression
def expr := (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3)

-- State the theorem
theorem simplified_value : expr = 10^1.7 :=
by
  sorry -- Proof omitted

end simplified_value_l166_166189


namespace water_depth_is_12_feet_l166_166868

variable (Ron_height Dean_height Water_depth : ℕ)

-- Given conditions
axiom H1 : Ron_height = 14
axiom H2 : Dean_height = Ron_height - 8
axiom H3 : Water_depth = 2 * Dean_height

-- Prove that the water depth is 12 feet
theorem water_depth_is_12_feet : Water_depth = 12 :=
by
  sorry

end water_depth_is_12_feet_l166_166868


namespace coeffs_of_quadratic_eq_l166_166414

theorem coeffs_of_quadratic_eq :
  ∃ a b c : ℤ, (2 * x^2 + x - 5 = 0) → (a = 2 ∧ b = 1 ∧ c = -5) :=
by
  sorry

end coeffs_of_quadratic_eq_l166_166414


namespace fraction_of_orange_juice_is_correct_l166_166393

noncomputable def fraction_of_orange_juice_in_mixture (V1 V2 juice1_ratio juice2_ratio : ℚ) : ℚ :=
  let juice1 := V1 * juice1_ratio
  let juice2 := V2 * juice2_ratio
  let total_juice := juice1 + juice2
  let total_volume := V1 + V2
  total_juice / total_volume

theorem fraction_of_orange_juice_is_correct :
  fraction_of_orange_juice_in_mixture 800 500 (1/4) (1/3) = 7 / 25 :=
by sorry

end fraction_of_orange_juice_is_correct_l166_166393


namespace triangle_segments_l166_166687

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_segments (a : ℕ) (h : a > 0) :
  ¬ triangle_inequality 1 2 3 ∧
  ¬ triangle_inequality 4 5 10 ∧
  triangle_inequality 5 10 13 ∧
  ¬ triangle_inequality (2 * a) (3 * a) (6 * a) :=
by
  -- Proof goes here
  sorry

end triangle_segments_l166_166687


namespace identify_counterfeit_coin_correct_l166_166398

noncomputable def identify_counterfeit_coin (coins : Fin 8 → ℝ) : ℕ :=
  sorry

theorem identify_counterfeit_coin_correct (coins : Fin 8 → ℝ) (h_fake : 
  ∃ i : Fin 8, ∀ j : Fin 8, j ≠ i → coins i > coins j) : 
  ∃ i : Fin 8, identify_counterfeit_coin coins = i ∧ ∀ j : Fin 8, j ≠ i → coins i > coins j :=
by
  sorry

end identify_counterfeit_coin_correct_l166_166398


namespace probability_sum_equals_6_l166_166473

theorem probability_sum_equals_6 : 
  let possible_outcomes := 36
  let favorable_outcomes := 5
  (favorable_outcomes / possible_outcomes : ℚ) = 5 / 36 := 
by 
  sorry

end probability_sum_equals_6_l166_166473


namespace calculate_ray_grocery_bill_l166_166399

noncomputable def ray_grocery_total_cost : ℝ :=
let hamburger_meat_price := 5.0
let crackers_price := 3.5
let frozen_vegetables_price := 2.0 * 4
let cheese_price := 3.5
let chicken_price := 6.5
let cereal_price := 4.0
let wine_price := 10.0
let cookies_price := 3.0

let discount_hamburger_meat := hamburger_meat_price * 0.10
let discount_crackers := crackers_price * 0.10
let discount_frozen_vegetables := frozen_vegetables_price * 0.10
let discount_cheese := cheese_price * 0.05
let discount_chicken := chicken_price * 0.05
let discount_wine := wine_price * 0.15

let discounted_hamburger_meat_price := hamburger_meat_price - discount_hamburger_meat
let discounted_crackers_price := crackers_price - discount_crackers
let discounted_frozen_vegetables_price := frozen_vegetables_price - discount_frozen_vegetables
let discounted_cheese_price := cheese_price - discount_cheese
let discounted_chicken_price := chicken_price - discount_chicken
let discounted_wine_price := wine_price - discount_wine

let total_discounted_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  discounted_wine_price +
  cookies_price

let food_items_total_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  cookies_price

let food_sales_tax := food_items_total_price * 0.06
let wine_sales_tax := discounted_wine_price * 0.09

let total_with_tax := total_discounted_price + food_sales_tax + wine_sales_tax

total_with_tax

theorem calculate_ray_grocery_bill :
  ray_grocery_total_cost = 42.51 :=
sorry

end calculate_ray_grocery_bill_l166_166399


namespace num_customers_left_more_than_remaining_l166_166415

theorem num_customers_left_more_than_remaining (initial remaining : ℕ) (h : initial = 11 ∧ remaining = 3) : (initial - remaining) = (remaining + 5) :=
by sorry

end num_customers_left_more_than_remaining_l166_166415


namespace toothpicks_needed_l166_166614

-- Defining the number of rows in the large equilateral triangle.
def rows : ℕ := 10

-- Formula to compute the total number of smaller equilateral triangles.
def total_small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

-- Number of small triangles in this specific case.
def num_small_triangles : ℕ := total_small_triangles rows

-- Total toothpicks without sharing sides.
def total_sides_no_sharing (n : ℕ) : ℕ := 3 * num_small_triangles

-- Adjust for shared toothpicks internally.
def shared_toothpicks (n : ℕ) : ℕ := (total_sides_no_sharing n - 3 * rows) / 2 + 3 * rows

-- Total boundary toothpicks.
def boundary_toothpicks (n : ℕ) : ℕ := 3 * rows

-- Final total number of toothpicks required.
def total_toothpicks (n : ℕ) : ℕ := shared_toothpicks n + boundary_toothpicks n

-- The theorem to be proved
theorem toothpicks_needed : total_toothpicks rows = 98 :=
by
  -- You can complete the proof.
  sorry

end toothpicks_needed_l166_166614


namespace peculiar_looking_less_than_500_l166_166526

def is_composite (n : ℕ) : Prop :=
  1 < n ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def peculiar_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (n % 2 = 0 ∨ n % 3 = 0 ∨ n % 7 = 0 ∨ n % 11 = 0)

theorem peculiar_looking_less_than_500 :
  ∃ n, n = 33 ∧ ∀ k, k < 500 → peculiar_looking k → k = n :=
sorry

end peculiar_looking_less_than_500_l166_166526


namespace lateral_surface_area_of_prism_l166_166509

theorem lateral_surface_area_of_prism (h : ℝ) (angle : ℝ) (h_pos : 0 < h) (angle_eq : angle = 60) :
  ∃ S : ℝ, S = 6 * h^2 :=
by
  sorry

end lateral_surface_area_of_prism_l166_166509


namespace suff_and_nec_eq_triangle_l166_166208

noncomputable def triangle (A B C: ℝ) (a b c : ℝ) : Prop :=
(B + C = 2 * A) ∧ (b + c = 2 * a)

theorem suff_and_nec_eq_triangle (A B C a b c : ℝ) (h : triangle A B C a b c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
sorry

end suff_and_nec_eq_triangle_l166_166208


namespace power_function_general_form_l166_166471

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_general_form (α : ℝ) :
  ∃ y : ℝ, ∃ α : ℝ, f 3 α = y ∧ ∀ x : ℝ, f x α = x ^ α :=
by
  sorry

end power_function_general_form_l166_166471


namespace percentage_markup_l166_166375

theorem percentage_markup (sell_price : ℝ) (cost_price : ℝ)
  (h_sell : sell_price = 8450) (h_cost : cost_price = 6500) : 
  (sell_price - cost_price) / cost_price * 100 = 30 :=
by
  sorry

end percentage_markup_l166_166375


namespace parallel_lines_slope_l166_166290

-- Define the given conditions
def line1_slope (x : ℝ) : ℝ := 6
def line2_slope (c : ℝ) (x : ℝ) : ℝ := 3 * c

-- State the proof problem
theorem parallel_lines_slope (c : ℝ) : 
  (∀ x : ℝ, line1_slope x = line2_slope c x) → c = 2 :=
by
  intro h
  -- Intro provides a human-readable variable and corresponding proof obligation
  -- The remainder of the proof would follow here, but instead,
  -- we use "sorry" to indicate an incomplete proof
  sorry

end parallel_lines_slope_l166_166290


namespace inequality_l166_166777

theorem inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 :=
sorry

end inequality_l166_166777


namespace find_value_of_expression_l166_166871

theorem find_value_of_expression
  (k m : ℕ)
  (hk : 3^(k - 1) = 9)
  (hm : 4^(m + 2) = 64) :
  2^(3*k + 2*m) = 2^11 :=
by 
  sorry

end find_value_of_expression_l166_166871


namespace find_number_l166_166896

-- Statement of the problem in Lean 4
theorem find_number (n : ℝ) (h : n / 3000 = 0.008416666666666666) : n = 25.25 :=
sorry

end find_number_l166_166896


namespace inequality_of_sums_l166_166188

theorem inequality_of_sums
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1)
  (h2 : 0 < a2)
  (h3 : a1 > a2)
  (h4 : b1 ≥ a1)
  (h5 : b1 * b2 ≥ a1 * a2) :
  b1 + b2 ≥ a1 + a2 :=
by
  -- Here we don't provide the proof
  sorry

end inequality_of_sums_l166_166188


namespace problem_solution_l166_166337

theorem problem_solution
  (a b c : ℝ)
  (habc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
by
  sorry

end problem_solution_l166_166337


namespace contractor_total_amount_l166_166557

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l166_166557


namespace distinct_positive_integers_solution_l166_166231

theorem distinct_positive_integers_solution (x y : ℕ) (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : 1 / x + 1 / y = 2 / 7) : (x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4) :=
by
  sorry -- proof to be filled in.

end distinct_positive_integers_solution_l166_166231


namespace volume_of_rectangular_prism_l166_166087

theorem volume_of_rectangular_prism {l w h : ℝ} 
  (h1 : l * w = 12) 
  (h2 : w * h = 18) 
  (h3 : l * h = 24) : 
  l * w * h = 72 :=
by
  sorry

end volume_of_rectangular_prism_l166_166087


namespace log_expression_equals_four_l166_166736

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l166_166736


namespace second_class_students_l166_166964

-- Define the conditions
variables (x : ℕ)
variable (sum_marks_first_class : ℕ := 35 * 40)
variable (sum_marks_second_class : ℕ := x * 60)
variable (total_students : ℕ := 35 + x)
variable (total_marks_all_students : ℕ := total_students * 5125 / 100)

-- The theorem to prove
theorem second_class_students : 
  1400 + (x * 60) = (35 + x) * 5125 / 100 →
  x = 45 :=
by
  sorry

end second_class_students_l166_166964


namespace scientific_notation_3080000_l166_166523

theorem scientific_notation_3080000 : (3080000 : ℝ) = 3.08 * 10^6 := 
by
  sorry

end scientific_notation_3080000_l166_166523


namespace find_n_for_perfect_square_l166_166660

theorem find_n_for_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℤ), n^2 + 5 * n + 13 = m^2 ∧ n = 4 :=
by
  sorry

end find_n_for_perfect_square_l166_166660


namespace quadratic_inequality_range_l166_166904

theorem quadratic_inequality_range (a x : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end quadratic_inequality_range_l166_166904


namespace linear_expressions_constant_multiple_l166_166525

theorem linear_expressions_constant_multiple 
    (a b c p q r : ℝ)
    (h : (a*x + p)^2 + (b*x + q)^2 = (c*x + r)^2) : 
    a*b ≠ 0 → p*q ≠ 0 → (a / b = p / q) :=
by
  -- Given: (ax + p)^2 + (bx + q)^2 = (cx + r)^2
  -- Prove: a / b = p / q, implying that A(x) and B(x) can be expressed as the constant times C(x)
  sorry

end linear_expressions_constant_multiple_l166_166525


namespace total_fruits_correct_l166_166464

def total_fruits 
  (Jason_watermelons : Nat) (Jason_pineapples : Nat)
  (Mark_watermelons : Nat) (Mark_pineapples : Nat)
  (Sandy_watermelons : Nat) (Sandy_pineapples : Nat) : Nat :=
  Jason_watermelons + Jason_pineapples +
  Mark_watermelons + Mark_pineapples +
  Sandy_watermelons + Sandy_pineapples

theorem total_fruits_correct :
  total_fruits 37 56 68 27 11 14 = 213 :=
by
  sorry

end total_fruits_correct_l166_166464


namespace no_solution_for_x_l166_166239

open Real

theorem no_solution_for_x (m : ℝ) : ¬ ∃ x : ℝ, (sin (3 * x) * cos (↑60 - x) + 1) / (sin (↑60 - 7 * x) - cos (↑30 + x) + m) = 0 :=
by
  sorry

end no_solution_for_x_l166_166239


namespace problem_l166_166583

theorem problem (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -1/2 ∨ x > 1/3)) : 
  a + b = -14 :=
sorry

end problem_l166_166583


namespace earnings_from_jam_l166_166171

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end earnings_from_jam_l166_166171


namespace gcf_270_108_150_l166_166940

theorem gcf_270_108_150 : Nat.gcd (Nat.gcd 270 108) 150 = 30 := 
  sorry

end gcf_270_108_150_l166_166940


namespace quadratic_roots_range_l166_166895

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (x1^2 - 2 * x1 + m - 2 = 0) ∧ 
    (x2^2 - 2 * x2 + m - 2 = 0)) → m < 3 := 
by 
  sorry

end quadratic_roots_range_l166_166895


namespace price_change_theorem_l166_166053

-- Define initial prices
def candy_box_price_before : ℝ := 10
def soda_can_price_before : ℝ := 9
def popcorn_bag_price_before : ℝ := 5
def gum_pack_price_before : ℝ := 2

-- Define price changes
def candy_box_price_increase := candy_box_price_before * 0.25
def soda_can_price_decrease := soda_can_price_before * 0.15
def popcorn_bag_price_factor := 2
def gum_pack_price_change := 0

-- Compute prices after the policy changes
def candy_box_price_after := candy_box_price_before + candy_box_price_increase
def soda_can_price_after := soda_can_price_before - soda_can_price_decrease
def popcorn_bag_price_after := popcorn_bag_price_before * popcorn_bag_price_factor
def gum_pack_price_after := gum_pack_price_before

-- Compute total costs
def total_cost_before := candy_box_price_before + soda_can_price_before + popcorn_bag_price_before + gum_pack_price_before
def total_cost_after := candy_box_price_after + soda_can_price_after + popcorn_bag_price_after + gum_pack_price_after

-- The statement to be proven
theorem price_change_theorem :
  total_cost_before = 26 ∧ total_cost_after = 32.15 :=
by
  -- This part requires proof, add 'sorry' for now
  sorry

end price_change_theorem_l166_166053


namespace option_c_correct_l166_166093

theorem option_c_correct : (3 * Real.sqrt 2) ^ 2 = 18 :=
by 
  -- Proof to be provided here
  sorry

end option_c_correct_l166_166093


namespace monroe_collection_legs_l166_166475

theorem monroe_collection_legs : 
  let ants := 12 
  let spiders := 8 
  let beetles := 15 
  let centipedes := 5 
  let legs_ants := 6 
  let legs_spiders := 8 
  let legs_beetles := 6 
  let legs_centipedes := 100
  (ants * legs_ants + spiders * legs_spiders + beetles * legs_beetles + centipedes * legs_centipedes = 726) := 
by 
  sorry

end monroe_collection_legs_l166_166475


namespace complement_set_l166_166260

open Set

variable (U : Set ℝ) (M : Set ℝ)

theorem complement_set :
  U = univ ∧ M = {x | x^2 - 2 * x ≤ 0} → (U \ M) = {x | x < 0 ∨ x > 2} :=
by
  intros
  sorry

end complement_set_l166_166260


namespace find_x_l166_166705

variable (x : ℝ)
variable (l : ℝ) (w : ℝ)

def length := 4 * x + 1
def width := x + 7

theorem find_x (h1 : l = length x) (h2 : w = width x) (h3 : l * w = 2 * (2 * l + 2 * w)) :
  x = (-9 + Real.sqrt 481) / 8 :=
by
  subst_vars
  sorry

end find_x_l166_166705


namespace max_elevation_l166_166304

def elevation (t : ℝ) : ℝ := 144 * t - 18 * t^2

theorem max_elevation : ∃ t : ℝ, elevation t = 288 :=
by
  use 4
  sorry

end max_elevation_l166_166304


namespace z_is_negative_y_intercept_l166_166803

-- Define the objective function as an assumption or condition
def objective_function (x y z : ℝ) : Prop := z = 3 * x - y

-- Define what we need to prove: z is the negative of the y-intercept 
def negative_y_intercept (x y z : ℝ) : Prop := ∃ m b, (y = m * x + b) ∧ m = 3 ∧ b = -z

-- The theorem we need to prove
theorem z_is_negative_y_intercept (x y z : ℝ) (h : objective_function x y z) : negative_y_intercept x y z :=
  sorry

end z_is_negative_y_intercept_l166_166803


namespace integer_values_of_a_l166_166506

theorem integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^4 + 4 * x^3 + a * x^2 + 8 = 0) ↔ (a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2) :=
sorry

end integer_values_of_a_l166_166506


namespace tina_wins_more_than_losses_l166_166886

theorem tina_wins_more_than_losses 
  (initial_wins : ℕ)
  (additional_wins : ℕ)
  (first_loss : ℕ)
  (doubled_wins : ℕ)
  (second_loss : ℕ)
  (total_wins : ℕ)
  (total_losses : ℕ)
  (final_difference : ℕ) :
  initial_wins = 10 →
  additional_wins = 5 →
  first_loss = 1 →
  doubled_wins = 30 →
  second_loss = 1 →
  total_wins = initial_wins + additional_wins + doubled_wins →
  total_losses = first_loss + second_loss →
  final_difference = total_wins - total_losses →
  final_difference = 43 :=
by
  sorry

end tina_wins_more_than_losses_l166_166886


namespace airfare_price_for_BD_l166_166776

theorem airfare_price_for_BD (AB AC AD CD BC : ℝ) (hAB : AB = 2000) (hAC : AC = 1600) (hAD : AD = 2500) 
    (hCD : CD = 900) (hBC : BC = 1200) (proportional_pricing : ∀ x y : ℝ, x * (y / x) = y) : 
    ∃ BD : ℝ, BD = 1500 :=
by
  sorry

end airfare_price_for_BD_l166_166776


namespace solution_to_eq_l166_166587

def eq1 (x y z t : ℕ) : Prop := x * y - x * z + y * t = 182
def cond_numbers (n : ℕ) : Prop := n = 12 ∨ n = 14 ∨ n = 37 ∨ n = 65

theorem solution_to_eq 
  (x y z t : ℕ) 
  (hx : cond_numbers x) 
  (hy : cond_numbers y) 
  (hz : cond_numbers z) 
  (ht : cond_numbers t) 
  (h : eq1 x y z t) : 
  (x = 12 ∧ y = 37 ∧ z = 65 ∧ t = 14) ∨ 
  (x = 37 ∧ y = 12 ∧ z = 14 ∧ t = 65) := 
sorry

end solution_to_eq_l166_166587


namespace minimum_degree_of_g_l166_166052

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

theorem minimum_degree_of_g :
  (5 * f - 3 * g = h) →
  (Polynomial.degree f = 10) →
  (Polynomial.degree h = 11) →
  (Polynomial.degree g = 11) :=
sorry

end minimum_degree_of_g_l166_166052


namespace evaluate_expression_correct_l166_166900

noncomputable def evaluate_expression :=
  abs (-1) - ((-3.14 + Real.pi) ^ 0) + (2 ^ (-1 : ℤ)) + (Real.cos (Real.pi / 6)) ^ 2

theorem evaluate_expression_correct : evaluate_expression = 5 / 4 := by sorry

end evaluate_expression_correct_l166_166900


namespace notable_features_points_l166_166249

namespace Points3D

def is_first_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y > 0) ∧ (z > 0)
def is_second_octant (x y z : ℝ) : Prop := (x < 0) ∧ (y > 0) ∧ (z > 0)
def is_eighth_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y < 0) ∧ (z < 0)
def lies_in_YOZ_plane (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z ≠ 0)
def lies_on_OY_axis (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z = 0)
def is_origin (x y z : ℝ) : Prop := (x = 0) ∧ (y = 0) ∧ (z = 0)

theorem notable_features_points :
  is_first_octant 3 2 6 ∧
  is_second_octant (-2) 3 1 ∧
  is_eighth_octant 1 (-4) (-2) ∧
  is_eighth_octant 1 (-2) (-1) ∧
  lies_in_YOZ_plane 0 4 1 ∧
  lies_on_OY_axis 0 2 0 ∧
  is_origin 0 0 0 :=
by
  sorry

end Points3D

end notable_features_points_l166_166249


namespace original_percent_acid_l166_166578

open Real

variables (a w : ℝ)

theorem original_percent_acid 
  (h1 : (a + 2) / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 4) = 1 / 5) :
  a / (a + w) = 1 / 5 :=
sorry

end original_percent_acid_l166_166578


namespace even_function_maximum_l166_166560

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def has_maximum_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ ∀ y : ℝ, a ≤ y ∧ y ≤ b → f y ≤ f x

theorem even_function_maximum 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_max_1_7 : has_maximum_on_interval f 1 7) :
  has_maximum_on_interval f (-7) (-1) :=
sorry

end even_function_maximum_l166_166560


namespace ebay_ordered_cards_correct_l166_166683

noncomputable def initial_cards := 4
noncomputable def father_cards := 13
noncomputable def cards_given_to_dexter := 29
noncomputable def cards_kept := 20
noncomputable def bad_cards := 4

theorem ebay_ordered_cards_correct :
  let total_before_ebay := initial_cards + father_cards
  let total_after_giving_and_keeping := cards_given_to_dexter + cards_kept
  let ordered_before_bad := total_after_giving_and_keeping - total_before_ebay
  let ebay_ordered_cards := ordered_before_bad + bad_cards
  ebay_ordered_cards = 36 :=
by
  sorry

end ebay_ordered_cards_correct_l166_166683


namespace intersection_M_N_l166_166984

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3 * x = 0}

theorem intersection_M_N : M ∩ N = {0} :=
by sorry

end intersection_M_N_l166_166984


namespace hannahs_adblock_not_block_l166_166601

theorem hannahs_adblock_not_block (x : ℝ) (h1 : 0.8 * x = 0.16) : x = 0.2 :=
by {
  sorry
}

end hannahs_adblock_not_block_l166_166601


namespace ted_cookies_eaten_l166_166535

def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def days_baking : ℕ := 6
def cookies_per_day : ℕ := trays_per_day * cookies_per_tray
def total_cookies_baked : ℕ := days_baking * cookies_per_day
def cookies_eaten_by_frank : ℕ := days_baking
def cookies_before_ted : ℕ := total_cookies_baked - cookies_eaten_by_frank
def cookies_left_after_ted : ℕ := 134

theorem ted_cookies_eaten : cookies_before_ted - cookies_left_after_ted = 4 := by
  sorry

end ted_cookies_eaten_l166_166535


namespace minimum_black_edges_5x5_l166_166540

noncomputable def minimum_black_edges_on_border (n : ℕ) : ℕ :=
if n = 5 then 5 else 0

theorem minimum_black_edges_5x5 : 
  minimum_black_edges_on_border 5 = 5 :=
by sorry

end minimum_black_edges_5x5_l166_166540


namespace problem_f_2011_2012_l166_166942

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2011_2012 :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f (1-x) = f (1+x)) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = 2^x - 1) →
  f 2011 + f 2012 = -1 :=
by
  intros h1 h2 h3
  sorry

end problem_f_2011_2012_l166_166942


namespace angle_triple_complement_l166_166941

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l166_166941


namespace regular_tetrahedron_surface_area_l166_166279

theorem regular_tetrahedron_surface_area {h : ℝ} (h_pos : h > 0) :
  ∃ (S : ℝ), S = (3 * h^2 * Real.sqrt 3) / 2 :=
sorry

end regular_tetrahedron_surface_area_l166_166279


namespace square_side_length_l166_166735

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by 
  sorry

end square_side_length_l166_166735


namespace scores_are_sample_l166_166420

-- Define the total number of students
def total_students : ℕ := 5000

-- Define the number of selected students for sampling
def selected_students : ℕ := 200

-- Define a predicate that checks if a selection is a sample
def is_sample (total selected : ℕ) : Prop :=
  selected < total

-- The proposition that needs to be proven
theorem scores_are_sample : is_sample total_students selected_students := 
by 
  -- Proof of the theorem is omitted.
  sorry

end scores_are_sample_l166_166420


namespace sequence_expression_l166_166222

theorem sequence_expression (a : ℕ → ℚ)
  (h1 : a 1 = 2 / 3)
  (h2 : ∀ n : ℕ, a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, a n = 2 / (3 * n) :=
sorry

end sequence_expression_l166_166222


namespace three_legged_tables_count_l166_166929

theorem three_legged_tables_count (x y : ℕ) (h1 : 3 * x + 4 * y = 23) (h2 : 2 ≤ x) (h3 : 2 ≤ y) : x = 5 := 
sorry

end three_legged_tables_count_l166_166929


namespace smallest_number_of_roses_to_buy_l166_166033

-- Definitions representing the conditions
def group_size1 : ℕ := 9
def group_size2 : ℕ := 19

-- Statement representing the problem and solution
theorem smallest_number_of_roses_to_buy : Nat.lcm group_size1 group_size2 = 171 := 
by 
  sorry

end smallest_number_of_roses_to_buy_l166_166033


namespace find_a1_in_arithmetic_sequence_l166_166901

theorem find_a1_in_arithmetic_sequence (d n a_n : ℤ) (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10) :
  ∃ a1 : ℤ, a1 = -38 :=
by
  sorry

end find_a1_in_arithmetic_sequence_l166_166901


namespace number_of_stickers_used_to_decorate_l166_166134

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end number_of_stickers_used_to_decorate_l166_166134


namespace sequence_count_l166_166785

def num_sequences (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem sequence_count :
  let x := 490
  let y := 510
  let a : (n : ℕ) → ℕ := fun n => if n = 0 then 0 else if n = 1000 then 2020 else sorry
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1000 → (a (k + 1) - a k = 1 ∨ a (k + 1) - a k = 3)) →
  (∃ binomial_coeff, binomial_coeff = num_sequences 1000 490) :=
by sorry

end sequence_count_l166_166785


namespace inequality_proof_l166_166747

theorem inequality_proof {x y z : ℝ}
  (h1 : x + 2 * y + 4 * z ≥ 3)
  (h2 : y - 3 * x + 2 * z ≥ 5) :
  y - x + 2 * z ≥ 3 :=
by
  sorry

end inequality_proof_l166_166747


namespace solution_set_l166_166531

variable (f : ℝ → ℝ)

def cond1 := ∀ x, f x = f (-x)
def cond2 := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
def cond3 := f (1/3) = 0

theorem solution_set (hf1 : cond1 f) (hf2 : cond2 f) (hf3 : cond3 f) :
  { x : ℝ | f (Real.log x / Real.log (1/8)) > 0 } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | 2 < x } :=
sorry

end solution_set_l166_166531


namespace weight_of_b_l166_166644

variable {A B C : ℤ}

def condition1 (A B C : ℤ) : Prop := (A + B + C) / 3 = 45
def condition2 (A B : ℤ) : Prop := (A + B) / 2 = 42
def condition3 (B C : ℤ) : Prop := (B + C) / 2 = 43

theorem weight_of_b (A B C : ℤ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B) 
  (h3 : condition3 B C) : 
  B = 35 := 
by
  sorry

end weight_of_b_l166_166644


namespace find_e_l166_166111

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end find_e_l166_166111


namespace Deepak_age_l166_166752

variable (R D : ℕ)

theorem Deepak_age 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26) : D = 15 := 
sorry

end Deepak_age_l166_166752


namespace area_of_hexagon_l166_166615

-- Definitions of the angles and side lengths
def angle_A := 120
def angle_B := 120
def angle_C := 120
def angle_D := 150

def FA := 2
def AB := 2
def BC := 2
def CD := 3
def DE := 3
def EF := 3

-- Theorem statement for the area of hexagon ABCDEF
theorem area_of_hexagon : 
  (angle_A = 120 ∧ angle_B = 120 ∧ angle_C = 120 ∧ angle_D = 150 ∧
   FA = 2 ∧ AB = 2 ∧ BC = 2 ∧ CD = 3 ∧ DE = 3 ∧ EF = 3) →
  (∃ area : ℝ, area = 7.5 * Real.sqrt 3) :=
by
  sorry

end area_of_hexagon_l166_166615


namespace margo_total_distance_l166_166642

theorem margo_total_distance (time_to_friend : ℝ) (time_back_home : ℝ) (average_rate : ℝ)
  (total_time_hours : ℝ) (total_miles : ℝ) :
  time_to_friend = 12 / 60 ∧
  time_back_home = 24 / 60 ∧
  total_time_hours = (12 / 60) + (24 / 60) ∧
  average_rate = 3 ∧
  total_miles = average_rate * total_time_hours →
  total_miles = 1.8 :=
by
  sorry

end margo_total_distance_l166_166642


namespace focus_of_parabola_l166_166237

theorem focus_of_parabola (x y : ℝ) (h : x^2 = -y) : (0, -1/4) = (0, -1/4) :=
by sorry

end focus_of_parabola_l166_166237


namespace total_time_correct_l166_166067

def greta_time : ℝ := 6.5
def george_time : ℝ := greta_time - 1.5
def gloria_time : ℝ := 2 * george_time
def gary_time : ℝ := (george_time + gloria_time) + 1.75
def gwen_time : ℝ := (greta_time + george_time) - 0.40 * (greta_time + george_time)
def total_time : ℝ := greta_time + george_time + gloria_time + gary_time + gwen_time

theorem total_time_correct : total_time = 45.15 := by
  sorry

end total_time_correct_l166_166067


namespace rotated_square_height_l166_166756

noncomputable def height_of_B (side_length : ℝ) (rotation_angle : ℝ) : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let vertical_component := diagonal * Real.sin rotation_angle
  vertical_component

theorem rotated_square_height :
  height_of_B 1 (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end rotated_square_height_l166_166756


namespace div_neg_rev_l166_166097

theorem div_neg_rev (a b : ℝ) (h : a > b) : (a / -3) < (b / -3) :=
by
  sorry

end div_neg_rev_l166_166097


namespace smallest_number_diminished_by_16_divisible_l166_166112

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end smallest_number_diminished_by_16_divisible_l166_166112


namespace exists_four_consecutive_with_square_divisors_l166_166091

theorem exists_four_consecutive_with_square_divisors :
  ∃ n : ℕ, n = 3624 ∧
  (∃ d1, d1^2 > 1 ∧ d1^2 ∣ n) ∧ 
  (∃ d2, d2^2 > 1 ∧ d2^2 ∣ (n + 1)) ∧ 
  (∃ d3, d3^2 > 1 ∧ d3^2 ∣ (n + 2)) ∧ 
  (∃ d4, d4^2 > 1 ∧ d4^2 ∣ (n + 3)) :=
sorry

end exists_four_consecutive_with_square_divisors_l166_166091


namespace greatest_value_of_squares_l166_166982

-- Given conditions
variables (a b c d : ℝ)
variables (h1 : a + b = 20)
variables (h2 : ab + c + d = 105)
variables (h3 : ad + bc = 225)
variables (h4 : cd = 144)

theorem greatest_value_of_squares : a^2 + b^2 + c^2 + d^2 ≤ 150 := by
  sorry

end greatest_value_of_squares_l166_166982


namespace beijing_time_conversion_l166_166719

-- Define the conversion conditions
def new_clock_hours_in_day : Nat := 10
def new_clock_minutes_per_hour : Nat := 100
def new_clock_time_at_5_beijing_time : Nat := 12 * 60  -- converting 12 noon to minutes


-- Define the problem to prove the corresponding Beijing time 
theorem beijing_time_conversion :
  new_clock_minutes_per_hour * 5 = 500 → 
  new_clock_time_at_5_beijing_time = 720 →
  (720 + 175 * 1.44) = 4 * 60 + 12 :=
by
  intros h1 h2
  sorry

end beijing_time_conversion_l166_166719


namespace find_expression_l166_166589

theorem find_expression (E a : ℝ) 
  (h1 : (E + (3 * a - 8)) / 2 = 69) 
  (h2 : a = 26) : 
  E = 68 :=
sorry

end find_expression_l166_166589


namespace max_b_c_l166_166008

theorem max_b_c (a b c : ℤ) (ha : a > 0) 
  (h1 : a - b + c = 4) 
  (h2 : 4 * a + 2 * b + c = 1) 
  (h3 : (b ^ 2) - 4 * a * c > 0) :
  -3 * a + 2 = -4 := 
sorry

end max_b_c_l166_166008


namespace solution_set_of_inequality_l166_166118

variable {R : Type} [LinearOrderedField R] (f : R → R)

-- Conditions
def monotonically_increasing_on_nonnegatives := 
  ∀ x y : R, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

def odd_function_shifted_one := 
  ∀ x : R, f (-x) = 2 - f (x)

-- The problem
theorem solution_set_of_inequality
  (mono_inc : monotonically_increasing_on_nonnegatives f)
  (odd_shift : odd_function_shifted_one f) :
  {x : R | f (3 * x + 4) + f (1 - x) < 2} = {x : R | x < -5 / 2} :=
by
  sorry

end solution_set_of_inequality_l166_166118


namespace product_of_sequence_l166_166580

theorem product_of_sequence :
  (1 + 1 / 1) * (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) *
  (1 + 1 / 6) * (1 + 1 / 7) * (1 + 1 / 8) = 9 :=
by sorry

end product_of_sequence_l166_166580


namespace yanna_afternoon_baking_l166_166684

noncomputable def butter_cookies_in_afternoon (B : ℕ) : Prop :=
  let biscuits_afternoon := 20
  let butter_cookies_morning := 20
  let biscuits_morning := 40
  (biscuits_afternoon = B + 30) → B = 20

theorem yanna_afternoon_baking (h : butter_cookies_in_afternoon 20) : 20 = 20 :=
by {
  sorry
}

end yanna_afternoon_baking_l166_166684


namespace smallest_t_in_colored_grid_l166_166358

theorem smallest_t_in_colored_grid :
  ∃ (t : ℕ), (t > 0) ∧
  (∀ (coloring : Fin (100*100) → ℕ),
      (∀ (n : ℕ), (∃ (squares : Finset (Fin (100*100))), squares.card ≤ 104 ∧ ∀ x ∈ squares, coloring x = n)) →
      (∃ (rectangle : Finset (Fin (100*100))),
        (rectangle.card = t ∧ (t = 1 ∨ (t = 2 ∨ ∃ (l : ℕ), (l = 12 ∧ rectangle.card = l) ∧ (∃ (c : ℕ), (c = 3 ∧ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∃(s1 s2 s3 : Fin (100*100)), (s1 ∈ rectangle ∧ coloring s1 = a) ∧ (s2 ∈ rectangle ∧ coloring s2 = b) ∧ (s3 ∈ rectangle ∧ coloring s3 = c))))))))) :=
sorry

end smallest_t_in_colored_grid_l166_166358


namespace inequality_correctness_l166_166898

variable (a b : ℝ)
variable (h1 : a < b) (h2 : b < 0)

theorem inequality_correctness : a^2 > ab ∧ ab > b^2 := by
  sorry

end inequality_correctness_l166_166898


namespace more_ones_than_twos_in_digital_roots_l166_166456

/-- Define the digital root (i.e., repeated sum of digits until a single digit). -/
def digitalRoot (n : Nat) : Nat :=
  if n == 0 then 0 else 1 + (n - 1) % 9

/-- Statement of the problem: For numbers 1 to 1,000,000, the count of digital root 1 is higher than the count of digital root 2. -/
theorem more_ones_than_twos_in_digital_roots :
  (Finset.filter (fun n => digitalRoot n = 1) (Finset.range 1000000)).card >
  (Finset.filter (fun n => digitalRoot n = 2) (Finset.range 1000000)).card :=
by
  sorry

end more_ones_than_twos_in_digital_roots_l166_166456


namespace slope_y_intercept_product_l166_166861

theorem slope_y_intercept_product (m b : ℝ) (hm : m = -1/2) (hb : b = 4/5) : -1 < m * b ∧ m * b < 0 :=
by
  sorry

end slope_y_intercept_product_l166_166861


namespace traffic_lights_states_l166_166273

theorem traffic_lights_states (n k : ℕ) : 
  (k ≤ n) → 
  (∃ (ways : ℕ), ways = 3^k * 2^(n - k)) :=
by
  sorry

end traffic_lights_states_l166_166273


namespace mul_exponents_l166_166383

theorem mul_exponents (a : ℝ) : ((-2 * a) ^ 2) * (a ^ 4) = 4 * a ^ 6 := by
  sorry

end mul_exponents_l166_166383


namespace function_graph_intersection_l166_166799

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end function_graph_intersection_l166_166799


namespace find_a_l166_166334

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x + a

theorem find_a (a : ℝ) :
  (∃! x : ℝ, f x a = 0) → (a = -2 ∨ a = 2) :=
sorry

end find_a_l166_166334


namespace triangle_area_inscribed_in_circle_l166_166482

theorem triangle_area_inscribed_in_circle (R : ℝ) 
    (h_pos : R > 0) 
    (h_ratio : ∃ (x : ℝ)(hx : x > 0), 2*x + 5*x + 17*x = 2*π) :
  (∃ (area : ℝ), area = (R^2 / 4)) :=
by
  sorry

end triangle_area_inscribed_in_circle_l166_166482


namespace find_a_values_l166_166619

theorem find_a_values (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 27 * x^3) (h₃ : a - b = 2 * x) :
  a = 3.041 * x ∨ a = -1.041 * x :=
by
  sorry

end find_a_values_l166_166619


namespace not_divisible_by_1955_l166_166232

theorem not_divisible_by_1955 (n : ℤ) : ¬ ∃ k : ℤ, (n^2 + n + 1) = 1955 * k :=
by
  sorry

end not_divisible_by_1955_l166_166232


namespace popsicles_eaten_l166_166518

theorem popsicles_eaten (total_minutes : ℕ) (minutes_per_popsicle : ℕ) (h : total_minutes = 405) (k : minutes_per_popsicle = 12) :
  (total_minutes / minutes_per_popsicle) = 33 :=
by
  sorry

end popsicles_eaten_l166_166518


namespace total_number_of_coins_l166_166924

theorem total_number_of_coins (n : ℕ) (h : 4 * n - 4 = 240) : n^2 = 3721 :=
by
  sorry

end total_number_of_coins_l166_166924


namespace maries_trip_distance_l166_166511

theorem maries_trip_distance (x : ℚ)
  (h1 : x = x / 4 + 15 + x / 6) :
  x = 180 / 7 :=
by
  sorry

end maries_trip_distance_l166_166511


namespace gcd_bc_eq_one_l166_166045

theorem gcd_bc_eq_one (a b c x y : ℕ)
  (h1 : Nat.gcd a b = 120)
  (h2 : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  Nat.gcd b c = 1 :=
by
  sorry

end gcd_bc_eq_one_l166_166045


namespace vessel_base_length_l166_166905

variables (L : ℝ) (edge : ℝ) (W : ℝ) (h : ℝ)
def volume_cube := edge^3
def volume_rise := L * W * h

theorem vessel_base_length :
  (volume_cube 16 = volume_rise L 15 13.653333333333334) →
  L = 20 :=
by sorry

end vessel_base_length_l166_166905


namespace quadratic_average_of_roots_l166_166507

theorem quadratic_average_of_roots (a b c : ℝ) (h_eq : a ≠ 0) (h_b : b = -6) (h_c : c = 3) 
  (discriminant : (b^2 - 4 * a * c) = 12) : 
  (b^2 - 4 * a * c = 12) → ((-b / (2 * a)) / 2 = 1.5) :=
by
  have a_val : a = 2 := sorry
  sorry

end quadratic_average_of_roots_l166_166507


namespace max_product_of_two_integers_sum_2000_l166_166830

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l166_166830


namespace min_value_of_sum_of_squares_l166_166983

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 10) : 
  x^2 + y^2 + z^2 ≥ 100 / 29 :=
sorry

end min_value_of_sum_of_squares_l166_166983


namespace length_reduction_percentage_to_maintain_area_l166_166959

theorem length_reduction_percentage_to_maintain_area
  (L W : ℝ)
  (new_width : ℝ := W * (1 + 28.2051282051282 / 100))
  (new_length : ℝ := L * (1 - 21.9512195121951 / 100))
  (original_area : ℝ := L * W) :
  original_area = new_length * new_width := by
  sorry

end length_reduction_percentage_to_maintain_area_l166_166959


namespace original_number_is_144_l166_166814

theorem original_number_is_144 (A B C : ℕ) (A_digit : A < 10) (B_digit : B < 10) (C_digit : C < 10)
  (h1 : 100 * A + 10 * B + B = 144)
  (h2 : A * B * B = 10 * A + C)
  (h3 : (10 * A + C) % 10 = C) : 100 * A + 10 * B + B = 144 := 
sorry

end original_number_is_144_l166_166814


namespace distinct_x_intercepts_l166_166977

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, (x + 5) * (x^2 + 5 * x - 6) = 0 ↔ x ∈ s :=
by { 
  sorry 
}

end distinct_x_intercepts_l166_166977


namespace problem_l166_166046

/-
A problem involving natural numbers a and b
where:
1. Their sum is 20000
2. One of them (b) is divisible by 5
3. Erasing the units digit of b gives the other number a

We want to prove their difference is 16358
-/

def nat_sum_and_difference (a b : ℕ) : Prop :=
  a + b = 20000 ∧
  b % 5 = 0 ∧
  (b % 10 = 0 ∧ b / 10 = a ∨ b % 10 = 5 ∧ (b - 5) / 10 = a)

theorem problem (a b : ℕ) (h : nat_sum_and_difference a b) : b - a = 16358 := 
  sorry

end problem_l166_166046


namespace count_positive_n_l166_166544

def is_factorable (n : ℕ) : Prop :=
  ∃ a b : ℤ, (a + b = -2) ∧ (a * b = - (n:ℤ))

theorem count_positive_n : 
  (∃ (S : Finset ℕ), S.card = 45 ∧ ∀ n ∈ S, (1 ≤ n ∧ n ≤ 2000) ∧ is_factorable n) :=
by
  -- Placeholder for the proof
  sorry

end count_positive_n_l166_166544


namespace students_speaking_both_languages_l166_166192

theorem students_speaking_both_languages:
  ∀ (total E T N B : ℕ),
    total = 150 →
    E = 55 →
    T = 85 →
    N = 30 →
    (total - N) = 120 →
    (E + T - B) = 120 → B = 20 :=
by
  intros total E T N B h_total h_E h_T h_N h_langs h_equiv
  sorry

end students_speaking_both_languages_l166_166192


namespace calculate_fraction_l166_166364

theorem calculate_fraction : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end calculate_fraction_l166_166364


namespace geometric_sequence_thm_proof_l166_166965

noncomputable def geometric_sequence_thm (a : ℕ → ℤ) : Prop :=
  (∃ r : ℤ, ∃ a₀ : ℤ, ∀ n : ℕ, a n = a₀ * r ^ n) ∧
  (a 2) * (a 10) = 4 ∧
  (a 2) + (a 10) > 0 →
  (a 6) = 2

theorem geometric_sequence_thm_proof (a : ℕ → ℤ) :
  geometric_sequence_thm a :=
  by
  sorry

end geometric_sequence_thm_proof_l166_166965


namespace solve_for_a_l166_166077

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solve_for_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x / ((x + 1) * (x - a)))
  (h_odd : is_odd_function f) :
  a = 1 :=
sorry

end solve_for_a_l166_166077


namespace projections_relationship_l166_166706

theorem projections_relationship (a b r : ℝ) (h : r ≠ 0) :
  (∃ α β : ℝ, a = r * Real.cos α ∧ b = r * Real.cos β ∧ (Real.cos α)^2 + (Real.cos β)^2 = 1) → (a^2 + b^2 = r^2) :=
by
  sorry

end projections_relationship_l166_166706


namespace austin_hours_on_mondays_l166_166021

-- Define the conditions
def earning_per_hour : ℕ := 5
def hours_wednesday : ℕ := 1
def hours_friday : ℕ := 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

-- Define the proof problem
theorem austin_hours_on_mondays (M : ℕ) :
  earning_per_hour * weeks * (M + hours_wednesday + hours_friday) = bicycle_cost → M = 2 :=
by 
  intro h
  sorry

end austin_hours_on_mondays_l166_166021


namespace reef_age_in_decimal_l166_166018

def octal_to_decimal (n: Nat) : Nat :=
  match n with
  | 367 => 7 * (8^0) + 6 * (8^1) + 3 * (8^2)
  | _   => 0  -- Placeholder for other values if needed

theorem reef_age_in_decimal : octal_to_decimal 367 = 247 := by
  sorry

end reef_age_in_decimal_l166_166018


namespace regular_price_of_pony_jeans_l166_166166

-- Define the regular price of fox jeans
def fox_jeans_price := 15

-- Define the given conditions
def pony_discount_rate := 0.18
def total_savings := 9
def total_discount_rate := 0.22

-- State the problem: Prove the regular price of pony jeans
theorem regular_price_of_pony_jeans : 
  ∃ P, P * pony_discount_rate = 3.6 :=
by
  sorry

end regular_price_of_pony_jeans_l166_166166


namespace probability_below_8_l166_166575

theorem probability_below_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) : 
  1 - (p10 + p9 + p8) = 0.40 :=
by 
  rw [h1, h2, h3]
  sorry

end probability_below_8_l166_166575


namespace terminal_side_second_or_third_quadrant_l166_166012

-- Definitions and conditions directly from part a)
def sin (x : ℝ) : ℝ := sorry
def tan (x : ℝ) : ℝ := sorry
def terminal_side_in_quadrant (x : ℝ) (q : ℕ) : Prop := sorry

-- Proving the mathematically equivalent proof
theorem terminal_side_second_or_third_quadrant (x : ℝ) :
  sin x * tan x < 0 →
  (terminal_side_in_quadrant x 2 ∨ terminal_side_in_quadrant x 3) :=
by
  sorry

end terminal_side_second_or_third_quadrant_l166_166012


namespace min_value_fraction_sum_l166_166449

theorem min_value_fraction_sum (p q r a b : ℝ) (hpq : 0 < p) (hq : p < q) (hr : q < r)
  (h_sum : p + q + r = a) (h_prod_sum : p * q + q * r + r * p = b) (h_prod : p * q * r = 48) :
  ∃ (min_val : ℝ), min_val = (1 / p) + (2 / q) + (3 / r) ∧ min_val = 3 / 2 :=
sorry

end min_value_fraction_sum_l166_166449


namespace system_of_equations_abs_diff_l166_166625

theorem system_of_equations_abs_diff 
  (x y m n : ℝ) 
  (h₁ : 2 * x - y = m)
  (h₂ : x + m * y = n)
  (hx : x = 2)
  (hy : y = 1) : 
  |m - n| = 2 :=
by
  sorry

end system_of_equations_abs_diff_l166_166625


namespace non_adjacent_placements_l166_166828

theorem non_adjacent_placements (n : ℕ) : 
  let total_ways := n^2 * (n^2 - 1)
  let adjacent_ways := 2 * n^2 - 2 * n
  (total_ways - adjacent_ways) = n^4 - 3 * n^2 + 2 * n :=
by
  -- Proof is sorted out
  sorry

end non_adjacent_placements_l166_166828


namespace sophia_ate_pie_l166_166958

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end sophia_ate_pie_l166_166958


namespace pipe_B_filling_time_l166_166332

theorem pipe_B_filling_time (T_B : ℝ) 
  (A_filling_time : ℝ := 10) 
  (combined_filling_time: ℝ := 20/3)
  (A_rate : ℝ := 1 / A_filling_time)
  (combined_rate : ℝ := 1 / combined_filling_time) : 
  1 / T_B = combined_rate - A_rate → T_B = 20 := by 
  sorry

end pipe_B_filling_time_l166_166332


namespace lisa_additional_marbles_l166_166594

theorem lisa_additional_marbles (n : ℕ) (f : ℕ) (m : ℕ) (current_marbles : ℕ) : 
  n = 12 ∧ f = n ∧ m = (n * (n + 1)) / 2 ∧ current_marbles = 34 → 
  m - current_marbles = 44 :=
by
  intros
  sorry

end lisa_additional_marbles_l166_166594


namespace quadrilateral_angles_l166_166455

theorem quadrilateral_angles 
  (A B C D : Type) 
  (a d b c : Float)
  (hAD : a = d ∧ d = c) 
  (hBDC_twice_BDA : ∃ x : Float, b = 2 * x) 
  (hBDA_CAD_ratio : ∃ x : Float, d = 2/3 * x) :
  (∃ α β γ δ : Float, 
    α = 75 ∧ 
    β = 135 ∧ 
    γ = 60 ∧ 
    δ = 90) := 
sorry

end quadrilateral_angles_l166_166455


namespace exponent_sum_l166_166979

theorem exponent_sum (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^123 + i^223 + i^323 = -3 * i :=
by
  sorry

end exponent_sum_l166_166979


namespace pipes_fill_cistern_in_12_minutes_l166_166564

noncomputable def time_to_fill_cistern_with_pipes (A_fill : ℝ) (B_fill : ℝ) (C_empty : ℝ) : ℝ :=
  let A_rate := 1 / (12 * 3)          -- Pipe A's rate
  let B_rate := 1 / (8 * 3)           -- Pipe B's rate
  let C_rate := -1 / 24               -- Pipe C's rate
  let combined_rate := A_rate + B_rate - C_rate
  (1 / 3) / combined_rate             -- Time to fill remaining one-third

theorem pipes_fill_cistern_in_12_minutes :
  time_to_fill_cistern_with_pipes 12 8 24 = 12 :=
by
  sorry

end pipes_fill_cistern_in_12_minutes_l166_166564


namespace dot_product_value_l166_166899

variables (a b : ℝ × ℝ)

theorem dot_product_value
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a.1 * b.1 + a.2 * b.2 = -12 :=
sorry

end dot_product_value_l166_166899


namespace price_of_cork_l166_166704

theorem price_of_cork (C : ℝ) 
  (h₁ : ∃ (bottle_with_cork bottle_without_cork : ℝ), bottle_with_cork = 2.10 ∧ bottle_without_cork = C + 2.00 ∧ bottle_with_cork = C + bottle_without_cork) :
  C = 0.05 :=
by
  obtain ⟨bottle_with_cork, bottle_without_cork, hwc, hwoc, ht⟩ := h₁
  sorry

end price_of_cork_l166_166704


namespace smallest_number_divisible_1_to_10_l166_166047

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l166_166047


namespace volume_of_first_bottle_l166_166673

theorem volume_of_first_bottle (V_2 V_3 : ℕ) (V_total : ℕ):
  V_2 = 750 ∧ V_3 = 250 ∧ V_total = 3 * 1000 →
  (V_total - V_2 - V_3) / 1000 = 2 :=
by
  sorry

end volume_of_first_bottle_l166_166673


namespace find_square_side_length_l166_166612

open Nat

def original_square_side_length (s : ℕ) : Prop :=
  let length := s + 8
  let breadth := s + 4
  (2 * (length + breadth)) = 40 → s = 4

theorem find_square_side_length (s : ℕ) : original_square_side_length s := by
  sorry

end find_square_side_length_l166_166612


namespace simplify_expression_l166_166700

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := 
by
  sorry

end simplify_expression_l166_166700


namespace Julie_monthly_salary_l166_166520

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end Julie_monthly_salary_l166_166520


namespace problem_1_problem_2_problem_3_l166_166568

-- Problem 1: Prove that if the inequality |x-1| - |x-2| < a holds for all x in ℝ, then a > 1.
theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a) → a > 1 :=
sorry

-- Problem 2: Prove that if the inequality |x-1| - |x-2| < a has at least one real solution, then a > -1.
theorem problem_2 (a : ℝ) :
  (∃ x : ℝ, |x - 1| - |x - 2| < a) → a > -1 :=
sorry

-- Problem 3: Prove that if the solution set of the inequality |x-1| - |x-2| < a is empty, then a ≤ -1.
theorem problem_3 (a : ℝ) :
  (¬∃ x : ℝ, |x - 1| - |x - 2| < a) → a ≤ -1 :=
sorry

end problem_1_problem_2_problem_3_l166_166568


namespace distance_interval_l166_166270

-- Define the conditions based on the false statements:
variable (d : ℝ)

def false_by_alice : Prop := d < 8
def false_by_bob : Prop := d > 7
def false_by_charlie : Prop := d ≠ 6

theorem distance_interval (h_alice : false_by_alice d) (h_bob : false_by_bob d) (h_charlie : false_by_charlie d) :
  7 < d ∧ d < 8 :=
by
  sorry

end distance_interval_l166_166270


namespace inequality_proof_l166_166205

variables {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (min (a * b) (b * c)) (c * a) ≥ 1) :
  (↑((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) ^ (1 / 3 : ℝ)) ≤ ((a + b + c) / 3) ^ 2 + 1 :=
by
  sorry

end inequality_proof_l166_166205


namespace measure_of_α_l166_166889

variables (α β : ℝ)
-- Condition 1: α and β are complementary angles
def complementary := α + β = 180

-- Condition 2: Half of angle β is 30° less than α
def half_less_30 := α - (1 / 2) * β = 30

-- Theorem: Measure of angle α
theorem measure_of_α (α β : ℝ) (h1 : complementary α β) (h2 : half_less_30 α β) :
  α = 80 :=
by
  sorry

end measure_of_α_l166_166889


namespace power_inequality_l166_166596

open Nat

theorem power_inequality (a b : ℝ) (n : ℕ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a) + (1 / b) = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
  sorry

end power_inequality_l166_166596


namespace no_matching_option_for_fraction_l166_166693

theorem no_matching_option_for_fraction (m n : ℕ) (h : m = 16 ^ 500) : 
  (m / 8 ≠ 8 ^ 499) ∧ 
  (m / 8 ≠ 4 ^ 999) ∧ 
  (m / 8 ≠ 2 ^ 1998) ∧ 
  (m / 8 ≠ 4 ^ 498) ∧ 
  (m / 8 ≠ 2 ^ 1994) := 
by {
  sorry
}

end no_matching_option_for_fraction_l166_166693


namespace find_a_l166_166387

theorem find_a (a : ℝ) (α : ℝ) (h1 : ∃ (y : ℝ), (a, y) = (a, -2))
(h2 : Real.tan (π + α) = 1 / 3) : a = -6 :=
sorry

end find_a_l166_166387


namespace find_x_l166_166107

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 8 :=
by
  sorry

end find_x_l166_166107


namespace range_of_a_l166_166014

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*x + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x a > 0) ↔ a > -3 := 
by sorry

end range_of_a_l166_166014


namespace math_problem_l166_166599

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l166_166599


namespace find_line_through_and_perpendicular_l166_166254

def point (x y : ℝ) := (x, y)

def passes_through (P : ℝ × ℝ) (a b c : ℝ) :=
  a * P.1 + b * P.2 + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) :=
  a1 * a2 + b1 * b2 = 0

theorem find_line_through_and_perpendicular :
  ∃ c : ℝ, passes_through (1, -1) 1 1 c ∧ is_perpendicular 1 (-1) 1 1 → 
  c = 0 :=
by
  sorry

end find_line_through_and_perpendicular_l166_166254


namespace exponent_power_rule_l166_166082

theorem exponent_power_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 :=
by sorry

end exponent_power_rule_l166_166082


namespace determine_a_square_binomial_l166_166227

theorem determine_a_square_binomial (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, ax^2 + 24*x + 9 = (r*x + s)^2) → a = 16 :=
by
  sorry

end determine_a_square_binomial_l166_166227


namespace real_root_bound_l166_166017

noncomputable def P (x : ℝ) (n : ℕ) (ns : List ℕ) : ℝ :=
  1 + x^2 + x^5 + ns.foldr (λ n acc => x^n + acc) 0 + x^2008

theorem real_root_bound (n1 n2 : ℕ) (ns : List ℕ) (x : ℝ) :
  5 < n1 →
  List.Chain (λ a b => a < b) n1 (n2 :: ns) →
  n2 < 2008 →
  P x n1 (n2 :: ns) = 0 →
  x ≤ (1 - Real.sqrt 5) / 2 :=
sorry

end real_root_bound_l166_166017


namespace fraction_increase_l166_166080

-- Define the problem conditions and the proof statement
theorem fraction_increase (m n : ℤ) (hnz : n ≠ 0) (hnnz : n ≠ -1) (h : m < n) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) :=
by sorry

end fraction_increase_l166_166080


namespace election_total_votes_l166_166408

theorem election_total_votes (V: ℝ) (valid_votes: ℝ) (candidate_votes: ℝ) (invalid_rate: ℝ) (candidate_rate: ℝ) :
  candidate_rate = 0.75 →
  invalid_rate = 0.15 →
  candidate_votes = 357000 →
  valid_votes = (1 - invalid_rate) * V →
  candidate_votes = candidate_rate * valid_votes →
  V = 560000 :=
by
  intros candidate_rate_eq invalid_rate_eq candidate_votes_eq valid_votes_eq equation
  sorry

end election_total_votes_l166_166408


namespace unit_digit_smaller_by_four_l166_166751

theorem unit_digit_smaller_by_four (x : ℤ) : x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4 :=
by
  sorry

end unit_digit_smaller_by_four_l166_166751


namespace somu_age_to_father_age_ratio_l166_166695

theorem somu_age_to_father_age_ratio
  (S : ℕ) (F : ℕ)
  (h1 : S = 10)
  (h2 : S - 5 = (1/5) * (F - 5)) :
  S / F = 1 / 3 :=
by
  sorry

end somu_age_to_father_age_ratio_l166_166695


namespace parallel_vectors_y_value_l166_166216

theorem parallel_vectors_y_value (y : ℝ) :
  let a := (2, 3)
  let b := (4, y)
  ∃ y : ℝ, (2 : ℝ) / 4 = 3 / y → y = 6 :=
sorry

end parallel_vectors_y_value_l166_166216


namespace pyramid_value_l166_166645

theorem pyramid_value (a b c d e f : ℕ) (h_b : b = 6) (h_d : d = 20) (h_prod1 : d = b * (20 / b)) (h_prod2 : e = (20 / b) * c) (h_prod3 : f = c * (72 / c)) : a = b * c → a = 54 :=
by 
  -- Assuming the proof would assert the calculations done in the solution.
  sorry

end pyramid_value_l166_166645


namespace cristina_pace_correct_l166_166524

-- Definitions of the conditions
def head_start : ℕ := 30
def nicky_pace : ℕ := 3  -- meters per second
def time_for_catch_up : ℕ := 15  -- seconds

-- Distance covers by Nicky
def nicky_distance : ℕ := nicky_pace * time_for_catch_up

-- Total distance covered by Cristina to catch up Nicky
def cristina_distance : ℕ := nicky_distance + head_start

-- Cristina's pace
def cristina_pace : ℕ := cristina_distance / time_for_catch_up

-- Theorem statement
theorem cristina_pace_correct : cristina_pace = 5 := by 
  sorry

end cristina_pace_correct_l166_166524


namespace relationship_x_y_l166_166595

theorem relationship_x_y (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := 
by 
  sorry

end relationship_x_y_l166_166595


namespace intersection_M_N_l166_166215

def I : Set ℤ := {0, -1, -2, -3, -4}
def M : Set ℤ := {0, -1, -2}
def N : Set ℤ := {0, -3, -4}

theorem intersection_M_N : M ∩ N = {0} := 
by 
  sorry

end intersection_M_N_l166_166215


namespace area_ratio_independent_l166_166128

-- Definitions related to the problem
variables (AB BC CD : ℝ) (e f g : ℝ)

-- Let the lengths be defined as follows
def AB_def : Prop := AB = 2 * e
def BC_def : Prop := BC = 2 * f
def CD_def : Prop := CD = 2 * g

-- Let the areas be defined as follows
def area_quadrilateral (e f g : ℝ) : ℝ :=
  2 * (e + f) * (f + g)

def area_enclosed (e f g : ℝ) : ℝ :=
  (e + f + g) ^ 2 + f ^ 2 - e ^ 2 - g ^ 2

-- Prove the ratio is 2 / π
theorem area_ratio_independent (e f g : ℝ) (h1 : AB_def AB e)
  (h2 : BC_def BC f) (h3 : CD_def CD g) :
  (area_quadrilateral e f g) / ((area_enclosed e f g) * (π / 2)) = 2 / π :=
by
  sorry

end area_ratio_independent_l166_166128
