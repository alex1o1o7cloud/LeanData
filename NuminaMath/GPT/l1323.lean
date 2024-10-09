import Mathlib

namespace age_discrepancy_l1323_132316

theorem age_discrepancy (R G M F A : ℕ)
  (hR : R = 12)
  (hG : G = 7 * R)
  (hM : M = G / 2)
  (hF : F = M + 5)
  (hA : A = G - 8)
  (hDiff : A - F = 10) :
  false :=
by
  -- proofs and calculations leading to contradiction go here
  sorry

end age_discrepancy_l1323_132316


namespace qt_q_t_neq_2_l1323_132397

theorem qt_q_t_neq_2 (q t : ℕ) (hq : 0 < q) (ht : 0 < t) : q * t + q + t ≠ 2 :=
  sorry

end qt_q_t_neq_2_l1323_132397


namespace find_t_l1323_132301

theorem find_t
  (x y t : ℝ)
  (h1 : 2 ^ x = t)
  (h2 : 5 ^ y = t)
  (h3 : 1 / x + 1 / y = 2)
  (h4 : t ≠ 1) : 
  t = Real.sqrt 10 := 
by
  sorry

end find_t_l1323_132301


namespace problem_1_problem_2_l1323_132330

-- Definitions for sets A and B
def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 6
def B (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Problem (1): What is A ∩ B when m = 3
theorem problem_1 : ∀ (x : ℝ), A x → B x 3 → (-1 ≤ x ∧ x ≤ 4) := by
  intro x hA hB
  sorry

-- Problem (2): What is the range of m if A ⊆ B and m > 0
theorem problem_2 (m : ℝ) : m > 0 → (∀ x, A x → B x m) → (m ≥ 5) := by
  intros hm hAB
  sorry

end problem_1_problem_2_l1323_132330


namespace handshakes_at_gathering_l1323_132374

def total_handshakes (num_couples : ℕ) (exceptions : ℕ) : ℕ :=
  let num_people := 2 * num_couples
  let handshakes_per_person := num_people - exceptions - 1
  num_people * handshakes_per_person / 2

theorem handshakes_at_gathering : total_handshakes 6 2 = 54 := by
  sorry

end handshakes_at_gathering_l1323_132374


namespace speed_of_boat_in_still_water_l1323_132345

variable (V_b V_s t_up t_down : ℝ)

theorem speed_of_boat_in_still_water (h1 : t_up = 2 * t_down)
  (h2 : V_s = 18) 
  (h3 : ∀ d : ℝ, d = (V_b - V_s) * t_up ∧ d = (V_b + V_s) * t_down) : V_b = 54 :=
sorry

end speed_of_boat_in_still_water_l1323_132345


namespace product_of_possible_values_l1323_132338

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end product_of_possible_values_l1323_132338


namespace eval_expr_at_x_eq_neg6_l1323_132358

-- Define the given condition
def x : ℤ := -4

-- Define the expression to be simplified and evaluated
def expr (x y : ℤ) : ℤ := ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x)

-- The theorem stating the result of the evaluated expression
theorem eval_expr_at_x_eq_neg6 (y : ℤ) : expr (-4) y = -6 := 
by
  sorry

end eval_expr_at_x_eq_neg6_l1323_132358


namespace paul_money_last_weeks_l1323_132392

theorem paul_money_last_weeks (a b c: ℕ) (h1: a = 68) (h2: b = 13) (h3: c = 9) : 
  (a + b) / c = 9 := 
by 
  sorry

end paul_money_last_weeks_l1323_132392


namespace tank_capacity_l1323_132346

theorem tank_capacity (x : ℝ) (h : (5/12) * x = 150) : x = 360 :=
by
  sorry

end tank_capacity_l1323_132346


namespace ratio_of_books_on_each_table_l1323_132384

-- Define the conditions
variables (number_of_tables number_of_books : ℕ)
variables (R : ℕ) -- Ratio we need to find

-- State the conditions
def conditions := (number_of_tables = 500) ∧ (number_of_books = 100000)

-- Mathematical Problem Statement
theorem ratio_of_books_on_each_table (h : conditions number_of_tables number_of_books) :
    100000 = 500 * R → R = 200 :=
by
  sorry

end ratio_of_books_on_each_table_l1323_132384


namespace fourth_root_12960000_eq_60_l1323_132312

theorem fourth_root_12960000_eq_60 :
  (6^4 = 1296) →
  (10^4 = 10000) →
  (60^4 = 12960000) →
  (Real.sqrt (Real.sqrt 12960000) = 60) := 
by
  intros h1 h2 h3
  sorry

end fourth_root_12960000_eq_60_l1323_132312


namespace regular_polygon_perimeter_is_28_l1323_132326

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l1323_132326


namespace average_score_for_entire_class_l1323_132379

def total_students : ℕ := 100
def assigned_day_percentage : ℝ := 0.70
def make_up_day_percentage : ℝ := 0.30
def assigned_day_avg_score : ℝ := 65
def make_up_day_avg_score : ℝ := 95

theorem average_score_for_entire_class :
  (assigned_day_percentage * total_students * assigned_day_avg_score + make_up_day_percentage * total_students * make_up_day_avg_score) / total_students = 74 := by
  sorry

end average_score_for_entire_class_l1323_132379


namespace problem_eval_expression_l1323_132390

theorem problem_eval_expression :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end problem_eval_expression_l1323_132390


namespace roots_cubic_polynomial_l1323_132329

theorem roots_cubic_polynomial (a b c : ℝ) 
  (h1 : a^3 - 2*a - 2 = 0) 
  (h2 : b^3 - 2*b - 2 = 0) 
  (h3 : c^3 - 2*c - 2 = 0) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -18 :=
by
  sorry

end roots_cubic_polynomial_l1323_132329


namespace quadratic_roots_l1323_132394

theorem quadratic_roots (m n p : ℕ) (h : m.gcd p = 1) 
  (h1 : 3 * m^2 - 8 * m * p + p^2 = p^2 * n) : n = 13 :=
by sorry

end quadratic_roots_l1323_132394


namespace wall_length_l1323_132399

theorem wall_length (s : ℕ) (w : ℕ) (a_ratio : ℕ) (A_mirror : ℕ) (A_wall : ℕ) (L : ℕ) 
  (hs : s = 24) (hw : w = 42) (h_ratio : a_ratio = 2) 
  (hA_mirror : A_mirror = s * s) 
  (hA_wall : A_wall = A_mirror * a_ratio) 
  (h_area : A_wall = w * L) : L = 27 :=
  sorry

end wall_length_l1323_132399


namespace base_number_is_five_l1323_132382

theorem base_number_is_five (x k : ℝ) (h1 : x^k = 5) (h2 : x^(2 * k + 2) = 400) : x = 5 :=
by
  sorry

end base_number_is_five_l1323_132382


namespace inequality_integral_ln_bounds_l1323_132396

-- Define the conditions
variables (x a : ℝ)
variables (hx : 0 < x) (ha : x < a)

-- First part: inequality involving integral
theorem inequality_integral (hx : 0 < x) (ha : x < a) :
  (2 * x / a) < (∫ t in a - x..a + x, 1 / t) ∧ (∫ t in a - x..a + x, 1 / t) < x * (1 / (a + x) + 1 / (a - x)) :=
sorry

-- Second part: to prove 0.68 < ln(2) < 0.71 using the result of the first part
theorem ln_bounds :
  0.68 < Real.log 2 ∧ Real.log 2 < 0.71 :=
sorry

end inequality_integral_ln_bounds_l1323_132396


namespace valid_square_numbers_l1323_132373

noncomputable def is_valid_number (N P Q : ℕ) (q : ℕ) : Prop :=
  N = P * 10^q + Q ∧ N = 2 * P * Q

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem valid_square_numbers : 
  ∀ (N : ℕ), (∃ (P Q : ℕ) (q : ℕ), is_valid_number N P Q q) → is_perfect_square N :=
sorry

end valid_square_numbers_l1323_132373


namespace solve_fraction_equation_l1323_132313

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 5 * x - 14) - 1 / (x^2 - 15 * x - 18) = 0) →
  x = 2 ∨ x = -9 ∨ x = 6 ∨ x = -3 :=
sorry

end solve_fraction_equation_l1323_132313


namespace janice_typing_proof_l1323_132303

noncomputable def janice_typing : Prop :=
  let initial_speed := 6
  let error_speed := 8
  let corrected_speed := 5
  let typing_duration_initial := 20
  let typing_duration_corrected := 15
  let erased_sentences := 40
  let typing_duration_after_lunch := 18
  let total_sentences_end_of_day := 536

  let sentences_initial_typing := typing_duration_initial * error_speed
  let sentences_post_error_typing := typing_duration_corrected * initial_speed
  let sentences_final_typing := typing_duration_after_lunch * corrected_speed

  let sentences_total_typed := sentences_initial_typing + sentences_post_error_typing - erased_sentences + sentences_final_typing

  let sentences_started_with := total_sentences_end_of_day - sentences_total_typed

  sentences_started_with = 236

theorem janice_typing_proof : janice_typing := by
  sorry

end janice_typing_proof_l1323_132303


namespace age_difference_in_decades_l1323_132355

-- Declare the ages of x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the condition
def age_condition (x y z : ℝ) : Prop := x + y = y + z + 18

-- The proof problem statement
theorem age_difference_in_decades (h : age_condition x y z) : (x - z) / 10 = 1.8 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end age_difference_in_decades_l1323_132355


namespace range_of_sine_l1323_132398

theorem range_of_sine {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x ≥ Real.sqrt 2 / 2) :
  Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4 :=
by
  sorry

end range_of_sine_l1323_132398


namespace equilateral_sector_area_l1323_132342

noncomputable def area_of_equilateral_sector (r : ℝ) : ℝ :=
  if h : r = r then (1/2) * r^2 * 1 else 0

theorem equilateral_sector_area (r : ℝ) : r = 2 → area_of_equilateral_sector r = 2 :=
by
  intros hr
  rw [hr]
  unfold area_of_equilateral_sector
  split_ifs
  · norm_num
  · contradiction

end equilateral_sector_area_l1323_132342


namespace soap_bars_problem_l1323_132333

theorem soap_bars_problem :
  ∃ (N : ℤ), 200 < N ∧ N < 300 ∧ 2007 % N = 5 :=
sorry

end soap_bars_problem_l1323_132333


namespace max_value_of_quadratic_l1323_132300

-- Define the quadratic function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the main theorem of finding the maximum value
theorem max_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 11 := sorry

end max_value_of_quadratic_l1323_132300


namespace find_a_of_extreme_value_at_one_l1323_132376

-- Define the function f(x) = x^3 - a * x
def f (x a : ℝ) : ℝ := x^3 - a * x
  
-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 3 * x^2 - a

-- The theorem statement: for f(x) having an extreme value at x = 1, the corresponding a must be 3
theorem find_a_of_extreme_value_at_one (a : ℝ) : 
  (f' 1 a = 0) ↔ (a = 3) :=
by
  sorry

end find_a_of_extreme_value_at_one_l1323_132376


namespace find_x_value_l1323_132315

theorem find_x_value (x : ℝ) (h1 : x^2 + x = 6) (h2 : x^2 - 2 = 1) : x = 2 := sorry

end find_x_value_l1323_132315


namespace find_k_values_l1323_132348

theorem find_k_values :
    ∀ (k : ℚ),
    (∀ (a b : ℚ), (5 * a^2 + 7 * a + k = 0) ∧ (5 * b^2 + 7 * b + k = 0) ∧ |a - b| = a^2 + b^2 → k = 21 / 25 ∨ k = -21 / 25) :=
by
  sorry

end find_k_values_l1323_132348


namespace average_mark_of_excluded_students_l1323_132344

theorem average_mark_of_excluded_students (N A E A_R A_E : ℝ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hAR : A_R = 90) 
  (h_eq : N * A - E * A_E = (N - E) * A_R) : 
  A_E = 40 := 
by 
  sorry

end average_mark_of_excluded_students_l1323_132344


namespace min_sum_of_3_digit_numbers_l1323_132387

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_3_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999

theorem min_sum_of_3_digit_numbers : 
  ∃ (a b c : ℕ), 
    a ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    b ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    c ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    a + b = c ∧ 
    a + b + c = 459 := 
sorry

end min_sum_of_3_digit_numbers_l1323_132387


namespace part1_part2_l1323_132334

-- Part (1)
theorem part1 (a : ℝ) (A B : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hB : B = { x : ℝ | x^2 - a * x + a - 1 = 0 }) 
  (hUnion : A ∪ B = A) : 
  a = 2 ∨ a = 3 := 
sorry

-- Part (2)
theorem part2 (m : ℝ) (A C : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hC : C = { x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 5 = 0 }) 
  (hInter : A ∩ C = C) : 
  m ∈ Set.Iic (-3) := 
sorry

end part1_part2_l1323_132334


namespace four_digit_number_exists_l1323_132368

-- Definitions corresponding to the conditions in the problem
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def follows_scheme (n : ℕ) (d : ℕ) : Prop :=
  -- Placeholder for the scheme condition
  sorry

-- The Lean statement for the proof problem
theorem four_digit_number_exists :
  ∃ n d1 d2 : ℕ, is_four_digit_number n ∧ follows_scheme n d1 ∧ follows_scheme n d2 ∧ 
  (n = 1014 ∨ n = 1035 ∨ n = 1512) :=
by {
  -- Placeholder for proof steps
  sorry
}

end four_digit_number_exists_l1323_132368


namespace eccentricity_of_hyperbola_l1323_132378

noncomputable def hyperbola_eccentricity : ℝ → ℝ → ℝ → ℝ
| p, a, b => 
  let c := p / 2
  let e := c / a
  have h₁ : 9 * e^2 - 12 * e^2 / (e^2 - 1) = 1 := sorry
  e

theorem eccentricity_of_hyperbola (p a b : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity p a b = (Real.sqrt 7 + 2) / 3 :=
sorry

end eccentricity_of_hyperbola_l1323_132378


namespace sequence_general_term_l1323_132323

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, a n = S n - S (n-1) :=
by
  -- The proof will be filled in here
  sorry

end sequence_general_term_l1323_132323


namespace count_divisible_2_3_or_5_lt_100_l1323_132319
-- We need the Mathlib library for general mathematical functions

-- The main theorem statement
theorem count_divisible_2_3_or_5_lt_100 : 
  let A2 := Nat.floor (100 / 2)
  let A3 := Nat.floor (100 / 3)
  let A5 := Nat.floor (100 / 5)
  let A23 := Nat.floor (100 / 6)
  let A25 := Nat.floor (100 / 10)
  let A35 := Nat.floor (100 / 15)
  let A235 := Nat.floor (100 / 30)
  (A2 + A3 + A5 - A23 - A25 - A35 + A235) = 74 :=
by
  sorry

end count_divisible_2_3_or_5_lt_100_l1323_132319


namespace cleaning_time_if_anne_doubled_l1323_132357

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l1323_132357


namespace ellipse_circle_parallelogram_condition_l1323_132369

theorem ellipse_circle_parallelogram_condition
  (a b : ℝ)
  (C₀ : ∀ x y : ℝ, x^2 + y^2 = 1)
  (C₁ : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h : a > 0 ∧ b > 0 ∧ a > b) :
  1 / a^2 + 1 / b^2 = 1 := by
  sorry

end ellipse_circle_parallelogram_condition_l1323_132369


namespace problem1_problem2_l1323_132328

-- Definitions of sets A and B
def A : Set ℝ := { x | x > 1 }
def B (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Problem 1:
theorem problem1 (a : ℝ) : B a ⊆ A → 1 ≤ a :=
  sorry

-- Problem 2:
theorem problem2 (a : ℝ) : (A ∩ B a).Nonempty → 0 < a :=
  sorry

end problem1_problem2_l1323_132328


namespace total_chairs_taken_l1323_132365

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l1323_132365


namespace subtracted_value_l1323_132302

theorem subtracted_value (N V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 :=
by
  sorry

end subtracted_value_l1323_132302


namespace fraction_of_girls_l1323_132304

variable (total_students : ℕ) (number_of_boys : ℕ)

theorem fraction_of_girls (h1 : total_students = 160) (h2 : number_of_boys = 60) :
    (total_students - number_of_boys) / total_students = 5 / 8 := by
  sorry

end fraction_of_girls_l1323_132304


namespace percentage_of_females_l1323_132395

theorem percentage_of_females (total_passengers : ℕ)
  (first_class_percentage : ℝ) (male_fraction_first_class : ℝ)
  (females_coach_class : ℕ) (h1 : total_passengers = 120)
  (h2 : first_class_percentage = 0.10)
  (h3 : male_fraction_first_class = 1/3)
  (h4 : females_coach_class = 40) :
  (females_coach_class + (first_class_percentage * total_passengers - male_fraction_first_class * (first_class_percentage * total_passengers))) / total_passengers * 100 = 40 :=
by
  sorry

end percentage_of_females_l1323_132395


namespace silverware_probability_l1323_132380

-- Definitions based on the problem conditions
def total_silverware : ℕ := 8 + 10 + 7
def total_combinations : ℕ := Nat.choose total_silverware 4

def fork_combinations : ℕ := Nat.choose 8 2
def spoon_combinations : ℕ := Nat.choose 10 1
def knife_combinations : ℕ := Nat.choose 7 1

def favorable_combinations : ℕ := fork_combinations * spoon_combinations * knife_combinations
def specific_combination_probability : ℚ := favorable_combinations / total_combinations

-- The statement to prove the given probability
theorem silverware_probability :
  specific_combination_probability = 392 / 2530 :=
by
  sorry

end silverware_probability_l1323_132380


namespace range_of_k_l1323_132350

theorem range_of_k :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) :=
by
  sorry

end range_of_k_l1323_132350


namespace circle_problem_l1323_132381

theorem circle_problem (P : ℝ × ℝ) (QR : ℝ) (S : ℝ × ℝ) (k : ℝ)
  (h1 : P = (5, 12))
  (h2 : QR = 5)
  (h3 : S = (0, k))
  (h4 : dist (0,0) P = 13) -- OP = 13 from the origin to point P
  (h5 : dist (0,0) S = 8) -- OQ = 8 from the origin to point S
: k = 8 ∨ k = -8 :=
by sorry

end circle_problem_l1323_132381


namespace division_problem_l1323_132367

theorem division_problem : 75 / 0.05 = 1500 := 
  sorry

end division_problem_l1323_132367


namespace gcd_7854_13843_l1323_132353

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := 
  sorry

end gcd_7854_13843_l1323_132353


namespace simple_interest_amount_l1323_132339

noncomputable def simple_interest (P r t : ℝ) : ℝ := (P * r * t) / 100
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r / 100)^t - P

theorem simple_interest_amount:
  ∀ (P : ℝ), compound_interest P 5 2 = 51.25 → simple_interest P 5 2 = 50 :=
by
  intros P h
  -- this is where the proof would go
  sorry

end simple_interest_amount_l1323_132339


namespace set_intersection_complement_l1323_132324
open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}
def comp_T : Set ℕ := U \ T

theorem set_intersection_complement :
  S ∩ comp_T = {1, 5} := by
  sorry

end set_intersection_complement_l1323_132324


namespace inequality_solution_set_range_of_a_l1323_132354

noncomputable def f (x : ℝ) := abs (2 * x - 1) - abs (x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x > 0 } = { x : ℝ | x < -1 / 3 ∨ x > 3 } :=
sorry

theorem range_of_a (x0 : ℝ) (h : f x0 + 2 * a ^ 2 < 4 * a) :
  -1 / 2 < a ∧ a < 5 / 2 :=
sorry

end inequality_solution_set_range_of_a_l1323_132354


namespace max_profit_thousand_rubles_l1323_132349

theorem max_profit_thousand_rubles :
  ∃ x y : ℕ, 
    (80 * x + 100 * y = 2180) ∧ 
    (10 * x + 70 * y ≤ 700) ∧ 
    (23 * x + 40 * y ≤ 642) := 
by
  -- proof goes here
  sorry

end max_profit_thousand_rubles_l1323_132349


namespace sum_in_base4_eq_in_base5_l1323_132363

def base4_to_base5 (n : ℕ) : ℕ := sorry -- Placeholder for the conversion function

theorem sum_in_base4_eq_in_base5 :
  base4_to_base5 (203 + 112 + 321) = 2222 := 
sorry

end sum_in_base4_eq_in_base5_l1323_132363


namespace minimize_travel_time_l1323_132331

-- Definitions and conditions
def grid_size : ℕ := 7
def mid_point : ℕ := (grid_size + 1) / 2
def is_meeting_point (p : ℕ × ℕ) : Prop := 
  p = (mid_point, mid_point)

-- Main theorem statement to be proven
theorem minimize_travel_time : 
  ∃ (p : ℕ × ℕ), is_meeting_point p ∧
  (∀ (q : ℕ × ℕ), is_meeting_point q → p = q) :=
sorry

end minimize_travel_time_l1323_132331


namespace sum_of_constants_l1323_132371

theorem sum_of_constants (c d : ℝ) (h₁ : 16 = 2 * 4 + c) (h₂ : 16 = 4 * 4 + d) : c + d = 8 := by
  sorry

end sum_of_constants_l1323_132371


namespace minimum_value_l1323_132347

variable (a b : ℝ)
variable (ab_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (circle1 : ∀ x y, x^2 + y^2 + 2 * a * x + a^2 - 9 = 0)
variable (circle2 : ∀ x y, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0)
variable (centers_distance : a^2 + 4 * b^2 = 16)

theorem minimum_value :
  (4 / a^2 + 1 / b^2) = 1 := sorry

end minimum_value_l1323_132347


namespace evaluate_expressions_for_pos_x_l1323_132327

theorem evaluate_expressions_for_pos_x :
  (∀ x : ℝ, x > 0 → 6^x * x^3 = 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (3 * x)^(3 * x) ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → 3^x * x^6 ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (6 * x)^x ≠ 6^x * x^3) →
  ∃ n : ℕ, n = 1 := 
by
  sorry

end evaluate_expressions_for_pos_x_l1323_132327


namespace relative_error_approximation_l1323_132309

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  (1 / (1 + y) - (1 - y)) / (1 / (1 + y)) = y^2 :=
by
  sorry

end relative_error_approximation_l1323_132309


namespace value_of_f_two_l1323_132306

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_two :
  (∀ x : ℝ, f (1 / x) = 1 / (x + 1)) → f 2 = 2 / 3 := by
  intro h
  -- The proof would go here
  sorry

end value_of_f_two_l1323_132306


namespace pieces_equality_l1323_132336

-- Define the pieces of chocolate and their areas.
def piece1_area : ℝ := 6 -- Area of triangle EBC
def piece2_area : ℝ := 6 -- Area of triangle AEC
def piece3_area : ℝ := 6 -- Area of polygon AHGFD
def piece4_area : ℝ := 6 -- Area of polygon CFGH

-- State the problem: proving the equality of the areas.
theorem pieces_equality : piece1_area = piece2_area ∧ piece2_area = piece3_area ∧ piece3_area = piece4_area :=
by
  sorry

end pieces_equality_l1323_132336


namespace find_ab_pairs_l1323_132393

theorem find_ab_pairs (a b s : ℕ) (a_pos : a > 0) (b_pos : b > 0) (s_gt_one : s > 1) :
  (a = 2^s ∧ b = 2^(2*s) - 1) ↔
  (∃ p k : ℕ, Prime p ∧ (a^2 + b + 1 = p^k) ∧
   (a^2 + b + 1 ∣ b^2 - a^3 - 1) ∧
   ¬ (a^2 + b + 1 ∣ (a + b - 1)^2)) :=
sorry

end find_ab_pairs_l1323_132393


namespace target_hit_prob_l1323_132318

-- Probability definitions for A, B, and C
def prob_A := 1 / 2
def prob_B := 1 / 3
def prob_C := 1 / 4

-- Theorem to prove the probability of the target being hit
theorem target_hit_prob :
  (1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)) = 3 / 4 :=
by
  sorry

end target_hit_prob_l1323_132318


namespace cuboid_face_areas_l1323_132364

-- Conditions
variables (a b c S : ℝ)
-- Surface area of the sphere condition
theorem cuboid_face_areas 
  (h1 : a * b = 6) 
  (h2 : b * c = 10) 
  (h3 : a^2 + b^2 + c^2 = 76) 
  (h4 : 4 * π * 38 = 152 * π) :
  a * c = 15 :=
by 
  -- Prove that the solution matches the conclusion
  sorry

end cuboid_face_areas_l1323_132364


namespace find_x_l1323_132335

/-- Let r be the result of doubling both the base and exponent of a^b, 
and b does not equal to 0. If r equals the product of a^b by x^b,
then x equals 4a. -/
theorem find_x (a b x: ℝ) (h₁ : b ≠ 0) (h₂ : (2*a)^(2*b) = a^b * x^b) : x = 4*a := 
  sorry

end find_x_l1323_132335


namespace find_distance_BC_l1323_132311

variables {d_AB d_AC d_BC : ℝ}

theorem find_distance_BC
  (h1 : d_AB = d_AC + d_BC - 200)
  (h2 : d_AC = d_AB + d_BC - 300) :
  d_BC = 250 := 
sorry

end find_distance_BC_l1323_132311


namespace plums_total_correct_l1323_132337

-- Define the number of plums picked by Melanie, Dan, and Sally
def plums_melanie : ℕ := 4
def plums_dan : ℕ := 9
def plums_sally : ℕ := 3

-- Define the total number of plums picked
def total_plums : ℕ := plums_melanie + plums_dan + plums_sally

-- Theorem stating the total number of plums picked
theorem plums_total_correct : total_plums = 16 := by
  sorry

end plums_total_correct_l1323_132337


namespace eric_bike_speed_l1323_132375

def swim_distance : ℝ := 0.5
def swim_speed : ℝ := 1
def run_distance : ℝ := 2
def run_speed : ℝ := 8
def bike_distance : ℝ := 12
def total_time_limit : ℝ := 2

theorem eric_bike_speed :
  (swim_distance / swim_speed) + (run_distance / run_speed) + (bike_distance / (48/5)) < total_time_limit :=
by
  sorry

end eric_bike_speed_l1323_132375


namespace f_one_value_l1323_132388

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h_f_defined : ∀ x, x > 0 → ∃ y, f x = y
axiom h_f_strict_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom h_f_eq : ∀ x, x > 0 → f x * f (f x + 1/x) = 1

theorem f_one_value : f 1 = (1 + Real.sqrt 5) / 2 := 
by
  sorry

end f_one_value_l1323_132388


namespace circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l1323_132370

theorem circle_equation_tangent_y_axis_center_on_line_chord_length_condition :
  ∃ (x₀ y₀ r : ℝ), 
  (x₀ - 3 * y₀ = 0) ∧ 
  (r = |3 * y₀|) ∧ 
  ((x₀ + 3)^2 + (y₀ - 1)^2 = r^2 ∨ (x₀ - 3)^2 + (y₀ + 1)^2 = r^2) :=
sorry

end circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l1323_132370


namespace average_rate_of_change_l1323_132314

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α)
variable (x x₁ : α)
variable (h₁ : x ≠ x₁)

theorem average_rate_of_change : 
  (f x₁ - f x) / (x₁ - x) = (f x₁ - f x) / (x₁ - x) :=
by
  sorry

end average_rate_of_change_l1323_132314


namespace older_brother_pocket_money_l1323_132386

-- Definitions of the conditions
axiom sum_of_pocket_money (O Y : ℕ) : O + Y = 12000
axiom older_brother_more (O Y : ℕ) : O = Y + 1000

-- The statement to prove
theorem older_brother_pocket_money (O Y : ℕ) (h1 : O + Y = 12000) (h2 : O = Y + 1000) : O = 6500 :=
by
  exact sorry  -- Placeholder for the proof

end older_brother_pocket_money_l1323_132386


namespace race_outcomes_l1323_132325

-- Definition of participants
inductive Participant
| Abe 
| Bobby
| Charles
| Devin
| Edwin
| Frank
deriving DecidableEq

open Participant

def num_participants : ℕ := 6

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Proving the number of different 1st-2nd-3rd outcomes
theorem race_outcomes : factorial 6 / factorial 3 = 120 := by
  sorry

end race_outcomes_l1323_132325


namespace bowls_total_marbles_l1323_132307

theorem bowls_total_marbles :
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  C1 = 450 ∧ C3 = 225 ∧ (C1 + C2 + C3 = 1275) := 
by
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  have hC1 : C1 = 450 := by norm_num
  have hC3 : C3 = 225 := by norm_num
  have hTotal : C1 + C2 + C3 = 1275 := by norm_num
  exact ⟨hC1, hC3, hTotal⟩

end bowls_total_marbles_l1323_132307


namespace find_remainder_l1323_132356

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end find_remainder_l1323_132356


namespace sum_of_first_five_primes_with_units_digit_3_l1323_132351

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l1323_132351


namespace rhombus_area_l1323_132389

def diagonal1 : ℝ := 24
def diagonal2 : ℝ := 16

theorem rhombus_area : 0.5 * diagonal1 * diagonal2 = 192 :=
by
  sorry

end rhombus_area_l1323_132389


namespace find_integer_of_divisors_l1323_132305

theorem find_integer_of_divisors:
  ∃ (N : ℕ), (∀ (l m n : ℕ), N = (2^l) * (3^m) * (5^n) → 
  (2^120) * (3^60) * (5^90) = (2^l * 3^m * 5^n)^( ((l+1)*(m+1)*(n+1)) / 2 ) ) → 
  N = 18000 :=
sorry

end find_integer_of_divisors_l1323_132305


namespace linear_term_coefficient_l1323_132385

theorem linear_term_coefficient : (x - 1) * (1 / x + x) ^ 6 = a + b * x + c * x^2 + d * x^3 + e * x^4 + f * x^5 + g * x^6 →
  b = 20 :=
by
  sorry

end linear_term_coefficient_l1323_132385


namespace new_volume_is_correct_l1323_132372

variable (l w h : ℝ)

-- Conditions given in the problem
axiom volume : l * w * h = 4320
axiom surface_area : 2 * (l * w + w * h + h * l) = 1704
axiom edge_sum : 4 * (l + w + h) = 208

-- The proposition we need to prove:
theorem new_volume_is_correct : (l + 2) * (w + 2) * (h + 2) = 6240 :=
by
  -- Placeholder for the actual proof
  sorry

end new_volume_is_correct_l1323_132372


namespace mixture_replacement_l1323_132341

theorem mixture_replacement:
  ∀ (A B x : ℝ),
    A = 64 →
    B = A / 4 →
    (A - (4/5) * x) / (B + (4/5) * x) = 2 / 3 →
    x = 40 :=
by
  intros A B x hA hB hRatio
  sorry

end mixture_replacement_l1323_132341


namespace menkara_index_card_area_l1323_132352

theorem menkara_index_card_area :
  ∀ (length width: ℕ), 
  length = 5 → width = 7 → (length - 2) * width = 21 → 
  (length * (width - 2) = 25) :=
by
  intros length width h_length h_width h_area
  sorry

end menkara_index_card_area_l1323_132352


namespace trig_identity_l1323_132321

theorem trig_identity (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 2 / 3) : 
  Real.cos (2 * α + Real.pi / 3) = -1 / 9 :=
by
  sorry

end trig_identity_l1323_132321


namespace correct_statements_l1323_132359

variable (a b : ℝ)

theorem correct_statements (hab : a * b > 0) :
  (|a + b| > |a| ∧ |a + b| > |a - b|) ∧ (¬ (|a + b| < |b|)) ∧ (¬ (|a + b| < |a - b|)) :=
by
  -- The proof is omitted as per instructions
  sorry

end correct_statements_l1323_132359


namespace field_trip_classrooms_count_l1323_132343

variable (students : ℕ) (seats_per_bus : ℕ) (number_of_buses : ℕ) (total_classrooms : ℕ)

def fieldTrip 
    (students := 58)
    (seats_per_bus := 2)
    (number_of_buses := 29)
    (total_classrooms := 2) : Prop :=
  students = seats_per_bus * number_of_buses  ∧ total_classrooms = students / (students / total_classrooms)

theorem field_trip_classrooms_count : fieldTrip := by
  -- Proof goes here
  sorry

end field_trip_classrooms_count_l1323_132343


namespace age_problem_l1323_132377

theorem age_problem (x y : ℕ) 
  (h1 : 3 * x = 4 * y) 
  (h2 : 3 * y - x = 140) : x = 112 ∧ y = 84 := 
by 
  sorry

end age_problem_l1323_132377


namespace star_running_back_yardage_l1323_132322

-- Definitions
def total_yardage : ℕ := 150
def catching_passes_yardage : ℕ := 60
def running_yardage (total_yardage catching_passes_yardage : ℕ) : ℕ :=
  total_yardage - catching_passes_yardage

-- Statement to prove
theorem star_running_back_yardage :
  running_yardage total_yardage catching_passes_yardage = 90 := 
sorry

end star_running_back_yardage_l1323_132322


namespace find_C_share_l1323_132320

-- Definitions
variable (A B C : ℝ)
variable (H1 : A + B + C = 585)
variable (H2 : 4 * A = 6 * B)
variable (H3 : 6 * B = 3 * C)

-- Problem statement
theorem find_C_share (A B C : ℝ) (H1 : A + B + C = 585) (H2 : 4 * A = 6 * B) (H3 : 6 * B = 3 * C) : C = 260 :=
by
  sorry

end find_C_share_l1323_132320


namespace baking_completion_time_l1323_132310

theorem baking_completion_time (start_time : ℕ) (partial_bake_time : ℕ) (fraction_baked : ℕ) :
  start_time = 9 → partial_bake_time = 3 → fraction_baked = 4 →
  (start_time + (partial_bake_time * fraction_baked)) = 21 :=
by
  intros h_start h_partial h_fraction
  sorry

end baking_completion_time_l1323_132310


namespace area_relationship_l1323_132308

theorem area_relationship (P Q R : ℝ) (h_square : 10 * 10 = 100)
  (h_triangle1 : P + R = 50)
  (h_triangle2 : Q + R = 50) :
  P - Q = 0 :=
by
  sorry

end area_relationship_l1323_132308


namespace classes_after_drop_remaining_hours_of_classes_per_day_l1323_132317

def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_classes : ℕ := 1

theorem classes_after_drop 
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ) :
  initial_classes - dropped_classes = 3 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

theorem remaining_hours_of_classes_per_day
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ)
  (h : initial_classes - dropped_classes = 3) :
  hours_per_class * (initial_classes - dropped_classes) = 6 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

end classes_after_drop_remaining_hours_of_classes_per_day_l1323_132317


namespace avg_height_students_l1323_132340

theorem avg_height_students 
  (x : ℕ)  -- number of students in the first group
  (avg_height_first_group : ℕ)  -- average height of the first group
  (avg_height_second_group : ℕ)  -- average height of the second group
  (avg_height_combined_group : ℕ)  -- average height of the combined group
  (h1 : avg_height_first_group = 20)
  (h2 : avg_height_second_group = 20)
  (h3 : avg_height_combined_group = 20)
  (h4 : 20*x + 20*11 = 20*31) :
  x = 20 := 
  by {
    sorry
  }

end avg_height_students_l1323_132340


namespace find_missing_number_l1323_132360

theorem find_missing_number (x : ℕ) : (4 + 3) + (8 - 3 - x) = 11 → x = 1 :=
by
  sorry

end find_missing_number_l1323_132360


namespace find_m_in_hyperbola_l1323_132332

-- Define the problem in Lean 4
theorem find_m_in_hyperbola (m : ℝ) (x y : ℝ) (e : ℝ) (a_sq : ℝ := 9) (h_eq : e = 2) (h_hyperbola : x^2 / a_sq - y^2 / m = 1) : m = 27 :=
sorry

end find_m_in_hyperbola_l1323_132332


namespace calc_root_difference_l1323_132391

theorem calc_root_difference :
  ((81: ℝ)^(1/4) + (32: ℝ)^(1/5) - (49: ℝ)^(1/2)) = -2 :=
by
  have h1 : (81: ℝ)^(1/4) = 3 := by sorry
  have h2 : (32: ℝ)^(1/5) = 2 := by sorry
  have h3 : (49: ℝ)^(1/2) = 7 := by sorry
  rw [h1, h2, h3]
  norm_num

end calc_root_difference_l1323_132391


namespace prime_p_square_condition_l1323_132361

theorem prime_p_square_condition (p : ℕ) (h_prime : Prime p) (h_square : ∃ n : ℤ, 5^p + 4 * p^4 = n^2) :
  p = 31 :=
sorry

end prime_p_square_condition_l1323_132361


namespace triangle_parallel_vectors_l1323_132366

noncomputable def collinear {V : Type*} [AddCommGroup V] [Module ℝ V]
  (P₁ P₂ P₃ : V) : Prop :=
∃ t : ℝ, P₃ = P₁ + t • (P₂ - P₁)

theorem triangle_parallel_vectors
  (A B C C₁ A₁ B₁ C₂ A₂ B₂ : ℝ × ℝ)
  (h1 : collinear A B C₁) (h2 : collinear B C A₁) (h3 : collinear C A B₁)
  (ratio1 : ∀ (AC1 CB : ℝ), AC1 / CB = 1) (ratio2 : ∀ (BA1 AC : ℝ), BA1 / AC = 1) (ratio3 : ∀ (CB B1A : ℝ), CB / B1A = 1)
  (h4 : collinear A₁ B₁ C₂) (h5 : collinear B₁ C₁ A₂) (h6 : collinear C₁ A₁ B₂)
  (n : ℝ)
  (ratio4 : ∀ (A1C2 C2B1 : ℝ), A1C2 / C2B1 = n) (ratio5 : ∀ (B1A2 A2C1 : ℝ), B1A2 / A2C1 = n) (ratio6 : ∀ (C1B2 B2A1 : ℝ), C1B2 / B2A1 = n) :
  collinear A C A₂ ∧ collinear C B C₂ ∧ collinear B A B₂ :=
sorry

end triangle_parallel_vectors_l1323_132366


namespace coprime_condition_exists_l1323_132383

theorem coprime_condition_exists : ∃ (A B C : ℕ), (A > 0 ∧ B > 0 ∧ C > 0) ∧ (Nat.gcd (Nat.gcd A B) C = 1) ∧ 
  (A * Real.log 5 / Real.log 50 + B * Real.log 2 / Real.log 50 = C) ∧ (A + B + C = 4) :=
by {
  sorry
}

end coprime_condition_exists_l1323_132383


namespace watermelon_count_l1323_132362

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ)
  (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) : total_seeds / seeds_per_watermelon = 4 :=
by
  sorry

end watermelon_count_l1323_132362
