import Mathlib

namespace chessboard_colorings_l484_48404

-- Definitions based on conditions
def valid_chessboard_colorings_count : ℕ :=
  2 ^ 33

-- Theorem statement with the question, conditions, and the correct answer
theorem chessboard_colorings : 
  valid_chessboard_colorings_count = 2 ^ 33 := by
  sorry

end chessboard_colorings_l484_48404


namespace smallest_value_A_plus_B_plus_C_plus_D_l484_48487

variable (A B C D : ℤ)

-- Given conditions in Lean statement form
def isArithmeticSequence (A B C : ℤ) : Prop :=
  B - A = C - B

def isGeometricSequence (B C D : ℤ) : Prop :=
  (C / B : ℚ) = 4 / 3 ∧ (D / C : ℚ) = C / B

def givenConditions (A B C D : ℤ) : Prop :=
  isArithmeticSequence A B C ∧ isGeometricSequence B C D

-- The proof problem to validate the smallest possible value
theorem smallest_value_A_plus_B_plus_C_plus_D (h : givenConditions A B C D) :
  A + B + C + D = 43 :=
sorry

end smallest_value_A_plus_B_plus_C_plus_D_l484_48487


namespace total_students_in_class_l484_48436

theorem total_students_in_class : 
  ∀ (total_candies students_candies : ℕ), 
    total_candies = 901 → students_candies = 53 → 
    students_candies * (total_candies / students_candies) = total_candies ∧ 
    total_candies % students_candies = 0 → 
    total_candies / students_candies = 17 := 
by 
  sorry

end total_students_in_class_l484_48436


namespace divisibility_of_binomial_l484_48448

theorem divisibility_of_binomial (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ n-1 → p ∣ Nat.choose n x) ↔ ∃ m : ℕ, n = p^m := sorry

end divisibility_of_binomial_l484_48448


namespace log_one_third_nine_l484_48453

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_third_nine : log_base (1/3) 9 = -2 := by
  sorry

end log_one_third_nine_l484_48453


namespace gcd_factorials_l484_48493

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem gcd_factorials (n m : ℕ) (hn : n = 8) (hm : m = 10) :
  Nat.gcd (factorial n) (factorial m) = 40320 := by
  sorry

end gcd_factorials_l484_48493


namespace compare_negatives_l484_48489

theorem compare_negatives : -3 < -2 :=
by {
  -- Placeholder for proof
  sorry
}

end compare_negatives_l484_48489


namespace union_sets_l484_48439

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_sets : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_sets_l484_48439


namespace smallest_angle_in_triangle_l484_48443

theorem smallest_angle_in_triangle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180) : 
  3 * k = 45 := 
by sorry

end smallest_angle_in_triangle_l484_48443


namespace initial_assessed_value_l484_48485

theorem initial_assessed_value (V : ℝ) (tax_rate : ℝ) (new_value : ℝ) (tax_increase : ℝ) 
  (h1 : tax_rate = 0.10) 
  (h2 : new_value = 28000) 
  (h3 : tax_increase = 800) 
  (h4 : tax_rate * new_value = tax_rate * V + tax_increase) : 
  V = 20000 :=
by
  sorry

end initial_assessed_value_l484_48485


namespace percentage_y_less_than_x_l484_48459

variable (x y : ℝ)

-- given condition
axiom hyp : x = 11 * y

-- proof problem: Prove that the percentage y is less than x is (10/11) * 100
theorem percentage_y_less_than_x (x y : ℝ) (hyp : x = 11 * y) : 
  (x - y) / x * 100 = (10 / 11) * 100 :=
by
  sorry

end percentage_y_less_than_x_l484_48459


namespace company_annual_income_l484_48464

variable {p a : ℝ}

theorem company_annual_income (h : 280 * p + (a - 280) * (p + 2) = a * (p + 0.25)) : a = 320 := 
sorry

end company_annual_income_l484_48464


namespace work_completion_time_l484_48465

theorem work_completion_time 
    (A B : ℝ) 
    (h1 : A = 2 * B) 
    (h2 : (A + B) * 18 = 1) : 
    1 / A = 27 := 
by 
    sorry

end work_completion_time_l484_48465


namespace motorbike_speed_l484_48405

noncomputable def speed_of_motorbike 
  (V_train : ℝ) 
  (t_overtake : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  V_train - (train_length_m / 1000) * (3600 / t_overtake)

theorem motorbike_speed : 
  speed_of_motorbike 100 80 800.064 = 63.99712 :=
by
  -- this is where the proof steps would go
  sorry

end motorbike_speed_l484_48405


namespace lizette_overall_average_is_94_l484_48416

-- Defining the given conditions
def third_quiz_score : ℕ := 92
def first_two_quizzes_average : ℕ := 95
def total_quizzes : ℕ := 3

-- Calculating total points from the conditions
def total_points : ℕ := first_two_quizzes_average * 2 + third_quiz_score

-- Defining the overall average to prove
def overall_average : ℕ := total_points / total_quizzes

-- The theorem stating Lizette's overall average after taking the third quiz
theorem lizette_overall_average_is_94 : overall_average = 94 := by
  sorry

end lizette_overall_average_is_94_l484_48416


namespace min_value_of_expression_l484_48415

theorem min_value_of_expression (x y : ℝ) : (2 * x * y - 3) ^ 2 + (x - y) ^ 2 ≥ 1 :=
sorry

end min_value_of_expression_l484_48415


namespace ratio_of_a_to_c_l484_48408

theorem ratio_of_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3) 
  (h3 : d / b = 1 / 5) : a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l484_48408


namespace prove_a_value_l484_48461

theorem prove_a_value (a : ℝ) (h : (a - 2) * 0^2 + 0 + a^2 - 4 = 0) : a = -2 := 
by
  sorry

end prove_a_value_l484_48461


namespace union_of_intervals_l484_48430

open Set

theorem union_of_intervals :
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  M ∪ N = { x : ℝ | 1 < x ∧ x ≤ 5 } :=
by
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  sorry

end union_of_intervals_l484_48430


namespace pizza_slices_per_pizza_l484_48447

theorem pizza_slices_per_pizza (h : ∀ (mrsKaplanSlices bobbySlices pizzas : ℕ), 
  mrsKaplanSlices = 3 ∧ mrsKaplanSlices = bobbySlices / 4 ∧ pizzas = 2 → bobbySlices / pizzas = 6) : 
  ∃ (bobbySlices pizzas : ℕ), bobbySlices / pizzas = 6 :=
by
  existsi (3 * 4)
  existsi 2
  sorry

end pizza_slices_per_pizza_l484_48447


namespace problem_example_l484_48445

theorem problem_example (a : ℕ) (H1 : a ∈ ({a, b, c} : Set ℕ)) (H2 : 0 ∈ ({x | x^2 ≠ 0} : Set ℕ)) :
  a ∈ ({a, b, c} : Set ℕ) ∧ 0 ∈ ({x | x^2 ≠ 0} : Set ℕ) :=
by
  sorry

end problem_example_l484_48445


namespace nat_pow_eq_iff_divides_l484_48431

theorem nat_pow_eq_iff_divides (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : a = b^n :=
sorry

end nat_pow_eq_iff_divides_l484_48431


namespace units_digit_m_squared_plus_2_pow_m_l484_48442

-- Define the value of m
def m : ℕ := 2023^2 + 2^2023

-- Define the property we need to prove
theorem units_digit_m_squared_plus_2_pow_m :
  ((m^2 + 2^m) % 10) = 7 :=
by
  sorry

end units_digit_m_squared_plus_2_pow_m_l484_48442


namespace roots_product_l484_48450

theorem roots_product (x1 x2 : ℝ) (h : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 → x = x1 ∨ x = x2) : x1 * x2 = 1 :=
sorry

end roots_product_l484_48450


namespace units_digit_G_1000_l484_48424

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 2 :=
by
  sorry

end units_digit_G_1000_l484_48424


namespace repeating_decimal_sum_l484_48438

def repeating_decimal_to_fraction (d : ℕ) (n : ℕ) : ℚ := n / ((10^d) - 1)

theorem repeating_decimal_sum : 
  repeating_decimal_to_fraction 1 2 + repeating_decimal_to_fraction 2 2 + repeating_decimal_to_fraction 4 2 = 2474646 / 9999 := 
sorry

end repeating_decimal_sum_l484_48438


namespace average_of_seven_consecutive_l484_48432

theorem average_of_seven_consecutive (
  a : ℤ 
  ) (c : ℤ) 
  (h1 : c = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 7 := 
by 
  sorry

end average_of_seven_consecutive_l484_48432


namespace valid_third_side_l484_48479

-- Define a structure for the triangle with given sides
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the conditions using the triangle inequality theorem
def valid_triangle (T : Triangle) : Prop :=
  T.a + T.x > T.b ∧ T.b + T.x > T.a ∧ T.a + T.b > T.x

-- Given values of a and b, and the condition on x
def specific_triangle : Triangle :=
  { a := 4, b := 9, x := 6 }

-- Statement to prove valid_triangle holds for specific_triangle
theorem valid_third_side : valid_triangle specific_triangle :=
by
  -- Import or assumptions about inequalities can be skipped or replaced by sorry
  sorry

end valid_third_side_l484_48479


namespace find_a_l484_48468

-- Definitions and theorem statement
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}
def C (a : ℝ) : Set ℝ := {3}

theorem find_a (a : ℝ) : A a ∩ B a = C a → a = 2 :=
by
  sorry

end find_a_l484_48468


namespace rectangles_divided_into_13_squares_l484_48437

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l484_48437


namespace length_of_the_bridge_l484_48495

-- Conditions
def train_length : ℝ := 80
def train_speed_kmh : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Conversion factor
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Calculation
noncomputable def train_speed_ms : ℝ := train_speed_kmh * km_to_m / hr_to_s
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

-- Proof statement
theorem length_of_the_bridge : bridge_length = 295 :=
by
  sorry

end length_of_the_bridge_l484_48495


namespace simplify_expression_l484_48475

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x ^ 2 + 8 - (4 - 5 * x + 3 * y - 9 * x ^ 2) = 18 * x ^ 2 + 10 * x - 6 * y + 4 :=
by
  sorry

end simplify_expression_l484_48475


namespace total_visible_legs_l484_48411

-- Defining the conditions
def num_crows : ℕ := 4
def num_pigeons : ℕ := 3
def num_flamingos : ℕ := 5
def num_sparrows : ℕ := 8

def legs_per_crow : ℕ := 2
def legs_per_pigeon : ℕ := 2
def legs_per_flamingo : ℕ := 3
def legs_per_sparrow : ℕ := 2

-- Formulating the theorem that we need to prove
theorem total_visible_legs :
  (num_crows * legs_per_crow) +
  (num_pigeons * legs_per_pigeon) +
  (num_flamingos * legs_per_flamingo) +
  (num_sparrows * legs_per_sparrow) = 45 := by sorry

end total_visible_legs_l484_48411


namespace arithmetic_sequence_value_l484_48414

theorem arithmetic_sequence_value (a : ℝ) 
  (h1 : 2 * (2 * a + 1) = (a - 1) + (a + 4)) : a = 1 / 2 := 
by 
  sorry

end arithmetic_sequence_value_l484_48414


namespace number_of_dozens_l484_48486

theorem number_of_dozens (x : Nat) (h : x = 16 * (3 * 4)) : x / 12 = 16 :=
by
  sorry

end number_of_dozens_l484_48486


namespace evaluate_expression_l484_48484

theorem evaluate_expression :
  (⌈(19 / 7 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ / ⌈(35 / 7 : ℚ) + ⌈((7 * 19) / 35 : ℚ)⌉⌉) = (1 / 9 : ℚ) :=
by
  sorry

end evaluate_expression_l484_48484


namespace stratified_sampling_medium_supermarkets_l484_48418

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end stratified_sampling_medium_supermarkets_l484_48418


namespace max_value_of_inverse_l484_48490

noncomputable def f (x y z : ℝ) : ℝ := (1/4) * x^2 + 2 * y^2 + 16 * z^2

theorem max_value_of_inverse (x y z a b c : ℝ) (h : a + b + c = 1) (pos_intercepts : a > 0 ∧ b > 0 ∧ c > 0)
  (point_on_plane : (x/a + y/b + z/c = 1)) (pos_points : x > 0 ∧ y > 0 ∧ z > 0) :
  ∀ (k : ℕ), 21 ≤ k → k < (f x y z)⁻¹ :=
sorry

end max_value_of_inverse_l484_48490


namespace alice_needs_136_life_vests_l484_48454

-- Definitions from the problem statement
def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def instructors_per_class : ℕ := 10
def life_vest_probability : ℝ := 0.40

-- Calculate the total number of people
def total_people := num_classes * (students_per_class + instructors_per_class)

-- Calculate the expected number of students with life vests
def students_with_life_vests := (students_per_class : ℝ) * life_vest_probability
def total_students_with_life_vests := num_classes * students_with_life_vests

-- Calculate the number of life vests needed
def life_vests_needed := total_people - total_students_with_life_vests

-- Proof statement (missing the actual proof)
theorem alice_needs_136_life_vests : life_vests_needed = 136 := by
  sorry

end alice_needs_136_life_vests_l484_48454


namespace rectangle_to_cylinder_max_volume_ratio_l484_48429

/-- Given a rectangle with a perimeter of 12 and converting it into a cylinder 
with the height being the same as the width of the rectangle, prove that the 
ratio of the circumference of the cylinder's base to its height when the volume 
is maximized is 2:1. -/
theorem rectangle_to_cylinder_max_volume_ratio : 
  ∃ (x : ℝ), (2 * x + 2 * (6 - x)) = 12 → 2 * (6 - x) / x = 2 :=
sorry

end rectangle_to_cylinder_max_volume_ratio_l484_48429


namespace find_x_l484_48435

noncomputable def f (x : ℝ) := (30 : ℝ) / (x + 5)
noncomputable def h (x : ℝ) := 4 * (f⁻¹ x)

theorem find_x (x : ℝ) (hx : h x = 20) : x = 3 :=
by 
  -- Conditions
  let f_inv := f⁻¹
  have h_def : h x = 4 * f_inv x := rfl
  have f_def : f x = (30 : ℝ) / (x + 5) := rfl
  -- Needed Proof Steps
  sorry

end find_x_l484_48435


namespace triangle_inequality_range_x_l484_48427

theorem triangle_inequality_range_x (x : ℝ) :
  let a := 3;
  let b := 8;
  let c := 1 + 2 * x;
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔ (2 < x ∧ x < 5) :=
by
  sorry

end triangle_inequality_range_x_l484_48427


namespace min_value_of_x_l484_48469

open Real

-- Defining the conditions
def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := log x ≥ 2 * log 3 + (1/3) * log x

-- Statement of the theorem
theorem min_value_of_x (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x ≥ 27 :=
sorry

end min_value_of_x_l484_48469


namespace find_values_of_m_l484_48491

theorem find_values_of_m (m : ℤ) (h₁ : m > 2022) (h₂ : (2022 + m) ∣ (2022 * m)) : 
  m = 1011 ∨ m = 2022 :=
sorry

end find_values_of_m_l484_48491


namespace num_trains_encountered_l484_48434

noncomputable def train_travel_encounters : ℕ := 5

theorem num_trains_encountered (start_time : ℕ) (duration : ℕ) (daily_departure : ℕ) 
  (train_journey_duration : ℕ) (daily_start_interval : ℕ) 
  (end_time : ℕ) (number_encountered : ℕ) :
  (train_journey_duration = 3 * 24 * 60 + 30) → -- 3 days and 30 minutes in minutes
  (daily_start_interval = 24 * 60) →             -- interval between daily train starts (in minutes)
  (number_encountered = 5) :=
by
  sorry

end num_trains_encountered_l484_48434


namespace geometric_series_sum_l484_48417

theorem geometric_series_sum :
  let a := 2
  let r := 2
  let n := 11
  let S := a * (r^n - 1) / (r - 1)
  S = 4094 := by
  sorry

end geometric_series_sum_l484_48417


namespace log_expression_l484_48480

section log_problem

variable (log : ℝ → ℝ)
variable (m n : ℝ)

-- Assume the properties of logarithms:
-- 1. log(m^n) = n * log(m)
axiom log_pow (m : ℝ) (n : ℝ) : log (m ^ n) = n * log m
-- 2. log(m * n) = log(m) + log(n)
axiom log_mul (m n : ℝ) : log (m * n) = log m + log n
-- 3. log(1) = 0
axiom log_one : log 1 = 0

theorem log_expression : log 5 * log 2 + log (2 ^ 2) - log 2 = 0 := by
  sorry

end log_problem

end log_expression_l484_48480


namespace blue_marbles_l484_48481

theorem blue_marbles (r b : ℕ) (h_ratio : 3 * b = 5 * r) (h_red : r = 18) : b = 30 := by
  -- proof
  sorry

end blue_marbles_l484_48481


namespace product_pass_rate_l484_48498

variable {a b : ℝ} (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) (h_indep : true)

theorem product_pass_rate : (1 - a) * (1 - b) = 
((1 - a) * (1 - b)) :=
by
  sorry

end product_pass_rate_l484_48498


namespace max_investment_at_7_percent_l484_48444

variables (x y : ℝ)

theorem max_investment_at_7_percent 
  (h1 : x + y = 25000)
  (h2 : 0.07 * x + 0.12 * y ≥ 2450) : 
  x ≤ 11000 :=
sorry

end max_investment_at_7_percent_l484_48444


namespace negation_problem_l484_48494

variable {a b c : ℝ}

theorem negation_problem (h : a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) : 
  a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3 :=
sorry

end negation_problem_l484_48494


namespace T_n_lt_1_l484_48470

open Nat

def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := (a n : ℚ) / ((b n : ℚ) * (b (n + 1) : ℚ))

noncomputable def T (n : ℕ) : ℚ := (Finset.range (n + 1)).sum c

theorem T_n_lt_1 (n : ℕ) : T n < 1 := by
  sorry

end T_n_lt_1_l484_48470


namespace not_possible_2018_people_in_2019_minutes_l484_48457

-- Definitions based on conditions
def initial_people (t : ℕ) : ℕ := 0
def changed_people (x y : ℕ) : ℕ := 2 * x - y

theorem not_possible_2018_people_in_2019_minutes :
  ¬ ∃ (x y : ℕ), (x + y = 2019) ∧ (2 * x - y = 2018) :=
by
  sorry

end not_possible_2018_people_in_2019_minutes_l484_48457


namespace lcm_of_8_and_15_l484_48466

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 :=
by
  sorry

end lcm_of_8_and_15_l484_48466


namespace intersection_eq_1_2_l484_48482

-- Define the set M
def M : Set ℝ := {y : ℝ | -2 ≤ y ∧ y ≤ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | 1 < x}

-- The intersection of M and N
def intersection : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 2 }

-- Our goal is to prove that M ∩ N = (1, 2]
theorem intersection_eq_1_2 : (M ∩ N) = (Set.Ioo 1 2) :=
by
  sorry

end intersection_eq_1_2_l484_48482


namespace derivative_of_my_function_l484_48499

variable (x : ℝ)

noncomputable def my_function : ℝ :=
  (Real.cos (Real.sin 3))^2 + (Real.sin (29 * x))^2 / (29 * Real.cos (58 * x))

theorem derivative_of_my_function :
  deriv my_function x = Real.tan (58 * x) / Real.cos (58 * x) := 
sorry

end derivative_of_my_function_l484_48499


namespace linear_equation_solution_l484_48401

theorem linear_equation_solution (m : ℝ) (x : ℝ) (h : |m| - 2 = 1) (h_ne : m ≠ 3) :
  (2 * m - 6) * x^(|m|-2) = m^2 ↔ x = -(3/4) :=
by
  sorry

end linear_equation_solution_l484_48401


namespace depth_of_water_in_cistern_l484_48476

-- Define the given constants
def length_cistern : ℝ := 6
def width_cistern : ℝ := 5
def total_wet_area : ℝ := 57.5

-- Define the area of the bottom of the cistern
def area_bottom (length : ℝ) (width : ℝ) : ℝ := length * width

-- Define the area of the longer sides of the cistern in contact with water
def area_long_sides (length : ℝ) (depth : ℝ) : ℝ := 2 * length * depth

-- Define the area of the shorter sides of the cistern in contact with water
def area_short_sides (width : ℝ) (depth : ℝ) : ℝ := 2 * width * depth

-- Define the total wet surface area based on depth of the water
def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ := 
    area_bottom length width + area_long_sides length depth + area_short_sides width depth

-- Define the proof statement
theorem depth_of_water_in_cistern : ∃ h : ℝ, h = 1.25 ∧ total_wet_surface_area length_cistern width_cistern h = total_wet_area := 
by
  use 1.25
  sorry

end depth_of_water_in_cistern_l484_48476


namespace factorization_correct_l484_48478

theorem factorization_correct : ∀ x : ℝ, (x^2 - 2*x - 9 = 0) → ((x-1)^2 = 10) :=
by 
  intros x h
  sorry

end factorization_correct_l484_48478


namespace events_equally_likely_iff_N_eq_18_l484_48425

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l484_48425


namespace consecutive_roots_prime_q_l484_48419

theorem consecutive_roots_prime_q (p q : ℤ) (h1 : Prime q)
  (h2 : ∃ x1 x2 : ℤ, 
    x1 ≠ x2 ∧ 
    (x1 = x2 + 1 ∨ x1 = x2 - 1) ∧ 
    x1 + x2 = p ∧ 
    x1 * x2 = q) : (p = 3 ∨ p = -3) ∧ q = 2 :=
by
  sorry

end consecutive_roots_prime_q_l484_48419


namespace initial_number_proof_l484_48446

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

def certain_value : ℕ := (factor1 * factor2) * factor3

theorem initial_number_proof :
  initial_number - certain_value = result := by
  sorry

end initial_number_proof_l484_48446


namespace Ron_book_picking_times_l484_48412

theorem Ron_book_picking_times (couples members : ℕ) (weeks people : ℕ) (Ron wife picks_per_year : ℕ) 
  (h1 : couples = 3) 
  (h2 : members = 5) 
  (h3 : Ron = 1) 
  (h4 : wife = 1) 
  (h5 : weeks = 52) 
  (h6 : people = 2 * couples + members + Ron + wife) 
  (h7 : picks_per_year = weeks / people) 
  : picks_per_year = 4 :=
by
  -- Definition steps can be added here if needed, currently immediate from conditions h1 to h7
  sorry

end Ron_book_picking_times_l484_48412


namespace quarters_initial_l484_48456

-- Define the given conditions
def candies_cost_dimes : Nat := 4 * 3
def candies_cost_cents : Nat := candies_cost_dimes * 10
def lollipop_cost_quarters : Nat := 1
def lollipop_cost_cents : Nat := lollipop_cost_quarters * 25
def total_spent_cents : Nat := candies_cost_cents + lollipop_cost_cents
def money_left_cents : Nat := 195
def total_initial_money_cents : Nat := money_left_cents + total_spent_cents
def dimes_count : Nat := 19
def dimes_value_cents : Nat := dimes_count * 10

-- Prove that the number of quarters initially is 6
theorem quarters_initial (quarters_count : Nat) (h : quarters_count * 25 = total_initial_money_cents - dimes_value_cents) : quarters_count = 6 :=
by
  sorry

end quarters_initial_l484_48456


namespace second_concert_attendance_l484_48410

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119

theorem second_concert_attendance : first_concert_attendance + additional_people = 66018 := 
by 
  -- Proof is not discussed here, only the statement is required.
sorry

end second_concert_attendance_l484_48410


namespace minimal_side_length_of_room_l484_48400

theorem minimal_side_length_of_room (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ S : ℕ, S = 10 :=
by {
  sorry
}

end minimal_side_length_of_room_l484_48400


namespace unique_y_star_l484_48467

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_y_star :
  ∃! y : ℝ, star 4 y = 20 :=
by 
  sorry

end unique_y_star_l484_48467


namespace house_cost_l484_48473

-- Definitions of given conditions
def annual_salary : ℝ := 150000
def saving_rate : ℝ := 0.10
def downpayment_rate : ℝ := 0.20
def years_saving : ℝ := 6

-- Given the conditions, calculate annual savings and total savings after 6 years
def annual_savings : ℝ := annual_salary * saving_rate
def total_savings : ℝ := annual_savings * years_saving

-- Total savings represents 20% of the house cost
def downpayment : ℝ := total_savings

-- Prove the total cost of the house
theorem house_cost (downpayment : ℝ) (downpayment_rate : ℝ) : ℝ :=
  downpayment / downpayment_rate

lemma house_cost_correct : house_cost downpayment downpayment_rate = 450000 :=
by
  -- the proof would go here
  sorry

end house_cost_l484_48473


namespace solve_abc_values_l484_48449

theorem solve_abc_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 8/3) :
  abc = 1 ∨ abc = 37/3 :=
sorry

end solve_abc_values_l484_48449


namespace number_of_seven_banana_bunches_l484_48458

theorem number_of_seven_banana_bunches (total_bananas : ℕ) (eight_banana_bunches : ℕ) (seven_banana_bunches : ℕ) : 
    total_bananas = 83 → 
    eight_banana_bunches = 6 → 
    (∃ n : ℕ, seven_banana_bunches = n) → 
    8 * eight_banana_bunches + 7 * seven_banana_bunches = total_bananas → 
    seven_banana_bunches = 5 := by
  sorry

end number_of_seven_banana_bunches_l484_48458


namespace sam_after_joan_took_marbles_l484_48455

theorem sam_after_joan_took_marbles
  (original_yellow : ℕ)
  (marbles_taken_by_joan : ℕ)
  (remaining_yellow : ℕ)
  (h1 : original_yellow = 86)
  (h2 : marbles_taken_by_joan = 25)
  (h3 : remaining_yellow = original_yellow - marbles_taken_by_joan) :
  remaining_yellow = 61 :=
by
  sorry

end sam_after_joan_took_marbles_l484_48455


namespace leak_time_to_empty_cistern_l484_48472

theorem leak_time_to_empty_cistern :
  (1/6 - 1/8) = 1/24 → (1 / (1/24)) = 24 := by
sorry

end leak_time_to_empty_cistern_l484_48472


namespace florida_vs_georgia_license_plates_l484_48440

theorem florida_vs_georgia_license_plates :
  26 ^ 4 * 10 ^ 3 - 26 ^ 3 * 10 ^ 3 = 439400000 := by
  -- proof is omitted as directed
  sorry

end florida_vs_georgia_license_plates_l484_48440


namespace find_angle_B_l484_48492

-- Conditions
variable (A B C a b : ℝ)
variable (h1 : a = Real.sqrt 6)
variable (h2 : b = Real.sqrt 3)
variable (h3 : b + a * (Real.sin C - Real.cos C) = 0)

-- Target
theorem find_angle_B : B = Real.pi / 6 :=
sorry

end find_angle_B_l484_48492


namespace smallest_term_of_bn_div_an_is_four_l484_48483

theorem smallest_term_of_bn_div_an_is_four
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = 2 * S n)
  (h3 : b 1 = 16)
  (h4 : ∀ n, b (n + 1) - b n = 2 * n) :
  ∃ n : ℕ, ∀ m : ℕ, (m ≠ 4 → b m / a m > b 4 / a 4) ∧ (n = 4) := sorry

end smallest_term_of_bn_div_an_is_four_l484_48483


namespace sum_of_x_and_y_l484_48423

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_of_x_and_y_l484_48423


namespace solution_x_x_sub_1_eq_x_l484_48428

theorem solution_x_x_sub_1_eq_x (x : ℝ) : x * (x - 1) = x ↔ (x = 0 ∨ x = 2) :=
by {
  sorry
}

end solution_x_x_sub_1_eq_x_l484_48428


namespace functional_eq_l484_48462

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end functional_eq_l484_48462


namespace work_completion_l484_48497

theorem work_completion (A B C : ℚ) (hA : A = 1/21) (hB : B = 1/6) 
    (hCombined : A + B + C = 1/3.36) : C = 1/12 := by
  sorry

end work_completion_l484_48497


namespace sum_of_roots_of_quadratic_l484_48407

theorem sum_of_roots_of_quadratic (m n : ℝ) (h1 : m = 2 * n) (h2 : ∀ x : ℝ, x ^ 2 + m * x + n = 0) :
    m + n = 3 / 2 :=
sorry

end sum_of_roots_of_quadratic_l484_48407


namespace razorback_shop_revenue_from_jerseys_zero_l484_48477

theorem razorback_shop_revenue_from_jerseys_zero:
  let num_tshirts := 20
  let num_jerseys := 64
  let revenue_per_tshirt := 215
  let total_revenue_tshirts := 4300
  let total_revenue := total_revenue_tshirts
  let revenue_from_jerseys := total_revenue - total_revenue_tshirts
  revenue_from_jerseys = 0 := by
  sorry

end razorback_shop_revenue_from_jerseys_zero_l484_48477


namespace value_of_expression_l484_48426

theorem value_of_expression : (20 * 24) / (2 * 0 + 2 * 4) = 60 := sorry

end value_of_expression_l484_48426


namespace range_of_m_l484_48441

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x^2 + 2 * x - m - 1 = 0) → m ≥ -2 := 
by
  sorry

end range_of_m_l484_48441


namespace sin_15_cos_15_l484_48406

theorem sin_15_cos_15 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := by
  sorry

end sin_15_cos_15_l484_48406


namespace tan_product_eq_three_l484_48409

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l484_48409


namespace train_length_l484_48433

theorem train_length (speed_kmph : ℕ) (time_seconds : ℕ) (length_meters : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_seconds = 14)
  (h3 : length_meters = speed_kmph * 1000 * time_seconds / 3600)
  : length_meters = 280 := by
  sorry

end train_length_l484_48433


namespace find_s_l484_48471

theorem find_s (s : Real) (h : ⌊s⌋ + s = 15.4) : s = 7.4 :=
sorry

end find_s_l484_48471


namespace no_three_distinct_nat_numbers_sum_prime_l484_48451

theorem no_three_distinct_nat_numbers_sum_prime:
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Nat.Prime (a + b) ∧ Nat.Prime (a + c) ∧ Nat.Prime (b + c) := 
sorry

end no_three_distinct_nat_numbers_sum_prime_l484_48451


namespace arithmetic_seq_max_n_l484_48413

def arithmetic_seq_max_sum (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 > 0) ∧ (3 * (a 1 + 4 * d) = 5 * (a 1 + 7 * d)) ∧
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧
  (S 12 = -72 * d)

theorem arithmetic_seq_max_n
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  arithmetic_seq_max_sum a d S → n = 12 :=
by
  sorry

end arithmetic_seq_max_n_l484_48413


namespace yellow_yarns_count_l484_48460

theorem yellow_yarns_count (total_scarves red_yarn_count blue_yarn_count yellow_yarns scarves_per_yarn : ℕ) 
  (h1 : 3 = scarves_per_yarn)
  (h2 : red_yarn_count = 2)
  (h3 : blue_yarn_count = 6)
  (h4 : total_scarves = 36)
  :
  yellow_yarns = 4 :=
by 
  sorry

end yellow_yarns_count_l484_48460


namespace sum_of_series_eq_one_third_l484_48463

theorem sum_of_series_eq_one_third :
  ∑' k : ℕ, (2^k / (8^k - 1)) = 1 / 3 :=
sorry

end sum_of_series_eq_one_third_l484_48463


namespace num_divisors_fact8_l484_48488

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l484_48488


namespace largest_k_inequality_l484_48452

theorem largest_k_inequality {a b c : ℝ} (h1 : a ≤ b) (h2 : b ≤ c) (h3 : ab + bc + ca = 0) (h4 : abc = 1) :
  |a + b| ≥ 4 * |c| :=
sorry

end largest_k_inequality_l484_48452


namespace train_length_l484_48421

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_train : ℝ) 
  (h_speed : speed_kmph = 50)
  (h_time : time_sec = 18) 
  (h_length : length_train = 250) : 
  (speed_kmph * 1000 / 3600) * time_sec = length_train :=
by 
  rw [h_speed, h_time, h_length]
  sorry

end train_length_l484_48421


namespace units_digit_of_power_ends_in_nine_l484_48403

theorem units_digit_of_power_ends_in_nine (n : ℕ) (h : (3^n) % 10 = 9) : n % 4 = 2 :=
sorry

end units_digit_of_power_ends_in_nine_l484_48403


namespace bacteria_growth_time_l484_48474

theorem bacteria_growth_time (n0 : ℕ) (n : ℕ) (rate : ℕ) (time_step : ℕ) (final : ℕ)
  (h0 : n0 = 200)
  (h1 : rate = 3)
  (h2 : time_step = 5)
  (h3 : n = n0 * rate ^ final)
  (h4 : n = 145800) :
  final = 30 := 
sorry

end bacteria_growth_time_l484_48474


namespace transformation_C_factorization_l484_48496

open Function

theorem transformation_C_factorization (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by sorry

end transformation_C_factorization_l484_48496


namespace eccentricity_of_hyperbola_l484_48422

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_asymp : 3 * a + b = 0) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 10 :=
by
  sorry

end eccentricity_of_hyperbola_l484_48422


namespace max_sum_abc_l484_48402

theorem max_sum_abc (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) : a + b + c ≤ 3 :=
sorry

end max_sum_abc_l484_48402


namespace paper_thickness_after_folding_five_times_l484_48420

-- Definitions of initial conditions
def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 5

-- Target thickness after folding
def final_thickness (init_thickness : ℝ) (folds : ℕ) : ℝ :=
  (2 ^ folds) * init_thickness

-- Statement of the theorem
theorem paper_thickness_after_folding_five_times :
  final_thickness initial_thickness num_folds = 3.2 :=
by
  -- The proof (the implementation is replaced with sorry)
  sorry

end paper_thickness_after_folding_five_times_l484_48420
