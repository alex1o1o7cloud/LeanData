import Mathlib

namespace find_k_l241_241830

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

theorem find_k (k : ℝ) (h : f 5 - g 5 k = 24) : k = -16.36 := by
  sorry

end find_k_l241_241830


namespace sum_of_remaining_digit_is_correct_l241_241975

-- Define the local value calculation function for a particular digit with its place value
def local_value (digit place_value : ℕ) : ℕ := digit * place_value

-- Define the number in question
def number : ℕ := 2345

-- Define the local values for each digit in their respective place values
def local_value_2 : ℕ := local_value 2 1000
def local_value_3 : ℕ := local_value 3 100
def local_value_4 : ℕ := local_value 4 10
def local_value_5 : ℕ := local_value 5 1

-- Define the given sum of the local values
def given_sum : ℕ := 2345

-- Define the sum of the local values of the digits 2, 3, and 5
def sum_of_other_digits : ℕ := local_value_2 + local_value_3 + local_value_5

-- Define the target sum which is the sum of the local value of the remaining digit
def target_sum : ℕ := given_sum - sum_of_other_digits

-- Prove that the sum of the local value of the remaining digit is equal to 40
theorem sum_of_remaining_digit_is_correct : target_sum = 40 := 
by
  -- The proof will be provided here
  sorry

end sum_of_remaining_digit_is_correct_l241_241975


namespace ellipse_eccentricity_l241_241812

open Real

def ellipse_foci_x_axis (m : ℝ) : Prop :=
  ∃ a b c e,
    a = sqrt m ∧
    b = sqrt 6 ∧
    c = sqrt (m - 6) ∧
    e = c / a ∧
    e = 1 / 2

theorem ellipse_eccentricity (m : ℝ) (h : ellipse_foci_x_axis m) :
  m = 8 := by
  sorry

end ellipse_eccentricity_l241_241812


namespace simplify_fraction_l241_241465

theorem simplify_fraction (x : ℤ) :
  (2 * x - 3) / 4 + (3 * x + 5) / 5 - (x - 1) / 2 = (12 * x + 15) / 20 :=
by sorry

end simplify_fraction_l241_241465


namespace total_pages_is_360_l241_241394

-- Definitions from conditions
variable (A B : ℕ) -- Rates of printer A and printer B in pages per minute.
variable (total_pages : ℕ) -- Total number of pages of the task.

-- Given conditions
axiom h1 : 24 * (A + B) = total_pages -- Condition from both printers working together.
axiom h2 : 60 * A = total_pages -- Condition from printer A alone.
axiom h3 : B = A + 3 -- Condition of printer B printing 3 more pages per minute.

-- Goal: Prove the total number of pages is 360
theorem total_pages_is_360 : total_pages = 360 := 
by 
  sorry

end total_pages_is_360_l241_241394


namespace ratio_of_sums_l241_241504

theorem ratio_of_sums (total_sums : ℕ) (correct_sums : ℕ) (incorrect_sums : ℕ)
  (h1 : total_sums = 75)
  (h2 : incorrect_sums = 2 * correct_sums)
  (h3 : total_sums = correct_sums + incorrect_sums) :
  incorrect_sums / correct_sums = 2 :=
by
  -- Proof placeholder
  sorry

end ratio_of_sums_l241_241504


namespace primes_less_or_equal_F_l241_241788

-- Definition of F_n
def F (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- The main theorem statement
theorem primes_less_or_equal_F (n : ℕ) : ∃ S : Finset ℕ, S.card ≥ n + 1 ∧ ∀ p ∈ S, Nat.Prime p ∧ p ≤ F n := 
sorry

end primes_less_or_equal_F_l241_241788


namespace stratified_sampling_l241_241435

theorem stratified_sampling 
  (students_first_grade : ℕ)
  (students_second_grade : ℕ)
  (selected_first_grade : ℕ)
  (x : ℕ)
  (h1 : students_first_grade = 400)
  (h2 : students_second_grade = 360)
  (h3 : selected_first_grade = 60)
  (h4 : (selected_first_grade / students_first_grade : ℚ) = (x / students_second_grade : ℚ)) :
  x = 54 :=
sorry

end stratified_sampling_l241_241435


namespace trains_crossing_time_l241_241282

noncomputable def time_to_cross_each_other (L T1 T2 : ℝ) (H1 : L = 120) (H2 : T1 = 10) (H3 : T2 = 16) : ℝ :=
  let S1 := L / T1
  let S2 := L / T2
  let S := S1 + S2
  let D := L + L
  D / S

theorem trains_crossing_time : time_to_cross_each_other 120 10 16 (by rfl) (by rfl) (by rfl) = 240 / (12 + 7.5) :=
  sorry

end trains_crossing_time_l241_241282


namespace maximize_f_l241_241400

noncomputable def f (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximize_f (x : ℝ) (h : x < 5 / 4): ∃ M, (∀ y, (y < 5 / 4) → f y ≤ M) ∧ M = 1 := by
  sorry

end maximize_f_l241_241400


namespace apps_added_eq_sixty_l241_241598

-- Definitions derived from the problem conditions
def initial_apps : ℕ := 50
def removed_apps : ℕ := 10
def final_apps : ℕ := 100

-- Intermediate calculation based on the problem
def apps_after_removal : ℕ := initial_apps - removed_apps

-- The main theorem stating the mathematically equivalent proof problem
theorem apps_added_eq_sixty : final_apps - apps_after_removal = 60 :=
by
  sorry

end apps_added_eq_sixty_l241_241598


namespace range_of_m_l241_241308

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  ∀ m, m = 9 / 4 → (1 / x + 4 / y) ≥ m := 
by
  sorry

end range_of_m_l241_241308


namespace smallest_integer_expression_l241_241129

theorem smallest_integer_expression :
  ∃ m n : ℤ, 1237 * m + 78653 * n = 1 :=
sorry

end smallest_integer_expression_l241_241129


namespace sum_series_l241_241274

theorem sum_series :
  ∑' n:ℕ, (4 * n ^ 2 - 2 * n + 3) / 3 ^ n = 21 / 4 :=
sorry

end sum_series_l241_241274


namespace seventh_number_fifth_row_l241_241315

theorem seventh_number_fifth_row : 
  ∀ (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ), 
  (∀ i, 1 <= i ∧ i <= n  → b 1 i = 2 * i - 1) →
  (∀ j i, 2 <= j ∧ 1 <= i ∧ i <= n - (j-1)  → b j i = b (j-1) i + b (j-1) (i+1)) →
  (b : ℕ → ℕ → ℕ) →
  b 5 7 = 272 :=
by {
  sorry
}

end seventh_number_fifth_row_l241_241315


namespace triangle_angle_measure_l241_241007

/-- Proving the measure of angle x in a defined triangle -/
theorem triangle_angle_measure (A B C x : ℝ) (hA : A = 85) (hB : B = 35) (hC : C = 30) : x = 150 :=
by
  sorry

end triangle_angle_measure_l241_241007


namespace arithmetic_sum_problem_l241_241146

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sum_problem (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_sequence a)
  (h_S_def : ∀ n : ℕ, S n = sum_of_first_n_terms a n)
  (h_S13 : S 13 = 52) : a 4 + a 8 + a 9 = 12 :=
sorry

end arithmetic_sum_problem_l241_241146


namespace find_retail_price_l241_241606

-- Define the wholesale price
def wholesale_price : ℝ := 90

-- Define the profit as 20% of the wholesale price
def profit (w : ℝ) : ℝ := 0.2 * w

-- Define the selling price as the wholesale price plus the profit
def selling_price (w p : ℝ) : ℝ := w + p

-- Define the selling price as 90% of the retail price t
def discount_selling_price (t : ℝ) : ℝ := 0.9 * t

-- Prove that the retail price t is 120 given the conditions
theorem find_retail_price :
  ∃ t : ℝ, wholesale_price + (profit wholesale_price) = discount_selling_price t → t = 120 :=
by
  sorry

end find_retail_price_l241_241606


namespace minimum_value_inequality_l241_241413

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x * y * z * (x + y + z) = 1) : (x + y) * (y + z) ≥ 2 := 
sorry

end minimum_value_inequality_l241_241413


namespace noah_billed_amount_l241_241524

theorem noah_billed_amount
  (minutes_per_call : ℕ)
  (cost_per_minute : ℝ)
  (weeks_per_year : ℕ)
  (total_cost : ℝ)
  (h_minutes_per_call : minutes_per_call = 30)
  (h_cost_per_minute : cost_per_minute = 0.05)
  (h_weeks_per_year : weeks_per_year = 52)
  (h_total_cost : total_cost = 78) :
  (minutes_per_call * cost_per_minute * weeks_per_year = total_cost) :=
by
  sorry

end noah_billed_amount_l241_241524


namespace fertilizer_production_l241_241561

theorem fertilizer_production (daily_production : ℕ) (days : ℕ) (total_production : ℕ) 
  (h1 : daily_production = 105) 
  (h2 : days = 24) 
  (h3 : total_production = daily_production * days) : 
  total_production = 2520 := 
  by 
  -- skipping the proof
  sorry

end fertilizer_production_l241_241561


namespace solution_set_l241_241345

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom monotone_increasing : ∀ x y, x < y → f x ≤ f y
axiom f_at_3 : f 3 = 2

-- Proof statement
theorem solution_set : {x : ℝ | -2 ≤ f (3 - x) ∧ f (3 - x) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 6} :=
by {
  sorry
}

end solution_set_l241_241345


namespace simplest_square_root_l241_241314

theorem simplest_square_root (A B C D : Real) 
    (hA : A = Real.sqrt 0.1) 
    (hB : B = 1 / 2) 
    (hC : C = Real.sqrt 30) 
    (hD : D = Real.sqrt 18) : 
    C = Real.sqrt 30 := 
by 
    sorry

end simplest_square_root_l241_241314


namespace polynomial_evaluation_l241_241927

noncomputable def Q (x : ℝ) : ℝ :=
  x^4 + x^3 + 2 * x

theorem polynomial_evaluation :
  Q (3) = 114 := by
  -- We assume the conditions implicitly in this equivalence.
  sorry

end polynomial_evaluation_l241_241927


namespace brown_stripes_l241_241632

theorem brown_stripes (B G Bl : ℕ) (h1 : G = 3 * B) (h2 : Bl = 5 * G) (h3 : Bl = 60) : B = 4 :=
by {
  sorry
}

end brown_stripes_l241_241632


namespace parallel_lines_sufficient_necessity_l241_241418

theorem parallel_lines_sufficient_necessity (a : ℝ) :
  ¬ (a = 1 ↔ (∀ x : ℝ, a^2 * x + 1 = x - 1)) := 
sorry

end parallel_lines_sufficient_necessity_l241_241418


namespace gf_three_l241_241471

def f (x : ℕ) : ℕ := x^3 - 4 * x + 5
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem gf_three : g (f 3) = 1222 :=
by {
  -- We would need to prove the given mathematical statement here.
  sorry
}

end gf_three_l241_241471


namespace sequence_is_decreasing_l241_241142

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

theorem sequence_is_decreasing (a : ℕ → ℝ) (h1 : a 1 < 0) (h2 : is_geometric_sequence a (1/3)) :
  ∀ n, a (n + 1) < a n :=
by
  -- Here should be the proof
  sorry

end sequence_is_decreasing_l241_241142


namespace balloon_ratio_l241_241835

/-- Janice has 6 water balloons. --/
def Janice_balloons : Nat := 6

/-- Randy has half as many water balloons as Janice. --/
def Randy_balloons : Nat := Janice_balloons / 2

/-- Cynthia has 12 water balloons. --/
def Cynthia_balloons : Nat := 12

/-- The ratio of Cynthia's water balloons to Randy's water balloons is 4:1. --/
theorem balloon_ratio : Cynthia_balloons / Randy_balloons = 4 := by
  sorry

end balloon_ratio_l241_241835


namespace boys_in_class_l241_241341

theorem boys_in_class
  (g b : ℕ)
  (h_ratio : g = (3 * b) / 5)
  (h_total : g + b = 32) :
  b = 20 :=
sorry

end boys_in_class_l241_241341


namespace find_m_l241_241294

-- Define the sets A and B and the conditions
def A : Set ℝ := {x | x ≥ 3}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Define the conditions on these sets
def conditions (m : ℝ) : Prop :=
  (∀ x, x ∈ A ∨ x ∈ B m) ∧ (∀ x, ¬(x ∈ A ∧ x ∈ B m))

-- State the theorem
theorem find_m : ∃ m : ℝ, conditions m ∧ m = 3 :=
  sorry

end find_m_l241_241294


namespace geometric_sequence_ab_product_l241_241042

theorem geometric_sequence_ab_product (a b : ℝ) (h₁ : 2 ≤ a) (h₂ : a ≤ 16) (h₃ : 2 ≤ b) (h₄ : b ≤ 16)
  (h₅ : ∃ r : ℝ, a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) : a * b = 32 :=
by
  sorry

end geometric_sequence_ab_product_l241_241042


namespace fraction_of_quarters_from_1860_to_1869_l241_241220

theorem fraction_of_quarters_from_1860_to_1869
  (total_quarters : ℕ) (quarters_from_1860s : ℕ)
  (h1 : total_quarters = 30) (h2 : quarters_from_1860s = 15) :
  (quarters_from_1860s : ℚ) / (total_quarters : ℚ) = 1 / 2 := by
  sorry

end fraction_of_quarters_from_1860_to_1869_l241_241220


namespace simplify_fraction_l241_241181

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l241_241181


namespace no_good_polygon_in_division_of_equilateral_l241_241005

def is_equilateral_polygon (P : List Point) : Prop :=
  -- Definition of equilateral polygon
  sorry

def is_good_polygon (P : List Point) : Prop :=
  -- Definition of good polygon (having a pair of parallel sides)
  sorry

def is_divided_by_non_intersecting_diagonals (P : List Point) (polygons : List (List Point)) : Prop :=
  -- Definition for dividing by non-intersecting diagonals into several polygons
  sorry

def have_same_odd_sides (polygons : List (List Point)) : Prop :=
  -- Definition for all polygons having the same odd number of sides
  sorry

theorem no_good_polygon_in_division_of_equilateral (P : List Point) (polygons : List (List Point)) :
  is_equilateral_polygon P →
  is_divided_by_non_intersecting_diagonals P polygons →
  have_same_odd_sides polygons →
  ¬ ∃ gp ∈ polygons, is_good_polygon gp :=
by
  intro h_eq h_div h_odd
  intro h_good
  -- Proof goes here
  sorry

end no_good_polygon_in_division_of_equilateral_l241_241005


namespace Gianna_daily_savings_l241_241128

theorem Gianna_daily_savings 
  (total_saved : ℕ) (days_in_year : ℕ) 
  (H1 : total_saved = 14235) 
  (H2 : days_in_year = 365) : 
  total_saved / days_in_year = 39 := 
by 
  sorry

end Gianna_daily_savings_l241_241128


namespace hours_learning_english_each_day_l241_241233

theorem hours_learning_english_each_day (total_hours : ℕ) (days : ℕ) (learning_hours_per_day : ℕ) 
  (h1 : total_hours = 12) 
  (h2 : days = 2) 
  (h3 : total_hours = learning_hours_per_day * days) : 
  learning_hours_per_day = 6 := 
by
  sorry

end hours_learning_english_each_day_l241_241233


namespace polygon_num_sides_l241_241259

-- Define the given conditions
def perimeter : ℕ := 150
def side_length : ℕ := 15

-- State the theorem to prove the number of sides of the polygon
theorem polygon_num_sides (P : ℕ) (s : ℕ) (hP : P = perimeter) (hs : s = side_length) : P / s = 10 :=
by
  sorry

end polygon_num_sides_l241_241259


namespace bianca_drawing_time_at_home_l241_241926

-- Define the conditions
def drawing_time_at_school : ℕ := 22
def total_drawing_time : ℕ := 41

-- Define the calculation for drawing time at home
def drawing_time_at_home : ℕ := total_drawing_time - drawing_time_at_school

-- The proof goal
theorem bianca_drawing_time_at_home : drawing_time_at_home = 19 := by
  sorry

end bianca_drawing_time_at_home_l241_241926


namespace num_students_in_research_study_group_prob_diff_classes_l241_241757

-- Define the number of students in each class and the number of students selected from class (2)
def num_students_class1 : ℕ := 18
def num_students_class2 : ℕ := 27
def selected_from_class2 : ℕ := 3

-- Prove the number of students in the research study group
theorem num_students_in_research_study_group : 
  (∃ (m : ℕ), (m / 18 = 3 / 27) ∧ (m + selected_from_class2 = 5)) := 
by
  sorry

-- Prove the probability that the students speaking in both activities come from different classes
theorem prob_diff_classes : 
  (12 / 25 = 12 / 25) :=
by
  sorry

end num_students_in_research_study_group_prob_diff_classes_l241_241757


namespace school_fee_correct_l241_241175

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end school_fee_correct_l241_241175


namespace quadratic_distinct_roots_l241_241283

theorem quadratic_distinct_roots (a : ℝ) : 
  (a > -1 ∧ a ≠ 3) ↔ 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ 
    (a - 3) * x₂^2 - 4 * x₂ - 1 = 0 :=
by
  sorry

end quadratic_distinct_roots_l241_241283


namespace problem1_problem2_problem3_problem4_problem5_problem6_l241_241602

-- Problem 1
theorem problem1 : (-20 + 3 - (-5) - 7 : Int) = -19 := sorry

-- Problem 2
theorem problem2 : (-2.4 - 3.7 - 4.6 + 5.7 : Real) = -5 := sorry

-- Problem 3
theorem problem3 : (-0.25 + ((-3 / 7) * (4 / 5)) : Real) = (-83 / 140) := sorry

-- Problem 4
theorem problem4 : ((-1 / 2) * (-8) + (-6)^2 : Real) = 40 := sorry

-- Problem 5
theorem problem5 : ((-1 / 12 - 1 / 36 + 1 / 6) * (-36) : Real) = -2 := sorry

-- Problem 6
theorem problem6 : (-1^4 + (-2) + (-1 / 3) - abs (-9) : Real) = -37 / 3 := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l241_241602


namespace n_is_power_of_three_l241_241380

theorem n_is_power_of_three {n : ℕ} (hn_pos : 0 < n) (p : Nat.Prime (4^n + 2^n + 1)) :
  ∃ (a : ℕ), n = 3^a :=
by
  sorry

end n_is_power_of_three_l241_241380


namespace problem_statement_l241_241666

theorem problem_statement 
  (h1 : 17 ≡ 3 [MOD 7])
  (h2 : 3^1 ≡ 3 [MOD 7])
  (h3 : 3^2 ≡ 2 [MOD 7])
  (h4 : 3^3 ≡ 6 [MOD 7])
  (h5 : 3^4 ≡ 4 [MOD 7])
  (h6 : 3^5 ≡ 5 [MOD 7])
  (h7 : 3^6 ≡ 1 [MOD 7])
  (h8 : 3^100 ≡ 4 [MOD 7]) :
  17^100 ≡ 4 [MOD 7] :=
by sorry

end problem_statement_l241_241666


namespace list_scores_lowest_highest_l241_241414

variable (M Q S T : ℕ)

axiom Quay_thinks : Q = T
axiom Marty_thinks : M > T
axiom Shana_thinks : S < T
axiom Tana_thinks : T ≠ max M (max Q (max S T)) ∧ T ≠ min M (min Q (min S T))

theorem list_scores_lowest_highest : (S < T) ∧ (T = Q) ∧ (Q < M) ↔ (S < T) ∧ (T < M) :=
by
  sorry

end list_scores_lowest_highest_l241_241414


namespace parallelogram_sides_l241_241267

theorem parallelogram_sides (x y : ℝ) 
    (h1 : 4 * y + 2 = 12) 
    (h2 : 6 * x - 2 = 10)
    (h3 : 10 + 12 + (6 * x - 2) + (4 * y + 2) = 68) :
    x + y = 4.5 := 
by
  -- Proof to be provided
  sorry

end parallelogram_sides_l241_241267


namespace taimour_paints_fence_alone_in_15_hours_l241_241306

theorem taimour_paints_fence_alone_in_15_hours :
  ∀ (T : ℝ), (∀ (J : ℝ), J = T / 2 → (1 / J + 1 / T = 1 / 5)) → T = 15 :=
by
  intros T h
  have h1 := h (T / 2) rfl
  sorry

end taimour_paints_fence_alone_in_15_hours_l241_241306


namespace third_term_of_sequence_l241_241660

theorem third_term_of_sequence (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = (1 / 2) * a n + (1 / (2 * n))) : a 3 = 3 / 4 := by
  sorry

end third_term_of_sequence_l241_241660


namespace promotional_rate_ratio_is_one_third_l241_241100

-- Define the conditions
def normal_monthly_charge : ℕ := 30
def extra_fee : ℕ := 15
def total_paid : ℕ := 175

-- Define the total data plan amount equation
def calculate_total (P : ℕ) : ℕ :=
  P + 2 * normal_monthly_charge + (normal_monthly_charge + extra_fee) + 2 * normal_monthly_charge

theorem promotional_rate_ratio_is_one_third (P : ℕ) (hP : calculate_total P = total_paid) :
  P * 3 = normal_monthly_charge :=
by sorry

end promotional_rate_ratio_is_one_third_l241_241100


namespace average_age_before_new_students_joined_l241_241015

theorem average_age_before_new_students_joined 
  (A : ℝ) 
  (N : ℕ) 
  (new_students_average_age : ℝ) 
  (average_age_drop : ℝ) 
  (original_class_strength : ℕ)
  (hN : N = 17) 
  (h_new_students : new_students_average_age = 32)
  (h_age_drop : average_age_drop = 4)
  (h_strength : original_class_strength = 17)
  (h_equation : 17 * A + 17 * new_students_average_age = (2 * original_class_strength) * (A - average_age_drop)) :
  A = 40 :=
by sorry

end average_age_before_new_students_joined_l241_241015


namespace total_points_other_team_members_l241_241851

variable (x y : ℕ)

theorem total_points_other_team_members :
  (1 / 3 * x + 3 / 8 * x + 18 + y = x) ∧ (y ≤ 24) → y = 17 :=
by
  intro h
  have h1 : 1 / 3 * x + 3 / 8 * x + 18 + y = x := h.1
  have h2 : y ≤ 24 := h.2
  sorry

end total_points_other_team_members_l241_241851


namespace root_interval_sum_l241_241771

theorem root_interval_sum (a b : Int) (h1 : b - a = 1) (h2 : ∃ x, a < x ∧ x < b ∧ (x^3 - x + 1) = 0) : a + b = -3 := 
sorry

end root_interval_sum_l241_241771


namespace least_y_l241_241346

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end least_y_l241_241346


namespace probability_of_selection_l241_241673

-- defining necessary parameters and the systematic sampling method
def total_students : ℕ := 52
def selected_students : ℕ := 10
def exclusion_probability := 2 / total_students
def inclusion_probability_exclude := selected_students / (total_students - 2)
def final_probability := (1 - exclusion_probability) * inclusion_probability_exclude

-- the main theorem stating the probability calculation
theorem probability_of_selection :
  final_probability = 5 / 26 :=
by
  -- we skip the proof part and end with sorry since it is not required
  sorry

end probability_of_selection_l241_241673


namespace particle_speed_correct_l241_241038

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 9)

noncomputable def particle_speed : ℝ :=
  Real.sqrt (3 ^ 2 + 5 ^ 2)

theorem particle_speed_correct : particle_speed = Real.sqrt 34 := by
  sorry

end particle_speed_correct_l241_241038


namespace sue_final_answer_is_67_l241_241421

-- Declare the initial value Ben thinks of
def ben_initial_number : ℕ := 4

-- Ben's calculation function
def ben_number (b : ℕ) : ℕ := ((b + 2) * 3) + 5

-- Sue's calculation function
def sue_number (x : ℕ) : ℕ := ((x - 3) * 3) + 7

-- Define the final number Sue calculates
def final_sue_number : ℕ := sue_number (ben_number ben_initial_number)

-- Prove that Sue's final number is 67
theorem sue_final_answer_is_67 : final_sue_number = 67 :=
by 
  sorry

end sue_final_answer_is_67_l241_241421


namespace good_students_options_l241_241950

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l241_241950


namespace largest_possible_n_l241_241691

theorem largest_possible_n (b g : ℕ) (n : ℕ) (h1 : g = 3 * b)
  (h2 : ∀ (boy : ℕ), boy < b → ∀ (girlfriend : ℕ), girlfriend < g → girlfriend ≤ 2013)
  (h3 : ∀ (girl : ℕ), girl < g → ∀ (boyfriend : ℕ), boyfriend < b → boyfriend ≥ n) :
  n ≤ 671 := by
    sorry

end largest_possible_n_l241_241691


namespace checkerboard_red_squares_l241_241292

/-- Define the properties of the checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (colors : ℕ → ℕ → String)
  (corner_color : String)

/-- Our checkerboard patterning function -/
def checkerboard_colors (i j : ℕ) : String :=
  match (i + j) % 3 with
  | 0 => "blue"
  | 1 => "yellow"
  | _ => "red"

/-- Our checkerboard of size 33x33 -/
def chubby_checkerboard : Checkerboard :=
  { size := 33,
    colors := checkerboard_colors,
    corner_color := "blue" }

/-- Proof that the number of red squares is 363 -/
theorem checkerboard_red_squares (b : Checkerboard) (h1 : b.size = 33) (h2 : b.colors = checkerboard_colors) : ∃ n, n = 363 :=
  by sorry

end checkerboard_red_squares_l241_241292


namespace deepak_current_age_l241_241439

theorem deepak_current_age (A D : ℕ) (h1 : A / D = 5 / 7) (h2 : A + 6 = 36) : D = 42 :=
sorry

end deepak_current_age_l241_241439


namespace sum_of_ages_is_24_l241_241890

def age_problem :=
  ∃ (x y z : ℕ), 2 * x^2 + y^2 + z^2 = 194 ∧ (x + x + y + z = 24)

theorem sum_of_ages_is_24 : age_problem :=
by
  sorry

end sum_of_ages_is_24_l241_241890


namespace greatest_num_of_coins_l241_241081

-- Define the total amount of money Carlos has in U.S. coins.
def total_value : ℝ := 5.45

-- Define the value of each type of coin.
def quarter_value : ℝ := 0.25
def dime_value : ℝ := 0.10
def nickel_value : ℝ := 0.05

-- Define the number of quarters, dimes, and nickels Carlos has.
def num_coins (q : ℕ) := quarter_value * q + dime_value * q + nickel_value * q

-- The main theorem: Carlos can have at most 13 quarters, dimes, and nickels.
theorem greatest_num_of_coins (q : ℕ) :
  num_coins q = total_value → q ≤ 13 :=
sorry

end greatest_num_of_coins_l241_241081


namespace min_value_expression_l241_241095

theorem min_value_expression (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 
  = 5^(5/4) - 10 * Real.sqrt (5^(1/4)) + 5 := 
sorry

end min_value_expression_l241_241095


namespace math_proof_l241_241456

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : Nat) : Nat :=
  (factorial n) / ((factorial k) * (factorial (n - k)))

theorem math_proof :
  binom 20 6 * factorial 6 = 27907200 :=
by
  sorry

end math_proof_l241_241456


namespace sufficient_but_not_necessary_l241_241644

theorem sufficient_but_not_necessary {a b : ℝ} (h₁ : a < b) (h₂ : b < 0) : 
  (a^2 > b^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by
  sorry

end sufficient_but_not_necessary_l241_241644


namespace decagon_triangle_probability_l241_241198

theorem decagon_triangle_probability : 
  let total_vertices := 10
  let total_triangles := Nat.choose total_vertices 3
  let favorable_triangles := 10
  (total_triangles > 0) → 
  (favorable_triangles / total_triangles : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l241_241198


namespace trapezoid_area_is_8_l241_241273

noncomputable def trapezoid_area
  (AB CD : ℝ)        -- lengths of the bases
  (h : ℝ)            -- height (distance between the bases)
  : ℝ :=
  0.5 * (AB + CD) * h

theorem trapezoid_area_is_8 
  (AB CD : ℝ) 
  (h : ℝ) 
  (K M : ℝ) 
  (height_condition : h = 2)
  (AB_condition : AB = 5)
  (CD_condition : CD = 3)
  (K_midpoint : K = AB / 2) 
  (M_midpoint : M = CD / 2)
  : trapezoid_area AB CD h = 8 :=
by
  rw [trapezoid_area, AB_condition, CD_condition, height_condition]
  norm_num

end trapezoid_area_is_8_l241_241273


namespace number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l241_241060

theorem number_of_two_digit_factors_2_pow_18_minus_1_is_zero :
  (∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ n ∣ (2^18 - 1)) = false :=
by sorry

end number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l241_241060


namespace path_count_1800_l241_241221

-- Define the coordinates of the points
def A := (0, 8)
def B := (4, 5)
def C := (7, 2)
def D := (9, 0)

-- Function to calculate the number of combinatorial paths
def comb_paths (steps_right steps_down : ℕ) : ℕ :=
  Nat.choose (steps_right + steps_down) steps_right

-- Define the number of steps for each segment
def steps_A_B := (4, 2)  -- 4 right, 2 down
def steps_B_C := (3, 3)  -- 3 right, 3 down
def steps_C_D := (2, 2)  -- 2 right, 2 down

-- Calculate the number of paths for each segment
def paths_A_B := comb_paths steps_A_B.1 steps_A_B.2
def paths_B_C := comb_paths steps_B_C.1 steps_B_C.2
def paths_C_D := comb_paths steps_C_D.1 steps_C_D.2

-- Calculate the total number of paths combining all segments
def total_paths : ℕ :=
  paths_A_B * paths_B_C * paths_C_D

theorem path_count_1800 :
  total_paths = 1800 := by
  sorry

end path_count_1800_l241_241221


namespace product_evaluation_l241_241023

theorem product_evaluation : 
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_evaluation_l241_241023


namespace trig_identity_l241_241915

theorem trig_identity : 4 * Real.sin (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l241_241915


namespace smallest_four_digit_divisible_by_35_l241_241281

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end smallest_four_digit_divisible_by_35_l241_241281


namespace flower_bed_length_l241_241508

theorem flower_bed_length (a b : ℝ) :
  ∀ width : ℝ, (6 * a^2 - 4 * a * b + 2 * a = 2 * a * width) → width = 3 * a - 2 * b + 1 :=
by
  intros width h
  sorry

end flower_bed_length_l241_241508


namespace redistribution_amount_l241_241811

theorem redistribution_amount
    (earnings : Fin 5 → ℕ)
    (h : earnings = ![18, 22, 30, 35, 45]) :
    (earnings 4 - ((earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5)) = 15 :=
by
  sorry

end redistribution_amount_l241_241811


namespace find_blue_chips_l241_241861

def num_chips_satisfies (n m : ℕ) : Prop :=
  (n > m) ∧ (n + m > 2) ∧ (n + m < 50) ∧
  (n * (n - 1) + m * (m - 1)) = 2 * n * m

theorem find_blue_chips (n : ℕ) :
  (∃ m : ℕ, num_chips_satisfies n m) → 
  n = 3 ∨ n = 6 ∨ n = 10 ∨ n = 15 ∨ n = 21 ∨ n = 28 :=
by
  sorry

end find_blue_chips_l241_241861


namespace sum_of_cubes_l241_241931

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l241_241931


namespace square_chord_length_eq_l241_241648

def radius1 := 10
def radius2 := 7
def centers_distance := 15
def chord_length (x : ℝ) := 2 * x

theorem square_chord_length_eq :
    ∀ (x : ℝ), chord_length x = 15 →
    (10 + x)^2 - 200 * (Real.sqrt ((1 + 19.0 / 35.0) / 2)) = 200 - 200 * Real.sqrt (27.0 / 35.0) :=
sorry

end square_chord_length_eq_l241_241648


namespace total_socks_l241_241163

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end total_socks_l241_241163


namespace remainder_369963_div_6_is_3_l241_241244

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def remainder_when_divided (a b : ℕ) (r : ℕ) : Prop := a % b = r

theorem remainder_369963_div_6_is_3 :
  remainder_when_divided 369963 6 3 :=
by
  have h₁ : 369963 % 2 = 1 := by
    sorry -- It is known that 369963 is not divisible by 2.
  have h₂ : 369963 % 3 = 0 := by
    sorry -- It is known that 369963 is divisible by 3.
  have h₃ : 369963 % 6 = 3 := by
    sorry -- From the above properties, derive that the remainder when 369963 is divided by 6 is 3.
  exact h₃

end remainder_369963_div_6_is_3_l241_241244


namespace problem1_solution_problem2_solution_l241_241072

theorem problem1_solution (x : ℝ) : (x^2 - 4 * x = 5) → (x = 5 ∨ x = -1) :=
by sorry

theorem problem2_solution (x : ℝ) : (2 * x^2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1/2) :=
by sorry

end problem1_solution_problem2_solution_l241_241072


namespace find_square_digit_l241_241585

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_digits_31_42_7s (s : ℕ) : ℕ :=
  3 + 1 + 4 + 2 + 7 + s

-- The main theorem to prove
theorem find_square_digit (d : ℕ) (h0 : is_even d) (h1 : (sum_digits_31_42_7s d) % 3 = 0) : d = 4 :=
by
  sorry

end find_square_digit_l241_241585


namespace actual_size_of_plot_l241_241378

/-
Theorem: The actual size of the plot of land is 61440 acres.
Given:
- The plot of land is a rectangle.
- The map dimensions are 12 cm by 8 cm.
- 1 cm on the map equals 1 mile in reality.
- One square mile equals 640 acres.
-/

def map_length_cm := 12
def map_width_cm := 8
def cm_to_miles := 1 -- 1 cm equals 1 mile
def mile_to_acres := 640 -- 1 square mile is 640 acres

theorem actual_size_of_plot
  (length_cm : ℕ) (width_cm : ℕ) (cm_to_miles : ℕ → ℕ) (mile_to_acres : ℕ → ℕ) :
  length_cm = 12 → width_cm = 8 →
  (cm_to_miles 1 = 1) →
  (mile_to_acres 1 = 640) →
  (length_cm * width_cm * mile_to_acres (cm_to_miles 1 * cm_to_miles 1) = 61440) :=
by
  intros
  sorry

end actual_size_of_plot_l241_241378


namespace largest_apartment_size_l241_241700

theorem largest_apartment_size (rent_per_sqft : ℝ) (budget : ℝ) (s : ℝ) :
  rent_per_sqft = 0.9 →
  budget = 630 →
  s = budget / rent_per_sqft →
  s = 700 :=
by
  sorry

end largest_apartment_size_l241_241700


namespace machine_minutes_worked_l241_241954

-- Definitions based on conditions
def shirts_made_yesterday : ℕ := 9
def shirts_per_minute : ℕ := 3

-- The proof problem statement
theorem machine_minutes_worked (shirts_made_yesterday shirts_per_minute : ℕ) : 
  shirts_made_yesterday / shirts_per_minute = 3 := 
by
  sorry

end machine_minutes_worked_l241_241954


namespace geom_sequence_third_term_l241_241918

theorem geom_sequence_third_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a n = a 1 * r ^ (n - 1)) (h_cond : a 1 * a 5 = a 3) : a 3 = 1 :=
sorry

end geom_sequence_third_term_l241_241918


namespace intersection_P_Q_l241_241211

open Set

noncomputable def P : Set ℝ := {x | abs (x - 1) < 4}
noncomputable def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2) }

theorem intersection_P_Q :
  (P ∩ Q) = {x : ℝ | -2 < x ∧ x < 5} :=
by
  sorry

end intersection_P_Q_l241_241211


namespace survival_rate_is_98_l241_241286

def total_flowers := 150
def unsurviving_flowers := 3
def surviving_flowers := total_flowers - unsurviving_flowers

theorem survival_rate_is_98 : (surviving_flowers : ℝ) / total_flowers * 100 = 98 := by
  sorry

end survival_rate_is_98_l241_241286


namespace total_books_l241_241342

-- Define the given conditions
def books_per_shelf : ℕ := 8
def mystery_shelves : ℕ := 12
def picture_shelves : ℕ := 9

-- Define the number of books on each type of shelves
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := picture_shelves * books_per_shelf

-- Define the statement to prove
theorem total_books : total_mystery_books + total_picture_books = 168 := by
  sorry

end total_books_l241_241342


namespace infinite_points_with_sum_of_squares_condition_l241_241620

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle centered at origin with given radius
def isWithinCircle (P : Point2D) (r : ℝ) :=
  P.x^2 + P.y^2 ≤ r^2

-- Define the distance squared from a point to another point
def dist2 (P Q : Point2D) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the problem
theorem infinite_points_with_sum_of_squares_condition :
  ∃ P : Point2D, isWithinCircle P 1 → (dist2 P ⟨-1, 0⟩ + dist2 P ⟨1, 0⟩ = 3) :=
by  
  sorry

end infinite_points_with_sum_of_squares_condition_l241_241620


namespace inequality_proof_l241_241546

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end inequality_proof_l241_241546


namespace sqrt_factorial_div_l241_241900

theorem sqrt_factorial_div:
  Real.sqrt (↑(Nat.factorial 9) / 90) = 4 * Real.sqrt 42 := 
by
  -- Steps of the proof
  sorry

end sqrt_factorial_div_l241_241900


namespace smallest_six_digit_odd_div_by_125_l241_241513

theorem smallest_six_digit_odd_div_by_125 : 
  ∃ n : ℕ, n = 111375 ∧ 
           100000 ≤ n ∧ n < 1000000 ∧ 
           (∀ d : ℕ, d ∈ (n.digits 10) → d % 2 = 1) ∧ 
           n % 125 = 0 :=
by
  sorry

end smallest_six_digit_odd_div_by_125_l241_241513


namespace adjugate_power_null_l241_241251

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)

def adjugate (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ := sorry

theorem adjugate_power_null (A : Matrix (Fin n) (Fin n) ℂ) (m : ℕ) (hm : 0 < m) (h : (adjugate A) ^ m = 0) : 
  (adjugate A) ^ 2 = 0 := 
sorry

end adjugate_power_null_l241_241251


namespace vertical_axis_residuals_of_residual_plot_l241_241940

theorem vertical_axis_residuals_of_residual_plot :
  ∀ (vertical_axis : Type), 
  (vertical_axis = Residuals ∨ 
   vertical_axis = SampleNumber ∨ 
   vertical_axis = EstimatedValue) →
  (vertical_axis = Residuals) :=
by
  sorry

end vertical_axis_residuals_of_residual_plot_l241_241940


namespace find_m_value_l241_241229

theorem find_m_value :
  ∃ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 ∧ m = 11 :=
by
  -- The proof would follow here.
  sorry

end find_m_value_l241_241229


namespace max_sum_x_y_min_diff_x_y_l241_241786

def circle_points (x y : ℤ) : Prop := (x - 1)^2 + (y + 2)^2 = 36

theorem max_sum_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x + y ≥ x' + y') :=
  by sorry

theorem min_diff_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x - y ≤ x' - y') :=
  by sorry

end max_sum_x_y_min_diff_x_y_l241_241786


namespace identity_function_l241_241458

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : ∀ n : ℕ, f n = n :=
by
  sorry

end identity_function_l241_241458


namespace tangent_circle_line_l241_241152

theorem tangent_circle_line (r : ℝ) (h_pos : 0 < r) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 = r^2) 
  (h_line : ∀ x y : ℝ, x + y = r + 1) : 
  r = 1 + Real.sqrt 2 := 
by 
  sorry

end tangent_circle_line_l241_241152


namespace sum_powers_of_ab_l241_241226

theorem sum_powers_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1)
  (h3 : a^2 + b^2 = 7) (h4 : a^3 + b^3 = 18) (h5 : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 :=
sorry

end sum_powers_of_ab_l241_241226


namespace positive_numbers_inequality_l241_241596

theorem positive_numbers_inequality
  (x y z : ℝ)
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x * y + y * z + z * x = 6) :
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
   1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
   1 / (2 * Real.sqrt 2 + z^2 * (x + y))) <= 
  (1 / (x * y * z)) :=
by
  sorry

end positive_numbers_inequality_l241_241596


namespace polygon_diagonals_formula_l241_241063

theorem polygon_diagonals_formula (n : ℕ) (h₁ : n = 5) (h₂ : 2 * n = (n * (n - 3)) / 2) :
  ∃ D : ℕ, D = n * (n - 3) / 2 :=
by
  sorry

end polygon_diagonals_formula_l241_241063


namespace tan_alpha_value_trigonometric_expression_value_l241_241483

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  Real.tan α = 2 :=
sorry

theorem trigonometric_expression_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  (4 * Real.sin (π - α) + 2 * Real.cos (2 * π - α)) / (Real.sin (π / 2 - α) + Real.sin (-α)) = -10 := 
sorry

end tan_alpha_value_trigonometric_expression_value_l241_241483


namespace isabella_hair_length_end_of_year_l241_241541

/--
Isabella's initial hair length.
-/
def initial_hair_length : ℕ := 18

/--
Isabella's hair growth over the year.
-/
def hair_growth : ℕ := 6

/--
Prove that Isabella's hair length at the end of the year is 24 inches.
-/
theorem isabella_hair_length_end_of_year : initial_hair_length + hair_growth = 24 := by
  sorry

end isabella_hair_length_end_of_year_l241_241541


namespace minimize_PA2_plus_PB2_plus_PC2_l241_241381

def PA (x y : ℝ) : ℝ := (x - 3) ^ 2 + (y + 1) ^ 2
def PB (x y : ℝ) : ℝ := (x + 1) ^ 2 + (y - 4) ^ 2
def PC (x y : ℝ) : ℝ := (x - 1) ^ 2 + (y + 6) ^ 2

theorem minimize_PA2_plus_PB2_plus_PC2 :
  ∃ x y : ℝ, (PA x y + PB x y + PC x y) = 64 :=
by
  use 1
  use -1
  simp [PA, PB, PC]
  sorry

end minimize_PA2_plus_PB2_plus_PC2_l241_241381


namespace original_bet_is_40_l241_241133

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end original_bet_is_40_l241_241133


namespace B_profit_percentage_l241_241703

theorem B_profit_percentage (cost_price_A : ℝ) (profit_A : ℝ) (selling_price_C : ℝ) 
  (h1 : cost_price_A = 154) 
  (h2 : profit_A = 0.20) 
  (h3 : selling_price_C = 231) : 
  (selling_price_C - (cost_price_A * (1 + profit_A))) / (cost_price_A * (1 + profit_A)) * 100 = 25 :=
by
  sorry

end B_profit_percentage_l241_241703


namespace hypotenuse_length_is_13_l241_241514

theorem hypotenuse_length_is_13 (a b c : ℝ) (ha : a = 5) (hb : b = 12)
  (hrt : a ^ 2 + b ^ 2 = c ^ 2) : c = 13 :=
by
  -- to complete the proof, fill in the details here
  sorry

end hypotenuse_length_is_13_l241_241514


namespace area_of_circumscribed_circle_l241_241289

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l241_241289


namespace intersection_of_perpendicular_lines_l241_241531

theorem intersection_of_perpendicular_lines (x y : ℝ) : 
  (y = 3 * x + 4) ∧ (y = -1/3 * x + 4) → (x = 0 ∧ y = 4) :=
by
  sorry

end intersection_of_perpendicular_lines_l241_241531


namespace Kyle_monthly_income_l241_241339

theorem Kyle_monthly_income :
  let rent := 1250
  let utilities := 150
  let retirement_savings := 400
  let groceries_eatingout := 300
  let insurance := 200
  let miscellaneous := 200
  let car_payment := 350
  let gas_maintenance := 350
  rent + utilities + retirement_savings + groceries_eatingout + insurance + miscellaneous + car_payment + gas_maintenance = 3200 :=
by
  -- Informal proof was provided in the solution.
  sorry

end Kyle_monthly_income_l241_241339


namespace tan_fraction_identity_l241_241041

theorem tan_fraction_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
by 
  sorry

end tan_fraction_identity_l241_241041


namespace pairings_without_alice_and_bob_l241_241268

theorem pairings_without_alice_and_bob (n : ℕ) (h : n = 12) : 
    ∃ k : ℕ, k = ((n * (n - 1)) / 2) - 1 ∧ k = 65 :=
by
  sorry

end pairings_without_alice_and_bob_l241_241268


namespace probability_even_sum_is_half_l241_241236

-- Definitions for probability calculations
def prob_even_A : ℚ := 2 / 5
def prob_odd_A : ℚ := 3 / 5
def prob_even_B : ℚ := 1 / 2
def prob_odd_B : ℚ := 1 / 2

-- Sum is even if both are even or both are odd
def prob_even_sum := prob_even_A * prob_even_B + prob_odd_A * prob_odd_B

-- Theorem stating the final probability
theorem probability_even_sum_is_half : prob_even_sum = 1 / 2 := by
  sorry

end probability_even_sum_is_half_l241_241236


namespace car_b_speed_l241_241114

theorem car_b_speed (v : ℕ) (h1 : ∀ (v : ℕ), CarA_speed = 3 * v)
                   (h2 : ∀ (time : ℕ), CarA_time = 6)
                   (h3 : ∀ (time : ℕ), CarB_time = 2)
                   (h4 : Car_total_distance = 1000) :
    v = 50 :=
by
  sorry

end car_b_speed_l241_241114


namespace Eliane_schedule_combinations_l241_241338

def valid_schedule_combinations : ℕ :=
  let mornings := 6 * 3 -- 6 days (Monday to Saturday) each with 3 time slots
  let afternoons := 5 * 2 -- 5 days (Monday to Friday) each with 2 time slots
  let mon_or_fri_comb := 2 * 3 * 3 * 2 -- Morning on Monday or Friday
  let sat_comb := 1 * 3 * 4 * 2 -- Morning on Saturday
  let tue_wed_thu_comb := 3 * 3 * 2 * 2 -- Morning on Tuesday, Wednesday, or Thursday
  mon_or_fri_comb + sat_comb + tue_wed_thu_comb

theorem Eliane_schedule_combinations :
  valid_schedule_combinations = 96 := by
  sorry

end Eliane_schedule_combinations_l241_241338


namespace weight_of_person_replaced_l241_241161

def initial_total_weight (W : ℝ) : ℝ := W
def new_person_weight : ℝ := 137
def average_increase : ℝ := 7.2
def group_size : ℕ := 10

theorem weight_of_person_replaced 
(W : ℝ) 
(weight_replaced : ℝ) 
(h1 : (W / group_size) + average_increase = (W - weight_replaced + new_person_weight) / group_size) : 
weight_replaced = 65 := 
sorry

end weight_of_person_replaced_l241_241161


namespace inequality_preserved_l241_241287

variable {a b c : ℝ}

theorem inequality_preserved (h : abs ((a^2 + b^2 - c^2) / (a * b)) < 2) :
    abs ((b^2 + c^2 - a^2) / (b * c)) < 2 ∧ abs ((c^2 + a^2 - b^2) / (c * a)) < 2 := 
sorry

end inequality_preserved_l241_241287


namespace pow_mod_equality_l241_241148

theorem pow_mod_equality (h : 2^3 ≡ 1 [MOD 7]) : 2^30 ≡ 1 [MOD 7] :=
sorry

end pow_mod_equality_l241_241148


namespace constant_ratio_of_arithmetic_sequence_l241_241653

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n-1) * d

-- The main theorem stating the result
theorem constant_ratio_of_arithmetic_sequence 
  (a : ℕ → ℝ) (c : ℝ) (h_seq : arithmetic_sequence a)
  (h_const : ∀ n : ℕ, a n ≠ 0 ∧ a (2 * n) ≠ 0 ∧ a n / a (2 * n) = c) :
  c = 1 ∨ c = 1 / 2 :=
sorry

end constant_ratio_of_arithmetic_sequence_l241_241653


namespace race_time_l241_241183

theorem race_time (v_A v_B : ℝ) (t_A t_B : ℝ) (h1 : v_A = 1000 / t_A) (h2 : v_B = 952 / (t_A + 6)) (h3 : v_A = v_B) : t_A = 125 :=
by
  sorry

end race_time_l241_241183


namespace correctly_calculated_value_l241_241543

theorem correctly_calculated_value :
  ∀ (x : ℕ), (x * 15 = 45) → ((x * 5) * 10 = 150) := 
by
  intro x
  intro h
  sorry

end correctly_calculated_value_l241_241543


namespace cross_section_is_rectangle_l241_241169

def RegularTetrahedron : Type := sorry

def Plane : Type := sorry

variable (T : RegularTetrahedron) (P : Plane)

-- Conditions
axiom regular_tetrahedron (T : RegularTetrahedron) : Prop
axiom plane_intersects_tetrahedron (P : Plane) (T : RegularTetrahedron) : Prop
axiom plane_parallel_opposite_edges (P : Plane) (T : RegularTetrahedron) : Prop

-- The cross-section formed by intersecting a regular tetrahedron with a plane
-- that is parallel to two opposite edges is a rectangle.
theorem cross_section_is_rectangle (T : RegularTetrahedron) (P : Plane) 
  (hT : regular_tetrahedron T) 
  (hI : plane_intersects_tetrahedron P T) 
  (hP : plane_parallel_opposite_edges P T) :
  ∃ (shape : Type), shape = Rectangle := 
  sorry

end cross_section_is_rectangle_l241_241169


namespace quadratic_intersection_l241_241365

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  ∃ x y : ℝ, (y = a * x^2 + b * x + c) ∧ (y = a * (x - h)^2 + b * (x - h) + d)
    ∧ x = (d - c) / b
    ∧ y = a * (d - c)^2 / b^2 + d :=
by {
  sorry
}

end quadratic_intersection_l241_241365


namespace f_x_minus_1_pass_through_l241_241515

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + x

theorem f_x_minus_1_pass_through (a : ℝ) : f a (1 - 1) = 0 :=
by
  -- Proof is omitted here
  sorry

end f_x_minus_1_pass_through_l241_241515


namespace roots_quadratic_sum_of_squares_l241_241206

theorem roots_quadratic_sum_of_squares :
  ∀ x1 x2 : ℝ, (x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) → x1^2 + x2^2 = 6 :=
by
  intros x1 x2 h
  -- proof goes here
  sorry

end roots_quadratic_sum_of_squares_l241_241206


namespace jason_work_hours_l241_241872

variable (x y : ℕ)

def working_hours : Prop :=
  (4 * x + 6 * y = 88) ∧
  (x + y = 18)

theorem jason_work_hours (h : working_hours x y) : y = 8 :=
  by
    sorry

end jason_work_hours_l241_241872


namespace will_pages_needed_l241_241419

theorem will_pages_needed :
  let new_cards_2020 := 8
  let old_cards := 10
  let duplicates := 2
  let cards_per_page := 3
  let unique_old_cards := old_cards - duplicates
  let pages_needed_for_2020 := (new_cards_2020 + cards_per_page - 1) / cards_per_page -- ceil(new_cards_2020 / cards_per_page)
  let pages_needed_for_old := (unique_old_cards + cards_per_page - 1) / cards_per_page -- ceil(unique_old_cards / cards_per_page)
  let pages_needed := pages_needed_for_2020 + pages_needed_for_old
  pages_needed = 6 :=
by
  sorry

end will_pages_needed_l241_241419


namespace hockey_season_duration_l241_241393

theorem hockey_season_duration 
  (total_games : ℕ)
  (games_per_month : ℕ)
  (h_total : total_games = 182)
  (h_monthly : games_per_month = 13) : 
  total_games / games_per_month = 14 := 
by
  sorry

end hockey_season_duration_l241_241393


namespace lotion_cost_l241_241586

variable (shampoo_conditioner_cost lotion_total_spend: ℝ)
variable (num_lotions num_lotions_cost_target: ℕ)
variable (free_shipping_threshold additional_spend_needed: ℝ)

noncomputable def cost_of_each_lotion := lotion_total_spend / num_lotions

theorem lotion_cost
    (h1 : shampoo_conditioner_cost = 10)
    (h2 : num_lotions = 3)
    (h3 : additional_spend_needed = 12)
    (h4 : free_shipping_threshold = 50)
    (h5 : (shampoo_conditioner_cost * 2) + additional_spend_needed + lotion_total_spend = free_shipping_threshold) :
    cost_of_each_lotion = 10 :=
by
  sorry

end lotion_cost_l241_241586


namespace car_speed_l241_241322

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end car_speed_l241_241322


namespace value_of_S6_l241_241320

theorem value_of_S6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 :=
by sorry

end value_of_S6_l241_241320


namespace Razorback_total_revenue_l241_241222

def t_shirt_price : ℕ := 51
def t_shirt_discount : ℕ := 8
def hat_price : ℕ := 28
def hat_discount : ℕ := 5
def t_shirts_sold : ℕ := 130
def hats_sold : ℕ := 85

def discounted_t_shirt_price : ℕ := t_shirt_price - t_shirt_discount
def discounted_hat_price : ℕ := hat_price - hat_discount

def revenue_from_t_shirts : ℕ := t_shirts_sold * discounted_t_shirt_price
def revenue_from_hats : ℕ := hats_sold * discounted_hat_price

def total_revenue : ℕ := revenue_from_t_shirts + revenue_from_hats

theorem Razorback_total_revenue : total_revenue = 7545 := by
  unfold total_revenue
  unfold revenue_from_t_shirts
  unfold revenue_from_hats
  unfold discounted_t_shirt_price
  unfold discounted_hat_price
  unfold t_shirts_sold
  unfold hats_sold
  unfold t_shirt_price
  unfold t_shirt_discount
  unfold hat_price
  unfold hat_discount
  sorry

end Razorback_total_revenue_l241_241222


namespace fraction_of_quarters_in_1790s_l241_241197

theorem fraction_of_quarters_in_1790s (total_coins : ℕ) (coins_in_1790s : ℕ) :
  total_coins = 30 ∧ coins_in_1790s = 7 → 
  (coins_in_1790s : ℚ) / total_coins = 7 / 30 :=
by
  sorry

end fraction_of_quarters_in_1790s_l241_241197


namespace positive_difference_of_perimeters_is_zero_l241_241368

-- Definitions of given conditions
def rect1_length : ℕ := 5
def rect1_width : ℕ := 1
def rect2_first_rect_length : ℕ := 3
def rect2_first_rect_width : ℕ := 2
def rect2_second_rect_length : ℕ := 1
def rect2_second_rect_width : ℕ := 2

-- Perimeter calculation functions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def rect1_perimeter := perimeter rect1_length rect1_width
def rect2_extended_length : ℕ := rect2_first_rect_length + rect2_second_rect_length
def rect2_extended_width : ℕ := rect2_first_rect_width
def rect2_perimeter := perimeter rect2_extended_length rect2_extended_width

-- The positive difference of the perimeters
def positive_difference (a b : ℕ) : ℕ := if a > b then a - b else b - a

-- The Lean 4 statement to be proven
theorem positive_difference_of_perimeters_is_zero :
    positive_difference rect1_perimeter rect2_perimeter = 0 := by
  sorry

end positive_difference_of_perimeters_is_zero_l241_241368


namespace min_sum_abc_l241_241881

theorem min_sum_abc (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c + b * c + c = 2014) : a + b + c = 40 :=
sorry

end min_sum_abc_l241_241881


namespace range_y_eq_2cosx_minus_1_range_y_eq_sq_2sinx_minus_1_plus_3_l241_241379

open Real

theorem range_y_eq_2cosx_minus_1 : 
  (∀ x : ℝ, -1 ≤ cos x ∧ cos x ≤ 1) →
  (∀ y : ℝ, y = 2 * (cos x) - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  intros h1 y h2
  sorry

theorem range_y_eq_sq_2sinx_minus_1_plus_3 : 
  (∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1) →
  (∀ y : ℝ, y = (2 * (sin x) - 1)^2 + 3 → 3 ≤ y ∧ y ≤ 12) :=
by
  intros h1 y h2
  sorry

end range_y_eq_2cosx_minus_1_range_y_eq_sq_2sinx_minus_1_plus_3_l241_241379


namespace reflect_across_x_axis_l241_241468

-- Definitions for the problem conditions
def initial_point : ℝ × ℝ := (-2, 1)
def reflected_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The statement to be proved
theorem reflect_across_x_axis :
  reflected_point initial_point = (-2, -1) :=
  sorry

end reflect_across_x_axis_l241_241468


namespace simplify_cbrt_expr_l241_241993

-- Define the cube root function.
def cbrt (x : ℝ) : ℝ := x^(1/3)

-- Define the original expression under the cube root.
def original_expr : ℝ := 40^3 + 70^3 + 100^3

-- Define the simplified expression.
def simplified_expr : ℝ := 10 * cbrt 1407

theorem simplify_cbrt_expr : cbrt original_expr = simplified_expr := by
  -- Declaration that proof is not provided to ensure Lean statement is complete.
  sorry

end simplify_cbrt_expr_l241_241993


namespace john_spends_on_memory_cards_l241_241792

theorem john_spends_on_memory_cards :
  (10 * (3 * 365)) / 50 * 60 = 13140 :=
by
  sorry

end john_spends_on_memory_cards_l241_241792


namespace pencils_per_box_l241_241741

theorem pencils_per_box (boxes : ℕ) (total_pencils : ℕ) (h1 : boxes = 3) (h2 : total_pencils = 27) : (total_pencils / boxes) = 9 := 
by
  sorry

end pencils_per_box_l241_241741


namespace neznaika_incorrect_l241_241721

-- Define the average consumption conditions
def average_consumption_december (total_consumption total_days_cons_december : ℕ) : Prop :=
  total_consumption = 10 * total_days_cons_december

def average_consumption_january (total_consumption total_days_cons_january : ℕ) : Prop :=
  total_consumption = 5 * total_days_cons_january

-- Define the claim to be disproven
def neznaika_claim (days_december_at_least_10 days_january_at_least_10 : ℕ) : Prop :=
  days_december_at_least_10 > days_january_at_least_10

-- Proof statement that the claim is incorrect
theorem neznaika_incorrect (total_days_cons_december total_days_cons_january total_consumption_dec total_consumption_jan : ℕ)
    (days_december_at_least_10 days_january_at_least_10 : ℕ)
    (h1 : average_consumption_december total_consumption_dec total_days_cons_december)
    (h2 : average_consumption_january total_consumption_jan total_days_cons_january)
    (h3 : total_days_cons_december = 31)
    (h4 : total_days_cons_january = 31)
    (h5 : days_december_at_least_10 ≤ total_days_cons_december)
    (h6 : days_january_at_least_10 ≤ total_days_cons_january)
    (h7 : days_december_at_least_10 = 1)
    (h8 : days_january_at_least_10 = 15) : 
    ¬ neznaika_claim days_december_at_least_10 days_january_at_least_10 :=
by
  sorry

end neznaika_incorrect_l241_241721


namespace exercise_l241_241527

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l241_241527


namespace find_a_l241_241968

theorem find_a 
  (x y a : ℝ)
  (h₁ : x - 3 ≤ 0)
  (h₂ : y - a ≤ 0)
  (h₃ : x + y ≥ 0)
  (h₄ : ∃ (x y : ℝ), 2*x + y = 10): a = 4 :=
sorry

end find_a_l241_241968


namespace smallest_same_terminal_1000_l241_241932

def has_same_terminal_side (theta phi : ℝ) : Prop :=
  ∃ n : ℤ, theta = phi + n * 360

theorem smallest_same_terminal_1000 : ∀ θ : ℝ,
  θ ≥ 0 → θ < 360 → has_same_terminal_side θ 1000 → θ = 280 :=
by
  sorry

end smallest_same_terminal_1000_l241_241932


namespace distribution_count_l241_241212

-- Making the function for counting the number of valid distributions
noncomputable def countValidDistributions : ℕ :=
  let cases1 := 4                            -- One box contains all five balls
  let cases2 := 4 * 3                        -- One box has 4 balls, another has 1
  let cases3 := 4 * 3                        -- One box has 3 balls, another has 2
  let cases4 := 6 * 2                        -- Two boxes have 2 balls, and one has 1
  let cases5 := 4 * 3                        -- One box has 3 balls, and two boxes have 1 each
  cases1 + cases2 + cases3 + cases4 + cases5 -- Sum of all cases

-- Theorem statement: the count of valid distributions equals 52
theorem distribution_count : countValidDistributions = 52 := 
  by
    sorry

end distribution_count_l241_241212


namespace quadrilateral_sides_l241_241088

noncomputable def circle_radius : ℝ := 25
noncomputable def diagonal1_length : ℝ := 48
noncomputable def diagonal2_length : ℝ := 40

theorem quadrilateral_sides :
  ∃ (a b c d : ℝ),
    (a = 5 * Real.sqrt 10 ∧ 
    b = 9 * Real.sqrt 10 ∧ 
    c = 13 * Real.sqrt 10 ∧ 
    d = 15 * Real.sqrt 10) ∧ 
    (diagonal1_length = 48 ∧ 
    diagonal2_length = 40 ∧ 
    circle_radius = 25) :=
sorry

end quadrilateral_sides_l241_241088


namespace circles_ordered_by_radius_l241_241549

def circle_radii_ordered (rA rB rC : ℝ) : Prop :=
  rA < rC ∧ rC < rB

theorem circles_ordered_by_radius :
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  circle_radii_ordered rA rB rC :=
by
  intros
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  show circle_radii_ordered rA rB rC
  sorry

end circles_ordered_by_radius_l241_241549


namespace prove_angle_sum_l241_241521

open Real

theorem prove_angle_sum (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : cos α / sin β + cos β / sin α = 2) : 
  α + β = π / 2 := 
sorry

end prove_angle_sum_l241_241521


namespace rectangular_garden_side_length_l241_241141

theorem rectangular_garden_side_length (a b : ℝ) (h1 : 2 * a + 2 * b = 60) (h2 : a * b = 200) (h3 : b = 10) : a = 20 :=
by
  sorry

end rectangular_garden_side_length_l241_241141


namespace relationship_between_a_b_c_l241_241728

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (Real.cos (34 * Real.pi / 180) - Real.sin (34 * Real.pi / 180))
noncomputable def b : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (128 * Real.pi / 180) + Real.cos (40 * Real.pi / 180) * Real.cos (38 * Real.pi / 180)
noncomputable def c : ℝ := (1 / 2) * (Real.cos (80 * Real.pi / 180) - 2 * (Real.cos (50 * Real.pi / 180))^2 + 1)

theorem relationship_between_a_b_c : b > a ∧ a > c :=
  sorry

end relationship_between_a_b_c_l241_241728


namespace stratified_sampling_counts_l241_241562

-- Defining the given conditions
def num_elderly : ℕ := 27
def num_middle_aged : ℕ := 54
def num_young : ℕ := 81
def total_sample : ℕ := 42

-- Proving the required stratified sample counts
theorem stratified_sampling_counts :
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  elderly_count = 7 ∧ middle_aged_count = 14 ∧ young_count = 21 :=
by 
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  have h1 : elderly_count = 7 := by sorry
  have h2 : middle_aged_count = 14 := by sorry
  have h3 : young_count = 21 := by sorry
  exact ⟨h1, h2, h3⟩

end stratified_sampling_counts_l241_241562


namespace option_A_option_C_l241_241313

variable {a : ℕ → ℝ} (q : ℝ)
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n + 1) = q * (a n)

def decreasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n > a (n + 1)

theorem option_A (h₁ : a 1 > 0) (hq : geometric_sequence a q) : 0 < q ∧ q < 1 → decreasing_sequence a := 
  sorry

theorem option_C (h₁ : a 1 < 0) (hq : geometric_sequence a q) : q > 1 → decreasing_sequence a := 
  sorry

end option_A_option_C_l241_241313


namespace power_greater_than_one_million_l241_241484

theorem power_greater_than_one_million (α β γ δ : ℝ) (ε ζ η : ℕ)
  (h1 : α = 1.01) (h2 : β = 1.001) (h3 : γ = 1.000001) 
  (h4 : δ = 1000000) 
  (h_eps : ε = 99999900) (h_zet : ζ = 999999000) (h_eta : η = 999999000000) :
  α^ε > δ ∧ β^ζ > δ ∧ γ^η > δ :=
by
  sorry

end power_greater_than_one_million_l241_241484


namespace range_of_a_l241_241999

open Set

def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ 2 * a + 1 }
def B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

theorem range_of_a (a : ℝ) (h : A a ∪ B = B) : a ∈ Iio (-2) ∪ Icc (-1) (3 / 2) :=
by
  sorry

end range_of_a_l241_241999


namespace sequence_convergence_l241_241018

noncomputable def alpha : ℝ := sorry
def bounded (a : ℕ → ℝ) : Prop := ∃ M > 0, ∀ n, ‖a n‖ ≤ M

-- Translation of the math problem
theorem sequence_convergence (a : ℕ → ℝ) (ha : bounded a) (hα : 0 < alpha ∧ alpha ≤ 1) 
  (ineq : ∀ n ≥ 2, a (n+1) ≤ alpha * a n + (1 - alpha) * a (n-1)) : 
  ∃ l, ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖a n - l‖ < ε := 
sorry

end sequence_convergence_l241_241018


namespace simplify_expression_l241_241360

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) =
  16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end simplify_expression_l241_241360


namespace fraction_of_repeating_decimal_l241_241557

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l241_241557


namespace electric_car_travel_distance_l241_241853

theorem electric_car_travel_distance {d_electric d_diesel : ℕ} 
  (h1 : d_diesel = 120) 
  (h2 : d_electric = d_diesel + 50 * d_diesel / 100) : 
  d_electric = 180 := 
by 
  sorry

end electric_car_travel_distance_l241_241853


namespace Jackson_money_is_125_l241_241617

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end Jackson_money_is_125_l241_241617


namespace problem1_problem2_l241_241842

-- Definition and conditions
def i := Complex.I

-- Problem 1
theorem problem1 : (2 + 2 * i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i)) ^ 2010 = -1 := 
by
  sorry

-- Problem 2
theorem problem2 : (4 - i^5) * (6 + 2 * i^7) + (7 + i^11) * (4 - 3 * i) = 47 - 39 * i := 
by
  sorry

end problem1_problem2_l241_241842


namespace find_de_l241_241297

def magic_square (f : ℕ × ℕ → ℕ) : Prop :=
  (f (0, 0) = 30) ∧ (f (0, 1) = 20) ∧ (f (0, 2) = f (0, 2)) ∧
  (f (1, 0) = f (1, 0)) ∧ (f (1, 1) = f (1, 1)) ∧ (f (1, 2) = f (1, 2)) ∧
  (f (2, 0) = 24) ∧ (f (2, 1) = 32) ∧ (f (2, 2) = f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (1, 0) + f (1, 1) + f (1, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (2, 0) + f (2, 1) + f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (0, 0) + f (1, 0) + f (2, 0)) ∧
  (f (0, 0) + f (1, 1) + f (2, 2) = f (0, 2) + f (1, 1) + f (2, 0)) 

theorem find_de (f : ℕ × ℕ → ℕ) (h : magic_square f) : 
  (f (1, 0) + f (1, 1) = 54) :=
sorry

end find_de_l241_241297


namespace find_origin_coordinates_l241_241718

variable (x y : ℝ)

def original_eq (x y : ℝ) := x^2 - y^2 - 2*x - 2*y - 1 = 0

def transformed_eq (x' y' : ℝ) := x'^2 - y'^2 = 1

theorem find_origin_coordinates (x y : ℝ) :
  original_eq (x - 1) (y + 1) ↔ transformed_eq x y :=
by
  sorry

end find_origin_coordinates_l241_241718


namespace solve_for_x_l241_241801

theorem solve_for_x (x : ℝ) (h : (x - 75) / 3 = (8 - 3 * x) / 4) : 
  x = 324 / 13 :=
sorry

end solve_for_x_l241_241801


namespace contradiction_example_l241_241840

theorem contradiction_example (a b c d : ℝ) 
(h1 : a + b = 1) 
(h2 : c + d = 1) 
(h3 : ac + bd > 1) : 
¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_example_l241_241840


namespace distribution_scheme_count_l241_241813

-- Definitions based on conditions
variable (village1 village2 village3 village4 : Type)
variables (quota1 quota2 quota3 quota4 : ℕ)

-- Conditions as given in the problem
def valid_distribution (v1 v2 v3 v4 : ℕ) : Prop :=
  v1 = 1 ∧ v2 = 2 ∧ v3 = 3 ∧ v4 = 4

-- The goal is to prove the number of permutations is equal to 24
theorem distribution_scheme_count :
  (∃ v1 v2 v3 v4 : ℕ, valid_distribution v1 v2 v3 v4) → 
  (4 * 3 * 2 * 1 = 24) :=
by 
  sorry

end distribution_scheme_count_l241_241813


namespace volleyball_count_l241_241094

theorem volleyball_count (x y z : ℕ) (h1 : x + y + z = 20) (h2 : 6 * x + 3 * y + z = 33) : z = 15 :=
by
  sorry

end volleyball_count_l241_241094


namespace Jason_seashells_l241_241895

theorem Jason_seashells (initial_seashells given_to_Tim remaining_seashells : ℕ) :
  initial_seashells = 49 → given_to_Tim = 13 → remaining_seashells = initial_seashells - given_to_Tim →
  remaining_seashells = 36 :=
by intros; sorry

end Jason_seashells_l241_241895


namespace b_range_l241_241317

noncomputable def f (a b x : ℝ) := (x - 1) * Real.log x - a * x + a + b

theorem b_range (a b : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = 0 ∧ f a b x2 = 0) :
  b < 0 :=
sorry

end b_range_l241_241317


namespace phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l241_241144

def even_digits : Set ℕ := { 0, 2, 4, 6, 8 }
def odd_digits : Set ℕ := { 1, 3, 5, 7, 9 }

theorem phone_numbers_even : (4 * 5^6) = 62500 := by
  sorry

theorem phone_numbers_odd : 5^7 = 78125 := by
  sorry

theorem phone_numbers_ratio
  (evens : (4 * 5^6) = 62500)
  (odds : 5^7 = 78125) :
  (78125 / 62500 : ℝ) = 1.25 := by
    sorry

end phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l241_241144


namespace volume_of_box_l241_241441

-- Define the dimensions of the box
variables (L W H : ℝ)

-- Define the conditions as hypotheses
def side_face_area : Prop := H * W = 288
def top_face_area : Prop := L * W = 1.5 * 288
def front_face_area : Prop := L * H = 0.5 * (L * W)

-- Define the volume of the box
def box_volume : ℝ := L * W * H

-- The proof statement
theorem volume_of_box (h1 : side_face_area H W) (h2 : top_face_area L W) (h3 : front_face_area L H W) : box_volume L W H = 5184 :=
by
  sorry

end volume_of_box_l241_241441


namespace lisa_speed_correct_l241_241929

def eugene_speed := 5

def carlos_speed := (3 / 4) * eugene_speed

def lisa_speed := (4 / 3) * carlos_speed

theorem lisa_speed_correct : lisa_speed = 5 := by
  sorry

end lisa_speed_correct_l241_241929


namespace length_sum_l241_241030

theorem length_sum : 
  let m := 1 -- Meter as base unit
  let cm := 0.01 -- 1 cm in meters
  let mm := 0.001 -- 1 mm in meters
  2 * m + 3 * cm + 5 * mm = 2.035 * m :=
by sorry

end length_sum_l241_241030


namespace find_number_l241_241665

theorem find_number (N : ℝ) 
  (h1 : (5 / 6) * N = (5 / 16) * N + 200) : 
  N = 384 :=
sorry

end find_number_l241_241665


namespace largest_x_to_floor_ratio_l241_241111

theorem largest_x_to_floor_ratio : ∃ x : ℝ, (⌊x⌋ / x = 9 / 10 ∧ ∀ y : ℝ, (⌊y⌋ / y = 9 / 10 → y ≤ x)) :=
sorry

end largest_x_to_floor_ratio_l241_241111


namespace smallest_sum_of_two_squares_l241_241834

theorem smallest_sum_of_two_squares :
  ∃ n : ℕ, (∀ m : ℕ, m < n → (¬ (∃ a b c d e f : ℕ, m = a^2 + b^2 ∧  m = c^2 + d^2 ∧ m = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))))) ∧
          (∃ a b c d e f : ℕ, n = a^2 + b^2 ∧  n = c^2 + d^2 ∧ n = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))) :=
sorry

end smallest_sum_of_two_squares_l241_241834


namespace jim_caught_fish_l241_241969

variable (ben judy billy susie jim caught_back total_filets : ℕ)

def caught_fish : ℕ :=
  ben + judy + billy + susie + jim - caught_back

theorem jim_caught_fish (h_ben : ben = 4)
                        (h_judy : judy = 1)
                        (h_billy : billy = 3)
                        (h_susie : susie = 5)
                        (h_caught_back : caught_back = 3)
                        (h_total_filets : total_filets = 24)
                        (h_filets_per_fish : ∀ f : ℕ, total_filets = f * 2 → caught_fish ben judy billy susie jim caught_back = f) :
  jim = 2 :=
by
  -- Proof goes here
  sorry

end jim_caught_fish_l241_241969


namespace loaned_books_count_l241_241645

variable (x : ℕ) -- x is the number of books loaned out during the month

theorem loaned_books_count 
  (initial_books : ℕ) (returned_percentage : ℚ) (remaining_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : returned_percentage = 0.80)
  (h3 : remaining_books = 66) :
  x = 45 :=
by
  -- Proof can be inserted here
  sorry

end loaned_books_count_l241_241645


namespace number_satisfies_equation_l241_241136

theorem number_satisfies_equation :
  ∃ x : ℝ, (x^2 + 100 = (x - 20)^2) ∧ x = 7.5 :=
by
  use 7.5
  sorry

end number_satisfies_equation_l241_241136


namespace smallest_sum_of_four_consecutive_primes_divisible_by_five_l241_241563

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    b = a + 2 ∧ c = b + 4 ∧ d = c + 2 ∧
    (a + b + c + d) % 5 = 0 ∧ (a + b + c + d = 60) := sorry

end smallest_sum_of_four_consecutive_primes_divisible_by_five_l241_241563


namespace sam_sandwich_shop_cost_l241_241867

theorem sam_sandwich_shop_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let fries_cost := 2
  let num_sandwiches := 3
  let num_sodas := 7
  let num_fries := 5
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_fries * fries_cost
  total_cost = 43 :=
by
  sorry

end sam_sandwich_shop_cost_l241_241867


namespace sqrt_of_four_is_pm_two_l241_241652

theorem sqrt_of_four_is_pm_two (y : ℤ) : y * y = 4 → y = 2 ∨ y = -2 := by
  sorry

end sqrt_of_four_is_pm_two_l241_241652


namespace max_value_of_y_in_interval_l241_241662

theorem max_value_of_y_in_interval (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : 
  ∃ y_max, ∀ x, 0 < x ∧ x < 1 / 3 → x * (1 - 3 * x) ≤ y_max ∧ y_max = 1 / 12 :=
by sorry

end max_value_of_y_in_interval_l241_241662


namespace range_of_m_for_point_in_second_quadrant_l241_241607

theorem range_of_m_for_point_in_second_quadrant (m : ℝ) :
  (m - 3 < 0) ∧ (m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  -- The proof will be inserted here.
  sorry

end range_of_m_for_point_in_second_quadrant_l241_241607


namespace exams_in_fourth_year_l241_241153

noncomputable def student_exam_counts 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 ∧ 
  a_5 = 3 * a_1 ∧ 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5

theorem exams_in_fourth_year 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) (h : student_exam_counts a_1 a_2 a_3 a_4 a_5) : 
  a_4 = 8 :=
sorry

end exams_in_fourth_year_l241_241153


namespace no_real_roots_for_pair_2_2_3_l241_241608

noncomputable def discriminant (A B : ℝ) : ℝ :=
  let a := 1 - 2 * B
  let b := -B
  let c := -A + A * B
  b ^ 2 - 4 * a * c

theorem no_real_roots_for_pair_2_2_3 : discriminant 2 (2 / 3) < 0 := by
  sorry

end no_real_roots_for_pair_2_2_3_l241_241608


namespace real_roots_quadratic_l241_241118

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k - 6 = 0) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
by {
  sorry
}

end real_roots_quadratic_l241_241118


namespace smallest_non_unit_digit_multiple_of_five_l241_241898

theorem smallest_non_unit_digit_multiple_of_five :
  ∀ (d : ℕ), ((d = 0) ∨ (d = 5)) → (d ≠ 1 ∧ d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) :=
by {
  sorry
}

end smallest_non_unit_digit_multiple_of_five_l241_241898


namespace calc_f_2005_2007_zero_l241_241199

variable {R : Type} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_function (f : R → R) (p : R) : Prop :=
  ∀ x, f (x + p) = f x

theorem calc_f_2005_2007_zero
  {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_period : periodic_function f 4) :
  f 2005 + f 2006 + f 2007 = 0 :=
sorry

end calc_f_2005_2007_zero_l241_241199


namespace omega_value_l241_241337

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem omega_value (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (h_x1 : f ω x₁ = -2) (h_x2 : f ω x₂ = 0) (h_min : |x₁ - x₂| = Real.pi) :
  ω = 1 / 2 := 
by 
  sorry

end omega_value_l241_241337


namespace simplify_expr_l241_241766

theorem simplify_expr (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) =
    8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 :=
by
  sorry

end simplify_expr_l241_241766


namespace range_of_a_l241_241844

noncomputable def f (x a : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 3 < a :=
by
  sorry

end range_of_a_l241_241844


namespace total_boxes_count_l241_241647

theorem total_boxes_count
  (initial_boxes : ℕ := 2013)
  (boxes_per_operation : ℕ := 13)
  (operations : ℕ := 2013)
  (non_empty_boxes : ℕ := 2013)
  (total_boxes : ℕ := initial_boxes + boxes_per_operation * operations) :
  non_empty_boxes = operations → total_boxes = 28182 :=
by
  sorry

end total_boxes_count_l241_241647


namespace find_a1_l241_241917

noncomputable def a (n : ℕ) : ℤ := sorry -- the definition of sequence a_n is not computable without initial terms
noncomputable def S (n : ℕ) : ℤ := sorry -- similarly, the definition of S_n without initial terms isn't given

axiom recurrence_relation (n : ℕ) (h : n ≥ 3): 
  a (n) = a (n - 1) - a (n - 2)

axiom S9 : S 9 = 6
axiom S10 : S 10 = 5

theorem find_a1 : a 1 = 1 :=
by
  sorry

end find_a1_l241_241917


namespace brad_reads_26_pages_per_day_l241_241843

-- Define conditions
def greg_daily_reading : ℕ := 18
def brad_extra_pages : ℕ := 8

-- Define Brad's daily reading
def brad_daily_reading : ℕ := greg_daily_reading + brad_extra_pages

-- The theorem to be proven
theorem brad_reads_26_pages_per_day : brad_daily_reading = 26 := by
  sorry

end brad_reads_26_pages_per_day_l241_241843


namespace factorization_correct_l241_241269

theorem factorization_correct: ∀ (x : ℝ), (x^2 - 9 = (x + 3) * (x - 3)) := 
sorry

end factorization_correct_l241_241269


namespace domain_of_f_l241_241404

noncomputable def f (x : ℝ) := 2 ^ (Real.sqrt (3 - x)) + 1 / (x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ (x ≤ 3 ∧ x ≠ 1) :=
by
  sorry

end domain_of_f_l241_241404


namespace range_of_k_for_ellipse_l241_241053

theorem range_of_k_for_ellipse (k : ℝ) :
  (4 - k > 0) ∧ (k - 1 > 0) ∧ (4 - k ≠ k - 1) ↔ (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  sorry

end range_of_k_for_ellipse_l241_241053


namespace upper_limit_of_people_l241_241477

theorem upper_limit_of_people (T : ℕ) (h1 : (3/7) * T = 24) (h2 : T > 50) : T ≤ 56 :=
by
  -- The steps to solve this proof would go here.
  sorry

end upper_limit_of_people_l241_241477


namespace emily_collected_total_eggs_l241_241014

def eggs_in_setA : ℕ := (200 * 36) + (250 * 24)
def eggs_in_setB : ℕ := (375 * 42) - 80
def eggs_in_setC : ℕ := (560 / 2 * 50) + (560 / 2 * 32)

def total_eggs_collected : ℕ := eggs_in_setA + eggs_in_setB + eggs_in_setC

theorem emily_collected_total_eggs : total_eggs_collected = 51830 := by
  -- proof goes here
  sorry

end emily_collected_total_eggs_l241_241014


namespace geometric_series_sum_l241_241737

theorem geometric_series_sum :
  let a := -1
  let r := -3
  let n := 8
  let S := (a * (r ^ n - 1)) / (r - 1)
  S = 1640 :=
by 
  sorry 

end geometric_series_sum_l241_241737


namespace find_pool_depth_l241_241424

noncomputable def pool_depth (rate volume capacity_percent time length width : ℝ) :=
  volume / (length * width * rate * time / capacity_percent)

theorem find_pool_depth :
  pool_depth 60 75000 0.8 1000 150 50 = 10 := by
  simp [pool_depth] -- Simplifying the complex expression should lead to the solution.
  sorry

end find_pool_depth_l241_241424


namespace sum_of_variables_is_16_l241_241331

theorem sum_of_variables_is_16 (A B C D E : ℕ)
    (h1 : C + E = 4) 
    (h2 : B + E = 7) 
    (h3 : B + D = 6) 
    (h4 : A = 6)
    (hdistinct : ∀ x y, x ≠ y → (x ≠ A ∧ x ≠ B ∧ x ≠ C ∧ x ≠ D ∧ x ≠ E) ∧ (y ≠ A ∧ y ≠ B ∧ y ≠ C ∧ y ≠ D ∧ y ≠ E)) :
    A + B + C + D + E = 16 :=
by
    sorry

end sum_of_variables_is_16_l241_241331


namespace polynomial_solution_l241_241667

theorem polynomial_solution (P : ℝ → ℝ) :
  (∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))) →
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x ^ 4 + β * x ^ 2 :=
by
  sorry

end polynomial_solution_l241_241667


namespace problem_statement_l241_241040

variable {x : Real}
variable {m : Int}
variable {n : Int}

theorem problem_statement (h1 : x^m = 5) (h2 : x^n = 10) : x^(2 * m - n) = 5 / 2 :=
by
  sorry

end problem_statement_l241_241040


namespace no_negative_roots_of_P_l241_241893

def P (x : ℝ) : ℝ := x^4 - 5 * x^3 + 3 * x^2 - 7 * x + 1

theorem no_negative_roots_of_P : ∀ x : ℝ, P x = 0 → x ≥ 0 := 
by 
    sorry

end no_negative_roots_of_P_l241_241893


namespace problem_l241_241238

theorem problem (a b c : ℝ) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end problem_l241_241238


namespace car_and_truck_arrival_time_simultaneous_l241_241390

theorem car_and_truck_arrival_time_simultaneous {t_car t_truck : ℕ} 
    (h1 : t_car = 8 * 60 + 16) -- Car leaves at 08:16
    (h2 : t_truck = 9 * 60) -- Truck leaves at 09:00
    (h3 : t_car_arrive = 10 * 60 + 56) -- Car arrives at 10:56
    (h4 : t_truck_arrive = 12 * 60 + 20) -- Truck arrives at 12:20
    (h5 : t_truck_exit = t_car_exit + 2) -- Truck leaves tunnel 2 minutes after car
    : (t_car_exit + t_car_tunnel_time = 10 * 60) ∧ (t_truck_exit + t_truck_tunnel_time = 10 * 60) :=
  sorry

end car_and_truck_arrival_time_simultaneous_l241_241390


namespace distinct_pairs_reciprocal_sum_l241_241803

theorem distinct_pairs_reciprocal_sum : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ (m n : ℕ), ((m, n) ∈ S) ↔ (m > 0 ∧ n > 0 ∧ (1/m + 1/n = 1/5))) ∧ S.card = 3 :=
sorry

end distinct_pairs_reciprocal_sum_l241_241803


namespace average_runs_next_10_matches_l241_241768

theorem average_runs_next_10_matches (avg_first_10 : ℕ) (avg_all_20 : ℕ) (n_matches : ℕ) (avg_next_10 : ℕ) :
  avg_first_10 = 40 ∧ avg_all_20 = 35 ∧ n_matches = 10 → avg_next_10 = 30 :=
by
  intros h
  sorry

end average_runs_next_10_matches_l241_241768


namespace triangle_max_third_side_l241_241517

theorem triangle_max_third_side (D E F : ℝ) (a b : ℝ) (h1 : a = 8) (h2 : b = 15) 
(h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1) 
: ∃ c : ℝ, c = 13 :=
by
  sorry

end triangle_max_third_side_l241_241517


namespace election_vote_percentage_l241_241115

theorem election_vote_percentage 
  (total_students : ℕ)
  (winner_percentage : ℝ)
  (loser_percentage : ℝ)
  (vote_difference : ℝ)
  (P : ℝ)
  (H1 : total_students = 2000)
  (H2 : winner_percentage = 0.55)
  (H3 : loser_percentage = 0.45)
  (H4 : vote_difference = 50)
  (H5 : 0.1 * P * (total_students / 100) = vote_difference) :
  P = 25 := 
sorry

end election_vote_percentage_l241_241115


namespace kelly_points_l241_241941

theorem kelly_points (K : ℕ) 
  (h1 : 12 + 2 * 12 + K + 2 * K + 12 / 2 = 69) : K = 9 := by
  sorry

end kelly_points_l241_241941


namespace volleyball_club_lineups_l241_241860
-- Import the required Lean library

-- Define the main problem
theorem volleyball_club_lineups :
  let total_players := 18
  let quadruplets := 4
  let starters := 6
  let eligible_lineups := Nat.choose 18 6 - Nat.choose 14 2 - Nat.choose 14 6
  eligible_lineups = 15470 :=
by
  sorry

end volleyball_club_lineups_l241_241860


namespace find_m_l241_241047

-- Defining vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b : ℝ × ℝ := (1, -1)

-- Proving that if b is perpendicular to (a + 2b), then m = 6
theorem find_m (m : ℝ) :
  let a_vec := a m
  let b_vec := b
  let sum_vec := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (b_vec.1 * sum_vec.1 + b_vec.2 * sum_vec.2 = 0) → m = 6 :=
by
  intros a_vec b_vec sum_vec perp_cond
  sorry

end find_m_l241_241047


namespace expected_area_convex_hull_correct_l241_241427

def point_placement (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

def convex_hull_area (points : Finset (ℕ × ℤ)) : ℚ := 
  -- Definition of the area calculation goes here. This is a placeholder.
  0  -- Placeholder for the actual calculation

noncomputable def expected_convex_hull_area : ℚ := 
  -- Calculation of the expected area, which is complex and requires integration of the probability.
  sorry  -- Placeholder for the actual expected value

theorem expected_area_convex_hull_correct : 
  expected_convex_hull_area = 1793 / 128 :=
sorry

end expected_area_convex_hull_correct_l241_241427


namespace solve_player_coins_l241_241215

def player_coins (n m k: ℕ) : Prop :=
  ∃ k, 
  (m = k * (n - 1) + 50) ∧ 
  (3 * m = 7 * n * k - 3 * k + 74) ∧ 
  (m = 69)

theorem solve_player_coins (n m k : ℕ) : player_coins n m k :=
by {
  sorry
}

end solve_player_coins_l241_241215


namespace problem_statement_l241_241639

open Real

theorem problem_statement (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * π)
  (h₁ : 2 * cos x ≤ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))
  ∧ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x)) ≤ sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := sorry

end problem_statement_l241_241639


namespace square_div_by_144_l241_241371

theorem square_div_by_144 (n : ℕ) (h1 : ∃ (k : ℕ), n = 12 * k) : ∃ (m : ℕ), n^2 = 144 * m :=
by
  sorry

end square_div_by_144_l241_241371


namespace pick_two_black_cards_l241_241859

-- Definition: conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13
def black_suits : ℕ := 2
def red_suits : ℕ := 2
def total_black_cards : ℕ := black_suits * cards_per_suit

-- Theorem: number of ways to pick two different black cards
theorem pick_two_black_cards :
  (total_black_cards * (total_black_cards - 1)) = 650 :=
by
  -- proof here
  sorry

end pick_two_black_cards_l241_241859


namespace greatest_whole_number_satisfying_inequality_l241_241366

theorem greatest_whole_number_satisfying_inequality :
  ∃ x : ℕ, (∀ y : ℕ, y < 1 → y ≤ x) ∧ 4 * x - 3 < 2 - x :=
sorry

end greatest_whole_number_satisfying_inequality_l241_241366


namespace rainfall_ratio_l241_241280

theorem rainfall_ratio (R1 R2 : ℕ) (hR2 : R2 = 24) (hTotal : R1 + R2 = 40) : 
  R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l241_241280


namespace min_distance_from_P_to_origin_l241_241349

noncomputable def distance_to_origin : ℝ := 8 / 5

theorem min_distance_from_P_to_origin
  (P : ℝ × ℝ)
  (hA : P.1^2 + P.2^2 = 1)
  (hB : (P.1 - 3)^2 + (P.2 + 4)^2 = 10)
  (h_tangent : PE = PD) :
  dist P (0, 0) = distance_to_origin := 
sorry

end min_distance_from_P_to_origin_l241_241349


namespace line_circle_intersect_a_le_0_l241_241214

theorem line_circle_intersect_a_le_0 :
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2 * x - 2 * y + 1 = 0) →
  a ≤ 0 :=
sorry

end line_circle_intersect_a_le_0_l241_241214


namespace sum_of_star_tips_l241_241453

/-- Given ten points that are evenly spaced on a circle and connected to form a 10-pointed star,
prove that the sum of the angle measurements of the ten tips of the star is 720 degrees. -/
theorem sum_of_star_tips (n : ℕ) (h : n = 10) :
  (10 * 72 = 720) :=
by
  sorry

end sum_of_star_tips_l241_241453


namespace ratio_of_larger_to_smaller_l241_241001

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l241_241001


namespace range_of_a_l241_241191

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^3 - a * x^2 - 4 * a * x + 4 * a^2 - 1 = 0 ∧ ∀ y : ℝ, 
  (y ≠ x → y^3 - a * y^2 - 4 * a * y + 4 * a^2 - 1 ≠ 0)) ↔ a < 3 / 4 := 
sorry

end range_of_a_l241_241191


namespace exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l241_241729

/-- There exists a way to completely tile a 5x6 board with dominos without leaving any gaps. -/
theorem exists_tiling_5x6_no_gaps :
  ∃ (tiling : List (Set (Fin 5 × Fin 6))), True := 
sorry

/-- It is not possible to tile a 5x6 board with dominos such that gaps are left. -/
theorem no_tiling_5x6_with_gaps :
  ¬ ∃ (tiling : List (Set (Fin 5 × Fin 6))), False := 
sorry

/-- It is impossible to tile a 6x6 board with dominos. -/
theorem no_tiling_6x6 :
  ¬ ∃ (tiling : List (Set (Fin 6 × Fin 6))), True := 
sorry

end exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l241_241729


namespace exam_total_students_l241_241522
-- Import the necessary Lean libraries

-- Define the problem conditions and the proof goal
theorem exam_total_students (T : ℕ) (h1 : 27 * T / 100 ≤ T) (h2 : 54 * T / 100 ≤ T) (h3 : 57 = 19 * T / 100) :
  T = 300 :=
  sorry  -- Proof is omitted here.

end exam_total_students_l241_241522


namespace sum_of_angles_in_figure_l241_241951

theorem sum_of_angles_in_figure : 
  let triangles := 3
  let angles_in_triangle := 180
  let square_angles := 4 * 90
  (triangles * angles_in_triangle + square_angles) = 900 := by
  sorry

end sum_of_angles_in_figure_l241_241951


namespace river_length_l241_241177

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end river_length_l241_241177


namespace square_sum_l241_241234

theorem square_sum (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = -2) : a^2 + b^2 = 68 := 
by 
  sorry

end square_sum_l241_241234


namespace infinite_nested_radicals_solution_l241_241719

theorem infinite_nested_radicals_solution :
  ∃ x : ℝ, 
    (∃ y z : ℝ, (y = (x * y)^(1/3) ∧ z = (x + z)^(1/3)) ∧ y = z) ∧ 
    0 < x ∧ x = (3 + Real.sqrt 5) / 2 := 
sorry

end infinite_nested_radicals_solution_l241_241719


namespace estimate_number_of_blue_cards_l241_241179

-- Define the given conditions:
def red_cards : ℕ := 8
def frequency_blue_card : ℚ := 0.6

-- Define the statement that needs to be proved:
theorem estimate_number_of_blue_cards (x : ℕ) 
  (h : (x : ℚ) / (x + red_cards) = frequency_blue_card) : 
  x = 12 :=
  sorry

end estimate_number_of_blue_cards_l241_241179


namespace number_of_games_in_season_l241_241253

-- Define the number of teams and divisions
def num_teams := 20
def num_divisions := 4
def teams_per_division := 5

-- Define the games played within and between divisions
def intra_division_games_per_team := 12  -- 4 teams * 3 games each
def inter_division_games_per_team := 15  -- (20 - 5) teams * 1 game each

-- Define the total number of games played by each team
def total_games_per_team := intra_division_games_per_team + inter_division_games_per_team

-- Define the total number of games played (double-counting needs to be halved)
def total_games (num_teams : ℕ) (total_games_per_team : ℕ) : ℕ :=
  (num_teams * total_games_per_team) / 2

-- The theorem to be proven
theorem number_of_games_in_season :
  total_games num_teams total_games_per_team = 270 :=
by
  sorry

end number_of_games_in_season_l241_241253


namespace inequality_solution_l241_241829

theorem inequality_solution (a x : ℝ) : 
  (x^2 - (a + 1) * x + a) ≤ 0 ↔ 
  (a > 1 → (1 ≤ x ∧ x ≤ a)) ∧ 
  (a = 1 → x = 1) ∧ 
  (a < 1 → (a ≤ x ∧ x ≤ 1)) :=
by 
  sorry

end inequality_solution_l241_241829


namespace right_triangle_eqn_roots_indeterminate_l241_241984

theorem right_triangle_eqn_roots_indeterminate 
  (a b c : ℝ) (h : a^2 + c^2 = b^2) : 
  ¬(∃ Δ, Δ = 4 - 4 * c^2 ∧ (Δ > 0 ∨ Δ = 0 ∨ Δ < 0)) →
  (¬∃ x, a * (x^2 - 1) - 2 * x + b * (x^2 + 1) = 0 ∨
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ a * (x₁^2 - 1) - 2 * x₁ + b * (x₁^2 + 1) = 0 ∧ a * (x₂^2 - 1) - 2 * x₂ + b * (x₂^2 + 1) = 0) :=
by
  sorry

end right_triangle_eqn_roots_indeterminate_l241_241984


namespace maximize_profit_l241_241577

noncomputable def profit_function (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 40 then
  -2 * x^2 + 120 * x - 300
else if 40 < x ∧ x ≤ 100 then
  -x - 3600 / x + 1800
else
  0

theorem maximize_profit :
  profit_function 60 = 1680 ∧
  ∀ x, 0 < x ∧ x ≤ 100 → profit_function x ≤ 1680 := 
sorry

end maximize_profit_l241_241577


namespace xiao_ying_should_pay_l241_241500

variable (x y z : ℝ)

def equation1 := 3 * x + 7 * y + z = 14
def equation2 := 4 * x + 10 * y + z = 16
def equation3 := 2 * (x + y + z) = 20

theorem xiao_ying_should_pay :
  equation1 x y z →
  equation2 x y z →
  equation3 x y z :=
by
  intros h1 h2
  sorry

end xiao_ying_should_pay_l241_241500


namespace trip_time_difference_l241_241802

theorem trip_time_difference (speed distance1 distance2 : ℕ) (h1 : speed > 0) (h2 : distance2 > distance1) 
  (h3 : speed = 60) (h4 : distance1 = 540) (h5 : distance2 = 570) : 
  (distance2 - distance1) / speed * 60 = 30 := 
by
  sorry

end trip_time_difference_l241_241802


namespace system_solution_l241_241774

theorem system_solution :
  ∃ x y : ℝ, (3 * x + y = 11 ∧ x - y = 1) ∧ (x = 3 ∧ y = 2) := 
by
  sorry

end system_solution_l241_241774


namespace simplify_fraction_l241_241887

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 :=
by
  sorry

end simplify_fraction_l241_241887


namespace colin_avg_time_l241_241265

def totalTime (a b c d : ℕ) : ℕ := a + b + c + d

def averageTime (total_time miles : ℕ) : ℕ := total_time / miles

theorem colin_avg_time :
  let first_mile := 6
  let second_mile := 5
  let third_mile := 5
  let fourth_mile := 4
  let total_time := totalTime first_mile second_mile third_mile fourth_mile
  4 > 0 -> averageTime total_time 4 = 5 :=
by
  intros
  -- proof goes here
  sorry

end colin_avg_time_l241_241265


namespace greatest_sum_l241_241034

theorem greatest_sum {x y : ℤ} (h₁ : x^2 + y^2 = 49) : x + y ≤ 9 :=
sorry

end greatest_sum_l241_241034


namespace medium_kite_area_l241_241059

-- Define the points and the spacing on the grid
structure Point :=
mk :: (x : ℕ) (y : ℕ)

def medium_kite_vertices : List Point :=
[Point.mk 0 4, Point.mk 4 10, Point.mk 12 4, Point.mk 4 0]

def grid_spacing : ℕ := 2

-- Function to calculate the area of a kite given list of vertices and spacing
noncomputable def area_medium_kite (vertices : List Point) (spacing : ℕ) : ℕ := sorry

-- The theorem to be proved
theorem medium_kite_area (vertices : List Point) (spacing : ℕ) :
  vertices = medium_kite_vertices ∧ spacing = grid_spacing → area_medium_kite vertices spacing = 288 := 
by {
  -- The detailed proof would go here
  sorry
}

end medium_kite_area_l241_241059


namespace find_f_5pi_div_3_l241_241961

variable (f : ℝ → ℝ)

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem find_f_5pi_div_3
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_function f π)
  (h_def : ∀ x, 0 ≤ x → x ≤ π/2 → f x = Real.sin x) :
  f (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end find_f_5pi_div_3_l241_241961


namespace find_original_manufacturing_cost_l241_241782

noncomputable def originalManufacturingCost (P : ℝ) : ℝ := 0.70 * P

theorem find_original_manufacturing_cost (P : ℝ) (currentCost : ℝ) 
  (h1 : currentCost = 50) 
  (h2 : currentCost = P - 0.50 * P) : originalManufacturingCost P = 70 :=
by
  -- The actual proof steps would go here, but we'll add sorry for now
  sorry

end find_original_manufacturing_cost_l241_241782


namespace find_b_for_continuity_at_2_l241_241848

noncomputable def f (x : ℝ) (b : ℝ) :=
if x ≤ 2 then 3 * x^2 + 1 else b * x - 6

theorem find_b_for_continuity_at_2
  (b : ℝ) 
  (h_cont : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) :
  b = 19 / 2 := by sorry

end find_b_for_continuity_at_2_l241_241848


namespace function_monotonically_increasing_iff_range_of_a_l241_241641

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem function_monotonically_increasing_iff_range_of_a (a : ℝ) :
  (∀ x, (deriv (f a) x) ≥ 0) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
by
  sorry

end function_monotonically_increasing_iff_range_of_a_l241_241641


namespace find_b_l241_241457

open Real

variables {A B C a b c : ℝ}

theorem find_b 
  (hA : A = π / 4) 
  (h1 : 2 * b * sin B - c * sin C = 2 * a * sin A) 
  (h_area : 1 / 2 * b * c * sin A = 3) : 
  b = 3 := 
sorry

end find_b_l241_241457


namespace warriors_can_defeat_dragon_l241_241576

theorem warriors_can_defeat_dragon (n : ℕ) (h : n = 20^20) :
  (∀ n, n % 2 = 0 ∨ n % 3 = 0) → (∃ m, m = 0) := 
sorry

end warriors_can_defeat_dragon_l241_241576


namespace john_annual_profit_l241_241569

namespace JohnProfit

def number_of_people_subletting := 3
def rent_per_person_per_month := 400
def john_rent_per_month := 900
def months_in_year := 12

theorem john_annual_profit 
  (h1 : number_of_people_subletting = 3)
  (h2 : rent_per_person_per_month = 400)
  (h3 : john_rent_per_month = 900)
  (h4 : months_in_year = 12) : 
  (number_of_people_subletting * rent_per_person_per_month - john_rent_per_month) * months_in_year = 3600 :=
by
  sorry

end JohnProfit

end john_annual_profit_l241_241569


namespace find_x_l241_241460

theorem find_x (x : ℝ) (h : 15 * x + 16 * x + 19 * x + 11 = 161) : x = 3 :=
sorry

end find_x_l241_241460


namespace max_three_digit_sum_l241_241558

theorem max_three_digit_sum :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ 101 * A + 11 * B + 11 * C = 986 := 
sorry

end max_three_digit_sum_l241_241558


namespace perfect_squares_digit_4_5_6_l241_241936

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end perfect_squares_digit_4_5_6_l241_241936


namespace resultant_force_correct_l241_241203

-- Define the conditions
def P1 : ℝ := 80
def P2 : ℝ := 130
def distance : ℝ := 12.035
def theta1 : ℝ := 125
def theta2 : ℝ := 135.1939

-- Calculate the correct answer
def result_magnitude : ℝ := 209.299
def result_direction : ℝ := 131.35

-- The goal statement to be proved
theorem resultant_force_correct :
  ∃ (R : ℝ) (theta_R : ℝ), 
    R = result_magnitude ∧ theta_R = result_direction := 
sorry

end resultant_force_correct_l241_241203


namespace least_positive_integer_x_l241_241555

theorem least_positive_integer_x :
  ∃ x : ℕ, (x > 0) ∧ (∃ k : ℕ, (2 * x + 51) = k * 59) ∧ x = 4 :=
by
  -- Lean statement
  sorry

end least_positive_integer_x_l241_241555


namespace tan_beta_half_l241_241277

theorem tan_beta_half (α β : ℝ)
    (h1 : Real.tan α = 1 / 3)
    (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
    Real.tan β = 1 / 2 := 
sorry

end tan_beta_half_l241_241277


namespace Ram_has_amount_l241_241909

theorem Ram_has_amount (R G K : ℕ)
    (h1 : R = 7 * G / 17)
    (h2 : G = 7 * K / 17)
    (h3 : K = 3757) : R = 637 := by
  sorry

end Ram_has_amount_l241_241909


namespace second_set_number_l241_241711

theorem second_set_number (x : ℕ) (sum1 : ℕ) (avg2 : ℕ) (total_avg : ℕ)
  (h1 : sum1 = 98) (h2 : avg2 = 11) (h3 : total_avg = 8)
  (h4 : 16 + x ≠ 0) :
  (98 + avg2 * x = total_avg * (x + 16)) → x = 10 :=
by
  sorry

end second_set_number_l241_241711


namespace prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l241_241924

noncomputable def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 1)^2 = 4

noncomputable def cartesian_eq_C2 (x y : ℝ) : Prop :=
  (4 * x - y - 1 = 0)

noncomputable def min_distance_C1_C2 : ℝ :=
  (10 * Real.sqrt 17 / 17) - 2

theorem prove_cartesian_eq_C1 (x y t : ℝ) (h : x = -2 + 2 * Real.cos t ∧ y = 1 + 2 * Real.sin t) :
  cartesian_eq_C1 x y :=
sorry

theorem prove_cartesian_eq_C2 (ρ θ : ℝ) (h : 4 * ρ * Real.cos θ - ρ * Real.sin θ - 1 = 0) :
  cartesian_eq_C2 (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

theorem prove_min_distance_C1_C2 (h1 : ∀ x y, cartesian_eq_C1 x y) (h2 : ∀ x y, cartesian_eq_C2 x y) :
  ∀ P Q : ℝ × ℝ, (cartesian_eq_C1 P.1 P.2) → (cartesian_eq_C2 Q.1 Q.2) →
  (min_distance_C1_C2 = (Real.sqrt (4^2 + (-1)^2) / Real.sqrt 17) - 2) :=
sorry

end prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l241_241924


namespace angle_SR_XY_is_70_l241_241333

-- Define the problem conditions
variables (X Y Z V H S R : Type) 
variables (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ)

-- Set the conditions
def triangleXYZ (X Y Z V H S R : Type) (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ) : Prop :=
  angleX = 40 ∧ angleY = 70 ∧ XY = 12 ∧ XV = 2 ∧ YH = 2 ∧
  ∃ S R, S = (XY / 2) ∧ R = ((XV + YH) / 2)

-- Construct the theorem to be proven
theorem angle_SR_XY_is_70 {X Y Z V H S R : Type} 
  {angleX angleY angleZ angleSRXY : ℝ} 
  {XY XV YH : ℝ} : 
  triangleXYZ X Y Z V H S R angleX angleY angleZ angleSRXY XY XV YH →
  angleSRXY = 70 :=
by
  -- Placeholder proof steps
  sorry

end angle_SR_XY_is_70_l241_241333


namespace problem_solution_l241_241121

-- Definitions
def has_property_P (A : List ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ A.length →
    (A.get! (j - 1) + A.get! (i - 1) ∈ A ∨ A.get! (j - 1) - A.get! (i - 1) ∈ A)

def sequence_01234 := [0, 2, 4, 6]

-- Propositions
def proposition_1 : Prop := has_property_P sequence_01234

def proposition_2 (A : List ℕ) : Prop := 
  has_property_P A → (A.headI = 0)

def proposition_3 (A : List ℕ) : Prop :=
  has_property_P A → A.headI ≠ 0 →
  ∀ k, 1 ≤ k ∧ k < A.length → A.get! (A.length - 1) - A.get! (A.length - 1 - k) = A.get! k

def proposition_4 (A : List ℕ) : Prop :=
  has_property_P A → A.length = 3 →
  A.get! 2 = A.get! 0 + A.get! 1

-- Main statement
theorem problem_solution : 
  (proposition_1) ∧
  (∃ A, ¬ (proposition_2 A)) ∧
  (∃ A, proposition_3 A) ∧
  (∃ A, proposition_4 A) →
  3 = 3 := 
by sorry

end problem_solution_l241_241121


namespace ratio_naomi_to_katherine_l241_241815

theorem ratio_naomi_to_katherine 
  (katherine_time : ℕ) 
  (naomi_total_time : ℕ) 
  (websites_naomi : ℕ)
  (hk : katherine_time = 20)
  (hn : naomi_total_time = 750)
  (wn : websites_naomi = 30) : 
  naomi_total_time / websites_naomi / katherine_time = 5 / 4 := 
by sorry

end ratio_naomi_to_katherine_l241_241815


namespace least_number_of_colors_needed_l241_241099

-- Define the tessellation of hexagons
structure HexagonalTessellation :=
(adjacent : (ℕ × ℕ) → (ℕ × ℕ) → Prop)
(symm : ∀ {a b : ℕ × ℕ}, adjacent a b → adjacent b a)
(irrefl : ∀ a : ℕ × ℕ, ¬ adjacent a a)
(hex_property : ∀ a : ℕ × ℕ, ∃ b1 b2 b3 b4 b5 b6,
  adjacent a b1 ∧ adjacent a b2 ∧ adjacent a b3 ∧ adjacent a b4 ∧ adjacent a b5 ∧ adjacent a b6)

-- Define a coloring function for a HexagonalTessellation
def coloring (T : HexagonalTessellation) (colors : ℕ) :=
(∀ (a b : ℕ × ℕ), T.adjacent a b → a ≠ b → colors ≥ 1 → colors ≤ 3)

-- Statement to prove the minimum number of colors required
theorem least_number_of_colors_needed (T : HexagonalTessellation) :
  ∃ colors, coloring T colors ∧ colors = 3 :=
sorry

end least_number_of_colors_needed_l241_241099


namespace ellipse_x_intercepts_l241_241544

noncomputable def distances_sum (x : ℝ) (y : ℝ) (f₁ f₂ : ℝ × ℝ) :=
  (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2)) + (Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2))

def is_on_ellipse (x y : ℝ) : Prop := 
  distances_sum x y (0, 3) (4, 0) = 7

theorem ellipse_x_intercepts 
  (h₀ : is_on_ellipse 0 0) 
  (hx_intercept : ∀ x : ℝ, is_on_ellipse x 0 → x = 0 ∨ x = 20 / 7) :
  ∀ x : ℝ, is_on_ellipse x 0 ↔ x = 0 ∨ x = 20 / 7 :=
by
  sorry

end ellipse_x_intercepts_l241_241544


namespace complement_U_A_l241_241230

open Set

def U : Set ℝ := {x | -3 < x ∧ x < 3}
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem complement_U_A : 
  (U \ A) = {x | -3 < x ∧ x ≤ -2} ∪ {x | 1 < x ∧ x < 3} :=
by
  sorry

end complement_U_A_l241_241230


namespace expression_value_as_fraction_l241_241877

theorem expression_value_as_fraction :
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 :=
by
  sorry

end expression_value_as_fraction_l241_241877


namespace correct_choice_l241_241196

def PropA : Prop := ∀ x : ℝ, x^2 + 3 < 0
def PropB : Prop := ∀ x : ℕ, x^2 ≥ 1
def PropC : Prop := ∃ x : ℤ, x^5 < 1
def PropD : Prop := ∃ x : ℚ, x^2 = 3

theorem correct_choice : ¬PropA ∧ ¬PropB ∧ PropC ∧ ¬PropD := by
  sorry

end correct_choice_l241_241196


namespace midpoint_trajectory_of_chord_l241_241933

theorem midpoint_trajectory_of_chord {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 / 3 + A.2^2 = 1) ∧ 
    (B.1^2 / 3 + B.2^2 = 1) ∧ 
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (x, y) ∧ 
    ∃ t : ℝ, ((-1, 0) = ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2))) -> 
  x^2 + x + 3 * y^2 = 0 :=
by sorry

end midpoint_trajectory_of_chord_l241_241933


namespace problem_l241_241000

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem problem 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y = 30) 
  (h5 : x * z = 60) 
  (h6 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := 
  sorry

end problem_l241_241000


namespace difference_of_squares_l241_241192

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) : 
  x^2 - y^2 = 200 := 
sorry

end difference_of_squares_l241_241192


namespace compare_abc_l241_241376

noncomputable def a : ℝ := (0.6)^(2/5)
noncomputable def b : ℝ := (0.4)^(2/5)
noncomputable def c : ℝ := (0.4)^(3/5)

theorem compare_abc : a > b ∧ b > c := 
by
  sorry

end compare_abc_l241_241376


namespace findPerpendicularLine_l241_241664

-- Defining the condition: the line passes through point (-1, 2)
def pointOnLine (x y : ℝ) (a b : ℝ) (c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Defining the condition: the line is perpendicular to 2x - 3y + 4 = 0
def isPerpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

-- The original line equation: 2x - 3y + 4 = 0
def originalLine (x y : ℝ) : Prop :=
  2 * x - 3 * y + 4 = 0

-- The target equation of the line: 3x + 2y - 1 = 0
def targetLine (x y : ℝ) : Prop :=
  3 * x + 2 * y - 1 = 0

theorem findPerpendicularLine :
  (pointOnLine (-1) 2 3 2 (-1)) ∧
  (isPerpendicular 3 2 2 (-3)) →
  (∀ x y, targetLine x y ↔ 3 * x + 2 * y - 1 = 0) :=
by
  sorry

end findPerpendicularLine_l241_241664


namespace range_m_condition_l241_241436

theorem range_m_condition {x y m : ℝ} (h1 : x^2 + (y - 1)^2 = 1) (h2 : x + y + m ≥ 0) : -1 < m :=
by
  sorry

end range_m_condition_l241_241436


namespace color_pairings_correct_l241_241325

noncomputable def num_color_pairings (bowls : ℕ) (glasses : ℕ) : ℕ :=
  bowls * glasses

theorem color_pairings_correct : 
  num_color_pairings 4 5 = 20 :=
by 
  -- proof omitted
  sorry

end color_pairings_correct_l241_241325


namespace range_of_a_l241_241438

-- Define conditions
def setA : Set ℝ := {x | x^2 - x ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- Problem statement in Lean 4
theorem range_of_a (a : ℝ) (h : setA ⊆ setB a) : a ≤ -2 :=
by
  sorry

end range_of_a_l241_241438


namespace pieces_per_box_l241_241777

theorem pieces_per_box (boxes : ℕ) (total_pieces : ℕ) (h_boxes : boxes = 7) (h_total : total_pieces = 21) : 
  total_pieces / boxes = 3 :=
by
  sorry

end pieces_per_box_l241_241777


namespace gcd_280_2155_l241_241800

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := 
sorry

end gcd_280_2155_l241_241800


namespace power_mod_1000_l241_241817

theorem power_mod_1000 (N : ℤ) (h : Int.gcd N 10 = 1) : (N ^ 101 ≡ N [ZMOD 1000]) :=
  sorry

end power_mod_1000_l241_241817


namespace combined_area_difference_l241_241131

theorem combined_area_difference :
  let area_11x11 := 2 * (11 * 11)
  let area_5_5x11 := 2 * (5.5 * 11)
  area_11x11 - area_5_5x11 = 121 :=
by
  sorry

end combined_area_difference_l241_241131


namespace distinct_bead_arrangements_on_bracelet_l241_241945

open Nat

-- Definition of factorial
def fact : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * fact n

-- Theorem stating the number of distinct arrangements of 7 beads on a bracelet
theorem distinct_bead_arrangements_on_bracelet : 
  fact 7 / 14 = 360 := 
by 
  sorry

end distinct_bead_arrangements_on_bracelet_l241_241945


namespace equal_share_of_marbles_l241_241973

-- Define the number of marbles bought by each friend based on the conditions
def wolfgang_marbles : ℕ := 16
def ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
def michael_marbles : ℕ := 2 * (wolfgang_marbles + ludo_marbles) / 3
def shania_marbles : ℕ := 2 * ludo_marbles
def gabriel_marbles : ℕ := (wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles) - 1
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles + gabriel_marbles
def marbles_per_friend : ℕ := total_marbles / 5

-- Mathematical equivalent proof problem
theorem equal_share_of_marbles : marbles_per_friend = 39 := by
  sorry

end equal_share_of_marbles_l241_241973


namespace tiling_problem_l241_241948

theorem tiling_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ n = 4 * k) 
  ↔ (∃ (L_tile T_tile : ℕ), n * n = 3 * L_tile + 4 * T_tile) :=
by
  sorry

end tiling_problem_l241_241948


namespace potassium_salt_average_molar_mass_l241_241385

noncomputable def average_molar_mass (total_weight : ℕ) (num_moles : ℕ) : ℕ :=
  total_weight / num_moles

theorem potassium_salt_average_molar_mass :
  let total_weight := 672
  let num_moles := 4
  average_molar_mass total_weight num_moles = 168 := by
    sorry

end potassium_salt_average_molar_mass_l241_241385


namespace smallest_odd_n_3_product_gt_5000_l241_241827

theorem smallest_odd_n_3_product_gt_5000 :
  ∃ n : ℕ, (∃ k : ℤ, n = 2 * k + 1 ∧ n > 0) ∧ (3 ^ ((n + 1)^2 / 8)) > 5000 ∧ n = 8 :=
by
  sorry

end smallest_odd_n_3_product_gt_5000_l241_241827


namespace solve_trig_problem_l241_241204

noncomputable def trig_problem (α : ℝ) : Prop :=
  α ∈ (Set.Ioo 0 (Real.pi / 2)) ∪ Set.Ioo (Real.pi / 2) Real.pi ∧
  ∃ r : ℝ, r ≠ 0 ∧ Real.sin α * r = Real.sin (2 * α) ∧ Real.sin (2 * α) * r = Real.sin (4 * α)

theorem solve_trig_problem (α : ℝ) (h : trig_problem α) : α = 2 * Real.pi / 3 :=
by
  sorry

end solve_trig_problem_l241_241204


namespace min_value_reciprocal_sum_l241_241353

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a = 1 ∧ b = 1) → (1 / a + 1 / b = 2) := by
  intros h
  sorry

end min_value_reciprocal_sum_l241_241353


namespace tv_sale_increase_l241_241336

theorem tv_sale_increase (P Q : ℝ) :
  let new_price := 0.9 * P
  let original_sale_value := P * Q
  let increased_percentage := 1.665
  ∃ x : ℝ, (new_price * (1 + x / 100) * Q = increased_percentage * original_sale_value) → x = 85 :=
by
  sorry

end tv_sale_increase_l241_241336


namespace simplify_and_evaluate_l241_241942

variable (a : ℕ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a^2 / (1 - 2 / a) = 7 / 5 :=
by
  -- Assign the condition
  let a := 5
  sorry -- skip the proof

end simplify_and_evaluate_l241_241942


namespace distribute_pencils_l241_241587

def number_of_ways_to_distribute_pencils (pencils friends : ℕ) : ℕ :=
  Nat.choose (pencils - friends + friends - 1) (friends - 1)

theorem distribute_pencils :
  number_of_ways_to_distribute_pencils 4 4 = 35 :=
by
  sorry

end distribute_pencils_l241_241587


namespace lemon_cookies_amount_l241_241117

def cookies_problem 
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) : Prop :=
  jenny_pb_cookies = 40 ∧
  jenny_cc_cookies = 50 ∧
  marcus_pb_cookies = 30 ∧
  total_pb_cookies = jenny_pb_cookies + marcus_pb_cookies ∧
  total_pb_cookies = 70 ∧
  total_non_pb_cookies = jenny_cc_cookies + marcus_lemon_cookies ∧
  total_pb_cookies = total_non_pb_cookies

theorem lemon_cookies_amount
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) :
  cookies_problem jenny_pb_cookies jenny_cc_cookies marcus_pb_cookies marcus_lemon_cookies total_pb_cookies total_non_pb_cookies →
  marcus_lemon_cookies = 20 :=
by
  sorry

end lemon_cookies_amount_l241_241117


namespace truncated_pyramid_volume_l241_241194

theorem truncated_pyramid_volume :
  let unit_cube_vol := 1
  let tetrahedron_base_area := 1 / 2
  let tetrahedron_height := 1 / 2
  let tetrahedron_vol := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let two_tetrahedra_vol := 2 * tetrahedron_vol
  let truncated_pyramid_vol := unit_cube_vol - two_tetrahedra_vol
  truncated_pyramid_vol = 5 / 6 :=
by
  sorry

end truncated_pyramid_volume_l241_241194


namespace tournament_games_count_l241_241520

-- Defining the problem conditions
def num_players : Nat := 12
def plays_twice : Bool := true

-- Theorem statement
theorem tournament_games_count (n : Nat) (plays_twice : Bool) (h : n = num_players ∧ plays_twice = true) :
  (n * (n - 1) * 2) = 264 := by
  sorry

end tournament_games_count_l241_241520


namespace desired_antifreeze_pct_in_colder_climates_l241_241789

-- Definitions for initial conditions
def initial_antifreeze_pct : ℝ := 0.10
def radiator_volume : ℝ := 4
def drained_volume : ℝ := 2.2857
def replacement_antifreeze_pct : ℝ := 0.80

-- Proof goal: Desired percentage of antifreeze in the mixture is 50%
theorem desired_antifreeze_pct_in_colder_climates :
  (drained_volume * replacement_antifreeze_pct + (radiator_volume - drained_volume) * initial_antifreeze_pct) / radiator_volume = 0.50 :=
by
  sorry

end desired_antifreeze_pct_in_colder_climates_l241_241789


namespace arithmetic_common_difference_l241_241369

-- Define the conditions of the arithmetic sequence
def a (n : ℕ) := 0 -- This is a placeholder definition since we only care about a_5 and a_12
def a5 : ℝ := 10
def a12 : ℝ := 31

-- State the proof problem
theorem arithmetic_common_difference :
  ∃ d : ℝ, a5 + 7 * d = a12 :=
by
  use 3
  simp [a5, a12]
  sorry

end arithmetic_common_difference_l241_241369


namespace sum_of_sines_leq_3_sqrt3_over_2_l241_241302

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sum_of_sines_leq_3_sqrt3_over_2_l241_241302


namespace seating_arrangement_ways_l241_241704

-- Define the problem conditions in Lean 4
def number_of_ways_to_seat (total_chairs : ℕ) (total_people : ℕ) := 
  Nat.factorial total_chairs / Nat.factorial (total_chairs - total_people)

-- Define the specific theorem to be proved
theorem seating_arrangement_ways : number_of_ways_to_seat 8 5 = 6720 :=
by
  sorry

end seating_arrangement_ways_l241_241704


namespace bug_total_distance_l241_241649

def total_distance_bug (start : ℤ) (pos1 : ℤ) (pos2 : ℤ) (pos3 : ℤ) : ℤ :=
  abs (pos1 - start) + abs (pos2 - pos1) + abs (pos3 - pos2)

theorem bug_total_distance :
  total_distance_bug 3 (-4) 6 2 = 21 :=
by
  -- We insert a sorry here to indicate the proof is skipped.
  sorry

end bug_total_distance_l241_241649


namespace three_distinct_divisors_l241_241330

theorem three_distinct_divisors (M : ℕ) : (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ M ∧ b ∣ M ∧ c ∣ M ∧ (∀ d, d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬ d ∣ M)) ↔ (∃ p : ℕ, Prime p ∧ M = p^2) := 
by sorry

end three_distinct_divisors_l241_241330


namespace intersection_complement_l241_241529

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x : ℝ | x > 0 }

-- Define the complement of B
def complement_B : Set ℝ := { x : ℝ | x ≤ 0 }

-- The theorem we need to prove
theorem intersection_complement :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 0 } := 
by
  sorry

end intersection_complement_l241_241529


namespace range_of_a_l241_241070

theorem range_of_a:
  (∃ x : ℝ, 1 ≤ x ∧ |x - a| + x - 4 ≤ 0) → (-2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l241_241070


namespace lunch_break_duration_l241_241696

def rate_sandra : ℝ := 0 -- Sandra's painting rate in houses per hour
def rate_helpers : ℝ := 0 -- Combined rate of the three helpers in houses per hour
def lunch_break : ℝ := 0 -- Lunch break duration in hours

axiom monday_condition : (8 - lunch_break) * (rate_sandra + rate_helpers) = 0.6
axiom tuesday_condition : (6 - lunch_break) * rate_helpers = 0.3
axiom wednesday_condition : (2 - lunch_break) * rate_sandra = 0.1

theorem lunch_break_duration : lunch_break = 0.5 :=
by {
  sorry
}

end lunch_break_duration_l241_241696


namespace sum_of_digits_of_7_pow_1974_l241_241375

-- Define the number \(7^{1974}\)
def num := 7^1974

-- Function to extract the last two digits
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Function to compute the sum of the tens and units digits
def sum_tens_units (n : ℕ) : ℕ :=
  let last_two := last_two_digits n
  (last_two / 10) + (last_two % 10)

theorem sum_of_digits_of_7_pow_1974 : sum_tens_units num = 9 := by
  sorry

end sum_of_digits_of_7_pow_1974_l241_241375


namespace students_play_both_football_and_tennis_l241_241022

theorem students_play_both_football_and_tennis 
  (T : ℕ) (F : ℕ) (L : ℕ) (N : ℕ) (B : ℕ)
  (hT : T = 38) (hF : F = 26) (hL : L = 20) (hN : N = 9) :
  B = F + L - (T - N) → B = 17 :=
by 
  intros h
  rw [hT, hF, hL, hN] at h
  exact h

end students_play_both_football_and_tennis_l241_241022


namespace quadratic_has_solutions_l241_241962

theorem quadratic_has_solutions :
  (1 + Real.sqrt 2)^2 - 2 * (1 + Real.sqrt 2) - 1 = 0 ∧ 
  (1 - Real.sqrt 2)^2 - 2 * (1 - Real.sqrt 2) - 1 = 0 :=
by
  sorry

end quadratic_has_solutions_l241_241962


namespace tub_emptying_time_l241_241537

variables (x C D T : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)

theorem tub_emptying_time (h1 : 4 * (D - x) = (5 / 7) * C) :
  T = 8 / (5 + (28 * x) / C) :=
by sorry

end tub_emptying_time_l241_241537


namespace part_I_part_II_l241_241361

-- Part I
theorem part_I (x : ℝ) : (|x + 1| + |x - 4| ≤ 2 * |x - 4|) ↔ (x < 1.5) :=
sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x : ℝ, |x + a| + |x - 4| ≥ 3) → (a ≤ -7 ∨ a ≥ -1) :=
sorry

end part_I_part_II_l241_241361


namespace tim_total_payment_correct_l241_241964

-- Define the conditions stated in the problem
def doc_visit_cost : ℝ := 300
def insurance_coverage_percent : ℝ := 0.75
def cat_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60

-- Define the amounts covered by insurance 
def insurance_coverage_amount : ℝ := doc_visit_cost * insurance_coverage_percent
def tim_payment_for_doc_visit : ℝ := doc_visit_cost - insurance_coverage_amount
def tim_payment_for_cat_visit : ℝ := cat_visit_cost - pet_insurance_coverage

-- Define the total payment Tim needs to make
def tim_total_payment : ℝ := tim_payment_for_doc_visit + tim_payment_for_cat_visit

-- State the main theorem
theorem tim_total_payment_correct : tim_total_payment = 135 := by
  sorry

end tim_total_payment_correct_l241_241964


namespace profit_percentage_calculation_l241_241486

noncomputable def profit_percentage (SP CP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem profit_percentage_calculation (SP : ℝ) (h : CP = 0.92 * SP) : |profit_percentage SP (0.92 * SP) - 8.70| < 0.01 :=
by
  sorry

end profit_percentage_calculation_l241_241486


namespace watermelon_percentage_l241_241589

theorem watermelon_percentage (total_drink : ℕ)
  (orange_percentage : ℕ)
  (grape_juice : ℕ)
  (watermelon_amount : ℕ)
  (W : ℕ) :
  total_drink = 300 →
  orange_percentage = 25 →
  grape_juice = 105 →
  watermelon_amount = total_drink - (orange_percentage * total_drink) / 100 - grape_juice →
  W = (watermelon_amount * 100) / total_drink →
  W = 40 :=
sorry

end watermelon_percentage_l241_241589


namespace census_suitable_survey_l241_241765

theorem census_suitable_survey (A B C D : Prop) : 
  D := 
sorry

end census_suitable_survey_l241_241765


namespace bug_total_distance_l241_241491

theorem bug_total_distance 
  (p₀ p₁ p₂ p₃ : ℤ) 
  (h₀ : p₀ = 0) 
  (h₁ : p₁ = 4) 
  (h₂ : p₂ = -3) 
  (h₃ : p₃ = 7) : 
  |p₁ - p₀| + |p₂ - p₁| + |p₃ - p₂| = 21 :=
by 
  sorry

end bug_total_distance_l241_241491


namespace remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l241_241020

theorem remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14 
  (a b c d e f g h : ℤ) 
  (h1 : a = 11085)
  (h2 : b = 11087)
  (h3 : c = 11089)
  (h4 : d = 11091)
  (h5 : e = 11093)
  (h6 : f = 11095)
  (h7 : g = 11097)
  (h8 : h = 11099) :
  (2 * (a + b + c + d + e + f + g + h)) % 14 = 2 := 
by
  sorry

end remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l241_241020


namespace completely_factored_form_l241_241071

theorem completely_factored_form (x : ℤ) :
  (12 * x ^ 3 + 95 * x - 6) - (-3 * x ^ 3 + 5 * x - 6) = 15 * x * (x ^ 2 + 6) :=
by
  sorry

end completely_factored_form_l241_241071


namespace clara_cookies_l241_241004

theorem clara_cookies (n : ℕ) :
  (15 * n - 1) % 11 = 0 → n = 3 := 
sorry

end clara_cookies_l241_241004


namespace students_selected_juice_l241_241068

def fraction_of_students_choosing_juice (students_selected_juice_ratio students_selected_soda_ratio : ℚ) : ℚ :=
  students_selected_juice_ratio / students_selected_soda_ratio

def num_students_selecting (students_selected_soda : ℕ) (fraction_juice : ℚ) : ℚ :=
  fraction_juice * students_selected_soda

theorem students_selected_juice (students_selected_soda : ℕ) : students_selected_soda = 120 ∧
    (fraction_of_students_choosing_juice 0.15 0.75) = 1/5 →
    num_students_selecting students_selected_soda (fraction_of_students_choosing_juice 0.15 0.75) = 24 :=
by
  intros h
  sorry

end students_selected_juice_l241_241068


namespace lobster_distribution_l241_241590

theorem lobster_distribution :
  let HarborA := 50
  let HarborB := 70.5
  let HarborC := (2 / 3) * HarborB
  let HarborD := HarborA - 0.15 * HarborA
  let Sum := HarborA + HarborB + HarborC + HarborD
  let HooperBay := 3 * Sum
  let Total := HooperBay + Sum
  Total = 840 := by
  sorry

end lobster_distribution_l241_241590


namespace determine_weights_l241_241891

-- Definitions
variable {W : Type} [AddCommGroup W] [OrderedAddCommMonoid W]
variable (w : Fin 20 → W) -- List of weights for 20 people
variable (s : W) -- Total sum of weights
variable (lower upper : W) -- Lower and upper weight limits

-- Conditions
def weight_constraints : Prop :=
  (∀ i, lower ≤ w i ∧ w i ≤ upper) ∧ (Finset.univ.sum w = s)

-- Problem statement
theorem determine_weights (w : Fin 20 → ℝ) :
  weight_constraints w 60 90 3040 →
  ∃ w : Fin 20 → ℝ, weight_constraints w 60 90 3040 := by
  sorry

end determine_weights_l241_241891


namespace hexagon_largest_angle_l241_241512

-- Definitions for conditions
def hexagon_interior_angle_sum : ℝ := 720  -- Sum of all interior angles of hexagon

def angle_A : ℝ := 100
def angle_B : ℝ := 120

-- Define x for angles C and D
variables (x : ℝ)
def angle_C : ℝ := x
def angle_D : ℝ := x
def angle_F : ℝ := 3 * x + 10

-- The formal statement to prove
theorem hexagon_largest_angle (x : ℝ) : 
  100 + 120 + x + x + (3 * x + 10) = 720 → 
  3 * x + 10 = 304 :=
by 
  sorry

end hexagon_largest_angle_l241_241512


namespace total_distance_correct_l241_241430

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end total_distance_correct_l241_241430


namespace time_interval_for_birth_and_death_rates_l241_241943

theorem time_interval_for_birth_and_death_rates
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (population_net_increase_per_day : ℝ)
  (number_of_minutes_per_day : ℝ)
  (net_increase_per_interval : ℝ)
  (time_intervals_per_day : ℝ)
  (time_interval_in_minutes : ℝ):

  birth_rate = 10 →
  death_rate = 2 →
  population_net_increase_per_day = 345600 →
  number_of_minutes_per_day = 1440 →
  net_increase_per_interval = birth_rate - death_rate →
  time_intervals_per_day = population_net_increase_per_day / net_increase_per_interval →
  time_interval_in_minutes = number_of_minutes_per_day / time_intervals_per_day →
  time_interval_in_minutes = 48 :=
by
  intros
  sorry

end time_interval_for_birth_and_death_rates_l241_241943


namespace distance_from_home_to_school_l241_241714

theorem distance_from_home_to_school
  (x y : ℝ)
  (h1 : x = y / 3)
  (h2 : x = (y + 18) / 5) : x = 9 := 
by
  sorry

end distance_from_home_to_school_l241_241714


namespace correct_options_l241_241078

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l241_241078


namespace total_ranking_sequences_at_end_l241_241130

-- Define the teams
inductive Team
| E
| F
| G
| H

open Team

-- Conditions of the problem
def split_groups : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

def saturday_matches : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

-- Function to count total ranking sequences
noncomputable def total_ranking_sequences : ℕ := 4

-- Define the main theorem
theorem total_ranking_sequences_at_end : total_ranking_sequences = 4 :=
by
  sorry

end total_ranking_sequences_at_end_l241_241130


namespace line_passes_point_l241_241611

theorem line_passes_point (k : ℝ) :
  ((1 + 4 * k) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k)) = 0 :=
by
  sorry

end line_passes_point_l241_241611


namespace impossible_all_black_l241_241426

def initial_white_chessboard (n : ℕ) : Prop :=
  n = 0

def move_inverts_three (move : ℕ → ℕ) : Prop :=
  ∀ n, move n = n + 3 ∨ move n = n - 3

theorem impossible_all_black (move : ℕ → ℕ) (n : ℕ) (initial : initial_white_chessboard n) (invert : move_inverts_three move) : ¬ ∃ k, move^[k] n = 64 :=
by sorry

end impossible_all_black_l241_241426


namespace total_amount_l241_241551

theorem total_amount (x : ℝ) (hC : 2 * x = 70) :
  let B_share := 1.25 * x
  let C_share := 2 * x
  let D_share := 0.7 * x
  let E_share := 0.5 * x
  let A_share := x
  B_share + C_share + D_share + E_share + A_share = 190.75 :=
by
  sorry

end total_amount_l241_241551


namespace option_a_correct_option_c_correct_option_d_correct_l241_241572

theorem option_a_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (1 / a > 1 / b) :=
sorry

theorem option_c_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (Real.sqrt (-a) > Real.sqrt (-b)) :=
sorry

theorem option_d_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (|a| > -b) :=
sorry

end option_a_correct_option_c_correct_option_d_correct_l241_241572


namespace fraction_covered_by_triangle_l241_241791

structure Point where
  x : ℤ
  y : ℤ

def area_of_triangle (A B C : Point) : ℚ :=
  (1/2 : ℚ) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_grid (length width : ℤ) : ℚ :=
  (length * width : ℚ)

def fraction_of_grid_covered (A B C : Point) (length width : ℤ) : ℚ :=
  (area_of_triangle A B C) / (area_of_grid length width)

theorem fraction_covered_by_triangle :
  fraction_of_grid_covered ⟨2, 4⟩ ⟨7, 2⟩ ⟨6, 5⟩ 8 6 = 13 / 96 :=
by
  sorry

end fraction_covered_by_triangle_l241_241791


namespace octal_addition_l241_241010

theorem octal_addition (x y : ℕ) (h1 : x = 1 * 8^3 + 4 * 8^2 + 6 * 8^1 + 3 * 8^0)
                     (h2 : y = 2 * 8^2 + 7 * 8^1 + 5 * 8^0) :
  x + y = 1 * 8^3 + 7 * 8^2 + 5 * 8^1 + 0 * 8^0 := sorry

end octal_addition_l241_241010


namespace minimum_n_for_obtuse_triangle_l241_241868

def α₀ : ℝ := 60 
def β₀ : ℝ := 59.999
def γ₀ : ℝ := 60.001

def α (n : ℕ) : ℝ := (-2)^n * (α₀ - 60) + 60
def β (n : ℕ) : ℝ := (-2)^n * (β₀ - 60) + 60
def γ (n : ℕ) : ℝ := (-2)^n * (γ₀ - 60) + 60

theorem minimum_n_for_obtuse_triangle : ∃ n : ℕ, β n > 90 ∧ ∀ m : ℕ, m < n → β m ≤ 90 :=
by sorry

end minimum_n_for_obtuse_triangle_l241_241868


namespace train_speeds_proof_l241_241069

-- Defining the initial conditions
variables (v_g v_p v_e : ℝ)
variables (t_g t_p t_e : ℝ) -- t_g, t_p, t_e are the times for goods, passenger, and express trains respectively

-- Conditions given in the problem
def goods_train_speed := v_g 
def passenger_train_speed := 90 
def express_train_speed := 1.5 * 90

-- Passenger train catches up with the goods train after 4 hours
def passenger_goods_catchup := 90 * 4 = v_g * (t_g + 4) - v_g * t_g

-- Express train catches up with the passenger train after 3 hours
def express_passenger_catchup := 1.5 * 90 * 3 = 90 * (3 + 4)

-- Theorem to prove the speeds of each train
theorem train_speeds_proof (h1 : 90 * 4 = v_g * (t_g + 4) - v_g * t_g)
                           (h2 : 1.5 * 90 * 3 = 90 * (3 + 4)) :
    v_g = 90 ∧ v_p = 90 ∧ v_e = 135 :=
by {
  sorry
}

end train_speeds_proof_l241_241069


namespace nickel_chocolates_l241_241511

theorem nickel_chocolates (N : ℕ) (h : 7 = N + 2) : N = 5 :=
by
  sorry

end nickel_chocolates_l241_241511


namespace line_ellipse_intersection_l241_241173

theorem line_ellipse_intersection (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m ∧ (x^2 / 4 + y^2 / 2 = 1)) →
  (-3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2) ∨
  (m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2) ∨ 
  (m < -3 * Real.sqrt 2 ∨ m > 3 * Real.sqrt 2) :=
sorry

end line_ellipse_intersection_l241_241173


namespace wrong_observation_value_l241_241388

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean correct_value wrong_value : ℚ) 
  (h₁ : n = 50)
  (h₂ : initial_mean = 36)
  (h₃ : corrected_mean = 36.5)
  (h₄ : correct_value = 60)
  (h₅ : n * corrected_mean = n * initial_mean - wrong_value + correct_value) :
  wrong_value = 35 := by
  have htotal₁ : n * initial_mean = 1800 := by sorry
  have htotal₂ : n * corrected_mean = 1825 := by sorry
  linarith

end wrong_observation_value_l241_241388


namespace elena_total_pens_l241_241751

theorem elena_total_pens (price_x price_y total_cost : ℝ) (num_x : ℕ) (hx1 : price_x = 4.0) (hx2 : price_y = 2.2) 
  (hx3 : total_cost = 42.0) (hx4 : num_x = 6) : 
  ∃ num_total : ℕ, num_total = 14 :=
by
  sorry

end elena_total_pens_l241_241751


namespace linear_function_difference_l241_241758

-- Define the problem in Lean.
theorem linear_function_difference (g : ℕ → ℝ) (h : ∀ x y : ℕ, g x = 3 * x + g 0) (h_condition : g 4 - g 1 = 9) : g 10 - g 1 = 27 := 
by
  sorry -- Proof is omitted.

end linear_function_difference_l241_241758


namespace sum_of_averages_is_six_l241_241661

variable (a b c d e : ℕ)

def average_teacher : ℚ :=
  (5 * a + 4 * b + 3 * c + 2 * d + e) / (a + b + c + d + e)

def average_kati : ℚ :=
  (5 * e + 4 * d + 3 * c + 2 * b + a) / (a + b + c + d + e)

theorem sum_of_averages_is_six (a b c d e : ℕ) : 
    average_teacher a b c d e + average_kati a b c d e = 6 := by
  sorry

end sum_of_averages_is_six_l241_241661


namespace sum_central_square_l241_241912

noncomputable def table_sum : ℕ := 10200
noncomputable def a : ℕ := 1200
noncomputable def central_sum : ℕ := 720

theorem sum_central_square :
  ∃ (a : ℕ), table_sum = a * (1 + (1 / 3) + (1 / 9) + (1 / 27)) * (1 + (1 / 4) + (1 / 16) + (1 / 64)) ∧ 
              central_sum = (a / 3) + (a / 12) + (a / 9) + (a / 36) :=
by
  sorry

end sum_central_square_l241_241912


namespace fill_in_the_blanks_l241_241347

theorem fill_in_the_blanks :
  (9 / 18 = 0.5) ∧
  (27 / 54 = 0.5) ∧
  (50 / 100 = 0.5) ∧
  (10 / 20 = 0.5) ∧
  (5 / 10 = 0.5) :=
by
  sorry

end fill_in_the_blanks_l241_241347


namespace remainder_division_l241_241536

theorem remainder_division (exists_quotient : ∃ q r : ℕ, r < 5 ∧ N = 5 * 5 + r)
    (exists_quotient_prime : ∃ k : ℕ, N = 11 * k + 3) :
  ∃ r : ℕ, r = 0 ∧ N % 5 = r := 
sorry

end remainder_division_l241_241536


namespace equation_no_solution_B_l241_241870

theorem equation_no_solution_B :
  ¬(∃ x : ℝ, |-3 * x| + 5 = 0) :=
sorry

end equation_no_solution_B_l241_241870


namespace sequence_value_proof_l241_241348

theorem sequence_value_proof : 
  (∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n : ℕ, a (2 * n) = 2 * n * a n) ∧ 
    a (2^50) = 2^1276) :=
sorry

end sequence_value_proof_l241_241348


namespace apples_per_friend_l241_241545

def Benny_apples : Nat := 5
def Dan_apples : Nat := 2 * Benny_apples
def Total_apples : Nat := Benny_apples + Dan_apples
def Number_of_friends : Nat := 3

theorem apples_per_friend : Total_apples / Number_of_friends = 5 := by
  sorry

end apples_per_friend_l241_241545


namespace number_of_people_with_cards_greater_than_0p3_l241_241636

theorem number_of_people_with_cards_greater_than_0p3 :
  (∃ (number_of_people : ℕ),
     number_of_people = (if 0.3 < 0.8 then 1 else 0) +
                        (if 0.3 < (1 / 2) then 1 else 0) +
                        (if 0.3 < 0.9 then 1 else 0) +
                        (if 0.3 < (1 / 3) then 1 else 0)) →
  number_of_people = 4 :=
by
  sorry

end number_of_people_with_cards_greater_than_0p3_l241_241636


namespace general_term_formula_of_a_l241_241391

def S (n : ℕ) : ℚ := (3 / 2) * n^2 - 2 * n

def a (n : ℕ) : ℚ :=
  if n = 1 then (3 / 2) - 2
  else 2 * (3 / 2) * n - (3 / 2) - 2

theorem general_term_formula_of_a :
  ∀ n : ℕ, n > 0 → a n = 3 * n - (7 / 2) :=
by
  intros n hn
  sorry

end general_term_formula_of_a_l241_241391


namespace common_ratio_of_geometric_sequence_l241_241039

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def is_geometric_sequence (x y z : ℝ) (q : ℝ) : Prop :=
  y^2 = x * z

theorem common_ratio_of_geometric_sequence 
    (a_n : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a_n) 
    (a1 a3 a5 : ℝ)
    (h1 : a1 = a_n 1 + 1) 
    (h3 : a3 = a_n 3 + 3) 
    (h5 : a5 = a_n 5 + 5) 
    (h_geom : is_geometric_sequence a1 a3 a5 1) : 
  1 = 1 :=
by
  sorry

end common_ratio_of_geometric_sequence_l241_241039


namespace correct_option_c_l241_241832

theorem correct_option_c (a : ℝ) : (-2 * a) ^ 3 = -8 * a ^ 3 :=
sorry

end correct_option_c_l241_241832


namespace single_interval_condition_l241_241542

-- Definitions: k and l are integers
variables (k l : ℤ)

-- Condition: The given condition for l
theorem single_interval_condition : l = Int.floor (k ^ 2 / 4) :=
sorry

end single_interval_condition_l241_241542


namespace length_of_segment_AB_l241_241790

theorem length_of_segment_AB :
  ∀ A B : ℝ × ℝ,
  (∃ x y : ℝ, y^2 = 8 * x ∧ y = (y - 0) / (4 - 2) * (x - 2))
  ∧ (A.1 + B.1) / 2 = 4
  → dist A B = 12 := 
by
  sorry

end length_of_segment_AB_l241_241790


namespace triangle_perimeter_l241_241878

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_perimeter {a b c : ℕ} (h : is_triangle 15 11 19) : 15 + 11 + 19 = 45 := by
  sorry

end triangle_perimeter_l241_241878


namespace min_value_sum_inverse_sq_l241_241423

theorem min_value_sum_inverse_sq (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 1) : 
  (39 + 1/x + 4/y + 9/z) ≥ 25 :=
by
    sorry

end min_value_sum_inverse_sq_l241_241423


namespace sum_x_y_is_4_l241_241978

theorem sum_x_y_is_4 {x y : ℝ} (h : x / (1 - (I : ℂ)) + y / (1 - 2 * I) = 5 / (1 - 3 * I)) : x + y = 4 :=
sorry

end sum_x_y_is_4_l241_241978


namespace product_of_three_numbers_l241_241243

theorem product_of_three_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 200)
  (h2 : n = 8 * x)
  (h3 : n = y - 5)
  (h4 : n = z + 5) :
  x * y * z = 372462 :=
by sorry

end product_of_three_numbers_l241_241243


namespace ram_initial_deposit_l241_241981

theorem ram_initial_deposit :
  ∃ P: ℝ, P + 100 = 1100 ∧ 1.20 * 1100 = 1320 ∧ P * 1.32 = 1320 ∧ P = 1000 :=
by
  existsi (1000 : ℝ)
  sorry

end ram_initial_deposit_l241_241981


namespace yellow_not_greater_than_green_l241_241506

theorem yellow_not_greater_than_green
    (G Y S : ℕ)
    (h1 : G + Y + S = 100)
    (h2 : G + S / 2 = 50)
    (h3 : Y + S / 2 = 50) : ¬ Y > G :=
sorry

end yellow_not_greater_than_green_l241_241506


namespace pushups_fri_is_39_l241_241716

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end pushups_fri_is_39_l241_241716


namespace math_problem_proof_l241_241713

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l241_241713


namespace find_n_l241_241066

-- Define the parameters of the arithmetic sequence
def a1 : ℤ := 1
def d : ℤ := 3
def a_n : ℤ := 298

-- The general formula for the nth term in an arithmetic sequence
def an (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The theorem to prove that n equals 100 given the conditions
theorem find_n (n : ℕ) (h : an n = a_n) : n = 100 :=
by
  sorry

end find_n_l241_241066


namespace proposition_relationship_l241_241922
-- Import library

-- Statement of the problem
theorem proposition_relationship (p q : Prop) (hpq : p ∨ q) (hnp : ¬p) : ¬p ∧ q :=
  by
  sorry

end proposition_relationship_l241_241922


namespace pipe_length_difference_l241_241982

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end pipe_length_difference_l241_241982


namespace silenos_time_l241_241232

theorem silenos_time :
  (∃ x : ℝ, ∃ b: ℝ, (x - 2 = x / 2) ∧ (b = x / 3)) → (∃ x : ℝ, x = 3) :=
by sorry

end silenos_time_l241_241232


namespace conic_sections_l241_241247

theorem conic_sections (x y : ℝ) :
  y^4 - 9 * x^4 = 3 * y^2 - 4 →
  (∃ c : ℝ, (c = 5/2 ∨ c = 1) ∧ y^2 - 3 * x^2 = c) :=
by
  sorry

end conic_sections_l241_241247


namespace moles_of_HCl_required_l241_241688

noncomputable def numberOfMolesHClRequired (moles_AgNO3 : ℕ) : ℕ :=
  if moles_AgNO3 = 3 then 3 else 0

-- Theorem statement
theorem moles_of_HCl_required : numberOfMolesHClRequired 3 = 3 := by
  sorry

end moles_of_HCl_required_l241_241688


namespace johns_mistake_l241_241862

theorem johns_mistake (a b : ℕ) (h1 : 10000 * a + b = 11 * a * b)
  (h2 : 100 ≤ a ∧ a ≤ 999) (h3 : 1000 ≤ b ∧ b ≤ 9999) : a + b = 1093 :=
sorry

end johns_mistake_l241_241862


namespace closest_point_exists_l241_241612

def closest_point_on_line_to_point (x : ℝ) (y : ℝ) : Prop :=
  ∃(p : ℝ × ℝ), p = (3, 1) ∧ ∀(q : ℝ × ℝ), q.2 = (q.1 + 3) / 3 → dist p (3, 2) ≤ dist q (3, 2)

theorem closest_point_exists :
  closest_point_on_line_to_point 3 2 :=
sorry

end closest_point_exists_l241_241612


namespace cost_to_replace_and_install_l241_241028

theorem cost_to_replace_and_install (s l : ℕ) 
  (h1 : l = 3 * s) (h2 : 2 * s + 2 * l = 640) 
  (cost_per_foot : ℕ) (cost_per_gate : ℕ) (installation_cost_per_gate : ℕ) 
  (h3 : cost_per_foot = 5) (h4 : cost_per_gate = 150) (h5 : installation_cost_per_gate = 75) : 
  (s * cost_per_foot + 2 * (cost_per_gate + installation_cost_per_gate)) = 850 := 
by 
  sorry

end cost_to_replace_and_install_l241_241028


namespace range_of_t_l241_241334

variable (t : ℝ)

def point_below_line (x y a b c : ℝ) : Prop :=
  a * x - b * y + c < 0

theorem range_of_t (t : ℝ) : point_below_line 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
  sorry

end range_of_t_l241_241334


namespace value_of_a_plus_d_l241_241518

variable (a b c d : ℝ)

theorem value_of_a_plus_d 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := 
by 
  sorry

end value_of_a_plus_d_l241_241518


namespace doll_cost_is_one_l241_241744

variable (initial_amount : ℕ) (end_amount : ℕ) (number_of_dolls : ℕ)

-- Conditions
def given_conditions : Prop :=
  initial_amount = 100 ∧
  end_amount = 97 ∧
  number_of_dolls = 3

-- Question: Proving the cost of each doll
def cost_per_doll (initial_amount end_amount number_of_dolls : ℕ) : ℕ :=
  (initial_amount - end_amount) / number_of_dolls

theorem doll_cost_is_one (h : given_conditions initial_amount end_amount number_of_dolls) :
  cost_per_doll initial_amount end_amount number_of_dolls = 1 :=
by
  sorry

end doll_cost_is_one_l241_241744


namespace sum_of_numbers_l241_241839

variable (x y S : ℝ)
variable (H1 : x + y = S)
variable (H2 : x * y = 375)
variable (H3 : (1 / x) + (1 / y) = 0.10666666666666667)

theorem sum_of_numbers (H1 : x + y = S) (H2 : x * y = 375) (H3 : (1 / x) + (1 / y) = 0.10666666666666667) : S = 40 :=
by {
  sorry
}

end sum_of_numbers_l241_241839


namespace range_of_function_l241_241819

theorem range_of_function : 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 12) ∧ 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 18) ∧ 
  (∀ y : ℝ, (12 ≤ y ∧ y ≤ 18) → 
    ∃ x : ℝ, y = |x + 5| - |x - 3| + 4) :=
by
  sorry

end range_of_function_l241_241819


namespace sum_of_angles_l241_241354

theorem sum_of_angles (ABC ABD : ℝ) (n_octagon n_triangle : ℕ) 
(h1 : n_octagon = 8) 
(h2 : n_triangle = 3) 
(h3 : ABC = 180 * (n_octagon - 2) / n_octagon)
(h4 : ABD = 180 * (n_triangle - 2) / n_triangle) : 
ABC + ABD = 195 :=
by {
  sorry
}

end sum_of_angles_l241_241354


namespace statements_correct_l241_241708

theorem statements_correct :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (∀ x : ℝ, (∀ x, x^2 + x + 1 ≠ 0) ↔ (∃ x, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) ↔ p ∧ q) ∧
  (∀ x : ℝ, (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬ (x^2 - 3*x + 2 > 0) → x ≤ 2)) :=
by
  sorry

end statements_correct_l241_241708


namespace find_sum_of_coefficients_l241_241668

theorem find_sum_of_coefficients
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + x^3 - 2 * x^2 + 17 * x - 5) :
  a + b + c + d = 5 :=
by
  sorry

end find_sum_of_coefficients_l241_241668


namespace polygon_sides_exterior_interior_sum_l241_241963

theorem polygon_sides_exterior_interior_sum (n : ℕ) (h : ((n - 2) * 180 = 360)) : n = 4 :=
by sorry

end polygon_sides_exterior_interior_sum_l241_241963


namespace roots_quadratic_expression_l241_241775

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + m - 2023 = 0) (h2 : n^2 + n - 2023 = 0) :
  m^2 + 2 * m + n = 2022 :=
by
  -- proof steps would go here
  sorry

end roots_quadratic_expression_l241_241775


namespace find_length_of_first_dimension_of_tank_l241_241335

theorem find_length_of_first_dimension_of_tank 
    (w : ℝ) (h : ℝ) (cost_per_sq_ft : ℝ) (total_cost : ℝ) (l : ℝ) :
    w = 5 → h = 3 → cost_per_sq_ft = 20 → total_cost = 1880 → 
    1880 = (2 * l * w + 2 * l * h + 2 * w * h) * cost_per_sq_ft →
    l = 4 := 
by
  intros hw hh hcost htotal heq
  sorry

end find_length_of_first_dimension_of_tank_l241_241335


namespace geometric_progression_common_point_l241_241416

theorem geometric_progression_common_point (a r : ℝ) :
  ∀ x y : ℝ, (a ≠ 0 ∧ x = 1 ∧ y = 0) ↔ (a * x + (a * r) * y = a * r^2) := by
  sorry

end geometric_progression_common_point_l241_241416


namespace remainder_1394_mod_2535_l241_241974

-- Definition of the least number satisfying the given conditions
def L : ℕ := 1394

-- Proof statement: proving the remainder of division
theorem remainder_1394_mod_2535 : (1394 % 2535) = 1394 :=
by sorry

end remainder_1394_mod_2535_l241_241974


namespace product_or_double_is_perfect_square_l241_241318

variable {a b c : ℤ}

-- Conditions
def sides_of_triangle (a b c : ℤ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def no_common_divisor (a b c : ℤ) : Prop := gcd (gcd a b) c = 1

def all_fractions_are_integers (a b c : ℤ) : Prop :=
  (a + b - c) ≠ 0 ∧ (b + c - a) ≠ 0 ∧ (c + a - b) ≠ 0 ∧
  ((a^2 + b^2 - c^2) % (a + b - c) = 0) ∧ 
  ((b^2 + c^2 - a^2) % (b + c - a) = 0) ∧ 
  ((c^2 + a^2 - b^2) % (c + a - b) = 0)

-- Mathematical proof problem statement in Lean 4
theorem product_or_double_is_perfect_square (a b c : ℤ) 
  (h1 : sides_of_triangle a b c)
  (h2 : no_common_divisor a b c)
  (h3 : all_fractions_are_integers a b c) :
  ∃ k : ℤ, k^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
           k^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := sorry

end product_or_double_is_perfect_square_l241_241318


namespace original_square_perimeter_l241_241873

theorem original_square_perimeter (x : ℝ) 
  (h1 : ∀ r, r = x ∨ r = 4 * x) 
  (h2 : 28 * x = 56) : 
  4 * (4 * x) = 32 :=
by
  -- We don't need to consider the proof as per instructions.
  sorry

end original_square_perimeter_l241_241873


namespace probability_same_suit_JQKA_l241_241805

theorem probability_same_suit_JQKA  : 
  let deck_size := 52 
  let prob_J := 4 / deck_size
  let prob_Q_given_J := 1 / (deck_size - 1) 
  let prob_K_given_JQ := 1 / (deck_size - 2)
  let prob_A_given_JQK := 1 / (deck_size - 3)
  prob_J * prob_Q_given_J * prob_K_given_JQ * prob_A_given_JQK = 1 / 1624350 :=
by
  sorry

end probability_same_suit_JQKA_l241_241805


namespace hexagon_area_l241_241309

noncomputable def area_of_hexagon (P Q R P' Q' R' : Point) (radius : ℝ) : ℝ :=
  -- a placeholder for the actual area calculation
  sorry 

theorem hexagon_area (P Q R P' Q' R' : Point) 
  (radius : ℝ) (perimeter : ℝ) :
  radius = 9 → perimeter = 42 →
  area_of_hexagon P Q R P' Q' R' radius = 189 := by
  intros h1 h2
  sorry

end hexagon_area_l241_241309


namespace additional_money_needed_for_free_shipping_l241_241429

-- Define the prices of the books
def price_book1 : ℝ := 13.00
def price_book2 : ℝ := 15.00
def price_book3 : ℝ := 10.00
def price_book4 : ℝ := 10.00

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Calculate the discounted prices
def discounted_price_book1 : ℝ := price_book1 * (1 - discount_rate)
def discounted_price_book2 : ℝ := price_book2 * (1 - discount_rate)

-- Sum of discounted prices of books
def total_cost : ℝ := discounted_price_book1 + discounted_price_book2 + price_book3 + price_book4

-- Free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- The proof statement
theorem additional_money_needed_for_free_shipping : additional_amount = 9.00 := by
  -- calculation steps omitted
  sorry

end additional_money_needed_for_free_shipping_l241_241429


namespace roots_odd_even_l241_241086

theorem roots_odd_even (n : ℤ) (x1 x2 : ℤ) (h_eqn : x1^2 + (4 * n + 1) * x1 + 2 * n = 0) (h_eqn' : x2^2 + (4 * n + 1) * x2 + 2 * n = 0) :
  ((x1 % 2 = 0 ∧ x2 % 2 ≠ 0) ∨ (x1 % 2 ≠ 0 ∧ x2 % 2 = 0)) :=
sorry

end roots_odd_even_l241_241086


namespace jennifer_total_discount_is_28_l241_241501

-- Define the conditions in the Lean context

def initial_whole_milk_cans : ℕ := 40 
def mark_whole_milk_cans : ℕ := 30 
def mark_skim_milk_cans : ℕ := 15 
def almond_milk_per_3_whole_milk : ℕ := 2 
def whole_milk_per_5_skim_milk : ℕ := 4 
def discount_per_10_whole_milk : ℕ := 4 
def discount_per_7_almond_milk : ℕ := 3 
def discount_per_3_almond_milk : ℕ := 1

def jennifer_additional_almond_milk := (mark_whole_milk_cans / 3) * almond_milk_per_3_whole_milk
def jennifer_additional_whole_milk := (mark_skim_milk_cans / 5) * whole_milk_per_5_skim_milk

def jennifer_whole_milk_cans := initial_whole_milk_cans + jennifer_additional_whole_milk
def jennifer_almond_milk_cans := jennifer_additional_almond_milk

def jennifer_whole_milk_discount := (jennifer_whole_milk_cans / 10) * discount_per_10_whole_milk
def jennifer_almond_milk_discount := 
  (jennifer_almond_milk_cans / 7) * discount_per_7_almond_milk + 
  ((jennifer_almond_milk_cans % 7) / 3) * discount_per_3_almond_milk

def total_jennifer_discount := jennifer_whole_milk_discount + jennifer_almond_milk_discount

-- Theorem stating the total discount 
theorem jennifer_total_discount_is_28 : total_jennifer_discount = 28 := by
  sorry

end jennifer_total_discount_is_28_l241_241501


namespace overall_percentage_increase_correct_l241_241185

def initial_salary : ℕ := 60
def first_raise_salary : ℕ := 90
def second_raise_salary : ℕ := 120
def gym_deduction : ℕ := 10

def final_salary : ℕ := second_raise_salary - gym_deduction
def salary_difference : ℕ := final_salary - initial_salary
def percentage_increase : ℚ := (salary_difference : ℚ) / initial_salary * 100

theorem overall_percentage_increase_correct :
  percentage_increase = 83.33 := by
  sorry

end overall_percentage_increase_correct_l241_241185


namespace no_solution_for_equation_l241_241880

theorem no_solution_for_equation :
  ¬ ∃ x : ℝ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  sorry

end no_solution_for_equation_l241_241880


namespace arith_seq_sum_first_110_l241_241507

variable {α : Type*} [OrderedRing α]

theorem arith_seq_sum_first_110 (a₁ d : α) :
  (10 * a₁ + 45 * d = 100) →
  (100 * a₁ + 4950 * d = 10) →
  (110 * a₁ + 5995 * d = -110) :=
by
  intros h1 h2
  sorry

end arith_seq_sum_first_110_l241_241507


namespace parabola_directrix_x_eq_neg1_eqn_l241_241263

theorem parabola_directrix_x_eq_neg1_eqn :
  (∀ y : ℝ, ∃ x : ℝ, x = -1 → y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_x_eq_neg1_eqn_l241_241263


namespace no_prime_pairs_sum_53_l241_241210

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l241_241210


namespace tangent_to_parabola_l241_241037

theorem tangent_to_parabola {k : ℝ} : 
  (∀ x y : ℝ, (4 * x + 3 * y + k = 0) ↔ (y ^ 2 = 16 * x)) → k = 9 :=
by
  sorry

end tangent_to_parabola_l241_241037


namespace find_sum_l241_241509

variable (a b : ℚ)

theorem find_sum :
  2 * a + 5 * b = 31 ∧ 4 * a + 3 * b = 35 → a + b = 68 / 7 := by
  sorry

end find_sum_l241_241509


namespace sequence_a10_l241_241796

theorem sequence_a10 : 
  (∃ (a : ℕ → ℤ), 
    a 1 = -1 ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n) - a (2*n - 1) = 2^(2*n-1)) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n + 1) - a (2*n) = 2^(2*n))) → 
  (∃ a : ℕ → ℤ, a 10 = 1021) :=
by
  intro h
  obtain ⟨a, h1, h2, h3⟩ := h
  sorry

end sequence_a10_l241_241796


namespace book_purchasing_methods_l241_241837

theorem book_purchasing_methods :
  ∃ (A B C D : ℕ),
  A + B + C + D = 10 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
  3 * A + 5 * B + 7 * C + 11 * D = 70 ∧
  (∃ N : ℕ, N = 4) :=
by sorry

end book_purchasing_methods_l241_241837


namespace m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l241_241485

-- Defining the sequence condition
def seq_condition (a : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n ≥ 2, a n ^ 2 - a (n + 1) * a (n - 1) = m * (a 2 - a 1) ^ 2

-- (1) Value of m for an arithmetic sequence with a non-zero common difference
theorem m_value_for_arithmetic_seq {a : ℕ → ℝ} (d : ℝ) (h_nonzero : d ≠ 0) :
  (∀ n, a (n + 1) = a n + d) → seq_condition a 1 :=
by
  sorry

-- (2) Minimum value of t given specific conditions
theorem min_value_t {t p : ℝ} (a : ℕ → ℝ) (h_p : 3 ≤ p ∧ p ≤ 5) :
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ (∀ n, t * a n + p ≥ n) → t = 1 / 32 :=
by
  sorry

-- (3) Smallest value of T for non-constant periodic sequence
theorem smallest_T_periodic_seq {a : ℕ → ℝ} {m : ℝ} (h_m_nonzero : m ≠ 0) :
  seq_condition a m → (∀ n, a (n + T) = a n) → (∃ T' > 0, ∀ T'', T'' > 0 → T'' = 3) :=
by
  sorry

end m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l241_241485


namespace second_day_speed_faster_l241_241305

def first_day_distance := 18
def first_day_speed := 3
def first_day_time := first_day_distance / first_day_speed
def second_day_time := first_day_time - 1
def third_day_speed := 5
def third_day_time := 3
def third_day_distance := third_day_speed * third_day_time
def total_distance := 53

theorem second_day_speed_faster :
  ∃ r2, (first_day_distance + (second_day_time * r2) + third_day_distance = total_distance) → (r2 - first_day_speed = 1) :=
by
  sorry

end second_day_speed_faster_l241_241305


namespace history_paper_pages_l241_241671

theorem history_paper_pages (days: ℕ) (pages_per_day: ℕ) (h₁: days = 3) (h₂: pages_per_day = 27) : days * pages_per_day = 81 := 
by
  sorry

end history_paper_pages_l241_241671


namespace father_age_l241_241158

theorem father_age (M F : ℕ) 
  (h1 : M = 2 * F / 5) 
  (h2 : M + 10 = (F + 10) / 2) : F = 50 :=
sorry

end father_age_l241_241158


namespace angle_between_clock_hands_at_7_30_l241_241726

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end angle_between_clock_hands_at_7_30_l241_241726


namespace mints_ratio_l241_241886

theorem mints_ratio (n : ℕ) (green_mints red_mints : ℕ) (h1 : green_mints + red_mints = n) (h2 : green_mints = 3 * (n / 4)) : green_mints / red_mints = 3 :=
by
  sorry

end mints_ratio_l241_241886


namespace solve_ab_sum_l241_241397

theorem solve_ab_sum (x a b : ℝ) (ha : ℕ) (hb : ℕ)
  (h1 : a = ha)
  (h2 : b = hb)
  (h3 : x = a + Real.sqrt b)
  (h4 : x^2 + 3 * x + 3 / x + 1 / x^2 = 26) :
  (ha + hb = 5) :=
sorry

end solve_ab_sum_l241_241397


namespace no_integer_pairs_satisfy_equation_l241_241921

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ m n : ℤ, m^3 + 8 * m^2 + 17 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1 :=
sorry

end no_integer_pairs_satisfy_equation_l241_241921


namespace tangents_product_is_constant_MN_passes_fixed_point_l241_241799

-- Define the parabola C and the tangency conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

variables {x1 y1 x2 y2 : ℝ}

-- Point G is on the axis of the parabola C (we choose the y-axis for part 2)
def point_G_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Two tangent points from G to the parabola at A (x1, y1) and B (x2, y2)
def tangent_points (G : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂

-- Question 1 proof statement
theorem tangents_product_is_constant (G : ℝ × ℝ) (hG : point_G_on_axis G)
  (hT : tangent_points G x1 y1 x2 y2) : x1 * x2 + y1 * y2 = -3 := sorry

variables {M N : ℝ × ℝ}

-- Question 2 proof statement
theorem MN_passes_fixed_point {G : ℝ × ℝ} (hG : G.1 = 0) (xM yM xN yN : ℝ)
 (hMA : parabola M.1 M.2) (hMB : parabola N.1 N.2)
 (h_perpendicular : (M.1 - G.1) * (N.1 - G.1) + (M.2 - G.2) * (N.2 - G.2) = 0)
 : ∃ P, P = (2, 5) := sorry

end tangents_product_is_constant_MN_passes_fixed_point_l241_241799


namespace least_pos_int_x_l241_241892

theorem least_pos_int_x (x : ℕ) (h1 : ∃ k : ℤ, (3 * x + 43) = 53 * k) 
  : x = 21 :=
sorry

end least_pos_int_x_l241_241892


namespace range_of_a_l241_241571

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 :=
by {
  sorry
}

end range_of_a_l241_241571


namespace min_sum_of_arithmetic_sequence_terms_l241_241447

open Real

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a m = a n + d * (m - n)

theorem min_sum_of_arithmetic_sequence_terms (a : ℕ → ℝ) 
  (hpos : ∀ n, a n > 0) 
  (harith : arithmetic_sequence a) 
  (hprod : a 1 * a 20 = 100) : 
  a 7 + a 14 ≥ 20 := sorry

end min_sum_of_arithmetic_sequence_terms_l241_241447


namespace min_n_for_binomial_constant_term_l241_241579

theorem min_n_for_binomial_constant_term : ∃ (n : ℕ), n > 0 ∧ 3 * n - 7 * ((3 * n) / 7) = 0 ∧ n = 7 :=
by {
  sorry
}

end min_n_for_binomial_constant_term_l241_241579


namespace sqrt_div_l241_241082

theorem sqrt_div (a b : ℝ) (h1 : a = 28) (h2 : b = 7) :
  Real.sqrt a / Real.sqrt b = 2 := 
by 
  sorry

end sqrt_div_l241_241082


namespace total_handshakes_five_people_l241_241499

theorem total_handshakes_five_people : 
  let n := 5
  let total_handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2
  total_handshakes 5 = 10 :=
by sorry

end total_handshakes_five_people_l241_241499


namespace min_value_A2_minus_B2_l241_241205

noncomputable def A (p q r : ℝ) : ℝ := 
  Real.sqrt (p + 3) + Real.sqrt (q + 6) + Real.sqrt (r + 12)

noncomputable def B (p q r : ℝ) : ℝ :=
  Real.sqrt (p + 2) + Real.sqrt (q + 2) + Real.sqrt (r + 2)

theorem min_value_A2_minus_B2
  (h₁ : 0 ≤ p)
  (h₂ : 0 ≤ q)
  (h₃ : 0 ≤ r) :
  ∃ (p q r : ℝ), A p q r ^ 2 - B p q r ^ 2 = 35 + 10 * Real.sqrt 10 := 
sorry

end min_value_A2_minus_B2_l241_241205


namespace prove_n_prime_l241_241149

theorem prove_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Prime p) (h1 : n > 0) (h2 : 3^n - 2^n = p^k) : Prime n :=
by {
  sorry
}

end prove_n_prime_l241_241149


namespace pizza_cost_l241_241681

theorem pizza_cost
  (initial_money_frank : ℕ)
  (initial_money_bill : ℕ)
  (final_money_bill : ℕ)
  (pizza_cost : ℕ)
  (number_of_pizzas : ℕ)
  (money_given_to_bill : ℕ) :
  initial_money_frank = 42 ∧
  initial_money_bill = 30 ∧
  final_money_bill = 39 ∧
  number_of_pizzas = 3 ∧
  money_given_to_bill = final_money_bill - initial_money_bill →
  3 * pizza_cost + money_given_to_bill = initial_money_frank →
  pizza_cost = 11 :=
by
  sorry

end pizza_cost_l241_241681


namespace margo_total_distance_travelled_l241_241540

noncomputable def total_distance_walked (walking_time_in_minutes: ℝ) (stopping_time_in_minutes: ℝ) (additional_walking_time_in_minutes: ℝ) (walking_speed: ℝ) : ℝ :=
  walking_speed * ((walking_time_in_minutes + stopping_time_in_minutes + additional_walking_time_in_minutes) / 60)

noncomputable def total_distance_cycled (cycling_time_in_minutes: ℝ) (cycling_speed: ℝ) : ℝ :=
  cycling_speed * (cycling_time_in_minutes / 60)

theorem margo_total_distance_travelled :
  let walking_time := 10
  let stopping_time := 15
  let additional_walking_time := 10
  let cycling_time := 15
  let walking_speed := 4
  let cycling_speed := 10

  total_distance_walked walking_time stopping_time additional_walking_time walking_speed +
  total_distance_cycled cycling_time cycling_speed = 4.8333 := 
by 
  sorry

end margo_total_distance_travelled_l241_241540


namespace simplify_and_evaluate_expression_l241_241326

variable (x : ℝ) (h : x = Real.sqrt 2 - 1)

theorem simplify_and_evaluate_expression : 
  (1 - 1 / (x + 1)) / (x / (x^2 + 2 * x + 1)) = Real.sqrt 2 :=
by
  -- Using the given definition of x
  have hx : x = Real.sqrt 2 - 1 := h
  
  -- Required proof should go here 
  sorry

end simplify_and_evaluate_expression_l241_241326


namespace income_increase_l241_241679

-- Definitions based on conditions
def original_price := 1.0
def original_items := 100.0
def discount := 0.10
def increased_sales := 0.15

-- Calculations for new values
def new_price := original_price * (1 - discount)
def new_items := original_items * (1 + increased_sales)
def original_income := original_price * original_items
def new_income := new_price * new_items

-- The percentage increase in income
def percentage_increase := ((new_income - original_income) / original_income) * 100

-- The theorem to prove that the percentage increase in gross income is 3.5%
theorem income_increase : percentage_increase = 3.5 := 
by
  -- This is where the proof would go
  sorry

end income_increase_l241_241679


namespace valid_exponent_rule_l241_241623

theorem valid_exponent_rule (a : ℝ) : (a^3)^2 = a^6 :=
by
  sorry

end valid_exponent_rule_l241_241623


namespace octopus_dressing_orders_l241_241731

/-- A robotic octopus has four legs, and each leg needs to wear a glove before it can wear a boot.
    Additionally, it has two tentacles that require one bracelet each before putting anything on the legs.
    The total number of valid dressing orders is 1,286,400. -/
theorem octopus_dressing_orders : 
  ∃ (n : ℕ), n = 1286400 :=
by
  sorry

end octopus_dressing_orders_l241_241731


namespace factor_expression_l241_241344

theorem factor_expression (x : ℝ) : 9 * x^2 + 3 * x = 3 * x * (3 * x + 1) := 
by
  sorry

end factor_expression_l241_241344


namespace cube_face_min_sum_l241_241570

open Set

theorem cube_face_min_sum (S : Finset ℕ)
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8})
  (h_faces_sum : ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → 
                    (a + b + c >= 10) ∨ 
                    (a + b + d >= 10) ∨ 
                    (a + c + d >= 10) ∨ 
                    (b + c + d >= 10)) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 16 :=
sorry

end cube_face_min_sum_l241_241570


namespace Carla_total_counts_l241_241003

def Monday_counts := (60 * 2) + (120 * 2) + (10 * 2)
def Tuesday_counts := (60 * 3) + (120 * 2) + (10 * 1)
def Wednesday_counts := (80 * 4) + (24 * 5)
def Thursday_counts := (60 * 1) + (80 * 2) + (120 * 3) + (10 * 4) + (24 * 5)
def Friday_counts := (60 * 1) + (120 * 2) + (80 * 2) + (10 * 3) + (24 * 3)

def total_counts := Monday_counts + Tuesday_counts + Wednesday_counts + Thursday_counts + Friday_counts

theorem Carla_total_counts : total_counts = 2552 :=
by 
  sorry

end Carla_total_counts_l241_241003


namespace friend_redistribution_l241_241124

-- Definitions of friends' earnings
def earnings := [18, 22, 26, 32, 47]

-- Definition of total earnings
def totalEarnings := earnings.sum

-- Definition of equal share
def equalShare := totalEarnings / earnings.length

-- The amount that the friend who earned 47 needs to redistribute
def redistributionAmount := 47 - equalShare

-- The goal to prove
theorem friend_redistribution:
  redistributionAmount = 18 := by
  sorry

end friend_redistribution_l241_241124


namespace probability_correct_l241_241261

-- Definitions of the problem components
def total_beads : Nat := 7
def red_beads : Nat := 4
def white_beads : Nat := 2
def green_bead : Nat := 1

-- The total number of permutations of the given multiset
def total_permutations : Nat :=
  Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial green_bead)

-- The number of valid permutations where no two neighboring beads are the same color
def valid_permutations : Nat := 14 -- As derived in the solution steps

-- The probability that no two neighboring beads are the same color
def probability_no_adjacent_same_color : Rat :=
  valid_permutations / total_permutations

-- The theorem to be proven
theorem probability_correct :
  probability_no_adjacent_same_color = 2 / 15 :=
by
  -- Proof omitted
  sorry

end probability_correct_l241_241261


namespace movie_box_office_growth_l241_241588

theorem movie_box_office_growth 
  (x : ℝ) 
  (r₁ r₃ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₃ : r₃ = 2.4) 
  (growth : r₃ = (1 + x) ^ 2) : 
  (1 + x) ^ 2 = 2.4 :=
by sorry

end movie_box_office_growth_l241_241588


namespace only_1996_is_leap_l241_241944

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))

def is_leap_year_1996 := is_leap_year 1996
def is_leap_year_1998 := is_leap_year 1998
def is_leap_year_2010 := is_leap_year 2010
def is_leap_year_2100 := is_leap_year 2100

theorem only_1996_is_leap : 
  is_leap_year_1996 ∧ ¬is_leap_year_1998 ∧ ¬is_leap_year_2010 ∧ ¬is_leap_year_2100 :=
by 
  -- proof will be added here later
  sorry

end only_1996_is_leap_l241_241944


namespace ratio_proof_l241_241054

variables {d l e : ℕ} -- Define variables representing the number of doctors, lawyers, and engineers
variables (hd : ℕ → ℕ) (hl : ℕ → ℕ) (he : ℕ → ℕ) (ho : ℕ → ℕ)

-- Condition: Average ages
def avg_age_doctors := 40 * d
def avg_age_lawyers := 55 * l
def avg_age_engineers := 35 * e

-- Condition: Overall average age is 45 years
def overall_avg_age := (40 * d + 55 * l + 35 * e) / (d + l + e)

theorem ratio_proof (h1 : 40 * d + 55 * l + 35 * e = 45 * (d + l + e)) : 
  d = l ∧ e = 2 * l :=
by
  sorry

end ratio_proof_l241_241054


namespace part1_part2_l241_241804

def A := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2 * x + 1 - m^2 ≤ 0}

theorem part1 (m : ℝ) (hm : m = 2) :
  A ∩ {x : ℝ | x < -1 ∨ 3 < x} = {x : ℝ | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} :=
sorry

theorem part2 :
  (∀ x, x ∈ A → x ∈ B (m : ℝ)) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end part1_part2_l241_241804


namespace P_subset_M_l241_241646

def P : Set ℝ := {x | x^2 - 6 * x + 9 = 0}
def M : Set ℝ := {x | x > 1}

theorem P_subset_M : P ⊂ M := by sorry

end P_subset_M_l241_241646


namespace derivative_at_zero_l241_241864

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem derivative_at_zero : deriv f 0 = 720 :=
by
  sorry

end derivative_at_zero_l241_241864


namespace find_a_l241_241694

theorem find_a (z a : ℂ) (h1 : ‖z‖ = 2) (h2 : (z - a)^2 = a) : a = 2 :=
sorry

end find_a_l241_241694


namespace betty_paid_total_l241_241258

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end betty_paid_total_l241_241258


namespace percentage_of_loss_l241_241976

theorem percentage_of_loss (CP SP : ℕ) (h1 : CP = 1750) (h2 : SP = 1610) : 
  (CP - SP) * 100 / CP = 8 := by
  sorry

end percentage_of_loss_l241_241976


namespace calculate_distribution_l241_241482

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end calculate_distribution_l241_241482


namespace total_time_equals_l241_241627

-- Define the distances and speeds
def first_segment_distance : ℝ := 50
def first_segment_speed : ℝ := 30
def second_segment_distance (b : ℝ) : ℝ := b
def second_segment_speed : ℝ := 80

-- Prove that the total time is equal to (400 + 3b) / 240 hours
theorem total_time_equals (b : ℝ) : 
  (first_segment_distance / first_segment_speed) + (second_segment_distance b / second_segment_speed) 
  = (400 + 3 * b) / 240 := 
by
  sorry

end total_time_equals_l241_241627


namespace class_size_l241_241246

theorem class_size (g : ℕ) (h1 : g + (g + 3) = 44) (h2 : g^2 + (g + 3)^2 = 540) : g + (g + 3) = 44 :=
by
  sorry

end class_size_l241_241246


namespace ratio_of_a_to_b_in_arithmetic_sequence_l241_241727

theorem ratio_of_a_to_b_in_arithmetic_sequence (a x b : ℝ) (h : a = 0 ∧ b = 2 * x) : (a / b) = 0 :=
  by sorry

end ratio_of_a_to_b_in_arithmetic_sequence_l241_241727


namespace cost_of_football_and_basketball_max_number_of_basketballs_l241_241762

-- Problem 1: Cost of one football and one basketball
theorem cost_of_football_and_basketball (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 310) 
  (h2 : 2 * x + 5 * y = 500) : 
  x = 50 ∧ y = 80 :=
sorry

-- Problem 2: Maximum number of basketballs
theorem max_number_of_basketballs (x : ℝ) 
  (h1 : 50 * (96 - x) + 80 * x ≤ 5800) 
  (h2 : x ≥ 0) 
  (h3 : x ≤ 96) : 
  x ≤ 33 :=
sorry

end cost_of_football_and_basketball_max_number_of_basketballs_l241_241762


namespace find_points_PQ_l241_241761

-- Define the points A, B, M, and E in 3D space
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨10, 0, 0⟩
def M : Point := ⟨5, 5, 0⟩
def E : Point := ⟨0, 0, 10⟩

-- Define the lines AB and EM
def line_AB (t : ℝ) : Point := ⟨10 * t, 0, 0⟩
def line_EM (s : ℝ) : Point := ⟨5 * s, 5 * s, 10 - 10 * s⟩

-- Define the points P and Q
def P (t : ℝ) : Point := line_AB t
def Q (s : ℝ) : Point := line_EM s

-- Define the distance function in 3D space
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

-- The main theorem
theorem find_points_PQ (t s : ℝ) (h1 : t = 0.4) (h2 : s = 0.8) :
  (P t = ⟨4, 0, 0⟩) ∧ (Q s = ⟨4, 4, 2⟩) ∧
  (distance (P t) (Q s) = distance (line_AB 0.4) (line_EM 0.8)) :=
by
  sorry

end find_points_PQ_l241_241761


namespace non_zero_real_y_satisfies_l241_241492

theorem non_zero_real_y_satisfies (y : ℝ) (h : y ≠ 0) : (8 * y) ^ 3 = (16 * y) ^ 2 → y = 1 / 2 :=
by
  -- Lean code placeholders
  sorry

end non_zero_real_y_satisfies_l241_241492


namespace each_person_gets_equal_share_l241_241560

-- Definitions based on the conditions
def number_of_friends: Nat := 4
def initial_chicken_wings: Nat := 9
def additional_chicken_wings: Nat := 7

-- The proof statement
theorem each_person_gets_equal_share (total_chicken_wings := initial_chicken_wings + additional_chicken_wings) : 
       total_chicken_wings / number_of_friends = 4 := 
by 
  sorry

end each_person_gets_equal_share_l241_241560


namespace WallLengthBy40Men_l241_241487

-- Definitions based on the problem conditions
def men1 : ℕ := 20
def length1 : ℕ := 112
def days1 : ℕ := 6

def men2 : ℕ := 40
variable (y : ℕ)  -- given 'y' days

-- Establish the relationship based on the given conditions
theorem WallLengthBy40Men :
  ∃ x : ℕ, x = (men2 / men1) * length1 * (y / days1) :=
by
  sorry

end WallLengthBy40Men_l241_241487


namespace roots_of_polynomial_l241_241298

def p (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (p (-1) = 0) ∧ (p 2 = 0) ∧ (p (-2) = 0) ∧ 
  ∀ x, p x = 0 → (x = -1 ∨ x = 2 ∨ x = -2) :=
by
  sorry

end roots_of_polynomial_l241_241298


namespace inequality_four_a_cubed_sub_l241_241472

theorem inequality_four_a_cubed_sub (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  4 * a^3 * (a - b) ≥ a^4 - b^4 :=
sorry

end inequality_four_a_cubed_sub_l241_241472


namespace rational_function_domain_l241_241168

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 - 4*x + 5) / (x^2 - 5*x + 4)

theorem rational_function_domain :
  {x : ℝ | ∃ y, h y = h x } = {x : ℝ | x ≠ 1 ∧ x ≠ 4} := 
sorry

end rational_function_domain_l241_241168


namespace expand_and_simplify_l241_241995

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5 * x - 66 :=
by
  sorry

end expand_and_simplify_l241_241995


namespace find_amount_with_r_l241_241965

variable (p q r : ℝ)

-- Condition 1: p, q, and r have Rs. 6000 among themselves.
def total_amount : Prop := p + q + r = 6000

-- Condition 2: r has two-thirds of the total amount with p and q.
def r_amount : Prop := r = (2 / 3) * (p + q)

theorem find_amount_with_r (h1 : total_amount p q r) (h2 : r_amount p q r) : r = 2400 := by
  sorry

end find_amount_with_r_l241_241965


namespace point_A_in_Quadrant_IV_l241_241125

-- Define the coordinates of point A
def A : ℝ × ℝ := (5, -4)

-- Define the quadrants based on x and y signs
def in_Quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_Quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_Quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_Quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Statement to prove that point A lies in Quadrant IV
theorem point_A_in_Quadrant_IV : in_Quadrant_IV A :=
by
  sorry

end point_A_in_Quadrant_IV_l241_241125


namespace sin_of_3halfpiplus2theta_l241_241374

theorem sin_of_3halfpiplus2theta (θ : ℝ) (h : Real.tan θ = 1 / 3) : Real.sin (3 * π / 2 + 2 * θ) = -4 / 5 := 
by 
  sorry

end sin_of_3halfpiplus2theta_l241_241374


namespace min_value_of_expression_l241_241732

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : 
  (1/x + 4/y + 9/z) >= 36/5 :=
sorry

end min_value_of_expression_l241_241732


namespace largest_prime_factor_4851_l241_241470

theorem largest_prime_factor_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 4851 → q ≤ p) :=
by
  -- todo: provide actual proof
  sorry

end largest_prime_factor_4851_l241_241470


namespace multiple_choice_questions_count_l241_241736

variable (M F : ℕ)

-- Conditions
def totalQuestions := M + F = 60
def totalStudyTime := 15 * M + 25 * F = 1200

-- Statement to prove
theorem multiple_choice_questions_count (h1 : totalQuestions M F) (h2 : totalStudyTime M F) : M = 30 := by
  sorry

end multiple_choice_questions_count_l241_241736


namespace first_term_of_geometric_series_l241_241532

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l241_241532


namespace sin_675_eq_neg_sqrt2_div_2_l241_241709

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end sin_675_eq_neg_sqrt2_div_2_l241_241709


namespace find_y_l241_241740

theorem find_y (x y : ℤ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 :=
sorry

end find_y_l241_241740


namespace johns_profit_l241_241132

-- Definitions based on Conditions
def original_price_per_bag : ℝ := 4
def discount_percentage : ℝ := 0.10
def discounted_price_per_bag := original_price_per_bag * (1 - discount_percentage)
def bags_bought : ℕ := 30
def cost_per_bag : ℝ := if bags_bought >= 20 then discounted_price_per_bag else original_price_per_bag
def total_cost := bags_bought * cost_per_bag
def bags_sold_to_adults : ℕ := 20
def bags_sold_to_children : ℕ := 10
def price_per_bag_for_adults : ℝ := 8
def price_per_bag_for_children : ℝ := 6
def revenue_from_adults := bags_sold_to_adults * price_per_bag_for_adults
def revenue_from_children := bags_sold_to_children * price_per_bag_for_children
def total_revenue := revenue_from_adults + revenue_from_children
def profit := total_revenue - total_cost

-- Lean Statement to be Proven
theorem johns_profit : profit = 112 :=
by
  sorry

end johns_profit_l241_241132


namespace completing_the_square_l241_241383

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l241_241383


namespace nut_game_winning_strategy_l241_241064

theorem nut_game_winning_strategy (N : ℕ) (h : N > 2) : ∃ second_player_wins : Prop, second_player_wins :=
sorry

end nut_game_winning_strategy_l241_241064


namespace average_salary_all_workers_l241_241613

-- Definitions based on the conditions
def technicians_avg_salary := 16000
def rest_avg_salary := 6000
def total_workers := 35
def technicians := 7
def rest_workers := total_workers - technicians

-- Prove that the average salary of all workers is 8000
theorem average_salary_all_workers :
  (technicians * technicians_avg_salary + rest_workers * rest_avg_salary) / total_workers = 8000 := by
  sorry

end average_salary_all_workers_l241_241613


namespace store_profit_l241_241753

theorem store_profit 
  (cost_per_item : ℕ)
  (selling_price_decrease : ℕ → ℕ)
  (profit : ℤ)
  (x : ℕ) :
  cost_per_item = 40 →
  (∀ x, selling_price_decrease x = 150 - 5 * (x - 50)) →
  profit = 1500 →
  (((x = 50 ∧ selling_price_decrease 50 = 150) ∨ (x = 70 ∧ selling_price_decrease 70 = 50)) ↔ (x = 50 ∨ x = 70) ∧ profit = 1500) :=
by
  sorry

end store_profit_l241_241753


namespace janet_percentage_l241_241110

-- Define the number of snowballs made by Janet and her brother
def janet_snowballs : ℕ := 50
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- State the theorem to be proved
theorem janet_percentage : percentage janet_snowballs total_snowballs = 25 := by
  sorry

end janet_percentage_l241_241110


namespace externally_tangent_circles_solution_l241_241420

theorem externally_tangent_circles_solution (R1 R2 d : Real)
  (h1 : R1 > 0) (h2 : R2 > 0) (h3 : R1 + R2 > d) :
  (1/R1) + (1/R2) = 2/d :=
sorry

end externally_tangent_circles_solution_l241_241420


namespace range_a_l241_241433

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else x - 1

theorem range_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 ≠ a^2 - 2 * a ∧ f x2 ≠ a^2 - 2 * a ∧ f x3 ≠ a^2 - 2 * a) ↔ (0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2) :=
by
  sorry

end range_a_l241_241433


namespace option_B_is_incorrect_l241_241025

-- Define the set A
def A := { x : ℤ | x ^ 2 - 4 = 0 }

-- Statement to prove that -2 is an element of A
theorem option_B_is_incorrect : -2 ∈ A :=
sorry

end option_B_is_incorrect_l241_241025


namespace Aunt_Lucy_gift_correct_l241_241923

def Jade_initial : ℕ := 38
def Julia_initial : ℕ := Jade_initial / 2
def Jack_initial : ℕ := 12
def John_initial : ℕ := 15
def Jane_initial : ℕ := 20

def Aunt_Mary_gift : ℕ := 65
def Aunt_Susan_gift : ℕ := 70

def total_initial : ℕ :=
  Jade_initial + Julia_initial + Jack_initial + John_initial + Jane_initial

def total_after_gifts : ℕ := 225
def total_gifts : ℕ := total_after_gifts - total_initial
def Aunt_Lucy_gift : ℕ := total_gifts - (Aunt_Mary_gift + Aunt_Susan_gift)

theorem Aunt_Lucy_gift_correct :
  Aunt_Lucy_gift = total_after_gifts - total_initial - (Aunt_Mary_gift + Aunt_Susan_gift) := by
  sorry

end Aunt_Lucy_gift_correct_l241_241923


namespace num_sets_satisfying_union_is_four_l241_241808

variable (M : Set ℕ) (N : Set ℕ)

def num_sets_satisfying_union : Prop :=
  M = {1, 2} ∧ (M ∪ N = {1, 2, 6} → (N = {6} ∨ N = {1, 6} ∨ N = {2, 6} ∨ N = {1, 2, 6}))

theorem num_sets_satisfying_union_is_four :
  (∃ M : Set ℕ, M = {1, 2}) →
  (∃ N : Set ℕ, M ∪ N = {1, 2, 6}) →
  (∃ (num_sets : ℕ), num_sets = 4) :=
by
  sorry

end num_sets_satisfying_union_is_four_l241_241808


namespace rationalize_sqrt_fraction_l241_241446

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l241_241446


namespace sequence_sqrt_l241_241615

theorem sequence_sqrt (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a n > 0)
  (h₃ : ∀ n, a (n+1 - 1) ^ 2 = a (n+1) ^ 2 + 4) :
  ∀ n, a n = Real.sqrt (4 * n - 3) :=
by
  sorry

end sequence_sqrt_l241_241615


namespace find_b_l241_241431

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_b (b : ℤ) (h : operation 11 b = 110) : b = 12 := 
by
  sorry

end find_b_l241_241431


namespace inequality_not_always_hold_l241_241340

variables {a b c d : ℝ}

theorem inequality_not_always_hold 
  (h1 : a > b) 
  (h2 : c > d) 
: ¬ (a + d > b + c) :=
  sorry

end inequality_not_always_hold_l241_241340


namespace yancheng_marathon_half_marathon_estimated_probability_l241_241538

noncomputable def estimated_probability
  (surveyed_participants_frequencies : List (ℕ × Real)) : Real :=
by
  -- Define the surveyed participants and their corresponding frequencies
  -- In this example, [(20, 0.35), (50, 0.40), (100, 0.39), (200, 0.415), (500, 0.418), (2000, 0.411)]
  sorry

theorem yancheng_marathon_half_marathon_estimated_probability :
  let surveyed_participants_frequencies := [
    (20, 0.350),
    (50, 0.400),
    (100, 0.390),
    (200, 0.415),
    (500, 0.418),
    (2000, 0.411)
  ]
  estimated_probability surveyed_participants_frequencies = 0.40 :=
by
  sorry

end yancheng_marathon_half_marathon_estimated_probability_l241_241538


namespace slope_positive_if_and_only_if_l241_241554

/-- Given points A(2, 1) and B(1, m^2), the slope of the line passing through them is positive,
if and only if m is in the range -1 < m < 1. -/
theorem slope_positive_if_and_only_if
  (m : ℝ) : 1 - m^2 > 0 ↔ -1 < m ∧ m < 1 :=
by
  sorry

end slope_positive_if_and_only_if_l241_241554


namespace probability_second_roll_twice_first_l241_241601

theorem probability_second_roll_twice_first :
  let outcomes := [(1, 2), (2, 4), (3, 6)]
  let total_outcomes := 36
  let favorable_outcomes := 3
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 :=
by
  sorry

end probability_second_roll_twice_first_l241_241601


namespace tangent_parallel_line_l241_241990

open Function

def f (x : ℝ) : ℝ := x^4 - x

def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel_line {P : ℝ × ℝ} (hP : ∃ x y, P = (x, y) ∧ f' x = 3) :
  P = (1, 0) := by
  sorry

end tangent_parallel_line_l241_241990


namespace number_of_tables_l241_241036

-- Define the conditions
def seats_per_table : ℕ := 8
def total_seating_capacity : ℕ := 32

-- Define the main statement using the conditions
theorem number_of_tables : total_seating_capacity / seats_per_table = 4 := by
  sorry

end number_of_tables_l241_241036


namespace heath_average_carrots_per_hour_l241_241616

theorem heath_average_carrots_per_hour 
  (rows1 rows2 : ℕ)
  (plants_per_row1 plants_per_row2 : ℕ)
  (hours1 hours2 : ℕ)
  (h1 : rows1 = 200)
  (h2 : rows2 = 200)
  (h3 : plants_per_row1 = 275)
  (h4 : plants_per_row2 = 325)
  (h5 : hours1 = 15)
  (h6 : hours2 = 25) :
  ((rows1 * plants_per_row1 + rows2 * plants_per_row2) / (hours1 + hours2) = 3000) :=
  by
  sorry

end heath_average_carrots_per_hour_l241_241616


namespace value_of_square_of_sum_l241_241356

theorem value_of_square_of_sum (x y: ℝ) 
(h1: 2 * x * (x + y) = 58) 
(h2: 3 * y * (x + y) = 111):
  (x + y)^2 = (169/5)^2 := by
  sorry

end value_of_square_of_sum_l241_241356


namespace simplify_polynomial_l241_241715

theorem simplify_polynomial (x y : ℝ) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 :=
by
  sorry

end simplify_polynomial_l241_241715


namespace proof_inequality_l241_241067

noncomputable def inequality (a b c : ℝ) : Prop :=
  a + 2 * b + c = 1 ∧ a^2 + b^2 + c^2 = 1 → -2/3 ≤ c ∧ c ≤ 1

theorem proof_inequality (a b c : ℝ) (h : a + 2 * b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : -2/3 ≤ c ∧ c ≤ 1 :=
by {
  sorry
}

end proof_inequality_l241_241067


namespace child_support_owed_l241_241814

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l241_241814


namespace parallelogram_height_l241_241448

theorem parallelogram_height
  (A b : ℝ)
  (h : ℝ)
  (h_area : A = 120)
  (h_base : b = 12)
  (h_formula : A = b * h) : h = 10 :=
by 
  sorry

end parallelogram_height_l241_241448


namespace odd_square_divisors_l241_241473

theorem odd_square_divisors (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (f g : ℕ), (f > g) ∧ (∀ d, d ∣ (n * n) → d % 4 = 1 ↔ (0 < f)) ∧ (∀ d, d ∣ (n * n) → d % 4 = 3 ↔ (0 < g)) :=
by
  sorry

end odd_square_divisors_l241_241473


namespace sin_value_l241_241640

theorem sin_value (alpha : ℝ) (h1 : -π / 6 < alpha ∧ alpha < π / 6)
  (h2 : Real.cos (alpha + π / 6) = 4 / 5) :
  Real.sin (2 * alpha + π / 12) = 17 * Real.sqrt 2 / 50 :=
by
    sorry

end sin_value_l241_241640


namespace hyperbola_equation_l241_241478

theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) 
  (h₃ : (2^2 / a^2) - (1^2 / b^2) = 1) (h₄ : a^2 + b^2 = 3) :
  (∀ x y : ℝ,  (x^2 / 2) - y^2 = 1) :=
by 
  sorry

end hyperbola_equation_l241_241478


namespace tetrahedron_circumscribed_sphere_radius_l241_241351

open Real

theorem tetrahedron_circumscribed_sphere_radius :
  ∀ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 5 →
    dist C D = 5 →
    dist A C = sqrt 34 →
    dist B D = sqrt 34 →
    dist A D = sqrt 41 →
    dist B C = sqrt 41 →
    ∃ (R : ℝ), R = 5 * sqrt 2 / 2 :=
by
  intros A B C D hAB hCD hAC hBD hAD hBC
  sorry

end tetrahedron_circumscribed_sphere_radius_l241_241351


namespace infinite_series_sum_l241_241255

theorem infinite_series_sum (c d : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : c > d) :
  (∑' n, 1 / (((2 * n + 1) * c - n * d) * ((2 * (n+1) - 1) * c - (n + 1 - 1) * d))) = 1 / ((c - d) * d) :=
sorry

end infinite_series_sum_l241_241255


namespace melanie_sale_revenue_correct_l241_241710

noncomputable def melanie_revenue : ℝ :=
let red_cost := 0.08
let green_cost := 0.10
let yellow_cost := 0.12
let red_gumballs := 15
let green_gumballs := 18
let yellow_gumballs := 22
let total_gumballs := red_gumballs + green_gumballs + yellow_gumballs
let total_cost := (red_cost * red_gumballs) + (green_cost * green_gumballs) + (yellow_cost * yellow_gumballs)
let discount := if total_gumballs >= 20 then 0.30 else if total_gumballs >= 10 then 0.20 else 0
let final_cost := total_cost * (1 - discount)
final_cost

theorem melanie_sale_revenue_correct : melanie_revenue = 3.95 :=
by
  -- All calculations and proofs omitted for brevity, as per instructions above
  sorry

end melanie_sale_revenue_correct_l241_241710


namespace difference_between_percentages_l241_241746

noncomputable def number : ℝ := 140

noncomputable def percentage_65 (x : ℝ) : ℝ := 0.65 * x

noncomputable def fraction_4_5 (x : ℝ) : ℝ := 0.8 * x

theorem difference_between_percentages 
  (x : ℝ) 
  (hx : x = number) 
  : (fraction_4_5 x) - (percentage_65 x) = 21 := 
by 
  sorry

end difference_between_percentages_l241_241746


namespace juan_distance_l241_241024

def running_time : ℝ := 80.0
def speed : ℝ := 10.0
def distance : ℝ := running_time * speed

theorem juan_distance :
  distance = 800.0 :=
by
  sorry

end juan_distance_l241_241024


namespace intersection_complement_l241_241218

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 2}

-- Statement to prove
theorem intersection_complement :
  (((I \ B) ∩ A : Set ℕ) = {3, 5}) :=
by
  sorry

end intersection_complement_l241_241218


namespace yola_past_weight_l241_241603

variable (W Y Y_past : ℕ)

-- Conditions
def condition1 : Prop := W = Y + 30
def condition2 : Prop := W = Y_past + 80
def condition3 : Prop := Y = 220

-- Theorem statement
theorem yola_past_weight : condition1 W Y → condition2 W Y_past → condition3 Y → Y_past = 170 :=
by
  intros h_condition1 h_condition2 h_condition3
  -- Placeholder for the proof, not required in the solution
  sorry

end yola_past_weight_l241_241603


namespace sum_of_three_squares_l241_241904

theorem sum_of_three_squares (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
  sorry

end sum_of_three_squares_l241_241904


namespace average_gas_mileage_round_trip_l241_241702

-- necessary definitions related to the problem conditions
def total_distance_one_way := 150
def fuel_efficiency_going := 35
def fuel_efficiency_return := 30
def round_trip_distance := total_distance_one_way + total_distance_one_way

-- calculation of gasoline used for each trip and total usage
def gasoline_used_going := total_distance_one_way / fuel_efficiency_going
def gasoline_used_return := total_distance_one_way / fuel_efficiency_return
def total_gasoline_used := gasoline_used_going + gasoline_used_return

-- calculation of average gas mileage
def average_gas_mileage := round_trip_distance / total_gasoline_used

-- the final theorem to prove the average gas mileage for the round trip 
theorem average_gas_mileage_round_trip : average_gas_mileage = 32 := 
by
  sorry

end average_gas_mileage_round_trip_l241_241702


namespace find_triangle_angles_l241_241797

theorem find_triangle_angles (α β γ : ℝ)
  (h1 : (180 - α) / (180 - β) = 13 / 9)
  (h2 : β - α = 45)
  (h3 : α + β + γ = 180) :
  (α = 33.75) ∧ (β = 78.75) ∧ (γ = 67.5) :=
by
  sorry

end find_triangle_angles_l241_241797


namespace tileable_contains_domino_l241_241102

theorem tileable_contains_domino {m n a b : ℕ} (h_m : m ≥ a) (h_n : n ≥ b) :
  (∀ (x : ℕ) (y : ℕ), x + a ≤ m → y + b ≤ n → ∃ (p : ℕ) (q : ℕ), p = x ∧ q = y) :=
sorry

end tileable_contains_domino_l241_241102


namespace Winnie_keeps_lollipops_l241_241550

-- Definitions based on the conditions provided
def total_lollipops : ℕ := 60 + 135 + 5 + 250
def number_of_friends : ℕ := 12

-- The theorem statement we need to prove
theorem Winnie_keeps_lollipops : total_lollipops % number_of_friends = 6 :=
by
  -- proof omitted as instructed
  sorry

end Winnie_keeps_lollipops_l241_241550


namespace jason_fish_count_ninth_day_l241_241850

def fish_growth_day1 := 8 * 3
def fish_growth_day2 := fish_growth_day1 * 3
def fish_growth_day3 := fish_growth_day2 * 3
def fish_day4_removed := 2 / 5 * fish_growth_day3
def fish_after_day4 := fish_growth_day3 - fish_day4_removed
def fish_growth_day5 := fish_after_day4 * 3
def fish_growth_day6 := fish_growth_day5 * 3
def fish_day6_removed := 3 / 7 * fish_growth_day6
def fish_after_day6 := fish_growth_day6 - fish_day6_removed
def fish_growth_day7 := fish_after_day6 * 3
def fish_growth_day8 := fish_growth_day7 * 3
def fish_growth_day9 := fish_growth_day8 * 3
def fish_final := fish_growth_day9 + 20

theorem jason_fish_count_ninth_day : fish_final = 18083 :=
by
  -- proof steps will go here
  sorry

end jason_fish_count_ninth_day_l241_241850


namespace smallest_m_plus_n_l241_241358

theorem smallest_m_plus_n : ∃ (m n : ℕ), m > 1 ∧ 
  (∃ (a b : ℝ), a = (1 : ℝ) / (m * n : ℝ) ∧ b = (m : ℝ) / (n : ℝ) ∧ b - a = (1 : ℝ) / 1007) ∧
  (∀ (k l : ℕ), k > 1 ∧ 
    (∃ (c d : ℝ), c = (1 : ℝ) / (k * l : ℝ) ∧ d = (k : ℝ) / (l : ℝ) ∧ d - c = (1 : ℝ) / 1007) → m + n ≤ k + l) ∧ 
  m + n = 19099 :=
sorry

end smallest_m_plus_n_l241_241358


namespace find_number_l241_241503

theorem find_number (x : ℝ) (h : 0.4 * x + 60 = x) : x = 100 :=
by
  sorry

end find_number_l241_241503


namespace P_2n_expression_l241_241245

noncomputable def a (n : ℕ) : ℕ :=
  2 * n + 1

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 2)

noncomputable def b (n : ℕ) : ℕ :=
  2 ^ (n - 1)

noncomputable def T (n : ℕ) : ℕ :=
  2 * b n - 1

noncomputable def c (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 / S n else a n * b n
  
noncomputable def P (n : ℕ) : ℕ :=
  if n % 2 = 0 then (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) else 0

theorem P_2n_expression (n : ℕ) : 
  P (2 * n) = (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) :=
sorry

end P_2n_expression_l241_241245


namespace find_c_l241_241642

theorem find_c (y c : ℝ) (h : y > 0) (h₂ : (8*y)/20 + (c*y)/10 = 0.7*y) : c = 6 :=
by
  sorry

end find_c_l241_241642


namespace horse_total_value_l241_241826

theorem horse_total_value (n : ℕ) (a r : ℕ) (h₁ : n = 32) (h₂ : a = 1) (h₃ : r = 2) :
  (a * (r ^ n - 1) / (r - 1)) = 4294967295 :=
by 
  rw [h₁, h₂, h₃]
  sorry

end horse_total_value_l241_241826


namespace arithmetic_sequence_sum_l241_241634

theorem arithmetic_sequence_sum (c d : ℕ) (h₁ : 3 + 5 = 8) (h₂ : 8 + 5 = 13) (h₃ : c = 13 + 5) (h₄ : d = 18 + 5) (h₅ : d + 5 = 28) : c + d = 41 :=
by
  sorry

end arithmetic_sequence_sum_l241_241634


namespace binomial_sum_eq_728_l241_241049

theorem binomial_sum_eq_728 :
  (Nat.choose 6 1) * 2^1 +
  (Nat.choose 6 2) * 2^2 +
  (Nat.choose 6 3) * 2^3 +
  (Nat.choose 6 4) * 2^4 +
  (Nat.choose 6 5) * 2^5 +
  (Nat.choose 6 6) * 2^6 = 728 :=
by
  sorry

end binomial_sum_eq_728_l241_241049


namespace total_gift_money_l241_241352

-- Definitions based on the conditions given in the problem
def initialAmount : ℕ := 159
def giftFromGrandmother : ℕ := 25
def giftFromAuntAndUncle : ℕ := 20
def giftFromParents : ℕ := 75

-- Lean statement to prove the total amount of money Chris has after receiving his birthday gifts
theorem total_gift_money : 
    initialAmount + giftFromGrandmother + giftFromAuntAndUncle + giftFromParents = 279 := by
sorry

end total_gift_money_l241_241352


namespace remainder_div_l241_241822

theorem remainder_div (n : ℕ) : (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 + 
  90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 - 90^7 * Nat.choose 10 7 + 
  90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 + 90^10 * Nat.choose 10 10) % 88 = 1 := by
  sorry

end remainder_div_l241_241822


namespace max_elements_of_valid_set_l241_241706

def valid_set (M : Finset ℤ) : Prop :=
  ∀ (a b c : ℤ), a ∈ M → b ∈ M → c ∈ M → (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (a + b ∈ M ∨ a + c ∈ M ∨ b + c ∈ M)

theorem max_elements_of_valid_set (M : Finset ℤ) (h : valid_set M) : M.card ≤ 7 :=
sorry

end max_elements_of_valid_set_l241_241706


namespace sum_of_areas_is_858_l241_241723

def length1 : ℕ := 1
def length2 : ℕ := 9
def length3 : ℕ := 25
def length4 : ℕ := 49
def length5 : ℕ := 81
def length6 : ℕ := 121

def base_width : ℕ := 3

def area (width : ℕ) (length : ℕ) : ℕ :=
  width * length

def total_area_of_rectangles : ℕ :=
  area base_width length1 +
  area base_width length2 +
  area base_width length3 +
  area base_width length4 +
  area base_width length5 +
  area base_width length6

theorem sum_of_areas_is_858 : total_area_of_rectangles = 858 := by
  sorry

end sum_of_areas_is_858_l241_241723


namespace compute_expression_l241_241271

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end compute_expression_l241_241271


namespace mod_inverse_13_997_l241_241553

-- The theorem statement
theorem mod_inverse_13_997 : ∃ x : ℕ, 0 ≤ x ∧ x < 997 ∧ (13 * x) % 997 = 1 ∧ x = 767 := 
by
  sorry

end mod_inverse_13_997_l241_241553


namespace sum_geometric_series_l241_241847

theorem sum_geometric_series (x : ℂ) (h₀ : x ≠ 1) (h₁ : x^10 - 3*x + 2 = 0) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end sum_geometric_series_l241_241847


namespace fred_final_baseball_cards_l241_241939

-- Conditions
def initial_cards : ℕ := 25
def sold_to_melanie : ℕ := 7
def traded_with_kevin : ℕ := 3
def bought_from_alex : ℕ := 5

-- Proof statement (Lean theorem)
theorem fred_final_baseball_cards : initial_cards - sold_to_melanie - traded_with_kevin + bought_from_alex = 20 := by
  sorry

end fred_final_baseball_cards_l241_241939


namespace simplify_and_evaluate_l241_241772

-- Math proof problem
theorem simplify_and_evaluate :
  ∀ (a : ℤ), a = -1 →
  (2 - a)^2 - (1 + a) * (a - 1) - a * (a - 3) = 5 :=
by
  intros a ha
  sorry

end simplify_and_evaluate_l241_241772


namespace triangle_problem_l241_241902

open Real

theorem triangle_problem (a b S : ℝ) (A B : ℝ) (hA_cos : cos A = (sqrt 6) / 3) (hA_val : a = 3) (hB_val : B = A + π / 2):
  b = 3 * sqrt 2 ∧
  S = (3 * sqrt 2) / 2 :=
by
  sorry

end triangle_problem_l241_241902


namespace digits_making_number_divisible_by_4_l241_241113

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l241_241113


namespace polygon_divided_l241_241452

theorem polygon_divided (p q r : ℕ) : p - q + r = 1 :=
sorry

end polygon_divided_l241_241452


namespace det_calculation_l241_241754

-- Given conditions
variables (p q r s : ℤ)
variable (h1 : p * s - q * r = -3)

-- Define the matrix and determinant
def matrix_determinant (a b c d : ℤ) := a * d - b * c

-- Problem statement
theorem det_calculation : matrix_determinant (p + 2 * r) (q + 2 * s) r s = -3 :=
by
  -- Proof goes here
  sorry

end det_calculation_l241_241754


namespace derivative_at_1_of_f_l241_241044

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1_of_f :
  (deriv f 1) = 2 * Real.log 2 - 3 :=
sorry

end derivative_at_1_of_f_l241_241044


namespace liquid_X_percentage_in_new_solution_l241_241182

noncomputable def solutionY_initial_kg : ℝ := 10
noncomputable def percentage_liquid_X : ℝ := 0.30
noncomputable def evaporated_water_kg : ℝ := 2
noncomputable def added_solutionY_kg : ℝ := 2

-- Calculate the amount of liquid X in the original solution
noncomputable def initial_liquid_X_kg : ℝ :=
  percentage_liquid_X * solutionY_initial_kg

-- Calculate the remaining weight after evaporation
noncomputable def remaining_weight_kg : ℝ :=
  solutionY_initial_kg - evaporated_water_kg

-- Calculate the amount of liquid X after evaporation
noncomputable def remaining_liquid_X_kg : ℝ := initial_liquid_X_kg

-- Since only water evaporates, remaining water weight
noncomputable def remaining_water_kg : ℝ :=
  remaining_weight_kg - remaining_liquid_X_kg

-- Calculate the amount of liquid X in the added solution
noncomputable def added_liquid_X_kg : ℝ :=
  percentage_liquid_X * added_solutionY_kg

-- Total liquid X in the new solution
noncomputable def new_liquid_X_kg : ℝ :=
  remaining_liquid_X_kg + added_liquid_X_kg

-- Calculate the water in the added solution
noncomputable def percentage_water : ℝ := 0.70
noncomputable def added_water_kg : ℝ :=
  percentage_water * added_solutionY_kg

-- Total water in the new solution
noncomputable def new_water_kg : ℝ :=
  remaining_water_kg + added_water_kg

-- Total weight of the new solution
noncomputable def new_total_weight_kg : ℝ :=
  remaining_weight_kg + added_solutionY_kg

-- Percentage of liquid X in the new solution
noncomputable def percentage_new_liquid_X : ℝ :=
  (new_liquid_X_kg / new_total_weight_kg) * 100

-- The proof statement
theorem liquid_X_percentage_in_new_solution :
  percentage_new_liquid_X = 36 :=
by
  sorry

end liquid_X_percentage_in_new_solution_l241_241182


namespace cosine_squared_identity_l241_241824

theorem cosine_squared_identity (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) : Real.cos (α - (π / 4)) ^ 2 = 2 / 3 :=
sorry

end cosine_squared_identity_l241_241824


namespace coordinates_of_vertex_B_equation_of_line_BC_l241_241201

noncomputable def vertex_A : (ℝ × ℝ) := (5, 1)
def bisector_expr (x y : ℝ) : Prop := x + y - 5 = 0
def median_CM_expr (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem coordinates_of_vertex_B (B : ℝ × ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  B = (2, 3) :=
sorry

theorem equation_of_line_BC (coeff_3x coeff_2y const : ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  coeff_3x = 3 ∧ coeff_2y = 2 ∧ const = -12 :=
sorry

end coordinates_of_vertex_B_equation_of_line_BC_l241_241201


namespace systematic_sampling_interval_people_l241_241467

theorem systematic_sampling_interval_people (total_employees : ℕ) (selected_employees : ℕ) (start_interval : ℕ) (end_interval : ℕ)
  (h_total : total_employees = 420)
  (h_selected : selected_employees = 21)
  (h_start_end : start_interval = 281)
  (h_end : end_interval = 420)
  : (end_interval - start_interval + 1) / (total_employees / selected_employees) = 7 := 
by
  -- sorry placeholder for proof
  sorry

end systematic_sampling_interval_people_l241_241467


namespace number_of_terms_in_expansion_l241_241807

def first_factor : List Char := ['x', 'y']
def second_factor : List Char := ['u', 'v', 'w', 'z', 's']

theorem number_of_terms_in_expansion :
  first_factor.length * second_factor.length = 10 :=
by
  -- Lean expects a proof here, but the problem statement specifies to use sorry to skip the proof.
  sorry

end number_of_terms_in_expansion_l241_241807


namespace Tanya_accompanied_two_l241_241810

-- Define the number of songs sung by each girl
def Anya_songs : ℕ := 8
def Tanya_songs : ℕ := 6
def Olya_songs : ℕ := 3
def Katya_songs : ℕ := 7

-- Assume each song is sung by three girls
def total_songs : ℕ := (Anya_songs + Tanya_songs + Olya_songs + Katya_songs) / 3

-- Define the number of times Tanya accompanied
def Tanya_accompanied : ℕ := total_songs - Tanya_songs

-- Prove that Tanya accompanied 2 times
theorem Tanya_accompanied_two : Tanya_accompanied = 2 :=
by sorry

end Tanya_accompanied_two_l241_241810


namespace greater_number_is_84_l241_241479

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : x + y - (x - y) = 64) :
  x = 84 :=
by sorry

end greater_number_is_84_l241_241479


namespace estimated_fish_in_pond_l241_241032

theorem estimated_fish_in_pond :
  ∀ (number_marked_first_catch total_second_catch number_marked_second_catch : ℕ),
    number_marked_first_catch = 100 →
    total_second_catch = 108 →
    number_marked_second_catch = 9 →
    ∃ est_total_fish : ℕ, (number_marked_second_catch / total_second_catch : ℝ) = (number_marked_first_catch / est_total_fish : ℝ) ∧ est_total_fish = 1200 := 
by
  intros number_marked_first_catch total_second_catch number_marked_second_catch
  sorry

end estimated_fish_in_pond_l241_241032


namespace grilled_cheese_sandwiches_l241_241312

theorem grilled_cheese_sandwiches (h g : ℕ) (c_ham c_grilled total_cheese : ℕ)
  (h_count : h = 10)
  (ham_cheese : c_ham = 2)
  (grilled_cheese : c_grilled = 3)
  (cheese_used : total_cheese = 50)
  (sandwich_eq : total_cheese = h * c_ham + g * c_grilled) :
  g = 10 :=
by
  sorry

end grilled_cheese_sandwiches_l241_241312


namespace circle_radius_l241_241925

theorem circle_radius (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 10) : r = 20 :=
by
  sorry

end circle_radius_l241_241925


namespace range_of_a_l241_241651

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ a ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by
  sorry

end range_of_a_l241_241651


namespace sector_area_120_deg_radius_3_l241_241680

theorem sector_area_120_deg_radius_3 (r : ℝ) (theta_deg : ℝ) (theta_rad : ℝ) (A : ℝ)
  (h1 : r = 3)
  (h2 : theta_deg = 120)
  (h3 : theta_rad = (2 * Real.pi / 3))
  (h4 : A = (1 / 2) * theta_rad * r^2) :
  A = 3 * Real.pi :=
  sorry

end sector_area_120_deg_radius_3_l241_241680


namespace equilateral_triangle_sum_perimeters_l241_241866

theorem equilateral_triangle_sum_perimeters (s : ℝ) (h : ∑' n, 3 * s / 2 ^ n = 360) : 
  s = 60 := 
by 
  sorry

end equilateral_triangle_sum_perimeters_l241_241866


namespace stools_chopped_up_l241_241432

variable (chairs tables stools : ℕ)
variable (sticks_per_chair sticks_per_table sticks_per_stool : ℕ)
variable (sticks_per_hour hours total_sticks_from_chairs tables_sticks required_sticks : ℕ)

theorem stools_chopped_up (h1 : sticks_per_chair = 6)
                         (h2 : sticks_per_table = 9)
                         (h3 : sticks_per_stool = 2)
                         (h4 : sticks_per_hour = 5)
                         (h5 : chairs = 18)
                         (h6 : tables = 6)
                         (h7 : hours = 34)
                         (h8 : total_sticks_from_chairs = chairs * sticks_per_chair)
                         (h9 : tables_sticks = tables * sticks_per_table)
                         (h10 : required_sticks = hours * sticks_per_hour)
                         (h11 : total_sticks_from_chairs + tables_sticks = 162) :
                         stools = 4 := by
  sorry

end stools_chopped_up_l241_241432


namespace division_multiplication_eval_l241_241548

theorem division_multiplication_eval : (18 / (5 + 2 - 3)) * 4 = 18 := 
by
  sorry

end division_multiplication_eval_l241_241548


namespace time_for_C_to_complete_work_l241_241934

variable (A B C : ℕ) (R : ℚ)

def work_completion_in_days (days : ℕ) (portion : ℚ) :=
  portion = 1 / days

theorem time_for_C_to_complete_work :
  work_completion_in_days A 8 →
  work_completion_in_days B 12 →
  work_completion_in_days (A + B + C) 4 →
  C = 24 :=
by
  sorry

end time_for_C_to_complete_work_l241_241934


namespace correct_operation_l241_241869

variables (a b : ℝ)

theorem correct_operation : 5 * a * b - 3 * a * b = 2 * a * b :=
by sorry

end correct_operation_l241_241869


namespace num_triangles_with_perimeter_9_l241_241496

-- Definitions for side lengths and their conditions
def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 9

-- The main theorem
theorem num_triangles_with_perimeter_9 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a ≤ b ∧ b ≤ c → is_triangle a b c ↔ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end num_triangles_with_perimeter_9_l241_241496


namespace roof_area_l241_241303

-- Definitions of the roof's dimensions based on the given conditions.
def length (w : ℝ) := 4 * w
def width (w : ℝ) := w
def difference (l w : ℝ) := l - w
def area (l w : ℝ) := l * w

-- The proof problem: Given the conditions, prove the area is 576 square feet.
theorem roof_area : ∀ w : ℝ, (length w) - (width w) = 36 → area (length w) (width w) = 576 := by
  intro w
  intro h_diff
  sorry

end roof_area_l241_241303


namespace diamond_comm_not_assoc_l241_241157

def diamond (a b : ℤ) : ℤ := (a * b + 5) / (a + b)

-- Lemma: Verify commutativity of the diamond operation
lemma diamond_comm (a b : ℤ) (ha : a > 1) (hb : b > 1) : 
  diamond a b = diamond b a := by
  sorry

-- Lemma: Verify non-associativity of the diamond operation
lemma diamond_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  sorry

-- Theorem: The diamond operation is commutative but not associative
theorem diamond_comm_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond a b = diamond b a ∧ diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  apply And.intro
  · apply diamond_comm
    apply ha
    apply hb
  · apply diamond_not_assoc
    apply ha
    apply hb
    apply hc

end diamond_comm_not_assoc_l241_241157


namespace relationship_between_mode_median_mean_l241_241079

def data_set : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode : ℕ := 60 -- derived from the problem conditions
def median : ℕ := 50 -- derived from the problem conditions
def mean : ℚ := 330 / 7 -- derived from the problem conditions

theorem relationship_between_mode_median_mean :
  mode > median ∧ median > mean :=
by
  sorry

end relationship_between_mode_median_mean_l241_241079


namespace zoo_with_hippos_only_l241_241248

variables {Z : Type} -- The type of all zoos
variables (H R G : Set Z) -- Subsets of zoos with hippos, rhinos, and giraffes respectively

-- Conditions
def condition1 : Prop := ∀ (z : Z), z ∈ H ∧ z ∈ R → z ∉ G
def condition2 : Prop := ∀ (z : Z), z ∈ R ∧ z ∉ G → z ∈ H
def condition3 : Prop := ∀ (z : Z), z ∈ H ∧ z ∈ G → z ∈ R

-- Goal
def goal : Prop := ∃ (z : Z), z ∈ H ∧ z ∉ G ∧ z ∉ R

-- Theorem statement
theorem zoo_with_hippos_only (h1 : condition1 H R G) (h2 : condition2 H R G) (h3 : condition3 H R G) : goal H R G :=
sorry

end zoo_with_hippos_only_l241_241248


namespace common_points_circle_ellipse_l241_241857

theorem common_points_circle_ellipse :
    (∃ (p1 p2: ℝ × ℝ),
        p1 ≠ p2 ∧
        (p1, p2).fst.1 ^ 2 + (p1, p2).fst.2 ^ 2 = 4 ∧
        9 * (p1, p2).fst.1 ^ 2 + 4 * (p1, p2).fst.2 ^ 2 = 36 ∧
        (p1, p2).snd.1 ^ 2 + (p1, p2).snd.2 ^ 2 = 4 ∧
        9 * (p1, p2).snd.1 ^ 2 + 4 * (p1, p2).snd.2 ^ 2 = 36) :=
sorry

end common_points_circle_ellipse_l241_241857


namespace Doug_age_l241_241658

theorem Doug_age
  (B : ℕ) (D : ℕ) (N : ℕ)
  (h1 : 2 * B = N)
  (h2 : B + D = 90)
  (h3 : 20 * N = 2000) : 
  D = 40 := sorry

end Doug_age_l241_241658


namespace suzie_store_revenue_l241_241150

theorem suzie_store_revenue 
  (S B : ℝ) 
  (h1 : B = S + 15) 
  (h2 : 22 * S + 16 * B = 460) : 
  8 * S + 32 * B = 711.60 :=
by
  sorry

end suzie_store_revenue_l241_241150


namespace geometric_sequence_a4_l241_241780

theorem geometric_sequence_a4 {a_2 a_6 a_4 : ℝ} 
  (h1 : ∃ a_1 r : ℝ, a_2 = a_1 * r ∧ a_6 = a_1 * r^5) 
  (h2 : a_2 * a_6 = 64) 
  (h3 : a_2 = a_1 * r)
  (h4 : a_6 = a_1 * r^5)
  : a_4 = 8 :=
by
  sorry

end geometric_sequence_a4_l241_241780


namespace next_term_geometric_sequence_l241_241530

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l241_241530


namespace rocket_coaster_total_cars_l241_241085

theorem rocket_coaster_total_cars (C_4 C_6 : ℕ) (h1 : C_4 = 9) (h2 : 4 * C_4 + 6 * C_6 = 72) :
  C_4 + C_6 = 15 :=
sorry

end rocket_coaster_total_cars_l241_241085


namespace salon_customers_l241_241599

theorem salon_customers (C : ℕ) (H : C * 2 + 5 = 33) : C = 14 :=
by {
  sorry
}

end salon_customers_l241_241599


namespace min_value_of_c_l241_241237

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m^2

noncomputable def isPerfectCube (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^3

theorem min_value_of_c (c : ℕ) :
  (∃ a b d e : ℕ, a = c-2 ∧ b = c-1 ∧ d = c+1 ∧ e = c+2 ∧ a < b ∧ b < c ∧ c < d ∧ d < e) ∧
  isPerfectSquare (3 * c) ∧
  isPerfectCube (5 * c) →
  c = 675 :=
sorry

end min_value_of_c_l241_241237


namespace mrs_generous_jelly_beans_l241_241928

-- Define necessary terms and state the problem
def total_children (x : ℤ) : ℤ := x + (x + 3)

theorem mrs_generous_jelly_beans :
  ∃ x : ℤ, x^2 + (x + 3)^2 = 490 ∧ total_children x = 31 :=
by {
  sorry
}

end mrs_generous_jelly_beans_l241_241928


namespace probability_at_least_four_8s_in_five_rolls_l241_241784

-- Definitions 
def prob_three_favorable : ℚ := 3 / 10

def prob_at_least_four_times_in_five_rolls : ℚ := 5 * (prob_three_favorable^4) * ((7 : ℚ)/10) + (prob_three_favorable)^5

-- The proof statement
theorem probability_at_least_four_8s_in_five_rolls : prob_at_least_four_times_in_five_rolls = 2859.3 / 10000 :=
by
  sorry

end probability_at_least_four_8s_in_five_rolls_l241_241784


namespace problem_statement_l241_241575

noncomputable def f (x : ℝ) := Real.log 9 * (Real.log x / Real.log 3)

theorem problem_statement : deriv f 2 + deriv f 2 = 1 := sorry

end problem_statement_l241_241575


namespace probability_of_consecutive_blocks_drawn_l241_241476

theorem probability_of_consecutive_blocks_drawn :
  let total_ways := (Nat.factorial 12)
  let favorable_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5) * (Nat.factorial 3)
  (favorable_ways / total_ways) = 1 / 4620 :=
by
  sorry

end probability_of_consecutive_blocks_drawn_l241_241476


namespace investment_amount_l241_241027

noncomputable def PV (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

theorem investment_amount (FV : ℝ) (r : ℝ) (n : ℕ) (PV : ℝ) : FV = 1000000 ∧ r = 0.08 ∧ n = 20 → PV = 1000000 / (1 + 0.08)^20 :=
by
  intros
  sorry

end investment_amount_l241_241027


namespace total_turnips_l241_241105

theorem total_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := 
by sorry

end total_turnips_l241_241105


namespace min_total_bags_l241_241290

theorem min_total_bags (x y : ℕ) (h : 15 * x + 8 * y = 1998) (hy_min : ∀ y', (15 * x + 8 * y' = 1998) → y ≤ y') :
  x + y = 140 :=
by
  sorry

end min_total_bags_l241_241290


namespace probability_median_five_l241_241147

theorem probability_median_five {S : Finset ℕ} (hS : S = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let n := 8
  let k := 5
  let total_ways := Nat.choose n k
  let ways_median_5 := Nat.choose 4 2 * Nat.choose 3 2
  (ways_median_5 : ℚ) / (total_ways : ℚ) = (9 : ℚ) / (28 : ℚ) :=
by
  sorry

end probability_median_five_l241_241147


namespace brian_books_chapters_l241_241580

variable (x : ℕ)

theorem brian_books_chapters (h1 : 1 ≤ x) (h2 : 20 + 2 * x + (20 + 2 * x) / 2 = 75) : x = 15 :=
sorry

end brian_books_chapters_l241_241580


namespace condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l241_241735

-- Definitions corresponding to each condition
def numMethods_participates_in_one_event (students events : ℕ) : ℕ :=
  events ^ students

def numMethods_event_limit_one_person (students events : ℕ) : ℕ :=
  students * (students - 1) * (students - 2)

def numMethods_person_limit_in_events (students events : ℕ) : ℕ :=
  students ^ events

-- Theorems to be proved
theorem condition1_num_registration_methods : 
  numMethods_participates_in_one_event 6 3 = 729 :=
by
  sorry

theorem condition2_num_registration_methods : 
  numMethods_event_limit_one_person 6 3 = 120 :=
by
  sorry

theorem condition3_num_registration_methods : 
  numMethods_person_limit_in_events 6 3 = 216 :=
by
  sorry

end condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l241_241735


namespace time_for_plastic_foam_drift_l241_241650

def boat_speed_in_still_water : ℝ := sorry
def speed_of_water_flow : ℝ := sorry
def distance_between_docks : ℝ := sorry

theorem time_for_plastic_foam_drift (x y s t : ℝ) 
(hx : 6 * (x + y) = s)
(hy : 8 * (x - y) = s)
(t_eq : t = s / y) : 
t = 48 := 
sorry

end time_for_plastic_foam_drift_l241_241650


namespace sum_of_factors_of_1000_l241_241293

-- Define what it means for an integer to not contain the digit '0'
def no_zero_digits (n : ℕ) : Prop :=
∀ c ∈ (n.digits 10), c ≠ 0

-- Define the problem statement
theorem sum_of_factors_of_1000 :
  ∃ (a b : ℕ), a * b = 1000 ∧ no_zero_digits a ∧ no_zero_digits b ∧ (a + b = 133) :=
sorry

end sum_of_factors_of_1000_l241_241293


namespace parabola_properties_l241_241329

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c : ℝ) (h₀ : a ≠ 0)
    (h₁ : parabola a b c (-1) = -1)
    (h₂ : parabola a b c 0 = 1)
    (h₃ : parabola a b c (-2) > 1) :
    (a * b * c > 0) ∧
    (∃ Δ : ℝ, Δ > 0 ∧ (Δ = b^2 - 4*a*c)) ∧
    (a + b + c > 7) :=
sorry

end parabola_properties_l241_241329


namespace contrapositive_statement_l241_241626

theorem contrapositive_statement :
  (∀ n : ℕ, (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0) →
  (∀ n : ℕ, n % 10 ≠ 0 → ¬(n % 2 = 0 ∧ n % 5 = 0)) :=
by
  sorry

end contrapositive_statement_l241_241626


namespace value_calculation_l241_241756

theorem value_calculation :
  6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 :=
by
  sorry

end value_calculation_l241_241756


namespace product_not_power_of_two_l241_241116

theorem product_not_power_of_two (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, (36 * a + b) * (a + 36 * b) ≠ 2^k :=
by
  sorry

end product_not_power_of_two_l241_241116


namespace positive_divisors_60_l241_241176

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l241_241176


namespace solve_eq1_solve_eq2_solve_eq3_l241_241451

theorem solve_eq1 (x : ℝ) : 5 * x - 2.9 = 12 → x = 1.82 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq2 (x : ℝ) : 10.5 * x + 0.6 * x = 44 → x = 3 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq3 (x : ℝ) : 8 * x / 2 = 1.5 → x = 0.375 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

end solve_eq1_solve_eq2_solve_eq3_l241_241451


namespace sin_cos_identity_l241_241624

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) + Real.cos (20 * Real.pi / 180) * Real.sin (140 * Real.pi / 180)) =
  (Real.sqrt 3 / 2) := by
  sorry

end sin_cos_identity_l241_241624


namespace window_width_is_28_l241_241017

noncomputable def window_width (y : ℝ) : ℝ :=
  12 * y + 4

theorem window_width_is_28 : ∃ (y : ℝ), window_width y = 28 :=
by
  -- The proof goes here
  sorry

end window_width_is_28_l241_241017


namespace joan_gemstone_samples_l241_241747

theorem joan_gemstone_samples
  (minerals_yesterday : ℕ)
  (gemstones : ℕ)
  (h1 : minerals_yesterday + 6 = 48)
  (h2 : gemstones = minerals_yesterday / 2) :
  gemstones = 21 :=
by
  sorry

end joan_gemstone_samples_l241_241747


namespace pages_allocation_correct_l241_241568

-- Define times per page for Alice, Bob, and Chandra
def t_A := 40
def t_B := 60
def t_C := 48

-- Define pages read by Alice, Bob, and Chandra
def pages_A := 295
def pages_B := 197
def pages_C := 420

-- Total pages in the novel
def total_pages := 912

-- Calculate the total time each one spends reading
def total_time_A := t_A * pages_A
def total_time_B := t_B * pages_B
def total_time_C := t_C * pages_C

-- Theorem: Prove the correct allocation of pages
theorem pages_allocation_correct : 
  total_pages = pages_A + pages_B + pages_C ∧
  total_time_A = total_time_B ∧
  total_time_B = total_time_C :=
by 
  -- Place end of proof here 
  sorry

end pages_allocation_correct_l241_241568


namespace minimum_degree_q_l241_241056

variable (p q r : Polynomial ℝ)

theorem minimum_degree_q (h1 : 2 * p + 5 * q = r)
                        (hp : p.degree = 7)
                        (hr : r.degree = 10) :
  q.degree = 10 :=
sorry

end minimum_degree_q_l241_241056


namespace square_binomial_formula_l241_241523

variable {x y : ℝ}

theorem square_binomial_formula :
  (2 * x + y) * (y - 2 * x) = y^2 - 4 * x^2 := 
  sorry

end square_binomial_formula_l241_241523


namespace frank_hamburger_goal_l241_241592

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end frank_hamburger_goal_l241_241592


namespace percentage_of_non_defective_products_l241_241593

-- Define the conditions
def totalProduction : ℕ := 100
def M1_production : ℕ := 25
def M2_production : ℕ := 35
def M3_production : ℕ := 40

def M1_defective_rate : ℝ := 0.02
def M2_defective_rate : ℝ := 0.04
def M3_defective_rate : ℝ := 0.05

-- Calculate the total defective units
noncomputable def total_defective_units : ℝ := 
  (M1_defective_rate * M1_production) + 
  (M2_defective_rate * M2_production) + 
  (M3_defective_rate * M3_production)

-- Calculate the percentage of defective products
noncomputable def defective_percentage : ℝ := (total_defective_units / totalProduction) * 100

-- Calculate the percentage of non-defective products
noncomputable def non_defective_percentage : ℝ := 100 - defective_percentage

-- The statement to prove
theorem percentage_of_non_defective_products :
  non_defective_percentage = 96.1 :=
by
  sorry

end percentage_of_non_defective_products_l241_241593


namespace man_walking_speed_l241_241670

theorem man_walking_speed (length_of_bridge : ℝ) (time_to_cross : ℝ) 
  (h1 : length_of_bridge = 1250) (h2 : time_to_cross = 15) : 
  (length_of_bridge / time_to_cross) * (60 / 1000) = 5 := 
sorry

end man_walking_speed_l241_241670


namespace stratified_sampling_probability_l241_241573

open Finset Nat

noncomputable def combin (n k : ℕ) : ℕ := choose n k

theorem stratified_sampling_probability :
  let total_balls := 40
  let red_balls := 16
  let blue_balls := 12
  let white_balls := 8
  let yellow_balls := 4
  let n_draw := 10
  let red_draw := 4
  let blue_draw := 3
  let white_draw := 2
  let yellow_draw := 1
  
  combin yellow_balls yellow_draw * combin white_balls white_draw * combin blue_balls blue_draw * combin red_balls red_draw = combin total_balls n_draw :=
sorry

end stratified_sampling_probability_l241_241573


namespace wall_length_proof_l241_241410

-- Define the initial conditions
def men1 : ℕ := 20
def days1 : ℕ := 8
def men2 : ℕ := 86
def days2 : ℕ := 8
def wall_length2 : ℝ := 283.8

-- Define the expected length of the wall for the first condition
def expected_length : ℝ := 65.7

-- The proof statement.
theorem wall_length_proof : ((men1 * days1) / (men2 * days2)) * wall_length2 = expected_length :=
sorry

end wall_length_proof_l241_241410


namespace find_r_in_geometric_series_l241_241138

theorem find_r_in_geometric_series
  (a r : ℝ)
  (h1 : a / (1 - r) = 15)
  (h2 : a / (1 - r^2) = 6) :
  r = 2 / 3 :=
sorry

end find_r_in_geometric_series_l241_241138


namespace convert_degrees_to_radians_l241_241256

theorem convert_degrees_to_radians : 
  (-390) * (Real.pi / 180) = - (13 * Real.pi / 6) := 
by 
  sorry

end convert_degrees_to_radians_l241_241256


namespace shirts_sold_correct_l241_241833

-- Define the conditions
def shoes_sold := 6
def cost_per_shoe := 3
def earnings_per_person := 27
def total_earnings := 2 * earnings_per_person
def earnings_from_shoes := shoes_sold * cost_per_shoe
def cost_per_shirt := 2
def earnings_from_shirts := total_earnings - earnings_from_shoes

-- Define the total number of shirts sold and the target value to prove
def shirts_sold : Nat := earnings_from_shirts / cost_per_shirt

-- Prove that shirts_sold is 18
theorem shirts_sold_correct : shirts_sold = 18 := by
  sorry

end shirts_sold_correct_l241_241833


namespace highest_power_of_3_divides_N_l241_241098

-- Define the range of two-digit numbers and the concatenation function
def concatTwoDigitIntegers : ℕ := sorry  -- Placeholder for the concatenation implementation

-- Integer N formed by concatenating integers from 31 to 68
def N := concatTwoDigitIntegers

-- The statement proving the highest power of 3 dividing N is 3^1
theorem highest_power_of_3_divides_N :
  (∃ k : ℕ, 3^k ∣ N ∧ ¬ 3^(k+1) ∣ N) ∧ 3^1 ∣ N ∧ ¬ 3^2 ∣ N :=
by
  sorry  -- Placeholder for the proof

end highest_power_of_3_divides_N_l241_241098


namespace people_in_group_10_l241_241519

-- Let n represent the number of people in the group.
def number_of_people_in_group (n : ℕ) : Prop :=
  let average_increase : ℚ := 3.2
  let weight_of_replaced_person : ℚ := 65
  let weight_of_new_person : ℚ := 97
  let weight_increase : ℚ := weight_of_new_person - weight_of_replaced_person
  weight_increase = average_increase * n

theorem people_in_group_10 :
  ∃ n : ℕ, number_of_people_in_group n ∧ n = 10 :=
by
  sorry

end people_in_group_10_l241_241519


namespace max_dominoes_l241_241084

theorem max_dominoes (m n : ℕ) (h : n ≥ m) :
  ∃ k, k = m * n - (m / 2 : ℕ) :=
by sorry

end max_dominoes_l241_241084


namespace find_BC_distance_l241_241610

-- Definitions of constants as per problem conditions
def ACB_angle : ℝ := 120
def AC_distance : ℝ := 2
def AB_distance : ℝ := 3

-- The theorem to prove the distance BC
theorem find_BC_distance (BC : ℝ) (h : AC_distance * AC_distance + (BC * BC) - 2 * AC_distance * BC * Real.cos (ACB_angle * Real.pi / 180) = AB_distance * AB_distance) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end find_BC_distance_l241_241610


namespace mixed_gender_appointment_schemes_l241_241633

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 
  else n * factorial (n - 1)

noncomputable def P (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

theorem mixed_gender_appointment_schemes : 
  let total_students := 9
  let total_permutations := P total_students 3
  let male_students := 5
  let female_students := 4
  let male_permutations := P male_students 3
  let female_permutations := P female_students 3
  total_permutations - (male_permutations + female_permutations) = 420 :=
by 
  sorry

end mixed_gender_appointment_schemes_l241_241633


namespace equation_holds_if_a_eq_neg_b_c_l241_241045

-- Define the conditions and equation
variables {a b c : ℝ} (h1 : a ≠ 0) (h2 : a + b ≠ 0)

-- Statement to be proved
theorem equation_holds_if_a_eq_neg_b_c : 
  (a = -(b + c)) ↔ (a + b + c) / a = (b + c) / (a + b) := 
sorry

end equation_holds_if_a_eq_neg_b_c_l241_241045


namespace average_price_of_racket_l241_241155

theorem average_price_of_racket
  (total_amount_made : ℝ)
  (number_of_pairs_sold : ℕ)
  (h1 : total_amount_made = 490) 
  (h2 : number_of_pairs_sold = 50) : 
  (total_amount_made / number_of_pairs_sold : ℝ) = 9.80 := 
  by
  sorry

end average_price_of_racket_l241_241155


namespace biquadratic_exactly_two_distinct_roots_l241_241300

theorem biquadratic_exactly_two_distinct_roots {a : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^4 + a*x1^2 + a - 1 = 0) ∧ (x2^4 + a*x2^2 + a - 1 = 0) ∧
   ∀ x, x^4 + a*x^2 + a - 1 = 0 → (x = x1 ∨ x = x2)) ↔ a < 1 :=
by
  sorry

end biquadratic_exactly_two_distinct_roots_l241_241300


namespace markup_calculation_l241_241905

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.25
def net_profit : ℝ := 12

def overhead := purchase_price * overhead_percentage
def total_cost := purchase_price + overhead
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_calculation : markup = 24 := by
  sorry

end markup_calculation_l241_241905


namespace compound_interest_rate_l241_241678

theorem compound_interest_rate
  (P : ℝ) (t : ℕ) (A : ℝ) (interest : ℝ)
  (hP : P = 6000)
  (ht : t = 2)
  (hA : A = 7260)
  (hInterest : interest = 1260.000000000001)
  (hA_eq : A = P + interest) :
  ∃ r : ℝ, (1 + r)^(t : ℝ) = A / P ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l241_241678


namespace pentagon_PTRSQ_area_proof_l241_241048

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end pentagon_PTRSQ_area_proof_l241_241048


namespace compute_expression_l241_241127

theorem compute_expression : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end compute_expression_l241_241127


namespace recruit_people_l241_241408

variable (average_contribution : ℝ) (total_funds_needed : ℝ) (current_funds : ℝ)

theorem recruit_people (h₁ : average_contribution = 10) (h₂ : current_funds = 200) (h₃ : total_funds_needed = 1000) : 
    (total_funds_needed - current_funds) / average_contribution = 80 := by
  sorry

end recruit_people_l241_241408


namespace remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l241_241707

theorem remainder_7_mul_12_pow_24_add_2_pow_24_mod_13 :
  (7 * 12^24 + 2^24) % 13 = 8 := by
  sorry

end remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l241_241707


namespace jason_current_cards_l241_241595

-- Define the initial number of Pokemon cards Jason had.
def initial_cards : ℕ := 9

-- Define the number of Pokemon cards Jason gave to his friends.
def given_away : ℕ := 4

-- Prove that the number of Pokemon cards he has now is 5.
theorem jason_current_cards : initial_cards - given_away = 5 := by
  sorry

end jason_current_cards_l241_241595


namespace find_prime_p_l241_241935

def f (x : ℕ) : ℕ :=
  (x^4 + 2 * x^3 + 4 * x^2 + 2 * x + 1)^5

theorem find_prime_p : ∃! p, Nat.Prime p ∧ f p = 418195493 := by
  sorry

end find_prime_p_l241_241935


namespace baseball_opponents_score_l241_241382

theorem baseball_opponents_score 
  (team_scores : List ℕ)
  (team_lost_scores : List ℕ)
  (team_won_scores : List ℕ)
  (opponent_lost_scores : List ℕ)
  (opponent_won_scores : List ℕ)
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : team_lost_scores = [1, 3, 5, 7, 9, 11])
  (h3 : team_won_scores = [6, 9, 12])
  (h4 : opponent_lost_scores = [3, 5, 7, 9, 11, 13])
  (h5 : opponent_won_scores = [2, 3, 4]) :
  (List.sum opponent_lost_scores + List.sum opponent_won_scores = 57) :=
sorry

end baseball_opponents_score_l241_241382


namespace total_selling_price_l241_241278

theorem total_selling_price (CP : ℕ) (num_toys : ℕ) (gain_toys : ℕ) (TSP : ℕ)
  (h1 : CP = 1300)
  (h2 : num_toys = 18)
  (h3 : gain_toys = 3) :
  TSP = 27300 := by
  sorry

end total_selling_price_l241_241278


namespace apple_cost_price_l241_241656

theorem apple_cost_price (SP : ℝ) (loss_frac : ℝ) (CP : ℝ) (h_SP : SP = 19) (h_loss_frac : loss_frac = 1 / 6) (h_loss : SP = CP - loss_frac * CP) : CP = 22.8 :=
by
  sorry

end apple_cost_price_l241_241656


namespace total_distance_covered_l241_241143

noncomputable def radius : ℝ := 0.242
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def number_of_revolutions : ℕ := 500
noncomputable def total_distance : ℝ := circumference * number_of_revolutions

theorem total_distance_covered :
  total_distance = 760 :=
by
  -- sorry Re-enable this line for the solver to automatically skip the proof 
  sorry

end total_distance_covered_l241_241143


namespace distance_point_to_line_zero_or_four_l241_241164

theorem distance_point_to_line_zero_or_four {b : ℝ} 
(h : abs (b - 2) / Real.sqrt 2 = Real.sqrt 2) : 
b = 0 ∨ b = 4 := 
sorry

end distance_point_to_line_zero_or_four_l241_241164


namespace track_length_l241_241123

theorem track_length (V_A V_B V_C : ℝ) (x : ℝ) 
  (h1 : x / V_A = (x - 1) / V_B) 
  (h2 : x / V_A = (x - 2) / V_C) 
  (h3 : x / V_B = (x - 1.01) / V_C) : 
  110 - x = 9 :=
by 
  sorry

end track_length_l241_241123


namespace trig_identity_l241_241828

theorem trig_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (α - 15 * Real.pi / 180) + Real.cos (105 * Real.pi / 180 - α) = -2 / 3 :=
sorry

end trig_identity_l241_241828


namespace total_questions_l241_241295

theorem total_questions (f s k : ℕ) (hf : f = 36) (hs : s = 2 * f) (hk : k = (f + s) / 2) :
  2 * (f + s + k) = 324 :=
by {
  sorry
}

end total_questions_l241_241295


namespace jericho_owes_annika_l241_241119

variable (J A M : ℝ)
variable (h1 : 2 * J = 60)
variable (h2 : M = A / 2)
variable (h3 : 30 - A - M = 9)

theorem jericho_owes_annika :
  A = 14 :=
by
  sorry

end jericho_owes_annika_l241_241119


namespace problem1_problem2_l241_241350

variable (x : ℝ)

-- Statement for the first problem
theorem problem1 : (-1 + 3 * x) * (-3 * x - 1) = 1 - 9 * x^2 := 
by
  sorry

-- Statement for the second problem
theorem problem2 : (x + 1)^2 - (1 - 3 * x) * (1 + 3 * x) = 10 * x^2 + 2 * x := 
by
  sorry

end problem1_problem2_l241_241350


namespace money_left_is_40_l241_241250

-- Define the quantities spent by Mildred and Candice, and the total money provided by their mom.
def MildredSpent : ℕ := 25
def CandiceSpent : ℕ := 35
def TotalGiven : ℕ := 100

-- Calculate the total spent and the remaining money.
def TotalSpent := MildredSpent + CandiceSpent
def MoneyLeft := TotalGiven - TotalSpent

-- Prove that the money left is 40.
theorem money_left_is_40 : MoneyLeft = 40 := by
  sorry

end money_left_is_40_l241_241250


namespace each_child_plays_for_90_minutes_l241_241972

-- Definitions based on the conditions
def total_playing_time : ℕ := 180
def children_playing_at_a_time : ℕ := 3
def total_children : ℕ := 6

-- The proof problem statement
theorem each_child_plays_for_90_minutes :
  (children_playing_at_a_time * total_playing_time) / total_children = 90 := by
  sorry

end each_child_plays_for_90_minutes_l241_241972


namespace find_x_l241_241495

theorem find_x
  (p q : ℝ)
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.33333333333333337) :
  x = 6 :=
sorry

end find_x_l241_241495


namespace drums_of_grapes_per_day_l241_241930

-- Definitions derived from conditions
def pickers := 235
def raspberry_drums_per_day := 100
def total_days := 77
def total_drums := 17017

-- Prove the main theorem
theorem drums_of_grapes_per_day : (total_drums - total_days * raspberry_drums_per_day) / total_days = 121 := by
  sorry

end drums_of_grapes_per_day_l241_241930


namespace abs_minus_five_plus_three_l241_241996

theorem abs_minus_five_plus_three : |(-5 + 3)| = 2 := 
by
  sorry

end abs_minus_five_plus_three_l241_241996


namespace prime_dvd_square_l241_241856

theorem prime_dvd_square (p n : ℕ) (hp : Nat.Prime p) (h : p ∣ n^2) : p ∣ n :=
  sorry

end prime_dvd_square_l241_241856


namespace saroj_age_proof_l241_241720

def saroj_present_age (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : ℕ :=
  sorry    -- calculation logic would be here but is not needed per instruction

noncomputable def question_conditions (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : Prop :=
  vimal_age_6_years_ago / 6 = saroj_age_6_years_ago / 5 ∧
  (vimal_age_6_years_ago + 10) / 11 = (saroj_age_6_years_ago + 10) / 10 ∧
  saroj_present_age vimal_age_6_years_ago saroj_age_6_years_ago = 16

theorem saroj_age_proof (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) :
  question_conditions vimal_age_6_years_ago saroj_age_6_years_ago :=
  sorry

end saroj_age_proof_l241_241720


namespace min_nS_n_l241_241019

open Function

noncomputable def a (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

noncomputable def S (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := n * a_1 + d * n * (n - 1) / 2

theorem min_nS_n (d : ℤ) (h_a7 : ∃ a_1 : ℤ, a 7 a_1 d = 5)
  (h_S5 : ∃ a_1 : ℤ, S 5 a_1 d = -55) :
  ∃ n : ℕ, n > 0 ∧ n * S n a_1 d = -343 :=
by
  sorry

end min_nS_n_l241_241019


namespace operation_correct_l241_241475

def operation (x y : ℝ) := x^2 + y^2 + 12

theorem operation_correct :
  operation (Real.sqrt 6) (Real.sqrt 6) = 23.999999999999996 :=
by
  -- proof omitted
  sorry

end operation_correct_l241_241475


namespace find_number_l241_241074

theorem find_number (x : ℕ) (h : 695 - 329 = x - 254) : x = 620 :=
sorry

end find_number_l241_241074


namespace fraction_checked_by_worker_y_l241_241677

variables (P X Y : ℕ)
variables (defective_rate_x defective_rate_y total_defective_rate : ℚ)
variables (h1 : X + Y = P)
variables (h2 : defective_rate_x = 0.005)
variables (h3 : defective_rate_y = 0.008)
variables (h4 : total_defective_rate = 0.007)
variables (defective_x : ℚ := 0.005 * X)
variables (defective_y : ℚ := 0.008 * Y)
variables (total_defective_products : ℚ := 0.007 * P)
variables (h5 : defective_x + defective_y = total_defective_products)

theorem fraction_checked_by_worker_y : Y / P = 2 / 3 :=
by sorry

end fraction_checked_by_worker_y_l241_241677


namespace max_k_C_l241_241464

theorem max_k_C (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  ∃ k : ℕ, (k = ((n + 1) / 2) ^ 2) := 
sorry

end max_k_C_l241_241464


namespace tank_emptying_time_correct_l241_241977

noncomputable def tank_emptying_time : ℝ :=
  let initial_volume := 1 / 5
  let fill_rate := 1 / 15
  let empty_rate := 1 / 6
  let combined_rate := fill_rate - empty_rate
  initial_volume / combined_rate

theorem tank_emptying_time_correct :
  tank_emptying_time = 2 :=
by
  -- Proof will be provided here
  sorry

end tank_emptying_time_correct_l241_241977


namespace TJs_average_time_l241_241752

theorem TJs_average_time 
  (total_distance : ℝ) 
  (distance_half : ℝ)
  (time_first_half : ℝ) 
  (time_second_half : ℝ) 
  (H1 : total_distance = 10) 
  (H2 : distance_half = total_distance / 2) 
  (H3 : time_first_half = 20) 
  (H4 : time_second_half = 30) :
  (time_first_half + time_second_half) / total_distance = 5 :=
by
  sorry

end TJs_average_time_l241_241752


namespace balls_balance_l241_241058

theorem balls_balance (G Y W B : ℕ) (h1 : G = 2 * B) (h2 : Y = 5 * B / 2) (h3 : W = 3 * B / 2) :
  5 * G + 3 * Y + 3 * W = 22 * B :=
by
  sorry

end balls_balance_l241_241058


namespace area_ratio_of_circles_l241_241565

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l241_241565


namespace range_of_m_l241_241055

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = 3) (h3 : x + y > 0) : m > -4 := by
  sorry

end range_of_m_l241_241055


namespace sum_fourth_powers_eq_t_l241_241254

theorem sum_fourth_powers_eq_t (a b t : ℝ) (h1 : a + b = t) (h2 : a^2 + b^2 = t) (h3 : a^3 + b^3 = t) : 
  a^4 + b^4 = t := 
by
  sorry

end sum_fourth_powers_eq_t_l241_241254


namespace total_walnut_trees_l241_241889

-- Define the conditions
def current_walnut_trees := 4
def new_walnut_trees := 6

-- State the lean proof problem
theorem total_walnut_trees : current_walnut_trees + new_walnut_trees = 10 := by
  sorry

end total_walnut_trees_l241_241889


namespace order_of_6_l241_241106

def f (x : ℤ) : ℤ := (x^2) % 13

theorem order_of_6 :
  ∀ n : ℕ, (∀ k < n, f^[k] 6 ≠ 6) → f^[n] 6 = 6 → n = 72 :=
by
  sorry

end order_of_6_l241_241106


namespace incorrect_statement_A_l241_241151

-- Define the statements based on conditions
def statementA : String := "INPUT \"MATH=\"; a+b+c"
def statementB : String := "PRINT \"MATH=\"; a+b+c"
def statementC : String := "a=b+c"
def statementD : String := "a=b-c"

-- Define a function to check if a statement is valid syntax
noncomputable def isValidSyntax : String → Prop :=
  λ stmt => 
    stmt = statementB ∨ stmt = statementC ∨ stmt = statementD

-- The proof problem
theorem incorrect_statement_A : ¬ isValidSyntax statementA :=
  sorry

end incorrect_statement_A_l241_241151


namespace cost_of_staying_23_days_l241_241547

def hostel_cost (days: ℕ) : ℝ :=
  if days ≤ 7 then
    days * 18
  else
    7 * 18 + (days - 7) * 14

theorem cost_of_staying_23_days : hostel_cost 23 = 350 :=
by
  sorry

end cost_of_staying_23_days_l241_241547


namespace area_of_triangle_l241_241307

theorem area_of_triangle (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 144) : 
  1/2 * a * b = 30 :=
by sorry

end area_of_triangle_l241_241307


namespace people_in_group_l241_241050

theorem people_in_group (n : ℕ) 
  (h1 : ∀ (new_weight old_weight : ℕ), old_weight = 70 → new_weight = 110 → (70 * n + (new_weight - old_weight) = 70 * n + 4 * n)) :
  n = 10 :=
sorry

end people_in_group_l241_241050


namespace mean_of_second_set_l241_241219

def mean (l: List ℕ) : ℚ :=
  (l.sum: ℚ) / l.length

theorem mean_of_second_set (x: ℕ) 
  (h: mean [28, x, 42, 78, 104] = 90): 
  mean [128, 255, 511, 1023, x] = 423 :=
by
  sorry

end mean_of_second_set_l241_241219


namespace trajectory_is_ellipse_l241_241533

noncomputable def trajectory_of_P (P : ℝ × ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.fst^2 + N.snd^2 = 8 ∧ 
                 ∃ (M : ℝ × ℝ), M.fst = 0 ∧ M.snd = N.snd ∧
                 P.fst = N.fst / 2 ∧ P.snd = N.snd

theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : trajectory_of_P P) : 
  P.fst^2 / 2 + P.snd^2 / 8 = 1 :=
by
  sorry

end trajectory_is_ellipse_l241_241533


namespace find_p_l241_241535

variables (a b c p : ℝ)

theorem find_p 
  (h1 : 9 / (a + b) = 13 / (c - b)) : 
  p = 22 :=
sorry

end find_p_l241_241535


namespace distance_between_foci_of_hyperbola_l241_241108

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), x^2 - 6 * x - 4 * y^2 - 8 * y = 27 → (4 * Real.sqrt 10) = 4 * Real.sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l241_241108


namespace product_of_two_numbers_l241_241755

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 :=
sorry

end product_of_two_numbers_l241_241755


namespace percentile_75_eq_95_l241_241112

def seventy_fifth_percentile (data : List ℕ) : ℕ := sorry

theorem percentile_75_eq_95 : seventy_fifth_percentile [92, 93, 88, 99, 89, 95] = 95 := 
sorry

end percentile_75_eq_95_l241_241112


namespace kates_discount_is_8_percent_l241_241412

-- Definitions based on the problem's conditions
def bobs_bill : ℤ := 30
def kates_bill : ℤ := 25
def total_paid : ℤ := 53
def total_without_discount : ℤ := bobs_bill + kates_bill
def discount_received : ℤ := total_without_discount - total_paid
def kates_discount_percentage : ℚ := (discount_received : ℚ) / kates_bill * 100

-- The theorem to prove
theorem kates_discount_is_8_percent : kates_discount_percentage = 8 :=
by
  sorry

end kates_discount_is_8_percent_l241_241412


namespace islanders_liars_l241_241897

inductive Person
| A
| B

open Person

def is_liar (p : Person) : Prop :=
  sorry -- placeholder for the actual definition

def makes_statement (p : Person) (statement : Prop) : Prop :=
  sorry -- placeholder for the actual definition

theorem islanders_liars :
  makes_statement A (is_liar A ∧ ¬ is_liar B) →
  is_liar A ∧ is_liar B :=
by
  sorry

end islanders_liars_l241_241897


namespace base8_to_base10_correct_l241_241398

def base8_to_base10_conversion : Prop :=
  (2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166)

theorem base8_to_base10_correct : base8_to_base10_conversion :=
by
  sorry

end base8_to_base10_correct_l241_241398


namespace cubic_roots_fraction_l241_241480

theorem cubic_roots_fraction 
  (a b c d : ℝ)
  (h_eq : ∀ x: ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) :
  c / d = -1 / 12 :=
by
  sorry

end cubic_roots_fraction_l241_241480


namespace range_of_a_l241_241505

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → 3 * x - a ≥ 0) → a ≤ 6 :=
by
  intros h
  sorry

end range_of_a_l241_241505


namespace find_number_l241_241442

variable (n : ℝ)

theorem find_number (h₁ : (0.47 * 1442 - 0.36 * n) + 63 = 3) : 
  n = 2049.28 := 
by 
  sorry

end find_number_l241_241442


namespace geometric_sequence_b_value_l241_241425

theorem geometric_sequence_b_value
  (b : ℝ)
  (hb_pos : b > 0)
  (hgeom : ∃ r : ℝ, 30 * r = b ∧ b * r = 3 / 8) :
  b = 7.5 := by
  sorry

end geometric_sequence_b_value_l241_241425


namespace first_term_of_geometric_series_l241_241405

theorem first_term_of_geometric_series (a r S : ℝ)
  (h_sum : S = a / (1 - r))
  (h_r : r = 1/3)
  (h_S : S = 18) :
  a = 12 :=
by
  sorry

end first_term_of_geometric_series_l241_241405


namespace work_together_days_l241_241171

-- Define the days it takes for A and B to complete the work individually.
def days_A : ℕ := 3
def days_B : ℕ := 6

-- Define the combined work rate.
def combined_work_rate : ℚ := (1 / days_A) + (1 / days_B)

-- State the theorem for the number of days A and B together can complete the work.
theorem work_together_days :
  1 / combined_work_rate = 2 := by
  sorry

end work_together_days_l241_241171


namespace sqrt_22_gt_4_l241_241013

theorem sqrt_22_gt_4 : Real.sqrt 22 > 4 := 
sorry

end sqrt_22_gt_4_l241_241013


namespace smallest_multiple_of_45_and_60_not_divisible_by_18_l241_241279

noncomputable def smallest_multiple_not_18 (n : ℕ) : Prop :=
  (n % 45 = 0) ∧
  (n % 60 = 0) ∧
  (n % 18 ≠ 0) ∧
  ∀ m : ℕ, (m % 45 = 0) ∧ (m % 60 = 0) ∧ (m % 18 ≠ 0) → n ≤ m

theorem smallest_multiple_of_45_and_60_not_divisible_by_18 : ∃ n : ℕ, smallest_multiple_not_18 n ∧ n = 810 := 
by
  existsi 810
  sorry

end smallest_multiple_of_45_and_60_not_divisible_by_18_l241_241279


namespace student_competition_distribution_l241_241675

theorem student_competition_distribution :
  ∃ f : Fin 4 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ x : Fin 4, f x = i ∧ ∃ y : Fin 4, f y = j) ∧ 
  (Finset.univ.image f).card = 3 := 
sorry

end student_competition_distribution_l241_241675


namespace find_divisor_nearest_to_3105_l241_241967

def nearest_divisible_number (n : ℕ) (d : ℕ) : ℕ :=
  if n % d = 0 then n else n + d - (n % d)

theorem find_divisor_nearest_to_3105 (d : ℕ) (h : nearest_divisible_number 3105 d = 3108) : d = 3 :=
by
  sorry

end find_divisor_nearest_to_3105_l241_241967


namespace range_of_g_le_2_minus_x_l241_241461

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ :=
if x ≥ 0 then f x else -f (-x)

theorem range_of_g_le_2_minus_x : {x : ℝ | g x ≤ 2 - x} = {x : ℝ | x ≤ 1} :=
by sorry

end range_of_g_le_2_minus_x_l241_241461


namespace eggs_collected_week_l241_241949

def num_chickens : ℕ := 6
def num_ducks : ℕ := 4
def num_geese : ℕ := 2
def eggs_per_chicken : ℕ := 3
def eggs_per_duck : ℕ := 2
def eggs_per_goose : ℕ := 1

def eggs_per_day (num_birds eggs_per_bird : ℕ) : ℕ := num_birds * eggs_per_bird

def eggs_collected_monday_to_saturday : ℕ :=
  6 * (eggs_per_day num_chickens eggs_per_chicken +
       eggs_per_day num_ducks eggs_per_duck +
       eggs_per_day num_geese eggs_per_goose)

def eggs_collected_sunday : ℕ :=
  eggs_per_day num_chickens (eggs_per_chicken - 1) +
  eggs_per_day num_ducks (eggs_per_duck - 1) +
  eggs_per_day num_geese (eggs_per_goose - 1)

def total_eggs_collected : ℕ :=
  eggs_collected_monday_to_saturday + eggs_collected_sunday

theorem eggs_collected_week : total_eggs_collected = 184 :=
by sorry

end eggs_collected_week_l241_241949


namespace log_x_squared_y_squared_l241_241389

theorem log_x_squared_y_squared (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x^2 * y^2) = 12 / 5 := 
by
  sorry

end log_x_squared_y_squared_l241_241389


namespace last_two_digits_of_floor_l241_241854

def last_two_digits (n : Nat) : Nat :=
  n % 100

theorem last_two_digits_of_floor :
  let x := 10^93
  let y := 10^31
  last_two_digits (Nat.floor (x / (y + 3))) = 8 :=
by
  sorry

end last_two_digits_of_floor_l241_241854


namespace fair_decision_l241_241310

def fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

theorem fair_decision (b c : ℕ) : fair_selection b c := by
  sorry

end fair_decision_l241_241310


namespace trainers_hours_split_equally_l241_241903

noncomputable def dolphins := 12
noncomputable def hours_per_dolphin := 5
noncomputable def trainers := 4

theorem trainers_hours_split_equally :
  (dolphins * hours_per_dolphin) / trainers = 15 :=
by
  sorry

end trainers_hours_split_equally_l241_241903


namespace initial_distance_between_Seonghyeon_and_Jisoo_l241_241750

theorem initial_distance_between_Seonghyeon_and_Jisoo 
  (D : ℝ)
  (h1 : 2000 = (D - 200) + 1000) : 
  D = 1200 :=
by
  sorry

end initial_distance_between_Seonghyeon_and_Jisoo_l241_241750


namespace distance_between_A_and_B_l241_241332

theorem distance_between_A_and_B :
  let A := (0, 0)
  let B := (-10, 24)
  dist A B = 26 :=
by
  sorry

end distance_between_A_and_B_l241_241332


namespace cost_price_is_975_l241_241428

-- Definitions from the conditions
def selling_price : ℝ := 1170
def profit_percentage : ℝ := 0.20

-- The proof statement
theorem cost_price_is_975 : (selling_price / (1 + profit_percentage)) = 975 := by
  sorry

end cost_price_is_975_l241_241428


namespace ian_leftover_money_l241_241734

def ianPayments (initial: ℝ) (colin: ℝ) (helen: ℝ) (benedict: ℝ) (emmaInitial: ℝ) (interest: ℝ) (avaAmount: ℝ) (conversionRate: ℝ) : ℝ :=
  let emmaTotal := emmaInitial + (interest * emmaInitial)
  let avaTotal := (avaAmount * 0.75) * conversionRate
  initial - (colin + helen + benedict + emmaTotal + avaTotal)

theorem ian_leftover_money :
  let initial := 100
  let colin := 20
  let twice_colin := 2 * colin
  let half_helen := twice_colin / 2
  let emmaInitial := 15
  let interest := 0.10
  let avaAmount := 8
  let conversionRate := 1.20
  ianPayments initial colin twice_colin half_helen emmaInitial interest avaAmount conversionRate = -3.70
:= by
  sorry

end ian_leftover_money_l241_241734


namespace gaoan_total_revenue_in_scientific_notation_l241_241526

theorem gaoan_total_revenue_in_scientific_notation :
  (21 * 10^9 : ℝ) = 2.1 * 10^9 :=
sorry

end gaoan_total_revenue_in_scientific_notation_l241_241526


namespace parabola_increasing_implies_a_lt_zero_l241_241208

theorem parabola_increasing_implies_a_lt_zero (a : ℝ) :
  (∀ x : ℝ, x < 0 → a * (2 * x) > 0) → a < 0 :=
by
  sorry

end parabola_increasing_implies_a_lt_zero_l241_241208


namespace no_quad_term_l241_241655

theorem no_quad_term (x m : ℝ) : 
  (2 * x^2 - 2 * (7 + 3 * x - 2 * x^2) + m * x^2) = -6 * x - 14 → m = -6 := 
by 
  sorry

end no_quad_term_l241_241655


namespace non_neg_int_solutions_inequality_l241_241781

theorem non_neg_int_solutions_inequality :
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} :=
by
  sorry

end non_neg_int_solutions_inequality_l241_241781


namespace problem1_problem2_l241_241301

-- Define Sn as given
def S (n : ℕ) : ℕ := (n ^ 2 + n) / 2

-- Define a sequence a_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define b_n using a_n = log_2 b_n
def b (n : ℕ) : ℕ := 2 ^ n

-- Define the sum of first n terms of sequence b_n
def T (n : ℕ) : ℕ := (2 ^ (n + 1)) - 2

-- Our main theorem statements
theorem problem1 (n : ℕ) : a n = n := by
  sorry

theorem problem2 (n : ℕ) : (Finset.range n).sum b = T n := by
  sorry

end problem1_problem2_l241_241301


namespace arithmetic_sequence_a3_value_l241_241006

theorem arithmetic_sequence_a3_value {a : ℕ → ℕ}
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
sorry

end arithmetic_sequence_a3_value_l241_241006


namespace area_first_side_l241_241189

-- Define dimensions of the box
variables (L W H : ℝ)

-- Define conditions
def area_WH : Prop := W * H = 72
def area_LH : Prop := L * H = 60
def volume_box : Prop := L * W * H = 720

-- Prove the area of the first side
theorem area_first_side (h1 : area_WH W H) (h2 : area_LH L H) (h3 : volume_box L W H) : L * W = 120 :=
by sorry

end area_first_side_l241_241189


namespace calculate_value_l241_241701

theorem calculate_value :
  let number := 1.375
  let coef := 0.6667
  let increment := 0.75
  coef * number + increment = 1.666675 :=
by
  sorry

end calculate_value_l241_241701


namespace tree_height_at_two_years_l241_241888

variable (h : ℕ → ℕ)

-- Given conditions
def condition1 := h 4 = 81
def condition2 := ∀ t : ℕ, h (t + 1) = 3 * h t

theorem tree_height_at_two_years
  (h_tripled : ∀ t : ℕ, h (t + 1) = 3 * h t)
  (h_at_four : h 4 = 81) :
  h 2 = 9 :=
by
  -- Formal proof will be provided here
  sorry

end tree_height_at_two_years_l241_241888


namespace sum_of_first_11_terms_l241_241061

theorem sum_of_first_11_terms (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + a 8 = 16) : (11 / 2) * (a 1 + a 11) = 88 :=
by
  sorry

end sum_of_first_11_terms_l241_241061


namespace k_value_l241_241759

theorem k_value (k : ℝ) (h : 10 * k * (-1)^3 - (-1) - 9 = 0) : k = -4 / 5 :=
by
  sorry

end k_value_l241_241759


namespace jason_spent_at_music_store_l241_241252

theorem jason_spent_at_music_store 
  (cost_flute : ℝ) (cost_music_tool : ℝ) (cost_song_book : ℝ)
  (h1 : cost_flute = 142.46)
  (h2 : cost_music_tool = 8.89)
  (h3 : cost_song_book = 7) :
  cost_flute + cost_music_tool + cost_song_book = 158.35 :=
by
  -- assumption proof
  sorry

end jason_spent_at_music_store_l241_241252


namespace min_value_of_linear_expression_l241_241225

theorem min_value_of_linear_expression {x y : ℝ} (h1 : 2 * x - y ≥ 0) (h2 : x + y - 3 ≥ 0) (h3 : y - x ≥ 0) :
  ∃ z, z = 2 * x + y ∧ z = 4 := by
  sorry

end min_value_of_linear_expression_l241_241225


namespace sufficient_but_not_necessary_l241_241323

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 2) : (1/x < 1/2 ∧ (∃ y : ℝ, 1/y < 1/2 ∧ y ≤ 2)) :=
by { sorry }

end sufficient_but_not_necessary_l241_241323


namespace largest_percentage_increase_is_2013_to_2014_l241_241689

-- Defining the number of students in each year as constants
def students_2010 : ℕ := 50
def students_2011 : ℕ := 56
def students_2012 : ℕ := 62
def students_2013 : ℕ := 68
def students_2014 : ℕ := 77
def students_2015 : ℕ := 81

-- Defining the percentage increase between consecutive years
def percentage_increase (a b : ℕ) : ℚ := ((b - a) : ℚ) / (a : ℚ)

-- Calculating all the percentage increases
def pi_2010_2011 := percentage_increase students_2010 students_2011
def pi_2011_2012 := percentage_increase students_2011 students_2012
def pi_2012_2013 := percentage_increase students_2012 students_2013
def pi_2013_2014 := percentage_increase students_2013 students_2014
def pi_2014_2015 := percentage_increase students_2014 students_2015

-- The theorem stating the largest percentage increase is between 2013 and 2014
theorem largest_percentage_increase_is_2013_to_2014 :
  max (pi_2010_2011) (max (pi_2011_2012) (max (pi_2012_2013) (max (pi_2013_2014) (pi_2014_2015)))) = pi_2013_2014 :=
sorry

end largest_percentage_increase_is_2013_to_2014_l241_241689


namespace scientific_notation_l241_241767

theorem scientific_notation (n : ℝ) (h : n = 1300000) : n = 1.3 * 10^6 :=
by {
  sorry
}

end scientific_notation_l241_241767


namespace max_square_plots_l241_241885

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end max_square_plots_l241_241885


namespace sum_of_roots_eq_a_plus_b_l241_241195

theorem sum_of_roots_eq_a_plus_b (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - (a + b) * x + (ab + 1) = 0 → (x = a ∨ x = b)) :
  a + b = a + b :=
by sorry

end sum_of_roots_eq_a_plus_b_l241_241195


namespace product_value_l241_241083

theorem product_value :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
    -- Skipping the actual proof
    sorry

end product_value_l241_241083


namespace OM_geq_ON_l241_241051

variables {A B C D E F G H P Q M N O : Type*}

-- Definitions for geometrical concepts
def is_intersection_of_diagonals (M : Type*) (A B C D : Type*) : Prop :=
-- M is the intersection of the diagonals AC and BD
sorry

def is_intersection_of_midlines (N : Type*) (A B C D : Type*) : Prop :=
-- N is the intersection of the midlines connecting the midpoints of opposite sides
sorry

def is_center_of_circumscribed_circle (O : Type*) (A B C D : Type*) : Prop :=
-- O is the center of the circumscribed circle around quadrilateral ABCD
sorry

-- Proof problem
theorem OM_geq_ON (A B C D M N O : Type*) 
  (hm : is_intersection_of_diagonals M A B C D)
  (hn : is_intersection_of_midlines N A B C D)
  (ho : is_center_of_circumscribed_circle O A B C D) : 
  ∃ (OM ON : ℝ), OM ≥ ON :=
sorry

end OM_geq_ON_l241_241051


namespace minimum_y_value_inequality_proof_l241_241684
-- Import necessary Lean library

-- Define a > 0, b > 0, and a + b = 1
variables {a b : ℝ}
variables (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 1)

-- Statement for Part (I): Prove the minimum value of y is 25/4
theorem minimum_y_value :
  (a + 1/a) * (b + 1/b) = 25/4 :=
sorry

-- Statement for Part (II): Prove the inequality
theorem inequality_proof :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 :=
sorry

end minimum_y_value_inequality_proof_l241_241684


namespace DongfangElementary_total_students_l241_241355

theorem DongfangElementary_total_students (x y : ℕ) 
  (h1 : x = y + 2)
  (h2 : 10 * (y + 2) = 22 * 11 * (y - 22))
  (h3 : x - x / 11 = 2 * (y - 22)) :
  x + y = 86 :=
by
  sorry

end DongfangElementary_total_students_l241_241355


namespace judy_hits_percentage_l241_241686

theorem judy_hits_percentage 
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (single_hits_percentage : ℚ) :
  total_hits = 35 →
  home_runs = 1 →
  triples = 1 →
  doubles = 5 →
  single_hits_percentage = (total_hits - (home_runs + triples + doubles)) / total_hits * 100 →
  single_hits_percentage = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end judy_hits_percentage_l241_241686


namespace profit_percentage_l241_241783

def cost_price : ℝ := 60
def selling_price : ℝ := 78

theorem profit_percentage : ((selling_price - cost_price) / cost_price) * 100 = 30 := 
by
  sorry

end profit_percentage_l241_241783


namespace non_congruent_rectangles_count_l241_241871

theorem non_congruent_rectangles_count :
  (∃ (l w : ℕ), l + w = 50 ∧ l ≠ w) ∧
  (∀ (l w : ℕ), l + w = 50 ∧ l ≠ w → l > w) →
  (∃ (n : ℕ), n = 24) :=
by
  sorry

end non_congruent_rectangles_count_l241_241871


namespace common_point_sufficient_condition_l241_241836

theorem common_point_sufficient_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3) → k ≤ -2 * Real.sqrt 2 :=
by
  -- Proof will go here
  sorry

end common_point_sufficient_condition_l241_241836


namespace pieces_of_gum_per_nickel_l241_241682

-- Definitions based on the given conditions
def initial_nickels : ℕ := 5
def remaining_nickels : ℕ := 2
def total_gum_pieces : ℕ := 6

-- We need to prove that Quentavious gets 2 pieces of gum per nickel.
theorem pieces_of_gum_per_nickel 
  (initial_nickels remaining_nickels total_gum_pieces : ℕ)
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum_pieces = 6) :
  total_gum_pieces / (initial_nickels - remaining_nickels) = 2 :=
by {
  sorry
}

end pieces_of_gum_per_nickel_l241_241682


namespace number_of_senior_citizen_tickets_sold_on_first_day_l241_241443

theorem number_of_senior_citizen_tickets_sold_on_first_day 
  (S : ℤ) (x : ℤ)
  (student_ticket_price : ℤ := 9)
  (first_day_sales : ℤ := 79)
  (second_day_sales : ℤ := 246) 
  (first_day_student_tickets_sold : ℤ := 3)
  (second_day_senior_tickets_sold : ℤ := 12)
  (second_day_student_tickets_sold : ℤ := 10) 
  (h1 : 12 * S + 10 * student_ticket_price = second_day_sales)
  (h2 : S * x + first_day_student_tickets_sold * student_ticket_price = first_day_sales) : 
  x = 4 :=
by
  sorry

end number_of_senior_citizen_tickets_sold_on_first_day_l241_241443


namespace cost_to_cover_wall_with_tiles_l241_241139

/--
There is a wall in the shape of a rectangle with a width of 36 centimeters (cm) and a height of 72 centimeters (cm).
On this wall, you want to attach tiles that are 3 centimeters (cm) and 4 centimeters (cm) in length and width, respectively,
without any empty space. If it costs 2500 won per tile, prove that the total cost to cover the wall is 540,000 won.

Conditions:
- width_wall = 36
- height_wall = 72
- width_tile = 3
- height_tile = 4
- cost_per_tile = 2500

Target:
- Total_cost = 540,000 won
-/
theorem cost_to_cover_wall_with_tiles :
  let width_wall := 36
  let height_wall := 72
  let width_tile := 3
  let height_tile := 4
  let cost_per_tile := 2500
  let area_wall := width_wall * height_wall
  let area_tile := width_tile * height_tile
  let number_of_tiles := area_wall / area_tile
  let total_cost := number_of_tiles * cost_per_tile
  total_cost = 540000 := by
  sorry

end cost_to_cover_wall_with_tiles_l241_241139


namespace fraction_equality_l241_241304

theorem fraction_equality : (2 + 4) / (1 + 2) = 2 := by
  sorry

end fraction_equality_l241_241304


namespace original_price_l241_241434

theorem original_price 
  (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : SP = 15)
  (h2 : gain_percent = 0.50)
  (h3 : SP = P * (1 + gain_percent)) :
  P = 10 :=
by
  sorry

end original_price_l241_241434


namespace solve_price_per_litre_second_oil_l241_241291

variable (P : ℝ)

def price_per_litre_second_oil :=
  10 * 55 + 5 * P = 15 * 58.67

theorem solve_price_per_litre_second_oil (h : price_per_litre_second_oil P) : P = 66.01 :=
  by
  sorry

end solve_price_per_litre_second_oil_l241_241291


namespace find_C_marks_l241_241717

theorem find_C_marks :
  let english := 90
  let math := 92
  let physics := 85
  let biology := 85
  let avg_marks := 87.8
  let total_marks := avg_marks * 5
  let other_marks := english + math + physics + biology
  ∃ C : ℝ, total_marks - other_marks = C ∧ C = 87 :=
by
  sorry

end find_C_marks_l241_241717


namespace graph_equation_l241_241773

theorem graph_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_equation_l241_241773


namespace smallest_natural_number_l241_241156

theorem smallest_natural_number (x : ℕ) : 
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) → x = 122 := 
by
  sorry

end smallest_natural_number_l241_241156


namespace initial_glass_bottles_count_l241_241629

namespace Bottles

variable (G P : ℕ)

/-- The weight of some glass bottles is 600 g. 
    The total weight of 4 glass bottles and 5 plastic bottles is 1050 g.
    A glass bottle is 150 g heavier than a plastic bottle.
    Prove that the number of glass bottles initially weighed is 3. -/
theorem initial_glass_bottles_count (h1 : G * (P + 150) = 600)
  (h2 : 4 * (P + 150) + 5 * P = 1050)
  (h3 : P + 150 > P) :
  G = 3 :=
  by sorry

end Bottles

end initial_glass_bottles_count_l241_241629


namespace exists_q_no_zero_in_decimal_l241_241916

theorem exists_q_no_zero_in_decimal : ∃ q : ℕ, ∀ (d : ℕ), q * 2 ^ 1967 ≠ 10 * d := 
sorry

end exists_q_no_zero_in_decimal_l241_241916


namespace f_injective_on_restricted_domain_l241_241605

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- Define the restricted domain
def f_restricted (x : ℝ) (h : -2 <= x) : ℝ := f x

-- The main statement to be proved
theorem f_injective_on_restricted_domain : 
  (∀ x1 x2 : {x // -2 <= x}, f_restricted x1.val x1.property = f_restricted x2.val x2.property → x1 = x2) := 
sorry

end f_injective_on_restricted_domain_l241_241605


namespace find_value_of_a_minus_b_l241_241077

variable (a b : ℝ)

theorem find_value_of_a_minus_b (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := 
sorry

end find_value_of_a_minus_b_l241_241077


namespace factor_expression_l241_241242

theorem factor_expression (x : ℝ) : 100 * x ^ 23 + 225 * x ^ 46 = 25 * x ^ 23 * (4 + 9 * x ^ 23) :=
by
  -- Proof steps will go here
  sorry

end factor_expression_l241_241242


namespace ratio_trumpet_to_flute_l241_241876

-- Given conditions
def flute_players : ℕ := 5
def trumpet_players (T : ℕ) : ℕ := T
def trombone_players (T : ℕ) : ℕ := T - 8
def drummers (T : ℕ) : ℕ := T - 8 + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players (T : ℕ) : ℕ := T - 8 + 3
def total_seats_needed (T : ℕ) : ℕ := 
  flute_players + trumpet_players T + trombone_players T + drummers T + clarinet_players + french_horn_players T

-- Proof statement
theorem ratio_trumpet_to_flute 
  (T : ℕ) (h : total_seats_needed T = 65) : trumpet_players T / flute_players = 3 :=
sorry

end ratio_trumpet_to_flute_l241_241876


namespace chloe_treasures_first_level_l241_241190

def chloe_treasures_score (T : ℕ) (score_per_treasure : ℕ) (treasures_second_level : ℕ) (total_score : ℕ) :=
  T * score_per_treasure + treasures_second_level * score_per_treasure = total_score

theorem chloe_treasures_first_level :
  chloe_treasures_score T 9 3 81 → T = 6 :=
by
  intro h
  sorry

end chloe_treasures_first_level_l241_241190


namespace number_of_matches_is_85_l241_241080

open Nat

/-- This definition calculates combinations of n taken k at a time. -/
def binom (n k : ℕ) : ℕ := n.choose k

/-- The calculation of total number of matches in the entire tournament. -/
def total_matches (groups teams_per_group : ℕ) : ℕ :=
  let matches_per_group := binom teams_per_group 2
  let total_matches_first_round := groups * matches_per_group
  let matches_final_round := binom groups 2
  total_matches_first_round + matches_final_round

/-- Theorem proving the total number of matches played is 85, given 5 groups with 6 teams each. -/
theorem number_of_matches_is_85 : total_matches 5 6 = 85 :=
  by
  sorry

end number_of_matches_is_85_l241_241080


namespace system_no_solution_iff_n_eq_neg_one_l241_241097

def no_solution_system (n : ℝ) : Prop :=
  ¬∃ x y z : ℝ, (n * x + y = 1) ∧ (n * y + z = 1) ∧ (x + n * z = 1)

theorem system_no_solution_iff_n_eq_neg_one (n : ℝ) : no_solution_system n ↔ n = -1 :=
sorry

end system_no_solution_iff_n_eq_neg_one_l241_241097


namespace andrew_kept_stickers_l241_241009

theorem andrew_kept_stickers :
  ∃ (b d f e g h : ℕ), b = 2000 ∧ d = (5 * b) / 100 ∧ f = d + 120 ∧ e = (d + f) / 2 ∧ g = 80 ∧ h = (e + g) / 5 ∧ (b - (d + f + e + g + h) = 1392) :=
sorry

end andrew_kept_stickers_l241_241009


namespace find_x_l241_241449

def determinant (a b c d : ℚ) : ℚ := a * d - b * c

theorem find_x (x : ℚ) (h : determinant (2 * x) (-4) x 1 = 18) : x = 3 :=
  sorry

end find_x_l241_241449


namespace largest_value_is_B_l241_241625

def exprA := 1 + 2 * 3 + 4
def exprB := 1 + 2 + 3 * 4
def exprC := 1 + 2 + 3 + 4
def exprD := 1 * 2 + 3 + 4
def exprE := 1 * 2 + 3 * 4

theorem largest_value_is_B : exprB = 15 ∧ exprB > exprA ∧ exprB > exprC ∧ exprB > exprD ∧ exprB > exprE := 
by
  sorry

end largest_value_is_B_l241_241625


namespace arithmetic_sequence_n_value_l241_241956

theorem arithmetic_sequence_n_value
  (a : ℕ → ℚ)
  (h1 : a 1 = 1 / 3)
  (h2 : a 2 + a 5 = 4)
  (h3 : a n = 33)
  : n = 50 :=
sorry

end arithmetic_sequence_n_value_l241_241956


namespace g_triple_evaluation_l241_241778

def g (x : ℤ) : ℤ := 
if x < 8 then x ^ 2 - 6 
else x - 15

theorem g_triple_evaluation :
  g (g (g 20)) = 4 :=
by sorry

end g_triple_evaluation_l241_241778


namespace find_y_l241_241109

theorem find_y (y : ℝ) (h : (15 + 28 + y) / 3 = 25) : y = 32 := by
  sorry

end find_y_l241_241109


namespace probability_at_least_one_each_color_in_bag_l241_241692

open BigOperators

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_at_least_one_each_color : ℚ :=
  let total_ways := num_combinations 9 5
  let favorable_ways := 27 + 27 + 27 -- 3 scenarios (2R+1B+2G, 2B+1R+2G, 2G+1R+2B)
  favorable_ways / total_ways

theorem probability_at_least_one_each_color_in_bag :
  prob_at_least_one_each_color = 9 / 14 :=
by
  sorry

end probability_at_least_one_each_color_in_bag_l241_241692


namespace total_num_animals_l241_241574

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end total_num_animals_l241_241574


namespace part1_l241_241699

def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

theorem part1 (a : ℝ) (h : a = 1) :
  (Set.compl B ∪ A a) = {x | x ≤ 1 ∨ x ≥ 2} :=
by
  sorry

end part1_l241_241699


namespace zero_point_in_range_l241_241092

theorem zero_point_in_range (a : ℝ) (x1 x2 x3 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : x1 < x2) (h4 : x2 < x3)
  (hx1 : (x1^3 - 4*x1 + a) = 0) (hx2 : (x2^3 - 4*x2 + a) = 0) (hx3 : (x3^3 - 4*x3 + a) = 0) :
  0 < x2 ∧ x2 < 1 :=
by
  sorry

end zero_point_in_range_l241_241092


namespace proof_complex_ratio_l241_241987

noncomputable def condition1 (x y : ℂ) (k : ℝ) : Prop :=
  (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1

theorem proof_complex_ratio (x y : ℂ) (k : ℝ) (h : condition1 x y k) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = (41 / 20 : ℂ) :=
by 
  sorry

end proof_complex_ratio_l241_241987


namespace prob_not_same_city_l241_241552

def prob_A_city_A : ℝ := 0.6
def prob_B_city_A : ℝ := 0.3

theorem prob_not_same_city :
  (prob_A_city_A * (1 - prob_B_city_A) + (1 - prob_A_city_A) * prob_B_city_A) = 0.54 :=
by 
  -- This is just a placeholder to indicate that the proof is skipped
  sorry

end prob_not_same_city_l241_241552


namespace coin_landing_heads_prob_l241_241991

theorem coin_landing_heads_prob (p : ℝ) (h : p^2 * (1 - p)^3 = 0.03125) : p = 0.5 :=
by
sorry

end coin_landing_heads_prob_l241_241991


namespace intersection_at_most_one_l241_241402

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the statement to be proved
theorem intersection_at_most_one (a : ℝ) :
  ∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2 :=
by
  sorry

end intersection_at_most_one_l241_241402


namespace total_price_of_hats_l241_241831

-- Declare the conditions as Lean definitions
def total_hats : Nat := 85
def green_hats : Nat := 38
def blue_hat_cost : Nat := 6
def green_hat_cost : Nat := 7

-- The question becomes proving the total cost of the hats is $548
theorem total_price_of_hats :
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  total_blue_cost + total_green_cost = 548 := by
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  show total_blue_cost + total_green_cost = 548
  sorry

end total_price_of_hats_l241_241831


namespace minimize_travel_expense_l241_241597

noncomputable def travel_cost_A (x : ℕ) : ℝ := 2000 * x * 0.75
noncomputable def travel_cost_B (x : ℕ) : ℝ := 2000 * (x - 1) * 0.8

theorem minimize_travel_expense (x : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x ≤ 15 → travel_cost_B x < travel_cost_A x) ∧
  (x = 16 → travel_cost_A x = travel_cost_B x) ∧
  (17 ≤ x ∧ x ≤ 25 → travel_cost_A x < travel_cost_B x) :=
by
  sorry

end minimize_travel_expense_l241_241597


namespace original_population_l241_241180

theorem original_population (n : ℕ) (h1 : n + 1500 * 85 / 100 = n - 45) : n = 8800 := 
by
  sorry

end original_population_l241_241180


namespace sine_cos_suffices_sine_cos_necessary_l241_241488

theorem sine_cos_suffices
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) :
  c > Real.sqrt (a^2 + b^2) :=
sorry

theorem sine_cos_necessary
  (a b c : ℝ)
  (h : c > Real.sqrt (a^2 + b^2)) :
  ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end sine_cos_suffices_sine_cos_necessary_l241_241488


namespace sum_of_interior_numbers_l241_241672

def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

theorem sum_of_interior_numbers :
  sum_interior 8 + sum_interior 9 + sum_interior 10 = 890 :=
by
  sorry

end sum_of_interior_numbers_l241_241672


namespace repeating_decimal_division_l241_241276

def repeating_decimal_081_as_fraction : ℚ := 9 / 11
def repeating_decimal_272_as_fraction : ℚ := 30 / 11

theorem repeating_decimal_division : 
  (repeating_decimal_081_as_fraction / repeating_decimal_272_as_fraction) = (3 / 10) := 
by 
  sorry

end repeating_decimal_division_l241_241276


namespace arithmetic_sums_l241_241422

theorem arithmetic_sums (d : ℤ) (p q : ℤ) (S : ℤ → ℤ)
  (hS : ∀ n, S n = p * n^2 + q * n)
  (h_eq : S 20 = S 40) : S 60 = 0 :=
by
  sorry

end arithmetic_sums_l241_241422


namespace min_cylinder_surface_area_l241_241444

noncomputable def h := Real.sqrt (5^2 - 4^2)
noncomputable def V_cone := (1 / 3) * Real.pi * 4^2 * h
noncomputable def V_cylinder (r h': ℝ) := Real.pi * r^2 * h'
noncomputable def h' (r: ℝ) := 16 / r^2
noncomputable def S (r: ℝ) := 2 * Real.pi * r^2 + (32 * Real.pi) / r

theorem min_cylinder_surface_area : 
  ∃ r, r = 2 ∧ ∀ r', r' ≠ 2 → S r' > S 2 := sorry

end min_cylinder_surface_area_l241_241444


namespace greatest_overlap_l241_241096

-- Defining the conditions based on the problem statement
def percentage_internet (n : ℕ) : Prop := n = 35
def percentage_snacks (m : ℕ) : Prop := m = 70

-- The theorem to prove the greatest possible overlap
theorem greatest_overlap (n m k : ℕ) (hn : percentage_internet n) (hm : percentage_snacks m) : 
  k ≤ 35 :=
by sorry

end greatest_overlap_l241_241096


namespace find_x_l241_241685

theorem find_x (x : ℝ) (h : 0.35 * 400 = 0.20 * x): x = 700 :=
sorry

end find_x_l241_241685


namespace no_power_of_two_divides_3n_plus_1_l241_241883

theorem no_power_of_two_divides_3n_plus_1 (n : ℕ) (hn : n > 1) : ¬ (2^n ∣ 3^n + 1) := sorry

end no_power_of_two_divides_3n_plus_1_l241_241883


namespace determine_radius_l241_241528

variable (R r : ℝ)

theorem determine_radius (h1 : R = 10) (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 :=
  sorry

end determine_radius_l241_241528


namespace work_complete_in_15_days_l241_241698

theorem work_complete_in_15_days :
  let A_rate := (1 : ℚ) / 20
  let B_rate := (1 : ℚ) / 30
  let C_rate := (1 : ℚ) / 10
  let all_together_rate := A_rate + B_rate + C_rate
  let work_2_days := 2 * all_together_rate
  let B_C_rate := B_rate + C_rate
  let work_next_2_days := 2 * B_C_rate
  let total_work_4_days := work_2_days + work_next_2_days
  let remaining_work := 1 - total_work_4_days
  let B_time := remaining_work / B_rate

  2 + 2 + B_time = 15 :=
by
  sorry

end work_complete_in_15_days_l241_241698


namespace g_eval_at_neg2_l241_241466

def g (x : ℝ) : ℝ := x^3 + 2*x - 4

theorem g_eval_at_neg2 : g (-2) = -16 := by
  sorry

end g_eval_at_neg2_l241_241466


namespace brittany_average_correct_l241_241122

def brittany_first_score : ℤ :=
78

def brittany_second_score : ℤ :=
84

def brittany_average_after_second_test (score1 score2 : ℤ) : ℤ :=
(score1 + score2) / 2

theorem brittany_average_correct : 
  brittany_average_after_second_test brittany_first_score brittany_second_score = 81 := 
by
  sorry

end brittany_average_correct_l241_241122


namespace cross_section_area_l241_241137

-- Definitions representing the conditions
variables (AK KD BP PC DM DC : ℝ)
variable (h : ℝ)
variable (Volume : ℝ)

-- Conditions
axiom hyp1 : AK = KD
axiom hyp2 : BP = PC
axiom hyp3 : DM = 0.4 * DC
axiom hyp4 : h = 1
axiom hyp5 : Volume = 5

-- Proof problem: Prove that the area S of the cross-section of the pyramid is 3
theorem cross_section_area (S : ℝ) : S = 3 :=
by sorry

end cross_section_area_l241_241137


namespace domain_v_l241_241228

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + x - 1)

theorem domain_v :
  {x : ℝ | x >= 0 ∧ Real.sqrt x + x - 1 ≠ 0} = {x : ℝ | x ∈ Set.Ico 0 (Real.sqrt 5 - 1) ∪ Set.Ioi (Real.sqrt 5 - 1)} :=
by
  sorry

end domain_v_l241_241228


namespace difference_of_numbers_l241_241631

theorem difference_of_numbers :
  ∃ (a b : ℕ), a + b = 36400 ∧ b = 100 * a ∧ b - a = 35640 :=
by
  sorry

end difference_of_numbers_l241_241631


namespace smallest_w_value_l241_241776

theorem smallest_w_value (x y z w : ℝ) 
    (hx : -2 ≤ x ∧ x ≤ 5) 
    (hy : -3 ≤ y ∧ y ≤ 7) 
    (hz : 4 ≤ z ∧ z ≤ 8) 
    (hw : w = x * y - z) : 
    w ≥ -23 :=
sorry

end smallest_w_value_l241_241776


namespace f_always_positive_l241_241403

def f (x : ℝ) : ℝ := x^2 + 3 * x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := 
by 
  sorry

end f_always_positive_l241_241403


namespace S6_eq_24_l241_241184

-- Definitions based on the conditions provided
def is_arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def S : ℕ → ℝ := sorry  -- Sum of the first n terms of some arithmetic sequence

-- Given conditions
axiom S2_eq_2 : S 2 = 2
axiom S4_eq_10 : S 4 = 10

-- The main theorem to prove
theorem S6_eq_24 : S 6 = 24 :=
by 
  sorry  -- Proof is omitted

end S6_eq_24_l241_241184


namespace smaller_two_digit_product_is_34_l241_241559

theorem smaller_two_digit_product_is_34 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 5082) : min a b = 34 :=
by
  sorry

end smaller_two_digit_product_is_34_l241_241559


namespace quadratic_square_binomial_l241_241093

theorem quadratic_square_binomial (a : ℝ) :
  (∃ d : ℝ, 9 * x ^ 2 - 18 * x + a = (3 * x + d) ^ 2) → a = 9 :=
by
  intro h
  match h with
  | ⟨d, h_eq⟩ => sorry

end quadratic_square_binomial_l241_241093


namespace proof_C_l241_241742

variable {a b c : Type} [LinearOrder a] [LinearOrder b] [LinearOrder c]
variable {y : Type}

-- Definitions for parallel and perpendicular relationships
def parallel (x1 x2 : Type) : Prop := sorry
def perp (x1 x2 : Type) : Prop := sorry

theorem proof_C (a b c : Type) [LinearOrder a] [LinearOrder b] [LinearOrder c] (y : Type):
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perp a y ∧ perp b y → parallel a b) :=
by
  sorry

end proof_C_l241_241742


namespace earnings_of_r_l241_241906

theorem earnings_of_r (P Q R : ℕ) (h1 : 9 * (P + Q + R) = 1710) (h2 : 5 * (P + R) = 600) (h3 : 7 * (Q + R) = 910) : 
  R = 60 :=
by
  -- proof will be provided here
  sorry

end earnings_of_r_l241_241906


namespace regression_analysis_correct_statement_l241_241809

variables (x : Type) (y : Type)

def is_deterministic (v : Type) : Prop := sorry -- A placeholder definition
def is_random (v : Type) : Prop := sorry -- A placeholder definition

theorem regression_analysis_correct_statement :
  (is_deterministic x) → (is_random y) →
  ("The independent variable is a deterministic variable, and the dependent variable is a random variable" = "C") :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end regression_analysis_correct_statement_l241_241809


namespace bus_speed_excluding_stoppages_l241_241994

theorem bus_speed_excluding_stoppages (v : Real) 
  (h1 : ∀ x, x = 41) 
  (h2 : ∀ y, y = 14.444444444444443 / 60) : 
  v = 54 := 
by
  -- Proving the statement. Proof steps are skipped.
  sorry

end bus_speed_excluding_stoppages_l241_241994


namespace value_of_one_TV_mixer_blender_l241_241779

variables (M T B : ℝ)

-- The given conditions
def eq1 : Prop := 2 * M + T + B = 10500
def eq2 : Prop := T + M + 2 * B = 14700

-- The problem: find the combined value of one TV, one mixer, and one blender
theorem value_of_one_TV_mixer_blender :
  eq1 M T B → eq2 M T B → (T + M + B = 18900) :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end value_of_one_TV_mixer_blender_l241_241779


namespace sandy_money_left_l241_241264

theorem sandy_money_left (total_money : ℝ) (spent_percentage : ℝ) (money_left : ℝ) : 
  total_money = 320 → spent_percentage = 0.30 → money_left = (total_money * (1 - spent_percentage)) → 
  money_left = 224 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end sandy_money_left_l241_241264


namespace pipe_B_fill_time_l241_241763

theorem pipe_B_fill_time (T_B : ℝ) : 
  (1/3 + 1/T_B - 1/4 = 1/3) → T_B = 4 :=
sorry

end pipe_B_fill_time_l241_241763


namespace total_numbers_l241_241637

theorem total_numbers (m j c : ℕ) (h1 : m = j + 20) (h2 : j = c - 40) (h3 : c = 80) : m + j + c = 180 := 
by sorry

end total_numbers_l241_241637


namespace inequality_proof_l241_241770

variable (a b c : ℝ)

noncomputable def specific_condition (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (1 / a + 1 / b + 1 / c = 1)

theorem inequality_proof (h : specific_condition a b c) :
  (a^a * b * c + b^b * c * a + c^c * a * b) ≥ 27 * (b * c + c * a + a * b) := 
by {
  sorry
}

end inequality_proof_l241_241770


namespace gcf_of_36_and_54_l241_241988

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := 
by
  sorry

end gcf_of_36_and_54_l241_241988


namespace inequality_holds_l241_241908

variable (a b c : ℝ)

theorem inequality_holds : 
  (a * b + b * c + c * a - 1)^2 ≤ (a^2 + 1) * (b^2 + 1) * (c^2 + 1) := 
by 
  sorry

end inequality_holds_l241_241908


namespace calc_f_five_times_l241_241450

def f (x : ℕ) : ℕ := if x % 2 = 0 then x / 2 else 5 * x + 1

theorem calc_f_five_times : f (f (f (f (f 5)))) = 166 :=
by 
  sorry

end calc_f_five_times_l241_241450


namespace largest_n_with_integer_solutions_l241_241011

theorem largest_n_with_integer_solutions : ∃ n, ∀ x y1 y2 y3 y4, 
 ( ((x + 1)^2 + y1^2) = ((x + 2)^2 + y2^2) ∧  ((x + 2)^2 + y2^2) = ((x + 3)^2 + y3^2) ∧ 
  ((x + 3)^2 + y3^2) = ((x + 4)^2 + y4^2)) → (n = 3) := sorry

end largest_n_with_integer_solutions_l241_241011


namespace positive_integer_sum_representation_l241_241947

theorem positive_integer_sum_representation :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → ∃ (a : Fin 2004 → ℕ), 
    (∀ i j : Fin 2004, i < j → a i < a j) ∧ 
    (∀ i : Fin 2003, a i ∣ a (i + 1)) ∧
    (n = (Finset.univ.sum a)) := 
sorry

end positive_integer_sum_representation_l241_241947


namespace exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l241_241979

theorem exists_xy_such_that_x2_add_y2_eq_n_mod_p
  (p : ℕ) [Fact (Nat.Prime p)] (n : ℤ)
  (hp1 : p > 5) :
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = n % p) :=
sorry

theorem p_mod_4_eq_1_implies_n_can_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp1 : p % 4 = 1) : 
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

theorem p_mod_4_eq_3_implies_n_cannot_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 4 = 3) :
  ¬(∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

end exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l241_241979


namespace sum_of_roots_eq_three_l241_241583

theorem sum_of_roots_eq_three {a b : ℝ} (h₁ : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (11 - a) = 0)
  (h₂ : a * 2^3 + (a + 3 * b) * 2^2 + (b - 4 * a) * 2 + (11 - a) = 0)
  (h₃ : a * 4^3 + (a + 3 * b) * 4^2 + (b - 4 * a) * 4 + (11 - a) = 0) :
  (-3) + 2 + 4 = 3 :=
by
  sorry

end sum_of_roots_eq_three_l241_241583


namespace alyssa_limes_correct_l241_241399

-- Definitions representing the conditions
def fred_limes : Nat := 36
def nancy_limes : Nat := 35
def total_limes : Nat := 103

-- Definition of the number of limes Alyssa picked
def alyssa_limes : Nat := total_limes - (fred_limes + nancy_limes)

-- The theorem we need to prove
theorem alyssa_limes_correct : alyssa_limes = 32 := by
  sorry

end alyssa_limes_correct_l241_241399


namespace marcus_saves_34_22_l241_241748

def max_spend : ℝ := 200
def shoe_price : ℝ := 120
def shoe_discount : ℝ := 0.30
def sock_price : ℝ := 25
def sock_discount : ℝ := 0.20
def shirt_price : ℝ := 55
def shirt_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def calc_discounted_price (price discount : ℝ) : ℝ := price * (1 - discount)

def total_cost_before_tax : ℝ :=
  calc_discounted_price shoe_price shoe_discount +
  calc_discounted_price sock_price sock_discount +
  calc_discounted_price shirt_price shirt_discount

def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

def final_cost : ℝ := total_cost_before_tax + sales_tax

def money_saved : ℝ := max_spend - final_cost

theorem marcus_saves_34_22 :
  money_saved = 34.22 :=
by sorry

end marcus_saves_34_22_l241_241748


namespace max_basketballs_l241_241730

theorem max_basketballs (x : ℕ) (h1 : 80 * x + 50 * (40 - x) ≤ 2800) : x ≤ 26 := sorry

end max_basketballs_l241_241730


namespace value_standard_deviations_from_mean_l241_241437

-- Define the mean (µ)
def μ : ℝ := 15.5

-- Define the standard deviation (σ)
def σ : ℝ := 1.5

-- Define the value X
def X : ℝ := 12.5

-- Prove that the Z-score is -2
theorem value_standard_deviations_from_mean : (X - μ) / σ = -2 := by
  sorry

end value_standard_deviations_from_mean_l241_241437


namespace Chris_buys_48_golf_balls_l241_241654

theorem Chris_buys_48_golf_balls (total_golf_balls : ℕ) (dozen_to_balls : ℕ → ℕ)
  (dan_buys : ℕ) (gus_buys : ℕ) (chris_buys : ℕ) :
  dozen_to_balls 1 = 12 →
  dan_buys = 5 →
  gus_buys = 2 →
  total_golf_balls = 132 →
  (chris_buys * 12) + (dan_buys * 12) + (gus_buys * 12) = total_golf_balls →
  chris_buys * 12 = 48 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Chris_buys_48_golf_balls_l241_241654


namespace joan_dimes_spent_l241_241960

theorem joan_dimes_spent (initial_dimes remaining_dimes spent_dimes : ℕ) 
    (h_initial: initial_dimes = 5) 
    (h_remaining: remaining_dimes = 3) : 
    spent_dimes = initial_dimes - remaining_dimes := 
by 
    sorry

end joan_dimes_spent_l241_241960


namespace gcd_3060_561_l241_241272

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 :=
by
  sorry

end gcd_3060_561_l241_241272


namespace gcf_450_144_l241_241823

theorem gcf_450_144 : Nat.gcd 450 144 = 18 := by
  sorry

end gcf_450_144_l241_241823


namespace scientific_notation_example_l241_241216

theorem scientific_notation_example : 0.00001 = 1 * 10^(-5) :=
sorry

end scientific_notation_example_l241_241216


namespace positive_integer_not_in_S_l241_241172

noncomputable def S : Set ℤ :=
  {n | ∃ (i : ℕ), n = 4^i * 3 ∨ n = -4^i * 2}

theorem positive_integer_not_in_S (n : ℤ) (hn : 0 < n) (hnS : n ∉ S) :
  ∃ (x y : ℤ), x ≠ y ∧ x ∈ S ∧ y ∈ S ∧ x + y = n :=
sorry

end positive_integer_not_in_S_l241_241172


namespace mixture_volume_correct_l241_241469

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end mixture_volume_correct_l241_241469


namespace sum_of_youngest_and_oldest_cousins_l241_241384

theorem sum_of_youngest_and_oldest_cousins :
  ∃ (ages : Fin 5 → ℝ), (∃ (a1 a5 : ℝ), ages 0 = a1 ∧ ages 4 = a5 ∧ a1 + a5 = 29) ∧
                        (∃ (median : ℝ), median = ages 2 ∧ median = 7) ∧
                        (∃ (mean : ℝ), mean = (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 ∧ mean = 10) :=
by sorry

end sum_of_youngest_and_oldest_cousins_l241_241384


namespace average_marks_for_class_l241_241489

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l241_241489


namespace toms_total_miles_l241_241029

-- Define the conditions as facts
def days_in_year : ℕ := 365
def first_part_days : ℕ := 183
def second_part_days : ℕ := days_in_year - first_part_days
def miles_per_day_first_part : ℕ := 30
def miles_per_day_second_part : ℕ := 35

-- State the final theorem
theorem toms_total_miles : 
  (first_part_days * miles_per_day_first_part) + (second_part_days * miles_per_day_second_part) = 11860 := by 
  sorry

end toms_total_miles_l241_241029


namespace divisibility_of_2b_by_a_l241_241101

theorem divisibility_of_2b_by_a (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (h_cond : ∃ᶠ m in at_top, ∃ᶠ n in at_top, (∃ k₁ : ℕ, m^2 + a * n + b = k₁^2) ∧ (∃ k₂ : ℕ, n^2 + a * m + b = k₂^2)) :
  a ∣ 2 * b :=
sorry

end divisibility_of_2b_by_a_l241_241101


namespace fraction_of_cats_l241_241202

theorem fraction_of_cats (C D : ℕ) 
  (h1 : C + D = 300)
  (h2 : 4 * D = 400) : 
  (C : ℚ) / (C + D) = 2 / 3 :=
by
  sorry

end fraction_of_cats_l241_241202


namespace odd_coefficients_in_binomial_expansion_l241_241285

theorem odd_coefficients_in_binomial_expansion :
  let a : Fin 9 → ℕ := fun k => Nat.choose 8 k
  (Finset.filter (fun k => a k % 2 = 1) (Finset.Icc 0 8)).card = 2 := by
  sorry

end odd_coefficients_in_binomial_expansion_l241_241285


namespace simplify_fraction_l241_241362

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : 
  (x / (x - y) - y / (x + y)) = (x^2 + y^2) / (x^2 - y^2) :=
sorry

end simplify_fraction_l241_241362


namespace h_value_l241_241739

theorem h_value (h : ℝ) : (∃ x : ℝ, x^3 + h * x + 5 = 0 ∧ x = 3) → h = -32 / 3 := by
  sorry

end h_value_l241_241739


namespace avg_of_two_numbers_l241_241090

theorem avg_of_two_numbers (a b c d : ℕ) (h_different: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_positive: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_average: (a + b + c + d) / 4 = 4)
  (h_max_diff: ∀ x y : ℕ, (x ≠ y ∧ x > 0 ∧ y > 0 ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ x ≠ d ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ y ≠ d) → (max x y - min x y <= max a d - min a d)) : 
  (a + b + c + d - min a (min b (min c d)) - max a (max b (max c d))) / 2 = 5 / 2 :=
by sorry

end avg_of_two_numbers_l241_241090


namespace required_barrels_of_pitch_l241_241395

def total_road_length : ℕ := 16
def bags_of_gravel_per_truckload : ℕ := 2
def barrels_of_pitch_per_truckload (bgt : ℕ) : ℚ := bgt / 5
def truckloads_per_mile : ℕ := 3

def miles_paved_day1 : ℕ := 4
def miles_paved_day2 : ℕ := (miles_paved_day1 * 2) - 1
def total_miles_paved_first_two_days : ℕ := miles_paved_day1 + miles_paved_day2
def remaining_miles_paved_day3 : ℕ := total_road_length - total_miles_paved_first_two_days

def truckloads_needed (miles : ℕ) : ℕ := miles * truckloads_per_mile
def barrels_of_pitch_needed (truckloads : ℕ) (bgt : ℕ) : ℚ := truckloads * barrels_of_pitch_per_truckload bgt

theorem required_barrels_of_pitch : 
  barrels_of_pitch_needed (truckloads_needed remaining_miles_paved_day3) bags_of_gravel_per_truckload = 6 := 
by
  sorry

end required_barrels_of_pitch_l241_241395


namespace percentage_of_315_out_of_900_is_35_l241_241008

theorem percentage_of_315_out_of_900_is_35 :
  (315 : ℝ) / 900 * 100 = 35 := 
by
  sorry

end percentage_of_315_out_of_900_is_35_l241_241008


namespace average_is_12_or_15_l241_241370

variable {N : ℝ} (h : 12 < N ∧ N < 22)

theorem average_is_12_or_15 : (∃ x ∈ ({12, 15} : Set ℝ), x = (9 + 15 + N) / 3) :=
by
  have h1 : 12 < (24 + N) / 3 := by sorry
  have h2 : (24 + N) / 3 < 15.3333 := by sorry
  sorry

end average_is_12_or_15_l241_241370


namespace calculate_expression_l241_241462

-- Define the conditions
def exp1 : ℤ := (-1)^(53)
def exp2 : ℤ := 2^(2^4 + 5^2 - 4^3)

-- State and skip the proof
theorem calculate_expression :
  exp1 + exp2 = -1 + 1 / (2^23) :=
by sorry

#check calculate_expression

end calculate_expression_l241_241462


namespace simplify_and_evaluate_expression_l241_241896

theorem simplify_and_evaluate_expression 
  (x y : ℤ) (hx : x = -3) (hy : y = -2) :
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l241_241896


namespace problem_l241_241793

theorem problem (a b : ℤ) (h1 : |a - 2| = 5) (h2 : |b| = 9) (h3 : a + b < 0) :
  a - b = 16 ∨ a - b = 6 := 
sorry

end problem_l241_241793


namespace simplify_neg_cube_square_l241_241749

theorem simplify_neg_cube_square (a : ℝ) : (-a^3)^2 = a^6 :=
by
  sorry

end simplify_neg_cube_square_l241_241749


namespace change_in_mean_and_median_l241_241534

-- Original attendance data
def original_data : List ℕ := [15, 23, 17, 19, 17, 20]

-- Corrected attendance data
def corrected_data : List ℕ := [15, 23, 17, 19, 17, 25]

-- Function to compute mean
def mean (data: List ℕ) : ℚ := (data.sum : ℚ) / data.length

-- Function to compute median
def median (data: List ℕ) : ℚ :=
  let sorted := data.toArray.qsort (· ≤ ·) |>.toList
  if sorted.length % 2 == 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

-- Lean statement verifying the expected change in mean and median
theorem change_in_mean_and_median :
  mean corrected_data - mean original_data = 1 ∧ median corrected_data = median original_data :=
by -- Note the use of 'by' to structure the proof
  sorry -- Proof omitted

end change_in_mean_and_median_l241_241534


namespace range_of_a_l241_241296

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ (a < 1 ∨ a > 3) := 
sorry

end range_of_a_l241_241296


namespace total_money_made_l241_241724

-- Define the given conditions.
def total_rooms : ℕ := 260
def single_rooms : ℕ := 64
def single_room_cost : ℕ := 35
def double_room_cost : ℕ := 60

-- Define the number of double rooms.
def double_rooms : ℕ := total_rooms - single_rooms

-- Define the total money made from single and double rooms.
def money_from_single_rooms : ℕ := single_rooms * single_room_cost
def money_from_double_rooms : ℕ := double_rooms * double_room_cost

-- State the theorem we want to prove.
theorem total_money_made : 
  (money_from_single_rooms + money_from_double_rooms) = 14000 :=
  by
    sorry -- Proof is omitted.

end total_money_made_l241_241724


namespace blue_balls_initial_count_l241_241567

theorem blue_balls_initial_count (B : ℕ)
  (h1 : 15 - 3 = 12)
  (h2 : (B - 3) / 12 = 1 / 3) :
  B = 7 :=
sorry

end blue_balls_initial_count_l241_241567


namespace blue_black_pen_ratio_l241_241493

theorem blue_black_pen_ratio (B K R : ℕ) 
  (h1 : B + K + R = 31) 
  (h2 : B = 18) 
  (h3 : K = R + 5) : 
  B / Nat.gcd B K = 2 ∧ K / Nat.gcd B K = 1 := 
by 
  sorry

end blue_black_pen_ratio_l241_241493


namespace organic_fertilizer_prices_l241_241075

theorem organic_fertilizer_prices
  (x y : ℝ)
  (h1 : x - y = 100)
  (h2 : 2 * x + y = 1700) :
  x = 600 ∧ y = 500 :=
by {
  sorry
}

end organic_fertilizer_prices_l241_241075


namespace max_ab_of_tangent_circles_l241_241838

theorem max_ab_of_tangent_circles (a b : ℝ) 
  (hC1 : ∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4)
  (hC2 : ∀ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1)
  (h_tangent : a + b = 3) :
  ab ≤ 9 / 4 :=
by
  sorry

end max_ab_of_tangent_circles_l241_241838


namespace derivative_at_one_l241_241914

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end derivative_at_one_l241_241914


namespace pencils_profit_goal_l241_241134

theorem pencils_profit_goal (n : ℕ) (price_purchase price_sale cost_goal : ℚ) (purchase_quantity : ℕ) 
  (h1 : price_purchase = 0.10) 
  (h2 : price_sale = 0.25) 
  (h3 : cost_goal = 100) 
  (h4 : purchase_quantity = 1500) 
  (h5 : n * price_sale ≥ purchase_quantity * price_purchase + cost_goal) :
  n ≥ 1000 :=
sorry

end pencils_profit_goal_l241_241134


namespace bus_stop_time_l241_241409

theorem bus_stop_time
  (speed_without_stoppage : ℝ := 54)
  (speed_with_stoppage : ℝ := 45)
  (distance_diff : ℝ := speed_without_stoppage - speed_with_stoppage)
  (distance : ℝ := distance_diff)
  (speed_km_per_min : ℝ := speed_without_stoppage / 60) :
  distance / speed_km_per_min = 10 :=
by
  -- The proof steps would go here.
  sorry

end bus_stop_time_l241_241409


namespace crop_yield_solution_l241_241957

variable (x y : ℝ)

axiom h1 : 3 * x + 6 * y = 4.7
axiom h2 : 5 * x + 3 * y = 5.5

theorem crop_yield_solution :
  x = 0.9 ∧ y = 1/3 :=
by
  sorry

end crop_yield_solution_l241_241957


namespace biased_coin_probability_l241_241818

theorem biased_coin_probability :
  let P1 := 3 / 4
  let P2 := 1 / 2
  let P3 := 3 / 4
  let P4 := 2 / 3
  let P5 := 1 / 3
  let P6 := 2 / 5
  let P7 := 3 / 7
  P1 * P2 * P3 * P4 * P5 * P6 * P7 = 3 / 560 :=
by sorry

end biased_coin_probability_l241_241818


namespace simplify_expression_l241_241998

variable {x y z : ℝ} 
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)

theorem simplify_expression :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x * y * z)⁻¹ * (x + y + z)⁻¹ :=
sorry

end simplify_expression_l241_241998


namespace Dan_picked_9_plums_l241_241769

-- Define the constants based on the problem
def M : ℕ := 4 -- Melanie's plums
def S : ℕ := 3 -- Sally's plums
def T : ℕ := 16 -- Total plums picked

-- The number of plums Dan picked
def D : ℕ := T - (M + S)

-- The theorem we want to prove
theorem Dan_picked_9_plums : D = 9 := by
  sorry

end Dan_picked_9_plums_l241_241769


namespace goldfish_to_pretzels_ratio_l241_241087

theorem goldfish_to_pretzels_ratio :
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := kids * items_per_baggie
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  ratio = 4 :=
by
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := 16 * 22 -- or kids * items_per_baggie for clarity
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  show ratio = 4
  · sorry

end goldfish_to_pretzels_ratio_l241_241087


namespace problem_statement_l241_241126

theorem problem_statement {a b c d : ℝ} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ (36 / (a - d)) :=
by
  sorry -- proof is omitted according to the instructions

end problem_statement_l241_241126


namespace part_one_part_two_l241_241497

-- Part (1)
theorem part_one (m : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → (2 * m < x ∧ x < 1 → -1 ≤ x ∧ x ≤ 2 ∧ - (1 / 2) ≤ m)) → 
  (m ≥ - (1 / 2)) :=
by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ x : ℤ, (2 * m < x ∧ x < 1) ∧ (x < -1 ∨ x > 2)) ∧ 
  (∀ y : ℤ, (2 * m < y ∧ y < 1) ∧ (y < -1 ∨ y > 2) → y = x) → 
  (- (3 / 2) ≤ m ∧ m < -1) :=
by sorry

end part_one_part_two_l241_241497


namespace product_of_numbers_l241_241952

variable (x y : ℕ)

theorem product_of_numbers : x + y = 120 ∧ x - y = 6 → x * y = 3591 := by
  sorry

end product_of_numbers_l241_241952


namespace product_of_a_values_has_three_solutions_eq_20_l241_241849

noncomputable def f (x : ℝ) : ℝ := abs ((x^2 - 10 * x + 25) / (x - 5) - (x^2 - 3 * x) / (3 - x))

def has_three_solutions (a : ℝ) : Prop :=
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ abs (abs (f x1) - 5) = a ∧ abs (abs (f x2) - 5) = a ∧ abs (abs (f x3) - 5) = a)

theorem product_of_a_values_has_three_solutions_eq_20 :
  ∃ a1 a2 : ℝ, has_three_solutions a1 ∧ has_three_solutions a2 ∧ a1 * a2 = 20 :=
sorry

end product_of_a_values_has_three_solutions_eq_20_l241_241849


namespace tom_teaching_years_l241_241140

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l241_241140


namespace arithmetic_sequence_1001th_term_l241_241474

theorem arithmetic_sequence_1001th_term (p q : ℤ)
  (h1 : 9 - p = (2 * q - 5))
  (h2 : (3 * p - q + 7) - 9 = (2 * q - 5)) :
  p + (1000 * (2 * q - 5)) = 5004 :=
by
  sorry

end arithmetic_sequence_1001th_term_l241_241474


namespace cans_purchased_l241_241498

variable (N P T : ℕ)

theorem cans_purchased (N P T : ℕ) : N * (5 * (T - 1)) / P = 5 * N * (T - 1) / P :=
by
  sorry

end cans_purchased_l241_241498


namespace angle_rotation_l241_241618

theorem angle_rotation (initial_angle : ℝ) (rotation : ℝ) :
  initial_angle = 30 → rotation = 450 → 
  ∃ (new_angle : ℝ), new_angle = 60 :=
by
  sorry

end angle_rotation_l241_241618


namespace orchids_initially_l241_241910

-- Definitions and Conditions
def initial_orchids (current_orchids: ℕ) (cut_orchids: ℕ) : ℕ :=
  current_orchids + cut_orchids

-- Proof statement
theorem orchids_initially (current_orchids: ℕ) (cut_orchids: ℕ) : initial_orchids current_orchids cut_orchids = 3 :=
by 
  have h1 : current_orchids = 7 := sorry
  have h2 : cut_orchids = 4 := sorry
  have h3 : initial_orchids current_orchids cut_orchids = 7 + 4 := sorry
  have h4 : initial_orchids current_orchids cut_orchids = 3 := sorry
  sorry

end orchids_initially_l241_241910


namespace find_m_l241_241377

theorem find_m (a : ℕ → ℝ) (m : ℕ) (h_pos : m > 0) 
  (h_a0 : a 0 = 37) (h_a1 : a 1 = 72) (h_am : a m = 0)
  (h_rec : ∀ k, 1 ≤ k ∧ k ≤ m - 1 → a (k + 1) = a (k - 1) - 3 / a k) :
  m = 889 :=
sorry

end find_m_l241_241377


namespace lemonade_water_quarts_l241_241494

theorem lemonade_water_quarts :
  let ratioWaterLemon := (4 : ℕ) / (1 : ℕ)
  let totalParts := 4 + 1
  let totalVolumeInGallons := 3
  let quartsPerGallon := 4
  let totalVolumeInQuarts := totalVolumeInGallons * quartsPerGallon
  let volumePerPart := totalVolumeInQuarts / totalParts
  let volumeWater := 4 * volumePerPart
  volumeWater = 9.6 :=
by
  -- placeholder for actual proof
  sorry

end lemonade_water_quarts_l241_241494


namespace incorrect_ratio_implies_l241_241033

variable {a b c d : ℝ} (h : a * d = b * c) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

theorem incorrect_ratio_implies :
  ¬ (c / b = a / d) :=
sorry

end incorrect_ratio_implies_l241_241033


namespace find_a_l241_241609

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 2 ↔ |a * x + 2| < 6) → a = -4 :=
by
  intro h
  sorry

end find_a_l241_241609


namespace arithmetic_mean_correct_l241_241284

-- Define the expressions
def expr1 (x : ℤ) := x + 12
def expr2 (y : ℤ) := y
def expr3 (x : ℤ) := 3 * x
def expr4 := 18
def expr5 (x : ℤ) := 3 * x + 6

-- The condition as a hypothesis
def condition (x y : ℤ) : Prop := (expr1 x + expr2 y + expr3 x + expr4 + expr5 x) / 5 = 30

-- The theorem to prove
theorem arithmetic_mean_correct : condition 6 72 :=
sorry

end arithmetic_mean_correct_l241_241284


namespace f_odd_f_monotonic_range_of_x_l241_241373

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd : ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x := by
  sorry

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ := by
  sorry

theorem range_of_x (x : ℝ) : f (1 / (x - 3)) + f (- 1 / 3) < 0 → x < 2 ∨ x > 6 := by
  sorry

end f_odd_f_monotonic_range_of_x_l241_241373


namespace find_chord_points_l241_241971

/-
Define a parabola and check if the points given form a chord that intersects 
the point (8,4) in the ratio 1:4.
-/

def parabola (P : ℝ × ℝ) : Prop :=
  P.snd^2 = 4 * P.fst

def divides_in_ratio (C A B : ℝ × ℝ) (m n : ℝ) : Prop :=
  (A.fst * n + B.fst * m = C.fst * (m + n)) ∧ 
  (A.snd * n + B.snd * m = C.snd * (m + n))

theorem find_chord_points :
  ∃ (P1 P2 : ℝ × ℝ),
  parabola P1 ∧
  parabola P2 ∧
  divides_in_ratio (8, 4) P1 P2 1 4 ∧ 
  ((P1 = (1, 2) ∧ P2 = (36, 12)) ∨ (P1 = (9, 6) ∧ P2 = (4, -4))) :=
sorry

end find_chord_points_l241_241971


namespace parabola_equation_l241_241619

variables (a b c p : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : p > 0)
variables (h_eccentricity : c / a = 2)
variables (h_b : b = Real.sqrt (3) * a)
variables (h_c : c = Real.sqrt (a^2 + b^2))
variables (d : ℝ) (h_distance : d = 2) (h_d_formula : d = (a * p) / (2 * c))

theorem parabola_equation (h : (a > 0) ∧ (b > 0) ∧ (p > 0) ∧ (c / a = 2) ∧ (b = (Real.sqrt 3) * a) ∧ (c = Real.sqrt (a^2 + b^2)) ∧ (d = 2) ∧ (d = (a * p) / (2 * c))) : x^2 = 16 * y :=
by {
  -- Lean does not require an actual proof here, so we use sorry.
  sorry
}

end parabola_equation_l241_241619


namespace find_f_of_2_l241_241012

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^3 + x^2 else 0

theorem find_f_of_2 :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, x < 0 → f x = x^3 + x^2) → f 2 = 4 :=
by
  intros h_odd h_def_neg
  sorry

end find_f_of_2_l241_241012


namespace quadratic_has_two_distinct_real_roots_l241_241564

theorem quadratic_has_two_distinct_real_roots :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 ∧ a*x^2 + b*x + c = 0 → (b^2 - 4*a*c) > 0 := 
sorry

end quadratic_has_two_distinct_real_roots_l241_241564


namespace initial_percentage_of_jasmine_water_l241_241186

-- Definitions
def v_initial : ℝ := 80
def v_jasmine_added : ℝ := 8
def v_water_added : ℝ := 12
def percentage_final : ℝ := 16
def v_final : ℝ := v_initial + v_jasmine_added + v_water_added

-- Lean 4 statement that frames the proof problem
theorem initial_percentage_of_jasmine_water (P : ℝ) :
  (P / 100) * v_initial + v_jasmine_added = (percentage_final / 100) * v_final → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_jasmine_water_l241_241186


namespace ordered_pair_count_l241_241392

noncomputable def count_pairs (n : ℕ) (c : ℕ) : ℕ := 
  (if n < c then 0 else n - c + 1)

theorem ordered_pair_count :
  (count_pairs 39 5 = 35) :=
sorry

end ordered_pair_count_l241_241392


namespace Ben_shirts_is_15_l241_241846

variable (Alex_shirts Joe_shirts Ben_shirts : Nat)

def Alex_has_4 : Alex_shirts = 4 := by sorry

def Joe_has_more_than_Alex : Joe_shirts = Alex_shirts + 3 := by sorry

def Ben_has_more_than_Joe : Ben_shirts = Joe_shirts + 8 := by sorry

theorem Ben_shirts_is_15 (h1 : Alex_shirts = 4) (h2 : Joe_shirts = Alex_shirts + 3) (h3 : Ben_shirts = Joe_shirts + 8) : Ben_shirts = 15 := by
  sorry

end Ben_shirts_is_15_l241_241846


namespace reciprocal_of_neg3_l241_241594

theorem reciprocal_of_neg3 : 1 / (-3: ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l241_241594


namespace red_pairs_count_l241_241970

def num_green_students : Nat := 63
def num_red_students : Nat := 69
def total_pairs : Nat := 66
def num_green_pairs : Nat := 27

theorem red_pairs_count : 
  (num_red_students - (num_green_students - num_green_pairs * 2)) / 2 = 30 := 
by sorry

end red_pairs_count_l241_241970


namespace nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l241_241997

theorem nat_forms_6n_plus_1_or_5 (x : ℕ) (h1 : ¬ (x % 2 = 0) ∧ ¬ (x % 3 = 0)) :
  ∃ n : ℕ, x = 6 * n + 1 ∨ x = 6 * n + 5 := 
sorry

theorem prod_6n_plus_1 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 1) = 6 * (6 * m * n + m + n) + 1 :=
sorry

theorem prod_6n_plus_5 (m n : ℕ) :
  (6 * m + 5) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + 5 * n + 4) + 1 :=
sorry

theorem prod_6n_plus_1_and_5 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + n) + 5 :=
sorry

end nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l241_241997


namespace ratio_fifth_terms_l241_241959

variable (a_n b_n S_n T_n : ℕ → ℚ)

-- Conditions
variable (h : ∀ n, S_n n / T_n n = (9 * n + 2) / (n + 7))

-- Define the 5th term
def a_5 (S_n : ℕ → ℚ) : ℚ := S_n 9 / 9
def b_5 (T_n : ℕ → ℚ) : ℚ := T_n 9 / 9

-- Prove that the ratio of the 5th terms is 83 / 16
theorem ratio_fifth_terms :
  (a_5 S_n) / (b_5 T_n) = 83 / 16 :=
by
  sorry

end ratio_fifth_terms_l241_241959


namespace Maxwell_age_l241_241445

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end Maxwell_age_l241_241445


namespace cara_neighbors_l241_241591

theorem cara_neighbors (friends : Finset Person) (mark : Person) (cara : Person) (h_mark : mark ∈ friends) (h_len : friends.card = 8) :
  ∃ pairs : Finset (Person × Person), pairs.card = 6 ∧
    ∀ (p : Person × Person), p ∈ pairs → p.1 = mark ∨ p.2 = mark :=
by
  -- The proof goes here.
  sorry

end cara_neighbors_l241_241591


namespace find_element_in_A_l241_241630

def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

def f (p : A) : B := (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem find_element_in_A : ∃ p : A, f p = (3, 1) ∧ p = (1, 1) := by
  sorry

end find_element_in_A_l241_241630


namespace recurring_decimal_to_fraction_l241_241643

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end recurring_decimal_to_fraction_l241_241643


namespace minimum_distance_between_extrema_is_2_sqrt_pi_l241_241002

noncomputable def minimum_distance_adjacent_extrema (a : ℝ) (h : a > 0) : ℝ := 2 * Real.sqrt Real.pi

theorem minimum_distance_between_extrema_is_2_sqrt_pi (a : ℝ) (h : a > 0) :
  minimum_distance_adjacent_extrema a h = 2 * Real.sqrt Real.pi := 
sorry

end minimum_distance_between_extrema_is_2_sqrt_pi_l241_241002


namespace find_equation_for_second_machine_l241_241091

theorem find_equation_for_second_machine (x : ℝ) : 
  (1 / 6) + (1 / x) = 1 / 3 ↔ (x = 6) := 
by 
  sorry

end find_equation_for_second_machine_l241_241091


namespace inscribed_circle_implies_rhombus_l241_241738

theorem inscribed_circle_implies_rhombus (AB : ℝ) (AD : ℝ)
  (h_parallelogram : AB = CD ∧ AD = BC) 
  (h_inscribed : AB + CD = AD + BC) : 
  AB = AD := by
  sorry

end inscribed_circle_implies_rhombus_l241_241738


namespace log_identity_l241_241683

theorem log_identity (c b : ℝ) (h1 : c = Real.log 81 / Real.log 4) (h2 : b = Real.log 3 / Real.log 2) : c = 2 * b := by
  sorry

end log_identity_l241_241683


namespace surface_area_of_rectangular_prism_l241_241657

def SurfaceArea (length : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  2 * ((length * width) + (width * height) + (height * length))

theorem surface_area_of_rectangular_prism 
  (l w h : ℕ) 
  (hl : l = 1) 
  (hw : w = 2) 
  (hh : h = 2) : 
  SurfaceArea l w h = 16 := by
  sorry

end surface_area_of_rectangular_prism_l241_241657


namespace rolls_sold_to_uncle_l241_241463

theorem rolls_sold_to_uncle (total_rolls : ℕ) (rolls_grandmother : ℕ) (rolls_neighbor : ℕ) (rolls_remaining : ℕ) (rolls_uncle : ℕ) :
  total_rolls = 12 →
  rolls_grandmother = 3 →
  rolls_neighbor = 3 →
  rolls_remaining = 2 →
  rolls_uncle = total_rolls - rolls_remaining - (rolls_grandmother + rolls_neighbor) →
  rolls_uncle = 4 :=
by
  intros h_total h_grandmother h_neighbor h_remaining h_compute
  rw [h_total, h_grandmother, h_neighbor, h_remaining] at h_compute
  exact h_compute

end rolls_sold_to_uncle_l241_241463


namespace express_inequality_l241_241359

theorem express_inequality (x : ℝ) : x + 4 ≥ -1 := sorry

end express_inequality_l241_241359


namespace students_not_in_biology_l241_241107

theorem students_not_in_biology (total_students : ℕ) (percent_enrolled : ℝ) (students_enrolled : ℕ) (students_not_enrolled : ℕ) : 
  total_students = 880 ∧ percent_enrolled = 32.5 ∧ total_students - students_enrolled = students_not_enrolled ∧ students_enrolled = 286 ∧ students_not_enrolled = 594 :=
by
  sorry

end students_not_in_biology_l241_241107


namespace problem_statement_l241_241764

noncomputable def count_propositions_and_true_statements 
  (statements : List String)
  (is_proposition : String → Bool)
  (is_true_proposition : String → Bool) 
  : Nat × Nat :=
  let props := statements.filter is_proposition
  let true_props := props.filter is_true_proposition
  (props.length, true_props.length)

theorem problem_statement : 
  (count_propositions_and_true_statements 
     ["Isn't an equilateral triangle an isosceles triangle?",
      "Are two lines perpendicular to the same line necessarily parallel?",
      "A number is either positive or negative",
      "What a beautiful coastal city Zhuhai is!",
      "If x + y is a rational number, then x and y are also rational numbers",
      "Construct △ABC ∼ △A₁B₁C₁"]
     (fun s => 
        s = "A number is either positive or negative" ∨ 
        s = "If x + y is a rational number, then x and y are also rational numbers")
     (fun s => false))
  = (2, 0) :=
by
  sorry

end problem_statement_l241_241764


namespace balloon_height_per_ounce_l241_241224

theorem balloon_height_per_ounce
    (total_money : ℕ)
    (sheet_cost : ℕ)
    (rope_cost : ℕ)
    (propane_cost : ℕ)
    (helium_price : ℕ)
    (max_height : ℕ)
    :
    total_money = 200 →
    sheet_cost = 42 →
    rope_cost = 18 →
    propane_cost = 14 →
    helium_price = 150 →
    max_height = 9492 →
    max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price) = 113 :=
by
  intros
  sorry

end balloon_height_per_ounce_l241_241224


namespace total_distance_traveled_l241_241454

theorem total_distance_traveled
  (bike_time_min : ℕ) (bike_rate_mph : ℕ)
  (jog_time_min : ℕ) (jog_rate_mph : ℕ)
  (total_time_min : ℕ)
  (h_bike_time : bike_time_min = 30)
  (h_bike_rate : bike_rate_mph = 6)
  (h_jog_time : jog_time_min = 45)
  (h_jog_rate : jog_rate_mph = 8)
  (h_total_time : total_time_min = 75) :
  (bike_rate_mph * bike_time_min / 60) + (jog_rate_mph * jog_time_min / 60) = 9 :=
by sorry

end total_distance_traveled_l241_241454


namespace math_problem_l241_241407

theorem math_problem :
  (625.3729 * (4500 + 2300 ^ 2) - Real.sqrt 84630) / (1500 ^ 3 * 48 ^ 2) = 0.0004257 :=
by
  sorry

end math_problem_l241_241407


namespace markup_percentage_l241_241052

theorem markup_percentage (PP SP SaleP : ℝ) (M : ℝ) (hPP : PP = 60) (h1 : SP = 60 + M * SP)
  (h2 : SaleP = SP * 0.8) (h3 : 4 = SaleP - PP) : M = 0.25 :=
by 
  sorry

end markup_percentage_l241_241052


namespace harry_less_than_half_selena_l241_241989

-- Definitions of the conditions
def selena_book_pages := 400
def harry_book_pages := 180
def half (n : ℕ) := n / 2

-- The theorem to prove that Harry's book is 20 pages less than half of Selena's book.
theorem harry_less_than_half_selena :
  harry_book_pages = half selena_book_pages - 20 := 
by
  sorry

end harry_less_than_half_selena_l241_241989


namespace parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l241_241669

section vector

variables {k : ℝ}
def a : ℝ × ℝ := (6, 2)
def b : ℝ × ℝ := (-2, k)

-- Parallel condition
theorem parallel_vectors : 
  (∀ c : ℝ, (6, 2) = -2 * (c * k, c)) → k = -2 / 3 :=
by 
  sorry

-- Perpendicular condition
theorem perpendicular_vectors : 
  6 * (-2) + 2 * k = 0 → k = 6 :=
by 
  sorry

-- Obtuse angle condition
theorem obtuse_angle_vectors : 
  6 * (-2) + 2 * k < 0 ∧ k ≠ -2 / 3 → k < 6 ∧ k ≠ -2 / 3 :=
by 
  sorry

end vector

end parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l241_241669


namespace parallelogram_area_l241_241367

theorem parallelogram_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let y_top := a
  let y_bottom := -b
  let x_left := -c + 2*y
  let x_right := d - 2*y 
  (d + c) * (a + b) = ad + ac + bd + bc :=
by
  sorry

end parallelogram_area_l241_241367


namespace sin_15_mul_sin_75_l241_241913

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 := 
by
  sorry

end sin_15_mul_sin_75_l241_241913


namespace solution_set_abs_inequality_l241_241955

theorem solution_set_abs_inequality (x : ℝ) : |3 - x| + |x - 7| ≤ 8 ↔ 1 ≤ x ∧ x ≤ 9 :=
sorry

end solution_set_abs_inequality_l241_241955


namespace wall_height_l241_241241

theorem wall_height (length width depth total_bricks: ℕ) (h: ℕ) (H_length: length = 20) (H_width: width = 4) (H_depth: depth = 2) (H_total_bricks: total_bricks = 800) :
  80 * depth * h = total_bricks → h = 5 :=
by
  intros H_eq
  sorry

end wall_height_l241_241241


namespace basketball_campers_l241_241200

theorem basketball_campers (total_campers soccer_campers football_campers : ℕ)
  (h_total : total_campers = 88)
  (h_soccer : soccer_campers = 32)
  (h_football : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 :=
by
  sorry

end basketball_campers_l241_241200


namespace total_cows_l241_241604

/-- A farmer divides his herd of cows among his four sons.
The first son receives 1/3 of the herd, the second son receives 1/6,
the third son receives 1/9, and the rest goes to the fourth son,
who receives 12 cows. Calculate the total number of cows in the herd
-/
theorem total_cows (n : ℕ) (h1 : (n : ℚ) * (1 / 3) + (n : ℚ) * (1 / 6) + (n : ℚ) * (1 / 9) + 12 = n) : n = 54 := by
  sorry

end total_cows_l241_241604


namespace integer_solutions_b_l241_241327

theorem integer_solutions_b (b : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ ∀ x : ℤ, x1 ≤ x ∧ x ≤ x2 → x^2 + b * x + 3 ≤ 0) ↔ b = -4 ∨ b = 4 := 
sorry

end integer_solutions_b_l241_241327


namespace min_value_N_l241_241120

theorem min_value_N (a b c d e f : ℤ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : 0 < e) (h₅ : 0 < f)
  (h_sum : a + b + c + d + e + f = 4020) :
  ∃ N : ℤ, N = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ N = 805 :=
by
  sorry

end min_value_N_l241_241120


namespace find_x_l241_241920

def vec := (ℝ × ℝ)

def a : vec := (1, 1)
def b (x : ℝ) : vec := (3, x)

def add_vec (v1 v2 : vec) : vec := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) (h : dot_product a (add_vec a (b x)) = 0) : x = -5 :=
by
  -- Proof steps (irrelevant for now)
  sorry

end find_x_l241_241920


namespace dichromate_molecular_weight_l241_241600

theorem dichromate_molecular_weight :
  let atomic_weight_Cr := 52.00
  let atomic_weight_O := 16.00
  let dichromate_num_Cr := 2
  let dichromate_num_O := 7
  (dichromate_num_Cr * atomic_weight_Cr + dichromate_num_O * atomic_weight_O) = 216.00 :=
by
  sorry

end dichromate_molecular_weight_l241_241600


namespace cone_lateral_area_l241_241316

noncomputable def lateral_area_of_cone (θ : ℝ) (r_base : ℝ) : ℝ :=
  if θ = 120 ∧ r_base = 2 then 
    12 * Real.pi 
  else 
    0 -- default case for the sake of definition, not used in our proof

theorem cone_lateral_area :
  lateral_area_of_cone 120 2 = 12 * Real.pi :=
by
  -- This is where the proof would go
  sorry

end cone_lateral_area_l241_241316


namespace y_intercept_of_linear_function_l241_241821

theorem y_intercept_of_linear_function 
  (k : ℝ)
  (h : (∃ k: ℝ, ∀ x y: ℝ, y = k * (x - 1) ∧ (x, y) = (-1, -2))) : 
  ∃ y : ℝ, (0, y) = (0, -1) :=
by {
  -- Skipping the proof as per the instruction
  sorry
}

end y_intercept_of_linear_function_l241_241821


namespace area_increase_l241_241188

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end area_increase_l241_241188


namespace min_speed_A_l241_241907

theorem min_speed_A (V_B V_C V_A : ℕ) (d_AB d_AC wind extra_speed : ℕ) :
  V_B = 50 →
  V_C = 70 →
  d_AB = 40 →
  d_AC = 280 →
  wind = 5 →
  V_A > ((d_AB * (V_A + wind + extra_speed)) / (d_AC - d_AB) - wind) :=
sorry

end min_speed_A_l241_241907


namespace almond_butter_servings_l241_241986

def convert_mixed_to_fraction (a b : ℤ) (n : ℕ) : ℚ :=
  (a * n + b) / n

def servings (total servings_fraction : ℚ) : ℚ :=
  total / servings_fraction

theorem almond_butter_servings :
  servings (convert_mixed_to_fraction 35 2 3) (convert_mixed_to_fraction 2 1 2) = 14 + 4 / 15 :=
by
  sorry

end almond_butter_servings_l241_241986


namespace width_of_paving_stone_l241_241275

-- Given conditions as definitions
def length_of_courtyard : ℝ := 40
def width_of_courtyard : ℝ := 16.5
def number_of_stones : ℕ := 132
def length_of_stone : ℝ := 2.5

-- Define the total area of the courtyard
def area_of_courtyard := length_of_courtyard * width_of_courtyard

-- Define the equation we need to prove
theorem width_of_paving_stone :
  (length_of_stone * W * number_of_stones = area_of_courtyard) → W = 2 :=
by
  sorry

end width_of_paving_stone_l241_241275


namespace inequality_xyz_l241_241299

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz / (x^3 + y^3 + xyz) + xyz / (y^3 + z^3 + xyz) + xyz / (z^3 + x^3 + xyz) ≤ 1) := by
  sorry

end inequality_xyz_l241_241299


namespace campers_went_rowing_and_hiking_in_all_l241_241193

def C_rm : Nat := 41
def C_hm : Nat := 4
def C_ra : Nat := 26

theorem campers_went_rowing_and_hiking_in_all : (C_rm + C_ra) + C_hm = 71 :=
by
  sorry

end campers_went_rowing_and_hiking_in_all_l241_241193


namespace ice_cream_arrangements_is_correct_l241_241516

-- Let us define the problem: counting the number of unique stacks of ice cream flavors
def ice_cream_scoops_arrangements : ℕ :=
  let total_scoops := 5
  let vanilla_scoops := 2
  Nat.factorial total_scoops / Nat.factorial vanilla_scoops

-- Assertion that needs to be proved
theorem ice_cream_arrangements_is_correct : ice_cream_scoops_arrangements = 60 := by
  -- Proof to be filled in; current placeholder
  sorry

end ice_cream_arrangements_is_correct_l241_241516


namespace cost_of_each_book_l241_241406

theorem cost_of_each_book 
  (B : ℝ)
  (num_books_plant : ℕ)
  (num_books_fish : ℕ)
  (num_magazines : ℕ)
  (cost_magazine : ℝ)
  (total_spent : ℝ) :
  num_books_plant = 9 →
  num_books_fish = 1 →
  num_magazines = 10 →
  cost_magazine = 2 →
  total_spent = 170 →
  10 * B + 10 * cost_magazine = total_spent →
  B = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cost_of_each_book_l241_241406


namespace solve_quadratic_eq_l241_241510

theorem solve_quadratic_eq (x : ℝ) : 4 * x ^ 2 - (x - 1) ^ 2 = 0 ↔ x = -1 ∨ x = 1 / 3 :=
by
  sorry

end solve_quadratic_eq_l241_241510


namespace simple_interest_time_l241_241621

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem simple_interest_time (SI CI : ℝ) (SI_given CI_given P_simp P_comp r_simp r_comp t_comp : ℝ) :
  SI = CI / 2 →
  CI = compound_interest P_comp r_comp 1 t_comp - P_comp →
  SI = simple_interest P_simp r_simp t_comp →
  P_simp = 1272 →
  r_simp = 0.10 →
  P_comp = 5000 →
  r_comp = 0.12 →
  t_comp = 2 →
  t_comp = 5 :=
by
  intros
  sorry

end simple_interest_time_l241_241621


namespace first_reduction_is_12_percent_l241_241938

theorem first_reduction_is_12_percent (P : ℝ) (x : ℝ) (h1 : (1 - x / 100) * 0.9 * P = 0.792 * P) : x = 12 :=
by
  sorry

end first_reduction_is_12_percent_l241_241938


namespace problem_statement_l241_241062

noncomputable def given_function (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - 2 * x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem problem_statement :
  is_even_function given_function ∧ smallest_positive_period given_function Real.pi :=
by
  sorry

end problem_statement_l241_241062


namespace tangent_lines_to_circle_passing_through_point_l241_241676

theorem tangent_lines_to_circle_passing_through_point :
  ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 1 → ((x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = -1)) :=
by
  sorry

end tangent_lines_to_circle_passing_through_point_l241_241676


namespace perimeter_greater_than_diagonals_l241_241057

namespace InscribedQuadrilateral

def is_convex_quadrilateral (AB BC CD DA AC BD: ℝ) : Prop :=
  -- Conditions for a convex quadrilateral (simple check)
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ AC > 0 ∧ BD > 0

def is_inscribed_in_circle (AB BC CD DA AC BD: ℝ) (r: ℝ) : Prop :=
  -- Check if quadrilateral is inscribed in a circle of radius 1
  r = 1

theorem perimeter_greater_than_diagonals 
  (AB BC CD DA AC BD: ℝ) 
  (r: ℝ)
  (h1 : is_convex_quadrilateral AB BC CD DA AC BD) 
  (h2 : is_inscribed_in_circle AB BC CD DA AC BD r) :
  0 < (AB + BC + CD + DA) - (AC + BD) ∧ (AB + BC + CD + DA) - (AC + BD) < 2 :=
by
  sorry 

end InscribedQuadrilateral

end perimeter_greater_than_diagonals_l241_241057


namespace total_wrappers_collected_l241_241743

theorem total_wrappers_collected :
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  Andy_wrappers + Max_wrappers + Zoe_wrappers = 74 :=
by
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  show Andy_wrappers + Max_wrappers + Zoe_wrappers = 74
  sorry

end total_wrappers_collected_l241_241743


namespace largest_K_inequality_l241_241690

theorem largest_K_inequality :
  ∃ K : ℕ, (K < 12) ∧ (10 * K = 110) := by
  use 11
  sorry

end largest_K_inequality_l241_241690


namespace find_train_speed_l241_241174

def train_speed (v t_pole t_stationary d_stationary : ℕ) : ℕ := v

theorem find_train_speed (v : ℕ) (t_pole : ℕ) (t_stationary : ℕ) (d_stationary : ℕ) :
  t_pole = 5 →
  t_stationary = 25 →
  d_stationary = 360 →
  25 * v = 5 * v + d_stationary →
  v = 18 :=
by intros h1 h2 h3 h4; sorry

end find_train_speed_l241_241174


namespace bobby_toy_cars_l241_241874

theorem bobby_toy_cars (initial_cars : ℕ) (increase_rate : ℕ → ℕ) (n : ℕ) :
  initial_cars = 16 →
  increase_rate 1 = initial_cars + (initial_cars / 2) →
  increase_rate 2 = increase_rate 1 + (increase_rate 1 / 2) →
  increase_rate 3 = increase_rate 2 + (increase_rate 2 / 2) →
  n = 3 →
  increase_rate n = 54 :=
by
  intros
  sorry

end bobby_toy_cars_l241_241874


namespace totalPlayers_l241_241145

def kabadiParticipants : ℕ := 50
def khoKhoParticipants : ℕ := 80
def soccerParticipants : ℕ := 30
def kabadiAndKhoKhoParticipants : ℕ := 15
def kabadiAndSoccerParticipants : ℕ := 10
def khoKhoAndSoccerParticipants : ℕ := 25
def allThreeParticipants : ℕ := 8

theorem totalPlayers : kabadiParticipants + khoKhoParticipants + soccerParticipants 
                       - kabadiAndKhoKhoParticipants - kabadiAndSoccerParticipants 
                       - khoKhoAndSoccerParticipants + allThreeParticipants = 118 :=
by 
  sorry

end totalPlayers_l241_241145


namespace depth_of_well_l241_241946

theorem depth_of_well 
  (t1 t2 : ℝ) 
  (d : ℝ) 
  (h1: t1 + t2 = 8) 
  (h2: d = 32 * t1^2) 
  (h3: t2 = d / 1100) 
  : d = 1348 := 
  sorry

end depth_of_well_l241_241946


namespace porche_project_time_l241_241239

theorem porche_project_time :
  let total_time := 180
  let math_time := 45
  let english_time := 30
  let science_time := 50
  let history_time := 25
  let homework_time := math_time + english_time + science_time + history_time 
  total_time - homework_time = 30 :=
by
  sorry

end porche_project_time_l241_241239


namespace gcd_of_28430_and_39674_l241_241798

theorem gcd_of_28430_and_39674 : Nat.gcd 28430 39674 = 2 := 
by 
  sorry

end gcd_of_28430_and_39674_l241_241798


namespace min_surveyed_consumers_l241_241262

theorem min_surveyed_consumers (N : ℕ) 
    (h10 : ∃ k : ℕ, N = 10 * k)
    (h30 : ∃ l : ℕ, N = 10 * l) 
    (h40 : ∃ m : ℕ, N = 5 * m) : 
    N = 10 :=
by
  sorry

end min_surveyed_consumers_l241_241262


namespace solve_four_tuple_l241_241622

-- Define the problem conditions
theorem solve_four_tuple (a b c d : ℝ) : 
    (ab + c + d = 3) → 
    (bc + d + a = 5) → 
    (cd + a + b = 2) → 
    (da + b + c = 6) → 
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by
  intros h1 h2 h3 h4
  sorry

end solve_four_tuple_l241_241622


namespace sum_of_three_squares_l241_241207

theorem sum_of_three_squares (n : ℕ) (h_pos : 0 < n) (h_square : ∃ m : ℕ, 3 * n + 1 = m^2) : ∃ x y z : ℕ, n + 1 = x^2 + y^2 + z^2 :=
by
  sorry

end sum_of_three_squares_l241_241207


namespace minimum_value_expression_l241_241046

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 4)

theorem minimum_value_expression : (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) = 192 := by
  sorry

end minimum_value_expression_l241_241046


namespace simplify_fraction_product_l241_241386

theorem simplify_fraction_product :
  8 * (15 / 14) * (-49 / 45) = - (28 / 3) :=
by
  sorry

end simplify_fraction_product_l241_241386


namespace Jill_water_volume_l241_241387

theorem Jill_water_volume 
  (n : ℕ) (h₀ : 3 * n = 48) :
  n * (1 / 4) + n * (1 / 2) + n * 1 = 28 := 
by 
  sorry

end Jill_water_volume_l241_241387


namespace selection_methods_l241_241722

theorem selection_methods (students lectures : ℕ) (h_stu : students = 4) (h_lect : lectures = 3) : 
  (lectures ^ students) = 81 := 
by
  rw [h_stu, h_lect]
  rfl

end selection_methods_l241_241722


namespace expression_for_B_A_greater_than_B_l241_241556

-- Define the polynomials A and B
def A (x : ℝ) := 3 * x^2 - 2 * x + 1
def B (x : ℝ) := 2 * x^2 - x - 3

-- Prove that the given expression for B validates the equation A + B = 5x^2 - 4x - 2.
theorem expression_for_B (x : ℝ) : A x + 2 * x^2 - x - 3 = 5 * x^2 - 4 * x - 2 :=
by {
  sorry
}

-- Prove that A is always greater than B for all values of x.
theorem A_greater_than_B (x : ℝ) : A x > B x :=
by {
  sorry
}

end expression_for_B_A_greater_than_B_l241_241556


namespace ellipse_equation_l241_241992

theorem ellipse_equation {a b : ℝ} 
  (center_origin : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x + y = 0)
  (foci_on_x : ∀ c : ℝ, c = a / 2)
  (perimeter_triangle : ∀ A B : ℝ, A + B + 2 * c = 16) :
  a = 4 ∧ b^2 = 12 → (∀ x y : ℝ, x^2/16 + y^2/12 = 1) :=
by
  sorry

end ellipse_equation_l241_241992


namespace part_a_part_b_l241_241415

-- Let γ and δ represent acute angles, γ < δ implies γ - sin γ < δ - sin δ 
theorem part_a (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  alpha - Real.sin alpha < beta - Real.sin beta := sorry

-- Let γ and δ represent acute angles, γ < δ implies tan γ - γ < tan δ - δ 
theorem part_b (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  Real.tan alpha - alpha < Real.tan beta - beta := sorry

end part_a_part_b_l241_241415


namespace prime_not_fourth_power_l241_241787

theorem prime_not_fourth_power (p : ℕ) (hp : p > 5) (prime : Prime p) : 
  ¬ ∃ a : ℕ, p = a^4 + 4 :=
by
  sorry

end prime_not_fourth_power_l241_241787


namespace combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l241_241663

theorem combined_sum_of_interior_numbers_of_eighth_and_ninth_rows :
  (2 ^ (8 - 1) - 2) + (2 ^ (9 - 1) - 2) = 380 :=
by
  -- The steps of the proof would go here, but for the purpose of this task:
  sorry

end combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l241_241663


namespace delphine_chocolates_l241_241031

theorem delphine_chocolates (x : ℕ) 
  (h1 : ∃ n, n = (2 * x - 3)) 
  (h2 : ∃ m, m = (x - 2))
  (h3 : ∃ p, p = (x - 3))
  (total_eq : x + (2 * x - 3) + (x - 2) + (x - 3) + 12 = 24) : 
  x = 4 := 
sorry

end delphine_chocolates_l241_241031


namespace triangle_formation_conditions_l241_241628

theorem triangle_formation_conditions (a b c : ℝ) :
  (a + b > c ∧ |a - b| < c) ↔ (a + b > c ∧ b + c > a ∧ c + a > b ∧ |a - b| < c ∧ |b - c| < a ∧ |c - a| < b) :=
sorry

end triangle_formation_conditions_l241_241628


namespace enclosed_area_is_43pi_l241_241858

noncomputable def enclosed_area (x y : ℝ) : Prop :=
  (x^2 - 6*x + y^2 + 10*y = 9)

theorem enclosed_area_is_43pi :
  (∃ x y : ℝ, enclosed_area x y) → 
  ∃ A : ℝ, A = 43 * Real.pi :=
by
  sorry

end enclosed_area_is_43pi_l241_241858


namespace treasure_probability_l241_241328

variable {Island : Type}

-- Define the probabilities.
def prob_treasure : ℚ := 1 / 3
def prob_trap : ℚ := 1 / 6
def prob_neither : ℚ := 1 / 2

-- Define the number of islands.
def num_islands : ℕ := 5

-- Define the probability of encountering exactly 4 islands with treasure and one with neither traps nor treasures.
theorem treasure_probability :
  (num_islands.choose 4) * (prob_ttreasure^4) * (prob_neither^1) = (5 : ℚ) * (1 / 81) * (1 / 2) :=
  by
  sorry

end treasure_probability_l241_241328


namespace sum_of_integers_l241_241958

theorem sum_of_integers (x y : ℤ) (h1 : x ^ 2 + y ^ 2 = 130) (h2 : x * y = 36) (h3 : x - y = 4) : x + y = 4 := 
by sorry

end sum_of_integers_l241_241958


namespace intersection_correct_union_correct_intersection_complement_correct_l241_241073

def U := ℝ
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}
def C_U_A : Set ℝ := {x | x ≤ 0 ∨ x > 2}
def C_U_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_correct : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

theorem union_correct : (A ∪ B) = {x : ℝ | x < -3 ∨ x > 0} :=
sorry

theorem intersection_complement_correct : (C_U_A ∩ C_U_B) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} :=
sorry

end intersection_correct_union_correct_intersection_complement_correct_l241_241073


namespace part_a_l241_241882

theorem part_a (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, a (n + 2) = a (n + 1) * a n + 1) :
  ∀ n, ¬ (4 ∣ a n) :=
by
  sorry

end part_a_l241_241882


namespace prime_sum_of_digits_base_31_l241_241270

-- Define the sum of digits function in base k
def sum_of_digits_in_base (k n : ℕ) : ℕ :=
  let digits := (Nat.digits k n)
  digits.foldr (· + ·) 0

theorem prime_sum_of_digits_base_31 (p : ℕ) (hp : Nat.Prime p) (h_bound : p < 20000) : 
  sum_of_digits_in_base 31 p = 49 ∨ sum_of_digits_in_base 31 p = 77 :=
by
  sorry

end prime_sum_of_digits_base_31_l241_241270


namespace g_of_f_three_l241_241985

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end g_of_f_three_l241_241985


namespace find_quarters_l241_241638

-- Define the conditions
def quarters_bounds (q : ℕ) : Prop :=
  8 < q ∧ q < 80

def stacks_mod4 (q : ℕ) : Prop :=
  q % 4 = 2

def stacks_mod6 (q : ℕ) : Prop :=
  q % 6 = 2

def stacks_mod8 (q : ℕ) : Prop :=
  q % 8 = 2

-- The theorem to prove
theorem find_quarters (q : ℕ) (h_bounds : quarters_bounds q) (h4 : stacks_mod4 q) (h6 : stacks_mod6 q) (h8 : stacks_mod8 q) : 
  q = 26 :=
by
  sorry

end find_quarters_l241_241638


namespace differential_equation_solution_l241_241135

def C1 : ℝ := sorry
def C2 : ℝ := sorry

noncomputable def y (x : ℝ) : ℝ := C1 * Real.cos x + C2 * Real.sin x
noncomputable def z (x : ℝ) : ℝ := -C1 * Real.sin x + C2 * Real.cos x

theorem differential_equation_solution : 
  (∀ x : ℝ, deriv y x = z x) ∧ 
  (∀ x : ℝ, deriv z x = -y x) :=
by
  sorry

end differential_equation_solution_l241_241135


namespace greg_initial_money_eq_36_l241_241178

theorem greg_initial_money_eq_36 
  (Earl_initial Fred_initial : ℕ)
  (Greg_initial : ℕ)
  (Earl_owes_Fred Fred_owes_Greg Greg_owes_Earl : ℕ)
  (Total_after_debt : ℕ)
  (hEarl_initial : Earl_initial = 90)
  (hFred_initial : Fred_initial = 48)
  (hEarl_owes_Fred : Earl_owes_Fred = 28)
  (hFred_owes_Greg : Fred_owes_Greg = 32)
  (hGreg_owes_Earl : Greg_owes_Earl = 40)
  (hTotal_after_debt : Total_after_debt = 130) :
  Greg_initial = 36 :=
sorry

end greg_initial_money_eq_36_l241_241178


namespace inequality_solution_set_l241_241566

theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 := by
sorry

end inequality_solution_set_l241_241566


namespace find_k_minus_r_l241_241733

theorem find_k_minus_r : 
  ∃ (k r : ℕ), k > 1 ∧ r < k ∧ 
  (1177 % k = r) ∧ (1573 % k = r) ∧ (2552 % k = r) ∧ 
  (k - r = 11) :=
sorry

end find_k_minus_r_l241_241733


namespace martin_spends_30_dollars_on_berries_l241_241879

def cost_per_package : ℝ := 2.0
def cups_per_package : ℝ := 1.0
def cups_per_day : ℝ := 0.5
def days : ℝ := 30

theorem martin_spends_30_dollars_on_berries :
  (days / (cups_per_package / cups_per_day)) * cost_per_package = 30 :=
by
  sorry

end martin_spends_30_dollars_on_berries_l241_241879


namespace sufficient_not_necessary_l241_241705

variable (a : ℝ)

theorem sufficient_not_necessary :
  (a > 1 → a^2 > a) ∧ (¬(a > 1) ∧ a^2 > a → a < 0) :=
by
  sorry

end sufficient_not_necessary_l241_241705


namespace find_constants_l241_241635

def equation1 (x p q : ℝ) : Prop := (x + p) * (x + q) * (x + 5) = 0
def equation2 (x p q : ℝ) : Prop := (x + 2 * p) * (x + 2) * (x + 3) = 0

def valid_roots1 (p q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ p q ∧ equation1 x₂ p q ∧
  x₁ = -5 ∨ x₁ = -q ∨ x₁ = -p

def valid_roots2 (p q : ℝ) : Prop :=
  ∃ x₃ x₄ : ℝ, x₃ ≠ x₄ ∧ equation2 x₃ p q ∧ equation2 x₄ p q ∧
  (x₃ = -2 * p ∨ x₃ = -2 ∨ x₃ = -3)

theorem find_constants (p q : ℝ) (h1 : valid_roots1 p q) (h2 : valid_roots2 p q) : 100 * p + q = 502 :=
by
  sorry

end find_constants_l241_241635


namespace find_remainder_l241_241288

theorem find_remainder : ∃ r : ℝ, r = 14 ∧ 13698 = (153.75280898876406 * 89) + r := 
by
  sorry

end find_remainder_l241_241288


namespace distinct_prime_factors_count_l241_241614

theorem distinct_prime_factors_count :
  ∀ (a b c d : ℕ),
  (a = 79) → (b = 3^4) → (c = 5 * 17) → (d = 3 * 29) →
  (∃ s : Finset ℕ, ∀ n ∈ s, Nat.Prime n ∧ 79 * 81 * 85 * 87 = s.prod id) :=
sorry

end distinct_prime_factors_count_l241_241614


namespace james_total_catch_l241_241065

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l241_241065


namespace first_term_of_geometric_sequence_l241_241712

theorem first_term_of_geometric_sequence (a r : ℚ) (h1 : a * r^2 = 12) (h2 : a * r^3 = 16) : a = 27 / 4 :=
by {
  sorry
}

end first_term_of_geometric_sequence_l241_241712


namespace magazines_cover_area_l241_241016

theorem magazines_cover_area (S : ℝ) (n : ℕ) (h_n_15 : n = 15) (h_cover : ∀ m ≤ n, ∃(Sm:ℝ), (Sm ≥ (m : ℝ) / n * S) ) :
  ∃ k : ℕ, k = n - 7 ∧ ∃ (Sk : ℝ), (Sk ≥ 8/15 * S) := 
by
  sorry

end magazines_cover_area_l241_241016


namespace central_angle_of_sector_l241_241582

theorem central_angle_of_sector (r A θ : ℝ) (hr : r = 2) (hA : A = 4) :
  θ = 2 :=
by
  sorry

end central_angle_of_sector_l241_241582


namespace n_squared_plus_one_divides_n_plus_one_l241_241745

theorem n_squared_plus_one_divides_n_plus_one (n : ℕ) (h : n^2 + 1 ∣ n + 1) : n = 1 :=
by
  sorry

end n_squared_plus_one_divides_n_plus_one_l241_241745


namespace solve_real_equation_l241_241417

theorem solve_real_equation (x : ℝ) (h : (x + 2)^4 + x^4 = 82) : x = 1 ∨ x = -3 :=
  sorry

end solve_real_equation_l241_241417


namespace joaozinho_multiplication_l241_241249

theorem joaozinho_multiplication :
  12345679 * 9 = 111111111 :=
by
  sorry

end joaozinho_multiplication_l241_241249


namespace carrey_fixed_amount_l241_241825

theorem carrey_fixed_amount :
  ∃ C : ℝ, 
    (C + 0.25 * 44.44444444444444 = 24 + 0.16 * 44.44444444444444) →
    C = 20 :=
by
  sorry

end carrey_fixed_amount_l241_241825


namespace ken_situps_l241_241076

variable (K : ℕ)

theorem ken_situps (h1 : Nathan = 2 * K)
                   (h2 : Bob = 3 * K / 2)
                   (h3 : Bob = K + 10) : 
                   K = 20 := 
by
  sorry

end ken_situps_l241_241076


namespace original_number_of_cats_l241_241160

theorem original_number_of_cats (C : ℕ) : 
  (C - 600) / 2 = 600 → C = 1800 :=
by
  sorry

end original_number_of_cats_l241_241160


namespace problem_equivalent_l241_241321

theorem problem_equivalent : ∀ m : ℝ, 2 * m^2 + m = -1 → 4 * m^2 + 2 * m + 5 = 3 := 
by
  intros m h
  sorry

end problem_equivalent_l241_241321


namespace car_travel_distance_l241_241502

theorem car_travel_distance (v d : ℕ) 
  (h1 : d = v * 7)
  (h2 : d = (v + 12) * 5) : 
  d = 210 := by 
  sorry

end car_travel_distance_l241_241502


namespace square_side_length_l241_241795

/-- Define OPEN as a square and T a point on side NO
    such that the areas of triangles TOP and TEN are 
    respectively 62 and 10. Prove that the side length 
    of the square is 12. -/
theorem square_side_length (s x y : ℝ) (T : x + y = s)
    (h1 : 0 < s) (h2 : 0 < x) (h3 : 0 < y)
    (a1 : 1 / 2 * x * s = 62)
    (a2 : 1 / 2 * y * s = 10) :
    s = 12 :=
by
    sorry

end square_side_length_l241_241795


namespace a100_gt_two_pow_99_l241_241154

theorem a100_gt_two_pow_99 
  (a : ℕ → ℤ) 
  (h1 : a 1 > a 0)
  (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l241_241154


namespace sum_first_five_terms_eq_15_l241_241035

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d 

variable (a : ℕ → ℝ) (h_arith_seq : is_arithmetic_sequence a) (h_a3 : a 3 = 3)

theorem sum_first_five_terms_eq_15 : (a 1 + a 2 + a 3 + a 4 + a 5 = 15) :=
sorry

end sum_first_five_terms_eq_15_l241_241035


namespace taylor_scores_l241_241820

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end taylor_scores_l241_241820


namespace soda_price_before_increase_l241_241894

theorem soda_price_before_increase
  (candy_box_after : ℝ)
  (soda_after : ℝ)
  (candy_box_increase : ℝ)
  (soda_increase : ℝ)
  (new_price_soda : soda_after = 9)
  (new_price_candy_box : candy_box_after = 10)
  (percent_candy_box_increase : candy_box_increase = 0.25)
  (percent_soda_increase : soda_increase = 0.50) :
  ∃ P : ℝ, 1.5 * P = 9 ∧ P = 6 := 
by
  sorry

end soda_price_before_increase_l241_241894


namespace math_problem_l241_241760

theorem math_problem
  (N O : ℝ)
  (h₁ : 96 / 100 = |(O - 5 * N) / (5 * N)|)
  (h₂ : 5 * N ≠ 0) :
  O = 0.2 * N :=
by
  sorry

end math_problem_l241_241760


namespace minimum_distance_from_circle_to_line_l241_241980

noncomputable def point_on_circle (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

def line_eq (p : ℝ × ℝ) : ℝ :=
  p.1 - p.2 + 4

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 4| / Real.sqrt (1^2 + 1^2)

theorem minimum_distance_from_circle_to_line :
  ∀ θ : ℝ, (∃ θ, distance_from_point_to_line (point_on_circle θ) = 2 * Real.sqrt 2 - 2) :=
by
  sorry

end minimum_distance_from_circle_to_line_l241_241980


namespace future_age_ratio_l241_241539

theorem future_age_ratio (j e x : ℕ) 
  (h1 : j - 3 = 5 * (e - 3)) 
  (h2 : j - 7 = 6 * (e - 7)) 
  (h3 : x = 17) : (j + x) / (e + x) = 3 := 
by
  sorry

end future_age_ratio_l241_241539


namespace valid_parameterizations_l241_241919

noncomputable def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

def is_valid_parameterization (p d : ℝ × ℝ) (m b : ℝ) : Prop :=
  lies_on_line p m b ∧ is_scalar_multiple d (2, 1)

theorem valid_parameterizations :
  (is_valid_parameterization (7, 18) (-1, -2) 2 4) ∧
  (is_valid_parameterization (1, 6) (5, 10) 2 4) ∧
  (is_valid_parameterization (2, 8) (20, 40) 2 4) ∧
  ¬ (is_valid_parameterization (-4, -4) (1, -1) 2 4) ∧
  ¬ (is_valid_parameterization (-3, -2) (0.5, 1) 2 4) :=
by {
  sorry
}

end valid_parameterizations_l241_241919


namespace simplify_336_to_fraction_l241_241863

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l241_241863


namespace probability_of_shaded_triangle_l241_241021

theorem probability_of_shaded_triangle 
  (triangles : Finset ℝ) 
  (shaded_triangles : Finset ℝ)
  (h1 : triangles = {1, 2, 3, 4, 5})
  (h2 : shaded_triangles = {1, 4})
  : (shaded_triangles.card / triangles.card) = 2 / 5 := 
  by
  sorry

end probability_of_shaded_triangle_l241_241021


namespace final_sum_l241_241693

-- Assuming an initial condition for the values on the calculators
def initial_values : List Int := [2, 1, -1]

-- Defining the operations to be applied on the calculators
def operations (vals : List Int) : List Int :=
  match vals with
  | [a, b, c] => [a * a, b * b * b, -c]
  | _ => vals  -- This case handles unexpected input formats

-- Applying the operations for 43 participants
def final_values (vals : List Int) (n : Nat) : List Int :=
  if n = 0 then vals
  else final_values (operations vals) (n - 1)

-- Prove that the final sum of the values on the calculators equals 2 ^ 2 ^ 43
theorem final_sum : 
  final_values initial_values 43 = [2 ^ 2 ^ 43, 1, -1] → 
  List.sum (final_values initial_values 43) = 2 ^ 2 ^ 43 :=
by
  intro h -- This introduces the hypothesis that the final values list equals the expected values
  sorry   -- Provide an ultimate proof for the statement.

end final_sum_l241_241693


namespace negation_proposition_l241_241455

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
sorry

end negation_proposition_l241_241455


namespace least_multiple_25_gt_500_l241_241966

theorem least_multiple_25_gt_500 : ∃ (k : ℕ), 25 * k > 500 ∧ (∀ m : ℕ, (25 * m > 500 → 25 * k ≤ 25 * m)) :=
by
  use 21
  sorry

end least_multiple_25_gt_500_l241_241966


namespace eval_nabla_l241_241170

def nabla (a b : ℕ) : ℕ := 3 + b^(a-1)

theorem eval_nabla : nabla (nabla 2 3) 4 = 1027 := by
  -- proof goes here
  sorry

end eval_nabla_l241_241170


namespace simplify_expression_l241_241213

theorem simplify_expression : 9 * (12 / 7) * ((-35) / 36) = -15 := by
  sorry

end simplify_expression_l241_241213


namespace square_root_properties_l241_241687

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end square_root_properties_l241_241687


namespace remainder_of_sum_of_squares_mod_8_l241_241785

theorem remainder_of_sum_of_squares_mod_8 :
  let a := 445876
  let b := 985420
  let c := 215546
  let d := 656452
  let e := 387295
  a % 8 = 4 → b % 8 = 4 → c % 8 = 6 → d % 8 = 4 → e % 8 = 7 →
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remainder_of_sum_of_squares_mod_8_l241_241785


namespace Raine_total_steps_l241_241240

-- Define the steps taken to and from school each day
def Monday_steps_to_school := 150
def Monday_steps_back := 170
def Tuesday_steps_to_school := 140
def Tuesday_steps_back := 140 + 30
def Wednesday_steps_to_school := 160
def Wednesday_steps_back := 210
def Thursday_steps_to_school := 150
def Thursday_steps_back := 140 + 30
def Friday_steps_to_school := 180
def Friday_steps_back := 200

-- Define total steps for each day
def Monday_total_steps := Monday_steps_to_school + Monday_steps_back
def Tuesday_total_steps := Tuesday_steps_to_school + Tuesday_steps_back
def Wednesday_total_steps := Wednesday_steps_to_school + Wednesday_steps_back
def Thursday_total_steps := Thursday_steps_to_school + Thursday_steps_back
def Friday_total_steps := Friday_steps_to_school + Friday_steps_back

-- Define the total steps for all five days
def total_steps :=
  Monday_total_steps +
  Tuesday_total_steps +
  Wednesday_total_steps +
  Thursday_total_steps +
  Friday_total_steps

-- Prove that the total steps equals 1700
theorem Raine_total_steps : total_steps = 1700 := 
by 
  unfold total_steps
  unfold Monday_total_steps Tuesday_total_steps Wednesday_total_steps Thursday_total_steps Friday_total_steps
  unfold Monday_steps_to_school Monday_steps_back
  unfold Tuesday_steps_to_school Tuesday_steps_back
  unfold Wednesday_steps_to_school Wednesday_steps_back
  unfold Thursday_steps_to_school Thursday_steps_back
  unfold Friday_steps_to_school Friday_steps_back
  sorry

end Raine_total_steps_l241_241240


namespace probability_of_distinct_dice_numbers_l241_241396

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l241_241396


namespace total_boys_went_down_slide_l241_241260

theorem total_boys_went_down_slide :
  let boys_first_10_minutes := 22
  let boys_next_5_minutes := 13
  let boys_last_20_minutes := 35
  (boys_first_10_minutes + boys_next_5_minutes + boys_last_20_minutes) = 70 :=
by
  sorry

end total_boys_went_down_slide_l241_241260


namespace min_value_expr_l241_241089

theorem min_value_expr (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 :=
by
  sorry

end min_value_expr_l241_241089


namespace club_membership_l241_241875

theorem club_membership (n : ℕ) 
  (h1 : n % 10 = 6)
  (h2 : n % 11 = 6)
  (h3 : 150 ≤ n)
  (h4 : n ≤ 300) : 
  n = 226 := 
sorry

end club_membership_l241_241875


namespace prod_eq_of_eqs_l241_241983

variable (a : ℝ) (m n p q : ℕ)
variable (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1)
variable (h4 : a^m + a^n = a^p + a^q) (h5 : a^{3*m} + a^{3*n} = a^{3*p} + a^{3*q})

theorem prod_eq_of_eqs : m * n = p * q := by
  sorry

end prod_eq_of_eqs_l241_241983


namespace S_of_1_eq_8_l241_241841

variable (x : ℝ)

-- Definition of original polynomial R(x)
def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

-- Definition of new polynomial S(x) created by adding 2 to each coefficient of R(x)
def S (x : ℝ) : ℝ := 5 * x^3 - 3 * x + 6

-- The theorem we want to prove
theorem S_of_1_eq_8 : S 1 = 8 := by
  sorry

end S_of_1_eq_8_l241_241841


namespace box_dimensions_correct_l241_241324

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l241_241324


namespace range_of_f_l241_241725

noncomputable def g (x : ℝ) := 15 - 2 * Real.cos (2 * x) - 4 * Real.sin x

noncomputable def f (x : ℝ) := Real.sqrt (g x ^ 2 - 245)

theorem range_of_f : (Set.range f) = Set.Icc 0 14 := sorry

end range_of_f_l241_241725


namespace find_x_of_floor_eq_72_l241_241209

theorem find_x_of_floor_eq_72 (x : ℝ) (hx_pos : 0 < x) (hx_eq : x * ⌊x⌋ = 72) : x = 9 :=
by 
  sorry

end find_x_of_floor_eq_72_l241_241209


namespace number_of_extreme_value_points_l241_241103

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem number_of_extreme_value_points : ∃! c : ℝ, c > 0 ∧ (deriv f c = 0) :=
by
  sorry

end number_of_extreme_value_points_l241_241103


namespace cubic_poly_sum_l241_241794

noncomputable def q (x : ℕ) : ℤ := sorry

axiom h0 : q 1 = 5
axiom h1 : q 6 = 24
axiom h2 : q 10 = 16
axiom h3 : q 15 = 34

theorem cubic_poly_sum :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) +
  (q 7) + (q 8) + (q 9) + (q 10) + (q 11) + (q 12) + (q 13) +
  (q 14) + (q 15) + (q 16) = 340 :=
by
  sorry

end cubic_poly_sum_l241_241794


namespace ratio_of_DE_EC_l241_241357

noncomputable def ratio_DE_EC (a x : ℝ) : ℝ :=
  let DE := a - x
  x / DE

theorem ratio_of_DE_EC (a : ℝ) (H1 : ∀ x, x = 5 * a / 7) :
  ratio_DE_EC a (5 * a / 7) = 5 / 2 :=
by
  sorry

end ratio_of_DE_EC_l241_241357


namespace directrix_of_parabola_l241_241490

-- Define the variables and constants
variables (x y a : ℝ) (h₁ : x^2 = 4 * a * y) (h₂ : x = -2) (h₃ : y = 1)

theorem directrix_of_parabola (h : (-2)^2 = 4 * a * 1) : y = -1 := 
by
  -- Our proof will happen here, but we omit the details
  sorry

end directrix_of_parabola_l241_241490


namespace third_vertex_y_coordinate_correct_l241_241855

noncomputable def third_vertex_y_coordinate (x1 y1 x2 y2 : ℝ) (h : y1 = y2) (h_dist : |x1 - x2| = 10) : ℝ :=
  y1 + 5 * Real.sqrt 3

theorem third_vertex_y_coordinate_correct : 
  third_vertex_y_coordinate 3 4 13 4 rfl (by norm_num) = 4 + 5 * Real.sqrt 3 :=
by
  sorry

end third_vertex_y_coordinate_correct_l241_241855


namespace correct_phone_call_sequence_l241_241162

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end correct_phone_call_sequence_l241_241162


namespace sum_modulo_seven_l241_241187

theorem sum_modulo_seven :
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999
  s % 7 = 2 :=
by
  sorry

end sum_modulo_seven_l241_241187


namespace lollipop_problem_l241_241231

def Henry_lollipops (A : Nat) : Nat := A + 30
def Diane_lollipops (A : Nat) : Nat := 2 * A
def Total_days (H A D : Nat) (daily_rate : Nat) : Nat := (H + A + D) / daily_rate

theorem lollipop_problem
  (A : Nat) (H : Nat) (D : Nat) (daily_rate : Nat)
  (h₁ : A = 60)
  (h₂ : H = Henry_lollipops A)
  (h₃ : D = Diane_lollipops A)
  (h₄ : daily_rate = 45)
  : Total_days H A D daily_rate = 6 := by
  sorry

end lollipop_problem_l241_241231


namespace any_power_ends_in_12890625_l241_241953

theorem any_power_ends_in_12890625 (a : ℕ) (m k : ℕ) (h : a = 10^m * k + 12890625) : ∀ (n : ℕ), 0 < n → ((a ^ n) % 10^8 = 12890625 % 10^8) :=
by
  intros
  sorry

end any_power_ends_in_12890625_l241_241953


namespace add_candies_to_equalize_l241_241159

-- Define the initial number of candies in basket A and basket B
def candiesInA : ℕ := 8
def candiesInB : ℕ := 17

-- Problem statement: Prove that adding 9 more candies to basket A
-- makes the number of candies in basket A equal to that in basket B.
theorem add_candies_to_equalize : ∃ n : ℕ, candiesInA + n = candiesInB :=
by
  use 9  -- The value we are adding to the candies in basket A
  sorry  -- Proof goes here

end add_candies_to_equalize_l241_241159


namespace running_speed_equiv_l241_241578

variable (R : ℝ)
variable (walking_speed : ℝ) (total_distance : ℝ) (total_time: ℝ) (distance_walked : ℝ) (distance_ran : ℝ)

theorem running_speed_equiv :
  walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4 →
  1 + (4 / R) = 1.5 →
  R = 8 :=
by
  intros H1 H2
  -- H1: Condition set (walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4)
  -- H2: Equation (1 + (4 / R) = 1.5)
  sorry

end running_speed_equiv_l241_241578


namespace no_nonzero_solutions_l241_241026

theorem no_nonzero_solutions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x = y^2 - y) ∧ (y^2 + y = z^2 - z) ∧ (z^2 + z = x^2 - x) → false :=
by
  sorry

end no_nonzero_solutions_l241_241026


namespace equivalence_condition_l241_241697

theorem equivalence_condition (a b c d : ℝ) (h : (a + b) / (b + c) = (c + d) / (d + a)) : 
  a = c ∨ a + b + c + d = 0 :=
sorry

end equivalence_condition_l241_241697


namespace avg_age_new_students_l241_241440

-- Definitions for the conditions
def initial_avg_age : ℕ := 14
def initial_student_count : ℕ := 10
def new_student_count : ℕ := 5
def new_avg_age : ℕ := initial_avg_age + 1

-- Lean statement for the proof problem
theorem avg_age_new_students :
  (initial_avg_age * initial_student_count + new_avg_age * new_student_count) / new_student_count = 17 :=
by
  sorry

end avg_age_new_students_l241_241440


namespace makes_at_least_one_shot_l241_241845
noncomputable section

/-- The probability of making the free throw. -/
def free_throw_make_prob : ℚ := 4/5

/-- The probability of making the high school 3-pointer. -/
def high_school_make_prob : ℚ := 1/2

/-- The probability of making the professional 3-pointer. -/
def pro_make_prob : ℚ := 1/3

/-- The probability of making at least one of the three shots. -/
theorem makes_at_least_one_shot :
  (1 - ((1 - free_throw_make_prob) * (1 - high_school_make_prob) * (1 - pro_make_prob))) = 14 / 15 :=
by
  sorry

end makes_at_least_one_shot_l241_241845


namespace worst_player_is_son_l241_241166

-- Define the types of players and relationships
inductive Sex
| male
| female

structure Player where
  name : String
  sex : Sex
  age : Nat

-- Define the four players
def woman := Player.mk "woman" Sex.female 30  -- Age is arbitrary
def brother := Player.mk "brother" Sex.male 30
def son := Player.mk "son" Sex.male 10
def daughter := Player.mk "daughter" Sex.female 10

-- Define the conditions
def opposite_sex (p1 p2 : Player) : Prop := p1.sex ≠ p2.sex
def same_age (p1 p2 : Player) : Prop := p1.age = p2.age

-- Define the worst player and the best player
variable (worst_player : Player) (best_player : Player)

-- Conditions as hypotheses
axiom twin_condition : ∃ twin : Player, (twin ≠ worst_player) ∧ (opposite_sex twin best_player)
axiom age_condition : same_age worst_player best_player
axiom not_same_player : worst_player ≠ best_player

-- Prove that the worst player is the son
theorem worst_player_is_son : worst_player = son :=
by
  sorry

end worst_player_is_son_l241_241166


namespace num_perfect_square_factors_l241_241459

-- Define the exponents and their corresponding number of perfect square factors
def num_square_factors (exp : ℕ) : ℕ := exp / 2 + 1

-- Define the product of the prime factorization
def product : ℕ := 2^12 * 3^15 * 7^18

-- State the theorem
theorem num_perfect_square_factors :
  (num_square_factors 12) * (num_square_factors 15) * (num_square_factors 18) = 560 := by
  sorry

end num_perfect_square_factors_l241_241459


namespace smallest_m_inequality_l241_241343

theorem smallest_m_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l241_241343


namespace sum_of_c_l241_241899

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℕ :=
  2^(n - 1)

-- Define the sequence c_n
def c (n : ℕ) : ℕ :=
  a n * b n

-- Define the sum S_n of the first n terms of c_n
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => c (i + 1))

-- The main Lean statement
theorem sum_of_c (n : ℕ) : S n = 3 + (n - 1) * 2^(n + 1) :=
  sorry

end sum_of_c_l241_241899


namespace paco_initial_cookies_l241_241584

theorem paco_initial_cookies (cookies_ate : ℕ) (cookies_left : ℕ) (cookies_initial : ℕ) 
  (h1 : cookies_ate = 15) (h2 : cookies_left = 78) :
  cookies_initial = cookies_ate + cookies_left → cookies_initial = 93 :=
by
  sorry

end paco_initial_cookies_l241_241584


namespace books_in_series_l241_241227

theorem books_in_series (books_watched : ℕ) (movies_watched : ℕ) (read_more_movies_than_books : books_watched + 3 = movies_watched) (watched_movies : movies_watched = 19) : books_watched = 16 :=
by sorry

end books_in_series_l241_241227


namespace remaining_blocks_to_walk_l241_241235

noncomputable def total_blocks : ℕ := 11 + 6 + 8
noncomputable def walked_blocks : ℕ := 5

theorem remaining_blocks_to_walk : total_blocks - walked_blocks = 20 := by
  sorry

end remaining_blocks_to_walk_l241_241235


namespace wire_ratio_is_one_l241_241266

theorem wire_ratio_is_one (a b : ℝ) (h1 : a = b) : a / b = 1 := by
  -- The proof goes here
  sorry

end wire_ratio_is_one_l241_241266


namespace remaining_cookies_l241_241865

variable (total_initial_cookies : ℕ)
variable (cookies_taken_day1 : ℕ := 3)
variable (cookies_taken_day2 : ℕ := 3)
variable (cookies_eaten_day2 : ℕ := 1)
variable (cookies_put_back_day2 : ℕ := 2)
variable (cookies_taken_by_junior : ℕ := 7)

theorem remaining_cookies (total_initial_cookies cookies_taken_day1 cookies_taken_day2
                          cookies_eaten_day2 cookies_put_back_day2 cookies_taken_by_junior : ℕ) :
  (total_initial_cookies = 2 * (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior))
  → (total_initial_cookies - (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior) = 11) :=
by
  sorry

end remaining_cookies_l241_241865


namespace probability_B_does_not_lose_l241_241884

def prob_A_wins : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Theorem: the probability that B does not lose is 70%.
theorem probability_B_does_not_lose : prob_A_wins + prob_draw ≤ 1 → 1 - prob_A_wins - (1 - prob_draw - prob_A_wins) = 0.7 := by
  sorry

end probability_B_does_not_lose_l241_241884


namespace trench_digging_l241_241937

theorem trench_digging 
  (t : ℝ) (T : ℝ) (work_units : ℝ)
  (h1 : 4 * t = 10)
  (h2 : T = 5 * t) :
  work_units = 80 :=
by
  sorry

end trench_digging_l241_241937


namespace eq_irrational_parts_l241_241411

theorem eq_irrational_parts (a b c d : ℝ) (h : a + b * (Real.sqrt 5) = c + d * (Real.sqrt 5)) : a = c ∧ b = d := 
by 
  sorry

end eq_irrational_parts_l241_241411


namespace line_intersects_y_axis_l241_241104

-- Define the points
def P1 : ℝ × ℝ := (3, 18)
def P2 : ℝ × ℝ := (-9, -6)

-- State that the line passing through P1 and P2 intersects the y-axis at (0, 12)
theorem line_intersects_y_axis :
  ∃ y : ℝ, (∃ m b : ℝ, ∀ x : ℝ, y = m * x + b ∧ (m = (P2.2 - P1.2) / (P2.1 - P1.1)) ∧ (P1.2 = m * P1.1 + b) ∧ (x = 0) ∧ y = 12) :=
sorry

end line_intersects_y_axis_l241_241104


namespace smallest_possible_denominator_l241_241581

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end smallest_possible_denominator_l241_241581


namespace initial_distance_between_fred_and_sam_l241_241806

-- Define the conditions as parameters
variables (initial_distance : ℝ)
          (fred_speed sam_speed meeting_distance : ℝ)
          (h_fred_speed : fred_speed = 5)
          (h_sam_speed : sam_speed = 5)
          (h_meeting_distance : meeting_distance = 25)

-- State the theorem
theorem initial_distance_between_fred_and_sam :
  initial_distance = meeting_distance + meeting_distance :=
by
  -- Inline proof structure (sorry means the proof is omitted here)
  sorry

end initial_distance_between_fred_and_sam_l241_241806


namespace anthony_more_shoes_than_jim_l241_241816

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem anthony_more_shoes_than_jim : (anthony_shoes - jim_shoes) = 2 :=
by
  sorry

end anthony_more_shoes_than_jim_l241_241816


namespace range_of_a_l241_241911

noncomputable def f (x a : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x
noncomputable def f' (x a : ℝ) : ℝ := 1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ 0) ↔ -1 / 3 ≤ a ∧ a ≤ 1 / 3 :=
sorry

end range_of_a_l241_241911


namespace g_at_3_l241_241319

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2

theorem g_at_3 : g 3 = 0 :=
by
  sorry

end g_at_3_l241_241319


namespace find_square_sum_l241_241695

theorem find_square_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2 / 7 :=
by
  sorry

end find_square_sum_l241_241695


namespace park_area_l241_241043

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end park_area_l241_241043


namespace find_certain_number_l241_241659

theorem find_certain_number (x : ℕ) (h1 : 172 = 4 * 43) (h2 : 43 - 172 / x = 28) (h3 : 172 % x = 7) : x = 11 := by
  sorry

end find_certain_number_l241_241659


namespace correct_samples_for_senior_l241_241311

-- Define the total number of students in each section
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_students : ℕ := junior_students + senior_students

-- Define the total number of samples to be drawn
def total_samples : ℕ := 60

-- Calculate the number of samples to be drawn from each section
def junior_samples : ℕ := total_samples * junior_students / total_students
def senior_samples : ℕ := total_samples - junior_samples

-- The theorem to prove
theorem correct_samples_for_senior :
  senior_samples = 20 :=
by
  sorry

end correct_samples_for_senior_l241_241311


namespace range_of_f_ge_1_l241_241674

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (x + 1) ^ 2 else 4 - Real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end range_of_f_ge_1_l241_241674


namespace number_of_girls_in_school_l241_241525

/-- Statement: There are 408 boys and some girls in a school which are to be divided into equal sections
of either boys or girls alone. The total number of sections thus formed is 26. Prove that the number 
of girls is 216. -/
theorem number_of_girls_in_school (n : ℕ) (n_boys : ℕ := 408) (total_sections : ℕ := 26)
  (h1 : n_boys = 408)
  (h2 : ∃ b g : ℕ, b + g = total_sections ∧ 408 / b = n / g ∧ b ∣ 408 ∧ g ∣ n) :
  n = 216 :=
by
  -- Proof would go here
  sorry

end number_of_girls_in_school_l241_241525


namespace perimeter_of_square_field_l241_241364

variable (s a p : ℕ)

-- Given conditions as definitions
def area_eq_side_squared (a s : ℕ) : Prop := a = s^2
def perimeter_eq_four_sides (p s : ℕ) : Prop := p = 4 * s
def given_equation (a p : ℕ) : Prop := 6 * a = 6 * (2 * p + 9)

-- The proof statement
theorem perimeter_of_square_field (s a p : ℕ) 
  (h1 : area_eq_side_squared a s)
  (h2 : perimeter_eq_four_sides p s)
  (h3 : given_equation a p) :
  p = 36 :=
by
  sorry

end perimeter_of_square_field_l241_241364


namespace collinear_points_l241_241217

axiom collinear (A B C : ℝ × ℝ × ℝ) : Prop

theorem collinear_points (c d : ℝ) (h : collinear (2, c, d) (c, 3, d) (c, d, 4)) : c + d = 6 :=
sorry

end collinear_points_l241_241217


namespace gum_distribution_l241_241901

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end gum_distribution_l241_241901


namespace A_investment_l241_241223

theorem A_investment (B_invest C_invest Total_profit A_share : ℝ) 
  (h1 : B_invest = 4200)
  (h2 : C_invest = 10500)
  (h3 : Total_profit = 12100)
  (h4 : A_share = 3630) 
  (h5 : ∀ {x : ℝ}, A_share / Total_profit = x / (x + B_invest + C_invest)) :
  ∃ A_invest : ℝ, A_invest = 6300 :=
by sorry

end A_investment_l241_241223


namespace plane_equation_rewriting_l241_241401

theorem plane_equation_rewriting (A B C D x y z p q r : ℝ)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (eq1 : A * x + B * y + C * z + D = 0)
  (hp : p = -D / A) (hq : q = -D / B) (hr : r = -D / C) :
  x / p + y / q + z / r = 1 :=
by
  sorry

end plane_equation_rewriting_l241_241401


namespace scientific_notation_example_l241_241257

theorem scientific_notation_example : (5.2 * 10^5) = 520000 := sorry

end scientific_notation_example_l241_241257


namespace cylinder_volume_increase_l241_241165

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) (hV : V = Real.pi * r^2 * h) :
    let new_height := 3 * h
    let new_radius := 2.5 * r
    let new_volume := Real.pi * (new_radius ^ 2) * new_height
    new_volume = 18.75 * V :=
by
  sorry

end cylinder_volume_increase_l241_241165


namespace train_speed_l241_241167

/-- 
A man sitting in a train which is traveling at a certain speed observes 
that a goods train, traveling in the opposite direction, takes 9 seconds 
to pass him. The goods train is 280 m long and its speed is 52 kmph. 
Prove that the speed of the train the man is sitting in is 60 kmph.
-/
theorem train_speed (t : ℝ) (h1 : 0 < t)
  (goods_speed_kmph : ℝ := 52)
  (goods_length_m : ℝ := 280)
  (time_seconds : ℝ := 9)
  (h2 : goods_length_m / time_seconds = (t + goods_speed_kmph) * (5 / 18)) :
  t = 60 :=
sorry

end train_speed_l241_241167


namespace part1_sales_volume_part2_price_reduction_l241_241363

noncomputable def daily_sales_volume (x : ℝ) : ℝ :=
  100 + 200 * x

noncomputable def profit_eq (x : ℝ) : Prop :=
  (4 - 2 - x) * (100 + 200 * x) = 300

theorem part1_sales_volume (x : ℝ) : daily_sales_volume x = 100 + 200 * x :=
sorry

theorem part2_price_reduction (hx : profit_eq (1 / 2)) : 1 / 2 = 1 / 2 :=
sorry

end part1_sales_volume_part2_price_reduction_l241_241363


namespace percent_increase_from_may_to_june_l241_241852

noncomputable def profit_increase_from_march_to_april (P : ℝ) : ℝ := 1.30 * P
noncomputable def profit_decrease_from_april_to_may (P : ℝ) : ℝ := 1.04 * P
noncomputable def profit_increase_from_march_to_june (P : ℝ) : ℝ := 1.56 * P

theorem percent_increase_from_may_to_june (P : ℝ) :
  (1.04 * P * (1 + 0.50)) = 1.56 * P :=
by
  sorry

end percent_increase_from_may_to_june_l241_241852


namespace part1_intersection_part2_range_of_m_l241_241481

-- Define the universal set and the sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part (1): When m = 3, find A ∩ B
theorem part1_intersection:
  A ∩ B 3 = {x | x < 0 ∨ x > 6} :=
sorry

-- Part (2): If B ∪ A = B, find the range of values for m
theorem part2_range_of_m (m : ℝ) :
  (B m ∪ A = B m) → (1 ≤ m ∧ m ≤ 3 / 2) :=
sorry

end part1_intersection_part2_range_of_m_l241_241481


namespace exists_special_sequence_l241_241372

open List
open Finset
open BigOperators

theorem exists_special_sequence :
  ∃ s : ℕ → ℕ,
    (∀ n, s n > 0) ∧
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    (∀ k, (∑ i in range (k + 1), s i) % (k + 1) = 0) :=
sorry  -- Proof from the provided solution steps.

end exists_special_sequence_l241_241372
