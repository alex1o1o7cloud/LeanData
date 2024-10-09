import Mathlib

namespace diaries_ratio_l1755_175508

variable (initial_diaries : ℕ)
variable (final_diaries : ℕ)
variable (lost_fraction : ℚ)
variable (bought_diaries : ℕ)

theorem diaries_ratio 
  (h1 : initial_diaries = 8)
  (h2 : final_diaries = 18)
  (h3 : lost_fraction = 1 / 4)
  (h4 : ∃ x : ℕ, (initial_diaries + x - lost_fraction * (initial_diaries + x) = final_diaries) ∧ x = 16) :
  (16 / initial_diaries : ℚ) = 2 := 
by
  sorry

end diaries_ratio_l1755_175508


namespace range_of_a_for_f_increasing_l1755_175577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a_for_f_increasing :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_for_f_increasing_l1755_175577


namespace Rick_is_three_times_Sean_l1755_175511

-- Definitions and assumptions
def Fritz_money : ℕ := 40
def Sean_money : ℕ := (Fritz_money / 2) + 4
def total_money : ℕ := 96

-- Rick's money can be derived from total_money - Sean_money
def Rick_money : ℕ := total_money - Sean_money

-- Claim to be proven
theorem Rick_is_three_times_Sean : Rick_money = 3 * Sean_money := 
by 
  -- Proof steps would go here
  sorry

end Rick_is_three_times_Sean_l1755_175511


namespace marks_in_chemistry_l1755_175534

-- Define the given conditions
def marks_english := 76
def marks_math := 65
def marks_physics := 82
def marks_biology := 85
def average_marks := 75
def number_subjects := 5

-- Define the theorem statement to prove David's marks in Chemistry
theorem marks_in_chemistry :
  let total_marks := marks_english + marks_math + marks_physics + marks_biology
  let total_marks_all_subjects := average_marks * number_subjects
  let marks_chemistry := total_marks_all_subjects - total_marks
  marks_chemistry = 67 :=
sorry

end marks_in_chemistry_l1755_175534


namespace no_such_function_exists_l1755_175571

open Set

theorem no_such_function_exists
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → y > x → f y > (y - x) * f x ^ 2) :
  False :=
sorry

end no_such_function_exists_l1755_175571


namespace manuscript_pages_l1755_175599

theorem manuscript_pages (P : ℝ)
  (h1 : 10 * (0.05 * P) + 10 * 5 = 250) : P = 400 :=
sorry

end manuscript_pages_l1755_175599


namespace evaluate_expression_l1755_175542

theorem evaluate_expression : - (16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end evaluate_expression_l1755_175542


namespace infinite_solutions_l1755_175550

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end infinite_solutions_l1755_175550


namespace math_problem_l1755_175549

noncomputable def x : ℝ := -2

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}
def C1 : Set ℝ := {1, 3}
def C2 : Set ℝ := {3, 4}

theorem math_problem
  (h1 : B x ⊆ A x) :
  x = -2 ∧ (B x ∪ C1 = A x ∨ B x ∪ C2 = A x) :=
by
  sorry

end math_problem_l1755_175549


namespace Nina_saves_enough_to_buy_video_game_in_11_weeks_l1755_175525

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end Nina_saves_enough_to_buy_video_game_in_11_weeks_l1755_175525


namespace range_of_sum_l1755_175548

theorem range_of_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + b + 3 = a * b) : 
a + b ≥ 6 := 
sorry

end range_of_sum_l1755_175548


namespace last_digit_of_3_pow_2004_l1755_175536

theorem last_digit_of_3_pow_2004 : (3 ^ 2004) % 10 = 1 := by
  sorry

end last_digit_of_3_pow_2004_l1755_175536


namespace worksheets_turned_in_l1755_175522

def initial_worksheets : ℕ := 34
def graded_worksheets : ℕ := 7
def remaining_worksheets : ℕ := initial_worksheets - graded_worksheets
def current_worksheets : ℕ := 63

theorem worksheets_turned_in :
  current_worksheets - remaining_worksheets = 36 :=
by
  sorry

end worksheets_turned_in_l1755_175522


namespace triangle_base_l1755_175585

theorem triangle_base (A h b : ℝ) (hA : A = 15) (hh : h = 6) (hbase : A = 0.5 * b * h) : b = 5 := by
  sorry

end triangle_base_l1755_175585


namespace negation_of_p_l1755_175552

-- Define the proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of p
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := sorry

end negation_of_p_l1755_175552


namespace problem_l1755_175530

section Problem
variables {n : ℕ } {k : ℕ} 

theorem problem (n : ℕ) (k : ℕ) (a : ℕ) (n_i : Fin k → ℕ) (h1 : ∀ i j, i ≠ j → Nat.gcd (n_i i) (n_i j) = 1) 
  (h2 : ∀ i, a^n_i i % n_i i = 1) (h3 : ∀ i, ¬(n_i i ∣ a - 1)) :
  ∃ (x : ℕ), x > 1 ∧ a^x % x = 1 ∧ x ≥ 2^(k + 1) - 2 := by
  sorry
end Problem

end problem_l1755_175530


namespace det_my_matrix_l1755_175543

def my_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, 1], ![-5, 5, -4], ![3, 3, 6]]

theorem det_my_matrix : my_matrix.det = 96 := by
  sorry

end det_my_matrix_l1755_175543


namespace find_natrual_numbers_l1755_175531

theorem find_natrual_numbers (k n : ℕ) (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : k ≥ 1) 
  (h2 : n ≥ 2) 
  (h3 : A ^ 3 = 0) 
  (h4 : A ^ k * B + B * A = 1) : 
  k = 1 ∧ Even n := 
sorry

end find_natrual_numbers_l1755_175531


namespace find_time_to_fill_tank_l1755_175500

noncomputable def time_to_fill_tanker (TA : ℝ) : Prop :=
  let RB := 1 / 40
  let fill_time := 29.999999999999993
  let half_fill_time := fill_time / 2
  let RAB := (1 / TA) + RB
  (RAB * half_fill_time = 1 / 2) → (TA = 120)

theorem find_time_to_fill_tank : ∃ TA, time_to_fill_tanker TA :=
by
  use 120
  sorry

end find_time_to_fill_tank_l1755_175500


namespace inverse_proportion_quadrants_l1755_175574

theorem inverse_proportion_quadrants (m : ℝ) : (∀ (x : ℝ), x ≠ 0 → y = (m - 2) / x → (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) ↔ m > 2 :=
by
  sorry

end inverse_proportion_quadrants_l1755_175574


namespace g_does_not_pass_through_fourth_quadrant_l1755_175563

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / x)

theorem g_does_not_pass_through_fourth_quadrant (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
    ¬(∃ x, x > 0 ∧ g x < 0) :=
by
    sorry

end g_does_not_pass_through_fourth_quadrant_l1755_175563


namespace ratio_of_x_intercepts_l1755_175501

theorem ratio_of_x_intercepts (b s t : ℝ) (h_b : b ≠ 0)
  (h1 : 0 = 8 * s + b)
  (h2 : 0 = 4 * t + b) :
  s / t = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l1755_175501


namespace arithmetic_sequence_primes_l1755_175512

theorem arithmetic_sequence_primes (a : ℕ) (d : ℕ) (primes_seq : ∀ n : ℕ, n < 15 → Nat.Prime (a + n * d))
  (distinct_primes : ∀ m n : ℕ, m < 15 → n < 15 → m ≠ n → a + m * d ≠ a + n * d) :
  d > 30000 := 
sorry

end arithmetic_sequence_primes_l1755_175512


namespace least_number_to_subtract_l1755_175573

theorem least_number_to_subtract (n : ℕ) (h : n = 42739) : 
    ∃ k, k = 4 ∧ (n - k) % 15 = 0 := by
  sorry

end least_number_to_subtract_l1755_175573


namespace dany_farm_bushels_l1755_175558

theorem dany_farm_bushels :
  let cows := 5
  let cows_bushels_per_day := 3
  let sheep := 4
  let sheep_bushels_per_day := 2
  let chickens := 8
  let chickens_bushels_per_day := 1
  let pigs := 6
  let pigs_bushels_per_day := 4
  let horses := 2
  let horses_bushels_per_day := 5
  cows * cows_bushels_per_day +
  sheep * sheep_bushels_per_day +
  chickens * chickens_bushels_per_day +
  pigs * pigs_bushels_per_day +
  horses * horses_bushels_per_day = 65 := by
  sorry

end dany_farm_bushels_l1755_175558


namespace committee_size_l1755_175523

theorem committee_size (n : ℕ)
  (h : ((n - 2 : ℕ) : ℚ) / ((n - 1) * (n - 2) / 2 : ℚ) = 0.4) :
  n = 6 :=
by
  sorry

end committee_size_l1755_175523


namespace solution1_solution2_l1755_175566

noncomputable def problem1 : ℝ :=
  40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12

theorem solution1 : problem1 = 43 := by
  sorry

noncomputable def problem2 : ℝ :=
  (-1 : ℝ) ^ 2021 + |(-9 : ℝ)| * (2 / 3) + (-3) / (1 / 5)

theorem solution2 : problem2 = -10 := by
  sorry

end solution1_solution2_l1755_175566


namespace part_one_part_two_l1755_175598

-- Definitions based on the conditions
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 - a}

-- Prove intersection A ∩ B = (0, 1)
theorem part_one : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

-- Prove range of a when A ∪ C = A
theorem part_two (a : ℝ) (h : A ∪ C a = A) : 1 < a := by
  sorry

end part_one_part_two_l1755_175598


namespace solve_system_of_equations_l1755_175590

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x + y = 55) 
  (h2 : x - y = 15) 
  (h3 : x > y) : 
  x = 35 ∧ y = 20 := 
sorry

end solve_system_of_equations_l1755_175590


namespace determine_s_l1755_175517

theorem determine_s 
  (s : ℝ) 
  (h : (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
       6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30) : 
  s = 4 :=
by
  sorry

end determine_s_l1755_175517


namespace long_diagonal_length_l1755_175526

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end long_diagonal_length_l1755_175526


namespace find_unknown_rate_l1755_175564

variable (x : ℕ)

theorem find_unknown_rate
    (c3 : ℕ := 3 * 100)
    (c5 : ℕ := 5 * 150)
    (n : ℕ := 10)
    (avg_price : ℕ := 160) 
    (h : c3 + c5 + 2 * x = avg_price * n) :
    x = 275 := 
by
  -- Proof goes here.
  sorry

end find_unknown_rate_l1755_175564


namespace min_value_expression_l1755_175591

theorem min_value_expression (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) : (x * y + x^2) ≥ 4 :=
sorry

end min_value_expression_l1755_175591


namespace find_product_abcd_l1755_175514

def prod_abcd (a b c d : ℚ) :=
  4 * a - 2 * b + 3 * c + 5 * d = 22 ∧
  2 * (d + c) = b - 2 ∧
  4 * b - c = a + 1 ∧
  c + 1 = 2 * d

theorem find_product_abcd (a b c d : ℚ) (h : prod_abcd a b c d) :
  a * b * c * d = -30751860 / 11338912 :=
sorry

end find_product_abcd_l1755_175514


namespace sin_beta_value_l1755_175592

theorem sin_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h1 : Real.cos α = 4 / 5) (h2 : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_value_l1755_175592


namespace num_people_present_l1755_175524

-- Given conditions
def associatePencilCount (A : ℕ) : ℕ := 2 * A
def assistantPencilCount (B : ℕ) : ℕ := B
def associateChartCount (A : ℕ) : ℕ := A
def assistantChartCount (B : ℕ) : ℕ := 2 * B

def totalPencils (A B : ℕ) : ℕ := associatePencilCount A + assistantPencilCount B
def totalCharts (A B : ℕ) : ℕ := associateChartCount A + assistantChartCount B

-- Prove the total number of people present
theorem num_people_present (A B : ℕ) (h1 : totalPencils A B = 11) (h2 : totalCharts A B = 16) : A + B = 9 :=
by
  sorry

end num_people_present_l1755_175524


namespace system_solution_l1755_175579

theorem system_solution :
  ∃ x y : ℝ, (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧ 
            (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
            x = -3 / 4 ∧ y = 1 / 2 :=
by
  sorry

end system_solution_l1755_175579


namespace monochromatic_triangle_probability_l1755_175502

noncomputable def probability_of_monochromatic_triangle_in_hexagon : ℝ := 0.968324

theorem monochromatic_triangle_probability :
  ∃ (H : Hexagon), probability_of_monochromatic_triangle_in_hexagon = 0.968324 :=
sorry

end monochromatic_triangle_probability_l1755_175502


namespace stock_return_to_original_l1755_175593

theorem stock_return_to_original (x : ℝ) : 
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  1.56 * (1 - p/100) = 1 :=
by
  intro x
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  show 1.56 * (1 - p / 100) = 1
  sorry

end stock_return_to_original_l1755_175593


namespace age_difference_is_18_l1755_175576

variable (A B C : ℤ)
variable (h1 : A + B > B + C)
variable (h2 : C = A - 18)

theorem age_difference_is_18 : (A + B) - (B + C) = 18 :=
by
  sorry

end age_difference_is_18_l1755_175576


namespace carpet_area_l1755_175586

theorem carpet_area (length_ft : ℕ) (width_ft : ℕ) (ft_per_yd : ℕ) (A_y : ℕ) 
  (h_length : length_ft = 15) (h_width : width_ft = 12) (h_ft_per_yd : ft_per_yd = 9) :
  A_y = (length_ft * width_ft) / ft_per_yd := 
by sorry

#check carpet_area

end carpet_area_l1755_175586


namespace max_b_div_a_plus_c_l1755_175544

-- Given positive numbers a, b, c
-- equation: b^2 + 2(a + c)b - ac = 0
-- Prove: ∀ a b c : ℝ (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : b^2 + 2*(a + c)*b - a*c = 0),
--         b/(a + c) ≤ (Real.sqrt 5 - 2)/2

theorem max_b_div_a_plus_c (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + 2 * (a + c) * b - a * c = 0) :
  b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
sorry

end max_b_div_a_plus_c_l1755_175544


namespace stratified_sampling_l1755_175569

-- Definition of conditions as hypothesis
def total_employees : ℕ := 100
def under_35 : ℕ := 45
def between_35_49 : ℕ := 25
def over_50 : ℕ := total_employees - under_35 - between_35_49
def sample_size : ℕ := 20
def sampling_ratio : ℚ := sample_size / total_employees

-- The target number of people from each group
def under_35_sample : ℚ := sampling_ratio * under_35
def between_35_49_sample : ℚ := sampling_ratio * between_35_49
def over_50_sample : ℚ := sampling_ratio * over_50

-- Problem statement
theorem stratified_sampling : 
  under_35_sample = 9 ∧ 
  between_35_49_sample = 5 ∧ 
  over_50_sample = 6 :=
  by
  sorry

end stratified_sampling_l1755_175569


namespace factor_expression_l1755_175541

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end factor_expression_l1755_175541


namespace remainder_of_trailing_zeroes_in_factorials_product_l1755_175553

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def product_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc * factorial x) 1 

def trailing_zeroes (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5 + trailing_zeroes (n / 5))

def trailing_zeroes_in_product (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc + trailing_zeroes x) 0 

theorem remainder_of_trailing_zeroes_in_factorials_product :
  let N := trailing_zeroes_in_product 150
  N % 500 = 45 :=
by
  sorry

end remainder_of_trailing_zeroes_in_factorials_product_l1755_175553


namespace circumference_difference_l1755_175581

theorem circumference_difference (r : ℝ) (width : ℝ) (hp : width = 10.504226244065093) : 
  2 * Real.pi * (r + width) - 2 * Real.pi * r = 66.00691339889247 := by
  sorry

end circumference_difference_l1755_175581


namespace swimming_pool_width_l1755_175528

theorem swimming_pool_width
  (length : ℝ)
  (lowered_height_inches : ℝ)
  (removed_water_gallons : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (volume_for_removal : ℝ)
  (width : ℝ) :
  length = 60 → 
  lowered_height_inches = 6 →
  removed_water_gallons = 4500 →
  gallons_per_cubic_foot = 7.5 →
  volume_for_removal = removed_water_gallons / gallons_per_cubic_foot →
  width = volume_for_removal / (length * (lowered_height_inches / 12)) →
  width = 20 :=
by
  intros h_length h_lowered_height h_removed_water h_gallons_per_cubic_foot h_volume_for_removal h_width
  sorry

end swimming_pool_width_l1755_175528


namespace correct_expression_after_removing_parentheses_l1755_175537

variable (a b c : ℝ)

theorem correct_expression_after_removing_parentheses :
  -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c :=
sorry

end correct_expression_after_removing_parentheses_l1755_175537


namespace weight_shifted_count_l1755_175595

def is_weight_shifted (a b x y : ℕ) : Prop :=
  a + b = 2 * (x + y) ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9

theorem weight_shifted_count : 
  ∃ count : ℕ, count = 225 ∧ 
  (∀ (a b x y : ℕ), is_weight_shifted a b x y → count = 225) := 
sorry

end weight_shifted_count_l1755_175595


namespace children_l1755_175540

variable (C : ℝ) -- Define the weight of a children's book

theorem children's_book_weight :
  (9 * 0.8 + 7 * C = 10.98) → C = 0.54 :=
by  
sorry

end children_l1755_175540


namespace total_amount_divided_l1755_175597

variables (T x : ℝ)
variables (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
variables (h₂ : T - x = 1100)

theorem total_amount_divided (T x : ℝ) 
  (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
  (h₂ : T - x = 1100) : 
  T = 1600 := 
sorry

end total_amount_divided_l1755_175597


namespace seeds_in_bucket_C_l1755_175513

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end seeds_in_bucket_C_l1755_175513


namespace equilateral_triangle_area_l1755_175557

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l1755_175557


namespace minimize_G_l1755_175575

noncomputable def F (p q : ℝ) : ℝ :=
  2 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G :
  ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 0.75 → G p = G 0 → p = 0 :=
by
  intro p hp hG
  -- The proof goes here
  sorry

end minimize_G_l1755_175575


namespace solve_m_correct_l1755_175560

noncomputable def solve_for_m (Q t h : ℝ) : ℝ :=
  if h >= 0 ∧ Q > 0 ∧ t > 0 then
    (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h))
  else
    0 -- Define default output for invalid inputs

theorem solve_m_correct (Q t h : ℝ) (m : ℝ) :
  Q = t / (1 + Real.sqrt h)^m → m = (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h)) :=
by
  intros h1
  rw [h1]
  sorry

end solve_m_correct_l1755_175560


namespace steps_in_staircase_l1755_175521

theorem steps_in_staircase (h1 : 120 / 20 = 6) (h2 : 180 / 6 = 30) : 
  ∃ n : ℕ, n = 30 :=
by
  -- the proof is omitted
  sorry

end steps_in_staircase_l1755_175521


namespace tan_alpha_values_l1755_175555

theorem tan_alpha_values (α : ℝ) (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := 
by sorry

end tan_alpha_values_l1755_175555


namespace dodecahedron_interior_diagonals_l1755_175519

-- Definition of a dodecahedron based on given conditions
structure Dodecahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (vertices_per_face : ℕ)
  (faces_per_vertex : ℕ)
  (interior_diagonals : ℕ)

-- Conditions provided in the problem
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_per_vertex := 3,
    interior_diagonals := 130 }

-- The theorem to prove that given a dodecahedron structure, it has the correct number of interior diagonals
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : d.interior_diagonals = 130 := by
  sorry

end dodecahedron_interior_diagonals_l1755_175519


namespace zeros_of_f_l1755_175584

def f (x : ℝ) : ℝ := (x^2 - 3 * x) * (x + 4)

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = 0 ∨ x = 3 ∨ x = -4 := by
  sorry

end zeros_of_f_l1755_175584


namespace problem1_problem2_l1755_175535

theorem problem1 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : a > b) : a + b = 7 ∨ a + b = 3 := 
by sorry

theorem problem2 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : |a + b| = |a| - |b|) : (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2) := 
by sorry

end problem1_problem2_l1755_175535


namespace frank_ryan_problem_ratio_l1755_175587

theorem frank_ryan_problem_ratio 
  (bill_problems : ℕ)
  (h1 : bill_problems = 20)
  (ryan_problems : ℕ)
  (h2 : ryan_problems = 2 * bill_problems)
  (frank_problems_per_type : ℕ)
  (h3 : frank_problems_per_type = 30)
  (types : ℕ)
  (h4 : types = 4) : 
  frank_problems_per_type * types / ryan_problems = 3 := by
  sorry

end frank_ryan_problem_ratio_l1755_175587


namespace unwanted_texts_per_week_l1755_175510

-- Define the conditions as constants
def messages_per_day_old : ℕ := 20
def messages_per_day_new : ℕ := 55
def days_per_week : ℕ := 7

-- Define the theorem stating the problem
theorem unwanted_texts_per_week (messages_per_day_old messages_per_day_new days_per_week 
  : ℕ) : (messages_per_day_new - messages_per_day_old) * days_per_week = 245 :=
by
  sorry

end unwanted_texts_per_week_l1755_175510


namespace project_completion_by_B_l1755_175580

-- Definitions of the given conditions
def person_A_work_rate := 1 / 10
def person_B_work_rate := 1 / 15
def days_A_worked := 3

-- Definition of the mathematical proof problem
theorem project_completion_by_B {x : ℝ} : person_A_work_rate * days_A_worked + person_B_work_rate * x = 1 :=
by
  sorry

end project_completion_by_B_l1755_175580


namespace a_gives_b_head_start_l1755_175515

theorem a_gives_b_head_start (Va Vb L H : ℝ) 
    (h1 : Va = (20 / 19) * Vb)
    (h2 : L / Va = (L - H) / Vb) : 
    H = (1 / 20) * L := sorry

end a_gives_b_head_start_l1755_175515


namespace part_one_part_two_l1755_175546

def f (a x : ℝ) : ℝ := |a - 4 * x| + |2 * a + x|

theorem part_one (x : ℝ) : f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 / 5 := 
sorry

theorem part_two (a x : ℝ) : f a x + f a (-1 / x) ≥ 10 := 
sorry

end part_one_part_two_l1755_175546


namespace condition_for_diff_of_roots_l1755_175533

/-- Statement: For a quadratic equation of the form x^2 + px + q = 0, if the difference of the roots is a, then the condition a^2 - p^2 = -4q holds. -/
theorem condition_for_diff_of_roots (p q a : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 - x2 = a) :
  a^2 - p^2 = -4 * q :=
sorry

end condition_for_diff_of_roots_l1755_175533


namespace range_of_a_l1755_175520

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

-- Define the theorem to be proved
theorem range_of_a (a : ℝ) (h₁ : A a ⊆ B) (h₂ : 2 - a < 2 + a) : 0 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l1755_175520


namespace gcd_of_360_and_150_l1755_175559

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end gcd_of_360_and_150_l1755_175559


namespace circle_properties_l1755_175503

noncomputable def pi : Real := 3.14
variable (C : Real) (diameter : Real) (radius : Real) (area : Real)

theorem circle_properties (h₀ : C = 31.4) :
  radius = C / (2 * pi) ∧
  diameter = 2 * radius ∧
  area = pi * radius^2 ∧
  radius = 5 ∧
  diameter = 10 ∧
  area = 78.5 :=
by
  sorry

end circle_properties_l1755_175503


namespace inequality_with_xy_l1755_175504

theorem inequality_with_xy
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3)) + (1 / (y + 3)) ≤ 2 / 5 :=
sorry

end inequality_with_xy_l1755_175504


namespace value_of_a_l1755_175572

variable (a : ℝ)

theorem value_of_a (h1 : (0.5 / 100) * a = 0.80) : a = 160 := by
  sorry

end value_of_a_l1755_175572


namespace x_squared_plus_y_squared_l1755_175547

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := 
by 
  sorry

end x_squared_plus_y_squared_l1755_175547


namespace elevation_after_descend_l1755_175506

theorem elevation_after_descend (initial_elevation : ℕ) (rate : ℕ) (time : ℕ) (final_elevation : ℕ) 
  (h_initial : initial_elevation = 400) 
  (h_rate : rate = 10) 
  (h_time : time = 5) 
  (h_final : final_elevation = initial_elevation - rate * time) : 
  final_elevation = 350 := 
by 
  sorry

end elevation_after_descend_l1755_175506


namespace value_of_f5_and_f_neg5_l1755_175539

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f5_and_f_neg5 (a b c : ℝ) (m : ℝ) (h : f a b c (-5) = m) :
  f a b c 5 + f a b c (-5) = 4 :=
sorry

end value_of_f5_and_f_neg5_l1755_175539


namespace slices_with_all_toppings_l1755_175527

-- Definitions
def slices_with_pepperoni (x y w : ℕ) : ℕ := 15 - x - y + w
def slices_with_mushrooms (x z w : ℕ) : ℕ := 16 - x - z + w
def slices_with_olives (y z w : ℕ) : ℕ := 10 - y - z + w

-- Problem's total validation condition
axiom total_slices_with_at_least_one_topping (x y z w : ℕ) :
  15 + 16 + 10 - x - y - z - 2 * w = 24

-- Statement to prove
theorem slices_with_all_toppings (x y z w : ℕ) (h : 15 + 16 + 10 - x - y - z - 2 * w = 24) : w = 2 :=
sorry

end slices_with_all_toppings_l1755_175527


namespace chairs_left_to_move_l1755_175596

theorem chairs_left_to_move (total_chairs : ℕ) (carey_chairs : ℕ) (pat_chairs : ℕ) (h1 : total_chairs = 74)
  (h2 : carey_chairs = 28) (h3 : pat_chairs = 29) : total_chairs - carey_chairs - pat_chairs = 17 :=
by 
  sorry

end chairs_left_to_move_l1755_175596


namespace left_person_truthful_right_person_lies_l1755_175589

theorem left_person_truthful_right_person_lies
  (L R M : Prop)
  (L_truthful_or_false : L ∨ ¬L)
  (R_truthful_or_false : R ∨ ¬R)
  (M_always_answers : M = (L → M) ∨ (¬L → M))
  (left_statement : L → (M = (L → M)))
  (right_statement : R → (M = (¬L → M))) :
  (L ∧ ¬R) ∨ (¬L ∧ R) :=
by
  sorry

end left_person_truthful_right_person_lies_l1755_175589


namespace find_x_l1755_175583

/-- Given real numbers x and y,
    under the condition that (y^3 + 2y - 1)/(y^3 + 2y - 3) = x/(x - 1),
    we want to prove that x = (y^3 + 2y - 1)/2 -/
theorem find_x (x y : ℝ) (h1 : y^3 + 2*y - 3 ≠ 0) (h2 : y^3 + 2*y - 1 ≠ 0)
  (h : x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3)) :
  x = (y^3 + 2*y - 1) / 2 :=
by sorry

end find_x_l1755_175583


namespace system_solution_l1755_175562

theorem system_solution (m n : ℚ) (x y : ℚ) 
  (h₁ : 2 * x + m * y = 5) 
  (h₂ : n * x - 3 * y = 2) 
  (h₃ : x = 3)
  (h₄ : y = 1) : 
  m / n = -3 / 5 :=
by sorry

end system_solution_l1755_175562


namespace all_fruits_fallen_by_twelfth_day_l1755_175507

noncomputable def magical_tree_falling_day : Nat :=
  let total_fruits := 58
  let initial_day_falls := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].foldl (· + ·) 0
  let continuation_falls := [1, 2].foldl (· + ·) 0
  let total_days := initial_day_falls + continuation_falls
  12

theorem all_fruits_fallen_by_twelfth_day :
  magical_tree_falling_day = 12 :=
by
  sorry

end all_fruits_fallen_by_twelfth_day_l1755_175507


namespace Aiyanna_has_more_cookies_l1755_175565

theorem Aiyanna_has_more_cookies (Alyssa_cookies : ℕ) (Aiyanna_cookies : ℕ) (hAlyssa : Alyssa_cookies = 129) (hAiyanna : Aiyanna_cookies = 140) : Aiyanna_cookies - Alyssa_cookies = 11 := 
by sorry

end Aiyanna_has_more_cookies_l1755_175565


namespace total_gallons_l1755_175567

def gallons_used (A F : ℕ) := F = 4 * A - 5

theorem total_gallons
  (A F : ℕ)
  (h1 : gallons_used A F)
  (h2 : F = 23) :
  A + F = 30 :=
by
  sorry

end total_gallons_l1755_175567


namespace eval_expression_l1755_175588

theorem eval_expression :
  -((18 / 3 * 8) - 80 + (4 ^ 2 * 2)) = 0 :=
by
  sorry

end eval_expression_l1755_175588


namespace pounds_of_sugar_l1755_175594

theorem pounds_of_sugar (x p : ℝ) (h1 : x * p = 216) (h2 : (x + 3) * (p - 1) = 216) : x = 24 :=
sorry

end pounds_of_sugar_l1755_175594


namespace find_n_in_arithmetic_sequence_l1755_175556

noncomputable def arithmetic_sequence (a1 d n : ℕ) := a1 + (n - 1) * d

theorem find_n_in_arithmetic_sequence (a1 d an : ℕ) (h1 : a1 = 1) (h2 : d = 5) (h3 : an = 2016) :
  ∃ n : ℕ, an = arithmetic_sequence a1 d n :=
  by
  sorry

end find_n_in_arithmetic_sequence_l1755_175556


namespace value_of_b_l1755_175509

theorem value_of_b (b : ℝ) (f g : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2 - b * x + 3) ∧ 
  (∀ x, g x = 2 * x^2 + b * x + 3) ∧ 
  (∀ x, g x = f (x + 6)) →
  b = 12 :=
by
  sorry

end value_of_b_l1755_175509


namespace smallest_positive_integer_divisible_by_8_11_15_l1755_175532

-- Define what it means for a number to be divisible by another
def divisible_by (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

-- Define a function to find the least common multiple of three numbers
noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Statement of the theorem
theorem smallest_positive_integer_divisible_by_8_11_15 : 
  ∀ n : ℕ, (n > 0) ∧ divisible_by n 8 ∧ divisible_by n 11 ∧ divisible_by n 15 ↔ n = 1320 :=
sorry -- Proof is omitted

end smallest_positive_integer_divisible_by_8_11_15_l1755_175532


namespace total_amount_divided_l1755_175545

theorem total_amount_divided 
    (A B C : ℝ) 
    (h1 : A = (2 / 3) * (B + C)) 
    (h2 : B = (2 / 3) * (A + C)) 
    (h3 : A = 160) : 
    A + B + C = 400 := 
by 
  sorry

end total_amount_divided_l1755_175545


namespace minimum_value_of_expression_l1755_175516

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end minimum_value_of_expression_l1755_175516


namespace part1_solution_part2_solution_l1755_175538

def part1 (m : ℝ) (x1 : ℝ) (x2 : ℝ) : Prop :=
  (m * x1 - 2) * (m * x2 - 2) = 4

theorem part1_solution : part1 (1/3) 9 18 :=
by 
  sorry

def part2 (m x1 x2 : ℕ) : Prop :=
  ((m * x1 - 2) * (m * x2 - 2) = 4)

def count_pairs : ℕ := 7

theorem part2_solution 
  (m x1 x2 : ℕ) 
  (h_pos : m > 0 ∧ x1 > 0 ∧ x2 > 0) : 
  ∃ c, c = count_pairs ∧ 
  (part2 m x1 x2) :=
by 
  sorry

end part1_solution_part2_solution_l1755_175538


namespace discount_policy_l1755_175551

-- Define the prices of the fruits
def lemon_price := 2
def papaya_price := 1
def mango_price := 4

-- Define the quantities Tom buys
def lemons_bought := 6
def papayas_bought := 4
def mangos_bought := 2

-- Define the total amount paid by Tom
def amount_paid := 21

-- Define the total number of fruits bought
def total_fruits_bought := lemons_bought + papayas_bought + mangos_bought

-- Define the total cost without discount
def total_cost_without_discount := 
  (lemons_bought * lemon_price) + 
  (papayas_bought * papaya_price) + 
  (mangos_bought * mango_price)

-- Calculate the discount
def discount := total_cost_without_discount - amount_paid

-- The discount policy
theorem discount_policy : discount = 3 ∧ total_fruits_bought = 12 :=
by 
  sorry

end discount_policy_l1755_175551


namespace domain_of_sqrt_fraction_l1755_175561

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (x - 2 ≥ 0 ∧ 5 - x > 0) ↔ (2 ≤ x ∧ x < 5) :=
by
  sorry

end domain_of_sqrt_fraction_l1755_175561


namespace ratio_addition_l1755_175578

theorem ratio_addition (x : ℤ) (h : 4 + x = 3 * (15 + x) / 4): x = 29 :=
by
  sorry

end ratio_addition_l1755_175578


namespace sufficient_but_not_necessary_condition_l1755_175505

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x + m = 0) ↔ m < 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l1755_175505


namespace other_root_of_equation_l1755_175518

theorem other_root_of_equation (m : ℝ) :
  (∃ (x : ℝ), 3 * x^2 + m * x = -2 ∧ x = -1) →
  (∃ (y : ℝ), 3 * y^2 + m * y + 2 = 0 ∧ y = -(-2 / 3)) :=
by
  sorry

end other_root_of_equation_l1755_175518


namespace min_value_sqrt_sum_l1755_175554

open Real

theorem min_value_sqrt_sum (x : ℝ) : 
    ∃ c : ℝ, (∀ x : ℝ, c ≤ sqrt (x^2 - 4 * x + 13) + sqrt (x^2 - 10 * x + 26)) ∧ 
             (sqrt ((17/4)^2 - 4 * (17/4) + 13) + sqrt ((17/4)^2 - 10 * (17/4) + 26) = 5 ∧ c = 5) := 
by
  sorry

end min_value_sqrt_sum_l1755_175554


namespace james_profit_l1755_175582

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l1755_175582


namespace green_apples_count_l1755_175529

variables (G R : ℕ)

def total_apples_collected (G R : ℕ) : Prop :=
  R + G = 496

def relation_red_green (G R : ℕ) : Prop :=
  R = 3 * G

theorem green_apples_count (G R : ℕ) (h1 : total_apples_collected G R) (h2 : relation_red_green G R) :
  G = 124 :=
by sorry

end green_apples_count_l1755_175529


namespace expression_value_l1755_175568

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end expression_value_l1755_175568


namespace add_base6_l1755_175570

def base6_to_base10 (n : Nat) : Nat :=
  let rec aux (n : Nat) (exp : Nat) : Nat :=
    match n with
    | 0     => 0
    | n + 1 => aux n (exp + 1) + (n % 6) * (6 ^ exp)
  aux n 0

def base10_to_base6 (n : Nat) : Nat :=
  let rec aux (n : Nat) : Nat :=
    if n = 0 then 0
    else
      let q := n / 6
      let r := n % 6
      r + 10 * aux q
  aux n

theorem add_base6 (a b : Nat) (h1 : base6_to_base10 a = 5) (h2 : base6_to_base10 b = 13) : base10_to_base6 (base6_to_base10 a + base6_to_base10 b) = 30 :=
by
  sorry

end add_base6_l1755_175570
