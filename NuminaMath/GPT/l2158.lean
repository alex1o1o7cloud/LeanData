import Mathlib

namespace lunas_phone_bill_percentage_l2158_215823

variables (H F P : ℝ)

theorem lunas_phone_bill_percentage :
  F = 0.60 * H ∧ H + F = 240 ∧ H + F + P = 249 →
  (P / F) * 100 = 10 :=
by
  intros
  sorry

end lunas_phone_bill_percentage_l2158_215823


namespace find_functions_l2158_215855

theorem find_functions (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ^ 2) :=
by
  sorry

end find_functions_l2158_215855


namespace lemonade_in_pitcher_l2158_215835

theorem lemonade_in_pitcher (iced_tea lemonade total_pitcher total_in_drink lemonade_ratio : ℚ)
  (h1 : iced_tea = 1/4)
  (h2 : lemonade = 5/4)
  (h3 : total_in_drink = iced_tea + lemonade)
  (h4 : lemonade_ratio = lemonade / total_in_drink)
  (h5 : total_pitcher = 18) :
  (total_pitcher * lemonade_ratio) = 15 :=
by
  sorry

end lemonade_in_pitcher_l2158_215835


namespace sandy_money_l2158_215872

theorem sandy_money (X : ℝ) (h1 : 0.70 * X = 224) : X = 320 := 
by {
  sorry
}

end sandy_money_l2158_215872


namespace paper_fold_ratio_l2158_215884

theorem paper_fold_ratio (paper_side : ℕ) (fold_fraction : ℚ) (cut_fraction : ℚ)
  (thin_section_width thick_section_width : ℕ) (small_width large_width : ℚ)
  (P_small P_large : ℚ) (ratio : ℚ) :
  paper_side = 6 →
  fold_fraction = 1 / 3 →
  cut_fraction = 2 / 3 →
  thin_section_width = 2 →
  thick_section_width = 4 →
  small_width = 2 →
  large_width = 16 / 3 →
  P_small = 2 * (6 + small_width) →
  P_large = 2 * (6 + large_width) →
  ratio = P_small / P_large →
  ratio = 12 / 17 :=
by
  sorry

end paper_fold_ratio_l2158_215884


namespace sally_reads_10_pages_on_weekdays_l2158_215843

def sallyReadsOnWeekdays (x : ℕ) (total_pages : ℕ) (weekdays : ℕ) (weekend_days : ℕ) (weekend_pages : ℕ) : Prop :=
  (weekdays + weekend_days * weekend_pages = total_pages) → (weekdays * x = total_pages - weekend_days * weekend_pages)

theorem sally_reads_10_pages_on_weekdays :
  sallyReadsOnWeekdays 10 180 10 4 20 :=
by
  intros h
  sorry  -- proof to be filled in

end sally_reads_10_pages_on_weekdays_l2158_215843


namespace red_marbles_l2158_215863

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end red_marbles_l2158_215863


namespace perpendicular_lines_condition_l2158_215881

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → m * x + y + 1 = 0 → False) ↔ m = 1 / 2 :=
by sorry

end perpendicular_lines_condition_l2158_215881


namespace distinct_lines_through_point_and_parabola_l2158_215879

noncomputable def num_distinct_lines : ℕ :=
  let num_divisors (n : ℕ) : ℕ :=
    have factors := [2^5, 3^2, 7]
    factors.foldl (fun acc f => acc * (f + 1)) 1
  (num_divisors 2016) / 2 -- as each pair (x_1, x_2) corresponds twice

theorem distinct_lines_through_point_and_parabola :
  num_distinct_lines = 36 :=
by
  sorry

end distinct_lines_through_point_and_parabola_l2158_215879


namespace base8_perfect_square_b_zero_l2158_215887

-- Define the base 8 representation and the perfect square condition
def base8_to_decimal (a b : ℕ) : ℕ := 512 * a + 64 + 8 * b + 4

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating that if the number in base 8 is a perfect square, then b = 0
theorem base8_perfect_square_b_zero (a b : ℕ) (h₀ : a ≠ 0) 
  (h₁ : is_perfect_square (base8_to_decimal a b)) : b = 0 :=
sorry

end base8_perfect_square_b_zero_l2158_215887


namespace selection_methods_l2158_215870

-- Define the number of students and lectures.
def numberOfStudents : Nat := 6
def numberOfLectures : Nat := 5

-- Define the problem as proving the number of selection methods equals 5^6.
theorem selection_methods : (numberOfLectures ^ numberOfStudents) = 15625 := by
  -- Include the proper mathematical equivalence statement
  sorry

end selection_methods_l2158_215870


namespace smallest_y_l2158_215826

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end smallest_y_l2158_215826


namespace solution_set_l2158_215807

noncomputable def solve_inequality : Set ℝ :=
  {x | (1 / (x - 1)) >= -1}

theorem solution_set :
  solve_inequality = {x | x ≤ 0} ∪ {x | x > 1} :=
by
  sorry

end solution_set_l2158_215807


namespace find_cost_of_crackers_l2158_215894

-- Definitions based on the given conditions
def cost_hamburger_meat : ℝ := 5.00
def cost_per_bag_vegetables : ℝ := 2.00
def number_of_bags_vegetables : ℕ := 4
def cost_cheese : ℝ := 3.50
def discount_rate : ℝ := 0.10
def total_after_discount : ℝ := 18

-- Definition of the box of crackers, which we aim to prove
def cost_crackers : ℝ := 3.50

-- The Lean statement for the proof
theorem find_cost_of_crackers
  (C : ℝ)
  (h : C = cost_crackers)
  (H : 0.9 * (cost_hamburger_meat + cost_per_bag_vegetables * number_of_bags_vegetables + cost_cheese + C) = total_after_discount) :
  C = 3.50 :=
  sorry

end find_cost_of_crackers_l2158_215894


namespace mary_shirt_fraction_l2158_215816

theorem mary_shirt_fraction (f : ℝ) : 
  26 * (1 - f) + 36 - 36 / 3 = 37 → f = 1 / 2 :=
by
  sorry

end mary_shirt_fraction_l2158_215816


namespace fractional_eq_has_positive_root_m_value_l2158_215868

-- Define the conditions and the proof goal
theorem fractional_eq_has_positive_root_m_value (m x : ℝ) (h1 : x - 2 ≠ 0) (h2 : 2 - x ≠ 0) (h3 : ∃ x > 0, (m / (x - 2)) = ((1 - x) / (2 - x)) - 3) : m = 1 :=
by
  -- Proof goes here
  sorry

end fractional_eq_has_positive_root_m_value_l2158_215868


namespace probability_of_sum_greater_than_15_l2158_215805

-- Definition of the dice and outcomes
def total_outcomes : ℕ := 6 * 6 * 6
def favorable_outcomes : ℕ := 10

-- Probability calculation
def probability_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

-- Theorem to be proven
theorem probability_of_sum_greater_than_15 : probability_sum_gt_15 = 5 / 108 := by
  sorry

end probability_of_sum_greater_than_15_l2158_215805


namespace sqrt_product_l2158_215804

theorem sqrt_product (h1 : Real.sqrt 81 = 9) 
                     (h2 : Real.sqrt 16 = 4) 
                     (h3 : Real.sqrt (Real.sqrt (Real.sqrt 64)) = 2 * Real.sqrt 2) : 
                     Real.sqrt 81 * Real.sqrt 16 * Real.sqrt (Real.sqrt (Real.sqrt 64)) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l2158_215804


namespace line_equation_with_slope_angle_135_and_y_intercept_neg1_l2158_215822

theorem line_equation_with_slope_angle_135_and_y_intercept_neg1 :
  ∃ k b : ℝ, k = -1 ∧ b = -1 ∧ ∀ x y : ℝ, y = k * x + b ↔ y = -x - 1 :=
by
  sorry

end line_equation_with_slope_angle_135_and_y_intercept_neg1_l2158_215822


namespace range_of_a_l2158_215845

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a^2 + a) * x + a^3 > 0 ↔ (x < a^2 ∨ x > a)) → (0 ≤ a ∧ a ≤ 1) :=
by
  intros h
  sorry

end range_of_a_l2158_215845


namespace set_intersection_l2158_215827

def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1, 2}

theorem set_intersection :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end set_intersection_l2158_215827


namespace exponential_function_solution_l2158_215811

theorem exponential_function_solution (a : ℝ) (h : a > 1)
  (h_max_min_diff : a - a⁻¹ = 1) : a = (Real.sqrt 5 + 1) / 2 :=
sorry

end exponential_function_solution_l2158_215811


namespace Jeanine_gave_fraction_of_pencils_l2158_215878

theorem Jeanine_gave_fraction_of_pencils
  (Jeanine_initial_pencils Clare_initial_pencils Jeanine_pencils_after Clare_pencils_after : ℕ)
  (h1 : Jeanine_initial_pencils = 18)
  (h2 : Clare_initial_pencils = Jeanine_initial_pencils / 2)
  (h3 : Jeanine_pencils_after = Clare_pencils_after + 3)
  (h4 : Clare_pencils_after = Clare_initial_pencils)
  (h5 : Jeanine_pencils_after + (Jeanine_initial_pencils - Jeanine_pencils_after) = Jeanine_initial_pencils) :
  (Jeanine_initial_pencils - Jeanine_pencils_after) / Jeanine_initial_pencils = 1 / 3 :=
by
  -- Proof here
  sorry

end Jeanine_gave_fraction_of_pencils_l2158_215878


namespace interest_rate_is_five_percent_l2158_215893

-- Define the principal amount P and the interest rate r.
variables (P : ℝ) (r : ℝ)

-- Define the conditions given in the problem
def simple_interest_condition : Prop := P * r * 2 = 40
def compound_interest_condition : Prop := P * (1 + r)^2 - P = 41

-- Define the goal statement to prove
theorem interest_rate_is_five_percent (h1 : simple_interest_condition P r) (h2 : compound_interest_condition P r) : r = 0.05 :=
sorry

end interest_rate_is_five_percent_l2158_215893


namespace rowing_distance_upstream_l2158_215861

theorem rowing_distance_upstream 
  (v : ℝ) (d : ℝ)
  (h1 : 75 = (v + 3) * 5)
  (h2 : d = (v - 3) * 5) :
  d = 45 :=
by {
  sorry
}

end rowing_distance_upstream_l2158_215861


namespace find_numbers_l2158_215817

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l2158_215817


namespace stratified_sampling_result_l2158_215806

-- Define the total number of students in each grade
def students_grade10 : ℕ := 1600
def students_grade11 : ℕ := 1200
def students_grade12 : ℕ := 800

-- Define the condition
def stratified_sampling (x : ℕ) : Prop :=
  (x / (students_grade10 + students_grade11 + students_grade12) = (20 / students_grade12))

-- The main statement to be proven
theorem stratified_sampling_result 
  (students_grade10 : ℕ)
  (students_grade11 : ℕ)
  (students_grade12 : ℕ)
  (sampled_from_grade12 : ℕ)
  (h_sampling : stratified_sampling 90)
  (h_sampled12 : sampled_from_grade12 = 20) :
  (90 - sampled_from_grade12 = 70) :=
  by
    sorry

end stratified_sampling_result_l2158_215806


namespace circle_area_difference_l2158_215867

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 30) (h2 : r2 = 7.5) : 
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end circle_area_difference_l2158_215867


namespace simplify_radicals_l2158_215873

theorem simplify_radicals :
  (Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by 
  sorry

end simplify_radicals_l2158_215873


namespace least_five_digit_is_15625_l2158_215825

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l2158_215825


namespace triangle_BD_length_l2158_215810

noncomputable def triangle_length_BD : ℝ :=
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  BD

theorem triangle_BD_length : triangle_length_BD = 63 :=
by
  -- Definitions and assumptions
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)

  -- Formal proof logic corresponding to solution steps
  sorry

end triangle_BD_length_l2158_215810


namespace hyperbola_eccentricity_l2158_215841

theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ e : ℝ, e = 5 / 4 ∧ (∀ x y : ℝ, (x^2 / 16) - (y^2 / m) = 1)) → m = 9 :=
by
  intro h
  sorry

end hyperbola_eccentricity_l2158_215841


namespace sin_70_eq_1_minus_2k_squared_l2158_215821

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by
  sorry

end sin_70_eq_1_minus_2k_squared_l2158_215821


namespace Dave_tiles_210_square_feet_l2158_215849

theorem Dave_tiles_210_square_feet
  (ratio_charlie_dave : ℕ := 5 / 7)
  (total_area : ℕ := 360)
  : ∀ (work_done_by_dave : ℕ), work_done_by_dave = 210 :=
by
  sorry

end Dave_tiles_210_square_feet_l2158_215849


namespace day_of_week_50th_day_of_year_N_minus_1_l2158_215851

def day_of_week (d : ℕ) (first_day : ℕ) : ℕ :=
  (first_day + d - 1) % 7

theorem day_of_week_50th_day_of_year_N_minus_1 
  (N : ℕ) 
  (day_250_N : ℕ) 
  (day_150_N_plus_1 : ℕ) 
  (h1 : day_250_N = 3)  -- 250th day of year N is Wednesday (3rd day of week, 0 = Sunday)
  (h2 : day_150_N_plus_1 = 3) -- 150th day of year N+1 is also Wednesday (3rd day of week, 0 = Sunday)
  : day_of_week 50 (day_of_week 1 ((day_of_week 1 day_250_N - 1 + 250) % 365 - 1 + 366)) = 6 := 
sorry

-- Explanation:
-- day_of_week function calculates the day of the week given the nth day of the year and the first day of the year.
-- Given conditions that 250th day of year N and 150th day of year N+1 are both Wednesdays (represented by 3 assuming Sunday = 0).
-- We need to derive that the 50th day of year N-1 is a Saturday (represented by 6 assuming Sunday = 0).

end day_of_week_50th_day_of_year_N_minus_1_l2158_215851


namespace marbles_per_customer_l2158_215883

theorem marbles_per_customer
  (initial_marbles remaining_marbles customers marbles_per_customer : ℕ)
  (h1 : initial_marbles = 400)
  (h2 : remaining_marbles = 100)
  (h3 : customers = 20)
  (h4 : initial_marbles - remaining_marbles = customers * marbles_per_customer) :
  marbles_per_customer = 15 :=
by
  sorry

end marbles_per_customer_l2158_215883


namespace car_rent_per_day_leq_30_l2158_215891

variable (D : ℝ) -- daily rental rate
variable (cost_per_mile : ℝ := 0.23) -- cost per mile
variable (daily_budget : ℝ := 76) -- daily budget
variable (distance : ℝ := 200) -- distance driven

theorem car_rent_per_day_leq_30 :
  D + cost_per_mile * distance ≤ daily_budget → D ≤ 30 :=
sorry

end car_rent_per_day_leq_30_l2158_215891


namespace probability_of_odd_sum_given_even_product_l2158_215818

open Nat

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_outcomes := total_outcomes - odd_outcomes
  let favorable_outcomes := 15 * 3^5
  favorable_outcomes / even_outcomes

theorem probability_of_odd_sum_given_even_product :
  probability_odd_sum_given_even_product = 91 / 324 :=
by
  sorry

end probability_of_odd_sum_given_even_product_l2158_215818


namespace mean_of_xyz_l2158_215850

theorem mean_of_xyz (a b c d e f g x y z : ℝ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 48)
  (h2 : (a + b + c + d + e + f + g + x + y + z) / 10 = 55) :
  (x + y + z) / 3 = 71.33333333333333 :=
by
  sorry

end mean_of_xyz_l2158_215850


namespace increase_by_multiplication_l2158_215833

theorem increase_by_multiplication (n : ℕ) (h : n = 14) : (15 * n) - n = 196 :=
by
  -- Skip the proof
  sorry

end increase_by_multiplication_l2158_215833


namespace class_duration_l2158_215831

theorem class_duration (x : ℝ) (h : 3 * x = 6) : x = 2 :=
by
  sorry

end class_duration_l2158_215831


namespace expected_value_is_correct_l2158_215813

noncomputable def expected_value_of_heads : ℝ :=
  let penny := 1 / 2 * 1
  let nickel := 1 / 2 * 5
  let dime := 1 / 2 * 10
  let quarter := 1 / 2 * 25
  let half_dollar := 1 / 2 * 50
  (penny + nickel + dime + quarter + half_dollar : ℝ)

theorem expected_value_is_correct : expected_value_of_heads = 45.5 := by
  sorry

end expected_value_is_correct_l2158_215813


namespace no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l2158_215856

-- Conditions: Expressing the sum of three reciprocals
def sum_of_reciprocals (a b c : ℕ) : ℚ := (1 / a) + (1 / b) + (1 / c)

-- Proof Problem 1: Prove that the sum of the reciprocals of any three positive integers cannot equal 9/11
theorem no_three_reciprocals_sum_to_nine_eleven :
  ∀ (a b c : ℕ), sum_of_reciprocals a b c ≠ 9 / 11 := sorry

-- Proof Problem 2: Prove that there exists no rational number between 41/42 and 1 that can be expressed as the sum of the reciprocals of three positive integers other than 41/42
theorem no_rational_between_fortyone_fortytwo_and_one :
  ∀ (K : ℚ), 41 / 42 < K ∧ K < 1 → ¬ (∃ (a b c : ℕ), sum_of_reciprocals a b c = K) := sorry

end no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l2158_215856


namespace isosceles_triangle_base_angles_l2158_215895

theorem isosceles_triangle_base_angles (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B ∨ B = C ∨ C = A) (h₃ : A = 80 ∨ B = 80 ∨ C = 80) :
  A = 50 ∨ B = 50 ∨ C = 50 ∨ A = 80 ∨ B = 80 ∨ C = 80 := 
by
  sorry

end isosceles_triangle_base_angles_l2158_215895


namespace sum_of_ratios_of_squares_l2158_215803

theorem sum_of_ratios_of_squares (r : ℚ) (a b c : ℤ) (h1 : r = 45 / 64) 
  (h2 : r = (a * (Real.sqrt b)) / c) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hc : c = 8) : a + b + c = 16 := 
by
  sorry

end sum_of_ratios_of_squares_l2158_215803


namespace average_salary_l2158_215886

theorem average_salary (A B C D E : ℕ) (hA : A = 8000) (hB : B = 5000) (hC : C = 14000) (hD : D = 7000) (hE : E = 9000) :
  (A + B + C + D + E) / 5 = 8800 :=
by
  -- the proof will be inserted here
  sorry

end average_salary_l2158_215886


namespace flight_distance_each_way_l2158_215859

variables (D : ℝ) (T_out T_return total_time : ℝ)

-- Defining conditions
def condition1 : Prop := T_out = D / 300
def condition2 : Prop := T_return = D / 500
def condition3 : Prop := total_time = 8

-- Given conditions
axiom h1 : condition1 D T_out
axiom h2 : condition2 D T_return
axiom h3 : condition3 total_time

-- The proof problem statement
theorem flight_distance_each_way : T_out + T_return = total_time → D = 1500 :=
by
  sorry

end flight_distance_each_way_l2158_215859


namespace find_subsequence_with_sum_n_l2158_215801

theorem find_subsequence_with_sum_n (n : ℕ) (a : Fin n → ℕ) (h1 : ∀ i, a i ∈ Finset.range n) 
  (h2 : (Finset.univ.sum a) < 2 * n) : 
  ∃ s : Finset (Fin n), s.sum a = n := 
sorry

end find_subsequence_with_sum_n_l2158_215801


namespace total_puppies_count_l2158_215837

def first_week_puppies : Nat := 20
def second_week_puppies : Nat := 2 * first_week_puppies / 5
def third_week_puppies : Nat := 3 * second_week_puppies / 8
def fourth_week_puppies : Nat := 2 * second_week_puppies
def fifth_week_puppies : Nat := first_week_puppies + 10
def sixth_week_puppies : Nat := 2 * third_week_puppies - 5
def seventh_week_puppies : Nat := 2 * sixth_week_puppies
def eighth_week_puppies : Nat := 5 * seventh_week_puppies / 6 / 1 -- Assuming rounding down to nearest whole number

def total_puppies : Nat :=
  first_week_puppies + second_week_puppies + third_week_puppies +
  fourth_week_puppies + fifth_week_puppies + sixth_week_puppies +
  seventh_week_puppies + eighth_week_puppies

theorem total_puppies_count : total_puppies = 81 := by
  sorry

end total_puppies_count_l2158_215837


namespace marcus_point_value_l2158_215829

theorem marcus_point_value 
  (team_total_points : ℕ)
  (marcus_percentage : ℚ)
  (three_point_goals : ℕ)
  (num_goals_type2 : ℕ)
  (score_type1 : ℕ)
  (score_type2 : ℕ)
  (total_marcus_points : ℚ)
  (points_type2 : ℚ)
  (three_point_value : ℕ := 3):
  team_total_points = 70 →
  marcus_percentage = 0.5 →
  three_point_goals = 5 →
  num_goals_type2 = 10 →
  total_marcus_points = marcus_percentage * team_total_points →
  score_type1 = three_point_goals * three_point_value →
  points_type2 = total_marcus_points - score_type1 →
  score_type2 = points_type2 / num_goals_type2 →
  score_type2 = 2 :=
by
  intros
  sorry

end marcus_point_value_l2158_215829


namespace binomial_theorem_example_l2158_215854

theorem binomial_theorem_example 
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : (2 - 1)^5 = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5)
  (h2 : (2 - (-1))^5 = a_0 - a_1 + a_2 * (-1)^2 - a_3 * (-1)^3 + a_4 * (-1)^4 - a_5 * (-1)^5)
  (h3 : a_5 = -1) :
  (a_0 + a_2 + a_4 : ℤ) / (a_1 + a_3 : ℤ) = -61 / 60 := 
sorry

end binomial_theorem_example_l2158_215854


namespace hyperbola_asymptotes_angle_l2158_215802

-- Define the given conditions and the proof problem
theorem hyperbola_asymptotes_angle (a b c : ℝ) (e : ℝ) (h1 : e = 2) 
  (h2 : e = c / a) (h3 : c = 2 * a) (h4 : b^2 + a^2 = c^2) : 
  ∃ θ : ℝ, θ = 60 :=
by 
  sorry -- Proof is omitted

end hyperbola_asymptotes_angle_l2158_215802


namespace ratio_of_areas_l2158_215809

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l2158_215809


namespace xiao_ming_total_evaluation_score_l2158_215844

theorem xiao_ming_total_evaluation_score 
  (regular midterm final : ℤ) (weight_regular weight_midterm weight_final : ℕ)
  (h1 : regular = 80)
  (h2 : midterm = 90)
  (h3 : final = 85)
  (h_weight_regular : weight_regular = 3)
  (h_weight_midterm : weight_midterm = 3)
  (h_weight_final : weight_final = 4) :
  (regular * weight_regular + midterm * weight_midterm + final * weight_final) /
    (weight_regular + weight_midterm + weight_final) = 85 :=
by
  sorry

end xiao_ming_total_evaluation_score_l2158_215844


namespace find_n_l2158_215847

-- Definitions based on conditions
variable (n : ℕ)  -- number of persons
variable (A : Fin n → Finset (Fin n))  -- acquaintance relation, specified as a set of neighbors for each person
-- Condition 1: Each person is acquainted with exactly 8 others
def acquaintances := ∀ i : Fin n, (A i).card = 8
-- Condition 2: Any two acquainted persons have exactly 4 common acquaintances
def common_acquaintances_adj := ∀ i j : Fin n, i ≠ j → j ∈ (A i) → (A i ∩ A j).card = 4
-- Condition 3: Any two non-acquainted persons have exactly 2 common acquaintances
def common_acquaintances_non_adj := ∀ i j : Fin n, i ≠ j → j ∉ (A i) → (A i ∩ A j).card = 2

-- Statement to prove
theorem find_n (h1 : acquaintances n A) (h2 : common_acquaintances_adj n A) (h3 : common_acquaintances_non_adj n A) :
  n = 21 := 
sorry

end find_n_l2158_215847


namespace three_segments_form_triangle_l2158_215862

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem three_segments_form_triangle :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 2 4 6 ∧
  ¬ can_form_triangle 2 2 4 ∧
    can_form_triangle 6 6 6 :=
by
  repeat {sorry}

end three_segments_form_triangle_l2158_215862


namespace Janet_previous_movie_length_l2158_215896

theorem Janet_previous_movie_length (L : ℝ) (H1 : 1.60 * L = 1920 / 100) : L / 60 = 0.20 :=
by
  sorry

end Janet_previous_movie_length_l2158_215896


namespace find_larger_number_of_two_l2158_215898

theorem find_larger_number_of_two (A B : ℕ) (hcf lcm : ℕ) (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 13)
  (h_factor2 : factor2 = 16)
  (h_lcm : lcm = hcf * factor1 * factor2)
  (h_A : A = hcf * m ∧ m = factor1)
  (h_B : B = hcf * n ∧ n = factor2):
  max A B = 368 := by
  sorry

end find_larger_number_of_two_l2158_215898


namespace dot_product_sum_eq_fifteen_l2158_215871

-- Define the vectors a, b, and c
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1, y)
def vec_c : ℝ × ℝ := (3, -6)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditions from the problem
def cond_perpendicular (x : ℝ) : Prop :=
  dot_product (vec_a x) vec_c = 0

def cond_parallel (y : ℝ) : Prop :=
  1 / 3 = y / -6

-- Lean statement for the problem
theorem dot_product_sum_eq_fifteen (x y : ℝ)
  (h1 : cond_perpendicular x) 
  (h2 : cond_parallel y) :
  dot_product (vec_a x + vec_b y) vec_c = 15 :=
sorry

end dot_product_sum_eq_fifteen_l2158_215871


namespace carnival_wait_time_l2158_215832

theorem carnival_wait_time :
  ∀ (T : ℕ), 4 * 60 = 4 * 30 + T + 4 * 15 → T = 60 :=
by
  intro T
  intro h
  sorry

end carnival_wait_time_l2158_215832


namespace prime_sum_mod_eighth_l2158_215875

theorem prime_sum_mod_eighth (p1 p2 p3 p4 p5 p6 p7 p8 : ℕ) 
  (h₁ : p1 = 2) 
  (h₂ : p2 = 3) 
  (h₃ : p3 = 5) 
  (h₄ : p4 = 7) 
  (h₅ : p5 = 11) 
  (h₆ : p6 = 13) 
  (h₇ : p7 = 17) 
  (h₈ : p8 = 19) : 
  ((p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8) = 1 :=
by
  sorry

end prime_sum_mod_eighth_l2158_215875


namespace total_prayers_in_a_week_l2158_215882

def prayers_per_week (pastor_prayers : ℕ → ℕ) : ℕ :=
  (pastor_prayers 0) + (pastor_prayers 1) + (pastor_prayers 2) +
  (pastor_prayers 3) + (pastor_prayers 4) + (pastor_prayers 5) + (pastor_prayers 6)

def pastor_paul (day : ℕ) : ℕ :=
  if day = 6 then 40 else 20

def pastor_bruce (day : ℕ) : ℕ :=
  if day = 6 then 80 else 10

def pastor_caroline (day : ℕ) : ℕ :=
  if day = 6 then 30 else 10

theorem total_prayers_in_a_week :
  prayers_per_week pastor_paul + prayers_per_week pastor_bruce + prayers_per_week pastor_caroline = 390 :=
sorry

end total_prayers_in_a_week_l2158_215882


namespace intersection_A_B_l2158_215853

open Set

def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_A_B : A ∩ B = { 1 } := by
  sorry

end intersection_A_B_l2158_215853


namespace no_adjacent_standing_probability_l2158_215866

noncomputable def probability_no_adjacent_standing : ℚ := 
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 123
  favorable_outcomes / total_outcomes

theorem no_adjacent_standing_probability :
  probability_no_adjacent_standing = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l2158_215866


namespace tan_75_eq_2_plus_sqrt_3_l2158_215860

theorem tan_75_eq_2_plus_sqrt_3 :
  Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_eq_2_plus_sqrt_3_l2158_215860


namespace find_a_l2158_215846

theorem find_a (a x : ℝ) : 
  ((x + a)^2 / (3 * x + 65) = 2) 
  ∧ (∃ x1 x2 : ℝ,  x1 ≠ x2 ∧ (x1 = x2 + 22 ∨ x2 = x1 + 22 )) 
  → a = 3 := 
sorry

end find_a_l2158_215846


namespace disqualified_team_participants_l2158_215888

theorem disqualified_team_participants
  (initial_teams : ℕ) (initial_avg : ℕ) (final_teams : ℕ) (final_avg : ℕ)
  (total_initial : ℕ) (total_final : ℕ) :
  initial_teams = 9 →
  initial_avg = 7 →
  final_teams = 8 →
  final_avg = 6 →
  total_initial = initial_teams * initial_avg →
  total_final = final_teams * final_avg →
  total_initial - total_final = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end disqualified_team_participants_l2158_215888


namespace range_of_a_sufficient_but_not_necessary_condition_l2158_215808

theorem range_of_a_sufficient_but_not_necessary_condition (a : ℝ) : 
  (-2 < x ∧ x < -1) → ((x + a) * (x + 1) < 0) → (a > 2) :=
sorry

end range_of_a_sufficient_but_not_necessary_condition_l2158_215808


namespace simplify_expression_1_simplify_expression_2_l2158_215877

-- Statement for the first problem
theorem simplify_expression_1 (a : ℝ) : 2 * a * (a - 3) - a^2 = a^2 - 6 * a := 
by sorry

-- Statement for the second problem
theorem simplify_expression_2 (x : ℝ) : (x - 1) * (x + 2) - x * (x + 1) = -2 := 
by sorry

end simplify_expression_1_simplify_expression_2_l2158_215877


namespace g_g_2_eq_78652_l2158_215836

def g (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem g_g_2_eq_78652 : g (g 2) = 78652 := by
  sorry

end g_g_2_eq_78652_l2158_215836


namespace smallestThreeDigitNumberWithPerfectSquare_l2158_215839

def isThreeDigitNumber (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

def formsPerfectSquare (a : ℕ) : Prop := ∃ n : ℕ, 1001 * a + 1 = n * n

theorem smallestThreeDigitNumberWithPerfectSquare :
  ∀ a : ℕ, isThreeDigitNumber a → formsPerfectSquare a → a = 183 :=
by
sorry

end smallestThreeDigitNumberWithPerfectSquare_l2158_215839


namespace pascal_triangle_fifth_number_l2158_215876

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l2158_215876


namespace max_pieces_l2158_215814

namespace CakeProblem

-- Define the dimensions of the cake and the pieces.
def cake_side : ℕ := 16
def piece_side : ℕ := 4

-- Define the areas of the cake and the pieces.
def cake_area : ℕ := cake_side * cake_side
def piece_area : ℕ := piece_side * piece_side

-- State the main problem to prove.
theorem max_pieces : cake_area / piece_area = 16 :=
by
  -- The proof is omitted.
  sorry

end CakeProblem

end max_pieces_l2158_215814


namespace georgie_window_ways_l2158_215852

theorem georgie_window_ways (n : Nat) (h : n = 8) :
  let ways := n * (n - 1)
  ways = 56 := by
  sorry

end georgie_window_ways_l2158_215852


namespace total_money_collected_l2158_215824

def hourly_wage : ℕ := 10 -- Marta's hourly wage 
def tips_collected : ℕ := 50 -- Tips collected by Marta
def hours_worked : ℕ := 19 -- Hours Marta worked

theorem total_money_collected : (hourly_wage * hours_worked + tips_collected = 240) :=
  sorry

end total_money_collected_l2158_215824


namespace combination_mod_100_l2158_215828

def totalDistinctHands : Nat := Nat.choose 60 12

def remainder (n : Nat) (m : Nat) : Nat := n % m

theorem combination_mod_100 :
  remainder totalDistinctHands 100 = R :=
sorry

end combination_mod_100_l2158_215828


namespace part1_part2_l2158_215848

noncomputable def setA : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }
noncomputable def setB (m : ℝ) : Set ℝ := { x | m - 1 < x ∧ x < 2*m + 1 }

theorem part1 (x : ℝ) : 
  setA ∪ setB 3 = { x | -1 ≤ x ∧ x < 7 } :=
sorry

theorem part2 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∃ x, x ∈ setB m ∧ x ∉ setA) ↔ 
  m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) :=
sorry

end part1_part2_l2158_215848


namespace cubic_eq_has_real_roots_l2158_215840

theorem cubic_eq_has_real_roots (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end cubic_eq_has_real_roots_l2158_215840


namespace depth_second_project_l2158_215838

def volume (depth length breadth : ℝ) : ℝ := depth * length * breadth

theorem depth_second_project (D : ℝ) : 
  (volume 100 25 30 = volume D 20 50) → D = 75 :=
by 
  sorry

end depth_second_project_l2158_215838


namespace all_numbers_positive_l2158_215897

noncomputable def condition (a : Fin 9 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 9)), S.card = 4 → S.sum (a : Fin 9 → ℝ) < (Finset.univ \ S).sum (a : Fin 9 → ℝ)

theorem all_numbers_positive (a : Fin 9 → ℝ) (h : condition a) : ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l2158_215897


namespace trains_clear_time_l2158_215820

theorem trains_clear_time :
  ∀ (length_A length_B length_C : ℕ)
    (speed_A_kmph speed_B_kmph speed_C_kmph : ℕ)
    (distance_AB distance_BC : ℕ),
  length_A = 160 ∧ length_B = 320 ∧ length_C = 480 ∧
  speed_A_kmph = 42 ∧ speed_B_kmph = 30 ∧ speed_C_kmph = 48 ∧
  distance_AB = 200 ∧ distance_BC = 300 →
  ∃ (time_clear : ℚ), time_clear = 50.78 :=
by
  intros length_A length_B length_C
         speed_A_kmph speed_B_kmph speed_C_kmph
         distance_AB distance_BC h
  sorry

end trains_clear_time_l2158_215820


namespace hallie_number_of_paintings_sold_l2158_215842

/-- 
Hallie is an artist. She wins an art contest, and she receives a $150 prize. 
She sells some of her paintings for $50 each. 
She makes a total of $300 from her art. 
How many paintings did she sell?
-/
theorem hallie_number_of_paintings_sold 
    (prize : ℕ)
    (price_per_painting : ℕ)
    (total_earnings : ℕ)
    (prize_eq : prize = 150)
    (price_eq : price_per_painting = 50)
    (total_eq : total_earnings = 300) :
    (total_earnings - prize) / price_per_painting = 3 :=
by
  sorry

end hallie_number_of_paintings_sold_l2158_215842


namespace angle_x_is_36_l2158_215892

theorem angle_x_is_36
    (x : ℝ)
    (h1 : 7 * x + 3 * x = 360)
    (h2 : 8 * x ≤ 360) :
    x = 36 := 
by {
  sorry
}

end angle_x_is_36_l2158_215892


namespace directrix_parabola_l2158_215869

theorem directrix_parabola (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end directrix_parabola_l2158_215869


namespace floor_e_eq_2_l2158_215834

noncomputable def e_approx : ℝ := 2.71828

theorem floor_e_eq_2 : ⌊e_approx⌋ = 2 :=
sorry

end floor_e_eq_2_l2158_215834


namespace A_less_B_C_A_relationship_l2158_215874

variable (a : ℝ)
def A := a + 2
def B := 2 * a^2 - 3 * a + 10
def C := a^2 + 5 * a - 3

theorem A_less_B : A a - B a < 0 := by
  sorry

theorem C_A_relationship :
  if a < -5 then C a > A a
  else if a = -5 then C a = A a
  else if a < 1 then C a < A a
  else if a = 1 then C a = A a
  else C a > A a := by
  sorry

end A_less_B_C_A_relationship_l2158_215874


namespace minimum_value_at_zero_l2158_215880

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

theorem minimum_value_at_zero : ∀ x : ℝ, f 0 ≤ f x :=
by
  sorry

end minimum_value_at_zero_l2158_215880


namespace simplify_expression_l2158_215830

theorem simplify_expression (x y : ℝ) : x^2 * y - 3 * x * y^2 + 2 * y * x^2 - y^2 * x = 3 * x^2 * y - 4 * x * y^2 :=
by
  sorry

end simplify_expression_l2158_215830


namespace coloring_scheme_count_l2158_215819

/-- Given the set of points in the Cartesian plane, where each point (m, n) with
    1 <= m, n <= 6 is colored either red or blue, the number of ways to color these points
    such that each unit square has exactly two red vertices is 126. -/
theorem coloring_scheme_count 
  (color : Fin 6 → Fin 6 → Bool)
  (colored_correctly : ∀ m n, (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ 
    (color m n = true ∨ color m n = false) :=
    sorry
  )
  : (∃ valid_coloring : Nat, valid_coloring = 126) :=
  sorry

end coloring_scheme_count_l2158_215819


namespace current_population_l2158_215864

theorem current_population (initial_population deaths_leaving_percentage : ℕ) (current_population : ℕ) :
  initial_population = 3161 → deaths_leaving_percentage = 5 →
  deaths_leaving_percentage / 100 * initial_population + deaths_leaving_percentage * (initial_population - deaths_leaving_percentage / 100 * initial_population) / 100 = initial_population - current_population →
  current_population = 2553 :=
 by
  sorry

end current_population_l2158_215864


namespace mod_remainder_w_l2158_215889

theorem mod_remainder_w (w : ℕ) (h : w = 3^39) : w % 13 = 1 :=
by
  sorry

end mod_remainder_w_l2158_215889


namespace max_modulus_l2158_215815

open Complex

noncomputable def max_modulus_condition (z : ℂ) : Prop :=
  abs (z - (0 + 2*Complex.I)) = 1

theorem max_modulus : ∀ z : ℂ, max_modulus_condition z → abs z ≤ 3 :=
  by sorry

end max_modulus_l2158_215815


namespace ilya_defeats_dragon_l2158_215865

noncomputable def prob_defeat : ℝ := 1 / 4 * 2 + 1 / 3 * 1 + 5 / 12 * 0

theorem ilya_defeats_dragon : prob_defeat = 1 := sorry

end ilya_defeats_dragon_l2158_215865


namespace problem_1992_AHSME_43_l2158_215800

theorem problem_1992_AHSME_43 (a b c : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : Odd a) (h2 : Odd b) : Odd (3^a + (b-1)^2 * c) :=
sorry

end problem_1992_AHSME_43_l2158_215800


namespace evaluate_expression_l2158_215899

theorem evaluate_expression (c : ℕ) (hc : c = 4) : 
  ((c^c - 2 * c * (c-2)^c + c^2)^c) = 431441456 :=
by
  rw [hc]
  sorry

end evaluate_expression_l2158_215899


namespace decagon_not_divided_properly_l2158_215857

theorem decagon_not_divided_properly :
  ∀ (n m : ℕ),
  (∃ black white : Finset ℕ, ∀ b ∈ black, ∀ w ∈ white,
    (b + w = 10) ∧ (b % 3 = 0) ∧ (w % 3 = 0)) →
  n - m = 10 → (n % 3 = 0) ∧ (m % 3 = 0) → 10 % 3 = 0 → False :=
by
  sorry

end decagon_not_divided_properly_l2158_215857


namespace number_of_desired_numbers_l2158_215885

-- Define a predicate for a four-digit number with the thousands digit 3
def isDesiredNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n / 1000) % 10 = 3

-- Statement of the theorem
theorem number_of_desired_numbers : 
  ∃ k, k = 1000 ∧ (∀ n, isDesiredNumber n ↔ 3000 ≤ n ∧ n < 4000) := 
by
  -- Proof omitted, using sorry to skip the proof
  sorry

end number_of_desired_numbers_l2158_215885


namespace A_intersect_B_l2158_215812

def A : Set ℝ := { x | abs x < 2 }
def B : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }

theorem A_intersect_B : A ∩ B = { x | -1 < x ∧ x < 2 } := by
  sorry

end A_intersect_B_l2158_215812


namespace line_does_not_pass_through_point_l2158_215858

theorem line_does_not_pass_through_point 
  (m : ℝ) (h : (2*m + 1)^2 - 4*(m^2 + 4) > 0) : 
  ¬((2*m - 3)*(-2) - 4*m + 7 = 1) :=
by
  sorry

end line_does_not_pass_through_point_l2158_215858


namespace circle_center_l2158_215890

theorem circle_center (x y : ℝ) : ∀ (h k : ℝ), (x^2 - 6*x + y^2 + 2*y = 9) → (x - h)^2 + (y - k)^2 = 19 → h = 3 ∧ k = -1 :=
by
  intros h k h_eq c_eq
  sorry

end circle_center_l2158_215890
