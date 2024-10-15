import Mathlib

namespace NUMINAMATH_GPT_poly_remainder_l1656_165643

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end NUMINAMATH_GPT_poly_remainder_l1656_165643


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l1656_165637

theorem breadth_of_rectangular_plot (b l A : ℕ) (h1 : A = 20 * b) (h2 : l = b + 10) 
    (h3 : A = l * b) : b = 10 := by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l1656_165637


namespace NUMINAMATH_GPT_mark_sold_9_boxes_less_than_n_l1656_165622

theorem mark_sold_9_boxes_less_than_n :
  ∀ (n M A : ℕ),
  n = 10 →
  M < n →
  A = n - 2 →
  M + A < n →
  M ≥ 1 →
  A ≥ 1 →
  M = 1 ∧ n - M = 9 :=
by
  intros n M A h_n h_M_lt_n h_A h_MA_lt_n h_M_ge_1 h_A_ge_1
  rw [h_n, h_A] at *
  sorry

end NUMINAMATH_GPT_mark_sold_9_boxes_less_than_n_l1656_165622


namespace NUMINAMATH_GPT_amount_each_student_should_pay_l1656_165608

noncomputable def total_rental_fee_per_book_per_half_hour : ℕ := 4000 
noncomputable def total_books : ℕ := 4
noncomputable def total_students : ℕ := 6
noncomputable def total_hours : ℕ := 3
noncomputable def total_half_hours : ℕ := total_hours * 2

noncomputable def total_fee_one_book : ℕ := total_rental_fee_per_book_per_half_hour * total_half_hours
noncomputable def total_fee_all_books : ℕ := total_fee_one_book * total_books

theorem amount_each_student_should_pay : total_fee_all_books / total_students = 16000 := by
  sorry

end NUMINAMATH_GPT_amount_each_student_should_pay_l1656_165608


namespace NUMINAMATH_GPT_ski_helmet_final_price_l1656_165620

variables (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
def final_price_after_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let after_first_discount := initial_price * (1 - discount1)
  let after_second_discount := after_first_discount * (1 - discount2)
  after_second_discount

theorem ski_helmet_final_price :
  final_price_after_discounts 120 0.40 0.20 = 57.60 := 
  sorry

end NUMINAMATH_GPT_ski_helmet_final_price_l1656_165620


namespace NUMINAMATH_GPT_team_C_has_most_uniform_height_l1656_165615

theorem team_C_has_most_uniform_height
  (S_A S_B S_C S_D : ℝ)
  (h_A : S_A = 0.13)
  (h_B : S_B = 0.11)
  (h_C : S_C = 0.09)
  (h_D : S_D = 0.15)
  (h_same_num_members : ∀ (a b c d : ℕ), a = b ∧ b = c ∧ c = d) 
  : S_C = min S_A (min S_B (min S_C S_D)) :=
by
  sorry

end NUMINAMATH_GPT_team_C_has_most_uniform_height_l1656_165615


namespace NUMINAMATH_GPT_amount_each_person_needs_to_raise_l1656_165638

theorem amount_each_person_needs_to_raise (Total_goal Already_collected Number_of_people : ℝ) 
(h1 : Total_goal = 2400) (h2 : Already_collected = 300) (h3 : Number_of_people = 8) : 
    (Total_goal - Already_collected) / Number_of_people = 262.5 := 
by
  sorry

end NUMINAMATH_GPT_amount_each_person_needs_to_raise_l1656_165638


namespace NUMINAMATH_GPT_yield_percentage_is_correct_l1656_165650

-- Defining the conditions and question
def market_value := 70
def face_value := 100
def dividend_percentage := 7
def annual_dividend := (dividend_percentage * face_value) / 100

-- Lean statement to prove the yield percentage
theorem yield_percentage_is_correct (market_value: ℕ) (annual_dividend: ℝ) : 
  ((annual_dividend / market_value) * 100) = 10 := 
by
  -- conditions from a)
  have market_value := 70
  have face_value := 100
  have dividend_percentage := 7
  have annual_dividend := (dividend_percentage * face_value) / 100
  
  -- proof will go here
  sorry

end NUMINAMATH_GPT_yield_percentage_is_correct_l1656_165650


namespace NUMINAMATH_GPT_necessary_and_sufficient_l1656_165632

variable (α β : ℝ)
variable (p : Prop := α > β)
variable (q : Prop := α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α)

theorem necessary_and_sufficient : (p ↔ q) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_l1656_165632


namespace NUMINAMATH_GPT_difference_between_extrema_l1656_165617

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x

theorem difference_between_extrema (a b : ℝ)
  (h1 : 3 * (2 : ℝ)^2 + 6 * a * (2 : ℝ) + 3 * b = 0)
  (h2 : 3 * (1 : ℝ)^2 + 6 * a * (1 : ℝ) + 3 * b = -3) :
  f 0 a b - f 2 a b = 4 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_extrema_l1656_165617


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1656_165607

theorem max_value_of_quadratic :
  ∃ y : ℚ, ∀ x : ℚ, -x^2 - 3 * x + 4 ≤ y :=
sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1656_165607


namespace NUMINAMATH_GPT_regular_tetrahedron_ratio_l1656_165668

/-- In plane geometry, the ratio of the radius of the circumscribed circle to the 
inscribed circle of an equilateral triangle is 2:1, --/
def ratio_radii_equilateral_triangle : ℚ := 2 / 1

/-- In space geometry, we study the relationship between the radii of the circumscribed
sphere and the inscribed sphere of a regular tetrahedron. --/
def ratio_radii_regular_tetrahedron : ℚ := 3 / 1

/-- Prove the ratio of the radius of the circumscribed sphere to the inscribed sphere
of a regular tetrahedron is 3 : 1, given the ratio is 2 : 1 for the equilateral triangle. --/
theorem regular_tetrahedron_ratio : 
  ratio_radii_equilateral_triangle = 2 / 1 → 
  ratio_radii_regular_tetrahedron = 3 / 1 :=
by
  sorry

end NUMINAMATH_GPT_regular_tetrahedron_ratio_l1656_165668


namespace NUMINAMATH_GPT_apples_per_pie_l1656_165642

theorem apples_per_pie
  (total_apples : ℕ) (apples_handed_out : ℕ) (remaining_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 96)
  (h2 : apples_handed_out = 42)
  (h3 : remaining_apples = total_apples - apples_handed_out)
  (h4 : remaining_apples = 54)
  (h5 : number_of_pies = 9) :
  remaining_apples / number_of_pies = 6 := by
  sorry

end NUMINAMATH_GPT_apples_per_pie_l1656_165642


namespace NUMINAMATH_GPT_min_value_part1_l1656_165677

open Real

theorem min_value_part1 (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_part1_l1656_165677


namespace NUMINAMATH_GPT_order_numbers_l1656_165666

theorem order_numbers : (5 / 2 : ℝ) < (3 : ℝ) ∧ (3 : ℝ) < Real.sqrt (10) := 
by
  sorry

end NUMINAMATH_GPT_order_numbers_l1656_165666


namespace NUMINAMATH_GPT_union_sets_l1656_165695

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_sets : A ∪ B = {2, 3, 5, 6} := by
  sorry

end NUMINAMATH_GPT_union_sets_l1656_165695


namespace NUMINAMATH_GPT_distinct_roots_and_ratios_l1656_165664

open Real

theorem distinct_roots_and_ratios (a b : ℝ) (h1 : a^2 - 3*a - 1 = 0) (h2 : b^2 - 3*b - 1 = 0) (h3 : a ≠ b) :
  b/a + a/b = -11 :=
sorry

end NUMINAMATH_GPT_distinct_roots_and_ratios_l1656_165664


namespace NUMINAMATH_GPT_average_score_of_male_students_l1656_165616

theorem average_score_of_male_students
  (female_students : ℕ) (male_students : ℕ) (female_avg_score : ℕ) (class_avg_score : ℕ)
  (h_female_students : female_students = 20)
  (h_male_students : male_students = 30)
  (h_female_avg_score : female_avg_score = 75)
  (h_class_avg_score : class_avg_score = 72) :
  (30 * (((class_avg_score * (female_students + male_students)) - (female_avg_score * female_students)) / male_students) = 70) :=
by
  -- Sorry for the proof
  sorry

end NUMINAMATH_GPT_average_score_of_male_students_l1656_165616


namespace NUMINAMATH_GPT_remainder_problem_l1656_165656

theorem remainder_problem :
  ((98 * 103 + 7) % 12) = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l1656_165656


namespace NUMINAMATH_GPT_negation_exists_or_l1656_165628

theorem negation_exists_or (x : ℝ) :
  ¬ (∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_GPT_negation_exists_or_l1656_165628


namespace NUMINAMATH_GPT_change_is_41_l1656_165672

-- Define the cost of shirts and sandals as given in the problem conditions
def cost_of_shirts : ℕ := 10 * 5
def cost_of_sandals : ℕ := 3 * 3
def total_cost : ℕ := cost_of_shirts + cost_of_sandals

-- Define the amount given
def amount_given : ℕ := 100

-- Calculate the change
def change := amount_given - total_cost

-- State the theorem
theorem change_is_41 : change = 41 := 
by 
  -- Filling this with justification steps would be the actual proof
  -- but it's not required, so we use 'sorry' to indicate the theorem
  sorry

end NUMINAMATH_GPT_change_is_41_l1656_165672


namespace NUMINAMATH_GPT_value_of_n_l1656_165606

-- Define required conditions
variables (n : ℕ) (f : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : n > 7
axiom cond2 : ∀ m k : ℕ, f m k = 2^(n - m) * Nat.choose m k

-- Given condition
axiom after_seventh_round : f 7 5 = 42

-- Theorem to prove
theorem value_of_n : n = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_n_l1656_165606


namespace NUMINAMATH_GPT_largest_square_side_length_largest_rectangle_dimensions_l1656_165661

variable (a b : ℝ)

-- Part a
theorem largest_square_side_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ s : ℝ, s = (a * b) / (a + b) :=
sorry

-- Part b
theorem largest_rectangle_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, (x = a / 2 ∧ y = b / 2) :=
sorry

end NUMINAMATH_GPT_largest_square_side_length_largest_rectangle_dimensions_l1656_165661


namespace NUMINAMATH_GPT_student_B_incorrect_l1656_165634

-- Define the quadratic function and the non-zero condition on 'a'
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x - 6

-- Conditions stated by the students
def student_A_condition (a b : ℝ) : Prop := -b / (2 * a) = 1
def student_B_condition (a b : ℝ) : Prop := quadratic a b 3 = -6
def student_C_condition (a b : ℝ) : Prop := (4 * a * (-6) - b^2) / (4 * a) = -8
def student_D_condition (a b : ℝ) : Prop := quadratic a b 3 = 0

-- The proof problem: Student B's conclusion is incorrect
theorem student_B_incorrect : 
  ∀ (a b : ℝ), 
  a ≠ 0 → 
  student_A_condition a b ∧ 
  student_C_condition a b ∧ 
  student_D_condition a b → 
  ¬ student_B_condition a b :=
by 
  -- problem converted to Lean problem format 
  -- based on the conditions provided
  sorry

end NUMINAMATH_GPT_student_B_incorrect_l1656_165634


namespace NUMINAMATH_GPT_water_speed_l1656_165611

theorem water_speed (v : ℝ) 
  (still_water_speed : ℝ := 4)
  (distance : ℝ := 10)
  (time : ℝ := 5)
  (effective_speed : ℝ := distance / time) 
  (h : still_water_speed - v = effective_speed) :
  v = 2 :=
by
  sorry

end NUMINAMATH_GPT_water_speed_l1656_165611


namespace NUMINAMATH_GPT_television_combinations_l1656_165651

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem television_combinations :
  ∃ (combinations : ℕ), 
  ∀ (A B total : ℕ), A = 4 → B = 5 → total = 3 →
  combinations = (combination 4 2 * combination 5 1 + combination 4 1 * combination 5 2) →
  combinations = 70 :=
sorry

end NUMINAMATH_GPT_television_combinations_l1656_165651


namespace NUMINAMATH_GPT_mark_total_payment_l1656_165636

def total_cost (work_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_hours * hourly_rate + part_cost

theorem mark_total_payment :
  total_cost 2 75 150 = 300 :=
by
  -- Proof omitted, sorry used to skip the proof
  sorry

end NUMINAMATH_GPT_mark_total_payment_l1656_165636


namespace NUMINAMATH_GPT_apr_sales_is_75_l1656_165613

-- Definitions based on conditions
def sales_jan : ℕ := 90
def sales_feb : ℕ := 50
def sales_mar : ℕ := 70
def avg_sales : ℕ := 72

-- Total sales of first three months
def total_sales_jan_to_mar : ℕ := sales_jan + sales_feb + sales_mar

-- Total sales considering average sales over 5 months
def total_sales : ℕ := avg_sales * 5

-- Defining April sales
def sales_apr (sales_may : ℕ) : ℕ := total_sales - total_sales_jan_to_mar - sales_may

theorem apr_sales_is_75 (sales_may : ℕ) : sales_apr sales_may = 75 :=
by
  unfold sales_apr total_sales total_sales_jan_to_mar avg_sales sales_jan sales_feb sales_mar
  -- Here we could insert more steps if needed to directly connect to the proof
  sorry


end NUMINAMATH_GPT_apr_sales_is_75_l1656_165613


namespace NUMINAMATH_GPT_MrKishore_petrol_expense_l1656_165685

theorem MrKishore_petrol_expense 
  (rent milk groceries education misc savings salary expenses petrol : ℝ)
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_education : education = 2500)
  (h_misc : misc = 700)
  (h_savings : savings = 1800)
  (h_salary : salary = 18000)
  (h_expenses_equation : expenses = rent + milk + groceries + education + petrol + misc)
  (h_savings_equation : savings = salary * 0.10)
  (h_total_equation : salary = expenses + savings) :
  petrol = 2000 :=
by
  sorry

end NUMINAMATH_GPT_MrKishore_petrol_expense_l1656_165685


namespace NUMINAMATH_GPT_solution1_solution2_l1656_165640

noncomputable def Problem1 : ℝ :=
  4 + (-2)^3 * 5 - (-0.28) / 4

theorem solution1 : Problem1 = -35.93 := by
  sorry

noncomputable def Problem2 : ℚ :=
  -1^4 - (1/6) * (2 - (-3)^2)

theorem solution2 : Problem2 = 1/6 := by
  sorry

end NUMINAMATH_GPT_solution1_solution2_l1656_165640


namespace NUMINAMATH_GPT_find_initial_number_l1656_165648

-- Define the initial equation
def initial_equation (x : ℤ) : Prop := x - 12 * 3 * 2 = 9938

-- Prove that the initial number x is equal to 10010 given initial_equation
theorem find_initial_number (x : ℤ) (h : initial_equation x) : x = 10010 :=
sorry

end NUMINAMATH_GPT_find_initial_number_l1656_165648


namespace NUMINAMATH_GPT_point_between_circles_l1656_165618

theorem point_between_circles 
  (a b c x1 x2 : ℝ)
  (ellipse_eq : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (quad_eq : a * x1^2 + b * x1 - c = 0)
  (quad_eq2 : a * x2^2 + b * x2 - c = 0)
  (sum_roots : x1 + x2 = -b / a)
  (prod_roots : x1 * x2 = -c / a) :
  1 < x1^2 + x2^2 ∧ x1^2 + x2^2 < 2 :=
sorry

end NUMINAMATH_GPT_point_between_circles_l1656_165618


namespace NUMINAMATH_GPT_range_of_m_l1656_165624

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1656_165624


namespace NUMINAMATH_GPT_prove_2x_plus_y_le_sqrt_11_l1656_165609

variable (x y : ℝ)
variable (h : 3 * x^2 + 2 * y^2 ≤ 6)

theorem prove_2x_plus_y_le_sqrt_11 : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end NUMINAMATH_GPT_prove_2x_plus_y_le_sqrt_11_l1656_165609


namespace NUMINAMATH_GPT_difference_of_results_l1656_165660

theorem difference_of_results (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (h_diff: a ≠ b) :
  (70 * a - 7 * a) - (70 * b - 7 * b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_results_l1656_165660


namespace NUMINAMATH_GPT_avg_price_of_towels_l1656_165687

def towlesScenario (t1 t2 t3 : ℕ) (price1 price2 price3 : ℕ) : ℕ :=
  (t1 * price1 + t2 * price2 + t3 * price3) / (t1 + t2 + t3)

theorem avg_price_of_towels :
  towlesScenario 3 5 2 100 150 500 = 205 := by
  sorry

end NUMINAMATH_GPT_avg_price_of_towels_l1656_165687


namespace NUMINAMATH_GPT_total_distance_traveled_l1656_165699

-- Define the parameters and conditions
def hoursPerDay : ℕ := 2
def daysPerWeek : ℕ := 5
def daysPeriod1 : ℕ := 3
def daysPeriod2 : ℕ := 2
def speedPeriod1 : ℕ := 12 -- speed in km/h from Monday to Wednesday
def speedPeriod2 : ℕ := 9 -- speed in km/h from Thursday to Friday

-- This is the theorem we want to prove
theorem total_distance_traveled : (daysPeriod1 * hoursPerDay * speedPeriod1) + (daysPeriod2 * hoursPerDay * speedPeriod2) = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l1656_165699


namespace NUMINAMATH_GPT_johns_actual_marks_l1656_165602

def actual_marks (T : ℝ) (x : ℝ) (incorrect : ℝ) (students : ℕ) (avg_increase : ℝ) :=
  (incorrect = 82) ∧ (students = 80) ∧ (avg_increase = 1/2) ∧
  ((T + incorrect) / students = (T + x) / students + avg_increase)

theorem johns_actual_marks (T : ℝ) :
  ∃ x : ℝ, actual_marks T x 82 80 (1/2) ∧ x = 42 :=
by
  sorry

end NUMINAMATH_GPT_johns_actual_marks_l1656_165602


namespace NUMINAMATH_GPT_a_capital_used_l1656_165697

theorem a_capital_used (C P x : ℕ) (h_b_contributes : 3 * C / 4 - C ≥ 0) 
(h_b_receives : 2 * P / 3 - P ≥ 0) 
(h_b_money_used : 10 > 0) 
(h_ratio : 1 / 2 = x / 30) 
: x = 15 :=
sorry

end NUMINAMATH_GPT_a_capital_used_l1656_165697


namespace NUMINAMATH_GPT_sin_double_angle_l1656_165669

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l1656_165669


namespace NUMINAMATH_GPT_noah_ate_burgers_l1656_165644

theorem noah_ate_burgers :
  ∀ (weight_hotdog weight_burger weight_pie : ℕ) 
    (mason_hotdog_weight : ℕ) 
    (jacob_pies noah_burgers mason_hotdogs : ℕ),
    weight_hotdog = 2 →
    weight_burger = 5 →
    weight_pie = 10 →
    (jacob_pies + 3 = noah_burgers) →
    (mason_hotdogs = 3 * jacob_pies) →
    (mason_hotdog_weight = 30) →
    (mason_hotdog_weight / weight_hotdog = mason_hotdogs) →
    noah_burgers = 8 :=
by
  intros weight_hotdog weight_burger weight_pie mason_hotdog_weight
         jacob_pies noah_burgers mason_hotdogs
         h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_noah_ate_burgers_l1656_165644


namespace NUMINAMATH_GPT_fraction_of_green_marbles_half_l1656_165694

-- Definitions based on given conditions
def initial_fraction (x : ℕ) : ℚ := 1 / 3

-- Number of blue, red, and green marbles initially
def blue_marbles (x : ℕ) : ℚ := initial_fraction x * x
def red_marbles (x : ℕ) : ℚ := initial_fraction x * x
def green_marbles (x : ℕ) : ℚ := initial_fraction x * x

-- Number of green marbles after doubling
def doubled_green_marbles (x : ℕ) : ℚ := 2 * green_marbles x

-- New total number of marbles
def new_total_marbles (x : ℕ) : ℚ := blue_marbles x + red_marbles x + doubled_green_marbles x

-- New fraction of green marbles after doubling
def new_fraction_of_green_marbles (x : ℕ) : ℚ := doubled_green_marbles x / new_total_marbles x

theorem fraction_of_green_marbles_half (x : ℕ) (hx : x > 0) :
  new_fraction_of_green_marbles x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_green_marbles_half_l1656_165694


namespace NUMINAMATH_GPT_solve_system_l1656_165679

variable (y : ℝ) (x1 x2 x3 x4 x5 : ℝ)

def system_of_equations :=
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x3

theorem solve_system :
  (y = 2 → x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∧
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) →
   x1 + x2 + x3 + x4 + x5 = 0 ∧ ∀ (x1 x5 : ℝ), system_of_equations y x1 x2 x3 x4 x5) :=
sorry

end NUMINAMATH_GPT_solve_system_l1656_165679


namespace NUMINAMATH_GPT_domain_of_rational_func_l1656_165691

noncomputable def rational_func (x : ℝ) : ℝ := (2 * x ^ 3 - 3 * x ^ 2 + 5 * x - 1) / (x ^ 2 - 5 * x + 6)

theorem domain_of_rational_func : 
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ↔ (∃ y : ℝ, rational_func y = x) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_rational_func_l1656_165691


namespace NUMINAMATH_GPT_jelly_sold_l1656_165600

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end NUMINAMATH_GPT_jelly_sold_l1656_165600


namespace NUMINAMATH_GPT_expression_D_divisible_by_9_l1656_165673

theorem expression_D_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
by
  sorry

end NUMINAMATH_GPT_expression_D_divisible_by_9_l1656_165673


namespace NUMINAMATH_GPT_find_t_from_tan_conditions_l1656_165614

theorem find_t_from_tan_conditions 
  (α t : ℝ)
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + Real.pi / 4) = 4 / t)
  (h3 : Real.tan (α + Real.pi / 4) = (Real.tan (Real.pi / 4) + Real.tan α) / (1 - Real.tan (Real.pi / 4) * Real.tan α)) :
  t = 2 := 
  by
  sorry

end NUMINAMATH_GPT_find_t_from_tan_conditions_l1656_165614


namespace NUMINAMATH_GPT_solution_to_axb_eq_0_l1656_165631

theorem solution_to_axb_eq_0 (a b x : ℝ) (h₀ : a ≠ 0) (h₁ : (0, 4) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) (h₂ : (-3, 0) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) :
  x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_axb_eq_0_l1656_165631


namespace NUMINAMATH_GPT_quadratic_condition_l1656_165625

theorem quadratic_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * x + 3 = 0) → a ≠ 0 :=
by 
  intro h
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_quadratic_condition_l1656_165625


namespace NUMINAMATH_GPT_evaluate_expression_l1656_165690

theorem evaluate_expression :
  (3 / 2) * ((8 / 3) * ((15 / 8) - (5 / 6))) / (((7 / 8) + (11 / 6)) / (13 / 4)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1656_165690


namespace NUMINAMATH_GPT_selling_prices_for_10_percent_profit_l1656_165652

theorem selling_prices_for_10_percent_profit
    (cost1 cost2 cost3 : ℝ)
    (cost1_eq : cost1 = 200)
    (cost2_eq : cost2 = 300)
    (cost3_eq : cost3 = 500)
    (profit_percent : ℝ)
    (profit_percent_eq : profit_percent = 0.10):
    ∃ s1 s2 s3 : ℝ,
      s1 = cost1 + 33.33 ∧
      s2 = cost2 + 33.33 ∧
      s3 = cost3 + 33.33 ∧
      s1 + s2 + s3 = 1100 :=
by
  sorry

end NUMINAMATH_GPT_selling_prices_for_10_percent_profit_l1656_165652


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1656_165682

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_mono : ∀ {x1 x2}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) (h_f1 : f 1 = 0) :
  {x | (x - 1) * f x > 0} = {x | -1 < x ∧ x < 1} ∪ {x | 1 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1656_165682


namespace NUMINAMATH_GPT_cake_eating_classmates_l1656_165601

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end NUMINAMATH_GPT_cake_eating_classmates_l1656_165601


namespace NUMINAMATH_GPT_sally_lost_orange_balloons_l1656_165639

theorem sally_lost_orange_balloons :
  ∀ (initial_orange_balloons lost_orange_balloons current_orange_balloons : ℕ),
  initial_orange_balloons = 9 →
  current_orange_balloons = 7 →
  lost_orange_balloons = initial_orange_balloons - current_orange_balloons →
  lost_orange_balloons = 2 :=
by
  intros initial_orange_balloons lost_orange_balloons current_orange_balloons
  intros h_init h_current h_lost
  rw [h_init, h_current] at h_lost
  exact h_lost

end NUMINAMATH_GPT_sally_lost_orange_balloons_l1656_165639


namespace NUMINAMATH_GPT_floor_square_of_sqrt_50_eq_49_l1656_165621

theorem floor_square_of_sqrt_50_eq_49 : (Int.floor (Real.sqrt 50))^2 = 49 := 
by
  sorry

end NUMINAMATH_GPT_floor_square_of_sqrt_50_eq_49_l1656_165621


namespace NUMINAMATH_GPT_total_marks_scored_l1656_165623

theorem total_marks_scored :
  let Keith_score := 3.5
  let Larry_score := Keith_score * 3.2
  let Danny_score := Larry_score + 5.7
  let Emma_score := (Danny_score * 2) - 1.2
  let Fiona_score := (Keith_score + Larry_score + Danny_score + Emma_score) / 4
  Keith_score + Larry_score + Danny_score + Emma_score + Fiona_score = 80.25 :=
by
  sorry

end NUMINAMATH_GPT_total_marks_scored_l1656_165623


namespace NUMINAMATH_GPT_geometric_sequence_properties_l1656_165654

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ) :
  a 1 = 1 / 2 ∧ a 4 = -4 → q = -2 ∧ (∀ n, a n = 1 / 2 * q ^ (n - 1)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l1656_165654


namespace NUMINAMATH_GPT_balloon_minimum_volume_l1656_165681

theorem balloon_minimum_volume 
  (p V : ℝ)
  (h1 : p * V = 24000)
  (h2 : p ≤ 40000) : 
  V ≥ 0.6 :=
  sorry

end NUMINAMATH_GPT_balloon_minimum_volume_l1656_165681


namespace NUMINAMATH_GPT_initial_number_of_men_l1656_165693

theorem initial_number_of_men (P : ℝ) (M : ℝ) (h1 : P = 15 * M * (P / (15 * M))) (h2 : P = 12.5 * (M + 200) * (P / (12.5 * (M + 200)))) : M = 1000 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l1656_165693


namespace NUMINAMATH_GPT_sum_of_non_solutions_l1656_165658

theorem sum_of_non_solutions (A B C x: ℝ) 
  (h1 : A = 2) 
  (h2 : B = C / 2) 
  (h3 : C = 28) 
  (eq_inf_solutions : ∀ x, (x ≠ -C ∧ x ≠ -14) → 
  (x + B) * (A * x + 56) = 2 * ((x + C) * (x + 14))) : 
  (-14 + -28) = -42 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_non_solutions_l1656_165658


namespace NUMINAMATH_GPT_cylinder_volume_transformation_l1656_165641

theorem cylinder_volume_transformation (π : ℝ) (r h : ℝ) (V : ℝ) (V_new : ℝ)
  (hV : V = π * r^2 * h) (hV_initial : V = 20) : V_new = π * (3 * r)^2 * (4 * h) :=
by
sorry

end NUMINAMATH_GPT_cylinder_volume_transformation_l1656_165641


namespace NUMINAMATH_GPT_find_common_ratio_l1656_165603

variable {a : ℕ → ℝ}
variable (q : ℝ)

-- Definition of geometric sequence condition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions
def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 2 + a 4 = 20) ∧ (a 3 + a 5 = 40)

-- Proposition to be proved
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a q) (h_cond : conditions a q) : q = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1656_165603


namespace NUMINAMATH_GPT_probability_selecting_cooking_l1656_165675

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_probability_selecting_cooking_l1656_165675


namespace NUMINAMATH_GPT_B_correct_A_inter_B_correct_l1656_165698

def A := {x : ℝ | 1 < x ∧ x < 8}
def B := {x : ℝ | x^2 - 5 * x - 14 ≥ 0}

theorem B_correct : B = {x : ℝ | x ≤ -2 ∨ x ≥ 7} := 
sorry

theorem A_inter_B_correct : A ∩ B = {x : ℝ | 7 ≤ x ∧ x < 8} :=
sorry

end NUMINAMATH_GPT_B_correct_A_inter_B_correct_l1656_165698


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1656_165649

-- Given the discriminant condition Δ = b^2 - 4ac > 0
theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) := 
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1656_165649


namespace NUMINAMATH_GPT_length_of_segment_AC_l1656_165605

theorem length_of_segment_AC :
  ∀ (a b h: ℝ),
    (a = b) →
    (h = a * Real.sqrt 2) →
    (4 = (a + b - h) / 2) →
    a = 4 * Real.sqrt 2 + 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_AC_l1656_165605


namespace NUMINAMATH_GPT_opposite_of_neg_three_l1656_165647

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_l1656_165647


namespace NUMINAMATH_GPT_number_line_steps_l1656_165657

theorem number_line_steps (n : ℕ) (total_distance : ℕ) (steps_to_x : ℕ) (x : ℕ)
  (h1 : total_distance = 32)
  (h2 : n = 8)
  (h3 : steps_to_x = 6)
  (h4 : x = (total_distance / n) * steps_to_x) :
  x = 24 := 
sorry

end NUMINAMATH_GPT_number_line_steps_l1656_165657


namespace NUMINAMATH_GPT_find_a_l1656_165627

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -1 - Real.log 2 := sorry

end NUMINAMATH_GPT_find_a_l1656_165627


namespace NUMINAMATH_GPT_sufficient_condition_parallel_planes_l1656_165683

-- Definitions for lines and planes
variable {Line Plane : Type}
variable {m n : Line}
variable {α β : Plane}

-- Relations between lines and planes
variable (parallel_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Condition for sufficient condition for α parallel β
theorem sufficient_condition_parallel_planes
  (h1 : parallel_line m n)
  (h2 : perpendicular_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  parallel_plane α β :=
sorry

end NUMINAMATH_GPT_sufficient_condition_parallel_planes_l1656_165683


namespace NUMINAMATH_GPT_zero_in_M_l1656_165689

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
by
  sorry

end NUMINAMATH_GPT_zero_in_M_l1656_165689


namespace NUMINAMATH_GPT_g_at_8_equals_minus_30_l1656_165670

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_8_equals_minus_30 :
  (∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) →
  g 8 = -30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_g_at_8_equals_minus_30_l1656_165670


namespace NUMINAMATH_GPT_find_t_l1656_165667

open Real

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem find_t (t : ℝ) 
  (area_eq_50 : area_of_triangle 3 15 15 0 0 t = 50) :
  t = 325 / 12 ∨ t = 125 / 12 := 
sorry

end NUMINAMATH_GPT_find_t_l1656_165667


namespace NUMINAMATH_GPT_cone_height_ratio_l1656_165659

theorem cone_height_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rolls_19_times : 19 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  h / r = 6 * Real.sqrt 10 :=
by
  -- problem setup and mathematical manipulations
  sorry

end NUMINAMATH_GPT_cone_height_ratio_l1656_165659


namespace NUMINAMATH_GPT_cubes_difference_l1656_165653

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end NUMINAMATH_GPT_cubes_difference_l1656_165653


namespace NUMINAMATH_GPT_min_value_3x_4y_l1656_165684

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 25 :=
sorry

end NUMINAMATH_GPT_min_value_3x_4y_l1656_165684


namespace NUMINAMATH_GPT_prime_iff_totient_divisor_sum_l1656_165630

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def euler_totient (n : ℕ) : ℕ := sorry  -- we assume implementation of Euler's Totient function
def divisor_sum (n : ℕ) : ℕ := sorry  -- we assume implementation of Divisor sum function

theorem prime_iff_totient_divisor_sum (n : ℕ) :
  (2 ≤ n) → (euler_totient n ∣ (n - 1)) → (n + 1 ∣ divisor_sum n) → is_prime n :=
  sorry

end NUMINAMATH_GPT_prime_iff_totient_divisor_sum_l1656_165630


namespace NUMINAMATH_GPT_factor_determines_d_l1656_165680

theorem factor_determines_d (d : ℚ) :
  (∀ x : ℚ, x - 4 ∣ d * x^3 - 8 * x^2 + 5 * d * x - 12) → d = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_factor_determines_d_l1656_165680


namespace NUMINAMATH_GPT_natural_number_pairs_l1656_165678

theorem natural_number_pairs (a b : ℕ) (p q : ℕ) :
  a ≠ b →
  (∃ p, a + b = 2^p) →
  (∃ q, ab + 1 = 2^q) →
  (a = 1 ∧ b = 2^p - 1 ∨ a = 2^q - 1 ∧ b = 2^q + 1) :=
by intro hne hp hq; sorry

end NUMINAMATH_GPT_natural_number_pairs_l1656_165678


namespace NUMINAMATH_GPT_min_value_75_l1656_165646

def min_value (x y z : ℝ) := x^2 + y^2 + z^2

theorem min_value_75 
  (x y z : ℝ) 
  (h1 : (x + 5) * (y - 5) = 0) 
  (h2 : (y + 5) * (z - 5) = 0) 
  (h3 : (z + 5) * (x - 5) = 0) :
  min_value x y z = 75 := 
sorry

end NUMINAMATH_GPT_min_value_75_l1656_165646


namespace NUMINAMATH_GPT_jury_deliberation_days_l1656_165633

theorem jury_deliberation_days
  (jury_selection_days trial_times jury_duty_days deliberation_hours_per_day hours_in_day : ℕ)
  (h1 : jury_selection_days = 2)
  (h2 : trial_times = 4)
  (h3 : jury_duty_days = 19)
  (h4 : deliberation_hours_per_day = 16)
  (h5 : hours_in_day = 24) :
  (jury_duty_days - jury_selection_days - (trial_times * jury_selection_days)) * deliberation_hours_per_day / hours_in_day = 6 := 
by
  sorry

end NUMINAMATH_GPT_jury_deliberation_days_l1656_165633


namespace NUMINAMATH_GPT_Winnie_the_Pooh_guarantee_kilogram_l1656_165686

noncomputable def guarantee_minimum_honey : Prop :=
  ∃ (a1 a2 a3 a4 a5 : ℝ), 
    a1 + a2 + a3 + a4 + a5 = 3 ∧
    min (min (a1 + a2) (a2 + a3)) (min (a3 + a4) (a4 + a5)) ≥ 1

theorem Winnie_the_Pooh_guarantee_kilogram :
  guarantee_minimum_honey :=
sorry

end NUMINAMATH_GPT_Winnie_the_Pooh_guarantee_kilogram_l1656_165686


namespace NUMINAMATH_GPT_range_of_a_l1656_165671

theorem range_of_a (a : ℝ)
  (A : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0})
  (B : Set ℝ := {x : ℝ | x ≥ a - 1})
  (H : A ∪ B = Set.univ) :
  a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1656_165671


namespace NUMINAMATH_GPT_certain_number_divisible_l1656_165662

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end NUMINAMATH_GPT_certain_number_divisible_l1656_165662


namespace NUMINAMATH_GPT_abs_neg_three_l1656_165604

theorem abs_neg_three : abs (-3) = 3 := 
by 
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1656_165604


namespace NUMINAMATH_GPT_range_of_k_for_obtuse_triangle_l1656_165676

theorem range_of_k_for_obtuse_triangle (k : ℝ) (a b c : ℝ) (h₁ : a = k) (h₂ : b = k + 2) (h₃ : c = k + 4) : 
  2 < k ∧ k < 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_obtuse_triangle_l1656_165676


namespace NUMINAMATH_GPT_intersection_S_T_l1656_165674

def S : Set ℝ := { y | y ≥ 0 }
def T : Set ℝ := { x | x > 1 }

theorem intersection_S_T :
  S ∩ T = { z | z > 1 } :=
sorry

end NUMINAMATH_GPT_intersection_S_T_l1656_165674


namespace NUMINAMATH_GPT_f_5_5_l1656_165692

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_even (x : ℝ) : f x = f (-x) := sorry

lemma f_recurrence (x : ℝ) : f (x + 2) = - (1 / f x) := sorry

lemma f_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) : f x = x := sorry

theorem f_5_5 : f 5.5 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_f_5_5_l1656_165692


namespace NUMINAMATH_GPT_unique_digit_10D4_count_unique_digit_10D4_l1656_165626

theorem unique_digit_10D4 (D : ℕ) (hD : D < 10) : 
  (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0 ↔ D = 4 :=
by
  sorry

theorem count_unique_digit_10D4 :
  ∃! D, (D < 10 ∧ (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0) :=
by
  use 4
  simp [unique_digit_10D4]
  sorry

end NUMINAMATH_GPT_unique_digit_10D4_count_unique_digit_10D4_l1656_165626


namespace NUMINAMATH_GPT_betty_eggs_used_l1656_165696

-- Conditions as definitions
def ratio_sugar_cream_cheese (sugar cream_cheese : ℚ) : Prop :=
  sugar / cream_cheese = 1 / 4

def ratio_vanilla_cream_cheese (vanilla cream_cheese : ℚ) : Prop :=
  vanilla / cream_cheese = 1 / 2

def ratio_eggs_vanilla (eggs vanilla : ℚ) : Prop :=
  eggs / vanilla = 2

-- Given conditions
def sugar_used : ℚ := 2 -- cups of sugar

-- The statement to prove
theorem betty_eggs_used (cream_cheese vanilla eggs : ℚ) 
  (h1 : ratio_sugar_cream_cheese sugar_used cream_cheese)
  (h2 : ratio_vanilla_cream_cheese vanilla cream_cheese)
  (h3 : ratio_eggs_vanilla eggs vanilla) :
  eggs = 8 :=
sorry

end NUMINAMATH_GPT_betty_eggs_used_l1656_165696


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1656_165619

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 8) (h3 : ∃ p q r, p = b ∧ q = b ∧ r = a ∧ p + q > r) : 
  a + b + b = 20 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1656_165619


namespace NUMINAMATH_GPT_find_denomination_of_oliver_bills_l1656_165610

-- Definitions based on conditions
def denomination (x : ℕ) : Prop :=
  let oliver_total := 10 * x + 3 * 5
  let william_total := 15 * 10 + 4 * 5
  oliver_total = william_total + 45

-- The Lean theorem statement
theorem find_denomination_of_oliver_bills (x : ℕ) : denomination x → x = 20 := by
  sorry

end NUMINAMATH_GPT_find_denomination_of_oliver_bills_l1656_165610


namespace NUMINAMATH_GPT_mountain_height_correct_l1656_165612

noncomputable def height_of_mountain : ℝ :=
  15 / (1 / Real.tan (Real.pi * 10 / 180) + 1 / Real.tan (Real.pi * 12 / 180))

theorem mountain_height_correct :
  abs (height_of_mountain - 1.445) < 0.001 :=
sorry

end NUMINAMATH_GPT_mountain_height_correct_l1656_165612


namespace NUMINAMATH_GPT_square_area_increase_l1656_165655

theorem square_area_increase (s : ℝ) (h : s > 0) :
  ((1.15 * s) ^ 2 - s ^ 2) / s ^ 2 * 100 = 32.25 :=
by
  sorry

end NUMINAMATH_GPT_square_area_increase_l1656_165655


namespace NUMINAMATH_GPT_factor_expression_l1656_165629

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end NUMINAMATH_GPT_factor_expression_l1656_165629


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l1656_165635

theorem factorize_difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) :=
sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l1656_165635


namespace NUMINAMATH_GPT_overall_average_score_l1656_165665

theorem overall_average_score
  (mean_morning mean_evening : ℕ)
  (ratio_morning_evening : ℚ) 
  (h1 : mean_morning = 90)
  (h2 : mean_evening = 80)
  (h3 : ratio_morning_evening = 4 / 5) : 
  ∃ overall_mean : ℚ, overall_mean = 84 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_score_l1656_165665


namespace NUMINAMATH_GPT_part1_a1_union_part2_A_subset_complement_B_l1656_165688

open Set Real

-- Definitions for Part (1)
def A : Set ℝ := {x | (x - 1) / (x - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a^2 - 1 < 0}

-- Statement for Part (1)
theorem part1_a1_union (a : ℝ) (h : a = 1) : A ∪ B 1 = {x | 0 < x ∧ x < 5} :=
sorry

-- Definitions for Part (2)
def complement_B (a : ℝ) : Set ℝ := {x | x ≤ a - 1 ∨ x ≥ a + 1}

-- Statement for Part (2)
theorem part2_A_subset_complement_B : (∀ x, (1 < x ∧ x < 5) → (x ≤ a - 1 ∨ x ≥ a + 1)) → (a ≤ 0 ∨ a ≥ 6) :=
sorry

end NUMINAMATH_GPT_part1_a1_union_part2_A_subset_complement_B_l1656_165688


namespace NUMINAMATH_GPT_largest_number_is_B_l1656_165663
open Real

noncomputable def A := 0.989
noncomputable def B := 0.998
noncomputable def C := 0.899
noncomputable def D := 0.9899
noncomputable def E := 0.8999

theorem largest_number_is_B :
  B = max (max (max (max A B) C) D) E :=
by
  sorry

end NUMINAMATH_GPT_largest_number_is_B_l1656_165663


namespace NUMINAMATH_GPT_area_triangle_PCB_correct_l1656_165645

noncomputable def area_of_triangle_PCB (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) : ℝ :=
  6

theorem area_triangle_PCB_correct (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) :
  area_APB = 4 ∧ area_CPD = 9 → area_of_triangle_PCB ABCD A B C D P AB_parallel_CD diagonals_intersect_P area_APB area_CPD = 6 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_PCB_correct_l1656_165645
