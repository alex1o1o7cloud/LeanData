import Mathlib

namespace star_addition_l238_23890

-- Definition of the binary operation "star"
def star (x y : ℤ) := 5 * x - 2 * y

-- Statement of the problem
theorem star_addition : star 3 4 + star 2 2 = 13 :=
by
  -- By calculation, we have:
  -- star 3 4 = 7 and star 2 2 = 6
  -- Thus, star 3 4 + star 2 2 = 7 + 6 = 13
  sorry

end star_addition_l238_23890


namespace number_value_l238_23826

theorem number_value (N : ℝ) (h : 0.40 * N = 180) : 
  (1/4) * (1/3) * (2/5) * N = 15 :=
by
  -- assume the conditions have been stated correctly
  sorry

end number_value_l238_23826


namespace jay_change_l238_23870

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l238_23870


namespace student_chose_121_l238_23896

theorem student_chose_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := by
  sorry

end student_chose_121_l238_23896


namespace shaded_area_l238_23829

theorem shaded_area (d : ℝ) (k : ℝ) (π : ℝ) (r : ℝ)
  (h_diameter : d = 6) 
  (h_radius_large : k = 5)
  (h_small_radius: r = d / 2) :
  ((π * (k * r)^2) - (π * r^2)) = 216 * π :=
by
  sorry

end shaded_area_l238_23829


namespace pens_sold_during_promotion_l238_23805

theorem pens_sold_during_promotion (x y n : ℕ) 
  (h_profit: 12 * x + 7 * y = 2011)
  (h_n: n = 2 * x + y) : 
  n = 335 := by
  sorry

end pens_sold_during_promotion_l238_23805


namespace kevin_leap_day_2024_is_monday_l238_23877

def days_between_leap_birthdays (years: ℕ) (leap_year_count: ℕ) : ℕ :=
  (years - leap_year_count) * 365 + leap_year_count * 366

def day_of_week_after_days (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

noncomputable def kevin_leap_day_weekday_2024 : ℕ :=
  let days := days_between_leap_birthdays 24 6
  let start_day := 2 -- Tuesday as 2 (assuming 0 = Sunday, 1 = Monday,..., 6 = Saturday)
  day_of_week_after_days start_day days

theorem kevin_leap_day_2024_is_monday :
  kevin_leap_day_weekday_2024 = 1 -- 1 represents Monday
  :=
by
  sorry

end kevin_leap_day_2024_is_monday_l238_23877


namespace solve_equation_l238_23802

theorem solve_equation : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ↔ x = 2 :=
by
  sorry

end solve_equation_l238_23802


namespace fraction_expression_l238_23887

theorem fraction_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3 / 8 := by
  sorry

end fraction_expression_l238_23887


namespace largest_difference_l238_23837

noncomputable def A := 3 * (1003 ^ 1004)
noncomputable def B := 1003 ^ 1004
noncomputable def C := 1002 * (1003 ^ 1003)
noncomputable def D := 3 * (1003 ^ 1003)
noncomputable def E := 1003 ^ 1003
noncomputable def F := 1003 ^ 1002

theorem largest_difference : 
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l238_23837


namespace find_x_plus_y_l238_23848

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 3 * y = 27) (h2 : 3 * x + 5 * y = 1) : x + y = 31 / 17 :=
by
  sorry

end find_x_plus_y_l238_23848


namespace students_attending_swimming_class_l238_23830

theorem students_attending_swimming_class 
  (total_students : ℕ) 
  (chess_percentage : ℕ) 
  (swimming_percentage : ℕ) 
  (number_of_students : ℕ)
  (chess_students := chess_percentage * total_students / 100)
  (swimming_students := swimming_percentage * chess_students / 100) 
  (condition1 : total_students = 2000)
  (condition2 : chess_percentage = 10)
  (condition3 : swimming_percentage = 50)
  (condition4 : number_of_students = chess_students) :
  swimming_students = 100 := 
by 
  sorry

end students_attending_swimming_class_l238_23830


namespace find_p_l238_23880

theorem find_p (n : ℝ) (p : ℝ) (h1 : p = 4 * n * (1 / (2 ^ 2009)) ^ Real.log 1) (h2 : n = 9 / 4) : p = 9 :=
by
  sorry

end find_p_l238_23880


namespace isabella_paintable_area_l238_23834

def total_paintable_area : ℕ :=
  let room1_area := 2 * (14 * 9) + 2 * (12 * 9) - 70
  let room2_area := 2 * (13 * 9) + 2 * (11 * 9) - 70
  let room3_area := 2 * (15 * 9) + 2 * (10 * 9) - 70
  let room4_area := 4 * (12 * 9) - 70
  room1_area + room2_area + room3_area + room4_area

theorem isabella_paintable_area : total_paintable_area = 1502 := by
  sorry

end isabella_paintable_area_l238_23834


namespace abc_values_l238_23806

theorem abc_values (a b c : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hab : b = a^2 / (2 - a^2)) 
  (hbc : c = b^2 / (2 - b^2)) 
  (hca : a = c^2 / (2 - c^2)) : 
  a + b + c = 6 ∨ a + b + c = -4 ∨ a + b + c = -6 :=
sorry

end abc_values_l238_23806


namespace clara_current_age_l238_23845

theorem clara_current_age (a c : ℕ) (h1 : a = 54) (h2 : (c - 41) = 3 * (a - 41)) : c = 80 :=
by
  -- This is where the proof would be constructed.
  sorry

end clara_current_age_l238_23845


namespace rectangle_perimeter_l238_23885

noncomputable def perimeter (a b c : ℕ) : ℕ :=
  2 * (a + b)

theorem rectangle_perimeter (p q: ℕ) (rel_prime: Nat.gcd p q = 1) :
  ∃ (a b c: ℕ), p = 2 * (a + b) ∧ p + q = 52 ∧ a = 5 ∧ b = 12 ∧ c = 7 :=
by
  sorry

end rectangle_perimeter_l238_23885


namespace max_bench_weight_support_l238_23821

/-- Definitions for the given problem conditions -/
def john_weight : ℝ := 250
def bar_weight : ℝ := 550
def total_weight : ℝ := john_weight + bar_weight
def safety_percentage : ℝ := 0.80

/-- Theorem stating the maximum weight the bench can support given the conditions -/
theorem max_bench_weight_support :
  ∀ (W : ℝ), safety_percentage * W = total_weight → W = 1000 :=
by
  sorry

end max_bench_weight_support_l238_23821


namespace domain_of_function_l238_23850

theorem domain_of_function :
  ∀ x : ℝ, (x - 1 ≥ 0) ↔ (x ≥ 1) ∧ (x + 1 ≠ 0) :=
by
  sorry

end domain_of_function_l238_23850


namespace kids_stay_home_lawrence_county_l238_23801

def total_kids_lawrence_county : ℕ := 1201565
def kids_camp_lawrence_county : ℕ := 610769

theorem kids_stay_home_lawrence_county : total_kids_lawrence_county - kids_camp_lawrence_county = 590796 := by
  sorry

end kids_stay_home_lawrence_county_l238_23801


namespace kevin_final_cards_l238_23818

-- Define the initial conditions and problem
def initial_cards : ℕ := 20
def found_cards : ℕ := 47
def lost_cards_1 : ℕ := 7
def lost_cards_2 : ℕ := 12
def won_cards : ℕ := 15

-- Define the function to calculate the final count
def final_cards (initial found lost1 lost2 won : ℕ) : ℕ :=
  (initial + found - lost1 - lost2 + won)

-- Statement of the problem to be proven
theorem kevin_final_cards :
  final_cards initial_cards found_cards lost_cards_1 lost_cards_2 won_cards = 63 :=
by
  sorry

end kevin_final_cards_l238_23818


namespace shaded_area_calculation_l238_23889

noncomputable section

-- Definition of the total area of the grid
def total_area (rows columns : ℕ) : ℝ :=
  rows * columns

-- Definition of the area of a right triangle
def triangle_area (base height : ℕ) : ℝ :=
  1 / 2 * base * height

-- Definition of the shaded area in the grid
def shaded_area (total_area triangle_area : ℝ) : ℝ :=
  total_area - triangle_area

-- Theorem stating the shaded area
theorem shaded_area_calculation :
  let rows := 4
  let columns := 13
  let height := 3
  shaded_area (total_area rows columns) (triangle_area columns height) = 32.5 :=
  sorry

end shaded_area_calculation_l238_23889


namespace horse_revolutions_l238_23853

theorem horse_revolutions (r1 r2 : ℝ) (rev1 rev2 : ℕ) (h1 : r1 = 30) (h2 : rev1 = 25) (h3 : r2 = 10) : 
  rev2 = 75 :=
by 
  sorry

end horse_revolutions_l238_23853


namespace fraction_of_reciprocal_l238_23876

theorem fraction_of_reciprocal (x : ℝ) (hx : 0 < x) (h : (2/3) * x = y / x) (hx1 : x = 1) : y = 2/3 :=
by
  sorry

end fraction_of_reciprocal_l238_23876


namespace calculate_expression_l238_23886

theorem calculate_expression : (36 / (9 + 2 - 6)) * 4 = 28.8 := 
by
    sorry

end calculate_expression_l238_23886


namespace find_other_package_size_l238_23839

variable (total_coffee : ℕ)
variable (total_5_ounce_packages : ℕ)
variable (num_other_packages : ℕ)
variable (other_package_size : ℕ)

theorem find_other_package_size
  (h1 : total_coffee = 85)
  (h2 : total_5_ounce_packages = num_other_packages + 2)
  (h3 : num_other_packages = 5)
  (h4 : 5 * total_5_ounce_packages + other_package_size * num_other_packages = total_coffee) :
  other_package_size = 10 :=
sorry

end find_other_package_size_l238_23839


namespace final_value_after_determinant_and_addition_l238_23892

theorem final_value_after_determinant_and_addition :
  let a := 5
  let b := 7
  let c := 3
  let d := 4
  let det := a * d - b * c
  det + 3 = 2 :=
by
  sorry

end final_value_after_determinant_and_addition_l238_23892


namespace number_of_boys_l238_23811

theorem number_of_boys
  (M W B : Nat)
  (total_earnings wages_of_men earnings_of_men : Nat)
  (num_men_eq_women : 5 * M = W)
  (num_men_eq_boys : 5 * M = B)
  (earnings_eq_90 : total_earnings = 90)
  (men_wages_6 : wages_of_men = 6)
  (men_earnings_eq_30 : earnings_of_men = M * wages_of_men) : 
  B = 5 := 
by
  sorry

end number_of_boys_l238_23811


namespace valid_combination_exists_l238_23833

def exists_valid_combination : Prop :=
  ∃ (a: Fin 7 → ℤ), (a 0 = 1) ∧
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 4) ∧ 
  (a 4 = 5) ∧ (a 5 = 6) ∧ (a 6 = 7) ∧
  ((a 0 = a 1 + a 2 + a 3 + a 4 - a 5 - a 6))

theorem valid_combination_exists :
  exists_valid_combination :=
by
  sorry

end valid_combination_exists_l238_23833


namespace dealer_overall_gain_l238_23861

noncomputable def dealer_gain_percentage (weight1 weight2 : ℕ) (cost_price : ℕ) : ℚ :=
  let actual_weight_sold := weight1 + weight2
  let supposed_weight_sold := 1000 + 1000
  let gain_item1 := cost_price - (weight1 / 1000) * cost_price
  let gain_item2 := cost_price - (weight2 / 1000) * cost_price
  let total_gain := gain_item1 + gain_item2
  let total_actual_cost := (actual_weight_sold / 1000) * cost_price
  (total_gain / total_actual_cost) * 100

theorem dealer_overall_gain :
  dealer_gain_percentage 900 850 100 = 14.29 := 
sorry

end dealer_overall_gain_l238_23861


namespace probability_correct_l238_23868

structure Bag :=
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

def marbles_drawn_sequence (bag : Bag) : ℚ :=
  let total_marbles := bag.blue + bag.green + bag.yellow
  let prob_blue_first := ↑bag.blue / total_marbles
  let prob_green_second := ↑bag.green / (total_marbles - 1)
  let prob_yellow_third := ↑bag.yellow / (total_marbles - 2)
  prob_blue_first * prob_green_second * prob_yellow_third

theorem probability_correct (bag : Bag) (h : bag = ⟨4, 6, 5⟩) : 
  marbles_drawn_sequence bag = 20 / 455 :=
by
  sorry

end probability_correct_l238_23868


namespace rectangle_k_value_l238_23859

theorem rectangle_k_value (x d : ℝ)
  (h_ratio : ∃ x, ∀ l w, l = 5 * x ∧ w = 4 * x)
  (h_diagonal : ∀ l w, l = 5 * x ∧ w = 4 * x → d^2 = (5 * x)^2 + (4 * x)^2)
  (h_area_written : ∃ k, ∀ A, A = (5 * x) * (4 * x) → A = k * d^2) :
  ∃ k, k = 20 / 41 := sorry

end rectangle_k_value_l238_23859


namespace required_fencing_l238_23860

-- Given definitions and conditions
def area (L W : ℕ) : ℕ := L * W

def fencing (W L : ℕ) : ℕ := 2 * W + L

theorem required_fencing
  (L W : ℕ)
  (hL : L = 10)
  (hA : area L W = 600) :
  fencing W L = 130 := by
  sorry

end required_fencing_l238_23860


namespace speedster_convertibles_approx_l238_23843

-- Definitions corresponding to conditions
def total_inventory : ℕ := 120
def num_non_speedsters : ℕ := 40
def num_speedsters : ℕ := 2 * total_inventory / 3
def num_speedster_convertibles : ℕ := 64

-- Theorem statement
theorem speedster_convertibles_approx :
  2 * total_inventory / 3 - num_non_speedsters + num_speedster_convertibles = total_inventory :=
sorry

end speedster_convertibles_approx_l238_23843


namespace problem_solution_l238_23879

theorem problem_solution (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 972) : (x + 2) * (x - 2) = 5 :=
by
  sorry

end problem_solution_l238_23879


namespace find_matrix_triples_elements_l238_23828

theorem find_matrix_triples_elements (M A : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ (a b c d : ℝ), A = ![![a, b], ![c, d]] -> M * A = ![![3 * a, 3 * b], ![3 * c, 3 * d]]) :
  M = ![![3, 0], ![0, 3]] :=
by
  sorry

end find_matrix_triples_elements_l238_23828


namespace cannot_finish_third_l238_23878

-- Definitions for the orders of runners
def order (a b : String) : Prop := a < b

-- The problem statement and conditions
def conditions (P Q R S T U : String) : Prop :=
  order P Q ∧ order P R ∧ order Q S ∧ order P U ∧ order U T ∧ order T Q

theorem cannot_finish_third (P Q R S T U : String) (h : conditions P Q R S T U) :
  (P = "third" → False) ∧ (S = "third" → False) :=
by
  sorry

end cannot_finish_third_l238_23878


namespace initially_tagged_fish_l238_23851

theorem initially_tagged_fish (second_catch_total : ℕ) (second_catch_tagged : ℕ)
  (total_fish_pond : ℕ) (approx_ratio : ℚ) 
  (h1 : second_catch_total = 50)
  (h2 : second_catch_tagged = 2)
  (h3 : total_fish_pond = 1750)
  (h4 : approx_ratio = (second_catch_tagged : ℚ) / second_catch_total) :
  ∃ T : ℕ, T = 70 :=
by
  sorry

end initially_tagged_fish_l238_23851


namespace necessary_but_not_sufficient_condition_l238_23807

theorem necessary_but_not_sufficient_condition (x y : ℝ) : 
  ((x > 1) ∨ (y > 2)) → (x + y > 3) ∧ ¬((x > 1) ∨ (y > 2) ↔ (x + y > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l238_23807


namespace thelma_tomato_count_l238_23844

-- Definitions and conditions
def slices_per_tomato : ℕ := 8
def slices_per_meal_per_person : ℕ := 20
def family_members : ℕ := 8
def total_slices_needed : ℕ := slices_per_meal_per_person * family_members
def tomatoes_needed : ℕ := total_slices_needed / slices_per_tomato

-- Statement of the theorem to be proved
theorem thelma_tomato_count :
  tomatoes_needed = 20 := by
  sorry

end thelma_tomato_count_l238_23844


namespace find_a_l238_23875

def A : Set ℝ := {-1, 0, 1}
noncomputable def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem find_a (a : ℝ) : (A ∩ B a = {0}) → a = -1 := by
  sorry

end find_a_l238_23875


namespace invitees_count_l238_23898

theorem invitees_count 
  (packages : ℕ) 
  (weight_per_package : ℕ) 
  (weight_per_burger : ℕ) 
  (total_people : ℕ)
  (H1 : packages = 4)
  (H2 : weight_per_package = 5)
  (H3 : weight_per_burger = 2)
  (H4 : total_people + 1 = (packages * weight_per_package) / weight_per_burger) :
  total_people = 9 := 
by
  sorry

end invitees_count_l238_23898


namespace trailing_zeros_30_factorial_l238_23869

-- Definitions directly from conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeros (n : ℕ) : ℕ :=
  let count_five_factors (k : ℕ) : ℕ :=
    k / 5 + k / 25 + k / 125 -- This generalizes for higher powers of 5 which is sufficient here.
  count_five_factors n

-- Mathematical proof problem statement
theorem trailing_zeros_30_factorial : trailing_zeros 30 = 7 := by
  sorry

end trailing_zeros_30_factorial_l238_23869


namespace smaller_of_two_digit_numbers_l238_23881

theorem smaller_of_two_digit_numbers (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4725) :
  min a b = 15 :=
sorry

end smaller_of_two_digit_numbers_l238_23881


namespace range_of_a_l238_23838

theorem range_of_a (a : ℝ) :
  (∀ x, (x - 2)/5 + 2 ≤ x - 4/5 ∨ x ≤ a) → a ≥ 3 :=
by
  sorry

end range_of_a_l238_23838


namespace savings_equal_in_820_weeks_l238_23858

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end savings_equal_in_820_weeks_l238_23858


namespace problem_A_inter_B_empty_l238_23810

section

def set_A : Set ℝ := {x | |x| ≥ 2}
def set_B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_A_inter_B_empty : set_A ∩ set_B = ∅ := 
  sorry

end

end problem_A_inter_B_empty_l238_23810


namespace fraction_simplification_l238_23803

theorem fraction_simplification :
  (36 / 19) * (57 / 40) * (95 / 171) = (3 / 2) :=
by
  sorry

end fraction_simplification_l238_23803


namespace find_alpha_plus_beta_l238_23831

variable (α β : ℝ)

def condition_1 : Prop := α^3 - 3*α^2 + 5*α = 1
def condition_2 : Prop := β^3 - 3*β^2 + 5*β = 5

theorem find_alpha_plus_beta (h1 : condition_1 α) (h2 : condition_2 β) : α + β = 2 := 
  sorry

end find_alpha_plus_beta_l238_23831


namespace no_such_p_l238_23899

theorem no_such_p : ¬ ∃ p : ℕ, p > 0 ∧ (∃ k : ℤ, 4 * p + 35 = k * (3 * p - 7)) :=
by
  sorry

end no_such_p_l238_23899


namespace time_for_10_strikes_l238_23871

-- Assume a clock takes 7 seconds to strike 7 times
def clock_time_for_N_strikes (N : ℕ) : ℕ :=
  if N = 7 then 7 else sorry  -- This would usually be a function, simplified here for the specific condition

-- Assume there are 6 intervals for 7 strikes
def intervals_between_strikes (N : ℕ) : ℕ :=
  if N = 7 then 6 else N - 1

-- Function to calculate total time for any number of strikes based on intervals and time per strike
def total_time_for_strikes (N : ℕ) : ℚ :=
  (intervals_between_strikes N) * (clock_time_for_N_strikes 7 / intervals_between_strikes 7 : ℚ)

theorem time_for_10_strikes : total_time_for_strikes 10 = 10.5 :=
by
  -- Insert proof here
  sorry

end time_for_10_strikes_l238_23871


namespace slope_of_tangent_line_at_A_l238_23835

noncomputable def f (x : ℝ) := x^2 + 3 * x

def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (sorry : ℝ)  -- Placeholder for the definition of the derivative

theorem slope_of_tangent_line_at_A : 
  derivative_at f 1 = 5 := 
sorry

end slope_of_tangent_line_at_A_l238_23835


namespace p_is_necessary_but_not_sufficient_for_q_l238_23884

-- Conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0
def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- Proof target
theorem p_is_necessary_but_not_sufficient_for_q : 
  (∀ a : ℝ, p a → q a) ∧ ¬(∀ a : ℝ, q a → p a) :=
sorry

end p_is_necessary_but_not_sufficient_for_q_l238_23884


namespace quadratic_solution_transformation_l238_23814

theorem quadratic_solution_transformation
  (m h k : ℝ)
  (h_nonzero : m ≠ 0)
  (x1 x2 : ℝ)
  (h_sol1 : m * (x1 - h)^2 - k = 0)
  (h_sol2 : m * (x2 - h)^2 - k = 0)
  (h_x1 : x1 = 2)
  (h_x2 : x2 = 5) :
  (∃ x1' x2', x1' = 1 ∧ x2' = 4 ∧ m * (x1' - h + 1)^2 = k ∧ m * (x2' - h + 1)^2 = k) :=
by 
  -- Proof here
  sorry

end quadratic_solution_transformation_l238_23814


namespace find_polynomial_q_l238_23865

theorem find_polynomial_q (q : ℝ → ℝ) :
  (∀ x : ℝ, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = (12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2)) →
  (∀ x : ℝ, q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2) :=
by 
  sorry

end find_polynomial_q_l238_23865


namespace smallest_b_for_factorization_l238_23827

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l238_23827


namespace ratio_of_ages_l238_23842

-- Necessary conditions as definitions in Lean
def combined_age (S D : ℕ) : Prop := S + D = 54
def sam_is_18 (S : ℕ) : Prop := S = 18

-- The statement that we need to prove
theorem ratio_of_ages (S D : ℕ) (h1 : combined_age S D) (h2 : sam_is_18 S) : S / D = 1 / 2 := by
  sorry

end ratio_of_ages_l238_23842


namespace total_children_with_cats_l238_23840

variable (D C B : ℕ)
variable (h1 : D = 18)
variable (h2 : B = 6)
variable (h3 : D + C + B = 30)

theorem total_children_with_cats : C + B = 12 := by
  sorry

end total_children_with_cats_l238_23840


namespace percentage_cut_l238_23822

theorem percentage_cut (S C : ℝ) (hS : S = 940) (hC : C = 611) :
  (C / S) * 100 = 65 := 
by
  rw [hS, hC]
  norm_num

end percentage_cut_l238_23822


namespace binary_to_decimal_1100_l238_23812

-- Define the binary number 1100
def binary_1100 : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0

-- State the theorem that we need to prove
theorem binary_to_decimal_1100 : binary_1100 = 12 := by
  rw [binary_1100]
  sorry

end binary_to_decimal_1100_l238_23812


namespace transform_to_zero_set_l238_23867

def S (p : ℕ) : Finset ℕ := Finset.range p

def P (p : ℕ) (x : ℕ) : ℕ := 3 * x ^ ((2 * p - 1) / 3) + x ^ ((p + 1) / 3) + x + 1

def remainder (n p : ℕ) : ℕ := n % p

theorem transform_to_zero_set (p k : ℕ) (hp : Nat.Prime p) (h_cong : p % 3 = 2) (hk : 0 < k) :
  (∃ n : ℕ, ∀ i ∈ S p, remainder (P p i) p = n) ∨ (∃ n : ℕ, ∀ i ∈ S p, remainder (i ^ k) p = n) ↔
  Nat.gcd k (p - 1) > 1 :=
sorry

end transform_to_zero_set_l238_23867


namespace find_value_of_N_l238_23863

theorem find_value_of_N (N : ℝ) : 
  2 * ((3.6 * N * 2.50) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002 → 
  N = 0.4800000000000001 :=
by
  sorry

end find_value_of_N_l238_23863


namespace arccos_cos_of_11_l238_23893

-- Define the initial conditions
def angle_in_radians (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 * Real.pi

def arccos_principal_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ Real.pi

-- Define the main theorem to be proved
theorem arccos_cos_of_11 :
  angle_in_radians 11 →
  arccos_principal_range (Real.arccos (Real.cos 11)) →
  Real.arccos (Real.cos 11) = 4.71682 :=
by
  -- Proof is not required
  sorry

end arccos_cos_of_11_l238_23893


namespace range_of_m_l238_23815

open Set

variable {m : ℝ}

def A : Set ℝ := { x | x^2 < 16 }
def B (m : ℝ) : Set ℝ := { x | x < m }

theorem range_of_m (h : A ∩ B m = A) : 4 ≤ m :=
by
  sorry

end range_of_m_l238_23815


namespace points_on_line_l238_23854

theorem points_on_line : 
    ∀ (P : ℝ × ℝ),
      (P = (1, 2) ∨ P = (0, 0) ∨ P = (2, 4) ∨ P = (5, 10) ∨ P = (-1, -2))
      → (∃ m b, m = 2 ∧ b = 0 ∧ P.2 = m * P.1 + b) :=
by
  sorry

end points_on_line_l238_23854


namespace sum_of_squares_bounds_l238_23855

-- Given quadrilateral vertices' distances from the nearest vertices of the square
variable (w x y z : ℝ)
-- The side length of the square
def side_length_square : ℝ := 1

-- Expression for the square of each side of the quadrilateral
def square_AB : ℝ := w^2 + x^2
def square_BC : ℝ := (side_length_square - x)^2 + y^2
def square_CD : ℝ := (side_length_square - y)^2 + z^2
def square_DA : ℝ := (side_length_square - z)^2 + (side_length_square - w)^2

-- Sum of the squares of the sides
def sum_of_squares := square_AB w x + square_BC x y + square_CD y z + square_DA z w

-- Proof that the sum of the squares is within the bounds [2, 4]
theorem sum_of_squares_bounds (hw : 0 ≤ w ∧ w ≤ side_length_square)
                              (hx : 0 ≤ x ∧ x ≤ side_length_square)
                              (hy : 0 ≤ y ∧ y ≤ side_length_square)
                              (hz : 0 ≤ z ∧ z ≤ side_length_square)
                              : 2 ≤ sum_of_squares w x y z ∧ sum_of_squares w x y z ≤ 4 := sorry

end sum_of_squares_bounds_l238_23855


namespace rhombus_area_correct_l238_23846

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 30 12 = 180 :=
by
  sorry

end rhombus_area_correct_l238_23846


namespace vector_parallel_l238_23823

theorem vector_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (3 * (2 * x + 1) - 4 * (2 - x) = 0) → (x = 1 / 2) :=
by
  intros a b h
  sorry

end vector_parallel_l238_23823


namespace birds_reduction_on_third_day_l238_23800

theorem birds_reduction_on_third_day
  {a b c : ℕ} 
  (h1 : a = 300)
  (h2 : b = 2 * a)
  (h3 : c = 1300)
  : (b - (c - (a + b))) = 200 :=
by sorry

end birds_reduction_on_third_day_l238_23800


namespace intersection_is_correct_l238_23862

namespace IntervalProofs

def setA := {x : ℝ | 3 * x^2 - 14 * x + 16 ≤ 0}
def setB := {x : ℝ | (3 * x - 7) / x > 0}

theorem intersection_is_correct :
  {x | 7 / 3 < x ∧ x ≤ 8 / 3} = setA ∩ setB :=
by
  sorry

end IntervalProofs

end intersection_is_correct_l238_23862


namespace ratio_AH_HD_triangle_l238_23882

theorem ratio_AH_HD_triangle (BC AC : ℝ) (angleC : ℝ) (H AD HD : ℝ) 
  (hBC : BC = 4) (hAC : AC = 3 * Real.sqrt 2) (hAngleC : angleC = 45) 
  (hAD : AD = 3) (hHD : HD = 1) : 
  (AH / HD) = 2 :=
by
  sorry

end ratio_AH_HD_triangle_l238_23882


namespace grandmother_cheapest_option_l238_23824

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end grandmother_cheapest_option_l238_23824


namespace largest_m_for_game_with_2022_grids_l238_23872

variables (n : ℕ) (f : ℕ → ℕ)

/- Definitions using conditions given -/

/-- Definition of the game and the marking process -/
def game (n : ℕ) : ℕ := 
  if n % 4 = 0 then n / 2 + 1
  else if n % 4 = 2 then n / 2 + 1
  else 0

/-- Main theorem statement -/
theorem largest_m_for_game_with_2022_grids : game 2022 = 1011 :=
by sorry

end largest_m_for_game_with_2022_grids_l238_23872


namespace groupDivisionWays_l238_23856

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end groupDivisionWays_l238_23856


namespace mark_initial_kept_percentage_l238_23817

-- Defining the conditions
def initial_friends : Nat := 100
def remaining_friends : Nat := 70
def percentage_contacted (P : ℝ) := 100 - P
def percentage_responded : ℝ := 0.5

-- Theorem statement: Mark initially kept 40% of his friends
theorem mark_initial_kept_percentage (P : ℝ) : 
  (P / 100 * initial_friends) + (percentage_contacted P / 100 * initial_friends * percentage_responded) = remaining_friends → 
  P = 40 := by
  sorry

end mark_initial_kept_percentage_l238_23817


namespace cubic_has_real_root_l238_23852

open Real

-- Define the conditions
variables (a0 a1 a2 a3 : ℝ) (h : a0 ≠ 0)

-- Define the cubic polynomial function
def cubic (x : ℝ) : ℝ :=
  a0 * x^3 + a1 * x^2 + a2 * x + a3

-- State the theorem
theorem cubic_has_real_root : ∃ x : ℝ, cubic a0 a1 a2 a3 x = 0 :=
by
  sorry

end cubic_has_real_root_l238_23852


namespace smallest_number_gt_sum_digits_1755_l238_23894

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end smallest_number_gt_sum_digits_1755_l238_23894


namespace min_games_needed_l238_23883

theorem min_games_needed (N : ℕ) : 
  (2 + N) * 10 ≥ 9 * (5 + N) ↔ N ≥ 25 := 
by {
  sorry
}

end min_games_needed_l238_23883


namespace sqrt_400_div_2_l238_23841

theorem sqrt_400_div_2 : (Nat.sqrt 400) / 2 = 10 := by
  sorry

end sqrt_400_div_2_l238_23841


namespace real_estate_profit_l238_23891

def purchase_price_first : ℝ := 350000
def purchase_price_second : ℝ := 450000
def purchase_price_third : ℝ := 600000

def gain_first : ℝ := 0.12
def loss_second : ℝ := 0.08
def gain_third : ℝ := 0.18

def selling_price_first : ℝ :=
  purchase_price_first + (purchase_price_first * gain_first)
def selling_price_second : ℝ :=
  purchase_price_second - (purchase_price_second * loss_second)
def selling_price_third : ℝ :=
  purchase_price_third + (purchase_price_third * gain_third)

def total_purchase_price : ℝ :=
  purchase_price_first + purchase_price_second + purchase_price_third
def total_selling_price : ℝ :=
  selling_price_first + selling_price_second + selling_price_third

def overall_gain : ℝ :=
  total_selling_price - total_purchase_price

theorem real_estate_profit :
  overall_gain = 114000 := by
  sorry

end real_estate_profit_l238_23891


namespace dance_team_members_l238_23849

theorem dance_team_members (a b c : ℕ)
  (h1 : a + b + c = 100)
  (h2 : b = 2 * a)
  (h3 : c = 2 * a + 10) :
  c = 46 := by
  sorry

end dance_team_members_l238_23849


namespace counterexample_exists_l238_23836

-- Define prime predicate
def is_prime (n : ℕ) : Prop :=
∀ m, m ∣ n → m = 1 ∨ m = n

def counterexample_to_statement (n : ℕ) : Prop :=
  is_prime n ∧ ¬ is_prime (n + 2)

theorem counterexample_exists : ∃ n ∈ [3, 5, 11, 17, 23], is_prime n ∧ ¬ is_prime (n + 2) :=
by
  sorry

end counterexample_exists_l238_23836


namespace largest_divisor_of_consecutive_even_product_l238_23809

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), ∃ k : ℤ, k = 24 ∧ 
  (2 * n) * (2 * n + 2) * (2 * n + 4) % k = 0 :=
by
  sorry

end largest_divisor_of_consecutive_even_product_l238_23809


namespace max_min_x_plus_y_on_circle_l238_23816

-- Define the conditions
def polar_eq (ρ θ : Real) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the standard form of the circle
def circle_eq (x y : Real) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the parametric equations of the circle
def parametric_eq (α : Real) (x y : Real) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos α ∧ y = 2 + Real.sqrt 2 * Real.sin α

-- Define the problem in Lean
theorem max_min_x_plus_y_on_circle :
  (∀ (ρ θ : Real), polar_eq ρ θ → circle_eq (ρ * Real.cos θ) (ρ * Real.sin θ)) →
  (∀ (α : Real), parametric_eq α (2 + Real.sqrt 2 * Real.cos α) (2 + Real.sqrt 2 * Real.sin α)) →
  (∀ (P : Real × Real), circle_eq P.1 P.2 → 2 ≤ P.1 + P.2 ∧ P.1 + P.2 ≤ 6) :=
by
  intros hpolar hparam P hcircle
  sorry

end max_min_x_plus_y_on_circle_l238_23816


namespace find_f_at_7_l238_23897

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_at_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  sorry

end find_f_at_7_l238_23897


namespace length_of_bridge_l238_23808

theorem length_of_bridge (speed : ℝ) (time_min : ℝ) (length : ℝ)
  (h_speed : speed = 5) (h_time : time_min = 15) :
  length = 1250 :=
sorry

end length_of_bridge_l238_23808


namespace abs_eq_self_nonneg_l238_23895

theorem abs_eq_self_nonneg (x : ℝ) : abs x = x ↔ x ≥ 0 :=
sorry

end abs_eq_self_nonneg_l238_23895


namespace cos_double_angle_zero_l238_23888

theorem cos_double_angle_zero
  (θ : ℝ)
  (a : ℝ×ℝ := (1, -Real.cos θ))
  (b : ℝ×ℝ := (1, 2 * Real.cos θ))
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.cos (2 * θ) = 0 :=
by sorry

end cos_double_angle_zero_l238_23888


namespace distinct_solutions_subtraction_l238_23873

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end distinct_solutions_subtraction_l238_23873


namespace set_intersection_complement_eq_l238_23832

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

noncomputable def complement (U B : Set ℕ) : Set ℕ := { x ∈ U | x ∉ B }

theorem set_intersection_complement_eq : (A ∩ (complement U B)) = {1, 3} := 
by 
  sorry

end set_intersection_complement_eq_l238_23832


namespace total_drink_volume_l238_23847

theorem total_drink_volume (oj wj gj : ℕ) (hoj : oj = 25) (hwj : wj = 40) (hgj : gj = 70) : (gj * 100) / (100 - oj - wj) = 200 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_drink_volume_l238_23847


namespace inequality_of_transformed_division_l238_23866

theorem inequality_of_transformed_division (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (h : A * 5 = B * 4) : A ≤ B := by
  sorry

end inequality_of_transformed_division_l238_23866


namespace florist_total_roses_l238_23825

-- Define the known quantities
def originalRoses : ℝ := 37.0
def firstPick : ℝ := 16.0
def secondPick : ℝ := 19.0

-- The theorem stating the total number of roses
theorem florist_total_roses : originalRoses + firstPick + secondPick = 72.0 :=
  sorry

end florist_total_roses_l238_23825


namespace five_power_l238_23864

theorem five_power (a : ℕ) (h : 5^a = 3125) : 5^(a - 3) = 25 := 
  sorry

end five_power_l238_23864


namespace two_buckets_have_40_liters_l238_23813

def liters_in_jug := 5
def jugs_in_bucket := 4
def liters_in_bucket := liters_in_jug * jugs_in_bucket
def buckets := 2

theorem two_buckets_have_40_liters :
  buckets * liters_in_bucket = 40 :=
by
  sorry

end two_buckets_have_40_liters_l238_23813


namespace remainder_when_divided_by_23_l238_23820

theorem remainder_when_divided_by_23 (y : ℕ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end remainder_when_divided_by_23_l238_23820


namespace chess_tournament_l238_23874

theorem chess_tournament (m p k n : ℕ) 
  (h1 : m * 9 = p * 6) 
  (h2 : m * n = k * 8) 
  (h3 : p * 2 = k * 6) : 
  n = 4 := 
by 
  sorry

end chess_tournament_l238_23874


namespace non_neg_sequence_l238_23857

theorem non_neg_sequence (a : ℝ) (x : ℕ → ℝ) (h0 : x 0 = 0)
  (h1 : ∀ n, x (n + 1) = 1 - a * Real.exp (x n)) (ha : a ≤ 1) :
  ∀ n, x n ≥ 0 := 
  sorry

end non_neg_sequence_l238_23857


namespace genevieve_errors_fixed_l238_23804

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end genevieve_errors_fixed_l238_23804


namespace find_YW_in_triangle_l238_23819

theorem find_YW_in_triangle
  (X Y Z W : Type)
  (d_XZ d_YZ d_XW d_CW : ℝ)
  (h_XZ : d_XZ = 10)
  (h_YZ : d_YZ = 10)
  (h_XW : d_XW = 12)
  (h_CW : d_CW = 5) : 
  YW = 29 / 12 :=
sorry

end find_YW_in_triangle_l238_23819
