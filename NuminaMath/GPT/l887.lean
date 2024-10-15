import Mathlib

namespace NUMINAMATH_GPT_complete_square_q_value_l887_88774

theorem complete_square_q_value :
  ∃ p q, (16 * x^2 - 32 * x - 512 = 0) ∧ ((x + p)^2 = q) → q = 33 := by
  sorry

end NUMINAMATH_GPT_complete_square_q_value_l887_88774


namespace NUMINAMATH_GPT_product_of_real_roots_l887_88773

theorem product_of_real_roots : 
  let f (x : ℝ) := x ^ Real.log x / Real.log 2 
  ∃ r1 r2 : ℝ, (f r1 = 16 ∧ f r2 = 16) ∧ (r1 * r2 = 1) := 
by
  sorry

end NUMINAMATH_GPT_product_of_real_roots_l887_88773


namespace NUMINAMATH_GPT_calculate_expression_l887_88779

theorem calculate_expression : 3 * Real.sqrt 2 - abs (Real.sqrt 2 - Real.sqrt 3) = 4 * Real.sqrt 2 - Real.sqrt 3 :=
  by sorry

end NUMINAMATH_GPT_calculate_expression_l887_88779


namespace NUMINAMATH_GPT_number_of_members_l887_88712

theorem number_of_members (n : ℕ) (h : n^2 = 9216) : n = 96 :=
sorry

end NUMINAMATH_GPT_number_of_members_l887_88712


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l887_88799

theorem simplify_and_evaluate_expression (m : ℕ) (h : m = 2) :
  ( (↑m + 1) / (↑m - 1) + 1 ) / ( (↑m + m^2) / (m^2 - 2*m + 1) ) - ( 2 - 2*↑m ) / ( m^2 - 1 ) = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l887_88799


namespace NUMINAMATH_GPT_polygon_diagonals_with_restricted_vertices_l887_88715

theorem polygon_diagonals_with_restricted_vertices
  (vertices : ℕ) (non_contributing_vertices : ℕ)
  (h_vertices : vertices = 35)
  (h_non_contributing_vertices : non_contributing_vertices = 5) :
  (vertices - non_contributing_vertices) * (vertices - non_contributing_vertices - 3) / 2 = 405 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_diagonals_with_restricted_vertices_l887_88715


namespace NUMINAMATH_GPT_gross_profit_value_l887_88750

theorem gross_profit_value
  (SP : ℝ) (C : ℝ) (GP : ℝ)
  (h1 : SP = 81)
  (h2 : GP = 1.7 * C)
  (h3 : SP = C + GP) :
  GP = 51 :=
by
  sorry

end NUMINAMATH_GPT_gross_profit_value_l887_88750


namespace NUMINAMATH_GPT_solve_for_s_l887_88767

theorem solve_for_s {x : ℝ} (h : 4 * x^2 - 8 * x - 320 = 0) : ∃ s, s = 81 :=
by 
  -- Introduce the conditions and the steps
  sorry

end NUMINAMATH_GPT_solve_for_s_l887_88767


namespace NUMINAMATH_GPT_donut_combinations_l887_88782

theorem donut_combinations (donuts types : ℕ) (at_least_one : ℕ) :
  donuts = 7 ∧ types = 5 ∧ at_least_one = 4 → ∃ combinations : ℕ, combinations = 100 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_donut_combinations_l887_88782


namespace NUMINAMATH_GPT_three_point_seven_five_as_fraction_l887_88727

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end NUMINAMATH_GPT_three_point_seven_five_as_fraction_l887_88727


namespace NUMINAMATH_GPT_sliderB_moves_distance_l887_88796

theorem sliderB_moves_distance :
  ∀ (A B : ℝ) (rod_length : ℝ),
    (A = 20) →
    (B = 15) →
    (rod_length = Real.sqrt (20^2 + 15^2)) →
    (rod_length = 25) →
    (B_new = 25 - 15) →
    B_new = 10 := by
  sorry

end NUMINAMATH_GPT_sliderB_moves_distance_l887_88796


namespace NUMINAMATH_GPT_car_cost_l887_88735

-- Define the weekly allowance in the first year
def first_year_allowance_weekly : ℕ := 50

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Calculate the total first year savings
def first_year_savings : ℕ := first_year_allowance_weekly * weeks_in_year

-- Define the hourly wage and weekly hours worked in the second year
def hourly_wage : ℕ := 9
def weekly_hours_worked : ℕ := 30

-- Calculate the total second year earnings
def second_year_earnings : ℕ := hourly_wage * weekly_hours_worked * weeks_in_year

-- Define the weekly spending in the second year
def weekly_spending : ℕ := 35

-- Calculate the total second year spending
def second_year_spending : ℕ := weekly_spending * weeks_in_year

-- Calculate the total second year savings
def second_year_savings : ℕ := second_year_earnings - second_year_spending

-- Calculate the total savings after two years
def total_savings : ℕ := first_year_savings + second_year_savings

-- Define the additional amount needed
def additional_amount_needed : ℕ := 2000

-- Calculate the total cost of the car
def total_cost_of_car : ℕ := total_savings + additional_amount_needed

-- Theorem statement
theorem car_cost : total_cost_of_car = 16820 := by
  -- The proof is omitted; it is enough to state the theorem
  sorry

end NUMINAMATH_GPT_car_cost_l887_88735


namespace NUMINAMATH_GPT_dilation_result_l887_88723

noncomputable def dilation (c a : ℂ) (k : ℝ) : ℂ := k * (c - a) + a

theorem dilation_result :
  dilation (3 - 1* I) (1 + 2* I) 4 = 9 + 6* I :=
by
  sorry

end NUMINAMATH_GPT_dilation_result_l887_88723


namespace NUMINAMATH_GPT_solve_equations_l887_88769

theorem solve_equations :
  (∀ x, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x, x^2 - 6 * x + 9 = 0 ↔ x = 3) ∧
  (∀ x, x^2 - 7 * x + 12 = 0 ↔ x = 3 ∨ x = 4) ∧
  (∀ x, 2 * x^2 - 3 * x - 5 = 0 ↔ x = 5 / 2 ∨ x = -1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solve_equations_l887_88769


namespace NUMINAMATH_GPT_Milly_took_extra_balloons_l887_88706

theorem Milly_took_extra_balloons :
  let total_packs := 3 + 2
  let balloons_per_pack := 6
  let total_balloons := total_packs * balloons_per_pack
  let even_split := total_balloons / 2
  let Floretta_balloons := 8
  let Milly_extra_balloons := even_split - Floretta_balloons
  Milly_extra_balloons = 7 := by
  sorry

end NUMINAMATH_GPT_Milly_took_extra_balloons_l887_88706


namespace NUMINAMATH_GPT_average_GPA_of_class_l887_88730

theorem average_GPA_of_class (n : ℕ) (h1 : n > 0) 
  (GPA1 : ℝ := 60) (GPA2 : ℝ := 66) 
  (students_ratio1 : ℝ := 1 / 3) (students_ratio2 : ℝ := 2 / 3) :
  let total_students := (students_ratio1 * n + students_ratio2 * n)
  let total_GPA := (students_ratio1 * n * GPA1 + students_ratio2 * n * GPA2)
  let average_GPA := total_GPA / total_students
  average_GPA = 64 := by
    sorry

end NUMINAMATH_GPT_average_GPA_of_class_l887_88730


namespace NUMINAMATH_GPT_simplify_logical_expression_l887_88793

variables (A B C : Bool)

theorem simplify_logical_expression :
  (A && !B || B && !C || B && C || A && B) = (A || B) :=
by { sorry }

end NUMINAMATH_GPT_simplify_logical_expression_l887_88793


namespace NUMINAMATH_GPT_solve_negative_integer_sum_l887_88747

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end NUMINAMATH_GPT_solve_negative_integer_sum_l887_88747


namespace NUMINAMATH_GPT_solution_set_of_even_function_l887_88753

theorem solution_set_of_even_function (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) 
  (h_def : ∀ x, 0 < x → f x = x^2 - 2*x - 3) : 
  { x : ℝ | f x > 0 } = { x | x > 3 } ∪ { x | x < -3 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_even_function_l887_88753


namespace NUMINAMATH_GPT_katie_books_ratio_l887_88709

theorem katie_books_ratio
  (d : ℕ)
  (k : ℚ)
  (g : ℕ)
  (total_books : ℕ)
  (hd : d = 6)
  (hk : ∃ k : ℚ, k = (k : ℚ))
  (hg : g = 5 * (d + k * d))
  (ht : total_books = d + k * d + g)
  (htotal : total_books = 54) :
  k = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_katie_books_ratio_l887_88709


namespace NUMINAMATH_GPT_no_positive_integer_n_exists_l887_88726

theorem no_positive_integer_n_exists {n : ℕ} (hn : n > 0) :
  ¬ ((∃ k, 5 * 10^(k - 1) ≤ 2^n ∧ 2^n < 6 * 10^(k - 1)) ∧
     (∃ m, 2 * 10^(m - 1) ≤ 5^n ∧ 5^n < 3 * 10^(m - 1))) :=
sorry

end NUMINAMATH_GPT_no_positive_integer_n_exists_l887_88726


namespace NUMINAMATH_GPT_replacement_parts_l887_88788

theorem replacement_parts (num_machines : ℕ) (parts_per_machine : ℕ) (week1_fail_rate : ℚ) (week2_fail_rate : ℚ) (week3_fail_rate : ℚ) :
  num_machines = 500 ->
  parts_per_machine = 6 ->
  week1_fail_rate = 0.10 ->
  week2_fail_rate = 0.30 ->
  week3_fail_rate = 0.60 ->
  (num_machines * parts_per_machine) * week1_fail_rate +
  (num_machines * parts_per_machine) * week2_fail_rate +
  (num_machines * parts_per_machine) * week3_fail_rate = 3000 := by
  sorry

end NUMINAMATH_GPT_replacement_parts_l887_88788


namespace NUMINAMATH_GPT_total_seats_l887_88707

theorem total_seats (s : ℕ) 
  (first_class : ℕ := 30) 
  (business_class : ℕ := (20 * s) / 100) 
  (premium_economy : ℕ := 15) 
  (economy_class : ℕ := s - first_class - business_class - premium_economy) 
  (total : first_class + business_class + premium_economy + economy_class = s) 
  : s = 288 := 
sorry

end NUMINAMATH_GPT_total_seats_l887_88707


namespace NUMINAMATH_GPT_rectangle_area_l887_88739

theorem rectangle_area (a b c : ℝ) :
  a = 15 ∧ b = 12 ∧ c = 1 / 3 →
  ∃ (AD AB : ℝ), 
  AD = (180 / 17) ∧ AB = (60 / 17) ∧ 
  (AD * AB = 10800 / 289) :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l887_88739


namespace NUMINAMATH_GPT_sara_pumpkins_l887_88725

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end NUMINAMATH_GPT_sara_pumpkins_l887_88725


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l887_88778

noncomputable def matrix_330 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (330 * Real.pi / 180), -Real.sin (330 * Real.pi / 180)],
    ![Real.sin (330 * Real.pi / 180), Real.cos (330 * Real.pi / 180)]
  ]

theorem smallest_positive_integer_n (n : ℕ) (h : matrix_330 ^ n = 1) : n = 12 := sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l887_88778


namespace NUMINAMATH_GPT_scoring_situations_4_students_l887_88783

noncomputable def number_of_scoring_situations (students : ℕ) (topicA_score : ℤ) (topicB_score : ℤ) : ℕ :=
  let combinations := Nat.choose 4 2
  let first_category := combinations * 2 * 2
  let second_category := 2 * combinations
  first_category + second_category

theorem scoring_situations_4_students : number_of_scoring_situations 4 100 90 = 36 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_scoring_situations_4_students_l887_88783


namespace NUMINAMATH_GPT_susan_hours_per_day_l887_88745

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end NUMINAMATH_GPT_susan_hours_per_day_l887_88745


namespace NUMINAMATH_GPT_factorization_1_factorization_2_l887_88738

variables {x y m n : ℝ}

theorem factorization_1 : x^3 + 2 * x^2 * y + x * y^2 = x * (x + y)^2 :=
sorry

theorem factorization_2 : 4 * m^2 - n^2 - 4 * m + 1 = (2 * m - 1 + n) * (2 * m - 1 - n) :=
sorry

end NUMINAMATH_GPT_factorization_1_factorization_2_l887_88738


namespace NUMINAMATH_GPT_intersection_M_N_l887_88760

-- Definitions of the domains M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | x > 0}

-- The goal is to prove that the intersection of M and N is equal to (0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l887_88760


namespace NUMINAMATH_GPT_best_fitting_model_is_model_3_l887_88757

-- Define models with their corresponding R^2 values
def R_squared_model_1 : ℝ := 0.72
def R_squared_model_2 : ℝ := 0.64
def R_squared_model_3 : ℝ := 0.98
def R_squared_model_4 : ℝ := 0.81

-- Define a proposition that model 3 has the best fitting effect
def best_fitting_model (R1 R2 R3 R4 : ℝ) : Prop :=
  R3 = max (max R1 R2) (max R3 R4)

-- State the theorem that we need to prove
theorem best_fitting_model_is_model_3 :
  best_fitting_model R_squared_model_1 R_squared_model_2 R_squared_model_3 R_squared_model_4 :=
by
  sorry

end NUMINAMATH_GPT_best_fitting_model_is_model_3_l887_88757


namespace NUMINAMATH_GPT_problem_a_problem_b_l887_88722

-- Define the polynomial P(x) = ax^3 + bx
def P (a b x : ℤ) : ℤ := a * x^3 + b * x

-- Define when a pair (a, b) is n-good
def is_ngood (n a b : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ P a b m - P a b k → n ∣ m - k

-- Define when a pair (a, b) is very good
def is_verygood (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ (infinitely_many_n : ℕ), is_ngood n a b

-- Problem (a): Find a pair (1, -51^2) which is 51-good but not very good
theorem problem_a : ∃ a b : ℤ, a = 1 ∧ b = -(51^2) ∧ is_ngood 51 a b ∧ ¬is_verygood a b := 
by {
  sorry
}

-- Problem (b): Show that all 2010-good pairs are very good
theorem problem_b : ∀ a b : ℤ, is_ngood 2010 a b → is_verygood a b := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_a_problem_b_l887_88722


namespace NUMINAMATH_GPT_difference_of_place_values_l887_88765

theorem difference_of_place_values :
  let n := 54179759
  let pos1 := 10000 * 7
  let pos2 := 10 * 7
  pos1 - pos2 = 69930 := by
  sorry

end NUMINAMATH_GPT_difference_of_place_values_l887_88765


namespace NUMINAMATH_GPT_spiral_2018_position_l887_88756

def T100_spiral : Matrix ℕ ℕ ℕ := sorry -- Definition of T100 as a spiral matrix

def pos_2018 := (34, 95) -- The given position we need to prove

theorem spiral_2018_position (i j : ℕ) (h₁ : T100_spiral 34 95 = 2018) : (i, j) = pos_2018 := by  
  sorry

end NUMINAMATH_GPT_spiral_2018_position_l887_88756


namespace NUMINAMATH_GPT_numSpaceDiagonals_P_is_241_l887_88720

noncomputable def numSpaceDiagonals (vertices : ℕ) (edges : ℕ) (tri_faces : ℕ) (quad_faces : ℕ) : ℕ :=
  let total_segments := (vertices * (vertices - 1)) / 2
  let face_diagonals := 2 * quad_faces
  total_segments - edges - face_diagonals

theorem numSpaceDiagonals_P_is_241 :
  numSpaceDiagonals 26 60 24 12 = 241 := by 
  sorry

end NUMINAMATH_GPT_numSpaceDiagonals_P_is_241_l887_88720


namespace NUMINAMATH_GPT_find_even_integer_l887_88768

theorem find_even_integer (x y z : ℤ) (h₁ : Even x) (h₂ : Odd y) (h₃ : Odd z)
  (h₄ : x < y) (h₅ : y < z) (h₆ : y - x > 5) (h₇ : z - x = 9) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_even_integer_l887_88768


namespace NUMINAMATH_GPT_cattle_train_speed_is_correct_l887_88703

-- Given conditions as definitions
def cattle_train_speed (x : ℝ) : ℝ := x
def diesel_train_speed (x : ℝ) : ℝ := x - 33
def cattle_train_distance (x : ℝ) : ℝ := 6 * x
def diesel_train_distance (x : ℝ) : ℝ := 12 * (x - 33)

-- Statement to prove
theorem cattle_train_speed_is_correct (x : ℝ) :
  cattle_train_distance x + diesel_train_distance x = 1284 → 
  x = 93.33 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cattle_train_speed_is_correct_l887_88703


namespace NUMINAMATH_GPT_angle_B_in_triangle_tan_A_given_c_eq_3a_l887_88700

theorem angle_B_in_triangle (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) : B = π / 3 := 
sorry

theorem tan_A_given_c_eq_3a (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) (h2 : c = 3 * a) : 
(Real.tan A) = Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_GPT_angle_B_in_triangle_tan_A_given_c_eq_3a_l887_88700


namespace NUMINAMATH_GPT_max_value_of_trig_expr_l887_88763

theorem max_value_of_trig_expr (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_max_value_of_trig_expr_l887_88763


namespace NUMINAMATH_GPT_determine_a_from_root_l887_88794

noncomputable def quadratic_eq (x a : ℝ) : Prop := x^2 - a = 0

theorem determine_a_from_root :
  (∃ a : ℝ, quadratic_eq 2 a) → (∃ a : ℝ, a = 4) :=
by
  intro h
  obtain ⟨a, ha⟩ := h
  use a
  have h_eq : 2^2 - a = 0 := ha
  linarith

end NUMINAMATH_GPT_determine_a_from_root_l887_88794


namespace NUMINAMATH_GPT_tan_theta_value_l887_88734

theorem tan_theta_value (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_value_l887_88734


namespace NUMINAMATH_GPT_largest_A_l887_88759

namespace EquivalentProofProblem

def F (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f (3 * x) ≥ f (f (2 * x)) + x

theorem largest_A (f : ℝ → ℝ) (hf : F f) (x : ℝ) (hx : x > 0) : 
  ∃ A, (∀ (f : ℝ → ℝ), F f → ∀ x, x > 0 → f x ≥ A * x) ∧ A = 1 / 2 :=
sorry

end EquivalentProofProblem

end NUMINAMATH_GPT_largest_A_l887_88759


namespace NUMINAMATH_GPT_scientific_notation_of_2135_billion_l887_88798

theorem scientific_notation_of_2135_billion :
  (2135 * 10^9 : ℝ) = 2.135 * 10^11 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_2135_billion_l887_88798


namespace NUMINAMATH_GPT_leila_money_left_l887_88710

theorem leila_money_left (initial_money spent_on_sweater spent_on_jewelry total_spent left_money : ℕ) 
  (h1 : initial_money = 160) 
  (h2 : spent_on_sweater = 40) 
  (h3 : spent_on_jewelry = 100) 
  (h4 : total_spent = spent_on_sweater + spent_on_jewelry) 
  (h5 : total_spent = 140) : 
  initial_money - total_spent = 20 := by
  sorry

end NUMINAMATH_GPT_leila_money_left_l887_88710


namespace NUMINAMATH_GPT_LittleRed_system_of_eqns_l887_88701

theorem LittleRed_system_of_eqns :
  ∃ (x y : ℝ), (2/60) * x + (3/60) * y = 1.5 ∧ x + y = 18 :=
sorry

end NUMINAMATH_GPT_LittleRed_system_of_eqns_l887_88701


namespace NUMINAMATH_GPT_slip_2_5_goes_to_B_l887_88744

-- Defining the slips and their values
def slips : List ℝ := [1.5, 2, 2, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

-- Defining the total sum of slips
def total_sum : ℝ := 52

-- Defining the cup sum values
def cup_sums : List ℝ := [11, 10, 9, 8, 7]

-- Conditions: slip with 4 goes into cup A, slip with 5 goes into cup D
def cup_A_contains : ℝ := 4
def cup_D_contains : ℝ := 5

-- Proof statement
theorem slip_2_5_goes_to_B : 
  ∃ (cup_A cup_B cup_C cup_D cup_E : List ℝ), 
    (cup_A.sum = 11 ∧ cup_B.sum = 10 ∧ cup_C.sum = 9 ∧ cup_D.sum = 8 ∧ cup_E.sum = 7) ∧
    (4 ∈ cup_A) ∧ (5 ∈ cup_D) ∧ (2.5 ∈ cup_B) :=
sorry

end NUMINAMATH_GPT_slip_2_5_goes_to_B_l887_88744


namespace NUMINAMATH_GPT_coordinates_of_C_l887_88752

noncomputable def point := (ℚ × ℚ)

def A : point := (2, 8)
def B : point := (6, 14)
def M : point := (4, 11)
def L : point := (6, 6)
def C : point := (14, 2)

-- midpoint formula definition
def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main statement to prove
theorem coordinates_of_C (hM : is_midpoint M A B) : C = (14, 2) :=
  sorry

end NUMINAMATH_GPT_coordinates_of_C_l887_88752


namespace NUMINAMATH_GPT_triangle_median_difference_l887_88777

theorem triangle_median_difference
    (A B C D E : Type)
    (BC_len : BC = 10)
    (AD_len : AD = 6)
    (BE_len : BE = 7.5) :
    ∃ X_max X_min : ℝ, 
    X_max = AB^2 + AC^2 + BC^2 ∧ 
    X_min = AB^2 + AC^2 + BC^2 ∧ 
    (X_max - X_min) = 56.25 :=
by
  sorry

end NUMINAMATH_GPT_triangle_median_difference_l887_88777


namespace NUMINAMATH_GPT_even_multiples_of_25_l887_88719

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0

theorem even_multiples_of_25 (a b : ℕ) (h1 : 249 ≤ a) (h2 : b ≤ 501) :
  (a = 250 ∨ a = 275 ∨ a = 300 ∨ a = 350 ∨ a = 400 ∨ a = 450) →
  (b = 275 ∨ b = 300 ∨ b = 350 ∨ b = 400 ∨ b = 450 ∨ b = 500) →
  (∃ n, n = 5 ∧ ∀ m, (is_multiple_of_25 m ∧ is_even m ∧ a ≤ m ∧ m ≤ b) ↔ m ∈ [a, b]) :=
by sorry

end NUMINAMATH_GPT_even_multiples_of_25_l887_88719


namespace NUMINAMATH_GPT_smaller_pack_size_l887_88740

theorem smaller_pack_size {x : ℕ} (total_eggs large_pack_size large_packs : ℕ) (eggs_in_smaller_packs : ℕ) :
  total_eggs = 79 → large_pack_size = 11 → large_packs = 5 → eggs_in_smaller_packs = total_eggs - large_pack_size * large_packs →
  x * 1 = eggs_in_smaller_packs → x = 24 :=
by sorry

end NUMINAMATH_GPT_smaller_pack_size_l887_88740


namespace NUMINAMATH_GPT_sum_of_squares_l887_88724

variable (a b c : ℝ)
variable (S : ℝ)

theorem sum_of_squares (h1 : ab + bc + ac = 131)
                       (h2 : a + b + c = 22) :
  a^2 + b^2 + c^2 = 222 :=
by
  -- Proof would be placed here
  sorry

end NUMINAMATH_GPT_sum_of_squares_l887_88724


namespace NUMINAMATH_GPT_n19_minus_n7_div_30_l887_88785

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end NUMINAMATH_GPT_n19_minus_n7_div_30_l887_88785


namespace NUMINAMATH_GPT_train_crosses_platform_in_20_seconds_l887_88775

theorem train_crosses_platform_in_20_seconds 
  (t : ℝ) (lp : ℝ) (lt : ℝ) (tp : ℝ) (sp : ℝ) (st : ℝ) 
  (pass_time : st = lt / tp) (lc : lp = 267) (lc_train : lt = 178) (cross_time : t = sp / st) : 
  t = 20 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_in_20_seconds_l887_88775


namespace NUMINAMATH_GPT_distance_between_parallel_lines_eq_l887_88705

open Real

theorem distance_between_parallel_lines_eq
  (h₁ : ∀ (x y : ℝ), 3 * x + y - 3 = 0 → Prop)
  (h₂ : ∀ (x y : ℝ), 6 * x + 2 * y + 1 = 0 → Prop) :
  ∃ d : ℝ, d = (7 / 20) * sqrt 10 :=
sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_eq_l887_88705


namespace NUMINAMATH_GPT_set_inter_compl_eq_l887_88766

def U := ℝ
def M : Set ℝ := { x | abs (x - 1/2) ≤ 5/2 }
def P : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def complement_U_M : Set ℝ := { x | x < -2 ∨ x > 3 }

theorem set_inter_compl_eq :
  (complement_U_M ∩ P) = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_GPT_set_inter_compl_eq_l887_88766


namespace NUMINAMATH_GPT_rosa_total_pages_called_l887_88721

variable (P_last P_this : ℝ)

theorem rosa_total_pages_called (h1 : P_last = 10.2) (h2 : P_this = 8.6) : P_last + P_this = 18.8 :=
by sorry

end NUMINAMATH_GPT_rosa_total_pages_called_l887_88721


namespace NUMINAMATH_GPT_find_bk_l887_88795

theorem find_bk
  (A B C D : ℝ)
  (BC : ℝ) (hBC : BC = 3)
  (AB CD : ℝ) (hAB_CD : AB = 2 * CD)
  (BK : ℝ) (hBK : BK = 2) :
  ∃ x a : ℝ, (x = BK) ∧ (AB = 2 * CD) ∧ ((2 * a + x) * (3 - x) = x * (a + 3 - x)) :=
by
  sorry

end NUMINAMATH_GPT_find_bk_l887_88795


namespace NUMINAMATH_GPT_arithmetic_sqrt_25_l887_88704

-- Define the arithmetic square root condition
def is_arithmetic_sqrt (x a : ℝ) : Prop :=
  0 ≤ x ∧ x^2 = a

-- Lean statement to prove the arithmetic square root of 25 is 5
theorem arithmetic_sqrt_25 : is_arithmetic_sqrt 5 25 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_25_l887_88704


namespace NUMINAMATH_GPT_mod_37_5_l887_88780

theorem mod_37_5 : 37 % 5 = 2 := 
by
  sorry

end NUMINAMATH_GPT_mod_37_5_l887_88780


namespace NUMINAMATH_GPT_subtract_23_result_l887_88748

variable {x : ℕ}

theorem subtract_23_result (h : x + 30 = 55) : x - 23 = 2 :=
sorry

end NUMINAMATH_GPT_subtract_23_result_l887_88748


namespace NUMINAMATH_GPT_domain_of_f_l887_88791

theorem domain_of_f (x : ℝ) : (2*x - x^2 > 0 ∧ x ≠ 1) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_domain_of_f_l887_88791


namespace NUMINAMATH_GPT_sum_of_integers_l887_88713

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 54) : x + y = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l887_88713


namespace NUMINAMATH_GPT_min_possible_value_l887_88749

theorem min_possible_value
  (a b c d e f g h : Int)
  (h_distinct : List.Nodup [a, b, c, d, e, f, g, h])
  (h_set_a : a ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_b : b ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_c : c ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_d : d ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_e : e ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_f : f ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_g : g ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_h : h ∈ [-9, -6, -3, 0, 1, 3, 6, 10]) :
  ∃ a b c d e f g h : Int,
  ((a + b + c + d)^2 + (e + f + g + h)^2) = 2
  :=
  sorry

end NUMINAMATH_GPT_min_possible_value_l887_88749


namespace NUMINAMATH_GPT_find_intended_number_l887_88716

theorem find_intended_number (n : ℕ) (h : 6 * n + 382 = 988) : n = 101 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_intended_number_l887_88716


namespace NUMINAMATH_GPT_trigonometric_identity_l887_88702

theorem trigonometric_identity (α : ℝ) : 
  - (Real.sin α) + (Real.sqrt 3) * (Real.cos α) = 2 * (Real.sin (α + 2 * Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l887_88702


namespace NUMINAMATH_GPT_tan_alpha_eq_neg2_l887_88717

theorem tan_alpha_eq_neg2 {α : ℝ} {x y : ℝ} (hx : x = -2) (hy : y = 4) (hM : (x, y) = (-2, 4)) :
  Real.tan α = -2 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg2_l887_88717


namespace NUMINAMATH_GPT_sum_first_70_odd_eq_4900_l887_88771

theorem sum_first_70_odd_eq_4900 (h : (70 * (70 + 1) = 4970)) :
  (70 * 70 = 4900) :=
by
  sorry

end NUMINAMATH_GPT_sum_first_70_odd_eq_4900_l887_88771


namespace NUMINAMATH_GPT_smallest_x_l887_88772

theorem smallest_x (x : ℤ) (h : x + 3 < 3 * x - 4) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l887_88772


namespace NUMINAMATH_GPT_reinforcement_arrival_days_l887_88714

theorem reinforcement_arrival_days (x : ℕ) (h : x = 2000) (provisions_days : ℕ) (provisions_days_initial : provisions_days = 54) 
(reinforcement : ℕ) (reinforcement_val : reinforcement = 1300) (remaining_days : ℕ) (remaining_days_val : remaining_days = 20) 
(total_men : ℕ) (total_men_val : total_men = 3300) (equation : 2000 * (54 - x) = 3300 * 20) : x = 21 := 
by
  have eq1 : 2000 * 54 - 2000 * x = 3300 * 20 := by sorry
  have eq2 : 108000 - 2000 * x = 66000 := by sorry
  have eq3 : 2000 * x = 42000 := by sorry
  have eq4 : x = 21000 / 2000 := by sorry
  have eq5 : x = 21 := by sorry
  sorry

end NUMINAMATH_GPT_reinforcement_arrival_days_l887_88714


namespace NUMINAMATH_GPT_ab_value_l887_88751

/-- 
  Given the conditions:
  - a - b = 10
  - a^2 + b^2 = 210
  Prove that ab = 55.
-/
theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 210) : a * b = 55 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l887_88751


namespace NUMINAMATH_GPT_only_point_D_lies_on_graph_l887_88781

def point := ℤ × ℤ

def lies_on_graph (f : ℤ → ℤ) (p : point) : Prop :=
  f p.1 = p.2

def f (x : ℤ) : ℤ := 2 * x - 1

theorem only_point_D_lies_on_graph :
  (lies_on_graph f (-1, 3) = false) ∧ 
  (lies_on_graph f (0, 1) = false) ∧ 
  (lies_on_graph f (1, -1) = false) ∧ 
  (lies_on_graph f (2, 3)) := 
by
  sorry

end NUMINAMATH_GPT_only_point_D_lies_on_graph_l887_88781


namespace NUMINAMATH_GPT_width_of_grass_field_l887_88758

-- Define the conditions
def length_of_grass_field : ℝ := 75
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2
def total_cost : ℝ := 1200

-- Define the width of the grass field as a variable
variable (w : ℝ)

-- Define the total length and width including the path
def total_length : ℝ := length_of_grass_field + 2 * path_width
def total_width (w : ℝ) : ℝ := w + 2 * path_width

-- Define the area of the path
def area_of_path (w : ℝ) : ℝ := (total_length * total_width w) - (length_of_grass_field * w)

-- Define the cost equation
def cost_eq (w : ℝ) : Prop := cost_per_sq_m * area_of_path w = total_cost

-- The theorem to prove
theorem width_of_grass_field : cost_eq 40 :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_width_of_grass_field_l887_88758


namespace NUMINAMATH_GPT_neg_distance_represents_west_l887_88746

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end NUMINAMATH_GPT_neg_distance_represents_west_l887_88746


namespace NUMINAMATH_GPT_solve_recursive_fraction_l887_88776

noncomputable def recursive_fraction (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | (n+1) => 1 + 1 / (recursive_fraction n x)

theorem solve_recursive_fraction (x : ℝ) (n : ℕ) :
  (recursive_fraction n x = x) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_solve_recursive_fraction_l887_88776


namespace NUMINAMATH_GPT_eq1_solution_eq2_solution_l887_88754

theorem eq1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 :=
by
  sorry

theorem eq2_solution (x : ℝ) (h : (1 / 2) * x - 6 = (3 / 4) * x) : x = -24 :=
by
  sorry

end NUMINAMATH_GPT_eq1_solution_eq2_solution_l887_88754


namespace NUMINAMATH_GPT_cone_base_radius_l887_88770

theorem cone_base_radius (r_paper : ℝ) (n_parts : ℕ) (r_cone_base : ℝ) 
  (h_radius_paper : r_paper = 16)
  (h_n_parts : n_parts = 4)
  (h_cone_part : r_cone_base = r_paper / n_parts) : r_cone_base = 4 := by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l887_88770


namespace NUMINAMATH_GPT_part1_part2_l887_88764

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x)

theorem part1 : f (Real.pi / 8) = Real.sqrt 2 + 1 := sorry

theorem part2 : (∀ x1 x2 : ℝ, f (x1 + Real.pi) = f x1) ∧ (∀ x : ℝ, f x ≥ 1 - Real.sqrt 2) := 
  sorry

-- Explanation:
-- part1 is for proving f(π/8) = √2 + 1
-- part2 handles proving the smallest positive period and the minimum value of the function.

end NUMINAMATH_GPT_part1_part2_l887_88764


namespace NUMINAMATH_GPT_find_a_find_k_max_l887_88761

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end NUMINAMATH_GPT_find_a_find_k_max_l887_88761


namespace NUMINAMATH_GPT_rectangles_260_261_272_273_have_similar_property_l887_88729

-- Defining a rectangle as a structure with width and height
structure Rectangle where
  width : ℕ
  height : ℕ

-- Given conditions
def r1 : Rectangle := ⟨16, 10⟩
def r2 : Rectangle := ⟨23, 7⟩

-- Hypothesis function indicating the dissection trick causing apparent equality
def dissection_trick (r1 r2 : Rectangle) : Prop :=
  (r1.width * r1.height : ℕ) = (r2.width * r2.height : ℕ) + 1

-- The statement of the proof problem
theorem rectangles_260_261_272_273_have_similar_property :
  ∃ (r3 r4 : Rectangle) (r5 r6 : Rectangle),
    dissection_trick r3 r4 ∧ dissection_trick r5 r6 ∧
    r3.width * r3.height = 260 ∧ r4.width * r4.height = 261 ∧
    r5.width * r5.height = 272 ∧ r6.width * r6.height = 273 :=
  sorry

end NUMINAMATH_GPT_rectangles_260_261_272_273_have_similar_property_l887_88729


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l887_88728

theorem point_in_fourth_quadrant (m : ℝ) (h1 : m + 2 > 0) (h2 : m < 0) : -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l887_88728


namespace NUMINAMATH_GPT_Donggil_cleaning_time_l887_88736

-- Define the total area of the school as A.
variable (A : ℝ)

-- Define the cleaning rates of Daehyeon (D) and Donggil (G).
variable (D G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (D + G) * 8 = (7 / 12) * A
def condition2 : Prop := D * 10 = (5 / 12) * A

-- The goal is to prove that Donggil can clean the entire area alone in 32 days.
theorem Donggil_cleaning_time : condition1 A D G ∧ condition2 A D → 32 * G = A :=
by
  sorry

end NUMINAMATH_GPT_Donggil_cleaning_time_l887_88736


namespace NUMINAMATH_GPT_floor_sqrt_120_eq_10_l887_88790

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end NUMINAMATH_GPT_floor_sqrt_120_eq_10_l887_88790


namespace NUMINAMATH_GPT_tangent_line_at_A_l887_88762

def f (x : ℝ) : ℝ := x ^ (1 / 2)

def tangent_line_equation (x y: ℝ) : Prop :=
  4 * x - 4 * y + 1 = 0

theorem tangent_line_at_A :
  tangent_line_equation (1/4) (f (1/4)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_A_l887_88762


namespace NUMINAMATH_GPT_simplify_and_evaluate_l887_88741

-- Define the expression
def expr (a : ℚ) : ℚ := (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2)

-- Given the condition
def a_value : ℚ := -1 / 3

-- State the theorem
theorem simplify_and_evaluate : expr a_value = 3 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l887_88741


namespace NUMINAMATH_GPT_unique_nets_of_a_cube_l887_88743

-- Definitions based on the conditions and the properties of the cube
def is_net (net: ℕ) : Prop :=
  -- A placeholder definition of a valid net
  sorry

def is_distinct_by_rotation_or_reflection (net1 net2: ℕ) : Prop :=
  -- Two nets are distinct if they cannot be transformed into each other by rotation or reflection
  sorry

-- The statement to be proved
theorem unique_nets_of_a_cube : ∃ n, n = 11 ∧ (∀ net, is_net net → ∃! net', is_net net' ∧ is_distinct_by_rotation_or_reflection net net') :=
sorry

end NUMINAMATH_GPT_unique_nets_of_a_cube_l887_88743


namespace NUMINAMATH_GPT_problem_statement_eq_l887_88786

noncomputable def given_sequence (a : ℝ) (n : ℕ) : ℝ :=
  a^n

noncomputable def Sn (a : ℝ) (n : ℕ) (an : ℝ) : ℝ :=
  (a / (a - 1)) * (an - 1)

noncomputable def bn (a : ℝ) (n : ℕ) : ℝ :=
  2 * (Sn a n (given_sequence a n)) / (given_sequence a n) + 1

noncomputable def cn (a : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (bn a n)

noncomputable def Tn (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc k => acc + cn a (k + 1)) 0

theorem problem_statement_eq :
  ∀ (a : ℝ) (n : ℕ), a ≠ 0 → a ≠ 1 →
  (bn a n = (3:ℝ)^n) →
  Tn (1 / 3) n = 3^(n+1) * (2 * n - 3) / 4 + 9 / 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_problem_statement_eq_l887_88786


namespace NUMINAMATH_GPT_find_k_l887_88789

noncomputable def vector_a : ℝ × ℝ := (-1, 1)
noncomputable def vector_b : ℝ × ℝ := (2, 3)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (-2, k)

def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) (h : perp (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) (vector_c k)) : k = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l887_88789


namespace NUMINAMATH_GPT_parallel_lines_slope_l887_88784

theorem parallel_lines_slope {a : ℝ} (h : -a / 3 = -2 / 3) : a = 2 := 
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l887_88784


namespace NUMINAMATH_GPT_positive_integers_sum_digits_less_than_9000_l887_88737

theorem positive_integers_sum_digits_less_than_9000 : 
  ∃ n : ℕ, n = 47 ∧ ∀ x : ℕ, (1 ≤ x ∧ x < 9000 ∧ (Nat.digits 10 x).sum = 5) → (Nat.digits 10 x).length = n :=
sorry

end NUMINAMATH_GPT_positive_integers_sum_digits_less_than_9000_l887_88737


namespace NUMINAMATH_GPT_two_a_minus_b_l887_88718

-- Definitions of vector components and parallelism condition
def is_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0
def vector_a : ℝ × ℝ := (1, -2)

-- Given assumptions
variable (m : ℝ)
def vector_b : ℝ × ℝ := (m, 4)

-- Theorem statement
theorem two_a_minus_b (h : is_parallel vector_a (vector_b m)) : 2 • vector_a - vector_b m = (4, -8) :=
sorry

end NUMINAMATH_GPT_two_a_minus_b_l887_88718


namespace NUMINAMATH_GPT_abs_gt_x_iff_x_lt_0_l887_88755

theorem abs_gt_x_iff_x_lt_0 (x : ℝ) : |x| > x ↔ x < 0 := 
by
  sorry

end NUMINAMATH_GPT_abs_gt_x_iff_x_lt_0_l887_88755


namespace NUMINAMATH_GPT_alice_bob_meet_l887_88733

theorem alice_bob_meet (t : ℝ) 
(h1 : ∀ s : ℝ, s = 30 * t) 
(h2 : ∀ b : ℝ, b = 29.5 * 60 ∨ b = 30.5 * 60)
(h3 : ∀ a : ℝ, a = 30 * t)
(h4 : ∀ a b : ℝ, a = b):
(t = 59 ∨ t = 61) :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meet_l887_88733


namespace NUMINAMATH_GPT_minimum_value_f_inequality_proof_l887_88787

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 1)

-- The minimal value of f(x)
def m : ℝ := 4

theorem minimum_value_f :
  (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, -3 ≤ x ∧ x ≤ 1 ∧ f x = m) :=
by
  sorry -- Proof that the minimum value of f(x) is 4 and occurs in the range -3 ≤ x ≤ 1

variables (p q r : ℝ)

-- Given condition that p^2 + 2q^2 + r^2 = 4
theorem inequality_proof (h : p^2 + 2 * q^2 + r^2 = m) : q * (p + r) ≤ 2 :=
by
  sorry -- Proof that q(p + r) ≤ 2 given p^2 + 2q^2 + r^2 = 4

end NUMINAMATH_GPT_minimum_value_f_inequality_proof_l887_88787


namespace NUMINAMATH_GPT_arithmetic_expression_value_l887_88711

theorem arithmetic_expression_value :
  68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_value_l887_88711


namespace NUMINAMATH_GPT_complement_intersection_l887_88742

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection (hU : ∀ x, x ∈ U) (hA : ∀ x, x ∈ A) (hB : ∀ x, x ∈ B) :
    (U \ A) ∩ (U \ B) = {7, 9} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l887_88742


namespace NUMINAMATH_GPT_classify_curve_l887_88731

-- Define the curve equation
def curve_equation (m : ℝ) : Prop := 
  ∃ (x y : ℝ), ((m - 3) * x^2 + (5 - m) * y^2 = 1)

-- Define the conditions for types of curves
def is_circle (m : ℝ) : Prop := 
  m = 4 ∧ (curve_equation m)

def is_ellipse (m : ℝ) : Prop := 
  (3 < m ∧ m < 5 ∧ m ≠ 4) ∧ (curve_equation m)

def is_hyperbola (m : ℝ) : Prop := 
  ((m > 5 ∨ m < 3) ∧ (curve_equation m))

-- Main theorem stating the type of curve
theorem classify_curve (m : ℝ) : 
  (is_circle m) ∨ (is_ellipse m) ∨ (is_hyperbola m) :=
sorry

end NUMINAMATH_GPT_classify_curve_l887_88731


namespace NUMINAMATH_GPT_min_u_condition_l887_88792

-- Define the function u and the condition
def u (x y : ℝ) : ℝ := x^2 + 4 * x + y^2 - 2 * y

def condition (x y : ℝ) : Prop := 2 * x + y ≥ 1

-- The statement we want to prove
theorem min_u_condition : ∃ (x y : ℝ), condition x y ∧ u x y = -9/5 := 
by
  sorry

end NUMINAMATH_GPT_min_u_condition_l887_88792


namespace NUMINAMATH_GPT_tangent_line_at_point_l887_88708

def tangent_line_equation (f : ℝ → ℝ) (slope : ℝ) (p : ℝ × ℝ) :=
  ∃ (a b c : ℝ), a * p.1 + b * p.2 + c = 0 ∧ a = slope ∧ p.2 = f p.1

noncomputable def curve (x : ℝ) : ℝ := x^3 + x + 1

theorem tangent_line_at_point : 
  tangent_line_equation curve 4 (1, 3) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l887_88708


namespace NUMINAMATH_GPT_probability_of_picking_peach_l887_88797

-- Define the counts of each type of fruit
def apples : ℕ := 5
def pears : ℕ := 3
def peaches : ℕ := 2

-- Define the total number of fruits
def total_fruits : ℕ := apples + pears + peaches

-- Define the probability of picking a peach
def probability_of_peach : ℚ := peaches / total_fruits

-- State the theorem
theorem probability_of_picking_peach : probability_of_peach = 1/5 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_picking_peach_l887_88797


namespace NUMINAMATH_GPT_roots_opposite_signs_l887_88732

theorem roots_opposite_signs (a b c: ℝ) 
  (h1 : (b^2 - a * c) > 0)
  (h2 : (b^4 - a^2 * c^2) < 0) :
  a * c < 0 :=
sorry

end NUMINAMATH_GPT_roots_opposite_signs_l887_88732
