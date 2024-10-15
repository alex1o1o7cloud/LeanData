import Mathlib

namespace NUMINAMATH_GPT_elem_of_M_l2097_209776

variable (U M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : U \ M = {1, 3})

theorem elem_of_M : 2 ∈ M :=
by {
  sorry
}

end NUMINAMATH_GPT_elem_of_M_l2097_209776


namespace NUMINAMATH_GPT_polygon_perimeter_l2097_209749

theorem polygon_perimeter (a b : ℕ) (h : adjacent_sides_perpendicular) :
  perimeter = 2 * (a + b) :=
sorry

end NUMINAMATH_GPT_polygon_perimeter_l2097_209749


namespace NUMINAMATH_GPT_tom_swim_time_l2097_209789

theorem tom_swim_time (t : ℝ) :
  (2 * t + 4 * t = 12) → t = 2 :=
by
  intro h
  have eq1 : 6 * t = 12 := by linarith
  linarith

end NUMINAMATH_GPT_tom_swim_time_l2097_209789


namespace NUMINAMATH_GPT_probability_zero_after_2017_days_l2097_209784

-- Define the people involved
inductive Person
| Lunasa | Merlin | Lyrica
deriving DecidableEq, Inhabited

open Person

-- Define the initial state with each person having their own distinct hat
def initial_state : Person → Person
| Lunasa => Lunasa
| Merlin => Merlin
| Lyrica => Lyrica

-- Define a function that represents switching hats between two people
def switch_hats (p1 p2 : Person) (state : Person → Person) : Person → Person :=
  λ p => if p = p1 then state p2 else if p = p2 then state p1 else state p

-- Define a function to represent the state after n days (iterations)
def iter_switch_hats (n : ℕ) : Person → Person :=
  sorry -- This would involve implementing the iterative random switching

-- Proposition: The probability that after 2017 days, every person has their own hat back is 0
theorem probability_zero_after_2017_days :
  iter_switch_hats 2017 = initial_state → false :=
by
  sorry

end NUMINAMATH_GPT_probability_zero_after_2017_days_l2097_209784


namespace NUMINAMATH_GPT_students_side_by_side_with_A_and_B_l2097_209748

theorem students_side_by_side_with_A_and_B (total students_from_club_A students_from_club_B: ℕ) 
    (h1 : total = 100)
    (h2 : students_from_club_A = 62)
    (h3 : students_from_club_B = 54) :
  ∃ p q r : ℕ, p + q + r = 100 ∧ p + q = 62 ∧ p + r = 54 ∧ p = 16 :=
by
  sorry

end NUMINAMATH_GPT_students_side_by_side_with_A_and_B_l2097_209748


namespace NUMINAMATH_GPT_total_money_shared_l2097_209738

/-- Assume there are four people Amanda, Ben, Carlos, and David, sharing an amount of money.
    Their portions are in the ratio 1:2:7:3.
    Amanda's portion is $20.
    Prove that the total amount of money shared by them is $260. -/
theorem total_money_shared (A B C D : ℕ) (h_ratio : A = 20 ∧ B = 2 * A ∧ C = 7 * A ∧ D = 3 * A) :
  A + B + C + D = 260 := by 
  sorry

end NUMINAMATH_GPT_total_money_shared_l2097_209738


namespace NUMINAMATH_GPT_remainder_when_divided_by_32_l2097_209713

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_32_l2097_209713


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l2097_209786

theorem largest_angle_in_triangle (k : ℕ) (h : 3 * k + 4 * k + 5 * k = 180) : 5 * k = 75 :=
  by
  -- This is a placeholder for the proof, which is not required as per instructions
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l2097_209786


namespace NUMINAMATH_GPT_compound_interest_principal_l2097_209731

theorem compound_interest_principal 
  (CI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hCI : CI = 315)
  (hR : R = 10)
  (hT : T = 2) :
  CI = P * ((1 + R / 100)^T - 1) → P = 1500 := by
  sorry

end NUMINAMATH_GPT_compound_interest_principal_l2097_209731


namespace NUMINAMATH_GPT_amount_of_pizza_needed_l2097_209754

theorem amount_of_pizza_needed :
  (1 / 2 + 1 / 3 + 1 / 6) = 1 := by
  sorry

end NUMINAMATH_GPT_amount_of_pizza_needed_l2097_209754


namespace NUMINAMATH_GPT_constants_solution_l2097_209750

theorem constants_solution : ∀ (x : ℝ), x ≠ 0 ∧ x^2 ≠ 2 →
  (2 * x^2 - 5 * x + 1) / (x^3 - 2 * x) = (-1 / 2) / x + (2.5 * x - 5) / (x^2 - 2) := by
  intros x hx
  sorry

end NUMINAMATH_GPT_constants_solution_l2097_209750


namespace NUMINAMATH_GPT_eq_holds_for_n_l2097_209785

theorem eq_holds_for_n (n : ℕ) (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a + b + c + d = n * Real.sqrt (a * b * c * d) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_GPT_eq_holds_for_n_l2097_209785


namespace NUMINAMATH_GPT_incorrect_statement_A_l2097_209795

/-- Let prob_beijing be the probability of rainfall in Beijing and prob_shanghai be the probability
of rainfall in Shanghai. We assert that statement (A) which claims "It is certain to rain in Beijing today, 
while it is certain not to rain in Shanghai" is incorrect given the probabilities. 
-/
theorem incorrect_statement_A (prob_beijing prob_shanghai : ℝ) 
  (h_beijing : prob_beijing = 0.8)
  (h_shanghai : prob_shanghai = 0.2)
  (statement_A : ¬ (prob_beijing = 1 ∧ prob_shanghai = 0)) : 
  true := 
sorry

end NUMINAMATH_GPT_incorrect_statement_A_l2097_209795


namespace NUMINAMATH_GPT_scientific_notation_of_192M_l2097_209716

theorem scientific_notation_of_192M : 192000000 = 1.92 * 10^8 :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_192M_l2097_209716


namespace NUMINAMATH_GPT_vector_properties_l2097_209728

-- Definitions of vectors
def vec_a : ℝ × ℝ := (3, 11)
def vec_b : ℝ × ℝ := (-1, -4)
def vec_c : ℝ × ℝ := (1, 3)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Linear combination of vector scaling and addition
def vec_sub_scal (u v : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (u.1 - k * v.1, u.2 - k * v.2)

-- Check if two vectors are parallel
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2

-- Lean statement for the proof problem
theorem vector_properties :
  dot_product vec_a vec_b = -47 ∧
  vec_sub_scal vec_a vec_b 2 = (5, 19) ∧
  dot_product (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2) vec_c ≠ 0 ∧
  parallel (vec_sub_scal vec_a vec_c 1) vec_b :=
by sorry

end NUMINAMATH_GPT_vector_properties_l2097_209728


namespace NUMINAMATH_GPT_number_of_valid_triangles_l2097_209719

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_triangles_l2097_209719


namespace NUMINAMATH_GPT_simplify_complex_expr_correct_l2097_209768

noncomputable def simplify_complex_expr (i : ℂ) (h : i^2 = -1) : ℂ :=
  3 * (4 - 2 * i) - 2 * i * (3 - 2 * i) + (1 + i) * (2 + i)

theorem simplify_complex_expr_correct (i : ℂ) (h : i^2 = -1) : 
  simplify_complex_expr i h = 9 - 9 * i :=
by
  sorry

end NUMINAMATH_GPT_simplify_complex_expr_correct_l2097_209768


namespace NUMINAMATH_GPT_general_term_a_l2097_209783

noncomputable def S (n : ℕ) : ℤ := 3^n - 2

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2 * 3^(n - 1)

theorem general_term_a (n : ℕ) (hn : n > 0) : a n = if n = 1 then 1 else 2 * 3^(n - 1) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_general_term_a_l2097_209783


namespace NUMINAMATH_GPT_find_k_l2097_209762

theorem find_k (k α β : ℝ)
  (h1 : (∀ x : ℝ, x^2 - (k-1) * x - 3*k - 2 = 0 → x = α ∨ x = β))
  (h2 : α^2 + β^2 = 17) :
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l2097_209762


namespace NUMINAMATH_GPT_probability_red_balls_fourth_draw_l2097_209771

theorem probability_red_balls_fourth_draw :
  let p_red := 2 / 10
  let p_white := 8 / 10
  p_red * p_red * p_white * p_white * 3 / 10 + 
  p_red * p_white * p_red * p_white * 2 / 10 + 
  p_white * p_red * p_red * p_red = 0.0434 :=
sorry

end NUMINAMATH_GPT_probability_red_balls_fourth_draw_l2097_209771


namespace NUMINAMATH_GPT_find_x_l2097_209707

-- Definition of the problem conditions
def angle_ABC : ℝ := 85
def angle_BAC : ℝ := 55
def sum_angles_triangle (a b c : ℝ) : Prop := a + b + c = 180
def corresponding_angle (a b : ℝ) : Prop := a = b
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90

-- The theorem to prove
theorem find_x :
  ∀ (x BCA : ℝ), sum_angles_triangle angle_ABC angle_BAC BCA ∧ corresponding_angle BCA 40 ∧ right_triangle_sum BCA x → x = 50 :=
by
  intros x BCA h
  sorry

end NUMINAMATH_GPT_find_x_l2097_209707


namespace NUMINAMATH_GPT_value_of_ratio_l2097_209778

theorem value_of_ratio (x y : ℝ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 2 * x + 3 * y = 8) :
    (2 / x + 3 / y) = 25 / 8 := 
by
  sorry

end NUMINAMATH_GPT_value_of_ratio_l2097_209778


namespace NUMINAMATH_GPT_f_at_2_l2097_209773

noncomputable def f (x : ℝ) (a b : ℝ) := a * Real.log x + b / x + x
noncomputable def g (x : ℝ) (a b : ℝ) := (a / x) - (b / (x ^ 2)) + 1

theorem f_at_2 (a b : ℝ) (ha : g 1 a b = 0) (hb : g 3 a b = 0) : f 2 a b = 1 / 2 - 4 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_f_at_2_l2097_209773


namespace NUMINAMATH_GPT_expand_expression_l2097_209725

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2097_209725


namespace NUMINAMATH_GPT_determine_disco_ball_price_l2097_209735

variable (x y z : ℝ)

-- Given conditions
def budget_constraint : Prop := 4 * x + 10 * y + 20 * z = 600
def food_cost : Prop := y = 0.85 * x
def decoration_cost : Prop := z = x / 2 - 10

-- Goal
theorem determine_disco_ball_price (h1 : budget_constraint x y z) (h2 : food_cost x y) (h3 : decoration_cost x z) :
  x = 35.56 :=
sorry 

end NUMINAMATH_GPT_determine_disco_ball_price_l2097_209735


namespace NUMINAMATH_GPT_percent_diamond_jewels_l2097_209787

def percent_beads : ℝ := 0.3
def percent_ruby_jewels : ℝ := 0.5

theorem percent_diamond_jewels (percent_beads percent_ruby_jewels : ℝ) : 
  (1 - percent_beads) * (1 - percent_ruby_jewels) = 0.35 :=
by
  -- We insert the proof steps here
  sorry

end NUMINAMATH_GPT_percent_diamond_jewels_l2097_209787


namespace NUMINAMATH_GPT_factorization_problem1_factorization_problem2_l2097_209718

-- Define the first problem: Factorization of 3x^2 - 27
theorem factorization_problem1 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) :=
by
  sorry 

-- Define the second problem: Factorization of (a + 1)(a - 5) + 9
theorem factorization_problem2 (a : ℝ) : (a + 1) * (a - 5) + 9 = (a - 2) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_problem1_factorization_problem2_l2097_209718


namespace NUMINAMATH_GPT_evaluate_y_correct_l2097_209744

noncomputable def evaluate_y (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - 4 * x + 4) + Real.sqrt (x^2 + 6 * x + 9) - 2

theorem evaluate_y_correct (x : ℝ) : 
  evaluate_y x = |x - 2| + |x + 3| - 2 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_y_correct_l2097_209744


namespace NUMINAMATH_GPT_total_annual_cost_l2097_209740

def daily_pills : ℕ := 2
def pill_cost : ℕ := 5
def medication_cost (daily_pills : ℕ) (pill_cost : ℕ) : ℕ := daily_pills * pill_cost
def insurance_coverage : ℚ := 0.80
def visit_cost : ℕ := 400
def visits_per_year : ℕ := 2
def annual_medication_cost (medication_cost : ℕ) (insurance_coverage : ℚ) : ℚ :=
  medication_cost * 365 * (1 - insurance_coverage)
def annual_visit_cost (visit_cost : ℕ) (visits_per_year : ℕ) : ℕ :=
  visit_cost * visits_per_year

theorem total_annual_cost : annual_medication_cost (medication_cost daily_pills pill_cost) insurance_coverage
  + annual_visit_cost visit_cost visits_per_year = 1530 := by
  sorry

end NUMINAMATH_GPT_total_annual_cost_l2097_209740


namespace NUMINAMATH_GPT_max_value_of_a_l2097_209737

theorem max_value_of_a (a b c d : ℤ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 := by 
  sorry

end NUMINAMATH_GPT_max_value_of_a_l2097_209737


namespace NUMINAMATH_GPT_find_original_number_l2097_209788

theorem find_original_number (x y : ℕ) (h1 : x + y = 8) (h2 : 10 * y + x = 10 * x + y + 18) : 10 * x + y = 35 := 
sorry

end NUMINAMATH_GPT_find_original_number_l2097_209788


namespace NUMINAMATH_GPT_additional_machines_needed_l2097_209701

theorem additional_machines_needed
  (machines : ℕ)
  (days : ℕ)
  (one_fourth_less_days : ℕ)
  (machine_days_total : ℕ)
  (machines_needed : ℕ)
  (additional_machines : ℕ) 
  (h1 : machines = 15) 
  (h2 : days = 36)
  (h3 : one_fourth_less_days = 27)
  (h4 : machine_days_total = machines * days)
  (h5 : machines_needed = machine_days_total / one_fourth_less_days) :
  additional_machines = machines_needed - machines → additional_machines = 5 :=
by
  admit -- sorry

end NUMINAMATH_GPT_additional_machines_needed_l2097_209701


namespace NUMINAMATH_GPT_geometric_progression_fourth_term_l2097_209700

theorem geometric_progression_fourth_term (a b c : ℝ) (r : ℝ) 
  (h1 : a = 2) (h2 : b = 2 * Real.sqrt 2) (h3 : c = 4) (h4 : r = Real.sqrt 2)
  (h5 : b = a * r) (h6 : c = b * r) :
  c * r = 4 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_geometric_progression_fourth_term_l2097_209700


namespace NUMINAMATH_GPT_unique_function_satisfies_sum_zero_l2097_209730

theorem unique_function_satisfies_sum_zero 
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x^3) = (f x)^3)
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2) : 
  f 0 + f 1 + f (-1) = 0 :=
sorry

end NUMINAMATH_GPT_unique_function_satisfies_sum_zero_l2097_209730


namespace NUMINAMATH_GPT_number_of_grade2_students_l2097_209745

theorem number_of_grade2_students (ratio1 ratio2 ratio3 : ℕ) (total_students : ℕ) (ratio_sum : ratio1 + ratio2 + ratio3 = 12)
  (total_sample_size : total_students = 240) : 
  total_students * ratio2 / (ratio1 + ratio2 + ratio3) = 80 :=
by
  have ratio1_val : ratio1 = 5 := sorry
  have ratio2_val : ratio2 = 4 := sorry
  have ratio3_val : ratio3 = 3 := sorry
  rw [ratio1_val, ratio2_val, ratio3_val] at ratio_sum
  rw [ratio1_val, ratio2_val, ratio3_val]
  exact sorry

end NUMINAMATH_GPT_number_of_grade2_students_l2097_209745


namespace NUMINAMATH_GPT_nap_time_is_correct_l2097_209706

-- Define the total trip time and the hours spent on each activity
def total_trip_time : ℝ := 15
def reading_time : ℝ := 2
def eating_time : ℝ := 1
def movies_time : ℝ := 3
def chatting_time : ℝ := 1
def browsing_time : ℝ := 0.75
def waiting_time : ℝ := 0.5
def working_time : ℝ := 2

-- Define the total activity time
def total_activity_time : ℝ := reading_time + eating_time + movies_time + chatting_time + browsing_time + waiting_time + working_time

-- Define the nap time as the difference between total trip time and total activity time
def nap_time : ℝ := total_trip_time - total_activity_time

-- Prove that the nap time is 4.75 hours
theorem nap_time_is_correct : nap_time = 4.75 :=
by
  -- Calculation hint, can be ignored
  -- nap_time = 15 - (2 + 1 + 3 + 1 + 0.75 + 0.5 + 2) = 15 - 10.25 = 4.75
  sorry

end NUMINAMATH_GPT_nap_time_is_correct_l2097_209706


namespace NUMINAMATH_GPT_total_license_groups_l2097_209753

-- Defining the given conditions
def letter_choices : Nat := 3
def digit_choices_per_slot : Nat := 10
def number_of_digit_slots : Nat := 5

-- Statement to prove that the total number of different license groups is 300000
theorem total_license_groups : letter_choices * (digit_choices_per_slot ^ number_of_digit_slots) = 300000 := by
  sorry

end NUMINAMATH_GPT_total_license_groups_l2097_209753


namespace NUMINAMATH_GPT_kabadi_kho_kho_players_l2097_209760

theorem kabadi_kho_kho_players (total_players kabadi_only kho_kho_only both_games : ℕ)
  (h1 : kabadi_only = 10)
  (h2 : kho_kho_only = 40)
  (h3 : total_players = 50)
  (h4 : kabadi_only + kho_kho_only - both_games = total_players) :
  both_games = 0 := by
  sorry

end NUMINAMATH_GPT_kabadi_kho_kho_players_l2097_209760


namespace NUMINAMATH_GPT_sum_of_reciprocals_factors_12_l2097_209734

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end NUMINAMATH_GPT_sum_of_reciprocals_factors_12_l2097_209734


namespace NUMINAMATH_GPT_employee_monthly_wage_l2097_209782

theorem employee_monthly_wage 
(revenue : ℝ)
(tax_rate : ℝ)
(marketing_rate : ℝ)
(operational_cost_rate : ℝ)
(wage_rate : ℝ)
(num_employees : ℕ)
(h_revenue : revenue = 400000)
(h_tax_rate : tax_rate = 0.10)
(h_marketing_rate : marketing_rate = 0.05)
(h_operational_cost_rate : operational_cost_rate = 0.20)
(h_wage_rate : wage_rate = 0.15)
(h_num_employees : num_employees = 10) :
(revenue * (1 - tax_rate) * (1 - marketing_rate) * (1 - operational_cost_rate) * wage_rate / num_employees = 4104) :=
by
  sorry

end NUMINAMATH_GPT_employee_monthly_wage_l2097_209782


namespace NUMINAMATH_GPT_opposite_of_minus_one_third_l2097_209722

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_minus_one_third_l2097_209722


namespace NUMINAMATH_GPT_radius_of_garden_outer_boundary_l2097_209764

-- Definitions based on the conditions from the problem statement
def fountain_diameter : ℝ := 12
def garden_width : ℝ := 10

-- Question translated to a proof statement
theorem radius_of_garden_outer_boundary :
  (fountain_diameter / 2 + garden_width) = 16 := 
by 
  sorry

end NUMINAMATH_GPT_radius_of_garden_outer_boundary_l2097_209764


namespace NUMINAMATH_GPT_avg_price_pen_is_correct_l2097_209752

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end NUMINAMATH_GPT_avg_price_pen_is_correct_l2097_209752


namespace NUMINAMATH_GPT_jeffrey_walks_to_mailbox_l2097_209772

theorem jeffrey_walks_to_mailbox :
  ∀ (D total_steps net_gain_per_set steps_per_set sets net_gain : ℕ),
    steps_per_set = 3 ∧ 
    net_gain = 1 ∧ 
    total_steps = 330 ∧ 
    net_gain_per_set = net_gain ∧ 
    sets = total_steps / steps_per_set ∧ 
    D = sets * net_gain →
    D = 110 :=
by
  intro D total_steps net_gain_per_set steps_per_set sets net_gain
  intro h
  sorry

end NUMINAMATH_GPT_jeffrey_walks_to_mailbox_l2097_209772


namespace NUMINAMATH_GPT_initial_population_l2097_209742

theorem initial_population (P : ℝ) 
    (h1 : 1.25 * P * 0.70 = 363650) : 
    P = 415600 :=
sorry

end NUMINAMATH_GPT_initial_population_l2097_209742


namespace NUMINAMATH_GPT_percentage_increase_in_savings_l2097_209757

theorem percentage_increase_in_savings
  (I : ℝ) -- Original income of Paulson
  (E : ℝ) -- Original expenditure of Paulson
  (hE : E = 0.75 * I) -- Paulson spends 75% of his income
  (h_inc_income : 1.2 * I = I + 0.2 * I) -- Income is increased by 20%
  (h_inc_expenditure : 0.825 * I = 0.75 * I + 0.1 * (0.75 * I)) -- Expenditure is increased by 10%
  : (0.375 * I - 0.25 * I) / (0.25 * I) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_savings_l2097_209757


namespace NUMINAMATH_GPT_lost_weights_l2097_209755

-- Define the weights
def weights : List ℕ := [43, 70, 57]

-- Total remaining weight after loss
def remaining_weight : ℕ := 20172

-- Number of weights lost
def weights_lost : ℕ := 4

-- Whether a given number of weights and types of weights match the remaining weight
def valid_loss (initial_count : ℕ) (lost_weight_count : ℕ) : Prop :=
  let total_initial_weight := initial_count * (weights.sum)
  let lost_weight := lost_weight_count * 57
  total_initial_weight - lost_weight = remaining_weight

-- Proposition we need to prove
theorem lost_weights (initial_count : ℕ) (h : valid_loss initial_count weights_lost) : ∀ w ∈ weights, w = 57 :=
by {
  sorry
}

end NUMINAMATH_GPT_lost_weights_l2097_209755


namespace NUMINAMATH_GPT_problem_a_b_l2097_209799

theorem problem_a_b (a b : ℝ) (h₁ : a + b = 10) (h₂ : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_b_l2097_209799


namespace NUMINAMATH_GPT_problem_1_problem_2_l2097_209710

theorem problem_1 {m : ℝ} (h₁ : 0 < m) (h₂ : ∀ x : ℝ, (m - |x + 2| ≥ 0) ↔ (-3 ≤ x ∧ x ≤ -1)) :
  m = 1 :=
sorry

theorem problem_2 {a b c : ℝ} (h₃ : 0 < a ∧ 0 < b ∧ 0 < c) (h₄ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1)
  : a + 2 * b + 3 * c ≥ 9 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2097_209710


namespace NUMINAMATH_GPT_probability_odd_even_draw_correct_l2097_209708

noncomputable def probability_odd_even_draw : ℚ := sorry

theorem probability_odd_even_draw_correct :
  probability_odd_even_draw = 17 / 45 := 
sorry

end NUMINAMATH_GPT_probability_odd_even_draw_correct_l2097_209708


namespace NUMINAMATH_GPT_pqr_value_l2097_209729

theorem pqr_value
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 29)
  (h_eq : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 :=
by
  sorry

end NUMINAMATH_GPT_pqr_value_l2097_209729


namespace NUMINAMATH_GPT_point_A_coordinates_l2097_209721

variable (a x y : ℝ)

def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

theorem point_A_coordinates (h1 : ∃ t : ℝ, ∀ x : ℝ, f a x = t * x + t) (h2 : x = 0) : (0, 2) = (0, f a 0) :=
by
  sorry

end NUMINAMATH_GPT_point_A_coordinates_l2097_209721


namespace NUMINAMATH_GPT_city_phone_number_remainder_l2097_209792

theorem city_phone_number_remainder :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := sorry

end NUMINAMATH_GPT_city_phone_number_remainder_l2097_209792


namespace NUMINAMATH_GPT_arc_length_l2097_209766

/-- Given a circle with a radius of 5 cm and a sector area of 11.25 cm², 
prove that the length of the arc forming the sector is 4.5 cm. --/
theorem arc_length (r : ℝ) (A : ℝ) (θ : ℝ) (arc_length : ℝ) 
  (h_r : r = 5) 
  (h_A : A = 11.25) 
  (h_area_formula : A = (θ / (2 * π)) * π * r ^ 2) 
  (h_arc_length_formula : arc_length = r * θ) :
  arc_length = 4.5 :=
sorry

end NUMINAMATH_GPT_arc_length_l2097_209766


namespace NUMINAMATH_GPT_volume_solid_correct_l2097_209702

noncomputable def volume_of_solid : ℝ := 
  let area_rhombus := 1250 -- Area of the rhombus calculated from the bounded region
  let height := 10 -- Given height of the solid
  area_rhombus * height -- Volume of the solid

theorem volume_solid_correct (height: ℝ := 10) :
  volume_of_solid = 12500 := by
  sorry

end NUMINAMATH_GPT_volume_solid_correct_l2097_209702


namespace NUMINAMATH_GPT_sequence_sum_l2097_209717

-- Assume the sum of first n terms of the sequence {a_n} is given by S_n = n^2 + n + 1
def S (n : ℕ) : ℕ := n^2 + n + 1

-- The sequence a_8 + a_9 + a_10 + a_11 + a_12 is what we want to prove equals 100.
theorem sequence_sum : S 12 - S 7 = 100 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l2097_209717


namespace NUMINAMATH_GPT_xiaoming_wait_probability_l2097_209743

-- Conditions
def green_light_duration : ℕ := 40
def red_light_duration : ℕ := 50
def total_light_cycle : ℕ := green_light_duration + red_light_duration
def waiting_time_threshold : ℕ := 20
def long_wait_interval : ℕ := 30 -- from problem (20 seconds to wait corresponds to 30 seconds interval)

-- Probability calculation
theorem xiaoming_wait_probability :
  ∀ (arrival_time : ℕ), arrival_time < total_light_cycle →
    (30 : ℝ) / (total_light_cycle : ℝ) = 1 / 3 := by sorry

end NUMINAMATH_GPT_xiaoming_wait_probability_l2097_209743


namespace NUMINAMATH_GPT_find_product_of_roots_plus_one_l2097_209779

-- Define the problem conditions
variables (x1 x2 : ℝ)
axiom sum_roots : x1 + x2 = 3
axiom prod_roots : x1 * x2 = 2

-- State the theorem corresponding to the proof problem
theorem find_product_of_roots_plus_one : (x1 + 1) * (x2 + 1) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_product_of_roots_plus_one_l2097_209779


namespace NUMINAMATH_GPT_find_two_digit_integers_l2097_209794

theorem find_two_digit_integers :
  ∃ (m n : ℕ), 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 ∧
    (∃ (a b : ℚ), a = m ∧ b = n ∧ (a + b) / 2 = b + a / 100) ∧ (m + n < 150) ∧ m = 50 ∧ n = 49 := 
by
  sorry

end NUMINAMATH_GPT_find_two_digit_integers_l2097_209794


namespace NUMINAMATH_GPT_length_of_PR_l2097_209741

-- Define the entities and conditions
variables (x y : ℝ)
variables (xy_area : ℝ := 125)
variables (PR_length : ℝ := 10 * Real.sqrt 5)

-- State the problem in Lean
theorem length_of_PR (x y : ℝ) (hxy : x * y = 125) :
  x^2 + (125 / x)^2 = (10 * Real.sqrt 5)^2 :=
sorry

end NUMINAMATH_GPT_length_of_PR_l2097_209741


namespace NUMINAMATH_GPT_miles_driven_on_Monday_l2097_209751

def miles_Tuesday : ℕ := 18
def miles_Wednesday : ℕ := 21
def avg_miles_per_day : ℕ := 17

theorem miles_driven_on_Monday (miles_Monday : ℕ) :
  (miles_Monday + miles_Tuesday + miles_Wednesday) / 3 = avg_miles_per_day →
  miles_Monday = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_miles_driven_on_Monday_l2097_209751


namespace NUMINAMATH_GPT_empty_rooms_le_1000_l2097_209758

/--
In a 50x50 grid where each cell can contain at most one tree, 
with the following rules: 
1. A pomegranate tree has at least one apple neighbor
2. A peach tree has at least one apple neighbor and one pomegranate neighbor
3. An empty room has at least one apple neighbor, one pomegranate neighbor, and one peach neighbor
Show that the number of empty rooms is not greater than 1000.
-/
theorem empty_rooms_le_1000 (apple pomegranate peach : ℕ) (empty : ℕ)
  (h1 : apple + pomegranate + peach + empty = 2500)
  (h2 : ∀ p, pomegranate ≥ p → apple ≥ 1)
  (h3 : ∀ p, peach ≥ p → apple ≥ 1 ∧ pomegranate ≥ 1)
  (h4 : ∀ e, empty ≥ e → apple ≥ 1 ∧ pomegranate ≥ 1 ∧ peach ≥ 1) :
  empty ≤ 1000 :=
sorry

end NUMINAMATH_GPT_empty_rooms_le_1000_l2097_209758


namespace NUMINAMATH_GPT_middle_number_is_five_l2097_209723

theorem middle_number_is_five
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 20)
  (h_sorted : a < b ∧ b < c)
  (h_bella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → x = a → y = b ∧ z = c)
  (h_della : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → y = b → x = a ∧ z = c)
  (h_nella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → z = c → x = a ∧ y = b) :
  b = 5 := sorry

end NUMINAMATH_GPT_middle_number_is_five_l2097_209723


namespace NUMINAMATH_GPT_total_percentage_increase_l2097_209703

def initial_salary : Float := 60
def first_raise (s : Float) : Float := s + 0.10 * s
def second_raise (s : Float) : Float := s + 0.15 * s
def deduction (s : Float) : Float := s - 0.05 * s
def promotion_raise (s : Float) : Float := s + 0.20 * s
def final_salary (s : Float) : Float := promotion_raise (deduction (second_raise (first_raise s)))

theorem total_percentage_increase :
  final_salary initial_salary = initial_salary * 1.4421 :=
by
  sorry

end NUMINAMATH_GPT_total_percentage_increase_l2097_209703


namespace NUMINAMATH_GPT_outfit_combinations_l2097_209736

def shirts : ℕ := 6
def pants : ℕ := 4
def hats : ℕ := 6

def pant_colors : Finset String := {"tan", "black", "blue", "gray"}
def shirt_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}

def total_combinations : ℕ := shirts * pants * hats
def restricted_combinations : ℕ := pant_colors.card

theorem outfit_combinations
    (hshirts : shirts = 6)
    (hpants : pants = 4)
    (hhats : hats = 6)
    (hpant_colors : pant_colors.card = 4)
    (hshirt_colors : shirt_colors.card = 6)
    (hhat_colors : hat_colors.card = 6)
    (hrestricted : restricted_combinations = pant_colors.card) :
    total_combinations - restricted_combinations = 140 := by
  sorry

end NUMINAMATH_GPT_outfit_combinations_l2097_209736


namespace NUMINAMATH_GPT_expected_value_min_of_subset_l2097_209765

noncomputable def expected_value_min (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : ℚ :=
  (n + 1) / (r + 1)

theorem expected_value_min_of_subset (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  expected_value_min n r h = (n + 1) / (r + 1) :=
sorry

end NUMINAMATH_GPT_expected_value_min_of_subset_l2097_209765


namespace NUMINAMATH_GPT_exists_fi_l2097_209796

theorem exists_fi (f : ℝ → ℝ) (h_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧ 
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧ 
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧ 
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧ 
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
by
  sorry

end NUMINAMATH_GPT_exists_fi_l2097_209796


namespace NUMINAMATH_GPT_first_group_men_8_l2097_209709

variable (x : ℕ)

theorem first_group_men_8 (h1 : x * 80 = 20 * 32) : x = 8 := by
  -- provide the proof here
  sorry

end NUMINAMATH_GPT_first_group_men_8_l2097_209709


namespace NUMINAMATH_GPT_smallest_angle_l2097_209780

theorem smallest_angle (largest_angle : ℝ) (a b : ℝ) (h1 : largest_angle = 120) (h2 : 3 * a = 2 * b) (h3 : largest_angle + a + b = 180) : b = 24 := by
  sorry

end NUMINAMATH_GPT_smallest_angle_l2097_209780


namespace NUMINAMATH_GPT_factorize_l2097_209739

theorem factorize (x : ℝ) : 72 * x ^ 11 + 162 * x ^ 22 = 18 * x ^ 11 * (4 + 9 * x ^ 11) :=
by
  sorry

end NUMINAMATH_GPT_factorize_l2097_209739


namespace NUMINAMATH_GPT_remainder_of_polynomial_l2097_209793

theorem remainder_of_polynomial (x : ℤ) : 
  (x^4 - 1) * (x^2 - 1) % (x^2 + x + 1) = 3 := 
sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l2097_209793


namespace NUMINAMATH_GPT_log_domain_inequality_l2097_209756

theorem log_domain_inequality {a : ℝ} : 
  (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ a > 1 :=
sorry

end NUMINAMATH_GPT_log_domain_inequality_l2097_209756


namespace NUMINAMATH_GPT_form_square_from_trapezoid_l2097_209763

noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

theorem form_square_from_trapezoid (a b h : ℝ) (trapezoid_area_eq_five : trapezoid_area a b h = 5) :
  ∃ s : ℝ, s^2 = 5 :=
by
  use (Real.sqrt 5)
  sorry

end NUMINAMATH_GPT_form_square_from_trapezoid_l2097_209763


namespace NUMINAMATH_GPT_carrie_spent_l2097_209732

-- Definitions derived from the problem conditions
def cost_of_one_tshirt : ℝ := 9.65
def number_of_tshirts : ℕ := 12

-- The statement to prove
theorem carrie_spent :
  cost_of_one_tshirt * number_of_tshirts = 115.80 :=
by
  sorry

end NUMINAMATH_GPT_carrie_spent_l2097_209732


namespace NUMINAMATH_GPT_dessert_eating_contest_l2097_209781

theorem dessert_eating_contest (a b c : ℚ) 
  (h1 : a = 5/6) 
  (h2 : b = 7/8) 
  (h3 : c = 1/2) :
  b - a = 1/24 ∧ a - c = 1/3 := 
by 
  sorry

end NUMINAMATH_GPT_dessert_eating_contest_l2097_209781


namespace NUMINAMATH_GPT_matvey_healthy_diet_l2097_209726

theorem matvey_healthy_diet (n b_1 p_1 : ℕ) (h1 : n * b_1 - (n * (n - 1)) / 2 = 264) (h2 : n * p_1 + (n * (n - 1)) / 2 = 187) :
  n = 11 :=
by
  let buns_diff_pears := b_1 - p_1 - (n - 1)
  have buns_def : 264 = n * buns_diff_pears + n * (n - 1) / 2 := sorry
  have pears_def : 187 = n * buns_diff_pears - n * (n - 1) / 2 := sorry
  have diff : 77 = n * buns_diff_pears := sorry
  sorry

end NUMINAMATH_GPT_matvey_healthy_diet_l2097_209726


namespace NUMINAMATH_GPT_average_salary_excluding_manager_l2097_209761

theorem average_salary_excluding_manager (A : ℝ) 
  (num_employees : ℝ := 20)
  (manager_salary : ℝ := 3300)
  (salary_increase : ℝ := 100)
  (total_salary_with_manager : ℝ := 21 * (A + salary_increase)) :
  20 * A + manager_salary = total_salary_with_manager → A = 1200 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_average_salary_excluding_manager_l2097_209761


namespace NUMINAMATH_GPT_probability_girl_selection_l2097_209733

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_girl_selection_l2097_209733


namespace NUMINAMATH_GPT_eval_expr_l2097_209705

theorem eval_expr : (900 ^ 2) / (262 ^ 2 - 258 ^ 2) = 389.4 := 
by
  sorry

end NUMINAMATH_GPT_eval_expr_l2097_209705


namespace NUMINAMATH_GPT_new_trailer_homes_added_l2097_209727

theorem new_trailer_homes_added
  (n : ℕ) (avg_age_3_years_ago avg_age_today age_increase new_home_age : ℕ) (k : ℕ) :
  n = 30 → avg_age_3_years_ago = 15 → avg_age_today = 12 → age_increase = 3 → new_home_age = 3 →
  (n * (avg_age_3_years_ago + age_increase) + k * new_home_age) / (n + k) = avg_age_today →
  k = 20 :=
by
  intros h_n h_avg_age_3y h_avg_age_today h_age_increase h_new_home_age h_eq
  sorry

end NUMINAMATH_GPT_new_trailer_homes_added_l2097_209727


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2097_209769

theorem arithmetic_sequence_sum :
  ∃ (a_n : ℕ → ℝ) (d : ℝ), 
  (∀ n, a_n n = a_n 0 + n * d) ∧
  d > 0 ∧
  a_n 0 + a_n 1 + a_n 2 = 15 ∧
  a_n 0 * a_n 1 * a_n 2 = 80 →
  a_n 10 + a_n 11 + a_n 12 = 135 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2097_209769


namespace NUMINAMATH_GPT_factory_ill_days_l2097_209715

theorem factory_ill_days
  (average_first_25_days : ℝ)
  (total_days : ℝ)
  (overall_average : ℝ)
  (ill_days_average : ℝ)
  (production_first_25_days_total : ℝ)
  (production_ill_days_total : ℝ)
  (x : ℝ) :
  average_first_25_days = 50 →
  total_days = 25 + x →
  overall_average = 48 →
  ill_days_average = 38 →
  production_first_25_days_total = 25 * 50 →
  production_ill_days_total = x * 38 →
  (25 * 50 + x * 38 = (25 + x) * 48) →
  x = 5 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_factory_ill_days_l2097_209715


namespace NUMINAMATH_GPT_remaining_amount_correct_l2097_209724

-- Definitions for the given conditions
def deposit_percentage : ℝ := 0.05
def deposit_amount : ℝ := 50

-- The correct answer we need to prove
def remaining_amount_to_be_paid : ℝ := 950

-- Stating the theorem (proof not required)
theorem remaining_amount_correct (total_price : ℝ) 
    (H1 : deposit_amount = total_price * deposit_percentage) : 
    total_price - deposit_amount = remaining_amount_to_be_paid :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_correct_l2097_209724


namespace NUMINAMATH_GPT_sum_of_integers_990_l2097_209770

theorem sum_of_integers_990 :
  ∃ (n m : ℕ), (n * (n + 1) = 990 ∧ (m - 1) * m * (m + 1) = 990 ∧ (n + n + 1 + m - 1 + m + m + 1 = 90)) :=
sorry

end NUMINAMATH_GPT_sum_of_integers_990_l2097_209770


namespace NUMINAMATH_GPT_decrypted_plaintext_l2097_209720

theorem decrypted_plaintext (a b c d : ℕ) : 
  (a + 2 * b = 14) → (2 * b + c = 9) → (2 * c + 3 * d = 23) → (4 * d = 28) → 
  (a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7) :=
by 
  intros h1 h2 h3 h4
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_decrypted_plaintext_l2097_209720


namespace NUMINAMATH_GPT_william_washed_2_normal_cars_l2097_209759

def time_spent_on_one_normal_car : Nat := 4 + 7 + 4 + 9

def time_spent_on_suv : Nat := 2 * time_spent_on_one_normal_car

def total_time_spent : Nat := 96

def time_spent_on_normal_cars : Nat := total_time_spent - time_spent_on_suv

def number_of_normal_cars : Nat := time_spent_on_normal_cars / time_spent_on_one_normal_car

theorem william_washed_2_normal_cars : number_of_normal_cars = 2 := by
  sorry

end NUMINAMATH_GPT_william_washed_2_normal_cars_l2097_209759


namespace NUMINAMATH_GPT_no_integer_solution_mx2_minus_sy2_eq_3_l2097_209704

theorem no_integer_solution_mx2_minus_sy2_eq_3 (m s : ℤ) (x y : ℤ) (h : m * s = 2000 ^ 2001) :
  ¬ (m * x ^ 2 - s * y ^ 2 = 3) :=
sorry

end NUMINAMATH_GPT_no_integer_solution_mx2_minus_sy2_eq_3_l2097_209704


namespace NUMINAMATH_GPT_trigonometric_identity_l2097_209711

theorem trigonometric_identity (x : ℝ) (h : Real.tan (x + Real.pi / 2) = 5) : 
  1 / (Real.sin x * Real.cos x) = -26 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2097_209711


namespace NUMINAMATH_GPT_sum_of_digits_in_binary_representation_of_315_l2097_209790

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_in_binary_representation_of_315_l2097_209790


namespace NUMINAMATH_GPT_vacation_fund_percentage_l2097_209798

variable (s : ℝ) (vs : ℝ)
variable (d : ℝ)
variable (v : ℝ)

-- conditions:
-- 1. Jill's net monthly salary
#check (s = 3700)
-- 2. Jill's discretionary income is one fifth of her salary
#check (d = s / 5)
-- 3. Savings percentage
#check (0.20 * d)
-- 4. Eating out and socializing percentage
#check (0.35 * d)
-- 5. Gifts and charitable causes
#check (111)

-- Prove: 
theorem vacation_fund_percentage : 
  s = 3700 -> d = s / 5 -> 
  (v * d + 0.20 * d + 0.35 * d + 111 = d) -> 
  v = 222 / 740 :=
by
  sorry -- proof skipped

end NUMINAMATH_GPT_vacation_fund_percentage_l2097_209798


namespace NUMINAMATH_GPT_find_a_cubed_l2097_209791

-- Definitions based on conditions
def varies_inversely (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement with given conditions
theorem find_a_cubed (a b : ℝ) (k : ℝ) (h1 : varies_inversely a b)
    (h2 : a = 2) (h3 : b = 4) (k_val : k = 2048) (b_new : b = 8) : a^3 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_cubed_l2097_209791


namespace NUMINAMATH_GPT_solve_system_l2097_209777

theorem solve_system (x y : ℝ) :
  (2 * y = (abs (2 * x + 3)) - (abs (2 * x - 3))) ∧ 
  (4 * x = (abs (y + 2)) - (abs (y - 2))) → 
  (-1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x) := 
by
  sorry

end NUMINAMATH_GPT_solve_system_l2097_209777


namespace NUMINAMATH_GPT_angle_ratio_l2097_209712

theorem angle_ratio (x y α β : ℝ)
  (h1 : y = x + β)
  (h2 : 2 * y = 2 * x + α) :
  α / β = 2 :=
by
  sorry

end NUMINAMATH_GPT_angle_ratio_l2097_209712


namespace NUMINAMATH_GPT_max_distance_without_fuel_depots_l2097_209767

def exploration_max_distance : ℕ :=
  360

-- Define the conditions
def cars_count : ℕ :=
  9

def full_tank_distance : ℕ :=
  40

def additional_gal_capacity : ℕ :=
  9

def total_gallons_per_car : ℕ :=
  1 + additional_gal_capacity

-- Define the distance calculation under the given constraints
theorem max_distance_without_fuel_depots (n : ℕ) (d_tank : ℕ) (d_add : ℕ) :
  ∀ (cars : ℕ), (cars = cars_count) →
  (d_tank = full_tank_distance) →
  (d_add = additional_gal_capacity) →
  ((cars * (1 + d_add)) * d_tank) / (2 * cars - 1) = exploration_max_distance :=
by
  intros _ hc ht ha
  rw [hc, ht, ha]
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_max_distance_without_fuel_depots_l2097_209767


namespace NUMINAMATH_GPT_local_maximum_at_1_2_l2097_209714

noncomputable def f (x1 x2 : ℝ) : ℝ := x2^2 - x1^2
def constraint (x1 x2 : ℝ) : Prop := x1 - 2 * x2 + 3 = 0
def is_local_maximum (f : ℝ → ℝ → ℝ) (x1 x2 : ℝ) : Prop := 
∃ ε > 0, ∀ (y1 y2 : ℝ), (constraint y1 y2 ∧ (y1 - x1)^2 + (y2 - x2)^2 < ε^2) → f y1 y2 ≤ f x1 x2

theorem local_maximum_at_1_2 : is_local_maximum f 1 2 :=
sorry

end NUMINAMATH_GPT_local_maximum_at_1_2_l2097_209714


namespace NUMINAMATH_GPT_weight_comparison_l2097_209775

theorem weight_comparison :
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  average = 45 ∧ median = 25 ∧ average - median = 20 :=
by
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  have h1 : average = 45 := sorry
  have h2 : median = 25 := sorry
  have h3 : average - median = 20 := sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_weight_comparison_l2097_209775


namespace NUMINAMATH_GPT_circle_condition_l2097_209746

theorem circle_condition (m : ℝ) :
    (4 * m) ^ 2 + 4 - 4 * 5 * m > 0 ↔ (m < 1 / 4 ∨ m > 1) := sorry

end NUMINAMATH_GPT_circle_condition_l2097_209746


namespace NUMINAMATH_GPT_victory_saved_less_l2097_209797

-- Definitions based on conditions
def total_savings : ℕ := 1900
def sam_savings : ℕ := 1000
def victory_savings : ℕ := total_savings - sam_savings

-- Prove that Victory saved $100 less than Sam
theorem victory_saved_less : sam_savings - victory_savings = 100 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_victory_saved_less_l2097_209797


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2097_209774

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2))
  (h2 : ∃ x y z : ℕ, (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  a + a + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2097_209774


namespace NUMINAMATH_GPT_area_acpq_eq_sum_areas_aekl_cdmn_l2097_209747

variables (A B C D E P Q M N K L : Point)

def is_acute_angled_triangle (A B C : Point) : Prop := sorry
def is_altitude (A B C D : Point) : Prop := sorry
def is_square (A P Q C : Point) : Prop := sorry
def is_rectangle (A E K L : Point) : Prop := sorry
def is_rectangle' (C D M N : Point) : Prop := sorry
def length (P Q : Point) : Real := sorry
def area (P Q R S : Point) : Real := sorry

-- Conditions
axiom abc_acute : is_acute_angled_triangle A B C
axiom ad_altitude : is_altitude A B C D
axiom ce_altitude : is_altitude C A B E
axiom acpq_square : is_square A P Q C
axiom aekl_rectangle : is_rectangle A E K L
axiom cdmn_rectangle : is_rectangle' C D M N
axiom al_eq_ab : length A L = length A B
axiom cn_eq_cb : length C N = length C B

-- Question proof statement
theorem area_acpq_eq_sum_areas_aekl_cdmn :
  area A C P Q = area A E K L + area C D M N :=
sorry

end NUMINAMATH_GPT_area_acpq_eq_sum_areas_aekl_cdmn_l2097_209747
