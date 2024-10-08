import Mathlib

namespace chocolate_bars_in_box_l195_195631

theorem chocolate_bars_in_box (x : ℕ) (h1 : 2 * (x - 4) = 18) : x = 13 := 
by {
  sorry
}

end chocolate_bars_in_box_l195_195631


namespace line_passes_through_fixed_point_equal_intercepts_line_equation_l195_195455

open Real

theorem line_passes_through_fixed_point (m : ℝ) : ∃ P : ℝ × ℝ, P = (4, 1) ∧ (m + 2) * P.1 - (m + 1) * P.2 - 3 * m - 7 = 0 := 
sorry

theorem equal_intercepts_line_equation (m : ℝ) :
  ((3 * m + 7) / (m + 2) = -(3 * m + 7) / (m + 1)) → (m = -3 / 2) → 
  (∀ (x y : ℝ), (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0 → x + y - 5 = 0) := 
sorry

end line_passes_through_fixed_point_equal_intercepts_line_equation_l195_195455


namespace probability_of_two_red_two_blue_l195_195441

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l195_195441


namespace minimum_sum_distances_square_l195_195008

noncomputable def minimum_sum_of_distances
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : ℝ :=
(1 + Real.sqrt 2) * d

theorem minimum_sum_distances_square
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : minimum_sum_of_distances A B d h_dist = (1 + Real.sqrt 2) * d := by
sorry

end minimum_sum_distances_square_l195_195008


namespace total_steps_l195_195754

theorem total_steps (steps_per_floor : ℕ) (n : ℕ) (m : ℕ) (h : steps_per_floor = 20) (hm : m = 11) (hn : n = 1) : 
  steps_per_floor * (m - n) = 200 :=
by
  sorry

end total_steps_l195_195754


namespace periodic_odd_fn_calc_l195_195723

theorem periodic_odd_fn_calc :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, 0 < x ∧ x < 1 → f x = 4^x) →
  f (-5 / 2) + f 2 = -2 :=
by
  intros f h
  sorry

end periodic_odd_fn_calc_l195_195723


namespace stacy_history_paper_pages_l195_195737

def stacy_paper := 1 -- Number of pages Stacy writes per day
def days_to_finish := 12 -- Number of days Stacy has to finish the paper

theorem stacy_history_paper_pages : stacy_paper * days_to_finish = 12 := by
  sorry

end stacy_history_paper_pages_l195_195737


namespace largest_n_divisible_l195_195560

theorem largest_n_divisible (n : ℕ) (h : (n ^ 3 + 144) % (n + 12) = 0) : n ≤ 84 :=
sorry

end largest_n_divisible_l195_195560


namespace james_chess_learning_time_l195_195554

theorem james_chess_learning_time (R : ℝ) 
    (h1 : R + 49 * R + 100 * (R + 49 * R) = 10100) 
    : R = 2 :=
by 
  sorry

end james_chess_learning_time_l195_195554


namespace problem_quadratic_inequality_l195_195717

theorem problem_quadratic_inequality
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ 4/9)
  (h3 : b = -a)
  (h4 : c = -2*a + 1)
  (h5 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 1) :
  3*a + 2*b + c ≠ 1/3 ∧ 3*a + 2*b + c ≠ 5/4 :=
by
  sorry

end problem_quadratic_inequality_l195_195717


namespace common_factor_of_polynomial_l195_195591

theorem common_factor_of_polynomial :
  ∀ (x : ℝ), (2 * x^2 - 8 * x) = 2 * x * (x - 4) := by
  sorry

end common_factor_of_polynomial_l195_195591


namespace ellipse_tangent_to_rectangle_satisfies_equation_l195_195444

theorem ellipse_tangent_to_rectangle_satisfies_equation
  (a b : ℝ) -- lengths of the semi-major and semi-minor axes of the ellipse
  (h_rect : 4 * a * b = 48) -- the area condition (since the rectangle sides are 2a and 2b)
  (h_ellipse_form : ∃ (a b : ℝ), ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) : 
  a = 4 ∧ b = 3 ∨ a = 3 ∧ b = 4 := 
sorry

end ellipse_tangent_to_rectangle_satisfies_equation_l195_195444


namespace green_valley_ratio_l195_195063

variable (j s : ℕ)

theorem green_valley_ratio (h : (3 / 4 : ℚ) * j = (1 / 2 : ℚ) * s) : s = 3 / 2 * j :=
by
  sorry

end green_valley_ratio_l195_195063


namespace max_candies_l195_195064

theorem max_candies (V M S : ℕ) (hv : V = 35) (hm : 1 ≤ M ∧ M < 35) (hs : S = 35 + M) (heven : Even S) : V + M + S = 136 :=
sorry

end max_candies_l195_195064


namespace range_of_a_l195_195547

variable {R : Type} [LinearOrderedField R]

def f (x a : R) : R := |x - 1| + |x - 2| - a

theorem range_of_a (h : ∀ x : R, f x a > 0) : a < 1 :=
by
  sorry

end range_of_a_l195_195547


namespace solve_fraction_equation_l195_195795

-- Defining the function f
def f (x : ℝ) : ℝ := x + 4

-- Statement of the problem
theorem solve_fraction_equation (x : ℝ) :
  (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ↔ x = 2 / 5 := by
  sorry

end solve_fraction_equation_l195_195795


namespace percentage_four_petals_l195_195513

def total_clovers : ℝ := 200
def percentage_three_petals : ℝ := 0.75
def percentage_two_petals : ℝ := 0.24
def earnings : ℝ := 554 -- cents

theorem percentage_four_petals :
  (total_clovers - (percentage_three_petals * total_clovers + percentage_two_petals * total_clovers)) / total_clovers * 100 = 1 := 
by sorry

end percentage_four_petals_l195_195513


namespace solve_for_m_l195_195988

noncomputable def has_positive_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (m / (x - 3) - 1 / (3 - x) = 2)

theorem solve_for_m (m : ℝ) : has_positive_root m → m = -1 :=
sorry

end solve_for_m_l195_195988


namespace radius_of_inscribed_circle_l195_195747

theorem radius_of_inscribed_circle (r1 r2 : ℝ) (AC BC AB : ℝ) 
  (h1 : AC = 2 * r1)
  (h2 : BC = 2 * r2)
  (h3 : AB = 2 * Real.sqrt (r1^2 + r2^2)) : 
  (r1 + r2 - Real.sqrt (r1^2 + r2^2)) = ((2 * r1 + 2 * r2 - 2 * Real.sqrt (r1^2 + r2^2)) / 2) := 
by
  sorry

end radius_of_inscribed_circle_l195_195747


namespace moving_circle_passes_through_focus_l195_195114

-- Given conditions
def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def is_tangent_to_line (circle_center_x : ℝ) : Prop :=
  circle_center_x + 2 = 0

-- Prove that the point (2,0) lies on the moving circle
theorem moving_circle_passes_through_focus (circle_center_x circle_center_y : ℝ) :
  is_on_parabola circle_center_x circle_center_y →
  is_tangent_to_line circle_center_x →
  (circle_center_x - 2)^2 + circle_center_y^2 = (circle_center_x + 2)^2 :=
by
  -- Proof skipped with sorry.
  sorry

end moving_circle_passes_through_focus_l195_195114


namespace budget_percentage_l195_195573

-- Define the given conditions
def basic_salary_per_hour : ℝ := 7.50
def commission_rate : ℝ := 0.16
def hours_worked : ℝ := 160
def total_sales : ℝ := 25000
def amount_for_insurance : ℝ := 260

-- Define the basic salary, commission, and total earnings
def basic_salary : ℝ := basic_salary_per_hour * hours_worked
def commission : ℝ := commission_rate * total_sales
def total_earnings : ℝ := basic_salary + commission
def amount_for_budget : ℝ := total_earnings - amount_for_insurance

-- Define the proof problem
theorem budget_percentage : (amount_for_budget / total_earnings) * 100 = 95 := by
  simp [basic_salary, commission, total_earnings, amount_for_budget]
  sorry

end budget_percentage_l195_195573


namespace find_c_l195_195258

def P (x : ℝ) (c : ℝ) : ℝ :=
  x^3 + 3*x^2 + c*x + 15

theorem find_c (c : ℝ) : (x - 3 = P x c → c = -23) := by
  sorry

end find_c_l195_195258


namespace simplify_expression_l195_195540

theorem simplify_expression :
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 :=
by
  sorry

end simplify_expression_l195_195540


namespace total_number_of_slices_l195_195447

def number_of_pizzas : ℕ := 7
def slices_per_pizza : ℕ := 2

theorem total_number_of_slices :
  number_of_pizzas * slices_per_pizza = 14 :=
by
  sorry

end total_number_of_slices_l195_195447


namespace mrs_smith_strawberries_l195_195797

theorem mrs_smith_strawberries (girls : ℕ) (strawberries_per_girl : ℕ) 
                                (h1 : girls = 8) (h2 : strawberries_per_girl = 6) :
    girls * strawberries_per_girl = 48 := by
  sorry

end mrs_smith_strawberries_l195_195797


namespace solve_equation1_solve_equation2_l195_195682

def equation1 (x : ℝ) := (x - 1) ^ 2 = 4
def equation2 (x : ℝ) := 2 * x ^ 3 = -16

theorem solve_equation1 (x : ℝ) (h : equation1 x) : x = 3 ∨ x = -1 := 
sorry

theorem solve_equation2 (x : ℝ) (h : equation2 x) : x = -2 := 
sorry

end solve_equation1_solve_equation2_l195_195682


namespace sum_of_a_b_l195_195579

-- Define the conditions in Lean
def a : ℝ := 1
def b : ℝ := 1

-- Define the proof statement
theorem sum_of_a_b : a + b = 2 := by
  sorry

end sum_of_a_b_l195_195579


namespace sum_of_first_15_terms_l195_195360

-- Given conditions: Sum of 4th and 12th term is 24
variable (a d : ℤ) (a_4 a_12 : ℤ)
variable (S : ℕ → ℤ)
variable (arithmetic_series_4_12_sum : 2 * a + 14 * d = 24)
variable (nth_term_def : ∀ n, a + (n - 1) * d = a_n)

-- Question: Sum of the first 15 terms of the progression
theorem sum_of_first_15_terms : S 15 = 180 := by
  sorry

end sum_of_first_15_terms_l195_195360


namespace percentage_reduction_l195_195652

theorem percentage_reduction (original reduced : ℕ) (h₁ : original = 260) (h₂ : reduced = 195) :
  (original - reduced) / original * 100 = 25 := by
  sorry

end percentage_reduction_l195_195652


namespace inequality_solution_l195_195823

theorem inequality_solution (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1 / x + 4 / y) ≥ 9 / 4 := 
sorry

end inequality_solution_l195_195823


namespace pics_per_album_eq_five_l195_195724

-- Definitions based on conditions
def pics_from_phone : ℕ := 5
def pics_from_camera : ℕ := 35
def total_pics : ℕ := pics_from_phone + pics_from_camera
def num_albums : ℕ := 8

-- Statement to prove
theorem pics_per_album_eq_five : total_pics / num_albums = 5 := by
  sorry

end pics_per_album_eq_five_l195_195724


namespace ink_cartridge_15th_month_l195_195187

def months_in_year : ℕ := 12
def first_change_month : ℕ := 1   -- January is the first month

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + (3 * (n - 1))) % months_in_year

theorem ink_cartridge_15th_month : nth_change_month 15 = 7 := by
  -- This is where the proof would go
  sorry

end ink_cartridge_15th_month_l195_195187


namespace problem_statement_l195_195752

theorem problem_statement (x y z : ℝ) (h : (x - z)^2 - 4 * (x - y) * (y - z) = 0) : x + z - 2 * y = 0 :=
sorry

end problem_statement_l195_195752


namespace average_salary_of_all_workers_is_correct_l195_195024

noncomputable def average_salary_all_workers (n_total n_tech : ℕ) (avg_salary_tech avg_salary_others : ℝ) : ℝ :=
  let n_others := n_total - n_tech
  let total_salary_tech := n_tech * avg_salary_tech
  let total_salary_others := n_others * avg_salary_others
  let total_salary := total_salary_tech + total_salary_others
  total_salary / n_total

theorem average_salary_of_all_workers_is_correct :
  average_salary_all_workers 21 7 12000 6000 = 8000 :=
by
  unfold average_salary_all_workers
  sorry

end average_salary_of_all_workers_is_correct_l195_195024


namespace lion_weight_l195_195120

theorem lion_weight :
  ∃ (L : ℝ), 
    (∃ (T P : ℝ), 
      L + T + P = 106.6 ∧ 
      P = T - 7.7 ∧ 
      T = L - 4.8) ∧ 
    L = 41.3 :=
by
  sorry

end lion_weight_l195_195120


namespace harkamal_total_amount_l195_195528

def cost_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_mangoes (quantity rate : ℕ) : ℕ := quantity * rate
def total_amount_paid (cost1 cost2 : ℕ) : ℕ := cost1 + cost2

theorem harkamal_total_amount :
  let grapes_quantity := 8
  let grapes_rate := 70
  let mangoes_quantity := 9
  let mangoes_rate := 65
  total_amount_paid (cost_grapes grapes_quantity grapes_rate) (cost_mangoes mangoes_quantity mangoes_rate) = 1145 := 
by
  sorry

end harkamal_total_amount_l195_195528


namespace largest_divisor_if_n_sq_div_72_l195_195144

theorem largest_divisor_if_n_sq_div_72 (n : ℕ) (h : n > 0) (h72 : 72 ∣ n^2) : ∃ m, m = 12 ∧ m ∣ n :=
by { sorry }

end largest_divisor_if_n_sq_div_72_l195_195144


namespace percentage_of_material_A_in_second_solution_l195_195630

theorem percentage_of_material_A_in_second_solution 
  (material_A_first_solution : ℝ)
  (material_B_first_solution : ℝ)
  (material_B_second_solution : ℝ)
  (material_A_mixture : ℝ)
  (percentage_first_solution_in_mixture : ℝ)
  (percentage_second_solution_in_mixture : ℝ)
  (total_mixture: ℝ)
  (hyp1 : material_A_first_solution = 20 / 100)
  (hyp2 : material_B_first_solution = 80 / 100)
  (hyp3 : material_B_second_solution = 70 / 100)
  (hyp4 : material_A_mixture = 22 / 100)
  (hyp5 : percentage_first_solution_in_mixture = 80 / 100)
  (hyp6 : percentage_second_solution_in_mixture = 20 / 100)
  (hyp7 : percentage_first_solution_in_mixture + percentage_second_solution_in_mixture = total_mixture)
  : ∃ (x : ℝ), x = 30 := by
  sorry

end percentage_of_material_A_in_second_solution_l195_195630


namespace part1_part2_l195_195733

noncomputable def f (x : ℝ) : ℝ := |x - 1| - 1
noncomputable def g (x : ℝ) : ℝ := -|x + 1| - 4

theorem part1 (x : ℝ) : f x ≤ 1 ↔ -1 ≤ x ∧ x ≤ 3 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ 4 :=
by
  sorry

end part1_part2_l195_195733


namespace range_of_m_l195_195701

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) → m ≤ 9 / 8 :=
by
  intro h
  -- We need to implement the proof here
  sorry

end range_of_m_l195_195701


namespace find_the_number_l195_195075

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end find_the_number_l195_195075


namespace sqrt_meaningful_range_l195_195293

theorem sqrt_meaningful_range (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l195_195293


namespace number_of_ears_pierced_l195_195629

-- Definitions for the conditions
def nosePiercingPrice : ℝ := 20
def earPiercingPrice := nosePiercingPrice + 0.5 * nosePiercingPrice
def totalAmountMade : ℝ := 390
def nosesPierced : ℕ := 6
def totalFromNoses := nosesPierced * nosePiercingPrice
def totalFromEars := totalAmountMade - totalFromNoses

-- The proof statement
theorem number_of_ears_pierced : totalFromEars / earPiercingPrice = 9 := by
  sorry

end number_of_ears_pierced_l195_195629


namespace third_day_sales_correct_l195_195847

variable (a : ℕ)

def firstDaySales := a
def secondDaySales := a + 4
def thirdDaySales := 2 * (a + 4) - 7
def expectedSales := 2 * a + 1

theorem third_day_sales_correct : thirdDaySales a = expectedSales a :=
by
  -- Main proof goes here
  sorry

end third_day_sales_correct_l195_195847


namespace beam_count_represents_number_of_beams_l195_195743

def price := 6210
def transport_cost_per_beam := 3
def beam_condition (x : ℕ) : Prop := 
  transport_cost_per_beam * x * (x - 1) = price

theorem beam_count_represents_number_of_beams (x : ℕ) :
  beam_condition x → (∃ n : ℕ, x = n) := 
sorry

end beam_count_represents_number_of_beams_l195_195743


namespace sum_of_roots_eq_five_thirds_l195_195900

-- Define the quadratic equation
def quadratic_eq (n : ℝ) : Prop := 3 * n^2 - 5 * n - 4 = 0

-- Prove that the sum of the solutions to the quadratic equation is 5/3
theorem sum_of_roots_eq_five_thirds :
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 5 / 3) :=
sorry

end sum_of_roots_eq_five_thirds_l195_195900


namespace value_of_diamond_l195_195448

def diamond (a b : ℕ) : ℕ := 4 * a + 2 * b

theorem value_of_diamond : diamond 6 3 = 30 :=
by {
  sorry
}

end value_of_diamond_l195_195448


namespace complex_problem_l195_195349

theorem complex_problem (z : ℂ) (h : (i * z + z) = 2) : z = 1 - i :=
sorry

end complex_problem_l195_195349


namespace nonagon_line_segments_not_adjacent_l195_195581

def nonagon_segments (n : ℕ) : ℕ :=
(n * (n - 3)) / 2

theorem nonagon_line_segments_not_adjacent (h : ∃ n, n = 9) :
  nonagon_segments 9 = 27 :=
by
  -- proof omitted
  sorry

end nonagon_line_segments_not_adjacent_l195_195581


namespace part1_part2_l195_195725

theorem part1 (a : ℝ) (x : ℝ) (h : a > 0) :
  (|x + 1/a| + |x - a + 1|) ≥ 1 :=
sorry

theorem part2 (a : ℝ) (h1 : a > 0) (h2 : |3 + 1/a| + |3 - a + 1| < 11/2) :
  2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part1_part2_l195_195725


namespace book_page_count_l195_195543

theorem book_page_count (x : ℝ) : 
    (x - (1 / 4 * x + 20)) - ((1 / 3 * (x - (1 / 4 * x + 20)) + 25)) - (1 / 2 * ((x - (1 / 4 * x + 20)) - (1 / 3 * (x - (1 / 4 * x + 20)) + 25)) + 30) = 70 →
    x = 480 :=
by
  sorry

end book_page_count_l195_195543


namespace John_other_trip_length_l195_195651

theorem John_other_trip_length :
  ∀ (fuel_per_km total_fuel first_trip_length other_trip_length : ℕ),
    fuel_per_km = 5 →
    total_fuel = 250 →
    first_trip_length = 20 →
    total_fuel / fuel_per_km - first_trip_length = other_trip_length →
    other_trip_length = 30 :=
by
  intros fuel_per_km total_fuel first_trip_length other_trip_length h1 h2 h3 h4
  sorry

end John_other_trip_length_l195_195651


namespace cos_of_complementary_angle_l195_195705

theorem cos_of_complementary_angle (Y Z : ℝ) (h : Y + Z = π / 2) 
  (sin_Y : Real.sin Y = 3 / 5) : Real.cos Z = 3 / 5 := 
  sorry

end cos_of_complementary_angle_l195_195705


namespace time_comparison_l195_195844

noncomputable def pedestrian_speed : Real := 6.5
noncomputable def cyclist_speed : Real := 20.0
noncomputable def distance_between_points_B_A : Real := 4 * Real.pi - 6.5
noncomputable def alley_distance : Real := 4 * Real.pi - 6.5
noncomputable def combined_speed_3 : Real := pedestrian_speed + cyclist_speed
noncomputable def combined_speed_2 : Real := 21.5
noncomputable def time_scenario_3 : Real := (4 * Real.pi - 6.5) / combined_speed_3
noncomputable def time_scenario_2 : Real := (10.5 - 2 * Real.pi) / combined_speed_2

theorem time_comparison : time_scenario_2 < time_scenario_3 :=
by
  sorry

end time_comparison_l195_195844


namespace new_pressure_of_nitrogen_gas_l195_195728

variable (p1 p2 v1 v2 k : ℝ)

theorem new_pressure_of_nitrogen_gas :
  (∀ p v, p * v = k) ∧ (p1 = 8) ∧ (v1 = 3) ∧ (p1 * v1 = k) ∧ (v2 = 7.5) →
  p2 = 3.2 :=
by
  intro h
  sorry

end new_pressure_of_nitrogen_gas_l195_195728


namespace integer_solution_count_eq_eight_l195_195944

theorem integer_solution_count_eq_eight : ∃ S : Finset (ℤ × ℤ), (∀ s ∈ S, 2 * s.1 ^ 2 + s.1 * s.2 - s.2 ^ 2 = 14 ∧ (s.1 = s.1 ∧ s.2 = s.2)) ∧ S.card = 8 :=
by
  sorry

end integer_solution_count_eq_eight_l195_195944


namespace find_number_l195_195234

theorem find_number (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) (number : ℕ) :
  quotient = 9 ∧ remainder = 1 ∧ divisor = 30 → number = 271 := by
  intro h
  sorry

end find_number_l195_195234


namespace smallest_number_condition_l195_195578

theorem smallest_number_condition :
  ∃ n, 
  (n > 0) ∧ 
  (∀ k, k < n → (n - 3) % 12 = 0 ∧ (n - 3) % 16 = 0 ∧ (n - 3) % 18 = 0 ∧ (n - 3) % 21 = 0 ∧ (n - 3) % 28 = 0 → k = 0) ∧
  (n - 3) % 12 = 0 ∧
  (n - 3) % 16 = 0 ∧
  (n - 3) % 18 = 0 ∧
  (n - 3) % 21 = 0 ∧
  (n - 3) % 28 = 0 ∧
  n = 1011 :=
sorry

end smallest_number_condition_l195_195578


namespace find_a2_l195_195387

def arithmetic_sequence (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n + d 

def sum_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a2 (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a a1 d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : a1 = -2010)
  (h4 : (S 2010) / 2010 - (S 2008) / 2008 = 2) :
  a 2 = -2008 :=
sorry

end find_a2_l195_195387


namespace rectangle_horizontal_length_l195_195738

variable (squareside rectheight : ℕ)

-- Condition: side of the square is 80 cm, vertical side length of the rectangle is 100 cm
def square_side_length := 80
def rect_vertical_length := 100

-- Question: Calculate the horizontal length of the rectangle
theorem rectangle_horizontal_length :
  (4 * square_side_length) = (2 * rect_vertical_length + 2 * rect_horizontal_length) -> rect_horizontal_length = 60 := by
  sorry

end rectangle_horizontal_length_l195_195738


namespace angle_BAO_eq_angle_CAH_l195_195976

noncomputable def is_triangle (A B C : Type) : Prop := sorry
noncomputable def orthocenter (A B C H : Type) : Prop := sorry
noncomputable def circumcenter (A B C O : Type) : Prop := sorry
noncomputable def angle (A B C : Type) : Type := sorry

theorem angle_BAO_eq_angle_CAH (A B C H O : Type) 
  (hABC : is_triangle A B C)
  (hH : orthocenter A B C H)
  (hO : circumcenter A B C O):
  angle B A O = angle C A H := 
  sorry

end angle_BAO_eq_angle_CAH_l195_195976


namespace neighbor_to_johnson_yield_ratio_l195_195759

-- Definitions
def johnsons_yield (months : ℕ) : ℕ := 80 * (months / 2)
def neighbors_yield_per_hectare (x : ℕ) (months : ℕ) : ℕ := 80 * x * (months / 2)
def total_neighor_yield (x : ℕ) (months : ℕ) : ℕ := 2 * neighbors_yield_per_hectare x months

-- Theorem statement
theorem neighbor_to_johnson_yield_ratio
  (x : ℕ)
  (h1 : johnsons_yield 6 = 240)
  (h2 : total_neighor_yield x 6 = 480 * x)
  (h3 : johnsons_yield 6 + total_neighor_yield x 6 = 1200)
  : x = 2 := by
sorry

end neighbor_to_johnson_yield_ratio_l195_195759


namespace take_home_pay_l195_195407

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l195_195407


namespace average_speed_is_42_l195_195848

theorem average_speed_is_42 (v t : ℝ) (h : t > 0)
  (h_eq : v * t = (v + 21) * (2/3) * t) : v = 42 :=
by
  sorry

end average_speed_is_42_l195_195848


namespace fixed_point_C_D_intersection_l195_195990

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

noncomputable def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ P.2 ≠ 0

noncomputable def line_CD_fixed_point (t : ℝ) (C D : ℝ × ℝ) : Prop :=
  let x1 := (36 - 2 * t^2) / (18 + t^2)
  let y1 := (12 * t) / (18 + t^2)
  let x2 := (2 * t^2 - 4) / (2 + t^2)
  let y2 := -(4 * t) / (t^2 + 2)
  C = (x1, y1) ∧ D = (x2, y2) →
  let k_CD := (4 * t) / (6 - t^2)
  ∀ (x y : ℝ), y + (4 * t) / (t^2 + 2) = k_CD * (x - (2 * t^2 - 4) / (t^2 + 2)) →
  y = 0 → x = 1

theorem fixed_point_C_D_intersection :
  ∀ (t : ℝ) (C D : ℝ × ℝ), point_on_line (4, t) →
  ellipse_equation C.1 C.2 →
  ellipse_equation D.1 D.2 →
  line_CD_fixed_point t C D :=
by
  intros t C D point_on_line_P ellipse_C ellipse_D
  sorry

end fixed_point_C_D_intersection_l195_195990


namespace inequality_one_over_a_plus_one_over_b_geq_4_l195_195489

theorem inequality_one_over_a_plus_one_over_b_geq_4 
    (a b : ℕ) (hapos : 0 < a) (hbpos : 0 < b) (h : a + b = 1) : 
    (1 : ℚ) / a + (1 : ℚ) / b ≥ 4 := 
  sorry

end inequality_one_over_a_plus_one_over_b_geq_4_l195_195489


namespace max_sum_of_arithmetic_sequence_l195_195666

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ)
  (d : ℤ) (h_a : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 2 + a 3 = 156)
  (h2 : a 2 + a 3 + a 4 = 147) :
  ∃ n : ℕ, n = 19 ∧ (∀ m : ℕ, S m ≤ S n) :=
sorry

end max_sum_of_arithmetic_sequence_l195_195666


namespace lattice_points_on_hyperbola_l195_195623

-- Define the problem
def countLatticePoints (n : ℤ) : ℕ :=
  let factoredCount := (2 + 1) * (2 + 1) * (4 + 1) -- Number of divisors of 2^2 * 3^2 * 5^4
  2 * factoredCount -- Each pair has two solutions considering positive and negative values

-- The theorem to be proven
theorem lattice_points_on_hyperbola : countLatticePoints 1800 = 90 := sorry

end lattice_points_on_hyperbola_l195_195623


namespace frank_total_points_l195_195304

def points_defeating_enemies (enemies : ℕ) (points_per_enemy : ℕ) : ℕ :=
  enemies * points_per_enemy

def total_points (points_from_enemies : ℕ) (completion_points : ℕ) : ℕ :=
  points_from_enemies + completion_points

theorem frank_total_points :
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  total_points points_from_enemies completion_points = 62 :=
by
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  -- Placeholder for proof
  sorry

end frank_total_points_l195_195304


namespace limit_of_sequence_N_of_epsilon_l195_195066

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end limit_of_sequence_N_of_epsilon_l195_195066


namespace marbles_in_jar_l195_195452

theorem marbles_in_jar (g y p : ℕ) (h1 : y + p = 7) (h2 : g + p = 10) (h3 : g + y = 5) :
  g + y + p = 11 :=
sorry

end marbles_in_jar_l195_195452


namespace arrange_athletes_l195_195161

theorem arrange_athletes :
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  (Nat.choose athletes country_athletes) *
  (Nat.choose (athletes - country_athletes) country_athletes) *
  (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
  (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520 :=
by
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  show (Nat.choose athletes country_athletes) *
       (Nat.choose (athletes - country_athletes) country_athletes) *
       (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
       (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520
  sorry

end arrange_athletes_l195_195161


namespace circle_eq_l195_195298

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l195_195298


namespace value_of_x_squared_plus_9y_squared_l195_195851

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l195_195851


namespace multiplier_is_five_l195_195057

-- condition 1: n = m * (n - 4)
-- condition 2: n = 5
-- question: prove m = 5

theorem multiplier_is_five (n m : ℝ) 
  (h1 : n = m * (n - 4)) 
  (h2 : n = 5) : m = 5 := 
  sorry

end multiplier_is_five_l195_195057


namespace hockey_season_length_l195_195981

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end hockey_season_length_l195_195981


namespace votes_difference_l195_195734

theorem votes_difference (V : ℝ) (h1 : 0.62 * V = 899) :
  |(0.62 * V) - (0.38 * V)| = 348 :=
by
  -- The solution goes here
  sorry

end votes_difference_l195_195734


namespace flare_initial_velocity_and_duration_l195_195300

noncomputable def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

theorem flare_initial_velocity_and_duration (v t : ℝ) :
  (h v 5 = 245) ↔ (v = 73.5) ∧ (5 < t ∧ t < 10) :=
by {
  sorry
}

end flare_initial_velocity_and_duration_l195_195300


namespace ms_cole_total_students_l195_195487

def students_6th : ℕ := 40
def students_4th : ℕ := 4 * students_6th
def students_7th : ℕ := 2 * students_4th

def total_students : ℕ := students_6th + students_4th + students_7th

theorem ms_cole_total_students :
  total_students = 520 :=
by
  sorry

end ms_cole_total_students_l195_195487


namespace seating_arrangement_correct_l195_195693

noncomputable def seatingArrangements (committee : Fin 10) : Nat :=
  Nat.factorial 9

theorem seating_arrangement_correct :
  seatingArrangements committee = 362880 :=
by sorry

end seating_arrangement_correct_l195_195693


namespace children_division_into_circles_l195_195416

theorem children_division_into_circles (n m k : ℕ) (hn : n = 5) (hm : m = 2) (trees_indistinguishable : true) (children_distinguishable : true) :
  ∃ ways, ways = 50 := 
by
  sorry

end children_division_into_circles_l195_195416


namespace area_outside_circle_of_equilateral_triangle_l195_195896

noncomputable def equilateral_triangle_area_outside_circle {a : ℝ} (h : a > 0) : ℝ :=
  let S1 := a^2 * Real.sqrt 3 / 4
  let S2 := Real.pi * (a / 3)^2
  let S3 := (Real.pi * (a / 3)^2 / 6) - (a^2 * Real.sqrt 3 / 36)
  S1 - S2 + 3 * S3

theorem area_outside_circle_of_equilateral_triangle
  (a : ℝ) (h : a > 0) :
  equilateral_triangle_area_outside_circle h = a^2 * (3 * Real.sqrt 3 - Real.pi) / 18 :=
sorry

end area_outside_circle_of_equilateral_triangle_l195_195896


namespace river_width_proof_l195_195592
noncomputable def river_width (V FR D : ℝ) : ℝ := V / (FR * D)

theorem river_width_proof :
  river_width 2933.3333333333335 33.33333333333333 4 = 22 :=
by
  simp [river_width]
  norm_num
  sorry

end river_width_proof_l195_195592


namespace length_of_MN_l195_195053

-- Define the lengths and trapezoid properties
variables (a b: ℝ)

-- Define the problem statement
theorem length_of_MN (a b: ℝ) :
  ∃ (MN: ℝ), ∀ (M N: ℝ) (is_trapezoid : True),
  (MN = 3 * a * b / (a + 2 * b)) :=
sorry

end length_of_MN_l195_195053


namespace range_of_a_l195_195676

theorem range_of_a (a : ℝ) (x : ℝ) :
  (x^2 - 4 * a * x + 3 * a^2 < 0 → (x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0)) → a < 0 → 
  (a ≤ -4 ∨ -2 / 3 ≤ a ∧ a < 0) :=
by
  sorry

end range_of_a_l195_195676


namespace matrix_pow_50_l195_195427

open Matrix

-- Define the given matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 2; -8, -5]

-- Define the expected result for C^50
def C_50 : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-199, -100; 400, 199]

-- Proposition asserting that C^50 equals the given result matrix
theorem matrix_pow_50 :
  C ^ 50 = C_50 := 
  by
  sorry

end matrix_pow_50_l195_195427


namespace sin_2x_and_tan_fraction_l195_195980

open Real

theorem sin_2x_and_tan_fraction (x : ℝ) (h : sin (π + x) + cos (π + x) = 1 / 2) :
  (sin (2 * x) = -3 / 4) ∧ ((1 + tan x) / (sin x * cos (x - π / 4)) = -8 * sqrt 2 / 3) :=
by
  sorry

end sin_2x_and_tan_fraction_l195_195980


namespace total_people_selected_l195_195773

-- Define the number of residents in each age group
def residents_21_to_35 : Nat := 840
def residents_36_to_50 : Nat := 700
def residents_51_to_65 : Nat := 560

-- Define the number of people selected from the 36 to 50 age group
def selected_36_to_50 : Nat := 100

-- Define the total number of residents
def total_residents : Nat := residents_21_to_35 + residents_36_to_50 + residents_51_to_65

-- Theorem: Prove that the total number of people selected in this survey is 300
theorem total_people_selected : (100 : ℕ) / (700 : ℕ) * (residents_21_to_35 + residents_36_to_50 + residents_51_to_65) = 300 :=
  by 
    sorry

end total_people_selected_l195_195773


namespace f_neg2_minus_f_neg3_l195_195972

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end f_neg2_minus_f_neg3_l195_195972


namespace probability_two_females_one_male_l195_195979

theorem probability_two_females_one_male :
  let total_contestants := 8
  let num_females := 5
  let num_males := 3
  let choose3 := Nat.choose total_contestants 3
  let choose2f := Nat.choose num_females 2
  let choose1m := Nat.choose num_males 1
  let favorable_outcomes := choose2f * choose1m
  choose3 ≠ 0 → (favorable_outcomes / choose3 : ℚ) = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l195_195979


namespace students_in_lower_grades_l195_195381

noncomputable def seniors : ℕ := 300
noncomputable def percentage_cars_seniors : ℝ := 0.40
noncomputable def percentage_cars_remaining : ℝ := 0.10
noncomputable def total_percentage_cars : ℝ := 0.15

theorem students_in_lower_grades (X : ℝ) :
  (0.15 * (300 + X) = 120 + 0.10 * X) → X = 1500 :=
by
  intro h
  sorry

end students_in_lower_grades_l195_195381


namespace rectangleY_has_tileD_l195_195671

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define tiles
def TileA : Tile := { top := 6, right := 3, bottom := 5, left := 2 }
def TileB : Tile := { top := 3, right := 6, bottom := 2, left := 5 }
def TileC : Tile := { top := 5, right := 7, bottom := 1, left := 2 }
def TileD : Tile := { top := 2, right := 5, bottom := 6, left := 3 }

-- Define rectangles (positioning)
inductive Rectangle
| W | X | Y | Z

-- Define which tile is in Rectangle Y
def tileInRectangleY : Tile → Prop :=
  fun t => t = TileD

-- Statement to prove
theorem rectangleY_has_tileD : tileInRectangleY TileD :=
by
  -- The final statement to be proven, skipping the proof itself with sorry
  sorry

end rectangleY_has_tileD_l195_195671


namespace evaluate_polynomial_l195_195536

theorem evaluate_polynomial
  (x : ℝ)
  (h1 : x^2 - 3 * x - 9 = 0)
  (h2 : 0 < x)
  : x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = 8 :=
sorry

end evaluate_polynomial_l195_195536


namespace determine_f_l195_195037

theorem determine_f (d e f : ℝ) 
  (h_eq : ∀ y : ℝ, (-3) = d * y^2 + e * y + f)
  (h_vertex : ∀ k : ℝ, -1 = d * (3 - k)^2 + e * (3 - k) + f) :
  f = -5 / 2 :=
sorry

end determine_f_l195_195037


namespace bus_stop_l195_195243

theorem bus_stop (M H : ℕ) 
  (h1 : H = 2 * (M - 15))
  (h2 : M - 15 = 5 * (H - 45)) :
  M = 40 ∧ H = 50 := 
sorry

end bus_stop_l195_195243


namespace ABCD_area_is_correct_l195_195152

-- Define rectangle ABCD with the given conditions
def ABCD_perimeter (x : ℝ) : Prop :=
  2 * (4 * x + x) = 160

-- Define the area to be proved
def ABCD_area (x : ℝ) : ℝ :=
  4 * (x ^ 2)

-- The proof problem: given the conditions, the area should be 1024 square centimeters
theorem ABCD_area_is_correct (x : ℝ) (h : ABCD_perimeter x) : 
  ABCD_area x = 1024 := 
by {
  sorry
}

end ABCD_area_is_correct_l195_195152


namespace calculate_expression_l195_195438

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end calculate_expression_l195_195438


namespace neg_70kg_represents_subtract_70kg_l195_195675

theorem neg_70kg_represents_subtract_70kg (add_30kg : Int) (concept_opposite : ∀ (x : Int), x = -(-x)) :
  -70 = -70 := 
by
  sorry

end neg_70kg_represents_subtract_70kg_l195_195675


namespace brothers_percentage_fewer_trees_l195_195951

theorem brothers_percentage_fewer_trees (total_trees initial_days brother_days : ℕ) (trees_per_day : ℕ) (total_brother_trees : ℕ) (percentage_fewer : ℕ):
  initial_days = 2 →
  brother_days = 3 →
  trees_per_day = 20 →
  total_trees = 196 →
  total_brother_trees = total_trees - (trees_per_day * initial_days) →
  percentage_fewer = ((total_brother_trees / brother_days - trees_per_day) * 100) / trees_per_day →
  percentage_fewer = 60 :=
by
  sorry

end brothers_percentage_fewer_trees_l195_195951


namespace stratified_sampling_third_year_students_l195_195190

theorem stratified_sampling_third_year_students 
  (N : ℕ) (N_1 : ℕ) (P_sophomore : ℝ) (n : ℕ) (N_2 : ℕ) :
  N = 2000 →
  N_1 = 760 →
  P_sophomore = 0.37 →
  n = 20 →
  N_2 = Nat.ceil (N - N_1 - P_sophomore * N) →
  Nat.floor ((n : ℝ) / (N : ℝ) * (N_2 : ℝ)) = 5 :=
by
  sorry

end stratified_sampling_third_year_students_l195_195190


namespace probability_not_sit_at_ends_l195_195587

theorem probability_not_sit_at_ends (h1: ∀ M J: ℕ, M ≠ J → M ≠ 1 ∧ M ≠ 8 ∧ J ≠ 1 ∧ J ≠ 8) : 
  (∃ p: ℚ, p = (3 / 7)) :=
by 
  sorry

end probability_not_sit_at_ends_l195_195587


namespace space_shuttle_speed_conversion_l195_195269

-- Define the given conditions
def speed_km_per_sec : ℕ := 6  -- Speed in km/s
def seconds_per_hour : ℕ := 3600  -- Seconds in an hour

-- Define the computed speed in km/hr
def expected_speed_km_per_hr : ℕ := 21600  -- Expected speed in km/hr

-- The main theorem statement to be proven
theorem space_shuttle_speed_conversion : speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hr := by
  sorry

end space_shuttle_speed_conversion_l195_195269


namespace shifted_polynomial_roots_are_shifted_l195_195077

noncomputable def original_polynomial : Polynomial ℝ := Polynomial.X ^ 3 - 5 * Polynomial.X + 7
noncomputable def shifted_polynomial : Polynomial ℝ := Polynomial.X ^ 3 + 6 * Polynomial.X ^ 2 + 7 * Polynomial.X + 5

theorem shifted_polynomial_roots_are_shifted :
  (∀ (a b c : ℝ), (original_polynomial.eval a = 0) ∧ (original_polynomial.eval b = 0) ∧ (original_polynomial.eval c = 0) 
    → (shifted_polynomial.eval (a - 2) = 0) ∧ (shifted_polynomial.eval (b - 2) = 0) ∧ (shifted_polynomial.eval (c - 2) = 0)) :=
by
  sorry

end shifted_polynomial_roots_are_shifted_l195_195077


namespace shauna_min_test_score_l195_195088

theorem shauna_min_test_score (score1 score2 score3 : ℕ) (h1 : score1 = 82) (h2 : score2 = 88) (h3 : score3 = 95) 
  (max_score : ℕ) (h4 : max_score = 100) (desired_avg : ℕ) (h5 : desired_avg = 85) :
  ∃ (score4 score5 : ℕ), score4 ≥ 75 ∧ score5 ≥ 75 ∧ (score1 + score2 + score3 + score4 + score5) / 5 = desired_avg :=
by
  -- proof here
  sorry

end shauna_min_test_score_l195_195088


namespace largest_n_value_l195_195768

theorem largest_n_value (n : ℕ) (h1: n < 100000) (h2: (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0) : n = 99996 := 
sorry

end largest_n_value_l195_195768


namespace problem_proof_l195_195370

noncomputable def problem (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y ≥ 0 ∧ x ^ 2019 + y = 1) → (x + y ^ 2019 > 1 - 1 / 300)

theorem problem_proof (x y : ℝ) : problem x y :=
by
  intros h
  sorry

end problem_proof_l195_195370


namespace students_who_did_not_receive_an_A_l195_195126

def total_students : ℕ := 40
def a_in_literature : ℕ := 10
def a_in_science : ℕ := 18
def a_in_both : ℕ := 6

theorem students_who_did_not_receive_an_A :
  total_students - ((a_in_literature + a_in_science) - a_in_both) = 18 :=
by
  sorry

end students_who_did_not_receive_an_A_l195_195126


namespace number_of_white_tiles_l195_195099

-- Definition of conditions in the problem
def side_length_large_square := 81
def area_large_square := side_length_large_square * side_length_large_square
def area_black_tiles := 81
def num_red_tiles := 154
def area_red_tiles := num_red_tiles * 4
def area_covered_by_black_and_red := area_black_tiles + area_red_tiles
def remaining_area_for_white_tiles := area_large_square - area_covered_by_black_and_red
def area_of_one_white_tile := 2
def expected_num_white_tiles := 2932

-- The theorem to prove
theorem number_of_white_tiles :
  remaining_area_for_white_tiles / area_of_one_white_tile = expected_num_white_tiles :=
by
  sorry

end number_of_white_tiles_l195_195099


namespace find_x_for_y_equals_six_l195_195744

variable (x y k : ℚ)

-- Conditions
def varies_inversely_as_square := x = k / y^2
def initial_condition := (y = 3 ∧ x = 1)

-- Problem Statement
theorem find_x_for_y_equals_six (h₁ : varies_inversely_as_square x y k) (h₂ : initial_condition x y) :
  ∃ k, (k = 9 ∧ x = k / 6^2 ∧ x = 1 / 4) :=
sorry

end find_x_for_y_equals_six_l195_195744


namespace polynomial_divisibility_l195_195655

theorem polynomial_divisibility (m : ℤ) : (3 * (-2)^2 + 5 * (-2) + m = 0) ↔ (m = -2) :=
by
  sorry

end polynomial_divisibility_l195_195655


namespace minimum_value_proof_l195_195508

noncomputable def minimum_value (a b : ℝ) (h : 0 < a ∧ 0 < b) : ℝ :=
  1 / (2 * a) + 1 / b

theorem minimum_value_proof (a b : ℝ) (h : 0 < a ∧ 0 < b)
  (line_bisects_circle : a + b = 1) : minimum_value a b h = (3 + 2 * Real.sqrt 2) / 2 := 
by
  sorry

end minimum_value_proof_l195_195508


namespace average_speed_is_50_l195_195090

-- Defining the conditions
def totalDistance : ℕ := 250
def totalTime : ℕ := 5

-- Defining the average speed
def averageSpeed := totalDistance / totalTime

-- The theorem statement
theorem average_speed_is_50 : averageSpeed = 50 := sorry

end average_speed_is_50_l195_195090


namespace total_profit_is_correct_l195_195522

-- Definitions of the investments
def A_initial_investment : ℝ := 12000
def B_investment : ℝ := 16000
def C_investment : ℝ := 20000
def D_investment : ℝ := 24000
def E_investment : ℝ := 18000
def C_profit_share : ℝ := 36000

-- Definitions of the time periods (in months)
def time_6_months : ℝ := 6
def time_12_months : ℝ := 12

-- Calculations of investment-months for each person
def A_investment_months : ℝ := A_initial_investment * time_6_months
def B_investment_months : ℝ := B_investment * time_12_months
def C_investment_months : ℝ := C_investment * time_12_months
def D_investment_months : ℝ := D_investment * time_12_months
def E_investment_months : ℝ := E_investment * time_6_months

-- Calculation of total investment-months
def total_investment_months : ℝ :=
  A_investment_months + B_investment_months + C_investment_months +
  D_investment_months + E_investment_months

-- The main theorem stating the total profit calculation
theorem total_profit_is_correct :
  ∃ TP : ℝ, (C_profit_share / C_investment_months) = (TP / total_investment_months) ∧ TP = 135000 :=
by
  sorry

end total_profit_is_correct_l195_195522


namespace max_of_four_expressions_l195_195204

theorem max_of_four_expressions :
  996 * 996 > 995 * 997 ∧ 996 * 996 > 994 * 998 ∧ 996 * 996 > 993 * 999 :=
by
  sorry

end max_of_four_expressions_l195_195204


namespace min_stamps_for_target_value_l195_195977

theorem min_stamps_for_target_value :
  ∃ (c f : ℕ), 5 * c + 7 * f = 50 ∧ ∀ (c' f' : ℕ), 5 * c' + 7 * f' = 50 → c + f ≤ c' + f' → c + f = 8 :=
by
  sorry

end min_stamps_for_target_value_l195_195977


namespace vans_needed_l195_195399

theorem vans_needed (boys girls students_per_van total_vans : ℕ) 
  (hb : boys = 60) 
  (hg : girls = 80) 
  (hv : students_per_van = 28) 
  (t : total_vans = (boys + girls) / students_per_van) : 
  total_vans = 5 := 
by {
  sorry
}

end vans_needed_l195_195399


namespace marble_group_size_l195_195633

-- Define the conditions
def num_marbles : ℕ := 220
def future_people (x : ℕ) : ℕ := x + 2
def marbles_per_person (x : ℕ) : ℕ := num_marbles / x
def marbles_if_2_more (x : ℕ) : ℕ := num_marbles / future_people x

-- Statement of the theorem
theorem marble_group_size (x : ℕ) :
  (marbles_per_person x - 1 = marbles_if_2_more x) ↔ x = 20 :=
sorry

end marble_group_size_l195_195633


namespace tom_initial_foreign_exchange_l195_195986

theorem tom_initial_foreign_exchange (x : ℝ) (y₀ y₁ y₂ y₃ y₄ : ℝ) :
  y₀ = x / 2 - 5 ∧
  y₁ = y₀ / 2 - 5 ∧
  y₂ = y₁ / 2 - 5 ∧
  y₃ = y₂ / 2 - 5 ∧
  y₄ = y₃ / 2 - 5 ∧
  y₄ - 5 = 100
  → x = 3355 :=
by
  intro h
  sorry

end tom_initial_foreign_exchange_l195_195986


namespace annual_rent_per_square_foot_l195_195565

theorem annual_rent_per_square_foot (length width : ℕ) (monthly_rent : ℕ)
  (h_length : length = 20) (h_width : width = 15) (h_monthly_rent : monthly_rent = 3600) :
  let area := length * width
  let annual_rent := monthly_rent * 12
  let annual_rent_per_sq_ft := annual_rent / area
  annual_rent_per_sq_ft = 144 := by
  sorry

end annual_rent_per_square_foot_l195_195565


namespace line_equation_l195_195502

theorem line_equation (t : ℝ) : 
  ∃ m b, (∀ x y : ℝ, (x, y) = (3 * t + 6, 5 * t - 7) → y = m * x + b) ∧
  m = 5 / 3 ∧ b = -17 :=
by
  use 5 / 3, -17
  sorry

end line_equation_l195_195502


namespace minimum_fencing_l195_195702

variable (a b z : ℝ)

def area_condition : Prop := a * b = 50
def length_condition : Prop := a + 2 * b = z

theorem minimum_fencing (h1 : area_condition a b) (h2 : length_condition a b z) : z ≥ 20 := 
  sorry

end minimum_fencing_l195_195702


namespace eccentricity_of_ellipse_l195_195957

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem eccentricity_of_ellipse {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
                                 (h_angle : Real.cos (Real.pi / 6) = b / a) :
    eccentricity a b = (Real.sqrt 6) / 3 := by
  sorry

end eccentricity_of_ellipse_l195_195957


namespace division_subtraction_l195_195203

theorem division_subtraction : 144 / (12 / 3) - 5 = 31 := by
  sorry

end division_subtraction_l195_195203


namespace min_x_y_l195_195256

open Real

theorem min_x_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 := 
sorry

end min_x_y_l195_195256


namespace simplify_expression_l195_195153

theorem simplify_expression (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 5) * (2 * x - 1) - 
  (2 * x - 1) * (x^2 + 2 * x - 8) + 
  (x^2 - 2 * x + 3) * (2 * x - 1) * (x - 2) = 
  8 * x^4 - 2 * x^3 - 5 * x^2 + 32 * x - 15 := 
  sorry

end simplify_expression_l195_195153


namespace planting_trees_system_of_equations_l195_195230

/-- This formalizes the problem where we have 20 young pioneers in total, 
each boy planted 3 trees, each girl planted 2 trees,
and together they planted a total of 52 tree seedlings.
We need to formalize proving that the system of linear equations is as follows:
x + y = 20
3x + 2y = 52
-/
theorem planting_trees_system_of_equations (x y : ℕ) (h1 : x + y = 20)
  (h2 : 3 * x + 2 * y = 52) : 
  (x + y = 20 ∧ 3 * x + 2 * y = 52) :=
by
  exact ⟨h1, h2⟩

end planting_trees_system_of_equations_l195_195230


namespace CA_eq_A_intersection_CB_eq_l195_195292

-- Definitions as per conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | x > 1 }

-- Proof problems as per questions and answers
theorem CA_eq : (U \ A) = { x : ℝ | x ≥ 2 } :=
by
  sorry

theorem A_intersection_CB_eq : (A ∩ (U \ B)) = { x : ℝ | x ≤ 1 } :=
by
  sorry

end CA_eq_A_intersection_CB_eq_l195_195292


namespace sum_of_roots_l195_195450

theorem sum_of_roots (m n : ℝ) (h1 : ∀ x, x^2 - 3 * x - 1 = 0 → x = m ∨ x = n) : m + n = 3 :=
sorry

end sum_of_roots_l195_195450


namespace car_R_speed_l195_195796

theorem car_R_speed (v : ℝ) (h1 : ∀ t_R t_P : ℝ, t_R * v = 800 ∧ t_P * (v + 10) = 800) (h2 : ∀ t_R t_P : ℝ, t_P + 2 = t_R) :
  v = 50 := by
  sorry

end car_R_speed_l195_195796


namespace bisections_needed_l195_195198

theorem bisections_needed (ε : ℝ) (ε_pos : ε = 0.01) (h : 0 < ε) : 
  ∃ n : ℕ, n ≤ 7 ∧ 1 / (2^n) < ε :=
by
  sorry

end bisections_needed_l195_195198


namespace number_of_true_propositions_is_zero_l195_195042

theorem number_of_true_propositions_is_zero :
  (∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0) →
  (¬ ∃ x : ℚ, x^2 = 2) →
  (¬ ∃ x : ℝ, x^2 + 1 = 0) →
  (∀ x : ℝ, 4 * x^2 ≤ 2 * x - 1 + 3 * x^2) →
  true :=  -- representing that the number of true propositions is 0
by
  intros h1 h2 h3 h4
  sorry

end number_of_true_propositions_is_zero_l195_195042


namespace cyclic_sum_inequality_l195_195601

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  ( ( (a - b) * (a - c) / (a + b + c) ) + 
    ( (b - c) * (b - d) / (b + c + d) ) + 
    ( (c - d) * (c - a) / (c + d + a) ) + 
    ( (d - a) * (d - b) / (d + a + b) ) ) ≥ 0 := 
by
  sorry

end cyclic_sum_inequality_l195_195601


namespace beads_pulled_out_l195_195898

theorem beads_pulled_out (white_beads black_beads : ℕ) (frac_black frac_white : ℚ) (h_black : black_beads = 90) (h_white : white_beads = 51) (h_frac_black : frac_black = (1/6)) (h_frac_white : frac_white = (1/3)) : 
  white_beads * frac_white + black_beads * frac_black = 32 := 
by
  sorry

end beads_pulled_out_l195_195898


namespace trees_died_more_than_survived_l195_195658

theorem trees_died_more_than_survived :
  ∀ (initial_trees survived_percent : ℕ),
    initial_trees = 25 →
    survived_percent = 40 →
    (initial_trees * survived_percent / 100) + (initial_trees - initial_trees * survived_percent / 100) -
    (initial_trees * survived_percent / 100) = 5 :=
by
  intro initial_trees survived_percent initial_trees_eq survived_percent_eq
  sorry

end trees_died_more_than_survived_l195_195658


namespace remainder_when_587421_divided_by_6_l195_195425

theorem remainder_when_587421_divided_by_6 :
  ¬ (587421 % 2 = 0) → (587421 % 3 = 0) → 587421 % 6 = 3 :=
by sorry

end remainder_when_587421_divided_by_6_l195_195425


namespace groups_of_four_on_plane_l195_195660

-- Define the points in the tetrahedron
inductive Point
| vertex : Point
| midpoint : Point

noncomputable def points : List Point :=
  [Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.midpoint,
   Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.vertex]

-- Condition: all 10 points are either vertices or midpoints of the edges of a tetrahedron 
def points_condition : ∀ p ∈ points, p = Point.vertex ∨ p = Point.midpoint := sorry

-- Function to count unique groups of four points lying on the same plane
noncomputable def count_groups : ℕ :=
  33  -- Given as the correct answer in the problem

-- Proof problem stating the count of groups
theorem groups_of_four_on_plane : count_groups = 33 :=
by 
  sorry -- Proof omitted

end groups_of_four_on_plane_l195_195660


namespace initial_percentage_decrease_l195_195404

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₁ : P > 0) (h₂ : 1.55 * (1 - x / 100) = 1.24) :
    x = 20 :=
by
  sorry

end initial_percentage_decrease_l195_195404


namespace floor_ceil_diff_l195_195435

theorem floor_ceil_diff (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌊x⌋ + x - ⌈x⌉ = x - 1 :=
sorry

end floor_ceil_diff_l195_195435


namespace income_of_deceased_is_correct_l195_195383

-- Definitions based on conditions
def family_income_before_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def family_income_after_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def income_of_deceased (total_before: ℝ) (total_after: ℝ) : ℝ := total_before - total_after

-- Given conditions
def avg_income_before : ℝ := 782
def avg_income_after : ℝ := 650
def num_members_before : ℕ := 4
def num_members_after : ℕ := 3

-- Mathematical statement
theorem income_of_deceased_is_correct : 
  income_of_deceased (family_income_before_death avg_income_before num_members_before) 
                     (family_income_after_death avg_income_after num_members_after) = 1178 :=
by
  sorry

end income_of_deceased_is_correct_l195_195383


namespace freshman_class_count_l195_195984

theorem freshman_class_count : ∃ n : ℤ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧ n = 49 := by
  sorry

end freshman_class_count_l195_195984


namespace charlie_delta_four_products_l195_195289

noncomputable def charlie_delta_purchase_ways : ℕ := 1363

theorem charlie_delta_four_products :
  let cakes := 6
  let cookies := 4
  let total := cakes + cookies
  ∃ ways : ℕ, ways = charlie_delta_purchase_ways :=
by
  sorry

end charlie_delta_four_products_l195_195289


namespace sandy_age_l195_195585

theorem sandy_age (S M : ℕ) (h1 : M = S + 18) (h2 : S * 9 = M * 7) : S = 63 := by
  sorry

end sandy_age_l195_195585


namespace boys_who_did_not_bring_laptops_l195_195627

-- Definitions based on the conditions.
def total_boys : ℕ := 20
def students_who_brought_laptops : ℕ := 25
def girls_who_brought_laptops : ℕ := 16

-- Main theorem statement.
theorem boys_who_did_not_bring_laptops : total_boys - (students_who_brought_laptops - girls_who_brought_laptops) = 11 := by
  sorry

end boys_who_did_not_bring_laptops_l195_195627


namespace greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l195_195958

theorem greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30 :
  ∃ d, d ∣ 480 ∧ d < 60 ∧ d ∣ 90 ∧ (∀ e, e ∣ 480 → e < 60 → e ∣ 90 → e ≤ d) ∧ d = 30 :=
sorry

end greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l195_195958


namespace income_growth_rate_l195_195076

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l195_195076


namespace jade_initial_pieces_l195_195316

theorem jade_initial_pieces (n w l p : ℕ) (hn : n = 11) (hw : w = 7) (hl : l = 23) (hp : p = n * w + l) : p = 100 :=
by
  sorry

end jade_initial_pieces_l195_195316


namespace find_b_l195_195182

-- Define the conditions as given in the problem
def poly1 (x : ℝ) : ℝ := x^2 - 2 * x - 1
def poly2 (x a b : ℝ) : ℝ := a * x^3 + b * x^2 + 1

-- Define the problem statement using these conditions
theorem find_b (a b : ℤ) (h : ∀ x, poly1 x = 0 → poly2 x a b = 0) : b = -3 :=
sorry

end find_b_l195_195182


namespace min_matches_to_win_champion_min_total_matches_if_wins_11_l195_195729

-- Define the conditions and problem in Lean 4
def teams := ["A", "B", "C"]
def players_per_team : ℕ := 9
def initial_matches : ℕ := 0

-- The minimum number of matches the champion team must win
theorem min_matches_to_win_champion (H : ∀ t ∈ teams, t ≠ "Champion" → players_per_team = 0) :
  initial_matches + 19 = 19 :=
by
  sorry

-- The minimum total number of matches if the champion team wins 11 matches
theorem min_total_matches_if_wins_11 (wins_by_champion : ℕ := 11) (H : wins_by_champion = 11) :
  initial_matches + wins_by_champion + (players_per_team * 2 - wins_by_champion) + 4 = 24 :=
by
  sorry

end min_matches_to_win_champion_min_total_matches_if_wins_11_l195_195729


namespace maximum_value_of_M_l195_195353

noncomputable def M (x : ℝ) : ℝ :=
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x)

theorem maximum_value_of_M : 
  ∃ x : ℝ, M x = (Real.sqrt 3) / 4 :=
sorry

end maximum_value_of_M_l195_195353


namespace find_b_fixed_point_extremum_l195_195503

theorem find_b_fixed_point_extremum (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, f x = x ^ 3 + b * x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ (∀ x : ℝ, deriv f x₀ = 3 * x₀ ^ 2 + b) ∧ deriv f x₀ = 0) →
  b = -3 :=
by
  sorry

end find_b_fixed_point_extremum_l195_195503


namespace hypotenuse_length_l195_195470

theorem hypotenuse_length (a b : ℕ) (h : a = 9 ∧ b = 12) : ∃ c : ℕ, c = 15 ∧ a * a + b * b = c * c :=
by
  sorry

end hypotenuse_length_l195_195470


namespace root_expression_value_l195_195327

theorem root_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : 2021 - 2 * a^2 - 2 * a = 2019 := 
by sorry

end root_expression_value_l195_195327


namespace sum_first_four_terms_l195_195469

theorem sum_first_four_terms (a : ℕ → ℤ) (h5 : a 5 = 5) (h6 : a 6 = 9) (h7 : a 7 = 13) : 
  a 1 + a 2 + a 3 + a 4 = -20 :=
sorry

end sum_first_four_terms_l195_195469


namespace wedding_cost_correct_l195_195363

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def john_guests : ℕ := 50
def wife_guest_increase : ℕ := john_guests * 60 / 100
def total_wedding_cost : ℕ := venue_cost + cost_per_guest * (john_guests + wife_guest_increase)

theorem wedding_cost_correct : total_wedding_cost = 50000 :=
by
  sorry

end wedding_cost_correct_l195_195363


namespace min_value_inequality_l195_195639

theorem min_value_inequality (p q r : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) (h₂ : 0 < r) :
  ( 3 * r / (p + 2 * q) + 3 * p / (2 * r + q) + 2 * q / (p + r) ) ≥ (29 / 6) := 
sorry

end min_value_inequality_l195_195639


namespace problem_solution_l195_195336

theorem problem_solution (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3) →
  (a₁ + a₂ + a₃ = 19) :=
by
  -- Given condition: for any real number x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3
  -- We need to prove: a₁ + a₂ + a₃ = 19
  sorry

end problem_solution_l195_195336


namespace train_length_l195_195436

theorem train_length (s : ℝ) (t : ℝ) (h_s : s = 60) (h_t : t = 10) :
  ∃ L : ℝ, L = 166.7 := by
  sorry

end train_length_l195_195436


namespace river_depth_is_correct_l195_195786

noncomputable def depth_of_river (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_min : ℝ) : ℝ :=
  let flow_rate_mpm := (flow_rate_kmph * 1000) / 60
  let cross_sectional_area := volume_per_min / flow_rate_mpm
  cross_sectional_area / width

theorem river_depth_is_correct :
  depth_of_river 65 6 26000 = 4 :=
by
  -- Steps to compute depth (converted from solution)
  sorry

end river_depth_is_correct_l195_195786


namespace square_perimeter_l195_195157

theorem square_perimeter (s : ℝ) (h1 : (2 * (s + s / 4)) = 40) :
  4 * s = 64 :=
by
  sorry

end square_perimeter_l195_195157


namespace office_needs_24_pencils_l195_195792

noncomputable def number_of_pencils (total_cost : ℝ) (cost_per_pencil : ℝ) (cost_per_folder : ℝ) (number_of_folders : ℕ) : ℝ :=
  (total_cost - (number_of_folders * cost_per_folder)) / cost_per_pencil

theorem office_needs_24_pencils :
  number_of_pencils 30 0.5 0.9 20 = 24 :=
by
  sorry

end office_needs_24_pencils_l195_195792


namespace zinc_weight_in_mixture_l195_195787

theorem zinc_weight_in_mixture (total_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) (total_parts : ℝ) (fraction_zinc : ℝ) (weight_zinc : ℝ) :
  zinc_ratio = 9 ∧ copper_ratio = 11 ∧ total_weight = 70 ∧ total_parts = zinc_ratio + copper_ratio ∧
  fraction_zinc = zinc_ratio / total_parts ∧ weight_zinc = fraction_zinc * total_weight →
  weight_zinc = 31.5 :=
by
  intros h
  sorry

end zinc_weight_in_mixture_l195_195787


namespace inequality_x2_y4_z6_l195_195938

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end inequality_x2_y4_z6_l195_195938


namespace determine_common_ratio_l195_195147

variable (a : ℕ → ℝ) (q : ℝ)

-- Given conditions
axiom a2 : a 2 = 1 / 2
axiom a5 : a 5 = 4
axiom geom_seq_def : ∀ n, a n = a 1 * q ^ (n - 1)

-- Prove the common ratio q == 2
theorem determine_common_ratio : q = 2 :=
by
  -- here we should unfold the proof steps given in the solution
  sorry

end determine_common_ratio_l195_195147


namespace smallest_natural_number_l195_195212

theorem smallest_natural_number :
  ∃ n : ℕ, (n > 0) ∧ (7 * n % 10000 = 2012) ∧ ∀ m : ℕ, (7 * m % 10000 = 2012) → (n ≤ m) :=
sorry

end smallest_natural_number_l195_195212


namespace BoatCrafters_boats_total_l195_195730

theorem BoatCrafters_boats_total
  (n_february: ℕ)
  (h_february: n_february = 5)
  (h_march: 3 * n_february = 15)
  (h_april: 3 * 15 = 45) :
  n_february + 15 + 45 = 65 := 
sorry

end BoatCrafters_boats_total_l195_195730


namespace opposite_pairs_l195_195054

theorem opposite_pairs :
  (3^2 = 9) ∧ (-3^2 = -9) ∧
  ¬ ((3^2 = 9 ∧ -2^3 = -8) ∧ 9 = -(-8)) ∧
  ¬ ((3^2 = 9 ∧ (-3)^2 = 9) ∧ 9 = -9) ∧
  ¬ ((-3^2 = -9 ∧ -(-3)^2 = -9) ∧ -9 = -(-9)) :=
by
  sorry

end opposite_pairs_l195_195054


namespace xy_in_N_l195_195718

def M := {x : ℤ | ∃ m : ℤ, x = 3 * m + 1}
def N := {y : ℤ | ∃ n : ℤ, y = 3 * n + 2}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : (x * y) ∈ N :=
by
  sorry

end xy_in_N_l195_195718


namespace remainder_div_by_13_l195_195765

-- Define conditions
variable (N : ℕ)
variable (k : ℕ)

-- Given condition
def condition := N = 39 * k + 19

-- Goal statement
theorem remainder_div_by_13 (h : condition N k) : N % 13 = 6 :=
sorry

end remainder_div_by_13_l195_195765


namespace rectangle_section_properties_l195_195013

structure Tetrahedron where
  edge_length : ℝ

structure RectangleSection where
  perimeter : ℝ
  area : ℝ

def regular_tetrahedron : Tetrahedron :=
  { edge_length := 1 }

theorem rectangle_section_properties :
  ∀ (rect : RectangleSection), 
  (∃ tetra : Tetrahedron, tetra = regular_tetrahedron) →
  (rect.perimeter = 2) ∧ (0 ≤ rect.area) ∧ (rect.area ≤ 1/4) :=
by
  -- Provide the hypothesis of the existence of such a tetrahedron and rectangular section
  sorry

end rectangle_section_properties_l195_195013


namespace farmers_acres_to_clean_l195_195229

-- Definitions of the main quantities
variables (A D : ℕ)

-- Conditions
axiom condition1 : A = 80 * D
axiom condition2 : 90 * (D - 1) + 30 = A

-- Theorem asserting the total number of acres to be cleaned
theorem farmers_acres_to_clean : A = 480 :=
by
  -- The proof would go here, but is omitted as per instructions
  sorry

end farmers_acres_to_clean_l195_195229


namespace problem1_problem2_l195_195751

-- Problem (I)
theorem problem1 (a b : ℝ) (h : a ≥ b ∧ b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := sorry

-- Problem (II)
theorem problem2 (a b c x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 10) 
  (h2 : x^2 + y^2 + z^2 = 40) 
  (h3 : a * x + b * y + c * z = 20) : 
  (a + b + c) / (x + y + z) = 1 / 2 := sorry

end problem1_problem2_l195_195751


namespace slope_l1_parallel_lines_math_proof_problem_l195_195874

-- Define the two lines
def l1 := ∀ x y : ℝ, x + 2 * y + 2 = 0
def l2 (a : ℝ) := ∀ x y : ℝ, a * x + y - 4 = 0

-- Define the assertions
theorem slope_l1 : ∀ x y : ℝ, x + 2 * y + 2 = 0 ↔ y = -1 / 2 * x - 1 := sorry

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) ↔ a = 1 / 2 := sorry

-- Using the assertions to summarize what we need to prove
theorem math_proof_problem (a : ℝ) :
  ((∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) → a = 1 / 2) ∧
  (∀ x y : ℝ, x + 2 * y + 2 = 0 → y = -1 / 2 * x - 1) := sorry

end slope_l1_parallel_lines_math_proof_problem_l195_195874


namespace ribbons_left_l195_195306

theorem ribbons_left {initial_ribbons morning_giveaway afternoon_giveaway ribbons_left : ℕ} 
    (h1 : initial_ribbons = 38) 
    (h2 : morning_giveaway = 14) 
    (h3 : afternoon_giveaway = 16) 
    (h4 : ribbons_left = initial_ribbons - (morning_giveaway + afternoon_giveaway)) : 
  ribbons_left = 8 := 
by 
  sorry

end ribbons_left_l195_195306


namespace arithmetic_sequence_sum_l195_195699

theorem arithmetic_sequence_sum (a b c : ℤ)
  (h1 : ∃ d : ℤ, a = 3 + d)
  (h2 : ∃ d : ℤ, b = 3 + 2 * d)
  (h3 : ∃ d : ℤ, c = 3 + 3 * d)
  (h4 : 3 + 3 * (c - 3) = 15) : a + b + c = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l195_195699


namespace sufficient_but_not_necessary_l195_195659

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x ≥ 2 ∧ y ≥ 2) → x + y ≥ 4 ∧ (¬ (x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_l195_195659


namespace problem_solution_l195_195556

theorem problem_solution (x1 x2 : ℝ) (h1 : x1^2 + x1 - 4 = 0) (h2 : x2^2 + x2 - 4 = 0) (h3 : x1 + x2 = -1) : 
  x1^3 - 5 * x2^2 + 10 = -19 := 
by 
  sorry

end problem_solution_l195_195556


namespace profit_share_ratio_l195_195854

theorem profit_share_ratio (P Q : ℝ) (hP : P = 40000) (hQ : Q = 60000) : P / Q = 2 / 3 :=
by
  rw [hP, hQ]
  norm_num

end profit_share_ratio_l195_195854


namespace Helga_articles_written_this_week_l195_195771

def articles_per_30_minutes : ℕ := 5
def work_hours_per_day : ℕ := 4
def work_days_per_week : ℕ := 5
def extra_hours_thursday : ℕ := 2
def extra_hours_friday : ℕ := 3

def articles_per_hour : ℕ := articles_per_30_minutes * 2
def regular_daily_articles : ℕ := articles_per_hour * work_hours_per_day
def regular_weekly_articles : ℕ := regular_daily_articles * work_days_per_week
def extra_thursday_articles : ℕ := articles_per_hour * extra_hours_thursday
def extra_friday_articles : ℕ := articles_per_hour * extra_hours_friday
def extra_weekly_articles : ℕ := extra_thursday_articles + extra_friday_articles
def total_weekly_articles : ℕ := regular_weekly_articles + extra_weekly_articles

theorem Helga_articles_written_this_week : total_weekly_articles = 250 := by
  sorry

end Helga_articles_written_this_week_l195_195771


namespace correct_operation_l195_195928

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end correct_operation_l195_195928


namespace fraction_simplification_l195_195266

theorem fraction_simplification (a b c x y : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), (y ≠ 0 → (y^2 / x^2) ≠ (y / x))) ∧
  (∀ (a b c : ℝ), (a + c^2) / (b + c^2) ≠ a / b) ∧
  (∀ (a b m : ℝ), ¬(m ≠ -1 → (a + b) / (m * a + m * b) = 1 / 2)) ∧
  (∃ a b : ℝ, (a - b) / (b - a) = -1) :=
  by
  sorry

end fraction_simplification_l195_195266


namespace triangle_area_proof_l195_195608

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (B : ℝ) (hB : B = 2 * Real.pi / 3) (hb : b = Real.sqrt 13) (h_sum : a + c = 4) :
  triangle_area a b c B = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_proof_l195_195608


namespace Agnes_age_now_l195_195607

variable (A : ℕ) (J : ℕ := 6)

theorem Agnes_age_now :
  (2 * (J + 13) = A + 13) → A = 25 :=
by
  intro h
  sorry

end Agnes_age_now_l195_195607


namespace zoo_gorilla_percentage_l195_195776

theorem zoo_gorilla_percentage :
  ∀ (visitors_per_hour : ℕ) (open_hours : ℕ) (gorilla_visitors : ℕ) (total_visitors : ℕ)
    (percentage : ℕ),
  visitors_per_hour = 50 → open_hours = 8 → gorilla_visitors = 320 →
  total_visitors = visitors_per_hour * open_hours →
  percentage = (gorilla_visitors * 100) / total_visitors →
  percentage = 80 :=
by
  intros visitors_per_hour open_hours gorilla_visitors total_visitors percentage
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3, h4] at h5
  exact h5

end zoo_gorilla_percentage_l195_195776


namespace greatest_value_of_NPMK_l195_195525

def is_digit (n : ℕ) : Prop := n < 10

theorem greatest_value_of_NPMK : 
  ∃ M K N P : ℕ, is_digit M ∧ is_digit K ∧ 
  M = K + 1 ∧ M = 9 ∧ K = 8 ∧ 
  1000 * N + 100 * P + 10 * M + K = 8010 ∧ 
  (100 * M + 10 * M + K) * M = 8010 := by
  sorry

end greatest_value_of_NPMK_l195_195525


namespace saber_toothed_frog_tails_l195_195401

def tails_saber_toothed_frog (n k : ℕ) (x : ℕ) : Prop :=
  5 * n + 4 * k = 100 ∧ n + x * k = 64

theorem saber_toothed_frog_tails : ∃ x, ∃ n k : ℕ, tails_saber_toothed_frog n k x ∧ x = 3 := 
by
  sorry

end saber_toothed_frog_tails_l195_195401


namespace gcd_1734_816_l195_195170

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end gcd_1734_816_l195_195170


namespace charles_draws_yesterday_after_work_l195_195052

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l195_195052


namespace factor_squared_of_symmetric_poly_l195_195662

theorem factor_squared_of_symmetric_poly (P : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ)
  (h_symm : ∀ x y, P x y = P y x)
  (h_factor : ∀ x y, (x - y) ∣ P x y) :
  ∀ x y, (x - y) ^ 2 ∣ P x y := 
sorry

end factor_squared_of_symmetric_poly_l195_195662


namespace ball_travel_approximately_80_l195_195279

noncomputable def ball_travel_distance : ℝ :=
  let h₀ := 20
  let ratio := 2 / 3
  h₀ + -- first descent
  h₀ * ratio + -- first ascent
  h₀ * ratio + -- second descent
  h₀ * ratio^2 + -- second ascent
  h₀ * ratio^2 + -- third descent
  h₀ * ratio^3 + -- third ascent
  h₀ * ratio^3 + -- fourth descent
  h₀ * ratio^4 -- fourth ascent

theorem ball_travel_approximately_80 :
  abs (ball_travel_distance - 80) < 1 :=
sorry

end ball_travel_approximately_80_l195_195279


namespace four_digit_numbers_condition_l195_195277

theorem four_digit_numbers_condition :
  ∃ (N : Nat), (1000 ≤ N ∧ N < 10000) ∧
               (∃ x a : Nat, N = 1000 * a + x ∧ x = 200 * a ∧ 1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end four_digit_numbers_condition_l195_195277


namespace find_sum_l195_195845

noncomputable def sumPutAtSimpleInterest (R: ℚ) (P: ℚ) := 
  let I := P * R * 5 / 100
  I + 90 = P * (R + 6) * 5 / 100 → P = 300

theorem find_sum (R: ℚ) (P: ℚ) : sumPutAtSimpleInterest R P := by
  sorry

end find_sum_l195_195845


namespace no_rational_roots_of_prime_3_digit_l195_195895

noncomputable def is_prime (n : ℕ) := Nat.Prime n

theorem no_rational_roots_of_prime_3_digit (a b c : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) 
(h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : 0 ≤ c ∧ c ≤ 9) 
(p := 100 * a + 10 * b + c) (hp : is_prime p) (h₃ : 100 ≤ p ∧ p ≤ 999) :
¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
sorry

end no_rational_roots_of_prime_3_digit_l195_195895


namespace owen_wins_with_n_bullseyes_l195_195942

-- Define the parameters and conditions
def initial_score_lead : ℕ := 60
def total_shots : ℕ := 120
def bullseye_points : ℕ := 9
def minimum_points_per_shot : ℕ := 3
def max_points_per_shot : ℕ := 9
def n : ℕ := 111

-- Define the condition for Owen's winning requirement
theorem owen_wins_with_n_bullseyes :
  6 * 111 + 360 > 1020 :=
by
  sorry

end owen_wins_with_n_bullseyes_l195_195942


namespace smallest_solution_l195_195529

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l195_195529


namespace camel_cost_is_5200_l195_195621

-- Definitions of costs in terms of Rs.
variable (C H O E : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : ∃ X : ℕ, X * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 130000

-- Theorem to prove
theorem camel_cost_is_5200 (hC : C = 5200) : C = 5200 :=
by sorry

end camel_cost_is_5200_l195_195621


namespace volleyballs_count_l195_195384

-- Definitions of sports item counts based on given conditions.
def soccer_balls := 20
def basketballs := soccer_balls + 5
def tennis_balls := 2 * soccer_balls
def baseballs := soccer_balls + 10
def hockey_pucks := tennis_balls / 2
def total_items := 180

-- Calculate the total number of known sports items.
def known_items_sum := soccer_balls + basketballs + tennis_balls + baseballs + hockey_pucks

-- Prove the number of volleyballs
theorem volleyballs_count : total_items - known_items_sum = 45 := by
  sorry

end volleyballs_count_l195_195384


namespace parallel_vectors_l195_195557

def vec_a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem parallel_vectors (x : ℝ) : vec_a x = (2, 4) → x = 2 := by
  sorry

end parallel_vectors_l195_195557


namespace cameras_not_in_both_l195_195379

-- Definitions for the given conditions
def shared_cameras : ℕ := 12
def sarah_cameras : ℕ := 24
def mike_unique_cameras : ℕ := 9

-- The proof statement
theorem cameras_not_in_both : (sarah_cameras - shared_cameras) + mike_unique_cameras = 21 := by
  sorry

end cameras_not_in_both_l195_195379


namespace min_value_a_plus_3b_plus_9c_l195_195218

theorem min_value_a_plus_3b_plus_9c {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a + 3*b + 9*c ≥ 27 :=
sorry

end min_value_a_plus_3b_plus_9c_l195_195218


namespace onions_total_l195_195192

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ) 
  (h1: Sara_onions = 4) (h2: Sally_onions = 5) (h3: Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 :=
by
  sorry

end onions_total_l195_195192


namespace min_ab_l195_195550

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a + b + 3 = a * b) : 9 ≤ a * b :=
sorry

end min_ab_l195_195550


namespace magician_earnings_l195_195763

theorem magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (decks_remaining : ℕ) (money_earned : ℕ) : 
    price_per_deck = 7 →
    initial_decks = 16 →
    decks_remaining = 8 →
    money_earned = (initial_decks - decks_remaining) * price_per_deck →
    money_earned = 56 :=
by
  intros hp hi hd he
  rw [hp, hi, hd] at he
  exact he

end magician_earnings_l195_195763


namespace correct_quotient_and_remainder_l195_195044

theorem correct_quotient_and_remainder:
  let incorrect_divisor := 47
  let incorrect_quotient := 5
  let incorrect_remainder := 8
  let incorrect_dividend := incorrect_divisor * incorrect_quotient + incorrect_remainder
  let correct_dividend := 243
  let correct_divisor := 74
  (correct_dividend / correct_divisor = 3 ∧ correct_dividend % correct_divisor = 21) :=
by sorry

end correct_quotient_and_remainder_l195_195044


namespace smallest_w_l195_195128

theorem smallest_w (w : ℕ) (h : 2^5 ∣ 936 * w ∧ 3^3 ∣ 936 * w ∧ 11^2 ∣ 936 * w) : w = 4356 :=
sorry

end smallest_w_l195_195128


namespace combined_wattage_l195_195939

theorem combined_wattage (w1 w2 w3 w4 : ℕ) (h1 : w1 = 60) (h2 : w2 = 80) (h3 : w3 = 100) (h4 : w4 = 120) :
  let nw1 := w1 + w1 / 4
  let nw2 := w2 + w2 / 4
  let nw3 := w3 + w3 / 4
  let nw4 := w4 + w4 / 4
  nw1 + nw2 + nw3 + nw4 = 450 :=
by
  sorry

end combined_wattage_l195_195939


namespace first_grade_muffins_total_l195_195820

theorem first_grade_muffins_total :
  let muffins_brier : ℕ := 218
  let muffins_macadams : ℕ := 320
  let muffins_flannery : ℕ := 417
  let muffins_smith : ℕ := 292
  let muffins_jackson : ℕ := 389
  muffins_brier + muffins_macadams + muffins_flannery + muffins_smith + muffins_jackson = 1636 :=
by
  apply sorry

end first_grade_muffins_total_l195_195820


namespace solution_set_of_quadratic_inequality_l195_195247

theorem solution_set_of_quadratic_inequality 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x < 0 ↔ x < -1 ∨ x > 1 / 3)
  (h₂ : ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) : 
  ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3 := 
by
  intro x
  exact h₂ x

end solution_set_of_quadratic_inequality_l195_195247


namespace perimeter_of_triangle_l195_195443

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 9) + (P.2^2 / 5) = 1

noncomputable def foci_position (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (-2, 0) ∧ F2 = (2, 0)

theorem perimeter_of_triangle :
  ∀ (P F1 F2 : ℝ × ℝ),
    point_on_ellipse P →
    foci_position F1 F2 →
    dist P F1 + dist P F2 + dist F1 F2 = 10 :=
by
  sorry

end perimeter_of_triangle_l195_195443


namespace pairs_of_mittens_correct_l195_195905

variables (pairs_of_plugs_added pairs_of_plugs_original plugs_total pairs_of_plugs_current pairs_of_mittens : ℕ)

theorem pairs_of_mittens_correct :
  pairs_of_plugs_added = 30 →
  plugs_total = 400 →
  pairs_of_plugs_current = plugs_total / 2 →
  pairs_of_plugs_current = pairs_of_plugs_original + pairs_of_plugs_added →
  pairs_of_mittens = pairs_of_plugs_original - 20 →
  pairs_of_mittens = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pairs_of_mittens_correct_l195_195905


namespace height_after_five_years_l195_195339

namespace PapayaTreeGrowth

def growth_first_year := true → ℝ
def growth_second_year (x : ℝ) := 1.5 * x
def growth_third_year (x : ℝ) := 1.5 * growth_second_year x
def growth_fourth_year (x : ℝ) := 2 * growth_third_year x
def growth_fifth_year (x : ℝ) := 0.5 * growth_fourth_year x

def total_growth (x : ℝ) := x + growth_second_year x + growth_third_year x +
                             growth_fourth_year x + growth_fifth_year x

theorem height_after_five_years (x : ℝ) (H : total_growth x = 23) : x = 2 :=
by
  sorry

end PapayaTreeGrowth

end height_after_five_years_l195_195339


namespace optimal_pricing_for_max_profit_l195_195686

noncomputable def sales_profit (x : ℝ) : ℝ :=
  -5 * x^3 + 45 * x^2 - 75 * x + 675

theorem optimal_pricing_for_max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x < 9 ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 9 → sales_profit y ≤ sales_profit 5 ∧ (14 - 5 = 9) :=
by
  sorry

end optimal_pricing_for_max_profit_l195_195686


namespace starting_number_of_range_l195_195755

theorem starting_number_of_range (N : ℕ) : ∃ (start : ℕ), 
  (∀ n, n ≥ start ∧ n ≤ 200 → ∃ k, 8 * k = n) ∧ -- All numbers between start and 200 inclusive are multiples of 8
  (∃ k, k = (200 / 8) ∧ 25 - k = 13.5) ∧ -- There are 13.5 multiples of 8 in the range
  start = 84 := 
sorry

end starting_number_of_range_l195_195755


namespace area_of_trapezoid_RSQT_l195_195790
-- Import the required library

-- Declare the geometrical setup and given areas
variables (PQ PR : ℝ)
variable (PQR_area : ℝ)
variable (small_triangle_area : ℝ)
variable (num_small_triangles : ℕ)
variable (inner_triangle_area : ℝ)
variable (trapezoid_RSQT_area : ℝ)

-- Define the conditions from part a)
def isosceles_triangle : Prop := PQ = PR
def triangle_PQR_area_given : Prop := PQR_area = 75
def small_triangle_area_given : Prop := small_triangle_area = 3
def num_small_triangles_given : Prop := num_small_triangles = 9
def inner_triangle_area_given : Prop := inner_triangle_area = 5 * small_triangle_area

-- Define the target statement (question == answer)
theorem area_of_trapezoid_RSQT :
  isosceles_triangle PQ PR ∧
  triangle_PQR_area_given PQR_area ∧
  small_triangle_area_given small_triangle_area ∧
  num_small_triangles_given num_small_triangles ∧
  inner_triangle_area_given small_triangle_area inner_triangle_area → 
  trapezoid_RSQT_area = 60 :=
sorry

end area_of_trapezoid_RSQT_l195_195790


namespace find_salary_l195_195782

theorem find_salary (S : ℤ) (food house_rent clothes left : ℤ) 
  (h_food : food = S / 5) 
  (h_house_rent : house_rent = S / 10) 
  (h_clothes : clothes = 3 * S / 5) 
  (h_left : left = 18000) 
  (h_spent : food + house_rent + clothes + left = S) : 
  S = 180000 :=
by {
  sorry
}

end find_salary_l195_195782


namespace max_value_y_l195_195649

noncomputable def max_y (a b c d : ℝ) : ℝ :=
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2

theorem max_value_y {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 10) : max_y a b c d = 40 := 
  sorry

end max_value_y_l195_195649


namespace Jasmine_total_weight_in_pounds_l195_195117

-- Definitions for the conditions provided
def weight_chips_ounces : ℕ := 20
def weight_cookies_ounces : ℕ := 9
def bags_chips : ℕ := 6
def tins_cookies : ℕ := 4 * bags_chips
def total_weight_ounces : ℕ := (weight_chips_ounces * bags_chips) + (weight_cookies_ounces * tins_cookies)
def total_weight_pounds : ℕ := total_weight_ounces / 16

-- The proof problem statement
theorem Jasmine_total_weight_in_pounds : total_weight_pounds = 21 := 
by
  sorry

end Jasmine_total_weight_in_pounds_l195_195117


namespace uncommon_card_cost_l195_195361

/--
Tom's deck contains 19 rare cards, 11 uncommon cards, and 30 common cards.
Each rare card costs $1.
Each common card costs $0.25.
The total cost of the deck is $32.
Prove that the cost of each uncommon card is $0.50.
-/
theorem uncommon_card_cost (x : ℝ): 
  let rare_count := 19
  let uncommon_count := 11
  let common_count := 30
  let rare_cost := 1
  let common_cost := 0.25
  let total_cost := 32
  (rare_count * rare_cost) + (common_count * common_cost) + (uncommon_count * x) = total_cost 
  → x = 0.5 :=
by
  sorry

end uncommon_card_cost_l195_195361


namespace find_pairs_l195_195000

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_l195_195000


namespace find_m_l195_195194

noncomputable def hex_to_dec (m : ℕ) : ℕ :=
  3 * 6^4 + m * 6^3 + 5 * 6^2 + 2

theorem find_m (m : ℕ) : hex_to_dec m = 4934 ↔ m = 4 := 
by
  sorry

end find_m_l195_195194


namespace impossible_cube_configuration_l195_195050

theorem impossible_cube_configuration :
  ∀ (cube: ℕ → ℕ) (n : ℕ), 
    (∀ n, 1 ≤ n ∧ n ≤ 27 → ∃ k, 1 ≤ k ∧ k ≤ 27 ∧ cube k = n) →
    (∀ n, 1 ≤ n ∧ n ≤ 27 → (cube 27 = 27 ∧ ∀ m, 1 ≤ m ∧ m ≤ 26 → cube m = 27 - m)) → 
    false :=
by
  intros cube n hcube htarget
  -- any detailed proof steps would go here, skipping with sorry
  sorry

end impossible_cube_configuration_l195_195050


namespace product_of_binomials_l195_195313

-- Definition of the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x - 3
def binomial2 (x : ℝ) : ℝ := x + 7

-- The theorem to be proved
theorem product_of_binomials (x : ℝ) : 
  binomial1 x * binomial2 x = 4 * x^2 + 25 * x - 21 :=
by
  sorry

end product_of_binomials_l195_195313


namespace triangle_area_l195_195418

theorem triangle_area (h b : ℝ) (Hhb : h < b) :
  let P := (0, b)
  let B := (b, 0)
  let D := (h, h)
  let PD := b - h
  let DB := b - h
  1 / 2 * PD * DB = 1 / 2 * (b - h) ^ 2 := by 
  sorry

end triangle_area_l195_195418


namespace miles_on_first_day_l195_195310

variable (x : ℝ)

/-- The distance traveled on the first day is x miles. -/
noncomputable def second_day_distance := (3/4) * x

/-- The distance traveled on the second day is (3/4)x miles. -/
noncomputable def third_day_distance := (1/2) * (x + second_day_distance x)

theorem miles_on_first_day
    (total_distance : x + second_day_distance x + third_day_distance x = 525)
    : x = 200 :=
sorry

end miles_on_first_day_l195_195310


namespace solve_arithmetic_sequence_l195_195109

theorem solve_arithmetic_sequence :
  ∀ (x : ℝ), x > 0 ∧ x^2 = (2^2 + 5^2) / 2 → x = Real.sqrt (29 / 2) :=
by
  intro x
  intro hx
  sorry

end solve_arithmetic_sequence_l195_195109


namespace remainder_when_200_divided_by_k_l195_195004

theorem remainder_when_200_divided_by_k 
  (k : ℕ) (k_pos : 0 < k)
  (h : 120 % k^2 = 12) :
  200 % k = 2 :=
sorry

end remainder_when_200_divided_by_k_l195_195004


namespace union_M_N_eq_N_l195_195377

def M := {x : ℝ | x^2 - 2 * x ≤ 0}
def N := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

theorem union_M_N_eq_N : M ∪ N = N := 
sorry

end union_M_N_eq_N_l195_195377


namespace hours_per_day_l195_195065

theorem hours_per_day (H : ℕ) : 
  (42 * 12 * H = 30 * 14 * 6) → 
  H = 5 := by
  sorry

end hours_per_day_l195_195065


namespace tens_digit_of_13_pow_3007_l195_195812

theorem tens_digit_of_13_pow_3007 : 
  (13 ^ 3007 / 10) % 10 = 1 :=
sorry

end tens_digit_of_13_pow_3007_l195_195812


namespace cube_sum_div_by_nine_l195_195945

theorem cube_sum_div_by_nine (n : ℕ) (hn : 0 < n) : (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 := by sorry

end cube_sum_div_by_nine_l195_195945


namespace total_treat_value_is_339100_l195_195778

def hotel_cost (cost_per_night : ℕ) (nights : ℕ) (discount : ℕ) : ℕ :=
  let total_cost := cost_per_night * nights
  total_cost - (total_cost * discount / 100)

def car_cost (base_price : ℕ) (tax : ℕ) : ℕ :=
  base_price + (base_price * tax / 100)

def house_cost (car_base_price : ℕ) (multiplier : ℕ) (property_tax : ℕ) : ℕ :=
  let house_value := car_base_price * multiplier
  house_value + (house_value * property_tax / 100)

def yacht_cost (hotel_value : ℕ) (car_value : ℕ) (multiplier : ℕ) (discount : ℕ) : ℕ :=
  let combined_value := hotel_value + car_value
  let yacht_value := combined_value * multiplier
  yacht_value - (yacht_value * discount / 100)

def gold_coins_cost (yacht_value : ℕ) (multiplier : ℕ) (tax : ℕ) : ℕ :=
  let gold_value := yacht_value * multiplier
  gold_value + (gold_value * tax / 100)

theorem total_treat_value_is_339100 :
  let hotel_value := hotel_cost 4000 2 5
  let car_value := car_cost 30000 10
  let house_value := house_cost 30000 4 2
  let yacht_value := yacht_cost 8000 30000 2 7
  let gold_coins_value := gold_coins_cost 76000 3 3
  hotel_value + car_value + house_value + yacht_value + gold_coins_value = 339100 :=
by sorry

end total_treat_value_is_339100_l195_195778


namespace smallest_norm_of_v_l195_195978

variables (v : ℝ × ℝ)

def vector_condition (v : ℝ × ℝ) : Prop :=
  ‖(v.1 - 2, v.2 + 4)‖ = 10

theorem smallest_norm_of_v
  (hv : vector_condition v) :
  ‖v‖ ≥ 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_of_v_l195_195978


namespace nat_pairs_solution_l195_195110

theorem nat_pairs_solution (x y : ℕ) :
  2^(2*x+1) + 2^x + 1 = y^2 → (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by
  sorry

end nat_pairs_solution_l195_195110


namespace camila_weeks_needed_l195_195221

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end camila_weeks_needed_l195_195221


namespace pure_imaginary_condition_l195_195558

def z1 : ℂ := 3 - 2 * Complex.I
def z2 (m : ℝ) : ℂ := 1 + m * Complex.I

theorem pure_imaginary_condition (m : ℝ) : z1 * z2 m ∈ {z : ℂ | z.re = 0} ↔ m = -3 / 2 := by
  sorry

end pure_imaginary_condition_l195_195558


namespace total_cost_is_26_30_l195_195911

open Real

-- Define the costs
def cost_snake_toy : ℝ := 11.76
def cost_cage : ℝ := 14.54

-- Define the total cost of purchases
def total_cost : ℝ := cost_snake_toy + cost_cage

-- Prove the total cost equals $26.30
theorem total_cost_is_26_30 : total_cost = 26.30 :=
by
  sorry

end total_cost_is_26_30_l195_195911


namespace alpha_div_beta_is_rational_l195_195309

noncomputable def alpha_is_multiple (α : ℝ) (k : ℕ) : Prop :=
  ∃ k : ℕ, α = k * (2 * Real.pi / 1996)

noncomputable def beta_is_multiple (β : ℝ) (m : ℕ) : Prop :=
  β ≠ 0 ∧ ∃ m : ℕ, β = m * (2 * Real.pi / 1996)

theorem alpha_div_beta_is_rational (α β : ℝ) (k m : ℕ)
  (hα : alpha_is_multiple α k) (hβ : beta_is_multiple β m) :
  ∃ r : ℚ, α / β = r := by
    sorry

end alpha_div_beta_is_rational_l195_195309


namespace non_periodic_decimal_l195_195564

variable {a : ℕ → ℕ}

-- Condition definitions
def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

def constraint (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ 10 * a n

-- Theorem statement
theorem non_periodic_decimal (a : ℕ → ℕ) 
  (h_inc : is_increasing_sequence a) 
  (h_constraint : constraint a) : 
  ¬ (∃ T : ℕ, ∀ n : ℕ, a (n + T) = a n) :=
sorry

end non_periodic_decimal_l195_195564


namespace jade_cal_difference_l195_195100

def Mabel_transactions : ℕ := 90

def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)

def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3

def Jade_transactions : ℕ := 85

theorem jade_cal_difference : Jade_transactions - Cal_transactions = 19 := by
  sorry

end jade_cal_difference_l195_195100


namespace find_a4_l195_195465
open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n / (2 * a n + 3)

theorem find_a4 (a : ℕ → ℚ) (h : seq a) : a 4 = 1 / 53 :=
by
  obtain ⟨h1, h_rec⟩ := h
  have a2 := h_rec 1 (by decide)
  have a3 := h_rec 2 (by decide)
  have a4 := h_rec 3 (by decide)
  -- Proof steps would go here
  sorry

end find_a4_l195_195465


namespace brownie_pieces_count_l195_195828

def area_of_pan (length width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

def number_of_pieces (pan_area piece_area : ℕ) : ℕ := pan_area / piece_area

theorem brownie_pieces_count :
  let pan_length := 24
  let pan_width := 15
  let piece_side := 3
  let pan_area := area_of_pan pan_length pan_width
  let piece_area := area_of_piece piece_side
  number_of_pieces pan_area piece_area = 40 :=
by
  sorry

end brownie_pieces_count_l195_195828


namespace white_surface_area_fraction_l195_195704

theorem white_surface_area_fraction :
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  (white_faces_exposed / surface_area) = (7 / 8) :=
by
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  have h_white_fraction : (white_faces_exposed / surface_area) = (7 / 8) := sorry
  exact h_white_fraction

end white_surface_area_fraction_l195_195704


namespace determine_number_l195_195462

theorem determine_number (x : ℝ) (number : ℝ) (h1 : number / x = 0.03) (h2 : x = 0.3) : number = 0.009 := by
  sorry

end determine_number_l195_195462


namespace find_x_l195_195914

theorem find_x (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (3, x)) (h : (a.fst * b.fst + a.snd * b.snd) = 3) : x = 3 :=
by
  sorry

end find_x_l195_195914


namespace trapezoid_segment_ratio_l195_195784

theorem trapezoid_segment_ratio (s l : ℝ) (h₁ : 3 * s + l = 1) (h₂ : 2 * l + 6 * s = 2) :
  l = 2 * s :=
by
  sorry

end trapezoid_segment_ratio_l195_195784


namespace min_value_inequality_l195_195284

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 3) :
  (1 / x) + (4 / y) + (9 / z) ≥ 12 := 
sorry

end min_value_inequality_l195_195284


namespace correct_system_of_equations_l195_195164

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l195_195164


namespace binomial_coeff_divisibility_l195_195685

theorem binomial_coeff_divisibility (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : n ∣ (Nat.choose n k) * Nat.gcd n k :=
sorry

end binomial_coeff_divisibility_l195_195685


namespace jen_problem_correct_answer_l195_195766

-- Definitions based on the conditions
def sum_178_269 : ℤ := 178 + 269
def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n - (n % 100) + 100 else n - (n % 100)

-- Prove the statement
theorem jen_problem_correct_answer :
  round_to_nearest_hundred sum_178_269 = 400 :=
by
  have h1 : sum_178_269 = 447 := rfl
  have h2 : round_to_nearest_hundred 447 = 400 := by sorry
  exact h2

end jen_problem_correct_answer_l195_195766


namespace binomial_coefficient_and_factorial_l195_195788

open Nat

/--
  Given:
    - The binomial coefficient definition: Nat.choose n k = n! / (k! * (n - k)!)
    - The factorial definition: Nat.factorial n = n * (n - 1) * ... * 1
  Prove:
    Nat.choose 60 3 * Nat.factorial 10 = 124467072000
-/
theorem binomial_coefficient_and_factorial :
  Nat.choose 60 3 * Nat.factorial 10 = 124467072000 :=
by
  sorry

end binomial_coefficient_and_factorial_l195_195788


namespace kevin_found_cards_l195_195904

-- Definitions from the conditions
def initial_cards : ℕ := 7
def final_cards : ℕ := 54

-- The proof goal
theorem kevin_found_cards : final_cards - initial_cards = 47 :=
by
  sorry

end kevin_found_cards_l195_195904


namespace math_problem_l195_195359

variable (a a' b b' c c' : ℝ)

theorem math_problem 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b * b) 
  (h3 : a' * c' ≥ b' * b') : 
  (a + a') * (c + c') ≥ (b + b') * (b + b') := 
by
  sorry

end math_problem_l195_195359


namespace hyperbola_range_l195_195541

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (2 + m) + y^2 / (m + 1) = 1)) → (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_range_l195_195541


namespace domain_of_f_l195_195331

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.log (x + 2)

theorem domain_of_f :
  {x : ℝ | (x ≠ 0) ∧ (x > -2)} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x)} :=
by
  sorry

end domain_of_f_l195_195331


namespace min_additional_games_l195_195393

def num_initial_games : ℕ := 4
def num_lions_won : ℕ := 3
def num_eagles_won : ℕ := 1
def win_threshold : ℝ := 0.90

theorem min_additional_games (M : ℕ) : (num_eagles_won + M) / (num_initial_games + M) ≥ win_threshold ↔ M ≥ 26 :=
by
  sorry

end min_additional_games_l195_195393


namespace lateral_surface_area_eq_total_surface_area_eq_l195_195260

def r := 3
def h := 10

theorem lateral_surface_area_eq : 2 * Real.pi * r * h = 60 * Real.pi := by
  sorry

theorem total_surface_area_eq : 2 * Real.pi * r * h + 2 * Real.pi * r^2 = 78 * Real.pi := by
  sorry

end lateral_surface_area_eq_total_surface_area_eq_l195_195260


namespace rectangle_width_decrease_l195_195519

theorem rectangle_width_decrease (a b : ℝ) (p x : ℝ) 
  (hp : p ≥ 0) (hx : x ≥ 0)
  (area_eq : a * b = (a * (1 + p / 100)) * (b * (1 - x / 100))) :
  x = (100 * p) / (100 + p) := 
by
  sorry

end rectangle_width_decrease_l195_195519


namespace daria_amount_owed_l195_195391

variable (savings : ℝ)
variable (couch_price : ℝ)
variable (table_price : ℝ)
variable (lamp_price : ℝ)
variable (total_cost : ℝ)
variable (amount_owed : ℝ)

theorem daria_amount_owed (h_savings : savings = 500)
                          (h_couch : couch_price = 750)
                          (h_table : table_price = 100)
                          (h_lamp : lamp_price = 50)
                          (h_total_cost : total_cost = couch_price + table_price + lamp_price)
                          (h_amount_owed : amount_owed = total_cost - savings) :
                          amount_owed = 400 :=
by
  sorry

end daria_amount_owed_l195_195391


namespace tangent_line_to_circle_l195_195501

theorem tangent_line_to_circle (a : ℝ) :
  (∃ k : ℝ, k = a ∧ (∀ x y : ℝ, y = x + 4 → (x - k)^2 + (y - 3)^2 = 8)) ↔ (a = 3 ∨ a = -5) := by
  sorry

end tangent_line_to_circle_l195_195501


namespace number_of_sides_of_polygon_l195_195600

-- Given definition about angles and polygons
def exterior_angle (sides: ℕ) : ℝ := 30

-- The sum of exterior angles of any polygon
def sum_exterior_angles : ℝ := 360

-- The proof statement
theorem number_of_sides_of_polygon (k : ℕ) 
  (h1 : exterior_angle k = 30) 
  (h2 : sum_exterior_angles = 360):
  k = 12 :=
sorry

end number_of_sides_of_polygon_l195_195600


namespace value_of_coins_is_77_percent_l195_195314

theorem value_of_coins_is_77_percent :
  let pennies := 2 * 1  -- value of two pennies in cents
  let nickel := 5       -- value of one nickel in cents
  let dimes := 2 * 10   -- value of two dimes in cents
  let half_dollar := 50 -- value of one half-dollar in cents
  let total_cents := pennies + nickel + dimes + half_dollar
  let dollar_in_cents := 100
  (total_cents / dollar_in_cents) * 100 = 77 :=
by
  sorry

end value_of_coins_is_77_percent_l195_195314


namespace correct_average_l195_195264

theorem correct_average (initial_avg : ℝ) (n : ℕ) (error1 : ℝ) (wrong_num : ℝ) (correct_num : ℝ) :
  initial_avg = 40.2 → n = 10 → error1 = 19 → wrong_num = 13 → correct_num = 31 →
  (initial_avg * n - error1 - wrong_num + correct_num) / n = 40.1 :=
by
  intros
  sorry

end correct_average_l195_195264


namespace largest_angle_in_triangle_l195_195996

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A = 45) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : 
  max A (max B C) = 75 :=
by
  -- Since no proof is needed, we mark it as sorry
  sorry

end largest_angle_in_triangle_l195_195996


namespace elberta_money_l195_195940

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) 
  (h1 : granny_smith = 120) 
  (h2 : anjou = granny_smith / 4) 
  (h3 : elberta = anjou + 5) : 
  elberta = 35 :=
by {
  sorry
}

end elberta_money_l195_195940


namespace min_value_of_m_l195_195878

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) := (Real.exp (-x) - Real.exp x) / 2

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → m * g x + h x ≥ 0) → m ≥ (Real.exp 2 - 1) / (Real.exp 2 + 1) :=
by
  intro h
  have key_ineq : ∀ x, -1 ≤ x ∧ x ≤ 1 → m ≥ 1 - 2 / (Real.exp (2 * x) + 1) := sorry
  sorry

end min_value_of_m_l195_195878


namespace division_result_is_correct_l195_195707

def division_result : ℚ := 132 / 6 / 3

theorem division_result_is_correct : division_result = 22 / 3 :=
by
  -- here, we would include the proof steps, but for now, we'll put sorry
  sorry

end division_result_is_correct_l195_195707


namespace fred_spending_correct_l195_195514

noncomputable def fred_total_spending : ℝ :=
  let football_price_each := 2.73
  let football_quantity := 2
  let football_tax_rate := 0.05
  let pokemon_price := 4.01
  let pokemon_tax_rate := 0.08
  let baseball_original_price := 10
  let baseball_discount_rate := 0.10
  let baseball_tax_rate := 0.06
  let football_total_before_tax := football_price_each * football_quantity
  let football_total_tax := football_total_before_tax * football_tax_rate
  let football_total := football_total_before_tax + football_total_tax
  let pokemon_total_tax := pokemon_price * pokemon_tax_rate
  let pokemon_total := pokemon_price + pokemon_total_tax
  let baseball_discount := baseball_original_price * baseball_discount_rate
  let baseball_discounted_price := baseball_original_price - baseball_discount
  let baseball_total_tax := baseball_discounted_price * baseball_tax_rate
  let baseball_total := baseball_discounted_price + baseball_total_tax
  football_total + pokemon_total + baseball_total

theorem fred_spending_correct :
  fred_total_spending = 19.6038 := 
  by
    sorry

end fred_spending_correct_l195_195514


namespace pentagon_segment_condition_l195_195434

-- Define the problem context and hypothesis
variable (a b c d e : ℝ)

theorem pentagon_segment_condition 
  (h₁ : a + b + c + d + e = 3)
  (h₂ : a ≤ b)
  (h₃ : b ≤ c)
  (h₄ : c ≤ d)
  (h₅ : d ≤ e) : 
  a < 3 / 2 ∧ b < 3 / 2 ∧ c < 3 / 2 ∧ d < 3 / 2 ∧ e < 3 / 2 := 
sorry

end pentagon_segment_condition_l195_195434


namespace find_rectangle_pairs_l195_195975

theorem find_rectangle_pairs (w l : ℕ) (hw : w > 0) (hl : l > 0) (h : w * l = 18) : 
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) ∨
  (w, l) = (6, 3) ∨ (w, l) = (9, 2) ∨ (w, l) = (18, 1) :=
by
  sorry

end find_rectangle_pairs_l195_195975


namespace find_tangent_circles_tangent_circle_at_given_point_l195_195297

noncomputable def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 1)^2 = 4

def is_tangent (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ (u v : ℝ), (u - a)^2 + (v - b)^2 = 1 ∧
  (x - u)^2 + (y - v)^2 = 4 ∧
  (x = u ∧ y = v)

theorem find_tangent_circles (x y a b : ℝ) (hx : circle_C x y)
  (ha_b : is_tangent x y a b) :
  (a = 5 ∧ b = -1) ∨ (a = 3 ∧ b = -1) :=
sorry

theorem tangent_circle_at_given_point (x y : ℝ) (hx : circle_C x y) (y_pos : y = -1)
  : ((x - 5)^2 + (y + 1)^2 = 1) ∨ ((x - 3)^2 + (y + 1)^2 = 1) :=
sorry

end find_tangent_circles_tangent_circle_at_given_point_l195_195297


namespace line_intersects_hyperbola_l195_195145

variables (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0)

def line (x y : ℝ) := a * x - y + b = 0

def hyperbola (x y : ℝ) := x^2 / (|a| / |b|) - y^2 / (|b| / |a|) = 1

theorem line_intersects_hyperbola :
  ∃ x y : ℝ, line a b x y ∧ hyperbola a b x y := 
sorry

end line_intersects_hyperbola_l195_195145


namespace chromium_percentage_new_alloy_l195_195330

-- Define the weights and chromium percentages of the alloys
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 35
def chromium_percent_alloy1 : ℝ := 0.15
def chromium_percent_alloy2 : ℝ := 0.08

-- Define the theorem to calculate the chromium percentage of the new alloy
theorem chromium_percentage_new_alloy :
  ((weight_alloy1 * chromium_percent_alloy1 + weight_alloy2 * chromium_percent_alloy2)
  / (weight_alloy1 + weight_alloy2) * 100) = 10.1 :=
by
  sorry

end chromium_percentage_new_alloy_l195_195330


namespace second_number_value_l195_195922

-- Definition of the problem conditions
variables (x y z : ℝ)
axiom h1 : z = 4.5 * y
axiom h2 : y = 2.5 * x
axiom h3 : (x + y + z) / 3 = 165

-- The goal is to prove y = 82.5 given the conditions h1, h2, and h3
theorem second_number_value : y = 82.5 :=
by
  sorry

end second_number_value_l195_195922


namespace contrapositive_of_square_inequality_l195_195953

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x > y → x^2 > y^2) ↔ (x^2 ≤ y^2 → x ≤ y) :=
sorry

end contrapositive_of_square_inequality_l195_195953


namespace solution_set_of_inequality_l195_195495

/-- Given an even function f that is monotonically increasing on [0, ∞) with f(3) = 0,
    show that the solution set for xf(2x - 1) < 0 is (-∞, -1) ∪ (0, 2). -/
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_value : f 3 = 0) :
  {x : ℝ | x * f (2*x - 1) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l195_195495


namespace binary_to_octal_of_101101110_l195_195210

def binaryToDecimal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 2 + b) 0 (Nat.digits 2 n)

def decimalToOctal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 10 + b) 0 (Nat.digits 8 n)

theorem binary_to_octal_of_101101110 :
  decimalToOctal (binaryToDecimal 0b101101110) = 556 :=
by sorry

end binary_to_octal_of_101101110_l195_195210


namespace angle_B_max_area_triangle_l195_195612
noncomputable section

open Real

variables {A B C a b c : ℝ}

-- Prove B = π / 3 given b sin A = √3 a cos B
theorem angle_B (h1 : b * sin A = sqrt 3 * a * cos B) : B = π / 3 :=
sorry

-- Prove if b = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_triangle (h1 : b * sin A = sqrt 3 * a * cos B) (h2 : b = 2 * sqrt 3) : 
    (1 / 2) * a * (a : ℝ) *  (sqrt 3 / 2 : ℝ) ≤ 3 * sqrt 3 :=
sorry

end angle_B_max_area_triangle_l195_195612


namespace ned_time_left_to_diffuse_bomb_l195_195731

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end ned_time_left_to_diffuse_bomb_l195_195731


namespace max_value_of_S_l195_195127

-- Define the sequence sum function
def S (n : ℕ) : ℤ :=
  -2 * (n : ℤ) ^ 3 + 21 * (n : ℤ) ^ 2 + 23 * (n : ℤ)

theorem max_value_of_S :
  ∃ (n : ℕ), S n = 504 ∧ 
             (∀ k : ℕ, S k ≤ 504) :=
sorry

end max_value_of_S_l195_195127


namespace percentage_problem_l195_195046

theorem percentage_problem (x : ℝ) (h : 0.30 * 0.15 * x = 18) : 0.15 * 0.30 * x = 18 :=
by
  sorry

end percentage_problem_l195_195046


namespace algebraic_expression_value_l195_195916

theorem algebraic_expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2 * x + 1) * (2 * x - 1) + x * (x - 1) = 1 :=
by
  sorry

end algebraic_expression_value_l195_195916


namespace OQ_value_l195_195777

variables {X Y Z N O Q R : Type}
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables [MetricSpace N] [MetricSpace O] [MetricSpace Q] [MetricSpace R]
variables (XY YZ XN NY ZO XO OZ YN XR OQ RQ : ℝ)
variables (triangle_XYZ : Triangle X Y Z)
variables (X_equal_midpoint_XY : XY = 540)
variables (Y_equal_midpoint_YZ : YZ = 360)
variables (XN_equal_NY : XN = NY)
variables (ZO_is_angle_bisector : is_angle_bisector Z O X Y)
variables (intersection_YN_ZO : Q = intersection YN ZO)
variables (N_midpoint_RQ : is_midpoint N R Q)
variables (XR_value : XR = 216)

theorem OQ_value : OQ = 216 := sorry

end OQ_value_l195_195777


namespace set_A_is_listed_correctly_l195_195524

def A : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem set_A_is_listed_correctly : A = {-2, -1, 0} := 
by
  sorry

end set_A_is_listed_correctly_l195_195524


namespace complex_triple_sum_eq_sqrt3_l195_195403

noncomputable section

open Complex

theorem complex_triple_sum_eq_sqrt3 {a b c : ℂ} (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : a + b + c ≠ 0) (h5 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3) : abs (a + b + c) = Real.sqrt 3 :=
by
  sorry

end complex_triple_sum_eq_sqrt3_l195_195403


namespace value_of_expression_l195_195208

variable {a b m n x : ℝ}

def opposite (a b : ℝ) : Prop := a = -b
def reciprocal (m n : ℝ) : Prop := m * n = 1
def distance_to_2 (x : ℝ) : Prop := abs (x - 2) = 3

theorem value_of_expression (h1 : opposite a b) (h2 : reciprocal m n) (h3 : distance_to_2 x) :
  (a + b - m * n) * x + (a + b)^2022 + (- m * n)^2023 = 
  if x = 5 then -6 else if x = -1 then 0 else sorry :=
by
  sorry

end value_of_expression_l195_195208


namespace range_of_a_l195_195205

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + Real.exp x - Real.exp (-x)

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l195_195205


namespace no_integer_y_makes_Q_perfect_square_l195_195420

def Q (y : ℤ) : ℤ := y^4 + 8 * y^3 + 18 * y^2 + 10 * y + 41

theorem no_integer_y_makes_Q_perfect_square :
  ¬ ∃ y : ℤ, ∃ b : ℤ, Q y = b^2 :=
by
  intro h
  rcases h with ⟨y, b, hQ⟩
  sorry

end no_integer_y_makes_Q_perfect_square_l195_195420


namespace setB_is_correct_l195_195711

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end setB_is_correct_l195_195711


namespace all_weights_equal_l195_195571

theorem all_weights_equal (w : Fin 13 → ℤ) 
  (h : ∀ (i : Fin 13), ∃ (a b : Multiset (Fin 12)),
    a + b = (Finset.univ.erase i).val ∧ Multiset.card a = 6 ∧ 
    Multiset.card b = 6 ∧ Multiset.sum (a.map w) = Multiset.sum (b.map w)) :
  ∀ i j, w i = w j :=
by sorry

end all_weights_equal_l195_195571


namespace solution_set_range_l195_195842

theorem solution_set_range (x : ℝ) : 
  (2 * |x - 10| + 3 * |x - 20| ≤ 35) ↔ (9 ≤ x ∧ x ≤ 23) :=
sorry

end solution_set_range_l195_195842


namespace find_expression_l195_195067

theorem find_expression (x y : ℝ) (h1 : 4 * x + y = 17) (h2 : x + 4 * y = 23) :
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 :=
by
  sorry

end find_expression_l195_195067


namespace uncle_age_when_seokjin_is_12_l195_195857

-- Definitions for the conditions
def mother_age_when_seokjin_born : ℕ := 32
def uncle_is_younger_by : ℕ := 3
def seokjin_age : ℕ := 12

-- Definition for the main hypothesis
theorem uncle_age_when_seokjin_is_12 :
  let mother_age_when_seokjin_is_12 := mother_age_when_seokjin_born + seokjin_age
  let uncle_age_when_seokjin_is_12 := mother_age_when_seokjin_is_12 - uncle_is_younger_by
  uncle_age_when_seokjin_is_12 = 41 :=
by
  sorry

end uncle_age_when_seokjin_is_12_l195_195857


namespace eval_f_l195_195597

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem eval_f : f (f (1 / 2)) = 1 :=
by
  sorry

end eval_f_l195_195597


namespace fill_pool_time_l195_195818

-- Define the conditions
def pool_volume : ℕ := 15000
def hoses1_rate : ℕ := 2
def hoses1_count : ℕ := 2
def hoses2_rate : ℕ := 3
def hoses2_count : ℕ := 2

-- Calculate the total delivery rate
def total_delivery_rate : ℕ :=
  (hoses1_rate * hoses1_count) + (hoses2_rate * hoses2_count)

-- Calculate the time to fill the pool in minutes
def time_to_fill_in_minutes : ℕ :=
  pool_volume / total_delivery_rate

-- Calculate the time to fill the pool in hours
def time_to_fill_in_hours : ℕ :=
  time_to_fill_in_minutes / 60

-- The theorem to prove
theorem fill_pool_time : time_to_fill_in_hours = 25 := by
  sorry

end fill_pool_time_l195_195818


namespace tribe_leadership_choices_l195_195696

open Nat

theorem tribe_leadership_choices (n m k l : ℕ) (h : n = 15) : 
  (choose 14 2 * choose 12 3 * choose 9 3 * 15 = 27392400) := 
  by sorry

end tribe_leadership_choices_l195_195696


namespace polynomial_roots_quartic_sum_l195_195518

noncomputable def roots_quartic_sum (a b c : ℂ) : ℂ :=
  a^4 + b^4 + c^4

theorem polynomial_roots_quartic_sum :
  ∀ (a b c : ℂ), (a^3 - 3 * a + 1 = 0) ∧ (b^3 - 3 * b + 1 = 0) ∧ (c^3 - 3 * c + 1 = 0) →
  (a + b + c = 0) ∧ (a * b + b * c + c * a = -3) ∧ (a * b * c = -1) →
  roots_quartic_sum a b c = 18 :=
by
  intros a b c hroot hsum
  sorry

end polynomial_roots_quartic_sum_l195_195518


namespace good_numbers_l195_195583

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_odd_prime (n : ℕ) : Prop :=
  Prime n ∧ n % 2 = 1

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_divisor (d + 1) (n + 1)

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ n = 1 ∨ is_odd_prime n :=
sorry

end good_numbers_l195_195583


namespace odd_factors_of_360_l195_195225

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l195_195225


namespace triangle_property_proof_l195_195253

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end triangle_property_proof_l195_195253


namespace sequence_general_formula_l195_195632

theorem sequence_general_formula :
  ∃ (a : ℕ → ℕ), 
    (a 1 = 4) ∧ 
    (∀ n : ℕ, a (n + 1) = a n + 3) ∧ 
    (∀ n : ℕ, a n = 3 * n + 1) :=
sorry

end sequence_general_formula_l195_195632


namespace find_M_value_l195_195132

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end find_M_value_l195_195132


namespace intersection_of_M_and_N_l195_195296

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := 
by sorry

end intersection_of_M_and_N_l195_195296


namespace parabola_equation_l195_195698

theorem parabola_equation (p : ℝ) (hp : 0 < p) (F : ℝ × ℝ) (Q : ℝ × ℝ) (PQ QF : ℝ)
  (hPQ : PQ = 8 / p) (hQF : QF = 8 / p + p / 2) (hDist : QF = 5 / 4 * PQ) : 
  ∃ x, y^2 = 4 * x :=
by
  sorry

end parabola_equation_l195_195698


namespace count_valid_sequences_l195_195091

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_sequence (x : ℕ → ℕ) : Prop :=
  (x 7 % 2 = 0) ∧ (∀ i < 7, (x i % 2 = 0 → x (i + 1) % 2 = 1) ∧ (x i % 2 = 1 → x (i + 1) % 2 = 0))

theorem count_valid_sequences : ∃ n, 
  n = 78125 ∧ 
  ∃ x : ℕ → ℕ, 
    (∀ i < 8, 0 ≤ x i ∧ x i ≤ 9) ∧ valid_sequence x :=
sorry

end count_valid_sequences_l195_195091


namespace necessary_but_not_sufficient_condition_l195_195563

variable (a : ℝ)

theorem necessary_but_not_sufficient_condition (h : 0 ≤ a ∧ a ≤ 4) :
  (∀ x : ℝ, x^2 + a * x + a > 0) → (0 ≤ a ∧ a ≤ 4 ∧ ¬ (∀ x : ℝ, x^2 + a * x + a > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l195_195563


namespace smallest_sum_divisible_by_5_l195_195757

-- Definition of a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of four consecutive primes greater than 5
def four_consecutive_primes_greater_than_five (a b c d : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ a > 5 ∧ b > 5 ∧ c > 5 ∧ d > 5 ∧ 
  b = a + 4 ∧ c = b + 6 ∧ d = c + 2

-- The statement to prove
theorem smallest_sum_divisible_by_5 :
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) % 5 = 0 ∧
   ∀ x y z w : ℕ, four_consecutive_primes_greater_than_five x y z w → (x + y + z + w) % 5 = 0 → a + b + c + d ≤ x + y + z + w) →
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) = 60) :=
by
  sorry

end smallest_sum_divisible_by_5_l195_195757


namespace solve_problem_l195_195680

open Real

noncomputable def problem (x : ℝ) : Prop :=
  (cos (2 * x / 5) - cos (2 * π / 15)) ^ 2 + (sin (2 * x / 3) - sin (4 * π / 9)) ^ 2 = 0

theorem solve_problem : ∀ t : ℤ, problem ((29 * π / 3) + 15 * π * t) :=
by
  intro t
  sorry

end solve_problem_l195_195680


namespace solve_equation_l195_195610

theorem solve_equation (a : ℝ) : 
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} :=
sorry

end solve_equation_l195_195610


namespace find_cos_A_l195_195948

theorem find_cos_A
  (A C : ℝ)
  (AB CD : ℝ)
  (AD BC : ℝ)
  (α : ℝ)
  (h1 : A = C)
  (h2 : AB = 150)
  (h3 : CD = 150)
  (h4 : AD ≠ BC)
  (h5 : AB + BC + CD + AD = 560)
  (h6 : A = α)
  (h7 : C = α)
  (BD₁ BD₂ : ℝ)
  (h8 : BD₁^2 = AD^2 + 150^2 - 2 * 150 * AD * Real.cos α)
  (h9 : BD₂^2 = BC^2 + 150^2 - 2 * 150 * BC * Real.cos α)
  (h10 : BD₁ = BD₂) :
  Real.cos A = 13 / 15 := 
sorry

end find_cos_A_l195_195948


namespace roger_ant_l195_195618

def expected_steps : ℚ := 11/3

theorem roger_ant (a b : ℕ) (h1 : expected_steps = a / b) (h2 : Nat.gcd a b = 1) : 100 * a + b = 1103 :=
sorry

end roger_ant_l195_195618


namespace perfect_square_trinomial_m_l195_195813

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a b : ℤ, (b^2 = 25) ∧ (a + b)^2 = x^2 - (m - 3) * x + 25) → (m = 13 ∨ m = -7) :=
by
  sorry

end perfect_square_trinomial_m_l195_195813


namespace symmetric_point_origin_l195_195840

-- Define the point P
structure Point3D where
  x : Int
  y : Int
  z : Int

def P : Point3D := { x := 1, y := 3, z := -5 }

-- Define the symmetric function w.r.t. the origin
def symmetric_with_origin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Define the expected result
def Q : Point3D := { x := -1, y := -3, z := 5 }

-- The theorem to prove
theorem symmetric_point_origin : symmetric_with_origin P = Q := by
  sorry

end symmetric_point_origin_l195_195840


namespace triangle_side_lengths_l195_195715

noncomputable def radius_inscribed_circle := 4/3
def sum_of_heights := 13

theorem triangle_side_lengths :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (h_a h_b h_c : ℕ), h_a ≠ h_b ∧ h_b ≠ h_c ∧ h_a ≠ h_c ∧
  h_a + h_b + h_c = sum_of_heights ∧
  r * (a + b + c) = 8 ∧ -- (since Δ = r * s, where s = (a + b + c)/2)
  1 / 2 * a * h_a = 1 / 2 * b * h_b ∧
  1 / 2 * b * h_b = 1 / 2 * c * h_c ∧
  a = 6 ∧ b = 4 ∧ c = 3 :=
sorry

end triangle_side_lengths_l195_195715


namespace yule_log_surface_area_increase_l195_195119

noncomputable def yuleLogIncreaseSurfaceArea : ℝ := 
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initialSurfaceArea := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let sliceHeight := h / n
  let sliceSurfaceArea := 2 * Real.pi * r * sliceHeight + 2 * Real.pi * r^2
  let totalSlicesSurfaceArea := n * sliceSurfaceArea
  let increaseSurfaceArea := totalSlicesSurfaceArea - initialSurfaceArea
  increaseSurfaceArea

theorem yule_log_surface_area_increase : yuleLogIncreaseSurfaceArea = 100 * Real.pi := by
  sorry

end yule_log_surface_area_increase_l195_195119


namespace touching_squares_same_color_probability_l195_195491

theorem touching_squares_same_color_probability :
  let m := 0
  let n := 1
  100 * m + n = 1 :=
by
  let m := 0
  let n := 1
  sorry -- Proof is omitted as per instructions

end touching_squares_same_color_probability_l195_195491


namespace jane_project_time_l195_195722

theorem jane_project_time
  (J : ℝ)
  (work_rate_jane_ashley : ℝ := 1 / J + 1 / 40)
  (time_together : ℝ := 15.2 - 8)
  (work_done_together : ℝ := time_together * work_rate_jane_ashley)
  (ashley_alone_time : ℝ := 8)
  (work_done_ashley : ℝ := ashley_alone_time / 40)
  (jane_alone_time : ℝ := 4)
  (work_done_jane_alone : ℝ := jane_alone_time / J) :
  7.2 * (1 / J + 1 / 40) + 8 / 40 + 4 / J = 1 ↔ J = 18.06 :=
by 
  sorry

end jane_project_time_l195_195722


namespace gcd_45123_32768_l195_195191

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end gcd_45123_32768_l195_195191


namespace total_crayons_l195_195394

def box1_crayons := 3 * (8 + 4 + 5)
def box2_crayons := 4 * (7 + 6 + 3)
def box3_crayons := 2 * (11 + 5 + 2)
def unique_box_crayons := 9 + 2 + 7

theorem total_crayons : box1_crayons + box2_crayons + box3_crayons + unique_box_crayons = 169 := by
  sorry

end total_crayons_l195_195394


namespace father_payment_l195_195924

variable (x y : ℤ)

theorem father_payment :
  5 * x - 3 * y = 24 :=
sorry

end father_payment_l195_195924


namespace perfect_square_trinomial_l195_195669

theorem perfect_square_trinomial (m : ℝ) : (∃ (a b : ℝ), (a * x + b) ^ 2 = x^2 + m * x + 16) -> (m = 8 ∨ m = -8) :=
sorry

end perfect_square_trinomial_l195_195669


namespace hexagon_side_equalities_l195_195663

variables {A B C D E F : Type}

-- Define the properties and conditions of the problem
noncomputable def convex_hexagon (A B C D E F : Type) : Prop :=
  True -- Since we neglect geometric properties in this abstract.

def parallel (a b : Type) : Prop := True -- placeholder for parallel condition
def equal_length (a b : Type) : Prop := True -- placeholder for length

-- Given conditions
variables (h1 : convex_hexagon A B C D E F)
variables (h2 : parallel AB DE)
variables (h3 : parallel BC FA)
variables (h4 : parallel CD FA)
variables (h5 : equal_length AB DE)

-- Statement to prove
theorem hexagon_side_equalities : equal_length BC DE ∧ equal_length CD FA := sorry

end hexagon_side_equalities_l195_195663


namespace product_of_ab_l195_195569

theorem product_of_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 13) : a * b = -6 :=
by
  sorry

end product_of_ab_l195_195569


namespace bus_average_speed_excluding_stoppages_l195_195055

theorem bus_average_speed_excluding_stoppages :
  ∀ v : ℝ, (32 / 60) * v = 40 → v = 75 :=
by
  intro v
  intro h
  sorry

end bus_average_speed_excluding_stoppages_l195_195055


namespace range_of_a_l195_195439

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, 2 < (x : ℝ) ∧ (x : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ y : ℤ, 2 < (y : ℝ) ∧ (y : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ z : ℤ, 2 < (z : ℝ) ∧ (z : ℝ) ≤ 2 * a - 1) ∧ 
  (∀ w : ℤ, 2 < (w : ℝ) ∧ (w : ℝ) ≤ 2 * a - 1 → w = 3 ∨ w = 4 ∨ w = 5) :=
  by
    sorry

end range_of_a_l195_195439


namespace inequality_on_positive_reals_l195_195473

variable {a b c : ℝ}

theorem inequality_on_positive_reals (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_on_positive_reals_l195_195473


namespace functional_equation_solution_l195_195673

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l195_195673


namespace arrange_numbers_in_ascending_order_l195_195239

noncomputable def S := 222 ^ 2
noncomputable def T := 22 ^ 22
noncomputable def U := 2 ^ 222
noncomputable def V := 22 ^ (2 ^ 2)
noncomputable def W := 2 ^ (22 ^ 2)
noncomputable def X := 2 ^ (2 ^ 22)
noncomputable def Y := 2 ^ (2 ^ (2 ^ 2))

theorem arrange_numbers_in_ascending_order :
  S < Y ∧ Y < V ∧ V < T ∧ T < U ∧ U < W ∧ W < X :=
sorry

end arrange_numbers_in_ascending_order_l195_195239


namespace not_odd_function_l195_195250

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x ^ 2 + 1)

theorem not_odd_function : ¬ is_odd_function f := by
  sorry

end not_odd_function_l195_195250


namespace ratio_of_b_plus_e_over_c_plus_f_l195_195750

theorem ratio_of_b_plus_e_over_c_plus_f 
  (a b c d e f : ℝ)
  (h1 : a + b = 2 * a + c)
  (h2 : a - 2 * b = 4 * c)
  (h3 : a + b + c = 21)
  (h4 : d + e = 3 * d + f)
  (h5 : d - 2 * e = 5 * f)
  (h6 : d + e + f = 32) :
  (b + e) / (c + f) = -3.99 :=
sorry

end ratio_of_b_plus_e_over_c_plus_f_l195_195750


namespace solve_inequality_l195_195628

theorem solve_inequality (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 6) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 2) :
  (x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Ioo 2 6)) → 
  ((x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Icc 3 5))) :=
by 
  introv h
  sorry

end solve_inequality_l195_195628


namespace water_usage_eq_13_l195_195148

theorem water_usage_eq_13 (m x : ℝ) (h : 16 * m = 10 * m + (x - 10) * 2 * m) : x = 13 :=
by sorry

end water_usage_eq_13_l195_195148


namespace dice_composite_probability_l195_195094

theorem dice_composite_probability (m n : ℕ) (h : Nat.gcd m n = 1) :
  (∃ m n : ℕ, (m * 36 = 29 * n) ∧ Nat.gcd m n = 1) → m + n = 65 :=
by {
  sorry
}

end dice_composite_probability_l195_195094


namespace range_of_d_l195_195877

noncomputable def sn (n a1 d : ℝ) := (n / 2) * (2 * a1 + (n - 1) * d)

theorem range_of_d (a1 d : ℝ) (h_eq : (sn 2 a1 d) * (sn 4 a1 d) / 2 + (sn 3 a1 d) ^ 2 / 9 + 2 = 0) :
  d ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
sorry

end range_of_d_l195_195877


namespace henry_age_is_29_l195_195941

-- Definitions and conditions
variable (Henry_age Jill_age : ℕ)

-- Condition 1: Sum of the present age of Henry and Jill is 48
def sum_of_ages : Prop := Henry_age + Jill_age = 48

-- Condition 2: Nine years ago, Henry was twice the age of Jill
def age_relation_nine_years_ago : Prop := Henry_age - 9 = 2 * (Jill_age - 9)

-- Theorem to prove
theorem henry_age_is_29 (H: ℕ) (J: ℕ)
  (h1 : sum_of_ages H J) 
  (h2 : age_relation_nine_years_ago H J) : H = 29 :=
by
  sorry

end henry_age_is_29_l195_195941


namespace radian_measure_of_acute_angle_l195_195681

theorem radian_measure_of_acute_angle 
  (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2)
  (θ : ℝ) (S U : ℝ) 
  (hS : S = U * 9 / 14) (h_total_area : (π * r1^2) + (π * r2^2) + (π * r3^2) = S + U) :
  θ = 1827 * π / 3220 :=
by
  -- proof goes here
  sorry

end radian_measure_of_acute_angle_l195_195681


namespace min_passengers_on_vehicle_with_no_adjacent_seats_l195_195576

-- Define the seating arrangement and adjacency rules

structure Seat :=
(row : Fin 2) (col : Fin 5)

def adjacent (a b : Seat) : Prop :=
(a.row = b.row ∧ (a.col = b.col + 1 ∨ a.col + 1 = b.col)) ∨
(a.col = b.col ∧ (a.row = b.row + 1 ∨ a.row + 1 = b.row))

def valid_seating (seated : List Seat) : Prop :=
∀ (i j : Seat), i ∈ seated → j ∈ seated → adjacent i j → false

def min_passengers : ℕ :=
5

theorem min_passengers_on_vehicle_with_no_adjacent_seats :
∃ seated : List Seat, valid_seating seated ∧ List.length seated = min_passengers :=
sorry

end min_passengers_on_vehicle_with_no_adjacent_seats_l195_195576


namespace arithmetic_mean_is_ten_l195_195959

theorem arithmetic_mean_is_ten (a b x : ℝ) (h₁ : a = 4) (h₂ : b = 16) (h₃ : x = (a + b) / 2) : x = 10 :=
by
  sorry

end arithmetic_mean_is_ten_l195_195959


namespace gcf_lcm_15_l195_195087

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_15 : 
  GCF (LCM 9 15) (LCM 10 21) = 15 :=
by 
  sorry

end gcf_lcm_15_l195_195087


namespace probability_abs_diff_gt_half_is_7_over_16_l195_195884

noncomputable def probability_abs_diff_gt_half : ℚ :=
  let p_tail := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping tails
  let p_head := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping heads
  let p_x_tail_y_tail := p_tail * p_tail   -- Both first flips tails
  let p_x1_y_tail := p_head * p_tail / 2     -- x = 1, y flip tails
  let p_x_tail_y0 := p_tail * p_head / 2     -- x flip tails, y = 0
  let p_x1_y0 := p_head * p_head / 4         -- x = 1, y = 0
  -- Individual probabilities for x − y > 1/2
  let p_x_tail_y_tail_diff := (1 : ℚ) / (8 : ℚ) * p_x_tail_y_tail
  let p_x1_y_tail_diff := (1 : ℚ) / (2 : ℚ) * p_x1_y_tail
  let p_x_tail_y0_diff := (1 : ℚ) / (2 : ℚ) * p_x_tail_y0
  let p_x1_y0_diff := (1 : ℚ) * p_x1_y0
  -- Combined probability for x − y > 1/2
  let p_x_y_diff_gt_half := p_x_tail_y_tail_diff +
                            p_x1_y_tail_diff +
                            p_x_tail_y0_diff +
                            p_x1_y0_diff
  -- Final probability for |x − y| > 1/2 is twice of x − y > 1/2
  2 * p_x_y_diff_gt_half

theorem probability_abs_diff_gt_half_is_7_over_16 :
  probability_abs_diff_gt_half = (7 : ℚ) / 16 := 
  sorry

end probability_abs_diff_gt_half_is_7_over_16_l195_195884


namespace initial_players_round_robin_l195_195111

-- Definitions of conditions
def num_matches_round_robin (x : ℕ) : ℕ := x * (x - 1) / 2
def num_matches_after_drop_out (x : ℕ) : ℕ := num_matches_round_robin x - 2 * (x - 4) + 1

-- The theorem statement
theorem initial_players_round_robin (x : ℕ) 
  (two_players_dropped : num_matches_after_drop_out x = 84) 
  (round_robin_condition : num_matches_round_robin x - 2 * (x - 4) + 1 = 84 ∨ num_matches_round_robin x - 2 * (x - 4) = 84) :
  x = 15 :=
sorry

end initial_players_round_robin_l195_195111


namespace probability_of_specific_individual_drawn_on_third_attempt_l195_195155

theorem probability_of_specific_individual_drawn_on_third_attempt :
  let population_size := 6
  let sample_size := 3
  let prob_not_drawn_first_attempt := 5 / 6
  let prob_not_drawn_second_attempt := 4 / 5
  let prob_drawn_third_attempt := 1 / 4
  (prob_not_drawn_first_attempt * prob_not_drawn_second_attempt * prob_drawn_third_attempt) = 1 / 6 :=
by sorry

end probability_of_specific_individual_drawn_on_third_attempt_l195_195155


namespace triangle_angles_ratio_l195_195871

theorem triangle_angles_ratio (A B C : ℕ) 
  (hA : A = 20)
  (hB : B = 3 * A)
  (hSum : A + B + C = 180) :
  (C / A) = 5 := 
by
  sorry

end triangle_angles_ratio_l195_195871


namespace triangle_side_length_l195_195265

theorem triangle_side_length (A B C : ℝ) (h1 : AC = Real.sqrt 2) (h2: AB = 2)
  (h3 : (Real.sqrt 3 * Real.sin A + Real.cos A) / (Real.sqrt 3 * Real.cos A - Real.sin A) = Real.tan (5 * Real.pi / 12)) :
  BC = Real.sqrt 2 := 
sorry

end triangle_side_length_l195_195265


namespace line_eq_circle_eq_l195_195017

section
  variable (A B : ℝ × ℝ)
  variable (A_eq : A = (4, 6))
  variable (B_eq : B = (-2, 4))

  theorem line_eq : ∃ (a b c : ℝ), (a, b, c) = (1, -3, 14) ∧ ∀ x y, (y - 6) = ((4 - 6) / (-2 - 4)) * (x - 4) → a * x + b * y + c = 0 :=
  sorry

  theorem circle_eq : ∃ (h k r : ℝ), (h, k, r) = (1, 5, 10) ∧ ∀ x y, (x - 1)^2 + (y - 5)^2 = 10 :=
  sorry
end

end line_eq_circle_eq_l195_195017


namespace joan_has_10_books_l195_195549

def toms_books := 38
def together_books := 48
def joans_books := together_books - toms_books

theorem joan_has_10_books : joans_books = 10 :=
by
  -- The proof goes here, but we'll add "sorry" to indicate it's a placeholder.
  sorry

end joan_has_10_books_l195_195549


namespace maria_total_distance_in_miles_l195_195139

theorem maria_total_distance_in_miles :
  ∀ (steps_per_mile : ℕ) (full_cycles : ℕ) (remaining_steps : ℕ),
    steps_per_mile = 1500 →
    full_cycles = 50 →
    remaining_steps = 25000 →
    (100000 * full_cycles + remaining_steps) / steps_per_mile = 3350 := by
  intros
  sorry

end maria_total_distance_in_miles_l195_195139


namespace total_spent_l195_195800

variable (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ)

/-- Conditions from the problem setup --/
def conditions :=
  T_L = 40 ∧
  J_L = 0.5 * T_L ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = 0.25 * T_L ∧
  J_C = 3 * J_L ∧
  C_C = 0.5 * C_L ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = 0.5 * J_C

/-- Total spent by Lisa --/
def total_Lisa := T_L + J_L + C_L + S_L

/-- Total spent by Carly --/
def total_Carly := T_C + J_C + C_C + S_C + D_C + A_C

/-- Combined total spent by Lisa and Carly --/
theorem total_spent :
  conditions T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_Lisa T_L J_L C_L S_L + total_Carly T_C J_C C_C S_C D_C A_C = 520 :=
by
  sorry

end total_spent_l195_195800


namespace decreasing_even_function_condition_l195_195516

theorem decreasing_even_function_condition (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, x < y → y < 0 → f y < f x) 
    (h2 : ∀ x : ℝ, f (-x) = f x) : f 13 < f 9 ∧ f 9 < f 1 := 
by
  sorry

end decreasing_even_function_condition_l195_195516


namespace monotonic_decreasing_intervals_l195_195016

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem monotonic_decreasing_intervals : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) ∧
  (∀ x : ℝ, (1 < x ∧ x < Real.exp 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) :=
by
  sorry

end monotonic_decreasing_intervals_l195_195016


namespace smaller_number_l195_195398

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 := by
  sorry

end smaller_number_l195_195398


namespace combined_salary_ABC_and_E_l195_195822

def salary_D : ℕ := 7000
def avg_salary : ℕ := 9000
def num_individuals : ℕ := 5

theorem combined_salary_ABC_and_E :
  (avg_salary * num_individuals - salary_D) = 38000 :=
by
  -- proof goes here
  sorry

end combined_salary_ABC_and_E_l195_195822


namespace sqrt_a_add_4b_eq_pm3_l195_195589

theorem sqrt_a_add_4b_eq_pm3
  (a b : ℝ)
  (A_sol : a * (-1) + 5 * (-1) = 15)
  (B_sol : 4 * 5 - b * 2 = -2) :
  (a + 4 * b)^(1/2) = 3 ∨ (a + 4 * b)^(1/2) = -3 := by
  sorry

end sqrt_a_add_4b_eq_pm3_l195_195589


namespace smallest_n_lil_wayne_rain_l195_195233

noncomputable def probability_rain (n : ℕ) : ℝ := 
  1 / 2 - 1 / 2^(n + 1)

theorem smallest_n_lil_wayne_rain :
  ∃ n : ℕ, probability_rain n > 0.499 ∧ (∀ m : ℕ, m < n → probability_rain m ≤ 0.499) ∧ n = 9 := 
by
  sorry

end smallest_n_lil_wayne_rain_l195_195233


namespace arithmetic_sequence_sum_l195_195104

theorem arithmetic_sequence_sum (a d x y : ℤ) 
  (h1 : a = 3) (h2 : d = 5) 
  (h3 : x = a + d) 
  (h4 : y = x + d) 
  (h5 : y = 18) 
  (h6 : x = 13) : x + y = 31 := by
  sorry

end arithmetic_sequence_sum_l195_195104


namespace percentage_difference_l195_195575

variable (x y : ℝ)
variable (p : ℝ)  -- percentage by which x is less than y

theorem percentage_difference (h1 : y = x * 1.3333333333333333) : p = 25 :=
by
  sorry

end percentage_difference_l195_195575


namespace factorize_expression_l195_195280

theorem factorize_expression (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) :=
by 
  sorry

end factorize_expression_l195_195280


namespace find_a_and_b_solve_inequality_l195_195794

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem find_a_and_b (a b : ℝ) (h : ∀ x : ℝ, f x a b > 0 ↔ x < 0 ∨ x > 2) : a = -2 ∧ b = 0 :=
by sorry

theorem solve_inequality (a b : ℝ) (m : ℝ) (h1 : a = -2) (h2 : b = 0) :
  (∀ x : ℝ, f x a b < m^2 - 1 ↔ 
    (m = 0 → ∀ x : ℝ, false) ∧
    (m > 0 → (1 - m < x ∧ x < 1 + m)) ∧
    (m < 0 → (1 + m < x ∧ x < 1 - m))) :=
by sorry

end find_a_and_b_solve_inequality_l195_195794


namespace probability_of_multiple_6_or_8_l195_195506

def is_probability_of_multiple_6_or_8 (n : ℕ) : Prop := 
  let num_multiples (k : ℕ) := n / k
  let multiples_6 := num_multiples 6
  let multiples_8 := num_multiples 8
  let multiples_24 := num_multiples 24
  let total_multiples := multiples_6 + multiples_8 - multiples_24
  total_multiples / n = 1 / 4

theorem probability_of_multiple_6_or_8 : is_probability_of_multiple_6_or_8 72 :=
  by sorry

end probability_of_multiple_6_or_8_l195_195506


namespace gondor_laptop_earning_l195_195907

theorem gondor_laptop_earning :
  ∃ L : ℝ, (3 * 10 + 5 * 10 + 2 * L + 4 * L = 200) → L = 20 :=
by
  use 20
  sorry

end gondor_laptop_earning_l195_195907


namespace kevin_started_with_cards_l195_195167

-- The definitions corresponding to the conditions in the problem
def ended_with : Nat := 54
def found_cards : Nat := 47
def started_with (ended_with found_cards : Nat) : Nat := ended_with - found_cards

-- The Lean statement for the proof problem itself
theorem kevin_started_with_cards : started_with ended_with found_cards = 7 := by
  sorry

end kevin_started_with_cards_l195_195167


namespace f_is_even_l195_195202

noncomputable def f (x : ℝ) : ℝ := x ^ 2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := 
by
  intros x
  sorry

end f_is_even_l195_195202


namespace find_local_min_l195_195254

def z (x y : ℝ) : ℝ := x^2 + 2 * y^2 - 2 * x * y - x - 2 * y

theorem find_local_min: ∃ (x y : ℝ), x = 2 ∧ y = 3/2 ∧ ∀ ⦃h : ℝ⦄, h ≠ 0 → z (2 + h) (3/2 + h) > z 2 (3/2) :=
by
  sorry

end find_local_min_l195_195254


namespace bob_hair_growth_time_l195_195936

theorem bob_hair_growth_time (initial_length final_length growth_rate monthly_to_yearly_conversion : ℝ) 
  (initial_cut : initial_length = 6) 
  (current_length : final_length = 36) 
  (growth_per_month : growth_rate = 0.5) 
  (months_in_year : monthly_to_yearly_conversion = 12) : 
  (final_length - initial_length) / (growth_rate * monthly_to_yearly_conversion) = 5 :=
by
  sorry

end bob_hair_growth_time_l195_195936


namespace sunset_time_l195_195006

def length_of_daylight_in_minutes := 11 * 60 + 12
def sunrise_time_in_minutes := 6 * 60 + 45
def sunset_time_in_minutes := sunrise_time_in_minutes + length_of_daylight_in_minutes
def sunset_time_hour := sunset_time_in_minutes / 60
def sunset_time_minute := sunset_time_in_minutes % 60
def sunset_time_12hr_format := if sunset_time_hour >= 12 
    then (sunset_time_hour - 12, sunset_time_minute)
    else (sunset_time_hour, sunset_time_minute)

theorem sunset_time : sunset_time_12hr_format = (5, 57) :=
by
  sorry

end sunset_time_l195_195006


namespace minimal_d1_l195_195533

theorem minimal_d1 :
  (∃ (S3 S6 : ℕ), 
    ∃ (d1 : ℚ), 
      S3 = d1 + (d1 + 1) + (d1 + 2) ∧ 
      S6 = d1 + (d1 + 1) + (d1 + 2) + (d1 + 3) + (d1 + 4) + (d1 + 5) ∧ 
      d1 = (5 * S3 - S6) / 9 ∧ 
      d1 ≥ 1 / 2) → 
  ∃ (d1 : ℚ), d1 = 5 / 9 := 
by 
  sorry

end minimal_d1_l195_195533


namespace arithmetic_sum_s6_l195_195875

theorem arithmetic_sum_s6 (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) 
  (h1 : ∀ n, a (n+1) - a n = d)
  (h2 : a 1 = 2)
  (h3 : S 4 = 20)
  (hS : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) :
  S 6 = 42 :=
by sorry

end arithmetic_sum_s6_l195_195875


namespace composite_polynomial_l195_195995

-- Definition that checks whether a number is composite
def is_composite (a : ℕ) : Prop := ∃ (b c : ℕ), b > 1 ∧ c > 1 ∧ a = b * c

-- Problem translated into a Lean 4 statement
theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  is_composite (n ^ (5 * n - 1) + n ^ (5 * n - 2) + n ^ (5 * n - 3) + n + 1) :=
sorry

end composite_polynomial_l195_195995


namespace scientific_notation_of_0_0000205_l195_195918

noncomputable def scientific_notation (n : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_0_0000205 :
  scientific_notation 0.0000205 = (2.05, -5) :=
sorry

end scientific_notation_of_0_0000205_l195_195918


namespace strictly_increasing_interval_l195_195956

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

theorem strictly_increasing_interval :
  (∀ k : ℤ, ∀ x : ℝ, 
    (2 * k * Real.pi - 5 * Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6) 
    → (f x) < (f (x + 1))) :=
by 
  sorry

end strictly_increasing_interval_l195_195956


namespace find_x_for_which_ffx_eq_fx_l195_195177

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_for_which_ffx_eq_fx :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end find_x_for_which_ffx_eq_fx_l195_195177


namespace boys_skip_count_l195_195346

theorem boys_skip_count 
  (x y : ℕ)
  (avg_jumps_boys : ℕ := 85)
  (avg_jumps_girls : ℕ := 92)
  (avg_jumps_all : ℕ := 88)
  (h1 : x = y + 10)
  (h2 : (85 * x + 92 * y) / (x + y) = 88) : x = 40 :=
  sorry

end boys_skip_count_l195_195346


namespace whiteboards_per_class_is_10_l195_195214

-- Definitions from conditions
def classes : ℕ := 5
def ink_per_whiteboard_ml : ℕ := 20
def cost_per_ml_cents : ℕ := 50
def total_cost_cents : ℕ := 100 * 100  -- converting $100 to cents

-- Following the solution, define other useful constants
def cost_per_whiteboard_cents : ℕ := ink_per_whiteboard_ml * cost_per_ml_cents
def total_cost_all_classes_cents : ℕ := classes * total_cost_cents
def total_whiteboards : ℕ := total_cost_all_classes_cents / cost_per_whiteboard_cents
def whiteboards_per_class : ℕ := total_whiteboards / classes

-- We want to prove that each class uses 10 whiteboards.
theorem whiteboards_per_class_is_10 : whiteboards_per_class = 10 :=
  sorry

end whiteboards_per_class_is_10_l195_195214


namespace chris_age_l195_195548

theorem chris_age (a b c : ℤ) (h1 : a + b + c = 45) (h2 : c - 5 = a)
  (h3 : c + 4 = 3 * (b + 4) / 4) : c = 15 :=
by
  sorry

end chris_age_l195_195548


namespace arithmetic_progression_pairs_count_l195_195432

theorem arithmetic_progression_pairs_count (x y : ℝ) 
  (h1 : x = (15 + y) / 2)
  (h2 : x + x * y = 2 * y) : 
  (∃ x1 y1, x1 = (15 + y1) / 2 ∧ x1 + x1 * y1 = 2 * y1 ∧ x1 = (9 + 3 * Real.sqrt 7) / 2 ∧ y1 = -6 + 3 * Real.sqrt 7) ∨ 
  (∃ x2 y2, x2 = (15 + y2) / 2 ∧ x2 + x2 * y2 = 2 * y2 ∧ x2 = (9 - 3 * Real.sqrt 7) / 2 ∧ y2 = -6 - 3 * Real.sqrt 7) := 
sorry

end arithmetic_progression_pairs_count_l195_195432


namespace three_digit_number_count_l195_195129

def total_three_digit_numbers : ℕ := 900

def count_ABA : ℕ := 9 * 9  -- 81

def count_ABC : ℕ := 9 * 9 * 8  -- 648

def valid_three_digit_numbers : ℕ := total_three_digit_numbers - (count_ABA + count_ABC)

theorem three_digit_number_count :
  valid_three_digit_numbers = 171 := by
  sorry

end three_digit_number_count_l195_195129


namespace pounds_per_ton_l195_195315

theorem pounds_per_ton (weight_pounds : ℕ) (weight_tons : ℕ) (h_weight : weight_pounds = 6000) (h_tons : weight_tons = 3) : 
  weight_pounds / weight_tons = 2000 :=
by
  sorry

end pounds_per_ton_l195_195315


namespace intersection_set_eq_l195_195340

-- Define M
def M : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1 }

-- Define N
def N : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 / 4) + (p.2 / 3) = 1 }

-- Define the intersection of M and N
def M_intersection_N := { x : ℝ | -4 ≤ x ∧ x ≤ 4 }

-- The theorem to be proved
theorem intersection_set_eq : 
  { p : ℝ × ℝ | p ∈ M ∧ p ∈ N } = { p : ℝ × ℝ | p.1 ∈ M_intersection_N } :=
sorry

end intersection_set_eq_l195_195340


namespace find_a_l195_195200

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a : 
  ( ∀ a : ℝ, 
    (∀ x : ℝ,  0 ≤ x ∧ x ≤ 1 → f a 0 + f a 1 = a) → a = 1/2 ) :=
sorry

end find_a_l195_195200


namespace constant_speed_total_distance_l195_195678

def travel_time : ℝ := 5.5
def distance_per_hour : ℝ := 100
def speed := distance_per_hour

theorem constant_speed : ∀ t : ℝ, (1 ≤ t) ∧ (t ≤ travel_time) → speed = distance_per_hour := 
by sorry

theorem total_distance : speed * travel_time = 550 :=
by sorry

end constant_speed_total_distance_l195_195678


namespace quarters_range_difference_l195_195507

theorem quarters_range_difference (n d q : ℕ) (h1 : n + d + q = 150) (h2 : 5 * n + 10 * d + 25 * q = 2000) :
  let max_quarters := 0
  let min_quarters := 62
  (max_quarters - min_quarters) = 62 :=
by
  let max_quarters := 0
  let min_quarters := 62
  sorry

end quarters_range_difference_l195_195507


namespace work_completed_in_initial_days_l195_195511

theorem work_completed_in_initial_days (x : ℕ) : 
  (100 * x = 50 * 40) → x = 20 :=
by
  sorry

end work_completed_in_initial_days_l195_195511


namespace odd_function_f1_eq_4_l195_195022

theorem odd_function_f1_eq_4 (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x < 0 → f x = x^2 + a * x)
  (h3 : f 2 = 6) : 
  f 1 = 4 :=
by sorry

end odd_function_f1_eq_4_l195_195022


namespace band_member_share_l195_195345

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l195_195345


namespace product_of_two_integers_l195_195007

def gcd_lcm_prod (x y : ℕ) :=
  Nat.gcd x y = 8 ∧ Nat.lcm x y = 48

theorem product_of_two_integers (x y : ℕ) (h : gcd_lcm_prod x y) : x * y = 384 :=
by
  sorry

end product_of_two_integers_l195_195007


namespace perimeter_of_garden_l195_195925

def area (length width : ℕ) : ℕ := length * width

def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem perimeter_of_garden :
  ∀ (l w : ℕ), area l w = 28 ∧ l = 7 → perimeter l w = 22 := by
  sorry

end perimeter_of_garden_l195_195925


namespace john_replace_bedroom_doors_l195_195882

variable (B O : ℕ)
variable (cost_outside cost_bedroom total_cost : ℕ)

def john_has_to_replace_bedroom_doors : Prop :=
  let outside_doors_replaced := 2
  let cost_of_outside_door := 20
  let cost_of_bedroom_door := 10
  let total_replacement_cost := 70
  O = outside_doors_replaced ∧
  cost_outside = cost_of_outside_door ∧
  cost_bedroom = cost_of_bedroom_door ∧
  total_cost = total_replacement_cost ∧
  20 * O + 10 * B = total_cost →
  B = 3

theorem john_replace_bedroom_doors : john_has_to_replace_bedroom_doors B O cost_outside cost_bedroom total_cost :=
sorry

end john_replace_bedroom_doors_l195_195882


namespace arithmetic_sequence_formula_geometric_sequence_sum_l195_195366

variables {a_n S_n b_n T_n : ℕ → ℚ} {a_3 S_3 a_5 b_3 T_3 : ℚ} {q : ℚ}

def is_arithmetic_sequence (a_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, a_n n = a_1 + (n - 1) * d

def sum_first_n_arithmetic (S_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

def is_geometric_sequence (b_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, b_n n = b_1 * q^(n-1)

def sum_first_n_geometric (T_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, T_n n = if q = 1 then n * b_1 else b_1 * (1 - q^n) / (1 - q)

theorem arithmetic_sequence_formula {a_1 d : ℚ} (h_arith : is_arithmetic_sequence a_n a_1 d)
    (h_sum : sum_first_n_arithmetic S_n a_1 d) (h1 : a_n 3 = 5) (h2 : S_n 3 = 9) :
    ∀ n, a_n n = 2 * n - 1 := sorry

theorem geometric_sequence_sum {b_1 : ℚ} (h_geom : is_geometric_sequence b_n b_1 q)
    (h_sum : sum_first_n_geometric T_n b_1 q) (h3 : q > 0) (h4 : b_n 3 = a_n 5) (h5 : T_n 3 = 13) :
    ∀ n, T_n n = (3^n - 1) / 2 := sorry

end arithmetic_sequence_formula_geometric_sequence_sum_l195_195366


namespace tank_capacity_l195_195897

-- Define the conditions given in the problem.
def tank_full_capacity (x : ℝ) : Prop :=
  (0.25 * x = 60) ∧ (0.15 * x = 36)

-- State the theorem that needs to be proved.
theorem tank_capacity : ∃ x : ℝ, tank_full_capacity x ∧ x = 240 := 
by 
  sorry

end tank_capacity_l195_195897


namespace no_square_number_divisible_by_six_between_50_and_120_l195_195380

theorem no_square_number_divisible_by_six_between_50_and_120 :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n * n) ∧ (x % 6 = 0) ∧ (50 < x ∧ x < 120) := 
sorry

end no_square_number_divisible_by_six_between_50_and_120_l195_195380


namespace pq_sub_l195_195082

-- Assuming the conditions
theorem pq_sub (p q : ℚ) 
  (h₁ : 3 / p = 4) 
  (h₂ : 3 / q = 18) : 
  p - q = 7 / 12 := 
  sorry

end pq_sub_l195_195082


namespace perfect_square_n_l195_195226

open Nat

theorem perfect_square_n (n : ℕ) : 
  (∃ k : ℕ, 2 ^ (n + 1) * n = k ^ 2) ↔ 
  (∃ m : ℕ, n = 2 * m ^ 2) ∨ (∃ odd_k : ℕ, n = odd_k ^ 2 ∧ odd_k % 2 = 1) := 
sorry

end perfect_square_n_l195_195226


namespace find_c_and_d_l195_195124

theorem find_c_and_d :
  ∀ (y c d : ℝ), (y^2 - 5 * y + 5 / y + 1 / (y^2) = 17) ∧ (y = c - Real.sqrt d) ∧ (0 < c) ∧ (0 < d) → (c + d = 106) :=
by
  intros y c d h
  sorry

end find_c_and_d_l195_195124


namespace equilateral_triangle_area_decrease_l195_195480

theorem equilateral_triangle_area_decrease :
  let original_area : ℝ := 100 * Real.sqrt 3
  let side_length_s := 20
  let decreased_side_length := side_length_s - 6
  let new_area := (decreased_side_length * decreased_side_length * Real.sqrt 3) / 4
  let decrease_in_area := original_area - new_area
  decrease_in_area = 51 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_decrease_l195_195480


namespace coefficients_sum_l195_195223

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

theorem coefficients_sum : 
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  ((2 * x - 1) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) 
  ∧ (a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8) :=
sorry

end coefficients_sum_l195_195223


namespace farm_cows_l195_195026

theorem farm_cows (x y : ℕ) (h : 4 * x + 2 * y = 20 + 3 * (x + y)) : x = 20 + y :=
sorry

end farm_cows_l195_195026


namespace solve_for_x_l195_195643

theorem solve_for_x (x : ℤ) (h : 13 * x + 14 * x + 17 * x + 11 = 143) : x = 3 :=
by sorry

end solve_for_x_l195_195643


namespace option_d_l195_195970

variable {R : Type*} [LinearOrderedField R]

theorem option_d (a b c d : R) (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by 
  sorry

end option_d_l195_195970


namespace second_year_students_sampled_l195_195101

def total_students (f s t : ℕ) : ℕ := f + s + t

def proportion_second_year (s total_stu : ℕ) : ℚ := s / total_stu

def sampled_second_year_students (p : ℚ) (n : ℕ) : ℚ := p * n

theorem second_year_students_sampled
  (f s t : ℕ) (n : ℕ)
  (h1 : f = 600)
  (h2 : s = 780)
  (h3 : t = 720)
  (h4 : n = 35) :
  sampled_second_year_students (proportion_second_year s (total_students f s t)) n = 13 := 
sorry

end second_year_students_sampled_l195_195101


namespace height_after_16_minutes_l195_195224

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  8 * Real.sin ((Real.pi / 6) * t - Real.pi / 2) + 10

theorem height_after_16_minutes : ferris_wheel_height 16 = 6 := by
  sorry

end height_after_16_minutes_l195_195224


namespace prime_square_sum_eq_square_iff_l195_195567

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l195_195567


namespace find_multiple_of_q_l195_195568

-- Definitions of x and y
def x (k q : ℤ) : ℤ := 55 + k * q
def y (q : ℤ) : ℤ := 4 * q + 41

-- The proof statement
theorem find_multiple_of_q (k : ℤ) : x k 7 = y 7 → k = 2 := by
  sorry

end find_multiple_of_q_l195_195568


namespace final_value_of_S_l195_195672

theorem final_value_of_S :
  ∀ (S n : ℕ), S = 1 → n = 1 →
  (∀ S n : ℕ, ¬ n > 3 → 
    (∃ S' n' : ℕ, S' = S + 2 * n ∧ n' = n + 1 ∧ 
      (∀ S n : ℕ, n > 3 → S' = 13))) :=
by 
  intros S n hS hn
  simp [hS, hn]
  sorry

end final_value_of_S_l195_195672


namespace probability_of_different_colors_is_correct_l195_195512

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end probability_of_different_colors_is_correct_l195_195512


namespace Milly_study_time_l195_195935

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end Milly_study_time_l195_195935


namespace crate_stacking_probability_l195_195086

theorem crate_stacking_probability :
  ∃ (p q : ℕ), (p.gcd q = 1) ∧ (p : ℚ) / q = 170 / 6561 ∧ (total_height = 50) ∧ (number_of_crates = 12) ∧ (orientation_probability = 1 / 3) :=
sorry

end crate_stacking_probability_l195_195086


namespace fractional_equation_solution_l195_195665

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l195_195665


namespace ratio_of_cats_l195_195748

-- Definitions from conditions
def total_animals_anthony := 12
def fraction_cats_anthony := 2 / 3
def extra_dogs_leonel := 7
def total_animals_both := 27

-- Calculate number of cats and dogs Anthony has
def cats_anthony := fraction_cats_anthony * total_animals_anthony
def dogs_anthony := total_animals_anthony - cats_anthony

-- Calculate number of dogs Leonel has
def dogs_leonel := dogs_anthony + extra_dogs_leonel

-- Calculate number of cats Leonel has
def cats_leonel := total_animals_both - (cats_anthony + dogs_anthony + dogs_leonel)

-- Prove the desired ratio
theorem ratio_of_cats : (cats_leonel / cats_anthony) = (1 / 2) := by
  -- Insert proof steps here
  sorry

end ratio_of_cats_l195_195748


namespace trees_died_in_typhoon_imply_all_died_l195_195561

-- Given conditions
def trees_initial := 3
def survived_trees (x : Int) := x
def died_trees (x : Int) := x + 23

-- Prove that the number of died trees is 3
theorem trees_died_in_typhoon_imply_all_died : ∀ x, 2 * survived_trees x + 23 = trees_initial → trees_initial = died_trees x := 
by
  intro x h
  sorry

end trees_died_in_typhoon_imply_all_died_l195_195561


namespace inequality_solution_l195_195268

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ (x > 8)) ↔
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) :=
sorry

end inequality_solution_l195_195268


namespace median_squared_formula_l195_195108

theorem median_squared_formula (a b c m : ℝ) (AC_is_median : 2 * m^2 + c^2 = a^2 + b^2) : 
  m^2 = (1/4) * (2 * a^2 + 2 * b^2 - c^2) := 
by
  sorry

end median_squared_formula_l195_195108


namespace inequality_l195_195332

variable {a b c : ℝ}

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := 
by 
  sorry

end inequality_l195_195332


namespace percentage_off_at_sale_l195_195921

theorem percentage_off_at_sale
  (sale_price original_price : ℝ)
  (h1 : sale_price = 140)
  (h2 : original_price = 350) :
  (original_price - sale_price) / original_price * 100 = 60 :=
by
  sorry

end percentage_off_at_sale_l195_195921


namespace percent_workday_in_meetings_l195_195039

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 3 * first_meeting_duration
def third_meeting_duration : ℕ := 2 * second_meeting_duration
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration
def workday_duration : ℕ := 10 * 60

theorem percent_workday_in_meetings : (total_meeting_time : ℚ) / workday_duration * 100 = 50 := by
  sorry

end percent_workday_in_meetings_l195_195039


namespace parabola_standard_equation_l195_195890

theorem parabola_standard_equation (x y : ℝ) : 
  (3 * x - 4 * y - 12 = 0) →
  (y = 0 → x = 4 ∨ y = -3 → x = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
by
  intros h_line h_intersect
  sorry

end parabola_standard_equation_l195_195890


namespace eqn_abs_3x_minus_2_solution_l195_195172

theorem eqn_abs_3x_minus_2_solution (x : ℝ) :
  (|x + 5| = 3 * x - 2) ↔ x = 7 / 2 :=
by
  sorry

end eqn_abs_3x_minus_2_solution_l195_195172


namespace heat_released_is_1824_l195_195641

def ΔH_f_NH3 : ℝ := -46  -- Enthalpy of formation of NH3 in kJ/mol
def ΔH_f_H2SO4 : ℝ := -814  -- Enthalpy of formation of H2SO4 in kJ/mol
def ΔH_f_NH4SO4 : ℝ := -909  -- Enthalpy of formation of (NH4)2SO4 in kJ/mol

def ΔH_rxn : ℝ :=
  2 * ΔH_f_NH4SO4 - (2 * ΔH_f_NH3 + ΔH_f_H2SO4)  -- Reaction enthalpy change

def heat_released : ℝ := 2 * ΔH_rxn  -- Heat released for 4 moles of NH3

theorem heat_released_is_1824 : heat_released = -1824 :=
by
  -- Theorem statement for proving heat released is 1824 kJ
  sorry

end heat_released_is_1824_l195_195641


namespace part_one_part_two_l195_195216

-- Part (1)
theorem part_one (x : ℝ) : x - (3 * x - 1) ≤ 2 * x + 3 → x ≥ -1 / 2 :=
by sorry

-- Part (2)
theorem part_two (x : ℝ) : 
  (3 * (x - 1) < 4 * x - 2) ∧ ((1 + 4 * x) / 3 > x - 1) → x > -1 :=
by sorry

end part_one_part_two_l195_195216


namespace sum_first_five_terms_l195_195826

theorem sum_first_five_terms (a1 a2 a3 : ℝ) (S5 : ℝ) 
  (h1 : a1 * a3 = 8 * a2)
  (h2 : (a1 + a2) = 24) :
  S5 = 31 :=
sorry

end sum_first_five_terms_l195_195826


namespace ratio_garbage_zane_dewei_l195_195811

-- Define the weights of garbage picked up by Daliah, Dewei, and Zane.
def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah_garbage - 2
def zane_garbage : ℝ := 62

-- The theorem that we need to prove
theorem ratio_garbage_zane_dewei : zane_garbage / dewei_garbage = 4 :=
by
  sorry

end ratio_garbage_zane_dewei_l195_195811


namespace function_d_is_odd_l195_195201

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Given function
def f (x : ℝ) : ℝ := x^3

-- Proof statement
theorem function_d_is_odd : is_odd_function f := 
by sorry

end function_d_is_odd_l195_195201


namespace sum_of_vars_l195_195433

variables (x y z w : ℤ)

theorem sum_of_vars (h1 : x - y + z = 7)
                    (h2 : y - z + w = 8)
                    (h3 : z - w + x = 4)
                    (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
by
  sorry

end sum_of_vars_l195_195433


namespace exist_polynomials_unique_polynomials_l195_195083

-- Problem statement: the function 'f'
variable (f : ℝ → ℝ → ℝ → ℝ)

-- Condition: f(w, w, w) = 0 for all w ∈ ℝ
axiom f_ww_ww_ww (w : ℝ) : f w w w = 0

-- Statement for existence of A, B, C
theorem exist_polynomials (f : ℝ → ℝ → ℝ → ℝ)
  (hf : ∀ w : ℝ, f w w w = 0) : 
  ∃ A B C : ℝ → ℝ → ℝ → ℝ, 
  (∀ w : ℝ, A w w w + B w w w + C w w w = 0) ∧ 
  ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x) :=
sorry

-- Statement for uniqueness of A, B, C
theorem unique_polynomials (f : ℝ → ℝ → ℝ → ℝ) 
  (A B C A' B' C' : ℝ → ℝ → ℝ → ℝ)
  (hf: ∀ w : ℝ, f w w w = 0)
  (h1 : ∀ w : ℝ, A w w w + B w w w + C w w w = 0)
  (h2 : ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x))
  (h3 : ∀ w : ℝ, A' w w w + B' w w w + C' w w w = 0)
  (h4 : ∀ x y z : ℝ, f x y z = A' x y z * (x - y) + B' x y z * (y - z) + C' x y z * (z - x)) : 
  A = A' ∧ B = B' ∧ C = C' :=
sorry

end exist_polynomials_unique_polynomials_l195_195083


namespace cost_of_items_l195_195283

namespace GardenCost

variables (B T C : ℝ)

/-- Given conditions defining the cost relationships and combined cost,
prove the specific costs of bench, table, and chair. -/
theorem cost_of_items
  (h1 : T + B + C = 650)
  (h2 : T = 2 * B - 50)
  (h3 : C = 1.5 * B - 25) :
  B = 161.11 ∧ T = 272.22 ∧ C = 216.67 :=
sorry

end GardenCost

end cost_of_items_l195_195283


namespace jill_tax_on_other_items_l195_195542

noncomputable def tax_on_other_items (total_spent clothing_tax_percent total_tax_percent : ℝ) : ℝ :=
  let clothing_spent := 0.5 * total_spent
  let food_spent := 0.25 * total_spent
  let other_spent := 0.25 * total_spent
  let clothing_tax := clothing_tax_percent * clothing_spent
  let total_tax := total_tax_percent * total_spent
  let tax_on_others := total_tax - clothing_tax
  (tax_on_others / other_spent) * 100

theorem jill_tax_on_other_items :
  let total_spent := 100
  let clothing_tax_percent := 0.1
  let total_tax_percent := 0.1
  tax_on_other_items total_spent clothing_tax_percent total_tax_percent = 20 := by
  sorry

end jill_tax_on_other_items_l195_195542


namespace m_plus_n_in_right_triangle_l195_195485

noncomputable def triangle (A B C : Point) : Prop :=
  ∃ (BD : ℕ) (x : ℕ) (y : ℕ),
  ∃ (AB BC AC : ℕ),
  ∃ (m n : ℕ),
  B ≠ C ∧
  C ≠ A ∧
  B ≠ A ∧
  m.gcd n = 1 ∧
  BD = 17^3 ∧
  BC = 17^2 * x ∧
  AB = 17 * x^2 ∧
  AC = 17 * x * y ∧
  BC^2 + AC^2 = AB^2 ∧
  (2 * 17 * x) = 17^2 ∧
  ∃ cB, cB = (BC : ℚ) / (AB : ℚ) ∧
  cB = (m : ℚ) / (n : ℚ)

theorem m_plus_n_in_right_triangle :
  ∀ (A B C : Point),
  A ≠ B ∧
  B ≠ C ∧
  C ≠ A ∧
  triangle A B C →
  ∃ m n : ℕ, m.gcd n = 1 ∧ m + n = 162 :=
sorry

end m_plus_n_in_right_triangle_l195_195485


namespace add_base8_l195_195849

theorem add_base8 : 
  let a := 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  let b := 5 * 8^2 + 7 * 8^1 + 3 * 8^0
  let c := 6 * 8^1 + 2 * 8^0
  let sum := a + b + c
  sum = 1 * 8^3 + 1 * 8^2 + 2 * 8^1 + 3 * 8^0 :=
by
  -- Proof skipped
  sorry

end add_base8_l195_195849


namespace proof_intersection_complement_l195_195367

open Set

variable (U : Set ℝ) (A B : Set ℝ)

theorem proof_intersection_complement:
  U = univ ∧ A = {x | -1 < x ∧ x ≤ 5} ∧ B = {x | x < 2} →
  A ∩ (U \ B) = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  intros h
  rcases h with ⟨hU, hA, hB⟩
  simp [hU, hA, hB]
  sorry

end proof_intersection_complement_l195_195367


namespace problem_l195_195342

theorem problem
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 320) :
  x * y = 64 ∧ x^3 + y^3 = 4160 :=
by
  sorry

end problem_l195_195342


namespace initial_amount_is_3_l195_195968

-- Define the initial amount of water in the bucket
def initial_water_amount (total water_added : ℝ) : ℝ :=
  total - water_added

-- Define the variables
def total : ℝ := 9.8
def water_added : ℝ := 6.8

-- State the problem
theorem initial_amount_is_3 : initial_water_amount total water_added = 3 := 
  by
    sorry

end initial_amount_is_3_l195_195968


namespace ann_has_30_more_cards_than_anton_l195_195867

theorem ann_has_30_more_cards_than_anton (heike_cards : ℕ) (anton_cards : ℕ) (ann_cards : ℕ) 
  (h1 : anton_cards = 3 * heike_cards)
  (h2 : ann_cards = 6 * heike_cards)
  (h3 : ann_cards = 60) : ann_cards - anton_cards = 30 :=
by
  sorry

end ann_has_30_more_cards_than_anton_l195_195867


namespace smallest_b_factors_l195_195664

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end smallest_b_factors_l195_195664


namespace bankers_discount_l195_195504

/-- Given the present worth (P) of Rs. 400 and the true discount (TD) of Rs. 20,
Prove that the banker's discount (BD) is Rs. 21. -/
theorem bankers_discount (P TD FV BD : ℝ) (hP : P = 400) (hTD : TD = 20) 
(hFV : FV = P + TD) (hBD : BD = (TD * FV) / P) : BD = 21 := 
by
  sorry

end bankers_discount_l195_195504


namespace remainder_sum_modulo_eleven_l195_195684

theorem remainder_sum_modulo_eleven :
  (88132 + 88133 + 88134 + 88135 + 88136 + 88137 + 88138 + 88139 + 88140 + 88141) % 11 = 1 :=
by
  sorry

end remainder_sum_modulo_eleven_l195_195684


namespace perfect_square_expression_l195_195552

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l195_195552


namespace distance_between_intersections_l195_195231

open Classical
open Real

noncomputable def curve1 (x y : ℝ) : Prop := y^2 = x
noncomputable def curve2 (x y : ℝ) : Prop := x + 2 * y = 10

theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ),
    (curve1 p1.1 p1.2) ∧ (curve2 p1.1 p1.2) ∧
    (curve1 p2.1 p2.2) ∧ (curve2 p2.1 p2.2) ∧
    (dist p1 p2 = 2 * sqrt 55) :=
by
  sorry

end distance_between_intersections_l195_195231


namespace inequality_abc_l195_195843

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l195_195843


namespace probability_same_color_is_117_200_l195_195866

/-- There are eight green balls, five red balls, and seven blue balls in a bag. 
    A ball is taken from the bag, its color recorded, then placed back in the bag.
    A second ball is taken and its color recorded. -/
def probability_two_balls_same_color : ℚ :=
  let pGreen := (8 : ℚ) / 20
  let pRed := (5 : ℚ) / 20
  let pBlue := (7 : ℚ) / 20
  pGreen^2 + pRed^2 + pBlue^2

theorem probability_same_color_is_117_200 : probability_two_balls_same_color = 117 / 200 := by
  sorry

end probability_same_color_is_117_200_l195_195866


namespace cube_mono_increasing_l195_195143

theorem cube_mono_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

end cube_mono_increasing_l195_195143


namespace neg_p_equiv_exists_leq_l195_195963

-- Define the given proposition p
def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- State the equivalence we need to prove
theorem neg_p_equiv_exists_leq :
  ¬ p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by {
  sorry  -- Proof is skipped as per instructions
}

end neg_p_equiv_exists_leq_l195_195963


namespace eve_age_l195_195653

variable (E : ℕ)

theorem eve_age (h1 : ∀ (a : ℕ), a = 9 → (E + 1) = 3 * (9 - 4)) : E = 14 := 
by
  have h2 : 9 - 4 = 5 := by norm_num
  have h3 : 3 * 5 = 15 := by norm_num
  have h4 : (E + 1) = 15 := h1 9 rfl
  linarith

end eve_age_l195_195653


namespace intersect_x_axis_once_l195_195329

theorem intersect_x_axis_once (k : ℝ) : 
  (∀ x : ℝ, (k - 3) * x^2 + 2 * x + 1 = 0 → x = 0) → (k = 3 ∨ k = 4) :=
by
  intro h
  sorry

end intersect_x_axis_once_l195_195329


namespace days_worked_per_week_l195_195634

theorem days_worked_per_week (total_toys_per_week toys_produced_each_day : ℕ) 
  (h1 : total_toys_per_week = 5505)
  (h2 : toys_produced_each_day = 1101)
  : total_toys_per_week / toys_produced_each_day = 5 :=
  by
    sorry

end days_worked_per_week_l195_195634


namespace minimum_value_of_f_at_zero_inequality_f_geq_term_l195_195350

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem minimum_value_of_f_at_zero (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≥ f a x ∧ f a x = 0) → a = 2 :=
by
  sorry

theorem inequality_f_geq_term (x : ℝ) (hx : x > 1) : 
  f 2 x ≥ 1 / x - Real.exp (1 - x) :=
by
  sorry

end minimum_value_of_f_at_zero_inequality_f_geq_term_l195_195350


namespace trigonometric_expression_evaluation_l195_195386

theorem trigonometric_expression_evaluation :
  (Real.cos (-585 * Real.pi / 180)) / 
  (Real.tan (495 * Real.pi / 180) + Real.sin (-690 * Real.pi / 180)) = Real.sqrt 2 :=
  sorry

end trigonometric_expression_evaluation_l195_195386


namespace find_n_l195_195149

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem find_n (n : ℤ) (h : ∃ x, n < x ∧ x < n+1 ∧ f x = 0) : n = 2 :=
sorry

end find_n_l195_195149


namespace a_oxen_count_l195_195555

-- Define the conditions from the problem
def total_rent : ℝ := 210
def c_share_rent : ℝ := 54
def oxen_b : ℝ := 12
def oxen_c : ℝ := 15
def months_b : ℝ := 5
def months_c : ℝ := 3
def months_a : ℝ := 7
def oxen_c_months : ℝ := oxen_c * months_c
def total_ox_months (oxen_a : ℝ) : ℝ := (oxen_a * months_a) + (oxen_b * months_b) + oxen_c_months

-- The theorem we want to prove
theorem a_oxen_count (oxen_a : ℝ) (h : c_share_rent / total_rent = oxen_c_months / total_ox_months oxen_a) :
  oxen_a = 10 := by sorry

end a_oxen_count_l195_195555


namespace initial_paper_count_l195_195232

theorem initial_paper_count (used left initial : ℕ) (h_used : used = 156) (h_left : left = 744) :
  initial = used + left :=
sorry

end initial_paper_count_l195_195232


namespace total_students_is_30_l195_195726

def students_per_bed : ℕ := 2 

def beds_per_room : ℕ := 2 

def students_per_couch : ℕ := 1 

def rooms_booked : ℕ := 6 

def total_students := (students_per_bed * beds_per_room + students_per_couch) * rooms_booked

theorem total_students_is_30 : total_students = 30 := by
  sorry

end total_students_is_30_l195_195726


namespace hexagon_largest_angle_l195_195188

theorem hexagon_largest_angle (x : ℝ) 
  (h_angles_sum : 80 + 100 + x + x + x + (2 * x + 20) = 720) : 
  (2 * x + 20) = 228 :=
by 
  sorry

end hexagon_largest_angle_l195_195188


namespace find_p_l195_195846

theorem find_p (A B C p q r s : ℝ) (h₀ : A ≠ 0)
  (h₁ : r + s = -B / A)
  (h₂ : r * s = C / A)
  (h₃ : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C + 2 * A^2 * C^2) / A^3 :=
sorry

end find_p_l195_195846


namespace problem_statement_l195_195457

variable (a b c : ℤ) -- Declare variables as integers

-- Define conditions based on the problem
def smallest_natural_number (a : ℤ) := a = 1
def largest_negative_integer (b : ℤ) := b = -1
def number_equal_to_its_opposite (c : ℤ) := c = 0

-- State the theorem
theorem problem_statement (h1 : smallest_natural_number a) 
                         (h2 : largest_negative_integer b) 
                         (h3 : number_equal_to_its_opposite c) : 
  a + b + c = 0 := 
  by 
    rw [h1, h2, h3] 
    simp

end problem_statement_l195_195457


namespace solve_fractional_equation_l195_195791

theorem solve_fractional_equation
  (x : ℝ)
  (h1 : x ≠ 0)
  (h2 : x ≠ 2)
  (h_eq : 2 / x - 1 / (x - 2) = 0) : 
  x = 4 := by
  sorry

end solve_fractional_equation_l195_195791


namespace largest_log_value_l195_195421

theorem largest_log_value :
  ∃ (x y z t : ℝ) (a b c : ℝ),
    x ≤ y ∧ y ≤ z ∧ z ≤ t ∧
    a = Real.log y / Real.log x ∧
    b = Real.log z / Real.log y ∧
    c = Real.log t / Real.log z ∧
    a = 15 ∧ b = 20 ∧ c = 21 ∧
    (∃ u v w, u = a * b ∧ v = b * c ∧ w = a * b * c ∧ w = 420) := sorry

end largest_log_value_l195_195421


namespace swimming_speed_eq_l195_195228

theorem swimming_speed_eq (S R H : ℝ) (h1 : R = 9) (h2 : H = 5) (h3 : H = (2 * S * R) / (S + R)) :
  S = 45 / 13 :=
by
  sorry

end swimming_speed_eq_l195_195228


namespace smallest_gcd_for_system_l195_195821

theorem smallest_gcd_for_system :
  ∃ n : ℕ, n > 0 ∧ 
    (∀ a b c : ℤ,
     gcd (gcd a b) c = n →
     ∃ x y z : ℤ, 
       (x + 2*y + 3*z = a) ∧ 
       (2*x + y - 2*z = b) ∧ 
       (3*x + y + 5*z = c)) ∧ 
  n = 28 :=
sorry

end smallest_gcd_for_system_l195_195821


namespace nico_reads_wednesday_l195_195372

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_l195_195372


namespace water_volume_per_minute_l195_195242

theorem water_volume_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ) 
  (H_depth : depth = 5) 
  (H_width : width = 35) 
  (H_flow_rate_kmph : flow_rate_kmph = 2) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 5832.75 :=
by
  sorry

end water_volume_per_minute_l195_195242


namespace penalty_kicks_calculation_l195_195967

def totalPlayers := 24
def goalkeepers := 4
def nonGoalkeeperShootsAgainstOneGoalkeeper := totalPlayers - 1
def totalPenaltyKicks := goalkeepers * nonGoalkeeperShootsAgainstOneGoalkeeper

theorem penalty_kicks_calculation : totalPenaltyKicks = 92 := by
  sorry

end penalty_kicks_calculation_l195_195967


namespace original_number_l195_195397

theorem original_number (x : ℝ) (h : 1.35 * x = 935) : x = 693 := by
  sorry

end original_number_l195_195397


namespace find_triplet_solution_l195_195357

theorem find_triplet_solution (m n x y : ℕ) (hm : 0 < m) (hcoprime : Nat.gcd m n = 1) 
 (heq : (x^2 + y^2)^m = (x * y)^n) : 
  ∃ a : ℕ, x = 2^a ∧ y = 2^a ∧ n = m + 1 :=
by sorry

end find_triplet_solution_l195_195357


namespace problem_solution_l195_195879

theorem problem_solution :
  (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := 
  by 
  sorry

end problem_solution_l195_195879


namespace triangle_angle_contradiction_l195_195824

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ α > 60 ∧ β > 60 ∧ γ > 60 → False :=
by
  sorry

end triangle_angle_contradiction_l195_195824


namespace proof_problem_l195_195538

variable {a b m n x : ℝ}

theorem proof_problem (h1 : a = -b) (h2 : m * n = 1) (h3 : m ≠ n) (h4 : |x| = 2) :
    (-2 * m * n + (b + a) / (m - n) - x = -4 ∧ x = 2) ∨
    (-2 * m * n + (b + a) / (m - n) - x = 0 ∧ x = -2) :=
by
  sorry

end proof_problem_l195_195538


namespace negation_at_most_three_l195_195982

theorem negation_at_most_three :
  ¬ (∀ n : ℕ, n ≤ 3) ↔ (∃ n : ℕ, n ≥ 4) :=
by
  sorry

end negation_at_most_three_l195_195982


namespace fraction_product_l195_195868

theorem fraction_product : (1 / 2) * (1 / 3) * (1 / 6) * 120 = 10 / 3 :=
by
  sorry

end fraction_product_l195_195868


namespace triangle_inequality_l195_195815

variable {x y z : ℝ}
variable {A B C : ℝ}

theorem triangle_inequality (hA: A > 0) (hB : B > 0) (hC : C > 0) (h_sum : A + B + C = π):
  x^2 + y^2 + z^2 ≥ 2 * y * z * Real.sin A + 2 * z * x * Real.sin B - 2 * x * y * Real.cos C := by
  sorry

end triangle_inequality_l195_195815


namespace arccos_one_half_eq_pi_div_three_l195_195141

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end arccos_one_half_eq_pi_div_three_l195_195141


namespace ray_initial_cents_l195_195081

theorem ray_initial_cents :
  ∀ (initial_cents : ℕ), 
    (∃ (peter_cents : ℕ), 
      peter_cents = 30 ∧
      ∃ (randi_cents : ℕ),
        randi_cents = 2 * peter_cents ∧
        randi_cents = peter_cents + 60 ∧
        peter_cents + randi_cents = initial_cents
    ) →
    initial_cents = 90 := 
by
    intros initial_cents h
    obtain ⟨peter_cents, hp, ⟨randi_cents, hr1, hr2, hr3⟩⟩ := h
    sorry

end ray_initial_cents_l195_195081


namespace find_other_number_l195_195248

theorem find_other_number (A B : ℕ) (HCF LCM : ℕ)
  (hA : A = 24)
  (hHCF: (HCF : ℚ) = 16)
  (hLCM: (LCM : ℚ) = 312)
  (hHCF_LCM: HCF * LCM = A * B) : 
  B = 208 :=
by
  sorry

end find_other_number_l195_195248


namespace original_amount_of_money_l195_195894

variable (took : ℕ) (now : ℕ) (initial : ℕ)

-- conditions from the problem
def conditions := (took = 2) ∧ (now = 3)

-- the statement to prove
theorem original_amount_of_money {took now initial : ℕ} (h : conditions took now) :
  initial = now + took ↔ initial = 5 :=
by {
  sorry
}

end original_amount_of_money_l195_195894


namespace boys_without_glasses_l195_195414

def total_students_with_glasses : ℕ := 36
def girls_with_glasses : ℕ := 21
def total_boys : ℕ := 30

theorem boys_without_glasses :
  total_boys - (total_students_with_glasses - girls_with_glasses) = 15 :=
by
  sorry

end boys_without_glasses_l195_195414


namespace perfect_square_a_value_l195_195079

theorem perfect_square_a_value (x y a : ℝ) :
  (∃ k : ℝ, x^2 + 2 * x * y + y^2 - a * (x + y) + 25 = k^2) →
  a = 10 ∨ a = -10 :=
sorry

end perfect_square_a_value_l195_195079


namespace simplify_fraction_l195_195446

-- Given
def num := 54
def denom := 972

-- Factorization condition
def factorization_54 : num = 2 * 3^3 := by 
  sorry

def factorization_972 : denom = 2^2 * 3^5 := by 
  sorry

-- GCD condition
def gcd_num_denom := 54

-- Division condition
def simplified_num := 1
def simplified_denom := 18

-- Statement to prove
theorem simplify_fraction : (num / denom) = (simplified_num / simplified_denom) := by 
  sorry

end simplify_fraction_l195_195446


namespace solve_equation_l195_195498

open Real

theorem solve_equation :
  ∀ x : ℝ, (
    (1 / ((x - 2) * (x - 3))) +
    (1 / ((x - 3) * (x - 4))) +
    (1 / ((x - 4) * (x - 5))) = (1 / 12)
  ) ↔ (x = 5 + sqrt 19 ∨ x = 5 - sqrt 19) := 
by 
  sorry

end solve_equation_l195_195498


namespace jana_height_l195_195932

theorem jana_height (Jess_height : ℕ) (h1 : Jess_height = 72) 
  (Kelly_height : ℕ) (h2 : Kelly_height = Jess_height - 3) 
  (Jana_height : ℕ) (h3 : Jana_height = Kelly_height + 5) : 
  Jana_height = 74 := by
  subst h1
  subst h2
  subst h3
  sorry

end jana_height_l195_195932


namespace reduction_percentage_toy_l195_195027

-- Definition of key parameters
def paintings_bought : ℕ := 10
def cost_per_painting : ℕ := 40
def toys_bought : ℕ := 8
def cost_per_toy : ℕ := 20
def total_cost : ℕ := (paintings_bought * cost_per_painting) + (toys_bought * cost_per_toy) -- $560
def painting_selling_price_per_unit : ℕ := cost_per_painting - (cost_per_painting * 10 / 100) -- $36
def total_loss : ℕ := 64

-- Define percentage reduction in the selling price of a wooden toy
variable {x : ℕ} -- Define x as a percentage value to be solved

-- Theorems to prove
theorem reduction_percentage_toy (x) : 
  (paintings_bought * painting_selling_price_per_unit) 
  + (toys_bought * (cost_per_toy - (cost_per_toy * x / 100))) 
  = (total_cost - total_loss) 
  → x = 15 := 
by
  sorry

end reduction_percentage_toy_l195_195027


namespace two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l195_195220

theorem two_pow_1000_mod_3 : 2^1000 % 3 = 1 := sorry
theorem two_pow_1000_mod_5 : 2^1000 % 5 = 1 := sorry
theorem two_pow_1000_mod_11 : 2^1000 % 11 = 1 := sorry
theorem two_pow_1000_mod_13 : 2^1000 % 13 = 3 := sorry

end two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l195_195220


namespace total_interest_rate_l195_195574

theorem total_interest_rate (I_total I_11: ℝ) (r_9 r_11: ℝ) (h1: I_total = 100000) (h2: I_11 = 12499.999999999998) (h3: I_11 < I_total):
  r_9 = 0.09 →
  r_11 = 0.11 →
  ( ((I_total - I_11) * r_9 + I_11 * r_11) / I_total * 100 = 9.25 ) :=
by
  sorry

end total_interest_rate_l195_195574


namespace sqrt_fraction_arith_sqrt_16_l195_195183

-- Prove that the square root of 4/9 is ±2/3
theorem sqrt_fraction (a b : ℕ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (h_a : a = 4) (h_b : b = 9) : 
    (Real.sqrt (a / (b : ℝ)) = abs (Real.sqrt a / Real.sqrt b)) :=
by
    rw [h_a, h_b]
    sorry

-- Prove that the arithmetic square root of √16 is 4.
theorem arith_sqrt_16 : Real.sqrt (Real.sqrt 16) = 4 :=
by
    sorry

end sqrt_fraction_arith_sqrt_16_l195_195183


namespace median_inequality_l195_195328

variables {α : ℝ} (A B C M : Point) (a b c : ℝ)

-- Definitions and conditions
def isTriangle (A B C : Point) : Prop := -- definition of triangle
sorry

def isMedian (A B C M : Point) : Prop := -- definition of median
sorry

-- Statement we want to prove
theorem median_inequality (h1 : isTriangle A B C) (h2 : isMedian A B C M) :
  2 * AM ≥ (b + c) * Real.cos (α / 2) :=
sorry

end median_inequality_l195_195328


namespace mario_age_difference_l195_195929

variable (Mario_age Maria_age : ℕ)

def age_conditions (Mario_age Maria_age difference : ℕ) : Prop :=
  Mario_age + Maria_age = 7 ∧
  Mario_age = 4 ∧
  Mario_age - Maria_age = difference

theorem mario_age_difference : ∃ (difference : ℕ), age_conditions 4 (4 - difference) difference ∧ difference = 1 := by
  sorry

end mario_age_difference_l195_195929


namespace woody_writing_time_l195_195419

open Real

theorem woody_writing_time (W : ℝ) 
  (h1 : ∃ n : ℝ, n * 12 = W * 12 + 3) 
  (h2 : 12 * W + (12 * W + 3) = 39) :
  W = 1.5 :=
by sorry

end woody_writing_time_l195_195419


namespace all_terms_are_integers_l195_195458

   noncomputable def a : ℕ → ℤ
   | 0 => 1
   | 1 => 1
   | 2 => 997
   | n + 3 => (1993 + a (n + 2) * a (n + 1)) / a n

   theorem all_terms_are_integers : ∀ n : ℕ, ∃ (a : ℕ → ℤ), 
     (a 1 = 1) ∧ 
     (a 2 = 1) ∧ 
     (a 3 = 997) ∧ 
     (∀ n : ℕ, a (n + 3) = (1993 + a (n + 2) * a (n + 1)) / a n) → 
     (∀ n : ℕ, ∃ k : ℤ, a n = k) := 
   by 
     sorry
   
end all_terms_are_integers_l195_195458


namespace rank_best_buy_LMS_l195_195819

theorem rank_best_buy_LMS (c_S q_S : ℝ) :
  let c_M := 1.75 * c_S
  let q_M := 1.1 * q_S
  let c_L := 1.25 * c_M
  let q_L := 1.5 * q_M
  (c_S / q_S) > (c_M / q_M) ∧ (c_M / q_M) > (c_L / q_L) :=
by
  sorry

end rank_best_buy_LMS_l195_195819


namespace standard_equation_line_standard_equation_circle_intersection_range_a_l195_195319

theorem standard_equation_line (a t x y : ℝ) (h1 : x = a - 2 * t * y) (h2 : y = -4 * t) : 
    2 * x - y - 2 * a = 0 :=
sorry

theorem standard_equation_circle (θ x y : ℝ) (h1 : x = 4 * Real.cos θ) (h2 : y = 4 * Real.sin θ) : 
    x ^ 2 + y ^ 2 = 16 :=
sorry

theorem intersection_range_a (a : ℝ) (h : ∃ (t θ : ℝ), (a - 2 * t * (-4 * t)) = 4 * (Real.cos θ) ∧ (-4 * t) = 4 * (Real.sin θ)) :
    -4 * Real.sqrt 5 <= a ∧ a <= 4 * Real.sqrt 5 :=
sorry

end standard_equation_line_standard_equation_circle_intersection_range_a_l195_195319


namespace intersection_complement_l195_195005

def set_M : Set ℝ := {x : ℝ | x^2 - x = 0}

def set_N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = 2 * n + 1}

theorem intersection_complement (h : UniversalSet = Set.univ) :
  set_M ∩ (UniversalSet \ set_N) = {0} := 
sorry

end intersection_complement_l195_195005


namespace domain_of_f_l195_195893

def domain_valid (x : ℝ) :=
  1 - x ≥ 0 ∧ 1 - x ≠ 1

theorem domain_of_f :
  ∀ x : ℝ, domain_valid x ↔ (x ∈ Set.Iio 0 ∪ Set.Ioc 0 1) :=
by
  sorry

end domain_of_f_l195_195893


namespace euro_operation_example_l195_195909

def euro_operation (x y : ℕ) : ℕ := 3 * x * y - x - y

theorem euro_operation_example : euro_operation 6 (euro_operation 4 2) = 300 := by
  sorry

end euro_operation_example_l195_195909


namespace value_of_fraction_l195_195708

variables (w x y : ℝ)

theorem value_of_fraction (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 :=
sorry

end value_of_fraction_l195_195708


namespace sin_cos_eq_l195_195801

theorem sin_cos_eq (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := sorry

end sin_cos_eq_l195_195801


namespace value_of_at_20_at_l195_195647

noncomputable def left_at (x : ℝ) : ℝ := 9 - x
noncomputable def right_at (x : ℝ) : ℝ := x - 9

theorem value_of_at_20_at : right_at (left_at 20) = -20 := by
  sorry

end value_of_at_20_at_l195_195647


namespace smallest_blocks_required_l195_195742

theorem smallest_blocks_required (L H : ℕ) (block_height block_long block_short : ℕ) 
  (vert_joins_staggered : Prop) (consistent_end_finish : Prop) : 
  L = 120 → H = 10 → block_height = 1 → block_long = 3 → block_short = 1 → 
  (vert_joins_staggered) → (consistent_end_finish) → 
  ∃ n, n = 415 :=
by
  sorry

end smallest_blocks_required_l195_195742


namespace range_of_a_l195_195720

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the monotonically increasing property on [0, ∞)
def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f →
  mono_increasing_on_nonneg f →
  (f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1) →
  (0 < a ∧ a ≤ 2) :=
by
  intros h_even h_mono h_ineq
  sorry

end range_of_a_l195_195720


namespace initial_number_of_friends_l195_195411

theorem initial_number_of_friends (X : ℕ) (H : 3 * (X - 3) = 15) : X = 8 :=
by
  sorry

end initial_number_of_friends_l195_195411


namespace matrix_sum_correct_l195_195364

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -1], ![3, 7]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![ -6, 8], ![5, -2]]
def mat_sum : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, 7], ![8, 5]]

theorem matrix_sum_correct : mat1 + mat2 = mat_sum :=
by
  rw [mat1, mat2]
  sorry

end matrix_sum_correct_l195_195364


namespace find_y_l195_195614

-- Definitions based on conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

-- Lean statement capturing the problem
theorem find_y
  (h1 : inversely_proportional x y)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (h4 : x = -12) :
  y = -56.25 :=
sorry  -- Proof omitted

end find_y_l195_195614


namespace inequality_does_not_hold_l195_195451

theorem inequality_does_not_hold {x y : ℝ} (h : x > y) : ¬ (-2 * x > -2 * y) ∧ (2023 * x > 2023 * y) ∧ (x - 1 > y - 1) ∧ (-x / 3 < -y / 3) :=
by {
  sorry
}

end inequality_does_not_hold_l195_195451


namespace colorings_equivalence_l195_195299

-- Define the problem setup
structure ProblemSetup where
  n : ℕ  -- Number of disks (8)
  blue : ℕ  -- Number of blue disks (3)
  red : ℕ  -- Number of red disks (3)
  green : ℕ  -- Number of green disks (2)
  rotations : ℕ  -- Number of rotations (4: 90°, 180°, 270°, 360°)
  reflections : ℕ  -- Number of reflections (8: 4 through vertices and 4 through midpoints)

def number_of_colorings (setup : ProblemSetup) : ℕ :=
  sorry -- This represents the complex implementation details

def correct_answer : ℕ := 43

theorem colorings_equivalence : ∀ (setup : ProblemSetup),
  setup.n = 8 → setup.blue = 3 → setup.red = 3 → setup.green = 2 → setup.rotations = 4 → setup.reflections = 8 →
  number_of_colorings setup = correct_answer :=
by
  intros setup h1 h2 h3 h4 h5 h6
  sorry

end colorings_equivalence_l195_195299


namespace smallest_number_among_neg2_neg1_0_pi_l195_195286

/-- The smallest number among -2, -1, 0, and π is -2. -/
theorem smallest_number_among_neg2_neg1_0_pi : min (min (min (-2 : ℝ) (-1)) 0) π = -2 := 
sorry

end smallest_number_among_neg2_neg1_0_pi_l195_195286


namespace minimum_beta_value_l195_195412

variable (α β : Real)

-- Defining the conditions given in the problem
def sin_alpha_condition : Prop := Real.sin α = -Real.sqrt 2 / 2
def cos_alpha_minus_beta_condition : Prop := Real.cos (α - β) = 1 / 2
def beta_greater_than_zero : Prop := β > 0

-- The theorem to be proven
theorem minimum_beta_value (h1 : sin_alpha_condition α) (h2 : cos_alpha_minus_beta_condition α β) (h3 : beta_greater_than_zero β) : β = Real.pi / 12 := 
sorry

end minimum_beta_value_l195_195412


namespace p_or_q_then_p_and_q_is_false_l195_195263

theorem p_or_q_then_p_and_q_is_false (p q : Prop) (hpq : p ∨ q) : ¬(p ∧ q) :=
sorry

end p_or_q_then_p_and_q_is_false_l195_195263


namespace prism_cutout_l195_195014

noncomputable def original_volume : ℕ := 15 * 5 * 4 -- Volume of the original prism
noncomputable def cutout_width : ℕ := 5

variables {x y : ℕ}

theorem prism_cutout:
  -- Given conditions
  (15 > 0) ∧ (5 > 0) ∧ (4 > 0) ∧ (x > 0) ∧ (y > 0) ∧ 
  -- The volume condition
  (original_volume - y * cutout_width * x = 120) →
  -- Prove that x + y = 15
  (x + y = 15) :=
sorry

end prism_cutout_l195_195014


namespace rhombus_diagonal_length_l195_195902

theorem rhombus_diagonal_length
  (d2 : ℝ)
  (h1 : d2 = 20)
  (area : ℝ)
  (h2 : area = 150) :
  ∃ d1 : ℝ, d1 = 15 ∧ (area = (d1 * d2) / 2) := by
  sorry

end rhombus_diagonal_length_l195_195902


namespace total_footprints_l195_195870

def pogo_footprints_per_meter : ℕ := 4
def grimzi_footprints_per_6_meters : ℕ := 3
def distance_traveled_meters : ℕ := 6000

theorem total_footprints : (pogo_footprints_per_meter * distance_traveled_meters) + (grimzi_footprints_per_6_meters * (distance_traveled_meters / 6)) = 27000 :=
by
  sorry

end total_footprints_l195_195870


namespace total_pairs_sold_l195_195018

theorem total_pairs_sold (H S : ℕ) 
    (soft_lens_cost hard_lens_cost : ℕ)
    (total_sales : ℕ)
    (h1 : soft_lens_cost = 150)
    (h2 : hard_lens_cost = 85)
    (h3 : S = H + 5)
    (h4 : soft_lens_cost * S + hard_lens_cost * H = total_sales)
    (h5 : total_sales = 1455) :
    H + S = 11 := 
  sorry

end total_pairs_sold_l195_195018


namespace parking_savings_l195_195347

theorem parking_savings
  (weekly_rent : ℕ := 10)
  (monthly_rent : ℕ := 40)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12)
  : weekly_rent * weeks_in_year - monthly_rent * months_in_year = 40 := 
by
  sorry

end parking_savings_l195_195347


namespace josh_money_left_l195_195445

theorem josh_money_left :
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  money_left = 15.87 :=
by
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  have h1 : total_spent = 84.13 := sorry
  have h2 : money_left = initial_money - 84.13 := sorry
  have h3 : money_left = 15.87 := sorry
  exact h3

end josh_money_left_l195_195445


namespace parallel_lines_m_eq_neg4_l195_195273

theorem parallel_lines_m_eq_neg4 (m : ℝ) (h1 : (m-2) ≠ -m) 
  (h2 : (m-2) / 3 = -m / (m + 2)) : m = -4 :=
sorry

end parallel_lines_m_eq_neg4_l195_195273


namespace range_of_a_l195_195134

theorem range_of_a (a : ℝ) :
  (0 + 0 + a) * (2 - 1 + a) < 0 ↔ (-1 < a ∧ a < 0) :=
by sorry

end range_of_a_l195_195134


namespace interior_triangles_from_chords_l195_195171

theorem interior_triangles_from_chords (h₁ : ∀ p₁ p₂ p₃ : Prop, ¬(p₁ ∧ p₂ ∧ p₃)) : 
  ∀ (nine_points_on_circle : Finset ℝ) (h₂ : nine_points_on_circle.card = 9), 
    ∃ (triangles : ℕ), triangles = 210 := 
by 
  sorry

end interior_triangles_from_chords_l195_195171


namespace simplified_expression_value_l195_195861

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l195_195861


namespace oranges_to_apples_equiv_apples_for_36_oranges_l195_195832

-- Conditions
def weight_equiv (oranges apples : ℕ) : Prop :=
  9 * oranges = 6 * apples

-- Question (Theorem to Prove)
theorem oranges_to_apples_equiv_apples_for_36_oranges:
  ∃ (apples : ℕ), apples = 24 ∧ weight_equiv 36 apples :=
by
  use 24
  sorry

end oranges_to_apples_equiv_apples_for_36_oranges_l195_195832


namespace calc_result_l195_195703

-- Define the operation and conditions
def my_op (a b c : ℝ) : ℝ :=
  3 * (a - b - c)^2

theorem calc_result (x y z : ℝ) : 
  my_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 :=
by
  sorry

end calc_result_l195_195703


namespace abs_gt_two_nec_but_not_suff_l195_195115

theorem abs_gt_two_nec_but_not_suff (x : ℝ) : (|x| > 2 → x < -2) ∧ (¬ (|x| > 2 ↔ x < -2)) := 
sorry

end abs_gt_two_nec_but_not_suff_l195_195115


namespace roshini_spent_on_sweets_l195_195961

variable (initial_amount friends_amount total_friends_amount sweets_amount : ℝ)

noncomputable def Roshini_conditions (initial_amount friends_amount total_friends_amount sweets_amount : ℝ) :=
  initial_amount = 10.50 ∧ friends_amount = 6.80 ∧ sweets_amount = 3.70 ∧ 2 * 3.40 = 6.80

theorem roshini_spent_on_sweets :
  ∀ (initial_amount friends_amount total_friends_amount sweets_amount : ℝ),
    Roshini_conditions initial_amount friends_amount total_friends_amount sweets_amount →
    initial_amount - friends_amount = sweets_amount :=
by
  intros initial_amount friends_amount total_friends_amount sweets_amount h
  cases h
  sorry

end roshini_spent_on_sweets_l195_195961


namespace polynomial_roots_l195_195048

theorem polynomial_roots (p q BD DC : ℝ) (h_sum : BD + DC = p) (h_prod : BD * DC = q^2) :
    Polynomial.roots (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C p * Polynomial.X + Polynomial.C (q^2)) = {BD, DC} :=
sorry

end polynomial_roots_l195_195048


namespace maximum_at_vertex_l195_195691

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_at_vertex (a b c x_0 : ℝ) (h_a : a < 0) (h_x0 : 2 * a * x_0 + b = 0) :
  ∀ x : ℝ, quadratic_function a b c x ≤ quadratic_function a b c x_0 :=
sorry

end maximum_at_vertex_l195_195691


namespace food_drive_total_cans_l195_195839

def total_cans_brought (M J R : ℕ) : ℕ := M + J + R

theorem food_drive_total_cans (M J R : ℕ) 
  (h1 : M = 4 * J) 
  (h2 : J = 2 * R + 5) 
  (h3 : M = 100) : 
  total_cans_brought M J R = 135 :=
by sorry

end food_drive_total_cans_l195_195839


namespace vector_parallel_solution_l195_195779

theorem vector_parallel_solution 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (x, -9)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  x = -6 :=
by
  sorry

end vector_parallel_solution_l195_195779


namespace apples_to_pears_l195_195059

theorem apples_to_pears :
  (3 / 4) * 12 = 9 → (2 / 3) * 6 = 4 :=
by {
  sorry
}

end apples_to_pears_l195_195059


namespace pentagon_diagonal_l195_195417

theorem pentagon_diagonal (a d : ℝ) (h : d^2 = a^2 + a * d) : 
  d = a * (Real.sqrt 5 + 1) / 2 :=
sorry

end pentagon_diagonal_l195_195417


namespace divide_nuts_equal_l195_195396

-- Define the conditions: sequence of 64 nuts where adjacent differ by 1 gram
def is_valid_sequence (seq : List Int) :=
  seq.length = 64 ∧ (∀ i < 63, (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ + 1) ∨ (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ - 1))

-- Main theorem statement: prove that the sequence can be divided into two groups with equal number of nuts and equal weights
theorem divide_nuts_equal (seq : List Int) (h : is_valid_sequence seq) :
  ∃ (s1 s2 : List Int), s1.length = 32 ∧ s2.length = 32 ∧ (s1.sum = s2.sum) :=
sorry

end divide_nuts_equal_l195_195396


namespace correct_average_calculation_l195_195838

-- Conditions as definitions
def incorrect_average := 5
def num_values := 10
def incorrect_num := 26
def correct_num := 36

-- Statement to prove
theorem correct_average_calculation : 
  (incorrect_average * num_values + (correct_num - incorrect_num)) / num_values = 6 :=
by
  -- Placeholder for the proof
  sorry

end correct_average_calculation_l195_195838


namespace mr_smith_total_cost_l195_195834

noncomputable def total_cost : ℝ :=
  let adult_price := 30
  let child_price := 15
  let teen_price := 25
  let senior_discount := 0.10
  let college_discount := 0.05
  let senior_price := adult_price * (1 - senior_discount)
  let college_price := adult_price * (1 - college_discount)
  let soda_price := 2
  let iced_tea_price := 3
  let coffee_price := 4
  let juice_price := 1.50
  let wine_price := 6
  let buffet_cost := 2 * adult_price + 2 * senior_price + 3 * child_price + teen_price + 2 * college_price
  let drinks_cost := 3 * soda_price + 2 * iced_tea_price + coffee_price + juice_price + 2 * wine_price
  buffet_cost + drinks_cost

theorem mr_smith_total_cost : total_cost = 270.50 :=
by
  sorry

end mr_smith_total_cost_l195_195834


namespace handshake_problem_l195_195186

-- Define the remainder operation
def r_mod (n : ℕ) (k : ℕ) : ℕ := n % k

-- Define the function F
def F (t : ℕ) : ℕ := r_mod (t^3) 5251

-- The lean theorem statement with the given conditions and expected results
theorem handshake_problem :
  ∃ (x y : ℕ),
    F x = 506 ∧
    F (x + 1) = 519 ∧
    F y = 229 ∧
    F (y + 1) = 231 ∧
    x = 102 ∧
    y = 72 :=
by
  sorry

end handshake_problem_l195_195186


namespace robot_distance_proof_l195_195954

noncomputable def distance (south1 south2 south3 east1 east2 : ℝ) : ℝ :=
  Real.sqrt ((south1 + south2 + south3)^2 + (east1 + east2)^2)

theorem robot_distance_proof :
  distance 1.2 1.8 1.0 1.0 2.0 = 5.0 :=
by
  sorry

end robot_distance_proof_l195_195954


namespace intersection_M_N_l195_195637

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := U \ complement_U_N

theorem intersection_M_N : M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_M_N_l195_195637


namespace sin_subtract_pi_over_6_l195_195118

theorem sin_subtract_pi_over_6 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hcos : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_subtract_pi_over_6_l195_195118


namespace find_f_7_5_l195_195490

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_2  : ∀ x, f (x + 2) = -f x
axiom initial_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- Proof goes here
  sorry

end find_f_7_5_l195_195490


namespace denomination_of_bill_l195_195426

def cost_berries : ℝ := 7.19
def cost_peaches : ℝ := 6.83
def change_received : ℝ := 5.98

theorem denomination_of_bill :
  (cost_berries + cost_peaches) + change_received = 20.0 := 
by 
  sorry

end denomination_of_bill_l195_195426


namespace sum_of_geometric_sequence_l195_195238

noncomputable def geometric_sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence (a₁ q : ℝ) (n : ℕ) 
  (h1 : a₁ + a₁ * q^3 = 10) 
  (h2 : a₁ * q + a₁ * q^4 = 20) : 
  geometric_sequence_sum a₁ q n = (10 / 9) * (2^n - 1) :=
by 
  sorry

end sum_of_geometric_sequence_l195_195238


namespace find_b_l195_195318

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 :=
sorry

end find_b_l195_195318


namespace equivalent_single_discount_l195_195891

variable (x : ℝ)
variable (original_price : ℝ := x)
variable (discount1 : ℝ := 0.15)
variable (discount2 : ℝ := 0.10)
variable (discount3 : ℝ := 0.05)

theorem equivalent_single_discount :
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let equivalent_discount := (1 - final_price / original_price)
  equivalent_discount = 0.27 := 
by 
  sorry

end equivalent_single_discount_l195_195891


namespace max_tickets_with_120_l195_195477

-- Define the cost of tickets
def cost_ticket (n : ℕ) : ℕ :=
  if n ≤ 5 then n * 15
  else 5 * 15 + (n - 5) * 12

-- Define the maximum number of tickets Jane can buy with 120 dollars
def max_tickets (money : ℕ) : ℕ :=
  if money ≤ 75 then money / 15
  else 5 + (money - 75) / 12

-- Prove that with 120 dollars, the maximum number of tickets Jane can buy is 8
theorem max_tickets_with_120 : max_tickets 120 = 8 :=
by
  sorry

end max_tickets_with_120_l195_195477


namespace base_of_square_eq_l195_195476

theorem base_of_square_eq (b : ℕ) (h : b > 6) : 
  (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 → b = 7 :=
by
  sorry

end base_of_square_eq_l195_195476


namespace ducks_joined_l195_195085

theorem ducks_joined (initial_ducks total_ducks ducks_joined : ℕ) 
  (h_initial: initial_ducks = 13)
  (h_total: total_ducks = 33) :
  initial_ducks + ducks_joined = total_ducks → ducks_joined = 20 :=
by
  intros h_equation
  rw [h_initial, h_total] at h_equation
  sorry

end ducks_joined_l195_195085


namespace martians_cannot_hold_hands_l195_195852

-- Define the number of hands each Martian possesses
def hands_per_martian := 3

-- Define the number of Martians
def number_of_martians := 7

-- Define the total number of hands
def total_hands := hands_per_martian * number_of_martians

-- Prove that it is not possible for the seven Martians to hold hands with each other
theorem martians_cannot_hold_hands :
  ¬ ∃ (pairs : ℕ), 2 * pairs = total_hands :=
by
  sorry

end martians_cannot_hold_hands_l195_195852


namespace isosceles_trapezoid_side_length_is_five_l195_195749

noncomputable def isosceles_trapezoid_side_length (b1 b2 area : ℝ) : ℝ :=
  let h := 2 * area / (b1 + b2)
  let base_diff_half := (b2 - b1) / 2
  Real.sqrt (h^2 + base_diff_half^2)
  
theorem isosceles_trapezoid_side_length_is_five :
  isosceles_trapezoid_side_length 6 12 36 = 5 := by
  sorry

end isosceles_trapezoid_side_length_is_five_l195_195749


namespace six_diggers_five_hours_l195_195985

theorem six_diggers_five_hours (holes_per_hour_per_digger : ℝ) 
  (h1 : 3 * holes_per_hour_per_digger * 3 = 3) :
  6 * (holes_per_hour_per_digger) * 5 = 10 :=
by
  -- The proof will go here, but we only need to state the theorem
  sorry

end six_diggers_five_hours_l195_195985


namespace find_constant_k_eq_l195_195096

theorem find_constant_k_eq : ∃ k : ℤ, (-x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4)) ↔ (k = -17) :=
by
  sorry

end find_constant_k_eq_l195_195096


namespace find_common_ratio_l195_195246

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Given conditions
lemma a2_eq_8 (a₁ q : ℝ) : geometric_sequence a₁ q 2 = 8 :=
by sorry

lemma a5_eq_64 (a₁ q : ℝ) : geometric_sequence a₁ q 5 = 64 :=
by sorry

-- The common ratio q
theorem find_common_ratio (a₁ q : ℝ) (hq : 0 < q) :
  (geometric_sequence a₁ q 2 = 8) → (geometric_sequence a₁ q 5 = 64) → q = 2 :=
by sorry

end find_common_ratio_l195_195246


namespace partial_fraction_decomposition_l195_195544

theorem partial_fraction_decomposition :
  ∃ A B C : ℚ, (∀ x : ℚ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
          (x^2 + 2 * x - 8) / (x^3 - x - 2) = A / (x + 1) + (B * x + C) / (x^2 - x + 2)) ∧
          A = -9/4 ∧ B = 13/4 ∧ C = -7/2 :=
sorry

end partial_fraction_decomposition_l195_195544


namespace number_of_possible_values_r_l195_195559

noncomputable def is_closest_approx (r : ℝ) : Prop :=
  (r >= 0.2857) ∧ (r < 0.2858)

theorem number_of_possible_values_r : 
  ∃ n : ℕ, (∀ r : ℝ, is_closest_approx r ↔ r = 0.2857 ∨ r = 0.2858 ∨ r = 0.2859) ∧ n = 3 :=
by
  sorry

end number_of_possible_values_r_l195_195559


namespace hyperbola_eccentricity_l195_195070

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_asymptote_parallel : b = 2 * a)
  (h_c_squared : c^2 = a^2 + b^2)
  (h_e_def : e = c / a) :
  e = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l195_195070


namespace factorize_cubic_l195_195545

theorem factorize_cubic (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_cubic_l195_195545


namespace graph_passes_through_fixed_point_l195_195661

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x-1)

theorem graph_passes_through_fixed_point (a : ℝ) : f a 1 = 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end graph_passes_through_fixed_point_l195_195661


namespace length_of_plot_l195_195354

-- Define the conditions
def width : ℝ := 60
def num_poles : ℕ := 60
def dist_between_poles : ℝ := 5
def num_intervals : ℕ := num_poles - 1
def perimeter : ℝ := num_intervals * dist_between_poles

-- Define the theorem and the correctness condition
theorem length_of_plot : 
  perimeter = 2 * (length + width) → 
  length = 87.5 :=
by
  sorry

end length_of_plot_l195_195354


namespace simplify_expression_l195_195196

variable (a b : ℝ)

theorem simplify_expression : (a + b) * (3 * a - b) - b * (a - b) = 3 * a ^ 2 + a * b :=
by
  sorry

end simplify_expression_l195_195196


namespace sales_fraction_l195_195636

theorem sales_fraction (A D : ℝ) (h : D = 2 * A) : D / (11 * A + D) = 2 / 13 :=
by
  sorry

end sales_fraction_l195_195636


namespace probability_of_x_gt_3y_is_correct_l195_195670

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle_width := 2016
  let rectangle_height := 2017
  let triangle_height := 672 -- 2016 / 3
  let triangle_area := 1 / 2 * rectangle_width * triangle_height
  let rectangle_area := rectangle_width * rectangle_height
  triangle_area / rectangle_area

theorem probability_of_x_gt_3y_is_correct :
  probability_x_gt_3y = 336 / 2017 :=
by
  -- Proof will be filled in later
  sorry

end probability_of_x_gt_3y_is_correct_l195_195670


namespace average_is_4_l195_195965

theorem average_is_4 (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := 
by 
  sorry 

end average_is_4_l195_195965


namespace amphibians_count_l195_195178

-- Define the conditions
def frogs : Nat := 7
def salamanders : Nat := 4
def tadpoles : Nat := 30
def newt : Nat := 1

-- Define the total number of amphibians observed by Hunter
def total_amphibians : Nat := frogs + salamanders + tadpoles + newt

-- State the theorem
theorem amphibians_count : total_amphibians = 42 := 
by 
  -- proof goes here
  sorry

end amphibians_count_l195_195178


namespace amusement_park_admission_fees_l195_195047

theorem amusement_park_admission_fees
  (num_children : ℕ) (num_adults : ℕ)
  (fee_child : ℝ) (fee_adult : ℝ)
  (total_people : ℕ) (expected_total_fees : ℝ) :
  num_children = 180 →
  fee_child = 1.5 →
  fee_adult = 4.0 →
  total_people = 315 →
  expected_total_fees = 810 →
  num_children + num_adults = total_people →
  (num_children : ℝ) * fee_child + (num_adults : ℝ) * fee_adult = expected_total_fees := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amusement_park_admission_fees_l195_195047


namespace sum_of_h_and_k_l195_195338

theorem sum_of_h_and_k (foci1 foci2 : ℝ × ℝ) (pt : ℝ × ℝ) (a b h k : ℝ) 
  (h_positive : a > 0) (b_positive : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = if (x, y) = pt then 1 else sorry)
  (foci_eq : foci1 = (1, 2) ∧ foci2 = (4, 2))
  (pt_eq : pt = (-1, 5)) :
  h + k = 4.5 :=
sorry

end sum_of_h_and_k_l195_195338


namespace service_center_milepost_l195_195526

theorem service_center_milepost :
  ∀ (first_exit seventh_exit service_fraction : ℝ), 
    first_exit = 50 →
    seventh_exit = 230 →
    service_fraction = 3 / 4 →
    (first_exit + service_fraction * (seventh_exit - first_exit) = 185) :=
by
  intros first_exit seventh_exit service_fraction h_first h_seventh h_fraction
  sorry

end service_center_milepost_l195_195526


namespace fair_share_of_bill_l195_195222

noncomputable def total_bill : Real := 139.00
noncomputable def tip_percent : Real := 0.10
noncomputable def num_people : Real := 6
noncomputable def expected_amount_per_person : Real := 25.48

theorem fair_share_of_bill :
  (total_bill + (tip_percent * total_bill)) / num_people = expected_amount_per_person :=
by
  sorry

end fair_share_of_bill_l195_195222


namespace solve_system_l195_195400

theorem solve_system : ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 8 * x + 5 * y = 7 ∧ x = 1 / 4 ∧ y = 1 :=
by
  sorry

end solve_system_l195_195400


namespace complement_A_union_B_range_of_m_l195_195344

def setA : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 5*x - 14) }
def setB : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (-x^2 - 7*x - 12) }
def setC (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem complement_A_union_B :
  (A ∪ B)ᶜ = Set.Ioo (-2 : ℝ) 7 :=
sorry

theorem range_of_m (m : ℝ) :
  (A ∪ setC m = A) → (m < 2 ∨ m ≥ 6) :=
sorry

end complement_A_union_B_range_of_m_l195_195344


namespace total_weight_of_4_moles_of_ba_cl2_l195_195619

-- Conditions
def atomic_weight_ba : ℝ := 137.33
def atomic_weight_cl : ℝ := 35.45
def moles_ba_cl2 : ℝ := 4

-- Molecular weight of BaCl2
def molecular_weight_ba_cl2 : ℝ := 
  atomic_weight_ba + 2 * atomic_weight_cl

-- Total weight of 4 moles of BaCl2
def total_weight : ℝ := 
  molecular_weight_ba_cl2 * moles_ba_cl2

-- Theorem stating the total weight of 4 moles of BaCl2
theorem total_weight_of_4_moles_of_ba_cl2 :
  total_weight = 832.92 :=
sorry

end total_weight_of_4_moles_of_ba_cl2_l195_195619


namespace mark_reading_pages_before_injury_l195_195616

theorem mark_reading_pages_before_injury:
  ∀ (h_increased: Nat) (pages_week: Nat), 
  (h_increased = 2 + (2 * 3/2)) ∧ (pages_week = 1750) → 100 = pages_week / 7 / h_increased * 2 := 
by
  sorry

end mark_reading_pages_before_injury_l195_195616


namespace solve_system_of_equations_l195_195807

theorem solve_system_of_equations :
  ∀ (x y z : ℚ), 
    (x * y = x + 2 * y ∧
     y * z = y + 3 * z ∧
     z * x = z + 4 * x) ↔
    (x = 0 ∧ y = 0 ∧ z = 0) ∨
    (x = 25 / 9 ∧ y = 25 / 7 ∧ z = 25 / 4) := by
  sorry

end solve_system_of_equations_l195_195807


namespace cost_of_sandwiches_and_smoothies_l195_195906

-- Define the cost of sandwiches and smoothies
def sandwich_cost := 4
def smoothie_cost := 3

-- Define the discount applicable
def sandwich_discount := 1
def total_sandwiches := 6
def total_smoothies := 7

-- Calculate the effective cost per sandwich considering discount
def effective_sandwich_cost := if total_sandwiches > 4 then sandwich_cost - sandwich_discount else sandwich_cost

-- Calculate the total cost for sandwiches
def sandwiches_cost := total_sandwiches * effective_sandwich_cost

-- Calculate the total cost for smoothies
def smoothies_cost := total_smoothies * smoothie_cost

-- Calculate the total cost
def total_cost := sandwiches_cost + smoothies_cost

-- The main statement to prove
theorem cost_of_sandwiches_and_smoothies : total_cost = 39 := by
  -- skip the proof
  sorry

end cost_of_sandwiches_and_smoothies_l195_195906


namespace max_a_value_l195_195650

theorem max_a_value : ∃ a b : ℕ, 1 < a ∧ a < b ∧
  (∀ x y : ℝ, y = -2 * x + 4033 ∧ y = |x - 1| + |x + a| + |x - b| → 
  a = 4031) := sorry

end max_a_value_l195_195650


namespace num_more_green_l195_195068

noncomputable def num_people : ℕ := 150
noncomputable def more_blue : ℕ := 90
noncomputable def both_green_blue : ℕ := 40
noncomputable def neither_green_blue : ℕ := 20

theorem num_more_green :
  (num_people + more_blue + both_green_blue + neither_green_blue) ≤ 150 →
  (more_blue - both_green_blue) + both_green_blue + neither_green_blue ≤ num_people →
  (num_people - 
  ((more_blue - both_green_blue) + both_green_blue + neither_green_blue)) + both_green_blue = 80 :=
by
    intros h1 h2
    sorry

end num_more_green_l195_195068


namespace simplify_expression_l195_195688

theorem simplify_expression : 4 * Real.sqrt (1 / 2) + 3 * Real.sqrt (1 / 3) - Real.sqrt 8 = Real.sqrt 3 := 
by 
  sorry

end simplify_expression_l195_195688


namespace least_number_divisible_l195_195390

theorem least_number_divisible (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 9 = 4) (h3 : n % 18 = 4) : n = 130 := sorry

end least_number_divisible_l195_195390


namespace count_multiples_5_or_7_but_not_35_l195_195371

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end count_multiples_5_or_7_but_not_35_l195_195371


namespace veronica_reroll_probability_is_correct_l195_195605

noncomputable def veronica_reroll_probability : ℚ :=
  let P := (5 : ℚ) / 54
  P

theorem veronica_reroll_probability_is_correct :
  veronica_reroll_probability = (5 : ℚ) / 54 := sorry

end veronica_reroll_probability_is_correct_l195_195605


namespace isosceles_triangle_base_l195_195038

theorem isosceles_triangle_base (b : ℝ) (h1 : 7 + 7 + b = 20) : b = 6 :=
by {
    sorry
}

end isosceles_triangle_base_l195_195038


namespace computer_game_cost_l195_195645

variable (ticket_cost : ℕ := 12)
variable (num_tickets : ℕ := 3)
variable (total_spent : ℕ := 102)

theorem computer_game_cost (C : ℕ) (h : C + num_tickets * ticket_cost = total_spent) : C = 66 :=
by
  -- Proof would go here
  sorry

end computer_game_cost_l195_195645


namespace min_value_of_expression_l195_195382

open Real

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h_perp : (x - 1) * 1 + 3 * y = 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_ab_perp : (a - 1) * 1 + 3 * b = 0), (1 / a) + (1 / (3 * b)) ≥ m) :=
by
  use 4
  sorry

end min_value_of_expression_l195_195382


namespace books_remaining_after_second_day_l195_195069

theorem books_remaining_after_second_day :
  let initial_books := 100
  let first_day_borrowed := 5 * 2
  let second_day_borrowed := 20
  let total_borrowed := first_day_borrowed + second_day_borrowed
  let remaining_books := initial_books - total_borrowed
  remaining_books = 70 :=
by
  sorry

end books_remaining_after_second_day_l195_195069


namespace percentage_calculation_l195_195933

def percentage_less_than_50000_towns : Float := 85

def percentage_less_than_20000_towns : Float := 20
def percentage_20000_to_49999_towns : Float := 65

theorem percentage_calculation :
  percentage_less_than_50000_towns = percentage_less_than_20000_towns + percentage_20000_to_49999_towns :=
by
  sorry

end percentage_calculation_l195_195933


namespace solution_system_linear_eqns_l195_195835

theorem solution_system_linear_eqns
    (a1 b1 c1 a2 b2 c2 : ℝ)
    (h1: a1 * 6 + b1 * 3 = c1)
    (h2: a2 * 6 + b2 * 3 = c2) :
    (4 * a1 * 22 + 3 * b1 * 33 = 11 * c1) ∧
    (4 * a2 * 22 + 3 * b2 * 33 = 11 * c2) :=
by
    sorry

end solution_system_linear_eqns_l195_195835


namespace total_number_of_trees_l195_195276

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l195_195276


namespace coordinate_fifth_point_l195_195880

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end coordinate_fifth_point_l195_195880


namespace temperature_lower_than_freezing_point_is_minus_three_l195_195577

-- Define the freezing point of water
def freezing_point := 0 -- in degrees Celsius

-- Define the temperature lower by a certain value
def lower_temperature (t: Int) (delta: Int) := t - delta

-- State the theorem to be proved
theorem temperature_lower_than_freezing_point_is_minus_three:
  lower_temperature freezing_point 3 = -3 := by
  sorry

end temperature_lower_than_freezing_point_is_minus_three_l195_195577


namespace son_work_time_l195_195270

theorem son_work_time (M S : ℝ) 
  (hM : M = 1 / 4)
  (hCombined : M + S = 1 / 3) : 
  S = 1 / 12 :=
by
  sorry

end son_work_time_l195_195270


namespace sum_of_squares_l195_195509

theorem sum_of_squares (a b : ℕ) (h₁ : a = 300000) (h₂ : b = 20000) : a^2 + b^2 = 9004000000 :=
by
  rw [h₁, h₂]
  sorry

end sum_of_squares_l195_195509


namespace complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l195_195062

theorem complete_even_square_diff_eqn : (10^2 - 8^2 = 4 * 9) :=
by sorry

theorem even_square_diff_multiple_of_four (n : ℕ) : (4 * (n + 1) * (n + 1) - 4 * n * n) % 4 = 0 :=
by sorry

theorem odd_square_diff_multiple_of_eight (m : ℕ) : ((2 * m + 1)^2 - (2 * m - 1)^2) % 8 = 0 :=
by sorry

end complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l195_195062


namespace lloyd_total_hours_worked_l195_195853

noncomputable def total_hours_worked (daily_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier: ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_hours := 7.5
  let regular_pay := regular_hours * regular_rate
  if total_earnings <= regular_pay then daily_hours else
  let overtime_pay := total_earnings - regular_pay
  let overtime_hours := overtime_pay / (regular_rate * overtime_multiplier)
  regular_hours + overtime_hours

theorem lloyd_total_hours_worked :
  total_hours_worked 7.5 5.50 1.5 66 = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l195_195853


namespace trash_can_prices_and_minimum_A_can_purchase_l195_195472

theorem trash_can_prices_and_minimum_A_can_purchase 
  (x y : ℕ) 
  (h₁ : 3 * x + 4 * y = 580)
  (h₂ : 6 * x + 5 * y = 860)
  (total_trash_cans : ℕ)
  (total_cost : ℕ)
  (cond₃ : total_trash_cans = 200)
  (cond₄ : 60 * (total_trash_cans - x) + 100 * x ≤ 15000) : 
  x = 60 ∧ y = 100 ∧ x ≥ 125 := 
sorry

end trash_can_prices_and_minimum_A_can_purchase_l195_195472


namespace printer_a_time_l195_195609

theorem printer_a_time :
  ∀ (A B : ℕ), 
  B = A + 4 → 
  A + B = 12 → 
  (480 / A = 120) :=
by 
  intros A B hB hAB
  sorry

end printer_a_time_l195_195609


namespace factorize_poly1_factorize_poly2_l195_195056

theorem factorize_poly1 (x : ℝ) : 2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4) :=
by
  sorry

theorem factorize_poly2 (x : ℝ) : x^2 - 14 * x + 49 = (x - 7) ^ 2 :=
by
  sorry

end factorize_poly1_factorize_poly2_l195_195056


namespace total_preparation_and_cooking_time_l195_195291

def time_to_chop_pepper := 3
def time_to_chop_onion := 4
def time_to_slice_mushroom := 2
def time_to_dice_tomato := 3
def time_to_grate_cheese := 1
def time_to_assemble_and_cook_omelet := 6

def num_peppers := 8
def num_onions := 4
def num_mushrooms := 6
def num_tomatoes := 6
def num_omelets := 10

theorem total_preparation_and_cooking_time :
  (num_peppers * time_to_chop_pepper) +
  (num_onions * time_to_chop_onion) +
  (num_mushrooms * time_to_slice_mushroom) +
  (num_tomatoes * time_to_dice_tomato) +
  (num_omelets * time_to_grate_cheese) +
  (num_omelets * time_to_assemble_and_cook_omelet) = 140 :=
by
  sorry

end total_preparation_and_cooking_time_l195_195291


namespace perpendicular_line_plane_l195_195158

variables {m : ℝ}

theorem perpendicular_line_plane (h : (4 / 2) = (2 / 1) ∧ (2 / 1) = (m / -1)) : m = -2 :=
by
  sorry

end perpendicular_line_plane_l195_195158


namespace exists_unique_representation_l195_195189

theorem exists_unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3 * x + y) / 2 :=
sorry

end exists_unique_representation_l195_195189


namespace total_price_l195_195255

theorem total_price (r w : ℕ) (hr : r = 4275) (hw : w = r - 1490) : r + w = 7060 :=
by
  sorry

end total_price_l195_195255


namespace ratio_black_white_l195_195596

-- Definitions of the parameters
variables (B W : ℕ)
variables (h1 : B + W = 200)
variables (h2 : 30 * B + 25 * W = 5500)

theorem ratio_black_white (B W : ℕ) (h1 : B + W = 200) (h2 : 30 * B + 25 * W = 5500) :
  B = W :=
by
  -- Proof omitted
  sorry

end ratio_black_white_l195_195596


namespace θ_values_l195_195930

-- Define the given conditions
def terminal_side_coincides (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + 360 * k

def θ_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- The main theorem
theorem θ_values (θ : ℝ) (h_terminal : terminal_side_coincides θ) (h_range : θ_in_range θ) :
  θ = 0 ∨ θ = 60 ∨ θ = 120 ∨ θ = 180 ∨ θ = 240 ∨ θ = 300 :=
sorry

end θ_values_l195_195930


namespace time_b_used_l195_195321

noncomputable def time_b_used_for_proof : ℚ :=
  let C : ℚ := 1
  let C_a : ℚ := 1 / 4 * C
  let t_a : ℚ := 15
  let p_a : ℚ := 1 / 3
  let p_b : ℚ := 2 / 3
  let ratio : ℚ := (C_a * t_a) / ((C - C_a) * (t_a * p_a / p_b))
  t_a * p_a / p_b

theorem time_b_used : time_b_used_for_proof = 10 / 3 := by
  sorry

end time_b_used_l195_195321


namespace meeting_time_l195_195195

noncomputable def start_time : ℕ := 13 -- 1 pm in 24-hour format
noncomputable def speed_A : ℕ := 5 -- in kmph
noncomputable def speed_B : ℕ := 7 -- in kmph
noncomputable def initial_distance : ℕ := 24 -- in km

theorem meeting_time : start_time + (initial_distance / (speed_A + speed_B)) = 15 :=
by
  sorry

end meeting_time_l195_195195


namespace find_n_l195_195638

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 16 = 48) (h_gcf : Nat.gcd n 16 = 4) : n = 12 :=
by
  sorry

end find_n_l195_195638


namespace number_with_all_8s_is_divisible_by_13_l195_195356

theorem number_with_all_8s_is_divisible_by_13 :
  ∀ (N : ℕ), (N = 8 * (10^1974 - 1) / 9) → 13 ∣ N :=
by
  sorry

end number_with_all_8s_is_divisible_by_13_l195_195356


namespace k_value_opposite_solutions_l195_195746

theorem k_value_opposite_solutions (k x1 x2 : ℝ) 
  (h1 : 3 * (2 * x1 - 1) = 1 - 2 * x1)
  (h2 : 8 - k = 2 * (x2 + 1))
  (opposite : x2 = -x1) :
  k = 7 :=
by sorry

end k_value_opposite_solutions_l195_195746


namespace greatest_possible_z_l195_195181

theorem greatest_possible_z (x y z : ℕ) (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hx_cond : 7 < x) (hy_cond : y < 15) (hx_lt_y : x < y) (hz_gt_zero : z > 0) 
  (hy_sub_x_div_z : (y - x) % z = 0) : z = 2 := 
sorry

end greatest_possible_z_l195_195181


namespace min_omega_condition_l195_195497

theorem min_omega_condition :
  ∃ (ω: ℝ) (k: ℤ), (ω > 0) ∧ (ω = 6 * k + 1 / 2) ∧ (∀ (ω' : ℝ), (ω' > 0) ∧ (∃ (k': ℤ), ω' = 6 * k' + 1 / 2) → ω ≤ ω') := 
sorry

end min_omega_condition_l195_195497


namespace soccer_team_points_l195_195001

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end soccer_team_points_l195_195001


namespace find_r_l195_195546

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 16.5) : r = 8.5 :=
sorry

end find_r_l195_195546


namespace fraction_equality_l195_195078

variables {a b : ℝ}

theorem fraction_equality (h : ab * (a + b) = 1) (ha : a > 0) (hb : b > 0) : 
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := 
sorry

end fraction_equality_l195_195078


namespace cups_remaining_l195_195209

-- Each definition only directly appears in the conditions problem
def required_cups : ℕ := 7
def added_cups : ℕ := 3

-- The proof problem capturing Joan needs to add 4 more cups of flour.
theorem cups_remaining : required_cups - added_cups = 4 := 
by
  -- The proof is skipped using sorry.
  sorry

end cups_remaining_l195_195209


namespace correct_number_of_eggs_to_buy_l195_195606

/-- Define the total number of eggs needed and the number of eggs given by Andrew -/
def total_eggs_needed : ℕ := 222
def eggs_given_by_andrew : ℕ := 155

/-- Define a statement asserting the correct number of eggs to buy -/
def remaining_eggs_to_buy : ℕ := total_eggs_needed - eggs_given_by_andrew

/-- The statement of the proof problem -/
theorem correct_number_of_eggs_to_buy : remaining_eggs_to_buy = 67 :=
by sorry

end correct_number_of_eggs_to_buy_l195_195606


namespace combination_2586_1_eq_2586_l195_195992

noncomputable def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_2586_1_eq_2586 : combination 2586 1 = 2586 := by
  sorry

end combination_2586_1_eq_2586_l195_195992


namespace regular_polygon_sides_l195_195496

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = 150 ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l195_195496


namespace proportionality_cube_and_fourth_root_l195_195219

variables (x y z : ℝ) (k j m n : ℝ)

theorem proportionality_cube_and_fourth_root (h1 : x = k * y^3) (h2 : y = j * z^(1/4)) : 
  ∃ m : ℝ, ∃ n : ℝ, x = m * z^n ∧ n = 3/4 :=
by
  sorry

end proportionality_cube_and_fourth_root_l195_195219


namespace thomas_percentage_l195_195774

/-- 
Prove that if Emmanuel gets 100 jelly beans out of a total of 200 jelly beans, and 
Barry and Emmanuel share the remainder in a 4:5 ratio, then Thomas takes 10% 
of the jelly beans.
-/
theorem thomas_percentage (total_jelly_beans : ℕ) (emmanuel_jelly_beans : ℕ)
  (barry_ratio : ℕ) (emmanuel_ratio : ℕ) (thomas_percentage : ℕ) :
  total_jelly_beans = 200 → emmanuel_jelly_beans = 100 → barry_ratio = 4 → emmanuel_ratio = 5 →
  thomas_percentage = 10 :=
by
  intros;
  sorry

end thomas_percentage_l195_195774


namespace find_number_l195_195464

theorem find_number (x : ℝ) (h : 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30) : x = 66 :=
by
  sorry

end find_number_l195_195464


namespace intersection_point_of_diagonals_l195_195706

noncomputable def intersection_of_diagonals (k m b : Real) : Real × Real :=
  let A := (0, b)
  let B := (0, -b)
  let C := (2 * b / (k - m), 2 * b * k / (k - m) - b)
  let D := (-2 * b / (k - m), -2 * b * k / (k - m) + b)
  (0, 0)

theorem intersection_point_of_diagonals (k m b : Real) :
  intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_point_of_diagonals_l195_195706


namespace product_of_repeating_decimal_l195_195058

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l195_195058


namespace total_time_over_weekend_l195_195107

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end total_time_over_weekend_l195_195107


namespace probability_of_selecting_girl_l195_195165

theorem probability_of_selecting_girl (boys girls : ℕ) (total_students : ℕ) (prob : ℚ) 
  (h1 : boys = 3) 
  (h2 : girls = 2) 
  (h3 : total_students = boys + girls) 
  (h4 : prob = girls / total_students) : 
  prob = 2 / 5 := 
sorry

end probability_of_selecting_girl_l195_195165


namespace max_f_value_l195_195492

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ :=
  n / (n + 32) / (n + 2)

theorem max_f_value : ∀ n : ℕ, f n ≤ (1 / 50) :=
sorry

end max_f_value_l195_195492


namespace permutations_of_BANANA_l195_195154

theorem permutations_of_BANANA : 
  let word := ["B", "A", "N", "A", "N", "A"]
  let total_letters := 6
  let repeated_A := 3
  (total_letters.factorial / repeated_A.factorial) = 120 :=
by
  sorry

end permutations_of_BANANA_l195_195154


namespace joseph_cards_percentage_left_l195_195271

theorem joseph_cards_percentage_left (h1 : ℕ := 16) (h2 : ℚ := 3/8) (h3 : ℕ := 2) :
  ((h1 - (h2 * h1 + h3)) / h1 * 100) = 50 :=
by
  sorry

end joseph_cards_percentage_left_l195_195271


namespace quadratic_axis_of_symmetry_l195_195883

theorem quadratic_axis_of_symmetry (b c : ℝ) (h : -b / 2 = 3) : b = 6 :=
by
  sorry

end quadratic_axis_of_symmetry_l195_195883


namespace other_factor_computation_l195_195160

theorem other_factor_computation (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
  a = 11 → b = 43 → c = 2 → d = 31 → e = 1311 → 33 ∣ 363 →
  a * b * c * d * e = 38428986 :=
by
  intros ha hb hc hd he hdiv
  rw [ha, hb, hc, hd, he]
  -- proof steps go here if required
  sorry

end other_factor_computation_l195_195160


namespace horner_value_at_3_l195_195103

noncomputable def horner (x : ℝ) : ℝ :=
  ((((0.5 * x + 4) * x + 0) * x - 3) * x + 1) * x - 1

theorem horner_value_at_3 : horner 3 = 5.5 :=
by
  sorry

end horner_value_at_3_l195_195103


namespace manufacturing_cost_before_decrease_l195_195888

def original_manufacturing_cost (P : ℝ) (C_now : ℝ) (profit_rate_now : ℝ) : ℝ :=
  P - profit_rate_now * P

theorem manufacturing_cost_before_decrease
  (P : ℝ)
  (C_now : ℝ)
  (profit_rate_now : ℝ)
  (profit_rate_original : ℝ)
  (H1 : C_now = P - profit_rate_now * P)
  (H2 : profit_rate_now = 0.50)
  (H3 : profit_rate_original = 0.20)
  (H4 : C_now = 50) :
  original_manufacturing_cost P C_now profit_rate_now = 80 :=
sorry

end manufacturing_cost_before_decrease_l195_195888


namespace maximum_value_of_expression_l195_195595

theorem maximum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz * (x + y + z)) / ((x + y)^2 * (y + z)^2) ≤ (1 / 4) :=
sorry

end maximum_value_of_expression_l195_195595


namespace tilly_counts_total_stars_l195_195337

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end tilly_counts_total_stars_l195_195337


namespace neznaika_discrepancy_l195_195413

theorem neznaika_discrepancy :
  let KL := 1 -- Assume we start with 1 kiloluna
  let kg := 1 -- Assume we start with 1 kilogram
  let snayka_kg (KL : ℝ) := (KL / 4) * 0.96 -- Conversion rule from kilolunas to kilograms by Snayka
  let neznaika_kl (kg : ℝ) := (kg * 4) * 1.04 -- Conversion rule from kilograms to kilolunas by Neznaika
  let correct_kl (kg : ℝ) := kg / 0.24 -- Correct conversion from kilograms to kilolunas
  
  let result_kl := (neznaika_kl 1) -- Neznaika's computed kilolunas for 1 kilogram
  let correct_kl_val := (correct_kl 1) -- Correct kilolunas for 1 kilogram
  let ratio := result_kl / correct_kl_val -- Ratio of Neznaika's value to Correct value
  let discrepancy := 100 * (1 - ratio) -- Discrepancy percentage

  result_kl = 4.16 ∧ correct_kl_val = 4.1667 ∧ discrepancy = 0.16 := 
by
  sorry

end neznaika_discrepancy_l195_195413


namespace speed_of_faster_train_l195_195116

noncomputable def speed_of_slower_train_kmph := 36
def time_to_cross_seconds := 12
def length_of_faster_train_meters := 120

-- Speed of train V_f in kmph 
theorem speed_of_faster_train 
  (relative_speed_mps : ℝ := length_of_faster_train_meters / time_to_cross_seconds)
  (speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * (1000 / 3600))
  (speed_of_faster_train_mps : ℝ := relative_speed_mps + speed_of_slower_train_mps)
  (speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * (3600 / 1000) )
  : speed_of_faster_train_kmph = 72 := 
sorry

end speed_of_faster_train_l195_195116


namespace range_of_m_if_real_roots_specific_m_given_conditions_l195_195237

open Real

-- Define the quadratic equation and its conditions
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x ^ 2 - x + 2 * m - 4 = 0
def has_real_roots (m : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2

-- Proof that m ≤ 17/8 if the quadratic equation has real roots
theorem range_of_m_if_real_roots (m : ℝ) : has_real_roots m → m ≤ 17 / 8 := 
sorry

-- Define a condition on the roots
def roots_condition (x1 x2 m : ℝ) : Prop := (x1 - 3) * (x2 - 3) = m ^ 2 - 1

-- Proof of specific m when roots condition is given
theorem specific_m_given_conditions (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ roots_condition x1 x2 m) → m = -1 :=
sorry

end range_of_m_if_real_roots_specific_m_given_conditions_l195_195237


namespace sqrt_36_eq_6_l195_195376

theorem sqrt_36_eq_6 : Real.sqrt 36 = 6 := by
  sorry

end sqrt_36_eq_6_l195_195376


namespace hyperbola_solution_l195_195431

noncomputable def hyperbola_focus_parabola_equiv_hyperbola : Prop :=
  ∀ (a b c : ℝ),
    -- Condition 1: One focus of the hyperbola coincides with the focus of the parabola y^2 = 4sqrt(7)x
    (c^2 = a^2 + b^2) ∧ (c^2 = 7) →

    -- Condition 2: The hyperbola intersects the line y = x - 1 at points M and N
    (∃ M N : ℝ × ℝ, (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1^2 / a^2) - (M.2^2 / b^2) = 1) ∧ ((N.1^2 / a^2) - (N.2^2 / b^2) = 1)) →

    -- Condition 3: The x-coordinate of the midpoint of MN is -2/3
    (∀ M N : ℝ × ℝ, 
    (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1 + N.1) / 2 = -2/3)) →

    -- Conclusion: The standard equation of the hyperbola is x^2 / 2 - y^2 / 5 = 1
    a^2 = 2 ∧ b^2 = 5 ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → (x^2 / 2) - (y^2 / 5) = 1)

-- Proof omitted
theorem hyperbola_solution : hyperbola_focus_parabola_equiv_hyperbola :=
by sorry

end hyperbola_solution_l195_195431


namespace range_of_k_l195_195989

theorem range_of_k (a k : ℝ) : 
  (∀ x y : ℝ, y^2 - x * y + 2 * x + k = 0 → (x = a ∧ y = -a)) →
  k ≤ 1/2 :=
by sorry

end range_of_k_l195_195989


namespace function_periodicity_l195_195695

theorem function_periodicity
  (f : ℝ → ℝ)
  (H_odd : ∀ x, f (-x) = -f x)
  (H_even_shift : ∀ x, f (x + 2) = f (-x + 2))
  (H_val_neg1 : f (-1) = -1)
  : f 2017 + f 2016 = 1 := 
sorry

end function_periodicity_l195_195695


namespace find_y_values_l195_195112

theorem find_y_values (x : ℝ) (y : ℝ) 
  (h : x^2 + 4 * ((x / (x + 3))^2) = 64) : 
  y = (x + 3)^2 * (x - 2) / (2 * x + 3) → 
  y = 250 / 3 :=
sorry

end find_y_values_l195_195112


namespace gain_percent_l195_195136

-- Definitions for the problem
variables (MP CP SP : ℝ)
def cost_price := CP = 0.64 * MP
def selling_price := SP = 0.88 * MP

-- The statement to prove
theorem gain_percent (h1 : cost_price MP CP) (h2 : selling_price MP SP) :
  (SP - CP) / CP * 100 = 37.5 := 
sorry

end gain_percent_l195_195136


namespace closed_chain_possible_l195_195163

-- Define the angle constraint
def angle_constraint (θ : ℝ) : Prop :=
  θ ≥ 150

-- Define meshing condition between two gears
def meshed_gears (θ : ℝ) : Prop :=
  angle_constraint θ

-- Define the general condition for a closed chain of gears
def closed_chain (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → meshed_gears 150

theorem closed_chain_possible : closed_chain 61 :=
by sorry

end closed_chain_possible_l195_195163


namespace width_of_each_glass_pane_l195_195697

noncomputable def width_of_pane (num_panes : ℕ) (total_area : ℝ) (length_of_pane : ℝ) : ℝ :=
  total_area / num_panes / length_of_pane

theorem width_of_each_glass_pane :
  width_of_pane 8 768 12 = 8 := by
  sorry

end width_of_each_glass_pane_l195_195697


namespace min_m_n_sum_divisible_by_27_l195_195033

theorem min_m_n_sum_divisible_by_27 (m n : ℕ) (h : 180 * m * (n - 2) % 27 = 0) : m + n = 6 :=
sorry

end min_m_n_sum_divisible_by_27_l195_195033


namespace weekly_allowance_l195_195802

variable (A : ℝ)   -- declaring A as a real number

theorem weekly_allowance (h1 : (3/5 * A) + 1/3 * (2/5 * A) + 1 = A) : 
  A = 3.75 :=
sorry

end weekly_allowance_l195_195802


namespace cook_carrots_l195_195920

theorem cook_carrots :
  ∀ (total_carrots : ℕ) (fraction_used_before_lunch : ℚ) (carrots_not_used_end_of_day : ℕ),
    total_carrots = 300 →
    fraction_used_before_lunch = 2 / 5 →
    carrots_not_used_end_of_day = 72 →
    let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
    let carrots_after_lunch := total_carrots - carrots_used_before_lunch
    let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
    (carrots_used_end_of_day / carrots_after_lunch) = 3 / 5 :=
by
  intros total_carrots fraction_used_before_lunch carrots_not_used_end_of_day
  intros h1 h2 h3
  let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
  let carrots_after_lunch := total_carrots - carrots_used_before_lunch
  let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
  have h : carrots_used_end_of_day / carrots_after_lunch = 3 / 5 := sorry
  exact h

end cook_carrots_l195_195920


namespace school_students_l195_195121

theorem school_students
  (total_students : ℕ)
  (students_in_both : ℕ)
  (students_chemistry : ℕ)
  (students_biology : ℕ)
  (students_only_chemistry : ℕ)
  (students_only_biology : ℕ)
  (h1 : total_students = students_only_chemistry + students_only_biology + students_in_both)
  (h2 : students_chemistry = 3 * students_biology)
  (students_in_both_eq : students_in_both = 5)
  (total_students_eq : total_students = 43) :
  students_only_chemistry + students_in_both = 36 :=
by
  sorry

end school_students_l195_195121


namespace cube_volume_is_8_l195_195215

theorem cube_volume_is_8 (a : ℕ) 
  (h_cond : (a+2) * (a-2) * a = a^3 - 8) : 
  a^3 = 8 := 
by
  sorry

end cube_volume_is_8_l195_195215


namespace num_of_adults_l195_195586

def students : ℕ := 22
def vans : ℕ := 3
def capacity_per_van : ℕ := 8

theorem num_of_adults : (vans * capacity_per_van) - students = 2 := by
  sorry

end num_of_adults_l195_195586


namespace cos_675_eq_sqrt2_div_2_l195_195780

theorem cos_675_eq_sqrt2_div_2 : Real.cos (675 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by 
  sorry

end cos_675_eq_sqrt2_div_2_l195_195780


namespace solve_for_x_l195_195931

theorem solve_for_x (h_perimeter_square : ∀(s : ℝ), 4 * s = 64)
  (h_height_triangle : ∀(h : ℝ), h = 48)
  (h_area_equal : ∀(s h x : ℝ), s * s = 1/2 * h * x) : 
  x = 32 / 3 := by
  sorry

end solve_for_x_l195_195931


namespace min_value_of_A2_minus_B2_nonneg_l195_195028

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3)

theorem min_value_of_A2_minus_B2_nonneg (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z) ^ 2 - (B x y z) ^ 2 ≥ 36 :=
by
  sorry

end min_value_of_A2_minus_B2_nonneg_l195_195028


namespace max_score_top_three_teams_l195_195767

theorem max_score_top_three_teams : 
  ∀ (teams : Finset String) (points : String → ℕ), 
    teams.card = 6 →
    (∀ team, team ∈ teams → (points team = 0 ∨ points team = 1 ∨ points team = 3)) →
    ∃ top_teams : Finset String, top_teams.card = 3 ∧ 
    (∀ team, team ∈ top_teams → points team = 24) := 
by sorry

end max_score_top_three_teams_l195_195767


namespace line_equation_perpendicular_l195_195772

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem line_equation_perpendicular (c : ℝ) :
  (∃ k : ℝ, x - 2 * y + k = 0) ∧ is_perpendicular 2 1 1 (-2) → x - 2 * y - 3 = 0 := by
  sorry

end line_equation_perpendicular_l195_195772


namespace percentage_reduction_in_price_l195_195369

-- Definitions for the conditions in the problem
def reduced_price_per_kg : ℕ := 30
def extra_oil_obtained_kg : ℕ := 10
def total_money_spent : ℕ := 1500

-- Definition of the original price per kg of oil
def original_price_per_kg : ℕ := 75

-- Statement to prove the percentage reduction
theorem percentage_reduction_in_price : 
  (original_price_per_kg - reduced_price_per_kg) * 100 / original_price_per_kg = 60 := by
  sorry

end percentage_reduction_in_price_l195_195369


namespace positive_integer_condition_l195_195535

theorem positive_integer_condition (x : ℝ) (hx : x ≠ 0) : 
  (∃ (n : ℤ), n > 0 ∧ (abs (x - abs x + 2) / x) = n) ↔ x = 2 :=
by
  sorry

end positive_integer_condition_l195_195535


namespace quadratic_inequality_solution_l195_195486

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 6*x - 16 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 8} := by
  sorry

end quadratic_inequality_solution_l195_195486


namespace sum_of_fractions_l195_195282

theorem sum_of_fractions : 
  (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := 
by
  sorry

end sum_of_fractions_l195_195282


namespace find_c_l195_195657

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem find_c (a b c S : ℝ) (C : ℝ) 
  (ha : a = 3) 
  (hC : C = 120) 
  (hS : S = 15 * Real.sqrt 3 / 4) 
  (hab : a * b = 15)
  (hc2 : c^2 = a^2 + b^2 - 2 * a * b * cos_deg C) :
  c = 7 :=
by 
  sorry

end find_c_l195_195657


namespace smallest_m_inequality_l195_195043

theorem smallest_m_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1) : 27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l195_195043


namespace boat_stream_speeds_l195_195736

variable (x y : ℝ)

theorem boat_stream_speeds (h1 : 20 + x ≠ 0) (h2 : 40 - y ≠ 0) :
  380 = 7 * x + 13 * y ↔ 
  26 * (40 - y) = 14 * (20 + x) :=
by { sorry }

end boat_stream_speeds_l195_195736


namespace matt_twice_james_age_in_5_years_l195_195885

theorem matt_twice_james_age_in_5_years :
  (∃ x : ℕ, (3 + 27 = 30) ∧ (Matt_current_age = 65) ∧ 
  (Matt_age_in_x_years = Matt_current_age + x) ∧ 
  (James_age_in_x_years = James_current_age + x) ∧ 
  (Matt_age_in_x_years = 2 * James_age_in_x_years) → x = 5) :=
sorry

end matt_twice_james_age_in_5_years_l195_195885


namespace matrix_vector_computation_l195_195775

-- Setup vectors and their corresponding matrix multiplication results
variables {R : Type*} [Field R]
variables {M : Matrix (Fin 2) (Fin 2) R} {u z : Fin 2 → R}

-- Conditions given in (a)
def condition1 : M.mulVec u = ![3, -4] :=
  sorry

def condition2 : M.mulVec z = ![-1, 6] :=
  sorry

-- Statement equivalent to the proof problem given in (c)
theorem matrix_vector_computation :
  M.mulVec (3 • u - 2 • z) = ![11, -24] :=
by
  -- Use the conditions to prove the theorem
  sorry

end matrix_vector_computation_l195_195775


namespace germination_probability_l195_195236

open Nat

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_of_success (p : ℚ) (k : ℕ) (n : ℕ) : ℚ :=
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem germination_probability :
  probability_of_success 0.9 5 7 = 0.124 := by
  sorry

end germination_probability_l195_195236


namespace nina_age_l195_195841

theorem nina_age : ∀ (M L A N : ℕ), 
  (M = L - 5) → 
  (L = A + 6) → 
  (N = A + 2) → 
  (M = 16) → 
  N = 17 :=
by
  intros M L A N h1 h2 h3 h4
  sorry

end nina_age_l195_195841


namespace calculate_expression_l195_195428

def inequality_holds (a b c d x : ℝ) : Prop :=
  (x - a) * (x - b) * (x - d) / (x - c) ≥ 0

theorem calculate_expression : 
  ∀ (a b c d : ℝ),
    a < b ∧ b < d ∧
    (∀ x : ℝ, 
      (inequality_holds a b c d x ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32))) →
    a + 2 * b + 3 * c + 4 * d = 160 :=
sorry

end calculate_expression_l195_195428


namespace circle_rational_points_l195_195227

theorem circle_rational_points :
  ( ∃ B : ℚ × ℚ, ∀ k : ℚ, B ∈ {p | p.1 ^ 2 + 2 * p.1 + p.2 ^ 2 = 1992} ) ∧ 
  ( (42 : ℤ)^2 + 2 * 42 + 12^2 = 1992 ) :=
by
  sorry

end circle_rational_points_l195_195227


namespace sector_central_angle_l195_195741

noncomputable def sector_angle (R L : ℝ) : ℝ := L / R

theorem sector_central_angle :
  ∃ R L : ℝ, 
    (R > 0) ∧ 
    (L > 0) ∧ 
    (1 / 2 * L * R = 5) ∧ 
    (2 * R + L = 9) ∧ 
    (sector_angle R L = 8 / 5 ∨ sector_angle R L = 5 / 2) :=
sorry

end sector_central_angle_l195_195741


namespace fraction_value_l195_195098

theorem fraction_value (a : ℕ) (h : a > 0) (h_eq : (a:ℝ) / (a + 35) = 0.7) : a = 82 :=
by
  -- Steps to prove the theorem here
  sorry

end fraction_value_l195_195098


namespace ratio_of_u_to_v_l195_195739

theorem ratio_of_u_to_v (b : ℚ) (hb : b ≠ 0) (u v : ℚ)
  (hu : u = -b / 8) (hv : v = -b / 12) :
  u / v = 3 / 2 :=
by sorry

end ratio_of_u_to_v_l195_195739


namespace problem_solution_l195_195806

theorem problem_solution (k : ℕ) (hk : k ≥ 2) : 
  (∀ m n : ℕ, 1 ≤ m ∧ m ≤ k → 1 ≤ n ∧ n ≤ k → m ≠ n → ¬ k ∣ (n^(n-1) - m^(m-1))) ↔ (k = 2 ∨ k = 3) :=
by
  sorry

end problem_solution_l195_195806


namespace car_speed_is_100_l195_195646

def avg_speed (d1 d2 t: ℕ) := (d1 + d2) / t = 80

theorem car_speed_is_100 
  (x : ℕ)
  (speed_second_hour : ℕ := 60)
  (total_time : ℕ := 2)
  (h : avg_speed x speed_second_hour total_time):
  x = 100 :=
by
  unfold avg_speed at h
  sorry

end car_speed_is_100_l195_195646


namespace tangent_line_l195_195197

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line (x y : ℝ) (h : f 2 = 6) : 13 * x - y - 20 = 0 :=
by
  -- Insert proof here
  sorry

end tangent_line_l195_195197


namespace ratio_of_areas_l195_195764

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C M : V}

-- Define the collinearity condition point M in the triangle plane with respect to vectors AB and AC
def point_condition (A B C M : V) : Prop :=
  5 • (M - A) = (B - A) + 3 • (C - A)

-- Define an area ratio function
def area_ratio_triangles (A B C M : V) [AddCommGroup V] [Module ℝ V] : ℝ :=
  sorry  -- Implementation of area ratio comparison, abstracted out for the given problem statement

-- The theorem to prove
theorem ratio_of_areas (hM : point_condition A B C M) : area_ratio_triangles A B C M = 3 / 5 :=
sorry

end ratio_of_areas_l195_195764


namespace polynomial_coeff_sum_l195_195368

theorem polynomial_coeff_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) (h : (2 * x - 3) ^ 5 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 160 :=
sorry

end polynomial_coeff_sum_l195_195368


namespace maria_cartons_needed_l195_195687

theorem maria_cartons_needed : 
  ∀ (total_needed strawberries blueberries raspberries blackberries : ℕ), 
  total_needed = 36 →
  strawberries = 4 →
  blueberries = 8 →
  raspberries = 3 →
  blackberries = 5 →
  (total_needed - (strawberries + blueberries + raspberries + blackberries) = 16) :=
by
  intros total_needed strawberries blueberries raspberries blackberries ht hs hb hr hb
  -- ... the proof would go here
  sorry

end maria_cartons_needed_l195_195687


namespace total_guitars_l195_195494

theorem total_guitars (Barbeck_guitars Steve_guitars Davey_guitars : ℕ) (h1 : Barbeck_guitars = 2 * Steve_guitars) (h2 : Davey_guitars = 3 * Barbeck_guitars) (h3 : Davey_guitars = 18) : Barbeck_guitars + Steve_guitars + Davey_guitars = 27 :=
by sorry

end total_guitars_l195_195494


namespace ordered_pairs_unique_solution_l195_195952

theorem ordered_pairs_unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 :=
by
  sorry

end ordered_pairs_unique_solution_l195_195952


namespace new_students_admitted_l195_195756

-- Definitions of the conditions
def original_students := 35
def increase_in_expenses := 42
def decrease_in_average_expense := 1
def original_expenditure := 420

-- Main statement: proving the number of new students admitted
theorem new_students_admitted : ∃ x : ℕ, 
  (original_expenditure + increase_in_expenses = 11 * (original_students + x)) ∧ 
  (x = 7) := 
sorry

end new_students_admitted_l195_195756


namespace remainder_of_P_div_by_D_is_333_l195_195295

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 8 * x^4 - 18 * x^3 + 27 * x^2 - 14 * x - 30

-- Define the divisor D(x) and simplify it, but this is not necessary for the theorem statement.
-- def D (x : ℝ) : ℝ := 4 * x - 12  

-- Prove the remainder is 333 when x = 3
theorem remainder_of_P_div_by_D_is_333 : P 3 = 333 := by
  sorry

end remainder_of_P_div_by_D_is_333_l195_195295


namespace perfect_square_trinomial_l195_195466

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l195_195466


namespace smallest_possible_b_l195_195604

theorem smallest_possible_b
  (a c b : ℤ)
  (h1 : a < c)
  (h2 : c < b)
  (h3 : c = (a + b) / 2)
  (h4 : b^2 / c = a) :
  b = 2 :=
sorry

end smallest_possible_b_l195_195604


namespace senior_discount_percentage_l195_195378

theorem senior_discount_percentage 
    (cost_shorts : ℕ)
    (count_shorts : ℕ)
    (cost_shirts : ℕ)
    (count_shirts : ℕ)
    (amount_paid : ℕ)
    (total_cost : ℕ := (cost_shorts * count_shorts) + (cost_shirts * count_shirts))
    (discount_received : ℕ := total_cost - amount_paid)
    (discount_percentage : ℚ := (discount_received : ℚ) / total_cost * 100) :
    count_shorts = 3 ∧ cost_shorts = 15 ∧ count_shirts = 5 ∧ cost_shirts = 17 ∧ amount_paid = 117 →
    discount_percentage = 10 := 
by
    sorry

end senior_discount_percentage_l195_195378


namespace smallest_integer_conditions_l195_195523

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Definition of having a prime factor less than a given number
def has_prime_factor_less_than (n k : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p < k

-- Problem statement
theorem smallest_integer_conditions :
  ∃ n : ℕ, n > 0 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ ¬ has_prime_factor_less_than n 60 ∧ ∀ m : ℕ, (m > 0 ∧ ¬ is_prime m ∧ ¬ is_square m ∧ ¬ has_prime_factor_less_than m 60) → n ≤ m :=
  sorry

end smallest_integer_conditions_l195_195523


namespace four_digit_number_l195_195211

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end four_digit_number_l195_195211


namespace average_salary_all_workers_l195_195960

theorem average_salary_all_workers 
  (n : ℕ) (avg_salary_technicians avg_salary_rest total_avg_salary : ℝ)
  (h1 : n = 7) 
  (h2 : avg_salary_technicians = 8000) 
  (h3 : avg_salary_rest = 6000)
  (h4 : total_avg_salary = avg_salary_technicians) : 
  total_avg_salary = 8000 :=
by sorry

end average_salary_all_workers_l195_195960


namespace train_length_l195_195521

theorem train_length (L : ℝ) 
  (h1 : (L / 20) = ((L + 1500) / 70)) : L = 600 := by
  sorry

end train_length_l195_195521


namespace pens_sold_l195_195915

theorem pens_sold (C : ℝ) (N : ℝ) (h_gain : 22 * C = 0.25 * N * C) : N = 88 :=
by {
  sorry
}

end pens_sold_l195_195915


namespace sin_double_angle_l195_195648

theorem sin_double_angle (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin (2 * x) = 3 / 5 := 
by 
  sorry

end sin_double_angle_l195_195648


namespace minimum_experiments_fractional_method_l195_195785

/--
A pharmaceutical company needs to optimize the cultivation temperature for a certain medicinal liquid through bioassay.
The experimental range is set from 29℃ to 63℃, with an accuracy requirement of ±1℃.
Prove that the minimum number of experiments required to ensure the best cultivation temperature is found using the fractional method is 7.
-/
theorem minimum_experiments_fractional_method
  (range_start : ℕ)
  (range_end : ℕ)
  (accuracy : ℕ)
  (fractional_method : ∀ (range_start range_end accuracy: ℕ), ℕ) :
  range_start = 29 → range_end = 63 → accuracy = 1 → fractional_method range_start range_end accuracy = 7 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end minimum_experiments_fractional_method_l195_195785


namespace find_a5_l195_195123

noncomputable def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
a₁ + (n - 1) * d

theorem find_a5 (a₁ d : ℚ) (h₁ : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 5 - arithmetic_sequence a₁ d 8 = 1)
(h₂ : arithmetic_sequence a₁ d 9 - arithmetic_sequence a₁ d 2 = 5) :
arithmetic_sequence a₁ d 5 = 6 :=
sorry

end find_a5_l195_195123


namespace proof_mn_squared_l195_195240

theorem proof_mn_squared (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end proof_mn_squared_l195_195240


namespace triangle_inequality_inequality_equality_condition_l195_195074

variable (a b c : ℝ)

-- indicating triangle inequality conditions
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_inequality_equality_condition_l195_195074


namespace least_positive_integer_n_l195_195991

theorem least_positive_integer_n : ∃ (n : ℕ), (1 / (n : ℝ) - 1 / (n + 1) < 1 / 100) ∧ ∀ m, m < n → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 100) :=
sorry

end least_positive_integer_n_l195_195991


namespace students_wrote_word_correctly_l195_195424

-- Definitions based on the problem conditions
def total_students := 50
def num_cat := 10
def num_rat := 18
def num_croc := total_students - num_cat - num_rat
def correct_cat := 15
def correct_rat := 15
def correct_total := correct_cat + correct_rat

-- Question: How many students wrote their word correctly?
-- Correct Answer: 8

theorem students_wrote_word_correctly : 
  num_cat + num_rat + num_croc = total_students 
  → correct_cat = 15 
  → correct_rat = 15 
  → correct_total = 30 
  → ∀ (num_correct_words : ℕ), num_correct_words = correct_total - num_croc 
  → num_correct_words = 8 := by 
  sorry

end students_wrote_word_correctly_l195_195424


namespace max_x_plus_y_l195_195810

theorem max_x_plus_y (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : x + y ≤ 2 * Real.sqrt (3) / 3 :=
sorry

end max_x_plus_y_l195_195810


namespace sum_cubes_first_39_eq_608400_l195_195656

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

theorem sum_cubes_first_39_eq_608400 : sum_of_cubes 39 = 608400 :=
by
  sorry

end sum_cubes_first_39_eq_608400_l195_195656


namespace jeopardy_episode_length_l195_195423

-- Definitions based on the conditions
def num_episodes_jeopardy : ℕ := 2
def num_episodes_wheel : ℕ := 2
def wheel_twice_jeopardy (J : ℝ) : ℝ := 2 * J
def total_time_watched : ℝ := 120 -- in minutes

-- Condition stating the total time watched in terms of J
def total_watching_time_formula (J : ℝ) : ℝ :=
  num_episodes_jeopardy * J + num_episodes_wheel * (wheel_twice_jeopardy J)

theorem jeopardy_episode_length : ∃ J : ℝ, total_watching_time_formula J = total_time_watched ∧ J = 20 :=
by
  use 20
  simp [total_watching_time_formula, wheel_twice_jeopardy, num_episodes_jeopardy, num_episodes_wheel, total_time_watched]
  sorry

end jeopardy_episode_length_l195_195423


namespace a5_b3_c_divisible_by_6_l195_195468

theorem a5_b3_c_divisible_by_6 (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) :=
by
  sorry

end a5_b3_c_divisible_by_6_l195_195468


namespace total_vertical_distance_of_rings_l195_195642

theorem total_vertical_distance_of_rings :
  let thickness := 2
  let top_outside_diameter := 20
  let bottom_outside_diameter := 4
  let n := (top_outside_diameter - bottom_outside_diameter) / thickness + 1
  let total_distance := n * thickness
  total_distance + thickness = 76 :=
by
  sorry

end total_vertical_distance_of_rings_l195_195642


namespace max_value_of_f_l195_195456

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : 
  ∃ x : ℝ, f x = 6 / 5 ∧ ∀ y : ℝ, f y ≤ 6 / 5 :=
sorry

end max_value_of_f_l195_195456


namespace find_u_plus_v_l195_195572

variables (u v : ℚ)

theorem find_u_plus_v (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : u + v = 27 / 43 := by
  sorry

end find_u_plus_v_l195_195572


namespace apples_total_l195_195312

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l195_195312


namespace chocolate_game_winner_l195_195207

theorem chocolate_game_winner (m n : ℕ) (h_m : m = 6) (h_n : n = 8) :
  (∃ k : ℕ, (48 - 1) - 2 * k = 0) ↔ true :=
by
  sorry

end chocolate_game_winner_l195_195207


namespace marks_in_physics_l195_195626

def marks_in_english : ℝ := 74
def marks_in_mathematics : ℝ := 65
def marks_in_chemistry : ℝ := 67
def marks_in_biology : ℝ := 90
def average_marks : ℝ := 75.6
def number_of_subjects : ℕ := 5

-- We need to show that David's marks in Physics are 82.
theorem marks_in_physics : ∃ (P : ℝ), P = 82 ∧ 
  ((marks_in_english + marks_in_mathematics + P + marks_in_chemistry + marks_in_biology) / number_of_subjects = average_marks) :=
by sorry

end marks_in_physics_l195_195626


namespace rent_fraction_l195_195481

theorem rent_fraction (B R : ℝ) 
  (food_and_beverages_spent : (1 / 4) * (1 - R) * B = 0.1875 * B) : 
  R = 0.25 :=
by
  -- proof skipped
  sorry

end rent_fraction_l195_195481


namespace clerical_percentage_after_reduction_l195_195475

theorem clerical_percentage_after_reduction
  (total_employees : ℕ)
  (clerical_fraction : ℚ)
  (reduction_fraction : ℚ)
  (h1 : total_employees = 3600)
  (h2 : clerical_fraction = 1/4)
  (h3 : reduction_fraction = 1/4) : 
  let initial_clerical := clerical_fraction * total_employees
  let reduced_clerical := (1 - reduction_fraction) * initial_clerical
  let let_go := initial_clerical - reduced_clerical
  let new_total := total_employees - let_go
  let clerical_percentage := (reduced_clerical / new_total) * 100
  clerical_percentage = 20 :=
by sorry

end clerical_percentage_after_reduction_l195_195475


namespace average_employees_per_week_l195_195373

-- Define the number of employees hired each week
variables (x : ℕ)
noncomputable def employees_first_week := x + 200
noncomputable def employees_second_week := x
noncomputable def employees_third_week := x + 150
noncomputable def employees_fourth_week := 400

-- Given conditions as hypotheses
axiom h1 : employees_third_week / 2 = employees_fourth_week / 2
axiom h2 : employees_fourth_week = 400

-- Prove the average number of employees hired per week is 225
theorem average_employees_per_week :
  (employees_first_week + employees_second_week + employees_third_week + employees_fourth_week) / 4 = 225 :=
by
  sorry

end average_employees_per_week_l195_195373


namespace base5_addition_correct_l195_195302

-- Definitions to interpret base-5 numbers
def base5_to_base10 (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc d => acc * 5 + d) 0

-- Conditions given in the problem
def num1 : ℕ := base5_to_base10 [2, 0, 1, 4]  -- (2014)_5 in base-10
def num2 : ℕ := base5_to_base10 [2, 2, 3]    -- (223)_5 in base-10

-- Statement to prove
theorem base5_addition_correct :
  base5_to_base10 ([2, 0, 1, 4]) + base5_to_base10 ([2, 2, 3]) = base5_to_base10 ([2, 2, 4, 2]) :=
by
  -- Proof goes here
  sorry

#print axioms base5_addition_correct

end base5_addition_correct_l195_195302


namespace solve_for_x_l195_195405

theorem solve_for_x (x y : ℝ) (h : 3 * x - 4 * y = 5) : x = (1 / 3) * (5 + 4 * y) :=
  sorry

end solve_for_x_l195_195405


namespace scientific_notation_chip_gate_width_l195_195375

theorem scientific_notation_chip_gate_width :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_chip_gate_width_l195_195375


namespace greatest_possible_sum_of_10_integers_l195_195803

theorem greatest_possible_sum_of_10_integers (a b c d e f g h i j : ℕ) 
  (h_prod : a * b * c * d * e * f * g * h * i * j = 1024) : 
  a + b + c + d + e + f + g + h + i + j ≤ 1033 :=
sorry

end greatest_possible_sum_of_10_integers_l195_195803


namespace compute_4_star_3_l195_195740

def custom_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem compute_4_star_3 : custom_op 4 3 = 13 :=
by
  sorry

end compute_4_star_3_l195_195740


namespace minimum_value_7a_4b_l195_195862

noncomputable def original_cond (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4

theorem minimum_value_7a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  original_cond a b ha hb → 7 * a + 4 * b = 9 / 4 :=
by
  sorry

end minimum_value_7a_4b_l195_195862


namespace smallest_positive_n_l195_195142

theorem smallest_positive_n (x y z : ℕ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^13) :=
by
  sorry

end smallest_positive_n_l195_195142


namespace simple_and_compound_interest_difference_l195_195021

theorem simple_and_compound_interest_difference (r : ℝ) :
  let P := 3600
  let t := 2
  let SI := P * r * t / 100
  let CI := P * (1 + r / 100)^t - P
  CI - SI = 225 → r = 25 := by
  intros
  sorry

end simple_and_compound_interest_difference_l195_195021


namespace simplify_and_evaluate_expression_l195_195241

theorem simplify_and_evaluate_expression (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
    (x ≠ 1) → (x ≠ -1) → (x ≠ 0) → 
    ((x / (x + 1) - (3 * x) / (x - 1)) / (x / (x^2 - 1))) = -8 :=
by 
  intro h3 h4 h5
  sorry

end simplify_and_evaluate_expression_l195_195241


namespace domain_of_function_l195_195553

noncomputable def is_defined (x : ℝ) : Prop :=
  (x + 4 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_function :
  ∀ x : ℝ, is_defined x ↔ x ≥ -4 ∧ x ≠ 0 :=
by
  sorry

end domain_of_function_l195_195553


namespace log_constant_expression_l195_195385

theorem log_constant_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hcond : x^2 + y^2 = 18 * x * y) :
  ∃ k : ℝ, (Real.log (x - y) / Real.log (Real.sqrt 2) - (1 / 2) * (Real.log x / Real.log (Real.sqrt 2) + Real.log y / Real.log (Real.sqrt 2))) = k :=
sorry

end log_constant_expression_l195_195385


namespace percentage_of_girls_who_like_basketball_l195_195106

theorem percentage_of_girls_who_like_basketball 
  (total_students : ℕ)
  (percentage_girls : ℝ)
  (percentage_boys_basketball : ℝ)
  (factor_girls_to_boys_not_basketball : ℝ)
  (total_students_eq : total_students = 25)
  (percentage_girls_eq : percentage_girls = 0.60)
  (percentage_boys_basketball_eq : percentage_boys_basketball = 0.40)
  (factor_girls_to_boys_not_basketball_eq : factor_girls_to_boys_not_basketball = 2) 
  : 
  ((factor_girls_to_boys_not_basketball * (total_students * (1 - percentage_girls) * (1 - percentage_boys_basketball))) / 
  (total_students * percentage_girls)) * 100 = 80 :=
by
  sorry

end percentage_of_girls_who_like_basketball_l195_195106


namespace regular_tire_price_l195_195084

theorem regular_tire_price 
  (x : ℝ) 
  (h1 : 3 * x + x / 2 = 300) 
  : x = 600 / 7 := 
sorry

end regular_tire_price_l195_195084


namespace smallest_area_of_right_triangle_l195_195089

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l195_195089


namespace find_initial_pomelos_l195_195856

theorem find_initial_pomelos (g w w' g' : ℕ) 
  (h1 : w = 3 * g)
  (h2 : w' = w - 90)
  (h3 : g' = g - 60)
  (h4 : w' = 4 * g' - 26) 
  : g = 176 :=
by
  sorry

end find_initial_pomelos_l195_195856


namespace number_of_valid_sequences_l195_195343

/--
The measures of the interior angles of a convex pentagon form an increasing arithmetic sequence.
Determine the number of such sequences possible if the pentagon is not equiangular, all of the angle
degree measures are positive integers less than 150 degrees, and the smallest angle is at least 60 degrees.
-/

theorem number_of_valid_sequences : ∃ n : ℕ, n = 5 ∧
  ∀ (x d : ℕ),
  x ≥ 60 ∧ x + 4 * d < 150 ∧ 5 * x + 10 * d = 540 ∧ (x + d ≠ x + 2 * d) := 
sorry

end number_of_valid_sequences_l195_195343


namespace part1_part2_l195_195966

def star (a b c d : ℝ) : ℝ := a * c - b * d

-- Part (1)
theorem part1 : star (-4) 3 2 (-6) = 10 := by
  sorry

-- Part (2)
theorem part2 (m : ℝ) (h : ∀ x : ℝ, star x (2 * x - 1) (m * x + 1) m = 0 → (m ≠ 0 → (((1 - 2 * m) ^ 2 - 4 * m * m) ≥ 0))) :
  (m ≤ 1 / 4 ∨ m < 0) ∧ m ≠ 0 := by
  sorry

end part1_part2_l195_195966


namespace transition_to_modern_population_reproduction_l195_195943

-- Defining the conditions as individual propositions
def A : Prop := ∃ (m b : ℝ), m < 0 ∧ b = 0
def B : Prop := ∃ (m b : ℝ), m < 0 ∧ b < 0
def C : Prop := ∃ (m b : ℝ), m > 0 ∧ b = 0
def D : Prop := ∃ (m b : ℝ), m > 0 ∧ b > 0

-- Defining the question as a property marking the transition from traditional to modern types of population reproduction
def Q : Prop := B

-- The proof problem
theorem transition_to_modern_population_reproduction :
  Q = B :=
by
  sorry

end transition_to_modern_population_reproduction_l195_195943


namespace running_time_of_BeastOfWar_is_100_l195_195320

noncomputable def Millennium := 120  -- minutes
noncomputable def AlphaEpsilon := Millennium - 30  -- minutes
noncomputable def BeastOfWar := AlphaEpsilon + 10  -- minutes
noncomputable def DeltaSquadron := 2 * BeastOfWar  -- minutes

theorem running_time_of_BeastOfWar_is_100 :
  BeastOfWar = 100 :=
by
  -- Proof goes here
  sorry

end running_time_of_BeastOfWar_is_100_l195_195320


namespace equivalent_shaded_areas_l195_195131

/- 
  Definitions and parameters:
  - l_sq: the side length of the larger square.
  - s_sq: the side length of the smaller square.
-/
variables (l_sq s_sq : ℝ)
  
-- The area of the larger square
def area_larger_square : ℝ := l_sq * l_sq
  
-- The area of the smaller square
def area_smaller_square : ℝ := s_sq * s_sq
  
-- The shaded area in diagram i
def shaded_area_diagram_i : ℝ := area_larger_square l_sq - area_smaller_square s_sq

-- The polygonal areas in diagrams ii and iii
variables (polygon_area_ii polygon_area_iii : ℝ)

-- The theorem to prove the equivalence of the areas
theorem equivalent_shaded_areas :
  polygon_area_ii = shaded_area_diagram_i l_sq s_sq ∧ polygon_area_iii = shaded_area_diagram_i l_sq s_sq :=
sorry

end equivalent_shaded_areas_l195_195131


namespace yellow_percentage_l195_195955

theorem yellow_percentage (s w : ℝ) 
  (h_cross : w * w + 4 * w * (s - 2 * w) = 0.49 * s * s) : 
  (w / s) ^ 2 = 0.2514 :=
by
  sorry

end yellow_percentage_l195_195955


namespace inequality_correct_l195_195335

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c :=
sorry

end inequality_correct_l195_195335


namespace range_of_function_l195_195732

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ y, y = sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x)) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l195_195732


namespace problem1_problem2_l195_195272

-- Problem 1
theorem problem1 (x : ℚ) (h : x = -1/3) : 6 * x^2 + 5 * x^2 - 2 * (3 * x - 2 * x^2) = 11 / 3 :=
by sorry

-- Problem 2
theorem problem2 (a b : ℚ) (ha : a = -2) (hb : b = -1) : 5 * a^2 - a * b - 2 * (3 * a * b - (a * b - 2 * a^2)) = -6 :=
by sorry

end problem1_problem2_l195_195272


namespace profit_from_ad_l195_195301

def advertising_cost : ℝ := 1000
def customers : ℕ := 100
def purchase_rate : ℝ := 0.8
def purchase_price : ℝ := 25

theorem profit_from_ad (advertising_cost customers purchase_rate purchase_price : ℝ) : 
  (customers * purchase_rate * purchase_price - advertising_cost) = 1000 :=
by
  -- assumptions as conditions
  let bought_customers := (customers : ℝ) * purchase_rate
  let revenue := bought_customers * purchase_price
  let profit := revenue - advertising_cost
  -- state the proof goal
  have goal : profit = 1000 :=
    sorry
  exact goal

end profit_from_ad_l195_195301


namespace companion_value_4164_smallest_N_satisfies_conditions_l195_195570

-- Define relevant functions
def G (N : ℕ) : ℕ :=
  let digits := [N / 1000 % 10, N / 100 % 10, N / 10 % 10, N % 10]
  digits.sum

def P (N : ℕ) : ℕ :=
  (N / 1000 % 10) * (N / 100 % 10)

def Q (N : ℕ) : ℕ :=
  (N / 10 % 10) * (N % 10)

def companion_value (N : ℕ) : ℚ :=
  |(G N : ℤ) / ((P N : ℤ) - (Q N : ℤ))|

-- Proof problem for part (1)
theorem companion_value_4164 : companion_value 4164 = 3 / 4 := sorry

-- Proof problem for part (2)
theorem smallest_N_satisfies_conditions :
  ∀ (N : ℕ), N > 1000 ∧ N < 10000 ∧ (∀ d, N / 10^d % 10 ≠ 0) ∧ (N / 1000 % 10 + N % 10) % 9 = 0 ∧ G N = 16 ∧ companion_value N = 4 → N = 2527 := sorry

end companion_value_4164_smallest_N_satisfies_conditions_l195_195570


namespace cylinder_base_radii_l195_195912

theorem cylinder_base_radii {l w : ℝ} (hl : l = 3 * Real.pi) (hw : w = Real.pi) :
  (∃ r : ℝ, l = 2 * Real.pi * r ∧ r = 3 / 2) ∨ (∃ r : ℝ, w = 2 * Real.pi * r ∧ r = 1 / 2) :=
sorry

end cylinder_base_radii_l195_195912


namespace correct_average_calculation_l195_195133

theorem correct_average_calculation (n : ℕ) (incorrect_avg correct_num wrong_num : ℕ) (incorrect_avg_eq : incorrect_avg = 21) (n_eq : n = 10) (correct_num_eq : correct_num = 36) (wrong_num_eq : wrong_num = 26) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 22 := by
  sorry

end correct_average_calculation_l195_195133


namespace students_opted_both_math_science_l195_195267

def total_students : ℕ := 40
def not_opted_math : ℕ := 10
def not_opted_science : ℕ := 15
def not_opted_either : ℕ := 2

theorem students_opted_both_math_science :
  let T := total_students
  let M' := not_opted_math
  let S' := not_opted_science
  let E := not_opted_either
  let B := (T - M') + (T - S') - (T - E)
  B = 17 :=
by
  sorry

end students_opted_both_math_science_l195_195267


namespace sequences_count_n3_sequences_count_n6_sequences_count_n9_l195_195410

inductive Shape
  | triangle
  | square
  | rectangle (k : ℕ)

open Shape

def transition (s : Shape) : List Shape :=
  match s with
  | triangle => [triangle, square]
  | square => [rectangle 1]
  | rectangle k =>
    if k = 0 then [rectangle 1] else [rectangle (k - 1), rectangle (k + 1)]

def count_sequences (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (shapes : List Shape) : ℕ :=
    if m = 0 then shapes.length
    else
      let next_shapes := shapes.bind transition
      aux (m - 1) next_shapes
  aux n [square]

theorem sequences_count_n3 : count_sequences 3 = 5 :=
  by sorry

theorem sequences_count_n6 : count_sequences 6 = 24 :=
  by sorry

theorem sequences_count_n9 : count_sequences 9 = 149 :=
  by sorry

end sequences_count_n3_sequences_count_n6_sequences_count_n9_l195_195410


namespace find_m2n_plus_mn2_minus_mn_l195_195355

def quadratic_roots (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0

theorem find_m2n_plus_mn2_minus_mn :
  ∃ m n : ℝ, quadratic_roots 1 2015 (-1) m n ∧ m^2 * n + m * n^2 - m * n = 2016 :=
by
  sorry

end find_m2n_plus_mn2_minus_mn_l195_195355


namespace odd_numbers_square_division_l195_195527

theorem odd_numbers_square_division (m n : ℤ) (hm : Odd m) (hn : Odd n) (h : m^2 - n^2 + 1 ∣ n^2 - 1) : ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := 
sorry

end odd_numbers_square_division_l195_195527


namespace Jane_buys_three_bagels_l195_195635

theorem Jane_buys_three_bagels (b m c : ℕ) (h1 : b + m + c = 5) (h2 : 80 * b + 60 * m + 100 * c = 400) : b = 3 := 
sorry

end Jane_buys_three_bagels_l195_195635


namespace man_total_earnings_l195_195323

-- Define the conditions
def total_days := 30
def wage_per_day := 10
def fine_per_absence := 2
def days_absent := 7
def days_worked := total_days - days_absent
def earned := days_worked * wage_per_day
def fine := days_absent * fine_per_absence
def total_earnings := earned - fine

-- State the theorem
theorem man_total_earnings : total_earnings = 216 := by
  -- Using the definitions provided, the proof should show that the calculations result in 216
  sorry

end man_total_earnings_l195_195323


namespace simplify_fraction_l195_195409

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end simplify_fraction_l195_195409


namespace case1_case2_case3_l195_195430

-- Definitions from conditions
def tens_digit_one : ℕ := sorry
def units_digit_one : ℕ := sorry
def units_digit_two : ℕ := sorry
def tens_digit_two : ℕ := sorry
def sum_units_digits_ten : Prop := units_digit_one + units_digit_two = 10
def same_digit : ℕ := sorry
def sum_tens_digits_ten : Prop := tens_digit_one + tens_digit_two = 10

-- The proof problems
theorem case1 (A B D : ℕ) (hBplusD : B + D = 10) :
  (10 * A + B) * (10 * A + D) = 100 * (A^2 + A) + B * D :=
sorry

theorem case2 (A B C : ℕ) (hAplusC : A + C = 10) :
  (10 * A + B) * (10 * C + B) = 100 * A * C + 100 * B + B^2 :=
sorry

theorem case3 (A B C : ℕ) (hAplusB : A + B = 10) :
  (10 * A + B) * (10 * C + C) = 100 * A * C + 100 * C + B * C :=
sorry

end case1_case2_case3_l195_195430


namespace lizette_third_quiz_score_l195_195829

theorem lizette_third_quiz_score :
  ∀ (x : ℕ),
  (2 * 95 + x) / 3 = 94 → x = 92 :=
by
  intro x h
  have h1 : 2 * 95 = 190 := by norm_num
  have h2 : 3 * 94 = 282 := by norm_num
  sorry

end lizette_third_quiz_score_l195_195829


namespace min_value_expression_l195_195562

theorem min_value_expression (a b : ℝ) (h1 : 2 * a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a) + ((1 - b) / b) = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_expression_l195_195562


namespace Irene_hours_worked_l195_195388

open Nat

theorem Irene_hours_worked (x totalHours : ℕ) : 
  (500 + 20 * x = 700) → 
  (totalHours = 40 + x) → 
  totalHours = 50 :=
by
  sorry

end Irene_hours_worked_l195_195388


namespace range_of_m_l195_195249

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + m + 8 ≥ 0) ↔ (-8 / 9 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l195_195249


namespace bottle_total_height_l195_195713

theorem bottle_total_height (r1 r2 water_height_up water_height_down : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 3) (h_water_height_up : water_height_up = 20) (h_water_height_down : water_height_down = 28) : 
    ∃ x : ℝ, (π * r1^2 * (x - water_height_up) = 9 * π * (x - water_height_down) ∧ x = 29) := 
by 
    sorry

end bottle_total_height_l195_195713


namespace john_bought_3_croissants_l195_195122

variable (c k : ℕ)

theorem john_bought_3_croissants
  (h1 : c + k = 5)
  (h2 : ∃ n : ℕ, 88 * c + 44 * k = 100 * n) :
  c = 3 :=
by
-- Proof omitted
sorry

end john_bought_3_croissants_l195_195122


namespace midpoint_x_coordinate_l195_195515

theorem midpoint_x_coordinate (M N : ℝ × ℝ)
  (hM : M.1 ^ 2 = 4 * M.2)
  (hN : N.1 ^ 2 = 4 * N.2)
  (h_dist : (Real.sqrt ((M.1 - 1)^2 + M.2^2)) + (Real.sqrt ((N.1 - 1)^2 + N.2^2)) = 6) :
  (M.1 + N.1) / 2 = 2 := 
sorry

end midpoint_x_coordinate_l195_195515


namespace sequence_term_geometric_l195_195809

theorem sequence_term_geometric :
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 →
    (∀ n, n ≥ 2 → (a n) / (a (n - 1)) = 2^(n-1)) →
    a 101 = 2^5050 :=
by
  sorry

end sequence_term_geometric_l195_195809


namespace ratio_first_to_second_l195_195667

theorem ratio_first_to_second (S F T : ℕ) 
  (hS : S = 60)
  (hT : T = F / 3)
  (hSum : F + S + T = 220) :
  F / S = 2 :=
by
  sorry

end ratio_first_to_second_l195_195667


namespace combined_cost_price_l195_195622

def cost_price_A : ℕ := (120 + 60) / 2
def cost_price_B : ℕ := (200 + 100) / 2
def cost_price_C : ℕ := (300 + 180) / 2

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C

theorem combined_cost_price :
  total_cost_price = 480 := by
  sorry

end combined_cost_price_l195_195622


namespace sum_eq_2184_l195_195901

variable (p q r s : ℝ)

-- Conditions
axiom h1 : r + s = 12 * p
axiom h2 : r * s = 14 * q
axiom h3 : p + q = 12 * r
axiom h4 : p * q = 14 * s
axiom distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

-- Problem: Prove that p + q + r + s = 2184
theorem sum_eq_2184 : p + q + r + s = 2184 := 
by {
  sorry
}

end sum_eq_2184_l195_195901


namespace ratio_of_dimensions_128_l195_195325

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end ratio_of_dimensions_128_l195_195325


namespace solve_equation_l195_195308

theorem solve_equation {x y z : ℝ} (h₁ : x + 95 / 12 * y + 4 * z = 0)
  (h₂ : 4 * x + 95 / 12 * y - 3 * z = 0)
  (h₃ : 3 * x + 5 * y - 4 * z = 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 * z / y^3 = -60 :=
sorry

end solve_equation_l195_195308


namespace not_possible_identical_nonzero_remainders_l195_195406

theorem not_possible_identical_nonzero_remainders :
  ¬ ∃ (a : ℕ → ℕ) (r : ℕ), (r > 0) ∧ (∀ i : Fin 100, a i % (a ((i + 1) % 100)) = r) :=
by
  sorry

end not_possible_identical_nonzero_remainders_l195_195406


namespace ratio_x_w_l195_195389

variable {x y z w : ℕ}

theorem ratio_x_w (h1 : x / y = 24) (h2 : z / y = 8) (h3 : z / w = 1 / 12) : x / w = 1 / 4 := by
  sorry

end ratio_x_w_l195_195389


namespace problem_1_l195_195760

noncomputable def derivative_y (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) : ℝ :=
  (2 * a) / (3 * (1 - y^2))

theorem problem_1 (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) :
  derivative_y a x y h = (2 * a) / (3 * (1 - y^2)) :=
sorry

end problem_1_l195_195760


namespace max_value_of_exp_l195_195580

theorem max_value_of_exp (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : 
  a^2 * b^3 * c ≤ 27 / 16 := 
  sorry

end max_value_of_exp_l195_195580


namespace savings_l195_195499

def distance_each_way : ℕ := 150
def round_trip_distance : ℕ := 2 * distance_each_way
def rental_cost_first_option : ℕ := 50
def rental_cost_second_option : ℕ := 90
def gasoline_efficiency : ℕ := 15
def gasoline_cost_per_liter : ℚ := 0.90
def gasoline_needed_for_trip : ℚ := round_trip_distance / gasoline_efficiency
def total_gasoline_cost : ℚ := gasoline_needed_for_trip * gasoline_cost_per_liter
def total_cost_first_option : ℚ := rental_cost_first_option + total_gasoline_cost
def total_cost_second_option : ℚ := rental_cost_second_option

theorem savings : total_cost_second_option - total_cost_first_option = 22 := by
  sorry

end savings_l195_195499


namespace total_miles_traveled_l195_195825

noncomputable def distance_to_first_museum : ℕ := 5
noncomputable def distance_to_second_museum : ℕ := 15
noncomputable def distance_to_cultural_center : ℕ := 10
noncomputable def extra_detour : ℕ := 3

theorem total_miles_traveled : 
  (2 * (distance_to_first_museum + extra_detour) + 2 * distance_to_second_museum + 2 * distance_to_cultural_center) = 66 :=
  by
  sorry

end total_miles_traveled_l195_195825


namespace find_num_boys_l195_195252

-- Definitions for conditions
def num_children : ℕ := 13
def num_girls (num_boys : ℕ) : ℕ := num_children - num_boys

-- We will assume we have a predicate representing the truthfulness of statements.
-- boys tell the truth to boys and lie to girls
-- girls tell the truth to girls and lie to boys

theorem find_num_boys (boys_truth_to_boys : Prop) 
                      (boys_lie_to_girls : Prop) 
                      (girls_truth_to_girls : Prop) 
                      (girls_lie_to_boys : Prop)
                      (alternating_statements : Prop) : 
  ∃ (num_boys : ℕ), num_boys = 7 := 
  sorry

end find_num_boys_l195_195252


namespace find_largest_value_l195_195032

theorem find_largest_value
  (h1: 0 < Real.sin 2) (h2: Real.sin 2 < 1)
  (h3: Real.log 2 / Real.log (1 / 3) < 0)
  (h4: Real.log (1 / 3) / Real.log (1 / 2) > 1) :
  Real.log (1 / 3) / Real.log (1 / 2) > Real.sin 2 ∧ 
  Real.log (1 / 3) / Real.log (1 / 2) > Real.log 2 / Real.log (1 / 3) := by
  sorry

end find_largest_value_l195_195032


namespace Jenny_ate_65_l195_195873

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end Jenny_ate_65_l195_195873


namespace measure_of_y_l195_195987

theorem measure_of_y (y : ℕ) (h₁ : 40 + 2 * y + y = 180) : y = 140 / 3 :=
by
  sorry

end measure_of_y_l195_195987


namespace f_zero_is_118_l195_195594

theorem f_zero_is_118
  (f : ℕ → ℕ)
  (eq1 : ∀ m n : ℕ, f (m^2 + n^2) = (f m - f n)^2 + f (2 * m * n))
  (eq2 : 8 * f 0 + 9 * f 1 = 2006) :
  f 0 = 118 :=
sorry

end f_zero_is_118_l195_195594


namespace sum_of_odd_integers_15_to_51_l195_195582

def odd_arithmetic_series_sum (a1 an d : ℤ) (n : ℕ) : ℤ :=
  (n * (a1 + an)) / 2

theorem sum_of_odd_integers_15_to_51 :
  odd_arithmetic_series_sum 15 51 2 19 = 627 :=
by
  sorry

end sum_of_odd_integers_15_to_51_l195_195582


namespace circle_condition_l195_195913

noncomputable def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, circle_eq x y m) → m < 1 / 4 :=
by
  sorry

end circle_condition_l195_195913


namespace white_socks_cost_proof_l195_195463

-- Define the cost of a single brown sock in cents
def brown_sock_cost (B : ℕ) : Prop :=
  15 * B = 300

-- Define the cost of two white socks in cents
def white_socks_cost (B : ℕ) (W : ℕ) : Prop :=
  W = B + 25

-- Statement of the problem
theorem white_socks_cost_proof : 
  ∃ B W : ℕ, brown_sock_cost B ∧ white_socks_cost B W ∧ W = 45 :=
by
  sorry

end white_socks_cost_proof_l195_195463


namespace variable_value_l195_195620

theorem variable_value (w x v : ℝ) (h1 : 5 / w + 5 / x = 5 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) : v = 0.25 :=
by
  sorry

end variable_value_l195_195620


namespace octahedron_non_blue_probability_l195_195307

theorem octahedron_non_blue_probability :
  let total_faces := 8
  let blue_faces := 3
  let red_faces := 3
  let green_faces := 2
  let non_blue_faces := total_faces - blue_faces
  (non_blue_faces / total_faces : ℚ) = (5 / 8 : ℚ) :=
by
  sorry

end octahedron_non_blue_probability_l195_195307


namespace find_a_l195_195859

noncomputable def f (a x : ℝ) : ℝ := a * x * (x - 2)^2

theorem find_a (a : ℝ) (h1 : a ≠ 0)
  (h2 : ∃ x : ℝ, f a x = 32) :
  a = 27 :=
sorry

end find_a_l195_195859


namespace fixed_point_of_parabola_l195_195352

theorem fixed_point_of_parabola (s : ℝ) : ∃ y : ℝ, y = 4 * 3^2 + s * 3 - 3 * s ∧ (3, y) = (3, 36) :=
by
  sorry

end fixed_point_of_parabola_l195_195352


namespace total_sounds_produced_l195_195926

-- Defining the total number of nails for one customer and the number of customers
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3

-- Proving the total number of nail trimming sounds for 3 customers = 60
theorem total_sounds_produced : nails_per_person * number_of_customers = 60 := by
  sorry

end total_sounds_produced_l195_195926


namespace question1_question2_l195_195537

theorem question1 (m : ℝ) (x : ℝ) :
  (∀ x, x^2 - m * x + (m - 1) ≥ 0) → m = 2 :=
by
  sorry

theorem question2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (n = (a + 1 / b) * (2 * b + 1 / (2 * a))) → n ≥ (9 / 2) :=
by
  sorry

end question1_question2_l195_195537


namespace total_money_shared_l195_195934

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l195_195934


namespace probability_X_greater_than_2_l195_195709

noncomputable def probability_distribution (i : ℕ) : ℝ :=
  if h : 1 ≤ i ∧ i ≤ 4 then i / 10 else 0

theorem probability_X_greater_than_2 :
  (probability_distribution 3 + probability_distribution 4) = 0.7 := by 
  sorry

end probability_X_greater_than_2_l195_195709


namespace tan_neg_3780_eq_zero_l195_195060

theorem tan_neg_3780_eq_zero : Real.tan (-3780 * Real.pi / 180) = 0 := 
by 
  sorry

end tan_neg_3780_eq_zero_l195_195060


namespace days_between_dates_l195_195694

-- Define the starting and ending dates
def start_date : Nat := 1990 * 365 + (19 + 2 * 31 + 28) -- March 19, 1990 (accounting for leap years before the start date)
def end_date : Nat   := 1996 * 365 + (23 + 2 * 31 + 29 + 366 * 2 + 365 * 3) -- March 23, 1996 (accounting for leap years)

-- Define the number of leap years between the dates
def leap_years : Nat := 2 -- 1992 and 1996

-- Total number of days
def total_days : Nat := (end_date - start_date + 1)

theorem days_between_dates : total_days = 2197 :=
by
  sorry

end days_between_dates_l195_195694


namespace evaluate_expression_l195_195625

theorem evaluate_expression :
  2 - (-3) - 4 + (-5) - 6 + 7 = -3 :=
by
  sorry

end evaluate_expression_l195_195625


namespace area_of_enclosing_square_is_100_l195_195761

noncomputable def radius : ℝ := 5

noncomputable def diameter_of_circle (r : ℝ) : ℝ := 2 * r

noncomputable def side_length_of_square (d : ℝ) : ℝ := d

noncomputable def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_enclosing_square_is_100 :
  area_of_square (side_length_of_square (diameter_of_circle radius)) = 100 :=
by
  sorry

end area_of_enclosing_square_is_100_l195_195761


namespace distance_between_vertices_l195_195735

-- Define the equations of the parabolas
def C_eq (x : ℝ) : ℝ := x^2 + 6 * x + 13
def D_eq (x : ℝ) : ℝ := -x^2 + 2 * x + 8

-- Define the vertices of the parabolas
def vertex_C : (ℝ × ℝ) := (-3, 4)
def vertex_D : (ℝ × ℝ) := (1, 9)

-- Prove that the distance between the vertices is sqrt 41
theorem distance_between_vertices : 
  dist (vertex_C) (vertex_D) = Real.sqrt 41 := 
by
  sorry

end distance_between_vertices_l195_195735


namespace min_value_sin_function_l195_195858

theorem min_value_sin_function (α β : ℝ) (h : -5 * (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 3 * Real.sin α) :
  ∃ x : ℝ, x = Real.sin α ∧ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 0 :=
sorry

end min_value_sin_function_l195_195858


namespace movie_revenue_multiple_correct_l195_195679

-- Definitions from the conditions
def opening_weekend_revenue : ℝ := 120 * 10^6
def company_share_fraction : ℝ := 0.60
def profit : ℝ := 192 * 10^6
def production_cost : ℝ := 60 * 10^6

-- The statement to prove
theorem movie_revenue_multiple_correct : 
  ∃ M : ℝ, (company_share_fraction * (opening_weekend_revenue * M) - production_cost = profit) ∧ M = 3.5 :=
by
  sorry

end movie_revenue_multiple_correct_l195_195679


namespace solution_set_ineq_l195_195235

theorem solution_set_ineq (x : ℝ) :
  (x - 1) / (1 - 2 * x) ≥ 0 ↔ (1 / 2 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_ineq_l195_195235


namespace homework_problems_l195_195257

noncomputable def problems_solved (p t : ℕ) : ℕ := p * t

theorem homework_problems (p t : ℕ) (h_eq: p * t = (3 * p - 5) * (t - 3))
  (h_pos_p: p > 0) (h_pos_t: t > 0) (h_p_ge_15: p ≥ 15) 
  (h_friend_did_20: (3 * p - 5) * (t - 3) ≥ 20) : 
  problems_solved p t = 100 :=
by
  sorry

end homework_problems_l195_195257


namespace moles_of_water_formed_l195_195805

-- Definitions
def moles_of_H2SO4 : Nat := 3
def moles_of_NaOH : Nat := 3
def moles_of_NaHSO4 : Nat := 3
def moles_of_H2O := moles_of_NaHSO4

-- Theorem
theorem moles_of_water_formed :
  moles_of_H2SO4 = 3 →
  moles_of_NaOH = 3 →
  moles_of_NaHSO4 = 3 →
  moles_of_H2O = 3 :=
by
  intros h1 h2 h3
  rw [moles_of_H2O]
  exact h3

end moles_of_water_formed_l195_195805


namespace iron_conducts_electricity_l195_195454

-- Define the predicates
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
noncomputable def Iron : Type := sorry
  
theorem iron_conducts_electricity (h1 : ∀ x, Metal x → ConductsElectricity x)
  (h2 : Metal Iron) : ConductsElectricity Iron :=
by
  sorry

end iron_conducts_electricity_l195_195454


namespace seven_times_equivalent_l195_195520

theorem seven_times_equivalent (n a b : ℤ) (h : n = a^2 + a * b + b^2) :
  ∃ (c d : ℤ), 7 * n = c^2 + c * d + d^2 :=
sorry

end seven_times_equivalent_l195_195520


namespace square_integer_2209_implies_value_l195_195176

theorem square_integer_2209_implies_value (x : ℤ) (h : x^2 = 2209) : (2*x + 1)*(2*x - 1) = 8835 :=
by sorry

end square_integer_2209_implies_value_l195_195176


namespace log_base_2_of_1024_l195_195532

theorem log_base_2_of_1024 (h : 2^10 = 1024) : Real.logb 2 1024 = 10 :=
by
  sorry

end log_base_2_of_1024_l195_195532


namespace find_sum_of_money_invested_l195_195162

theorem find_sum_of_money_invested (P : ℝ) (h1 : SI_15 = P * (15 / 100) * 2)
                                    (h2 : SI_12 = P * (12 / 100) * 2)
                                    (h3 : SI_15 - SI_12 = 720) : 
                                    P = 12000 :=
by
  -- Skipping the proof
  sorry

end find_sum_of_money_invested_l195_195162


namespace tangent_line_slope_angle_l195_195793

theorem tangent_line_slope_angle (θ : ℝ) : 
  (∃ k : ℝ, (∀ x y, k * x - y = 0) ∧ ∀ x y, x^2 + y^2 - 4 * x + 3 = 0) →
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end tangent_line_slope_angle_l195_195793


namespace three_digit_number_property_l195_195689

theorem three_digit_number_property :
  (∃ a b c : ℕ, 100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c = (a + b + c)^3) ↔
  (∃ a b c : ℕ, a = 5 ∧ b = 1 ∧ c = 2 ∧ 100 * a + 10 * b + c = 512) := sorry

end three_digit_number_property_l195_195689


namespace value_of_f2009_l195_195598

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f2009 
  (h_ineq1 : ∀ x : ℝ, f x ≤ f (x+4) + 4)
  (h_ineq2 : ∀ x : ℝ, f (x+2) ≥ f x + 2)
  (h_f1 : f 1 = 0) :
  f 2009 = 2008 :=
sorry

end value_of_f2009_l195_195598


namespace jane_current_age_l195_195097

noncomputable def JaneAge : ℕ := 34

theorem jane_current_age : 
  ∃ J : ℕ, 
    (∀ t : ℕ, t ≥ 18 ∧ t - 18 ≤ JaneAge - 18 → t ≤ JaneAge / 2) ∧
    (JaneAge - 12 = 23 - 12 * 2) ∧
    (23 = 23) →
    J = 34 := by
  sorry

end jane_current_age_l195_195097


namespace tom_took_out_beads_l195_195422

-- Definitions of the conditions
def green_beads : Nat := 1
def brown_beads : Nat := 2
def red_beads : Nat := 3
def beads_left_in_container : Nat := 4

-- Total initial beads
def total_beads : Nat := green_beads + brown_beads + red_beads

-- The Lean problem statement to prove
theorem tom_took_out_beads : (total_beads - beads_left_in_container) = 2 :=
by
  sorry

end tom_took_out_beads_l195_195422


namespace apples_per_basket_l195_195899

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) (h : total_apples = 629) (k : num_baskets = 37) :
  total_apples / num_baskets = 17 :=
by
  -- proof omitted
  sorry

end apples_per_basket_l195_195899


namespace factorization_correct_l195_195159

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end factorization_correct_l195_195159


namespace decreasing_function_range_l195_195962

theorem decreasing_function_range (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end decreasing_function_range_l195_195962


namespace negation_proposition_l195_195923

theorem negation_proposition : ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ ∃ x : ℝ, x > 0 ∧ x < 1 := 
by
  sorry

end negation_proposition_l195_195923


namespace probability_C_calc_l195_195973

noncomputable section

-- Define the given probabilities
def prob_A : ℚ := 3 / 8
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 3 / 16
def prob_D : ℚ := prob_C

-- The sum of probabilities equals 1
theorem probability_C_calc :
  prob_A + prob_B + prob_C + prob_D = 1 :=
by
  -- Simplifying directly, we can assert the correctness of given prob_C
  sorry

end probability_C_calc_l195_195973


namespace jenna_less_than_bob_l195_195287

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end jenna_less_than_bob_l195_195287


namespace calculate_fraction_l195_195919

theorem calculate_fraction : (5 / (8 / 13) / (10 / 7) = 91 / 16) :=
by
  sorry

end calculate_fraction_l195_195919


namespace p_nonnegative_iff_equal_l195_195727

def p (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem p_nonnegative_iff_equal (a b c : ℝ) : (∀ x : ℝ, p a b c x ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end p_nonnegative_iff_equal_l195_195727


namespace larger_number_is_1590_l195_195173

theorem larger_number_is_1590 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 :=
by
  sorry

end larger_number_is_1590_l195_195173


namespace tickets_spent_on_beanie_l195_195461

-- Define the initial conditions
def initial_tickets : ℕ := 25
def additional_tickets : ℕ := 15
def tickets_left : ℕ := 18

-- Define the total tickets
def total_tickets := initial_tickets + additional_tickets

-- Define what we're proving: the number of tickets spent on the beanie
theorem tickets_spent_on_beanie : initial_tickets + additional_tickets - tickets_left = 22 :=
by 
  -- Provide proof steps here
  sorry

end tickets_spent_on_beanie_l195_195461


namespace original_price_of_dish_l195_195244

theorem original_price_of_dish (P : ℝ) (h1 : ∃ P, John's_payment = (0.9 * P) + (0.15 * P))
                               (h2 : ∃ P, Jane's_payment = (0.9 * P) + (0.135 * P))
                               (h3 : John's_payment = Jane's_payment + 0.51) : P = 34 := by
  -- John's Payment
  let John's_payment := (0.9 * P) + (0.15 * P)
  -- Jane's Payment
  let Jane's_payment := (0.9 * P) + (0.135 * P)
  -- Condition that John paid $0.51 more than Jane
  have h3 : John's_payment = Jane's_payment + 0.51 := sorry
  -- From the given conditions, we need to prove P = 34
  sorry

end original_price_of_dish_l195_195244


namespace carnival_ticket_count_l195_195061

theorem carnival_ticket_count (ferris_wheel_rides bumper_car_rides ride_cost : ℕ) 
  (h1 : ferris_wheel_rides = 7) 
  (h2 : bumper_car_rides = 3) 
  (h3 : ride_cost = 5) : 
  ferris_wheel_rides + bumper_car_rides * ride_cost = 50 := 
by {
  -- proof omitted
  sorry
}

end carnival_ticket_count_l195_195061


namespace num_people_comparison_l195_195358

def num_people_1st_session (a : ℝ) : Prop := a > 0 -- Define the number for first session
def num_people_2nd_session (a : ℝ) : ℝ := 1.1 * a -- Define the number for second session
def num_people_3rd_session (a : ℝ) : ℝ := 0.99 * a -- Define the number for third session

theorem num_people_comparison (a b : ℝ) 
    (h1 : b = 0.99 * a): 
    a > b := 
by 
  -- insert the proof here
  sorry 

end num_people_comparison_l195_195358


namespace find_value_of_N_l195_195500

theorem find_value_of_N 
  (N : ℝ) 
  (h : (20 / 100) * N = (30 / 100) * 2500) 
  : N = 3750 := 
sorry

end find_value_of_N_l195_195500


namespace christine_sales_value_l195_195156

variable {X : ℝ}

def commission_rate : ℝ := 0.12
def personal_needs_percent : ℝ := 0.60
def savings_amount : ℝ := 1152
def savings_percent : ℝ := 0.40

theorem christine_sales_value:
  (savings_percent * (commission_rate * X) = savings_amount) → 
  (X = 24000) := 
by
  intro h
  sorry

end christine_sales_value_l195_195156


namespace max_value_of_f_l195_195206

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem: the maximum value of f(x) is 4
theorem max_value_of_f : ∃ x : ℝ, f x = 4 := sorry

end max_value_of_f_l195_195206


namespace banks_investments_count_l195_195770

-- Conditions
def revenue_per_investment_banks := 500
def revenue_per_investment_elizabeth := 900
def number_of_investments_elizabeth := 5
def extra_revenue_elizabeth := 500

-- Total revenue calculations
def total_revenue_elizabeth := number_of_investments_elizabeth * revenue_per_investment_elizabeth
def total_revenue_banks := total_revenue_elizabeth - extra_revenue_elizabeth

-- Number of investments for Mr. Banks
def number_of_investments_banks := total_revenue_banks / revenue_per_investment_banks

theorem banks_investments_count : number_of_investments_banks = 8 := by
  sorry

end banks_investments_count_l195_195770


namespace cubic_roots_sum_cubes_l195_195517

theorem cubic_roots_sum_cubes
  (p q r : ℂ)
  (h_eq_root : ∀ x, x = p ∨ x = q ∨ x = r → x^3 - 2 * x^2 + 3 * x - 1 = 0)
  (h_sum : p + q + r = 2)
  (h_prod_sum : p * q + q * r + r * p = 3)
  (h_prod : p * q * r = 1) :
  p^3 + q^3 + r^3 = -7 := by
  sorry

end cubic_roots_sum_cubes_l195_195517


namespace find_f2_l195_195334

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 2) : f a b 2 = 0 := by
  sorry

end find_f2_l195_195334


namespace num_toys_purchased_min_selling_price_l195_195716

variable (x m : ℕ)

-- Given conditions
axiom cond1 : 1500 / x + 5 = 3500 / (2 * x)
axiom cond2 : 150 * m - 5000 >= 1150

-- Required proof
theorem num_toys_purchased : x = 50 :=
by
  sorry

theorem min_selling_price : m >= 41 :=
by
  sorry

end num_toys_purchased_min_selling_price_l195_195716


namespace point_on_inverse_graph_and_sum_l195_195668

-- Definitions
variable (f : ℝ → ℝ)
variable (h : f 2 = 6)

-- Theorem statement
theorem point_on_inverse_graph_and_sum (hf : ∀ x, x = 2 → 3 = (f x) / 2) :
  (6, 1 / 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, (f⁻¹ x) / 2)} ∧
  (6 + (1 / 2) = 13 / 2) :=
by
  sorry

end point_on_inverse_graph_and_sum_l195_195668


namespace solution_y_values_l195_195303
-- Import the necessary libraries

-- Define the system of equations and the necessary conditions
def equation1 (x : ℝ) := x^2 - 6*x + 8 = 0
def equation2 (x y : ℝ) := 2*x - y = 6

-- The main theorem to be proven
theorem solution_y_values : ∃ x1 x2 y1 y2 : ℝ, 
  (equation1 x1 ∧ equation1 x2 ∧ equation2 x1 y1 ∧ equation2 x2 y2 ∧ 
  y1 = 2 ∧ y2 = -2) :=
by
  -- Use the provided solutions in the problem statement
  use 4, 2, 2, -2
  sorry  -- The details of the proof are omitted.

end solution_y_values_l195_195303


namespace minimize_y_l195_195983

noncomputable def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y (a b c : ℝ) : ∃ x : ℝ, (∀ x0 : ℝ, y x a b c ≤ y x0 a b c) ∧ x = (a + b + c) / 3 :=
by
  sorry

end minimize_y_l195_195983


namespace find_ratio_of_hyperbola_l195_195179

noncomputable def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem find_ratio_of_hyperbola (a b : ℝ) (h : a > b) 
  (h_asymptote_angle : ∀ α : ℝ, (y = ↑(b / a) * x -> α = 45)) :
  a / b = 1 :=
sorry

end find_ratio_of_hyperbola_l195_195179


namespace west_of_1km_l195_195168

def east_direction (d : Int) : Int :=
  d

def west_direction (d : Int) : Int :=
  -d

theorem west_of_1km :
  east_direction (2) = 2 →
  west_direction (1) = -1 := by
  sorry

end west_of_1km_l195_195168


namespace company_KW_price_l195_195910

theorem company_KW_price (A B : ℝ) (x : ℝ) (h1 : P = x * A) (h2 : P = 2 * B) (h3 : P = (6 / 7) * (A + B)) : x = 1.666666666666667 := 
sorry

end company_KW_price_l195_195910


namespace all_push_ups_total_l195_195808

-- Definitions derived from the problem's conditions
def ZacharyPushUps := 47
def DavidPushUps := ZacharyPushUps + 15
def EmilyPushUps := DavidPushUps * 2
def TotalPushUps := ZacharyPushUps + DavidPushUps + EmilyPushUps

-- The statement to be proved
theorem all_push_ups_total : TotalPushUps = 233 := by
  sorry

end all_push_ups_total_l195_195808


namespace complement_of_A_l195_195429

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | abs (x - 1) > 1 }

-- Define the problem statement
theorem complement_of_A :
  ∀ x : ℝ, x ∈ compl A ↔ x ∈ Icc 0 2 :=
by
  intro x
  rw [mem_compl_iff, mem_Icc]
  sorry

end complement_of_A_l195_195429


namespace monotonicity_of_f_range_of_a_l195_195045

noncomputable def f (a x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → (∀ x, f a x ≥ f a (2 * a) → x ≤ 0 ∨ x ≤ 2 * a)) ∧
  (a = 0 → ∀ x y, x ≤ y → f a x ≥ f a y) ∧
  (a > 0 → (∀ x, f a x ≤ f a 0 → x ≤ 0) ∧
           (∀ x, 0 < x ∧ x < 2 * a → f a x ≥ f a 2 * a) ∧
           (∀ x, 2 * a < x → f a x ≤ f a (2 * a))) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, a ≥ 1 / 2 →
  ∃ x1 : ℝ, x1 > 0 ∧ ∃ x2 : ℝ, f a x1 ≥ g a x2 :=
sorry

end monotonicity_of_f_range_of_a_l195_195045


namespace simplify_and_evaluate_l195_195348

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l195_195348


namespace bars_sold_this_week_l195_195035

-- Definitions based on conditions
def total_bars : Nat := 18
def bars_sold_last_week : Nat := 5
def bars_needed_to_sell : Nat := 6

-- Statement of the proof problem
theorem bars_sold_this_week : (total_bars - (bars_needed_to_sell + bars_sold_last_week)) = 2 := by
  -- proof goes here
  sorry

end bars_sold_this_week_l195_195035


namespace volume_pyramid_PABCD_is_384_l195_195025

noncomputable def volume_of_pyramid : ℝ :=
  let AB := 12
  let BC := 6
  let PA := Real.sqrt (20^2 - 12^2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem volume_pyramid_PABCD_is_384 :
  volume_of_pyramid = 384 := 
by
  sorry

end volume_pyramid_PABCD_is_384_l195_195025


namespace find_correct_fraction_l195_195950

theorem find_correct_fraction
  (mistake_frac : ℚ) (n : ℕ) (delta : ℚ)
  (correct_frac : ℚ) (number : ℕ)
  (h1 : mistake_frac = 5 / 6)
  (h2 : number = 288)
  (h3 : mistake_frac * number = correct_frac * number + delta)
  (h4 : delta = 150) :
  correct_frac = 5 / 32 :=
by
  sorry

end find_correct_fraction_l195_195950


namespace range_of_m_l195_195290

theorem range_of_m (m : ℝ) : (∀ (x : ℝ), |3 - x| + |5 + x| > m) → m < 8 :=
sorry

end range_of_m_l195_195290


namespace final_output_M_l195_195199

-- Definitions of the steps in the conditions
def initial_M : ℕ := 1
def increment_M1 (M : ℕ) : ℕ := M + 1
def increment_M2 (M : ℕ) : ℕ := M + 2

-- Define the final value of M after performing the operations
def final_M : ℕ := increment_M2 (increment_M1 initial_M)

-- The statement to prove
theorem final_output_M : final_M = 4 :=
by
  -- Placeholder for the actual proof
  sorry

end final_output_M_l195_195199


namespace harold_catches_up_at_12_miles_l195_195947

/-- 
Proof Problem: Given that Adrienne starts walking from X to Y at 3 miles per hour and one hour later Harold starts walking from X to Y at 4 miles per hour, prove that Harold covers 12 miles when he catches up to Adrienne.
-/
theorem harold_catches_up_at_12_miles :
  (∀ (T : ℕ), (ad_distance : ℕ) = 3 * (T + 1) → (ha_distance : ℕ) = 4 * T → ad_distance = ha_distance) →
  (∃ T : ℕ, ha_distance = 12) :=
by
  sorry

end harold_catches_up_at_12_miles_l195_195947


namespace find_a8_l195_195603

variable (a : ℕ+ → ℕ)

theorem find_a8 (h : ∀ m n : ℕ+, a (m * n) = a m * a n) (h2 : a 2 = 3) : a 8 = 27 := 
by
  sorry

end find_a8_l195_195603


namespace intersection_is_singleton_l195_195611

namespace ProofProblem

def M : Set ℤ := {-3, -2, -1}

def N : Set ℤ := {x : ℤ | (x + 2) * (x - 3) < 0}

theorem intersection_is_singleton : M ∩ N = {-1} :=
by
  sorry

end ProofProblem

end intersection_is_singleton_l195_195611


namespace abc_def_ratio_l195_195151

theorem abc_def_ratio (a b c d e f : ℝ)
    (h1 : a / b = 1 / 3)
    (h2 : b / c = 2)
    (h3 : c / d = 1 / 2)
    (h4 : d / e = 3)
    (h5 : e / f = 1 / 8) :
    (a * b * c) / (d * e * f) = 1 / 8 :=
by
  sorry

end abc_def_ratio_l195_195151


namespace Jhon_payment_per_day_l195_195683

theorem Jhon_payment_per_day
  (total_days : ℕ)
  (present_days : ℕ)
  (absent_pay : ℝ)
  (total_pay : ℝ)
  (Jhon_present_days : total_days = 60)
  (Jhon_presence : present_days = 35)
  (Jhon_absent_payment : absent_pay = 3.0)
  (Jhon_total_payment : total_pay = 170) :
  ∃ (P : ℝ), 
    P = 2.71 ∧ 
    total_pay = (present_days * P + (total_days - present_days) * absent_pay) := 
sorry

end Jhon_payment_per_day_l195_195683


namespace x_minus_y_div_x_eq_4_7_l195_195817

-- Definitions based on the problem's conditions
axiom y_div_x_eq_3_7 (x y : ℝ) : y / x = 3 / 7

-- The main problem to prove
theorem x_minus_y_div_x_eq_4_7 (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end x_minus_y_div_x_eq_4_7_l195_195817


namespace membership_percentage_change_l195_195721

theorem membership_percentage_change :
  let initial_membership := 100.0
  let first_fall_membership := initial_membership * 1.04
  let first_spring_membership := first_fall_membership * 0.95
  let second_fall_membership := first_spring_membership * 1.07
  let second_spring_membership := second_fall_membership * 0.97
  let third_fall_membership := second_spring_membership * 1.05
  let third_spring_membership := third_fall_membership * 0.81
  let final_membership := third_spring_membership
  let total_percentage_change := ((final_membership - initial_membership) / initial_membership) * 100.0
  total_percentage_change = -12.79 :=
by
  sorry

end membership_percentage_change_l195_195721


namespace no_real_roots_l195_195482

-- Define the polynomial P(X) = X^5
def P (X : ℝ) : ℝ := X^5

-- Prove that for every α ∈ ℝ*, the polynomial P(X + α) - P(X) has no real roots
theorem no_real_roots (α : ℝ) (hα : α ≠ 0) : ∀ (X : ℝ), P (X + α) ≠ P X :=
by sorry

end no_real_roots_l195_195482


namespace geometric_sequence_ratio_l195_195019

variable {a : ℕ → ℝ} -- Define the geometric sequence {a_n}

-- Conditions: The sequence is geometric with positive terms
variable (q : ℝ) (hq : q > 0) (hgeo : ∀ n, a (n + 1) = q * a n)

-- Additional condition: a2, 1/2 a3, and a1 form an arithmetic sequence
variable (hseq : a 1 - (1 / 2) * a 2 = (1 / 2) * a 2 - a 0)

theorem geometric_sequence_ratio :
  (a 3 + a 4) / (a 2 + a 3) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_sequence_ratio_l195_195019


namespace quadratic_equation_with_one_variable_is_B_l195_195484

def is_quadratic_equation_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + x + 3 = 0"

theorem quadratic_equation_with_one_variable_is_B :
  is_quadratic_equation_with_one_variable "x^2 + x + 3 = 0" :=
by
  sorry

end quadratic_equation_with_one_variable_is_B_l195_195484


namespace infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l195_195184

-- Problem 1: Infinitely many primes congruent to 3 modulo 4
theorem infinite_primes_congruent_3_mod_4 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 4 = 3) → ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ ps :=
by
  sorry

-- Problem 2: Infinitely many primes congruent to 5 modulo 6
theorem infinite_primes_congruent_5_mod_6 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 6 = 5) → ∃ q, Nat.Prime q ∧ q % 6 = 5 ∧ q ∉ ps :=
by
  sorry

end infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l195_195184


namespace total_cost_eq_57_l195_195617

namespace CandyCost

-- Conditions
def cost_of_caramel : ℕ := 3
def cost_of_candy_bar : ℕ := 2 * cost_of_caramel
def cost_of_cotton_candy : ℕ := (4 * cost_of_candy_bar) / 2

-- Define the total cost calculation
def total_cost : ℕ :=
  (6 * cost_of_candy_bar) + (3 * cost_of_caramel) + cost_of_cotton_candy

-- Theorem we want to prove
theorem total_cost_eq_57 : total_cost = 57 :=
by
  sorry  -- Proof to be provided

end CandyCost

end total_cost_eq_57_l195_195617


namespace driving_time_is_correct_l195_195745

-- Define conditions
def flight_departure : ℕ := 20 * 60 -- 8:00 pm in minutes since 0:00
def checkin_time : ℕ := flight_departure - 2 * 60 -- 2 hours early
def latest_leave_time : ℕ := 17 * 60 -- 5:00 pm in minutes since 0:00
def additional_time : ℕ := 15 -- 15 minutes to park and make their way to the terminal

-- Define question
def driving_time : ℕ := checkin_time - additional_time - latest_leave_time

-- Prove the expected answer
theorem driving_time_is_correct : driving_time = 45 :=
by
  -- omitting the proof
  sorry

end driving_time_is_correct_l195_195745


namespace total_bad_carrots_and_tomatoes_l195_195876

theorem total_bad_carrots_and_tomatoes 
  (vanessa_carrots : ℕ := 17)
  (vanessa_tomatoes : ℕ := 12)
  (mother_carrots : ℕ := 14)
  (mother_tomatoes : ℕ := 22)
  (brother_carrots : ℕ := 6)
  (brother_tomatoes : ℕ := 8)
  (good_carrots : ℕ := 28)
  (good_tomatoes : ℕ := 35) :
  (vanessa_carrots + mother_carrots + brother_carrots - good_carrots) + 
  (vanessa_tomatoes + mother_tomatoes + brother_tomatoes - good_tomatoes) = 16 := 
by
  sorry

end total_bad_carrots_and_tomatoes_l195_195876


namespace john_final_push_time_l195_195969

theorem john_final_push_time :
  ∃ t : ℝ, (∀ (d_j d_s : ℝ), d_j = 4.2 * t ∧ d_s = 3.7 * t ∧ (d_j = d_s + 14)) → t = 28 :=
by
  sorry

end john_final_push_time_l195_195969


namespace range_of_c_l195_195602

theorem range_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 1) : ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end range_of_c_l195_195602


namespace sequence_factorial_l195_195712

theorem sequence_factorial (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n > 0 → a n = n * a (n - 1)) :
  ∀ n : ℕ, a n = Nat.factorial n :=
by
  sorry

end sequence_factorial_l195_195712


namespace total_blocks_correct_l195_195998

-- Definitions given by the conditions in the problem
def red_blocks : ℕ := 18
def yellow_blocks : ℕ := red_blocks + 7
def blue_blocks : ℕ := red_blocks + 14

-- Theorem stating the goal to prove
theorem total_blocks_correct : red_blocks + yellow_blocks + blue_blocks = 75 := by
  -- Skipping the proof for now
  sorry

end total_blocks_correct_l195_195998


namespace value_of_y_l195_195758

theorem value_of_y (y m : ℕ) (h1 : ((1 ^ m) / (y ^ m)) * (1 ^ 16 / 4 ^ 16) = 1 / (2 * 10 ^ 31)) (h2 : m = 31) : 
  y = 5 := 
sorry

end value_of_y_l195_195758


namespace nth_equation_l195_195402

theorem nth_equation (n : ℕ) (hn : n ≠ 0) : 
  (↑n + 2) / ↑n - 2 / (↑n + 2) = ((↑n + 2)^2 + ↑n^2) / (↑n * (↑n + 2)) - 1 :=
by
  sorry

end nth_equation_l195_195402


namespace battery_charge_to_60_percent_l195_195677

noncomputable def battery_charge_time (initial_charge_percent : ℝ) (initial_time_minutes : ℕ) (additional_time_minutes : ℕ) : ℕ :=
  let rate_per_minute := initial_charge_percent / initial_time_minutes
  let additional_charge_percent := additional_time_minutes * rate_per_minute
  let total_percent := initial_charge_percent + additional_charge_percent
  if total_percent = 60 then
    initial_time_minutes + additional_time_minutes
  else
    sorry

theorem battery_charge_to_60_percent : battery_charge_time 20 60 120 = 180 :=
by
  -- The formal proof will be provided here.
  sorry

end battery_charge_to_60_percent_l195_195677


namespace clerical_percentage_after_reduction_l195_195031

-- Define the initial conditions
def total_employees : ℕ := 3600
def clerical_fraction : ℚ := 1/4
def reduction_fraction : ℚ := 1/4

-- Define the intermediate calculations
def initial_clerical_employees : ℚ := clerical_fraction * total_employees
def clerical_reduction : ℚ := reduction_fraction * initial_clerical_employees
def new_clerical_employees : ℚ := initial_clerical_employees - clerical_reduction
def total_employees_after_reduction : ℚ := total_employees - clerical_reduction

-- State the theorem
theorem clerical_percentage_after_reduction :
  (new_clerical_employees / total_employees_after_reduction) * 100 = 20 :=
sorry

end clerical_percentage_after_reduction_l195_195031


namespace result_is_21_l195_195892

theorem result_is_21 (n : ℕ) (h : n = 55) : (n / 5 + 10) = 21 :=
by
  sorry

end result_is_21_l195_195892


namespace factor_polynomial_l195_195789

theorem factor_polynomial (x : ℝ) :
  (x^3 - 12 * x + 16) = (x + 4) * ((x - 2)^2) :=
by
  sorry

end factor_polynomial_l195_195789


namespace find_real_solutions_l195_195700

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end find_real_solutions_l195_195700


namespace arithmetic_sequence_solution_l195_195374

theorem arithmetic_sequence_solution :
  ∃ (a1 d : ℤ), 
    (a1 + 3*d + (a1 + 4*d) + (a1 + 5*d) + (a1 + 6*d) = 56) ∧
    ((a1 + 3*d) * (a1 + 6*d) = 187) ∧
    (
      (a1 = 5 ∧ d = 2) ∨
      (a1 = 23 ∧ d = -2)
    ) :=
by
  sorry

end arithmetic_sequence_solution_l195_195374


namespace solution_exists_l195_195146

theorem solution_exists (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 :=
by
  sorry

end solution_exists_l195_195146


namespace cookies_count_l195_195783

theorem cookies_count :
  ∀ (Tom Lucy Millie Mike Frank : ℕ), 
  (Tom = 16) →
  (Lucy = Nat.sqrt Tom) →
  (Millie = 2 * Lucy) →
  (Mike = 3 * Millie) →
  (Frank = Mike / 2 - 3) →
  Frank = 9 :=
by
  intros Tom Lucy Millie Mike Frank hTom hLucy hMillie hMike hFrank
  have h1 : Tom = 16 := hTom
  have h2 : Lucy = Nat.sqrt Tom := hLucy
  have h3 : Millie = 2 * Lucy := hMillie
  have h4 : Mike = 3 * Millie := hMike
  have h5 : Frank = Mike / 2 - 3 := hFrank
  sorry

end cookies_count_l195_195783


namespace roadster_paving_company_cement_usage_l195_195837

theorem roadster_paving_company_cement_usage :
  let L := 10
  let T := 5.1
  L + T = 15.1 :=
by
  -- proof is omitted
  sorry

end roadster_paving_company_cement_usage_l195_195837


namespace range_of_a_l195_195798

open Set

-- Define proposition p
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0

-- Define proposition q
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define negation of p
def not_p (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define negation of q
def not_q (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Main theorem to prove the range of a
theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 1 → -3 ≤ x ∧ x ≤ 1) → a ∈ Icc (-3 : ℝ) (0 : ℝ) :=
by
  intro h
  -- skipped detailed proof
  sorry

end range_of_a_l195_195798


namespace fermats_little_theorem_for_q_plus_1_l195_195185

theorem fermats_little_theorem_for_q_plus_1 (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  (q + 1)^(q - 1) % q = 1 := by
  sorry

end fermats_little_theorem_for_q_plus_1_l195_195185


namespace find_a100_l195_195588

theorem find_a100 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, a n = (2 * (S n)^2) / (2 * (S n) - 1))
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 100 = -2 / 39203 := 
sorry

-- Explanation of the statement:
-- 'theorem find_a100': We define a theorem to find a_100.
-- 'a : ℕ → ℝ': a is a sequence of real numbers.
-- 'S : ℕ → ℝ': S is a sequence representing the sum of the first n terms.
-- 'h1' to 'h3': Given conditions from the problem statement.
-- 'a 100 = -2 / 39203' : The statement to prove.

end find_a100_l195_195588


namespace sum_a_b_eq_negative_one_l195_195294

theorem sum_a_b_eq_negative_one 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0)
  (h2 : ∀ x : ℝ, x^2 - a * x - b = 0 → x = 2 ∨ x = 3) :
  a + b = -1 := 
sorry

end sum_a_b_eq_negative_one_l195_195294


namespace sector_angle_measure_l195_195690

-- Define the variables
variables (r α : ℝ)

-- Define the conditions
def perimeter_condition := (2 * r + r * α = 4)
def area_condition := (1 / 2 * α * r^2 = 1)

-- State the theorem
theorem sector_angle_measure (h1 : perimeter_condition r α) (h2 : area_condition r α) : α = 2 :=
sorry

end sector_angle_measure_l195_195690


namespace max_min_difference_l195_195322

open Real

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 2 * y = 4) :
  ∃(max min : ℝ), (∀z, z = (|2 * x - y| / (|x| + |y|)) → z ≤ max) ∧ 
                  (∀z, z = (|2 * x - y| / (|x| + |y|)) → min ≤ z) ∧ 
                  (max - min = 5) :=
by
  sorry

end max_min_difference_l195_195322


namespace tile_floor_multiple_of_seven_l195_195971

theorem tile_floor_multiple_of_seven (n : ℕ) (a : ℕ)
  (h1 : n * n = 7 * a)
  (h2 : 4 * a / 7 + 3 * a / 7 = a) :
  ∃ k : ℕ, n = 7 * k := by
  sorry

end tile_floor_multiple_of_seven_l195_195971


namespace triangle_angle_range_l195_195615

theorem triangle_angle_range (α β γ : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α = 2 * γ)
  (h3 : α ≥ β)
  (h4 : β ≥ γ) :
  45 ≤ β ∧ β ≤ 72 := 
sorry

end triangle_angle_range_l195_195615


namespace books_per_shelf_l195_195640

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ) 
    (h1 : mystery_shelves = 5) (h2 : picture_shelves = 4) (h3 : total_books = 54) : 
    total_books / (mystery_shelves + picture_shelves) = 6 := 
by
  -- necessary preliminary steps and full proof will go here
  sorry

end books_per_shelf_l195_195640


namespace inequality_solution_I_inequality_solution_II_l195_195692

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| - |x + 1|

theorem inequality_solution_I (x : ℝ) : f x 1 > 2 ↔ x < -2 / 3 ∨ x > 4 :=
sorry 

noncomputable def g (x a : ℝ) : ℝ := f x a + |x + 1| + x

theorem inequality_solution_II (a : ℝ) : (∀ x, g x a > a ^ 2 - 1 / 2) ↔ (-1 / 2 < a ∧ a < 1) :=
sorry

end inequality_solution_I_inequality_solution_II_l195_195692


namespace reciprocal_of_mixed_number_l195_195714

def mixed_number := -1 - (4 / 5)

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number : reciprocal mixed_number = -5 / 9 := 
by
  sorry

end reciprocal_of_mixed_number_l195_195714


namespace rectangle_circle_diameter_l195_195566

theorem rectangle_circle_diameter:
  ∀ (m n : ℕ), (∃ (x : ℚ), m + n = 47 ∧ (∀ (r : ℚ), r = (20 / 7)) →
  (2 * r = (40 / 7))) :=
by
  sorry

end rectangle_circle_diameter_l195_195566


namespace ribbon_per_box_l195_195095

def total_ribbon : ℝ := 4.5
def remaining_ribbon : ℝ := 1
def number_of_boxes : ℕ := 5

theorem ribbon_per_box :
  (total_ribbon - remaining_ribbon) / number_of_boxes = 0.7 :=
by
  sorry

end ribbon_per_box_l195_195095


namespace brick_height_l195_195710

variable {l w : ℕ} (SA : ℕ)

theorem brick_height (h : ℕ) (l_eq : l = 10) (w_eq : w = 4) (SA_eq : SA = 136) 
    (surface_area_eq : SA = 2 * (l * w + l * h + w * h)) : h = 2 :=
by
  sorry

end brick_height_l195_195710


namespace k_value_l195_195827

open Real

noncomputable def k_from_roots (α β : ℝ) : ℝ := - (α + β)

theorem k_value (k : ℝ) (α β : ℝ) (h1 : α + β = -k) (h2 : α * β = 8) (h3 : (α+3) + (β+3) = k) (h4 : (α+3) * (β+3) = 12) : k = 3 :=
by
  -- Here we skip the proof as instructed.
  sorry

end k_value_l195_195827


namespace tan_C_l195_195002

theorem tan_C (A B C : ℝ) (hABC : A + B + C = π) (tan_A : Real.tan A = 1 / 2) 
  (cos_B : Real.cos B = 3 * Real.sqrt 10 / 10) : Real.tan C = -1 :=
by
  sorry

end tan_C_l195_195002


namespace vehicle_count_expression_l195_195488

variable (C B M : ℕ)

-- Given conditions
axiom wheel_count : 4 * C + 2 * B + 2 * M = 196
axiom bike_to_motorcycle : B = 2 * M

-- Prove that the number of cars can be expressed in terms of the number of motorcycles
theorem vehicle_count_expression : C = (98 - 3 * M) / 2 :=
by
  sorry

end vehicle_count_expression_l195_195488


namespace dasha_paper_strip_l195_195442

theorem dasha_paper_strip (a b c : ℕ) (h1 : a < b) (h2 : 2 * a * b + 2 * a * c - a^2 = 43) :
    ∃ (length width : ℕ), length = a ∧ width = b + c := by
  sorry

end dasha_paper_strip_l195_195442


namespace ratio_expression_l195_195274

theorem ratio_expression (a b c : ℝ) (ha : a / b = 20) (hb : b / c = 10) : (a + b) / (b + c) = 210 / 11 := by
  sorry

end ratio_expression_l195_195274


namespace sum_lent_is_1100_l195_195887

variables (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)

-- Given conditions
def interest_formula := I = P * r * t
def interest_difference := I = P - 572

-- Values
def rate := r = 0.06
def time := t = 8

theorem sum_lent_is_1100 : P = 1100 :=
by
  -- Definitions and axioms
  sorry

end sum_lent_is_1100_l195_195887


namespace value_of_a_l195_195015

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l195_195015


namespace ratio_of_ages_l195_195180

-- Definitions of the conditions
def son_current_age : ℕ := 28
def man_current_age : ℕ := son_current_age + 30
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem
theorem ratio_of_ages : (man_age_in_two_years / son_age_in_two_years) = 2 :=
by
  -- Skipping the proof steps
  sorry

end ratio_of_ages_l195_195180


namespace smallest_triangle_perimeter_l195_195150

theorem smallest_triangle_perimeter : ∃ (a b c : ℕ), a = 3 ∧ b = a + 1 ∧ c = b + 1 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 12 := by
  sorry

end smallest_triangle_perimeter_l195_195150


namespace polynomial_divisible_by_7_l195_195317

theorem polynomial_divisible_by_7 (n : ℤ) : 7 ∣ ((n + 7)^2 - n^2) :=
sorry

end polynomial_divisible_by_7_l195_195317


namespace remaining_tickets_l195_195262

def initial_tickets : ℝ := 49.0
def lost_tickets : ℝ := 6.0
def spent_tickets : ℝ := 25.0

theorem remaining_tickets : initial_tickets - lost_tickets - spent_tickets = 18.0 := by
  sorry

end remaining_tickets_l195_195262


namespace no_solution_for_equation_l195_195245

/-- The given equation expressed using letters as unique digits:
    ∑ (letters as digits) from БАРАНКА + БАРАБАН + КАРАБАС = ПАРАЗИТ
    We aim to prove that there are no valid digit assignments satisfying the equation. -/
theorem no_solution_for_equation :
  ∀ (b a r n k s p i t: ℕ),
  b ≠ a ∧ b ≠ r ∧ b ≠ n ∧ b ≠ k ∧ b ≠ s ∧ b ≠ p ∧ b ≠ i ∧ b ≠ t ∧
  a ≠ r ∧ a ≠ n ∧ a ≠ k ∧ a ≠ s ∧ a ≠ p ∧ a ≠ i ∧ a ≠ t ∧
  r ≠ n ∧ r ≠ k ∧ r ≠ s ∧ r ≠ p ∧ r ≠ i ∧ r ≠ t ∧
  n ≠ k ∧ n ≠ s ∧ n ≠ p ∧ n ≠ i ∧ n ≠ t ∧
  k ≠ s ∧ k ≠ p ∧ k ≠ i ∧ k ≠ t ∧
  s ≠ p ∧ s ≠ i ∧ s ≠ t ∧
  p ≠ i ∧ p ≠ t ∧
  i ≠ t →
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * n + k +
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * b + a + n +
  100000 * k + 10000 * a + 1000 * r + 100 * a + 10 * b + a + s ≠ 
  100000 * p + 10000 * a + 1000 * r + 100 * a + 10 * z + i + t :=
sorry

end no_solution_for_equation_l195_195245


namespace sum_of_roots_l195_195974

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) →
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) →
  a + b + c + d = 8 :=
by
  sorry

end sum_of_roots_l195_195974


namespace height_percentage_difference_l195_195175

theorem height_percentage_difference 
  (r1 h1 r2 h2 : ℝ) 
  (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2)
  (r2_eq_1_2_r1 : r2 = (6 / 5) * r1) :
  h1 = (36 / 25) * h2 :=
by
  sorry

end height_percentage_difference_l195_195175


namespace combinatorial_proof_l195_195869

noncomputable def combinatorial_identity (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) : ℕ :=
  let summation_term (i : ℕ) := Nat.choose k i * Nat.choose n (m - i)
  List.sum (List.map summation_term (List.range (k + 1)))

theorem combinatorial_proof (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) :
  combinatorial_identity n m k h1 h2 h3 = Nat.choose (n + k) m :=
sorry

end combinatorial_proof_l195_195869


namespace simplify_and_evaluate_expression_l195_195140

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = 2):
  ( ( (2 * m + 1) / m - 1 ) / ( (m^2 - 1) / m ) ) = 1 :=
by
  rw [h] -- Replace m by 2
  sorry

end simplify_and_evaluate_expression_l195_195140


namespace union_of_A_and_B_l195_195539

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end union_of_A_and_B_l195_195539


namespace magnitude_of_a_plus_b_in_range_l195_195530

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def theta_domain : Set ℝ := {θ : ℝ | -Real.pi / 2 < θ ∧ θ < Real.pi / 2}

open Real

theorem magnitude_of_a_plus_b_in_range (θ : ℝ) (hθ : θ ∈ theta_domain) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (cos θ, sin θ)
  1 < sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) ∧ sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) < (3 + 2 * sqrt 2) :=
sorry

end magnitude_of_a_plus_b_in_range_l195_195530


namespace children_on_bus_l195_195997

/-- Prove the number of children on the bus after the bus stop equals 14 given the initial conditions -/
theorem children_on_bus (initial_children : ℕ) (children_got_off : ℕ) (extra_children_got_on : ℕ) (final_children : ℤ) :
  initial_children = 5 →
  children_got_off = 63 →
  extra_children_got_on = 9 →
  final_children = (initial_children - children_got_off) + (children_got_off + extra_children_got_on) →
  final_children = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end children_on_bus_l195_195997


namespace min_value_of_f_inequality_a_b_l195_195392

theorem min_value_of_f :
  ∃ m : ℝ, m = 4 ∧ (∀ x : ℝ, |x + 3| + |x - 1| ≥ m) :=
sorry

theorem inequality_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1 / a + 4 / b ≥ 9 / 4) :=
sorry

end min_value_of_f_inequality_a_b_l195_195392


namespace trajectory_equation_necessary_not_sufficient_l195_195023

theorem trajectory_equation_necessary_not_sufficient :
  ∀ (x y : ℝ), (|x| = |y|) → (y = |x|) ↔ (necessary_not_sufficient) :=
by
  sorry

end trajectory_equation_necessary_not_sufficient_l195_195023


namespace lightsaber_ratio_l195_195036

theorem lightsaber_ratio (T L : ℕ) (hT : T = 1000) (hTotal : L + T = 3000) : L / T = 2 :=
by
  sorry

end lightsaber_ratio_l195_195036


namespace triangle_angle_inequality_l195_195613

theorem triangle_angle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  4 / A + 1 / (B + C) ≥ 9 / Real.pi := by
  sorry

end triangle_angle_inequality_l195_195613


namespace find_valid_pairs_l195_195510

-- Defining the conditions and target answer set.
def valid_pairs : List (Nat × Nat) := [(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)]

theorem find_valid_pairs (a b : Nat) :
  (∃ n m : Int, (a^2 + b = n * (b^2 - a)) ∧ (b^2 + a = m * (a^2 - b)))
  ↔ (a, b) ∈ valid_pairs :=
by sorry

end find_valid_pairs_l195_195510


namespace theta_half_quadrant_l195_195781

open Real

theorem theta_half_quadrant (θ : ℝ) (k : ℤ) 
  (h1 : 2 * k * π + 3 * π / 2 ≤ θ ∧ θ ≤ 2 * k * π + 2 * π) 
  (h2 : |cos (θ / 2)| = -cos (θ / 2)) : 
  k * π + 3 * π / 4 ≤ θ / 2 ∧ θ / 2 ≤ k * π + π ∧ cos (θ / 2) < 0 := 
sorry

end theta_half_quadrant_l195_195781


namespace probability_sum_even_for_three_cubes_l195_195251

-- Define the probability function
def probability_even_sum (n: ℕ) : ℚ :=
  if n > 0 then 1 / 2 else 0

theorem probability_sum_even_for_three_cubes : probability_even_sum 3 = 1 / 2 :=
by
  sorry

end probability_sum_even_for_three_cubes_l195_195251


namespace min_tip_percentage_l195_195460

namespace TipCalculation

def mealCost : Float := 35.50
def totalPaid : Float := 37.275
def maxTipPercent : Float := 0.08

theorem min_tip_percentage : ∃ (P : Float), (P / 100 * mealCost = (totalPaid - mealCost)) ∧ (P < maxTipPercent * 100) ∧ (P = 5) := by
  sorry

end TipCalculation

end min_tip_percentage_l195_195460


namespace line_through_point_parallel_l195_195872

theorem line_through_point_parallel (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) (hA : A = (2, 3)) (hl : ∀ x y, l x y ↔ 2 * x - 4 * y + 7 = 0) :
  ∃ m, (∀ x y, (2 * x - 4 * y + m = 0) ↔ (x - 2 * y + 4 = 0)) ∧ (2 * (A.1) - 4 * (A.2) + m = 0) := 
sorry

end line_through_point_parallel_l195_195872


namespace part1_proof_part2_proof_l195_195534

open Real

-- Definitions for the conditions
variables (x y z : ℝ)
variable (h₁ : 0 < x)
variable (h₂ : 0 < y)
variable (h₃ : 0 < z)

-- Part 1
theorem part1_proof : (1 / x + 1 / y ≥ 4 / (x + y)) :=
by sorry

-- Part 2
theorem part2_proof : (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) :=
by sorry

end part1_proof_part2_proof_l195_195534


namespace negation_of_p_l195_195311

theorem negation_of_p :
  (¬ (∀ x > 0, (x+1)*Real.exp x > 1)) ↔ 
  (∃ x ≤ 0, (x+1)*Real.exp x ≤ 1) :=
sorry

end negation_of_p_l195_195311


namespace arcsin_sqrt_one_half_l195_195011

theorem arcsin_sqrt_one_half : Real.arcsin (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  -- TODO: provide proof
  sorry

end arcsin_sqrt_one_half_l195_195011


namespace votes_diff_eq_70_l195_195833

noncomputable def T : ℝ := 350
def votes_against (T : ℝ) : ℝ := 0.40 * T
def votes_favor (T : ℝ) (X : ℝ) : ℝ := votes_against T + X

theorem votes_diff_eq_70 :
  ∃ X : ℝ, 350 = votes_against T + votes_favor T X → X = 70 :=
by
  sorry

end votes_diff_eq_70_l195_195833


namespace number_of_days_to_catch_fish_l195_195041

variable (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ)

theorem number_of_days_to_catch_fish (h1 : fish_per_day = 2) 
                                    (h2 : fillets_per_fish = 2) 
                                    (h3 : total_fillets = 120) : 
                                    (total_fillets / fillets_per_fish) / fish_per_day = 30 :=
by sorry

end number_of_days_to_catch_fish_l195_195041


namespace BB_digit_value_in_5BB3_l195_195864

theorem BB_digit_value_in_5BB3 (B : ℕ) (h : 2 * B + 8 % 9 = 0) : B = 5 :=
sorry

end BB_digit_value_in_5BB3_l195_195864


namespace find_first_half_speed_l195_195166

theorem find_first_half_speed (distance time total_time : ℝ) (v2 : ℝ)
    (h_distance : distance = 300) 
    (h_time : total_time = 11) 
    (h_v2 : v2 = 25) 
    (half_distance : distance / 2 = 150) :
    (150 / (total_time - (150 / v2)) = 30) :=
by
  sorry

end find_first_half_speed_l195_195166


namespace subject_selection_ways_l195_195927

theorem subject_selection_ways :
  let compulsory := 3 -- Chinese, Mathematics, English
  let choose_one := 2
  let choose_two := 6
  compulsory + choose_one * choose_two = 12 :=
by
  sorry

end subject_selection_ways_l195_195927


namespace alien_run_time_l195_195917

variable (v_r v_f : ℝ) -- velocities in km/h
variable (T_r T_f : ℝ) -- time in hours
variable (D_r D_f : ℝ) -- distances in kilometers

theorem alien_run_time :
  v_r = 15 ∧ v_f = 10 ∧ (T_f = T_r + 0.5) ∧ (D_r = D_f) ∧ (D_r = v_r * T_r) ∧ (D_f = v_f * T_f) → T_f = 1.5 :=
by
  intros h
  rcases h with ⟨_, _, _, _, _, _⟩
  -- proof goes here
  sorry

end alien_run_time_l195_195917


namespace infinitely_many_colorings_l195_195483

def colorings_exist (clr : ℕ → Prop) : Prop :=
  ∀ a b : ℕ, (clr a = clr b) ∧ (0 < a - 10 * b) → clr (a - 10 * b) = clr a

theorem infinitely_many_colorings : ∃ (clr : ℕ → Prop), colorings_exist clr :=
sorry

end infinitely_many_colorings_l195_195483


namespace average_visitors_other_days_l195_195903

theorem average_visitors_other_days 
  (avg_sunday : ℕ) (avg_day : ℕ)
  (num_days : ℕ) (sunday_offset : ℕ)
  (other_days_count : ℕ) (total_days : ℕ) 
  (total_avg_visitors : ℕ)
  (sunday_avg_visitors : ℕ) :
  avg_sunday = 150 →
  avg_day = 125 →
  num_days = 30 →
  sunday_offset = 5 →
  total_days = 30 →
  total_avg_visitors * total_days =
    (sunday_offset * sunday_avg_visitors) + (other_days_count * avg_sunday) →
  125 = total_avg_visitors →
  150 = sunday_avg_visitors →
  other_days_count = num_days - sunday_offset →
  (125 * 30 = (5 * 150) + (other_days_count * avg_sunday)) →
  avg_sunday = 120 :=
by
  sorry

end average_visitors_other_days_l195_195903


namespace cone_lateral_surface_area_l195_195860

theorem cone_lateral_surface_area (l d : ℝ) (h_l : l = 5) (h_d : d = 8) : 
  (π * (d / 2) * l) = 20 * π :=
by
  sorry

end cone_lateral_surface_area_l195_195860


namespace exists_multiple_of_10_of_three_distinct_integers_l195_195213

theorem exists_multiple_of_10_of_three_distinct_integers
    (a b c : ℤ) 
    (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    ∃ x y : ℤ, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ (10 ∣ (x^5 * y^3 - x^3 * y^5)) :=
by
  sorry

end exists_multiple_of_10_of_three_distinct_integers_l195_195213


namespace sum_of_a_b_either_1_or_neg1_l195_195261

theorem sum_of_a_b_either_1_or_neg1 (a b : ℝ) (h1 : a + a = 0) (h2 : b * b = 1) : a + b = 1 ∨ a + b = -1 :=
by {
  sorry
}

end sum_of_a_b_either_1_or_neg1_l195_195261


namespace symmetric_circle_l195_195831

-- Define given circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 8 * y + 12 = 0

-- Define the line of symmetry
def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 5 = 0

-- Define the symmetric circle equation we need to prove
def symm_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- Lean 4 theorem statement
theorem symmetric_circle (x y : ℝ) :
  (∃ a b : ℝ, circle_equation 2 4 ∧ line_equation a b ∧ (a, b) = (0, 0)) →
  symm_circle_equation x y :=
by sorry

end symmetric_circle_l195_195831


namespace packs_of_tuna_purchased_l195_195467

-- Definitions based on the conditions
def cost_per_pack_of_tuna : ℕ := 2
def cost_per_bottle_of_water : ℤ := (3 / 2)
def total_paid_by_Barbara : ℕ := 56
def money_spent_on_different_goods : ℕ := 40
def number_of_bottles_of_water : ℕ := 4

-- The proposition to prove
theorem packs_of_tuna_purchased :
  ∃ T : ℕ, total_paid_by_Barbara = cost_per_pack_of_tuna * T + cost_per_bottle_of_water * number_of_bottles_of_water + money_spent_on_different_goods ∧ T = 5 :=
by
  sorry

end packs_of_tuna_purchased_l195_195467


namespace veromont_clicked_ads_l195_195881

def ads_on_first_page := 12
def ads_on_second_page := 2 * ads_on_first_page
def ads_on_third_page := ads_on_second_page + 24
def ads_on_fourth_page := (3 / 4) * ads_on_second_page
def total_ads := ads_on_first_page + ads_on_second_page + ads_on_third_page + ads_on_fourth_page
def ads_clicked := (2 / 3) * total_ads

theorem veromont_clicked_ads : ads_clicked = 68 := 
by
  sorry

end veromont_clicked_ads_l195_195881


namespace find_b_l195_195174

theorem find_b (a b c d : ℝ) (h : ∃ k : ℝ, 2 * k = π ∧ k * (b / 2) = π) : b = 4 :=
by
  sorry

end find_b_l195_195174


namespace sequence_term_position_l195_195850

theorem sequence_term_position :
  ∃ n : ℕ, ∀ k : ℕ, (k = 7 + 6 * (n - 1)) → k = 2005 → n = 334 :=
by
  sorry

end sequence_term_position_l195_195850


namespace water_force_on_dam_l195_195855

-- Given conditions
def density : Real := 1000  -- kg/m^3
def gravity : Real := 10    -- m/s^2
def a : Real := 5.7         -- m
def b : Real := 9.0         -- m
def h : Real := 4.0         -- m

-- Prove that the force is 544000 N under the given conditions
theorem water_force_on_dam : ∃ (F : Real), F = 544000 :=
by
  sorry  -- proof goes here

end water_force_on_dam_l195_195855


namespace odometer_problem_l195_195814

theorem odometer_problem :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a + b + c ≤ 10 ∧ (11 * c - 10 * a - b) % 6 = 0 ∧ a^2 + b^2 + c^2 = 54 :=
by
  sorry

end odometer_problem_l195_195814


namespace complement_set_M_l195_195753

-- Definitions of sets based on given conditions
def universal_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def set_M : Set ℝ := {x | x^2 - x ≤ 0}

-- The proof statement that we need to prove
theorem complement_set_M :
  {x | 1 < x ∧ x ≤ 2} = universal_set \ set_M := by
  sorry

end complement_set_M_l195_195753


namespace quadratic_root_equation_l195_195479

-- Define the conditions given in the problem
variables (a b x : ℝ)

-- Assertion for a ≠ 0
axiom a_ne_zero : a ≠ 0

-- Root assumption
axiom root_assumption : (x^2 + b * x + a = 0) → x = -a

-- Lean statement to prove that b - a = 1
theorem quadratic_root_equation (h : x^2 + b * x + a = 0) : b - a = 1 :=
sorry

end quadratic_root_equation_l195_195479


namespace paperclips_exceed_200_at_friday_l195_195012

def paperclips_on_day (n : ℕ) : ℕ :=
  3 * 4^n

theorem paperclips_exceed_200_at_friday : 
  ∃ n : ℕ, n = 4 ∧ paperclips_on_day n > 200 :=
by
  sorry

end paperclips_exceed_200_at_friday_l195_195012


namespace stimulus_check_total_l195_195799

def find_stimulus_check (T : ℝ) : Prop :=
  let amount_after_wife := T * (3/5)
  let amount_after_first_son := amount_after_wife * (3/5)
  let amount_after_second_son := amount_after_first_son * (3/5)
  amount_after_second_son = 432

theorem stimulus_check_total (T : ℝ) : find_stimulus_check T → T = 2000 := by
  sorry

end stimulus_check_total_l195_195799


namespace number_of_sides_l195_195395

theorem number_of_sides (P l n : ℕ) (hP : P = 49) (hl : l = 7) (h : P = n * l) : n = 7 :=
by
  sorry

end number_of_sides_l195_195395


namespace capacity_of_buckets_l195_195408

theorem capacity_of_buckets :
  (∃ x : ℝ, 26 * x = 39 * 9) → (∃ x : ℝ, 26 * x = 351 ∧ x = 13.5) :=
by
  sorry

end capacity_of_buckets_l195_195408


namespace solve_lambda_l195_195804

variable (a b : ℝ × ℝ)
variable (lambda : ℝ)

def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

axiom a_def : a = (-3, 2)
axiom b_def : b = (-1, 0)
axiom perp_def : perpendicular (a.1 + lambda * b.1, a.2 + lambda * b.2) b

theorem solve_lambda : lambda = -3 :=
by
  sorry

end solve_lambda_l195_195804


namespace unique_three_positive_perfect_square_sums_to_100_l195_195474

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l195_195474


namespace arithmetic_sequence_find_side_length_l195_195105

variable (A B C a b c : ℝ)

-- Condition: Given that b(1 + cos(C)) = c(2 - cos(B))
variable (h : b * (1 + Real.cos C) = c * (2 - Real.cos B))

-- Question I: Prove that a + b = 2 * c
theorem arithmetic_sequence (h : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : a + b = 2 * c :=
sorry

-- Additional conditions for Question II
variable (C_eq : C = Real.pi / 3)
variable (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3)

-- Question II: Find c
theorem find_side_length (C_eq : C = Real.pi / 3) (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) : c = 4 :=
sorry

end arithmetic_sequence_find_side_length_l195_195105


namespace integer_values_of_b_l195_195946

theorem integer_values_of_b (b : ℤ) :
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ 
  b = -21 ∨ b = 19 ∨ b = -17 ∨ b = -4 ∨ b = 3 :=
by
  sorry

end integer_values_of_b_l195_195946


namespace initial_people_on_train_l195_195889

theorem initial_people_on_train {x y z u v w : ℤ} 
  (h1 : y = 29) (h2 : z = 17) (h3 : u = 27) (h4 : v = 35) (h5 : w = 116) :
  x - (y - z) + (v - u) = w → x = 120 := 
by sorry

end initial_people_on_train_l195_195889


namespace gcd_of_items_l195_195072

def numPens : ℕ := 891
def numPencils : ℕ := 810
def numNotebooks : ℕ := 1080
def numErasers : ℕ := 972

theorem gcd_of_items :
  Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numNotebooks) numErasers = 27 :=
by
  sorry

end gcd_of_items_l195_195072


namespace average_speed_return_trip_l195_195137

def speed1 : ℝ := 12 -- Speed for the first part of the trip in miles per hour
def distance1 : ℝ := 18 -- Distance for the first part of the trip in miles
def speed2 : ℝ := 10 -- Speed for the second part of the trip in miles per hour
def distance2 : ℝ := 18 -- Distance for the second part of the trip in miles
def total_round_trip_time : ℝ := 7.3 -- Total time for the round trip in hours

theorem average_speed_return_trip :
  let time1 := distance1 / speed1 -- Time taken for the first part of the trip
  let time2 := distance2 / speed2 -- Time taken for the second part of the trip
  let total_time_to_destination := time1 + time2 -- Total time for the trip to the destination
  let time_return_trip := total_round_trip_time - total_time_to_destination -- Time for the return trip
  let return_trip_distance := distance1 + distance2 -- Distance for the return trip (same as to the destination)
  let avg_speed_return_trip := return_trip_distance / time_return_trip -- Average speed for the return trip
  avg_speed_return_trip = 9 := 
by
  sorry

end average_speed_return_trip_l195_195137


namespace right_triangle_angles_l195_195092

theorem right_triangle_angles (c : ℝ) (t : ℝ) (h : t = c^2 / 8) :
  ∃(A B: ℝ), A = 90 ∧ (B = 75 ∨ B = 15) :=
by
  sorry

end right_triangle_angles_l195_195092


namespace two_R_theta_bounds_l195_195009

variables {R : ℝ} (θ : ℝ)
variables (h_pos : 0 < R) (h_triangle : (R + 1 + (R + 1/2)) > 2 *R)

-- Define that θ is the angle between sides R and R + 1/2
-- Here we assume θ is defined via the cosine rule for simplicity

noncomputable def angle_between_sides (R : ℝ) := 
  Real.arccos ((R^2 + (R + 1/2)^2 - 1^2) / (2 * R * (R + 1/2)))

-- State the theorem
theorem two_R_theta_bounds (h : θ = angle_between_sides R) : 
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
by
  sorry

end two_R_theta_bounds_l195_195009


namespace cubic_sum_of_roots_l195_195030

theorem cubic_sum_of_roots (a b c : ℝ) 
  (h1 : a + b + c = -1)
  (h2 : a * b + b * c + c * a = -333)
  (h3 : a * b * c = 1001) :
  a^3 + b^3 + c^3 = 2003 :=
sorry

end cubic_sum_of_roots_l195_195030


namespace area_of_triangle_l195_195908

variables (yellow_area green_area blue_area : ℝ)
variables (is_equilateral_triangle : Prop)
variables (centered_at_vertices : Prop)
variables (radius_less_than_height : Prop)

theorem area_of_triangle (h_yellow : yellow_area = 1000)
                        (h_green : green_area = 100)
                        (h_blue : blue_area = 1)
                        (h_triangle : is_equilateral_triangle)
                        (h_centered : centered_at_vertices)
                        (h_radius : radius_less_than_height) :
  ∃ (area : ℝ), area = 150 :=
by
  sorry

end area_of_triangle_l195_195908


namespace original_laborers_count_l195_195040

theorem original_laborers_count (L : ℕ) (h1 : (L - 7) * 10 = L * 6) : L = 18 :=
sorry

end original_laborers_count_l195_195040


namespace distance_apart_after_skating_l195_195590

theorem distance_apart_after_skating :
  let Ann_speed := 6 -- Ann's speed in miles per hour
  let Glenda_speed := 8 -- Glenda's speed in miles per hour
  let skating_time := 3 -- Time spent skating in hours
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  Total_Distance = 42 :=
by
  let Ann_speed := 6
  let Glenda_speed := 8
  let skating_time := 3
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  sorry

end distance_apart_after_skating_l195_195590


namespace niu_fraction_property_l195_195762

open Nat

-- Given mn <= 2009, where m, n are positive integers and (n/m) is in lowest terms
-- Prove that for adjacent terms in the sequence, m_k n_{k+1} - m_{k+1} n_k = 1.

noncomputable def is_numerator_denom_pair_in_seq (m n : ℕ) : Bool :=
  m > 0 ∧ n > 0 ∧ m * n ≤ 2009

noncomputable def are_sorted_adjacent_in_seq (m_k n_k m_k1 n_k1 : ℕ) : Bool :=
  m_k * n_k1 - m_k1 * n_k = 1

theorem niu_fraction_property :
  ∀ (m_k n_k m_k1 n_k1 : ℕ),
  is_numerator_denom_pair_in_seq m_k n_k →
  is_numerator_denom_pair_in_seq m_k1 n_k1 →
  m_k < m_k1 →
  are_sorted_adjacent_in_seq m_k n_k m_k1 n_k1
:=
sorry

end niu_fraction_property_l195_195762


namespace dormitory_to_city_distance_l195_195994

theorem dormitory_to_city_distance
  (D : ℝ)
  (h1 : (1/5) * D + (2/3) * D + 14 = D) :
  D = 105 :=
by
  sorry

end dormitory_to_city_distance_l195_195994


namespace geo_seq_sum_l195_195493

theorem geo_seq_sum (a : ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = 2^n + a) →
  a = -1 :=
sorry

end geo_seq_sum_l195_195493


namespace bob_total_profit_l195_195275

-- Define the given inputs
def n_dogs : ℕ := 2
def c_dog : ℝ := 250.00
def n_puppies : ℕ := 6
def c_food_vac : ℝ := 500.00
def c_ad : ℝ := 150.00
def p_puppy : ℝ := 350.00

-- The statement to prove
theorem bob_total_profit : 
  (n_puppies * p_puppy - (n_dogs * c_dog + c_food_vac + c_ad)) = 950.00 :=
by
  sorry

end bob_total_profit_l195_195275


namespace product_of_marbles_l195_195049

theorem product_of_marbles (R B : ℕ) (h1 : R - B = 12) (h2 : R + B = 52) : R * B = 640 := by
  sorry

end product_of_marbles_l195_195049


namespace powerjet_30_minutes_500_gallons_per_hour_l195_195135

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end powerjet_30_minutes_500_gallons_per_hour_l195_195135


namespace find_investment_sum_l195_195125

theorem find_investment_sum (P : ℝ)
  (h1 : SI_15 = P * (15 / 100) * 2)
  (h2 : SI_12 = P * (12 / 100) * 2)
  (h3 : SI_15 - SI_12 = 420) :
  P = 7000 :=
by
  sorry

end find_investment_sum_l195_195125


namespace total_miles_driven_l195_195333

-- Define the required variables and their types
variables (avg1 avg2 : ℝ) (gallons1 gallons2 : ℝ) (miles1 miles2 : ℝ)

-- State the conditions
axiom sum_avg_mpg : avg1 + avg2 = 75
axiom first_car_gallons : gallons1 = 25
axiom second_car_gallons : gallons2 = 35
axiom first_car_avg_mpg : avg1 = 40

-- Declare the function to calculate miles driven
def miles_driven (avg_mpg gallons : ℝ) : ℝ := avg_mpg * gallons

-- Declare the theorem for proof
theorem total_miles_driven : miles_driven avg1 gallons1 + miles_driven avg2 gallons2 = 2225 := by
  sorry

end total_miles_driven_l195_195333


namespace quotient_when_divided_by_44_l195_195010

theorem quotient_when_divided_by_44 (N Q P : ℕ) (h1 : N = 44 * Q) (h2 : N = 35 * P + 3) : Q = 12 :=
by {
  -- Proof
  sorry
}

end quotient_when_divided_by_44_l195_195010


namespace donuts_selection_l195_195071

theorem donuts_selection :
  (∃ g c p : ℕ, g + c + p = 6 ∧ g ≥ 1 ∧ c ≥ 1 ∧ p ≥ 1) →
  ∃ k : ℕ, k = 10 :=
by {
  -- The mathematical proof steps are omitted according to the instructions
  sorry
}

end donuts_selection_l195_195071


namespace point_in_fourth_quadrant_l195_195865

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l195_195865


namespace part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l195_195324

def f (a x : ℝ) : ℝ := a * x ^ 2 + (1 - a) * x + a - 2

theorem part1 (a : ℝ) : (∀ x : ℝ, f a x ≥ -2) ↔ a ≥ 1/3 :=
sorry

theorem part2_case1 (a : ℝ) (ha : a = 0) : ∀ x : ℝ, f a x < a - 1 ↔ x < 1 :=
sorry

theorem part2_case2 (a : ℝ) (ha : a > 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (-1 / a < x ∧ x < 1) :=
sorry

theorem part2_case3_1 (a : ℝ) (ha : a = -1) : ∀ x : ℝ, (f a x < a - 1) ↔ x ≠ 1 :=
sorry

theorem part2_case3_2 (a : ℝ) (ha : -1 < a ∧ a < 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > -1 / a ∨ x < 1) :=
sorry

theorem part2_case3_3 (a : ℝ) (ha : a < -1) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > 1 ∨ x < -1 / a) :=
sorry

end part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l195_195324


namespace solve_abs_equation_l195_195505

theorem solve_abs_equation (y : ℝ) (h : |y - 8| + 3 * y = 12) : y = 2 :=
sorry

end solve_abs_equation_l195_195505


namespace max_amount_xiao_li_spent_l195_195351

theorem max_amount_xiao_li_spent (a m n : ℕ) :
  33 ≤ m ∧ m < n ∧ n ≤ 37 ∧
  ∃ (x y : ℕ), 
  (25 * (a - x) + m * (a - y) + n * (x + y + a) = 700) ∧ 
  (25 * x + m * y + n * (3*a - x - y) = 1200) ∧
  ( 675 <= 700 - 25) :=
sorry

end max_amount_xiao_li_spent_l195_195351


namespace function_satisfies_conditions_l195_195362

def f (m n : ℕ) : ℕ := m * n

theorem function_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := 
by {
  sorry
}

end function_satisfies_conditions_l195_195362


namespace find_a_l195_195365

variable (a : ℝ)

def A := ({1, 2, a} : Set ℝ)
def B := ({1, a^2 - a} : Set ℝ)

theorem find_a (h : B a ⊆ A a) : a = -1 ∨ a = 0 :=
  sorry

end find_a_l195_195365


namespace ramu_profit_percent_l195_195816

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

theorem ramu_profit_percent :
  profit_percent 42000 13000 64500 = 17.27 :=
by
  -- Placeholder for the proof
  sorry

end ramu_profit_percent_l195_195816


namespace min_value_product_expression_l195_195138

theorem min_value_product_expression (x : ℝ) : ∃ m, m = -2746.25 ∧ (∀ y : ℝ, (13 - y) * (8 - y) * (13 + y) * (8 + y) ≥ m) :=
sorry

end min_value_product_expression_l195_195138


namespace intersection_A_B_l195_195341

def A : Set ℝ := {x | abs x < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l195_195341


namespace total_apples_l195_195093

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l195_195093


namespace sum_of_areas_of_disks_l195_195471

theorem sum_of_areas_of_disks (r : ℝ) (a b c : ℕ) (h : a + b + c = 123) :
  ∃ (r : ℝ), (15 * Real.pi * r^2 = Real.pi * ((105 / 4) - 15 * Real.sqrt 3) ∧ r = 1 - (Real.sqrt 3) / 2) := 
by
  sorry

end sum_of_areas_of_disks_l195_195471


namespace average_remaining_two_numbers_l195_195080

theorem average_remaining_two_numbers 
    (a b c d e f : ℝ)
    (h1 : (a + b + c + d + e + f) / 6 = 3.95)
    (h2 : (a + b) / 2 = 4.4)
    (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 3.6 := 
sorry

end average_remaining_two_numbers_l195_195080


namespace find_missing_number_l195_195624

theorem find_missing_number (x : ℕ) :
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 :=
by
  sorry

end find_missing_number_l195_195624


namespace probability_win_all_games_l195_195437

variable (p : ℚ) (n : ℕ)

-- Define the conditions
def probability_of_winning := p = 2 / 3
def number_of_games := n = 6
def independent_games := true

-- The theorem we want to prove
theorem probability_win_all_games (h₁ : probability_of_winning p)
                                   (h₂ : number_of_games n)
                                   (h₃ : independent_games) :
  p^n = 64 / 729 :=
sorry

end probability_win_all_games_l195_195437


namespace salary_reduction_l195_195551

variable (S R : ℝ) (P : ℝ)
variable (h1 : R = S * (1 - P/100))
variable (h2 : S = R * (1 + 53.84615384615385 / 100))

theorem salary_reduction : P = 35 :=
by sorry

end salary_reduction_l195_195551


namespace pumpkins_total_weight_l195_195836

-- Define the weights of the pumpkins as given in the conditions
def first_pumpkin_weight : ℝ := 4
def second_pumpkin_weight : ℝ := 8.7

-- Prove that the total weight of the two pumpkins is 12.7 pounds
theorem pumpkins_total_weight : first_pumpkin_weight + second_pumpkin_weight = 12.7 := by
  sorry

end pumpkins_total_weight_l195_195836


namespace tammy_haircuts_l195_195034

theorem tammy_haircuts (total_haircuts free_haircuts haircuts_to_next_free : ℕ) 
(h1 : free_haircuts = 5) 
(h2 : haircuts_to_next_free = 5) 
(h3 : total_haircuts = 79) : 
(haircuts_to_next_free = 5) :=
by {
  sorry
}

end tammy_haircuts_l195_195034


namespace problem_1_problem_2_problem_3_l195_195593

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) : (∀ x : ℝ, f (x + 1) = x^2 + 4*x + 1) → (∀ x : ℝ, f x = x^2 + 2*x - 2) :=
by
  intro h
  sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) : (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) → (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) → (∀ x : ℝ, f x = x + 3) :=
by
  intros h1 h2
  sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) : (∀ x : ℝ, 2 * f x + f (1 / x) = 3 * x) → (∀ x : ℝ, f x = 2 * x - 1 / x) :=
by
  intro h
  sorry

end problem_1_problem_2_problem_3_l195_195593


namespace points_per_bag_l195_195217

/-
Wendy had 11 bags but didn't recycle 2 of them. She would have earned 
45 points for recycling all 11 bags. Prove that Wendy earns 5 points 
per bag of cans she recycles.
-/

def total_bags : Nat := 11
def unrecycled_bags : Nat := 2
def recycled_bags : Nat := total_bags - unrecycled_bags
def total_points : Nat := 45

theorem points_per_bag : total_points / recycled_bags = 5 := by
  sorry

end points_per_bag_l195_195217


namespace arman_sister_age_l195_195964

-- Define the conditions
variables (S : ℝ) -- Arman's sister's age four years ago
variable (A : ℝ) -- Arman's age four years ago

-- Given conditions as hypotheses
axiom h1 : A = 6 * S -- Arman is six times older than his sister
axiom h2 : A + 8 = 40 -- In 4 years, Arman's age will be 40 (hence, A in 4 years should be A + 8)

-- Main theorem to prove
theorem arman_sister_age (h1 : A = 6 * S) (h2 : A + 8 = 40) : S = 16 / 3 :=
by
  sorry

end arman_sister_age_l195_195964


namespace axis_of_symmetry_of_function_l195_195863

theorem axis_of_symmetry_of_function 
  (f : ℝ → ℝ)
  (h : ∀ x, f x = 3 * Real.cos x - Real.sqrt 3 * Real.sin x)
  : ∃ k : ℤ, x = k * Real.pi - Real.pi / 6 ∧ x = Real.pi - Real.pi / 6 :=
sorry

end axis_of_symmetry_of_function_l195_195863


namespace value_of_expression_l195_195305

theorem value_of_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -1) : -a^2 - b^2 + a * b = -21 := by
  sorry

end value_of_expression_l195_195305


namespace chicks_increased_l195_195113

theorem chicks_increased (chicks_day1 chicks_day2: ℕ) (H1 : chicks_day1 = 23) (H2 : chicks_day2 = 12) : 
  chicks_day1 + chicks_day2 = 35 :=
by
  sorry

end chicks_increased_l195_195113


namespace alpha_plus_beta_l195_195288

theorem alpha_plus_beta (α β : ℝ) (hα_range : -Real.pi / 2 < α ∧ α < Real.pi / 2)
    (hβ_range : -Real.pi / 2 < β ∧ β < Real.pi / 2)
    (h_roots : ∃ (x1 x2 : ℝ), x1 = Real.tan α ∧ x2 = Real.tan β ∧ (x1^2 + 3 * Real.sqrt 3 * x1 + 4 = 0) ∧ (x2^2 + 3 * Real.sqrt 3 * x2 + 4 = 0)) :
    α + β = -2 * Real.pi / 3 :=
sorry

end alpha_plus_beta_l195_195288


namespace sum_of_squares_l195_195169

variable {x y z a b c : Real}
variable (h₁ : x * y = a) (h₂ : x * z = b) (h₃ : y * z = c)
variable (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem sum_of_squares : x^2 + y^2 + z^2 = (a * b)^2 / (a * b * c) + (a * c)^2 / (a * b * c) + (b * c)^2 / (a * b * c) := 
sorry

end sum_of_squares_l195_195169


namespace find_g_neg_five_l195_195102

-- Given function and its properties
variables (g : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom ax2 : ∀ (x : ℝ), g x ≠ 0
axiom ax3 : g 5 = 2

-- Theorem to prove
theorem find_g_neg_five : g (-5) = 8 :=
sorry

end find_g_neg_five_l195_195102


namespace volume_of_four_cubes_l195_195130

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l195_195130


namespace find_x_l195_195259

theorem find_x (x : ℝ) (h : x^29 * 4^15 = 2 * 10^29) : x = 5 := 
by 
  sorry

end find_x_l195_195259


namespace base_number_of_exponentiation_l195_195278

theorem base_number_of_exponentiation (n : ℕ) (some_number : ℕ) (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^22) (h2 : n = 21) : some_number = 4 :=
  sorry

end base_number_of_exponentiation_l195_195278


namespace two_digit_numbers_condition_l195_195003

theorem two_digit_numbers_condition :
  ∃ (x y : ℕ), x > y ∧ x < 100 ∧ y < 100 ∧ x - y = 56 ∧ (x^2 % 100) = (y^2 % 100) ∧
  ((x = 78 ∧ y = 22) ∨ (x = 22 ∧ y = 78)) :=
by sorry

end two_digit_numbers_condition_l195_195003


namespace sum_of_digits_least_N_l195_195769

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 * N : ℚ) / 5⌉) / (N + 1)

theorem sum_of_digits_least_N (k : ℕ) (h_k : k = 2) (h1 : ∀ N, P N k < 8 / 10 ) :
  ∃ N : ℕ, (N % 10) + (N / 10) = 1 ∧ (P N k < 8 / 10) ∧ (∀ M : ℕ, M < N → P M k ≥ 8 / 10) := by
  sorry

end sum_of_digits_least_N_l195_195769


namespace book_E_chapters_l195_195285

def total_chapters: ℕ := 97
def chapters_A: ℕ := 17
def chapters_B: ℕ := chapters_A + 5
def chapters_C: ℕ := chapters_B - 7
def chapters_D: ℕ := chapters_C * 2
def chapters_sum : ℕ := chapters_A + chapters_B + chapters_C + chapters_D

theorem book_E_chapters :
  total_chapters - chapters_sum = 13 :=
by
  sorry

end book_E_chapters_l195_195285


namespace original_number_is_24_l195_195654

def number_parts (x y original_number : ℝ) : Prop :=
  7 * x + 5 * y = 146 ∧ x = 13 ∧ original_number = x + y

theorem original_number_is_24 :
  ∃ (x y original_number : ℝ), number_parts x y original_number ∧ original_number = 24 :=
by
  sorry

end original_number_is_24_l195_195654


namespace solve_problem_l195_195440

theorem solve_problem (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 5) : (b - d) ^ 2 = 16 :=
by
  sorry

end solve_problem_l195_195440


namespace pencil_and_pen_cost_l195_195029

theorem pencil_and_pen_cost
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 3.75)
  (h2 : 2 * p + 3 * q = 4.05) :
  p + q = 1.56 :=
by
  sorry

end pencil_and_pen_cost_l195_195029


namespace earl_up_second_time_l195_195453

def earl_floors (n top start up1 down up2 dist : ℕ) : Prop :=
  start + up1 - down + up2 = top - dist

theorem earl_up_second_time 
  (start up1 down top dist : ℕ) 
  (h_start : start = 1) 
  (h_up1 : up1 = 5) 
  (h_down : down = 2) 
  (h_top : top = 20) 
  (h_dist : dist = 9) : 
  ∃ up2, earl_floors n top start up1 down up2 dist ∧ up2 = 7 :=
by
  use 7
  sorry

end earl_up_second_time_l195_195453


namespace find_sum_of_abs_roots_l195_195193

variable {p q r n : ℤ}

theorem find_sum_of_abs_roots (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2024) (h3 : p * q * r = -n) :
  |p| + |q| + |r| = 100 :=
  sorry

end find_sum_of_abs_roots_l195_195193


namespace min_odd_solution_l195_195051

theorem min_odd_solution (a m1 m2 n1 n2 : ℕ)
  (h1: a = m1^2 + n1^2)
  (h2: a^2 = m2^2 + n2^2)
  (h3: m1 - n1 = m2 - n2)
  (h4: a > 5)
  (h5: a % 2 = 1) :
  a = 261 :=
sorry

end min_odd_solution_l195_195051


namespace katy_books_l195_195674

theorem katy_books (x : ℕ) (h : x + 2 * x + (2 * x - 3) = 37) : x = 8 :=
by
  sorry

end katy_books_l195_195674


namespace max_kopeyka_coins_l195_195415

def coins (n : Nat) (k : Nat) : Prop :=
  k ≤ n / 4 + 1

theorem max_kopeyka_coins : coins 2001 501 :=
by
  sorry

end max_kopeyka_coins_l195_195415


namespace answer_l195_195999

def p : Prop := ∃ x > Real.exp 1, (1 / 2)^x > Real.log x
def q : Prop := ∀ a b : Real, a > 1 → b > 1 → Real.log a / Real.log b + 2 * (Real.log b / Real.log a) ≥ 2 * Real.sqrt 2

theorem answer : ¬ p ∧ q :=
by
  have h1 : ¬ p := sorry
  have h2 : q := sorry
  exact ⟨h1, h2⟩

end answer_l195_195999


namespace professors_initial_count_l195_195531

noncomputable def initialNumberOfProfessors (failureGradesLastYear : ℕ) (failureGradesNextYear : ℕ) (increaseProfessors : ℕ) : ℕ :=
if (failureGradesLastYear, failureGradesNextYear, increaseProfessors) = (6480, 11200, 3) then 5 else sorry

theorem professors_initial_count :
  initialNumberOfProfessors 6480 11200 3 = 5 := by {
  sorry
}

end professors_initial_count_l195_195531


namespace line_through_midpoint_l195_195073

theorem line_through_midpoint (x y : ℝ)
  (ellipse : x^2 / 25 + y^2 / 16 = 1)
  (midpoint : P = (2, 1)) :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (x = 32*y - 25*x - 89) :=
sorry

end line_through_midpoint_l195_195073


namespace evan_books_in_ten_years_l195_195644

def E4 : ℕ := 400
def E_now : ℕ := E4 - 80
def E2 : ℕ := E_now / 2
def E10 : ℕ := 6 * E2 + 120

theorem evan_books_in_ten_years : E10 = 1080 := by
sorry

end evan_books_in_ten_years_l195_195644


namespace comparison_inequalities_l195_195886

open Real

theorem comparison_inequalities
  (m : ℝ) (h1 : 3 ^ m = Real.exp 1) 
  (a : ℝ) (h2 : a = cos m) 
  (b : ℝ) (h3 : b = 1 - 1/2 * m^2)
  (c : ℝ) (h4 : c = sin m / m) :
  c > a ∧ a > b := by
  sorry

end comparison_inequalities_l195_195886


namespace trajectory_equation_l195_195599

theorem trajectory_equation :
  ∀ (N : ℝ × ℝ), (∃ (F : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (∃ b : ℝ, P = (0, b)) ∧ 
    (∃ a : ℝ, a ≠ 0 ∧ M = (a, 0)) ∧ 
    (N.fst = -(M.fst) ∧ N.snd = 2 * P.snd) ∧ 
    ((-M.fst) * F.fst + (-(M.snd)) * (-(P.snd)) = 0) ∧ 
    ((-M.fst, -M.snd) + (N.fst, N.snd) = (0,0))) → 
  (N.snd)^2 = 4 * (N.fst) :=
by
  intros N h
  sorry

end trajectory_equation_l195_195599


namespace maximal_x2009_l195_195020

theorem maximal_x2009 (x : ℕ → ℝ) 
    (h_seq : ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0)
    (h_x0 : x 0 = 1)
    (h_x20 : x 20 = 9)
    (h_x200 : x 200 = 6) :
    x 2009 ≤ 6 :=
sorry

end maximal_x2009_l195_195020


namespace soap_box_width_l195_195937

theorem soap_box_width
  (carton_length : ℝ) (carton_width : ℝ) (carton_height : ℝ)
  (box_length : ℝ) (box_height : ℝ) (max_boxes : ℝ) (carton_volume : ℝ)
  (box_volume : ℝ) (W : ℝ) : 
  carton_length = 25 →
  carton_width = 42 →
  carton_height = 60 →
  box_length = 6 →
  box_height = 6 →
  max_boxes = 250 →
  carton_volume = carton_length * carton_width * carton_height →
  box_volume = box_length * W * box_height →
  max_boxes * box_volume = carton_volume →
  W = 7 :=
sorry

end soap_box_width_l195_195937


namespace total_cleaning_time_l195_195949

-- Definition for the problem conditions
def time_to_clean_egg (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ) : ℕ :=
  (num_eggs * seconds_per_egg) / seconds_per_minute

def time_to_clean_toilet_paper (minutes_per_roll : ℕ) (num_rolls : ℕ) : ℕ :=
  num_rolls * minutes_per_roll

-- Main statement to prove the total cleaning time
theorem total_cleaning_time
  (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ)
  (minutes_per_roll : ℕ) (num_rolls : ℕ) :
  seconds_per_egg = 15 →
  num_eggs = 60 →
  seconds_per_minute = 60 →
  minutes_per_roll = 30 →
  num_rolls = 7 →
  time_to_clean_egg seconds_per_egg num_eggs seconds_per_minute +
  time_to_clean_toilet_paper minutes_per_roll num_rolls = 225 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cleaning_time_l195_195949


namespace meaning_of_implication_l195_195281

theorem meaning_of_implication (p q : Prop) : (p → q) = ((p → q) = True) :=
sorry

end meaning_of_implication_l195_195281


namespace average_interest_rate_l195_195993

theorem average_interest_rate (I : ℝ) (r1 r2 : ℝ) (y : ℝ)
  (h0 : I = 6000)
  (h1 : r1 = 0.05)
  (h2 : r2 = 0.07)
  (h3 : 0.05 * (6000 - y) = 0.07 * y) :
  ((r1 * (I - y) + r2 * y) / I) = 0.05833 :=
by
  sorry

end average_interest_rate_l195_195993


namespace system1_solution_l195_195326

variable (x y : ℝ)

theorem system1_solution :
  (3 * x - y = -1) ∧ (x + 2 * y = 9) ↔ (x = 1) ∧ (y = 4) := by
  sorry

end system1_solution_l195_195326


namespace product_of_large_integers_l195_195830

theorem product_of_large_integers :
  ∃ A B : ℤ, A > 10^2009 ∧ B > 10^2009 ∧ A * B = 3^(4^5) + 4^(5^6) :=
by
  sorry

end product_of_large_integers_l195_195830


namespace tan_beta_solution_l195_195719

theorem tan_beta_solution
  (α β : ℝ)
  (h₁ : Real.tan α = 2)
  (h₂ : Real.tan (α + β) = -1) :
  Real.tan β = 3 := 
sorry

end tan_beta_solution_l195_195719


namespace sum_binom_equals_220_l195_195459

/-- The binomial coefficient C(n, k) -/
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

/-- Prove that the sum C(2, 2) + C(3, 2) + C(4, 2) + ... + C(11, 2) equals 220 using the 
    binomial coefficient property C(n, r+1) + C(n, r) = C(n+1, r+1) -/
theorem sum_binom_equals_220 :
  binom 2 2 + binom 3 2 + binom 4 2 + binom 5 2 + binom 6 2 + binom 7 2 + 
  binom 8 2 + binom 9 2 + binom 10 2 + binom 11 2 = 220 := by
sorry

end sum_binom_equals_220_l195_195459


namespace compound_interest_doubling_time_l195_195478

theorem compound_interest_doubling_time :
  ∃ t : ℕ, (2 : ℝ) < (1 + (0.13 : ℝ))^t ∧ t = 6 :=
by
  sorry

end compound_interest_doubling_time_l195_195478


namespace isosceles_triangle_base_angle_l195_195449

theorem isosceles_triangle_base_angle (x : ℝ) 
  (h1 : ∀ (a b : ℝ), a + b + (20 + 2 * b) = 180)
  (h2 : 20 + 2 * x = 180 - 2 * x - x) : x = 40 :=
by sorry

end isosceles_triangle_base_angle_l195_195449


namespace abigail_money_loss_l195_195584

theorem abigail_money_loss {initial spent remaining lost : ℤ} 
  (h1 : initial = 11) 
  (h2 : spent = 2) 
  (h3 : remaining = 3) 
  (h4 : lost = initial - spent - remaining) : 
  lost = 6 := sorry

end abigail_money_loss_l195_195584
