import Mathlib

namespace NUMINAMATH_GPT_sarah_total_distance_walked_l1914_191443

noncomputable def total_distance : ℝ :=
  let rest_time : ℝ := 1 / 3
  let total_time : ℝ := 3.5
  let time_spent_walking : ℝ := total_time - rest_time -- time spent walking
  let uphill_speed : ℝ := 3 -- in mph
  let downhill_speed : ℝ := 4 -- in mph
  let d := time_spent_walking * (uphill_speed * downhill_speed) / (uphill_speed + downhill_speed) -- half distance D
  2 * d

theorem sarah_total_distance_walked :
  total_distance = 10.858 := sorry

end NUMINAMATH_GPT_sarah_total_distance_walked_l1914_191443


namespace NUMINAMATH_GPT_reciprocal_of_3_div_2_l1914_191459

def reciprocal (a : ℚ) : ℚ := a⁻¹

theorem reciprocal_of_3_div_2 : reciprocal (3 / 2) = 2 / 3 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_reciprocal_of_3_div_2_l1914_191459


namespace NUMINAMATH_GPT_math_problem_solution_l1914_191432

theorem math_problem_solution : 8 / 4 - 3 - 9 + 3 * 9 = 17 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_solution_l1914_191432


namespace NUMINAMATH_GPT_max_correct_answers_l1914_191463

theorem max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 25) 
  (h2 : 5 * c - 2 * w = 60) : 
  c ≤ 14 := 
sorry

end NUMINAMATH_GPT_max_correct_answers_l1914_191463


namespace NUMINAMATH_GPT_number_of_pages_l1914_191460

-- Define the conditions
def rate_of_printer_A (P : ℕ) : ℕ := P / 60
def rate_of_printer_B (P : ℕ) : ℕ := (P / 60) + 6

-- Define the combined rate condition
def combined_rate (P : ℕ) (R_A R_B : ℕ) : Prop := (R_A + R_B) = P / 24

-- The main theorem to prove
theorem number_of_pages :
  ∃ (P : ℕ), combined_rate P (rate_of_printer_A P) (rate_of_printer_B P) ∧ P = 720 := by
  sorry

end NUMINAMATH_GPT_number_of_pages_l1914_191460


namespace NUMINAMATH_GPT_second_smallest_five_digit_in_pascal_l1914_191427

theorem second_smallest_five_digit_in_pascal :
  ∃ (x : ℕ), (x > 10000) ∧ (∀ y : ℕ, (y ≠ 10000) → (y < x) → (y < 10000)) ∧ (x = 10001) :=
sorry

end NUMINAMATH_GPT_second_smallest_five_digit_in_pascal_l1914_191427


namespace NUMINAMATH_GPT_start_page_day2_correct_l1914_191481

variables (total_pages : ℕ) (percentage_read_day1 : ℝ) (start_page_day2 : ℕ)

theorem start_page_day2_correct
  (h1 : total_pages = 200)
  (h2 : percentage_read_day1 = 0.2)
  : start_page_day2 = total_pages * percentage_read_day1 + 1 :=
by
  sorry

end NUMINAMATH_GPT_start_page_day2_correct_l1914_191481


namespace NUMINAMATH_GPT_ratio_of_areas_of_similar_triangles_l1914_191453

-- Define the variables and conditions
variables {ABC DEF : Type} 
variables (hABCDEF : Similar ABC DEF) 
variables (perimeterABC perimeterDEF : ℝ)
variables (hpABC : perimeterABC = 3)
variables (hpDEF : perimeterDEF = 1)

-- The theorem statement
theorem ratio_of_areas_of_similar_triangles :
  (perimeterABC / perimeterDEF) ^ 2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_similar_triangles_l1914_191453


namespace NUMINAMATH_GPT_sym_diff_A_B_l1914_191454

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- Definition of the symmetric difference
def sym_diff (A B : Set ℕ) : Set ℕ := {x | (x ∈ A ∨ x ∈ B) ∧ x ∉ (A ∩ B)}

theorem sym_diff_A_B : sym_diff A B = {0, 3} := 
by 
  sorry

end NUMINAMATH_GPT_sym_diff_A_B_l1914_191454


namespace NUMINAMATH_GPT_probability_of_stock_price_increase_l1914_191446

namespace StockPriceProbability

variables (P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C : ℝ)

def P_D : ℝ := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

theorem probability_of_stock_price_increase :
    P_A = 0.6 → P_B = 0.3 → P_C = 0.1 → 
    P_D_given_A = 0.7 → P_D_given_B = 0.2 → P_D_given_C = 0.1 → 
    P_D P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C = 0.49 :=
by intros h₁ h₂ h₃ h₄ h₅ h₆; sorry

end StockPriceProbability

end NUMINAMATH_GPT_probability_of_stock_price_increase_l1914_191446


namespace NUMINAMATH_GPT_total_crayons_l1914_191484

def original_crayons := 41
def added_crayons := 12

theorem total_crayons : original_crayons + added_crayons = 53 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l1914_191484


namespace NUMINAMATH_GPT_coordinates_satisfy_l1914_191486

theorem coordinates_satisfy (x y : ℝ) : y * (x + 1) = x^2 - 1 ↔ (x = -1 ∨ y = x - 1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_satisfy_l1914_191486


namespace NUMINAMATH_GPT_mr_johnson_pill_intake_l1914_191456

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end NUMINAMATH_GPT_mr_johnson_pill_intake_l1914_191456


namespace NUMINAMATH_GPT_probability_not_all_same_color_l1914_191482

def num_colors := 3
def draws := 3
def total_outcomes := num_colors ^ draws

noncomputable def prob_same_color : ℚ := (3 / total_outcomes)
noncomputable def prob_not_same_color : ℚ := 1 - prob_same_color

theorem probability_not_all_same_color :
  prob_not_same_color = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_all_same_color_l1914_191482


namespace NUMINAMATH_GPT_probability_length_error_in_interval_l1914_191448

noncomputable def normal_dist_prob (μ σ : ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

theorem probability_length_error_in_interval :
  normal_dist_prob 0 3 3 6 = 0.1359 :=
by
  sorry

end NUMINAMATH_GPT_probability_length_error_in_interval_l1914_191448


namespace NUMINAMATH_GPT_candies_bought_is_18_l1914_191405

-- Define the original number of candies
def original_candies : ℕ := 9

-- Define the total number of candies after buying more
def total_candies : ℕ := 27

-- Define the function to calculate the number of candies bought
def candies_bought (o t : ℕ) : ℕ := t - o

-- The main theorem stating that the number of candies bought is 18
theorem candies_bought_is_18 : candies_bought original_candies total_candies = 18 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_candies_bought_is_18_l1914_191405


namespace NUMINAMATH_GPT_students_per_class_l1914_191465

-- Define the conditions
variables (c : ℕ) (h_c : c ≥ 1) (s : ℕ)

-- Define the total number of books read by one student per year
def books_per_student_per_year := 5 * 12

-- Define the total number of students
def total_number_of_students := c * s

-- Define the total number of books read by the entire student body
def total_books_read := total_number_of_students * books_per_student_per_year

-- The given condition that the entire student body reads 60 books in one year
axiom total_books_eq_60 : total_books_read = 60

theorem students_per_class (h_c : c ≥ 1) : s = 1 / c :=
by sorry

end NUMINAMATH_GPT_students_per_class_l1914_191465


namespace NUMINAMATH_GPT_second_number_l1914_191431

theorem second_number (x : ℕ) (h1 : ∃ k : ℕ, 1428 = 129 * k + 9)
  (h2 : ∃ m : ℕ, x = 129 * m + 13) (h_gcd : ∀ (d : ℕ), d ∣ (1428 - 9 : ℕ) ∧ d ∣ (x - 13 : ℕ) → d ≤ 129) :
  x = 1561 :=
by
  sorry

end NUMINAMATH_GPT_second_number_l1914_191431


namespace NUMINAMATH_GPT_find_C_D_l1914_191499

theorem find_C_D : ∃ C D, 
  (∀ x, x ≠ 3 → x ≠ 5 → (6*x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) ∧ 
  C = -15/2 ∧ D = 27/2 := by
  sorry

end NUMINAMATH_GPT_find_C_D_l1914_191499


namespace NUMINAMATH_GPT_tangent_line_through_origin_l1914_191490

theorem tangent_line_through_origin (f : ℝ → ℝ) (x : ℝ) (H1 : ∀ x < 0, f x = Real.log (-x))
  (H2 : ∀ x < 0, DifferentiableAt ℝ f x) (H3 : ∀ (x₀ : ℝ), x₀ < 0 → x₀ = -Real.exp 1 → deriv f x₀ = -1 / Real.exp 1)
  : ∀ x, -Real.exp 1 = x → ∀ y, y = -1 / Real.exp 1 * x → y = 0 → y = -1 / Real.exp 1 * x :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_origin_l1914_191490


namespace NUMINAMATH_GPT_overhead_cost_calculation_l1914_191479

-- Define the production cost per performance
def production_cost_performance : ℕ := 7000

-- Define the revenue per sold-out performance
def revenue_per_soldout_performance : ℕ := 16000

-- Define the number of performances needed to break even
def break_even_performances : ℕ := 9

-- Prove the overhead cost
theorem overhead_cost_calculation (O : ℕ) :
  (O + break_even_performances * production_cost_performance = break_even_performances * revenue_per_soldout_performance) →
  O = 81000 :=
by
  sorry

end NUMINAMATH_GPT_overhead_cost_calculation_l1914_191479


namespace NUMINAMATH_GPT_bottles_stolen_at_dance_l1914_191421

-- Define the initial conditions
def initial_bottles := 10
def bottles_lost_at_school := 2
def total_stickers := 21
def stickers_per_bottle := 3

-- Calculate remaining bottles after loss at school
def remaining_bottles_after_school := initial_bottles - bottles_lost_at_school

-- Calculate the remaining bottles after the theft
def remaining_bottles_after_theft := total_stickers / stickers_per_bottle

-- Prove the number of bottles stolen
theorem bottles_stolen_at_dance : remaining_bottles_after_school - remaining_bottles_after_theft = 1 :=
by
  sorry

end NUMINAMATH_GPT_bottles_stolen_at_dance_l1914_191421


namespace NUMINAMATH_GPT_figure_100_squares_l1914_191440

theorem figure_100_squares : (∃ f : ℕ → ℕ, f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 100 = 30301) :=
  sorry

end NUMINAMATH_GPT_figure_100_squares_l1914_191440


namespace NUMINAMATH_GPT_neg_exists_is_forall_l1914_191450

theorem neg_exists_is_forall: 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_exists_is_forall_l1914_191450


namespace NUMINAMATH_GPT_range_of_m_l1914_191497

theorem range_of_m (m : ℝ) (p : Prop) (q : Prop)
  (hp : (2 * m)^2 - 4 ≥ 0 ↔ p)
  (hq : 1 < (Real.sqrt (5 + m)) / (Real.sqrt 5) ∧ (Real.sqrt (5 + m)) / (Real.sqrt 5) < 2 ↔ q)
  (hnq : ¬q = False)
  (hpq : (p ∧ q) = False) :
  0 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1914_191497


namespace NUMINAMATH_GPT_expected_value_eight_l1914_191433

-- Define the 10-sided die roll outcomes
def outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the value function for a roll outcome
def value (x : ℕ) : ℕ :=
  if x % 2 = 0 then x  -- even value
  else 2 * x  -- odd value

-- Calculate the expected value
def expected_value : ℚ :=
  (1 / 10 : ℚ) * (2 + 2 + 6 + 4 + 10 + 6 + 14 + 8 + 18 + 10)

-- The theorem stating the expected value equals 8
theorem expected_value_eight :
  expected_value = 8 := by
  sorry

end NUMINAMATH_GPT_expected_value_eight_l1914_191433


namespace NUMINAMATH_GPT_original_average_speed_l1914_191480

theorem original_average_speed :
  ∀ (D : ℝ),
  (V = D / (5 / 6)) ∧ (60 = D / (2 / 3)) → V = 48 :=
by
  sorry

end NUMINAMATH_GPT_original_average_speed_l1914_191480


namespace NUMINAMATH_GPT_number_of_friends_l1914_191425

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l1914_191425


namespace NUMINAMATH_GPT_smallest_number_divisible_l1914_191457

theorem smallest_number_divisible (n : ℤ) : 
  (n + 7) % 25 = 0 ∧
  (n + 7) % 49 = 0 ∧
  (n + 7) % 15 = 0 ∧
  (n + 7) % 21 = 0 ↔ n = 3668 :=
by 
 sorry

end NUMINAMATH_GPT_smallest_number_divisible_l1914_191457


namespace NUMINAMATH_GPT_zoe_recycled_correctly_l1914_191449

-- Let Z be the number of pounds recycled by Zoe
def pounds_by_zoe (total_points : ℕ) (friends_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_points * pounds_per_point - friends_pounds

-- Given conditions
def total_points : ℕ := 6
def friends_pounds : ℕ := 23
def pounds_per_point : ℕ := 8

-- Lean statement for the proof problem
theorem zoe_recycled_correctly : pounds_by_zoe total_points friends_pounds pounds_per_point = 25 :=
by
  -- proof to be provided here
  sorry

end NUMINAMATH_GPT_zoe_recycled_correctly_l1914_191449


namespace NUMINAMATH_GPT_max_working_groups_l1914_191402

theorem max_working_groups (teachers groups : ℕ) (memberships_per_teacher group_size : ℕ) 
  (h_teachers : teachers = 36) (h_memberships_per_teacher : memberships_per_teacher = 2)
  (h_group_size : group_size = 4) 
  (h_max_memberships : teachers * memberships_per_teacher = 72) :
  groups ≤ 18 :=
by
  sorry

end NUMINAMATH_GPT_max_working_groups_l1914_191402


namespace NUMINAMATH_GPT_train_average_speed_l1914_191426

-- Define the variables used in the conditions
variables (D V : ℝ)
-- Condition: Distance D in 50 minutes at average speed V kmph
-- 50 minutes to hours conversion
def condition1 : D = V * (50 / 60) := sorry
-- Condition: Distance D in 40 minutes at speed 60 kmph
-- 40 minutes to hours conversion
def condition2 : D = 60 * (40 / 60) := sorry

-- Claim: Current average speed V
theorem train_average_speed : V = 48 :=
by
  -- Using the conditions to prove the claim
  sorry

end NUMINAMATH_GPT_train_average_speed_l1914_191426


namespace NUMINAMATH_GPT_M_gt_N_l1914_191415

-- Define the variables and conditions
variables (x y : ℝ)
noncomputable def M := x^2 + y^2
noncomputable def N := 2*x + 6*y - 11

-- State the theorem
theorem M_gt_N : M x y > N x y := by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_M_gt_N_l1914_191415


namespace NUMINAMATH_GPT_corrected_observations_mean_l1914_191494

noncomputable def corrected_mean (mean incorrect correct: ℚ) (n: ℕ) : ℚ :=
  let S_incorrect := mean * n
  let Difference := correct - incorrect
  let S_corrected := S_incorrect + Difference
  S_corrected / n

theorem corrected_observations_mean:
  corrected_mean 36 23 34 50 = 36.22 := by
  sorry

end NUMINAMATH_GPT_corrected_observations_mean_l1914_191494


namespace NUMINAMATH_GPT_evaluate_expression_l1914_191423

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1914_191423


namespace NUMINAMATH_GPT_equilateral_triangle_distances_l1914_191441

-- Defining the necessary conditions
variables {h x y z : ℝ}
variables (hx : 0 < h) (hx_cond : x + y + z = h)
variables (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y)

-- Lean 4 statement to express the proof problem
theorem equilateral_triangle_distances (hx : 0 < h) (hx_cond : x + y + z = h) (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x < h / 2 ∧ y < h / 2 ∧ z < h / 2 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_distances_l1914_191441


namespace NUMINAMATH_GPT_smallest_possible_value_of_c_l1914_191434

theorem smallest_possible_value_of_c (b c : ℝ) (h1 : 1 < b) (h2 : b < c)
    (h3 : ¬∃ (u v w : ℝ), u = 1 ∧ v = b ∧ w = c ∧ u + v > w ∧ u + w > v ∧ v + w > u)
    (h4 : ¬∃ (x y z : ℝ), x = 1 ∧ y = 1/b ∧ z = 1/c ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
    c = (5 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_c_l1914_191434


namespace NUMINAMATH_GPT_taxi_ride_cost_l1914_191496

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1914_191496


namespace NUMINAMATH_GPT_range_of_x_l1914_191438

theorem range_of_x (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (x + 2) / (x - 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l1914_191438


namespace NUMINAMATH_GPT_average_minutes_run_l1914_191464

-- Definitions
def third_graders (fi : ℕ) : ℕ := 6 * fi
def fourth_graders (fi : ℕ) : ℕ := 2 * fi
def fifth_graders (fi : ℕ) : ℕ := fi

-- Number of minutes run by each grade
def third_graders_minutes : ℕ := 10
def fourth_graders_minutes : ℕ := 18
def fifth_graders_minutes : ℕ := 8

-- Main theorem
theorem average_minutes_run 
  (fi : ℕ) 
  (t := third_graders fi) 
  (fr := fourth_graders fi) 
  (f := fifth_graders fi) 
  (minutes_total := 10 * t + 18 * fr + 8 * f) 
  (students_total := t + fr + f) :
  (students_total > 0) →
  (minutes_total : ℚ) / students_total = 104 / 9 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_l1914_191464


namespace NUMINAMATH_GPT_growth_rate_equation_l1914_191489

-- Given conditions
def revenue_january : ℕ := 36
def revenue_march : ℕ := 48

-- Problem statement
theorem growth_rate_equation (x : ℝ) 
  (h_january : revenue_january = 36)
  (h_march : revenue_march = 48) :
  36 * (1 + x) ^ 2 = 48 :=
sorry

end NUMINAMATH_GPT_growth_rate_equation_l1914_191489


namespace NUMINAMATH_GPT_n_minus_m_eq_zero_l1914_191462

-- Definitions based on the conditions
def m : ℝ := sorry
def n : ℝ := sorry
def i := Complex.I
def condition : Prop := m + i = (1 + 2 * i) - n * i

-- The theorem stating the equivalence proof problem
theorem n_minus_m_eq_zero (h : condition) : n - m = 0 :=
sorry

end NUMINAMATH_GPT_n_minus_m_eq_zero_l1914_191462


namespace NUMINAMATH_GPT_difference_in_surface_areas_l1914_191455

-- Define the conditions: volumes and number of cubes
def V_large : ℕ := 343
def n : ℕ := 343
def V_small : ℕ := 1

-- Define the function to calculate the side length of a cube given its volume
def side_length (V : ℕ) : ℕ := V^(1/3 : ℕ)

-- Specify the side lengths of the larger and smaller cubes
def s_large : ℕ := side_length V_large
def s_small : ℕ := side_length V_small

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- Specify the surface areas of the larger cube and the total of the smaller cubes
def SA_large : ℕ := surface_area s_large
def SA_small_total : ℕ := n * surface_area s_small

-- State the theorem to prove
theorem difference_in_surface_areas : SA_small_total - SA_large = 1764 :=
by {
  -- Intentionally omit proof, as per instructions
  sorry
}

end NUMINAMATH_GPT_difference_in_surface_areas_l1914_191455


namespace NUMINAMATH_GPT_diameter_of_large_circle_is_19_312_l1914_191411

noncomputable def diameter_large_circle (r_small : ℝ) (n : ℕ) : ℝ :=
  let side_length_inner_octagon := 2 * r_small
  let radius_inner_octagon := side_length_inner_octagon / (2 * Real.sin (Real.pi / n)) / 2
  let radius_large_circle := radius_inner_octagon + r_small
  2 * radius_large_circle

theorem diameter_of_large_circle_is_19_312 :
  diameter_large_circle 4 8 = 19.312 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_large_circle_is_19_312_l1914_191411


namespace NUMINAMATH_GPT_total_gas_cost_l1914_191492

theorem total_gas_cost 
  (x : ℝ)
  (cost_per_person_initial : ℝ := x / 5)
  (cost_per_person_new : ℝ := x / 8)
  (cost_difference : cost_per_person_initial - cost_per_person_new = 15) :
  x = 200 :=
sorry

end NUMINAMATH_GPT_total_gas_cost_l1914_191492


namespace NUMINAMATH_GPT_transformation_invariant_l1914_191409

-- Define the initial and transformed parabolas
def initial_parabola (x : ℝ) : ℝ := 2 * x^2
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3

-- Define the transformation process
def move_right_1 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def move_up_3 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

-- Concatenate transformations to form the final transformation
def combined_transformation (x : ℝ) : ℝ :=
  move_up_3 (move_right_1 initial_parabola) x

-- Statement to prove
theorem transformation_invariant :
  ∀ x : ℝ, combined_transformation x = transformed_parabola x := 
by {
  sorry
}

end NUMINAMATH_GPT_transformation_invariant_l1914_191409


namespace NUMINAMATH_GPT_Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l1914_191428

open Complex

noncomputable def Z (m : ℝ) : ℂ :=
  (m ^ 2 + 5 * m + 6) + (m ^ 2 - 2 * m - 15) * Complex.I

namespace ComplexNumbersProofs

-- Prove that Z is a real number if and only if m = -3 or m = 5
theorem Z_real_iff_m_eq_neg3_or_5 (m : ℝ) :
  (Z m).im = 0 ↔ (m = -3 ∨ m = 5) := 
by
  sorry

-- Prove that Z is a pure imaginary number if and only if m = -2
theorem Z_pure_imaginary_iff_m_eq_neg2 (m : ℝ) :
  (Z m).re = 0 ↔ (m = -2) := 
by
  sorry

-- Prove that the point corresponding to Z lies in the fourth quadrant if and only if -2 < m < 5
theorem Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5 (m : ℝ) :
  (Z m).re > 0 ∧ (Z m).im < 0 ↔ (-2 < m ∧ m < 5) :=
by
  sorry

end ComplexNumbersProofs

end NUMINAMATH_GPT_Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l1914_191428


namespace NUMINAMATH_GPT_max_area_parabola_l1914_191472

open Real

noncomputable def max_area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem max_area_parabola (a b c : ℝ) 
  (ha : a^2 = (a * a))
  (hb : b^2 = (b * b))
  (hc : c^2 = (c * c))
  (centroid_cond1 : (a + b + c) = 4)
  (centroid_cond2 : (a^2 + b^2 + c^2) = 6)
  : max_area_of_triangle (a^2, a) (b^2, b) (c^2, c) = (sqrt 3) / 9 := 
sorry

end NUMINAMATH_GPT_max_area_parabola_l1914_191472


namespace NUMINAMATH_GPT_sqrt_of_4_l1914_191437

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_4_l1914_191437


namespace NUMINAMATH_GPT_bananas_used_l1914_191468

-- Define the conditions
def bananas_per_loaf := 4
def loaves_monday := 3
def loaves_tuesday := 2 * loaves_monday

-- Define the total bananas used
def bananas_monday := loaves_monday * bananas_per_loaf
def bananas_tuesday := loaves_tuesday * bananas_per_loaf
def total_bananas := bananas_monday + bananas_tuesday

-- Theorem statement to prove the total bananas used is 36
theorem bananas_used : total_bananas = 36 := by
  sorry

end NUMINAMATH_GPT_bananas_used_l1914_191468


namespace NUMINAMATH_GPT_sail_pressure_l1914_191444

def pressure (k A V : ℝ) : ℝ := k * A * V^2

theorem sail_pressure (k : ℝ)
  (h_k : k = 1 / 800) 
  (A : ℝ) 
  (V : ℝ) 
  (P : ℝ)
  (h_initial : A = 1 ∧ V = 20 ∧ P = 0.5) 
  (A2 : ℝ) 
  (V2 : ℝ) 
  (h_doubled : A2 = 2 ∧ V2 = 30) :
  pressure k A2 V2 = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_sail_pressure_l1914_191444


namespace NUMINAMATH_GPT_sum_of_integers_l1914_191495

theorem sum_of_integers (n : ℤ) (h : n * (n + 2) = 20400) : n + (n + 2) = 286 ∨ n + (n + 2) = -286 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1914_191495


namespace NUMINAMATH_GPT_alice_vs_bob_payment_multiple_l1914_191401

theorem alice_vs_bob_payment_multiple :
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  total_alice_payment / bob_payment = 9 := by
  -- define the variables as per the conditions
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  -- define the target statement
  show total_alice_payment / bob_payment = 9
  sorry

end NUMINAMATH_GPT_alice_vs_bob_payment_multiple_l1914_191401


namespace NUMINAMATH_GPT_triangle_proportion_l1914_191416

theorem triangle_proportion (p q r x y : ℝ)
  (h1 : x / q = y / r)
  (h2 : x + y = p) :
  y / r = p / (q + r) := sorry

end NUMINAMATH_GPT_triangle_proportion_l1914_191416


namespace NUMINAMATH_GPT_smallest_clock_equiv_to_square_greater_than_10_l1914_191445

def clock_equiv (h k : ℕ) : Prop :=
  (h % 12) = (k % 12)

theorem smallest_clock_equiv_to_square_greater_than_10 : ∃ h > 10, clock_equiv h (h * h) ∧ ∀ h' > 10, clock_equiv h' (h' * h') → h ≤ h' :=
by
  sorry

end NUMINAMATH_GPT_smallest_clock_equiv_to_square_greater_than_10_l1914_191445


namespace NUMINAMATH_GPT_least_common_multiple_of_20_45_75_l1914_191474

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end NUMINAMATH_GPT_least_common_multiple_of_20_45_75_l1914_191474


namespace NUMINAMATH_GPT_tennis_to_soccer_ratio_l1914_191422

theorem tennis_to_soccer_ratio
  (total_balls : ℕ)
  (soccer_balls : ℕ)
  (basketball_offset : ℕ)
  (baseball_offset : ℕ)
  (volleyballs : ℕ)
  (tennis_balls : ℕ)
  (total_balls_eq : total_balls = 145)
  (soccer_balls_eq : soccer_balls = 20)
  (basketball_count : soccer_balls + basketball_offset = 20 + 5)
  (baseball_count : soccer_balls + baseball_offset = 20 + 10)
  (volleyballs_eq : volleyballs = 30)
  (accounted_balls : soccer_balls + (soccer_balls + basketball_offset) + (soccer_balls + baseball_offset) + volleyballs = 105)
  (tennis_balls_eq : tennis_balls = 145 - 105) :
  tennis_balls / soccer_balls = 2 :=
sorry

end NUMINAMATH_GPT_tennis_to_soccer_ratio_l1914_191422


namespace NUMINAMATH_GPT_probability_prime_sum_l1914_191483

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_prime_sum_l1914_191483


namespace NUMINAMATH_GPT_normal_distribution_interval_probability_l1914_191493

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
sorry

theorem normal_distribution_interval_probability
  (σ : ℝ) (hσ : σ > 0)
  (hprob : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8) :
  (normal_cdf 1 σ 2 - normal_cdf 1 σ 1) = 0.4 :=
sorry

end NUMINAMATH_GPT_normal_distribution_interval_probability_l1914_191493


namespace NUMINAMATH_GPT_find_function_satisfying_condition_l1914_191412

theorem find_function_satisfying_condition :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)) → 
                          (∀ x : ℝ, f x = 2 * x + c) :=
sorry

end NUMINAMATH_GPT_find_function_satisfying_condition_l1914_191412


namespace NUMINAMATH_GPT_fuel_consumption_l1914_191408

open Real

theorem fuel_consumption (initial_fuel : ℝ) (final_fuel : ℝ) (distance_covered : ℝ) (consumption_rate : ℝ) (fuel_left : ℝ) (x : ℝ) :
  initial_fuel = 60 ∧ final_fuel = 50 ∧ distance_covered = 100 ∧ 
  consumption_rate = (initial_fuel - final_fuel) / distance_covered ∧ consumption_rate = 0.1 ∧ 
  fuel_left = initial_fuel - consumption_rate * x ∧ x = 260 →
  fuel_left = 34 :=
by
  sorry

end NUMINAMATH_GPT_fuel_consumption_l1914_191408


namespace NUMINAMATH_GPT_rational_sum_abs_ratios_l1914_191403

theorem rational_sum_abs_ratios (a b c : ℚ) (h : |a * b * c| / (a * b * c) = 1) : (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) := 
sorry

end NUMINAMATH_GPT_rational_sum_abs_ratios_l1914_191403


namespace NUMINAMATH_GPT_geometric_progression_l1914_191469

theorem geometric_progression (b q : ℝ) :
  (b + b*q + b*q^2 + b*q^3 = -40) ∧ 
  (b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280) →
  (b = 2 ∧ q = -3) ∨ (b = -54 ∧ q = -1/3) :=
by sorry

end NUMINAMATH_GPT_geometric_progression_l1914_191469


namespace NUMINAMATH_GPT_total_trip_cost_l1914_191420

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end NUMINAMATH_GPT_total_trip_cost_l1914_191420


namespace NUMINAMATH_GPT_union_complement_eq_l1914_191413

open Set

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1}
def B : Set ℕ := {1, 2}

theorem union_complement_eq : A ∪ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l1914_191413


namespace NUMINAMATH_GPT_Ryan_dig_time_alone_l1914_191419

theorem Ryan_dig_time_alone :
  ∃ R : ℝ, ∀ Castel_time together_time,
    Castel_time = 6 ∧ together_time = 30 / 11 →
    (1 / R + 1 / Castel_time = 11 / 30) →
    R = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Ryan_dig_time_alone_l1914_191419


namespace NUMINAMATH_GPT_daphney_potatoes_l1914_191478

theorem daphney_potatoes (cost_per_2kg : ℕ) (total_paid : ℕ) (amount_per_kg : ℕ) (kg_bought : ℕ) 
  (h1 : cost_per_2kg = 6) (h2 : total_paid = 15) (h3 : amount_per_kg = cost_per_2kg / 2) 
  (h4 : kg_bought = total_paid / amount_per_kg) : kg_bought = 5 :=
by
  sorry

end NUMINAMATH_GPT_daphney_potatoes_l1914_191478


namespace NUMINAMATH_GPT_matrix_determinant_transformation_l1914_191414

theorem matrix_determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
  (p * (5 * r + 4 * s) - r * (5 * p + 4 * q)) = -12 :=
sorry

end NUMINAMATH_GPT_matrix_determinant_transformation_l1914_191414


namespace NUMINAMATH_GPT_no_int_solutions_p_mod_4_neg_1_l1914_191470

theorem no_int_solutions_p_mod_4_neg_1 :
  ∀ (p n : ℕ), (p % 4 = 3) → (∀ x y : ℕ, x^2 + y^2 ≠ p^n) :=
by
  intros
  sorry

end NUMINAMATH_GPT_no_int_solutions_p_mod_4_neg_1_l1914_191470


namespace NUMINAMATH_GPT_vasya_result_correct_l1914_191407

def num : ℕ := 10^1990 + (10^1989 * 6 - 1)
def denom : ℕ := 10 * (10^1989 * 6 - 1) + 4

theorem vasya_result_correct : (num / denom) = (1 / 4) := 
  sorry

end NUMINAMATH_GPT_vasya_result_correct_l1914_191407


namespace NUMINAMATH_GPT_number_of_valid_three_digit_numbers_l1914_191488

def three_digit_numbers_count : Nat :=
  let count_numbers (last_digit : Nat) (remaining_digits : List Nat) : Nat :=
    remaining_digits.length * (remaining_digits.erase last_digit).length

  let count_when_last_digit_is_0 :=
    count_numbers 0 [1, 2, 3, 4, 5, 6, 7, 8, 9]

  let count_when_last_digit_is_5 :=
    count_numbers 5 [0, 1, 2, 3, 4, 6, 7, 8, 9]

  count_when_last_digit_is_0 + count_when_last_digit_is_5

theorem number_of_valid_three_digit_numbers : three_digit_numbers_count = 136 := by
  sorry

end NUMINAMATH_GPT_number_of_valid_three_digit_numbers_l1914_191488


namespace NUMINAMATH_GPT_log_expression_value_l1914_191458

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_value :
  log10 8 + 3 * log10 4 - 2 * log10 2 + 4 * log10 25 + log10 16 = 11 := by
  sorry

end NUMINAMATH_GPT_log_expression_value_l1914_191458


namespace NUMINAMATH_GPT_valid_tickets_percentage_l1914_191410

theorem valid_tickets_percentage (cars : ℕ) (people_without_payment : ℕ) (P : ℚ) 
  (h_cars : cars = 300) (h_people_without_payment : people_without_payment = 30) 
  (h_total_valid_or_passes : (cars - people_without_payment = 270)) :
  P + (P / 5) = 90 → P = 75 :=
by
  sorry

end NUMINAMATH_GPT_valid_tickets_percentage_l1914_191410


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1914_191439

variable (a : ℕ → ℝ) (d : ℝ)
variable (a1 : ℝ) (h_d : d ≠ 0)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem common_ratio_of_geometric_sequence :
  (a 0 = a1) →
  (a 4 = a1 + 4 * d) →
  (a 16 = a1 + 16 * d) →
  (a1 + 4 * d) / a1 = (a1 + 16 * d) / (a1 + 4 * d) →
  (a1 + 16 * d) / (a1 + 4 * d) = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1914_191439


namespace NUMINAMATH_GPT_algebraic_expression_value_l1914_191452

theorem algebraic_expression_value (a b : ℤ) (h : 2 * (-3) - a + 2 * b = 0) : 2 * a - 4 * b + 1 = -11 := 
by {
  sorry
}

end NUMINAMATH_GPT_algebraic_expression_value_l1914_191452


namespace NUMINAMATH_GPT_min_fraction_sum_l1914_191473

theorem min_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  (∀ x, x = 1 / m + 2 / n → x ≥ 8) :=
  sorry

end NUMINAMATH_GPT_min_fraction_sum_l1914_191473


namespace NUMINAMATH_GPT_negate_proposition_l1914_191430

theorem negate_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l1914_191430


namespace NUMINAMATH_GPT_area_difference_l1914_191477

-- Definitions of the conditions
def length_rect := 60 -- length of the rectangular garden in feet
def width_rect := 20 -- width of the rectangular garden in feet

-- Compute the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Compute the perimeter of the rectangular garden
def perimeter_rect := 2 * (length_rect + width_rect)

-- Compute the side length of the square garden from the same perimeter
def side_square := perimeter_rect / 4

-- Compute the area of the square garden
def area_square := side_square * side_square

-- The goal is to prove the area difference
theorem area_difference : area_square - area_rect = 400 := by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_area_difference_l1914_191477


namespace NUMINAMATH_GPT_fish_added_l1914_191467

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end NUMINAMATH_GPT_fish_added_l1914_191467


namespace NUMINAMATH_GPT_find_m_l1914_191442

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (m : ℝ) (hS : ∀ n, S n = m * 2^(n-1) - 3) 
               (ha1 : a 1 = S 1) (han : ∀ n > 1, a n = S n - S (n - 1)) 
               (ratio : ∀ n > 1, a (n+1) / a n = 1/2): 
  m = 6 := 
sorry

end NUMINAMATH_GPT_find_m_l1914_191442


namespace NUMINAMATH_GPT_find_larger_number_l1914_191471

-- Define the conditions
variables (L S : ℕ)

theorem find_larger_number (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1914_191471


namespace NUMINAMATH_GPT_int_solution_l1914_191475

theorem int_solution (n : ℕ) (h1 : n ≥ 1) (h2 : n^2 ∣ 2^n + 1) : n = 1 ∨ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_int_solution_l1914_191475


namespace NUMINAMATH_GPT_hoseok_value_l1914_191461

theorem hoseok_value (x : ℕ) (h : x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end NUMINAMATH_GPT_hoseok_value_l1914_191461


namespace NUMINAMATH_GPT_spoons_in_set_l1914_191406

def number_of_spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) : ℕ :=
  let c := cost_five_spoons / 5
  let s := total_cost_set / c
  s

theorem spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) (h1 : total_cost_set = 21) (h2 : cost_five_spoons = 15) : 
  number_of_spoons_in_set total_cost_set cost_five_spoons = 7 :=
by
  sorry

end NUMINAMATH_GPT_spoons_in_set_l1914_191406


namespace NUMINAMATH_GPT_probability_interval_l1914_191424

noncomputable def Phi : ℝ → ℝ := sorry -- assuming Φ is a given function for CDF of a standard normal distribution

theorem probability_interval (h : Phi 1.98 = 0.9762) : 
  2 * Phi 1.98 - 1 = 0.9524 :=
by
  sorry

end NUMINAMATH_GPT_probability_interval_l1914_191424


namespace NUMINAMATH_GPT_reduced_price_is_16_l1914_191476

noncomputable def reduced_price_per_kg (P : ℝ) (r : ℝ) : ℝ :=
  0.9 * (P * (1 + r))

theorem reduced_price_is_16 (P r : ℝ) (h₀ : (0.9 : ℝ) * (P * (1 + r)) = 16) : 
  reduced_price_per_kg P r = 16 :=
by
  -- We have the hypothesis and we need to prove the result
  exact h₀

end NUMINAMATH_GPT_reduced_price_is_16_l1914_191476


namespace NUMINAMATH_GPT_sum_of_odd_base4_digits_of_152_and_345_l1914_191447

def base_4_digit_count (n : ℕ) : ℕ :=
    n.digits 4 |>.filter (λ x => x % 2 = 1) |>.length

theorem sum_of_odd_base4_digits_of_152_and_345 :
    base_4_digit_count 152 + base_4_digit_count 345 = 6 :=
by
    sorry

end NUMINAMATH_GPT_sum_of_odd_base4_digits_of_152_and_345_l1914_191447


namespace NUMINAMATH_GPT_average_price_per_book_l1914_191451

theorem average_price_per_book 
  (amount1 : ℝ)
  (books1 : ℕ)
  (amount2 : ℝ)
  (books2 : ℕ)
  (h1 : amount1 = 581)
  (h2 : books1 = 27)
  (h3 : amount2 = 594)
  (h4 : books2 = 20) :
  (amount1 + amount2) / (books1 + books2) = 25 := 
by
  sorry

end NUMINAMATH_GPT_average_price_per_book_l1914_191451


namespace NUMINAMATH_GPT_fraction_identity_l1914_191400

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end NUMINAMATH_GPT_fraction_identity_l1914_191400


namespace NUMINAMATH_GPT_part_a_part_b_l1914_191436

-- Given distinct primes p and q
variables (p q : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] (h : p ≠ q)

-- Prove p^q + q^p ≡ p + q (mod pq)
theorem part_a (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) :
  (p^q + q^p) % (p * q) = (p + q) % (p * q) := by
  sorry

-- Given distinct primes p and q, and neither are 2
theorem part_b (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) (hp2 : p ≠ 2) (hq2 : q ≠ 2) :
  Even (Nat.floor ((p^q + q^p) / (p * q))) := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1914_191436


namespace NUMINAMATH_GPT_cannot_obtain_100_pieces_l1914_191466

theorem cannot_obtain_100_pieces : ¬ ∃ n : ℕ, 1 + 2 * n = 100 := by
  sorry

end NUMINAMATH_GPT_cannot_obtain_100_pieces_l1914_191466


namespace NUMINAMATH_GPT_hotel_room_count_l1914_191418

theorem hotel_room_count {total_lamps lamps_per_room : ℕ} (h_total_lamps : total_lamps = 147) (h_lamps_per_room : lamps_per_room = 7) : total_lamps / lamps_per_room = 21 := by
  -- We will insert this placeholder auto-proof, as the actual arithmetic proof isn't the focus.
  sorry

end NUMINAMATH_GPT_hotel_room_count_l1914_191418


namespace NUMINAMATH_GPT_area_of_triangle_l1914_191417

theorem area_of_triangle (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (angle : ℝ) (h_side_ratio : side2 / side3 = 8 / 5)
  (h_side_opposite : side1 = 14)
  (h_angle_opposite : angle = 60) :
  (1/2 * side2 * side3 * Real.sin (angle * Real.pi / 180)) = 40 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1914_191417


namespace NUMINAMATH_GPT_intersection_A_B_l1914_191491

def A : Set ℤ := { x | (2 * x + 3) * (x - 4) < 0 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ Real.exp 1 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ (x : ℝ) ∈ B } = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1914_191491


namespace NUMINAMATH_GPT_dot_product_a_b_equals_neg5_l1914_191435

-- Defining vectors and conditions
structure vector2 := (x : ℝ) (y : ℝ)

def a : vector2 := ⟨2, 1⟩
def b (x : ℝ) : vector2 := ⟨x, -1⟩

-- Collinearity condition
def parallel (v w : vector2) : Prop :=
  v.x * w.y = v.y * w.x

-- Dot product definition
def dot_product (v w : vector2) : ℝ :=
  v.x * w.x + v.y * w.y

-- Given condition
theorem dot_product_a_b_equals_neg5 (x : ℝ) (h : parallel a ⟨a.x - x, a.y - (-1)⟩) : dot_product a (b x) = -5 :=
sorry

end NUMINAMATH_GPT_dot_product_a_b_equals_neg5_l1914_191435


namespace NUMINAMATH_GPT_minimum_distance_l1914_191487

def curve1 (x y : ℝ) : Prop := y^2 - 9 + 2*y*x - 12*x - 3*x^2 = 0
def curve2 (x y : ℝ) : Prop := y^2 + 3 - 4*x - 2*y + x^2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem minimum_distance 
  (A B : ℝ × ℝ) 
  (hA : curve1 A.1 A.2) 
  (hB : curve2 B.1 B.2) : 
  ∃ d, d = 2 * Real.sqrt 2 ∧ (∀ P Q : ℝ × ℝ, curve1 P.1 P.2 → curve2 Q.1 Q.2 → distance P.1 P.2 Q.1 Q.2 ≥ d) :=
sorry

end NUMINAMATH_GPT_minimum_distance_l1914_191487


namespace NUMINAMATH_GPT_additional_discount_percentage_l1914_191498

def initial_price : ℝ := 2000
def gift_cards : ℝ := 200
def initial_discount_rate : ℝ := 0.15
def final_price : ℝ := 1330

theorem additional_discount_percentage :
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  additional_discount_percentage = 11.33 :=
by
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  show additional_discount_percentage = 11.33
  sorry

end NUMINAMATH_GPT_additional_discount_percentage_l1914_191498


namespace NUMINAMATH_GPT_units_digit_of_17_pow_2025_l1914_191429

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2025 :
  units_digit (17 ^ 2025) = 7 :=
by sorry

end NUMINAMATH_GPT_units_digit_of_17_pow_2025_l1914_191429


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1914_191404

theorem repeating_decimal_to_fraction :
  (0.512341234123412341234 : ℝ) = (51229 / 99990 : ℝ) :=
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1914_191404


namespace NUMINAMATH_GPT_gp_sum_l1914_191485

theorem gp_sum (x : ℕ) (h : (30 + x) / (10 + x) = (60 + x) / (30 + x)) :
  x = 30 ∧ (10 + x) + (30 + x) + (60 + x) + (120 + x) = 340 :=
by {
  sorry
}

end NUMINAMATH_GPT_gp_sum_l1914_191485
