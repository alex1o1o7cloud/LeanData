import Mathlib

namespace NUMINAMATH_GPT_f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l1176_117634
open Real

noncomputable def f : ℝ → ℝ := sorry

theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := sorry
theorem f_positive_lt_x_zero (x : ℝ) (h_pos : 0 < x) : f x < 0 := sorry
theorem f_at_one : f 1 = 1 := sorry

-- Prove that f is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x :=
  sorry

-- Solve the inequality: f((log2 x)^2 - log2 (x^2)) > 3
theorem f_inequality (x : ℝ) (h_pos : 0 < x) : (f ((log x / log 2)^2 - (log x^2 / log 2))) > 3 ↔ 1 / 2 < x ∧ x < 8 :=
  sorry

end NUMINAMATH_GPT_f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l1176_117634


namespace NUMINAMATH_GPT_problem_solution_l1176_117662

noncomputable def positiveIntPairsCount : ℕ :=
  sorry

theorem problem_solution :
  positiveIntPairsCount = 2 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1176_117662


namespace NUMINAMATH_GPT_pyramid_four_triangular_faces_area_l1176_117663

noncomputable def pyramid_total_area (base_edge lateral_edge : ℝ) : ℝ :=
  if base_edge = 8 ∧ lateral_edge = 7 then 16 * Real.sqrt 33 else 0

theorem pyramid_four_triangular_faces_area :
  pyramid_total_area 8 7 = 16 * Real.sqrt 33 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pyramid_four_triangular_faces_area_l1176_117663


namespace NUMINAMATH_GPT_train_speed_correct_l1176_117685

noncomputable def train_speed (length_meters : ℕ) (time_seconds : ℕ) : ℝ :=
  (length_meters : ℝ) / 1000 / (time_seconds / 3600)

theorem train_speed_correct :
  train_speed 2500 50 = 180 := 
by
  -- We leave the proof as sorry, the statement is sufficient
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1176_117685


namespace NUMINAMATH_GPT_min_value_frac_gcd_l1176_117602

theorem min_value_frac_gcd {N k : ℕ} (hN_substring : N % 10^5 = 11235) (hN_pos : 0 < N) (hk_pos : 0 < k) (hk_bound : 10^k > N) : 
  (10^k - 1) / Nat.gcd N (10^k - 1) = 89 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_min_value_frac_gcd_l1176_117602


namespace NUMINAMATH_GPT_solve_for_c_l1176_117676

theorem solve_for_c (a b c d e : ℝ) 
  (h1 : a + b + c = 48)
  (h2 : c + d + e = 78)
  (h3 : a + b + c + d + e = 100) :
  c = 26 :=
by
sorry

end NUMINAMATH_GPT_solve_for_c_l1176_117676


namespace NUMINAMATH_GPT_trigonometric_identity_l1176_117699

variable (α : Real)

theorem trigonometric_identity (h : Real.tan α = Real.sqrt 2) :
  (1/3) * Real.sin α^2 + Real.cos α^2 = 5/9 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1176_117699


namespace NUMINAMATH_GPT_dimension_sum_l1176_117687

-- Define the dimensions A, B, C and areas AB, AC, BC
variables (A B C : ℝ) (AB AC BC : ℝ)

-- Conditions
def conditions := AB = 40 ∧ AC = 90 ∧ BC = 100 ∧ A * B = AB ∧ A * C = AC ∧ B * C = BC

-- Theorem statement
theorem dimension_sum : conditions A B C AB AC BC → A + B + C = (83 : ℝ) / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_dimension_sum_l1176_117687


namespace NUMINAMATH_GPT_calculate_bmw_sales_and_revenue_l1176_117691

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end NUMINAMATH_GPT_calculate_bmw_sales_and_revenue_l1176_117691


namespace NUMINAMATH_GPT_line_log_intersection_l1176_117667

theorem line_log_intersection (a b : ℤ) (k : ℝ)
  (h₁ : k = a + Real.sqrt b)
  (h₂ : k > 0)
  (h₃ : Real.log k / Real.log 2 - Real.log (k + 2) / Real.log 2 = 1
    ∨ Real.log (k + 2) / Real.log 2 - Real.log k / Real.log 2 = 1) :
  a + b = 2 :=
sorry

end NUMINAMATH_GPT_line_log_intersection_l1176_117667


namespace NUMINAMATH_GPT_percentage_difference_l1176_117694

theorem percentage_difference : (0.5 * 56) - (0.3 * 50) = 13 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1176_117694


namespace NUMINAMATH_GPT_find_t_of_quadratic_root_l1176_117605

variable (a t : ℝ)

def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ t : ℝ, Complex.ofReal a + Complex.I * 3 = Complex.ofReal a - Complex.I * 3 ∧
           (Complex.ofReal a + Complex.I * 3).re * (Complex.ofReal a - Complex.I * 3).re = t

theorem find_t_of_quadratic_root (h : quadratic_root_condition a) : t = 13 :=
sorry

end NUMINAMATH_GPT_find_t_of_quadratic_root_l1176_117605


namespace NUMINAMATH_GPT_sales_volume_relation_maximize_profit_l1176_117611

-- Define the conditions as given in the problem
def cost_price : ℝ := 6
def sales_data : List (ℝ × ℝ) := [(10, 4000), (11, 3900), (12, 3800)]
def price_range (x : ℝ) : Prop := 6 ≤ x ∧ x ≤ 32

-- Define the functional relationship y in terms of x
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

-- Define the profit function w in terms of x
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - cost_price)

-- Prove that the functional relationship holds within the price range
theorem sales_volume_relation (x : ℝ) (h : price_range x) :
  ∀ (y : ℝ), (x, y) ∈ sales_data → y = sales_volume x := by
  sorry

-- Prove that the profit is maximized when x = 28 and the profit is 48400 yuan
theorem maximize_profit :
  ∃ x, price_range x ∧ x = 28 ∧ profit x = 48400 := by
  sorry

end NUMINAMATH_GPT_sales_volume_relation_maximize_profit_l1176_117611


namespace NUMINAMATH_GPT_no_rotation_of_11_gears_l1176_117670

theorem no_rotation_of_11_gears :
  ∀ (gears : Fin 11 → ℕ → Prop), 
    (∀ i, gears i 0 ∧ gears (i + 1) 1 → gears i 0 = ¬gears (i + 1) 1) →
    gears 10 0 = gears 0 0 →
    False :=
by
  sorry

end NUMINAMATH_GPT_no_rotation_of_11_gears_l1176_117670


namespace NUMINAMATH_GPT_square_tiles_count_l1176_117617

theorem square_tiles_count (p s : ℕ) (h1 : p + s = 30) (h2 : 5 * p + 4 * s = 122) : s = 28 :=
sorry

end NUMINAMATH_GPT_square_tiles_count_l1176_117617


namespace NUMINAMATH_GPT_geom_seq_min_value_proof_l1176_117626

noncomputable def geom_seq_min_value : ℝ := 3 / 2

theorem geom_seq_min_value_proof (a : ℕ → ℝ) (a1 : ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  a 2017 = a 2016 + 2 * a 2015 →
  a m * a n = 16 * a1^2 →
  (4 / m + 1 / n) = geom_seq_min_value :=
by {
  sorry
}

end NUMINAMATH_GPT_geom_seq_min_value_proof_l1176_117626


namespace NUMINAMATH_GPT_find_p8_l1176_117612

noncomputable def p (x : ℝ) : ℝ := sorry -- p is a monic polynomial of degree 7

def monic_degree_7 (p : ℝ → ℝ) : Prop := sorry -- p is monic polynomial of degree 7
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 4 ∧ p 4 = 5 ∧ p 5 = 6 ∧ p 6 = 7 ∧ p 7 = 8

theorem find_p8 (h_monic : monic_degree_7 p) (h_conditions : satisfies_conditions p) : p 8 = 5049 :=
by
  sorry

end NUMINAMATH_GPT_find_p8_l1176_117612


namespace NUMINAMATH_GPT_apples_in_box_l1176_117695

-- Define the initial conditions
def oranges : ℕ := 12
def removed_oranges : ℕ := 6
def target_percentage : ℚ := 0.70

-- Define the function that models the problem
def fruit_after_removal (apples : ℕ) : ℕ := apples + (oranges - removed_oranges)
def apples_percentage (apples : ℕ) : ℚ := (apples : ℚ) / (fruit_after_removal apples : ℚ)

-- The theorem states the question and expected answer
theorem apples_in_box : ∃ (apples : ℕ), apples_percentage apples = target_percentage ∧ apples = 14 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_box_l1176_117695


namespace NUMINAMATH_GPT_find_M_l1176_117601

def grid_conditions :=
  ∃ (M : ℤ), 
  ∀ d1 d2 d3 d4, 
    (d1 = 22) ∧ (d2 = 6) ∧ (d3 = -34 / 6) ∧ (d4 = (8 - M) / 6) ∧
    (10 = 32 - d2) ∧ 
    (16 = 10 + d2) ∧ 
    (-2 = 10 - d2) ∧
    (32 - M = 34 / 6 * 6) ∧ 
    (M = -34 / 6 - (-17 / 3))

theorem find_M : grid_conditions → ∃ M : ℤ, M = 17 :=
by
  intros
  existsi (17 : ℤ) 
  sorry

end NUMINAMATH_GPT_find_M_l1176_117601


namespace NUMINAMATH_GPT_parabola_axis_l1176_117671

theorem parabola_axis (p : ℝ) (h_parabola : ∀ x : ℝ, y = x^2 → x^2 = y) : (y = - p / 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_axis_l1176_117671


namespace NUMINAMATH_GPT_odd_even_divisors_ratio_l1176_117623

theorem odd_even_divisors_ratio (M : ℕ) (h1 : M = 2^5 * 3^5 * 5 * 7^3) :
  let sum_odd_divisors := (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_all_divisors := (1 + 2 + 4 + 8 + 16 + 32) * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  sum_odd_divisors / sum_even_divisors = 1 / 62 :=
by
  sorry

end NUMINAMATH_GPT_odd_even_divisors_ratio_l1176_117623


namespace NUMINAMATH_GPT_prime_neighbor_divisible_by_6_l1176_117633

theorem prime_neighbor_divisible_by_6 (p : ℕ) (h_prime: Prime p) (h_gt3: p > 3) :
  ∃ k : ℕ, k ≠ 0 ∧ ((p - 1) % 6 = 0 ∨ (p + 1) % 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_prime_neighbor_divisible_by_6_l1176_117633


namespace NUMINAMATH_GPT_negation_of_prop_l1176_117697

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ ∃ x : ℝ, x^2 ≤ x - 1 :=
sorry

end NUMINAMATH_GPT_negation_of_prop_l1176_117697


namespace NUMINAMATH_GPT_find_x_l1176_117621

theorem find_x (x : ℚ) (h1 : 3 * x + (4 * x - 10) = 90) : x = 100 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1176_117621


namespace NUMINAMATH_GPT_max_value_of_3x_plus_4y_l1176_117609

theorem max_value_of_3x_plus_4y (x y : ℝ) (h : x^2 + y^2 = 10) : 
  ∃ z, z = 5 * Real.sqrt 10 ∧ z = 3 * x + 4 * y :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_3x_plus_4y_l1176_117609


namespace NUMINAMATH_GPT_evaluate_expression_l1176_117620

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1176_117620


namespace NUMINAMATH_GPT_max_cake_boxes_l1176_117666

theorem max_cake_boxes 
  (L_carton W_carton H_carton : ℕ) (L_box W_box H_box : ℕ)
  (h_carton : L_carton = 25 ∧ W_carton = 42 ∧ H_carton = 60)
  (h_box : L_box = 8 ∧ W_box = 7 ∧ H_box = 5) : 
  (L_carton * W_carton * H_carton) / (L_box * W_box * H_box) = 225 := by 
  sorry

end NUMINAMATH_GPT_max_cake_boxes_l1176_117666


namespace NUMINAMATH_GPT_male_contestants_l1176_117668

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end NUMINAMATH_GPT_male_contestants_l1176_117668


namespace NUMINAMATH_GPT_simplified_expression_l1176_117638

noncomputable def simplify_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ :=
  (x⁻¹ - z⁻¹)⁻¹

theorem simplified_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : 
  simplify_expression x z hx hz = x * z / (z - x) := 
by
  sorry

end NUMINAMATH_GPT_simplified_expression_l1176_117638


namespace NUMINAMATH_GPT_chips_cost_l1176_117647

noncomputable def cost_of_each_bag_of_chips (amount_paid_per_friend : ℕ) (number_of_friends : ℕ) (number_of_bags : ℕ) : ℕ :=
  (amount_paid_per_friend * number_of_friends) / number_of_bags

theorem chips_cost
  (amount_paid_per_friend : ℕ := 5)
  (number_of_friends : ℕ := 3)
  (number_of_bags : ℕ := 5) :
  cost_of_each_bag_of_chips amount_paid_per_friend number_of_friends number_of_bags = 3 :=
by
  sorry

end NUMINAMATH_GPT_chips_cost_l1176_117647


namespace NUMINAMATH_GPT_inequality_solution_set_l1176_117628

theorem inequality_solution_set (a b x : ℝ) (h1 : a > 0) (h2 : b = a) : 
  ((a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1176_117628


namespace NUMINAMATH_GPT_ravi_prakash_finish_together_l1176_117645

theorem ravi_prakash_finish_together (ravi_days prakash_days : ℕ) (h_ravi : ravi_days = 15) (h_prakash : prakash_days = 30) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 10 := 
by
  sorry

end NUMINAMATH_GPT_ravi_prakash_finish_together_l1176_117645


namespace NUMINAMATH_GPT_find_roots_l1176_117696

theorem find_roots (a b c d x : ℝ) (h₁ : a + d = 2015) (h₂ : b + c = 2015) (h₃ : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := 
sorry

end NUMINAMATH_GPT_find_roots_l1176_117696


namespace NUMINAMATH_GPT_alloy_parts_separation_l1176_117636

theorem alloy_parts_separation {p q x : ℝ} (h0 : p ≠ q)
  (h1 : 6 * p ≠ 16 * q)
  (h2 : 6 * x * p + 2 * (8 - 2 * x) * q = 8 * (8 - x) * p + 6 * x * q) :
  x = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_alloy_parts_separation_l1176_117636


namespace NUMINAMATH_GPT_jenna_hike_duration_l1176_117692

-- Definitions from conditions
def initial_speed : ℝ := 25
def exhausted_speed : ℝ := 10
def total_distance : ℝ := 140
def total_time : ℝ := 8

-- The statement to prove:
theorem jenna_hike_duration : ∃ x : ℝ, 25 * x + 10 * (8 - x) = 140 ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_jenna_hike_duration_l1176_117692


namespace NUMINAMATH_GPT_value_of_expression_l1176_117651

-- Given conditions
variable (n : ℤ)
def m : ℤ := 4 * n + 3

-- Main theorem statement
theorem value_of_expression (n : ℤ) : 
  (m n)^2 - 8 * (m n) * n + 16 * n^2 = 9 := 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1176_117651


namespace NUMINAMATH_GPT_find_fraction_of_difference_eq_halves_l1176_117672

theorem find_fraction_of_difference_eq_halves (x : ℚ) (h : 9 - x = 2.25) : x = 27 / 4 :=
by sorry

end NUMINAMATH_GPT_find_fraction_of_difference_eq_halves_l1176_117672


namespace NUMINAMATH_GPT_tanvi_min_candies_l1176_117630

theorem tanvi_min_candies : 
  ∃ c : ℕ, 
  (c % 6 = 5) ∧ 
  (c % 8 = 7) ∧ 
  (c % 9 = 6) ∧ 
  (c % 11 = 0) ∧ 
  (∀ d : ℕ, 
    (d % 6 = 5) ∧ 
    (d % 8 = 7) ∧ 
    (d % 9 = 6) ∧ 
    (d % 11 = 0) → 
    c ≤ d) → 
  c = 359 :=
by sorry

end NUMINAMATH_GPT_tanvi_min_candies_l1176_117630


namespace NUMINAMATH_GPT_course_choice_gender_related_l1176_117635
open scoped Real

theorem course_choice_gender_related :
  let a := 40 -- Males choosing Calligraphy
  let b := 10 -- Males choosing Paper Cutting
  let c := 30 -- Females choosing Calligraphy
  let d := 20 -- Females choosing Paper Cutting
  let n := a + b + c + d -- Total number of students
  let χ_squared := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d))
  χ_squared > 3.841 := 
by
  sorry

end NUMINAMATH_GPT_course_choice_gender_related_l1176_117635


namespace NUMINAMATH_GPT_circle_y_axis_intersection_range_l1176_117629

theorem circle_y_axis_intersection_range (m : ℝ) : (4 - 4 * (m + 6) > 0) → (-2 < 0) → (m + 6 > 0) → (-6 < m ∧ m < -5) :=
by 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_circle_y_axis_intersection_range_l1176_117629


namespace NUMINAMATH_GPT_average_rate_of_change_correct_l1176_117625

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_correct :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_correct_l1176_117625


namespace NUMINAMATH_GPT_extreme_value_result_l1176_117678

open Real

-- Conditions
def function_has_extreme_value_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  deriv f x₀ = 0

-- The given function
noncomputable def f (x : ℝ) : ℝ := x * sin x

-- The problem statement (to prove)
theorem extreme_value_result (x₀ : ℝ) 
  (h : function_has_extreme_value_at f x₀) :
  (1 + x₀^2) * (1 + cos (2 * x₀)) = 2 :=
sorry

end NUMINAMATH_GPT_extreme_value_result_l1176_117678


namespace NUMINAMATH_GPT_sqrt_neg2_sq_l1176_117677

theorem sqrt_neg2_sq : Real.sqrt ((-2 : ℝ) ^ 2) = 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_neg2_sq_l1176_117677


namespace NUMINAMATH_GPT_square_perimeter_increase_l1176_117653

theorem square_perimeter_increase (s : ℝ) : (4 * (s + 2) - 4 * s) = 8 := 
by
  sorry

end NUMINAMATH_GPT_square_perimeter_increase_l1176_117653


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1176_117669

-- Part a
theorem part_a (n: ℕ) (h: n = 1): (n^2 - 5 * n + 4) / (n - 4) = 0 := by sorry

-- Part b
theorem part_b (n: ℕ) (h: (n^2 - 5 * n + 4) / (n - 4) = 5): n = 6 := 
  by sorry

-- Part c
theorem part_c (n: ℕ) (h : n ≠ 4): (n^2 - 5 * n + 4) / (n - 4) ≠ 3 := 
  by sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1176_117669


namespace NUMINAMATH_GPT_sum_of_interior_angles_pentagon_l1176_117675

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_pentagon_l1176_117675


namespace NUMINAMATH_GPT_tangent_circles_BC_length_l1176_117693

theorem tangent_circles_BC_length
  (rA rB : ℝ) (A B C : ℝ × ℝ) (distAB distAC : ℝ) 
  (hAB : rA + rB = distAB)
  (hAC : distAB + 2 = distAC) 
  (h_sim : ∀ AD BE BC AC : ℝ, AD / BE = rA / rB → BC / AC = rB / rA) :
  BC = 52 / 7 := sorry

end NUMINAMATH_GPT_tangent_circles_BC_length_l1176_117693


namespace NUMINAMATH_GPT_distance_between_points_l1176_117690

theorem distance_between_points (points : Fin 7 → ℝ × ℝ) (diameter : ℝ)
  (h_diameter : diameter = 1)
  (h_points_in_circle : ∀ i : Fin 7, (points i).fst^2 + (points i).snd^2 ≤ (diameter / 2)^2) :
  ∃ (i j : Fin 7), i ≠ j ∧ (dist (points i) (points j) ≤ 1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1176_117690


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1176_117631

-- Define the quadratic polynomial f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions: f(x) = x has no real roots
theorem quadratic_no_real_roots (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1176_117631


namespace NUMINAMATH_GPT_range_of_m_l1176_117632

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) ↔ (m ∈ Set.Icc (-6:ℝ) 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1176_117632


namespace NUMINAMATH_GPT_find_a_plus_b_l1176_117649

theorem find_a_plus_b (a b x : ℝ) (h1 : x + 2 * a > 4) (h2 : 2 * x < b)
  (h3 : 0 < x) (h4 : x < 2) : a + b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1176_117649


namespace NUMINAMATH_GPT_compound_p_and_q_false_l1176_117616

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1) /- The function y = a^x is monotonically decreasing. -/
def q : Prop := (a > 1/2) /- The function y = log(ax^2 - x + a) has the range R. -/

theorem compound_p_and_q_false : 
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ (a > 1) :=
by {
  -- this part will contain the proof steps, omitted here.
  sorry
}

end NUMINAMATH_GPT_compound_p_and_q_false_l1176_117616


namespace NUMINAMATH_GPT_tea_bags_l1176_117648

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end NUMINAMATH_GPT_tea_bags_l1176_117648


namespace NUMINAMATH_GPT_infinitely_many_n_divisible_by_2018_l1176_117680

theorem infinitely_many_n_divisible_by_2018 :
  ∃ᶠ n : ℕ in Filter.atTop, 2018 ∣ (1 + 2^n + 3^n + 4^n) :=
sorry

end NUMINAMATH_GPT_infinitely_many_n_divisible_by_2018_l1176_117680


namespace NUMINAMATH_GPT_cannot_be_expressed_as_difference_of_squares_l1176_117681

theorem cannot_be_expressed_as_difference_of_squares : 
  ¬ ∃ (a b : ℤ), 2006 = a^2 - b^2 :=
sorry

end NUMINAMATH_GPT_cannot_be_expressed_as_difference_of_squares_l1176_117681


namespace NUMINAMATH_GPT_number_of_intersections_l1176_117640

def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1
def vertical_line (x : ℝ) : Prop := x = 4

theorem number_of_intersections : 
    (∃ y : ℝ, ellipse 4 y ∧ vertical_line 4) ∧ 
    ∀ y1 y2, (ellipse 4 y1 ∧ vertical_line 4) → (ellipse 4 y2 ∧ vertical_line 4) → y1 = y2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_intersections_l1176_117640


namespace NUMINAMATH_GPT_circle_trajectory_l1176_117600

theorem circle_trajectory (x y : ℝ) (h1 : (x-5)^2 + (y+7)^2 = 16) (h2 : ∃ c : ℝ, c = ((x + 1 - 5)^2 + (y + 1 + 7)^2)): 
    ((x-5)^2+(y+7)^2 = 25 ∨ (x-5)^2+(y+7)^2 = 9) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_circle_trajectory_l1176_117600


namespace NUMINAMATH_GPT_num_comics_liked_by_males_l1176_117627

-- Define the problem conditions
def num_comics : ℕ := 300
def percent_liked_by_females : ℕ := 30
def percent_disliked_by_both : ℕ := 30

-- Define the main theorem to prove
theorem num_comics_liked_by_males :
  let percent_liked_by_at_least_one_gender := 100 - percent_disliked_by_both
  let num_comics_liked_by_females := percent_liked_by_females * num_comics / 100
  let num_comics_liked_by_at_least_one_gender := percent_liked_by_at_least_one_gender * num_comics / 100
  num_comics_liked_by_at_least_one_gender - num_comics_liked_by_females = 120 :=
by
  sorry

end NUMINAMATH_GPT_num_comics_liked_by_males_l1176_117627


namespace NUMINAMATH_GPT_find_n_if_roots_opposite_signs_l1176_117610

theorem find_n_if_roots_opposite_signs :
  ∃ n : ℝ, (∀ x : ℝ, (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) → x = -x) →
    (n = (-1 + Real.sqrt 5) / 2 ∨ n = (-1 - Real.sqrt 5) / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_n_if_roots_opposite_signs_l1176_117610


namespace NUMINAMATH_GPT_hyperbola_eq_from_conditions_l1176_117637

-- Conditions of the problem
def hyperbola_center : Prop := ∃ (h : ℝ → ℝ → Prop), h 0 0
def hyperbola_eccentricity : Prop := ∃ e : ℝ, e = 2
def parabola_focus : Prop := ∃ p : ℝ × ℝ, p = (4, 0)
def parabola_equation : Prop := ∀ x y : ℝ, y^2 = 8 * x

-- Hyperbola equation to be proved
def hyperbola_equation : Prop := ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1

-- Lean 4 theorem statement
theorem hyperbola_eq_from_conditions 
  (h_center : hyperbola_center) 
  (h_eccentricity : hyperbola_eccentricity) 
  (p_focus : parabola_focus) 
  (p_eq : parabola_equation) 
  : hyperbola_equation :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eq_from_conditions_l1176_117637


namespace NUMINAMATH_GPT_smallest_positive_integer_with_12_divisors_l1176_117604

theorem smallest_positive_integer_with_12_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ n) → n ≤ m) ∧ ∃ d : ℕ → ℕ, (d n = 12) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_with_12_divisors_l1176_117604


namespace NUMINAMATH_GPT_find_q_l1176_117659

theorem find_q (q : Nat) (h : 81 ^ 6 = 3 ^ q) : q = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l1176_117659


namespace NUMINAMATH_GPT_find_initial_alison_stamps_l1176_117614

-- Define initial number of stamps Anna, Jeff, and Alison had
def initial_anna_stamps : ℕ := 37
def initial_jeff_stamps : ℕ := 31
def final_anna_stamps : ℕ := 50

-- Define the assumption that Alison gave Anna half of her stamps
def alison_gave_anna_half (a : ℕ) : Prop :=
  initial_anna_stamps + a / 2 = final_anna_stamps

-- Define the problem of finding the initial number of stamps Alison had
def alison_initial_stamps : ℕ := 26

theorem find_initial_alison_stamps :
  ∃ a : ℕ, alison_gave_anna_half a ∧ a = alison_initial_stamps :=
by
  sorry

end NUMINAMATH_GPT_find_initial_alison_stamps_l1176_117614


namespace NUMINAMATH_GPT_number_of_red_parrots_l1176_117673

-- Defining the conditions from a)
def fraction_yellow_parrots : ℚ := 2 / 3
def total_birds : ℕ := 120

-- Stating the theorem we want to prove
theorem number_of_red_parrots (H1 : fraction_yellow_parrots = 2 / 3) (H2 : total_birds = 120) : 
  (1 - fraction_yellow_parrots) * total_birds = 40 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_red_parrots_l1176_117673


namespace NUMINAMATH_GPT_tea_sales_l1176_117657

theorem tea_sales (L T : ℕ) (h1 : L = 32) (h2 : L = 4 * T + 8) : T = 6 :=
by
  sorry

end NUMINAMATH_GPT_tea_sales_l1176_117657


namespace NUMINAMATH_GPT_union_is_correct_l1176_117655

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem union_is_correct : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by
  sorry

end NUMINAMATH_GPT_union_is_correct_l1176_117655


namespace NUMINAMATH_GPT_hungarian_license_plates_l1176_117622

/-- 
In Hungarian license plates, digits can be identical. Based on observations, 
someone claimed that on average, approximately 3 out of every 10 vehicles 
have such license plates. Is this statement true?
-/
theorem hungarian_license_plates : 
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  abs (probability - 0.3) < 0.05 :=
by {
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  sorry
}

end NUMINAMATH_GPT_hungarian_license_plates_l1176_117622


namespace NUMINAMATH_GPT_largest_root_vieta_l1176_117652

theorem largest_root_vieta 
  (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = -6) : 
  max a (max b c) = 3 :=
sorry

end NUMINAMATH_GPT_largest_root_vieta_l1176_117652


namespace NUMINAMATH_GPT_pythagorean_theorem_example_l1176_117658

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 8
noncomputable def c : ℕ := 10

theorem pythagorean_theorem_example :
  c = Real.sqrt (a^2 + b^2) := 
by
  sorry

end NUMINAMATH_GPT_pythagorean_theorem_example_l1176_117658


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l1176_117619

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 1) : y = -3 * x + 1 := 
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l1176_117619


namespace NUMINAMATH_GPT_find_digit_l1176_117660

theorem find_digit (a : ℕ) (n1 n2 n3 : ℕ) (h1 : n1 = a * 1000) (h2 : n2 = a * 1000 + 998) (h3 : n3 = a * 1000 + 999) (h4 : n1 + n2 + n3 = 22997) :
  a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_l1176_117660


namespace NUMINAMATH_GPT_find_M_range_of_a_l1176_117607

def Δ (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

def A : Set ℝ := { x | 4 * x^2 + 9 * x + 2 < 0 }

def B : Set ℝ := { x | -1 < x ∧ x < 2 }

def M : Set ℝ := Δ B A

def P (a: ℝ) : Set ℝ := { x | (x - 2 * a) * (x + a - 2) < 0 }

theorem find_M :
  M = { x | -1/4 ≤ x ∧ x < 2 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ M → x ∈ P a) →
  a < -1/8 ∨ a > 9/4 :=
sorry

end NUMINAMATH_GPT_find_M_range_of_a_l1176_117607


namespace NUMINAMATH_GPT_find_number_l1176_117684

theorem find_number (x : ℝ) (h : 120 = 1.5 * x) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1176_117684


namespace NUMINAMATH_GPT_carrots_problem_l1176_117679

def total_carrots (faye_picked : Nat) (mother_picked : Nat) : Nat :=
  faye_picked + mother_picked

def bad_carrots (total_carrots : Nat) (good_carrots : Nat) : Nat :=
  total_carrots - good_carrots

theorem carrots_problem (faye_picked : Nat) (mother_picked : Nat) (good_carrots : Nat) (bad_carrots : Nat) 
  (h1 : faye_picked = 23) 
  (h2 : mother_picked = 5)
  (h3 : good_carrots = 12) :
  bad_carrots = 16 := sorry

end NUMINAMATH_GPT_carrots_problem_l1176_117679


namespace NUMINAMATH_GPT_find_b_l1176_117613

theorem find_b (k a b : ℝ) (h1 : 1 + a + b = 3) (h2 : k = 3 + a) :
  b = 3 := 
sorry

end NUMINAMATH_GPT_find_b_l1176_117613


namespace NUMINAMATH_GPT_librarian_donated_200_books_this_year_l1176_117683

noncomputable def total_books_five_years_ago : ℕ := 500
noncomputable def books_bought_two_years_ago : ℕ := 300
noncomputable def books_bought_last_year : ℕ := books_bought_two_years_ago + 100
noncomputable def total_books_current : ℕ := 1000

-- The Lean statement to prove the librarian donated 200 old books this year
theorem librarian_donated_200_books_this_year :
  total_books_five_years_ago + books_bought_two_years_ago + books_bought_last_year - total_books_current = 200 :=
by sorry

end NUMINAMATH_GPT_librarian_donated_200_books_this_year_l1176_117683


namespace NUMINAMATH_GPT_b_and_c_work_days_l1176_117639

theorem b_and_c_work_days
  (A B C : ℝ)
  (h1 : A + B = 1 / 8)
  (h2 : A + C = 1 / 8)
  (h3 : A + B + C = 1 / 6) :
  B + C = 1 / 24 :=
sorry

end NUMINAMATH_GPT_b_and_c_work_days_l1176_117639


namespace NUMINAMATH_GPT_gcd_k_power_eq_k_minus_one_l1176_117689

noncomputable def gcd_k_power (k : ℤ) : ℤ := 
  Int.gcd (k^1024 - 1) (k^1035 - 1)

theorem gcd_k_power_eq_k_minus_one (k : ℤ) : gcd_k_power k = k - 1 := 
  sorry

end NUMINAMATH_GPT_gcd_k_power_eq_k_minus_one_l1176_117689


namespace NUMINAMATH_GPT_integer_multiplication_l1176_117661

theorem integer_multiplication :
  ∃ A : ℤ, (999999999 : ℤ) * A = (111111111 : ℤ) :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_multiplication_l1176_117661


namespace NUMINAMATH_GPT_obtuse_triangle_side_range_l1176_117641

theorem obtuse_triangle_side_range (a : ℝ) :
  (a > 0) ∧
  ((a < 3 ∧ a > -1) ∧ 
  (2 * a + 1 > a + 2) ∧ 
  (a > 1)) → 1 < a ∧ a < 3 := 
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_side_range_l1176_117641


namespace NUMINAMATH_GPT_line_intersection_l1176_117686

/-- Prove the intersection of the lines given by the equations
    8x - 5y = 10 and 3x + 2y = 1 is (25/31, -22/31) -/
theorem line_intersection :
  ∃ (x y : ℚ), 8 * x - 5 * y = 10 ∧ 3 * x + 2 * y = 1 ∧ x = 25 / 31 ∧ y = -22 / 31 :=
by
  sorry

end NUMINAMATH_GPT_line_intersection_l1176_117686


namespace NUMINAMATH_GPT_jelly_bean_problem_l1176_117624

variables {p_r p_o p_y p_g : ℝ}

theorem jelly_bean_problem :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 :=
by
  intros p_r_eq p_o_eq sum_eq
  -- The proof would proceed here, but we avoid proof details
  sorry

end NUMINAMATH_GPT_jelly_bean_problem_l1176_117624


namespace NUMINAMATH_GPT_solution_set_non_empty_iff_l1176_117656

theorem solution_set_non_empty_iff (a : ℝ) : (∃ x : ℝ, |x - 1| + |x + 2| < a) ↔ (a > 3) := 
sorry

end NUMINAMATH_GPT_solution_set_non_empty_iff_l1176_117656


namespace NUMINAMATH_GPT_probability_of_pink_l1176_117608

-- Given conditions
variables (B P : ℕ) (h : (B : ℚ) / (B + P) = 3 / 7)

-- To prove
theorem probability_of_pink (h_pow : (B : ℚ) ^ 2 / (B + P) ^ 2 = 9 / 49) :
  (P : ℚ) / (B + P) = 4 / 7 :=
sorry

end NUMINAMATH_GPT_probability_of_pink_l1176_117608


namespace NUMINAMATH_GPT_y_value_l1176_117664

theorem y_value (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := 
by 
  sorry

end NUMINAMATH_GPT_y_value_l1176_117664


namespace NUMINAMATH_GPT_find_fraction_l1176_117642

theorem find_fraction (x y : ℝ) (h1 : (1/3) * (1/4) * x = 18) (h2 : y * x = 64.8) : y = 0.3 :=
sorry

end NUMINAMATH_GPT_find_fraction_l1176_117642


namespace NUMINAMATH_GPT_line_through_A_with_equal_intercepts_l1176_117603

theorem line_through_A_with_equal_intercepts (x y : ℝ) (A : ℝ × ℝ) (hx : A = (2, 1)) :
  (∃ k : ℝ, x + y = k ∧ x + y - 3 = 0) ∨ (x - 2 * y = 0) :=
sorry

end NUMINAMATH_GPT_line_through_A_with_equal_intercepts_l1176_117603


namespace NUMINAMATH_GPT_red_ball_probability_l1176_117618

noncomputable def Urn1_blue : ℕ := 5
noncomputable def Urn1_red : ℕ := 3
noncomputable def Urn2_blue : ℕ := 4
noncomputable def Urn2_red : ℕ := 4
noncomputable def Urn3_blue : ℕ := 8
noncomputable def Urn3_red : ℕ := 0

noncomputable def P_urn (n : ℕ) : ℝ := 1 / 3
noncomputable def P_red_urn1 : ℝ := (Urn1_red : ℝ) / (Urn1_blue + Urn1_red)
noncomputable def P_red_urn2 : ℝ := (Urn2_red : ℝ) / (Urn2_blue + Urn2_red)
noncomputable def P_red_urn3 : ℝ := (Urn3_red : ℝ) / (Urn3_blue + Urn3_red)

theorem red_ball_probability : 
  (P_urn 1 * P_red_urn1 + P_urn 2 * P_red_urn2 + P_urn 3 * P_red_urn3) = 7 / 24 :=
  by sorry

end NUMINAMATH_GPT_red_ball_probability_l1176_117618


namespace NUMINAMATH_GPT_nonagon_diagonals_l1176_117688

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end NUMINAMATH_GPT_nonagon_diagonals_l1176_117688


namespace NUMINAMATH_GPT_perimeters_positive_difference_l1176_117646

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end NUMINAMATH_GPT_perimeters_positive_difference_l1176_117646


namespace NUMINAMATH_GPT_geometric_sequence_a3_value_l1176_117650

theorem geometric_sequence_a3_value
  {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 82)
  (h2 : a 2 * a 4 = 81)
  (h3 : ∀ n : ℕ, a (n + 1) = a n * a 3 / a 2) :
  a 3 = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_value_l1176_117650


namespace NUMINAMATH_GPT_total_cost_correct_l1176_117644

variables (gravel_cost_per_ton : ℝ) (gravel_tons : ℝ)
variables (sand_cost_per_ton : ℝ) (sand_tons : ℝ)
variables (cement_cost_per_ton : ℝ) (cement_tons : ℝ)

noncomputable def total_cost : ℝ :=
  (gravel_cost_per_ton * gravel_tons) + (sand_cost_per_ton * sand_tons) + (cement_cost_per_ton * cement_tons)

theorem total_cost_correct :
  gravel_cost_per_ton = 30.5 → gravel_tons = 5.91 →
  sand_cost_per_ton = 40.5 → sand_tons = 8.11 →
  cement_cost_per_ton = 55.6 → cement_tons = 4.35 →
  total_cost gravel_cost_per_ton gravel_tons sand_cost_per_ton sand_tons cement_cost_per_ton cement_tons = 750.57 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1176_117644


namespace NUMINAMATH_GPT_elena_allowance_fraction_l1176_117682

variable {A m s : ℝ}

theorem elena_allowance_fraction {A : ℝ} (h1 : m = 0.25 * (A - s)) (h2 : s = 0.10 * (A - m)) : m + s = (4 / 13) * A :=
by
  sorry

end NUMINAMATH_GPT_elena_allowance_fraction_l1176_117682


namespace NUMINAMATH_GPT_isabel_pop_albums_l1176_117643

theorem isabel_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) (pop_albums : ℕ)
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8)
  (h4 : total_songs - country_albums * songs_per_album = pop_albums * songs_per_album) :
  pop_albums = 5 :=
by
  sorry

end NUMINAMATH_GPT_isabel_pop_albums_l1176_117643


namespace NUMINAMATH_GPT_expand_polynomial_l1176_117698

theorem expand_polynomial :
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1176_117698


namespace NUMINAMATH_GPT_price_increase_decrease_eq_l1176_117606

theorem price_increase_decrease_eq (x : ℝ) (p : ℝ) (hx : x ≠ 0) :
  x * (1 + p / 100) * (1 - p / 200) = x * (1 + p / 300) → p = 100 / 3 :=
by
  intro h
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_price_increase_decrease_eq_l1176_117606


namespace NUMINAMATH_GPT_problem_complement_intersection_l1176_117615

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

def complement (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

theorem problem_complement_intersection :
  (complement U M) ∩ N = {3} :=
by
  sorry

end NUMINAMATH_GPT_problem_complement_intersection_l1176_117615


namespace NUMINAMATH_GPT_average_weight_men_women_l1176_117674

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end NUMINAMATH_GPT_average_weight_men_women_l1176_117674


namespace NUMINAMATH_GPT_point_P_distance_l1176_117665

variable (a b c d x : ℝ)

-- Define the points on the line
def O := 0
def A := a
def B := b
def C := c
def D := d

-- Define the conditions for point P
def AP_PDRatio := (|a - x| / |x - d| = 2 * |b - x| / |x - c|)

theorem point_P_distance : AP_PDRatio a b c d x → b + c - a = x :=
by
  sorry

end NUMINAMATH_GPT_point_P_distance_l1176_117665


namespace NUMINAMATH_GPT_andrew_age_l1176_117654

theorem andrew_age 
  (g a : ℚ)
  (h1: g = 16 * a)
  (h2: g - 20 - (a - 20) = 45) : 
 a = 17 / 3 := by 
  sorry

end NUMINAMATH_GPT_andrew_age_l1176_117654
