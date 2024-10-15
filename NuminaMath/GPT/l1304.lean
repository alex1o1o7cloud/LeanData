import Mathlib

namespace NUMINAMATH_GPT_zoey_holidays_in_a_year_l1304_130490

-- Given conditions as definitions
def holidays_per_month : ℕ := 2
def months_in_a_year : ℕ := 12

-- Definition of the total holidays in a year
def total_holidays_in_year : ℕ := holidays_per_month * months_in_a_year

-- Proof statement
theorem zoey_holidays_in_a_year : total_holidays_in_year = 24 := 
by
  sorry

end NUMINAMATH_GPT_zoey_holidays_in_a_year_l1304_130490


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1304_130494

theorem max_value_of_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), -3 * y^2 + 18 * y - 5 ≤ -3 * x^2 + 18 * x - 5 ∧ -3 * x^2 + 18 * x - 5 = 22 :=
sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1304_130494


namespace NUMINAMATH_GPT_tax_on_clothing_l1304_130497

variable (T : ℝ)
variable (c : ℝ := 0.45 * T)
variable (f : ℝ := 0.45 * T)
variable (o : ℝ := 0.10 * T)
variable (x : ℝ)
variable (t_c : ℝ := x / 100 * c)
variable (t_f : ℝ := 0)
variable (t_o : ℝ := 0.10 * o)
variable (t : ℝ := 0.0325 * T)

theorem tax_on_clothing :
  t_c + t_o = t → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_tax_on_clothing_l1304_130497


namespace NUMINAMATH_GPT_solve_system_of_equations_l1304_130462

theorem solve_system_of_equations :
  ∃ (x y z : ℝ),
    (x^2 + y^2 + 8 * x - 6 * y = -20) ∧
    (x^2 + z^2 + 8 * x + 4 * z = -10) ∧
    (y^2 + z^2 - 6 * y + 4 * z = 0) ∧
    ((x = -3 ∧ y = 1 ∧ z = 1) ∨
     (x = -3 ∧ y = 1 ∧ z = -5) ∨
     (x = -3 ∧ y = 5 ∧ z = 1) ∨
     (x = -3 ∧ y = 5 ∧ z = -5) ∨
     (x = -5 ∧ y = 1 ∧ z = 1) ∨
     (x = -5 ∧ y = 1 ∧ z = -5) ∨
     (x = -5 ∧ y = 5 ∧ z = 1) ∨
     (x = -5 ∧ y = 5 ∧ z = -5)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1304_130462


namespace NUMINAMATH_GPT_area_of_triangle_is_27_over_5_l1304_130413

def area_of_triangle_bounded_by_y_axis_and_lines : ℚ :=
  let y_intercept_1 := -2
  let y_intercept_2 := 4
  let base := y_intercept_2 - y_intercept_1
  let x_intersection : ℚ := 9 / 5   -- Calculated using the system of equations
  1 / 2 * base * x_intersection

theorem area_of_triangle_is_27_over_5 :
  area_of_triangle_bounded_by_y_axis_and_lines = 27 / 5 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_27_over_5_l1304_130413


namespace NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l1304_130411

/-- In an arithmetic sequence, the sum of the first three terms is 9 and the third term is 8. 
    Prove that the seventh term is 28. -/
theorem arithmetic_sequence_seventh_term :
  ∃ (a d : ℤ), (a + (a + d) + (a + 2 * d) = 9) ∧ (a + 2 * d = 8) ∧ (a + 6 * d = 28) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l1304_130411


namespace NUMINAMATH_GPT_parallel_sufficient_not_necessary_l1304_130492

def line := Type
def parallel (l1 l2 : line) : Prop := sorry
def in_plane (l : line) : Prop := sorry

theorem parallel_sufficient_not_necessary (a β : line) :
  (parallel a β → ∃ γ, in_plane γ ∧ parallel a γ) ∧
  ¬( (∃ γ, in_plane γ ∧ parallel a γ) → parallel a β ) :=
by sorry

end NUMINAMATH_GPT_parallel_sufficient_not_necessary_l1304_130492


namespace NUMINAMATH_GPT_contrapositive_of_real_roots_l1304_130467

theorem contrapositive_of_real_roots (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_of_real_roots_l1304_130467


namespace NUMINAMATH_GPT_cost_price_decrease_proof_l1304_130414

theorem cost_price_decrease_proof (x y : ℝ) (a : ℝ) (h1 : y - x = x * a / 100)
    (h2 : y = (1 + a / 100) * x)
    (h3 : y - 0.9 * x = (0.9 * x * a / 100) + 0.9 * x * 20 / 100) : a = 80 :=
  sorry

end NUMINAMATH_GPT_cost_price_decrease_proof_l1304_130414


namespace NUMINAMATH_GPT_range_of_k_l1304_130475

noncomputable def f (x k : ℝ) : ℝ := 2^x + 3*x - k

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x < 2 ∧ f x k = 0) ↔ 5 ≤ k ∧ k < 10 :=
by sorry

end NUMINAMATH_GPT_range_of_k_l1304_130475


namespace NUMINAMATH_GPT_range_of_a_l1304_130485

noncomputable def quadratic_inequality_solution_set (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x - 4 < 0

theorem range_of_a :
  {a : ℝ | quadratic_inequality_solution_set a} = {a | -16 < a ∧ a ≤ 0} := 
sorry

end NUMINAMATH_GPT_range_of_a_l1304_130485


namespace NUMINAMATH_GPT_coffee_merchant_mixture_price_l1304_130423

theorem coffee_merchant_mixture_price
  (c1 c2 : ℝ) (w1 w2 total_cost mixture_price : ℝ)
  (h_c1 : c1 = 9)
  (h_c2 : c2 = 12)
  (h_w1w2 : w1 = 25 ∧ w2 = 25)
  (h_total_weight : w1 + w2 = 100)
  (h_total_cost : total_cost = w1 * c1 + w2 * c2)
  (h_mixture_price : mixture_price = total_cost / (w1 + w2)) :
  mixture_price = 5.25 :=
by sorry

end NUMINAMATH_GPT_coffee_merchant_mixture_price_l1304_130423


namespace NUMINAMATH_GPT_percentage_difference_l1304_130448

variable (x y : ℝ)
variable (hxy : x = 6 * y)

theorem percentage_difference : ((x - y) / x) * 100 = 83.33 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1304_130448


namespace NUMINAMATH_GPT_vertical_asymptote_sum_l1304_130417

theorem vertical_asymptote_sum :
  ∀ x y : ℝ, (4 * x^2 + 8 * x + 3 = 0) → (4 * y^2 + 8 * y + 3 = 0) → x ≠ y → x + y = -2 :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_sum_l1304_130417


namespace NUMINAMATH_GPT_num_common_points_l1304_130408

-- Definitions of the given conditions:
def line1 (x y : ℝ) := x + 2 * y - 3 = 0
def line2 (x y : ℝ) := 4 * x - y + 1 = 0
def line3 (x y : ℝ) := 2 * x - y - 5 = 0
def line4 (x y : ℝ) := 3 * x + 4 * y - 8 = 0

-- The proof goal:
theorem num_common_points : 
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2) ∧ (line3 p.1 p.2 ∨ line4 p.1 p.2) :=
sorry

end NUMINAMATH_GPT_num_common_points_l1304_130408


namespace NUMINAMATH_GPT_workers_in_first_group_l1304_130449

-- Define the first condition: Some workers collect 48 kg of cotton in 4 days
def cotton_collected_by_W_workers_in_4_days (W : ℕ) : ℕ := 48

-- Define the second condition: 9 workers collect 72 kg of cotton in 2 days
def cotton_collected_by_9_workers_in_2_days : ℕ := 72

-- Define the rate of cotton collected per worker per day for both scenarios
def rate_per_worker_first_group (W : ℕ) : ℕ :=
cotton_collected_by_W_workers_in_4_days W / (W * 4)

def rate_per_worker_second_group : ℕ :=
cotton_collected_by_9_workers_in_2_days / (9 * 2)

-- Given the rates are the same for both groups, prove W = 3
theorem workers_in_first_group (W : ℕ) (h : rate_per_worker_first_group W = rate_per_worker_second_group) : W = 3 :=
sorry

end NUMINAMATH_GPT_workers_in_first_group_l1304_130449


namespace NUMINAMATH_GPT_slope_of_intersection_line_l1304_130451

theorem slope_of_intersection_line 
    (x y : ℝ)
    (h1 : x^2 + y^2 - 6*x + 4*y - 20 = 0)
    (h2 : x^2 + y^2 - 2*x - 6*y + 10 = 0) :
    ∃ m : ℝ, m = 0.4 := 
sorry

end NUMINAMATH_GPT_slope_of_intersection_line_l1304_130451


namespace NUMINAMATH_GPT_difference_of_squares_l1304_130498

theorem difference_of_squares (n : ℤ) : 4 - n^2 = (2 + n) * (2 - n) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1304_130498


namespace NUMINAMATH_GPT_faye_initial_books_l1304_130468

theorem faye_initial_books (X : ℕ) (h : (X - 3) + 48 = 79) : X = 34 :=
sorry

end NUMINAMATH_GPT_faye_initial_books_l1304_130468


namespace NUMINAMATH_GPT_number_of_red_cars_l1304_130409

theorem number_of_red_cars (B R : ℕ) (h1 : R / B = 3 / 8) (h2 : B = 70) : R = 26 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_cars_l1304_130409


namespace NUMINAMATH_GPT_shortest_chord_value_of_m_l1304_130405

theorem shortest_chord_value_of_m :
  (∃ m : ℝ,
      (∀ x y : ℝ, mx + y - 2 * m - 1 = 0) ∧
      (∀ x y : ℝ, x ^ 2 + y ^ 2 - 2 * x - 4 * y = 0) ∧
      (mx + y - 2 * m - 1 = 0 → ∃ x y : ℝ, (x, y) = (2, 1))
  ) → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_value_of_m_l1304_130405


namespace NUMINAMATH_GPT_sum_infinite_series_l1304_130431

theorem sum_infinite_series :
  ∑' n : ℕ, (3 * (n+1) + 2) / ((n+1) * (n+2) * (n+4)) = 29 / 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_infinite_series_l1304_130431


namespace NUMINAMATH_GPT_positive_real_numbers_l1304_130420

theorem positive_real_numbers
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : b * c + c * a + a * b > 0)
  (h3 : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end NUMINAMATH_GPT_positive_real_numbers_l1304_130420


namespace NUMINAMATH_GPT_equilateral_triangle_area_l1304_130493

theorem equilateral_triangle_area (perimeter : ℝ) (h1 : perimeter = 120) :
  ∃ A : ℝ, A = 400 * Real.sqrt 3 ∧
    (∃ s : ℝ, s = perimeter / 3 ∧ A = (Real.sqrt 3 / 4) * (s ^ 2)) :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_l1304_130493


namespace NUMINAMATH_GPT_min_a_value_l1304_130471

theorem min_a_value 
  (a x y : ℤ) 
  (h1 : x - y^2 = a) 
  (h2 : y - x^2 = a) 
  (h3 : x ≠ y) 
  (h4 : |x| ≤ 10) : 
  a = -111 :=
sorry

end NUMINAMATH_GPT_min_a_value_l1304_130471


namespace NUMINAMATH_GPT_domain_condition_implies_m_range_range_condition_implies_m_range_l1304_130469

noncomputable def f (x m : ℝ) : ℝ := Real.log (x^2 - 2 * m * x + m + 2)

def condition1 (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 2 * m * x + m + 2 > 0)

def condition2 (m : ℝ) : Prop :=
  ∃ y : ℝ, (∀ x : ℝ, y = Real.log (x^2 - 2 * m * x + m + 2))

theorem domain_condition_implies_m_range (m : ℝ) :
  condition1 m → -1 < m ∧ m < 2 :=
sorry

theorem range_condition_implies_m_range (m : ℝ) :
  condition2 m → (m ≤ -1 ∨ m ≥ 2) :=
sorry

end NUMINAMATH_GPT_domain_condition_implies_m_range_range_condition_implies_m_range_l1304_130469


namespace NUMINAMATH_GPT_peter_walks_more_time_l1304_130455

-- Define the total distance Peter has to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def distance_walked : ℝ := 1.0

-- Define Peter's walking pace in minutes per mile
def walking_pace : ℝ := 20.0

-- Prove that Peter has to walk 30 more minutes to reach the grocery store
theorem peter_walks_more_time : walking_pace * (total_distance - distance_walked) = 30 :=
by
  sorry

end NUMINAMATH_GPT_peter_walks_more_time_l1304_130455


namespace NUMINAMATH_GPT_sequence_property_l1304_130478

theorem sequence_property (x : ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = 1 + x ^ (n + 1) + x ^ (n + 2)) (h_given : (a 2) ^ 2 = (a 1) * (a 3)) :
  ∀ n ≥ 3, (a n) ^ 2 = (a (n - 1)) * (a (n + 1)) :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_sequence_property_l1304_130478


namespace NUMINAMATH_GPT_question_inequality_l1304_130436

theorem question_inequality
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (cond : a + b ≤ 4) :
  (1 / a + 1 / b) ≥ 1 := 
sorry

end NUMINAMATH_GPT_question_inequality_l1304_130436


namespace NUMINAMATH_GPT_longest_side_similar_triangle_l1304_130496

theorem longest_side_similar_triangle (a b c : ℝ) (p : ℝ) (h₀ : a = 8) (h₁ : b = 15) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) (h₄ : p = 160) :
  ∃ x : ℝ, (8 * x) + (15 * x) + (17 * x) = p ∧ 17 * x = 68 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_similar_triangle_l1304_130496


namespace NUMINAMATH_GPT_sum_of_triangulars_iff_sum_of_squares_l1304_130443

-- Definitions of triangular numbers and sums of squares
def isTriangular (n : ℕ) : Prop := ∃ k, n = k * (k + 1) / 2
def isSumOfTwoTriangulars (m : ℕ) : Prop := ∃ x y, m = (x * (x + 1) / 2) + (y * (y + 1) / 2)
def isSumOfTwoSquares (n : ℕ) : Prop := ∃ a b, n = a * a + b * b

-- Main theorem statement
theorem sum_of_triangulars_iff_sum_of_squares (m : ℕ) (h_pos : 0 < m) : 
  isSumOfTwoTriangulars m ↔ isSumOfTwoSquares (4 * m + 1) :=
sorry

end NUMINAMATH_GPT_sum_of_triangulars_iff_sum_of_squares_l1304_130443


namespace NUMINAMATH_GPT_kevin_stone_count_l1304_130470

theorem kevin_stone_count :
  ∃ (N : ℕ), (∀ (n k : ℕ), 2007 = 9 * n + 11 * k → N = 20) := 
sorry

end NUMINAMATH_GPT_kevin_stone_count_l1304_130470


namespace NUMINAMATH_GPT_balls_in_boxes_l1304_130404

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1304_130404


namespace NUMINAMATH_GPT_intersection_is_correct_l1304_130446

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_is_correct : A ∩ B = {2, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1304_130446


namespace NUMINAMATH_GPT_joe_cut_kids_hair_l1304_130491

theorem joe_cut_kids_hair
  (time_women minutes_women count_women : ℕ)
  (time_men minutes_men count_men : ℕ)
  (time_kid minutes_kid : ℕ)
  (total_minutes: ℕ) : 
  minutes_women = 50 → 
  minutes_men = 15 →
  minutes_kid = 25 →
  count_women = 3 →
  count_men = 2 →
  total_minutes = 255 →
  (count_women * minutes_women + count_men * minutes_men + time_kid * minutes_kid) = total_minutes →
  time_kid = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof is not provided, hence stating sorry.
  sorry

end NUMINAMATH_GPT_joe_cut_kids_hair_l1304_130491


namespace NUMINAMATH_GPT_problem_one_problem_two_l1304_130481

-- Problem 1
theorem problem_one : -9 + 5 * (-6) - 18 / (-3) = -33 :=
by
  sorry

-- Problem 2
theorem problem_two : ((-3/4) - (5/8) + (9/12)) * (-24) + (-8) / (2/3) = -6 :=
by
  sorry

end NUMINAMATH_GPT_problem_one_problem_two_l1304_130481


namespace NUMINAMATH_GPT_compute_xy_l1304_130428

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end NUMINAMATH_GPT_compute_xy_l1304_130428


namespace NUMINAMATH_GPT_find_m_l1304_130425

theorem find_m (m : ℤ) (y : ℤ) : 
  (y^2 + m * y + 2) % (y - 1) = (m + 3) ∧ 
  (y^2 + m * y + 2) % (y + 1) = (3 - m) ∧
  (m + 3 = 3 - m) → m = 0 :=
sorry

end NUMINAMATH_GPT_find_m_l1304_130425


namespace NUMINAMATH_GPT_binomial_12_10_eq_66_l1304_130400

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end NUMINAMATH_GPT_binomial_12_10_eq_66_l1304_130400


namespace NUMINAMATH_GPT_convert_10203_base4_to_base10_l1304_130460

def base4_to_base10 (n : ℕ) (d₀ d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  d₄ * 4^4 + d₃ * 4^3 + d₂ * 4^2 + d₁ * 4^1 + d₀ * 4^0

theorem convert_10203_base4_to_base10 :
  base4_to_base10 10203 3 0 2 0 1 = 291 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_convert_10203_base4_to_base10_l1304_130460


namespace NUMINAMATH_GPT_proof_solution_l1304_130457

def proof_problem : Prop :=
  ∀ (s c p d : ℝ), 
  4 * s + 8 * c + p + 2 * d = 5.00 → 
  5 * s + 11 * c + p + 3 * d = 6.50 → 
  s + c + p + d = 1.50

theorem proof_solution : proof_problem :=
  sorry

end NUMINAMATH_GPT_proof_solution_l1304_130457


namespace NUMINAMATH_GPT_probability_same_color_l1304_130472

-- Definitions for the conditions
def blue_balls : Nat := 8
def yellow_balls : Nat := 5
def total_balls : Nat := blue_balls + yellow_balls

def prob_two_balls_same_color : ℚ :=
  (blue_balls/total_balls) * (blue_balls/total_balls) + (yellow_balls/total_balls) * (yellow_balls/total_balls)

-- Lean statement to be proved
theorem probability_same_color : prob_two_balls_same_color = 89 / 169 :=
by
  -- The proof is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_probability_same_color_l1304_130472


namespace NUMINAMATH_GPT_factor_quadratic_expression_l1304_130459

theorem factor_quadratic_expression (a b : ℤ) (h: 25 * -198 = -4950 ∧ a + b = -195 ∧ a * b = -4950) : a + 2 * b = -420 :=
sorry

end NUMINAMATH_GPT_factor_quadratic_expression_l1304_130459


namespace NUMINAMATH_GPT_equal_constants_l1304_130442

theorem equal_constants (a b : ℝ) :
  (∃ᶠ n in at_top, ⌊a * n + b⌋ ≥ ⌊a + b * n⌋) →
  (∃ᶠ m in at_top, ⌊a + b * m⌋ ≥ ⌊a * m + b⌋) →
  a = b :=
by
  sorry

end NUMINAMATH_GPT_equal_constants_l1304_130442


namespace NUMINAMATH_GPT_solve_derivative_equation_l1304_130463

theorem solve_derivative_equation :
  (∃ n : ℤ, ∀ x,
    x = 2 * n * Real.pi ∨
    x = 2 * n * Real.pi - 2 * Real.arctan (3 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_solve_derivative_equation_l1304_130463


namespace NUMINAMATH_GPT_angelina_speed_l1304_130464

theorem angelina_speed (v : ℝ) (h1 : 840 / v - 40 = 240 / v) :
  2 * v = 30 :=
by
  sorry

end NUMINAMATH_GPT_angelina_speed_l1304_130464


namespace NUMINAMATH_GPT_sum_of_interior_angles_l1304_130424

theorem sum_of_interior_angles (n : ℕ) 
  (h : 180 * (n - 2) = 3600) :
  180 * (n + 2 - 2) = 3960 ∧ 180 * (n - 2 - 2) = 3240 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l1304_130424


namespace NUMINAMATH_GPT_stratified_sampling_groupD_l1304_130495

-- Definitions for the conditions
def totalDistrictCount : ℕ := 38
def groupADistrictCount : ℕ := 4
def groupBDistrictCount : ℕ := 10
def groupCDistrictCount : ℕ := 16
def groupDDistrictCount : ℕ := 8
def numberOfCitiesToSelect : ℕ := 9

-- Define stratified sampling calculation with a floor function or rounding
noncomputable def numberSelectedFromGroupD : ℕ := (groupDDistrictCount * numberOfCitiesToSelect) / totalDistrictCount

-- The theorem to prove 
theorem stratified_sampling_groupD : numberSelectedFromGroupD = 2 := by
  sorry -- This is where the proof would go

end NUMINAMATH_GPT_stratified_sampling_groupD_l1304_130495


namespace NUMINAMATH_GPT_preimage_of_3_1_is_2_half_l1304_130482

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2 * p.2, p.1 - 2 * p.2)

theorem preimage_of_3_1_is_2_half :
  (∃ x y : ℝ, f (x, y) = (3, 1) ∧ (x = 2 ∧ y = 1/2)) :=
by
  sorry

end NUMINAMATH_GPT_preimage_of_3_1_is_2_half_l1304_130482


namespace NUMINAMATH_GPT_mandy_total_cost_after_discount_l1304_130461

-- Define the conditions
def packs_black_shirts : ℕ := 6
def packs_yellow_shirts : ℕ := 8
def packs_green_socks : ℕ := 5

def items_per_pack_black_shirts : ℕ := 7
def items_per_pack_yellow_shirts : ℕ := 4
def items_per_pack_green_socks : ℕ := 5

def cost_per_pack_black_shirts : ℕ := 25
def cost_per_pack_yellow_shirts : ℕ := 15
def cost_per_pack_green_socks : ℕ := 10

def discount_rate : ℚ := 0.10

-- Calculate the total number of each type of item
def total_black_shirts : ℕ := packs_black_shirts * items_per_pack_black_shirts
def total_yellow_shirts : ℕ := packs_yellow_shirts * items_per_pack_yellow_shirts
def total_green_socks : ℕ := packs_green_socks * items_per_pack_green_socks

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ :=
  (packs_black_shirts * cost_per_pack_black_shirts) +
  (packs_yellow_shirts * cost_per_pack_yellow_shirts) +
  (packs_green_socks * cost_per_pack_green_socks)

-- Calculate the total cost after discount
def discount_amount : ℚ := discount_rate * total_cost_before_discount
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount

-- Problem to prove: Total cost after discount is $288
theorem mandy_total_cost_after_discount : total_cost_after_discount = 288 := by
  sorry

end NUMINAMATH_GPT_mandy_total_cost_after_discount_l1304_130461


namespace NUMINAMATH_GPT_first_number_in_a10_l1304_130426

-- Define a function that captures the sequence of the first number in each sum 'a_n'.
def first_in_an (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1)) / 2 

-- State the theorem we want to prove
theorem first_number_in_a10 : first_in_an 10 = 91 := 
  sorry

end NUMINAMATH_GPT_first_number_in_a10_l1304_130426


namespace NUMINAMATH_GPT_sale_price_of_sarees_after_discounts_l1304_130444

theorem sale_price_of_sarees_after_discounts :
  let original_price := 400.0
  let discount_1 := 0.15
  let discount_2 := 0.08
  let discount_3 := 0.07
  let discount_4 := 0.10
  let price_after_first_discount := original_price * (1 - discount_1)
  let price_after_second_discount := price_after_first_discount * (1 - discount_2)
  let price_after_third_discount := price_after_second_discount * (1 - discount_3)
  let final_price := price_after_third_discount * (1 - discount_4)
  final_price = 261.81 := by
    -- Sorry is used to skip the proof
    sorry

end NUMINAMATH_GPT_sale_price_of_sarees_after_discounts_l1304_130444


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1304_130484

noncomputable def given_quadratic (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

noncomputable def sin_cos_eq_quadratic_roots (θ m : ℝ) : Prop := 
  let sinθ := Real.sin θ
  let cosθ := Real.cos θ
  given_quadratic sinθ m = 0 ∧ given_quadratic cosθ m = 0

theorem problem_part1 (θ : ℝ) (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ)) = (3 + 5 * Real.sqrt 3) / 4 :=
sorry

theorem problem_part2 {θ : ℝ} (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  m = Real.sqrt 3 / 4 :=
sorry

theorem problem_part3 (m : ℝ) (sinθ1 cosθ1 sinθ2 cosθ2 : ℝ) (θ1 θ2 : ℝ)
  (H1 : sinθ1 = Real.sqrt 3 / 2 ∧ cosθ1 = 1 / 2 ∧ θ1 = Real.pi / 3)
  (H2 : sinθ2 = 1 / 2 ∧ cosθ2 = Real.sqrt 3 / 2 ∧ θ2 = Real.pi / 6) : 
  ∃ θ, sin_cos_eq_quadratic_roots θ m ∧ 
       (Real.sin θ = sinθ1 ∧ Real.cos θ = cosθ1 ∨ Real.sin θ = sinθ2 ∧ Real.cos θ = cosθ2) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1304_130484


namespace NUMINAMATH_GPT_min_abs_sum_l1304_130433

-- Definitions based on given conditions for the problem
variable (p q r s : ℤ)
variable (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
variable (h : (matrix 2 2 ℤ ![(p, q), (r, s)]) ^ 2 = matrix 2 2 ℤ ![(9, 0), (0, 9)])

-- Statement of the proof problem
theorem min_abs_sum :
  |p| + |q| + |r| + |s| = 8 :=
by
  sorry

end NUMINAMATH_GPT_min_abs_sum_l1304_130433


namespace NUMINAMATH_GPT_wire_cut_problem_l1304_130407

theorem wire_cut_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq_area : (a / 4) ^ 2 = π * (b / (2 * π)) ^ 2) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_problem_l1304_130407


namespace NUMINAMATH_GPT_bus_capacities_rental_plan_l1304_130406

variable (x y : ℕ)
variable (m n : ℕ)

theorem bus_capacities :
  3 * x + 2 * y = 195 ∧ 2 * x + 4 * y = 210 → x = 45 ∧ y = 30 :=
by
  sorry

theorem rental_plan :
  7 * m + 3 * n = 20 ∧ m + n ≤ 7 ∧ 65 * m + 45 * n + 30 * (7 - m - n) = 310 →
  m = 2 ∧ n = 2 ∧ 7 - m - n = 3 :=
by
  sorry

end NUMINAMATH_GPT_bus_capacities_rental_plan_l1304_130406


namespace NUMINAMATH_GPT_intersection_eq_l1304_130402

theorem intersection_eq {A : Set ℕ} {B : Set ℕ} 
  (hA : A = {0, 1, 2, 3, 4, 5, 6}) 
  (hB : B = {x | ∃ n ∈ A, x = 2 * n}) : 
  A ∩ B = {0, 2, 4, 6} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1304_130402


namespace NUMINAMATH_GPT_solve_for_a_l1304_130434

theorem solve_for_a (a x : ℝ) (h : x = 3) (eqn : a * x - 5 = x + 1) : a = 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_solve_for_a_l1304_130434


namespace NUMINAMATH_GPT_total_cost_one_each_l1304_130487

theorem total_cost_one_each (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 6.3)
  (h2 : 4 * x + 10 * y + z = 8.4) :
  x + y + z = 2.1 :=
  sorry

end NUMINAMATH_GPT_total_cost_one_each_l1304_130487


namespace NUMINAMATH_GPT_fraction_addition_l1304_130454

theorem fraction_addition : (1 / 3) + (5 / 12) = 3 / 4 := 
sorry

end NUMINAMATH_GPT_fraction_addition_l1304_130454


namespace NUMINAMATH_GPT_compound_oxygen_atoms_l1304_130450

theorem compound_oxygen_atoms (H C O : Nat) (mw : Nat) (H_weight C_weight O_weight : Nat) 
  (h_H : H = 2)
  (h_C : C = 1)
  (h_mw : mw = 62)
  (h_H_weight : H_weight = 1)
  (h_C_weight : C_weight = 12)
  (h_O_weight : O_weight = 16)
  : O = 3 :=
by
  sorry

end NUMINAMATH_GPT_compound_oxygen_atoms_l1304_130450


namespace NUMINAMATH_GPT_sequence_2019_value_l1304_130403

theorem sequence_2019_value :
  ∃ a : ℕ → ℤ, (∀ n ≥ 4, a n = a (n-1) * a (n-3)) ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ a 2019 = -1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_2019_value_l1304_130403


namespace NUMINAMATH_GPT_common_divisors_4n_7n_l1304_130483

theorem common_divisors_4n_7n (n : ℕ) (h1 : n < 50) 
    (h2 : (Nat.gcd (4 * n + 5) (7 * n + 6) > 1)) :
    n = 7 ∨ n = 18 ∨ n = 29 ∨ n = 40 := 
  sorry

end NUMINAMATH_GPT_common_divisors_4n_7n_l1304_130483


namespace NUMINAMATH_GPT_cafe_location_l1304_130427

-- Definition of points and conditions
structure Point where
  x : ℤ
  y : ℚ

def mark : Point := { x := 1, y := 8 }
def sandy : Point := { x := -5, y := 0 }

-- The problem statement
theorem cafe_location :
  ∃ cafe : Point, cafe.x = -3 ∧ cafe.y = 8/3 := by
  sorry

end NUMINAMATH_GPT_cafe_location_l1304_130427


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1304_130499

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -1/3 < x ∧ x < 1 → -3 * x^2 + 8 * x + 1 < 0 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1304_130499


namespace NUMINAMATH_GPT_kite_height_30_sqrt_43_l1304_130419

theorem kite_height_30_sqrt_43
  (c d h : ℝ)
  (h1 : h^2 + c^2 = 170^2)
  (h2 : h^2 + d^2 = 150^2)
  (h3 : c^2 + d^2 = 160^2) :
  h = 30 * Real.sqrt 43 := by
  sorry

end NUMINAMATH_GPT_kite_height_30_sqrt_43_l1304_130419


namespace NUMINAMATH_GPT_reginald_apples_sold_l1304_130432

theorem reginald_apples_sold 
  (apple_price : ℝ) 
  (bike_cost : ℝ)
  (repair_percentage : ℝ)
  (remaining_fraction : ℝ)
  (discount_apples : ℕ)
  (free_apples : ℕ)
  (total_apples_sold : ℕ) : 
  apple_price = 1.25 → 
  bike_cost = 80 → 
  repair_percentage = 0.25 → 
  remaining_fraction = 0.2 → 
  discount_apples = 5 → 
  free_apples = 1 → 
  (∃ (E : ℝ), (125 = E ∧ total_apples_sold = 120)) → 
  total_apples_sold = 120 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end NUMINAMATH_GPT_reginald_apples_sold_l1304_130432


namespace NUMINAMATH_GPT_solve_expression_l1304_130412

noncomputable def given_expression : ℝ :=
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2 / 3) - Real.log 4 + Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4) + Nat.factorial 4 / Nat.factorial 2

theorem solve_expression : given_expression = 59.6862 :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l1304_130412


namespace NUMINAMATH_GPT_birds_total_distance_l1304_130447

-- Define the speeds of the birds
def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30

-- Define the flying time for each bird
def flying_time : ℕ := 2

-- Calculate the total distance flown by all birds
def total_distance_flown : ℕ := (eagle_speed * flying_time) +
                                 (falcon_speed * flying_time) +
                                 (pelican_speed * flying_time) +
                                 (hummingbird_speed * flying_time)

-- The goal is to prove that the total distance flown by all birds is 248 miles
theorem birds_total_distance : total_distance_flown = 248 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_birds_total_distance_l1304_130447


namespace NUMINAMATH_GPT_maximize_take_home_pay_l1304_130452

-- Define the tax system condition
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay condition
def take_home_pay (y : ℝ) : ℝ := 100 * y^2 - tax y

-- The theorem to prove the maximum take-home pay is achieved at a specific income level
theorem maximize_take_home_pay : 
  ∃ y : ℝ, take_home_pay y = 100 * 50^2 - 50^3 := sorry

end NUMINAMATH_GPT_maximize_take_home_pay_l1304_130452


namespace NUMINAMATH_GPT_tangent_parallel_coordinates_l1304_130453

theorem tangent_parallel_coordinates :
  (∃ (x1 y1 x2 y2 : ℝ), 
    (y1 = x1^3 - 2) ∧ (y2 = x2^3 - 2) ∧ 
    ((3 * x1^2 = 3) ∧ (3 * x2^2 = 3)) ∧ 
    ((x1 = 1 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -3))) :=
sorry

end NUMINAMATH_GPT_tangent_parallel_coordinates_l1304_130453


namespace NUMINAMATH_GPT_point_movement_l1304_130437

theorem point_movement (P : ℤ) (hP : P = -5) (k : ℤ) (hk : (k = 3 ∨ k = -3)) :
  P + k = -8 ∨ P + k = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_point_movement_l1304_130437


namespace NUMINAMATH_GPT_upper_left_region_l1304_130456

theorem upper_left_region (t : ℝ) : (2 - 2 * t + 4 ≤ 0) → (t ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_upper_left_region_l1304_130456


namespace NUMINAMATH_GPT_sparrows_on_fence_l1304_130445

-- Define the number of sparrows initially on the fence
def initial_sparrows : ℕ := 2

-- Define the number of sparrows that joined later
def additional_sparrows : ℕ := 4

-- Define the number of sparrows that flew away
def sparrows_flew_away : ℕ := 3

-- Define the final number of sparrows on the fence
def final_sparrows : ℕ := initial_sparrows + additional_sparrows - sparrows_flew_away

-- Prove that the final number of sparrows on the fence is 3
theorem sparrows_on_fence : final_sparrows = 3 := by
  sorry

end NUMINAMATH_GPT_sparrows_on_fence_l1304_130445


namespace NUMINAMATH_GPT_solve_for_x_l1304_130476

theorem solve_for_x (x : ℤ) : (16 : ℝ) ^ (3 * x - 5) = ((1 : ℝ) / 4) ^ (2 * x + 6) → x = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1304_130476


namespace NUMINAMATH_GPT_polygon_sides_l1304_130415

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 := sorry

end NUMINAMATH_GPT_polygon_sides_l1304_130415


namespace NUMINAMATH_GPT_actual_height_of_boy_is_236_l1304_130422

-- Define the problem conditions
def average_height (n : ℕ) (avg : ℕ) := n * avg
def incorrect_total_height := average_height 35 180
def correct_total_height := average_height 35 178
def wrong_height := 166
def height_difference := incorrect_total_height - correct_total_height

-- Proving the actual height of the boy whose height was wrongly written
theorem actual_height_of_boy_is_236 : 
  wrong_height + height_difference = 236 := sorry

end NUMINAMATH_GPT_actual_height_of_boy_is_236_l1304_130422


namespace NUMINAMATH_GPT_number_of_yellow_highlighters_l1304_130438

-- Definitions based on the given conditions
def total_highlighters : Nat := 12
def pink_highlighters : Nat := 6
def blue_highlighters : Nat := 4

-- Statement to prove the question equals the correct answer given the conditions
theorem number_of_yellow_highlighters : 
  ∃ y : Nat, y = total_highlighters - (pink_highlighters + blue_highlighters) := 
by
  -- TODO: The proof will be filled in here
  sorry

end NUMINAMATH_GPT_number_of_yellow_highlighters_l1304_130438


namespace NUMINAMATH_GPT_supplement_comp_greater_l1304_130458

theorem supplement_comp_greater {α β : ℝ} (h : α + β = 90) : 180 - α = β + 90 :=
by
  sorry

end NUMINAMATH_GPT_supplement_comp_greater_l1304_130458


namespace NUMINAMATH_GPT_fraction_of_shaded_area_l1304_130440

theorem fraction_of_shaded_area (total_length total_width : ℕ) (total_area : ℕ)
  (quarter_fraction half_fraction : ℚ)
  (h1 : total_length = 15) 
  (h2 : total_width = 20)
  (h3 : total_area = total_length * total_width)
  (h4 : quarter_fraction = 1 / 4)
  (h5 : half_fraction = 1 / 2) :
  (half_fraction * quarter_fraction * total_area) / total_area = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_shaded_area_l1304_130440


namespace NUMINAMATH_GPT_first_stack_height_l1304_130479

theorem first_stack_height (x : ℕ) (h1 : x + (x + 2) + (x - 3) + (x + 2) = 21) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_first_stack_height_l1304_130479


namespace NUMINAMATH_GPT_part2_l1304_130473

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem part2 (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f 1 x1 = f 1 x2) : x1 + x2 > 2 := by
  have f_x1 := h2
  sorry

end NUMINAMATH_GPT_part2_l1304_130473


namespace NUMINAMATH_GPT_find_triples_l1304_130410

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def power_of_p (p n : ℕ) : Prop := ∃ (k : ℕ), n = p^k

-- Given the conditions
variable (p x y : ℕ)
variable (h_prime : is_prime p)
variable (h_pos_x : x > 0)
variable (h_pos_y : y > 0)

-- The problem statement
theorem find_triples (h1 : power_of_p p (x^(p-1) + y)) (h2 : power_of_p p (x + y^(p-1))) : 
  (p = 3 ∧ x = 2 ∧ y = 5) ∨
  (p = 3 ∧ x = 5 ∧ y = 2) ∨
  (p = 2 ∧ ∃ (n i : ℕ), n > 0 ∧ i > 0 ∧ x = n ∧ y = 2^i - n ∧ 0 < n ∧ n < 2^i) := 
sorry

end NUMINAMATH_GPT_find_triples_l1304_130410


namespace NUMINAMATH_GPT_sum_ratios_l1304_130401

variable (a b d : ℕ)

def A_n (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def arithmetic_sum (a n d : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_ratios (k : ℕ) (h1 : 2 * (a + d) = 7 * k) (h2 : 4 * (a + 3 * d) = 6 * k) :
  arithmetic_sum a 7 d / arithmetic_sum a 3 d = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_ratios_l1304_130401


namespace NUMINAMATH_GPT_total_working_days_l1304_130439

variables (x a b c : ℕ)

-- Given conditions
axiom bus_morning : b + c = 6
axiom bus_afternoon : a + c = 18
axiom train_commute : a + b = 14

-- Proposition to prove
theorem total_working_days : x = a + b + c → x = 19 :=
by
  -- Placeholder for Lean's automatic proof generation
  sorry

end NUMINAMATH_GPT_total_working_days_l1304_130439


namespace NUMINAMATH_GPT_gingerbread_percentage_red_hats_l1304_130486

def total_gingerbread_men (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ) : ℕ :=
  n_red_hats + n_blue_boots - n_both

def percentage_with_red_hats (n_red_hats : ℕ) (total : ℕ) : ℕ :=
  (n_red_hats * 100) / total

theorem gingerbread_percentage_red_hats 
  (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ)
  (h_red_hats : n_red_hats = 6)
  (h_blue_boots : n_blue_boots = 9)
  (h_both : n_both = 3) : 
  percentage_with_red_hats n_red_hats (total_gingerbread_men n_red_hats n_blue_boots n_both) = 50 := by
  sorry

end NUMINAMATH_GPT_gingerbread_percentage_red_hats_l1304_130486


namespace NUMINAMATH_GPT_initial_fee_is_correct_l1304_130421

noncomputable def initial_fee (total_charge : ℝ) (charge_per_segment : ℝ) (segment_length : ℝ) (distance : ℝ) : ℝ :=
  total_charge - (⌊distance / segment_length⌋ * charge_per_segment)

theorem initial_fee_is_correct :
  initial_fee 4.5 0.25 (2/5) 3.6 = 2.25 :=
by 
  sorry

end NUMINAMATH_GPT_initial_fee_is_correct_l1304_130421


namespace NUMINAMATH_GPT_equation_represents_point_l1304_130441

theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2 * a * x + 2 * b * y + a^2 + b^2 = 0 ↔ x = -a ∧ y = -b := 
by sorry

end NUMINAMATH_GPT_equation_represents_point_l1304_130441


namespace NUMINAMATH_GPT_find_x_squared_plus_inverse_squared_l1304_130416

theorem find_x_squared_plus_inverse_squared (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + (1 / x)^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_inverse_squared_l1304_130416


namespace NUMINAMATH_GPT_unique_solution_for_a_l1304_130465

def system_has_unique_solution (a : ℝ) (x y : ℝ) : Prop :=
(x^2 + y^2 + 2 * x ≤ 1) ∧ (x - y + a = 0)

theorem unique_solution_for_a (a x y : ℝ) :
  (system_has_unique_solution 3 x y ∨ system_has_unique_solution (-1) x y)
  ∧ (((a = 3) → (x, y) = (-2, 1)) ∨ ((a = -1) → (x, y) = (0, -1))) :=
sorry

end NUMINAMATH_GPT_unique_solution_for_a_l1304_130465


namespace NUMINAMATH_GPT_trigonometric_identity_l1304_130489

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1304_130489


namespace NUMINAMATH_GPT_factorize_expression_l1304_130418

theorem factorize_expression (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l1304_130418


namespace NUMINAMATH_GPT_find_B_share_l1304_130430

theorem find_B_share (x : ℕ) (x_pos : 0 < x) (C_share_difference : 5 * x = 4 * x + 1000) (B_share_eq : 3 * x = B) : B = 3000 :=
by
  sorry

end NUMINAMATH_GPT_find_B_share_l1304_130430


namespace NUMINAMATH_GPT_apples_to_pears_l1304_130435

theorem apples_to_pears :
  (∀ (apples oranges pears : ℕ),
  12 * apples = 6 * oranges →
  3 * oranges = 5 * pears →
  24 * apples = 20 * pears) :=
by
  intros apples oranges pears h₁ h₂
  sorry

end NUMINAMATH_GPT_apples_to_pears_l1304_130435


namespace NUMINAMATH_GPT_divide_angle_into_parts_l1304_130480

-- Definitions based on the conditions
def given_angle : ℝ := 19

/-- 
Theorem: An angle of 19 degrees can be divided into 19 equal parts using a compass and a ruler,
and each part will measure 1 degree.
-/
theorem divide_angle_into_parts (angle : ℝ) (n : ℕ) (h1 : angle = given_angle) (h2 : n = 19) : angle / n = 1 :=
by
  -- Proof to be filled out
  sorry

end NUMINAMATH_GPT_divide_angle_into_parts_l1304_130480


namespace NUMINAMATH_GPT_average_age_of_team_is_23_l1304_130477

noncomputable def average_age_team (A : ℝ) : Prop :=
  let captain_age := 27
  let wicket_keeper_age := 28
  let team_size := 11
  let remaining_players := team_size - 2
  let remaining_average_age := A - 1
  11 * A = 55 + 9 * (A - 1)

theorem average_age_of_team_is_23 : average_age_team 23 := by
  sorry

end NUMINAMATH_GPT_average_age_of_team_is_23_l1304_130477


namespace NUMINAMATH_GPT_empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l1304_130488

noncomputable def A (a : ℝ) : Set ℝ := { x | a*x^2 - 3*x + 2 = 0 }

theorem empty_set_a_gt_nine_over_eight (a : ℝ) : A a = ∅ ↔ a > 9 / 8 :=
by
  sorry

theorem singleton_set_a_values (a : ℝ) : (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) :=
by
  sorry

theorem at_most_one_element_set_a_range (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) →
  (A a = ∅ ∨ ∃ x, A a = {x}) ↔ (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end NUMINAMATH_GPT_empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l1304_130488


namespace NUMINAMATH_GPT_four_digit_number_divisibility_l1304_130474

theorem four_digit_number_divisibility : ∃ x : ℕ, 
  (let n := 1000 + x * 100 + 50 + x; 
   ∃ k₁ k₂ : ℤ, (n = 36 * k₁) ∧ ((10 * 5 + x) = 4 * k₂) ∧ ((2 * x + 6) % 9 = 0)) :=
sorry

end NUMINAMATH_GPT_four_digit_number_divisibility_l1304_130474


namespace NUMINAMATH_GPT_tan_half_alpha_l1304_130429

theorem tan_half_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 24 / 25) : Real.tan (α / 2) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_half_alpha_l1304_130429


namespace NUMINAMATH_GPT_total_length_segments_l1304_130466

noncomputable def segment_length (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment : ℕ) :=
  let total_length := rect_horizontal_1 + rect_horizontal_2 + rect_vertical
  total_length - 8 + left_segment

theorem total_length_segments
  (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment total_left : ℕ)
  (h1 : rect_horizontal_1 = 10)
  (h2 : rect_horizontal_2 = 3)
  (h3 : rect_vertical = 12)
  (h4 : left_segment = 8)
  (h5 : total_left = 19)
  : segment_length rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment = total_left :=
sorry

end NUMINAMATH_GPT_total_length_segments_l1304_130466
