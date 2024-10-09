import Mathlib

namespace find_constants_l571_57120

theorem find_constants (P Q R : ℚ) 
  (h : ∀ x : ℚ, x ≠ 4 → x ≠ 2 → (5 * x + 1) / ((x - 4) * (x - 2) ^ 2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) :
  P = 21 / 4 ∧ Q = 15 ∧ R = -11 / 2 :=
by
  sorry

end find_constants_l571_57120


namespace problem_a_problem_b_problem_c_problem_d_l571_57194

theorem problem_a : 37.3 / (1 / 2) = 74.6 := by
  sorry

theorem problem_b : 0.45 - (1 / 20) = 0.4 := by
  sorry

theorem problem_c : (33 / 40) * (10 / 11) = 0.75 := by
  sorry

theorem problem_d : 0.375 + (1 / 40) = 0.4 := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l571_57194


namespace max_electronic_thermometers_l571_57115

-- Definitions
def budget : ℕ := 300
def price_mercury : ℕ := 3
def price_electronic : ℕ := 10
def total_students : ℕ := 53

-- The theorem statement
theorem max_electronic_thermometers : 
  (∃ x : ℕ, x ≤ total_students ∧ 10 * x + 3 * (total_students - x) ≤ budget ∧ 
            ∀ y : ℕ, y ≤ total_students ∧ 10 * y + 3 * (total_students - y) ≤ budget → y ≤ x) :=
sorry

end max_electronic_thermometers_l571_57115


namespace num_rectangles_in_5x5_grid_l571_57125

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l571_57125


namespace hyperbola_eccentricity_l571_57111

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymptotes : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  (c / a = 5 / 4) ∨ (c / a = 5 / 3) :=
by
  -- Proof omitted
  sorry

end hyperbola_eccentricity_l571_57111


namespace parabola_focus_l571_57110

theorem parabola_focus (x y : ℝ) :
  (∃ x, y = 4 * x^2 + 8 * x - 5) →
  (x, y) = (-1, -8.9375) :=
by
  sorry

end parabola_focus_l571_57110


namespace find_rectangle_width_l571_57107

variable (length_square : ℕ) (length_rectangle : ℕ) (width_rectangle : ℕ)

-- Given conditions
def square_side_length := 700
def rectangle_length := 400
def square_perimeter := 4 * square_side_length
def rectangle_perimeter := square_perimeter / 2
def rectangle_perimeter_eq := 2 * length_rectangle + 2 * width_rectangle

-- Statement to prove
theorem find_rectangle_width :
  (square_perimeter = 2800) →
  (rectangle_perimeter = 1400) →
  (length_rectangle = 400) →
  (rectangle_perimeter_eq = 1400) →
  (width_rectangle = 300) :=
by
  intros
  sorry

end find_rectangle_width_l571_57107


namespace graph_squares_count_l571_57189

theorem graph_squares_count :
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  non_diagonal_squares / 2 = 88 :=
by
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  have h : (non_diagonal_squares / 2 = 88) := sorry
  exact h

end graph_squares_count_l571_57189


namespace irreducible_fraction_l571_57178

-- Definition of gcd
def my_gcd (m n : Int) : Int :=
  gcd m n

-- Statement of the problem
theorem irreducible_fraction (a : Int) : my_gcd (a^3 + 2 * a) (a^4 + 3 * a^2 + 1) = 1 :=
by
  sorry

end irreducible_fraction_l571_57178


namespace find_b_plus_m_l571_57158

-- Definitions of the constants and functions based on the given conditions.
variables (m b : ℝ)

-- The first line equation passing through (5, 8).
def line1 := 8 = m * 5 + 3

-- The second line equation passing through (5, 8).
def line2 := 8 = 4 * 5 + b

-- The goal statement we need to prove.
theorem find_b_plus_m (h1 : line1 m) (h2 : line2 b) : b + m = -11 :=
sorry

end find_b_plus_m_l571_57158


namespace find_m_collinear_l571_57100

-- Definition of a point in 2D space
structure Point2D where
  x : ℤ
  y : ℤ

-- Predicate to check if three points are collinear 
def collinear_points (p1 p2 p3 : Point2D) : Prop :=
  (p3.x - p2.x) * (p2.y - p1.y) = (p2.x - p1.x) * (p3.y - p2.y)

-- Given points A, B, and C
def A : Point2D := ⟨2, 3⟩
def B (m : ℤ) : Point2D := ⟨-4, m⟩
def C : Point2D := ⟨-12, -1⟩

-- Theorem stating the value of m such that points A, B, and C are collinear
theorem find_m_collinear : ∃ (m : ℤ), collinear_points A (B m) C ∧ m = 9 / 7 := sorry

end find_m_collinear_l571_57100


namespace selling_price_per_unit_profit_per_unit_after_discount_l571_57181

-- Define the initial cost per unit
variable (a : ℝ)

-- Problem statement for part 1: Selling price per unit is 1.22a yuan
theorem selling_price_per_unit (a : ℝ) : 1.22 * a = a + 0.22 * a :=
by
  sorry

-- Problem statement for part 2: Profit per unit after 15% discount is still 0.037a yuan
theorem profit_per_unit_after_discount (a : ℝ) : 
  (1.22 * a * 0.85) - a = 0.037 * a :=
by
  sorry

end selling_price_per_unit_profit_per_unit_after_discount_l571_57181


namespace arccos_of_sqrt3_div_2_l571_57193

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l571_57193


namespace max_area_of_triangle_ABC_l571_57167

noncomputable def max_area_triangle_ABC: ℝ :=
  let QA := 3
  let QB := 4
  let QC := 5
  let BC := 6
  -- Given these conditions, prove the maximum area of triangle ABC
  19

theorem max_area_of_triangle_ABC 
  (QA QB QC BC : ℝ) 
  (h1 : QA = 3) 
  (h2 : QB = 4) 
  (h3 : QC = 5) 
  (h4 : BC = 6) 
  (h5 : QB * QB + BC * BC = QC * QC) -- The right angle condition at Q
  : max_area_triangle_ABC = 19 :=
by sorry

end max_area_of_triangle_ABC_l571_57167


namespace car_winning_probability_l571_57108

noncomputable def probability_of_winning (P_X P_Y P_Z : ℚ) : ℚ :=
  P_X + P_Y + P_Z

theorem car_winning_probability :
  let P_X := (1 : ℚ) / 6
  let P_Y := (1 : ℚ) / 10
  let P_Z := (1 : ℚ) / 8
  probability_of_winning P_X P_Y P_Z = 47 / 120 :=
by
  sorry

end car_winning_probability_l571_57108


namespace coins_left_zero_when_divided_by_9_l571_57153

noncomputable def smallestCoinCount (n: ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_left_zero_when_divided_by_9 (n : ℕ) (h : smallestCoinCount n) (h_min: ∀ m : ℕ, smallestCoinCount m → n ≤ m) :
  n % 9 = 0 :=
sorry

end coins_left_zero_when_divided_by_9_l571_57153


namespace range_of_a_l571_57136

noncomputable def range_of_a_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, |x + 1| + |x - a| ≤ 2

theorem range_of_a : ∀ a : ℝ, range_of_a_condition a → (-3 : ℝ) ≤ a ∧ a ≤ 1 :=
by
  intros a h
  sorry

end range_of_a_l571_57136


namespace larger_group_men_count_l571_57157

-- Define the conditions
def total_man_days (men : ℕ) (days : ℕ) : ℕ := men * days

-- Define the total work for 36 men in 18 days
def work_by_36_men_in_18_days : ℕ := total_man_days 36 18

-- Define the number of days the larger group takes
def days_for_larger_group : ℕ := 8

-- Problem Statement: Prove that if 36 men take 18 days to complete the work, and a larger group takes 8 days, then the larger group consists of 81 men.
theorem larger_group_men_count : 
  ∃ (M : ℕ), total_man_days M days_for_larger_group = work_by_36_men_in_18_days ∧ M = 81 := 
by
  -- Here would go the proof steps
  sorry

end larger_group_men_count_l571_57157


namespace cricket_problem_solved_l571_57147

noncomputable def cricket_problem : Prop :=
  let run_rate_10 := 3.2
  let target := 252
  let required_rate := 5.5
  let overs_played := 10
  let total_overs := 50
  let runs_scored := run_rate_10 * overs_played
  let runs_remaining := target - runs_scored
  let overs_remaining := total_overs - overs_played
  (runs_remaining / overs_remaining = required_rate)

theorem cricket_problem_solved : cricket_problem :=
by
  sorry

end cricket_problem_solved_l571_57147


namespace min_value_expression_l571_57185

theorem min_value_expression (a b c : ℝ) (h : 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (4 / c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end min_value_expression_l571_57185


namespace product_of_base8_digits_of_8654_l571_57148

theorem product_of_base8_digits_of_8654 : 
  let base10 := 8654
  let base8_rep := [2, 0, 7, 1, 6] -- Representing 8654(10) to 20716(8)
  (base8_rep.prod = 0) :=
  sorry

end product_of_base8_digits_of_8654_l571_57148


namespace tangent_line_length_l571_57184

noncomputable def curve_C (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def problem_conditions : Prop :=
  curve_C 0 = 4 ∧ cartesian 4 0 = (4, 0)

theorem tangent_line_length :
  problem_conditions → 
  ∃ l : ℝ, l = 2 :=
by
  sorry

end tangent_line_length_l571_57184


namespace base_conversion_to_zero_l571_57180

theorem base_conversion_to_zero (A B : ℕ) (hA : 0 ≤ A ∧ A < 12) (hB : 0 ≤ B ∧ B < 5) 
    (h1 : 12 * A + B = 5 * B + A) : 12 * A + B = 0 :=
by
  sorry

end base_conversion_to_zero_l571_57180


namespace travel_time_without_walking_l571_57152

-- Definitions based on the problem's conditions
def walking_time_without_escalator (x y : ℝ) : Prop := 75 * x = y
def walking_time_with_escalator (x k y : ℝ) : Prop := 30 * (x + k) = y

-- Main theorem: Time taken to travel the distance with the escalator alone
theorem travel_time_without_walking (x k y : ℝ) (h1 : walking_time_without_escalator x y) (h2 : walking_time_with_escalator x k y) : y / k = 50 :=
by
  sorry

end travel_time_without_walking_l571_57152


namespace bulb_works_longer_than_4000_hours_l571_57165

noncomputable def P_X := 0.5
noncomputable def P_Y := 0.3
noncomputable def P_Z := 0.2

noncomputable def P_4000_given_X := 0.59
noncomputable def P_4000_given_Y := 0.65
noncomputable def P_4000_given_Z := 0.70

noncomputable def P_4000 := 
  P_X * P_4000_given_X + P_Y * P_4000_given_Y + P_Z * P_4000_given_Z

theorem bulb_works_longer_than_4000_hours : P_4000 = 0.63 :=
by
  sorry

end bulb_works_longer_than_4000_hours_l571_57165


namespace lucy_groceries_total_l571_57183

theorem lucy_groceries_total (cookies noodles : ℕ) (h1 : cookies = 12) (h2 : noodles = 16) : cookies + noodles = 28 :=
by
  sorry

end lucy_groceries_total_l571_57183


namespace number_of_positive_integer_pairs_l571_57101

theorem number_of_positive_integer_pairs (x y : ℕ) (h : 20 * x + 6 * y = 2006) : 
  ∃ n, n = 34 ∧ ∀ (x y : ℕ), 20 * x + 6 * y = 2006 → 0 < x → 0 < y → 
  (∃ k, x = 3 * k + 1 ∧ y = 331 - 10 * k ∧ 0 ≤ k ∧ k ≤ 33) :=
sorry

end number_of_positive_integer_pairs_l571_57101


namespace derek_history_test_l571_57137

theorem derek_history_test :
  let ancient_questions := 20
  let medieval_questions := 25
  let modern_questions := 35
  let total_questions := ancient_questions + medieval_questions + modern_questions

  let derek_ancient_correct := 0.60 * ancient_questions
  let derek_medieval_correct := 0.56 * medieval_questions
  let derek_modern_correct := 0.70 * modern_questions

  let derek_total_correct := derek_ancient_correct + derek_medieval_correct + derek_modern_correct

  let passing_score := 0.65 * total_questions
  (derek_total_correct < passing_score) →
  passing_score - derek_total_correct = 2
  := by
  sorry

end derek_history_test_l571_57137


namespace vector_subtraction_l571_57106

-- Define the given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- State the theorem that the vector subtraction b - a equals (2, -1)
theorem vector_subtraction : b - a = (2, -1) :=
by
  -- Proof is omitted and replaced with sorry
  sorry

end vector_subtraction_l571_57106


namespace total_floor_area_covered_l571_57191

theorem total_floor_area_covered (A B C : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : B = 24) 
  (h3 : C = 19) : 
  A - (B - C) - 2 * C = 138 := 
by sorry

end total_floor_area_covered_l571_57191


namespace product_closest_value_l571_57151

theorem product_closest_value (a b : ℝ) (ha : a = 0.000321) (hb : b = 7912000) :
  abs ((a * b) - 2523) < min (abs ((a * b) - 2500)) (min (abs ((a * b) - 2700)) (min (abs ((a * b) - 3100)) (abs ((a * b) - 2000)))) := by
  sorry

end product_closest_value_l571_57151


namespace continuous_tape_length_l571_57144

theorem continuous_tape_length :
  let num_sheets := 15
  let sheet_length_cm := 25
  let overlap_cm := 0.5 
  let total_length_without_overlap := num_sheets * sheet_length_cm
  let num_overlaps := num_sheets - 1
  let total_overlap_length := num_overlaps * overlap_cm
  let total_length_cm := total_length_without_overlap - total_overlap_length
  let total_length_m := total_length_cm / 100
  total_length_m = 3.68 := 
by {
  sorry
}

end continuous_tape_length_l571_57144


namespace parabola_vertex_in_other_l571_57196

theorem parabola_vertex_in_other (p q a : ℝ) (h₁ : a ≠ 0) 
  (h₂ : ∀ (x : ℝ),  x = a → pa^2 = p * x^2) 
  (h₃ : ∀ (x : ℝ),  x = 0 → 0 = q * (x - a)^2 + pa^2) : 
  p + q = 0 := 
sorry

end parabola_vertex_in_other_l571_57196


namespace minimum_value_f_l571_57195

open Real

noncomputable def f (x : ℝ) : ℝ :=
  x + (3 * x) / (x^2 + 3) + (x * (x + 3)) / (x^2 + 1) + (3 * (x + 1)) / (x * (x^2 + 1))

theorem minimum_value_f (x : ℝ) (hx : x > 0) : f x ≥ 7 :=
by
  -- Proof omitted
  sorry

end minimum_value_f_l571_57195


namespace number_of_cows_on_farm_l571_57133

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end number_of_cows_on_farm_l571_57133


namespace union_sets_eq_real_l571_57172

def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x < 1}

theorem union_sets_eq_real : A ∪ B = Set.univ :=
by
  sorry

end union_sets_eq_real_l571_57172


namespace felicia_flour_amount_l571_57182

-- Define the conditions as constants
def white_sugar := 1 -- cups
def brown_sugar := 1 / 4 -- cups
def oil := 1 / 2 -- cups
def scoop := 1 / 4 -- cups
def total_scoops := 15 -- number of scoops

-- Define the proof statement
theorem felicia_flour_amount : 
  (total_scoops * scoop - (white_sugar + brown_sugar / scoop + oil / scoop)) * scoop = 2 :=
by
  sorry

end felicia_flour_amount_l571_57182


namespace intersection_of_complements_l571_57159

open Set

variable (U A B : Set Nat)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 4, 5})
variable (hB : B = {2, 4, 6, 8})

theorem intersection_of_complements :
  A ∩ (U \ B) = {3, 5} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l571_57159


namespace min_value_prime_factorization_l571_57128

/-- Let x and y be positive integers and assume 5 * x ^ 7 = 13 * y ^ 11.
  If x has a prime factorization of the form a ^ c * b ^ d, then the minimum possible value of a + b + c + d is 31. -/
theorem min_value_prime_factorization (x y a b c d : ℕ) (hx_pos : x > 0) (hy_pos: y > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos: c > 0) (hd_pos: d > 0)
    (h_eq : 5 * x ^ 7 = 13 * y ^ 11) (h_fact : x = a^c * b^d) : a + b + c + d = 31 :=
by
  sorry

end min_value_prime_factorization_l571_57128


namespace train_time_to_cross_tree_l571_57190

-- Definitions based on conditions
def length_of_train := 1200 -- in meters
def time_to_pass_platform := 150 -- in seconds
def length_of_platform := 300 -- in meters
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_pass_platform
def time_to_cross_tree := length_of_train / speed_of_train

-- Theorem stating the main question
theorem train_time_to_cross_tree : time_to_cross_tree = 120 := by
  sorry

end train_time_to_cross_tree_l571_57190


namespace mean_equal_l571_57117

theorem mean_equal (y : ℚ) :
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := 
by
  sorry

end mean_equal_l571_57117


namespace calculate_product_sum_l571_57138

theorem calculate_product_sum :
  17 * (17/18) + 35 * (35/36) = 50 + 1/12 :=
by sorry

end calculate_product_sum_l571_57138


namespace fourth_term_arithmetic_sequence_l571_57156

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end fourth_term_arithmetic_sequence_l571_57156


namespace find_smaller_number_l571_57174

theorem find_smaller_number (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 124) : x = 31 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l571_57174


namespace probability_between_C_and_E_l571_57176

theorem probability_between_C_and_E
  (AB AD BC BE : ℝ)
  (h₁ : AB = 4 * AD)
  (h₂ : AB = 8 * BC)
  (h₃ : AB = 2 * BE) : 
  (AB / 2 - AB / 8) / AB = 3 / 8 :=
by 
  sorry

end probability_between_C_and_E_l571_57176


namespace ratio_B_to_A_l571_57145

-- Definitions for conditions
def w_B : ℕ := 275 -- weight of element B in grams
def w_X : ℕ := 330 -- total weight of compound X in grams

-- Statement to prove
theorem ratio_B_to_A : (w_B:ℚ) / (w_X - w_B) = 5 :=
by 
  sorry

end ratio_B_to_A_l571_57145


namespace work_days_together_l571_57146

theorem work_days_together (A B : Type) (R_A R_B : ℝ) 
  (h1 : R_A = 1/2 * R_B) (h2 : R_B = 1 / 27) : 
  (1 / (R_A + R_B)) = 18 :=
by
  sorry

end work_days_together_l571_57146


namespace proof_ratio_QP_over_EF_l571_57162

noncomputable def rectangle_theorem : Prop :=
  ∃ (A B C D E F G P Q : ℝ × ℝ),
    -- Coordinates of the rectangle vertices
    A = (0, 4) ∧ B = (5, 4) ∧ C = (5, 0) ∧ D = (0, 0) ∧
    -- Coordinates of points E, F, and G on the sides of the rectangle
    E = (4, 4) ∧ F = (2, 0) ∧ G = (5, 1) ∧
    -- Coordinates of intersection points P and Q
    P = (20 / 7, 12 / 7) ∧ Q = (40 / 13, 28 / 13) ∧
    -- Ratio of distances PQ and EF
    (dist P Q)/(dist E F) = 10 / 91

theorem proof_ratio_QP_over_EF : rectangle_theorem :=
sorry

end proof_ratio_QP_over_EF_l571_57162


namespace sphere_tangent_plane_normal_line_l571_57177

variable {F : ℝ → ℝ → ℝ → ℝ}
def sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 5 = 0

def tangent_plane (x y z : ℝ) : Prop := 2*x + y + 2*z - 15 = 0

def normal_line (x y z : ℝ) : Prop := (x - 3) / 2 = (y + 1) / 1 ∧ (y + 1) / 1 = (z - 5) / 2

theorem sphere_tangent_plane_normal_line :
  sphere 3 (-1) 5 →
  tangent_plane 3 (-1) 5 ∧ normal_line 3 (-1) 5 :=
by
  intros h
  constructor
  sorry
  sorry

end sphere_tangent_plane_normal_line_l571_57177


namespace find_length_of_FC_l571_57192

theorem find_length_of_FC (DC CB AD AB ED FC : ℝ) (h1 : DC = 9) (h2 : CB = 10) (h3 : AB = (1 / 3) * AD) (h4 : ED = (2 / 3) * AD) : 
  FC = 13 := by
  sorry

end find_length_of_FC_l571_57192


namespace frosting_cupcakes_in_10_minutes_l571_57140

def speed_Cagney := 1 / 20 -- Cagney frosts 1 cupcake every 20 seconds
def speed_Lacey := 1 / 30 -- Lacey frosts 1 cupcake every 30 seconds
def speed_Jamie := 1 / 15 -- Jamie frosts 1 cupcake every 15 seconds

def combined_speed := speed_Cagney + speed_Lacey + speed_Jamie -- Combined frosting rate (cupcakes per second)

def total_seconds := 10 * 60 -- 10 minutes converted to seconds

def number_of_cupcakes := combined_speed * total_seconds -- Total number of cupcakes frosted in 10 minutes

theorem frosting_cupcakes_in_10_minutes :
  number_of_cupcakes = 90 := by
  sorry

end frosting_cupcakes_in_10_minutes_l571_57140


namespace john_share_l571_57131

theorem john_share
  (total_amount : ℝ)
  (john_ratio jose_ratio binoy_ratio : ℝ)
  (total_amount_eq : total_amount = 6000)
  (ratios_eq : john_ratio = 2 ∧ jose_ratio = 4 ∧ binoy_ratio = 6) :
  (john_ratio / (john_ratio + jose_ratio + binoy_ratio)) * total_amount = 1000 :=
by
  -- Here we would derive the proof, but just use sorry for the moment.
  sorry

end john_share_l571_57131


namespace find_a_value_l571_57124

theorem find_a_value 
  (A : Set ℝ := {x | x^2 - 4 ≤ 0})
  (B : Set ℝ := {x | 2 * x + a ≤ 0})
  (intersection : A ∩ B = {x | -2 ≤ x ∧ x ≤ 1}) : a = -2 :=
by
  sorry

end find_a_value_l571_57124


namespace ash_cloud_ratio_l571_57113

theorem ash_cloud_ratio
  (distance_ashes_shot_up : ℕ)
  (radius_ash_cloud : ℕ)
  (h1 : distance_ashes_shot_up = 300)
  (h2 : radius_ash_cloud = 2700) :
  (2 * radius_ash_cloud) / distance_ashes_shot_up = 18 :=
by
  sorry

end ash_cloud_ratio_l571_57113


namespace necessary_condition_l571_57118

theorem necessary_condition (A B C D : Prop) (h1 : A > B → C < D) : A > B → C < D := by
  exact h1 -- This is just a placeholder for the actual hypothesis, a required assumption in our initial problem statement

end necessary_condition_l571_57118


namespace chromium_percentage_new_alloy_l571_57126

-- Conditions as definitions
def first_alloy_chromium_percentage : ℝ := 12
def second_alloy_chromium_percentage : ℝ := 8
def first_alloy_weight : ℝ := 10
def second_alloy_weight : ℝ := 30

-- Final proof statement
theorem chromium_percentage_new_alloy : 
  ((first_alloy_chromium_percentage / 100 * first_alloy_weight +
    second_alloy_chromium_percentage / 100 * second_alloy_weight) /
  (first_alloy_weight + second_alloy_weight)) * 100 = 9 :=
by
  sorry

end chromium_percentage_new_alloy_l571_57126


namespace max_subset_count_l571_57105

-- Define the problem conditions in Lean 4
def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬ (a + b) % 5 = 0

theorem max_subset_count :
  ∃ (T : Finset ℕ), (is_valid_subset T) ∧ T.card = 18 := by
  sorry

end max_subset_count_l571_57105


namespace incorrect_transformation_D_l571_57150

theorem incorrect_transformation_D (x : ℝ) (hx1 : x + 1 ≠ 0) : 
  (2 - x) / (x + 1) ≠ (x - 2) / (1 + x) := 
by 
  sorry

end incorrect_transformation_D_l571_57150


namespace polygon_sides_l571_57129

-- Define the given conditions
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def sum_exterior_angles : ℕ := 360

-- Define the theorem
theorem polygon_sides (n : ℕ) (h : sum_interior_angles n = 3 * sum_exterior_angles + 180) : n = 9 :=
sorry

end polygon_sides_l571_57129


namespace mean_noon_temperature_l571_57168

def temperatures : List ℝ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_noon_temperature :
  (List.sum temperatures) / (temperatures.length) = 770 / 9 := by
  sorry

end mean_noon_temperature_l571_57168


namespace inequality_holds_for_triangle_sides_l571_57161

theorem inequality_holds_for_triangle_sides (a : ℝ) : 
  (∀ (x y z : ℕ), x + y > z ∧ y + z > x ∧ z + x > y → (x^2 + y^2 + z^2 ≤ a * (x * y + y * z + z * x))) ↔ (1 ≤ a ∧ a ≤ 6 / 5) :=
by sorry

end inequality_holds_for_triangle_sides_l571_57161


namespace distance_from_dormitory_to_city_l571_57104

theorem distance_from_dormitory_to_city (D : ℝ)
  (h1 : (1 / 5) * D + (2 / 3) * D + 4 = D) : D = 30 := by
  sorry

end distance_from_dormitory_to_city_l571_57104


namespace trees_died_l571_57170

theorem trees_died 
  (original_trees : ℕ) 
  (cut_trees : ℕ) 
  (remaining_trees : ℕ) 
  (died_trees : ℕ)
  (h1 : original_trees = 86)
  (h2 : cut_trees = 23)
  (h3 : remaining_trees = 48)
  (h4 : original_trees - died_trees - cut_trees = remaining_trees) : 
  died_trees = 15 :=
by
  sorry

end trees_died_l571_57170


namespace pond_length_l571_57155

theorem pond_length (V W D L : ℝ) (hV : V = 1600) (hW : W = 10) (hD : D = 8) :
  L = 20 ↔ V = L * W * D :=
by
  sorry

end pond_length_l571_57155


namespace purple_chip_value_l571_57173

theorem purple_chip_value 
  (x : ℕ)
  (blue_chip_value : 1 = 1)
  (green_chip_value : 5 = 5)
  (red_chip_value : 11 = 11)
  (purple_chip_condition1 : x > 5)
  (purple_chip_condition2 : x < 11)
  (product_of_points : ∃ b g p r, (b = 1 ∨ b = 1) ∧ (g = 5 ∨ g = 5) ∧ (p = x ∨ p = x) ∧ (r = 11 ∨ r = 11) ∧ b * g * p * r = 28160) : 
  x = 7 :=
sorry

end purple_chip_value_l571_57173


namespace complex_division_l571_57142

theorem complex_division (z1 z2 : ℂ) (h1 : z1 = 1 + 1 * Complex.I) (h2 : z2 = 0 + 2 * Complex.I) :
  z2 / z1 = 1 + Complex.I :=
by
  sorry

end complex_division_l571_57142


namespace total_fruit_in_buckets_l571_57169

theorem total_fruit_in_buckets (A B C : ℕ) 
  (h1 : A = B + 4)
  (h2 : B = C + 3)
  (h3 : C = 9) :
  A + B + C = 37 := by
  sorry

end total_fruit_in_buckets_l571_57169


namespace original_number_is_16_l571_57149

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l571_57149


namespace number_of_clips_after_k_steps_l571_57135

theorem number_of_clips_after_k_steps (k : ℕ) : 
  ∃ (c : ℕ), c = 2^(k-1) + 1 :=
by sorry

end number_of_clips_after_k_steps_l571_57135


namespace amount_left_after_spending_l571_57164

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end amount_left_after_spending_l571_57164


namespace find_m_l571_57123

theorem find_m (m : ℝ) (a b : ℝ) (r s : ℝ) (S1 S2 : ℝ)
  (h1 : a = 10)
  (h2 : b = 10)
  (h3 : 10 * r = 5)
  (h4 : S1 = 20)
  (h5 : 10 * s = 5 + m)
  (h6 : S2 = 100 / (5 - m))
  (h7 : S2 = 3 * S1) :
  m = 10 / 3 := by
  sorry

end find_m_l571_57123


namespace seq_formula_l571_57160

def S (n : ℕ) (a : ℕ → ℤ) : ℤ := 2 * a n + 1

theorem seq_formula (a : ℕ → ℤ) (S_n : ℕ → ℤ)
  (hS : ∀ n, S_n n = S n a) :
  a = fun n => -2^(n-1) := by
  sorry

end seq_formula_l571_57160


namespace total_sweaters_calculated_l571_57134

def monday_sweaters := 8
def tuesday_sweaters := monday_sweaters + 2
def wednesday_sweaters := tuesday_sweaters - 4
def thursday_sweaters := tuesday_sweaters - 4
def friday_sweaters := monday_sweaters / 2

def total_sweaters := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem total_sweaters_calculated : total_sweaters = 34 := 
by sorry

end total_sweaters_calculated_l571_57134


namespace brick_width_l571_57141

theorem brick_width (L W : ℕ) (l : ℕ) (b : ℕ) (n : ℕ) (A B : ℕ) 
    (courtyard_area_eq : A = L * W * 10000)
    (brick_area_eq : B = l * b)
    (total_bricks_eq : A = n * B)
    (courtyard_dims : L = 30 ∧ W = 16)
    (brick_len : l = 20)
    (num_bricks : n = 24000) :
    b = 10 := by
  sorry

end brick_width_l571_57141


namespace value_of_x_minus_y_l571_57175

theorem value_of_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end value_of_x_minus_y_l571_57175


namespace equal_distribution_arithmetic_seq_l571_57119

theorem equal_distribution_arithmetic_seq :
  ∃ (a1 d : ℚ), (a1 + (a1 + d) = (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d)) ∧ 
                (5 * a1 + 10 / 2 * d = 5) ∧ 
                (a1 = 4 / 3) :=
by
  sorry

end equal_distribution_arithmetic_seq_l571_57119


namespace sum_of_squares_of_roots_l571_57187

theorem sum_of_squares_of_roots : 
  (∃ (a b c d : ℝ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^4 - 15 * x^2 + 56 = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
    (a^2 + b^2 + c^2 + d^2 = 30)) :=
sorry

end sum_of_squares_of_roots_l571_57187


namespace max_value_ab_bc_cd_l571_57143

theorem max_value_ab_bc_cd (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 120) : ab + bc + cd ≤ 3600 :=
by {
  sorry
}

end max_value_ab_bc_cd_l571_57143


namespace jane_buys_bagels_l571_57154

variable (b m : ℕ)
variable (h1 : b + m = 7)
variable (h2 : 65 * b + 40 * m % 100 = 80)
variable (h3 : 40 * b + 40 * m % 100 = 0)

theorem jane_buys_bagels : b = 4 := by sorry

end jane_buys_bagels_l571_57154


namespace find_r_l571_57179

variable (k r : ℝ)

theorem find_r (h1 : 5 = k * 2^r) (h2 : 45 = k * 8^r) : r = (1/2) * Real.log 9 / Real.log 2 :=
sorry

end find_r_l571_57179


namespace fraction_of_quarters_1840_1849_equals_4_over_15_l571_57163

noncomputable def fraction_of_states_from_1840s (total_states : ℕ) (states_from_1840s : ℕ) : ℚ := 
  states_from_1840s / total_states

theorem fraction_of_quarters_1840_1849_equals_4_over_15 :
  fraction_of_states_from_1840s 30 8 = 4 / 15 := 
by
  sorry

end fraction_of_quarters_1840_1849_equals_4_over_15_l571_57163


namespace alissa_total_amount_spent_correct_l571_57116
-- Import necessary Lean library

-- Define the costs of individual items
def football_cost : ℝ := 8.25
def marbles_cost : ℝ := 6.59
def puzzle_cost : ℝ := 12.10
def action_figure_cost : ℝ := 15.29
def board_game_cost : ℝ := 23.47

-- Define the discount rate and the sales tax rate
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.06

-- Define the total cost before discount
def total_cost_before_discount : ℝ :=
  football_cost + marbles_cost + puzzle_cost + action_figure_cost + board_game_cost

-- Define the discount amount
def discount : ℝ := total_cost_before_discount * discount_rate

-- Define the total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount

-- Define the sales tax amount
def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_after_discount + sales_tax

-- Prove that the total amount spent is $62.68
theorem alissa_total_amount_spent_correct : total_amount_spent = 62.68 := 
  by 
    sorry

end alissa_total_amount_spent_correct_l571_57116


namespace sam_fish_count_l571_57114

/-- Let S be the number of fish Sam has. -/
def num_fish_sam : ℕ := sorry

/-- Joe has 8 times as many fish as Sam, which gives 8S fish. -/
def num_fish_joe (S : ℕ) : ℕ := 8 * S

/-- Harry has 4 times as many fish as Joe, hence 32S fish. -/
def num_fish_harry (S : ℕ) : ℕ := 32 * S

/-- Harry has 224 fish. -/
def harry_fish : ℕ := 224

/-- Prove that Sam has 7 fish given the conditions above. -/
theorem sam_fish_count : num_fish_harry num_fish_sam = harry_fish → num_fish_sam = 7 := by
  sorry

end sam_fish_count_l571_57114


namespace exponent_identity_l571_57109

variable (x : ℝ) (m n : ℝ)
axiom h1 : x^m = 6
axiom h2 : x^n = 9

theorem exponent_identity : x^(2 * m - n) = 4 :=
by
  sorry

end exponent_identity_l571_57109


namespace sum_seven_terms_l571_57166

-- Define the arithmetic sequence and sum of first n terms
variable {a : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S : ℕ → ℝ} -- The sum of the first n terms S_n

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition: a_4 = 4
def a_4_eq_4 (a : ℕ → ℝ) : Prop :=
  a 4 = 4

-- Proposition we want to prove: S_7 = 28 given a_4 = 4
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (ha : is_arithmetic_sequence a)
  (hS : sum_of_arithmetic_sequence a S)
  (h : a_4_eq_4 a) : 
  S 7 = 28 := 
sorry

end sum_seven_terms_l571_57166


namespace problem_statement_l571_57122

-- Definitions for conditions
def cond_A : Prop := ∃ B : ℝ, B = 45 ∨ B = 135
def cond_B : Prop := ∃ C : ℝ, C = 90
def cond_C : Prop := false
def cond_D : Prop := ∃ B : ℝ, 0 < B ∧ B < 60

-- Prove that only cond_A has two possibilities
theorem problem_statement : cond_A ∧ ¬cond_B ∧ ¬cond_C ∧ ¬cond_D :=
by 
  -- Lean proof goes here
  sorry

end problem_statement_l571_57122


namespace emus_count_l571_57127

theorem emus_count (E : ℕ) (heads : ℕ) (legs : ℕ) 
  (h_heads : ∀ e : ℕ, heads = e) 
  (h_legs : ∀ e : ℕ, legs = 2 * e)
  (h_total : heads + legs = 60) : 
  E = 20 :=
by sorry

end emus_count_l571_57127


namespace green_to_blue_ratio_l571_57132

-- Definition of the problem conditions
variable (G B R : ℕ)
variable (H1 : 2 * G = R)
variable (H2 : B = 80)
variable (H3 : R = 1280)

-- Theorem statement: the ratio of the green car's speed to the blue car's speed is 8:1
theorem green_to_blue_ratio (G B R : ℕ) (H1 : 2 * G = R) (H2 : B = 80) (H3 : R = 1280) :
  G / B = 8 :=
by
  sorry

end green_to_blue_ratio_l571_57132


namespace sum_of_percentages_l571_57121

theorem sum_of_percentages : (20 / 100 : ℝ) * 40 + (25 / 100 : ℝ) * 60 = 23 := 
by 
  -- Sorry skips the proof
  sorry

end sum_of_percentages_l571_57121


namespace abs_inequality_solution_l571_57188

theorem abs_inequality_solution (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := 
sorry

end abs_inequality_solution_l571_57188


namespace difference_between_max_and_min_l571_57171

noncomputable def maxThree (a b c : ℝ) : ℝ :=
  max a (max b c)

noncomputable def minThree (a b c : ℝ) : ℝ :=
  min a (min b c)

theorem difference_between_max_and_min :
  maxThree 0.12 0.23 0.22 - minThree 0.12 0.23 0.22 = 0.11 :=
by
  sorry

end difference_between_max_and_min_l571_57171


namespace minimum_value_of_quadratic_l571_57197

theorem minimum_value_of_quadratic :
  ∃ x : ℝ, (x = 6) ∧ (∀ y : ℝ, (y^2 - 12 * y + 32) ≥ -4) :=
sorry

end minimum_value_of_quadratic_l571_57197


namespace distinct_colored_triangle_l571_57130

open Finset

variables {n k : ℕ} (hn : 0 < n) (hk : 3 ≤ k)
variables (K : SimpleGraph (Fin n))
variables (color : Edge (Fin n) → Fin k)
variables (connected_subgraph : ∀ i : Fin k, ∀ u v : Fin n, u ≠ v → (∃ p : Walk (Fin n) u v, ∀ {e}, e ∈ p.edges → color e = i))

theorem distinct_colored_triangle :
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  color (A, B) ≠ color (B, C) ∧
  color (B, C) ≠ color (C, A) ∧
  color (C, A) ≠ color (A, B) :=
sorry

end distinct_colored_triangle_l571_57130


namespace kevin_correct_answer_l571_57199

theorem kevin_correct_answer (k : ℝ) (h : (20 + 1) * (6 + k) = 126 + 21 * k) :
  (20 + 1 * 6 + k) = 21 := by
sorry

end kevin_correct_answer_l571_57199


namespace balance_proof_l571_57186

variable (a b c : ℕ)

theorem balance_proof (h1 : 5 * a + 2 * b = 15 * c) (h2 : 2 * a = b + 3 * c) : 4 * b = 7 * c :=
sorry

end balance_proof_l571_57186


namespace length_of_ae_l571_57112

def consecutive_points_on_line (a b c d e : ℝ) : Prop :=
  ∃ (ab bc cd de : ℝ), 
  ab = 5 ∧ 
  bc = 2 * cd ∧ 
  de = 4 ∧ 
  a + ab = b ∧ 
  b + bc = c ∧ 
  c + cd = d ∧ 
  d + de = e ∧
  a + ab + bc = c -- ensuring ac = 11

theorem length_of_ae (a b c d e : ℝ) 
  (h1 : consecutive_points_on_line a b c d e) 
  (h2 : a + 5 = b)
  (h3 : b + 2 * (c - b) = c)
  (h4 : d - c = 3)
  (h5 : d + 4 = e)
  (h6 : a + 5 + 2 * (c - b) = c) :
  e - a = 18 :=
sorry

end length_of_ae_l571_57112


namespace bumper_cars_initial_count_l571_57102

variable {X : ℕ}

theorem bumper_cars_initial_count (h : (X - 6) + 3 = 6) : X = 9 := 
by
  sorry

end bumper_cars_initial_count_l571_57102


namespace price_of_shoes_on_tuesday_is_correct_l571_57103

theorem price_of_shoes_on_tuesday_is_correct :
  let price_thursday : ℝ := 30
  let price_friday : ℝ := price_thursday * 1.2
  let price_monday : ℝ := price_friday - price_friday * 0.15
  let price_tuesday : ℝ := price_monday - price_monday * 0.1
  price_tuesday = 27.54 := 
by
  sorry

end price_of_shoes_on_tuesday_is_correct_l571_57103


namespace Mrs_Hilt_walks_to_fountain_l571_57198

theorem Mrs_Hilt_walks_to_fountain :
  ∀ (distance trips : ℕ), distance = 30 → trips = 4 → distance * trips = 120 :=
by
  intros distance trips h_distance h_trips
  sorry

end Mrs_Hilt_walks_to_fountain_l571_57198


namespace arrangement_of_athletes_l571_57139

def num_arrangements (n : ℕ) (available_tracks_for_A : ℕ) (permutations_remaining : ℕ) : ℕ :=
  n * available_tracks_for_A * permutations_remaining

theorem arrangement_of_athletes :
  num_arrangements 2 3 24 = 144 :=
by
  sorry

end arrangement_of_athletes_l571_57139
