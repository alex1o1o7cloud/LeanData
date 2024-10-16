import Mathlib

namespace NUMINAMATH_CALUDE_f_seven_eq_neg_seventeen_l2146_214625

/-- Given a function f(x) = ax^7 + bx^3 + cx - 5, if f(-7) = 7, then f(7) = -17 -/
theorem f_seven_eq_neg_seventeen 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5)
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_eq_neg_seventeen_l2146_214625


namespace NUMINAMATH_CALUDE_intersection_complement_equals_zero_l2146_214603

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 - x = 0}

-- Define set N
def N : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

-- The theorem to prove
theorem intersection_complement_equals_zero : M ∩ (U \ N) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_zero_l2146_214603


namespace NUMINAMATH_CALUDE_rectangular_prism_inequality_l2146_214641

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0)
  (h_diagonal : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_inequality_l2146_214641


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_minimum_value_condition_l2146_214658

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3|

-- Define the function g
def g (x m : ℝ) : ℝ := f (x + m) + f (x - m)

-- Theorem for part (1)
theorem solution_set_of_inequality :
  {x : ℝ | f x > 5 - |x + 2|} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

-- Theorem for part (2)
theorem minimum_value_condition (m : ℝ) :
  (∀ x : ℝ, g x m ≥ 4) ∧ (∃ x : ℝ, g x m = 4) → m = 1 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_minimum_value_condition_l2146_214658


namespace NUMINAMATH_CALUDE_fractional_part_theorem_l2146_214637

theorem fractional_part_theorem (x : ℝ) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * x - m| ≤ 1 / n := by sorry

end NUMINAMATH_CALUDE_fractional_part_theorem_l2146_214637


namespace NUMINAMATH_CALUDE_exam_score_problem_l2146_214634

theorem exam_score_problem (correct_score : ℕ) (incorrect_score : ℤ) 
  (total_score : ℕ) (correct_answers : ℕ) :
  correct_score = 4 →
  incorrect_score = -1 →
  total_score = 150 →
  correct_answers = 42 →
  ∃ (total_questions : ℕ), 
    total_questions = correct_answers + (correct_score * correct_answers - total_score) := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2146_214634


namespace NUMINAMATH_CALUDE_emily_hens_count_l2146_214653

def total_eggs : Float := 303.0
def eggs_per_hen : Float := 10.82142857

theorem emily_hens_count : 
  (total_eggs / eggs_per_hen).round = 28 := by sorry

end NUMINAMATH_CALUDE_emily_hens_count_l2146_214653


namespace NUMINAMATH_CALUDE_cat_toy_cost_l2146_214615

def initial_amount : ℚ := 1173 / 100
def amount_left : ℚ := 151 / 100

theorem cat_toy_cost : initial_amount - amount_left = 1022 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cat_toy_cost_l2146_214615


namespace NUMINAMATH_CALUDE_white_more_probable_l2146_214689

def yellow_balls : ℕ := 3
def white_balls : ℕ := 5
def total_balls : ℕ := yellow_balls + white_balls

def prob_yellow : ℚ := yellow_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem white_more_probable : prob_white > prob_yellow := by
  sorry

end NUMINAMATH_CALUDE_white_more_probable_l2146_214689


namespace NUMINAMATH_CALUDE_sons_age_l2146_214607

theorem sons_age (father_age son_age : ℕ) 
  (h1 : 2 * son_age + father_age = 70)
  (h2 : 2 * father_age + son_age = 95)
  (h3 : father_age = 40) : son_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l2146_214607


namespace NUMINAMATH_CALUDE_system_solution_l2146_214694

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 4 * y = -7) ∧ 
  (7 * x - 3 * y = 5) ∧ 
  (x = 41 / 19) ∧ 
  (y = 64 / 19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2146_214694


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l2146_214630

/-- Proves that given Terrell hiked 8.2 miles on Saturday and 9.8 miles in total,
    the distance he hiked on Sunday is 1.6 miles. -/
theorem terrell_hike_distance (saturday_distance : Real) (total_distance : Real)
    (h1 : saturday_distance = 8.2)
    (h2 : total_distance = 9.8) :
    total_distance - saturday_distance = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l2146_214630


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l2146_214667

def remaining_pastries (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

theorem baker_remaining_pastries :
  remaining_pastries 56 29 = 27 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l2146_214667


namespace NUMINAMATH_CALUDE_det_special_matrix_l2146_214661

theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![x + 2, x, x; x, x + 2, x; x, x, x + 2] = 8 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2146_214661


namespace NUMINAMATH_CALUDE_cable_cost_per_person_l2146_214693

/-- Represents the cable program tiers and discount rates --/
structure CableProgram where
  tier1_channels : ℕ := 100
  tier1_cost : ℚ := 100
  tier2_channels : ℕ := 150
  tier2_cost : ℚ := 75
  tier3_channels : ℕ := 200
  tier4_channels : ℕ := 250
  discount_200 : ℚ := 0.1
  discount_300 : ℚ := 0.15
  discount_500 : ℚ := 0.2

/-- Calculates the cost for a given number of channels --/
def calculateCost (program : CableProgram) (channels : ℕ) : ℚ :=
  sorry

/-- Applies the appropriate discount based on the number of channels --/
def applyDiscount (program : CableProgram) (cost : ℚ) (channels : ℕ) : ℚ :=
  sorry

/-- Theorem: The cost per person for 375 channels split among 4 people is $57.11 --/
theorem cable_cost_per_person (program : CableProgram) :
  let total_cost := calculateCost program 375
  let discounted_cost := applyDiscount program total_cost 375
  let cost_per_person := discounted_cost / 4
  cost_per_person = 57.11 := by
  sorry

end NUMINAMATH_CALUDE_cable_cost_per_person_l2146_214693


namespace NUMINAMATH_CALUDE_rachel_colored_pictures_l2146_214685

def coloring_book_problem (book1_pictures book2_pictures remaining_pictures : ℕ) : Prop :=
  let total_pictures := book1_pictures + book2_pictures
  let colored_pictures := total_pictures - remaining_pictures
  colored_pictures = 44

theorem rachel_colored_pictures :
  coloring_book_problem 23 32 11 := by
  sorry

end NUMINAMATH_CALUDE_rachel_colored_pictures_l2146_214685


namespace NUMINAMATH_CALUDE_unique_four_digit_palindromic_square_l2146_214613

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem unique_four_digit_palindromic_square : 
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_palindromic_square_l2146_214613


namespace NUMINAMATH_CALUDE_choose_team_with_smaller_variance_l2146_214632

-- Define the teams
inductive Team
  | A
  | B

-- Define the properties of the teams
def average_height : ℝ := 1.72
def variance (t : Team) : ℝ :=
  match t with
  | Team.A => 1.2
  | Team.B => 5.6

-- Define a function to determine which team has more uniform heights
def more_uniform_heights (t1 t2 : Team) : Prop :=
  variance t1 < variance t2

-- Theorem statement
theorem choose_team_with_smaller_variance :
  more_uniform_heights Team.A Team.B :=
sorry

end NUMINAMATH_CALUDE_choose_team_with_smaller_variance_l2146_214632


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2146_214606

-- Define the triangle ABC
theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  -- Conditions
  a = 1 →
  B = π / 4 → -- 45° in radians
  S = 2 →
  S = (1 / 2) * a * c * Real.sin B →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  -- Conclusion
  c = 4 * Real.sqrt 2 ∧ b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l2146_214606


namespace NUMINAMATH_CALUDE_cathys_total_money_l2146_214676

def cathys_money (initial_balance dad_contribution : ℕ) : ℕ :=
  initial_balance + dad_contribution + 2 * dad_contribution

theorem cathys_total_money :
  cathys_money 12 25 = 87 := by
  sorry

end NUMINAMATH_CALUDE_cathys_total_money_l2146_214676


namespace NUMINAMATH_CALUDE_odd_number_grouping_l2146_214680

theorem odd_number_grouping (n : ℕ) (odd_number : ℕ) : 
  (odd_number = 2007) →
  (∀ k : ℕ, k < n → (k^2 < 1004 ∧ 1004 ≤ (k+1)^2)) →
  (n = 32) :=
by sorry

end NUMINAMATH_CALUDE_odd_number_grouping_l2146_214680


namespace NUMINAMATH_CALUDE_m_salary_percentage_l2146_214687

def total_salary : ℝ := 550
def n_salary : ℝ := 250

theorem m_salary_percentage : 
  (total_salary - n_salary) / n_salary * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_m_salary_percentage_l2146_214687


namespace NUMINAMATH_CALUDE_circle_max_distance_squared_l2146_214647

theorem circle_max_distance_squared (x y : ℝ) (h : x^2 + (y - 2)^2 = 1) :
  x^2 + y^2 ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_circle_max_distance_squared_l2146_214647


namespace NUMINAMATH_CALUDE_zeros_after_one_in_power_l2146_214699

theorem zeros_after_one_in_power (n : ℕ) (h : 10000 = 10^4) :
  10000^50 = 10^200 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_power_l2146_214699


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l2146_214662

-- Define the arithmetic-geometric sequence and its properties
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

-- Define S_n as the sum of the first n terms of a_n
def S (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S a n + a (n + 1)

-- Define T_n as the sum of the first n terms of S_n
def T (S : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => T S n + S (n + 1)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_ag : arithmetic_geometric_sequence a)
  (h_S3 : S a 3 = 7)
  (h_S6 : S a 6 = 63) :
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (∀ n : ℕ, S a n = 2^n - 1) ∧
  (∀ n : ℕ, T (S a) n = 2^(n + 1) - n - 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l2146_214662


namespace NUMINAMATH_CALUDE_median_in_group_two_l2146_214643

/-- Represents the labor time groups --/
inductive LaborGroup
  | One
  | Two
  | Three
  | Four

/-- The frequency of each labor group --/
def frequency : LaborGroup → Nat
  | LaborGroup.One => 10
  | LaborGroup.Two => 20
  | LaborGroup.Three => 12
  | LaborGroup.Four => 8

/-- The total number of surveyed students --/
def totalStudents : Nat := 50

/-- The cumulative frequency up to and including a given group --/
def cumulativeFrequency (g : LaborGroup) : Nat :=
  match g with
  | LaborGroup.One => frequency LaborGroup.One
  | LaborGroup.Two => frequency LaborGroup.One + frequency LaborGroup.Two
  | LaborGroup.Three => frequency LaborGroup.One + frequency LaborGroup.Two + frequency LaborGroup.Three
  | LaborGroup.Four => totalStudents

/-- The median position --/
def medianPosition : Nat := totalStudents / 2

theorem median_in_group_two :
  cumulativeFrequency LaborGroup.One < medianPosition ∧
  medianPosition ≤ cumulativeFrequency LaborGroup.Two :=
sorry

end NUMINAMATH_CALUDE_median_in_group_two_l2146_214643


namespace NUMINAMATH_CALUDE_polynomial_sum_of_terms_l2146_214698

def polynomial (x : ℝ) : ℝ := 4 * x^2 - 3 * x - 2

def term1 (x : ℝ) : ℝ := 4 * x^2
def term2 (x : ℝ) : ℝ := -3 * x
def term3 : ℝ := -2

theorem polynomial_sum_of_terms :
  ∀ x : ℝ, polynomial x = term1 x + term2 x + term3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_terms_l2146_214698


namespace NUMINAMATH_CALUDE_count_integers_satisfying_equation_l2146_214623

-- Define the function g
def g (n : ℤ) : ℤ := ⌈(101 * n : ℚ) / 102⌉ - ⌊(102 * n : ℚ) / 103⌋

-- State the theorem
theorem count_integers_satisfying_equation : 
  (∃ (S : Finset ℤ), (∀ n ∈ S, g n = 1) ∧ (∀ n ∉ S, g n ≠ 1) ∧ Finset.card S = 10506) :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_equation_l2146_214623


namespace NUMINAMATH_CALUDE_wife_departure_time_l2146_214688

/-- Prove that given a man driving at 40 miles/hr and his wife driving at 50 miles/hr,
    if they meet in 2 hours, the wife left 24 minutes after the man. -/
theorem wife_departure_time (man_speed wife_speed meeting_time : ℝ) 
  (h1 : man_speed = 40)
  (h2 : wife_speed = 50)
  (h3 : meeting_time = 2) : 
  (wife_speed * meeting_time - man_speed * meeting_time) / wife_speed * 60 = 24 := by
sorry


end NUMINAMATH_CALUDE_wife_departure_time_l2146_214688


namespace NUMINAMATH_CALUDE_range_of_a_l2146_214673

-- Define the propositions p and q
def p (a x : ℝ) : Prop := 3 * a < x ∧ x < a
def q (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (a < 0) →
  (∀ x : ℝ, ¬(p a x) → ¬(q x)) →
  (∃ x : ℝ, ¬(p a x) ∧ q x) →
  -2/3 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2146_214673


namespace NUMINAMATH_CALUDE_april_flower_sale_l2146_214657

/-- April's flower sale problem -/
theorem april_flower_sale (initial_roses : ℕ) (remaining_roses : ℕ) (price_per_rose : ℕ) :
  initial_roses = 13 →
  remaining_roses = 4 →
  price_per_rose = 4 →
  (initial_roses - remaining_roses) * price_per_rose = 36 := by
sorry

end NUMINAMATH_CALUDE_april_flower_sale_l2146_214657


namespace NUMINAMATH_CALUDE_vector_decomposition_l2146_214684

def x : ℝ × ℝ × ℝ := (-13, 2, 18)
def p : ℝ × ℝ × ℝ := (1, 1, 4)
def q : ℝ × ℝ × ℝ := (-3, 0, 2)
def r : ℝ × ℝ × ℝ := (1, 2, -1)

theorem vector_decomposition :
  x = (2 : ℝ) • p + (5 : ℝ) • q := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2146_214684


namespace NUMINAMATH_CALUDE_square_area_error_l2146_214627

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.05 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 10.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2146_214627


namespace NUMINAMATH_CALUDE_browser_tabs_remaining_l2146_214651

theorem browser_tabs_remaining (initial_tabs : ℕ) : 
  initial_tabs = 400 → 
  (initial_tabs - initial_tabs / 4 - (initial_tabs - initial_tabs / 4) * 2 / 5) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_browser_tabs_remaining_l2146_214651


namespace NUMINAMATH_CALUDE_open_box_volume_l2146_214654

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h_length : sheet_length = 100)
  (h_width : sheet_width = 50)
  (h_cut : cut_length = 10) :
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 24000 := by
sorry


end NUMINAMATH_CALUDE_open_box_volume_l2146_214654


namespace NUMINAMATH_CALUDE_min_value_expression_l2146_214624

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2146_214624


namespace NUMINAMATH_CALUDE_number_of_valid_paths_l2146_214601

-- Define the grid dimensions
def columns : ℕ := 10
def rows : ℕ := 4

-- Define the forbidden segment
def forbidden_column : ℕ := 6
def forbidden_row_start : ℕ := 2
def forbidden_row_end : ℕ := 3

-- Define the total number of steps
def total_steps : ℕ := columns + rows

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Function to calculate the number of paths between two points
def paths_between (col_diff row_diff : ℕ) : ℕ := 
  binomial (col_diff + row_diff) row_diff

-- Theorem statement
theorem number_of_valid_paths : 
  paths_between columns rows - 
  (paths_between forbidden_column (rows - forbidden_row_end) * 
   paths_between (columns - forbidden_column) (forbidden_row_end)) = 861 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_paths_l2146_214601


namespace NUMINAMATH_CALUDE_blue_red_face_ratio_l2146_214639

theorem blue_red_face_ratio (n : ℕ) (h : n = 13) : 
  let red_area := 6 * n^2
  let total_area := 6 * n^3
  let blue_area := total_area - red_area
  blue_area / red_area = 12 := by sorry

end NUMINAMATH_CALUDE_blue_red_face_ratio_l2146_214639


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l2146_214618

theorem bakery_flour_usage (wheat_flour : ℝ) (white_flour : ℝ) 
  (h1 : wheat_flour = 0.2)
  (h2 : white_flour = 0.1) :
  wheat_flour + white_flour = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l2146_214618


namespace NUMINAMATH_CALUDE_function_greater_than_three_sixteenths_l2146_214690

/-- The function f(x) = x^2 + 2mx + m is greater than 3/16 for all x if and only if 1/4 < m < 3/4 -/
theorem function_greater_than_three_sixteenths (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*m*x + m > 3/16) ↔ (1/4 < m ∧ m < 3/4) :=
sorry

end NUMINAMATH_CALUDE_function_greater_than_three_sixteenths_l2146_214690


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2146_214648

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2146_214648


namespace NUMINAMATH_CALUDE_cyclists_average_speed_cyclists_average_speed_is_22_point_5_l2146_214629

/-- Cyclist's average speed problem -/
theorem cyclists_average_speed (total_distance : ℝ) (initial_speed : ℝ) 
  (speed_increase : ℝ) (distance_fraction : ℝ) : ℝ :=
  let new_speed := initial_speed * (1 + speed_increase)
  let time_first_part := (distance_fraction * total_distance) / initial_speed
  let time_second_part := ((1 - distance_fraction) * total_distance) / new_speed
  let total_time := time_first_part + time_second_part
  total_distance / total_time

/-- Proof of the cyclist's average speed -/
theorem cyclists_average_speed_is_22_point_5 :
  cyclists_average_speed 1 20 0.2 (1/3) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_average_speed_cyclists_average_speed_is_22_point_5_l2146_214629


namespace NUMINAMATH_CALUDE_calculation_proof_l2146_214691

theorem calculation_proof :
  ((-1 - (1 + 0.5) * (1/3) + (-4)) = -11/2) ∧
  ((-8^2 + 3 * (-2)^2 + (-6) + (-1/3)^2) = -521/9) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2146_214691


namespace NUMINAMATH_CALUDE_term_2023_equals_2023_term_2023_in_group_64_term_2023_is_7th_in_group_l2146_214619

/-- Definition of the sequence term -/
def sequenceTerm (n : ℕ) : ℕ :=
  let start := 1 + n * (n - 1) / 2
  (n * (2 * start + (n - 1))) / 2

/-- Theorem stating that the 2023rd term of the sequence is 2023 -/
theorem term_2023_equals_2023 : sequenceTerm 64 = 2023 := by
  sorry

/-- Theorem stating that the 2023rd term is in the 64th group -/
theorem term_2023_in_group_64 :
  (63 * 64) / 2 < 2023 ∧ 2023 ≤ (64 * 65) / 2 := by
  sorry

/-- Theorem stating that the 2023rd term is the 7th term in its group -/
theorem term_2023_is_7th_in_group :
  2023 - (63 * 64) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_term_2023_equals_2023_term_2023_in_group_64_term_2023_is_7th_in_group_l2146_214619


namespace NUMINAMATH_CALUDE_probability_of_matching_pair_l2146_214642

def blue_socks : ℕ := 12
def red_socks : ℕ := 10

def total_socks : ℕ := blue_socks + red_socks

def ways_to_pick_two (n : ℕ) : ℕ := n * (n - 1) / 2

def matching_pairs : ℕ := ways_to_pick_two blue_socks + ways_to_pick_two red_socks

def total_combinations : ℕ := ways_to_pick_two total_socks

theorem probability_of_matching_pair :
  (matching_pairs : ℚ) / total_combinations = 111 / 231 := by sorry

end NUMINAMATH_CALUDE_probability_of_matching_pair_l2146_214642


namespace NUMINAMATH_CALUDE_total_players_specific_l2146_214682

/-- The number of players in a sports event with overlapping groups --/
def totalPlayers (kabadi khoKho soccer kabadi_khoKho kabadi_soccer khoKho_soccer all_three : ℕ) : ℕ :=
  kabadi + khoKho + soccer - kabadi_khoKho - kabadi_soccer - khoKho_soccer + all_three

/-- Theorem stating the total number of players given the specific conditions --/
theorem total_players_specific : totalPlayers 50 80 30 15 10 25 8 = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_players_specific_l2146_214682


namespace NUMINAMATH_CALUDE_optimal_production_value_l2146_214695

/-- Represents the production plan for products A and B -/
structure ProductionPlan where
  a : ℝ  -- Amount of product A in kg
  b : ℝ  -- Amount of product B in kg

/-- Calculates the total value of a production plan -/
def totalValue (plan : ProductionPlan) : ℝ :=
  600 * plan.a + 400 * plan.b

/-- Checks if a production plan is feasible given the raw material constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  4 * plan.a + 2 * plan.b ≤ 100 ∧  -- Raw material A constraint
  2 * plan.a + 3 * plan.b ≤ 120    -- Raw material B constraint

/-- The optimal production plan -/
def optimalPlan : ProductionPlan :=
  { a := 7.5, b := 35 }

theorem optimal_production_value :
  (∀ plan : ProductionPlan, isFeasible plan → totalValue plan ≤ totalValue optimalPlan) ∧
  isFeasible optimalPlan ∧
  totalValue optimalPlan = 18500 := by
  sorry

end NUMINAMATH_CALUDE_optimal_production_value_l2146_214695


namespace NUMINAMATH_CALUDE_adjacent_numbers_selection_l2146_214610

theorem adjacent_numbers_selection (n : ℕ) (k : ℕ) : 
  n = 49 → k = 6 → 
  (Nat.choose n k) - (Nat.choose (n - k + 1) k) = 
  (Nat.choose n k) - (Nat.choose 44 k) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_numbers_selection_l2146_214610


namespace NUMINAMATH_CALUDE_johns_total_payment_l2146_214652

def nike_cost : ℝ := 150
def work_boots_cost : ℝ := 120
def tax_rate : ℝ := 0.1

def total_cost : ℝ := nike_cost + work_boots_cost + (nike_cost + work_boots_cost) * tax_rate

theorem johns_total_payment : total_cost = 297 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_payment_l2146_214652


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l2146_214686

-- Problem 1
theorem problem_one : 27 - (-12) + 3 - 7 = 35 := by sorry

-- Problem 2
theorem problem_two : (-3 - 1/3) * 2/5 * (-2 - 1/2) / (-10/7) = -7/3 := by sorry

-- Problem 3
theorem problem_three : (3/4 - 7/8 - 7/12) * (-12) = 17/2 := by sorry

-- Problem 4
theorem problem_four : 4 / (-2/3)^2 + 1 + (-1)^2023 = 9 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l2146_214686


namespace NUMINAMATH_CALUDE_indicator_light_signals_l2146_214660

/-- The number of indicator lights in a row -/
def num_lights : ℕ := 8

/-- The number of lights displayed at a time -/
def lights_displayed : ℕ := 4

/-- The number of adjacent lights among those displayed -/
def adjacent_lights : ℕ := 3

/-- The number of colors each light can display -/
def colors_per_light : ℕ := 2

/-- The total number of different signals that can be displayed -/
def total_signals : ℕ := 320

theorem indicator_light_signals :
  (num_lights = 8) →
  (lights_displayed = 4) →
  (adjacent_lights = 3) →
  (colors_per_light = 2) →
  total_signals = 320 := by sorry

end NUMINAMATH_CALUDE_indicator_light_signals_l2146_214660


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_five_l2146_214616

theorem unique_square_divisible_by_five : ∃! y : ℕ, 
  (∃ n : ℕ, y = n^2) ∧ 
  (∃ k : ℕ, y = 5 * k) ∧ 
  50 < y ∧ y < 120 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_five_l2146_214616


namespace NUMINAMATH_CALUDE_a_not_periodic_l2146_214604

/-- The first digit of a positive integer -/
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

/-- The sequence a_n where a_n is the first digit of n^2 -/
def a (n : ℕ) : ℕ :=
  firstDigit (n * n)

/-- A sequence is periodic if there exists a positive integer p such that
    for all n ≥ some N, a(n+p) = a(n) -/
def isPeriodic (f : ℕ → ℕ) : Prop :=
  ∃ p N : ℕ, p > 0 ∧ ∀ n ≥ N, f (n + p) = f n

/-- The sequence a_n is not periodic -/
theorem a_not_periodic : ¬ isPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_a_not_periodic_l2146_214604


namespace NUMINAMATH_CALUDE_probability_of_pair_after_removal_l2146_214692

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset (Fin 13 × Fin 4))
  (card_count : cards.card = 52)

/-- Represents the deck after removing a pair and a single card -/
def RemainingDeck (d : Deck) : Finset (Fin 13 × Fin 4) :=
  d.cards.filter (λ _ => true)  -- This is a placeholder; actual implementation would remove cards

/-- Probability of selecting a matching pair from the remaining deck -/
def ProbabilityOfPair (d : Deck) : ℚ :=
  67 / 1176

/-- Main theorem: The probability of selecting a matching pair is 67/1176 -/
theorem probability_of_pair_after_removal (d : Deck) : 
  ProbabilityOfPair d = 67 / 1176 := by
  sorry

#eval (67 : ℕ) + 1176  -- Should output 1243

end NUMINAMATH_CALUDE_probability_of_pair_after_removal_l2146_214692


namespace NUMINAMATH_CALUDE_distance_to_softball_park_l2146_214650

/-- Represents the problem of calculating the distance to the softball park -/
def softball_park_distance (efficiency : ℝ) (initial_gas : ℝ) 
  (to_school : ℝ) (to_restaurant : ℝ) (to_friend : ℝ) (to_home : ℝ) : ℝ :=
  efficiency * initial_gas - (to_school + to_restaurant + to_friend + to_home)

/-- Theorem stating that the distance to the softball park is 6 miles -/
theorem distance_to_softball_park :
  softball_park_distance 19 2 15 2 4 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_softball_park_l2146_214650


namespace NUMINAMATH_CALUDE_problem_solution_l2146_214664

theorem problem_solution :
  ∀ (x y z : ℝ),
  (x + x = y * x) →
  (x + x = z * z) →
  (y = 3) →
  (x * z = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2146_214664


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_real_l2146_214612

theorem sqrt_x_minus_5_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) ↔ x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_real_l2146_214612


namespace NUMINAMATH_CALUDE_number_of_boys_l2146_214649

/-- Proves that the number of boys in a group is 5 given specific conditions about weights --/
theorem number_of_boys (num_girls : ℕ) (num_total : ℕ) (avg_girls : ℚ) (avg_boys : ℚ) (avg_total : ℚ) :
  num_girls = 5 →
  num_total = 10 →
  avg_girls = 45 →
  avg_boys = 55 →
  avg_total = 50 →
  ∃ (num_boys : ℕ), num_boys = 5 ∧ num_girls + num_boys = num_total ∧
    (num_girls : ℚ) * avg_girls + (num_boys : ℚ) * avg_boys = (num_total : ℚ) * avg_total :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_boys_l2146_214649


namespace NUMINAMATH_CALUDE_sin_cos_transformation_given_condition_l2146_214674

theorem sin_cos_transformation (x : ℝ) :
  4 * Real.sin x * Real.cos x = 2 * Real.sin (2 * x + π / 6) :=
by
  sorry

-- Additional theorem to represent the given condition
theorem given_condition (x : ℝ) :
  Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * x - π / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_given_condition_l2146_214674


namespace NUMINAMATH_CALUDE_specific_sculpture_surface_area_l2146_214636

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  bottomLayerCount : ℕ
  middleLayerCount : ℕ
  topLayerCount : ℕ
  cubeEdgeLength : ℝ

/-- Calculates the exposed surface area of a cube sculpture -/
def exposedSurfaceArea (sculpture : CubeSculpture) : ℝ :=
  sorry

/-- The theorem stating that the specific sculpture has 55 square meters of exposed surface area -/
theorem specific_sculpture_surface_area :
  let sculpture : CubeSculpture := {
    bottomLayerCount := 9
    middleLayerCount := 8
    topLayerCount := 3
    cubeEdgeLength := 1
  }
  exposedSurfaceArea sculpture = 55 := by
  sorry

end NUMINAMATH_CALUDE_specific_sculpture_surface_area_l2146_214636


namespace NUMINAMATH_CALUDE_weight_of_ten_moles_l2146_214617

/-- Represents an iron oxide compound with the number of iron and oxygen atoms -/
structure IronOxide where
  iron_atoms : ℕ
  oxygen_atoms : ℕ

/-- Calculates the molar mass of an iron oxide compound -/
def molar_mass (compound : IronOxide) : ℝ :=
  55.85 * compound.iron_atoms + 16.00 * compound.oxygen_atoms

/-- Calculates the weight of a given number of moles of an iron oxide compound -/
def weight (moles : ℝ) (compound : IronOxide) : ℝ :=
  moles * molar_mass compound

/-- Theorem: The weight of 10 moles of an iron oxide compound is 10 times its molar mass -/
theorem weight_of_ten_moles (compound : IronOxide) :
  weight 10 compound = 10 * molar_mass compound := by
  sorry

#check weight_of_ten_moles

end NUMINAMATH_CALUDE_weight_of_ten_moles_l2146_214617


namespace NUMINAMATH_CALUDE_rectangle_width_l2146_214666

theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 1) → w = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2146_214666


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2146_214675

/-- Given a quadratic equation k^2x^2 + (4k-1)x + 4 = 0 with two distinct real roots,
    the range of values for k is k < 1/8 and k ≠ 0 -/
theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k^2 * x₁^2 + (4*k - 1) * x₁ + 4 = 0 ∧
    k^2 * x₂^2 + (4*k - 1) * x₂ + 4 = 0) →
  k < 1/8 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2146_214675


namespace NUMINAMATH_CALUDE_prize_distributions_count_l2146_214671

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 7

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := num_bowlers - 1

/-- The number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- The total number of possible prize distributions -/
def total_distributions : ℕ := outcomes_per_game ^ num_games

/-- Theorem stating that the number of possible prize distributions is 64 -/
theorem prize_distributions_count :
  total_distributions = 64 := by sorry

end NUMINAMATH_CALUDE_prize_distributions_count_l2146_214671


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l2146_214670

theorem fraction_to_zero_power (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a / b : ℚ) ^ (0 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l2146_214670


namespace NUMINAMATH_CALUDE_find_x_l2146_214608

theorem find_x : ∃ x : ℚ, x * 9999 = 724827405 ∧ x = 72492.75 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2146_214608


namespace NUMINAMATH_CALUDE_martins_bells_l2146_214644

theorem martins_bells (S B : ℤ) : 
  S = B / 3 + B^2 / 4 →
  S + B = 52 →
  B > 0 →
  B = 12 := by
sorry

end NUMINAMATH_CALUDE_martins_bells_l2146_214644


namespace NUMINAMATH_CALUDE_inverse_function_problem_l2146_214677

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_problem (h : ∀ x > 0, f⁻¹ x = x^2) : f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l2146_214677


namespace NUMINAMATH_CALUDE_set_B_is_empty_l2146_214640

theorem set_B_is_empty : {x : ℝ | x^2 + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_set_B_is_empty_l2146_214640


namespace NUMINAMATH_CALUDE_investment_division_l2146_214611

/-- 
Given a total amount of 3200 divided into two parts, where one part is invested at 3% 
and the other at 5%, and the total annual interest is 144, prove that the amount 
of the first part (invested at 3%) is 800.
-/
theorem investment_division (x : ℝ) : 
  x ≥ 0 ∧ 
  3200 - x ≥ 0 ∧ 
  0.03 * x + 0.05 * (3200 - x) = 144 → 
  x = 800 := by
  sorry

#check investment_division

end NUMINAMATH_CALUDE_investment_division_l2146_214611


namespace NUMINAMATH_CALUDE_cow_count_l2146_214678

/-- Given a group of cows and hens, prove that the number of cows is 4 when the total number of legs
    is 8 more than twice the number of heads. -/
theorem cow_count (cows hens : ℕ) : 
  (4 * cows + 2 * hens = 2 * (cows + hens) + 8) → cows = 4 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l2146_214678


namespace NUMINAMATH_CALUDE_dog_treat_expenditure_l2146_214621

/-- Represents the cost and nutritional value of dog treats -/
structure DogTreat where
  cost : ℚ
  np : ℕ

/-- Calculates the discounted price based on quantity and discount rate -/
def discountedPrice (regularPrice : ℚ) (quantity : ℕ) (discountRate : ℚ) : ℚ :=
  regularPrice * (1 - discountRate)

/-- Theorem: The total expenditure on dog treats for the month is $11.70 -/
theorem dog_treat_expenditure :
  let treatA : DogTreat := { cost := 0.1, np := 1 }
  let treatB : DogTreat := { cost := 0.15, np := 2 }
  let quantityA : ℕ := 50
  let quantityB : ℕ := 60
  let discountRateA : ℚ := 0.1
  let discountRateB : ℚ := 0.2
  let totalNP : ℕ := quantityA * treatA.np + quantityB * treatB.np
  let regularPriceA : ℚ := treatA.cost * quantityA
  let regularPriceB : ℚ := treatB.cost * quantityB
  let discountedPriceA : ℚ := discountedPrice regularPriceA quantityA discountRateA
  let discountedPriceB : ℚ := discountedPrice regularPriceB quantityB discountRateB
  let totalExpenditure : ℚ := discountedPriceA + discountedPriceB
  totalNP ≥ 40 ∧ totalExpenditure = 11.7 := by
  sorry


end NUMINAMATH_CALUDE_dog_treat_expenditure_l2146_214621


namespace NUMINAMATH_CALUDE_bacteria_increase_l2146_214663

/-- Given an original bacteria count of 600 and a current count of 8917,
    prove that the increase in bacteria count is 8317. -/
theorem bacteria_increase (original : ℕ) (current : ℕ) 
  (h1 : original = 600) (h2 : current = 8917) : 
  current - original = 8317 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_increase_l2146_214663


namespace NUMINAMATH_CALUDE_tim_has_33_books_l2146_214696

/-- The number of books Tim has, given the initial conditions -/
def tims_books (benny_initial : ℕ) (sandy_received : ℕ) (total : ℕ) : ℕ :=
  total - (benny_initial - sandy_received)

/-- Theorem stating that Tim has 33 books under the given conditions -/
theorem tim_has_33_books :
  tims_books 24 10 47 = 33 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_33_books_l2146_214696


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2146_214602

/-- Represents a seating arrangement of 12 people around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Represents a married couple -/
structure Couple :=
  (husband : Fin 12)
  (wife : Fin 12)

/-- Checks if two positions are adjacent on the round table -/
def isAdjacent (a b : Fin 12) : Prop :=
  (a + 1) % 12 = b ∨ (b + 1) % 12 = a

/-- Checks if two positions are two seats apart on the round table -/
def isTwoApart (a b : Fin 12) : Prop :=
  (a + 2) % 12 = b ∨ (b + 2) % 12 = a

/-- Checks if two positions are across from each other on the round table -/
def isAcross (a b : Fin 12) : Prop :=
  (a + 6) % 12 = b

/-- Checks if a seating arrangement satisfies all conditions -/
def isValidArrangement (s : SeatingArrangement) (couples : List Couple) : Prop :=
  ∀ i j : Fin 12, i ≠ j →
    (¬isAdjacent (s i) (s j) ∨ (i % 2 = 0 ↔ j % 2 = 1)) ∧
    (∀ c : Couple, c ∈ couples →
      ¬isAdjacent (s c.husband) (s c.wife) ∧
      ¬isTwoApart (s c.husband) (s c.wife) ∧
      ¬isAcross (s c.husband) (s c.wife))

/-- The main theorem stating that there are exactly 17280 valid seating arrangements -/
theorem seating_arrangements_count :
  ∃! (n : ℕ), ∃ (arrangements : Finset SeatingArrangement) (couples : List Couple),
    couples.length = 6 ∧
    arrangements.card = n ∧
    (∀ s ∈ arrangements, isValidArrangement s couples) ∧
    (∀ s, isValidArrangement s couples → s ∈ arrangements) ∧
    n = 17280 :=
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2146_214602


namespace NUMINAMATH_CALUDE_sin_sum_l2146_214622

theorem sin_sum (α β : ℝ) : Real.sin (α + β) = Real.sin α * Real.cos β + Real.cos α * Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_l2146_214622


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2146_214697

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the eccentricity of the hyperbola is √(17)/3 -/
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (F : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : b = 1)
  (h4 : F = (c, 0))
  (h5 : B = (0, 1))
  (h6 : A.1^2 / a^2 - A.2^2 / b^2 = 1)  -- A is on the hyperbola
  (h7 : A.1^2 = 4 * A.2)                -- A is on the parabola
  (h8 : (A.1 - B.1, A.2 - B.2) = 3 * (F.1 - A.1, F.2 - A.2))  -- BA = 3AF
  : c / a = Real.sqrt 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2146_214697


namespace NUMINAMATH_CALUDE_m_range_l2146_214656

def f (x : ℝ) : ℝ := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2146_214656


namespace NUMINAMATH_CALUDE_fraction_inequality_l2146_214626

theorem fraction_inequality (x : ℝ) : (x + 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2146_214626


namespace NUMINAMATH_CALUDE_carol_cupcakes_l2146_214620

/-- The number of cupcakes Carol has after initially making some, selling some, and making more. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (made_more : ℕ) : ℕ :=
  initial - sold + made_more

/-- Theorem stating that Carol has 49 cupcakes in total -/
theorem carol_cupcakes : total_cupcakes 30 9 28 = 49 := by
  sorry

end NUMINAMATH_CALUDE_carol_cupcakes_l2146_214620


namespace NUMINAMATH_CALUDE_river_speed_calculation_l2146_214683

-- Define the swimmer's speed in still water
variable (a : ℝ) 

-- Define the speed of the river flow
def river_speed : ℝ := 0.02

-- Define the time the swimmer swam upstream before realizing the loss
def upstream_time : ℝ := 0.5

-- Define the distance downstream where the swimmer catches up to the bottle
def downstream_distance : ℝ := 1.2

-- Theorem statement
theorem river_speed_calculation (h : ∀ a > 0, 
  (downstream_distance + upstream_time * (a - 60 * river_speed)) / (a + 60 * river_speed) = 
  downstream_distance / (60 * river_speed) - upstream_time) : 
  river_speed = 0.02 := by sorry

end NUMINAMATH_CALUDE_river_speed_calculation_l2146_214683


namespace NUMINAMATH_CALUDE_remainder_172_pow_172_mod_13_l2146_214638

theorem remainder_172_pow_172_mod_13 : 172^172 % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_172_pow_172_mod_13_l2146_214638


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l2146_214646

theorem batsman_running_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (h1 : total_runs = 125)
  (h2 : boundaries = 5)
  (h3 : sixes = 5) :
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l2146_214646


namespace NUMINAMATH_CALUDE_number_components_l2146_214665

def number : ℕ := 1234000000

theorem number_components : 
  (number / 100000000 = 12) ∧ 
  ((number / 10000000) % 10 = 3) ∧ 
  ((number / 1000000) % 10 = 4) := by
  sorry

end NUMINAMATH_CALUDE_number_components_l2146_214665


namespace NUMINAMATH_CALUDE_apple_pie_price_per_pound_l2146_214633

/-- Given the following conditions for an apple pie:
  - The pie serves 8 people
  - 2 pounds of apples are needed
  - Pre-made pie crust costs $2.00
  - Lemon costs $0.50
  - Butter costs $1.50
  - Each serving of pie costs $1
  Prove that the price per pound of apples is $2.00 -/
theorem apple_pie_price_per_pound (servings : ℕ) (apple_pounds : ℝ) 
  (crust_cost lemon_cost butter_cost serving_cost : ℝ) :
  servings = 8 → 
  apple_pounds = 2 → 
  crust_cost = 2 → 
  lemon_cost = 0.5 → 
  butter_cost = 1.5 → 
  serving_cost = 1 → 
  (servings * serving_cost - (crust_cost + lemon_cost + butter_cost)) / apple_pounds = 2 :=
by sorry


end NUMINAMATH_CALUDE_apple_pie_price_per_pound_l2146_214633


namespace NUMINAMATH_CALUDE_sqrt_123400_l2146_214669

theorem sqrt_123400 (h1 : Real.sqrt 12.34 = 3.512) (h2 : Real.sqrt 123.4 = 11.108) :
  Real.sqrt 123400 = 351.2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_123400_l2146_214669


namespace NUMINAMATH_CALUDE_factor_expression_l2146_214645

theorem factor_expression (a b : ℝ) : 56 * b^2 * a^2 + 168 * b * a = 56 * b * a * (b * a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2146_214645


namespace NUMINAMATH_CALUDE_part1_part2_l2146_214659

-- Define the operation F
def F (a b x y : ℝ) : ℝ := a * x + b * y

-- Theorem for part 1
theorem part1 (a b : ℝ) : 
  F a b (-1) 3 = 2 ∧ F a b 1 (-2) = 8 → a = 28 ∧ b = 10 := by sorry

-- Theorem for part 2
theorem part2 (a b : ℝ) :
  b ≥ 0 ∧ F a b 2 1 = 5 → a ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2146_214659


namespace NUMINAMATH_CALUDE_three_sets_sum_18_with_6_l2146_214681

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem three_sets_sum_18_with_6 :
  (Finset.filter (fun s : Finset ℕ => 
    s.card = 3 ∧ 
    s ⊆ S ∧ 
    6 ∈ s ∧ 
    s.sum id = 18
  ) (S.powerset)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_sets_sum_18_with_6_l2146_214681


namespace NUMINAMATH_CALUDE_initial_brownies_count_l2146_214635

/-- Represents the number of days in a week -/
def week : ℕ := 7

/-- Represents the number of cookies eaten per day -/
def cookiesPerDay : ℕ := 3

/-- Represents the number of brownies eaten per day -/
def browniesPerDay : ℕ := 3

/-- Represents the difference between cookies and brownies after a week -/
def cookieBrownieDifference : ℕ := 36

/-- 
Theorem: If a person eats 3 cookies and 3 brownies per day for a week, 
and ends up with 36 more cookies than brownies, 
then they must have started with 36 brownies.
-/
theorem initial_brownies_count 
  (initialCookies initialBrownies : ℕ) : 
  initialCookies - week * cookiesPerDay = initialBrownies - week * browniesPerDay + cookieBrownieDifference →
  initialBrownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_brownies_count_l2146_214635


namespace NUMINAMATH_CALUDE_max_NF_value_slope_AN_l2146_214609

-- Define the ellipse parameters
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
axiom major_minor_ratio : 2*a = (3*Real.sqrt 5 / 5) * (2*b)
axiom point_D_on_ellipse : ellipse a b (-1) (2*Real.sqrt 10 / 3)
axiom vector_relation (x₀ y₀ x₁ y₁ : ℝ) (hx₀ : x₀ > 0) (hy₀ : y₀ > 0) :
  ellipse a b x₀ y₀ → ellipse a b x₁ y₁ → (x₀, y₀) = 2 * (x₁ + 3, y₁)

-- State the theorems to be proved
theorem max_NF_value :
  ∃ (c : ℝ), c^2 = a^2 - b^2 ∧ a + c = 5 :=
sorry

theorem slope_AN :
  ∃ (x₀ y₀ : ℝ), ellipse a b x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0 ∧ y₀ / x₀ = 5 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_NF_value_slope_AN_l2146_214609


namespace NUMINAMATH_CALUDE_video_game_enemies_l2146_214600

/-- The number of points earned per enemy defeated -/
def points_per_enemy : ℕ := 5

/-- The number of enemies left undefeated -/
def enemies_left : ℕ := 6

/-- The total points earned when all but 6 enemies are defeated -/
def total_points : ℕ := 10

/-- The total number of enemies in the level -/
def total_enemies : ℕ := 8

theorem video_game_enemies :
  total_enemies = (total_points / points_per_enemy) + enemies_left := by
  sorry

end NUMINAMATH_CALUDE_video_game_enemies_l2146_214600


namespace NUMINAMATH_CALUDE_maci_blue_pens_l2146_214631

/-- The number of blue pens Maci needs -/
def num_blue_pens : ℕ := sorry

/-- The number of red pens Maci needs -/
def num_red_pens : ℕ := 15

/-- The cost of a blue pen in cents -/
def blue_pen_cost : ℕ := 10

/-- The cost of a red pen in cents -/
def red_pen_cost : ℕ := 2 * blue_pen_cost

/-- The total cost of all pens in cents -/
def total_cost : ℕ := 400

theorem maci_blue_pens :
  num_blue_pens * blue_pen_cost + num_red_pens * red_pen_cost = total_cost ∧
  num_blue_pens = 10 :=
sorry

end NUMINAMATH_CALUDE_maci_blue_pens_l2146_214631


namespace NUMINAMATH_CALUDE_average_fruits_per_basket_l2146_214605

-- Define the number of baskets
def num_baskets : ℕ := 5

-- Define the number of fruits in each basket
def basket_A : ℕ := 15
def basket_B : ℕ := 30
def basket_C : ℕ := 20
def basket_D : ℕ := 25
def basket_E : ℕ := 35

-- Define the total number of fruits
def total_fruits : ℕ := basket_A + basket_B + basket_C + basket_D + basket_E

-- Theorem: The average number of fruits per basket is 25
theorem average_fruits_per_basket : 
  total_fruits / num_baskets = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_fruits_per_basket_l2146_214605


namespace NUMINAMATH_CALUDE_horner_v4_value_l2146_214668

def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc c => horner_step acc c x) 0

theorem horner_v4_value :
  let coeffs := [1, 0, 1, 0, 0, -2, 3]
  let x := 2
  let v0 := 3
  let v1 := horner_step v0 (-2) x
  let v2 := horner_step v1 0 x
  let v3 := horner_step v2 1 x
  let v4 := horner_step v3 0 x
  v4 = 34 ∧ horner_method coeffs x = f x := by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l2146_214668


namespace NUMINAMATH_CALUDE_square_perimeter_l2146_214628

/-- The perimeter of a square is 160 cm, given that its area is five times
    the area of a rectangle with dimensions 32 cm * 10 cm. -/
theorem square_perimeter (square_area rectangle_area : ℝ) : 
  square_area = 5 * rectangle_area →
  rectangle_area = 32 * 10 →
  4 * Real.sqrt square_area = 160 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2146_214628


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2146_214679

/-- Given that the solution set of ax^2 + bx - 1 > 0 is {x | -1/2 < x < -1/3},
    prove that the solution set of x^2 - bx - a ≥ 0 is {x | x ≤ -3 or x ≥ -2} -/
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 + b*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) →
  (∀ x, x^2 - b*x - a ≥ 0 ↔ x ≤ -3 ∨ x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2146_214679


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_one_two_left_closed_l2146_214672

/-- The set M of real numbers x such that (x + 3)(x - 2) < 0 -/
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}

/-- The set N of real numbers x such that 1 ≤ x ≤ 3 -/
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

/-- The theorem stating that the intersection of M and N is equal to the interval [1, 2) -/
theorem M_intersect_N_eq_one_two_left_closed :
  M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_one_two_left_closed_l2146_214672


namespace NUMINAMATH_CALUDE_digit_81_of_325_over_999_l2146_214655

theorem digit_81_of_325_over_999 (n : ℕ) (h : n = 81) :
  (325 : ℚ) / 999 * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_81_of_325_over_999_l2146_214655


namespace NUMINAMATH_CALUDE_probability_even_sum_l2146_214614

def set_a : Finset ℕ := {11, 44, 55}
def set_b : Finset ℕ := {1}

def is_sum_even (x : ℕ) (y : ℕ) : Bool :=
  (x + y) % 2 = 0

def count_even_sums : ℕ :=
  (set_a.filter (λ x => is_sum_even x 1)).card

theorem probability_even_sum :
  (count_even_sums : ℚ) / (set_a.card : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l2146_214614
