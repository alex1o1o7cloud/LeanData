import Mathlib

namespace third_square_is_G_l1760_176049

-- Define the type for squares
inductive Square | A | B | C | D | E | F | G | H

-- Define the placement order
def placement_order : List Square := [Square.F, Square.H, Square.G, Square.D, Square.A, Square.B, Square.C, Square.E]

-- Define the property of being fully visible
def is_fully_visible (s : Square) : Prop := s = Square.E

-- Define the property of being partially visible
def is_partially_visible (s : Square) : Prop := s ≠ Square.E

-- Define the property of being the last placed square
def is_last_placed (s : Square) : Prop := s = Square.E

-- Theorem statement
theorem third_square_is_G :
  (∀ s : Square, s ∈ placement_order) →
  (List.length placement_order = 8) →
  (∀ s : Square, s ≠ Square.E → is_partially_visible s) →
  is_fully_visible Square.E →
  is_last_placed Square.E →
  placement_order[2] = Square.G :=
by sorry

end third_square_is_G_l1760_176049


namespace prob_two_pairs_eq_nine_twentytwo_l1760_176097

/-- Represents the number of socks of each color --/
def socks_per_color : ℕ := 3

/-- Represents the number of colors --/
def num_colors : ℕ := 4

/-- Represents the total number of socks --/
def total_socks : ℕ := socks_per_color * num_colors

/-- Represents the number of socks drawn --/
def socks_drawn : ℕ := 5

/-- Calculates the probability of drawing exactly two pairs of socks with different colors --/
def prob_two_pairs : ℚ :=
  (num_colors.choose 2 * (num_colors - 2).choose 1 * socks_per_color.choose 2 * socks_per_color.choose 2 * socks_per_color.choose 1) /
  (total_socks.choose socks_drawn)

theorem prob_two_pairs_eq_nine_twentytwo : prob_two_pairs = 9 / 22 := by
  sorry

end prob_two_pairs_eq_nine_twentytwo_l1760_176097


namespace max_m_inequality_l1760_176045

theorem max_m_inequality (m : ℝ) : 
  (∀ a b : ℝ, (a / Real.exp a - b)^2 ≥ m - (a - b + 3)^2) → m ≤ 9/2 :=
by sorry

end max_m_inequality_l1760_176045


namespace train_length_l1760_176004

/-- Proves that a train passing through a tunnel under specific conditions has a length of 100 meters -/
theorem train_length (tunnel_length : ℝ) (total_time : ℝ) (inside_time : ℝ) 
  (h1 : tunnel_length = 500)
  (h2 : total_time = 30)
  (h3 : inside_time = 20)
  (h4 : total_time > 0)
  (h5 : inside_time > 0)
  (h6 : total_time > inside_time) :
  ∃ (train_length : ℝ), 
    train_length = 100 ∧ 
    (tunnel_length + train_length) / total_time = (tunnel_length - train_length) / inside_time :=
by sorry


end train_length_l1760_176004


namespace calculation_proof_l1760_176070

theorem calculation_proof :
  (4 + (-2)^3 * 5 - (-0.28) / 4 = -35.93) ∧
  (-1^4 - 1/6 * (2 - (-3)^2) = 1/6) := by
  sorry

end calculation_proof_l1760_176070


namespace cycle_selling_price_l1760_176076

/-- Calculates the final selling price of a cycle given initial cost, profit percentage, discount percentage, and sales tax percentage. -/
def finalSellingPrice (costPrice : ℚ) (profitPercentage : ℚ) (discountPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let markedPrice := costPrice * (1 + profitPercentage / 100)
  let discountedPrice := markedPrice * (1 - discountPercentage / 100)
  discountedPrice * (1 + salesTaxPercentage / 100)

/-- Theorem stating that the final selling price of the cycle is 936.32 given the specified conditions. -/
theorem cycle_selling_price :
  finalSellingPrice 800 10 5 12 = 936.32 := by
  sorry


end cycle_selling_price_l1760_176076


namespace sum_of_digits_of_sum_f_equals_8064_l1760_176008

-- Define the function f
def f (k : ℕ) : ℕ :=
  -- The smallest positive integer not written on the blackboard
  -- after the process described in the problem
  sorry

-- Define the sum of f(2k) from k=1 to 1008
def sum_f : ℕ :=
  (Finset.range 1008).sum (λ k => f (2 * (k + 1)))

-- Define a function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

-- The main theorem
theorem sum_of_digits_of_sum_f_equals_8064 :
  sum_of_digits sum_f = 8064 :=
sorry

end sum_of_digits_of_sum_f_equals_8064_l1760_176008


namespace parabola_a_value_l1760_176079

/-- A parabola with equation y = ax^2 + bx + c, vertex at (3, -2), and passing through (0, -50) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_condition : vertex_y = a * vertex_x^2 + b * vertex_x + c
  point_condition : point_y = a * point_x^2 + b * point_x + c

/-- The theorem stating that the value of 'a' for the given parabola is -16/3 -/
theorem parabola_a_value (p : Parabola) 
  (h1 : p.vertex_x = 3) 
  (h2 : p.vertex_y = -2) 
  (h3 : p.point_x = 0) 
  (h4 : p.point_y = -50) : 
  p.a = -16/3 := by
  sorry

end parabola_a_value_l1760_176079


namespace reflected_ray_equation_l1760_176030

-- Define the points and line
def M : ℝ × ℝ := (-3, 4)
def N : ℝ × ℝ := (2, 6)
def l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the reflection property
def is_reflection (M N M' : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), M' = (a, b) ∧
  (b - M.2) / (a - M.1) = -1 ∧
  (M.1 + a) / 2 - (b + M.2) / 2 + 3 = 0

-- Define the property of a line passing through two points
def line_through_points (P Q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - P.2) * (Q.1 - P.1) = (x - P.1) * (Q.2 - P.2)

-- Theorem statement
theorem reflected_ray_equation 
  (h_reflection : ∃ M' : ℝ × ℝ, is_reflection M N M' l) :
  ∀ x y : ℝ, line_through_points M' N x y ↔ 6 * x - y - 6 = 0 :=
sorry

end reflected_ray_equation_l1760_176030


namespace school_student_count_miyoung_school_students_l1760_176098

theorem school_student_count (grades classes_per_grade : ℕ) 
  (rank_from_front rank_from_back : ℕ) : ℕ :=
  let students_per_class := rank_from_front + rank_from_back - 1
  let students_per_grade := classes_per_grade * students_per_class
  let total_students := grades * students_per_grade
  total_students

theorem miyoung_school_students : 
  school_student_count 3 12 12 12 = 828 := by
  sorry

end school_student_count_miyoung_school_students_l1760_176098


namespace third_angle_of_triangle_l1760_176006

theorem third_angle_of_triangle (a b c : ℝ) : 
  a + b + c = 180 → a = 50 → b = 80 → c = 50 := by sorry

end third_angle_of_triangle_l1760_176006


namespace roses_kept_l1760_176035

theorem roses_kept (total : ℕ) (to_mother : ℕ) (to_grandmother : ℕ) (to_sister : ℕ) 
  (h1 : total = 20)
  (h2 : to_mother = 6)
  (h3 : to_grandmother = 9)
  (h4 : to_sister = 4) : 
  total - (to_mother + to_grandmother + to_sister) = 1 := by
  sorry

end roses_kept_l1760_176035


namespace z_percentage_of_1000_l1760_176022

theorem z_percentage_of_1000 (x y z : ℝ) : 
  x = (3/5) * 4864 →
  y = (2/3) * 9720 →
  z = (1/4) * 800 →
  (z / 1000) * 100 = 20 :=
by sorry

end z_percentage_of_1000_l1760_176022


namespace greatest_integer_less_than_N_div_100_l1760_176065

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def sum_of_fractions : ℚ :=
  1 / (factorial 2 * factorial 17) +
  1 / (factorial 3 * factorial 16) +
  1 / (factorial 4 * factorial 15) +
  1 / (factorial 5 * factorial 14) +
  1 / (factorial 6 * factorial 13) +
  1 / (factorial 7 * factorial 12) +
  1 / (factorial 8 * factorial 11) +
  1 / (factorial 9 * factorial 10)

def N : ℚ := sum_of_fractions * factorial 18

theorem greatest_integer_less_than_N_div_100 :
  ⌊N / 100⌋ = 137 :=
sorry

end greatest_integer_less_than_N_div_100_l1760_176065


namespace conditions_on_m_l1760_176002

/-- The set A defined by the quadratic equation mx² - 2x + 1 = 0 -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 - 2 * x + 1 = 0}

/-- Theorem stating the conditions on m for different properties of set A -/
theorem conditions_on_m :
  (∀ m : ℝ, A m = ∅ ↔ m > 1) ∧
  (∀ m : ℝ, (∃ x : ℝ, A m = {x}) ↔ m = 0 ∨ m = 1) ∧
  (∀ m : ℝ, (∃ x : ℝ, x ∈ A m ∧ x > 1/2 ∧ x < 2) ↔ m > 0 ∧ m ≤ 1) :=
by sorry

end conditions_on_m_l1760_176002


namespace jesses_friends_bananas_l1760_176057

/-- The total number of bananas given the number of friends and bananas per friend -/
def total_bananas (num_friends : ℝ) (bananas_per_friend : ℝ) : ℝ :=
  num_friends * bananas_per_friend

/-- Theorem: Jesse's friends have 63.0 bananas in total -/
theorem jesses_friends_bananas :
  total_bananas 3.0 21.0 = 63.0 := by
  sorry

end jesses_friends_bananas_l1760_176057


namespace fathers_age_l1760_176063

theorem fathers_age (son_age father_age : ℕ) : 
  father_age = 3 * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 8 →
  father_age = 27 := by
sorry

end fathers_age_l1760_176063


namespace p_sufficient_not_necessary_for_q_l1760_176000

-- Define the conditions
def p (x : ℝ) : Prop := Real.log (x - 3) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l1760_176000


namespace vases_to_arrange_l1760_176014

/-- Proves that the number of vases of flowers to be arranged is 256,
    given that Jane can arrange 16 vases per day and needs 16 days to finish all arrangements. -/
theorem vases_to_arrange (vases_per_day : ℕ) (days_needed : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days_needed = 16) : 
  vases_per_day * days_needed = 256 := by
  sorry

end vases_to_arrange_l1760_176014


namespace base_7_2534_equals_956_l1760_176033

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_2534_equals_956 :
  base_7_to_10 [4, 3, 5, 2] = 956 := by
  sorry

end base_7_2534_equals_956_l1760_176033


namespace average_of_remaining_numbers_l1760_176099

/-- Given 6 numbers with specified averages, prove the average of the remaining 2 numbers -/
theorem average_of_remaining_numbers
  (total_average : Real)
  (first_pair_average : Real)
  (second_pair_average : Real)
  (h1 : total_average = 3.95)
  (h2 : first_pair_average = 3.8)
  (h3 : second_pair_average = 3.85) :
  (6 * total_average - 2 * first_pair_average - 2 * second_pair_average) / 2 = 4.2 := by
  sorry

end average_of_remaining_numbers_l1760_176099


namespace cube_sum_from_sum_and_square_sum_l1760_176078

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
  sorry

end cube_sum_from_sum_and_square_sum_l1760_176078


namespace all_points_on_single_line_l1760_176021

/-- A point in a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line. -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear. -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, pointOnLine p l ∧ pointOnLine q l ∧ pointOnLine r l

/-- The main theorem. -/
theorem all_points_on_single_line (k : ℕ) (points : Fin k → Point)
    (h : ∀ i j : Fin k, i ≠ j → ∃ m : Fin k, m ≠ i ∧ m ≠ j ∧ collinear (points i) (points j) (points m)) :
    ∃ l : Line, ∀ i : Fin k, pointOnLine (points i) l := by
  sorry

end all_points_on_single_line_l1760_176021


namespace stratified_sampling_management_l1760_176064

theorem stratified_sampling_management (total_employees : ℕ) (management : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 150)
  (h2 : management = 15)
  (h3 : sample_size = 30) :
  (management * sample_size) / total_employees = 3 :=
by
  sorry

end stratified_sampling_management_l1760_176064


namespace m_range_isosceles_perimeter_l1760_176056

-- Define the triangle ABC
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the specific triangle from the problem
def triangleABC (m : ℝ) : Triangle where
  AB := 17
  BC := 8
  AC := 2 * m - 1

-- Theorem for the range of m
theorem m_range (m : ℝ) : 
  (∃ t : Triangle, t = triangleABC m ∧ t.AB + t.BC > t.AC ∧ t.AB + t.AC > t.BC ∧ t.AC + t.BC > t.AB) 
  ↔ (5 < m ∧ m < 13) := by sorry

-- Theorem for the perimeter when isosceles
theorem isosceles_perimeter (m : ℝ) :
  (∃ t : Triangle, t = triangleABC m ∧ (t.AB = t.AC ∨ t.AB = t.BC ∨ t.AC = t.BC)) →
  (∃ t : Triangle, t = triangleABC m ∧ t.AB + t.BC + t.AC = 42) := by sorry

end m_range_isosceles_perimeter_l1760_176056


namespace coloring_exists_l1760_176036

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2010}

theorem coloring_exists :
  ∃ (f : ℕ → Fin 5),
    ∀ (a d : ℕ),
      a ∈ M →
      d > 0 →
      (∀ k, k ∈ Finset.range 9 → (a + k * d) ∈ M) →
      ∃ (i j : Fin 9), i ≠ j ∧ f (a + i * d) ≠ f (a + j * d) :=
by sorry

end coloring_exists_l1760_176036


namespace gcd_90_150_l1760_176082

theorem gcd_90_150 : Nat.gcd 90 150 = 30 := by
  sorry

end gcd_90_150_l1760_176082


namespace cycle_reappearance_l1760_176044

theorem cycle_reappearance (letter_seq_length digit_seq_length : ℕ) 
  (h1 : letter_seq_length = 9)
  (h2 : digit_seq_length = 4) : 
  Nat.lcm letter_seq_length digit_seq_length = 36 := by
  sorry

end cycle_reappearance_l1760_176044


namespace simplify_expression_l1760_176037

theorem simplify_expression (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end simplify_expression_l1760_176037


namespace five_Y_three_equals_four_l1760_176058

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_four : Y 5 3 = 4 := by
  sorry

end five_Y_three_equals_four_l1760_176058


namespace find_z_l1760_176095

theorem find_z (M N : Set ℂ) (i : ℂ) (z : ℂ) : 
  M = {1, 2, z * i} →
  N = {3, 4} →
  M ∩ N = {4} →
  i * i = -1 →
  z = 4 * i :=
by sorry

end find_z_l1760_176095


namespace university_diploma_percentage_l1760_176024

theorem university_diploma_percentage
  (no_diploma_with_job : Real)
  (diploma_without_job : Real)
  (job_of_choice : Real)
  (h1 : no_diploma_with_job = 0.18)
  (h2 : diploma_without_job = 0.25)
  (h3 : job_of_choice = 0.4) :
  (job_of_choice - no_diploma_with_job) + (diploma_without_job * (1 - job_of_choice)) = 0.37 := by
sorry

end university_diploma_percentage_l1760_176024


namespace xiaomin_final_score_l1760_176081

/-- Calculates the final score for the "Book-loving Youth" selection -/
def final_score (honor_score : ℝ) (speech_score : ℝ) : ℝ :=
  0.4 * honor_score + 0.6 * speech_score

/-- Theorem: Xiaomin's final score in the "Book-loving Youth" selection is 86 points -/
theorem xiaomin_final_score :
  let honor_score := 80
  let speech_score := 90
  final_score honor_score speech_score = 86 :=
by
  sorry

end xiaomin_final_score_l1760_176081


namespace brother_age_in_five_years_l1760_176026

/-- Given the ages of Nick and his siblings, prove the brother's age in 5 years -/
theorem brother_age_in_five_years
  (nick_age : ℕ)
  (sister_age_diff : ℕ)
  (h_nick_age : nick_age = 13)
  (h_sister_age_diff : sister_age_diff = 6)
  (brother_age : ℕ)
  (h_brother_age : brother_age = (nick_age + (nick_age + sister_age_diff)) / 2) :
  brother_age + 5 = 21 := by
sorry


end brother_age_in_five_years_l1760_176026


namespace jackson_pbj_sandwiches_l1760_176061

/-- Calculates the number of peanut butter and jelly sandwiches Jackson eats during the school year -/
def pbj_sandwiches (weeks : ℕ) (wed_holidays : ℕ) (fri_holidays : ℕ) (ham_cheese_interval : ℕ) (wed_absences : ℕ) (fri_absences : ℕ) : ℕ :=
  let total_wed := weeks
  let total_fri := weeks
  let remaining_wed := total_wed - wed_holidays - wed_absences
  let remaining_fri := total_fri - fri_holidays - fri_absences
  let ham_cheese_weeks := weeks / ham_cheese_interval
  let ham_cheese_wed := ham_cheese_weeks
  let ham_cheese_fri := ham_cheese_weeks * 2
  let pbj_wed := remaining_wed - ham_cheese_wed
  let pbj_fri := remaining_fri - ham_cheese_fri
  pbj_wed + pbj_fri

/-- Theorem stating that Jackson eats 37 peanut butter and jelly sandwiches during the school year -/
theorem jackson_pbj_sandwiches :
  pbj_sandwiches 36 2 3 4 1 2 = 37 := by
  sorry

end jackson_pbj_sandwiches_l1760_176061


namespace estimated_area_is_10_l1760_176009

/-- The function representing the lower bound of the area -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The upper bound of the area -/
def upper_bound : ℝ := 5

/-- The total square area -/
def total_area : ℝ := 16

/-- The total number of experiments -/
def total_experiments : ℕ := 1000

/-- The number of points that fall within the desired area -/
def points_within : ℕ := 625

/-- Theorem stating that the estimated area is 10 -/
theorem estimated_area_is_10 : 
  (total_area * (points_within : ℝ) / total_experiments) = 10 := by
  sorry

end estimated_area_is_10_l1760_176009


namespace correct_average_calculation_l1760_176019

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 20 →
  incorrect_num = 26 →
  correct_num = 86 →
  (n : ℚ) * incorrect_avg - incorrect_num + correct_num = n * 26 := by
  sorry

end correct_average_calculation_l1760_176019


namespace remainder_sum_l1760_176055

theorem remainder_sum (x y : ℤ) (hx : x % 90 = 75) (hy : y % 120 = 115) :
  (x + y) % 30 = 10 := by
  sorry

end remainder_sum_l1760_176055


namespace stewart_farm_sheep_count_l1760_176039

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (horse_food_per_day total_food : ℝ),
    sheep * 7 = horses * 6 →
    horse_food_per_day = 230 →
    total_food = 12880 →
    horses * horse_food_per_day = total_food →
    sheep = 48 := by
  sorry

end stewart_farm_sheep_count_l1760_176039


namespace function_shift_l1760_176048

theorem function_shift (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) = x^2 - 2*x - 3) →
  (∀ x : ℝ, f x = x^2 - 4*x) := by
  sorry

end function_shift_l1760_176048


namespace plot_width_calculation_l1760_176011

/-- Calculates the width of a rectangular plot given its length and fence specifications. -/
theorem plot_width_calculation (length width : ℝ) (num_poles : ℕ) (pole_distance : ℝ) : 
  length = 90 ∧ 
  num_poles = 28 ∧ 
  pole_distance = 10 ∧ 
  (num_poles - 1) * pole_distance = 2 * (length + width) →
  width = 45 := by
sorry

end plot_width_calculation_l1760_176011


namespace intersection_difference_l1760_176031

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 3 * x + 5

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | parabola1 x = parabola2 x}

-- Theorem statement
theorem intersection_difference : 
  ∃ (p r : ℝ), p ∈ intersection_points ∧ r ∈ intersection_points ∧ 
  r ≥ p ∧ ∀ x ∈ intersection_points, (x = p ∨ x = r) ∧ 
  r - p = 3/5 := by
  sorry

end intersection_difference_l1760_176031


namespace smaller_two_digit_factor_l1760_176015

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 5488 → 
  min a b = 56 := by
sorry

end smaller_two_digit_factor_l1760_176015


namespace magical_stack_size_with_157_fixed_l1760_176093

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (pile_a : Finset ℕ := Finset.range n)
  (pile_b : Finset ℕ := Finset.range n)
  (card_157_position : ℕ)
  (card_157_retains_position : card_157_position = 157)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem: The number of cards in a magical stack where card 157 retains its position is 470 --/
theorem magical_stack_size_with_157_fixed (stack : MagicalStack) :
  magical_stack_size stack = 470 := by sorry

end magical_stack_size_with_157_fixed_l1760_176093


namespace inner_rectangle_side_length_l1760_176013

/-- Given a square with side length a and four congruent right triangles removed from its corners,
    this theorem proves the relationship between the original square's side length,
    the area removed, and the resulting inner rectangle's side length. -/
theorem inner_rectangle_side_length
  (a : ℝ)
  (h1 : a ≥ 24 * Real.sqrt 3)
  (h2 : 6 * (4 * Real.sqrt 3)^2 = 288) :
  a - 24 * Real.sqrt 3 = a - 6 * (4 * Real.sqrt 3) :=
by sorry

end inner_rectangle_side_length_l1760_176013


namespace pencil_cost_solution_l1760_176046

/-- The cost of Daniel's purchase -/
def purchase_problem (magazine_cost pencil_cost coupon_discount total_spent : ℚ) : Prop :=
  magazine_cost + pencil_cost - coupon_discount = total_spent

theorem pencil_cost_solution :
  ∃ (pencil_cost : ℚ),
    purchase_problem 0.85 pencil_cost 0.35 1 ∧ pencil_cost = 0.50 := by
  sorry

end pencil_cost_solution_l1760_176046


namespace divisibility_puzzle_l1760_176074

theorem divisibility_puzzle (a : ℤ) :
  (∃! n : Fin 4, ¬ ((n = 0 → a % 2 = 0) ∧
                    (n = 1 → a % 4 = 0) ∧
                    (n = 2 → a % 12 = 0) ∧
                    (n = 3 → a % 24 = 0))) →
  ¬ (a % 24 = 0) :=
by sorry

end divisibility_puzzle_l1760_176074


namespace square_of_sum_equality_l1760_176032

theorem square_of_sum_equality : 31^2 + 2*(31)*(5 + 3) + (5 + 3)^2 = 1521 := by
  sorry

end square_of_sum_equality_l1760_176032


namespace intersection_sum_l1760_176066

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y + 1)^2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 4 :=
by
  sorry

end intersection_sum_l1760_176066


namespace work_completion_proof_l1760_176080

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 15

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 20

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.41666666666666663

/-- The number of days A and B worked together -/
def days_worked_together : ℝ := 5

theorem work_completion_proof :
  let work_rate_a := 1 / a_days
  let work_rate_b := 1 / b_days
  let combined_rate := work_rate_a + work_rate_b
  combined_rate * days_worked_together = 1 - work_left :=
by sorry

end work_completion_proof_l1760_176080


namespace tour_group_size_tour_group_size_exists_l1760_176088

/-- Represents the possible solutions for the number of people in the tour group -/
inductive TourGroupSize
  | eight : TourGroupSize
  | thirteen : TourGroupSize

/-- Checks if a given number of adults and children satisfies the ticket price conditions -/
def validTicketCombination (adults : ℕ) (children : ℕ) : Prop :=
  8 * adults + 3 * children = 44

/-- Calculates the total number of people in the tour group -/
def groupSize (adults : ℕ) (children : ℕ) : ℕ :=
  adults + children

/-- Theorem stating that the only valid tour group sizes are 8 or 13 -/
theorem tour_group_size :
  ∀ (adults children : ℕ),
    validTicketCombination adults children →
    (groupSize adults children = 8 ∨ groupSize adults children = 13) :=
by sorry

/-- Theorem stating that both 8 and 13 are possible tour group sizes -/
theorem tour_group_size_exists :
  (∃ (adults children : ℕ), validTicketCombination adults children ∧ groupSize adults children = 8) ∧
  (∃ (adults children : ℕ), validTicketCombination adults children ∧ groupSize adults children = 13) :=
by sorry

end tour_group_size_tour_group_size_exists_l1760_176088


namespace problem_solution_l1760_176005

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a*x

theorem problem_solution (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  sorry

end problem_solution_l1760_176005


namespace latus_rectum_of_parabola_l1760_176034

/-- Given a parabola with equation y² = 4x, its latus rectum has the equation x = -1 -/
theorem latus_rectum_of_parabola :
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (x₀ : ℝ), x₀ = -1 ∧ ∀ (y₀ : ℝ), (y₀^2 = 4*x₀ → x₀ = -1)) :=
by sorry

end latus_rectum_of_parabola_l1760_176034


namespace cake_mix_buyers_l1760_176072

/-- Proof of the number of buyers purchasing cake mix -/
theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) 
  (h1 : total = 100)
  (h2 : muffin = 40)
  (h3 : both = 17)
  (h4 : neither_prob = 27 / 100) : 
  ∃ cake : ℕ, cake = 50 ∧ 
    cake + muffin - both = total - (neither_prob * total).num := by
  sorry

end cake_mix_buyers_l1760_176072


namespace simplify_sqrt_expression_l1760_176041

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7 * x^6 + 1)) / (3 * x^3) :=
sorry

end simplify_sqrt_expression_l1760_176041


namespace placement_count_no_restriction_placement_count_with_restriction_l1760_176012

/-- The number of booths in the exhibition room -/
def total_booths : ℕ := 9

/-- The number of exhibits to be displayed -/
def num_exhibits : ℕ := 3

/-- Calculates the number of ways to place exhibits under the given conditions -/
def calculate_placements (max_distance : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of placement options without distance restriction -/
theorem placement_count_no_restriction : calculate_placements total_booths = 60 :=
  sorry

/-- Theorem stating the number of placement options with distance restriction -/
theorem placement_count_with_restriction : calculate_placements 2 = 48 :=
  sorry

end placement_count_no_restriction_placement_count_with_restriction_l1760_176012


namespace z_share_per_x_rupee_l1760_176018

/-- Proof that z gets 0.50 rupees for each rupee x gets --/
theorem z_share_per_x_rupee
  (total : ℝ)
  (y_share : ℝ)
  (y_per_x : ℝ)
  (h_total : total = 156)
  (h_y_share : y_share = 36)
  (h_y_per_x : y_per_x = 0.45)
  : ∃ (z_per_x : ℝ), z_per_x = 0.50 ∧
    ∃ (units : ℝ), units * (1 + y_per_x + z_per_x) = total ∧
                   units * y_per_x = y_share :=
by
  sorry


end z_share_per_x_rupee_l1760_176018


namespace three_integers_sum_l1760_176016

theorem three_integers_sum (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 →
  a * b * c = 216000 →
  Nat.gcd a b = 1 → Nat.gcd a c = 1 → Nat.gcd b c = 1 →
  a + b + c = 184 :=
by sorry

end three_integers_sum_l1760_176016


namespace diagonals_25_sided_polygon_l1760_176059

/-- The number of diagonals in a convex polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 25 sides is 275 -/
theorem diagonals_25_sided_polygon :
  numDiagonals 25 = 275 := by
  sorry

end diagonals_25_sided_polygon_l1760_176059


namespace root_zero_implies_a_half_l1760_176003

theorem root_zero_implies_a_half (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + 2*a - 1 = 0 ∧ x = 0) → a = 1/2 := by
  sorry

end root_zero_implies_a_half_l1760_176003


namespace counting_game_result_l1760_176020

/-- Represents the counting game with students in a circle. -/
def CountingGame (n : ℕ) (start : ℕ) (last : ℕ) : Prop :=
  ∃ (process : ℕ → ℕ → ℕ), 
    (process 0 start = last) ∧ 
    (∀ k, k > 0 → process k start ≠ last → 
      process (k+1) start = process k (((process k start + 2) % n) + 1))

/-- The main theorem stating that if student 37 is the last remaining
    in a circle of 40 students, then the initial student was number 5. -/
theorem counting_game_result : CountingGame 40 5 37 := by
  sorry


end counting_game_result_l1760_176020


namespace ara_final_height_theorem_l1760_176086

/-- Represents the growth and heights of Shea and Ara -/
structure HeightGrowth where
  initial_height : ℝ
  shea_growth_percent : ℝ
  ara_growth_fraction : ℝ
  shea_final_height : ℝ

/-- Calculates Ara's final height based on the given conditions -/
def calculate_ara_height (hg : HeightGrowth) : ℝ :=
  let shea_growth := hg.initial_height * hg.shea_growth_percent
  let ara_growth := shea_growth * hg.ara_growth_fraction
  hg.initial_height + ara_growth

/-- Theorem stating that Ara's final height is approximately 60.67 inches -/
theorem ara_final_height_theorem (hg : HeightGrowth) 
  (h1 : hg.initial_height > 0)
  (h2 : hg.shea_growth_percent = 0.25)
  (h3 : hg.ara_growth_fraction = 1/3)
  (h4 : hg.shea_final_height = 70) :
  ∃ ε > 0, |calculate_ara_height hg - 60.67| < ε :=
sorry

end ara_final_height_theorem_l1760_176086


namespace least_positive_integer_with_remainders_l1760_176010

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 3 = 0 ∧ m % 4 = 1 ∧ m % 5 = 2 → n ≤ m :=
by
  use 57
  sorry

end least_positive_integer_with_remainders_l1760_176010


namespace sandy_net_spent_l1760_176062

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_return : ℚ := 7.43

theorem sandy_net_spent (shorts_cost shirt_cost jacket_return : ℚ) :
  shorts_cost = 13.99 →
  shirt_cost = 12.14 →
  jacket_return = 7.43 →
  shorts_cost + shirt_cost - jacket_return = 18.70 :=
by sorry

end sandy_net_spent_l1760_176062


namespace divisible_by_6_up_to_88_characterization_l1760_176094

def divisible_by_6_up_to_88 : Set ℕ :=
  {n : ℕ | 1 < n ∧ n ≤ 88 ∧ n % 6 = 0}

theorem divisible_by_6_up_to_88_characterization :
  divisible_by_6_up_to_88 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84} := by
  sorry

end divisible_by_6_up_to_88_characterization_l1760_176094


namespace karen_beats_tom_l1760_176068

theorem karen_beats_tom (karen_speed : ℝ) (tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  karen_delay = 4 / 60 →
  winning_margin = 4 →
  (tom_speed * (karen_delay + (winning_margin + tom_speed * karen_delay) / (karen_speed - tom_speed))) = 21 :=
by sorry

end karen_beats_tom_l1760_176068


namespace recycling_drive_target_l1760_176043

/-- Calculates the target amount of kilos for a recycling drive given the number of sections,
    amount collected per section in two weeks, and additional amount needed. -/
def recycling_target (sections : ℕ) (kilos_per_section_two_weeks : ℕ) (additional_kilos : ℕ) : ℕ :=
  let kilos_per_section_per_week := kilos_per_section_two_weeks / 2
  let kilos_per_section_three_weeks := kilos_per_section_per_week * 3
  let total_collected := kilos_per_section_three_weeks * sections
  total_collected + additional_kilos

/-- The recycling drive target matches the calculated amount. -/
theorem recycling_drive_target :
  recycling_target 6 280 320 = 2840 := by
  sorry

end recycling_drive_target_l1760_176043


namespace inequality_sum_l1760_176054

theorem inequality_sum (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : x < a) (h2 : y < b) : x + y < a + b := by
  sorry

end inequality_sum_l1760_176054


namespace integral_proofs_l1760_176090

theorem integral_proofs :
  ∀ x : ℝ,
    (deriv (λ y => Real.arctan (Real.log y)) x = 1 / (x * (1 + Real.log x ^ 2))) ∧
    (deriv (λ y => Real.arctan (Real.exp y)) x = Real.exp x / (1 + Real.exp (2 * x))) ∧
    (deriv (λ y => Real.arctan (Real.sin y)) x = Real.cos x / (1 + Real.sin x ^ 2)) :=
by sorry

end integral_proofs_l1760_176090


namespace sequence_sum_l1760_176071

/-- Given an 8-term sequence where C = 10 and the sum of any three consecutive terms is 40,
    prove that A + H = 30 -/
theorem sequence_sum (A B C D E F G H : ℝ) 
  (hC : C = 10)
  (hABC : A + B + C = 40)
  (hBCD : B + C + D = 40)
  (hCDE : C + D + E = 40)
  (hDEF : D + E + F = 40)
  (hEFG : E + F + G = 40)
  (hFGH : F + G + H = 40) :
  A + H = 30 := by
  sorry

end sequence_sum_l1760_176071


namespace specific_l_shape_perimeter_l1760_176053

/-- An L-shaped figure formed by squares -/
structure LShapedFigure where
  squareSideLength : ℕ
  baseSquares : ℕ
  stackedSquares : ℕ

/-- Calculate the perimeter of an L-shaped figure -/
def perimeter (figure : LShapedFigure) : ℕ :=
  2 * figure.squareSideLength * (figure.baseSquares + figure.stackedSquares + 1)

/-- Theorem: The perimeter of the specific L-shaped figure is 14 units -/
theorem specific_l_shape_perimeter :
  let figure : LShapedFigure := ⟨2, 3, 2⟩
  perimeter figure = 14 := by
  sorry

end specific_l_shape_perimeter_l1760_176053


namespace x_squared_plus_reciprocal_l1760_176001

theorem x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^4 + 1/x^4 = 47 → x^2 + 1/x^2 = 7 := by sorry

end x_squared_plus_reciprocal_l1760_176001


namespace baker_remaining_cakes_l1760_176029

/-- Given a baker who made 167 cakes and sold 108 cakes, prove that the number of cakes remaining is 59. -/
theorem baker_remaining_cakes (cakes_made : ℕ) (cakes_sold : ℕ) 
  (h1 : cakes_made = 167) (h2 : cakes_sold = 108) : 
  cakes_made - cakes_sold = 59 := by
  sorry

#check baker_remaining_cakes

end baker_remaining_cakes_l1760_176029


namespace lisa_marble_difference_l1760_176028

/-- Proves that Lisa has 19 more marbles than Cindy after the marble exchange -/
theorem lisa_marble_difference (cindy_initial : ℕ) (lisa_initial : ℕ) (marbles_given : ℕ) : 
  cindy_initial = 20 →
  cindy_initial = lisa_initial + 5 →
  marbles_given = 12 →
  (lisa_initial + marbles_given) - (cindy_initial - marbles_given) = 19 :=
by
  sorry

end lisa_marble_difference_l1760_176028


namespace euler_totient_properties_l1760_176052

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Definition: p is prime -/
def is_prime (p : ℕ) : Prop := sorry

theorem euler_totient_properties (p : ℕ) (α : ℕ) (h : is_prime p) (h' : α > 0) :
  (phi 17 = 16) ∧
  (phi p = p - 1) ∧
  (phi (p^2) = p * (p - 1)) ∧
  (phi (p^α) = p^(α-1) * (p - 1)) :=
sorry

end euler_totient_properties_l1760_176052


namespace vote_count_theorem_l1760_176085

theorem vote_count_theorem (votes_A votes_B : ℕ) : 
  (votes_B = (20 * votes_A) / 21) →  -- B's votes are 20/21 of A's
  (votes_A > votes_B) →  -- A wins
  (votes_B + 4 > votes_A - 4) →  -- If B gains 4 votes, B would win
  (votes_A < 168) →  -- derived from the inequality in the solution
  (∀ (x : ℕ), x < votes_A → x ≠ votes_A ∨ (20 * x) / 21 ≠ votes_B) →  -- A's vote count is minimal
  ((votes_A = 147 ∧ votes_B = 140) ∨ (votes_A = 126 ∧ votes_B = 120)) :=
by
  sorry  -- Proof omitted

end vote_count_theorem_l1760_176085


namespace bicycle_tire_swap_theorem_l1760_176050

/-- Represents the characteristics and behavior of a bicycle with swappable tires -/
structure Bicycle where
  front_tire_life : ℝ
  rear_tire_life : ℝ

/-- Calculates the maximum distance a bicycle can travel with one tire swap -/
def max_distance_with_swap (b : Bicycle) : ℝ := sorry

/-- Calculates the optimal distance at which to swap tires -/
def optimal_swap_distance (b : Bicycle) : ℝ := sorry

/-- Theorem stating the maximum distance and optimal swap point for a specific bicycle -/
theorem bicycle_tire_swap_theorem (b : Bicycle) 
  (h1 : b.front_tire_life = 11000)
  (h2 : b.rear_tire_life = 9000) :
  max_distance_with_swap b = 9900 ∧ optimal_swap_distance b = 4950 := by sorry

end bicycle_tire_swap_theorem_l1760_176050


namespace sugar_profit_percentage_l1760_176042

theorem sugar_profit_percentage 
  (total_sugar : ℝ) 
  (sugar_at_18_percent : ℝ) 
  (overall_profit_percentage : ℝ) :
  total_sugar = 1000 →
  sugar_at_18_percent = 600 →
  overall_profit_percentage = 14 →
  ∃ (unknown_profit_percentage : ℝ),
    unknown_profit_percentage = 80 ∧
    sugar_at_18_percent * (18 / 100) + 
    (total_sugar - sugar_at_18_percent) * (unknown_profit_percentage / 100) = 
    total_sugar * (overall_profit_percentage / 100) :=
by
  sorry

end sugar_profit_percentage_l1760_176042


namespace point_B_coordinates_l1760_176025

-- Define the point type
structure Point := (x : ℝ) (y : ℝ)

-- Define the problem statement
theorem point_B_coordinates (A B : Point) (h1 : A.x = 1 ∧ A.y = -1) 
  (h2 : (B.x - A.x)^2 + (B.y - A.y)^2 = 3^2) 
  (h3 : B.x = A.x) : 
  (B = Point.mk 1 (-4) ∨ B = Point.mk 1 2) := by
  sorry


end point_B_coordinates_l1760_176025


namespace dubblefud_red_balls_l1760_176089

/-- The game of dubblefud with red, blue, and green balls -/
def dubblefud (r b g : ℕ) : Prop :=
  2^r * 4^b * 5^g = 16000 ∧ b = g

theorem dubblefud_red_balls :
  ∃ (r b g : ℕ), dubblefud r b g ∧ r = 6 :=
sorry

end dubblefud_red_balls_l1760_176089


namespace unique_prime_in_set_l1760_176027

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B : ℕ) : ℕ := 303200 + B

theorem unique_prime_in_set :
  ∃! B : ℕ, B ≤ 9 ∧ is_prime (six_digit_number B) :=
sorry

end unique_prime_in_set_l1760_176027


namespace n_minus_m_range_l1760_176007

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then Real.exp x - 1 else 3/2 * x + 1

theorem n_minus_m_range (m n : ℝ) (h1 : m < n) (h2 : f m = f n) : 
  2/3 < n - m ∧ n - m ≤ Real.log (3/2) + 1/3 := by sorry

end n_minus_m_range_l1760_176007


namespace curve_symmetry_condition_l1760_176040

/-- Given a curve y = x + p/x where p ≠ 0, this theorem states that the condition for two distinct
points on the curve to be symmetric with respect to the line y = x is satisfied if and only if p < 0 -/
theorem curve_symmetry_condition (p : ℝ) (hp : p ≠ 0) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   x₁ + p / x₁ = x₂ + p / x₂ ∧
   x₁ + p / x₁ + x₂ + p / x₂ = x₁ + x₂) ↔ 
  p < 0 := by
  sorry

end curve_symmetry_condition_l1760_176040


namespace max_value_of_expression_l1760_176092

theorem max_value_of_expression (t : ℝ) :
  (∃ (max : ℝ), max = (1 / 8) ∧
    ∀ (t : ℝ), ((3^t - 2*t^2)*t) / (9^t) ≤ max ∧
    ∃ (t_max : ℝ), ((3^t_max - 2*t_max^2)*t_max) / (9^t_max) = max) :=
by sorry

end max_value_of_expression_l1760_176092


namespace abrahams_a_students_l1760_176077

theorem abrahams_a_students (total_students : ℕ) (total_a_students : ℕ) (abraham_students : ℕ) :
  total_students = 40 →
  total_a_students = 25 →
  abraham_students = 10 →
  (abraham_students : ℚ) / total_students * total_a_students = (abraham_students : ℕ) →
  ∃ (abraham_a_students : ℕ), 
    (abraham_a_students : ℚ) / abraham_students = (total_a_students : ℚ) / total_students ∧
    abraham_a_students = 6 :=
by sorry

end abrahams_a_students_l1760_176077


namespace lcm_5_6_8_9_l1760_176083

theorem lcm_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end lcm_5_6_8_9_l1760_176083


namespace arithmetic_proof_l1760_176069

theorem arithmetic_proof : (1) - 2^3 / (-1/5) - 1/2 * (-4)^2 = 32 := by
  sorry

end arithmetic_proof_l1760_176069


namespace tea_consumption_l1760_176038

theorem tea_consumption (total : ℕ) (days : ℕ) (diff : ℕ) : 
  total = 120 → days = 6 → diff = 4 → 
  ∃ (first : ℕ), 
    (first + 3 * diff = 22) ∧ 
    (days * (2 * first + (days - 1) * diff) / 2 = total) := by
  sorry

end tea_consumption_l1760_176038


namespace bill_drew_eight_squares_l1760_176084

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := 4

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 88

/-- Theorem: Bill drew 8 squares -/
theorem bill_drew_eight_squares :
  ∃ (num_squares : ℕ),
    num_squares * square_sides + 
    num_triangles * triangle_sides + 
    num_pentagons * pentagon_sides = total_lines ∧
    num_squares = 8 := by
  sorry

end bill_drew_eight_squares_l1760_176084


namespace sum_and_product_equality_l1760_176017

theorem sum_and_product_equality : 2357 + 3572 + 5723 + 7235 * 2 = 26122 := by
  sorry

end sum_and_product_equality_l1760_176017


namespace nested_average_equals_two_thirds_l1760_176087

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_two_thirds :
  avg3 (avg3 1 2 0) (avg2 0 2) 0 = 2/3 := by sorry

end nested_average_equals_two_thirds_l1760_176087


namespace harry_seashells_count_l1760_176060

theorem harry_seashells_count :
  ∀ (seashells : ℕ),
    -- Initial collection
    34 + seashells + 29 = 34 + seashells + 29 →
    -- Total items lost
    25 = 25 →
    -- Items left at the end
    59 = 59 →
    -- Proof that seashells = 21
    seashells = 21 := by
  sorry

end harry_seashells_count_l1760_176060


namespace perpendicular_sum_equals_perimeter_l1760_176067

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define perpendicular points on angle bisectors
structure PerpendicularPoints (T : Triangle) :=
  (A1 A2 B1 B2 C1 C2 : ℝ × ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem perpendicular_sum_equals_perimeter (T : Triangle) (P : PerpendicularPoints T) :
  2 * (distance P.A1 P.A2 + distance P.B1 P.B2 + distance P.C1 P.C2) =
  distance T.A T.B + distance T.B T.C + distance T.C T.A := by sorry

end perpendicular_sum_equals_perimeter_l1760_176067


namespace smallest_multiple_of_45_with_0_and_8_l1760_176051

def is_multiple_of_45 (n : ℕ) : Prop := n % 45 = 0

def contains_only_0_and_8 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 8

theorem smallest_multiple_of_45_with_0_and_8 :
  ∃ (n : ℕ), is_multiple_of_45 n ∧ contains_only_0_and_8 n ∧
  (∀ m : ℕ, m < n → ¬(is_multiple_of_45 m ∧ contains_only_0_and_8 m)) ∧
  n = 8888888880 :=
sorry

end smallest_multiple_of_45_with_0_and_8_l1760_176051


namespace rectangular_prism_painted_faces_l1760_176091

theorem rectangular_prism_painted_faces (a : ℕ) : 
  2 < a → a < 5 → (a - 2) * 3 * 4 = 4 * 3 + 4 * 4 → a = 4 := by sorry

end rectangular_prism_painted_faces_l1760_176091


namespace gcd_30_problem_l1760_176075

theorem gcd_30_problem (n : ℕ) : 
  70 ≤ n ∧ n ≤ 90 → Nat.gcd 30 n = 10 → n = 70 ∨ n = 80 := by sorry

end gcd_30_problem_l1760_176075


namespace range_of_m_main_theorem_l1760_176096

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + m*p.1 + 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (A m ∩ B).Nonempty → m ∈ Set.Iic (-1) := by
  sorry

-- Define the range of m
def range_m : Set ℝ := {m : ℝ | ∃ x : ℝ, (A m ∩ B).Nonempty}

-- State the main theorem
theorem main_theorem : range_m = Set.Iic (-1) := by
  sorry

end range_of_m_main_theorem_l1760_176096


namespace martha_clothes_count_l1760_176047

def jacket_ratio : ℕ := 2
def tshirt_ratio : ℕ := 3
def jackets_bought : ℕ := 4
def tshirts_bought : ℕ := 9

def total_clothes : ℕ := 
  (jackets_bought + jackets_bought / jacket_ratio) + 
  (tshirts_bought + tshirts_bought / tshirt_ratio)

theorem martha_clothes_count : total_clothes = 18 := by
  sorry

end martha_clothes_count_l1760_176047


namespace multiple_births_quintuplets_l1760_176073

theorem multiple_births_quintuplets (total_babies : ℕ) 
  (triplets_to_quintuplets : ℕ → ℕ) 
  (twins_to_triplets : ℕ → ℕ) 
  (quadruplets_to_quintuplets : ℕ → ℕ) 
  (h1 : total_babies = 1540)
  (h2 : ∀ q, triplets_to_quintuplets q = 6 * q)
  (h3 : ∀ t, twins_to_triplets t = 2 * t)
  (h4 : ∀ q, quadruplets_to_quintuplets q = 3 * q)
  (h5 : ∀ q, 2 * (twins_to_triplets (triplets_to_quintuplets q)) + 
             3 * (triplets_to_quintuplets q) + 
             4 * (quadruplets_to_quintuplets q) + 
             5 * q = total_babies) : 
  ∃ q : ℚ, q = 7700 / 59 ∧ 5 * q = (quintuplets_babies : ℚ) := by
  sorry

end multiple_births_quintuplets_l1760_176073


namespace three_colors_sufficient_and_necessary_l1760_176023

/-- A function that returns the minimum number of colors needed to uniquely identify n keys on a single keychain. -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 2 then n else 3

/-- Theorem stating that for n ≥ 3 keys on a single keychain, 3 colors are sufficient and necessary to uniquely identify each key. -/
theorem three_colors_sufficient_and_necessary (n : ℕ) (h : n ≥ 3) :
  min_colors n = 3 := by sorry

end three_colors_sufficient_and_necessary_l1760_176023
