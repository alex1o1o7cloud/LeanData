import Mathlib

namespace game_download_proof_l1298_129898

/-- Proves that the amount downloaded before the connection slowed down is 310 MB -/
theorem game_download_proof (total_size : ℕ) (current_speed : ℕ) (remaining_time : ℕ) 
  (h1 : total_size = 880)
  (h2 : current_speed = 3)
  (h3 : remaining_time = 190) :
  total_size - current_speed * remaining_time = 310 := by
  sorry

end game_download_proof_l1298_129898


namespace circle_center_distance_l1298_129840

/-- Given a circle with equation x^2 + y^2 - 6x + 8y + 4 = 0 and a point (19, 11),
    the distance between the center of the circle and the point is √481. -/
theorem circle_center_distance (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 8*y + 4 = 0) → 
  Real.sqrt ((19 - x)^2 + (11 - y)^2) = Real.sqrt 481 := by
  sorry

end circle_center_distance_l1298_129840


namespace smallest_dual_base_representation_l1298_129851

/-- Represents a number in a given base with two identical digits --/
def twoDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid in a given base --/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (C D : Nat),
    isValidDigit C 4 ∧
    isValidDigit D 6 ∧
    twoDigitNumber C 4 = 35 ∧
    twoDigitNumber D 6 = 35 ∧
    (∀ (C' D' : Nat),
      isValidDigit C' 4 →
      isValidDigit D' 6 →
      twoDigitNumber C' 4 = twoDigitNumber D' 6 →
      twoDigitNumber C' 4 ≥ 35) :=
by sorry

end smallest_dual_base_representation_l1298_129851


namespace rakesh_salary_l1298_129893

/-- Rakesh's salary calculation -/
theorem rakesh_salary (salary : ℝ) : 
  (salary * (1 - 0.15) * (1 - 0.30) = 2380) → salary = 4000 := by
  sorry

end rakesh_salary_l1298_129893


namespace prob_third_draw_exactly_l1298_129877

/-- Simple random sampling without replacement from a finite population -/
structure SimpleRandomSampling where
  population_size : ℕ
  sample_size : ℕ
  h_sample_size : sample_size ≤ population_size

/-- The probability of drawing a specific individual on the nth draw,
    given they were not drawn in the previous n-1 draws -/
def prob_draw_on_nth (srs : SimpleRandomSampling) (n : ℕ) : ℚ :=
  if n ≤ srs.sample_size
  then 1 / (srs.population_size - n + 1)
  else 0

/-- The probability of not drawing a specific individual on the nth draw,
    given they were not drawn in the previous n-1 draws -/
def prob_not_draw_on_nth (srs : SimpleRandomSampling) (n : ℕ) : ℚ :=
  if n ≤ srs.sample_size
  then (srs.population_size - n) / (srs.population_size - n + 1)
  else 1

theorem prob_third_draw_exactly
  (srs : SimpleRandomSampling)
  (h : srs.population_size = 6 ∧ srs.sample_size = 3) :
  prob_not_draw_on_nth srs 1 * prob_not_draw_on_nth srs 2 * prob_draw_on_nth srs 3 = 1/6 := by
  sorry

end prob_third_draw_exactly_l1298_129877


namespace greatest_integer_2pi_minus_6_l1298_129846

theorem greatest_integer_2pi_minus_6 :
  Int.floor (2 * Real.pi - 6) = 0 :=
sorry

end greatest_integer_2pi_minus_6_l1298_129846


namespace probability_nine_heads_in_twelve_flips_l1298_129806

theorem probability_nine_heads_in_twelve_flips :
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1-p)^(n-k) = 220/4096 := by
sorry

end probability_nine_heads_in_twelve_flips_l1298_129806


namespace greatest_value_quadratic_inequality_l1298_129881

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 9 ∧
  (∀ x : ℝ, x^2 - 12*x + 27 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 12*x_max + 27 ≤ 0) :=
by sorry

end greatest_value_quadratic_inequality_l1298_129881


namespace seven_thirteenths_of_3940_percent_of_25000_l1298_129836

theorem seven_thirteenths_of_3940_percent_of_25000 : 
  (7 / 13 * 3940) / 25000 * 100 = 8.484 := by
  sorry

end seven_thirteenths_of_3940_percent_of_25000_l1298_129836


namespace smaller_angle_measure_l1298_129888

-- Define a parallelogram
structure Parallelogram where
  -- Smaller angle
  angle1 : ℝ
  -- Larger angle
  angle2 : ℝ
  -- Condition: angle2 exceeds angle1 by 70 degrees
  angle_diff : angle2 = angle1 + 70
  -- Condition: adjacent angles are supplementary
  supplementary : angle1 + angle2 = 180

-- Theorem statement
theorem smaller_angle_measure (p : Parallelogram) : p.angle1 = 55 := by
  sorry

end smaller_angle_measure_l1298_129888


namespace five_eighths_of_twelve_fifths_l1298_129875

theorem five_eighths_of_twelve_fifths : (5 / 8 : ℚ) * (12 / 5 : ℚ) = (3 / 2 : ℚ) := by
  sorry

end five_eighths_of_twelve_fifths_l1298_129875


namespace min_value_of_function_equality_holds_l1298_129815

theorem min_value_of_function (x : ℝ) (h : x > 0) : 3 * x + 12 / x^2 ≥ 9 := by
  sorry

theorem equality_holds : ∃ x : ℝ, x > 0 ∧ 3 * x + 12 / x^2 = 9 := by
  sorry

end min_value_of_function_equality_holds_l1298_129815


namespace number_divided_by_three_equals_number_minus_five_l1298_129895

theorem number_divided_by_three_equals_number_minus_five : 
  ∃ x : ℚ, x / 3 = x - 5 ∧ x = 15 / 2 := by
  sorry

end number_divided_by_three_equals_number_minus_five_l1298_129895


namespace find_number_l1298_129814

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 57 :=
  sorry

end find_number_l1298_129814


namespace three_numbers_in_unit_interval_l1298_129863

theorem three_numbers_in_unit_interval (x y z : ℝ) :
  (0 ≤ x ∧ x < 1) → (0 ≤ y ∧ y < 1) → (0 ≤ z ∧ z < 1) →
  ∃ a b : ℝ, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2) :=
by sorry

end three_numbers_in_unit_interval_l1298_129863


namespace happy_cattle_ranch_population_l1298_129824

/-- The number of cows after n years, given an initial population and growth rate -/
def cowPopulation (initialPopulation : ℕ) (growthRate : ℚ) (years : ℕ) : ℚ :=
  initialPopulation * (1 + growthRate) ^ years

/-- Theorem: The cow population on Happy Cattle Ranch after 2 years -/
theorem happy_cattle_ranch_population :
  cowPopulation 200 (1/2) 2 = 450 := by
  sorry

end happy_cattle_ranch_population_l1298_129824


namespace max_product_of_fractions_l1298_129801

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_product_of_fractions (A B C D : ℕ) 
  (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D) :
  (∀ (W X Y Z : ℕ), is_digit W → is_digit X → is_digit Y → is_digit Z →
    W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
    (A : ℚ) / B * (C : ℚ) / D ≥ (W : ℚ) / X * (Y : ℚ) / Z) →
  (A : ℚ) / B * (C : ℚ) / D = 36 := by
sorry

end max_product_of_fractions_l1298_129801


namespace min_cubes_for_specific_box_l1298_129857

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

theorem min_cubes_for_specific_box :
  min_cubes_for_box 7 18 3 9 = 42 := by
  sorry

end min_cubes_for_specific_box_l1298_129857


namespace sum_of_digits_of_large_number_l1298_129894

/-- The sum of the digits of 10^85 - 85 -/
def sumOfDigits : ℕ := 753

/-- The number represented by 10^85 - 85 -/
def largeNumber : ℕ := 10^85 - 85

theorem sum_of_digits_of_large_number :
  (largeNumber.digits 10).sum = sumOfDigits := by sorry

end sum_of_digits_of_large_number_l1298_129894


namespace expression_evaluation_l1298_129802

theorem expression_evaluation : (-1)^10 * 2 + (-2)^3 / 4 = 0 := by
  sorry

end expression_evaluation_l1298_129802


namespace product_of_larger_numbers_l1298_129818

theorem product_of_larger_numbers (A B C : ℝ) 
  (h1 : B - A = C - B) 
  (h2 : A * B = 85) 
  (h3 : B = 10) : 
  B * C = 115 := by
sorry

end product_of_larger_numbers_l1298_129818


namespace eighth_term_of_specific_arithmetic_sequence_l1298_129882

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem eighth_term_of_specific_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = -1)
  (h_diff : ∃ d : ℤ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 8 = -22 :=
sorry

end eighth_term_of_specific_arithmetic_sequence_l1298_129882


namespace sector_angle_l1298_129862

/-- Given a circular sector with circumference 8 and area 4, 
    prove that the central angle in radians is 2. -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_circumference : α * r + 2 * r = 8) 
  (h_area : (1 / 2) * α * r^2 = 4) : 
  α = 2 := by
  sorry

end sector_angle_l1298_129862


namespace complement_of_M_in_U_l1298_129899

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {4, 5}

theorem complement_of_M_in_U :
  (U \ M) = {1, 2, 3} := by sorry

end complement_of_M_in_U_l1298_129899


namespace binomial_coefficient_identity_l1298_129817

theorem binomial_coefficient_identity (n k : ℕ) (h1 : k ≤ n) (h2 : n ≥ 1) :
  Nat.choose n k = Nat.choose (n - 1) (k - 1) + Nat.choose (n - 1) k := by
  sorry

end binomial_coefficient_identity_l1298_129817


namespace train_arrival_interval_l1298_129844

def minutes_between (h1 m1 h2 m2 : ℕ) : ℕ :=
  (h2 * 60 + m2) - (h1 * 60 + m1)

theorem train_arrival_interval (x : ℕ) : 
  x > 0 → 
  minutes_between 10 10 10 55 % x = 0 → 
  minutes_between 10 55 11 58 % x = 0 → 
  x = 9 :=
sorry

end train_arrival_interval_l1298_129844


namespace power_multiplication_l1298_129810

theorem power_multiplication (m : ℝ) : m^5 * m = m^6 := by
  sorry

end power_multiplication_l1298_129810


namespace shoe_probability_l1298_129823

def total_pairs : ℕ := 20
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 5
def gray_pairs : ℕ := 3
def white_pairs : ℕ := 4

theorem shoe_probability :
  let total_shoes := total_pairs * 2
  let prob_black := (black_pairs * 2 / total_shoes) * (black_pairs / (total_shoes - 1))
  let prob_brown := (brown_pairs * 2 / total_shoes) * (brown_pairs / (total_shoes - 1))
  let prob_gray := (gray_pairs * 2 / total_shoes) * (gray_pairs / (total_shoes - 1))
  let prob_white := (white_pairs * 2 / total_shoes) * (white_pairs / (total_shoes - 1))
  prob_black + prob_brown + prob_gray + prob_white = 19 / 130 := by
  sorry

end shoe_probability_l1298_129823


namespace intersection_P_Q_l1298_129800

def P : Set ℝ := {x | |x - 1| < 4}
def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2)}

theorem intersection_P_Q : P ∩ Q = Set.Ioo (-2 : ℝ) 5 := by sorry

end intersection_P_Q_l1298_129800


namespace maze_exit_probabilities_l1298_129855

/-- Represents the three passages in the maze -/
inductive Passage
| one
| two
| three

/-- Time taken to exit each passage -/
def exit_time (p : Passage) : ℕ :=
  match p with
  | Passage.one => 1
  | Passage.two => 2
  | Passage.three => 3

/-- The probability of selecting a passage when n passages are available -/
def select_prob (n : ℕ) : ℚ :=
  1 / n

theorem maze_exit_probabilities :
  let p_one_hour := select_prob 3
  let p_more_than_three_hours := 
    select_prob 3 * select_prob 2 + 
    select_prob 3 * select_prob 2 + 
    select_prob 3 * select_prob 2
  (p_one_hour = 1/3) ∧ 
  (p_more_than_three_hours = 1/2) := by
  sorry

end maze_exit_probabilities_l1298_129855


namespace unique_number_satisfying_conditions_l1298_129873

/-- Function to replace 2s with 5s and 5s with 2s in a number -/
def replaceDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a 5-digit odd number -/
def isFiveDigitOdd (n : ℕ) : Prop := sorry

theorem unique_number_satisfying_conditions :
  ∀ x y : ℕ,
    isFiveDigitOdd x →
    y = replaceDigits x →
    y = 2 * (x + 1) →
    x = 29995 := by sorry

end unique_number_satisfying_conditions_l1298_129873


namespace probability_calculation_l1298_129826

theorem probability_calculation (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) 
  (remaining : ℕ) (h1 : total_students = 2006) (h2 : eliminated = 6) 
  (h3 : selected = 50) (h4 : remaining = total_students - eliminated) :
  (eliminated : ℚ) / (total_students : ℚ) = 3 / 1003 ∧ 
  (selected : ℚ) / (remaining : ℚ) = 25 / 1003 := by
  sorry

#check probability_calculation

end probability_calculation_l1298_129826


namespace abc_positive_l1298_129816

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem abc_positive 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h0 : quadratic a b c 0 = -2)
  (h1 : quadratic a b c 1 = -2)
  (hneg_half : quadratic a b c (-1/2) > 0) :
  a * b * c > 0 := by
  sorry

end abc_positive_l1298_129816


namespace attendants_with_both_tools_l1298_129889

theorem attendants_with_both_tools (pencil_users : ℕ) (pen_users : ℕ) (single_tool_users : ℕ) : 
  pencil_users = 25 →
  pen_users = 15 →
  single_tool_users = 20 →
  pencil_users + pen_users - single_tool_users = 10 := by
sorry

end attendants_with_both_tools_l1298_129889


namespace sum_with_radical_conjugate_l1298_129883

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 5005
  let y : ℝ := 15 + Real.sqrt 5005
  x + y = 30 := by
  sorry

end sum_with_radical_conjugate_l1298_129883


namespace coordinate_translation_l1298_129878

/-- Given a translation of the coordinate system where point A moves from (-1, 3) to (-3, -1),
    prove that the new origin O' has coordinates (2, 4). -/
theorem coordinate_translation (A_old A_new O'_new : ℝ × ℝ) : 
  A_old = (-1, 3) → A_new = (-3, -1) → O'_new = (2, 4) := by
  sorry

end coordinate_translation_l1298_129878


namespace special_triangle_sides_l1298_129876

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The altitudes (heights) of the triangle
  ha : ℕ
  hb : ℕ
  hc : ℕ
  -- The radius of the inscribed circle
  r : ℝ
  -- Conditions
  radius_condition : r = 4/3
  altitudes_sum : ha + hb + hc = 13
  altitude_relation : 1/ha + 1/hb + 1/hc = 3/4

/-- Theorem about the side lengths of the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) : 
  t.a = 32 / Real.sqrt 15 ∧ 
  t.b = 24 / Real.sqrt 15 ∧ 
  t.c = 16 / Real.sqrt 15 :=
by sorry

end special_triangle_sides_l1298_129876


namespace fraction_inequality_l1298_129828

theorem fraction_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  (x₁ + 1) / (x₂ + 1) > x₁ / x₂ := by sorry

end fraction_inequality_l1298_129828


namespace sequence_max_value_l1298_129808

theorem sequence_max_value (n : ℤ) : -n^2 + 15*n + 3 ≤ 59 := by
  sorry

end sequence_max_value_l1298_129808


namespace parabola_c_value_l1298_129870

-- Define the parabola equation
def parabola (a b c : ℝ) (x y : ℝ) : Prop := x = a * y^2 + b * y + c

-- Define the vertex of the parabola
def vertex (x y : ℝ) : Prop := x = 5 ∧ y = 3

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := x = 3 ∧ y = 5

-- Theorem statement
theorem parabola_c_value :
  ∀ (a b c : ℝ),
  (∀ x y, vertex x y → parabola a b c x y) →
  (∀ x y, point_on_parabola x y → parabola a b c x y) →
  a = -1 →
  c = -4 := by
  sorry

end parabola_c_value_l1298_129870


namespace statement_1_false_statement_2_true_statement_3_true_statement_4_false_statement_5_true_l1298_129868

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Statement 1
theorem statement_1_false : ∃ x y z : ℝ, (heartsuit (heartsuit x y) z) ≠ (heartsuit x (heartsuit y z)) := by sorry

-- Statement 2
theorem statement_2_true : ∀ x y : ℝ, 3 * (heartsuit x y) = heartsuit (3 * x) (3 * y) := by sorry

-- Statement 3
theorem statement_3_true : ∀ x y : ℝ, heartsuit x (-y) = heartsuit (-x) y := by sorry

-- Statement 4
theorem statement_4_false : ∃ x : ℝ, heartsuit x x ≠ x := by sorry

-- Statement 5
theorem statement_5_true : ∀ x y : ℝ, heartsuit x y ≥ 0 := by sorry

end statement_1_false_statement_2_true_statement_3_true_statement_4_false_statement_5_true_l1298_129868


namespace point_on_line_l1298_129819

/-- A line in the xy-plane defined by two points -/
structure Line where
  x1 : ℚ
  y1 : ℚ
  x2 : ℚ
  y2 : ℚ

/-- Check if a point (x, y) lies on the given line -/
def Line.contains (l : Line) (x y : ℚ) : Prop :=
  (y - l.y1) * (l.x2 - l.x1) = (x - l.x1) * (l.y2 - l.y1)

theorem point_on_line (l : Line) (x : ℚ) :
  l.x1 = 1 ∧ l.y1 = 9 ∧ l.x2 = -2 ∧ l.y2 = -1 →
  l.contains x 2 →
  x = -11/10 := by
  sorry

end point_on_line_l1298_129819


namespace worksheets_turned_in_l1298_129842

/-- 
Given:
- initial_worksheets: The initial number of worksheets to grade
- graded_worksheets: The number of worksheets graded
- final_worksheets: The final number of worksheets to grade

Prove that the number of worksheets turned in after grading is 36.
-/
theorem worksheets_turned_in 
  (initial_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (final_worksheets : ℕ) 
  (h1 : initial_worksheets = 34)
  (h2 : graded_worksheets = 7)
  (h3 : final_worksheets = 63) :
  final_worksheets - (initial_worksheets - graded_worksheets) = 36 := by
  sorry

end worksheets_turned_in_l1298_129842


namespace class_average_calculation_l1298_129858

theorem class_average_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_average : ℚ)
  (group2_students : ℕ) (group2_average : ℚ) :
  total_students = 30 →
  group1_students = 24 →
  group2_students = 6 →
  group1_average = 85 / 100 →
  group2_average = 92 / 100 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 864 / 1000 := by
  sorry

end class_average_calculation_l1298_129858


namespace concert_revenue_proof_l1298_129854

/-- Calculates the total revenue of a concert given ticket prices and attendance numbers. -/
def concertRevenue (adultPrice : ℕ) (adultAttendance : ℕ) (childAttendance : ℕ) : ℕ :=
  adultPrice * adultAttendance + (adultPrice / 2) * childAttendance

/-- Proves that the total revenue of the concert is $5122 given the specified conditions. -/
theorem concert_revenue_proof :
  concertRevenue 26 183 28 = 5122 := by
  sorry

#eval concertRevenue 26 183 28

end concert_revenue_proof_l1298_129854


namespace chipmunk_families_went_away_l1298_129820

theorem chipmunk_families_went_away (original : ℕ) (left : ℕ) (h1 : original = 86) (h2 : left = 21) :
  original - left = 65 := by
  sorry

end chipmunk_families_went_away_l1298_129820


namespace polynomial_equality_l1298_129869

/-- Given that 2x^5 + 4x^3 + 3x + 4 + g(x) = x^4 - 2x^3 + 3,
    prove that g(x) = -2x^5 + x^4 - 6x^3 - 3x - 1 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 2 * x^5 + 4 * x^3 + 3 * x + 4 + g x = x^4 - 2 * x^3 + 3) :
  g x = -2 * x^5 + x^4 - 6 * x^3 - 3 * x - 1 := by
  sorry

end polynomial_equality_l1298_129869


namespace rachel_setup_time_l1298_129886

/-- Represents the time in hours for Rachel's speed painting process. -/
structure PaintingTime where
  setup : ℝ
  paintingPerVideo : ℝ
  cleanup : ℝ
  editAndPostPerVideo : ℝ
  totalPerVideo : ℝ
  batchSize : ℕ

/-- The setup time for Rachel's speed painting process is 1 hour. -/
theorem rachel_setup_time (t : PaintingTime) : t.setup = 1 :=
  by
  have h1 : t.paintingPerVideo = 1 := by sorry
  have h2 : t.cleanup = 1 := by sorry
  have h3 : t.editAndPostPerVideo = 1.5 := by sorry
  have h4 : t.totalPerVideo = 3 := by sorry
  have h5 : t.batchSize = 4 := by sorry
  
  have total_batch_time : t.setup + t.batchSize * (t.paintingPerVideo + t.editAndPostPerVideo) + t.cleanup = t.batchSize * t.totalPerVideo :=
    by sorry
  
  -- The proof goes here
  sorry


end rachel_setup_time_l1298_129886


namespace product_sum_inequality_l1298_129834

theorem product_sum_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 := by
  sorry

end product_sum_inequality_l1298_129834


namespace restaurant_order_combinations_l1298_129835

theorem restaurant_order_combinations :
  let main_dish_options : ℕ := 12
  let side_dish_options : ℕ := 5
  let person_count : ℕ := 2
  main_dish_options ^ person_count * side_dish_options = 720 :=
by sorry

end restaurant_order_combinations_l1298_129835


namespace division_problem_l1298_129892

theorem division_problem (N : ℕ) (n : ℕ) (h1 : N > 0) :
  (∀ k : ℕ, k ≤ n → ∃ part : ℚ, part = N / (k * (k + 1))) →
  (N / (n * (n + 1)) = N / 400) →
  n = 20 := by
sorry

end division_problem_l1298_129892


namespace sum_of_five_numbers_l1298_129813

theorem sum_of_five_numbers : 1357 + 2468 + 3579 + 4680 + 5791 = 17875 := by
  sorry

end sum_of_five_numbers_l1298_129813


namespace parallel_vectors_m_value_l1298_129809

/-- Given two vectors a and b in R^3, if a is parallel to b, then m = -2 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 3 → ℝ := ![2*m+1, 3, m-1]
  let b : Fin 3 → ℝ := ![2, m, -m]
  (∃ (k : ℝ), a = k • b) → m = -2 := by
  sorry

end parallel_vectors_m_value_l1298_129809


namespace jons_toaster_cost_l1298_129890

/-- Calculates the total cost of a toaster purchase with given parameters. -/
def toaster_total_cost (msrp : ℝ) (standard_insurance_rate : ℝ) (premium_insurance_additional : ℝ) 
                       (tax_rate : ℝ) (recycling_fee : ℝ) : ℝ :=
  let standard_insurance := msrp * standard_insurance_rate
  let premium_insurance := standard_insurance + premium_insurance_additional
  let subtotal := msrp + premium_insurance
  let tax := subtotal * tax_rate
  subtotal + tax + recycling_fee

/-- Theorem stating that the total cost for Jon's toaster purchase is $69.50 -/
theorem jons_toaster_cost : 
  toaster_total_cost 30 0.2 7 0.5 5 = 69.5 := by
  sorry

end jons_toaster_cost_l1298_129890


namespace number_categorization_l1298_129845

/-- Define the set of numbers we're working with -/
def numbers : Set ℚ := {-3.14, 22/7, 0, 2023}

/-- Define the set of negative rational numbers -/
def negative_rationals : Set ℚ := {x : ℚ | x < 0}

/-- Define the set of positive fractions -/
def positive_fractions : Set ℚ := {x : ℚ | x > 0 ∧ x ≠ ⌊x⌋}

/-- Define the set of non-negative integers -/
def non_negative_integers : Set ℤ := {x : ℤ | x ≥ 0}

/-- Define the set of natural numbers (including 0) -/
def natural_numbers : Set ℕ := Set.univ

/-- Theorem stating the categorization of the given numbers -/
theorem number_categorization :
  (-3.14 ∈ negative_rationals) ∧
  (22/7 ∈ positive_fractions) ∧
  (0 ∈ non_negative_integers) ∧
  (2023 ∈ non_negative_integers) ∧
  (0 ∈ natural_numbers) ∧
  (2023 ∈ natural_numbers) :=
by sorry

end number_categorization_l1298_129845


namespace max_height_is_three_l1298_129867

/-- Represents a rectangular prism formed by unit cubes -/
structure RectangularPrism where
  base_area : ℕ
  height : ℕ

/-- The volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℕ :=
  prism.base_area * prism.height

/-- The set of all possible rectangular prisms with a given base area -/
def possible_prisms (base_area : ℕ) (total_cubes : ℕ) : Set RectangularPrism :=
  {prism | prism.base_area = base_area ∧ volume prism ≤ total_cubes}

/-- The theorem stating that the maximum height of a rectangular prism
    with base area 4 and 12 total cubes is 3 -/
theorem max_height_is_three :
  ∀ (prism : RectangularPrism),
    prism ∈ possible_prisms 4 12 →
    prism.height ≤ 3 ∧
    ∃ (max_prism : RectangularPrism),
      max_prism ∈ possible_prisms 4 12 ∧
      max_prism.height = 3 :=
sorry

end max_height_is_three_l1298_129867


namespace middle_integer_is_five_l1298_129885

/-- Given three consecutive one-digit, positive, odd integers where their sum is
    one-seventh of their product, the middle integer is 5. -/
theorem middle_integer_is_five : 
  ∀ n : ℕ, 
    (n > 0 ∧ n < 10) →  -- one-digit positive integer
    (n % 2 = 1) →  -- odd integer
    (∃ (a b : ℕ), a = n - 2 ∧ b = n + 2 ∧  -- consecutive odd integers
      a > 0 ∧ b < 10 ∧  -- all are one-digit positive
      a % 2 = 1 ∧ b % 2 = 1 ∧  -- all are odd
      (a + n + b) = (a * n * b) / 7) →  -- sum is one-seventh of product
    n = 5 :=
by sorry

end middle_integer_is_five_l1298_129885


namespace car_wash_soap_cost_l1298_129841

/-- The cost of each bottle of car wash soap -/
def bottle_cost (washes_per_bottle : ℕ) (total_washes : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_washes / washes_per_bottle)

/-- Theorem stating that the cost of each bottle is $4 -/
theorem car_wash_soap_cost :
  bottle_cost 4 20 20 = 4 := by
  sorry

end car_wash_soap_cost_l1298_129841


namespace input_statement_is_INPUT_l1298_129830

-- Define the possible statement types
inductive Statement
  | PRINT
  | INPUT
  | IF
  | WHILE

-- Define the function of each statement
def statementFunction (s : Statement) : String :=
  match s with
  | Statement.PRINT => "output"
  | Statement.INPUT => "input"
  | Statement.IF => "conditional execution"
  | Statement.WHILE => "looping"

-- Theorem to prove
theorem input_statement_is_INPUT :
  ∃ s : Statement, statementFunction s = "input" ∧ s = Statement.INPUT :=
  sorry

end input_statement_is_INPUT_l1298_129830


namespace elaine_rent_percentage_l1298_129850

/-- Elaine's earnings last year -/
def last_year_earnings : ℝ := 1

/-- Percentage of earnings spent on rent last year -/
def last_year_rent_percentage : ℝ := 20

/-- Earnings increase percentage this year -/
def earnings_increase : ℝ := 20

/-- Percentage of earnings spent on rent this year -/
def this_year_rent_percentage : ℝ := 30

/-- Increase in rent amount from last year to this year -/
def rent_increase : ℝ := 180

theorem elaine_rent_percentage :
  last_year_rent_percentage = 20 :=
by
  sorry

#check elaine_rent_percentage

end elaine_rent_percentage_l1298_129850


namespace jordan_machine_input_l1298_129837

theorem jordan_machine_input (x : ℝ) : 2 * x + 3 - 5 = 27 → x = 14.5 := by
  sorry

end jordan_machine_input_l1298_129837


namespace abs_diff_roots_quadratic_l1298_129833

theorem abs_diff_roots_quadratic : 
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 10
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  |r₁ - r₂| = 3 := by
sorry


end abs_diff_roots_quadratic_l1298_129833


namespace parabola_a_range_l1298_129821

/-- A parabola that opens downwards -/
structure DownwardParabola where
  a : ℝ
  eq : ℝ → ℝ := λ x => a * x^2 - 2 * a * x + 3
  opens_downward : a < 0

/-- The theorem stating the range of 'a' for a downward parabola with positive y-values in (0, 3) -/
theorem parabola_a_range (p : DownwardParabola) 
  (h : ∀ x, 0 < x → x < 3 → p.eq x > 0) : 
  -1 < p.a ∧ p.a < 0 := by
  sorry


end parabola_a_range_l1298_129821


namespace opposite_faces_in_cube_l1298_129856

structure Cube where
  faces : Fin 6 → Char
  top : Fin 6
  front : Fin 6
  right : Fin 6
  back : Fin 6
  left : Fin 6
  bottom : Fin 6
  unique_faces : ∀ i j, i ≠ j → faces i ≠ faces j

def is_opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  f1 ≠ f2 ∧ f1 ≠ c.top ∧ f1 ≠ c.bottom ∧ 
  f2 ≠ c.top ∧ f2 ≠ c.bottom ∧
  (f1 = c.front ∧ f2 = c.back ∨
   f1 = c.back ∧ f2 = c.front ∨
   f1 = c.left ∧ f2 = c.right ∨
   f1 = c.right ∧ f2 = c.left)

theorem opposite_faces_in_cube (c : Cube) 
  (h1 : c.faces c.top = 'A')
  (h2 : c.faces c.front = 'B')
  (h3 : c.faces c.right = 'C')
  (h4 : c.faces c.back = 'D')
  (h5 : c.faces c.left = 'E') :
  is_opposite c c.front c.back :=
by sorry

end opposite_faces_in_cube_l1298_129856


namespace stamp_collection_l1298_129807

theorem stamp_collection (aj kj cj bj : ℕ) : 
  kj = aj / 2 →
  cj = 2 * kj + 5 →
  bj = 3 * aj - 3 →
  aj + kj + cj + bj = 1472 →
  aj = 267 := by
sorry

end stamp_collection_l1298_129807


namespace similar_triangle_coordinates_l1298_129860

/-- Given two points A and B in a Cartesian coordinate system, with O as the origin and center of
    similarity, and triangle A'B'O similar to triangle ABO with a similarity ratio of 1:2,
    prove that the coordinates of B' are (-3, -2). -/
theorem similar_triangle_coordinates (A B : ℝ × ℝ) (h_A : A = (-4, 2)) (h_B : B = (-6, -4)) :
  let O : ℝ × ℝ := (0, 0)
  let similarity_ratio : ℝ := 1 / 2
  let B' : ℝ × ℝ := (similarity_ratio * B.1, similarity_ratio * B.2)
  B' = (-3, -2) := by
  sorry

end similar_triangle_coordinates_l1298_129860


namespace specific_window_side_length_l1298_129827

/-- Represents a square window with glass panes -/
structure SquareWindow where
  /-- Number of panes in each row/column -/
  panes_per_side : ℕ
  /-- Width of a single pane -/
  pane_width : ℝ
  /-- Width of borders between panes and around the window -/
  border_width : ℝ

/-- Calculates the side length of the square window -/
def window_side_length (w : SquareWindow) : ℝ :=
  w.panes_per_side * w.pane_width + (w.panes_per_side + 1) * w.border_width

/-- Theorem stating the side length of the specific window described in the problem -/
theorem specific_window_side_length :
  ∃ w : SquareWindow,
    w.panes_per_side = 3 ∧
    w.pane_width * 3 = w.pane_width * w.panes_per_side ∧
    w.border_width = 3 ∧
    window_side_length w = 42 := by
  sorry

end specific_window_side_length_l1298_129827


namespace tom_seashells_per_day_l1298_129805

/-- Represents the number of seashells Tom found each day -/
def seashells_per_day (total_seashells : ℕ) (days_at_beach : ℕ) : ℕ :=
  total_seashells / days_at_beach

/-- Theorem stating that Tom found 7 seashells per day -/
theorem tom_seashells_per_day :
  seashells_per_day 35 5 = 7 := by
  sorry

end tom_seashells_per_day_l1298_129805


namespace smallest_x_satisfying_equation_l1298_129859

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℤ) - x * (⌊x⌋ : ℤ) = 7 ∧ 
  ∀ (y : ℝ), y > 0 → (⌊y^2⌋ : ℤ) - y * (⌊y⌋ : ℤ) = 7 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_x_satisfying_equation_l1298_129859


namespace parallel_lines_x_value_l1298_129829

/-- A line in a 2D plane --/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Check if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  (l1.point1.1 = l1.point2.1) = (l2.point1.1 = l2.point2.1)

theorem parallel_lines_x_value (l1 l2 : Line) (x : ℝ) :
  l1.point1 = (-1, -2) →
  l1.point2 = (-1, 4) →
  l2.point1 = (2, 1) →
  l2.point2 = (x, 6) →
  parallel l1 l2 →
  x = 2 := by
  sorry

end parallel_lines_x_value_l1298_129829


namespace system_solution_l1298_129852

theorem system_solution : ∃ (x y : ℝ), x - y = 3 ∧ x + y = 1 ∧ x = 2 ∧ y = -1 := by
  sorry

end system_solution_l1298_129852


namespace points_on_circle_l1298_129811

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a motion (isometry) of the plane -/
structure Motion where
  transform : Point → Point

/-- A system of points with the given property -/
structure PointSystem where
  points : List Point
  motion_property : ∀ (p q : Point), p ∈ points → q ∈ points → 
    ∃ (m : Motion), m.transform p = q ∧ (∀ x ∈ points, m.transform x ∈ points)

/-- Definition of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The main theorem -/
theorem points_on_circle (sys : PointSystem) : 
  ∃ (c : Circle), ∀ p ∈ sys.points, 
    (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2 := by
  sorry

end points_on_circle_l1298_129811


namespace largest_five_digit_with_product_10080_l1298_129803

def digit_product (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_10080 :
  ∀ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ digit_product n = 10080 → n ≤ 98754 :=
by sorry

end largest_five_digit_with_product_10080_l1298_129803


namespace cell_count_after_twelve_days_l1298_129843

/-- Represents the cell growth and death process over 12 days -/
def cell_growth (initial_cells : ℕ) (split_interval : ℕ) (total_days : ℕ) (death_day : ℕ) (cells_died : ℕ) : ℕ :=
  let cycles := total_days / split_interval
  let final_count := initial_cells * 2^cycles
  if death_day ≤ total_days then final_count - cells_died else final_count

/-- Theorem stating the number of cells after 12 days -/
theorem cell_count_after_twelve_days :
  cell_growth 5 3 12 9 3 = 77 := by
  sorry

end cell_count_after_twelve_days_l1298_129843


namespace domain_of_shifted_sum_l1298_129887

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 0 4

-- State the theorem
theorem domain_of_shifted_sum (hf : Set.range f = dom_f) :
  {x : ℝ | ∃ y, y ∈ dom_f ∧ x + 1 = y} ∩ {x : ℝ | ∃ y, y ∈ dom_f ∧ x - 1 = y} = Set.Icc 1 3 := by
  sorry

end domain_of_shifted_sum_l1298_129887


namespace equation_solution_l1298_129849

theorem equation_solution : ∃ (x₁ x₂ : ℚ), x₁ = 7/4 ∧ x₂ = 1/4 ∧
  (16 * (x₁ - 1)^2 - 9 = 0) ∧ (16 * (x₂ - 1)^2 - 9 = 0) := by
  sorry

end equation_solution_l1298_129849


namespace not_always_perfect_square_exists_l1298_129804

/-- Given an n-digit number x, prove that there doesn't always exist a non-negative integer y ≤ 9
    and an integer z such that 10^(n+1) * z + 10x + y is a perfect square. -/
theorem not_always_perfect_square_exists (n : ℕ) : 
  ∃ x : ℕ, (10^n ≤ x ∧ x < 10^(n+1)) →
    ¬∃ (y z : ℤ), 0 ≤ y ∧ y ≤ 9 ∧ ∃ (k : ℤ), 10^(n+1) * z + 10 * x + y = k^2 :=
by sorry

end not_always_perfect_square_exists_l1298_129804


namespace system_no_solution_l1298_129812

theorem system_no_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y = 1 ∧ 2 * x + a * y = 1) ↔ a = -2 :=
by sorry

end system_no_solution_l1298_129812


namespace area_of_triangle_ABC_l1298_129874

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Calculates the area of a triangle given three points -/
def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC' :
  let A : ℝ × ℝ := (3, 4)
  let B' := reflect_y_axis A
  let C' := reflect_y_eq_x B'
  triangle_area A B' C' = 21 := by sorry

end area_of_triangle_ABC_l1298_129874


namespace max_value_expression_max_value_achievable_l1298_129879

theorem max_value_expression (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) ≤ 9 * Real.sqrt 2 :=
by sorry

theorem max_value_achievable :
  ∃ (a b : ℝ), a ≥ 1 ∧ b ≥ 1 ∧
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) = 9 * Real.sqrt 2 :=
by sorry

end max_value_expression_max_value_achievable_l1298_129879


namespace reading_pattern_equation_l1298_129822

/-- Represents the total number of words in the book "Mencius" -/
def total_words : ℕ := 34685

/-- Represents the number of days taken to read the book -/
def days : ℕ := 3

/-- Represents the relationship between words read on consecutive days -/
def daily_increase_factor : ℕ := 2

/-- Theorem stating the correct equation for the reading pattern -/
theorem reading_pattern_equation (x : ℕ) :
  x + daily_increase_factor * x + daily_increase_factor^2 * x = total_words →
  x + 2*x + 4*x = total_words :=
by sorry

end reading_pattern_equation_l1298_129822


namespace square_root_of_49_l1298_129861

theorem square_root_of_49 : (Real.sqrt 49)^2 = 49 := by
  sorry

end square_root_of_49_l1298_129861


namespace birthday_candles_sharing_l1298_129891

/-- 
Given that Ambika has 4 birthday candles and Aniyah has 6 times as many,
this theorem proves that when they put their candles together and share them equally,
each will have 14 candles.
-/
theorem birthday_candles_sharing (ambika_candles : ℕ) (aniyah_multiplier : ℕ) :
  ambika_candles = 4 →
  aniyah_multiplier = 6 →
  let aniyah_candles := ambika_candles * aniyah_multiplier
  let total_candles := ambika_candles + aniyah_candles
  total_candles / 2 = 14 := by
  sorry


end birthday_candles_sharing_l1298_129891


namespace organize_toys_time_l1298_129884

/-- The time in minutes it takes to organize all toys given the following conditions:
  * There are 50 toys to organize
  * 4 toys are put into the box every 45 seconds
  * 3 toys are taken out immediately after each 45-second interval
-/
def organizeToys (totalToys : ℕ) (putIn : ℕ) (takeOut : ℕ) (cycleTime : ℚ) : ℚ :=
  let netIncrease : ℕ := putIn - takeOut
  let almostFullCycles : ℕ := (totalToys - putIn) / netIncrease
  let almostFullTime : ℚ := (almostFullCycles : ℚ) * cycleTime
  let finalCycleTime : ℚ := cycleTime
  (almostFullTime + finalCycleTime) / 60

theorem organize_toys_time :
  organizeToys 50 4 3 (45 / 60) = 35.25 := by
  sorry

end organize_toys_time_l1298_129884


namespace laura_payment_l1298_129853

/-- The amount Laura gave to the cashier --/
def amount_given_to_cashier (pants_price : ℕ) (shirts_price : ℕ) (pants_quantity : ℕ) (shirts_quantity : ℕ) (change : ℕ) : ℕ :=
  pants_price * pants_quantity + shirts_price * shirts_quantity + change

/-- Theorem stating that Laura gave $250 to the cashier --/
theorem laura_payment : amount_given_to_cashier 54 33 2 4 10 = 250 := by
  sorry

end laura_payment_l1298_129853


namespace min_removals_for_no_products_l1298_129865

theorem min_removals_for_no_products (n : ℕ) (hn : n = 1982) :
  ∃ (S : Finset ℕ),
    S.card = 43 ∧ 
    (∀ k ∈ Finset.range (n + 1) \ S, k = 1 ∨ k ≥ 45) ∧
    (∀ a b k, a ∈ Finset.range (n + 1) \ S → b ∈ Finset.range (n + 1) \ S → 
      k ∈ Finset.range (n + 1) \ S → a ≠ b → a * b ≠ k) ∧
    (∀ T : Finset ℕ, T.card < 43 → 
      ∃ a b k, a ∈ Finset.range (n + 1) \ T → b ∈ Finset.range (n + 1) \ T → 
        k ∈ Finset.range (n + 1) \ T → a ≠ b → a * b = k) :=
by sorry

end min_removals_for_no_products_l1298_129865


namespace number_percentage_problem_l1298_129839

theorem number_percentage_problem (N : ℚ) : 
  (4/5 : ℚ) * (3/8 : ℚ) * N = 24 → (5/2 : ℚ) * N = 200 := by
  sorry

end number_percentage_problem_l1298_129839


namespace parabola_arc_projection_difference_l1298_129897

/-- 
Given a parabola y = x^2 + px + q and two rays y = x and y = 2x for x ≥ 0,
prove that the difference between the projection of the right arc and 
the projection of the left arc on the x-axis is equal to 1.
-/
theorem parabola_arc_projection_difference 
  (p q : ℝ) : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ < x₂) ∧ (x₃ < x₄) ∧
    (x₁^2 + p*x₁ + q = x₁) ∧ 
    (x₂^2 + p*x₂ + q = x₂) ∧
    (x₃^2 + p*x₃ + q = 2*x₃) ∧ 
    (x₄^2 + p*x₄ + q = 2*x₄) ∧
    (x₄ - x₂) - (x₁ - x₃) = 1 := by
  sorry

end parabola_arc_projection_difference_l1298_129897


namespace apartment_fractions_l1298_129864

theorem apartment_fractions (one_bedroom : Real) (two_bedroom : Real) 
  (h1 : one_bedroom = 0.17)
  (h2 : one_bedroom + two_bedroom = 0.5) :
  two_bedroom = 0.33 := by
sorry

end apartment_fractions_l1298_129864


namespace triangular_area_l1298_129832

/-- The area of the triangular part of a piece of land -/
theorem triangular_area (total_length total_width rect_length rect_width : ℝ) 
  (h1 : total_length = 20)
  (h2 : total_width = 6)
  (h3 : rect_length = 15)
  (h4 : rect_width = 6) :
  total_length * total_width - rect_length * rect_width = 30 := by
  sorry

end triangular_area_l1298_129832


namespace q_satisfies_conditions_l1298_129880

/-- The quadratic polynomial q(x) that satisfies specific conditions -/
def q (x : ℚ) : ℚ := -25/11 * x^2 + 75/11 * x + 450/11

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 8 = -50 := by
  sorry

end q_satisfies_conditions_l1298_129880


namespace expand_expression_l1298_129847

theorem expand_expression (x y : ℝ) : (16*x + 18 - 7*y) * 3*x = 48*x^2 + 54*x - 21*x*y := by
  sorry

end expand_expression_l1298_129847


namespace divisor_sum_difference_bound_l1298_129866

/-- Sum of counts of positive even divisors of numbers from 1 to n -/
def D1 (n : ℕ) : ℕ := sorry

/-- Sum of counts of positive odd divisors of numbers from 1 to n -/
def D2 (n : ℕ) : ℕ := sorry

/-- The difference between D2 and D1 is no greater than n -/
theorem divisor_sum_difference_bound (n : ℕ) : D2 n - D1 n ≤ n := by sorry

end divisor_sum_difference_bound_l1298_129866


namespace student_distribution_theorem_l1298_129896

/-- The number of ways to distribute students into dormitories -/
def distribute_students (total_students : ℕ) (num_dorms : ℕ) (min_per_dorm : ℕ) (max_per_dorm : ℕ) : ℕ := sorry

/-- The number of ways to distribute students with one student excluded from one dorm -/
def distribute_students_with_exclusion (total_students : ℕ) (num_dorms : ℕ) (min_per_dorm : ℕ) (max_per_dorm : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to distribute 5 students into 3 dormitories with constraints -/
theorem student_distribution_theorem :
  distribute_students_with_exclusion 5 3 1 2 = 60 := by sorry

end student_distribution_theorem_l1298_129896


namespace total_students_present_l1298_129838

/-- Represents a kindergarten session with registered and absent students -/
structure Session where
  registered : ℕ
  absent : ℕ

/-- Calculates the number of present students in a session -/
def presentStudents (s : Session) : ℕ := s.registered - s.absent

/-- Represents the kindergarten school data -/
structure KindergartenSchool where
  morning : Session
  earlyAfternoon : Session
  lateAfternoon : Session
  earlyEvening : Session
  lateEvening : Session
  transferredOut : ℕ
  newRegistrations : ℕ
  newAttending : ℕ

/-- The main theorem to prove -/
theorem total_students_present (school : KindergartenSchool)
  (h1 : school.morning = { registered := 75, absent := 9 })
  (h2 : school.earlyAfternoon = { registered := 72, absent := 12 })
  (h3 : school.lateAfternoon = { registered := 90, absent := 15 })
  (h4 : school.earlyEvening = { registered := 50, absent := 6 })
  (h5 : school.lateEvening = { registered := 60, absent := 10 })
  (h6 : school.transferredOut = 3)
  (h7 : school.newRegistrations = 3)
  (h8 : school.newAttending = 1) :
  presentStudents school.morning +
  presentStudents school.earlyAfternoon +
  presentStudents school.lateAfternoon +
  presentStudents school.earlyEvening +
  presentStudents school.lateEvening -
  school.transferredOut +
  school.newAttending = 293 := by
  sorry

end total_students_present_l1298_129838


namespace expression_evaluation_l1298_129848

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^2)^y * (y^3)^x / ((y^2)^y * (x^3)^x) = x^(2*y - 3*x) * y^(3*x - 2*y) := by
  sorry

end expression_evaluation_l1298_129848


namespace mo_negative_bo_positive_l1298_129825

-- Define the two types of people
inductive PersonType
| Positive
| Negative

-- Define a person with a type
structure Person where
  name : String
  type : PersonType

-- Define the property of asking a question
def asksQuestion (p : Person) (q : Prop) : Prop :=
  match p.type with
  | PersonType.Positive => q
  | PersonType.Negative => ¬q

-- Define Mo and Bo
def Mo : Person := { name := "Mo", type := PersonType.Negative }
def Bo : Person := { name := "Bo", type := PersonType.Positive }

-- Define the question Mo asked
def moQuestion : Prop := Mo.type = PersonType.Negative ∧ Bo.type = PersonType.Negative

-- Theorem stating that Mo is negative and Bo is positive
theorem mo_negative_bo_positive :
  asksQuestion Mo moQuestion ∧ (Mo.type = PersonType.Negative ∧ Bo.type = PersonType.Positive) :=
by sorry


end mo_negative_bo_positive_l1298_129825


namespace polly_tweets_theorem_l1298_129831

/-- Polly's tweeting behavior -/
structure PollyTweets where
  happy_rate : ℕ
  hungry_rate : ℕ
  mirror_rate : ℕ
  happy_duration : ℕ
  hungry_duration : ℕ
  mirror_duration : ℕ

/-- Calculate the total number of tweets -/
def total_tweets (p : PollyTweets) : ℕ :=
  p.happy_rate * p.happy_duration +
  p.hungry_rate * p.hungry_duration +
  p.mirror_rate * p.mirror_duration

/-- Theorem: Polly's total tweets equal 1340 -/
theorem polly_tweets_theorem (p : PollyTweets) 
  (h1 : p.happy_rate = 18)
  (h2 : p.hungry_rate = 4)
  (h3 : p.mirror_rate = 45)
  (h4 : p.happy_duration = 20)
  (h5 : p.hungry_duration = 20)
  (h6 : p.mirror_duration = 20) :
  total_tweets p = 1340 := by
  sorry

end polly_tweets_theorem_l1298_129831


namespace square_root_sequence_l1298_129871

theorem square_root_sequence (n : ℕ) : 
  (∀ k ∈ Finset.range 35, Int.floor (Real.sqrt (n^2 + k : ℝ)) = n) ↔ n = 17 := by
sorry

end square_root_sequence_l1298_129871


namespace A_union_complement_B_eq_l1298_129872

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,4}

theorem A_union_complement_B_eq : A ∪ (U \ B) = {1,2,3,5} := by sorry

end A_union_complement_B_eq_l1298_129872
