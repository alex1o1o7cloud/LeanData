import Mathlib

namespace NUMINAMATH_CALUDE_basketball_tryouts_l59_5988

theorem basketball_tryouts (girls : ℕ) (boys : ℕ) (called_back : ℕ) (not_selected : ℕ) : 
  boys = 14 →
  called_back = 2 →
  not_selected = 21 →
  girls + boys = called_back + not_selected →
  girls = 9 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l59_5988


namespace NUMINAMATH_CALUDE_neighbor_birth_year_l59_5951

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year ≤ 9999

def first_two_digits (year : ℕ) : ℕ :=
  year / 100

def last_two_digits (year : ℕ) : ℕ :=
  year % 100

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def diff_of_digits (n : ℕ) : ℕ :=
  (n / 10) - (n % 10)

theorem neighbor_birth_year :
  ∀ year : ℕ, is_valid_year year →
    (sum_of_digits (first_two_digits year) = diff_of_digits (last_two_digits year)) →
    year = 1890 :=
by sorry

end NUMINAMATH_CALUDE_neighbor_birth_year_l59_5951


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l59_5979

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2^(n-1) - 2

/-- The problem statement -/
theorem six_balls_two_boxes : distribute_balls 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l59_5979


namespace NUMINAMATH_CALUDE_triangle_shortest_side_l59_5915

theorem triangle_shortest_side 
  (a b c : ℕ) 
  (h : ℕ) 
  (area : ℕ) 
  (ha : a = 24) 
  (hperim : a + b + c = 55) 
  (harea : area = a * h / 2) 
  (hherons : area^2 = ((a + b + c) / 2) * (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) : 
  min b c = 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_shortest_side_l59_5915


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l59_5957

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
    2 * x₁^2 + (m + 1) * x₁ + m = 0 ∧
    2 * x₂^2 + (m + 1) * x₂ + m = 0) →
  m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l59_5957


namespace NUMINAMATH_CALUDE_parabola_directrix_l59_5912

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 6 * x + 2

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -11/12

/-- Theorem: The directrix of the given parabola is y = -11/12 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    ∃ f : ℝ × ℝ, (f.2 - p.2)^2 = 4 * 3 * ((p.1 - f.1)^2 + (p.2 - d)^2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l59_5912


namespace NUMINAMATH_CALUDE_class_average_weight_l59_5904

theorem class_average_weight (students_A students_B : ℕ) (avg_weight_A avg_weight_B : ℝ) :
  students_A = 36 →
  students_B = 24 →
  avg_weight_A = 30 →
  avg_weight_B = 30 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l59_5904


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l59_5922

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + 2*x - 3 > 0 ↔ x < -3 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l59_5922


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l59_5955

theorem other_root_of_quadratic (m : ℝ) : 
  (1 : ℝ)^2 + m * 1 - 5 = 0 → 
  (-5 : ℝ)^2 + m * (-5) - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l59_5955


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_equation_C_has_equal_roots_l59_5936

theorem quadratic_equal_roots (a b c : ℝ) (h : a ≠ 0) :
  (b^2 - 4*a*c = 0) ↔ ∃! x, a*x^2 + b*x + c = 0 :=
sorry

theorem equation_C_has_equal_roots :
  ∃! x, x^2 + 12*x + 36 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_equation_C_has_equal_roots_l59_5936


namespace NUMINAMATH_CALUDE_six_awards_four_students_l59_5963

/-- The number of ways to distribute awards to students. -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The theorem stating the correct number of ways to distribute 6 awards to 4 students. -/
theorem six_awards_four_students :
  distribute_awards 6 4 = 1260 :=
sorry

end NUMINAMATH_CALUDE_six_awards_four_students_l59_5963


namespace NUMINAMATH_CALUDE_abs_x_plus_y_equals_three_l59_5995

theorem abs_x_plus_y_equals_three (x y : ℝ) 
  (eq1 : |x| + 2*y = 2) 
  (eq2 : 2*|x| + y = 7) : 
  |x| + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_plus_y_equals_three_l59_5995


namespace NUMINAMATH_CALUDE_triangle_problem_l59_5977

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions and the statements to prove --/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.b * cos t.C + t.c = 2 * t.a) 
  (h2 : cos t.A = 1 / 7) : 
  t.B = π / 3 ∧ t.c / t.a = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l59_5977


namespace NUMINAMATH_CALUDE_wages_problem_l59_5966

/-- Given a sum of money that can pay x's wages for 36 days and y's wages for 45 days,
    prove that it can pay both x and y's wages together for 20 days. -/
theorem wages_problem (S : ℝ) (x y : ℝ → ℝ) :
  (∃ (Wx Wy : ℝ), Wx > 0 ∧ Wy > 0 ∧ S = 36 * Wx ∧ S = 45 * Wy) →
  ∃ D : ℝ, D = 20 ∧ S = D * (x 1 + y 1) :=
by sorry

end NUMINAMATH_CALUDE_wages_problem_l59_5966


namespace NUMINAMATH_CALUDE_bianca_coloring_books_l59_5940

/-- Represents the number of coloring books Bianca gave away -/
def books_given_away : ℕ := 6

/-- Represents Bianca's initial number of coloring books -/
def initial_books : ℕ := 45

/-- Represents the number of coloring books Bianca bought -/
def books_bought : ℕ := 20

/-- Represents Bianca's final number of coloring books -/
def final_books : ℕ := 59

theorem bianca_coloring_books : 
  initial_books - books_given_away + books_bought = final_books :=
by sorry

end NUMINAMATH_CALUDE_bianca_coloring_books_l59_5940


namespace NUMINAMATH_CALUDE_mikes_remaining_nickels_l59_5923

/-- Given Mike's initial number of nickels and the number borrowed by his dad,
    proves that the number of nickels Mike has now is the difference between
    the initial number and the borrowed number. -/
theorem mikes_remaining_nickels
  (initial_nickels : ℕ)
  (borrowed_nickels : ℕ)
  (h1 : initial_nickels = 87)
  (h2 : borrowed_nickels = 75)
  : initial_nickels - borrowed_nickels = 12 := by
  sorry

end NUMINAMATH_CALUDE_mikes_remaining_nickels_l59_5923


namespace NUMINAMATH_CALUDE_expected_winnings_is_one_sixth_l59_5991

/-- A strange die with 6 sides -/
inductive DieSide
  | one
  | two
  | three
  | four
  | five
  | six

/-- Probability of rolling each side of the die -/
def probability (s : DieSide) : ℚ :=
  match s with
  | DieSide.one => 1/4
  | DieSide.two => 1/4
  | DieSide.three => 1/6
  | DieSide.four => 1/6
  | DieSide.five => 1/6
  | DieSide.six => 1/12

/-- Winnings (or losses) for each outcome -/
def winnings (s : DieSide) : ℤ :=
  match s with
  | DieSide.one => 2
  | DieSide.two => 2
  | DieSide.three => 4
  | DieSide.four => 4
  | DieSide.five => -6
  | DieSide.six => -12

/-- Expected value of winnings -/
def expected_winnings : ℚ :=
  (probability DieSide.one * winnings DieSide.one) +
  (probability DieSide.two * winnings DieSide.two) +
  (probability DieSide.three * winnings DieSide.three) +
  (probability DieSide.four * winnings DieSide.four) +
  (probability DieSide.five * winnings DieSide.five) +
  (probability DieSide.six * winnings DieSide.six)

theorem expected_winnings_is_one_sixth :
  expected_winnings = 1/6 := by sorry

end NUMINAMATH_CALUDE_expected_winnings_is_one_sixth_l59_5991


namespace NUMINAMATH_CALUDE_no_real_solutions_l59_5948

theorem no_real_solutions : ¬∃ x : ℝ, x + 36 / (x - 3) = -9 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l59_5948


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l59_5971

/-- A right-angled triangle with one angle of 30° and base length of 6 units has a base length of 6 units. -/
theorem right_triangle_base_length (a b c : ℝ) (θ : ℝ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  θ = π/6 →  -- 30° angle in radians
  a = 6 →  -- base length
  a = 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_base_length_l59_5971


namespace NUMINAMATH_CALUDE_multiply_by_9999_l59_5908

theorem multiply_by_9999 : ∃ x : ℕ, x * 9999 = 4690640889 ∧ x = 469131 := by sorry

end NUMINAMATH_CALUDE_multiply_by_9999_l59_5908


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l59_5960

theorem multiple_with_binary_digits (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m ≠ 0 ∧ n ∣ m ∧ (Nat.digits 10 m).length ≤ n ∧ ∀ d ∈ Nat.digits 10 m, d = 0 ∨ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l59_5960


namespace NUMINAMATH_CALUDE_abs_sum_min_value_l59_5992

theorem abs_sum_min_value (x : ℚ) : 
  ∃ (min : ℚ), min = 5 ∧ (∀ y : ℚ, |y - 2| + |y + 3| ≥ min) ∧ (|x - 2| + |x + 3| = min) :=
sorry

end NUMINAMATH_CALUDE_abs_sum_min_value_l59_5992


namespace NUMINAMATH_CALUDE_triangle_midpoint_line_sum_l59_5952

/-- Given a triangle ABC with vertices A(0,6), B(0,0), C(10,0), and D the midpoint of AB,
    the sum of the slope and y-intercept of line CD is 27/10 -/
theorem triangle_midpoint_line_sum (A B C D : ℝ × ℝ) : 
  A = (0, 6) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2
  m + b = 27 / 10 := by sorry

end NUMINAMATH_CALUDE_triangle_midpoint_line_sum_l59_5952


namespace NUMINAMATH_CALUDE_product_equals_eight_l59_5986

theorem product_equals_eight : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l59_5986


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l59_5907

/-- Subtraction of two specific decimal numbers -/
theorem subtraction_of_decimals : (678.90 : ℝ) - (123.45 : ℝ) = 555.55 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l59_5907


namespace NUMINAMATH_CALUDE_placards_per_person_l59_5937

def total_placards : ℕ := 5682
def people_entered : ℕ := 2841

theorem placards_per_person :
  total_placards / people_entered = 2 := by
  sorry

end NUMINAMATH_CALUDE_placards_per_person_l59_5937


namespace NUMINAMATH_CALUDE_soap_box_length_l59_5954

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Theorem: Given the carton and soap box dimensions, if 360 soap boxes fit exactly in the carton,
    then the length of a soap box is 7 inches -/
theorem soap_box_length
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton.length = 30 ∧ carton.width = 42 ∧ carton.height = 60)
  (h2 : soap.width = 6 ∧ soap.height = 5)
  (h3 : boxVolume carton = 360 * boxVolume soap) :
  soap.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_length_l59_5954


namespace NUMINAMATH_CALUDE_worker_completion_times_l59_5945

def job_completion_time (worker1_time worker2_time : ℝ) : Prop :=
  (1 / worker1_time + 1 / worker2_time = 1 / 8) ∧
  (worker1_time = worker2_time - 12)

theorem worker_completion_times :
  ∃ (worker1_time worker2_time : ℝ),
    job_completion_time worker1_time worker2_time ∧
    worker1_time = 24 ∧
    worker2_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_worker_completion_times_l59_5945


namespace NUMINAMATH_CALUDE_mistaken_subtraction_l59_5919

theorem mistaken_subtraction (x : ℤ) : x - 59 = 43 → x - 46 = 56 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_subtraction_l59_5919


namespace NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l59_5943

/-- The fraction of water remaining in a radiator after multiple replacements with antifreeze -/
def water_fraction (total_capacity : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  (1 - replacement_volume / total_capacity) ^ num_replacements

/-- The fraction of water remaining in a 20-quart radiator after 4 replacements of 5 quarts each -/
theorem water_fraction_after_four_replacements :
  water_fraction 20 5 4 = 81 / 256 := by
  sorry

#eval water_fraction 20 5 4

end NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l59_5943


namespace NUMINAMATH_CALUDE_relay_race_time_l59_5958

/-- Represents the time taken by each runner in the relay race -/
structure RelayTimes where
  mary : ℕ
  susan : ℕ
  jen : ℕ
  tiffany : ℕ

/-- Calculates the total time of the relay race -/
def total_time (times : RelayTimes) : ℕ :=
  times.mary + times.susan + times.jen + times.tiffany

/-- Theorem stating the total time of the relay race -/
theorem relay_race_time : ∃ (times : RelayTimes),
  times.mary = 2 * times.susan ∧
  times.susan = times.jen + 10 ∧
  times.jen = 30 ∧
  times.tiffany = times.mary - 7 ∧
  total_time times = 223 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_l59_5958


namespace NUMINAMATH_CALUDE_gcd_18_30_l59_5921

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l59_5921


namespace NUMINAMATH_CALUDE_riku_sticker_count_l59_5949

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := 85

/-- The ratio of Riku's stickers to Kristoff's stickers -/
def riku_to_kristoff_ratio : ℕ := 25

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := kristoff_stickers * riku_to_kristoff_ratio

theorem riku_sticker_count : riku_stickers = 2125 := by
  sorry

end NUMINAMATH_CALUDE_riku_sticker_count_l59_5949


namespace NUMINAMATH_CALUDE_exists_m_divides_polynomial_l59_5913

theorem exists_m_divides_polynomial (p : ℕ) (h_prime : Nat.Prime p) (h_cong : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ p ∣ (m^3 + m^2 - 2*m - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divides_polynomial_l59_5913


namespace NUMINAMATH_CALUDE_expression_simplification_l59_5975

theorem expression_simplification (x : ℝ) (h : x = 1) :
  (x^2 - 4*x + 4) / (2*x) / ((x^2 - 2*x) / x^2) + 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l59_5975


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l59_5925

theorem imaginary_part_of_z (z : ℂ) (h : (z - 2*Complex.I)*Complex.I = 1 + Complex.I) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l59_5925


namespace NUMINAMATH_CALUDE_distribute_5_3_l59_5965

/-- The number of ways to distribute n distinct items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags -/
theorem distribute_5_3 : distribute 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l59_5965


namespace NUMINAMATH_CALUDE_incorrect_calculation_l59_5978

theorem incorrect_calculation (x y : ℝ) : 
  (-2 * x^2 * y^2)^3 / (-x * y)^3 ≠ -2 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l59_5978


namespace NUMINAMATH_CALUDE_curve_T_and_fixed_point_l59_5998

-- Define the points A, B, C, and O
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (0, -1)
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (x + 1) * (x - 1) + y * y = y * (y + 1)

-- Define the curve T
def curve_T (x y : ℝ) : Prop := y = x^2 - 1

-- Define the tangent line at point P
def tangent_line (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₀, y₀) := P
  y - y₀ = 2 * x₀ * (x - x₀)

-- Define the line y = -5/4
def line_y_eq_neg_5_4 (x y : ℝ) : Prop := y = -5/4

-- Define the circle with diameter PQ
def circle_PQ (P Q H : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  let (xh, yh) := H
  (xh - xp) * (xh - xq) + (yh - yp) * (yh - yq) = 0

-- State the theorem
theorem curve_T_and_fixed_point :
  -- Part 1: The trajectory of point M is curve T
  (∀ M : ℝ × ℝ, condition_M M ↔ curve_T M.1 M.2) ∧
  -- Part 2: The circle with diameter PQ passes through a fixed point
  (∀ P : ℝ × ℝ, P.1 ≠ 0 → curve_T P.1 P.2 →
    ∃ Q : ℝ × ℝ,
      tangent_line P Q.1 Q.2 ∧
      line_y_eq_neg_5_4 Q.1 Q.2 ∧
      circle_PQ P Q (0, -3/4)) := by
  sorry

end NUMINAMATH_CALUDE_curve_T_and_fixed_point_l59_5998


namespace NUMINAMATH_CALUDE_john_chess_probability_l59_5918

theorem john_chess_probability (p_win : ℚ) (h : p_win = 2 / 5) : 1 - p_win = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_john_chess_probability_l59_5918


namespace NUMINAMATH_CALUDE_batsman_average_l59_5933

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  inningScore : ℝ
  averageIncrease : ℝ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem: Given the conditions, the batsman's new average is 55 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.inningScore = 95)
  (h2 : b.averageIncrease = 2.5)
  : newAverage b = 55 := by
  sorry

#eval newAverage { initialAverage := 52.5, inningScore := 95, averageIncrease := 2.5 }

end NUMINAMATH_CALUDE_batsman_average_l59_5933


namespace NUMINAMATH_CALUDE_complex_equation_solution_l59_5926

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : 
  a + b * i = -1 + 2 * i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l59_5926


namespace NUMINAMATH_CALUDE_cube_root_of_1331_l59_5906

theorem cube_root_of_1331 (y : ℝ) (h1 : y > 0) (h2 : y^3 = 1331) : y = 11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_1331_l59_5906


namespace NUMINAMATH_CALUDE_cubic_function_sign_properties_l59_5905

/-- Given a cubic function with three real roots, prove specific sign properties -/
theorem cubic_function_sign_properties 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = x^3 - 6*x^2 + 9*x - a*b*c)
  (h2 : a < b ∧ b < c)
  (h3 : f a = 0 ∧ f b = 0 ∧ f c = 0) :
  f 0 * f 1 < 0 ∧ f 0 * f 3 > 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_sign_properties_l59_5905


namespace NUMINAMATH_CALUDE_x_range_l59_5994

theorem x_range (x : ℝ) : (x^2 - 4 < 0 ∨ |x| = 2) → x ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l59_5994


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l59_5917

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  eq : ℝ → ℝ → Prop := λ x y => x^2 / a^2 + y^2 / b^2 = 1
  vertex : ℝ × ℝ := (0, -1)
  focus_distance : ℝ := 3

/-- The line that intersects the ellipse -/
structure IntersectingLine where
  k : ℝ
  m : ℝ
  h_k : k ≠ 0
  eq : ℝ → ℝ → Prop := λ x y => y = k * x + m

/-- Main theorem about the ellipse and intersecting line -/
theorem ellipse_and_line_properties (e : Ellipse) (l : IntersectingLine) :
  (e.eq = λ x y => x^2 / 3 + y^2 = 1) ∧
  (∀ M N : ℝ × ℝ, e.eq M.1 M.2 → e.eq N.1 N.2 → l.eq M.1 M.2 → l.eq N.1 N.2 → M ≠ N →
    (dist M e.vertex = dist N e.vertex) → (1/2 < l.m ∧ l.m < 2)) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_line_properties_l59_5917


namespace NUMINAMATH_CALUDE_terez_cows_l59_5929

theorem terez_cows (total : ℕ) (females : ℕ) (pregnant : ℕ) : 
  2 * females = total → 
  2 * pregnant = females → 
  pregnant = 11 → 
  total = 44 := by
sorry

end NUMINAMATH_CALUDE_terez_cows_l59_5929


namespace NUMINAMATH_CALUDE_ellipse_property_l59_5920

-- Define the basic concepts
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a moving point
def MovingPoint := ℝ → Point

-- Define the concept of an ellipse
def is_ellipse (trajectory : MovingPoint) : Prop := sorry

-- Define the concept of constant sum of distances
def constant_sum_distances (trajectory : MovingPoint) (f1 f2 : Point) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, distance (trajectory t) f1 + distance (trajectory t) f2 = k

-- State the theorem
theorem ellipse_property :
  (∀ trajectory : MovingPoint, ∀ f1 f2 : Point,
    is_ellipse trajectory → constant_sum_distances trajectory f1 f2) ∧
  (∃ trajectory : MovingPoint, ∃ f1 f2 : Point,
    constant_sum_distances trajectory f1 f2 ∧ ¬is_ellipse trajectory) :=
sorry

end NUMINAMATH_CALUDE_ellipse_property_l59_5920


namespace NUMINAMATH_CALUDE_oilseed_germination_theorem_l59_5934

/-- The average germination rate of oilseeds -/
def average_germination_rate : ℝ := 0.96

/-- The total number of oilseeds -/
def total_oilseeds : ℕ := 2000

/-- The number of oilseeds that cannot germinate -/
def non_germinating_oilseeds : ℕ := 80

/-- Theorem stating that given the average germination rate,
    approximately 80 out of 2000 oilseeds cannot germinate -/
theorem oilseed_germination_theorem :
  ⌊(1 - average_germination_rate) * total_oilseeds⌋ = non_germinating_oilseeds :=
sorry

end NUMINAMATH_CALUDE_oilseed_germination_theorem_l59_5934


namespace NUMINAMATH_CALUDE_power_product_equals_one_l59_5927

theorem power_product_equals_one : (0.25 ^ 2023) * (4 ^ 2023) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_one_l59_5927


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l59_5987

theorem inverse_proposition_false : ¬ (∀ a b : ℝ, a^2 = b^2 → a = b) := by sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l59_5987


namespace NUMINAMATH_CALUDE_union_when_m_neg_one_subset_iff_m_range_disjoint_iff_m_range_l59_5969

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = {x | -2 < x ∧ x < 3} := by sorry

-- Theorem 2
theorem subset_iff_m_range :
  ∀ m, A ⊆ B m ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem disjoint_iff_m_range :
  ∀ m, A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end NUMINAMATH_CALUDE_union_when_m_neg_one_subset_iff_m_range_disjoint_iff_m_range_l59_5969


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l59_5944

theorem cubic_equation_solution (x y z : ℕ) : 
  x^3 + 4*y^3 = 16*z^3 + 4*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l59_5944


namespace NUMINAMATH_CALUDE_parallel_lines_parameter_sum_l59_5947

/-- Given two parallel lines with a specific distance between them, prove that the sum of their parameters is either 3 or -3. -/
theorem parallel_lines_parameter_sum (n m : ℝ) : 
  (∀ x y : ℝ, 2 * x + y + n = 0 ↔ 4 * x + m * y - 4 = 0) →  -- parallelism condition
  (∃ d : ℝ, d = (3 / 5) * Real.sqrt 5 ∧ 
    d = |n + 2| / Real.sqrt 5) →  -- distance condition
  m = 2 →  -- parallelism implies m = 2
  (m + n = 3 ∨ m + n = -3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_parameter_sum_l59_5947


namespace NUMINAMATH_CALUDE_max_value_5x_3y_l59_5980

theorem max_value_5x_3y (x y : ℝ) (h : x^2 + y^2 = 10*x + 8*y + 10) :
  ∃ (M : ℝ), M = 105 ∧ 5*x + 3*y ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 10*x₀ + 8*y₀ + 10 ∧ 5*x₀ + 3*y₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_5x_3y_l59_5980


namespace NUMINAMATH_CALUDE_wallet_cost_proof_l59_5999

/-- The cost of a pair of sneakers -/
def sneaker_cost : ℕ := 100

/-- The cost of a backpack -/
def backpack_cost : ℕ := 100

/-- The cost of a pair of jeans -/
def jeans_cost : ℕ := 50

/-- The total amount spent by Leonard and Michael -/
def total_spent : ℕ := 450

/-- The cost of the wallet -/
def wallet_cost : ℕ := 50

theorem wallet_cost_proof :
  wallet_cost + 2 * sneaker_cost + backpack_cost + 2 * jeans_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_wallet_cost_proof_l59_5999


namespace NUMINAMATH_CALUDE_oranges_in_bin_l59_5914

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 40 → thrown_away = 37 → final = 10 → final - (initial - thrown_away) = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_bin_l59_5914


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l59_5946

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)}

def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem set_intersection_theorem : N ∩ (U \ M) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l59_5946


namespace NUMINAMATH_CALUDE_wyatts_money_l59_5941

theorem wyatts_money (bread_quantity : ℕ) (juice_quantity : ℕ) 
  (bread_price : ℕ) (juice_price : ℕ) (money_left : ℕ) :
  bread_quantity = 5 →
  juice_quantity = 4 →
  bread_price = 5 →
  juice_price = 2 →
  money_left = 41 →
  bread_quantity * bread_price + juice_quantity * juice_price + money_left = 74 :=
by sorry

end NUMINAMATH_CALUDE_wyatts_money_l59_5941


namespace NUMINAMATH_CALUDE_solve_ice_cream_problem_l59_5997

def ice_cream_problem (aaron_savings : ℚ) (carson_savings : ℚ) (dinner_bill_ratio : ℚ) 
  (ice_cream_cost_per_scoop : ℚ) (change_per_person : ℚ) : Prop :=
  let total_savings := aaron_savings + carson_savings
  let dinner_cost := dinner_bill_ratio * total_savings
  let remaining_money := total_savings - dinner_cost
  let ice_cream_total_cost := remaining_money - 2 * change_per_person
  let total_scoops := ice_cream_total_cost / ice_cream_cost_per_scoop
  (total_scoops / 2 : ℚ) = 6

theorem solve_ice_cream_problem :
  ice_cream_problem 40 40 (3/4) (3/2) 1 :=
by
  sorry

#check solve_ice_cream_problem

end NUMINAMATH_CALUDE_solve_ice_cream_problem_l59_5997


namespace NUMINAMATH_CALUDE_tan_435_degrees_l59_5932

theorem tan_435_degrees : Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_435_degrees_l59_5932


namespace NUMINAMATH_CALUDE_pedal_to_original_triangle_l59_5981

/-- Given the sides of a pedal triangle, calculate the sides of the original triangle --/
theorem pedal_to_original_triangle 
  (a₁ b₁ c₁ : ℝ) 
  (h_pos : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁) :
  ∃ (a b c : ℝ),
    let s₁ := (a₁ + b₁ + c₁) / 2
    a = a₁ * Real.sqrt (b₁ * c₁ / ((s₁ - b₁) * (s₁ - c₁))) ∧
    b = b₁ * Real.sqrt (a₁ * c₁ / ((s₁ - a₁) * (s₁ - c₁))) ∧
    c = c₁ * Real.sqrt (a₁ * b₁ / ((s₁ - a₁) * (s₁ - b₁))) :=
by
  sorry


end NUMINAMATH_CALUDE_pedal_to_original_triangle_l59_5981


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l59_5984

theorem washing_machine_capacity 
  (shirts : ℕ) 
  (sweaters : ℕ) 
  (loads : ℕ) 
  (h1 : shirts = 19) 
  (h2 : sweaters = 8) 
  (h3 : loads = 3) : 
  (shirts + sweaters) / loads = 9 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l59_5984


namespace NUMINAMATH_CALUDE_point_of_tangency_parabolas_l59_5928

/-- The point of tangency of two parabolas -/
theorem point_of_tangency_parabolas :
  let f (x : ℝ) := x^2 + 8*x + 15
  let g (y : ℝ) := y^2 + 16*y + 63
  let point : ℝ × ℝ := (-7/2, -15/2)
  (f (point.1) = point.2 ∧ g (point.2) = point.1) ∧
  ∀ x y : ℝ, (f x = y ∧ g y = x) → (x, y) = point :=
by sorry


end NUMINAMATH_CALUDE_point_of_tangency_parabolas_l59_5928


namespace NUMINAMATH_CALUDE_completing_square_transformation_l59_5916

theorem completing_square_transformation (x : ℝ) : 
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l59_5916


namespace NUMINAMATH_CALUDE_burger_composition_l59_5976

theorem burger_composition (total_weight filler_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_burger_composition_l59_5976


namespace NUMINAMATH_CALUDE_circle_radius_circle_radius_proof_l59_5900

/-- The radius of a circle with center (2, -3) passing through (5, -7) is 5 -/
theorem circle_radius : ℝ → Prop :=
  fun r : ℝ =>
    let center : ℝ × ℝ := (2, -3)
    let point : ℝ × ℝ := (5, -7)
    (center.1 - point.1)^2 + (center.2 - point.2)^2 = r^2 → r = 5

/-- Proof of the theorem -/
theorem circle_radius_proof : circle_radius 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_circle_radius_proof_l59_5900


namespace NUMINAMATH_CALUDE_multiply_65_55_l59_5982

theorem multiply_65_55 : 65 * 55 = 3575 := by sorry

end NUMINAMATH_CALUDE_multiply_65_55_l59_5982


namespace NUMINAMATH_CALUDE_reflected_hyperbola_l59_5996

/-- Given a hyperbola with equation xy = 1 reflected over the line y = 2x,
    the resulting hyperbola has the equation 12y² + 7xy - 12x² = 25 -/
theorem reflected_hyperbola (x y : ℝ) :
  (∃ x₀ y₀, x₀ * y₀ = 1 ∧ 
   ∃ x₁ y₁, y₁ = 2 * x₁ ∧
   ∃ x₂ y₂, (x₂ - x₀) = (y₁ - y₀) ∧ (y₂ - y₀) = -(x₁ - x₀) ∧
   x = x₂ ∧ y = y₂) →
  12 * y^2 + 7 * x * y - 12 * x^2 = 25 :=
by sorry


end NUMINAMATH_CALUDE_reflected_hyperbola_l59_5996


namespace NUMINAMATH_CALUDE_river_problem_solution_l59_5909

/-- Represents the problem of a boat traveling along a river -/
structure RiverProblem where
  total_distance : ℝ
  total_time : ℝ
  upstream_distance : ℝ
  downstream_distance : ℝ
  hTotalDistance : total_distance = 10
  hTotalTime : total_time = 5
  hEqualTime : upstream_distance / downstream_distance = 2 / 3

/-- Solution to the river problem -/
structure RiverSolution where
  current_speed : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Theorem stating the solution to the river problem -/
theorem river_problem_solution (p : RiverProblem) : 
  ∃ (s : RiverSolution), 
    s.current_speed = 5 / 12 ∧ 
    s.upstream_time = 3 ∧ 
    s.downstream_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_river_problem_solution_l59_5909


namespace NUMINAMATH_CALUDE_product_of_fractions_l59_5930

theorem product_of_fractions :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l59_5930


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l59_5953

-- Define the rhombus
structure Rhombus where
  perimeter : ℝ
  diagonal_difference : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

-- State the theorem
theorem rhombus_diagonals (r : Rhombus) :
  r.perimeter = 100 ∧ r.diagonal_difference = 34 →
  r.diagonal1 = 14 ∧ r.diagonal2 = 48 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_diagonals_l59_5953


namespace NUMINAMATH_CALUDE_barbara_has_winning_strategy_l59_5942

/-- A game played on a matrix where two players alternately fill entries --/
structure MatrixGame where
  n : ℕ
  entries : Fin n → Fin n → ℝ

/-- A strategy for the second player in the matrix game --/
def SecondPlayerStrategy (n : ℕ) := 
  (Fin n → Fin n → ℝ) → Fin n → Fin n → ℝ

/-- The determinant of a matrix is zero if two of its rows are identical --/
axiom det_zero_if_identical_rows {n : ℕ} (M : Fin n → Fin n → ℝ) :
  (∃ i j, i ≠ j ∧ (∀ k, M i k = M j k)) → Matrix.det M = 0

/-- The second player can always make two rows identical --/
axiom second_player_can_make_identical_rows (n : ℕ) :
  ∃ (strategy : SecondPlayerStrategy n),
    ∀ (game : MatrixGame),
    game.n = n →
    ∃ i j, i ≠ j ∧ (∀ k, game.entries i k = game.entries j k)

theorem barbara_has_winning_strategy :
  ∃ (strategy : SecondPlayerStrategy 2008),
    ∀ (game : MatrixGame),
    game.n = 2008 →
    Matrix.det game.entries = 0 := by
  sorry

end NUMINAMATH_CALUDE_barbara_has_winning_strategy_l59_5942


namespace NUMINAMATH_CALUDE_bianca_recycling_points_l59_5959

theorem bianca_recycling_points 
  (points_per_bag : ℕ) 
  (total_bags : ℕ) 
  (bags_not_recycled : ℕ) 
  (h1 : points_per_bag = 5)
  (h2 : total_bags = 17)
  (h3 : bags_not_recycled = 8) :
  (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by sorry

end NUMINAMATH_CALUDE_bianca_recycling_points_l59_5959


namespace NUMINAMATH_CALUDE_solution_characterization_l59_5983

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The plane z = x -/
def midplane (p : Point3D) : Prop := p.z = p.x

/-- The sphere with center A and radius r -/
def sphere (A : Point3D) (r : ℝ) (p : Point3D) : Prop :=
  (p.x - A.x)^2 + (p.y - A.y)^2 + (p.z - A.z)^2 = r^2

/-- The set of points satisfying both conditions -/
def solution_set (A : Point3D) (r : ℝ) : Set Point3D :=
  {p : Point3D | sphere A r p ∧ midplane p}

theorem solution_characterization (A : Point3D) (r : ℝ) :
  ∀ p : Point3D, p ∈ solution_set A r ↔ 
    (p.x - A.x)^2 + (p.y - A.y)^2 + (p.x - A.z)^2 = r^2 :=
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l59_5983


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l59_5939

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l59_5939


namespace NUMINAMATH_CALUDE_solution_set_l59_5973

def equation (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8

theorem solution_set : {x : ℝ | equation x} = {7, -2} := by sorry

end NUMINAMATH_CALUDE_solution_set_l59_5973


namespace NUMINAMATH_CALUDE_compound_interest_problem_l59_5924

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem compound_interest_problem :
  let principal : ℝ := 3600
  let rate : ℝ := 0.05
  let time : ℕ := 2
  let final_amount : ℝ := 3969
  compound_interest principal rate time = final_amount := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l59_5924


namespace NUMINAMATH_CALUDE_min_age_difference_proof_l59_5970

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- The number of days in a common year -/
def daysInCommonYear : ℕ := 365

/-- The year Adil was born -/
def adilBirthYear : ℕ := 2015

/-- The year Bav was born -/
def bavBirthYear : ℕ := 2018

/-- The minimum age difference in days between Adil and Bav -/
def minAgeDifference : ℕ := daysInLeapYear + daysInCommonYear + 1

theorem min_age_difference_proof :
  minAgeDifference = 732 :=
sorry

end NUMINAMATH_CALUDE_min_age_difference_proof_l59_5970


namespace NUMINAMATH_CALUDE_sale_price_for_50_percent_profit_l59_5974

/-- Represents the cost and pricing of an article -/
structure Article where
  cost : ℝ
  profit_price : ℝ
  loss_price : ℝ

/-- The conditions of the problem -/
def problem_conditions (a : Article) : Prop :=
  a.profit_price - a.cost = a.cost - a.loss_price ∧
  a.profit_price = 892 ∧
  1005 = 1.5 * a.cost

/-- The theorem to be proved -/
theorem sale_price_for_50_percent_profit (a : Article) 
  (h : problem_conditions a) : 
  1.5 * a.cost = 1005 := by
  sorry

#check sale_price_for_50_percent_profit

end NUMINAMATH_CALUDE_sale_price_for_50_percent_profit_l59_5974


namespace NUMINAMATH_CALUDE_gift_cost_l59_5972

/-- Proves that the cost of the gift is $250 given the specified conditions --/
theorem gift_cost (erika_savings : ℕ) (cake_cost : ℕ) (leftover : ℕ) :
  erika_savings = 155 →
  cake_cost = 25 →
  leftover = 5 →
  ∃ (gift_cost : ℕ), 
    gift_cost = 250 ∧
    erika_savings + gift_cost / 2 = gift_cost + cake_cost + leftover :=
by
  sorry

end NUMINAMATH_CALUDE_gift_cost_l59_5972


namespace NUMINAMATH_CALUDE_complex_multiplication_l59_5931

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (1 - i) * i = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l59_5931


namespace NUMINAMATH_CALUDE_min_value_theorem_l59_5993

theorem min_value_theorem (a b c : ℝ) 
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  ∃ m : ℝ, m = -4 ∧ ∀ k : ℝ, (∃ a' b' c' : ℝ, 
    (∀ x y : ℝ, x + 2*y - 3 ≤ a'*x + b'*y + c' ∧ a'*x + b'*y + c' ≤ x + 2*y + 3) ∧
    k = a' + 2*b' - 3*c') → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l59_5993


namespace NUMINAMATH_CALUDE_smallest_sum_of_ten_numbers_l59_5901

theorem smallest_sum_of_ten_numbers (S : Finset ℕ) : 
  S.card = 10 ∧ 
  (∀ T ⊆ S, T.card = 5 → Even (T.prod id)) ∧
  Odd (S.sum id) →
  65 ≤ S.sum id :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_ten_numbers_l59_5901


namespace NUMINAMATH_CALUDE_smaller_fraction_problem_l59_5910

theorem smaller_fraction_problem (x y : ℚ) 
  (sum_cond : x + y = 7/8)
  (prod_cond : x * y = 1/12) :
  min x y = (7 - Real.sqrt 17) / 16 := by
  sorry

end NUMINAMATH_CALUDE_smaller_fraction_problem_l59_5910


namespace NUMINAMATH_CALUDE_laundry_ratio_l59_5903

def wednesday_loads : ℕ := 6
def thursday_loads : ℕ := 2 * wednesday_loads
def saturday_loads : ℕ := wednesday_loads / 3
def total_loads : ℕ := 26

def friday_loads : ℕ := total_loads - (wednesday_loads + thursday_loads + saturday_loads)

theorem laundry_ratio :
  (friday_loads : ℚ) / thursday_loads = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_laundry_ratio_l59_5903


namespace NUMINAMATH_CALUDE_ellipse_a_value_l59_5938

-- Define the ellipse equation
def ellipse_equation (a x y : ℝ) : Prop := x^2 / a^2 + y^2 / 2 = 1

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem ellipse_a_value :
  ∃ (a : ℝ), 
    (∀ (x y : ℝ), ellipse_equation a x y → 
      ∃ (c : ℝ), c = 2 ∧ a^2 = 2 + c^2) ∧ 
    (∀ (x y : ℝ), parabola_equation x y → 
      ∃ (f : ℝ × ℝ), f = parabola_focus) →
  a = Real.sqrt 6 ∨ a = -Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_a_value_l59_5938


namespace NUMINAMATH_CALUDE_expression_value_l59_5956

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l59_5956


namespace NUMINAMATH_CALUDE_triangle_side_equation_l59_5968

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude equations
def altitude1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def altitude2 (x y : ℝ) : Prop := x + y = 0

-- Define the theorem
theorem triangle_side_equation (ABC : Triangle) 
  (h1 : ABC.A = (1, 2))
  (h2 : altitude1 (ABC.B.1) (ABC.B.2) ∨ altitude1 (ABC.C.1) (ABC.C.2))
  (h3 : altitude2 (ABC.B.1) (ABC.B.2) ∨ altitude2 (ABC.C.1) (ABC.C.2)) :
  ∃ (a b c : ℝ), a * ABC.B.1 + b * ABC.B.2 + c = 0 ∧
                 a * ABC.C.1 + b * ABC.C.2 + c = 0 ∧
                 (a, b, c) = (2, 3, 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_equation_l59_5968


namespace NUMINAMATH_CALUDE_shirt_discount_problem_l59_5990

theorem shirt_discount_problem (original_price : ℝ) : 
  (0.75 * (0.75 * original_price) = 19) → 
  original_price = 33.78 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_problem_l59_5990


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l59_5989

theorem polynomial_division_quotient : 
  let dividend := fun x : ℚ => 10 * x^3 - 5 * x^2 + 8 * x - 9
  let divisor := fun x : ℚ => 3 * x - 4
  let quotient := fun x : ℚ => (10/3) * x^2 - (55/9) * x - 172/27
  ∀ x : ℚ, dividend x = divisor x * quotient x + (-971/27) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l59_5989


namespace NUMINAMATH_CALUDE_cinema_seating_arrangement_l59_5911

def number_of_arrangements (n : ℕ) (must_together : ℕ) (must_not_together : ℕ) : ℕ :=
  (must_together.factorial * (n - must_together + 1).factorial) -
  (must_together.factorial * must_not_together.factorial * (n - must_together - must_not_together + 2).factorial)

theorem cinema_seating_arrangement :
  number_of_arrangements 6 2 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seating_arrangement_l59_5911


namespace NUMINAMATH_CALUDE_painted_cubes_count_l59_5935

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  red_faces : Nat
  blue_faces : Nat

/-- Calculates the number of painted unit cubes in a PaintedCube -/
def num_painted_cubes (cube : PaintedCube) : Nat :=
  cube.size ^ 3 - (cube.size - 2) ^ 3

/-- Theorem: In a 5x5x5 cube with 2 red faces and 4 blue faces, 101 unit cubes are painted -/
theorem painted_cubes_count (cube : PaintedCube) 
  (h_size : cube.size = 5)
  (h_red : cube.red_faces = 2)
  (h_blue : cube.blue_faces = 4) :
  num_painted_cubes cube = 101 := by
  sorry

#check painted_cubes_count

end NUMINAMATH_CALUDE_painted_cubes_count_l59_5935


namespace NUMINAMATH_CALUDE_heather_aprons_tomorrow_l59_5967

/-- The number of aprons Heather should sew tomorrow -/
def aprons_tomorrow (total : ℕ) (initial : ℕ) (today_multiplier : ℕ) : ℕ :=
  (total - (initial + today_multiplier * initial)) / 2

/-- Theorem: Given the conditions, Heather should sew 49 aprons tomorrow -/
theorem heather_aprons_tomorrow :
  aprons_tomorrow 150 13 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_heather_aprons_tomorrow_l59_5967


namespace NUMINAMATH_CALUDE_crocodile_coloring_exists_l59_5964

/-- A coloring function for an infinite checkerboard -/
def ColoringFunction := ℤ → ℤ → Fin 2

/-- The "crocodile" move on a checkerboard -/
def crocodileMove (m n : ℤ) (x y : ℤ) : Set (ℤ × ℤ) :=
  {(x + m, y + n), (x + m, y - n), (x - m, y + n), (x - m, y - n),
   (x + n, y + m), (x + n, y - m), (x - n, y + m), (x - n, y - m)}

/-- Theorem: For any m and n, there exists a coloring function such that
    any two squares connected by a crocodile move have different colors -/
theorem crocodile_coloring_exists (m n : ℤ) :
  ∃ (f : ColoringFunction),
    ∀ (x y : ℤ), ∀ (x' y' : ℤ), (x', y') ∈ crocodileMove m n x y →
      f x y ≠ f x' y' := by
  sorry

end NUMINAMATH_CALUDE_crocodile_coloring_exists_l59_5964


namespace NUMINAMATH_CALUDE_debugging_time_l59_5902

theorem debugging_time (total_hours : ℝ) (flow_chart_fraction : ℝ) (coding_fraction : ℝ)
  (h1 : total_hours = 48)
  (h2 : flow_chart_fraction = 1/4)
  (h3 : coding_fraction = 3/8)
  (h4 : flow_chart_fraction + coding_fraction < 1) :
  total_hours * (1 - flow_chart_fraction - coding_fraction) = 18 :=
by sorry

end NUMINAMATH_CALUDE_debugging_time_l59_5902


namespace NUMINAMATH_CALUDE_commission_calculation_l59_5985

/-- Calculates the commission earned from selling a coupe and an SUV --/
theorem commission_calculation (coupe_price : ℝ) (suv_price_multiplier : ℝ) (commission_rate : ℝ) :
  coupe_price = 30000 →
  suv_price_multiplier = 2 →
  commission_rate = 0.02 →
  coupe_price * suv_price_multiplier * commission_rate + coupe_price * commission_rate = 1800 := by
  sorry

end NUMINAMATH_CALUDE_commission_calculation_l59_5985


namespace NUMINAMATH_CALUDE_g_at_six_l59_5961

def g (x : ℝ) : ℝ := 2*x^4 - 19*x^3 + 30*x^2 - 12*x - 72

theorem g_at_six : g 6 = 288 := by
  sorry

end NUMINAMATH_CALUDE_g_at_six_l59_5961


namespace NUMINAMATH_CALUDE_probability_not_red_blue_purple_l59_5950

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_not_red_blue_purple :
  let favorable_outcomes := total_balls - (red_balls + blue_balls + purple_balls)
  (favorable_outcomes : ℚ) / total_balls = 35 / 48 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_blue_purple_l59_5950


namespace NUMINAMATH_CALUDE_largest_k_for_g_range_l59_5962

/-- The function g(x) defined as x^2 - 7x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + k

/-- Theorem stating that the largest value of k such that 4 is in the range of g(x) is 65/4 -/
theorem largest_k_for_g_range : 
  (∃ (k : ℝ), ∀ (k' : ℝ), (∃ (x : ℝ), g k' x = 4) → k' ≤ k) ∧ 
  (∃ (x : ℝ), g (65/4) x = 4) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_for_g_range_l59_5962
