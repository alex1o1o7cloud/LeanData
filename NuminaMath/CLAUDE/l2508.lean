import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2508_250872

theorem equation_solution :
  {x : ℂ | x^4 - 81 = 0} = {3, -3, 3*I, -3*I} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2508_250872


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2508_250812

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 6*x*y - 4*x + 9*y - 366 = 0 ↔ (x = 3 ∧ y = 14) ∨ (x = -24 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2508_250812


namespace NUMINAMATH_CALUDE_exists_number_not_divisible_by_3_with_digit_product_3_l2508_250889

def numbers : List Nat := [4621, 4631, 4641, 4651, 4661]

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def is_divisible_by_3 (n : Nat) : Prop :=
  n % 3 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem exists_number_not_divisible_by_3_with_digit_product_3 :
  ∃ n ∈ numbers, ¬(is_divisible_by_3 n) ∧ (units_digit n) * (tens_digit n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_not_divisible_by_3_with_digit_product_3_l2508_250889


namespace NUMINAMATH_CALUDE_daniel_noodles_left_l2508_250857

/-- Given that Daniel initially had 66 noodles and gave 12 noodles to William,
    prove that he now has 54 noodles. -/
theorem daniel_noodles_left (initial : ℕ) (given : ℕ) (h1 : initial = 66) (h2 : given = 12) :
  initial - given = 54 := by sorry

end NUMINAMATH_CALUDE_daniel_noodles_left_l2508_250857


namespace NUMINAMATH_CALUDE_usual_time_to_school_l2508_250804

theorem usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_rate > 0 →
  usual_time > 0 →
  (5/4 * usual_rate) * (usual_time - 4) = usual_rate * usual_time →
  usual_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_school_l2508_250804


namespace NUMINAMATH_CALUDE_ap_square_identity_l2508_250877

/-- Three consecutive terms of an arithmetic progression -/
structure ArithmeticProgressionTerms (α : Type*) [Add α] [Sub α] where
  a : α
  b : α
  c : α
  is_ap : b - a = c - b

/-- Theorem: For any three consecutive terms of an arithmetic progression,
    a^2 + 8bc = (2b + c)^2 -/
theorem ap_square_identity {α : Type*} [CommRing α] (terms : ArithmeticProgressionTerms α) :
  terms.a ^ 2 + 8 * terms.b * terms.c = (2 * terms.b + terms.c) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ap_square_identity_l2508_250877


namespace NUMINAMATH_CALUDE_square_window_side_length_l2508_250868

/-- Given a square window opening formed by two rectangular frames, 
    prove that the side length of the square is 5 when the perimeter 
    of the left frame is 14 and the perimeter of the right frame is 16. -/
theorem square_window_side_length 
  (a : ℝ) -- side length of the square window
  (b : ℝ) -- width of the left rectangular frame
  (h1 : 2 * a + 2 * b = 14) -- perimeter of the left frame
  (h2 : 4 * a - 2 * b = 16) -- perimeter of the right frame
  : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_window_side_length_l2508_250868


namespace NUMINAMATH_CALUDE_angle_supplement_l2508_250859

theorem angle_supplement (angle : ℝ) : 
  (90 - angle = angle - 18) → (180 - angle = 126) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l2508_250859


namespace NUMINAMATH_CALUDE_unique_solution_pqr_l2508_250888

theorem unique_solution_pqr : 
  ∀ p q r : ℕ,
  Prime p → Prime q → Even r → r > 0 →
  p^3 + q^2 = 4*r^2 + 45*r + 103 →
  p = 7 ∧ q = 2 ∧ r = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_pqr_l2508_250888


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l2508_250852

/-- Represents a lamp in a room -/
structure Lamp where
  room : Nat
  state : Bool

/-- Represents a switch controlling a pair of lamps -/
structure Switch where
  lamp1 : Lamp
  lamp2 : Lamp

/-- Configuration of lamps and switches -/
structure Configuration (k : Nat) where
  lamps : Fin (6 * k) → Lamp
  switches : Fin (3 * k) → Switch
  rooms : Fin (2 * k)

/-- Predicate to check if a room has at least one lamp on and one off -/
def validRoom (config : Configuration k) (room : Fin (2 * k)) : Prop :=
  ∃ (l1 l2 : Fin (6 * k)), 
    (config.lamps l1).room = room ∧ 
    (config.lamps l2).room = room ∧ 
    (config.lamps l1).state = true ∧ 
    (config.lamps l2).state = false

/-- Main theorem statement -/
theorem exists_valid_configuration (k : Nat) (h : k > 0) : 
  ∃ (config : Configuration k), ∀ (room : Fin (2 * k)), validRoom config room :=
sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l2508_250852


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_25_l2508_250876

theorem three_digit_perfect_cube_divisible_by_25 : 
  ∃! (n : ℕ), 100 ≤ 125 * n^3 ∧ 125 * n^3 ≤ 999 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_25_l2508_250876


namespace NUMINAMATH_CALUDE_sofa_bench_arrangement_l2508_250825

/-- The number of ways to arrange n indistinguishable objects of one type
    and k indistinguishable objects of another type in a row -/
def arrangements (n k : ℕ) : ℕ := Nat.choose (n + k) n

/-- Theorem: There are 210 distinct ways to arrange 6 indistinguishable objects
    of one type and 4 indistinguishable objects of another type in a row -/
theorem sofa_bench_arrangement : arrangements 6 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sofa_bench_arrangement_l2508_250825


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2508_250886

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, n > 20 ∧ n % 6 = 5 ∧ n % 7 = 3 ∧ 
  ∀ m : ℕ, m > 20 ∧ m % 6 = 5 ∧ m % 7 = 3 → n ≤ m :=
by
  use 59
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2508_250886


namespace NUMINAMATH_CALUDE_fraction_expression_equality_l2508_250890

theorem fraction_expression_equality : 
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 9) = 1548 / 805 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_equality_l2508_250890


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l2508_250824

-- Define a cube structure
structure Cube where
  size : ℕ
  unit_cubes : ℕ

-- Define a plane that bisects the diagonal
structure BisectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

-- Define the function to count intersected cubes
def count_intersected_cubes (c : Cube) (p : BisectingPlane) : ℕ :=
  sorry

-- Theorem statement
theorem intersected_cubes_count 
  (c : Cube) 
  (p : BisectingPlane) 
  (h1 : c.size = 4) 
  (h2 : c.unit_cubes = 64) 
  (h3 : p.perpendicular_to_diagonal = true) 
  (h4 : p.bisects_diagonal = true) : 
  count_intersected_cubes c p = 24 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l2508_250824


namespace NUMINAMATH_CALUDE_john_payment_l2508_250845

def hearing_aid_cost : ℝ := 2500
def insurance_coverage_percent : ℝ := 80
def number_of_hearing_aids : ℕ := 2

theorem john_payment (total_cost : ℝ) (insurance_payment : ℝ) (john_payment : ℝ) :
  total_cost = hearing_aid_cost * number_of_hearing_aids →
  insurance_payment = (insurance_coverage_percent / 100) * total_cost →
  john_payment = total_cost - insurance_payment →
  john_payment = 1000 := by sorry

end NUMINAMATH_CALUDE_john_payment_l2508_250845


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2508_250849

theorem absolute_value_equality (x : ℝ) (h : x > 2) :
  |x - Real.sqrt ((x - 3)^2)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2508_250849


namespace NUMINAMATH_CALUDE_generalized_inequality_l2508_250847

theorem generalized_inequality (x : ℝ) (n : ℕ) (h : x > 0) :
  x^n + n/x > n + 1 := by sorry

end NUMINAMATH_CALUDE_generalized_inequality_l2508_250847


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l2508_250850

theorem expression_equals_negative_one (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : b ≠ a) (hd : b ≠ -a) :
  (a / (a + b) + b / (a - b)) / (b / (a + b) - a / (a - b)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l2508_250850


namespace NUMINAMATH_CALUDE_beong_gun_number_l2508_250848

theorem beong_gun_number : ∃ x : ℚ, (x / 11 + 156 = 178) ∧ (x = 242) := by
  sorry

end NUMINAMATH_CALUDE_beong_gun_number_l2508_250848


namespace NUMINAMATH_CALUDE_complex_real_part_theorem_l2508_250821

theorem complex_real_part_theorem (a : ℝ) : 
  (((a - Complex.I) / (3 + Complex.I)).re = 1/2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_part_theorem_l2508_250821


namespace NUMINAMATH_CALUDE_range_of_m_l2508_250858

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m - 1}

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (A ∩ (U \ B m) = A) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2508_250858


namespace NUMINAMATH_CALUDE_movie_book_difference_l2508_250827

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 17

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 11

/-- Theorem: The difference between the number of movies and books in the 'crazy silly school' series is 6 -/
theorem movie_book_difference : num_movies - num_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_movie_book_difference_l2508_250827


namespace NUMINAMATH_CALUDE_furniture_production_l2508_250805

theorem furniture_production (total_wood : ℕ) (table_wood : ℕ) (chair_wood : ℕ) (tables_made : ℕ) :
  total_wood = 672 →
  table_wood = 12 →
  chair_wood = 8 →
  tables_made = 24 →
  (total_wood - tables_made * table_wood) / chair_wood = 48 :=
by sorry

end NUMINAMATH_CALUDE_furniture_production_l2508_250805


namespace NUMINAMATH_CALUDE_square_of_difference_formula_l2508_250895

theorem square_of_difference_formula (m n : ℝ) : 
  ¬ ∃ (a b : ℝ), (m - n) * (-m + n) = (a - b) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_formula_l2508_250895


namespace NUMINAMATH_CALUDE_motion_analysis_l2508_250840

-- Define the motion function
def s (t : ℝ) : ℝ := t^2 + 2*t - 3

-- Define velocity as the derivative of s
def v (t : ℝ) : ℝ := 2*t + 2

-- Define acceleration as the derivative of v
def a : ℝ := 2

theorem motion_analysis :
  v 2 = 6 ∧ a = 2 :=
sorry

end NUMINAMATH_CALUDE_motion_analysis_l2508_250840


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2508_250829

-- Define the curve f(x) = 2x³ - 3x
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem tangent_line_at_origin :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y = m * x) →                   -- Equation of a line through (0,0)
    (∃ (t : ℝ), t ≠ 0 →
      y = f t ∧                     -- Point (t, f(t)) is on the curve
      (f t - 0) / (t - 0) = m) →    -- Slope of secant line
    m = -3                          -- Slope of the tangent line
    :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2508_250829


namespace NUMINAMATH_CALUDE_largest_sales_increase_2011_l2508_250884

-- Define the sales data for each year
def sales : Fin 8 → ℕ
  | 0 => 20
  | 1 => 24
  | 2 => 27
  | 3 => 26
  | 4 => 28
  | 5 => 33
  | 6 => 32
  | 7 => 35

-- Define the function to calculate the sales increase between two consecutive years
def salesIncrease (i : Fin 7) : ℤ :=
  (sales (i.succ : Fin 8) : ℤ) - (sales i : ℤ)

-- Define the theorem to prove
theorem largest_sales_increase_2011 :
  ∃ i : Fin 7, salesIncrease i = 5 ∧
  ∀ j : Fin 7, salesIncrease j ≤ 5 ∧
  (i : ℕ) + 2006 = 2011 :=
by sorry

end NUMINAMATH_CALUDE_largest_sales_increase_2011_l2508_250884


namespace NUMINAMATH_CALUDE_P_less_than_Q_l2508_250834

theorem P_less_than_Q (a : ℝ) (h : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 7) < Real.sqrt (a + 3) + Real.sqrt (a + 4) := by
sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l2508_250834


namespace NUMINAMATH_CALUDE_discount_calculation_l2508_250870

theorem discount_calculation (CP : ℝ) (CP_pos : CP > 0) : 
  let MP := 1.12 * CP
  let SP := 0.99 * CP
  MP - SP = 0.13 * CP := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l2508_250870


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2508_250819

/-- 
For any triangle ABC, we define:
- ha, hb, hc as the altitudes
- ra, rb, rc as the exradii
- r as the inradius
-/
theorem triangle_inequalities (A B C : Point) 
  (ha hb hc : ℝ) (ra rb rc : ℝ) (r : ℝ) :
  (ha > 0 ∧ hb > 0 ∧ hc > 0) →
  (ra > 0 ∧ rb > 0 ∧ rc > 0) →
  (r > 0) →
  (ha * hb * hc ≥ 27 * r^3) ∧ (ra * rb * rc ≥ 27 * r^3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2508_250819


namespace NUMINAMATH_CALUDE_tom_remaining_seashells_l2508_250878

def initial_seashells : ℕ := 5
def seashells_given_away : ℕ := 2

theorem tom_remaining_seashells : 
  initial_seashells - seashells_given_away = 3 := by sorry

end NUMINAMATH_CALUDE_tom_remaining_seashells_l2508_250878


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2508_250867

-- Problem 1
theorem problem_1 (a b : ℝ) : 
  (abs a = 5) → 
  (abs b = 3) → 
  (abs (a - b) = b - a) → 
  ((a - b = -8) ∨ (a - b = -2)) :=
sorry

-- Problem 2
theorem problem_2 (a b c d m : ℝ) :
  (a + b = 0) →
  (c * d = 1) →
  (abs m = 2) →
  (abs (a + b) / m - c * d + m^2 = 3) :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2508_250867


namespace NUMINAMATH_CALUDE_cyclic_fraction_sum_l2508_250818

theorem cyclic_fraction_sum (x y z w t : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ w ≠ x)
  (h_eq : x + 1/y = t ∧ y + 1/z = t ∧ z + 1/w = t ∧ w + 1/x = t) :
  t = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_fraction_sum_l2508_250818


namespace NUMINAMATH_CALUDE_math_competition_score_ratio_l2508_250811

theorem math_competition_score_ratio :
  let sammy_score : ℕ := 20
  let gab_score : ℕ := 2 * sammy_score
  let opponent_score : ℕ := 85
  let total_score : ℕ := opponent_score + 55
  let cher_score : ℕ := total_score - (sammy_score + gab_score)
  (cher_score : ℚ) / (gab_score : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_score_ratio_l2508_250811


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2508_250864

/-- Given that Dan spent $13 in total on a candy bar and a chocolate, 
    and the chocolate costs $6, prove that the candy bar costs $7. -/
theorem candy_bar_cost (total_spent : ℕ) (chocolate_cost : ℕ) (candy_bar_cost : ℕ) : 
  total_spent = 13 → chocolate_cost = 6 → candy_bar_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2508_250864


namespace NUMINAMATH_CALUDE_constant_product_l2508_250828

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define curve C
def curve_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define points A, P, and B on curve C
def point_on_C (p : ℝ × ℝ) : Prop := curve_C p.1 p.2

theorem constant_product (A P B S T : ℝ × ℝ) 
  (hA : point_on_C A) (hP : point_on_C P) (hB : point_on_C B)
  (hB_sym : B.1 = A.1 ∧ B.2 = -A.2)
  (hS : S.2 = 0 ∧ (P.2 - A.2) * (S.1 - A.1) = (P.1 - A.1) * (S.2 - A.2))
  (hT : T.2 = 0 ∧ (P.2 - B.2) * (T.1 - B.1) = (P.1 - B.1) * (T.2 - B.2))
  (hP_ne_A : P.1 ≠ A.1 ∨ P.2 ≠ A.2) :
  |S.1| * |T.1| = 9 :=
sorry

end NUMINAMATH_CALUDE_constant_product_l2508_250828


namespace NUMINAMATH_CALUDE_friend_spent_ten_l2508_250856

def lunch_problem (total : ℝ) (difference : ℝ) : Prop :=
  ∃ (your_cost friend_cost : ℝ),
    your_cost + friend_cost = total ∧
    friend_cost = your_cost + difference ∧
    friend_cost = 10

theorem friend_spent_ten :
  lunch_problem 17 3 :=
sorry

end NUMINAMATH_CALUDE_friend_spent_ten_l2508_250856


namespace NUMINAMATH_CALUDE_snow_probability_l2508_250860

theorem snow_probability (p : ℝ) (h : p = 3/4) : 
  1 - (1 - p)^5 = 1023/1024 := by
sorry

end NUMINAMATH_CALUDE_snow_probability_l2508_250860


namespace NUMINAMATH_CALUDE_solve_inequality_max_a_value_l2508_250802

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for part I
theorem solve_inequality :
  ∀ x : ℝ, f x > 4 ↔ x < -1.5 ∨ x > 2.5 := by sorry

-- Theorem for part II
theorem max_a_value :
  ∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∀ b : ℝ, (∀ x : ℝ, f x ≥ b) → b ≤ a) ∧ a = 3 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_max_a_value_l2508_250802


namespace NUMINAMATH_CALUDE_determinant_transformation_l2508_250882

theorem determinant_transformation (x y z w : ℝ) :
  (x * w - y * z = 3) →
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end NUMINAMATH_CALUDE_determinant_transformation_l2508_250882


namespace NUMINAMATH_CALUDE_total_fruit_punch_l2508_250817

def orange_punch : Real := 4.5
def cherry_punch : Real := 2 * orange_punch
def apple_juice : Real := cherry_punch - 1.5
def pineapple_juice : Real := 3
def grape_punch : Real := apple_juice + 0.5 * apple_juice

theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_punch_l2508_250817


namespace NUMINAMATH_CALUDE_electricity_bill_theorem_l2508_250894

/-- Represents a meter reading with three tariff zones -/
structure MeterReading where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Represents tariff rates for electricity -/
structure TariffRates where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Calculates the electricity bill based on meter readings and tariff rates -/
def calculate_bill (previous : MeterReading) (current : MeterReading) (rates : TariffRates) : ℝ :=
  (current.peak - previous.peak) * rates.peak +
  (current.night - previous.night) * rates.night +
  (current.half_peak - previous.half_peak) * rates.half_peak

/-- Theorem: Maximum additional payment and expected difference -/
theorem electricity_bill_theorem 
  (previous : MeterReading)
  (current : MeterReading)
  (rates : TariffRates)
  (actual_payment : ℝ)
  (h1 : rates.peak = 4.03)
  (h2 : rates.night = 1.01)
  (h3 : rates.half_peak = 3.39)
  (h4 : actual_payment = 660.72)
  (h5 : current.peak > previous.peak)
  (h6 : current.night > previous.night)
  (h7 : current.half_peak > previous.half_peak) :
  ∃ (max_additional_payment expected_difference : ℝ),
    max_additional_payment = 397.34 ∧
    expected_difference = 19.30 :=
sorry

end NUMINAMATH_CALUDE_electricity_bill_theorem_l2508_250894


namespace NUMINAMATH_CALUDE_no_rain_time_l2508_250809

theorem no_rain_time (total_time rain_time : ℕ) (h1 : total_time = 8) (h2 : rain_time = 2) :
  total_time - rain_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_time_l2508_250809


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2508_250871

def sequence_a (n : ℕ) : ℝ := 2^n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sequence_T (n : ℕ) : ℝ := n * 2^(n+1) - 2^(n+1) + 2

theorem smallest_n_for_inequality :
  ∀ n : ℕ, (∀ k < n, sequence_T k - k * 2^(k+1) + 50 ≥ 0) ∧
           (sequence_T n - n * 2^(n+1) + 50 < 0) →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2508_250871


namespace NUMINAMATH_CALUDE_remainder_101_35_mod_100_l2508_250883

theorem remainder_101_35_mod_100 : 101^35 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_35_mod_100_l2508_250883


namespace NUMINAMATH_CALUDE_min_value_expression_l2508_250880

theorem min_value_expression (x y : ℝ) : (x*y + 1)^2 + (x^2 + y^2)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2508_250880


namespace NUMINAMATH_CALUDE_subadditive_sequence_inequality_l2508_250813

/-- A non-negative sequence satisfying the subadditivity property -/
def SubadditiveSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧ (∀ m n, a (m + n) ≤ a m + a n)

/-- The main theorem to be proved -/
theorem subadditive_sequence_inequality (a : ℕ → ℝ) (h : SubadditiveSequence a) :
    ∀ m n, m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m := by
  sorry

end NUMINAMATH_CALUDE_subadditive_sequence_inequality_l2508_250813


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l2508_250861

theorem divisibility_by_eleven (n : ℤ) : 
  (11 : ℤ) ∣ ((n + 11)^2 - n^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l2508_250861


namespace NUMINAMATH_CALUDE_officers_on_duty_l2508_250887

theorem officers_on_duty (total_female_officers : ℕ) 
  (female_on_duty_percentage : ℚ) (female_ratio_on_duty : ℚ) :
  total_female_officers = 300 →
  female_on_duty_percentage = 2/5 →
  female_ratio_on_duty = 1/2 →
  (female_on_duty_percentage * total_female_officers : ℚ) / female_ratio_on_duty = 240 := by
  sorry

end NUMINAMATH_CALUDE_officers_on_duty_l2508_250887


namespace NUMINAMATH_CALUDE_students_in_both_chorus_and_band_l2508_250800

theorem students_in_both_chorus_and_band :
  ∀ (total chorus band neither both : ℕ),
    total = 50 →
    chorus = 18 →
    band = 26 →
    neither = 8 →
    total = chorus + band - both + neither →
    both = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_both_chorus_and_band_l2508_250800


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l2508_250865

theorem power_of_two_plus_one (b m n : ℕ) : 
  b > 1 → 
  m > n → 
  (∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) → 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l2508_250865


namespace NUMINAMATH_CALUDE_star_calculation_l2508_250879

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ (4 ⋆ 6)) = -152877 -/
theorem star_calculation : star 2 (star 3 (star 4 6)) = -152877 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l2508_250879


namespace NUMINAMATH_CALUDE_target_destruction_probabilities_l2508_250830

/-- Represents the probability of a person hitting a target -/
def HitProbability := Fin 2 → Fin 2 → ℚ

/-- The probability of person A hitting targets -/
def probA : HitProbability := fun i j => 
  if i = j then 1/2 else 1/2

/-- The probability of person B hitting targets -/
def probB : HitProbability := fun i j => 
  if i = 0 ∧ j = 0 then 1/3
  else if i = 1 ∧ j = 1 then 2/5
  else 0

/-- The probability of a target being destroyed -/
def targetDestroyed (i : Fin 2) : ℚ :=
  probA i i * probB i i

/-- The probability of exactly one target being destroyed -/
def oneTargetDestroyed : ℚ :=
  (targetDestroyed 0) * (1 - probA 1 1) * (1 - probB 1 1) +
  (targetDestroyed 1) * (1 - probA 0 0) * (1 - probB 0 0)

theorem target_destruction_probabilities :
  (targetDestroyed 0 = 1/6) ∧
  (oneTargetDestroyed = 3/10) := by sorry

end NUMINAMATH_CALUDE_target_destruction_probabilities_l2508_250830


namespace NUMINAMATH_CALUDE_geometric_sequence_50th_term_l2508_250897

/-- The 50th term of a geometric sequence with first term 8 and second term -16 -/
theorem geometric_sequence_50th_term :
  let a₁ : ℝ := 8
  let a₂ : ℝ := -16
  let r : ℝ := a₂ / a₁
  let aₙ (n : ℕ) : ℝ := a₁ * r^(n - 1)
  aₙ 50 = -8 * 2^49 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_50th_term_l2508_250897


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l2508_250801

theorem boat_speed_ratio (b r : ℝ) (h1 : b > 0) (h2 : r > 0) 
  (h3 : (b - r)⁻¹ = 2 * (b + r)⁻¹) 
  (s1 s2 : ℝ) (h4 : s1 > 0) (h5 : s2 > 0)
  (h6 : b * (1/4) + b * (3/4) = b) :
  b / (s1 + s2) = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l2508_250801


namespace NUMINAMATH_CALUDE_square_root_range_l2508_250851

theorem square_root_range (x : ℝ) : 3 - 2*x ≥ 0 → x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_range_l2508_250851


namespace NUMINAMATH_CALUDE_power_of_point_l2508_250838

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a line passing through two points
structure Line where
  point1 : Point
  point2 : Point

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the intersection of a line and a circle
def intersect (l : Line) (c : Circle) : Option (Point × Point) := sorry

-- Theorem statement
theorem power_of_point (S : Circle) (P A B A1 B1 : Point) 
  (l1 l2 : Line) : 
  l1.point1 = P → l2.point1 = P → 
  intersect l1 S = some (A, B) → 
  intersect l2 S = some (A1, B1) → 
  distance P A * distance P B = distance P A1 * distance P B1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_point_l2508_250838


namespace NUMINAMATH_CALUDE_xy_yz_xz_equals_60_l2508_250844

theorem xy_yz_xz_equals_60 
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 36)
  (eq3 : z^2 + x*z + x^2 = 111) :
  x*y + y*z + x*z = 60 := by
sorry

end NUMINAMATH_CALUDE_xy_yz_xz_equals_60_l2508_250844


namespace NUMINAMATH_CALUDE_our_number_not_perfect_square_l2508_250831

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the number we want to prove is not a perfect square
def our_number : ℕ := 4^2021

-- Theorem statement
theorem our_number_not_perfect_square : ¬ (is_perfect_square our_number) := by
  sorry

end NUMINAMATH_CALUDE_our_number_not_perfect_square_l2508_250831


namespace NUMINAMATH_CALUDE_two_blue_marbles_probability_l2508_250891

def total_marbles : ℕ := 3 + 4 + 9

def blue_marbles : ℕ := 4

def probability_two_blue_marbles : ℚ :=
  (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1))

theorem two_blue_marbles_probability :
  probability_two_blue_marbles = 1 / 20 :=
sorry

end NUMINAMATH_CALUDE_two_blue_marbles_probability_l2508_250891


namespace NUMINAMATH_CALUDE_school_purchase_cost_l2508_250863

/-- Calculates the total cost of pencils and pens after all applicable discounts -/
def totalCostAfterDiscounts (pencilPrice penPrice : ℚ) (pencilCount penCount : ℕ) 
  (pencilDiscountThreshold penDiscountThreshold : ℕ) 
  (pencilDiscountRate penDiscountRate additionalDiscountRate : ℚ)
  (additionalDiscountThreshold : ℚ) : ℚ :=
  sorry

theorem school_purchase_cost : 
  let pencilPrice : ℚ := 2.5
  let penPrice : ℚ := 3.5
  let pencilCount : ℕ := 38
  let penCount : ℕ := 56
  let pencilDiscountThreshold : ℕ := 30
  let penDiscountThreshold : ℕ := 50
  let pencilDiscountRate : ℚ := 0.1
  let penDiscountRate : ℚ := 0.15
  let additionalDiscountRate : ℚ := 0.05
  let additionalDiscountThreshold : ℚ := 250

  totalCostAfterDiscounts pencilPrice penPrice pencilCount penCount 
    pencilDiscountThreshold penDiscountThreshold 
    pencilDiscountRate penDiscountRate additionalDiscountRate
    additionalDiscountThreshold = 239.5 := by
  sorry

end NUMINAMATH_CALUDE_school_purchase_cost_l2508_250863


namespace NUMINAMATH_CALUDE_larger_integer_proof_l2508_250896

theorem larger_integer_proof (x y : ℕ+) 
  (h1 : y - x = 8)
  (h2 : x * y = 272) : 
  y = 20 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l2508_250896


namespace NUMINAMATH_CALUDE_line_equation_proof_l2508_250874

/-- Given two lines in the 2D plane, we define them as parallel if they have the same slope. -/
def parallel_lines (m1 b1 m2 b2 : ℝ) : Prop :=
  m1 = m2

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line given by y = mx + b -/
def point_on_line (p : Point) (m b : ℝ) : Prop :=
  p.y = m * p.x + b

theorem line_equation_proof :
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x - y + 3 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2 * x - y - 8 = 0
  let point_A : Point := ⟨2, -4⟩
  parallel_lines 2 3 2 (-8) ∧ point_on_line point_A 2 (-8) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2508_250874


namespace NUMINAMATH_CALUDE_vector_expression_l2508_250822

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

theorem vector_expression : c = 3 • a - b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l2508_250822


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l2508_250839

-- Define the properties of the function f
def is_odd_and_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = -f x)

-- State the theorem
theorem odd_periodic_function_value (f : ℝ → ℝ) (h : is_odd_and_periodic f) : 
  f 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l2508_250839


namespace NUMINAMATH_CALUDE_solution_interval_l2508_250823

theorem solution_interval (f : ℝ → ℝ) (k : ℝ) : 
  (∃ x, f x = 0 ∧ k < x ∧ x < k + 1/2) →
  (∃ n : ℤ, k = n * 1/2) →
  (∀ x, f x = x^3 - 4 + x) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_interval_l2508_250823


namespace NUMINAMATH_CALUDE_problem_solution_l2508_250881

theorem problem_solution : ∃ N : ℕ, 
  (N / (555 + 445) = 2 * (555 - 445)) ∧ 
  (N % (555 + 445) = 50) ∧ 
  N = 220050 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2508_250881


namespace NUMINAMATH_CALUDE_min_value_expression_l2508_250892

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2508_250892


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2508_250833

/-- A quadratic function f(x) = x^2 - 6x + c intersects the x-axis at only one point
    if and only if its discriminant is zero. -/
def intersects_once (c : ℝ) : Prop :=
  ((-6)^2 - 4*1*c) = 0

/-- The theorem states that if a quadratic function f(x) = x^2 - 6x + c
    intersects the x-axis at only one point, then c = 9. -/
theorem quadratic_intersection (c : ℝ) :
  intersects_once c → c = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2508_250833


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2508_250869

/-- Represents an isosceles triangle with vertex angle 80°, leg length a, and base length b -/
structure IsoscelesTriangle where
  a : ℝ  -- length of the legs
  b : ℝ  -- length of the base
  h₁ : a > 0
  h₂ : b > 0

/-- Calculates the area of an isosceles triangle -/
noncomputable def triangleArea (t : IsoscelesTriangle) : ℝ :=
  (t.a^3 * t.b) / (4 * (t.b^2 - t.a^2))

/-- Theorem stating that the area of the isosceles triangle with vertex angle 80° is (a^3 * b) / (4 * (b^2 - a^2)) -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) :
  triangleArea t = (t.a^3 * t.b) / (4 * (t.b^2 - t.a^2)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2508_250869


namespace NUMINAMATH_CALUDE_unique_poly_pair_l2508_250806

/-- A polynomial of degree 3 -/
def Poly3 (R : Type*) [CommRing R] := R → R

/-- The evaluation of a polynomial at a point -/
def eval (p : Poly3 ℝ) (x : ℝ) : ℝ := p x

/-- The composition of two polynomials -/
def comp (p q : Poly3 ℝ) : Poly3 ℝ := λ x ↦ p (q x)

/-- The cube of a polynomial -/
def cube (p : Poly3 ℝ) : Poly3 ℝ := λ x ↦ (p x)^3

theorem unique_poly_pair (f g : Poly3 ℝ) 
  (h1 : f ≠ g)
  (h2 : ∀ x, eval (comp f f) x = eval (cube g) x)
  (h3 : ∀ x, eval (comp f g) x = eval (cube f) x)
  (h4 : eval f 0 = 1) :
  (∀ x, f x = (1 - x)^3) ∧ (∀ x, g x = (x - 1)^3 + 1) := by
  sorry


end NUMINAMATH_CALUDE_unique_poly_pair_l2508_250806


namespace NUMINAMATH_CALUDE_mask_probability_l2508_250866

theorem mask_probability (regular_ratio surgical_ratio regular_ear_loop_ratio surgical_ear_loop_ratio : Real) 
  (h1 : regular_ratio = 0.8)
  (h2 : surgical_ratio = 0.2)
  (h3 : regular_ear_loop_ratio = 0.1)
  (h4 : surgical_ear_loop_ratio = 0.2)
  (h5 : regular_ratio + surgical_ratio = 1) :
  regular_ratio * regular_ear_loop_ratio + surgical_ratio * surgical_ear_loop_ratio = 0.12 := by
sorry

end NUMINAMATH_CALUDE_mask_probability_l2508_250866


namespace NUMINAMATH_CALUDE_train_length_l2508_250873

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 12 → ∃ length : ℝ, 
  (abs (length - 200.04) < 0.01) ∧ (length = speed * (1000 / 3600) * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2508_250873


namespace NUMINAMATH_CALUDE_marks_reading_time_l2508_250832

/-- Calculates Mark's new weekly reading time given his daily reading time,
    the number of days in a week, and his planned increase in weekly reading time. -/
def new_weekly_reading_time (daily_reading_time : ℕ) (days_in_week : ℕ) (weekly_increase : ℕ) : ℕ :=
  daily_reading_time * days_in_week + weekly_increase

/-- Proves that Mark's new weekly reading time is 18 hours -/
theorem marks_reading_time :
  new_weekly_reading_time 2 7 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_marks_reading_time_l2508_250832


namespace NUMINAMATH_CALUDE_expression_evaluation_l2508_250841

theorem expression_evaluation : (10^9) / ((2 * 10^6) * 3) = 500/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2508_250841


namespace NUMINAMATH_CALUDE_maria_quiz_goal_l2508_250875

theorem maria_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (quizzes_taken : ℕ) (as_earned : ℕ) (remaining_lower_a : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 70 / 100 →
  quizzes_taken = 35 →
  as_earned = 28 →
  remaining_lower_a = 11 →
  (as_earned + (total_quizzes - quizzes_taken - remaining_lower_a) : ℚ) / total_quizzes ≥ goal_percentage := by
  sorry

#check maria_quiz_goal

end NUMINAMATH_CALUDE_maria_quiz_goal_l2508_250875


namespace NUMINAMATH_CALUDE_all_multiples_contain_two_l2508_250820

def numbers : List ℕ := [418, 244, 816, 426, 24]

def containsTwo (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 2

theorem all_multiples_contain_two :
  ∀ n ∈ numbers, containsTwo (3 * n) :=
by sorry

end NUMINAMATH_CALUDE_all_multiples_contain_two_l2508_250820


namespace NUMINAMATH_CALUDE_part_one_part_two_l2508_250854

-- Define the function f
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

-- Part I
theorem part_one : 
  {x : ℝ | f x (-1) (-1) ≥ x} = {x : ℝ | x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2)} := by sorry

-- Part II
theorem part_two (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2) 
  (h4 : a ≤ -3 ∨ a ≥ 3) : 
  m = 1/3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2508_250854


namespace NUMINAMATH_CALUDE_missing_bricks_count_l2508_250815

/-- Represents a brick wall -/
structure BrickWall where
  total_positions : ℕ
  filled_positions : ℕ
  h_filled_le_total : filled_positions ≤ total_positions

/-- The number of missing bricks in a wall -/
def missing_bricks (wall : BrickWall) : ℕ :=
  wall.total_positions - wall.filled_positions

/-- Theorem stating that the number of missing bricks in the given wall is 26 -/
theorem missing_bricks_count (wall : BrickWall) 
  (h_total : wall.total_positions = 60)
  (h_filled : wall.filled_positions = 34) : 
  missing_bricks wall = 26 := by
sorry


end NUMINAMATH_CALUDE_missing_bricks_count_l2508_250815


namespace NUMINAMATH_CALUDE_debby_stuffed_animal_tickets_l2508_250826

/-- The number of tickets Debby spent on various items at the arcade -/
structure ArcadeTickets where
  hat : ℕ
  yoyo : ℕ
  stuffed_animal : ℕ
  total : ℕ

/-- Theorem about Debby's ticket spending at the arcade -/
theorem debby_stuffed_animal_tickets (d : ArcadeTickets) 
  (hat_tickets : d.hat = 2)
  (yoyo_tickets : d.yoyo = 2)
  (total_tickets : d.total = 14)
  (sum_correct : d.hat + d.yoyo + d.stuffed_animal = d.total) :
  d.stuffed_animal = 10 := by
  sorry

end NUMINAMATH_CALUDE_debby_stuffed_animal_tickets_l2508_250826


namespace NUMINAMATH_CALUDE_additive_implies_zero_and_odd_l2508_250853

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y is zero at 0 and odd -/
theorem additive_implies_zero_and_odd (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_additive_implies_zero_and_odd_l2508_250853


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l2508_250810

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y k : ℝ) : Prop := y = k*x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0 ∧
    x₁ ≠ x₂

-- Define the distance ratio condition
def distance_ratio (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ((x₁ - 0)^2 + (y₁ - 1)^2).sqrt = 3 * ((x₂ - 0)^2 + (y₂ - 1)^2).sqrt

-- The theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  intersection_points k →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    distance_ratio x₁ y₁ x₂ y₂) →
  k = (2 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l2508_250810


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2508_250808

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_cost := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that John's remaining money is 50 - 15p dollars -/
theorem johns_remaining_money (p : ℝ) : money_left p = 50 - 15 * p := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2508_250808


namespace NUMINAMATH_CALUDE_min_distance_C1_to_C2_sum_distances_PA_PB_l2508_250843

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x + 2

-- Define the ellipse C1'
def C1' (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define point P
def P : ℝ × ℝ := (-1, 1)

-- Theorem for the minimum distance
theorem min_distance_C1_to_C2 :
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧
  (∀ (x y : ℝ), C1 x y → ∀ (x' y' : ℝ), C2 x' y' →
    Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d) ∧
  (∃ (x y : ℝ), C1 x y ∧ ∃ (x' y' : ℝ), C2 x' y' ∧
    Real.sqrt ((x - x')^2 + (y - y')^2) = d) :=
sorry

-- Theorem for the sum of distances
theorem sum_distances_PA_PB :
  ∃ (A B : ℝ × ℝ), C1' A.1 A.2 ∧ C1' B.1 B.2 ∧
  C2 A.1 A.2 ∧ C2 B.1 B.2 ∧
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_min_distance_C1_to_C2_sum_distances_PA_PB_l2508_250843


namespace NUMINAMATH_CALUDE_proposition_A_necessary_not_sufficient_l2508_250814

/-- Proposition A: The inequality x^2 + 2ax + 4 ≤ 0 has solutions -/
def proposition_A (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2

/-- Proposition B: The function f(x) = log_a(x + a - 2) is always positive on the interval (1, +∞) -/
def proposition_B (a : ℝ) : Prop := a ≥ 2

theorem proposition_A_necessary_not_sufficient :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  ¬(∀ a : ℝ, proposition_A a → proposition_B a) := by
  sorry

end NUMINAMATH_CALUDE_proposition_A_necessary_not_sufficient_l2508_250814


namespace NUMINAMATH_CALUDE_right_triangle_properties_l2508_250893

theorem right_triangle_properties (A B C : ℝ) (h_right : A = 90) (h_tan : Real.tan C = 5) (h_hypotenuse : A = 80) :
  let AB := 80 * (5 / Real.sqrt 26)
  let BC := 80 / Real.sqrt 26
  (AB = 80 * (5 / Real.sqrt 26)) ∧ (BC / AB = 1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l2508_250893


namespace NUMINAMATH_CALUDE_triangle_side_values_l2508_250899

theorem triangle_side_values (n : ℕ) : 
  (3 * n - 3 > 0) ∧ 
  (2 * n + 12 > 0) ∧ 
  (2 * n + 7 > 0) ∧ 
  (3 * n - 3 + 2 * n + 7 > 2 * n + 12) ∧
  (3 * n - 3 + 2 * n + 12 > 2 * n + 7) ∧
  (2 * n + 7 + 2 * n + 12 > 3 * n - 3) ∧
  (2 * n + 12 > 2 * n + 7) ∧
  (2 * n + 7 > 3 * n - 3) →
  (∃ (count : ℕ), count = 7 ∧ 
    (∀ (m : ℕ), (m ≥ 1 ∧ m ≤ count) ↔ 
      (∃ (k : ℕ), k ≥ 3 ∧ k ≤ 9 ∧
        (3 * k - 3 > 0) ∧ 
        (2 * k + 12 > 0) ∧ 
        (2 * k + 7 > 0) ∧ 
        (3 * k - 3 + 2 * k + 7 > 2 * k + 12) ∧
        (3 * k - 3 + 2 * k + 12 > 2 * k + 7) ∧
        (2 * k + 7 + 2 * k + 12 > 3 * k - 3) ∧
        (2 * k + 12 > 2 * k + 7) ∧
        (2 * k + 7 > 3 * k - 3)))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l2508_250899


namespace NUMINAMATH_CALUDE_largest_square_area_l2508_250846

-- Define the right triangle XYZ
structure RightTriangle where
  xy : ℝ  -- length of side XY
  xz : ℝ  -- length of side XZ
  yz : ℝ  -- length of hypotenuse YZ
  right_angle : xy^2 + xz^2 = yz^2  -- Pythagorean theorem

-- Define the theorem
theorem largest_square_area (t : RightTriangle) 
  (sum_of_squares : t.xy^2 + t.xz^2 + t.yz^2 = 450) :
  t.yz^2 = 225 := by
  sorry


end NUMINAMATH_CALUDE_largest_square_area_l2508_250846


namespace NUMINAMATH_CALUDE_marcy_serves_36_people_l2508_250836

/-- Represents the makeup supplies and application rates --/
structure MakeupSupplies where
  lip_gloss_per_tube : ℕ
  mascara_per_tube : ℕ
  lip_gloss_tubs : ℕ
  lip_gloss_tubes_per_tub : ℕ
  mascara_tubs : ℕ
  mascara_tubes_per_tub : ℕ

/-- Calculates the number of people that can be served with the given makeup supplies --/
def people_served (supplies : MakeupSupplies) : ℕ :=
  min
    (supplies.lip_gloss_tubs * supplies.lip_gloss_tubes_per_tub * supplies.lip_gloss_per_tube)
    (supplies.mascara_tubs * supplies.mascara_tubes_per_tub * supplies.mascara_per_tube)

/-- Theorem stating that Marcy can serve exactly 36 people with her makeup supplies --/
theorem marcy_serves_36_people :
  let supplies := MakeupSupplies.mk 3 5 6 2 4 3
  people_served supplies = 36 := by
  sorry

#eval people_served (MakeupSupplies.mk 3 5 6 2 4 3)

end NUMINAMATH_CALUDE_marcy_serves_36_people_l2508_250836


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l2508_250803

/-- Given a polynomial division where:
    - p(x) is a polynomial of degree 17
    - g(x) is the divisor polynomial
    - The quotient polynomial has degree 9
    - The remainder polynomial has degree 5
    Then the degree of g(x) is 8. -/
theorem polynomial_division_degree (p g q r : Polynomial ℝ) : 
  Polynomial.degree p = 17 →
  p = g * q + r →
  Polynomial.degree q = 9 →
  Polynomial.degree r = 5 →
  Polynomial.degree g = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l2508_250803


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2508_250837

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2508_250837


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l2508_250855

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (1 - a)*x + 1 ≥ 0) → a ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l2508_250855


namespace NUMINAMATH_CALUDE_min_value_expression_l2508_250816

theorem min_value_expression (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 1) 
  (hy : -0.5 ≤ y ∧ y ≤ 1) 
  (hz : -0.5 ≤ z ∧ z ≤ 1) : 
  3 / ((1 - x) * (1 - y) * (1 - z)) + 3 / ((1 + x) * (1 + y) * (1 + z)) ≥ 6 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 3 / ((1 - x) * (1 - y) * (1 - z)) + 3 / ((1 + x) * (1 + y) * (1 + z)) = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2508_250816


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l2508_250807

theorem power_of_negative_cube (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l2508_250807


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2508_250842

/-- The product of the coordinates of the midpoint of a line segment with endpoints (4, -1) and (-2, 7) is 3. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -1
  let x2 : ℝ := -2
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = 3 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2508_250842


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l2508_250885

theorem sqrt_nine_factorial_over_126 :
  let nine_factorial : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let one_twenty_six : ℕ := 2 * 7 * 9
  (nine_factorial / one_twenty_six : ℚ).sqrt = 12 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l2508_250885


namespace NUMINAMATH_CALUDE_perfect_apples_l2508_250835

theorem perfect_apples (total : ℕ) (small_fraction : ℚ) (unripe_fraction : ℚ) :
  total = 30 →
  small_fraction = 1/6 →
  unripe_fraction = 1/3 →
  (total : ℚ) - small_fraction * total - unripe_fraction * total = 15 := by
  sorry

end NUMINAMATH_CALUDE_perfect_apples_l2508_250835


namespace NUMINAMATH_CALUDE_f_of_2_equals_12_l2508_250862

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem f_of_2_equals_12 : f 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_12_l2508_250862


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l2508_250898

theorem polynomial_identity_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l2508_250898
