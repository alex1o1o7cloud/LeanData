import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2470_247094

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2470_247094


namespace NUMINAMATH_CALUDE_all_girls_same_color_probability_l2470_247062

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 10

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 30

/-- Represents the number of girls selecting marbles -/
def num_girls : ℕ := 15

/-- The probability that all girls select the same colored marble -/
def probability_same_color : ℚ := 0

theorem all_girls_same_color_probability :
  marbles_per_color = 10 →
  total_marbles = 30 →
  num_girls = 15 →
  probability_same_color = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_girls_same_color_probability_l2470_247062


namespace NUMINAMATH_CALUDE_score_79_implies_93_correct_l2470_247040

/-- Represents the grading system for a test -/
structure TestGrade where
  total_questions : ℕ
  correct_answers : ℕ
  score : ℤ

/-- Theorem stating that for a 100-question test with the given grading system,
    a score of 79 implies 93 correct answers -/
theorem score_79_implies_93_correct
  (test : TestGrade)
  (h1 : test.total_questions = 100)
  (h2 : test.score = test.correct_answers - 2 * (test.total_questions - test.correct_answers))
  (h3 : test.score = 79) :
  test.correct_answers = 93 := by
  sorry

end NUMINAMATH_CALUDE_score_79_implies_93_correct_l2470_247040


namespace NUMINAMATH_CALUDE_angle_phi_value_l2470_247022

-- Define the problem statement
theorem angle_phi_value (φ : Real) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin (20 * π / 180) = Real.cos φ - Real.sin φ) : 
  φ = 25 * π / 180 := by
  sorry

#check angle_phi_value

end NUMINAMATH_CALUDE_angle_phi_value_l2470_247022


namespace NUMINAMATH_CALUDE_problem_statement_l2470_247077

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * b^3 / 5 = 1000) 
  (h2 : a * b = 2) : 
  a^3 * b^2 / 3 = 2 / 705 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2470_247077


namespace NUMINAMATH_CALUDE_smallest_sum_solution_l2470_247086

theorem smallest_sum_solution : ∃ (a b c : ℕ), 
  (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ 
  (∀ (x y z : ℕ), (x * z + 2 * y * z + x + 2 * y = z^2 + z + 6) → 
    (a + b + c ≤ x + y + z)) ∧
  (a = 2 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_solution_l2470_247086


namespace NUMINAMATH_CALUDE_two_thirds_of_fifteen_fourths_l2470_247029

theorem two_thirds_of_fifteen_fourths (x : ℚ) : x = 15 / 4 → (2 / 3) * x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_fifteen_fourths_l2470_247029


namespace NUMINAMATH_CALUDE_hiking_problem_l2470_247061

/-- Hiking Problem -/
theorem hiking_problem (up_rate : ℝ) (up_time : ℝ) (down_dist : ℝ) (rate_ratio : ℝ) :
  up_time = 2 →
  down_dist = 18 →
  rate_ratio = 1.5 →
  up_rate * up_time = down_dist / rate_ratio →
  up_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_hiking_problem_l2470_247061


namespace NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l2470_247069

/-- Represents the value of 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The tourism revenue in billions of yuan -/
def tourism_revenue : ℝ := 12.41

theorem tourism_revenue_scientific_notation : 
  tourism_revenue * billion = 1.241 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l2470_247069


namespace NUMINAMATH_CALUDE_set_operation_result_l2470_247090

-- Define the sets A, B, and C
def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

-- State the theorem
theorem set_operation_result : (A ∪ B) ∩ C = {3, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l2470_247090


namespace NUMINAMATH_CALUDE_expression_equals_ten_l2470_247081

theorem expression_equals_ten :
  let a : ℚ := 3
  let b : ℚ := 2
  let c : ℚ := 2
  (c * a^3 + c * b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_ten_l2470_247081


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2470_247072

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x - 2) + 12 / Real.sqrt (3 * x - 2) = 8 ↔ x = 2 ∨ x = 38 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2470_247072


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l2470_247009

theorem cylinder_radius_proof (r : ℝ) : 
  let h : ℝ := 3
  let volume (r h : ℝ) := π * r^2 * h
  let volume_increase_height := volume r (h + 3) - volume r h
  let volume_increase_radius := volume (r + 3) h - volume r h
  volume_increase_height = volume_increase_radius →
  r = 3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l2470_247009


namespace NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l2470_247032

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Theorem: For a parabola y^2 = 2px and a line passing through its focus
    with an inclination angle of 60°, intersecting the parabola at points A and B
    in the first and fourth quadrants respectively, the ratio |AF| / |BF| = 3 -/
theorem parabola_line_intersection_ratio 
  (para : Parabola) 
  (l : Line) 
  (A B : Point) 
  (h1 : l.point = Point.mk (para.p / 2) 0)  -- Focus of the parabola
  (h2 : l.angle = π / 3)  -- 60° in radians
  (h3 : A.x > 0 ∧ A.y > 0)  -- A in first quadrant
  (h4 : B.x > 0 ∧ B.y < 0)  -- B in fourth quadrant
  (h5 : A.y^2 = 2 * para.p * A.x)  -- A on parabola
  (h6 : B.y^2 = 2 * para.p * B.x)  -- B on parabola
  : abs (A.x - para.p / 2) / abs (B.x - para.p / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l2470_247032


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2470_247027

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2470_247027


namespace NUMINAMATH_CALUDE_pizza_combinations_l2470_247034

/-- The number of available pizza toppings -/
def num_toppings : ℕ := 8

/-- The number of one-topping pizzas that can be ordered -/
def one_topping_pizzas : ℕ := num_toppings

/-- The number of two-topping pizzas that can be ordered -/
def two_topping_pizzas : ℕ := (num_toppings.choose 2)

/-- The number of three-topping pizzas that can be ordered -/
def three_topping_pizzas : ℕ := (num_toppings.choose 3)

/-- The total number of different one-, two-, and three-topping pizzas that can be ordered -/
def total_pizzas : ℕ := one_topping_pizzas + two_topping_pizzas + three_topping_pizzas

theorem pizza_combinations :
  total_pizzas = 92 := by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2470_247034


namespace NUMINAMATH_CALUDE_jason_gave_nine_cards_l2470_247078

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason has left -/
def remaining_cards : ℕ := 4

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem jason_gave_nine_cards : cards_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_nine_cards_l2470_247078


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2470_247035

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  b * Real.cos C = c * Real.cos B → B = C := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2470_247035


namespace NUMINAMATH_CALUDE_passing_mark_is_200_l2470_247019

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total_marks : ℕ
  passing_mark : ℕ

/-- Defines the conditions of the exam as described in the problem. -/
def exam_conditions (e : Exam) : Prop :=
  (e.total_marks * 30 / 100 + 50 = e.passing_mark) ∧
  (e.total_marks * 45 / 100 = e.passing_mark + 25)

/-- Theorem stating that under the given conditions, the passing mark is 200. -/
theorem passing_mark_is_200 :
  ∃ e : Exam, exam_conditions e ∧ e.passing_mark = 200 := by
  sorry


end NUMINAMATH_CALUDE_passing_mark_is_200_l2470_247019


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l2470_247047

/-- Given a configuration of circles where four congruent smaller circles
    are arranged inside a larger circle such that their diameters align
    with the diameter of the larger circle, this theorem states that
    the radius of each smaller circle is one-fourth of the radius of the larger circle. -/
theorem smaller_circle_radius (R : ℝ) (r : ℝ) 
    (h1 : R = 8) -- The radius of the larger circle is 8 meters
    (h2 : 4 * r = R) -- Four smaller circle diameters align with the larger circle diameter
    : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l2470_247047


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l2470_247070

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l2470_247070


namespace NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_174_main_result_l2470_247041

theorem least_number_with_remainder (n : ℕ) : 
  (n % 34 = 4 ∧ n % 5 = 4) → n ≥ 174 := by
  sorry

theorem least_number_is_174 : 
  174 % 34 = 4 ∧ 174 % 5 = 4 := by
  sorry

theorem main_result : 
  ∀ n : ℕ, (n % 34 = 4 ∧ n % 5 = 4) → n ≥ 174 ∧ (174 % 34 = 4 ∧ 174 % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_174_main_result_l2470_247041


namespace NUMINAMATH_CALUDE_function_properties_l2470_247002

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

-- State the theorem
theorem function_properties (a : ℝ) (h : a > 0) :
  (∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, f a x ≥ m) ∧
  ((∀ x : ℝ, f a x > 0) → a > 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ → f a x₁ < f a x₂) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2470_247002


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_sum_of_A_and_B_proof_l2470_247087

theorem sum_of_A_and_B : ℕ → ℕ → Prop :=
  fun A B =>
    (A < 10 ∧ B < 10) →  -- A and B are single digit numbers
    (A = 2 + 4) →        -- A is 4 greater than 2
    (B - 3 = 1) →        -- 3 less than B is 1
    A + B = 10           -- The sum of A and B is 10

-- Proof
theorem sum_of_A_and_B_proof : sum_of_A_and_B 6 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_sum_of_A_and_B_proof_l2470_247087


namespace NUMINAMATH_CALUDE_fiona_pages_587_equal_reading_time_l2470_247063

/-- Represents the book reading scenario -/
structure BookReading where
  totalPages : ℕ
  fionaSpeed : ℕ  -- seconds per page
  davidSpeed : ℕ  -- seconds per page

/-- Calculates the number of pages Fiona should read for equal reading time -/
def fionaPages (br : BookReading) : ℕ :=
  (br.totalPages * br.davidSpeed) / (br.fionaSpeed + br.davidSpeed)

/-- Theorem stating that Fiona should read 587 pages -/
theorem fiona_pages_587 (br : BookReading) 
  (h1 : br.totalPages = 900)
  (h2 : br.fionaSpeed = 40)
  (h3 : br.davidSpeed = 75) : 
  fionaPages br = 587 := by
  sorry

/-- Theorem stating that Fiona and David spend equal time reading -/
theorem equal_reading_time (br : BookReading) 
  (h1 : br.totalPages = 900)
  (h2 : br.fionaSpeed = 40)
  (h3 : br.davidSpeed = 75) : 
  br.fionaSpeed * (fionaPages br) = br.davidSpeed * (br.totalPages - fionaPages br) := by
  sorry

end NUMINAMATH_CALUDE_fiona_pages_587_equal_reading_time_l2470_247063


namespace NUMINAMATH_CALUDE_factorization_d_is_valid_l2470_247065

/-- Represents a polynomial factorization -/
def IsFactorization (left right : ℝ → ℝ) : Prop :=
  ∀ x, left x = right x ∧ 
       ∃ p q : ℝ → ℝ, right = fun y ↦ p y * q y

/-- The specific factorization we want to prove -/
def FactorizationD (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The factored form -/
def FactoredFormD (x : ℝ) : ℝ := (x + 2)^2

/-- Theorem stating that FactorizationD is a valid factorization -/
theorem factorization_d_is_valid : IsFactorization FactorizationD FactoredFormD := by
  sorry

end NUMINAMATH_CALUDE_factorization_d_is_valid_l2470_247065


namespace NUMINAMATH_CALUDE_english_only_enrollment_l2470_247056

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 45)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : german ≥ both) :
  total - german + both = 23 := by
  sorry

end NUMINAMATH_CALUDE_english_only_enrollment_l2470_247056


namespace NUMINAMATH_CALUDE_max_fraction_value_l2470_247045

theorem max_fraction_value (a b c d : ℕ) 
  (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hd : d < 10) :
  (∀ w x y z : ℕ, 0 < w → w < x → x < y → y < z → z < 10 → 
    (a - b : ℚ) / (c - d : ℚ) ≥ (w - x : ℚ) / (y - z : ℚ)) →
  (a - b : ℚ) / (c - d : ℚ) = -6 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_value_l2470_247045


namespace NUMINAMATH_CALUDE_P_on_x_axis_AP_parallel_y_axis_l2470_247008

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of point P with coordinates (m+1, 2m-4) -/
def P (m : ℝ) : Point :=
  { x := m + 1, y := 2 * m - 4 }

/-- Point A with coordinates (-5, 2) -/
def A : Point :=
  { x := -5, y := 2 }

/-- Theorem: If P lies on the x-axis, then its coordinates are (3,0) -/
theorem P_on_x_axis (m : ℝ) : P m = { x := 3, y := 0 } ↔ (P m).y = 0 := by
  sorry

/-- Theorem: If AP is parallel to y-axis, then P's coordinates are (-5,-16) -/
theorem AP_parallel_y_axis (m : ℝ) : P m = { x := -5, y := -16 } ↔ (P m).x = A.x := by
  sorry

end NUMINAMATH_CALUDE_P_on_x_axis_AP_parallel_y_axis_l2470_247008


namespace NUMINAMATH_CALUDE_greatest_product_base_seven_l2470_247075

/-- Represents a positive integer in base 7 --/
def BaseSeven := List Nat

/-- Converts a decimal number to base 7 --/
def toBaseSeven (n : Nat) : BaseSeven :=
  sorry

/-- Calculates the product of digits in a base 7 number --/
def productOfDigits (n : BaseSeven) : Nat :=
  sorry

/-- Theorem: The greatest possible product of digits in base 7 for numbers less than 2300 --/
theorem greatest_product_base_seven :
  (∃ (n : Nat), n < 2300 ∧
    (∀ (m : Nat), m < 2300 →
      productOfDigits (toBaseSeven m) ≤ productOfDigits (toBaseSeven n)) ∧
    productOfDigits (toBaseSeven n) = 1080) :=
  sorry

end NUMINAMATH_CALUDE_greatest_product_base_seven_l2470_247075


namespace NUMINAMATH_CALUDE_fishing_tournament_l2470_247017

theorem fishing_tournament (jacob_initial : ℕ) : 
  (7 * jacob_initial - 23 = jacob_initial + 26 - 1) → jacob_initial = 8 := by sorry

end NUMINAMATH_CALUDE_fishing_tournament_l2470_247017


namespace NUMINAMATH_CALUDE_piggy_bank_dimes_l2470_247021

/-- Proves that given $5.55 in dimes and quarters, with three more dimes than quarters, the number of dimes is 18 -/
theorem piggy_bank_dimes (total : ℚ) (dimes quarters : ℕ) : 
  total = (5 : ℚ) + (55 : ℚ) / 100 →
  dimes = quarters + 3 →
  (10 : ℚ) * dimes + (25 : ℚ) * quarters = total * 100 →
  dimes = 18 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_dimes_l2470_247021


namespace NUMINAMATH_CALUDE_rest_of_body_length_l2470_247080

theorem rest_of_body_length
  (total_height : ℝ)
  (leg_ratio : ℝ)
  (head_ratio : ℝ)
  (h1 : total_height = 60)
  (h2 : leg_ratio = 1 / 3)
  (h3 : head_ratio = 1 / 4)
  : total_height - (leg_ratio * total_height + head_ratio * total_height) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rest_of_body_length_l2470_247080


namespace NUMINAMATH_CALUDE_difference_between_fractions_l2470_247052

theorem difference_between_fractions (n : ℝ) (h : n = 140) : (4/5 * n) - (65/100 * n) = 21 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_fractions_l2470_247052


namespace NUMINAMATH_CALUDE_pen_cost_l2470_247005

theorem pen_cost (cost : ℝ) (has : ℝ) (needs : ℝ) : 
  has = cost / 3 → needs = 20 → has + needs = cost → cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l2470_247005


namespace NUMINAMATH_CALUDE_fraction_sum_cubes_l2470_247025

theorem fraction_sum_cubes : (5 / 6 : ℚ)^3 + (3 / 5 : ℚ)^3 = 21457 / 27000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_cubes_l2470_247025


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2470_247082

/-- The complex number z defined as 2/(1-i) - 2i^3 is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant :
  let z : ℂ := 2 / (1 - Complex.I) - 2 * Complex.I^3
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2470_247082


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2470_247059

theorem exponent_multiplication (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2470_247059


namespace NUMINAMATH_CALUDE_square_EC_dot_ED_l2470_247073

/-- Square ABCD with side length 2 and E as midpoint of AB -/
structure Square2D where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  is_square : A.1 = B.1 ∧ A.2 = D.2 ∧ C.1 = D.1 ∧ C.2 = B.2
  side_length : ‖B - A‖ = 2
  E_midpoint : E = (A + B) / 2

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem square_EC_dot_ED (s : Square2D) :
  dot_product (s.C - s.E) (s.D - s.E) = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_EC_dot_ED_l2470_247073


namespace NUMINAMATH_CALUDE_rainstorm_multiple_rainstorm_multiple_proof_l2470_247007

/-- Given the conditions of a rainstorm, prove that the multiple of the first hour's
    rain amount that determines the second hour's rain (minus 7 inches) is equal to 2. -/
theorem rainstorm_multiple : ℝ → Prop :=
  fun x =>
    let first_hour_rain := 5
    let second_hour_rain := x * first_hour_rain + 7
    let total_rain := 22
    first_hour_rain + second_hour_rain = total_rain →
    x = 2

/-- Proof of the rainstorm_multiple theorem -/
theorem rainstorm_multiple_proof : rainstorm_multiple 2 := by
  sorry

end NUMINAMATH_CALUDE_rainstorm_multiple_rainstorm_multiple_proof_l2470_247007


namespace NUMINAMATH_CALUDE_class_enrollment_l2470_247023

theorem class_enrollment (q1_correct q2_correct both_correct not_taken : ℕ) 
  (h1 : q1_correct = 25)
  (h2 : q2_correct = 22)
  (h3 : not_taken = 5)
  (h4 : both_correct = 22) :
  q1_correct + q2_correct - both_correct + not_taken = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_class_enrollment_l2470_247023


namespace NUMINAMATH_CALUDE_range_of_m_l2470_247093

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊂ A → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2470_247093


namespace NUMINAMATH_CALUDE_sunzi_car_problem_l2470_247020

theorem sunzi_car_problem (x : ℕ) : 
  (x / 4 : ℚ) + 1 = (x - 9 : ℚ) / 3 ↔ 
  (∃ (cars : ℕ), 
    (x / 4 + 1 = cars) ∧ 
    ((x - 9) / 3 = cars - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_car_problem_l2470_247020


namespace NUMINAMATH_CALUDE_cube_root_negative_l2470_247015

theorem cube_root_negative (a : ℝ) (k : ℝ) (h : k^3 = a) : 
  ((-a : ℝ)^(1/3 : ℝ) : ℝ) = -k := by sorry

end NUMINAMATH_CALUDE_cube_root_negative_l2470_247015


namespace NUMINAMATH_CALUDE_complex_polynomial_root_abs_d_l2470_247074

theorem complex_polynomial_root_abs_d (a b c d : ℤ) : 
  (a * (Complex.I + 3) ^ 5 + b * (Complex.I + 3) ^ 4 + c * (Complex.I + 3) ^ 3 + 
   d * (Complex.I + 3) ^ 2 + c * (Complex.I + 3) + b + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 16 := by
sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_abs_d_l2470_247074


namespace NUMINAMATH_CALUDE_smallest_start_for_five_odd_squares_l2470_247004

theorem smallest_start_for_five_odd_squares : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (s : Finset ℕ), s.card = 5 ∧ 
    (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ Odd m ∧ ∃ k : ℕ, m = k^2) ∧
    (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ Odd m ∧ (∃ k : ℕ, m = k^2) → m ∈ s)) ∧
  (∀ n' : ℕ, 0 < n' ∧ n' < n → 
    ¬∃ (s : Finset ℕ), s.card = 5 ∧ 
      (∀ m ∈ s, n' ≤ m ∧ m ≤ 100 ∧ Odd m ∧ ∃ k : ℕ, m = k^2) ∧
      (∀ m : ℕ, n' ≤ m ∧ m ≤ 100 ∧ Odd m ∧ (∃ k : ℕ, m = k^2) → m ∈ s)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_start_for_five_odd_squares_l2470_247004


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l2470_247031

/-- Given that 4 pages cost 6 cents, prove that $15 (1500 cents) will allow copying 1000 pages. -/
theorem pages_copied_for_fifteen_dollars :
  let pages_per_six_cents : ℚ := 4
  let cents_per_four_pages : ℚ := 6
  let total_cents : ℚ := 1500
  (total_cents * pages_per_six_cents) / cents_per_four_pages = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l2470_247031


namespace NUMINAMATH_CALUDE_alpha_value_l2470_247076

/-- A structure representing the relationship between α, β, and γ -/
structure Relationship where
  α : ℝ
  β : ℝ
  γ : ℝ
  k : ℝ
  h1 : α = k * γ / β

/-- The theorem stating the relationship between α, β, and γ -/
theorem alpha_value (r : Relationship) (h2 : r.α = 4) (h3 : r.β = 27) (h4 : r.γ = 3) :
  ∃ (r' : Relationship), r'.β = -81 ∧ r'.γ = 9 ∧ r'.α = -4 :=
sorry

end NUMINAMATH_CALUDE_alpha_value_l2470_247076


namespace NUMINAMATH_CALUDE_smallest_n_for_interval_multiple_l2470_247028

theorem smallest_n_for_interval_multiple : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 → 
    ∃ (k : ℕ), (m : ℚ) / 1993 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 1994) ∧
  (∀ (n' : ℕ), 0 < n' ∧ n' < n → 
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 ∧
      ∀ (k : ℕ), ¬((m : ℚ) / 1993 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 1994)) ∧
  n = 3987 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_interval_multiple_l2470_247028


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2470_247096

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2470_247096


namespace NUMINAMATH_CALUDE_dog_food_duration_l2470_247067

-- Define the number of dogs
def num_dogs : ℕ := 4

-- Define the number of meals per day
def meals_per_day : ℕ := 2

-- Define the amount of food per meal in grams
def food_per_meal : ℕ := 250

-- Define the number of sacks of dog food
def num_sacks : ℕ := 2

-- Define the weight of each sack in kilograms
def sack_weight : ℕ := 50

-- Define the number of grams in a kilogram
def grams_per_kg : ℕ := 1000

-- Theorem statement
theorem dog_food_duration : 
  (num_sacks * sack_weight * grams_per_kg) / (num_dogs * meals_per_day * food_per_meal) = 50 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_duration_l2470_247067


namespace NUMINAMATH_CALUDE_propositions_truth_values_l2470_247091

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_solution (x : ℚ) : Prop := x^2 + x - 2 = 0

theorem propositions_truth_values :
  (is_prime 3 ∨ is_even 3) ∧
  ¬(is_prime 3 ∧ is_even 3) ∧
  ¬(¬is_prime 3) ∧
  (is_solution (-2) ∨ is_solution 1) ∧
  (is_solution (-2) ∧ is_solution 1) ∧
  ¬(¬is_solution (-2)) := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_values_l2470_247091


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_equal_distances_m_value_l2470_247095

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def A : Point := { x := -3, y := -4 }
def B : Point := { x := 6, y := 3 }

def perpendicular_bisector (p1 p2 : Point) : Line := sorry

def distance_to_line (p : Point) (l : Line) : ℝ := sorry

theorem perpendicular_bisector_equation :
  perpendicular_bisector A B = { a := 9, b := 7, c := -10 } := by sorry

theorem equal_distances_m_value (m : ℝ) :
  let l : Line := { a := 1, b := m, c := 1 }
  distance_to_line A l = distance_to_line B l → m = 5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_equal_distances_m_value_l2470_247095


namespace NUMINAMATH_CALUDE_library_card_lineup_l2470_247016

theorem library_card_lineup : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_library_card_lineup_l2470_247016


namespace NUMINAMATH_CALUDE_inequality_proof_l2470_247013

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (a^2 + b^2 + c^2) ≥ 9 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2470_247013


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2470_247033

def is_in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_in_third_quadrant :
  is_in_third_quadrant (-3) (-4) := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2470_247033


namespace NUMINAMATH_CALUDE_cartesian_polar_equivalence_l2470_247050

-- Define the set of points in Cartesian coordinates
def cartesian_set : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define the set of points in polar coordinates
def polar_set : Set (ℝ × ℝ) := {p | Real.sqrt (p.1^2 + p.2^2) = 3}

-- Theorem stating the equivalence of the two sets
theorem cartesian_polar_equivalence : cartesian_set = polar_set := by sorry

end NUMINAMATH_CALUDE_cartesian_polar_equivalence_l2470_247050


namespace NUMINAMATH_CALUDE_luna_kibble_remaining_l2470_247046

/-- The amount of kibble remaining in the bag after feeding Luna for a day -/
def remaining_kibble (initial_amount : ℕ) (mary_morning : ℕ) (mary_evening : ℕ) 
  (frank_afternoon : ℕ) : ℕ :=
  initial_amount - (mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon)

/-- Theorem stating the remaining amount of kibble in Luna's bag -/
theorem luna_kibble_remaining : 
  remaining_kibble 12 1 1 1 = 7 := by sorry

end NUMINAMATH_CALUDE_luna_kibble_remaining_l2470_247046


namespace NUMINAMATH_CALUDE_distinct_roots_of_quadratic_l2470_247039

theorem distinct_roots_of_quadratic (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - a*x₁ - 2 = 0 ∧ x₂^2 - a*x₂ - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_of_quadratic_l2470_247039


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2470_247001

/-- Given a geometric sequence {a_n} with a₁ = 3, if 4a₁, 2a₂, a₃ form an arithmetic sequence,
    then the common ratio of the geometric sequence is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term condition
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- arithmetic sequence condition
  q = 2 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2470_247001


namespace NUMINAMATH_CALUDE_mary_received_more_l2470_247024

/-- Calculates the profit share difference between two partners in a business --/
def profit_share_difference (mary_investment : ℚ) (harry_investment : ℚ) (total_profit : ℚ) : ℚ :=
  let equal_share := (1 / 3) * total_profit / 2
  let investment_based_profit := (2 / 3) * total_profit
  let mary_investment_share := (mary_investment / (mary_investment + harry_investment)) * investment_based_profit
  let harry_investment_share := (harry_investment / (mary_investment + harry_investment)) * investment_based_profit
  let mary_total := equal_share + mary_investment_share
  let harry_total := equal_share + harry_investment_share
  mary_total - harry_total

/-- Theorem stating that Mary received $800 more than Harry --/
theorem mary_received_more (mary_investment harry_investment total_profit : ℚ) :
  mary_investment = 700 →
  harry_investment = 300 →
  total_profit = 3000 →
  profit_share_difference mary_investment harry_investment total_profit = 800 := by
  sorry

#eval profit_share_difference 700 300 3000

end NUMINAMATH_CALUDE_mary_received_more_l2470_247024


namespace NUMINAMATH_CALUDE_tangent_curves_alpha_l2470_247071

theorem tangent_curves_alpha (f g : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, g x = α * x^2) →
  (∃ x₀, f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀) →
  α = Real.exp 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_curves_alpha_l2470_247071


namespace NUMINAMATH_CALUDE_milk_expense_calculation_l2470_247014

def monthly_salary : ℝ := 18000

theorem milk_expense_calculation (rent : ℝ) (groceries : ℝ) (education : ℝ) (petrol : ℝ) (misc : ℝ) (savings : ℝ) :
  rent = 5000 →
  groceries = 4500 →
  education = 2500 →
  petrol = 2000 →
  misc = 700 →
  savings = 1800 →
  savings = 0.1 * monthly_salary →
  ∃ (milk : ℝ), milk = monthly_salary - (rent + groceries + education + petrol + misc + savings) ∧ milk = 1500 := by
  sorry

end NUMINAMATH_CALUDE_milk_expense_calculation_l2470_247014


namespace NUMINAMATH_CALUDE_complex_fraction_imaginary_l2470_247089

theorem complex_fraction_imaginary (a : ℝ) : 
  (∃ (b : ℝ), b ≠ 0 ∧ (a - Complex.I) / (2 + Complex.I) = Complex.I * b) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_imaginary_l2470_247089


namespace NUMINAMATH_CALUDE_matilda_age_l2470_247003

/-- Given the ages of Louis, Jerica, and Matilda, prove Matilda's age -/
theorem matilda_age (louis_age jerica_age matilda_age : ℕ) : 
  louis_age = 14 →
  jerica_age = 2 * louis_age →
  matilda_age = jerica_age + 7 →
  matilda_age = 35 := by
sorry

end NUMINAMATH_CALUDE_matilda_age_l2470_247003


namespace NUMINAMATH_CALUDE_brenda_bracelets_l2470_247000

theorem brenda_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) (h1 : total_stones = 36) (h2 : stones_per_bracelet = 12) :
  total_stones / stones_per_bracelet = 3 := by
  sorry

end NUMINAMATH_CALUDE_brenda_bracelets_l2470_247000


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2470_247037

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_problem 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = seq.S 3) 
  (h2 : seq.a 6 = 12) : 
  seq.a 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2470_247037


namespace NUMINAMATH_CALUDE_customers_per_car_l2470_247079

/-- Proves that there are 5 customers in each car given the problem conditions --/
theorem customers_per_car :
  let num_cars : ℕ := 10
  let sports_sales : ℕ := 20
  let music_sales : ℕ := 30
  let total_sales : ℕ := sports_sales + music_sales
  let total_customers : ℕ := total_sales
  let customers_per_car : ℕ := total_customers / num_cars
  customers_per_car = 5 := by
  sorry

end NUMINAMATH_CALUDE_customers_per_car_l2470_247079


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l2470_247051

theorem cubic_roots_sum_squares (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 12 = 0 →
  q^3 - 15*q^2 + 25*q - 12 = 0 →
  r^3 - 15*r^2 + 25*r - 12 = 0 →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l2470_247051


namespace NUMINAMATH_CALUDE_jim_total_cost_l2470_247060

def total_cost (lamp_price bulb_price bedside_table_price decorative_item_price : ℝ)
               (lamp_quantity bulb_quantity bedside_table_quantity decorative_item_quantity : ℕ)
               (lamp_discount bulb_discount bedside_table_discount decorative_item_discount : ℝ)
               (lamp_tax_rate bulb_tax_rate bedside_table_tax_rate decorative_item_tax_rate : ℝ) : ℝ :=
  let lamp_cost := lamp_quantity * lamp_price * (1 - lamp_discount) * (1 + lamp_tax_rate)
  let bulb_cost := bulb_quantity * bulb_price * (1 - bulb_discount) * (1 + bulb_tax_rate)
  let bedside_table_cost := bedside_table_quantity * bedside_table_price * (1 - bedside_table_discount) * (1 + bedside_table_tax_rate)
  let decorative_item_cost := decorative_item_quantity * decorative_item_price * (1 - decorative_item_discount) * (1 + decorative_item_tax_rate)
  lamp_cost + bulb_cost + bedside_table_cost + decorative_item_cost

theorem jim_total_cost :
  total_cost 12 8 25 10 2 6 3 4 0.2 0.3 0 0.15 0.05 0.05 0.06 0.04 = 170.30 := by
  sorry

end NUMINAMATH_CALUDE_jim_total_cost_l2470_247060


namespace NUMINAMATH_CALUDE_equilateral_triangles_in_cube_l2470_247042

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- An equilateral triangle is a triangle in which all three sides have the same length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The number of equilateral triangles that can be formed with vertices of a cube -/
def num_equilateral_triangles_in_cube (c : Cube) : ℕ :=
  8

/-- Theorem: The number of equilateral triangles that can be formed with vertices of a cube is 8 -/
theorem equilateral_triangles_in_cube (c : Cube) :
  num_equilateral_triangles_in_cube c = 8 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_in_cube_l2470_247042


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2470_247058

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2470_247058


namespace NUMINAMATH_CALUDE_grid_division_exists_l2470_247006

/-- Represents a figure cut from the grid -/
structure Figure where
  area : ℕ
  externalPerimeter : ℕ
  internalPerimeter : ℕ

/-- Represents the division of the 9x9 grid -/
structure GridDivision where
  a : Figure
  b : Figure
  c : Figure

/-- The proposition to be proved -/
theorem grid_division_exists : ∃ (d : GridDivision),
  -- The grid is 9x9
  (9 * 9 = d.a.area + d.b.area + d.c.area) ∧
  -- All figures have equal area
  (d.a.area = d.b.area) ∧ (d.b.area = d.c.area) ∧
  -- The perimeter of c equals the sum of perimeters of a and b
  (d.c.externalPerimeter + d.c.internalPerimeter = 
   d.a.externalPerimeter + d.a.internalPerimeter + 
   d.b.externalPerimeter + d.b.internalPerimeter) ∧
  -- The sum of external perimeters is the perimeter of the 9x9 grid
  (d.a.externalPerimeter + d.b.externalPerimeter + d.c.externalPerimeter = 4 * 9) ∧
  -- The sum of a and b's internal perimeters equals c's internal perimeter
  (d.a.internalPerimeter + d.b.internalPerimeter = d.c.internalPerimeter) :=
sorry

end NUMINAMATH_CALUDE_grid_division_exists_l2470_247006


namespace NUMINAMATH_CALUDE_black_socks_bought_is_12_l2470_247083

/-- The number of pairs of black socks Dmitry bought -/
def black_socks_bought : ℕ := sorry

/-- The initial number of blue sock pairs -/
def initial_blue : ℕ := 14

/-- The initial number of black sock pairs -/
def initial_black : ℕ := 24

/-- The initial number of white sock pairs -/
def initial_white : ℕ := 10

/-- The total number of sock pairs after buying more black socks -/
def total_after : ℕ := initial_blue + initial_white + initial_black + black_socks_bought

/-- The number of black sock pairs after buying more -/
def black_after : ℕ := initial_black + black_socks_bought

theorem black_socks_bought_is_12 : 
  black_socks_bought = 12 ∧ 
  black_after = (3 : ℚ) / 5 * total_after := by sorry

end NUMINAMATH_CALUDE_black_socks_bought_is_12_l2470_247083


namespace NUMINAMATH_CALUDE_count_valid_formations_l2470_247066

/-- The number of musicians in the marching band -/
def total_musicians : ℕ := 420

/-- The minimum number of musicians per row -/
def min_per_row : ℕ := 12

/-- The maximum number of musicians per row -/
def max_per_row : ℕ := 50

/-- A predicate that checks if a pair (s, t) forms a valid formation -/
def is_valid_formation (s t : ℕ) : Prop :=
  s * t = total_musicians ∧ min_per_row ≤ t ∧ t ≤ max_per_row

/-- The theorem stating that there are exactly 8 valid formations -/
theorem count_valid_formations :
  ∃! (formations : Finset (ℕ × ℕ)),
    formations.card = 8 ∧
    ∀ (s t : ℕ), (s, t) ∈ formations ↔ is_valid_formation s t :=
by sorry

end NUMINAMATH_CALUDE_count_valid_formations_l2470_247066


namespace NUMINAMATH_CALUDE_function_properties_imply_b_range_l2470_247068

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def zero_points_count (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (zeros : Finset ℝ), zeros.card = n ∧ (∀ x ∈ zeros, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem function_properties_imply_b_range (f : ℝ → ℝ) (b : ℝ) :
  is_odd_function f →
  has_period f 4 →
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log (x^2 - x + b)) →
  zero_points_count f (-2) 2 5 →
  (1/4 < b ∧ b ≤ 1) ∨ b = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_imply_b_range_l2470_247068


namespace NUMINAMATH_CALUDE_stable_painted_area_l2470_247026

/-- Calculates the total area to be painted for a rectangular stable with a chimney -/
def total_painted_area (width length height chim_width chim_length chim_height : ℝ) : ℝ :=
  let wall_area_1 := 2 * 2 * (width * height)
  let wall_area_2 := 2 * 2 * (length * height)
  let roof_area := width * length
  let ceiling_area := width * length
  let chimney_area := 4 * (chim_width * chim_height) + (chim_width * chim_length)
  wall_area_1 + wall_area_2 + roof_area + ceiling_area + chimney_area

/-- Theorem stating that the total area to be painted for the given stable is 1060 sq yd -/
theorem stable_painted_area :
  total_painted_area 12 15 6 2 2 2 = 1060 := by
  sorry

end NUMINAMATH_CALUDE_stable_painted_area_l2470_247026


namespace NUMINAMATH_CALUDE_salt_solution_problem_l2470_247011

theorem salt_solution_problem (x : ℝ) : 
  x > 0 →  -- Ensure x is positive
  let initial_salt := 0.2 * x
  let after_evaporation := 0.75 * x
  let final_volume := after_evaporation + 7 + 14
  let final_salt := initial_salt + 14
  (final_salt / final_volume = 1/3) →
  x = 140 :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_problem_l2470_247011


namespace NUMINAMATH_CALUDE_work_completion_time_l2470_247010

theorem work_completion_time (b a_and_b : ℚ) (hb : b = 35) (hab : a_and_b = 20 / 11) :
  let a : ℚ := (1 / a_and_b - 1 / b)⁻¹
  a = 700 / 365 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2470_247010


namespace NUMINAMATH_CALUDE_simplest_fraction_with_conditions_l2470_247099

theorem simplest_fraction_with_conditions (a b : ℕ) : 
  (a : ℚ) / b = 45 / 56 →
  ∃ (x : ℕ), a = x^2 →
  ∃ (y : ℕ), b = y^3 →
  ∃ (c d : ℕ), (c : ℚ) / d = 1 ∧ 
    (∀ (e f : ℕ), (e : ℚ) / f = 45 / 56 → 
      (∃ (g : ℕ), e = g^2) → 
      (∃ (h : ℕ), f = h^3) → 
      (c : ℚ) / d ≤ (e : ℚ) / f) :=
by sorry

end NUMINAMATH_CALUDE_simplest_fraction_with_conditions_l2470_247099


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2470_247098

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point on the curve where we want to find the tangent line -/
def x₀ : ℝ := 1

/-- The y-coordinate of the point on the curve -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line at the point (x₀, y₀) -/
def m : ℝ := f' x₀

theorem tangent_line_equation :
  ∀ x y : ℝ, (x - x₀) = m * (y - y₀) ↔ x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2470_247098


namespace NUMINAMATH_CALUDE_prob_three_red_modified_deck_l2470_247084

/-- A deck of cards with red and black suits -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_red_cards : red_cards ≤ total_cards)

/-- The probability of drawing three red cards in a row -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards * (d.red_cards - 1) * (d.red_cards - 2)) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- The deck described in the problem -/
def modified_deck : Deck :=
  { total_cards := 60,
    red_cards := 36,
    h_red_cards := by norm_num }

theorem prob_three_red_modified_deck :
  prob_three_red modified_deck = 140 / 673 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_modified_deck_l2470_247084


namespace NUMINAMATH_CALUDE_base_number_proof_l2470_247018

theorem base_number_proof (x : ℝ) : 16^7 = x^14 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2470_247018


namespace NUMINAMATH_CALUDE_system_solution_l2470_247054

theorem system_solution (x y z b : ℝ) : 
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) →
  ((b = 0 ∧ ((x = 0 ∧ z = -y) ∨ (y = 0 ∧ z = -x))) ∨
   (b ≠ 0 ∧ z = 0 ∧ 
    ((x = (1 + Real.sqrt (-1/2)) * b ∧ y = (1 - Real.sqrt (-1/2)) * b) ∨
     (x = (1 - Real.sqrt (-1/2)) * b ∧ y = (1 + Real.sqrt (-1/2)) * b)))) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2470_247054


namespace NUMINAMATH_CALUDE_min_sum_and_min_product_l2470_247049

/-- An arithmetic sequence with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of first n terms
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- The specific arithmetic sequence satisfying given conditions -/
def special_sequence (a : ArithmeticSequence) : Prop :=
  a.S 10 = 0 ∧ a.S 15 = 25

theorem min_sum_and_min_product (a : ArithmeticSequence) 
  (h : special_sequence a) :
  (∀ n : ℕ, n > 0 → a.S n ≥ a.S 5) ∧ 
  (∀ n : ℕ, n > 0 → n * (a.S n) ≥ -49) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_and_min_product_l2470_247049


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2470_247088

-- Define the equation
def equation (x : ℝ) : Prop :=
  2 * Real.cos (2 * x) * (Real.cos (2 * x) - Real.cos (2000 * Real.pi ^ 2 / x)) = Real.cos (4 * x) - 1

-- Define the set of all positive real solutions
def solution_set : Set ℝ := {x | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x ∈ solution_set) ∧
                    (∀ x ∈ solution_set, x ∈ S) ∧
                    (Finset.sum S id = 136 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2470_247088


namespace NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l2470_247097

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Determines if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Given point A in the second quadrant, prove that point B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant (m n : ℝ) (h : isInSecondQuadrant ⟨m, n⟩) :
  isInFourthQuadrant ⟨2*n - m, -n + m⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l2470_247097


namespace NUMINAMATH_CALUDE_negation_of_existence_logarithm_l2470_247044

theorem negation_of_existence_logarithm (x : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_logarithm_l2470_247044


namespace NUMINAMATH_CALUDE_min_value_product_l2470_247055

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 9) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 57 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a + b/a + c/b + a/c = 9 ∧
    (a/b + b/c + c/a) * (b/a + c/b + a/c) = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2470_247055


namespace NUMINAMATH_CALUDE_exactly_one_zero_two_zeros_greater_than_neg_one_l2470_247064

-- Define the function f(x) in terms of m
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 3*m + 4

-- Theorem for condition 1
theorem exactly_one_zero (m : ℝ) :
  (∃! x, f m x = 0) ↔ (m = 4 ∨ m = -1) :=
sorry

-- Theorem for condition 2
theorem two_zeros_greater_than_neg_one (m : ℝ) :
  (∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ 
  (m > -5 ∧ m < -1) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_zero_two_zeros_greater_than_neg_one_l2470_247064


namespace NUMINAMATH_CALUDE_symmetry_probability_l2470_247053

/-- Represents a point on the grid --/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The center point of the grid --/
def centerPoint : GridPoint :=
  ⟨5, 5⟩

/-- The set of all points on the grid --/
def allPoints : Finset GridPoint :=
  sorry

/-- The set of all points except the center point --/
def nonCenterPoints : Finset GridPoint :=
  sorry

/-- Predicate to check if a line through two points is a line of symmetry --/
def isSymmetryLine (p q : GridPoint) : Prop :=
  sorry

/-- The set of points that form symmetry lines with the center point --/
def symmetryPoints : Finset GridPoint :=
  sorry

theorem symmetry_probability :
    (symmetryPoints.card : ℚ) / (nonCenterPoints.card : ℚ) = 1 / 3 :=
  sorry

end NUMINAMATH_CALUDE_symmetry_probability_l2470_247053


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2470_247043

-- Problem 1
theorem problem_1 : (-7) - (-8) + (-9) - 14 = -22 := by sorry

-- Problem 2
theorem problem_2 : (-4) * (-3)^2 - 14 / (-7) = -34 := by sorry

-- Problem 3
theorem problem_3 : (3/10 - 1/4 + 4/5) * (-20) = -17 := by sorry

-- Problem 4
theorem problem_4 : (-2)^2 / |1-3| + 3 * (1/2 - 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2470_247043


namespace NUMINAMATH_CALUDE_total_time_wasted_l2470_247030

def traffic_wait_time : ℝ := 2
def freeway_exit_time_multiplier : ℝ := 4

theorem total_time_wasted : 
  traffic_wait_time + freeway_exit_time_multiplier * traffic_wait_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_time_wasted_l2470_247030


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2470_247038

theorem prime_sum_theorem (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_less : p < q) (h_eq : p * q + p^2 + q^2 = 199) : 
  (Finset.range (q - p)).sum (fun k => 2 / ((p + k) * (p + k + 1))) = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2470_247038


namespace NUMINAMATH_CALUDE_altitude_triangle_min_side_range_l2470_247036

/-- A triangle with side lengths a, b, c, perimeter 1, and altitudes that form a new triangle -/
structure AltitudeTriangle where
  a : Real
  b : Real
  c : Real
  perimeter_one : a + b + c = 1
  altitudes_form_triangle : 1/a + 1/b > 1/c ∧ 1/b + 1/c > 1/a ∧ 1/c + 1/a > 1/b
  a_smallest : a ≤ b ∧ a ≤ c

theorem altitude_triangle_min_side_range (t : AltitudeTriangle) :
  (3 - Real.sqrt 5) / 4 < t.a ∧ t.a ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_altitude_triangle_min_side_range_l2470_247036


namespace NUMINAMATH_CALUDE_conors_work_week_l2470_247085

/-- Conor's vegetable chopping problem -/
theorem conors_work_week (eggplants carrots potatoes total : ℕ) 
  (h1 : eggplants = 12)
  (h2 : carrots = 9)
  (h3 : potatoes = 8)
  (h4 : total = 116) : 
  total / (eggplants + carrots + potatoes) = 4 := by
  sorry

#check conors_work_week

end NUMINAMATH_CALUDE_conors_work_week_l2470_247085


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2470_247048

theorem sin_cos_identity : 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) - 
  Real.sin (69 * π / 180) * Real.cos (9 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2470_247048


namespace NUMINAMATH_CALUDE_no_division_between_valid_numbers_l2470_247092

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ),
    d₁ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₂ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₃ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₄ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₅ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₆ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₇ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ d₁ ≠ d₇ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ d₂ ≠ d₇ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ d₃ ≠ d₇ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ d₄ ≠ d₇ ∧
    d₅ ≠ d₆ ∧ d₅ ≠ d₇ ∧
    d₆ ≠ d₇ ∧
    n = d₁ * 1000000 + d₂ * 100000 + d₃ * 10000 + d₄ * 1000 + d₅ * 100 + d₆ * 10 + d₇

theorem no_division_between_valid_numbers :
  ∀ a b : ℕ, is_valid_number a → is_valid_number b → a ≠ b → ¬(a ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_no_division_between_valid_numbers_l2470_247092


namespace NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2470_247012

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  r : ℝ
  h_geom : ∀ n, b (n + 1) = r * b n

/-- The theorem statement -/
theorem arithmetic_geometric_relation (seq : ArithmeticSequence)
    (h_geom : ∃ (g : GeometricSequence), 
      g.b 1 = seq.a 2 ∧ g.b 2 = seq.a 3 ∧ g.b 3 = seq.a 7) :
    (∃ (g : GeometricSequence), 
      g.b 1 = seq.a 2 ∧ g.b 2 = seq.a 3 ∧ g.b 3 = seq.a 7 ∧ g.r = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2470_247012


namespace NUMINAMATH_CALUDE_parallel_lines_a_perpendicular_lines_a_l2470_247057

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - 3/2 = 0

-- Parallel lines condition
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Perpendicular lines condition
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- Theorem for parallel lines
theorem parallel_lines_a (a : ℝ) :
  parallel a → a = 4 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines_a (a : ℝ) :
  perpendicular a → a = 0 ∨ a = -20/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_perpendicular_lines_a_l2470_247057
