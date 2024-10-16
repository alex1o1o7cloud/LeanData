import Mathlib

namespace NUMINAMATH_CALUDE_negative_five_plus_abs_negative_three_equals_negative_two_l1979_197996

theorem negative_five_plus_abs_negative_three_equals_negative_two :
  -5 + |(-3)| = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_plus_abs_negative_three_equals_negative_two_l1979_197996


namespace NUMINAMATH_CALUDE_largest_B_divisible_by_nine_l1979_197958

def number (B : Nat) : Nat := 5 * 100000 + B * 10000 + 4 * 1000 + 8 * 100 + 6 * 10 + 1

theorem largest_B_divisible_by_nine :
  ∀ B : Nat, B < 10 →
    (∃ m : Nat, number B = 9 * m) →
    B ≤ 9 ∧
    (∀ C : Nat, C < 10 → C > B → ¬∃ n : Nat, number C = 9 * n) :=
by sorry

end NUMINAMATH_CALUDE_largest_B_divisible_by_nine_l1979_197958


namespace NUMINAMATH_CALUDE_perpendicular_vectors_cos2theta_l1979_197952

theorem perpendicular_vectors_cos2theta (θ : ℝ) : 
  let a : ℝ × ℝ := (1, Real.cos θ)
  let b : ℝ × ℝ := (-1, 2 * Real.cos θ)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.cos (2 * θ) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_cos2theta_l1979_197952


namespace NUMINAMATH_CALUDE_exists_divisible_figure_l1979_197943

/-- Represents a geometric shape --/
structure Shape :=
  (area : ℝ)

/-- Represents a T-shaped piece --/
def T_shape : Shape :=
  { area := 3 }

/-- Represents the set of five specific pieces --/
def five_pieces : Finset Shape :=
  sorry

/-- A figure that can be divided into different sets of pieces --/
structure DivisibleFigure :=
  (total_area : ℝ)
  (can_divide_into_four_T : Prop)
  (can_divide_into_five_pieces : Prop)

/-- The existence of a figure that satisfies both division conditions --/
theorem exists_divisible_figure : 
  ∃ (fig : DivisibleFigure), 
    fig.can_divide_into_four_T ∧ 
    fig.can_divide_into_five_pieces :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_figure_l1979_197943


namespace NUMINAMATH_CALUDE_range_of_m_l1979_197988

-- Define a decreasing function on [-1, 1]
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x > f y

-- Main theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : IsDecreasingOn f)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  m ∈ Set.Ioo 0 1 := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_range_of_m_l1979_197988


namespace NUMINAMATH_CALUDE_largest_s_value_l1979_197992

/-- The interior angle of a regular n-gon -/
def interior_angle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- The largest possible value of s for regular polygons Q_1 (r-gon) and Q_2 (s-gon) -/
theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) 
  (h_ratio : interior_angle r / interior_angle s = 39 / 38) : 
  s ≤ 76 ∧ ∃ (r' : ℕ), r' ≥ 76 ∧ interior_angle r' / interior_angle 76 = 39 / 38 :=
sorry

end NUMINAMATH_CALUDE_largest_s_value_l1979_197992


namespace NUMINAMATH_CALUDE_girls_in_school_l1979_197900

/-- The number of girls in a school after new students join -/
def total_girls (initial_girls new_girls : ℕ) : ℕ :=
  initial_girls + new_girls

/-- Theorem stating that the total number of girls after new students joined is 1414 -/
theorem girls_in_school (initial_girls new_girls : ℕ) 
  (h1 : initial_girls = 732) 
  (h2 : new_girls = 682) : 
  total_girls initial_girls new_girls = 1414 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l1979_197900


namespace NUMINAMATH_CALUDE_decimal_binary_equality_l1979_197957

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec toBinary (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- Define a function to convert binary to decimal
def binaryToDecimal (bits : List Nat) : Nat :=
  bits.foldl (fun acc bit => 2 * acc + bit) 0

-- Theorem statement
theorem decimal_binary_equality :
  (decimalToBinary 25 ≠ [1, 0, 1, 1, 0]) ∧
  (decimalToBinary 13 = [1, 1, 0, 1]) ∧
  (decimalToBinary 11 ≠ [1, 1, 0, 0]) ∧
  (decimalToBinary 10 ≠ [1, 0]) :=
by sorry

end NUMINAMATH_CALUDE_decimal_binary_equality_l1979_197957


namespace NUMINAMATH_CALUDE_triangle_properties_l1979_197939

theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Acute triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →  -- Sine rule
  (Real.sin A + Real.sin B)^2 = (2 * Real.sin B + Real.sin C) * Real.sin C →  -- Given equation
  Real.sin A > Real.sqrt 3 / 3 →  -- Given inequality
  (c - a = a * Real.cos C) ∧ (c > a) ∧ (C > π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1979_197939


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_perfect_square_coefficient_l1979_197974

/-- A quadratic expression is a perfect square if and only if its discriminant is zero -/
theorem quadratic_is_perfect_square (a b c : ℝ) :
  (∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2) ↔ b^2 = 4 * a * c := by sorry

/-- The main theorem: If 6x^2 + cx + 16 is a perfect square, then c = 8√6 -/
theorem perfect_square_coefficient (c : ℝ) :
  (∃ p q : ℝ, ∀ x, 6 * x^2 + c * x + 16 = (p * x + q)^2) → c = 8 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_perfect_square_coefficient_l1979_197974


namespace NUMINAMATH_CALUDE_shortest_distance_moving_points_l1979_197905

/-- The shortest distance between two points moving along perpendicular edges of a square -/
theorem shortest_distance_moving_points (side_length : ℝ) (v1 v2 : ℝ) 
  (h1 : side_length = 10)
  (h2 : v1 = 30 / 100)
  (h3 : v2 = 40 / 100) :
  ∃ t : ℝ, ∃ x y : ℝ,
    x = v1 * t ∧
    y = v2 * t ∧
    ∀ s : ℝ, (v1 * s - side_length)^2 + (v2 * s)^2 ≥ x^2 + y^2 ∧
    Real.sqrt (x^2 + y^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_moving_points_l1979_197905


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1979_197917

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - 1| > a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1979_197917


namespace NUMINAMATH_CALUDE_rain_all_three_days_l1979_197956

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.2

theorem rain_all_three_days :
  let prob_all_days := prob_rain_friday * prob_rain_saturday * prob_rain_sunday
  prob_all_days = 0.04 := by sorry

end NUMINAMATH_CALUDE_rain_all_three_days_l1979_197956


namespace NUMINAMATH_CALUDE_davids_physics_marks_l1979_197961

def english_marks : ℕ := 45
def math_marks : ℕ := 35
def chemistry_marks : ℕ := 47
def biology_marks : ℕ := 55
def average_marks : ℚ := 46.8
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 52 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l1979_197961


namespace NUMINAMATH_CALUDE_find_divisor_l1979_197933

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 507 → quotient = 61 → remainder = 19 →
  ∃ (divisor : Nat), dividend = divisor * quotient + remainder ∧ divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1979_197933


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1979_197983

theorem sqrt_sum_equality : 
  Real.sqrt 9 + Real.sqrt (9 + 11) + Real.sqrt (9 + 11 + 13) + 
  Real.sqrt (9 + 11 + 13 + 15) + Real.sqrt (9 + 11 + 13 + 15 + 17) = 
  3 + 2 * Real.sqrt 5 + Real.sqrt 33 + 4 * Real.sqrt 3 + Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1979_197983


namespace NUMINAMATH_CALUDE_three_lines_exist_l1979_197991

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : Option ℝ
  intercept : ℝ

/-- The hyperbola x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- A line passes through the point (√2, 0) -/
def passesThrough (l : Line) : Prop :=
  match l.slope with
  | none => l.intercept = Real.sqrt 2
  | some m => l.intercept = -m * Real.sqrt 2

/-- A line has exactly one common point with the hyperbola -/
def hasOneCommonPoint (l : Line) : Prop :=
  ∃! p : ℝ × ℝ, (match l.slope with
    | none => p.1 = l.intercept ∧ p.2 = 0
    | some m => p.2 = m * p.1 + l.intercept) ∧ hyperbola p.1 p.2

/-- The main theorem: there are exactly 3 lines satisfying the conditions -/
theorem three_lines_exist :
  ∃! (lines : Finset Line), lines.card = 3 ∧
    ∀ l ∈ lines, passesThrough l ∧ hasOneCommonPoint l :=
sorry

end NUMINAMATH_CALUDE_three_lines_exist_l1979_197991


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1979_197907

/-- Given a geometric sequence with first term 243 and eighth term 32,
    the sixth term of the sequence is 1. -/
theorem geometric_sequence_sixth_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 243 →
    a * r^7 = 32 →
    a * r^5 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1979_197907


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1979_197970

theorem quadratic_equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = -3 ∧ 
  (∀ x : ℝ, x^2 + 2*x + 1 = 4 ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1979_197970


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l1979_197960

theorem sun_radius_scientific_notation :
  696000 = 6.96 * (10 ^ 5) := by sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l1979_197960


namespace NUMINAMATH_CALUDE_range_of_a_for_surjective_f_l1979_197995

/-- A piecewise function f dependent on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else 2^(x - 1)

/-- The theorem stating the relationship between the range of f and the range of a -/
theorem range_of_a_for_surjective_f :
  (∀ a : ℝ, Set.range (f a) = Set.univ) ↔ (Set.Icc 0 (1/2) : Set ℝ) = {a : ℝ | 0 ≤ a ∧ a < 1/2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_surjective_f_l1979_197995


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l1979_197973

theorem shirt_price_calculation (total_cost sweater_price shirt_price : ℝ) :
  total_cost = 80.34 →
  sweater_price - shirt_price = 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l1979_197973


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1979_197904

/-- Given a geometric sequence {a_n} with a₃ = 6 and S₃ = 18,
    prove that the common ratio q is either 1 or -1/2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = 6)
  (h_S3 : a 1 + a 2 + a 3 = 18) :
  q = 1 ∨ q = -1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1979_197904


namespace NUMINAMATH_CALUDE_odd_function_range_l1979_197911

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h_odd : IsOdd f)
    (h_neg : ∀ x, x < 0 → f x = 9*x + a^2/x + 7)
    (h_pos : ∀ x, x ≥ 0 → f x ≥ a + 1) :
  a ≤ -8/7 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_range_l1979_197911


namespace NUMINAMATH_CALUDE_excellent_students_probability_l1979_197979

/-- The probability of selecting exactly 4 excellent students when randomly choosing 7 students from a class of 10 students, where 6 are excellent, is equal to 0.5. -/
theorem excellent_students_probability :
  let total_students : ℕ := 10
  let excellent_students : ℕ := 6
  let selected_students : ℕ := 7
  let target_excellent : ℕ := 4
  (Nat.choose excellent_students target_excellent * Nat.choose (total_students - excellent_students) (selected_students - target_excellent)) / Nat.choose total_students selected_students = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_excellent_students_probability_l1979_197979


namespace NUMINAMATH_CALUDE_factorization_problem_fraction_simplification_l1979_197997

-- Factorization problem
theorem factorization_problem (m : ℝ) : m^3 - 4*m^2 + 4*m = m*(m-2)^2 := by
  sorry

-- Fraction simplification problem
theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_fraction_simplification_l1979_197997


namespace NUMINAMATH_CALUDE_flu_infection_equation_l1979_197951

/-- 
Given:
- One person initially has the flu
- Each person infects x people on average in each round
- There are two rounds of infection
- After two rounds, 144 people have the flu

Prove that (1 + x)^2 = 144 correctly represents the total number of infected people.
-/
theorem flu_infection_equation (x : ℝ) : (1 + x)^2 = 144 :=
sorry

end NUMINAMATH_CALUDE_flu_infection_equation_l1979_197951


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1979_197986

theorem trigonometric_identity (t : ℝ) : 
  1 + Real.sin (t/2) * Real.sin t - Real.cos (t/2) * (Real.sin t)^2 = 
  2 * (Real.cos (π/4 - t/2))^2 ↔ 
  ∃ k : ℤ, t = k * π := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1979_197986


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_l1979_197950

/-- Represents a digital time display in 24-hour format -/
structure TimeDisplay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60
  seconds_valid : seconds < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  n.repr.foldl (fun sum c => sum + c.toNat - '0'.toNat) 0

/-- Calculates the sum of digits for a time display -/
def sumOfTimeDigits (t : TimeDisplay) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The largest possible sum of digits in a 24-hour format digital watch display is 38 -/
theorem largest_sum_of_digits : ∀ t : TimeDisplay, sumOfTimeDigits t ≤ 38 := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_l1979_197950


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l1979_197977

theorem simple_random_sampling_probability 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (selected_students : ℕ) 
  (h1 : total_students = 100) 
  (h2 : male_students = 25) 
  (h3 : selected_students = 20) 
  (h4 : male_students ≤ total_students) 
  (h5 : selected_students ≤ total_students) : 
  (selected_students : ℚ) / total_students = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l1979_197977


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_equal_absolute_value_l1979_197963

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (m + 3) * x + m + 2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: The absolute values of the roots are equal iff m = -1 or m = -3
theorem roots_equal_absolute_value (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ |x₁| = |x₂|) ↔
  (m = -1 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_equal_absolute_value_l1979_197963


namespace NUMINAMATH_CALUDE_max_sum_squares_l1979_197989

theorem max_sum_squares (m n : ℕ) : 
  m ∈ Finset.range 101 ∧ 
  n ∈ Finset.range 101 ∧ 
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 10946 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l1979_197989


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1979_197954

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ ¬((a - b) * a^2 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1979_197954


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l1979_197987

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, 3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 9 * x - 4 * y = 6

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 7 * x + y = 17

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ (x y : ℝ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l1979_197987


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1979_197932

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1979_197932


namespace NUMINAMATH_CALUDE_height_prediction_approximate_l1979_197946

/-- Regression model for height prediction -/
def height_model (x : ℝ) : ℝ := 7.19 * x + 73.93

/-- The age at which we want to predict the height -/
def prediction_age : ℝ := 10

/-- The predicted height at the given age -/
def predicted_height : ℝ := height_model prediction_age

theorem height_prediction_approximate :
  ∃ ε > 0, abs (predicted_height - 145.83) < ε :=
sorry

end NUMINAMATH_CALUDE_height_prediction_approximate_l1979_197946


namespace NUMINAMATH_CALUDE_power_function_through_2_4_l1979_197972

/-- A power function that passes through the point (2, 4) is equivalent to f(x) = x^2 -/
theorem power_function_through_2_4 (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x^α) →  -- f is a power function with exponent α
  f 2 = 4 →           -- f passes through the point (2, 4)
  ∀ x, f x = x^2 :=   -- f is equivalent to x^2
by sorry

end NUMINAMATH_CALUDE_power_function_through_2_4_l1979_197972


namespace NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l1979_197984

/-- The number of large glasses Peter bought -/
def num_large_glasses (small_cost large_cost total_money num_small change : ℕ) : ℕ :=
  (total_money - change - small_cost * num_small) / large_cost

/-- Proof that Peter bought 5 large glasses -/
theorem peter_bought_five_large_glasses :
  num_large_glasses 3 5 50 8 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l1979_197984


namespace NUMINAMATH_CALUDE_divisibility_condition_l1979_197953

theorem divisibility_condition (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  n ∣ (1 + m^(3^n) + m^(2*3^n)) ↔ ∃ t : ℕ+, n = 3 ∧ m = 3 * t - 2 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1979_197953


namespace NUMINAMATH_CALUDE_evening_emails_count_l1979_197990

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon and evening combined -/
def afternoon_and_evening_emails : ℕ := 13

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := afternoon_and_evening_emails - afternoon_emails

theorem evening_emails_count : evening_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_evening_emails_count_l1979_197990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1979_197942

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) (m : ℕ) 
  (h_m : m > 1)
  (h_condition : seq.a (m - 1) + seq.a (m + 1) - (seq.a m)^2 = 0)
  (h_sum : seq.S (2 * m - 1) = 38) :
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1979_197942


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1979_197937

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 3 * x + 2 > 0) ↔ (b < x ∧ x < 1)) → 
  (a = -5 ∧ b = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1979_197937


namespace NUMINAMATH_CALUDE_smallest_zucchini_count_l1979_197929

def is_divisible_by_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k * k * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_6 n ∧ is_perfect_square (n / 2) ∧ is_perfect_cube (n / 3)

theorem smallest_zucchini_count :
  satisfies_conditions 648 ∧ ∀ m : ℕ, m < 648 → ¬(satisfies_conditions m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_zucchini_count_l1979_197929


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1979_197927

theorem decimal_multiplication : (0.25 : ℝ) * 0.75 * 0.1 = 0.01875 := by sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1979_197927


namespace NUMINAMATH_CALUDE_min_exponent_sum_520_l1979_197964

/-- Given a natural number n, returns the minimum sum of exponents when expressing n as a sum of at least two distinct powers of 2 -/
def min_exponent_sum (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the minimum sum of exponents when expressing 520 as a sum of at least two distinct powers of 2 is 12 -/
theorem min_exponent_sum_520 : min_exponent_sum 520 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_exponent_sum_520_l1979_197964


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_angle_l1979_197980

theorem isosceles_right_triangle_angle (a h : ℝ) (θ : Real) : 
  a > 0 → -- leg length is positive
  h > 0 → -- hypotenuse length is positive
  h = a * Real.sqrt 2 → -- Pythagorean theorem for isosceles right triangle
  h^2 = 4 * a * Real.cos θ → -- given condition
  0 < θ ∧ θ < Real.pi / 2 → -- θ is an acute angle
  θ = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_angle_l1979_197980


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1979_197938

/-- Represents a rectangular field with a width that is one-third of its length -/
structure RectangularField where
  length : ℝ
  width : ℝ
  width_is_third_of_length : width = length / 3
  perimeter_is_72 : 2 * (length + width) = 72

/-- The area of a rectangular field with the given conditions is 243 square meters -/
theorem rectangular_field_area (field : RectangularField) : field.length * field.width = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1979_197938


namespace NUMINAMATH_CALUDE_river_distance_l1979_197920

/-- The distance between two points on a river, given boat speeds and time difference -/
theorem river_distance (v_down v_up : ℝ) (time_diff : ℝ) (h1 : v_down = 20)
  (h2 : v_up = 15) (h3 : time_diff = 5) :
  ∃ d : ℝ, d = 300 ∧ d / v_up - d / v_down = time_diff :=
by sorry

end NUMINAMATH_CALUDE_river_distance_l1979_197920


namespace NUMINAMATH_CALUDE_prob_two_heads_with_second_tail_l1979_197926

/-- A fair coin flip sequence that ends with either two heads or two tails in a row -/
inductive CoinFlipSequence
| TH : CoinFlipSequence → CoinFlipSequence
| TT : CoinFlipSequence
| HH : CoinFlipSequence

/-- The probability of a specific coin flip sequence -/
def probability (seq : CoinFlipSequence) : ℚ :=
  match seq with
  | CoinFlipSequence.TH s => (1/2) * probability s
  | CoinFlipSequence.TT => (1/2) * (1/2)
  | CoinFlipSequence.HH => (1/2) * (1/2)

/-- The probability of getting two heads in a row while seeing a second tail before seeing a second head -/
def probTwoHeadsWithSecondTail : ℚ :=
  (1/2) * (1/2) * (1/2) * (1/3)

theorem prob_two_heads_with_second_tail :
  probTwoHeadsWithSecondTail = 1/24 :=
sorry

end NUMINAMATH_CALUDE_prob_two_heads_with_second_tail_l1979_197926


namespace NUMINAMATH_CALUDE_birds_on_fence_l1979_197902

/-- Given an initial number of birds and a final total number of birds,
    calculate the number of birds that joined. -/
def birds_joined (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 2 initial birds and 6 final birds,
    the number of birds that joined is 4. -/
theorem birds_on_fence : birds_joined 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1979_197902


namespace NUMINAMATH_CALUDE_ricks_books_l1979_197930

theorem ricks_books (N : ℕ) : (N / 2 / 2 / 2 / 2 = 25) → N = 400 := by
  sorry

end NUMINAMATH_CALUDE_ricks_books_l1979_197930


namespace NUMINAMATH_CALUDE_expression_factorization_l1979_197966

theorem expression_factorization (a b c : ℝ) :
  (((a^2 + 1) - (b^2 + 1))^3 + ((b^2 + 1) - (c^2 + 1))^3 + ((c^2 + 1) - (a^2 + 1))^3) /
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (b + c) * (c + a) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l1979_197966


namespace NUMINAMATH_CALUDE_intersection_range_l1979_197934

/-- Hyperbola C centered at the origin with right focus at (2,0) and real axis length 2√3 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- Line l with equation y = kx + √2 -/
def line_l (k x : ℝ) (y : ℝ) : Prop := y = k * x + Real.sqrt 2

/-- Predicate to check if a point (x, y) is on the left branch of hyperbola C -/
def on_left_branch (x y : ℝ) : Prop := hyperbola_C x y ∧ x < 0

/-- Theorem stating the range of k for which line l intersects the left branch of hyperbola C at two points -/
theorem intersection_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    on_left_branch x₁ y₁ ∧ 
    on_left_branch x₂ y₂ ∧ 
    line_l k x₁ y₁ ∧ 
    line_l k x₂ y₂) ↔ 
  Real.sqrt 3 / 3 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1979_197934


namespace NUMINAMATH_CALUDE_andrew_fruit_purchase_l1979_197925

/-- The total cost of a fruit purchase, including tax -/
def totalCost (grapeQuantity mangoQuantity grapePrice mangoPrice grapeTaxRate mangoTaxRate : ℚ) : ℚ :=
  let grapeCost := grapeQuantity * grapePrice
  let mangoCost := mangoQuantity * mangoPrice
  let grapeTax := grapeCost * grapeTaxRate
  let mangoTax := mangoCost * mangoTaxRate
  grapeCost + mangoCost + grapeTax + mangoTax

/-- The theorem stating the total cost of Andrew's fruit purchase -/
theorem andrew_fruit_purchase :
  totalCost 8 9 70 55 (8/100) (11/100) = 1154.25 := by
  sorry

end NUMINAMATH_CALUDE_andrew_fruit_purchase_l1979_197925


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l1979_197906

/-- The cost of one blue pill in dollars -/
def blue_pill_cost : ℚ := 11

/-- The number of days in the treatment period -/
def days : ℕ := 21

/-- The daily discount in dollars after the first week -/
def daily_discount : ℚ := 2

/-- The number of days with discount -/
def discount_days : ℕ := 14

/-- The total cost without discount for the entire period -/
def total_cost_without_discount : ℚ := 735

/-- The number of blue pills taken daily -/
def daily_blue_pills : ℕ := 2

/-- The number of orange pills taken daily -/
def daily_orange_pills : ℕ := 1

/-- The cost difference between orange and blue pills in dollars -/
def orange_blue_cost_difference : ℚ := 2

theorem blue_pill_cost_proof :
  blue_pill_cost * (daily_blue_pills * days + daily_orange_pills * days) +
  orange_blue_cost_difference * (daily_orange_pills * days) -
  daily_discount * discount_days = total_cost_without_discount - daily_discount * discount_days :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l1979_197906


namespace NUMINAMATH_CALUDE_rattlesnake_count_l1979_197968

theorem rattlesnake_count (total_snakes : ℕ) (boa_constrictors : ℕ) : 
  total_snakes = 200 →
  boa_constrictors = 40 →
  total_snakes = boa_constrictors + 3 * boa_constrictors + (total_snakes - (boa_constrictors + 3 * boa_constrictors)) →
  total_snakes - (boa_constrictors + 3 * boa_constrictors) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rattlesnake_count_l1979_197968


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l1979_197944

-- Define the triangles and their side lengths
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

def triangle_PQR : Triangle :=
  { side1 := 10,
    side2 := 12,
    side3 := 0 }  -- We don't know the length of PR

def triangle_STU : Triangle :=
  { side1 := 5,
    side2 := 0,  -- We need to prove this is 6
    side3 := 0 }  -- We need to prove this is 6

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.side1 = k * t1.side1 ∧
    t2.side2 = k * t1.side2 ∧
    t2.side3 = k * t1.side3

-- Define the theorem
theorem triangle_similarity_theorem :
  similar triangle_PQR triangle_STU →
  triangle_STU.side2 = 6 ∧
  triangle_STU.side3 = 6 ∧
  triangle_STU.side1 + triangle_STU.side2 + triangle_STU.side3 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l1979_197944


namespace NUMINAMATH_CALUDE_lowest_discount_l1979_197945

theorem lowest_discount (cost_price marked_price : ℝ) (min_profit_margin : ℝ) : 
  cost_price = 100 → 
  marked_price = 150 → 
  min_profit_margin = 0.05 → 
  ∃ (discount : ℝ), 
    discount = 0.7 ∧ 
    marked_price * discount = cost_price * (1 + min_profit_margin) ∧
    ∀ (d : ℝ), d > discount → marked_price * d > cost_price * (1 + min_profit_margin) :=
by sorry


end NUMINAMATH_CALUDE_lowest_discount_l1979_197945


namespace NUMINAMATH_CALUDE_five_girls_five_boys_arrangements_l1979_197941

/-- The number of ways to arrange n girls and n boys around a circular table
    such that no two people of the same gender sit next to each other -/
def alternatingArrangements (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

/-- Theorem: There are 28800 ways to arrange 5 girls and 5 boys around a circular table
    such that no two people of the same gender sit next to each other -/
theorem five_girls_five_boys_arrangements :
  alternatingArrangements 5 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_five_girls_five_boys_arrangements_l1979_197941


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1979_197913

theorem three_numbers_sum (a b c x y z : ℝ) : 
  (x + y = z + a) → 
  (x + z = y + b) → 
  (y + z = x + c) → 
  (x = (a + b - c) / 2) ∧ 
  (y = (a - b + c) / 2) ∧ 
  (z = (-a + b + c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1979_197913


namespace NUMINAMATH_CALUDE_valuable_files_count_l1979_197993

def initial_download : ℕ := 800
def first_deletion_rate : ℚ := 70 / 100
def second_download : ℕ := 400
def second_deletion_rate : ℚ := 3 / 5

theorem valuable_files_count :
  (initial_download - (initial_download * first_deletion_rate).floor) +
  (second_download - (second_download * second_deletion_rate).floor) = 400 := by
  sorry

end NUMINAMATH_CALUDE_valuable_files_count_l1979_197993


namespace NUMINAMATH_CALUDE_disease_cases_2005_2015_l1979_197994

/-- Calculates the number of disease cases in a given year, assuming a linear decrease. -/
def cases_in_year (initial_year initial_cases final_year final_cases target_year : ℕ) : ℕ :=
  initial_cases - (initial_cases - final_cases) * (target_year - initial_year) / (final_year - initial_year)

/-- Theorem stating the number of disease cases in 2005 and 2015 given the conditions. -/
theorem disease_cases_2005_2015 :
  cases_in_year 1970 300000 2020 100 2005 = 90070 ∧
  cases_in_year 1970 300000 2020 100 2015 = 30090 :=
by
  sorry

#eval cases_in_year 1970 300000 2020 100 2005
#eval cases_in_year 1970 300000 2020 100 2015

end NUMINAMATH_CALUDE_disease_cases_2005_2015_l1979_197994


namespace NUMINAMATH_CALUDE_intersection_M_N_l1979_197915

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x : ℕ | x - 1 ≥ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1979_197915


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1979_197931

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 9/2 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1/x + 1/y + 1/z = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1979_197931


namespace NUMINAMATH_CALUDE_triangle_coordinates_l1979_197935

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  P : Point
  Q : Point
  R : Point

/-- Predicate to check if a line segment is horizontal -/
def isHorizontal (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Predicate to check if a line segment is vertical -/
def isVertical (p1 p2 : Point) : Prop :=
  p1.x = p2.x

theorem triangle_coordinates (t : Triangle) 
  (h1 : isHorizontal t.P t.R)
  (h2 : isVertical t.P t.Q)
  (h3 : t.R.y = -2)
  (h4 : t.Q.x = -11) :
  t.P.x = -11 ∧ t.P.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_coordinates_l1979_197935


namespace NUMINAMATH_CALUDE_final_strawberry_count_l1979_197918

/-- The number of strawberry plants after n months of doubling, starting from an initial number. -/
def plants_after_months (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

/-- The theorem stating the final number of strawberry plants -/
theorem final_strawberry_count :
  let initial_plants : ℕ := 3
  let months_passed : ℕ := 3
  let plants_given_away : ℕ := 4
  (plants_after_months initial_plants months_passed) - plants_given_away = 20 :=
by sorry

end NUMINAMATH_CALUDE_final_strawberry_count_l1979_197918


namespace NUMINAMATH_CALUDE_value_of_a_l1979_197924

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1979_197924


namespace NUMINAMATH_CALUDE_a_less_than_b_l1979_197901

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) 
  (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l1979_197901


namespace NUMINAMATH_CALUDE_parabola_f_value_l1979_197978

/-- A parabola with equation y = dx^2 + ex + f, vertex at (3, -5), and passing through (4, -3) has f = 13 -/
theorem parabola_f_value (d e f : ℝ) : 
  (∀ x y : ℝ, y = d*x^2 + e*x + f) →  -- Parabola equation
  (-5 : ℝ) = d*(3:ℝ)^2 + e*(3:ℝ) + f → -- Vertex at (3, -5)
  (-3 : ℝ) = d*(4:ℝ)^2 + e*(4:ℝ) + f → -- Passes through (4, -3)
  f = 13 := by
sorry

end NUMINAMATH_CALUDE_parabola_f_value_l1979_197978


namespace NUMINAMATH_CALUDE_expenditure_increase_l1979_197949

theorem expenditure_increase 
  (income : ℝ) 
  (expenditure : ℝ) 
  (savings : ℝ) 
  (new_income : ℝ) 
  (new_expenditure : ℝ) 
  (new_savings : ℝ) 
  (h1 : expenditure = 0.75 * income) 
  (h2 : savings = income - expenditure) 
  (h3 : new_income = 1.2 * income) 
  (h4 : new_savings = 1.4999999999999996 * savings) 
  (h5 : new_savings = new_income - new_expenditure) : 
  new_expenditure = 1.1 * expenditure := by
sorry

end NUMINAMATH_CALUDE_expenditure_increase_l1979_197949


namespace NUMINAMATH_CALUDE_regular_polygon_30_degree_exterior_angle_l1979_197982

/-- A regular polygon with exterior angles measuring 30° has 12 sides -/
theorem regular_polygon_30_degree_exterior_angle (n : ℕ) :
  (n > 0) →
  (360 / n = 30) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_30_degree_exterior_angle_l1979_197982


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1979_197947

theorem expression_equals_zero (θ : Real) (h : Real.tan θ = 5) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1979_197947


namespace NUMINAMATH_CALUDE_range_of_a_for_intersecting_circles_l1979_197965

/-- Two circles intersect if and only if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
axiom circles_intersect_iff_distance_between_centers (x₁ y₁ x₂ y₂ r₁ r₂ : ℝ) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 > (r₁ - r₂)^2) ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 < (r₁ + r₂)^2)

/-- The range of a for intersecting circles -/
theorem range_of_a_for_intersecting_circles (a : ℝ) :
  (∃ x y : ℝ, (x + 2)^2 + (y - a)^2 = 1 ∧ (x - a)^2 + (y - 5)^2 = 16) ↔
  1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_intersecting_circles_l1979_197965


namespace NUMINAMATH_CALUDE_complementary_angle_measure_l1979_197998

-- Define the angle
def angle : ℝ := 45

-- Define the relationship between supplementary and complementary angles
def supplementary_complementary_relation (supplementary complementary : ℝ) : Prop :=
  supplementary = 3 * complementary

-- Define the supplementary angle
def supplementary (a : ℝ) : ℝ := 180 - a

-- Define the complementary angle
def complementary (a : ℝ) : ℝ := 90 - a

-- Theorem statement
theorem complementary_angle_measure :
  supplementary_complementary_relation (supplementary angle) (complementary angle) →
  complementary angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_measure_l1979_197998


namespace NUMINAMATH_CALUDE_chessboard_placements_l1979_197908

/-- Represents a standard 8x8 chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents the different types of chess pieces -/
inductive ChessPiece
| Rook
| King
| Bishop
| Knight
| Queen

/-- Returns true if two pieces of the given type at the given positions do not attack each other -/
def not_attacking (piece : ChessPiece) (pos1 pos2 : Chessboard) : Prop := sorry

/-- Counts the number of ways to place two identical pieces on the chessboard without attacking each other -/
def count_placements (piece : ChessPiece) : ℕ := sorry

theorem chessboard_placements :
  (count_placements ChessPiece.Rook = 1568) ∧
  (count_placements ChessPiece.King = 1806) ∧
  (count_placements ChessPiece.Bishop = 1736) ∧
  (count_placements ChessPiece.Knight = 1848) ∧
  (count_placements ChessPiece.Queen = 1288) := by sorry

end NUMINAMATH_CALUDE_chessboard_placements_l1979_197908


namespace NUMINAMATH_CALUDE_max_gum_pieces_is_31_l1979_197912

/-- Represents the number of coins Quentavious has -/
structure Coins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Represents the exchange rates for gum pieces -/
structure ExchangeRates where
  nickel_rate : ℕ
  dime_rate : ℕ
  quarter_rate : ℕ

/-- Represents the maximum number of coins that can be exchanged -/
structure MaxExchange where
  max_nickels : ℕ
  max_dimes : ℕ
  max_quarters : ℕ

/-- Calculates the maximum number of gum pieces Quentavious can get -/
def max_gum_pieces (coins : Coins) (rates : ExchangeRates) (max_exchange : MaxExchange) 
  (keep_nickels keep_dimes : ℕ) : ℕ :=
  let exchangeable_nickels := min (coins.nickels - keep_nickels) max_exchange.max_nickels
  let exchangeable_dimes := min (coins.dimes - keep_dimes) max_exchange.max_dimes
  let exchangeable_quarters := min coins.quarters max_exchange.max_quarters
  exchangeable_nickels * rates.nickel_rate + 
  exchangeable_dimes * rates.dime_rate + 
  exchangeable_quarters * rates.quarter_rate

/-- Theorem stating that the maximum number of gum pieces Quentavious can get is 31 -/
theorem max_gum_pieces_is_31 
  (coins : Coins)
  (rates : ExchangeRates)
  (max_exchange : MaxExchange)
  (h_coins : coins = ⟨5, 6, 4⟩)
  (h_rates : rates = ⟨2, 3, 5⟩)
  (h_max_exchange : max_exchange = ⟨3, 4, 2⟩)
  (h_keep_nickels : 2 ≤ coins.nickels)
  (h_keep_dimes : 1 ≤ coins.dimes) :
  max_gum_pieces coins rates max_exchange 2 1 = 31 :=
sorry

end NUMINAMATH_CALUDE_max_gum_pieces_is_31_l1979_197912


namespace NUMINAMATH_CALUDE_sum_reciprocal_products_l1979_197967

theorem sum_reciprocal_products (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_products_eq : x*y + y*z + z*x = 11)
  (product_eq : x*y*z = 6) :
  x/(y*z) + y/(z*x) + z/(x*y) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_products_l1979_197967


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1979_197985

theorem exponent_multiplication (m : ℝ) : 5 * m^2 * m^3 = 5 * m^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1979_197985


namespace NUMINAMATH_CALUDE_max_value_expression_l1979_197910

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_eq : c^2 = a^2 + b^2) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) ≤ 2 * a^2 + b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) = 2 * a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1979_197910


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1979_197955

theorem product_of_three_numbers (a b c : ℝ) : 
  (a + b + c = 44) → 
  (a^2 + b^2 + c^2 = 890) → 
  (a^2 + b^2 > 2*c^2) → 
  (b^2 + c^2 > 2*a^2) → 
  (c^2 + a^2 > 2*b^2) → 
  (a * b * c = 23012) := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1979_197955


namespace NUMINAMATH_CALUDE_woodys_weekly_allowance_l1979_197975

/-- Woody's weekly allowance problem -/
theorem woodys_weekly_allowance 
  (console_cost : ℕ) 
  (initial_savings : ℕ) 
  (weeks_to_save : ℕ) 
  (h1 : console_cost = 282)
  (h2 : initial_savings = 42)
  (h3 : weeks_to_save = 10) :
  (console_cost - initial_savings) / weeks_to_save = 24 := by
  sorry

end NUMINAMATH_CALUDE_woodys_weekly_allowance_l1979_197975


namespace NUMINAMATH_CALUDE_quadrilateral_sum_l1979_197928

/-- Given a quadrilateral PQRS with vertices P(a, b), Q(a, -b), R(-a, -b), and S(-a, b),
    where a and b are positive integers and a > b, if the area of PQRS is 32,
    then a + b = 5. -/
theorem quadrilateral_sum (a b : ℕ) (ha : a > b) (hb : b > 0)
  (harea : (2 * a) * (2 * b) = 32) : a + b = 5 := by
  sorry

#check quadrilateral_sum

end NUMINAMATH_CALUDE_quadrilateral_sum_l1979_197928


namespace NUMINAMATH_CALUDE_two_lucky_tickets_exist_l1979_197940

/-- A ticket number is a 6-digit integer -/
def TicketNumber := { n : ℕ // n ≥ 100000 ∧ n < 1000000 }

/-- Sum of the first three digits of a ticket number -/
def sumFirstThree (n : TicketNumber) : ℕ := 
  (n.val / 100000) + ((n.val / 10000) % 10) + ((n.val / 1000) % 10)

/-- Sum of the last three digits of a ticket number -/
def sumLastThree (n : TicketNumber) : ℕ := 
  ((n.val / 100) % 10) + ((n.val / 10) % 10) + (n.val % 10)

/-- A ticket is lucky if the sum of its first three digits equals the sum of its last three digits -/
def isLucky (n : TicketNumber) : Prop := sumFirstThree n = sumLastThree n

/-- There exist two lucky tickets among ten consecutive tickets -/
theorem two_lucky_tickets_exist : 
  ∃ (n : TicketNumber) (a b : ℕ), 0 ≤ a ∧ a < b ∧ b ≤ 9 ∧ 
    isLucky ⟨n.val + a, sorry⟩ ∧ isLucky ⟨n.val + b, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_two_lucky_tickets_exist_l1979_197940


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1979_197999

theorem inequality_equivalence (x : ℝ) : (x + 1) / 2 ≥ x / 3 ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1979_197999


namespace NUMINAMATH_CALUDE_max_four_digit_product_of_primes_l1979_197909

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem max_four_digit_product_of_primes :
  ∃ (n x y : ℕ),
    n = x * y * (10 * x + y) ∧
    is_prime x ∧
    is_prime y ∧
    is_prime (10 * x + y) ∧
    x < 5 ∧
    y < 5 ∧
    x ≠ y ∧
    1000 ≤ n ∧
    n < 10000 ∧
    (∀ (m x' y' : ℕ),
      m = x' * y' * (10 * x' + y') →
      is_prime x' →
      is_prime y' →
      is_prime (10 * x' + y') →
      x' < 5 →
      y' < 5 →
      x' ≠ y' →
      1000 ≤ m →
      m < 10000 →
      m ≤ n) ∧
    n = 138 :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_product_of_primes_l1979_197909


namespace NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1979_197936

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Define acute angle condition
def acute_angle (A B C : ℝ × ℝ) : Prop :=
  let BA := (A.1 - B.1, A.2 - B.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  BA.1 * BC.1 + BA.2 * BC.2 > 0

-- Theorem 1: Collinearity condition
theorem collinearity_condition :
  ∀ m : ℝ, collinear OA OB (OC m) ↔ m = 1/2 := sorry

-- Theorem 2: Acute angle condition
theorem acute_angle_condition :
  ∀ m : ℝ, acute_angle OA OB (OC m) ↔ m ∈ Set.Ioo (-3/4 : ℝ) (1/2) ∪ Set.Ioi (1/2) := sorry

end NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1979_197936


namespace NUMINAMATH_CALUDE_club_average_age_l1979_197962

theorem club_average_age (num_females num_males num_children : ℕ)
                         (avg_age_females avg_age_males avg_age_children : ℚ) :
  num_females = 12 →
  num_males = 20 →
  num_children = 8 →
  avg_age_females = 28 →
  avg_age_males = 40 →
  avg_age_children = 10 →
  let total_sum := num_females * avg_age_females + num_males * avg_age_males + num_children * avg_age_children
  let total_people := num_females + num_males + num_children
  (total_sum / total_people : ℚ) = 30.4 := by
  sorry

end NUMINAMATH_CALUDE_club_average_age_l1979_197962


namespace NUMINAMATH_CALUDE_recurrence_closed_form_l1979_197916

def recurrence_sequence (a : ℕ → ℝ) : Prop :=
  (a 0 = 3) ∧ (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 4 * a (n - 1) - 3 * a (n - 2))

theorem recurrence_closed_form (a : ℕ → ℝ) (h : recurrence_sequence a) :
  ∀ n : ℕ, a n = 3^n + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_recurrence_closed_form_l1979_197916


namespace NUMINAMATH_CALUDE_spider_web_paths_l1979_197914

/-- The number of paths from (0, 0) to (m, n) on a grid, moving only right and up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The coordinates of the fly -/
def flyPosition : ℕ × ℕ := (5, 3)

theorem spider_web_paths :
  gridPaths flyPosition.1 flyPosition.2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_l1979_197914


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l1979_197921

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (A B : ℤ), (n - 6) / 15 = A ∧ (n - 5) / 24 = B) := by
sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l1979_197921


namespace NUMINAMATH_CALUDE_micro_lesson_production_properties_l1979_197948

/-- Represents the cost and profit structure of an online education micro-lesson production team -/
structure MicroLessonProduction where
  cost_2A_3B : ℕ  -- Cost of producing 2 A type and 3 B type micro-lessons
  cost_3A_4B : ℕ  -- Cost of producing 3 A type and 4 B type micro-lessons
  price_A : ℕ     -- Selling price of A type micro-lesson
  price_B : ℕ     -- Selling price of B type micro-lesson
  days_per_month : ℕ  -- Number of production days per month

/-- Theorem stating the properties of the micro-lesson production system -/
theorem micro_lesson_production_properties (p : MicroLessonProduction)
  (h1 : p.cost_2A_3B = 2900)
  (h2 : p.cost_3A_4B = 4100)
  (h3 : p.price_A = 1500)
  (h4 : p.price_B = 1000)
  (h5 : p.days_per_month = 22) :
  ∃ (cost_A cost_B : ℕ) (profit_function : ℕ → ℕ) (max_profit max_profit_days : ℕ),
    cost_A = 700 ∧
    cost_B = 500 ∧
    (∀ a : ℕ, 0 < a ∧ a ≤ 66 / 7 → profit_function a = 50 * a + 16500) ∧
    max_profit = 16900 ∧
    max_profit_days = 8 ∧
    (∀ a : ℕ, 0 < a ∧ a ≤ 66 / 7 → profit_function a ≤ max_profit) :=
by sorry


end NUMINAMATH_CALUDE_micro_lesson_production_properties_l1979_197948


namespace NUMINAMATH_CALUDE_probability_of_valid_triangle_l1979_197903

-- Define a regular 15-gon
def regular_15gon : Set (ℝ × ℝ) := sorry

-- Define a function to get all segments in the 15-gon
def all_segments (polygon : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

-- Define a function to check if three segments form a triangle with positive area
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the total number of ways to choose 3 segments
def total_combinations : ℕ := Nat.choose 105 3

-- Define the number of valid triangles
def valid_triangles : ℕ := sorry

-- Theorem statement
theorem probability_of_valid_triangle :
  (valid_triangles : ℚ) / total_combinations = 713 / 780 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_triangle_l1979_197903


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l1979_197981

def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem tree_height_after_two_years 
  (h : tree_height (tree_height h0 2) 2 = 81) : tree_height h0 2 = 9 :=
by
  sorry

#check tree_height_after_two_years

end NUMINAMATH_CALUDE_tree_height_after_two_years_l1979_197981


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l1979_197922

theorem bobby_candy_consumption (morning afternoon evening total : ℕ) : 
  morning = 26 →
  afternoon = 3 * morning →
  evening = afternoon / 2 →
  total = morning + afternoon + evening →
  total = 143 := by
sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l1979_197922


namespace NUMINAMATH_CALUDE_unique_n_value_l1979_197976

theorem unique_n_value : ∃! n : ℕ, 
  50 < n ∧ n < 120 ∧ 
  ∃ k : ℕ, n = 8 * k ∧
  n % 7 = 3 ∧
  n % 9 = 3 ∧
  n = 192 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_value_l1979_197976


namespace NUMINAMATH_CALUDE_evaluate_expression_l1979_197923

theorem evaluate_expression (a x : ℝ) (h : x = a + 9) : x - a + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1979_197923


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l1979_197959

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wall_dimensions : Dimensions := ⟨800, 600, 22.5⟩

/-- The known dimensions of a brick in centimeters -/
def brick_dimensions (height : ℝ) : Dimensions := ⟨80, 11.25, height⟩

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 2000

/-- Theorem stating that the height of each brick is 6 cm -/
theorem brick_height_is_6cm :
  ∃ (h : ℝ), h = 6 ∧ 
  volume wall_dimensions = ↑num_bricks * volume (brick_dimensions h) := by
  sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l1979_197959


namespace NUMINAMATH_CALUDE_exists_xAxis_visitsAllLines_l1979_197919

/-- Represents a line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Configuration of n lines in a plane -/
structure LineConfiguration where
  n : ℕ
  lines : Fin n → Line
  not_parallel : ∀ i j, i ≠ j → (lines i).slope ≠ (lines j).slope
  not_perpendicular : ∀ i j, i ≠ j → (lines i).slope * (lines j).slope ≠ -1
  not_concurrent : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬∃ (x y : ℝ), (y = (lines i).slope * x + (lines i).intercept) ∧
                   (y = (lines j).slope * x + (lines j).intercept) ∧
                   (y = (lines k).slope * x + (lines k).intercept)

/-- A point visits all lines if it intersects with each line -/
def visitsAllLines (cfg : LineConfiguration) (xAxis : Line) : Prop :=
  ∀ i, ∃ x, xAxis.slope * x + xAxis.intercept = (cfg.lines i).slope * x + (cfg.lines i).intercept

/-- Main theorem: There exists a line that can be chosen as x-axis to visit all lines -/
theorem exists_xAxis_visitsAllLines (cfg : LineConfiguration) :
  ∃ xAxis, visitsAllLines cfg xAxis := by
  sorry

end NUMINAMATH_CALUDE_exists_xAxis_visitsAllLines_l1979_197919


namespace NUMINAMATH_CALUDE_triangle_formation_l1979_197969

/-- Triangle inequality theorem check for three sides --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 2 3 4 ∧
  ¬can_form_triangle 5 1 3 ∧
  ¬can_form_triangle 2 4 2 ∧
  ¬can_form_triangle 3 3 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1979_197969


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l1979_197971

theorem forty_percent_of_number (n : ℝ) : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 15 → (40/100 : ℝ) * n = 180 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l1979_197971
