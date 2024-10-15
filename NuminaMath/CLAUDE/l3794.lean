import Mathlib

namespace NUMINAMATH_CALUDE_b_share_is_1500_l3794_379452

/-- Calculates the share of the second child (B) when distributing money among three children in a given ratio -/
def calculate_b_share (total_money : ℚ) (ratio_a ratio_b ratio_c : ℕ) : ℚ :=
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := total_money / total_parts
  ratio_b * part_value

/-- Theorem stating that given $4500 distributed in the ratio 2:3:4, B's share is $1500 -/
theorem b_share_is_1500 :
  calculate_b_share 4500 2 3 4 = 1500 := by
  sorry

#eval calculate_b_share 4500 2 3 4

end NUMINAMATH_CALUDE_b_share_is_1500_l3794_379452


namespace NUMINAMATH_CALUDE_units_digit_27_45_l3794_379450

theorem units_digit_27_45 : (27 ^ 45) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_45_l3794_379450


namespace NUMINAMATH_CALUDE_difference_of_squares_l3794_379489

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3794_379489


namespace NUMINAMATH_CALUDE_pebble_collection_sum_l3794_379426

/-- The sum of an arithmetic sequence with first term 2, common difference 3, and 15 terms -/
def arithmetic_sum : ℕ → ℕ
| n => n * (4 + 3 * (n - 1)) / 2

/-- Theorem stating that the sum of the first 15 terms of the arithmetic sequence is 345 -/
theorem pebble_collection_sum : arithmetic_sum 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_sum_l3794_379426


namespace NUMINAMATH_CALUDE_one_true_proposition_l3794_379498

-- Define the basic concepts
def Point : Type := ℝ × ℝ
def Triangle (A B C : Point) : Prop := True  -- Simplified definition
def Isosceles (A B C : Point) : Prop := True  -- Simplified definition

-- Define the original proposition
def original_prop (A B C : Point) : Prop :=
  A.1 = B.1 ∧ A.2 = B.2 → Isosceles A B C

-- Define the converse proposition
def converse_prop (A B C : Point) : Prop :=
  Isosceles A B C → A.1 = B.1 ∧ A.2 = B.2

-- Define the inverse proposition
def inverse_prop (A B C : Point) : Prop :=
  ¬(A.1 = B.1 ∧ A.2 = B.2) → ¬(Isosceles A B C)

-- Define the contrapositive proposition
def contrapositive_prop (A B C : Point) : Prop :=
  ¬(Isosceles A B C) → ¬(A.1 = B.1 ∧ A.2 = B.2)

-- The theorem to be proved
theorem one_true_proposition (A B C : Point) :
  (original_prop A B C) ∧
  (¬(converse_prop A B C) ∨ ¬(inverse_prop A B C)) ∧
  (contrapositive_prop A B C) :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l3794_379498


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l3794_379467

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Theorem: In a trapezoid with given properties, the length of the other side is 10 -/
theorem trapezoid_side_length (t : Trapezoid) 
  (h_area : t.area = 164)
  (h_altitude : t.altitude = 8)
  (h_base1 : t.base1 = 10)
  (h_base2 : t.base2 = 17) :
  t.base2 - t.base1 = 10 := by
  sorry

#check trapezoid_side_length

end NUMINAMATH_CALUDE_trapezoid_side_length_l3794_379467


namespace NUMINAMATH_CALUDE_negative_two_cubed_l3794_379497

theorem negative_two_cubed : (-2 : ℤ)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_l3794_379497


namespace NUMINAMATH_CALUDE_second_year_sample_size_l3794_379402

/-- Represents the distribution of students across four years -/
structure StudentDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the total number of students -/
def total_students (d : StudentDistribution) : ℕ :=
  d.first + d.second + d.third + d.fourth

/-- Calculates the number of students to sample from a specific year -/
def sample_size_for_year (total_population : ℕ) (year_population : ℕ) (sample_size : ℕ) : ℕ :=
  (year_population * sample_size) / total_population

theorem second_year_sample_size 
  (total_population : ℕ) 
  (distribution : StudentDistribution) 
  (sample_size : ℕ) :
  total_population = 5000 →
  distribution = { first := 5, second := 4, third := 3, fourth := 1 } →
  sample_size = 260 →
  sample_size_for_year total_population distribution.second sample_size = 80 := by
  sorry

#check second_year_sample_size

end NUMINAMATH_CALUDE_second_year_sample_size_l3794_379402


namespace NUMINAMATH_CALUDE_tv_price_increase_l3794_379443

theorem tv_price_increase (initial_price : ℝ) (first_increase : ℝ) : 
  first_increase > 0 →
  (initial_price * (1 + first_increase / 100) * 1.4 = initial_price * 1.82) →
  first_increase = 30 := by
sorry

end NUMINAMATH_CALUDE_tv_price_increase_l3794_379443


namespace NUMINAMATH_CALUDE_line_ratio_sum_l3794_379496

/-- Given two lines l₁ and l₂, and points P₁ and P₂ on these lines respectively,
    prove that the sum of certain ratios of the line coefficients equals 3. -/
theorem line_ratio_sum (a₁ b₁ c₁ a₂ b₂ c₂ x₁ y₁ x₂ y₂ : ℝ) : 
  a₁ * x₁ + b₁ * y₁ = c₁ →
  a₂ * x₂ + b₂ * y₂ = c₂ →
  a₁ + b₁ = c₁ →
  a₂ + b₂ = 2 * c₂ →
  (∀ x₁ y₁ x₂ y₂, (x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ 1/2) →
  c₁ / a₁ + a₂ / c₂ = 3 := by
sorry

end NUMINAMATH_CALUDE_line_ratio_sum_l3794_379496


namespace NUMINAMATH_CALUDE_arianna_sleep_hours_l3794_379466

/-- Represents the number of hours in a day. -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Arianna spends at work. -/
def work_hours : ℕ := 6

/-- Represents the number of hours Arianna spends in class. -/
def class_hours : ℕ := 3

/-- Represents the number of hours Arianna spends at the gym. -/
def gym_hours : ℕ := 2

/-- Represents the number of hours Arianna spends on other daily chores. -/
def chore_hours : ℕ := 5

/-- Represents the number of hours Arianna sleeps. -/
def sleep_hours : ℕ := hours_in_day - (work_hours + class_hours + gym_hours + chore_hours)

/-- Theorem stating that Arianna sleeps for 8 hours a day. -/
theorem arianna_sleep_hours : sleep_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_arianna_sleep_hours_l3794_379466


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3794_379494

theorem complex_equation_solution :
  ∃ x : ℂ, (5 : ℂ) - 3 * Complex.I * x = (7 : ℂ) - Complex.I * x ∧ x = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3794_379494


namespace NUMINAMATH_CALUDE_arbitrarily_large_power_l3794_379492

theorem arbitrarily_large_power (a : ℝ) (h : a > 1) :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, a^x > y :=
by sorry

end NUMINAMATH_CALUDE_arbitrarily_large_power_l3794_379492


namespace NUMINAMATH_CALUDE_square_remainder_l3794_379476

theorem square_remainder (n x : ℤ) (h : n % x = 3) : (n^2) % x = 9 % x := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l3794_379476


namespace NUMINAMATH_CALUDE_problem_statement_l3794_379410

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_statement (x x₁ x₂ : ℝ) :
  (∀ x > 0, f x ≥ 1) ∧
  (∀ x > 1, f x < g x) ∧
  (x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g x₁ = g x₂ → x₁ * x₂ < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3794_379410


namespace NUMINAMATH_CALUDE_abs_x_y_sum_l3794_379407

theorem abs_x_y_sum (x y : ℝ) : 
  (|x| = 7 ∧ |y| = 9 ∧ |x + y| = -(x + y)) → (x - y = 16 ∨ x - y = -16) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_y_sum_l3794_379407


namespace NUMINAMATH_CALUDE_bridge_length_l3794_379409

theorem bridge_length 
  (left_bank : ℚ) 
  (right_bank : ℚ) 
  (river_width : ℚ) :
  left_bank = 1/4 →
  right_bank = 1/3 →
  river_width = 120 →
  (1 - left_bank - right_bank) * (288 : ℚ) = river_width :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3794_379409


namespace NUMINAMATH_CALUDE_probability_closer_to_point1_l3794_379475

/-- The rectangular region from which point P is selected -/
def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The area of the rectangular region -/
def RectangleArea : ℝ := 6

/-- The point (1,1) -/
def Point1 : ℝ × ℝ := (1, 1)

/-- The point (4,1) -/
def Point2 : ℝ × ℝ := (4, 1)

/-- The region where points are closer to (1,1) than to (4,1) -/
def CloserRegion : Set (ℝ × ℝ) :=
  {p ∈ Rectangle | dist p Point1 < dist p Point2}

/-- The area of the region closer to (1,1) -/
def CloserRegionArea : ℝ := 5

/-- The probability of a randomly selected point being closer to (1,1) than to (4,1) -/
theorem probability_closer_to_point1 :
  CloserRegionArea / RectangleArea = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_closer_to_point1_l3794_379475


namespace NUMINAMATH_CALUDE_total_cakes_is_fifteen_l3794_379455

/-- The number of cakes served during lunch -/
def lunch_cakes : ℕ := 6

/-- The number of cakes served during dinner -/
def dinner_cakes : ℕ := 9

/-- The total number of cakes served today -/
def total_cakes : ℕ := lunch_cakes + dinner_cakes

/-- Proof that the total number of cakes served today is 15 -/
theorem total_cakes_is_fifteen : total_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_is_fifteen_l3794_379455


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3794_379458

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x^3 / (x - 2) > 0)) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3794_379458


namespace NUMINAMATH_CALUDE_square_area_error_l3794_379486

theorem square_area_error (a : ℝ) (h : a > 0) :
  let measured_side := a * (1 + 0.08)
  let actual_area := a^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1664 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3794_379486


namespace NUMINAMATH_CALUDE_second_store_cars_l3794_379471

-- Define the number of stores
def num_stores : ℕ := 5

-- Define the car counts for known stores
def first_store : ℕ := 30
def third_store : ℕ := 14
def fourth_store : ℕ := 21
def fifth_store : ℕ := 25

-- Define the mean
def mean : ℚ := 20.8

-- Define the theorem
theorem second_store_cars :
  ∃ (second_store : ℕ),
    (first_store + second_store + third_store + fourth_store + fifth_store) / num_stores = mean ∧
    second_store = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_store_cars_l3794_379471


namespace NUMINAMATH_CALUDE_wheel_circumference_proof_l3794_379478

/-- The circumference of the front wheel -/
def front_wheel_circumference : ℝ := 24

/-- The circumference of the rear wheel -/
def rear_wheel_circumference : ℝ := 18

/-- The distance traveled -/
def distance : ℝ := 360

theorem wheel_circumference_proof :
  (distance / front_wheel_circumference = distance / rear_wheel_circumference + 4) ∧
  (distance / (front_wheel_circumference - 3) = distance / (rear_wheel_circumference - 3) + 6) →
  (front_wheel_circumference = 24 ∧ rear_wheel_circumference = 18) :=
by sorry

end NUMINAMATH_CALUDE_wheel_circumference_proof_l3794_379478


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3794_379469

/-- The coordinates of the foci of an ellipse given by the equation mx^2 + ny^2 + mn = 0,
    where m < n < 0 -/
theorem ellipse_foci_coordinates (m n : ℝ) (h1 : m < n) (h2 : n < 0) :
  let equation := fun (x y : ℝ) => m * x^2 + n * y^2 + m * n
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) ↔ 
      (x, y) ∈ {p : ℝ × ℝ | p.1^2 / (-n) + p.2^2 / (-m) = 1 ∧ p.1^2 + p.2^2 > 1}) ∧
    c^2 = n - m :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3794_379469


namespace NUMINAMATH_CALUDE_uphill_divisible_by_nine_count_l3794_379433

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def UphillInteger (n : ℕ) : Prop := sorry

/-- Check if a natural number ends with 6 -/
def EndsWithSix (n : ℕ) : Prop := sorry

/-- Count the number of uphill integers ending in 6 that are divisible by 9 -/
def CountUphillDivisibleBySix : ℕ := sorry

theorem uphill_divisible_by_nine_count : CountUphillDivisibleBySix = 2 := by sorry

end NUMINAMATH_CALUDE_uphill_divisible_by_nine_count_l3794_379433


namespace NUMINAMATH_CALUDE_min_value_expression_l3794_379480

theorem min_value_expression (x y z : ℝ) (h : x - 2*y + 2*z = 5) :
  ∃ (min : ℝ), min = 36 ∧ ∀ (x' y' z' : ℝ), x' - 2*y' + 2*z' = 5 → 
    (x' + 5)^2 + (y' - 1)^2 + (z' + 3)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3794_379480


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3794_379474

theorem exponential_equation_solution : ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (27 : ℝ)^4 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3794_379474


namespace NUMINAMATH_CALUDE_expand_expression_l3794_379463

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3794_379463


namespace NUMINAMATH_CALUDE_f_strictly_increasing_on_interval_l3794_379446

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- State the theorem
theorem f_strictly_increasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_on_interval_l3794_379446


namespace NUMINAMATH_CALUDE_video_recorder_markup_l3794_379493

theorem video_recorder_markup (wholesale_cost : ℝ) (employee_discount : ℝ) (employee_paid : ℝ) :
  wholesale_cost = 200 →
  employee_discount = 0.30 →
  employee_paid = 168 →
  ∃ (markup : ℝ), 
    employee_paid = (1 - employee_discount) * (wholesale_cost * (1 + markup)) ∧
    markup = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_markup_l3794_379493


namespace NUMINAMATH_CALUDE_curve_translation_l3794_379437

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.sin x - 2 * y + 3 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (1 + y) * Real.cos x - 2 * y + 1 = 0

-- Theorem statement
theorem curve_translation :
  ∀ (x y : ℝ),
  original_curve (x + π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l3794_379437


namespace NUMINAMATH_CALUDE_artwork_transaction_l3794_379490

/-- Converts a number from base s to base 10 -/
def to_base_10 (digits : List Nat) (s : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * s^i) 0

theorem artwork_transaction (s : Nat) : 
  s > 1 →
  to_base_10 [0, 3, 5] s + to_base_10 [0, 3, 2, 1] s = to_base_10 [0, 0, 0, 2] s →
  s = 8 := by
sorry

end NUMINAMATH_CALUDE_artwork_transaction_l3794_379490


namespace NUMINAMATH_CALUDE_factorial_15_base_18_trailing_zeros_l3794_379436

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def countTrailingZerosBase18 (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

theorem factorial_15_base_18_trailing_zeros :
  countTrailingZerosBase18 (factorial 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_15_base_18_trailing_zeros_l3794_379436


namespace NUMINAMATH_CALUDE_madeline_leisure_hours_l3794_379456

def total_hours_in_week : ℕ := 24 * 7

def class_hours : ℕ := 18
def homework_hours : ℕ := 4 * 7
def extracurricular_hours : ℕ := 3 * 3
def tutoring_hours : ℕ := 1 * 2
def work_hours : ℕ := 5 + 4 + 4 + 7
def sleep_hours : ℕ := 8 * 7

def total_scheduled_hours : ℕ := 
  class_hours + homework_hours + extracurricular_hours + tutoring_hours + work_hours + sleep_hours

theorem madeline_leisure_hours : 
  total_hours_in_week - total_scheduled_hours = 35 := by sorry

end NUMINAMATH_CALUDE_madeline_leisure_hours_l3794_379456


namespace NUMINAMATH_CALUDE_sets_equality_and_inclusion_l3794_379425

def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x - 6*m^2 ≤ 0}

theorem sets_equality_and_inclusion (m : ℝ) (h : m > 0) :
  A = {x : ℝ | -2 ≤ x ∧ x ≤ 6} ∧
  B m = {x : ℝ | -2*m ≤ x ∧ x ≤ 3*m} ∧
  (A ⊆ B m ↔ m ≥ 2) ∧
  (B m ⊆ A ↔ 0 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sets_equality_and_inclusion_l3794_379425


namespace NUMINAMATH_CALUDE_john_plays_two_periods_l3794_379406

def points_per_4_minutes : ℕ := 2 * 2 + 1 * 3
def minutes_per_period : ℕ := 12
def total_points : ℕ := 42

theorem john_plays_two_periods :
  (total_points / (points_per_4_minutes * (minutes_per_period / 4))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_plays_two_periods_l3794_379406


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3794_379444

/-- The polynomial with unknown coefficients a and b -/
def P (x a b : ℝ) : ℝ := 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12

/-- The given factor of the polynomial -/
def F (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2

/-- Theorem stating that the polynomial P has the factor F when a = -26.5 and b = -40 -/
theorem polynomial_factorization (x : ℝ) : 
  ∃ (Q : ℝ → ℝ), P x (-26.5) (-40) = F x * Q x := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3794_379444


namespace NUMINAMATH_CALUDE_certain_number_problem_l3794_379499

theorem certain_number_problem (x : ℝ) : 
  3 + x + 333 + 33.3 = 399.6 → x = 30.3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3794_379499


namespace NUMINAMATH_CALUDE_leadership_structure_count_15_l3794_379482

/-- The number of ways to select a leadership structure from a group of people. -/
def leadershipStructureCount (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * (n - 3).choose 3 * (n - 6).choose 3

/-- Theorem stating that the number of ways to select a leadership structure
    from 15 people is 2,717,880. -/
theorem leadership_structure_count_15 :
  leadershipStructureCount 15 = 2717880 := by
  sorry

end NUMINAMATH_CALUDE_leadership_structure_count_15_l3794_379482


namespace NUMINAMATH_CALUDE_area_quadrilateral_OBEC_l3794_379488

/-- A line with slope -3 passing through (3,6) -/
def line1 (x y : ℝ) : Prop := y - 6 = -3 * (x - 3)

/-- The x-coordinate of point A where line1 intersects the x-axis -/
def point_A : ℝ := 5

/-- The y-coordinate of point B where line1 intersects the y-axis -/
def point_B : ℝ := 15

/-- A line passing through points (6,0) and (3,6) -/
def line2 (x y : ℝ) : Prop := y = 2 * x - 12

/-- The area of quadrilateral OBEC -/
def area_OBEC : ℝ := 72

theorem area_quadrilateral_OBEC :
  line1 3 6 →
  line1 point_A 0 →
  line1 0 point_B →
  line2 3 6 →
  line2 6 0 →
  area_OBEC = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_quadrilateral_OBEC_l3794_379488


namespace NUMINAMATH_CALUDE_polynomial_identity_l3794_379464

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3794_379464


namespace NUMINAMATH_CALUDE_fifth_month_sales_l3794_379429

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_3 : ℕ := 6855
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 7991
def average_sales : ℕ := 7000
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sales ∧
    sales_5 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l3794_379429


namespace NUMINAMATH_CALUDE_leap_year_hours_l3794_379439

/-- The number of days in a leap year -/
def days_in_leap_year : ℕ := 366

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

/-- The number of hours in a leap year -/
def hours_in_leap_year : ℕ := days_in_leap_year * hours_in_day

theorem leap_year_hours :
  hours_in_leap_year = 8784 :=
by sorry

end NUMINAMATH_CALUDE_leap_year_hours_l3794_379439


namespace NUMINAMATH_CALUDE_symmetry_axis_of_sine_function_l3794_379470

/-- Given that cos(2π/3 - φ) = cosφ, prove that x = 5π/6 is a symmetry axis of f(x) = sin(x - φ) -/
theorem symmetry_axis_of_sine_function (φ : ℝ) 
  (h : Real.cos (2 * Real.pi / 3 - φ) = Real.cos φ) :
  ∀ x : ℝ, Real.sin (x - φ) = Real.sin ((5 * Real.pi / 3 - x) - φ) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_sine_function_l3794_379470


namespace NUMINAMATH_CALUDE_percentage_increase_l3794_379479

theorem percentage_increase (original_earnings new_earnings : ℝ) :
  original_earnings = 60 →
  new_earnings = 68 →
  (new_earnings - original_earnings) / original_earnings * 100 = (68 - 60) / 60 * 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3794_379479


namespace NUMINAMATH_CALUDE_water_left_in_bathtub_l3794_379441

/-- The amount of water left in a bathtub after a faucet drips and water evaporates --/
theorem water_left_in_bathtub
  (drip_rate : ℝ)
  (evap_rate : ℝ)
  (time : ℝ)
  (dumped : ℝ)
  (h1 : drip_rate = 40)
  (h2 : evap_rate = 200)
  (h3 : time = 9)
  (h4 : dumped = 12000) :
  drip_rate * time * 60 - evap_rate * time - dumped = 7800 :=
by sorry

end NUMINAMATH_CALUDE_water_left_in_bathtub_l3794_379441


namespace NUMINAMATH_CALUDE_barbaras_score_l3794_379422

theorem barbaras_score (total_students : ℕ) (students_without_barbara : ℕ) 
  (avg_without_barbara : ℚ) (avg_with_barbara : ℚ) :
  total_students = 20 →
  students_without_barbara = 19 →
  avg_without_barbara = 78 →
  avg_with_barbara = 79 →
  (total_students * avg_with_barbara - students_without_barbara * avg_without_barbara : ℚ) = 98 := by
  sorry

#check barbaras_score

end NUMINAMATH_CALUDE_barbaras_score_l3794_379422


namespace NUMINAMATH_CALUDE_mode_of_throws_l3794_379483

def throw_results : List Float := [7.6, 8.5, 8.6, 8.5, 9.1, 8.5, 8.4, 8.6, 9.2, 7.3]

def mode (l : List Float) : Float :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_throws :
  mode throw_results = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_throws_l3794_379483


namespace NUMINAMATH_CALUDE_logarithm_problem_l3794_379427

noncomputable def a : ℝ := Real.log 55 / Real.log 50
noncomputable def b : ℝ := Real.log 20 / Real.log 55

theorem logarithm_problem (a b : ℝ) (h1 : a = Real.log 55 / Real.log 50) (h2 : b = Real.log 20 / Real.log 55) :
  Real.log (2662 * Real.sqrt 10) / Real.log 250 = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_problem_l3794_379427


namespace NUMINAMATH_CALUDE_sum_of_squares_l3794_379454

variables {x y z w a b c d : ℝ}

theorem sum_of_squares (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x * w = d)
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : d ≠ 0) :
  x^2 + y^2 + z^2 + w^2 = (a * b + b * d + d * a)^2 / (a * b * d) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3794_379454


namespace NUMINAMATH_CALUDE_price_change_theorem_l3794_379408

theorem price_change_theorem (initial_price : ℝ) (initial_price_positive : 0 < initial_price) :
  let price_after_increase := initial_price * (1 + 0.35)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  (final_price - initial_price) / initial_price = 0.03275 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l3794_379408


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l3794_379461

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b) / a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l3794_379461


namespace NUMINAMATH_CALUDE_inequality_proof_l3794_379405

theorem inequality_proof (y : ℝ) (h : y > 0) :
  2 * y ≥ 3 - 1 / y^2 ∧ (2 * y = 3 - 1 / y^2 ↔ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3794_379405


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3794_379420

theorem complex_number_simplification :
  (2 - 5 * Complex.I) - (-3 + 7 * Complex.I) - 4 * (-1 + 2 * Complex.I) = 1 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3794_379420


namespace NUMINAMATH_CALUDE_compute_expression_l3794_379419

theorem compute_expression : 3 * 3^4 - 4^55 / 4^54 = 239 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l3794_379419


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3794_379468

theorem modulus_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3794_379468


namespace NUMINAMATH_CALUDE_prob_over_60_and_hypertension_is_9_percent_l3794_379430

/-- The probability of a person being over 60 years old in the region -/
def prob_over_60 : ℝ := 0.2

/-- The probability of a person having hypertension given they are over 60 -/
def prob_hypertension_given_over_60 : ℝ := 0.45

/-- The probability of a person being both over 60 and having hypertension -/
def prob_over_60_and_hypertension : ℝ := prob_over_60 * prob_hypertension_given_over_60

theorem prob_over_60_and_hypertension_is_9_percent :
  prob_over_60_and_hypertension = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_prob_over_60_and_hypertension_is_9_percent_l3794_379430


namespace NUMINAMATH_CALUDE_odd_numbers_product_equality_l3794_379449

theorem odd_numbers_product_equality (a b c d : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  a < b → b < c → c < d →
  a * d = b * c →
  ∃ k l : ℕ, a + d = 2^k ∧ b + c = 2^l →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_numbers_product_equality_l3794_379449


namespace NUMINAMATH_CALUDE_exponent_division_l3794_379417

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^15 / x^3 = x^12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3794_379417


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3794_379428

/-- A symmetric trapezoid with an inscribed and circumscribed circle -/
structure SymmetricTrapezoid where
  -- The lengths of the parallel sides
  a : ℝ
  b : ℝ
  -- The radius of the circumscribed circle
  R : ℝ
  -- The radius of the inscribed circle
  ρ : ℝ
  -- Conditions
  h_symmetric : a ≥ b
  h_R : R = 1
  h_inscribed : ρ > 0
  h_center_bisects : ∃ (K : ℝ × ℝ), K.1^2 + K.2^2 = (R/2)^2

/-- The radius of the inscribed circle in the symmetric trapezoid -/
theorem inscribed_circle_radius (T : SymmetricTrapezoid) : T.ρ = Real.sqrt (9/40) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3794_379428


namespace NUMINAMATH_CALUDE_first_number_is_55_l3794_379473

def number_list : List ℕ := [55, 57, 58, 59, 62, 62, 63, 65, 65]

theorem first_number_is_55 (average_is_60 : (number_list.sum / number_list.length : ℚ) = 60) :
  number_list.head? = some 55 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_55_l3794_379473


namespace NUMINAMATH_CALUDE_geometric_sequence_implies_c_equals_six_l3794_379424

/-- A function f(x) = x^2 + x + c where f(1), f(2), and f(3) form a geometric sequence. -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + x + c

/-- The theorem stating that if f(1), f(2), and f(3) form a geometric sequence, then c = 6. -/
theorem geometric_sequence_implies_c_equals_six (c : ℝ) :
  (∃ r : ℝ, f c 2 = f c 1 * r ∧ f c 3 = f c 2 * r) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_implies_c_equals_six_l3794_379424


namespace NUMINAMATH_CALUDE_x_equals_negative_x_is_valid_l3794_379453

/-- An assignment statement is valid if it assigns a value to a variable -/
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (val : String), stmt = var ++ " = " ++ val

/-- The statement "x = -x" -/
def statement : String := "x = -x"

/-- Theorem: The statement "x = -x" is a valid assignment statement -/
theorem x_equals_negative_x_is_valid : is_valid_assignment statement := by
  sorry

end NUMINAMATH_CALUDE_x_equals_negative_x_is_valid_l3794_379453


namespace NUMINAMATH_CALUDE_divisibility_of_concatenated_numbers_l3794_379447

theorem divisibility_of_concatenated_numbers (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 →
  100 ≤ b ∧ b < 1000 →
  37 ∣ (a + b) →
  37 ∣ (1000 * a + b) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_concatenated_numbers_l3794_379447


namespace NUMINAMATH_CALUDE_circle_properties_l3794_379438

-- Define the circle P
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def intersects_x_axis (P : Circle) : Prop :=
  P.radius^2 - P.center.2^2 = 2

def intersects_y_axis (P : Circle) : Prop :=
  P.radius^2 - P.center.1^2 = 3

def distance_to_y_eq_x (P : Circle) : Prop :=
  |P.center.2 - P.center.1| = 1

-- Define the theorem
theorem circle_properties (P : Circle) 
  (hx : intersects_x_axis P) 
  (hy : intersects_y_axis P) 
  (hd : distance_to_y_eq_x P) : 
  (∃ a b : ℝ, P.center = (a, b) ∧ b^2 - a^2 = 1) ∧ 
  ((P.center = (0, 1) ∧ P.radius = Real.sqrt 3) ∨ 
   (P.center = (0, -1) ∧ P.radius = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3794_379438


namespace NUMINAMATH_CALUDE_problem_solution_l3794_379448

theorem problem_solution : 
  (0.027 ^ (-1/3 : ℝ)) + (16 ^ 3) ^ (1/4 : ℝ) - 3⁻¹ + ((2 : ℝ).sqrt - 1) ^ (0 : ℝ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3794_379448


namespace NUMINAMATH_CALUDE_arithmetic_expressions_l3794_379403

theorem arithmetic_expressions :
  let expr1 := (3.6 - 0.8) * (1.8 + 2.05)
  let expr2 := (34.28 / 2) - (16.2 / 4)
  (expr1 = (3.6 - 0.8) * (1.8 + 2.05)) ∧
  (expr2 = (34.28 / 2) - (16.2 / 4)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_l3794_379403


namespace NUMINAMATH_CALUDE_max_profit_price_l3794_379477

/-- Represents the profit function for a store selling items -/
def profit_function (purchase_price : ℝ) (base_price : ℝ) (base_quantity : ℝ) (price_sensitivity : ℝ) (x : ℝ) : ℝ :=
  (x - purchase_price) * (base_quantity - price_sensitivity * (x - base_price))

theorem max_profit_price (purchase_price : ℝ) (base_price : ℝ) (base_quantity : ℝ) (price_sensitivity : ℝ) 
    (h1 : purchase_price = 20)
    (h2 : base_price = 30)
    (h3 : base_quantity = 400)
    (h4 : price_sensitivity = 20) : 
  ∃ (max_price : ℝ), max_price = 35 ∧ 
    ∀ (x : ℝ), profit_function purchase_price base_price base_quantity price_sensitivity x ≤ 
               profit_function purchase_price base_price base_quantity price_sensitivity max_price :=
by sorry

#check max_profit_price

end NUMINAMATH_CALUDE_max_profit_price_l3794_379477


namespace NUMINAMATH_CALUDE_min_n_for_cuboid_sum_l3794_379460

theorem min_n_for_cuboid_sum (n : ℕ) : (∀ m : ℕ, m > 0 ∧ 128 * m > 2011 → n ≤ m) ∧ n > 0 ∧ 128 * n > 2011 ↔ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_cuboid_sum_l3794_379460


namespace NUMINAMATH_CALUDE_sphere_speeds_solution_l3794_379472

/-- Represents the speeds of two spheres moving towards the vertex of a right angle --/
structure SphereSpeeds where
  small : ℝ
  large : ℝ

/-- The problem setup and conditions --/
def sphereProblem (s : SphereSpeeds) : Prop :=
  let r₁ := 2 -- radius of smaller sphere
  let r₂ := 3 -- radius of larger sphere
  let d₁ := 6 -- initial distance of smaller sphere from vertex
  let d₂ := 16 -- initial distance of larger sphere from vertex
  let t₁ := 1 -- time after which distance between centers is measured
  let t₂ := 3 -- time at which spheres collide
  -- Initial positions
  (d₁ - s.small * t₁) ^ 2 + (d₂ - s.large * t₁) ^ 2 = 13 ^ 2 ∧
  -- Collision positions
  (d₁ - s.small * t₂) ^ 2 + (d₂ - s.large * t₂) ^ 2 = (r₁ + r₂) ^ 2

/-- The theorem stating the solution to the sphere problem --/
theorem sphere_speeds_solution :
  ∃ s : SphereSpeeds, sphereProblem s ∧ s.small = 1 ∧ s.large = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_speeds_solution_l3794_379472


namespace NUMINAMATH_CALUDE_modulus_of_z_l3794_379495

theorem modulus_of_z (z : ℂ) (h : (z + Complex.I) * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3794_379495


namespace NUMINAMATH_CALUDE_eddie_study_games_l3794_379431

/-- Calculates the maximum number of games that can be played in a study block -/
def max_games (study_block_minutes : ℕ) (homework_minutes : ℕ) (game_duration : ℕ) : ℕ :=
  (study_block_minutes - homework_minutes) / game_duration

/-- Theorem stating that given the specific conditions, the maximum number of games is 7 -/
theorem eddie_study_games :
  max_games 60 25 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eddie_study_games_l3794_379431


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3794_379413

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = 2 + 8*I) : Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3794_379413


namespace NUMINAMATH_CALUDE_converse_of_zero_product_is_false_l3794_379400

theorem converse_of_zero_product_is_false :
  ¬ (∀ (a b : ℝ), a * b = 0 → a = 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_zero_product_is_false_l3794_379400


namespace NUMINAMATH_CALUDE_parallel_vectors_xy_value_l3794_379487

/-- Given two parallel vectors a and b in R³, prove that xy = -1/4 --/
theorem parallel_vectors_xy_value (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2*x, 1, 3)
  let b : ℝ × ℝ × ℝ := (1, -2*y, 9)
  (∃ (k : ℝ), a = k • b) → x * y = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_xy_value_l3794_379487


namespace NUMINAMATH_CALUDE_sin_squared_plus_cos_squared_equals_one_l3794_379457

-- Define a point on a unit circle
def PointOnUnitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the relationship between x, y, and θ on the unit circle
def UnitCirclePoint (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.cos θ ∧ y = Real.sin θ

-- Theorem statement
theorem sin_squared_plus_cos_squared_equals_one (θ : ℝ) :
  ∃ x y : ℝ, UnitCirclePoint θ x y → (Real.sin θ)^2 + (Real.cos θ)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_plus_cos_squared_equals_one_l3794_379457


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l3794_379432

theorem unique_solution_3x_4y_5z : 
  ∀ x y z : ℕ+, 
    (3 : ℕ)^(x : ℕ) + (4 : ℕ)^(y : ℕ) = (5 : ℕ)^(z : ℕ) → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

#check unique_solution_3x_4y_5z

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l3794_379432


namespace NUMINAMATH_CALUDE_complex_equation_solution_count_l3794_379481

theorem complex_equation_solution_count : 
  ∃! (c : ℝ), Complex.abs (2/3 - c * Complex.I) = 5/6 ∧ Complex.im (3 + c * Complex.I) > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_count_l3794_379481


namespace NUMINAMATH_CALUDE_smile_area_l3794_379415

/-- The area of the "smile" region formed by two sectors and a semicircle -/
theorem smile_area : 
  ∀ (r₁ r₂ : ℝ) (θ : ℝ),
  r₁ = 3 → r₂ = 2 → θ = π/4 →
  2 * (1/2 * r₁^2 * θ) + 1/2 * π * r₂^2 = 17*π/4 :=
by sorry

end NUMINAMATH_CALUDE_smile_area_l3794_379415


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3794_379445

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 3*x + 2) * (x^2 + 7*x + 12) + (x^2 + 5*x - 6) = (x^2 + 5*x + 2) * (x^2 + 5*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3794_379445


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3794_379434

theorem consecutive_integers_product_sum :
  ∀ x y z : ℤ,
  (y = x + 1) →
  (z = y + 1) →
  (x * y * z = 336) →
  (x + y + z = 21) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3794_379434


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l3794_379484

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (Dishonest : U → Prop)

-- Define the conditions
variable (some_students_dishonest : ∃ x, Student x ∧ Dishonest x)
variable (all_club_members_honest : ∀ x, ClubMember x → ¬Dishonest x)

-- Theorem to prove
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_club_members_l3794_379484


namespace NUMINAMATH_CALUDE_jims_bulb_purchase_l3794_379414

theorem jims_bulb_purchase : 
  let lamp_cost : ℕ := 7
  let bulb_cost : ℕ := lamp_cost - 4
  let num_lamps : ℕ := 2
  let total_cost : ℕ := 32
  let bulbs_cost : ℕ := total_cost - (num_lamps * lamp_cost)
  ∃ (num_bulbs : ℕ), num_bulbs * bulb_cost = bulbs_cost ∧ num_bulbs = 6
  := by sorry

end NUMINAMATH_CALUDE_jims_bulb_purchase_l3794_379414


namespace NUMINAMATH_CALUDE_fraction_simplification_l3794_379465

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3794_379465


namespace NUMINAMATH_CALUDE_exists_k_undecided_tournament_l3794_379421

/-- A tournament is a complete directed graph where each edge represents a match outcome. -/
def Tournament (n : ℕ) := Fin n → Fin n → Bool

/-- A tournament is k-undecided if for every set of k players, there exists a player who has defeated all of them. -/
def IsKUndecided (k : ℕ) (n : ℕ) (t : Tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k →
    ∃ (p : Fin n), p ∉ A ∧ ∀ (a : Fin n), a ∈ A → t p a = true

/-- For every positive integer k, there exists a k-undecided tournament with more than k players. -/
theorem exists_k_undecided_tournament (k : ℕ+) :
  ∃ (n : ℕ) (t : Tournament n), n > k ∧ IsKUndecided k n t :=
sorry

end NUMINAMATH_CALUDE_exists_k_undecided_tournament_l3794_379421


namespace NUMINAMATH_CALUDE_jim_diving_hours_l3794_379435

/-- The number of gold coins Jim finds per hour -/
def gold_coins_per_hour : ℕ := 25

/-- The number of gold coins in the treasure chest -/
def chest_coins : ℕ := 100

/-- The number of smaller bags Jim found -/
def num_smaller_bags : ℕ := 2

/-- The number of gold coins in each smaller bag -/
def coins_per_smaller_bag : ℕ := chest_coins / 2

/-- The total number of gold coins Jim found -/
def total_coins : ℕ := chest_coins + num_smaller_bags * coins_per_smaller_bag

/-- Theorem: Jim spent 8 hours scuba diving -/
theorem jim_diving_hours : total_coins / gold_coins_per_hour = 8 := by
  sorry

end NUMINAMATH_CALUDE_jim_diving_hours_l3794_379435


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l3794_379404

theorem root_reciprocal_sum (m : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*(m+1)*α + m + 4 = 0 ∧ 
              β^2 - 2*(m+1)*β + m + 4 = 0 ∧ 
              α ≠ β ∧
              1/α + 1/β = 1) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l3794_379404


namespace NUMINAMATH_CALUDE_fraction_simplification_l3794_379442

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) :
  (2 * a) / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3794_379442


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_proposition_l3794_379418

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x > 0) ↔ (∃ x > 0, x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_proposition_l3794_379418


namespace NUMINAMATH_CALUDE_standard_deck_two_card_selections_l3794_379451

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = suits * cards_per_suit)

/-- The number of ways to select two different cards from a deck, where order matters -/
def two_card_selections (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1)

/-- Theorem: The number of ways to select two different cards from a standard deck of 52 cards, where order matters, is 2652 -/
theorem standard_deck_two_card_selections :
  ∃ (d : Deck), d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ two_card_selections d = 2652 := by
  sorry

end NUMINAMATH_CALUDE_standard_deck_two_card_selections_l3794_379451


namespace NUMINAMATH_CALUDE_expression_simplification_l3794_379401

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1) / x / (x - 1 / x) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3794_379401


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3794_379440

theorem trigonometric_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  let half_sum := (α + β) / 2
  let half_gamma := γ / 2
  (Real.sin half_sum - Real.cos half_gamma = 0) ∧
  (Real.tan half_gamma + Real.tan half_sum - (1 / Real.tan half_sum + 1 / Real.tan half_gamma) = 0) ∧
  (Real.sin half_sum ^ 2 + (1 / Real.tan half_sum) * (1 / Real.tan half_gamma) - Real.cos half_gamma ^ 2 = 1) ∧
  (Real.cos half_sum ^ 2 + Real.tan half_sum * Real.tan half_gamma + Real.cos half_gamma ^ 2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3794_379440


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l3794_379462

theorem complex_imaginary_part (a : ℝ) (z : ℂ) : 
  z = 1 + a * I →  -- z is of the form 1 + ai
  a > 0 →  -- z is in the first quadrant
  Complex.abs z = Real.sqrt 5 →  -- |z| = √5
  z.im = 2 :=  -- The imaginary part of z is 2
by sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l3794_379462


namespace NUMINAMATH_CALUDE_wheat_packets_in_gunny_bag_l3794_379412

/-- The maximum number of wheat packets that can be accommodated in a gunny bag -/
def max_wheat_packets (bag_capacity : ℝ) (ton_to_kg : ℝ) (kg_to_g : ℝ) 
  (packet_weight_pounds : ℝ) (packet_weight_ounces : ℝ) 
  (pound_to_kg : ℝ) (ounce_to_g : ℝ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of wheat packets in the gunny bag -/
theorem wheat_packets_in_gunny_bag : 
  max_wheat_packets 13 1000 1000 16 4 0.453592 28.3495 = 1763 := by
  sorry

end NUMINAMATH_CALUDE_wheat_packets_in_gunny_bag_l3794_379412


namespace NUMINAMATH_CALUDE_min_circle_area_l3794_379491

/-- Given a line ax + by = 1 passing through point A(b, a), where O is the origin (0, 0),
    the minimum area of the circle with center O and radius OA is π. -/
theorem min_circle_area (a b : ℝ) (h : a * b = 1 / 2) :
  (π : ℝ) ≤ π * (a^2 + b^2) ∧ ∃ (a₀ b₀ : ℝ), a₀ * b₀ = 1 / 2 ∧ π * (a₀^2 + b₀^2) = π :=
by sorry

end NUMINAMATH_CALUDE_min_circle_area_l3794_379491


namespace NUMINAMATH_CALUDE_lucia_weekly_dance_cost_l3794_379423

/-- Represents the cost of dance classes for a week -/
def total_dance_cost (hip_hop_classes ballet_classes jazz_classes : ℕ) 
  (hip_hop_cost ballet_cost jazz_cost : ℕ) : ℕ :=
  hip_hop_classes * hip_hop_cost + ballet_classes * ballet_cost + jazz_classes * jazz_cost

/-- Proves that Lucia's weekly dance class cost is $52 -/
theorem lucia_weekly_dance_cost : 
  total_dance_cost 2 2 1 10 12 8 = 52 := by
  sorry

end NUMINAMATH_CALUDE_lucia_weekly_dance_cost_l3794_379423


namespace NUMINAMATH_CALUDE_brazilian_coffee_price_l3794_379459

/-- Proves that the price of Brazilian coffee is $3.75 per pound given the conditions of the coffee mix problem. -/
theorem brazilian_coffee_price
  (total_mix : ℝ)
  (columbian_price : ℝ)
  (final_mix_price : ℝ)
  (columbian_amount : ℝ)
  (h_total_mix : total_mix = 100)
  (h_columbian_price : columbian_price = 8.75)
  (h_final_mix_price : final_mix_price = 6.35)
  (h_columbian_amount : columbian_amount = 52) :
  let brazilian_amount : ℝ := total_mix - columbian_amount
  let brazilian_price : ℝ := (total_mix * final_mix_price - columbian_amount * columbian_price) / brazilian_amount
  brazilian_price = 3.75 := by
sorry


end NUMINAMATH_CALUDE_brazilian_coffee_price_l3794_379459


namespace NUMINAMATH_CALUDE_divisibility_counterexample_l3794_379416

theorem divisibility_counterexample : 
  ∃ (a b c : ℤ), (a ∣ b * c) ∧ ¬(a ∣ b) ∧ ¬(a ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_counterexample_l3794_379416


namespace NUMINAMATH_CALUDE_evaluate_expression_l3794_379411

theorem evaluate_expression : Real.sqrt (Real.sqrt 81) + Real.sqrt 256 - Real.sqrt 49 = 12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3794_379411


namespace NUMINAMATH_CALUDE_complement_of_A_l3794_379485

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 ≥ 0}

-- State the theorem
theorem complement_of_A :
  (Set.univ : Set ℝ) \ A = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3794_379485
