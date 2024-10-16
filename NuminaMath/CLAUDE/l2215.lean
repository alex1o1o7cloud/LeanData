import Mathlib

namespace NUMINAMATH_CALUDE_jack_waiting_time_l2215_221591

/-- The total waiting time in hours for Jack's travel to Canada -/
def total_waiting_time (customs_hours : ℕ) (quarantine_days : ℕ) : ℕ :=
  customs_hours + 24 * quarantine_days

/-- Theorem stating that Jack's total waiting time is 356 hours -/
theorem jack_waiting_time :
  total_waiting_time 20 14 = 356 := by
  sorry

end NUMINAMATH_CALUDE_jack_waiting_time_l2215_221591


namespace NUMINAMATH_CALUDE_solve_quadratic_l2215_221524

theorem solve_quadratic (x : ℝ) (h1 : x^2 - 4*x = 0) (h2 : x ≠ 0) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_l2215_221524


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2215_221567

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (f : ℝ × ℝ) (M N : ℝ × ℝ) :
  (∃ c : ℝ, c > 0 ∧ f = (c, 0) ∧ ∀ x y : ℝ, y^2 = 4 * Real.sqrt 7 * x → (x - c)^2 + y^2 = c^2) →  -- focus coincides with parabola focus
  (M.1 - 1 = M.2 ∧ N.1 - 1 = N.2) →  -- M and N are on the line y = x - 1
  ((M.1 + N.1) / 2 = -2/3) →  -- x-coordinate of midpoint is -2/3
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ 
    ((x - f.1)^2 / (a^2 + b^2) + (y - f.2)^2 / (a^2 + b^2) = 1 ∧
     (x + f.1)^2 / (a^2 + b^2) + (y - f.2)^2 / (a^2 + b^2) = 1)) →
  ∃ x y : ℝ, x^2/2 - y^2/5 = 1 ↔
    ((x - f.1)^2 / 7 + (y - f.2)^2 / 7 = 1 ∧
     (x + f.1)^2 / 7 + (y - f.2)^2 / 7 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2215_221567


namespace NUMINAMATH_CALUDE_workshop_efficiency_l2215_221551

theorem workshop_efficiency (x : ℝ) (h : x > 0) : 
  (3000 / x) - (3000 / (2.5 * x)) = (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_workshop_efficiency_l2215_221551


namespace NUMINAMATH_CALUDE_video_recorder_markup_l2215_221521

theorem video_recorder_markup (wholesale_cost : ℝ) (employee_discount : ℝ) (employee_paid : ℝ) :
  wholesale_cost = 200 →
  employee_discount = 0.30 →
  employee_paid = 168 →
  ∃ (markup : ℝ), 
    employee_paid = (1 - employee_discount) * (wholesale_cost * (1 + markup)) ∧
    markup = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_markup_l2215_221521


namespace NUMINAMATH_CALUDE_complex_power_difference_l2215_221536

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^40 - (1 - i)^40 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2215_221536


namespace NUMINAMATH_CALUDE_complex_magnitude_l2215_221502

theorem complex_magnitude (i z : ℂ) : 
  i * i = -1 → i * z = 1 - i → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2215_221502


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l2215_221562

/-- Represents Conor's vegetable chopping capacity and work schedule --/
structure VegetableChopper where
  eggplants_per_day : ℕ
  carrots_per_day : ℕ
  potatoes_per_day : ℕ
  work_days_per_week : ℕ

/-- Calculates the total number of vegetables chopped in a week --/
def total_vegetables_per_week (c : VegetableChopper) : ℕ :=
  (c.eggplants_per_day + c.carrots_per_day + c.potatoes_per_day) * c.work_days_per_week

/-- Theorem stating that Conor can chop 116 vegetables in a week --/
theorem conor_weekly_vegetables :
  ∃ c : VegetableChopper,
    c.eggplants_per_day = 12 ∧
    c.carrots_per_day = 9 ∧
    c.potatoes_per_day = 8 ∧
    c.work_days_per_week = 4 ∧
    total_vegetables_per_week c = 116 :=
by
  sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l2215_221562


namespace NUMINAMATH_CALUDE_unique_right_triangle_l2215_221598

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (3,4,5) forms a right triangle --/
theorem unique_right_triangle :
  (¬ isRightTriangle 2 3 4) ∧
  (¬ isRightTriangle 3 4 6) ∧
  (isRightTriangle 3 4 5) ∧
  (¬ isRightTriangle 4 5 6) :=
by sorry

#check unique_right_triangle

end NUMINAMATH_CALUDE_unique_right_triangle_l2215_221598


namespace NUMINAMATH_CALUDE_circle_equation_l2215_221599

theorem circle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (4, 6)
  let center_line (x y : ℝ) := x - 2*y - 1 = 0
  ∃ (h k : ℝ), 
    center_line h k ∧ 
    (h - A.1)^2 + (k - A.2)^2 = (h - B.1)^2 + (k - B.2)^2 ∧
    (x - h)^2 + (y - k)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2215_221599


namespace NUMINAMATH_CALUDE_class_average_mark_l2215_221587

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 10 →
  excluded_students = 5 →
  excluded_avg = 50 →
  remaining_avg = 90 →
  (total_students * (total_students * excluded_avg + (total_students - excluded_students) * remaining_avg) / total_students) / total_students = 70 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2215_221587


namespace NUMINAMATH_CALUDE_product_division_equality_l2215_221501

theorem product_division_equality : (400 * 7000) / (100^1) = 28000 := by
  sorry

end NUMINAMATH_CALUDE_product_division_equality_l2215_221501


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l2215_221575

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of G_1000 is 2 -/
theorem units_digit_G_1000 : units_digit (G 1000) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l2215_221575


namespace NUMINAMATH_CALUDE_unique_solution_l2215_221593

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  x = 11 ∧ y = 9 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2215_221593


namespace NUMINAMATH_CALUDE_expression_evaluation_l2215_221509

theorem expression_evaluation :
  65 + (160 / 8) + (35 * 12) - 450 - (504 / 7) = -17 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2215_221509


namespace NUMINAMATH_CALUDE_intersection_area_is_nine_l2215_221500

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the intersection of two rectangles -/
structure Intersection where
  rect1 : Rectangle
  rect2 : Rectangle
  isSquare : Bool
  area : ℝ

/-- Theorem: Area of intersection between two specific rectangles -/
theorem intersection_area_is_nine 
  (r1 : Rectangle) 
  (r2 : Rectangle) 
  (i : Intersection) 
  (h1 : r1.width = 4 ∧ r1.height = 12) 
  (h2 : r2.width = 3 ∧ r2.height = 7) 
  (h3 : i.rect1 = r1 ∧ i.rect2 = r2) 
  (h4 : i.isSquare = true) : 
  i.area = 9 := by
  sorry


end NUMINAMATH_CALUDE_intersection_area_is_nine_l2215_221500


namespace NUMINAMATH_CALUDE_solution_value_l2215_221537

theorem solution_value (a b x y : ℝ) : 
  x = 2 ∧ 
  y = -1 ∧ 
  a * x + b * y = 1 ∧ 
  b * x + a * y = 7 → 
  (a + b) * (a - b) = -16 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l2215_221537


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l2215_221518

theorem polynomial_uniqueness (Q : ℝ → ℝ) :
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2) →
  Q (-1) = 3 →
  Q 3 = 15 →
  ∀ x, Q x = -2 * x^2 + 6 * x - 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l2215_221518


namespace NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l2215_221553

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds :
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l2215_221553


namespace NUMINAMATH_CALUDE_medal_ratio_is_two_to_one_l2215_221572

/-- The ratio of swimming medals to track medals -/
def medal_ratio (total_medals track_medals badminton_medals : ℕ) : ℚ :=
  let swimming_medals := total_medals - track_medals - badminton_medals
  (swimming_medals : ℚ) / track_medals

/-- Theorem stating that the ratio of swimming medals to track medals is 2:1 -/
theorem medal_ratio_is_two_to_one :
  medal_ratio 20 5 5 = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_medal_ratio_is_two_to_one_l2215_221572


namespace NUMINAMATH_CALUDE_complex_square_equality_l2215_221552

theorem complex_square_equality (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 3 + 4 * Complex.I →
  a + b * Complex.I = 2 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2215_221552


namespace NUMINAMATH_CALUDE_divisibility_condition_l2215_221565

theorem divisibility_condition (n : ℕ) : 
  (2^n + n) ∣ (8^n + n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2215_221565


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2215_221581

theorem integer_sum_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 15)
  (h2 : x.val * y.val = 56) :
  x.val + y.val = Real.sqrt 449 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2215_221581


namespace NUMINAMATH_CALUDE_max_value_complex_l2215_221519

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + 3*z + Complex.I*2) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_l2215_221519


namespace NUMINAMATH_CALUDE_justins_dogs_l2215_221531

theorem justins_dogs (camden_dogs rico_dogs justin_dogs : ℕ) : 
  camden_dogs = (3 * rico_dogs) / 4 →
  rico_dogs = justin_dogs + 10 →
  camden_dogs * 4 = 72 →
  justin_dogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_justins_dogs_l2215_221531


namespace NUMINAMATH_CALUDE_area_conversion_time_conversion_l2215_221588

-- Define the conversion factors
def square_meters_per_hectare : ℝ := 10000
def minutes_per_hour : ℝ := 60

-- Define the input values
def area_in_square_meters : ℝ := 123000
def time_in_hours : ℝ := 4.25

-- Theorem for area conversion
theorem area_conversion :
  area_in_square_meters / square_meters_per_hectare = 12.3 := by sorry

-- Theorem for time conversion
theorem time_conversion :
  ∃ (whole_hours minutes : ℕ),
    whole_hours = 4 ∧
    minutes = 15 ∧
    time_in_hours = whole_hours + (minutes : ℝ) / minutes_per_hour := by sorry

end NUMINAMATH_CALUDE_area_conversion_time_conversion_l2215_221588


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2215_221517

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem not_sufficient_not_necessary 
  (m : Line) (α β : Plane) 
  (h_perp_planes : perpendicular_planes α β) :
  ¬(∀ m α β, parallel m α → perpendicular m β) ∧ 
  ¬(∀ m α β, perpendicular m β → parallel m α) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2215_221517


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2215_221526

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 - 2*x + x^2) = 9 ↔ x = 1 + Real.sqrt 78 ∨ x = 1 - Real.sqrt 78 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2215_221526


namespace NUMINAMATH_CALUDE_nancys_hourly_wage_l2215_221504

/-- Proves that Nancy needs to make $10 per hour to pay the rest of her tuition --/
theorem nancys_hourly_wage (tuition : ℝ) (scholarship : ℝ) (work_hours : ℝ) :
  tuition = 22000 →
  scholarship = 3000 →
  work_hours = 200 →
  (tuition / 2 - scholarship - 2 * scholarship) / work_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancys_hourly_wage_l2215_221504


namespace NUMINAMATH_CALUDE_increasing_equivalent_l2215_221579

/-- A function is increasing on an interval if its graph always rises when viewed from left to right. -/
def IncreasingOnInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem increasing_equivalent {f : ℝ → ℝ} {I : Set ℝ} :
  IncreasingOnInterval f I ↔
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_increasing_equivalent_l2215_221579


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2215_221585

/-- The simultaneous equations y = kx + 5 and y = (3k - 2)x + 6 have at least one solution
    in terms of real numbers (x, y) if and only if k ≠ 1 -/
theorem simultaneous_equations_solution (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 5 ∧ y = (3 * k - 2) * x + 6) ↔ k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2215_221585


namespace NUMINAMATH_CALUDE_cos_equality_proof_l2215_221535

theorem cos_equality_proof (n : ℤ) : 
  n = 43 ∧ -180 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l2215_221535


namespace NUMINAMATH_CALUDE_ratio_x_y_is_two_l2215_221545

theorem ratio_x_y_is_two (x y a : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (eq1 : x^3 + Real.log x + 2*a^2 = 0) 
  (eq2 : 4*y^3 + Real.log (Real.sqrt y) + Real.log (Real.sqrt 2) + a^2 = 0) : 
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_y_is_two_l2215_221545


namespace NUMINAMATH_CALUDE_negation_of_neither_even_l2215_221529

theorem negation_of_neither_even (a b : ℤ) : 
  ¬(¬(Even a) ∧ ¬(Even b)) ↔ (Even a ∨ Even b) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_neither_even_l2215_221529


namespace NUMINAMATH_CALUDE_common_factor_of_polynomials_l2215_221596

theorem common_factor_of_polynomials (a b : ℝ) :
  ∃ (k₁ k₂ : ℝ), (4 * a^2 - 2 * a * b = (2 * a - b) * k₁) ∧
                 (4 * a^2 - b^2 = (2 * a - b) * k₂) := by
  sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomials_l2215_221596


namespace NUMINAMATH_CALUDE_calculate_expression_l2215_221548

theorem calculate_expression : (-Real.sqrt 6)^2 - 3 * Real.sqrt 2 * Real.sqrt 18 = -12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2215_221548


namespace NUMINAMATH_CALUDE_initial_stock_value_l2215_221532

/-- Represents the daily change in stock value -/
def daily_change : ℤ := 1

/-- Represents the number of days until the stock reaches $200 -/
def days_to_target : ℕ := 100

/-- Represents the target value of the stock -/
def target_value : ℤ := 200

/-- Theorem stating that the initial stock value is $101 -/
theorem initial_stock_value (V : ℤ) :
  V + (days_to_target - 1) * daily_change = target_value →
  V = 101 := by
  sorry

end NUMINAMATH_CALUDE_initial_stock_value_l2215_221532


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2215_221590

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≥ 100 → n.mod 17 = 0 → n ≥ 102 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2215_221590


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2215_221563

def range_start : ℕ := 1
def range_end : ℕ := 60
def num_choices : ℕ := 3

def is_multiple_of_four (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

theorem probability_at_least_one_multiple_of_four :
  let total_numbers := range_end - range_start + 1
  let multiples_of_four := (range_end / 4) - ((range_start - 1) / 4)
  let prob_not_multiple := (total_numbers - multiples_of_four) / total_numbers
  (1 : ℚ) - (prob_not_multiple ^ num_choices) = 37 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2215_221563


namespace NUMINAMATH_CALUDE_fixed_point_range_l2215_221597

/-- A function f: ℝ → ℝ has a fixed point if there exists an x such that f(x) = x -/
def HasFixedPoint (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = x

/-- The quadratic function f(x) = x^2 + x + a -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + x + a

theorem fixed_point_range (a : ℝ) :
  HasFixedPoint (f a) → a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_range_l2215_221597


namespace NUMINAMATH_CALUDE_theater_revenue_specific_case_l2215_221511

def theater_revenue (orchestra_price balcony_price : ℕ) 
                    (total_tickets balcony_orchestra_diff : ℕ) : ℕ :=
  let orchestra_tickets := (total_tickets - balcony_orchestra_diff) / 2
  let balcony_tickets := total_tickets - orchestra_tickets
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets

theorem theater_revenue_specific_case :
  theater_revenue 12 8 360 140 = 3320 := by
  sorry

end NUMINAMATH_CALUDE_theater_revenue_specific_case_l2215_221511


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2215_221507

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2215_221507


namespace NUMINAMATH_CALUDE_min_value_a5_plus_a6_l2215_221512

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r d : ℝ), r > 1 ∧ d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n + d

theorem min_value_a5_plus_a6 (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
  ∃ (min : ℝ), min = 48 ∧ ∀ x, (∃ b, ArithmeticGeometricSequence b ∧ 
    b 4 + b 3 - 2 * b 2 - 2 * b 1 = 6 ∧ b 5 + b 6 = x) → x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_a5_plus_a6_l2215_221512


namespace NUMINAMATH_CALUDE_range_of_m_l2215_221576

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) * (x - 1) < 0}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < 1}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∅ ≠ B m) ∧ 
  (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ 
  (∃ y : ℝ, y ∈ A ∧ y ∉ B m) →
  -1 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2215_221576


namespace NUMINAMATH_CALUDE_square_2023_position_l2215_221558

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DABC
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the function to get the nth square position
def nthSquarePosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.DABC
  | 2 => SquarePosition.CBAD
  | 3 => SquarePosition.DCBA
  | _ => SquarePosition.ABCD -- This case is not actually possible

theorem square_2023_position : nthSquarePosition 2023 = SquarePosition.DABC := by
  sorry

end NUMINAMATH_CALUDE_square_2023_position_l2215_221558


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l2215_221554

/-- Given a geometric sequence where the 5th term is 2 and the 8th term is 16,
    prove that the 11th term is 128. -/
theorem geometric_sequence_11th_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n m, a (n + 1) / a n = a (m + 1) / a m)  -- Geometric sequence condition
  (h_5th : a 5 = 2)  -- 5th term is 2
  (h_8th : a 8 = 16)  -- 8th term is 16
  : a 11 = 128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l2215_221554


namespace NUMINAMATH_CALUDE_octagon_triangle_angle_sum_l2215_221541

theorem octagon_triangle_angle_sum :
  ∀ (ABC ABD : ℝ),
  (∃ (n : ℕ), n = 8 ∧ ABC = 180 * (n - 2) / n) →
  (∃ (m : ℕ), m = 3 ∧ ABD = 180 * (m - 2) / m) →
  ABC + ABD = 195 := by
sorry

end NUMINAMATH_CALUDE_octagon_triangle_angle_sum_l2215_221541


namespace NUMINAMATH_CALUDE_isosceles_triangle_line_equation_l2215_221557

/-- An isosceles triangle AOB with given properties -/
structure IsoscelesTriangle where
  /-- Point O is at the origin -/
  O : ℝ × ℝ := (0, 0)
  /-- Point A coordinates -/
  A : ℝ × ℝ := (1, 3)
  /-- Point B is on the positive x-axis -/
  B : ℝ × ℝ
  /-- B's y-coordinate is 0 -/
  h_B_on_x_axis : B.2 = 0
  /-- B's x-coordinate is positive -/
  h_B_positive_x : B.1 > 0
  /-- AO = AB (isosceles property) -/
  h_isosceles : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- The equation of line AB in an isosceles triangle AOB is y-3 = -3(x-1) -/
theorem isosceles_triangle_line_equation (t : IsoscelesTriangle) :
  ∀ x y : ℝ, (y - 3 = -3 * (x - 1)) ↔ (∃ k : ℝ, x = t.A.1 + k * (t.B.1 - t.A.1) ∧ y = t.A.2 + k * (t.B.2 - t.A.2)) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_line_equation_l2215_221557


namespace NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l2215_221520

/-- Represents a solution with its acid concentration and volume -/
structure Solution where
  concentration : Real
  volume : Real

/-- Calculates the amount of pure acid in a solution -/
def pureAcidAmount (s : Solution) : Real :=
  s.concentration * s.volume

/-- Theorem: The total amount of pure acid in a mixture of solutions is the sum of pure acid amounts from each solution -/
theorem total_pure_acid_in_mixture (solutionA solutionB solutionC : Solution)
  (hA : solutionA.concentration = 0.20 ∧ solutionA.volume = 8)
  (hB : solutionB.concentration = 0.35 ∧ solutionB.volume = 5)
  (hC : solutionC.concentration = 0.15 ∧ solutionC.volume = 3) :
  pureAcidAmount solutionA + pureAcidAmount solutionB + pureAcidAmount solutionC = 3.8 := by
  sorry


end NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l2215_221520


namespace NUMINAMATH_CALUDE_special_school_total_students_l2215_221530

/-- A school with blind and deaf students -/
structure School where
  blind_students : ℕ
  deaf_students : ℕ

/-- The total number of students in the school -/
def total_students (s : School) : ℕ := s.blind_students + s.deaf_students

/-- Theorem: The total number of students in the special school is 180 -/
theorem special_school_total_students :
  ∃ (s : School), s.blind_students = 45 ∧ s.deaf_students = 3 * s.blind_students ∧ total_students s = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_special_school_total_students_l2215_221530


namespace NUMINAMATH_CALUDE_water_in_tank_after_rain_l2215_221506

/-- Given an initial amount of water, a water flow rate, and a rainstorm duration,
    calculate the final amount of water in the tank. -/
def final_water_amount (initial_amount : ℝ) (flow_rate : ℝ) (duration : ℝ) : ℝ :=
  initial_amount + flow_rate * duration

/-- Theorem stating that given the specific conditions in the problem,
    the final amount of water in the tank is 280 L. -/
theorem water_in_tank_after_rain : final_water_amount 100 2 90 = 280 := by
  sorry

end NUMINAMATH_CALUDE_water_in_tank_after_rain_l2215_221506


namespace NUMINAMATH_CALUDE_performance_orders_count_l2215_221589

/-- The number of ways to select 4 programs from 8 options -/
def total_options : ℕ := 8

/-- The number of programs to be selected -/
def selected_programs : ℕ := 4

/-- The number of special programs (A and B) -/
def special_programs : ℕ := 2

/-- The number of non-special programs -/
def other_programs : ℕ := total_options - special_programs

/-- Calculates the number of performance orders with only one special program -/
def orders_with_one_special : ℕ :=
  special_programs * (Nat.choose other_programs (selected_programs - 1)) * (Nat.factorial selected_programs)

/-- Calculates the number of performance orders with both special programs -/
def orders_with_both_special : ℕ :=
  (Nat.choose other_programs (selected_programs - 2)) * (Nat.factorial 2) * (Nat.factorial (selected_programs - 2))

/-- The total number of valid performance orders -/
def total_orders : ℕ := orders_with_one_special + orders_with_both_special

theorem performance_orders_count :
  total_orders = 2860 :=
sorry

end NUMINAMATH_CALUDE_performance_orders_count_l2215_221589


namespace NUMINAMATH_CALUDE_farmer_cows_problem_l2215_221540

theorem farmer_cows_problem (initial_cows : ℕ) (final_cows : ℕ) (new_cows : ℕ) : 
  initial_cows = 51 →
  final_cows = 42 →
  (3 : ℚ) / 4 * (initial_cows + new_cows) = final_cows →
  new_cows = 5 := by
sorry

end NUMINAMATH_CALUDE_farmer_cows_problem_l2215_221540


namespace NUMINAMATH_CALUDE_expand_product_l2215_221582

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2215_221582


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2215_221568

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 5 → total_games = 20 → (n * (n - 1)) / 2 = total_games → n - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2215_221568


namespace NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l2215_221525

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l2215_221525


namespace NUMINAMATH_CALUDE_cyclist_speed_l2215_221510

/-- Proves that a cyclist's speed is 24 km/h given specific conditions -/
theorem cyclist_speed (hiker_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) 
  (hiker_speed_positive : 0 < hiker_speed)
  (cyclist_travel_time_positive : 0 < cyclist_travel_time)
  (hiker_catch_up_time_positive : 0 < hiker_catch_up_time)
  (hiker_speed_val : hiker_speed = 4)
  (cyclist_travel_time_val : cyclist_travel_time = 5 / 60)
  (hiker_catch_up_time_val : hiker_catch_up_time = 25 / 60) : 
  ∃ (cyclist_speed : ℝ), cyclist_speed = 24 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_speed_l2215_221510


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l2215_221560

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 5 -/
def num_y_intercepts : ℕ := 0

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 4 * y + 5

theorem parabola_y_intercepts :
  (∀ y : ℝ, parabola_equation y ≠ 0) ∧ num_y_intercepts = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l2215_221560


namespace NUMINAMATH_CALUDE_gcf_of_36_and_54_l2215_221546

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_54_l2215_221546


namespace NUMINAMATH_CALUDE_estate_area_calculation_l2215_221533

/-- Represents the side length of the square on the map in inches -/
def map_side_length : ℝ := 12

/-- Represents the scale of the map in miles per inch -/
def map_scale : ℝ := 100

/-- Calculates the actual side length of the estate in miles -/
def actual_side_length : ℝ := map_side_length * map_scale

/-- Calculates the actual area of the estate in square miles -/
def actual_area : ℝ := actual_side_length ^ 2

/-- Theorem stating that the actual area of the estate is 1440000 square miles -/
theorem estate_area_calculation : actual_area = 1440000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l2215_221533


namespace NUMINAMATH_CALUDE_chalk_problem_l2215_221550

theorem chalk_problem (total_people : ℕ) (added_chalk : ℕ) (lost_chalk : ℕ) (final_per_person : ℚ) :
  total_people = 11 →
  added_chalk = 28 →
  lost_chalk = 4 →
  final_per_person = 5.5 →
  ∃ (original_chalk : ℕ), original_chalk = 37 ∧ 
    (↑original_chalk - ↑lost_chalk + ↑added_chalk : ℚ) = ↑total_people * final_per_person :=
by sorry

end NUMINAMATH_CALUDE_chalk_problem_l2215_221550


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l2215_221559

/-- A function f: ℝ → ℝ that is symmetric about x = 2 and has exactly four distinct real roots -/
def SymmetricFourRootFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∃! (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0)

/-- The sum of the four distinct real roots of a SymmetricFourRootFunction is 8 -/
theorem sum_of_roots_is_eight (f : ℝ → ℝ) (h : SymmetricFourRootFunction f) :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    a + b + c + d = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l2215_221559


namespace NUMINAMATH_CALUDE_fill_time_with_leak_l2215_221570

/-- Time to fill the cistern without a leak (in hours) -/
def fill_time : ℝ := 8

/-- Time to empty the full cistern through the leak (in hours) -/
def empty_time : ℝ := 24

/-- Theorem: The time to fill the cistern with a leak is 12 hours -/
theorem fill_time_with_leak : 
  (1 / fill_time - 1 / empty_time)⁻¹ = 12 := by sorry

end NUMINAMATH_CALUDE_fill_time_with_leak_l2215_221570


namespace NUMINAMATH_CALUDE_max_value_of_z_l2215_221503

/-- Given real numbers x and y satisfying the conditions,
    prove that the maximum value of z = 2x - y is 5 -/
theorem max_value_of_z (x y : ℝ) 
  (h1 : x - 2*y + 2 ≥ 0) 
  (h2 : x + y ≤ 1) 
  (h3 : y + 1 ≥ 0) : 
  ∃ (z : ℝ), z = 2*x - y ∧ z ≤ 5 ∧ ∀ (w : ℝ), w = 2*x - y → w ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2215_221503


namespace NUMINAMATH_CALUDE_tangerines_count_l2215_221505

/-- The number of tangerines in a fruit basket -/
def num_tangerines (total fruits bananas apples pears : ℕ) : ℕ :=
  total - (bananas + apples + pears)

/-- Theorem: There are 13 tangerines in the fruit basket -/
theorem tangerines_count :
  let total := 60
  let bananas := 32
  let apples := 10
  let pears := 5
  num_tangerines total bananas apples pears = 13 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_count_l2215_221505


namespace NUMINAMATH_CALUDE_smile_area_l2215_221514

/-- The area of the "smile" region formed by two sectors and a semicircle -/
theorem smile_area : 
  ∀ (r₁ r₂ : ℝ) (θ : ℝ),
  r₁ = 3 → r₂ = 2 → θ = π/4 →
  2 * (1/2 * r₁^2 * θ) + 1/2 * π * r₂^2 = 17*π/4 :=
by sorry

end NUMINAMATH_CALUDE_smile_area_l2215_221514


namespace NUMINAMATH_CALUDE_commission_calculation_l2215_221592

/-- Calculates the commission amount given a commission rate and total sales -/
def calculate_commission (rate : ℚ) (sales : ℚ) : ℚ :=
  rate * sales

theorem commission_calculation :
  let rate : ℚ := 25 / 1000  -- 2.5% expressed as a rational number
  let sales : ℚ := 600
  calculate_commission rate sales = 15 := by
  sorry

end NUMINAMATH_CALUDE_commission_calculation_l2215_221592


namespace NUMINAMATH_CALUDE_no_solution_condition_l2215_221580

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (m * x - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2215_221580


namespace NUMINAMATH_CALUDE_snail_return_time_l2215_221549

/-- Represents the movement of a point on a plane -/
structure PointMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the position of the point at a given time -/
def Position := ℝ × ℝ

/-- Returns the position of the point after a given time -/
noncomputable def positionAfterTime (m : PointMovement) (t : ℝ) : Position :=
  sorry

/-- Checks if the point has returned to its starting position -/
def hasReturnedToStart (m : PointMovement) (t : ℝ) : Prop :=
  positionAfterTime m t = (0, 0)

/-- The main theorem to prove -/
theorem snail_return_time (m : PointMovement) 
    (h1 : m.speed > 0)
    (h2 : m.turnInterval = 15)
    (h3 : m.turnAngle = 90) :
    ∀ t : ℝ, hasReturnedToStart m t → ∃ n : ℕ, t = 60 * n := by
  sorry

end NUMINAMATH_CALUDE_snail_return_time_l2215_221549


namespace NUMINAMATH_CALUDE_parabola_max_value_l2215_221515

def f (x : ℝ) : ℝ := -(x + 1)^2 + 3

theorem parabola_max_value :
  ∀ x : ℝ, f x ≤ 3 ∧ ∃ x₀ : ℝ, f x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_value_l2215_221515


namespace NUMINAMATH_CALUDE_daughter_weight_l2215_221586

/-- Represents the weights of family members -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  child : ℝ

/-- The conditions of the family weight problem -/
def FamilyWeightProblem (w : FamilyWeights) : Prop :=
  w.grandmother + w.daughter + w.child = 110 ∧
  w.daughter + w.child = 60 ∧
  w.child = (1 / 5) * w.grandmother

/-- The theorem stating that given the conditions, the daughter's weight is 50 kg -/
theorem daughter_weight (w : FamilyWeights) : 
  FamilyWeightProblem w → w.daughter = 50 := by
  sorry


end NUMINAMATH_CALUDE_daughter_weight_l2215_221586


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l2215_221555

theorem longest_segment_in_quarter_circle (r : ℝ) (h : r = 9) :
  let sector_chord_length_squared := 2 * r^2
  sector_chord_length_squared = 162 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l2215_221555


namespace NUMINAMATH_CALUDE_andrew_total_donation_l2215_221539

/-- Calculates the total donation amount for a geometric series of donations -/
def totalDonation (initialAmount : ℕ) (commonRatio : ℕ) (startAge : ℕ) (currentAge : ℕ) : ℕ :=
  let numberOfTerms := currentAge - startAge + 1
  initialAmount * (commonRatio ^ numberOfTerms - 1) / (commonRatio - 1)

/-- Theorem stating that Andrew's total donation equals 3,669,609k -/
theorem andrew_total_donation :
  totalDonation 7000 2 11 29 = 3669609000 := by
  sorry


end NUMINAMATH_CALUDE_andrew_total_donation_l2215_221539


namespace NUMINAMATH_CALUDE_playground_area_is_297_l2215_221595

/-- Calculates the area of a rectangular playground given the specified conditions --/
def playground_area (total_posts : ℕ) (post_spacing : ℕ) : ℕ :=
  let shorter_side_posts := 4  -- Including corners
  let longer_side_posts := 3 * shorter_side_posts
  let shorter_side_length := post_spacing * (shorter_side_posts - 1)
  let longer_side_length := post_spacing * (longer_side_posts - 1)
  shorter_side_length * longer_side_length

/-- Theorem stating that the area of the playground under given conditions is 297 square yards --/
theorem playground_area_is_297 :
  playground_area 24 3 = 297 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_is_297_l2215_221595


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2215_221523

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 1 → b = 2 → c^2 = a^2 + b^2 → c = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2215_221523


namespace NUMINAMATH_CALUDE_sin_minus_cos_special_angle_l2215_221527

/-- Given an angle α whose terminal side passes through the point (3a, -4a) where a < 0,
    prove that sinα - cosα = 7/5 -/
theorem sin_minus_cos_special_angle (a : ℝ) (α : Real) (h : a < 0) 
    (h_terminal : ∃ k : ℝ, k > 0 ∧ k * Real.cos α = 3 * a ∧ k * Real.sin α = -4 * a) :
    Real.sin α - Real.cos α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_special_angle_l2215_221527


namespace NUMINAMATH_CALUDE_encryption_theorem_l2215_221528

/-- Represents the encryption table --/
def encryption_table : Fin 16 → Fin 16 := sorry

/-- Applies the encryption once to a string of 16 characters --/
def apply_encryption (s : String) : String := sorry

/-- Applies the encryption n times to a string --/
def apply_encryption_n_times (s : String) (n : ℕ) : String := sorry

/-- The last three characters of a string --/
def last_three (s : String) : String := sorry

theorem encryption_theorem :
  ∀ s : String,
  last_three s = "уао" →
  apply_encryption_n_times (apply_encryption s) 2014 = s →
  ∃ t : String, last_three t = "чку" ∧ apply_encryption_n_times t 2015 = s :=
sorry

end NUMINAMATH_CALUDE_encryption_theorem_l2215_221528


namespace NUMINAMATH_CALUDE_average_five_equals_three_l2215_221564

theorem average_five_equals_three (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + 3) / 6 = 3 →
  (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_five_equals_three_l2215_221564


namespace NUMINAMATH_CALUDE_circle_radius_increase_circle_radius_increase_is_five_over_pi_l2215_221544

/-- Represents the change in radius when a circle's circumference increases from 30 to 40 inches -/
theorem circle_radius_increase : ℝ → Prop :=
  fun Δr =>
    ∃ (r₁ r₂ : ℝ),
      (2 * Real.pi * r₁ = 30) ∧
      (2 * Real.pi * r₂ = 40) ∧
      (r₂ - r₁ = Δr) ∧
      (Δr = 5 / Real.pi)

/-- Proves that the radius increase is 5/π inches -/
theorem circle_radius_increase_is_five_over_pi :
  circle_radius_increase (5 / Real.pi) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_circle_radius_increase_is_five_over_pi_l2215_221544


namespace NUMINAMATH_CALUDE_prob_A_hit_given_target_hit_l2215_221569

/-- The probability of A hitting the target -/
def prob_A_hit : ℚ := 3/5

/-- The probability of B hitting the target -/
def prob_B_hit : ℚ := 4/5

/-- The probability of the target being hit by either A or B -/
def prob_target_hit : ℚ := 1 - (1 - prob_A_hit) * (1 - prob_B_hit)

/-- The probability of A hitting the target (regardless of B) -/
def prob_A_hit_total : ℚ := prob_A_hit * (1 - prob_B_hit) + prob_A_hit * prob_B_hit

theorem prob_A_hit_given_target_hit :
  prob_A_hit_total / prob_target_hit = 15/23 :=
sorry

end NUMINAMATH_CALUDE_prob_A_hit_given_target_hit_l2215_221569


namespace NUMINAMATH_CALUDE_possible_perimeters_only_possible_perimeters_l2215_221577

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the possible ways to cut the original rectangle -/
inductive Cut
  | Vertical
  | Horizontal
  | Mixed

/-- The original rectangle -/
def originalRect : Rectangle := { length := 6, width := 3 }

/-- Theorem stating the possible perimeters of the resulting rectangles -/
theorem possible_perimeters :
  ∃ (c : Cut) (r : Rectangle),
    (c = Cut.Vertical ∧ perimeter r = 14) ∨
    (c = Cut.Horizontal ∧ perimeter r = 10) ∨
    (c = Cut.Mixed ∧ perimeter r = 10.5) :=
  sorry

/-- Theorem stating that these are the only possible perimeters -/
theorem only_possible_perimeters :
  ∀ (c : Cut) (r : Rectangle),
    (perimeter r ≠ 14 ∧ perimeter r ≠ 10 ∧ perimeter r ≠ 10.5) →
    ¬(∃ (r1 r2 : Rectangle), 
      perimeter r = perimeter r1 ∧
      perimeter r = perimeter r2 ∧
      r.length + r1.length + r2.length = originalRect.length ∧
      r.width = r1.width ∧ r.width = r2.width ∧ r.width = originalRect.width) :=
  sorry

end NUMINAMATH_CALUDE_possible_perimeters_only_possible_perimeters_l2215_221577


namespace NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l2215_221594

theorem one_fourth_in_one_eighth :
  (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l2215_221594


namespace NUMINAMATH_CALUDE_tims_change_l2215_221584

/-- Tim's change calculation -/
theorem tims_change (initial_amount : ℕ) (spent_amount : ℕ) (change : ℕ) : 
  initial_amount = 50 → spent_amount = 45 → change = initial_amount - spent_amount → change = 5 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_l2215_221584


namespace NUMINAMATH_CALUDE_cos_45_cos_15_plus_sin_45_sin_15_l2215_221578

theorem cos_45_cos_15_plus_sin_45_sin_15 :
  Real.cos (45 * π / 180) * Real.cos (15 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_cos_15_plus_sin_45_sin_15_l2215_221578


namespace NUMINAMATH_CALUDE_triangle_division_ratio_l2215_221508

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the sum of squares of distances from vertices to division points -/
def S (t : Triangle) (n : ℕ) : ℝ := sorry

/-- Theorem: The ratio of S to the sum of squared side lengths is a specific rational function of n -/
theorem triangle_division_ratio (t : Triangle) (n : ℕ) (h : n > 0) :
  S t n / (t.a^2 + t.b^2 + t.c^2) = (n - 1) * (5 * n - 1) / (6 * n) := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_ratio_l2215_221508


namespace NUMINAMATH_CALUDE_race_result_theorem_l2215_221522

-- Define the girls
inductive Girl : Type
  | Anna : Girl
  | Bella : Girl
  | Csilla : Girl
  | Dora : Girl

-- Define the positions
inductive Position : Type
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

def race_result : Girl → Position := sorry

-- Define the statements
def anna_statement : Prop := race_result Girl.Anna ≠ Position.First ∧ race_result Girl.Anna ≠ Position.Fourth
def bella_statement : Prop := race_result Girl.Bella ≠ Position.First
def csilla_statement : Prop := race_result Girl.Csilla = Position.First
def dora_statement : Prop := race_result Girl.Dora = Position.Fourth

-- Define the condition that three statements are true and one is false
def statements_condition : Prop :=
  (anna_statement ∧ bella_statement ∧ csilla_statement ∧ ¬dora_statement) ∨
  (anna_statement ∧ bella_statement ∧ ¬csilla_statement ∧ dora_statement) ∨
  (anna_statement ∧ ¬bella_statement ∧ csilla_statement ∧ dora_statement) ∨
  (¬anna_statement ∧ bella_statement ∧ csilla_statement ∧ dora_statement)

-- Theorem to prove
theorem race_result_theorem :
  statements_condition →
  (¬dora_statement ∧ race_result Girl.Csilla = Position.First) := by
  sorry

end NUMINAMATH_CALUDE_race_result_theorem_l2215_221522


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2215_221543

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem complex_equation_solution :
  (3 - 4*I + z)*I = 2 + I → z = -2 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2215_221543


namespace NUMINAMATH_CALUDE_original_mean_calculation_l2215_221547

theorem original_mean_calculation (n : ℕ) (decrement : ℝ) (new_mean : ℝ) (h1 : n = 50) (h2 : decrement = 47) (h3 : new_mean = 153) :
  ∃ (original_mean : ℝ), original_mean * n = new_mean * n + decrement * n ∧ original_mean = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l2215_221547


namespace NUMINAMATH_CALUDE_water_left_in_bathtub_l2215_221516

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

end NUMINAMATH_CALUDE_water_left_in_bathtub_l2215_221516


namespace NUMINAMATH_CALUDE_min_value_a_l2215_221534

theorem min_value_a (a : ℝ) : 
  (∀ x > a, x + 4 / (x - a) ≥ 5) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_value_a_l2215_221534


namespace NUMINAMATH_CALUDE_addition_problems_l2215_221556

theorem addition_problems :
  (15 + (-22) = -7) ∧
  ((-13) + (-8) = -21) ∧
  ((-0.9) + 1.5 = 0.6) ∧
  (1/2 + (-2/3) = -1/6) := by
  sorry

end NUMINAMATH_CALUDE_addition_problems_l2215_221556


namespace NUMINAMATH_CALUDE_unique_digit_B_l2215_221574

-- Define the number as a function of B
def number (B : Nat) : Nat := 58709310 + B

-- Theorem statement
theorem unique_digit_B :
  ∀ B : Nat,
  B < 10 →
  (number B) % 2 = 0 →
  (number B) % 3 = 0 →
  (number B) % 4 = 0 →
  (number B) % 5 = 0 →
  (number B) % 6 = 0 →
  (number B) % 10 = 0 →
  B = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_B_l2215_221574


namespace NUMINAMATH_CALUDE_psychology_majors_percentage_l2215_221513

theorem psychology_majors_percentage (total_students : ℝ) (h1 : total_students > 0) : 
  let freshmen := 0.60 * total_students
  let liberal_arts_freshmen := 0.40 * freshmen
  let psych_majors := 0.048 * total_students
  psych_majors / liberal_arts_freshmen = 0.20 := by
sorry

end NUMINAMATH_CALUDE_psychology_majors_percentage_l2215_221513


namespace NUMINAMATH_CALUDE_rod_cutting_l2215_221583

theorem rod_cutting (rod_length : Real) (num_pieces : Nat) (piece_length_cm : Real) : 
  rod_length = 29.75 ∧ num_pieces = 35 → piece_length_cm = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2215_221583


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2016_l2215_221538

/-- An arithmetic sequence with first term a₁ and common difference d -/
def ArithmeticSequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def ArithmeticSum (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum_2016 :
  ∀ (d : ℤ),
  let a₁ : ℤ := -2016
  let S : ℕ → ℤ := ArithmeticSum a₁ d
  (S 20 / 20 - S 18 / 18 = 2) →
  (S 2016 = -2016) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2016_l2215_221538


namespace NUMINAMATH_CALUDE_powerjet_pump_l2215_221571

/-- The amount of water pumped in a given time -/
def water_pumped (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: A pump operating at 500 gallons per hour will pump 250 gallons in 30 minutes -/
theorem powerjet_pump (rate : ℝ) (time : ℝ) (h1 : rate = 500) (h2 : time = 1/2) : 
  water_pumped rate time = 250 := by
  sorry

end NUMINAMATH_CALUDE_powerjet_pump_l2215_221571


namespace NUMINAMATH_CALUDE_exam_marks_lost_l2215_221566

theorem exam_marks_lost (total_questions : ℕ) (marks_per_correct : ℕ) (total_marks : ℕ) (correct_answers : ℕ)
  (h1 : total_questions = 80)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 120)
  (h4 : correct_answers = 40) :
  (marks_per_correct * correct_answers - total_marks) / (total_questions - correct_answers) = 1 := by
sorry

end NUMINAMATH_CALUDE_exam_marks_lost_l2215_221566


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2215_221561

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, ∀ x : ℤ, x^2 + 2*(m-3) + 16 = (x + k)^2) → (m = -1 ∨ m = 7) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2215_221561


namespace NUMINAMATH_CALUDE_remainder_problem_l2215_221573

theorem remainder_problem (x : ℤ) (h1 : x % 62 = 7) (h2 : ∃ n : ℤ, (x + n) % 31 = 18) : 
  ∃ n : ℕ, n > 0 ∧ (x + n) % 31 = 18 ∧ ∀ m : ℕ, m > 0 ∧ (x + m) % 31 = 18 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2215_221573


namespace NUMINAMATH_CALUDE_root_difference_of_quadratic_l2215_221542

theorem root_difference_of_quadratic (r₁ r₂ : ℝ) : 
  r₁^2 - 9*r₁ + 14 = 0 → 
  r₂^2 - 9*r₂ + 14 = 0 → 
  r₁ + r₂ = r₁ * r₂ → 
  |r₁ - r₂| = 5 := by
sorry

end NUMINAMATH_CALUDE_root_difference_of_quadratic_l2215_221542
