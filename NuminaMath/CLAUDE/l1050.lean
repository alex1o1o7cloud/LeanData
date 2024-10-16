import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l1050_105031

theorem complex_equation_solution (z : ℂ) : z / (1 - I) = 3 + 2*I → z = 5 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1050_105031


namespace NUMINAMATH_CALUDE_optimal_allocation_l1050_105003

/-- Represents the allocation of workers to different part types. -/
structure WorkerAllocation :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents the hourly production rates for each part type. -/
structure ProductionRates :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents the assembly requirements for each part type. -/
structure AssemblyRequirements :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- The total number of workers available. -/
def totalWorkers : ℕ := 45

/-- The production rates for each part type. -/
def rates : ProductionRates :=
  { a := 30, b := 25, c := 20 }

/-- The assembly requirements for each part type. -/
def requirements : AssemblyRequirements :=
  { a := 3, b := 5, c := 4 }

/-- Checks if the given worker allocation satisfies all conditions. -/
def isValidAllocation (alloc : WorkerAllocation) : Prop :=
  alloc.a + alloc.b + alloc.c = totalWorkers ∧
  alloc.a * rates.a = requirements.a * (alloc.a * rates.a + alloc.b * rates.b + alloc.c * rates.c) / (requirements.a + requirements.b + requirements.c) ∧
  alloc.b * rates.b = requirements.b * (alloc.a * rates.a + alloc.b * rates.b + alloc.c * rates.c) / (requirements.a + requirements.b + requirements.c) ∧
  alloc.c * rates.c = requirements.c * (alloc.a * rates.a + alloc.b * rates.b + alloc.c * rates.c) / (requirements.a + requirements.b + requirements.c)

theorem optimal_allocation :
  isValidAllocation { a := 9, b := 18, c := 18 } :=
sorry

end NUMINAMATH_CALUDE_optimal_allocation_l1050_105003


namespace NUMINAMATH_CALUDE_complex_distance_problem_l1050_105004

theorem complex_distance_problem (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs (z - 1) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_problem_l1050_105004


namespace NUMINAMATH_CALUDE_two_ice_cream_cones_cost_l1050_105063

/-- The cost of an ice cream cone in cents -/
def ice_cream_cost : ℕ := 99

/-- The number of ice cream cones -/
def num_cones : ℕ := 2

/-- Theorem: The cost of 2 ice cream cones is 198 cents -/
theorem two_ice_cream_cones_cost : 
  ice_cream_cost * num_cones = 198 := by
  sorry

end NUMINAMATH_CALUDE_two_ice_cream_cones_cost_l1050_105063


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1050_105081

theorem consecutive_even_numbers_sum (n : ℕ) (sum : ℕ) (start : ℕ) : 
  (sum = (n / 2) * (2 * start + (n - 1) * 2)) →
  (start = 32) →
  (sum = 140) →
  (n = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1050_105081


namespace NUMINAMATH_CALUDE_simplify_tan_cot_expression_l1050_105026

theorem simplify_tan_cot_expression :
  let tan_60 := Real.sqrt 3
  let cot_60 := 1 / Real.sqrt 3
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_cot_expression_l1050_105026


namespace NUMINAMATH_CALUDE_new_average_weight_l1050_105071

theorem new_average_weight (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 6 →
  a = 78 →
  (b + c + d + e) / 4 = 79 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l1050_105071


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1050_105016

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_a4 : a 4 = 8)
  (h_q : q = -2) :
  a 7 = -64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1050_105016


namespace NUMINAMATH_CALUDE_complex_roots_circle_l1050_105077

theorem complex_roots_circle (z : ℂ) : 
  (z + 1)^6 = 243 * z^6 → Complex.abs (z - Complex.ofReal (1/8)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_circle_l1050_105077


namespace NUMINAMATH_CALUDE_light_travel_distance_l1050_105067

/-- The distance light travels in one year, in miles -/
def light_year_distance : ℕ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℕ := 200

/-- Theorem stating that light travels 1174 × 10^12 miles in 200 years -/
theorem light_travel_distance : 
  (light_year_distance * years : ℚ) = 1174 * (10^12 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1050_105067


namespace NUMINAMATH_CALUDE_operation_result_l1050_105052

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_result (star mul : Operation) 
  (h : apply_op star 12 2 / apply_op mul 9 3 = 2) :
  apply_op star 7 3 / apply_op mul 12 6 = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1050_105052


namespace NUMINAMATH_CALUDE_read_book_in_seven_weeks_l1050_105048

/-- The number of weeks required to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  (total_pages + pages_per_week - 1) / pages_per_week

/-- Theorem stating that it takes 7 weeks to read a 2100-page book at a rate of 300 pages per week. -/
theorem read_book_in_seven_weeks :
  let total_pages : ℕ := 2100
  let pages_per_day : ℕ := 100
  let days_per_week : ℕ := 3
  let pages_per_week : ℕ := pages_per_day * days_per_week
  weeks_to_read total_pages pages_per_week = 7 := by
  sorry

end NUMINAMATH_CALUDE_read_book_in_seven_weeks_l1050_105048


namespace NUMINAMATH_CALUDE_kyungsoo_string_shorter_l1050_105034

/-- Conversion factor from centimeters to millimeters -/
def cm_to_mm : ℚ := 10

/-- Length of Inhyuk's string in centimeters -/
def inhyuk_length_cm : ℚ := 97.5

/-- Base length of Kyungsoo's string in centimeters -/
def kyungsoo_base_length_cm : ℚ := 97

/-- Additional length of Kyungsoo's string in millimeters -/
def kyungsoo_additional_length_mm : ℚ := 3

/-- Theorem stating that Kyungsoo's string is shorter than Inhyuk's -/
theorem kyungsoo_string_shorter :
  kyungsoo_base_length_cm * cm_to_mm + kyungsoo_additional_length_mm <
  inhyuk_length_cm * cm_to_mm := by
  sorry

end NUMINAMATH_CALUDE_kyungsoo_string_shorter_l1050_105034


namespace NUMINAMATH_CALUDE_sin_315_degrees_l1050_105002

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l1050_105002


namespace NUMINAMATH_CALUDE_expansion_equality_l1050_105041

theorem expansion_equality (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l1050_105041


namespace NUMINAMATH_CALUDE_expression_evaluation_l1050_105042

theorem expression_evaluation : 
  Real.sqrt ((16^10 + 8^10 + 2^30) / (16^4 + 8^11 + 2^20)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1050_105042


namespace NUMINAMATH_CALUDE_last_remaining_number_l1050_105008

/-- Represents the marking process on a list of numbers -/
def MarkingProcess (n : ℕ) (skip : ℕ) (l : List ℕ) : List ℕ :=
  sorry

/-- Represents a single pass of the marking process -/
def SinglePass (n : ℕ) (skip : ℕ) (l : List ℕ) : List ℕ :=
  sorry

/-- Represents the entire process of marking and skipping -/
def FullProcess (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the last remaining number is 21 -/
theorem last_remaining_number : FullProcess 50 = 21 :=
  sorry

end NUMINAMATH_CALUDE_last_remaining_number_l1050_105008


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_l1050_105039

/-- Given an arithmetic sequence {a_n} with first term a₁ = 1 and common difference d = 3,
    prove that the value of n for which aₙ = 2005 is 669. -/
theorem arithmetic_sequence_2005 (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
  a 1 = 1 →                    -- First term is 1
  ∃ n : ℕ, a n = 2005 ∧ n = 669 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_l1050_105039


namespace NUMINAMATH_CALUDE_point_y_coordinate_l1050_105079

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the distance from a point to the focus
def distance_to_focus (x y : ℝ) : ℝ := 4

-- Define the y-coordinate of the directrix
def directrix_y : ℝ := -1

-- Theorem statement
theorem point_y_coordinate (x y : ℝ) :
  is_on_parabola x y →
  distance_to_focus x y = 4 →
  y = 3 := by sorry

end NUMINAMATH_CALUDE_point_y_coordinate_l1050_105079


namespace NUMINAMATH_CALUDE_january_salary_l1050_105045

/-- Given the average salaries for two sets of four months and the salary for May,
    prove that the salary for January is 4100. -/
theorem january_salary (jan feb mar apr may : ℕ)
  (h1 : (jan + feb + mar + apr) / 4 = 8000)
  (h2 : (feb + mar + apr + may) / 4 = 8600)
  (h3 : may = 6500) :
  jan = 4100 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l1050_105045


namespace NUMINAMATH_CALUDE_oranges_per_group_l1050_105020

def total_oranges : ℕ := 356
def orange_groups : ℕ := 178

theorem oranges_per_group : total_oranges / orange_groups = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_group_l1050_105020


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1050_105011

theorem purely_imaginary_complex_number (a : ℝ) : 
  (a^2 - a - 2 = 0) ∧ (|a - 1| - 1 ≠ 0) → a = -1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1050_105011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l1050_105033

/-- Given an arithmetic sequence {a_n} with a_1 = 1 and common difference d = 3,
    prove that the 100th term is equal to 298. -/
theorem arithmetic_sequence_100th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 1) → 
    (∀ n : ℕ, a (n + 1) - a n = 3) → 
    (a 100 = 298) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l1050_105033


namespace NUMINAMATH_CALUDE_class_weighted_average_l1050_105058

/-- Calculates the weighted average score for a class with three groups of students -/
theorem class_weighted_average (total_students : ℕ) 
  (group1_count : ℕ) (group1_avg : ℚ)
  (group2_count : ℕ) (group2_avg : ℚ)
  (group3_count : ℕ) (group3_avg : ℚ)
  (h1 : total_students = group1_count + group2_count + group3_count)
  (h2 : total_students = 30)
  (h3 : group1_count = 12)
  (h4 : group2_count = 10)
  (h5 : group3_count = 8)
  (h6 : group1_avg = 72 / 100)
  (h7 : group2_avg = 85 / 100)
  (h8 : group3_avg = 92 / 100) :
  (group1_count * group1_avg + 2 * group2_count * group2_avg + group3_count * group3_avg) / 
  (group1_count + 2 * group2_count + group3_count) = 825 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_class_weighted_average_l1050_105058


namespace NUMINAMATH_CALUDE_point_is_centroid_l1050_105027

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC in a real inner product space, if P is any point in the space and G satisfies PG = 1/3(PA + PB + PC), then G is the centroid of triangle ABC -/
theorem point_is_centroid (A B C P G : V) :
  (G - P) = (1 / 3 : ℝ) • ((A - P) + (B - P) + (C - P)) →
  G = (1 / 3 : ℝ) • (A + B + C) :=
sorry

end NUMINAMATH_CALUDE_point_is_centroid_l1050_105027


namespace NUMINAMATH_CALUDE_intersection_solution_set_l1050_105021

theorem intersection_solution_set (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ (x^2 - 2*x - 3 < 0 ∧ x^2 + x - 6 < 0)) → 
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_intersection_solution_set_l1050_105021


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_squares_l1050_105010

theorem unique_prime_sum_of_squares (p k x y a b : ℤ) : 
  Prime p → 
  p = 4 * k + 1 → 
  p = x^2 + y^2 → 
  p = a^2 + b^2 → 
  (x = a ∧ y = b) ∨ (x = -a ∧ y = -b) ∨ (x = b ∧ y = -a) ∨ (x = -b ∧ y = a) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_squares_l1050_105010


namespace NUMINAMATH_CALUDE_tara_quarters_l1050_105014

theorem tara_quarters : ∃! q : ℕ, 
  8 < q ∧ q < 80 ∧ 
  q % 4 = 2 ∧
  q % 6 = 2 ∧
  q % 8 = 2 ∧
  q = 26 := by sorry

end NUMINAMATH_CALUDE_tara_quarters_l1050_105014


namespace NUMINAMATH_CALUDE_vector_angle_cosine_l1050_105006

theorem vector_angle_cosine (α β : Real) (a b : ℝ × ℝ) :
  -π/2 < α ∧ α < 0 ∧ 0 < β ∧ β < π/2 →
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  ‖a - b‖ = Real.sqrt 10 / 5 →
  Real.cos α = 12/13 →
  Real.cos (α - β) = 4/5 ∧ Real.cos β = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_cosine_l1050_105006


namespace NUMINAMATH_CALUDE_target_line_correct_l1050_105074

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel_lines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The given line 2x - y + 1 = 0 -/
def given_line : Line2D :=
  { a := 2, b := -1, c := 1 }

/-- Point A (-1, 0) -/
def point_A : Point2D :=
  { x := -1, y := 0 }

/-- The line we need to prove -/
def target_line : Line2D :=
  { a := 2, b := -1, c := 2 }

theorem target_line_correct :
  point_on_line point_A target_line ∧
  parallel_lines target_line given_line := by
  sorry

end NUMINAMATH_CALUDE_target_line_correct_l1050_105074


namespace NUMINAMATH_CALUDE_company_women_count_l1050_105076

theorem company_women_count (total_workers : ℕ) 
  (h1 : total_workers / 3 = total_workers - (2 * total_workers / 3))  -- One-third don't have retirement plan
  (h2 : (total_workers / 3) / 5 = total_workers / 15)  -- 20% of workers without plan are women
  (h3 : (2 * total_workers / 3) * 2 / 5 = (2 * total_workers / 3) - ((2 * total_workers / 3) * 3 / 5))  -- 40% of workers with plan are men
  (h4 : 144 = (2 * total_workers / 3) * 2 / 5)  -- 144 men in the company
  : (total_workers / 15 + (2 * total_workers / 3) * 3 / 5 = 252) := by
  sorry

end NUMINAMATH_CALUDE_company_women_count_l1050_105076


namespace NUMINAMATH_CALUDE_complex_number_additive_inverse_l1050_105047

theorem complex_number_additive_inverse (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_additive_inverse_l1050_105047


namespace NUMINAMATH_CALUDE_final_stamp_count_l1050_105013

/-- Represents the number of stamps in Tom's collection -/
def stamps_collection (initial : ℕ) (mike_gift : ℕ) : ℕ → ℕ
  | harry_gift => initial + mike_gift + harry_gift

/-- Theorem: Tom's final stamp collection contains 3,061 stamps -/
theorem final_stamp_count :
  let initial := 3000
  let mike_gift := 17
  let harry_gift := 2 * mike_gift + 10
  stamps_collection initial mike_gift harry_gift = 3061 := by
  sorry

#check final_stamp_count

end NUMINAMATH_CALUDE_final_stamp_count_l1050_105013


namespace NUMINAMATH_CALUDE_circle_properties_l1050_105009

/-- Circle with center (6,8) and radius 10 -/
def Circle := {p : ℝ × ℝ | (p.1 - 6)^2 + (p.2 - 8)^2 = 100}

/-- The circle passes through the origin -/
axiom origin_on_circle : (0, 0) ∈ Circle

/-- P is the point where the circle intersects the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Q is the point on the circle with maximum y-coordinate -/
def Q : ℝ × ℝ := (6, 18)

/-- R is the point on the circle forming a right angle with P and Q -/
def R : ℝ × ℝ := (0, 16)

/-- S and T are the points on the circle forming 45-degree angles with P and Q -/
def S : ℝ × ℝ := (14, 14)
def T : ℝ × ℝ := (-2, 2)

theorem circle_properties :
  P ∈ Circle ∧
  Q ∈ Circle ∧
  R ∈ Circle ∧
  S ∈ Circle ∧
  T ∈ Circle ∧
  P.2 = 0 ∧
  ∀ p ∈ Circle, p.2 ≤ Q.2 ∧
  (R.1 - Q.1) * (P.1 - Q.1) + (R.2 - Q.2) * (P.2 - Q.2) = 0 ∧
  (S.1 - Q.1) * (P.1 - Q.1) + (S.2 - Q.2) * (P.2 - Q.2) =
    (T.1 - Q.1) * (P.1 - Q.1) + (T.2 - Q.2) * (P.2 - Q.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1050_105009


namespace NUMINAMATH_CALUDE_peters_horses_l1050_105059

/-- The number of horses Peter has -/
def num_horses : ℕ := 4

/-- The amount of oats each horse eats per feeding -/
def oats_per_feeding : ℕ := 4

/-- The number of oat feedings per day -/
def oat_feedings_per_day : ℕ := 2

/-- The amount of grain each horse eats per day -/
def grain_per_day : ℕ := 3

/-- The number of days Peter feeds his horses -/
def feeding_days : ℕ := 3

/-- The total amount of food Peter needs for all his horses for the given days -/
def total_food : ℕ := 132

theorem peters_horses :
  num_horses * (oats_per_feeding * oat_feedings_per_day + grain_per_day) * feeding_days = total_food :=
by sorry

end NUMINAMATH_CALUDE_peters_horses_l1050_105059


namespace NUMINAMATH_CALUDE_number_of_students_l1050_105078

theorem number_of_students (student_avg : ℝ) (teacher_age : ℝ) (new_avg : ℝ) :
  student_avg = 26 →
  teacher_age = 52 →
  new_avg = 27 →
  ∃ n : ℕ, (n : ℝ) * student_avg + teacher_age = (n + 1) * new_avg ∧ n = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1050_105078


namespace NUMINAMATH_CALUDE_coin_denominations_l1050_105029

/-- Represents a pair of coin denominations -/
structure CoinPair where
  a : ℕ+
  b : ℕ+

/-- Checks if an amount can be paid exactly with given coin denominations -/
def canPayExactly (pair : CoinPair) (amount : ℕ) : Prop :=
  ∃ (x y : ℕ), x * pair.a.val + y * pair.b.val = amount

/-- The main theorem stating the conditions and the result -/
theorem coin_denominations (pair : CoinPair) :
  (∀ n > 53, canPayExactly pair n) →
  ¬(canPayExactly pair 53) →
  (pair.a = 2 ∧ pair.b = 55) ∨ (pair.a = 3 ∧ pair.b = 28) :=
by sorry

end NUMINAMATH_CALUDE_coin_denominations_l1050_105029


namespace NUMINAMATH_CALUDE_simplify_expression_l1050_105064

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12 + 15*x + 18 = 33*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1050_105064


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1050_105082

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def swap_hundreds_units (n : ℕ) : ℕ :=
  (units_digit n) * 100 + (tens_digit n) * 10 + (hundreds_digit n)

theorem three_digit_number_problem :
  ∀ n : ℕ, is_three_digit_number n →
    (tens_digit n)^2 = (hundreds_digit n) * (units_digit n) →
    n - (swap_hundreds_units n) = 297 →
    (n = 300 ∨ n = 421) :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l1050_105082


namespace NUMINAMATH_CALUDE_cube_root_plus_square_root_l1050_105012

theorem cube_root_plus_square_root : 
  ∃ (x : ℝ), (x = 4 ∨ x = -8) ∧ x = ((-64 : ℝ)^(1/2))^(1/3) + (36 : ℝ)^(1/2) :=
sorry

end NUMINAMATH_CALUDE_cube_root_plus_square_root_l1050_105012


namespace NUMINAMATH_CALUDE_commercial_reduction_l1050_105036

def original_length : ℝ := 30
def reduction_percentage : ℝ := 0.30

theorem commercial_reduction :
  original_length * (1 - reduction_percentage) = 21 := by
  sorry

end NUMINAMATH_CALUDE_commercial_reduction_l1050_105036


namespace NUMINAMATH_CALUDE_scarves_per_box_l1050_105043

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_items : ℕ) : 
  num_boxes = 4 → 
  mittens_per_box = 6 → 
  total_items = 32 → 
  (total_items - num_boxes * mittens_per_box) / num_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_scarves_per_box_l1050_105043


namespace NUMINAMATH_CALUDE_election_probabilities_l1050_105005

structure Student where
  name : String
  prob_elected : ℚ

def A : Student := { name := "A", prob_elected := 4/5 }
def B : Student := { name := "B", prob_elected := 3/5 }
def C : Student := { name := "C", prob_elected := 7/10 }

def students : List Student := [A, B, C]

-- Probability that exactly one student is elected
def prob_exactly_one_elected (students : List Student) : ℚ :=
  sorry

-- Probability that at most two students are elected
def prob_at_most_two_elected (students : List Student) : ℚ :=
  sorry

theorem election_probabilities :
  (prob_exactly_one_elected students = 47/250) ∧
  (prob_at_most_two_elected students = 83/125) := by
  sorry

end NUMINAMATH_CALUDE_election_probabilities_l1050_105005


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l1050_105095

theorem smallest_k_with_remainders : ∃ k : ℕ, 
  k > 1 ∧
  k % 17 = 1 ∧
  k % 11 = 1 ∧
  k % 6 = 2 ∧
  ∀ m : ℕ, m > 1 → m % 17 = 1 → m % 11 = 1 → m % 6 = 2 → k ≤ m :=
by
  use 188
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l1050_105095


namespace NUMINAMATH_CALUDE_alteredLucas_53_mod_5_l1050_105068

def alteredLucas : ℕ → ℕ
  | 0 => 1
  | 1 => 4
  | n + 2 => alteredLucas n + alteredLucas (n + 1)

theorem alteredLucas_53_mod_5 : alteredLucas 52 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_alteredLucas_53_mod_5_l1050_105068


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_3_sqrt_3_over_4_l1050_105092

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition1 (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - t.b * t.c

def satisfies_condition2 (t : Triangle) : Prop :=
  t.a = Real.sqrt 7

def satisfies_condition3 (t : Triangle) : Prop :=
  t.c - t.b = 2

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfies_condition1 t) :
  t.A = π / 3 := by sorry

-- Theorem 2
theorem area_is_3_sqrt_3_over_4 (t : Triangle) 
  (h1 : satisfies_condition1 t) 
  (h2 : satisfies_condition2 t) 
  (h3 : satisfies_condition3 t) :
  (1/2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_3_sqrt_3_over_4_l1050_105092


namespace NUMINAMATH_CALUDE_blood_pressure_analysis_l1050_105055

def systolic_pressure : List ℝ := [151, 148, 140, 139, 140, 136, 140]
def diastolic_pressure : List ℝ := [90, 92, 88, 88, 90, 80, 88]

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem blood_pressure_analysis :
  (median systolic_pressure = 140) ∧
  (mode diastolic_pressure = 88) ∧
  (mean systolic_pressure = 142) ∧
  (variance diastolic_pressure = 88 / 7) :=
by sorry

end NUMINAMATH_CALUDE_blood_pressure_analysis_l1050_105055


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1050_105093

theorem cement_mixture_weight : 
  ∀ W : ℝ, 
    (5/14 + 3/10 + 2/9 + 1/7) * W + 2.5 = W → 
    W = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l1050_105093


namespace NUMINAMATH_CALUDE_max_carlson_jars_l1050_105060

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamState where
  carlson_weight : ℕ  -- Total weight of Carlson's jars
  baby_weight : ℕ    -- Total weight of Baby's jars
  carlson_jars : ℕ   -- Number of Carlson's jars

/-- Conditions of the jam problem -/
def jam_conditions (state : JamState) : Prop :=
  state.carlson_weight = 13 * state.baby_weight ∧
  ∃ (lightest : ℕ), 
    lightest > 0 ∧
    (state.carlson_weight - lightest) = 8 * (state.baby_weight + lightest)

/-- The theorem to be proved -/
theorem max_carlson_jars : 
  ∀ (state : JamState), 
    jam_conditions state → 
    state.carlson_jars ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l1050_105060


namespace NUMINAMATH_CALUDE_games_played_so_far_l1050_105017

/-- Proves that the number of games played so far is 15, given the conditions of the problem -/
theorem games_played_so_far 
  (total_games : ℕ) 
  (current_average : ℚ) 
  (goal_average : ℚ) 
  (required_average : ℚ) 
  (h1 : total_games = 20)
  (h2 : current_average = 26)
  (h3 : goal_average = 30)
  (h4 : required_average = 42)
  : ∃ (x : ℕ), x = 15 ∧ 
    x * current_average + (total_games - x) * required_average = total_games * goal_average := by
  sorry

end NUMINAMATH_CALUDE_games_played_so_far_l1050_105017


namespace NUMINAMATH_CALUDE_roots_derivative_sum_negative_l1050_105022

open Real

theorem roots_derivative_sum_negative (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
  (a * x₁ - log x₁ = 0) → (a * x₂ - log x₂ = 0) →
  (a - 1 / x₁) + (a - 1 / x₂) < 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_derivative_sum_negative_l1050_105022


namespace NUMINAMATH_CALUDE_sales_growth_equation_correct_l1050_105075

/-- Represents the sales growth scenario of a product over two years -/
structure SalesGrowth where
  initialSales : ℝ
  salesIncrease : ℝ
  growthRate : ℝ

/-- The equation for the sales growth scenario is correct -/
def isCorrectEquation (sg : SalesGrowth) : Prop :=
  20 * (1 + sg.growthRate)^2 - 20 = 3.12

/-- The given sales data matches the equation -/
theorem sales_growth_equation_correct (sg : SalesGrowth) 
  (h1 : sg.initialSales = 200000)
  (h2 : sg.salesIncrease = 31200) :
  isCorrectEquation sg := by
  sorry

end NUMINAMATH_CALUDE_sales_growth_equation_correct_l1050_105075


namespace NUMINAMATH_CALUDE_quadrilateral_cyclic_l1050_105073

-- Define the points
variable (A B C D P O B' D' X : EuclideanPlane)

-- Define the conditions
def is_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

def is_intersection (P : EuclideanPlane) (AB CD : Set EuclideanPlane) : Prop := sorry

def is_perpendicular_bisector_intersection (O : EuclideanPlane) (AB CD : Set EuclideanPlane) : Prop := sorry

def not_on_line (O : EuclideanPlane) (AB : Set EuclideanPlane) : Prop := sorry

def is_reflection (B' : EuclideanPlane) (B : EuclideanPlane) (OP : Set EuclideanPlane) : Prop := sorry

def meet_on_line (AB' CD' OP : Set EuclideanPlane) : Prop := sorry

def is_cyclic (A B C D : EuclideanPlane) : Prop := sorry

-- State the theorem
theorem quadrilateral_cyclic 
  (h1 : is_quadrilateral A B C D)
  (h2 : is_intersection P {A, B} {C, D})
  (h3 : is_perpendicular_bisector_intersection O {A, B} {C, D})
  (h4 : not_on_line O {A, B})
  (h5 : not_on_line O {C, D})
  (h6 : is_reflection B' B {O, P})
  (h7 : is_reflection D' D {O, P})
  (h8 : meet_on_line {A, B'} {C, D'} {O, P}) :
  is_cyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_cyclic_l1050_105073


namespace NUMINAMATH_CALUDE_stating_max_wickets_theorem_l1050_105049

/-- Represents the maximum number of wickets a bowler can take in one over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the number of overs bowled by the bowler in an innings -/
def overs_bowled : ℕ := 6

/-- Represents the number of players in a cricket team -/
def players_per_team : ℕ := 11

/-- Represents the maximum number of wickets that can be taken in an innings -/
def max_wickets_in_innings : ℕ := players_per_team - 1

/-- 
Theorem stating that the maximum number of wickets a bowler can take in an innings
is the minimum of the theoretical maximum (max_wickets_per_over * overs_bowled) 
and the actual maximum (max_wickets_in_innings)
-/
theorem max_wickets_theorem : 
  min (max_wickets_per_over * overs_bowled) max_wickets_in_innings = max_wickets_in_innings := by
  sorry

end NUMINAMATH_CALUDE_stating_max_wickets_theorem_l1050_105049


namespace NUMINAMATH_CALUDE_product_inequality_l1050_105087

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1050_105087


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_three_sqrt_two_over_two_l1050_105053

theorem sqrt_sum_equals_three_sqrt_two_over_two 
  (a b : ℝ) (h1 : a + b = -6) (h2 : a * b = 8) :
  Real.sqrt (b / a) + Real.sqrt (a / b) = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_three_sqrt_two_over_two_l1050_105053


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_3_l1050_105025

theorem complex_exp_13pi_over_3 : Complex.exp (13 * π * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_3_l1050_105025


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_negative_five_l1050_105015

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℤ := sorry

-- Define the general term of the expansion
def generalTerm (r : ℕ) : ℤ := (-1)^r * binomial 5 r

-- Define the exponent of x in the general term
def exponent (r : ℕ) : ℚ := (5 - 3*r) / 2

theorem coefficient_of_x_is_negative_five :
  ∃ (r : ℕ), exponent r = 1 ∧ generalTerm r = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_negative_five_l1050_105015


namespace NUMINAMATH_CALUDE_intersection_limit_l1050_105066

noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 8)

theorem intersection_limit :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 
    0 < |m| ∧ |m| < δ ∧ -8 < m ∧ m < 8 → 
    |((L (-m) - L m) / m) - 1 / (2 * Real.sqrt 2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_intersection_limit_l1050_105066


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l1050_105062

theorem opposite_of_negative_five :
  ∀ x : ℤ, ((-5 : ℤ) + x = 0) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l1050_105062


namespace NUMINAMATH_CALUDE_solve_for_k_l1050_105070

theorem solve_for_k : ∀ k : ℝ, (2 * k * 1 - (-7) = -1) → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l1050_105070


namespace NUMINAMATH_CALUDE_parking_probability_l1050_105097

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required -/
def required_spaces : ℕ := 3

/-- The probability of finding the required adjacent empty spaces -/
def probability_of_parking : ℚ := 12501 / 15504

theorem parking_probability :
  (total_spaces : ℕ) = 20 →
  (parked_cars : ℕ) = 15 →
  (required_spaces : ℕ) = 3 →
  probability_of_parking = 12501 / 15504 := by
  sorry

end NUMINAMATH_CALUDE_parking_probability_l1050_105097


namespace NUMINAMATH_CALUDE_division_result_l1050_105007

theorem division_result : 
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 := by sorry

end NUMINAMATH_CALUDE_division_result_l1050_105007


namespace NUMINAMATH_CALUDE_calculate_walking_speed_l1050_105050

/-- Given two people walking towards each other, this theorem calculates the speed of one person given the total distance, the speed of the other person, and the distance traveled by the first person. -/
theorem calculate_walking_speed 
  (total_distance : ℝ) 
  (brad_speed : ℝ) 
  (maxwell_distance : ℝ) 
  (h1 : total_distance = 40) 
  (h2 : brad_speed = 5) 
  (h3 : maxwell_distance = 15) : 
  ∃ (maxwell_speed : ℝ), maxwell_speed = 3 := by
  sorry

#check calculate_walking_speed

end NUMINAMATH_CALUDE_calculate_walking_speed_l1050_105050


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l1050_105030

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l1050_105030


namespace NUMINAMATH_CALUDE_dinner_cost_calculation_l1050_105001

/-- The total cost of dinner for Bret and his co-workers -/
def dinner_cost (num_people : ℕ) (main_meal_cost : ℚ) (num_appetizers : ℕ) (appetizer_cost : ℚ) (tip_percentage : ℚ) (rush_order_fee : ℚ) : ℚ :=
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := tip_percentage * subtotal
  subtotal + tip + rush_order_fee

/-- Theorem stating the total cost of dinner -/
theorem dinner_cost_calculation :
  dinner_cost 4 12 2 6 (1/5) 5 = 77 :=
by sorry

end NUMINAMATH_CALUDE_dinner_cost_calculation_l1050_105001


namespace NUMINAMATH_CALUDE_E27D6_divisibility_l1050_105019

/-- A number in the form E27D6 where E and D are single digits -/
def E27D6 (E D : ℕ) : ℕ := E * 10000 + 27000 + D * 10 + 6

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

theorem E27D6_divisibility (E D : ℕ) :
  is_single_digit E →
  is_single_digit D →
  E27D6 E D % 8 = 0 →
  ∃ (sum : ℕ), sum = D + E ∧ 1 ≤ sum ∧ sum ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_E27D6_divisibility_l1050_105019


namespace NUMINAMATH_CALUDE_nancy_antacids_per_month_l1050_105040

/-- Calculates the number of antacids Nancy takes per month -/
def antacids_per_month (indian_antacids : ℕ) (mexican_antacids : ℕ) (other_antacids : ℕ)
  (indian_freq : ℕ) (mexican_freq : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let days_per_week := 7
  let other_days := days_per_week - indian_freq - mexican_freq
  let weekly_antacids := indian_antacids * indian_freq + mexican_antacids * mexican_freq + other_antacids * other_days
  weekly_antacids * weeks_per_month

/-- Proves that Nancy takes 60 antacids per month given the specified conditions -/
theorem nancy_antacids_per_month :
  antacids_per_month 3 2 1 3 2 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_nancy_antacids_per_month_l1050_105040


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1050_105090

theorem largest_divisible_n : ∃ (n : ℕ), n = 180 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 20) ∣ (m^3 + 1000))) ∧ 
  ((n + 20) ∣ (n^3 + 1000)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1050_105090


namespace NUMINAMATH_CALUDE_parabola_shift_l1050_105038

/-- A parabola shifted 1 unit left and 4 units down -/
def shifted_parabola (x : ℝ) : ℝ := 3 * (x + 1)^2 - 4

/-- The original parabola -/
def original_parabola (x : ℝ) : ℝ := 3 * x^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) - 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1050_105038


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1050_105051

theorem smallest_number_of_eggs : ∀ (n : ℕ),
  n > 200 ∧
  ∃ (c : ℕ), n = 15 * c - 3 ∧
  c ≥ 14 →
  n ≥ 207 ∧
  ∃ (m : ℕ), m = 207 ∧ m > 200 ∧ ∃ (d : ℕ), m = 15 * d - 3 ∧ d ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1050_105051


namespace NUMINAMATH_CALUDE_black_midwest_percentage_is_31_l1050_105098

/-- Represents the population data for different ethnic groups in different regions --/
structure PopulationData :=
  (ne_white : ℕ) (mw_white : ℕ) (south_white : ℕ) (west_white : ℕ)
  (ne_black : ℕ) (mw_black : ℕ) (south_black : ℕ) (west_black : ℕ)
  (ne_asian : ℕ) (mw_asian : ℕ) (south_asian : ℕ) (west_asian : ℕ)
  (ne_hispanic : ℕ) (mw_hispanic : ℕ) (south_hispanic : ℕ) (west_hispanic : ℕ)

/-- Calculates the percentage of Black population in the Midwest --/
def black_midwest_percentage (data : PopulationData) : ℚ :=
  let total_black := data.ne_black + data.mw_black + data.south_black + data.west_black
  (data.mw_black : ℚ) / total_black * 100

/-- Rounds a rational number to the nearest integer --/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- The main theorem stating that the rounded percentage of Black population in the Midwest is 31% --/
theorem black_midwest_percentage_is_31 (data : PopulationData) 
  (h : data = { ne_white := 45, mw_white := 55, south_white := 60, west_white := 40,
                ne_black := 6, mw_black := 12, south_black := 18, west_black := 3,
                ne_asian := 2, mw_asian := 2, south_asian := 2, west_asian := 5,
                ne_hispanic := 2, mw_hispanic := 3, south_hispanic := 4, west_hispanic := 6 }) :
  round_to_nearest (black_midwest_percentage data) = 31 := by
  sorry

end NUMINAMATH_CALUDE_black_midwest_percentage_is_31_l1050_105098


namespace NUMINAMATH_CALUDE_b_oxen_count_main_theorem_l1050_105028

/-- Represents the number of oxen and months for each person --/
structure Grazing :=
  (oxen : ℕ)
  (months : ℕ)

/-- Calculates the total grazing cost --/
def total_cost (a b c : Grazing) (cost_per_ox_month : ℚ) : ℚ :=
  (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months : ℚ) * cost_per_ox_month

/-- Theorem: Given the conditions, b put 12 oxen for grazing --/
theorem b_oxen_count (total_rent : ℚ) (c_rent : ℚ) : ℕ :=
  let a : Grazing := ⟨10, 7⟩
  let b : Grazing := ⟨12, 5⟩  -- We claim b put 12 oxen
  let c : Grazing := ⟨15, 3⟩
  let cost_per_ox_month : ℚ := c_rent / (c.oxen * c.months)
  have h1 : total_cost a b c cost_per_ox_month = total_rent := by sorry
  have h2 : c.oxen * c.months * cost_per_ox_month = c_rent := by sorry
  b.oxen

/-- The main theorem stating that given the conditions, b put 12 oxen for grazing --/
theorem main_theorem : b_oxen_count 140 36 = 12 := by sorry

end NUMINAMATH_CALUDE_b_oxen_count_main_theorem_l1050_105028


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_sqrt_three_over_two_l1050_105065

theorem sin_cos_difference_equals_sqrt_three_over_two :
  Real.sin (5 * π / 180) * Real.cos (55 * π / 180) -
  Real.cos (175 * π / 180) * Real.sin (55 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_sqrt_three_over_two_l1050_105065


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_squared_l1050_105096

theorem min_value_of_quadratic_squared (x : ℝ) : 
  ∃ (y : ℝ), (x^2 + 6*x + 2)^2 ≥ 0 ∧ (y^2 + 6*y + 2)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_squared_l1050_105096


namespace NUMINAMATH_CALUDE_somu_age_problem_l1050_105084

/-- Somu's age problem -/
theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 7 = (father_age - 7) / 5 →
  somu_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_somu_age_problem_l1050_105084


namespace NUMINAMATH_CALUDE_log_equation_solution_l1050_105024

theorem log_equation_solution (x : ℝ) :
  x > 0 → (4 * Real.log x / Real.log 3 = Real.log (6 * x) / Real.log 3) ↔ x = (6 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1050_105024


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1050_105000

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1050_105000


namespace NUMINAMATH_CALUDE_cos_sin_18_equality_l1050_105037

theorem cos_sin_18_equality :
  let cos_18 : ℝ := (Real.sqrt 5 + 1) / 4
  let sin_18 : ℝ := (Real.sqrt 5 - 1) / 4
  4 * cos_18^2 - 1 = 1 / (4 * sin_18^2) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_18_equality_l1050_105037


namespace NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l1050_105018

theorem quadratic_coefficient_of_equation (x : ℝ) : 
  (2*x + 1) * (3*x - 2) = x^2 + 2 → 
  ∃ a b c : ℝ, a = 5 ∧ a*x^2 + b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l1050_105018


namespace NUMINAMATH_CALUDE_D_l1050_105056

def D' : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => D' (n + 2) + D' (n + 1) + D' n

theorem D'_parity_2024_2025_2026 :
  Even (D' 2024) ∧ Odd (D' 2025) ∧ Odd (D' 2026) :=
by
  sorry

end NUMINAMATH_CALUDE_D_l1050_105056


namespace NUMINAMATH_CALUDE_parabola_directrix_l1050_105089

/-- The parabola is defined by the equation x^2 = 4y -/
def parabola_eq (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix of a parabola that opens upward is given by y = -p, where p is the distance from the vertex to the focus -/
def directrix_eq (y p : ℝ) : Prop := y = -p

/-- For a parabola in the form x^2 = 4py, p represents the distance from the vertex to the focus -/
def p_value (p : ℝ) : Prop := 4*p = 4

theorem parabola_directrix :
  ∃ p, p_value p ∧ ∀ x y, parabola_eq x y → directrix_eq (-1) p :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1050_105089


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l1050_105099

/-- Proves that for a rectangular hall with width being half the length and area 288 sq. m, 
    the difference between length and width is 12 meters -/
theorem rectangular_hall_dimension_difference 
  (length width : ℝ) 
  (h1 : width = length / 2) 
  (h2 : length * width = 288) : 
  length - width = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l1050_105099


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l1050_105061

/-- The amount of oil leaked before engineers started fixing the pipe -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked while engineers were working -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem oil_leak_calculation :
  total_oil_leaked = 11687 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l1050_105061


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1050_105046

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≥ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1050_105046


namespace NUMINAMATH_CALUDE_tom_has_24_blue_marbles_l1050_105083

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The difference between Jason's and Tom's blue marbles -/
def marble_difference : ℕ := 20

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := jason_blue_marbles - marble_difference

theorem tom_has_24_blue_marbles : tom_blue_marbles = 24 := by
  sorry

end NUMINAMATH_CALUDE_tom_has_24_blue_marbles_l1050_105083


namespace NUMINAMATH_CALUDE_food_bank_remaining_l1050_105094

/-- Calculates the amount of food remaining in the food bank after four weeks of donations and distributions. -/
theorem food_bank_remaining (week1_donation : ℝ) (week2_factor : ℝ) (week3_increase : ℝ) (week4_decrease : ℝ)
  (week1_given_out : ℝ) (week2_given_out : ℝ) (week3_given_out : ℝ) (week4_given_out : ℝ) :
  week1_donation = 40 →
  week2_factor = 1.5 →
  week3_increase = 1.25 →
  week4_decrease = 0.9 →
  week1_given_out = 0.6 →
  week2_given_out = 0.7 →
  week3_given_out = 0.8 →
  week4_given_out = 0.5 →
  let week2_donation := week1_donation * week2_factor
  let week3_donation := week2_donation * week3_increase
  let week4_donation := week3_donation * week4_decrease
  let week1_remaining := week1_donation * (1 - week1_given_out)
  let week2_remaining := week2_donation * (1 - week2_given_out)
  let week3_remaining := week3_donation * (1 - week3_given_out)
  let week4_remaining := week4_donation * (1 - week4_given_out)
  week1_remaining + week2_remaining + week3_remaining + week4_remaining = 82.75 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_remaining_l1050_105094


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1050_105044

theorem quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → x^2 + 2*a*x + 1 ≥ 0) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1050_105044


namespace NUMINAMATH_CALUDE_platform_length_l1050_105072

/-- Given a train of length 1200 m that takes 120 sec to pass a tree and 150 sec to pass a platform, 
    prove that the length of the platform is 300 m. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_tree : ℝ) 
  (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 150) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 300 := by
sorry


end NUMINAMATH_CALUDE_platform_length_l1050_105072


namespace NUMINAMATH_CALUDE_or_implies_and_implies_not_equivalent_l1050_105023

theorem or_implies_and_implies_not_equivalent :
  ¬(∀ (A B C : Prop), ((A ∨ B) → C) ↔ ((A ∧ B) → C)) := by
sorry

end NUMINAMATH_CALUDE_or_implies_and_implies_not_equivalent_l1050_105023


namespace NUMINAMATH_CALUDE_johns_cost_per_minute_l1050_105088

/-- Calculates the cost per minute for long distance calls given the total bill, monthly fee, and minutes used. -/
def cost_per_minute (total_bill : ℚ) (monthly_fee : ℚ) (minutes_used : ℚ) : ℚ :=
  (total_bill - monthly_fee) / minutes_used

/-- Proves that the cost per minute for John's long distance calls is $0.25 given the specified conditions. -/
theorem johns_cost_per_minute :
  let total_bill : ℚ := 12.02
  let monthly_fee : ℚ := 5
  let minutes_used : ℚ := 28.08
  cost_per_minute total_bill monthly_fee minutes_used = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_johns_cost_per_minute_l1050_105088


namespace NUMINAMATH_CALUDE_apartment_building_floors_l1050_105035

/-- Represents an apartment building with the given specifications. -/
structure ApartmentBuilding where
  floors : ℕ
  apartments_per_floor : ℕ
  people_per_apartment : ℕ
  total_people : ℕ

/-- Calculates the number of people on a full floor. -/
def people_on_full_floor (building : ApartmentBuilding) : ℕ :=
  building.apartments_per_floor * building.people_per_apartment

/-- Calculates the number of people on a half-capacity floor. -/
def people_on_half_capacity_floor (building : ApartmentBuilding) : ℕ :=
  (building.apartments_per_floor / 2) * building.people_per_apartment

/-- Theorem stating that given the conditions, the apartment building has 12 floors. -/
theorem apartment_building_floors
  (building : ApartmentBuilding)
  (h1 : building.apartments_per_floor = 10)
  (h2 : building.people_per_apartment = 4)
  (h3 : building.total_people = 360)
  (h4 : building.total_people = 
    (building.floors / 2 * people_on_full_floor building) + 
    (building.floors / 2 * people_on_half_capacity_floor building)) :
  building.floors = 12 := by
  sorry


end NUMINAMATH_CALUDE_apartment_building_floors_l1050_105035


namespace NUMINAMATH_CALUDE_expression_value_l1050_105086

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 6 * y) / (x - 2 * y) = 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1050_105086


namespace NUMINAMATH_CALUDE_platform_length_l1050_105080

/-- Given a train and platform with specific properties, prove the platform length --/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 36) :
  ∃ platform_length : ℝ,
    platform_length = 25 ∧
    (train_length + platform_length) / time_cross_platform = train_length / time_cross_pole :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1050_105080


namespace NUMINAMATH_CALUDE_pole_area_after_cuts_l1050_105032

/-- The area of a rectangular pole after two cuts -/
theorem pole_area_after_cuts (original_length original_width : ℝ)
  (length_cut_percentage width_cut_percentage : ℝ) :
  original_length = 20 →
  original_width = 2 →
  length_cut_percentage = 0.3 →
  width_cut_percentage = 0.25 →
  let new_length := original_length * (1 - length_cut_percentage)
  let new_width := original_width * (1 - width_cut_percentage)
  new_length * new_width = 21 := by
  sorry

end NUMINAMATH_CALUDE_pole_area_after_cuts_l1050_105032


namespace NUMINAMATH_CALUDE_altitude_inradius_equality_l1050_105085

/-- Triangle ABC with side lengths a, b, c, altitudes h_a, h_b, h_c, and inradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  r : ℝ

/-- The theorem states that the sum of altitudes equals 9 times the inradius 
    if and only if the triangle is equilateral -/
theorem altitude_inradius_equality (t : Triangle) : 
  t.h_a + t.h_b + t.h_c = 9 * t.r ↔ t.a = t.b ∧ t.b = t.c :=
sorry

end NUMINAMATH_CALUDE_altitude_inradius_equality_l1050_105085


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1050_105069

/-- The lateral area of a cone with height 3 and slant height 5 is 20π. -/
theorem cone_lateral_area (h : ℝ) (s : ℝ) (r : ℝ) :
  h = 3 →
  s = 5 →
  r^2 + h^2 = s^2 →
  (1/2 : ℝ) * (2 * π * r) * s = 20 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1050_105069


namespace NUMINAMATH_CALUDE_count_eight_digit_cyclic_fixed_points_l1050_105054

def is_eight_digit (n : ℕ) : Prop := 10^7 ≤ n ∧ n < 10^8

def last_digit (n : ℕ) : ℕ := n % 10

def cyclic_permutation (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  (n % 10) * 10^(d-1) + n / 10

def iterative_permutation (n : ℕ) (k : ℕ) : ℕ :=
  Nat.iterate cyclic_permutation k n

theorem count_eight_digit_cyclic_fixed_points :
  (∃ (S : Finset ℕ), (∀ a ∈ S, is_eight_digit a ∧ last_digit a ≠ 0 ∧
    iterative_permutation a 4 = a) ∧ S.card = 9^4) := by sorry

end NUMINAMATH_CALUDE_count_eight_digit_cyclic_fixed_points_l1050_105054


namespace NUMINAMATH_CALUDE_equation_solution_l1050_105091

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 12 ∧ x = 168 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1050_105091


namespace NUMINAMATH_CALUDE_total_amount_is_1195_l1050_105057

/-- The total amount paid for grapes and mangoes -/
def total_amount (grapes_quantity : ℕ) (grapes_rate : ℕ) (mangoes_quantity : ℕ) (mangoes_rate : ℕ) : ℕ :=
  grapes_quantity * grapes_rate + mangoes_quantity * mangoes_rate

/-- Theorem stating that the total amount paid is 1195 -/
theorem total_amount_is_1195 :
  total_amount 10 70 9 55 = 1195 := by
  sorry

#eval total_amount 10 70 9 55

end NUMINAMATH_CALUDE_total_amount_is_1195_l1050_105057
