import Mathlib

namespace NUMINAMATH_CALUDE_union_perimeter_bound_l355_35593

/-- A disc in a 2D plane -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- A set of discs satisfying the problem conditions -/
structure DiscSet where
  discs : Set Disc
  segment_length : ℝ
  centers_on_segment : ∀ d ∈ discs, ∃ x : ℝ, d.center = (x, 0) ∧ 0 ≤ x ∧ x ≤ segment_length
  radii_bounded : ∀ d ∈ discs, d.radius ≤ 1

/-- The perimeter of the union of discs -/
noncomputable def union_perimeter (ds : DiscSet) : ℝ := sorry

/-- The main theorem -/
theorem union_perimeter_bound (ds : DiscSet) :
  union_perimeter ds ≤ 4 * ds.segment_length + 8 := by
  sorry

end NUMINAMATH_CALUDE_union_perimeter_bound_l355_35593


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l355_35569

/-- Given a triangle with area 30, a side of length 12, and a median to that side of length 8,
    the sine of the angle between the side and the median is 5/8. -/
theorem triangle_sine_theorem (A : ℝ) (a m θ : ℝ) 
    (h_area : A = 30)
    (h_side : a = 12)
    (h_median : m = 8)
    (h_angle : 0 < θ ∧ θ < π / 2)
    (h_triangle_area : A = 1/2 * a * m * Real.sin θ) : 
  Real.sin θ = 5/8 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l355_35569


namespace NUMINAMATH_CALUDE_perfect_square_condition_l355_35556

theorem perfect_square_condition (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l355_35556


namespace NUMINAMATH_CALUDE_derivative_of_sine_function_l355_35557

open Real

theorem derivative_of_sine_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => 3 * sin (2 * x - π / 6)
  deriv y x = 6 * cos (2 * x - π / 6) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_sine_function_l355_35557


namespace NUMINAMATH_CALUDE_scale_division_theorem_l355_35564

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 10 * 12 + 5

/-- The number of equal parts the scale is divided into -/
def num_parts : ℕ := 5

/-- The length of each part in inches -/
def part_length : ℕ := scale_length / num_parts

/-- Converts inches to feet and remaining inches -/
def inches_to_feet_and_inches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

theorem scale_division_theorem :
  inches_to_feet_and_inches part_length = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_scale_division_theorem_l355_35564


namespace NUMINAMATH_CALUDE_solve_for_y_l355_35546

theorem solve_for_y : ∃ y : ℝ, (2 * y) / 5 = 10 ∧ y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l355_35546


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l355_35506

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 15*x + 54 = (x + a) * (x + b)) →
  (∀ x, x^2 - 17*x + 72 = (x - b) * (x - c)) →
  a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l355_35506


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l355_35562

theorem sin_plus_cos_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2))
  (h2 : Real.tan (α - 7 * π) = -3 / 4) : 
  Real.sin α + Real.cos α = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l355_35562


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l355_35505

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one (n m : ℕ) : x n = y m → n = 0 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l355_35505


namespace NUMINAMATH_CALUDE_airport_distance_l355_35548

/-- The distance from David's home to the airport in miles. -/
def distance_to_airport : ℝ := 160

/-- David's initial speed in miles per hour. -/
def initial_speed : ℝ := 40

/-- The increase in David's speed in miles per hour. -/
def speed_increase : ℝ := 20

/-- The time in hours David would be late if he continued at the initial speed. -/
def time_late : ℝ := 0.75

/-- The time in hours David arrives early with increased speed. -/
def time_early : ℝ := 0.25

/-- Theorem stating that the distance to the airport is 160 miles. -/
theorem airport_distance : 
  ∃ (t : ℝ), 
    distance_to_airport = initial_speed * (t + time_late) ∧
    distance_to_airport - initial_speed = (initial_speed + speed_increase) * (t - 1 - time_early) :=
by
  sorry


end NUMINAMATH_CALUDE_airport_distance_l355_35548


namespace NUMINAMATH_CALUDE_walking_time_calculation_walk_two_miles_time_l355_35599

/-- Calculates the time taken to walk a given distance at a constant pace -/
theorem walking_time_calculation (distance : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  (distance > 0) → (total_distance > 0) → (total_time > 0) →
  (total_distance / total_time * total_time = total_distance) →
  (distance / (total_distance / total_time) = distance * total_time / total_distance) := by
  sorry

/-- Proves that walking 2 miles takes 1 hour given the conditions -/
theorem walk_two_miles_time :
  ∃ (pace : ℝ),
    (2 : ℝ) / pace = 1 ∧
    pace * 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_walk_two_miles_time_l355_35599


namespace NUMINAMATH_CALUDE_office_gender_ratio_l355_35501

/-- Given an office with 60 employees, if a meeting of 4 men and 6 women
    reduces the number of women on the office floor by 20%,
    then the ratio of men to women in the office is 1:1. -/
theorem office_gender_ratio
  (total_employees : ℕ)
  (meeting_men : ℕ)
  (meeting_women : ℕ)
  (women_reduction_percent : ℚ)
  (h1 : total_employees = 60)
  (h2 : meeting_men = 4)
  (h3 : meeting_women = 6)
  (h4 : women_reduction_percent = 1/5)
  : (total_employees / 2 : ℚ) = (total_employees - (total_employees / 2) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_office_gender_ratio_l355_35501


namespace NUMINAMATH_CALUDE_blue_balls_count_l355_35551

def probability_two_red (red green blue : ℕ) : ℚ :=
  (red.choose 2 : ℚ) / ((red + green + blue).choose 2 : ℚ)

theorem blue_balls_count (red green : ℕ) (prob : ℚ) :
  red = 7 →
  green = 4 →
  probability_two_red red green (blue : ℕ) = (175 : ℚ) / 1000 →
  blue = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l355_35551


namespace NUMINAMATH_CALUDE_cookies_left_l355_35519

theorem cookies_left (initial_cookies eaten_cookies : ℕ) 
  (h1 : initial_cookies = 93)
  (h2 : eaten_cookies = 15) :
  initial_cookies - eaten_cookies = 78 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l355_35519


namespace NUMINAMATH_CALUDE_negation_equivalence_l355_35542

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x < x ^ 2) ↔ (∀ x : ℝ, (2 : ℝ) ^ x ≥ x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l355_35542


namespace NUMINAMATH_CALUDE_smallest_base_for_100_l355_35580

theorem smallest_base_for_100 :
  ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 100 → 100 ≥ x^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_l355_35580


namespace NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_l355_35504

/-- A function f: ℝ → ℝ is periodic with period T if f(x + T) = f(x) for all x ∈ ℝ -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem derivative_of_periodic_is_periodic (f : ℝ → ℝ) (f' : ℝ → ℝ) (T : ℝ) 
    (hf : Differentiable ℝ f) (hperiodic : IsPeriodic f T) :
    IsPeriodic f' T :=
  sorry

end NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_l355_35504


namespace NUMINAMATH_CALUDE_condition_one_implies_right_triangle_condition_two_implies_right_triangle_condition_three_not_implies_right_triangle_condition_four_implies_right_triangle_l355_35518

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_of_angles : A + B + C = 180

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Condition 1
theorem condition_one_implies_right_triangle (t : Triangle) 
  (h : t.A + t.B = t.C) : is_right_triangle t :=
sorry

-- Condition 2
theorem condition_two_implies_right_triangle (t : Triangle) 
  (h : ∃ (k : Real), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k) : is_right_triangle t :=
sorry

-- Condition 3
theorem condition_three_not_implies_right_triangle : ∃ (t : Triangle), 
  (t.A = t.B ∧ t.B = t.C) ∧ ¬(is_right_triangle t) :=
sorry

-- Condition 4
theorem condition_four_implies_right_triangle (t : Triangle) 
  (h : t.A = 90 - t.B) : is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_one_implies_right_triangle_condition_two_implies_right_triangle_condition_three_not_implies_right_triangle_condition_four_implies_right_triangle_l355_35518


namespace NUMINAMATH_CALUDE_julia_rental_cost_l355_35587

/-- Calculates the total cost of a car rental --/
def calculateRentalCost (dailyRate : ℝ) (mileageRate : ℝ) (days : ℝ) (miles : ℝ) : ℝ :=
  dailyRate * days + mileageRate * miles

/-- Proves that Julia's car rental cost is $46.12 --/
theorem julia_rental_cost :
  let dailyRate : ℝ := 29
  let mileageRate : ℝ := 0.08
  let days : ℝ := 1
  let miles : ℝ := 214
  calculateRentalCost dailyRate mileageRate days miles = 46.12 := by
  sorry

end NUMINAMATH_CALUDE_julia_rental_cost_l355_35587


namespace NUMINAMATH_CALUDE_function_properties_l355_35566

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- State the theorem
theorem function_properties :
  -- 1. The maximum value of f is 4
  (∃ (x : ℝ), f x = 4) ∧ (∀ (x : ℝ), f x ≤ 4) ∧
  -- 2. The solution set of f(x) < 1
  (∀ (x : ℝ), f x < 1 ↔ (x < -4 ∨ x > 0)) ∧
  -- 3. The maximum value of ab + bc given the constraints
  (∀ (a b c : ℝ), a > 0 → b > 0 → a^2 + 2*b^2 + c^2 = 4 → ab + bc ≤ 2) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 + c^2 = 4 ∧ ab + bc = 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l355_35566


namespace NUMINAMATH_CALUDE_distance_between_vertices_l355_35539

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) :
  let f1 := fun x : ℝ => x^2 + a*x + b
  let f2 := fun x : ℝ => x^2 + d*x + e
  let vertex1 := (-a/2, f1 (-a/2))
  let vertex2 := (-d/2, f2 (-d/2))
  a = -4 ∧ b = 7 ∧ d = 6 ∧ e = 20 →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l355_35539


namespace NUMINAMATH_CALUDE_inequality_proof_l355_35543

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2*x^6*y^6 ≤ π/2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l355_35543


namespace NUMINAMATH_CALUDE_weighted_average_combined_class_l355_35550

/-- Given two classes of students, prove that the weighted average of the combined class
    is equal to the sum of the products of each class's student count and average mark,
    divided by the total number of students. -/
theorem weighted_average_combined_class
  (n₁ : ℕ) (n₂ : ℕ) (x₁ : ℚ) (x₂ : ℚ)
  (h₁ : n₁ = 58)
  (h₂ : n₂ = 52)
  (h₃ : x₁ = 67)
  (h₄ : x₂ = 82) :
  (n₁ * x₁ + n₂ * x₂) / (n₁ + n₂ : ℚ) = (58 * 67 + 52 * 82) / (58 + 52 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_combined_class_l355_35550


namespace NUMINAMATH_CALUDE_no_solution_exists_l355_35512

theorem no_solution_exists : ¬∃ (a b : ℕ+), a^2 - 23 = b^11 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l355_35512


namespace NUMINAMATH_CALUDE_triangle_existence_l355_35530

/-- Given an angle and two segments representing differences between sides,
    prove the existence of a triangle with these properties. -/
theorem triangle_existence (A : Real) (d e : ℝ) : ∃ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- positive side lengths
  a - c = d ∧              -- given difference d
  b - c = e ∧              -- given difference e
  0 < A ∧ A < π ∧          -- valid angle measure
  -- The angle A is the smallest in the triangle
  A ≤ Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧
  A ≤ Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) :=
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l355_35530


namespace NUMINAMATH_CALUDE_max_value_abc_l355_35503

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683/472392 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l355_35503


namespace NUMINAMATH_CALUDE_system_equation_solution_l355_35570

theorem system_equation_solution (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l355_35570


namespace NUMINAMATH_CALUDE_range_of_a_l355_35585

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x > 0, Monotone (fun x => Real.log x / Real.log a)

def Q (a : ℝ) : Prop := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : P a ∨ Q a) 
  (h2 : ¬(P a ∧ Q a)) : 
  a > 2 ∨ (-2 < a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l355_35585


namespace NUMINAMATH_CALUDE_inequality_equivalence_l355_35558

theorem inequality_equivalence (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l355_35558


namespace NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_l355_35538

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (a : ℝ) (x : ℝ) : ℝ := 2 * |x| + a

-- Theorem for part (1)
theorem solution_set_when_a_neg_one :
  {x : ℝ | f x ≤ g (-1) x} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ ≥ (1/2) * g a x₀) → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_l355_35538


namespace NUMINAMATH_CALUDE_additional_sticks_needed_l355_35584

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  large_rectangle : Rectangle
  total_sticks : ℕ
  num_small_rectangles : ℕ
  small_rectangle_types : List Rectangle

/-- The main theorem statement -/
theorem additional_sticks_needed 
  (setup : ProblemSetup)
  (h1 : setup.large_rectangle = ⟨8, 12⟩)
  (h2 : setup.total_sticks = 40)
  (h3 : setup.num_small_rectangles = 40)
  (h4 : setup.small_rectangle_types = [⟨1, 2⟩, ⟨1, 3⟩])
  : ∃ (additional_sticks : ℕ), additional_sticks = 116 ∧
    ∃ (small_rectangles : List Rectangle),
      small_rectangles.length = setup.num_small_rectangles ∧
      (∀ r ∈ small_rectangles, r ∈ setup.small_rectangle_types) ∧
      (small_rectangles.map (λ r => r.width * r.height)).sum = 
        setup.large_rectangle.width * setup.large_rectangle.height :=
by
  sorry


end NUMINAMATH_CALUDE_additional_sticks_needed_l355_35584


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l355_35559

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (p : Point) 
  (l1 : Line) 
  (l2 : Line) : 
  p.liesOn l2 ∧ l2.isParallelTo l1 → 
  l2 = Line.mk 1 (-2) 7 :=
by
  sorry

#check line_through_point_parallel_to_line 
  (Point.mk (-1) 3) 
  (Line.mk 1 (-2) 3) 
  (Line.mk 1 (-2) 7)

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l355_35559


namespace NUMINAMATH_CALUDE_eleventh_grade_sample_l355_35552

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def grade_ratio : Fin 3 → ℕ
| 0 => 3  -- 10th grade
| 1 => 3  -- 11th grade
| 2 => 4  -- 12th grade

/-- The total sample size -/
def sample_size : ℕ := 50

/-- Calculates the number of students to be sampled from a specific grade -/
def students_to_sample (grade : Fin 3) : ℕ :=
  (grade_ratio grade * sample_size) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2)

theorem eleventh_grade_sample :
  students_to_sample 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_grade_sample_l355_35552


namespace NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l355_35536

/-- Proves that the percentage of alcohol in solution y is 30% -/
theorem alcohol_percentage_solution_y :
  let solution_x_volume : ℝ := 200
  let solution_y_volume : ℝ := 600
  let solution_x_percentage : ℝ := 10
  let final_mixture_percentage : ℝ := 25
  let total_volume : ℝ := solution_x_volume + solution_y_volume
  let solution_y_percentage : ℝ := 
    ((final_mixture_percentage / 100) * total_volume - (solution_x_percentage / 100) * solution_x_volume) / 
    solution_y_volume * 100
  solution_y_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l355_35536


namespace NUMINAMATH_CALUDE_quadratic_solution_unique_l355_35533

theorem quadratic_solution_unique (x : ℝ) :
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0 → x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_unique_l355_35533


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l355_35529

theorem largest_multiple_of_15_under_500 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 ∧ 5 ∣ n → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l355_35529


namespace NUMINAMATH_CALUDE_extra_fruits_theorem_l355_35553

/-- Represents the quantities of fruits ordered and wanted --/
structure FruitQuantities where
  redAppleOrdered : Nat
  redAppleWanted : Nat
  greenAppleOrdered : Nat
  greenAppleWanted : Nat
  orangeOrdered : Nat
  orangeWanted : Nat
  bananaOrdered : Nat
  bananaWanted : Nat

/-- Represents the extra fruits for each type --/
structure ExtraFruits where
  redApple : Nat
  greenApple : Nat
  orange : Nat
  banana : Nat

/-- Calculates the extra fruits given the ordered and wanted quantities --/
def calculateExtraFruits (quantities : FruitQuantities) : ExtraFruits :=
  { redApple := quantities.redAppleOrdered - quantities.redAppleWanted,
    greenApple := quantities.greenAppleOrdered - quantities.greenAppleWanted,
    orange := quantities.orangeOrdered - quantities.orangeWanted,
    banana := quantities.bananaOrdered - quantities.bananaWanted }

/-- The theorem stating that the calculated extra fruits match the expected values --/
theorem extra_fruits_theorem (quantities : FruitQuantities) 
  (h : quantities = { redAppleOrdered := 6, redAppleWanted := 5,
                      greenAppleOrdered := 15, greenAppleWanted := 8,
                      orangeOrdered := 10, orangeWanted := 6,
                      bananaOrdered := 8, bananaWanted := 7 }) :
  calculateExtraFruits quantities = { redApple := 1, greenApple := 7, orange := 4, banana := 1 } := by
  sorry

end NUMINAMATH_CALUDE_extra_fruits_theorem_l355_35553


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l355_35576

/-- Given a square carpet with side length 12 feet, containing one large shaded square
    with side length S and eight smaller congruent shaded squares with side length T,
    where 12:S = S:T = 4, prove that the total shaded area is 13.5 square feet. -/
theorem carpet_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  S^2 + 8 * T^2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l355_35576


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l355_35596

/-- Given two hyperbolas with the same asymptotes, prove that M = 576/25 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, y^2/16 - x^2/25 = 1 ↔ x^2/36 - y^2/M = 1) → 
  (∀ x y : ℝ, y = (4/5)*x ↔ y = (Real.sqrt M / 6)*x) → 
  M = 576/25 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l355_35596


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l355_35574

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 9 + c + d) / 5 = 18 → (c + d) / 2 = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l355_35574


namespace NUMINAMATH_CALUDE_final_cafeteria_count_l355_35555

def total_students : ℕ := 300

def initial_cafeteria : ℕ := (2 * total_students) / 5
def initial_outside : ℕ := (3 * total_students) / 10
def initial_classroom : ℕ := total_students - initial_cafeteria - initial_outside

def outside_to_cafeteria : ℕ := (40 * initial_outside) / 100
def cafeteria_to_outside : ℕ := 5
def classroom_to_cafeteria : ℕ := (15 * initial_classroom + 50) / 100  -- Rounded up
def outside_to_classroom : ℕ := 2

theorem final_cafeteria_count :
  initial_cafeteria + outside_to_cafeteria - cafeteria_to_outside + classroom_to_cafeteria = 165 :=
sorry

end NUMINAMATH_CALUDE_final_cafeteria_count_l355_35555


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l355_35511

/-- A line y = 2x + c is tangent to the parabola y^2 = 8x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = 2 * p.1 + c ∧ p.2^2 = 8 * p.1) ↔ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l355_35511


namespace NUMINAMATH_CALUDE_megan_deleted_files_l355_35510

/-- Calculates the number of deleted files given the initial number of files,
    the number of folders after organizing, and the number of files per folder. -/
def deleted_files (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : ℕ :=
  initial_files - (num_folders * files_per_folder)

/-- Proves that Megan deleted 21 files given the problem conditions. -/
theorem megan_deleted_files :
  deleted_files 93 9 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_megan_deleted_files_l355_35510


namespace NUMINAMATH_CALUDE_sally_takes_home_17_pens_l355_35535

/-- Calculates the number of pens Sally takes home --/
def pens_taken_home (total_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_left := total_pens - pens_given
  let pens_in_locker := pens_left / 2
  pens_left - pens_in_locker

/-- Proves that Sally takes home 17 pens --/
theorem sally_takes_home_17_pens :
  pens_taken_home 342 44 7 = 17 := by
  sorry

#eval pens_taken_home 342 44 7

end NUMINAMATH_CALUDE_sally_takes_home_17_pens_l355_35535


namespace NUMINAMATH_CALUDE_middle_elementary_students_l355_35509

theorem middle_elementary_students (total : ℕ) 
  (h_total : total = 12000)
  (h_elementary : (15 : ℚ) / 16 * total = upper_elementary + middle_elementary)
  (h_not_upper : (1 : ℚ) / 2 * total = junior_high + middle_elementary)
  (h_groups : total = junior_high + upper_elementary + middle_elementary) :
  middle_elementary = 4875 := by
  sorry

end NUMINAMATH_CALUDE_middle_elementary_students_l355_35509


namespace NUMINAMATH_CALUDE_reverse_digits_problem_l355_35541

/-- Given a two-digit number, returns the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The problem statement -/
theorem reverse_digits_problem : ∃ (v : ℕ), 57 + v = reverse_digits 57 :=
  sorry

end NUMINAMATH_CALUDE_reverse_digits_problem_l355_35541


namespace NUMINAMATH_CALUDE_e_pi_third_in_first_quadrant_l355_35571

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Define the first quadrant
def first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

-- Theorem statement
theorem e_pi_third_in_first_quadrant :
  first_quadrant (Complex.exp (Complex.I * (π / 3))) :=
sorry

end NUMINAMATH_CALUDE_e_pi_third_in_first_quadrant_l355_35571


namespace NUMINAMATH_CALUDE_point_C_y_coordinate_sum_of_digits_l355_35547

/-- The function representing the graph y = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Sum of digits of a real number -/
noncomputable def sumOfDigits (y : ℝ) : ℕ := sorry

theorem point_C_y_coordinate_sum_of_digits 
  (A B C : ℝ × ℝ) 
  (hA : A.2 = f A.1) 
  (hB : B.2 = f B.1) 
  (hC : C.2 = f C.1) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (hParallel : A.2 = B.2) 
  (hArea : abs ((B.1 - A.1) * (C.2 - A.2)) / 2 = 100) :
  sumOfDigits C.2 = 6 := by sorry

end NUMINAMATH_CALUDE_point_C_y_coordinate_sum_of_digits_l355_35547


namespace NUMINAMATH_CALUDE_problem_solution_l355_35578

theorem problem_solution (m n : ℝ) : 
  (Real.sqrt (1 - m))^2 + |n + 2| = 0 → m - n = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l355_35578


namespace NUMINAMATH_CALUDE_complex_power_difference_l355_35554

/-- Given that i^2 = -1, prove that (1+2i)^24 - (1-2i)^24 = 0 -/
theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i)^24 - (1 - 2*i)^24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l355_35554


namespace NUMINAMATH_CALUDE_simple_interest_principal_l355_35575

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 25)
  (h2 : rate = 25 / 4)
  (h3 : time = 73 / 365)
  : (interest * 100) / (rate * time) = 2000 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l355_35575


namespace NUMINAMATH_CALUDE_root_product_sum_zero_l355_35502

noncomputable def sqrt10 : ℝ := Real.sqrt 10

theorem root_product_sum_zero 
  (y₁ y₂ y₃ : ℝ) 
  (h_roots : ∀ x, x^3 - 6*sqrt10*x^2 + 10 = 0 ↔ x = y₁ ∨ x = y₂ ∨ x = y₃)
  (h_order : y₁ < y₂ ∧ y₂ < y₃) : 
  y₂ * (y₁ + y₃) = 0 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_zero_l355_35502


namespace NUMINAMATH_CALUDE_fraction_simplification_l355_35583

theorem fraction_simplification : (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l355_35583


namespace NUMINAMATH_CALUDE_y_value_l355_35598

theorem y_value : (2023^2 - 1012) / 2023 = 2023 - 1012/2023 := by sorry

end NUMINAMATH_CALUDE_y_value_l355_35598


namespace NUMINAMATH_CALUDE_function_forms_with_common_tangent_l355_35567

/-- Given two functions f and g, prove that they have the specified forms
    when they pass through (2, 0) and have a common tangent at that point. -/
theorem function_forms_with_common_tangent 
  (f g : ℝ → ℝ) 
  (hf : ∃ a : ℝ, ∀ x, f x = 2 * x^3 + a * x)
  (hg : ∃ b c : ℝ, ∀ x, g x = b * x^2 + c)
  (pass_through : f 2 = 0 ∧ g 2 = 0)
  (common_tangent : (deriv f) 2 = (deriv g) 2) :
  (∀ x, f x = 2 * x^3 - 8 * x) ∧ 
  (∀ x, g x = 4 * x^2 - 16) := by
sorry

end NUMINAMATH_CALUDE_function_forms_with_common_tangent_l355_35567


namespace NUMINAMATH_CALUDE_smores_theorem_l355_35516

def smores_problem (graham_crackers : ℕ) (marshmallows : ℕ) : ℕ :=
  let smores_from_graham := graham_crackers / 2
  smores_from_graham - marshmallows

theorem smores_theorem (graham_crackers marshmallows : ℕ) :
  graham_crackers = 48 →
  marshmallows = 6 →
  smores_problem graham_crackers marshmallows = 18 :=
by sorry

end NUMINAMATH_CALUDE_smores_theorem_l355_35516


namespace NUMINAMATH_CALUDE_expression_equals_twenty_times_ten_to_1234_l355_35520

theorem expression_equals_twenty_times_ten_to_1234 :
  (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_twenty_times_ten_to_1234_l355_35520


namespace NUMINAMATH_CALUDE_tallest_player_height_calculation_l355_35507

/-- The height of the tallest player on a basketball team, given the height of the shortest player and the difference in height between the tallest and shortest players. -/
def tallest_player_height (shortest_player_height : ℝ) (height_difference : ℝ) : ℝ :=
  shortest_player_height + height_difference

/-- Theorem stating that given a shortest player height of 68.25 inches and a height difference of 9.5 inches, the tallest player's height is 77.75 inches. -/
theorem tallest_player_height_calculation :
  tallest_player_height 68.25 9.5 = 77.75 := by
  sorry

end NUMINAMATH_CALUDE_tallest_player_height_calculation_l355_35507


namespace NUMINAMATH_CALUDE_sum_of_squares_l355_35514

theorem sum_of_squares (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^2 + 9) / a = (b^2 + 9) / b ∧ (b^2 + 9) / b = (c^2 + 9) / c) : 
  a^2 + b^2 + c^2 = -27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l355_35514


namespace NUMINAMATH_CALUDE_solve_system_1_l355_35568

theorem solve_system_1 (x y : ℝ) : 
  2 * x + 3 * y = 16 ∧ x + 4 * y = 13 → x = 5 ∧ y = 2 := by
  sorry

#check solve_system_1

end NUMINAMATH_CALUDE_solve_system_1_l355_35568


namespace NUMINAMATH_CALUDE_tan_double_angle_l355_35590

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l355_35590


namespace NUMINAMATH_CALUDE_right_triangle_area_l355_35572

theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 34) : 
  (1/2) * a * b = 510 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l355_35572


namespace NUMINAMATH_CALUDE_convex_polygon_perimeter_bound_l355_35579

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool  -- We simplify the convexity check to a boolean for this statement

/-- A square in 2D space -/
structure Square where
  center : Real × Real
  side_length : Real

/-- Check if a point is inside or on the boundary of a square -/
def point_in_square (p : Real × Real) (s : Square) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  let half_side := s.side_length / 2
  x ≥ cx - half_side ∧ x ≤ cx + half_side ∧
  y ≥ cy - half_side ∧ y ≤ cy + half_side

/-- Check if a polygon is contained in a square -/
def polygon_in_square (p : ConvexPolygon) (s : Square) : Prop :=
  ∀ v ∈ p.vertices, point_in_square v s

/-- Calculate the perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : Real :=
  sorry  -- The actual calculation is omitted for brevity

/-- The main theorem -/
theorem convex_polygon_perimeter_bound (p : ConvexPolygon) (s : Square) :
  p.is_convex = true →
  s.side_length = 1 →
  polygon_in_square p s →
  perimeter p ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_perimeter_bound_l355_35579


namespace NUMINAMATH_CALUDE_chord_length_parabola_l355_35595

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem chord_length_parabola (C : Parabola) (l : Line) (A B : Point) :
  C.equation = (fun x y => x^2 = 4*y) →
  l.intercept = 1 →
  C.equation A.x A.y →
  C.equation B.x B.y →
  (A.y + B.y) / 2 = 5 →
  ∃ k, l.slope = k ∧ k^2 = 2 →
  ∃ AB : ℝ, AB = 6 ∧ AB = Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_parabola_l355_35595


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_6_minus_5_4_l355_35526

theorem least_prime_factor_of_5_6_minus_5_4 :
  Nat.minFac (5^6 - 5^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_6_minus_5_4_l355_35526


namespace NUMINAMATH_CALUDE_sequence_even_terms_l355_35531

def sequence_property (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ d : ℕ, d > 0 ∧ d < 10 ∧ d ∣ x (n-1) ∧ x n = x (n-1) + d

theorem sequence_even_terms (x : ℕ → ℕ) (h : sequence_property x) :
  (∃ n : ℕ, Even (x n)) ∧ (∀ m : ℕ, ∃ n : ℕ, n > m ∧ Even (x n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_even_terms_l355_35531


namespace NUMINAMATH_CALUDE_exactly_two_favor_policy_l355_35508

/-- The probability of a person favoring the policy -/
def p : ℝ := 0.6

/-- The number of people surveyed -/
def n : ℕ := 5

/-- The number of people who favor the policy in the desired outcome -/
def k : ℕ := 2

/-- The probability of exactly k out of n people favoring the policy -/
def prob_exactly_k (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_favor_policy :
  prob_exactly_k p n k = 0.2304 := by sorry

end NUMINAMATH_CALUDE_exactly_two_favor_policy_l355_35508


namespace NUMINAMATH_CALUDE_quadrilateral_area_l355_35573

/-- The area of a quadrilateral given its four sides and the angle between diagonals -/
theorem quadrilateral_area (a b c d ω : ℝ) (h_pos : 0 < ω ∧ ω < π) :
  ∃ t : ℝ, t = (1/4) * (b^2 + d^2 - a^2 - c^2) * Real.tan ω :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l355_35573


namespace NUMINAMATH_CALUDE_parabola_vertex_l355_35528

/-- Given a quadratic function f(x) = -x^2 + cx + d where c and d are real numbers,
    and the solution to f(x) ≤ 0 is (-∞, -4] ∪ [6, ∞),
    prove that the vertex of the parabola is (1, 25). -/
theorem parabola_vertex (c d : ℝ) :
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) →
  ∃ (vertex : ℝ × ℝ), vertex = (1, 25) ∧
    ∀ x, -x^2 + c*x + d ≤ -(x - vertex.1)^2 + vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l355_35528


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_a_no_solution_l355_35594

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for the solution of f(x) > 5
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = Set.Iio (-4/3) ∪ Set.Ioi 2 := by sorry

-- Theorem for the range of a when 1/(f(x)-4) = a has no solution
theorem range_of_a_no_solution :
  {a : ℝ | ∀ x, 1/(f x - 4) ≠ a} = Set.Ioo (-2/3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_a_no_solution_l355_35594


namespace NUMINAMATH_CALUDE_S_not_algorithmically_solvable_l355_35540

-- Define a type for expressions
inductive Expression
  | finite : Nat → Expression  -- Represents finite sums
  | infinite : Expression      -- Represents infinite sums

-- Define what it means for an expression to be algorithmically solvable
def is_algorithmically_solvable (e : Expression) : Prop :=
  match e with
  | Expression.finite _ => True
  | Expression.infinite => False

-- Define the infinite sum S = 1 + 2 + 3 + ...
def S : Expression := Expression.infinite

-- Theorem statement
theorem S_not_algorithmically_solvable :
  ¬(is_algorithmically_solvable S) :=
sorry

end NUMINAMATH_CALUDE_S_not_algorithmically_solvable_l355_35540


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l355_35586

-- Define a geometric sequence with positive terms
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Main theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l355_35586


namespace NUMINAMATH_CALUDE_sequence_relation_l355_35577

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : (y n)^2 = 3 * (x n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l355_35577


namespace NUMINAMATH_CALUDE_wider_can_radius_l355_35524

/-- Given two cylindrical cans with the same volume, where the height of one can is double 
    the height of the other, and the radius of the narrower can is 8 units, 
    the radius of the wider can is 8√2 units. -/
theorem wider_can_radius (h : ℝ) (r : ℝ) (h_pos : h > 0) : 
  π * 8^2 * (2*h) = π * r^2 * h → r = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_wider_can_radius_l355_35524


namespace NUMINAMATH_CALUDE_complex_trajectory_l355_35545

theorem complex_trajectory (z : ℂ) (x y : ℝ) :
  z = x + y * Complex.I →
  Complex.abs z ^ 2 - 2 * Complex.abs z - 3 = 0 →
  x ^ 2 + y ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_complex_trajectory_l355_35545


namespace NUMINAMATH_CALUDE_factorial_division_l355_35500

theorem factorial_division : 8 / 3 = 6720 :=
by
  -- Define 8! as given in the problem
  have h1 : 8 = 40320 := by sorry
  
  -- Define 3! (not given in the problem, but necessary for the proof)
  have h2 : 3 = 6 := by sorry
  
  -- Prove that 8! ÷ 3! = 6720
  sorry

end NUMINAMATH_CALUDE_factorial_division_l355_35500


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_parallel_intersections_perpendicular_line_in_plane_l355_35527

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (outside : Point → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (in_plane : Point → Plane → Prop)
variable (in_line : Point → Line → Prop)

-- Theorem 1
theorem unique_perpendicular_line 
  (p : Point) (π : Plane) (h : outside p π) :
  ∃! l : Line, perpendicular l π ∧ in_line p l :=
sorry

-- Theorem 2
theorem parallel_intersections 
  (π₁ π₂ π₃ : Plane) (h : parallel_planes π₁ π₂) :
  parallel (intersect π₁ π₃) (intersect π₂ π₃) :=
sorry

-- Theorem 3
theorem perpendicular_line_in_plane 
  (π₁ π₂ : Plane) (p : Point) (l : Line)
  (h₁ : perpendicular_planes π₁ π₂) (h₂ : in_plane p π₁)
  (h₃ : perpendicular l π₂) (h₄ : in_line p l) :
  ∀ q : Point, in_line q l → in_plane q π₁ :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_parallel_intersections_perpendicular_line_in_plane_l355_35527


namespace NUMINAMATH_CALUDE_animal_count_animal_group_count_l355_35521

theorem animal_count (total_horses : ℕ) (cow_cow_diff : ℕ) : ℕ :=
  let total_animals := 2 * (total_horses + cow_cow_diff)
  total_animals

theorem animal_group_count : animal_count 75 10 = 170 := by
  sorry

end NUMINAMATH_CALUDE_animal_count_animal_group_count_l355_35521


namespace NUMINAMATH_CALUDE_three_distinct_real_roots_l355_35592

/-- A cubic polynomial with specific conditions -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  b_neg : b < 0
  ab_eq_9c : a * b = 9 * c

/-- The polynomial function -/
def polynomial (p : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Theorem stating that the polynomial has three different real roots -/
theorem three_distinct_real_roots (p : CubicPolynomial) :
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    polynomial p x = 0 ∧ polynomial p y = 0 ∧ polynomial p z = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_distinct_real_roots_l355_35592


namespace NUMINAMATH_CALUDE_min_square_area_for_rectangles_l355_35523

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.height) (r1.height + r2.width)

/-- Theorem: The smallest square area to fit a 3x4 and a 4x5 rectangle with one rotated is 81 -/
theorem min_square_area_for_rectangles :
  let r1 : Rectangle := ⟨3, 4⟩
  let r2 : Rectangle := ⟨4, 5⟩
  (minSquareSide r1 r2) ^ 2 = 81 := by
  sorry

#eval (minSquareSide ⟨3, 4⟩ ⟨4, 5⟩) ^ 2

end NUMINAMATH_CALUDE_min_square_area_for_rectangles_l355_35523


namespace NUMINAMATH_CALUDE_fundraiser_hourly_rate_l355_35544

/-- Proves that if 8 volunteers working 40 hours each at $18 per hour raise the same total amount
    as 12 volunteers working 32 hours each, then the hourly rate for the second group is $15. -/
theorem fundraiser_hourly_rate
  (volunteers_last_week : ℕ)
  (hours_last_week : ℕ)
  (rate_last_week : ℚ)
  (volunteers_this_week : ℕ)
  (hours_this_week : ℕ)
  (h1 : volunteers_last_week = 8)
  (h2 : hours_last_week = 40)
  (h3 : rate_last_week = 18)
  (h4 : volunteers_this_week = 12)
  (h5 : hours_this_week = 32)
  (h6 : volunteers_last_week * hours_last_week * rate_last_week =
        volunteers_this_week * hours_this_week * (15 : ℚ)) :
  15 = (volunteers_last_week * hours_last_week * rate_last_week) /
       (volunteers_this_week * hours_this_week) :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_hourly_rate_l355_35544


namespace NUMINAMATH_CALUDE_hexagonal_prism_volume_l355_35589

-- Define the hexagonal prism
structure HexagonalPrism where
  sideEdgeLength : ℝ
  lateralSurfaceAreaQuadPrism : ℝ

-- Define the theorem
theorem hexagonal_prism_volume 
  (prism : HexagonalPrism)
  (h1 : prism.sideEdgeLength = 3)
  (h2 : prism.lateralSurfaceAreaQuadPrism = 30) :
  ∃ (volume : ℝ), volume = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_volume_l355_35589


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_minimal_m_l355_35597

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 10) ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define the sufficient condition
def sufficient (m : ℝ) : Prop :=
  ∀ x, q x m → p x

-- Define the not necessary condition
def not_necessary (m : ℝ) : Prop :=
  ∃ x, p x ∧ ¬(q x m)

-- Main theorem
theorem sufficient_but_not_necessary_condition (m : ℝ) 
  (h1 : m ≥ 3) (h2 : m > 0) : 
  sufficient m ∧ not_necessary m := by
  sorry

-- Prove that this is the minimal value of m
theorem minimal_m :
  ∀ m < 3, ¬(sufficient m ∧ not_necessary m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_minimal_m_l355_35597


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l355_35534

/-- Given a bag marked at $240 with a 50% discount, prove that the discounted price is $120. -/
theorem discounted_price_calculation (marked_price : ℝ) (discount_rate : ℝ) :
  marked_price = 240 →
  discount_rate = 0.5 →
  marked_price * (1 - discount_rate) = 120 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l355_35534


namespace NUMINAMATH_CALUDE_acute_triangle_angle_inequality_iff_sine_inequality_l355_35525

theorem acute_triangle_angle_inequality_iff_sine_inequality 
  (A B C : Real) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (A > B ∧ B > C) ↔ (Real.sin (2*A) < Real.sin (2*B) ∧ Real.sin (2*B) < Real.sin (2*C)) :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_inequality_iff_sine_inequality_l355_35525


namespace NUMINAMATH_CALUDE_tomato_seeds_planted_l355_35513

/-- The total number of tomato seeds planted by Mike, Ted, and Sarah -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon + sarah_morning + sarah_afternoon

theorem tomato_seeds_planted :
  ∃ (mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon : ℕ),
    mike_morning = 50 ∧
    ted_morning = 2 * mike_morning ∧
    sarah_morning = mike_morning + 30 ∧
    mike_afternoon = 60 ∧
    ted_afternoon = mike_afternoon - 20 ∧
    sarah_afternoon = sarah_morning + 20 ∧
    total_seeds mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon = 430 :=
by sorry

end NUMINAMATH_CALUDE_tomato_seeds_planted_l355_35513


namespace NUMINAMATH_CALUDE_golden_ratio_function_l355_35560

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_function (f : ℝ → ℝ) :
  (∀ x > 0, Monotone f) →
  (∀ x > 0, f x > 0) →
  (∀ x > 0, f x * f (f x + 1 / x) = 1) →
  f 1 = φ := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_function_l355_35560


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l355_35522

/-- Given an arithmetic sequence with common difference 2,
    if a_1, a_3, and a_4 form a geometric sequence, then a_1 = -8 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l355_35522


namespace NUMINAMATH_CALUDE_faster_train_speed_l355_35561

/-- Proves that the speed of the faster train is 20/3 m/s given the specified conditions -/
theorem faster_train_speed
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h_length : train_length = 100)
  (h_time : crossing_time = 20)
  (h_speed_ratio : ∃ (v : ℝ), v > 0 ∧ faster_speed = 2 * v ∧ slower_speed = v)
  (h_relative_speed : relative_speed = faster_speed + slower_speed)
  (h_distance : total_distance = 2 * train_length)
  (h_speed_formula : relative_speed = total_distance / crossing_time) :
  faster_speed = 20 / 3 :=
sorry

end NUMINAMATH_CALUDE_faster_train_speed_l355_35561


namespace NUMINAMATH_CALUDE_parallel_resistors_existence_l355_35565

theorem parallel_resistors_existence : ∃ (R R₁ R₂ : ℕ+), 
  R.val * (R₁.val + R₂.val) = R₁.val * R₂.val ∧ 
  R.val > 0 ∧ R₁.val > 0 ∧ R₂.val > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_existence_l355_35565


namespace NUMINAMATH_CALUDE_abs_neg_2022_l355_35581

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2022_l355_35581


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l355_35549

theorem arithmetic_geometric_progression_ratio 
  (a₁ d : ℝ) (h : d ≠ 0) : 
  let a₂ := a₁ + d
  let a₃ := a₁ + 2*d
  let r := a₂ * a₃ / (a₁ * a₂)
  (r * r = 1 ∧ (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) → r = -2 := by
  sorry

#check arithmetic_geometric_progression_ratio

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l355_35549


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l355_35517

theorem simplify_sqrt_expression :
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l355_35517


namespace NUMINAMATH_CALUDE_orange_buckets_total_l355_35582

theorem orange_buckets_total (bucket1 bucket2 bucket3 : ℕ) : 
  bucket1 = 22 →
  bucket2 = bucket1 + 17 →
  bucket3 = bucket2 - 11 →
  bucket1 + bucket2 + bucket3 = 89 := by
sorry

end NUMINAMATH_CALUDE_orange_buckets_total_l355_35582


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l355_35588

/-- Given two concentric circles where the radius of the outer circle is twice
    the radius of the inner circle, and the width of the gray region between
    them is 3 feet, prove that the area of the gray region is 21π square feet. -/
theorem area_between_concentric_circles (r : ℝ) : 
  r > 0 → -- Inner circle radius is positive
  2 * r - r = 3 → -- Width of gray region is 3
  π * (2 * r)^2 - π * r^2 = 21 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l355_35588


namespace NUMINAMATH_CALUDE_noah_sticker_count_l355_35532

/-- Given the number of stickers for Kristoff, calculate the number of stickers Noah has -/
def noahs_stickers (kristoff : ℕ) : ℕ :=
  let riku : ℕ := 25 * kristoff
  let lila : ℕ := 2 * (kristoff + riku)
  kristoff * lila - 3

theorem noah_sticker_count : noahs_stickers 85 = 375697 := by
  sorry

end NUMINAMATH_CALUDE_noah_sticker_count_l355_35532


namespace NUMINAMATH_CALUDE_bag_probabilities_l355_35591

structure Bag where
  red_balls : ℕ
  blue_balls : ℕ
  red_ones : ℕ
  blue_ones : ℕ

def sample_bag : Bag := ⟨6, 3, 2, 1⟩

def prob_one_red_sum_three (b : Bag) : ℚ :=
  16 / 81

def prob_first_red (b : Bag) : ℚ :=
  b.red_balls / (b.red_balls + b.blue_balls)

def prob_second_one (b : Bag) : ℚ :=
  1 / 3

theorem bag_probabilities (b : Bag) 
  (h1 : b.red_balls = 6) 
  (h2 : b.blue_balls = 3) 
  (h3 : b.red_ones = 2) 
  (h4 : b.blue_ones = 1) :
  prob_one_red_sum_three b = 16 / 81 ∧
  prob_first_red b = 2 / 3 ∧
  prob_second_one b = 1 / 3 ∧
  prob_first_red b * prob_second_one b = 
    (b.red_ones / (b.red_balls + b.blue_balls - 1) + 
     (b.red_balls - b.red_ones) * b.blue_ones / ((b.red_balls + b.blue_balls) * (b.red_balls + b.blue_balls - 1))) := by
  sorry

#check bag_probabilities

end NUMINAMATH_CALUDE_bag_probabilities_l355_35591


namespace NUMINAMATH_CALUDE_train_length_calculation_l355_35537

/-- The length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 46) (h2 : v2 = 36) (h3 : t = 27) :
  let relative_speed := (v1 - v2) * (5 / 18)
  let distance := relative_speed * t
  let train_length := distance / 2
  train_length = 37.5 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l355_35537


namespace NUMINAMATH_CALUDE_sphere_radius_from_depression_l355_35563

/-- The radius of a sphere that creates a circular depression with given diameter and depth when partially submerged. -/
def sphere_radius (depression_diameter : ℝ) (depression_depth : ℝ) : ℝ :=
  13

/-- Theorem stating that a sphere with radius 13cm creates a circular depression
    with diameter 24cm and depth 8cm when partially submerged. -/
theorem sphere_radius_from_depression :
  sphere_radius 24 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_depression_l355_35563


namespace NUMINAMATH_CALUDE_sector_area_l355_35515

/-- Given a sector with radius 4 cm and arc length 12 cm, its area is 24 cm². -/
theorem sector_area (radius : ℝ) (arc_length : ℝ) (area : ℝ) : 
  radius = 4 → arc_length = 12 → area = (1/2) * arc_length * radius → area = 24 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l355_35515
