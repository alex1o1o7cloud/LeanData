import Mathlib

namespace NUMINAMATH_CALUDE_least_three_digit_with_product_24_l989_98993

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_product_24 :
  (is_three_digit 234) ∧
  (digit_product 234 = 24) ∧
  (∀ m : ℕ, is_three_digit m → digit_product m = 24 → 234 ≤ m) :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_product_24_l989_98993


namespace NUMINAMATH_CALUDE_binomial_150_150_l989_98967

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l989_98967


namespace NUMINAMATH_CALUDE_probability_white_or_red_l989_98951

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def red_balls : ℕ := 5

def total_balls : ℕ := white_balls + black_balls + red_balls

def favorable_outcomes : ℕ := white_balls + red_balls

theorem probability_white_or_red :
  (favorable_outcomes : ℚ) / total_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_or_red_l989_98951


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l989_98982

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 - 9 * x₁ + 5 = 0) → 
  (6 * x₂^2 - 9 * x₂ + 5 = 0) → 
  x₁^2 + x₂^2 = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l989_98982


namespace NUMINAMATH_CALUDE_inequality_range_l989_98971

theorem inequality_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 
    Real.sin (2 * θ) - (2 * Real.sqrt 2 + Real.sqrt 2 * a) * Real.sin (θ + π / 4) - 
    (2 * Real.sqrt 2 / Real.cos (θ - π / 4)) > -3 - 2 * a) → 
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l989_98971


namespace NUMINAMATH_CALUDE_distribute_eight_to_two_groups_l989_98904

/-- The number of ways to distribute n distinct objects into 2 non-empty groups -/
def distribute_to_two_groups (n : ℕ) : ℕ :=
  2^n - 2

/-- The theorem stating that distributing 8 distinct objects into 2 non-empty groups results in 254 possibilities -/
theorem distribute_eight_to_two_groups :
  distribute_to_two_groups 8 = 254 := by
  sorry

#eval distribute_to_two_groups 8

end NUMINAMATH_CALUDE_distribute_eight_to_two_groups_l989_98904


namespace NUMINAMATH_CALUDE_original_mean_l989_98943

theorem original_mean (n : ℕ) (decrement : ℝ) (updated_mean : ℝ) (h1 : n = 50) (h2 : decrement = 34) (h3 : updated_mean = 166) : 
  (n : ℝ) * updated_mean + n * decrement = n * 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_l989_98943


namespace NUMINAMATH_CALUDE_intersection_point_on_diagonal_l989_98990

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Check if two lines intersect -/
def linesIntersect (l1 l2 : Line3D) : Prop :=
  sorry

theorem intersection_point_on_diagonal (A B C D E F G H P : Point3D)
  (AB : Line3D) (BC : Line3D) (CD : Line3D) (DA : Line3D)
  (EF : Line3D) (GH : Line3D) (AC : Line3D)
  (ABC : Plane3D) (ADC : Plane3D) :
  pointOnLine E AB →
  pointOnLine F BC →
  pointOnLine G CD →
  pointOnLine H DA →
  linesIntersect EF GH →
  pointOnLine P EF →
  pointOnLine P GH →
  pointOnPlane E ABC →
  pointOnPlane F ABC →
  pointOnPlane G ADC →
  pointOnPlane H ADC →
  pointOnLine P AC :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_diagonal_l989_98990


namespace NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l989_98957

theorem quadratic_equality_implies_coefficient (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l989_98957


namespace NUMINAMATH_CALUDE_binomial_10_2_l989_98954

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

/-- Theorem: The binomial coefficient (10 choose 2) equals 45 -/
theorem binomial_10_2 : binomial 10 2 = 45 := by sorry

end NUMINAMATH_CALUDE_binomial_10_2_l989_98954


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l989_98973

/-- Represents the savings from Coupon A (20% discount) -/
def savingsA (price : ℝ) : ℝ := 0.2 * price

/-- Represents the savings from Coupon B ($50 flat discount) -/
def savingsB : ℝ := 50

/-- Represents the savings from Coupon C (30% discount on amount over $200) -/
def savingsC (price : ℝ) : ℝ := 0.3 * (price - 200)

/-- The minimum price where Coupon A saves at least as much as Coupons B and C -/
def minPrice : ℝ := 250

/-- The maximum price where Coupon A saves at least as much as Coupons B and C -/
def maxPrice : ℝ := 600

theorem coupon_savings_difference :
  ∀ price : ℝ, price > 200 →
  (savingsA price ≥ savingsB ∧ savingsA price ≥ savingsC price) →
  minPrice ≤ price ∧ price ≤ maxPrice →
  maxPrice - minPrice = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l989_98973


namespace NUMINAMATH_CALUDE_sample_size_is_twenty_l989_98961

/-- Represents the total number of employees in the company -/
def total_employees : ℕ := 1000

/-- Represents the number of middle-aged workers in the company -/
def middle_aged_workers : ℕ := 350

/-- Represents the number of middle-aged workers in the sample -/
def sample_middle_aged : ℕ := 7

/-- Theorem stating that the sample size is 20 given the conditions -/
theorem sample_size_is_twenty :
  ∃ (sample_size : ℕ),
    (sample_middle_aged : ℚ) / sample_size = middle_aged_workers / total_employees ∧
    sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_twenty_l989_98961


namespace NUMINAMATH_CALUDE_problem_solution_l989_98930

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define the theorem
theorem problem_solution :
  -- Given conditions
  ∃ m : ℝ,
    m > 0 ∧
    (∀ x : ℝ, f (x + 5) ≤ 3 * m ↔ -7 ≤ x ∧ x ≤ -1) ∧
    -- Part 1: The value of m is 1
    m = 1 ∧
    -- Part 2: Maximum value of 2a√(1+b²) is 2√2
    (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 + b^2 = 3 * m →
      2 * a * Real.sqrt (1 + b^2) ≤ 2 * Real.sqrt 2) ∧
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a^2 + b^2 = 3 * m ∧
      2 * a * Real.sqrt (1 + b^2) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l989_98930


namespace NUMINAMATH_CALUDE_triangle_side_length_l989_98907

open Real

-- Define the triangle
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  -- Given conditions
  c = 4 * sqrt 2 →
  B = π / 4 →  -- 45° in radians
  S = 2 →
  -- Area formula
  S = (1 / 2) * a * c * sin B →
  -- Law of Cosines
  b^2 = a^2 + c^2 - 2*a*c*(cos B) →
  -- Conclusion
  b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l989_98907


namespace NUMINAMATH_CALUDE_pitchers_prepared_is_six_l989_98968

/-- Represents the number of glasses of lemonade a single pitcher can serve. -/
def glasses_per_pitcher : ℕ := 5

/-- Represents the total number of glasses of lemonade served. -/
def total_glasses_served : ℕ := 30

/-- Calculates the number of pitchers needed to serve the given number of glasses. -/
def pitchers_needed (total_glasses : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  total_glasses / glasses_per_pitcher

/-- Proves that the number of pitchers prepared is 6. -/
theorem pitchers_prepared_is_six :
  pitchers_needed total_glasses_served glasses_per_pitcher = 6 := by
  sorry

end NUMINAMATH_CALUDE_pitchers_prepared_is_six_l989_98968


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l989_98909

theorem negation_of_existence_inequality (p : Prop) :
  (¬ p ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0) ↔
  (p ↔ ∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l989_98909


namespace NUMINAMATH_CALUDE_line_perpendicular_theorem_l989_98969

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_perpendicular_theorem
  (m n : Line3D) (α β : Plane3D)
  (h1 : ¬ parallel_planes α β)
  (h2 : perpendicular m α)
  (h3 : ¬ parallel n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_theorem_l989_98969


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l989_98944

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (Complex.I - 1)^2 + 4 / (Complex.I + 1)
  (z.im = -3) := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l989_98944


namespace NUMINAMATH_CALUDE_correct_num_technicians_l989_98941

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℚ
  technician_salary : ℚ
  other_salary : ℚ

/-- The number of technicians in the workshop -/
def num_technicians (w : Workshop) : ℕ :=
  7  -- We'll prove this is correct

/-- The given workshop scenario -/
def given_workshop : Workshop :=
  { total_workers := 56
    avg_salary := 6750
    technician_salary := 12000
    other_salary := 6000 }

/-- Theorem stating that the number of technicians in the given workshop is correct -/
theorem correct_num_technicians :
    let n := num_technicians given_workshop
    let m := given_workshop.total_workers - n
    n + m = given_workshop.total_workers ∧
    (n * given_workshop.technician_salary + m * given_workshop.other_salary) / given_workshop.total_workers = given_workshop.avg_salary :=
  sorry


end NUMINAMATH_CALUDE_correct_num_technicians_l989_98941


namespace NUMINAMATH_CALUDE_punch_bowl_theorem_l989_98927

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Mark adds in the second refill -/
def second_refill : ℝ := 4

/-- The amount of punch Sally drinks -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to completely fill the bowl at the end -/
def final_addition : ℝ := 12

/-- The initial amount of punch Mark added to the bowl -/
def initial_amount : ℝ := 4

theorem punch_bowl_theorem :
  let after_cousin := initial_amount / 2
  let after_second_refill := after_cousin + second_refill
  let after_sally := after_second_refill - sally_drinks
  after_sally + final_addition = bowl_capacity :=
by sorry

end NUMINAMATH_CALUDE_punch_bowl_theorem_l989_98927


namespace NUMINAMATH_CALUDE_max_value_log_sum_l989_98922

theorem max_value_log_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a * b = 1000) :
  Real.sqrt (1 + Real.log a / Real.log 10) + Real.sqrt (1 + Real.log b / Real.log 10) ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_log_sum_l989_98922


namespace NUMINAMATH_CALUDE_zero_subset_M_l989_98964

def M : Set ℤ := {x : ℤ | |x| < 5}

theorem zero_subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_M_l989_98964


namespace NUMINAMATH_CALUDE_min_abs_sum_squared_matrix_l989_98906

-- Define the matrix type
def Matrix2x2 (α : Type) := Fin 2 → Fin 2 → α

-- Define the matrix multiplication
def matMul (A B : Matrix2x2 ℤ) : Matrix2x2 ℤ :=
  λ i j => (Finset.univ.sum λ k => A i k * B k j)

-- Define the identity matrix
def identityMatrix : Matrix2x2 ℤ :=
  λ i j => if i = j then 9 else 0

-- Define the absolute value sum
def absSum (a b c d : ℤ) : ℤ :=
  |a| + |b| + |c| + |d|

theorem min_abs_sum_squared_matrix :
  ∃ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (matMul (λ i j => match i, j with
      | 0, 0 => a
      | 0, 1 => b
      | 1, 0 => c
      | 1, 1 => d) (λ i j => match i, j with
      | 0, 0 => a
      | 0, 1 => b
      | 1, 0 => c
      | 1, 1 => d)) = identityMatrix ∧
    (∀ (a' b' c' d' : ℤ),
      a' ≠ 0 → b' ≠ 0 → c' ≠ 0 → d' ≠ 0 →
      (matMul (λ i j => match i, j with
        | 0, 0 => a'
        | 0, 1 => b'
        | 1, 0 => c'
        | 1, 1 => d') (λ i j => match i, j with
        | 0, 0 => a'
        | 0, 1 => b'
        | 1, 0 => c'
        | 1, 1 => d')) = identityMatrix →
      absSum a b c d ≤ absSum a' b' c' d') ∧
    absSum a b c d = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_min_abs_sum_squared_matrix_l989_98906


namespace NUMINAMATH_CALUDE_x_value_in_equation_l989_98910

theorem x_value_in_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 * x^2 + 24 * x * y = 2 * x^3 + 3 * x^2 * y^2) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_equation_l989_98910


namespace NUMINAMATH_CALUDE_root_existence_and_bounds_l989_98936

theorem root_existence_and_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x₁ x₂ : ℝ),
    (1 / x₁ + 1 / (x₁ - a) + 1 / (x₁ + b) = 0) ∧
    (1 / x₂ + 1 / (x₂ - a) + 1 / (x₂ + b) = 0) ∧
    (a / 3 ≤ x₁ ∧ x₁ ≤ 2 * a / 3) ∧
    (-2 * b / 3 ≤ x₂ ∧ x₂ ≤ -b / 3) :=
by sorry

end NUMINAMATH_CALUDE_root_existence_and_bounds_l989_98936


namespace NUMINAMATH_CALUDE_freshmen_psych_majors_percentage_l989_98978

/-- The percentage of freshmen psychology majors in the School of Liberal Arts
    among all students at a certain college. -/
theorem freshmen_psych_majors_percentage
  (total_students : ℕ)
  (freshmen_percentage : ℚ)
  (liberal_arts_percentage : ℚ)
  (psychology_percentage : ℚ)
  (h1 : freshmen_percentage = 2/5)
  (h2 : liberal_arts_percentage = 1/2)
  (h3 : psychology_percentage = 1/2)
  : (freshmen_percentage * liberal_arts_percentage * psychology_percentage : ℚ) = 1/10 := by
  sorry

#check freshmen_psych_majors_percentage

end NUMINAMATH_CALUDE_freshmen_psych_majors_percentage_l989_98978


namespace NUMINAMATH_CALUDE_teacher_age_l989_98924

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 21 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 42 :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l989_98924


namespace NUMINAMATH_CALUDE_laura_triathlon_speed_l989_98923

theorem laura_triathlon_speed :
  ∃ x : ℝ, x > 0 ∧ (20 / (2 * x + 1)) + (5 / x) + (5 / 60) = 110 / 60 := by
  sorry

end NUMINAMATH_CALUDE_laura_triathlon_speed_l989_98923


namespace NUMINAMATH_CALUDE_minimum_students_for_photo_l989_98979

def photo_cost (x : ℝ) : ℝ := 5 + (x - 2) * 0.8

theorem minimum_students_for_photo : 
  ∃ x : ℝ, x ≥ 17 ∧ 
  (∀ y : ℝ, y ≥ x → photo_cost y / y ≤ 1) ∧
  (∀ z : ℝ, z < x → photo_cost z / z > 1) :=
sorry

end NUMINAMATH_CALUDE_minimum_students_for_photo_l989_98979


namespace NUMINAMATH_CALUDE_vector_difference_l989_98925

/-- Given two 2D vectors a and b, prove that their difference is (5, -3) -/
theorem vector_difference (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) (hb : b = (-3, 4)) : 
  a - b = (5, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_l989_98925


namespace NUMINAMATH_CALUDE_smallest_square_perimeter_l989_98974

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.sideLength

/-- Represents three concentric squares -/
structure ConcentricSquares where
  largest : Square
  middle : Square
  smallest : Square
  distanceBetweenSides : ℝ

/-- The theorem stating the perimeter of the smallest square in the given configuration -/
theorem smallest_square_perimeter (cs : ConcentricSquares)
    (h1 : cs.largest.sideLength = 22)
    (h2 : cs.distanceBetweenSides = 3)
    (h3 : cs.middle.sideLength = cs.largest.sideLength - 2 * cs.distanceBetweenSides)
    (h4 : cs.smallest.sideLength = cs.middle.sideLength - 2 * cs.distanceBetweenSides) :
    cs.smallest.perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_perimeter_l989_98974


namespace NUMINAMATH_CALUDE_ellipse_properties_l989_98998

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

-- Define the conditions
def is_valid_ellipse (e : Ellipse) : Prop :=
  ∃ c : ℝ, 
    e.a + c = Real.sqrt 2 + 1 ∧
    e.a = Real.sqrt 2 * c ∧
    e.a^2 = e.b^2 + c^2

-- Define the standard equation
def standard_equation (e : Ellipse) : Prop :=
  e.a^2 = 2 ∧ e.b^2 = 1

-- Define the line passing through the left focus
def line_through_focus (k : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, y t = k * (x t + 1)

-- Define the condition for the midpoint
def midpoint_on_line (x y : ℝ → ℝ) : Prop :=
  ∃ t₁ t₂, x ((t₁ + t₂)/2) + y ((t₁ + t₂)/2) = 0

-- The main theorem
theorem ellipse_properties (e : Ellipse) (h : is_valid_ellipse e) :
  standard_equation e ∧
  (∀ k x y, line_through_focus k x y → midpoint_on_line x y →
    (k = 0 ∨ k = 1/2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l989_98998


namespace NUMINAMATH_CALUDE_value_between_seven_and_eight_l989_98994

theorem value_between_seven_and_eight :
  7 < (3 * Real.sqrt 15 + 2 * Real.sqrt 5) * Real.sqrt (1/5) ∧
  (3 * Real.sqrt 15 + 2 * Real.sqrt 5) * Real.sqrt (1/5) < 8 := by
  sorry

end NUMINAMATH_CALUDE_value_between_seven_and_eight_l989_98994


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l989_98934

theorem shopkeeper_gain_percentage (marked_price cost_price : ℝ) :
  marked_price > 0 ∧ cost_price > 0 ∧
  0.9 * marked_price = 1.17 * cost_price →
  (marked_price - cost_price) / cost_price = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l989_98934


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l989_98947

theorem dolphin_training_hours 
  (num_dolphins : ℕ) 
  (num_trainers : ℕ) 
  (hours_per_trainer : ℕ) 
  (h1 : num_dolphins = 4) 
  (h2 : num_trainers = 2) 
  (h3 : hours_per_trainer = 6) : 
  (num_trainers * hours_per_trainer) / num_dolphins = 3 := by
sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l989_98947


namespace NUMINAMATH_CALUDE_shelving_orders_eq_1280_l989_98948

/-- The number of books in total -/
def total_books : ℕ := 10

/-- The label of the book that has already been shelved -/
def shelved_book : ℕ := 9

/-- Calculate the number of different possible orders for shelving the remaining books -/
def shelving_orders : ℕ :=
  (Finset.range (total_books - 1)).sum (fun k =>
    (Nat.choose (total_books - 2) k) * (k + 2))

/-- Theorem stating that the number of different possible orders for shelving the remaining books is 1280 -/
theorem shelving_orders_eq_1280 : shelving_orders = 1280 := by
  sorry

end NUMINAMATH_CALUDE_shelving_orders_eq_1280_l989_98948


namespace NUMINAMATH_CALUDE_book_price_decrease_l989_98929

theorem book_price_decrease (P : ℝ) (x : ℝ) : 
  P - ((P - (x / 100) * P) * 1.2) = 10.000000000000014 → 
  x = 50 / 3 := by
sorry

end NUMINAMATH_CALUDE_book_price_decrease_l989_98929


namespace NUMINAMATH_CALUDE_circle_equation_l989_98901

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y + c = 0

/-- Checks if a point lies on a given circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem to prove -/
theorem circle_equation : ∃ (c : Circle),
  (c.center.x + c.center.y - 2 = 0) ∧
  pointOnCircle ⟨1, -1⟩ c ∧
  pointOnCircle ⟨-1, 1⟩ c ∧
  c.center = ⟨1, 1⟩ ∧
  c.radius = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l989_98901


namespace NUMINAMATH_CALUDE_player_a_wins_l989_98945

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a move in the game -/
inductive Move
  | right : Move
  | up : Move

/-- Represents the game state -/
structure GameState where
  piecePosition : Point
  markedPoints : Set Point
  movesLeft : ℕ

/-- The game rules -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.right => 
    let newPos := Point.mk (state.piecePosition.x + 1) state.piecePosition.y
    newPos ∉ state.markedPoints
  | Move.up => 
    let newPos := Point.mk state.piecePosition.x (state.piecePosition.y + 1)
    newPos ∉ state.markedPoints

/-- Player A's strategy -/
def strategyA (k : ℕ) (state : GameState) : Point := sorry

/-- Theorem: Player A has a winning strategy for any positive k -/
theorem player_a_wins (k : ℕ) (h : k > 0) : 
  ∃ (strategy : GameState → Point), 
    ∀ (initialState : GameState),
      (∀ (move : Move), ¬isValidMove initialState move) ∨
      (∃ (finalState : GameState), 
        finalState.markedPoints = insert (strategy initialState) initialState.markedPoints ∧
        ∀ (move : Move), ¬isValidMove finalState move) := by
  sorry

end NUMINAMATH_CALUDE_player_a_wins_l989_98945


namespace NUMINAMATH_CALUDE_burn_represents_8615_l989_98987

/-- Represents a mapping from characters to digits -/
def DigitMapping := Char → Fin 10

/-- The sequence of characters used in the code -/
def codeSequence : List Char := ['G', 'R', 'E', 'A', 'T', 'N', 'U', 'M', 'B', 'S']

/-- Creates a mapping from the code sequence to digits 0-9 -/
def createMapping (seq : List Char) : DigitMapping :=
  fun c => match seq.indexOf? c with
    | some i => ⟨i, by sorry⟩
    | none => 0

/-- The mapping for our specific code -/
def mapping : DigitMapping := createMapping codeSequence

/-- Converts a string to a number using the given mapping -/
def stringToNumber (s : String) (m : DigitMapping) : Nat :=
  s.foldr (fun c acc => acc * 10 + m c) 0

theorem burn_represents_8615 :
  stringToNumber "BURN" mapping = 8615 := by sorry

end NUMINAMATH_CALUDE_burn_represents_8615_l989_98987


namespace NUMINAMATH_CALUDE_pizza_order_l989_98981

theorem pizza_order (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : people = 18) 
  (h2 : slices_per_person = 3) 
  (h3 : slices_per_pizza = 9) : 
  (people * slices_per_person) / slices_per_pizza = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l989_98981


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l989_98915

/-- Given that a and b are inversely proportional and a = 3b when a + b = 60,
    prove that b = -67.5 when a = -10 -/
theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) : 
  (∀ x y, x * y = k → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) →  -- inverse proportion
  (∃ a' b', a' + b' = 60 ∧ a' = 3 * b') →                   -- condition when sum is 60
  (a = -10) →                                               -- given a value
  (b = -67.5) :=                                            -- to prove
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l989_98915


namespace NUMINAMATH_CALUDE_sea_glass_ratio_l989_98920

/-- Sea glass collection problem -/
theorem sea_glass_ratio : 
  ∀ (blanche_green blanche_red rose_red rose_blue dorothy_total : ℕ),
  blanche_green = 12 →
  blanche_red = 3 →
  rose_red = 9 →
  rose_blue = 11 →
  dorothy_total = 57 →
  ∃ (dorothy_red dorothy_blue : ℕ),
    dorothy_blue = 3 * rose_blue ∧
    dorothy_red + dorothy_blue = dorothy_total ∧
    2 * (blanche_red + rose_red) = dorothy_red :=
by sorry

end NUMINAMATH_CALUDE_sea_glass_ratio_l989_98920


namespace NUMINAMATH_CALUDE_lawn_width_l989_98976

theorem lawn_width (area : ℝ) (length : ℝ) (width : ℝ) 
  (h1 : area = 20)
  (h2 : length = 4)
  (h3 : area = length * width) : 
  width = 5 := by
sorry

end NUMINAMATH_CALUDE_lawn_width_l989_98976


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l989_98946

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection point Q
def Q (k m : ℝ) : ℝ × ℝ := (-4, k*(-4) + m)

-- Define the left focus F
def F : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem ellipse_constant_product (k m : ℝ) (A B P : ℝ × ℝ) :
  E A.1 A.2 →
  E B.1 B.2 →
  E P.1 P.2 →
  l k m A.1 A.2 →
  l k m B.1 B.2 →
  P = (A.1 + B.1, A.2 + B.2) →
  (P.1 - F.1) * (Q k m).1 + (P.2 - F.2) * (Q k m).2 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l989_98946


namespace NUMINAMATH_CALUDE_twentieth_term_is_220_l989_98916

def a (n : ℕ) : ℚ := (1/2) * n * (n + 2)

theorem twentieth_term_is_220 : a 20 = 220 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_220_l989_98916


namespace NUMINAMATH_CALUDE_min_value_C_over_D_l989_98903

theorem min_value_C_over_D (x C D : ℝ) (hx : x ≠ 0) 
  (hC : x^2 + 1/x^2 = C) (hD : x + 1/x = D) (hCpos : C > 0) (hDpos : D > 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ y, y = C / D → y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_C_over_D_l989_98903


namespace NUMINAMATH_CALUDE_triangle_theorem_l989_98988

theorem triangle_theorem (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a * (Real.sin A - Real.sin B) + b * Real.sin B = c * Real.sin C → -- Line condition
  2 * (Real.cos (A / 2))^2 - 2 * (Real.sin (B / 2))^2 = Real.sqrt 3 / 2 → -- Given equation
  A < B → -- Given inequality
  C = π / 3 ∧ c / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l989_98988


namespace NUMINAMATH_CALUDE_ellipse_k_range_l989_98942

/-- The equation of an ellipse in terms of parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧ 
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ Set.Ioo (-3 : ℝ) (-1/2) ∪ Set.Ioo (-1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l989_98942


namespace NUMINAMATH_CALUDE_equation_transformation_l989_98965

theorem equation_transformation :
  ∀ x : ℝ, x^2 - 6*x = 0 ↔ (x - 3)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_equation_transformation_l989_98965


namespace NUMINAMATH_CALUDE_loes_speed_l989_98955

/-- Proves that Loe's speed is 50 mph given the conditions of the problem -/
theorem loes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  time = 1.5 →
  final_distance = 15 →
  ∃ (loe_speed : ℝ), loe_speed = 50 ∧
    teena_speed * time - loe_speed * time = final_distance + initial_distance :=
by sorry

end NUMINAMATH_CALUDE_loes_speed_l989_98955


namespace NUMINAMATH_CALUDE_withdrawn_players_matches_l989_98950

/-- Represents a table tennis tournament -/
structure TableTennisTournament where
  n : ℕ  -- Total number of players
  r : ℕ  -- Number of matches played among the 3 withdrawn players

/-- The number of matches played by remaining players -/
def remainingMatches (t : TableTennisTournament) : ℕ :=
  (t.n - 3) * (t.n - 4) / 2

/-- The total number of matches played in the tournament -/
def totalMatches (t : TableTennisTournament) : ℕ :=
  remainingMatches t + (3 * 2 - t.r)

/-- Theorem stating the number of matches played among withdrawn players -/
theorem withdrawn_players_matches (t : TableTennisTournament) : 
  t.n > 3 ∧ totalMatches t = 50 → t.r = 1 := by sorry

end NUMINAMATH_CALUDE_withdrawn_players_matches_l989_98950


namespace NUMINAMATH_CALUDE_norm_took_110_photos_l989_98928

/-- The number of photos taken by Norm given the conditions of the problem -/
def norm_photos (lisa mike norm : ℕ) : Prop :=
  (lisa + mike = mike + norm - 60) ∧ 
  (norm = 2 * lisa + 10) ∧
  (norm = 110)

/-- Theorem stating that Norm took 110 photos given the problem conditions -/
theorem norm_took_110_photos :
  ∃ (lisa mike norm : ℕ), norm_photos lisa mike norm :=
by
  sorry

end NUMINAMATH_CALUDE_norm_took_110_photos_l989_98928


namespace NUMINAMATH_CALUDE_range_of_a_l989_98919

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 + 5*a*x + 6*a^2 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = A) → a < 0 → -1/2 ≥ a ∧ a > -4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l989_98919


namespace NUMINAMATH_CALUDE_marys_max_earnings_l989_98984

/-- Calculates the maximum weekly earnings for a worker with the given parameters. -/
def max_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (overtime_rate_increase : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that Mary's maximum weekly earnings are $410 -/
theorem marys_max_earnings :
  max_weekly_earnings 45 20 8 (1/4) = 410 := by
  sorry

#eval max_weekly_earnings 45 20 8 (1/4)

end NUMINAMATH_CALUDE_marys_max_earnings_l989_98984


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l989_98914

theorem problem_1 : 2023^2 - 2024 * 2022 = 1 := by sorry

theorem problem_2 (a b c : ℝ) : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l989_98914


namespace NUMINAMATH_CALUDE_correct_average_l989_98983

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℕ) :
  n = 10 ∧ initial_avg = 14 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n * initial_avg + (correct_num - wrong_num)) / n = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l989_98983


namespace NUMINAMATH_CALUDE_quadratic_a_value_main_quadratic_theorem_l989_98926

/-- A quadratic function with vertex form y = a(x - h)^2 + k, where (h, k) is the vertex -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The theorem stating the value of 'a' for a quadratic function with given properties -/
theorem quadratic_a_value (f : QuadraticFunction) 
  (vertex_condition : f.h = -3 ∧ f.k = 0)
  (point_condition : f.a * (2 - f.h)^2 + f.k = -36) :
  f.a = -36/25 := by
  sorry

/-- The main theorem proving the value of 'a' for the given quadratic function -/
theorem main_quadratic_theorem :
  ∃ f : QuadraticFunction, 
    f.h = -3 ∧ 
    f.k = 0 ∧ 
    f.a * (2 - f.h)^2 + f.k = -36 ∧
    f.a = -36/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_a_value_main_quadratic_theorem_l989_98926


namespace NUMINAMATH_CALUDE_zach_savings_l989_98962

/-- Represents the financial situation of Zach saving for a bike --/
structure BikeSavings where
  bikeCost : ℕ
  weeklyAllowance : ℕ
  lawnMowingPay : ℕ
  babysittingRate : ℕ
  babysittingHours : ℕ
  additionalNeeded : ℕ

/-- Calculates the amount Zach has already saved --/
def amountSaved (s : BikeSavings) : ℕ :=
  s.bikeCost - (s.weeklyAllowance + s.lawnMowingPay + s.babysittingRate * s.babysittingHours) - s.additionalNeeded

/-- Theorem stating that for Zach's specific situation, he has already saved $65 --/
theorem zach_savings : 
  let s : BikeSavings := {
    bikeCost := 100,
    weeklyAllowance := 5,
    lawnMowingPay := 10,
    babysittingRate := 7,
    babysittingHours := 2,
    additionalNeeded := 6
  }
  amountSaved s = 65 := by sorry

end NUMINAMATH_CALUDE_zach_savings_l989_98962


namespace NUMINAMATH_CALUDE_exotic_courses_divisibility_l989_98966

/-- Represents a country with airports -/
structure Country where
  airports : ℕ

/-- Represents the flight system between two countries -/
structure FlightSystem where
  countryA : Country
  countryB : Country
  flightsPerAirport : ℕ
  noInternalFlights : Bool

/-- Represents an exotic traveling course -/
structure ExoticTravelingCourse where
  flightSystem : FlightSystem
  courseLength : ℕ

/-- The number of all exotic traveling courses -/
def numberOfExoticCourses (f : FlightSystem) : ℕ :=
  sorry

theorem exotic_courses_divisibility (f : FlightSystem) 
  (h1 : f.countryA.airports = f.countryB.airports)
  (h2 : f.countryA.airports ≥ 2)
  (h3 : f.flightsPerAirport = 3)
  (h4 : f.noInternalFlights = true) :
  ∃ k : ℕ, numberOfExoticCourses f = 8 * f.countryA.airports * k ∧ Even k :=
sorry

end NUMINAMATH_CALUDE_exotic_courses_divisibility_l989_98966


namespace NUMINAMATH_CALUDE_three_equal_numbers_sum_300_l989_98975

theorem three_equal_numbers_sum_300 :
  ∃ (x : ℕ), x + x + x = 300 ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_three_equal_numbers_sum_300_l989_98975


namespace NUMINAMATH_CALUDE_probability_A_equals_B_l989_98908

open Set
open MeasureTheory
open Real

-- Define the set of valid pairs (a, b)
def ValidPairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, b) := p
               cos (cos a) = cos (cos b) ∧
               -5*π/2 ≤ a ∧ a ≤ 5*π/2 ∧
               -5*π/2 ≤ b ∧ b ≤ 5*π/2}

-- Define the set of pairs where A = B
def EqualPairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, b) := p; a = b}

-- Define the probability measure on ValidPairs
noncomputable def ProbMeasure : Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_A_equals_B :
  ProbMeasure (ValidPairs ∩ EqualPairs) / ProbMeasure ValidPairs = 1/5 :=
sorry

end NUMINAMATH_CALUDE_probability_A_equals_B_l989_98908


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l989_98940

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a^2 = b^2 + c^2 → Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l989_98940


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l989_98992

/-- Configuration of tangent semicircles and circle -/
structure TangentConfiguration where
  R : ℝ  -- Radius of the large semicircle
  r : ℝ  -- Radius of the circle
  x : ℝ  -- Radius of the small semicircle
  tangent : R > 0 ∧ r > 0 ∧ x > 0  -- All radii are positive
  large_semicircle : R = 12  -- Large semicircle has radius 12
  circle : r = 6  -- Circle has radius 6
  pythagorean : r^2 + (R - x)^2 = (r + x)^2  -- Pythagorean theorem for tangent configuration

/-- The radius of the small semicircle in the tangent configuration is 4 -/
theorem small_semicircle_radius (config : TangentConfiguration) : config.x = 4 :=
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l989_98992


namespace NUMINAMATH_CALUDE_difference_of_squares_l989_98931

theorem difference_of_squares (n : ℕ) : 
  (n = 105 → ∃! k : ℕ, k = 4 ∧ ∃ s : Finset (ℕ × ℕ), s.card = k ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ x^2 - y^2 = n) ∧
  (n = 106 → ¬∃ (x y : ℕ), x^2 - y^2 = n) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l989_98931


namespace NUMINAMATH_CALUDE_cost_in_usd_l989_98963

/-- The cost of coffee and snack in USD given their prices in yen and the exchange rate -/
theorem cost_in_usd (coffee_yen : ℕ) (snack_yen : ℕ) (exchange_rate : ℚ) : 
  coffee_yen = 250 → snack_yen = 150 → exchange_rate = 1 / 100 →
  (coffee_yen + snack_yen : ℚ) * exchange_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_in_usd_l989_98963


namespace NUMINAMATH_CALUDE_contingency_and_sampling_theorem_l989_98991

/-- Represents the contingency table --/
structure ContingencyTable :=
  (male_running : ℕ)
  (male_not_running : ℕ)
  (female_running : ℕ)
  (female_not_running : ℕ)

/-- Calculates the K^2 value for the contingency table --/
def calculate_k_squared (table : ContingencyTable) : ℚ :=
  let n := table.male_running + table.male_not_running + table.female_running + table.female_not_running
  let a := table.male_running
  let b := table.male_not_running
  let c := table.female_running
  let d := table.female_not_running
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Calculates the expected value of females selected in the sampling process --/
def expected_females_selected (male_count female_count : ℕ) : ℚ :=
  (0 * (male_count * (male_count - 1)) + 
   1 * (2 * male_count * female_count) + 
   2 * (female_count * (female_count - 1))) / 
  ((male_count + female_count) * (male_count + female_count - 1))

/-- Main theorem to prove --/
theorem contingency_and_sampling_theorem 
  (table : ContingencyTable) 
  (h_total : table.male_running + table.male_not_running + table.female_running + table.female_not_running = 80)
  (h_male_running : table.male_running = 20)
  (h_male_not_running : table.male_not_running = 20)
  (h_female_not_running : table.female_not_running = 30) :
  calculate_k_squared table < (6635 : ℚ) / 1000 ∧ 
  expected_females_selected 2 3 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_contingency_and_sampling_theorem_l989_98991


namespace NUMINAMATH_CALUDE_total_legs_farmer_brown_l989_98938

/-- The number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- The number of legs for a sheep -/
def sheep_legs : ℕ := 4

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := 5

/-- Theorem stating the total number of legs among the animals Farmer Brown fed -/
theorem total_legs_farmer_brown : 
  num_chickens * chicken_legs + num_sheep * sheep_legs = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_farmer_brown_l989_98938


namespace NUMINAMATH_CALUDE_max_value_quadratic_l989_98912

theorem max_value_quadratic (x y : ℝ) : 
  4 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≤ -13 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l989_98912


namespace NUMINAMATH_CALUDE_remainder_divisibility_l989_98937

theorem remainder_divisibility (N : ℕ) (h : N % 125 = 40) : N % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l989_98937


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_PQ_l989_98996

def P : ℝ × ℝ := (4, 0)
def Q : ℝ × ℝ := (0, 2)

theorem circle_equation_with_diameter_PQ :
  let center := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let radius_squared := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_squared ↔
    (x - 2)^2 + (y - 1)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_circle_equation_with_diameter_PQ_l989_98996


namespace NUMINAMATH_CALUDE_line_intercepts_l989_98995

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

/-- The x-axis intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-axis intercept of the line -/
def y_intercept : ℝ := -3

/-- Theorem: The x-intercept and y-intercept of the line 3x - 2y - 6 = 0 are 2 and -3 respectively -/
theorem line_intercepts : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept :=
sorry

end NUMINAMATH_CALUDE_line_intercepts_l989_98995


namespace NUMINAMATH_CALUDE_expected_value_of_product_l989_98986

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def product_sum : ℕ := (marbles.powerset.filter (fun s => s.card = 2)).sum (fun s => s.prod id)

def total_combinations : ℕ := Nat.choose 6 2

theorem expected_value_of_product :
  (product_sum : ℚ) / total_combinations = 35 / 3 :=
sorry

end NUMINAMATH_CALUDE_expected_value_of_product_l989_98986


namespace NUMINAMATH_CALUDE_employee_hire_year_l989_98960

/-- Rule of 70 provision: An employee can retire when their age plus years of employment is at least 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1968

/-- The age at which the employee was hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible to retire -/
def retirement_year : ℕ := 2006

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) hire_age ∧
  ∀ y, y > hire_year → ¬rule_of_70 (hire_age + (y - hire_year)) hire_age :=
by sorry

end NUMINAMATH_CALUDE_employee_hire_year_l989_98960


namespace NUMINAMATH_CALUDE_discount_percentage_l989_98921

theorem discount_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 150)
  (h2 : sale_price = 135) :
  (original_price - sale_price) / original_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l989_98921


namespace NUMINAMATH_CALUDE_vector_dot_product_equals_22_l989_98939

/-- Given two vectors AB and BC in ℝ², where BC has a magnitude of √10,
    prove that the dot product of AB and AC equals 22. -/
theorem vector_dot_product_equals_22 
  (AB : ℝ × ℝ) 
  (BC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : ∃ t > 0, BC = (3, t)) 
  (h3 : Real.sqrt ((BC.1)^2 + (BC.2)^2) = Real.sqrt 10) : 
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_equals_22_l989_98939


namespace NUMINAMATH_CALUDE_x_value_proof_l989_98972

theorem x_value_proof (x : ℕ) 
  (h1 : x > 0) 
  (h2 : (x * x) / 100 = 16) 
  (h3 : 4 ∣ x) : 
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l989_98972


namespace NUMINAMATH_CALUDE_expand_and_simplify_l989_98985

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  3 / 7 * (7 / x + 14 * x^3) = 3 / x + 6 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l989_98985


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l989_98900

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) →
  (a 1 + (2 * a 2 - a 1) / 2 = a 3 / 2) →
  (a 10 + a 11) / (a 8 + a 9) = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l989_98900


namespace NUMINAMATH_CALUDE_john_completion_time_l989_98997

/-- The time it takes for John to complete the task alone -/
def john_time : ℝ := 20

/-- The time it takes for Jane to complete the task alone -/
def jane_time : ℝ := 10

/-- The total time they worked together -/
def total_time : ℝ := 10

/-- The time Jane worked before stopping -/
def jane_work_time : ℝ := 5

theorem john_completion_time :
  (jane_work_time * (1 / john_time + 1 / jane_time) + (total_time - jane_work_time) * (1 / john_time) = 1) →
  john_time = 20 := by
sorry

end NUMINAMATH_CALUDE_john_completion_time_l989_98997


namespace NUMINAMATH_CALUDE_max_cookies_andy_l989_98956

/-- The number of cookies baked by the siblings -/
def total_cookies : ℕ := 36

/-- Andy's cookies -/
def andy_cookies : ℕ → ℕ := λ x => x

/-- Aaron's cookies -/
def aaron_cookies : ℕ → ℕ := λ x => 2 * x

/-- Alexa's cookies -/
def alexa_cookies : ℕ → ℕ := λ x => total_cookies - x - 2 * x

/-- The maximum number of cookies Andy could have eaten -/
def max_andy_cookies : ℕ := 12

theorem max_cookies_andy :
  ∀ x : ℕ,
  x ≤ max_andy_cookies ∧
  andy_cookies x + aaron_cookies x + alexa_cookies x = total_cookies ∧
  alexa_cookies x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l989_98956


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_volume_l989_98953

/-- The volume of a regular hexagonal pyramid with base side length a and lateral surface area 10 times larger than the base area -/
theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) : 
  let base_area := (3 * Real.sqrt 3 / 2) * a^2
  let lateral_area := 10 * base_area
  let height := (3 * a * Real.sqrt 33) / 2
  let volume := (1 / 3) * base_area * height
  volume = (9 * a^3 * Real.sqrt 11) / 4 := by
sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_volume_l989_98953


namespace NUMINAMATH_CALUDE_min_value_theorem_l989_98902

/-- Given a function y = a^x + b where b > 0, a > 1, and 3 = a + b, 
    the minimum value of (4 / (a - 1)) + (1 / b) is 9/2 -/
theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a > 1) (h3 : 3 = a + b) :
  (∀ x : ℝ, (4 / (a - 1)) + (1 / b) ≥ 9/2) ∧ 
  (∃ x : ℝ, (4 / (a - 1)) + (1 / b) = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l989_98902


namespace NUMINAMATH_CALUDE_total_birds_in_pet_store_l989_98959

/-- Represents the number of birds in a cage -/
structure CageBirds where
  parrots : Nat
  finches : Nat
  canaries : Nat
  parakeets : Nat

/-- The pet store's bird inventory -/
def petStore : List CageBirds := [
  { parrots := 9, finches := 4, canaries := 7, parakeets := 0 },
  { parrots := 5, finches := 10, canaries := 0, parakeets := 8 },
  { parrots := 0, finches := 7, canaries := 3, parakeets := 15 },
  { parrots := 10, finches := 12, canaries := 0, parakeets := 5 }
]

/-- Calculates the total number of birds in a cage -/
def totalBirdsInCage (cage : CageBirds) : Nat :=
  cage.parrots + cage.finches + cage.canaries + cage.parakeets

/-- Theorem: The total number of birds in the pet store is 95 -/
theorem total_birds_in_pet_store :
  (petStore.map totalBirdsInCage).sum = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_pet_store_l989_98959


namespace NUMINAMATH_CALUDE_inequality_solution_l989_98970

-- Define the inequality function
def f (x : ℝ) : ℝ := (x^2 - 4) * (x - 6)^2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

-- Theorem stating that the solution set is correct
theorem inequality_solution : 
  {x : ℝ | f x ≤ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l989_98970


namespace NUMINAMATH_CALUDE_katya_age_l989_98977

/-- Represents the ages of the children in the family -/
structure FamilyAges where
  anya : ℕ
  katya : ℕ
  vasya : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.anya + ages.katya = 19 ∧
  ages.anya + ages.vasya = 14 ∧
  ages.katya + ages.vasya = 7

/-- The theorem to prove Katya's age -/
theorem katya_age (ages : FamilyAges) (h : satisfiesConditions ages) : ages.katya = 6 := by
  sorry


end NUMINAMATH_CALUDE_katya_age_l989_98977


namespace NUMINAMATH_CALUDE_tan_triple_angle_l989_98949

theorem tan_triple_angle (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l989_98949


namespace NUMINAMATH_CALUDE_wall_length_is_850_l989_98913

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

/-- The main theorem stating that under given conditions, the wall length is 850 cm -/
theorem wall_length_is_850 (brick : BrickDimensions)
    (wall : WallDimensions) (num_bricks : ℕ) :
    brick.length = 25 →
    brick.width = 11.25 →
    brick.height = 6 →
    wall.width = 600 →
    wall.height = 22.5 →
    num_bricks = 6800 →
    brickVolume brick * num_bricks = wallVolume wall →
    wall.length = 850 := by
  sorry


end NUMINAMATH_CALUDE_wall_length_is_850_l989_98913


namespace NUMINAMATH_CALUDE_root_ratio_sum_zero_l989_98911

theorem root_ratio_sum_zero (a b n p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) 
  (h_roots : ∃ (x y : ℝ), x / y = a / b ∧ p * x^2 + n * x + n = 0 ∧ p * y^2 + n * y + n = 0) :
  Real.sqrt (a / b) + Real.sqrt (b / a) + Real.sqrt (n / p) = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_ratio_sum_zero_l989_98911


namespace NUMINAMATH_CALUDE_shelter_cats_count_l989_98989

theorem shelter_cats_count (total animals : ℕ) (cats dogs : ℕ) : 
  total = 60 →
  cats = dogs + 20 →
  cats + dogs = total →
  cats = 40 :=
by sorry

end NUMINAMATH_CALUDE_shelter_cats_count_l989_98989


namespace NUMINAMATH_CALUDE_problem_solution_l989_98933

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem problem_solution :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c (-x) = -(f a b c x)) →
  f a b c 1 = 3 →
  f a b c 2 = 12 →
  (a = 1 ∧ b = 0 ∧ c = 2) ∧
  (∀ x y : ℝ, x < y → f a b c x < f a b c y) ∧
  (∀ m n : ℝ, m^3 - 3*m^2 + 5*m = 5 → n^3 - 3*n^2 + 5*n = 1 → m + n = 2) ∧
  (∀ k : ℝ, (∀ x : ℝ, 0 < x ∧ x < 1 → f a b c (x^2 - 4) + f a b c (k*x + 2*k) < 0) → k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l989_98933


namespace NUMINAMATH_CALUDE_function_extrema_l989_98917

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 4*x + 4

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4

-- Theorem statement
theorem function_extrema (a : ℝ) : 
  (f' a 1 = -3) → 
  (a = 1/3) ∧ 
  (∀ x, f (1/3) x ≤ 28/3) ∧ 
  (∀ x, f (1/3) x ≥ -4/3) ∧
  (∃ x, f (1/3) x = 28/3) ∧ 
  (∃ x, f (1/3) x = -4/3) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_l989_98917


namespace NUMINAMATH_CALUDE_limit_x_plus_sin_x_power_sin_x_plus_x_l989_98905

/-- The limit of (x + sin x)^(sin x + x) as x approaches π is π^π. -/
theorem limit_x_plus_sin_x_power_sin_x_plus_x (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π| ∧ |x - π| < δ →
    |(x + Real.sin x)^(Real.sin x + x) - π^π| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_plus_sin_x_power_sin_x_plus_x_l989_98905


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l989_98952

/-- Given a quadratic equation x^2 - 6x + 4 = 0, 
    its equivalent form using the completing the square method is (x - 3)^2 = 5 -/
theorem quadratic_completing_square : 
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l989_98952


namespace NUMINAMATH_CALUDE_non_constant_polynomial_not_always_palindrome_l989_98935

/-- A number is a palindrome if it reads the same from left to right as it reads from right to left in base 10 -/
def is_palindrome (n : ℤ) : Prop := sorry

/-- The theorem states that for any non-constant polynomial with integer coefficients, 
    there exists a positive integer n such that p(n) is not a palindrome number -/
theorem non_constant_polynomial_not_always_palindrome 
  (p : Polynomial ℤ) (h : ¬ (p.degree = 0)) : 
  ∃ (n : ℕ), ¬ is_palindrome (p.eval n) := by sorry

end NUMINAMATH_CALUDE_non_constant_polynomial_not_always_palindrome_l989_98935


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l989_98999

theorem sum_of_reciprocal_squares (a b c : ℝ) : 
  a^3 - 12*a^2 + 14*a + 3 = 0 →
  b^3 - 12*b^2 + 14*b + 3 = 0 →
  c^3 - 12*c^2 + 14*c + 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 268/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l989_98999


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l989_98918

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / (a^2 - 2*a + 1) / (1 + 2 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l989_98918


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l989_98932

theorem no_integer_tangent_length (t₁ m : ℕ) : 
  (∃ (m : ℕ), m % 2 = 1 ∧ m < 24 ∧ t₁^2 = m * (24 - m)) → False :=
sorry

end NUMINAMATH_CALUDE_no_integer_tangent_length_l989_98932


namespace NUMINAMATH_CALUDE_markup_rate_l989_98980

theorem markup_rate (S : ℝ) (h1 : S > 0) : 
  let profit := 0.20 * S
  let expenses := 0.20 * S
  let cost := S - profit - expenses
  (S - cost) / cost * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_markup_rate_l989_98980


namespace NUMINAMATH_CALUDE_investment_plans_count_l989_98958

/-- The number of ways to distribute projects across cities -/
def distribute_projects (num_projects : ℕ) (num_cities : ℕ) (max_per_city : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of investment plans -/
theorem investment_plans_count : distribute_projects 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l989_98958
