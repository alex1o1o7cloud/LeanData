import Mathlib

namespace NUMINAMATH_CALUDE_completely_overlapping_implies_congruent_l2986_298675

/-- Two triangles are completely overlapping if all their corresponding vertices coincide. -/
def CompletelyOverlapping (T1 T2 : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T1 ↔ p ∈ T2

/-- Two triangles are congruent if they have the same size and shape. -/
def Congruent (T1 T2 : Set (ℝ × ℝ)) : Prop :=
  ∃ f : ℝ × ℝ → ℝ × ℝ, Isometry f ∧ f '' T1 = T2

/-- If two triangles completely overlap, then they are congruent. -/
theorem completely_overlapping_implies_congruent
  (T1 T2 : Set (ℝ × ℝ)) (h : CompletelyOverlapping T1 T2) :
  Congruent T1 T2 := by
  sorry

end NUMINAMATH_CALUDE_completely_overlapping_implies_congruent_l2986_298675


namespace NUMINAMATH_CALUDE_broomstick_race_permutations_l2986_298693

theorem broomstick_race_permutations :
  let n : ℕ := 4  -- number of participants
  ∀ (participants : Finset (Fin n)),  -- set of participants
  Finset.card participants = n →  -- ensure we have exactly 4 participants
  Fintype.card (Equiv.Perm participants) = 24 :=  -- number of permutations is 24
by
  sorry

end NUMINAMATH_CALUDE_broomstick_race_permutations_l2986_298693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2986_298696

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2986_298696


namespace NUMINAMATH_CALUDE_counterexample_exists_l2986_298659

-- Define the set of numbers to check
def numbers : List Nat := [25, 35, 39, 49, 51]

-- Define what it means for a number to be composite
def isComposite (n : Nat) : Prop := ¬ Nat.Prime n

-- Define the counterexample property
def isCounterexample (n : Nat) : Prop := isComposite n ∧ Nat.Prime (n - 2)

-- Theorem to prove
theorem counterexample_exists : ∃ n ∈ numbers, isCounterexample n := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2986_298659


namespace NUMINAMATH_CALUDE_m_range_l2986_298691

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem m_range (m : ℝ) : (¬(p m ∨ q m)) → m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_m_range_l2986_298691


namespace NUMINAMATH_CALUDE_tuesday_lost_revenue_l2986_298648

/-- Represents a movie theater with its capacity, ticket price, and tickets sold. -/
structure MovieTheater where
  capacity : ℕ
  ticketPrice : ℚ
  ticketsSold : ℕ

/-- Calculates the lost revenue for a movie theater. -/
def lostRevenue (theater : MovieTheater) : ℚ :=
  (theater.capacity - theater.ticketsSold) * theater.ticketPrice

/-- Theorem stating that the lost revenue for the given theater scenario is $208.00. -/
theorem tuesday_lost_revenue :
  let theater : MovieTheater := ⟨50, 8, 24⟩
  lostRevenue theater = 208 := by sorry

end NUMINAMATH_CALUDE_tuesday_lost_revenue_l2986_298648


namespace NUMINAMATH_CALUDE_graph_properties_of_y_squared_equals_sin_x_squared_l2986_298607

theorem graph_properties_of_y_squared_equals_sin_x_squared :
  ∃ f : ℝ → Set ℝ, 
    (∀ x y, y ∈ f x ↔ y^2 = Real.sin (x^2)) ∧ 
    (0 ∈ f 0) ∧ 
    (∀ x y, y ∈ f x → -y ∈ f x) ∧
    (∀ x, (∃ y, y ∈ f x) → Real.sin (x^2) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_graph_properties_of_y_squared_equals_sin_x_squared_l2986_298607


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l2986_298623

theorem quadratic_equation_transformation (x : ℝ) :
  x^2 + 6*x - 1 = 0 →
  ∃ (m n : ℝ), (x + m)^2 = n ∧ m - n = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l2986_298623


namespace NUMINAMATH_CALUDE_base_nine_subtraction_l2986_298601

/-- Represents a number in base 9 --/
def BaseNine : Type := ℕ

/-- Converts a base 9 number to its decimal (base 10) representation --/
def to_decimal (n : BaseNine) : ℕ := sorry

/-- Converts a decimal (base 10) number to its base 9 representation --/
def from_decimal (n : ℕ) : BaseNine := sorry

/-- Subtracts two base 9 numbers --/
def base_nine_sub (a b : BaseNine) : BaseNine := sorry

/-- The main theorem to prove --/
theorem base_nine_subtraction :
  base_nine_sub (from_decimal 256) (from_decimal 143) = from_decimal 113 := by sorry

end NUMINAMATH_CALUDE_base_nine_subtraction_l2986_298601


namespace NUMINAMATH_CALUDE_triangle_problem_l2986_298687

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (Real.tan t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B)) →
  (Real.sin (t.B - t.A) = Real.cos t.C) →
  (t.A = π/4 ∧ t.C = π/3) ∧
  (((1/2) * t.a * t.c * Real.sin t.B = 3 + Real.sqrt 3) →
   (t.a = 2 * Real.sqrt 2 ∧ t.c = 2 * Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l2986_298687


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l2986_298637

/-- Function f(x) = |x - 1| + |x + 2| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

/-- Function g(x) = |x + 1| - |x - a| + a -/
def g (a x : ℝ) : ℝ := |x + 1| - |x - a| + a

/-- The solution set of f(x) + g(x) < 6 when a = 1 is (-4, 1) -/
theorem solution_set_theorem :
  {x : ℝ | f x + g 1 x < 6} = Set.Ioo (-4) 1 := by sorry

/-- For any real numbers x₁ and x₂, f(x₁) ≥ g(x₂) if and only if a ∈ (-∞, 1] -/
theorem range_of_a_theorem :
  ∀ (a : ℝ), (∀ (x₁ x₂ : ℝ), f x₁ ≥ g a x₂) ↔ a ∈ Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l2986_298637


namespace NUMINAMATH_CALUDE_problem_solution_l2986_298692

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x - 11
def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem problem_solution :
  ∀ a k : ℝ,
  (f' a (-1) = 0 → a = -2) ∧
  (∃ x y : ℝ, f a x = k * x + 9 ∧ f' a x = k ∧ g x = k * x + 9 ∧ (3 * 2 * x + 6) = k → k = 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2986_298692


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_water_percentage_is_90_l2986_298643

/-- The percentage of water in fresh grapes, given the conditions of the drying process -/
theorem water_percentage_in_fresh_grapes : ℝ → Prop :=
  fun p =>
    let fresh_weight : ℝ := 25
    let dried_weight : ℝ := 3.125
    let dried_water_percentage : ℝ := 20
    let fresh_solid_content : ℝ := fresh_weight * (100 - p) / 100
    let dried_solid_content : ℝ := dried_weight * (100 - dried_water_percentage) / 100
    fresh_solid_content = dried_solid_content →
    p = 90

/-- The theorem stating that the water percentage in fresh grapes is 90% -/
theorem water_percentage_is_90 : water_percentage_in_fresh_grapes 90 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_water_percentage_is_90_l2986_298643


namespace NUMINAMATH_CALUDE_line_segment_length_l2986_298677

/-- The length of a line segment with endpoints (1,2) and (4,10) is √73. -/
theorem line_segment_length : Real.sqrt ((4 - 1)^2 + (10 - 2)^2) = Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l2986_298677


namespace NUMINAMATH_CALUDE_junior_teachers_sampled_count_l2986_298638

/-- Represents the number of teachers in each category -/
structure TeacherCounts where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the sample size for stratified sampling -/
def SampleSize : Nat := 50

/-- Calculates the number of junior teachers in a stratified sample -/
def juniorTeachersSampled (counts : TeacherCounts) (sampleSize : Nat) : Nat :=
  (sampleSize * counts.junior) / counts.total

/-- Theorem: The number of junior teachers sampled is 20 -/
theorem junior_teachers_sampled_count 
  (counts : TeacherCounts) 
  (h1 : counts.total = 200)
  (h2 : counts.senior = 20)
  (h3 : counts.intermediate = 100)
  (h4 : counts.junior = 80) :
  juniorTeachersSampled counts SampleSize = 20 := by
  sorry

#eval juniorTeachersSampled { total := 200, senior := 20, intermediate := 100, junior := 80 } SampleSize

end NUMINAMATH_CALUDE_junior_teachers_sampled_count_l2986_298638


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2986_298610

theorem perfect_square_condition (n : ℤ) : 
  (∃ k : ℤ, n^2 + 6*n + 1 = k^2) ↔ (n = -6 ∨ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2986_298610


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2986_298672

/-- Given a geometric sequence {a_n} where a₃a₅a₇a₉a₁₁ = 243, prove that a₁₀² / a₁₃ = 3 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_product : a 3 * a 5 * a 7 * a 9 * a 11 = 243) :
  a 10 ^ 2 / a 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2986_298672


namespace NUMINAMATH_CALUDE_smallest_n_perfect_powers_l2986_298652

theorem smallest_n_perfect_powers : ∃ (n : ℕ), 
  (n = 151875) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (∃ k : ℕ, 3 * m = k^2) → 
    (∃ l : ℕ, 5 * m = l^5) → False) ∧
  (∃ k : ℕ, 3 * n = k^2) ∧
  (∃ l : ℕ, 5 * n = l^5) := by
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_powers_l2986_298652


namespace NUMINAMATH_CALUDE_sum_remainder_l2986_298690

theorem sum_remainder (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a % 13 = 3 → b % 13 = 5 → c % 13 = 7 → d % 13 = 9 → e % 13 = 12 →
  (a + b + c + d + e) % 13 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l2986_298690


namespace NUMINAMATH_CALUDE_circle_square_radius_l2986_298653

theorem circle_square_radius (s : ℝ) (r : ℝ) : 
  s^2 = 9/16 →                  -- Area of the square is 9/16
  π * r^2 = 9/16 →              -- Area of the circle is 9/16
  2 * r = s →                   -- Diameter of circle equals side length of square
  r = 3/8 := by                 -- Radius of the circle is 3/8
sorry

end NUMINAMATH_CALUDE_circle_square_radius_l2986_298653


namespace NUMINAMATH_CALUDE_no_even_rectangle_with_sum_120_l2986_298636

/-- Represents a rectangle with positive even integer side lengths -/
structure EvenRectangle where
  length : ℕ
  width : ℕ
  length_positive : length > 0
  width_positive : width > 0
  length_even : Even length
  width_even : Even width

/-- Calculates the area of an EvenRectangle -/
def area (r : EvenRectangle) : ℕ := r.length * r.width

/-- Calculates the modified perimeter of an EvenRectangle -/
def modifiedPerimeter (r : EvenRectangle) : ℕ := 2 * (r.length + r.width) + 6

/-- Theorem stating that there's no EvenRectangle with A + P' = 120 -/
theorem no_even_rectangle_with_sum_120 :
  ∀ r : EvenRectangle, area r + modifiedPerimeter r ≠ 120 := by
  sorry

end NUMINAMATH_CALUDE_no_even_rectangle_with_sum_120_l2986_298636


namespace NUMINAMATH_CALUDE_candy_distribution_solution_l2986_298670

def candy_distribution (n : ℕ) : Prop :=
  let initial_candy : ℕ := 120
  let first_phase_passes : ℕ := 40
  let first_phase_candy := first_phase_passes
  let second_phase_candy := initial_candy - first_phase_candy
  let total_passes := first_phase_passes + (second_phase_candy / 2)
  (n ∣ total_passes) ∧ (n > 0) ∧ (n ≤ total_passes)

theorem candy_distribution_solution :
  candy_distribution 40 ∧ ∀ m : ℕ, m ≠ 40 → ¬(candy_distribution m) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_solution_l2986_298670


namespace NUMINAMATH_CALUDE_pasture_rent_is_175_l2986_298633

/-- Represents the rent share of a person based on their oxen and months of grazing -/
structure RentShare where
  oxen : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given the rent shares and one known payment -/
def calculateTotalRent (shares : List RentShare) (knownShare : RentShare) (knownPayment : ℕ) : ℕ :=
  let totalOxenMonths := shares.foldl (fun acc s => acc + s.oxen * s.months) 0
  let knownShareOxenMonths := knownShare.oxen * knownShare.months
  (totalOxenMonths * knownPayment) / knownShareOxenMonths

/-- Theorem: The total rent of the pasture is 175 given the problem conditions -/
theorem pasture_rent_is_175 :
  let shares := [
    RentShare.mk 10 7,  -- A's share
    RentShare.mk 12 5,  -- B's share
    RentShare.mk 15 3   -- C's share
  ]
  let knownShare := RentShare.mk 15 3  -- C's share
  let knownPayment := 45  -- C's payment
  calculateTotalRent shares knownShare knownPayment = 175 := by
  sorry


end NUMINAMATH_CALUDE_pasture_rent_is_175_l2986_298633


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2986_298686

theorem polynomial_factorization (m : ℤ) : 
  (∃ (a b c d e f : ℤ), ∀ (x y : ℤ), 
    x^2 + 2*x*y + 2*x + m*y + 2*m = (a*x + b*y + c) * (d*x + e*y + f)) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2986_298686


namespace NUMINAMATH_CALUDE_parabola_directrix_l2986_298621

/-- Given a parabola defined by x = -1/8 * y^2, its directrix is x = 1/2 -/
theorem parabola_directrix (y : ℝ) : 
  let x := -1/8 * y^2
  let a := -1/8
  let focus_x := 1 / (4 * a)
  let directrix_x := -focus_x
  directrix_x = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2986_298621


namespace NUMINAMATH_CALUDE_probability_two_black_two_white_l2986_298662

def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4
def drawn_black : ℕ := 2
def drawn_white : ℕ := 2

theorem probability_two_black_two_white :
  (Nat.choose black_balls drawn_black * Nat.choose white_balls drawn_white) /
  Nat.choose total_balls drawn_balls = 7 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_two_white_l2986_298662


namespace NUMINAMATH_CALUDE_average_daily_attendance_l2986_298695

/-- Calculates the average daily attendance for a week given the attendance data --/
theorem average_daily_attendance
  (monday : ℕ)
  (tuesday : ℕ)
  (wednesday_friday : ℕ)
  (saturday : ℕ)
  (sunday : ℕ)
  (absent_monday_join_wednesday : ℕ)
  (h1 : monday = 10)
  (h2 : tuesday = 15)
  (h3 : wednesday_friday = 10)
  (h4 : saturday = 8)
  (h5 : sunday = 12)
  (h6 : absent_monday_join_wednesday = 3)
  : (monday + tuesday + (wednesday_friday * 3 + absent_monday_join_wednesday) + saturday + sunday) / 7 = 78 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_attendance_l2986_298695


namespace NUMINAMATH_CALUDE_john_initial_payment_l2986_298642

def soda_cost : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

theorem john_initial_payment :
  num_sodas * soda_cost + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_initial_payment_l2986_298642


namespace NUMINAMATH_CALUDE_digit_equation_solution_l2986_298635

theorem digit_equation_solution :
  ∀ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 →
    100 * A + 10 * B + C = 3 * (A + B + C) + 294 →
    (A + B + C) * (100 * A + 10 * B + C) = 2295 →
    A = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l2986_298635


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2986_298604

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a5_eq_3 : a 5 = 3
  a4_times_a7_eq_45 : a 4 * a 7 = 45

/-- The main theorem about the specific ratio in the geometric sequence -/
theorem geometric_sequence_ratio
  (seq : GeometricSequence) :
  (seq.a 7 - seq.a 9) / (seq.a 5 - seq.a 7) = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2986_298604


namespace NUMINAMATH_CALUDE_element_in_M_l2986_298688

def M : Set (ℕ × ℕ) := {(2, 3)}

theorem element_in_M : (2, 3) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_M_l2986_298688


namespace NUMINAMATH_CALUDE_expression_simplification_l2986_298614

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  ((x * y + 2) * (x * y - 2) + (x * y - 2)^2) / (x * y) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2986_298614


namespace NUMINAMATH_CALUDE_train_speed_l2986_298616

/-- Proves that a train with given length, crossing a platform of given length in a given time, has a specific speed in km/h -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) : 
  train_length = 450 ∧ 
  platform_length = 250.056 ∧ 
  crossing_time = 20 →
  (train_length + platform_length) / crossing_time * 3.6 = 126.01008 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l2986_298616


namespace NUMINAMATH_CALUDE_experts_win_probability_l2986_298644

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expert_score : ℕ := 3

/-- The current score of Viewers -/
def viewer_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability of Experts winning the game from the current state -/
theorem experts_win_probability : 
  p^4 + 4 * p^3 * q = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_l2986_298644


namespace NUMINAMATH_CALUDE_age_difference_l2986_298657

theorem age_difference (masc_age sam_age : ℕ) : 
  masc_age > sam_age →
  masc_age + sam_age = 27 →
  masc_age = 17 →
  sam_age = 10 →
  masc_age - sam_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2986_298657


namespace NUMINAMATH_CALUDE_certain_positive_integer_value_l2986_298650

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem certain_positive_integer_value :
  ∀ (i k m n : Nat),
    factorial 8 = 2^i * 3^k * 5^m * 7^n →
    i + k + m + n = 11 →
    n = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_positive_integer_value_l2986_298650


namespace NUMINAMATH_CALUDE_employee_average_salary_l2986_298622

theorem employee_average_salary 
  (num_employees : ℕ) 
  (manager_salary : ℕ) 
  (average_increase : ℕ) 
  (h1 : num_employees = 18)
  (h2 : manager_salary = 5800)
  (h3 : average_increase = 200) :
  let total_with_manager := (num_employees + 1) * (average_employee_salary + average_increase)
  let total_without_manager := num_employees * average_employee_salary + manager_salary
  total_with_manager = total_without_manager →
  average_employee_salary = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_average_salary_l2986_298622


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_coverage_l2986_298625

theorem chocolate_chip_cookie_coverage : 
  let cookie_radius : ℝ := 3
  let chip_radius : ℝ := 0.3
  let cookie_area : ℝ := π * cookie_radius^2
  let chip_area : ℝ := π * chip_radius^2
  let coverage_ratio : ℝ := 1/4
  let num_chips : ℕ := 25
  (↑num_chips * chip_area = coverage_ratio * cookie_area) ∧ 
  (∀ k : ℕ, k ≠ num_chips → ↑k * chip_area ≠ coverage_ratio * cookie_area) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_coverage_l2986_298625


namespace NUMINAMATH_CALUDE_inequality_not_hold_l2986_298661

theorem inequality_not_hold (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l2986_298661


namespace NUMINAMATH_CALUDE_contradiction_proof_l2986_298647

theorem contradiction_proof (a b c d : ℝ) 
  (sum1 : a + b = 1) 
  (sum2 : c + d = 1) 
  (prod_sum : a * c + b * d > 1) 
  (all_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : 
  False :=
sorry

end NUMINAMATH_CALUDE_contradiction_proof_l2986_298647


namespace NUMINAMATH_CALUDE_wire_cutting_l2986_298626

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 49 →
  ratio = 2 / 5 →
  shorter_piece + ratio⁻¹ * shorter_piece = total_length →
  shorter_piece = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l2986_298626


namespace NUMINAMATH_CALUDE_sum_of_edges_for_given_pyramid_l2986_298646

/-- Regular hexagonal pyramid with given edge lengths -/
structure RegularHexagonalPyramid where
  base_edge : ℝ
  lateral_edge : ℝ

/-- Sum of all edges of a regular hexagonal pyramid -/
def sum_of_edges (p : RegularHexagonalPyramid) : ℝ :=
  6 * p.base_edge + 6 * p.lateral_edge

/-- Theorem: The sum of all edges of a regular hexagonal pyramid with base edge 8 and lateral edge 13 is 126 -/
theorem sum_of_edges_for_given_pyramid :
  let p : RegularHexagonalPyramid := ⟨8, 13⟩
  sum_of_edges p = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edges_for_given_pyramid_l2986_298646


namespace NUMINAMATH_CALUDE_two_piggy_banks_value_l2986_298627

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "dime" => 10
  | _ => 0

/-- Represents the number of coins in a piggy bank -/
def coins_in_bank (coin : String) : ℕ :=
  match coin with
  | "penny" => 100
  | "dime" => 50
  | _ => 0

/-- Calculates the total value in cents for a single piggy bank -/
def piggy_bank_value : ℕ :=
  coin_value "penny" * coins_in_bank "penny" +
  coin_value "dime" * coins_in_bank "dime"

/-- Calculates the total value in dollars for two piggy banks -/
def total_value : ℚ :=
  (2 * piggy_bank_value : ℚ) / 100

theorem two_piggy_banks_value : total_value = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_piggy_banks_value_l2986_298627


namespace NUMINAMATH_CALUDE_tiling_colors_l2986_298612

/-- Represents the type of tiling: squares or hexagons -/
inductive TilingType
  | Squares
  | Hexagons

/-- Calculates the number of colors needed for a specific tiling type and grid parameters -/
def number_of_colors (t : TilingType) (k l : ℕ) : ℕ :=
  match t with
  | TilingType.Squares => k^2 + l^2
  | TilingType.Hexagons => k^2 + k*l + l^2

/-- Theorem stating the number of colors needed for a valid tiling -/
theorem tiling_colors (t : TilingType) (k l : ℕ) (h : k ≠ 0 ∨ l ≠ 0) :
  ∃ (n : ℕ), n = number_of_colors t k l ∧ n > 0 :=
by sorry

end NUMINAMATH_CALUDE_tiling_colors_l2986_298612


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l2986_298663

theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_radius := 2.5 * r
  let new_height := 3 * h
  (π * new_radius^2 * new_height) / (π * r^2 * h) = 18.75 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l2986_298663


namespace NUMINAMATH_CALUDE_sum_remainder_l2986_298698

theorem sum_remainder (a b c : ℕ) (ha : a % 36 = 15) (hb : b % 36 = 22) (hc : c % 36 = 9) :
  (a + b + c) % 36 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l2986_298698


namespace NUMINAMATH_CALUDE_sinusoidal_oscillations_l2986_298689

/-- A sinusoidal function that completes 5 oscillations from 0 to 2π has b = 5 -/
theorem sinusoidal_oscillations (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, (a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * Real.pi) + c) + d)) →
  (∃ n : ℕ, n = 5 ∧ ∀ x : ℝ, a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * Real.pi / n) + c) + d) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_oscillations_l2986_298689


namespace NUMINAMATH_CALUDE_exists_triangle_no_isosceles_triangle_l2986_298609

/-- The set of stick lengths -/
def stick_lengths : List ℝ := [1, 1.9, 1.9^2, 1.9^3, 1.9^4, 1.9^5, 1.9^6, 1.9^7, 1.9^8, 1.9^9]

/-- Function to check if three lengths can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three lengths can form an isosceles triangle -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  is_triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

/-- Theorem stating that a triangle can be formed from the given stick lengths -/
theorem exists_triangle : ∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_triangle a b c :=
sorry

/-- Theorem stating that an isosceles triangle cannot be formed from the given stick lengths -/
theorem no_isosceles_triangle : ¬∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_isosceles_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_no_isosceles_triangle_l2986_298609


namespace NUMINAMATH_CALUDE_residue_of_negative_thousand_mod_33_l2986_298654

theorem residue_of_negative_thousand_mod_33 :
  ∃ (k : ℤ), -1000 = 33 * k + 23 ∧ (0 ≤ 23 ∧ 23 < 33) := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_thousand_mod_33_l2986_298654


namespace NUMINAMATH_CALUDE_dress_hemming_time_l2986_298669

/-- The time required to hem a dress given its length, stitch size, and stitching rate -/
theorem dress_hemming_time 
  (dress_length : ℝ) 
  (stitch_length : ℝ) 
  (stitches_per_minute : ℝ) 
  (h1 : dress_length = 3) -- dress length in feet
  (h2 : stitch_length = 1/4 / 12) -- stitch length in feet (1/4 inch converted to feet)
  (h3 : stitches_per_minute = 24) :
  dress_length / (stitch_length * stitches_per_minute) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dress_hemming_time_l2986_298669


namespace NUMINAMATH_CALUDE_total_peanuts_l2986_298667

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 4

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 4

/-- Theorem: The total number of peanuts in the box is 8 -/
theorem total_peanuts : initial_peanuts + added_peanuts = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l2986_298667


namespace NUMINAMATH_CALUDE_five_people_seven_chairs_arrangement_l2986_298600

/-- The number of ways to arrange people in chairs with one person fixed -/
def arrangement_count (total_chairs : ℕ) (total_people : ℕ) (fixed_position : ℕ) : ℕ :=
  (total_chairs - 1).factorial / (total_chairs - total_people).factorial

/-- Theorem: Five people can be arranged in seven chairs with one person fixed in the middle in 360 ways -/
theorem five_people_seven_chairs_arrangement : 
  arrangement_count 7 5 4 = 360 := by
sorry

end NUMINAMATH_CALUDE_five_people_seven_chairs_arrangement_l2986_298600


namespace NUMINAMATH_CALUDE_quadratic_through_points_l2986_298664

/-- A quadratic function that passes through (-1, 2) and (1, y) must have y = 2 -/
theorem quadratic_through_points (a : ℝ) (y : ℝ) : 
  a ≠ 0 → (2 = a * (-1)^2) → (y = a * 1^2) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_points_l2986_298664


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l2986_298673

/-- An ellipse with equation x²/4 + y² = 1 -/
structure Ellipse where
  eq : ∀ x y : ℝ, x^2/4 + y^2 = 1

/-- The left focus of the ellipse -/
def leftFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def rightFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- A point on the ellipse where a line perpendicular to the x-axis passing through the left focus intersects the ellipse -/
def intersectionPoint (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focus_distance (e : Ellipse) :
  distance (intersectionPoint e) (rightFocus e) = 7/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l2986_298673


namespace NUMINAMATH_CALUDE_g_50_equals_zero_l2986_298681

theorem g_50_equals_zero
  (g : ℕ → ℕ)
  (h : ∀ a b : ℕ, 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b)) :
  g 50 = 0 := by
sorry

end NUMINAMATH_CALUDE_g_50_equals_zero_l2986_298681


namespace NUMINAMATH_CALUDE_percy_swimming_hours_l2986_298632

/-- Percy's daily swimming hours on weekdays -/
def weekday_hours : ℕ := 2

/-- Number of weekdays Percy swims per week -/
def weekdays_per_week : ℕ := 5

/-- Percy's weekend swimming hours -/
def weekend_hours : ℕ := 3

/-- Number of weeks -/
def num_weeks : ℕ := 4

/-- Total swimming hours over the given number of weeks -/
def total_swimming_hours : ℕ := 
  num_weeks * (weekday_hours * weekdays_per_week + weekend_hours)

theorem percy_swimming_hours : total_swimming_hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_percy_swimming_hours_l2986_298632


namespace NUMINAMATH_CALUDE_regular_polygon_27_diagonals_has_9_sides_l2986_298680

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon with 27 diagonals has 9 sides -/
theorem regular_polygon_27_diagonals_has_9_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 27 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_27_diagonals_has_9_sides_l2986_298680


namespace NUMINAMATH_CALUDE_t_shaped_area_l2986_298608

/-- The area of a T-shaped region formed by subtracting two squares and a rectangle from a larger square --/
theorem t_shaped_area (side_large : ℝ) (side_small : ℝ) (rect_length rect_width : ℝ) : 
  side_large = side_small + rect_length →
  side_large = 6 →
  side_small = 2 →
  rect_length = 4 →
  rect_width = 2 →
  side_large^2 - (2 * side_small^2 + rect_length * rect_width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shaped_area_l2986_298608


namespace NUMINAMATH_CALUDE_theresa_video_games_l2986_298678

theorem theresa_video_games (tory julia theresa : ℕ) : 
  tory = 6 → 
  julia = tory / 3 → 
  theresa = 3 * julia + 5 → 
  theresa = 11 := by
sorry

end NUMINAMATH_CALUDE_theresa_video_games_l2986_298678


namespace NUMINAMATH_CALUDE_intersection_point_sum_l2986_298651

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  sorry

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (a b c : Point) : ℚ :=
  sorry

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Represents the intersection point of a line with CD -/
structure IntersectionPoint where
  p : ℕ
  q : ℕ
  r : ℕ
  s : ℕ

/-- The main theorem -/
theorem intersection_point_sum (a b c d : Point) (l : Line) (i : IntersectionPoint) :
  a = Point.mk 0 0 →
  b = Point.mk 2 4 →
  c = Point.mk 6 6 →
  d = Point.mk 8 0 →
  l.p1 = a →
  isPointOnLine (Point.mk (i.p / i.q) (i.r / i.s)) l →
  isPointOnLine (Point.mk (i.p / i.q) (i.r / i.s)) (Line.mk c d) →
  triangleArea a (Point.mk (i.p / i.q) (i.r / i.s)) d = (1/3) * quadrilateralArea a b c d →
  i.p + i.q + i.r + i.s = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l2986_298651


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l2986_298603

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (3 - 2 * x) = 1 / (3 - 2 * x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l2986_298603


namespace NUMINAMATH_CALUDE_B_grazed_for_5_months_l2986_298605

/-- Represents the number of months B grazed his cows -/
def B_months : ℕ := sorry

/-- Represents the total rent of the field in rupees -/
def total_rent : ℕ := 6500

/-- Represents A's share of the rent in rupees -/
def A_rent : ℕ := 1440

/-- Represents the number of cows grazed by each milkman -/
def cows : Fin 4 → ℕ
  | 0 => 24  -- A's cows
  | 1 => 10  -- B's cows
  | 2 => 35  -- C's cows
  | 3 => 21  -- D's cows

/-- Represents the number of months each milkman grazed their cows -/
def months : Fin 4 → ℕ
  | 0 => 3             -- A's months
  | 1 => B_months      -- B's months (unknown)
  | 2 => 4             -- C's months
  | 3 => 3             -- D's months

/-- Calculates the total cow-months for all milkmen -/
def total_cow_months : ℕ := sorry

/-- The main theorem stating that B grazed his cows for 5 months -/
theorem B_grazed_for_5_months : B_months = 5 := by sorry

end NUMINAMATH_CALUDE_B_grazed_for_5_months_l2986_298605


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2986_298634

/-- Given a triangle ABC with sides defined by the equations:
    BC: x cos θ₁ + y sin θ₁ - p₁ = 0
    CA: x cos θ₂ + y sin θ₂ - p₂ = 0
    AB: x cos θ₃ + y sin θ₃ - p₃ = 0
    This theorem states that any point P(x, y) on the circumcircle of ABC
    satisfies the given equation. -/
theorem circumcircle_equation (θ₁ θ₂ θ₃ p₁ p₂ p₃ x y : ℝ) :
  (x * Real.cos θ₂ + y * Real.sin θ₂ - p₂) * (x * Real.cos θ₃ + y * Real.sin θ₃ - p₃) * Real.sin (θ₂ - θ₃) +
  (x * Real.cos θ₃ + y * Real.sin θ₃ - p₃) * (x * Real.cos θ₁ + y * Real.sin θ₁ - p₁) * Real.sin (θ₃ - θ₁) +
  (x * Real.cos θ₁ + y * Real.sin θ₁ - p₁) * (x * Real.cos θ₂ + y * Real.sin θ₂ - p₂) * Real.sin (θ₁ - θ₂) = 0 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l2986_298634


namespace NUMINAMATH_CALUDE_original_polygon_sides_l2986_298658

-- Define the number of sides of the original polygon
def n : ℕ := sorry

-- Define the sum of interior angles of the new polygon
def new_polygon_angle_sum : ℝ := 2520

-- Theorem statement
theorem original_polygon_sides :
  (n + 1 - 2) * 180 = new_polygon_angle_sum → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_polygon_sides_l2986_298658


namespace NUMINAMATH_CALUDE_orcs_per_squad_l2986_298613

theorem orcs_per_squad (total_weight : ℕ) (num_squads : ℕ) (weight_per_orc : ℕ) :
  total_weight = 1200 →
  num_squads = 10 →
  weight_per_orc = 15 →
  (total_weight / weight_per_orc) / num_squads = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_orcs_per_squad_l2986_298613


namespace NUMINAMATH_CALUDE_prob_one_tail_theorem_l2986_298619

/-- The probability of getting exactly one tail in 5 flips of a biased coin -/
def prob_one_tail_in_five_flips (p : ℝ) : ℝ :=
  5 * p * (1 - p)^4

/-- Theorem: The probability of getting exactly one tail in 5 flips of a biased coin -/
theorem prob_one_tail_theorem (p q : ℝ) 
  (h_prob : 0 ≤ p ∧ p ≤ 1) 
  (h_sum : p + q = 1) :
  prob_one_tail_in_five_flips p = 5 * p * q^4 :=
sorry

end NUMINAMATH_CALUDE_prob_one_tail_theorem_l2986_298619


namespace NUMINAMATH_CALUDE_train_speed_on_time_l2986_298671

/-- The speed at which a train arrives on time, given the journey length and late arrival information. -/
theorem train_speed_on_time 
  (journey_length : ℝ) 
  (late_speed : ℝ) 
  (late_time : ℝ) 
  (h1 : journey_length = 15) 
  (h2 : late_speed = 50) 
  (h3 : late_time = 0.25) : 
  (journey_length / ((journey_length / late_speed) - late_time) = 300) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_on_time_l2986_298671


namespace NUMINAMATH_CALUDE_line_equation_in_triangle_l2986_298668

/-- Given a line passing through (-2b, 0) forming a triangular region in the second quadrant with area S, 
    its equation is 2Sx - b^2y + 4bS = 0 --/
theorem line_equation_in_triangle (b S : ℝ) (h_b : b ≠ 0) (h_S : S > 0) : 
  ∃ (m k : ℝ), 
    (∀ (x y : ℝ), y = m * x + k → 
      (x = -2*b ∧ y = 0) ∨ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 0 ∧ y > 0)) ∧
    (1/2 * 2*b * (S/b) = S) ∧
    (∀ (x y : ℝ), 2*S*x - b^2*y + 4*b*S = 0 ↔ y = m * x + k) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_in_triangle_l2986_298668


namespace NUMINAMATH_CALUDE_cube_root_of_216_l2986_298666

theorem cube_root_of_216 (y : ℝ) : (Real.sqrt y)^3 = 216 → y = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_216_l2986_298666


namespace NUMINAMATH_CALUDE_mr_blue_flower_bed_yield_l2986_298620

/-- Represents the dimensions and yield of a flower bed -/
structure FlowerBed where
  length_paces : ℕ
  width_paces : ℕ
  pace_length : ℝ
  yield_per_sqft : ℝ

/-- Calculates the expected rose petal yield from a flower bed -/
def expected_yield (fb : FlowerBed) : ℝ :=
  (fb.length_paces : ℝ) * fb.pace_length *
  (fb.width_paces : ℝ) * fb.pace_length *
  fb.yield_per_sqft

/-- Theorem stating the expected yield for Mr. Blue's flower bed -/
theorem mr_blue_flower_bed_yield :
  let fb : FlowerBed := {
    length_paces := 18,
    width_paces := 24,
    pace_length := 1.5,
    yield_per_sqft := 0.4
  }
  expected_yield fb = 388.8 := by sorry

end NUMINAMATH_CALUDE_mr_blue_flower_bed_yield_l2986_298620


namespace NUMINAMATH_CALUDE_total_cost_theorem_l2986_298699

/-- Represents the cost of utensils in Moneda -/
structure UtensilCost where
  teaspoon : ℕ
  tablespoon : ℕ
  dessertSpoon : ℕ

/-- Represents the number of utensils Clara has -/
structure UtensilCount where
  teaspoon : ℕ
  tablespoon : ℕ
  dessertSpoon : ℕ

/-- Calculates the total cost of exchanged utensils and souvenirs in euros -/
def totalCostInEuros (costs : UtensilCost) (counts : UtensilCount) 
  (monedaToEuro : ℚ) (souvenirCostDollars : ℕ) (euroToDollar : ℚ) : ℚ :=
  sorry

/-- Theorem stating the total cost in euros -/
theorem total_cost_theorem (costs : UtensilCost) (counts : UtensilCount) 
  (monedaToEuro : ℚ) (souvenirCostDollars : ℕ) (euroToDollar : ℚ) :
  costs.teaspoon = 9 ∧ costs.tablespoon = 12 ∧ costs.dessertSpoon = 18 ∧
  counts.teaspoon = 7 ∧ counts.tablespoon = 10 ∧ counts.dessertSpoon = 12 ∧
  monedaToEuro = 0.04 ∧ souvenirCostDollars = 40 ∧ euroToDollar = 1.15 →
  totalCostInEuros costs counts monedaToEuro souvenirCostDollars euroToDollar = 50.74 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l2986_298699


namespace NUMINAMATH_CALUDE_grid_lines_formula_l2986_298606

/-- The number of straight lines needed to draw an n × n square grid -/
def gridLines (n : ℕ) : ℕ := 2 * (n + 1)

/-- Theorem stating that the number of straight lines needed to draw an n × n square grid is 2(n + 1) -/
theorem grid_lines_formula (n : ℕ) : gridLines n = 2 * (n + 1) := by
  sorry

#check grid_lines_formula

end NUMINAMATH_CALUDE_grid_lines_formula_l2986_298606


namespace NUMINAMATH_CALUDE_equation_condition_l2986_298611

theorem equation_condition (x y z : ℤ) :
  x * (x - y) + y * (y - z) + z * (z - x) = 0 → x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equation_condition_l2986_298611


namespace NUMINAMATH_CALUDE_birds_and_nests_difference_l2986_298645

theorem birds_and_nests_difference :
  let num_birds : ℕ := 6
  let num_nests : ℕ := 3
  num_birds - num_nests = 3 :=
by sorry

end NUMINAMATH_CALUDE_birds_and_nests_difference_l2986_298645


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2986_298655

/-- The y-coordinate of the point on the y-axis equidistant from (1, 0) and (4, 3) -/
theorem equidistant_point_y_coordinate : ∃ y : ℝ, 
  (Real.sqrt ((1 - 0)^2 + (0 - y)^2) = Real.sqrt ((4 - 0)^2 + (3 - y)^2)) ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2986_298655


namespace NUMINAMATH_CALUDE_number_division_problem_l2986_298697

theorem number_division_problem : ∃! x : ℕ, 
  ∃ q : ℕ, x = 7 * q ∧ q + x + 7 = 175 ∧ x = 147 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2986_298697


namespace NUMINAMATH_CALUDE_max_andy_cookies_l2986_298618

def total_cookies : ℕ := 30

def valid_distribution (andy_cookies : ℕ) : Prop :=
  andy_cookies + 3 * andy_cookies ≤ total_cookies

theorem max_andy_cookies :
  ∃ (max : ℕ), valid_distribution max ∧
    ∀ (n : ℕ), valid_distribution n → n ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_andy_cookies_l2986_298618


namespace NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l2986_298682

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l2986_298682


namespace NUMINAMATH_CALUDE_quiz_win_probability_l2986_298630

/-- Represents a quiz with multiple-choice questions. -/
structure Quiz where
  num_questions : ℕ
  num_choices : ℕ

/-- Represents the outcome of a quiz attempt. -/
structure QuizOutcome where
  correct_answers : ℕ

/-- The probability of getting a single question correct. -/
def single_question_probability (q : Quiz) : ℚ :=
  1 / q.num_choices

/-- The probability of winning the quiz. -/
def win_probability (q : Quiz) : ℚ :=
  let p := single_question_probability q
  (p ^ q.num_questions) +  -- All correct
  q.num_questions * (p ^ 3 * (1 - p))  -- Exactly 3 correct

/-- The theorem stating the probability of winning the quiz. -/
theorem quiz_win_probability (q : Quiz) (h1 : q.num_questions = 4) (h2 : q.num_choices = 3) :
  win_probability q = 1 / 9 := by
  sorry

#eval win_probability {num_questions := 4, num_choices := 3}

end NUMINAMATH_CALUDE_quiz_win_probability_l2986_298630


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2986_298660

theorem power_tower_mod_500 : 4^(4^(4^4)) ≡ 36 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2986_298660


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l2986_298685

theorem cubic_polynomial_problem (a b c : ℝ) (Q : ℝ → ℝ) :
  (∀ x, x^3 - 2*x^2 + 4*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∃ p q r s : ℝ, ∀ x, Q x = p*x^3 + q*x^2 + r*x + s) →
  Q a = b + c - 3 →
  Q b = a + c - 3 →
  Q c = a + b - 3 →
  Q (a + b + c) = -17 →
  (∀ x, Q x = -20/7*x^3 + 34/7*x^2 - 12/7*x + 13/7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l2986_298685


namespace NUMINAMATH_CALUDE_product_of_roots_l2986_298683

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ y : ℝ, (x + 3) * (x - 4) = 20 ∧ (y + 3) * (y - 4) = 20 ∧ x * y = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2986_298683


namespace NUMINAMATH_CALUDE_skincare_fraction_is_two_fifths_l2986_298617

/-- Represents Susie's babysitting and spending scenario -/
structure BabysittingScenario where
  hours_per_day : ℕ
  rate_per_hour : ℕ
  days_per_week : ℕ
  makeup_fraction : ℚ
  money_left : ℕ

/-- Calculates the fraction spent on skincare products given a babysitting scenario -/
def skincare_fraction (scenario : BabysittingScenario) : ℚ :=
  -- Definition to be proved
  2 / 5

/-- Theorem stating that given the specific scenario, the fraction spent on skincare is 2/5 -/
theorem skincare_fraction_is_two_fifths :
  let scenario : BabysittingScenario := {
    hours_per_day := 3,
    rate_per_hour := 10,
    days_per_week := 7,
    makeup_fraction := 3 / 10,
    money_left := 63
  }
  skincare_fraction scenario = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_skincare_fraction_is_two_fifths_l2986_298617


namespace NUMINAMATH_CALUDE_monotonicity_condition_max_k_value_l2986_298674

noncomputable section

def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem monotonicity_condition (t : ℝ) :
  (is_monotonic f (-2) t) ↔ -2 < t ∧ t ≤ 0 := by sorry

theorem max_k_value :
  ∃ k : ℕ, k = 6 ∧ 
  (∀ x : ℝ, x > 0 → (f x / Real.exp x) + 7*x - 2 > k * (x * Real.log x - 1)) ∧
  (∀ m : ℕ, m > k → ∃ x : ℝ, x > 0 ∧ (f x / Real.exp x) + 7*x - 2 ≤ m * (x * Real.log x - 1)) := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_condition_max_k_value_l2986_298674


namespace NUMINAMATH_CALUDE_game_points_theorem_l2986_298602

theorem game_points_theorem (eric : ℕ) (mark : ℕ) (samanta : ℕ) : 
  mark = eric + eric / 2 →
  samanta = mark + 8 →
  eric + mark + samanta = 32 →
  eric = 6 := by
sorry

end NUMINAMATH_CALUDE_game_points_theorem_l2986_298602


namespace NUMINAMATH_CALUDE_remainder_3456_div_23_l2986_298679

theorem remainder_3456_div_23 : 3456 % 23 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3456_div_23_l2986_298679


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2986_298639

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x, x^2 - 16*x + 60 = (x - a)*(x - b)) : 
  3*b - a = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2986_298639


namespace NUMINAMATH_CALUDE_gwen_homework_l2986_298631

def homework_problem (math_problems science_problems finished_problems : ℕ) : Prop :=
  let total_problems := math_problems + science_problems
  total_problems - finished_problems = 5

theorem gwen_homework :
  homework_problem 18 11 24 := by sorry

end NUMINAMATH_CALUDE_gwen_homework_l2986_298631


namespace NUMINAMATH_CALUDE_prime_power_plus_one_prime_l2986_298656

theorem prime_power_plus_one_prime (x y z : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧ x^y + 1 = z → (x = 2 ∧ y = 2 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_plus_one_prime_l2986_298656


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2986_298640

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 4) :
  a^4 + b^4 + c^4 = 6.833 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2986_298640


namespace NUMINAMATH_CALUDE_domain_of_log2_l2986_298676

-- Define the logarithm function with base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem stating that the domain of log₂x is the set of all positive real numbers
theorem domain_of_log2 :
  {x : ℝ | ∃ y, log2 x = y} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_log2_l2986_298676


namespace NUMINAMATH_CALUDE_inscribed_polygon_sides_l2986_298615

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the configuration of polygons in the problem -/
structure PolygonConfiguration :=
  (central : RegularPolygon)
  (inscribed : RegularPolygon)
  (num_inscribed : ℕ)

/-- The sum of interior angles at a contact point -/
def contact_angle_sum : ℝ := 360

/-- The condition that the vertices of the central polygon touch the centers of the inscribed polygons -/
def touches_centers (config : PolygonConfiguration) : Prop :=
  sorry

/-- The theorem stating that in the given configuration, the number of sides of the inscribed polygons must be 6 -/
theorem inscribed_polygon_sides
  (config : PolygonConfiguration)
  (h1 : config.central.sides = 12)
  (h2 : config.num_inscribed = 6)
  (h3 : touches_centers config)
  (h4 : contact_angle_sum = 360) :
  config.inscribed.sides = 6 :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_sides_l2986_298615


namespace NUMINAMATH_CALUDE_shooting_competition_stability_l2986_298628

/-- Represents a participant in the shooting competition -/
structure Participant where
  name : String
  variance : ℝ

/-- Defines when a participant has more stable performance -/
def more_stable (p1 p2 : Participant) : Prop :=
  p1.variance < p2.variance

theorem shooting_competition_stability :
  let A : Participant := ⟨"A", 3⟩
  let B : Participant := ⟨"B", 1.2⟩
  more_stable B A := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_stability_l2986_298628


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_hypotenuse_ratio_l2986_298649

theorem right_triangle_perimeter_hypotenuse_ratio 
  (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a := 3*x + 3*y
  let b := 4*x
  let c := 4*y
  let perimeter := a + b + c
  (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2) →
  (perimeter / a = 7/3 ∨ perimeter / b = 56/25 ∨ perimeter / c = 56/25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_hypotenuse_ratio_l2986_298649


namespace NUMINAMATH_CALUDE_nonzero_sum_reciprocal_equality_l2986_298694

theorem nonzero_sum_reciprocal_equality (a b c : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c ≠ 0)
  (h5 : 1/a + 1/b + 1/c = 1/(a + b + c)) :
  1/a^1999 + 1/b^1999 + 1/c^1999 = 1/(a^1999 + b^1999 + c^1999) := by
  sorry

end NUMINAMATH_CALUDE_nonzero_sum_reciprocal_equality_l2986_298694


namespace NUMINAMATH_CALUDE_a4_plus_b4_equals_17_l2986_298684

theorem a4_plus_b4_equals_17 (a b : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) : a^4 + b^4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_a4_plus_b4_equals_17_l2986_298684


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2986_298629

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ (∃ x : ℝ, x^2 - x + 1/4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2986_298629


namespace NUMINAMATH_CALUDE_largest_y_in_special_right_triangle_l2986_298665

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_y_in_special_right_triangle (x y z : ℕ) 
  (h1 : is_prime x ∧ is_prime y ∧ is_prime z)
  (h2 : x + y + z = 90)
  (h3 : y < x)
  (h4 : y > z) :
  y ≤ 47 ∧ ∃ (x' z' : ℕ), is_prime x' ∧ is_prime z' ∧ x' + 47 + z' = 90 ∧ 47 < x' ∧ 47 > z' :=
sorry

end NUMINAMATH_CALUDE_largest_y_in_special_right_triangle_l2986_298665


namespace NUMINAMATH_CALUDE_gcf_of_40_120_80_l2986_298641

theorem gcf_of_40_120_80 : Nat.gcd 40 (Nat.gcd 120 80) = 40 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_40_120_80_l2986_298641


namespace NUMINAMATH_CALUDE_lucky_lila_problem_l2986_298624

theorem lucky_lila_problem (a b c d e : ℤ) : 
  a = 5 → b = 3 → c = 2 → d = 6 →
  (a - b + c * d - e = a - (b + (c * (d - e)))) →
  e = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lila_problem_l2986_298624
