import Mathlib

namespace NUMINAMATH_CALUDE_student_marks_difference_l587_58780

/-- Given a student's marks in physics, chemistry, and mathematics,
    prove that the total marks exceed the physics marks by 140,
    given that the average of chemistry and mathematics marks is 70. -/
theorem student_marks_difference 
  (P C M : ℕ)  -- Marks in Physics, Chemistry, and Mathematics
  (h_avg : (C + M) / 2 = 70)  -- Average of Chemistry and Mathematics is 70
  : (P + C + M) - P = 140 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_difference_l587_58780


namespace NUMINAMATH_CALUDE_tan_theta_value_l587_58787

theorem tan_theta_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2)
  (h2 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2 * θ) = 3) :
  Real.tan θ = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_theta_value_l587_58787


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l587_58754

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (4, -3) and (-8, 7) is equal to -4. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -3
  let x2 : ℝ := -8
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l587_58754


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l587_58755

theorem trigonometric_simplification (α : ℝ) :
  (1 + Real.cos α + Real.cos (2 * α) + Real.cos (3 * α)) /
  (Real.cos α + 2 * (Real.cos α)^2 - 1) = 2 * Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l587_58755


namespace NUMINAMATH_CALUDE_teacher_assignment_schemes_l587_58731

theorem teacher_assignment_schemes (male_teachers : Nat) (female_teachers : Nat) : 
  male_teachers = 5 → 
  female_teachers = 4 → 
  (Nat.factorial 9 / Nat.factorial 6) - 
  (Nat.factorial 5 / Nat.factorial 2 + Nat.factorial 4 / Nat.factorial 1) = 420 := by
  sorry

end NUMINAMATH_CALUDE_teacher_assignment_schemes_l587_58731


namespace NUMINAMATH_CALUDE_complex_equation_solution_l587_58718

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Real.sqrt 2 * Complex.I) = -(Real.sqrt 2) * Complex.I →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l587_58718


namespace NUMINAMATH_CALUDE_complex_equation_solution_l587_58740

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + Complex.I) * z = -1 + 5 * Complex.I → z = 2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l587_58740


namespace NUMINAMATH_CALUDE_rectangle_area_l587_58738

/-- Theorem: Area of a rectangle with length-to-width ratio 4:3 and diagonal d -/
theorem rectangle_area (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 4 / 3 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = 12 / 25 * d ^ 2 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l587_58738


namespace NUMINAMATH_CALUDE_abes_age_problem_l587_58782

/-- Abe's age problem -/
theorem abes_age_problem (present_age : ℕ) (x : ℕ) 
  (h1 : present_age = 28)
  (h2 : present_age + (present_age - x) = 35) :
  present_age + x = 49 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_problem_l587_58782


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l587_58758

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l587_58758


namespace NUMINAMATH_CALUDE_number_of_blue_balls_l587_58770

/-- The number of blue balls originally in the box -/
def B : ℕ := sorry

/-- The number of red balls originally in the box -/
def R : ℕ := sorry

/-- Theorem stating the number of blue balls originally in the box -/
theorem number_of_blue_balls : 
  B = R + 17 ∧ 
  (B + 57) + (R + 18) - 44 = 502 → 
  B = 244 := by sorry

end NUMINAMATH_CALUDE_number_of_blue_balls_l587_58770


namespace NUMINAMATH_CALUDE_division_simplification_l587_58775

theorem division_simplification (x y : ℝ) (h : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l587_58775


namespace NUMINAMATH_CALUDE_value_of_x_l587_58717

theorem value_of_x :
  ∀ (x a b c d : ℤ),
    x = a + 7 →
    a = b + 9 →
    b = c + 15 →
    c = d + 25 →
    d = 60 →
    x = 116 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l587_58717


namespace NUMINAMATH_CALUDE_intercept_sum_lower_bound_l587_58741

/-- A line passing through (1,3) intersecting positive x and y axes -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  passes_through_P : 1 / a + 3 / b = 1

/-- The sum of intercepts is at least 4 + 2√3 -/
theorem intercept_sum_lower_bound (l : InterceptLine) : l.a + l.b ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

#check intercept_sum_lower_bound

end NUMINAMATH_CALUDE_intercept_sum_lower_bound_l587_58741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l587_58725

/-- An arithmetic sequence with sum of first 5 terms equal to 15 and second term equal to 5 has common difference -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ)  -- arithmetic sequence
  (S : ℕ → ℝ)  -- sum function
  (h1 : S 5 = 15)  -- sum of first 5 terms is 15
  (h2 : a 2 = 5)   -- second term is 5
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2)  -- sum formula for arithmetic sequence
  (h4 : ∀ n, a (n + 1) = a n + d)  -- definition of arithmetic sequence
  : d = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l587_58725


namespace NUMINAMATH_CALUDE_complex_sixth_root_of_negative_sixteen_l587_58727

theorem complex_sixth_root_of_negative_sixteen :
  ∀ z : ℂ, z^6 = -16 ↔ z = Complex.I * 2 ∨ z = Complex.I * (-2) := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_root_of_negative_sixteen_l587_58727


namespace NUMINAMATH_CALUDE_initial_apples_correct_l587_58756

/-- The number of apples the cafeteria had initially -/
def initial_apples : ℕ := 23

/-- The number of apples used for lunch -/
def apples_used : ℕ := 20

/-- The number of apples bought -/
def apples_bought : ℕ := 6

/-- The number of apples remaining after transactions -/
def remaining_apples : ℕ := 9

/-- Theorem stating that the initial number of apples is correct -/
theorem initial_apples_correct : 
  initial_apples - apples_used + apples_bought = remaining_apples := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_correct_l587_58756


namespace NUMINAMATH_CALUDE_log3_graph_properties_l587_58704

-- Define the logarithm function base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the graph of y = log₃(x)
def graph_log3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = log3 p.1 ∧ p.1 > 0}

-- Define the x-axis and y-axis
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Theorem statement
theorem log3_graph_properties :
  (∃ p, p ∈ graph_log3 ∩ x_axis) ∧
  (graph_log3 ∩ y_axis = ∅) :=
by sorry

end NUMINAMATH_CALUDE_log3_graph_properties_l587_58704


namespace NUMINAMATH_CALUDE_johns_car_trade_in_value_l587_58734

/-- Calculates the trade-in value of John's car based on his Uber earnings, initial car purchase price, and profit. -/
def trade_in_value (uber_earnings profit initial_car_price : ℕ) : ℕ :=
  initial_car_price - (uber_earnings - profit)

/-- Theorem stating that John's car trade-in value is $6,000 given the provided conditions. -/
theorem johns_car_trade_in_value :
  trade_in_value 30000 18000 18000 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_johns_car_trade_in_value_l587_58734


namespace NUMINAMATH_CALUDE_parallelogram_area_l587_58743

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area :
  let base : ℝ := 20
  let height : ℝ := 4
  let area : ℝ := base * height
  area = 80 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l587_58743


namespace NUMINAMATH_CALUDE_upper_bound_necessary_not_sufficient_l587_58760

variable {α : Type*} [PartialOrder α]
variable (I : Set α) (f : α → ℝ) (M : ℝ)

def is_upper_bound (f : α → ℝ) (M : ℝ) (I : Set α) : Prop :=
  ∀ x ∈ I, f x ≤ M

def is_maximum (f : α → ℝ) (M : ℝ) (I : Set α) : Prop :=
  (is_upper_bound f M I) ∧ (∃ x ∈ I, f x = M)

theorem upper_bound_necessary_not_sufficient :
  (is_upper_bound f M I → is_maximum f M I) ∧
  ¬(is_maximum f M I → is_upper_bound f M I) :=
sorry

end NUMINAMATH_CALUDE_upper_bound_necessary_not_sufficient_l587_58760


namespace NUMINAMATH_CALUDE_total_students_suggestion_l587_58793

theorem total_students_suggestion (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 324)
  (h2 : bacon = 374)
  (h3 : tomatoes = 128) :
  mashed_potatoes + bacon + tomatoes = 826 := by
  sorry

end NUMINAMATH_CALUDE_total_students_suggestion_l587_58793


namespace NUMINAMATH_CALUDE_regular_decagon_exterior_angle_regular_decagon_exterior_angle_is_36_l587_58753

/-- The exterior angle of a regular decagon is 36 degrees. -/
theorem regular_decagon_exterior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let interior_angle_sum : ℝ := 180 * (n - 2)
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- Proof that the exterior angle of a regular decagon is 36 degrees. -/
theorem regular_decagon_exterior_angle_is_36 : 
  regular_decagon_exterior_angle = 36 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_exterior_angle_regular_decagon_exterior_angle_is_36_l587_58753


namespace NUMINAMATH_CALUDE_solve_equation_l587_58707

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l587_58707


namespace NUMINAMATH_CALUDE_division_with_remainder_l587_58797

theorem division_with_remainder (n : ℕ) (h1 : n % 17 ≠ 0) (h2 : n / 17 = 25) :
  n ≤ 441 ∧ n ≥ 426 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l587_58797


namespace NUMINAMATH_CALUDE_mutual_acquaintance_exists_l587_58705

/-- Represents a diplomatic reception with a fixed number of participants. -/
structure DiplomaticReception where
  participants : Nat
  heardOf : Nat → Nat → Prop
  heardOfCount : Nat → Nat

/-- The minimum number of people each participant has heard of that guarantees mutual acquaintance. -/
def minHeardOfCount : Nat := 50

/-- Theorem stating that if each participant has heard of at least 50 others,
    there must be a pair who have heard of each other. -/
theorem mutual_acquaintance_exists (reception : DiplomaticReception)
    (h1 : reception.participants = 99)
    (h2 : ∀ i, i < reception.participants → reception.heardOfCount i ≥ minHeardOfCount)
    (h3 : ∀ i j, i < reception.participants → j < reception.participants → 
         reception.heardOf i j → reception.heardOfCount i > 0) :
    ∃ i j, i < reception.participants ∧ j < reception.participants ∧ 
    i ≠ j ∧ reception.heardOf i j ∧ reception.heardOf j i := by
  sorry

end NUMINAMATH_CALUDE_mutual_acquaintance_exists_l587_58705


namespace NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l587_58767

/-- Represents Mr Yadav's monthly finances --/
structure YadavFinances where
  salary : ℝ
  consumable_percentage : ℝ
  clothes_transport_percentage : ℝ
  yearly_savings : ℝ

/-- Calculates the monthly amount spent on clothes and transport --/
def monthly_clothes_transport (y : YadavFinances) : ℝ :=
  y.salary * (1 - y.consumable_percentage) * y.clothes_transport_percentage

/-- Theorem stating the amount spent on clothes and transport --/
theorem yadav_clothes_transport_expenditure (y : YadavFinances) 
  (h1 : y.consumable_percentage = 0.6)
  (h2 : y.clothes_transport_percentage = 0.5)
  (h3 : y.yearly_savings = 19008)
  (h4 : y.yearly_savings = 12 * (y.salary * (1 - y.consumable_percentage) * (1 - y.clothes_transport_percentage))) :
  monthly_clothes_transport y = 1584 := by
  sorry

#eval monthly_clothes_transport { salary := 7920, consumable_percentage := 0.6, clothes_transport_percentage := 0.5, yearly_savings := 19008 }

end NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l587_58767


namespace NUMINAMATH_CALUDE_max_third_altitude_l587_58719

/-- A triangle with two known altitudes and one unknown integer altitude -/
structure TriangleWithAltitudes where
  /-- The length of the first known altitude -/
  h₁ : ℝ
  /-- The length of the second known altitude -/
  h₂ : ℝ
  /-- The length of the unknown altitude (assumed to be an integer) -/
  h₃ : ℤ
  /-- Condition that h₁ and h₂ are 3 and 9 (in either order) -/
  known_altitudes : (h₁ = 3 ∧ h₂ = 9) ∨ (h₁ = 9 ∧ h₂ = 3)

/-- The theorem stating that the maximum possible integer length for h₃ is 4 -/
theorem max_third_altitude (t : TriangleWithAltitudes) : t.h₃ ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_third_altitude_l587_58719


namespace NUMINAMATH_CALUDE_seven_boys_without_calculators_l587_58721

/-- Represents the number of boys in Miss Parker's class who did not bring calculators. -/
def boys_without_calculators (total_students : ℕ) (boys_in_class : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) : ℕ :=
  boys_in_class - (students_with_calculators - girls_with_calculators)

/-- Theorem stating that 7 boys in Miss Parker's class did not bring calculators. -/
theorem seven_boys_without_calculators :
  boys_without_calculators 24 18 26 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_boys_without_calculators_l587_58721


namespace NUMINAMATH_CALUDE_equation_solution_l587_58748

theorem equation_solution :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l587_58748


namespace NUMINAMATH_CALUDE_y_intercept_of_line_a_l587_58762

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- The given line y = 2x + 4 -/
def given_line : Line :=
  { slope := 2, point := (0, 4) }

/-- Line a, which is parallel to the given line and passes through (2, 5) -/
def line_a : Line :=
  { slope := given_line.slope, point := (2, 5) }

theorem y_intercept_of_line_a :
  y_intercept line_a = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_a_l587_58762


namespace NUMINAMATH_CALUDE_simplify_fraction_l587_58783

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) : (x + 1) / x - 1 / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l587_58783


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l587_58708

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point (s, t) represents the average values of x and y -/
structure AveragePoint where
  s : ℝ
  t : ℝ

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect (t₁ t₂ : RegressionLine) (avg : AveragePoint) :
  (avg.s * t₁.slope + t₁.intercept = avg.t) →
  (avg.s * t₂.slope + t₂.intercept = avg.t) →
  ∃ (x y : ℝ), x = avg.s ∧ y = avg.t ∧ 
    y = x * t₁.slope + t₁.intercept ∧
    y = x * t₂.slope + t₂.intercept := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersect_l587_58708


namespace NUMINAMATH_CALUDE_count_integer_pairs_l587_58785

theorem count_integer_pairs : ∃ (count : ℕ), 
  count = (Finset.filter (fun p : ℕ × ℕ => 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 2887 ∧ 
    (7 : ℝ)^n < 3^m ∧ 3^m < 3^(m+3) ∧ 3^(m+3) < 7^(n+1))
  (Finset.product (Finset.range 2888) (Finset.range (3^2889 / 7^1233 + 1)))).card ∧
  3^2888 < 7^1233 ∧ 7^1233 < 3^2889 ∧
  count = 2466 :=
by sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l587_58785


namespace NUMINAMATH_CALUDE_monomial_difference_l587_58747

theorem monomial_difference (m n : ℤ) : 
  (∃ (a : ℝ) (p q : ℤ), ∀ (x y : ℝ), 9 * x^(m-2) * y^2 - (-3 * x^3 * y^(n+1)) = a * x^p * y^q) → 
  n - m = -4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_difference_l587_58747


namespace NUMINAMATH_CALUDE_lower_right_is_one_l587_58746

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if a number appears exactly once in each row -/
def unique_in_rows (g : Grid) : Prop :=
  ∀ i n, (∃! j, g i j = n) ∧ (1 ≤ n ∧ n ≤ 5)

/-- Checks if a number appears exactly once in each column -/
def unique_in_columns (g : Grid) : Prop :=
  ∀ j n, (∃! i, g i j = n) ∧ (1 ≤ n ∧ n ≤ 5)

/-- Initial grid configuration -/
def initial_grid : Grid :=
  fun i j =>
    if i = 0 ∧ j = 0 then 1
    else if i = 0 ∧ j = 2 then 2
    else if i = 1 ∧ j = 0 then 2
    else if i = 1 ∧ j = 1 then 4
    else if i = 2 ∧ j = 3 then 5
    else if i = 3 ∧ j = 1 then 5
    else 0  -- placeholder for empty cells

/-- The main theorem -/
theorem lower_right_is_one :
  ∀ g : Grid,
    (∀ i j, initial_grid i j ≠ 0 → g i j = initial_grid i j) →
    unique_in_rows g →
    unique_in_columns g →
    g 4 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_lower_right_is_one_l587_58746


namespace NUMINAMATH_CALUDE_factorial_division_equality_l587_58710

theorem factorial_division_equality : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_equality_l587_58710


namespace NUMINAMATH_CALUDE_car_speed_ratio_l587_58749

theorem car_speed_ratio : 
  ∀ (speed_A speed_B : ℝ),
    speed_B = 50 →
    speed_A * 6 + speed_B * 2 = 1000 →
    speed_A / speed_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_ratio_l587_58749


namespace NUMINAMATH_CALUDE_expression_equality_l587_58768

theorem expression_equality (a x : ℝ) (h : a^(2*x) = Real.sqrt 2 - 1) :
  (a^(3*x) + a^(-3*x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l587_58768


namespace NUMINAMATH_CALUDE_arrange_six_books_two_pairs_l587_58769

/-- The number of ways to arrange books with some identical copies -/
def arrange_books (total : ℕ) (identical_pairs : ℕ) (unique : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial 2 ^ identical_pairs)

/-- Theorem: Arranging 6 books with 2 identical pairs and 2 unique books -/
theorem arrange_six_books_two_pairs : arrange_books 6 2 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arrange_six_books_two_pairs_l587_58769


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l587_58711

theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 4 * X^2 + 5 * X - 3
  let p₂ : Polynomial ℤ := 6 * X^2 - 5 * X + 1
  (p₁ * p₂).coeff 2 = -39 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l587_58711


namespace NUMINAMATH_CALUDE_zoey_holidays_l587_58703

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of holidays Zoey takes per month -/
def holidays_per_month : ℕ := 2

/-- The total number of holidays Zoey takes in a year -/
def total_holidays : ℕ := months_in_year * holidays_per_month

theorem zoey_holidays : total_holidays = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoey_holidays_l587_58703


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l587_58786

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → p x) ↔ (∃ x : ℝ, x > 0 ∧ ¬(p x)) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l587_58786


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l587_58761

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let circumference := π * r
  let base_radius := circumference / (2 * π)
  let slant_height := r
  let cone_height := Real.sqrt (slant_height^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * cone_height
  volume = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l587_58761


namespace NUMINAMATH_CALUDE_complete_gear_exists_l587_58798

/-- Represents a gear with a certain number of teeth and missing teeth positions -/
structure Gear where
  num_teeth : Nat
  missing_teeth : Finset Nat

/-- The problem statement -/
theorem complete_gear_exists (gear1 gear2 : Gear)
  (h1 : gear1.num_teeth = 14)
  (h2 : gear2.num_teeth = 14)
  (h3 : gear1.missing_teeth.card = 4)
  (h4 : gear2.missing_teeth.card = 4) :
  ∃ (rotation : Nat), ∀ (pos : Nat),
    pos ∈ gear1.missing_teeth →
    (pos + rotation) % gear1.num_teeth ∉ gear2.missing_teeth :=
sorry

end NUMINAMATH_CALUDE_complete_gear_exists_l587_58798


namespace NUMINAMATH_CALUDE_customers_remaining_l587_58794

theorem customers_remaining (initial : ℕ) (difference : ℕ) (final : ℕ) : 
  initial = 19 → difference = 15 → final = initial - difference → final = 4 := by
  sorry

end NUMINAMATH_CALUDE_customers_remaining_l587_58794


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l587_58716

def is_valid_price (y : ℕ) : Prop :=
  y > 0 ∧ 90 % y = 0 ∧ 100 % y = 0

theorem ticket_price_possibilities :
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), S.card = n ∧ ∀ y ∈ S, is_valid_price y) :=
sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l587_58716


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l587_58777

/-- The sum of the infinite geometric series 1 - (1/4) + (1/4)^2 - (1/4)^3 + ... -/
def infiniteGeometricSeriesSum : ℚ := 4/5

/-- The first term of the series -/
def a : ℚ := 1

/-- The common ratio of the series -/
def r : ℚ := -1/4

/-- Theorem: The sum of the infinite geometric series 1 - (1/4) + (1/4)^2 - (1/4)^3 + ... is 4/5 -/
theorem infinite_geometric_series_sum :
  infiniteGeometricSeriesSum = a / (1 - r) :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l587_58777


namespace NUMINAMATH_CALUDE_worker_number_40th_segment_l587_58733

/-- Calculates the individual number of a worker in systematic sampling -/
def systematicSamplingNumber (totalStaff : ℕ) (segments : ℕ) (startNumber : ℕ) (segmentIndex : ℕ) : ℕ :=
  startNumber + (segmentIndex - 1) * (totalStaff / segments)

/-- Proves that the individual number of the worker from the 40th segment is 394 -/
theorem worker_number_40th_segment :
  systematicSamplingNumber 620 62 4 40 = 394 := by
  sorry

#eval systematicSamplingNumber 620 62 4 40

end NUMINAMATH_CALUDE_worker_number_40th_segment_l587_58733


namespace NUMINAMATH_CALUDE_sqrt_14400_l587_58709

theorem sqrt_14400 : Real.sqrt 14400 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14400_l587_58709


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l587_58774

variables (x y : ℝ)

theorem polynomial_multiplication :
  (2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9) * (3 * x^7) =
  6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 := by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l587_58774


namespace NUMINAMATH_CALUDE_kamari_toys_l587_58735

theorem kamari_toys (kamari_toys : ℕ) (anais_toys : ℕ) :
  anais_toys = kamari_toys + 30 →
  kamari_toys + anais_toys = 160 →
  kamari_toys = 65 := by
sorry

end NUMINAMATH_CALUDE_kamari_toys_l587_58735


namespace NUMINAMATH_CALUDE_monthly_compound_interest_greater_than_yearly_l587_58799

theorem monthly_compound_interest_greater_than_yearly :
  1 + 5 / 100 < (1 + 5 / (12 * 100)) ^ 12 := by
  sorry

end NUMINAMATH_CALUDE_monthly_compound_interest_greater_than_yearly_l587_58799


namespace NUMINAMATH_CALUDE_f_minimum_at_three_halves_l587_58788

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem f_minimum_at_three_halves :
  ∃ (y : ℝ), ∀ (x : ℝ), f (3/2) ≤ f x :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_three_halves_l587_58788


namespace NUMINAMATH_CALUDE_train_crossing_time_l587_58792

/-- Time for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 270 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l587_58792


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l587_58750

theorem restaurant_cooks_count :
  ∀ (C W : ℕ),
  (C : ℚ) / W = 3 / 10 →
  (C : ℚ) / (W + 12) = 3 / 14 →
  C = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l587_58750


namespace NUMINAMATH_CALUDE_fourth_grade_classroom_count_l587_58715

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of pet hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 3

/-- The number of pet guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 1

/-- The difference between the total number of students and the total number of pets -/
def student_pet_difference : ℕ := 90

theorem fourth_grade_classroom_count :
  num_classrooms * students_per_classroom - 
  num_classrooms * (hamsters_per_classroom + guinea_pigs_per_classroom) = 
  student_pet_difference := by sorry

end NUMINAMATH_CALUDE_fourth_grade_classroom_count_l587_58715


namespace NUMINAMATH_CALUDE_divisor_congruence_l587_58722

theorem divisor_congruence (p n d : ℕ) : 
  Prime p → d ∣ ((n + 1)^p - n^p) → d ≡ 1 [MOD p] := by sorry

end NUMINAMATH_CALUDE_divisor_congruence_l587_58722


namespace NUMINAMATH_CALUDE_ellipse_properties_l587_58706

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the foci
def foci (e : Ellipse) : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_properties (e : Ellipse) (A : PointOnEllipse e)
  (h_A : A.x = 1 ∧ A.y = 1)
  (h_foci_dist : let (F1, F2) := foci e
                 Real.sqrt ((A.x - F1)^2 + (A.y - F1)^2) +
                 Real.sqrt ((A.x - F2)^2 + (A.y - F2)^2) = 4) :
  (e.a = 2 ∧ e.b^2 = 4/3) ∧
  (∀ x y, x + 3*y - 4 = 0 ↔ x^2/4 + 3*y^2/4 = 1) ∧
  (∀ C D : PointOnEllipse e,
    let k₁ := (C.y - A.y) / (C.x - A.x)
    let k₂ := (D.y - A.y) / (D.x - A.x)
    k₁ * k₂ = -1 →
    (D.y - C.y) / (D.x - C.x) = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l587_58706


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l587_58726

/-- A function satisfying the given functional equation -/
def f : ℝ → ℝ := fun _ ↦ 1

/-- The main theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y) ∧ (f 0 = 1) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l587_58726


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l587_58773

theorem brown_eyed_brunettes (total : ℕ) (blondes : ℕ) (brunettes : ℕ) (blue_eyed : ℕ) (brown_eyed : ℕ) (blue_eyed_blondes : ℕ) :
  total = 60 →
  blondes + brunettes = total →
  blue_eyed + brown_eyed = total →
  brunettes = 35 →
  blue_eyed_blondes = 20 →
  brown_eyed = 22 →
  brown_eyed - (blondes - blue_eyed_blondes) = 17 :=
by sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l587_58773


namespace NUMINAMATH_CALUDE_product_nonnegative_proof_l587_58759

theorem product_nonnegative_proof :
  -- Original proposition
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0) ∧
  -- Contrapositive is true
  (∀ x y : ℝ, x * y < 0 → x < 0 ∨ y < 0) ∧
  -- Converse is false
  ¬(∀ x y : ℝ, x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0) ∧
  -- Negation is false
  ¬(∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x * y < 0) := by
sorry

end NUMINAMATH_CALUDE_product_nonnegative_proof_l587_58759


namespace NUMINAMATH_CALUDE_original_number_proof_l587_58732

theorem original_number_proof :
  ∀ (original_number : ℤ),
    original_number + 3377 = 13200 →
    original_number = 9823 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l587_58732


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l587_58713

theorem cube_volume_from_surface_area (surface_area : ℝ) (h : surface_area = 600) :
  let side_length := Real.sqrt (surface_area / 6)
  side_length ^ 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l587_58713


namespace NUMINAMATH_CALUDE_basketball_volleyball_problem_l587_58736

/-- Given the conditions of the basketball and volleyball purchase problem,
    prove the prices of the balls and the minimum total cost. -/
theorem basketball_volleyball_problem
  (basketball_price volleyball_price : ℕ)
  (total_balls min_cost : ℕ) :
  (3 * basketball_price + volleyball_price = 360) →
  (5 * basketball_price + 3 * volleyball_price = 680) →
  (total_balls = 100) →
  (∀ x y, x + y = total_balls → x ≥ 3 * y → 
    basketball_price * x + volleyball_price * y ≥ min_cost) →
  (basketball_price = 100 ∧ 
   volleyball_price = 60 ∧
   min_cost = 9000) :=
by sorry

end NUMINAMATH_CALUDE_basketball_volleyball_problem_l587_58736


namespace NUMINAMATH_CALUDE_perpendicular_bisector_focus_condition_l587_58751

/-- A point on a parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * x^2

/-- The perpendicular bisector of two points passes through the focus of the parabola -/
def perpendicular_bisector_passes_through_focus (A B : PointOnParabola) : Prop :=
  let midpoint := ((A.x + B.x) / 2, (A.y + B.y) / 2)
  let slope := if A.x = B.x then 0 else (B.y - A.y) / (B.x - A.x)
  let perp_slope := if slope = 0 then 0 else -1 / slope
  ∃ (t : ℝ), midpoint.1 + t * perp_slope = 0 ∧ midpoint.2 + t = 1/8

/-- Theorem: The perpendicular bisector passes through the focus iff x₁ + x₂ = 0 -/
theorem perpendicular_bisector_focus_condition (A B : PointOnParabola) :
  perpendicular_bisector_passes_through_focus A B ↔ A.x + B.x = 0 := by
  sorry

/-- The equation of the perpendicular bisector when x₁ = 1 and x₂ = -3 -/
def perpendicular_bisector_equation (A B : PointOnParabola) (h₁ : A.x = 1) (h₂ : B.x = -3) : 
  ∃ (a b c : ℝ), a * A.x + b * A.y + c = 0 ∧ a * B.x + b * B.y + c = 0 ∧ (a, b, c) = (1, -4, 41) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_focus_condition_l587_58751


namespace NUMINAMATH_CALUDE_complex_equation_solution_l587_58720

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + 3 * Complex.I = (b + Complex.I) * Complex.I → a = -1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l587_58720


namespace NUMINAMATH_CALUDE_least_number_to_add_l587_58742

theorem least_number_to_add (n : ℕ) : 
  (∀ m : ℕ, m < 234 → ¬(5 ∣ (2496 + m) ∧ 7 ∣ (2496 + m) ∧ 13 ∣ (2496 + m))) ∧ 
  (5 ∣ (2496 + 234) ∧ 7 ∣ (2496 + 234) ∧ 13 ∣ (2496 + 234)) := by
  sorry

end NUMINAMATH_CALUDE_least_number_to_add_l587_58742


namespace NUMINAMATH_CALUDE_polynomial_factorization_l587_58772

theorem polynomial_factorization (x : ℝ) : x^4 + 256 = (x^2 - 8*x + 16) * (x^2 + 8*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l587_58772


namespace NUMINAMATH_CALUDE_quadratic_factorization_l587_58789

theorem quadratic_factorization (C E : ℤ) :
  (∀ x, 20 * x^2 - 87 * x + 91 = (C * x - 13) * (E * x - 7)) →
  C * E + C = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l587_58789


namespace NUMINAMATH_CALUDE_gcd_equals_2023_l587_58763

theorem gcd_equals_2023 (a b c : ℕ+) 
  (h : Nat.gcd a b + Nat.gcd a c + Nat.gcd b c = b + c + 2023) : 
  Nat.gcd b c = 2023 := by
  sorry

end NUMINAMATH_CALUDE_gcd_equals_2023_l587_58763


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l587_58723

theorem a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq :
  (∃ a b : ℝ, a = b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l587_58723


namespace NUMINAMATH_CALUDE_initial_boys_count_l587_58757

theorem initial_boys_count (t : ℕ) : 
  t > 0 →  -- Ensure the group is non-empty
  (t / 2 : ℚ) = (t : ℚ) * (1 / 2 : ℚ) →  -- Initially 50% boys
  ((t / 2 - 4 : ℚ) / (t + 2 : ℚ) = (2 / 5 : ℚ)) →  -- After changes, 40% boys
  t / 2 = 24 := by  -- Initial number of boys is 24
sorry

end NUMINAMATH_CALUDE_initial_boys_count_l587_58757


namespace NUMINAMATH_CALUDE_expression_simplification_l587_58791

theorem expression_simplification (x : ℤ) (h : x = 2018) :
  x^2 + 2*x - x*(x + 1) = 2018 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l587_58791


namespace NUMINAMATH_CALUDE_alice_shoe_probability_l587_58779

/-- Represents the number of pairs for each color of shoes --/
structure ShoePairs where
  black : Nat
  brown : Nat
  white : Nat
  gray : Nat

/-- Calculates the probability of picking two shoes of the same color
    with one being left and one being right --/
def probability_same_color_different_feet (pairs : ShoePairs) : Rat :=
  let total_shoes := 2 * (pairs.black + pairs.brown + pairs.white + pairs.gray)
  let prob_black := (2 * pairs.black) * pairs.black / (total_shoes * (total_shoes - 1))
  let prob_brown := (2 * pairs.brown) * pairs.brown / (total_shoes * (total_shoes - 1))
  let prob_white := (2 * pairs.white) * pairs.white / (total_shoes * (total_shoes - 1))
  let prob_gray := (2 * pairs.gray) * pairs.gray / (total_shoes * (total_shoes - 1))
  prob_black + prob_brown + prob_white + prob_gray

theorem alice_shoe_probability :
  probability_same_color_different_feet ⟨7, 4, 3, 1⟩ = 25 / 145 := by
  sorry

end NUMINAMATH_CALUDE_alice_shoe_probability_l587_58779


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l587_58764

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l587_58764


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_two_roots_l587_58729

theorem at_least_one_quadratic_has_two_roots (p q₁ q₂ : ℝ) (h : p = q₁ + q₂ + 1) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ + q₁ = 0 ∧ x₂^2 + x₂ + q₁ = 0) ∨
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + p*y₁ + q₂ = 0 ∧ y₂^2 + p*y₂ + q₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_two_roots_l587_58729


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l587_58701

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_third_term (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 2 + a 3 + a 4 + a 5 = 20 →
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l587_58701


namespace NUMINAMATH_CALUDE_art_department_probability_l587_58752

theorem art_department_probability : 
  let total_students : ℕ := 4
  let students_per_grade : ℕ := 2
  let selected_students : ℕ := 2
  let different_grade_selections : ℕ := students_per_grade * students_per_grade
  let total_selections : ℕ := Nat.choose total_students selected_students
  (different_grade_selections : ℚ) / total_selections = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_art_department_probability_l587_58752


namespace NUMINAMATH_CALUDE_a_share_is_3690_l587_58790

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 3690 given the specified investments and total profit. -/
theorem a_share_is_3690 :
  calculate_share_of_profit 6300 4200 10500 12300 = 3690 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_3690_l587_58790


namespace NUMINAMATH_CALUDE_tile_problem_l587_58724

theorem tile_problem (n : ℕ) (total_tiles : ℕ) : 
  (total_tiles = n^2 + 64) ∧ (total_tiles = (n+1)^2 - 25) → total_tiles = 2000 := by
sorry

end NUMINAMATH_CALUDE_tile_problem_l587_58724


namespace NUMINAMATH_CALUDE_translator_selection_ways_l587_58700

-- Define the staff members
def total_staff : ℕ := 7
def english_only : ℕ := 3
def japanese_only : ℕ := 2
def bilingual : ℕ := 2

-- Define the required translators
def english_translators : ℕ := 3
def japanese_translators : ℕ := 2

-- Define the function to calculate the number of ways to select translators
def select_translators : ℕ := 27

-- Theorem statement
theorem translator_selection_ways :
  select_translators = 27 :=
sorry

end NUMINAMATH_CALUDE_translator_selection_ways_l587_58700


namespace NUMINAMATH_CALUDE_sum_of_quotient_digits_l587_58728

def dividend : ℕ := 111111
def divisor : ℕ := 3

def quotient : ℕ := dividend / divisor

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_quotient_digits :
  sum_of_digits quotient = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_quotient_digits_l587_58728


namespace NUMINAMATH_CALUDE_eulers_formula_l587_58765

/-- A connected planar graph -/
structure PlanarGraph where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  is_connected : Prop
  is_planar : Prop

/-- Euler's formula for connected planar graphs -/
theorem eulers_formula (G : PlanarGraph) : G.V - G.E + G.F = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l587_58765


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l587_58766

theorem reciprocal_of_negative_two :
  ((-2 : ℝ)⁻¹ : ℝ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l587_58766


namespace NUMINAMATH_CALUDE_girls_in_study_group_l587_58737

theorem girls_in_study_group (n : ℕ) :
  (Nat.choose 6 2 - Nat.choose (6 - n) 2 = 12) →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_study_group_l587_58737


namespace NUMINAMATH_CALUDE_fashion_markup_l587_58776

theorem fashion_markup (original_price : ℝ) (markup1 markup2 markup3 : ℝ) 
  (h1 : markup1 = 0.35)
  (h2 : markup2 = 0.25)
  (h3 : markup3 = 0.45) :
  let price1 := original_price * (1 + markup1)
  let price2 := price1 * (1 + markup2)
  let final_price := price2 * (1 + markup3)
  (final_price - original_price) / original_price * 100 = 144.69 := by
sorry

end NUMINAMATH_CALUDE_fashion_markup_l587_58776


namespace NUMINAMATH_CALUDE_banana_groups_indeterminate_l587_58714

theorem banana_groups_indeterminate 
  (total_bananas : ℕ) 
  (total_oranges : ℕ) 
  (orange_groups : ℕ) 
  (oranges_per_group : ℕ) 
  (h1 : total_bananas = 142) 
  (h2 : total_oranges = 356) 
  (h3 : orange_groups = 178) 
  (h4 : oranges_per_group = 2) 
  (h5 : total_oranges = orange_groups * oranges_per_group) : 
  ∀ (banana_groups : ℕ), ¬ (∃ (bananas_per_group : ℕ), total_bananas = banana_groups * bananas_per_group) :=
by sorry

end NUMINAMATH_CALUDE_banana_groups_indeterminate_l587_58714


namespace NUMINAMATH_CALUDE_mean_height_of_players_l587_58744

def heights : List ℕ := [145, 149, 151, 151, 157, 158, 163, 163, 164, 167, 168, 169, 170, 175]

def total_players : ℕ := heights.length

def sum_of_heights : ℕ := heights.sum

theorem mean_height_of_players :
  (sum_of_heights : ℚ) / (total_players : ℚ) = 160.714 := by sorry

end NUMINAMATH_CALUDE_mean_height_of_players_l587_58744


namespace NUMINAMATH_CALUDE_cornbread_pieces_l587_58712

-- Define the dimensions of the pan
def pan_length : ℕ := 20
def pan_width : ℕ := 18

-- Define the dimensions of each piece of cornbread
def piece_length : ℕ := 2
def piece_width : ℕ := 2

-- Theorem to prove
theorem cornbread_pieces :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 := by
  sorry


end NUMINAMATH_CALUDE_cornbread_pieces_l587_58712


namespace NUMINAMATH_CALUDE_cubic_equation_product_l587_58795

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2015 ∧ y₁^3 - 3*x₁^2*y₁ = 2014)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2015 ∧ y₂^3 - 3*x₂^2*y₂ = 2014)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2015 ∧ y₃^3 - 3*x₃^2*y₃ = 2014) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -4/1007 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l587_58795


namespace NUMINAMATH_CALUDE_part_i_part_ii_l587_58745

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x + 1 + 2 / x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 1 / x - 2 / (x^2)

-- Theorem for part I
theorem part_i :
  (∃ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, x > 0 → (f x - f a = -1/a * (x - a)) → (x = 0 → f x = 4)) ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f' x < 0) ∧
    (∀ x : ℝ, x > 2 → f' x > 0)) :=
sorry

-- Theorem for part II
theorem part_ii :
  (∃ k : ℤ, k = 7 ∧
    (∀ x : ℝ, x > 1 → 2 * f x > k * (1 - 1/x)) ∧
    (∀ m : ℤ, m > k → ∃ x : ℝ, x > 1 ∧ 2 * f x ≤ m * (1 - 1/x))) :=
sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l587_58745


namespace NUMINAMATH_CALUDE_simplify_fraction_l587_58702

theorem simplify_fraction : (45 : ℚ) / 75 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l587_58702


namespace NUMINAMATH_CALUDE_combination_sequence_implies_value_l587_58730

theorem combination_sequence_implies_value (n : ℕ) : 
  (2 * (n.choose 5) = (n.choose 4) + (n.choose 6)) → 
  (n.choose 12) = 91 := by sorry

end NUMINAMATH_CALUDE_combination_sequence_implies_value_l587_58730


namespace NUMINAMATH_CALUDE_cobbler_working_hours_l587_58781

/-- Represents the number of pairs of shoes a cobbler can mend in an hour -/
def shoes_per_hour : ℕ := 3

/-- Represents the number of hours the cobbler works on Friday -/
def friday_hours : ℕ := 3

/-- Represents the total number of pairs of shoes the cobbler can mend in a week -/
def total_shoes_per_week : ℕ := 105

/-- Represents the number of working days from Monday to Thursday -/
def working_days : ℕ := 4

theorem cobbler_working_hours :
  ∃ (h : ℕ), h * working_days * shoes_per_hour + friday_hours * shoes_per_hour = total_shoes_per_week ∧ h = 8 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_working_hours_l587_58781


namespace NUMINAMATH_CALUDE_max_guaranteed_sum_l587_58739

/-- A strategy for placing signs in the game -/
def Strategy := Fin 20 → Bool

/-- The set of numbers used in the game -/
def GameNumbers : Finset ℕ := Finset.range 20

/-- Calculate the sum given a strategy -/
def calculateSum (s : Strategy) : ℤ :=
  (Finset.sum GameNumbers fun i => if s i then (i + 1) else -(i + 1))

/-- Player B's objective is to maximize the absolute value of the sum -/
def playerBObjective (s : Strategy) : ℕ := Int.natAbs (calculateSum s)

/-- The theorem stating the maximum value Player B can guarantee -/
theorem max_guaranteed_sum :
  ∃ (s : Strategy), ∀ (t : Strategy), playerBObjective s ≥ 30 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_sum_l587_58739


namespace NUMINAMATH_CALUDE_circular_seating_arrangements_l587_58796

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem circular_seating_arrangements :
  let total_people : ℕ := 8
  let seats : ℕ := 7
  let reserved_seats : ℕ := 1
  let remaining_people : ℕ := total_people - reserved_seats
  let people_to_arrange : ℕ := seats - reserved_seats
  
  (choose remaining_people people_to_arrange * factorial people_to_arrange) / seats = 720 := by
sorry

end NUMINAMATH_CALUDE_circular_seating_arrangements_l587_58796


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l587_58771

theorem buckingham_palace_visitors :
  let current_day_visitors : ℕ := 132
  let previous_day_visitors : ℕ := 274
  let total_visitors : ℕ := 406
  let days_considered : ℕ := 2
  current_day_visitors + previous_day_visitors = total_visitors →
  days_considered = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l587_58771


namespace NUMINAMATH_CALUDE_lawn_length_l587_58784

/-- Given a rectangular lawn with specified conditions, prove its length is 70 meters -/
theorem lawn_length (width : ℝ) (road_width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) : 
  width = 60 → 
  road_width = 10 → 
  total_cost = 3600 → 
  cost_per_sqm = 3 → 
  ∃ (length : ℝ), 
    (road_width * length + road_width * (width - road_width)) * cost_per_sqm = total_cost ∧ 
    length = 70 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_l587_58784


namespace NUMINAMATH_CALUDE_simplify_expression_l587_58778

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 - 2*b) - 4*b^2 = 9*b^4 - 10*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l587_58778
