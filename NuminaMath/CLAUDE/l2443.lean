import Mathlib

namespace NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l2443_244351

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, if n is divisible by 1998, 
    then the sum of its digits is greater than or equal to 27 -/
theorem divisible_by_1998_digit_sum (n : ℕ) : 
  n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l2443_244351


namespace NUMINAMATH_CALUDE_subtract_negative_four_minus_negative_seven_l2443_244386

theorem subtract_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem four_minus_negative_seven : 4 - (-7) = 11 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_four_minus_negative_seven_l2443_244386


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2443_244381

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) →  -- geometric sequence condition
  (a 5) ^ 2 + 2016 * (a 5) + 9 = 0 →  -- a_5 is a root of the equation
  (a 9) ^ 2 + 2016 * (a 9) + 9 = 0 →  -- a_9 is a root of the equation
  a 7 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2443_244381


namespace NUMINAMATH_CALUDE_solution_set_nonempty_implies_m_greater_than_five_l2443_244312

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 4| + m

-- State the theorem
theorem solution_set_nonempty_implies_m_greater_than_five :
  (∃ x : ℝ, f x < g x m) → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_implies_m_greater_than_five_l2443_244312


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l2443_244318

theorem square_minus_product_equals_one : 2015^2 - 2016 * 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l2443_244318


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l2443_244363

/-- Given a tetrahedron with product of opposite edges equal to 1,
    angles α, β, γ between opposite edges, and face circumradii R₁, R₂, R₃, R₄,
    prove that sin²α + sin²β + sin²γ ≥ 1/√(R₁R₂R₃R₄) -/
theorem tetrahedron_inequality
  (α β γ R₁ R₂ R₃ R₄ : ℝ)
  (h_positive : R₁ > 0 ∧ R₂ > 0 ∧ R₃ > 0 ∧ R₄ > 0)
  (h_product : ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l → 
    ∃ (a_ij a_kl : ℝ), a_ij * a_kl = 1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 ≥ 1 / Real.sqrt (R₁ * R₂ * R₃ * R₄) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l2443_244363


namespace NUMINAMATH_CALUDE_car_speed_problem_l2443_244329

/-- Proves that car R's speed is 30 mph given the conditions of the problem -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ)
  (h1 : distance = 300)
  (h2 : time_diff = 2)
  (h3 : speed_diff = 10)
  (h4 : distance / (car_r_speed + speed_diff) + time_diff = distance / car_r_speed)
  : car_r_speed = 30 :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l2443_244329


namespace NUMINAMATH_CALUDE_gcd_119_34_l2443_244364

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_119_34_l2443_244364


namespace NUMINAMATH_CALUDE_max_value_a_l2443_244361

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 2 * c)
  (h3 : c < 5 * d)
  (h4 : d < 50) :
  a ≤ 1460 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1460 ∧ 
    a' < 3 * b' ∧ 
    b' < 2 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_a_l2443_244361


namespace NUMINAMATH_CALUDE_evaluate_expression_l2443_244346

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : 2*x - b + 5 = b + 23 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2443_244346


namespace NUMINAMATH_CALUDE_problem_1_l2443_244325

theorem problem_1 : 6 * (1/3 - 1/2) - 3^2 / (-12) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2443_244325


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2443_244393

theorem min_distance_to_line (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ 8 * x₀ + 15 * y₀ = 120 ∧
  (∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → 8 * x' + 15 * y' = 120 → 
    Real.sqrt (x₀^2 + y₀^2) ≤ Real.sqrt (x'^2 + y'^2)) ∧
  Real.sqrt (x₀^2 + y₀^2) = 120 / 17 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2443_244393


namespace NUMINAMATH_CALUDE_abc_product_l2443_244357

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : 1 / a + 1 / b + 1 / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l2443_244357


namespace NUMINAMATH_CALUDE_stamp_arrangement_count_l2443_244337

/-- Represents a stamp with its value in cents -/
structure Stamp where
  value : Nat
  deriving Repr

/-- Represents an arrangement of stamps -/
def Arrangement := List Stamp

/-- Checks if an arrangement is valid (sums to 15 cents) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  (arr.map (·.value)).sum = 15

/-- Checks if two arrangements are considered the same -/
def isSameArrangement (arr1 arr2 : Arrangement) : Bool :=
  sorry  -- Implementation details omitted

/-- Generates all possible stamp arrangements -/
def generateArrangements (stamps : List (Nat × Nat)) : List Arrangement :=
  sorry  -- Implementation details omitted

/-- Counts unique arrangements -/
def countUniqueArrangements (arrangements : List Arrangement) : Nat :=
  sorry  -- Implementation details omitted

/-- The main theorem to prove -/
theorem stamp_arrangement_count :
  let stamps := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
  let arrangements := generateArrangements stamps
  let validArrangements := arrangements.filter isValidArrangement
  countUniqueArrangements validArrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_stamp_arrangement_count_l2443_244337


namespace NUMINAMATH_CALUDE_total_dogs_count_l2443_244309

/-- The number of boxes of stuffed toy dogs -/
def num_boxes : ℕ := 15

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 8

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem total_dogs_count : total_dogs = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_count_l2443_244309


namespace NUMINAMATH_CALUDE_arc_length_sector_l2443_244374

/-- The arc length of a circular sector with central angle 90° and radius 6 is 3π. -/
theorem arc_length_sector (θ : ℝ) (r : ℝ) (h1 : θ = 90) (h2 : r = 6) :
  (θ / 360) * (2 * Real.pi * r) = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_arc_length_sector_l2443_244374


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_four_sqrt_five_l2443_244300

theorem sqrt_difference_equals_negative_four_sqrt_five :
  Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5) = -4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_four_sqrt_five_l2443_244300


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l2443_244330

theorem min_value_sqrt_expression (x : ℝ) :
  Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3) ≥ Real.sqrt 7 ∧
  (Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3) = Real.sqrt 7 ↔ x = Real.sqrt 3 / 4 ∨ x = -Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l2443_244330


namespace NUMINAMATH_CALUDE_circumcircles_intersect_at_single_point_l2443_244311

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define central symmetry
def centrally_symmetric (t1 t2 : Triangle) (center : Point) : Prop :=
  ∃ (O : Point),
    (t1.A.x + t2.A.x) / 2 = O.x ∧ (t1.A.y + t2.A.y) / 2 = O.y ∧
    (t1.B.x + t2.B.x) / 2 = O.x ∧ (t1.B.y + t2.B.y) / 2 = O.y ∧
    (t1.C.x + t2.C.x) / 2 = O.x ∧ (t1.C.y + t2.C.y) / 2 = O.y

-- Define circumcircle
def circumcircle (t : Triangle) : Circle :=
  sorry

-- Define intersection of circles
def intersect (c1 c2 : Circle) : Set Point :=
  sorry

theorem circumcircles_intersect_at_single_point
  (ABC A₁B₁C₁ : Triangle)
  (h : centrally_symmetric ABC A₁B₁C₁ (Point.mk 0 0)) :
  ∃ (S : Point),
    S ∈ intersect (circumcircle ABC) (circumcircle (Triangle.mk A₁B₁C₁.A ABC.B A₁B₁C₁.C)) ∧
    S ∈ intersect (circumcircle (Triangle.mk A₁B₁C₁.A A₁B₁C₁.B ABC.C)) (circumcircle (Triangle.mk ABC.A A₁B₁C₁.B A₁B₁C₁.C)) :=
sorry

end NUMINAMATH_CALUDE_circumcircles_intersect_at_single_point_l2443_244311


namespace NUMINAMATH_CALUDE_smallest_abs_rational_l2443_244362

theorem smallest_abs_rational : ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_rational_l2443_244362


namespace NUMINAMATH_CALUDE_not_perfect_square_l2443_244389

theorem not_perfect_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
  ¬ ∃ k : ℕ, 3^m + 3^n + 1 = k^2 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2443_244389


namespace NUMINAMATH_CALUDE_insertPluses_l2443_244302

/-- The number of ones in the original number -/
def n : ℕ := 15

/-- The number of plus signs to be inserted -/
def k : ℕ := 9

/-- The number of spaces between the ones where plus signs can be inserted -/
def spaces : ℕ := n - 1

-- Statement of the theorem
theorem insertPluses : 
  (Nat.choose spaces k : ℕ) = (2002 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_insertPluses_l2443_244302


namespace NUMINAMATH_CALUDE_family_children_count_l2443_244322

theorem family_children_count :
  ∀ (num_children : ℕ),
    (5 * (num_children + 3) + 2 * num_children + 4 * 3 = 55) →
    num_children = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_family_children_count_l2443_244322


namespace NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l2443_244326

/-- Represents a 12-hour digital clock with a defect that displays 1 instead of 2 --/
structure DefectiveClock :=
  (hours : Fin 12)
  (minutes : Fin 60)

/-- Checks if the given hour is displayed correctly --/
def hour_correct (h : Fin 12) : Bool :=
  h ≠ 2 ∧ h ≠ 12

/-- Checks if the given minute is displayed correctly --/
def minute_correct (m : Fin 60) : Bool :=
  m % 10 ≠ 2 ∧ m / 10 ≠ 2

/-- The fraction of the day during which the clock displays the correct time --/
def correct_time_fraction (clock : DefectiveClock) : ℚ :=
  (5 : ℚ) / 8

theorem defective_clock_correct_time_fraction :
  ∀ (clock : DefectiveClock),
  correct_time_fraction clock = (5 : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l2443_244326


namespace NUMINAMATH_CALUDE_quadratic_two_roots_implies_a_le_two_l2443_244344

/-- 
Given a quadratic equation x^2 - 4x + 2a = 0 with parameter a,
if the equation has two real roots, then a ≤ 2.
-/
theorem quadratic_two_roots_implies_a_le_two : 
  ∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x^2 - 4*x + 2*a = 0 ∧ y^2 - 4*y + 2*a = 0) → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_implies_a_le_two_l2443_244344


namespace NUMINAMATH_CALUDE_savings_proof_l2443_244340

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the person's savings are 3400 -/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (h1 : income = 17000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 3400 := by
  sorry

#eval calculate_savings 17000 5 4

end NUMINAMATH_CALUDE_savings_proof_l2443_244340


namespace NUMINAMATH_CALUDE_paper_cup_probability_l2443_244387

theorem paper_cup_probability (total_tosses : ℕ) (mouth_up_occurrences : ℕ) 
  (h1 : total_tosses = 200) (h2 : mouth_up_occurrences = 48) :
  (mouth_up_occurrences : ℚ) / total_tosses = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_paper_cup_probability_l2443_244387


namespace NUMINAMATH_CALUDE_hydropower_station_calculations_l2443_244391

-- Define constants
def generator_power : Real := 24.5 * 1000  -- in watts
def generator_voltage : Real := 350
def line_resistance : Real := 4
def power_loss_percentage : Real := 0.05
def user_voltage : Real := 220

-- Define the theorem
theorem hydropower_station_calculations :
  let line_current := Real.sqrt ((power_loss_percentage * generator_power) / line_resistance)
  let step_up_ratio := (generator_power / generator_voltage) / line_current
  let step_down_input_voltage := generator_voltage - line_current * line_resistance
  let step_down_ratio := step_down_input_voltage / user_voltage
  (line_current = 17.5) ∧
  (step_up_ratio = 4) ∧
  (step_down_ratio = 133 / 22) := by
  sorry

end NUMINAMATH_CALUDE_hydropower_station_calculations_l2443_244391


namespace NUMINAMATH_CALUDE_shortest_ribbon_length_l2443_244382

theorem shortest_ribbon_length (ribbon_length : ℕ) : 
  (ribbon_length % 2 = 0 ∧ ribbon_length % 5 = 0) → 
  ribbon_length ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_shortest_ribbon_length_l2443_244382


namespace NUMINAMATH_CALUDE_right_triangle_area_with_specific_ratios_l2443_244310

/-- 
Theorem: Area of a right triangle with specific leg and hypotenuse relationships

Given a right triangle where:
- One leg is 1/3 longer than the other leg
- The same leg is 1/3 shorter than the hypotenuse

The area of this triangle is equal to 2/3 times the square of the shorter leg.
-/
theorem right_triangle_area_with_specific_ratios 
  (a b c : ℝ) -- a, b are legs, c is hypotenuse
  (h_right : a^2 + b^2 = c^2) -- right triangle condition
  (h_leg_ratio : a = (4/3) * b) -- one leg is 1/3 longer than the other
  (h_hyp_ratio : a = (2/3) * c) -- the same leg is 1/3 shorter than hypotenuse
  : (1/2) * a * b = (2/3) * b^2 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_with_specific_ratios_l2443_244310


namespace NUMINAMATH_CALUDE_income_of_M_l2443_244307

theorem income_of_M (M N O : ℝ) 
  (avg_MN : (M + N) / 2 = 5050)
  (avg_NO : (N + O) / 2 = 6250)
  (avg_MO : (M + O) / 2 = 5200) :
  M = 2666.67 := by
  sorry

end NUMINAMATH_CALUDE_income_of_M_l2443_244307


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_proof_37_seats_for_150_l2443_244396

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats + 3) / 4

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

/-- Proves that 37 is the minimum number of occupied seats required
    for 150 total seats to ensure the next person sits next to someone. -/
theorem proof_37_seats_for_150 :
  ∀ n : ℕ, n < min_occupied_seats 150 →
    ∃ arrangement : Fin 150 → Bool,
      (∀ i : Fin 150, arrangement i = true → i.val < n) ∧
      ∃ j : Fin 150, (∀ k : Fin 150, k.val = j.val - 1 ∨ k.val = j.val + 1 → arrangement k = false) := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_proof_37_seats_for_150_l2443_244396


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l2443_244345

-- Define the foci
def F₁ : ℝ × ℝ := (0, 4)
def F₂ : ℝ × ℝ := (6, 4)

-- Define the ellipse
def Ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  Real.sqrt ((x - x₁)^2 + (y - y₁)^2) + Real.sqrt ((x - x₂)^2 + (y - y₂)^2) = 10

-- Define the ellipse equation parameters
def h : ℝ := sorry
def k : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry

-- State the theorem
theorem ellipse_parameter_sum :
  h + k + a + b = 16 := by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l2443_244345


namespace NUMINAMATH_CALUDE_last_released_position_l2443_244305

/-- Represents the state of the ransom process -/
structure RansomState where
  remaining_captives : ℕ
  purses_on_table : ℕ
  last_released_position : ℕ

/-- Simulates the ransom process for Robin Hood's captives -/
def ransom_process (initial_captives : ℕ) : ℕ → RansomState := sorry

/-- Theorem stating the position of the last released captive based on the final number of purses -/
theorem last_released_position 
  (initial_captives : ℕ) 
  (final_purses : ℕ) :
  initial_captives = 7 →
  (final_purses = 28 → (ransom_process initial_captives final_purses).last_released_position = 7) ∧
  (final_purses = 27 → 
    ((ransom_process initial_captives final_purses).last_released_position = 6 ∨
     (ransom_process initial_captives final_purses).last_released_position = 7)) :=
by sorry

end NUMINAMATH_CALUDE_last_released_position_l2443_244305


namespace NUMINAMATH_CALUDE_find_k_l2443_244353

theorem find_k (k : ℝ) (h : 16 / k = 4) : k = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2443_244353


namespace NUMINAMATH_CALUDE_rahul_share_l2443_244372

/-- Calculates the share of payment for a worker given the total payment and the time taken by both workers to complete the job individually --/
def calculateShare (totalPayment : ℚ) (worker1Time : ℚ) (worker2Time : ℚ) : ℚ :=
  let worker1Rate := 1 / worker1Time
  let worker2Rate := 1 / worker2Time
  let totalRate := worker1Rate + worker2Rate
  let worker1Share := worker1Rate / totalRate
  worker1Share * totalPayment

/-- Proves that Rahul's share of the payment is $42 given the specified conditions --/
theorem rahul_share :
  let rahulTime := 3
  let rajeshTime := 2
  let totalPayment := 105
  calculateShare totalPayment rahulTime rajeshTime = 42 := by
  sorry

#eval calculateShare 105 3 2

end NUMINAMATH_CALUDE_rahul_share_l2443_244372


namespace NUMINAMATH_CALUDE_fourth_grade_students_end_of_year_l2443_244335

/-- Calculates the total number of students at the end of the year given the initial number,
    students added during the year, and new students who came to school. -/
def total_students (initial : ℝ) (added : ℝ) (new_students : ℝ) : ℝ :=
  initial + added + new_students

/-- Proves that given the specific numbers in the problem, the total number of students
    at the end of the year is 56.0. -/
theorem fourth_grade_students_end_of_year :
  total_students 10.0 4.0 42.0 = 56.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_end_of_year_l2443_244335


namespace NUMINAMATH_CALUDE_negative_two_b_cubed_l2443_244378

theorem negative_two_b_cubed (b : ℝ) : (-2 * b)^3 = -8 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_b_cubed_l2443_244378


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_l2443_244342

theorem increasing_sequence_condition (a : ℕ → ℝ) (b : ℝ) :
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) →
  (∀ n : ℕ, n > 0 → a n = n^2 + b*n) →
  b > -3 :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_condition_l2443_244342


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l2443_244360

/-- Given a curve y = ax + ln x with a tangent line y = 2x + b at the point (1, a), prove that b = -1 -/
theorem tangent_line_intercept (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x => a * x + Real.log x) →  -- Curve definition
  (∃ (g : ℝ → ℝ), g = λ x => 2 * x + b) →           -- Tangent line definition
  (∃ (x₀ : ℝ), x₀ = 1 ∧ f x₀ = a) →                 -- Point of tangency
  (∀ x, (deriv f) x = a + 1 / x) →                  -- Derivative of f
  (deriv f) 1 = 2 →                                 -- Slope at x = 1 equals 2
  b = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l2443_244360


namespace NUMINAMATH_CALUDE_trig_simplification_l2443_244397

theorem trig_simplification :
  (Real.cos (40 * π / 180)) / (Real.cos (25 * π / 180) * Real.sqrt (1 - Real.sin (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2443_244397


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l2443_244313

theorem square_diagonal_ratio (s S : ℝ) (h_perimeter_ratio : 4 * S = 4 * (4 * s)) :
  S * Real.sqrt 2 = 4 * (s * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l2443_244313


namespace NUMINAMATH_CALUDE_circles_intersect_l2443_244365

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 - 12*x + y^2 - 8*y - 12 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 - 10*y + 34 = 0

/-- The shortest distance between the two circles -/
def shortest_distance : ℝ := 0

/-- Theorem stating that the shortest distance between the two circles is 0 -/
theorem circles_intersect : 
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ shortest_distance = 0 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l2443_244365


namespace NUMINAMATH_CALUDE_algebraic_expression_inconsistency_l2443_244388

theorem algebraic_expression_inconsistency (a b : ℤ) :
  (-a + b = -1) ∧ (a + b = 5) ∧ (4*a + b = 14) →
  (2*a + b ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_inconsistency_l2443_244388


namespace NUMINAMATH_CALUDE_train_length_l2443_244367

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 18 → speed_kmh * (1000 / 3600) * time_s = 900 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2443_244367


namespace NUMINAMATH_CALUDE_quadratic_solution_l2443_244370

theorem quadratic_solution (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9) - 36 = 0) → c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2443_244370


namespace NUMINAMATH_CALUDE_tangent_line_to_sine_curve_l2443_244379

theorem tangent_line_to_sine_curve (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.sin (t + Real.pi / 3)
  let point : ℝ × ℝ := (0, Real.sqrt 3 / 2)
  let tangent_equation : ℝ → ℝ → Prop := λ x y => x - 2 * y + Real.sqrt 3 = 0
  (∀ t, f t = Real.sin (t + Real.pi / 3)) →
  (point.1 = 0 ∧ point.2 = Real.sqrt 3 / 2) →
  (∃ k, ∀ x, tangent_equation x (k * x + point.2)) →
  tangent_equation x y = (x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_tangent_line_to_sine_curve_l2443_244379


namespace NUMINAMATH_CALUDE_training_schedule_days_l2443_244348

/-- Calculates the number of days required to complete a training schedule. -/
def trainingDays (totalHours : ℕ) (multiplicationMinutes : ℕ) (divisionMinutes : ℕ) : ℕ :=
  let totalMinutes := totalHours * 60
  let dailyMinutes := multiplicationMinutes + divisionMinutes
  totalMinutes / dailyMinutes

/-- Proves that the training schedule takes 10 days to complete. -/
theorem training_schedule_days :
  trainingDays 5 10 20 = 10 := by
  sorry

#eval trainingDays 5 10 20

end NUMINAMATH_CALUDE_training_schedule_days_l2443_244348


namespace NUMINAMATH_CALUDE_remainder_problem_l2443_244359

theorem remainder_problem (N : ℤ) (h : N % 296 = 75) : N % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2443_244359


namespace NUMINAMATH_CALUDE_zoo_visitors_l2443_244320

theorem zoo_visitors (adult_price kid_price total_sales num_kids : ℕ) 
  (h1 : adult_price = 28)
  (h2 : kid_price = 12)
  (h3 : total_sales = 3864)
  (h4 : num_kids = 203) :
  ∃ (num_adults : ℕ), 
    adult_price * num_adults + kid_price * num_kids = total_sales ∧
    num_adults + num_kids = 254 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l2443_244320


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2443_244383

/-- A perfect square trinomial in the form x^2 + ax + 4 -/
def is_perfect_square_trinomial (a : ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, x^2 + a*x + 4 = (x + b)^2

/-- If x^2 + ax + 4 is a perfect square trinomial, then a = ±4 -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  is_perfect_square_trinomial a → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2443_244383


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l2443_244384

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l2443_244384


namespace NUMINAMATH_CALUDE_root_sum_magnitude_l2443_244332

theorem root_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ →
  r₁^2 + p*r₁ + 9 = 0 →
  r₂^2 + p*r₂ + 9 = 0 →
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_root_sum_magnitude_l2443_244332


namespace NUMINAMATH_CALUDE_art_gallery_visitors_prove_initial_girls_l2443_244395

theorem art_gallery_visitors : ℕ → ℕ → Prop :=
  fun girls boys =>
    -- After 15 girls left, there were twice as many boys as girls remaining
    boys = 2 * (girls - 15) ∧
    -- After 45 boys left, there were five times as many girls as boys remaining
    (girls - 15) = 5 * (boys - 45) ∧
    -- The number of girls initially in the gallery is 40
    girls = 40

-- The theorem to prove
theorem prove_initial_girls : ∃ (girls boys : ℕ), art_gallery_visitors girls boys :=
  sorry

end NUMINAMATH_CALUDE_art_gallery_visitors_prove_initial_girls_l2443_244395


namespace NUMINAMATH_CALUDE_compound_interest_theorem_specific_case_calculation_l2443_244354

/-- Compound interest calculation function -/
def compound_interest (a : ℝ) (r : ℝ) (x : ℕ) : ℝ :=
  a * (1 + r) ^ x

/-- Theorem for compound interest calculation -/
theorem compound_interest_theorem (a r : ℝ) (x : ℕ) :
  compound_interest a r x = a * (1 + r) ^ x :=
by sorry

/-- Specific case calculation -/
theorem specific_case_calculation :
  let a : ℝ := 1000
  let r : ℝ := 0.0225
  let x : ℕ := 4
  abs (compound_interest a r x - 1093.08) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_theorem_specific_case_calculation_l2443_244354


namespace NUMINAMATH_CALUDE_voice_area_greater_than_ground_area_l2443_244306

/-- The side length of the square ground in meters -/
def ground_side : ℝ := 25

/-- The maximum distance the trainer's voice can be heard in meters -/
def voice_range : ℝ := 140

/-- The area of the ground where the trainer's voice can be heard is greater than the area of the square ground -/
theorem voice_area_greater_than_ground_area : π * voice_range^2 > ground_side^2 := by
  sorry

end NUMINAMATH_CALUDE_voice_area_greater_than_ground_area_l2443_244306


namespace NUMINAMATH_CALUDE_gift_contribution_theorem_l2443_244347

theorem gift_contribution_theorem (n : ℕ) (min_contribution max_contribution total : ℝ) :
  n = 12 →
  min_contribution = 1 →
  max_contribution = 9 →
  (∀ person, person ∈ Finset.range n → min_contribution ≤ person) →
  (∀ person, person ∈ Finset.range n → person ≤ max_contribution) →
  total = (n - 1) * min_contribution + max_contribution →
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_gift_contribution_theorem_l2443_244347


namespace NUMINAMATH_CALUDE_mikaela_paint_containers_l2443_244316

/-- Represents the number of paint containers Mikaela initially bought. -/
def initial_containers : ℕ := 8

/-- Represents the number of walls Mikaela initially planned to paint. -/
def planned_walls : ℕ := 4

/-- Represents the number of containers used for the ceiling. -/
def ceiling_containers : ℕ := 1

/-- Represents the number of containers left over. -/
def leftover_containers : ℕ := 3

/-- Represents the number of walls Mikaela actually painted. -/
def painted_walls : ℕ := 3

theorem mikaela_paint_containers :
  initial_containers = 
    ceiling_containers + leftover_containers + (planned_walls - painted_walls) :=
by sorry

end NUMINAMATH_CALUDE_mikaela_paint_containers_l2443_244316


namespace NUMINAMATH_CALUDE_scooter_price_l2443_244377

theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l2443_244377


namespace NUMINAMATH_CALUDE_children_ticket_price_l2443_244317

/-- The price of an adult ticket in dollars -/
def adult_price : ℚ := 8

/-- The total revenue in dollars -/
def total_revenue : ℚ := 236

/-- The total number of tickets sold -/
def total_tickets : ℕ := 34

/-- The number of adult tickets sold -/
def adult_tickets : ℕ := 12

/-- The price of a children's ticket in dollars -/
def children_price : ℚ := (total_revenue - adult_price * adult_tickets) / (total_tickets - adult_tickets)

theorem children_ticket_price :
  children_price = 6.36 := by sorry

end NUMINAMATH_CALUDE_children_ticket_price_l2443_244317


namespace NUMINAMATH_CALUDE_angle_identities_l2443_244350

theorem angle_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.cos α = 1 / 3) : 
  Real.tan α = 2 * Real.sqrt 2 ∧ 
  (Real.sqrt 2 * Real.sin (Real.pi + α) + 2 * Real.cos α) / 
  (Real.cos α - Real.sqrt 2 * Real.cos (Real.pi / 2 + α)) = -2 / 5 := by
sorry

end NUMINAMATH_CALUDE_angle_identities_l2443_244350


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2443_244369

theorem angle_measure_proof (x : ℝ) (h1 : x + (3 * x + 10) = 90) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2443_244369


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l2443_244376

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_is_45 : exterior_angle_regular_octagon = 45 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l2443_244376


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2443_244319

theorem cubic_root_sum (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → 
  (β^3 - β - 1 = 0) → 
  (γ^3 - γ - 1 = 0) → 
  ((1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2443_244319


namespace NUMINAMATH_CALUDE_solve_parking_problem_l2443_244303

def parking_problem (initial_balance : ℚ) (first_three_cost : ℚ) (fourth_cost_ratio : ℚ) (fifth_cost_ratio : ℚ) (roommate_payment_ratio : ℚ) : Prop :=
  let total_first_three := 3 * first_three_cost
  let fourth_ticket_cost := fourth_cost_ratio * first_three_cost
  let fifth_ticket_cost := fifth_cost_ratio * first_three_cost
  let total_cost := total_first_three + fourth_ticket_cost + fifth_ticket_cost
  let roommate_payment := roommate_payment_ratio * total_cost
  let james_payment := total_cost - roommate_payment
  let remaining_balance := initial_balance - james_payment
  remaining_balance = 871.88

theorem solve_parking_problem :
  parking_problem 1200 250 (1/4) (1/2) 0.65 :=
sorry

end NUMINAMATH_CALUDE_solve_parking_problem_l2443_244303


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2443_244399

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) :
  parallel m n → 
  perpendicular_line_plane n β → 
  perpendicular_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2443_244399


namespace NUMINAMATH_CALUDE_prism_height_l2443_244336

/-- Regular prism with base ABC and top A₁B₁C₁ -/
structure RegularPrism where
  a : ℝ  -- side length of the base
  h : ℝ  -- height of the prism
  M : ℝ × ℝ × ℝ  -- midpoint of AC
  N : ℝ × ℝ × ℝ  -- midpoint of A₁B₁

/-- The projection of MN onto BA₁ is a/(2√6) -/
def projection_condition (prism : RegularPrism) : Prop :=
  ∃ (proj : ℝ), proj = prism.a / (2 * Real.sqrt 6)

/-- The theorem stating the possible heights of the prism -/
theorem prism_height (prism : RegularPrism) 
  (h_proj : projection_condition prism) :
  prism.h = prism.a / Real.sqrt 2 ∨ 
  prism.h = prism.a / (2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_prism_height_l2443_244336


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l2443_244334

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_negative_fraction :
  reciprocal (-5/4) = -4/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l2443_244334


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2443_244343

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_monotone_increasing_on_positive (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem solution_set_of_inequality 
  (h_odd : is_odd f)
  (h_monotone : is_monotone_increasing_on_positive f)
  (h_f1 : f 1 = 0) :
  {x : ℝ | (f x - f (-x)) / x > 0} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2443_244343


namespace NUMINAMATH_CALUDE_jim_reading_pages_l2443_244390

/-- Calculates the number of pages Jim reads per week after changing his reading speed and time --/
def pages_read_per_week (
  regular_rate : ℝ)
  (technical_rate : ℝ)
  (regular_time : ℝ)
  (technical_time : ℝ)
  (regular_speed_increase : ℝ)
  (technical_speed_increase : ℝ)
  (regular_time_reduction : ℝ)
  (technical_time_reduction : ℝ) : ℝ :=
  let new_regular_rate := regular_rate * regular_speed_increase
  let new_technical_rate := technical_rate * technical_speed_increase
  let new_regular_time := regular_time - regular_time_reduction
  let new_technical_time := technical_time - technical_time_reduction
  (new_regular_rate * new_regular_time) + (new_technical_rate * new_technical_time)

theorem jim_reading_pages : 
  pages_read_per_week 40 30 10 5 1.5 1.3 4 2 = 477 := by
  sorry

end NUMINAMATH_CALUDE_jim_reading_pages_l2443_244390


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2443_244315

theorem first_player_winning_strategy (a c : ℤ) : ∃ (x y z : ℤ), 
  x^3 + a*x^2 - x + c = 0 ∧ 
  y^3 + a*y^2 - y + c = 0 ∧ 
  z^3 + a*z^2 - z + c = 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2443_244315


namespace NUMINAMATH_CALUDE_hot_day_price_correct_l2443_244394

/-- Represents the lemonade stand operation --/
structure LemonadeStand where
  totalDays : ℕ
  hotDays : ℕ
  cupsPerDay : ℕ
  costPerCup : ℚ
  totalProfit : ℚ
  hotDayPriceIncrease : ℚ

/-- Calculates the price of a cup on a hot day --/
def hotDayPrice (stand : LemonadeStand) : ℚ :=
  let regularPrice := (stand.totalProfit + stand.totalDays * stand.cupsPerDay * stand.costPerCup) /
    (stand.cupsPerDay * (stand.totalDays + stand.hotDays * stand.hotDayPriceIncrease))
  regularPrice * (1 + stand.hotDayPriceIncrease)

/-- Theorem stating that the hot day price is correct --/
theorem hot_day_price_correct (stand : LemonadeStand) : 
  stand.totalDays = 10 ∧ 
  stand.hotDays = 4 ∧ 
  stand.cupsPerDay = 32 ∧ 
  stand.costPerCup = 3/4 ∧ 
  stand.totalProfit = 200 ∧
  stand.hotDayPriceIncrease = 1/4 →
  hotDayPrice stand = 25/16 := by
  sorry

#eval hotDayPrice {
  totalDays := 10
  hotDays := 4
  cupsPerDay := 32
  costPerCup := 3/4
  totalProfit := 200
  hotDayPriceIncrease := 1/4
}

end NUMINAMATH_CALUDE_hot_day_price_correct_l2443_244394


namespace NUMINAMATH_CALUDE_base_10_256_to_base_4_l2443_244314

-- Define a function to convert a natural number to its base 4 representation
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- State the theorem
theorem base_10_256_to_base_4 :
  toBase4 256 = [1, 0, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_10_256_to_base_4_l2443_244314


namespace NUMINAMATH_CALUDE_field_division_l2443_244324

theorem field_division (total_area smaller_area : ℝ) (h1 : total_area = 500) (h2 : smaller_area = 225) :
  ∃ (larger_area difference_value : ℝ),
    larger_area + smaller_area = total_area ∧
    larger_area - smaller_area = difference_value / 5 ∧
    difference_value = 250 :=
by sorry

end NUMINAMATH_CALUDE_field_division_l2443_244324


namespace NUMINAMATH_CALUDE_population_growth_factors_l2443_244333

/-- Represents a population of organisms -/
structure Population where
  density : ℝ
  genotypeFrequency : ℝ
  kValue : ℝ

/-- Factors affecting population growth -/
inductive GrowthFactor
  | BirthRate
  | DeathRate
  | CarryingCapacity

/-- Represents ideal conditions for population growth -/
def idealConditions : Prop := sorry

/-- Main factors affecting population growth under ideal conditions -/
def mainFactors : Set GrowthFactor := sorry

theorem population_growth_factors :
  idealConditions →
  mainFactors = {GrowthFactor.BirthRate, GrowthFactor.DeathRate} ∧
  GrowthFactor.CarryingCapacity ∉ mainFactors :=
sorry

end NUMINAMATH_CALUDE_population_growth_factors_l2443_244333


namespace NUMINAMATH_CALUDE_sum_of_squares_positive_and_negative_l2443_244398

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_positive_and_negative :
  2 * (sum_of_squares 50) = 85850 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_positive_and_negative_l2443_244398


namespace NUMINAMATH_CALUDE_jake_has_seven_balls_l2443_244368

/-- The number of balls Audrey has -/
def audrey_balls : ℕ := 41

/-- The difference in the number of balls between Audrey and Jake -/
def difference : ℕ := 34

/-- The number of balls Jake has -/
def jake_balls : ℕ := audrey_balls - difference

/-- Theorem stating that Jake has 7 balls -/
theorem jake_has_seven_balls : jake_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_balls_l2443_244368


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2443_244355

theorem least_positive_integer_congruence (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 9 = 8) ∧ 
  (∀ x : ℕ, x < b → ¬((x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ (x % 9 = 8))) →
  b = 179 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2443_244355


namespace NUMINAMATH_CALUDE_rectangle_formations_l2443_244341

theorem rectangle_formations (h : ℕ) (v : ℕ) (h_val : h = 5) (v_val : v = 4) :
  (Nat.choose h 2) * (Nat.choose v 2) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_formations_l2443_244341


namespace NUMINAMATH_CALUDE_gcf_48_72_l2443_244323

theorem gcf_48_72 : Nat.gcd 48 72 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcf_48_72_l2443_244323


namespace NUMINAMATH_CALUDE_tempo_value_calculation_l2443_244380

/-- The original value of a tempo given insurance and premium information -/
def tempoOriginalValue (insuredFraction : ℚ) (premiumRate : ℚ) (premiumAmount : ℚ) : ℚ :=
  premiumAmount / (premiumRate * insuredFraction)

/-- Theorem stating the original value of the tempo given the problem conditions -/
theorem tempo_value_calculation :
  let insuredFraction : ℚ := 4 / 5
  let premiumRate : ℚ := 13 / 1000
  let premiumAmount : ℚ := 910
  tempoOriginalValue insuredFraction premiumRate premiumAmount = 87500 := by
  sorry

#eval tempoOriginalValue (4/5) (13/1000) 910

end NUMINAMATH_CALUDE_tempo_value_calculation_l2443_244380


namespace NUMINAMATH_CALUDE_scores_analysis_l2443_244371

def scores : List ℕ := [7, 5, 9, 7, 4, 8, 9, 9, 7, 5]

def mode (l : List ℕ) : Set ℕ := sorry

def variance (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def percentile (l : List ℕ) (p : ℚ) : ℚ := sorry

theorem scores_analysis :
  (mode scores = {7, 9}) ∧
  (variance scores = 3) ∧
  (mean scores = 7) ∧
  (percentile scores (70/100) = 17/2) := by sorry

end NUMINAMATH_CALUDE_scores_analysis_l2443_244371


namespace NUMINAMATH_CALUDE_binomial_expansion_equal_coefficients_l2443_244352

theorem binomial_expansion_equal_coefficients (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_equal_coefficients_l2443_244352


namespace NUMINAMATH_CALUDE_conference_tables_theorem_l2443_244321

/-- Represents the available table sizes -/
inductive TableSize
  | Four
  | Six
  | Eight

/-- Calculates the minimum number of tables needed -/
def minTablesNeeded (totalInvited : ℕ) (noShows : ℕ) (tableSizes : List TableSize) : ℕ :=
  sorry

/-- Theorem stating the minimum number of tables needed for the given problem -/
theorem conference_tables_theorem (totalInvited noShows : ℕ) (tableSizes : List TableSize) :
  totalInvited = 75 →
  noShows = 33 →
  tableSizes = [TableSize.Four, TableSize.Six, TableSize.Eight] →
  minTablesNeeded totalInvited noShows tableSizes = 6 :=
sorry

end NUMINAMATH_CALUDE_conference_tables_theorem_l2443_244321


namespace NUMINAMATH_CALUDE_marathon_day_three_miles_l2443_244392

/-- Calculates the miles run on the third day of a three-day running schedule -/
def milesOnDayThree (totalMiles : ℝ) (day1Percent : ℝ) (day2Percent : ℝ) : ℝ :=
  let day1Miles := totalMiles * day1Percent
  let remainingAfterDay1 := totalMiles - day1Miles
  let day2Miles := remainingAfterDay1 * day2Percent
  totalMiles - day1Miles - day2Miles

/-- Theorem stating that given the specific conditions, the miles run on day 3 is 28 -/
theorem marathon_day_three_miles :
  milesOnDayThree 70 0.2 0.5 = 28 := by
  sorry

#eval milesOnDayThree 70 0.2 0.5

end NUMINAMATH_CALUDE_marathon_day_three_miles_l2443_244392


namespace NUMINAMATH_CALUDE_shoes_lost_l2443_244308

theorem shoes_lost (initial_pairs : ℕ) (max_pairs_left : ℕ) (h1 : initial_pairs = 25) (h2 : max_pairs_left = 20) :
  initial_pairs * 2 - max_pairs_left * 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_shoes_lost_l2443_244308


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l2443_244385

/-- Given a circle with center (0,k) where k > 6, if the circle is tangent to the lines y = x, y = -x, and y = 6, then its radius is 6√2 + 6. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 6) :
  let C := { p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - k)^2 = r^2 }
  let L1 := { p : ℝ × ℝ | p.2 = p.1 }
  let L2 := { p : ℝ × ℝ | p.2 = -p.1 }
  let L3 := { p : ℝ × ℝ | p.2 = 6 }
  (∃ (p1 : ℝ × ℝ), p1 ∈ C ∧ p1 ∈ L1) →
  (∃ (p2 : ℝ × ℝ), p2 ∈ C ∧ p2 ∈ L2) →
  (∃ (p3 : ℝ × ℝ), p3 ∈ C ∧ p3 ∈ L3) →
  r = 6 * (Real.sqrt 2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l2443_244385


namespace NUMINAMATH_CALUDE_correct_product_l2443_244339

-- Define a function to reverse the digits of a three-digit number
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

-- Define the theorem
theorem correct_product (a b : ℕ) : 
  (100 ≤ a ∧ a < 1000) →  -- a is a three-digit number
  (0 < b) →               -- b is positive
  (reverse_digits a * b = 396) →  -- erroneous product condition
  (a * b = 693) :=        -- correct product
by sorry

end NUMINAMATH_CALUDE_correct_product_l2443_244339


namespace NUMINAMATH_CALUDE_equation_solution_l2443_244375

theorem equation_solution : ∃ x : ℝ, 3 * x + 6 = |(-23 + 9)| ∧ x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2443_244375


namespace NUMINAMATH_CALUDE_savings_ratio_l2443_244358

def debt : ℕ := 40
def lulu_savings : ℕ := 6
def nora_savings : ℕ := 5 * lulu_savings
def remaining_per_person : ℕ := 2

theorem savings_ratio (tamara_savings : ℕ) 
  (h1 : nora_savings + lulu_savings + tamara_savings = debt + 3 * remaining_per_person) :
  nora_savings / tamara_savings = 3 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l2443_244358


namespace NUMINAMATH_CALUDE_student_count_l2443_244373

/-- The number of students in the class -/
def n : ℕ := sorry

/-- The total number of tokens -/
def total_tokens : ℕ := 960

/-- The number of tokens each student gives to the teacher -/
def tokens_to_teacher : ℕ := 4

theorem student_count :
  (n > 0) ∧
  (total_tokens % n = 0) ∧
  (∃ k : ℕ, k > 0 ∧ total_tokens / n - tokens_to_teacher = k ∧ k * (n + 1) = total_tokens) →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_student_count_l2443_244373


namespace NUMINAMATH_CALUDE_water_flow_restrictor_l2443_244366

theorem water_flow_restrictor (original_rate : ℝ) (reduced_rate : ℝ) : 
  original_rate = 5 →
  reduced_rate = 0.6 * original_rate - 1 →
  reduced_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_water_flow_restrictor_l2443_244366


namespace NUMINAMATH_CALUDE_consecutive_divisible_numbers_exist_l2443_244327

theorem consecutive_divisible_numbers_exist : ∃ (n : ℕ),
  (∀ (i : Fin 11), ∃ (k : ℕ), n + i.val = k * (2 * i.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_divisible_numbers_exist_l2443_244327


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l2443_244349

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflect_over_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The sum of coordinate values of two points -/
def sum_of_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let c : Point := ⟨x, 8⟩
  let d : Point := reflect_over_x_axis c
  sum_of_coordinates c d = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l2443_244349


namespace NUMINAMATH_CALUDE_simplify_fraction_l2443_244331

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2443_244331


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l2443_244304

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ+) : ℕ+ := sorry

/-- The theorem stating that the 11th number with digit sum 13 is 166 -/
theorem eleventh_number_with_digit_sum_13 : 
  nthNumberWithDigitSum13 11 = 166 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l2443_244304


namespace NUMINAMATH_CALUDE_monica_second_third_classes_l2443_244328

/-- Represents the number of students in Monica's classes -/
structure MonicasClasses where
  total_classes : Nat
  first_class : Nat
  fourth_class : Nat
  fifth_sixth_classes : Nat
  total_students : Nat

/-- The number of students in Monica's second and third classes combined -/
def students_in_second_third_classes (m : MonicasClasses) : Nat :=
  m.total_students - (m.first_class + m.fourth_class + m.fifth_sixth_classes)

/-- Theorem stating the number of students in Monica's second and third classes -/
theorem monica_second_third_classes :
  ∀ (m : MonicasClasses),
  m.total_classes = 6 →
  m.first_class = 20 →
  m.fourth_class = m.first_class / 2 →
  m.fifth_sixth_classes = 28 * 2 →
  m.total_students = 136 →
  students_in_second_third_classes m = 50 := by
  sorry

end NUMINAMATH_CALUDE_monica_second_third_classes_l2443_244328


namespace NUMINAMATH_CALUDE_fraction_difference_l2443_244301

theorem fraction_difference (p q : ℝ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (10 / 12 : ℝ) - (3 / 21 : ℝ) = 29 / 42 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_l2443_244301


namespace NUMINAMATH_CALUDE_julies_savings_l2443_244338

-- Define the initial savings amount
variable (S : ℝ)

-- Define the interest rate
variable (r : ℝ)

-- Define the time period
def t : ℝ := 2

-- Define the simple interest earned
def simple_interest : ℝ := 120

-- Define the compound interest earned
def compound_interest : ℝ := 126

-- Theorem statement
theorem julies_savings :
  (simple_interest = (S / 2) * r * t) ∧
  (compound_interest = (S / 2) * ((1 + r)^t - 1)) →
  S = 1200 := by
sorry

end NUMINAMATH_CALUDE_julies_savings_l2443_244338


namespace NUMINAMATH_CALUDE_equation_solution_l2443_244356

theorem equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ (x / (x - 1) - 1 = 1) :=
by
  use 2
  constructor
  · constructor
    · norm_num
    · field_simp
      ring
  · intro y hy
    have h1 : y ≠ 1 := hy.1
    have h2 : y / (y - 1) - 1 = 1 := hy.2
    -- Proof steps would go here
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2443_244356
