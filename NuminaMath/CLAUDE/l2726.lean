import Mathlib

namespace NUMINAMATH_CALUDE_air_quality_probability_l2726_272606

theorem air_quality_probability (p_one_day : ℝ) (p_two_days : ℝ) 
  (h1 : p_one_day = 0.8) 
  (h2 : p_two_days = 0.6) : 
  p_two_days / p_one_day = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l2726_272606


namespace NUMINAMATH_CALUDE_blue_string_length_l2726_272668

/-- Given the lengths of three strings (red, white, and blue) with specific relationships,
    prove that the blue string is 5 metres long. -/
theorem blue_string_length (red white blue : ℝ) : 
  red = 8 →
  white = 5 * red →
  white = 8 * blue →
  blue = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_string_length_l2726_272668


namespace NUMINAMATH_CALUDE_ellipse_parabola_properties_l2726_272605

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the line y = k(x - 4)
def line_k (x y k : ℝ) : Prop := y = k * (x - 4)

-- Define the line x = 1
def line_x1 (x : ℝ) : Prop := x = 1

theorem ellipse_parabola_properties 
  (a b c : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_ecc : c / a = Real.sqrt 3 / 2) 
  (h_focus : ∃ x y, ellipse_C x y a b ∧ parabola_E x y ∧ (x = a ∨ x = -a ∨ y = b ∨ y = -b)) :
  -- 1. Equation of C
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 = 1) ∧
  -- 2. Collinearity of A, P, and N
  (∀ k x_M y_M x_N y_N x_P, 
    k ≠ 0 →
    ellipse_C x_M y_M a b →
    ellipse_C x_N y_N a b →
    line_k x_M y_M k →
    line_k x_N y_N k →
    line_x1 x_P →
    ∃ y_P, line_k x_P y_P k →
    ∃ t, t * (x_P + 2) = 3 ∧ t * y_P = k * (x_N + 2)) ∧
  -- 3. Maximum area of triangle OMN
  (∃ S : ℝ, 
    (∀ k x_M y_M x_N y_N, 
      k ≠ 0 →
      ellipse_C x_M y_M a b →
      ellipse_C x_N y_N a b →
      line_k x_M y_M k →
      line_k x_N y_N k →
      (1/2) * abs (x_M * y_N - x_N * y_M) ≤ S) ∧
    (∃ k x_M y_M x_N y_N,
      k ≠ 0 →
      ellipse_C x_M y_M a b →
      ellipse_C x_N y_N a b →
      line_k x_M y_M k →
      line_k x_N y_N k →
      (1/2) * abs (x_M * y_N - x_N * y_M) = S) ∧
    S = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_properties_l2726_272605


namespace NUMINAMATH_CALUDE_other_number_proof_l2726_272698

theorem other_number_proof (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by sorry

end NUMINAMATH_CALUDE_other_number_proof_l2726_272698


namespace NUMINAMATH_CALUDE_wallpaper_overlap_l2726_272646

theorem wallpaper_overlap (total_area : ℝ) (large_wall_area : ℝ) (two_layer_area : ℝ) (three_layer_area : ℝ) (four_layer_area : ℝ) 
  (h1 : total_area = 500)
  (h2 : large_wall_area = 280)
  (h3 : two_layer_area = 54)
  (h4 : three_layer_area = 28)
  (h5 : four_layer_area = 14) :
  ∃ (six_layer_area : ℝ), 
    six_layer_area = 9 ∧ 
    total_area = (large_wall_area - two_layer_area - three_layer_area) + 
                 2 * two_layer_area + 
                 3 * three_layer_area + 
                 4 * four_layer_area + 
                 6 * six_layer_area :=
by sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_l2726_272646


namespace NUMINAMATH_CALUDE_quadrilateral_inequalities_l2726_272663

-- Define a structure for a convex quadrilateral
structure ConvexQuadrilateral :=
  (a b c d t : ℝ)
  (area_positive : t > 0)
  (sides_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (q : ConvexQuadrilateral) : Prop := sorry

-- State the theorem
theorem quadrilateral_inequalities (q : ConvexQuadrilateral) :
  (2 * q.t ≤ q.a * q.b + q.c * q.d) ∧
  (2 * q.t ≤ q.a * q.c + q.b * q.d) ∧
  ((2 * q.t = q.a * q.b + q.c * q.d) ∨ (2 * q.t = q.a * q.c + q.b * q.d) → is_cyclic q) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequalities_l2726_272663


namespace NUMINAMATH_CALUDE_amanda_hourly_rate_l2726_272612

/-- Amanda's cleaning service hourly rate calculation -/
theorem amanda_hourly_rate :
  let monday_hours : ℝ := 7.5
  let tuesday_hours : ℝ := 3
  let thursday_hours : ℝ := 4
  let saturday_hours : ℝ := 6
  let total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours
  let total_earnings : ℝ := 410
  total_earnings / total_hours = 20 := by
sorry

end NUMINAMATH_CALUDE_amanda_hourly_rate_l2726_272612


namespace NUMINAMATH_CALUDE_range_of_a_l2726_272682

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 - 2*x + a > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a - 1)^x < (a - 1)^y

-- Define the theorem
theorem range_of_a :
  (∃ a, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a, 1 < a ∧ a ≤ 2) ∧ (∀ a, (1 < a ∧ a ≤ 2) → (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2726_272682


namespace NUMINAMATH_CALUDE_train_passing_platform_l2726_272662

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (time_to_cross_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_to_cross_point = 120)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / time_to_cross_point) = 190 :=
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2726_272662


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l2726_272616

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  3.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem :
  let totalWheels := 18
  let frontAxleWheels := 2
  let otherAxleWheels := 4
  let axles := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  toll axles = 5.00 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_theorem_l2726_272616


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l2726_272601

/-- Calculates the total cost of a shopping trip, including discounts and sales tax -/
theorem shopping_cost_calculation 
  (tshirt_price sweater_price jacket_price : ℚ)
  (jacket_discount sales_tax : ℚ)
  (tshirt_quantity sweater_quantity jacket_quantity : ℕ)
  (h1 : tshirt_price = 8)
  (h2 : sweater_price = 18)
  (h3 : jacket_price = 80)
  (h4 : jacket_discount = 1/10)
  (h5 : sales_tax = 1/20)
  (h6 : tshirt_quantity = 6)
  (h7 : sweater_quantity = 4)
  (h8 : jacket_quantity = 5) :
  let tshirt_cost := tshirt_quantity * tshirt_price
  let sweater_cost := sweater_quantity * sweater_price
  let jacket_cost := jacket_quantity * jacket_price * (1 - jacket_discount)
  let subtotal := tshirt_cost + sweater_cost + jacket_cost
  let total := subtotal * (1 + sales_tax)
  total = 504 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_calculation_l2726_272601


namespace NUMINAMATH_CALUDE_negation_equivalence_l2726_272632

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2726_272632


namespace NUMINAMATH_CALUDE_toy_cost_price_l2726_272658

def toy_problem (num_toys : ℕ) (selling_price : ℚ) (gain_ratio : ℕ) :=
  (num_toys : ℚ) * (selling_price / num_toys) / (num_toys + gain_ratio : ℚ)

theorem toy_cost_price :
  toy_problem 25 62500 5 = 2083 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2726_272658


namespace NUMINAMATH_CALUDE_second_sum_calculation_l2726_272696

/-- Proves that given the conditions, the second sum is 1704 --/
theorem second_sum_calculation (total : ℝ) (first_part : ℝ) (second_part : ℝ) 
  (h1 : total = 2769)
  (h2 : total = first_part + second_part)
  (h3 : (first_part * 3 * 8 / 100) = (second_part * 5 * 3 / 100)) :
  second_part = 1704 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_calculation_l2726_272696


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2726_272661

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2726_272661


namespace NUMINAMATH_CALUDE_line_moved_down_l2726_272671

/-- The equation of a line obtained by moving y = 2x down 3 units -/
def moved_line (x y : ℝ) : Prop := y = 2 * x - 3

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := y = 2 * x

/-- Moving a line down by a certain number of units subtracts that number from the y-coordinate -/
axiom move_down (a b : ℝ) : ∀ x y, original_line x y → moved_line x (y - b) → b = 3

theorem line_moved_down : 
  ∀ x y, original_line x y → moved_line x (y - 3) :=
sorry

end NUMINAMATH_CALUDE_line_moved_down_l2726_272671


namespace NUMINAMATH_CALUDE_means_inequality_l2726_272648

theorem means_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_means_inequality_l2726_272648


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2726_272670

theorem arithmetic_calculation : 8 / 4 - 3^2 + 4 * 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2726_272670


namespace NUMINAMATH_CALUDE_good_student_count_l2726_272608

/-- Represents a student in the class -/
inductive Student
| Good
| Troublemaker

/-- The total number of students in the class -/
def totalStudents : Nat := 25

/-- The number of students making the first claim -/
def firstClaimCount : Nat := 5

/-- The number of students making the second claim -/
def secondClaimCount : Nat := 20

/-- Represents the statements made by students -/
structure Statements where
  firstClaim : Bool  -- True if the statement is true
  secondClaim : Bool -- True if the statement is true

/-- Checks if the first claim is consistent with the given number of good students -/
def checkFirstClaim (goodCount : Nat) : Bool :=
  totalStudents - goodCount > (totalStudents - 1) / 2

/-- Checks if the second claim is consistent with the given number of good students -/
def checkSecondClaim (goodCount : Nat) : Bool :=
  totalStudents - goodCount = 3 * (goodCount - 1)

/-- Checks if the given number of good students is consistent with all statements -/
def isConsistent (goodCount : Nat) (statements : Statements) : Bool :=
  (statements.firstClaim = checkFirstClaim goodCount) &&
  (statements.secondClaim = checkSecondClaim goodCount)

/-- Theorem: The number of good students is either 5 or 7 -/
theorem good_student_count :
  ∃ (statements : Statements),
    (isConsistent 5 statements ∨ isConsistent 7 statements) ∧
    ∀ (n : Nat), n ≠ 5 ∧ n ≠ 7 → ¬ isConsistent n statements :=
by sorry

end NUMINAMATH_CALUDE_good_student_count_l2726_272608


namespace NUMINAMATH_CALUDE_line_through_point_intersecting_circle_l2726_272602

/-- A line passing through a point and intersecting a circle -/
theorem line_through_point_intersecting_circle 
  (M : ℝ × ℝ) 
  (A B : ℝ × ℝ) 
  (h_M : M = (1, 0))
  (h_circle : ∀ P : ℝ × ℝ, P ∈ {P | P.1^2 + P.2^2 = 5} ↔ A ∈ {P | P.1^2 + P.2^2 = 5} ∧ B ∈ {P | P.1^2 + P.2^2 = 5})
  (h_first_quadrant : A.1 > 0 ∧ A.2 > 0)
  (h_vector_relation : B - M = 2 • (A - M))
  : ∃ (m c : ℝ), m = 1 ∧ c = -1 ∧ ∀ P : ℝ × ℝ, P ∈ {P | P.1 = m * P.2 + c} ↔ (A ∈ {P | P.1 = m * P.2 + c} ∧ B ∈ {P | P.1 = m * P.2 + c} ∧ M ∈ {P | P.1 = m * P.2 + c}) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_intersecting_circle_l2726_272602


namespace NUMINAMATH_CALUDE_f_of_2_equals_1_l2726_272603

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^3 - (x - 1) + 1

-- State the theorem
theorem f_of_2_equals_1 : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_1_l2726_272603


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2726_272685

theorem unique_solution_quadratic_system (y : ℚ) 
  (eq1 : 10 * y^2 + 9 * y - 2 = 0)
  (eq2 : 30 * y^2 + 77 * y - 14 = 0) : 
  y = 1/5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2726_272685


namespace NUMINAMATH_CALUDE_subtract_base6_l2726_272652

/-- Convert a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Theorem: Subtracting 35₆ from 131₆ in base 6 is equal to 52₆ --/
theorem subtract_base6 :
  let a := [1, 3, 1]  -- 131₆
  let b := [3, 5]     -- 35₆
  let result := [5, 2] -- 52₆
  base10ToBase6 (base6ToBase10 a - base6ToBase10 b) = result := by
  sorry

end NUMINAMATH_CALUDE_subtract_base6_l2726_272652


namespace NUMINAMATH_CALUDE_doctor_team_count_l2726_272679

/-- The number of ways to choose a team of doctors under specific conditions -/
def choose_doctor_team (total_doctors : ℕ) (pediatricians surgeons general_practitioners : ℕ) 
  (team_size : ℕ) : ℕ :=
  (pediatricians.choose 1) * (surgeons.choose 1) * (general_practitioners.choose 1) * 
  ((total_doctors - 3).choose (team_size - 3))

/-- Theorem stating the number of ways to choose a team of 5 doctors from 25 doctors, 
    with specific specialty requirements -/
theorem doctor_team_count : 
  choose_doctor_team 25 5 10 10 5 = 115500 := by
  sorry

end NUMINAMATH_CALUDE_doctor_team_count_l2726_272679


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2726_272690

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 
    12 ∣ (k * (k + 1) * (k + 2) * (k + 3)) ∧
    ∀ (m : ℕ), m > 12 → 
      ∃ (j : ℕ), j > 0 ∧ ¬(m ∣ (j * (j + 1) * (j + 2) * (j + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2726_272690


namespace NUMINAMATH_CALUDE_no_natural_function_satisfies_equation_l2726_272627

theorem no_natural_function_satisfies_equation :
  ¬ ∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_function_satisfies_equation_l2726_272627


namespace NUMINAMATH_CALUDE_polynomial_equality_l2726_272630

theorem polynomial_equality (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2726_272630


namespace NUMINAMATH_CALUDE_factorization_existence_l2726_272686

theorem factorization_existence : ∃ (a b c : ℤ), 
  (∀ x, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ∧ (a = 8 ∨ a = 12) := by
  sorry

end NUMINAMATH_CALUDE_factorization_existence_l2726_272686


namespace NUMINAMATH_CALUDE_sheet_area_difference_l2726_272607

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem sheet_area_difference : 
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 9
  let sheet2_length : ℝ := 4.5
  let sheet2_width : ℝ := 11
  let combined_area (l w : ℝ) := 2 * l * w
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 99 := by
  sorry


end NUMINAMATH_CALUDE_sheet_area_difference_l2726_272607


namespace NUMINAMATH_CALUDE_min_xy_value_l2726_272695

theorem min_xy_value (x y : ℝ) :
  (∃ (n : ℕ), n = 12) →
  1 + Real.cos (2 * x + 3 * y - 1) ^ 2 = (x^2 + y^2 + 2*(x+1)*(1-y)) / (x-y+1) →
  ∀ (z : ℝ), x * y ≥ 1/25 ∧ (∃ (a b : ℝ), a * b = 1/25) :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l2726_272695


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2726_272615

theorem polynomial_remainder (s : ℤ) : (s^11 + 1) % (s + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2726_272615


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l2726_272669

/-- The number of handshakes in a gathering of couples where each person shakes hands
    with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, the total number of handshakes is 60 -/
theorem six_couples_handshakes :
  handshakes 6 = 60 := by
  sorry


end NUMINAMATH_CALUDE_six_couples_handshakes_l2726_272669


namespace NUMINAMATH_CALUDE_range_of_expression_l2726_272654

theorem range_of_expression (x y : ℝ) (h : 4 * x^2 - 2 * Real.sqrt 3 * x * y + 4 * y^2 = 13) :
  10 - 4 * Real.sqrt 3 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 10 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l2726_272654


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l2726_272697

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l2726_272697


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2726_272620

theorem division_remainder_problem (D : ℕ) : 
  D = 12 * 63 + (D % 12) →  -- Incorrect division equation
  D = 21 * 36 + (D % 21) →  -- Correct division equation
  D % 21 = 0 :=             -- Remainder of correct division is 0
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2726_272620


namespace NUMINAMATH_CALUDE_sin_equation_solution_l2726_272645

theorem sin_equation_solution (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  Real.sin (n * π / 180) = Real.sin (680 * π / 180) →
  n = 40 ∨ n = 140 := by
sorry

end NUMINAMATH_CALUDE_sin_equation_solution_l2726_272645


namespace NUMINAMATH_CALUDE_train_crossing_time_l2726_272651

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 900 →
  train_speed_kmh = 108 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2726_272651


namespace NUMINAMATH_CALUDE_age_problem_l2726_272689

theorem age_problem (oleg serezha misha : ℕ) : 
  serezha = oleg + 1 →
  misha = serezha + 1 →
  40 < oleg + serezha + misha →
  oleg + serezha + misha < 45 →
  oleg = 13 ∧ serezha = 14 ∧ misha = 15 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2726_272689


namespace NUMINAMATH_CALUDE_baseball_cards_distribution_l2726_272621

theorem baseball_cards_distribution (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) :
  total_cards = 24 →
  num_friends = 4 →
  total_cards = num_friends * cards_per_friend →
  cards_per_friend = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_distribution_l2726_272621


namespace NUMINAMATH_CALUDE_charcoal_drawings_l2726_272641

theorem charcoal_drawings (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ)
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7)
  (h4 : total = colored_pencil + blending_marker + (total - colored_pencil - blending_marker)) :
  total - colored_pencil - blending_marker = 4 := by
sorry

end NUMINAMATH_CALUDE_charcoal_drawings_l2726_272641


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2726_272666

/-- Given that for all k ∈ ℝ, the line y - kx - 1 = 0 always intersects 
    with the ellipse x²/4 + y²/m = 1, prove that the range of m is [1, 4) ∪ (4, +∞) -/
theorem ellipse_line_intersection_range (m : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y - k*x - 1 = 0 ∧ x^2/4 + y^2/m = 1) ↔ 
  (m ∈ Set.Icc 1 4 ∪ Set.Ioi 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2726_272666


namespace NUMINAMATH_CALUDE_sum_of_products_l2726_272656

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 75)
  (h2 : y^2 + y*z + z^2 = 4)
  (h3 : z^2 + x*z + x^2 = 79) :
  x*y + y*z + x*z = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2726_272656


namespace NUMINAMATH_CALUDE_expand_product_l2726_272699

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2726_272699


namespace NUMINAMATH_CALUDE_product_and_quotient_of_geometric_sequences_l2726_272628

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem product_and_quotient_of_geometric_sequences
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  is_geometric_sequence (λ n => a n * b n) ∧
  is_geometric_sequence (λ n => a n / b n) :=
sorry

end NUMINAMATH_CALUDE_product_and_quotient_of_geometric_sequences_l2726_272628


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_l2726_272672

/-- The number of odd numbers in the nth row of the triangular arrangement -/
def odd_numbers_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers in the first n rows -/
def sum_odd_numbers (n : ℕ) : ℕ :=
  (odd_numbers_in_row n + 1) * n / 2

/-- The nth positive odd number -/
def nth_odd_number (n : ℕ) : ℕ := 2 * n - 1

theorem fifth_number_21st_row :
  let total_before := sum_odd_numbers 20
  let position := total_before + 5
  nth_odd_number position = 809 := by sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_l2726_272672


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l2726_272626

theorem inverse_proportion_inequality (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  k < 0 →
  y₁ = k / (-3) →
  y₂ = k / (-2) →
  y₃ = k / 3 →
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l2726_272626


namespace NUMINAMATH_CALUDE_slower_time_to_top_l2726_272684

/-- The time taken by the slower of two people to reach the top of a building --/
def time_to_top (stories : ℕ) (run_time : ℕ) (elevator_time : ℕ) (stop_time : ℕ) : ℕ :=
  max
    (stories * run_time)
    (stories * elevator_time + (stories - 1) * stop_time)

/-- Theorem stating that the slower person takes 217 seconds to reach the top floor --/
theorem slower_time_to_top :
  time_to_top 20 10 8 3 = 217 := by
  sorry

end NUMINAMATH_CALUDE_slower_time_to_top_l2726_272684


namespace NUMINAMATH_CALUDE_job_completion_time_l2726_272659

/-- The time it takes for Annie to complete the job alone -/
def annie_time : ℝ := 9

/-- The time the person works before stopping -/
def person_partial_time : ℝ := 4

/-- The time it takes Annie to complete the remaining work after the person stops -/
def annie_completion_time : ℝ := 6

/-- The time it takes for the person to complete the job alone -/
def person_total_time : ℝ := 12

theorem job_completion_time :
  (person_partial_time / person_total_time) + (annie_completion_time / annie_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l2726_272659


namespace NUMINAMATH_CALUDE_mutually_exclusive_head_l2726_272614

-- Define the set of people
variable (People : Type)

-- Define the property of standing at the head of the line
variable (stands_at_head : People → Prop)

-- Define A and B as specific people
variable (A B : People)

-- Axiom: A and B are distinct people
axiom A_neq_B : A ≠ B

-- Axiom: Only one person can stand at the head of the line
axiom one_at_head : ∀ (x y : People), stands_at_head x ∧ stands_at_head y → x = y

-- Theorem: The events "A stands at the head of the line" and "B stands at the head of the line" are mutually exclusive
theorem mutually_exclusive_head : 
  ¬(stands_at_head A ∧ stands_at_head B) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_head_l2726_272614


namespace NUMINAMATH_CALUDE_one_fourth_of_7_2_l2726_272617

theorem one_fourth_of_7_2 : (7.2 : ℚ) / 4 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_7_2_l2726_272617


namespace NUMINAMATH_CALUDE_quadratic_root_in_arithmetic_sequence_l2726_272653

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ c ≥ b ≥ 0,
    if the quadratic ax^2 + cx + b has exactly one root, then this root is -2 + √3. -/
theorem quadratic_root_in_arithmetic_sequence (a b c : ℝ) 
    (seq : ∃ (d : ℝ), c = a - d ∧ b = a - 2*d) 
    (order : a ≥ c ∧ c ≥ b ∧ b ≥ 0) 
    (one_root : ∃! x : ℝ, a*x^2 + c*x + b = 0) :
  ∃ (x : ℝ), a*x^2 + c*x + b = 0 ∧ x = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_arithmetic_sequence_l2726_272653


namespace NUMINAMATH_CALUDE_l_shape_area_is_52_l2726_272623

/-- The area of an 'L' shaped figure formed from a rectangle with given dimensions,
    after subtracting a corner rectangle and an inner rectangle. -/
def l_shape_area (large_length large_width corner_length corner_width inner_length inner_width : ℕ) : ℕ :=
  large_length * large_width - (corner_length * corner_width + inner_length * inner_width)

/-- Theorem stating that the area of the specific 'L' shaped figure is 52 square units. -/
theorem l_shape_area_is_52 :
  l_shape_area 10 6 3 2 2 1 = 52 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_is_52_l2726_272623


namespace NUMINAMATH_CALUDE_horner_third_step_equals_12_l2726_272649

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_step (a : ℝ) (x : ℝ) (prev : ℝ) : ℝ := prev * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (horner_step · x) 0

theorem horner_third_step_equals_12 :
  let coeffs := [2, 0, -3, 2, 1, -3]
  let x := 2
  let v3 := (horner_method (coeffs.take 4) x)
  v3 = 12 := by sorry

end NUMINAMATH_CALUDE_horner_third_step_equals_12_l2726_272649


namespace NUMINAMATH_CALUDE_x_twelfth_power_l2726_272664

theorem x_twelfth_power (x : ℂ) (h : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l2726_272664


namespace NUMINAMATH_CALUDE_sandis_initial_amount_l2726_272637

/-- Proves that Sandi's initial amount was $300 given the conditions of the problem -/
theorem sandis_initial_amount (sandi_initial : ℝ) : 
  (3 * sandi_initial + 150 = 1050) → sandi_initial = 300 := by
  sorry

end NUMINAMATH_CALUDE_sandis_initial_amount_l2726_272637


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2726_272678

theorem complex_number_modulus : ∃ (z : ℂ), z = (2 * Complex.I) / (1 + Complex.I) ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2726_272678


namespace NUMINAMATH_CALUDE_polynomial_product_no_x4_x3_terms_l2726_272600

theorem polynomial_product_no_x4_x3_terms :
  let P (x : ℝ) := 2 * x^3 - 5 * x^2 + 7 * x - 8
  let Q (x : ℝ) := a * x^2 + b * x + 11
  (∀ x, (P x) * (Q x) = 8 * x^5 - 17 * x^2 - 3 * x - 88) →
  a = 4 ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_no_x4_x3_terms_l2726_272600


namespace NUMINAMATH_CALUDE_f_of_g_of_3_equals_29_l2726_272631

def f (x : ℝ) : ℝ := 3 * x - 4

def g (x : ℝ) : ℝ := 2 * x + 3

theorem f_of_g_of_3_equals_29 : f (2 + g 3) = 29 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_equals_29_l2726_272631


namespace NUMINAMATH_CALUDE_range_of_a_l2726_272667

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- State the theorem
theorem range_of_a (a : ℝ) : (¬(p a) ∧ q a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2726_272667


namespace NUMINAMATH_CALUDE_rotate_D_180_about_origin_l2726_272691

def rotate_180_about_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotate_D_180_about_origin :
  let D : ℝ × ℝ := (-6, 2)
  rotate_180_about_origin D = (6, -2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_D_180_about_origin_l2726_272691


namespace NUMINAMATH_CALUDE_cafe_location_l2726_272657

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Checks if a point divides a line segment in a given ratio -/
def divides_segment (p1 p2 p : Point) (m n : ℚ) : Prop :=
  p.x = (n * p1.x + m * p2.x) / (m + n) ∧
  p.y = (n * p1.y + m * p2.y) / (m + n)

theorem cafe_location :
  let mark := Point.mk 1 8
  let sandy := Point.mk (-5) 0
  let cafe := Point.mk (-3) (8/3)
  divides_segment mark sandy cafe 1 2 := by
  sorry

end NUMINAMATH_CALUDE_cafe_location_l2726_272657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2726_272692

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2726_272692


namespace NUMINAMATH_CALUDE_two_even_dice_probability_l2726_272681

/-- The probability of rolling an even number on a fair 8-sided die -/
def prob_even : ℚ := 1/2

/-- The number of ways to choose 2 dice out of 3 -/
def ways_to_choose : ℕ := 3

/-- The probability of exactly two dice showing even numbers when rolling three fair 8-sided dice -/
def prob_two_even : ℚ := ways_to_choose * (prob_even^2 * (1 - prob_even))

theorem two_even_dice_probability : prob_two_even = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_two_even_dice_probability_l2726_272681


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_eq_two_l2726_272622

-- Define vectors in R²
def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

-- Define vector addition and subtraction
def add_vectors (u v : Fin 2 → ℝ) : Fin 2 → ℝ := ![u 0 + v 0, u 1 + v 1]
def sub_vectors (u v : Fin 2 → ℝ) : Fin 2 → ℝ := ![u 0 - v 0, u 1 - v 1]

-- Define parallel vectors
def parallel (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 1 = u 1 * v 0

-- Theorem statement
theorem vector_parallel_implies_x_eq_two (x : ℝ) :
  parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_eq_two_l2726_272622


namespace NUMINAMATH_CALUDE_min_value_of_parallel_lines_l2726_272643

theorem min_value_of_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  (∀ x y : ℝ, 2 * a + 3 * b ≥ 25) ∧ (∃ x y : ℝ, 2 * a + 3 * b = 25) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_parallel_lines_l2726_272643


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l2726_272619

/-- The sum of the tens digit and the ones digit of (3+4)^11 is 7 -/
theorem sum_of_digits_of_seven_to_eleven : 
  let n : ℕ := (3 + 4)^11
  let ones_digit : ℕ := n % 10
  let tens_digit : ℕ := (n / 10) % 10
  ones_digit + tens_digit = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l2726_272619


namespace NUMINAMATH_CALUDE_f_property_f_expression_l2726_272687

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2 - 4

-- State the theorem
theorem f_property : ∀ x : ℝ, f (1 + x) = x^2 + 2*x - 1 := by
  sorry

-- Prove that f(x) = x^2 - 4
theorem f_expression : ∀ x : ℝ, f x = x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_property_f_expression_l2726_272687


namespace NUMINAMATH_CALUDE_kenya_peanuts_l2726_272644

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_extra : ℕ) : 
  jose_peanuts = 85 → kenya_extra = 48 → jose_peanuts + kenya_extra = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l2726_272644


namespace NUMINAMATH_CALUDE_smallest_x_for_digit_sum_50_l2726_272655

def sequence_sum (x : ℕ) : ℕ := 100 * x + 4950

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem smallest_x_for_digit_sum_50 :
  ∀ x : ℕ, x < 99950 → digit_sum (sequence_sum x) ≠ 50 ∧
  digit_sum (sequence_sum 99950) = 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_digit_sum_50_l2726_272655


namespace NUMINAMATH_CALUDE_side_length_eq_twice_radius_l2726_272613

/-- A square with a circle inscribed such that the circle is tangent to two adjacent sides
    and passes through one vertex of the square. -/
structure InscribedCircleSquare where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The side length of the square -/
  s : ℝ
  /-- The circle is tangent to two adjacent sides of the square -/
  tangent_to_sides : True
  /-- The circle passes through one vertex of the square -/
  passes_through_vertex : True

/-- The side length of a square with an inscribed circle tangent to two adjacent sides
    and passing through one vertex is equal to twice the radius of the circle. -/
theorem side_length_eq_twice_radius (square : InscribedCircleSquare) :
  square.s = 2 * square.r := by
  sorry

end NUMINAMATH_CALUDE_side_length_eq_twice_radius_l2726_272613


namespace NUMINAMATH_CALUDE_square_of_ten_n_plus_five_l2726_272677

theorem square_of_ten_n_plus_five (n : ℕ) : (10 * n + 5)^2 = 100 * n * (n + 1) + 25 := by
  sorry

#eval (10 * 199 + 5)^2  -- Should output 3980025

end NUMINAMATH_CALUDE_square_of_ten_n_plus_five_l2726_272677


namespace NUMINAMATH_CALUDE_original_mixture_volume_l2726_272675

theorem original_mixture_volume 
  (original_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (new_alcohol_percentage : ℝ)
  (h1 : original_alcohol_percentage = 0.25)
  (h2 : added_water = 3)
  (h3 : new_alcohol_percentage = 0.20833333333333336)
  : ∃ (original_volume : ℝ),
    original_volume * original_alcohol_percentage / (original_volume + added_water) = new_alcohol_percentage ∧
    original_volume = 15 :=
by sorry

end NUMINAMATH_CALUDE_original_mixture_volume_l2726_272675


namespace NUMINAMATH_CALUDE_lucky_larry_challenge_l2726_272639

theorem lucky_larry_challenge (a b c d e f : ℤ) :
  a = 2 ∧ b = 4 ∧ c = 6 ∧ d = 8 ∧ f = 5 →
  (a + b - c + d - e + f = a + (b - (c + (d - (e + f))))) ↔ e = 8 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_challenge_l2726_272639


namespace NUMINAMATH_CALUDE_seating_arrangements_l2726_272660

/-- The number of ways to arrange n people in k seats -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose n items from k items -/
def choose (n k : ℕ) : ℕ := sorry

theorem seating_arrangements : 
  let total_seats : ℕ := 6
  let people : ℕ := 3
  let all_arrangements := arrange people total_seats
  let no_adjacent_empty := choose (total_seats - people + 1) people * arrange people people
  let all_empty_adjacent := choose (total_seats - people + 1) 1 * arrange people people
  all_arrangements - no_adjacent_empty - all_empty_adjacent = 72 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2726_272660


namespace NUMINAMATH_CALUDE_quadratic_sum_l2726_272635

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The theorem stating that for a quadratic function with given properties, a + b - c = -7 -/
theorem quadratic_sum (a b c : ℤ) :
  let f := QuadraticFunction a b c
  (f 2 = 5) →  -- The graph passes through (2, 5)
  (∀ x, f x ≥ f 1) →  -- The vertex is at x = 1
  (f 1 = 3) →  -- The y-coordinate of the vertex is 3
  a + b - c = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2726_272635


namespace NUMINAMATH_CALUDE_collinear_relation_vector_relation_l2726_272676

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ → ℝ × ℝ := λ a b => (a, b)

-- Define vector from A to B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector from A to C
def AC (a b : ℝ) : ℝ × ℝ := (a - A.1, b - A.2)

-- Define collinearity condition
def collinear (a b : ℝ) : Prop :=
  ∃ (t : ℝ), AC a b = (t * AB.1, t * AB.2)

-- Theorem 1: If A, B, and C are collinear, then a = 2-b
theorem collinear_relation (a b : ℝ) :
  collinear a b → a = 2 - b := by sorry

-- Theorem 2: If AC = 2AB, then C = (5, -3)
theorem vector_relation :
  ∃ (a b : ℝ), AC a b = (2 * AB.1, 2 * AB.2) ∧ C a b = (5, -3) := by sorry

end NUMINAMATH_CALUDE_collinear_relation_vector_relation_l2726_272676


namespace NUMINAMATH_CALUDE_expression_simplification_l2726_272665

theorem expression_simplification : 
  ((3 + 4 + 5 + 6 + 7) / 3) + ((3 * 6 + 12) / 4) = 95 / 6 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2726_272665


namespace NUMINAMATH_CALUDE_unique_twin_prime_sum_prime_power_l2726_272650

-- Define twin primes
def is_twin_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (p + 2)

-- Define prime power
def is_prime_power (n : ℕ) : Prop :=
  ∃ (q k : ℕ), Nat.Prime q ∧ k > 0 ∧ n = q^k

-- Theorem statement
theorem unique_twin_prime_sum_prime_power :
  ∃! (p : ℕ), is_twin_prime p ∧ is_prime_power (p + (p + 2)) :=
sorry

end NUMINAMATH_CALUDE_unique_twin_prime_sum_prime_power_l2726_272650


namespace NUMINAMATH_CALUDE_clock_strikes_in_day_l2726_272624

def clock_strikes (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

def total_strikes : Nat :=
  (List.range 24).map clock_strikes |> List.sum

theorem clock_strikes_in_day : total_strikes = 156 := by
  sorry

end NUMINAMATH_CALUDE_clock_strikes_in_day_l2726_272624


namespace NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l2726_272636

theorem tan_value_from_sin_plus_cos (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l2726_272636


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2726_272673

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ (∃ x : ℝ, x^2 ≤ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2726_272673


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l2726_272640

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_sales (profit_per_shirt : ℕ) (total_profit : ℕ) 
    (h1 : profit_per_shirt = 9)
    (h2 : total_profit = 2205) :
  total_profit / profit_per_shirt = 245 := by
  sorry

#check razorback_tshirt_sales

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l2726_272640


namespace NUMINAMATH_CALUDE_michael_truck_meetings_l2726_272674

/-- Represents the problem of Michael and the garbage truck --/
structure GarbageTruckProblem where
  michael_speed : ℝ
  michael_delay : ℝ
  pail_spacing : ℝ
  truck_speed : ℝ
  truck_stop_duration : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def number_of_meetings (problem : GarbageTruckProblem) : ℕ :=
  sorry

/-- The specific problem instance --/
def our_problem : GarbageTruckProblem :=
  { michael_speed := 3
  , michael_delay := 20
  , pail_spacing := 300
  , truck_speed := 12
  , truck_stop_duration := 45
  , initial_distance := 300 }

/-- Theorem stating that Michael and the truck meet exactly 6 times --/
theorem michael_truck_meetings :
  number_of_meetings our_problem = 6 := by
  sorry

end NUMINAMATH_CALUDE_michael_truck_meetings_l2726_272674


namespace NUMINAMATH_CALUDE_same_color_prob_eq_half_l2726_272683

/-- The probability of drawing two balls of the same color from an urn -/
def same_color_prob (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

/-- Theorem: The probability of drawing two balls of the same color is 1/2 iff n = 1 or n = 9 -/
theorem same_color_prob_eq_half (n : ℕ) :
  same_color_prob n = 1/2 ↔ n = 1 ∨ n = 9 := by
  sorry

#eval same_color_prob 1  -- Should output 1/2
#eval same_color_prob 9  -- Should output 1/2

end NUMINAMATH_CALUDE_same_color_prob_eq_half_l2726_272683


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2726_272611

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 2 ∧ b > 2 ∧
    n = 2 * a + 1 ∧
    n = 1 * b + 2 ∧
    (∀ (m : ℕ) (c d : ℕ), 
      c > 2 ∧ d > 2 ∧
      m = 2 * c + 1 ∧
      m = 1 * d + 2 →
      n ≤ m) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2726_272611


namespace NUMINAMATH_CALUDE_equal_pair_proof_l2726_272688

theorem equal_pair_proof : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_equal_pair_proof_l2726_272688


namespace NUMINAMATH_CALUDE_train_length_l2726_272694

/-- The length of a train given relative speeds and passing time -/
theorem train_length (v1 v2 t : ℝ) (h1 : v1 = 36) (h2 : v2 = 45) (h3 : t = 4) :
  (v1 + v2) * (5 / 18) * t = 90 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2726_272694


namespace NUMINAMATH_CALUDE_inverse_function_property_l2726_272642

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property of f and f_inv being inverse functions
def are_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the theorem
theorem inverse_function_property
  (h1 : are_inverse f f_inv)
  (h2 : f 2 = -1) :
  f_inv (-1) = 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2726_272642


namespace NUMINAMATH_CALUDE_prob_three_odd_less_than_one_eighth_l2726_272633

def n : ℕ := 2016

def odd_count : ℕ := n / 2

def prob_three_odd : ℚ :=
  (odd_count : ℚ) / n *
  ((odd_count - 1) : ℚ) / (n - 1) *
  ((odd_count - 2) : ℚ) / (n - 2)

theorem prob_three_odd_less_than_one_eighth :
  prob_three_odd < 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_less_than_one_eighth_l2726_272633


namespace NUMINAMATH_CALUDE_average_age_of_group_l2726_272634

/-- The average age of a group of seventh-graders and their guardians -/
def average_age (num_students : ℕ) (student_avg_age : ℚ) (num_guardians : ℕ) (guardian_avg_age : ℚ) : ℚ :=
  ((num_students : ℚ) * student_avg_age + (num_guardians : ℚ) * guardian_avg_age) / ((num_students + num_guardians) : ℚ)

/-- Theorem stating that the average age of 40 seventh-graders (average age 13) and 60 guardians (average age 40) is 29.2 -/
theorem average_age_of_group : average_age 40 13 60 40 = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_group_l2726_272634


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2726_272647

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 70 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 7 + a 8 + a 9 + a 14 = 70

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_condition a) : 
  a 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2726_272647


namespace NUMINAMATH_CALUDE_prime_digits_imply_prime_count_l2726_272629

theorem prime_digits_imply_prime_count (n : ℕ) (x : ℕ) : 
  (x = (10^n - 1) / 9) →  -- x is an integer with n digits, all equal to 1
  Nat.Prime x →           -- x is prime
  Nat.Prime n :=          -- n is prime
by sorry

end NUMINAMATH_CALUDE_prime_digits_imply_prime_count_l2726_272629


namespace NUMINAMATH_CALUDE_birthday_crayons_l2726_272638

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 291

/-- The number of crayons Paul had lost or given away -/
def crayons_lost_or_given : ℕ := 315

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost_or_given

theorem birthday_crayons : total_crayons = 606 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l2726_272638


namespace NUMINAMATH_CALUDE_sphere_volume_after_drilling_l2726_272625

/-- The remaining volume of a sphere after drilling two cylindrical holes -/
theorem sphere_volume_after_drilling (sphere_diameter : ℝ) (hole1_depth hole1_diameter hole2_depth hole2_diameter : ℝ) : 
  sphere_diameter = 12 ∧ 
  hole1_depth = 5 ∧ 
  hole1_diameter = 1 ∧ 
  hole2_depth = 5 ∧ 
  hole2_diameter = 1.5 → 
  (4 / 3 * π * (sphere_diameter / 2)^3) - (π * (hole1_diameter / 2)^2 * hole1_depth) - (π * (hole2_diameter / 2)^2 * hole2_depth) = 283.9375 * π := by
  sorry

#check sphere_volume_after_drilling

end NUMINAMATH_CALUDE_sphere_volume_after_drilling_l2726_272625


namespace NUMINAMATH_CALUDE_vendor_profit_l2726_272618

/-- Vendor's profit calculation --/
theorem vendor_profit : 
  let apple_buy_price : ℚ := 3 / 2
  let apple_sell_price : ℚ := 2
  let orange_buy_price : ℚ := 2.7 / 3
  let orange_sell_price : ℚ := 1
  let apple_discount_rate : ℚ := 1 / 10
  let orange_discount_rate : ℚ := 3 / 20
  let num_apples : ℕ := 5
  let num_oranges : ℕ := 5

  let discounted_apple_price := apple_sell_price * (1 - apple_discount_rate)
  let discounted_orange_price := orange_sell_price * (1 - orange_discount_rate)

  let total_cost := num_apples * apple_buy_price + num_oranges * orange_buy_price
  let total_revenue := num_apples * discounted_apple_price + num_oranges * discounted_orange_price

  total_revenue - total_cost = 1.25 := by sorry

end NUMINAMATH_CALUDE_vendor_profit_l2726_272618


namespace NUMINAMATH_CALUDE_system_solution_l2726_272693

theorem system_solution (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : z^2 = x^2 + y^2) 
  (h3 : (z + c)^2 = (x + a)^2 + (y + b)^2) : 
  y = (b/a) * x ∧ z = (c/a) * x :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2726_272693


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2726_272609

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ¬∃ n : ℕ, is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2726_272609


namespace NUMINAMATH_CALUDE_largest_number_in_set_l2726_272610

def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

theorem largest_number_in_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : (a + b + c) / 3 = 6)
  (h_median : b = 6)
  (h_smallest : a = 2) : 
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l2726_272610


namespace NUMINAMATH_CALUDE_prob_even_first_odd_second_l2726_272680

/-- The number of sides on a standard die -/
def sides : ℕ := 6

/-- The number of even outcomes on a standard die -/
def evenOutcomes : ℕ := 3

/-- The number of odd outcomes on a standard die -/
def oddOutcomes : ℕ := 3

/-- The probability of rolling an even number on one die -/
def probEven : ℚ := evenOutcomes / sides

/-- The probability of rolling an odd number on one die -/
def probOdd : ℚ := oddOutcomes / sides

theorem prob_even_first_odd_second : probEven * probOdd = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_first_odd_second_l2726_272680


namespace NUMINAMATH_CALUDE_helen_cookies_l2726_272604

/-- The number of cookies Helen baked in total -/
def total_cookies : ℕ := 574

/-- The number of cookies Helen baked this morning -/
def morning_cookies : ℕ := 139

/-- The number of cookies Helen baked yesterday -/
def yesterday_cookies : ℕ := total_cookies - morning_cookies

theorem helen_cookies : yesterday_cookies = 435 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l2726_272604
