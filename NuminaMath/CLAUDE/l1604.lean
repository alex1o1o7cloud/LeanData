import Mathlib

namespace NUMINAMATH_CALUDE_max_angle_B_in_arithmetic_sequence_triangle_l1604_160461

theorem max_angle_B_in_arithmetic_sequence_triangle :
  ∀ (a b c : ℝ) (A B C : ℝ),
    0 < a ∧ 0 < b ∧ 0 < c →
    0 < A ∧ 0 < B ∧ 0 < C →
    A + B + C = π →
    b^2 = a * c →  -- arithmetic sequence condition
    B ≤ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_angle_B_in_arithmetic_sequence_triangle_l1604_160461


namespace NUMINAMATH_CALUDE_rational_smallest_abs_value_and_monomial_degree_l1604_160422

-- Define the concept of absolute value for rational numbers
def abs_rat (q : ℚ) : ℚ := max q (-q)

-- Define the degree of a monomial
def monomial_degree (a b c : ℕ) : ℕ := a + b + c

theorem rational_smallest_abs_value_and_monomial_degree :
  (∀ q : ℚ, abs_rat q ≥ 0) ∧
  (∀ q : ℚ, abs_rat q = 0 ↔ q = 0) ∧
  (monomial_degree 2 1 0 = 3) :=
sorry

end NUMINAMATH_CALUDE_rational_smallest_abs_value_and_monomial_degree_l1604_160422


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_one_l1604_160441

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (x - 4) / (x - 3) - m - 4 = m / (3 - x)

-- Define the theorem
theorem equation_positive_root_implies_m_equals_one :
  (∃ x : ℝ, x > 0 ∧ equation x m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_one_l1604_160441


namespace NUMINAMATH_CALUDE_tangent_slope_sin_pi_over_four_l1604_160465

theorem tangent_slope_sin_pi_over_four :
  let f : ℝ → ℝ := fun x ↦ Real.sin x
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_sin_pi_over_four_l1604_160465


namespace NUMINAMATH_CALUDE_equation_solution_l1604_160453

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
  x₁ = (40 + Real.sqrt 1636) / 2 ∧ 
  x₂ = (-20 + Real.sqrt 388) / 2 ∧
  ∀ (x : ℝ), x > 0 → 
  ((3 / 5) * (2 * x^2 - 2) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4)) ↔ 
  (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1604_160453


namespace NUMINAMATH_CALUDE_average_marks_l1604_160464

theorem average_marks (n : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) (h1 : n = 6) (h2 : avg_five = 74) (h3 : sixth_mark = 62) :
  ((avg_five * (n - 1) + sixth_mark) / n : ℝ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l1604_160464


namespace NUMINAMATH_CALUDE_anthony_pizza_fraction_l1604_160439

theorem anthony_pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) : 
  total_slices = 16 → 
  whole_slice = 1 / total_slices → 
  shared_slice = 1 / (2 * total_slices) → 
  whole_slice + 2 * shared_slice = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pizza_fraction_l1604_160439


namespace NUMINAMATH_CALUDE_probability_two_defective_approx_l1604_160485

/-- The probability of selecting two defective smartphones from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1)

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
  abs (probability_two_defective 220 84 - 1447/10000) < ε :=
sorry

end NUMINAMATH_CALUDE_probability_two_defective_approx_l1604_160485


namespace NUMINAMATH_CALUDE_terminating_decimal_count_l1604_160491

theorem terminating_decimal_count : 
  (Finset.filter (fun n : ℕ => n % 13 = 0) (Finset.range 543)).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_count_l1604_160491


namespace NUMINAMATH_CALUDE_cos_squared_difference_equals_sqrt3_over_2_l1604_160481

theorem cos_squared_difference_equals_sqrt3_over_2 :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_difference_equals_sqrt3_over_2_l1604_160481


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1604_160489

/-- Given a rectangle with one side of length 18 and the sum of its area and perimeter
    equal to 2016, prove that its perimeter is 234. -/
theorem rectangle_perimeter (a : ℝ) : 
  a > 0 → 
  18 * a + 2 * (18 + a) = 2016 → 
  2 * (18 + a) = 234 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1604_160489


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1604_160427

theorem quadratic_maximum : 
  (∀ s : ℝ, -3 * s^2 + 24 * s + 15 ≤ 63) ∧ 
  (∃ s : ℝ, -3 * s^2 + 24 * s + 15 = 63) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1604_160427


namespace NUMINAMATH_CALUDE_roses_to_grandmother_l1604_160480

/-- Given that Ian had a certain number of roses and distributed them in a specific way,
    this theorem proves how many roses he gave to his grandmother. -/
theorem roses_to_grandmother (total : ℕ) (to_mother : ℕ) (to_sister : ℕ) (kept : ℕ) 
    (h1 : total = 20)
    (h2 : to_mother = 6)
    (h3 : to_sister = 4)
    (h4 : kept = 1) :
    total - (to_mother + to_sister + kept) = 9 := by
  sorry

end NUMINAMATH_CALUDE_roses_to_grandmother_l1604_160480


namespace NUMINAMATH_CALUDE_x_squared_in_set_l1604_160451

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({0, 1, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l1604_160451


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1604_160473

/-- The eccentricity of an ellipse with equation x²/4 + y²/9 = 1 is √5/3 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1604_160473


namespace NUMINAMATH_CALUDE_frog_vertical_side_probability_l1604_160488

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square area -/
def Square : Set Point := {p | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Represents a vertical side of the square -/
def VerticalSide : Set Point := {p | p.x = 0 ∨ p.x = 5}

/-- Represents a single jump of the frog -/
def Jump (p : Point) : Set Point :=
  {q | (q.x = p.x ∧ (q.y = p.y + 1 ∨ q.y = p.y - 1)) ∨
       (q.y = p.y ∧ (q.x = p.x + 1 ∨ q.x = p.x - 1))}

/-- The probability of ending on a vertical side given the starting point -/
noncomputable def ProbVerticalSide (p : Point) : ℝ := sorry

/-- The theorem stating the probability of ending on a vertical side is 1/2 -/
theorem frog_vertical_side_probability :
  ProbVerticalSide ⟨2, 2⟩ = 1/2 := by sorry

end NUMINAMATH_CALUDE_frog_vertical_side_probability_l1604_160488


namespace NUMINAMATH_CALUDE_company_x_employees_l1604_160471

theorem company_x_employees (full_time : ℕ) (worked_year : ℕ) (neither : ℕ) (both : ℕ) :
  full_time = 80 →
  worked_year = 100 →
  neither = 20 →
  both = 30 →
  full_time + worked_year - both + neither = 170 := by
  sorry

end NUMINAMATH_CALUDE_company_x_employees_l1604_160471


namespace NUMINAMATH_CALUDE_power_of_seven_mod_eight_l1604_160431

theorem power_of_seven_mod_eight : 7^123 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_eight_l1604_160431


namespace NUMINAMATH_CALUDE_library_books_count_l1604_160498

theorem library_books_count (old_books : ℕ) 
  (h1 : old_books + 300 + 400 - 200 = 1000) : old_books = 500 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1604_160498


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l1604_160400

theorem difference_of_squares_example : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l1604_160400


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_a_negative_l1604_160470

/-- The function f(x) = ax³ + 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

/-- Theorem: If f(x) has a unique zero point x₀ > 0, then a ∈ (-∞, 0) -/
theorem unique_positive_zero_implies_a_negative
  (a : ℝ)
  (h_unique : ∃! x₀ : ℝ, f a x₀ = 0)
  (h_positive : ∃ x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) :
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_a_negative_l1604_160470


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_for_f_always_greater_than_1_l1604_160458

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + 4| + |x - a|

-- Theorem for part I
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -8/3 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem for part II
theorem range_of_a_for_f_always_greater_than_1 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 1) ↔ (a < -3 ∨ a > -1) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_for_f_always_greater_than_1_l1604_160458


namespace NUMINAMATH_CALUDE_find_y_value_l1604_160413

theorem find_y_value (x y z : ℚ) : 
  x + y + z = 150 → x + 8 = y - 8 → x + 8 = 4 * z → y = 224 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1604_160413


namespace NUMINAMATH_CALUDE_xiaofang_english_score_l1604_160467

/-- Represents the scores of four subjects -/
structure Scores where
  chinese : ℝ
  math : ℝ
  english : ℝ
  science : ℝ

/-- The average score of four subjects is 88 -/
def avg_four (s : Scores) : Prop :=
  (s.chinese + s.math + s.english + s.science) / 4 = 88

/-- The average score of the first two subjects is 93 -/
def avg_first_two (s : Scores) : Prop :=
  (s.chinese + s.math) / 2 = 93

/-- The average score of the last three subjects is 87 -/
def avg_last_three (s : Scores) : Prop :=
  (s.math + s.english + s.science) / 3 = 87

/-- Xiaofang's English test score is 95 -/
theorem xiaofang_english_score (s : Scores) 
  (h1 : avg_four s) (h2 : avg_first_two s) (h3 : avg_last_three s) : 
  s.english = 95 := by
  sorry

end NUMINAMATH_CALUDE_xiaofang_english_score_l1604_160467


namespace NUMINAMATH_CALUDE_foci_of_given_hyperbola_l1604_160404

/-- A hyperbola is defined by its equation and foci coordinates -/
structure Hyperbola where
  a_squared : ℝ
  b_squared : ℝ
  equation : (x y : ℝ) → Prop := λ x y => x^2 / a_squared - y^2 / b_squared = 1

/-- The foci of a hyperbola are the two fixed points used in its geometric definition -/
def foci (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = h.a_squared + h.b_squared ∧ p.2 = 0}

/-- The given hyperbola from the problem -/
def given_hyperbola : Hyperbola :=
  { a_squared := 7
    b_squared := 3 }

/-- The theorem states that the foci of the given hyperbola are (√10, 0) and (-√10, 0) -/
theorem foci_of_given_hyperbola :
  foci given_hyperbola = {(Real.sqrt 10, 0), (-Real.sqrt 10, 0)} := by
  sorry

end NUMINAMATH_CALUDE_foci_of_given_hyperbola_l1604_160404


namespace NUMINAMATH_CALUDE_intersection_M_N_l1604_160412

-- Define the sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- State the theorem
theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1604_160412


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1604_160448

theorem solution_set_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1604_160448


namespace NUMINAMATH_CALUDE_triangle_altitude_bound_l1604_160425

/-- For any triangle with perimeter 2, there exists at least one altitude that is less than or equal to 1/√3. -/
theorem triangle_altitude_bound (a b c : ℝ) (h_perimeter : a + b + c = 2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ h : ℝ, h ≤ 1 / Real.sqrt 3 ∧ (h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / a ∨
                                  h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / b ∨
                                  h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / c) :=
by sorry


end NUMINAMATH_CALUDE_triangle_altitude_bound_l1604_160425


namespace NUMINAMATH_CALUDE_three_four_five_triangle_l1604_160484

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three given lengths can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

/-- Theorem stating that the lengths 3, 4, and 5 can form a triangle. -/
theorem three_four_five_triangle :
  can_form_triangle 3 4 5 := by
  sorry


end NUMINAMATH_CALUDE_three_four_five_triangle_l1604_160484


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1604_160493

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.A * l2.A + l1.B * l2.B = 0

/-- The first line: 4x - (a+1)y + 9 = 0 -/
def line1 (a : ℝ) : Line :=
  { A := 4, B := -(a+1), C := 9 }

/-- The second line: (a^2-1)x - ay + 6 = 0 -/
def line2 (a : ℝ) : Line :=
  { A := a^2-1, B := -a, C := 6 }

/-- Statement: a = -1 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem perpendicular_condition :
  (∀ a : ℝ, a = -1 → are_perpendicular (line1 a) (line2 a)) ∧
  (∃ a : ℝ, a ≠ -1 ∧ are_perpendicular (line1 a) (line2 a)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1604_160493


namespace NUMINAMATH_CALUDE_particle_and_account_max_l1604_160486

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 150 * t - 15 * t^2 + 50

-- Define the account balance function
def accountBalance (t : ℝ) : ℝ := 1000 * (1 + 0.05 * t)

-- Theorem statement
theorem particle_and_account_max (t : ℝ) :
  (∀ s : ℝ, elevation s ≤ elevation t) →
  elevation t = 425 ∧ accountBalance (t / 12) = 1020.83 := by
  sorry


end NUMINAMATH_CALUDE_particle_and_account_max_l1604_160486


namespace NUMINAMATH_CALUDE_same_number_of_digits_l1604_160437

/-- 
Given a natural number n, if k is the number of digits in 1974^n,
then 1974^n + 2^n < 10^k.
This implies that 1974^n and 1974^n + 2^n have the same number of digits.
-/
theorem same_number_of_digits (n : ℕ) : 
  let k := (Nat.log 10 (1974^n) + 1)
  1974^n + 2^n < 10^k := by
  sorry

end NUMINAMATH_CALUDE_same_number_of_digits_l1604_160437


namespace NUMINAMATH_CALUDE_valid_license_plates_count_l1604_160477

/-- Represents a license plate with 4 characters -/
structure LicensePlate :=
  (first : Char) (second : Char) (third : Nat) (fourth : Nat)

/-- Checks if a character is a letter (A-Z) -/
def isLetter (c : Char) : Bool :=
  'A' ≤ c ∧ c ≤ 'Z'

/-- Checks if a number is a single digit (0-9) -/
def isDigit (n : Nat) : Bool :=
  n < 10

/-- Checks if a license plate satisfies all conditions -/
def isValidLicensePlate (plate : LicensePlate) : Prop :=
  isLetter plate.first ∧
  isLetter plate.second ∧
  isDigit plate.third ∧
  isDigit plate.fourth ∧
  plate.third = plate.fourth ∧
  (plate.first.toNat = plate.third ∨ plate.second.toNat = plate.third)

/-- The number of valid license plates -/
def numValidLicensePlates : Nat :=
  (26 * 26) * 10

theorem valid_license_plates_count :
  numValidLicensePlates = 6760 :=
by sorry

end NUMINAMATH_CALUDE_valid_license_plates_count_l1604_160477


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1604_160408

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 6*x + y^2 + 8*y - k = 0 ↔ (x + 3)^2 + (y + 4)^2 = 10^2) → 
  k = 75 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1604_160408


namespace NUMINAMATH_CALUDE_trajectory_equation_l1604_160449

/-- The trajectory of a point P(x,y) satisfying a specific condition with respect to fixed points M and N -/
theorem trajectory_equation (x y : ℝ) : 
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  let P : ℝ × ℝ := (x, y)
  let MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
  let MP : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)
  let NP : ℝ × ℝ := (P.1 - N.1, P.2 - N.2)
  ‖MN‖ * ‖MP‖ + MN.1 * NP.1 + MN.2 * NP.2 = 0 →
  y^2 = -8*x := by
sorry


end NUMINAMATH_CALUDE_trajectory_equation_l1604_160449


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1604_160496

theorem consecutive_integers_average (a c : ℤ) : 
  (∀ k ∈ Finset.range 7, k + a > 0) →  -- Positive integers condition
  c = (Finset.sum (Finset.range 7) (λ k => k + a)) / 7 →  -- Average condition
  (Finset.sum (Finset.range 7) (λ k => k + c)) / 7 = a + 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1604_160496


namespace NUMINAMATH_CALUDE_right_triangle_area_l1604_160443

/-- The area of a right triangle with hypotenuse 13 and one side 5 is 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1604_160443


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l1604_160497

def shelf_difference (initial_action_figures : ℕ) (initial_books : ℕ) (added_books : ℕ) : ℕ :=
  initial_action_figures - (initial_books + added_books)

theorem jerry_shelf_difference :
  shelf_difference 7 2 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l1604_160497


namespace NUMINAMATH_CALUDE_tom_missed_no_games_l1604_160424

/-- The number of hockey games Tom missed this year -/
def games_missed_this_year (games_this_year games_last_year total_games : ℕ) : ℕ :=
  total_games - (games_this_year + games_last_year)

/-- Theorem: Tom missed 0 hockey games this year -/
theorem tom_missed_no_games :
  games_missed_this_year 4 9 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tom_missed_no_games_l1604_160424


namespace NUMINAMATH_CALUDE_transformed_system_solution_l1604_160445

theorem transformed_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * 6 + b₁ * 3 = c₁)
  (h₂ : a₂ * 6 + b₂ * 3 = c₂) :
  (4 * a₁ * 22 + 3 * b₁ * 33 = 11 * c₁) ∧ 
  (4 * a₂ * 22 + 3 * b₂ * 33 = 11 * c₂) := by
sorry

end NUMINAMATH_CALUDE_transformed_system_solution_l1604_160445


namespace NUMINAMATH_CALUDE_solution_equality_l1604_160468

theorem solution_equality (x y : ℝ) : 
  |x + y - 2| + (2 * x - 3 * y + 5)^2 = 0 → 
  ((x = 1 ∧ y = 9) ∨ (x = 5 ∧ y = 5)) := by
sorry

end NUMINAMATH_CALUDE_solution_equality_l1604_160468


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1604_160476

/-- Given two vectors a and b in ℝ², where a is perpendicular to (a + b), prove that the y-coordinate of b is -3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.1 = -1) 
  (h'' : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) : ℝ) = 0) : 
  b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1604_160476


namespace NUMINAMATH_CALUDE_prize_money_calculation_l1604_160426

def total_amount : ℕ := 300
def paintings_sold : ℕ := 3
def price_per_painting : ℕ := 50

theorem prize_money_calculation :
  total_amount - (paintings_sold * price_per_painting) = 150 := by
  sorry

end NUMINAMATH_CALUDE_prize_money_calculation_l1604_160426


namespace NUMINAMATH_CALUDE_initial_conditions_squares_in_figure_100_l1604_160420

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 2 * n + 1

/-- The sequence satisfies the given initial conditions -/
theorem initial_conditions :
  f 0 = 1 ∧ f 1 = 6 ∧ f 2 = 17 ∧ f 3 = 34 := by sorry

/-- The number of squares in figure 100 is 30201 -/
theorem squares_in_figure_100 :
  f 100 = 30201 := by sorry

end NUMINAMATH_CALUDE_initial_conditions_squares_in_figure_100_l1604_160420


namespace NUMINAMATH_CALUDE_product_of_digits_less_than_number_l1604_160492

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def digit_product (n : ℕ) : ℕ :=
  (digits n).prod

theorem product_of_digits_less_than_number (N : ℕ) (h : N > 9) :
  digit_product N < N :=
sorry

end NUMINAMATH_CALUDE_product_of_digits_less_than_number_l1604_160492


namespace NUMINAMATH_CALUDE_log_weight_l1604_160415

theorem log_weight (log_length : ℕ) (weight_per_foot : ℕ) (cut_pieces : ℕ) : 
  log_length = 20 → 
  weight_per_foot = 150 → 
  cut_pieces = 2 → 
  (log_length / cut_pieces) * weight_per_foot = 1500 :=
by sorry

end NUMINAMATH_CALUDE_log_weight_l1604_160415


namespace NUMINAMATH_CALUDE_inverse_function_property_l1604_160406

theorem inverse_function_property (f : ℝ → ℝ) (hf : Function.Bijective f) 
  (h : ∀ x : ℝ, f (x + 1) + f (-x - 4) = 2) :
  ∀ x : ℝ, (Function.invFun f) (2011 - x) + (Function.invFun f) (x - 2009) = -3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1604_160406


namespace NUMINAMATH_CALUDE_solve_equation_l1604_160472

theorem solve_equation (x : ℝ) (n : ℝ) (h1 : 5 / (n + 1 / x) = 1) (h2 : x = 1) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1604_160472


namespace NUMINAMATH_CALUDE_max_product_fg_l1604_160440

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions on the ranges of f and g
axiom f_range : ∀ x, -3 ≤ f x ∧ f x ≤ 4
axiom g_range : ∀ x, -3 ≤ g x ∧ g x ≤ 2

-- Theorem stating the maximum value of f(x) · g(x)
theorem max_product_fg : 
  ∃ x : ℝ, ∀ y : ℝ, f y * g y ≤ f x * g x ∧ f x * g x = 12 :=
sorry

end NUMINAMATH_CALUDE_max_product_fg_l1604_160440


namespace NUMINAMATH_CALUDE_john_basketball_shots_l1604_160435

theorem john_basketball_shots 
  (initial_shots : ℕ) 
  (initial_percentage : ℚ) 
  (additional_shots : ℕ) 
  (new_percentage : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_percentage = 2/5)
  (h3 : additional_shots = 10)
  (h4 : new_percentage = 11/25) :
  (new_percentage * (initial_shots + additional_shots)).floor - 
  (initial_percentage * initial_shots).floor = 6 := by
sorry

end NUMINAMATH_CALUDE_john_basketball_shots_l1604_160435


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1604_160429

theorem complex_equation_solution (n : ℝ) : 
  (1 : ℂ) / (1 + Complex.I) = (1 : ℂ) / 2 - n * Complex.I → n = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1604_160429


namespace NUMINAMATH_CALUDE_coefficient_of_x_5_l1604_160469

-- Define the polynomials
def p (x : ℝ) : ℝ := x^6 - 4*x^5 + 6*x^4 - 5*x^3 + 3*x^2 - 2*x + 1
def q (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + x^2 + 4*x + 5

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem to prove
theorem coefficient_of_x_5 : 
  ∃ c, ∀ x, product x = c * x^5 + (fun x => x^6 * (-23) + (fun x => x^4 * 0 + x^3 * 0 + x^2 * 0 + x * 0 + 0) x) x :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_5_l1604_160469


namespace NUMINAMATH_CALUDE_speed_ratio_problem_l1604_160456

/-- The ratio of speeds between two people walking in opposite directions -/
theorem speed_ratio_problem (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  (v₁ / v₂ * 60 = v₂ / v₁ * 60 + 35) → v₁ / v₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_problem_l1604_160456


namespace NUMINAMATH_CALUDE_club_distribution_theorem_l1604_160428

-- Define the type for inhabitants
variable {Inhabitant : Type}

-- Define the type for clubs as sets of inhabitants
variable {Club : Type}

-- Define the property that every two clubs have a common member
def have_common_member (clubs : Set Club) (members : Club → Set Inhabitant) : Prop :=
  ∀ c1 c2 : Club, c1 ∈ clubs → c2 ∈ clubs → c1 ≠ c2 → ∃ i : Inhabitant, i ∈ members c1 ∩ members c2

-- Define the assignment of compasses and rulers
def valid_assignment (clubs : Set Club) (members : Club → Set Inhabitant) 
  (has_compass : Inhabitant → Prop) (has_ruler : Inhabitant → Prop) : Prop :=
  (∀ c : Club, c ∈ clubs → 
    (∃ i : Inhabitant, i ∈ members c ∧ has_compass i) ∧ 
    (∃ i : Inhabitant, i ∈ members c ∧ has_ruler i)) ∧
  (∃! i : Inhabitant, has_compass i ∧ has_ruler i)

-- The main theorem
theorem club_distribution_theorem 
  (clubs : Set Club) (members : Club → Set Inhabitant) 
  (h : have_common_member clubs members) :
  ∃ (has_compass : Inhabitant → Prop) (has_ruler : Inhabitant → Prop),
    valid_assignment clubs members has_compass has_ruler :=
sorry

end NUMINAMATH_CALUDE_club_distribution_theorem_l1604_160428


namespace NUMINAMATH_CALUDE_correct_calculation_l1604_160411

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1604_160411


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1604_160403

theorem sqrt_expression_equality : 
  (Real.sqrt 48 - Real.sqrt 27) / Real.sqrt 3 + Real.sqrt 6 * 2 * Real.sqrt (1/3) = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1604_160403


namespace NUMINAMATH_CALUDE_movement_increases_dimension_l1604_160459

/-- Dimension of geometric objects -/
inductive GeometricDimension
  | point
  | line
  | surface
  deriving Repr

/-- Function that returns the dimension of the object formed by moving an object of a given dimension -/
def dimensionAfterMovement (d : GeometricDimension) : GeometricDimension :=
  match d with
  | GeometricDimension.point => GeometricDimension.line
  | GeometricDimension.line => GeometricDimension.surface
  | GeometricDimension.surface => GeometricDimension.surface

/-- Theorem stating that moving a point forms a line and moving a line forms a surface -/
theorem movement_increases_dimension :
  (dimensionAfterMovement GeometricDimension.point = GeometricDimension.line) ∧
  (dimensionAfterMovement GeometricDimension.line = GeometricDimension.surface) :=
by sorry

end NUMINAMATH_CALUDE_movement_increases_dimension_l1604_160459


namespace NUMINAMATH_CALUDE_rational_identity_product_l1604_160423

theorem rational_identity_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (42 * x - 51) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -2981.25 := by sorry

end NUMINAMATH_CALUDE_rational_identity_product_l1604_160423


namespace NUMINAMATH_CALUDE_badminton_match_duration_l1604_160410

theorem badminton_match_duration :
  ∀ (hours minutes : ℕ),
    hours = 12 ∧ minutes = 25 →
    hours * 60 + minutes = 745 := by sorry

end NUMINAMATH_CALUDE_badminton_match_duration_l1604_160410


namespace NUMINAMATH_CALUDE_car_speed_problem_l1604_160457

theorem car_speed_problem (d : ℝ) (v : ℝ) (h1 : d > 0) (h2 : v > 0) :
  let t := d / v
  let return_time := 2 * t
  let total_distance := 2 * d
  let total_time := t + return_time
  (total_distance / total_time = 30) → v = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1604_160457


namespace NUMINAMATH_CALUDE_prob_one_day_both_first_class_conditional_prob_both_first_class_expected_daily_profit_l1604_160433

-- Define the probabilities and costs
def p_first_class : ℝ := 0.5
def p_second_class : ℝ := 0.4
def cost_per_unit : ℝ := 2000
def price_first_class : ℝ := 10000
def price_second_class : ℝ := 8000
def loss_substandard : ℝ := 1000

-- Define the probability of both products being first-class in one day
def p_both_first_class : ℝ := p_first_class * p_first_class

-- Define the probability of exactly one first-class product in a day
def p_one_first_class : ℝ := 2 * p_first_class * (1 - p_first_class)

-- Define the daily profit function
def daily_profit (n_first n_second n_substandard : ℕ) : ℝ :=
  n_first * (price_first_class - cost_per_unit) +
  n_second * (price_second_class - cost_per_unit) +
  n_substandard * (-cost_per_unit - loss_substandard)

-- Theorem 1: Probability of exactly one day with both first-class products in three days
theorem prob_one_day_both_first_class :
  (3 : ℝ) * p_both_first_class * (1 - p_both_first_class)^2 = 27/64 := by sorry

-- Theorem 2: Conditional probability of both products being first-class given one is first-class
theorem conditional_prob_both_first_class :
  p_both_first_class / (p_both_first_class + p_one_first_class) = 1/3 := by sorry

-- Theorem 3: Expected daily profit
theorem expected_daily_profit :
  p_both_first_class * daily_profit 2 0 0 +
  p_one_first_class * daily_profit 1 1 0 +
  (p_second_class * p_second_class) * daily_profit 0 2 0 +
  (2 * p_first_class * (1 - p_first_class - p_second_class)) * daily_profit 1 0 1 +
  (2 * p_second_class * (1 - p_first_class - p_second_class)) * daily_profit 0 1 1 +
  ((1 - p_first_class - p_second_class) * (1 - p_first_class - p_second_class)) * daily_profit 0 0 2 = 12200 := by sorry

end NUMINAMATH_CALUDE_prob_one_day_both_first_class_conditional_prob_both_first_class_expected_daily_profit_l1604_160433


namespace NUMINAMATH_CALUDE_sum_digits_base_seven_999_l1604_160483

/-- Represents a number in base 7 as a list of digits (least significant digit first) -/
def BaseSevenRepresentation := List Nat

/-- Converts a natural number to its base 7 representation -/
def toBaseSeven (n : Nat) : BaseSevenRepresentation :=
  sorry

/-- Computes the sum of digits in a base 7 representation -/
def sumDigitsBaseSeven (rep : BaseSevenRepresentation) : Nat :=
  sorry

theorem sum_digits_base_seven_999 :
  sumDigitsBaseSeven (toBaseSeven 999) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base_seven_999_l1604_160483


namespace NUMINAMATH_CALUDE_backpack_price_relationship_l1604_160421

/-- Represents the relationship between backpack purchases and prices -/
theorem backpack_price_relationship (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for division
  (h2 : 810 > 0) -- Total spent on type A is positive
  (h3 : 600 > 0) -- Total spent on type B is positive
  (h4 : x + 20 > 0) -- Ensure denominator is positive
  : 810 / (x + 20) = (600 / x) * (1 - 10 / 100) :=
by sorry

end NUMINAMATH_CALUDE_backpack_price_relationship_l1604_160421


namespace NUMINAMATH_CALUDE_average_of_sixty_results_l1604_160418

theorem average_of_sixty_results (A : ℝ) : 
  (60 * A + 40 * 60) / 100 = 48 → A = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_of_sixty_results_l1604_160418


namespace NUMINAMATH_CALUDE_team_a_win_probabilities_l1604_160414

/-- Probability of Team A winning a single game -/
def p_a : ℝ := 0.6

/-- Probability of Team B winning a single game -/
def p_b : ℝ := 0.4

/-- Sum of probabilities for a single game is 1 -/
axiom prob_sum : p_a + p_b = 1

/-- Probability of Team A winning in a best-of-three format -/
def p_a_bo3 : ℝ := p_a^2 + 2 * p_a^2 * p_b

/-- Probability of Team A winning in a best-of-five format -/
def p_a_bo5 : ℝ := p_a^3 + 3 * p_a^3 * p_b + 6 * p_a^3 * p_b^2

/-- Theorem: Probabilities of Team A winning in best-of-three and best-of-five formats -/
theorem team_a_win_probabilities : 
  p_a_bo3 = 0.648 ∧ p_a_bo5 = 0.68256 :=
sorry

end NUMINAMATH_CALUDE_team_a_win_probabilities_l1604_160414


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1604_160446

def f (x : ℝ) : ℝ := (x - 2)^2 + 1

theorem quadratic_inequality : f 2 < f 3 ∧ f 3 < f 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1604_160446


namespace NUMINAMATH_CALUDE_complex_expression_equals_eight_l1604_160490

theorem complex_expression_equals_eight :
  (1 / Real.sqrt 0.04) + ((1 / Real.sqrt 27) ^ (1/3)) + 
  ((Real.sqrt 2 + 1)⁻¹) - (2 ^ (1/2)) + ((-2) ^ 0) = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_eight_l1604_160490


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_m_range_l1604_160466

/-- An ellipse centered at the origin with right focus at (1,0) and one vertex at (0,√3) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- Two points are symmetric about the line y = x + m -/
def SymmetricAboutLine (p q : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (t : ℝ), p.1 + q.1 = 2 * t ∧ p.2 + q.2 = 2 * (t + m)

theorem ellipse_symmetric_points_m_range :
  ∀ (m : ℝ),
    (∃ (p q : ℝ × ℝ), p ∈ Ellipse ∧ q ∈ Ellipse ∧ p ≠ q ∧ SymmetricAboutLine p q m) →
    -Real.sqrt 7 / 7 < m ∧ m < Real.sqrt 7 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_m_range_l1604_160466


namespace NUMINAMATH_CALUDE_gcd_15225_20335_35475_l1604_160462

theorem gcd_15225_20335_35475 : Nat.gcd 15225 (Nat.gcd 20335 35475) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15225_20335_35475_l1604_160462


namespace NUMINAMATH_CALUDE_expression_simplification_l1604_160436

theorem expression_simplification :
  (((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4)) = 43 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1604_160436


namespace NUMINAMATH_CALUDE_ball_count_proof_l1604_160454

/-- Theorem: Given a bag of balls with specific color counts and probability,
    prove the total number of balls. -/
theorem ball_count_proof
  (white green yellow red purple : ℕ)
  (prob_not_red_or_purple : ℚ)
  (h1 : white = 50)
  (h2 : green = 20)
  (h3 : yellow = 10)
  (h4 : red = 17)
  (h5 : purple = 3)
  (h6 : prob_not_red_or_purple = 4/5) :
  white + green + yellow + red + purple = 100 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l1604_160454


namespace NUMINAMATH_CALUDE_pillow_average_price_l1604_160494

/-- Given 4 pillows with an average cost of $5 and an additional pillow costing $10,
    prove that the average price of all 5 pillows is $6 -/
theorem pillow_average_price (n : ℕ) (avg_cost : ℚ) (additional_cost : ℚ) :
  n = 4 ∧ avg_cost = 5 ∧ additional_cost = 10 →
  ((n : ℚ) * avg_cost + additional_cost) / ((n : ℚ) + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pillow_average_price_l1604_160494


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l1604_160409

theorem factorial_equation_solution : ∃ (n : ℕ), n > 0 ∧ (n + 1).factorial + (n + 3).factorial = n.factorial * 1190 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l1604_160409


namespace NUMINAMATH_CALUDE_dot_product_range_l1604_160450

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (angle_BAC : Real)
  (length_AB : Real)
  (length_AC : Real)

-- Define a point D on side BC
def PointOnBC (triangle : Triangle) := 
  {D : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • triangle.B + t • triangle.C}

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_range (triangle : Triangle) 
  (h1 : triangle.angle_BAC = 2*π/3)  -- 120° in radians
  (h2 : triangle.length_AB = 2)
  (h3 : triangle.length_AC = 1) :
  ∀ D ∈ PointOnBC triangle, 
    -5 ≤ dot_product (D - triangle.A) (triangle.C - triangle.B) ∧ 
    dot_product (D - triangle.A) (triangle.C - triangle.B) ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_dot_product_range_l1604_160450


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l1604_160442

/-- A polynomial of degree 5 -/
def P (a b c d e : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The statement that P has five distinct x-intercepts -/
def has_five_distinct_roots (a b c d e : ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ : ℝ), (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧
                           r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧
                           r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧
                           r₄ ≠ r₅) ∧
                          (∀ x : ℝ, P a b c d e x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅)

theorem coefficient_d_nonzero (a b c d e : ℝ) 
  (h1 : has_five_distinct_roots a b c d e)
  (h2 : P a b c d e 0 = 0) : -- One root is at (0,0)
  d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l1604_160442


namespace NUMINAMATH_CALUDE_savings_equality_l1604_160432

theorem savings_equality (your_initial : ℕ) (friend_initial : ℕ) (friend_weekly : ℕ) (weeks : ℕ) 
  (h1 : your_initial = 160)
  (h2 : friend_initial = 210)
  (h3 : friend_weekly = 5)
  (h4 : weeks = 25) :
  ∃ your_weekly : ℕ, 
    your_initial + weeks * your_weekly = friend_initial + weeks * friend_weekly ∧ 
    your_weekly = 7 := by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l1604_160432


namespace NUMINAMATH_CALUDE_screws_per_section_l1604_160434

def initial_screws : ℕ := 8
def buy_multiplier : ℕ := 2
def num_sections : ℕ := 4

theorem screws_per_section :
  (initial_screws + initial_screws * buy_multiplier) / num_sections = 6 := by
  sorry

end NUMINAMATH_CALUDE_screws_per_section_l1604_160434


namespace NUMINAMATH_CALUDE_lunch_combinations_l1604_160447

theorem lunch_combinations (first_course main_course dessert : ℕ) :
  first_course = 4 → main_course = 5 → dessert = 3 →
  first_course * main_course * dessert = 60 := by
sorry

end NUMINAMATH_CALUDE_lunch_combinations_l1604_160447


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1604_160482

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1604_160482


namespace NUMINAMATH_CALUDE_unique_intersection_l1604_160417

-- Define the three lines
def line1 (x y : ℝ) : Prop := 3 * x - 9 * y + 18 = 0
def line2 (x y : ℝ) : Prop := 6 * x - 18 * y - 36 = 0
def line3 (x : ℝ) : Prop := x - 3 = 0

-- Define what it means for a point to be on all three lines
def onAllLines (x y : ℝ) : Prop :=
  line1 x y ∧ line2 x y ∧ line3 x

-- Theorem statement
theorem unique_intersection :
  ∃! p : ℝ × ℝ, onAllLines p.1 p.2 ∧ p = (3, 3) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1604_160417


namespace NUMINAMATH_CALUDE_matrix_addition_theorem_l1604_160455

/-- Given matrices A and B, prove that C = 2A + B is equal to the expected result. -/
theorem matrix_addition_theorem (A B : Matrix (Fin 2) (Fin 2) ℤ) : 
  A = !![2, 1; 3, 4] → 
  B = !![0, -5; -1, 6] → 
  2 • A + B = !![4, -3; 5, 14] := by
  sorry

end NUMINAMATH_CALUDE_matrix_addition_theorem_l1604_160455


namespace NUMINAMATH_CALUDE_expression_evaluation_l1604_160401

theorem expression_evaluation : ∃ (n m k : ℕ),
  (n > 0 ∧ m > 0 ∧ k > 0) ∧
  (2 * n - 1 = 2025) ∧
  (2 * m = 2024) ∧
  (2^k = 1024) →
  (Finset.sum (Finset.range n) (λ i => 2 * i + 5)) -
  (Finset.sum (Finset.range m) (λ i => 2 * i + 4)) +
  2 * (Finset.sum (Finset.range k) (λ i => 2^i)) = 5104 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1604_160401


namespace NUMINAMATH_CALUDE_root_equation_sum_l1604_160405

theorem root_equation_sum (a : ℝ) (h : a^2 + a - 1 = 0) : 
  (1 - a) / a + a / (1 + a) = 1 := by sorry

end NUMINAMATH_CALUDE_root_equation_sum_l1604_160405


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1604_160438

theorem decimal_sum_to_fraction :
  (0.3 : ℚ) + 0.04 + 0.005 + 0.0006 + 0.00007 = 34567 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1604_160438


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1604_160402

theorem imaginary_part_of_complex_division : 
  Complex.im ((3 + 4 * Complex.I) / Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1604_160402


namespace NUMINAMATH_CALUDE_middle_manager_sample_size_l1604_160407

/-- Calculates the number of middle-level managers to be sampled in a stratified sampling scenario -/
theorem middle_manager_sample_size (total_employees : ℕ) (middle_managers : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : middle_managers = 150)
  (h3 : sample_size = 200) :
  (middle_managers : ℚ) / (total_employees : ℚ) * (sample_size : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_middle_manager_sample_size_l1604_160407


namespace NUMINAMATH_CALUDE_villager_A_motorcycle_fraction_l1604_160452

/-- Represents the scenario of two villagers and a motorcycle traveling to a station -/
structure TravelScenario where
  totalDistance : ℝ := 1
  walkingSpeed : ℝ
  motorcycleSpeed : ℝ
  simultaneousArrival : Prop

/-- The main theorem stating the fraction of journey villager A travels by motorcycle -/
theorem villager_A_motorcycle_fraction (scenario : TravelScenario) 
  (h1 : scenario.motorcycleSpeed = 9 * scenario.walkingSpeed)
  (h2 : scenario.simultaneousArrival) : 
  ∃ (x : ℝ), x = 5/6 ∧ x * scenario.totalDistance = scenario.totalDistance - scenario.walkingSpeed / scenario.motorcycleSpeed * scenario.totalDistance :=
by sorry

end NUMINAMATH_CALUDE_villager_A_motorcycle_fraction_l1604_160452


namespace NUMINAMATH_CALUDE_middle_integer_of_consecutive_evens_l1604_160475

theorem middle_integer_of_consecutive_evens (n : ℕ) : 
  n > 0 ∧ n < 10 ∧ n % 2 = 0 ∧
  (n - 2) > 0 ∧ (n + 2) < 10 ∧
  (n - 2) + n + (n + 2) = ((n - 2) * n * (n + 2)) / 8 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_middle_integer_of_consecutive_evens_l1604_160475


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l1604_160416

theorem largest_lcm_with_18 : 
  (Finset.image (fun x => Nat.lcm 18 x) {3, 6, 9, 12, 15, 18}).max = some 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l1604_160416


namespace NUMINAMATH_CALUDE_man_daily_wage_l1604_160444

/-- The daily wage of a man -/
def M : ℝ := sorry

/-- The daily wage of a woman -/
def W : ℝ := sorry

/-- The total wages of 24 men and 16 women per day -/
def total_wages : ℝ := 11600

/-- The number of men -/
def num_men : ℕ := 24

/-- The number of women -/
def num_women : ℕ := 16

/-- The wages of 24 men and 16 women amount to Rs. 11600 per day -/
axiom wage_equation : num_men * M + num_women * W = total_wages

/-- Half the number of men and 37 women earn the same amount per day -/
axiom half_men_equation : (num_men / 2) * M + 37 * W = total_wages

theorem man_daily_wage : M = 350 := by sorry

end NUMINAMATH_CALUDE_man_daily_wage_l1604_160444


namespace NUMINAMATH_CALUDE_u_eq_complement_a_union_b_l1604_160430

/-- The universal set U -/
def U : Finset Nat := {1, 2, 3, 4, 5, 7}

/-- Set A -/
def A : Finset Nat := {4, 7}

/-- Set B -/
def B : Finset Nat := {1, 3, 4, 7}

/-- Theorem stating that U is equal to the union of the complement of A in U and B -/
theorem u_eq_complement_a_union_b : U = (U \ A) ∪ B := by sorry

end NUMINAMATH_CALUDE_u_eq_complement_a_union_b_l1604_160430


namespace NUMINAMATH_CALUDE_find_correct_divisor_l1604_160495

theorem find_correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X / (D + 12) = 70) (h3 : X / D = 40) : D = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_correct_divisor_l1604_160495


namespace NUMINAMATH_CALUDE_geometric_progression_cubed_sum_l1604_160499

theorem geometric_progression_cubed_sum
  (b s : ℝ) (h1 : -1 < s) (h2 : s < 1) :
  let series := fun n => b^3 * s^(3*n)
  (∑' n, series n) = b^3 / (1 - s^3) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_cubed_sum_l1604_160499


namespace NUMINAMATH_CALUDE_kris_herbert_age_difference_l1604_160487

/-- The age difference between two people --/
def age_difference (age1 : ℕ) (age2 : ℕ) : ℕ := 
  if age1 ≥ age2 then age1 - age2 else age2 - age1

/-- Theorem: The age difference between Kris and Herbert is 10 years --/
theorem kris_herbert_age_difference : 
  let kris_age : ℕ := 24
  let herbert_age_next_year : ℕ := 15
  let herbert_age : ℕ := herbert_age_next_year - 1
  age_difference kris_age herbert_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_kris_herbert_age_difference_l1604_160487


namespace NUMINAMATH_CALUDE_quadrilateral_division_l1604_160474

/-- A triangle is a type representing a geometric triangle. -/
structure Triangle

/-- A quadrilateral is a type representing a geometric quadrilateral. -/
structure Quadrilateral

/-- Represents the concept of dividing a shape into equal parts. -/
def can_be_divided_into (n : ℕ) (T : Type) : Prop :=
  ∃ (parts : Fin n → T), ∀ i j : Fin n, i ≠ j → parts i ≠ parts j

/-- Any triangle can be divided into 4 equal triangles. -/
axiom triangle_division : can_be_divided_into 4 Triangle

/-- The main theorem: there exists a quadrilateral that can be divided into 7 equal triangles. -/
theorem quadrilateral_division : ∃ (Q : Quadrilateral), can_be_divided_into 7 Triangle :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_division_l1604_160474


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1604_160460

/-- A random variable following a normal distribution with mean 1 and standard deviation σ -/
def ξ (σ : ℝ) : Type := Unit

/-- The probability that ξ is less than a given value -/
def P_less_than (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The theorem stating that if P(ξ < 0) = 0.3, then P(ξ < 2) = 0.7 for ξ ~ N(1, σ²) -/
theorem normal_distribution_probability (σ : ℝ) (h : P_less_than σ 0 = 0.3) :
  P_less_than σ 2 = 0.7 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1604_160460


namespace NUMINAMATH_CALUDE_factorial_sum_l1604_160478

theorem factorial_sum : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 6 * Nat.factorial 6 = 40200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_l1604_160478


namespace NUMINAMATH_CALUDE_bunny_teddy_ratio_l1604_160479

def initial_teddies : ℕ := 5
def koala_bears : ℕ := 1
def additional_teddies_per_bunny : ℕ := 2
def total_mascots : ℕ := 51

def bunnies : ℕ := (total_mascots - initial_teddies - koala_bears) / (additional_teddies_per_bunny + 1)

theorem bunny_teddy_ratio :
  bunnies / initial_teddies = 3 ∧ bunnies % initial_teddies = 0 := by
  sorry

end NUMINAMATH_CALUDE_bunny_teddy_ratio_l1604_160479


namespace NUMINAMATH_CALUDE_first_12_average_l1604_160419

theorem first_12_average (total_count : Nat) (total_average : ℝ) (last_12_average : ℝ) (result_13 : ℝ) :
  total_count = 25 →
  total_average = 18 →
  last_12_average = 20 →
  result_13 = 90 →
  (((total_count : ℝ) * total_average - 12 * last_12_average - result_13) / 12 : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_12_average_l1604_160419


namespace NUMINAMATH_CALUDE_cards_kept_away_is_seven_l1604_160463

/-- The number of cards in a standard deck -/
def standard_deck : ℕ := 52

/-- The number of cards used in the game -/
def cards_used : ℕ := 45

/-- The number of cards kept away -/
def cards_kept_away : ℕ := standard_deck - cards_used

theorem cards_kept_away_is_seven : cards_kept_away = 7 := by
  sorry

end NUMINAMATH_CALUDE_cards_kept_away_is_seven_l1604_160463
