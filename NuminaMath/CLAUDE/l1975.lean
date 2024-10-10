import Mathlib

namespace line_intercept_sum_l1975_197538

/-- Given a line with equation 2x - 5y + 10 = 0, prove that the absolute value of the sum of its x and y intercepts is 3. -/
theorem line_intercept_sum (a b : ℝ) : 
  (2 * a - 5 * 0 + 10 = 0) →  -- x-intercept condition
  (2 * 0 - 5 * b + 10 = 0) →  -- y-intercept condition
  |a + b| = 3 := by
sorry

end line_intercept_sum_l1975_197538


namespace sum_of_roots_even_function_l1975_197516

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define a function that has exactly four roots
def HasFourRoots (f : ℝ → ℝ) : Prop := ∃ a b c d : ℝ, 
  (a < b ∧ b < c ∧ c < d) ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem sum_of_roots_even_function (f : ℝ → ℝ) 
  (h_even : EvenFunction f) (h_four_roots : HasFourRoots f) : 
  ∃ a b c d : ℝ, (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧ (a + b + c + d = 0) :=
sorry

end sum_of_roots_even_function_l1975_197516


namespace car_speed_l1975_197503

/-- Theorem: Given a car travels 300 miles in 5 hours, its speed is 60 miles per hour. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 300) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : speed = 60 := by
  sorry

end car_speed_l1975_197503


namespace sphere_radius_from_surface_area_l1975_197550

theorem sphere_radius_from_surface_area :
  ∀ (S R : ℝ), S = 4 * Real.pi → S = 4 * Real.pi * R^2 → R = 1 :=
by
  sorry

end sphere_radius_from_surface_area_l1975_197550


namespace fractional_exponent_simplification_l1975_197590

theorem fractional_exponent_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2 * b ^ (1/4))) / (((a * (b ^ (1/2))) ^ (1/2))) = a ^ (1/2) := by
  sorry

end fractional_exponent_simplification_l1975_197590


namespace john_scores_42_points_l1975_197500

/-- Calculates the total points scored by John given the specified conditions -/
def total_points_scored (points_per_interval : ℕ) (interval_duration : ℕ) (num_periods : ℕ) (period_duration : ℕ) : ℕ :=
  let total_duration := num_periods * period_duration
  let num_intervals := total_duration / interval_duration
  points_per_interval * num_intervals

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points : 
  let points_per_interval := 2 * 2 + 1 * 3  -- 2 two-point shots and 1 three-point shot
  let interval_duration := 4                -- every 4 minutes
  let num_periods := 2                      -- 2 periods
  let period_duration := 12                 -- each period is 12 minutes
  total_points_scored points_per_interval interval_duration num_periods period_duration = 42 := by
  sorry


end john_scores_42_points_l1975_197500


namespace star_commutative_iff_on_lines_l1975_197514

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem statement
theorem star_commutative_iff_on_lines (x y : ℝ) :
  star x y = star y x ↔ x = 0 ∨ y = 0 ∨ x + y = 0 ∨ x = y :=
sorry

end star_commutative_iff_on_lines_l1975_197514


namespace larger_circle_radius_l1975_197596

-- Define the radii of the three inner circles
def r₁ : ℝ := 2
def r₂ : ℝ := 3
def r₃ : ℝ := 10

-- Define the centers of the three inner circles
variable (A B C : ℝ × ℝ)

-- Define the center and radius of the larger circle
variable (O : ℝ × ℝ)
variable (R : ℝ)

-- Define the condition that all circles are touching one another
def circles_touching (A B C : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  (dist A B = r₁ + r₂) ∧ (dist B C = r₂ + r₃) ∧ (dist A C = r₁ + r₃)

-- Define the condition that the larger circle contains the three inner circles
def larger_circle_contains (O : ℝ × ℝ) (R : ℝ) (A B C : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  (dist O A = R - r₁) ∧ (dist O B = R - r₂) ∧ (dist O C = R - r₃)

-- The main theorem
theorem larger_circle_radius 
  (h₁ : circles_touching A B C r₁ r₂ r₃)
  (h₂ : larger_circle_contains O R A B C r₁ r₂ r₃) :
  R = 15 := by
  sorry

end larger_circle_radius_l1975_197596


namespace infinite_series_sum_l1975_197565

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 2))) = 1/2 := by sorry

end infinite_series_sum_l1975_197565


namespace sum_of_roots_l1975_197555

theorem sum_of_roots (k d : ℝ) (x₁ x₂ : ℝ) (h₁ : 4 * x₁^2 - k * x₁ = d)
    (h₂ : 4 * x₂^2 - k * x₂ = d) (h₃ : x₁ ≠ x₂) (h₄ : d ≠ 0) :
  x₁ + x₂ = k / 4 := by
sorry

end sum_of_roots_l1975_197555


namespace triangle_theorem_l1975_197597

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : t.a + t.c = 4) :
  t.B = π / 3 ∧ 
  ((t.a = 1 ∧ t.c = 3) ∨ (t.a = 3 ∧ t.c = 1)) := by
  sorry


end triangle_theorem_l1975_197597


namespace parallelogram_area_l1975_197566

/-- The area of a parallelogram with given side lengths and included angle -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (ha : a = 12) (hb : b = 18) (hθ : θ = 45 * π / 180) :
  abs (a * b * Real.sin θ - 152.73) < 0.01 := by
  sorry

end parallelogram_area_l1975_197566


namespace namjoon_books_l1975_197531

/-- The number of books Namjoon has in total -/
def total_books (a b c : ℕ) : ℕ := a + b + c

/-- Theorem stating the total number of books Namjoon has -/
theorem namjoon_books :
  ∀ (a b c : ℕ),
  a = 35 →
  b = a - 16 →
  c = b + 35 →
  total_books a b c = 108 := by
  sorry

end namjoon_books_l1975_197531


namespace sum_of_numbers_l1975_197567

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 7 ∨ b = 7 ∨ c = 7)

/-- Checks if a natural number contains the digit 3 -/
def containsThree (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 3 ∨ b = 3 ∨ c = 3)

theorem sum_of_numbers (A : ThreeDigitNumber) (B C : TwoDigitNumber) :
  ((containsSeven A.val ∧ containsSeven B.val) ∨
   (containsSeven A.val ∧ containsSeven C.val) ∨
   (containsSeven B.val ∧ containsSeven C.val)) →
  (containsThree B.val ∧ containsThree C.val) →
  (A.val + B.val + C.val = 208) →
  (B.val + C.val = 76) →
  A.val + B.val + C.val = 247 := by
  sorry

end sum_of_numbers_l1975_197567


namespace calculator_transformation_l1975_197529

/-- Transformation function for the calculator -/
def transform (a b : Int) : Int × Int :=
  match (a + b) % 4 with
  | 0 => (a + 1, b)
  | 1 => (a, b + 1)
  | 2 => (a - 1, b)
  | _ => (a, b - 1)

/-- Apply the transformation n times -/
def transformN (n : Nat) (a b : Int) : Int × Int :=
  match n with
  | 0 => (a, b)
  | n + 1 => 
    let (x, y) := transformN n a b
    transform x y

theorem calculator_transformation :
  transformN 6 1 12 = (-2, 15) :=
by sorry

end calculator_transformation_l1975_197529


namespace root_of_cubic_equation_l1975_197515

theorem root_of_cubic_equation :
  ∃ x : ℝ, (1/2 : ℝ) * x^3 + 4 = 0 ∧ x = -2 := by
  sorry

end root_of_cubic_equation_l1975_197515


namespace expression_evaluation_l1975_197506

theorem expression_evaluation : 
  let a : ℚ := 2
  let b : ℚ := 1/3
  3*(a^2 - a*b + 7) - 2*(3*a*b - a^2 + 1) + 3 = 36 := by sorry

end expression_evaluation_l1975_197506


namespace problem_solution_l1975_197581

theorem problem_solution (A B : ℝ) 
  (h1 : A + 2 * B = 814.8)
  (h2 : A = 10 * B) : 
  A - B = 611.1 := by
sorry

end problem_solution_l1975_197581


namespace inradius_exradius_inequality_l1975_197586

/-- Given a triangle ABC with inradius r, exradius r' touching side AB, and length c of side AB,
    prove that 4rr' ≤ c^2 -/
theorem inradius_exradius_inequality (r r' c : ℝ) (hr : r > 0) (hr' : r' > 0) (hc : c > 0) :
  4 * r * r' ≤ c^2 := by
  sorry

end inradius_exradius_inequality_l1975_197586


namespace online_store_sales_analysis_l1975_197591

/-- Represents the daily sales volume as a function of selling price -/
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 180

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 60) * (daily_sales_volume x)

/-- The original selling price -/
def original_price : ℝ := 80

/-- The cost price of each item -/
def cost_price : ℝ := 60

/-- The valid range for the selling price -/
def valid_price_range (x : ℝ) : Prop := 60 ≤ x ∧ x ≤ 80

theorem online_store_sales_analysis 
  (x : ℝ) 
  (h : valid_price_range x) :
  (daily_sales_volume x = -2 * x + 180) ∧
  (∃ x₁, daily_profit x₁ = 432 ∧ x₁ = 72) ∧
  (∃ x₂, ∀ y, valid_price_range y → daily_profit x₂ ≥ daily_profit y ∧ x₂ = 75) := by
  sorry

end online_store_sales_analysis_l1975_197591


namespace max_leap_years_in_period_l1975_197559

/-- A calendrical system where leap years occur every three years -/
structure ModifiedCalendar where
  leapYearInterval : ℕ
  leapYearInterval_eq : leapYearInterval = 3

/-- The number of years in the period we're considering -/
def periodLength : ℕ := 100

/-- The maximum number of leap years in the given period -/
def maxLeapYears (c : ModifiedCalendar) : ℕ :=
  periodLength / c.leapYearInterval

/-- Theorem stating that the maximum number of leap years in a 100-year period is 33 -/
theorem max_leap_years_in_period (c : ModifiedCalendar) :
  maxLeapYears c = 33 := by
  sorry

#check max_leap_years_in_period

end max_leap_years_in_period_l1975_197559


namespace power_fraction_equality_l1975_197511

theorem power_fraction_equality : (10 ^ 20 : ℚ) / (50 ^ 10) = 2 ^ 10 := by sorry

end power_fraction_equality_l1975_197511


namespace odd_function_sum_l1975_197557

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_sum (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : HasPeriod (fun x ↦ f (2 * x + 1)) 5) (h3 : f 1 = 5) :
  f 2009 + f 2010 = 0 := by sorry

end odd_function_sum_l1975_197557


namespace consecutive_integers_not_sum_of_squares_l1975_197525

theorem consecutive_integers_not_sum_of_squares :
  ∃ m : ℕ+, ∀ k : ℕ, k < 2017 → ¬∃ a b : ℤ, (m + k : ℤ) = a^2 + b^2 := by
  sorry

end consecutive_integers_not_sum_of_squares_l1975_197525


namespace no_very_convex_function_l1975_197578

/-- A function is very convex if it satisfies the given inequality for all real x and y -/
def VeryConvex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

/-- Theorem stating that no very convex function exists -/
theorem no_very_convex_function : ¬ ∃ f : ℝ → ℝ, VeryConvex f := by
  sorry


end no_very_convex_function_l1975_197578


namespace balloon_difference_l1975_197517

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of balloons Jake initially brought to the park -/
def jake_initial_balloons : ℕ := 3

/-- The number of additional balloons Jake bought at the park -/
def jake_additional_balloons : ℕ := 4

/-- The total number of balloons Jake had at the park -/
def jake_total_balloons : ℕ := jake_initial_balloons + jake_additional_balloons

/-- Theorem stating the difference in balloons between Jake and Allan -/
theorem balloon_difference : jake_total_balloons - allan_balloons = 1 := by
  sorry

end balloon_difference_l1975_197517


namespace equilateral_triangle_area_perimeter_ratio_l1975_197524

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let perimeter : ℝ := 3 * side_length
  area / perimeter^2 = Real.sqrt 3 / 36 := by sorry

end equilateral_triangle_area_perimeter_ratio_l1975_197524


namespace range_of_a_l1975_197587

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 < 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 ≤ 0 ∧ x^2 - 8*x - 20 ≥ 0) ∧
  (a > 0) →
  a ≥ 9 :=
by sorry

end range_of_a_l1975_197587


namespace hyperbola_upper_focus_l1975_197562

/-- Given a hyperbola with equation y^2/16 - x^2/9 = 1, prove that the coordinates of the upper focus are (0, 5) -/
theorem hyperbola_upper_focus (x y : ℝ) :
  (y^2 / 16) - (x^2 / 9) = 1 →
  ∃ (a b c : ℝ),
    a = 4 ∧
    b = 3 ∧
    c^2 = a^2 + b^2 ∧
    (0, c) = (0, 5) :=
by sorry

end hyperbola_upper_focus_l1975_197562


namespace arithmetic_sequence_squares_existence_and_uniqueness_l1975_197589

/-- Proves the existence and uniqueness of k that satisfies the given conditions --/
theorem arithmetic_sequence_squares_existence_and_uniqueness :
  ∃! k : ℤ, ∃ n d : ℤ,
    (n - d)^2 = 36 + k ∧
    n^2 = 300 + k ∧
    (n + d)^2 = 596 + k ∧
    k = 925 := by
  sorry

end arithmetic_sequence_squares_existence_and_uniqueness_l1975_197589


namespace comparison_of_fractions_l1975_197526

theorem comparison_of_fractions :
  (1/2 : ℚ) < (2/2 : ℚ) →
  (1 - 5/6 : ℚ) > (1 - 7/6 : ℚ) →
  (-π : ℝ) < -3.14 →
  (-2/3 : ℚ) > (-4/5 : ℚ) :=
by
  sorry

end comparison_of_fractions_l1975_197526


namespace sprint_jog_difference_l1975_197545

-- Define the distances
def sprint_distance : ℚ := 875 / 1000
def jog_distance : ℚ := 75 / 100

-- Theorem statement
theorem sprint_jog_difference :
  sprint_distance - jog_distance = 125 / 1000 := by
  sorry

end sprint_jog_difference_l1975_197545


namespace supermarket_can_display_l1975_197537

/-- Sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a₁ aₙ n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The problem statement -/
theorem supermarket_can_display :
  let a₁ : ℕ := 28  -- first term
  let aₙ : ℕ := 1   -- last term
  let n : ℕ := 10   -- number of terms
  arithmeticSequenceSum a₁ aₙ n = 145 := by
  sorry


end supermarket_can_display_l1975_197537


namespace license_plate_count_l1975_197593

/-- Represents the number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- Represents the number of possible letters (A-Z) -/
def letter_choices : ℕ := 26

/-- Represents the number of digits in a license plate -/
def num_digits : ℕ := 6

/-- Represents the number of adjacent letters in a license plate -/
def num_adjacent_letters : ℕ := 2

/-- Represents the number of positions for the adjacent letter pair -/
def adjacent_letter_positions : ℕ := 7

/-- Represents the number of positions for the optional letter -/
def optional_letter_positions : ℕ := 2

/-- Calculates the total number of distinct license plates -/
def total_license_plates : ℕ :=
  adjacent_letter_positions * 
  optional_letter_positions * 
  digit_choices^num_digits * 
  letter_choices^(num_adjacent_letters + 1)

/-- Theorem stating that the total number of distinct license plates is 936,520,000 -/
theorem license_plate_count : total_license_plates = 936520000 := by
  sorry

end license_plate_count_l1975_197593


namespace sin_585_degrees_l1975_197576

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l1975_197576


namespace quadratic_inequality_l1975_197549

/-- A quadratic function f(x) = x^2 + 4x + c, where c is a constant. -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- Theorem stating that for the quadratic function f(x) = x^2 + 4x + c,
    the inequality f(1) > f(0) > f(-2) holds for any constant c. -/
theorem quadratic_inequality (c : ℝ) : f c 1 > f c 0 ∧ f c 0 > f c (-2) := by
  sorry

end quadratic_inequality_l1975_197549


namespace austin_robot_purchase_l1975_197542

theorem austin_robot_purchase (num_robots : ℕ) (robot_cost tax change : ℚ) : 
  num_robots = 7 → 
  robot_cost = 8.75 → 
  tax = 7.22 → 
  change = 11.53 → 
  (num_robots : ℚ) * robot_cost + tax + change = 80 :=
by sorry

end austin_robot_purchase_l1975_197542


namespace parallel_lines_a_value_l1975_197543

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes (-A/B) are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ A1 / B1 = A2 / B2

/-- Two lines in the form Ax + By + C = 0 are identical if and only if their coefficients are proportional -/
def identical (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ A1 = k * A2 ∧ B1 = k * B2 ∧ C1 = k * C2

theorem parallel_lines_a_value : 
  ∃! a : ℝ, parallel (a + 1) 3 3 1 (a - 1) 1 ∧ ¬identical (a + 1) 3 3 1 (a - 1) 1 ∧ a = -2 := by
  sorry

end parallel_lines_a_value_l1975_197543


namespace specific_cube_structure_surface_area_l1975_197598

/-- A solid structure composed of unit cubes -/
structure CubeStructure :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)
  (total_cubes : ℕ)

/-- Calculate the surface area of a CubeStructure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.length * s.width + s.length * s.height + s.width * s.height)

/-- Theorem stating that a specific CubeStructure has a surface area of 78 square units -/
theorem specific_cube_structure_surface_area :
  ∃ (s : CubeStructure), s.length = 5 ∧ s.width = 3 ∧ s.height = 3 ∧ s.total_cubes = 15 ∧ surface_area s = 78 :=
by
  sorry

end specific_cube_structure_surface_area_l1975_197598


namespace fraction_closest_to_longest_side_specific_trapezoid_l1975_197507

/-- Represents a trapezoid field -/
structure TrapezoidField where
  base1 : ℝ
  base2 : ℝ
  angle1 : ℝ
  angle2 : ℝ

/-- The fraction of area closer to the longest side of the trapezoid field -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

/-- Theorem stating the fraction of area closest to the longest side for the given trapezoid -/
theorem fraction_closest_to_longest_side_specific_trapezoid :
  let field : TrapezoidField := {
    base1 := 200,
    base2 := 100,
    angle1 := 45,
    angle2 := 135
  }
  fraction_closest_to_longest_side field = 7/12 := by
  sorry

end fraction_closest_to_longest_side_specific_trapezoid_l1975_197507


namespace matrix_multiplication_result_l1975_197558

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 5, 7]
  let B : Matrix (Fin 2) (Fin 3) ℤ := !![2, 1, 4; 1, 0, -2]
  A * B = !![5, 3, 14; 17, 5, 6] := by
sorry

end matrix_multiplication_result_l1975_197558


namespace range_of_a_l1975_197599

/-- Proposition p: The function y=(a-1)^x is increasing with respect to x -/
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) ^ x < (a - 1) ^ y

/-- Proposition q: The inequality -3^x ≤ a is true for all positive real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → -3 ^ x ≤ a

/-- The range of a given the conditions -/
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : -1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l1975_197599


namespace calculate_expression_l1975_197583

theorem calculate_expression : (-3 : ℚ) * (1/3) / (-1/3) * 3 = 9 := by
  sorry

end calculate_expression_l1975_197583


namespace tree_planting_impossibility_l1975_197579

theorem tree_planting_impossibility :
  ∀ (arrangement : List ℕ),
    (arrangement.length = 50) →
    (∀ n : ℕ, n ∈ arrangement → 1 ≤ n ∧ n ≤ 25) →
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 → (arrangement.count n = 2)) →
    ¬(∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 →
      ∃ (i j : ℕ), i < j ∧ 
        arrangement.nthLe i sorry = n ∧
        arrangement.nthLe j sorry = n ∧
        (j - i = 2 ∨ j - i = 4)) :=
by sorry

end tree_planting_impossibility_l1975_197579


namespace blue_marbles_count_l1975_197540

/-- Given a bag of marbles with a 3:5 ratio of red to blue marbles and 18 red marbles,
    prove that there are 30 blue marbles. -/
theorem blue_marbles_count (red_count : ℕ) (ratio_red : ℕ) (ratio_blue : ℕ)
    (h_red_count : red_count = 18)
    (h_ratio : ratio_red = 3 ∧ ratio_blue = 5) :
    red_count * ratio_blue / ratio_red = 30 := by
  sorry

end blue_marbles_count_l1975_197540


namespace charles_cleaning_time_l1975_197546

theorem charles_cleaning_time 
  (alice_time : ℝ) 
  (bob_time : ℝ) 
  (charles_time : ℝ) 
  (h1 : alice_time = 20) 
  (h2 : bob_time = 3/4 * alice_time) 
  (h3 : charles_time = 2/3 * bob_time) : 
  charles_time = 10 := by
sorry

end charles_cleaning_time_l1975_197546


namespace sum_of_specific_terms_l1975_197521

/-- The sum of the 7th to 10th terms of a sequence defined by S_n = 2n^2 - 3n + 1 is 116 -/
theorem sum_of_specific_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = 2 * n^2 - 3 * n + 1) →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  a 7 + a 8 + a 9 + a 10 = 116 := by
sorry

end sum_of_specific_terms_l1975_197521


namespace calculate_F_2_f_3_l1975_197568

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 3*a + 2
def F (a b : ℝ) : ℝ := b^2 + a + 1

-- State the theorem
theorem calculate_F_2_f_3 : F 2 (f 3) = 7 := by
  sorry

end calculate_F_2_f_3_l1975_197568


namespace toy_store_revenue_l1975_197556

theorem toy_store_revenue (D : ℝ) (h1 : D > 0) : 
  let nov := (2 / 5 : ℝ) * D
  let jan := (1 / 2 : ℝ) * nov
  let avg := (nov + jan) / 2
  D / avg = 10 / 3 := by sorry

end toy_store_revenue_l1975_197556


namespace joan_books_l1975_197551

theorem joan_books (initial_books sold_books : ℕ) 
  (h1 : initial_books = 33)
  (h2 : sold_books = 26) :
  initial_books - sold_books = 7 := by
  sorry

end joan_books_l1975_197551


namespace final_season_premiere_l1975_197592

/-- The number of days needed to watch all episodes of a TV series -/
def days_to_watch (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Proof that it takes 10 days to watch all episodes -/
theorem final_season_premiere :
  days_to_watch 4 15 6 = 10 := by
  sorry

#eval days_to_watch 4 15 6

end final_season_premiere_l1975_197592


namespace count_less_than_ten_l1975_197575

def travel_times : List Nat := [10, 12, 15, 6, 3, 8, 9]

def less_than_ten (n : Nat) : Bool := n < 10

theorem count_less_than_ten :
  (travel_times.filter less_than_ten).length = 4 := by
  sorry

end count_less_than_ten_l1975_197575


namespace paper_pickup_sum_l1975_197584

theorem paper_pickup_sum : 127.5 + 345.25 + 518.75 = 991.5 := by
  sorry

end paper_pickup_sum_l1975_197584


namespace beta_value_l1975_197582

theorem beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_α_β : Real.sin (α - β) = -(Real.sqrt 10) / 10) :
  β = π/4 := by
sorry

end beta_value_l1975_197582


namespace smallest_sum_proof_l1975_197577

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/5, 1/4 + 1/6, 1/4 + 1/9, 1/4 + 1/8, 1/4 + 1/7]
  (∀ s ∈ sums, 1/4 + 1/9 ≤ s) ∧ (1/4 + 1/9 = 13/36) := by
  sorry

end smallest_sum_proof_l1975_197577


namespace arithmetic_sequence_75th_term_l1975_197539

/-- Given an arithmetic sequence where the first term is 3 and the 25th term is 51,
    prove that the 75th term is 151. -/
theorem arithmetic_sequence_75th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                                -- first term is 3
    a 24 = 51 →                              -- 25th term is 51
    a 74 = 151 :=                            -- 75th term is 151
by
  sorry

end arithmetic_sequence_75th_term_l1975_197539


namespace diamond_ratio_equals_five_thirds_l1975_197585

-- Define the diamond operation
def diamond (n m : ℤ) : ℤ := n^2 * m^3

-- Theorem statement
theorem diamond_ratio_equals_five_thirds :
  (diamond 3 5) / (diamond 5 3) = 5 / 3 := by
  sorry

end diamond_ratio_equals_five_thirds_l1975_197585


namespace f_min_max_l1975_197509

-- Define the function
def f (x : ℝ) : ℝ := 1 + 3*x - x^3

-- State the theorem
theorem f_min_max : 
  (∃ x : ℝ, f x = -1) ∧ 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = 3) ∧ 
  (∀ x : ℝ, f x ≤ 3) :=
sorry

end f_min_max_l1975_197509


namespace stratified_sampling_theorem_l1975_197544

structure Department where
  total : ℕ
  males : ℕ
  females : ℕ

def sample_size : ℕ := 3

def dept_A : Department := ⟨10, 6, 4⟩
def dept_B : Department := ⟨5, 3, 2⟩

def total_staff : ℕ := dept_A.total + dept_B.total

def stratified_sample (d : Department) : ℕ :=
  (sample_size * d.total) / total_staff

def prob_at_least_one_female (d : Department) (n : ℕ) : ℚ :=
  1 - (Nat.choose d.males n : ℚ) / (Nat.choose d.total n : ℚ)

def prob_male_count (k : ℕ) : ℚ := 
  if k = 0 then 4 / 75
  else if k = 1 then 22 / 75
  else if k = 2 then 34 / 75
  else if k = 3 then 1 / 3
  else 0

def expected_male_count : ℚ := 2

theorem stratified_sampling_theorem :
  (stratified_sample dept_A = 2) ∧
  (stratified_sample dept_B = 1) ∧
  (prob_at_least_one_female dept_A 2 = 2 / 3) ∧
  (∀ k, 0 ≤ k ∧ k ≤ 3 → prob_male_count k = prob_male_count k) ∧
  (Finset.sum (Finset.range 4) (λ k => k * prob_male_count k) = expected_male_count) := by
  sorry

end stratified_sampling_theorem_l1975_197544


namespace sum_of_roots_quadratic_l1975_197504

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m^2 - 4*m - 2 = 0) → (n^2 - 4*n - 2 = 0) → m + n = 4 := by
  sorry

end sum_of_roots_quadratic_l1975_197504


namespace converse_statement_l1975_197528

theorem converse_statement (a b : ℝ) :
  (∀ a b, a > 1 ∧ b > 1 → a + b > 2) →
  (∀ a b, a + b ≤ 2 → a ≤ 1 ∨ b ≤ 1) :=
by sorry

end converse_statement_l1975_197528


namespace sum_and_ratio_to_difference_l1975_197552

theorem sum_and_ratio_to_difference (x y : ℝ) 
  (sum_eq : x + y = 399)
  (ratio_eq : x / y = 0.9) : 
  y - x = 21 := by
sorry

end sum_and_ratio_to_difference_l1975_197552


namespace convenient_logistics_boxes_l1975_197553

/-- Represents the number of large boxes -/
def large_boxes : ℕ := 8

/-- Represents the number of small boxes -/
def small_boxes : ℕ := 21 - large_boxes

/-- The total number of bottles -/
def total_bottles : ℕ := 2000

/-- The capacity of a large box -/
def large_box_capacity : ℕ := 120

/-- The capacity of a small box -/
def small_box_capacity : ℕ := 80

/-- The total number of boxes -/
def total_boxes : ℕ := 21

theorem convenient_logistics_boxes :
  large_boxes * large_box_capacity + small_boxes * small_box_capacity = total_bottles ∧
  large_boxes + small_boxes = total_boxes :=
by sorry

end convenient_logistics_boxes_l1975_197553


namespace loss_ratio_is_one_third_l1975_197563

/-- Represents a baseball team's season statistics -/
structure BaseballSeason where
  total_games : ℕ
  away_games : ℕ
  home_game_wins : ℕ
  away_game_wins : ℕ
  home_game_wins_extra_innings : ℕ

/-- Calculates the ratio of away game losses to home game losses not in extra innings -/
def loss_ratio (season : BaseballSeason) : Rat :=
  let home_games := season.total_games - season.away_games
  let home_game_losses := home_games - season.home_game_wins
  let away_game_losses := season.away_games - season.away_game_wins
  away_game_losses / home_game_losses

/-- Theorem stating the loss ratio for the given season is 1/3 -/
theorem loss_ratio_is_one_third (season : BaseballSeason) 
  (h1 : season.total_games = 45)
  (h2 : season.away_games = 15)
  (h3 : season.home_game_wins = 6)
  (h4 : season.away_game_wins = 7)
  (h5 : season.home_game_wins_extra_innings = 3) :
  loss_ratio season = 1/3 := by
  sorry

end loss_ratio_is_one_third_l1975_197563


namespace min_gb_for_y_cheaper_l1975_197519

/-- Cost of Plan X in cents for g gigabytes -/
def cost_x (g : ℕ) : ℕ := 15 * g

/-- Cost of Plan Y in cents for g gigabytes -/
def cost_y (g : ℕ) : ℕ :=
  if g ≤ 500 then
    3000 + 8 * g
  else
    3000 + 8 * 500 + 6 * (g - 500)

/-- Predicate to check if Plan Y is cheaper than Plan X for g gigabytes -/
def y_cheaper_than_x (g : ℕ) : Prop :=
  cost_y g < cost_x g

theorem min_gb_for_y_cheaper :
  ∀ g : ℕ, g < 778 → ¬(y_cheaper_than_x g) ∧
  y_cheaper_than_x 778 :=
sorry

end min_gb_for_y_cheaper_l1975_197519


namespace count_cubic_polynomials_satisfying_property_l1975_197571

/-- A polynomial function of degree 3 -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The property that f(x)f(-x) = f(x³) -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that there are exactly 16 cubic polynomials satisfying the property -/
theorem count_cubic_polynomials_satisfying_property :
  ∃! (s : Finset (ℝ → ℝ)),
    (∀ f ∈ s, ∃ a b c d, f = CubicPolynomial a b c d ∧ SatisfiesProperty f) ∧
    Finset.card s = 16 := by sorry

end count_cubic_polynomials_satisfying_property_l1975_197571


namespace quadratic_polynomial_unique_l1975_197547

theorem quadratic_polynomial_unique (q : ℝ → ℝ) :
  (q = λ x => (67/30) * x^2 - (39/10) * x - 2/15) ↔
  (q (-1) = 6 ∧ q 2 = 1 ∧ q 4 = 20) :=
by sorry

end quadratic_polynomial_unique_l1975_197547


namespace probability_three_red_balls_l1975_197527

/-- The probability of picking 3 red balls from a bag containing 7 red, 9 blue, and 5 green balls -/
theorem probability_three_red_balls (red blue green : ℕ) (total : ℕ) : 
  red = 7 → blue = 9 → green = 5 → total = red + blue + green →
  (red / total) * ((red - 1) / (total - 1)) * ((red - 2) / (total - 2)) = 1 / 38 := by
  sorry

end probability_three_red_balls_l1975_197527


namespace intersection_of_A_and_B_l1975_197512

-- Define set A
def A : Set ℝ := {x | Real.sqrt (x^2 - 1) / Real.sqrt x = 0}

-- Define set B
def B : Set ℝ := {y | -2 ≤ y ∧ y ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end intersection_of_A_and_B_l1975_197512


namespace pascal_triangle_row20_element5_value_l1975_197532

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The fifth element (k = 4) in Row 20 of Pascal's triangle -/
def pascal_triangle_row20_element5 : ℕ := binomial 20 4

theorem pascal_triangle_row20_element5_value :
  pascal_triangle_row20_element5 = 4845 := by
  sorry

end pascal_triangle_row20_element5_value_l1975_197532


namespace balloon_count_correct_l1975_197505

/-- The number of red balloons Fred has -/
def fred_balloons : ℕ := 10

/-- The number of red balloons Sam has -/
def sam_balloons : ℕ := 46

/-- The number of red balloons Dan has -/
def dan_balloons : ℕ := 16

/-- The total number of red balloons -/
def total_balloons : ℕ := 72

theorem balloon_count_correct : fred_balloons + sam_balloons + dan_balloons = total_balloons := by
  sorry

end balloon_count_correct_l1975_197505


namespace product_of_difference_and_sum_of_squares_l1975_197530

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 34) : 
  a * b = 4.5 := by
sorry

end product_of_difference_and_sum_of_squares_l1975_197530


namespace circle_parabola_tangency_l1975_197533

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a parabola with equation y = x^2 + 1 -/
def Parabola : Point → Prop :=
  fun p => p.y = p.x^2 + 1

/-- Check if a circle is tangent to the parabola at two points -/
def IsTangent (c : Circle) (p1 p2 : Point) : Prop :=
  Parabola p1 ∧ Parabola p2 ∧
  (c.center.x - p1.x)^2 + (c.center.y - p1.y)^2 = c.radius^2 ∧
  (c.center.x - p2.x)^2 + (c.center.y - p2.y)^2 = c.radius^2

/-- The main theorem -/
theorem circle_parabola_tangency 
  (c : Circle) (p1 p2 : Point) (h : IsTangent c p1 p2) :
  c.center.y - p1.y = p1.x^2 - 1/2 :=
by
  sorry


end circle_parabola_tangency_l1975_197533


namespace order_of_abc_l1975_197561

theorem order_of_abc (a b c : ℝ) : 
  a = (Real.exp 1)⁻¹ → 
  b = (Real.log 3) / 3 → 
  c = (Real.log 4) / 4 → 
  a > b ∧ b > c := by
  sorry

end order_of_abc_l1975_197561


namespace inequality_proof_l1975_197508

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  a^3 / (b+c+d) + b^3 / (c+d+a) + c^3 / (d+a+b) + d^3 / (a+b+c) ≥ 1/3 := by
sorry

end inequality_proof_l1975_197508


namespace parabola_c_value_l1975_197513

/-- A parabola in the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ := p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 2 = -4 →   -- vertex (-4, 2)
  p.x_coord 4 = -2 →   -- point (-2, 4)
  p.x_coord 0 = -2 →   -- point (-2, 0)
  p.c = -2 := by sorry

end parabola_c_value_l1975_197513


namespace frustum_cut_off_height_l1975_197588

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  originalHeight : ℝ
  frustumHeight : ℝ
  upperRadius : ℝ
  lowerRadius : ℝ

/-- Calculates the height of the smaller cone cut off from the original cone -/
def cutOffHeight (f : Frustum) : ℝ :=
  f.originalHeight - f.frustumHeight

theorem frustum_cut_off_height (f : Frustum) 
  (h1 : f.originalHeight = 30)
  (h2 : f.frustumHeight = 18)
  (h3 : f.upperRadius = 6)
  (h4 : f.lowerRadius = 10) :
  cutOffHeight f = 12 := by
sorry

end frustum_cut_off_height_l1975_197588


namespace largest_nine_digit_divisible_by_127_l1975_197554

theorem largest_nine_digit_divisible_by_127 :
  ∀ n : ℕ, n ≤ 999999999 ∧ n % 127 = 0 → n ≤ 999999945 :=
by
  sorry

end largest_nine_digit_divisible_by_127_l1975_197554


namespace cubic_sum_ge_product_sum_l1975_197518

theorem cubic_sum_ge_product_sum (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
  u^3 + v^3 ≥ u^2 * v + v^2 * u := by
sorry

end cubic_sum_ge_product_sum_l1975_197518


namespace grape_juice_amount_l1975_197541

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem: The amount of grape juice in the drink is 105 ounces -/
theorem grape_juice_amount (drink : FruitDrink) 
  (h1 : drink.total = 300)
  (h2 : drink.orange_percent = 0.25)
  (h3 : drink.watermelon_percent = 0.40)
  (h4 : drink.grape_ounces = drink.total - (drink.orange_percent * drink.total + drink.watermelon_percent * drink.total)) :
  drink.grape_ounces = 105 := by
  sorry

end grape_juice_amount_l1975_197541


namespace arithmetic_sequence_problem_l1975_197595

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: If a_6 = S_3 = 12 in an arithmetic sequence, then a_8 = 16 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = 12) (h2 : seq.S 3 = 12) : seq.a 8 = 16 := by
  sorry


end arithmetic_sequence_problem_l1975_197595


namespace arithmetic_sequence_general_term_l1975_197564

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  t : ℝ
  h1 : 0 < d
  h2 : a 1 = 1
  h3 : ∀ n, 2 * (a n * a (n + 1) + 1) = t * n * (1 + a n)
  h4 : ∀ n, a (n + 1) = a n + d

/-- The general term of the arithmetic sequence is 2n - 1 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, n > 0 → seq.a n = 2 * n - 1 := by sorry

end arithmetic_sequence_general_term_l1975_197564


namespace solve_complex_equation_l1975_197520

theorem solve_complex_equation (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (a - i)^2 = 2*i) : a = -1 := by
  sorry

end solve_complex_equation_l1975_197520


namespace quadratic_two_distinct_roots_l1975_197522

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m = 0 ∧ y^2 + 2*y + m = 0) ↔ m < 1 :=
sorry

end quadratic_two_distinct_roots_l1975_197522


namespace concentric_circles_chord_theorem_l1975_197580

/-- Represents two concentric circles with chords of the outer circle tangent to the inner circle -/
structure ConcentricCircles where
  outer : ℝ → ℝ → Prop
  inner : ℝ → ℝ → Prop
  is_concentric : Prop
  tangent_chords : Prop

/-- The angle between two adjacent chords -/
def chord_angle (c : ConcentricCircles) : ℝ := 60

/-- The number of chords needed to complete a full circle -/
def num_chords (c : ConcentricCircles) : ℕ := 3

theorem concentric_circles_chord_theorem (c : ConcentricCircles) :
  chord_angle c = 60 → num_chords c = 3 := by sorry

end concentric_circles_chord_theorem_l1975_197580


namespace stone_pile_combination_l1975_197569

/-- Two piles are considered similar if their sizes differ by at most a factor of two -/
def similar (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A combining operation takes two piles and creates a new pile with their combined size -/
def combine (x y : ℕ) : ℕ := x + y

/-- A sequence of combining operations -/
def combineSequence : List (ℕ × ℕ) → List ℕ
  | [] => []
  | (x, y) :: rest => combine x y :: combineSequence rest

/-- The theorem states that for any number of stones, there exists a sequence of
    combining operations that results in a single pile, using only similar piles -/
theorem stone_pile_combination (n : ℕ) :
  ∃ (seq : List (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ seq → similar x y) ∧
    (combineSequence seq = [n]) ∧
    (seq.foldl (λ acc (x, y) => acc - 1) n = 1) :=
  sorry

end stone_pile_combination_l1975_197569


namespace log_sum_equality_l1975_197574

theorem log_sum_equality : 
  Real.log 8 / Real.log 2 + 3 * (Real.log 4 / Real.log 2) + 
  2 * (Real.log 16 / Real.log 8) + (Real.log 64 / Real.log 4) = 44 / 3 := by
  sorry

end log_sum_equality_l1975_197574


namespace line_separate_from_circle_l1975_197572

/-- A point inside a circle that is not the center of the circle -/
structure PointInsideCircle (a : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  inside : x₀^2 + y₀^2 < a^2
  not_center : (x₀, y₀) ≠ (0, 0)

/-- The line determined by the point inside the circle -/
def line_equation (a : ℝ) (p : PointInsideCircle a) (x y : ℝ) : Prop :=
  p.x₀ * x + p.y₀ * y = a^2

/-- The circle equation -/
def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = a^2

theorem line_separate_from_circle (a : ℝ) (ha : a > 0) (p : PointInsideCircle a) :
  ∀ x y : ℝ, line_equation a p x y → ¬circle_equation a x y :=
sorry

end line_separate_from_circle_l1975_197572


namespace cone_volume_from_circle_sector_l1975_197536

/-- The volume of a cone formed by rolling up a three-quarter sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 4) :
  let circumference := (3/4) * (2 * π * r)
  let base_radius := circumference / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * height = 3 * π * Real.sqrt 7 := by
  sorry

end cone_volume_from_circle_sector_l1975_197536


namespace inverse_log_inequality_l1975_197523

theorem inverse_log_inequality (n : ℝ) (h1 : n ≥ 2) :
  (1 / Real.log n) > (1 / (n - 1) - 1 / (n + 1)) :=
by
  -- Proof goes here
  sorry

-- Given condition
axiom log_inequality (x : ℝ) (h : x > 1) : Real.log x < x - 1

end inverse_log_inequality_l1975_197523


namespace min_value_of_function_l1975_197535

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, y - 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b condition
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 9 :=
by sorry

end min_value_of_function_l1975_197535


namespace quadratic_non_real_roots_l1975_197560

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 9 = 0 → x.im ≠ 0) ↔ -6 < b ∧ b < 6 := by
sorry

end quadratic_non_real_roots_l1975_197560


namespace solve_linear_equation_l1975_197501

theorem solve_linear_equation (x y : ℝ) :
  2 * x - 3 * y = 4 → y = (2 * x - 4) / 3 := by
  sorry

end solve_linear_equation_l1975_197501


namespace rectangle_perimeter_rectangle_perimeter_proof_l1975_197548

/-- The perimeter of a rectangle with width 16 and length 19 is 70 -/
theorem rectangle_perimeter : ℕ → ℕ → ℕ
  | 16, 19 => 70
  | _, _ => 0  -- Default case for other inputs

/-- The perimeter of a rectangle is twice the sum of its length and width -/
def perimeter (width length : ℕ) : ℕ := 2 * (width + length)

theorem rectangle_perimeter_proof (width length : ℕ) (h1 : width = 16) (h2 : length = 19) :
  perimeter width length = rectangle_perimeter width length := by
  sorry

end rectangle_perimeter_rectangle_perimeter_proof_l1975_197548


namespace father_sons_average_age_l1975_197502

/-- The average age of a father and his two sons -/
def average_age (father_age son1_age son2_age : ℕ) : ℚ :=
  (father_age + son1_age + son2_age) / 3

/-- Theorem stating the average age of the father and his two sons -/
theorem father_sons_average_age :
  ∀ (father_age son1_age son2_age : ℕ),
  father_age = 32 →
  son1_age - son2_age = 4 →
  (son1_age - 5 + son2_age - 5) / 2 = 15 →
  average_age father_age son1_age son2_age = 24 :=
by
  sorry


end father_sons_average_age_l1975_197502


namespace concentric_circles_theorem_l1975_197534

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) :=
  (h : R > r)

/-- Points on the circles -/
structure Points (R r : ℝ) extends ConcentricCircles R r :=
  (P : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = r^2)
  (hA : A.1^2 + A.2^2 = r^2)
  (hB : B.1^2 + B.2^2 = R^2)
  (hC : C.1^2 + C.2^2 = R^2)
  (hPerp : (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0)

/-- The theorem to be proved -/
theorem concentric_circles_theorem (R r : ℝ) (pts : Points R r) :
  let BC := (pts.B.1 - pts.C.1)^2 + (pts.B.2 - pts.C.2)^2
  let CA := (pts.C.1 - pts.A.1)^2 + (pts.C.2 - pts.A.2)^2
  let AB := (pts.A.1 - pts.B.1)^2 + (pts.A.2 - pts.B.2)^2
  let midpoint := ((pts.A.1 + pts.B.1) / 2, (pts.A.2 + pts.B.2) / 2)
  (BC + CA + AB = 6 * R^2 + 2 * r^2) ∧
  ((midpoint.1 + r/2)^2 + midpoint.2^2 = (R/2)^2) :=
sorry

end concentric_circles_theorem_l1975_197534


namespace negation_relationship_l1975_197570

theorem negation_relationship (x : ℝ) :
  (¬(x^2 + x - 6 > 0) → ¬(16 - x^2 < 0)) ∧
  ¬(¬(16 - x^2 < 0) → ¬(x^2 + x - 6 > 0)) :=
by sorry

end negation_relationship_l1975_197570


namespace x_plus_y_value_l1975_197573

theorem x_plus_y_value (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 17) 
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 21 / 17 := by
sorry

end x_plus_y_value_l1975_197573


namespace complex_number_extrema_l1975_197510

theorem complex_number_extrema (x y : ℝ) (z : ℂ) (h : z = x + y * I) 
  (h_bound : Complex.abs (z - I) ≤ 1) :
  let A := x * (Complex.abs (z - I)^2 - 1)
  ∃ (z_max z_min : ℂ),
    (∀ w : ℂ, Complex.abs (w - I) ≤ 1 → 
      x * (Complex.abs (w - I)^2 - 1) ≤ 2 * Real.sqrt 3 / 9) ∧
    (∀ w : ℂ, Complex.abs (w - I) ≤ 1 → 
      x * (Complex.abs (w - I)^2 - 1) ≥ -2 * Real.sqrt 3 / 9) ∧
    z_max = Real.sqrt 3 / 3 + I ∧
    z_min = -Real.sqrt 3 / 3 + I ∧
    x * (Complex.abs (z_max - I)^2 - 1) = 2 * Real.sqrt 3 / 9 ∧
    x * (Complex.abs (z_min - I)^2 - 1) = -2 * Real.sqrt 3 / 9 :=
by sorry

end complex_number_extrema_l1975_197510


namespace lcm_180_504_l1975_197594

theorem lcm_180_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end lcm_180_504_l1975_197594
