import Mathlib

namespace NUMINAMATH_CALUDE_salon_earnings_l4010_401067

/-- Calculates the total earnings from hair salon services -/
def total_earnings (haircut_price style_price coloring_price treatment_price : ℕ)
                   (haircuts styles colorings treatments : ℕ) : ℕ :=
  haircut_price * haircuts +
  style_price * styles +
  coloring_price * colorings +
  treatment_price * treatments

/-- Theorem stating that given specific prices and quantities, the total earnings are 871 -/
theorem salon_earnings :
  total_earnings 12 25 35 50 8 5 10 6 = 871 := by
  sorry

end NUMINAMATH_CALUDE_salon_earnings_l4010_401067


namespace NUMINAMATH_CALUDE_distance_minus_one_to_2023_l4010_401062

/-- The distance between two points on a number line -/
def distance (a b : ℝ) : ℝ := |b - a|

/-- Theorem: The distance between points representing -1 and 2023 on a number line is 2024 -/
theorem distance_minus_one_to_2023 : distance (-1) 2023 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_distance_minus_one_to_2023_l4010_401062


namespace NUMINAMATH_CALUDE_smallest_n_cube_and_square_l4010_401060

theorem smallest_n_cube_and_square : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 4 * n = a^3) ∧ 
  (∃ (b : ℕ), 5 * n = b^2) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (c : ℕ), 4 * m = c^3) → 
    (∃ (d : ℕ), 5 * m = d^2) → 
    m ≥ n) ∧
  n = 400 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_cube_and_square_l4010_401060


namespace NUMINAMATH_CALUDE_f_positive_iff_x_gt_one_l4010_401007

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (deriv f) x > f x)
variable (h2 : f 1 = 0)

-- State the theorem
theorem f_positive_iff_x_gt_one :
  (∀ x, f x > 0 ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_x_gt_one_l4010_401007


namespace NUMINAMATH_CALUDE_nikkas_stamp_collection_l4010_401025

theorem nikkas_stamp_collection :
  ∀ (total_stamps : ℕ) 
    (chinese_percentage : ℚ) 
    (us_percentage : ℚ) 
    (japanese_stamps : ℕ),
  chinese_percentage = 35 / 100 →
  us_percentage = 20 / 100 →
  japanese_stamps = 45 →
  (1 - chinese_percentage - us_percentage) * total_stamps = japanese_stamps →
  total_stamps = 100 := by
sorry

end NUMINAMATH_CALUDE_nikkas_stamp_collection_l4010_401025


namespace NUMINAMATH_CALUDE_log_ratio_sixteen_four_l4010_401008

theorem log_ratio_sixteen_four : (Real.log 16) / (Real.log 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_sixteen_four_l4010_401008


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l4010_401055

theorem geometric_series_r_value (a r : ℝ) (h1 : |r| < 1) : 
  (∑' n, a * r^n) = 15 ∧ (∑' n, a * r^(2*n)) = 6 → r = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l4010_401055


namespace NUMINAMATH_CALUDE_kennel_long_furred_dogs_l4010_401071

/-- The number of long-furred dogs in a kennel --/
def long_furred_dogs (total : ℕ) (brown : ℕ) (neither : ℕ) (long_furred_brown : ℕ) : ℕ :=
  total - neither - brown + long_furred_brown

/-- Theorem stating the number of long-furred dogs in the kennel --/
theorem kennel_long_furred_dogs :
  long_furred_dogs 45 17 8 9 = 29 := by
  sorry

#eval long_furred_dogs 45 17 8 9

end NUMINAMATH_CALUDE_kennel_long_furred_dogs_l4010_401071


namespace NUMINAMATH_CALUDE_no_nonzero_rational_solution_l4010_401081

theorem no_nonzero_rational_solution :
  ∀ (x y z : ℚ), x^3 + 3*y^3 + 9*z^3 = 9*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_rational_solution_l4010_401081


namespace NUMINAMATH_CALUDE_steak_cost_calculation_l4010_401040

/-- Calculate the total cost of steaks with a buy two get one free offer and a discount --/
theorem steak_cost_calculation (price_per_pound : ℝ) (pounds_bought : ℝ) (discount_rate : ℝ) : 
  price_per_pound = 15 →
  pounds_bought = 24 →
  discount_rate = 0.1 →
  (pounds_bought * price_per_pound) * (1 - discount_rate) = 324 := by
sorry

end NUMINAMATH_CALUDE_steak_cost_calculation_l4010_401040


namespace NUMINAMATH_CALUDE_diagonal_cells_in_rectangle_diagonal_cells_199_991_l4010_401038

theorem diagonal_cells_in_rectangle : ℕ → ℕ → ℕ
  | m, n => m + n - Nat.gcd m n

theorem diagonal_cells_199_991 :
  diagonal_cells_in_rectangle 199 991 = 1189 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cells_in_rectangle_diagonal_cells_199_991_l4010_401038


namespace NUMINAMATH_CALUDE_probability_theorem_l4010_401015

def total_candidates : ℕ := 9
def boys : ℕ := 5
def girls : ℕ := 4
def volunteers : ℕ := 4

def probability_1girl_3boys : ℚ := 20 / 63

def P (n : ℕ) : ℚ := 
  (Nat.choose boys n * Nat.choose girls (volunteers - n)) / Nat.choose total_candidates volunteers

theorem probability_theorem :
  (probability_1girl_3boys = P 3) ∧
  (∀ n : ℕ, P n ≥ 3/4 → n ≤ 2) ∧
  (P 2 ≥ 3/4) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l4010_401015


namespace NUMINAMATH_CALUDE_zhang_qiujian_suanjing_problem_l4010_401079

theorem zhang_qiujian_suanjing_problem (a : ℕ → ℚ) :
  (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
  a 1 + a 2 + a 3 = 4 →                         -- sum of first 3 terms
  a 8 + a 9 + a 10 = 3 →                        -- sum of last 3 terms
  a 5 + a 6 = 7/3 :=                            -- sum of 5th and 6th terms
by sorry

end NUMINAMATH_CALUDE_zhang_qiujian_suanjing_problem_l4010_401079


namespace NUMINAMATH_CALUDE_andrews_age_proof_l4010_401051

/-- Andrew's age in years -/
def andrew_age : ℝ := 7.875

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℝ := 9 * andrew_age

/-- Age difference between Andrew and his grandfather at Andrew's birth -/
def age_difference : ℝ := 63

theorem andrews_age_proof : 
  grandfather_age - andrew_age = age_difference ∧ 
  grandfather_age = 9 * andrew_age ∧ 
  andrew_age = 7.875 := by sorry

end NUMINAMATH_CALUDE_andrews_age_proof_l4010_401051


namespace NUMINAMATH_CALUDE_unique_integer_square_Q_l4010_401073

/-- Q is a function that maps an integer to an integer -/
def Q (x : ℤ) : ℤ := x^4 + 4*x^3 + 6*x^2 - x + 41

/-- There exists exactly one integer x such that Q(x) is a perfect square -/
theorem unique_integer_square_Q : ∃! x : ℤ, ∃ y : ℤ, Q x = y^2 := by sorry

end NUMINAMATH_CALUDE_unique_integer_square_Q_l4010_401073


namespace NUMINAMATH_CALUDE_correct_calculation_l4010_401003

theorem correct_calculation (x : ℤ) : x - 6 = 51 → 6 * x = 342 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4010_401003


namespace NUMINAMATH_CALUDE_sine_of_sum_inverse_sine_tangent_l4010_401056

theorem sine_of_sum_inverse_sine_tangent : 
  Real.sin (Real.arcsin (3/5) + Real.arctan 2) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_sum_inverse_sine_tangent_l4010_401056


namespace NUMINAMATH_CALUDE_line_slope_l4010_401058

/-- The slope of the line given by the equation x/4 + y/3 = 2 is -3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 + y / 3 = 2) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l4010_401058


namespace NUMINAMATH_CALUDE_archer_weekly_spending_is_1056_l4010_401006

/-- The archer's weekly spending on arrows -/
def archer_weekly_spending (shots_per_day : ℕ) (days_per_week : ℕ) 
  (recovery_rate : ℚ) (arrow_cost : ℚ) (team_payment_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let unrecovered_arrows := total_shots * (1 - recovery_rate)
  let total_cost := unrecovered_arrows * arrow_cost
  total_cost * (1 - team_payment_rate)

/-- Theorem: The archer spends $1056 on arrows per week -/
theorem archer_weekly_spending_is_1056 :
  archer_weekly_spending 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end NUMINAMATH_CALUDE_archer_weekly_spending_is_1056_l4010_401006


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l4010_401076

/-- The eccentricity of an ellipse with major axis length three times its minor axis length -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a = 3 * b) (h5 : a^2 = b^2 + c^2) : c / a = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l4010_401076


namespace NUMINAMATH_CALUDE_exactly_three_solutions_l4010_401054

-- Define the system of equations
def satisfies_system (a b c : ℤ) : Prop :=
  a * b + c = 17 ∧ a + b * c = 19

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset (ℤ × ℤ × ℤ)), 
    (∀ (x : ℤ × ℤ × ℤ), x ∈ s ↔ satisfies_system x.1 x.2.1 x.2.2) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_l4010_401054


namespace NUMINAMATH_CALUDE_sector_area_from_arc_and_angle_l4010_401096

/-- Given an arc length of 28 cm and a central angle of 240°, 
    the area of the sector is 294/π cm² -/
theorem sector_area_from_arc_and_angle 
  (arc_length : ℝ) 
  (central_angle : ℝ) 
  (h1 : arc_length = 28) 
  (h2 : central_angle = 240) : 
  (1/2) * arc_length * (arc_length / (central_angle * (π / 180))) = 294 / π :=
by sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_and_angle_l4010_401096


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4010_401083

theorem negation_of_universal_proposition (m : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4010_401083


namespace NUMINAMATH_CALUDE_point_2023_coordinates_l4010_401018

/-- Defines the x-coordinate of the nth point in the sequence -/
def x_coord (n : ℕ) : ℤ := 2 * n - 1

/-- Defines the y-coordinate of the nth point in the sequence -/
def y_coord (n : ℕ) : ℤ := (-1 : ℤ) ^ (n - 1) * 2 ^ n

/-- Theorem stating the coordinates of the 2023rd point -/
theorem point_2023_coordinates :
  (x_coord 2023, y_coord 2023) = (4045, 2 ^ 2023) := by
  sorry

end NUMINAMATH_CALUDE_point_2023_coordinates_l4010_401018


namespace NUMINAMATH_CALUDE_range_of_m_l4010_401095

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 :=
by
  sorry

-- Define the final result
def result : Set ℝ := {m | m ≥ 9}

end NUMINAMATH_CALUDE_range_of_m_l4010_401095


namespace NUMINAMATH_CALUDE_sara_height_l4010_401028

/-- Proves that Sara's height is 45 inches given the relative heights of Sara, Joe, Roy, Mark, and Julie. -/
theorem sara_height (
  julie_height : ℕ)
  (mark_taller_than_julie : ℕ)
  (roy_taller_than_mark : ℕ)
  (joe_taller_than_roy : ℕ)
  (sara_taller_than_joe : ℕ)
  (h_julie : julie_height = 33)
  (h_mark : mark_taller_than_julie = 1)
  (h_roy : roy_taller_than_mark = 2)
  (h_joe : joe_taller_than_roy = 3)
  (h_sara : sara_taller_than_joe = 6) :
  julie_height + mark_taller_than_julie + roy_taller_than_mark + joe_taller_than_roy + sara_taller_than_joe = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_height_l4010_401028


namespace NUMINAMATH_CALUDE_tan_315_degrees_l4010_401085

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l4010_401085


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_3_and_4_with_tens_greater_than_ones_l4010_401042

/-- A function that returns true if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The main theorem to be proved -/
theorem two_digit_divisible_by_3_and_4_with_tens_greater_than_ones :
  ∃! (s : Finset ℕ), 
    s.card = 4 ∧ 
    (∀ n ∈ s, 
      n > 0 ∧ 
      is_two_digit n ∧ 
      n % 3 = 0 ∧ 
      n % 4 = 0 ∧ 
      tens_digit n > ones_digit n) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_3_and_4_with_tens_greater_than_ones_l4010_401042


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4010_401064

def A : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
def B : Set (ℝ × ℝ) := {p | 2 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {(2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4010_401064


namespace NUMINAMATH_CALUDE_raghu_investment_l4010_401057

theorem raghu_investment
  (vishal_investment : ℝ)
  (trishul_investment : ℝ)
  (raghu_investment : ℝ)
  (vishal_more_than_trishul : vishal_investment = 1.1 * trishul_investment)
  (trishul_less_than_raghu : trishul_investment = 0.9 * raghu_investment)
  (total_investment : vishal_investment + trishul_investment + raghu_investment = 6936) :
  raghu_investment = 2400 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l4010_401057


namespace NUMINAMATH_CALUDE_eight_books_three_piles_l4010_401098

/-- The number of ways to divide n identical objects into k non-empty groups -/
def divide_objects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to divide 8 identical books into 3 piles -/
theorem eight_books_three_piles : divide_objects 8 3 = 5 := by sorry

end NUMINAMATH_CALUDE_eight_books_three_piles_l4010_401098


namespace NUMINAMATH_CALUDE_middle_dimension_at_least_six_l4010_401072

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : Real
  width : Real
  height : Real

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : Real
  height : Real

/-- Checks if a cylinder fits upright in a crate -/
def cylinderFitsUpright (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.length)

theorem middle_dimension_at_least_six 
  (crate : CrateDimensions)
  (h1 : crate.length = 3)
  (h2 : crate.height = 12)
  (h3 : cylinderFitsUpright crate { radius := 3, height := 12 }) :
  crate.width ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_middle_dimension_at_least_six_l4010_401072


namespace NUMINAMATH_CALUDE_find_a_and_b_l4010_401052

theorem find_a_and_b (a b d : ℤ) : 
  (∃ x : ℝ, Real.sqrt (x - a) + Real.sqrt (x + b) = 7 ∧ x = 12) →
  (∃ x : ℝ, Real.sqrt (x + a) + Real.sqrt (x + d) = 7 ∧ x = 13) →
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_find_a_and_b_l4010_401052


namespace NUMINAMATH_CALUDE_smallest_beneficial_discount_l4010_401059

theorem smallest_beneficial_discount : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (1 - m / 100) > (1 - 20 / 100) * (1 - 20 / 100) ∨
    (1 - m / 100) > (1 - 10 / 100) * (1 - 15 / 100) ∨
    (1 - m / 100) > (1 - 8 / 100) * (1 - 8 / 100) * (1 - 8 / 100)) ∧
  (1 - n / 100) ≤ (1 - 20 / 100) * (1 - 20 / 100) ∧
  (1 - n / 100) ≤ (1 - 10 / 100) * (1 - 15 / 100) ∧
  (1 - n / 100) ≤ (1 - 8 / 100) * (1 - 8 / 100) * (1 - 8 / 100) ∧
  n = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_beneficial_discount_l4010_401059


namespace NUMINAMATH_CALUDE_sandbox_sand_weight_l4010_401077

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.length * r.width

/-- Calculates the total area of two rectangles -/
def totalArea (r1 r2 : Rectangle) : ℕ := rectangleArea r1 + rectangleArea r2

/-- Calculates the number of bags needed to fill an area -/
def bagsNeeded (area : ℕ) (areaPerBag : ℕ) : ℕ := (area + areaPerBag - 1) / areaPerBag

/-- Theorem: The total weight of sand needed to fill the sandbox -/
theorem sandbox_sand_weight :
  let rectangle1 : Rectangle := ⟨50, 30⟩
  let rectangle2 : Rectangle := ⟨20, 15⟩
  let areaPerBag : ℕ := 80
  let weightPerBag : ℕ := 30
  let totalSandboxArea : ℕ := totalArea rectangle1 rectangle2
  let bags : ℕ := bagsNeeded totalSandboxArea areaPerBag
  bags * weightPerBag = 690 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_sand_weight_l4010_401077


namespace NUMINAMATH_CALUDE_certain_number_threshold_l4010_401094

theorem certain_number_threshold (k : ℤ) : 0.0010101 * (10 : ℝ)^(k : ℝ) > 10.101 → k ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_threshold_l4010_401094


namespace NUMINAMATH_CALUDE_smallest_divisible_by_5_13_7_l4010_401012

theorem smallest_divisible_by_5_13_7 : ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n → n ≥ 455 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_5_13_7_l4010_401012


namespace NUMINAMATH_CALUDE_fraction_value_l4010_401087

theorem fraction_value : (2 * 3 + 4) / (2 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l4010_401087


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l4010_401069

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetric_points_sum (x y : ℝ) :
  symmetric_wrt_origin (x, -2) (3, y) → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l4010_401069


namespace NUMINAMATH_CALUDE_M_equals_N_l4010_401065

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | 1/x < 1}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l4010_401065


namespace NUMINAMATH_CALUDE_trig_inequality_l4010_401031

theorem trig_inequality (x : ℝ) : 1 ≤ Real.sin x ^ 10 + 10 * Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 10 ∧ 
  Real.sin x ^ 10 + 10 * Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 10 ≤ 41 / 16 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l4010_401031


namespace NUMINAMATH_CALUDE_grade_assignments_count_l4010_401009

/-- The number of possible grades a professor can assign to each student. -/
def num_grades : ℕ := 4

/-- The number of students in the class. -/
def num_students : ℕ := 15

/-- The number of ways to assign grades to all students in the class. -/
def num_grade_assignments : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 4^15. -/
theorem grade_assignments_count :
  num_grade_assignments = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_count_l4010_401009


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l4010_401011

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let circle_radius : ℝ := 8
  let rectangle_area := rectangle_length * rectangle_width
  let circle_area := π * circle_radius^2
  let overlap_area := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 48 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l4010_401011


namespace NUMINAMATH_CALUDE_inequality_proof_l4010_401036

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4010_401036


namespace NUMINAMATH_CALUDE_not_prime_expression_l4010_401099

theorem not_prime_expression (n : ℕ) (h : n > 2) :
  ¬ Nat.Prime (n^(n^n) - 6*n^n + 5) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_expression_l4010_401099


namespace NUMINAMATH_CALUDE_evaluate_expression_l4010_401093

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 1) :
  z * (z - 4 * x) = -15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4010_401093


namespace NUMINAMATH_CALUDE_problem_solution_l4010_401032

-- Define the set of possible values
def S : Set ℕ := {0, 1, 3}

-- Define the properties
def prop1 (a b c : ℕ) : Prop := a ≠ 3
def prop2 (a b c : ℕ) : Prop := b = 3
def prop3 (a b c : ℕ) : Prop := c ≠ 0

theorem problem_solution (a b c : ℕ) :
  {a, b, c} = S →
  (prop1 a b c ∨ prop2 a b c ∨ prop3 a b c) →
  (¬(prop1 a b c ∧ prop2 a b c) ∧ ¬(prop1 a b c ∧ prop3 a b c) ∧ ¬(prop2 a b c ∧ prop3 a b c)) →
  100 * a + 10 * b + c = 301 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4010_401032


namespace NUMINAMATH_CALUDE_triangle_longest_side_l4010_401048

/-- Given a triangle with side lengths 10, y+5, and 3y-2, and a perimeter of 50,
    prove that the longest side length is 25.75. -/
theorem triangle_longest_side (y : ℝ) : 
  10 + (y + 5) + (3 * y - 2) = 50 →
  max 10 (max (y + 5) (3 * y - 2)) = 25.75 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l4010_401048


namespace NUMINAMATH_CALUDE_worker_count_l4010_401044

theorem worker_count (work_amount : ℝ) : ∃ (workers : ℕ), 
  (workers : ℝ) * 75 = work_amount ∧ 
  (workers + 10 : ℝ) * 65 = work_amount ∧ 
  workers = 65 := by
sorry

end NUMINAMATH_CALUDE_worker_count_l4010_401044


namespace NUMINAMATH_CALUDE_drums_per_day_l4010_401020

/-- Given that 266 pickers fill 90 drums in 5 days, prove that the number of drums filled per day is 18. -/
theorem drums_per_day (pickers : ℕ) (total_drums : ℕ) (days : ℕ) 
  (h1 : pickers = 266) 
  (h2 : total_drums = 90) 
  (h3 : days = 5) : 
  total_drums / days = 18 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l4010_401020


namespace NUMINAMATH_CALUDE_lao_you_fen_max_profit_l4010_401066

/-- Represents the cost and quantity information for Lao You Fen brands -/
structure LaoYouFen where
  cost_a : ℝ
  cost_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Calculates the profit given the quantities of each brand -/
def profit (l : LaoYouFen) (qa qb : ℝ) : ℝ :=
  (13 - l.cost_a) * qa + (13 - l.cost_b) * qb

/-- Theorem stating the maximum profit for Lao You Fen sales -/
theorem lao_you_fen_max_profit (l : LaoYouFen) :
  l.cost_b = l.cost_a + 2 →
  2700 / l.cost_a = 3300 / l.cost_b →
  l.quantity_a + l.quantity_b = 800 →
  l.quantity_a ≤ 3 * l.quantity_b →
  (∀ qa qb : ℝ, qa + qb = 800 → qa ≤ 3 * qb → profit l qa qb ≤ 2800) ∧
  profit l 600 200 = 2800 :=
sorry

end NUMINAMATH_CALUDE_lao_you_fen_max_profit_l4010_401066


namespace NUMINAMATH_CALUDE_license_plate_count_l4010_401061

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of license plates with the given conditions -/
def num_license_plates : ℕ := num_letters^3 * num_odd_digits^2 * num_even_digits

theorem license_plate_count : num_license_plates = 2197000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l4010_401061


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l4010_401033

/-- Represents a rectangular piece of paper --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of the rectangle when folded along its width --/
def perimeterFoldedWidth (r : Rectangle) : ℝ := 2 * r.length + r.width

/-- The perimeter of the rectangle when folded along its length --/
def perimeterFoldedLength (r : Rectangle) : ℝ := 2 * r.width + r.length

/-- The area of the rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem folded_rectangle_perimeter 
  (r : Rectangle) 
  (h1 : area r = 140)
  (h2 : perimeterFoldedWidth r = 34) :
  perimeterFoldedLength r = 38 := by
  sorry

#check folded_rectangle_perimeter

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l4010_401033


namespace NUMINAMATH_CALUDE_cubic_root_magnitude_l4010_401013

theorem cubic_root_magnitude (q : ℝ) (r₁ r₂ r₃ : ℝ) : 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ →
  r₁^3 + q*r₁^2 + 6*r₁ + 9 = 0 →
  r₂^3 + q*r₂^2 + 6*r₂ + 9 = 0 →
  r₃^3 + q*r₃^2 + 6*r₃ + 9 = 0 →
  (q^2 * 6^2 - 4 * 6^3 - 4*q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9)) ≠ 0 →
  max (|r₁|) (max (|r₂|) (|r₃|)) > 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_magnitude_l4010_401013


namespace NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l4010_401016

/-- The probability that x + y ≤ 6 when (x, y) is randomly selected from a rectangle where 0 ≤ x ≤ 4 and 0 ≤ y ≤ 5 -/
theorem probability_x_plus_y_leq_6 :
  let rectangle_area : ℝ := 4 * 5
  let favorable_area : ℝ := 15
  favorable_area / rectangle_area = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l4010_401016


namespace NUMINAMATH_CALUDE_eight_digit_factorization_comparison_l4010_401037

theorem eight_digit_factorization_comparison :
  let total_eight_digit_numbers := 99999999 - 10000000 + 1
  let four_digit_numbers := 9999 - 1000 + 1
  let products_of_four_digit_numbers := four_digit_numbers.choose 2 + four_digit_numbers
  total_eight_digit_numbers - products_of_four_digit_numbers > products_of_four_digit_numbers := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_factorization_comparison_l4010_401037


namespace NUMINAMATH_CALUDE_fourth_side_length_l4010_401082

/-- A quadrilateral inscribed in a circle with radius 150√2, where three sides have lengths 150, 150, and 150√3 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first side -/
  side1 : ℝ
  /-- The length of the second side -/
  side2 : ℝ
  /-- The length of the third side -/
  side3 : ℝ
  /-- The length of the fourth side -/
  side4 : ℝ
  /-- The radius is 150√2 -/
  radius_eq : radius = 150 * Real.sqrt 2
  /-- The first side has length 150 -/
  side1_eq : side1 = 150
  /-- The second side has length 150 -/
  side2_eq : side2 = 150
  /-- The third side has length 150√3 -/
  side3_eq : side3 = 150 * Real.sqrt 3

/-- The theorem stating that the fourth side has length 150√7 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.side4 = 150 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l4010_401082


namespace NUMINAMATH_CALUDE_number_problem_l4010_401023

theorem number_problem : ∃ x : ℝ, x^2 + 75 = (x - 20)^2 ∧ x = 8.125 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4010_401023


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4010_401049

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4010_401049


namespace NUMINAMATH_CALUDE_shortest_path_on_sphere_intersection_l4010_401027

/-- The shortest path on a sphere's surface between the two most distant points of its intersection with a plane --/
theorem shortest_path_on_sphere_intersection (R d : ℝ) (h1 : R = 2) (h2 : d = 1) :
  let r := Real.sqrt (R^2 - d^2)
  let θ := 2 * Real.arccos (d / R)
  θ / (2 * Real.pi) * (2 * Real.pi * r) = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_on_sphere_intersection_l4010_401027


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l4010_401050

/-- The area of the region outside a circle of radius 2 and inside two circles of radius 4 
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem shaded_area_calculation : 
  ∃ (r₁ r₂ : ℝ) (A B : ℝ × ℝ),
    r₁ = 2 ∧ 
    r₂ = 4 ∧
    A.1^2 + A.2^2 = r₁^2 ∧
    B.1^2 + B.2^2 = r₁^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * r₁)^2 →
    let shaded_area := 
      2 * (π * r₂^2 / 6 - r₁ * (r₂^2 - r₁^2).sqrt / 2 - π * r₁^2 / 4)
    shaded_area = (20 / 3) * π - 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l4010_401050


namespace NUMINAMATH_CALUDE_epidemic_test_analysis_l4010_401063

/-- Represents a class of students with their test scores and statistics -/
structure ClassData where
  scores : List Nat
  frequency_table : List (Nat × Nat)
  mean : Nat
  mode : Nat
  median : Nat
  variance : Float

/-- The data for the entire school -/
structure SchoolData where
  total_students : Nat
  class_a : ClassData
  class_b : ClassData

/-- Definition of excellent performance -/
def excellent_score : Nat := 90

/-- The given school data -/
def school_data : SchoolData := {
  total_students := 600,
  class_a := {
    scores := [78, 83, 89, 97, 98, 85, 100, 94, 87, 90, 93, 92, 99, 95, 100],
    frequency_table := [(1, 75), (1, 80), (3, 85), (4, 90), (6, 95)],
    mean := 92,
    mode := 100,
    median := 93,
    variance := 41.07
  },
  class_b := {
    scores := [91, 92, 94, 90, 93],
    frequency_table := [(1, 75), (2, 80), (3, 85), (5, 90), (4, 95)],
    mean := 90,
    mode := 87,
    median := 91,
    variance := 50.2
  }
}

theorem epidemic_test_analysis (data : SchoolData := school_data) :
  (data.class_a.mode = 100) ∧
  (data.class_b.median = 91) ∧
  (((data.class_a.frequency_table.filter (λ x => x.2 ≥ 90)).map (λ x => x.1)).sum +
   ((data.class_b.frequency_table.filter (λ x => x.2 ≥ 90)).map (λ x => x.1)).sum) * 20 = 380 ∧
  (data.class_a.mean > data.class_b.mean ∧ data.class_a.variance < data.class_b.variance) := by
  sorry

end NUMINAMATH_CALUDE_epidemic_test_analysis_l4010_401063


namespace NUMINAMATH_CALUDE_last_score_is_90_l4010_401043

def scores : List Nat := [72, 77, 85, 90, 94]

def isValidOrder (order : List Nat) : Prop :=
  order.length = 5 ∧
  order.toFinset = scores.toFinset ∧
  ∀ k : Fin 5, (order.take k.val.succ).sum % k.val.succ = 0

theorem last_score_is_90 :
  ∀ order : List Nat, isValidOrder order → order.getLast? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_last_score_is_90_l4010_401043


namespace NUMINAMATH_CALUDE_plumber_distribution_l4010_401024

/-- The number of ways to distribute n plumbers to k residences,
    where all plumbers are assigned, each plumber goes to only one residence,
    and each residence has at least one plumber. -/
def distributionSchemes (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 plumbers to 4 residences
    results in 240 different distribution schemes. -/
theorem plumber_distribution :
  distributionSchemes 5 4 = 240 := by sorry

end NUMINAMATH_CALUDE_plumber_distribution_l4010_401024


namespace NUMINAMATH_CALUDE_sum_of_squares_l4010_401090

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4010_401090


namespace NUMINAMATH_CALUDE_electronics_not_all_on_sale_l4010_401092

-- Define the universe of discourse
variable (E : Type) [Nonempty E]

-- Define the predicate for "on sale"
variable (on_sale : E → Prop)

-- Define the store
variable (store : Set E)

-- Assume the store is not empty
variable (h_store_nonempty : store.Nonempty)

-- The main theorem
theorem electronics_not_all_on_sale
  (h : ¬∀ (e : E), e ∈ store → on_sale e) :
  (∃ (e : E), e ∈ store ∧ ¬on_sale e) ∧
  (¬∀ (e : E), e ∈ store → on_sale e) :=
by sorry


end NUMINAMATH_CALUDE_electronics_not_all_on_sale_l4010_401092


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l4010_401097

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle (in degrees) for rotational symmetry -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ (p : RegularPolygon 17),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l4010_401097


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1540_l4010_401014

theorem sum_of_extreme_prime_factors_of_1540 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ 
    largest.Prime ∧
    smallest ∣ 1540 ∧ 
    largest ∣ 1540 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1540 → p ≥ smallest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1540 → p ≤ largest) ∧
    smallest + largest = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1540_l4010_401014


namespace NUMINAMATH_CALUDE_seat_representation_l4010_401046

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (column : ℕ)

/-- The notation for representing seats in the movie theater -/
def seat_notation (r : ℕ) (c : ℕ) : Seat := ⟨r, c⟩

/-- Theorem stating that if (5, 2) represents the seat in the 5th row and 2nd column,
    then (7, 3) represents the seat in the 7th row and 3rd column -/
theorem seat_representation :
  (seat_notation 5 2 = ⟨5, 2⟩) →
  (seat_notation 7 3 = ⟨7, 3⟩) :=
by sorry

end NUMINAMATH_CALUDE_seat_representation_l4010_401046


namespace NUMINAMATH_CALUDE_final_value_exceeds_initial_l4010_401029

theorem final_value_exceeds_initial (p q r M : ℝ) 
  (hp : p > 0) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) * (1 + r / 100) > M ↔ 
  p > (100 * (q - r + q * r / 100)) / (100 - q + r + q * r / 100) :=
by sorry

end NUMINAMATH_CALUDE_final_value_exceeds_initial_l4010_401029


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l4010_401068

/-- Definition of the sequence a_n -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

/-- Main theorem: 2^k divides a_n if and only if 2^k divides n -/
theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ a n ↔ 2^k ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l4010_401068


namespace NUMINAMATH_CALUDE_locus_of_symmetric_point_l4010_401091

/-- Given a parabola y = x^2, a fixed point A(a, 0) where a ≠ 0, and a moving point P on the parabola,
    the point Q symmetric to A with respect to P has the locus y = (1/2)(x + a)^2 -/
theorem locus_of_symmetric_point (a : ℝ) (ha : a ≠ 0) :
  ∀ x₁ y₁ x y : ℝ,
  y₁ = x₁^2 →                        -- P(x₁, y₁) is on the parabola y = x^2
  x = 2*a - x₁ →                     -- x-coordinate of Q
  y = -y₁ →                          -- y-coordinate of Q
  y = (1/2) * (x + a)^2 := by sorry

end NUMINAMATH_CALUDE_locus_of_symmetric_point_l4010_401091


namespace NUMINAMATH_CALUDE_composition_equation_solution_l4010_401053

/-- Given two functions f and g, and a condition on their composition, prove the value of b. -/
theorem composition_equation_solution (f g : ℝ → ℝ) (b : ℝ) 
  (hf : ∀ x, f x = 3 * x - 2)
  (hg : ∀ x, g x = 7 - 2 * x)
  (h_comp : g (f b) = 1) : 
  b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l4010_401053


namespace NUMINAMATH_CALUDE_existence_of_greater_indices_l4010_401001

theorem existence_of_greater_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end NUMINAMATH_CALUDE_existence_of_greater_indices_l4010_401001


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l4010_401039

/-- The value of 'a' for a hyperbola with equation x²/a² - y² = 1, a > 0, and eccentricity √5 -/
theorem hyperbola_a_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1) 
  (h3 : ∃ c : ℝ, c / a = Real.sqrt 5) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l4010_401039


namespace NUMINAMATH_CALUDE_last_three_digits_sum_sum_of_last_three_digits_l4010_401047

theorem last_three_digits_sum (C : ℕ) : ∃ (k : ℕ), 7^(4+C) = 1000 * k + 601 := by sorry

theorem sum_of_last_three_digits (C : ℕ) : (6 + 0 + 1 : ℕ) = 7 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_sum_sum_of_last_three_digits_l4010_401047


namespace NUMINAMATH_CALUDE_road_length_theorem_l4010_401089

/-- Represents the distance between two markers on the road. -/
structure MarkerDistance where
  fromA : ℕ
  fromB : ℕ

/-- The road between cities A and B -/
structure Road where
  length : ℕ
  marker1 : MarkerDistance
  marker2 : MarkerDistance

/-- Conditions for a valid road configuration -/
def isValidRoad (r : Road) : Prop :=
  (r.marker1.fromA + r.marker1.fromB = r.length) ∧
  (r.marker2.fromA + r.marker2.fromB = r.length) ∧
  (r.marker2.fromA = r.marker1.fromA + 10) ∧
  ((r.marker1.fromA = 2 * r.marker1.fromB ∨ r.marker1.fromB = 2 * r.marker1.fromA) ∧
   (r.marker2.fromA = 3 * r.marker2.fromB ∨ r.marker2.fromB = 3 * r.marker2.fromA))

theorem road_length_theorem :
  ∀ r : Road, isValidRoad r → (r.length = 120 ∨ r.length = 24) ∧
  (∀ d : ℕ, d ≠ 120 ∧ d ≠ 24 → ¬∃ r' : Road, r'.length = d ∧ isValidRoad r') :=
by sorry

end NUMINAMATH_CALUDE_road_length_theorem_l4010_401089


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l4010_401005

/-- Given a line y = ax + b passing through points (0, 2) and (-3, 0),
    prove that the solution to ax + b = 0 is x = -3. -/
theorem line_intersection_x_axis 
  (a b : ℝ) 
  (h1 : 2 = a * 0 + b) 
  (h2 : 0 = a * (-3) + b) : 
  ∀ x, a * x + b = 0 ↔ x = -3 :=
sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l4010_401005


namespace NUMINAMATH_CALUDE_vector_sum_length_l4010_401075

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_length (a b : ℝ × ℝ) : 
  angle_between a b = π / 3 →
  a = (3, -4) →
  Real.sqrt ((a.1)^2 + (a.2)^2) = 2 →
  Real.sqrt (((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2)) = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_length_l4010_401075


namespace NUMINAMATH_CALUDE_sqrt_n_factorial_inequality_l4010_401074

theorem sqrt_n_factorial_inequality (n : ℕ) (hn : n > 0) :
  Real.sqrt n < (n.factorial : ℝ) ^ (1 / n : ℝ) ∧ (n.factorial : ℝ) ^ (1 / n : ℝ) < (n + 1 : ℝ) / 2 := by
  sorry

#check sqrt_n_factorial_inequality

end NUMINAMATH_CALUDE_sqrt_n_factorial_inequality_l4010_401074


namespace NUMINAMATH_CALUDE_prob_sum_five_l4010_401004

/-- A uniformly dense cubic die -/
structure Die :=
  (faces : Fin 6)

/-- The result of throwing a die twice -/
def TwoThrows := Die × Die

/-- The sum of points from two throws -/
def sum_points (t : TwoThrows) : ℕ :=
  t.1.faces.val + 1 + t.2.faces.val + 1

/-- The set of all possible outcomes when throwing a die twice -/
def all_outcomes : Finset TwoThrows :=
  sorry

/-- The set of outcomes where the sum of points is 5 -/
def sum_five : Finset TwoThrows :=
  sorry

/-- The probability of an event occurring when throwing a die twice -/
def prob (event : Finset TwoThrows) : ℚ :=
  (event.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_sum_five :
  prob sum_five = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_five_l4010_401004


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l4010_401010

theorem smallest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 > 1 ↔ n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l4010_401010


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l4010_401086

theorem recurring_decimal_to_fraction : (6 / 10 : ℚ) + (23 / 99 : ℚ) = 412 / 495 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l4010_401086


namespace NUMINAMATH_CALUDE_positive_solution_x_l4010_401030

theorem positive_solution_x (x y z : ℝ) : 
  x > 0 →
  x * y + 3 * x + 4 * y + 10 = 30 →
  y * z + 4 * y + 2 * z + 8 = 6 →
  x * z + 4 * x + 3 * z + 12 = 30 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l4010_401030


namespace NUMINAMATH_CALUDE_y_derivative_l4010_401084

noncomputable def y (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l4010_401084


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l4010_401026

/-- Represents the position function of a particle -/
def S (t : ℝ) : ℝ := 2 * t^3

/-- Represents the velocity function of a particle -/
def V (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_velocity_at_3 :
  V 3 = 54 :=
sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l4010_401026


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l4010_401002

theorem cupcakes_per_package 
  (initial_cupcakes : ℕ) 
  (eaten_cupcakes : ℕ) 
  (num_packages : ℕ) 
  (h1 : initial_cupcakes = 20)
  (h2 : eaten_cupcakes = 11)
  (h3 : num_packages = 3)
  (h4 : eaten_cupcakes < initial_cupcakes) :
  (initial_cupcakes - eaten_cupcakes) / num_packages = 3 :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l4010_401002


namespace NUMINAMATH_CALUDE_function_equality_l4010_401035

theorem function_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (f x + f y)) = f x + y) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_l4010_401035


namespace NUMINAMATH_CALUDE_min_sum_given_product_l4010_401000

theorem min_sum_given_product (x y : ℤ) (h : x * y = 144) : 
  ∀ a b : ℤ, a * b = 144 → x + y ≤ a + b ∧ ∃ c d : ℤ, c * d = 144 ∧ c + d = -145 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l4010_401000


namespace NUMINAMATH_CALUDE_product_of_roots_l4010_401017

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 125 = 0) → 
  (∃ a b c : ℝ, x^3 - 15*x^2 + 75*x - 125 = (x - a) * (x - b) * (x - c) ∧ a * b * c = 125) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l4010_401017


namespace NUMINAMATH_CALUDE_range_of_a_l4010_401080

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1

def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∃ x₁ > 0, ∃ x₂, f a x₁ ≥ g a x₂) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4010_401080


namespace NUMINAMATH_CALUDE_fraction_of_a_equal_half_b_l4010_401022

/-- Given two amounts a and b, where their sum is 1210 and b is 484,
    prove that the fraction of a's amount equal to half of b's amount is 1/3 -/
theorem fraction_of_a_equal_half_b (a b : ℕ) : 
  a + b = 1210 → b = 484 → ∃ f : ℚ, f * a = (1 / 2) * b ∧ f = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_a_equal_half_b_l4010_401022


namespace NUMINAMATH_CALUDE_division_problem_l4010_401021

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 199 →
  divisor = 18 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4010_401021


namespace NUMINAMATH_CALUDE_sqrt_three_not_in_P_l4010_401045

-- Define the set P
def P : Set ℝ := {x | x^2 - Real.sqrt 2 * x ≤ 0}

-- State the theorem
theorem sqrt_three_not_in_P : Real.sqrt 3 ∉ P := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_not_in_P_l4010_401045


namespace NUMINAMATH_CALUDE_martha_juice_bottles_l4010_401034

/-- Calculates the number of juice bottles left after a week -/
def bottles_left (initial_refrigerator : ℕ) (initial_pantry : ℕ) (bought : ℕ) (drunk : ℕ) : ℕ :=
  initial_refrigerator + initial_pantry + bought - drunk

/-- Proves that given the initial conditions and actions, 10 bottles are left -/
theorem martha_juice_bottles : bottles_left 4 4 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_martha_juice_bottles_l4010_401034


namespace NUMINAMATH_CALUDE_quadratic_function_coefficients_l4010_401088

/-- Given a quadratic function f(x) = 2(x-3)^2 + 2, prove that it can be expressed
    as ax^2 + bx + c where a = 2, b = -12, and c = 20 -/
theorem quadratic_function_coefficients :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = 2 * (x - 3)^2 + 2) ∧
    (∃ (a b c : ℝ), a = 2 ∧ b = -12 ∧ c = 20 ∧ 
      ∀ x, f x = a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficients_l4010_401088


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4010_401078

theorem complex_modulus_problem (z : ℂ) (h : (z - 2*Complex.I)*(1 - Complex.I) = -2) :
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4010_401078


namespace NUMINAMATH_CALUDE_third_month_sale_l4010_401041

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_4 : ℕ := 7230
def sales_5 : ℕ := 6562
def sales_6 : ℕ := 7991
def average_sale : ℕ := 7000
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ (sales_3 : ℕ),
    sales_3 = num_months * average_sale - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l4010_401041


namespace NUMINAMATH_CALUDE_even_function_theorem_l4010_401070

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function -/
def Domain (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x, x ∈ s ↔ ∃ y, f x = y

theorem even_function_theorem (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + 3 * a + b
  IsEven f ∧ Domain f (Set.Icc (a - 1) (2 * a)) →
  a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_theorem_l4010_401070


namespace NUMINAMATH_CALUDE_tims_manicure_cost_l4010_401019

/-- The total cost of a manicure with tip -/
def total_cost (base_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  base_cost * (1 + tip_percentage)

/-- Theorem: Tim's total payment for a $30 manicure with a 30% tip is $39 -/
theorem tims_manicure_cost :
  total_cost 30 0.3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tims_manicure_cost_l4010_401019
