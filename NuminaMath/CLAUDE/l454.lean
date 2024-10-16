import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_difference_l454_45493

/-- Given a quadratic equation 5x^2 - 11x - 14 = 0, prove that the positive difference
    between its roots is √401/5 and that p + q = 406 --/
theorem quadratic_root_difference (x : ℝ) : 
  let a : ℝ := 5
  let b : ℝ := -11
  let c : ℝ := -14
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  let difference := |root1 - root2|
  difference = Real.sqrt 401 / 5 ∧ 401 + 5 = 406 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l454_45493


namespace NUMINAMATH_CALUDE_sum_of_ages_l454_45418

/-- Given the present ages of Henry and Jill, prove that their sum is 48 years. -/
theorem sum_of_ages (henry_age jill_age : ℕ) 
  (henry_present : henry_age = 29)
  (jill_present : jill_age = 19)
  (past_relation : henry_age - 9 = 2 * (jill_age - 9)) :
  henry_age + jill_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l454_45418


namespace NUMINAMATH_CALUDE_platform_length_l454_45404

/-- Calculates the length of a platform given train speed and crossing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- Train speed in kmph
  (h2 : platform_crossing_time = 30)  -- Time to cross platform in seconds
  (h3 : man_crossing_time = 15)  -- Time to cross man in seconds
  : ∃ (platform_length : ℝ), platform_length = 300 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l454_45404


namespace NUMINAMATH_CALUDE_expression_equals_x_plus_one_l454_45424

theorem expression_equals_x_plus_one (x : ℝ) (h : x ≠ -1) :
  (x^2 + 2*x + 1) / (x + 1) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_x_plus_one_l454_45424


namespace NUMINAMATH_CALUDE_largest_a_is_four_l454_45426

/-- The largest coefficient of x^4 in a polynomial that satisfies the given conditions -/
noncomputable def largest_a : ℝ := 4

/-- A polynomial of degree 4 with real coefficients -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The condition that the polynomial is between 0 and 1 on [-1, 1] -/
def satisfies_condition (a b c d e : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 0 ≤ polynomial a b c d e x ∧ polynomial a b c d e x ≤ 1

/-- The theorem stating that 4 is the largest possible value for a -/
theorem largest_a_is_four :
  (∃ b c d e : ℝ, satisfies_condition largest_a b c d e) ∧
  (∀ a : ℝ, a > largest_a → ¬∃ b c d e : ℝ, satisfies_condition a b c d e) :=
sorry

end NUMINAMATH_CALUDE_largest_a_is_four_l454_45426


namespace NUMINAMATH_CALUDE_ray_nickels_left_l454_45467

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Define Ray's initial amount in cents
def initial_amount : ℕ := 95

-- Define the amount given to Peter in cents
def amount_to_peter : ℕ := 25

-- Theorem stating that Ray will have 4 nickels left
theorem ray_nickels_left : 
  let amount_to_randi := 2 * amount_to_peter
  let total_given := amount_to_peter + amount_to_randi
  let remaining_cents := initial_amount - total_given
  remaining_cents / nickel_value = 4 := by
sorry

end NUMINAMATH_CALUDE_ray_nickels_left_l454_45467


namespace NUMINAMATH_CALUDE_sum_three_numbers_l454_45413

theorem sum_three_numbers (a b c M : ℤ) : 
  a + b + c = 75 ∧ 
  a + 4 = M ∧ 
  b - 5 = M ∧ 
  3 * c = M → 
  M = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l454_45413


namespace NUMINAMATH_CALUDE_age_multiplier_problem_l454_45478

theorem age_multiplier_problem (A : ℕ) (N : ℚ) : 
  A = 50 → (A + 5) * N - 5 * (A - 5) = A → N = 5 := by
sorry

end NUMINAMATH_CALUDE_age_multiplier_problem_l454_45478


namespace NUMINAMATH_CALUDE_f_10_equals_144_l454_45486

def f : ℕ → ℕ
  | 0 => 0  -- define f(0) as 0 for completeness
  | 1 => 2
  | 2 => 3
  | (n + 3) => f (n + 2) + f (n + 1)

theorem f_10_equals_144 : f 10 = 144 := by sorry

end NUMINAMATH_CALUDE_f_10_equals_144_l454_45486


namespace NUMINAMATH_CALUDE_fencing_cost_l454_45454

-- Define the ratio of sides
def ratio_length : ℚ := 3
def ratio_width : ℚ := 4

-- Define the area of the field
def area : ℚ := 8112

-- Define the cost per meter in rupees
def cost_per_meter : ℚ := 25 / 100

-- Theorem statement
theorem fencing_cost :
  let x : ℚ := (area / (ratio_length * ratio_width)) ^ (1/2)
  let length : ℚ := ratio_length * x
  let width : ℚ := ratio_width * x
  let perimeter : ℚ := 2 * (length + width)
  let total_cost : ℚ := perimeter * cost_per_meter
  total_cost = 91 := by sorry

end NUMINAMATH_CALUDE_fencing_cost_l454_45454


namespace NUMINAMATH_CALUDE_small_boxes_count_l454_45431

theorem small_boxes_count (total_bars : ℕ) (bars_per_box : ℕ) (h1 : total_bars = 375) (h2 : bars_per_box = 25) :
  total_bars / bars_per_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l454_45431


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l454_45472

/-- Proves that the line y - 2 = mx + m passes through the point (-1, 2) for any real m -/
theorem line_passes_through_fixed_point (m : ℝ) : 
  2 - 2 = m * (-1) + m := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l454_45472


namespace NUMINAMATH_CALUDE_box_height_proof_l454_45462

theorem box_height_proof (h w l : ℝ) (volume : ℝ) : 
  l = 3 * h →
  l = 4 * w →
  volume = l * w * h →
  volume = 3888 →
  h = 12 := by
sorry

end NUMINAMATH_CALUDE_box_height_proof_l454_45462


namespace NUMINAMATH_CALUDE_ampersand_composition_l454_45453

-- Define the & operation for the case &=9-x
def ampersand1 (x : ℤ) : ℤ := 9 - x

-- Define the & operation for the case &x = x - 9
def ampersand2 (x : ℤ) : ℤ := x - 9

-- Theorem to prove
theorem ampersand_composition : ampersand1 (ampersand2 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l454_45453


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l454_45496

-- Define the ratio of the sides
def side_ratio : ℚ := 3 / 4

-- Define the area of the field in square meters
def field_area : ℝ := 7500

-- Define the cost of fencing in paise per meter
def fencing_cost_paise : ℝ := 25

-- Theorem statement
theorem fencing_cost_calculation :
  let length : ℝ := Real.sqrt (field_area * side_ratio / (side_ratio + 1))
  let width : ℝ := length / side_ratio
  let perimeter : ℝ := 2 * (length + width)
  let total_cost : ℝ := perimeter * fencing_cost_paise / 100
  total_cost = 87.5 := by sorry

end NUMINAMATH_CALUDE_fencing_cost_calculation_l454_45496


namespace NUMINAMATH_CALUDE_square_root_problem_l454_45444

theorem square_root_problem (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (3*a + 2)^2 = x ∧ (a + 14)^2 = x) →
  (∃ (x : ℝ), x > 0 ∧ (3*a + 2)^2 = x ∧ (a + 14)^2 = x ∧ x = 100) :=
by sorry

end NUMINAMATH_CALUDE_square_root_problem_l454_45444


namespace NUMINAMATH_CALUDE_car_speed_in_kmph_l454_45433

/-- Proves that a car covering 375 meters in 15 seconds has a speed of 90 kmph -/
theorem car_speed_in_kmph : 
  let distance : ℝ := 375 -- distance in meters
  let time : ℝ := 15 -- time in seconds
  let conversion_factor : ℝ := 3.6 -- conversion factor from m/s to kmph
  (distance / time) * conversion_factor = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_in_kmph_l454_45433


namespace NUMINAMATH_CALUDE_middle_school_students_l454_45415

theorem middle_school_students (elementary : ℕ) (middle : ℕ) : 
  elementary = 4 * middle - 3 →
  elementary + middle = 247 →
  middle = 50 := by
sorry

end NUMINAMATH_CALUDE_middle_school_students_l454_45415


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l454_45494

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (n : ℕ), n > 0 → (n * (n + 1) * (n + 2) * (n + 3)) % d = 0) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ (m * (m + 1) * (m + 2) * (m + 3)) % k ≠ 0) ∧
  d = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l454_45494


namespace NUMINAMATH_CALUDE_abcdef_hex_bits_proof_l454_45485

/-- The number of bits required to represent ABCDEF₁₆ in binary -/
def abcdef_hex_to_bits : ℕ := 24

/-- The decimal value of ABCDEF₁₆ -/
def abcdef_hex_to_decimal : ℕ := 11293375

theorem abcdef_hex_bits_proof :
  abcdef_hex_to_bits = 24 ∧
  2^23 < abcdef_hex_to_decimal ∧
  abcdef_hex_to_decimal < 2^24 := by
  sorry

#eval abcdef_hex_to_bits
#eval abcdef_hex_to_decimal

end NUMINAMATH_CALUDE_abcdef_hex_bits_proof_l454_45485


namespace NUMINAMATH_CALUDE_gain_percentage_l454_45482

/-- Given an article sold for $110 with a gain of $10, prove that the gain percentage is 10%. -/
theorem gain_percentage (selling_price : ℝ) (gain : ℝ) (h1 : selling_price = 110) (h2 : gain = 10) :
  (gain / (selling_price - gain)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_gain_percentage_l454_45482


namespace NUMINAMATH_CALUDE_crossnumber_puzzle_l454_45477

/-- A number is a two-digit number if it's between 10 and 99 inclusive. -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The statement of the crossnumber puzzle -/
theorem crossnumber_puzzle :
  ∃! (a b c d : ℕ),
    isTwoDigit a ∧ isTwoDigit b ∧ isTwoDigit c ∧ isTwoDigit d ∧
    Nat.Prime a ∧
    ∃ (m n p : ℕ), b = m^2 ∧ c = n^2 ∧ d = p^2 ∧
    tensDigit a = unitsDigit b ∧
    unitsDigit a = tensDigit d ∧
    c = d :=
sorry

end NUMINAMATH_CALUDE_crossnumber_puzzle_l454_45477


namespace NUMINAMATH_CALUDE_election_win_margin_l454_45403

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    winner_votes = 3744 →
    winner_votes = (52 : ℕ) * total_votes / 100 →
    loser_votes = total_votes - winner_votes →
    winner_votes - loser_votes = 288 :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l454_45403


namespace NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l454_45421

/-- The number of points on the circle -/
def n : ℕ := 20

/-- The number of ways to choose 2 points from n points -/
def choose_diameter (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of ways to choose 2 points from the remaining n-2 points -/
def choose_remaining (n : ℕ) : ℕ := (n - 2) * (n - 3) / 2

/-- The total number of cyclic quadrilaterals with one right angle -/
def total_quadrilaterals (n : ℕ) : ℕ := choose_diameter n * choose_remaining n

theorem cyclic_quadrilaterals_count :
  total_quadrilaterals n = 29070 :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l454_45421


namespace NUMINAMATH_CALUDE_alkyne_ch_bond_polarization_l454_45420

-- Define the hybridization states
inductive Hybridization
| sp
| sp2
| sp3

-- Define a function to represent the s-character percentage
def sCharacter (h : Hybridization) : ℚ :=
  match h with
  | .sp  => 1/2
  | .sp2 => 1/3
  | .sp3 => 1/4

-- Define a function to represent electronegativity
def electronegativity (h : Hybridization) : ℝ := sorry

-- Define a function to represent bond polarization strength
def bondPolarizationStrength (h : Hybridization) : ℝ := sorry

-- Theorem statement
theorem alkyne_ch_bond_polarization :
  (∀ h : Hybridization, h ≠ Hybridization.sp → electronegativity Hybridization.sp > electronegativity h) ∧
  (∀ h : Hybridization, bondPolarizationStrength h = electronegativity h) ∧
  (bondPolarizationStrength Hybridization.sp > bondPolarizationStrength Hybridization.sp2) ∧
  (bondPolarizationStrength Hybridization.sp > bondPolarizationStrength Hybridization.sp3) := by
  sorry

end NUMINAMATH_CALUDE_alkyne_ch_bond_polarization_l454_45420


namespace NUMINAMATH_CALUDE_line_contains_diameter_l454_45411

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 8 = 0, 
    prove that the line 2x + y + 1 = 0 contains a diameter of the circle -/
theorem line_contains_diameter (x y : ℝ) :
  (x^2 + y^2 - 2*x + 6*y + 8 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 2*x₁ + 6*y₁ + 8 = 0) ∧
    (x₂^2 + y₂^2 - 2*x₂ + 6*y₂ + 8 = 0) ∧
    (2*x₁ + y₁ + 1 = 0) ∧
    (2*x₂ + y₂ + 1 = 0) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = (2*1)^2 + (2*(-3))^2)) :=
by sorry

end NUMINAMATH_CALUDE_line_contains_diameter_l454_45411


namespace NUMINAMATH_CALUDE_gcd_490_910_l454_45492

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_490_910_l454_45492


namespace NUMINAMATH_CALUDE_rationalize_and_product_l454_45419

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2:ℝ) + Real.sqrt 5) / ((3:ℝ) - Real.sqrt 5) = A + B * Real.sqrt C) ∧
  (A * B * C = 275) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l454_45419


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l454_45430

variable (m : ℝ)

def quadratic_equation (x : ℝ) := m * x^2 + (m - 3) * x + 1

theorem quadratic_roots_conditions :
  (∀ x, quadratic_equation m x ≠ 0 → m > 1) ∧
  ((∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ↔ 0 < m ∧ m < 1) ∧
  ((∃ x y, x > 0 ∧ y < 0 ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ↔ m < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l454_45430


namespace NUMINAMATH_CALUDE_claire_age_l454_45406

def age_problem (gabriel fiona ethan claire : ℕ) : Prop :=
  (gabriel = fiona - 2) ∧
  (fiona = ethan + 5) ∧
  (ethan = claire + 6) ∧
  (gabriel = 21)

theorem claire_age :
  ∀ gabriel fiona ethan claire : ℕ,
  age_problem gabriel fiona ethan claire →
  claire = 12 := by
sorry

end NUMINAMATH_CALUDE_claire_age_l454_45406


namespace NUMINAMATH_CALUDE_constant_sum_of_squares_l454_45461

/-- Defines an ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Defines a point P on the major axis of C -/
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Defines the direction vector of line l -/
def direction_vector : ℝ × ℝ := (2, 1)

/-- Defines the line l passing through P with the given direction vector -/
def line_l (m t : ℝ) : ℝ × ℝ := (m + 2*t, t)

/-- Defines the intersection points of line l and ellipse C -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l m t ∧ ellipse_C p.1 p.2}

/-- States that |PA|² + |PB|² is constant for all valid m -/
theorem constant_sum_of_squares (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) :
  ∃ A B, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B ∧
    (A.1 - m)^2 + A.2^2 + (B.1 - m)^2 + B.2^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_constant_sum_of_squares_l454_45461


namespace NUMINAMATH_CALUDE_sales_solution_l454_45484

def sales_problem (month1 month2 month4 month5 month6 average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := month1 + month2 + month4 + month5 + month6
  let month3 := total - known_sum
  month3 = 7855

theorem sales_solution :
  sales_problem 7435 7920 8230 7560 6000 7500 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l454_45484


namespace NUMINAMATH_CALUDE_gcd_conditions_and_sum_of_digits_l454_45460

/-- The least positive integer greater than 1000 satisfying the given GCD conditions -/
def n : ℕ := sorry

/-- Sum of digits function -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem gcd_conditions_and_sum_of_digits :
  n > 1000 ∧
  Nat.gcd 75 (n + 150) = 25 ∧
  Nat.gcd (n + 75) 150 = 75 ∧
  (∀ k, k > 1000 → Nat.gcd 75 (k + 150) = 25 → Nat.gcd (k + 75) 150 = 75 → k ≥ n) ∧
  sum_of_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_gcd_conditions_and_sum_of_digits_l454_45460


namespace NUMINAMATH_CALUDE_expand_product_l454_45471

theorem expand_product (x : ℝ) : (7*x + 5) * (5*x^2 - 2*x + 4) = 35*x^3 + 11*x^2 + 18*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l454_45471


namespace NUMINAMATH_CALUDE_activity_ratio_theorem_l454_45442

/-- Represents the ratio of time spent on two activities -/
structure TimeRatio where
  activity1 : ℝ
  activity2 : ℝ

/-- Calculates the score based on time spent on an activity -/
def calculateScore (pointsPerHour : ℝ) (hours : ℝ) : ℝ :=
  pointsPerHour * hours

/-- Theorem stating the relationship between activities and score -/
theorem activity_ratio_theorem (timeActivity1 : ℝ) (pointsPerHour : ℝ) (finalScore : ℝ) :
  timeActivity1 = 9 →
  pointsPerHour = 15 →
  finalScore = 45 →
  ∃ (ratio : TimeRatio),
    ratio.activity1 = timeActivity1 ∧
    ratio.activity2 = finalScore / pointsPerHour ∧
    ratio.activity2 / ratio.activity1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_activity_ratio_theorem_l454_45442


namespace NUMINAMATH_CALUDE_cos_graph_shift_l454_45445

theorem cos_graph_shift (x : ℝ) :
  3 * Real.cos (2 * x - π / 3) = 3 * Real.cos (2 * (x - π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_cos_graph_shift_l454_45445


namespace NUMINAMATH_CALUDE_trajectory_equation_MN_range_l454_45407

-- Define the circle P
structure CircleP where
  center : ℝ × ℝ
  passes_through_F : center.1^2 + center.2^2 = (center.1 - 1)^2 + center.2^2
  tangent_to_l : center.1 + 1 = abs center.2

-- Define the circle F
def circleF (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory C
def trajectoryC (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points M and N
structure Intersection (p : CircleP) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  on_circle_P : (M.1 - p.center.1)^2 + (M.2 - p.center.2)^2 = (p.center.1 - 1)^2 + p.center.2^2
              ∧ (N.1 - p.center.1)^2 + (N.2 - p.center.2)^2 = (p.center.1 - 1)^2 + p.center.2^2
  on_circle_F : circleF M.1 M.2 ∧ circleF N.1 N.2

-- Theorem statements
theorem trajectory_equation (p : CircleP) : trajectoryC p.center.1 p.center.2 := by sorry

theorem MN_range (p : CircleP) (i : Intersection p) : 
  Real.sqrt 3 ≤ Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) ∧ 
  Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) < 2 := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_MN_range_l454_45407


namespace NUMINAMATH_CALUDE_apple_basket_problem_l454_45475

theorem apple_basket_problem (x : ℕ) : 
  (x / 2 - 2) - ((x / 2 - 2) / 2 - 3) = 24 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l454_45475


namespace NUMINAMATH_CALUDE_cubic_inequality_l454_45451

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l454_45451


namespace NUMINAMATH_CALUDE_raspberry_juice_volume_l454_45465

/-- Proves that the original volume of raspberry juice is 6 quarts -/
theorem raspberry_juice_volume : ∀ (original_volume : ℚ),
  (original_volume / 12 + 1 = 3) →
  (original_volume / 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_raspberry_juice_volume_l454_45465


namespace NUMINAMATH_CALUDE_sum_of_digits_18_to_21_l454_45469

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def sum_of_digits_range (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => sum_of_digits (a + i))

theorem sum_of_digits_18_to_21 :
  sum_of_digits_range 18 21 = 24 :=
by
  sorry

-- The following definition is provided as a condition from the problem
axiom sum_of_digits_0_to_99 : sum_of_digits_range 0 99 = 900

end NUMINAMATH_CALUDE_sum_of_digits_18_to_21_l454_45469


namespace NUMINAMATH_CALUDE_arrangements_with_A_or_B_at_ends_eq_84_l454_45414

/-- The number of ways to arrange n distinct objects --/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 5 people in a row with at least one of A or B at the ends --/
def arrangements_with_A_or_B_at_ends : ℕ :=
  permutations 5 - (3 * 2 * permutations 3)

theorem arrangements_with_A_or_B_at_ends_eq_84 :
  arrangements_with_A_or_B_at_ends = 84 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_A_or_B_at_ends_eq_84_l454_45414


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l454_45435

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ),
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
    (a * b * c) % (a + 2012) = 0 ∧
    (a * b * c) % (b + 2012) = 0 ∧
    (a * b * c) % (c + 2012) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l454_45435


namespace NUMINAMATH_CALUDE_weight_vest_savings_l454_45447

theorem weight_vest_savings (weight_vest_cost plate_weight plate_cost_per_pound
                             weight_vest_200_cost weight_vest_200_discount : ℕ) :
  weight_vest_cost = 250 →
  plate_weight = 200 →
  plate_cost_per_pound = 12 / 10 →
  weight_vest_200_cost = 700 →
  weight_vest_200_discount = 100 →
  (weight_vest_200_cost - weight_vest_200_discount) - 
  (weight_vest_cost + plate_weight * plate_cost_per_pound) = 110 := by
sorry

end NUMINAMATH_CALUDE_weight_vest_savings_l454_45447


namespace NUMINAMATH_CALUDE_red_candies_count_l454_45481

/-- Represents the number of candies of each color -/
structure CandyCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- The conditions of the candy problem -/
def candy_conditions (c : CandyCounts) : Prop :=
  c.yellow = 3 * c.red - 20 ∧
  c.blue = c.yellow / 2 ∧
  c.red + c.blue = 90

/-- The theorem stating that there are 40 red candies -/
theorem red_candies_count :
  ∃ c : CandyCounts, candy_conditions c ∧ c.red = 40 := by
  sorry

end NUMINAMATH_CALUDE_red_candies_count_l454_45481


namespace NUMINAMATH_CALUDE_polynomial_sum_l454_45468

theorem polynomial_sum (x : ℝ) : (x^2 + 3*x - 4) + (-3*x + 1) = x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l454_45468


namespace NUMINAMATH_CALUDE_torn_sheets_count_l454_45463

/-- Represents a book with numbered pages -/
structure Book where
  /-- Each sheet contains two pages -/
  pages_per_sheet : Nat
  /-- The first torn-out page number -/
  first_torn_page : Nat
  /-- The last torn-out page number -/
  last_torn_page : Nat

/-- Check if two numbers have the same digits -/
def same_digits (a b : Nat) : Prop := sorry

/-- Calculate the number of torn-out sheets -/
def torn_sheets (book : Book) : Nat := sorry

/-- Main theorem -/
theorem torn_sheets_count (book : Book) :
  book.pages_per_sheet = 2 →
  book.first_torn_page = 185 →
  same_digits book.first_torn_page book.last_torn_page →
  Even book.last_torn_page →
  torn_sheets book = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l454_45463


namespace NUMINAMATH_CALUDE_original_number_theorem_l454_45405

theorem original_number_theorem (x : ℝ) : 
  12 * ((x * 0.5 - 10) / 6) = 15 → x = 35 := by
sorry

end NUMINAMATH_CALUDE_original_number_theorem_l454_45405


namespace NUMINAMATH_CALUDE_military_unit_reorganization_l454_45489

theorem military_unit_reorganization (x : ℕ) : 
  (x * (x + 5) = 5 * (x + 845)) → 
  (x * (x + 5) = 4550) := by
  sorry

end NUMINAMATH_CALUDE_military_unit_reorganization_l454_45489


namespace NUMINAMATH_CALUDE_slope_condition_l454_45440

/-- Given two points A(-3, 10) and B(5, y) in a coordinate plane, 
    if the slope of the line through A and B is -4/3, then y = -2/3. -/
theorem slope_condition (y : ℚ) : 
  let A : ℚ × ℚ := (-3, 10)
  let B : ℚ × ℚ := (5, y)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -4/3 → y = -2/3 := by
sorry

end NUMINAMATH_CALUDE_slope_condition_l454_45440


namespace NUMINAMATH_CALUDE_tetrahedron_vertex_equality_l454_45400

structure Tetrahedron where
  vertices : Fin 4 → ℝ
  vertex_positive : ∀ i, vertices i > 0

def face_sum (t : Tetrahedron) (i j k : Fin 4) : ℝ :=
  t.vertices i * t.vertices j + t.vertices j * t.vertices k + t.vertices k * t.vertices i

theorem tetrahedron_vertex_equality (t1 t2 : Tetrahedron) :
  (∀ (i j k : Fin 4), face_sum t1 i j k = face_sum t2 i j k) →
  (∀ (i : Fin 4), t1.vertices i = t2.vertices i) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_vertex_equality_l454_45400


namespace NUMINAMATH_CALUDE_complex_power_four_l454_45499

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_four (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l454_45499


namespace NUMINAMATH_CALUDE_percentage_problem_l454_45425

theorem percentage_problem (P : ℝ) : (P / 100) * 150 - 40 = 50 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l454_45425


namespace NUMINAMATH_CALUDE_f_sum_equals_sqrt2_minus_1_l454_45427

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, 0 ≤ x ∧ x < 1 → f x = 2 * x - 1)

theorem f_sum_equals_sqrt2_minus_1 (f : ℝ → ℝ) (hf : f_properties f) :
  f (1/2) + f 1 + f (3/2) + f (5/2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_sqrt2_minus_1_l454_45427


namespace NUMINAMATH_CALUDE_quadratic_no_roots_if_geometric_sequence_l454_45422

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- A quadratic function f(x) = ax² + bx + c has no real roots if and only if
    its discriminant is negative. -/
def HasNoRealRoots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem quadratic_no_roots_if_geometric_sequence (a b c : ℝ) (ha : a ≠ 0) :
  IsGeometricSequence a b c → HasNoRealRoots a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_if_geometric_sequence_l454_45422


namespace NUMINAMATH_CALUDE_lunks_for_two_dozen_bananas_l454_45456

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℕ) : ℕ := l / 2

/-- Exchange rate between kunks and bananas -/
def kunks_to_bananas (k : ℕ) : ℕ := 2 * k

/-- Number of lunks needed to buy a given number of bananas -/
def lunks_for_bananas (b : ℕ) : ℕ :=
  let kunks_needed := (b + 5) / 6 * 3  -- Round up division
  2 * kunks_needed

theorem lunks_for_two_dozen_bananas :
  lunks_for_bananas 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_lunks_for_two_dozen_bananas_l454_45456


namespace NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l454_45491

theorem remainder_of_55_power_55_plus_55_mod_56 :
  (55^55 + 55) % 56 = 54 := by sorry

end NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l454_45491


namespace NUMINAMATH_CALUDE_train_length_calculation_l454_45412

/-- The length of a train given its speed, the speed of a man it's passing, and the time it takes to cross the man completely. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 53.99568034557235 →
  ∃ (train_length : ℝ), abs (train_length - 900) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l454_45412


namespace NUMINAMATH_CALUDE_handshake_problem_l454_45479

theorem handshake_problem (n : ℕ) (s : ℕ) : 
  n * (n - 1) / 2 + s = 159 → n = 18 ∧ s = 6 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l454_45479


namespace NUMINAMATH_CALUDE_tangent_and_cosine_identities_l454_45443

theorem tangent_and_cosine_identities 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : 0 < β ∧ β < π) 
  (h3 : (Real.tan α)^2 - 5*(Real.tan α) + 6 = 0) 
  (h4 : (Real.tan β)^2 - 5*(Real.tan β) + 6 = 0) : 
  Real.tan (α + β) = -1 ∧ Real.cos (α - β) = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_cosine_identities_l454_45443


namespace NUMINAMATH_CALUDE_range_when_proposition_false_l454_45466

theorem range_when_proposition_false (x : ℝ) :
  x^2 - 5*x + 4 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_when_proposition_false_l454_45466


namespace NUMINAMATH_CALUDE_garden_plants_correct_l454_45474

/-- Calculates the total number of plants in Papi Calot's garden -/
def garden_plants : ℕ × ℕ × ℕ :=
  let potato_rows := 8
  let potato_alt1 := 22
  let potato_alt2 := 25
  let potato_extra := 18

  let carrot_rows := 12
  let carrot_start := 30
  let carrot_increment := 5
  let carrot_extra := 24

  let onion_repetitions := 4
  let onion_row1 := 15
  let onion_row2 := 20
  let onion_row3 := 25
  let onion_extra := 12

  let potatoes := (potato_rows / 2 * potato_alt1 + potato_rows / 2 * potato_alt2) + potato_extra
  let carrots := (carrot_rows * (2 * carrot_start + (carrot_rows - 1) * carrot_increment)) / 2 + carrot_extra
  let onions := onion_repetitions * (onion_row1 + onion_row2 + onion_row3) + onion_extra

  (potatoes, carrots, onions)

theorem garden_plants_correct :
  garden_plants = (206, 714, 252) :=
by sorry

end NUMINAMATH_CALUDE_garden_plants_correct_l454_45474


namespace NUMINAMATH_CALUDE_square_difference_equality_l454_45401

theorem square_difference_equality : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l454_45401


namespace NUMINAMATH_CALUDE_tray_height_l454_45432

/-- Given a square with side length 150 and cuts starting at √50 from each corner
    meeting at a 45° angle on the diagonal, the perpendicular height of the
    resulting tray is √1470. -/
theorem tray_height (square_side : ℝ) (cut_start : ℝ) (cut_angle : ℝ) :
  square_side = 150 →
  cut_start = Real.sqrt 50 →
  cut_angle = 45 →
  ∃ (height : ℝ), height = Real.sqrt 1470 :=
by sorry

end NUMINAMATH_CALUDE_tray_height_l454_45432


namespace NUMINAMATH_CALUDE_shark_difference_l454_45488

theorem shark_difference (cape_may_sharks daytona_beach_sharks : ℕ) 
  (h1 : cape_may_sharks = 32) 
  (h2 : daytona_beach_sharks = 12) : 
  cape_may_sharks - 2 * daytona_beach_sharks = 8 := by
  sorry

end NUMINAMATH_CALUDE_shark_difference_l454_45488


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l454_45402

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (segmentNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  let firstStudent := 20 - interval  -- Given that student 20 is in the second segment
  firstStudent + (segmentNumber - 1) * interval

/-- Theorem statement for the systematic sampling problem -/
theorem systematic_sampling_problem :
  let totalStudents : ℕ := 700
  let sampleSize : ℕ := 50
  let targetSegment : ℕ := 5
  systematicSample totalStudents sampleSize targetSegment = 62 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_problem_l454_45402


namespace NUMINAMATH_CALUDE_billy_ate_nine_apples_wednesday_l454_45438

/-- The number of apples Billy ate each day of the week -/
structure AppleConsumption where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Billy's apple consumption throughout the week -/
def billy_apple_conditions (ac : AppleConsumption) : Prop :=
  ac.monday = 2 ∧
  ac.tuesday = 2 * ac.monday ∧
  ac.thursday = 4 * ac.friday ∧
  ac.friday = ac.monday / 2 ∧
  ac.monday + ac.tuesday + ac.wednesday + ac.thursday + ac.friday = 20

/-- Theorem stating that given the conditions, Billy ate 9 apples on Wednesday -/
theorem billy_ate_nine_apples_wednesday (ac : AppleConsumption) 
  (h : billy_apple_conditions ac) : ac.wednesday = 9 := by
  sorry


end NUMINAMATH_CALUDE_billy_ate_nine_apples_wednesday_l454_45438


namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l454_45416

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ (p : Prism), p.edges = 18 → num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l454_45416


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l454_45470

-- Define the function f(x) = (x - 1)^2 - 2
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y := by
  sorry

-- Note: Set.Ici 1 represents the interval [1, +∞)

end NUMINAMATH_CALUDE_f_increasing_on_interval_l454_45470


namespace NUMINAMATH_CALUDE_system_solution_l454_45490

theorem system_solution (x y z : ℝ) : 
  x^2 + y^2 = -x + 3*y + z ∧ 
  y^2 + z^2 = x + 3*y - z ∧ 
  x^2 + z^2 = 2*x + 2*y - z ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l454_45490


namespace NUMINAMATH_CALUDE_english_failure_percentage_l454_45434

/-- The percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 25

/-- The percentage of students who failed in both Hindi and English -/
def failed_both : ℝ := 27

/-- The percentage of students who passed in both subjects -/
def passed_both : ℝ := 54

/-- The percentage of students who failed in English -/
def failed_english : ℝ := 100 - passed_both - failed_hindi + failed_both

theorem english_failure_percentage :
  failed_english = 48 :=
sorry

end NUMINAMATH_CALUDE_english_failure_percentage_l454_45434


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l454_45457

def is_valid_number (n : ℕ) : Prop :=
  (100000 ≤ n) ∧ (n < 1000000) ∧ (n / 100000 = 1) ∧
  ((n % 100000) * 10 + 1 = 3 * n)

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l454_45457


namespace NUMINAMATH_CALUDE_unit_digit_of_fraction_l454_45429

theorem unit_digit_of_fraction (n : ℕ) :
  (33 * 10) / (2^1984) % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_fraction_l454_45429


namespace NUMINAMATH_CALUDE_solution_when_a_is_one_solution_when_a_greater_than_two_solution_when_a_equals_two_solution_when_a_less_than_two_l454_45487

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (a+2)*x + 2*a < 0

-- Theorem for a = 1
theorem solution_when_a_is_one :
  ∀ x : ℝ, inequality 1 x ↔ 1 < x ∧ x < 2 :=
sorry

-- Theorem for a > 2
theorem solution_when_a_greater_than_two :
  ∀ a x : ℝ, a > 2 → (inequality a x ↔ 2 < x ∧ x < a) :=
sorry

-- Theorem for a = 2
theorem solution_when_a_equals_two :
  ∀ x : ℝ, ¬(inequality 2 x) :=
sorry

-- Theorem for a < 2
theorem solution_when_a_less_than_two :
  ∀ a x : ℝ, a < 2 → (inequality a x ↔ a < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_when_a_is_one_solution_when_a_greater_than_two_solution_when_a_equals_two_solution_when_a_less_than_two_l454_45487


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l454_45464

-- Define the rotation and reflection transformations
def rotate90CounterClockwise (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (h, k) := center
  let (x, y) := point
  (h - (y - k), k + (x - h))

def reflectAboutYEqualsX (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (y, x)

-- State the theorem
theorem point_transformation_theorem (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let rotated := rotate90CounterClockwise (2, 3) P
  let final := reflectAboutYEqualsX rotated
  final = (4, -5) → b - a = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l454_45464


namespace NUMINAMATH_CALUDE_max_expensive_product_price_is_30900_l454_45423

/-- Represents a company's product line -/
structure ProductLine where
  total_products : Nat
  average_price : ℝ
  min_price : ℝ
  num_below_threshold : Nat
  threshold : ℝ

/-- Calculates the maximum possible price for the most expensive product -/
def max_expensive_product_price (pl : ProductLine) : ℝ :=
  let total_value := pl.total_products * pl.average_price
  let min_value_below_threshold := pl.num_below_threshold * pl.min_price
  let remaining_products := pl.total_products - pl.num_below_threshold
  let remaining_value := total_value - min_value_below_threshold
  let value_at_threshold := (remaining_products - 1) * pl.threshold
  remaining_value - value_at_threshold

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_product_price_is_30900 :
  let pl := ProductLine.mk 40 1800 500 15 1400
  max_expensive_product_price pl = 30900 := by
  sorry

end NUMINAMATH_CALUDE_max_expensive_product_price_is_30900_l454_45423


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l454_45476

theorem quadratic_function_properties (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*b*x + b^2 + b - 5 = 0) →
  (∀ x < 3.5, (2*x - 2*b) < 0) →
  3.5 ≤ b ∧ b ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l454_45476


namespace NUMINAMATH_CALUDE_min_value_theorem_l454_45448

theorem min_value_theorem (x y a b : ℝ) (h1 : x - 2*y - 2 ≤ 0) 
  (h2 : x + y - 2 ≤ 0) (h3 : 2*x - y + 2 ≥ 0) (ha : a > 0) (hb : b > 0)
  (h4 : ∀ (x' y' : ℝ), x' - 2*y' - 2 ≤ 0 → x' + y' - 2 ≤ 0 → 2*x' - y' + 2 ≥ 0 
    → a*x' + b*y' + 5 ≥ a*x + b*y + 5)
  (h5 : a*x + b*y + 5 = 2) :
  (2/a + 3/b : ℝ) ≥ (10 + 4*Real.sqrt 6)/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l454_45448


namespace NUMINAMATH_CALUDE_validMSetIs0And8_l454_45497

/-- The function f(x) = x^2 + mx - 2m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2*m - 1

/-- Predicate to check if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Predicate to check if all roots of f are integers -/
def hasOnlyIntegerRoots (m : ℝ) : Prop :=
  ∀ x : ℝ, f m x = 0 → isInteger x

/-- The set of m values for which f has only integer roots -/
def validMSet : Set ℝ := {m | hasOnlyIntegerRoots m}

/-- Theorem stating that the set of valid m values is {0, -8} -/
theorem validMSetIs0And8 : validMSet = {0, -8} := by sorry

end NUMINAMATH_CALUDE_validMSetIs0And8_l454_45497


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l454_45437

theorem least_number_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 34 = 4 ∧ 
  n % 5 = 4 ∧
  ∀ m : ℕ, m > 0 ∧ m % 34 = 4 ∧ m % 5 = 4 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l454_45437


namespace NUMINAMATH_CALUDE_remainder_of_a_l454_45410

theorem remainder_of_a (a : ℤ) :
  (a^100 % 73 = 2) → (a^101 % 73 = 69) → (a % 73 = 71) := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_a_l454_45410


namespace NUMINAMATH_CALUDE_solution_check_l454_45446

theorem solution_check (x : ℝ) : x = 2 ↔ -1/3 * x + 2/3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_check_l454_45446


namespace NUMINAMATH_CALUDE_tetrahedron_reciprocal_squares_equality_l454_45439

/-- A tetrahedron with heights and distances between opposite edges. -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  h₁_pos : 0 < h₁
  h₂_pos : 0 < h₂
  h₃_pos : 0 < h₃
  h₄_pos : 0 < h₄
  d₁_pos : 0 < d₁
  d₂_pos : 0 < d₂
  d₃_pos : 0 < d₃

/-- The sum of reciprocal squares of heights equals the sum of reciprocal squares of distances. -/
theorem tetrahedron_reciprocal_squares_equality (t : Tetrahedron) :
    1 / t.h₁^2 + 1 / t.h₂^2 + 1 / t.h₃^2 + 1 / t.h₄^2 = 1 / t.d₁^2 + 1 / t.d₂^2 + 1 / t.d₃^2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_reciprocal_squares_equality_l454_45439


namespace NUMINAMATH_CALUDE_average_of_multiples_10_to_300_l454_45459

def multiples_of_10 (n : ℕ) : List ℕ :=
  List.filter (fun x => x % 10 = 0) (List.range (n + 1))

theorem average_of_multiples_10_to_300 :
  let sequence := multiples_of_10 300
  (sequence.sum / sequence.length : ℚ) = 155 := by
sorry

end NUMINAMATH_CALUDE_average_of_multiples_10_to_300_l454_45459


namespace NUMINAMATH_CALUDE_orange_count_l454_45458

theorem orange_count (initial : ℕ) : 
  initial - 9 + 38 = 60 → initial = 31 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l454_45458


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l454_45450

def annual_salary : ℝ := 40000
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800
def take_home_pay : ℝ := 27200

theorem tax_percentage_calculation :
  let healthcare_deduction := annual_salary * healthcare_rate
  let total_non_tax_deductions := healthcare_deduction + union_dues
  let amount_before_taxes := annual_salary - total_non_tax_deductions
  let tax_deduction := amount_before_taxes - take_home_pay
  let tax_percentage := (tax_deduction / annual_salary) * 100
  tax_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l454_45450


namespace NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l454_45428

theorem flower_shop_carnation_percentage :
  let carnations : ℝ := 1  -- Arbitrary non-zero value for carnations
  let violets : ℝ := (1/3) * carnations
  let tulips : ℝ := (1/4) * violets
  let roses : ℝ := tulips
  let total : ℝ := carnations + violets + tulips + roses
  (carnations / total) * 100 = 200/3 := by
sorry

end NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l454_45428


namespace NUMINAMATH_CALUDE_employees_in_all_restaurants_l454_45473

/-- The number of employees trained to work in all three restaurants at a resort --/
theorem employees_in_all_restaurants (total_employees : ℕ) 
  (family_buffet dining_room snack_bar in_two_restaurants : ℕ) : 
  total_employees = 39 →
  family_buffet = 19 →
  dining_room = 18 →
  snack_bar = 12 →
  in_two_restaurants = 4 →
  ∃ (in_all_restaurants : ℕ),
    family_buffet + dining_room + snack_bar - in_two_restaurants - 2 * in_all_restaurants = total_employees ∧
    in_all_restaurants = 5 :=
by sorry

end NUMINAMATH_CALUDE_employees_in_all_restaurants_l454_45473


namespace NUMINAMATH_CALUDE_prob_no_adjacent_same_five_people_l454_45498

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of faces on the standard die -/
def d : ℕ := 6

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ :=
  (d - 1)^(n - 1) * (d - 2) / d^n

theorem prob_no_adjacent_same_five_people (h : n = 5) :
  prob_no_adjacent_same = 625 / 1944 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_same_five_people_l454_45498


namespace NUMINAMATH_CALUDE_number_of_grandchildren_excluding_shelby_l454_45495

/-- Proves the number of grandchildren excluding Shelby, given the inheritance details --/
theorem number_of_grandchildren_excluding_shelby
  (total_inheritance : ℕ)
  (shelby_share : ℕ)
  (remaining_share : ℕ)
  (one_grandchild_share : ℕ)
  (h1 : total_inheritance = 124600)
  (h2 : shelby_share = total_inheritance / 2)
  (h3 : remaining_share = total_inheritance - shelby_share)
  (h4 : one_grandchild_share = 6230)
  (h5 : remaining_share % one_grandchild_share = 0) :
  remaining_share / one_grandchild_share = 10 := by
  sorry

#check number_of_grandchildren_excluding_shelby

end NUMINAMATH_CALUDE_number_of_grandchildren_excluding_shelby_l454_45495


namespace NUMINAMATH_CALUDE_bus_passenger_count_l454_45455

/-- Calculates the total number of passengers transported by a bus. -/
def totalPassengers (numTrips : ℕ) (initialPassengers : ℕ) (passengerDecrease : ℕ) : ℕ :=
  (numTrips * (2 * initialPassengers - (numTrips - 1) * passengerDecrease)) / 2

/-- Proves that the total number of passengers transported is 1854. -/
theorem bus_passenger_count : totalPassengers 18 120 2 = 1854 := by
  sorry

#eval totalPassengers 18 120 2

end NUMINAMATH_CALUDE_bus_passenger_count_l454_45455


namespace NUMINAMATH_CALUDE_inequality_solution_set_l454_45441

theorem inequality_solution_set (x : ℝ) : 
  (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l454_45441


namespace NUMINAMATH_CALUDE_seven_not_spheric_spheric_power_is_spheric_l454_45480

/-- A rational number is spheric if it is the sum of three squares of rational numbers. -/
def is_spheric (r : ℚ) : Prop :=
  ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := by
  sorry

theorem spheric_power_is_spheric (r : ℚ) (n : ℕ) (hn : n > 1) :
  is_spheric r → is_spheric (r^n) := by
  sorry

end NUMINAMATH_CALUDE_seven_not_spheric_spheric_power_is_spheric_l454_45480


namespace NUMINAMATH_CALUDE_river_current_speed_l454_45417

/-- The speed of the river's current in miles per hour -/
def river_speed : ℝ := 9.8

/-- The distance traveled downstream and upstream in miles -/
def distance : ℝ := 21

/-- The woman's initial canoeing speed in still water (miles per hour) -/
noncomputable def initial_speed : ℝ := 
  Real.sqrt (river_speed^2 + 7 * river_speed)

/-- Time difference between upstream and downstream journeys in hours -/
def time_difference : ℝ := 3

/-- Time difference after increasing paddling speed by 50% in hours -/
def reduced_time_difference : ℝ := 0.75

theorem river_current_speed :
  (distance / (initial_speed + river_speed) + time_difference 
    = distance / (initial_speed - river_speed)) ∧
  (distance / (1.5 * initial_speed + river_speed) + reduced_time_difference 
    = distance / (1.5 * initial_speed - river_speed)) :=
sorry

end NUMINAMATH_CALUDE_river_current_speed_l454_45417


namespace NUMINAMATH_CALUDE_b_over_a_range_l454_45452

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

-- Define the function f
def f (x a b c : ℝ) : ℝ := x^2 + c^2 - a^2 - a*b

-- State the theorem
theorem b_over_a_range (t : AcuteTriangle) 
  (h : ∃! x, f x t.a t.b t.c = 0) : 
  1 < t.b / t.a ∧ t.b / t.a < 2 := by
  sorry

end NUMINAMATH_CALUDE_b_over_a_range_l454_45452


namespace NUMINAMATH_CALUDE_paint_coverage_per_quart_l454_45483

/-- Represents the cost of paint per quart in dollars -/
def paint_cost_per_quart : ℝ := 3.20

/-- Represents the total cost to paint the cube in dollars -/
def total_paint_cost : ℝ := 192

/-- Represents the length of one edge of the cube in feet -/
def cube_edge_length : ℝ := 10

/-- Theorem stating the coverage of one quart of paint in square feet -/
theorem paint_coverage_per_quart : 
  (6 * cube_edge_length^2) / (total_paint_cost / paint_cost_per_quart) = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_coverage_per_quart_l454_45483


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l454_45408

/-- Two parallel lines with a given distance -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  distance : ℝ
  is_parallel : m = 8
  satisfies_distance : distance = 3

/-- The sum m + n for parallel lines with the given properties is either 48 or -12 -/
theorem parallel_lines_sum (lines : ParallelLines) : lines.m + lines.n = 48 ∨ lines.m + lines.n = -12 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l454_45408


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l454_45409

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- The sequence of integer pairs sorted by sum and then by first element -/
def sortedPairs : List IntPair := sorry

/-- The 60th element in the sortedPairs sequence -/
def sixtiethPair : IntPair := sorry

/-- Theorem stating that the 60th pair in the sequence is (5,7) -/
theorem sixtieth_pair_is_five_seven : 
  sixtiethPair = IntPair.mk 5 7 := by sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l454_45409


namespace NUMINAMATH_CALUDE_trapezoid_BE_length_l454_45449

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a trapezoid ABCD with point F outside -/
structure Trapezoid :=
  (A B C D F : Point)
  (is_trapezoid : sorry)  -- Condition that ABCD is a trapezoid
  (F_on_AD_extension : sorry)  -- Condition that F is on the extension of AD

/-- Given a trapezoid, find point E on AC such that E is on BF -/
def find_E (t : Trapezoid) : Point :=
  sorry

/-- Given a trapezoid, find point G on the extension of DC such that FG is parallel to BC -/
def find_G (t : Trapezoid) : Point :=
  sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem trapezoid_BE_length (t : Trapezoid) :
  let E := find_E t
  let G := find_G t
  distance t.B E = 30 :=
  sorry

end NUMINAMATH_CALUDE_trapezoid_BE_length_l454_45449


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l454_45436

theorem expression_simplification_and_evaluation (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1)) = (2 + x) / (2 - x) ∧
  (2 + 0) / (2 - 0) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l454_45436
