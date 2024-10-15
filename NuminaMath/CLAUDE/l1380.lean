import Mathlib

namespace NUMINAMATH_CALUDE_ratio_sum_problem_l1380_138012

theorem ratio_sum_problem (w x y : ℝ) (hw_x : w / x = 1 / 6) (hw_y : w / y = 1 / 5) :
  (x + y) / y = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l1380_138012


namespace NUMINAMATH_CALUDE_leahs_calculation_l1380_138096

theorem leahs_calculation (y : ℕ) (h : (y + 4) * 5 = 140) : y * 5 + 4 = 124 := by
  sorry

end NUMINAMATH_CALUDE_leahs_calculation_l1380_138096


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1380_138002

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, x)
  are_parallel a b → x = -4 := by
    sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1380_138002


namespace NUMINAMATH_CALUDE_complex_product_sum_l1380_138094

theorem complex_product_sum (a b : ℝ) : 
  (1 + Complex.I) * (2 + Complex.I) = Complex.mk a b → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l1380_138094


namespace NUMINAMATH_CALUDE_tan_sin_inequality_l1380_138053

open Real

theorem tan_sin_inequality (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (tan x - x) / (x - sin x) > 2 ∧
  ∃ (a : ℝ), a = 3 ∧ ∀ b > a, ∃ y ∈ Set.Ioo 0 (π/2), tan y + 2 * sin y - b * y ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_tan_sin_inequality_l1380_138053


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l1380_138088

/-- The exchange rate from U.S. dollars to British pounds -/
def exchange_rate : ℚ := 8 / 5

/-- The amount spent in British pounds -/
def amount_spent : ℚ := 72

/-- The remaining amount in British pounds as a function of initial U.S. dollars -/
def remaining (d : ℚ) : ℚ := 4 * d

theorem currency_exchange_problem (d : ℚ) : 
  (exchange_rate * d - amount_spent = remaining d) → d = -30 := by
  sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l1380_138088


namespace NUMINAMATH_CALUDE_transformations_of_g_reflection_about_y_axis_horizontal_stretch_horizontal_shift_l1380_138045

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f ((1 - x) / 2)

-- Theorem stating the transformations
theorem transformations_of_g (x : ℝ) :
  g x = f (-(x - 1) / 2) :=
sorry

-- Theorem stating the reflection about y-axis
theorem reflection_about_y_axis (x : ℝ) :
  g (-x + 1) = f x :=
sorry

-- Theorem stating the horizontal stretch
theorem horizontal_stretch (x : ℝ) :
  g (2 * x + 1) = f x :=
sorry

-- Theorem stating the horizontal shift
theorem horizontal_shift (x : ℝ) :
  g (x + 1) = f (x / 2) :=
sorry

end NUMINAMATH_CALUDE_transformations_of_g_reflection_about_y_axis_horizontal_stretch_horizontal_shift_l1380_138045


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1380_138033

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 1) :
  (1/a + 1/b) ≥ 4*Real.sqrt 3 + 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1380_138033


namespace NUMINAMATH_CALUDE_min_value_of_F_l1380_138081

theorem min_value_of_F (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (min : ℝ), min = 6/11 ∧ ∀ (F : ℝ), F = 2*x^2 + 3*y^2 + z^2 → F ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_F_l1380_138081


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l1380_138048

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C

/-- The theorem stating that A = π/3 given the condition -/
theorem angle_A_value (t : Triangle) (h : triangle_condition t) : t.A = π / 3 :=
sorry

/-- The theorem stating the maximum area when a = 4 -/
theorem max_area (t : Triangle) (h : triangle_condition t) (ha : t.a = 4) :
  (∀ t' : Triangle, triangle_condition t' → t'.a = 4 → 
    t'.b * t'.c * Real.sin t'.A / 2 ≤ 4 * Real.sqrt 3) ∧
  (∃ t' : Triangle, triangle_condition t' ∧ t'.a = 4 ∧ 
    t'.b * t'.c * Real.sin t'.A / 2 = 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l1380_138048


namespace NUMINAMATH_CALUDE_nested_even_function_is_even_l1380_138082

-- Define an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- State the theorem
theorem nested_even_function_is_even
  (g : ℝ → ℝ)
  (h : even_function g) :
  even_function (fun x ↦ g (g (g (g x)))) :=
by sorry

end NUMINAMATH_CALUDE_nested_even_function_is_even_l1380_138082


namespace NUMINAMATH_CALUDE_product_sum_multiplier_l1380_138083

theorem product_sum_multiplier (a b k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : a + b = k * (a * b)) (h2 : 1 / a + 1 / b = 6) : k = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_multiplier_l1380_138083


namespace NUMINAMATH_CALUDE_range_a_theorem_l1380_138040

/-- The range of a satisfying both conditions -/
def range_a : Set ℝ :=
  {a | (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4)}

/-- Condition 1: For all x ∈ ℝ, ax^2 + ax + 1 > 0 -/
def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Condition 2: The standard equation of the hyperbola is (x^2)/(1-a) + (y^2)/(a-3) = 1 -/
def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (1 - a) + y^2 / (a - 3) = 1

/-- The main theorem stating that if both conditions are satisfied, then a is in the specified range -/
theorem range_a_theorem (a : ℝ) :
  condition1 a ∧ condition2 a → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1380_138040


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l1380_138049

theorem fraction_sum_equals_two : 
  (1 : ℚ) / 2 + (1 : ℚ) / 2 + (1 : ℚ) / 3 + (1 : ℚ) / 3 + (1 : ℚ) / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l1380_138049


namespace NUMINAMATH_CALUDE_equal_distribution_l1380_138023

def total_amount : ℕ := 42900
def num_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem equal_distribution :
  total_amount / num_persons = amount_per_person := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l1380_138023


namespace NUMINAMATH_CALUDE_magistrate_seating_arrangements_l1380_138091

/-- Represents the number of people of each nationality -/
def magistrates : Finset (Nat × Nat) := {(2, 3), (1, 4)}

/-- The total number of magistrate members -/
def total_members : Nat := (magistrates.sum (λ x => x.1))

/-- Calculates the number of valid seating arrangements -/
def valid_arrangements (m : Finset (Nat × Nat)) (total : Nat) : Nat :=
  sorry

theorem magistrate_seating_arrangements :
  valid_arrangements magistrates total_members = 1895040 := by
  sorry

end NUMINAMATH_CALUDE_magistrate_seating_arrangements_l1380_138091


namespace NUMINAMATH_CALUDE_max_distance_trig_points_l1380_138067

theorem max_distance_trig_points (α β : ℝ) : 
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  ∃ (max_dist : ℝ), max_dist = 2 ∧ ∀ (α' β' : ℝ), 
    let P' := (Real.cos α', Real.sin α')
    let Q' := (Real.cos β', Real.sin β')
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_trig_points_l1380_138067


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1380_138087

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof :
  ∃! x : ℕ, x > 0 ∧ 
    (∃ y : ℕ, y > 0 ∧ 
      x + y = 4728 ∧ 
      is_divisible_by (x + y) 27 ∧
      is_divisible_by (x + y) 35 ∧
      is_divisible_by (x + y) 25 ∧
      is_divisible_by (x + y) 21) ∧
    (∀ z : ℕ, z > 0 ∧ 
      (∃ w : ℕ, w > 0 ∧ 
        z + w = 4728 ∧ 
        is_divisible_by (z + w) 27 ∧
        is_divisible_by (z + w) 35 ∧
        is_divisible_by (z + w) 25 ∧
        is_divisible_by (z + w) 21) → 
      x ≤ z) ∧
  x = 4725 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1380_138087


namespace NUMINAMATH_CALUDE_license_plate_count_l1380_138072

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 4

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

theorem license_plate_count : 
  num_digits ^ digits_count * num_letters ^ letters_count * block_positions = 878800000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1380_138072


namespace NUMINAMATH_CALUDE_count_integers_with_digit_sum_two_is_correct_l1380_138038

/-- The count of positive integers between 10^7 and 10^8 whose sum of digits is equal to 2 -/
def count_integers_with_digit_sum_two : ℕ :=
  let lower_bound := 10^7
  let upper_bound := 10^8 - 1
  let digit_sum := 2
  28  -- The actual count, to be proven

/-- Theorem stating that the count of positive integers between 10^7 and 10^8
    whose sum of digits is equal to 2 is correct -/
theorem count_integers_with_digit_sum_two_is_correct :
  count_integers_with_digit_sum_two = 
    (Finset.filter (fun n => (Finset.sum (Finset.range 8) (fun i => (n / 10^i) % 10) = 2))
      (Finset.range (10^8 - 10^7))).card :=
by
  sorry

#eval count_integers_with_digit_sum_two

end NUMINAMATH_CALUDE_count_integers_with_digit_sum_two_is_correct_l1380_138038


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l1380_138095

theorem imaginary_part_of_complex (z : ℂ) :
  z = 2 + (1 / (3 * I)) → z.im = -1/3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l1380_138095


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l1380_138010

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l1380_138010


namespace NUMINAMATH_CALUDE_unique_box_configuration_l1380_138084

/-- Represents a square piece --/
structure Square :=
  (id : Nat)

/-- Represents the F-shaped figure --/
def FShape := List Square

/-- Represents additional squares --/
def AdditionalSquares := List Square

/-- Represents a possible configuration of an open rectangular box --/
def BoxConfiguration := List Square

/-- A function that attempts to form an open rectangular box --/
def formBox (f : FShape) (add : AdditionalSquares) : Option BoxConfiguration :=
  sorry

/-- The main theorem stating there's exactly one valid configuration --/
theorem unique_box_configuration 
  (f : FShape) 
  (add : AdditionalSquares) 
  (h1 : f.length = 7) 
  (h2 : add.length = 3) : 
  ∃! (box : BoxConfiguration), formBox f add = some box :=
sorry

end NUMINAMATH_CALUDE_unique_box_configuration_l1380_138084


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1380_138004

theorem sqrt_equation_solutions :
  {x : ℝ | Real.sqrt (4 * x - 3) + 10 / Real.sqrt (4 * x - 3) = 7} = {7/4, 7} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1380_138004


namespace NUMINAMATH_CALUDE_m_values_l1380_138069

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, -1/2, 1/3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_m_values_l1380_138069


namespace NUMINAMATH_CALUDE_valid_parameterizations_l1380_138079

/-- The line equation y = -3x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

/-- A point is on the line if it satisfies the line equation -/
def point_on_line (p : ℝ × ℝ) : Prop :=
  line_equation p.1 p.2

/-- The direction vector is valid if it's parallel to (1, -3) -/
def valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, -3*k)

/-- A parameterization is valid if its point is on the line and its direction is valid -/
def valid_parameterization (p v : ℝ × ℝ) : Prop :=
  point_on_line p ∧ valid_direction v

theorem valid_parameterizations :
  valid_parameterization (4/3, 0) (1, -3) ∧
  valid_parameterization (-2, 10) (-3, 9) :=
by sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l1380_138079


namespace NUMINAMATH_CALUDE_samir_stairs_count_l1380_138080

theorem samir_stairs_count (s : ℕ) : 
  (s + (s / 2 + 18) = 495) → s = 318 := by
  sorry

end NUMINAMATH_CALUDE_samir_stairs_count_l1380_138080


namespace NUMINAMATH_CALUDE_ball_count_proof_l1380_138062

theorem ball_count_proof (total : ℕ) (p_yellow p_blue p_red : ℚ) 
  (h_total : total = 80)
  (h_yellow : p_yellow = 1/4)
  (h_blue : p_blue = 7/20)
  (h_red : p_red = 2/5)
  (h_sum : p_yellow + p_blue + p_red = 1) :
  ∃ (yellow blue red : ℕ),
    yellow = 20 ∧ 
    blue = 28 ∧ 
    red = 32 ∧
    yellow + blue + red = total ∧
    (yellow : ℚ) / total = p_yellow ∧
    (blue : ℚ) / total = p_blue ∧
    (red : ℚ) / total = p_red :=
by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l1380_138062


namespace NUMINAMATH_CALUDE_area_of_triangle_ABF_l1380_138044

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus F
def F : ℝ × ℝ := (-2, 0)

-- Define the intersection line
def intersection_line (x : ℝ) : Prop := x = 2

-- Define the points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, -3)

-- Theorem statement
theorem area_of_triangle_ABF :
  hyperbola A.1 A.2 ∧
  hyperbola B.1 B.2 ∧
  intersection_line A.1 ∧
  intersection_line B.1 →
  (1/2 : ℝ) * |A.2 - B.2| * |A.1 - F.1| = 12 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABF_l1380_138044


namespace NUMINAMATH_CALUDE_words_lost_oz_l1380_138098

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 67

/-- The number of words lost when prohibiting one letter in a language with only one or two-letter words --/
def words_lost (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

theorem words_lost_oz :
  words_lost num_letters = 135 := by
  sorry

end NUMINAMATH_CALUDE_words_lost_oz_l1380_138098


namespace NUMINAMATH_CALUDE_five_div_sqrt_five_times_one_over_sqrt_five_equals_one_l1380_138078

theorem five_div_sqrt_five_times_one_over_sqrt_five_equals_one :
  ∀ (sqrt_five : ℝ), sqrt_five > 0 → sqrt_five * sqrt_five = 5 →
  5 / sqrt_five * (1 / sqrt_five) = 1 := by
sorry

end NUMINAMATH_CALUDE_five_div_sqrt_five_times_one_over_sqrt_five_equals_one_l1380_138078


namespace NUMINAMATH_CALUDE_sum_congruence_l1380_138055

theorem sum_congruence : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l1380_138055


namespace NUMINAMATH_CALUDE_gcd_lcm_240_360_l1380_138026

theorem gcd_lcm_240_360 : 
  (Nat.gcd 240 360 = 120) ∧ (Nat.lcm 240 360 = 720) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_240_360_l1380_138026


namespace NUMINAMATH_CALUDE_min_additional_flights_for_40_percent_rate_l1380_138066

/-- Calculates the on-time rate given the number of on-time flights and total flights -/
def onTimeRate (onTime : ℕ) (total : ℕ) : ℚ :=
  onTime / total

/-- Represents the airport's flight departure scenario -/
structure AirportScenario where
  lateFlights : ℕ
  initialOnTimeFlights : ℕ
  additionalOnTimeFlights : ℕ

/-- Theorem: At least 1 additional on-time flight is needed for the on-time rate to exceed 40% -/
theorem min_additional_flights_for_40_percent_rate 
  (scenario : AirportScenario) 
  (h1 : scenario.lateFlights = 1)
  (h2 : scenario.initialOnTimeFlights = 3) :
  (∀ x : ℕ, x < 1 → 
    onTimeRate (scenario.initialOnTimeFlights + x) 
               (scenario.lateFlights + scenario.initialOnTimeFlights + x) ≤ 2/5) ∧
  (onTimeRate (scenario.initialOnTimeFlights + 1) 
              (scenario.lateFlights + scenario.initialOnTimeFlights + 1) > 2/5) :=
by sorry

end NUMINAMATH_CALUDE_min_additional_flights_for_40_percent_rate_l1380_138066


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_row20_l1380_138021

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem sum_fifth_sixth_row20 : 
  pascal_triangle 20 4 + pascal_triangle 20 5 = 20349 := by sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_row20_l1380_138021


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1380_138057

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : ∃ (M N Q : ℝ × ℝ),
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  M ∈ C ∧ N ∈ C ∧
  (N.1 - M.1) * 2 * c = 0 ∧
  (N.2 - M.2) * (F2.1 - F1.1) = 0 ∧
  (F2.1 - F1.1) = 4 * Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
  Q ∈ C ∧
  (Q.1 - F1.1)^2 + (Q.2 - F1.2)^2 = (N.1 - Q.1)^2 + (N.2 - Q.2)^2 →
  c^2 / a^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1380_138057


namespace NUMINAMATH_CALUDE_max_parts_two_planes_l1380_138056

-- Define a plane in 3D space
def Plane3D : Type := Unit

-- Define the possible relationships between two planes
inductive PlaneRelationship
| Parallel
| Intersecting

-- Function to calculate the number of parts based on the relationship
def numParts (rel : PlaneRelationship) : ℕ :=
  match rel with
  | PlaneRelationship.Parallel => 3
  | PlaneRelationship.Intersecting => 4

-- Theorem statement
theorem max_parts_two_planes (p1 p2 : Plane3D) :
  ∃ (rel : PlaneRelationship), ∀ (r : PlaneRelationship), numParts rel ≥ numParts r :=
sorry

end NUMINAMATH_CALUDE_max_parts_two_planes_l1380_138056


namespace NUMINAMATH_CALUDE_largest_percentage_increase_l1380_138029

def students : Fin 7 → ℕ
  | 0 => 80  -- 2010
  | 1 => 85  -- 2011
  | 2 => 88  -- 2012
  | 3 => 90  -- 2013
  | 4 => 95  -- 2014
  | 5 => 100 -- 2015
  | 6 => 120 -- 2016

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 6 := 5 -- Represents 2015 to 2016

theorem largest_percentage_increase :
  ∀ i : Fin 6, percentageIncrease (students i) (students (i.succ)) ≤ 
    percentageIncrease (students largestIncreaseYears) (students (largestIncreaseYears.succ)) := by
  sorry

end NUMINAMATH_CALUDE_largest_percentage_increase_l1380_138029


namespace NUMINAMATH_CALUDE_alex_walk_distance_l1380_138025

def south_movement : ℝ := 50 + 15
def north_movement : ℝ := 30
def west_movement : ℝ := 80
def east_movement : ℝ := 40

def net_south : ℝ := south_movement - north_movement
def net_west : ℝ := west_movement - east_movement

theorem alex_walk_distance : 
  ∃ (length_AB : ℝ), length_AB = (net_south^2 + net_west^2).sqrt :=
sorry

end NUMINAMATH_CALUDE_alex_walk_distance_l1380_138025


namespace NUMINAMATH_CALUDE_mason_hotdog_weight_l1380_138071

/-- Represents the weight of different food items in ounces -/
def food_weight : (String → Nat)
| "hotdog" => 2
| "burger" => 5
| "pie" => 10
| _ => 0

/-- The number of burgers Noah ate -/
def noah_burgers : Nat := 8

/-- Calculates the number of pies Jacob ate -/
def jacob_pies : Nat := noah_burgers - 3

/-- Calculates the number of hotdogs Mason ate -/
def mason_hotdogs : Nat := 3 * jacob_pies

/-- Theorem stating that the total weight of hotdogs Mason ate is 30 ounces -/
theorem mason_hotdog_weight :
  mason_hotdogs * food_weight "hotdog" = 30 := by
  sorry

end NUMINAMATH_CALUDE_mason_hotdog_weight_l1380_138071


namespace NUMINAMATH_CALUDE_sophia_ate_five_percent_l1380_138022

-- Define the percentages eaten by each person
def caden_percent : ℝ := 20
def zoe_percent : ℝ := caden_percent + 0.5 * caden_percent
def noah_percent : ℝ := zoe_percent + 0.5 * zoe_percent
def sophia_percent : ℝ := 100 - (caden_percent + zoe_percent + noah_percent)

-- Theorem statement
theorem sophia_ate_five_percent :
  sophia_percent = 5 := by
  sorry

end NUMINAMATH_CALUDE_sophia_ate_five_percent_l1380_138022


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1380_138042

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 6) 
  (eq2 : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1380_138042


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_2012021_5_l1380_138085

/-- The area of a quadrilateral with vertices at (1, 2), (1, 1), (4, 1), and (2009, 2010) -/
def quadrilateral_area : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (4, 1)
  let D : ℝ × ℝ := (2009, 2010)
  -- Area calculation goes here
  0 -- Placeholder

/-- Theorem stating that the area of the quadrilateral is 2012021.5 square units -/
theorem quadrilateral_area_is_2012021_5 : quadrilateral_area = 2012021.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_2012021_5_l1380_138085


namespace NUMINAMATH_CALUDE_hamiltonian_circuit_theorem_l1380_138077

/-- Represents a rectangle on a grid with unit cells -/
structure GridRectangle where
  m : ℕ  -- width
  n : ℕ  -- height

/-- Determines if a Hamiltonian circuit exists on the grid rectangle -/
def has_hamiltonian_circuit (rect : GridRectangle) : Prop :=
  rect.m > 0 ∧ rect.n > 0 ∧ (Odd rect.m ∨ Odd rect.n)

/-- Calculates the length of the Hamiltonian circuit when it exists -/
def hamiltonian_circuit_length (rect : GridRectangle) : ℕ :=
  (rect.m + 1) * (rect.n + 1)

theorem hamiltonian_circuit_theorem (rect : GridRectangle) :
  has_hamiltonian_circuit rect ↔
    ∃ (path_length : ℕ), 
      path_length = hamiltonian_circuit_length rect ∧
      path_length > 0 :=
sorry

end NUMINAMATH_CALUDE_hamiltonian_circuit_theorem_l1380_138077


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l1380_138009

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ m : ℕ, Nat.lcm x (Nat.lcm 15 21) = 105) → 
  x ≤ 105 ∧ 
  ∃ y : ℕ, y > 105 → ¬(∃ m : ℕ, Nat.lcm y (Nat.lcm 15 21) = 105) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l1380_138009


namespace NUMINAMATH_CALUDE_no_divisible_by_99_ab32_l1380_138054

theorem no_divisible_by_99_ab32 : ∀ a b : ℕ, 
  0 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 
  ¬(∃ k : ℕ, 1000 * a + 100 * b + 32 = 99 * k) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_99_ab32_l1380_138054


namespace NUMINAMATH_CALUDE_vector_problem_l1380_138041

/-- Given vectors AB and BC in R², prove that -1/2 * AC equals the specified vector. -/
theorem vector_problem (AB BC : ℝ × ℝ) (h1 : AB = (3, 7)) (h2 : BC = (-2, 3)) :
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  (-1/2 : ℝ) • AC = (-1/2, -5) := by sorry

end NUMINAMATH_CALUDE_vector_problem_l1380_138041


namespace NUMINAMATH_CALUDE_three_color_theorem_l1380_138060

theorem three_color_theorem (a b : ℕ) : 
  ∃ (f : ℤ → Fin 3), ∀ x : ℤ, f x ≠ f (x + a) ∧ f x ≠ f (x + b) := by
  sorry

end NUMINAMATH_CALUDE_three_color_theorem_l1380_138060


namespace NUMINAMATH_CALUDE_teal_sales_theorem_l1380_138013

/-- Represents the types of pies sold in the bakery -/
inductive PieType
| Pumpkin
| Custard

/-- Represents the properties of a pie -/
structure Pie where
  pieType : PieType
  slicesPerPie : Nat
  pricePerSlice : Nat
  piesCount : Nat

/-- Calculates the total sales for a given pie -/
def totalSales (pie : Pie) : Nat :=
  pie.slicesPerPie * pie.pricePerSlice * pie.piesCount

/-- Theorem: Teal's total sales from pumpkin and custard pies equal $340 -/
theorem teal_sales_theorem (pumpkinPie custardPie : Pie)
    (h_pumpkin : pumpkinPie.pieType = PieType.Pumpkin ∧
                 pumpkinPie.slicesPerPie = 8 ∧
                 pumpkinPie.pricePerSlice = 5 ∧
                 pumpkinPie.piesCount = 4)
    (h_custard : custardPie.pieType = PieType.Custard ∧
                 custardPie.slicesPerPie = 6 ∧
                 custardPie.pricePerSlice = 6 ∧
                 custardPie.piesCount = 5) :
    totalSales pumpkinPie + totalSales custardPie = 340 := by
  sorry

end NUMINAMATH_CALUDE_teal_sales_theorem_l1380_138013


namespace NUMINAMATH_CALUDE_square_difference_l1380_138028

theorem square_difference (n : ℕ) (h : (n + 1)^2 = n^2 + 2*n + 1) :
  (n - 1)^2 = n^2 - (2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_l1380_138028


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1380_138063

theorem solve_exponential_equation :
  ∃ x : ℝ, 4^x = Real.sqrt 64 ∧ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1380_138063


namespace NUMINAMATH_CALUDE_unique_perfect_square_Q_l1380_138000

/-- The polynomial Q(x) = x^4 + 6x^3 + 13x^2 + 3x - 19 -/
def Q (x : ℤ) : ℤ := x^4 + 6*x^3 + 13*x^2 + 3*x - 19

/-- A function that checks if a given integer is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, m^2 = n

/-- Theorem stating that there is exactly one integer x for which Q(x) is a perfect square -/
theorem unique_perfect_square_Q : ∃! x : ℤ, is_perfect_square (Q x) :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_Q_l1380_138000


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_primes_l1380_138070

theorem smallest_five_digit_divisible_by_primes : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
  n = 11550 :=
by sorry

#check smallest_five_digit_divisible_by_primes

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_primes_l1380_138070


namespace NUMINAMATH_CALUDE_career_preference_theorem_l1380_138006

/-- Represents the number of degrees in a circle that should be allocated to a career
    preference in a class with a given boy-to-girl ratio and preference ratios. -/
def career_preference_degrees (boy_ratio girl_ratio : ℚ) 
                               (boy_preference girl_preference : ℚ) : ℚ :=
  let total_parts := boy_ratio + girl_ratio
  let boy_parts := (boy_preference * boy_ratio) / total_parts
  let girl_parts := (girl_preference * girl_ratio) / total_parts
  let preference_fraction := (boy_parts + girl_parts) / 1
  preference_fraction * 360

/-- Theorem stating that for a class with a 2:3 ratio of boys to girls, 
    where 1/3 of boys and 2/3 of girls prefer a career, 
    192 degrees should be used to represent this preference in a circle graph. -/
theorem career_preference_theorem : 
  career_preference_degrees 2 3 (1/3) (2/3) = 192 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_theorem_l1380_138006


namespace NUMINAMATH_CALUDE_work_completion_time_l1380_138099

theorem work_completion_time (b a_and_b : ℝ) (hb : b = 12) (hab : a_and_b = 3) : 
  let a := (1 / a_and_b - 1 / b)⁻¹
  a = 4 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1380_138099


namespace NUMINAMATH_CALUDE_curved_octagon_area_l1380_138003

/-- A closed curve composed of circular arcs centered on an octagon's vertices -/
structure CurvedOctagon where
  /-- Number of circular arcs -/
  n_arcs : ℕ
  /-- Length of each circular arc -/
  arc_length : ℝ
  /-- Side length of the regular octagon -/
  octagon_side : ℝ

/-- The area enclosed by the curved octagon -/
noncomputable def enclosed_area (co : CurvedOctagon) : ℝ :=
  sorry

/-- Theorem stating the enclosed area of a specific curved octagon -/
theorem curved_octagon_area :
  let co : CurvedOctagon := {
    n_arcs := 12,
    arc_length := 3 * Real.pi / 4,
    octagon_side := 3
  }
  enclosed_area co = 18 * (1 + Real.sqrt 2) + 81 * Real.pi / 8 :=
sorry

end NUMINAMATH_CALUDE_curved_octagon_area_l1380_138003


namespace NUMINAMATH_CALUDE_elisa_lap_time_improvement_l1380_138035

/-- Calculates the improvement in lap time given current and previous swimming performance -/
def lap_time_improvement (current_laps : ℕ) (current_time : ℕ) (previous_laps : ℕ) (previous_time : ℕ) : ℚ :=
  (previous_time : ℚ) / (previous_laps : ℚ) - (current_time : ℚ) / (current_laps : ℚ)

/-- Proves that Elisa's lap time improvement is 0.5 minutes per lap -/
theorem elisa_lap_time_improvement :
  lap_time_improvement 15 30 20 50 = 1/2 := by sorry

end NUMINAMATH_CALUDE_elisa_lap_time_improvement_l1380_138035


namespace NUMINAMATH_CALUDE_power_function_through_one_l1380_138061

theorem power_function_through_one (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^a
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_through_one_l1380_138061


namespace NUMINAMATH_CALUDE_estimated_y_at_25_l1380_138032

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.5 * x - 0.81

/-- Theorem: The estimated value of y is 11.69 when x = 25 -/
theorem estimated_y_at_25 : linear_regression 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_at_25_l1380_138032


namespace NUMINAMATH_CALUDE_M_divisible_by_51_l1380_138007

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Definition of the function that concatenates numbers from 1 to n
  sorry

theorem M_divisible_by_51 :
  ∃ M : ℕ, M = concatenate_numbers 50 ∧ M % 51 = 0 :=
sorry

end NUMINAMATH_CALUDE_M_divisible_by_51_l1380_138007


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1380_138018

theorem inequality_solution_set :
  {x : ℝ | x - 3 / x > 2} = {x : ℝ | -1 < x ∧ x < 0 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1380_138018


namespace NUMINAMATH_CALUDE_primality_extension_l1380_138039

theorem primality_extension (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n)) :=
by sorry

end NUMINAMATH_CALUDE_primality_extension_l1380_138039


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l1380_138058

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := b^2 - a*b

-- Theorem statement
theorem equation_has_two_distinct_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ otimes (m - 2) x₁ = m ∧ otimes (m - 2) x₂ = m :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l1380_138058


namespace NUMINAMATH_CALUDE_min_value_T_l1380_138086

theorem min_value_T (a b c : ℝ) 
  (h1 : ∀ x : ℝ, 1/a * x^2 + 6*x + c ≥ 0)
  (h2 : a*b > 1)
  (h3 : ∃ x : ℝ, 1/a * x^2 + 6*x + c = 0) :
  1/(2*(a*b - 1)) + a*(b + 2*c)/(a*b - 1) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_T_l1380_138086


namespace NUMINAMATH_CALUDE_problem_statement_l1380_138064

theorem problem_statement (a b : ℝ) (h : a + b - 3 = 0) :
  2*a^2 + 4*a*b + 2*b^2 - 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1380_138064


namespace NUMINAMATH_CALUDE_sterling_total_questions_l1380_138068

/-- Represents the candy reward system and Sterling's performance --/
structure CandyReward where
  correct_reward : ℕ
  incorrect_penalty : ℕ
  correct_answers : ℕ
  total_questions : ℕ
  hypothetical_candy : ℕ

/-- Theorem stating that Sterling answered 9 questions in total --/
theorem sterling_total_questions 
  (reward : CandyReward) 
  (h1 : reward.correct_reward = 3)
  (h2 : reward.incorrect_penalty = 2)
  (h3 : reward.correct_answers = 7)
  (h4 : reward.hypothetical_candy = 31)
  (h5 : reward.hypothetical_candy = 
    (reward.correct_answers + 2) * reward.correct_reward - 
    (reward.total_questions - reward.correct_answers - 2) * reward.incorrect_penalty) : 
  reward.total_questions = 9 := by
  sorry

end NUMINAMATH_CALUDE_sterling_total_questions_l1380_138068


namespace NUMINAMATH_CALUDE_jasper_chip_sales_l1380_138024

/-- Given the conditions of Jasper's sales, prove that he sold 27 bags of chips. -/
theorem jasper_chip_sales :
  ∀ (chips hotdogs drinks : ℕ),
    hotdogs = chips - 8 →
    drinks = hotdogs + 12 →
    drinks = 31 →
    chips = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_jasper_chip_sales_l1380_138024


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1380_138017

theorem inequality_equivalence (x : ℝ) :
  |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1380_138017


namespace NUMINAMATH_CALUDE_provider_selection_ways_l1380_138001

def total_providers : ℕ := 25
def s_providers : ℕ := 6
def num_siblings : ℕ := 4

theorem provider_selection_ways : 
  (total_providers * s_providers * (total_providers - 2) * (total_providers - 3) = 75900) := by
  sorry

end NUMINAMATH_CALUDE_provider_selection_ways_l1380_138001


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1380_138090

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos α - Real.sin α = -Real.sqrt 5 / 5) : 
  (Real.sin α * Real.cos α = 2 / 5) ∧ 
  (Real.sin α + Real.cos α = 3 * Real.sqrt 5 / 5) ∧ 
  ((2 * Real.sin α * Real.cos α - Real.cos α + 1) / (1 - Real.tan α) = (-9 + Real.sqrt 5) / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1380_138090


namespace NUMINAMATH_CALUDE_symmetric_probability_is_one_over_429_l1380_138092

/-- Represents a coloring of the 13-square array -/
def Coloring := Fin 13 → Bool

/-- The total number of squares in the array -/
def totalSquares : ℕ := 13

/-- The number of red squares -/
def redSquares : ℕ := 8

/-- The number of blue squares -/
def blueSquares : ℕ := 5

/-- Predicate to check if a coloring is symmetric under 90-degree rotation -/
def isSymmetric (c : Coloring) : Prop := sorry

/-- The number of symmetric colorings -/
def symmetricColorings : ℕ := 3

/-- The total number of possible colorings -/
def totalColorings : ℕ := Nat.choose totalSquares blueSquares

/-- The probability of selecting a symmetric coloring -/
def symmetricProbability : ℚ := symmetricColorings / totalColorings

theorem symmetric_probability_is_one_over_429 : 
  symmetricProbability = 1 / 429 := by sorry

end NUMINAMATH_CALUDE_symmetric_probability_is_one_over_429_l1380_138092


namespace NUMINAMATH_CALUDE_constant_expression_l1380_138051

theorem constant_expression (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hsum : x + y = 1) :
  x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l1380_138051


namespace NUMINAMATH_CALUDE_edward_earnings_l1380_138046

/-- Calculate Edward's earnings for a week --/
theorem edward_earnings (regular_rate : ℝ) (regular_hours overtime_hours : ℕ) : 
  regular_rate = 7 →
  regular_hours = 40 →
  overtime_hours = 5 →
  let overtime_rate := 2 * regular_rate
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings = 350 := by sorry

end NUMINAMATH_CALUDE_edward_earnings_l1380_138046


namespace NUMINAMATH_CALUDE_find_x_and_y_l1380_138005

theorem find_x_and_y :
  ∀ x y : ℝ,
  x > y →
  x + y = 55 →
  x - y = 15 →
  x = 35 ∧ y = 20 := by
sorry

end NUMINAMATH_CALUDE_find_x_and_y_l1380_138005


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l1380_138050

theorem lcm_of_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l1380_138050


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1380_138031

theorem smallest_multiple_of_6_and_15 (c : ℕ) :
  (c > 0 ∧ 6 ∣ c ∧ 15 ∣ c) → c ≥ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1380_138031


namespace NUMINAMATH_CALUDE_rope_length_proof_l1380_138093

/-- The length of the rope in meters -/
def rope_length : ℝ := 1.15

/-- The fraction of the rope that was used -/
def used_fraction : ℝ := 0.4

/-- The remaining length of the rope in meters -/
def remaining_length : ℝ := 0.69

theorem rope_length_proof : 
  rope_length * (1 - used_fraction) = remaining_length :=
by sorry

end NUMINAMATH_CALUDE_rope_length_proof_l1380_138093


namespace NUMINAMATH_CALUDE_diamond_three_eight_l1380_138020

-- Define the ◇ operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y + 2

-- Theorem statement
theorem diamond_three_eight : diamond 3 8 = 62 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_eight_l1380_138020


namespace NUMINAMATH_CALUDE_jasons_games_this_month_l1380_138075

/-- 
Given that:
- Jason went to 17 games last month
- Jason plans to go to 16 games next month
- Jason will attend 44 games in all

Prove that Jason went to 11 games this month.
-/
theorem jasons_games_this_month 
  (games_last_month : ℕ) 
  (games_next_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44) :
  total_games - (games_last_month + games_next_month) = 11 := by
  sorry

end NUMINAMATH_CALUDE_jasons_games_this_month_l1380_138075


namespace NUMINAMATH_CALUDE_job_completion_time_l1380_138073

/-- The time taken by two workers to complete a job together, given their relative efficiencies and the time taken by one worker alone. -/
theorem job_completion_time 
  (efficiency_a : ℝ) 
  (efficiency_b : ℝ) 
  (time_a_alone : ℝ) 
  (h1 : efficiency_a = efficiency_b + 0.6 * efficiency_b) 
  (h2 : time_a_alone = 35) 
  : (1 / (1 / time_a_alone + efficiency_b / (efficiency_a * time_a_alone))) = 25 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1380_138073


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1380_138052

theorem pentagon_square_side_ratio :
  ∀ (p s : ℝ),
  p > 0 → s > 0 →
  5 * p = 20 →
  4 * s = 20 →
  p / s = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1380_138052


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1380_138076

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 8) (h₃ : a₃ = 13) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 148 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1380_138076


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1380_138014

/-- Simple interest calculation -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  interest = 4016.25 →
  rate = 11 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 7302.27 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1380_138014


namespace NUMINAMATH_CALUDE_row_time_ratio_l1380_138011

/-- Proves the ratio of time taken to row up and down a river -/
theorem row_time_ratio (man_speed : ℝ) (stream_speed : ℝ)
  (h1 : man_speed = 24)
  (h2 : stream_speed = 12) :
  (man_speed - stream_speed) / (man_speed + stream_speed) = 1 / 3 := by
  sorry

#check row_time_ratio

end NUMINAMATH_CALUDE_row_time_ratio_l1380_138011


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1380_138089

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  9 * x^2 + 36 * x + 4 * y^2 - 8 * y + 20 = 0

/-- The distance between the foci of the ellipse -/
def foci_distance : ℝ := 0

/-- Theorem stating that the distance between the foci of the given ellipse is 0 -/
theorem ellipse_foci_distance :
  ∀ x y : ℝ, ellipse_equation x y → foci_distance = 0 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1380_138089


namespace NUMINAMATH_CALUDE_parabola_vertex_l1380_138030

def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 8 = 0

def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x' y' : ℝ), eq x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

theorem parabola_vertex :
  is_vertex (-2) 2 parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1380_138030


namespace NUMINAMATH_CALUDE_triangle_properties_l1380_138065

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  2 * c * Real.sin A = Real.sqrt 3 * a →
  b = 2 →
  c = Real.sqrt 7 →
  C = π/3 ∧ (1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1380_138065


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l1380_138037

/-- A five-digit number is a natural number between 10000 and 99999 inclusive. -/
def FiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The property that defines our target number. -/
def SatisfiesCondition (x : ℕ) : Prop :=
  FiveDigitNumber x ∧ 7 * 10^5 + x = 5 * (10 * x + 7)

theorem unique_five_digit_number : 
  ∃! x : ℕ, SatisfiesCondition x ∧ x = 14285 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l1380_138037


namespace NUMINAMATH_CALUDE_sticker_remainder_l1380_138047

theorem sticker_remainder (a b c : ℤ) 
  (ha : a % 5 = 1)
  (hb : b % 5 = 4)
  (hc : c % 5 = 3) : 
  (a + b + c) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sticker_remainder_l1380_138047


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l1380_138036

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_part_of_product : Complex.im ((1 - 2*i) * i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l1380_138036


namespace NUMINAMATH_CALUDE_normal_curve_properties_l1380_138034

/- Normal distribution density function -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

theorem normal_curve_properties (μ σ : ℝ) (h : σ > 0) :
  /- 1. Symmetry about x = μ -/
  (∀ x : ℝ, normal_pdf μ σ (μ + x) = normal_pdf μ σ (μ - x)) ∧
  /- 2. Always positive -/
  (∀ x : ℝ, normal_pdf μ σ x > 0) ∧
  /- 3. Global maximum at x = μ -/
  (∀ x : ℝ, normal_pdf μ σ x ≤ normal_pdf μ σ μ) ∧
  /- 4. Axis of symmetry is x = μ -/
  (∀ x : ℝ, normal_pdf μ σ (μ + x) = normal_pdf μ σ (μ - x)) ∧
  /- 5. σ determines the spread -/
  (∀ σ₁ σ₂ : ℝ, σ₁ ≠ σ₂ → ∃ x : ℝ, normal_pdf μ σ₁ x ≠ normal_pdf μ σ₂ x) ∧
  /- 6. Larger σ, flatter and wider curve -/
  (∀ σ₁ σ₂ : ℝ, σ₁ > σ₂ → ∃ x : ℝ, x ≠ μ ∧ normal_pdf μ σ₁ x > normal_pdf μ σ₂ x) ∧
  /- 7. Smaller σ, taller and slimmer curve -/
  (∀ σ₁ σ₂ : ℝ, σ₁ < σ₂ → normal_pdf μ σ₁ μ > normal_pdf μ σ₂ μ) :=
by sorry

end NUMINAMATH_CALUDE_normal_curve_properties_l1380_138034


namespace NUMINAMATH_CALUDE_sum_squares_ge_sum_products_l1380_138008

theorem sum_squares_ge_sum_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_ge_sum_products_l1380_138008


namespace NUMINAMATH_CALUDE_inequality_preservation_l1380_138016

theorem inequality_preservation (a b : ℝ) : a < b → 1 - a > 1 - b := by sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1380_138016


namespace NUMINAMATH_CALUDE_sphere_volume_surface_ratio_l1380_138027

/-- The ratio of volume to surface area of a sphere with an inscribed regular hexagon -/
theorem sphere_volume_surface_ratio (area_hexagon : ℝ) (distance : ℝ) :
  area_hexagon = 3 * Real.sqrt 3 / 2 →
  distance = 2 * Real.sqrt 2 →
  ∃ (V S : ℝ), V / S = 1 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_ratio_l1380_138027


namespace NUMINAMATH_CALUDE_garden_breadth_l1380_138015

/-- The perimeter of a rectangle given its length and breadth -/
def perimeter (length breadth : ℝ) : ℝ := 2 * (length + breadth)

/-- Theorem: For a rectangular garden with perimeter 500 m and length 150 m, the breadth is 100 m -/
theorem garden_breadth :
  ∃ (breadth : ℝ), perimeter 150 breadth = 500 ∧ breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l1380_138015


namespace NUMINAMATH_CALUDE_unique_solution_from_distinct_points_l1380_138097

/-- Given two distinct points on a line, the system of equations formed by these points always has a unique solution -/
theorem unique_solution_from_distinct_points 
  (k : ℝ) (a b₁ b₂ a₁ a₂ : ℝ) :
  (a ≠ 2) →  -- P₁ and P₂ are distinct
  (b₁ = k * a + 1) →  -- P₁ is on the line
  (b₂ = k * 2 + 1) →  -- P₂ is on the line
  ∃! (x y : ℝ), a₁ * x + b₁ * y = 1 ∧ a₂ * x + b₂ * y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_from_distinct_points_l1380_138097


namespace NUMINAMATH_CALUDE_baseball_glove_discount_percentage_l1380_138074

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_cleats_price : ℝ := 10
def baseball_cleats_pairs : ℕ := 2
def total_amount : ℝ := 79

theorem baseball_glove_discount_percentage :
  let other_items_total : ℝ := baseball_cards_price + baseball_bat_price + baseball_cleats_price * baseball_cleats_pairs
  let glove_sale_price : ℝ := total_amount - other_items_total
  let discount_percentage : ℝ := (baseball_glove_original_price - glove_sale_price) / baseball_glove_original_price * 100
  discount_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_baseball_glove_discount_percentage_l1380_138074


namespace NUMINAMATH_CALUDE_car_count_on_river_road_l1380_138043

theorem car_count_on_river_road (buses cars bikes : ℕ) : 
  (3 : ℕ) * cars = 7 * buses →
  (3 : ℕ) * bikes = 10 * buses →
  cars = buses + 90 →
  bikes = buses + 140 →
  cars = 150 := by
sorry

end NUMINAMATH_CALUDE_car_count_on_river_road_l1380_138043


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1380_138019

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Theorem statement
theorem quadratic_function_properties :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ f x) ∧ -- Maximum value is 2
  (∃ (x : ℝ), f x = x + 1) ∧ -- Vertex lies on y = x + 1
  (f 3 = -2) ∧ -- Passes through (3, -2)
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 →
    f x ≤ 2 ∧ -- Maximum value in [0, 3] is 2
    f x ≥ -2 ∧ -- Minimum value in [0, 3] is -2
    (f x = 2 → x = 1) ∧ -- Maximum occurs at x = 1
    (f x = -2 → x = 3)) -- Minimum occurs at x = 3
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1380_138019


namespace NUMINAMATH_CALUDE_app_total_cost_l1380_138059

/-- Calculates the total cost of an app with online access -/
def total_cost (initial_price : ℕ) (monthly_fee : ℕ) (months : ℕ) : ℕ :=
  initial_price + monthly_fee * months

/-- Proves that the total cost for the given conditions is $21 -/
theorem app_total_cost : total_cost 5 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_app_total_cost_l1380_138059
