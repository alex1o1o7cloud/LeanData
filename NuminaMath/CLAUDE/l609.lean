import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_relation_l609_60941

theorem quadratic_root_relation (p : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + 3 = 0 ∧ y^2 + p*y + 3 = 0 ∧ y = 3*x) → 
  (p = 4 ∨ p = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l609_60941


namespace NUMINAMATH_CALUDE_parabolas_imply_right_triangle_l609_60944

/-- Two parabolas intersecting the x-axis at the same non-origin point -/
def intersecting_parabolas (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0

/-- The triangle formed by sides a, b, and c is right-angled -/
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

theorem parabolas_imply_right_triangle (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_neq : a ≠ c) 
  (h_intersect : intersecting_parabolas a b c) : 
  right_angled_triangle a b c := by
  sorry

end NUMINAMATH_CALUDE_parabolas_imply_right_triangle_l609_60944


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l609_60931

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l609_60931


namespace NUMINAMATH_CALUDE_pool_water_calculation_l609_60953

/-- Calculates the amount of water in a pool after five hours of filling and a leak -/
def water_in_pool (rate1 : ℕ) (rate2 : ℕ) (rate3 : ℕ) (leak : ℕ) : ℕ :=
  rate1 + 2 * rate2 + rate3 - leak

theorem pool_water_calculation :
  water_in_pool 8 10 14 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_pool_water_calculation_l609_60953


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l609_60996

theorem system_of_equations_solutions :
  (∃ x y : ℚ, x + y = 3 ∧ x - y = 1 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℚ, 2*x + y = 3 ∧ x - 2*y = 1 ∧ x = 7/5 ∧ y = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l609_60996


namespace NUMINAMATH_CALUDE_shaded_area_constant_l609_60909

/-- The total area of two triangles formed by joining the ends of two 1 cm segments 
    on opposite sides of an 8 cm square is always 4 cm², regardless of the segments' positions. -/
theorem shaded_area_constant (h : ℝ) (h_range : 0 ≤ h ∧ h ≤ 8) : 
  (1/2 * 1 * h) + (1/2 * 1 * (8 - h)) = 4 := by sorry

end NUMINAMATH_CALUDE_shaded_area_constant_l609_60909


namespace NUMINAMATH_CALUDE_two_digit_number_ratio_l609_60962

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  tens_valid : tens ≥ 1 ∧ tens ≤ 9
  units_valid : units ≤ 9

def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.tens + n.units

def TwoDigitNumber.interchanged (n : TwoDigitNumber) : ℕ :=
  10 * n.units + n.tens

theorem two_digit_number_ratio (n : TwoDigitNumber) 
  (h1 : n.value - n.interchanged = 36)
  (h2 : (n.tens + n.units) - (n.tens - n.units) = 8) :
  n.tens = 2 * n.units :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_ratio_l609_60962


namespace NUMINAMATH_CALUDE_largest_angle_measure_l609_60981

/-- A triangle XYZ is obtuse and isosceles with one of the equal angles measuring 30 degrees. -/
structure ObtuseIsoscelesTriangle where
  X : ℝ
  Y : ℝ
  Z : ℝ
  sum_180 : X + Y + Z = 180
  obtuse : Z > 90
  isosceles : X = Y
  x_measure : X = 30

/-- The largest interior angle of an obtuse isosceles triangle with one equal angle measuring 30 degrees is 120 degrees. -/
theorem largest_angle_measure (t : ObtuseIsoscelesTriangle) : t.Z = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_measure_l609_60981


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l609_60917

def quadratic_function (a b x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_function_properties (a b : ℝ) :
  -- Part 1: Range on [-1,1] when b = 1
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, quadratic_function a 1 x ∈ 
    if a ≤ -2 then Set.Icc a (-a)
    else if 0 < a ∧ a ≤ 2 then Set.Icc (1 - a^2/4) (2-a)
    else if -2 < a ∧ a ≤ 0 then Set.Icc (1 - a^2/4) (2+a)
    else Set.Icc (-a) a) ∧
  -- Part 2: Existence of k when roots are between consecutive integers
  (∃ x₁ x₂ m : ℝ, x₁ < x₂ ∧ ∃ k : ℤ, m = ↑k ∧ m < x₁ ∧ x₂ < m + 1 ∧
    quadratic_function a b x₁ = 0 ∧ quadratic_function a b x₂ = 0 →
    ∃ k : ℤ, |quadratic_function a b ↑k| ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l609_60917


namespace NUMINAMATH_CALUDE_product_of_roots_l609_60957

theorem product_of_roots (x : ℝ) : 
  (x^2 - 4*x - 42 = 0) → 
  ∃ y : ℝ, (y^2 - 4*y - 42 = 0) ∧ (x * y = -42) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l609_60957


namespace NUMINAMATH_CALUDE_larger_number_proof_l609_60911

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1515) (h3 : L = 16 * S + 15) : L = 1617 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l609_60911


namespace NUMINAMATH_CALUDE_square_plot_area_l609_60982

/-- Given a square plot with a fence, prove that the area is 289 square feet
    when the price per foot is 54 and the total cost is 3672. -/
theorem square_plot_area (side_length : ℝ) : 
  side_length > 0 →
  (4 * side_length * 54 = 3672) →
  side_length^2 = 289 := by
  sorry


end NUMINAMATH_CALUDE_square_plot_area_l609_60982


namespace NUMINAMATH_CALUDE_exists_set_without_triangle_l609_60945

/-- A set of 10 segment lengths --/
def SegmentSet : Type := Fin 10 → ℝ

/-- Predicate to check if three segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Theorem stating that there exists a set of 10 segments where no three can form a triangle --/
theorem exists_set_without_triangle : 
  ∃ (s : SegmentSet), ∀ (i j k : Fin 10), i ≠ j → j ≠ k → i ≠ k → 
    ¬(can_form_triangle (s i) (s j) (s k)) := by
  sorry

end NUMINAMATH_CALUDE_exists_set_without_triangle_l609_60945


namespace NUMINAMATH_CALUDE_distinct_odd_numbers_count_l609_60983

-- Define the given number as a list of digits
def given_number : List Nat := [3, 4, 3, 9, 6]

-- Function to check if a number is odd
def is_odd (n : Nat) : Bool :=
  n % 2 = 1

-- Function to count distinct permutations
def count_distinct_permutations (digits : List Nat) : Nat :=
  sorry

-- Function to count distinct odd permutations
def count_distinct_odd_permutations (digits : List Nat) : Nat :=
  sorry

-- Theorem statement
theorem distinct_odd_numbers_count :
  count_distinct_odd_permutations given_number = 36 := by
  sorry

end NUMINAMATH_CALUDE_distinct_odd_numbers_count_l609_60983


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l609_60973

/-- The motion equation of a ball rolling down an inclined plane -/
def motion_equation (t : ℝ) : ℝ := t^2

/-- The velocity function derived from the motion equation -/
def velocity (t : ℝ) : ℝ := 2 * t

theorem instantaneous_velocity_at_5 : velocity 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l609_60973


namespace NUMINAMATH_CALUDE_binary_1011001_to_base6_l609_60950

/-- Converts a binary (base-2) number to its decimal (base-10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base-10) number to its base-6 representation -/
def decimal_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The binary representation of 1011001 -/
def binary_1011001 : List Bool := [true, false, false, true, true, false, true]

theorem binary_1011001_to_base6 :
  decimal_to_base6 (binary_to_decimal binary_1011001) = [2, 2, 5] :=
sorry

end NUMINAMATH_CALUDE_binary_1011001_to_base6_l609_60950


namespace NUMINAMATH_CALUDE_ethans_rowing_time_l609_60902

/-- Proves that Ethan's rowing time is 25 minutes given the conditions -/
theorem ethans_rowing_time (total_time : ℕ) (ethan_time : ℕ) :
  total_time = 75 →
  total_time = ethan_time + 2 * ethan_time →
  ethan_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_ethans_rowing_time_l609_60902


namespace NUMINAMATH_CALUDE_probability_all_girls_l609_60914

def total_members : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5
def chosen_members : ℕ := 3

theorem probability_all_girls :
  (Nat.choose num_girls chosen_members : ℚ) / (Nat.choose total_members chosen_members) = 1 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_girls_l609_60914


namespace NUMINAMATH_CALUDE_cubic_roots_same_abs_value_iff_l609_60946

-- Define the polynomial type
def CubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

-- Define the property that all roots have the same absolute value
def AllRootsSameAbsValue (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ z : ℂ, f z.re = 0 → Complex.abs z = k

-- Theorem statement
theorem cubic_roots_same_abs_value_iff (a b c : ℝ) :
  AllRootsSameAbsValue (CubicPolynomial a b c) → (a = 0 ↔ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_same_abs_value_iff_l609_60946


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l609_60938

/-- Given that x^4 - x^3 + x^2 + ax + b is a perfect square polynomial, prove that b = 9/64 -/
theorem perfect_square_polynomial (a b : ℝ) : 
  (∃ p q r : ℝ, ∀ x : ℝ, x^4 - x^3 + x^2 + a*x + b = (p*x^2 + q*x + r)^2) →
  b = 9/64 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l609_60938


namespace NUMINAMATH_CALUDE_polynomial_multiplication_correction_l609_60948

theorem polynomial_multiplication_correction (x a b : ℚ) : 
  (2*x-a)*(3*x+b) = 6*x^2 + 11*x - 10 →
  (2*x+a)*(x+b) = 2*x^2 - 9*x + 10 →
  (2*x+a)*(3*x+b) = 6*x^2 - 19*x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_correction_l609_60948


namespace NUMINAMATH_CALUDE_f_composition_of_three_l609_60964

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_composition_of_three : f (f (f (f 3))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l609_60964


namespace NUMINAMATH_CALUDE_four_digit_number_property_l609_60924

theorem four_digit_number_property (m : ℕ) : 
  1000 ≤ m ∧ m ≤ 2025 →
  ∃ (n : ℕ), n > 0 ∧ Nat.Prime (m - n) ∧ ∃ (k : ℕ), m * n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_property_l609_60924


namespace NUMINAMATH_CALUDE_two_colored_line_exists_l609_60963

-- Define the color type
inductive Color
| Red
| Blue
| Green
| Yellow

-- Define the grid
def Grid := ℤ × ℤ → Color

-- Define the property that vertices of any 1x1 square are painted in different colors
def ValidColoring (g : Grid) : Prop :=
  ∀ x y : ℤ, 
    g (x, y) ≠ g (x + 1, y) ∧
    g (x, y) ≠ g (x, y + 1) ∧
    g (x, y) ≠ g (x + 1, y + 1) ∧
    g (x + 1, y) ≠ g (x, y + 1) ∧
    g (x + 1, y) ≠ g (x + 1, y + 1) ∧
    g (x, y + 1) ≠ g (x + 1, y + 1)

-- Define a line in the grid
def Line := ℤ → ℤ × ℤ

-- Define the property that a line has nodes painted in exactly two colors
def TwoColoredLine (g : Grid) (l : Line) : Prop :=
  ∃ c1 c2 : Color, c1 ≠ c2 ∧ ∀ z : ℤ, g (l z) = c1 ∨ g (l z) = c2

-- The main theorem
theorem two_colored_line_exists (g : Grid) (h : ValidColoring g) : 
  ∃ l : Line, TwoColoredLine g l := by
  sorry

end NUMINAMATH_CALUDE_two_colored_line_exists_l609_60963


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_103_not_six_digit_palindrome_l609_60971

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a six-digit palindrome -/
def isSixDigitPalindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ 
  (n / 100000 = n % 10) ∧ 
  ((n / 10000) % 10 = (n / 10) % 10) ∧
  ((n / 1000) % 10 = (n / 100) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_times_103_not_six_digit_palindrome :
  isThreeDigitPalindrome 131 ∧
  ¬(isSixDigitPalindrome (131 * 103)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 131 → isSixDigitPalindrome (n * 103) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_103_not_six_digit_palindrome_l609_60971


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l609_60956

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 4) : 
  ∃ (diagonal : ℝ), diagonal = 4 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l609_60956


namespace NUMINAMATH_CALUDE_triangle_interior_angle_l609_60994

theorem triangle_interior_angle (a b : ℝ) (ha : a = 110) (hb : b = 120) : 
  ∃ x : ℝ, x = 50 ∧ x + (360 - (a + b)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_angle_l609_60994


namespace NUMINAMATH_CALUDE_number_of_refills_l609_60900

def total_spent : ℕ := 63
def cost_per_refill : ℕ := 21

theorem number_of_refills : total_spent / cost_per_refill = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_refills_l609_60900


namespace NUMINAMATH_CALUDE_evaluate_expression_l609_60920

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) : y * (2 * y - x) = 24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l609_60920


namespace NUMINAMATH_CALUDE_gcd_123456_789012_l609_60954

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_123456_789012_l609_60954


namespace NUMINAMATH_CALUDE_total_cost_calculation_l609_60926

def silverware_cost : ℝ := 20
def plate_cost_ratio : ℝ := 0.5

theorem total_cost_calculation :
  let plate_cost := plate_cost_ratio * silverware_cost
  silverware_cost + plate_cost = 30 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l609_60926


namespace NUMINAMATH_CALUDE_a_55_mod_45_l609_60929

/-- Definition of a_n as a function that concatenates integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- The remainder when a_55 is divided by 45 is 10 -/
theorem a_55_mod_45 : a 55 % 45 = 10 := by sorry

end NUMINAMATH_CALUDE_a_55_mod_45_l609_60929


namespace NUMINAMATH_CALUDE_certain_number_proof_l609_60992

theorem certain_number_proof (x : ℝ) : (15 * x) / 100 = 0.04863 → x = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l609_60992


namespace NUMINAMATH_CALUDE_encyclopedia_pages_l609_60928

/-- The Encyclopedia of Life and Everything Else --/
structure Encyclopedia where
  chapters : Nat
  pages_per_chapter : Nat

/-- Calculate the total number of pages in the encyclopedia --/
def total_pages (e : Encyclopedia) : Nat :=
  e.chapters * e.pages_per_chapter

/-- Theorem: The encyclopedia has 9384 pages in total --/
theorem encyclopedia_pages :
  ∃ (e : Encyclopedia), e.chapters = 12 ∧ e.pages_per_chapter = 782 ∧ total_pages e = 9384 := by
  sorry

end NUMINAMATH_CALUDE_encyclopedia_pages_l609_60928


namespace NUMINAMATH_CALUDE_powers_of_two_in_arithmetic_sequence_l609_60934

theorem powers_of_two_in_arithmetic_sequence (k : ℕ) :
  (∃ n : ℕ, 2^k = 6*n + 8) ↔ (k > 1 ∧ k % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_powers_of_two_in_arithmetic_sequence_l609_60934


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l609_60949

theorem complex_number_magnitude (z : ℂ) : z = 2 / (1 - Complex.I) + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l609_60949


namespace NUMINAMATH_CALUDE_train_speed_calculation_l609_60916

/-- Given a train with length 280.0224 meters that crosses a post in 25.2 seconds, 
    its speed is 40.0032 km/hr. -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) 
    (h1 : train_length = 280.0224) 
    (h2 : crossing_time = 25.2) : 
  (train_length / 1000) / (crossing_time / 3600) = 40.0032 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l609_60916


namespace NUMINAMATH_CALUDE_carpet_cost_per_meter_l609_60918

/-- Calculates the cost per meter of carpet in paise given room dimensions, carpet width, and total cost --/
theorem carpet_cost_per_meter 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (carpet_width_cm : ℝ) 
  (total_cost_rupees : ℝ) 
  (h1 : room_length = 15)
  (h2 : room_width = 6)
  (h3 : carpet_width_cm = 75)
  (h4 : total_cost_rupees = 36) : 
  (total_cost_rupees * 100) / (room_length * room_width / (carpet_width_cm / 100)) = 30 := by
  sorry

#check carpet_cost_per_meter

end NUMINAMATH_CALUDE_carpet_cost_per_meter_l609_60918


namespace NUMINAMATH_CALUDE_map_distance_example_l609_60990

/-- Given a map scale and an actual distance, calculates the distance on the map -/
def map_distance (scale : ℚ) (actual_distance : ℚ) : ℚ :=
  actual_distance * scale

/-- Theorem: For a map with scale 1:5000000 and actual distance 400km, the map distance is 8cm -/
theorem map_distance_example : 
  let scale : ℚ := 1 / 5000000
  let actual_distance : ℚ := 400 * 100000  -- 400km in cm
  map_distance scale actual_distance = 8 := by
  sorry

#eval map_distance (1 / 5000000) (400 * 100000)

end NUMINAMATH_CALUDE_map_distance_example_l609_60990


namespace NUMINAMATH_CALUDE_bus_ride_difference_l609_60932

theorem bus_ride_difference (vince_ride : ℝ) (zachary_ride : ℝ)
  (h1 : vince_ride = 0.625)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.125 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l609_60932


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l609_60993

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l609_60993


namespace NUMINAMATH_CALUDE_gumball_calculation_l609_60980

/-- The number of gumballs originally in the dispenser -/
def original_gumballs : ℝ := 100

/-- The fraction of gumballs remaining after each day -/
def daily_remaining_fraction : ℝ := 0.7

/-- The number of days that have passed -/
def days : ℕ := 3

/-- The number of gumballs remaining after 3 days -/
def remaining_gumballs : ℝ := 34.3

/-- Theorem stating that the original number of gumballs is correct -/
theorem gumball_calculation :
  original_gumballs * daily_remaining_fraction ^ days = remaining_gumballs := by
  sorry

end NUMINAMATH_CALUDE_gumball_calculation_l609_60980


namespace NUMINAMATH_CALUDE_min_value_of_arithmetic_sequence_l609_60937

/-- An arithmetic sequence of positive terms -/
def ArithmeticSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ+, a (n + 1) = a n + q

theorem min_value_of_arithmetic_sequence (a : ℕ+ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_2018 : a 2018 = a 2017 + 2 * a 2016)
    (h_exist : ∃ m n : ℕ+, Real.sqrt (a m * a n) = 4 * a 1) :
    (∃ m n : ℕ+, 1 / m + 5 / n = 1 + Real.sqrt 5 / 3) ∧
    (∀ m n : ℕ+, 1 / m + 5 / n ≥ 1 + Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_arithmetic_sequence_l609_60937


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l609_60906

theorem product_of_sum_and_difference : 
  let a : ℝ := 4.93
  let b : ℝ := 3.78
  (a + b) * (a - b) = 10.0165 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l609_60906


namespace NUMINAMATH_CALUDE_jonah_profit_l609_60913

def pineapples : ℕ := 60
def base_price : ℚ := 2
def discount_rate : ℚ := 20 / 100
def rings_per_pineapple : ℕ := 12
def single_ring_price : ℚ := 4
def bundle_size : ℕ := 6
def bundle_price : ℚ := 20
def bundles_sold : ℕ := 35
def single_rings_sold : ℕ := 150

def discounted_price : ℚ := base_price * (1 - discount_rate)
def total_cost : ℚ := pineapples * discounted_price
def bundle_revenue : ℚ := bundles_sold * bundle_price
def single_ring_revenue : ℚ := single_rings_sold * single_ring_price
def total_revenue : ℚ := bundle_revenue + single_ring_revenue
def profit : ℚ := total_revenue - total_cost

theorem jonah_profit : profit = 1204 := by
  sorry

end NUMINAMATH_CALUDE_jonah_profit_l609_60913


namespace NUMINAMATH_CALUDE_term_206_of_specific_sequence_l609_60923

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem term_206_of_specific_sequence :
  let a₁ := 10
  let a₂ := -10
  let r := a₂ / a₁
  geometric_sequence a₁ r 206 = -10 := by sorry

end NUMINAMATH_CALUDE_term_206_of_specific_sequence_l609_60923


namespace NUMINAMATH_CALUDE_triangle_side_length_l609_60958

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Define the triangle
  A + B + C = Real.pi →  -- Sum of angles in a triangle is π radians
  A = Real.pi / 6 →      -- 30° in radians
  C = 7 * Real.pi / 12 → -- 105° in radians
  b = 8 →                -- Given side length
  -- Law of Sines
  b / Real.sin B = a / Real.sin A →
  -- Conclusion
  a = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l609_60958


namespace NUMINAMATH_CALUDE_age_problem_solution_l609_60955

/-- Represents the age relationship between a father and daughter -/
structure AgeProblem where
  daughter_age : ℕ
  father_age : ℕ
  years_ago : ℕ
  years_future : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : AgeProblem) : Prop :=
  p.father_age = 3 * p.daughter_age ∧
  (p.father_age - p.years_ago) = 5 * (p.daughter_age - p.years_ago)

/-- The future condition we want to prove -/
def future_condition (p : AgeProblem) : Prop :=
  (p.father_age + p.years_future) = 2 * (p.daughter_age + p.years_future)

/-- The theorem to prove -/
theorem age_problem_solution :
  ∀ p : AgeProblem,
    problem_conditions p →
    (p.years_future = 14 ↔ future_condition p) :=
by
  sorry


end NUMINAMATH_CALUDE_age_problem_solution_l609_60955


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l609_60935

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Check if a line passes through a point -/
def passesThrough (l : Line) (p : Point) : Prop := sorry

theorem tangent_line_to_circle (c : Circle) (p : Point) :
  c.center = Point.mk 2 0 →
  c.radius = 2 →
  p = Point.mk 4 5 →
  ∀ l : Line, (isTangent l c ∧ passesThrough l p) ↔ 
    (l = Line.mk 21 (-20) 16 ∨ l = Line.mk 1 0 (-4)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l609_60935


namespace NUMINAMATH_CALUDE_fishing_trip_total_l609_60969

def total_fish (pikes sturgeons herrings : ℕ) : ℕ :=
  pikes + sturgeons + herrings

theorem fishing_trip_total : total_fish 30 40 75 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fishing_trip_total_l609_60969


namespace NUMINAMATH_CALUDE_stone_145_is_2_l609_60910

/-- The number of stones in the arrangement -/
def num_stones : ℕ := 14

/-- The period of the counting sequence -/
def period : ℕ := 26

/-- The target count we're looking for -/
def target_count : ℕ := 145

/-- Function to convert the new count to the original stone number -/
def count_to_original (n : ℕ) : ℕ :=
  if n % period ≤ num_stones then n % period
  else period - (n % period) + 1

theorem stone_145_is_2 :
  count_to_original target_count = 2 := by sorry

end NUMINAMATH_CALUDE_stone_145_is_2_l609_60910


namespace NUMINAMATH_CALUDE_simplify_power_l609_60960

theorem simplify_power (y : ℝ) : (3 * y^2)^4 = 81 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_l609_60960


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l609_60977

theorem recurring_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n.gcd d = 1) ∧
  (7 + 318 / 999 : ℚ) = n / d :=
sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l609_60977


namespace NUMINAMATH_CALUDE_triangle_angle_A_l609_60970

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) (h3 : B = π / 4) :
  let A := Real.arcsin ((a * Real.sin B) / b)
  A = π / 3 ∨ A = 2 * π / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l609_60970


namespace NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_line_l609_60984

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_plane_perpendicular_line 
  (l m : Line) (α : Plane) :
  l ≠ m →
  parallel l α →
  perpendicular m α →
  perpendicularLines l m :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_line_l609_60984


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l609_60919

/-- The radius of an inscribed circle in a right triangle -/
theorem inscribed_circle_radius_right_triangle (a b c r : ℝ) 
  (h_right : a^2 + c^2 = b^2) -- Pythagorean theorem for right triangle
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
  : r = (a + c - b) / 2 ↔ 
    -- Definition of inscribed circle: 
    -- The circle touches all three sides of the triangle
    ∃ (x y : ℝ), 
      x > 0 ∧ y > 0 ∧
      x + y = b ∧
      x + r = c ∧
      y + r = a :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l609_60919


namespace NUMINAMATH_CALUDE_sqrt_5x_plus_y_squared_l609_60988

theorem sqrt_5x_plus_y_squared (x y : ℝ) 
  (h : Real.sqrt (x - 1) + (3 * x + y - 1)^2 = 0) : 
  Real.sqrt (5 * x + y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5x_plus_y_squared_l609_60988


namespace NUMINAMATH_CALUDE_expression_simplification_l609_60968

theorem expression_simplification (a : ℝ) : 
  a^3 * a^5 + (a^2)^4 + (-2*a^4)^2 - 10*a^10 / (5*a^2) = 4*a^8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l609_60968


namespace NUMINAMATH_CALUDE_car_meeting_problem_l609_60979

/-- Represents a car with a speed and initial position -/
structure Car where
  speed : ℝ
  initial_position : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  car_x : Car
  car_y : Car
  first_meeting_time : ℝ
  distance_between_meetings : ℝ

/-- The theorem statement -/
theorem car_meeting_problem (setup : ProblemSetup)
  (h1 : setup.car_x.speed = 50)
  (h2 : setup.first_meeting_time = 1)
  (h3 : setup.distance_between_meetings = 20)
  (h4 : setup.car_x.initial_position = 0)
  (h5 : setup.car_y.initial_position = setup.car_x.initial_position + 
        setup.car_x.speed * setup.first_meeting_time + 
        setup.car_y.speed * setup.first_meeting_time) :
  setup.car_y.initial_position - setup.car_x.initial_position = 110 ∧
  setup.car_y.speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_meeting_problem_l609_60979


namespace NUMINAMATH_CALUDE_factorization_equality_l609_60927

theorem factorization_equality (a : ℝ) : a * (a - 2) + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l609_60927


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l609_60999

theorem smallest_n_congruence : ∃! n : ℕ+, (3 * n : ℤ) ≡ 568 [ZMOD 34] ∧ 
  ∀ m : ℕ+, (3 * m : ℤ) ≡ 568 [ZMOD 34] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l609_60999


namespace NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l609_60995

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l609_60995


namespace NUMINAMATH_CALUDE_daily_wage_of_c_l609_60998

theorem daily_wage_of_c (a b c : ℕ) (total_earning : ℕ) : 
  a * 6 + b * 9 + c * 4 = total_earning →
  4 * a = 3 * b →
  5 * a = 3 * c →
  total_earning = 1554 →
  c = 105 := by
  sorry

end NUMINAMATH_CALUDE_daily_wage_of_c_l609_60998


namespace NUMINAMATH_CALUDE_muslim_boys_percentage_l609_60939

/-- The percentage of Muslim boys in a school -/
def percentage_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) : ℚ :=
  let non_muslim_boys := (hindu_percentage + sikh_percentage) * total_boys + other_boys
  let muslim_boys := total_boys - non_muslim_boys
  (muslim_boys / total_boys) * 100

/-- Theorem stating that the percentage of Muslim boys is approximately 44% -/
theorem muslim_boys_percentage :
  let total_boys : ℕ := 850
  let hindu_percentage : ℚ := 28 / 100
  let sikh_percentage : ℚ := 10 / 100
  let other_boys : ℕ := 153
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
    |percentage_muslim_boys total_boys hindu_percentage sikh_percentage other_boys - 44| < ε :=
sorry

end NUMINAMATH_CALUDE_muslim_boys_percentage_l609_60939


namespace NUMINAMATH_CALUDE_rectangle_x_value_l609_60901

/-- A rectangular figure with specified segment lengths -/
structure RectangularFigure where
  top_segment1 : ℝ
  top_segment2 : ℝ
  top_segment3 : ℝ
  bottom_segment1 : ℝ
  bottom_segment2 : ℝ
  bottom_segment3 : ℝ

/-- The property that the total length of top and bottom sides are equal -/
def is_valid_rectangle (r : RectangularFigure) : Prop :=
  r.top_segment1 + r.top_segment2 + r.top_segment3 = r.bottom_segment1 + r.bottom_segment2 + r.bottom_segment3

/-- The theorem stating that X must be 6 for the given rectangular figure -/
theorem rectangle_x_value :
  ∀ (x : ℝ),
  is_valid_rectangle ⟨3, 2, x, 4, 2, 5⟩ → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l609_60901


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_prism_side_length_l609_60959

theorem isosceles_right_triangle_prism_side_length 
  (XY XZ : ℝ) (height volume : ℝ) : 
  XY = XZ →  -- Base triangle is isosceles
  height = 6 →  -- Height of the prism
  volume = 27 →  -- Volume of the prism
  volume = (1/2 * XY * XY) * height →  -- Volume formula for triangular prism
  XY = 3 ∧ XZ = 3  -- Conclusion: side lengths are 3
  :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_prism_side_length_l609_60959


namespace NUMINAMATH_CALUDE_rectangle_area_l609_60951

theorem rectangle_area (p : ℝ) (h1 : p = 160) : 
  ∃ (s : ℝ), s > 0 ∧ p = 8 * s ∧ 4 * s^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l609_60951


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l609_60921

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℤ, x^2 + 2*x - 1 < 0) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l609_60921


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l609_60967

theorem necessary_but_not_sufficient
  (A B C : Set α)
  (hAnonempty : A.Nonempty)
  (hBnonempty : B.Nonempty)
  (hCnonempty : C.Nonempty)
  (hUnion : A ∪ B = C)
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l609_60967


namespace NUMINAMATH_CALUDE_cos_2theta_plus_pi_3_l609_60974

theorem cos_2theta_plus_pi_3 (θ : Real) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : 1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 2) : 
  Real.cos (2 * θ + π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_plus_pi_3_l609_60974


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l609_60942

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l609_60942


namespace NUMINAMATH_CALUDE_gabby_makeup_set_l609_60907

/-- The amount of money Gabby's mom gave her -/
def moms_gift (cost savings needed_after : ℕ) : ℕ :=
  cost - savings - needed_after

/-- Proof that Gabby's mom gave her $20 -/
theorem gabby_makeup_set : moms_gift 65 35 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gabby_makeup_set_l609_60907


namespace NUMINAMATH_CALUDE_lattice_point_decomposition_l609_60904

/-- Represents a point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a parallelogram OABC where O is the origin -/
structure Parallelogram where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is in or on a triangle -/
def inTriangle (P Q R S : LatticePoint) : Prop := sorry

/-- Vector addition -/
def vecAdd (P Q : LatticePoint) : LatticePoint := sorry

theorem lattice_point_decomposition 
  (OABC : Parallelogram) 
  (P : LatticePoint) 
  (h : inTriangle P OABC.A OABC.B OABC.C) :
  ∃ (Q R : LatticePoint), 
    inTriangle Q (LatticePoint.mk 0 0) OABC.A OABC.C ∧ 
    inTriangle R (LatticePoint.mk 0 0) OABC.A OABC.C ∧
    P = vecAdd Q R := by sorry

end NUMINAMATH_CALUDE_lattice_point_decomposition_l609_60904


namespace NUMINAMATH_CALUDE_insurance_payment_calculation_l609_60966

/-- The amount of a quarterly insurance payment in dollars. -/
def quarterly_payment : ℕ := 378

/-- The number of quarters in a year. -/
def quarters_per_year : ℕ := 4

/-- The annual insurance payment in dollars. -/
def annual_payment : ℕ := quarterly_payment * quarters_per_year

theorem insurance_payment_calculation :
  annual_payment = 1512 :=
by sorry

end NUMINAMATH_CALUDE_insurance_payment_calculation_l609_60966


namespace NUMINAMATH_CALUDE_heaviest_lightest_difference_l609_60925

def pumpkin_contest (brad_weight jessica_weight betty_weight : ℝ) : Prop :=
  jessica_weight = brad_weight / 2 ∧
  betty_weight = 4 * jessica_weight ∧
  brad_weight = 54

theorem heaviest_lightest_difference (brad_weight jessica_weight betty_weight : ℝ) 
  (h : pumpkin_contest brad_weight jessica_weight betty_weight) :
  max betty_weight (max brad_weight jessica_weight) - 
  min betty_weight (min brad_weight jessica_weight) = 81 := by
  sorry

end NUMINAMATH_CALUDE_heaviest_lightest_difference_l609_60925


namespace NUMINAMATH_CALUDE_f_g_minus_g_f_l609_60905

def f (x : ℝ) : ℝ := 4 * x + 8

def g (x : ℝ) : ℝ := 2 * x - 3

theorem f_g_minus_g_f : ∀ x : ℝ, f (g x) - g (f x) = -17 := by
  sorry

end NUMINAMATH_CALUDE_f_g_minus_g_f_l609_60905


namespace NUMINAMATH_CALUDE_intersection_and_complement_union_condition_implies_m_range_l609_60912

-- Define the sets
def U : Set ℝ := {x | 1 < x ∧ x < 7}
def A1 : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B1 : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

def A2 : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
def B2 (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- Theorem for the first part
theorem intersection_and_complement :
  (A1 ∩ B1 = {x | 3 ≤ x ∧ x < 5}) ∧
  (U \ A1 = {x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7)}) :=
sorry

-- Theorem for the second part
theorem union_condition_implies_m_range :
  ∀ m, (A2 ∪ B2 m = A2) → m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_complement_union_condition_implies_m_range_l609_60912


namespace NUMINAMATH_CALUDE_main_result_l609_60976

/-- A function satisfying the given property for all real numbers -/
def satisfies_property (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^3 * g a = a^3 * g c

/-- The main theorem -/
theorem main_result (g : ℝ → ℝ) (h1 : satisfies_property g) (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end NUMINAMATH_CALUDE_main_result_l609_60976


namespace NUMINAMATH_CALUDE_tv_selection_theorem_l609_60985

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of Type A TVs -/
def typeA : ℕ := 4

/-- The number of Type B TVs -/
def typeB : ℕ := 5

/-- The total number of TVs to be selected -/
def selectTotal : ℕ := 3

/-- The number of ways to select TVs satisfying the given conditions -/
def selectWays : ℕ :=
  typeA * binomial typeB (selectTotal - 1) +
  binomial typeA (selectTotal - 1) * typeB

theorem tv_selection_theorem : selectWays = 70 := by sorry

end NUMINAMATH_CALUDE_tv_selection_theorem_l609_60985


namespace NUMINAMATH_CALUDE_triple_solution_l609_60972

theorem triple_solution (k : ℕ) (hk : k > 0) :
  ∀ a b c : ℕ, 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 3 * k + 1 →
    a * b + b * c + c * a = 3 * k^2 + 2 * k →
    (a = k + 1 ∧ b = k ∧ c = k) :=
by sorry

end NUMINAMATH_CALUDE_triple_solution_l609_60972


namespace NUMINAMATH_CALUDE_pool_filling_time_l609_60943

/-- The number of hours it takes for a swimming pool to reach full capacity -/
def full_capacity_hours : ℕ := 8

/-- The factor by which the water volume increases each hour -/
def volume_increase_factor : ℕ := 3

/-- The fraction of the pool's capacity we're interested in -/
def target_fraction : ℚ := 1 / 9

/-- The number of hours it takes to reach the target fraction of capacity -/
def target_hours : ℕ := 6

theorem pool_filling_time :
  (volume_increase_factor ^ (full_capacity_hours - target_hours) : ℚ) = 1 / target_fraction :=
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l609_60943


namespace NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l609_60997

theorem smallest_n_for_floor_equation : ∃ (x : ℤ), ⌊(10 : ℝ)^7 / x⌋ = 1989 ∧ ∀ (n : ℕ), n < 7 → ¬∃ (x : ℤ), ⌊(10 : ℝ)^n / x⌋ = 1989 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l609_60997


namespace NUMINAMATH_CALUDE_boat_downstream_speed_l609_60986

/-- Given a boat's speed in still water and its upstream speed, calculate its downstream speed. -/
theorem boat_downstream_speed
  (still_water_speed : ℝ)
  (upstream_speed : ℝ)
  (h1 : still_water_speed = 7)
  (h2 : upstream_speed = 4) :
  still_water_speed + (still_water_speed - upstream_speed) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_speed_l609_60986


namespace NUMINAMATH_CALUDE_compute_expression_l609_60922

theorem compute_expression : 3 * 3^4 - 9^27 / 9^25 = 162 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l609_60922


namespace NUMINAMATH_CALUDE_total_clam_shells_is_43_l609_60991

/-- The number of clam shells found by Sam, Mary, and Lucy -/
def clam_shells (name : String) : ℕ :=
  match name with
  | "Sam" => 8
  | "Mary" => 20
  | "Lucy" => 15
  | _ => 0

/-- The total number of clam shells found by Sam, Mary, and Lucy -/
def total_clam_shells : ℕ :=
  clam_shells "Sam" + clam_shells "Mary" + clam_shells "Lucy"

/-- Theorem stating that the total number of clam shells found is 43 -/
theorem total_clam_shells_is_43 : total_clam_shells = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_clam_shells_is_43_l609_60991


namespace NUMINAMATH_CALUDE_student_difference_l609_60987

/-- Given that the sum of students in grades 1 and 2 is 30 more than the sum of students in grades 2 and 5,
    prove that the difference between the number of students in grade 1 and grade 5 is 30. -/
theorem student_difference (g1 g2 g5 : ℕ) (h : g1 + g2 = g2 + g5 + 30) : g1 - g5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_difference_l609_60987


namespace NUMINAMATH_CALUDE_cost_of_graveling_roads_l609_60965

/-- The cost of graveling two intersecting roads on a rectangular lawn. -/
theorem cost_of_graveling_roads
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sq_m : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sq_m = 2) :
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * cost_per_sq_m = 2600 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_graveling_roads_l609_60965


namespace NUMINAMATH_CALUDE_remaining_water_l609_60961

theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/4 → remaining = initial - used → remaining = 1/4 := by sorry

end NUMINAMATH_CALUDE_remaining_water_l609_60961


namespace NUMINAMATH_CALUDE_paiges_science_problems_l609_60908

theorem paiges_science_problems 
  (math_problems : ℕ) 
  (total_problems : ℕ → ℕ → ℕ) 
  (finished_problems : ℕ) 
  (remaining_problems : ℕ) 
  (h1 : math_problems = 43)
  (h2 : ∀ m s, total_problems m s = m + s)
  (h3 : finished_problems = 44)
  (h4 : remaining_problems = 11)
  (h5 : ∀ s, remaining_problems = total_problems math_problems s - finished_problems) :
  ∃ s : ℕ, s = 12 ∧ total_problems math_problems s = finished_problems + remaining_problems :=
sorry

end NUMINAMATH_CALUDE_paiges_science_problems_l609_60908


namespace NUMINAMATH_CALUDE_parabola_vertex_l609_60975

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -(x - 5)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (5, 3)

/-- Theorem: The vertex of the parabola y = -(x-5)^2 + 3 is (5, 3) -/
theorem parabola_vertex :
  ∀ (x : ℝ), parabola_equation x ≤ parabola_equation (vertex.1) ∧
  parabola_equation (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l609_60975


namespace NUMINAMATH_CALUDE_max_value_abcd_l609_60936

theorem max_value_abcd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (sum_eq_3 : a + b + c + d = 3) :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end NUMINAMATH_CALUDE_max_value_abcd_l609_60936


namespace NUMINAMATH_CALUDE_perry_dana_game_difference_l609_60933

theorem perry_dana_game_difference (phil_games dana_games charlie_games perry_games : ℕ) : 
  phil_games = 12 →
  charlie_games = dana_games - 2 →
  phil_games = charlie_games + 3 →
  perry_games = phil_games + 4 →
  perry_games - dana_games = 5 := by
sorry

end NUMINAMATH_CALUDE_perry_dana_game_difference_l609_60933


namespace NUMINAMATH_CALUDE_product_even_implies_factor_even_l609_60903

theorem product_even_implies_factor_even (a b : ℕ) : 
  Even (a * b) → Even a ∨ Even b := by sorry

end NUMINAMATH_CALUDE_product_even_implies_factor_even_l609_60903


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l609_60947

/-- The total surface area of a hemisphere with base area 225π is 675π. -/
theorem hemisphere_surface_area : 
  ∀ r : ℝ, 
  r > 0 → 
  π * r^2 = 225 * π → 
  2 * π * r^2 + π * r^2 = 675 * π :=
by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l609_60947


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l609_60915

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C, 
    if b^2 + c^2 - bc = a^2 and b/c = tan(B) / tan(C), 
    then the triangle is equilateral. -/
theorem triangle_is_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 + c^2 - b*c = a^2) 
  (h2 : b/c = Real.tan B / Real.tan C) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) :
  a = b ∧ b = c := by
  sorry


end NUMINAMATH_CALUDE_triangle_is_equilateral_l609_60915


namespace NUMINAMATH_CALUDE_division_problem_l609_60978

theorem division_problem (n : ℕ) (h1 : n % 11 = 1) (h2 : n / 11 = 13) : n = 144 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l609_60978


namespace NUMINAMATH_CALUDE_intersection_M_N_l609_60930

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {-1, 1, 2}

theorem intersection_M_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l609_60930


namespace NUMINAMATH_CALUDE_equation_roots_l609_60989

theorem equation_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    (0.5 : ℝ)^(x^2 - m*x + 0.5*m - 1.5) = (Real.sqrt 8)^(m - 1) ∧
    (0.5 : ℝ)^(y^2 - m*y + 0.5*m - 1.5) = (Real.sqrt 8)^(m - 1))
  ↔ (m < 2 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l609_60989


namespace NUMINAMATH_CALUDE_smallest_with_15_divisors_l609_60940

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a given natural number has exactly 15 positive divisors -/
def has_15_divisors (n : ℕ+) : Prop := num_divisors n = 15

theorem smallest_with_15_divisors :
  (∀ m : ℕ+, m < 24 → ¬(has_15_divisors m)) ∧ has_15_divisors 24 := by sorry

end NUMINAMATH_CALUDE_smallest_with_15_divisors_l609_60940


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l609_60952

theorem max_value_sum_of_roots (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 48) :
  (∀ a b : ℝ, 3 * a^2 + 4 * b^2 = 48 →
    Real.sqrt (x^2 + y^2 - 4*x + 4) + Real.sqrt (x^2 + y^2 - 2*x + 4*y + 5) ≥
    Real.sqrt (a^2 + b^2 - 4*a + 4) + Real.sqrt (a^2 + b^2 - 2*a + 4*b + 5)) ∧
  (∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * y₀^2 = 48 ∧
    Real.sqrt (x₀^2 + y₀^2 - 4*x₀ + 4) + Real.sqrt (x₀^2 + y₀^2 - 2*x₀ + 4*y₀ + 5) = 8 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l609_60952
