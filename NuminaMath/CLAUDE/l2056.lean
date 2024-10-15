import Mathlib

namespace NUMINAMATH_CALUDE_a_periodic_with_period_5_l2056_205629

/-- The sequence a_n defined as 6^n mod 100 -/
def a (n : ℕ) : ℕ := (6^n) % 100

/-- The period of the sequence a_n -/
def period : ℕ := 5

theorem a_periodic_with_period_5 :
  (∀ n ≥ 2, a (n + period) = a n) ∧
  (∀ k < period, ∃ m ≥ 2, a (m + k) ≠ a m) :=
sorry

end NUMINAMATH_CALUDE_a_periodic_with_period_5_l2056_205629


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2056_205607

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 6) = Real.sqrt 6 / 2) → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2056_205607


namespace NUMINAMATH_CALUDE_toucan_count_l2056_205694

/-- The total number of toucans after joining all limbs -/
def total_toucans (initial_first initial_second initial_third joining_first joining_second joining_third : ℝ) : ℝ :=
  (initial_first + joining_first) + (initial_second + joining_second) + (initial_third + joining_third)

/-- Theorem stating the total number of toucans after joining -/
theorem toucan_count : 
  total_toucans 3.5 4.25 2.75 1.5 0.6 1.2 = 13.8 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l2056_205694


namespace NUMINAMATH_CALUDE_percentage_of_democrats_l2056_205635

theorem percentage_of_democrats (D R : ℝ) : 
  D + R = 100 →
  0.7 * D + 0.2 * R = 50 →
  D = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_democrats_l2056_205635


namespace NUMINAMATH_CALUDE_black_and_white_films_count_l2056_205667

-- Define variables
variable (x y : ℚ)
variable (B : ℚ)

-- Define the theorem
theorem black_and_white_films_count :
  (6 * y) / ((y / x) / 100 * B + 6 * y) = 20 / 21 →
  B = 30 * x := by
sorry

end NUMINAMATH_CALUDE_black_and_white_films_count_l2056_205667


namespace NUMINAMATH_CALUDE_chessboard_division_theorem_l2056_205617

/-- Represents a 6x6 chessboard --/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- Represents a 2x1 domino on the chessboard --/
structure Domino where
  x : Fin 6
  y : Fin 6
  horizontal : Bool

/-- A configuration of dominoes on the chessboard --/
def DominoConfiguration := List Domino

/-- Checks if a given line (horizontal or vertical) intersects any domino --/
def lineIntersectsDomino (line : Nat) (horizontal : Bool) (config : DominoConfiguration) : Bool :=
  sorry

/-- The main theorem --/
theorem chessboard_division_theorem (config : DominoConfiguration) :
  config.length = 18 → ∃ (line : Nat) (horizontal : Bool),
    line < 6 ∧ ¬lineIntersectsDomino line horizontal config :=
  sorry

end NUMINAMATH_CALUDE_chessboard_division_theorem_l2056_205617


namespace NUMINAMATH_CALUDE_sons_age_is_correct_l2056_205619

/-- The age of the son -/
def sons_age : ℕ := 23

/-- The age of the father -/
def fathers_age : ℕ := sons_age + 25

theorem sons_age_is_correct : 
  (fathers_age + 2 = 2 * (sons_age + 2)) ∧ 
  (fathers_age = sons_age + 25) ∧ 
  (sons_age = 23) := by
  sorry

end NUMINAMATH_CALUDE_sons_age_is_correct_l2056_205619


namespace NUMINAMATH_CALUDE_stadium_length_in_feet_l2056_205646

/-- Converts yards to feet using the standard conversion factor. -/
def yards_to_feet (yards : ℕ) : ℕ := yards * 3

/-- The length of the sports stadium in yards. -/
def stadium_length_yards : ℕ := 61

theorem stadium_length_in_feet :
  yards_to_feet stadium_length_yards = 183 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_in_feet_l2056_205646


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2056_205665

theorem triangle_max_perimeter (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  ∃ (n : ℕ), n = 62 ∧ ∀ (s : ℝ), s > 0 → a + s > b → b + s > a → n > a + b + s :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2056_205665


namespace NUMINAMATH_CALUDE_single_filter_price_l2056_205618

/-- The price of a camera lens filter kit containing 5 filters -/
def kit_price : ℝ := 87.50

/-- The price of the first type of filter -/
def filter1_price : ℝ := 16.45

/-- The price of the second type of filter -/
def filter2_price : ℝ := 14.05

/-- The discount rate when purchasing the kit -/
def discount_rate : ℝ := 0.08

/-- The number of filters of the first type -/
def num_filter1 : ℕ := 2

/-- The number of filters of the second type -/
def num_filter2 : ℕ := 2

/-- The number of filters of the unknown type -/
def num_filter3 : ℕ := 1

/-- The total number of filters in the kit -/
def total_filters : ℕ := num_filter1 + num_filter2 + num_filter3

theorem single_filter_price (x : ℝ) : 
  (num_filter1 : ℝ) * filter1_price + (num_filter2 : ℝ) * filter2_price + (num_filter3 : ℝ) * x = 
  kit_price / (1 - discount_rate) → x = 34.11 := by
  sorry

end NUMINAMATH_CALUDE_single_filter_price_l2056_205618


namespace NUMINAMATH_CALUDE_complex_calculation_l2056_205675

theorem complex_calculation : (1 - Complex.I) - (-3 + 2 * Complex.I) + (4 - 6 * Complex.I) = 8 - 9 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l2056_205675


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l2056_205668

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 5*a| ≤ 3) ↔ (a = 3/4 ∨ a = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l2056_205668


namespace NUMINAMATH_CALUDE_solve_invitations_l2056_205621

def invitations_problem (I : ℝ) : Prop :=
  let rsvp_rate : ℝ := 0.9
  let show_up_rate : ℝ := 0.8
  let no_gift_attendees : ℕ := 10
  let thank_you_cards : ℕ := 134
  
  (rsvp_rate * show_up_rate * I - no_gift_attendees : ℝ) = thank_you_cards

theorem solve_invitations : ∃ I : ℝ, invitations_problem I ∧ I = 200 := by
  sorry

end NUMINAMATH_CALUDE_solve_invitations_l2056_205621


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2056_205682

theorem quadratic_equation_properties :
  ∀ s t : ℝ, 2 * s^2 + 3 * s - 1 = 0 → 2 * t^2 + 3 * t - 1 = 0 → s ≠ t →
  (s + t = -3/2) ∧
  (s * t = -1/2) ∧
  (s^2 + t^2 = 13/4) ∧
  (|1/s - 1/t| = Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2056_205682


namespace NUMINAMATH_CALUDE_other_pencil_length_l2056_205676

/-- Given two pencils with a total length of 24 cubes, where one pencil is 12 cubes long,
    prove that the other pencil is also 12 cubes long. -/
theorem other_pencil_length (total_length : ℕ) (first_pencil : ℕ) (h1 : total_length = 24) (h2 : first_pencil = 12) :
  total_length - first_pencil = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_pencil_length_l2056_205676


namespace NUMINAMATH_CALUDE_complement_of_A_l2056_205605

-- Define the set A
def A : Set ℝ := {x | x^2 + 3*x ≥ 0} ∪ {x | 2*x > 1}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | -3 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2056_205605


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_l2056_205648

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A circle with radius r -/
structure Circle where
  r : ℝ
  h : r > 0

/-- The theorem statement -/
theorem ellipse_circle_tangent (C : Ellipse) (O : Circle) :
  C.a = 2 * Real.sqrt 2 →  -- Left vertex at (-2√2, 0)
  O.r = 2 →  -- Circle equation: x² + y² = 4
  (∃ F : ℝ × ℝ, F.1 = -Real.sqrt 2 ∧ F.2 = 0 ∧
    ∃ A B : ℝ × ℝ, 
      -- A and B are on the circle
      (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧
      -- Line AB passes through F
      (B.2 - A.2) * (F.1 - A.1) = (F.2 - A.2) * (B.1 - A.1)) →
  C.a^2 + C.b^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_l2056_205648


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2056_205695

theorem abs_neg_three_eq_three : |(-3 : ℚ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2056_205695


namespace NUMINAMATH_CALUDE_trip_duration_l2056_205657

/-- A car trip with varying speeds -/
structure CarTrip where
  totalTime : ℝ
  averageSpeed : ℝ

/-- The conditions of the car trip -/
def tripConditions (trip : CarTrip) : Prop :=
  ∃ (additionalTime : ℝ),
    trip.totalTime = 4 + additionalTime ∧
    50 * 4 + 80 * additionalTime = 65 * trip.totalTime ∧
    trip.averageSpeed = 65

/-- The theorem stating that the trip duration is 8 hours -/
theorem trip_duration (trip : CarTrip) 
    (h : tripConditions trip) : trip.totalTime = 8 := by
  sorry

#check trip_duration

end NUMINAMATH_CALUDE_trip_duration_l2056_205657


namespace NUMINAMATH_CALUDE_original_number_proof_l2056_205677

theorem original_number_proof (x : ℝ) : 
  (x * 1.2 * 0.6 = 1080) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2056_205677


namespace NUMINAMATH_CALUDE_find_d_l2056_205645

theorem find_d (a b c d : ℕ+) 
  (eq1 : a ^ 2 = c * (d + 20))
  (eq2 : b ^ 2 = c * (d - 18)) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l2056_205645


namespace NUMINAMATH_CALUDE_factor_tree_value_l2056_205600

-- Define the variables
def W : ℕ := 7
def Y : ℕ := 7 * 11
def Z : ℕ := 13 * W
def X : ℕ := Y * Z

-- State the theorem
theorem factor_tree_value : X = 7007 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l2056_205600


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_165_l2056_205624

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

theorem arithmetic_progression_sum_165 :
  ∃ ap : ArithmeticProgression,
    sum_n_terms ap 15 = 200 ∧
    sum_n_terms ap 150 = 150 ∧
    sum_n_terms ap 165 = -3064 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_165_l2056_205624


namespace NUMINAMATH_CALUDE_triangle_area_l2056_205633

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3 under the following conditions:
    - (2b - √3c) / (√3a) = cos(C) / cos(A)
    - B = π/6
    - The median AM on side BC has length √7 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (M : ℝ) : 
  (2 * b - Real.sqrt 3 * c) / (Real.sqrt 3 * a) = Real.cos C / Real.cos A →
  B = π / 6 →
  M = Real.sqrt 7 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2056_205633


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l2056_205660

theorem different_tens_digit_probability :
  let n : ℕ := 5  -- number of integers to choose
  let lower_bound : ℕ := 10  -- lower bound of the range
  let upper_bound : ℕ := 59  -- upper bound of the range
  let total_numbers : ℕ := upper_bound - lower_bound + 1  -- total numbers in the range
  let tens_digits : ℕ := 5  -- number of different tens digits in the range
  let numbers_per_tens : ℕ := 10  -- numbers available for each tens digit

  -- Probability of choosing n integers with different tens digits
  (numbers_per_tens ^ n : ℚ) / (total_numbers.choose n) = 2500 / 52969 :=
by sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l2056_205660


namespace NUMINAMATH_CALUDE_problem_statement_l2056_205691

theorem problem_statement (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : Real.log x / Real.log 4 ^ 3 + Real.log y / Real.log 5 ^ 3 + 9 = 
        12 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) : 
  x^2 + y^2 = 64 + 25 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2056_205691


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2056_205666

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2) :
  a / (a^2 + 2*a + 1) / (1 - a / (a + 1)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2056_205666


namespace NUMINAMATH_CALUDE_car_speed_proof_l2056_205659

/-- Proves that a car traveling at speed v km/h takes 20 seconds longer to travel 1 kilometer 
    than it would at 36 km/h if and only if v = 30 km/h. -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 36) * 3600 = 20 ↔ v = 30 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2056_205659


namespace NUMINAMATH_CALUDE_quotient_in_third_quadrant_l2056_205685

/-- Given complex numbers z₁ and z₂ where z₁ = 1 - 2i and the points corresponding to z₁ and z₂ 
    are symmetric about the imaginary axis, the point corresponding to z₂/z₁ lies in the third 
    quadrant of the complex plane. -/
theorem quotient_in_third_quadrant (z₁ z₂ : ℂ) 
    (h₁ : z₁ = 1 - 2*I) 
    (h₂ : z₂.re = -z₁.re ∧ z₂.im = z₁.im) : 
    (z₂ / z₁).re < 0 ∧ (z₂ / z₁).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_quotient_in_third_quadrant_l2056_205685


namespace NUMINAMATH_CALUDE_hexagon_properties_l2056_205644

/-- A regular hexagon with diagonals -/
structure RegularHexagonWithDiagonals where
  /-- The area of the regular hexagon -/
  area : ℝ
  /-- The hexagon is regular -/
  is_regular : Bool
  /-- All diagonals are drawn -/
  diagonals_drawn : Bool

/-- The number of parts the hexagon is divided into by its diagonals -/
def num_parts (h : RegularHexagonWithDiagonals) : ℕ := sorry

/-- The area of the new regular hexagon formed by combining all quadrilateral parts -/
def new_hexagon_area (h : RegularHexagonWithDiagonals) : ℝ := sorry

/-- Theorem about the properties of a regular hexagon with diagonals -/
theorem hexagon_properties (h : RegularHexagonWithDiagonals) 
  (h_area : h.area = 144)
  (h_regular : h.is_regular = true)
  (h_diagonals : h.diagonals_drawn = true) :
  num_parts h = 24 ∧ new_hexagon_area h = 48 := by sorry

end NUMINAMATH_CALUDE_hexagon_properties_l2056_205644


namespace NUMINAMATH_CALUDE_expression_simplification_l2056_205680

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 4) * (9 * p - 12) = 89 * p - 84 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2056_205680


namespace NUMINAMATH_CALUDE_william_tax_is_800_l2056_205651

/-- Represents the farm tax system in a village -/
structure FarmTaxSystem where
  total_tax : ℝ
  taxable_land_percentage : ℝ
  william_land_percentage : ℝ

/-- Calculates the farm tax paid by Mr. William -/
def william_tax (system : FarmTaxSystem) : ℝ :=
  system.total_tax * system.william_land_percentage

/-- Theorem stating that Mr. William's farm tax is $800 -/
theorem william_tax_is_800 (system : FarmTaxSystem) 
  (h1 : system.total_tax = 5000)
  (h2 : system.taxable_land_percentage = 0.6)
  (h3 : system.william_land_percentage = 0.16) : 
  william_tax system = 800 := by
  sorry


end NUMINAMATH_CALUDE_william_tax_is_800_l2056_205651


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_12_l2056_205671

theorem binomial_coefficient_21_12 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 12 = 125970) →
  (Nat.choose 21 13 = 203490) →
  (Nat.choose 21 12 = 125970) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_12_l2056_205671


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l2056_205622

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let circle_radius : ℝ := 12
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := π * circle_radius^2
  let quarter_circle_area : ℝ := circle_area / 4
  rectangle_area + (circle_area - quarter_circle_area) = 96 + 108 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l2056_205622


namespace NUMINAMATH_CALUDE_braiding_time_for_dance_team_l2056_205623

/-- Calculates the time in minutes to braid dancers' hair -/
def braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℕ :=
  (num_dancers * braids_per_dancer * seconds_per_braid) / 60

/-- Proves that braiding 8 dancers' hair with 5 braids each, taking 30 seconds per braid, results in 20 minutes total -/
theorem braiding_time_for_dance_team : braiding_time 8 5 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_braiding_time_for_dance_team_l2056_205623


namespace NUMINAMATH_CALUDE_symmetric_derivative_implies_cosine_possible_l2056_205698

/-- A function whose derivative's graph is symmetric about the origin -/
class SymmetricDerivative (f : ℝ → ℝ) : Prop :=
  (symmetric : ∀ x : ℝ, (deriv f) x = -(deriv f) (-x))

/-- The theorem stating that if f'(x) is symmetric about the origin, 
    then f(x) = 3cos(x) is a possible expression for f(x) -/
theorem symmetric_derivative_implies_cosine_possible 
  (f : ℝ → ℝ) [SymmetricDerivative f] : 
  ∃ g : ℝ → ℝ, (∀ x, f x = 3 * Real.cos x) ∧ SymmetricDerivative g :=
sorry

end NUMINAMATH_CALUDE_symmetric_derivative_implies_cosine_possible_l2056_205698


namespace NUMINAMATH_CALUDE_count_numbers_correct_l2056_205696

/-- The count of n-digit numbers composed of digits 1, 2, and 3, where each digit appears at least once -/
def count_numbers (n : ℕ) : ℕ :=
  3^n - 3 * 2^n + 3

/-- Theorem stating that count_numbers gives the correct count -/
theorem count_numbers_correct (n : ℕ) :
  count_numbers n = (3^n : ℕ) - 3 * (2^n : ℕ) + 3 :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_correct_l2056_205696


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2056_205686

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  total_population : ℕ
  male_population : ℕ
  female_population : ℕ
  female_sample : ℕ
  male_sample : ℕ

/-- Checks if a stratified sample is valid according to the stratified sampling principle -/
def is_valid_stratified_sample (s : StratifiedSample) : Prop :=
  s.female_population * s.male_sample = s.male_population * s.female_sample

theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_population = 680)
  (h2 : s.male_population = 360)
  (h3 : s.female_population = 320)
  (h4 : s.female_sample = 16)
  (h5 : is_valid_stratified_sample s) :
  s.male_sample = 18 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2056_205686


namespace NUMINAMATH_CALUDE_equation_solution_l2056_205650

theorem equation_solution : 
  ∃! y : ℚ, (4 * y - 5) / (5 * y - 15) = 7 / 10 ∧ y = -11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2056_205650


namespace NUMINAMATH_CALUDE_vertex_y_coordinate_is_zero_l2056_205654

-- Define a trinomial function
def trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition that (f(x))^3 - f(x) = 0 has three real roots
def has_three_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (f x₁)^3 - f x₁ = 0 ∧ (f x₂)^3 - f x₂ = 0 ∧ (f x₃)^3 - f x₃ = 0

-- Theorem statement
theorem vertex_y_coordinate_is_zero 
  (a b c : ℝ) 
  (h : has_three_real_roots (trinomial a b c)) :
  let f := trinomial a b c
  let vertex_y := f (- b / (2 * a))
  vertex_y = 0 := by
sorry

end NUMINAMATH_CALUDE_vertex_y_coordinate_is_zero_l2056_205654


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2056_205639

/-- The number of n-digit numbers formed using the digits 1, 2, and 3, where each digit is used at least once -/
def valid_numbers (n : ℕ) : ℕ :=
  3^n - 3 * 2^n + 3

/-- Theorem stating that for n ≥ 3, the number of n-digit numbers formed using the digits 1, 2, and 3, 
    where each digit is used at least once, is equal to 3^n - 3 * 2^n + 3 -/
theorem count_valid_numbers (n : ℕ) (h : n ≥ 3) : 
  (valid_numbers n) = (3^n - 3 * 2^n + 3) := by
  sorry


end NUMINAMATH_CALUDE_count_valid_numbers_l2056_205639


namespace NUMINAMATH_CALUDE_interval_of_decrease_l2056_205620

/-- The function f(x) = -x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -2*x + 4

theorem interval_of_decrease (x : ℝ) :
  x ≥ 2 → (∀ y, y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l2056_205620


namespace NUMINAMATH_CALUDE_function_always_negative_l2056_205637

theorem function_always_negative
  (f : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, (2 - x) * f x + x * deriv f x < 0) :
  ∀ x : ℝ, f x < 0 :=
by sorry

end NUMINAMATH_CALUDE_function_always_negative_l2056_205637


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2056_205674

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R > 0 ∧ R / 100 * P * 10 = 4/5 * P ∧ R = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2056_205674


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2056_205614

theorem factorial_divisibility (p : ℕ) (h : Prime p) : 
  (Nat.factorial (p^2)) % (Nat.factorial p)^(p+1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2056_205614


namespace NUMINAMATH_CALUDE_other_lateral_side_length_l2056_205655

/-- A trapezoid with the property that a line through the midpoint of one lateral side
    divides it into two quadrilaterals, each with an inscribed circle -/
structure SpecialTrapezoid where
  /-- Length of one base -/
  a : ℝ
  /-- Length of the other base -/
  b : ℝ
  /-- The trapezoid has the special property -/
  has_special_property : Bool

/-- The length of the other lateral side in a special trapezoid -/
def other_lateral_side (t : SpecialTrapezoid) : ℝ :=
  t.a + t.b

theorem other_lateral_side_length (t : SpecialTrapezoid) 
  (h : t.has_special_property = true) : 
  other_lateral_side t = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_other_lateral_side_length_l2056_205655


namespace NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l2056_205653

theorem binomial_sum_divides_power_of_two (n : ℕ) : 
  n > 3 → (1 + n.choose 1 + n.choose 2 + n.choose 3 ∣ 2^2000) ↔ (n = 7 ∨ n = 23) := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l2056_205653


namespace NUMINAMATH_CALUDE_equilibrium_instability_l2056_205643

/-- The system of differential equations -/
def system (x y : ℝ) : ℝ × ℝ :=
  (y^3 + x^5, x^3 + y^5)

/-- The Lyapunov function -/
def v (x y : ℝ) : ℝ :=
  x^4 - y^4

/-- The time derivative of the Lyapunov function -/
def dv_dt (x y : ℝ) : ℝ :=
  4 * (x^8 - y^8)

/-- Theorem stating the instability of the equilibrium point (0, 0) -/
theorem equilibrium_instability :
  ∃ (ε : ℝ), ε > 0 ∧
  ∀ (δ : ℝ), δ > 0 →
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 < δ^2 ∧
  ∃ (t : ℝ), t > 0 ∧
  let (x, y) := system x₀ y₀
  x^2 + y^2 > ε^2 :=
sorry

end NUMINAMATH_CALUDE_equilibrium_instability_l2056_205643


namespace NUMINAMATH_CALUDE_square_of_binomial_identity_l2056_205664

/-- The square of a binomial formula -/
def square_of_binomial (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

/-- Expression A -/
def expr_A (a b : ℝ) : ℝ := (a + b) * (a + b)

/-- Expression B -/
def expr_B (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)

/-- Expression C -/
def expr_C (a : ℝ) : ℝ := (a - 3) * (3 - a)

/-- Expression D -/
def expr_D (a b : ℝ) : ℝ := (2*a - b) * (-2*a + 3*b)

theorem square_of_binomial_identity (a b : ℝ) :
  expr_A a b = square_of_binomial a b ∧
  ∃ x y, expr_B x y ≠ square_of_binomial x y ∧
  ∃ a, expr_C a ≠ square_of_binomial (a - 3) 3 ∧
  ∃ a b, expr_D a b ≠ square_of_binomial (2*a - b) (-2*a + 3*b) :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_identity_l2056_205664


namespace NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l2056_205634

theorem unique_real_sqrt_negative_square : 
  ∃! x : ℝ, ∃ y : ℝ, y ^ 2 = -(x + 2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l2056_205634


namespace NUMINAMATH_CALUDE_second_price_increase_l2056_205689

theorem second_price_increase (P : ℝ) (x : ℝ) (h : P > 0) :
  (1.15 * P) * (1 + x / 100) = 1.4375 * P → x = 25 := by
sorry

end NUMINAMATH_CALUDE_second_price_increase_l2056_205689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2056_205603

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_is_term (a : ℕ → ℕ) : Prop :=
  ∀ p s, ∃ t, a p + a s = a t

theorem arithmetic_sequence_property (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d →
  a 1 = 12 →
  d > 0 →
  sum_is_term a →
  d = 6 ∨ d = 3 ∨ d = 2 ∨ d = 1 →
  d = 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2056_205603


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2056_205678

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, m)
  parallel a b → m = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2056_205678


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l2056_205692

theorem n_times_n_plus_one_div_by_three (n : ℤ) (h : 1 ≤ n ∧ n ≤ 99) :
  ∃ k : ℤ, n * (n + 1) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l2056_205692


namespace NUMINAMATH_CALUDE_pen_price_correct_max_pens_correct_l2056_205636

-- Define the original price of a pen
def original_pen_price : ℝ := 4

-- Define the discount rate for pens in the first part
def discount_rate : ℝ := 0.1

-- Define the total budget
def budget : ℝ := 360

-- Define the number of additional pens that can be bought after discount
def additional_pens : ℕ := 10

-- Define the total number of items to be purchased
def total_items : ℕ := 80

-- Define the original price of a pencil case
def pencil_case_price : ℝ := 10

-- Define the discount rate for both items in the second part
def discount_rate_2 : ℝ := 0.2

-- Define the minimum total purchase amount
def min_purchase_amount : ℝ := 400

theorem pen_price_correct :
  budget / original_pen_price + additional_pens = budget / (original_pen_price * (1 - discount_rate)) :=
sorry

theorem max_pens_correct :
  ∀ y : ℕ, y ≤ 50 →
  y ≤ total_items →
  min_purchase_amount ≤ original_pen_price * (1 - discount_rate_2) * y + pencil_case_price * (1 - discount_rate_2) * (total_items - y) :=
sorry

#check pen_price_correct
#check max_pens_correct

end NUMINAMATH_CALUDE_pen_price_correct_max_pens_correct_l2056_205636


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_19_l2056_205638

theorem remainder_of_3_pow_19 : 3^19 % 1162261460 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_19_l2056_205638


namespace NUMINAMATH_CALUDE_altitude_polynomial_l2056_205604

/-- Given a cubic polynomial with rational coefficients whose roots are the side lengths of a triangle,
    the altitudes of this triangle are roots of a polynomial of sixth degree with rational coefficients. -/
theorem altitude_polynomial (a b c d : ℚ) (r₁ r₂ r₃ : ℝ) :
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) →
  (r₁ + r₂ > r₃ ∧ r₂ + r₃ > r₁ ∧ r₃ + r₁ > r₂) →
  ∃ (p q s t u v w : ℚ),
    ∀ h₁ h₂ h₃ : ℝ,
      (h₁ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₁ ∧
       h₂ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₂ ∧
       h₃ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₃) →
      p * h₁^6 + q * h₁^5 + s * h₁^4 + t * h₁^3 + u * h₁^2 + v * h₁ + w = 0 ∧
      p * h₂^6 + q * h₂^5 + s * h₂^4 + t * h₂^3 + u * h₂^2 + v * h₂ + w = 0 ∧
      p * h₃^6 + q * h₃^5 + s * h₃^4 + t * h₃^3 + u * h₃^2 + v * h₃ + w = 0 :=
by sorry

end NUMINAMATH_CALUDE_altitude_polynomial_l2056_205604


namespace NUMINAMATH_CALUDE_three_solutions_cosine_sine_equation_l2056_205601

theorem three_solutions_cosine_sine_equation :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, 0 < x ∧ x < 3 * Real.pi ∧ 3 * (Real.cos x)^2 + 2 * (Real.sin x)^2 = 2) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_cosine_sine_equation_l2056_205601


namespace NUMINAMATH_CALUDE_concentric_circles_equal_areas_l2056_205625

/-- Given a circle of radius R divided by two concentric circles into three equal areas,
    prove that the radii of the concentric circles are R/√3 and R√(2/3) -/
theorem concentric_circles_equal_areas (R : ℝ) (R₁ R₂ : ℝ) (h₁ : R > 0) :
  (π * R₁^2 = π * R^2 / 3) ∧ 
  (π * R₂^2 - π * R₁^2 = π * R^2 / 3) ∧ 
  (π * R^2 - π * R₂^2 = π * R^2 / 3) →
  (R₁ = R / Real.sqrt 3) ∧ (R₂ = R * Real.sqrt (2/3)) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_equal_areas_l2056_205625


namespace NUMINAMATH_CALUDE_boat_journey_time_l2056_205699

/-- Calculates the total time for a round trip boat journey affected by a stream -/
theorem boat_journey_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 9) 
  (h2 : stream_speed = 6) 
  (h3 : distance = 300) : 
  (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_boat_journey_time_l2056_205699


namespace NUMINAMATH_CALUDE_fraction_simplification_l2056_205681

theorem fraction_simplification :
  (240 : ℚ) / 20 * 6 / 150 * 12 / 5 = 48 / 125 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2056_205681


namespace NUMINAMATH_CALUDE_download_time_360GB_50MBps_l2056_205652

/-- Calculates the download time in hours for a given program size and download speed -/
def downloadTime (programSizeGB : ℕ) (downloadSpeedMBps : ℕ) : ℚ :=
  let programSizeMB := programSizeGB * 1000
  let downloadTimeSeconds := programSizeMB / downloadSpeedMBps
  downloadTimeSeconds / 3600

/-- Proves that downloading a 360 GB program at 50 MB/s takes 2 hours -/
theorem download_time_360GB_50MBps :
  downloadTime 360 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_download_time_360GB_50MBps_l2056_205652


namespace NUMINAMATH_CALUDE_rabbits_ate_27_watermelons_l2056_205656

/-- The number of watermelons eaten by rabbits, given initial and remaining counts. -/
def watermelons_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem stating that 27 watermelons were eaten by rabbits. -/
theorem rabbits_ate_27_watermelons : 
  watermelons_eaten 35 8 = 27 := by sorry

end NUMINAMATH_CALUDE_rabbits_ate_27_watermelons_l2056_205656


namespace NUMINAMATH_CALUDE_train_distance_example_l2056_205640

/-- The total distance traveled by a train given its speed and time -/
def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train traveling at 85 km/h for 4 hours covers 340 km -/
theorem train_distance_example : train_distance 85 4 = 340 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_example_l2056_205640


namespace NUMINAMATH_CALUDE_triangle_equilateral_proof_l2056_205647

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if certain conditions are met, the triangle is equilateral with A = π/3. -/
theorem triangle_equilateral_proof (a b c A B C : ℝ) : 
  0 < A ∧ A < π →  -- Angle A is between 0 and π
  0 < B ∧ B < π →  -- Angle B is between 0 and π
  0 < C ∧ C < π →  -- Angle C is between 0 and π
  A + B + C = π →  -- Sum of angles in a triangle
  2 * a * Real.cos B = 2 * c - b →  -- Given condition
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 4 →  -- Area condition
  a = Real.sqrt 3 →  -- Given side length
  A = π/3 ∧ a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_proof_l2056_205647


namespace NUMINAMATH_CALUDE_winter_holiday_activities_l2056_205631

theorem winter_holiday_activities (total : ℕ) (skating : ℕ) (skiing : ℕ) (both : ℕ) :
  total = 30 →
  skating = 20 →
  skiing = 9 →
  both = 5 →
  total - (skating + skiing - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_winter_holiday_activities_l2056_205631


namespace NUMINAMATH_CALUDE_walmart_cards_sent_eq_two_l2056_205606

/-- Represents the gift card scenario --/
structure GiftCardScenario where
  bestBuyCards : ℕ
  bestBuyValue : ℕ
  walmartCards : ℕ
  walmartValue : ℕ
  sentBestBuy : ℕ
  remainingValue : ℕ

/-- Calculates the number of Walmart gift cards sent --/
def walmartsCardsSent (s : GiftCardScenario) : ℕ :=
  let totalInitialValue := s.bestBuyCards * s.bestBuyValue + s.walmartCards * s.walmartValue
  let sentValue := totalInitialValue - s.remainingValue
  let sentWalmartValue := sentValue - s.sentBestBuy * s.bestBuyValue
  sentWalmartValue / s.walmartValue

/-- Theorem stating the number of Walmart gift cards sent --/
theorem walmart_cards_sent_eq_two (s : GiftCardScenario) 
  (h1 : s.bestBuyCards = 6)
  (h2 : s.bestBuyValue = 500)
  (h3 : s.walmartCards = 9)
  (h4 : s.walmartValue = 200)
  (h5 : s.sentBestBuy = 1)
  (h6 : s.remainingValue = 3900) :
  walmartsCardsSent s = 2 := by
  sorry


end NUMINAMATH_CALUDE_walmart_cards_sent_eq_two_l2056_205606


namespace NUMINAMATH_CALUDE_peach_problem_l2056_205608

theorem peach_problem (martine benjy gabrielle : ℕ) : 
  martine = 2 * benjy + 6 →
  benjy = gabrielle / 3 →
  martine = 16 →
  gabrielle = 15 := by
sorry

end NUMINAMATH_CALUDE_peach_problem_l2056_205608


namespace NUMINAMATH_CALUDE_solve_for_y_l2056_205669

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2056_205669


namespace NUMINAMATH_CALUDE_range_of_m_l2056_205658

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * m * x + 9 ≥ 0) → 
  m ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2056_205658


namespace NUMINAMATH_CALUDE_dad_vacuum_time_l2056_205684

theorem dad_vacuum_time (downstairs upstairs : ℕ) : 
  upstairs = 2 * downstairs + 5 →
  downstairs + upstairs = 38 →
  upstairs = 27 := by
sorry

end NUMINAMATH_CALUDE_dad_vacuum_time_l2056_205684


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l2056_205610

/-- The height difference between Janet's sandcastle and her sister's sandcastle --/
def height_difference : ℝ :=
  let janet_height : ℝ := 3.6666666666666665
  let sister_height : ℝ := 2.3333333333333335
  janet_height - sister_height

/-- Theorem stating that the height difference is 1.333333333333333 feet --/
theorem sandcastle_height_difference :
  height_difference = 1.333333333333333 := by sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l2056_205610


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l2056_205673

/-- Theorem: For a hyperbola with given conditions, a = 1 and b = 4 -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y, x^2 / a - y^2 / b = 1) →  -- Hyperbola equation
  (∃ k, ∀ x y, 2*x + y = 0 → y = k*x) →  -- One asymptote
  (∃ x y, x^2 + y^2 = 5 ∧ y = 0) →  -- One focus
  a = 1 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l2056_205673


namespace NUMINAMATH_CALUDE_third_term_of_specific_sequence_l2056_205693

/-- Represents a geometric sequence of positive integers -/
structure GeometricSequence where
  first_term : ℕ
  common_ratio : ℕ
  first_term_pos : 0 < first_term

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℕ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem third_term_of_specific_sequence :
  ∀ (seq : GeometricSequence),
    seq.first_term = 5 →
    nth_term seq 4 = 320 →
    nth_term seq 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_specific_sequence_l2056_205693


namespace NUMINAMATH_CALUDE_integral_f_equals_pi_over_four_l2056_205626

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + Real.tan (Real.sqrt 2 * x))

theorem integral_f_equals_pi_over_four :
  ∫ x in (0)..(Real.pi / 2), f x = Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_pi_over_four_l2056_205626


namespace NUMINAMATH_CALUDE_price_decrease_proof_l2056_205679

theorem price_decrease_proof (original_price : ℝ) (decrease_percentage : ℝ) (new_price : ℝ) :
  decrease_percentage = 24 →
  new_price = 421.05263157894734 →
  new_price = original_price * (1 - decrease_percentage / 100) :=
by
  sorry

#eval 421.05263157894734 -- To show the exact value used in the problem

end NUMINAMATH_CALUDE_price_decrease_proof_l2056_205679


namespace NUMINAMATH_CALUDE_unique_a_value_l2056_205672

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, 
  (A a ∩ B).Nonempty ∧ 
  Set.Nonempty (A a ∩ B) ∧
  (A a ∩ C) = ∅ ∧
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2056_205672


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2056_205632

theorem original_denominator_proof (d : ℚ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3) / (d + 3) = 2 / 3 →
  d = 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2056_205632


namespace NUMINAMATH_CALUDE_inequality_proof_l2056_205611

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2056_205611


namespace NUMINAMATH_CALUDE_rectangle_diagonal_length_l2056_205662

/-- The length of the diagonal of a rectangle with specific properties -/
theorem rectangle_diagonal_length : ∀ (a b d : ℝ), 
  a > 0 → 
  b = 2 * a → 
  a = 40 * Real.sqrt 2 → 
  d^2 = a^2 + b^2 → 
  d = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_length_l2056_205662


namespace NUMINAMATH_CALUDE_mothers_birthday_knowledge_l2056_205628

/-- Represents the distribution of students' knowledge about their parents' birthdays -/
structure BirthdayKnowledge where
  total : ℕ
  only_father : ℕ
  only_mother : ℕ
  both_parents : ℕ
  neither_parent : ℕ

/-- Theorem stating that 22 students know their mother's birthday -/
theorem mothers_birthday_knowledge (bk : BirthdayKnowledge) 
  (h1 : bk.total = 40)
  (h2 : bk.only_father = 10)
  (h3 : bk.only_mother = 12)
  (h4 : bk.both_parents = 22)
  (h5 : bk.neither_parent = 26)
  (h6 : bk.total = bk.only_father + bk.only_mother + bk.both_parents + bk.neither_parent) :
  bk.only_mother + bk.both_parents = 22 := by
  sorry

end NUMINAMATH_CALUDE_mothers_birthday_knowledge_l2056_205628


namespace NUMINAMATH_CALUDE_exam_score_distribution_l2056_205670

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  mean : ℝ
  stdDev : ℝ
  totalStudents : ℕ

/-- Calculates the number of students who scored at least a given threshold -/
def studentsAboveThreshold (dist : ScoreDistribution) (threshold : ℝ) : ℕ :=
  sorry

/-- The exam score distribution -/
def examScores : ScoreDistribution :=
  { mean := 110
    stdDev := 10
    totalStudents := 50 }

theorem exam_score_distribution :
  (studentsAboveThreshold examScores 90 = 49) ∧
  (studentsAboveThreshold examScores 120 = 8) := by
  sorry

end NUMINAMATH_CALUDE_exam_score_distribution_l2056_205670


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2056_205649

def M : Set ℝ := {x | 1 < x ∧ x < 4}

theorem sqrt_inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  |Real.sqrt (a * b) - 2| < |2 * Real.sqrt a - Real.sqrt b| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2056_205649


namespace NUMINAMATH_CALUDE_salary_problem_l2056_205615

theorem salary_problem (total : ℝ) (a_spend_percent : ℝ) (b_spend_percent : ℝ)
  (h_total : total = 7000)
  (h_a_spend : a_spend_percent = 95)
  (h_b_spend : b_spend_percent = 85)
  (h_equal_savings : (100 - a_spend_percent) * a_salary = (100 - b_spend_percent) * (total - a_salary)) :
  a_salary = 5250 :=
by
  sorry

#check salary_problem

end NUMINAMATH_CALUDE_salary_problem_l2056_205615


namespace NUMINAMATH_CALUDE_horner_method_correct_f_3_equals_283_l2056_205661

def f (x : ℝ) : ℝ := x^5 + x^3 + x^2 + x + 1

def horner_eval (x : ℝ) : ℝ := ((((1 * x + 0) * x + 1) * x + 1) * x + 1) * x + 1

theorem horner_method_correct (x : ℝ) : f x = horner_eval x := by sorry

theorem f_3_equals_283 : f 3 = 283 := by sorry

end NUMINAMATH_CALUDE_horner_method_correct_f_3_equals_283_l2056_205661


namespace NUMINAMATH_CALUDE_problem_statement_l2056_205612

theorem problem_statement : (12 : ℕ)^3 * 6^2 / 432 = 144 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2056_205612


namespace NUMINAMATH_CALUDE_max_increase_year_1998_l2056_205641

def sales : Fin 11 → ℝ
  | 0 => 3.0
  | 1 => 4.5
  | 2 => 5.1
  | 3 => 7.0
  | 4 => 8.5
  | 5 => 9.7
  | 6 => 10.7
  | 7 => 12.0
  | 8 => 13.2
  | 9 => 13.7
  | 10 => 7.5

def year_increase (i : Fin 10) : ℝ :=
  sales (i.succ) - sales i

theorem max_increase_year_1998 :
  ∃ i : Fin 10, (i.val + 1995 = 1998) ∧
    ∀ j : Fin 10, year_increase j ≤ year_increase i :=
by sorry

end NUMINAMATH_CALUDE_max_increase_year_1998_l2056_205641


namespace NUMINAMATH_CALUDE_average_work_hours_l2056_205616

theorem average_work_hours (total_people : ℕ) (people_on_duty : ℕ) (hours_per_day : ℕ) :
  total_people = 8 →
  people_on_duty = 3 →
  hours_per_day = 24 →
  (hours_per_day * people_on_duty : ℚ) / total_people = 9 := by
sorry

end NUMINAMATH_CALUDE_average_work_hours_l2056_205616


namespace NUMINAMATH_CALUDE_total_books_combined_l2056_205613

theorem total_books_combined (keith_books jason_books amanda_books sophie_books : ℕ)
  (h1 : keith_books = 20)
  (h2 : jason_books = 21)
  (h3 : amanda_books = 15)
  (h4 : sophie_books = 30) :
  keith_books + jason_books + amanda_books + sophie_books = 86 := by
sorry

end NUMINAMATH_CALUDE_total_books_combined_l2056_205613


namespace NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l2056_205697

theorem saltwater_animals_per_aquarium 
  (num_aquariums : ℕ) 
  (total_animals : ℕ) 
  (h1 : num_aquariums = 26) 
  (h2 : total_animals = 52) 
  (h3 : total_animals % num_aquariums = 0) :
  total_animals / num_aquariums = 2 := by
sorry

end NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l2056_205697


namespace NUMINAMATH_CALUDE_condition_equivalence_l2056_205627

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem statement
theorem condition_equivalence (a b : ℝ) :
  (a + b > 0) ↔ (f a + f b > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l2056_205627


namespace NUMINAMATH_CALUDE_power_of_square_l2056_205630

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l2056_205630


namespace NUMINAMATH_CALUDE_age_puzzle_solution_l2056_205642

/-- Represents a person in the age puzzle -/
structure Person where
  name : String
  age : Nat

/-- The conditions of the age puzzle -/
def AgePuzzle (tamara lena marina : Person) : Prop :=
  tamara.age = lena.age - 2 ∧
  tamara.age = marina.age + 1 ∧
  lena.age = marina.age + 3 ∧
  marina.age < tamara.age

/-- The theorem stating the unique solution to the age puzzle -/
theorem age_puzzle_solution :
  ∃! (tamara lena marina : Person),
    tamara.name = "Tamara" ∧
    lena.name = "Lena" ∧
    marina.name = "Marina" ∧
    AgePuzzle tamara lena marina ∧
    tamara.age = 23 ∧
    lena.age = 25 ∧
    marina.age = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_solution_l2056_205642


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l2056_205683

theorem smallest_angle_solution (y : Real) : 
  (∀ θ : Real, θ > 0 ∧ θ < y → 10 * Real.sin θ * Real.cos θ ^ 3 - 10 * Real.sin θ ^ 3 * Real.cos θ ≠ Real.sqrt 2) ∧
  (10 * Real.sin y * Real.cos y ^ 3 - 10 * Real.sin y ^ 3 * Real.cos y = Real.sqrt 2) →
  y = 11.25 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l2056_205683


namespace NUMINAMATH_CALUDE_bryans_total_amount_l2056_205609

/-- The total amount received from selling precious stones -/
def total_amount (num_stones : ℕ) (price_per_stone : ℕ) : ℕ :=
  num_stones * price_per_stone

/-- Theorem: Bryan's total amount from selling 8 stones at 1785 dollars each is 14280 dollars -/
theorem bryans_total_amount :
  total_amount 8 1785 = 14280 := by
  sorry

end NUMINAMATH_CALUDE_bryans_total_amount_l2056_205609


namespace NUMINAMATH_CALUDE_euler_family_mean_age_is_68_over_7_l2056_205687

/-- The mean age of the Euler family's children -/
def euler_family_mean_age : ℚ :=
  let ages : List ℕ := [6, 6, 6, 6, 12, 16, 16]
  (ages.sum : ℚ) / ages.length

/-- Theorem stating the mean age of the Euler family's children -/
theorem euler_family_mean_age_is_68_over_7 :
  euler_family_mean_age = 68 / 7 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_is_68_over_7_l2056_205687


namespace NUMINAMATH_CALUDE_archer_probability_l2056_205663

def prob_not_both_hit (prob_A prob_B : ℚ) : ℚ :=
  1 - (prob_A * prob_B)

theorem archer_probability :
  let prob_A : ℚ := 1/3
  let prob_B : ℚ := 1/2
  prob_not_both_hit prob_A prob_B = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_archer_probability_l2056_205663


namespace NUMINAMATH_CALUDE_not_exist_prime_power_of_six_plus_nineteen_l2056_205690

theorem not_exist_prime_power_of_six_plus_nineteen :
  ∀ n : ℕ, ¬ Nat.Prime (6^n + 19) := by
  sorry

end NUMINAMATH_CALUDE_not_exist_prime_power_of_six_plus_nineteen_l2056_205690


namespace NUMINAMATH_CALUDE_length_CD_l2056_205602

theorem length_CD (AB D C : ℝ) : 
  AB = 48 →                 -- Length of AB is 48
  D = AB / 3 →              -- AD is 1/3 of AB
  C = AB / 2 →              -- C is the midpoint of AB
  C - D = 8 :=              -- Length of CD is 8
by
  sorry


end NUMINAMATH_CALUDE_length_CD_l2056_205602


namespace NUMINAMATH_CALUDE_square_area_on_parallel_lines_l2056_205688

/-- Represents a line in a 2D plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Checks if three lines are parallel -/
def are_parallel (l1 l2 l3 : Line) : Prop := sorry

/-- Calculates the perpendicular distance between two lines -/
def perpendicular_distance (l1 l2 : Line) : ℝ := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Calculates the area of a square -/
def square_area (s : Square) : ℝ := sorry

/-- The main theorem -/
theorem square_area_on_parallel_lines 
  (l1 l2 l3 : Line) 
  (s : Square) :
  are_parallel l1 l2 l3 →
  perpendicular_distance l1 l2 = 3 →
  perpendicular_distance l2 l3 = 3 →
  point_on_line s.a l1 →
  point_on_line s.b l3 →
  point_on_line s.c l2 →
  square_area s = 45 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parallel_lines_l2056_205688
