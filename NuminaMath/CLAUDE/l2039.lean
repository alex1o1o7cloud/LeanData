import Mathlib

namespace NUMINAMATH_CALUDE_divisor_count_l2039_203946

/-- 
Given a positive integer n such that 140n^4 has exactly 140 positive integer divisors,
prove that 100n^5 has exactly 24 positive integer divisors.
-/
theorem divisor_count (n : ℕ+) 
  (h : (Finset.card (Nat.divisors (140 * n^4))) = 140) : 
  (Finset.card (Nat.divisors (100 * n^5))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_l2039_203946


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2039_203963

/-- Represents the problem from "The Compendious Book on Calculation by Completion and Balancing" --/
theorem ancient_chinese_math_problem (x y : ℕ) : 
  (∀ (room_capacity : ℕ), 
    (room_capacity = 7 → 7 * x + 7 = y) ∧ 
    (room_capacity = 9 → 9 * (x - 1) = y)) ↔ 
  (7 * x + 7 = y ∧ 9 * (x - 1) = y) :=
sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2039_203963


namespace NUMINAMATH_CALUDE_credit_sales_ratio_l2039_203973

theorem credit_sales_ratio (total_sales cash_sales : ℚ) 
  (h1 : total_sales = 80)
  (h2 : cash_sales = 48) :
  (total_sales - cash_sales) / total_sales = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_credit_sales_ratio_l2039_203973


namespace NUMINAMATH_CALUDE_quotient_problem_l2039_203921

theorem quotient_problem (dividend : ℕ) (k : ℕ) (divisor : ℕ) :
  dividend = 64 → k = 8 → divisor = k → dividend / divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l2039_203921


namespace NUMINAMATH_CALUDE_solve_otimes_equation_l2039_203934

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

-- Theorem statement
theorem solve_otimes_equation : 
  ∃! x : ℝ, otimes (-4) (x + 3) = 6 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_solve_otimes_equation_l2039_203934


namespace NUMINAMATH_CALUDE_train_speed_l2039_203914

theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 500) (h2 : crossing_time = 10) :
  train_length / crossing_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2039_203914


namespace NUMINAMATH_CALUDE_angle_Y_measure_l2039_203900

-- Define the structure for lines and angles
structure Geometry where
  Line : Type
  Angle : Type
  measure : Angle → ℝ
  parallel : Line → Line → Prop
  intersect : Line → Line → Prop
  angleOn : Line → Angle → Prop
  transversal : Line → Line → Line → Prop

-- State the theorem
theorem angle_Y_measure (G : Geometry) 
  (p q t yz : G.Line) (X Z Y : G.Angle) :
  G.parallel p q →
  G.parallel p yz →
  G.parallel q yz →
  G.transversal t p q →
  G.intersect t yz →
  G.angleOn p X →
  G.angleOn q Z →
  G.measure X = 100 →
  G.measure Z = 110 →
  G.measure Y = 40 := by
  sorry


end NUMINAMATH_CALUDE_angle_Y_measure_l2039_203900


namespace NUMINAMATH_CALUDE_square_sum_equals_69_l2039_203992

/-- Given a system of equations, prove that x₀² + y₀² = 69 -/
theorem square_sum_equals_69 
  (x₀ y₀ c : ℝ) 
  (h1 : x₀ * y₀ = 6)
  (h2 : x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2) :
  x₀^2 + y₀^2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_69_l2039_203992


namespace NUMINAMATH_CALUDE_repeating_fraction_sixteen_equals_five_thirty_ninths_l2039_203981

/-- Given a positive integer k, this function returns the value of the infinite geometric series
    4/k + 5/k^2 + 4/k^3 + 5/k^4 + ... -/
def repeating_fraction (k : ℕ) : ℚ :=
  (4 * k + 5) / (k^2 - 1)

/-- The theorem states that for k = 16, the repeating fraction equals 5/39 -/
theorem repeating_fraction_sixteen_equals_five_thirty_ninths :
  repeating_fraction 16 = 5 / 39 := by
  sorry

end NUMINAMATH_CALUDE_repeating_fraction_sixteen_equals_five_thirty_ninths_l2039_203981


namespace NUMINAMATH_CALUDE_negative_angle_quadrant_l2039_203928

/-- If an angle α is in the third quadrant, then -α is in the second quadrant -/
theorem negative_angle_quadrant (α : Real) : 
  (∃ k : ℤ, k * 2 * π + π < α ∧ α < k * 2 * π + 3 * π / 2) → 
  (∃ m : ℤ, m * 2 * π + π / 2 < -α ∧ -α < m * 2 * π + π) :=
by sorry

end NUMINAMATH_CALUDE_negative_angle_quadrant_l2039_203928


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2039_203931

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := 2*x^2 - 2*x + 1

/-- The main theorem about the quadratic function f -/
theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2 * x) ∧
  f 0 = 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, -1/2 ≤ f x ∧ f x ≤ 1) ∧
  (∀ a : ℝ,
    (a ≤ -1/2 → ∀ x ∈ Set.Icc a (a + 1), 2*a^2 + 2*a + 3 ≤ f x ∧ f x ≤ 2*a^2 - 2*a + 1) ∧
    (-1/2 < a ∧ a ≤ 0 → ∀ x ∈ Set.Icc a (a + 1), -1/2 ≤ f x ∧ f x ≤ 2*a^2 - 2*a + 1) ∧
    (0 ≤ a ∧ a < 1/2 → ∀ x ∈ Set.Icc a (a + 1), -1/2 ≤ f x ∧ f x ≤ 2*a^2 + 2*a + 3) ∧
    (1/2 ≤ a → ∀ x ∈ Set.Icc a (a + 1), 2*a^2 - 2*a + 1 ≤ f x ∧ f x ≤ 2*a^2 + 2*a + 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2039_203931


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_T_l2039_203903

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  value : Nat
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Generates the next term in the sequence based on the current term -/
def next_term (n : ThreeDigitInt) : ThreeDigitInt :=
  { value := (n.value % 100) * 10 + (n.value / 100),
    is_three_digit := sorry }

/-- Generates a sequence of three terms starting with the given number -/
def generate_sequence (start : ThreeDigitInt) : Fin 3 → ThreeDigitInt
| 0 => start
| 1 => next_term start
| 2 => next_term (next_term start)

/-- Calculates the sum of all terms in a sequence -/
def sequence_sum (start : ThreeDigitInt) : Nat :=
  (generate_sequence start 0).value +
  (generate_sequence start 1).value +
  (generate_sequence start 2).value

/-- The starting number for the first sequence -/
def start1 : ThreeDigitInt :=
  { value := 312,
    is_three_digit := sorry }

/-- The starting number for the second sequence -/
def start2 : ThreeDigitInt :=
  { value := 231,
    is_three_digit := sorry }

/-- The sum of all terms from both sequences -/
def T : Nat := sequence_sum start1 + sequence_sum start2

theorem largest_prime_factor_of_T :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ T ∧ ∀ (q : Nat), Nat.Prime q → q ∣ T → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_T_l2039_203903


namespace NUMINAMATH_CALUDE_circle_in_diamond_l2039_203953

-- Define the sets M and N
def M (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 2}

-- State the theorem
theorem circle_in_diamond (a : ℝ) (h : a > 0) :
  M a ⊆ N ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_circle_in_diamond_l2039_203953


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l2039_203976

theorem camping_trip_percentage (total_students : ℝ) (students_over_100 : ℝ) (students_100_or_less : ℝ) :
  students_over_100 = 0.16 * total_students →
  students_over_100 + students_100_or_less = 0.64 * total_students →
  (students_over_100 + students_100_or_less) / total_students = 0.64 :=
by
  sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l2039_203976


namespace NUMINAMATH_CALUDE_basketball_team_callback_l2039_203979

/-- The number of students called back for the basketball team. -/
def students_called_back (girls boys not_called : ℕ) : ℕ :=
  girls + boys - not_called

/-- Theorem stating that 26 students were called back for the basketball team. -/
theorem basketball_team_callback : students_called_back 39 4 17 = 26 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_callback_l2039_203979


namespace NUMINAMATH_CALUDE_special_prize_winner_l2039_203915

-- Define the type for students
inductive Student : Type
  | one | two | three | four | five | six

-- Define the type for predictors
inductive Predictor : Type
  | A | B | C | D

-- Define the prediction function
def prediction (p : Predictor) (s : Student) : Prop :=
  match p with
  | Predictor.A => s = Student.one ∨ s = Student.two
  | Predictor.B => s ≠ Student.three
  | Predictor.C => s ≠ Student.four ∧ s ≠ Student.five ∧ s ≠ Student.six
  | Predictor.D => s = Student.four ∨ s = Student.five ∨ s = Student.six

-- Define the theorem
theorem special_prize_winner (winner : Student) :
  (∃! p : Predictor, prediction p winner) →
  winner = Student.three :=
by sorry

end NUMINAMATH_CALUDE_special_prize_winner_l2039_203915


namespace NUMINAMATH_CALUDE_range_of_x2_plus_y2_l2039_203940

theorem range_of_x2_plus_y2 (x y : ℝ) (h : (x + 2)^2 + y^2/4 = 1) :
  ∃ (min max : ℝ), min = 1 ∧ max = 28/3 ∧
  (x^2 + y^2 ≥ min ∧ x^2 + y^2 ≤ max) ∧
  (∀ z, (∃ a b : ℝ, (a + 2)^2 + b^2/4 = 1 ∧ z = a^2 + b^2) → z ≥ min ∧ z ≤ max) :=
sorry

end NUMINAMATH_CALUDE_range_of_x2_plus_y2_l2039_203940


namespace NUMINAMATH_CALUDE_inscribed_cube_properties_l2039_203974

/-- A cube inscribed in a hemisphere -/
structure InscribedCube (R : ℝ) where
  -- The edge length of the cube
  a : ℝ
  -- The distance from the center of the hemisphere base to a vertex of the square face
  r : ℝ
  -- Four vertices of the cube are on the surface of the hemisphere
  vertices_on_surface : a ^ 2 + r ^ 2 = R ^ 2
  -- Four vertices of the cube are on the circular boundary of the hemisphere's base
  vertices_on_base : r = a * (Real.sqrt 2) / 2

/-- The edge length and distance properties of a cube inscribed in a hemisphere -/
theorem inscribed_cube_properties (R : ℝ) (h : R > 0) :
  ∃ (cube : InscribedCube R),
    cube.a = R * Real.sqrt (2/3) ∧
    cube.r = R / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_properties_l2039_203974


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2039_203923

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2039_203923


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2039_203972

-- Problem 1
theorem problem_one (a b c : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : |c| = 3) (horder : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 := by sorry

-- Problem 2
theorem problem_two (a b c d : ℚ) (hab : |a - b| ≤ 9) (hcd : |c - d| ≤ 16) (habcd : |a - b - c + d| = 25) :
  |b - a| - |d - c| = -7 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2039_203972


namespace NUMINAMATH_CALUDE_project_selection_count_l2039_203943

theorem project_selection_count : ∀ (n_key m_key n_general m_general : ℕ),
  n_key = 4 →
  m_key = 2 →
  n_general = 6 →
  m_general = 2 →
  (Nat.choose n_key m_key * Nat.choose (n_general - 1) (m_general - 1)) +
  (Nat.choose (n_key - 1) (m_key - 1) * Nat.choose n_general m_general) -
  (Nat.choose (n_key - 1) (m_key - 1) * Nat.choose (n_general - 1) (m_general - 1)) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_project_selection_count_l2039_203943


namespace NUMINAMATH_CALUDE_train_cars_problem_l2039_203926

theorem train_cars_problem (total_cars engine_and_caboose passenger_cars cargo_cars : ℕ) :
  total_cars = 71 →
  engine_and_caboose = 2 →
  cargo_cars = passenger_cars / 2 + 3 →
  total_cars = passenger_cars + cargo_cars + engine_and_caboose →
  passenger_cars = 44 := by
sorry

end NUMINAMATH_CALUDE_train_cars_problem_l2039_203926


namespace NUMINAMATH_CALUDE_expression_evaluation_l2039_203956

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 11) + 2 = -x^4 + 3*x^3 - 5*x^2 + 11*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2039_203956


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l2039_203930

theorem product_of_two_digit_numbers (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) → 
  (10 ≤ b ∧ b < 100) → 
  a * b = 4680 → 
  min a b = 40 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l2039_203930


namespace NUMINAMATH_CALUDE_lcm_of_308_and_275_l2039_203965

theorem lcm_of_308_and_275 :
  let a := 308
  let b := 275
  let hcf := 11
  let lcm := Nat.lcm a b
  (Nat.gcd a b = hcf) → (lcm = 7700) := by
sorry

end NUMINAMATH_CALUDE_lcm_of_308_and_275_l2039_203965


namespace NUMINAMATH_CALUDE_total_money_earned_l2039_203920

/-- The price per kg of fish in dollars -/
def price_per_kg : ℝ := 20

/-- The amount of fish in kg caught in the past four months -/
def past_four_months_catch : ℝ := 80

/-- The amount of fish in kg caught today -/
def today_catch : ℝ := 2 * past_four_months_catch

/-- The total amount of fish in kg caught in the past four months including today -/
def total_catch : ℝ := past_four_months_catch + today_catch

/-- Theorem: The total money earned by Erica in the past four months including today is $4800 -/
theorem total_money_earned : price_per_kg * total_catch = 4800 := by
  sorry

end NUMINAMATH_CALUDE_total_money_earned_l2039_203920


namespace NUMINAMATH_CALUDE_smallest_number_l2039_203958

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def number_a : Nat := base_to_decimal [5, 8] 9
def number_b : Nat := base_to_decimal [0, 1, 2] 6
def number_c : Nat := base_to_decimal [0, 0, 0, 1] 4
def number_d : Nat := base_to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number :
  number_d < number_a ∧ number_d < number_b ∧ number_d < number_c :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l2039_203958


namespace NUMINAMATH_CALUDE_sqrt_diff_inequality_l2039_203970

theorem sqrt_diff_inequality (k : ℕ) (h : k ≥ 2) :
  Real.sqrt k - Real.sqrt (k - 1) > Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_sqrt_diff_inequality_l2039_203970


namespace NUMINAMATH_CALUDE_archery_score_distribution_l2039_203936

theorem archery_score_distribution :
  ∃! (a b c d : ℕ),
    a + b + c + d = 10 ∧
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 ∧
    8*a + 12*b + 14*c + 18*d = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_archery_score_distribution_l2039_203936


namespace NUMINAMATH_CALUDE_range_of_a_l2039_203938

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (∀ y ∈ Set.Icc (-4) 32, ∃ x ∈ Set.Icc (-4) a, f x = y) →
  a ∈ Set.Icc 2 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2039_203938


namespace NUMINAMATH_CALUDE_actual_average_height_l2039_203907

/-- The number of boys in the class -/
def num_boys : ℕ := 60

/-- The initial calculated average height in cm -/
def initial_avg : ℝ := 185

/-- The recorded heights of the three boys with errors -/
def recorded_heights : Fin 3 → ℝ := ![170, 195, 160]

/-- The actual heights of the three boys -/
def actual_heights : Fin 3 → ℝ := ![140, 165, 190]

/-- The actual average height of the boys in the class -/
def actual_avg : ℝ := 184.50

theorem actual_average_height :
  let total_initial := initial_avg * num_boys
  let total_difference := (recorded_heights 0 - actual_heights 0) +
                          (recorded_heights 1 - actual_heights 1) +
                          (recorded_heights 2 - actual_heights 2)
  let corrected_total := total_initial - total_difference
  corrected_total / num_boys = actual_avg := by sorry

end NUMINAMATH_CALUDE_actual_average_height_l2039_203907


namespace NUMINAMATH_CALUDE_train_distance_problem_l2039_203924

theorem train_distance_problem (speed1 speed2 distance_difference : ℝ) 
  (h1 : speed1 = 16)
  (h2 : speed2 = 21)
  (h3 : distance_difference = 60)
  (time : ℝ)
  (h4 : time > 0)
  (distance1 : ℝ)
  (h5 : distance1 = speed1 * time)
  (distance2 : ℝ)
  (h6 : distance2 = speed2 * time)
  (h7 : distance2 = distance1 + distance_difference) :
  distance1 + distance2 = 444 := by
sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2039_203924


namespace NUMINAMATH_CALUDE_saree_price_problem_l2039_203964

theorem saree_price_problem (P : ℝ) : 
  P * (1 - 0.1) * (1 - 0.05) = 171 → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_problem_l2039_203964


namespace NUMINAMATH_CALUDE_six_ronna_scientific_notation_l2039_203939

/-- Represents the number of zeros after the number for the "ronna" prefix -/
def ronna_zeros : ℕ := 27

/-- Converts a number with the "ronna" prefix to scientific notation -/
def ronna_to_scientific (n : ℝ) : ℝ := n * (10 ^ ronna_zeros)

/-- Theorem stating that 6 ronna is equal to 6 × 10^27 -/
theorem six_ronna_scientific_notation : ronna_to_scientific 6 = 6 * (10 ^ 27) := by
  sorry

end NUMINAMATH_CALUDE_six_ronna_scientific_notation_l2039_203939


namespace NUMINAMATH_CALUDE_percentage_of_girls_in_class_l2039_203969

theorem percentage_of_girls_in_class (girls boys : ℕ) (h1 : girls = 10) (h2 : boys = 15) :
  (girls : ℚ) / ((girls : ℚ) + (boys : ℚ)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_in_class_l2039_203969


namespace NUMINAMATH_CALUDE_snail_distance_is_20_l2039_203904

def snail_path : List ℤ := [0, 4, -3, 6]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

def total_distance (path : List ℤ) : ℕ :=
  match path with
  | [] => 0
  | [_] => 0
  | x :: y :: rest => distance x y + total_distance (y :: rest)

theorem snail_distance_is_20 : total_distance snail_path = 20 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_is_20_l2039_203904


namespace NUMINAMATH_CALUDE_horner_method_v₃_l2039_203949

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 3*x + 2

def horner_v₃ (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v₀ := 1
  let v₁ := x + (-5)
  let v₂ := v₁ * x + 6
  v₂ * x + 0

theorem horner_method_v₃ :
  horner_v₃ f (-2) = -40 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l2039_203949


namespace NUMINAMATH_CALUDE_min_distance_at_median_l2039_203901

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- Represents the distance between two points -/
def distance (p q : Point) : ℝ :=
  |p.x - q.x|

/-- Given 9 points on a line, the sum of distances from an arbitrary point to all 9 points
    is minimized when the arbitrary point coincides with the 5th point -/
theorem min_distance_at_median (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ p₉ : Point) 
    (h : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ 
         p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x ∧ p₈.x < p₉.x) :
  ∀ p : Point, 
    distance p p₁ + distance p p₂ + distance p p₃ + distance p p₄ + 
    distance p p₅ + distance p p₆ + distance p p₇ + distance p p₈ + distance p p₉ ≥
    distance p₅ p₁ + distance p₅ p₂ + distance p₅ p₃ + distance p₅ p₄ + 
    distance p₅ p₅ + distance p₅ p₆ + distance p₅ p₇ + distance p₅ p₈ + distance p₅ p₉ :=
by sorry

end NUMINAMATH_CALUDE_min_distance_at_median_l2039_203901


namespace NUMINAMATH_CALUDE_fraction_difference_l2039_203948

theorem fraction_difference (a b : ℝ) (h : b / a = 2) :
  b / (a + b) - a / (a + b) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_l2039_203948


namespace NUMINAMATH_CALUDE_square_sum_equality_l2039_203983

theorem square_sum_equality (x y : ℝ) (h : x + y = -2) : x^2 + y^2 + 2*x*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l2039_203983


namespace NUMINAMATH_CALUDE_iced_tea_consumption_iced_tea_consumption_is_198_l2039_203987

theorem iced_tea_consumption : ℝ → Prop :=
  fun total_consumption =>
    ∃ (rob_size : ℝ),
      let mary_size : ℝ := 1.75 * rob_size
      let rob_remaining : ℝ := (1/3) * rob_size
      let mary_remaining : ℝ := (1/3) * mary_size
      let mary_share : ℝ := (1/4) * mary_remaining + 3
      let rob_total : ℝ := (2/3) * rob_size + mary_share
      let mary_total : ℝ := (2/3) * mary_size - mary_share
      rob_total = mary_total ∧
      total_consumption = rob_size + mary_size ∧
      total_consumption = 198

theorem iced_tea_consumption_is_198 : iced_tea_consumption 198 := by
  sorry

end NUMINAMATH_CALUDE_iced_tea_consumption_iced_tea_consumption_is_198_l2039_203987


namespace NUMINAMATH_CALUDE_afternoon_campers_count_l2039_203997

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 44

/-- The difference between morning and afternoon campers -/
def difference : ℕ := 5

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := morning_campers - difference

theorem afternoon_campers_count : afternoon_campers = 39 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_campers_count_l2039_203997


namespace NUMINAMATH_CALUDE_prime_power_sum_l2039_203982

theorem prime_power_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 3250 → 2*w + 3*x + 4*y + 5*z = 19 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2039_203982


namespace NUMINAMATH_CALUDE_license_plate_count_l2039_203971

/-- The number of letters in the Rotokas alphabet -/
def rotokas_alphabet_size : ℕ := 12

/-- The set of allowed first letters -/
def first_letters : Finset Char := {'G', 'K', 'P'}

/-- The required last letter -/
def last_letter : Char := 'T'

/-- The forbidden letter -/
def forbidden_letter : Char := 'R'

/-- The length of the license plate -/
def license_plate_length : ℕ := 5

/-- Calculates the number of valid license plates -/
def count_license_plates : ℕ :=
  first_letters.card * (rotokas_alphabet_size - 5) * (rotokas_alphabet_size - 6) * (rotokas_alphabet_size - 7)

theorem license_plate_count :
  count_license_plates = 630 :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l2039_203971


namespace NUMINAMATH_CALUDE_parabola_directrix_l2039_203947

/-- The directrix of the parabola y = (x^2 - 8x + 12) / 16 is y = -17/64 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => (x^2 - 8*x + 12) / 16
  ∃ (a b c : ℝ), (∀ x, f x = a * (x - b)^2 + c) ∧
                 (a ≠ 0) ∧
                 (c - 1 / (4 * a) = -17/64) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2039_203947


namespace NUMINAMATH_CALUDE_speed_conversion_l2039_203922

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed_mps : ℝ := 23.3352

/-- Calculated speed in kilometers per hour -/
def calculated_speed_kmph : ℝ := 84.00672

theorem speed_conversion : given_speed_mps * mps_to_kmph = calculated_speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l2039_203922


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l2039_203996

theorem circle_diameter_ratio (D C : ℝ → Prop) (r_D r_C : ℝ) : 
  (∀ x, C x → D x) →  -- C is inside D
  (2 * r_D = 20) →    -- Diameter of D is 20 cm
  (π * r_D^2 - π * r_C^2 = 2 * π * r_C^2) →  -- Ratio of shaded area to area of C is 2:1
  2 * r_C = 20 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l2039_203996


namespace NUMINAMATH_CALUDE_triangle_side_length_l2039_203989

theorem triangle_side_length (a b c : ℝ) (S : ℝ) (hA : a = 4) (hB : b = 5) (hS : S = 5 * Real.sqrt 3) :
  c = Real.sqrt 21 ∨ c = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2039_203989


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l2039_203960

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A number is a perfect square if it's the square of some natural number. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- The main theorem stating that an arithmetic progression of positive integers
    with at least one perfect square contains infinitely many perfect squares. -/
theorem infinitely_many_perfect_squares
  (a : ℕ → ℕ)
  (h_arith : ArithmeticProgression a)
  (h_positive : ∀ n, a n > 0)
  (h_one_square : ∃ n, IsPerfectSquare (a n)) :
  ∀ k : ℕ, ∃ n > k, IsPerfectSquare (a n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l2039_203960


namespace NUMINAMATH_CALUDE_christian_initial_savings_l2039_203961

/-- The price of the perfume in dollars -/
def perfume_price : ℚ := 50

/-- Sue's initial savings in dollars -/
def sue_initial : ℚ := 7

/-- The number of yards Christian mowed -/
def yards_mowed : ℕ := 4

/-- The price Christian charged per yard in dollars -/
def price_per_yard : ℚ := 5

/-- The number of dogs Sue walked -/
def dogs_walked : ℕ := 6

/-- The price Sue charged per dog in dollars -/
def price_per_dog : ℚ := 2

/-- The additional amount needed in dollars -/
def additional_needed : ℚ := 6

/-- Christian's earnings from mowing yards -/
def christian_earnings : ℚ := yards_mowed * price_per_yard

/-- Sue's earnings from walking dogs -/
def sue_earnings : ℚ := dogs_walked * price_per_dog

/-- Total money they have after their work -/
def total_after_work : ℚ := christian_earnings + sue_earnings + sue_initial

/-- Christian's initial savings -/
def christian_initial : ℚ := perfume_price - total_after_work - additional_needed

theorem christian_initial_savings : christian_initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_christian_initial_savings_l2039_203961


namespace NUMINAMATH_CALUDE_reciprocal_location_l2039_203998

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- A complex number is inside the unit circle if its norm is less than 1 -/
def inside_unit_circle (z : ℂ) : Prop :=
  Complex.abs z < 1

/-- A complex number is in the second quadrant if its real part is negative and imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- A complex number is outside the unit circle if its norm is greater than 1 -/
def outside_unit_circle (z : ℂ) : Prop :=
  Complex.abs z > 1

theorem reciprocal_location (F : ℂ) :
  in_third_quadrant F ∧ inside_unit_circle F →
  in_second_quadrant (1 / F) ∧ outside_unit_circle (1 / F) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_location_l2039_203998


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2039_203942

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 8) * (x + 1) = -12
def equation2 (x : ℝ) : Prop := 2 * x^2 + 4 * x - 1 = 0

-- Theorem for equation 1
theorem solution_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = -4 ∧ x₂ = -5 :=
by sorry

-- Theorem for equation 2
theorem solution_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation2 x₁ ∧ equation2 x₂ ∧
  x₁ = (-2 + Real.sqrt 6) / 2 ∧ x₂ = (-2 - Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2039_203942


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l2039_203925

/-- Represents a seating arrangement for two families in two cars -/
structure SeatingArrangement where
  audi : Finset (Fin 6)
  jetta : Finset (Fin 6)

/-- The set of all valid seating arrangements -/
def validArrangements : Finset SeatingArrangement :=
  sorry

/-- The number of adults in the group -/
def numAdults : Nat := 4

/-- The number of children in the group -/
def numChildren : Nat := 2

/-- The maximum capacity of each car -/
def maxCapacity : Nat := 4

/-- Theorem stating the number of valid seating arrangements -/
theorem count_valid_arrangements :
  Finset.card validArrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l2039_203925


namespace NUMINAMATH_CALUDE_janice_age_l2039_203954

def current_year : ℕ := 2021
def mark_birth_year : ℕ := 1976

def graham_age_difference : ℕ := 3

theorem janice_age :
  let mark_age : ℕ := current_year - mark_birth_year
  let graham_age : ℕ := mark_age - graham_age_difference
  let janice_age : ℕ := graham_age / 2
  janice_age = 21 := by sorry

end NUMINAMATH_CALUDE_janice_age_l2039_203954


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2039_203909

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 599 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 :
  (is_prime 599) ∧ 
  (digit_sum 599 = 23) ∧ 
  (∀ n : ℕ, n < 599 → ¬(is_prime n ∧ digit_sum n = 23)) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2039_203909


namespace NUMINAMATH_CALUDE_arrangements_theorem_l2039_203966

-- Define the number of officers and intersections
def num_officers : ℕ := 5
def num_intersections : ℕ := 3

-- Define the function to calculate the number of arrangements
def arrangements_with_AB_together : ℕ := sorry

-- State the theorem
theorem arrangements_theorem : arrangements_with_AB_together = 36 := by sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l2039_203966


namespace NUMINAMATH_CALUDE_overlapping_circles_common_chord_l2039_203913

theorem overlapping_circles_common_chord 
  (r : ℝ) 
  (h1 : r = 12) 
  (h2 : r > 0) : 
  let d := r -- distance between centers
  let x := Real.sqrt (r^2 - (r/2)^2) -- half-length of common chord
  2 * x = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_overlapping_circles_common_chord_l2039_203913


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l2039_203952

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125 degrees. -/
theorem supplement_of_complement_of_35_degrees : 
  let original_angle : ℝ := 35
  let complement : ℝ := 90 - original_angle
  let supplement : ℝ := 180 - complement
  supplement = 125 := by
sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l2039_203952


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2039_203905

/-- A rectangle with side lengths 2 and 1 -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : (B.1 - A.1) * (C.2 - B.2) = (C.1 - B.1) * (B.2 - A.2)
  AB_length : dist A B = 2
  BC_length : dist B C = 1

/-- The sum of vectors AB, AD, and AC in a rectangle -/
def vector_sum (r : Rectangle) : ℝ × ℝ :=
  (r.B.1 - r.A.1 + r.D.1 - r.A.1 + r.C.1 - r.A.1,
   r.B.2 - r.A.2 + r.D.2 - r.A.2 + r.C.2 - r.A.2)

/-- The magnitude of the sum of vectors AB, AD, and AC in a rectangle with side lengths 2 and 1 is 2√5 -/
theorem vector_sum_magnitude (r : Rectangle) : 
  Real.sqrt ((vector_sum r).1 ^ 2 + (vector_sum r).2 ^ 2) = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_vector_sum_magnitude_l2039_203905


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l2039_203967

theorem aquarium_fish_count (stingrays sharks eels : ℕ) : 
  stingrays = 28 →
  sharks = 2 * stingrays →
  eels = 3 * stingrays →
  stingrays + sharks + eels = 168 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l2039_203967


namespace NUMINAMATH_CALUDE_seminar_ratio_l2039_203985

theorem seminar_ratio (total_attendees : ℕ) (avg_age_all : ℚ) (avg_age_doctors : ℚ) (avg_age_lawyers : ℚ)
  (h_total : total_attendees = 20)
  (h_avg_all : avg_age_all = 45)
  (h_avg_doctors : avg_age_doctors = 40)
  (h_avg_lawyers : avg_age_lawyers = 55) :
  ∃ (num_doctors num_lawyers : ℚ),
    num_doctors + num_lawyers = total_attendees ∧
    (num_doctors * avg_age_doctors + num_lawyers * avg_age_lawyers) / total_attendees = avg_age_all ∧
    num_doctors / num_lawyers = 2 := by
  sorry


end NUMINAMATH_CALUDE_seminar_ratio_l2039_203985


namespace NUMINAMATH_CALUDE_pasture_feeding_theorem_l2039_203912

/-- Represents a pasture with growing grass -/
structure Pasture where
  dailyGrowthRate : ℕ
  initialGrass : ℕ

/-- Calculates the number of days a pasture can feed a given number of cows -/
def feedingDays (p : Pasture) (cows : ℕ) : ℕ :=
  (p.initialGrass + p.dailyGrowthRate * cows) / cows

theorem pasture_feeding_theorem (p : Pasture) : 
  feedingDays p 10 = 20 → 
  feedingDays p 15 = 10 → 
  p.dailyGrowthRate = 5 ∧ 
  feedingDays p 30 = 4 := by
  sorry

#check pasture_feeding_theorem

end NUMINAMATH_CALUDE_pasture_feeding_theorem_l2039_203912


namespace NUMINAMATH_CALUDE_wedge_volume_specific_case_l2039_203937

/-- Represents a cylindrical log with a wedge cut out. -/
structure WedgedLog where
  diameter : ℝ
  firstCutAngle : ℝ
  secondCutAngle : ℝ
  intersectionPoint : ℕ

/-- Calculates the volume of the wedge cut from the log. -/
def wedgeVolume (log : WedgedLog) : ℝ :=
  sorry

/-- Theorem stating the volume of the wedge under specific conditions. -/
theorem wedge_volume_specific_case :
  let log : WedgedLog := {
    diameter := 16,
    firstCutAngle := 90,  -- perpendicular cut
    secondCutAngle := 60,
    intersectionPoint := 1
  }
  wedgeVolume log = 512 * Real.pi := by sorry

end NUMINAMATH_CALUDE_wedge_volume_specific_case_l2039_203937


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_50n_two_satisfies_condition_two_is_smallest_l2039_203927

theorem smallest_n_for_sqrt_50n (n : ℕ) : (∃ k : ℕ, k * k = 50 * n) → n ≥ 2 := by
  sorry

theorem two_satisfies_condition : ∃ k : ℕ, k * k = 50 * 2 := by
  sorry

theorem two_is_smallest : ∀ n : ℕ, n > 0 → n < 2 → ¬(∃ k : ℕ, k * k = 50 * n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_50n_two_satisfies_condition_two_is_smallest_l2039_203927


namespace NUMINAMATH_CALUDE_pure_gala_trees_l2039_203906

/-- Represents the apple orchard problem --/
def apple_orchard_problem (T F G : ℕ) : Prop :=
  (F : ℚ) + 0.1 * T = 238 ∧
  F = (3/4 : ℚ) * T ∧
  G = T - F

/-- Theorem stating the number of pure Gala trees --/
theorem pure_gala_trees : ∃ T F G : ℕ, 
  apple_orchard_problem T F G ∧ G = 70 := by
  sorry

end NUMINAMATH_CALUDE_pure_gala_trees_l2039_203906


namespace NUMINAMATH_CALUDE_max_area_at_two_l2039_203945

open Real

noncomputable def tangentArea (m : ℝ) : ℝ :=
  if 1 ≤ m ∧ m ≤ 2 then
    4 * (4 - m) * (exp m)
  else if 2 < m ∧ m ≤ 5 then
    8 * (exp m)
  else
    0

theorem max_area_at_two :
  ∀ m : ℝ, 1 ≤ m ∧ m ≤ 5 → tangentArea m ≤ tangentArea 2 := by
  sorry

#check max_area_at_two

end NUMINAMATH_CALUDE_max_area_at_two_l2039_203945


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2039_203988

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℤ)
  (h_arithmetic : isArithmeticSequence a)
  (h_a4 : a 4 = -4)
  (h_a8 : a 8 = 4) :
  a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2039_203988


namespace NUMINAMATH_CALUDE_max_gel_pens_l2039_203957

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- Checks if the given pen counts satisfy all conditions -/
def is_valid_count (counts : PenCounts) : Prop :=
  counts.ballpoint + counts.gel + counts.fountain = 20 ∧
  counts.ballpoint > 0 ∧ counts.gel > 0 ∧ counts.fountain > 0 ∧
  10 * counts.ballpoint + 50 * counts.gel + 80 * counts.fountain = 1000

/-- Theorem stating that the maximum number of gel pens is 13 -/
theorem max_gel_pens : 
  (∃ (counts : PenCounts), is_valid_count counts ∧ counts.gel = 13) ∧
  (∀ (counts : PenCounts), is_valid_count counts → counts.gel ≤ 13) :=
sorry

end NUMINAMATH_CALUDE_max_gel_pens_l2039_203957


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2039_203918

/-- Calculates the amount after n years given an initial amount and annual increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating that an initial amount of 1600 increasing by 1/8 annually becomes 2025 after 2 years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 1600
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 2025 := by
  sorry


end NUMINAMATH_CALUDE_amount_after_two_years_l2039_203918


namespace NUMINAMATH_CALUDE_like_terms_imply_m_and_n_l2039_203933

/-- Two algebraic expressions are like terms if their variables have the same base and exponents -/
def are_like_terms (expr1 expr2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ x y, ∃ c1 c2 : ℝ, expr1 x y = c1 * (x^(expr1 1 0) * y^(expr1 0 1)) ∧
                      expr2 x y = c2 * (x^(expr2 1 0) * y^(expr2 0 1)) ∧
                      expr1 1 0 = expr2 1 0 ∧
                      expr1 0 1 = expr2 0 1

theorem like_terms_imply_m_and_n (m n : ℕ) :
  are_like_terms (λ x y => -3 * x^(m-1) * y^3) (λ x y => 4 * x * y^(m+n)) →
  m = 2 ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_imply_m_and_n_l2039_203933


namespace NUMINAMATH_CALUDE_geometry_problem_l2039_203993

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (intersection : Plane → Plane → Line)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem geometry_problem 
  (l m : Line) (α β γ : Plane)
  (h1 : intersection β γ = l)
  (h2 : parallel l α)
  (h3 : subset m α)
  (h4 : perpendicular m γ) :
  perpendicularPlanes α γ ∧ perpendicularLines l m := by
  sorry

end NUMINAMATH_CALUDE_geometry_problem_l2039_203993


namespace NUMINAMATH_CALUDE_inequality_theorem_l2039_203975

theorem inequality_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2039_203975


namespace NUMINAMATH_CALUDE_negation_of_positive_quadratic_l2039_203986

theorem negation_of_positive_quadratic (x : ℝ) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_quadratic_l2039_203986


namespace NUMINAMATH_CALUDE_roots_are_cosines_of_triangle_angles_l2039_203994

-- Define the polynomial p(x)
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the condition
def condition (a b c : ℝ) : Prop := a^2 - 2*b - 2*c = 1

-- Theorem statement
theorem roots_are_cosines_of_triangle_angles 
  (a b c : ℝ) 
  (h_positive_roots : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    ∀ t : ℝ, p a b c t = 0 ↔ t = x ∨ t = y ∨ t = z) :
  condition a b c ↔ 
  ∃ A B C : ℝ, 
    0 < A ∧ A < π/2 ∧
    0 < B ∧ B < π/2 ∧
    0 < C ∧ C < π/2 ∧
    A + B + C = π ∧
    (∀ t : ℝ, p a b c t = 0 ↔ t = Real.cos A ∨ t = Real.cos B ∨ t = Real.cos C) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_cosines_of_triangle_angles_l2039_203994


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2039_203999

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 4) * (8 / x^2 + 12 * x - 5) = 6 / x^2 + 9 * x - 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2039_203999


namespace NUMINAMATH_CALUDE_gray_eyed_brunettes_l2039_203978

theorem gray_eyed_brunettes (total : ℕ) (green_eyed_blondes : ℕ) (brunettes : ℕ) (gray_eyed : ℕ)
  (h1 : total = 60)
  (h2 : green_eyed_blondes = 20)
  (h3 : brunettes = 35)
  (h4 : gray_eyed = 25) :
  total - brunettes - green_eyed_blondes = gray_eyed - (total - brunettes) + green_eyed_blondes :=
by
  sorry

#check gray_eyed_brunettes

end NUMINAMATH_CALUDE_gray_eyed_brunettes_l2039_203978


namespace NUMINAMATH_CALUDE_math_majors_consecutive_seats_probability_l2039_203929

/-- The number of people sitting at the round table. -/
def totalPeople : ℕ := 12

/-- The number of math majors. -/
def mathMajors : ℕ := 5

/-- The number of physics majors. -/
def physicsMajors : ℕ := 4

/-- The number of biology majors. -/
def biologyMajors : ℕ := 3

/-- The probability of all math majors sitting in consecutive seats. -/
def probabilityConsecutiveSeats : ℚ := 1 / 66

theorem math_majors_consecutive_seats_probability :
  probabilityConsecutiveSeats = (totalPeople : ℚ) / (totalPeople.choose mathMajors) := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_seats_probability_l2039_203929


namespace NUMINAMATH_CALUDE_best_play_win_probability_best_play_win_probability_multi_l2039_203941

/-- The probability that the best play wins with a majority of votes in a two-play competition. -/
theorem best_play_win_probability (n : ℕ) : ℝ :=
  let total_mothers : ℕ := 2 * n
  let confident_mothers : ℕ := n
  let non_confident_mothers : ℕ := n
  let vote_for_best_prob : ℝ := 1 / 2
  let vote_for_child_prob : ℝ := 1 / 2
  1 - (1 / 2) ^ n

/-- The probability that the best play wins with a majority of votes in a multi-play competition. -/
theorem best_play_win_probability_multi (n s : ℕ) : ℝ :=
  let total_mothers : ℕ := s * n
  let confident_mothers : ℕ := n
  let non_confident_mothers : ℕ := (s - 1) * n
  let vote_for_best_prob : ℝ := 1 / 2
  let vote_for_child_prob : ℝ := 1 / 2
  1 - (1 / 2) ^ ((s - 1) * n)

#check best_play_win_probability
#check best_play_win_probability_multi

end NUMINAMATH_CALUDE_best_play_win_probability_best_play_win_probability_multi_l2039_203941


namespace NUMINAMATH_CALUDE_sum_of_pairwise_ratios_geq_three_halves_l2039_203968

theorem sum_of_pairwise_ratios_geq_three_halves 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_ratios_geq_three_halves_l2039_203968


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2039_203951

theorem arithmetic_calculations :
  (8 / (8 / 17) = 17) ∧
  ((6 / 11) / 3 = 2 / 11) ∧
  ((5 / 4) * (1 / 5) = 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2039_203951


namespace NUMINAMATH_CALUDE_function_properties_l2039_203959

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem function_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x ∈ Set.Icc (f a 1) (f a (exp 1))) ∧
  (a = -4 → ∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x ≤ f a (exp 1)) ∧
  (a = -4 → f a (exp 1) = (exp 1)^2 - 4) ∧
  (∃ n : ℕ, n ≤ 2 ∧ (∃ S : Finset ℝ, S.card = n ∧ ∀ x ∈ S, x ∈ Set.Icc 1 (exp 1) ∧ f a x = 0)) ∧
  (a > 0 → ¬∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 (exp 1) → x₂ ∈ Set.Icc 1 (exp 1) →
    |f a x₁ - f a x₂| ≤ |1/x₁ - 1/x₂|) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2039_203959


namespace NUMINAMATH_CALUDE_isosceles_60_is_equilateral_l2039_203980

-- Define an isosceles triangle with one 60° angle
def IsoscelesTriangleWith60Degree (α β γ : ℝ) : Prop :=
  (α = β ∨ β = γ ∨ γ = α) ∧ (α = 60 ∨ β = 60 ∨ γ = 60)

-- Theorem statement
theorem isosceles_60_is_equilateral (α β γ : ℝ) :
  IsoscelesTriangleWith60Degree α β γ →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_60_is_equilateral_l2039_203980


namespace NUMINAMATH_CALUDE_right_triangle_area_l2039_203995

theorem right_triangle_area (a b c m : ℝ) : 
  a = 10 →                -- One leg is 10
  m = 13 →                -- Shortest median is 13
  m^2 = (2*a^2 + 2*c^2 - b^2) / 4 →  -- Apollonius's theorem
  a^2 + b^2 = c^2 →       -- Pythagorean theorem
  a * b / 2 = 10 * Real.sqrt 69 :=   -- Area of the triangle
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2039_203995


namespace NUMINAMATH_CALUDE_no_natural_solutions_l2039_203919

theorem no_natural_solutions : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l2039_203919


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l2039_203955

/-- Number of ways to make n substitutions in a soccer game -/
def substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (12 - k) * substitutions k

/-- Total number of ways to make 0 to 4 substitutions -/
def totalSubstitutions : ℕ :=
  (List.range 5).map substitutions |>.sum

/-- The remainder when the total number of substitutions is divided by 1000 -/
theorem soccer_substitutions_remainder :
  totalSubstitutions % 1000 = 522 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l2039_203955


namespace NUMINAMATH_CALUDE_longer_string_length_l2039_203932

theorem longer_string_length 
  (total_length : ℕ) 
  (difference : ℕ) 
  (h1 : total_length = 348) 
  (h2 : difference = 72) : 
  ∃ (longer shorter : ℕ), 
    longer + shorter = total_length ∧ 
    longer - shorter = difference ∧ 
    longer = 210 := by
  sorry

end NUMINAMATH_CALUDE_longer_string_length_l2039_203932


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_nine_l2039_203950

theorem product_of_fractions_equals_nine (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0)
  (a_neq_b : a ≠ b)
  (a_neq_c : a ≠ c)
  (b_neq_c : b ≠ c) :
  ((a - b) / c + (b - c) / a + (c - a) / b) * (c / (a - b) + a / (b - c) + b / (c - a)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_nine_l2039_203950


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2039_203990

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 5

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 200

/-- Theorem stating the cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 320 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2039_203990


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2039_203991

theorem geometric_series_common_ratio :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -8/3
  let a₃ : ℚ := 64/21
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  r = -14/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2039_203991


namespace NUMINAMATH_CALUDE_burger_composition_l2039_203917

theorem burger_composition (total_weight filler_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_burger_composition_l2039_203917


namespace NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l2039_203962

/-- Represents the number of flies eaten daily in a swamp ecosystem -/
def flies_eaten_daily (gharials : ℕ) (fish_per_gharial : ℕ) (frogs_per_fish : ℕ) (flies_per_frog : ℕ) : ℕ :=
  gharials * fish_per_gharial * frogs_per_fish * flies_per_frog

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_daily 9 15 8 30 = 32400 := by
  sorry

end NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l2039_203962


namespace NUMINAMATH_CALUDE_petes_number_l2039_203902

theorem petes_number : ∃ x : ℝ, 3 * (2 * x + 15) = 141 ∧ x = 16 := by sorry

end NUMINAMATH_CALUDE_petes_number_l2039_203902


namespace NUMINAMATH_CALUDE_price_after_two_reductions_l2039_203911

-- Define the price reductions
def first_reduction : ℝ := 0.1  -- 10%
def second_reduction : ℝ := 0.14  -- 14%

-- Define the theorem
theorem price_after_two_reductions :
  let original_price : ℝ := 100
  let price_after_first_reduction := original_price * (1 - first_reduction)
  let final_price := price_after_first_reduction * (1 - second_reduction)
  final_price / original_price = 0.774 :=
by sorry

end NUMINAMATH_CALUDE_price_after_two_reductions_l2039_203911


namespace NUMINAMATH_CALUDE_max_sphere_radius_l2039_203984

-- Define the glass shape function
def glass_shape (x : ℝ) : ℝ := x^4

-- Define the circle equation
def circle_equation (x y r : ℝ) : Prop := x^2 + (y - r)^2 = r^2

-- Define the condition that the circle contains the origin
def contains_origin (r : ℝ) : Prop := circle_equation 0 0 r

-- Define the condition that the circle lies above or on the glass shape
def above_glass_shape (x y r : ℝ) : Prop := 
  circle_equation x y r → y ≥ glass_shape x

-- State the theorem
theorem max_sphere_radius : 
  ∃ (r : ℝ), r = (3 * 2^(1/3)) / 4 ∧ 
  (∀ (x y : ℝ), above_glass_shape x y r) ∧
  contains_origin r ∧
  (∀ (r' : ℝ), r' > r → ¬(∀ (x y : ℝ), above_glass_shape x y r') ∨ ¬(contains_origin r')) :=
sorry

end NUMINAMATH_CALUDE_max_sphere_radius_l2039_203984


namespace NUMINAMATH_CALUDE_greg_situps_l2039_203977

-- Define the number of sit-ups Peter did
def peter_situps : ℕ := 24

-- Define the ratio of Peter's sit-ups to Greg's sit-ups
def ratio_peter : ℕ := 3
def ratio_greg : ℕ := 4

-- Theorem to prove
theorem greg_situps : 
  (peter_situps * ratio_greg) / ratio_peter = 32 := by
  sorry

end NUMINAMATH_CALUDE_greg_situps_l2039_203977


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2039_203916

theorem fraction_equals_zero (x : ℝ) :
  (1 - x^2) / (x - 1) = 0 ∧ x ≠ 1 → x = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2039_203916


namespace NUMINAMATH_CALUDE_binomial_gcd_divisibility_l2039_203910

theorem binomial_gcd_divisibility (n k : ℕ+) :
  ∃ m : ℕ, m * n = Nat.choose n.val k.val * Nat.gcd n.val k.val := by
  sorry

end NUMINAMATH_CALUDE_binomial_gcd_divisibility_l2039_203910


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l2039_203944

/-- A composite figure made of squares and triangles -/
structure CompositeFigure where
  squareSideLength : ℝ
  triangleSideLength : ℝ
  numSquares : ℕ
  numTriangles : ℕ

/-- Calculate the perimeter of the composite figure -/
def perimeter (figure : CompositeFigure) : ℝ :=
  let squareContribution := 2 * figure.squareSideLength * (figure.numSquares + 2)
  let triangleContribution := figure.triangleSideLength * figure.numTriangles
  squareContribution + triangleContribution

/-- Theorem: The perimeter of the specific composite figure is 17 -/
theorem specific_figure_perimeter :
  let figure : CompositeFigure :=
    { squareSideLength := 2
      triangleSideLength := 1
      numSquares := 4
      numTriangles := 3 }
  perimeter figure = 17 := by
  sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l2039_203944


namespace NUMINAMATH_CALUDE_triangle_on_bottom_l2039_203935

/-- Represents the positions of faces on a cube -/
inductive CubeFace
  | Top
  | Bottom
  | East
  | South
  | West
  | North

/-- Represents the flattened cube configuration -/
structure FlattenedCube where
  faces : List CubeFace
  triangle_position : CubeFace

/-- The specific flattened cube configuration from the problem -/
def problem_cube : FlattenedCube := sorry

/-- Theorem stating that the triangle is on the bottom face in the given configuration -/
theorem triangle_on_bottom (c : FlattenedCube) : c.triangle_position = CubeFace.Bottom := by
  sorry

end NUMINAMATH_CALUDE_triangle_on_bottom_l2039_203935


namespace NUMINAMATH_CALUDE_set_union_problem_l2039_203908

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l2039_203908
