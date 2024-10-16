import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_area_l1891_189164

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Theorem: Area of quadrilateral ABCD is 62.5√3 -/
theorem quadrilateral_area (ABCD : Quadrilateral) :
  angle ABCD.A ABCD.B ABCD.C = π / 2 →
  angle ABCD.A ABCD.C ABCD.D = π / 3 →
  distance ABCD.A ABCD.C = 25 →
  distance ABCD.C ABCD.D = 10 →
  ∃ E : Point, distance ABCD.A E = 15 →
  area ABCD = 62.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1891_189164


namespace NUMINAMATH_CALUDE_range_of_m_l1891_189103

def P (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0

theorem range_of_m : ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1891_189103


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l1891_189141

theorem subtraction_of_large_numbers : 
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l1891_189141


namespace NUMINAMATH_CALUDE_trip_charges_eq_14_l1891_189120

/-- Represents the daily mileage and charging capacity for a 7-day trip. -/
structure TripData where
  daily_mileage : Fin 7 → ℕ
  initial_charging_capacity : ℕ
  daily_capacity_increment : ℕ
  weather_reduction_days : Finset (Fin 7)
  weather_reduction_percent : ℚ
  stop_interval : ℕ
  stop_days : Finset (Fin 7)

/-- Calculates the number of charges needed for a given day. -/
def charges_needed (data : TripData) (day : Fin 7) : ℕ :=
  sorry

/-- Calculates the total number of charges needed for the entire trip. -/
def total_charges (data : TripData) : ℕ :=
  sorry

/-- The main theorem stating that the total number of charges for the given trip data is 14. -/
theorem trip_charges_eq_14 : ∃ (data : TripData),
  data.daily_mileage = ![135, 259, 159, 189, 210, 156, 240] ∧
  data.initial_charging_capacity = 106 ∧
  data.daily_capacity_increment = 15 ∧
  data.weather_reduction_days = {3, 6} ∧
  data.weather_reduction_percent = 5 / 100 ∧
  data.stop_interval = 55 ∧
  data.stop_days = {1, 5} ∧
  total_charges data = 14 :=
  sorry

end NUMINAMATH_CALUDE_trip_charges_eq_14_l1891_189120


namespace NUMINAMATH_CALUDE_scores_statistics_l1891_189197

def scores : List ℕ := [98, 88, 90, 92, 90, 94]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def average (l : List ℕ) : ℚ := sorry

theorem scores_statistics :
  mode scores = 90 ∧
  median scores = 91 ∧
  average scores = 92 := by sorry

end NUMINAMATH_CALUDE_scores_statistics_l1891_189197


namespace NUMINAMATH_CALUDE_apple_count_difference_l1891_189134

theorem apple_count_difference (initial_green : ℕ) (red_green_difference : ℕ) (delivered_green : ℕ) : 
  initial_green = 546 →
  red_green_difference = 1850 →
  delivered_green = 2725 →
  (initial_green + delivered_green) - (initial_green + red_green_difference) = 875 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_count_difference_l1891_189134


namespace NUMINAMATH_CALUDE_factors_of_36_l1891_189186

def number : ℕ := 36

-- Sum of positive factors
def sum_of_factors (n : ℕ) : ℕ := sorry

-- Product of prime factors
def product_of_prime_factors (n : ℕ) : ℕ := sorry

theorem factors_of_36 :
  sum_of_factors number = 91 ∧ product_of_prime_factors number = 6 := by sorry

end NUMINAMATH_CALUDE_factors_of_36_l1891_189186


namespace NUMINAMATH_CALUDE_min_contribution_l1891_189191

/-- Proves that given 10 people contributing a total of $20.00, with a maximum individual contribution of $11, the minimum amount each person must have contributed is $2.00. -/
theorem min_contribution (num_people : ℕ) (total_contribution : ℚ) (max_individual : ℚ) :
  num_people = 10 ∧ 
  total_contribution = 20 ∧ 
  max_individual = 11 →
  ∃ (min_contribution : ℚ),
    min_contribution = 2 ∧
    num_people * min_contribution = total_contribution ∧
    ∀ (individual : ℚ),
      individual ≥ min_contribution ∧
      individual ≤ max_individual ∧
      (num_people - 1) * min_contribution + individual = total_contribution :=
by sorry

end NUMINAMATH_CALUDE_min_contribution_l1891_189191


namespace NUMINAMATH_CALUDE_number_subtraction_division_l1891_189140

theorem number_subtraction_division : ∃! x : ℝ, (x - 5) / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_division_l1891_189140


namespace NUMINAMATH_CALUDE_puppies_remaining_l1891_189132

def initial_puppies : ℕ := 12
def puppies_given_away : ℕ := 7

theorem puppies_remaining (initial : ℕ) (given_away : ℕ) :
  initial = initial_puppies →
  given_away = puppies_given_away →
  initial - given_away = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_puppies_remaining_l1891_189132


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l1891_189160

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l1891_189160


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1891_189198

def f (x : ℤ) : ℤ := 3 * x + 2

def iterate_f (n : ℕ) (x : ℤ) : ℤ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, ∃ k : ℤ, iterate_f 100 m.val = 1988 * k := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1891_189198


namespace NUMINAMATH_CALUDE_expression_evaluation_l1891_189192

theorem expression_evaluation (x y z : ℚ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  (x + 3) / (x + 2) * (y - 2) / (y - 3) * (z + 9) / (z + 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1891_189192


namespace NUMINAMATH_CALUDE_exactly_two_linear_functions_l1891_189180

/-- Two quadratic trinomials -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Linear function -/
structure LinearFunction where
  m : ℝ
  n : ℝ

/-- Evaluate a quadratic trinomial at a given x -/
def evaluate_quadratic (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Evaluate a linear function at a given x -/
def evaluate_linear (l : LinearFunction) (x : ℝ) : ℝ :=
  l.m * x + l.n

/-- The main theorem -/
theorem exactly_two_linear_functions (P Q : QuadraticTrinomial) :
  ∃! (l₁ l₂ : LinearFunction),
    (∀ x : ℝ, evaluate_quadratic P x = evaluate_quadratic Q (evaluate_linear l₁ x)) ∧
    (∀ x : ℝ, evaluate_quadratic P x = evaluate_quadratic Q (evaluate_linear l₂ x)) ∧
    l₁ ≠ l₂ :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_linear_functions_l1891_189180


namespace NUMINAMATH_CALUDE_triangle_inequality_l1891_189161

theorem triangle_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1891_189161


namespace NUMINAMATH_CALUDE_quadratic_root_implies_d_value_l1891_189144

theorem quadratic_root_implies_d_value (d : ℝ) :
  (∀ x : ℝ, 2 * x^2 + 8 * x + d = 0 ↔ x = (-8 + Real.sqrt 16) / 4 ∨ x = (-8 - Real.sqrt 16) / 4) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_d_value_l1891_189144


namespace NUMINAMATH_CALUDE_negation_p_necessary_not_sufficient_l1891_189124

theorem negation_p_necessary_not_sufficient (p q : Prop) :
  (¬(¬p → ¬(p ∨ q))) ∧ (∃ (p q : Prop), ¬p ∧ (p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_negation_p_necessary_not_sufficient_l1891_189124


namespace NUMINAMATH_CALUDE_james_run_time_l1891_189178

/-- The time it takes James to run 100 meters given John's performance and their speed differences -/
theorem james_run_time (john_total_time john_initial_distance john_initial_time total_distance
  james_initial_distance james_initial_time speed_difference : ℝ)
  (h1 : john_total_time = 13)
  (h2 : john_initial_distance = 4)
  (h3 : john_initial_time = 1)
  (h4 : total_distance = 100)
  (h5 : james_initial_distance = 10)
  (h6 : james_initial_time = 2)
  (h7 : speed_difference = 2)
  : ∃ james_total_time : ℝ, james_total_time = 11 :=
by sorry

end NUMINAMATH_CALUDE_james_run_time_l1891_189178


namespace NUMINAMATH_CALUDE_min_value_of_f_l1891_189147

theorem min_value_of_f (x : Real) (h : x ∈ Set.Icc (π/4) (5*π/12)) : 
  let f := fun (x : Real) => (Real.sin x)^2 - 2*(Real.cos x)^2 / (Real.sin x * Real.cos x)
  ∃ (m : Real), m = -1 ∧ ∀ y ∈ Set.Icc (π/4) (5*π/12), f y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1891_189147


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l1891_189133

theorem smallest_number_with_remainder_two (n : ℕ) : 
  (n % 3 = 2 ∧ n % 4 = 2 ∧ n % 6 = 2 ∧ n % 8 = 2) → n ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l1891_189133


namespace NUMINAMATH_CALUDE_accuracy_of_150_38_million_l1891_189149

/-- Represents a number in millions with two decimal places -/
structure MillionNumber where
  value : ℝ
  isMillions : value ≥ 0
  twoDecimalPlaces : ∃ n : ℕ, value = (n : ℝ) / 100

/-- Represents the accuracy of a number in terms of place value -/
inductive PlaceValue
  | Hundred
  | Thousand
  | TenThousand
  | HundredThousand
  | Million

/-- Given a MillionNumber, returns its accuracy in terms of PlaceValue -/
def getAccuracy (n : MillionNumber) : PlaceValue :=
  PlaceValue.Hundred

/-- Theorem stating that 150.38 million is accurate to the hundred place -/
theorem accuracy_of_150_38_million :
  let n : MillionNumber := ⟨150.38, by norm_num, ⟨15038, by norm_num⟩⟩
  getAccuracy n = PlaceValue.Hundred := by
  sorry

end NUMINAMATH_CALUDE_accuracy_of_150_38_million_l1891_189149


namespace NUMINAMATH_CALUDE_saree_discount_problem_l1891_189177

/-- Proves that given a saree with an original price of 600, after a 20% discount
    and a second discount resulting in a final price of 456, the second discount percentage is 5% -/
theorem saree_discount_problem (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ)
    (h1 : original_price = 600)
    (h2 : first_discount = 20)
    (h3 : final_price = 456) :
    let price_after_first_discount := original_price * (1 - first_discount / 100)
    let second_discount_amount := price_after_first_discount - final_price
    let second_discount_percentage := (second_discount_amount / price_after_first_discount) * 100
    second_discount_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l1891_189177


namespace NUMINAMATH_CALUDE_min_value_expression_l1891_189100

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + 
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 
    (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + 
    (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1891_189100


namespace NUMINAMATH_CALUDE_ticket_draw_theorem_l1891_189123

theorem ticket_draw_theorem (total : ℕ) (blue green red yellow orange : ℕ) : 
  total = 400 ∧ 
  blue + green + red + yellow + orange = total ∧
  blue * 2 = green ∧ 
  green * 2 = red ∧
  green * 3 = yellow ∧
  yellow * 2 = orange →
  (∃ n : ℕ, n ≤ 196 ∧ 
    (∀ m : ℕ, m < n → 
      (m ≤ blue ∨ m ≤ green ∨ m ≤ red ∨ m ≤ yellow ∨ m ≤ orange) ∧ 
      m < 50)) ∧
  (∃ color : ℕ, color ≥ 50 ∧ 
    (color = blue ∨ color = green ∨ color = red ∨ color = yellow ∨ color = orange) ∧
    color ≤ 196) := by
  sorry

end NUMINAMATH_CALUDE_ticket_draw_theorem_l1891_189123


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1891_189173

theorem right_triangle_perimeter : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- all sides are positive integers
  b = 4 ∧                   -- one leg measures 4
  a^2 + b^2 = c^2 ∧         -- right-angled triangle (Pythagorean theorem)
  a + b + c = 12            -- perimeter is 12
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1891_189173


namespace NUMINAMATH_CALUDE_chocolate_bar_squares_l1891_189148

theorem chocolate_bar_squares (gerald_bars : ℕ) (students : ℕ) (squares_per_student : ℕ) :
  gerald_bars = 7 →
  students = 24 →
  squares_per_student = 7 →
  (gerald_bars + 2 * gerald_bars) * (squares_in_each_bar : ℕ) = students * squares_per_student →
  squares_in_each_bar = 8 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bar_squares_l1891_189148


namespace NUMINAMATH_CALUDE_parabola_properties_l1891_189108

/-- Parabola with symmetric axis at x = -2 passing through (1, -2) and c > 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  symmetric_axis : a * (-2) + b = 0
  passes_through : a * 1^2 + b * 1 + c = -2
  c_positive : c > 0

theorem parabola_properties (p : Parabola) :
  p.a < 0 ∧ 16 * p.a + p.c > 4 * p.b := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1891_189108


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_sum_l1891_189117

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def isGeometricSequence (a b c : ℕ) : Prop :=
  (fib b) ^ 2 = (fib a) * (fib c)

theorem fibonacci_geometric_sequence_sum (a b c : ℕ) :
  isGeometricSequence a b c ∧ 
  fib a ≤ fib b ∧ 
  fib b ≤ fib c ∧ 
  a + b + c = 1500 → 
  a = 499 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_sum_l1891_189117


namespace NUMINAMATH_CALUDE_S_infinite_l1891_189111

/-- The expression 2^(n^3+1) - 3^(n^2+1) + 5^(n+1) for positive integer n -/
def f (n : ℕ+) : ℤ := 2^(n.val^3+1) - 3^(n.val^2+1) + 5^(n.val+1)

/-- The set of prime numbers that divide f(n) for some positive integer n -/
def S : Set ℕ := {p : ℕ | Nat.Prime p ∧ ∃ (n : ℕ+), ∃ (k : ℤ), f n = k * p}

/-- Theorem stating that the set S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l1891_189111


namespace NUMINAMATH_CALUDE_diameter_endpoint_theorem_l1891_189110

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle --/
structure Diameter where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Theorem: Given a circle with center at (0,0) and one endpoint of a diameter at (3,4),
    the other endpoint of the diameter is at (-3, -4) --/
theorem diameter_endpoint_theorem (c : Circle) (d : Diameter) :
  c.center = (0, 0) ∧ d.endpoint1 = (3, 4) →
  d.endpoint2 = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_diameter_endpoint_theorem_l1891_189110


namespace NUMINAMATH_CALUDE_work_completion_time_l1891_189158

theorem work_completion_time (x y : ℕ) (h1 : x = 14) 
  (h2 : (5 : ℝ) * ((1 : ℝ) / x + (1 : ℝ) / y) = 0.6071428571428572) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1891_189158


namespace NUMINAMATH_CALUDE_tape_length_calculation_l1891_189112

/-- Calculate the total length of overlapping tape sheets -/
def totalTapeLength (sheetLength : ℝ) (overlap : ℝ) (numSheets : ℕ) : ℝ :=
  sheetLength + (numSheets - 1 : ℝ) * (sheetLength - overlap)

/-- Theorem: The total length of 64 sheets of tape, each 25 cm long, 
    with a 3 cm overlap between consecutive sheets, is 1411 cm -/
theorem tape_length_calculation :
  totalTapeLength 25 3 64 = 1411 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_calculation_l1891_189112


namespace NUMINAMATH_CALUDE_complex_fraction_ratio_l1891_189113

theorem complex_fraction_ratio : 
  let z : ℂ := (2 + Complex.I) / Complex.I
  let a : ℝ := z.re
  let b : ℝ := z.im
  b / a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_ratio_l1891_189113


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l1891_189131

/-- The set of polynomials with coefficients from {0,1,...,p-1} and degree less than p -/
def K_p (p : ℕ) : Set (Polynomial ℤ) :=
  {f | ∀ i, (f.coeff i < p ∧ f.coeff i ≥ 0) ∧ f.degree < p}

/-- The condition that for all pairs of polynomials P,Q in K_p, 
    if P(Q(n)) ≡ n (mod p) for all integers n, then deg(P) = deg(Q) -/
def condition (p : ℕ) : Prop :=
  ∀ P Q : Polynomial ℤ, P ∈ K_p p → Q ∈ K_p p →
    (∀ n : ℤ, (P.comp Q).eval n ≡ n [ZMOD p]) →
    P.degree = Q.degree

theorem prime_condition_characterization :
  ∀ p : ℕ, p.Prime → (condition p ↔ p ∈ ({2, 3, 5, 7} : Set ℕ)) := by
  sorry

#check prime_condition_characterization

end NUMINAMATH_CALUDE_prime_condition_characterization_l1891_189131


namespace NUMINAMATH_CALUDE_max_power_of_two_divides_l1891_189122

/-- The highest power of 2 dividing a natural number -/
def v2 (n : ℕ) : ℕ := sorry

/-- The maximum power of 2 dividing (2019^n - 1) / 2018 for positive integer n -/
def max_power_of_two (n : ℕ+) : ℕ :=
  if n.val % 2 = 1 then 0 else v2 n.val + 1

/-- Theorem stating the maximum power of 2 dividing the given expression -/
theorem max_power_of_two_divides (n : ℕ+) :
  (2019^n.val - 1) / 2018 % 2^(max_power_of_two n) = 0 ∧
  ∀ k > max_power_of_two n, (2019^n.val - 1) / 2018 % 2^k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_max_power_of_two_divides_l1891_189122


namespace NUMINAMATH_CALUDE_natural_number_equation_solutions_l1891_189105

theorem natural_number_equation_solutions :
  ∀ a b : ℕ,
  a^b + b^a = 10 * b^(a-2) + 100 ↔ (a = 109 ∧ b = 1) ∨ (a = 7 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_equation_solutions_l1891_189105


namespace NUMINAMATH_CALUDE_line_properties_l1891_189115

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def y_coord (l : Line3D) (x : ℝ) : ℝ := sorry

/-- The intersection point of the line with the z=0 plane -/
def z_plane_intersection (l : Line3D) : ℝ × ℝ × ℝ := sorry

theorem line_properties (l : Line3D) 
  (h1 : l.point1 = (1, 3, 2)) 
  (h2 : l.point2 = (4, 3, -1)) : 
  y_coord l 7 = 3 ∧ z_plane_intersection l = (3, 3, 0) := by sorry

end NUMINAMATH_CALUDE_line_properties_l1891_189115


namespace NUMINAMATH_CALUDE_max_b_value_l1891_189171

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + 3 * b * x

-- State the theorem
theorem max_b_value (a b : ℝ) (ha : a < 0) (hb : b > 0) 
  (hf : ∀ x ∈ Set.Icc 0 1, f a b x ∈ Set.Icc 0 1) : 
  b ≤ Real.sqrt 3 / 2 ∧ ∃ x ∈ Set.Icc 0 1, f a (Real.sqrt 3 / 2) x = 1 := by
sorry

end NUMINAMATH_CALUDE_max_b_value_l1891_189171


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l1891_189146

def n : ℕ := 1004

theorem floor_expression_equals_eight :
  ⌊(1005^3 : ℚ) / (1003 * 1004) - (1003^3 : ℚ) / (1004 * 1005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l1891_189146


namespace NUMINAMATH_CALUDE_clock_face_ratio_l1891_189193

theorem clock_face_ratio (s : ℝ) (h : s = 4) : 
  let r : ℝ := s / 2
  let circle_area : ℝ := π * r^2
  let triangle_area : ℝ := r^2
  let sector_area : ℝ := circle_area / 12
  sector_area / triangle_area = π / 2 := by sorry

end NUMINAMATH_CALUDE_clock_face_ratio_l1891_189193


namespace NUMINAMATH_CALUDE_basketball_not_football_l1891_189107

theorem basketball_not_football (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : basketball = 15)
  (h3 : football = 8)
  (h4 : neither = 8) :
  ∃ (x : ℕ), x = basketball - (basketball + football - total + neither) ∧ x = 14 :=
by sorry

end NUMINAMATH_CALUDE_basketball_not_football_l1891_189107


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_five_is_smallest_smallest_base_is_five_l1891_189176

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 4 → (∃ n : ℕ, 4 * b + 5 = n^2) → b ≥ 5 :=
by
  sorry

theorem five_is_smallest :
  ∃ n : ℕ, 4 * 5 + 5 = n^2 :=
by
  sorry

theorem smallest_base_is_five :
  ∀ b : ℕ, b > 4 ∧ (∃ n : ℕ, 4 * b + 5 = n^2) → b ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_five_is_smallest_smallest_base_is_five_l1891_189176


namespace NUMINAMATH_CALUDE_discriminant_positive_increasing_when_m_le_8_l1891_189130

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m - 2) * x - m

-- Theorem 1: The discriminant is always positive
theorem discriminant_positive (m : ℝ) : m^2 + 4 > 0 := by sorry

-- Theorem 2: The function is increasing for x ≥ 3 when m ≤ 8
theorem increasing_when_m_le_8 (m : ℝ) (h : m ≤ 8) :
  ∀ x ≥ 3, ∀ y > x, f m y > f m x := by sorry

end NUMINAMATH_CALUDE_discriminant_positive_increasing_when_m_le_8_l1891_189130


namespace NUMINAMATH_CALUDE_puzzle_solution_l1891_189139

/-- Given positive integers A and B less than 10 satisfying the equation 21A104 × 11 = 2B8016 × 9, 
    prove that A = 1 and B = 5. -/
theorem puzzle_solution (A B : ℕ) 
  (h1 : 0 < A ∧ A < 10) 
  (h2 : 0 < B ∧ B < 10) 
  (h3 : 21 * 100000 + A * 10000 + 104 * 11 = 2 * 100000 + B * 10000 + 8016 * 9) : 
  A = 1 ∧ B = 5 := by
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1891_189139


namespace NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l1891_189181

/-- The length of the real axis of a hyperbola with equation x^2 - y^2/9 = 1 is 2 -/
theorem real_axis_length_of_hyperbola :
  let hyperbola_equation := fun (x y : ℝ) => x^2 - y^2/9 = 1
  ∃ a : ℝ, a > 0 ∧ hyperbola_equation = fun (x y : ℝ) => x^2/a^2 - y^2/(9*a^2) = 1 →
  (real_axis_length : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l1891_189181


namespace NUMINAMATH_CALUDE_range_of_a_l1891_189119

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1891_189119


namespace NUMINAMATH_CALUDE_power_sum_divisibility_and_quotient_units_digit_l1891_189185

theorem power_sum_divisibility_and_quotient_units_digit :
  (∃ k : ℕ, 4^1987 + 6^1987 = 10 * k) ∧
  (∃ m : ℕ, 4^1987 + 6^1987 = 5 * m) ∧
  (∃ n : ℕ, (4^1987 + 6^1987) / 5 = 10 * n + 0) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_divisibility_and_quotient_units_digit_l1891_189185


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1891_189151

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - x + a < 0 ↔ -1 < x ∧ x < 2) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1891_189151


namespace NUMINAMATH_CALUDE_ladder_problem_l1891_189162

-- Define the ladder setup
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the theorem
theorem ladder_problem :
  -- Part 1: Horizontal distance
  ∃ (horizontal_distance : ℝ),
    horizontal_distance^2 + wall_height^2 = ladder_length^2 ∧
    horizontal_distance = 5 ∧
  -- Part 2: Height reached by 8-meter ladder
  ∃ (height_8m : ℝ),
    height_8m = (wall_height * 8) / ladder_length ∧
    height_8m = 96 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1891_189162


namespace NUMINAMATH_CALUDE_parabola_bound_l1891_189145

theorem parabola_bound (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) →
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1) := by
sorry

end NUMINAMATH_CALUDE_parabola_bound_l1891_189145


namespace NUMINAMATH_CALUDE_successive_discounts_result_l1891_189153

/-- Calculates the final price after applying successive discounts -/
def finalPrice (initialPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that applying successive discounts of 20%, 10%, and 5% to a good 
    with an actual price of Rs. 9941.52 results in a final selling price of Rs. 6800.00 -/
theorem successive_discounts_result (ε : ℝ) (h : ε > 0) :
  ∃ (result : ℝ), abs (finalPrice 9941.52 0.20 0.10 0.05 - 6800.00) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_successive_discounts_result_l1891_189153


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l1891_189142

theorem half_abs_diff_squares : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l1891_189142


namespace NUMINAMATH_CALUDE_equation_solutions_l1891_189175

theorem equation_solutions (m : ℕ+) :
  ∀ x y z : ℕ+, (x^2 + y^2)^m.val = (x * y)^z.val →
  ∃ k n : ℕ+, x = 2^k.val ∧ y = 2^k.val ∧ z = (1 + 2*k.val)*n.val ∧ m = 2*k.val*n.val :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1891_189175


namespace NUMINAMATH_CALUDE_stratified_sample_under35_l1891_189136

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  total : ℕ
  under35 : ℕ
  between35and49 : ℕ
  over50 : ℕ

/-- Calculates the number of employees to be drawn from a specific group in stratified sampling -/
def stratifiedSampleSize (groups : EmployeeGroups) (sampleTotal : ℕ) (groupSize : ℕ) : ℕ :=
  (groupSize * sampleTotal) / groups.total

/-- Theorem stating that in the given scenario, 25 employees under 35 should be drawn -/
theorem stratified_sample_under35 (groups : EmployeeGroups) (sampleTotal : ℕ) :
  groups.total = 500 →
  groups.under35 = 125 →
  groups.between35and49 = 280 →
  groups.over50 = 95 →
  sampleTotal = 100 →
  stratifiedSampleSize groups sampleTotal groups.under35 = 25 := by
  sorry

#check stratified_sample_under35

end NUMINAMATH_CALUDE_stratified_sample_under35_l1891_189136


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1891_189104

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1^2 + 2^2 = (2*a^2 - a^2)) → (2*a = b) → 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2 - y^2/4 = 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1891_189104


namespace NUMINAMATH_CALUDE_distance_to_center_of_gravity_l1891_189159

/-- Regular hexagon with side length a -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Square cut out from the hexagon -/
structure CutOutSquare (hex : RegularHexagon) where
  diagonal : ℝ
  diagonal_eq_side : diagonal = hex.side_length

/-- Remaining plate after cutting out the square -/
structure RemainingPlate (hex : RegularHexagon) (square : CutOutSquare hex) where

/-- Center of gravity of the remaining plate -/
noncomputable def center_of_gravity (plate : RemainingPlate hex square) : ℝ × ℝ := sorry

/-- Distance from the hexagon center to the center of gravity -/
noncomputable def distance_to_center (plate : RemainingPlate hex square) : ℝ :=
  let cog := center_of_gravity plate
  Real.sqrt ((cog.1 ^ 2) + (cog.2 ^ 2))

/-- Main theorem: The distance from the hexagon center to the center of gravity of the remaining plate -/
theorem distance_to_center_of_gravity 
  (hex : RegularHexagon) 
  (square : CutOutSquare hex) 
  (plate : RemainingPlate hex square) : 
  distance_to_center plate = (3 * Real.sqrt 3 + 1) / 52 * hex.side_length := by
  sorry

end NUMINAMATH_CALUDE_distance_to_center_of_gravity_l1891_189159


namespace NUMINAMATH_CALUDE_books_left_after_donation_l1891_189129

/-- Calculates the total number of books left after donation --/
def booksLeftAfterDonation (
  mysteryShelvesCount : ℕ)
  (mysteryBooksPerShelf : ℕ)
  (pictureBooksShelvesCount : ℕ)
  (pictureBooksPerShelf : ℕ)
  (autobiographyShelvesCount : ℕ)
  (autobiographyBooksPerShelf : ℝ)
  (cookbookShelvesCount : ℕ)
  (cookbookBooksPerShelf : ℝ)
  (mysteryBooksDonated : ℕ)
  (pictureBooksdonated : ℕ)
  (autobiographiesDonated : ℕ)
  (cookbooksDonated : ℕ) : ℝ :=
  let totalBooksBeforeDonation :=
    (mysteryShelvesCount * mysteryBooksPerShelf : ℝ) +
    (pictureBooksShelvesCount * pictureBooksPerShelf : ℝ) +
    (autobiographyShelvesCount : ℝ) * autobiographyBooksPerShelf +
    (cookbookShelvesCount : ℝ) * cookbookBooksPerShelf
  let totalBooksDonated :=
    (mysteryBooksDonated + pictureBooksdonated + autobiographiesDonated + cookbooksDonated : ℝ)
  totalBooksBeforeDonation - totalBooksDonated

theorem books_left_after_donation :
  booksLeftAfterDonation 3 9 5 12 4 8.5 2 11.5 7 8 3 5 = 121 := by
  sorry

end NUMINAMATH_CALUDE_books_left_after_donation_l1891_189129


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1891_189170

theorem geometric_sequence_problem (a : ℝ) (h : a > 0) :
  let r : ℝ := 1/2
  let n : ℕ := 6
  let sum : ℝ := a * (1 - r^n) / (1 - r)
  sum = 189 → a * r = 48 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1891_189170


namespace NUMINAMATH_CALUDE_max_sum_geometric_sequence_l1891_189106

/-- Given integers a, b, and c forming a strictly increasing geometric sequence with abc = 216,
    the maximum value of a + b + c is 43. -/
theorem max_sum_geometric_sequence (a b c : ℤ) : 
  a < b ∧ b < c ∧                 -- strictly increasing
  (∃ r : ℤ, r > 1 ∧ b = a * r ∧ c = b * r) ∧  -- geometric sequence
  a * b * c = 216 →               -- product condition
  (∀ x y z : ℤ, 
    x < y ∧ y < z ∧
    (∃ r : ℤ, r > 1 ∧ y = x * r ∧ z = y * r) ∧
    x * y * z = 216 →
    x + y + z ≤ a + b + c) ∧
  a + b + c = 43 := by
sorry

end NUMINAMATH_CALUDE_max_sum_geometric_sequence_l1891_189106


namespace NUMINAMATH_CALUDE_train_length_calculation_l1891_189157

/-- The length of a train given crossing time and speeds --/
theorem train_length_calculation (crossing_time : ℝ) (man_speed : ℝ) (train_speed : ℝ) :
  crossing_time = 39.99680025597952 →
  man_speed = 2 →
  train_speed = 56 →
  ∃ (train_length : ℝ), abs (train_length - 599.95) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1891_189157


namespace NUMINAMATH_CALUDE_small_triangles_to_cover_large_l1891_189125

/-- The number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem small_triangles_to_cover_large (large_side small_side : ℝ) : 
  large_side = 12 → small_side = 2 → 
  (large_side^2 / small_side^2 : ℝ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_small_triangles_to_cover_large_l1891_189125


namespace NUMINAMATH_CALUDE_birthday_party_attendees_l1891_189168

theorem birthday_party_attendees :
  ∀ (n : ℕ),
  (12 * (n + 2) = 16 * n) →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_party_attendees_l1891_189168


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1891_189143

theorem arithmetic_sequence_length (a₁ aₙ d : ℕ) (h : aₙ = a₁ + (n - 1) * d) : 
  a₁ = 2 → aₙ = 1007 → d = 5 → n = 202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1891_189143


namespace NUMINAMATH_CALUDE_sin_sum_max_in_acute_triangle_l1891_189137

-- Define the convexity property for a function on an interval
def IsConvex (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y t : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- State the theorem
theorem sin_sum_max_in_acute_triangle :
  IsConvex Real.sin 0 (Real.pi / 2) →
  ∀ A B C : ℝ,
    0 < A ∧ A < Real.pi / 2 →
    0 < B ∧ B < Real.pi / 2 →
    0 < C ∧ C < Real.pi / 2 →
    A + B + C = Real.pi →
    Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_max_in_acute_triangle_l1891_189137


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_l1891_189196

theorem bus_stop_walking_time 
  (usual_speed : ℝ) 
  (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_speed * usual_time = (4/5 * usual_speed) * (usual_time + 7)) :
  usual_time = 28 := by
sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_l1891_189196


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l1891_189138

/-- Binomial distribution -/
def binomial_distribution (n : ℕ) (p : ℝ) : ℝ → ℝ := sorry

/-- Probability of a random variable being greater than or equal to a value -/
def prob_ge (X : ℝ → ℝ) (k : ℝ) : ℝ := sorry

theorem binomial_probability_problem (p : ℝ) :
  let ξ := binomial_distribution 2 p
  let η := binomial_distribution 4 p
  prob_ge ξ 1 = 5/9 →
  prob_ge η 2 = 11/27 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l1891_189138


namespace NUMINAMATH_CALUDE_building_units_l1891_189101

theorem building_units (total : ℕ) (restaurants : ℕ) : 
  (2 * restaurants = total / 4) →
  (restaurants = 75) →
  (total = 300) := by
sorry

end NUMINAMATH_CALUDE_building_units_l1891_189101


namespace NUMINAMATH_CALUDE_problem_solution_l1891_189152

def proposition (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - 2*m*x - 3*m^2 < 0

def set_A : Set ℝ := {m | proposition m}

def set_B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

theorem problem_solution :
  (set_A = Set.Ioi (-2) ∪ Set.Iio (2/3)) ∧
  {a | set_A ⊆ set_B a ∧ set_A ≠ set_B a} = Set.Iic (-3) ∪ Set.Ici (5/3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1891_189152


namespace NUMINAMATH_CALUDE_two_people_completion_time_l1891_189188

/-- Represents the amount of work done on a given day -/
def work_on_day (n : ℕ) : ℕ := 2^(n-1)

/-- Represents the total amount of work done up to and including a given day -/
def total_work (n : ℕ) : ℕ := 2^n - 1

/-- The number of days it takes one person to complete the job -/
def days_for_one_person : ℕ := 12

/-- The theorem stating that two people working together will complete the job in 11 days -/
theorem two_people_completion_time :
  ∃ (n : ℕ), n = 11 ∧ total_work n = total_work days_for_one_person := by
  sorry

end NUMINAMATH_CALUDE_two_people_completion_time_l1891_189188


namespace NUMINAMATH_CALUDE_f_inequality_l1891_189195

noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1))

theorem f_inequality : f (1 / Real.exp 1) < f 0 ∧ f 0 < f (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1891_189195


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l1891_189183

/-- Defines a circle equation passing through three points -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the circle equation passes through the given points -/
theorem circle_passes_through_points :
  CircleEquation 0 0 ∧ CircleEquation 4 0 ∧ CircleEquation (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l1891_189183


namespace NUMINAMATH_CALUDE_multiplier_problem_l1891_189156

theorem multiplier_problem (x m : ℝ) (h1 : x = -10) (h2 : m * x - 8 = -12) : m = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l1891_189156


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l1891_189190

-- Define a decreasing function on ℝ
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem decreasing_function_inequality (f : ℝ → ℝ) (h : DecreasingOn f) : f 3 > f 5 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l1891_189190


namespace NUMINAMATH_CALUDE_phill_slice_left_l1891_189154

-- Define the initial state of the pizza
def initial_pizza : ℕ := 1

-- Define the number of cuts
def number_of_cuts : ℕ := 3

-- Define the function to calculate slices after cuts
def slices_after_cuts (pizza : ℕ) (cuts : ℕ) : ℕ := pizza * (2^cuts)

-- Define the number of friends who get 1 slice
def friends_one_slice : ℕ := 3

-- Define the number of friends who get 2 slices
def friends_two_slices : ℕ := 2

-- Define the function to calculate distributed slices
def distributed_slices (one_slice : ℕ) (two_slices : ℕ) : ℕ := one_slice * 1 + two_slices * 2

-- Theorem: Phill has 1 slice left
theorem phill_slice_left : 
  slices_after_cuts initial_pizza number_of_cuts - 
  distributed_slices friends_one_slice friends_two_slices = 1 := by
  sorry

end NUMINAMATH_CALUDE_phill_slice_left_l1891_189154


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1891_189128

theorem order_of_logarithmic_expressions (x : ℝ) 
  (h1 : x ∈ Set.Ioo (Real.exp (-1)) 1)
  (a b c : ℝ) 
  (ha : a = Real.log x)
  (hb : b = 2 * Real.log x)
  (hc : c = (Real.log x) ^ 3) : 
  b < a ∧ a < c := by
sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1891_189128


namespace NUMINAMATH_CALUDE_polygon_sides_l1891_189184

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 720 → ∃ n : ℕ, n = 6 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1891_189184


namespace NUMINAMATH_CALUDE_ravi_overall_profit_l1891_189118

/-- Calculates the overall profit or loss for Ravi's sales -/
theorem ravi_overall_profit (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 4 / 100 →
  mobile_profit_percent = 10 / 100 →
  (refrigerator_cost * (1 - refrigerator_loss_percent) +
   mobile_cost * (1 + mobile_profit_percent) -
   (refrigerator_cost + mobile_cost) : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_ravi_overall_profit_l1891_189118


namespace NUMINAMATH_CALUDE_fans_with_all_items_l1891_189165

def stadium_capacity : ℕ := 5000
def hotdog_interval : ℕ := 75
def soda_interval : ℕ := 45
def popcorn_interval : ℕ := 50
def max_all_items : ℕ := 100

theorem fans_with_all_items :
  let lcm := Nat.lcm (Nat.lcm hotdog_interval soda_interval) popcorn_interval
  min (stadium_capacity / lcm) max_all_items = 11 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l1891_189165


namespace NUMINAMATH_CALUDE_inequality_proof_ratio_proof_l1891_189116

-- Part I
theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by sorry

-- Part II
theorem ratio_proof (a b c x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : a^2 + b^2 + c^2 = 10) (h8 : x^2 + y^2 + z^2 = 40) (h9 : a*x + b*y + c*z = 20) :
  (a + b + c) / (x + y + z) = 1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_ratio_proof_l1891_189116


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l1891_189167

theorem fraction_sum_difference : 7/6 + 5/4 - 3/2 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l1891_189167


namespace NUMINAMATH_CALUDE_inverse_negation_equivalence_l1891_189189

-- Define a quadrilateral type
structure Quadrilateral where
  isParallelogram : Prop
  oppositeSidesEqual : Prop

-- Define the original proposition
def originalProposition (q : Quadrilateral) : Prop :=
  q.oppositeSidesEqual → q.isParallelogram

-- Define the inverse negation
def inverseNegation (q : Quadrilateral) : Prop :=
  ¬q.isParallelogram → ¬q.oppositeSidesEqual

-- Theorem stating the equivalence of the inverse negation
theorem inverse_negation_equivalence :
  ∀ q : Quadrilateral, inverseNegation q ↔ ¬(originalProposition q) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_equivalence_l1891_189189


namespace NUMINAMATH_CALUDE_berry_theorem_l1891_189166

def berry_problem (total_needed : ℕ) (strawberry_cartons : ℕ) (blueberry_cartons : ℕ) : ℕ :=
  total_needed - (strawberry_cartons + blueberry_cartons)

theorem berry_theorem (total_needed strawberry_cartons blueberry_cartons : ℕ) :
  berry_problem total_needed strawberry_cartons blueberry_cartons =
  total_needed - (strawberry_cartons + blueberry_cartons) :=
by
  sorry

#eval berry_problem 42 2 7

end NUMINAMATH_CALUDE_berry_theorem_l1891_189166


namespace NUMINAMATH_CALUDE_a_salary_is_5250_l1891_189199

/-- Proof that A's salary is $5250 given the conditions of the problem -/
theorem a_salary_is_5250 (a b : ℝ) : 
  a + b = 7000 →                   -- Total salary is $7000
  0.05 * a = 0.15 * b →            -- Savings are equal
  a = 5250 := by
    sorry

end NUMINAMATH_CALUDE_a_salary_is_5250_l1891_189199


namespace NUMINAMATH_CALUDE_bryan_spent_1500_l1891_189121

/-- The total amount spent by Bryan on t-shirts and pants -/
def total_spent (num_tshirts : ℕ) (tshirt_price : ℕ) (num_pants : ℕ) (pants_price : ℕ) : ℕ :=
  num_tshirts * tshirt_price + num_pants * pants_price

/-- Theorem: Bryan spent $1500 on 5 t-shirts at $100 each and 4 pairs of pants at $250 each -/
theorem bryan_spent_1500 : total_spent 5 100 4 250 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_bryan_spent_1500_l1891_189121


namespace NUMINAMATH_CALUDE_x_lt_one_necessary_not_sufficient_l1891_189163

theorem x_lt_one_necessary_not_sufficient :
  ∀ x : ℝ,
  (∀ x, (1 / x > 1 → x < 1)) ∧
  (∃ x, x < 1 ∧ ¬(1 / x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_one_necessary_not_sufficient_l1891_189163


namespace NUMINAMATH_CALUDE_decagon_triangles_l1891_189109

theorem decagon_triangles : ∀ (n : ℕ), n = 10 → (n.choose 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1891_189109


namespace NUMINAMATH_CALUDE_exponential_function_properties_l1891_189179

theorem exponential_function_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, (x = 0 → a^x = 1) ∧
            (x = 1 → a^x = a) ∧
            (x = -1 → a^x = 1/a) ∧
            (x < 0 → a^x > 0 ∧ ∀ ε > 0, ∃ N : ℝ, ∀ y < N, 0 < a^y ∧ a^y < ε)) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_properties_l1891_189179


namespace NUMINAMATH_CALUDE_miss_adamson_class_size_l1891_189126

theorem miss_adamson_class_size :
  let num_classes : ℕ := 4
  let sheets_per_student : ℕ := 5
  let total_sheets : ℕ := 400
  let total_students : ℕ := total_sheets / sheets_per_student
  let students_per_class : ℕ := total_students / num_classes
  students_per_class = 20 := by
  sorry

end NUMINAMATH_CALUDE_miss_adamson_class_size_l1891_189126


namespace NUMINAMATH_CALUDE_perfect_square_form_l1891_189182

theorem perfect_square_form (k : ℕ+) : ∃ (n : ℕ+) (a : ℤ), a^2 = n * 2^(k : ℕ) - 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_form_l1891_189182


namespace NUMINAMATH_CALUDE_adult_panda_consumption_is_138_l1891_189172

/-- The daily bamboo consumption of an adult panda -/
def adult_panda_daily_consumption : ℕ := 138

/-- The daily bamboo consumption of a baby panda -/
def baby_panda_daily_consumption : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total bamboo consumption of both pandas in a week -/
def total_weekly_consumption : ℕ := 1316

/-- Theorem stating that the adult panda's daily bamboo consumption is 138 pounds -/
theorem adult_panda_consumption_is_138 :
  adult_panda_daily_consumption = 
    (total_weekly_consumption - baby_panda_daily_consumption * days_in_week) / days_in_week :=
by sorry

end NUMINAMATH_CALUDE_adult_panda_consumption_is_138_l1891_189172


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l1891_189127

theorem pure_imaginary_value (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 3*a + 2 : ℝ) + (a - 2 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l1891_189127


namespace NUMINAMATH_CALUDE_log_product_interval_l1891_189187

theorem log_product_interval :
  1 < Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 ∧
  Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 < 2 :=
by sorry

end NUMINAMATH_CALUDE_log_product_interval_l1891_189187


namespace NUMINAMATH_CALUDE_jesse_blocks_theorem_l1891_189150

/-- The number of blocks Jesse used to build the building -/
def building_blocks : ℕ := 80

/-- The number of blocks Jesse used to build the farmhouse -/
def farmhouse_blocks : ℕ := 123

/-- The number of blocks Jesse used to build the fenced-in area -/
def fenced_area_blocks : ℕ := 57

/-- The number of blocks Jesse has left -/
def remaining_blocks : ℕ := 84

/-- The total number of blocks Jesse started with -/
def total_blocks : ℕ := building_blocks + farmhouse_blocks + fenced_area_blocks + remaining_blocks

theorem jesse_blocks_theorem : total_blocks = 344 := by
  sorry

end NUMINAMATH_CALUDE_jesse_blocks_theorem_l1891_189150


namespace NUMINAMATH_CALUDE_equal_cost_at_200_unique_equal_cost_l1891_189194

/-- Represents the price per book in yuan -/
def base_price : ℕ := 40

/-- Represents the discount factor for supplier A -/
def discount_a : ℚ := 9/10

/-- Represents the discount factor for supplier B on books over 100 -/
def discount_b : ℚ := 8/10

/-- Represents the threshold for supplier B's discount -/
def threshold : ℕ := 100

/-- Calculates the cost for supplier A given the number of books -/
def cost_a (n : ℕ) : ℚ := n * base_price * discount_a

/-- Calculates the cost for supplier B given the number of books -/
def cost_b (n : ℕ) : ℚ :=
  if n ≤ threshold then n * base_price
  else threshold * base_price + (n - threshold) * base_price * discount_b

/-- Theorem stating that the costs are equal when 200 books are ordered -/
theorem equal_cost_at_200 : cost_a 200 = cost_b 200 := by sorry

/-- Theorem stating that 200 is the unique number of books where costs are equal -/
theorem unique_equal_cost (n : ℕ) : cost_a n = cost_b n ↔ n = 200 := by sorry

end NUMINAMATH_CALUDE_equal_cost_at_200_unique_equal_cost_l1891_189194


namespace NUMINAMATH_CALUDE_fixed_tangent_circle_l1891_189174

-- Define the main circle
def main_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the property of the chords
def chord_property (OA OB : ℝ) : Prop := OA * OB = 2

-- Define the tangent circle
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem fixed_tangent_circle 
  (O A B : ℝ × ℝ) 
  (hA : main_circle A.1 A.2) 
  (hB : main_circle B.1 B.2)
  (hOA : O = (0, 0))
  (hchord : chord_property (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2)) 
                           (Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)))
  (hAB : A ≠ B) :
  ∃ (P : ℝ × ℝ), tangent_circle P.1 P.2 ∧ 
    (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1) :=
sorry

end NUMINAMATH_CALUDE_fixed_tangent_circle_l1891_189174


namespace NUMINAMATH_CALUDE_equal_roots_coefficients_l1891_189114

def polynomial (x p q : ℝ) : ℝ := x^4 - 10*x^3 + 37*x^2 + p*x + q

theorem equal_roots_coefficients :
  ∀ (p q : ℝ),
  (∃ (x₁ x₃ : ℝ), 
    (∀ x : ℝ, polynomial x p q = 0 ↔ x = x₁ ∨ x = x₃) ∧
    (x₁ + x₃ = 5) ∧
    (x₁ * x₃ = 6)) →
  p = -60 ∧ q = 36 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_coefficients_l1891_189114


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1891_189169

theorem fractional_equation_solution :
  ∃ (x : ℝ), (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1891_189169


namespace NUMINAMATH_CALUDE_other_factor_l1891_189155

def f (k : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - 18*x^2 + 52*x + k

theorem other_factor (k : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, f k x = (x - 2) * c) → 
  (∃ d : ℝ, ∀ x : ℝ, f k x = (x + 5) * d) :=
sorry

end NUMINAMATH_CALUDE_other_factor_l1891_189155


namespace NUMINAMATH_CALUDE_smallest_four_digit_pascal_l1891_189135

/-- Pascal's triangle is represented as a function from row and column to natural number -/
def pascal : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal n k + pascal n (k + 1)

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_pascal :
  ∃ (r c : ℕ), isFourDigit (pascal r c) ∧
    ∀ (r' c' : ℕ), isFourDigit (pascal r' c') → pascal r c ≤ pascal r' c' :=
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_pascal_l1891_189135


namespace NUMINAMATH_CALUDE_lion_king_star_wars_profit_ratio_l1891_189102

/-- The ratio of profits between two movies -/
def profit_ratio (cost1 revenue1 cost2 revenue2 : ℚ) : ℚ :=
  (revenue1 - cost1) / (revenue2 - cost2)

/-- Theorem: The ratio of The Lion King's profit to Star Wars' profit is 1:2 -/
theorem lion_king_star_wars_profit_ratio :
  profit_ratio 10 200 25 405 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_lion_king_star_wars_profit_ratio_l1891_189102
