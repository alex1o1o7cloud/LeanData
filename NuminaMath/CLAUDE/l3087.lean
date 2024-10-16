import Mathlib

namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l3087_308717

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (75 * m) % 345 = 225 → m ≥ n) ∧
    n = 118 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l3087_308717


namespace NUMINAMATH_CALUDE_room_width_calculation_l3087_308764

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  area : ℝ
  length : ℝ
  width : ℝ

/-- Theorem: Given a room with area 12.0 sq ft and length 1.5 ft, its width is 8.0 ft -/
theorem room_width_calculation (room : RoomDimensions) 
  (h_area : room.area = 12.0) 
  (h_length : room.length = 1.5) : 
  room.width = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3087_308764


namespace NUMINAMATH_CALUDE_square_difference_l3087_308752

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3087_308752


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l3087_308739

theorem coefficient_x_squared (x : ℝ) : 
  let expansion := (1 + 1/x + 1/x^2) * (1 + x^2)^5
  ∃ a b c d e : ℝ, expansion = a*x^2 + b*x + c + d/x + e/x^2 ∧ a = 15 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l3087_308739


namespace NUMINAMATH_CALUDE_apples_in_crate_l3087_308711

/-- The number of apples in a crate -/
def apples_per_crate : ℕ := sorry

/-- The number of crates delivered -/
def crates_delivered : ℕ := 12

/-- The number of rotten apples -/
def rotten_apples : ℕ := 4

/-- The number of apples that fit in each box -/
def apples_per_box : ℕ := 10

/-- The number of boxes filled with good apples -/
def filled_boxes : ℕ := 50

theorem apples_in_crate :
  apples_per_crate * crates_delivered = filled_boxes * apples_per_box + rotten_apples ∧
  apples_per_crate = 42 := by sorry

end NUMINAMATH_CALUDE_apples_in_crate_l3087_308711


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3087_308769

theorem complex_fraction_simplification :
  (7 : ℂ) + 18 * I / (3 - 4 * I) = -(51 : ℚ) / 25 + (82 : ℚ) / 25 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3087_308769


namespace NUMINAMATH_CALUDE_circle_and_sphere_sum_l3087_308787

theorem circle_and_sphere_sum (c : ℝ) (h : c = 18 * Real.pi) :
  let r := c / (2 * Real.pi)
  (Real.pi * r^2) + (4/3 * Real.pi * r^3) = 1053 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_and_sphere_sum_l3087_308787


namespace NUMINAMATH_CALUDE_halfway_between_one_fifth_and_one_third_l3087_308786

theorem halfway_between_one_fifth_and_one_third :
  (1 / 5 : ℚ) / 2 + (1 / 3 : ℚ) / 2 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_fifth_and_one_third_l3087_308786


namespace NUMINAMATH_CALUDE_imaginary_product_implies_zero_l3087_308747

/-- If the product of (1-ai) and i is a pure imaginary number, then a = 0 -/
theorem imaginary_product_implies_zero (a : ℝ) : 
  (∃ b : ℝ, (1 - a * Complex.I) * Complex.I = b * Complex.I) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_product_implies_zero_l3087_308747


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l3087_308714

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Define the inverse function f^(-1)
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≥ 0 then Real.sqrt y else -Real.sqrt (-y)

-- Theorem statement
theorem inverse_sum_equals_negative_six :
  f_inv 9 + f_inv (-81) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l3087_308714


namespace NUMINAMATH_CALUDE_base_prime_1260_l3087_308780

/-- Base prime representation of a number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Prime factorization of 1260 -/
def PrimeFactorization1260 : List (ℕ × ℕ) :=
  [(2, 2), (3, 2), (5, 1), (7, 1)]

/-- Theorem: The base prime representation of 1260 is [2, 2, 1, 2] -/
theorem base_prime_1260 : 
  BasePrimeRepresentation 1260 = [2, 2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_1260_l3087_308780


namespace NUMINAMATH_CALUDE_min_radius_circle_line_intersection_l3087_308763

theorem min_radius_circle_line_intersection (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  let circle := fun (x y : Real) => (x - Real.cos θ)^2 + (y - Real.sin θ)^2
  let line := fun (x y : Real) => 2 * x - y - 10
  ∃ (r : Real), r > 0 ∧ ∃ (x y : Real), circle x y = r^2 ∧ line x y = 0 →
  ∀ (r' : Real), (∃ (x y : Real), circle x y = r'^2 ∧ line x y = 0) → r' ≥ 2 * Real.sqrt 5 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_radius_circle_line_intersection_l3087_308763


namespace NUMINAMATH_CALUDE_fourth_roll_three_prob_l3087_308749

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_three_prob : ℚ := 1 / 2
def biased_die_other_prob : ℚ := 1 / 10

-- Define the probability of selecting each die
def die_selection_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the event of rolling three threes in a row
def three_threes_event : Prop := True

-- Theorem statement
theorem fourth_roll_three_prob :
  three_threes_event →
  (die_selection_prob * fair_die_prob^3 * fair_die_prob +
   die_selection_prob * biased_die_three_prob^3 * biased_die_three_prob) /
  (die_selection_prob * fair_die_prob^3 +
   die_selection_prob * biased_die_three_prob^3) = 41 / 84 :=
by sorry

end NUMINAMATH_CALUDE_fourth_roll_three_prob_l3087_308749


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l3087_308703

/-- Represents the weather forecast for a day --/
structure WeatherForecast where
  sun_chance : ℝ
  rain_chance1 : ℝ
  rain_amount1 : ℝ
  rain_chance2 : ℝ
  rain_amount2 : ℝ

/-- Calculates the expected rainfall for a given weather forecast --/
def expected_rainfall (forecast : WeatherForecast) : ℝ :=
  forecast.sun_chance * 0 + forecast.rain_chance1 * forecast.rain_amount1 + 
  forecast.rain_chance2 * forecast.rain_amount2

/-- The weather forecast for weekdays --/
def weekday_forecast : WeatherForecast := {
  sun_chance := 0.3,
  rain_chance1 := 0.2,
  rain_amount1 := 5,
  rain_chance2 := 0.5,
  rain_amount2 := 8
}

/-- The weather forecast for weekend days --/
def weekend_forecast : WeatherForecast := {
  sun_chance := 0.5,
  rain_chance1 := 0.25,
  rain_amount1 := 2,
  rain_chance2 := 0.25,
  rain_amount2 := 6
}

/-- The number of weekdays --/
def num_weekdays : ℕ := 5

/-- The number of weekend days --/
def num_weekend_days : ℕ := 2

theorem expected_total_rainfall : 
  (num_weekdays : ℝ) * expected_rainfall weekday_forecast + 
  (num_weekend_days : ℝ) * expected_rainfall weekend_forecast = 29 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l3087_308703


namespace NUMINAMATH_CALUDE_fifth_term_value_l3087_308744

/-- A geometric sequence with positive terms satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  first_second_sum : a 1 + 2 * a 2 = 4
  fourth_squared : a 4 ^ 2 = 4 * a 3 * a 7

/-- The fifth term of the geometric sequence is 1/8 -/
theorem fifth_term_value (seq : GeometricSequence) : seq.a 5 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3087_308744


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l3087_308765

/-- Reflects a point (x, y) about the line y = -x --/
def reflectAboutNegativeX (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  reflectAboutNegativeX original_center = (3, -8) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l3087_308765


namespace NUMINAMATH_CALUDE_commission_calculation_l3087_308736

/-- The original commission held by the company for John --/
def original_commission : ℕ := sorry

/-- The advance agency fees taken by John --/
def advance_fees : ℕ := 8280

/-- The amount given to John by the accountant after one month --/
def amount_given : ℕ := 18500

/-- The incentive amount given to John --/
def incentive_amount : ℕ := 1780

/-- Theorem stating the relationship between the original commission and other amounts --/
theorem commission_calculation : 
  original_commission = amount_given + advance_fees - incentive_amount :=
by sorry

end NUMINAMATH_CALUDE_commission_calculation_l3087_308736


namespace NUMINAMATH_CALUDE_star_property_l3087_308799

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.three
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four

theorem star_property : 
  star (star Element.three Element.two) (star Element.two Element.one) = Element.four := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3087_308799


namespace NUMINAMATH_CALUDE_system_solution_l3087_308789

theorem system_solution :
  ∃! (x y : ℚ), 2 * x - 3 * y = 1 ∧ (y + 1) / 4 + 1 = (x + 2) / 3 ∧ x = 3 ∧ y = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3087_308789


namespace NUMINAMATH_CALUDE_cubic_inequality_l3087_308795

theorem cubic_inequality (x : ℝ) : x > 1 → 2 * x^3 > x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3087_308795


namespace NUMINAMATH_CALUDE_train_length_l3087_308790

/-- Calculates the length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh / 3.6 * time_s = 225 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3087_308790


namespace NUMINAMATH_CALUDE_gcd_30_and_70_to_80_l3087_308741

theorem gcd_30_and_70_to_80 : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 := by sorry

end NUMINAMATH_CALUDE_gcd_30_and_70_to_80_l3087_308741


namespace NUMINAMATH_CALUDE_one_real_solution_l3087_308782

/-- The number of distinct real solutions to the equation (x-5)(x^2 + 5x + 10) = 0 -/
def num_solutions : ℕ := 1

/-- The equation (x-5)(x^2 + 5x + 10) = 0 has exactly one real solution -/
theorem one_real_solution : num_solutions = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_real_solution_l3087_308782


namespace NUMINAMATH_CALUDE_or_propositions_true_l3087_308725

-- Define the properties of square and rectangle diagonals
def square_diagonals_perpendicular : Prop := True
def rectangle_diagonals_bisect : Prop := True

-- Theorem statement
theorem or_propositions_true : 
  ((2 = 2) ∨ (2 > 2)) ∧ 
  (square_diagonals_perpendicular ∨ rectangle_diagonals_bisect) := by
  sorry

end NUMINAMATH_CALUDE_or_propositions_true_l3087_308725


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3087_308713

theorem min_value_a_plus_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b^2 = 4) :
  a + b ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ a₀ * b₀^2 = 4 ∧ a₀ + b₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3087_308713


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3087_308735

def inequality_system (x : ℝ) : Prop :=
  (3*x - 2)/3 ≥ 1 ∧ 3*x + 5 > 4*x - 2

def integer_solutions : Set ℤ := {2, 3, 4, 5, 6}

theorem inequality_system_integer_solutions :
  ∀ (n : ℤ), n ∈ integer_solutions ↔ inequality_system (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3087_308735


namespace NUMINAMATH_CALUDE_prob_same_length_is_one_fifth_l3087_308751

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of diagonals of each distinct length -/
def num_diagonals_per_length : ℕ := 3

/-- The probability of selecting two elements of the same length from T -/
def prob_same_length : ℚ := sorry

theorem prob_same_length_is_one_fifth :
  prob_same_length = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_is_one_fifth_l3087_308751


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3087_308757

theorem consecutive_odd_numbers_sum (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 2) + (x + 4) = (x + 4) + 52) →  -- sum condition
  (x = 25) :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3087_308757


namespace NUMINAMATH_CALUDE_tens_digit_of_7_pow_2005_l3087_308728

/-- The last two digits of 7^n follow a cycle of length 4 -/
def last_two_digits_cycle : List (Fin 100) := [7, 49, 43, 1]

/-- The tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_7_pow_2005 :
  tens_digit (7^2005 % 100) = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_7_pow_2005_l3087_308728


namespace NUMINAMATH_CALUDE_reflection_line_equation_l3087_308794

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The reflection of a point over a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

/-- The line of reflection given three points and their reflections -/
def reflectionLine (p q r p' q' r' : Point) : Line :=
  sorry

/-- Theorem stating that the line of reflection for the given points has the equation y = (3/5)x + 3/5 -/
theorem reflection_line_equation :
  let p := Point.mk (-2) 1
  let q := Point.mk 3 5
  let r := Point.mk 6 3
  let p' := Point.mk (-4) (-1)
  let q' := Point.mk 1 1
  let r' := Point.mk 4 (-1)
  let l := reflectionLine p q r p' q' r'
  l.slope = 3/5 ∧ l.intercept = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l3087_308794


namespace NUMINAMATH_CALUDE_congruence_problem_l3087_308724

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 19 = 3 → (3 * x + 15) % 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3087_308724


namespace NUMINAMATH_CALUDE_range_of_c_l3087_308775

theorem range_of_c (a b c : ℝ) 
  (ha : 6 < a ∧ a < 10) 
  (hb : a / 2 ≤ b ∧ b ≤ 2 * a) 
  (hc : c = a + b) : 
  9 < c ∧ c < 30 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l3087_308775


namespace NUMINAMATH_CALUDE_train_speed_l3087_308737

-- Define the length of the train in meters
def train_length : ℝ := 90

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 9

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed : 
  (train_length / crossing_time) * conversion_factor = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3087_308737


namespace NUMINAMATH_CALUDE_max_photo_area_l3087_308760

/-- Given a rectangular frame with area 59.6 square centimeters,
    prove that the maximum area of each of four equal-sized,
    non-overlapping photos within the frame is 14.9 square centimeters. -/
theorem max_photo_area (frame_area : ℝ) (num_photos : ℕ) :
  frame_area = 59.6 ∧ num_photos = 4 →
  (frame_area / num_photos : ℝ) = 14.9 := by
  sorry

end NUMINAMATH_CALUDE_max_photo_area_l3087_308760


namespace NUMINAMATH_CALUDE_linear_function_theorem_l3087_308731

theorem linear_function_theorem (k b : ℝ) :
  (∃ (x y : ℝ), y = k * x + b ∧ x = 0 ∧ y = -2) →
  (1/2 * |2/k| * 2 = 3) →
  ((k = 2/3 ∧ b = -2) ∨ (k = -2/3 ∧ b = -2)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l3087_308731


namespace NUMINAMATH_CALUDE_consultant_decision_probability_l3087_308704

theorem consultant_decision_probability :
  let p : ℝ := 0.8  -- probability of each consultant being correct
  let n : ℕ := 3    -- number of consultants
  let k : ℕ := 2    -- minimum number of correct opinions for a correct decision
  -- probability of making the correct decision
  (Finset.sum (Finset.range (n + 1 - k)) (λ i => 
    (n.choose (n - i)) * p^(n - i) * (1 - p)^i)) = 0.896 := by
  sorry

end NUMINAMATH_CALUDE_consultant_decision_probability_l3087_308704


namespace NUMINAMATH_CALUDE_floor_difference_count_l3087_308797

theorem floor_difference_count (x y z : ℝ) : 
  (⌊x⌋ = 5) → (⌊y⌋ = -3) → (⌊z⌋ = -1) → 
  (∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ ∃ (x' y' z' : ℝ), 
    ⌊x'⌋ = 5 ∧ ⌊y'⌋ = -3 ∧ ⌊z'⌋ = -1 ∧ ⌊x' - y' - z'⌋ = n) ∧ 
  Finset.card S = 3) := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_count_l3087_308797


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3087_308742

/-- Theorem: For a hyperbola passing through the point (4, √3) with asymptotes y = ± (1/2)x, 
    its standard equation is x²/4 - y² = 1 -/
theorem hyperbola_standard_equation 
  (passes_through : (4 : ℝ)^2 / 4 - 3 = 1) 
  (asymptotes : ∀ (x y : ℝ), y = (1/2) * x ∨ y = -(1/2) * x) :
  ∀ (x y : ℝ), x^2 / 4 - y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3087_308742


namespace NUMINAMATH_CALUDE_y_divisible_by_48_l3087_308796

theorem y_divisible_by_48 (y : ℤ) 
  (h1 : (2 + y) % 8 = 3^2 % 8)
  (h2 : (4 + y) % 64 = 2^2 % 64)
  (h3 : (6 + y) % 216 = 7^2 % 216) :
  y % 48 = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_divisible_by_48_l3087_308796


namespace NUMINAMATH_CALUDE_f_value_at_2_l3087_308774

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l3087_308774


namespace NUMINAMATH_CALUDE_rose_count_l3087_308726

theorem rose_count (total : ℕ) : 
  (300 ≤ total ∧ total ≤ 400) →
  (∃ x : ℕ, total = 21 * x + 13) →
  (∃ y : ℕ, total = 15 * y - 8) →
  total = 307 := by
sorry

end NUMINAMATH_CALUDE_rose_count_l3087_308726


namespace NUMINAMATH_CALUDE_car_sale_profit_l3087_308709

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let buying_price := 0.80 * P
  let selling_price := 1.3600000000000001 * P
  let percentage_increase := (selling_price / buying_price - 1) * 100
  percentage_increase = 70.00000000000002 := by
sorry

end NUMINAMATH_CALUDE_car_sale_profit_l3087_308709


namespace NUMINAMATH_CALUDE_cube_of_0_09_times_0_0007_l3087_308727

theorem cube_of_0_09_times_0_0007 : (0.09 : ℝ)^3 * 0.0007 = 0.0000005103 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_0_09_times_0_0007_l3087_308727


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3087_308754

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ (∀ y : ℝ, y^2 - 2*y + k = 0 → y = x)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3087_308754


namespace NUMINAMATH_CALUDE_fraction_simplification_l3087_308729

theorem fraction_simplification : 1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3087_308729


namespace NUMINAMATH_CALUDE_chicken_cost_proof_l3087_308759

def total_cost : ℝ := 16
def beef_pounds : ℝ := 3
def beef_price_per_pound : ℝ := 4
def oil_price : ℝ := 1
def people_paying : ℕ := 3
def individual_payment : ℝ := 1

theorem chicken_cost_proof :
  total_cost - (beef_pounds * beef_price_per_pound + oil_price) = people_paying * individual_payment := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_proof_l3087_308759


namespace NUMINAMATH_CALUDE_multiply_23_by_4_l3087_308707

theorem multiply_23_by_4 : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_23_by_4_l3087_308707


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l3087_308773

theorem halloween_candy_problem (debby_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  debby_candy = 32 →
  eaten_candy = 35 →
  remaining_candy = 39 →
  ∃ (sister_candy : ℕ), 
    debby_candy + sister_candy = eaten_candy + remaining_candy ∧
    sister_candy = 42 :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l3087_308773


namespace NUMINAMATH_CALUDE_child_running_speed_l3087_308768

/-- The child's running speed in meters per minute -/
def child_speed : ℝ := 74

/-- The sidewalk's speed in meters per minute -/
def sidewalk_speed : ℝ := child_speed - 55

theorem child_running_speed 
  (h1 : (child_speed + sidewalk_speed) * 4 = 372)
  (h2 : (child_speed - sidewalk_speed) * 3 = 165) :
  child_speed = 74 := by sorry

end NUMINAMATH_CALUDE_child_running_speed_l3087_308768


namespace NUMINAMATH_CALUDE_fundamental_disagreement_essence_l3087_308719

-- Define philosophical viewpoints
def materialist_viewpoint : String := "Without scenery, where does emotion come from?"
def idealist_viewpoint : String := "Without emotion, where does scenery come from?"

-- Define the concept of fundamental disagreement
def fundamental_disagreement (v1 v2 : String) : Prop := sorry

-- Define the essence of the world
inductive WorldEssence
| Material
| Consciousness

-- Theorem statement
theorem fundamental_disagreement_essence :
  fundamental_disagreement materialist_viewpoint idealist_viewpoint ↔
  ∃ (e : WorldEssence), (e = WorldEssence.Material ∨ e = WorldEssence.Consciousness) :=
sorry

end NUMINAMATH_CALUDE_fundamental_disagreement_essence_l3087_308719


namespace NUMINAMATH_CALUDE_no_special_polyhedron_l3087_308706

-- Define a polyhedron structure
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangle_faces : ℕ
  pentagon_faces : ℕ
  even_degree_vertices : ℕ

-- Define the conditions for our specific polyhedron
def SpecialPolyhedron (p : Polyhedron) : Prop :=
  p.faces = p.triangle_faces + p.pentagon_faces ∧
  p.pentagon_faces = 1 ∧
  p.vertices = p.even_degree_vertices ∧
  p.vertices - p.edges + p.faces = 2 ∧  -- Euler's formula
  3 * p.triangle_faces + 5 * p.pentagon_faces = 2 * p.edges

-- Theorem stating that such a polyhedron does not exist
theorem no_special_polyhedron :
  ¬ ∃ (p : Polyhedron), SpecialPolyhedron p :=
sorry

end NUMINAMATH_CALUDE_no_special_polyhedron_l3087_308706


namespace NUMINAMATH_CALUDE_mailman_problem_l3087_308758

theorem mailman_problem (total_junk_mail : ℕ) (white_mailboxes : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ) :
  total_junk_mail = 48 →
  white_mailboxes = 2 →
  red_mailboxes = 3 →
  mail_per_house = 6 →
  white_mailboxes + red_mailboxes + (total_junk_mail - (white_mailboxes + red_mailboxes) * mail_per_house) / mail_per_house = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mailman_problem_l3087_308758


namespace NUMINAMATH_CALUDE_class_fund_solution_l3087_308701

/-- Represents the number of bills in a class fund -/
structure ClassFund where
  bills_10 : ℕ
  bills_20 : ℕ

/-- Calculates the total amount in the fund -/
def total_amount (fund : ClassFund) : ℕ :=
  10 * fund.bills_10 + 20 * fund.bills_20

theorem class_fund_solution :
  ∃ (fund : ClassFund),
    total_amount fund = 120 ∧
    fund.bills_10 = 2 * fund.bills_20 ∧
    fund.bills_20 = 3 := by
  sorry

end NUMINAMATH_CALUDE_class_fund_solution_l3087_308701


namespace NUMINAMATH_CALUDE_sum_of_greater_is_greater_l3087_308771

theorem sum_of_greater_is_greater (a b c d : ℝ) : a > b → c > d → a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_greater_is_greater_l3087_308771


namespace NUMINAMATH_CALUDE_relationship_abc_l3087_308733

theorem relationship_abc (a b c : ℝ) 
  (h : Real.exp a + a = Real.log b + b ∧ Real.log b + b = Real.sqrt c + c ∧ Real.sqrt c + c = Real.sin 1) : 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3087_308733


namespace NUMINAMATH_CALUDE_problem_statement_l3087_308746

theorem problem_statement (x : ℕ) (h : x = 3) : x + x * (Nat.factorial x)^x = 651 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3087_308746


namespace NUMINAMATH_CALUDE_chord_ratio_implies_slope_l3087_308779

theorem chord_ratio_implies_slope (k : ℝ) (h1 : k > 0) : 
  let l := {(x, y) : ℝ × ℝ | y = k * x}
  let C1 := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  let C2 := {(x, y) : ℝ × ℝ | (x - 3)^2 + y^2 = 1}
  let chord1 := {p : ℝ × ℝ | p ∈ l ∩ C1}
  let chord2 := {p : ℝ × ℝ | p ∈ l ∩ C2}
  (∃ (p q : ℝ × ℝ), p ∈ chord1 ∧ q ∈ chord1 ∧ p ≠ q) →
  (∃ (r s : ℝ × ℝ), r ∈ chord2 ∧ s ∈ chord2 ∧ r ≠ s) →
  (∃ (p q r s : ℝ × ℝ), p ∈ chord1 ∧ q ∈ chord1 ∧ r ∈ chord2 ∧ s ∈ chord2 ∧
    dist p q / dist r s = 3) →
  k = 1/3 :=
by sorry


end NUMINAMATH_CALUDE_chord_ratio_implies_slope_l3087_308779


namespace NUMINAMATH_CALUDE_horner_rule_v3_l3087_308720

def horner_rule (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def polynomial (x : ℝ) : ℝ :=
  3 * x^4 - x^2 + 2 * x + 1

theorem horner_rule_v3 (x : ℝ) (h : x = 2) :
  let a := [1, 2, 0, -1, 3]
  let v₃ := horner_rule (a.take 4) x
  v₃ = 20 := by sorry

end NUMINAMATH_CALUDE_horner_rule_v3_l3087_308720


namespace NUMINAMATH_CALUDE_distribute_seven_books_four_friends_l3087_308700

/-- The number of ways to distribute n identical books among k friends, 
    where each friend must have at least one book -/
def distribute_books (n k : ℕ) : ℕ := sorry

/-- Theorem: Distributing 7 books among 4 friends results in 34 ways -/
theorem distribute_seven_books_four_friends : 
  distribute_books 7 4 = 34 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_books_four_friends_l3087_308700


namespace NUMINAMATH_CALUDE_committee_formation_count_l3087_308770

theorem committee_formation_count (n : ℕ) (k : ℕ) : n = 30 → k = 5 →
  (n.choose 1) * ((n - 1).choose 1) * ((n - 2).choose (k - 2)) = 2850360 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3087_308770


namespace NUMINAMATH_CALUDE_rubber_boat_lost_at_4pm_l3087_308708

/-- Represents the time when the rubber boat was lost (in hours before 5 PM) -/
def time_lost : ℝ := 1

/-- Represents the speed of the ship in still water -/
def ship_speed : ℝ := 1

/-- Represents the speed of the river flow -/
def river_speed : ℝ := 1

/-- Theorem stating that the rubber boat was lost at 4 PM -/
theorem rubber_boat_lost_at_4pm :
  (5 - time_lost) * (ship_speed - river_speed) + (6 - time_lost) * river_speed = ship_speed + river_speed :=
by sorry

end NUMINAMATH_CALUDE_rubber_boat_lost_at_4pm_l3087_308708


namespace NUMINAMATH_CALUDE_inequality_theorem_l3087_308740

theorem inequality_theorem (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (b^2 / a + a^2 / b) ≤ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3087_308740


namespace NUMINAMATH_CALUDE_honda_second_shift_production_l3087_308730

/-- Represents the production of cars in a Honda factory --/
structure CarProduction where
  second_shift : ℕ
  day_shift : ℕ
  total : ℕ

/-- The conditions of the Honda car production problem --/
def honda_production : CarProduction :=
  { second_shift := 0,  -- placeholder, will be proven
    day_shift := 0,     -- placeholder, will be proven
    total := 5500 }

/-- The theorem stating the solution to the Honda car production problem --/
theorem honda_second_shift_production :
  ∃ (p : CarProduction),
    p.day_shift = 4 * p.second_shift ∧
    p.total = p.day_shift + p.second_shift ∧
    p.total = honda_production.total ∧
    p.second_shift = 1100 := by
  sorry

end NUMINAMATH_CALUDE_honda_second_shift_production_l3087_308730


namespace NUMINAMATH_CALUDE_function_equation_solution_l3087_308705

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3087_308705


namespace NUMINAMATH_CALUDE_solution_difference_l3087_308715

/-- Given that r and s are distinct solutions to the equation (6x-18)/(x^2+4x-21) = x+3,
    and r > s, prove that r - s = 10. -/
theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (6*r - 18) / (r^2 + 4*r - 21) = r + 3 →
  (6*s - 18) / (s^2 + 4*s - 21) = s + 3 →
  r > s →
  r - s = 10 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3087_308715


namespace NUMINAMATH_CALUDE_square_difference_equation_l3087_308778

theorem square_difference_equation : 9^2 - 8^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equation_l3087_308778


namespace NUMINAMATH_CALUDE_count_flippy_divisible_by_18_l3087_308767

def is_flippy (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ 
    n = a * 100000 + b * 10000 + a * 1000 + b * 100 + a * 10 + b

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

theorem count_flippy_divisible_by_18 :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_flippy n ∧ is_six_digit n ∧ n % 18 = 0) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_flippy_divisible_by_18_l3087_308767


namespace NUMINAMATH_CALUDE_product_remainder_l3087_308753

theorem product_remainder (a b m : ℕ) (h : a * b = 145 * 155) (hm : m = 12) : 
  (a * b) % m = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3087_308753


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l3087_308798

theorem tennis_tournament_matches (total_players : ℕ) (bye_players : ℕ) (first_round_players : ℕ) (first_round_matches : ℕ) :
  total_players = 128 →
  bye_players = 36 →
  first_round_players = 92 →
  first_round_matches = 46 →
  first_round_players = 2 * first_round_matches →
  total_players = bye_players + first_round_players →
  (∃ (total_matches : ℕ), total_matches = first_round_matches + (total_players - 1) ∧ total_matches = 127) :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l3087_308798


namespace NUMINAMATH_CALUDE_nine_times_2010_equals_201_l3087_308756

-- Define the operation
def diamond (a b : ℚ) : ℚ := (a * b) / (a + b)

-- Define a function that applies the operation n times
def apply_n_times (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => diamond (apply_n_times n x) x

-- Theorem statement
theorem nine_times_2010_equals_201 :
  apply_n_times 9 2010 = 201 := by sorry

end NUMINAMATH_CALUDE_nine_times_2010_equals_201_l3087_308756


namespace NUMINAMATH_CALUDE_xyz_sum_equals_96_l3087_308785

theorem xyz_sum_equals_96 
  (x y z : ℝ) 
  (hpos_x : x > 0) 
  (hpos_y : y > 0) 
  (hpos_z : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 64)
  (eq3 : z^2 + x*z + x^2 = 172) : 
  x*y + y*z + x*z = 96 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_equals_96_l3087_308785


namespace NUMINAMATH_CALUDE_min_value_d_l3087_308712

/-- Given positive integers a, b, c, and d where a < b < c < d, and a system of equations
    with exactly one solution, the minimum value of d is 602. -/
theorem min_value_d (a b c d : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : ∃! (x y : ℝ), 3 * x + y = 3004 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d = 602 := by
  sorry

end NUMINAMATH_CALUDE_min_value_d_l3087_308712


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_div_by_five_l3087_308743

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of four consecutive primes starting from the nth prime -/
def sumFourConsecutivePrimes (n : ℕ) : ℕ :=
  nthPrime n + nthPrime (n + 1) + nthPrime (n + 2) + nthPrime (n + 3)

/-- The main theorem -/
theorem smallest_sum_four_consecutive_primes_div_by_five :
  ∀ n : ℕ, sumFourConsecutivePrimes n % 5 = 0 → sumFourConsecutivePrimes n ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_div_by_five_l3087_308743


namespace NUMINAMATH_CALUDE_three_digit_numbers_34_times_sum_of_digits_l3087_308718

theorem three_digit_numbers_34_times_sum_of_digits : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n = 34 * (n / 100 + (n / 10 % 10) + (n % 10))} = 
  {102, 204, 306, 408} := by
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_34_times_sum_of_digits_l3087_308718


namespace NUMINAMATH_CALUDE_no_overlapping_attendees_l3087_308792

theorem no_overlapping_attendees (total_guests : ℕ) 
  (oates_attendees hall_attendees singh_attendees brown_attendees : ℕ) :
  total_guests = 350 ∧
  oates_attendees = 105 ∧
  hall_attendees = 98 ∧
  singh_attendees = 82 ∧
  brown_attendees = 65 ∧
  oates_attendees + hall_attendees + singh_attendees + brown_attendees = total_guests →
  (∃ (overlapping_attendees : ℕ), overlapping_attendees = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_overlapping_attendees_l3087_308792


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l3087_308750

theorem greatest_power_under_500 (a b : ℕ) : 
  a > 0 → b > 1 → a^b < 500 → (∀ x y : ℕ, x > 0 → y > 1 → x^y < 500 → x^y ≤ a^b) → a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_under_500_l3087_308750


namespace NUMINAMATH_CALUDE_vector_magnitude_l3087_308723

/-- Given plane vectors a and b with angle π/2 between them, |a| = 1, and |b| = √3, prove |2a - b| = √7 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a • b = 0) → (‖a‖ = 1) → (‖b‖ = Real.sqrt 3) → ‖2 • a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3087_308723


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l3087_308772

theorem sugar_solution_replacement (initial_sugar_percent : ℝ) 
                                   (final_sugar_percent : ℝ) 
                                   (second_sugar_percent : ℝ) 
                                   (replaced_portion : ℝ) : 
  initial_sugar_percent = 10 →
  final_sugar_percent = 16 →
  second_sugar_percent = 34 →
  (100 - replaced_portion) * initial_sugar_percent / 100 + 
    replaced_portion * second_sugar_percent / 100 = 
    final_sugar_percent →
  replaced_portion = 25 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l3087_308772


namespace NUMINAMATH_CALUDE_social_gathering_attendance_l3087_308777

theorem social_gathering_attendance
  (num_men : ℕ)
  (dances_per_man : ℕ)
  (dances_per_woman : ℕ)
  (h_num_men : num_men = 15)
  (h_dances_per_man : dances_per_man = 4)
  (h_dances_per_woman : dances_per_woman = 3) :
  (num_men * dances_per_man) / dances_per_woman = 20 := by
sorry

end NUMINAMATH_CALUDE_social_gathering_attendance_l3087_308777


namespace NUMINAMATH_CALUDE_p_iff_q_l3087_308721

theorem p_iff_q : ∀ x : ℝ, (x > 1 ∨ x < -1) ↔ |x + 1| + |x - 1| > 2 := by
  sorry

end NUMINAMATH_CALUDE_p_iff_q_l3087_308721


namespace NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l3087_308783

/-- 
Given odd integers a and b, the number of odd terms 
in the expansion of (a+b)^8 is equal to 2.
-/
theorem odd_terms_in_binomial_expansion (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) : 
  (Finset.filter (fun i => Odd (Nat.choose 8 i * a^(8-i) * b^i)) 
    (Finset.range 9)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l3087_308783


namespace NUMINAMATH_CALUDE_unique_max_f_and_sum_of_digits_l3087_308716

/-- Number of positive integer divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- Function f(n) = d(n) / n^(1/4) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val ^ (1/4 : ℝ)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem unique_max_f_and_sum_of_digits :
  ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N.val = 18 := by sorry

end NUMINAMATH_CALUDE_unique_max_f_and_sum_of_digits_l3087_308716


namespace NUMINAMATH_CALUDE_part1_part2_l3087_308732

-- Define the quadratic expression
def quadratic (a x : ℝ) : ℝ := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- Part 1
theorem part1 : 
  ∀ x : ℝ, quadratic (-2) x < 0 ↔ x ≠ -1 :=
sorry

-- Part 2
theorem part2 : 
  (∀ x : ℝ, quadratic a x < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3087_308732


namespace NUMINAMATH_CALUDE_hot_tea_sales_average_l3087_308791

/-- Represents the linear relationship between temperature and cups of hot tea sold -/
structure HotDrinkSales where
  slope : ℝ
  intercept : ℝ

/-- Calculates the average cups of hot tea sold given average temperature -/
def average_sales (model : HotDrinkSales) (avg_temp : ℝ) : ℝ :=
  model.slope * avg_temp + model.intercept

theorem hot_tea_sales_average (model : HotDrinkSales) (avg_temp : ℝ) 
    (h1 : model.slope = -2)
    (h2 : model.intercept = 58)
    (h3 : avg_temp = 12) :
    average_sales model avg_temp = 34 := by
  sorry

#check hot_tea_sales_average

end NUMINAMATH_CALUDE_hot_tea_sales_average_l3087_308791


namespace NUMINAMATH_CALUDE_elise_remaining_money_l3087_308722

/-- Calculates the remaining money for Elise given her initial amount, savings, and expenses. -/
def remaining_money (initial : ℕ) (savings : ℕ) (comic_expense : ℕ) (puzzle_expense : ℕ) : ℕ :=
  initial + savings - comic_expense - puzzle_expense

/-- Proves that Elise's remaining money is $1 given her initial amount, savings, and expenses. -/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

#eval remaining_money 8 13 2 18

end NUMINAMATH_CALUDE_elise_remaining_money_l3087_308722


namespace NUMINAMATH_CALUDE_initial_inventory_l3087_308734

def bookshop_inventory (initial_books : ℕ) : Prop :=
  let saturday_instore := 37
  let saturday_online := 128
  let sunday_instore := 2 * saturday_instore
  let sunday_online := saturday_online + 34
  let shipment := 160
  let current_books := 502
  initial_books = current_books + saturday_instore + saturday_online + sunday_instore + sunday_online - shipment

theorem initial_inventory : ∃ (x : ℕ), bookshop_inventory x ∧ x = 743 := by
  sorry

end NUMINAMATH_CALUDE_initial_inventory_l3087_308734


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l3087_308766

/-- Represents the maximum distance a car can travel by switching tires -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  let switch_point := front_tire_life / 2
  switch_point + min (front_tire_life - switch_point) (rear_tire_life - switch_point)

/-- Theorem stating the maximum distance a car can travel with given tire lives -/
theorem max_distance_for_given_tires :
  max_distance 21000 28000 = 24000 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l3087_308766


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3087_308755

def f (x : ℝ) := -x^2 + 2

theorem max_min_values_of_f :
  let a := -1
  let b := 3
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 2 ∧ f x_min = -7 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3087_308755


namespace NUMINAMATH_CALUDE_non_honda_red_percentage_is_51_25_l3087_308776

/-- Represents the car population in Chennai -/
structure CarPopulation where
  total : Nat
  honda : Nat
  toyota : Nat
  ford : Nat
  other : Nat
  honda_red_ratio : Rat
  toyota_red_ratio : Rat
  ford_red_ratio : Rat
  other_red_ratio : Rat

/-- Calculates the percentage of non-Honda cars that are red -/
def non_honda_red_percentage (pop : CarPopulation) : Rat :=
  let non_honda_total := pop.toyota + pop.ford + pop.other
  let non_honda_red := pop.toyota * pop.toyota_red_ratio + 
                       pop.ford * pop.ford_red_ratio + 
                       pop.other * pop.other_red_ratio
  (non_honda_red / non_honda_total) * 100

/-- The main theorem stating that the percentage of non-Honda cars that are red is 51.25% -/
theorem non_honda_red_percentage_is_51_25 (pop : CarPopulation) 
  (h1 : pop.total = 900)
  (h2 : pop.honda = 500)
  (h3 : pop.toyota = 200)
  (h4 : pop.ford = 150)
  (h5 : pop.other = 50)
  (h6 : pop.honda_red_ratio = 9/10)
  (h7 : pop.toyota_red_ratio = 3/4)
  (h8 : pop.ford_red_ratio = 3/10)
  (h9 : pop.other_red_ratio = 2/5) :
  non_honda_red_percentage pop = 51.25 := by
  sorry

#eval non_honda_red_percentage {
  total := 900,
  honda := 500,
  toyota := 200,
  ford := 150,
  other := 50,
  honda_red_ratio := 9/10,
  toyota_red_ratio := 3/4,
  ford_red_ratio := 3/10,
  other_red_ratio := 2/5
}

end NUMINAMATH_CALUDE_non_honda_red_percentage_is_51_25_l3087_308776


namespace NUMINAMATH_CALUDE_twenty_sheets_joined_length_l3087_308781

/-- The length of joined papers given the number of sheets, length per sheet, and overlap length -/
def joinedPapersLength (numSheets : ℕ) (sheetLength : ℝ) (overlapLength : ℝ) : ℝ :=
  numSheets * sheetLength - (numSheets - 1) * overlapLength

/-- Theorem stating that 20 sheets of 10 cm paper with 0.5 cm overlap results in 190.5 cm total length -/
theorem twenty_sheets_joined_length :
  joinedPapersLength 20 10 0.5 = 190.5 := by
  sorry

#eval joinedPapersLength 20 10 0.5

end NUMINAMATH_CALUDE_twenty_sheets_joined_length_l3087_308781


namespace NUMINAMATH_CALUDE_drunkard_theorem_l3087_308745

structure PubSystem where
  states : Finset (Fin 4)
  transition : Fin 4 → Fin 4 → ℚ
  start_state : Fin 4

def drunkard_walk (ps : PubSystem) (n : ℕ) : Fin 4 → ℚ :=
  sorry

theorem drunkard_theorem (ps : PubSystem) :
  (ps.states = {0, 1, 2, 3}) →
  (ps.transition 0 1 = 1/3) →
  (ps.transition 0 2 = 1/3) →
  (ps.transition 0 3 = 0) →
  (ps.transition 1 0 = 1/2) →
  (ps.transition 1 2 = 1/3) →
  (ps.transition 1 3 = 1/2) →
  (ps.transition 2 0 = 1/2) →
  (ps.transition 2 1 = 1/3) →
  (ps.transition 2 3 = 1/2) →
  (ps.transition 3 1 = 1/3) →
  (ps.transition 3 2 = 1/3) →
  (ps.transition 3 0 = 0) →
  (ps.start_state = 0) →
  ((drunkard_walk ps 5) 2 = 55/162) ∧
  (∀ n > 5, (drunkard_walk ps n) 1 > (drunkard_walk ps n) 0 ∧
            (drunkard_walk ps n) 2 > (drunkard_walk ps n) 0 ∧
            (drunkard_walk ps n) 1 > (drunkard_walk ps n) 3 ∧
            (drunkard_walk ps n) 2 > (drunkard_walk ps n) 3) :=
by sorry

end NUMINAMATH_CALUDE_drunkard_theorem_l3087_308745


namespace NUMINAMATH_CALUDE_percentage_problem_l3087_308784

theorem percentage_problem (x : ℝ) : (23 / 100) * x = 150 → x = 15000 / 23 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3087_308784


namespace NUMINAMATH_CALUDE_luke_fish_fillets_l3087_308762

/-- Calculates the number of fillets per fish given the total number of fish caught and total fillets obtained. -/
def filletsPerFish (fishPerDay : ℕ) (days : ℕ) (totalFillets : ℕ) : ℚ :=
  totalFillets / (fishPerDay * days)

/-- Proves that the number of fillets per fish is 2 given the problem conditions. -/
theorem luke_fish_fillets : filletsPerFish 2 30 120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_luke_fish_fillets_l3087_308762


namespace NUMINAMATH_CALUDE_scientific_notation_34_million_l3087_308748

theorem scientific_notation_34_million : 
  ∃ (a : ℝ) (n : ℤ), 34000000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.4 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_34_million_l3087_308748


namespace NUMINAMATH_CALUDE_race_time_comparison_l3087_308738

theorem race_time_comparison 
  (a : ℝ) (V : ℝ) 
  (h1 : a > 0) (h2 : V > 0) : 
  let planned_time := a / V
  let first_half_time := a / (2 * 1.25 * V)
  let second_half_time := a / (2 * 0.8 * V)
  let actual_time := first_half_time + second_half_time
  actual_time > planned_time :=
by sorry

end NUMINAMATH_CALUDE_race_time_comparison_l3087_308738


namespace NUMINAMATH_CALUDE_two_talent_students_l3087_308702

theorem two_talent_students (total : ℕ) (all_three : ℕ) (cant_sing : ℕ) (cant_dance : ℕ) (cant_act : ℕ) : 
  total = 150 →
  all_three = 10 →
  cant_sing = 70 →
  cant_dance = 90 →
  cant_act = 50 →
  ∃ (two_talents : ℕ), two_talents = 80 ∧ 
    (total - cant_sing) + (total - cant_dance) + (total - cant_act) - two_talents - 2 * all_three = total :=
by sorry

end NUMINAMATH_CALUDE_two_talent_students_l3087_308702


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3087_308788

theorem part_to_whole_ratio (N : ℚ) (x : ℚ) (h1 : N = 280) (h2 : x + 4 = N / 4 - 10) : x / N = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3087_308788


namespace NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l3087_308793

-- Define the functions
def f (x : ℝ) := abs x
def g (x : ℝ) := -x^2 - 5*x - 4

-- Define the vertical distance function
def vertical_distance (x : ℝ) := abs (f x - g x)

-- Theorem statement
theorem min_vertical_distance_is_zero :
  ∃ x : ℝ, vertical_distance x = 0 ∧ ∀ y : ℝ, vertical_distance y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l3087_308793


namespace NUMINAMATH_CALUDE_circle_equation_l3087_308761

/-- Theorem: Equation of a circle with specific properties -/
theorem circle_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ y : ℝ, (0 - a)^2 + (y - b)^2 = (a^2 + b^2) → |y| ≤ 1) →
  (∀ x : ℝ, (x - a)^2 + (0 - b)^2 = (a^2 + b^2) → |x| ≤ 2) →
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 
    (x₁ - a)^2 + (0 - b)^2 = (a^2 + b^2) ∧
    (x₂ - a)^2 + (0 - b)^2 = (a^2 + b^2) ∧
    (x₂ - x₁) / (4 - (x₂ - x₁)) = 3) →
  a = Real.sqrt 7 ∧ b = 2 ∧ a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3087_308761


namespace NUMINAMATH_CALUDE_solution_set_ln_inequality_l3087_308710

theorem solution_set_ln_inequality :
  {x : ℝ | x > 0 ∧ 2 - Real.log x ≥ 0} = Set.Ioo 0 (Real.exp 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_ln_inequality_l3087_308710
