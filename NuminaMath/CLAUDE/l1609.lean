import Mathlib

namespace NUMINAMATH_CALUDE_k_range_for_two_distinct_roots_l1609_160918

/-- A quadratic equation ax^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
axiom two_distinct_roots_iff_positive_discriminant (a b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- The range of k for which kx^2 - 6x + 9 = 0 has two distinct real roots -/
theorem k_range_for_two_distinct_roots :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x + 9 = 0 ∧ k * y^2 - 6 * y + 9 = 0) ↔ k < 1 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_two_distinct_roots_l1609_160918


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1609_160993

/-- Calculates the profit percentage when selling a number of articles at the cost price of a different number of articles. -/
def profit_percentage (articles_sold : ℕ) (articles_cost_price : ℕ) : ℚ :=
  ((articles_cost_price : ℚ) - (articles_sold : ℚ)) / (articles_sold : ℚ) * 100

/-- Theorem stating that when a shopkeeper sells 50 articles at the cost price of 60 articles, the profit percentage is 20%. -/
theorem shopkeeper_profit : profit_percentage 50 60 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1609_160993


namespace NUMINAMATH_CALUDE_digit_replacement_theorem_l1609_160925

theorem digit_replacement_theorem : ∃ (x y z w : ℕ), 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ w ≤ 9 ∧
  42 * (10 * x + 8) = 2000 + 100 * y + 10 * z + w ∧
  (x + y + z + w) % 2 = 1 ∧
  2000 ≤ 42 * (10 * x + 8) ∧ 42 * (10 * x + 8) < 3000 :=
by sorry

end NUMINAMATH_CALUDE_digit_replacement_theorem_l1609_160925


namespace NUMINAMATH_CALUDE_scientific_notation_36000_l1609_160927

/-- Proves that 36000 is equal to 3.6 * 10^4 in scientific notation -/
theorem scientific_notation_36000 :
  36000 = 3.6 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_36000_l1609_160927


namespace NUMINAMATH_CALUDE_missing_entry_is_L_l1609_160976

/-- Represents the possible entries in the table -/
inductive TableEntry
| W
| Q
| L

/-- Represents a position in the 3x3 table -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Represents the 3x3 table -/
def Table := Position → TableEntry

/-- The given table with known entries -/
def givenTable : Table :=
  fun pos => match pos with
  | ⟨0, 0⟩ => TableEntry.W
  | ⟨0, 2⟩ => TableEntry.Q
  | ⟨1, 0⟩ => TableEntry.L
  | ⟨1, 1⟩ => TableEntry.Q
  | ⟨1, 2⟩ => TableEntry.W
  | ⟨2, 0⟩ => TableEntry.Q
  | ⟨2, 1⟩ => TableEntry.W
  | ⟨2, 2⟩ => TableEntry.L
  | _ => TableEntry.W  -- Default value for unknown positions

theorem missing_entry_is_L :
  givenTable ⟨0, 1⟩ = TableEntry.L :=
sorry

end NUMINAMATH_CALUDE_missing_entry_is_L_l1609_160976


namespace NUMINAMATH_CALUDE_third_grade_class_size_l1609_160916

/-- Represents the number of students in each third grade class -/
def third_grade_students : ℕ := sorry

/-- Represents the total number of classes -/
def total_classes : ℕ := 5 + 4 + 4

/-- Represents the total number of students in fourth and fifth grades -/
def fourth_fifth_students : ℕ := 4 * 28 + 4 * 27

/-- Represents the cost of lunch per student in cents -/
def lunch_cost_per_student : ℕ := 210 + 50 + 20

/-- Represents the total cost of all lunches in cents -/
def total_lunch_cost : ℕ := 103600

theorem third_grade_class_size :
  third_grade_students = 30 ∧
  third_grade_students * 5 * lunch_cost_per_student +
  fourth_fifth_students * lunch_cost_per_student = total_lunch_cost :=
sorry

end NUMINAMATH_CALUDE_third_grade_class_size_l1609_160916


namespace NUMINAMATH_CALUDE_table_capacity_l1609_160970

theorem table_capacity (invited : ℕ) (no_shows : ℕ) (tables : ℕ) : 
  invited = 18 → no_shows = 12 → tables = 2 → 
  (invited - no_shows) / tables = 3 := by sorry

end NUMINAMATH_CALUDE_table_capacity_l1609_160970


namespace NUMINAMATH_CALUDE_slope_range_l1609_160919

-- Define the line l passing through point P(-2, 2) with slope k
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y - 2 = k * (x + 2)}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 + 12*x + 35 = 0}

-- Define the condition that circles with centers on l and radius 1 have no common points with C
def no_common_points (k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ line_l k →
    ∀ (a b : ℝ), (a - x)^2 + (b - y)^2 ≤ 1 →
      (a, b) ∉ circle_C

-- State the theorem
theorem slope_range :
  ∀ k : ℝ, no_common_points k →
    k < 0 ∨ k > 4/3 :=
sorry

end NUMINAMATH_CALUDE_slope_range_l1609_160919


namespace NUMINAMATH_CALUDE_sum_digits_of_valid_hex_count_l1609_160905

/-- Represents a hexadecimal digit -/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Represents a hexadecimal number -/
def HexNumber := List HexDigit

/-- Converts a natural number to hexadecimal representation -/
def toHex (n : ℕ) : HexNumber :=
  sorry

/-- Checks if a hexadecimal number contains only numeric digits and doesn't start with 0 -/
def isValidHex (h : HexNumber) : Bool :=
  sorry

/-- Counts valid hexadecimal numbers in the first n positive integers -/
def countValidHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number -/
def sumDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem sum_digits_of_valid_hex_count :
  sumDigits (countValidHex 2000) = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_digits_of_valid_hex_count_l1609_160905


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_5_l1609_160943

def is_multiple_of_5 (n : ℕ) : Prop :=
  ∃ k : ℤ, 7 * (n - 3)^7 - 2 * n^3 + 21 * n - 36 = 5 * k

theorem largest_n_multiple_of_5 :
  ∀ n : ℕ, n < 100000 → is_multiple_of_5 n → n ≤ 99998 ∧
  is_multiple_of_5 99998 ∧
  99998 < 100000 :=
sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_5_l1609_160943


namespace NUMINAMATH_CALUDE_surface_area_difference_after_cube_removal_l1609_160995

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the surface area difference after cube removal -/
def surfaceAreaDifference (length width height cubeEdge : ℝ) : ℝ :=
  let originalArea := surfaceArea length width height
  let removedArea := 3 * cubeEdge ^ 2
  let addedArea := cubeEdge ^ 2
  originalArea - removedArea + addedArea - originalArea

theorem surface_area_difference_after_cube_removal :
  surfaceAreaDifference 5 4 3 2 = -8 := by sorry

end NUMINAMATH_CALUDE_surface_area_difference_after_cube_removal_l1609_160995


namespace NUMINAMATH_CALUDE_parabola_range_l1609_160997

-- Define the function f(x) = x^2 - 4x + 5
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem parabola_range :
  ∀ y : ℝ, (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = y) ↔ 1 ≤ y ∧ y < 5 := by sorry

end NUMINAMATH_CALUDE_parabola_range_l1609_160997


namespace NUMINAMATH_CALUDE_f_extrema_l1609_160951

def f (x : ℝ) := 3 * x^4 - 6 * x^2 + 4

theorem f_extrema :
  (∀ x ∈ Set.Icc (-1) 3, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 1) ∧
  (∀ x ∈ Set.Icc (-1) 3, f x ≤ 193) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 193) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l1609_160951


namespace NUMINAMATH_CALUDE_power_of_p_is_one_l1609_160955

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property of being a positive even integer with a positive units digit -/
def isPositiveEvenWithPositiveUnitsDigit (p : ℕ) : Prop :=
  p > 0 ∧ p % 2 = 0 ∧ unitsDigit p > 0

theorem power_of_p_is_one (p : ℕ) (k : ℕ) 
  (h1 : isPositiveEvenWithPositiveUnitsDigit p)
  (h2 : unitsDigit (p + 1) = 7)
  (h3 : unitsDigit (p^3) - unitsDigit (p^k) = 0) :
  k = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_p_is_one_l1609_160955


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1609_160992

theorem fraction_evaluation : 
  let x : ℚ := 5
  (x^6 - 16*x^3 + x^2 + 64) / (x^3 - 8) = 4571 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1609_160992


namespace NUMINAMATH_CALUDE_new_shoes_duration_l1609_160981

/-- The duration of new shoes given repair and purchase costs -/
theorem new_shoes_duration (repair_cost : ℝ) (repair_duration : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  repair_cost = 11.50 →
  repair_duration = 1 →
  new_cost = 28.00 →
  cost_increase_percentage = 0.2173913043478261 →
  ∃ (new_duration : ℝ),
    new_duration = 2 ∧
    (new_cost / new_duration) = (repair_cost / repair_duration) * (1 + cost_increase_percentage) :=
by
  sorry

end NUMINAMATH_CALUDE_new_shoes_duration_l1609_160981


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l1609_160920

theorem equidistant_point_on_x_axis : ∃ x : ℝ, 
  (x^2 + 6*x + 9 = x^2 + 25) ∧ (x = 8/3) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l1609_160920


namespace NUMINAMATH_CALUDE_train_tickets_theorem_l1609_160903

/-- Calculates the number of different tickets needed for a train route -/
def number_of_tickets (intermediate_stops : ℕ) : ℕ :=
  intermediate_stops * (intermediate_stops + 3) + 2

/-- Theorem stating that a train route with 5 intermediate stops requires 42 different tickets -/
theorem train_tickets_theorem :
  number_of_tickets 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_tickets_theorem_l1609_160903


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1609_160900

/-- Given that i² = -1, prove that w = -2i/3 is the solution to the equation 3 - iw = 1 + 2iw -/
theorem complex_equation_solution (i : ℂ) (h : i^2 = -1) :
  let w : ℂ := -2*i/3
  3 - i*w = 1 + 2*i*w := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1609_160900


namespace NUMINAMATH_CALUDE_wedge_volume_l1609_160968

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (angle : ℝ) : 
  d = 16 →
  angle = 60 →
  (π * (d / 2)^2 * d) / 2 = 512 * π :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_l1609_160968


namespace NUMINAMATH_CALUDE_range_of_m_l1609_160931

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | x^2 + (m+2)*x + 1 = 0}

-- State the theorem
theorem range_of_m (m : ℝ) : (A m ∩ {x : ℝ | x ≠ 0} ≠ ∅) → (-4 < m ∧ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1609_160931


namespace NUMINAMATH_CALUDE_two_queens_or_at_least_one_jack_probability_l1609_160953

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Jacks in a standard deck -/
def num_jacks : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The probability of drawing either two Queens or at least 1 Jack from a standard deck when selecting 2 cards randomly -/
def prob_two_queens_or_at_least_one_jack : ℚ := 2 / 13

theorem two_queens_or_at_least_one_jack_probability :
  prob_two_queens_or_at_least_one_jack = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_two_queens_or_at_least_one_jack_probability_l1609_160953


namespace NUMINAMATH_CALUDE_ellipse_sum_l1609_160969

theorem ellipse_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →
  (h = 3 ∧ k = -5) →
  (a = 7 ∧ b = 4) →
  h + k + a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l1609_160969


namespace NUMINAMATH_CALUDE_stating_repeating_decimal_equals_fraction_l1609_160907

/-- Represents a repeating decimal where the fractional part is 0.325325325... -/
def repeating_decimal : ℚ := 3/10 + 25/990

/-- The fraction 161/495 in its lowest terms -/
def target_fraction : ℚ := 161/495

/-- 
Theorem stating that the repeating decimal 0.3̅25̅ is equal to the fraction 161/495
-/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_stating_repeating_decimal_equals_fraction_l1609_160907


namespace NUMINAMATH_CALUDE_foci_coordinates_l1609_160954

/-- Definition of a hyperbola with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Definition of the distance from center to focus for this hyperbola -/
def c : ℝ := 2

/-- The coordinates of the foci of the hyperbola x^2 - y^2/3 = 1 are (±2, 0) -/
theorem foci_coordinates :
  ∀ x y : ℝ, hyperbola x y → (x = c ∨ x = -c) ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_foci_coordinates_l1609_160954


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l1609_160921

/-- Calculates the profit percentage for a cricket bat sale --/
theorem cricket_bat_profit_percentage 
  (selling_price : ℝ) 
  (initial_profit : ℝ) 
  (tax_rate : ℝ) 
  (discount_rate : ℝ) 
  (h1 : selling_price = 850)
  (h2 : initial_profit = 255)
  (h3 : tax_rate = 0.07)
  (h4 : discount_rate = 0.05) : 
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 25.71) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l1609_160921


namespace NUMINAMATH_CALUDE_recurrence_sequence_a8_l1609_160958

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem recurrence_sequence_a8 
  (a : ℕ → ℕ) 
  (h : RecurrenceSequence a) 
  (h7 : a 7 = 120) : 
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a8_l1609_160958


namespace NUMINAMATH_CALUDE_square_and_cube_difference_l1609_160939

theorem square_and_cube_difference (a b : ℝ) 
  (sum_eq : a + b = 8) 
  (diff_eq : a - b = 4) : 
  a^2 - b^2 = 32 ∧ a^3 - b^3 = 208 := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_difference_l1609_160939


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l1609_160936

theorem no_alpha_sequence_exists :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ),
    (0 < α ∧ α < 1) ∧
    (∀ n, 0 < a n) ∧
    (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l1609_160936


namespace NUMINAMATH_CALUDE_proposition_count_l1609_160962

theorem proposition_count : ∃! n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x * y ≥ 0) ∧ 
  (∀ x y : ℝ, x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0 ∨ x ≤ 0 ∧ y ≤ 0) ∧
  (∃ x y : ℝ, ¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)) ∧
  (∀ x y : ℝ, x * y < 0 → x < 0 ∨ y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_proposition_count_l1609_160962


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1609_160991

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ (2 * x^2 - 7)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1609_160991


namespace NUMINAMATH_CALUDE_platform_length_l1609_160977

/-- Given a train with the following properties:
  * Length: 300 meters
  * Starting from rest
  * Constant acceleration
  * Crosses a signal pole in 24 seconds
  * Crosses a platform in 39 seconds
  Prove that the length of the platform is approximately 492.19 meters. -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : pole_time = 24)
  (h3 : platform_time = 39) :
  ∃ (platform_length : ℝ), 
    (abs (platform_length - 492.19) < 0.01) ∧ 
    (∃ (a : ℝ), 
      (train_length = (1/2) * a * pole_time^2) ∧
      (train_length + platform_length = (1/2) * a * platform_time^2)) :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1609_160977


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l1609_160988

theorem probability_at_least_one_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = red_balls + white_balls →
  total_balls = 4 →
  red_balls = 2 →
  white_balls = 2 →
  drawn_balls = 2 →
  (Nat.choose total_balls drawn_balls - Nat.choose white_balls drawn_balls) / 
    Nat.choose total_balls drawn_balls = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l1609_160988


namespace NUMINAMATH_CALUDE_parabola_shift_l1609_160964

/-- Given a parabola y = 5x², shifting it 2 units left and 3 units up results in y = 5(x + 2)² + 3 -/
theorem parabola_shift (x y : ℝ) :
  (y = 5 * x^2) →
  (∃ y_shifted : ℝ, y_shifted = 5 * (x + 2)^2 + 3 ∧
    y_shifted = y + 3 ∧
    ∀ x_orig : ℝ, y = 5 * x_orig^2 → x = x_orig - 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1609_160964


namespace NUMINAMATH_CALUDE_max_andy_consumption_l1609_160963

def total_cookies : ℕ := 36

def cookie_distribution (andy alexa ann : ℕ) : Prop :=
  ∃ k : ℕ+, alexa = k * andy ∧ ann = 2 * andy ∧ andy + alexa + ann = total_cookies

def max_andy_cookies : ℕ := 9

theorem max_andy_consumption :
  ∀ andy alexa ann : ℕ,
    cookie_distribution andy alexa ann →
    andy ≤ max_andy_cookies :=
by sorry

end NUMINAMATH_CALUDE_max_andy_consumption_l1609_160963


namespace NUMINAMATH_CALUDE_tony_running_speed_l1609_160983

/-- The distance to the store in miles -/
def distance : ℝ := 4

/-- Tony's walking speed in miles per hour -/
def walking_speed : ℝ := 2

/-- The average time Tony spends to get to the store in minutes -/
def average_time : ℝ := 56

/-- Tony's running speed in miles per hour -/
def running_speed : ℝ := 10

theorem tony_running_speed :
  let time_walking := (distance / walking_speed) * 60
  let time_running := (distance / running_speed) * 60
  (time_walking + 2 * time_running) / 3 = average_time :=
by sorry

end NUMINAMATH_CALUDE_tony_running_speed_l1609_160983


namespace NUMINAMATH_CALUDE_triangle_properties_l1609_160914

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  2 * Real.cos (t.A + t.B) = -1

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = Real.pi / 3 ∧
  t.c = Real.sqrt 6 ∧
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1609_160914


namespace NUMINAMATH_CALUDE_strawberries_in_buckets_l1609_160904

theorem strawberries_in_buckets (total_strawberries : ℕ) (num_buckets : ℕ) (removed_per_bucket : ℕ) :
  total_strawberries = 300 →
  num_buckets = 5 →
  removed_per_bucket = 20 →
  (total_strawberries / num_buckets) - removed_per_bucket = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberries_in_buckets_l1609_160904


namespace NUMINAMATH_CALUDE_sum_lent_proof_l1609_160928

/-- Proves that given a sum P lent at 4% per annum simple interest,
    if the interest after 4 years is Rs. 1260 less than P, then P = 1500. -/
theorem sum_lent_proof (P : ℝ) : 
  (P * (4 / 100) * 4 = P - 1260) → P = 1500 := by sorry

end NUMINAMATH_CALUDE_sum_lent_proof_l1609_160928


namespace NUMINAMATH_CALUDE_correct_hourly_wage_l1609_160994

/-- The hourly wage for a manufacturing plant worker --/
def hourly_wage : ℝ :=
  12.50

/-- The piece rate per widget --/
def piece_rate : ℝ :=
  0.16

/-- The number of widgets produced in a week --/
def widgets_per_week : ℕ :=
  1000

/-- The number of hours worked in a week --/
def hours_per_week : ℕ :=
  40

/-- The total earnings for a week --/
def total_earnings : ℝ :=
  660

theorem correct_hourly_wage :
  hourly_wage * hours_per_week + piece_rate * widgets_per_week = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_correct_hourly_wage_l1609_160994


namespace NUMINAMATH_CALUDE_mirror_solution_l1609_160938

/-- Represents the number of reflections seen in the house of mirrors --/
structure Reflections where
  sarah_tall : ℕ
  sarah_wide : ℕ
  ellie_tall : ℕ
  ellie_wide : ℕ
  tall_visits : ℕ
  wide_visits : ℕ
  total : ℕ

/-- The house of mirrors problem --/
def mirror_problem : Reflections where
  sarah_tall := 10
  sarah_wide := 5
  ellie_tall := 6  -- This is what we want to prove
  ellie_wide := 3
  tall_visits := 3
  wide_visits := 5
  total := 88

/-- Theorem stating that the given configuration solves the mirror problem --/
theorem mirror_solution :
  let r := mirror_problem
  r.sarah_tall * r.tall_visits + r.sarah_wide * r.wide_visits +
  r.ellie_tall * r.tall_visits + r.ellie_wide * r.wide_visits = r.total :=
by sorry

end NUMINAMATH_CALUDE_mirror_solution_l1609_160938


namespace NUMINAMATH_CALUDE_square_tiles_count_l1609_160987

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) (square_tiles : ℕ) (pentagonal_tiles : ℕ) :
  total_tiles = 30 →
  total_edges = 110 →
  total_tiles = square_tiles + pentagonal_tiles →
  4 * square_tiles + 5 * pentagonal_tiles = total_edges →
  square_tiles = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l1609_160987


namespace NUMINAMATH_CALUDE_max_intersections_four_circles_l1609_160957

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of intersections between a line and a circle --/
def intersectionCount (l : Line) (c : Circle) : ℕ := sorry

/-- Checks if four circles are coplanar --/
def areCoplanar (c1 c2 c3 c4 : Circle) : Prop := sorry

/-- Theorem: The maximum number of intersections between a line and four coplanar circles is 8 --/
theorem max_intersections_four_circles (c1 c2 c3 c4 : Circle) (l : Line) :
  areCoplanar c1 c2 c3 c4 →
  (intersectionCount l c1 + intersectionCount l c2 + intersectionCount l c3 + intersectionCount l c4) ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_four_circles_l1609_160957


namespace NUMINAMATH_CALUDE_problem_grid_squares_l1609_160934

/-- Represents a grid with vertical and horizontal lines -/
structure Grid :=
  (vertical_lines : ℕ)
  (horizontal_lines : ℕ)
  (column_widths : List ℕ)
  (row_heights : List ℕ)

/-- Counts the number of squares that can be traced in a given grid -/
def count_squares (g : Grid) : ℕ := sorry

/-- The specific grid described in the problem -/
def problem_grid : Grid :=
  { vertical_lines := 5
  , horizontal_lines := 6
  , column_widths := [1, 2, 1, 1]
  , row_heights := [2, 1, 1, 1] }

/-- Theorem stating that the number of squares in the problem grid is 23 -/
theorem problem_grid_squares :
  count_squares problem_grid = 23 := by sorry

end NUMINAMATH_CALUDE_problem_grid_squares_l1609_160934


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1609_160915

/-- Given a run of 12 miles in 90 minutes, prove that the average speed is 8 miles per hour -/
theorem average_speed_calculation (distance : ℝ) (time_minutes : ℝ) (h1 : distance = 12) (h2 : time_minutes = 90) :
  distance / (time_minutes / 60) = 8 := by
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1609_160915


namespace NUMINAMATH_CALUDE_parabolas_coincide_l1609_160945

/-- Represents a parabola with leading coefficient 1 -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  b : ℝ

/-- Returns the length of the segment intercepted by a line on a parabola -/
noncomputable def interceptLength (para : Parabola) (l : Line) : ℝ :=
  Real.sqrt ((para.p - l.k)^2 - 4*(para.q - l.b))

/-- Two lines are non-parallel if their slopes are different -/
def nonParallel (l₁ l₂ : Line) : Prop :=
  l₁.k ≠ l₂.k

theorem parabolas_coincide
  (Γ₁ Γ₂ : Parabola)
  (l₁ l₂ : Line)
  (h_nonparallel : nonParallel l₁ l₂)
  (h_equal_length₁ : interceptLength Γ₁ l₁ = interceptLength Γ₂ l₁)
  (h_equal_length₂ : interceptLength Γ₁ l₂ = interceptLength Γ₂ l₂) :
  Γ₁ = Γ₂ := by
  sorry

end NUMINAMATH_CALUDE_parabolas_coincide_l1609_160945


namespace NUMINAMATH_CALUDE_only_component_life_uses_experiments_l1609_160984

/-- Represents a method of data collection --/
inductive DataCollectionMethod
  | Observation
  | Experiment
  | Investigation

/-- Represents the different scenarios --/
inductive Scenario
  | TemperatureMeasurement
  | ComponentLifeDetermination
  | TVRatings
  | CounterfeitDetection

/-- Maps each scenario to its typical data collection method --/
def typicalMethod (s : Scenario) : DataCollectionMethod :=
  match s with
  | Scenario.TemperatureMeasurement => DataCollectionMethod.Observation
  | Scenario.ComponentLifeDetermination => DataCollectionMethod.Experiment
  | Scenario.TVRatings => DataCollectionMethod.Investigation
  | Scenario.CounterfeitDetection => DataCollectionMethod.Investigation

theorem only_component_life_uses_experiments :
  ∀ s : Scenario, typicalMethod s = DataCollectionMethod.Experiment ↔ s = Scenario.ComponentLifeDetermination :=
by sorry


end NUMINAMATH_CALUDE_only_component_life_uses_experiments_l1609_160984


namespace NUMINAMATH_CALUDE_even_increasing_property_l1609_160989

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function is increasing on an interval if f(x) ≤ f(y) whenever x ≤ y in that interval -/
def IsIncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_increasing_property (f : ℝ → ℝ) (h_even : IsEven f) 
    (h_incr : IsIncreasingOn f (Set.Iic 0)) :
    ∀ a : ℝ, f (a^2) > f (a^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_property_l1609_160989


namespace NUMINAMATH_CALUDE_prob_all_cats_before_lunch_l1609_160941

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of animals -/
def totalAnimals : ℕ := 7

/-- The number of cats -/
def numCats : ℕ := 2

/-- The number of dogs -/
def numDogs : ℕ := 5

/-- The number of animals to be groomed before lunch -/
def numGroomed : ℕ := 4

/-- The probability of grooming all cats before lunch -/
def probAllCats : ℚ := (choose numDogs (numGroomed - numCats)) / (choose totalAnimals numGroomed)

theorem prob_all_cats_before_lunch : probAllCats = 2/7 := by sorry

end NUMINAMATH_CALUDE_prob_all_cats_before_lunch_l1609_160941


namespace NUMINAMATH_CALUDE_factor_quadratic_l1609_160902

theorem factor_quadratic (x t : ℝ) : 
  (x - t) ∣ (10 * x^2 + 23 * x - 7) ↔ 
  t = (-23 + Real.sqrt 809) / 20 ∨ t = (-23 - Real.sqrt 809) / 20 := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l1609_160902


namespace NUMINAMATH_CALUDE_perfect_matching_exists_l1609_160979

/-- Represents a polygon with unit area -/
structure UnitPolygon where
  -- Add necessary fields here
  area : ℝ
  area_eq_one : area = 1

/-- Represents a square sheet of side length 2019 cut into 2019² unit polygons -/
structure Sheet where
  side_length : ℕ
  side_length_eq_2019 : side_length = 2019
  polygons : Finset UnitPolygon
  polygon_count : polygons.card = side_length * side_length

/-- Represents the intersection between two polygons from different sheets -/
def intersects (p1 p2 : UnitPolygon) : Prop :=
  sorry

/-- The main theorem -/
theorem perfect_matching_exists (sheet1 sheet2 : Sheet) : ∃ (matching : Finset (UnitPolygon × UnitPolygon)), 
  matching.card = 2019 * 2019 ∧ 
  (∀ (p1 p2 : UnitPolygon), (p1, p2) ∈ matching → p1 ∈ sheet1.polygons ∧ p2 ∈ sheet2.polygons ∧ intersects p1 p2) ∧
  (∀ p1 ∈ sheet1.polygons, ∃! p2, (p1, p2) ∈ matching) ∧
  (∀ p2 ∈ sheet2.polygons, ∃! p1, (p1, p2) ∈ matching) :=
sorry

end NUMINAMATH_CALUDE_perfect_matching_exists_l1609_160979


namespace NUMINAMATH_CALUDE_g_value_l1609_160971

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^4 - x^2 - 3
axiom sum_eq : ∀ x, f x + g x = 3 * x^2 - 1

-- State the theorem
theorem g_value : ∀ x, g x = -x^4 + 4 * x^2 + 2 := by sorry

end NUMINAMATH_CALUDE_g_value_l1609_160971


namespace NUMINAMATH_CALUDE_coin_distribution_l1609_160906

theorem coin_distribution (total : ℕ) (ways : ℕ) 
  (h_total : total = 1512)
  (h_ways : ways = 1512)
  (h_denominations : ∃ (c₂ c₅ c₁₀ c₂₀ c₅₀ c₁₀₀ c₂₀₀ : ℕ),
    (c₂ ≥ 1 ∧ c₅ ≥ 1 ∧ c₁₀ ≥ 1 ∧ c₂₀ ≥ 1 ∧ c₅₀ ≥ 1 ∧ c₁₀₀ ≥ 1 ∧ c₂₀₀ ≥ 1) ∧
    (2 * c₂ + 5 * c₅ + 10 * c₁₀ + 20 * c₂₀ + 50 * c₅₀ + 100 * c₁₀₀ + 200 * c₂₀₀ = total) ∧
    ((c₂ + 1) * (c₅ + 1) * (c₁₀ + 1) * (c₂₀ + 1) * (c₅₀ + 1) * (c₁₀₀ + 1) * (c₂₀₀ + 1) = ways)) :
  ∃! (c₂ c₅ c₁₀ c₂₀ c₅₀ c₁₀₀ c₂₀₀ : ℕ),
    (c₂ = 1 ∧ c₅ = 2 ∧ c₁₀ = 1 ∧ c₂₀ = 2 ∧ c₅₀ = 1 ∧ c₁₀₀ = 2 ∧ c₂₀₀ = 6) ∧
    (2 * c₂ + 5 * c₅ + 10 * c₁₀ + 20 * c₂₀ + 50 * c₅₀ + 100 * c₁₀₀ + 200 * c₂₀₀ = total) ∧
    ((c₂ + 1) * (c₅ + 1) * (c₁₀ + 1) * (c₂₀ + 1) * (c₅₀ + 1) * (c₁₀₀ + 1) * (c₂₀₀ + 1) = ways) :=
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l1609_160906


namespace NUMINAMATH_CALUDE_boyds_male_friends_percentage_l1609_160910

theorem boyds_male_friends_percentage 
  (julian_total : ℕ) 
  (julian_boys_percent : ℚ) 
  (boyd_total : ℕ) 
  (boyd_girls_multiplier : ℕ) : 
  julian_total = 80 → 
  julian_boys_percent = 60 / 100 → 
  boyd_total = 100 → 
  boyd_girls_multiplier = 2 → 
  (boyd_total - boyd_girls_multiplier * (julian_total * (1 - julian_boys_percent))) / boyd_total = 36 / 100 := by
  sorry

end NUMINAMATH_CALUDE_boyds_male_friends_percentage_l1609_160910


namespace NUMINAMATH_CALUDE_circle_equation_l1609_160929

/-- A circle C with given properties -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (center_in_first_quadrant : center.1 > 0 ∧ center.2 > 0)
  (tangent_to_line : |4 * center.1 - 3 * center.2| = 5 * radius)
  (tangent_to_x_axis : center.2 = radius)
  (radius_is_one : radius = 1)

/-- The standard equation of a circle -/
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem: The standard equation of the circle C is (x-2)^2 + (y-1)^2 = 1 -/
theorem circle_equation (c : Circle) :
  ∀ x y : ℝ, standard_equation c x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1609_160929


namespace NUMINAMATH_CALUDE_length_OB_is_sqrt_13_l1609_160944

-- Define the point A
def A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define the projection B of A onto the yOz plane
def B : ℝ × ℝ × ℝ := (0, A.2.1, A.2.2)

-- Define the origin O
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Theorem to prove
theorem length_OB_is_sqrt_13 : 
  Real.sqrt ((B.1 - O.1)^2 + (B.2.1 - O.2.1)^2 + (B.2.2 - O.2.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_length_OB_is_sqrt_13_l1609_160944


namespace NUMINAMATH_CALUDE_string_measurement_l1609_160913

theorem string_measurement (string_length : Real) (cut_fraction : Real) : 
  string_length = 2/3 → 
  cut_fraction = 1/4 → 
  (1 - cut_fraction) * string_length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_string_measurement_l1609_160913


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1609_160998

theorem ratio_of_sum_and_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1609_160998


namespace NUMINAMATH_CALUDE_overlapping_triangles_sum_l1609_160952

/-- Represents a triangle with angles a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180

/-- Configuration of two pairs of overlapping triangles -/
structure OverlappingTriangles where
  t1 : Triangle
  t2 : Triangle

/-- The sum of all distinct angles in a configuration of two pairs of overlapping triangles is 360° -/
theorem overlapping_triangles_sum (ot : OverlappingTriangles) : 
  ot.t1.a + ot.t1.b + ot.t1.c + ot.t2.a + ot.t2.b + ot.t2.c = 360 := by
  sorry


end NUMINAMATH_CALUDE_overlapping_triangles_sum_l1609_160952


namespace NUMINAMATH_CALUDE_daves_apps_l1609_160926

theorem daves_apps (initial_files : ℕ) (final_apps : ℕ) (final_files : ℕ) (deleted_apps : ℕ) :
  initial_files = 77 →
  final_apps = 5 →
  final_files = 23 →
  deleted_apps = 11 →
  final_apps + deleted_apps = 16 := by
  sorry

end NUMINAMATH_CALUDE_daves_apps_l1609_160926


namespace NUMINAMATH_CALUDE_total_questions_is_100_l1609_160923

/-- Represents the scoring system and test results for a student. -/
structure TestResult where
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  total_questions : ℕ

/-- Defines the properties of a valid test result based on the given conditions. -/
def is_valid_test_result (tr : TestResult) : Prop :=
  tr.score = tr.correct_responses - 2 * tr.incorrect_responses ∧
  tr.total_questions = tr.correct_responses + tr.incorrect_responses

/-- Theorem stating that given the conditions, the total number of questions is 100. -/
theorem total_questions_is_100 (tr : TestResult) 
  (h1 : is_valid_test_result tr) 
  (h2 : tr.score = 64) 
  (h3 : tr.correct_responses = 88) : 
  tr.total_questions = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_is_100_l1609_160923


namespace NUMINAMATH_CALUDE_brenda_stones_count_brenda_bought_36_stones_l1609_160933

theorem brenda_stones_count : ℕ → ℕ → ℕ
  | num_bracelets, stones_per_bracelet => 
    num_bracelets * stones_per_bracelet

theorem brenda_bought_36_stones 
  (num_bracelets : ℕ) 
  (stones_per_bracelet : ℕ) 
  (h1 : num_bracelets = 3) 
  (h2 : stones_per_bracelet = 12) : 
  brenda_stones_count num_bracelets stones_per_bracelet = 36 := by
  sorry

end NUMINAMATH_CALUDE_brenda_stones_count_brenda_bought_36_stones_l1609_160933


namespace NUMINAMATH_CALUDE_megan_earnings_after_discount_l1609_160996

/-- Calculates Megan's earnings from selling necklaces at a garage sale with a discount --/
theorem megan_earnings_after_discount :
  let bead_necklaces : ℕ := 7
  let bead_price : ℕ := 5
  let gem_necklaces : ℕ := 3
  let gem_price : ℕ := 15
  let discount_rate : ℚ := 1/5  -- 20% as a rational number
  
  let total_before_discount := bead_necklaces * bead_price + gem_necklaces * gem_price
  let discount_amount := (total_before_discount : ℚ) * discount_rate
  let earnings_after_discount := (total_before_discount : ℚ) - discount_amount
  
  earnings_after_discount = 64 := by sorry

end NUMINAMATH_CALUDE_megan_earnings_after_discount_l1609_160996


namespace NUMINAMATH_CALUDE_expense_difference_l1609_160917

def road_trip_expenses (alex_paid bob_paid carol_paid : ℚ) 
                       (a b : ℚ) : Prop :=
  let total := alex_paid + bob_paid + carol_paid
  let share := total / 3
  let alex_owes := share - alex_paid
  let bob_receives := bob_paid - share
  let carol_receives := carol_paid - share
  (alex_owes = a) ∧ (bob_receives + b = carol_receives) ∧ (a - b = 30)

theorem expense_difference :
  road_trip_expenses 120 150 210 40 10 := by sorry

end NUMINAMATH_CALUDE_expense_difference_l1609_160917


namespace NUMINAMATH_CALUDE_cubeRoot_of_negative_27_l1609_160912

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_27 : cubeRoot (-27) = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubeRoot_of_negative_27_l1609_160912


namespace NUMINAMATH_CALUDE_job_completion_times_l1609_160956

/-- Represents the productivity of a worker -/
structure Productivity where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a worker -/
structure Worker where
  productivity : Productivity

/-- Represents a job with three workers -/
structure Job where
  worker1 : Worker
  worker2 : Worker
  worker3 : Worker
  total_work : ℝ
  total_work_pos : total_work > 0
  third_worker_productivity : worker3.productivity.rate = (worker1.productivity.rate + worker2.productivity.rate) / 2
  work_condition : 48 * worker3.productivity.rate + 10 * worker1.productivity.rate = 
                   48 * worker3.productivity.rate + 15 * worker2.productivity.rate

/-- The theorem to be proved -/
theorem job_completion_times (job : Job) :
  let time1 := job.total_work / job.worker1.productivity.rate
  let time2 := job.total_work / job.worker2.productivity.rate
  let time3 := job.total_work / job.worker3.productivity.rate
  (time1 = 50 ∧ time2 = 75 ∧ time3 = 60) := by
  sorry

end NUMINAMATH_CALUDE_job_completion_times_l1609_160956


namespace NUMINAMATH_CALUDE_n_gon_division_l1609_160974

/-- The number of parts into which the diagonals of an n-gon divide it, 
    given that no three diagonals intersect at one point. -/
def numberOfParts (n : ℕ) : ℚ :=
  1 + (n * (n - 3) / 2) + (n * (n - 1) * (n - 2) * (n - 3) / 24)

/-- Theorem stating that the number of parts into which the diagonals of an n-gon divide it,
    given that no three diagonals intersect at one point, is equal to the formula. -/
theorem n_gon_division (n : ℕ) (h : n ≥ 3) : 
  numberOfParts n = 1 + (n * (n - 3) / 2) + (n * (n - 1) * (n - 2) * (n - 3) / 24) := by
  sorry

end NUMINAMATH_CALUDE_n_gon_division_l1609_160974


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l1609_160960

theorem log_2_base_10_bounds : ∃ (log_2_base_10 : ℝ),
  (10 : ℝ) ^ 3 = 1000 ∧
  (10 : ℝ) ^ 4 = 10000 ∧
  (2 : ℝ) ^ 10 = 1024 ∧
  (2 : ℝ) ^ 11 = 2048 ∧
  (2 : ℝ) ^ 12 = 4096 ∧
  (2 : ℝ) ^ 13 = 8192 ∧
  (∀ x > 0, (10 : ℝ) ^ (log_2_base_10 * Real.log x) = x) ∧
  3 / 10 < log_2_base_10 ∧
  log_2_base_10 < 4 / 13 :=
by sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l1609_160960


namespace NUMINAMATH_CALUDE_right_triangle_with_condition_l1609_160985

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def satisfies_condition (a b c : ℕ) : Prop :=
  a + b = c + 6

theorem right_triangle_with_condition :
  ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    a ≤ b →
    is_right_triangle a b c →
    satisfies_condition a b c →
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
by
  sorry

#check right_triangle_with_condition

end NUMINAMATH_CALUDE_right_triangle_with_condition_l1609_160985


namespace NUMINAMATH_CALUDE_moses_extra_amount_l1609_160972

def total_amount : ℝ := 50
def moses_percentage : ℝ := 0.4

theorem moses_extra_amount :
  let moses_share := moses_percentage * total_amount
  let remainder := total_amount - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by sorry

end NUMINAMATH_CALUDE_moses_extra_amount_l1609_160972


namespace NUMINAMATH_CALUDE_smallest_sum_X_plus_c_l1609_160980

theorem smallest_sum_X_plus_c : ∀ (X c : ℕ),
  X < 5 → 
  X > 0 →
  c > 6 →
  (31 * X = 4 * c + 4) →
  ∀ (Y d : ℕ), Y < 5 → Y > 0 → d > 6 → (31 * Y = 4 * d + 4) →
  X + c ≤ Y + d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_X_plus_c_l1609_160980


namespace NUMINAMATH_CALUDE_point_in_region_range_l1609_160940

theorem point_in_region_range (a : ℝ) : 
  (2 * a + 3 < 3) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_range_l1609_160940


namespace NUMINAMATH_CALUDE_angle_CAG_measure_l1609_160961

-- Define the points
variable (A B C F G : ℝ × ℝ)

-- Define the properties of the configuration
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

def is_rectangle (B C F G : ℝ × ℝ) : Prop := sorry

def shared_side (A B C F G : ℝ × ℝ) : Prop := sorry

def longer_side (B C F G : ℝ × ℝ) : Prop := sorry

-- Define the angle measure function
def angle_measure (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_CAG_measure 
  (h1 : is_equilateral A B C)
  (h2 : is_rectangle B C F G)
  (h3 : shared_side A B C F G)
  (h4 : longer_side B C F G) :
  angle_measure C A G = 15 := by sorry

end NUMINAMATH_CALUDE_angle_CAG_measure_l1609_160961


namespace NUMINAMATH_CALUDE_only_153_and_407_are_cube_sum_numbers_l1609_160922

-- Define a function to calculate the sum of cubes of digits
def sumOfCubesOfDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds^3 + tens^3 + ones^3

-- Define the property for a number to be a cube sum number
def isCubeSumNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n = sumOfCubesOfDigits n

-- Theorem statement
theorem only_153_and_407_are_cube_sum_numbers :
  ∀ n : ℕ, isCubeSumNumber n ↔ n = 153 ∨ n = 407 := by sorry

end NUMINAMATH_CALUDE_only_153_and_407_are_cube_sum_numbers_l1609_160922


namespace NUMINAMATH_CALUDE_problem_statement_l1609_160942

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem statement
theorem problem_statement : (p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1609_160942


namespace NUMINAMATH_CALUDE_notebooks_bought_is_four_l1609_160966

/-- The cost of one pencil -/
def pencil_cost : ℚ := sorry

/-- The cost of one notebook -/
def notebook_cost : ℚ := sorry

/-- The number of notebooks bought in the second case -/
def notebooks_bought : ℕ := sorry

/-- The cost of 8 dozen pencils and 2 dozen notebooks is 520 rupees -/
axiom eq1 : 96 * pencil_cost + 24 * notebook_cost = 520

/-- The cost of 3 pencils and some number of notebooks is 60 rupees -/
axiom eq2 : 3 * pencil_cost + notebooks_bought * notebook_cost = 60

/-- The sum of the cost of 1 pencil and 1 notebook is 15.512820512820513 rupees -/
axiom eq3 : pencil_cost + notebook_cost = 15.512820512820513

theorem notebooks_bought_is_four : notebooks_bought = 4 := by sorry

end NUMINAMATH_CALUDE_notebooks_bought_is_four_l1609_160966


namespace NUMINAMATH_CALUDE_star_example_l1609_160965

/-- The star operation for fractions -/
def star (m n p q : ℚ) : ℚ := (m + 1) * (p - 1) * ((q + 1) / (n - 1))

/-- Theorem stating that 5/7 ★ 9/4 = 40 -/
theorem star_example : star 5 7 9 4 = 40 := by sorry

end NUMINAMATH_CALUDE_star_example_l1609_160965


namespace NUMINAMATH_CALUDE_area_ABC_is_72_l1609_160975

-- Define the points X, Y, and Z
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

-- Define the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem area_ABC_is_72 :
  ∃ (A B C : ℝ × ℝ),
    triangleArea X Y Z = 0.1111111111111111 * triangleArea A B C ∧
    triangleArea A B C = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_ABC_is_72_l1609_160975


namespace NUMINAMATH_CALUDE_peter_class_size_l1609_160947

/-- The number of students in Peter's class -/
def students_in_class : ℕ := 11

/-- The number of hands in the class, excluding Peter's -/
def hands_without_peter : ℕ := 20

/-- The number of hands each student has -/
def hands_per_student : ℕ := 2

/-- Theorem: The number of students in Peter's class is 11 -/
theorem peter_class_size :
  students_in_class = hands_without_peter / hands_per_student + 1 :=
by sorry

end NUMINAMATH_CALUDE_peter_class_size_l1609_160947


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1609_160986

theorem largest_divisor_of_five_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℤ), (k * (k+1) * (k+2) * (k+3) * (k+4)) % n = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (l : ℤ), (l * (l+1) * (l+2) * (l+3) * (l+4)) % m ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1609_160986


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l1609_160937

theorem cats_sold_during_sale 
  (initial_siamese : ℕ) 
  (initial_house : ℕ) 
  (cats_left : ℕ) 
  (h1 : initial_siamese = 13)
  (h2 : initial_house = 5)
  (h3 : cats_left = 8) :
  initial_siamese + initial_house - cats_left = 10 := by
sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l1609_160937


namespace NUMINAMATH_CALUDE_twin_prime_square_diff_sum_l1609_160948

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p = q + 2 ∨ q = p + 2)

theorem twin_prime_square_diff_sum (p q : ℕ) : 
  is_twin_prime p q → 
  (is_prime (p^2 - p*q + q^2) ↔ ((p = 5 ∧ q = 3) ∨ (p = 3 ∧ q = 5))) :=
sorry

end NUMINAMATH_CALUDE_twin_prime_square_diff_sum_l1609_160948


namespace NUMINAMATH_CALUDE_range_of_f_l1609_160973

def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = y}
  S = Set.Icc (-2 : ℝ) 7 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1609_160973


namespace NUMINAMATH_CALUDE_test_points_calculation_l1609_160967

theorem test_points_calculation (total_problems : ℕ) (computation_problems : ℕ) 
  (computation_points : ℕ) (word_points : ℕ) :
  total_problems = 30 →
  computation_problems = 20 →
  computation_points = 3 →
  word_points = 5 →
  (computation_problems * computation_points) + 
  ((total_problems - computation_problems) * word_points) = 110 := by
sorry

end NUMINAMATH_CALUDE_test_points_calculation_l1609_160967


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1609_160909

theorem solution_set_of_inequality (x : ℝ) :
  x^2 < 2*x ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1609_160909


namespace NUMINAMATH_CALUDE_sweater_shirt_price_difference_l1609_160946

theorem sweater_shirt_price_difference : 
  let shirt_total : ℕ := 360
  let shirt_count : ℕ := 20
  let sweater_total : ℕ := 900
  let sweater_count : ℕ := 45
  let shirt_avg : ℚ := shirt_total / shirt_count
  let sweater_avg : ℚ := sweater_total / sweater_count
  sweater_avg - shirt_avg = 2 := by
sorry

end NUMINAMATH_CALUDE_sweater_shirt_price_difference_l1609_160946


namespace NUMINAMATH_CALUDE_price_fluctuation_l1609_160950

theorem price_fluctuation (p : ℝ) (original_price : ℝ) : 
  (original_price * (1 + p / 100) * (1 - p / 100) = 1) →
  (original_price = 10000 / (10000 - p^2)) :=
by sorry

end NUMINAMATH_CALUDE_price_fluctuation_l1609_160950


namespace NUMINAMATH_CALUDE_dog_does_not_catch_hare_l1609_160959

/-- Represents the chase scenario between a dog and a hare -/
structure ChaseScenario where
  dog_speed : ℝ
  hare_speed : ℝ
  initial_distance : ℝ
  bushes_distance : ℝ

/-- Determines if the dog catches the hare before it reaches the bushes -/
def dog_catches_hare (scenario : ChaseScenario) : Prop :=
  let relative_speed := scenario.dog_speed - scenario.hare_speed
  let catch_time := scenario.initial_distance / relative_speed
  let hare_distance := scenario.hare_speed * catch_time
  hare_distance < scenario.bushes_distance

/-- The theorem stating that the dog does not catch the hare -/
theorem dog_does_not_catch_hare (scenario : ChaseScenario)
  (h1 : scenario.dog_speed = 17)
  (h2 : scenario.hare_speed = 14)
  (h3 : scenario.initial_distance = 150)
  (h4 : scenario.bushes_distance = 520) :
  ¬(dog_catches_hare scenario) := by
  sorry

#check dog_does_not_catch_hare

end NUMINAMATH_CALUDE_dog_does_not_catch_hare_l1609_160959


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l1609_160908

/-- Represents a parabola of the form x = 3y^2 - 9y + 5 -/
def Parabola := { p : ℝ × ℝ | p.1 = 3 * p.2^2 - 9 * p.2 + 5 }

/-- The x-coordinate of the x-intercept -/
def a : ℝ := 5

/-- The y-coordinates of the y-intercepts -/
def b : ℝ := sorry
def c : ℝ := sorry

theorem parabola_intercepts_sum : a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l1609_160908


namespace NUMINAMATH_CALUDE_second_day_hours_proof_l1609_160924

/-- Represents the number of hours worked on the second day -/
def hours_second_day : ℕ := 8

/-- The hourly rate paid to each worker -/
def hourly_rate : ℕ := 10

/-- The total payment received by both workers -/
def total_payment : ℕ := 660

/-- The number of hours worked on the first day -/
def hours_first_day : ℕ := 10

/-- The number of hours worked on the third day -/
def hours_third_day : ℕ := 15

/-- The number of workers -/
def num_workers : ℕ := 2

theorem second_day_hours_proof :
  hours_second_day * num_workers * hourly_rate +
  hours_first_day * num_workers * hourly_rate +
  hours_third_day * num_workers * hourly_rate = total_payment :=
by sorry

end NUMINAMATH_CALUDE_second_day_hours_proof_l1609_160924


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1609_160982

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1609_160982


namespace NUMINAMATH_CALUDE_distance_calculation_l1609_160999

/-- The distance between Cara's and Don's homes -/
def distance_between_homes : ℝ := 45

/-- Cara's walking speed in km/h -/
def cara_speed : ℝ := 6

/-- Don's walking speed in km/h -/
def don_speed : ℝ := 5

/-- The distance Cara walks before meeting Don in km -/
def cara_distance : ℝ := 30

/-- The time difference between Cara's and Don's start in hours -/
def time_difference : ℝ := 2

theorem distance_calculation :
  distance_between_homes = cara_distance + don_speed * (cara_distance / cara_speed - time_difference) :=
sorry

end NUMINAMATH_CALUDE_distance_calculation_l1609_160999


namespace NUMINAMATH_CALUDE_seventh_term_equals_33_l1609_160978

/-- An arithmetic sequence with 15 terms, first term 3, and last term 72 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  3 + (n - 1) * ((72 - 3) / 14)

/-- The 7th term of the arithmetic sequence -/
def seventh_term : ℚ := arithmetic_sequence 7

theorem seventh_term_equals_33 : ⌊seventh_term⌋ = 33 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_equals_33_l1609_160978


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1609_160930

theorem inequality_equivalence (x : ℝ) : x - 1 > 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1609_160930


namespace NUMINAMATH_CALUDE_hawks_score_l1609_160990

theorem hawks_score (total_score : ℕ) (eagles_margin : ℕ) (eagles_three_pointers : ℕ) 
  (h1 : total_score = 82)
  (h2 : eagles_margin = 18)
  (h3 : eagles_three_pointers = 6) : 
  total_score / 2 - eagles_margin / 2 = 32 := by
  sorry

#check hawks_score

end NUMINAMATH_CALUDE_hawks_score_l1609_160990


namespace NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l1609_160932

/-- The volume of a solid obtained by rotating an equilateral triangle -/
theorem equilateral_triangle_rotation_volume (a : ℝ) (ha : a > 0) :
  let h := a * Real.sqrt 3 / 2
  let V := 2 * π * (a / 2)^2 * h
  V = π * a^3 * Real.sqrt 3 / 4 := by
  sorry

#check equilateral_triangle_rotation_volume

end NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l1609_160932


namespace NUMINAMATH_CALUDE_two_books_adjacent_probability_l1609_160911

theorem two_books_adjacent_probability (n : ℕ) (h : n = 10) :
  let total_arrangements := n.factorial
  let favorable_arrangements := ((n - 1).factorial * 2)
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_two_books_adjacent_probability_l1609_160911


namespace NUMINAMATH_CALUDE_smallest_an_l1609_160901

theorem smallest_an (n : ℕ+) (x : ℝ) :
  (((x^(2^(n.val+1)) + 1) / 2) ^ (1 / (2^n.val))) ≤ 2^(n.val-1) * (x-1)^2 + x :=
sorry

end NUMINAMATH_CALUDE_smallest_an_l1609_160901


namespace NUMINAMATH_CALUDE_water_drip_theorem_l1609_160935

/-- Represents the water dripping scenario -/
structure WaterDrip where
  drops_per_second : ℝ
  ml_per_drop : ℝ

/-- Calculates the volume of water dripped given time in minutes -/
def volume_dripped (w : WaterDrip) (minutes : ℝ) : ℝ :=
  w.drops_per_second * w.ml_per_drop * 60 * minutes

/-- The main theorem about the water dripping scenario -/
theorem water_drip_theorem (w : WaterDrip) 
    (h1 : w.drops_per_second = 2)
    (h2 : w.ml_per_drop = 0.05) : 
  (∀ x, volume_dripped w x = 6 * x) ∧ 
  (volume_dripped w 50 = 300) :=
sorry

#check water_drip_theorem

end NUMINAMATH_CALUDE_water_drip_theorem_l1609_160935


namespace NUMINAMATH_CALUDE_pythagorean_triple_value_l1609_160949

theorem pythagorean_triple_value (a : ℝ) : 
  (3 : ℝ)^2 + a^2 = 5^2 → a = 4 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_value_l1609_160949
