import Mathlib

namespace NUMINAMATH_CALUDE_sarahs_wallet_l105_10541

theorem sarahs_wallet (total_amount : ℕ) (total_bills : ℕ) (five_dollar_count : ℕ) (ten_dollar_count : ℕ) : 
  total_amount = 100 →
  total_bills = 15 →
  five_dollar_count + ten_dollar_count = total_bills →
  5 * five_dollar_count + 10 * ten_dollar_count = total_amount →
  five_dollar_count = 10 := by
sorry

end NUMINAMATH_CALUDE_sarahs_wallet_l105_10541


namespace NUMINAMATH_CALUDE_unique_solution_l105_10526

/-- Represents a six-digit number with distinct digits -/
structure SixDigitNumber where
  digits : Fin 6 → Fin 10
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- The equation 6 × AOBMEP = 7 × MEPAOB -/
def EquationHolds (n : SixDigitNumber) : Prop :=
  6 * (100000 * n.digits 0 + 10000 * n.digits 1 + 1000 * n.digits 2 +
       100 * n.digits 3 + 10 * n.digits 4 + n.digits 5) =
  7 * (100000 * n.digits 3 + 10000 * n.digits 4 + 1000 * n.digits 5 +
       100 * n.digits 0 + 10 * n.digits 1 + n.digits 2)

/-- The unique solution to the equation -/
def Solution : SixDigitNumber where
  digits := fun i => match i with
    | 0 => 5  -- A
    | 1 => 3  -- O
    | 2 => 8  -- B
    | 3 => 4  -- M
    | 4 => 6  -- E
    | 5 => 1  -- P
  distinct := by sorry

theorem unique_solution :
  ∀ n : SixDigitNumber, EquationHolds n ↔ n = Solution := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l105_10526


namespace NUMINAMATH_CALUDE_total_amount_is_454_5_l105_10549

/-- Represents the share distribution problem -/
def ShareDistribution (w x y z p : ℚ) : Prop :=
  x = (3/2) * w ∧
  y = (1/3) * w ∧
  z = (3/4) * w ∧
  p = (5/8) * w ∧
  y = 36

/-- Theorem stating that the total amount is 454.5 rupees -/
theorem total_amount_is_454_5 (w x y z p : ℚ) 
  (h : ShareDistribution w x y z p) : 
  w + x + y + z + p = 454.5 := by
  sorry

#eval (454.5 : ℚ)

end NUMINAMATH_CALUDE_total_amount_is_454_5_l105_10549


namespace NUMINAMATH_CALUDE_converse_not_true_l105_10583

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem converse_not_true :
  ∃ (b : Line) (α β : Plane),
    (subset b β ∧ perp b α) ∧ ¬(plane_perp β α) :=
sorry

end NUMINAMATH_CALUDE_converse_not_true_l105_10583


namespace NUMINAMATH_CALUDE_parabola_vertex_l105_10582

/-- The vertex of a parabola is the point where it turns. For a parabola with equation
    y² + 8y + 2x + 11 = 0, this theorem states that the vertex is (5/2, -4). -/
theorem parabola_vertex (x y : ℝ) : 
  y^2 + 8*y + 2*x + 11 = 0 → (x = 5/2 ∧ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l105_10582


namespace NUMINAMATH_CALUDE_down_payment_ratio_l105_10562

theorem down_payment_ratio (total_cost balance_due daily_payment : ℚ) 
  (h1 : total_cost = 120)
  (h2 : balance_due = 60)
  (h3 : daily_payment = 6)
  (h4 : balance_due = daily_payment * 10) :
  (total_cost - balance_due) / total_cost = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_down_payment_ratio_l105_10562


namespace NUMINAMATH_CALUDE_sam_seashells_l105_10510

/-- Given that Sam found 35 seashells and gave 18 to Joan, prove that he now has 17 seashells. -/
theorem sam_seashells (initial : ℕ) (given_away : ℕ) (h1 : initial = 35) (h2 : given_away = 18) :
  initial - given_away = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l105_10510


namespace NUMINAMATH_CALUDE_power_of_five_mod_eight_l105_10556

theorem power_of_five_mod_eight : 5^1082 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_eight_l105_10556


namespace NUMINAMATH_CALUDE_tile_formation_theorem_l105_10596

/-- Represents a 4x4 tile --/
def Tile := Matrix (Fin 4) (Fin 4) Bool

/-- Checks if a tile has alternating colors on its outside row and column --/
def hasAlternatingOutside (t : Tile) : Prop :=
  (∀ i, t 0 i ≠ t 0 (i + 1)) ∧
  (∀ i, t i 0 ≠ t (i + 1) 0)

/-- Represents the property that a tile can be formed by combining two pieces --/
def canBeFormedByPieces (t : Tile) : Prop :=
  hasAlternatingOutside t

theorem tile_formation_theorem (t : Tile) :
  ¬(canBeFormedByPieces t) ↔ ¬(hasAlternatingOutside t) :=
sorry

end NUMINAMATH_CALUDE_tile_formation_theorem_l105_10596


namespace NUMINAMATH_CALUDE_regular_polygon_problem_l105_10569

theorem regular_polygon_problem (n : ℕ) (n_gt_2 : n > 2) :
  (n - 2) * 180 = 3 * 360 + 180 →
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_problem_l105_10569


namespace NUMINAMATH_CALUDE_circle_properties_l105_10534

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 36

-- Theorem statement
theorem circle_properties :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
    C = (-1, 3) ∧
    r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l105_10534


namespace NUMINAMATH_CALUDE_stop_to_qons_l105_10578

/-- Represents a letter in a 2D coordinate system -/
structure Letter where
  char : Char
  x : ℝ
  y : ℝ

/-- Represents a word as a list of letters -/
def Word := List Letter

/-- Rotates a letter 180° clockwise about the origin -/
def rotate180 (l : Letter) : Letter :=
  { l with x := -l.x, y := -l.y }

/-- Reflects a letter in the x-axis -/
def reflectX (l : Letter) : Letter :=
  { l with y := -l.y }

/-- Applies both transformations to a letter -/
def transform (l : Letter) : Letter :=
  reflectX (rotate180 l)

/-- Applies the transformation to a word -/
def transformWord (w : Word) : Word :=
  w.map transform

/-- The initial word "stop" -/
def initialWord : Word := sorry

/-- The expected final word "qons" -/
def finalWord : Word := sorry

theorem stop_to_qons :
  transformWord initialWord = finalWord := by sorry

end NUMINAMATH_CALUDE_stop_to_qons_l105_10578


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l105_10538

/-- The line equation passing through a fixed point for all values of a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 2) * x + (a + 1) * y + 6 = 0

/-- The fixed point coordinates -/
def fixed_point : ℝ × ℝ := (2, -2)

/-- Theorem stating that the line passes through the fixed point for all a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), line_equation a (fixed_point.1) (fixed_point.2) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l105_10538


namespace NUMINAMATH_CALUDE_fraction_identity_l105_10560

theorem fraction_identity (n : ℕ) : 
  2 / ((2 * n - 1) * (2 * n + 1)) = 1 / (2 * n - 1) - 1 / (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_identity_l105_10560


namespace NUMINAMATH_CALUDE_nell_cards_given_to_jeff_l105_10522

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff : ℕ := 304 - 276

theorem nell_cards_given_to_jeff :
  cards_given_to_jeff = 28 :=
by sorry

end NUMINAMATH_CALUDE_nell_cards_given_to_jeff_l105_10522


namespace NUMINAMATH_CALUDE_exists_product_in_A_l105_10547

/-- The set A(m, n) containing all integers of the form x^2 + mx + n for x ∈ ℤ -/
def A (m n : ℤ) : Set ℤ :=
  {y | ∃ x : ℤ, y = x^2 + m*x + n}

/-- For any integers m and n, there exist three distinct integers a, b, c in A(m, n) such that a = b * c -/
theorem exists_product_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a = b * c :=
by sorry

end NUMINAMATH_CALUDE_exists_product_in_A_l105_10547


namespace NUMINAMATH_CALUDE_petyas_torn_sheets_l105_10587

/-- Represents a book with consecutively numbered pages -/
structure Book where
  firstTornPage : ℕ
  lastTornPage : ℕ

/-- Checks if two numbers have the same digits -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Calculates the number of sheets torn out from a book -/
def sheetsTornOut (book : Book) : ℕ := sorry

/-- Theorem stating the number of sheets torn out by Petya -/
theorem petyas_torn_sheets (book : Book) : 
  book.firstTornPage = 185 ∧ 
  sameDigits book.firstTornPage book.lastTornPage ∧
  book.lastTornPage > book.firstTornPage ∧
  Even book.lastTornPage →
  sheetsTornOut book = 167 := by
  sorry

end NUMINAMATH_CALUDE_petyas_torn_sheets_l105_10587


namespace NUMINAMATH_CALUDE_max_value_polynomial_l105_10593

theorem max_value_polynomial (a b : ℝ) (h : a^2 + 4*b^2 = 4) :
  ∃ M : ℝ, M = 16 ∧ ∀ x y : ℝ, x^2 + 4*y^2 = 4 → 3*x^5*y - 40*x^3*y^3 + 48*x*y^5 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l105_10593


namespace NUMINAMATH_CALUDE_total_amount_proof_l105_10508

theorem total_amount_proof (a b c : ℕ) : 
  a = 3 * b → 
  b = c + 25 → 
  b = 134 → 
  a + b + c = 645 := by
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l105_10508


namespace NUMINAMATH_CALUDE_max_value_of_a_l105_10595

-- Define the operation
def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  (∃ a_max : ℝ, a ≤ a_max ∧ a_max = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l105_10595


namespace NUMINAMATH_CALUDE_patio_layout_change_l105_10579

theorem patio_layout_change (initial_tiles initial_rows initial_columns : ℕ) 
  (new_rows : ℕ) (h1 : initial_tiles = 160) (h2 : initial_rows = 10) 
  (h3 : initial_columns * initial_rows = initial_tiles)
  (h4 : new_rows = initial_rows + 4) :
  ∃ (new_columns : ℕ), 
    new_columns * new_rows = initial_tiles ∧ 
    initial_columns - new_columns = 5 := by
  sorry

end NUMINAMATH_CALUDE_patio_layout_change_l105_10579


namespace NUMINAMATH_CALUDE_frisbee_sales_theorem_l105_10568

/-- The total number of frisbees sold given the conditions -/
def total_frisbees : ℕ := 64

/-- The price of the cheaper frisbees -/
def price_cheap : ℕ := 3

/-- The price of the more expensive frisbees -/
def price_expensive : ℕ := 4

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 196

/-- The minimum number of expensive frisbees sold -/
def min_expensive : ℕ := 4

theorem frisbee_sales_theorem :
  ∃ (cheap expensive : ℕ),
    cheap + expensive = total_frisbees ∧
    cheap * price_cheap + expensive * price_expensive = total_receipts ∧
    expensive ≥ min_expensive :=
by
  sorry

end NUMINAMATH_CALUDE_frisbee_sales_theorem_l105_10568


namespace NUMINAMATH_CALUDE_independence_day_absentees_l105_10586

theorem independence_day_absentees (total_children : ℕ) (bananas : ℕ) (present_children : ℕ) : 
  total_children = 740 →
  bananas = total_children * 2 →
  bananas = present_children * 4 →
  total_children - present_children = 370 := by
sorry

end NUMINAMATH_CALUDE_independence_day_absentees_l105_10586


namespace NUMINAMATH_CALUDE_unique_solution_l105_10550

theorem unique_solution (x y z : ℝ) 
  (hx : x > 3) (hy : y > 3) (hz : z > 3)
  (h : (x + 4)^2 / (y + z - 4) + (y + 6)^2 / (z + x - 6) + (z + 8)^2 / (x + y - 8) = 48) :
  x = 11 ∧ y = 10 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l105_10550


namespace NUMINAMATH_CALUDE_distance_and_closest_point_theorem_l105_10518

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by a point and a direction vector -/
structure Line3D where
  point : Point3D
  direction : Vector3D

def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ :=
  sorry

def closest_point_on_line (p : Point3D) (l : Line3D) : Point3D :=
  sorry

theorem distance_and_closest_point_theorem :
  let p := Point3D.mk 3 4 5
  let l := Line3D.mk (Point3D.mk 2 3 1) (Vector3D.mk 1 (-1) 2)
  distance_point_to_line p l = Real.sqrt 6 / 3 ∧
  closest_point_on_line p l = Point3D.mk (10/3) (5/3) (11/3) := by
  sorry

end NUMINAMATH_CALUDE_distance_and_closest_point_theorem_l105_10518


namespace NUMINAMATH_CALUDE_complement_P_subset_Q_l105_10552

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- State the theorem
theorem complement_P_subset_Q : (Set.univ \ P) ⊆ Q := by sorry

end NUMINAMATH_CALUDE_complement_P_subset_Q_l105_10552


namespace NUMINAMATH_CALUDE_grain_milling_theorem_l105_10500

/-- The amount of grain needed to be milled, in pounds -/
def grain_amount : ℚ := 111 + 1/9

/-- The milling fee percentage -/
def milling_fee_percent : ℚ := 1/10

/-- The amount of flour remaining after paying the fee, in pounds -/
def remaining_flour : ℚ := 100

theorem grain_milling_theorem :
  (1 - milling_fee_percent) * grain_amount = remaining_flour :=
by sorry

end NUMINAMATH_CALUDE_grain_milling_theorem_l105_10500


namespace NUMINAMATH_CALUDE_meaningful_expression_l105_10570

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y * y = x ∧ y ≥ 0) ∧ x ≠ 2 ↔ x ≥ 0 ∧ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l105_10570


namespace NUMINAMATH_CALUDE_go_stones_perimeter_l105_10575

/-- The number of stones on one side of the square arrangement. -/
def side_length : ℕ := 6

/-- The number of sides in a square. -/
def num_sides : ℕ := 4

/-- The number of corners in a square. -/
def num_corners : ℕ := 4

/-- Calculates the number of stones on the perimeter of a square arrangement. -/
def perimeter_stones (n : ℕ) : ℕ := n * num_sides - num_corners

theorem go_stones_perimeter :
  perimeter_stones side_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_perimeter_l105_10575


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_8_l105_10559

theorem least_n_factorial_divisible_by_8 : 
  ∃ n : ℕ, n > 0 ∧ 8 ∣ n.factorial ∧ ∀ m : ℕ, m > 0 → m < n → ¬(8 ∣ m.factorial) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_8_l105_10559


namespace NUMINAMATH_CALUDE_kirill_height_l105_10517

/-- Represents the heights of three siblings -/
structure SiblingHeights where
  kirill : ℝ
  brother : ℝ
  sister : ℝ

/-- The conditions of the problem -/
def height_conditions (h : SiblingHeights) : Prop :=
  h.brother = h.kirill + 14 ∧
  h.sister = 2 * h.kirill ∧
  h.kirill + h.brother + h.sister = 264

/-- Theorem stating Kirill's height given the conditions -/
theorem kirill_height (h : SiblingHeights) 
  (hc : height_conditions h) : h.kirill = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_kirill_height_l105_10517


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l105_10592

/-- Given a rectangular room with specified dimensions and total paving cost,
    calculate the rate per square meter for paving the floor. -/
theorem paving_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 9)
    (h2 : width = 4.75)
    (h3 : total_cost = 38475) : 
  total_cost / (length * width) = 900 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l105_10592


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l105_10535

/-- Represents a parabola of the form x = 3y^2 - 9y + 4 -/
def Parabola : ℝ → ℝ := λ y => 3 * y^2 - 9 * y + 4

/-- The x-intercept of the parabola -/
def a : ℝ := Parabola 0

/-- The y-intercepts of the parabola -/
def y_intercepts : Set ℝ := {y | Parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), y_intercepts = {b, c} ∧ a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l105_10535


namespace NUMINAMATH_CALUDE_tangent_line_inclination_angle_range_l105_10580

open Real Set

theorem tangent_line_inclination_angle_range :
  ∀ x : ℝ, 
  let P : ℝ × ℝ := (x, Real.sin x)
  let θ := Real.arctan (Real.cos x)
  θ ∈ Icc 0 (π/4) ∪ Ico (3*π/4) π := by
sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_angle_range_l105_10580


namespace NUMINAMATH_CALUDE_units_digit_17_pow_27_l105_10543

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result we want to prove -/
theorem units_digit_17_pow_27 : unitsDigit (17^27) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_27_l105_10543


namespace NUMINAMATH_CALUDE_percent_of_percent_l105_10548

theorem percent_of_percent : (3 / 100) / (5 / 100) * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l105_10548


namespace NUMINAMATH_CALUDE_current_rate_l105_10503

/-- The rate of the current given a man's rowing speed and time ratio -/
theorem current_rate (man_speed : ℝ) (time_ratio : ℝ) : 
  man_speed = 3.6 ∧ time_ratio = 2 → 
  ∃ c : ℝ, c = 1.2 ∧ (man_speed - c) / (man_speed + c) = 1 / time_ratio :=
by sorry

end NUMINAMATH_CALUDE_current_rate_l105_10503


namespace NUMINAMATH_CALUDE_trouser_price_decrease_trouser_price_decrease_result_l105_10573

/-- Calculates the final percent decrease in price for a trouser purchase with given conditions. -/
theorem trouser_price_decrease (original_price : ℝ) (clearance_discount : ℝ) 
  (german_vat : ℝ) (us_vat : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - clearance_discount)
  let price_with_german_vat := discounted_price * (1 + german_vat)
  let price_in_usd := price_with_german_vat * exchange_rate
  let final_price := price_in_usd * (1 + us_vat)
  let original_price_usd := original_price * exchange_rate
  let percent_decrease := (original_price_usd - final_price) / original_price_usd * 100
  percent_decrease

/-- The final percent decrease in price is approximately 10.0359322%. -/
theorem trouser_price_decrease_result : 
  abs (trouser_price_decrease 100 0.3 0.19 0.08 1.18 - 10.0359322) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_trouser_price_decrease_trouser_price_decrease_result_l105_10573


namespace NUMINAMATH_CALUDE_tape_length_calculation_l105_10576

/-- The total length of overlapping tape sheets -/
def total_tape_length (n : ℕ) (sheet_length : ℝ) (overlap : ℝ) : ℝ :=
  sheet_length + (n - 1 : ℝ) * (sheet_length - overlap)

/-- Theorem: The total length of 15 sheets of tape, each 20 cm long and overlapping by 5 cm, is 230 cm -/
theorem tape_length_calculation :
  total_tape_length 15 20 5 = 230 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_calculation_l105_10576


namespace NUMINAMATH_CALUDE_thousandths_place_of_three_sixteenths_l105_10558

theorem thousandths_place_of_three_sixteenths (f : Rat) (d : ℕ) : 
  f = 3 / 16 →
  d = (⌊f * 1000⌋ % 10) →
  d = 7 := by
sorry

end NUMINAMATH_CALUDE_thousandths_place_of_three_sixteenths_l105_10558


namespace NUMINAMATH_CALUDE_initial_average_age_l105_10537

theorem initial_average_age (n : ℕ) (new_age : ℕ) (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : new_age = 35)
  (h3 : new_average = 17) :
  ∃ initial_average : ℚ, 
    initial_average = 15 ∧ 
    (n : ℚ) * initial_average + new_age = ((n : ℚ) + 1) * new_average :=
by sorry

end NUMINAMATH_CALUDE_initial_average_age_l105_10537


namespace NUMINAMATH_CALUDE_div_negative_powers_l105_10598

theorem div_negative_powers (a : ℝ) (h : a ≠ 0) : -28 * a^3 / (7 * a) = -4 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_div_negative_powers_l105_10598


namespace NUMINAMATH_CALUDE_total_combinations_l105_10516

/-- The number of color choices available for painting the box. -/
def num_colors : ℕ := 4

/-- The number of decoration choices available for the box. -/
def num_decorations : ℕ := 3

/-- The number of painting method choices available. -/
def num_methods : ℕ := 3

/-- Theorem stating the total number of combinations for painting the box. -/
theorem total_combinations : num_colors * num_decorations * num_methods = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_l105_10516


namespace NUMINAMATH_CALUDE_candy_boxes_minimum_l105_10563

theorem candy_boxes_minimum (x y m : ℕ) : 
  x + y = 176 → 
  m > 1 → 
  x + 16 = m * (y - 16) + 31 → 
  x ≥ 131 :=
by sorry

end NUMINAMATH_CALUDE_candy_boxes_minimum_l105_10563


namespace NUMINAMATH_CALUDE_school_population_l105_10591

/-- Given a school population where:
  * The number of boys is 4 times the number of girls
  * The number of girls is 8 times the number of teachers
This theorem proves that the total number of boys, girls, and teachers
is equal to 41/32 times the number of boys. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = (41 * b) / 32 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l105_10591


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l105_10509

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of triangles in the large equilateral triangle -/
def num_triangles : ℕ := 6

/-- Represents the number of corner triangles -/
def num_corners : ℕ := 3

/-- Represents the number of edge triangles -/
def num_edges : ℕ := 2

/-- Represents the number of center triangles -/
def num_center : ℕ := 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of distinguishable large equilateral triangles -/
def num_distinguishable_triangles : ℕ :=
  -- Corner configurations
  (num_colors + 
   num_colors * (num_colors - 1) + 
   choose num_colors num_corners) *
  -- Edge configurations
  (num_colors ^ num_edges) *
  -- Center configurations
  num_colors

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 61440 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l105_10509


namespace NUMINAMATH_CALUDE_scenario_1_scenario_2_l105_10574

-- Define the lines l₁ and l₂
def l₁ (a b x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a x y : ℝ) : Prop := (a - 1) * x + y + 2 = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (1 - a) = -b

-- Define parallelism of lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Theorem for Scenario 1
theorem scenario_1 (a b : ℝ) : 
  l₁ a b (-3) (-1) ∧ perpendicular a b → a = 2 ∧ b = 2 :=
sorry

-- Theorem for Scenario 2
theorem scenario_2 (a b : ℝ) :
  parallel a b ∧ (4 / b = -3) → a = 4 ∧ b = -4/3 :=
sorry

end NUMINAMATH_CALUDE_scenario_1_scenario_2_l105_10574


namespace NUMINAMATH_CALUDE_andrew_payment_l105_10532

/-- The amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1376 to the shopkeeper -/
theorem andrew_payment : total_amount 14 54 10 62 = 1376 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l105_10532


namespace NUMINAMATH_CALUDE_train_passing_time_specific_train_passing_time_l105_10505

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length > 0 → train_speed > man_speed → train_speed > 0 → man_speed ≥ 0 →
  ∃ (t : ℝ), t > 0 ∧ t < 19 ∧ t * (train_speed - man_speed) * (5 / 18) = train_length :=
by sorry

/-- Specific instance of the train passing time problem -/
theorem specific_train_passing_time :
  ∃ (t : ℝ), t > 0 ∧ t < 19 ∧ t * (68 - 8) * (5 / 18) = 300 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_specific_train_passing_time_l105_10505


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l105_10536

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/21
  let a₃ : ℚ := 64/63
  let r : ℚ := a₂ / a₁
  r = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l105_10536


namespace NUMINAMATH_CALUDE_square_2601_difference_of_squares_l105_10512

theorem square_2601_difference_of_squares (x : ℤ) (h : x^2 = 2601) :
  (x + 2) * (x - 2) = 2597 := by
sorry

end NUMINAMATH_CALUDE_square_2601_difference_of_squares_l105_10512


namespace NUMINAMATH_CALUDE_lecture_duration_in_minutes_l105_10529

-- Define the duration of the lecture
def lecture_hours : ℕ := 8
def lecture_minutes : ℕ := 45

-- Define the conversion factor
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem lecture_duration_in_minutes :
  lecture_hours * minutes_per_hour + lecture_minutes = 525 := by
  sorry

end NUMINAMATH_CALUDE_lecture_duration_in_minutes_l105_10529


namespace NUMINAMATH_CALUDE_lee_cookies_l105_10530

/-- Given that Lee can make 24 cookies with 4 cups of flour, 
    this theorem proves that he can make 30 cookies with 5 cups of flour. -/
theorem lee_cookies (cookies_per_4_cups : ℕ) (h : cookies_per_4_cups = 24) :
  (cookies_per_4_cups * 5 / 4 : ℚ) = 30 := by
  sorry


end NUMINAMATH_CALUDE_lee_cookies_l105_10530


namespace NUMINAMATH_CALUDE_relationship_abc_l105_10599

theorem relationship_abc (a b c : ℝ) (ha : a = Real.log 3 / Real.log 0.5)
  (hb : b = Real.sqrt 2) (hc : c = Real.sqrt 0.5) : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l105_10599


namespace NUMINAMATH_CALUDE_volleyball_score_ratio_l105_10507

theorem volleyball_score_ratio :
  let lizzie_score : ℕ := 4
  let nathalie_score : ℕ := lizzie_score + 3
  let combined_score : ℕ := lizzie_score + nathalie_score
  let team_total : ℕ := 50
  let teammates_score : ℕ := 17
  ∃ (m : ℕ), 
    m * combined_score = team_total - lizzie_score - nathalie_score - teammates_score ∧
    m * combined_score = 2 * combined_score :=
by
  sorry

#check volleyball_score_ratio

end NUMINAMATH_CALUDE_volleyball_score_ratio_l105_10507


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l105_10567

theorem factorization_difference_of_squares (a : ℝ) : 
  a^2 - 9 = (a + 3) * (a - 3) := by sorry

#check factorization_difference_of_squares

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l105_10567


namespace NUMINAMATH_CALUDE_photo_arrangements_l105_10523

/-- The number of ways to arrange 1 teacher and 4 students in a row with the teacher in the middle -/
def arrangements_count : ℕ := 24

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of ways to arrange the students -/
def student_arrangements : ℕ := Nat.factorial num_students

theorem photo_arrangements :
  arrangements_count = student_arrangements := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l105_10523


namespace NUMINAMATH_CALUDE_inequality_relationships_l105_10577

theorem inequality_relationships (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧
  (1 / (a - b) < 1 / a) ∧
  (|a| > |b|) ∧
  (a^4 > b^4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationships_l105_10577


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l105_10590

theorem not_necessary_not_sufficient_condition (a b : ℝ) : 
  ¬(((a > 0 ∧ b > 0) → (a * b < ((a + b) / 2)^2)) ∧
    ((a * b < ((a + b) / 2)^2) → (a > 0 ∧ b > 0))) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l105_10590


namespace NUMINAMATH_CALUDE_gilda_marbles_theorem_l105_10524

/-- The percentage of marbles Gilda has left after giving away to her friends and brother -/
def gildasRemainingMarbles : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.20)
  let afterCarlos := afterEbony * (1 - 0.15)
  let afterJimmy := afterCarlos * (1 - 0.10)
  afterJimmy

/-- Theorem stating that Gilda has approximately 43% of her original marbles left -/
theorem gilda_marbles_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |gildasRemainingMarbles - 43| < ε :=
sorry

end NUMINAMATH_CALUDE_gilda_marbles_theorem_l105_10524


namespace NUMINAMATH_CALUDE_area_inner_octagon_l105_10531

/-- The area of a regular octagon formed by connecting the midpoints of four alternate sides of a regular octagon with side length 12 cm. -/
theorem area_inner_octagon (side_length : ℝ) (h_side : side_length = 12) : 
  ∃ area : ℝ, area = 576 + 288 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_area_inner_octagon_l105_10531


namespace NUMINAMATH_CALUDE_triangle_area_l105_10521

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b = 7 →
  c = 5 →
  B = 2 * π / 3 →
  (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l105_10521


namespace NUMINAMATH_CALUDE_max_value_theorem_l105_10540

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 - 2*a*b + 9*b^2 - c = 0) :
  ∃ (max_abc : ℝ), 
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 - 2*x*y + 9*y^2 - z = 0 → 
      x*y/z ≤ max_abc) →
    (3/a + 1/b - 12/c ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l105_10540


namespace NUMINAMATH_CALUDE_rain_probability_l105_10555

-- Define the probability of rain on any given day
def p_rain : ℝ := 0.5

-- Define the number of days
def n : ℕ := 6

-- Define the number of rainy days we're interested in
def k : ℕ := 4

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem rain_probability :
  (binomial_coefficient n k : ℝ) * p_rain ^ k * (1 - p_rain) ^ (n - k) = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l105_10555


namespace NUMINAMATH_CALUDE_race_distance_l105_10566

/-- 
Proves that given the conditions of two runners A and B, 
the race distance is 160 meters.
-/
theorem race_distance (t_A t_B : ℝ) (lead : ℝ) : 
  t_A = 28 →  -- A's time
  t_B = 32 →  -- B's time
  lead = 20 → -- A's lead over B at finish
  ∃ d : ℝ, d = 160 ∧ d / t_A = (d - lead) / t_B :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l105_10566


namespace NUMINAMATH_CALUDE_factory_production_rate_l105_10501

/-- Represents the production setup of a factory --/
structure Factory where
  original_machines : ℕ
  original_hours : ℕ
  new_machine_hours : ℕ
  price_per_kg : ℕ
  daily_earnings : ℕ

/-- Calculates the hourly production rate of a single machine --/
def hourly_production_rate (f : Factory) : ℚ :=
  let total_machine_hours := f.original_machines * f.original_hours + f.new_machine_hours
  let daily_production := f.daily_earnings / f.price_per_kg
  daily_production / total_machine_hours

/-- Theorem stating the hourly production rate of a single machine --/
theorem factory_production_rate (f : Factory) 
  (h1 : f.original_machines = 3)
  (h2 : f.original_hours = 23)
  (h3 : f.new_machine_hours = 12)
  (h4 : f.price_per_kg = 50)
  (h5 : f.daily_earnings = 8100) :
  hourly_production_rate f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_rate_l105_10501


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l105_10528

theorem quadratic_equation_roots (m : ℕ+) : 
  (∃ x : ℝ, x^2 - 2*x + 2*(m : ℝ) - 1 = 0) → 
  (m = 1 ∧ ∀ x : ℝ, x^2 - 2*x + 2*(m : ℝ) - 1 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l105_10528


namespace NUMINAMATH_CALUDE_repeated_digit_sum_2_power_2004_l105_10557

/-- The digit sum function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Repeated application of digit_sum until a single digit is reached -/
def repeated_digit_sum (n : ℕ) : ℕ := sorry

/-- 2^2004 mod 9 ≡ 1 -/
lemma power_two_2004_mod_9 : 2^2004 % 9 = 1 := sorry

/-- For any natural number n, n ≡ digit_sum(n) (mod 9) -/
lemma digit_sum_congruence (n : ℕ) : n % 9 = digit_sum n % 9 := sorry

/-- The main theorem -/
theorem repeated_digit_sum_2_power_2004 : 
  repeated_digit_sum (2^2004) = 1 := sorry

end NUMINAMATH_CALUDE_repeated_digit_sum_2_power_2004_l105_10557


namespace NUMINAMATH_CALUDE_complex_multiplication_l105_10554

theorem complex_multiplication (Q E D : ℂ) : 
  Q = 7 + 3*I ∧ E = 2 + I ∧ D = 7 - 3*I → Q * E * D = 116 + 58*I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l105_10554


namespace NUMINAMATH_CALUDE_city_population_problem_l105_10585

theorem city_population_problem (p : ℝ) : 
  (0.84 * (p + 2500) + 500 = p + 2680) → p = 500 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l105_10585


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_l105_10561

/-- Given a triangle with side lengths a, b, and c, 
    the sum of the ratios of each side length to the difference between 
    the sum of the other two sides and itself is greater than or equal to 3. -/
theorem triangle_side_ratio_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_l105_10561


namespace NUMINAMATH_CALUDE_symmetry_properties_l105_10520

/-- A function f: ℝ → ℝ is symmetric about the line x=a if f(a-x) = f(a+x) for all x ∈ ℝ -/
def symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

/-- A function f: ℝ → ℝ is symmetric about the y-axis if f(x) = f(-x) for all x ∈ ℝ -/
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Two functions f, g: ℝ → ℝ have graphs symmetric about the y-axis if f(x) = g(-x) for all x ∈ ℝ -/
def graphs_symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- Two functions f, g: ℝ → ℝ have graphs symmetric about the line x=a if f(x) = g(2a-x) for all x ∈ ℝ -/
def graphs_symmetric_about_line (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = g (2*a - x)

theorem symmetry_properties (f : ℝ → ℝ) :
  (symmetric_about_line f 4) ∧
  ((∀ x, f (4 - x) = f (x - 4)) → symmetric_about_y_axis f) ∧
  (graphs_symmetric_about_y_axis (fun x ↦ f (4 - x)) (fun x ↦ f (4 + x))) ∧
  (graphs_symmetric_about_line (fun x ↦ f (4 - x)) (fun x ↦ f (x - 4)) 4) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l105_10520


namespace NUMINAMATH_CALUDE_fraction_of_juniors_studying_japanese_l105_10594

/-- Proves that the fraction of juniors studying Japanese is 3/4 given the specified conditions. -/
theorem fraction_of_juniors_studying_japanese :
  ∀ (j s : ℕ), -- j: number of juniors, s: number of seniors
  s = 2 * j → -- senior class is twice the size of junior class
  ∃ (x : ℚ), -- x: fraction of juniors studying Japanese
  (1 / 8 : ℚ) * s + x * j = (1 / 3 : ℚ) * (j + s) ∧ -- equation based on given conditions
  x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_juniors_studying_japanese_l105_10594


namespace NUMINAMATH_CALUDE_subtract_negatives_l105_10544

theorem subtract_negatives : -3 - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l105_10544


namespace NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l105_10506

theorem reciprocal_of_sqrt_two :
  (1 : ℝ) / Real.sqrt 2 = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l105_10506


namespace NUMINAMATH_CALUDE_salesman_pears_sold_l105_10565

/-- The amount of pears sold by a salesman in a day -/
theorem salesman_pears_sold (morning_sales afternoon_sales : ℕ) 
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : morning_sales = 120)
  (h3 : afternoon_sales = 240) : 
  morning_sales + afternoon_sales = 360 := by
  sorry

end NUMINAMATH_CALUDE_salesman_pears_sold_l105_10565


namespace NUMINAMATH_CALUDE_sin_150_degrees_l105_10545

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l105_10545


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l105_10553

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the fixed point A
def point_A : ℝ × ℝ := (4, 3)

-- Define a point P outside circle O
def point_P (a b : ℝ) : Prop := a^2 + b^2 > 5

-- Define the tangent line condition
def is_tangent (a b : ℝ) : Prop := ∃ (t : ℝ), circle_O (a + t) (b + t) ∧ ∀ (s : ℝ), s ≠ t → ¬ circle_O (a + s) (b + s)

-- Define the equality of lengths PQ and PA
def length_equality (a b : ℝ) : Prop := (a - 4)^2 + (b - 3)^2 = a^2 + b^2 - 5

theorem circle_tangent_properties (a b : ℝ) 
  (h1 : point_P a b) 
  (h2 : is_tangent a b) 
  (h3 : length_equality a b) :
  -- 1. Relationship between a and b
  (4 * a + 3 * b - 15 = 0) ∧
  -- 2. Minimum length of PQ
  (∀ (x y : ℝ), point_P x y → is_tangent x y → length_equality x y → 
    (x - 4)^2 + (y - 3)^2 ≥ 16) ∧
  -- 3. Equation of circle P with minimum radius
  (∃ (r : ℝ), r = 3 - Real.sqrt 5 ∧
    ∀ (x y : ℝ), (x - 12/5)^2 + (y - 9/5)^2 = r^2 →
      ∃ (t : ℝ), circle_O (x + t) (y + t)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l105_10553


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_6_l105_10511

theorem smallest_common_multiple_of_5_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 5 ∣ m → 6 ∣ m → n ≤ m :=
by sorry

#check smallest_common_multiple_of_5_and_6

end NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_6_l105_10511


namespace NUMINAMATH_CALUDE_equation_holds_l105_10571

theorem equation_holds (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l105_10571


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l105_10513

/-- Given two adjacent points (1,2) and (4,6) on a square, prove that the area of the square is 25. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l105_10513


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l105_10527

theorem rectangular_solid_depth (l w sa : ℝ) (h : ℝ) : 
  l = 6 → w = 5 → sa = 104 → sa = 2 * l * w + 2 * l * h + 2 * w * h → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l105_10527


namespace NUMINAMATH_CALUDE_optimal_strategy_l105_10542

-- Define the warehouse options
structure Warehouse where
  monthly_rent : ℝ
  repossession_probability : ℝ
  repossession_time : ℕ

-- Define the company's parameters
structure Company where
  planning_horizon : ℕ
  moving_cost : ℝ

-- Define the purchase option
structure PurchaseOption where
  total_price : ℝ
  installment_period : ℕ

def calculate_total_cost (w : Warehouse) (c : Company) (years : ℕ) : ℝ :=
  sorry

def calculate_purchase_cost (p : PurchaseOption) : ℝ :=
  sorry

theorem optimal_strategy (w1 w2 : Warehouse) (c : Company) (p : PurchaseOption) :
  w1.monthly_rent = 80000 ∧
  w2.monthly_rent = 20000 ∧
  w2.repossession_probability = 0.5 ∧
  w2.repossession_time = 5 ∧
  c.planning_horizon = 60 ∧
  c.moving_cost = 150000 ∧
  p.total_price = 3000000 ∧
  p.installment_period = 36 →
  calculate_total_cost w2 c 1 + calculate_purchase_cost p <
  min (calculate_total_cost w1 c 5) (calculate_total_cost w2 c 5) :=
sorry

#check optimal_strategy

end NUMINAMATH_CALUDE_optimal_strategy_l105_10542


namespace NUMINAMATH_CALUDE_rectangle_length_reduction_l105_10589

theorem rectangle_length_reduction (original_length original_width : ℝ) 
  (h : original_length > 0 ∧ original_width > 0) :
  let new_width := original_width * (1 + 0.4285714285714287)
  let new_length := original_length * 0.7
  original_length * original_width = new_length * new_width :=
by
  sorry

#check rectangle_length_reduction

end NUMINAMATH_CALUDE_rectangle_length_reduction_l105_10589


namespace NUMINAMATH_CALUDE_line_slope_is_one_l105_10597

-- Define the line using its point-slope form
def line_equation (x y : ℝ) : Prop := y + 1 = x - 2

-- State the theorem
theorem line_slope_is_one :
  ∀ x y : ℝ, line_equation x y → (y - (y + 1)) / (x - (x - 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l105_10597


namespace NUMINAMATH_CALUDE_sum_of_digits_l105_10525

-- Define the variables as natural numbers
variable (a b c d : ℕ)

-- Define the conditions
axiom different_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom sum_hundreds_ones : a + c = 10
axiom sum_tens : b + c = 8
axiom sum_hundreds : a + d = 11
axiom result_sum : 100 * a + 10 * b + c + 100 * d + 10 * c + a = 1180

-- State the theorem
theorem sum_of_digits : a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l105_10525


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l105_10502

/-- Calculates the rate of mixed oil per litre given the volumes and rates of three different oils. -/
theorem mixed_oil_rate (v1 v2 v3 r1 r2 r3 : ℚ) : 
  v1 > 0 ∧ v2 > 0 ∧ v3 > 0 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 →
  (v1 * r1 + v2 * r2 + v3 * r3) / (v1 + v2 + v3) = 
    (15 * 50 + 8 * 75 + 10 * 65) / (15 + 8 + 10) :=
by
  sorry

#eval (15 * 50 + 8 * 75 + 10 * 65) / (15 + 8 + 10)

end NUMINAMATH_CALUDE_mixed_oil_rate_l105_10502


namespace NUMINAMATH_CALUDE_monday_rainfall_rate_l105_10533

/-- Represents the rainfall data for three days -/
structure RainfallData where
  monday_hours : ℝ
  monday_rate : ℝ
  tuesday_hours : ℝ
  tuesday_rate : ℝ
  wednesday_hours : ℝ
  wednesday_rate : ℝ
  total_rainfall : ℝ

/-- Theorem stating that given the rainfall conditions, the rate on Monday was 1 inch per hour -/
theorem monday_rainfall_rate (data : RainfallData)
  (h1 : data.monday_hours = 7)
  (h2 : data.tuesday_hours = 4)
  (h3 : data.tuesday_rate = 2)
  (h4 : data.wednesday_hours = 2)
  (h5 : data.wednesday_rate = 2 * data.tuesday_rate)
  (h6 : data.total_rainfall = 23)
  (h7 : data.total_rainfall = data.monday_hours * data.monday_rate + 
                              data.tuesday_hours * data.tuesday_rate + 
                              data.wednesday_hours * data.wednesday_rate) :
  data.monday_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_rate_l105_10533


namespace NUMINAMATH_CALUDE_unique_a_satisfies_condition_l105_10581

/-- Converts a base-25 number to its decimal representation modulo 12 -/
def base25ToDecimalMod12 (digits : List Nat) : Nat :=
  (digits.reverse.enum.map (fun (i, d) => d * (25^i % 12)) |>.sum) % 12

/-- The given number in base 25 -/
def number : List Nat := [3, 1, 4, 2, 6, 5, 2, 3]

theorem unique_a_satisfies_condition :
  ∃! a : ℕ, 0 ≤ a ∧ a ≤ 14 ∧ (base25ToDecimalMod12 number - a) % 12 = 0 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_satisfies_condition_l105_10581


namespace NUMINAMATH_CALUDE_division_fifteen_by_negative_five_l105_10588

theorem division_fifteen_by_negative_five : (15 : ℤ) / (-5 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_division_fifteen_by_negative_five_l105_10588


namespace NUMINAMATH_CALUDE_factor_calculation_l105_10564

theorem factor_calculation (x : ℝ) : 60 * x - 138 = 102 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l105_10564


namespace NUMINAMATH_CALUDE_petya_cannot_guarantee_win_l105_10539

/-- Represents a position on the 9x9 board -/
structure Position :=
  (x : Fin 9)
  (y : Fin 9)

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- The game state -/
structure GameState :=
  (position : Position)
  (lastDirection : Direction)
  (currentPlayer : Player)

/-- Checks if a move is valid for a given player -/
def isValidMove (player : Player) (lastDir : Direction) (newDir : Direction) : Prop :=
  match player with
  | Player.Petya => newDir = lastDir ∨ newDir = Direction.Right
  | Player.Vasya => newDir = lastDir ∨ newDir = Direction.Left

/-- Checks if a position is on the board -/
def isOnBoard (pos : Position) : Prop :=
  0 ≤ pos.x ∧ pos.x < 9 ∧ 0 ≤ pos.y ∧ pos.y < 9

/-- Theorem stating that Petya cannot guarantee a win -/
theorem petya_cannot_guarantee_win :
  ∀ (strategy : GameState → Direction),
  ∃ (counterStrategy : GameState → Direction),
  ∃ (finalState : GameState),
  (finalState.currentPlayer = Player.Petya ∧ 
   ¬∃ (dir : Direction), isValidMove Player.Petya finalState.lastDirection dir ∧ 
                         isOnBoard (finalState.position)) :=
sorry

end NUMINAMATH_CALUDE_petya_cannot_guarantee_win_l105_10539


namespace NUMINAMATH_CALUDE_inequality_proof_l105_10584

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l105_10584


namespace NUMINAMATH_CALUDE_wall_width_calculation_l105_10515

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 25 →
  wall_height = 2 →
  num_bricks = 25000 →
  ∃ (wall_width : ℝ), wall_width = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l105_10515


namespace NUMINAMATH_CALUDE_tourist_assignment_count_l105_10572

/-- The number of ways to assign tourists to scenic spots -/
def assignmentCount (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else if n = k then Nat.factorial k
  else (Nat.choose n 2) * (Nat.factorial k)

theorem tourist_assignment_count :
  assignmentCount 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tourist_assignment_count_l105_10572


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_l105_10514

theorem sqrt_sum_equals_two (a b : ℝ) (h : a^2 + b^2 = 4) :
  (a * (b - 4))^(1/3) + ((a * b - 3 * a + 2 * b - 6) : ℝ)^(1/2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_l105_10514


namespace NUMINAMATH_CALUDE_fourth_root_sum_squared_l105_10519

theorem fourth_root_sum_squared : 
  (Real.rpow (7 + 3 * Real.sqrt 5) (1/4) + Real.rpow (7 - 3 * Real.sqrt 5) (1/4))^4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sum_squared_l105_10519


namespace NUMINAMATH_CALUDE_large_positive_integer_product_l105_10504

theorem large_positive_integer_product : ∃ n : ℕ, n > 10^100 ∧ 
  (2+3)*(2^2+3^2)*(2^4-3^4)*(2^8+3^8)*(2^16-3^16)*(2^32+3^32)*(2^64-3^64) = n := by
  sorry

end NUMINAMATH_CALUDE_large_positive_integer_product_l105_10504


namespace NUMINAMATH_CALUDE_purely_imaginary_x_equals_one_l105_10546

-- Define a complex number
def complex_number (x : ℝ) : ℂ := (x^2 - 1) + (x + 1) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_x_equals_one :
  ∀ x : ℝ, is_purely_imaginary (complex_number x) → x = 1 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_x_equals_one_l105_10546


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_510_l105_10551

theorem sin_n_equals_cos_510 (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * π / 180) = Real.cos (510 * π / 180) → n = -60 := by sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_510_l105_10551
