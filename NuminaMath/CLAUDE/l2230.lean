import Mathlib

namespace NUMINAMATH_CALUDE_birds_meeting_time_l2230_223037

/-- The time taken for two birds flying in opposite directions to meet -/
theorem birds_meeting_time 
  (duck_time : ℝ) 
  (goose_time : ℝ) 
  (duck_time_positive : duck_time > 0)
  (goose_time_positive : goose_time > 0) :
  ∃ x : ℝ, x > 0 ∧ (1 / duck_time + 1 / goose_time) * x = 1 :=
sorry

end NUMINAMATH_CALUDE_birds_meeting_time_l2230_223037


namespace NUMINAMATH_CALUDE_expression_simplification_l2230_223060

theorem expression_simplification (a b x y : ℝ) :
  (2*a - (4*a + 5*b) + 2*(3*a - 4*b) = 4*a - 13*b) ∧
  (5*x^2 - 2*(3*y^2 - 5*x^2) + (-4*y^2 + 7*x*y) = 15*x^2 - 10*y^2 + 7*x*y) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2230_223060


namespace NUMINAMATH_CALUDE_brother_statement_contradiction_l2230_223043

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the brother's behavior
structure Brother where
  lying_days : Set Day
  today : Day

-- Define the brother's statement
def brother_statement (b : Brother) : Prop :=
  b.today ∈ b.lying_days

-- Theorem: The brother's statement leads to a contradiction
theorem brother_statement_contradiction (b : Brother) :
  ¬(brother_statement b ↔ ¬(brother_statement b)) :=
sorry

end NUMINAMATH_CALUDE_brother_statement_contradiction_l2230_223043


namespace NUMINAMATH_CALUDE_roller_coaster_rides_l2230_223042

/-- Given a roller coaster that costs 5 tickets per ride and a person with 10 tickets,
    prove that the number of possible rides is 2. -/
theorem roller_coaster_rides (total_tickets : ℕ) (cost_per_ride : ℕ) (h1 : total_tickets = 10) (h2 : cost_per_ride = 5) :
  total_tickets / cost_per_ride = 2 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_rides_l2230_223042


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l2230_223053

theorem complex_magnitude_one (z : ℂ) (r : ℝ) 
  (h1 : |r| < 2) 
  (h2 : z + z⁻¹ = r) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l2230_223053


namespace NUMINAMATH_CALUDE_max_min_f_on_I_l2230_223052

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_f_on_I :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 := by
sorry

end NUMINAMATH_CALUDE_max_min_f_on_I_l2230_223052


namespace NUMINAMATH_CALUDE_external_diagonals_theorem_l2230_223015

/-- Checks if a set of three numbers could be the lengths of external diagonals of a right regular prism -/
def valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 ≥ c^2 ∧
  b^2 + c^2 ≥ a^2 ∧
  a^2 + c^2 ≥ b^2

theorem external_diagonals_theorem :
  ¬(valid_external_diagonals 3 4 6) ∧
  valid_external_diagonals 3 4 5 ∧
  valid_external_diagonals 5 6 8 ∧
  valid_external_diagonals 5 8 9 ∧
  valid_external_diagonals 6 8 10 :=
by sorry

end NUMINAMATH_CALUDE_external_diagonals_theorem_l2230_223015


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l2230_223092

theorem x_equals_one_sufficient_not_necessary (x : ℝ) :
  (x = 1 → x * (x - 1) = 0) ∧ (∃ y : ℝ, y ≠ 1 ∧ y * (y - 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l2230_223092


namespace NUMINAMATH_CALUDE_g_is_self_inverse_l2230_223038

-- Define a function f that is symmetric about y=x-1
def f : ℝ → ℝ := sorry

-- Define the property of f being symmetric about y=x-1
axiom f_symmetric : ∀ x y : ℝ, f x = y ↔ f (y + 1) = x + 1

-- Define g in terms of f
def g : ℝ → ℝ := λ x => f (x + 1)

-- State the theorem
theorem g_is_self_inverse : ∀ x : ℝ, g (g x) = x := by sorry

end NUMINAMATH_CALUDE_g_is_self_inverse_l2230_223038


namespace NUMINAMATH_CALUDE_expression_value_l2230_223044

theorem expression_value (a b c : ℤ) (ha : a = 12) (hb : b = 8) (hc : c = 3) :
  ((a - b + c) - (a - (b + c))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2230_223044


namespace NUMINAMATH_CALUDE_dusty_single_layer_purchase_l2230_223071

/-- Represents the cost and quantity of cake slices purchased by Dusty -/
structure CakePurchase where
  single_layer_price : ℕ
  double_layer_price : ℕ
  double_layer_quantity : ℕ
  payment : ℕ
  change : ℕ

/-- Calculates the number of single layer cake slices purchased -/
def single_layer_quantity (purchase : CakePurchase) : ℕ :=
  (purchase.payment - purchase.change - purchase.double_layer_price * purchase.double_layer_quantity) / purchase.single_layer_price

/-- Theorem stating that Dusty bought 7 single layer cake slices -/
theorem dusty_single_layer_purchase :
  let purchase := CakePurchase.mk 4 7 5 100 37
  single_layer_quantity purchase = 7 := by
  sorry

end NUMINAMATH_CALUDE_dusty_single_layer_purchase_l2230_223071


namespace NUMINAMATH_CALUDE_volume_error_percentage_l2230_223000

theorem volume_error_percentage (L W H : ℝ) (L_meas W_meas H_meas : ℝ)
  (h_L : L_meas = 1.08 * L)
  (h_W : W_meas = 1.12 * W)
  (h_H : H_meas = 1.05 * H) :
  let V_true := L * W * H
  let V_calc := L_meas * W_meas * H_meas
  (V_calc - V_true) / V_true * 100 = 25.424 := by
sorry

end NUMINAMATH_CALUDE_volume_error_percentage_l2230_223000


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2230_223014

theorem smallest_positive_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 5 = 4) ∧ 
  (a % 7 = 6) ∧ 
  (∀ b : ℕ, b > 0 ∧ b % 5 = 4 ∧ b % 7 = 6 → a ≤ b) ∧
  (a = 34) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2230_223014


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l2230_223048

/-- Given a function g where g(4) = 8, and h defined as h(x) = (g(x))^2 + 1,
    prove that the sum of coordinates of the point (4, h(4)) is 69. -/
theorem sum_of_coordinates (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 8 → 
  (∀ x, h x = (g x)^2 + 1) → 
  4 + h 4 = 69 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l2230_223048


namespace NUMINAMATH_CALUDE_executive_board_selection_l2230_223079

theorem executive_board_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_selection_l2230_223079


namespace NUMINAMATH_CALUDE_negative_p_exponent_product_l2230_223019

theorem negative_p_exponent_product (p : ℝ) : (-p)^2 * (-p)^3 = -p^5 := by sorry

end NUMINAMATH_CALUDE_negative_p_exponent_product_l2230_223019


namespace NUMINAMATH_CALUDE_monic_quadratic_root_l2230_223061

theorem monic_quadratic_root (x : ℂ) : x^2 + 4*x + 9 = 0 ↔ x = -2 - Complex.I * Real.sqrt 5 ∨ x = -2 + Complex.I * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_monic_quadratic_root_l2230_223061


namespace NUMINAMATH_CALUDE_cube_root_negative_l2230_223046

theorem cube_root_negative (a : ℝ) (k : ℝ) (h : k^3 = a) : 
  ((-a : ℝ)^(1/3 : ℝ) : ℝ) = -k := by sorry

end NUMINAMATH_CALUDE_cube_root_negative_l2230_223046


namespace NUMINAMATH_CALUDE_rainfall_second_week_l2230_223054

/-- Proves that given a total rainfall of 40 inches over two weeks, 
    where the second week's rainfall is 1.5 times the first week's, 
    the rainfall in the second week is 24 inches. -/
theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) : 
  total_rainfall = 40 ∧ ratio = 1.5 → 
  ∃ (first_week : ℝ), 
    first_week + ratio * first_week = total_rainfall ∧ 
    ratio * first_week = 24 := by
  sorry

#check rainfall_second_week

end NUMINAMATH_CALUDE_rainfall_second_week_l2230_223054


namespace NUMINAMATH_CALUDE_apple_loss_fraction_l2230_223050

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

theorem apple_loss_fraction :
  fractionOfLoss 19 18 = 1 / 19 := by
  sorry

end NUMINAMATH_CALUDE_apple_loss_fraction_l2230_223050


namespace NUMINAMATH_CALUDE_remaining_files_count_l2230_223069

def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def initial_document_files : ℕ := 12
def initial_photo_files : ℕ := 30
def initial_app_files : ℕ := 7

def deleted_video_files : ℕ := 15
def deleted_document_files : ℕ := 10
def deleted_photo_files : ℕ := 18
def deleted_app_files : ℕ := 3

theorem remaining_files_count :
  initial_music_files +
  (initial_video_files - deleted_video_files) +
  (initial_document_files - deleted_document_files) +
  (initial_photo_files - deleted_photo_files) +
  (initial_app_files - deleted_app_files) = 28 := by
  sorry

end NUMINAMATH_CALUDE_remaining_files_count_l2230_223069


namespace NUMINAMATH_CALUDE_line_through_circle_center_l2230_223045

/-- The equation of a line passing through the center of a given circle with a specific slope angle -/
theorem line_through_circle_center (x y : ℝ) : 
  (∀ x y, (x + 1)^2 + (y - 2)^2 = 4) →  -- Circle equation
  (∃ m b : ℝ, y = m * x + b ∧ m = 1) →  -- Line with slope 1 (45° angle)
  (∃ x₀ y₀ : ℝ, (x₀ + 1)^2 + (y₀ - 2)^2 = 4 ∧ y₀ = x₀ + 3) →  -- Line passes through circle center
  x - y + 3 = 0  -- Resulting line equation
:= by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l2230_223045


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2230_223065

/-- Proves that the cost of a child ticket is $3.50 given the specified conditions -/
theorem child_ticket_cost (adult_price : ℝ) (total_tickets : ℕ) (total_cost : ℝ) (adult_tickets : ℕ) : ℝ :=
  let child_tickets := total_tickets - adult_tickets
  let child_price := (total_cost - (adult_price * adult_tickets)) / child_tickets
  by
    -- Assuming:
    have h1 : adult_price = 5.50 := by sorry
    have h2 : total_tickets = 21 := by sorry
    have h3 : total_cost = 83.50 := by sorry
    have h4 : adult_tickets = 5 := by sorry

    -- Proof goes here
    sorry

    -- Conclusion
    -- child_price = 3.50

end NUMINAMATH_CALUDE_child_ticket_cost_l2230_223065


namespace NUMINAMATH_CALUDE_M_values_l2230_223027

theorem M_values (a b : ℚ) (h : a * b ≠ 0) :
  let M := (2 * abs a) / a + (3 * b) / abs b
  M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 :=
sorry

end NUMINAMATH_CALUDE_M_values_l2230_223027


namespace NUMINAMATH_CALUDE_max_plate_valid_l2230_223018

-- Define a custom type for characters that can be on a number plate
inductive PlateChar
| Zero
| Six
| Nine
| H
| O

-- Define a function to check if a character looks the same upside down
def looks_same_upside_down (c : PlateChar) : Bool :=
  match c with
  | PlateChar.Zero => true
  | PlateChar.H => true
  | PlateChar.O => true
  | _ => false

-- Define a function to get the upside down version of a character
def upside_down (c : PlateChar) : PlateChar :=
  match c with
  | PlateChar.Six => PlateChar.Nine
  | PlateChar.Nine => PlateChar.Six
  | c => c

-- Define a number plate as a list of PlateChar
def NumberPlate := List PlateChar

-- Define the specific number plate we want to check
def max_plate : NumberPlate :=
  [PlateChar.Six, PlateChar.Zero, PlateChar.H, PlateChar.O, PlateChar.H, PlateChar.Zero, PlateChar.Nine]

-- Theorem: Max's plate is valid when turned upside down
theorem max_plate_valid : 
  max_plate.reverse.map upside_down = max_plate :=
by sorry

end NUMINAMATH_CALUDE_max_plate_valid_l2230_223018


namespace NUMINAMATH_CALUDE_add_1857_minutes_to_noon_l2230_223009

/-- Represents a time of day --/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  is_pm : Bool

/-- Adds minutes to a given time --/
def addMinutes (t : TimeOfDay) (m : Nat) : TimeOfDay :=
  sorry

/-- Checks if two times are equal --/
def timeEqual (t1 t2 : TimeOfDay) : Prop :=
  t1.hours = t2.hours ∧ t1.minutes = t2.minutes ∧ t1.is_pm = t2.is_pm

theorem add_1857_minutes_to_noon :
  let noon := TimeOfDay.mk 12 0 true
  let result := TimeOfDay.mk 6 57 false
  timeEqual (addMinutes noon 1857) result := by
  sorry

end NUMINAMATH_CALUDE_add_1857_minutes_to_noon_l2230_223009


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2230_223034

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicularity for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line (a+2)x+3ay+1=0 -/
def line1 (a : ℝ) : Line :=
  { a := a + 2, b := 3 * a, c := 1 }

/-- The second line (a-2)x+(a+2)y-3=0 -/
def line2 (a : ℝ) : Line :=
  { a := a - 2, b := a + 2, c := -3 }

theorem perpendicular_condition (a : ℝ) :
  (a = -2 → perpendicular (line1 a) (line2 a)) ∧
  (∃ b : ℝ, b ≠ -2 ∧ perpendicular (line1 b) (line2 b)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2230_223034


namespace NUMINAMATH_CALUDE_cards_left_l2230_223083

/-- Given that Nell had 242 cards initially and gave away 136 cards,
    prove that she has 106 cards left. -/
theorem cards_left (initial_cards given_away_cards : ℕ) 
  (h1 : initial_cards = 242)
  (h2 : given_away_cards = 136) :
  initial_cards - given_away_cards = 106 := by
  sorry

end NUMINAMATH_CALUDE_cards_left_l2230_223083


namespace NUMINAMATH_CALUDE_inequality_implies_interval_bound_l2230_223005

theorem inequality_implies_interval_bound 
  (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_interval_bound_l2230_223005


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2230_223078

theorem second_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (first_discount : ℝ)
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : first_discount = 19.954259576901087)
  : ∃ (second_discount : ℝ), 
    abs (second_discount - 12.552) < 0.001 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2230_223078


namespace NUMINAMATH_CALUDE_coopers_age_l2230_223003

/-- Given the ages of four people with specific relationships, prove Cooper's age --/
theorem coopers_age (cooper dante maria emily : ℕ) : 
  cooper + dante + maria + emily = 62 →
  dante = 2 * cooper →
  maria = dante + 1 →
  emily = 3 * cooper →
  cooper = 8 :=
by sorry

end NUMINAMATH_CALUDE_coopers_age_l2230_223003


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l2230_223066

theorem slope_of_line_from_equation (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : (3 : ℝ) / x₁ + (4 : ℝ) / y₁ = 0)
  (h₃ : (3 : ℝ) / x₂ + (4 : ℝ) / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -(4 : ℝ) / 3 := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l2230_223066


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2230_223095

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 63 = (X - 3) * q + 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2230_223095


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2230_223085

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b = 4 → (1 / x + 1 / y) ≤ (1 / a + 1 / b) ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 4 ∧ 1 / x + 1 / y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2230_223085


namespace NUMINAMATH_CALUDE_cricket_bat_price_l2230_223002

theorem cricket_bat_price (profit_A_to_B profit_B_to_C profit_C_to_D final_price : ℝ)
  (h1 : profit_A_to_B = 0.2)
  (h2 : profit_B_to_C = 0.25)
  (h3 : profit_C_to_D = 0.3)
  (h4 : final_price = 400) :
  ∃ (original_price : ℝ),
    original_price = final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D)) :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l2230_223002


namespace NUMINAMATH_CALUDE_garden_length_l2230_223080

/-- Proves that the length of the larger garden is 90 yards given the conditions -/
theorem garden_length (w : ℝ) (l : ℝ) : 
  l = 3 * w →  -- larger garden length is three times its width
  360 = 2 * l + 2 * w + 2 * (w / 2) + 2 * (l / 2) →  -- total fencing equals 360 yards
  l = 90 := by
  sorry

#check garden_length

end NUMINAMATH_CALUDE_garden_length_l2230_223080


namespace NUMINAMATH_CALUDE_adams_earnings_l2230_223035

/-- Adam's daily earnings problem -/
theorem adams_earnings (daily_earnings : ℝ) : 
  (daily_earnings * 0.9 * 30 = 1080) → daily_earnings = 40 := by
  sorry

end NUMINAMATH_CALUDE_adams_earnings_l2230_223035


namespace NUMINAMATH_CALUDE_range_of_f_l2230_223067

def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-5) 17 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2230_223067


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_l2230_223072

-- Define the concept of "like terms" for our specific case
def are_like_terms (m n : ℤ) : Prop :=
  m + 3 = 4 ∧ n + 3 = 1

-- State the theorem
theorem like_terms_imply_sum (m n : ℤ) :
  are_like_terms m n → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_l2230_223072


namespace NUMINAMATH_CALUDE_system_solution_l2230_223008

theorem system_solution (m : ℤ) : 
  (∃ (x y : ℝ), 
    x - 2*y = m ∧ 
    2*x + 3*y = 2*m - 3 ∧ 
    3*x + y ≥ 0 ∧ 
    x + 5*y < 0) ↔ 
  (m = 1 ∨ m = 2) := by sorry

end NUMINAMATH_CALUDE_system_solution_l2230_223008


namespace NUMINAMATH_CALUDE_x_squared_minus_2x_minus_3_is_quadratic_l2230_223049

/-- Definition of a quadratic equation -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = 0 → true

/-- The equation x² - 2x - 3 = 0 is a quadratic equation -/
theorem x_squared_minus_2x_minus_3_is_quadratic :
  is_quadratic_equation 1 (-2) (-3) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_2x_minus_3_is_quadratic_l2230_223049


namespace NUMINAMATH_CALUDE_zoo_visitors_l2230_223081

theorem zoo_visitors (sandwiches_per_person : ℝ) (total_sandwiches : ℕ) :
  sandwiches_per_person = 3.0 →
  total_sandwiches = 657 →
  ↑total_sandwiches / sandwiches_per_person = 219 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_l2230_223081


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2230_223058

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0) → 
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ 
  (c > 0 ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2230_223058


namespace NUMINAMATH_CALUDE_pet_shop_stock_worth_l2230_223004

/-- The total worth of the stock in a pet shop -/
def stock_worth (num_puppies num_kittens puppy_price kitten_price : ℕ) : ℕ :=
  num_puppies * puppy_price + num_kittens * kitten_price

/-- Theorem stating that the stock worth is 100 given the specific conditions -/
theorem pet_shop_stock_worth :
  stock_worth 2 4 20 15 = 100 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_stock_worth_l2230_223004


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l2230_223097

theorem cubic_roots_sum_squares (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 12 = 0 →
  q^3 - 15*q^2 + 25*q - 12 = 0 →
  r^3 - 15*r^2 + 25*r - 12 = 0 →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l2230_223097


namespace NUMINAMATH_CALUDE_fraction_simplification_l2230_223089

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a^2 / b^2
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2230_223089


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2230_223036

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 4 ∧
  point.x = -2 ∧ point.y = -3 →
  ∃ (l : Line), 
    pointOnLine point l ∧ 
    perpendicular l given_line ∧
    l.a = 3 ∧ l.b = 2 ∧ l.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2230_223036


namespace NUMINAMATH_CALUDE_difference_between_fractions_l2230_223098

theorem difference_between_fractions (n : ℝ) (h : n = 140) : (4/5 * n) - (65/100 * n) = 21 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_fractions_l2230_223098


namespace NUMINAMATH_CALUDE_triangle_area_l2230_223077

-- Define the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept_1 : ℝ := -3
def x_intercept_2 : ℝ := 4

-- Define the y-intercept
def y_intercept : ℝ := f 0

-- Theorem statement
theorem triangle_area : 
  let base := x_intercept_2 - x_intercept_1
  let height := y_intercept
  (1 / 2 : ℝ) * base * height = 168 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2230_223077


namespace NUMINAMATH_CALUDE_derivative_fifth_root_cube_l2230_223031

theorem derivative_fifth_root_cube (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => x^(3/5)) x = 3 / (5 * x^(2/5)) :=
sorry

end NUMINAMATH_CALUDE_derivative_fifth_root_cube_l2230_223031


namespace NUMINAMATH_CALUDE_radical_simplification_l2230_223090

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (3 * p^5) * Real.sqrt (4 * p^2) / Real.sqrt (2 * p) = 6 * p^(9/2) * Real.sqrt (5/2) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l2230_223090


namespace NUMINAMATH_CALUDE_diamond_value_l2230_223013

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Converts a number in base 9 to base 10 -/
def base9To10 (d : Digit) : ℕ :=
  9 * d.val + 5

/-- Converts a number in base 10 to itself -/
def base10To10 (d : Digit) : ℕ :=
  10 * d.val + 2

theorem diamond_value :
  ∃ (d : Digit), base9To10 d = base10To10 d ∧ d.val = 3 := by sorry

end NUMINAMATH_CALUDE_diamond_value_l2230_223013


namespace NUMINAMATH_CALUDE_possible_values_for_e_l2230_223029

def is_digit (n : ℕ) : Prop := n < 10

def distinct (a b c e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ e ∧ b ≠ c ∧ b ≠ e ∧ c ≠ e

def subtraction_equation (a b c e : ℕ) : Prop :=
  10000 * a + 1000 * b + 100 * b + 10 * c + b -
  (10000 * b + 1000 * c + 100 * a + 10 * c + b) =
  10000 * c + 1000 * e + 100 * b + 10 * e + e

theorem possible_values_for_e :
  ∃ (s : Finset ℕ),
    (∀ e ∈ s, is_digit e) ∧
    (∀ e ∈ s, ∃ (a b c : ℕ),
      is_digit a ∧ is_digit b ∧ is_digit c ∧
      distinct a b c e ∧
      subtraction_equation a b c e) ∧
    s.card = 10 :=
sorry

end NUMINAMATH_CALUDE_possible_values_for_e_l2230_223029


namespace NUMINAMATH_CALUDE_cost_price_per_meter_correct_cost_price_fabric_C_is_120_l2230_223099

/-- Calculates the cost price per meter of fabric given the selling price, number of meters, and profit per meter. -/
def costPricePerMeter (sellingPrice : ℚ) (meters : ℚ) (profitPerMeter : ℚ) : ℚ :=
  (sellingPrice - meters * profitPerMeter) / meters

/-- Represents the fabric types and their properties -/
structure FabricType where
  name : String
  sellingPrice : ℚ
  meters : ℚ
  profitPerMeter : ℚ

/-- Theorem stating that the cost price per meter calculation is correct for all fabric types -/
theorem cost_price_per_meter_correct (fabric : FabricType) :
  costPricePerMeter fabric.sellingPrice fabric.meters fabric.profitPerMeter =
  (fabric.sellingPrice - fabric.meters * fabric.profitPerMeter) / fabric.meters :=
by
  sorry

/-- The three fabric types given in the problem -/
def fabricA : FabricType := ⟨"A", 6000, 45, 12⟩
def fabricB : FabricType := ⟨"B", 10800, 60, 15⟩
def fabricC : FabricType := ⟨"C", 3900, 30, 10⟩

/-- Theorem stating that the cost price per meter for fabric C is 120 -/
theorem cost_price_fabric_C_is_120 :
  costPricePerMeter fabricC.sellingPrice fabricC.meters fabricC.profitPerMeter = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_correct_cost_price_fabric_C_is_120_l2230_223099


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l2230_223001

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, 
    a > 0 ∧ 
    a ≠ 1 ∧ 
    (∀ x y : ℝ, x < y → f a x < f a y) →
    a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l2230_223001


namespace NUMINAMATH_CALUDE_circle_symmetry_min_value_l2230_223059

/-- The minimum value of 1/a + 3/b for a circle symmetric to a line --/
theorem circle_symmetry_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_symmetry : ∃ (x y : ℝ), x^2 + y^2 + 2*x - 6*y + 1 = 0 ∧ a*x - b*y + 3 = 0) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∃ (x y : ℝ), x^2 + y^2 + 2*x - 6*y + 1 = 0 ∧ a'*x - b'*y + 3 = 0) → 
    1/a + 3/b ≤ 1/a' + 3/b') ∧
  (1/a + 3/b = 16/3) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_min_value_l2230_223059


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2230_223051

/-- Calculates the remaining money after John's pizza and drink purchase. -/
def remaining_money (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 2 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Proves that John's remaining money is equal to 50 - 8q. -/
theorem johns_remaining_money (q : ℝ) : remaining_money q = 50 - 8 * q := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2230_223051


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_error_l2230_223006

theorem rectangular_prism_volume_error (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let actual_volume := a * b * c
  let measured_volume := (a * 1.08) * (b * 0.90) * (c * 0.94)
  let error_percentage := (measured_volume - actual_volume) / actual_volume * 100
  error_percentage = -2.728 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_error_l2230_223006


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2230_223007

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 - 2*x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2230_223007


namespace NUMINAMATH_CALUDE_cubic_values_quadratic_polynomial_l2230_223041

theorem cubic_values_quadratic_polynomial 
  (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c) :
  ∃ (p q r : ℤ) (x₁ x₂ x₃ : ℤ), 
    p > 0 ∧ 
    (p * x₁^2 + q * x₁ + r = a^3) ∧
    (p * x₂^2 + q * x₂ + r = b^3) ∧
    (p * x₃^2 + q * x₃ + r = c^3) :=
sorry

end NUMINAMATH_CALUDE_cubic_values_quadratic_polynomial_l2230_223041


namespace NUMINAMATH_CALUDE_equation_solutions_l2230_223021

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 3 ∧ x₂ = 5) ∧ 
  (∀ x : ℝ, (x - 2)^6 + (x - 6)^6 = 64 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2230_223021


namespace NUMINAMATH_CALUDE_triangle_inradius_inequality_l2230_223024

/-- 
For any triangle ABC with side lengths a, b, c and inradius r, 
the inequality 24√3 r³ ≤ (-a+b+c)(a-b+c)(a+b-c) holds.
-/
theorem triangle_inradius_inequality (a b c r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inradius : r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c))) :
  24 * Real.sqrt 3 * r^3 ≤ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_inequality_l2230_223024


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2230_223017

/-- The longest segment that can fit inside a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2230_223017


namespace NUMINAMATH_CALUDE_scalene_to_right_triangle_l2230_223062

theorem scalene_to_right_triangle 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) :
  ∃ x : ℝ, (a + x)^2 + (b + x)^2 = (c + x)^2 :=
sorry

end NUMINAMATH_CALUDE_scalene_to_right_triangle_l2230_223062


namespace NUMINAMATH_CALUDE_eraser_buyers_difference_l2230_223063

theorem eraser_buyers_difference : ∀ (price : ℕ) (fifth_graders fourth_graders : ℕ),
  price > 0 →
  fifth_graders * price = 325 →
  fourth_graders * price = 460 →
  fourth_graders = 40 →
  fourth_graders - fifth_graders = 27 := by
sorry

end NUMINAMATH_CALUDE_eraser_buyers_difference_l2230_223063


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2230_223075

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2230_223075


namespace NUMINAMATH_CALUDE_mean_median_difference_l2230_223057

/-- Represents the score distribution in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_75_percent : ℚ
  score_82_percent : ℚ
  score_87_percent : ℚ
  score_90_percent : ℚ
  score_98_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (sd : ScoreDistribution) : ℚ :=
  (75 * sd.score_75_percent + 82 * sd.score_82_percent + 87 * sd.score_87_percent +
   90 * sd.score_90_percent + 98 * sd.score_98_percent) / 100

/-- Calculates the median score given a score distribution -/
def median_score (sd : ScoreDistribution) : ℚ := 87

/-- The main theorem stating the difference between mean and median scores -/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.total_students = 10)
  (h2 : sd.score_75_percent = 15)
  (h3 : sd.score_82_percent = 10)
  (h4 : sd.score_87_percent = 40)
  (h5 : sd.score_90_percent = 20)
  (h6 : sd.score_98_percent = 15) :
  |mean_score sd - median_score sd| = 9 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2230_223057


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_2310_l2230_223073

theorem sum_of_prime_factors_2310 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (2310 + 1))) id) = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_2310_l2230_223073


namespace NUMINAMATH_CALUDE_function_properties_l2230_223068

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : φ ∈ Set.Ioo (-π) 0)
  (h2 : ∀ x, f x φ = f (π/4 - x) φ) :
  φ = -3*π/4 ∧
  (∀ k : ℤ, ∀ x : ℝ, 5*π/8 + k*π ≤ x ∧ x ≤ 9*π/8 + k*π → 
    ∀ y : ℝ, x < y → f y φ < f x φ) ∧
  Set.range (fun x => f x φ) = Set.Icc (-3) (3*Real.sqrt 2/2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2230_223068


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2230_223047

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) :
  ∀ a b : ℝ, a > 0 → b > 0 → (a + 1) * (b + 1) = 9 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 9 ∧ x + y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2230_223047


namespace NUMINAMATH_CALUDE_cistern_leak_emptying_time_l2230_223026

theorem cistern_leak_emptying_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 8) 
  (h2 : leak_fill_time = 10) : 
  (1 / (1 / leak_fill_time - 1 / normal_fill_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_emptying_time_l2230_223026


namespace NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l2230_223020

/-- Cost function for Plan A -/
def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

/-- Cost function for Plan B -/
def costB (x : ℕ) : ℕ := 2500 + 4 * x

/-- Theorem stating the minimum number of minutes for Plan B to be cheaper -/
theorem min_minutes_for_plan_b_cheaper :
  ∀ x : ℕ, x < 301 → costA x ≤ costB x ∧
  ∀ y : ℕ, y ≥ 301 → costB y < costA y := by
  sorry

#check min_minutes_for_plan_b_cheaper

end NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l2230_223020


namespace NUMINAMATH_CALUDE_swallow_flock_capacity_l2230_223016

/-- Represents the carrying capacity of different types of swallows -/
structure SwallowCapacity where
  american : ℕ
  european : ℕ
  african : ℕ

/-- Represents the composition of a flock of swallows -/
structure SwallowFlock where
  american : ℕ
  european : ℕ
  african : ℕ

/-- Calculates the total number of swallows in a flock -/
def totalSwallows (flock : SwallowFlock) : ℕ :=
  flock.american + flock.european + flock.african

/-- Calculates the maximum weight a flock can carry -/
def maxCarryWeight (capacity : SwallowCapacity) (flock : SwallowFlock) : ℕ :=
  flock.american * capacity.american +
  flock.european * capacity.european +
  flock.african * capacity.african

/-- Theorem stating the maximum carrying capacity of a specific flock of swallows -/
theorem swallow_flock_capacity
  (capacity : SwallowCapacity)
  (flock : SwallowFlock)
  (h1 : capacity.american = 5)
  (h2 : capacity.european = 10)
  (h3 : capacity.african = 15)
  (h4 : flock.american = 45)
  (h5 : flock.european = 30)
  (h6 : flock.african = 75)
  (h7 : totalSwallows flock = 150)
  (h8 : flock.american * 2 = flock.european * 3)
  (h9 : flock.american * 5 = flock.african * 3) :
  maxCarryWeight capacity flock = 1650 := by
  sorry


end NUMINAMATH_CALUDE_swallow_flock_capacity_l2230_223016


namespace NUMINAMATH_CALUDE_probability_all_black_is_correct_l2230_223093

def urn_black_balls : ℕ := 10
def urn_white_balls : ℕ := 5
def total_balls : ℕ := urn_black_balls + urn_white_balls
def drawn_balls : ℕ := 2

def probability_all_black : ℚ := (urn_black_balls.choose drawn_balls) / (total_balls.choose drawn_balls)

theorem probability_all_black_is_correct :
  probability_all_black = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_probability_all_black_is_correct_l2230_223093


namespace NUMINAMATH_CALUDE_parkingLotSpaces_l2230_223056

/-- Represents a car parking lot with three sections. -/
structure ParkingLot where
  section1 : ℕ
  section2 : ℕ
  section3 : ℕ

/-- Calculates the total number of spaces in the parking lot. -/
def totalSpaces (lot : ParkingLot) : ℕ :=
  lot.section1 + lot.section2 + lot.section3

/-- Theorem stating the total number of spaces in the parking lot. -/
theorem parkingLotSpaces : ∃ (lot : ParkingLot), 
  lot.section1 = 320 ∧ 
  lot.section2 = 440 ∧ 
  lot.section2 = lot.section3 + 200 ∧
  totalSpaces lot = 1000 := by
  sorry

end NUMINAMATH_CALUDE_parkingLotSpaces_l2230_223056


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2230_223028

/-- The x-intercept of the line -4x + 6y = 24 is (-6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  -4 * x + 6 * y = 24 → y = 0 → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2230_223028


namespace NUMINAMATH_CALUDE_martin_family_ice_cream_l2230_223086

/-- The cost of ice cream for the Martin family at the mall --/
def ice_cream_cost (double_scoop_price : ℕ) : Prop :=
  let kiddie_scoop_price : ℕ := 3
  let regular_scoop_price : ℕ := 4
  let num_regular_scoops : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie_scoops : ℕ := 2   -- Two children
  let num_double_scoops : ℕ := 3   -- Three teenage children
  let total_cost : ℕ := 32
  (num_regular_scoops * regular_scoop_price +
   num_kiddie_scoops * kiddie_scoop_price +
   num_double_scoops * double_scoop_price) = total_cost

theorem martin_family_ice_cream : ice_cream_cost 6 := by
  sorry

end NUMINAMATH_CALUDE_martin_family_ice_cream_l2230_223086


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l2230_223055

theorem exponential_equation_solution :
  ∃ m : ℤ, (3 : ℝ)^m * 9^m = 81^(m - 24) ∧ m = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l2230_223055


namespace NUMINAMATH_CALUDE_shaded_area_four_circles_l2230_223082

/-- The area of the shaded region formed by the intersection of four circles -/
theorem shaded_area_four_circles (r : ℝ) (h : r = 5) : 
  let circle_area := π * r^2
  let quarter_circle_area := circle_area / 4
  let triangle_area := r^2 / 2
  let shaded_segment := quarter_circle_area - triangle_area
  4 * shaded_segment = 25 * π - 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_four_circles_l2230_223082


namespace NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l2230_223025

-- Define a type for data sets
structure DataSet where
  -- Add necessary fields for a data set
  nonEmpty : Bool

-- Define a predicate for whether a regression equation exists for a data set
def hasRegressionEquation (d : DataSet) : Prop := sorry

-- Theorem stating that not every data set has a regression equation
theorem not_all_data_sets_have_regression_equation :
  ¬ ∀ (d : DataSet), hasRegressionEquation d := by
  sorry


end NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l2230_223025


namespace NUMINAMATH_CALUDE_blue_pens_to_pencils_ratio_l2230_223088

theorem blue_pens_to_pencils_ratio 
  (blue_pens black_pens red_pens pencils : ℕ) : 
  black_pens = blue_pens + 10 →
  pencils = 8 →
  red_pens = pencils - 2 →
  blue_pens + black_pens + red_pens = 48 →
  blue_pens = 2 * pencils :=
by sorry

end NUMINAMATH_CALUDE_blue_pens_to_pencils_ratio_l2230_223088


namespace NUMINAMATH_CALUDE_sin_1050_degrees_l2230_223091

theorem sin_1050_degrees : Real.sin (1050 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1050_degrees_l2230_223091


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_odd_l2230_223039

theorem consecutive_integers_square_sum_odd (a b c M : ℤ) : 
  (a = b + 1 ∨ b = a + 1) →  -- a and b are consecutive integers
  c = a * b →               -- c = ab
  M^2 = a^2 + b^2 + c^2 →   -- M^2 = a^2 + b^2 + c^2
  Odd (M^2) :=              -- M^2 is an odd number
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_odd_l2230_223039


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2230_223087

theorem expression_simplification_and_evaluation :
  let x := Real.tan (45 * π / 180) + Real.cos (30 * π / 180)
  (x / (x^2 - 1)) * ((x - 1) / x - 2) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2230_223087


namespace NUMINAMATH_CALUDE_thursday_productivity_l2230_223070

/-- Represents the relationship between cups of coffee and lines of code written --/
structure CoffeeProductivity where
  k : ℝ  -- Proportionality constant
  coffee_to_code : ℝ → ℝ  -- Function that converts cups of coffee to lines of code

/-- Given the conditions from the problem, prove that the programmer wrote 250 lines of code on Thursday --/
theorem thursday_productivity (cp : CoffeeProductivity) 
  (h1 : cp.coffee_to_code 3 = 150)  -- Wednesday's data
  (h2 : ∀ c, cp.coffee_to_code c = cp.k * c)  -- Direct proportionality
  : cp.coffee_to_code 5 = 250 := by
  sorry

#check thursday_productivity

end NUMINAMATH_CALUDE_thursday_productivity_l2230_223070


namespace NUMINAMATH_CALUDE_bed_frame_cost_l2230_223010

theorem bed_frame_cost (bed_price : ℝ) (total_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) :
  bed_price = 10 * total_price →
  discount_rate = 0.2 →
  final_price = (1 - discount_rate) * (bed_price + total_price) →
  final_price = 660 →
  total_price = 75 := by
sorry

end NUMINAMATH_CALUDE_bed_frame_cost_l2230_223010


namespace NUMINAMATH_CALUDE_x_value_proof_l2230_223012

theorem x_value_proof (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2230_223012


namespace NUMINAMATH_CALUDE_tim_score_l2230_223040

def single_line_score : ℕ := 1000
def tetris_multiplier : ℕ := 8
def consecutive_multiplier : ℕ := 2

def calculate_score (singles : ℕ) (regular_tetrises : ℕ) (consecutive_tetrises : ℕ) : ℕ :=
  singles * single_line_score +
  regular_tetrises * (single_line_score * tetris_multiplier) +
  consecutive_tetrises * (single_line_score * tetris_multiplier * consecutive_multiplier)

theorem tim_score : calculate_score 6 2 2 = 54000 := by
  sorry

end NUMINAMATH_CALUDE_tim_score_l2230_223040


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2230_223084

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2230_223084


namespace NUMINAMATH_CALUDE_artichoke_dip_theorem_l2230_223076

/-- The amount of money Hakeem has to spend on artichokes -/
def budget : ℚ := 15

/-- The cost of one artichoke -/
def artichoke_cost : ℚ := 5/4

/-- The number of artichokes needed to make a batch of dip -/
def artichokes_per_batch : ℕ := 3

/-- The amount of dip (in ounces) produced from one batch -/
def dip_per_batch : ℚ := 5

/-- The maximum amount of dip (in ounces) that can be made with the given budget -/
def max_dip : ℚ := 20

theorem artichoke_dip_theorem :
  (budget / artichoke_cost).floor * (dip_per_batch / artichokes_per_batch) = max_dip :=
sorry

end NUMINAMATH_CALUDE_artichoke_dip_theorem_l2230_223076


namespace NUMINAMATH_CALUDE_bills_weights_theorem_l2230_223032

/-- The total weight of sand in two jugs filled partially with sand -/
def total_weight (jug_capacity : ℝ) (fill_percentage : ℝ) (num_jugs : ℕ) (sand_density : ℝ) : ℝ :=
  jug_capacity * fill_percentage * (num_jugs : ℝ) * sand_density

/-- Theorem stating the total weight of sand in Bill's improvised weights -/
theorem bills_weights_theorem :
  total_weight 2 0.7 2 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bills_weights_theorem_l2230_223032


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l2230_223022

theorem stratified_sampling_third_year_count 
  (total_sample : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (third_year : ℕ) 
  (h1 : total_sample = 200)
  (h2 : first_year = 1300)
  (h3 : second_year = 1200)
  (h4 : third_year = 1500) :
  (third_year * total_sample) / (first_year + second_year + third_year) = 75 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l2230_223022


namespace NUMINAMATH_CALUDE_rhombus_area_l2230_223074

/-- The area of a rhombus with vertices at (0, 3.5), (11, 0), (0, -3.5), and (-11, 0) is 77 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (11, 0), (0, -3.5), (-11, 0)]
  let vertical_diagonal : ℝ := |3.5 - (-3.5)|
  let horizontal_diagonal : ℝ := |11 - (-11)|
  let area : ℝ := (vertical_diagonal * horizontal_diagonal) / 2
  area = 77 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2230_223074


namespace NUMINAMATH_CALUDE_parabola_translation_up_one_unit_l2230_223023

/-- Represents a parabola in the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b + d }

theorem parabola_translation_up_one_unit :
  let original := Parabola.mk 3 0
  let translated := translate_vertical original 1
  translated = Parabola.mk 3 1 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_up_one_unit_l2230_223023


namespace NUMINAMATH_CALUDE_lisa_scenery_photos_l2230_223096

-- Define the variables
def animal_photos : ℕ := 10
def flower_photos : ℕ := 3 * animal_photos
def scenery_photos : ℕ := flower_photos - 10
def total_photos : ℕ := 45

-- Theorem to prove
theorem lisa_scenery_photos :
  scenery_photos = 20 ∧
  animal_photos + flower_photos + scenery_photos = total_photos :=
by sorry

end NUMINAMATH_CALUDE_lisa_scenery_photos_l2230_223096


namespace NUMINAMATH_CALUDE_kerosene_mixture_problem_l2230_223064

theorem kerosene_mixture_problem (first_liquid_percentage : ℝ) 
  (first_liquid_parts : ℝ) (second_liquid_parts : ℝ) (mixture_percentage : ℝ) :
  first_liquid_percentage = 25 →
  first_liquid_parts = 6 →
  second_liquid_parts = 4 →
  mixture_percentage = 27 →
  let total_parts := first_liquid_parts + second_liquid_parts
  let second_liquid_percentage := 
    (mixture_percentage * total_parts - first_liquid_percentage * first_liquid_parts) / second_liquid_parts
  second_liquid_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_mixture_problem_l2230_223064


namespace NUMINAMATH_CALUDE_distributive_property_l2230_223030

theorem distributive_property (m : ℝ) : m * (m - 1) = m^2 - m := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l2230_223030


namespace NUMINAMATH_CALUDE_intersection_when_m_is_5_intersection_equals_B_iff_l2230_223011

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Statement 1: When m = 5, A ∩ B = {x | 6 ≤ x < 9}
theorem intersection_when_m_is_5 : 
  A ∩ B 5 = {x : ℝ | 6 ≤ x ∧ x < 9} := by sorry

-- Statement 2: A ∩ B = B if and only if m ∈ (-∞, 5)
theorem intersection_equals_B_iff :
  ∀ m : ℝ, A ∩ B m = B m ↔ m < 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_5_intersection_equals_B_iff_l2230_223011


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l2230_223033

/-- Given a polynomial g(x) with three distinct roots that are also roots of f(x),
    prove that f(2) = -16342.5 -/
theorem polynomial_root_problem (p q d : ℝ) : 
  let g : ℝ → ℝ := λ x => x^3 + p*x^2 + 2*x + 15
  let f : ℝ → ℝ := λ x => x^4 + 2*x^3 + q*x^2 + 150*x + d
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  f 2 = -16342.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l2230_223033


namespace NUMINAMATH_CALUDE_necessary_condition_for_false_proposition_l2230_223094

theorem necessary_condition_for_false_proposition (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 1 ≤ 0) → (-2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_false_proposition_l2230_223094
