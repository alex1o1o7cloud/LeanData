import Mathlib

namespace NUMINAMATH_CALUDE_erasers_ratio_l2402_240270

def erasers_problem (hanna rachel tanya_red tanya_total : ℕ) : Prop :=
  hanna = 4 ∧
  tanya_total = 20 ∧
  hanna = 2 * rachel ∧
  rachel = tanya_red / 2 - 3 ∧
  tanya_red ≤ tanya_total

theorem erasers_ratio :
  ∀ hanna rachel tanya_red tanya_total,
    erasers_problem hanna rachel tanya_red tanya_total →
    (tanya_red : ℚ) / tanya_total = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_erasers_ratio_l2402_240270


namespace NUMINAMATH_CALUDE_problem_solution_l2402_240276

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.exp x - t * (x + 1)

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f t x + t / Real.exp x

theorem problem_solution :
  (∀ t : ℝ, (∀ x : ℝ, x > 0 → f t x ≥ 0) → t ≤ 1) ∧
  (∀ t : ℝ, t ≤ -1 →
    (∀ m : ℝ, (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ →
      y₁ = g t x₁ → y₂ = g t x₂ → (y₂ - y₁) / (x₂ - x₁) > m) → m < 3)) ∧
  (∀ n : ℕ, n > 0 →
    Real.log (1 + n) < (Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℝ))) ∧
    (Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℝ))) ≤ 1 + Real.log n) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2402_240276


namespace NUMINAMATH_CALUDE_survey_respondents_l2402_240296

/-- Represents the number of people preferring each brand in a survey -/
structure BrandPreference where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents given brand preferences -/
def totalRespondents (pref : BrandPreference) : ℕ :=
  pref.x + pref.y + pref.z

/-- Theorem stating the total number of respondents in the survey -/
theorem survey_respondents :
  ∀ (pref : BrandPreference),
    pref.x = 150 ∧
    5 * pref.z = pref.x ∧
    3 * pref.z = pref.y →
    totalRespondents pref = 270 := by
  sorry


end NUMINAMATH_CALUDE_survey_respondents_l2402_240296


namespace NUMINAMATH_CALUDE_inequality_theorem_range_theorem_l2402_240234

-- Theorem 1
theorem inequality_theorem (x y : ℝ) : x^2 + 2*y^2 ≥ 2*x*y + 2*y - 1 := by
  sorry

-- Theorem 2
theorem range_theorem (a b : ℝ) (h1 : -2 < a ∧ a ≤ 3) (h2 : 1 ≤ b ∧ b < 2) :
  -1 < a + b ∧ a + b < 5 ∧ -10 < 2*a - 3*b ∧ 2*a - 3*b ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_range_theorem_l2402_240234


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2402_240278

theorem least_number_with_remainder (n : ℕ) : n = 256 →
  (∃ k : ℕ, n = 18 * k + 4) ∧
  (∀ m : ℕ, m < n → ¬(∃ j : ℕ, m = 18 * j + 4)) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2402_240278


namespace NUMINAMATH_CALUDE_cat_mouse_position_after_323_moves_l2402_240263

-- Define the positions for the cat
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

-- Define the positions for the mouse
inductive MousePosition
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle
  | TopLeft

-- Function to calculate cat's position after n moves
def catPositionAfterMoves (n : ℕ) : CatPosition :=
  match n % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

-- Function to calculate mouse's position after n moves
def mousePositionAfterMoves (n : ℕ) : MousePosition :=
  match n % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem cat_mouse_position_after_323_moves :
  (catPositionAfterMoves 323 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 323 = MousePosition.RightMiddle) :=
by sorry

end NUMINAMATH_CALUDE_cat_mouse_position_after_323_moves_l2402_240263


namespace NUMINAMATH_CALUDE_complex_in_third_quadrant_l2402_240295

def complex (a b : ℝ) := a + b * Complex.I

theorem complex_in_third_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) * z = Complex.I ^ 3) :
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_third_quadrant_l2402_240295


namespace NUMINAMATH_CALUDE_sixteen_million_scientific_notation_l2402_240262

/-- Given a number n, returns true if it's in scientific notation -/
def is_scientific_notation (n : ℝ) : Prop :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ n = a * (10 : ℝ) ^ b

theorem sixteen_million_scientific_notation :
  is_scientific_notation 16000000 ∧
  16000000 = 1.6 * (10 : ℝ) ^ 7 :=
sorry

end NUMINAMATH_CALUDE_sixteen_million_scientific_notation_l2402_240262


namespace NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_l2402_240224

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Theorem for the ellipse
theorem ellipse_properties :
  ∀ x y : ℝ, ellipse x y →
  (∃ c : ℝ, c = Real.sqrt 2 ∧ 
   ((x - c)^2 + y^2 = 4 ∨ (x + c)^2 + y^2 = 4)) ∧
  (x = -2 * Real.sqrt 2 ∨ x = 2 * Real.sqrt 2) :=
sorry

-- Theorem for the hyperbola
theorem hyperbola_properties :
  ∀ x y : ℝ, hyperbola x y →
  (hyperbola (Real.sqrt 2) 2) ∧
  (∃ k : ℝ, k = 2 ∧ (y = k*x ∨ y = -k*x)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_l2402_240224


namespace NUMINAMATH_CALUDE_yonderland_license_plates_l2402_240268

/-- The number of possible letters in each position of a license plate. -/
def numLetters : ℕ := 26

/-- The number of possible digits in each position of a license plate. -/
def numDigits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def numLetterPositions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def numDigitPositions : ℕ := 4

/-- The total number of valid license plates in Yonderland. -/
def totalLicensePlates : ℕ := numLetters ^ numLetterPositions * numDigits ^ numDigitPositions

theorem yonderland_license_plates : totalLicensePlates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_yonderland_license_plates_l2402_240268


namespace NUMINAMATH_CALUDE_cards_distribution_l2402_240207

/-- Given 60 cards dealt to 9 people as evenly as possible, 
    the number of people with fewer than 7 cards is 3. -/
theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
    (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  people_with_fewer = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2402_240207


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2402_240202

theorem linear_equation_condition (m : ℤ) : (|m| - 2 = 1 ∧ m - 3 ≠ 0) ↔ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2402_240202


namespace NUMINAMATH_CALUDE_circle_properties_l2402_240269

/-- Given a circle C with equation x^2 + y^2 - 2x - 2y - 2 = 0,
    prove that its radius is 2 and its center is at (1, 1) -/
theorem circle_properties (x y : ℝ) :
  x^2 + y^2 - 2*x - 2*y - 2 = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 1) ∧ radius = 2 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2402_240269


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2402_240213

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Define an isosceles triangle type
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  h_isosceles : side1 = side2
  h_equation1 : equation side1
  h_equation2 : equation side2
  h_triangle_inequality : side1 + side2 > base ∧ side1 + base > side2 ∧ side2 + base > side1

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.side1 + t.side2 + t.base = 10 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2402_240213


namespace NUMINAMATH_CALUDE_factorization_equality_l2402_240232

theorem factorization_equality (a b : ℝ) :
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2 = (3*a^2 + b^2) * (a^2 + 3*b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2402_240232


namespace NUMINAMATH_CALUDE_parabola_focus_l2402_240211

/-- The parabola defined by the equation y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The parabola opens upwards -/
def opens_upwards (p : (ℝ → ℝ → Prop)) : Prop := sorry

/-- The focus lies on the y-axis -/
def focus_on_y_axis (f : Focus) : Prop := f.x = 0

/-- Theorem stating that the focus of the given parabola is at (0, 1) -/
theorem parabola_focus :
  ∃ (f : Focus),
    (∀ x y, parabola x y → opens_upwards parabola) ∧
    focus_on_y_axis f ∧
    f.x = 0 ∧ f.y = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2402_240211


namespace NUMINAMATH_CALUDE_multiplication_closed_in_P_l2402_240204

def P : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^2}

theorem multiplication_closed_in_P : 
  ∀ a b : ℕ, a ∈ P → b ∈ P → (a * b) ∈ P := by
  sorry

end NUMINAMATH_CALUDE_multiplication_closed_in_P_l2402_240204


namespace NUMINAMATH_CALUDE_total_students_l2402_240264

theorem total_students (group_a group_b : ℕ) : 
  (group_a : ℚ) / group_b = 3 / 2 →
  (group_a : ℚ) * (1 / 10) - (group_b : ℚ) * (1 / 5) = 190 →
  group_b = 650 →
  group_a + group_b = 1625 := by
sorry


end NUMINAMATH_CALUDE_total_students_l2402_240264


namespace NUMINAMATH_CALUDE_committee_selection_with_fixed_member_l2402_240261

/-- The number of ways to select a committee with a fixed member -/
def select_committee (total : ℕ) (committee_size : ℕ) (fixed_members : ℕ) : ℕ :=
  Nat.choose (total - fixed_members) (committee_size - fixed_members)

/-- Theorem: Selecting a 4-person committee from 12 people with one fixed member -/
theorem committee_selection_with_fixed_member :
  select_committee 12 4 1 = 165 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_with_fixed_member_l2402_240261


namespace NUMINAMATH_CALUDE_consecutive_pages_product_l2402_240273

theorem consecutive_pages_product (n : ℕ) : 
  n > 0 ∧ n + (n + 1) = 217 → n * (n + 1) = 11772 := by
sorry

end NUMINAMATH_CALUDE_consecutive_pages_product_l2402_240273


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_6_with_digit_sum_12_l2402_240291

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_multiple_of_6_with_digit_sum_12 :
  ∀ n : ℕ, is_four_digit n → n % 6 = 0 → digit_sum n = 12 → n ≥ 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_6_with_digit_sum_12_l2402_240291


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l2402_240297

theorem restaurant_bill_theorem (num_teenagers : ℕ) (avg_meal_cost : ℝ) (gratuity_rate : ℝ) :
  num_teenagers = 7 →
  avg_meal_cost = 100 →
  gratuity_rate = 0.20 →
  let total_before_gratuity := num_teenagers * avg_meal_cost
  let gratuity := total_before_gratuity * gratuity_rate
  let total_bill := total_before_gratuity + gratuity
  total_bill = 840 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l2402_240297


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l2402_240227

theorem butterflies_in_garden (initial : ℕ) (flew_away_fraction : ℚ) (remaining : ℕ) : 
  initial = 9 → 
  flew_away_fraction = 1/3 → 
  remaining = initial - (initial * flew_away_fraction).num →
  remaining = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l2402_240227


namespace NUMINAMATH_CALUDE_monochromatic_solution_exists_l2402_240281

def Color := Bool

def NumberSet : Set Nat := {1, 2, 3, 4, 5}

def Coloring := Nat → Color

theorem monochromatic_solution_exists (c : Coloring) : 
  ∃ (x y z : Nat), x ∈ NumberSet ∧ y ∈ NumberSet ∧ z ∈ NumberSet ∧ 
  x + y = z ∧ c x = c y ∧ c y = c z :=
sorry

end NUMINAMATH_CALUDE_monochromatic_solution_exists_l2402_240281


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2402_240239

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ (min : ℝ), min = 6 ∧ ∀ x y : ℝ, x + y = 2 → 3^x + 3^y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2402_240239


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2402_240209

theorem chess_tournament_games (n : ℕ) (h : n = 19) : 
  (n * (n - 1)) / 2 = 171 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2402_240209


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2402_240219

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def N : Set ℝ := {x : ℝ | Real.exp (Real.log 2 * (1 - x)) > 1}

-- Define the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2402_240219


namespace NUMINAMATH_CALUDE_xyz_acronym_length_l2402_240259

theorem xyz_acronym_length :
  let straight_segments : ℕ := 6
  let slanted_segments : ℕ := 6
  let straight_length : ℝ := 1
  let slanted_length : ℝ := Real.sqrt 2
  (straight_segments : ℝ) * straight_length + (slanted_segments : ℝ) * slanted_length = 6 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_acronym_length_l2402_240259


namespace NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l2402_240299

theorem root_in_interval_implies_k_range :
  ∀ k : ℝ, 
  (∃ x : ℝ, x ∈ (Set.Ioo 2 3) ∧ x^2 + (1-k)*x - 2*(k+1) = 0) →
  k ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l2402_240299


namespace NUMINAMATH_CALUDE_sum_squares_formula_l2402_240260

theorem sum_squares_formula (m n : ℝ) (h : m + n = 3) : 
  2*m^2 + 4*m*n + 2*n^2 - 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_formula_l2402_240260


namespace NUMINAMATH_CALUDE_chocolate_cost_proof_l2402_240280

/-- The cost of the chocolate given the total spent and the cost of the candy bar -/
def chocolate_cost (total_spent : ℝ) (candy_bar_cost : ℝ) : ℝ :=
  total_spent - candy_bar_cost

theorem chocolate_cost_proof (total_spent candy_bar_cost : ℝ) 
  (h1 : total_spent = 13)
  (h2 : candy_bar_cost = 7) :
  chocolate_cost total_spent candy_bar_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_proof_l2402_240280


namespace NUMINAMATH_CALUDE_white_chips_percentage_l2402_240201

theorem white_chips_percentage
  (total : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (h1 : blue = 3)
  (h2 : blue = total / 10)
  (h3 : green = 12) :
  (total - blue - green) * 100 / total = 50 :=
sorry

end NUMINAMATH_CALUDE_white_chips_percentage_l2402_240201


namespace NUMINAMATH_CALUDE_log_sum_equal_one_power_mult_equal_l2402_240293

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem for the first expression
theorem log_sum_equal_one : log10 2 + log10 5 = 1 := by sorry

-- Theorem for the second expression
theorem power_mult_equal : 4 * (-100)^4 = 400000000 := by sorry

end NUMINAMATH_CALUDE_log_sum_equal_one_power_mult_equal_l2402_240293


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l2402_240210

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l2402_240210


namespace NUMINAMATH_CALUDE_fraction_comparison_l2402_240254

def numerator (x : ℝ) : ℝ := 5 * x + 3

def denominator (x : ℝ) : ℝ := 8 - 3 * x

theorem fraction_comparison (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  numerator x > denominator x ↔ 5/8 < x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2402_240254


namespace NUMINAMATH_CALUDE_seashells_to_find_l2402_240285

def current_seashells : ℕ := 307
def target_seashells : ℕ := 500

theorem seashells_to_find : target_seashells - current_seashells = 193 := by
  sorry

end NUMINAMATH_CALUDE_seashells_to_find_l2402_240285


namespace NUMINAMATH_CALUDE_problem_solution_l2402_240214

theorem problem_solution : 
  ∃ x : ℝ, ((15 - 2 + 4) / 2) * x = 77 ∧ x = 77 / 8.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2402_240214


namespace NUMINAMATH_CALUDE_cody_money_theorem_l2402_240203

def cody_money_problem (initial_money birthday_gift game_price discount friend_debt : ℝ) : Prop :=
  let total_before_purchase := initial_money + birthday_gift
  let discount_amount := game_price * discount
  let actual_game_cost := game_price - discount_amount
  let money_after_purchase := total_before_purchase - actual_game_cost
  let final_amount := money_after_purchase + friend_debt
  final_amount = 48.90

theorem cody_money_theorem :
  cody_money_problem 45 9 19 0.1 12 := by
  sorry

end NUMINAMATH_CALUDE_cody_money_theorem_l2402_240203


namespace NUMINAMATH_CALUDE_davids_english_marks_l2402_240266

/-- Represents the marks of a student in various subjects -/
structure Marks where
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  english : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem davids_english_marks (m : Marks) (h1 : m.mathematics = 60) 
    (h2 : m.physics = 78) (h3 : m.chemistry = 60) (h4 : m.biology = 65) 
    (h5 : average [m.mathematics, m.physics, m.chemistry, m.biology, m.english] = 66.6) :
    m.english = 70 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l2402_240266


namespace NUMINAMATH_CALUDE_product_pricing_and_profit_maximization_l2402_240237

/-- Represents the purchase and selling prices of products A and B -/
structure ProductPrices where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ

/-- Represents the number of units purchased for products A and B -/
structure PurchaseUnits where
  units_A : ℕ
  units_B : ℕ

/-- Calculates the total cost of purchasing given units of products A and B -/
def total_cost (prices : ProductPrices) (units : PurchaseUnits) : ℝ :=
  prices.purchase_price_A * units.units_A + prices.purchase_price_B * units.units_B

/-- Calculates the total profit from selling given units of products A and B -/
def total_profit (prices : ProductPrices) (units : PurchaseUnits) : ℝ :=
  (prices.selling_price_A - prices.purchase_price_A) * units.units_A +
  (prices.selling_price_B - prices.purchase_price_B) * units.units_B

theorem product_pricing_and_profit_maximization
  (prices : ProductPrices)
  (h1 : prices.purchase_price_B = 80)
  (h2 : prices.selling_price_A = 300)
  (h3 : prices.selling_price_B = 100)
  (h4 : total_cost prices { units_A := 50, units_B := 25 } = 15000)
  (h5 : ∀ units : PurchaseUnits, units.units_A + units.units_B = 300 → units.units_B ≥ 2 * units.units_A) :
  prices.purchase_price_A = 260 ∧
  ∃ (max_units : PurchaseUnits),
    max_units.units_A + max_units.units_B = 300 ∧
    max_units.units_B ≥ 2 * max_units.units_A ∧
    max_units.units_A = 100 ∧
    max_units.units_B = 200 ∧
    total_profit prices max_units = 8000 ∧
    ∀ (units : PurchaseUnits),
      units.units_A + units.units_B = 300 →
      units.units_B ≥ 2 * units.units_A →
      total_profit prices units ≤ total_profit prices max_units := by
  sorry

end NUMINAMATH_CALUDE_product_pricing_and_profit_maximization_l2402_240237


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2402_240220

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a + b = 2 → a * b ≤ 1) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a * b ≤ 1 ∧ a + b ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2402_240220


namespace NUMINAMATH_CALUDE_probability_two_red_cards_modified_deck_l2402_240258

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)

/-- The probability of drawing two red cards in succession from the modified deck -/
def probability_two_red_cards (deck : ModifiedDeck) : ℚ :=
  (deck.red_cards * (deck.red_cards - 1)) / (deck.total_cards * (deck.total_cards - 1))

/-- Theorem stating the probability of drawing two red cards from the modified deck -/
theorem probability_two_red_cards_modified_deck :
  ∃ (deck : ModifiedDeck),
    deck.total_cards = 60 ∧
    deck.red_cards = 24 ∧
    deck.suits = 5 ∧
    deck.cards_per_suit = 12 ∧
    probability_two_red_cards deck = 92 / 590 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_cards_modified_deck_l2402_240258


namespace NUMINAMATH_CALUDE_total_rowing_campers_l2402_240282

def morning_rowing : ℕ := 13
def afternoon_rowing : ℕ := 21
def morning_hiking : ℕ := 59

theorem total_rowing_campers :
  morning_rowing + afternoon_rowing = 34 := by sorry

end NUMINAMATH_CALUDE_total_rowing_campers_l2402_240282


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l2402_240283

/-- Represents the number of students sampled from a grade -/
structure SampledStudents :=
  (freshmen : ℕ)
  (sophomores : ℕ)
  (juniors : ℕ)

/-- Calculates the stratified sample size for a grade -/
def stratifiedSampleSize (gradeTotal : ℕ) (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (gradeTotal * sampleSize) / totalStudents

/-- Theorem stating the correct stratified sampling for the given school -/
theorem correct_stratified_sampling :
  let totalStudents : ℕ := 2700
  let freshmenTotal : ℕ := 900
  let sophomoresTotal : ℕ := 1200
  let juniorsTotal : ℕ := 600
  let sampleSize : ℕ := 135
  let result : SampledStudents := {
    freshmen := stratifiedSampleSize freshmenTotal totalStudents sampleSize,
    sophomores := stratifiedSampleSize sophomoresTotal totalStudents sampleSize,
    juniors := stratifiedSampleSize juniorsTotal totalStudents sampleSize
  }
  result.freshmen = 45 ∧ result.sophomores = 60 ∧ result.juniors = 30 :=
by sorry

end NUMINAMATH_CALUDE_correct_stratified_sampling_l2402_240283


namespace NUMINAMATH_CALUDE_total_fat_ingested_l2402_240215

def fat_content (fish : String) : ℝ :=
  match fish with
  | "herring" => 40
  | "eel" => 20
  | "pike" => 30
  | "salmon" => 35
  | "halibut" => 50
  | _ => 0

def cooking_loss_rate : ℝ := 0.1
def indigestible_rate : ℝ := 0.08

def digestible_fat (fish : String) : ℝ :=
  let initial_fat := fat_content fish
  let after_cooking := initial_fat * (1 - cooking_loss_rate)
  after_cooking * (1 - indigestible_rate)

def fish_counts : List (String × ℕ) := [
  ("herring", 40),
  ("eel", 30),
  ("pike", 25),
  ("salmon", 20),
  ("halibut", 15)
]

theorem total_fat_ingested :
  (fish_counts.map (λ (fish, count) => (digestible_fat fish) * count)).sum = 3643.2 := by
  sorry

end NUMINAMATH_CALUDE_total_fat_ingested_l2402_240215


namespace NUMINAMATH_CALUDE_total_energy_consumption_l2402_240294

/-- Calculate total electric energy consumption for given appliances over 30 days -/
theorem total_energy_consumption
  (fan_power : Real) (fan_hours : Real)
  (computer_power : Real) (computer_hours : Real)
  (ac_power : Real) (ac_hours : Real)
  (h : fan_power = 75 ∧ fan_hours = 8 ∧
       computer_power = 100 ∧ computer_hours = 5 ∧
       ac_power = 1500 ∧ ac_hours = 3) :
  (fan_power / 1000 * fan_hours +
   computer_power / 1000 * computer_hours +
   ac_power / 1000 * ac_hours) * 30 = 168 := by
sorry

end NUMINAMATH_CALUDE_total_energy_consumption_l2402_240294


namespace NUMINAMATH_CALUDE_circle_roll_path_length_l2402_240246

/-- The total path length of a point on a circle rolling without slipping -/
theorem circle_roll_path_length
  (r : ℝ) -- radius of the circle
  (θ_flat : ℝ) -- angle rolled on flat surface in radians
  (θ_slope : ℝ) -- angle rolled on slope in radians
  (h_radius : r = 4 / Real.pi)
  (h_flat : θ_flat = 3 * Real.pi / 2)
  (h_slope : θ_slope = Real.pi / 2)
  (h_total : θ_flat + θ_slope = 2 * Real.pi) :
  2 * Real.pi * r = 8 :=
sorry

end NUMINAMATH_CALUDE_circle_roll_path_length_l2402_240246


namespace NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l2402_240206

/-- Two angles share the same terminal side if their difference is a multiple of 360° -/
def ShareTerminalSide (a b : ℝ) : Prop :=
  ∃ k : ℤ, a - b = 360 * k

/-- The main theorem -/
theorem angle_not_sharing_terminal_side :
  let angles : List ℝ := [330, -30, 680, -1110]
  ∀ a ∈ angles, a ≠ 680 → ShareTerminalSide a (-750) ∧
  ¬ ShareTerminalSide 680 (-750) := by
  sorry


end NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l2402_240206


namespace NUMINAMATH_CALUDE_salary_increase_to_original_l2402_240265

/-- Proves that a 56.25% increase is required to regain the original salary after a 30% reduction and 10% bonus --/
theorem salary_increase_to_original (S : ℝ) (S_pos : S > 0) : 
  let reduced_salary := 0.7 * S
  let bonus := 0.1 * S
  let new_salary := reduced_salary + bonus
  (S - new_salary) / new_salary = 0.5625 := by sorry

end NUMINAMATH_CALUDE_salary_increase_to_original_l2402_240265


namespace NUMINAMATH_CALUDE_polynomial_remainder_remainder_theorem_cube_area_is_six_probability_half_tan_135_is_negative_one_l2402_240247

-- Problem 1
theorem polynomial_remainder : Int → Int := 
  fun x ↦ 2 * x^3 - 3 * x^2 + x - 1

theorem remainder_theorem (p : Int → Int) (a : Int) :
  p (-1) = -7 → ∃ q : Int → Int, ∀ x, p x = (x + 1) * q x + -7 := by sorry

-- Problem 2
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

theorem cube_area_is_six : cube_surface_area 1 = 6 := by sorry

-- Problem 3
def probability_white (red white : ℕ) : ℚ := white / (red + white)

theorem probability_half : probability_white 10 10 = 1/2 := by sorry

-- Problem 4
theorem tan_135_is_negative_one : Real.tan (135 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_remainder_theorem_cube_area_is_six_probability_half_tan_135_is_negative_one_l2402_240247


namespace NUMINAMATH_CALUDE_tom_reading_pages_l2402_240252

def pages_read (initial_speed : ℕ) (time : ℕ) (speed_factor : ℕ) : ℕ :=
  initial_speed * speed_factor * time

theorem tom_reading_pages : pages_read 12 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_pages_l2402_240252


namespace NUMINAMATH_CALUDE_captain_age_l2402_240267

/-- Represents the ages of the crew members -/
structure CrewAges where
  sailor : ℕ
  boatswain : ℕ
  engineer : ℕ
  cabinBoy : ℕ
  helmsman : ℕ
  captain : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def validCrewAges (ages : CrewAges) : Prop :=
  ages.sailor = 20 ∧
  ages.boatswain = ages.sailor + 4 ∧
  ages.helmsman = 2 * ages.cabinBoy ∧
  ages.helmsman = ages.engineer + 6 ∧
  ages.boatswain - ages.cabinBoy = ages.engineer - ages.boatswain ∧
  (ages.sailor + ages.boatswain + ages.engineer + ages.cabinBoy + ages.helmsman + ages.captain) / 6 = 28

theorem captain_age (ages : CrewAges) (h : validCrewAges ages) : ages.captain = 40 := by
  sorry

end NUMINAMATH_CALUDE_captain_age_l2402_240267


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2402_240225

-- Define the sets U and A
def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2*x - 1 ∧ 2*x - 1 < 5}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = Set.Icc 0 2 ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2402_240225


namespace NUMINAMATH_CALUDE_triangle_base_length_l2402_240274

/-- Given a triangle with area 24.36 and height 5.8, its base length is 8.4 -/
theorem triangle_base_length : 
  ∀ (base : ℝ), 
    (24.36 = (base * 5.8) / 2) → 
    base = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2402_240274


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2402_240221

theorem quadratic_minimum (x : ℝ) : x^2 + 8*x + 3 ≥ -13 ∧ 
  (x^2 + 8*x + 3 = -13 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2402_240221


namespace NUMINAMATH_CALUDE_dot_product_value_l2402_240212

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def f (a b : n) (x : ℝ) : ℝ := ‖a + x • b‖

theorem dot_product_value (a b : n) 
  (ha : ‖a‖ = Real.sqrt 2) 
  (hb : ‖b‖ = Real.sqrt 2)
  (hmin : ∀ x : ℝ, f a b x ≥ 1)
  (hf : ∃ x : ℝ, f a b x = 1) :
  inner a b = Real.sqrt 2 ∨ inner a b = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_dot_product_value_l2402_240212


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2402_240216

theorem simplify_and_evaluate (x : ℝ) (h : x = 1) : 
  (4 / (x^2 - 4)) / (2 / (x - 2)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2402_240216


namespace NUMINAMATH_CALUDE_one_less_than_negative_two_l2402_240298

theorem one_less_than_negative_two : -2 - 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_negative_two_l2402_240298


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2402_240257

theorem consecutive_integers_average (a b : ℤ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4)) / 5 = a + 4) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2402_240257


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2402_240256

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2402_240256


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2402_240255

/-- Given (1-2x)^7 = a + a₁x + a₂x² + ... + a₇x⁷, prove that a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 12 -/
theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2402_240255


namespace NUMINAMATH_CALUDE_senior_class_college_attendance_l2402_240249

theorem senior_class_college_attendance 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (percent_not_attending : ℝ) 
  (h1 : num_boys = 300)
  (h2 : num_girls = 240)
  (h3 : percent_not_attending = 0.3)
  : (((1 - percent_not_attending) * (num_boys + num_girls)) / (num_boys + num_girls)) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_senior_class_college_attendance_l2402_240249


namespace NUMINAMATH_CALUDE_quadratic_root_l2402_240230

theorem quadratic_root (a b c : ℚ) (r : ℝ) : 
  a ≠ 0 → 
  r = 2 * Real.sqrt 2 - 3 → 
  a * r^2 + b * r + c = 0 → 
  a * (1 : ℝ) = 1 ∧ b = 6 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l2402_240230


namespace NUMINAMATH_CALUDE_triamoeba_population_after_one_week_l2402_240279

/-- Represents the population of Triamoebas after a given number of days -/
def triamoeba_population (initial_population : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_rate ^ days

/-- Theorem stating that the Triamoeba population after 7 days is 2187 -/
theorem triamoeba_population_after_one_week :
  triamoeba_population 1 3 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_triamoeba_population_after_one_week_l2402_240279


namespace NUMINAMATH_CALUDE_smallest_consecutive_integer_sum_l2402_240243

/-- The sum of 15 consecutive positive integers that is a perfect square -/
def consecutiveIntegerSum (n : ℕ) : ℕ := 15 * (n + 7)

/-- The sum is a perfect square -/
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_consecutive_integer_sum :
  (∃ n : ℕ, isPerfectSquare (consecutiveIntegerSum n)) →
  (∀ m : ℕ, isPerfectSquare (consecutiveIntegerSum m) → consecutiveIntegerSum m ≥ 225) :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_integer_sum_l2402_240243


namespace NUMINAMATH_CALUDE_line_family_envelope_and_return_points_l2402_240251

/-- A family of lines connecting points on the complex plane -/
structure LineFamilyOnComplex (k : ℝ) :=
  (k_nonzero : k ≠ 0)
  (k_not_one : k ≠ 1)
  (k_not_neg_one : k ≠ -1)

/-- The envelope curve of a family of lines -/
inductive EnvelopeCurve
  | Hypocycloid
  | Epicycloid

/-- The number of return points for a given k -/
def num_return_points (k : ℤ) : ℕ := Int.natAbs (k - 1)

/-- Main theorem about the envelope of the line family and return points -/
theorem line_family_envelope_and_return_points (k : ℝ) (fam : LineFamilyOnComplex k) :
  (∃ curve : EnvelopeCurve, true) ∧ 
  (∀ k' : ℤ, k' ≠ 0 ∧ k' ≠ 1 ∧ k' ≠ -1 → num_return_points k' = Int.natAbs (k' - 1)) :=
sorry

end NUMINAMATH_CALUDE_line_family_envelope_and_return_points_l2402_240251


namespace NUMINAMATH_CALUDE_line_in_three_quadrants_coeff_products_l2402_240233

/-- A line passing through the first, second, and third quadrants -/
structure LineInThreeQuadrants where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_first_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y + c = 0
  passes_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ a * x + b * y + c = 0
  passes_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0

/-- Theorem: If a line passes through the first, second, and third quadrants, 
    then the product of its coefficients satisfies ab < 0 and bc < 0 -/
theorem line_in_three_quadrants_coeff_products (line : LineInThreeQuadrants) :
  line.a * line.b < 0 ∧ line.b * line.c < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_in_three_quadrants_coeff_products_l2402_240233


namespace NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l2402_240241

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 5| + (y - 2)^2 = 0 → x = -5 ∧ y = 2 ∧ x^y = 25 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l2402_240241


namespace NUMINAMATH_CALUDE_shark_teeth_multiple_l2402_240226

theorem shark_teeth_multiple : 
  let tiger_teeth : ℕ := 180
  let hammerhead_teeth : ℕ := tiger_teeth / 6
  let sum_teeth : ℕ := tiger_teeth + hammerhead_teeth
  let great_white_teeth : ℕ := 420
  great_white_teeth / sum_teeth = 2 := by
  sorry

end NUMINAMATH_CALUDE_shark_teeth_multiple_l2402_240226


namespace NUMINAMATH_CALUDE_point_satisfies_conditions_l2402_240235

/-- A point on a line with equal distance to coordinate axes -/
def point_on_line_equal_distance (x y : ℝ) : Prop :=
  y = -2 * x + 2 ∧ (x = y ∨ x = -y)

/-- The point (2/3, 2/3) satisfies the conditions -/
theorem point_satisfies_conditions : point_on_line_equal_distance (2/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_point_satisfies_conditions_l2402_240235


namespace NUMINAMATH_CALUDE_tangent_line_length_l2402_240292

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define point P
def P : ℝ × ℝ := (-2, 5)

-- Define the tangent line (abstractly, as we don't know its equation)
def tangent_line (Q : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b ∧ circle_equation x y → (x, y) = Q

-- Theorem statement
theorem tangent_line_length :
  ∃ (Q : ℝ × ℝ), circle_equation Q.1 Q.2 ∧ tangent_line Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_length_l2402_240292


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2402_240231

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ b : ℝ, (2 - Complex.I) * (a + 2 * Complex.I) = Complex.I * b) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2402_240231


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2402_240253

/-- The lateral area of a cone with base radius 3 cm and height 4 cm is 15π cm². -/
theorem cone_lateral_area :
  let r : ℝ := 3  -- radius in cm
  let h : ℝ := 4  -- height in cm
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  let lateral_area : ℝ := π * r * l  -- lateral area formula
  lateral_area = 15 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2402_240253


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2402_240284

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2402_240284


namespace NUMINAMATH_CALUDE_evaluate_expression_l2402_240289

theorem evaluate_expression : -1^2010 + (-1)^2011 + 1^2012 - 1^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2402_240289


namespace NUMINAMATH_CALUDE_three_digit_mean_rearrangement_l2402_240240

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  7 * a = 3 * b + 4 * c

def solution_set : Set ℕ :=
  {111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592}

theorem three_digit_mean_rearrangement (n : ℕ) :
  is_valid_number n ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_three_digit_mean_rearrangement_l2402_240240


namespace NUMINAMATH_CALUDE_basketball_game_scores_l2402_240238

/-- Represents the scores of a team in a basketball game -/
structure TeamScores where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Checks if a sequence of four numbers forms a geometric sequence -/
def isGeometricSequence (a b c d : ℕ) : Prop :=
  ∃ r : ℚ, r > 1 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- Checks if a sequence of four numbers forms an arithmetic sequence -/
def isArithmeticSequence (a b c d : ℕ) : Prop :=
  ∃ diff : ℕ, diff > 0 ∧ b = a + diff ∧ c = b + diff ∧ d = c + diff

/-- The main theorem about the basketball game -/
theorem basketball_game_scores
  (alpha : TeamScores)
  (beta : TeamScores)
  (h1 : alpha.first = beta.first)  -- Tied at the end of first quarter
  (h2 : isGeometricSequence alpha.first alpha.second alpha.third alpha.fourth)
  (h3 : isArithmeticSequence beta.first beta.second beta.third beta.fourth)
  (h4 : alpha.first + alpha.second + alpha.third + alpha.fourth =
        beta.first + beta.second + beta.third + beta.fourth + 2)  -- Alpha won by 2 points
  (h5 : alpha.first + alpha.second + alpha.third + alpha.fourth +
        beta.first + beta.second + beta.third + beta.fourth < 200)  -- Total score under 200
  : alpha.first + alpha.second + beta.first + beta.second = 30 :=
by sorry


end NUMINAMATH_CALUDE_basketball_game_scores_l2402_240238


namespace NUMINAMATH_CALUDE_find_number_l2402_240245

theorem find_number : ∃ x : ℚ, (x + 32/113) * 113 = 9637 ∧ x = 85 := by sorry

end NUMINAMATH_CALUDE_find_number_l2402_240245


namespace NUMINAMATH_CALUDE_square_ratio_side_lengths_l2402_240205

theorem square_ratio_side_lengths : 
  ∃ (a b c : ℕ), 
    (a * a * b : ℚ) / (c * c) = 50 / 98 ∧ 
    a = 5 ∧ 
    b = 1 ∧ 
    c = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_lengths_l2402_240205


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2402_240222

theorem complete_square_quadratic (x : ℝ) :
  ∃ (a b : ℝ), (x^2 + 10*x - 3 = 0) ↔ ((x + a)^2 = b) ∧ b = 28 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2402_240222


namespace NUMINAMATH_CALUDE_divisibility_of_3_105_plus_4_105_l2402_240236

theorem divisibility_of_3_105_plus_4_105 :
  let n : ℕ := 3^105 + 4^105
  (∃ k : ℕ, n = 13 * k) ∧
  (∃ k : ℕ, n = 49 * k) ∧
  (∃ k : ℕ, n = 181 * k) ∧
  (∃ k : ℕ, n = 379 * k) ∧
  (∀ k : ℕ, n ≠ 5 * k) ∧
  (∀ k : ℕ, n ≠ 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_3_105_plus_4_105_l2402_240236


namespace NUMINAMATH_CALUDE_max_x_placement_l2402_240277

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if there are three X's in a row in any direction --/
def has_three_in_a_row (g : Grid) : Bool :=
  sorry

/-- Counts the number of X's in the grid --/
def count_x (g : Grid) : Nat :=
  sorry

/-- Theorem stating the maximum number of X's that can be placed --/
theorem max_x_placement :
  ∃ (g : Grid), count_x g = 13 ∧ ¬has_three_in_a_row g ∧
  ∀ (h : Grid), count_x h > 13 → has_three_in_a_row h :=
sorry

end NUMINAMATH_CALUDE_max_x_placement_l2402_240277


namespace NUMINAMATH_CALUDE_allowance_equation_l2402_240242

/-- The student's monthly allowance in USD -/
def monthly_allowance : ℝ := 29.65

/-- Proposition: Given the spending pattern, the monthly allowance satisfies the equation -/
theorem allowance_equation : 
  (5 / 42 : ℝ) * monthly_allowance = 3 / 0.85 := by
  sorry

end NUMINAMATH_CALUDE_allowance_equation_l2402_240242


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_ellipse_foci_l2402_240290

/-- Given an ellipse and a hyperbola E, if the hyperbola has the foci of the ellipse as its vertices,
    then the equation of the hyperbola E can be determined. -/
theorem hyperbola_equation_from_ellipse_foci (x y : ℝ) :
  (x^2 / 10 + y^2 / 5 = 1) →  -- Equation of the ellipse
  (∃ k : ℝ, 3*x + 4*y = k) →  -- Asymptote equation of hyperbola E
  (∃ a b : ℝ, a^2 = 5 ∧ x^2 / 10 + y^2 / 5 = 1 → (x = a ∨ x = -a) ∧ y = 0) →  -- Foci of ellipse as vertices of hyperbola
  (x^2 / 5 - 16*y^2 / 45 = 1)  -- Equation of hyperbola E
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_ellipse_foci_l2402_240290


namespace NUMINAMATH_CALUDE_second_divisor_is_24_l2402_240200

theorem second_divisor_is_24 (m n : ℕ) (h1 : m % 288 = 47) (h2 : m % n = 23) (h3 : n > 23) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_is_24_l2402_240200


namespace NUMINAMATH_CALUDE_cost_function_correct_l2402_240244

/-- The cost function for shipping a parcel -/
def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then
    12 + 4 * (P - 1)
  else
    27 + 4 * P - 21

/-- Theorem stating the correctness of the cost function -/
theorem cost_function_correct (P : ℕ) :
  (P ≤ 5 → cost P = 12 + 4 * (P - 1)) ∧
  (P > 5 → cost P = 27 + 4 * P - 21) := by
  sorry

end NUMINAMATH_CALUDE_cost_function_correct_l2402_240244


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l2402_240218

theorem product_and_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 3 / y) : x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l2402_240218


namespace NUMINAMATH_CALUDE_gcd_210_294_l2402_240287

theorem gcd_210_294 : Nat.gcd 210 294 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_210_294_l2402_240287


namespace NUMINAMATH_CALUDE_four_digit_sum_l2402_240271

theorem four_digit_sum (a b c d : Nat) : 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  (∃ (x y z : Nat), x < 10 ∧ y < 10 ∧ z < 10 ∧
    ((1000 * a + 100 * b + 10 * c + d) + (100 * x + 10 * y + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * a + 10 * y + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * a + 10 * b + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * x + 10 * y + c) = 6031)) →
  a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l2402_240271


namespace NUMINAMATH_CALUDE_min_value_theorem_l2402_240229

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 3| - t

-- State the theorem
theorem min_value_theorem (t : ℝ) (a b : ℝ) :
  (∀ x, f t (x + 2) ≤ 0 ↔ x ∈ Set.Icc (-1) 3) →
  (a > 0 ∧ b > 0) →
  (a * b - 2 * a - 8 * b = 2 * t - 2) →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ - 2 * a₀ - 8 * b₀ = 2 * t - 2 ∧
    ∀ a' b', a' > 0 → b' > 0 → a' * b' - 2 * a' - 8 * b' = 2 * t - 2 → a₀ + 2 * b₀ ≤ a' + 2 * b') →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ - 2 * a₀ - 8 * b₀ = 2 * t - 2 ∧ a₀ + 2 * b₀ = 36) :=
by sorry


end NUMINAMATH_CALUDE_min_value_theorem_l2402_240229


namespace NUMINAMATH_CALUDE_inequality_proof_l2402_240288

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2402_240288


namespace NUMINAMATH_CALUDE_total_courses_is_200_l2402_240248

/-- The number of college courses attended by Max -/
def max_courses : ℕ := 40

/-- The number of college courses attended by Sid -/
def sid_courses : ℕ := 4 * max_courses

/-- The total number of college courses attended by Max and Sid -/
def total_courses : ℕ := max_courses + sid_courses

/-- Theorem stating that the total number of courses attended by Max and Sid is 200 -/
theorem total_courses_is_200 : total_courses = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_courses_is_200_l2402_240248


namespace NUMINAMATH_CALUDE_sum_with_divisibility_conditions_l2402_240250

theorem sum_with_divisibility_conditions : 
  ∃ (a b : ℕ), a + b = 316 ∧ a % 13 = 0 ∧ b % 11 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_with_divisibility_conditions_l2402_240250


namespace NUMINAMATH_CALUDE_stock_price_change_l2402_240286

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  let week1 := initial_price * 1.3
  let week2 := week1 * 0.75
  let week3 := week2 * 1.2
  let week4 := week3 * 0.85
  week4 = initial_price := by sorry

end NUMINAMATH_CALUDE_stock_price_change_l2402_240286


namespace NUMINAMATH_CALUDE_sum_of_factors_l2402_240228

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 30 = (x + b) * (x - c)) →
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2402_240228


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2402_240217

-- Define a function f over the real numbers
variable (f : ℝ → ℝ)

-- Define the reflection operation
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- Theorem statement
theorem reflection_across_x_axis (x y : ℝ) :
  (y = f x) ↔ (-y = (reflect f) x) :=
sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2402_240217


namespace NUMINAMATH_CALUDE_grain_milling_problem_l2402_240272

theorem grain_milling_problem (grain_weight : ℚ) : 
  (grain_weight * (1 - 1/10) = 100) → grain_weight = 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_grain_milling_problem_l2402_240272


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2402_240223

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (x - 1 : ℚ) / 2 < x / 3 ∧ (2 * x - 5 : ℤ) ≤ 3 * (x - 2)}
  S = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2402_240223


namespace NUMINAMATH_CALUDE_shoe_box_problem_l2402_240275

theorem shoe_box_problem (num_pairs : ℕ) (prob_match : ℚ) :
  num_pairs = 9 →
  prob_match = 1 / 17 →
  (num_pairs * 2 : ℕ) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2402_240275


namespace NUMINAMATH_CALUDE_ab_geq_2_sufficient_not_necessary_for_a2_plus_b2_geq_4_l2402_240208

theorem ab_geq_2_sufficient_not_necessary_for_a2_plus_b2_geq_4 :
  (∀ a b : ℝ, a * b ≥ 2 → a^2 + b^2 ≥ 4) ∧
  (∃ a b : ℝ, a^2 + b^2 ≥ 4 ∧ a * b < 2) :=
by sorry

end NUMINAMATH_CALUDE_ab_geq_2_sufficient_not_necessary_for_a2_plus_b2_geq_4_l2402_240208
