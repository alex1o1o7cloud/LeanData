import Mathlib

namespace NUMINAMATH_CALUDE_total_barking_dogs_l1608_160823

theorem total_barking_dogs (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_dogs = 30 → additional_dogs = 10 → initial_dogs + additional_dogs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_barking_dogs_l1608_160823


namespace NUMINAMATH_CALUDE_water_jar_problem_l1608_160843

theorem water_jar_problem (small_jar large_jar : ℝ) 
  (h1 : small_jar > 0) 
  (h2 : large_jar > 0) 
  (h3 : small_jar * (1/4) = large_jar * (1/5)) : 
  (1/5) * small_jar + (1/4) * large_jar = (1/2) * large_jar := by
  sorry

end NUMINAMATH_CALUDE_water_jar_problem_l1608_160843


namespace NUMINAMATH_CALUDE_fraction_division_difference_l1608_160896

theorem fraction_division_difference : (5 / 3) / (1 / 6) - 2 / 3 = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_difference_l1608_160896


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1608_160858

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2/a^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- State that the focus of the parabola is the right focus of the hyperbola
axiom focus_equality : ∃ a : ℝ, hyperbola (parabola_focus.1) (parabola_focus.2) a

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y a : ℝ, parabola x y → hyperbola x y a → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1608_160858


namespace NUMINAMATH_CALUDE_irrational_and_rational_numbers_l1608_160805

theorem irrational_and_rational_numbers : 
  (¬ ∃ (p q : ℤ), π = (p : ℚ) / (q : ℚ)) ∧ 
  (∃ (p q : ℤ), (22 : ℚ) / (7 : ℚ) = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), (0 : ℚ) = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), (-2 : ℚ) = (p : ℚ) / (q : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_irrational_and_rational_numbers_l1608_160805


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1608_160893

theorem two_digit_number_problem : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n % 10 = n / 10 + 4) ∧ 
  (n * (n / 10 + n % 10) = 208) ∧
  n = 26 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1608_160893


namespace NUMINAMATH_CALUDE_diego_extra_cans_l1608_160898

theorem diego_extra_cans (martha_cans : ℕ) (total_cans : ℕ) (diego_cans : ℕ) : 
  martha_cans = 90 →
  total_cans = 145 →
  diego_cans = total_cans - martha_cans →
  diego_cans - martha_cans / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_diego_extra_cans_l1608_160898


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l1608_160883

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem relay_race_arrangements : permutations 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l1608_160883


namespace NUMINAMATH_CALUDE_coin_distribution_l1608_160852

theorem coin_distribution (a b c d e : ℚ) : 
  -- The amounts form an arithmetic sequence
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) →
  -- The total number of coins is 5
  a + b + c + d + e = 5 →
  -- The sum of first two equals the sum of last three
  a + b = c + d + e →
  -- B receives 7/6 coins
  b = 7/6 := by sorry

end NUMINAMATH_CALUDE_coin_distribution_l1608_160852


namespace NUMINAMATH_CALUDE_sum_256_125_base5_l1608_160844

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_256_125_base5 :
  addBase5 (toBase5 256) (toBase5 125) = [3, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_256_125_base5_l1608_160844


namespace NUMINAMATH_CALUDE_crude_oil_mixture_theorem_l1608_160821

/-- Represents the percentage of hydrocarbons in crude oil from the first source -/
def first_source_percentage : ℝ := 25

/-- Represents the total amount of crude oil needed in gallons -/
def total_crude_oil : ℝ := 50

/-- Represents the desired percentage of hydrocarbons in the final mixture -/
def final_mixture_percentage : ℝ := 55

/-- Represents the amount of crude oil from the second source in gallons -/
def second_source_amount : ℝ := 30

/-- Represents the percentage of hydrocarbons in crude oil from the second source -/
def second_source_percentage : ℝ := 75

/-- Theorem stating that given the conditions, the percentage of hydrocarbons
    in the first source is 25% -/
theorem crude_oil_mixture_theorem :
  (first_source_percentage / 100 * (total_crude_oil - second_source_amount) +
   second_source_percentage / 100 * second_source_amount) / total_crude_oil * 100 =
  final_mixture_percentage := by
  sorry

end NUMINAMATH_CALUDE_crude_oil_mixture_theorem_l1608_160821


namespace NUMINAMATH_CALUDE_donny_money_left_l1608_160885

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem donny_money_left : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end NUMINAMATH_CALUDE_donny_money_left_l1608_160885


namespace NUMINAMATH_CALUDE_theater_eye_colors_l1608_160877

theorem theater_eye_colors (total : ℕ) (blue brown black green : ℕ) : 
  total = 100 →
  blue = 19 →
  brown = total / 2 →
  black = total / 4 →
  green = total - (blue + brown + black) →
  green = 6 := by
sorry

end NUMINAMATH_CALUDE_theater_eye_colors_l1608_160877


namespace NUMINAMATH_CALUDE_correct_number_of_elements_l1608_160816

theorem correct_number_of_elements 
  (n : ℕ) 
  (S : ℝ) 
  (initial_average : ℝ) 
  (correct_average : ℝ) 
  (wrong_number : ℝ) 
  (correct_number : ℝ) 
  (h1 : initial_average = 15) 
  (h2 : correct_average = 16) 
  (h3 : wrong_number = 26) 
  (h4 : correct_number = 36) 
  (h5 : (S + wrong_number) / n = initial_average) 
  (h6 : (S + correct_number) / n = correct_average) : 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_correct_number_of_elements_l1608_160816


namespace NUMINAMATH_CALUDE_square_side_length_l1608_160873

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1608_160873


namespace NUMINAMATH_CALUDE_binomial_18_10_l1608_160804

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 32318 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1608_160804


namespace NUMINAMATH_CALUDE_route_length_difference_l1608_160813

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the trip details of Jerry and Beth -/
structure TripDetails where
  jerry_speed : ℝ
  jerry_time : ℝ
  beth_speed : ℝ
  beth_extra_time : ℝ

/-- Theorem stating the difference in route lengths -/
theorem route_length_difference (trip : TripDetails) : 
  trip.jerry_speed = 40 →
  trip.jerry_time = 0.5 →
  trip.beth_speed = 30 →
  trip.beth_extra_time = 1/3 →
  distance trip.beth_speed (trip.jerry_time + trip.beth_extra_time) - 
  distance trip.jerry_speed trip.jerry_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_route_length_difference_l1608_160813


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1608_160879

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1

theorem tangent_line_equation (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f) x₀ = 2 →
  ∃ y₀ : ℝ, y₀ = f x₀ ∧ 2 * x - y - Real.exp + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1608_160879


namespace NUMINAMATH_CALUDE_interest_difference_is_520_l1608_160802

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem stating that the difference between the principal and 
    the simple interest is $520 under the given conditions -/
theorem interest_difference_is_520 :
  let principal : ℝ := 1000
  let rate : ℝ := 0.06
  let time : ℝ := 8
  principal - simple_interest principal rate time = 520 := by
sorry


end NUMINAMATH_CALUDE_interest_difference_is_520_l1608_160802


namespace NUMINAMATH_CALUDE_hair_cut_length_l1608_160835

/-- The amount of hair cut off is equal to the difference between the initial hair length and the final hair length. -/
theorem hair_cut_length (initial_length final_length cut_length : ℕ) 
  (h1 : initial_length = 18)
  (h2 : final_length = 9)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l1608_160835


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l1608_160827

theorem max_rectangular_pen_area (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (width height : ℝ), 
    width > 0 ∧ height > 0 ∧
    2 * (width + height) = perimeter ∧
    ∀ (w h : ℝ), w > 0 → h > 0 → 2 * (w + h) = perimeter → w * h ≤ width * height ∧
    width * height = 225 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l1608_160827


namespace NUMINAMATH_CALUDE_complex_division_l1608_160859

theorem complex_division : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l1608_160859


namespace NUMINAMATH_CALUDE_gumball_probability_l1608_160822

theorem gumball_probability (blue_prob : ℝ) (pink_prob : ℝ) : 
  (blue_prob ^ 2 = 25 / 49) → 
  (blue_prob + pink_prob = 1) → 
  (pink_prob = 2 / 7) := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l1608_160822


namespace NUMINAMATH_CALUDE_morgan_hula_hoop_time_l1608_160853

/-- Given information about hula hooping times for Nancy, Casey, and Morgan,
    prove that Morgan can hula hoop for 21 minutes. -/
theorem morgan_hula_hoop_time :
  ∀ (nancy casey morgan : ℕ),
    nancy = 10 →
    casey = nancy - 3 →
    morgan = 3 * casey →
    morgan = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_morgan_hula_hoop_time_l1608_160853


namespace NUMINAMATH_CALUDE_complex_root_quadratic_equation_l1608_160812

theorem complex_root_quadratic_equation (b c : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I * Real.sqrt 2 : ℂ) ^ 2 + b * (1 - Complex.I * Real.sqrt 2) + c = 0 →
  b = -2 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_equation_l1608_160812


namespace NUMINAMATH_CALUDE_prob_no_defective_bulbs_l1608_160848

/-- The probability of selecting 4 non-defective bulbs out of 10 bulbs, where 4 are defective -/
theorem prob_no_defective_bulbs (total : ℕ) (defective : ℕ) (select : ℕ) :
  total = 10 →
  defective = 4 →
  select = 4 →
  (Nat.choose (total - defective) select : ℚ) / (Nat.choose total select : ℚ) = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_defective_bulbs_l1608_160848


namespace NUMINAMATH_CALUDE_max_books_with_200_dollars_l1608_160828

/-- The maximum number of books that can be purchased with a given budget and book price -/
def maxBooks (budget : ℕ) (bookPrice : ℕ) : ℕ :=
  (budget * 100) / bookPrice

/-- Theorem: Given a book price of $45 and a budget of $200, the maximum number of books that can be purchased is 444 -/
theorem max_books_with_200_dollars : maxBooks 200 45 = 444 := by
  sorry

end NUMINAMATH_CALUDE_max_books_with_200_dollars_l1608_160828


namespace NUMINAMATH_CALUDE_point_conversion_value_l1608_160832

/-- Calculates the value of each point conversion in James' football season. -/
theorem point_conversion_value
  (touchdowns_per_game : ℕ)
  (points_per_touchdown : ℕ)
  (num_games : ℕ)
  (num_conversions : ℕ)
  (old_record : ℕ)
  (points_above_record : ℕ)
  (h1 : touchdowns_per_game = 4)
  (h2 : points_per_touchdown = 6)
  (h3 : num_games = 15)
  (h4 : num_conversions = 6)
  (h5 : old_record = 300)
  (h6 : points_above_record = 72) :
  (old_record + points_above_record - touchdowns_per_game * points_per_touchdown * num_games) / num_conversions = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_conversion_value_l1608_160832


namespace NUMINAMATH_CALUDE_select_five_from_ten_l1608_160874

theorem select_five_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 5 → Nat.choose n k = 252 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_ten_l1608_160874


namespace NUMINAMATH_CALUDE_paulines_garden_tomato_kinds_l1608_160882

/-- Represents Pauline's garden -/
structure Garden where
  rows : ℕ
  spaces_per_row : ℕ
  tomato_kinds : ℕ
  tomatoes_per_kind : ℕ
  cucumber_kinds : ℕ
  cucumbers_per_kind : ℕ
  potatoes : ℕ
  remaining_spaces : ℕ

/-- Theorem representing the problem -/
theorem paulines_garden_tomato_kinds (g : Garden) 
  (h1 : g.rows = 10)
  (h2 : g.spaces_per_row = 15)
  (h3 : g.tomatoes_per_kind = 5)
  (h4 : g.cucumber_kinds = 5)
  (h5 : g.cucumbers_per_kind = 4)
  (h6 : g.potatoes = 30)
  (h7 : g.remaining_spaces = 85)
  (h8 : g.rows * g.spaces_per_row = 
        g.tomato_kinds * g.tomatoes_per_kind + 
        g.cucumber_kinds * g.cucumbers_per_kind + 
        g.potatoes + g.remaining_spaces) : 
  g.tomato_kinds = 3 := by
  sorry

end NUMINAMATH_CALUDE_paulines_garden_tomato_kinds_l1608_160882


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1608_160872

theorem sum_of_solutions (x : ℝ) : (x + 16 / x = 12) → (∃ y : ℝ, y + 16 / y = 12 ∧ y ≠ x) → x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1608_160872


namespace NUMINAMATH_CALUDE_no_isosceles_right_triangle_with_perimeter_60_l1608_160863

theorem no_isosceles_right_triangle_with_perimeter_60 :
  ¬ ∃ (a c : ℕ), 
    a > 0 ∧ 
    c > 0 ∧ 
    c * c = 2 * a * a ∧  -- Pythagorean theorem for isosceles right triangle
    2 * a + c = 60 :=    -- Perimeter condition
by sorry

end NUMINAMATH_CALUDE_no_isosceles_right_triangle_with_perimeter_60_l1608_160863


namespace NUMINAMATH_CALUDE_anniversary_products_l1608_160895

/-- Commemorative albums and bone china cups problem -/
theorem anniversary_products (total_cost album_cost cup_cost album_price cup_price : ℝ)
  (h1 : total_cost = 312000)
  (h2 : album_cost = 3 * cup_cost)
  (h3 : album_cost + cup_cost = total_cost)
  (h4 : album_price = 1.5 * cup_price)
  (h5 : cup_cost / cup_price - 4 * (album_cost / album_price) = 1600) :
  album_cost = 240000 ∧ cup_cost = 72000 ∧ album_price = 45 ∧ cup_price = 30 := by
sorry

end NUMINAMATH_CALUDE_anniversary_products_l1608_160895


namespace NUMINAMATH_CALUDE_smallest_argument_in_circle_l1608_160884

open Complex

theorem smallest_argument_in_circle (p : ℂ) : 
  abs (p - 25 * I) ≤ 15 → arg p ≥ arg (12 + 16 * I) := by
  sorry

end NUMINAMATH_CALUDE_smallest_argument_in_circle_l1608_160884


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1608_160862

/-- Represents a square tile with unit length sides -/
structure UnitTile where
  x : ℕ
  y : ℕ

/-- Represents a figure made of unit tiles -/
structure TileFigure where
  tiles : List UnitTile

/-- Calculates the perimeter of a figure made of unit tiles -/
def perimeter (figure : TileFigure) : ℕ :=
  sorry

/-- Checks if a tile is adjacent to any tile in the figure -/
def isAdjacent (tile : UnitTile) (figure : TileFigure) : Bool :=
  sorry

theorem perimeter_after_adding_tiles :
  ∃ (original : TileFigure) (new1 new2 : UnitTile),
    (original.tiles.length = 16) ∧
    (∀ t ∈ original.tiles, t.x < 4 ∧ t.y < 4) ∧
    (isAdjacent new1 original) ∧
    (isAdjacent new2 original) ∧
    (perimeter (TileFigure.mk (new1 :: new2 :: original.tiles)) = 18) :=
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1608_160862


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l1608_160836

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun i => 2 * i + 1) = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l1608_160836


namespace NUMINAMATH_CALUDE_max_non_managers_l1608_160810

theorem max_non_managers (managers : ℕ) (ratio_managers : ℕ) (ratio_non_managers : ℕ) :
  managers = 11 →
  ratio_managers = 7 →
  ratio_non_managers = 37 →
  ∀ non_managers : ℕ,
    (managers : ℚ) / (non_managers : ℚ) > (ratio_managers : ℚ) / (ratio_non_managers : ℚ) →
    non_managers ≤ 58 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l1608_160810


namespace NUMINAMATH_CALUDE_min_boxes_to_eliminate_l1608_160881

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $250,000 -/
def high_value_boxes : ℕ := 6

/-- The desired probability of holding a high-value box -/
def desired_probability : ℚ := 1/3

/-- The function to calculate the minimum number of boxes to eliminate -/
def boxes_to_eliminate : ℕ := total_boxes - (high_value_boxes * 3)

/-- Theorem stating the minimum number of boxes to eliminate -/
theorem min_boxes_to_eliminate :
  boxes_to_eliminate = 12 := by sorry

end NUMINAMATH_CALUDE_min_boxes_to_eliminate_l1608_160881


namespace NUMINAMATH_CALUDE_walt_age_l1608_160838

theorem walt_age (walt_age music_teacher_age : ℕ) : 
  music_teacher_age = 3 * walt_age →
  music_teacher_age + 12 = 2 * (walt_age + 12) →
  walt_age = 12 := by
sorry

end NUMINAMATH_CALUDE_walt_age_l1608_160838


namespace NUMINAMATH_CALUDE_amazon_pack_price_is_correct_l1608_160886

/-- The cost of a single lighter at the gas station in dollars -/
def gas_station_price : ℚ := 1.75

/-- The number of lighters Amanda wants to buy -/
def num_lighters : ℕ := 24

/-- The amount Amanda saves by buying online in dollars -/
def savings : ℚ := 32

/-- The cost of a pack of twelve lighters on Amazon in dollars -/
def amazon_pack_price : ℚ := 5

theorem amazon_pack_price_is_correct :
  amazon_pack_price = 5 ∧
  2 * amazon_pack_price = num_lighters * gas_station_price - savings :=
by sorry

end NUMINAMATH_CALUDE_amazon_pack_price_is_correct_l1608_160886


namespace NUMINAMATH_CALUDE_jade_savings_l1608_160806

def monthly_income : ℝ := 1600

def living_expenses_ratio : ℝ := 0.75
def insurance_ratio : ℝ := 0.2

def savings (income : ℝ) (living_ratio : ℝ) (insurance_ratio : ℝ) : ℝ :=
  income - (income * living_ratio) - (income * insurance_ratio)

theorem jade_savings : savings monthly_income living_expenses_ratio insurance_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_jade_savings_l1608_160806


namespace NUMINAMATH_CALUDE_cubic_arithmetic_progression_roots_l1608_160819

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_in_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), p.a * (r - d)^3 + p.b * (r - d)^2 + p.c * (r - d) + p.d = 0 ∧
                p.a * r^3 + p.b * r^2 + p.c * r + p.d = 0 ∧
                p.a * (r + d)^3 + p.b * (r + d)^2 + p.c * (r + d) + p.d = 0

/-- Two roots of a cubic polynomial are not real -/
def two_roots_not_real (p : CubicPolynomial) : Prop :=
  ∃ (z w : ℂ), p.a * z^3 + p.b * z^2 + p.c * z + p.d = 0 ∧
                p.a * w^3 + p.b * w^2 + p.c * w + p.d = 0 ∧
                z.im ≠ 0 ∧ w.im ≠ 0 ∧ z ≠ w

theorem cubic_arithmetic_progression_roots (a : ℝ) :
  let p := CubicPolynomial.mk 1 (-7) 20 a
  roots_in_arithmetic_progression p ∧ two_roots_not_real p → a = -574/27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_arithmetic_progression_roots_l1608_160819


namespace NUMINAMATH_CALUDE_expression_upper_bound_l1608_160814

theorem expression_upper_bound (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 2) :
  Real.sqrt (a^3 + (2-b)^3) + Real.sqrt (b^3 + (2-c)^3) + 
  Real.sqrt (c^3 + (2-d)^3) + Real.sqrt (d^3 + (3-a)^3) ≤ 5 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_upper_bound_l1608_160814


namespace NUMINAMATH_CALUDE_vet_recommendation_difference_l1608_160837

/-- Given a total number of vets and percentages recommending two different brands of dog food,
    prove that the difference in the number of vets recommending each brand is as expected. -/
theorem vet_recommendation_difference
  (total_vets : ℕ)
  (puppy_kibble_percent : ℚ)
  (yummy_dog_kibble_percent : ℚ)
  (h_total : total_vets = 1000)
  (h_puppy : puppy_kibble_percent = 1/5)
  (h_yummy : yummy_dog_kibble_percent = 3/10) :
  (total_vets : ℚ) * yummy_dog_kibble_percent - (total_vets : ℚ) * puppy_kibble_percent = 100 :=
by sorry


end NUMINAMATH_CALUDE_vet_recommendation_difference_l1608_160837


namespace NUMINAMATH_CALUDE_white_balls_count_l1608_160860

theorem white_balls_count (red_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) : 
  red_balls = 12 → 
  prob_white = 2/3 → 
  (white_balls : ℚ) / (white_balls + red_balls) = prob_white →
  white_balls = 24 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1608_160860


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1608_160811

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₂ + a₄) / (a₁ + a₃) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1608_160811


namespace NUMINAMATH_CALUDE_marble_comparison_l1608_160824

theorem marble_comparison (katrina mabel amanda carlos diana : ℕ) : 
  mabel = 5 * katrina →
  amanda + 12 = 2 * katrina →
  carlos = 3 * katrina →
  diana = 2 * katrina + (katrina / 2) →
  mabel = 85 →
  mabel = amanda + carlos + diana - 30 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_comparison_l1608_160824


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_l1608_160847

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Predicate to check if m divides n^2 + 1 and n divides m^2 + 1 -/
def satisfies_condition (m n : ℕ) : Prop :=
  (n^2 + 1) % m = 0 ∧ (m^2 + 1) % n = 0

theorem infinite_pairs_exist :
  ∀ k : ℕ, ∃ m n : ℕ,
    m > k ∧
    n > k ∧
    satisfies_condition m n ∧
    m = fib (2 * n + 1) ∧
    n = fib (2 * n - 1) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_l1608_160847


namespace NUMINAMATH_CALUDE_two_extremum_function_properties_l1608_160868

/-- A function with two distinct extremum points -/
structure TwoExtremumFunction where
  f : ℝ → ℝ
  a : ℝ
  x1 : ℝ
  x2 : ℝ
  h_def : ∀ x, f x = x^2 + a * Real.log (x + 1)
  h_extremum : x1 < x2
  h_distinct : ∃ y, x1 < y ∧ y < x2

/-- The main theorem about the properties of the function -/
theorem two_extremum_function_properties (g : TwoExtremumFunction) :
  0 < g.a ∧ g.a < 1/2 ∧ 0 < g.f g.x2 / g.x1 ∧ g.f g.x2 / g.x1 < -1/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_two_extremum_function_properties_l1608_160868


namespace NUMINAMATH_CALUDE_coin_identification_strategy_exists_l1608_160842

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a coin -/
structure Coin :=
(id : Nat)
(is_genuine : Bool)

/-- Represents the scale used for weighing -/
def Scale := List Coin → List Coin → WeighingResult

/-- Represents the strategy for identifying genuine coins -/
def IdentificationStrategy := Scale → List Coin → List Coin

theorem coin_identification_strategy_exists :
  ∀ (coins : List Coin),
    coins.length = 8 →
    (coins.filter (λ c => c.is_genuine)).length = 7 →
    ∃ (strategy : IdentificationStrategy),
      ∀ (scale : Scale),
        let identified := strategy scale coins
        identified.length ≥ 5 ∧
        ∀ c ∈ identified, c.is_genuine ∧
        ∀ c ∈ identified, c ∉ (coins.filter (λ c => ¬c.is_genuine)) :=
by sorry

end NUMINAMATH_CALUDE_coin_identification_strategy_exists_l1608_160842


namespace NUMINAMATH_CALUDE_smallest_x_value_l1608_160894

theorem smallest_x_value (x : ℝ) : 
  (((14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x) = (7 * x - 2)) → x ≥ 4/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1608_160894


namespace NUMINAMATH_CALUDE_teal_greenish_count_teal_greenish_proof_l1608_160851

def total_surveyed : ℕ := 120
def kinda_blue : ℕ := 70
def both : ℕ := 35
def neither : ℕ := 20

theorem teal_greenish_count : ℕ :=
  total_surveyed - (kinda_blue - both) - both - neither
  
theorem teal_greenish_proof : teal_greenish_count = 65 := by
  sorry

end NUMINAMATH_CALUDE_teal_greenish_count_teal_greenish_proof_l1608_160851


namespace NUMINAMATH_CALUDE_student_number_problem_l1608_160845

theorem student_number_problem (x : ℝ) : 8 * x - 138 = 102 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1608_160845


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l1608_160825

theorem ratio_sum_theorem (w x y : ℝ) 
  (h1 : w / x = 1 / 6)
  (h2 : w / y = 1 / 5) :
  (x + y) / y = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l1608_160825


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1608_160850

theorem cubic_equation_root (a b : ℚ) : 
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 45 = 0 ∧ x = -2 - 5*Real.sqrt 3) →
  a = 239/71 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1608_160850


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1608_160866

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1608_160866


namespace NUMINAMATH_CALUDE_function_value_l1608_160803

/-- Given a function f(x) = x^α that passes through (2, √2/2), prove f(4) = 1/2 -/
theorem function_value (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) :
  f 2 = Real.sqrt 2 / 2 → f 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l1608_160803


namespace NUMINAMATH_CALUDE_dino_money_theorem_l1608_160871

/-- Calculates Dino's remaining money at the end of the month -/
def dino_remaining_money (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem: Dino's remaining money at the end of the month is $500 -/
theorem dino_money_theorem : dino_remaining_money 20 30 5 10 20 40 500 = 500 := by
  sorry

end NUMINAMATH_CALUDE_dino_money_theorem_l1608_160871


namespace NUMINAMATH_CALUDE_cookie_ratio_l1608_160834

theorem cookie_ratio (monday_cookies : ℕ) (total_cookies : ℕ) 
  (h1 : monday_cookies = 32)
  (h2 : total_cookies = 92)
  (h3 : ∃ f : ℚ, 
    monday_cookies + f * monday_cookies + (3 * f * monday_cookies - 4) = total_cookies) :
  ∃ f : ℚ, f = 1/2 ∧ 
    monday_cookies + f * monday_cookies + (3 * f * monday_cookies - 4) = total_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l1608_160834


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1608_160865

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-2) - 3
  f 2 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1608_160865


namespace NUMINAMATH_CALUDE_existence_condition_l1608_160854

theorem existence_condition (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2*x - a ≥ 0) ↔ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_condition_l1608_160854


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1608_160888

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - i) / (2 + 5*i) = (1:ℂ) / 29 - (17:ℂ) / 29 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1608_160888


namespace NUMINAMATH_CALUDE_probability_product_24_l1608_160864

def is_valid_die_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def product_equals_24 (a b c d : ℕ) : Prop :=
  is_valid_die_roll a ∧ is_valid_die_roll b ∧ is_valid_die_roll c ∧ is_valid_die_roll d ∧ a * b * c * d = 24

def count_valid_permutations : ℕ := 36

def total_outcomes : ℕ := 6^4

theorem probability_product_24 :
  (count_valid_permutations : ℚ) / total_outcomes = 1 / 36 :=
sorry

end NUMINAMATH_CALUDE_probability_product_24_l1608_160864


namespace NUMINAMATH_CALUDE_school_population_proof_l1608_160856

theorem school_population_proof :
  ∀ (n : ℕ) (senior_class : ℕ) (total_selected : ℕ) (other_selected : ℕ),
  senior_class = 900 →
  total_selected = 20 →
  other_selected = 14 →
  (total_selected - other_selected : ℚ) / senior_class = total_selected / n →
  n = 3000 := by
sorry

end NUMINAMATH_CALUDE_school_population_proof_l1608_160856


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1608_160861

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1608_160861


namespace NUMINAMATH_CALUDE_max_cyclic_product_permutation_l1608_160855

def cyclic_product (xs : List ℕ) : ℕ :=
  let n := xs.length
  List.sum (List.zipWith (· * ·) xs (xs.rotateLeft 1))

theorem max_cyclic_product_permutation :
  let perms := List.permutations [1, 2, 3, 4, 5]
  let max_val := perms.map cyclic_product |>.maximum?
  let max_count := (perms.filter (λ p ↦ cyclic_product p = max_val.getD 0)).length
  (max_val.getD 0 = 48) ∧ (max_count = 10) := by
  sorry

end NUMINAMATH_CALUDE_max_cyclic_product_permutation_l1608_160855


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1608_160875

/-- Represents a number in base 6 as XX₆ -/
def base6 (x : ℕ) : ℕ := 6 * x + x

/-- Represents a number in base 8 as YY₈ -/
def base8 (y : ℕ) : ℕ := 8 * y + y

/-- Checks if a digit is valid in base 6 -/
def validBase6Digit (x : ℕ) : Prop := x ≤ 5

/-- Checks if a digit is valid in base 8 -/
def validBase8Digit (y : ℕ) : Prop := y ≤ 7

theorem smallest_dual_base_representation :
  ∃ (x y : ℕ), validBase6Digit x ∧ validBase8Digit y ∧
    base6 x = base8 y ∧
    base6 x = 63 ∧
    (∀ (x' y' : ℕ), validBase6Digit x' → validBase8Digit y' →
      base6 x' = base8 y' → base6 x' ≥ 63) :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1608_160875


namespace NUMINAMATH_CALUDE_every_three_connected_graph_without_K5_K33_is_planar_l1608_160817

-- Define a graph type
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define 3-connectivity
def isThreeConnected (G : Graph V) : Prop := sorry

-- Define subgraph relation
def isSubgraph (H G : Graph V) : Prop := sorry

-- Define K^5 graph
def K5 (V : Type) : Graph V := sorry

-- Define K_{3,3} graph
def K33 (V : Type) : Graph V := sorry

-- Define planarity
def isPlanar (G : Graph V) : Prop := sorry

-- The main theorem
theorem every_three_connected_graph_without_K5_K33_is_planar 
  (G : Graph V) 
  (h1 : isThreeConnected G) 
  (h2 : ¬ isSubgraph (K5 V) G) 
  (h3 : ¬ isSubgraph (K33 V) G) : 
  isPlanar G := by sorry

end NUMINAMATH_CALUDE_every_three_connected_graph_without_K5_K33_is_planar_l1608_160817


namespace NUMINAMATH_CALUDE_theater_company_max_members_l1608_160809

/-- The number of columns in the rectangular formation -/
def n : ℕ := 15

/-- The total number of members in the theater company -/
def total_members : ℕ := n * (n + 9)

/-- Theorem stating that the maximum number of members satisfying the given conditions is 360 -/
theorem theater_company_max_members :
  (∃ k : ℕ, total_members = k^2 + 3) ∧
  (total_members = n * (n + 9)) ∧
  (∀ m > total_members, ¬(∃ j : ℕ, m = j^2 + 3) ∨ ¬(∃ p : ℕ, m = p * (p + 9))) ∧
  total_members = 360 := by
  sorry

end NUMINAMATH_CALUDE_theater_company_max_members_l1608_160809


namespace NUMINAMATH_CALUDE_walter_school_allocation_l1608_160829

/-- Represents Walter's work schedule and earnings --/
structure WorkSchedule where
  days_per_week : ℕ
  hours_per_day : ℕ
  hourly_rate : ℚ
  school_allocation_ratio : ℚ

/-- Calculates the amount Walter allocates for school given his work schedule --/
def school_allocation (schedule : WorkSchedule) : ℚ :=
  schedule.days_per_week * schedule.hours_per_day * schedule.hourly_rate * schedule.school_allocation_ratio

/-- Theorem stating that Walter allocates $75 for school each week --/
theorem walter_school_allocation :
  let walter_schedule : WorkSchedule := {
    days_per_week := 5,
    hours_per_day := 4,
    hourly_rate := 5,
    school_allocation_ratio := 3/4
  }
  school_allocation walter_schedule = 75 := by
  sorry

end NUMINAMATH_CALUDE_walter_school_allocation_l1608_160829


namespace NUMINAMATH_CALUDE_board_cutting_theorem_l1608_160897

def is_valid_board_size (n : ℕ) : Prop :=
  ∃ m : ℕ, n * n = 5 * m ∧ n > 5

theorem board_cutting_theorem (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n * n = m + 4 * m) ↔ is_valid_board_size n :=
sorry

end NUMINAMATH_CALUDE_board_cutting_theorem_l1608_160897


namespace NUMINAMATH_CALUDE_bat_ball_cost_difference_l1608_160815

/-- The cost difference between a ball and a bat -/
def cost_difference (x y : ℝ) : ℝ := y - x

/-- The problem statement -/
theorem bat_ball_cost_difference :
  ∀ x y : ℝ,
  (2 * x + 3 * y = 1300) →
  (3 * x + 2 * y = 1200) →
  cost_difference x y = 100 := by
sorry

end NUMINAMATH_CALUDE_bat_ball_cost_difference_l1608_160815


namespace NUMINAMATH_CALUDE_valid_queues_count_l1608_160831

/-- Represents the amount a customer has: 
    1 for 50 cents (exact change), -1 for one dollar (needs change) -/
inductive CustomerMoney : Type
  | exact : CustomerMoney
  | needsChange : CustomerMoney

/-- A queue of customers -/
def CustomerQueue := List CustomerMoney

/-- The nth Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Checks if a queue is valid (cashier can always give change) -/
def isValidQueue (queue : CustomerQueue) : Prop := sorry

/-- Counts the number of valid queues for 2n customers -/
def countValidQueues (n : ℕ) : ℕ := sorry

/-- Theorem: The number of valid queues for 2n customers 
    (n with exact change, n needing change) is the nth Catalan number -/
theorem valid_queues_count (n : ℕ) : 
  countValidQueues n = catalanNumber n := by sorry

end NUMINAMATH_CALUDE_valid_queues_count_l1608_160831


namespace NUMINAMATH_CALUDE_unattainable_value_l1608_160833

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃y, y = (2 - x) / (3 * x + 4) ∧ y = -1/3 := by
sorry

end NUMINAMATH_CALUDE_unattainable_value_l1608_160833


namespace NUMINAMATH_CALUDE_min_pieces_to_control_l1608_160808

/-- Represents a rhombus game board -/
structure GameBoard where
  angle : ℝ
  side_divisions : ℕ

/-- Represents a piece on the game board -/
structure Piece where
  position : ℕ × ℕ

/-- Checks if a piece controls a given position -/
def controls (p : Piece) (pos : ℕ × ℕ) : Prop := sorry

/-- Checks if a set of pieces controls all positions on the board -/
def controls_all (pieces : Finset Piece) (board : GameBoard) : Prop := sorry

/-- The main theorem stating the minimum number of pieces required -/
theorem min_pieces_to_control (board : GameBoard) :
  board.angle = 60 ∧ board.side_divisions = 9 →
  ∃ (pieces : Finset Piece),
    pieces.card = 6 ∧
    controls_all pieces board ∧
    ∀ (other_pieces : Finset Piece),
      controls_all other_pieces board →
      other_pieces.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_pieces_to_control_l1608_160808


namespace NUMINAMATH_CALUDE_odd_numbers_product_equality_l1608_160890

theorem odd_numbers_product_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_numbers_product_equality_l1608_160890


namespace NUMINAMATH_CALUDE_no_perfect_square_203_base_n_l1608_160899

theorem no_perfect_square_203_base_n : 
  ¬ ∃ n : ℤ, 4 ≤ n ∧ n ≤ 18 ∧ ∃ k : ℤ, 2 * n^2 + 3 = k^2 :=
sorry

end NUMINAMATH_CALUDE_no_perfect_square_203_base_n_l1608_160899


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l1608_160800

theorem easter_egg_distribution (red_eggs orange_eggs min_eggs : ℕ) 
  (h1 : red_eggs = 30)
  (h2 : orange_eggs = 45)
  (h3 : min_eggs = 5) :
  ∃ (eggs_per_basket : ℕ), 
    eggs_per_basket ≥ min_eggs ∧ 
    eggs_per_basket ∣ red_eggs ∧ 
    eggs_per_basket ∣ orange_eggs ∧
    ∀ (n : ℕ), n > eggs_per_basket → ¬(n ∣ red_eggs ∧ n ∣ orange_eggs) :=
by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l1608_160800


namespace NUMINAMATH_CALUDE_red_purple_probability_l1608_160869

def total_balls : ℕ := 120
def red_balls : ℕ := 20
def purple_balls : ℕ := 5

theorem red_purple_probability : 
  (red_balls * purple_balls * 2 : ℚ) / (total_balls * (total_balls - 1)) = 5 / 357 := by
  sorry

end NUMINAMATH_CALUDE_red_purple_probability_l1608_160869


namespace NUMINAMATH_CALUDE_adam_has_14_apples_l1608_160840

-- Define the number of apples Jackie has
def jackie_apples : ℕ := 9

-- Define Adam's apples in relation to Jackie's
def adam_apples : ℕ := jackie_apples + 5

-- Theorem statement
theorem adam_has_14_apples : adam_apples = 14 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_14_apples_l1608_160840


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1608_160880

-- Define the grid dimensions
def grid_width : Nat := 4
def grid_height : Nat := 5

-- Define a type for grid positions
structure GridPosition where
  x : Nat
  y : Nat

-- Define the initially shaded squares
def initial_shaded : List GridPosition := [
  { x := 1, y := 4 },
  { x := 4, y := 1 }
]

-- Define a function to check if a position is within the grid
def is_valid_position (pos : GridPosition) : Prop :=
  pos.x ≥ 0 ∧ pos.x < grid_width ∧ pos.y ≥ 0 ∧ pos.y < grid_height

-- Define a function to check if a list of positions creates horizontal and vertical symmetry
def is_symmetric (shaded : List GridPosition) : Prop :=
  ∀ pos : GridPosition, is_valid_position pos →
    (pos ∈ shaded ↔ 
     { x := grid_width - 1 - pos.x, y := pos.y } ∈ shaded ∧
     { x := pos.x, y := grid_height - 1 - pos.y } ∈ shaded)

-- The main theorem
theorem min_additional_squares_for_symmetry :
  ∃ (additional : List GridPosition),
    (∀ pos ∈ additional, pos ∉ initial_shaded) ∧
    is_symmetric (initial_shaded ++ additional) ∧
    additional.length = 6 ∧
    (∀ (other : List GridPosition),
      (∀ pos ∈ other, pos ∉ initial_shaded) →
      is_symmetric (initial_shaded ++ other) →
      other.length ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1608_160880


namespace NUMINAMATH_CALUDE_ring_endomorphism_properties_division_ring_commutativity_l1608_160841

structure RingWithEndomorphism (R : Type) [Ring R] :=
  (f : R → R)
  (f_surjective : Function.Surjective f)
  (f_hom : ∀ x y, f (x + y) = f x + f y)
  (f_hom_mul : ∀ x y, f (x * y) = f x * f y)
  (f_commutes : ∀ x, x * f x = f x * x)

theorem ring_endomorphism_properties {R : Type} [Ring R] (S : RingWithEndomorphism R) :
  (∀ x y : R, x * S.f y - S.f y * x = S.f x * y - y * S.f x) ∧
  (∀ x y : R, x * (x * y - y * x) = S.f x * (x * y - y * x)) :=
sorry

theorem division_ring_commutativity {R : Type} [DivisionRing R] (S : RingWithEndomorphism R) :
  (∃ x : R, S.f x ≠ x) → (∀ a b : R, a * b = b * a) :=
sorry

end NUMINAMATH_CALUDE_ring_endomorphism_properties_division_ring_commutativity_l1608_160841


namespace NUMINAMATH_CALUDE_five_digit_number_formation_l1608_160830

/-- A two-digit number is between 10 and 99, inclusive -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A three-digit number is between 100 and 999, inclusive -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The five-digit number formed by placing x to the left of y -/
def FiveDigitNumber (x y : ℕ) : ℕ := 1000 * x + y

theorem five_digit_number_formation (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : ThreeDigitNumber y) :
  FiveDigitNumber x y = 1000 * x + y := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_formation_l1608_160830


namespace NUMINAMATH_CALUDE_max_parallelograms_in_hexagon_l1608_160857

-- Define the regular hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the parallelogram
def parallelogram (side1 : ℝ) (side2 : ℝ) (angle1 : ℝ) (angle2 : ℝ) : Set (ℝ × ℝ) := sorry

-- Define a function to count non-overlapping parallelograms in a hexagon
def count_parallelograms (h : Set (ℝ × ℝ)) (p : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem max_parallelograms_in_hexagon :
  let h := regular_hexagon 3
  let p := parallelogram 1 2 (π/3) (2*π/3)
  count_parallelograms h p = 12 := by sorry

end NUMINAMATH_CALUDE_max_parallelograms_in_hexagon_l1608_160857


namespace NUMINAMATH_CALUDE_movie_pause_point_l1608_160878

/-- Proves that the pause point in a movie is halfway through, given the total length and remaining time. -/
theorem movie_pause_point (total_length remaining : ℕ) (h1 : total_length = 60) (h2 : remaining = 30) :
  total_length - remaining = 30 := by
  sorry

end NUMINAMATH_CALUDE_movie_pause_point_l1608_160878


namespace NUMINAMATH_CALUDE_interval_partition_existence_l1608_160807

theorem interval_partition_existence : ∃ (x : Fin 10 → ℝ), 
  (∀ i, x i ∈ Set.Icc (0 : ℝ) 1) ∧ 
  (∀ k : Fin 9, k.val + 2 ≤ 10 → 
    ∀ i j : Fin (k.val + 2), i ≠ j → 
      ⌊(k.val + 2 : ℝ) * x i⌋ ≠ ⌊(k.val + 2 : ℝ) * x j⌋) :=
sorry

end NUMINAMATH_CALUDE_interval_partition_existence_l1608_160807


namespace NUMINAMATH_CALUDE_new_sailor_weight_l1608_160818

theorem new_sailor_weight (n : ℕ) (original_weight replaced_weight : ℝ) 
  (h1 : n = 8)
  (h2 : replaced_weight = 56)
  (h3 : ∀ (total_weight new_weight : ℝ), 
    (total_weight + new_weight - replaced_weight) / n = total_weight / n + 1) :
  ∃ (new_weight : ℝ), new_weight = 64 := by
sorry

end NUMINAMATH_CALUDE_new_sailor_weight_l1608_160818


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1608_160892

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1608_160892


namespace NUMINAMATH_CALUDE_right_triangle_product_divisible_by_60_l1608_160889

theorem right_triangle_product_divisible_by_60 
  (a b c : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  60 ∣ (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_product_divisible_by_60_l1608_160889


namespace NUMINAMATH_CALUDE_optimal_rectangle_area_l1608_160870

theorem optimal_rectangle_area 
  (perimeter : ℝ) 
  (min_length : ℝ) 
  (min_width : ℝ) 
  (h_perimeter : perimeter = 360) 
  (h_min_length : min_length = 90) 
  (h_min_width : min_width = 50) : 
  ∃ (length width : ℝ), 
    length ≥ min_length ∧ 
    width ≥ min_width ∧ 
    2 * (length + width) = perimeter ∧ 
    length * width = 8100 ∧ 
    ∀ (l w : ℝ), 
      l ≥ min_length → 
      w ≥ min_width → 
      2 * (l + w) = perimeter → 
      l * w ≤ 8100 :=
sorry

end NUMINAMATH_CALUDE_optimal_rectangle_area_l1608_160870


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1608_160876

/-- The number of games played in a round-robin tournament -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 8 players, where each player plays every other player once,
    the total number of games played is 28. -/
theorem chess_tournament_games :
  gamesPlayed 8 = 28 := by
  sorry

#eval gamesPlayed 8  -- This should output 28

end NUMINAMATH_CALUDE_chess_tournament_games_l1608_160876


namespace NUMINAMATH_CALUDE_tens_digit_of_sum_l1608_160849

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_sum : tens_digit (2^1500 + 5^768) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_sum_l1608_160849


namespace NUMINAMATH_CALUDE_total_black_dots_l1608_160867

theorem total_black_dots (num_butterflies : ℕ) (black_dots_per_butterfly : ℕ) 
  (h1 : num_butterflies = 397) 
  (h2 : black_dots_per_butterfly = 12) : 
  num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end NUMINAMATH_CALUDE_total_black_dots_l1608_160867


namespace NUMINAMATH_CALUDE_wyatt_orange_juice_purchase_l1608_160820

def orange_juice_cartons (initial_money : ℕ) (bread_loaves : ℕ) (bread_cost : ℕ) (juice_cost : ℕ) (remaining_money : ℕ) : ℕ :=
  (initial_money - remaining_money - bread_loaves * bread_cost) / juice_cost

theorem wyatt_orange_juice_purchase :
  orange_juice_cartons 74 5 5 2 41 = 4 := by
  sorry

end NUMINAMATH_CALUDE_wyatt_orange_juice_purchase_l1608_160820


namespace NUMINAMATH_CALUDE_martha_cards_l1608_160839

theorem martha_cards (initial_cards given_cards : ℝ) 
  (h1 : initial_cards = 76.0)
  (h2 : given_cards = 3.0) : 
  initial_cards - given_cards = 73.0 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l1608_160839


namespace NUMINAMATH_CALUDE_total_new_people_count_l1608_160887

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def people_immigrated : ℕ := 16320

/-- The total number of new people in the country last year -/
def total_new_people : ℕ := people_born + people_immigrated

/-- Theorem stating that the total number of new people is 106491 -/
theorem total_new_people_count : total_new_people = 106491 := by
  sorry

end NUMINAMATH_CALUDE_total_new_people_count_l1608_160887


namespace NUMINAMATH_CALUDE_odd_increasing_function_inequality_l1608_160801

-- Define an odd function that is increasing on [0,+∞)
def OddIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, 0 ≤ x → x < y → f x < f y)

-- State the theorem
theorem odd_increasing_function_inequality (f : ℝ → ℝ) (h : OddIncreasingFunction f) :
  ∀ x : ℝ, f (Real.log x) < 0 → 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_increasing_function_inequality_l1608_160801


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1608_160891

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- first leg
  b : ℝ  -- second leg
  c : ℝ  -- hypotenuse
  h : ℝ  -- altitude to hypotenuse
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  altitude_prop : h * c = a * b  -- property of altitude in right triangle

-- State the theorem
theorem right_triangle_inequality (t : RightTriangle) : t.a + t.b < t.c + t.h := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1608_160891


namespace NUMINAMATH_CALUDE_work_completion_time_l1608_160846

/-- Given that:
    1. A can complete the work in 15 days
    2. A and B working together for 5 days complete 0.5833333333333334 of the work
    Prove that B can complete the work alone in 20 days -/
theorem work_completion_time (a_time : ℝ) (b_time : ℝ) 
  (h1 : a_time = 15)
  (h2 : 5 * (1 / a_time + 1 / b_time) = 0.5833333333333334) :
  b_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1608_160846


namespace NUMINAMATH_CALUDE_sum_of_integers_l1608_160826

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
  (x : ℝ) + y = Real.sqrt 565 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1608_160826
