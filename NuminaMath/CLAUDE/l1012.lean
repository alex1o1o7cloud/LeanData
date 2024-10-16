import Mathlib

namespace NUMINAMATH_CALUDE_sin_double_angle_for_point_on_terminal_side_l1012_101296

theorem sin_double_angle_for_point_on_terminal_side :
  ∀ α : ℝ,
  let P : ℝ × ℝ := (-4, -6 * Real.sin (150 * π / 180))
  (P.1 = -4 ∧ P.2 = -6 * Real.sin (150 * π / 180)) →
  Real.sin (2 * α) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_point_on_terminal_side_l1012_101296


namespace NUMINAMATH_CALUDE_find_n_l1012_101200

theorem find_n : ∃ n : ℝ, (256 : ℝ)^(1/4) = 4^n ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_find_n_l1012_101200


namespace NUMINAMATH_CALUDE_divisor_of_smallest_six_digit_multiple_l1012_101203

def smallest_six_digit_number : Nat := 100000
def given_number : Nat := 100011
def divisor : Nat := 33337

theorem divisor_of_smallest_six_digit_multiple :
  (given_number = smallest_six_digit_number + 11) →
  (∀ n : Nat, n < given_number → n < smallest_six_digit_number ∨ given_number % n ≠ 0) →
  (given_number % divisor = 0) →
  (given_number / divisor = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisor_of_smallest_six_digit_multiple_l1012_101203


namespace NUMINAMATH_CALUDE_banana_apple_ratio_l1012_101232

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Checks if the fruit basket satisfies the given conditions -/
def validBasket (basket : FruitBasket) : Prop :=
  basket.oranges = 6 ∧
  basket.apples = basket.oranges - 2 ∧
  basket.peaches * 2 = basket.bananas ∧
  basket.oranges + basket.apples + basket.bananas + basket.peaches = 28

/-- Theorem stating that in a valid fruit basket, the ratio of bananas to apples is 3:1 -/
theorem banana_apple_ratio (basket : FruitBasket) (h : validBasket basket) :
  basket.bananas = 3 * basket.apples := by
  sorry

end NUMINAMATH_CALUDE_banana_apple_ratio_l1012_101232


namespace NUMINAMATH_CALUDE_logarithm_identity_l1012_101290

theorem logarithm_identity (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log x / Real.log (a * b) = (Real.log x / Real.log a * Real.log x / Real.log b) /
    (Real.log x / Real.log a + Real.log x / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_identity_l1012_101290


namespace NUMINAMATH_CALUDE_red_to_blue_bead_ratio_l1012_101224

theorem red_to_blue_bead_ratio :
  let red_beads : ℕ := 30
  let blue_beads : ℕ := 20
  (red_beads : ℚ) / blue_beads = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_to_blue_bead_ratio_l1012_101224


namespace NUMINAMATH_CALUDE_weeks_to_buy_bike_l1012_101276

def bike_cost : ℕ := 650
def birthday_money : ℕ := 60 + 45 + 25
def weekly_earnings : ℕ := 20

theorem weeks_to_buy_bike :
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = bike_cost ∧ weeks = 26 :=
by sorry

end NUMINAMATH_CALUDE_weeks_to_buy_bike_l1012_101276


namespace NUMINAMATH_CALUDE_harry_terry_calculation_l1012_101297

theorem harry_terry_calculation (x : ℤ) : 
  let H := 12 - (3 + 7) + x
  let T := 12 - 3 + 7 + x
  H - T + x = -14 + x := by
sorry

end NUMINAMATH_CALUDE_harry_terry_calculation_l1012_101297


namespace NUMINAMATH_CALUDE_reflect_point_5_neg3_l1012_101228

/-- Given a point P in the Cartesian coordinate system, 
    this function returns its coordinates with respect to the y-axis. -/
def reflect_y_axis (x y : ℝ) : ℝ × ℝ := (-x, y)

/-- The coordinates of P(5,-3) with respect to the y-axis are (-5,-3). -/
theorem reflect_point_5_neg3 : 
  reflect_y_axis 5 (-3) = (-5, -3) := by sorry

end NUMINAMATH_CALUDE_reflect_point_5_neg3_l1012_101228


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1012_101259

def I : Set ℕ := Set.univ

def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 10}

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def B : Set ℕ := {x | isPrime x}

theorem intersection_complement_equals_set : A ∩ (I \ B) = {4, 6, 8, 9, 10} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1012_101259


namespace NUMINAMATH_CALUDE_square_plus_square_l1012_101279

theorem square_plus_square (x : ℝ) : x^2 + x^2 = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_square_l1012_101279


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1012_101212

/-- An isosceles triangle with two given side lengths -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side1_pos : side1 > 0
  side2_pos : side2 > 0

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : Set ℝ :=
  if t.side1 = t.side2 then
    {2 * t.side1 + t.side2}
  else
    {2 * t.side1 + t.side2, t.side1 + 2 * t.side2}

theorem isosceles_triangle_perimeter :
  ∀ (t : IsoscelesTriangle),
    (t.side1 = 4 ∧ t.side2 = 6) ∨ (t.side1 = 6 ∧ t.side2 = 4) →
      perimeter t = {14, 16} ∧
    (t.side1 = 2 ∧ t.side2 = 6) ∨ (t.side1 = 6 ∧ t.side2 = 2) →
      perimeter t = {14} :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1012_101212


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1012_101258

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 && y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 2 * Real.sqrt 3 ∧ θ = 5 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1012_101258


namespace NUMINAMATH_CALUDE_min_value_of_a_l1012_101257

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≤ a) → 
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1012_101257


namespace NUMINAMATH_CALUDE_product_equals_32_l1012_101282

theorem product_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l1012_101282


namespace NUMINAMATH_CALUDE_cubic_sum_coefficients_l1012_101268

/-- A cubic function f(x) = ax^3 + bx^2 + cx + d -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_sum_coefficients (a b c d : ℝ) :
  (∀ x, cubic_function a b c d (x + 2) = 2 * x^3 - x^2 + 5 * x + 3) →
  a + b + c + d = -5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_coefficients_l1012_101268


namespace NUMINAMATH_CALUDE_man_swimming_speed_l1012_101214

/-- The speed of a man in still water, given his downstream and upstream swimming distances and times. -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h1 : downstream_distance = 51) 
  (h2 : upstream_distance = 18) 
  (h3 : time = 3) :
  ∃ (man_speed stream_speed : ℝ), 
    downstream_distance = (man_speed + stream_speed) * time ∧ 
    upstream_distance = (man_speed - stream_speed) * time ∧ 
    man_speed = 11.5 := by
sorry

end NUMINAMATH_CALUDE_man_swimming_speed_l1012_101214


namespace NUMINAMATH_CALUDE_equation_system_solutions_l1012_101255

/-- A system of two equations with two unknowns x and y -/
def equation_system (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

/-- The theorem stating that the equation system has only three specific solutions -/
theorem equation_system_solutions :
  ∀ x y : ℝ, equation_system x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l1012_101255


namespace NUMINAMATH_CALUDE_willow_play_time_l1012_101231

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Willow played football in minutes -/
def football_time : ℕ := 60

/-- The time Willow played basketball in minutes -/
def basketball_time : ℕ := 60

/-- The total time Willow played in hours -/
def total_time_hours : ℚ := (football_time + basketball_time) / minutes_per_hour

theorem willow_play_time : total_time_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_willow_play_time_l1012_101231


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1012_101242

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3 > 0) →
  (2*x - 1 > 0) →
  (x + 3) * (2*x - 1) = 12*x + 5 →
  x = (7 + Real.sqrt 113) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1012_101242


namespace NUMINAMATH_CALUDE_s_of_one_eq_394_div_25_l1012_101223

/-- Given functions t and s, prove that s(1) = 394/25 -/
theorem s_of_one_eq_394_div_25 
  (t : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h1 : ∀ x, t x = 5 * x - 12)
  (h2 : ∀ x, s (t x) = x^2 + 5 * x - 4) :
  s 1 = 394 / 25 := by
  sorry

end NUMINAMATH_CALUDE_s_of_one_eq_394_div_25_l1012_101223


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1012_101280

theorem tic_tac_toe_tie_probability (john_win_prob martha_win_prob : ℚ) 
  (h1 : john_win_prob = 4 / 9)
  (h2 : martha_win_prob = 5 / 12) :
  1 - (john_win_prob + martha_win_prob) = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1012_101280


namespace NUMINAMATH_CALUDE_investment_problem_l1012_101204

/-- Represents the investment scenario and proves A's investment amount --/
theorem investment_problem (x : ℝ) : 
  x > 0 ∧ 
  (x * 12) / (x * 12 + 200 * 6) * 100 = 60 →
  x = 150 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l1012_101204


namespace NUMINAMATH_CALUDE_correct_popularity_order_l1012_101252

/-- Represents the activities available for the sports day --/
inductive Activity
| dodgeball
| chessTournament
| track
| swimming

/-- Returns the fraction of students preferring a given activity --/
def preference (a : Activity) : Rat :=
  match a with
  | Activity.dodgeball => 3/8
  | Activity.chessTournament => 9/24
  | Activity.track => 5/16
  | Activity.swimming => 1/3

/-- Compares two activities based on their popularity --/
def morePopularThan (a b : Activity) : Prop :=
  preference a > preference b

/-- States that the given order of activities is correct based on popularity --/
theorem correct_popularity_order :
  morePopularThan Activity.swimming Activity.dodgeball ∧
  morePopularThan Activity.dodgeball Activity.chessTournament ∧
  morePopularThan Activity.chessTournament Activity.track :=
by sorry

end NUMINAMATH_CALUDE_correct_popularity_order_l1012_101252


namespace NUMINAMATH_CALUDE_parking_lot_motorcycles_l1012_101253

theorem parking_lot_motorcycles :
  let total_vehicles : ℕ := 24
  let total_wheels : ℕ := 86
  let car_wheels : ℕ := 4
  let motorcycle_wheels : ℕ := 3
  ∃ (cars motorcycles : ℕ),
    cars + motorcycles = total_vehicles ∧
    car_wheels * cars + motorcycle_wheels * motorcycles = total_wheels ∧
    motorcycles = 10 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_motorcycles_l1012_101253


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1012_101267

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 2*x + 5)}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 4}

-- Define the universal set U
def U : Type := ℝ

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioo (-1 : ℝ) (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1012_101267


namespace NUMINAMATH_CALUDE_initial_men_correct_l1012_101210

/-- The number of days it takes to dig the entire tunnel with the initial workforce -/
def initial_days : ℝ := 30

/-- The number of days worked before adding more men -/
def days_before_addition : ℝ := 10

/-- The number of additional men added to the workforce -/
def additional_men : ℕ := 20

/-- The number of days it takes to complete the tunnel after adding more men -/
def remaining_days : ℝ := 10.000000000000002

/-- The initial number of men digging the tunnel -/
def initial_men : ℕ := 6

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct :
  (initial_men : ℝ) * initial_days =
    (initial_men + additional_men) * remaining_days * (2/3) :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l1012_101210


namespace NUMINAMATH_CALUDE_james_total_spent_l1012_101250

def entry_fee : ℕ := 20
def num_rounds : ℕ := 2
def num_friends : ℕ := 5
def james_drinks : ℕ := 6
def drink_cost : ℕ := 6
def food_cost : ℕ := 14
def tip_percentage : ℚ := 30 / 100

def total_spent : ℕ := 163

theorem james_total_spent : 
  entry_fee + 
  (num_rounds * num_friends * drink_cost) + 
  (james_drinks * drink_cost) + 
  food_cost + 
  (((num_rounds * num_friends * drink_cost) + (james_drinks * drink_cost) + food_cost : ℚ) * tip_percentage).floor = 
  total_spent := by
  sorry

end NUMINAMATH_CALUDE_james_total_spent_l1012_101250


namespace NUMINAMATH_CALUDE_cone_base_radius_l1012_101237

/-- Given a cone with surface area 24π cm² and its lateral surface unfolded is a semicircle,
    the radius of the base circle is 2√2 cm. -/
theorem cone_base_radius (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  (π * r^2 + π * r * l = 24 * π) → 
  (π * l = 2 * π * r) → 
  r = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1012_101237


namespace NUMINAMATH_CALUDE_factor_implies_absolute_value_l1012_101217

/-- Given a polynomial 3x^4 - mx^2 + nx - p with factors (x-3) and (x+4), 
    prove that |m+2n-4p| = 20 -/
theorem factor_implies_absolute_value (m n p : ℤ) : 
  (∃ (a b : ℤ), (3 * X^4 - m * X^2 + n * X - p) = 
    (X - 3) * (X + 4) * (a * X^2 + b * X + (3 * a - 4 * b))) →
  |m + 2*n - 4*p| = 20 := by
  sorry


end NUMINAMATH_CALUDE_factor_implies_absolute_value_l1012_101217


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1012_101246

/-- The curve function f(x) = x³ + 1 -/
def f (x : ℝ) : ℝ := x^3 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3 * x - y + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1012_101246


namespace NUMINAMATH_CALUDE_modulus_of_z_l1012_101272

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem modulus_of_z : 
  let z : ℂ := (7 - Complex.I) / (1 + Complex.I)
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1012_101272


namespace NUMINAMATH_CALUDE_triangle_third_side_l1012_101271

theorem triangle_third_side (a b x : ℝ) : 
  a = 3 ∧ b = 9 ∧ 
  (a + b > x) ∧ (a + x > b) ∧ (b + x > a) →
  x = 10 → True :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l1012_101271


namespace NUMINAMATH_CALUDE_expression_simplification_l1012_101291

theorem expression_simplification :
  ∀ x : ℝ, ((3*x^2 + 2*x - 1) + x^2*2)*4 + (5 - 2/2)*(3*x^2 + 6*x - 8) = 32*x^2 + 32*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1012_101291


namespace NUMINAMATH_CALUDE_min_rooms_in_apartment_l1012_101284

/-- Represents an apartment with rooms and doors. -/
structure Apartment where
  rooms : ℕ
  doors : ℕ
  at_most_one_door_between_rooms : Bool
  at_most_one_door_to_outside : Bool

/-- Checks if the apartment configuration is valid. -/
def is_valid_apartment (a : Apartment) : Prop :=
  a.at_most_one_door_between_rooms ∧
  a.at_most_one_door_to_outside ∧
  a.doors = 12

/-- Theorem: The minimum number of rooms in a valid apartment is 5. -/
theorem min_rooms_in_apartment (a : Apartment) 
  (h : is_valid_apartment a) : a.rooms ≥ 5 := by
  sorry

#check min_rooms_in_apartment

end NUMINAMATH_CALUDE_min_rooms_in_apartment_l1012_101284


namespace NUMINAMATH_CALUDE_total_distance_is_176_l1012_101229

/-- Represents a series of linked rings with specific properties -/
structure LinkedRings where
  thickness : ℝ
  topOutsideDiameter : ℝ
  smallestOutsideDiameter : ℝ
  diameterDecrease : ℝ

/-- Calculates the total vertical distance covered by the linked rings -/
def totalVerticalDistance (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total vertical distance is 176 cm for the given conditions -/
theorem total_distance_is_176 (rings : LinkedRings) 
  (h1 : rings.thickness = 2)
  (h2 : rings.topOutsideDiameter = 30)
  (h3 : rings.smallestOutsideDiameter = 10)
  (h4 : rings.diameterDecrease = 2) :
  totalVerticalDistance rings = 176 :=
sorry

end NUMINAMATH_CALUDE_total_distance_is_176_l1012_101229


namespace NUMINAMATH_CALUDE_book_selection_combinations_l1012_101289

theorem book_selection_combinations :
  let mystery_books : ℕ := 3
  let fantasy_books : ℕ := 4
  let biography_books : ℕ := 3
  let total_combinations := mystery_books * fantasy_books * biography_books
  total_combinations = 36 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l1012_101289


namespace NUMINAMATH_CALUDE_average_of_numbers_sixth_and_seventh_sum_l1012_101285

def numbers : List ℝ := [54, 55, 57, 58, 59, 63, 65, 65]

theorem average_of_numbers : 
  (List.sum numbers) / (List.length numbers : ℝ) = 60 :=
by sorry

theorem sixth_and_seventh_sum : 
  List.sum (List.drop 5 (List.take 7 numbers)) = 54 :=
by sorry

#check average_of_numbers
#check sixth_and_seventh_sum

end NUMINAMATH_CALUDE_average_of_numbers_sixth_and_seventh_sum_l1012_101285


namespace NUMINAMATH_CALUDE_second_number_calculation_l1012_101294

theorem second_number_calculation (A B : ℝ) : 
  A = 3200 → 
  0.1 * A = 0.2 * B + 190 → 
  B = 650 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l1012_101294


namespace NUMINAMATH_CALUDE_problem_solution_l1012_101298

theorem problem_solution (k a b c x y z : ℝ) 
  (hk : k ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h1 : x * y = a * k * (x + y))
  (h2 : x * z = b * k * (x + z))
  (h3 : y * z = c * k * (y + z)) :
  x = (2 * a * b * c) / (k * (a * c + b * c - a * b)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1012_101298


namespace NUMINAMATH_CALUDE_modified_geometric_series_sum_l1012_101222

/-- The sum of a modified geometric series -/
theorem modified_geometric_series_sum 
  (a r : ℝ) 
  (h_r : -1 < r ∧ r < 1) :
  let series_sum : ℕ → ℝ := λ n => a^2 * r^(3*n)
  ∑' n, series_sum n = a^2 / (1 - r^3) := by
  sorry

end NUMINAMATH_CALUDE_modified_geometric_series_sum_l1012_101222


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_l1012_101234

theorem negation_of_existence_is_universal : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_l1012_101234


namespace NUMINAMATH_CALUDE_find_divisor_l1012_101219

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) : 
  dividend = quotient * divisor + remainder → 
  dividend = 22 → 
  quotient = 7 → 
  remainder = 1 → 
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1012_101219


namespace NUMINAMATH_CALUDE_operation_result_l1012_101248

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem operation_result :
  op (op Element.three Element.one) (op Element.four Element.two) = Element.three := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1012_101248


namespace NUMINAMATH_CALUDE_nonnegative_integer_pairs_l1012_101295

theorem nonnegative_integer_pairs (x y : ℕ) : (x * y + 2)^2 = x^2 + y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_integer_pairs_l1012_101295


namespace NUMINAMATH_CALUDE_book_order_theorem_l1012_101249

/-- Represents the book "Journey to the West" -/
inductive JourneyToWest

/-- Represents the book "Morning Blossoms and Evening Blossoms" -/
inductive MorningBlossoms

/-- Represents the initial order of books -/
structure InitialOrder where
  mb_cost : ℕ
  jw_cost : ℕ
  mb_price_ratio : ℚ
  mb_quantity_diff : ℕ

/-- Represents the additional order constraints -/
structure AdditionalOrderConstraints where
  total_books : ℕ
  min_mb : ℕ
  max_cost : ℕ

/-- Calculates the unit prices based on the initial order -/
def calculate_unit_prices (order : InitialOrder) : ℚ × ℚ :=
  sorry

/-- Calculates the number of possible ordering schemes and the lowest total cost -/
def calculate_additional_order (constraints : AdditionalOrderConstraints) (mb_price jw_price : ℚ) : ℕ × ℕ :=
  sorry

theorem book_order_theorem (initial_order : InitialOrder) (constraints : AdditionalOrderConstraints) :
  initial_order.mb_cost = 14000 ∧
  initial_order.jw_cost = 7000 ∧
  initial_order.mb_price_ratio = 1.4 ∧
  initial_order.mb_quantity_diff = 300 ∧
  constraints.total_books = 10 ∧
  constraints.min_mb = 3 ∧
  constraints.max_cost = 124 →
  let (jw_price, mb_price) := calculate_unit_prices initial_order
  let (schemes, lowest_cost) := calculate_additional_order constraints mb_price jw_price
  jw_price = 10 ∧ mb_price = 14 ∧ schemes = 4 ∧ lowest_cost = 112 :=
sorry

end NUMINAMATH_CALUDE_book_order_theorem_l1012_101249


namespace NUMINAMATH_CALUDE_root_implies_m_value_l1012_101281

theorem root_implies_m_value (m : ℝ) : 
  (Complex.I + 1)^2 + m * (Complex.I + 1) + 2 = 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l1012_101281


namespace NUMINAMATH_CALUDE_first_investment_rate_l1012_101206

/-- Represents the interest rate problem --/
structure InterestRateProblem where
  firstInvestment : ℝ
  secondInvestment : ℝ
  totalInterest : ℝ
  knownRate : ℝ
  firstRate : ℝ

/-- The interest rate problem satisfies the given conditions --/
def validProblem (p : InterestRateProblem) : Prop :=
  p.secondInvestment = p.firstInvestment - 100 ∧
  p.secondInvestment = 400 ∧
  p.knownRate = 0.07 ∧
  p.totalInterest = 73 ∧
  p.firstInvestment * p.firstRate + p.secondInvestment * p.knownRate = p.totalInterest

/-- The theorem stating that the first investment's interest rate is 0.15 --/
theorem first_investment_rate (p : InterestRateProblem) 
  (h : validProblem p) : p.firstRate = 0.15 := by
  sorry


end NUMINAMATH_CALUDE_first_investment_rate_l1012_101206


namespace NUMINAMATH_CALUDE_set_equality_l1012_101211

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def P : Set ℕ := {1, 3, 6}

theorem set_equality : (U \ M) ∩ (U \ P) = {2, 7, 8} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1012_101211


namespace NUMINAMATH_CALUDE_quadratic_form_nonnegative_l1012_101244

theorem quadratic_form_nonnegative (a b c : ℝ) :
  (∀ (f g : ℝ × ℝ), a * (f.1^2 + f.2^2) + b * (f.1 * g.1 + f.2 * g.2) + c * (g.1^2 + g.2^2) ≥ 0) ↔
  (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_nonnegative_l1012_101244


namespace NUMINAMATH_CALUDE_triangle_properties_l1012_101243

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = π) →
  (c * Real.sin B = b * Real.cos (C - π/6) ∨
   Real.cos B = (2*a - b) / (2*c) ∨
   (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 3 * a * b) →
  (Real.sin C = Real.sqrt 3 / 2 ∧
   (c = Real.sqrt 3 ∧ a = 3*b → 
    1/2 * a * b * Real.sin C = 9 * Real.sqrt 3 / 28)) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l1012_101243


namespace NUMINAMATH_CALUDE_binomial_coefficient_8_4_l1012_101277

theorem binomial_coefficient_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_8_4_l1012_101277


namespace NUMINAMATH_CALUDE_distance_between_walkers_l1012_101264

/-- Proves the distance between two people walking towards each other after a given time -/
theorem distance_between_walkers 
  (playground_length : ℝ) 
  (speed_hyosung : ℝ) 
  (speed_mimi : ℝ) 
  (time : ℝ) 
  (h1 : playground_length = 2.5)
  (h2 : speed_hyosung = 0.08)
  (h3 : speed_mimi = 2.4 / 60)
  (h4 : time = 15) :
  playground_length - (speed_hyosung + speed_mimi) * time = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_walkers_l1012_101264


namespace NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l1012_101218

theorem geometric_mean_of_3_and_12 :
  let b : ℝ := 3
  let c : ℝ := 12
  Real.sqrt (b * c) = 6 := by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l1012_101218


namespace NUMINAMATH_CALUDE_solution_xy_l1012_101275

theorem solution_xy (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x + y ≠ 0) 
  (h3 : (x + y) / x = y / (x + y)) 
  (h4 : x = 2 * y) : 
  x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_solution_xy_l1012_101275


namespace NUMINAMATH_CALUDE_harry_friday_speed_l1012_101205

-- Define Harry's running speeds
def monday_speed : ℝ := 10
def tuesday_to_thursday_increase : ℝ := 0.5
def friday_increase : ℝ := 0.6

-- Define the function to calculate speed increase
def speed_increase (base_speed : ℝ) (increase_percentage : ℝ) : ℝ :=
  base_speed * (1 + increase_percentage)

-- Theorem statement
theorem harry_friday_speed :
  let tuesday_to_thursday_speed := speed_increase monday_speed tuesday_to_thursday_increase
  let friday_speed := speed_increase tuesday_to_thursday_speed friday_increase
  friday_speed = 24 := by sorry

end NUMINAMATH_CALUDE_harry_friday_speed_l1012_101205


namespace NUMINAMATH_CALUDE_range_of_m_for_union_equality_l1012_101220

/-- The set A of solutions to x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}

/-- The set B of solutions to x^2 - 2x + m = 0, parameterized by m -/
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + m = 0}

/-- The theorem stating the range of m for which A ∪ B = A -/
theorem range_of_m_for_union_equality :
  {m : ℝ | A ∪ B m = A} = {m : ℝ | m ≥ 1} := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_union_equality_l1012_101220


namespace NUMINAMATH_CALUDE_min_q_value_l1012_101230

def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_q_value (a : ℕ) :
  (∀ x, 1 ≤ x ∧ x < a → q x < 1/2) ∧ q a ≥ 1/2 → a = 7 :=
sorry

end NUMINAMATH_CALUDE_min_q_value_l1012_101230


namespace NUMINAMATH_CALUDE_country_y_total_exports_l1012_101216

/-- Proves that the total yearly exports of country Y are $127.5 million given the specified conditions -/
theorem country_y_total_exports :
  ∀ (total_exports : ℝ),
  (0.2 * total_exports * (1/6) = 4.25) →
  total_exports = 127.5 := by
sorry

end NUMINAMATH_CALUDE_country_y_total_exports_l1012_101216


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l1012_101263

/-- Given a circle with center (2,3) and a point (8,7) on the circle,
    the slope of the line tangent to the circle at (8,7) is -3/2. -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (2, 3) →
  point = (8, 7) →
  (((point.2 - center.2) / (point.1 - center.1)) * (-1 / ((point.2 - center.2) / (point.1 - center.1)))) = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l1012_101263


namespace NUMINAMATH_CALUDE_profit_percentage_formula_l1012_101262

theorem profit_percentage_formula (C S M n : ℝ) (P : ℝ) 
  (h1 : S > 0) 
  (h2 : C > 0)
  (h3 : n > 0)
  (h4 : M = (2 / n) * C) 
  (h5 : P = (M / S) * 100) :
  P = 200 / (n + 2) := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_formula_l1012_101262


namespace NUMINAMATH_CALUDE_hillarys_deposit_l1012_101256

/-- Hillary's flea market earnings and deposit problem -/
theorem hillarys_deposit (crafts_sold : ℕ) (price_per_craft extra_tip remaining_cash : ℝ) 
  (h1 : crafts_sold = 3)
  (h2 : price_per_craft = 12)
  (h3 : extra_tip = 7)
  (h4 : remaining_cash = 25) :
  let total_earnings := crafts_sold * price_per_craft + extra_tip
  total_earnings - remaining_cash = 18 := by sorry

end NUMINAMATH_CALUDE_hillarys_deposit_l1012_101256


namespace NUMINAMATH_CALUDE_magic_8_ball_theorem_l1012_101288

def magic_8_ball_probability : ℚ := 242112 / 823543

theorem magic_8_ball_theorem (n : ℕ) (k : ℕ) (p : ℚ) 
  (h1 : n = 7) 
  (h2 : k = 3) 
  (h3 : p = 3 / 7) :
  Nat.choose n k * p^k * (1 - p)^(n - k) = magic_8_ball_probability := by
  sorry

#check magic_8_ball_theorem

end NUMINAMATH_CALUDE_magic_8_ball_theorem_l1012_101288


namespace NUMINAMATH_CALUDE_cricket_player_innings_l1012_101286

/-- A cricket player's innings problem -/
theorem cricket_player_innings (current_average : ℚ) (next_innings_runs : ℕ) (average_increase : ℚ) :
  current_average = 25 →
  next_innings_runs = 121 →
  average_increase = 6 →
  (∃ n : ℕ, (n * current_average + next_innings_runs) / (n + 1) = current_average + average_increase ∧ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_cricket_player_innings_l1012_101286


namespace NUMINAMATH_CALUDE_johns_fee_calculation_l1012_101283

/-- The one-time sitting fee for John's Photo World -/
def johns_fee : ℝ := 125

/-- The price per sheet at John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The price per sheet at Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_fee : ℝ := 140

/-- The number of sheets being compared -/
def num_sheets : ℕ := 12

theorem johns_fee_calculation :
  johns_fee = sams_fee + num_sheets * sams_price_per_sheet - num_sheets * johns_price_per_sheet :=
by sorry

end NUMINAMATH_CALUDE_johns_fee_calculation_l1012_101283


namespace NUMINAMATH_CALUDE_combination_permutation_properties_l1012_101287

-- Define combination function
def C (n m : ℕ) : ℕ := 
  if m ≤ n then Nat.choose n m else 0

-- Define permutation function
def A (n m : ℕ) : ℕ := 
  if m ≤ n then Nat.factorial n / Nat.factorial (n - m) else 0

-- Theorem statement
theorem combination_permutation_properties (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (C n m = C n (n - m)) ∧
  (C (n + 1) m = C n (m - 1) + C n m) ∧
  (A n m = C n m * A m m) ∧
  (A (n + 1) (m + 1) ≠ (m + 1) * A n m) := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_properties_l1012_101287


namespace NUMINAMATH_CALUDE_axis_of_symmetry_y₂_greater_y₁_l1012_101239

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  t : ℝ
  y₁ : ℝ
  y₂ : ℝ
  h_a_pos : a > 0
  h_point : m = a * 2^2 + b * 2 + c
  h_axis : t = -b / (2 * a)
  h_y₁ : y₁ = a * (-1)^2 + b * (-1) + c
  h_y₂ : y₂ = a * 3^2 + b * 3 + c

/-- When m = c, the axis of symmetry is at x = 1 -/
theorem axis_of_symmetry (p : Parabola) (h : p.m = p.c) : p.t = 1 := by sorry

/-- When c < m, y₂ > y₁ -/
theorem y₂_greater_y₁ (p : Parabola) (h : p.c < p.m) : p.y₂ > p.y₁ := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_y₂_greater_y₁_l1012_101239


namespace NUMINAMATH_CALUDE_parallel_lines_point_l1012_101227

def line (m : ℝ) (b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ m b₁ b₂, l₁ = line m b₁ ∧ l₂ = line m b₂

def angle_of_inclination (l : Set (ℝ × ℝ)) (θ : ℝ) : Prop :=
  ∃ m b, l = line m b ∧ m = Real.tan θ

theorem parallel_lines_point (a : ℝ) : 
  let l₁ : Set (ℝ × ℝ) := line 1 0
  let l₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 = -2 ∧ p.2 = -1) ∨ (p.1 = 3 ∧ p.2 = a)}
  angle_of_inclination l₁ (π/4) → parallel l₁ l₂ → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_point_l1012_101227


namespace NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l1012_101241

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : f a 1 < 3 → -2/3 < a ∧ a < 4/3 := by sorry

-- Theorem for the lower bound of f(x)
theorem lower_bound_of_f (a x : ℝ) : a ≥ 1 → f a x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l1012_101241


namespace NUMINAMATH_CALUDE_bread_cost_l1012_101266

def total_cost : ℕ := 42
def banana_cost : ℕ := 12
def milk_cost : ℕ := 7
def apple_cost : ℕ := 14

theorem bread_cost : 
  total_cost - (banana_cost + milk_cost + apple_cost) = 9 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l1012_101266


namespace NUMINAMATH_CALUDE_parabola_translation_l1012_101235

/-- Represents a vertical translation of a function -/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- Represents a horizontal translation of a function -/
def horizontalTranslation (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x + h)

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := fun x ↦ x^2

/-- The resulting parabola after translation -/
def resultingParabola : ℝ → ℝ := fun x ↦ (x + 1)^2 + 3

theorem parabola_translation :
  verticalTranslation (horizontalTranslation originalParabola 1) 3 = resultingParabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1012_101235


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l1012_101245

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ∨
  (a > 0 ∧
    (∀ x y, x < y ∧ y < Real.log a → f a x > f a y) ∧
    (∀ x y, Real.log a < x ∧ x < y → f a x < f a y)) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ (∀ z, z ≠ x ∧ z ≠ y → f a z ≠ 0) ↔
    (0 < a ∧ a < 1) ∨ (a > 1)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l1012_101245


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1012_101238

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity 5/3, its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2) / a^2 = 25 / 9 →
  ∃ k : ℝ, k = 4/3 ∧ (∀ x y : ℝ, y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1012_101238


namespace NUMINAMATH_CALUDE_simplify_expression_l1012_101274

theorem simplify_expression : (5 + 7 + 8) / 3 - 2 / 3 = 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1012_101274


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1012_101269

/-- 
Proves that the speed of a train is 72 km/hr, given its length, 
the platform length it crosses, and the time it takes to cross the platform.
-/
theorem train_speed_calculation (train_length platform_length crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 220)
  (h3 : crossing_time = 26) :
  (train_length + platform_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1012_101269


namespace NUMINAMATH_CALUDE_total_population_two_villages_l1012_101209

/-- The total population of two villages given partial information about each village's population -/
theorem total_population_two_villages
  (village1_90_percent : ℝ)
  (village2_80_percent : ℝ)
  (h1 : village1_90_percent = 45000)
  (h2 : village2_80_percent = 64000) :
  (village1_90_percent / 0.9 + village2_80_percent / 0.8) = 130000 :=
by sorry

end NUMINAMATH_CALUDE_total_population_two_villages_l1012_101209


namespace NUMINAMATH_CALUDE_unique_function_property_l1012_101270

theorem unique_function_property (f : ℕ → ℕ) : 
  (∀ (a b c : ℕ), (f a + f b + f c - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) → 
  (∀ (a : ℕ), f a = a ^ 2) := by
sorry

end NUMINAMATH_CALUDE_unique_function_property_l1012_101270


namespace NUMINAMATH_CALUDE_function_transformation_l1012_101225

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) : 
  (∀ y, f (y + 1) = 3 * y + 2) → f x = 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l1012_101225


namespace NUMINAMATH_CALUDE_zoe_winter_clothing_l1012_101265

/-- The number of boxes of winter clothing Zoe has. -/
def num_boxes : ℕ := 8

/-- The number of scarves in each box. -/
def scarves_per_box : ℕ := 4

/-- The number of mittens in each box. -/
def mittens_per_box : ℕ := 6

/-- The total number of pieces of winter clothing Zoe has. -/
def total_pieces : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem zoe_winter_clothing :
  total_pieces = 80 :=
by sorry

end NUMINAMATH_CALUDE_zoe_winter_clothing_l1012_101265


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1012_101233

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 4 ∧
  ∀ (m : ℕ), m > 0 → m % 3 = 1 → m % 5 = 3 → m % 6 = 4 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1012_101233


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1012_101215

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}
def B : Set ℝ := {x | x > 5/2}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | x > 6} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1012_101215


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1012_101201

theorem isosceles_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  ∃ (s : Real), s > 0 ∧ Real.sin A = s ∧ Real.sin B = s := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1012_101201


namespace NUMINAMATH_CALUDE_aeroplane_speed_l1012_101236

theorem aeroplane_speed (distance : ℝ) (time1 : ℝ) (time2 : ℝ) (speed2 : ℝ) :
  time1 = 6 →
  time2 = 14 / 3 →
  speed2 = 540 →
  distance = speed2 * time2 →
  distance = (distance / time1) * time1 →
  distance / time1 = 420 := by
sorry

end NUMINAMATH_CALUDE_aeroplane_speed_l1012_101236


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1012_101221

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the folding and crease
structure Folding (rect : Rectangle) :=
  (A' : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

-- Define the given dimensions
def given_dimensions (rect : Rectangle) (fold : Folding rect) : Prop :=
  let (ax, ay) := rect.A
  let (ex, ey) := fold.E
  let (fx, fy) := fold.F
  let (cx, cy) := rect.C
  Real.sqrt ((ax - ex)^2 + (ay - ey)^2) = 6 ∧
  Real.sqrt ((ex - rect.B.1)^2 + (ey - rect.B.2)^2) = 15 ∧
  Real.sqrt ((cx - fx)^2 + (cy - fy)^2) = 5

-- Define the theorem
theorem rectangle_perimeter (rect : Rectangle) (fold : Folding rect) :
  given_dimensions rect fold →
  (let perimeter := 2 * (Real.sqrt ((rect.A.1 - rect.B.1)^2 + (rect.A.2 - rect.B.2)^2) +
                         Real.sqrt ((rect.B.1 - rect.C.1)^2 + (rect.B.2 - rect.C.2)^2))
   perimeter = 808) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l1012_101221


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_l1012_101299

/-- The equation x²(x+y+1) = y²(x+y+1) represents three lines that do not all pass through a common point -/
theorem equation_represents_three_lines (x y : ℝ) : 
  (x^2 * (x + y + 1) = y^2 * (x + y + 1)) ↔ 
  ((y = -x) ∨ (y = x) ∨ (y = -x - 1)) ∧ 
  ¬(∃ p : ℝ × ℝ, (p.1 = p.2 ∧ p.2 = -p.1) ∧ 
                 (p.1 = -p.2 - 1 ∧ p.2 = p.1) ∧ 
                 (p.1 = p.2 ∧ p.2 = -p.1 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_l1012_101299


namespace NUMINAMATH_CALUDE_crayons_per_pack_l1012_101247

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) :
  total_crayons / num_packs = 15 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_pack_l1012_101247


namespace NUMINAMATH_CALUDE_max_b_value_l1012_101292

noncomputable section

variable (a b : ℝ)

def f (x : ℝ) := (3/2) * x^2 - 2*a*x

def g (x : ℝ) := a^2 * Real.log x + b

def common_point (x : ℝ) := f a x = g a b x

def common_tangent (x : ℝ) := deriv (f a) x = deriv (g a b) x

theorem max_b_value (h1 : a > 0) 
  (h2 : ∃ x > 0, common_point a b x ∧ common_tangent a b x) :
  ∃ b_max : ℝ, b_max = 1 / (2 * Real.exp 2) ∧ 
  (∀ b : ℝ, (∃ x > 0, common_point a b x ∧ common_tangent a b x) → b ≤ b_max) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l1012_101292


namespace NUMINAMATH_CALUDE_twelfth_root_of_unity_l1012_101261

open Complex

theorem twelfth_root_of_unity : 
  let z : ℂ := (Complex.tan (π / 6) + I) / (Complex.tan (π / 6) - I)
  z = exp (I * π / 3) ∧ z^12 = 1 := by sorry

end NUMINAMATH_CALUDE_twelfth_root_of_unity_l1012_101261


namespace NUMINAMATH_CALUDE_product_evaluation_l1012_101213

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1012_101213


namespace NUMINAMATH_CALUDE_cubic_equation_value_l1012_101208

theorem cubic_equation_value (a b : ℝ) :
  (a * (-2)^3 + b * (-2) - 7 = 9) →
  (a * 2^3 + b * 2 - 7 = -23) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l1012_101208


namespace NUMINAMATH_CALUDE_aquafaba_needed_l1012_101278

/-- The number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- The number of cakes Christine is making -/
def num_cakes : ℕ := 2

/-- The number of egg whites required for each cake -/
def egg_whites_per_cake : ℕ := 8

/-- Theorem stating the total number of tablespoons of aquafaba needed -/
theorem aquafaba_needed : 
  aquafaba_per_egg * num_cakes * egg_whites_per_cake = 32 := by
  sorry

end NUMINAMATH_CALUDE_aquafaba_needed_l1012_101278


namespace NUMINAMATH_CALUDE_folded_paper_distance_l1012_101251

theorem folded_paper_distance (paper_area : ℝ) (h_area : paper_area = 18) : 
  let side_length : ℝ := Real.sqrt paper_area
  let folded_leg : ℝ := Real.sqrt 12
  let distance : ℝ := Real.sqrt (2 * folded_leg ^ 2)
  distance = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l1012_101251


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1012_101273

theorem sum_of_fourth_powers (a b : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a * b = 5) : 
  a^4 + b^4 = 150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1012_101273


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1012_101226

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∃ a : ℝ, A ∩ B a = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1012_101226


namespace NUMINAMATH_CALUDE_initial_members_count_l1012_101240

/-- The number of initial earning members in a family -/
def initial_members : ℕ := sorry

/-- The initial average monthly income of the family -/
def initial_average : ℕ := 735

/-- The new average monthly income after one member died -/
def new_average : ℕ := 590

/-- The income of the deceased member -/
def deceased_income : ℕ := 1170

/-- Theorem stating the number of initial earning members -/
theorem initial_members_count : initial_members = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_members_count_l1012_101240


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1012_101260

def coin_tosses : ℕ := 8
def heads_count : ℕ := 3

theorem probability_three_heads_in_eight_tosses :
  (Nat.choose coin_tosses heads_count) / (2 ^ coin_tosses) = 7 / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1012_101260


namespace NUMINAMATH_CALUDE_football_yards_lost_l1012_101254

theorem football_yards_lost (yards_gained yards_progress : ℤ) 
  (h1 : yards_gained = 8)
  (h2 : yards_progress = 3) :
  ∃ yards_lost : ℤ, yards_lost + yards_gained = yards_progress ∧ yards_lost = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_football_yards_lost_l1012_101254


namespace NUMINAMATH_CALUDE_club_officer_selection_l1012_101293

/-- Represents the number of ways to choose officers in a club -/
def chooseOfficers (totalMembers boyCount girlCount : ℕ) : ℕ :=
  totalMembers * (if boyCount = girlCount then boyCount else 0) * (boyCount - 1)

/-- Theorem stating the number of ways to choose officers in the given conditions -/
theorem club_officer_selection :
  let totalMembers := 24
  let boyCount := 12
  let girlCount := 12
  chooseOfficers totalMembers boyCount girlCount = 3168 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l1012_101293


namespace NUMINAMATH_CALUDE_rectangle_division_exists_l1012_101207

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a division of a rectangle into parts -/
structure RectangleDivision where
  parts : List ℝ

/-- Checks if a division is valid for a given rectangle -/
def isValidDivision (r : Rectangle) (d : RectangleDivision) : Prop :=
  d.parts.length = 4 ∧ d.parts.sum = r.width * r.height

/-- The main theorem to be proved -/
theorem rectangle_division_exists : ∃ (d : RectangleDivision), 
  isValidDivision ⟨6, 10⟩ d ∧ 
  d.parts = [8, 12, 16, 24] := by
  sorry


end NUMINAMATH_CALUDE_rectangle_division_exists_l1012_101207


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1012_101202

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x^2 * |x| = 3*x + 4 ∧ 
  ∀ (y : ℝ), y^2 * |y| = 3*y + 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1012_101202
