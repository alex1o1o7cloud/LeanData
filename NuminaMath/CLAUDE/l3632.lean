import Mathlib

namespace NUMINAMATH_CALUDE_sum_perpendiculars_equals_altitude_l3632_363294

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a triangle is isosceles with AB = AC -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

/-- Calculates the altitude of a triangle -/
noncomputable def Triangle.altitude (t : Triangle) : ℝ := sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
noncomputable def perpendicularDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Checks if a point is inside or on a triangle -/
def Triangle.containsPoint (t : Triangle) (p : Point) : Prop := sorry

/-- Theorem: Sum of perpendiculars equals altitude for isosceles triangle -/
theorem sum_perpendiculars_equals_altitude (t : Triangle) (p : Point) :
  t.isIsosceles →
  t.containsPoint p →
  perpendicularDistance p t.B t.C + 
  perpendicularDistance p t.C t.A + 
  perpendicularDistance p t.A t.B = 
  t.altitude := by sorry

end NUMINAMATH_CALUDE_sum_perpendiculars_equals_altitude_l3632_363294


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l3632_363248

/-- Given a rectangular pen with a perimeter of 60 feet, the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * (x + y) = 60 →
  x * y ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l3632_363248


namespace NUMINAMATH_CALUDE_alison_has_4000_l3632_363296

-- Define the amounts of money for each person
def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := brittany_money / 2

-- Theorem statement
theorem alison_has_4000 : alison_money = 4000 := by
  sorry

end NUMINAMATH_CALUDE_alison_has_4000_l3632_363296


namespace NUMINAMATH_CALUDE_min_distance_is_3420_div_181_l3632_363269

/-- Triangle ABC with right angle at B, side lengths, and intersecting circles --/
structure RightTriangleWithCircles where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Angle condition
  right_angle : ab^2 + bc^2 = ac^2
  -- Side length values
  ab_eq : ab = 19
  bc_eq : bc = 180
  ac_eq : ac = 181
  -- Midpoints
  m : ℝ × ℝ  -- midpoint of AB
  n : ℝ × ℝ  -- midpoint of BC
  -- Intersection points
  d : ℝ × ℝ
  e : ℝ × ℝ
  p : ℝ × ℝ
  -- Conditions for D and E
  d_on_circle_m : (d.1 - m.1)^2 + (d.2 - m.2)^2 = (ac/2)^2
  d_on_circle_n : (d.1 - n.1)^2 + (d.2 - n.2)^2 = (ac/2)^2
  e_on_circle_m : (e.1 - m.1)^2 + (e.2 - m.2)^2 = (ac/2)^2
  e_on_circle_n : (e.1 - n.1)^2 + (e.2 - n.2)^2 = (ac/2)^2
  -- P is on AC
  p_on_ac : p.2 = 0
  -- DE intersects AC at P
  p_on_de : ∃ (t : ℝ), p = (1 - t) • d + t • e

/-- The minimum of DP and EP is 3420/181 --/
theorem min_distance_is_3420_div_181 (triangle : RightTriangleWithCircles) :
  min ((triangle.d.1 - triangle.p.1)^2 + (triangle.d.2 - triangle.p.2)^2)
      ((triangle.e.1 - triangle.p.1)^2 + (triangle.e.2 - triangle.p.2)^2) = (3420/181)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_is_3420_div_181_l3632_363269


namespace NUMINAMATH_CALUDE_special_tetrahedron_ratio_bounds_l3632_363276

/-- Represents a tetrahedron with specific edge length properties -/
structure SpecialTetrahedron where
  -- The edge lengths
  a : ℝ
  b : ℝ
  -- Conditions on edge lengths
  h_positive : 0 < a ∧ 0 < b
  h_pa_eq_pb : true  -- Represents PA = PB = a
  h_pc_eq_sides : true  -- Represents PC = AB = BC = CA = b
  h_a_lt_b : a < b

/-- The ratio a/b in a special tetrahedron is bounded -/
theorem special_tetrahedron_ratio_bounds (t : SpecialTetrahedron) :
  Real.sqrt (2 - Real.sqrt 3) < t.a / t.b ∧ t.a / t.b < 1 := by
  sorry


end NUMINAMATH_CALUDE_special_tetrahedron_ratio_bounds_l3632_363276


namespace NUMINAMATH_CALUDE_malaria_parasite_length_l3632_363252

theorem malaria_parasite_length : 0.0000015 = 1.5 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_malaria_parasite_length_l3632_363252


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3632_363287

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (|x| < 1 → x < a) ∧ ¬(x < a → |x| < 1)) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3632_363287


namespace NUMINAMATH_CALUDE_isosceles_triangle_lateral_side_length_l3632_363251

/-- Given an isosceles triangle with vertex angle α and the sum of two different heights l,
    the length of a lateral side is l * tan(α/2) / (1 + 2 * sin(α/2)). -/
theorem isosceles_triangle_lateral_side_length
  (α l : ℝ) (h_α : 0 < α ∧ α < π) (h_l : l > 0) :
  ∃ (side_length : ℝ),
    side_length = l * Real.tan (α / 2) / (1 + 2 * Real.sin (α / 2)) ∧
    ∃ (height1 height2 : ℝ),
      height1 + height2 = l ∧
      height1 ≠ height2 ∧
      ∃ (base : ℝ),
        height1 = side_length * Real.cos (α / 2) ∧
        height2 = base / 2 * Real.tan (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_lateral_side_length_l3632_363251


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3632_363217

/-- Given two regular polygons with the same perimeter, where the first has 24 sides
    and its side length is three times that of the second, prove the second has 72 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure side length is positive
  24 * (3 * s) = n * s →  -- Same perimeter condition
  n = 72 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3632_363217


namespace NUMINAMATH_CALUDE_part_one_part_two_l3632_363258

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part I
theorem part_one : 
  let a : ℝ := -4
  {x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 6} := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 3|) → -1 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3632_363258


namespace NUMINAMATH_CALUDE_boy_running_speed_l3632_363254

/-- The speed of a boy running around a square field -/
theorem boy_running_speed (side_length : Real) (time : Real) : 
  side_length = 60 → time = 72 → (4 * side_length) / time * (3600 / 1000) = 12 := by
  sorry

end NUMINAMATH_CALUDE_boy_running_speed_l3632_363254


namespace NUMINAMATH_CALUDE_bouquet_lilies_percentage_l3632_363216

theorem bouquet_lilies_percentage (F : ℚ) (F_pos : F > 0) : 
  let purple_flowers := (7 / 10) * F
  let purple_tulips := (1 / 2) * purple_flowers
  let yellow_flowers := F - purple_flowers
  let yellow_lilies := (2 / 3) * yellow_flowers
  let total_lilies := (purple_flowers - purple_tulips) + yellow_lilies
  (total_lilies / F) * 100 = 55 := by sorry

end NUMINAMATH_CALUDE_bouquet_lilies_percentage_l3632_363216


namespace NUMINAMATH_CALUDE_S_intersections_empty_l3632_363230

def S (n : ℕ) : Set ℕ :=
  {x | ∃ g : ℕ, g ≥ 2 ∧ x = (g^n - 1) / (g - 1)}

theorem S_intersections_empty :
  (S 3 ∩ S 4 = ∅) ∧ (S 3 ∩ S 5 = ∅) := by
  sorry

end NUMINAMATH_CALUDE_S_intersections_empty_l3632_363230


namespace NUMINAMATH_CALUDE_zain_coins_count_and_value_l3632_363205

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100
def half_dollar_value : ℚ := 50 / 100

def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def emerie_pennies : ℕ := 10
def emerie_half_dollars : ℕ := 2

def zain_more_coins : ℕ := 10

def zain_quarters : ℕ := emerie_quarters + zain_more_coins
def zain_dimes : ℕ := emerie_dimes + zain_more_coins
def zain_nickels : ℕ := emerie_nickels + zain_more_coins
def zain_pennies : ℕ := emerie_pennies + zain_more_coins
def zain_half_dollars : ℕ := emerie_half_dollars + zain_more_coins

def zain_total_coins : ℕ := zain_quarters + zain_dimes + zain_nickels + zain_pennies + zain_half_dollars

def zain_total_value : ℚ :=
  zain_quarters * quarter_value +
  zain_dimes * dime_value +
  zain_nickels * nickel_value +
  zain_pennies * penny_value +
  zain_half_dollars * half_dollar_value

theorem zain_coins_count_and_value :
  zain_total_coins = 80 ∧ zain_total_value ≤ 20 := by sorry

end NUMINAMATH_CALUDE_zain_coins_count_and_value_l3632_363205


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3632_363282

theorem quadratic_inequality_roots (k : ℝ) : 
  (∀ x : ℝ, -x^2 + k*x + 4 < 0 ↔ x < 2 ∨ x > 3) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3632_363282


namespace NUMINAMATH_CALUDE_bike_speed_l3632_363223

/-- Given a bike moving at a constant speed that covers 5400 meters in 9 minutes,
    prove that its speed is 10 meters per second. -/
theorem bike_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) 
    (h1 : distance = 5400)
    (h2 : time_minutes = 9)
    (h3 : speed = distance / (time_minutes * 60)) : 
    speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_bike_speed_l3632_363223


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3632_363239

theorem problem_1 : 2 * Real.cos (45 * π / 180) + (π - Real.sqrt 3) ^ 0 - Real.sqrt 8 = 1 - Real.sqrt 2 := by
  sorry

theorem problem_2 (m : ℝ) (h : m ≠ 1) : 
  ((2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1))) = (m - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3632_363239


namespace NUMINAMATH_CALUDE_toms_friend_decks_l3632_363214

/-- The problem of calculating how many decks Tom's friend bought -/
theorem toms_friend_decks :
  ∀ (cost_per_deck : ℕ) (toms_decks : ℕ) (total_spent : ℕ),
    cost_per_deck = 8 →
    toms_decks = 3 →
    total_spent = 64 →
    ∃ (friends_decks : ℕ),
      friends_decks * cost_per_deck + toms_decks * cost_per_deck = total_spent ∧
      friends_decks = 5 :=
by sorry

end NUMINAMATH_CALUDE_toms_friend_decks_l3632_363214


namespace NUMINAMATH_CALUDE_brick_width_is_10_l3632_363206

-- Define the dimensions of the brick and wall
def brick_length : ℝ := 20
def brick_height : ℝ := 7.5
def wall_length : ℝ := 2700  -- 27 m in cm
def wall_width : ℝ := 200    -- 2 m in cm
def wall_height : ℝ := 75    -- 0.75 m in cm
def num_bricks : ℕ := 27000

-- Theorem to prove the width of the brick
theorem brick_width_is_10 :
  ∃ (brick_width : ℝ),
    brick_width = 10 ∧
    brick_length * brick_width * brick_height * num_bricks =
    wall_length * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_brick_width_is_10_l3632_363206


namespace NUMINAMATH_CALUDE_bollards_contract_l3632_363247

theorem bollards_contract (total : ℕ) (installed : ℕ) (remaining : ℕ) : 
  installed = (3 * total) / 4 →
  remaining = 2000 →
  remaining = total / 4 →
  total = 8000 := by
  sorry

end NUMINAMATH_CALUDE_bollards_contract_l3632_363247


namespace NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l3632_363256

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 5

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l3632_363256


namespace NUMINAMATH_CALUDE_gold_calculation_l3632_363234

-- Define the amount of gold Greg has
def gregs_gold : ℕ := 20

-- Define Katie's gold in terms of Greg's
def katies_gold : ℕ := 4 * gregs_gold

-- Define the total amount of gold
def total_gold : ℕ := gregs_gold + katies_gold

-- Theorem to prove
theorem gold_calculation : total_gold = 100 := by
  sorry

end NUMINAMATH_CALUDE_gold_calculation_l3632_363234


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3632_363208

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x + y)^4 = a₁*x^4 + a₂*x^3*y + a₃*x^2*y^2 + a₄*x*y^3 + a₅*y^4) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3632_363208


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3632_363238

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not among the 12 lowest-scoring players
  total_players : ℕ := n + 12
  total_points : ℕ := n * (n - 1) + 132
  games_played : ℕ := (total_players * (total_players - 1)) / 2

/-- The theorem stating that the total number of players is 24 -/
theorem chess_tournament_players (t : ChessTournament) : t.total_players = 24 := by
  sorry

#check chess_tournament_players

end NUMINAMATH_CALUDE_chess_tournament_players_l3632_363238


namespace NUMINAMATH_CALUDE_smaller_bill_value_l3632_363278

/-- The value of the smaller denomination bill -/
def x : ℕ := sorry

/-- The total number of bills Anna has -/
def total_bills : ℕ := 12

/-- The number of smaller denomination bills Anna has -/
def smaller_bills : ℕ := 4

/-- The value of a $10 bill -/
def ten_dollar : ℕ := 10

/-- The total value of all bills in dollars -/
def total_value : ℕ := 100

theorem smaller_bill_value :
  x * smaller_bills + (total_bills - smaller_bills) * ten_dollar = total_value ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_bill_value_l3632_363278


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3632_363241

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 1 → a + b ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3632_363241


namespace NUMINAMATH_CALUDE_raritet_encounters_l3632_363242

/-- Represents the number of days it takes for a ferry to travel between Dzerzhinsk and Lvov --/
def travel_time : ℕ := 8

/-- Represents the number of ferries departing from Dzerzhinsk during Raritet's journey --/
def ferries_during_journey : ℕ := travel_time

/-- Represents the number of ferries already en route when Raritet departs --/
def ferries_en_route : ℕ := travel_time

/-- Represents the ferry arriving in Lvov when Raritet departs --/
def arriving_ferry : ℕ := 1

/-- Theorem stating the total number of ferries Raritet meets --/
theorem raritet_encounters :
  ferries_during_journey + ferries_en_route + arriving_ferry = 17 :=
sorry

end NUMINAMATH_CALUDE_raritet_encounters_l3632_363242


namespace NUMINAMATH_CALUDE_complex_magnitude_difference_zero_l3632_363281

theorem complex_magnitude_difference_zero : Complex.abs (3 - 5*Complex.I) - Complex.abs (3 + 5*Complex.I) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_difference_zero_l3632_363281


namespace NUMINAMATH_CALUDE_inequality_proof_l3632_363203

theorem inequality_proof (x y : ℝ) (h : x > y) : 
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3632_363203


namespace NUMINAMATH_CALUDE_sqrt2_minus_1_power_form_l3632_363222

theorem sqrt2_minus_1_power_form (n : ℕ) :
  ∃ k : ℤ, (Real.sqrt 2 - 1) ^ n = Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_minus_1_power_form_l3632_363222


namespace NUMINAMATH_CALUDE_inverse_variation_l3632_363272

/-- Given that r and s vary inversely, and s = 0.35 when r = 1200, 
    prove that s = 0.175 when r = 2400 -/
theorem inverse_variation (r s : ℝ) (h : r * s = 1200 * 0.35) :
  r = 2400 → s = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l3632_363272


namespace NUMINAMATH_CALUDE_area_of_polygon_AIHFGD_l3632_363210

-- Define the points
variable (A B C D E F G H I : ℝ × ℝ)

-- Define the squares
def is_square (P Q R S : ℝ × ℝ) : Prop := sorry

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Define midpoint
def is_midpoint (M P Q : ℝ × ℝ) : Prop := sorry

theorem area_of_polygon_AIHFGD :
  is_square A B C D →
  is_square E F G D →
  area [A, B, C, D] = 25 →
  area [E, F, G, D] = 25 →
  is_midpoint H B C →
  is_midpoint H E F →
  is_midpoint I A B →
  area [A, I, H, F, G, D] = 25 := by
  sorry

end NUMINAMATH_CALUDE_area_of_polygon_AIHFGD_l3632_363210


namespace NUMINAMATH_CALUDE_magazine_cost_lynne_magazine_cost_l3632_363211

/-- The cost of each magazine given Lynne's purchase details -/
theorem magazine_cost (cat_books : ℕ) (solar_books : ℕ) (magazines : ℕ) 
  (book_price : ℕ) (total_spent : ℕ) : ℕ :=
  let total_books := cat_books + solar_books
  let book_cost := total_books * book_price
  let magazine_total_cost := total_spent - book_cost
  magazine_total_cost / magazines

/-- Proof that each magazine costs $4 given Lynne's purchase details -/
theorem lynne_magazine_cost : 
  magazine_cost 7 2 3 7 75 = 4 := by
  sorry

end NUMINAMATH_CALUDE_magazine_cost_lynne_magazine_cost_l3632_363211


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3632_363232

/-- A quadratic function that intersects the x-axis at (0,0) and (-2,0) and has a minimum value of -1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x

theorem quadratic_function_properties :
  (f 0 = 0) ∧
  (f (-2) = 0) ∧
  (∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (∃ x₀, f x₀ = -1) ∧
  (∀ x, f x = x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3632_363232


namespace NUMINAMATH_CALUDE_tiffany_cans_l3632_363225

theorem tiffany_cans (monday_bags : ℕ) (next_day_bags : ℕ) 
  (h1 : monday_bags = 8) 
  (h2 : monday_bags = next_day_bags + 1) : 
  next_day_bags = 7 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_cans_l3632_363225


namespace NUMINAMATH_CALUDE_power_division_calculation_l3632_363246

theorem power_division_calculation : ((6^6 / 6^5)^3 * 8^3) / 4^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_power_division_calculation_l3632_363246


namespace NUMINAMATH_CALUDE_sally_pens_ratio_l3632_363257

def sally_pens_problem (initial_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) (pens_taken_home : ℕ) : Prop :=
  let pens_distributed := num_students * pens_per_student
  let pens_remaining := initial_pens - pens_distributed
  let pens_in_locker := pens_remaining - pens_taken_home
  pens_in_locker = pens_taken_home

theorem sally_pens_ratio : sally_pens_problem 342 44 7 17 := by
  sorry

end NUMINAMATH_CALUDE_sally_pens_ratio_l3632_363257


namespace NUMINAMATH_CALUDE_evaluate_expression_l3632_363228

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3632_363228


namespace NUMINAMATH_CALUDE_min_money_for_city_l3632_363298

/-- Represents the resources needed to build a city -/
structure CityResources where
  ore : ℕ
  wheat : ℕ

/-- Represents the market prices and exchange rates -/
structure MarketPrices where
  ore_price : ℕ
  wheat_bundle_price : ℕ
  wheat_bundle_size : ℕ
  wheat_to_ore_rate : ℕ

/-- The problem setup -/
def city_building_problem (work_days : ℕ) (daily_ore_production : ℕ) 
  (city_resources : CityResources) (market_prices : MarketPrices) : Prop :=
  ∃ (initial_money : ℕ),
    initial_money = 9 ∧
    work_days * daily_ore_production + 
    (market_prices.wheat_bundle_size - city_resources.wheat) = 
    city_resources.ore ∧
    initial_money + 
    (work_days * daily_ore_production - city_resources.ore) * market_prices.ore_price = 
    (market_prices.wheat_bundle_size / city_resources.wheat) * market_prices.wheat_bundle_price

/-- The theorem to be proved -/
theorem min_money_for_city : 
  city_building_problem 3 1 
    { ore := 3, wheat := 2 } 
    { ore_price := 3, 
      wheat_bundle_price := 12, 
      wheat_bundle_size := 3, 
      wheat_to_ore_rate := 1 } :=
by
  sorry


end NUMINAMATH_CALUDE_min_money_for_city_l3632_363298


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3632_363273

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a

theorem max_value_implies_a_equals_one :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≤ 1) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 1) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3632_363273


namespace NUMINAMATH_CALUDE_main_theorem_l3632_363236

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the main theorem
theorem main_theorem (x : ℝ) (h : (lg x)^2 * lg (10 * x) < 0) :
  (1 / lg (10 * x)) * Real.sqrt ((lg x)^2 + (lg (10 * x))^2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l3632_363236


namespace NUMINAMATH_CALUDE_concrete_volume_l3632_363290

-- Define constants
def sidewalk_width : ℚ := 4/3  -- in yards
def sidewalk_length : ℚ := 80/3  -- in yards
def sidewalk_thickness : ℚ := 1/9  -- in yards
def border_width : ℚ := 2/3  -- in yards (1 foot on each side)
def border_thickness : ℚ := 1/18  -- in yards

-- Define the theorem
theorem concrete_volume : 
  let sidewalk_volume := sidewalk_width * sidewalk_length * sidewalk_thickness
  let border_volume := border_width * sidewalk_length * border_thickness
  let total_volume := sidewalk_volume + border_volume
  ⌈total_volume⌉ = 6 := by
sorry


end NUMINAMATH_CALUDE_concrete_volume_l3632_363290


namespace NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_chord_length_y_axis_shortest_chord_equation_l3632_363212

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 16

-- Define the line l
def line_l (x y a : ℝ) : Prop := x - a * y + 3 * a - 2 = 0

-- Statement A
theorem line_l_passes_through_fixed_point (a : ℝ) :
  ∃ x y, line_l x y a ∧ x = 2 ∧ y = 3 :=
sorry

-- Statement B
theorem chord_length_y_axis :
  ∃ y₁ y₂, circle_C 0 y₁ ∧ circle_C 0 y₂ ∧ y₂ - y₁ = 2 * Real.sqrt 15 :=
sorry

-- Statement D
theorem shortest_chord_equation (a : ℝ) :
  (∀ x y, line_l x y a → circle_C x y → 
    ∀ x' y', line_l x' y' a → circle_C x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - (-1))^2 + (y - 1)^2) →
  ∃ k, a = -3/2 ∧ k * (3 * x + 2 * y - 12) = x - a * y + 3 * a - 2 :=
sorry

end NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_chord_length_y_axis_shortest_chord_equation_l3632_363212


namespace NUMINAMATH_CALUDE_equilateral_triangle_locus_l3632_363262

-- Define an equilateral triangle ABC
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the reflection of a point over a line
def ReflectPointOverLine (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the set of points P satisfying PA^2 = PB^2 + PC^2
def SatisfyingPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist P A ^ 2 = dist P B ^ 2 + dist P C ^ 2}

-- Theorem statement
theorem equilateral_triangle_locus 
  (A B C : ℝ × ℝ) 
  (h : EquilateralTriangle A B C) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = ReflectPointOverLine A B C ∧ 
    radius = dist A B ∧
    SatisfyingPoints A B C = {P : ℝ × ℝ | dist P center = radius} :=
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_locus_l3632_363262


namespace NUMINAMATH_CALUDE_sports_day_participation_l3632_363226

/-- Given that the number of participants in a school sports day this year (m) 
    is a 10% increase from last year, prove that the number of participants 
    last year was m / (1 + 10%). -/
theorem sports_day_participation (m : ℝ) : 
  let last_year := m / (1 + 10 / 100)
  let increase_rate := 10 / 100
  m = last_year * (1 + increase_rate) → 
  last_year = m / (1 + increase_rate) := by
sorry


end NUMINAMATH_CALUDE_sports_day_participation_l3632_363226


namespace NUMINAMATH_CALUDE_max_notebooks_purchasable_l3632_363229

def available_funds : ℚ := 21.45
def notebook_cost : ℚ := 2.75

theorem max_notebooks_purchasable :
  ∀ n : ℕ, (n : ℚ) * notebook_cost ≤ available_funds ↔ n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchasable_l3632_363229


namespace NUMINAMATH_CALUDE_circle_max_distance_l3632_363297

/-- Given a circle with equation x^2 + y^2 + 4x - 2y - 4 = 0, 
    the maximum value of x^2 + y^2 is 14 + 6√5 -/
theorem circle_max_distance (x y : ℝ) : 
  x^2 + y^2 + 4*x - 2*y - 4 = 0 → 
  x^2 + y^2 ≤ 14 + 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_max_distance_l3632_363297


namespace NUMINAMATH_CALUDE_trees_difference_l3632_363267

theorem trees_difference (initial_trees : ℕ) (died_trees : ℕ) : 
  initial_trees = 14 → died_trees = 9 → died_trees - (initial_trees - died_trees) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trees_difference_l3632_363267


namespace NUMINAMATH_CALUDE_eighth_of_2_38_l3632_363213

theorem eighth_of_2_38 (x : ℕ) :
  (1 / 8 : ℝ) * (2 : ℝ)^38 = (2 : ℝ)^x → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_2_38_l3632_363213


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3632_363244

theorem quadratic_roots_relation (c d : ℚ) : 
  (∃ r s : ℚ, r + s = 3/5 ∧ r * s = -8/5) →
  (∃ p q : ℚ, p + q = -c ∧ p * q = d ∧ p = r - 3 ∧ q = s - 3) →
  d = 28/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3632_363244


namespace NUMINAMATH_CALUDE_sum_in_base6_l3632_363275

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  let ones := n % 6
  let sixes := n / 6
  sixes * 6 + ones

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ :=
  let sixes := n / 6
  let ones := n % 6
  sixes * 10 + ones

/-- Theorem: The sum of 5₆ and 21₆ in base 6 is equal to 30₆ --/
theorem sum_in_base6 : base10ToBase6 (base6ToBase10 5 + base6ToBase10 21) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base6_l3632_363275


namespace NUMINAMATH_CALUDE_cone_slant_height_l3632_363270

/-- The slant height of a cone with base radius 1 and lateral surface that unfolds into a semicircle -/
def slant_height : ℝ := 2

/-- The base radius of the cone -/
def base_radius : ℝ := 1

/-- Theorem: The slant height of a cone with base radius 1 and lateral surface that unfolds into a semicircle is 2 -/
theorem cone_slant_height :
  let r := base_radius
  let s := slant_height
  r = 1 ∧ 2 * π * r = π * s → s = 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3632_363270


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3632_363293

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 11 + 4 * Real.sqrt 6 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 1/a + 1/b = 1 → 
  3*a/(a-1) + 8*b/(b-1) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3632_363293


namespace NUMINAMATH_CALUDE_triangle_side_length_l3632_363264

theorem triangle_side_length (PQ PR PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 3.5) :
  ∃ QR : ℝ, QR = 9 ∧ PM^2 = (1/2) * (PQ^2 + PR^2 + QR^2) - (1/4) * QR^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3632_363264


namespace NUMINAMATH_CALUDE_students_behind_hoseok_l3632_363200

/-- Given a line of students with the following properties:
  * There are 20 students in total
  * 11 students are in front of Yoongi
  * Hoseok is directly behind Yoongi
  Prove that there are 7 students behind Hoseok -/
theorem students_behind_hoseok (total : ℕ) (front_yoongi : ℕ) (hoseok_pos : ℕ) : 
  total = 20 → front_yoongi = 11 → hoseok_pos = front_yoongi + 2 → 
  total - hoseok_pos = 7 := by sorry

end NUMINAMATH_CALUDE_students_behind_hoseok_l3632_363200


namespace NUMINAMATH_CALUDE_village_population_original_inhabitants_l3632_363240

theorem village_population (final_population : ℕ) : ℕ :=
  let initial_reduction := 0.9
  let secondary_reduction := 0.75
  let total_reduction := initial_reduction * secondary_reduction
  (final_population : ℝ) / total_reduction
    |> round
    |> Int.toNat

/-- The original number of inhabitants in a village, given the final population after two reductions -/
theorem original_inhabitants : village_population 5265 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_village_population_original_inhabitants_l3632_363240


namespace NUMINAMATH_CALUDE_cookies_per_pack_l3632_363274

/-- Given that Candy baked four trays with 24 cookies each and divided them equally into eight packs,
    prove that the number of cookies in each pack is 12. -/
theorem cookies_per_pack :
  let num_trays : ℕ := 4
  let cookies_per_tray : ℕ := 24
  let num_packs : ℕ := 8
  let total_cookies : ℕ := num_trays * cookies_per_tray
  let cookies_per_pack : ℕ := total_cookies / num_packs
  cookies_per_pack = 12 := by sorry

end NUMINAMATH_CALUDE_cookies_per_pack_l3632_363274


namespace NUMINAMATH_CALUDE_division_by_fraction_l3632_363235

theorem division_by_fraction : (10 + 6) / (1 / 4) = 64 := by
  sorry

end NUMINAMATH_CALUDE_division_by_fraction_l3632_363235


namespace NUMINAMATH_CALUDE_four_tangent_circles_l3632_363285

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other --/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Counts the number of circles with radius 5 tangent to both given circles --/
def count_tangent_circles (c1 c2 : Circle) : ℕ :=
  sorry

theorem four_tangent_circles (c1 c2 : Circle) :
  c1.radius = 2 →
  c2.radius = 2 →
  are_tangent c1 c2 →
  count_tangent_circles c1 c2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_four_tangent_circles_l3632_363285


namespace NUMINAMATH_CALUDE_combination_permutation_equality_l3632_363265

theorem combination_permutation_equality (n : ℕ) (hn : n > 0) :
  3 * (Nat.choose (n - 1) (n - 5)) = 5 * (Nat.factorial (n - 2) / Nat.factorial (n - 4)) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_combination_permutation_equality_l3632_363265


namespace NUMINAMATH_CALUDE_equation_holds_l3632_363243

theorem equation_holds (a b : ℝ) : a^2 - b^2 - (-2*b^2) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l3632_363243


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3632_363279

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3632_363279


namespace NUMINAMATH_CALUDE_division_problem_l3632_363250

theorem division_problem (dividend : ℕ) (divisor : ℝ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 13698 →
  divisor = 153.75280898876406 →
  remainder = 14 →
  quotient = 89 →
  (dividend : ℝ) = divisor * quotient + remainder := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3632_363250


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3632_363266

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 400 → initial_money = 1000 :=
by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l3632_363266


namespace NUMINAMATH_CALUDE_circle_common_chord_l3632_363289

theorem circle_common_chord (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = a^2) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + a*y - 6 = 0) ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = a^2 ∧ 
    x₂^2 + y₂^2 + a*y₂ - 6 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) → 
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_circle_common_chord_l3632_363289


namespace NUMINAMATH_CALUDE_plot_length_is_sixty_l3632_363237

/-- Given a rectangular plot with the following properties:
    1. The length is 20 meters more than the breadth.
    2. The cost of fencing the plot at 26.50 per meter is Rs. 5300.
    This theorem proves that the length of the plot is 60 meters. -/
theorem plot_length_is_sixty (breadth : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 60 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_sixty_l3632_363237


namespace NUMINAMATH_CALUDE_polygon_20_vertices_has_170_diagonals_l3632_363204

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 vertices has 170 diagonals -/
theorem polygon_20_vertices_has_170_diagonals :
  num_diagonals 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_polygon_20_vertices_has_170_diagonals_l3632_363204


namespace NUMINAMATH_CALUDE_count_students_without_A_l3632_363283

/-- The number of students who did not receive an A in any subject -/
def students_without_A (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (science_A : ℕ) 
  (math_history_A : ℕ) (history_science_A : ℕ) (science_math_A : ℕ) (all_subjects_A : ℕ) : ℕ :=
  total_students - (history_A + math_A + science_A - math_history_A - history_science_A - science_math_A + all_subjects_A)

theorem count_students_without_A :
  students_without_A 50 9 15 12 5 3 4 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_students_without_A_l3632_363283


namespace NUMINAMATH_CALUDE_min_sum_of_intercepts_equality_condition_l3632_363271

theorem min_sum_of_intercepts (a b : ℝ) : 
  a > 0 → b > 0 → (4 / a + 1 / b = 1) → a + b ≥ 9 := by
  sorry

theorem equality_condition (a b : ℝ) :
  a > 0 → b > 0 → (4 / a + 1 / b = 1) → (a + b = 9) → (a = 6 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_intercepts_equality_condition_l3632_363271


namespace NUMINAMATH_CALUDE_remainder_of_y_l3632_363295

theorem remainder_of_y (y : ℤ) 
  (h1 : (4 + y) % 8 = 3^2 % 8)
  (h2 : (6 + y) % 27 = 2^3 % 27)
  (h3 : (8 + y) % 125 = 3^3 % 125) :
  y % 30 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_y_l3632_363295


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3632_363207

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + 2 * Real.sqrt x) = 4 → x = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3632_363207


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3632_363263

-- Define a positive geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ (n : ℕ), a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_condition1 : a 1 * a 8 = 4 * a 5)
  (h_condition2 : (a 4 + 2 * a 6) / 2 = 18) :
  a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3632_363263


namespace NUMINAMATH_CALUDE_coordinates_of_point_B_l3632_363292

def point := ℝ × ℝ

theorem coordinates_of_point_B 
  (A B : point) 
  (length_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (parallel_to_x : A.2 = B.2)
  (coord_A : A = (-1, 3)) :
  B = (-6, 3) ∨ B = (4, 3) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_B_l3632_363292


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l3632_363253

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point bisects a line segment --/
def bisectsSegment (p : Point) (l : Line) : Prop :=
  ∃ (p1 p2 : Point), pointOnLine p1 l ∧ pointOnLine p2 l ∧ 
    p.x = (p1.x + p2.x) / 2 ∧ p.y = (p1.y + p2.y) / 2

/-- Check if a line lies between two other lines --/
def linesBetween (l : Line) (l1 l2 : Line) : Prop :=
  ∀ (p : Point), pointOnLine p l → 
    (l1.a * p.x + l1.b * p.y + l1.c) * (l2.a * p.x + l2.b * p.y + l2.c) ≤ 0

theorem line_satisfies_conditions : 
  let P : Point := ⟨3, 0⟩
  let L : Line := ⟨8, -1, -24⟩
  let L1 : Line := ⟨2, -1, -2⟩
  let L2 : Line := ⟨1, 1, 3⟩
  pointOnLine P L ∧ 
  bisectsSegment P L ∧
  linesBetween L L1 L2 :=
by sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l3632_363253


namespace NUMINAMATH_CALUDE_max_advancing_teams_l3632_363201

/-- The number of teams in the tournament -/
def num_teams : ℕ := 8

/-- The minimum number of points required to advance -/
def min_points_to_advance : ℕ := 15

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a draw -/
def draw_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- The maximum number of teams that can advance to the next round -/
theorem max_advancing_teams :
  ∃ (n : ℕ), n ≤ max_total_points / min_points_to_advance ∧
             n = 5 ∧
             (∀ m : ℕ, m > n → m * min_points_to_advance > max_total_points) :=
by sorry

end NUMINAMATH_CALUDE_max_advancing_teams_l3632_363201


namespace NUMINAMATH_CALUDE_new_average_after_grace_marks_l3632_363259

theorem new_average_after_grace_marks 
  (num_students : ℕ) 
  (original_average : ℚ) 
  (grace_marks : ℚ) :
  num_students = 35 →
  original_average = 37 →
  grace_marks = 3 →
  (num_students : ℚ) * original_average + num_students * grace_marks = num_students * 40 :=
by sorry

end NUMINAMATH_CALUDE_new_average_after_grace_marks_l3632_363259


namespace NUMINAMATH_CALUDE_operation_is_multiplication_l3632_363219

theorem operation_is_multiplication : 
  ((0.137 + 0.098)^2 - (0.137 - 0.098)^2) / (0.137 * 0.098) = 4 := by
  sorry

end NUMINAMATH_CALUDE_operation_is_multiplication_l3632_363219


namespace NUMINAMATH_CALUDE_log_equation_range_l3632_363220

theorem log_equation_range (a : ℝ) :
  (∃ y : ℝ, y = Real.log (5 - a) / Real.log (a - 2)) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
sorry

end NUMINAMATH_CALUDE_log_equation_range_l3632_363220


namespace NUMINAMATH_CALUDE_razorback_revenue_per_shirt_l3632_363277

/-- Razorback t-shirt shop sales data -/
structure TShirtSales where
  total_shirts : ℕ
  game_shirts : ℕ
  game_revenue : ℕ

/-- Calculate the revenue per t-shirt -/
def revenue_per_shirt (sales : TShirtSales) : ℚ :=
  sales.game_revenue / sales.game_shirts

/-- Theorem: The revenue per t-shirt is $98 -/
theorem razorback_revenue_per_shirt :
  let sales : TShirtSales := {
    total_shirts := 163,
    game_shirts := 89,
    game_revenue := 8722
  }
  revenue_per_shirt sales = 98 := by
  sorry

end NUMINAMATH_CALUDE_razorback_revenue_per_shirt_l3632_363277


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l3632_363202

theorem three_digit_number_operation (a b c : ℕ) : 
  a = c - 3 → 
  0 ≤ a ∧ a < 10 → 
  0 ≤ b ∧ b < 10 → 
  0 ≤ c ∧ c < 10 → 
  (2 * (100 * a + 10 * b + c) - (100 * c + 10 * b + a)) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l3632_363202


namespace NUMINAMATH_CALUDE_perfect_square_conversion_l3632_363209

theorem perfect_square_conversion (a b : ℝ) : 9 * a^4 * b^2 - 42 * a^2 * b = (3 * a^2 * b - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_conversion_l3632_363209


namespace NUMINAMATH_CALUDE_johns_journey_distance_l3632_363261

/-- Calculates the total distance traveled given two journey segments -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem: The total distance traveled in John's journey is 255 miles -/
theorem johns_journey_distance :
  total_distance 45 2 55 3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_distance_l3632_363261


namespace NUMINAMATH_CALUDE_expression_evaluation_l3632_363221

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3*x^3 - 5*x^2 + 8*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3632_363221


namespace NUMINAMATH_CALUDE_council_vote_change_l3632_363299

theorem council_vote_change (total : ℕ) (initial_for initial_against : ℕ) 
  (revote_for revote_against : ℕ) :
  total = 350 →
  initial_for + initial_against = total →
  initial_against > initial_for →
  revote_for + revote_against = total →
  revote_for > revote_against →
  (revote_for - revote_against) = 2 * (initial_against - initial_for) →
  revote_for = (10 * initial_against) / 9 →
  revote_for - initial_for = 66 := by
  sorry

end NUMINAMATH_CALUDE_council_vote_change_l3632_363299


namespace NUMINAMATH_CALUDE_rice_distribution_l3632_363224

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) : 
  total_weight = 29 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound / num_containers : ℚ) = 29 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l3632_363224


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3632_363288

theorem combined_mean_of_two_sets (set1_mean set2_mean : ℚ) :
  set1_mean = 18 →
  set2_mean = 16 →
  (7 * set1_mean + 8 * set2_mean) / 15 = 254 / 15 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3632_363288


namespace NUMINAMATH_CALUDE_podium_height_l3632_363218

/-- The height of the podium given two configurations of books -/
theorem podium_height (l w : ℝ) (h : ℝ) : 
  l + h - w = 40 → w + h - l = 34 → h = 37 := by sorry

end NUMINAMATH_CALUDE_podium_height_l3632_363218


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3632_363291

/-- A rectangular prism that is not a cube -/
structure RectangularPrism where
  /-- The number of faces of the rectangular prism -/
  faces : ℕ
  /-- Each face is a rectangle -/
  faces_are_rectangles : True
  /-- The number of diagonals in each rectangular face -/
  diagonals_per_face : ℕ
  /-- The number of space diagonals in the rectangular prism -/
  space_diagonals : ℕ
  /-- The rectangular prism has exactly 6 faces -/
  face_count : faces = 6
  /-- Each rectangular face has exactly 2 diagonals -/
  face_diagonal_count : diagonals_per_face = 2
  /-- The rectangular prism has exactly 4 space diagonals -/
  space_diagonal_count : space_diagonals = 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (rp : RectangularPrism) : ℕ :=
  rp.faces * rp.diagonals_per_face + rp.space_diagonals

/-- Theorem: A rectangular prism (not a cube) has 16 diagonals -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) : total_diagonals rp = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3632_363291


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l3632_363245

/-- Represents the number of students to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (stratum_size : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_size * sample_size) / total_population

/-- The problem statement -/
theorem grade_10_sample_size :
  stratified_sample_size 4500 1200 150 = 40 := by
  sorry


end NUMINAMATH_CALUDE_grade_10_sample_size_l3632_363245


namespace NUMINAMATH_CALUDE_both_chromatids_contain_N15_l3632_363286

/-- Represents a chromatid -/
structure Chromatid where
  hasN15 : Bool

/-- Represents a chromosome with two chromatids -/
structure Chromosome where
  chromatid1 : Chromatid
  chromatid2 : Chromatid

/-- Represents a cell at the tetraploid stage -/
structure TetraploidCell where
  chromosomes : List Chromosome

/-- Represents the initial condition of progenitor cells -/
def initialProgenitorCell : Bool := true

/-- Represents the culture medium containing N -/
def cultureMediumWithN : Bool := true

/-- Theorem stating that both chromatids contain N15 at the tetraploid stage -/
theorem both_chromatids_contain_N15 (cell : TetraploidCell) 
  (h1 : initialProgenitorCell = true) 
  (h2 : cultureMediumWithN = true) : 
  ∀ c ∈ cell.chromosomes, c.chromatid1.hasN15 ∧ c.chromatid2.hasN15 := by
  sorry


end NUMINAMATH_CALUDE_both_chromatids_contain_N15_l3632_363286


namespace NUMINAMATH_CALUDE_square_root_equation_implies_y_minus_x_equals_two_l3632_363268

theorem square_root_equation_implies_y_minus_x_equals_two (x y : ℝ) :
  Real.sqrt (x + 1) - Real.sqrt (-1 - x) = (x + y)^2 → y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_implies_y_minus_x_equals_two_l3632_363268


namespace NUMINAMATH_CALUDE_factor_expression_l3632_363255

theorem factor_expression (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3632_363255


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3632_363233

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3632_363233


namespace NUMINAMATH_CALUDE_independence_day_absentees_l3632_363280

theorem independence_day_absentees (total_children : ℕ) (bananas : ℕ) (present_children : ℕ) : 
  total_children = 740 →
  bananas = total_children * 2 →
  bananas = present_children * 4 →
  total_children - present_children = 370 := by
sorry

end NUMINAMATH_CALUDE_independence_day_absentees_l3632_363280


namespace NUMINAMATH_CALUDE_crab_price_proof_l3632_363260

/-- Proves that the price per crab is $3 given the conditions of John's crab selling business -/
theorem crab_price_proof (baskets_per_week : ℕ) (crabs_per_basket : ℕ) (collection_frequency : ℕ) (total_revenue : ℕ) :
  baskets_per_week = 3 →
  crabs_per_basket = 4 →
  collection_frequency = 2 →
  total_revenue = 72 →
  (total_revenue : ℚ) / (baskets_per_week * crabs_per_basket * collection_frequency) = 3 := by
  sorry

#check crab_price_proof

end NUMINAMATH_CALUDE_crab_price_proof_l3632_363260


namespace NUMINAMATH_CALUDE_candy_count_l3632_363215

theorem candy_count (initial_bags : ℕ) (initial_cookies : ℕ) (remaining_bags : ℕ) 
  (h1 : initial_bags = 14)
  (h2 : initial_cookies = 28)
  (h3 : remaining_bags = 2)
  (h4 : initial_cookies % initial_bags = 0) :
  initial_cookies - (remaining_bags * (initial_cookies / initial_bags)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3632_363215


namespace NUMINAMATH_CALUDE_price_adjustment_l3632_363231

theorem price_adjustment (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let increased_price := original_price * (1 + 30 / 100)
  let decrease_factor := 3 / 13
  increased_price * (1 - decrease_factor) = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_adjustment_l3632_363231


namespace NUMINAMATH_CALUDE_saheed_earnings_l3632_363284

theorem saheed_earnings (vika_earnings kayla_earnings saheed_earnings : ℕ) : 
  vika_earnings = 84 →
  kayla_earnings = vika_earnings - 30 →
  saheed_earnings = 4 * kayla_earnings →
  saheed_earnings = 216 := by
  sorry

end NUMINAMATH_CALUDE_saheed_earnings_l3632_363284


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l3632_363227

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, x > 0 ∧ |x + 4| = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l3632_363227


namespace NUMINAMATH_CALUDE_esteban_exercise_days_l3632_363249

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of minutes in an hour -/
def minutesInHour : ℕ := 60

/-- Represents Natasha's daily exercise time in minutes -/
def natashasDailyExercise : ℕ := 30

/-- Represents Esteban's daily exercise time in minutes -/
def estebansDailyExercise : ℕ := 10

/-- Represents the total exercise time of Natasha and Esteban in hours -/
def totalExerciseTime : ℕ := 5

/-- Theorem stating that Esteban exercised for 9 days -/
theorem esteban_exercise_days : 
  ∃ (estebanDays : ℕ), 
    estebanDays * estebansDailyExercise + 
    daysInWeek * natashasDailyExercise = 
    totalExerciseTime * minutesInHour ∧ 
    estebanDays = 9 := by
  sorry

end NUMINAMATH_CALUDE_esteban_exercise_days_l3632_363249
