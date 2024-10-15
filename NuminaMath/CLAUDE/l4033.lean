import Mathlib

namespace NUMINAMATH_CALUDE_inverse_217_mod_397_l4033_403350

theorem inverse_217_mod_397 : ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ (217 * a) % 397 = 1 :=
by
  use 161
  sorry

end NUMINAMATH_CALUDE_inverse_217_mod_397_l4033_403350


namespace NUMINAMATH_CALUDE_line_circle_intersection_distance_l4033_403378

/-- The line y = kx + 1 intersects the circle (x - 2)² + y² = 9 at two points with distance 4 apart -/
theorem line_circle_intersection_distance (k : ℝ) : ∃ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₁ = k * x₁ + 1) ∧
  (y₂ = k * x₂ + 1) ∧
  ((x₁ - 2)^2 + y₁^2 = 9) ∧
  ((x₂ - 2)^2 + y₂^2 = 9) ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) := by
  sorry

#check line_circle_intersection_distance

end NUMINAMATH_CALUDE_line_circle_intersection_distance_l4033_403378


namespace NUMINAMATH_CALUDE_flour_for_one_loaf_l4033_403353

/-- Given that 5 cups of flour are needed for 2 loaves of bread,
    prove that 2.5 cups of flour are needed for 1 loaf of bread. -/
theorem flour_for_one_loaf (total_flour : ℝ) (total_loaves : ℝ) 
  (h1 : total_flour = 5)
  (h2 : total_loaves = 2) :
  total_flour / total_loaves = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_one_loaf_l4033_403353


namespace NUMINAMATH_CALUDE_fish_tank_balls_l4033_403332

/-- The total number of balls in a fish tank with goldfish, platyfish, and angelfish -/
def total_balls (goldfish platyfish angelfish : ℕ) 
                (goldfish_balls platyfish_balls angelfish_balls : ℚ) : ℚ :=
  (goldfish : ℚ) * goldfish_balls + 
  (platyfish : ℚ) * platyfish_balls + 
  (angelfish : ℚ) * angelfish_balls

/-- Theorem stating the total number of balls in the fish tank -/
theorem fish_tank_balls : 
  total_balls 5 8 4 12.5 7.5 4.5 = 140.5 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_balls_l4033_403332


namespace NUMINAMATH_CALUDE_weekly_rental_cost_l4033_403345

/-- The weekly rental cost of a parking space, given the monthly cost,
    yearly savings, and number of months and weeks in a year. -/
theorem weekly_rental_cost (monthly_cost : ℕ) (yearly_savings : ℕ) 
                            (months_per_year : ℕ) (weeks_per_year : ℕ) :
  monthly_cost = 42 →
  yearly_savings = 16 →
  months_per_year = 12 →
  weeks_per_year = 52 →
  (monthly_cost * months_per_year + yearly_savings) / weeks_per_year = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_weekly_rental_cost_l4033_403345


namespace NUMINAMATH_CALUDE_part1_part2_l4033_403309

/-- Represents a hotel accommodation scenario for a tour group -/
structure HotelAccommodation where
  totalPeople : ℕ
  singleRooms : ℕ
  tripleRooms : ℕ
  singleRoomPrice : ℕ
  tripleRoomPrice : ℕ
  menCount : ℕ

/-- Calculates the total cost for one night -/
def totalCost (h : HotelAccommodation) : ℕ :=
  h.singleRooms * h.singleRoomPrice + h.tripleRooms * h.tripleRoomPrice

/-- Part 1: Proves that given a total cost of 1530 yuan, the number of single rooms rented is 1 -/
theorem part1 (h : HotelAccommodation) 
    (hTotal : h.totalPeople = 33)
    (hSinglePrice : h.singleRoomPrice = 100)
    (hTriplePrice : h.tripleRoomPrice = 130)
    (hCost : totalCost h = 1530)
    (hSingleAvailable : h.singleRooms ≤ 4) :
  h.singleRooms = 1 := by
  sorry

/-- Part 2: Proves that given 3 single rooms and 19 men, the minimum cost is 1600 yuan -/
theorem part2 (h : HotelAccommodation) 
    (hTotal : h.totalPeople = 33)
    (hSinglePrice : h.singleRoomPrice = 100)
    (hTriplePrice : h.tripleRoomPrice = 130)
    (hSingleRooms : h.singleRooms = 3)
    (hMenCount : h.menCount = 19) :
  ∃ (minCost : ℕ), minCost = 1600 ∧ ∀ (cost : ℕ), totalCost h ≥ minCost := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l4033_403309


namespace NUMINAMATH_CALUDE_expression_simplification_l4033_403311

theorem expression_simplification (x : ℝ) (h : x = -3) :
  (x - 3) * (x + 4) - (x - x^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4033_403311


namespace NUMINAMATH_CALUDE_right_triangle_area_l4033_403326

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 49) (h3 : c^2 = 225) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 28 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4033_403326


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_eight_l4033_403356

theorem consecutive_integers_sqrt_eight (a b : ℤ) : 
  (a < Real.sqrt 8 ∧ Real.sqrt 8 < b) → 
  (b = a + 1) → 
  (b ^ a : ℝ) = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_eight_l4033_403356


namespace NUMINAMATH_CALUDE_paco_initial_cookies_l4033_403314

/-- Proves that Paco had 40 cookies initially given the problem conditions -/
theorem paco_initial_cookies :
  ∀ x : ℕ,
  x - 2 + 37 = 75 →
  x = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_paco_initial_cookies_l4033_403314


namespace NUMINAMATH_CALUDE_water_current_speed_l4033_403397

/-- The speed of a water current given swimmer's speed and time against current -/
theorem water_current_speed (swimmer_speed : ℝ) (distance : ℝ) (time : ℝ) :
  swimmer_speed = 4 →
  distance = 5 →
  time = 2.5 →
  ∃ (current_speed : ℝ), 
    time = distance / (swimmer_speed - current_speed) ∧
    current_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_current_speed_l4033_403397


namespace NUMINAMATH_CALUDE_bird_nest_problem_l4033_403390

theorem bird_nest_problem (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) 
  (h2 : birds = nests + 3) : 
  nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_problem_l4033_403390


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l4033_403324

/-- The volume of a cone with the same radius and height as a cylinder with volume 81π cm³ is 27π cm³ -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 81 * π → (1/3) * π * r^2 * h = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l4033_403324


namespace NUMINAMATH_CALUDE_marcus_pebble_count_l4033_403335

/-- Given an initial number of pebbles, calculate the number of pebbles
    after skipping half and receiving more. -/
def final_pebble_count (initial : ℕ) (received : ℕ) : ℕ :=
  initial / 2 + received

/-- Theorem stating that given 18 initial pebbles and 30 received pebbles,
    the final count is 39. -/
theorem marcus_pebble_count :
  final_pebble_count 18 30 = 39 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebble_count_l4033_403335


namespace NUMINAMATH_CALUDE_cookies_eaten_l4033_403393

theorem cookies_eaten (original : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  original = 18 → remaining = 9 → eaten = original - remaining → eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l4033_403393


namespace NUMINAMATH_CALUDE_james_money_calculation_l4033_403372

theorem james_money_calculation (num_bills : ℕ) (bill_value : ℕ) (existing_amount : ℕ) : 
  num_bills = 3 → bill_value = 20 → existing_amount = 75 → 
  num_bills * bill_value + existing_amount = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_calculation_l4033_403372


namespace NUMINAMATH_CALUDE_karen_wall_paint_area_l4033_403360

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℝ := d.height * d.width

/-- Represents Karen's living room wall with its components -/
structure Wall where
  dimensions : Dimensions
  window : Dimensions
  door : Dimensions

/-- Calculates the area to be painted on the wall -/
def areaToPaint (w : Wall) : ℝ :=
  area w.dimensions - area w.window - area w.door

theorem karen_wall_paint_area :
  let wall : Wall := {
    dimensions := { height := 10, width := 15 },
    window := { height := 3, width := 5 },
    door := { height := 2, width := 6 }
  }
  areaToPaint wall = 123 := by sorry

end NUMINAMATH_CALUDE_karen_wall_paint_area_l4033_403360


namespace NUMINAMATH_CALUDE_function_property_implies_zero_l4033_403339

theorem function_property_implies_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (y^2) = f (x^2 + y)) : 
  f (-2017) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_zero_l4033_403339


namespace NUMINAMATH_CALUDE_new_refrigerator_cost_l4033_403303

/-- The daily cost of electricity for Kurt's old refrigerator in dollars -/
def old_cost : ℚ := 85/100

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The amount Kurt saves in a month with his new refrigerator in dollars -/
def monthly_savings : ℚ := 12

/-- The daily cost of electricity for Kurt's new refrigerator in dollars -/
def new_cost : ℚ := 45/100

theorem new_refrigerator_cost :
  (days_in_month : ℚ) * old_cost - (days_in_month : ℚ) * new_cost = monthly_savings :=
by sorry

end NUMINAMATH_CALUDE_new_refrigerator_cost_l4033_403303


namespace NUMINAMATH_CALUDE_sum_in_range_l4033_403368

theorem sum_in_range : ∃ (s : ℚ), 
  s = 3 + 1/8 + 4 + 1/3 + 6 + 1/21 ∧ 13 < s ∧ s < 14.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l4033_403368


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l4033_403366

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- Represents an isosceles triangle. -/
structure IsoscelesTriangle extends Triangle where
  isosceles : (a = b) ∨ (b = c) ∨ (c = a)

/-- 
Given an isosceles triangle with perimeter 17 and one side length 4,
prove that the other two sides must both be 6.5.
-/
theorem isosceles_triangle_side_lengths :
  ∀ t : IsoscelesTriangle,
    t.a + t.b + t.c = 17 →
    (t.a = 4 ∨ t.b = 4 ∨ t.c = 4) →
    ((t.a = 6.5 ∧ t.b = 6.5 ∧ t.c = 4) ∨
     (t.a = 6.5 ∧ t.b = 4 ∧ t.c = 6.5) ∨
     (t.a = 4 ∧ t.b = 6.5 ∧ t.c = 6.5)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l4033_403366


namespace NUMINAMATH_CALUDE_roots_on_circle_l4033_403316

theorem roots_on_circle : ∃ (r : ℝ), r = 2/3 ∧
  ∀ (z : ℂ), (z - 2)^6 = 64*z^6 → Complex.abs (z - 2/3) = r := by
  sorry

end NUMINAMATH_CALUDE_roots_on_circle_l4033_403316


namespace NUMINAMATH_CALUDE_min_buses_for_given_route_l4033_403358

/-- Represents the bus route configuration -/
structure BusRoute where
  one_way_time : ℕ
  stop_time : ℕ
  departure_interval : ℕ

/-- Calculates the minimum number of buses required for a given bus route -/
def min_buses_required (route : BusRoute) : ℕ :=
  let round_trip_time := 2 * (route.one_way_time + route.stop_time)
  (round_trip_time / route.departure_interval)

/-- Theorem stating that the minimum number of buses required for the given conditions is 20 -/
theorem min_buses_for_given_route :
  let route := BusRoute.mk 50 10 6
  min_buses_required route = 20 := by
  sorry

#eval min_buses_required (BusRoute.mk 50 10 6)

end NUMINAMATH_CALUDE_min_buses_for_given_route_l4033_403358


namespace NUMINAMATH_CALUDE_average_age_decrease_l4033_403364

theorem average_age_decrease (original_strength : ℕ) (original_avg_age : ℝ) 
  (new_students : ℕ) (new_avg_age : ℝ) : 
  original_strength = 17 →
  original_avg_age = 40 →
  new_students = 17 →
  new_avg_age = 32 →
  let new_total_strength := original_strength + new_students
  let new_avg_age := (original_strength * original_avg_age + new_students * new_avg_age) / new_total_strength
  original_avg_age - new_avg_age = 4 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l4033_403364


namespace NUMINAMATH_CALUDE_tuesday_flower_sales_l4033_403369

/-- The number of flowers sold by Ginger on Tuesday -/
def total_flowers (lilacs roses gardenias : ℕ) : ℕ :=
  lilacs + roses + gardenias

/-- Theorem stating the total number of flowers sold on Tuesday -/
theorem tuesday_flower_sales :
  ∀ (lilacs roses gardenias : ℕ),
    lilacs = 10 →
    roses = 3 * lilacs →
    gardenias = lilacs / 2 →
    total_flowers lilacs roses gardenias = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_tuesday_flower_sales_l4033_403369


namespace NUMINAMATH_CALUDE_weight_of_HClO2_l4033_403340

/-- The molar mass of HClO2 in g/mol -/
def molar_mass_HClO2 : ℝ := 68.46

/-- The number of moles of HClO2 -/
def moles_HClO2 : ℝ := 6

/-- The weight of HClO2 in grams -/
def weight_HClO2 : ℝ := molar_mass_HClO2 * moles_HClO2

theorem weight_of_HClO2 :
  weight_HClO2 = 410.76 := by sorry

end NUMINAMATH_CALUDE_weight_of_HClO2_l4033_403340


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4033_403302

/-- Given that x varies inversely with y³ and x = 8 when y = 1, prove that x = 1 when y = 2 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ y, x * y^3 = k) →  -- x varies inversely with y³
  (8 * 1^3 = k) →       -- x = 8 when y = 1
  (1 * 2^3 = k) →       -- x = 1 when y = 2
  True := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4033_403302


namespace NUMINAMATH_CALUDE_equation_solution_l4033_403313

theorem equation_solution : 
  let f (x : ℝ) := (x^2 + 3*x - 4)^2 + (2*x^2 - 7*x + 6)^2 - (3*x^2 - 4*x + 2)^2
  ∀ x : ℝ, f x = 0 ↔ x = -4 ∨ x = 1 ∨ x = 3/2 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4033_403313


namespace NUMINAMATH_CALUDE_power_calculation_l4033_403329

theorem power_calculation : (16^6 * 8^3) / 4^11 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l4033_403329


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4033_403304

/-- The line mx + y - m - 1 = 0 passes through the point (1, 1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m * 1 + 1 - m - 1 = 0) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4033_403304


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l4033_403321

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_times_111_not_five_digit_palindrome :
  ∀ n : ℕ, isThreeDigitPalindrome n →
    (n < 111 ∨ isFiveDigitPalindrome (n * 111)) →
    ¬isThreeDigitPalindrome 111 ∨ isFiveDigitPalindrome (111 * 111) :=
by
  sorry

#check smallest_three_digit_palindrome_times_111_not_five_digit_palindrome

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l4033_403321


namespace NUMINAMATH_CALUDE_danny_in_position_three_l4033_403347

-- Define the people
inductive Person : Type
| Amelia : Person
| Blake : Person
| Claire : Person
| Danny : Person

-- Define the positions
inductive Position : Type
| One : Position
| Two : Position
| Three : Position
| Four : Position

-- Define the seating arrangement
def Seating := Person → Position

-- Define opposite positions
def opposite (p : Position) : Position :=
  match p with
  | Position.One => Position.Three
  | Position.Two => Position.Four
  | Position.Three => Position.One
  | Position.Four => Position.Two

-- Define adjacent positions
def adjacent (p1 p2 : Position) : Prop :=
  (p1 = Position.One ∧ p2 = Position.Two) ∨
  (p1 = Position.Two ∧ p2 = Position.Three) ∨
  (p1 = Position.Three ∧ p2 = Position.Four) ∨
  (p1 = Position.Four ∧ p2 = Position.One) ∨
  (p2 = Position.One ∧ p1 = Position.Two) ∨
  (p2 = Position.Two ∧ p1 = Position.Three) ∨
  (p2 = Position.Three ∧ p1 = Position.Four) ∨
  (p2 = Position.Four ∧ p1 = Position.One)

-- Define between positions
def between (p1 p2 p3 : Position) : Prop :=
  (adjacent p1 p2 ∧ adjacent p2 p3) ∨
  (adjacent p3 p1 ∧ adjacent p1 p2)

-- Theorem statement
theorem danny_in_position_three 
  (s : Seating)
  (claire_in_one : s Person.Claire = Position.One)
  (not_blake_opposite_claire : s Person.Blake ≠ opposite (s Person.Claire))
  (not_amelia_between_blake_claire : ¬ between (s Person.Blake) (s Person.Amelia) (s Person.Claire)) :
  s Person.Danny = Position.Three :=
by sorry

end NUMINAMATH_CALUDE_danny_in_position_three_l4033_403347


namespace NUMINAMATH_CALUDE_triangle_inequality_example_l4033_403346

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_example : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_example_l4033_403346


namespace NUMINAMATH_CALUDE_range_of_m_l4033_403348

/-- Given a function f with derivative f', we define g and prove a property about m. -/
theorem range_of_m (f : ℝ → ℝ) (f' : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, HasDerivAt f (f' x) x) →  -- f has derivative f' for all x
  (∀ x, g x = f x - (1/2) * x^2) →  -- definition of g
  (∀ x, f' x < x) →  -- condition on f'
  (f (4 - m) - f m ≥ 8 - 4*m) →  -- given inequality
  m ≥ 2 :=  -- conclusion: m is in [2, +∞)
sorry

end NUMINAMATH_CALUDE_range_of_m_l4033_403348


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l4033_403342

/-- 
Given an arithmetic progression where the sum of the 4th and 12th terms is 8,
prove that the sum of the first 15 terms is 60.
-/
theorem arithmetic_progression_sum (a d : ℝ) : 
  (a + 3*d) + (a + 11*d) = 8 → 
  (15 : ℝ) / 2 * (2*a + 14*d) = 60 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l4033_403342


namespace NUMINAMATH_CALUDE_quadrilateral_S_l4033_403363

/-- Given a quadrilateral with sides a, b, c, d and an angle A (where A ≠ 90°),
    S is equal to (a^2 + d^2 - b^2 - c^2) / (4 * tan(A)) -/
theorem quadrilateral_S (a b c d : ℝ) (A : ℝ) (h : A ≠ π / 2) :
  let S := (a^2 + d^2 - b^2 - c^2) / (4 * Real.tan A)
  ∃ (S : ℝ), S = (a^2 + d^2 - b^2 - c^2) / (4 * Real.tan A) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_S_l4033_403363


namespace NUMINAMATH_CALUDE_only_one_solves_l4033_403379

def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3
def prob_C : ℚ := 1/4

theorem only_one_solves : 
  let prob_only_one := 
    (prob_A * (1 - prob_B) * (1 - prob_C)) + 
    ((1 - prob_A) * prob_B * (1 - prob_C)) + 
    ((1 - prob_A) * (1 - prob_B) * prob_C)
  prob_only_one = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_only_one_solves_l4033_403379


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4033_403382

theorem complex_equation_solution (z : ℂ) : (1 - 2*I)*z = Complex.abs (3 + 4*I) → z = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4033_403382


namespace NUMINAMATH_CALUDE_symmetric_angles_sum_l4033_403396

theorem symmetric_angles_sum (α β : Real) : 
  0 < α ∧ α < 2 * Real.pi ∧ 
  0 < β ∧ β < 2 * Real.pi ∧ 
  α = 2 * Real.pi - β → 
  α + β = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_symmetric_angles_sum_l4033_403396


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l4033_403365

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500)
  (h2 : music_students = 40)
  (h3 : art_students = 20)
  (h4 : both_students = 10)
  : total_students - (music_students + art_students - both_students) = 450 :=
by
  sorry

#check students_taking_neither_music_nor_art

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l4033_403365


namespace NUMINAMATH_CALUDE_inequality_preservation_l4033_403367

theorem inequality_preservation (a b : ℝ) (h : a < b) : 2 - a > 2 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l4033_403367


namespace NUMINAMATH_CALUDE_volleyball_league_female_fraction_l4033_403371

theorem volleyball_league_female_fraction 
  (last_year_male : ℕ)
  (total_increase : ℝ)
  (male_increase : ℝ)
  (female_increase : ℝ)
  (h1 : last_year_male = 30)
  (h2 : total_increase = 0.15)
  (h3 : male_increase = 0.10)
  (h4 : female_increase = 0.25) :
  let this_year_male : ℝ := last_year_male * (1 + male_increase)
  let last_year_female : ℝ := last_year_male * (1 + total_increase) / (2 + male_increase + female_increase) - last_year_male
  let this_year_female : ℝ := last_year_female * (1 + female_increase)
  let total_this_year : ℝ := this_year_male + this_year_female
  (this_year_female / total_this_year) = 25 / 47 := by
sorry

end NUMINAMATH_CALUDE_volleyball_league_female_fraction_l4033_403371


namespace NUMINAMATH_CALUDE_dividend_calculation_l4033_403307

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 10) : 
  divisor * quotient + remainder = 163 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4033_403307


namespace NUMINAMATH_CALUDE_expression_value_at_three_l4033_403315

theorem expression_value_at_three :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64) / (x^4 + 8) = 89 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l4033_403315


namespace NUMINAMATH_CALUDE_valid_purchase_plans_l4033_403333

/-- Represents a purchasing plan for basketballs and footballs -/
structure PurchasePlan where
  basketballs : ℕ
  footballs : ℕ

/-- Checks if a purchase plan is valid according to the given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.basketballs + p.footballs = 20 ∧
  p.basketballs > p.footballs ∧
  80 * p.basketballs + 50 * p.footballs ≤ 1400

theorem valid_purchase_plans :
  ∀ (p : PurchasePlan), isValidPlan p ↔ 
    (p = ⟨11, 9⟩ ∨ p = ⟨12, 8⟩ ∨ p = ⟨13, 7⟩) :=
by sorry

end NUMINAMATH_CALUDE_valid_purchase_plans_l4033_403333


namespace NUMINAMATH_CALUDE_park_area_l4033_403327

/-- Proves that a rectangular park with sides in ratio 3:2 and fencing cost of $225 at 90 ps per meter has an area of 3750 square meters -/
theorem park_area (length width perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 3 / 2 →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.9 →
  total_cost = 225 →
  total_cost = perimeter * cost_per_meter →
  length * width = 3750 :=
by sorry

end NUMINAMATH_CALUDE_park_area_l4033_403327


namespace NUMINAMATH_CALUDE_zoo_badge_problem_l4033_403391

/-- Represents the commemorative badges sold by the zoo -/
inductive Badge
| A
| B

/-- Represents the cost and selling prices of badges -/
structure BadgePrices where
  cost_A : ℝ
  cost_B : ℝ
  sell_A : ℝ
  sell_B : ℝ

/-- Represents the purchasing plan for badges -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

/-- Calculates the total cost of a purchasing plan -/
def total_cost (prices : BadgePrices) (plan : PurchasePlan) : ℝ :=
  prices.cost_A * plan.num_A + prices.cost_B * plan.num_B

/-- Calculates the total profit of a purchasing plan -/
def total_profit (prices : BadgePrices) (plan : PurchasePlan) : ℝ :=
  (prices.sell_A - prices.cost_A) * plan.num_A + (prices.sell_B - prices.cost_B) * plan.num_B

/-- Theorem representing the zoo's badge problem -/
theorem zoo_badge_problem (prices : BadgePrices) 
  (h1 : prices.cost_A = prices.cost_B + 4)
  (h2 : 6 * prices.cost_A = 10 * prices.cost_B)
  (h3 : prices.sell_A = 13)
  (h4 : prices.sell_B = 8)
  : 
  prices.cost_A = 10 ∧ 
  prices.cost_B = 6 ∧
  ∃ (optimal_plan : PurchasePlan),
    optimal_plan.num_A + optimal_plan.num_B = 400 ∧
    total_cost prices optimal_plan ≤ 2800 ∧
    total_profit prices optimal_plan = 900 ∧
    ∀ (plan : PurchasePlan),
      plan.num_A + plan.num_B = 400 →
      total_cost prices plan ≤ 2800 →
      total_profit prices plan ≤ total_profit prices optimal_plan :=
by sorry


end NUMINAMATH_CALUDE_zoo_badge_problem_l4033_403391


namespace NUMINAMATH_CALUDE_expression_evaluation_l4033_403336

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The product of powers of x from 1 to n -/
def prod_powers (x : ℕ) (n : ℕ) : ℕ := x ^ sum_first_n n

/-- The product of powers of x for multiples of 3 up to 3n -/
def prod_powers_mult3 (x : ℕ) (n : ℕ) : ℕ := x ^ (3 * sum_first_n n)

theorem expression_evaluation (x : ℕ) (hx : x = 3) :
  prod_powers x 20 / prod_powers_mult3 x 10 = x ^ 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4033_403336


namespace NUMINAMATH_CALUDE_power_of_power_l4033_403386

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4033_403386


namespace NUMINAMATH_CALUDE_compare_fractions_l4033_403385

theorem compare_fractions :
  (-7/2 : ℚ) < (-7/3 : ℚ) ∧ (-3/4 : ℚ) > (-4/5 : ℚ) := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l4033_403385


namespace NUMINAMATH_CALUDE_two_colors_probability_l4033_403370

/-- The number of black balls in the bin -/
def black_balls : ℕ := 10

/-- The number of white balls in the bin -/
def white_balls : ℕ := 8

/-- The number of red balls in the bin -/
def red_balls : ℕ := 6

/-- The total number of balls in the bin -/
def total_balls : ℕ := black_balls + white_balls + red_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 4

/-- The probability of drawing 2 balls of one color and 2 balls of another color -/
theorem two_colors_probability : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 2 + 
   Nat.choose black_balls 2 * Nat.choose red_balls 2 + 
   Nat.choose white_balls 2 * Nat.choose red_balls 2) / 
  Nat.choose total_balls drawn_balls = 157 / 845 := by sorry

end NUMINAMATH_CALUDE_two_colors_probability_l4033_403370


namespace NUMINAMATH_CALUDE_hash_example_l4033_403389

def hash (a b c d : ℝ) : ℝ := d * b^2 - 5 * a * c

theorem hash_example : hash 2 3 1 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_hash_example_l4033_403389


namespace NUMINAMATH_CALUDE_construct_square_and_dodecagon_l4033_403374

/-- A point in a 2D plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a 2D plane --/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a compass --/
structure Compass :=
  (create_circle : Point → ℝ → Circle)

/-- Represents a square --/
structure Square :=
  (vertices : Fin 4 → Point)

/-- Represents a regular dodecagon --/
structure RegularDodecagon :=
  (vertices : Fin 12 → Point)

/-- Theorem stating that a square and a regular dodecagon can be constructed using only a compass --/
theorem construct_square_and_dodecagon 
  (A B : Point) 
  (compass : Compass) : 
  ∃ (square : Square) (dodecagon : RegularDodecagon),
    (square.vertices 0 = A ∧ square.vertices 1 = B) ∧
    (dodecagon.vertices 0 = A ∧ dodecagon.vertices 1 = B) :=
sorry

end NUMINAMATH_CALUDE_construct_square_and_dodecagon_l4033_403374


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4033_403338

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -63 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4033_403338


namespace NUMINAMATH_CALUDE_student_selection_properties_l4033_403384

/-- Represents the selection of students from different grades -/
structure StudentSelection where
  total : Nat
  first_year : Nat
  second_year : Nat
  third_year : Nat
  selected : Nat

/-- Calculate the probability of selecting students from different grades -/
def prob_different_grades (s : StudentSelection) : Rat :=
  (s.first_year.choose 1 * s.second_year.choose 1 * s.third_year.choose 1) /
  (s.total.choose s.selected)

/-- Calculate the mathematical expectation of the number of first-year students selected -/
def expectation_first_year (s : StudentSelection) : Rat :=
  (0 * (s.total - s.first_year).choose s.selected +
   1 * (s.first_year.choose 1 * (s.total - s.first_year).choose (s.selected - 1)) +
   2 * (s.first_year.choose 2 * (s.total - s.first_year).choose (s.selected - 2))) /
  (s.total.choose s.selected)

/-- The main theorem stating the properties of the student selection problem -/
theorem student_selection_properties (s : StudentSelection) 
  (h1 : s.total = 5)
  (h2 : s.first_year = 2)
  (h3 : s.second_year = 2)
  (h4 : s.third_year = 1)
  (h5 : s.selected = 3) :
  prob_different_grades s = 2/5 ∧ expectation_first_year s = 6/5 := by
  sorry

#eval prob_different_grades ⟨5, 2, 2, 1, 3⟩
#eval expectation_first_year ⟨5, 2, 2, 1, 3⟩

end NUMINAMATH_CALUDE_student_selection_properties_l4033_403384


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l4033_403362

theorem polynomial_coefficient_equality : 
  ∃! (a b c : ℝ), ∀ (x : ℝ), 
    2*x^4 + x^3 - 41*x^2 + 83*x - 45 = (a*x^2 + b*x + c)*(x^2 + 4*x + 9) ∧ 
    a = 2 ∧ b = -7 ∧ c = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l4033_403362


namespace NUMINAMATH_CALUDE_third_shape_symmetric_l4033_403337

-- Define a type for F-like shapes
inductive FLikeShape
| first
| second
| third
| fourth
| fifth

-- Define a function to check if a shape has reflection symmetry
def has_reflection_symmetry (shape : FLikeShape) : Prop :=
  match shape with
  | FLikeShape.third => True
  | _ => False

-- Theorem statement
theorem third_shape_symmetric :
  ∃ (shape : FLikeShape), has_reflection_symmetry shape ∧ shape = FLikeShape.third :=
by
  sorry

#check third_shape_symmetric

end NUMINAMATH_CALUDE_third_shape_symmetric_l4033_403337


namespace NUMINAMATH_CALUDE_max_area_difference_rectangles_l4033_403357

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The maximum difference between areas of two rectangles with perimeter 144 is 1225 -/
theorem max_area_difference_rectangles :
  ∃ (r1 r2 : Rectangle),
    perimeter r1 = 144 ∧
    perimeter r2 = 144 ∧
    ∀ (r3 r4 : Rectangle),
      perimeter r3 = 144 →
      perimeter r4 = 144 →
      area r1 - area r2 ≥ area r3 - area r4 ∧
      area r1 - area r2 = 1225 :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_rectangles_l4033_403357


namespace NUMINAMATH_CALUDE_sin_240_degrees_l4033_403399

theorem sin_240_degrees : Real.sin (240 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l4033_403399


namespace NUMINAMATH_CALUDE_initial_oak_trees_l4033_403395

theorem initial_oak_trees (final_trees : ℕ) (cut_trees : ℕ) : final_trees = 7 → cut_trees = 2 → final_trees + cut_trees = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_oak_trees_l4033_403395


namespace NUMINAMATH_CALUDE_seeds_planted_wednesday_l4033_403334

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := sorry

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := 2

/-- The total number of seeds planted -/
def total_seeds : ℕ := 22

/-- Theorem stating that the number of seeds planted on Wednesday is 20 -/
theorem seeds_planted_wednesday :
  seeds_wednesday = total_seeds - seeds_thursday ∧ seeds_wednesday = 20 := by sorry

end NUMINAMATH_CALUDE_seeds_planted_wednesday_l4033_403334


namespace NUMINAMATH_CALUDE_lcm_problem_l4033_403341

theorem lcm_problem (d n : ℕ) : 
  d > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm d n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ d) →
  n = 230 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l4033_403341


namespace NUMINAMATH_CALUDE_sin_pi_over_six_l4033_403355

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_pi_over_six_l4033_403355


namespace NUMINAMATH_CALUDE_dividend_divisor_properties_l4033_403361

theorem dividend_divisor_properties : ∃ (dividend : Nat) (divisor : Nat),
  dividend = 957 ∧ divisor = 75 ∧
  (dividend / divisor = (divisor / 10 + divisor % 10)) ∧
  (dividend % divisor = 57) ∧
  ((dividend % divisor) * (dividend / divisor) + divisor = 759) := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_properties_l4033_403361


namespace NUMINAMATH_CALUDE_boys_in_line_l4033_403323

/-- If a boy in a single line is 19th from both ends, then the total number of boys is 37 -/
theorem boys_in_line (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ k ≤ n ∧ k = 19 ∧ n - k + 1 = 19) → n = 37 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_line_l4033_403323


namespace NUMINAMATH_CALUDE_no_negative_roots_l4033_403322

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l4033_403322


namespace NUMINAMATH_CALUDE_additional_daily_intake_l4033_403398

/-- Proves that given a total milk consumption goal and a time frame, 
    the additional daily intake required can be calculated. -/
theorem additional_daily_intake 
  (total_milk : ℝ) 
  (weeks : ℝ) 
  (current_daily : ℝ) 
  (h1 : total_milk = 105) 
  (h2 : weeks = 3) 
  (h3 : current_daily = 3) : 
  ∃ (additional : ℝ), 
    additional = (total_milk / (weeks * 7)) - current_daily ∧ 
    additional = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_daily_intake_l4033_403398


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l4033_403352

theorem quadratic_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioo 2 3, Monotone (fun x => x^2 - 2*a*x + 1) ∨ StrictMono (fun x => x^2 - 2*a*x + 1)) ↔
  (a ≤ 2 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l4033_403352


namespace NUMINAMATH_CALUDE_expression_evaluation_l4033_403330

theorem expression_evaluation : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4033_403330


namespace NUMINAMATH_CALUDE_unique_perfect_cube_l4033_403375

theorem unique_perfect_cube (Z K : ℤ) : 
  (1000 < Z) → (Z < 1500) → (K > 1) → (Z = K^3) → 
  (∃! k : ℤ, k > 1 ∧ 1000 < k^3 ∧ k^3 < 1500 ∧ Z = k^3) ∧ (K = 11) := by
sorry

end NUMINAMATH_CALUDE_unique_perfect_cube_l4033_403375


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l4033_403310

/-- Returns the last four digits of a number in base 10 -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- Checks if a number satisfies the conditions of the problem -/
def satisfiesConditions (n : ℕ) : Prop :=
  (n > 0) ∧ (lastFourDigits n = lastFourDigits (n^2)) ∧ ((n - 2) % 7 = 0)

theorem smallest_satisfying_number :
  satisfiesConditions 625 ∧ ∀ n < 625, ¬(satisfiesConditions n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l4033_403310


namespace NUMINAMATH_CALUDE_train_passengers_l4033_403325

theorem train_passengers (adults_first : ℕ) (children_first : ℕ) 
  (adults_second : ℕ) (children_second : ℕ) (got_off : ℕ) (total : ℕ) : 
  children_first = adults_first - 17 →
  adults_second = 57 →
  children_second = 18 →
  got_off = 44 →
  total = 502 →
  adults_first + children_first + adults_second + children_second - got_off = total →
  adults_first = 244 := by
sorry

end NUMINAMATH_CALUDE_train_passengers_l4033_403325


namespace NUMINAMATH_CALUDE_process_output_for_4_l4033_403380

/-- A function representing the process described in the flowchart --/
def process (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the process outputs 3 when given input 4 --/
theorem process_output_for_4 : process 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_process_output_for_4_l4033_403380


namespace NUMINAMATH_CALUDE_box_two_three_l4033_403306

/-- Define the box operation -/
def box (a b : ℝ) : ℝ := a * (b^2 + 3) - b + 1

/-- Theorem: The value of (2) □ (3) is 22 -/
theorem box_two_three : box 2 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_box_two_three_l4033_403306


namespace NUMINAMATH_CALUDE_candied_apples_count_l4033_403343

/-- The number of candied apples that were made -/
def num_apples : ℕ := 15

/-- The price of each candied apple in dollars -/
def apple_price : ℚ := 2

/-- The number of candied grapes -/
def num_grapes : ℕ := 12

/-- The price of each candied grape in dollars -/
def grape_price : ℚ := 3/2

/-- The total earnings from selling all items in dollars -/
def total_earnings : ℚ := 48

theorem candied_apples_count :
  num_apples * apple_price + num_grapes * grape_price = total_earnings :=
sorry

end NUMINAMATH_CALUDE_candied_apples_count_l4033_403343


namespace NUMINAMATH_CALUDE_min_distance_to_2i_l4033_403312

theorem min_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 - 1) = Complex.abs (z * (z - Complex.I))) :
  ∃ (w : ℂ), Complex.abs (w - 2 * Complex.I) = 1 ∧ 
  ∀ (z : ℂ), Complex.abs (z - 2 * Complex.I) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_2i_l4033_403312


namespace NUMINAMATH_CALUDE_min_abs_phi_l4033_403377

/-- Given a function f(x) = 2sin(ωx + φ) with ω > 0, prove that the minimum value of |φ| is π/2 
    under the following conditions:
    1. Three consecutive intersection points with y = b (0 < b < 2) are at x = π/6, 5π/6, 7π/6
    2. f(x) reaches its minimum value at x = 3π/2 -/
theorem min_abs_phi (ω : ℝ) (φ : ℝ) (b : ℝ) (h_ω : ω > 0) (h_b : 0 < b ∧ b < 2) : 
  (∃ (k : ℤ), φ = 2 * π * k - 3 * π / 2) →
  (∀ (x : ℝ), 2 * Real.sin (ω * x + φ) = b → 
    (x = π / 6 ∨ x = 5 * π / 6 ∨ x = 7 * π / 6)) →
  (∀ (x : ℝ), 2 * Real.sin (ω * 3 * π / 2 + φ) ≤ 2 * Real.sin (ω * x + φ)) →
  ω = 2 ∧ (∀ (ψ : ℝ), |ψ| ≥ π / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_abs_phi_l4033_403377


namespace NUMINAMATH_CALUDE_weight_fluctuation_l4033_403351

theorem weight_fluctuation (initial_weight : ℕ) (initial_loss : ℕ) (final_gain : ℕ) : 
  initial_weight = 99 →
  initial_loss = 12 →
  final_gain = 6 →
  initial_weight - initial_loss + 2 * initial_loss - 3 * initial_loss + final_gain = 81 := by
  sorry

end NUMINAMATH_CALUDE_weight_fluctuation_l4033_403351


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l4033_403394

theorem smallest_positive_integer_ending_in_3_divisible_by_11 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 11 = 0 → m ≥ n :=
by
  use 33
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l4033_403394


namespace NUMINAMATH_CALUDE_exists_prime_not_cube_root_l4033_403387

theorem exists_prime_not_cube_root (p q : ℕ) : 
  ∃ q : ℕ, Prime q ∧ ∀ p : ℕ, Prime p → ¬∃ n : ℕ, n^3 = p^2 + q :=
sorry

end NUMINAMATH_CALUDE_exists_prime_not_cube_root_l4033_403387


namespace NUMINAMATH_CALUDE_anca_rest_time_l4033_403319

-- Define the constants
def bruce_speed : ℝ := 50
def anca_speed : ℝ := 60
def total_distance : ℝ := 200

-- Define the theorem
theorem anca_rest_time :
  let bruce_time := total_distance / bruce_speed
  let anca_drive_time := total_distance / anca_speed
  let rest_time := bruce_time - anca_drive_time
  rest_time * 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_anca_rest_time_l4033_403319


namespace NUMINAMATH_CALUDE_power_equation_l4033_403320

theorem power_equation (k m n : ℕ) 
  (h1 : 3^(k - 1) = 81) 
  (h2 : 4^(m + 2) = 256) 
  (h3 : 5^(n - 3) = 625) : 
  2^(4*k - 3*m + 5*n) = 2^49 := by
sorry

end NUMINAMATH_CALUDE_power_equation_l4033_403320


namespace NUMINAMATH_CALUDE_gcd_13n_plus_4_8n_plus_3_max_9_l4033_403301

theorem gcd_13n_plus_4_8n_plus_3_max_9 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 9) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_gcd_13n_plus_4_8n_plus_3_max_9_l4033_403301


namespace NUMINAMATH_CALUDE_meal_sales_tax_percentage_l4033_403354

/-- The maximum total spending allowed for the meal -/
def total_limit : ℝ := 50

/-- The maximum cost of food allowed -/
def max_food_cost : ℝ := 40.98

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.15

/-- The maximum sales tax percentage that satisfies the conditions -/
def max_sales_tax_percentage : ℝ := 6.1

/-- Theorem stating that the maximum sales tax percentage is approximately 6.1% -/
theorem meal_sales_tax_percentage :
  ∀ (sales_tax_percentage : ℝ),
    sales_tax_percentage ≤ max_sales_tax_percentage →
    max_food_cost + (sales_tax_percentage / 100 * max_food_cost) +
    (tip_percentage * (max_food_cost + (sales_tax_percentage / 100 * max_food_cost))) ≤ total_limit :=
by sorry

end NUMINAMATH_CALUDE_meal_sales_tax_percentage_l4033_403354


namespace NUMINAMATH_CALUDE_price_quantity_change_l4033_403328

theorem price_quantity_change (original_price original_quantity : ℝ) :
  let price_increase_factor := 1.20
  let quantity_decrease_factor := 0.70
  let new_cost := original_price * price_increase_factor * original_quantity * quantity_decrease_factor
  let original_cost := original_price * original_quantity
  new_cost / original_cost = 0.84 :=
by sorry

end NUMINAMATH_CALUDE_price_quantity_change_l4033_403328


namespace NUMINAMATH_CALUDE_trajectory_curve_intersection_range_l4033_403308

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the moving point M and its projection N on AB
def M : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, y)
def N : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, 0)

-- Define vectors
def vec_MN (m : ℝ × ℝ) : ℝ × ℝ := (0, -(m.2))
def vec_AN (n : ℝ × ℝ) : ℝ × ℝ := (n.1 + 1, 0)
def vec_BN (n : ℝ × ℝ) : ℝ × ℝ := (n.1 - 1, 0)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition for point M
def condition (m : ℝ × ℝ) : Prop :=
  let n := N m
  (vec_MN m).1^2 + (vec_MN m).2^2 = dot_product (vec_AN n) (vec_BN n)

-- Define the trajectory curve E
def curve_E (p : ℝ × ℝ) : Prop := p.1^2 - p.2^2 = 1

-- Define the line l
def line_l (k : ℝ) (p : ℝ × ℝ) : Prop := p.2 = k * p.1 - 1

-- Theorem statements
theorem trajectory_curve : ∀ m : ℝ × ℝ, condition m ↔ curve_E m := by sorry

theorem intersection_range : ∀ k : ℝ,
  (∃ p : ℝ × ℝ, curve_E p ∧ line_l k p) ↔ -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_trajectory_curve_intersection_range_l4033_403308


namespace NUMINAMATH_CALUDE_tangent_at_P_tangent_through_P_l4033_403373

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the tangent line at P
theorem tangent_at_P :
  let (x₀, y₀) := P
  let slope := (3 * x₀^2 - 3 : ℝ)
  (∀ x, f x₀ + slope * (x - x₀) = -2) :=
sorry

-- Theorem for the tangent lines passing through P
theorem tangent_through_P :
  let (x₀, y₀) := P
  (∃ x₁ : ℝ, 
    (∀ x, f x₁ + (3 * x₁^2 - 3) * (x - x₁) = -2) ∨
    (∀ x, f x₁ + (3 * x₁^2 - 3) * (x - x₁) = -9/4*x + 1/4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_at_P_tangent_through_P_l4033_403373


namespace NUMINAMATH_CALUDE_original_number_l4033_403376

theorem original_number (x : ℝ) : x * 1.1 = 660 ↔ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l4033_403376


namespace NUMINAMATH_CALUDE_hex_F2E1_equals_62177_l4033_403331

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_to_decimal (s : String) : Nat :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal number F2E1 -/
def hex_number : String := "F2E1"

/-- Theorem stating that F2E1 in hexadecimal is equal to 62177 in decimal -/
theorem hex_F2E1_equals_62177 : hex_to_decimal hex_number = 62177 := by
  sorry

end NUMINAMATH_CALUDE_hex_F2E1_equals_62177_l4033_403331


namespace NUMINAMATH_CALUDE_binomial_inequality_l4033_403359

theorem binomial_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : x ≠ 0) (h3 : n ≥ 2) :
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l4033_403359


namespace NUMINAMATH_CALUDE_negative_expression_l4033_403383

theorem negative_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  b + 3 * b^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_expression_l4033_403383


namespace NUMINAMATH_CALUDE_remainder_theorem_l4033_403305

theorem remainder_theorem : 2^9 * 3^10 + 14 ≡ 2 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4033_403305


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l4033_403318

/-- Given a wire cut into two pieces of lengths x and y, where x forms a square and y forms a regular pentagon with equal perimeters, prove that x/y = 1 -/
theorem wire_cut_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  (4 * (x / 4) = 5 * (y / 5)) → x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l4033_403318


namespace NUMINAMATH_CALUDE_smallest_cube_ending_544_l4033_403344

theorem smallest_cube_ending_544 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 544 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_544_l4033_403344


namespace NUMINAMATH_CALUDE_floor_negative_seven_halves_l4033_403317

theorem floor_negative_seven_halves : 
  ⌊(-7 : ℚ) / 2⌋ = -4 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_halves_l4033_403317


namespace NUMINAMATH_CALUDE_remaining_pages_l4033_403349

/-- Calculates the remaining pages in a pad after various projects --/
theorem remaining_pages (initial_pages : ℕ) : 
  initial_pages = 120 → 
  (initial_pages / 2 - 
   (initial_pages / 4 + 10 + initial_pages * 15 / 100) / 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pages_l4033_403349


namespace NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l4033_403300

/-- Represents the decimal expansion 7.836836836... -/
def recurring_decimal : ℚ := 7 + 836 / 999

/-- The fraction representation of the recurring decimal -/
def fraction : ℚ := 7829 / 999

theorem recurring_decimal_equals_fraction :
  recurring_decimal = fraction := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l4033_403300


namespace NUMINAMATH_CALUDE_work_completion_time_l4033_403392

-- Define work rates as fractions of work completed per hour
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_BC : ℚ := 1 / 3

-- Define the time taken for A and C together
def time_AC : ℚ := 2

theorem work_completion_time :
  let work_rate_C : ℚ := work_rate_BC - work_rate_B
  let work_rate_AC : ℚ := work_rate_A + work_rate_C
  time_AC = 1 / work_rate_AC :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4033_403392


namespace NUMINAMATH_CALUDE_jerry_books_count_l4033_403388

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 4

/-- The number of shelves Jerry needs -/
def shelves_needed : ℕ := 3

/-- The total number of books Jerry has to put away -/
def total_books : ℕ := books_per_shelf * shelves_needed

theorem jerry_books_count : total_books = 12 := by
  sorry

end NUMINAMATH_CALUDE_jerry_books_count_l4033_403388


namespace NUMINAMATH_CALUDE_vector_equality_implies_coordinates_l4033_403381

/-- Given four points A, B, C, D in a plane, where vector AB equals vector CD,
    prove that the coordinates of C and D satisfy specific values. -/
theorem vector_equality_implies_coordinates (A B C D : ℝ × ℝ) :
  A = (1, 2) →
  B = (5, 4) →
  C.2 = 3 →
  D.1 = -3 →
  B.1 - A.1 = D.1 - C.1 →
  B.2 - A.2 = D.2 - C.2 →
  C.1 = -7 ∧ D.2 = 5 := by
sorry

end NUMINAMATH_CALUDE_vector_equality_implies_coordinates_l4033_403381
