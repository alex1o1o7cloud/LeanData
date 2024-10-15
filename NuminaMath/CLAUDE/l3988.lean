import Mathlib

namespace NUMINAMATH_CALUDE_six_circle_arrangement_possible_l3988_398885

/-- A configuration of 6 circles in a plane -/
structure CircleConfiguration where
  positions : Fin 6 → ℝ × ℝ

/-- Predicate to check if a configuration allows a 7th circle to touch all 6 -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ i : Fin 6, 
    let (x, y) := config.positions i
    let (cx, cy) := center
    (x - cx)^2 + (y - cy)^2 = 4  -- Assuming unit radius for simplicity

/-- Predicate to check if a configuration can be achieved without measurements or lifting -/
def AchievableWithoutMeasurement (config : CircleConfiguration) : Prop :=
  sorry  -- This would require a formal definition of "without measurement"

theorem six_circle_arrangement_possible :
  ∃ (config : CircleConfiguration), 
    ValidConfiguration config ∧ AchievableWithoutMeasurement config :=
sorry

end NUMINAMATH_CALUDE_six_circle_arrangement_possible_l3988_398885


namespace NUMINAMATH_CALUDE_shadow_point_theorem_l3988_398869

-- Define shadow point
def isShadowPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y > x, f y > f x

-- State the theorem
theorem shadow_point_theorem (f : ℝ → ℝ) (a b : ℝ) 
  (hf : Continuous f) 
  (hab : a < b)
  (h_shadow : ∀ x ∈ Set.Ioo a b, isShadowPoint f x)
  (ha_not_shadow : ¬ isShadowPoint f a)
  (hb_not_shadow : ¬ isShadowPoint f b) :
  (∀ x ∈ Set.Ioo a b, f x ≤ f b) ∧ f a = f b :=
sorry

end NUMINAMATH_CALUDE_shadow_point_theorem_l3988_398869


namespace NUMINAMATH_CALUDE_min_value_theorem_l3988_398802

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + y ≥ 1 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    3 * x₀ + y₀ = 1 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3988_398802


namespace NUMINAMATH_CALUDE_park_outer_diameter_l3988_398831

/-- Represents the dimensions of a circular park with concentric areas. -/
structure ParkDimensions where
  pond_diameter : ℝ
  seating_width : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park. -/
def outer_diameter (park : ParkDimensions) : ℝ :=
  park.pond_diameter + 2 * (park.seating_width + park.garden_width + park.path_width)

/-- Theorem stating that for a park with given dimensions, the outer diameter is 64 feet. -/
theorem park_outer_diameter :
  let park := ParkDimensions.mk 20 4 10 8
  outer_diameter park = 64 := by
  sorry


end NUMINAMATH_CALUDE_park_outer_diameter_l3988_398831


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l3988_398812

theorem bakery_flour_usage (wheat_flour : Real) (total_flour : Real) (white_flour : Real) :
  wheat_flour = 0.2 →
  total_flour = 0.3 →
  white_flour = total_flour - wheat_flour →
  white_flour = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l3988_398812


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3988_398891

/-- The number of distinct convex quadrilaterals that can be formed by selecting 4 vertices
    from 12 distinct points on the circumference of a circle. -/
def num_quadrilaterals : ℕ := 495

/-- The number of ways to choose 4 items from a set of 12 items. -/
def choose_4_from_12 : ℕ := Nat.choose 12 4

theorem quadrilaterals_on_circle :
  num_quadrilaterals = choose_4_from_12 :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3988_398891


namespace NUMINAMATH_CALUDE_triangle_base_length_l3988_398856

theorem triangle_base_length (area height : ℝ) (h1 : area = 16) (h2 : height = 4) :
  (2 * area) / height = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3988_398856


namespace NUMINAMATH_CALUDE_tangent_coincidence_implies_a_range_l3988_398860

/-- Piecewise function f(x) defined as x^2 + x + a for x < 0, and -1/x for x > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x + a else -1/x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1 else 1/x^2

theorem tangent_coincidence_implies_a_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ 
   f_derivative a x₁ = f_derivative a x₂ ∧
   f a x₁ - (f_derivative a x₁ * x₁) = f a x₂ - (f_derivative a x₂ * x₂)) →
  -2 < a ∧ a < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_coincidence_implies_a_range_l3988_398860


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l3988_398859

/-- Given an ellipse and two symmetric points on it, prove the range of m -/
theorem ellipse_symmetric_points_range (x₁ y₁ x₂ y₂ m : ℝ) : 
  (x₁^2 / 4 + y₁^2 / 3 = 1) →  -- Point A on ellipse
  (x₂^2 / 4 + y₂^2 / 3 = 1) →  -- Point B on ellipse
  ((y₁ + y₂) / 2 = 4 * ((x₁ + x₂) / 2) + m) →  -- A and B symmetric about y = 4x + m
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →  -- A and B are distinct
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l3988_398859


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3988_398873

theorem quadratic_roots_problem (x₁ x₂ k : ℝ) : 
  (x₁^2 - 3*x₁ + k = 0) →
  (x₂^2 - 3*x₂ + k = 0) →
  (x₁ = 2*x₂) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3988_398873


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l3988_398847

theorem sqrt_eight_div_sqrt_two_eq_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l3988_398847


namespace NUMINAMATH_CALUDE_initial_pizzas_count_l3988_398820

/-- The number of pizzas returned by customers. -/
def returned_pizzas : ℕ := 6

/-- The number of pizzas successfully served to customers. -/
def served_pizzas : ℕ := 3

/-- The total number of pizzas initially served by the restaurant. -/
def total_pizzas : ℕ := returned_pizzas + served_pizzas

theorem initial_pizzas_count : total_pizzas = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_pizzas_count_l3988_398820


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l3988_398870

theorem consecutive_numbers_product_divisibility (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, ∀ p : ℕ,
    Prime p →
    (p ≤ 2*n + 1 ↔ ∃ i : ℕ, i < n ∧ p ∣ (k + i)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l3988_398870


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3988_398854

theorem y_intercept_of_line (x y : ℝ) : 
  (x + y - 1 = 0) → (0 + y - 1 = 0 → y = 1) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3988_398854


namespace NUMINAMATH_CALUDE_sports_club_members_l3988_398803

/-- A sports club with members who play badminton, tennis, both, or neither -/
structure SportsClub where
  badminton : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ

/-- The total number of members in the sports club -/
def total_members (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub),
    club.badminton = 17 ∧
    club.tennis = 21 ∧
    club.both = 10 ∧
    club.neither = 2 ∧
    total_members club = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l3988_398803


namespace NUMINAMATH_CALUDE_smallest_n_boxes_l3988_398838

theorem smallest_n_boxes : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(17 * m - 3) % 11 = 0) ∧ 
  (17 * n - 3) % 11 = 0 ∧ 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_boxes_l3988_398838


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3988_398834

/-- Checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

theorem right_triangle_sets :
  ¬(isRightTriangle 4 6 8) ∧
  (isRightTriangle 5 12 13) ∧
  (isRightTriangle 6 8 10) ∧
  (isRightTriangle 7 24 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3988_398834


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l3988_398867

/-- The number of digits to the right of the decimal point when a positive rational number is expressed as a decimal. -/
def decimal_digits (q : ℚ) : ℕ :=
  sorry

/-- The fraction in question -/
def fraction : ℚ := (4^7) / (8^5 * 1250)

/-- Theorem stating that the number of digits to the right of the decimal point
    in the decimal representation of the given fraction is 3 -/
theorem fraction_decimal_digits :
  decimal_digits fraction = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l3988_398867


namespace NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l3988_398810

theorem complex_subtraction_and_multiplication :
  (5 - 4*I : ℂ) - 2*(3 + 6*I) = -1 - 16*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l3988_398810


namespace NUMINAMATH_CALUDE_bus_network_routes_count_l3988_398861

/-- A bus network in a city. -/
structure BusNetwork where
  /-- The set of bus stops. -/
  stops : Type
  /-- The set of bus routes. -/
  routes : Type
  /-- Predicate indicating if a stop is on a route. -/
  on_route : stops → routes → Prop

/-- Properties of a valid bus network. -/
class ValidBusNetwork (bn : BusNetwork) where
  /-- From any stop to any other stop, you can get there without a transfer. -/
  no_transfer : ∀ (s₁ s₂ : bn.stops), ∃ (r : bn.routes), bn.on_route s₁ r ∧ bn.on_route s₂ r
  /-- For any pair of routes, there is exactly one stop where you can transfer from one route to the other. -/
  unique_transfer : ∀ (r₁ r₂ : bn.routes), ∃! (s : bn.stops), bn.on_route s r₁ ∧ bn.on_route s r₂
  /-- Each route has exactly three stops. -/
  three_stops : ∀ (r : bn.routes), ∃! (s₁ s₂ s₃ : bn.stops), 
    bn.on_route s₁ r ∧ bn.on_route s₂ r ∧ bn.on_route s₃ r ∧ 
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃

/-- The theorem stating the relationship between the number of routes and stops. -/
theorem bus_network_routes_count {bn : BusNetwork} [ValidBusNetwork bn] [Fintype bn.stops] [Fintype bn.routes] : 
  Fintype.card bn.routes = Fintype.card bn.stops * (Fintype.card bn.stops - 1) + 1 :=
sorry

end NUMINAMATH_CALUDE_bus_network_routes_count_l3988_398861


namespace NUMINAMATH_CALUDE_bullet_hole_displacement_l3988_398844

/-- The displacement of the second hole relative to the first hole when a bullet is fired perpendicular to a moving train -/
theorem bullet_hole_displacement 
  (c : Real) -- speed of the train in km/h
  (c_prime : Real) -- speed of the bullet in m/s
  (a : Real) -- width of the train car in meters
  (h1 : c = 60) -- train speed is 60 km/h
  (h2 : c_prime = 40) -- bullet speed is 40 m/s
  (h3 : a = 4) -- train car width is 4 meters
  : (a * c * 1000 / 3600) / c_prime = 1.667 := by sorry

end NUMINAMATH_CALUDE_bullet_hole_displacement_l3988_398844


namespace NUMINAMATH_CALUDE_arrangement_pattern_sixtieth_number_is_eighteen_l3988_398811

/-- Represents the value in a specific position of the arrangement -/
def arrangementValue (position : ℕ) : ℕ :=
  let rowNum := (position - 1) / 3 + 1
  3 * rowNum

/-- The arrangement follows the specified pattern -/
theorem arrangement_pattern (n : ℕ) :
  ∀ k, k ≤ 3 * n → arrangementValue (3 * (n - 1) + k) = 3 * n :=
  sorry

/-- The 60th number in the arrangement is 18 -/
theorem sixtieth_number_is_eighteen :
  arrangementValue 60 = 18 :=
  sorry

end NUMINAMATH_CALUDE_arrangement_pattern_sixtieth_number_is_eighteen_l3988_398811


namespace NUMINAMATH_CALUDE_insufficient_info_for_production_l3988_398895

structure MachineRates where
  A : ℝ
  B : ℝ
  C : ℝ

def total_production (rates : MachineRates) (hours : ℝ) : ℝ :=
  hours * (rates.A + rates.B + rates.C)

theorem insufficient_info_for_production (P : ℝ) :
  ∀ (rates : MachineRates),
    7 * rates.A + 11 * rates.B = 305 →
    8 * rates.A + 22 * rates.C = P →
    ∃ (rates' : MachineRates),
      7 * rates'.A + 11 * rates'.B = 305 ∧
      8 * rates'.A + 22 * rates'.C = P ∧
      total_production rates 8 ≠ total_production rates' 8 :=
by
  sorry

#check insufficient_info_for_production

end NUMINAMATH_CALUDE_insufficient_info_for_production_l3988_398895


namespace NUMINAMATH_CALUDE_fraction_simplification_l3988_398832

theorem fraction_simplification : 
  (2015^2 : ℚ) / (2014^2 + 2016^2 - 2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3988_398832


namespace NUMINAMATH_CALUDE_lucky_number_in_13_consecutive_l3988_398816

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def isLucky (n : ℕ) : Prop := sumOfDigits n % 7 = 0

/-- Any sequence of 13 consecutive numbers contains at least one lucky number -/
theorem lucky_number_in_13_consecutive (n : ℕ) : 
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 12 ∧ isLucky k := by sorry

end NUMINAMATH_CALUDE_lucky_number_in_13_consecutive_l3988_398816


namespace NUMINAMATH_CALUDE_sports_books_count_l3988_398840

/-- Given the total number of books and the number of school books,
    prove that the number of sports books is 39. -/
theorem sports_books_count (total_books school_books : ℕ)
    (h1 : total_books = 58)
    (h2 : school_books = 19) :
    total_books - school_books = 39 := by
  sorry

end NUMINAMATH_CALUDE_sports_books_count_l3988_398840


namespace NUMINAMATH_CALUDE_area_not_above_x_axis_is_half_l3988_398836

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four points -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculates the area of a parallelogram -/
def parallelogramArea (pg : Parallelogram) : ℝ :=
  sorry

/-- Calculates the area of the portion of a parallelogram below or on the x-axis -/
def areaNotAboveXAxis (pg : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem area_not_above_x_axis_is_half (pg : Parallelogram) :
  pg.p = ⟨4, 2⟩ ∧ pg.q = ⟨-2, -2⟩ ∧ pg.r = ⟨-6, -6⟩ ∧ pg.s = ⟨0, -2⟩ →
  areaNotAboveXAxis pg = (parallelogramArea pg) / 2 :=
sorry

end NUMINAMATH_CALUDE_area_not_above_x_axis_is_half_l3988_398836


namespace NUMINAMATH_CALUDE_puzzle_completion_time_l3988_398883

/-- Calculates the time to complete puzzles given the number of puzzles, pieces per puzzle, and completion rate. -/
def time_to_complete_puzzles (num_puzzles : ℕ) (pieces_per_puzzle : ℕ) (pieces_per_interval : ℕ) (interval_minutes : ℕ) : ℕ :=
  let total_pieces := num_puzzles * pieces_per_puzzle
  let pieces_per_minute := pieces_per_interval / interval_minutes
  total_pieces / pieces_per_minute

/-- Proves that completing 2 puzzles of 2000 pieces each at a rate of 100 pieces per 10 minutes takes 400 minutes. -/
theorem puzzle_completion_time :
  time_to_complete_puzzles 2 2000 100 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_completion_time_l3988_398883


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3988_398837

theorem inequality_solution_set : 
  {x : ℝ | x^2 - 2*x - 5 > 2*x} = {x : ℝ | x > 5 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3988_398837


namespace NUMINAMATH_CALUDE_time_after_3339_minutes_l3988_398815

/-- Represents a time of day -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat
  deriving Repr

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : TimeOfDay
  deriving Repr

def minutesToDateTime (startDateTime : DateTime) (elapsedMinutes : Nat) : DateTime :=
  sorry

theorem time_after_3339_minutes :
  let startDateTime := DateTime.mk (Date.mk 2020 12 31) (TimeOfDay.mk 0 0)
  let endDateTime := minutesToDateTime startDateTime 3339
  endDateTime = DateTime.mk (Date.mk 2021 1 2) (TimeOfDay.mk 7 39) := by
  sorry

end NUMINAMATH_CALUDE_time_after_3339_minutes_l3988_398815


namespace NUMINAMATH_CALUDE_waste_processing_growth_equation_l3988_398827

/-- Represents the growth of processing capacity over two months -/
def processing_capacity_growth (initial_capacity : ℝ) (final_capacity : ℝ) (growth_rate : ℝ) : Prop :=
  initial_capacity * (1 + growth_rate)^2 = final_capacity

/-- The equation correctly models the company's waste processing capacity growth -/
theorem waste_processing_growth_equation :
  processing_capacity_growth 1000 1200 x ↔ 1000 * (1 + x)^2 = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_waste_processing_growth_equation_l3988_398827


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3988_398826

theorem inscribed_cube_surface_area :
  let outer_cube_edge : ℝ := 12
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_surface_area : ℝ := 6 * inner_cube_edge ^ 2
  inner_cube_surface_area = 288 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3988_398826


namespace NUMINAMATH_CALUDE_pizza_eaten_after_six_trips_l3988_398887

def eat_pizza (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem pizza_eaten_after_six_trips :
  eat_pizza 6 = 665/729 := by sorry

end NUMINAMATH_CALUDE_pizza_eaten_after_six_trips_l3988_398887


namespace NUMINAMATH_CALUDE_gcd_168_54_264_l3988_398821

theorem gcd_168_54_264 : Nat.gcd 168 (Nat.gcd 54 264) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_168_54_264_l3988_398821


namespace NUMINAMATH_CALUDE_screamers_lineup_count_l3988_398808

-- Define the total number of players
def total_players : ℕ := 12

-- Define the number of players in a lineup
def lineup_size : ℕ := 5

-- Define a function to calculate combinations
def combinations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.choose n r

-- Theorem statement
theorem screamers_lineup_count : 
  combinations (total_players - 2) (lineup_size - 1) * 2 + 
  combinations (total_players - 2) lineup_size = 672 := by
  sorry


end NUMINAMATH_CALUDE_screamers_lineup_count_l3988_398808


namespace NUMINAMATH_CALUDE_sheep_per_herd_l3988_398852

theorem sheep_per_herd (total_sheep : ℕ) (num_herds : ℕ) (h1 : total_sheep = 60) (h2 : num_herds = 3) :
  total_sheep / num_herds = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheep_per_herd_l3988_398852


namespace NUMINAMATH_CALUDE_point_in_region_l3988_398819

theorem point_in_region (m : ℝ) :
  m^2 - 3*m + 2 > 0 ↔ m < 1 ∨ m > 2 := by sorry

end NUMINAMATH_CALUDE_point_in_region_l3988_398819


namespace NUMINAMATH_CALUDE_investment_timing_l3988_398829

/-- Proves that B invested after 6 months given the conditions of the investment problem -/
theorem investment_timing (a_investment : ℕ) (b_investment : ℕ) (total_profit : ℕ) (a_profit : ℕ) :
  a_investment = 150 →
  b_investment = 200 →
  total_profit = 100 →
  a_profit = 60 →
  ∃ x : ℕ,
    x = 6 ∧
    (a_investment * 12 : ℚ) / (b_investment * (12 - x)) = (a_profit : ℚ) / (total_profit - a_profit) :=
by
  sorry


end NUMINAMATH_CALUDE_investment_timing_l3988_398829


namespace NUMINAMATH_CALUDE_cookie_box_cost_l3988_398899

/-- Given Faye's initial money, her mother's contribution, cupcake purchases, and remaining money,
    prove that each box of cookies costs $3. -/
theorem cookie_box_cost (initial_money : ℚ) (cupcake_price : ℚ) (num_cupcakes : ℕ) 
  (num_cookie_boxes : ℕ) (money_left : ℚ) :
  initial_money = 20 →
  cupcake_price = 3/2 →
  num_cupcakes = 10 →
  num_cookie_boxes = 5 →
  money_left = 30 →
  let total_money := initial_money + 2 * initial_money
  let money_after_cupcakes := total_money - (cupcake_price * num_cupcakes)
  let cookie_boxes_cost := money_after_cupcakes - money_left
  cookie_boxes_cost / num_cookie_boxes = 3 :=
by sorry


end NUMINAMATH_CALUDE_cookie_box_cost_l3988_398899


namespace NUMINAMATH_CALUDE_classroom_area_less_than_hectare_l3988_398872

-- Define the area of 1 hectare in square meters
def hectare_area : ℝ := 10000

-- Define the typical area of a classroom in square meters
def typical_classroom_area : ℝ := 60

-- Theorem stating that a typical classroom area is much less than a hectare
theorem classroom_area_less_than_hectare :
  typical_classroom_area < hectare_area ∧ typical_classroom_area / hectare_area < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_classroom_area_less_than_hectare_l3988_398872


namespace NUMINAMATH_CALUDE_algebraic_identities_l3988_398886

theorem algebraic_identities (a b : ℝ) : 
  ((-a)^2 * (a^2)^2 / a^3 = a^3) ∧ 
  ((a + b) * (a - b) - (a - b)^2 = 2*a*b - 2*b^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l3988_398886


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3988_398806

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 36 = 0) ↔ m = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3988_398806


namespace NUMINAMATH_CALUDE_missing_figure_proof_l3988_398824

theorem missing_figure_proof : ∃ x : ℝ, (0.25 / 100) * x = 0.04 ∧ x = 16 := by sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l3988_398824


namespace NUMINAMATH_CALUDE_hotel_guests_count_l3988_398800

theorem hotel_guests_count (oates_count hall_count both_count : ℕ) 
  (ho : oates_count = 40)
  (hh : hall_count = 70)
  (hb : both_count = 10) :
  oates_count + hall_count - both_count = 100 := by
  sorry

end NUMINAMATH_CALUDE_hotel_guests_count_l3988_398800


namespace NUMINAMATH_CALUDE_proton_origin_probability_proton_max_probability_at_six_l3988_398871

/-- Represents the probability of a proton being at a specific position after n moves --/
def ProtonProbability (n : ℕ) (position : ℤ) : ℚ :=
  sorry

/-- The probability of the proton being at the origin after 4 moves --/
theorem proton_origin_probability : ProtonProbability 4 0 = 3/8 :=
  sorry

/-- The number of moves that maximizes the probability of the proton being at position 6 --/
def MaxProbabilityMoves : Finset ℕ :=
  sorry

/-- The probability of the proton being at position 6 is maximized when the number of moves is either 34 or 36 --/
theorem proton_max_probability_at_six :
  MaxProbabilityMoves = {34, 36} :=
  sorry

end NUMINAMATH_CALUDE_proton_origin_probability_proton_max_probability_at_six_l3988_398871


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3988_398813

theorem multiplication_subtraction_equality : 75 * 1414 - 25 * 1414 = 70700 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3988_398813


namespace NUMINAMATH_CALUDE_basketball_court_measurements_l3988_398889

theorem basketball_court_measurements :
  ∃! (A B C D E F : ℝ),
    A - B = C ∧
    D = 2 * (A + B) ∧
    E = A * B ∧
    F = 3 ∧
    A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧ F > 0 ∧
    ({A, B, C, D, E, F} : Set ℝ) = {86, 13, 420, 15, 28, 3} ∧
    A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 :=
by sorry

end NUMINAMATH_CALUDE_basketball_court_measurements_l3988_398889


namespace NUMINAMATH_CALUDE_sales_discount_effect_l3988_398801

theorem sales_discount_effect (discount : ℝ) 
  (h1 : discount = 10)
  (h2 : (1 - discount / 100) * 1.12 = 1.008) : 
  discount = 10 := by
sorry

end NUMINAMATH_CALUDE_sales_discount_effect_l3988_398801


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l3988_398868

theorem diophantine_equation_solvable (p : ℕ) (hp : Nat.Prime p) : 
  ∃ (x y z : ℤ), x^2 + y^2 + p * z = 2003 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l3988_398868


namespace NUMINAMATH_CALUDE_feuerbach_theorem_l3988_398830

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The midpoint circle of a triangle -/
def midpointCircle (t : Triangle) : Circle := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircles of a triangle -/
def excircles (t : Triangle) : Fin 3 → Circle := sorry

/-- Two circles are tangent -/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- Feuerbach's theorem -/
theorem feuerbach_theorem (t : Triangle) : 
  (areTangent (midpointCircle t) (incircle t)) ∧ 
  (∀ i : Fin 3, areTangent (midpointCircle t) (excircles t i)) := by
  sorry

end NUMINAMATH_CALUDE_feuerbach_theorem_l3988_398830


namespace NUMINAMATH_CALUDE_lights_remaining_on_l3988_398804

def total_lights : ℕ := 2013

def lights_on_after_switches (n : ℕ) : ℕ :=
  n - (n / 2 + n / 3 + n / 5 - n / 6 - n / 10 - n / 15 + n / 30)

theorem lights_remaining_on :
  lights_on_after_switches total_lights = 1006 := by
  sorry

end NUMINAMATH_CALUDE_lights_remaining_on_l3988_398804


namespace NUMINAMATH_CALUDE_equation_solution_l3988_398853

theorem equation_solution : 
  ∃! x : ℚ, (3 * x - 17) / 4 = (x + 12) / 5 ∧ x = 133 / 11 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3988_398853


namespace NUMINAMATH_CALUDE_only_25_is_five_times_greater_than_last_digit_l3988_398881

def lastDigit (n : Nat) : Nat :=
  n % 10

theorem only_25_is_five_times_greater_than_last_digit :
  ∀ n : Nat, n > 0 → (n = 5 * lastDigit n + lastDigit n) → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_only_25_is_five_times_greater_than_last_digit_l3988_398881


namespace NUMINAMATH_CALUDE_apartment_complex_flashlights_joas_apartment_complex_flashlights_l3988_398851

/-- Calculates the total number of emergency flashlights in an apartment complex -/
theorem apartment_complex_flashlights (total_buildings : ℕ) 
  (stories_per_building : ℕ) (families_per_floor_type1 : ℕ) 
  (families_per_floor_type2 : ℕ) (flashlights_per_family : ℕ) : ℕ :=
  let half_buildings := total_buildings / 2
  let families_type1 := half_buildings * stories_per_building * families_per_floor_type1
  let families_type2 := half_buildings * stories_per_building * families_per_floor_type2
  let total_families := families_type1 + families_type2
  total_families * flashlights_per_family

/-- The number of emergency flashlights in Joa's apartment complex -/
theorem joas_apartment_complex_flashlights : 
  apartment_complex_flashlights 8 15 4 5 2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_apartment_complex_flashlights_joas_apartment_complex_flashlights_l3988_398851


namespace NUMINAMATH_CALUDE_complex_product_equality_complex_sum_equality_l3988_398809

-- Define the complex number i
def i : ℂ := Complex.I

-- Part 1
theorem complex_product_equality : 
  (1 : ℂ) * (1 - i) * (-1/2 + (Real.sqrt 3)/2 * i) * (1 + i) = -1 + Real.sqrt 3 * i := by sorry

-- Part 2
theorem complex_sum_equality :
  (2 + 2*i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i))^2010 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_product_equality_complex_sum_equality_l3988_398809


namespace NUMINAMATH_CALUDE_stock_value_order_l3988_398876

def initial_investment : ℝ := 200

def omega_year1_change : ℝ := 1.15
def bravo_year1_change : ℝ := 0.70
def zeta_year1_change : ℝ := 1.00

def omega_year2_change : ℝ := 0.90
def bravo_year2_change : ℝ := 1.30
def zeta_year2_change : ℝ := 1.00

def omega_final : ℝ := initial_investment * omega_year1_change * omega_year2_change
def bravo_final : ℝ := initial_investment * bravo_year1_change * bravo_year2_change
def zeta_final : ℝ := initial_investment * zeta_year1_change * zeta_year2_change

theorem stock_value_order : bravo_final < zeta_final ∧ zeta_final < omega_final :=
by sorry

end NUMINAMATH_CALUDE_stock_value_order_l3988_398876


namespace NUMINAMATH_CALUDE_prime_divisor_fourth_power_l3988_398814

theorem prime_divisor_fourth_power (n : ℕ+) 
  (h : ∀ d : ℕ+, d ∣ n → ¬(n^2 ≤ d^4 ∧ d^4 ≤ n^3)) : 
  ∃ p : ℕ, p.Prime ∧ p ∣ n ∧ p^4 > n^3 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_fourth_power_l3988_398814


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_36_l3988_398875

theorem sum_of_roots_equals_36 : ∃ (x₁ x₂ x₃ : ℝ),
  (∀ x, (11 - x)^3 + (13 - x)^3 = (24 - 2*x)^3 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ + x₂ + x₃ = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_36_l3988_398875


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l3988_398862

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℕ, ∃ m : ℕ,
    (120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
    (∀ k : ℕ, k > 120 → ∃ p : ℕ, ¬(k ∣ (p * (p + 1) * (p + 2) * (p + 3) * (p + 4)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l3988_398862


namespace NUMINAMATH_CALUDE_fifth_sample_number_l3988_398888

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (first_sample : ℕ) (group : ℕ) : ℕ :=
  first_sample + (total / sample_size) * (group - 1)

/-- Theorem: In a systematic sampling of 100 samples from 2000 items, 
    if the first sample is numbered 11, then the fifth sample will be numbered 91 -/
theorem fifth_sample_number :
  systematic_sample 2000 100 11 5 = 91 := by
  sorry

end NUMINAMATH_CALUDE_fifth_sample_number_l3988_398888


namespace NUMINAMATH_CALUDE_b_31_mod_33_l3988_398849

/-- Definition of b_n as the concatenation of integers from 1 to n --/
def b (n : ℕ) : ℕ :=
  -- This is a placeholder definition. The actual implementation would be more complex.
  sorry

/-- Theorem stating that b_31 mod 33 = 11 --/
theorem b_31_mod_33 : b 31 % 33 = 11 := by
  sorry

end NUMINAMATH_CALUDE_b_31_mod_33_l3988_398849


namespace NUMINAMATH_CALUDE_largest_negative_integer_l3988_398828

theorem largest_negative_integer : 
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l3988_398828


namespace NUMINAMATH_CALUDE_total_books_eq_sum_l3988_398817

/-- The total number of different books in the 'crazy silly school' series -/
def total_books : ℕ := sorry

/-- The number of books already read from the series -/
def books_read : ℕ := 8

/-- The number of books left to read from the series -/
def books_left : ℕ := 6

/-- Theorem stating that the total number of books is equal to the sum of books read and books left to read -/
theorem total_books_eq_sum : total_books = books_read + books_left := by sorry

end NUMINAMATH_CALUDE_total_books_eq_sum_l3988_398817


namespace NUMINAMATH_CALUDE_alice_marbles_distinct_choices_l3988_398879

/-- Represents the colors of marbles --/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents the marble collection --/
structure MarbleCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of distinct ways to choose 2 marbles --/
def distinctChoices (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that for Alice's marble collection, there are 9 distinct ways to choose 2 marbles --/
theorem alice_marbles_distinct_choices :
  let aliceCollection : MarbleCollection := ⟨3, 2, 1, 4⟩
  distinctChoices aliceCollection = 9 :=
sorry

end NUMINAMATH_CALUDE_alice_marbles_distinct_choices_l3988_398879


namespace NUMINAMATH_CALUDE_nth_equation_l3988_398877

theorem nth_equation (n : ℕ) : 
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1) : ℚ) = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_l3988_398877


namespace NUMINAMATH_CALUDE_finite_seq_nat_countable_l3988_398835

-- Define the type for finite sequences of natural numbers
def FiniteSeqNat := List Nat

-- Statement of the theorem
theorem finite_seq_nat_countable : 
  ∃ f : FiniteSeqNat → Nat, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_finite_seq_nat_countable_l3988_398835


namespace NUMINAMATH_CALUDE_joshua_oranges_expenditure_l3988_398825

/-- The amount Joshua spent on buying oranges -/
def joshua_spent (num_oranges : ℕ) (selling_price profit : ℚ) : ℚ :=
  (num_oranges : ℚ) * (selling_price - profit)

/-- Theorem stating the amount Joshua spent on oranges -/
theorem joshua_oranges_expenditure :
  joshua_spent 25 0.60 0.10 = 12.50 := by
  sorry

end NUMINAMATH_CALUDE_joshua_oranges_expenditure_l3988_398825


namespace NUMINAMATH_CALUDE_absolute_difference_l3988_398841

/-- Given a set of five numbers {m, n, 9, 8, 10} with an average of 9 and a variance of 2, |m - n| = 4 -/
theorem absolute_difference (m n : ℝ) 
  (h_avg : (m + n + 9 + 8 + 10) / 5 = 9)
  (h_var : ((m - 9)^2 + (n - 9)^2 + (9 - 9)^2 + (8 - 9)^2 + (10 - 9)^2) / 5 = 2) :
  |m - n| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l3988_398841


namespace NUMINAMATH_CALUDE_tanya_work_days_l3988_398880

/-- Given Sakshi can do a piece of work in 12 days and Tanya is 20% more efficient than Sakshi,
    prove that Tanya can complete the same piece of work in 10 days. -/
theorem tanya_work_days (sakshi_days : ℝ) (tanya_efficiency : ℝ) :
  sakshi_days = 12 →
  tanya_efficiency = 1.2 →
  (sakshi_days / tanya_efficiency) = 10 := by
sorry

end NUMINAMATH_CALUDE_tanya_work_days_l3988_398880


namespace NUMINAMATH_CALUDE_real_root_of_cubic_l3988_398896

/-- Given a cubic polynomial with real coefficients c and d, 
    if -3 + 2i is a root, then 53/5 is the real root. -/
theorem real_root_of_cubic (c d : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (fun x : ℂ => c * x ^ 3 - x ^ 2 + d * x + 30) (-3 + 2 * Complex.I) = 0 →
  (fun x : ℝ => c * x ^ 3 - x ^ 2 + d * x + 30) (53 / 5) = 0 :=
by sorry

end NUMINAMATH_CALUDE_real_root_of_cubic_l3988_398896


namespace NUMINAMATH_CALUDE_system_solution_l3988_398823

theorem system_solution (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3988_398823


namespace NUMINAMATH_CALUDE_equation_solution_l3988_398805

theorem equation_solution (x : ℝ) : x ≠ 2 → (-x^2 = (4*x + 2) / (x - 2)) ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3988_398805


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3988_398898

-- Define the set of real numbers that satisfy the inequality
def solution_set : Set ℝ := {x | x ≠ 0 ∧ (1 / x < x)}

-- Theorem statement
theorem inequality_solution_set : 
  solution_set = {x | -1 < x ∧ x < 0} ∪ {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3988_398898


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_for_quadratic_l3988_398866

theorem no_positive_integer_solutions_for_quadratic :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
    ¬∃ x : ℕ, x > 0 ∧ x^2 - (A + 1) * x + A * 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_for_quadratic_l3988_398866


namespace NUMINAMATH_CALUDE_trig_simplification_l3988_398845

theorem trig_simplification :
  (Real.cos (20 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3988_398845


namespace NUMINAMATH_CALUDE_divides_power_sum_l3988_398882

theorem divides_power_sum (a b c : ℤ) (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∀ k : ℕ, (a + b + c) ∣ (a^(2^k) + b^(2^k) + c^(2^k)) :=
sorry

end NUMINAMATH_CALUDE_divides_power_sum_l3988_398882


namespace NUMINAMATH_CALUDE_largest_reciprocal_l3988_398842

theorem largest_reciprocal (a b c d e : ℚ) 
  (ha : a = 5/6) (hb : b = 1/2) (hc : c = 3) (hd : d = 8/3) (he : e = 240) :
  (1 / b > 1 / a) ∧ (1 / b > 1 / c) ∧ (1 / b > 1 / d) ∧ (1 / b > 1 / e) :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l3988_398842


namespace NUMINAMATH_CALUDE_regular_pentagons_are_similar_l3988_398874

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 angles of equal measure. -/
structure RegularPentagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two shapes are similar if they have the same shape but not necessarily the same size. -/
def are_similar (p1 p2 : RegularPentagon) : Prop :=
  ∃ k : ℝ, k > 0 ∧ p1.side_length = k * p2.side_length

/-- Theorem: Any two regular pentagons are similar. -/
theorem regular_pentagons_are_similar (p1 p2 : RegularPentagon) : are_similar p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagons_are_similar_l3988_398874


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3988_398890

theorem partial_fraction_decomposition (D E F : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 6 →
    1 / (x^3 - 3*x^2 - 4*x + 12) = D / (x - 1) + E / (x + 2) + F / (x + 2)^2) →
  D = -1/15 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3988_398890


namespace NUMINAMATH_CALUDE_binary_199_ones_minus_zeros_l3988_398843

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary' (n : ℕ) : List Bool :=
    if n = 0 then [] else (n % 2 = 1) :: toBinary' (n / 2)
  toBinary' n

/-- Count the number of true values in a list of booleans -/
def countTrue (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Count the number of false values in a list of booleans -/
def countFalse (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_199_ones_minus_zeros :
  let binary := toBinary 199
  let ones := countTrue binary
  let zeros := countFalse binary
  ones - zeros = 2 := by sorry

end NUMINAMATH_CALUDE_binary_199_ones_minus_zeros_l3988_398843


namespace NUMINAMATH_CALUDE_tailwind_speed_l3988_398846

def plane_speed_with_tailwind : ℝ := 460
def plane_speed_against_tailwind : ℝ := 310

theorem tailwind_speed : ∃ (plane_speed tailwind_speed : ℝ),
  plane_speed + tailwind_speed = plane_speed_with_tailwind ∧
  plane_speed - tailwind_speed = plane_speed_against_tailwind ∧
  tailwind_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_tailwind_speed_l3988_398846


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l3988_398893

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) where
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (n : ℕ) (X : BinomialDistribution n) (k : ℕ) : ℝ :=
  (n.choose k) * X.p^k * (1 - X.p)^(n - k)

theorem binomial_probability_problem (X : BinomialDistribution 4) 
  (h3 : X.p < 1/2) 
  (h4 : binomialPMF 4 X 2 = 8/27) : 
  binomialPMF 4 X 1 = 32/81 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l3988_398893


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3988_398878

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47,
    prove that the 8th term is 71. -/
theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℤ)  -- The arithmetic sequence
  (h1 : a 4 = 23)  -- The 4th term is 23
  (h2 : a 6 = 47)  -- The 6th term is 47
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- The sequence is arithmetic
  : a 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3988_398878


namespace NUMINAMATH_CALUDE_intersection_A_B_l3988_398818

-- Define the sets A and B
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3988_398818


namespace NUMINAMATH_CALUDE_graph_shift_l3988_398858

/-- Given a function f: ℝ → ℝ, prove that f(x - 2) + 1 is equivalent to
    shifting the graph of f(x) right by 2 units and up by 1 unit. -/
theorem graph_shift (f : ℝ → ℝ) (x : ℝ) :
  f (x - 2) + 1 = (fun y ↦ f (y - 2)) (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_graph_shift_l3988_398858


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l3988_398848

/-- A geometric sequence with specific partial sums -/
structure GeometricSequence where
  S : ℝ  -- Sum of first 2 terms
  T : ℝ  -- Sum of first 4 terms
  R : ℝ  -- Sum of first 6 terms

/-- Theorem stating the relation between partial sums of a geometric sequence -/
theorem geometric_sequence_sum_relation (seq : GeometricSequence) :
  seq.S^2 + seq.T^2 = seq.S * (seq.T + seq.R) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l3988_398848


namespace NUMINAMATH_CALUDE_donnas_truck_weight_l3988_398850

-- Define the given weights and quantities
def bridge_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000

-- Define the theorem
theorem donnas_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let dryers_weight := dryers * dryer_weight
  let total_weight := empty_truck_weight + soda_weight + produce_weight + dryers_weight
  total_weight = 24000 := by
  sorry

end NUMINAMATH_CALUDE_donnas_truck_weight_l3988_398850


namespace NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l3988_398863

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| - 10

-- State the theorem
theorem min_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (f x₀ = -10) ∧ (x₀ = -3) := by
  sorry

end NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l3988_398863


namespace NUMINAMATH_CALUDE_music_store_sales_calculation_l3988_398807

/-- Represents the sales data for a mall with two stores -/
structure MallSales where
  num_cars : ℕ
  customers_per_car : ℕ
  sports_store_sales : ℕ

/-- Calculates the number of sales made by the music store -/
def music_store_sales (mall : MallSales) : ℕ :=
  mall.num_cars * mall.customers_per_car - mall.sports_store_sales

/-- Theorem: The music store sales is equal to the total customers minus sports store sales -/
theorem music_store_sales_calculation (mall : MallSales) 
  (h1 : mall.num_cars = 10)
  (h2 : mall.customers_per_car = 5)
  (h3 : mall.sports_store_sales = 20) :
  music_store_sales mall = 30 := by
  sorry

end NUMINAMATH_CALUDE_music_store_sales_calculation_l3988_398807


namespace NUMINAMATH_CALUDE_third_number_in_set_l3988_398855

theorem third_number_in_set (x : ℝ) : 
  (20 + 40 + x) / 3 = (10 + 70 + 16) / 3 + 8 → x = 60 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_set_l3988_398855


namespace NUMINAMATH_CALUDE_root_equation_and_product_l3988_398833

theorem root_equation_and_product (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (2*a - 1)*x + a^2 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ + 2) * (x₂ + 2) = 11 →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_root_equation_and_product_l3988_398833


namespace NUMINAMATH_CALUDE_hike_pace_proof_l3988_398822

/-- Proves that given the conditions of the hike, the pace to the destination is 4 miles per hour -/
theorem hike_pace_proof (distance : ℝ) (return_pace : ℝ) (total_time : ℝ) (pace_to : ℝ) : 
  distance = 12 → 
  return_pace = 6 → 
  total_time = 5 → 
  distance / pace_to + distance / return_pace = total_time → 
  pace_to = 4 := by
sorry

end NUMINAMATH_CALUDE_hike_pace_proof_l3988_398822


namespace NUMINAMATH_CALUDE_water_amount_equals_sugar_amount_l3988_398894

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  flour : ℚ
  water : ℚ
  sugar : ℚ

/-- The original recipe ratio -/
def original_ratio : RecipeRatio := ⟨10, 6, 3⟩

/-- The new recipe ratio -/
def new_ratio : RecipeRatio := 
  let flour_water_doubled := original_ratio.flour / original_ratio.water * 2
  let flour_sugar_halved := original_ratio.flour / original_ratio.sugar / 2
  ⟨
    flour_water_doubled * original_ratio.water,
    original_ratio.water,
    flour_sugar_halved * original_ratio.sugar
  ⟩

/-- Amount of sugar in the new recipe -/
def sugar_amount : ℚ := 4

theorem water_amount_equals_sugar_amount : 
  (new_ratio.water / new_ratio.sugar) * sugar_amount = sugar_amount := by
  sorry

end NUMINAMATH_CALUDE_water_amount_equals_sugar_amount_l3988_398894


namespace NUMINAMATH_CALUDE_probability_no_shaded_square_l3988_398865

/-- Represents a rectangle in the grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The grid configuration --/
def grid_width : Nat := 201
def grid_height : Nat := 3
def shaded_column : Nat := grid_width / 2 + 1

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Bool :=
  r.left ≤ shaded_column && shaded_column ≤ r.right

/-- Counts the total number of possible rectangles --/
def total_rectangles : Nat :=
  (grid_width.choose 2) * (grid_height.choose 2)

/-- Counts the number of rectangles that contain a shaded square --/
def shaded_rectangles : Nat :=
  grid_height * (shaded_column - 1) * (grid_width - shaded_column)

/-- The main theorem --/
theorem probability_no_shaded_square :
  (total_rectangles - shaded_rectangles) / total_rectangles = 100 / 201 := by
  sorry


end NUMINAMATH_CALUDE_probability_no_shaded_square_l3988_398865


namespace NUMINAMATH_CALUDE_eugene_pencils_l3988_398864

theorem eugene_pencils (initial_pencils : ℝ) (pencils_given : ℝ) :
  initial_pencils = 51.0 →
  pencils_given = 6.0 →
  initial_pencils - pencils_given = 45.0 :=
by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l3988_398864


namespace NUMINAMATH_CALUDE_equation_solution_l3988_398884

theorem equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3988_398884


namespace NUMINAMATH_CALUDE_trajectory_and_line_equations_l3988_398857

-- Define the points
def A : ℝ × ℝ := (0, 3)
def O : ℝ × ℝ := (0, 0)
def N : ℝ × ℝ := (-1, 3)

-- Define the moving point M
def M : ℝ × ℝ → Prop := fun (x, y) ↦ 
  (x - A.1)^2 + (y - A.2)^2 = 4 * ((x - O.1)^2 + (y - O.2)^2)

-- Define the trajectory
def Trajectory : ℝ × ℝ → Prop := fun (x, y) ↦ 
  x^2 + (y + 1)^2 = 4

-- Define the line equations
def Line1 : ℝ × ℝ → Prop := fun (x, y) ↦ x = -1
def Line2 : ℝ × ℝ → Prop := fun (x, y) ↦ 15*x + 8*y - 9 = 0

theorem trajectory_and_line_equations :
  (∀ p, M p ↔ Trajectory p) ∧
  (∃ l, (l = Line1 ∨ l = Line2) ∧
        (l N) ∧
        (∃ p q : ℝ × ℝ, p ≠ q ∧ Trajectory p ∧ Trajectory q ∧ l p ∧ l q ∧
          (p.1 - q.1)^2 + (p.2 - q.2)^2 = 12)) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_line_equations_l3988_398857


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3988_398897

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ x : ℕ, p x) ↔ (∀ x : ℕ, ¬ p x) := by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℕ, x^2 ≥ x) ↔ (∀ x : ℕ, x^2 < x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3988_398897


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l3988_398839

/-- If (m+1)x + 3y^m = 5 is a linear equation in x and y, then m = 1 -/
theorem linear_equation_m_value (m : ℝ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, (m + 1) * x + 3 * y^m = a * x + b * y + c) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l3988_398839


namespace NUMINAMATH_CALUDE_manuscript_review_theorem_l3988_398892

/-- Represents the review process for a manuscript --/
structure ManuscriptReview where
  initial_pass_prob : ℝ
  third_expert_pass_prob : ℝ

/-- Calculates the probability of a manuscript being accepted --/
def acceptance_probability (review : ManuscriptReview) : ℝ :=
  review.initial_pass_prob ^ 2 + 
  2 * review.initial_pass_prob * (1 - review.initial_pass_prob) * review.third_expert_pass_prob

/-- Represents the distribution of accepted manuscripts --/
def manuscript_distribution (n : ℕ) (p : ℝ) : List (ℕ × ℝ) :=
  sorry

/-- Theorem stating the probability of acceptance and the distribution of accepted manuscripts --/
theorem manuscript_review_theorem (review : ManuscriptReview) 
    (h1 : review.initial_pass_prob = 0.5)
    (h2 : review.third_expert_pass_prob = 0.3) :
  acceptance_probability review = 0.4 ∧ 
  manuscript_distribution 4 (acceptance_probability review) = 
    [(0, 0.1296), (1, 0.3456), (2, 0.3456), (3, 0.1536), (4, 0.0256)] :=
  sorry

end NUMINAMATH_CALUDE_manuscript_review_theorem_l3988_398892
