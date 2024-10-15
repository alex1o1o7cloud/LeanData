import Mathlib

namespace NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l1706_170657

-- Define the cough rates and time
def georgia_cough_rate : ℕ := 5
def robert_cough_rate : ℕ := 2 * georgia_cough_rate
def time_minutes : ℕ := 20

-- Define the total coughs function
def total_coughs (georgia_rate : ℕ) (robert_rate : ℕ) (time : ℕ) : ℕ :=
  georgia_rate * time + robert_rate * time

-- Theorem statement
theorem total_coughs_after_20_minutes :
  total_coughs georgia_cough_rate robert_cough_rate time_minutes = 300 := by
  sorry


end NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l1706_170657


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1706_170696

theorem min_value_theorem (x : ℝ) :
  (x^2 + 12) / Real.sqrt (x^2 + x + 5) ≥ 2 * Real.sqrt 7 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, (x^2 + 12) / Real.sqrt (x^2 + x + 5) = 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1706_170696


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1706_170678

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1706_170678


namespace NUMINAMATH_CALUDE_commodity_price_increase_l1706_170676

/-- The annual price increase of commodity X in cents -/
def annual_increase_X : ℝ := 30

/-- The annual price increase of commodity Y in cents -/
def annual_increase_Y : ℝ := 20

/-- The price of commodity X in 2001 in dollars -/
def price_X_2001 : ℝ := 4.20

/-- The price of commodity Y in 2001 in dollars -/
def price_Y_2001 : ℝ := 4.40

/-- The number of years between 2001 and 2010 -/
def years : ℕ := 9

/-- The difference in price between X and Y in 2010 in cents -/
def price_difference_2010 : ℝ := 70

theorem commodity_price_increase :
  annual_increase_X = 30 ∧
  price_X_2001 + (annual_increase_X / 100 * years) =
  price_Y_2001 + (annual_increase_Y / 100 * years) + price_difference_2010 / 100 :=
by sorry

end NUMINAMATH_CALUDE_commodity_price_increase_l1706_170676


namespace NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l1706_170693

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of days from Monday to a given day -/
def daysFromMonday (day : DayOfWeek) : Nat :=
  match day with
  | .Monday => 0
  | .Tuesday => 1
  | .Wednesday => 2
  | .Thursday => 3
  | .Friday => 4
  | .Saturday => 5
  | .Sunday => 6

def dailyConsumption : Rat := 3/5
def initialCans : Nat := 8

theorem cat_food_finished_on_sunday :
  ∃ (day : DayOfWeek),
    (daysFromMonday day + 1) * dailyConsumption > initialCans ∧
    (daysFromMonday day) * dailyConsumption ≤ initialCans ∧
    day = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l1706_170693


namespace NUMINAMATH_CALUDE_uvw_sum_squared_product_bound_l1706_170613

theorem uvw_sum_squared_product_bound (u v w : ℝ) 
  (h_nonneg : u ≥ 0 ∧ v ≥ 0 ∧ w ≥ 0) 
  (h_sum : u + v + w = 2) : 
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ 
  u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_uvw_sum_squared_product_bound_l1706_170613


namespace NUMINAMATH_CALUDE_no_solution_for_room_occupancy_l1706_170641

theorem no_solution_for_room_occupancy : ¬∃ (x y : ℕ), x + y = 2019 ∧ 2 * x - y = 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_room_occupancy_l1706_170641


namespace NUMINAMATH_CALUDE_geometric_propositions_l1706_170679

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the basic relations
variable (belongs_to : Point → Line → Prop)
variable (lies_in : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersect_lines : Line → Line → Prop)
variable (intersect_line_plane : Line → Plane → Prop)
variable (intersect_planes : Plane → Plane → Line)

-- Define the theorem
theorem geometric_propositions 
  (a b : Line) (α β γ : Plane) :
  (-- Proposition 2
   intersect_lines a b ∧ 
   (¬ line_in_plane a α) ∧ (¬ line_in_plane a β) ∧
   (¬ line_in_plane b α) ∧ (¬ line_in_plane b β) ∧
   parallel_line_plane a α ∧ parallel_line_plane a β ∧
   parallel_line_plane b α ∧ parallel_line_plane b β →
   parallel_planes α β) ∧
  (-- Proposition 3
   line_in_plane a α ∧ 
   parallel_line_plane a β ∧
   intersect_planes α β = b →
   parallel_lines a b) :=
sorry

end NUMINAMATH_CALUDE_geometric_propositions_l1706_170679


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1706_170665

/-- Given two quadratic functions with two distinct roots each, prove that a third related quadratic function has no real roots -/
theorem quadratic_intersection (a b c : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*a*x₁ + b^2 = 0 ∧ x₂^2 + 2*a*x₂ + b^2 = 0) ∧
   (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + 2*b*y₁ + c^2 = 0 ∧ y₂^2 + 2*b*y₂ + c^2 = 0)) →
  (∀ z : ℝ, z^2 + 2*c*z + a^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l1706_170665


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1706_170664

theorem complex_equation_solution :
  ∃ z : ℂ, (5 - 3 * Complex.I * z = 2 + 5 * Complex.I * z) ∧ (z = -3 * Complex.I / 8) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1706_170664


namespace NUMINAMATH_CALUDE_f_min_value_l1706_170684

/-- The function f(x) = (x^2 + 33) / x for x ∈ ℕ* -/
def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

/-- The minimum value of f(x) is 23/2 -/
theorem f_min_value : ∀ x : ℕ+, f x ≥ 23/2 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l1706_170684


namespace NUMINAMATH_CALUDE_robert_remaining_kicks_l1706_170644

/-- Calculates the remaining kicks needed to reach a goal. -/
def remaining_kicks (total_goal : ℕ) (kicks_before_break : ℕ) (kicks_after_break : ℕ) : ℕ :=
  total_goal - (kicks_before_break + kicks_after_break)

/-- Proves that given the specific conditions, the remaining kicks is 19. -/
theorem robert_remaining_kicks : 
  remaining_kicks 98 43 36 = 19 := by
  sorry

end NUMINAMATH_CALUDE_robert_remaining_kicks_l1706_170644


namespace NUMINAMATH_CALUDE_three_coins_same_probability_l1706_170669

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The number of specific coins we're interested in -/
def num_specific_coins : ℕ := 3

/-- The probability of three specific coins out of five all coming up the same -/
theorem three_coins_same_probability :
  (2^(num_specific_coins - 1) * 2^(num_coins - num_specific_coins)) / 2^num_coins = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_three_coins_same_probability_l1706_170669


namespace NUMINAMATH_CALUDE_sum_of_three_distinct_divisors_l1706_170635

theorem sum_of_three_distinct_divisors (n : ℕ+) :
  (∃ (d₁ d₂ d₃ : ℕ+), d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧
    d₁ + d₂ + d₃ = n) ↔
  (∃ k : ℕ+, n = 6 * k) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_distinct_divisors_l1706_170635


namespace NUMINAMATH_CALUDE_team_total_score_l1706_170600

/-- Given a team of 10 people in a shooting competition, prove that their total score is 905 points. -/
theorem team_total_score 
  (team_size : ℕ) 
  (best_score : ℕ) 
  (hypothetical_best : ℕ) 
  (hypothetical_average : ℕ) :
  team_size = 10 →
  best_score = 95 →
  hypothetical_best = 110 →
  hypothetical_average = 92 →
  (hypothetical_best - best_score + (team_size * hypothetical_average)) = (team_size * 905) :=
by sorry

end NUMINAMATH_CALUDE_team_total_score_l1706_170600


namespace NUMINAMATH_CALUDE_divisibility_of_binomial_coefficient_l1706_170683

theorem divisibility_of_binomial_coefficient (m n d : ℕ) : 
  0 < m → 0 < n → m ≤ n → d = Nat.gcd m n → 
  ∃ k : ℤ, (d : ℤ) * (Nat.choose n m : ℤ) = k * (n : ℤ) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_binomial_coefficient_l1706_170683


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1706_170609

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  ((b * Real.cos C) / Real.cos B + c = (2 * Real.sqrt 3 / 3) * a) →
  B = π / 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_angle_proof_l1706_170609


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1706_170689

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6 →
  b = -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6 →
  c = Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6 →
  d = -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6 →
  (1/a + 1/b + 1/c + 1/d)^2 = 3/50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1706_170689


namespace NUMINAMATH_CALUDE_expand_expression_l1706_170672

theorem expand_expression (a b c : ℝ) : (a + b - c) * (a - b - c) = a^2 - 2*a*c + c^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1706_170672


namespace NUMINAMATH_CALUDE_skater_practice_hours_l1706_170618

/-- Given a skater's practice schedule, calculate the total weekly practice hours. -/
theorem skater_practice_hours (weekend_hours : ℕ) (additional_weekday_hours : ℕ) : 
  weekend_hours = 8 → additional_weekday_hours = 17 → 
  weekend_hours + (weekend_hours + additional_weekday_hours) = 33 := by
  sorry

#check skater_practice_hours

end NUMINAMATH_CALUDE_skater_practice_hours_l1706_170618


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1706_170658

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ bridge_length : ℝ,
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length ∧
    bridge_length = 217.5 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1706_170658


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1706_170652

theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) :
  current_speed = 8 →
  downstream_distance = 6.283333333333333 →
  downstream_time = 13 / 60 →
  ∃ (boat_speed : ℝ), 
    boat_speed = 21 ∧ 
    (boat_speed + current_speed) * downstream_time = downstream_distance :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1706_170652


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l1706_170656

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (q : ℝ) (q_pos : 0 < q) :
  (4 / 3 * Real.pi * q ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * q) ^ 3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l1706_170656


namespace NUMINAMATH_CALUDE_clock_strike_theorem_l1706_170616

/-- Represents the number of seconds it takes for a clock to strike a given number of times. -/
def strike_time (num_strikes : ℕ) (seconds : ℝ) : Prop :=
  num_strikes > 0 ∧ seconds > 0 ∧ 
  (seconds / (num_strikes - 1 : ℝ)) = (8 : ℝ) / (5 - 1 : ℝ)

/-- Theorem stating that if a clock takes 8 seconds to strike 5 times, 
    it will take 18 seconds to strike 10 times. -/
theorem clock_strike_theorem :
  strike_time 5 8 → strike_time 10 18 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_strike_theorem_l1706_170616


namespace NUMINAMATH_CALUDE_factorization_equality_l1706_170695

theorem factorization_equality (x : ℝ) : 
  x^2 * (x - 3) - 4 * (x - 3) = (x - 3) * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1706_170695


namespace NUMINAMATH_CALUDE_ball_travel_distance_l1706_170643

/-- The distance traveled by the center of a ball on a track with three semicircular arcs -/
theorem ball_travel_distance (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : 
  ball_diameter = 8 →
  R₁ = 110 →
  R₂ = 70 →
  R₃ = 90 →
  (π / 2) * ((R₁ - ball_diameter / 2) + (R₂ + ball_diameter / 2) + (R₃ - ball_diameter / 2)) = 266 * π := by
  sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l1706_170643


namespace NUMINAMATH_CALUDE_subsets_common_element_probability_l1706_170654

def S : Finset ℕ := {1, 2, 3, 4}

theorem subsets_common_element_probability :
  let subsets := S.powerset
  let total_pairs := subsets.card * subsets.card
  let disjoint_pairs := (Finset.range 5).sum (λ k => (Nat.choose 4 k) * (2^(4 - k)))
  (total_pairs - disjoint_pairs) / total_pairs = 175 / 256 := by
sorry

end NUMINAMATH_CALUDE_subsets_common_element_probability_l1706_170654


namespace NUMINAMATH_CALUDE_complex_power_sum_l1706_170645

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^600 + 1/z^600 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1706_170645


namespace NUMINAMATH_CALUDE_fourth_root_difference_l1706_170601

theorem fourth_root_difference : (81 : ℝ) ^ (1/4) - (1296 : ℝ) ^ (1/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_difference_l1706_170601


namespace NUMINAMATH_CALUDE_missing_digit_is_one_l1706_170606

/-- Converts a number from base 3 to base 10 -/
def base3ToBase10 (digit1 digit2 : ℕ) : ℕ :=
  digit1 * 3 + digit2

/-- Converts a number from base 12 to base 10 -/
def base12ToBase10 (digit1 digit2 : ℕ) : ℕ :=
  digit1 * 12 + digit2

/-- The main theorem stating that the missing digit is 1 -/
theorem missing_digit_is_one :
  ∃ (triangle : ℕ), 
    triangle < 10 ∧ 
    base3ToBase10 5 triangle = base12ToBase10 triangle 4 ∧ 
    triangle = 1 := by
  sorry

#check missing_digit_is_one

end NUMINAMATH_CALUDE_missing_digit_is_one_l1706_170606


namespace NUMINAMATH_CALUDE_quadratic_equation_constant_l1706_170663

theorem quadratic_equation_constant (C : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    x₁ > x₂ ∧ 
    x₁ - x₂ = 5.5 ∧ 
    2 * x₁^2 + 5 * x₁ - C = 0 ∧ 
    2 * x₂^2 + 5 * x₂ - C = 0) → 
  C = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_constant_l1706_170663


namespace NUMINAMATH_CALUDE_savings_is_84_l1706_170646

/-- Represents the pricing structure and needs for a window purchase scenario -/
structure WindowPurchase where
  regular_price : ℕ  -- Regular price per window
  free_window_threshold : ℕ  -- Number of windows purchased to get one free
  bulk_discount_threshold : ℕ  -- Number of windows needed for bulk discount
  bulk_discount_rate : ℚ  -- Discount rate for bulk purchases
  alice_needs : ℕ  -- Number of windows Alice needs
  bob_needs : ℕ  -- Number of windows Bob needs

/-- Calculates the price for a given number of windows -/
def calculate_price (wp : WindowPurchase) (num_windows : ℕ) : ℚ :=
  let paid_windows := num_windows - (num_windows / wp.free_window_threshold)
  let base_price := (paid_windows * wp.regular_price : ℚ)
  if num_windows ≥ wp.bulk_discount_threshold
  then base_price * (1 - wp.bulk_discount_rate)
  else base_price

/-- Calculates the savings when purchasing windows together versus separately -/
def savings_difference (wp : WindowPurchase) : ℚ :=
  let separate_cost := calculate_price wp wp.alice_needs + calculate_price wp wp.bob_needs
  let combined_cost := calculate_price wp (wp.alice_needs + wp.bob_needs)
  (wp.alice_needs + wp.bob_needs) * wp.regular_price - separate_cost - 
  ((wp.alice_needs + wp.bob_needs) * wp.regular_price - combined_cost)

/-- The main theorem stating the savings difference -/
theorem savings_is_84 (wp : WindowPurchase) 
  (h1 : wp.regular_price = 120)
  (h2 : wp.free_window_threshold = 5)
  (h3 : wp.bulk_discount_threshold = 10)
  (h4 : wp.bulk_discount_rate = 1/10)
  (h5 : wp.alice_needs = 9)
  (h6 : wp.bob_needs = 11) : 
  savings_difference wp = 84 := by
  sorry

end NUMINAMATH_CALUDE_savings_is_84_l1706_170646


namespace NUMINAMATH_CALUDE_johnny_table_planks_l1706_170640

/-- The number of planks needed for a table's surface -/
def planks_for_surface (total_tables : ℕ) (total_planks : ℕ) (legs_per_table : ℕ) : ℕ :=
  (total_planks / total_tables) - legs_per_table

theorem johnny_table_planks 
  (total_tables : ℕ) 
  (total_planks : ℕ) 
  (legs_per_table : ℕ) 
  (h1 : total_tables = 5) 
  (h2 : total_planks = 45) 
  (h3 : legs_per_table = 4) : 
  planks_for_surface total_tables total_planks legs_per_table = 5 := by
sorry

end NUMINAMATH_CALUDE_johnny_table_planks_l1706_170640


namespace NUMINAMATH_CALUDE_diagonal_intersection_minimizes_sum_distances_l1706_170661

-- Define a quadrilateral in 2D space
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p q : Point) : ℝ := sorry

-- Function to find the intersection of two line segments
def intersectionPoint (p1 p2 q1 q2 : Point) : Point := sorry

-- Function to calculate the sum of distances from a point to all vertices of a quadrilateral
def sumDistances (quad : Quadrilateral) (p : Point) : ℝ := sorry

-- Theorem stating that the intersection of diagonals minimizes the sum of distances
theorem diagonal_intersection_minimizes_sum_distances (quad : Quadrilateral) :
  let M := intersectionPoint quad.A quad.C quad.B quad.D
  ∀ p : Point, sumDistances quad M ≤ sumDistances quad p :=
sorry

end NUMINAMATH_CALUDE_diagonal_intersection_minimizes_sum_distances_l1706_170661


namespace NUMINAMATH_CALUDE_cylinder_speed_squared_l1706_170677

/-- The acceleration due to gravity in m/s^2 -/
def g : ℝ := 9.8

/-- The height of the incline in meters -/
def h : ℝ := 3.0

/-- The speed of the cylinder at the bottom of the incline in m/s -/
def v : ℝ := sorry

theorem cylinder_speed_squared (m : ℝ) (m_pos : m > 0) :
  v^2 = 2 * g * h := by sorry

end NUMINAMATH_CALUDE_cylinder_speed_squared_l1706_170677


namespace NUMINAMATH_CALUDE_linda_candy_count_l1706_170623

def candy_problem (initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

theorem linda_candy_count : candy_problem 34 28 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_linda_candy_count_l1706_170623


namespace NUMINAMATH_CALUDE_amoeba_count_day_10_l1706_170638

def amoeba_count (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % 3 = 0 then (3 * amoeba_count (n - 1)) / 2
  else 2 * amoeba_count (n - 1)

theorem amoeba_count_day_10 : amoeba_count 10 = 432 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_day_10_l1706_170638


namespace NUMINAMATH_CALUDE_concurrent_diagonals_l1706_170627

/-- Regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- The spiral similarity center of two regular polygons -/
def spiralSimilarityCenter (p q : RegularPolygon 100) : ℝ × ℝ :=
  sorry

/-- Intersection point of two line segments -/
def intersectionPoint (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The R points defined by the intersection of sides of p and q -/
def R (p q : RegularPolygon 100) (i : Fin 100) : ℝ × ℝ :=
  intersectionPoint (p.vertices i) (p.vertices (i+1)) (q.vertices i) (q.vertices (i+1))

/-- A diagonal of the 200-gon formed by R points -/
def diagonal (p q : RegularPolygon 100) (i : Fin 50) : Set (ℝ × ℝ) :=
  sorry

theorem concurrent_diagonals (p q : RegularPolygon 100) :
  ∃ (center : ℝ × ℝ), ∀ (i : Fin 50), center ∈ diagonal p q i := by
  sorry

end NUMINAMATH_CALUDE_concurrent_diagonals_l1706_170627


namespace NUMINAMATH_CALUDE_inequality_proof_l1706_170610

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 
  (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1706_170610


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1706_170698

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![9, -3; 2, 2]
def C : Matrix (Fin 2) (Fin 2) ℝ := !![27, -9; 32, -16]

theorem matrix_multiplication_result : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1706_170698


namespace NUMINAMATH_CALUDE_product_and_sum_with_reciprocal_bounds_l1706_170687

/-- Given positive real numbers a and b that sum to 1, this theorem proves
    the range of their product and the minimum value of their product plus its reciprocal. -/
theorem product_and_sum_with_reciprocal_bounds (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
    (0 < a * b ∧ a * b ≤ 1/4) ∧
    (∀ x : ℝ, x > 0 → x ≤ 1/4 → a * b + 1 / (a * b) ≤ x + 1 / x) ∧
    a * b + 1 / (a * b) = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_with_reciprocal_bounds_l1706_170687


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1706_170614

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3015 * x + 3020 * y = 3024)
  (eq2 : 3017 * x + 3022 * y = 3026) : 
  x - y = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1706_170614


namespace NUMINAMATH_CALUDE_flu_infection_rate_l1706_170692

theorem flu_infection_rate : ∃ x : ℝ, x > 0 ∧ 1 + x + x^2 = 121 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l1706_170692


namespace NUMINAMATH_CALUDE_no_14_consecutive_divisible_by_primes_lt_13_exist_21_consecutive_divisible_by_primes_lt_17_l1706_170607

/-- The set of primes less than 13 -/
def primes_lt_13 : Set Nat := {p | Nat.Prime p ∧ p < 13}

/-- The set of primes less than 17 -/
def primes_lt_17 : Set Nat := {p | Nat.Prime p ∧ p < 17}

/-- A function that checks if a number is divisible by any prime in a given set -/
def divisible_by_any_prime (n : Nat) (primes : Set Nat) : Prop :=
  ∃ p ∈ primes, n % p = 0

/-- Theorem stating that there do not exist 14 consecutive positive integers
    each divisible by a prime less than 13 -/
theorem no_14_consecutive_divisible_by_primes_lt_13 :
  ¬ ∃ start : Nat, ∀ i ∈ Finset.range 14, divisible_by_any_prime (start + i) primes_lt_13 :=
sorry

/-- Theorem stating that there exist 21 consecutive positive integers
    each divisible by a prime less than 17 -/
theorem exist_21_consecutive_divisible_by_primes_lt_17 :
  ∃ start : Nat, ∀ i ∈ Finset.range 21, divisible_by_any_prime (start + i) primes_lt_17 :=
sorry

end NUMINAMATH_CALUDE_no_14_consecutive_divisible_by_primes_lt_13_exist_21_consecutive_divisible_by_primes_lt_17_l1706_170607


namespace NUMINAMATH_CALUDE_solution_exists_l1706_170605

theorem solution_exists : ∃ a : ℝ, (-6) * (a^2) = 3 * (4*a + 2) ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l1706_170605


namespace NUMINAMATH_CALUDE_iterative_average_difference_l1706_170659

def iterative_average (seq : List ℚ) : ℚ :=
  seq.foldl (λ acc x => (acc + x) / 2) (seq.head!)

def max_sequence : List ℚ := [6, 5, 4, 3, 2, 1]
def min_sequence : List ℚ := [1, 2, 3, 4, 5, 6]

theorem iterative_average_difference :
  iterative_average max_sequence - iterative_average min_sequence = 1 := by
  sorry

end NUMINAMATH_CALUDE_iterative_average_difference_l1706_170659


namespace NUMINAMATH_CALUDE_equation_solution_l1706_170660

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = 13/4 ∧ 
  (∀ x : ℝ, x - 3 = 4 * (x - 3)^2 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1706_170660


namespace NUMINAMATH_CALUDE_subtraction_as_addition_of_negative_l1706_170662

theorem subtraction_as_addition_of_negative (a b : ℚ) : a - b = a + (-b) := by sorry

end NUMINAMATH_CALUDE_subtraction_as_addition_of_negative_l1706_170662


namespace NUMINAMATH_CALUDE_focus_of_parabola_l1706_170690

/-- A parabola is defined by the equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola is a point on its axis of symmetry -/
def Focus (p : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem: The focus of the parabola y^2 = 4x is at the point (2, 0) -/
theorem focus_of_parabola :
  Focus Parabola = (2, 0) := by sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l1706_170690


namespace NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l1706_170604

theorem not_product_of_consecutive_numbers (n : ℕ) :
  ¬ ∃ k : ℕ, 2 * (6^n + 1) = k * (k + 1) := by
sorry

end NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l1706_170604


namespace NUMINAMATH_CALUDE_initial_men_count_l1706_170615

/-- The initial number of men working on the construction job -/
def initial_men : ℕ := sorry

/-- The total amount of work to be done -/
def total_work : ℝ := sorry

/-- Half of the job is finished in 15 days with the initial number of men -/
axiom half_job_rate : (initial_men : ℝ) * 15 = total_work / 2

/-- The remaining job is completed in 25 days with two fewer men -/
axiom remaining_job_rate : (initial_men - 2 : ℝ) * 25 = total_work / 2

/-- The initial number of men is 5 -/
theorem initial_men_count : initial_men = 5 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_l1706_170615


namespace NUMINAMATH_CALUDE_stimulus_savings_amount_l1706_170619

def stimulus_distribution (initial_amount : ℚ) : ℚ :=
  let wife_share := (2 / 5) * initial_amount
  let after_wife := initial_amount - wife_share
  let first_son_share := (2 / 5) * after_wife
  let after_first_son := after_wife - first_son_share
  let second_son_share := (40 / 100) * after_first_son
  after_first_son - second_son_share

theorem stimulus_savings_amount :
  stimulus_distribution 2000 = 432 := by
  sorry

end NUMINAMATH_CALUDE_stimulus_savings_amount_l1706_170619


namespace NUMINAMATH_CALUDE_no_integer_roots_for_both_equations_l1706_170620

theorem no_integer_roots_for_both_equations :
  ¬∃ (b c : ℝ), 
    (∃ (p q : ℤ), p ≠ q ∧ (p : ℝ)^2 + b*(p : ℝ) + c = 0 ∧ (q : ℝ)^2 + b*(q : ℝ) + c = 0) ∧
    (∃ (r s : ℤ), r ≠ s ∧ 2*(r : ℝ)^2 + (b+1)*(r : ℝ) + (c+1) = 0 ∧ 2*(s : ℝ)^2 + (b+1)*(s : ℝ) + (c+1) = 0) :=
sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_both_equations_l1706_170620


namespace NUMINAMATH_CALUDE_problem_solution_l1706_170625

def f (k : ℝ) (x : ℝ) : ℝ := k - |x - 3|

theorem problem_solution (k a b c : ℝ) :
  (∀ x, f k (x + 3) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (1 / (k * a) + 1 / (2 * k * b) + 1 / (3 * k * c) = 1) →
  (k = 1 ∧ 1/9 * a + 2/9 * b + 3/9 * c ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1706_170625


namespace NUMINAMATH_CALUDE_solution_set_product_l1706_170602

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (a b : ℝ)
variable (f g : ℝ → ℝ)

/-- The solution set for f(x) > 0 -/
def SolutionSetF : Set ℝ := Set.Ioo (a^2) b

/-- The solution set for g(x) > 0 -/
def SolutionSetG : Set ℝ := Set.Ioo (a^2/2) (b/2)

/-- The conditions given in the problem -/
structure ProblemConditions where
  f_odd : OddFunction f
  g_odd : OddFunction g
  f_solution : SolutionSetF a b = {x | f x > 0}
  g_solution : SolutionSetG a b = {x | g x > 0}
  b_gt_2a_squared : b > 2 * (a^2)

/-- The theorem to be proved -/
theorem solution_set_product (h : ProblemConditions a b f g) :
  {x | f x * g x > 0} = Set.Ioo (-b/2) (-a^2) ∪ Set.Ioo (a^2) (b/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_product_l1706_170602


namespace NUMINAMATH_CALUDE_leah_bought_three_boxes_l1706_170621

/-- The number of boxes of birdseed Leah bought -/
def boxes_bought (existing_boxes weeks parrot_consumption cockatiel_consumption box_content : ℕ) : ℕ :=
  let total_consumption := weeks * (parrot_consumption + cockatiel_consumption)
  let total_boxes_needed := (total_consumption + box_content - 1) / box_content
  total_boxes_needed - existing_boxes

/-- Theorem stating that Leah bought 3 boxes of birdseed -/
theorem leah_bought_three_boxes :
  boxes_bought 5 12 100 50 225 = 3 := by sorry

end NUMINAMATH_CALUDE_leah_bought_three_boxes_l1706_170621


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l1706_170617

/-- The cost price of one meter of cloth given the selling price and profit per meter -/
theorem cost_price_per_meter
  (cloth_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : cloth_length = 75)
  (h2 : selling_price = 4950)
  (h3 : profit_per_meter = 15) :
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 51 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l1706_170617


namespace NUMINAMATH_CALUDE_monotonic_sine_phi_range_l1706_170650

theorem monotonic_sine_phi_range (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = -2 * Real.sin (2 * x + φ)) →
  (|φ| < π) →
  (∀ x ∈ Set.Ioo (π / 5) ((5 / 8) * π), StrictMono f) →
  φ ∈ Set.Ioo (π / 10) (π / 4) := by
sorry

end NUMINAMATH_CALUDE_monotonic_sine_phi_range_l1706_170650


namespace NUMINAMATH_CALUDE_two_distinct_roots_l1706_170631

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m + 1)*x + m^2 - 1

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (2*m + 1)^2 - 4*(m^2 - 1)

/-- Theorem: The quadratic function f has two distinct real roots if and only if m > -5/4 -/
theorem two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m > -5/4 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l1706_170631


namespace NUMINAMATH_CALUDE_eleven_skill_players_wait_l1706_170655

/-- Represents the football team's water consumption during practice --/
structure FootballTeamWater where
  linemen : ℕ
  skillPlayers : ℕ
  linemenConsumption : ℕ
  skillPlayerConsumption : ℕ
  initialWater : ℕ
  refillWater : ℕ

/-- Calculates the number of skill position players who must wait for a drink --/
def waitingSkillPlayers (team : FootballTeamWater) : ℕ :=
  let linemenTotal := team.linemen * team.linemenConsumption
  let remainingWater := team.initialWater + team.refillWater - linemenTotal
  let drinkingSkillPlayers := remainingWater / team.skillPlayerConsumption
  team.skillPlayers - min team.skillPlayers drinkingSkillPlayers

/-- Theorem stating that 11 skill position players must wait for a drink --/
theorem eleven_skill_players_wait (team : FootballTeamWater) 
  (h1 : team.linemen = 20)
  (h2 : team.skillPlayers = 18)
  (h3 : team.linemenConsumption = 12)
  (h4 : team.skillPlayerConsumption = 10)
  (h5 : team.initialWater = 190)
  (h6 : team.refillWater = 120) :
  waitingSkillPlayers team = 11 := by
  sorry

#eval waitingSkillPlayers {
  linemen := 20,
  skillPlayers := 18,
  linemenConsumption := 12,
  skillPlayerConsumption := 10,
  initialWater := 190,
  refillWater := 120
}

end NUMINAMATH_CALUDE_eleven_skill_players_wait_l1706_170655


namespace NUMINAMATH_CALUDE_fraction_simplification_l1706_170612

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) : (3 * a) / (3 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1706_170612


namespace NUMINAMATH_CALUDE_math_science_students_l1706_170629

theorem math_science_students :
  -- Total number of students
  ∀ (total : ℕ),
  -- Number of students in Science class
  ∀ (science : ℕ),
  -- Number of students in Math class
  ∀ (math : ℕ),
  -- Conditions
  total = 30 →
  science + math + 2 = total →
  math + 2 = 3 * (science + 2) →
  -- Conclusion
  math - science = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_math_science_students_l1706_170629


namespace NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l1706_170680

theorem max_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
    a > 0 →
    d > 0 →
    a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60 →
    ∀ (b e : ℕ),
      b > 0 →
      e > 0 →
      b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 60 →
      (a + d) ≤ 7 ∧
      (∃ a d : ℕ, a > 0 ∧ d > 0 ∧ a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60 ∧ a + d = 7) :=
by sorry

#check max_second_term_arithmetic_sequence

end NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l1706_170680


namespace NUMINAMATH_CALUDE_derivative_constant_sine_l1706_170688

theorem derivative_constant_sine (y : ℝ → ℝ) (h : y = λ _ => Real.sin (π / 3)) :
  deriv y = λ _ => 0 := by sorry

end NUMINAMATH_CALUDE_derivative_constant_sine_l1706_170688


namespace NUMINAMATH_CALUDE_candle_burning_theorem_l1706_170630

theorem candle_burning_theorem (ℓ : ℝ) (h : ℓ > 0) : 
  let t := 180 -- 3 hours in minutes
  let f := λ x : ℝ => ℓ * (1 - x / 240) -- stub length of 4-hour candle
  let g := λ x : ℝ => ℓ * (1 - x / 360) -- stub length of 6-hour candle
  g t = 3 * f t := by sorry

end NUMINAMATH_CALUDE_candle_burning_theorem_l1706_170630


namespace NUMINAMATH_CALUDE_shoeing_problem_solution_l1706_170675

/-- Represents the shoeing problem with given parameters -/
structure ShoeingProblem where
  blacksmiths : ℕ
  horses : ℕ
  time_per_hoof : ℕ
  hooves_per_horse : ℕ
  min_hooves_on_ground : ℕ

/-- Calculates the minimum time required to shoe all horses -/
def minimum_shoeing_time (problem : ShoeingProblem) : ℕ :=
  let total_hooves := problem.horses * problem.hooves_per_horse
  let total_time := total_hooves * problem.time_per_hoof
  let time_per_blacksmith := total_time / problem.blacksmiths
  let horses_at_once := problem.blacksmiths / problem.hooves_per_horse
  let sets_needed := (problem.horses + horses_at_once - 1) / horses_at_once
  sets_needed * time_per_blacksmith

/-- Theorem stating that for the given problem, the minimum shoeing time is 125 minutes -/
theorem shoeing_problem_solution :
  let problem : ShoeingProblem := {
    blacksmiths := 48,
    horses := 60,
    time_per_hoof := 5,
    hooves_per_horse := 4,
    min_hooves_on_ground := 3
  }
  minimum_shoeing_time problem = 125 := by
  sorry

#eval minimum_shoeing_time {
  blacksmiths := 48,
  horses := 60,
  time_per_hoof := 5,
  hooves_per_horse := 4,
  min_hooves_on_ground := 3
}

end NUMINAMATH_CALUDE_shoeing_problem_solution_l1706_170675


namespace NUMINAMATH_CALUDE_smoking_students_not_hospitalized_l1706_170628

theorem smoking_students_not_hospitalized 
  (total_students : ℕ) 
  (smoking_percentage : ℚ) 
  (hospitalized_percentage : ℚ) 
  (h1 : total_students = 300)
  (h2 : smoking_percentage = 40 / 100)
  (h3 : hospitalized_percentage = 70 / 100) :
  ⌊total_students * smoking_percentage - 
   (total_students * smoking_percentage * hospitalized_percentage)⌋ = 36 := by
sorry

end NUMINAMATH_CALUDE_smoking_students_not_hospitalized_l1706_170628


namespace NUMINAMATH_CALUDE_beth_coin_sale_l1706_170603

theorem beth_coin_sale (initial_coins : ℕ) (gift_coins : ℕ) : 
  initial_coins = 125 → gift_coins = 35 → 
  (initial_coins + gift_coins) / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_beth_coin_sale_l1706_170603


namespace NUMINAMATH_CALUDE_stratified_sample_young_employees_l1706_170608

/-- Calculates the number of employees to be drawn from a specific age group in a stratified sample. -/
def stratifiedSampleSize (totalEmployees : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupSize * sampleSize) / totalEmployees

/-- Proves that the number of employees no older than 45 to be drawn in a stratified sample is 15. -/
theorem stratified_sample_young_employees :
  let totalEmployees : ℕ := 200
  let youngEmployees : ℕ := 120
  let sampleSize : ℕ := 25
  stratifiedSampleSize totalEmployees youngEmployees sampleSize = 15 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_young_employees_l1706_170608


namespace NUMINAMATH_CALUDE_wednesday_spending_ratio_is_three_to_eight_l1706_170699

/-- Represents Bob's spending pattern and final amount --/
structure BobsSpending where
  initial : ℚ
  monday_spent : ℚ
  tuesday_spent : ℚ
  final : ℚ

/-- Calculates the ratio of Wednesday's spending to Tuesday's remaining amount --/
def wednesdaySpendingRatio (s : BobsSpending) : ℚ × ℚ :=
  let monday_left := s.initial - s.monday_spent
  let tuesday_left := monday_left - s.tuesday_spent
  let wednesday_spent := tuesday_left - s.final
  (wednesday_spent, tuesday_left)

/-- Theorem stating the ratio of Wednesday's spending to Tuesday's remaining amount --/
theorem wednesday_spending_ratio_is_three_to_eight (s : BobsSpending) 
  (h1 : s.initial = 80)
  (h2 : s.monday_spent = s.initial / 2)
  (h3 : s.tuesday_spent = (s.initial - s.monday_spent) / 5)
  (h4 : s.final = 20) :
  wednesdaySpendingRatio s = (3, 8) := by
  sorry


end NUMINAMATH_CALUDE_wednesday_spending_ratio_is_three_to_eight_l1706_170699


namespace NUMINAMATH_CALUDE_inequality_proof_l1706_170651

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1706_170651


namespace NUMINAMATH_CALUDE_power_mod_23_l1706_170653

theorem power_mod_23 : 17^1988 % 23 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_23_l1706_170653


namespace NUMINAMATH_CALUDE_max_representations_l1706_170639

/-- The number of colors used for coding -/
def n : ℕ := 5

/-- The number of ways to choose a single color -/
def single_color_choices (m : ℕ) : ℕ := m

/-- The number of ways to choose a pair of two different colors -/
def color_pair_choices (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The total number of unique representations -/
def total_representations (m : ℕ) : ℕ := single_color_choices m + color_pair_choices m

/-- Theorem: Given 5 colors, the maximum number of unique representations is 15 -/
theorem max_representations : total_representations n = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_representations_l1706_170639


namespace NUMINAMATH_CALUDE_capacitance_calculation_l1706_170685

theorem capacitance_calculation (U ε Q : ℝ) (hε : ε > 0) (hU : U ≠ 0) :
  ∃ C : ℝ, C > 0 ∧ C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_capacitance_calculation_l1706_170685


namespace NUMINAMATH_CALUDE_f_min_at_neg_one_l1706_170686

/-- The quadratic function f(x) = 3x^2 + 6x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

/-- The theorem stating that the minimum of f occurs at x = -1 -/
theorem f_min_at_neg_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -1 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_one_l1706_170686


namespace NUMINAMATH_CALUDE_turkey_roasting_time_l1706_170671

/-- Represents the turkey roasting problem --/
structure TurkeyRoasting where
  num_turkeys : ℕ
  weight_per_turkey : ℕ
  start_time : ℕ
  end_time : ℕ

/-- Calculates the roasting time per pound --/
def roasting_time_per_pound (tr : TurkeyRoasting) : ℚ :=
  let total_time := tr.end_time - tr.start_time
  let total_weight := tr.num_turkeys * tr.weight_per_turkey
  (total_time : ℚ) / total_weight

/-- The main theorem stating that the roasting time per pound is 15 minutes --/
theorem turkey_roasting_time (tr : TurkeyRoasting)
  (h1 : tr.num_turkeys = 2)
  (h2 : tr.weight_per_turkey = 16)
  (h3 : tr.start_time = 10 * 60)  -- 10:00 am in minutes
  (h4 : tr.end_time = 18 * 60)    -- 6:00 pm in minutes
  : roasting_time_per_pound tr = 15 := by
  sorry

#eval roasting_time_per_pound { num_turkeys := 2, weight_per_turkey := 16, start_time := 10 * 60, end_time := 18 * 60 }

end NUMINAMATH_CALUDE_turkey_roasting_time_l1706_170671


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1706_170697

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ n : ℕ, n > 2 ∧ D = n * (n - 3) / 2 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1706_170697


namespace NUMINAMATH_CALUDE_tangent_slope_three_points_point_on_curve_l1706_170632

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_slope_three_points (x : ℝ) :
  (f' x = 3) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem point_on_curve (x : ℝ) :
  (f' x = 3 ∧ f x = x^3) ↔ ((x = 1 ∧ f x = 1) ∨ (x = -1 ∧ f x = -1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_three_points_point_on_curve_l1706_170632


namespace NUMINAMATH_CALUDE_mixture_theorem_l1706_170611

/-- Represents a chemical solution with a given percentage of chemical a -/
structure Solution :=
  (percent_a : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution_x : Solution)
  (solution_y : Solution)
  (percent_x : ℝ)

/-- Calculates the percentage of chemical a in a mixture -/
def percent_a_in_mixture (m : Mixture) : ℝ :=
  m.solution_x.percent_a * m.percent_x + m.solution_y.percent_a * (1 - m.percent_x)

theorem mixture_theorem :
  let x : Solution := ⟨0.30⟩
  let y : Solution := ⟨0.40⟩
  let mixture : Mixture := ⟨x, y, 0.80⟩
  percent_a_in_mixture mixture = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_mixture_theorem_l1706_170611


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l1706_170681

theorem partial_fraction_decomposition_A (x A B C : ℝ) :
  (1 : ℝ) / (x^3 - x^2 - 17*x + 45) = A / (x + 5) + B / (x - 3) + C / (x + 3) →
  A = (1 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l1706_170681


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l1706_170637

theorem mary_baseball_cards (X : ℕ) : 
  X - 8 + 26 + 40 = 84 → X = 26 := by
sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l1706_170637


namespace NUMINAMATH_CALUDE_mosquito_shadow_speed_l1706_170649

/-- The speed of a mosquito's shadow across the bottom of a water body -/
theorem mosquito_shadow_speed 
  (v : Real) 
  (h : Real) 
  (t : Real) 
  (cos_incidence : Real) : 
  v = 1 → 
  h = 3 → 
  t = 5 → 
  cos_incidence = 0.6 → 
  ∃ (shadow_speed : Real), 
    shadow_speed = 1.6 ∨ shadow_speed = 0 :=
by sorry

end NUMINAMATH_CALUDE_mosquito_shadow_speed_l1706_170649


namespace NUMINAMATH_CALUDE_tanner_savings_l1706_170642

/-- The amount of money Tanner is left with after saving and spending -/
def money_left (september_savings october_savings november_savings video_game_cost : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - video_game_cost

/-- Theorem stating that Tanner is left with $41 -/
theorem tanner_savings : 
  money_left 17 48 25 49 = 41 := by
  sorry

end NUMINAMATH_CALUDE_tanner_savings_l1706_170642


namespace NUMINAMATH_CALUDE_log_equation_solution_l1706_170622

theorem log_equation_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx2 : x ≠ 2) :
  (Real.log x + Real.log y = Real.log (x + 2*y)) ↔ (y = x / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1706_170622


namespace NUMINAMATH_CALUDE_count_numbers_divisible_by_33_l1706_170648

def is_valid_number (x y : ℕ) : Prop :=
  x ≤ 9 ∧ y ≤ 9

def number_value (x y : ℕ) : ℕ :=
  2007000000 + x * 100000 + 2008 + y

theorem count_numbers_divisible_by_33 :
  ∃ (S : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ S ↔ 
      is_valid_number p.1 p.2 ∧ 
      (number_value p.1 p.2) % 33 = 0) ∧
    Finset.card S = 3 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_divisible_by_33_l1706_170648


namespace NUMINAMATH_CALUDE_range_of_a_l1706_170667

-- Define a decreasing function f on the real numbers
def f : ℝ → ℝ := sorry

-- State that f is decreasing
axiom f_decreasing : ∀ x y, x < y → f x > f y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (f (1 - a) < f (2 * a - 5)) ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1706_170667


namespace NUMINAMATH_CALUDE_congruence_solution_l1706_170674

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1706_170674


namespace NUMINAMATH_CALUDE_unique_solution_inequality_system_l1706_170670

theorem unique_solution_inequality_system :
  ∃! (a b : ℤ), 11 > 2 * a - b ∧
                25 > 2 * b - a ∧
                42 < 3 * b - a ∧
                46 < 2 * a + b :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_system_l1706_170670


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l1706_170647

theorem lcm_gcd_problem (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 18) :
  ∃ (a' c' : ℕ), Nat.lcm a' b = 20 ∧ Nat.lcm b c' = 18 ∧
    ∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 18 →
      Nat.lcm a' c' + 2 * Nat.gcd a' c' ≤ Nat.lcm x y + 2 * Nat.gcd x y :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l1706_170647


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1706_170682

theorem triangle_angle_A (a b : ℝ) (A B : ℝ) (hb : b = 2 * Real.sqrt 3) (ha : a = 2) (hB : B = π / 3) :
  A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l1706_170682


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l1706_170668

theorem cos_five_pi_thirds_plus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (5 * π / 3 + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l1706_170668


namespace NUMINAMATH_CALUDE_function_evaluation_l1706_170626

theorem function_evaluation :
  ((-1)^4 + (-1)^3 + 1) / ((-1)^2 + 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l1706_170626


namespace NUMINAMATH_CALUDE_total_distance_traveled_l1706_170633

/-- Converts kilometers per hour to miles per hour -/
def kph_to_mph (kph : ℝ) : ℝ := kph * 0.621371

/-- Calculates distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem total_distance_traveled :
  let walk_time : ℝ := 90 / 60  -- 90 minutes in hours
  let walk_speed : ℝ := 3       -- 3 mph
  let rest_time : ℝ := 15 / 60  -- 15 minutes in hours
  let cycle_time : ℝ := 45 / 60 -- 45 minutes in hours
  let cycle_speed : ℝ := kph_to_mph 20 -- 20 kph converted to mph
  let total_time : ℝ := 2.5     -- 2 hours and 30 minutes
  let walk_distance := distance walk_speed walk_time
  let cycle_distance := distance cycle_speed cycle_time
  let total_distance := walk_distance + cycle_distance
  ∃ ε > 0, |total_distance - 13.82| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l1706_170633


namespace NUMINAMATH_CALUDE_area_of_triangle_from_square_centers_l1706_170624

/-- The area of a triangle formed by the centers of three adjacent squares surrounding a central square --/
theorem area_of_triangle_from_square_centers (central_square_side : ℝ) (h : central_square_side = 2) :
  let outer_square_diagonal := central_square_side * Real.sqrt 2
  let triangle_side := outer_square_diagonal
  let triangle_area := Real.sqrt 3 / 4 * triangle_side^2
  triangle_area = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_from_square_centers_l1706_170624


namespace NUMINAMATH_CALUDE_polynomial_zero_at_sqrt_three_halves_l1706_170634

def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*x^2*y + b₈*x*y^2 + b₉*y^3

theorem polynomial_zero_at_sqrt_three_halves 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) 
  (h₁ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0)
  (h₂ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0)
  (h₃ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0)
  (h₄ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0)
  (h₅ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0)
  (h₆ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0)
  (h₇ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 (-1) = 0)
  (h₈ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 1 = 0) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (Real.sqrt (3/2)) (Real.sqrt (3/2)) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_zero_at_sqrt_three_halves_l1706_170634


namespace NUMINAMATH_CALUDE_max_sum_of_distinct_factors_l1706_170666

theorem max_sum_of_distinct_factors (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b^2 * c^3 = 1350 →
  ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    x * y^2 * z^3 = 1350 →
    a + b + c ≥ x + y + z :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_distinct_factors_l1706_170666


namespace NUMINAMATH_CALUDE_mitch_family_milk_consumption_l1706_170694

/-- The amount of regular milk consumed by Mitch's family in 1 week -/
def regular_milk : ℚ := 1/2

/-- The amount of soy milk consumed by Mitch's family in 1 week -/
def soy_milk : ℚ := 1/10

/-- The total amount of milk consumed by Mitch's family in 1 week -/
def total_milk : ℚ := regular_milk + soy_milk

theorem mitch_family_milk_consumption :
  total_milk = 3/5 := by sorry

end NUMINAMATH_CALUDE_mitch_family_milk_consumption_l1706_170694


namespace NUMINAMATH_CALUDE_no_integer_solution_l1706_170673

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 8 * x + 3 * y^2 = 5 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1706_170673


namespace NUMINAMATH_CALUDE_mixed_fraction_calculation_l1706_170636

theorem mixed_fraction_calculation :
  (13/4 - 13/5 + 21/4 + (-42/5) : ℚ) = -5/2 ∧
  (-(3^2) - (-5 + 3) + 27 / (-3) * (1/3) : ℚ) = -10 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_calculation_l1706_170636


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_min_value_attained_l1706_170691

/-- Given two parallel vectors a and b, where a = (1,3) and b = (x,1-y),
    and x and y are positive real numbers, 
    the minimum value of 3/x + 1/y is 16 -/
theorem min_value_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → (1 : ℝ) / x = 3 / (1 - y) → (3 / x + 1 / y) ≥ 16 := by
  sorry

/-- The minimum value is attained when 3/x + 1/y = 16 -/
theorem min_value_attained (x y : ℝ) : 
  x > 0 → y > 0 → (1 : ℝ) / x = 3 / (1 - y) → 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (1 : ℝ) / x₀ = 3 / (1 - y₀) ∧ 3 / x₀ + 1 / y₀ = 16) := by
  sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_min_value_attained_l1706_170691
