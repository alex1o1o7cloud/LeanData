import Mathlib

namespace NUMINAMATH_CALUDE_rubber_band_difference_l1702_170231

theorem rubber_band_difference (total : ℕ) (aira_initial : ℕ) (samantha_extra : ℕ) (equal_share : ℕ)
  (h1 : total = 18)
  (h2 : aira_initial = 4)
  (h3 : samantha_extra = 5)
  (h4 : equal_share = 6) :
  let samantha_initial := aira_initial + samantha_extra
  let joe_initial := total - samantha_initial - aira_initial
  joe_initial - aira_initial = 1 := by sorry

end NUMINAMATH_CALUDE_rubber_band_difference_l1702_170231


namespace NUMINAMATH_CALUDE_remainder_theorem_application_l1702_170228

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^6 + E * x^4 + F * x^2 + 5
  (q 2 = 17) → (q (-2) = 17) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_application_l1702_170228


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1702_170294

theorem intersection_of_sets :
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {1, 3}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1702_170294


namespace NUMINAMATH_CALUDE_max_value_expression_l1702_170266

theorem max_value_expression (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → 2*a*m*c + a*m + m*c + c*a ≤ 2*A*M*C + A*M + M*C + C*A) →
  2*A*M*C + A*M + M*C + C*A = 325 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1702_170266


namespace NUMINAMATH_CALUDE_power_of_two_equations_l1702_170293

theorem power_of_two_equations :
  (∃! n x : ℕ+, 2^(n : ℕ) + 1 = (x : ℕ)^2 ∧ n = 3 ∧ x = 3) ∧
  (∃! n x : ℕ+, 2^(n : ℕ) = (x : ℕ)^2 + 1 ∧ n = 1 ∧ x = 1) := by
  sorry

#check power_of_two_equations

end NUMINAMATH_CALUDE_power_of_two_equations_l1702_170293


namespace NUMINAMATH_CALUDE_field_area_in_acres_l1702_170215

-- Define the field dimensions
def field_length : ℕ := 30
def width_plus_diagonal : ℕ := 50

-- Define the conversion rate
def square_steps_per_acre : ℕ := 240

-- Theorem statement
theorem field_area_in_acres :
  ∃ (width : ℕ),
    width^2 + field_length^2 = (width_plus_diagonal - width)^2 ∧
    (field_length * width) / square_steps_per_acre = 2 :=
by sorry

end NUMINAMATH_CALUDE_field_area_in_acres_l1702_170215


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l1702_170288

theorem dvd_pack_cost (total_cost : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) :
  total_cost = 2673 →
  num_packs = 33 →
  cost_per_pack = total_cost / num_packs →
  cost_per_pack = 81 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l1702_170288


namespace NUMINAMATH_CALUDE_floor_nested_expression_l1702_170237

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem floor_nested_expression : floor (-2.3 + floor 1.6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_nested_expression_l1702_170237


namespace NUMINAMATH_CALUDE_max_value_of_f_l1702_170229

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2

-- Define the interval
def interval : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 4 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1702_170229


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1702_170286

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of 'a' for which the given lines are parallel -/
theorem parallel_lines_a_value :
  (∀ x y : ℝ, 3 * y + 6 * a = 9 * x ↔ y - 2 = (2 * a - 3) * x) → a = 3 := by
  sorry

#check parallel_lines_a_value

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1702_170286


namespace NUMINAMATH_CALUDE_dealership_sales_theorem_l1702_170220

/-- Represents the ratio of trucks to minivans sold -/
def truck_to_minivan_ratio : ℚ := 5 / 3

/-- Number of trucks expected to be sold -/
def expected_trucks : ℕ := 45

/-- Price of each truck in dollars -/
def truck_price : ℕ := 25000

/-- Price of each minivan in dollars -/
def minivan_price : ℕ := 20000

/-- Calculates the expected number of minivans to be sold -/
def expected_minivans : ℕ := (expected_trucks * 3) / 5

/-- Calculates the total revenue from truck and minivan sales -/
def total_revenue : ℕ := expected_trucks * truck_price + expected_minivans * minivan_price

theorem dealership_sales_theorem :
  expected_minivans = 27 ∧ total_revenue = 1665000 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_theorem_l1702_170220


namespace NUMINAMATH_CALUDE_total_blocks_l1702_170227

def num_boxes : ℕ := 2
def blocks_per_box : ℕ := 6

theorem total_blocks : num_boxes * blocks_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_l1702_170227


namespace NUMINAMATH_CALUDE_rectangle_width_l1702_170257

theorem rectangle_width (width length perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 288 →
  width = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1702_170257


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1702_170239

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 9/16 -/
theorem max_y_coordinate_sin_3theta : 
  let r : ℝ → ℝ := λ θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 9/16 :=
by
  sorry


end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1702_170239


namespace NUMINAMATH_CALUDE_prime_sum_problem_l1702_170200

theorem prime_sum_problem (p q r s : ℕ) : 
  Prime p → Prime q → Prime r → Prime s →
  p < q → q < r → r < s →
  p * q * r * s + 1 = 4^(p + q) →
  r + s = 274 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l1702_170200


namespace NUMINAMATH_CALUDE_friend_distribution_l1702_170283

theorem friend_distribution (F : ℕ) (h1 : F > 0) : 
  (100 / F : ℚ) - (100 / (F + 5) : ℚ) = 1 → F = 20 := by
  sorry

end NUMINAMATH_CALUDE_friend_distribution_l1702_170283


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_difference_one_l1702_170295

/-- The quadratic equation x^2 + (m+3)x + m+2 = 0 -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (m+3)*x + m+2

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (m+3)^2 - 4*(m+2)

theorem quadratic_always_real_roots (m : ℝ) :
  discriminant m ≥ 0 := by sorry

theorem roots_difference_one (m : ℝ) :
  (∃ a b, quadratic m a = 0 ∧ quadratic m b = 0 ∧ |a - b| = 1) →
  (m = -2 ∨ m = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_difference_one_l1702_170295


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l1702_170280

theorem ceiling_floor_calculation : 
  ⌈(12 / 5 : ℚ) * (((-19 : ℚ) / 4) - 3)⌉ - ⌊(12 / 5 : ℚ) * ⌊(-19 : ℚ) / 4⌋⌋ = -6 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l1702_170280


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l1702_170279

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that if a_2 and a_10 of an arithmetic sequence are roots of x^2 + 12x - 8 = 0, then a_6 = -6 -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2)^2 + 12*(a 2) - 8 = 0 →
  (a 10)^2 + 12*(a 10) - 8 = 0 →
  a 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l1702_170279


namespace NUMINAMATH_CALUDE_fifty_percent_greater_than_88_l1702_170281

theorem fifty_percent_greater_than_88 (x : ℝ) : x = 88 * 1.5 → x = 132 := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_greater_than_88_l1702_170281


namespace NUMINAMATH_CALUDE_paperback_copies_sold_l1702_170290

theorem paperback_copies_sold (hardback_copies : ℕ) (total_copies : ℕ) : 
  hardback_copies = 36000 →
  total_copies = 440000 →
  ∃ paperback_copies : ℕ, 
    paperback_copies = 9 * hardback_copies ∧
    hardback_copies + paperback_copies = total_copies ∧
    paperback_copies = 360000 := by
  sorry

end NUMINAMATH_CALUDE_paperback_copies_sold_l1702_170290


namespace NUMINAMATH_CALUDE_optimal_bus_rental_l1702_170219

/-- Represents the rental problem for buses -/
structure BusRental where
  cost_a : ℕ  -- Cost of renting one bus A
  cost_b : ℕ  -- Cost of renting one bus B
  capacity_a : ℕ  -- Capacity of bus A
  capacity_b : ℕ  -- Capacity of bus B
  total_people : ℕ  -- Total number of people to transport
  total_buses : ℕ  -- Total number of buses to rent

/-- Calculates the total cost for a given number of buses A and B -/
def total_cost (br : BusRental) (num_a : ℕ) (num_b : ℕ) : ℕ :=
  num_a * br.cost_a + num_b * br.cost_b

/-- Calculates the total capacity for a given number of buses A and B -/
def total_capacity (br : BusRental) (num_a : ℕ) (num_b : ℕ) : ℕ :=
  num_a * br.capacity_a + num_b * br.capacity_b

/-- Theorem stating that renting 2 buses A and 6 buses B minimizes the cost -/
theorem optimal_bus_rental (br : BusRental) 
  (h1 : br.cost_a + br.cost_b = 500)
  (h2 : 2 * br.cost_a + 3 * br.cost_b = 1300)
  (h3 : br.capacity_a = 15)
  (h4 : br.capacity_b = 25)
  (h5 : br.total_people = 180)
  (h6 : br.total_buses = 8) :
  ∀ (num_a num_b : ℕ), 
    num_a + num_b = br.total_buses →
    total_capacity br num_a num_b ≥ br.total_people →
    total_cost br 2 6 ≤ total_cost br num_a num_b :=
sorry

end NUMINAMATH_CALUDE_optimal_bus_rental_l1702_170219


namespace NUMINAMATH_CALUDE_inequality_proof_1_inequality_proof_2_l1702_170271

theorem inequality_proof_1 (x : ℝ) : 
  abs (x + 2) + abs (x - 2) > 6 ↔ x < -3 ∨ x > 3 := by sorry

theorem inequality_proof_2 (x : ℝ) : 
  abs (2*x - 1) - abs (x - 3) > 5 ↔ x < -7 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_1_inequality_proof_2_l1702_170271


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1702_170275

theorem min_value_of_function (x : ℝ) (h : x > 0) : (x^2 + 1) / x ≥ 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : (x^2 + 1) / x = 2 ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1702_170275


namespace NUMINAMATH_CALUDE_tom_marble_combinations_l1702_170274

/-- Represents the number of marbles of each color -/
structure MarbleSet :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- Calculates the number of ways to choose 2 marbles from a given set -/
def chooseTwo (s : MarbleSet) : ℕ :=
  sorry

/-- Tom's marble set -/
def tomMarbles : MarbleSet :=
  { red := 1, blue := 1, green := 2, yellow := 3 }

theorem tom_marble_combinations :
  chooseTwo tomMarbles = 19 :=
sorry

end NUMINAMATH_CALUDE_tom_marble_combinations_l1702_170274


namespace NUMINAMATH_CALUDE_power_of_twenty_l1702_170258

theorem power_of_twenty : (20 : ℕ) ^ ((20 : ℕ) / 2) = 102400000000000000000 := by
  sorry

end NUMINAMATH_CALUDE_power_of_twenty_l1702_170258


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1702_170236

theorem imaginary_part_of_z (z : ℂ) (h : (Complex.I - 1) * z = 2) : 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1702_170236


namespace NUMINAMATH_CALUDE_external_tangent_intercept_l1702_170285

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in slope-intercept form --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a line is external tangent to two circles --/
def isExternalTangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

theorem external_tangent_intercept :
  let c1 : Circle := { center := (3, -2), radius := 3 }
  let c2 : Circle := { center := (15, 8), radius := 8 }
  ∀ l : Line,
    l.slope > 0 →
    isExternalTangent l c1 c2 →
    l.intercept = 720 / 11 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intercept_l1702_170285


namespace NUMINAMATH_CALUDE_problem_solution_l1702_170206

theorem problem_solution :
  ∀ (a b c : ℝ),
    (∃ (x : ℝ), x > 0 ∧ (a - 2)^2 = x ∧ (7 - 2*a)^2 = x) →
    ((3*b + 1)^(1/3) = -2) →
    (c = ⌊Real.sqrt 39⌋) →
    (a = 5 ∧ b = -3 ∧ c = 6 ∧ 
     (∃ (y : ℝ), y^2 = 5*a + 2*b - c ∧ (y = Real.sqrt 13 ∨ y = -Real.sqrt 13))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1702_170206


namespace NUMINAMATH_CALUDE_lending_rate_calculation_l1702_170223

def borrowed_amount : ℝ := 7000
def borrowed_time : ℝ := 2
def borrowed_rate : ℝ := 4
def gain_per_year : ℝ := 140

theorem lending_rate_calculation :
  let borrowed_interest := borrowed_amount * borrowed_rate * borrowed_time / 100
  let total_gain := gain_per_year * borrowed_time
  let total_interest_earned := borrowed_interest + total_gain
  let lending_rate := (total_interest_earned * 100) / (borrowed_amount * borrowed_time)
  lending_rate = 6 := by sorry

end NUMINAMATH_CALUDE_lending_rate_calculation_l1702_170223


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1702_170249

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x < -1 → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1702_170249


namespace NUMINAMATH_CALUDE_max_notebooks_purchasable_l1702_170260

def total_money : ℚ := 30
def notebook_cost : ℚ := 2.4

theorem max_notebooks_purchasable :
  ⌊total_money / notebook_cost⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchasable_l1702_170260


namespace NUMINAMATH_CALUDE_geometric_solid_sum_of_edges_l1702_170217

/-- Represents a rectangular solid with sides in geometric progression -/
structure GeometricSolid where
  a : ℝ  -- shortest side
  r : ℝ  -- common ratio
  h : r > 0  -- ensure positive ratio

/-- Volume of a GeometricSolid -/
def volume (s : GeometricSolid) : ℝ := s.a * (s.a * s.r) * (s.a * s.r * s.r)

/-- Surface area of a GeometricSolid -/
def surfaceArea (s : GeometricSolid) : ℝ :=
  2 * (s.a * (s.a * s.r) + s.a * (s.a * s.r * s.r) + (s.a * s.r) * (s.a * s.r * s.r))

/-- Sum of lengths of all edges of a GeometricSolid -/
def sumOfEdges (s : GeometricSolid) : ℝ := 4 * (s.a + (s.a * s.r) + (s.a * s.r * s.r))

/-- Theorem statement -/
theorem geometric_solid_sum_of_edges :
  ∀ s : GeometricSolid,
    volume s = 125 →
    surfaceArea s = 150 →
    sumOfEdges s = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_solid_sum_of_edges_l1702_170217


namespace NUMINAMATH_CALUDE_football_season_duration_l1702_170243

theorem football_season_duration (total_games : ℕ) (games_per_month : ℕ) 
  (h1 : total_games = 323) 
  (h2 : games_per_month = 19) : 
  total_games / games_per_month = 17 := by
  sorry

end NUMINAMATH_CALUDE_football_season_duration_l1702_170243


namespace NUMINAMATH_CALUDE_part_one_part_two_l1702_170238

/-- The function f(x) defined in the problem -/
def f (a c x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

/-- Part 1 of the problem -/
theorem part_one (a : ℝ) :
  f a 19 1 > 0 ↔ -2 < a ∧ a < 8 := by sorry

/-- Part 2 of the problem -/
theorem part_two (a c : ℝ) :
  (∀ x : ℝ, f a c x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1702_170238


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1702_170210

theorem unique_solution_3x_4y_5z : 
  ∀ x y z : ℕ+, 3^(x:ℕ) + 4^(y:ℕ) = 5^(z:ℕ) → x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1702_170210


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l1702_170273

theorem bryans_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 34) (h2 : books_per_shelf = 17) :
  total_books / books_per_shelf = 2 :=
by sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l1702_170273


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_dividing_powers_l1702_170298

theorem infinitely_many_pairs_dividing_powers (d : ℤ) 
  (h1 : d > 1) (h2 : d % 4 = 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (a + b : ℤ) ∣ (a^b + b^a : ℤ) := by
  sorry

#check infinitely_many_pairs_dividing_powers

end NUMINAMATH_CALUDE_infinitely_many_pairs_dividing_powers_l1702_170298


namespace NUMINAMATH_CALUDE_function_property_l1702_170214

/-- Given a function f(x) = 2√3 sin(3ωx + π/3) where ω > 0,
    if f(x+θ) is an even function with a period of 2π,
    then θ = 7π/6 -/
theorem function_property (ω θ : ℝ) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sqrt 3 * Real.sin (3 * ω * x + π / 3)
  (∀ x, f (x + θ) = f (-x - θ)) ∧  -- f(x+θ) is even
  (∀ x, f (x + θ) = f (x + θ + 2 * π)) →  -- f(x+θ) has period 2π
  θ = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1702_170214


namespace NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_l1702_170261

theorem max_value_constraint (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 
  2 * x + y ≤ Real.sqrt 11 := by
sorry

theorem max_value_attained : ∃ (x y : ℝ), 3 * x^2 + 2 * y^2 ≤ 6 ∧ 2 * x + y = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_l1702_170261


namespace NUMINAMATH_CALUDE_fiftieth_term_divisible_by_five_l1702_170211

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem fiftieth_term_divisible_by_five : 
  5 ∣ modifiedLucas 49 := by sorry

end NUMINAMATH_CALUDE_fiftieth_term_divisible_by_five_l1702_170211


namespace NUMINAMATH_CALUDE_a_squared_minus_b_squared_l1702_170269

theorem a_squared_minus_b_squared (a b : ℚ) 
  (h1 : a + b = 2/3) 
  (h2 : a - b = 1/6) : 
  a^2 - b^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_minus_b_squared_l1702_170269


namespace NUMINAMATH_CALUDE_cabbage_area_is_one_square_foot_l1702_170222

/-- Represents the cabbage garden --/
structure CabbageGarden where
  side_length : ℝ
  num_cabbages : ℕ

/-- The increase in cabbages from last year to this year --/
def cabbage_increase : ℕ := 211

/-- The number of cabbages grown this year --/
def cabbages_this_year : ℕ := 11236

/-- Calculates the area of a square garden --/
def garden_area (g : CabbageGarden) : ℝ := g.side_length ^ 2

theorem cabbage_area_is_one_square_foot 
  (last_year : CabbageGarden) 
  (this_year : CabbageGarden) 
  (h1 : this_year.num_cabbages = cabbages_this_year)
  (h2 : this_year.num_cabbages = last_year.num_cabbages + cabbage_increase)
  (h3 : garden_area this_year - garden_area last_year = cabbage_increase) :
  (garden_area this_year - garden_area last_year) / cabbage_increase = 1 := by
  sorry

#check cabbage_area_is_one_square_foot

end NUMINAMATH_CALUDE_cabbage_area_is_one_square_foot_l1702_170222


namespace NUMINAMATH_CALUDE_zongzi_purchase_problem_l1702_170203

/-- Represents the cost and quantity information for zongzi purchases -/
structure ZongziPurchase where
  cost_A : ℝ  -- Cost per bag of brand A zongzi
  cost_B : ℝ  -- Cost per bag of brand B zongzi
  quantity_A : ℕ  -- Quantity of brand A zongzi
  quantity_B : ℕ  -- Quantity of brand B zongzi
  total_cost : ℝ  -- Total cost of the purchase

/-- Theorem representing the zongzi purchase problem -/
theorem zongzi_purchase_problem 
  (purchase1 : ZongziPurchase)
  (purchase2 : ZongziPurchase)
  (h1 : purchase1.quantity_A = 100 ∧ purchase1.quantity_B = 150 ∧ purchase1.total_cost = 7000)
  (h2 : purchase2.quantity_A = 180 ∧ purchase2.quantity_B = 120 ∧ purchase2.total_cost = 8100)
  (h3 : purchase1.cost_A = purchase2.cost_A ∧ purchase1.cost_B = purchase2.cost_B) :
  ∃ (optimal_purchase : ZongziPurchase),
    purchase1.cost_A = 25 ∧
    purchase1.cost_B = 30 ∧
    optimal_purchase.quantity_A = 200 ∧
    optimal_purchase.quantity_B = 100 ∧
    optimal_purchase.total_cost = 8000 ∧
    optimal_purchase.quantity_A + optimal_purchase.quantity_B = 300 ∧
    optimal_purchase.quantity_A ≤ 2 * optimal_purchase.quantity_B ∧
    ∀ (other_purchase : ZongziPurchase),
      other_purchase.quantity_A + other_purchase.quantity_B = 300 →
      other_purchase.quantity_A ≤ 2 * other_purchase.quantity_B →
      other_purchase.total_cost ≥ optimal_purchase.total_cost := by
  sorry


end NUMINAMATH_CALUDE_zongzi_purchase_problem_l1702_170203


namespace NUMINAMATH_CALUDE_exam_score_problem_l1702_170225

theorem exam_score_problem (total_questions : ℕ) (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 140) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1702_170225


namespace NUMINAMATH_CALUDE_circular_arrangements_count_l1702_170224

/-- The number of ways to arrange n people in a circle with r people between A and B -/
def circularArrangements (n : ℕ) (r : ℕ) : ℕ :=
  2 * Nat.factorial (n - 2)

/-- Theorem: The number of circular arrangements with r people between A and B -/
theorem circular_arrangements_count (n : ℕ) (r : ℕ) 
  (h₁ : n ≥ 3) 
  (h₂ : r < n / 2 - 1) : 
  circularArrangements n r = 2 * Nat.factorial (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangements_count_l1702_170224


namespace NUMINAMATH_CALUDE_improved_running_distance_l1702_170246

/-- Proves that a runner who can cover 40 yards in 5 seconds and improves their speed by 40% will cover 112 yards in 10 seconds -/
theorem improved_running_distance 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (improvement_percentage : ℝ) 
  (new_time : ℝ) :
  initial_distance = 40 ∧ 
  initial_time = 5 ∧ 
  improvement_percentage = 40 ∧ 
  new_time = 10 → 
  (initial_distance * (1 + improvement_percentage / 100) * (new_time / initial_time)) = 112 :=
by sorry

end NUMINAMATH_CALUDE_improved_running_distance_l1702_170246


namespace NUMINAMATH_CALUDE_product_pass_rate_l1702_170213

-- Define the defect rates for each step
variable (a b : ℝ)

-- Assume the defect rates are between 0 and 1
variable (ha : 0 ≤ a ∧ a ≤ 1)
variable (hb : 0 ≤ b ∧ b ≤ 1)

-- Define the pass rate of the product
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem statement
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  pass_rate a b = (1 - a) * (1 - b) :=
by
  sorry

end NUMINAMATH_CALUDE_product_pass_rate_l1702_170213


namespace NUMINAMATH_CALUDE_inequality_proof_l1702_170287

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.log (Real.exp 2 / x) ≤ (1 + x) / x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1702_170287


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_root_l1702_170235

theorem fourth_power_of_cube_root (x : ℝ) : 
  x = (3 + Real.sqrt (1 + Real.sqrt 5)) ^ (1/3) → x^4 = 9 + 12 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_root_l1702_170235


namespace NUMINAMATH_CALUDE_rep_for_A_percent_is_20_l1702_170240

/-- Represents the voting scenario in a city -/
structure VotingScenario where
  total_voters : ℝ
  dem_percent : ℝ
  rep_percent : ℝ
  dem_for_A_percent : ℝ
  total_for_A_percent : ℝ
  rep_for_A_percent : ℝ

/-- The conditions of the voting scenario -/
def city_voting : VotingScenario :=
  { total_voters := 100, -- Assuming 100 for simplicity
    dem_percent := 60,
    rep_percent := 40,
    dem_for_A_percent := 85,
    total_for_A_percent := 59,
    rep_for_A_percent := 20 }

theorem rep_for_A_percent_is_20 (v : VotingScenario) (h1 : v.dem_percent + v.rep_percent = 100) 
    (h2 : v.dem_percent = 60) (h3 : v.dem_for_A_percent = 85) (h4 : v.total_for_A_percent = 59) :
  v.rep_for_A_percent = 20 := by
  sorry

#check rep_for_A_percent_is_20

end NUMINAMATH_CALUDE_rep_for_A_percent_is_20_l1702_170240


namespace NUMINAMATH_CALUDE_training_hours_calculation_l1702_170272

/-- Given a person trains for a specific number of hours per day and a total number of days,
    calculate the total hours spent training. -/
def total_training_hours (hours_per_day : ℕ) (total_days : ℕ) : ℕ :=
  hours_per_day * total_days

/-- Theorem: A person training for 5 hours every day for 42 days spends 210 hours in total. -/
theorem training_hours_calculation :
  let hours_per_day : ℕ := 5
  let initial_days : ℕ := 30
  let additional_days : ℕ := 12
  let total_days : ℕ := initial_days + additional_days
  total_training_hours hours_per_day total_days = 210 := by
  sorry

#check training_hours_calculation

end NUMINAMATH_CALUDE_training_hours_calculation_l1702_170272


namespace NUMINAMATH_CALUDE_reciprocal_nonexistence_l1702_170248

theorem reciprocal_nonexistence (a : ℝ) : (¬∃x : ℝ, x * a = 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_nonexistence_l1702_170248


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1702_170233

theorem min_value_on_circle (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → 
  ∃ (min : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1702_170233


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1702_170262

theorem polar_to_rectangular_conversion :
  let r : ℝ := 10
  let θ : ℝ := 5 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 5 ∧ y = -5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1702_170262


namespace NUMINAMATH_CALUDE_no_square_divisibility_l1702_170241

theorem no_square_divisibility (a b : ℕ) (α : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hsum : a + b = 2^α) (hα : α ≥ 1) :
  ¬∃ (k : ℕ), k > 1 ∧ (k^2 ∣ a^k + b^k) := by
  sorry

end NUMINAMATH_CALUDE_no_square_divisibility_l1702_170241


namespace NUMINAMATH_CALUDE_carla_chicken_farm_problem_l1702_170254

/-- The percentage of chickens that died in Carla's farm -/
def percentage_died (initial_chickens final_chickens : ℕ) : ℚ :=
  let bought_chickens := final_chickens - initial_chickens
  let died_chickens := bought_chickens / 10
  (died_chickens : ℚ) / initial_chickens * 100

theorem carla_chicken_farm_problem :
  percentage_died 400 1840 = 40 := by
  sorry

end NUMINAMATH_CALUDE_carla_chicken_farm_problem_l1702_170254


namespace NUMINAMATH_CALUDE_total_sales_equals_250_l1702_170230

/-- Represents the commission rate as a percentage -/
def commission_rate : ℚ := 5 / 100

/-- Represents the commission earned in Rupees -/
def commission_earned : ℚ := 25 / 2

/-- Calculates the total sales given the commission rate and commission earned -/
def total_sales (rate : ℚ) (earned : ℚ) : ℚ := earned / rate

/-- Theorem stating that the total sales equal 250 Rupees -/
theorem total_sales_equals_250 : 
  total_sales commission_rate commission_earned = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_equals_250_l1702_170230


namespace NUMINAMATH_CALUDE_total_yellow_marbles_l1702_170259

theorem total_yellow_marbles (mary joan peter : ℕ) 
  (h1 : mary = 9) 
  (h2 : joan = 3) 
  (h3 : peter = 7) : 
  mary + joan + peter = 19 := by
sorry

end NUMINAMATH_CALUDE_total_yellow_marbles_l1702_170259


namespace NUMINAMATH_CALUDE_digit_change_sum_inequality_l1702_170242

/-- Changes each digit of a positive integer by 1 (either up or down) -/
def change_digits (n : ℕ) : ℕ :=
  sorry

theorem digit_change_sum_inequality (a b : ℕ) :
  let c := a + b
  let a' := change_digits a
  let b' := change_digits b
  let c' := change_digits c
  a' + b' ≠ c' :=
by sorry

end NUMINAMATH_CALUDE_digit_change_sum_inequality_l1702_170242


namespace NUMINAMATH_CALUDE_cubic_equation_root_c_value_l1702_170212

theorem cubic_equation_root_c_value : ∃ (c d : ℚ),
  ((-2 : ℝ) - 3 * Real.sqrt 5) ^ 3 + c * ((-2 : ℝ) - 3 * Real.sqrt 5) ^ 2 + 
  d * ((-2 : ℝ) - 3 * Real.sqrt 5) + 50 = 0 → c = 114 / 41 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_c_value_l1702_170212


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l1702_170207

theorem smallest_positive_integer (a : ℝ) : 
  ∃ (b : ℤ), (∀ (x : ℝ), (x + 2) * (x + 5) * (x + 8) * (x + 11) + b > 0) ∧ 
  (∀ (c : ℤ), c < b → ∃ (y : ℝ), (y + 2) * (y + 5) * (y + 8) * (y + 11) + c ≤ 0) ∧
  b = 82 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l1702_170207


namespace NUMINAMATH_CALUDE_point_to_line_distance_l1702_170245

/-- The distance from a point to a line in 2D space -/
theorem point_to_line_distance
  (x₀ y₀ a b c : ℝ) (h : a^2 + b^2 ≠ 0) :
  let d := |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)
  ∀ x y, a * x + b * y + c = 0 → 
    d ≤ Real.sqrt ((x - x₀)^2 + (y - y₀)^2) :=
by sorry

end NUMINAMATH_CALUDE_point_to_line_distance_l1702_170245


namespace NUMINAMATH_CALUDE_juniors_in_sample_l1702_170252

/-- Proves that given the conditions, the number of juniors in the sample is 78 -/
theorem juniors_in_sample (total_students : ℕ) (freshmen : ℕ) (sophomores_juniors_diff : ℕ) 
  (freshmen_sample : ℕ) (h1 : total_students = 1290) (h2 : freshmen = 480) 
  (h3 : sophomores_juniors_diff = 30) (h4 : freshmen_sample = 96) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_juniors_in_sample_l1702_170252


namespace NUMINAMATH_CALUDE_smallest_angle_in_quadrilateral_l1702_170218

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_angle_in_quadrilateral (p q r s : ℕ) : 
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > q → q > r → r > s →
  p + q + r = 270 →
  p + q + r + s = 360 →
  s ≥ 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_quadrilateral_l1702_170218


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1702_170256

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 165826) :
  ∃ x : ℕ, x = 2 ∧ 
    (∀ y : ℕ, y < x → ¬(4 ∣ (n - y))) ∧
    (4 ∣ (n - x)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1702_170256


namespace NUMINAMATH_CALUDE_lcm_924_660_l1702_170250

theorem lcm_924_660 : Nat.lcm 924 660 = 4620 := by
  sorry

end NUMINAMATH_CALUDE_lcm_924_660_l1702_170250


namespace NUMINAMATH_CALUDE_f_minimum_value_l1702_170244

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x*y + y^2) - 3 * (x + y) + 5

/-- The theorem stating the minimum value of f(x, y) -/
theorem f_minimum_value :
  (∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ 2) ∧ f (1/2) (1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1702_170244


namespace NUMINAMATH_CALUDE_dragon_cannot_be_killed_l1702_170216

/-- Represents the possible number of heads Arthur can cut off in a single swipe --/
inductive CutOff
  | fifteen
  | seventeen
  | twenty
  | five

/-- Represents the number of heads that grow back after a cut --/
def regrow (c : CutOff) : ℕ :=
  match c with
  | CutOff.fifteen => 24
  | CutOff.seventeen => 2
  | CutOff.twenty => 14
  | CutOff.five => 17

/-- Represents a single action of cutting off heads and regrowing --/
def action (c : CutOff) : ℤ :=
  match c with
  | CutOff.fifteen => 24 - 15
  | CutOff.seventeen => 2 - 17
  | CutOff.twenty => 14 - 20
  | CutOff.five => 17 - 5

/-- The main theorem stating that it's impossible to kill the dragon --/
theorem dragon_cannot_be_killed :
  ∀ (n : ℕ) (actions : List CutOff),
    (100 + (actions.map action).sum : ℤ) % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_dragon_cannot_be_killed_l1702_170216


namespace NUMINAMATH_CALUDE_type_b_sample_count_l1702_170277

/-- Represents the number of items of type B in a stratified sample -/
def stratifiedSampleCount (totalPopulation : ℕ) (typeBPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (typeBPopulation * sampleSize) / totalPopulation

/-- Theorem stating that the number of type B items in the sample is 15 -/
theorem type_b_sample_count :
  stratifiedSampleCount 5000 1250 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_type_b_sample_count_l1702_170277


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1702_170282

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 31) : 
  a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1702_170282


namespace NUMINAMATH_CALUDE_bridge_length_l1702_170201

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time_s : ℝ) : 
  train_length = 160 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1702_170201


namespace NUMINAMATH_CALUDE_problem_solution_l1702_170234

theorem problem_solution : ∃ m : ℚ, 
  (∃ x₁ x₂ : ℚ, 5*m + 3*x₁ = 1 + x₁ ∧ 2*x₂ + m = 3*m ∧ x₁ = x₂ + 2) →
  7*m^2 - 1 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1702_170234


namespace NUMINAMATH_CALUDE_family_size_family_size_is_four_l1702_170263

theorem family_size (current_avg_age : ℝ) (youngest_age : ℝ) (birth_avg_age : ℝ) : ℝ :=
  let n := (youngest_age * birth_avg_age) / (current_avg_age - birth_avg_age - youngest_age)
  n

#check family_size 20 10 12.5

theorem family_size_is_four :
  family_size 20 10 12.5 = 4 := by sorry

end NUMINAMATH_CALUDE_family_size_family_size_is_four_l1702_170263


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1702_170284

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1702_170284


namespace NUMINAMATH_CALUDE_fraction_equality_l1702_170297

theorem fraction_equality (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y) 
  (h : (y * z - x^2) / (1 - x) = (x * z - y^2) / (1 - y)) : 
  (y * z - x^2) / (1 - x) = x + y + z ∧ (x * z - y^2) / (1 - y) = x + y + z := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1702_170297


namespace NUMINAMATH_CALUDE_geometric_sequence_existence_l1702_170253

theorem geometric_sequence_existence : ∃ (a r : ℝ), 
  a * r = 2 ∧ 
  a * r^3 = 6 ∧ 
  a = -2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_existence_l1702_170253


namespace NUMINAMATH_CALUDE_winter_olympics_souvenir_sales_l1702_170226

/-- Daily sales volume as a function of selling price -/
def daily_sales (x : ℝ) : ℝ := -10 * x + 740

/-- Daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := daily_sales x * (x - 40)

/-- The selling price is between 44 and 52 yuan -/
def valid_price (x : ℝ) : Prop := 44 ≤ x ∧ x ≤ 52

theorem winter_olympics_souvenir_sales :
  ∃ (x : ℝ), valid_price x ∧
  (daily_profit x = 2400 → x = 50) ∧
  (∀ y, valid_price y → daily_profit y ≤ daily_profit 52) ∧
  daily_profit 52 = 2640 := by
  sorry


end NUMINAMATH_CALUDE_winter_olympics_souvenir_sales_l1702_170226


namespace NUMINAMATH_CALUDE_free_throw_success_rate_increase_l1702_170255

theorem free_throw_success_rate_increase 
  (initial_attempts : ℕ) 
  (initial_successes : ℕ) 
  (additional_attempts : ℕ) 
  (additional_success_rate : ℚ) : 
  initial_attempts = 10 → 
  initial_successes = 4 → 
  additional_attempts = 16 → 
  additional_success_rate = 3/4 → 
  round ((((initial_successes + additional_success_rate * additional_attempts) / 
    (initial_attempts + additional_attempts)) - 
    (initial_successes / initial_attempts)) * 100) = 22 := by
  sorry

#check free_throw_success_rate_increase

end NUMINAMATH_CALUDE_free_throw_success_rate_increase_l1702_170255


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1702_170299

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1702_170299


namespace NUMINAMATH_CALUDE_movie_marathon_duration_l1702_170264

theorem movie_marathon_duration :
  let movie1 : ℝ := 2
  let movie2 : ℝ := movie1 * 1.5
  let movie3 : ℝ := movie1 + movie2 - 1
  movie1 + movie2 + movie3 = 9 := by sorry

end NUMINAMATH_CALUDE_movie_marathon_duration_l1702_170264


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l1702_170267

/-- 
Given a real number m < 1, prove that the complex number 1 + (m-1)i 
is located in the fourth quadrant of the complex plane.
-/
theorem complex_in_fourth_quadrant (m : ℝ) (h : m < 1) : 
  let z : ℂ := 1 + (m - 1) * I
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l1702_170267


namespace NUMINAMATH_CALUDE_incorrect_equation_l1702_170221

/-- Represents a decimal number with a non-repeating segment followed by a repeating segment -/
structure DecimalNumber where
  X : ℕ  -- non-repeating segment
  Y : ℕ  -- repeating segment
  u : ℕ  -- number of digits in X
  v : ℕ  -- number of digits in Y

/-- Converts a DecimalNumber to its real value -/
def toReal (z : DecimalNumber) : ℚ :=
  (z.X : ℚ) / 10^z.u + (z.Y : ℚ) / (10^z.u * (10^z.v - 1))

/-- The main theorem stating that the given equation does not hold for all DecimalNumbers -/
theorem incorrect_equation (z : DecimalNumber) : 
  ¬(10^(2*z.u) * (10^z.v - 1) * toReal z = (z.Y : ℚ) * ((z.X : ℚ)^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_l1702_170221


namespace NUMINAMATH_CALUDE_passed_candidates_count_l1702_170276

/-- Prove the number of passed candidates given total candidates and average marks -/
theorem passed_candidates_count
  (total_candidates : ℕ)
  (avg_all : ℚ)
  (avg_passed : ℚ)
  (avg_failed : ℚ)
  (h_total : total_candidates = 120)
  (h_avg_all : avg_all = 35)
  (h_avg_passed : avg_passed = 39)
  (h_avg_failed : avg_failed = 15) :
  ∃ (passed_candidates : ℕ), passed_candidates = 100 ∧
    passed_candidates ≤ total_candidates ∧
    (passed_candidates : ℚ) * avg_passed +
    (total_candidates - passed_candidates : ℚ) * avg_failed =
    (total_candidates : ℚ) * avg_all :=
by sorry

end NUMINAMATH_CALUDE_passed_candidates_count_l1702_170276


namespace NUMINAMATH_CALUDE_existence_of_n_l1702_170208

theorem existence_of_n : ∃ n : ℕ, n > 0 ∧ (1.001 : ℝ)^n > 10 ∧ (0.999 : ℝ)^n < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l1702_170208


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1702_170270

theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (target_income : ℝ) (available_rates : List ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.03 →
  second_rate = 0.045 →
  target_income = 580 →
  available_rates = [0.05, 0.055, 0.06, 0.065, 0.07] →
  ∃ (optimal_rate : ℝ), 
    optimal_rate ∈ available_rates ∧
    optimal_rate = 0.07 ∧
    ∀ (rate : ℝ), rate ∈ available_rates →
      |((target_income - (first_investment * first_rate + second_investment * second_rate)) / 
        (total_investment - first_investment - second_investment)) - optimal_rate| ≤
      |((target_income - (first_investment * first_rate + second_investment * second_rate)) / 
        (total_investment - first_investment - second_investment)) - rate| :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1702_170270


namespace NUMINAMATH_CALUDE_quadratic_radicals_simplification_l1702_170265

theorem quadratic_radicals_simplification :
  (∀ a b m n : ℝ, a > 0 ∧ b > 0 ∧ m > 0 ∧ n > 0 →
    m^2 + n^2 = a ∧ m * n = Real.sqrt b →
    Real.sqrt (a + 2 * Real.sqrt b) = m + n) ∧
  Real.sqrt (6 + 2 * Real.sqrt 5) = Real.sqrt 5 + 1 ∧
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 ∧
  (∀ a : ℝ, Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5 →
    a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radicals_simplification_l1702_170265


namespace NUMINAMATH_CALUDE_toy_shop_spending_l1702_170202

def total_spent (trevor_spending : ℕ) (reed_spending : ℕ) (quinn_spending : ℕ) (years : ℕ) : ℕ :=
  (trevor_spending + reed_spending + quinn_spending) * years

theorem toy_shop_spending (trevor_spending reed_spending quinn_spending : ℕ) :
  trevor_spending = reed_spending + 20 →
  reed_spending = 2 * quinn_spending →
  trevor_spending = 80 →
  total_spent trevor_spending reed_spending quinn_spending 4 = 680 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_shop_spending_l1702_170202


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l1702_170247

theorem complex_subtraction_simplification :
  (7 - 3*I) - (2 + 5*I) = 5 - 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l1702_170247


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l1702_170289

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when two dice are rolled -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (sum > 7) -/
def favorableOutcomes : ℕ := totalOutcomes - 21

/-- The probability of the sum being greater than 7 when two fair dice are rolled -/
def probSumGreaterThanSeven : ℚ := favorableOutcomes / totalOutcomes

theorem prob_sum_greater_than_seven :
  probSumGreaterThanSeven = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l1702_170289


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l1702_170204

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l1702_170204


namespace NUMINAMATH_CALUDE_log_minus_x_decreasing_l1702_170268

theorem log_minus_x_decreasing (a b c : ℝ) (h : 1 < a ∧ a < b ∧ b < c) :
  Real.log a - a > Real.log b - b ∧ Real.log b - b > Real.log c - c := by
  sorry

end NUMINAMATH_CALUDE_log_minus_x_decreasing_l1702_170268


namespace NUMINAMATH_CALUDE_total_lemons_l1702_170296

def lemon_problem (levi jayden eli ian : ℕ) : Prop :=
  levi = 5 ∧
  jayden = levi + 6 ∧
  3 * jayden = eli ∧
  2 * eli = ian ∧
  levi + jayden + eli + ian = 115

theorem total_lemons : ∃ levi jayden eli ian : ℕ, lemon_problem levi jayden eli ian := by
  sorry

end NUMINAMATH_CALUDE_total_lemons_l1702_170296


namespace NUMINAMATH_CALUDE_evaluate_expression_l1702_170205

theorem evaluate_expression : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1702_170205


namespace NUMINAMATH_CALUDE_triangle_inradius_l1702_170251

/-- The inradius of a triangle with perimeter 36 and area 45 is 2.5 -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) : 
  perimeter = 36 → area = 45 → inradius = area / (perimeter / 2) → inradius = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1702_170251


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1702_170232

theorem decimal_multiplication (a b c : ℚ) (h1 : a = 0.025) (h2 : b = 3.84) (h3 : c = 0.096) 
  (h4 : (25 : ℕ) * 384 = 9600) : a * b = c := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1702_170232


namespace NUMINAMATH_CALUDE_second_box_difference_l1702_170278

/-- Represents the amount of cereal in ounces for each box. -/
structure CerealBoxes where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Defines the properties of the cereal boxes based on the problem conditions. -/
def validCerealBoxes (boxes : CerealBoxes) : Prop :=
  boxes.first = 14 ∧
  boxes.second = boxes.first / 2 ∧
  boxes.second < boxes.third ∧
  boxes.first + boxes.second + boxes.third = 33

/-- Theorem stating that the difference between the third and second box is 5 ounces. -/
theorem second_box_difference (boxes : CerealBoxes) 
  (h : validCerealBoxes boxes) : boxes.third - boxes.second = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_box_difference_l1702_170278


namespace NUMINAMATH_CALUDE_mitten_plug_difference_l1702_170292

theorem mitten_plug_difference (mittens : ℕ) (added_plugs : ℕ) (total_plugs : ℕ) : 
  mittens = 150 → added_plugs = 30 → total_plugs = 400 →
  (total_plugs / 2 - added_plugs) - mittens = 20 := by
  sorry

end NUMINAMATH_CALUDE_mitten_plug_difference_l1702_170292


namespace NUMINAMATH_CALUDE_fence_cost_l1702_170209

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 144 → price_per_foot = 58 → cost = 2784 → 
  cost = 4 * Real.sqrt area * price_per_foot := by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l1702_170209


namespace NUMINAMATH_CALUDE_sin_double_angle_from_infinite_sum_l1702_170291

theorem sin_double_angle_from_infinite_sum (θ : ℝ) 
  (h : ∑' n, (Real.sin θ)^(2*n) = 4) : 
  Real.sin (2 * θ) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_from_infinite_sum_l1702_170291
