import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_equals_sqrt_semiperimeter_l1023_102343

theorem triangle_area_equals_sqrt_semiperimeter 
  (x y z : ℝ) (a b c s Δ : ℝ) 
  (ha : a = x / y + y / z)
  (hb : b = y / z + z / x)
  (hc : c = z / x + x / y)
  (hs : s = (a + b + c) / 2) :
  Δ = Real.sqrt s := by sorry

end NUMINAMATH_CALUDE_triangle_area_equals_sqrt_semiperimeter_l1023_102343


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l1023_102322

def orange_chips : Nat := 4
def green_chips : Nat := 3
def blue_chips : Nat := 5
def total_chips : Nat := orange_chips + green_chips + blue_chips

def satisfying_arrangements : Nat := orange_chips.factorial * green_chips.factorial * blue_chips.factorial

theorem consecutive_color_draw_probability : 
  (satisfying_arrangements : ℚ) / (total_chips.factorial : ℚ) = 1 / 665280 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l1023_102322


namespace NUMINAMATH_CALUDE_total_wheels_l1023_102302

/-- The number of bikes that can be assembled in the garage -/
def bikes_assemblable : ℕ := 10

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- Theorem: The total number of wheels in the garage is 20 -/
theorem total_wheels : bikes_assemblable * wheels_per_bike = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l1023_102302


namespace NUMINAMATH_CALUDE_probability_three_odd_less_than_eighth_l1023_102364

def range_size : ℕ := 2023
def odd_count : ℕ := (range_size + 1) / 2

theorem probability_three_odd_less_than_eighth :
  (odd_count : ℚ) / range_size *
  ((odd_count - 1) : ℚ) / (range_size - 1) *
  ((odd_count - 2) : ℚ) / (range_size - 2) <
  1 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_three_odd_less_than_eighth_l1023_102364


namespace NUMINAMATH_CALUDE_molecular_weight_C6H8O7_moles_l1023_102352

/-- The molecular weight of a single molecule of C6H8O7 in g/mol -/
def molecular_weight_C6H8O7 : ℝ := 192.124

/-- The total molecular weight in grams -/
def total_weight : ℝ := 960

/-- Theorem stating that the molecular weight of a certain number of moles of C6H8O7 is equal to the total weight -/
theorem molecular_weight_C6H8O7_moles : 
  ∃ (n : ℝ), n * molecular_weight_C6H8O7 = total_weight :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_C6H8O7_moles_l1023_102352


namespace NUMINAMATH_CALUDE_train_length_l1023_102304

/-- The length of a train given specific conditions --/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 200 →
  passing_time = 41 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 210 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1023_102304


namespace NUMINAMATH_CALUDE_factorial_squared_ge_power_l1023_102329

theorem factorial_squared_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_ge_power_l1023_102329


namespace NUMINAMATH_CALUDE_sum_lent_problem_l1023_102312

theorem sum_lent_problem (P : ℝ) : 
  P > 0 →  -- Assuming the sum lent is positive
  (8 * 0.06 * P) = P - 572 → 
  P = 1100 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_problem_l1023_102312


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1023_102389

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  sumInteriorNumbers 4 = 6 ∧
  sumInteriorNumbers 5 = 14 →
  sumInteriorNumbers 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1023_102389


namespace NUMINAMATH_CALUDE_largest_expression_l1023_102395

theorem largest_expression : 
  let a := 3 + 1 + 2 + 9
  let b := 3 * 1 + 2 + 9
  let c := 3 + 1 * 2 + 9
  let d := 3 + 1 + 2 * 9
  let e := 3 * 1 * 2 * 9
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l1023_102395


namespace NUMINAMATH_CALUDE_red_pens_per_student_red_pens_calculation_l1023_102387

theorem red_pens_per_student (students : ℕ) (black_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) (pens_taken_second_month : ℕ) 
  (remaining_pens_per_student : ℕ) : ℕ :=
  let total_black_pens := students * black_pens_per_student
  let total_pens_taken := pens_taken_first_month + pens_taken_second_month
  let total_remaining_pens := students * remaining_pens_per_student
  let initial_total_pens := total_pens_taken + total_remaining_pens
  let total_red_pens := initial_total_pens - total_black_pens
  total_red_pens / students

theorem red_pens_calculation :
  red_pens_per_student 3 43 37 41 79 = 62 := by
  sorry

end NUMINAMATH_CALUDE_red_pens_per_student_red_pens_calculation_l1023_102387


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1023_102374

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of two repeating decimals 0.̅6 and 0.̅2 minus 0.̅4 equals 4/9 -/
theorem repeating_decimal_sum_diff :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1023_102374


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1023_102360

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 2 ↔ (m - 1) * x < Real.sqrt (4 * x - x^2)) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1023_102360


namespace NUMINAMATH_CALUDE_ticket_revenue_calculation_l1023_102354

/-- Calculates the total revenue from ticket sales given the following conditions:
  * Child ticket cost: $6
  * Adult ticket cost: $9
  * Total tickets sold: 225
  * Number of adult tickets: 175
-/
theorem ticket_revenue_calculation (child_cost adult_cost total_tickets adult_tickets : ℕ) 
  (h1 : child_cost = 6)
  (h2 : adult_cost = 9)
  (h3 : total_tickets = 225)
  (h4 : adult_tickets = 175) :
  child_cost * (total_tickets - adult_tickets) + adult_cost * adult_tickets = 1875 :=
by sorry

end NUMINAMATH_CALUDE_ticket_revenue_calculation_l1023_102354


namespace NUMINAMATH_CALUDE_point_on_graph_l1023_102384

/-- The function f(x) = -2x + 3 --/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- The point (1, 1) --/
def point : ℝ × ℝ := (1, 1)

/-- Theorem: The point (1, 1) lies on the graph of f(x) = -2x + 3 --/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l1023_102384


namespace NUMINAMATH_CALUDE_bird_cost_l1023_102318

/-- The cost of birds at a pet store -/
theorem bird_cost (small_bird_cost large_bird_cost : ℚ) : 
  large_bird_cost = 2 * small_bird_cost →
  5 * large_bird_cost + 3 * small_bird_cost = 
    5 * small_bird_cost + 3 * large_bird_cost + 20 →
  small_bird_cost = 10 ∧ large_bird_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_bird_cost_l1023_102318


namespace NUMINAMATH_CALUDE_meaningful_exponent_range_l1023_102397

theorem meaningful_exponent_range (x : ℝ) : 
  (∃ y : ℝ, (2*x - 3)^0 = y) ↔ x ≠ 3/2 := by sorry

end NUMINAMATH_CALUDE_meaningful_exponent_range_l1023_102397


namespace NUMINAMATH_CALUDE_point_b_coordinates_l1023_102376

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetricPoint (p q : Point3D) : Point3D :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y, 2 * q.z - p.z⟩

def vector (p q : Point3D) : Point3D :=
  ⟨q.x - p.x, q.y - p.y, q.z - p.z⟩

theorem point_b_coordinates
  (A : Point3D)
  (P : Point3D)
  (A_prime : Point3D)
  (B_prime : Point3D)
  (h1 : A = ⟨-1, 3, -3⟩)
  (h2 : P = ⟨1, 2, 3⟩)
  (h3 : A_prime = symmetricPoint A P)
  (h4 : vector A_prime B_prime = ⟨3, 1, 5⟩)
  : ∃ B : Point3D, (B = ⟨-4, 2, -8⟩ ∧ symmetricPoint B P = B_prime) :=
sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l1023_102376


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l1023_102309

/-- A point (x, y) is inside a circle if the left side of the circle's equation is less than the right side -/
def is_inside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*y - 4 < 0

/-- The theorem stating the range of a for which the point (a+1, a-1) is inside the given circle -/
theorem point_inside_circle_range (a : ℝ) :
  is_inside_circle (a+1) (a-1) a ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l1023_102309


namespace NUMINAMATH_CALUDE_difference_of_squares_l1023_102365

theorem difference_of_squares (a b : ℝ) : a^2 - 9*b^2 = (a + 3*b) * (a - 3*b) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1023_102365


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1023_102341

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.tan (2 * α) = -Real.cos α / (2 + Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1023_102341


namespace NUMINAMATH_CALUDE_blue_area_after_transformations_l1023_102349

/-- Represents the fraction of blue area remaining after a single transformation -/
def blue_fraction_after_one_transform : ℚ := 3/4

/-- Represents the number of transformations -/
def num_transformations : ℕ := 3

/-- Represents the fraction of the original area that remains blue after all transformations -/
def final_blue_fraction : ℚ := (blue_fraction_after_one_transform) ^ num_transformations

theorem blue_area_after_transformations :
  final_blue_fraction = 27/64 := by sorry

end NUMINAMATH_CALUDE_blue_area_after_transformations_l1023_102349


namespace NUMINAMATH_CALUDE_soccer_ball_properties_l1023_102320

/-- A soccer-ball polyhedron has faces that are m-gons or n-gons (m ≠ n),
    and in every vertex, three faces meet: two m-gons and one n-gon. -/
structure SoccerBallPolyhedron where
  m : ℕ
  n : ℕ
  m_ne_n : m ≠ n
  vertex_config : 2 * ((m - 2) * π / m) + ((n - 2) * π / n) = 2 * π

theorem soccer_ball_properties (P : SoccerBallPolyhedron) :
  Even P.m ∧ P.m = 6 ∧ P.n = 5 := by
  sorry

#check soccer_ball_properties

end NUMINAMATH_CALUDE_soccer_ball_properties_l1023_102320


namespace NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1023_102355

theorem sum_of_squares_not_prime (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : a * b = c * d) : 
  ¬ Nat.Prime (Int.natAbs (a^2 + b^2 + c^2 + d^2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1023_102355


namespace NUMINAMATH_CALUDE_max_profit_at_six_l1023_102316

/-- The profit function for a certain product -/
def profit_function (x : ℝ) : ℝ := -2 * x^3 + 18 * x^2

/-- The derivative of the profit function -/
def profit_derivative (x : ℝ) : ℝ := -6 * x^2 + 36 * x

theorem max_profit_at_six :
  ∃ (x : ℝ), x > 0 ∧
  (∀ (y : ℝ), y > 0 → profit_function y ≤ profit_function x) ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_six_l1023_102316


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l1023_102371

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 45 = 0 ∧
  (n + 1) % 60 = 0 ∧
  (n + 1) % 72 = 0 ∧
  (n + 1) % 81 = 0 ∧
  (n + 1) % 100 = 0 ∧
  (n + 1) % 120 = 0

theorem smallest_divisible_number :
  is_divisible_by_all 16199 ∧
  ∀ m : ℕ, m < 16199 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l1023_102371


namespace NUMINAMATH_CALUDE_max_gum_pieces_is_31_l1023_102310

/-- Represents the number of coins Quentavious has -/
structure Coins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Represents the exchange rates for gum pieces -/
structure ExchangeRates where
  nickel_rate : ℕ
  dime_rate : ℕ
  quarter_rate : ℕ

/-- Represents the maximum number of coins that can be exchanged -/
structure MaxExchange where
  max_nickels : ℕ
  max_dimes : ℕ
  max_quarters : ℕ

/-- Calculates the maximum number of gum pieces Quentavious can get -/
def max_gum_pieces (coins : Coins) (rates : ExchangeRates) (max_exchange : MaxExchange) 
  (keep_nickels keep_dimes : ℕ) : ℕ :=
  let exchangeable_nickels := min (coins.nickels - keep_nickels) max_exchange.max_nickels
  let exchangeable_dimes := min (coins.dimes - keep_dimes) max_exchange.max_dimes
  let exchangeable_quarters := min coins.quarters max_exchange.max_quarters
  exchangeable_nickels * rates.nickel_rate + 
  exchangeable_dimes * rates.dime_rate + 
  exchangeable_quarters * rates.quarter_rate

/-- Theorem stating that the maximum number of gum pieces Quentavious can get is 31 -/
theorem max_gum_pieces_is_31 
  (coins : Coins)
  (rates : ExchangeRates)
  (max_exchange : MaxExchange)
  (h_coins : coins = ⟨5, 6, 4⟩)
  (h_rates : rates = ⟨2, 3, 5⟩)
  (h_max_exchange : max_exchange = ⟨3, 4, 2⟩)
  (h_keep_nickels : 2 ≤ coins.nickels)
  (h_keep_dimes : 1 ≤ coins.dimes) :
  max_gum_pieces coins rates max_exchange 2 1 = 31 :=
sorry

end NUMINAMATH_CALUDE_max_gum_pieces_is_31_l1023_102310


namespace NUMINAMATH_CALUDE_zachary_pushups_l1023_102377

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (h1 : david_pushups = 62) (h2 : difference = 15) :
  david_pushups - difference = 47 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l1023_102377


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1023_102338

theorem greatest_two_digit_multiple_of_17 :
  ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 17 ∣ m → m ≤ n) ∧
  17 ∣ n ∧ 10 ≤ n ∧ n ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1023_102338


namespace NUMINAMATH_CALUDE_sum_interior_angles_limited_diagonal_polygon_l1023_102363

/-- A polygon where at most 6 diagonals can be drawn from any vertex -/
structure LimitedDiagonalPolygon where
  vertices : ℕ
  diagonals_limit : vertices - 3 = 6

/-- The sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

/-- Theorem: The sum of interior angles of a LimitedDiagonalPolygon is 1260° -/
theorem sum_interior_angles_limited_diagonal_polygon (p : LimitedDiagonalPolygon) :
  sum_interior_angles p.vertices = 1260 := by
  sorry

#eval sum_interior_angles 9  -- Expected output: 1260

end NUMINAMATH_CALUDE_sum_interior_angles_limited_diagonal_polygon_l1023_102363


namespace NUMINAMATH_CALUDE_two_number_difference_l1023_102390

theorem two_number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) :
  y - x = 80 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l1023_102390


namespace NUMINAMATH_CALUDE_petes_journey_distance_l1023_102325

/-- Represents the distance of each segment of Pete's journey in blocks -/
structure JourneySegments where
  toGarage : ℕ
  toPostOffice : ℕ
  toLibrary : ℕ
  toFriend : ℕ

/-- Calculates the total distance of Pete's round trip journey -/
def totalDistance (segments : JourneySegments) : ℕ :=
  2 * (segments.toGarage + segments.toPostOffice + segments.toLibrary + segments.toFriend)

/-- Pete's actual journey segments -/
def petesJourney : JourneySegments :=
  { toGarage := 5
  , toPostOffice := 20
  , toLibrary := 8
  , toFriend := 10 }

/-- Theorem stating that Pete's total journey distance is 86 blocks -/
theorem petes_journey_distance : totalDistance petesJourney = 86 := by
  sorry


end NUMINAMATH_CALUDE_petes_journey_distance_l1023_102325


namespace NUMINAMATH_CALUDE_min_value_theorem_l1023_102348

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 2 * a₀ + 3 * b₀ + 6 * c₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1023_102348


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1023_102335

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1023_102335


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l1023_102315

theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (apple_pie_lovers : ℕ) 
  (chocolate_cake_lovers : ℕ) 
  (neither_dessert_lovers : ℕ) 
  (h1 : total_students = 35)
  (h2 : apple_pie_lovers = 20)
  (h3 : chocolate_cake_lovers = 17)
  (h4 : neither_dessert_lovers = 10) :
  total_students - neither_dessert_lovers + apple_pie_lovers + chocolate_cake_lovers - total_students = 12 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l1023_102315


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l1023_102381

theorem imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : 
  1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l1023_102381


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1023_102379

theorem quadratic_roots_sum_product (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 201 →
  p*q + r*s = -28743/12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1023_102379


namespace NUMINAMATH_CALUDE_no_discriminant_for_quartic_l1023_102386

theorem no_discriminant_for_quartic (P : ℝ → ℝ → ℝ → ℝ → ℝ) :
  ∃ (a b c d : ℝ),
    (∃ (r₁ r₂ r₃ r₄ : ℝ), ∀ (x : ℝ),
      x^4 + a*x^3 + b*x^2 + c*x + d = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) ∧
      P a b c d < 0) ∨
    ((¬ ∃ (r₁ r₂ r₃ r₄ : ℝ), ∀ (x : ℝ),
      x^4 + a*x^3 + b*x^2 + c*x + d = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄)) ∧
      P a b c d ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_discriminant_for_quartic_l1023_102386


namespace NUMINAMATH_CALUDE_trip_time_calculation_l1023_102301

/-- Represents the time for a trip given two different speeds -/
def trip_time (initial_speed initial_time new_speed : ℚ) : ℚ :=
  (initial_speed * initial_time) / new_speed

theorem trip_time_calculation (initial_speed initial_time new_speed : ℚ) :
  initial_speed = 80 →
  initial_time = 16/3 →
  new_speed = 50 →
  trip_time initial_speed initial_time new_speed = 128/15 := by
  sorry

#eval trip_time 80 (16/3) 50

end NUMINAMATH_CALUDE_trip_time_calculation_l1023_102301


namespace NUMINAMATH_CALUDE_total_rats_l1023_102342

theorem total_rats (elodie hunter kenia : ℕ) : 
  elodie = 30 →
  elodie = hunter + 10 →
  kenia = 3 * (elodie + hunter) →
  elodie + hunter + kenia = 200 := by
sorry

end NUMINAMATH_CALUDE_total_rats_l1023_102342


namespace NUMINAMATH_CALUDE_dinner_meals_count_l1023_102330

/-- Represents the number of meals in a restaurant scenario -/
structure RestaurantMeals where
  lunch_prepared : ℕ
  lunch_sold : ℕ
  dinner_prepared : ℕ

/-- Calculates the total number of meals available for dinner -/
def meals_for_dinner (r : RestaurantMeals) : ℕ :=
  (r.lunch_prepared - r.lunch_sold) + r.dinner_prepared

/-- Theorem stating the number of meals available for dinner in the given scenario -/
theorem dinner_meals_count (r : RestaurantMeals) 
  (h1 : r.lunch_prepared = 17) 
  (h2 : r.lunch_sold = 12) 
  (h3 : r.dinner_prepared = 5) : 
  meals_for_dinner r = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_meals_count_l1023_102330


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1023_102391

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - I) : 2 / z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1023_102391


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l1023_102337

/-- The distance from any focus of the hyperbola x^2 - y^2 = 1 to any of its asymptotes is 1 -/
theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), x^2 - y^2 = 1 →
  ∀ (fx : ℝ), fx^2 = 2 →
  ∀ (a b : ℝ), a^2 = 1 ∧ b^2 = 1 ∧ a * b = 0 →
  |a * fx + b * 0| / Real.sqrt (a^2 + b^2) = 1 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l1023_102337


namespace NUMINAMATH_CALUDE_lcm_of_3_8_9_12_l1023_102358

theorem lcm_of_3_8_9_12 : Nat.lcm 3 (Nat.lcm 8 (Nat.lcm 9 12)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_3_8_9_12_l1023_102358


namespace NUMINAMATH_CALUDE_carly_butterfly_practice_l1023_102383

/-- The number of hours Carly practices butterfly stroke per day -/
def butterfly_hours : ℝ := 3

/-- The number of days per week Carly practices butterfly stroke -/
def butterfly_days_per_week : ℕ := 4

/-- The number of hours Carly practices backstroke per day -/
def backstroke_hours : ℝ := 2

/-- The number of days per week Carly practices backstroke -/
def backstroke_days_per_week : ℕ := 6

/-- The total number of hours Carly practices swimming in a month -/
def total_hours_per_month : ℝ := 96

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

theorem carly_butterfly_practice :
  butterfly_hours * (butterfly_days_per_week * weeks_per_month) +
  backstroke_hours * (backstroke_days_per_week * weeks_per_month) =
  total_hours_per_month := by sorry

end NUMINAMATH_CALUDE_carly_butterfly_practice_l1023_102383


namespace NUMINAMATH_CALUDE_smallest_permutation_number_is_1089_l1023_102340

/-- A function that returns true if two natural numbers are permutations of each other's digits -/
def is_digit_permutation (a b : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number satisfying the permutation condition when multiplied by 9 -/
noncomputable def smallest_permutation_number : ℕ := sorry

theorem smallest_permutation_number_is_1089 :
  smallest_permutation_number = 1089 ∧
  is_digit_permutation smallest_permutation_number (9 * smallest_permutation_number) ∧
  ∀ n < smallest_permutation_number, ¬is_digit_permutation n (9 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_permutation_number_is_1089_l1023_102340


namespace NUMINAMATH_CALUDE_max_identical_papers_l1023_102317

def heart_stickers : ℕ := 240
def star_stickers : ℕ := 162
def smiley_stickers : ℕ := 90
def sun_stickers : ℕ := 54

def ratio_heart_to_smiley (n : ℕ) : Prop :=
  2 * (n * smiley_stickers) = n * heart_stickers

def ratio_star_to_sun (n : ℕ) : Prop :=
  3 * (n * sun_stickers) = n * star_stickers

def all_stickers_used (n : ℕ) : Prop :=
  n * (heart_stickers / n + star_stickers / n + smiley_stickers / n + sun_stickers / n) =
    heart_stickers + star_stickers + smiley_stickers + sun_stickers

theorem max_identical_papers : 
  ∃ (n : ℕ), n = 18 ∧ 
    ratio_heart_to_smiley n ∧ 
    ratio_star_to_sun n ∧ 
    all_stickers_used n ∧ 
    ∀ (m : ℕ), m > n → 
      ¬(ratio_heart_to_smiley m ∧ ratio_star_to_sun m ∧ all_stickers_used m) :=
by sorry

end NUMINAMATH_CALUDE_max_identical_papers_l1023_102317


namespace NUMINAMATH_CALUDE_n_pointed_star_value_l1023_102326

/-- Represents an n-pointed star. -/
structure PointedStar where
  n : ℕ
  segment_length : ℝ
  angle_a : ℝ
  angle_b : ℝ

/-- Theorem stating the properties of the n-pointed star and the value of n. -/
theorem n_pointed_star_value (star : PointedStar) :
  star.segment_length = 2 * star.n ∧
  star.angle_a = star.angle_b - 10 ∧
  star.n > 2 →
  star.n = 36 := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_value_l1023_102326


namespace NUMINAMATH_CALUDE_valid_integers_count_l1023_102327

/-- The number of permutations of 6 distinct elements -/
def total_permutations : ℕ := 720

/-- The number of permutations satisfying the first condition (1 left of 2) -/
def permutations_condition1 : ℕ := total_permutations / 2

/-- The number of permutations satisfying both conditions (1 left of 2 and 3 left of 4) -/
def permutations_both_conditions : ℕ := permutations_condition1 / 2

/-- Theorem stating the number of valid 6-digit integers -/
theorem valid_integers_count : permutations_both_conditions = 180 := by sorry

end NUMINAMATH_CALUDE_valid_integers_count_l1023_102327


namespace NUMINAMATH_CALUDE_sum_remainder_thirteen_l1023_102394

theorem sum_remainder_thirteen : ∃ k : ℕ, (8930 + 8931 + 8932 + 8933 + 8934) = 13 * k + 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_thirteen_l1023_102394


namespace NUMINAMATH_CALUDE_grade_distribution_l1023_102359

theorem grade_distribution (frac_A frac_B frac_C frac_D : ℝ) 
  (h1 : frac_A = 0.6)
  (h2 : frac_B = 0.25)
  (h3 : frac_C = 0.1)
  (h4 : frac_D = 0.05) :
  frac_A + frac_B + frac_C + frac_D = 1 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l1023_102359


namespace NUMINAMATH_CALUDE_james_final_amounts_l1023_102334

def calculate_final_amounts (initial_gold : ℕ) (tax_rate : ℚ) (divorce_loss : ℚ) 
  (investment_percentage : ℚ) (stock_gain : ℕ) (exchange_rates : List ℚ) : ℕ × ℕ × ℕ :=
  sorry

theorem james_final_amounts :
  let initial_gold : ℕ := 60
  let tax_rate : ℚ := 1/10
  let divorce_loss : ℚ := 1/2
  let investment_percentage : ℚ := 1/4
  let stock_gain : ℕ := 1
  let exchange_rates : List ℚ := [5, 7, 3]
  let (silver_bars, remaining_gold, stock_investment) := 
    calculate_final_amounts initial_gold tax_rate divorce_loss investment_percentage stock_gain exchange_rates
  silver_bars = 99 ∧ remaining_gold = 3 ∧ stock_investment = 6 :=
by sorry

end NUMINAMATH_CALUDE_james_final_amounts_l1023_102334


namespace NUMINAMATH_CALUDE_not_square_sum_divisor_l1023_102361

theorem not_square_sum_divisor (n : ℕ) (d : ℕ) (h : d ∣ 2 * n^2) :
  ¬∃ (x : ℕ), n^2 + d = x^2 := by
sorry

end NUMINAMATH_CALUDE_not_square_sum_divisor_l1023_102361


namespace NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1023_102362

/-- A quadratic polynomial with non-negative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The statement of the theorem -/
theorem quadratic_polynomial_inequality (P : QuadraticPolynomial) (x y : ℝ) :
  (P.eval (x * y))^2 ≤ (P.eval (x^2)) * (P.eval (y^2)) := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1023_102362


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1023_102333

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1023_102333


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1023_102350

/-- Given two vectors are parallel if their coordinates are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 = t * w.1 ∧ v.2 = t * w.2

theorem parallel_vectors_k_value :
  let e₁ : ℝ × ℝ := (1, 0)
  let e₂ : ℝ × ℝ := (0, 1)
  let a : ℝ × ℝ := (e₁.1 - 2 * e₂.1, e₁.2 - 2 * e₂.2)
  ∀ k : ℝ,
    let b : ℝ × ℝ := (k * e₁.1 + e₂.1, k * e₁.2 + e₂.2)
    are_parallel a b → k = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1023_102350


namespace NUMINAMATH_CALUDE_number_problem_l1023_102353

theorem number_problem (N : ℚ) : 
  (N / (4/5) = (4/5) * N + 27) → N = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1023_102353


namespace NUMINAMATH_CALUDE_div_exp_eq_pow_specific_calculation_l1023_102370

/-- Division exponentiation for rational numbers -/
def div_exp (a : ℚ) (n : ℕ) : ℚ :=
  if n ≤ 1 then a else (1 / a) ^ (n - 2)

/-- Theorem for division exponentiation -/
theorem div_exp_eq_pow (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  div_exp a n = (1 / a) ^ (n - 2) :=
sorry

/-- Theorem for specific calculation -/
theorem specific_calculation :
  2^2 * div_exp (-1/3) 4 / div_exp (-2) 3 - div_exp (-3) 2 = -73 :=
sorry

end NUMINAMATH_CALUDE_div_exp_eq_pow_specific_calculation_l1023_102370


namespace NUMINAMATH_CALUDE_square_area_with_line_area_of_square_ABCD_l1023_102366

/-- A square with a line passing through it -/
structure SquareWithLine where
  /-- The side length of the square -/
  side : ℝ
  /-- The distance from vertex A to the line -/
  dist_A : ℝ
  /-- The distance from vertex C to the line -/
  dist_C : ℝ
  /-- The line passes through the midpoint of AB -/
  midpoint_AB : dist_A = side / 2
  /-- The line intersects BC -/
  intersects_BC : dist_C < side

/-- The theorem stating the area of the square given the conditions -/
theorem square_area_with_line (s : SquareWithLine) (h1 : s.dist_A = 4) (h2 : s.dist_C = 7) : 
  s.side ^ 2 = 185 := by
  sorry

/-- The main theorem proving the area of the square ABCD is 185 -/
theorem area_of_square_ABCD : ∃ s : SquareWithLine, s.side ^ 2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_line_area_of_square_ABCD_l1023_102366


namespace NUMINAMATH_CALUDE_inverse_of_i_minus_two_i_inv_l1023_102357

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem inverse_of_i_minus_two_i_inv (h : i^2 = -1) :
  (i - 2 * i⁻¹)⁻¹ = -i / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_i_minus_two_i_inv_l1023_102357


namespace NUMINAMATH_CALUDE_birds_optimal_speed_l1023_102336

/-- Represents the problem of finding the optimal speed for Mr. Bird's commute --/
theorem birds_optimal_speed (d : ℝ) (t : ℝ) : 
  d > 0 → -- distance is positive
  t > 0 → -- time is positive
  d = 50 * (t + 1/12) → -- equation for 50 mph
  d = 70 * (t - 1/12) → -- equation for 70 mph
  ∃ (speed : ℝ), 
    speed = d / t ∧ 
    speed = 70 ∧ 
    speed > 50 :=
by sorry

end NUMINAMATH_CALUDE_birds_optimal_speed_l1023_102336


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1023_102382

/-- Given three types of tea mixed in a specific ratio, calculate the price of the mixture per kg -/
theorem tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℕ) : 
  price1 = 126 →
  price2 = 135 →
  price3 = 175.5 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3 : ℚ) = 153 := by
sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l1023_102382


namespace NUMINAMATH_CALUDE_fourth_number_is_28_l1023_102306

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def sequence_property (a b c d : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧
  (digit_sum a + digit_sum b + digit_sum c + digit_sum d) * 4 = a + b + c + d

theorem fourth_number_is_28 :
  ∃ (d : ℕ), sequence_property 46 19 63 d ∧ d = 28 :=
sorry

end NUMINAMATH_CALUDE_fourth_number_is_28_l1023_102306


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l1023_102375

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ 1 < x} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l1023_102375


namespace NUMINAMATH_CALUDE_morning_shells_count_l1023_102308

/-- The number of shells Lino picked up in the afternoon -/
def afternoon_shells : ℕ := 324

/-- The total number of shells Lino picked up -/
def total_shells : ℕ := 616

/-- The number of shells Lino picked up in the morning -/
def morning_shells : ℕ := total_shells - afternoon_shells

theorem morning_shells_count : morning_shells = 292 := by
  sorry

end NUMINAMATH_CALUDE_morning_shells_count_l1023_102308


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1023_102398

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1023_102398


namespace NUMINAMATH_CALUDE_gym_towels_theorem_l1023_102332

/-- Represents the number of guests entering the gym each hour -/
structure GymHours :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- Calculates the total number of towels used based on gym hours -/
def totalTowels (hours : GymHours) : ℕ :=
  hours.first + hours.second + hours.third + hours.fourth

/-- Theorem: Given the specified conditions, the total number of towels used is 285 -/
theorem gym_towels_theorem (hours : GymHours) 
  (h1 : hours.first = 50)
  (h2 : hours.second = hours.first + hours.first / 5)
  (h3 : hours.third = hours.second + hours.second / 4)
  (h4 : hours.fourth = hours.third + hours.third / 3)
  : totalTowels hours = 285 := by
  sorry

#eval totalTowels { first := 50, second := 60, third := 75, fourth := 100 }

end NUMINAMATH_CALUDE_gym_towels_theorem_l1023_102332


namespace NUMINAMATH_CALUDE_cubic_factorization_l1023_102351

theorem cubic_factorization (x : ℝ) : x^3 - 2*x^2 + x = x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1023_102351


namespace NUMINAMATH_CALUDE_water_height_in_cylinder_l1023_102339

/-- The height of water in a cylinder when poured from a cone -/
theorem water_height_in_cylinder (cone_radius cone_height cyl_radius : ℝ) 
  (h_cone_radius : cone_radius = 12)
  (h_cone_height : cone_height = 18)
  (h_cyl_radius : cyl_radius = 24) : 
  (1 / 3 * π * cone_radius^2 * cone_height) / (π * cyl_radius^2) = 1.5 := by
  sorry

#check water_height_in_cylinder

end NUMINAMATH_CALUDE_water_height_in_cylinder_l1023_102339


namespace NUMINAMATH_CALUDE_sin_cos_identity_sin_tan_simplification_l1023_102392

-- Question 1
theorem sin_cos_identity :
  Real.sin (34 * π / 180) * Real.sin (26 * π / 180) - 
  Real.sin (56 * π / 180) * Real.cos (26 * π / 180) = -1/2 := by sorry

-- Question 2
theorem sin_tan_simplification :
  Real.sin (50 * π / 180) * (Real.sqrt 3 * Real.tan (10 * π / 180) + 1) = 
  Real.cos (20 * π / 180) / Real.cos (10 * π / 180) := by sorry

end NUMINAMATH_CALUDE_sin_cos_identity_sin_tan_simplification_l1023_102392


namespace NUMINAMATH_CALUDE_jellybean_ratio_l1023_102372

/-- Proves that the ratio of jellybeans Shannon refilled to the total taken out
    by Samantha and Shelby is 1/2, given the initial count, the amounts taken
    by Samantha and Shelby, and the final count. -/
theorem jellybean_ratio (initial : ℕ) (samantha_taken : ℕ) (shelby_taken : ℕ) (final : ℕ)
  (h1 : initial = 90)
  (h2 : samantha_taken = 24)
  (h3 : shelby_taken = 12)
  (h4 : final = 72) :
  (final - (initial - (samantha_taken + shelby_taken))) / (samantha_taken + shelby_taken) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l1023_102372


namespace NUMINAMATH_CALUDE_blake_purchase_change_l1023_102399

/-- The amount of change Blake will receive after his purchase -/
def blakes_change (lollipop_count : ℕ) (chocolate_pack_count : ℕ) (lollipop_price : ℕ) (bill_count : ℕ) (bill_value : ℕ) : ℕ :=
  let chocolate_pack_price := 4 * lollipop_price
  let total_cost := lollipop_count * lollipop_price + chocolate_pack_count * chocolate_pack_price
  let amount_paid := bill_count * bill_value
  amount_paid - total_cost

theorem blake_purchase_change :
  blakes_change 4 6 2 6 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_purchase_change_l1023_102399


namespace NUMINAMATH_CALUDE_inequality_proof_l1023_102388

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b)^2 / (2 * (a + b)) ≤ Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ∧
  Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ≤ (a - b)^2 / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1023_102388


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1023_102319

/-- The point corresponding to (1+3i)(3-i) is located in the first quadrant of the complex plane. -/
theorem point_in_first_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1023_102319


namespace NUMINAMATH_CALUDE_min_value_theorem_l1023_102324

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 128/x^4 ≥ 256 ∧ ∃ y > 0, y^2 + 12*y + 128/y^4 = 256 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1023_102324


namespace NUMINAMATH_CALUDE_lines_perpendicular_l1023_102305

/-- Two lines with slopes that are roots of x^2 - mx - 1 = 0 are perpendicular --/
theorem lines_perpendicular (m : ℝ) (k₁ k₂ : ℝ) : 
  k₁^2 - m*k₁ - 1 = 0 → k₂^2 - m*k₂ - 1 = 0 → k₁ * k₂ = -1 := by
  sorry

#check lines_perpendicular

end NUMINAMATH_CALUDE_lines_perpendicular_l1023_102305


namespace NUMINAMATH_CALUDE_negative_nine_less_than_negative_two_l1023_102331

theorem negative_nine_less_than_negative_two : -9 < -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_nine_less_than_negative_two_l1023_102331


namespace NUMINAMATH_CALUDE_not_algorithm_quadratic_roots_l1023_102367

/-- Represents a statement that might be an algorithm --/
inductive Statement
  | travel_plan : Statement
  | linear_equation_steps : Statement
  | quadratic_equation_roots : Statement
  | sum_calculation : Statement

/-- Predicate to determine if a statement is an algorithm --/
def is_algorithm (s : Statement) : Prop :=
  match s with
  | Statement.travel_plan => True
  | Statement.linear_equation_steps => True
  | Statement.quadratic_equation_roots => False
  | Statement.sum_calculation => True

theorem not_algorithm_quadratic_roots :
  ¬(is_algorithm Statement.quadratic_equation_roots) ∧
  (is_algorithm Statement.travel_plan) ∧
  (is_algorithm Statement.linear_equation_steps) ∧
  (is_algorithm Statement.sum_calculation) := by
  sorry

end NUMINAMATH_CALUDE_not_algorithm_quadratic_roots_l1023_102367


namespace NUMINAMATH_CALUDE_wrexham_orchestra_max_members_l1023_102300

theorem wrexham_orchestra_max_members :
  ∀ m : ℕ,
  (∃ k : ℕ, 30 * m = 31 * k + 7) →
  30 * m < 1200 →
  (∀ n : ℕ, (∃ j : ℕ, 30 * n = 31 * j + 7) → 30 * n < 1200 → 30 * n ≤ 30 * m) →
  30 * m = 720 :=
by sorry

end NUMINAMATH_CALUDE_wrexham_orchestra_max_members_l1023_102300


namespace NUMINAMATH_CALUDE_parabola_point_relation_l1023_102347

-- Define the parabola function
def parabola (x c : ℝ) : ℝ := -x^2 + 6*x + c

-- Define the theorem
theorem parabola_point_relation (c y₁ y₂ y₃ : ℝ) :
  parabola 1 c = y₁ →
  parabola 3 c = y₂ →
  parabola 4 c = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ := by
  sorry


end NUMINAMATH_CALUDE_parabola_point_relation_l1023_102347


namespace NUMINAMATH_CALUDE_simplify_expression_l1023_102380

theorem simplify_expression (n : ℕ) : (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1023_102380


namespace NUMINAMATH_CALUDE_inequality_solution_l1023_102345

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4/3 ∨ x > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1023_102345


namespace NUMINAMATH_CALUDE_ascending_order_abc_l1023_102314

theorem ascending_order_abc :
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4/5 : ℝ)
  let c := Real.sin 1
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l1023_102314


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1023_102328

-- Define a function f that satisfies the symmetry condition
def f (x : ℝ) : ℝ := sorry

-- State the symmetry condition
axiom symmetry_condition : ∀ x, f x = f (4 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1023_102328


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1023_102307

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2, 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1023_102307


namespace NUMINAMATH_CALUDE_sin_double_angle_circle_l1023_102303

theorem sin_double_angle_circle (α : Real) :
  let P : ℝ × ℝ := (1, 2)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  (P.1^2 + P.2^2 = r^2) →  -- Point P is on the circle
  (P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) →  -- P is on the terminal side of α
  Real.sin (2 * α) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_circle_l1023_102303


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1023_102321

theorem quadratic_factorization (x : ℝ) :
  ∃ (m n : ℤ), 6 * x^2 - 5 * x - 6 = (6 * x + m) * (x + n) ∧ m - n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1023_102321


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l1023_102373

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

/-- Proves that the athlete's heart beats 21600 times during the 30-mile race. -/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 120  -- heartbeats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 21600 := by
  sorry

#eval total_heartbeats 120 6 30

end NUMINAMATH_CALUDE_athlete_heartbeats_l1023_102373


namespace NUMINAMATH_CALUDE_eulers_formula_l1023_102356

/-- A closed polyhedron is a structure with a number of edges, faces, and vertices. -/
structure ClosedPolyhedron where
  edges : ℕ
  faces : ℕ
  vertices : ℕ

/-- Euler's formula for polyhedra states that for any closed polyhedron, 
    the number of edges plus 2 is equal to the sum of the number of faces and vertices. -/
theorem eulers_formula (p : ClosedPolyhedron) : p.edges + 2 = p.faces + p.vertices := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1023_102356


namespace NUMINAMATH_CALUDE_unique_orthogonal_chord_l1023_102385

-- Define the quadratic function
def f (p q x : ℝ) : ℝ := x^2 - 2*p*x + q

-- State the theorem
theorem unique_orthogonal_chord (p q : ℝ) :
  p > 0 ∧ q > 0 ∧  -- p and q are positive
  (∀ x, f p q x ≠ 0) ∧  -- graph doesn't intersect x-axis
  (∃! a, a > 0 ∧
    f p q (p - a) = f p q (p + a) ∧  -- AB parallel to x-axis
    (p - a) * (p + a) + (f p q (p - a))^2 = 0)  -- angle AOB = π/2
  → q = 1/4 := by
sorry

end NUMINAMATH_CALUDE_unique_orthogonal_chord_l1023_102385


namespace NUMINAMATH_CALUDE_women_in_sports_club_l1023_102344

/-- The number of women in a sports club -/
def number_of_women (total_members participants : ℕ) : ℕ :=
  let women := 3 * (total_members - participants) / 2
  women

/-- Theorem: The number of women in the sports club is 21 -/
theorem women_in_sports_club :
  number_of_women 36 22 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_sports_club_l1023_102344


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l1023_102393

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l1023_102393


namespace NUMINAMATH_CALUDE_monotonic_at_most_one_solution_l1023_102313

-- Define a monotonic function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- State the theorem
theorem monotonic_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) (h : Monotonic f) :
  ∃! x, f x = c ∨ (∀ x, f x ≠ c) :=
sorry

end NUMINAMATH_CALUDE_monotonic_at_most_one_solution_l1023_102313


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1023_102311

theorem three_numbers_sum (a b c x y z : ℝ) : 
  (x + y = z + a) → 
  (x + z = y + b) → 
  (y + z = x + c) → 
  (x = (a + b - c) / 2) ∧ 
  (y = (a - b + c) / 2) ∧ 
  (z = (-a + b + c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1023_102311


namespace NUMINAMATH_CALUDE_tangent_line_to_two_parabolas_l1023_102368

/-- Given curves C₁: y = x² and C₂: y = -(x - 2)², prove that the line l: y = -2x + 3 is tangent to both C₁ and C₂ -/
theorem tangent_line_to_two_parabolas :
  let C₁ : ℝ → ℝ := λ x ↦ x^2
  let C₂ : ℝ → ℝ := λ x ↦ -(x - 2)^2
  let l : ℝ → ℝ := λ x ↦ -2*x + 3
  (∃ x₁, (C₁ x₁ = l x₁) ∧ (deriv C₁ x₁ = deriv l x₁)) ∧
  (∃ x₂, (C₂ x₂ = l x₂) ∧ (deriv C₂ x₂ = deriv l x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_two_parabolas_l1023_102368


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l1023_102323

/-- Soccer camp ratio problem -/
theorem soccer_camp_ratio :
  ∀ (total_kids soccer_kids : ℕ),
  total_kids = 2000 →
  soccer_kids * 3 = 750 * 4 →
  soccer_kids * 2 = total_kids :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_ratio_l1023_102323


namespace NUMINAMATH_CALUDE_work_completion_men_count_l1023_102378

/-- Given that 42 men can complete a piece of work in 18 days, and another group can complete
    the same work in 28 days, prove that the second group consists of 27 men. -/
theorem work_completion_men_count :
  ∀ (work : ℝ) (men_group2 : ℕ),
    work = 42 * 18 →
    work = men_group2 * 28 →
    men_group2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l1023_102378


namespace NUMINAMATH_CALUDE_additional_money_needed_l1023_102396

/-- Given a football team, budget, and cost per football, calculate the additional money needed --/
theorem additional_money_needed 
  (num_players : ℕ) 
  (budget : ℕ) 
  (cost_per_football : ℕ) 
  (h1 : num_players = 22)
  (h2 : budget = 1500)
  (h3 : cost_per_football = 69) : 
  (num_players * cost_per_football - budget : ℤ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l1023_102396


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_parallelepiped_l1023_102369

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  diagonal : ℝ
  angle_with_base : ℝ
  base_area : ℝ

-- Define the theorem
theorem lateral_surface_area_of_parallelepiped (p : RectangularParallelepiped) 
  (h1 : p.diagonal = 10)
  (h2 : p.angle_with_base = Real.pi / 3)  -- 60 degrees in radians
  (h3 : p.base_area = 12) :
  ∃ (lateral_area : ℝ), lateral_area = 70 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_parallelepiped_l1023_102369


namespace NUMINAMATH_CALUDE_triangle_height_l1023_102346

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 3 → area = 9 → area = (base * height) / 2 → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1023_102346
