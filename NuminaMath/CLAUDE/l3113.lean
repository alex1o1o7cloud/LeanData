import Mathlib

namespace NUMINAMATH_CALUDE_inequality_for_negative_reals_l3113_311301

theorem inequality_for_negative_reals (a b : ℝ) : 
  a < b → b < 0 → a + 1/b < b + 1/a := by sorry

end NUMINAMATH_CALUDE_inequality_for_negative_reals_l3113_311301


namespace NUMINAMATH_CALUDE_pole_height_l3113_311311

theorem pole_height (cable_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ)
  (h1 : cable_ground_distance = 5)
  (h2 : person_distance = 4)
  (h3 : person_height = 3) :
  let pole_height := cable_ground_distance * person_height / (cable_ground_distance - person_distance)
  pole_height = 15 := by sorry

end NUMINAMATH_CALUDE_pole_height_l3113_311311


namespace NUMINAMATH_CALUDE_casper_window_problem_l3113_311341

theorem casper_window_problem (total_windows : ℕ) (locked_windows : ℕ) : 
  total_windows = 8 → 
  locked_windows = 1 → 
  (total_windows - locked_windows) * (total_windows - locked_windows - 1) = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_casper_window_problem_l3113_311341


namespace NUMINAMATH_CALUDE_point_on_y_axis_implies_a_equals_two_l3113_311398

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
axiom point_on_y_axis (x y : ℝ) : (x, y) ∈ {p : ℝ × ℝ | p.1 = 0} ↔ x = 0

/-- The theorem states that if the point A(a-2, 2a+8) lies on the y-axis, then a = 2 -/
theorem point_on_y_axis_implies_a_equals_two (a : ℝ) :
  (a - 2, 2 * a + 8) ∈ {p : ℝ × ℝ | p.1 = 0} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_implies_a_equals_two_l3113_311398


namespace NUMINAMATH_CALUDE_garrick_nickels_count_l3113_311319

/-- The number of cents in a dime -/
def dime_value : ℕ := 10

/-- The number of cents in a quarter -/
def quarter_value : ℕ := 25

/-- The number of cents in a nickel -/
def nickel_value : ℕ := 5

/-- The number of cents in a penny -/
def penny_value : ℕ := 1

/-- The number of dimes Cindy tossed -/
def cindy_dimes : ℕ := 5

/-- The number of quarters Eric flipped -/
def eric_quarters : ℕ := 3

/-- The number of pennies Ivy dropped -/
def ivy_pennies : ℕ := 60

/-- The total amount of money in the pond in cents -/
def total_cents : ℕ := 200

/-- The number of nickels Garrick threw into the pond -/
def garrick_nickels : ℕ := (total_cents - (cindy_dimes * dime_value + eric_quarters * quarter_value + ivy_pennies * penny_value)) / nickel_value

theorem garrick_nickels_count : garrick_nickels = 3 := by
  sorry

end NUMINAMATH_CALUDE_garrick_nickels_count_l3113_311319


namespace NUMINAMATH_CALUDE_orthogonal_projection_area_l3113_311309

/-- A plane polygon -/
structure PlanePolygon where
  area : ℝ

/-- An orthogonal projection of a plane polygon onto another plane -/
structure OrthogonalProjection (P : PlanePolygon) where
  area : ℝ
  angle : ℝ  -- Angle between the original plane and the projection plane

/-- 
Theorem: The area of the orthogonal projection of a plane polygon 
onto a plane is equal to the area of the polygon being projected, 
multiplied by the cosine of the angle between the projection plane 
and the plane of the polygon.
-/
theorem orthogonal_projection_area 
  (P : PlanePolygon) (proj : OrthogonalProjection P) : 
  proj.area = P.area * Real.cos proj.angle := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_projection_area_l3113_311309


namespace NUMINAMATH_CALUDE_function_property_l3113_311318

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem function_property (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ > f x₂) →
  a < 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3113_311318


namespace NUMINAMATH_CALUDE_coordinate_sum_theorem_l3113_311329

/-- Given a function f where f(3) = 4, the sum of the coordinates of the point (x, y) 
    satisfying 4y = 2f(2x) + 7 is equal to 5.25. -/
theorem coordinate_sum_theorem (f : ℝ → ℝ) (hf : f 3 = 4) :
  ∃ (x y : ℝ), 4 * y = 2 * f (2 * x) + 7 ∧ x + y = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_theorem_l3113_311329


namespace NUMINAMATH_CALUDE_buttons_pattern_l3113_311323

/-- Represents the number of buttons in the nth box -/
def buttonsInBox (n : ℕ) : ℕ := 3^(n - 1)

/-- Represents the total number of buttons up to the nth box -/
def totalButtons (n : ℕ) : ℕ := (3^n - 1) / 2

theorem buttons_pattern (n : ℕ) (h : n > 0) :
  (buttonsInBox 1 = 1) ∧
  (buttonsInBox 2 = 3) ∧
  (buttonsInBox 3 = 9) ∧
  (buttonsInBox 4 = 27) ∧
  (buttonsInBox 5 = 81) →
  (∀ k : ℕ, k > 0 → buttonsInBox k = 3^(k - 1)) ∧
  (totalButtons n = (3^n - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_buttons_pattern_l3113_311323


namespace NUMINAMATH_CALUDE_largest_prime_factor_l3113_311316

theorem largest_prime_factor : 
  (∃ (p : ℕ), Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 15^4) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → q ≤ p) ∧
  (Nat.Prime 241 ∧ 241 ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l3113_311316


namespace NUMINAMATH_CALUDE_stamps_needed_l3113_311371

/-- The weight of one piece of paper in ounces -/
def paper_weight : ℚ := 1/5

/-- The number of pieces of paper used -/
def num_papers : ℕ := 8

/-- The weight of the envelope in ounces -/
def envelope_weight : ℚ := 2/5

/-- The number of stamps needed per ounce -/
def stamps_per_ounce : ℕ := 1

/-- The theorem stating the number of stamps needed for Jessica's letter -/
theorem stamps_needed : 
  ⌈(num_papers * paper_weight + envelope_weight) * stamps_per_ounce⌉ = 2 := by
  sorry

end NUMINAMATH_CALUDE_stamps_needed_l3113_311371


namespace NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l3113_311395

theorem x_fourth_plus_y_fourth (x y : ℕ+) (h : y * x^2 + x * y^2 = 70) : x^4 + y^4 = 641 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l3113_311395


namespace NUMINAMATH_CALUDE_point_transformation_l3113_311337

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define rotation function
def rotate90ClockwiseAround (center : Point) (p : Point) : Point :=
  let (cx, cy) := center
  let (x, y) := p
  (cx + (y - cy), cy - (x - cx))

-- Define reflection function
def reflectAboutYEqualsX (p : Point) : Point :=
  let (x, y) := p
  (y, x)

-- Main theorem
theorem point_transformation (c d : ℝ) :
  let Q : Point := (c, d)
  let center : Point := (2, 3)
  let finalPoint : Point := (4, -1)
  (reflectAboutYEqualsX (rotate90ClockwiseAround center Q) = finalPoint) →
  (d - c = -1) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3113_311337


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_three_l3113_311386

theorem smallest_digit_divisible_by_three : 
  ∃ (x : ℕ), x < 10 ∧ 
  (526000 + x * 100 + 18) % 3 = 0 ∧
  ∀ (y : ℕ), y < x → y < 10 → (526000 + y * 100 + 18) % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_three_l3113_311386


namespace NUMINAMATH_CALUDE_purple_yellow_ratio_l3113_311360

/-- Represents the number of flowers of each color in the garden -/
structure GardenFlowers where
  yellow : ℕ
  purple : ℕ
  green : ℕ

/-- Conditions of the garden -/
def gardenConditions (g : GardenFlowers) : Prop :=
  g.yellow = 10 ∧
  g.green = (g.yellow + g.purple) / 4 ∧
  g.yellow + g.purple + g.green = 35

/-- Theorem stating the relationship between purple and yellow flowers -/
theorem purple_yellow_ratio (g : GardenFlowers) 
  (h : gardenConditions g) : g.purple * 10 = g.yellow * 18 := by
  sorry

#check purple_yellow_ratio

end NUMINAMATH_CALUDE_purple_yellow_ratio_l3113_311360


namespace NUMINAMATH_CALUDE_shop_earnings_l3113_311305

theorem shop_earnings : 
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1
  let cola_sold : ℕ := 15
  let juice_sold : ℕ := 12
  let water_sold : ℕ := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88
  := by sorry

end NUMINAMATH_CALUDE_shop_earnings_l3113_311305


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l3113_311363

theorem parabola_y_relationship (x₁ x₂ y₁ y₂ : ℝ) : 
  (y₁ = x₁^2 - 3) →  -- Point A lies on the parabola
  (y₂ = x₂^2 - 3) →  -- Point B lies on the parabola
  (0 < x₁) →         -- x₁ is positive
  (x₁ < x₂) →        -- x₁ is less than x₂
  y₁ < y₂ :=         -- Conclusion: y₁ is less than y₂
by sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l3113_311363


namespace NUMINAMATH_CALUDE_power_function_below_identity_l3113_311368

theorem power_function_below_identity (α : ℝ) : 
  (∀ x : ℝ, x > 1 → x^α < x) → α < 1 := by sorry

end NUMINAMATH_CALUDE_power_function_below_identity_l3113_311368


namespace NUMINAMATH_CALUDE_mall_profit_analysis_l3113_311381

def average_daily_sales : ℝ := 20
def profit_per_shirt : ℝ := 40
def additional_sales_per_yuan : ℝ := 2

def daily_profit (x : ℝ) : ℝ :=
  (profit_per_shirt - x) * (average_daily_sales + additional_sales_per_yuan * x)

theorem mall_profit_analysis :
  ∃ (f : ℝ → ℝ),
    (∀ x, daily_profit x = f x) ∧
    (f x = -2 * x^2 + 60 * x + 800) ∧
    (∃ x_max, ∀ x, f x ≤ f x_max ∧ x_max = 15) ∧
    (∃ x1 x2, x1 ≠ x2 ∧ f x1 = 1200 ∧ f x2 = 1200 ∧ (x1 = 10 ∨ x1 = 20) ∧ (x2 = 10 ∨ x2 = 20)) :=
by sorry

end NUMINAMATH_CALUDE_mall_profit_analysis_l3113_311381


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3113_311339

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 689 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3113_311339


namespace NUMINAMATH_CALUDE_diamond_calculation_l3113_311367

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation : 
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3113_311367


namespace NUMINAMATH_CALUDE_distance_is_60km_l3113_311372

/-- A ship's journey relative to a lighthouse -/
structure ShipJourney where
  speed : ℝ
  time : ℝ
  initial_angle : ℝ
  final_angle : ℝ

/-- Calculate the distance between the ship and the lighthouse at the end of the journey -/
def distance_to_lighthouse (journey : ShipJourney) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the distance to the lighthouse is 60 km -/
theorem distance_is_60km (journey : ShipJourney) 
  (h1 : journey.speed = 15)
  (h2 : journey.time = 4)
  (h3 : journey.initial_angle = π / 3)
  (h4 : journey.final_angle = π / 6) :
  distance_to_lighthouse journey = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_60km_l3113_311372


namespace NUMINAMATH_CALUDE_school_attendance_problem_l3113_311313

theorem school_attendance_problem (girls : ℕ) (percentage_increase : ℚ) (boys : ℕ) :
  girls = 5000 →
  percentage_increase = 40 / 100 →
  (boys : ℚ) + percentage_increase * (boys : ℚ) = (boys : ℚ) + (girls : ℚ) →
  boys = 12500 := by
sorry

end NUMINAMATH_CALUDE_school_attendance_problem_l3113_311313


namespace NUMINAMATH_CALUDE_tree_planting_variance_l3113_311399

def group_data : List (Nat × Nat) := [(5, 3), (6, 4), (7, 3)]

def total_groups : Nat := (group_data.map Prod.snd).sum

theorem tree_planting_variance :
  let mean : Rat := (group_data.map (λ (x, y) => x * y)).sum / total_groups
  let variance : Rat := (group_data.map (λ (x, y) => y * ((x : Rat) - mean)^2)).sum / total_groups
  variance = 6/10 := by sorry

end NUMINAMATH_CALUDE_tree_planting_variance_l3113_311399


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3113_311357

/-- Given a function f: ℝ → ℝ with a tangent line 2x - y - 3 = 0 at x = 2,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f 2 → 2*x - y - 3 = 0 ↔ y = f x) : 
    f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3113_311357


namespace NUMINAMATH_CALUDE_line_x_intercept_l3113_311370

/-- The x-intercept of a line passing through (2, -2) and (6, 10) is 8/3 -/
theorem line_x_intercept :
  let p1 : ℝ × ℝ := (2, -2)
  let p2 : ℝ × ℝ := (6, 10)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let x_intercept : ℝ := -b / m
  x_intercept = 8/3 := by
sorry


end NUMINAMATH_CALUDE_line_x_intercept_l3113_311370


namespace NUMINAMATH_CALUDE_no_eight_digit_six_times_l3113_311347

theorem no_eight_digit_six_times : ¬ ∃ (N : ℕ), 
  (10000000 ≤ N) ∧ (N < 100000000) ∧
  (∃ (p q : ℕ), N = 10000 * p + q ∧ q < 10000 ∧ 10000 * q + p = 6 * N) :=
sorry

end NUMINAMATH_CALUDE_no_eight_digit_six_times_l3113_311347


namespace NUMINAMATH_CALUDE_blue_candles_l3113_311332

/-- The number of blue candles on a birthday cake -/
theorem blue_candles (total : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ)
  (h1 : total = 79)
  (h2 : yellow = 27)
  (h3 : red = 14)
  (h4 : blue = total - yellow - red) :
  blue = 38 := by
  sorry

end NUMINAMATH_CALUDE_blue_candles_l3113_311332


namespace NUMINAMATH_CALUDE_inequality_proof_l3113_311358

theorem inequality_proof (a b c d : ℝ) 
  (h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d)
  (h_product : (a-b)*(b-c)*(c-d)*(d-a) = -3) :
  (a + b + c + d = 6 → d < 0.36) ∧
  (a^2 + b^2 + c^2 + d^2 = 14 → (a+c)*(b+d) ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3113_311358


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3113_311390

/-- Given a geometric sequence of positive integers where the first term is 5 and the fourth term is 500,
    prove that the third term is equal to 5 * 100^(2/3). -/
theorem geometric_sequence_third_term :
  ∀ (seq : ℕ → ℕ),
    (∀ n, seq (n + 1) / seq n = seq 2 / seq 1) →  -- Geometric sequence condition
    seq 1 = 5 →                                   -- First term is 5
    seq 4 = 500 →                                 -- Fourth term is 500
    seq 3 = 5 * 100^(2/3) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3113_311390


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_square_l3113_311392

theorem sum_and_reciprocal_square (m : ℝ) (h : m + 1/m = 5) : 
  m^2 + 1/m^2 + m + 1/m = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_square_l3113_311392


namespace NUMINAMATH_CALUDE_ellipse_inscribed_circle_radius_specific_ellipse_inscribed_circle_radius_l3113_311354

/-- The radius of the largest circle inside an ellipse with its center at a focus --/
theorem ellipse_inscribed_circle_radius (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let c := Real.sqrt (a^2 - b^2)
  let r := c - b
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x - c)^2 + y^2 ≥ r^2) ∧
  (∃ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) :=
by sorry

/-- The specific case for the given ellipse --/
theorem specific_ellipse_inscribed_circle_radius :
  let a : ℝ := 6
  let b : ℝ := 5
  let c := Real.sqrt (a^2 - b^2)
  let r := c - a
  r = Real.sqrt 11 - 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_circle_radius_specific_ellipse_inscribed_circle_radius_l3113_311354


namespace NUMINAMATH_CALUDE_tangent_line_circle_minimum_l3113_311364

theorem tangent_line_circle_minimum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2 ∧ 
    ∀ x' y' : ℝ, (x' - b)^2 + (y' - 1)^2 ≤ 2 → (x' + y' + a)^2 > 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x y : ℝ, x + y + a' = 0 ∧ (x - b')^2 + (y - 1)^2 = 2 ∧ 
      ∀ x' y' : ℝ, (x' - b')^2 + (y' - 1)^2 ≤ 2 → (x' + y' + a')^2 > 0) → 
    (3 - 2*b')^2 / (2*a') ≥ (3 - 2*b)^2 / (2*a)) →
  (3 - 2*b)^2 / (2*a) = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_minimum_l3113_311364


namespace NUMINAMATH_CALUDE_reflection_path_exists_l3113_311355

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a path from A to B with two reflections in a triangle -/
structure ReflectionPath (t : Triangle) where
  P : Point -- Point of reflection on side BC
  Q : Point -- Point of reflection on side CA

/-- Angle at vertex C of a triangle -/
def angle_C (t : Triangle) : ℝ := sorry

/-- Theorem stating the condition for the existence of a reflection path -/
theorem reflection_path_exists (t : Triangle) : 
  (∃ path : ReflectionPath t, True) ↔ (π/4 < angle_C t ∧ angle_C t < π/3) := by sorry

end NUMINAMATH_CALUDE_reflection_path_exists_l3113_311355


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3113_311310

/-- A quadratic expression in x and y with a parameter k -/
def quadratic (x y : ℝ) (k : ℝ) : ℝ := 2 * x^2 - 6 * y^2 + x * y + k * x + 6

/-- Predicate to check if an expression is a product of two linear factors -/
def is_product_of_linear_factors (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d e g : ℝ), ∀ x y, f x y = (a * x + b * y + c) * (d * x + e * y + g)

/-- Theorem stating that if the quadratic expression is factorizable, then k = 7 or k = -7 -/
theorem quadratic_factorization (k : ℝ) :
  is_product_of_linear_factors (quadratic · · k) → k = 7 ∨ k = -7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3113_311310


namespace NUMINAMATH_CALUDE_ratio_problem_l3113_311356

theorem ratio_problem (x : ℝ) : 
  (5 : ℝ) * x = 60 → x = 12 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3113_311356


namespace NUMINAMATH_CALUDE_mary_warmth_hours_l3113_311324

/-- The number of sticks of wood produced by chopping up furniture and the number of hours Mary can keep warm. -/
def furniture_to_warmth (chair_sticks table_sticks cabinet_sticks stool_sticks : ℕ)
  (chairs tables cabinets stools : ℕ) (sticks_per_hour : ℕ) : ℕ :=
  let total_sticks := chair_sticks * chairs + table_sticks * tables + 
                      cabinet_sticks * cabinets + stool_sticks * stools
  total_sticks / sticks_per_hour

/-- Theorem stating that Mary can keep warm for 64 hours given the specified conditions. -/
theorem mary_warmth_hours : 
  furniture_to_warmth 8 12 16 3 25 12 5 8 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_mary_warmth_hours_l3113_311324


namespace NUMINAMATH_CALUDE_ladder_height_proof_l3113_311322

def ceiling_height : ℝ := 300
def fixture_below_ceiling : ℝ := 15
def alice_height : ℝ := 170
def alice_normal_reach : ℝ := 55
def extra_reach_needed : ℝ := 5

theorem ladder_height_proof :
  let fixture_height := ceiling_height - fixture_below_ceiling
  let total_reach_needed := fixture_height
  let alice_max_reach := alice_height + alice_normal_reach + extra_reach_needed
  let ladder_height := total_reach_needed - alice_max_reach
  ladder_height = 60 := by sorry

end NUMINAMATH_CALUDE_ladder_height_proof_l3113_311322


namespace NUMINAMATH_CALUDE_chess_games_count_l3113_311352

/-- The number of combinations of k items from a set of n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of players in the chess group -/
def num_players : ℕ := 50

/-- The number of players in each game -/
def players_per_game : ℕ := 2

theorem chess_games_count : combinations num_players players_per_game = 1225 := by
  sorry

end NUMINAMATH_CALUDE_chess_games_count_l3113_311352


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_l3113_311348

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distributeDistinguishableBalls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes -/
theorem seven_balls_two_boxes :
  distributeDistinguishableBalls 7 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_l3113_311348


namespace NUMINAMATH_CALUDE_circle_equation_l3113_311345

-- Define the circle
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 5}

-- Define the line
def Line := {(x, y) : ℝ × ℝ | x - 2*y = 0}

-- Theorem statement
theorem circle_equation :
  ∃ (a : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ Circle a → (x - a)^2 + y^2 = 5) ∧ 
    (∃ (x y : ℝ), (x, y) ∈ Circle a ∩ Line) ∧
    (a = 5 ∨ a = -5) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3113_311345


namespace NUMINAMATH_CALUDE_laptop_cost_ratio_l3113_311383

theorem laptop_cost_ratio : 
  ∀ (first_laptop_cost second_laptop_cost : ℝ),
    first_laptop_cost = 500 →
    first_laptop_cost + second_laptop_cost = 2000 →
    second_laptop_cost / first_laptop_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_laptop_cost_ratio_l3113_311383


namespace NUMINAMATH_CALUDE_puzzle_spells_bach_l3113_311317

/-- Represents a musical symbol --/
inductive MusicalSymbol
  | DoubleFlatSolKey
  | ATenorClef
  | CAltoClef
  | BNaturalSolKey

/-- Represents the interpretation rules --/
def interpretSymbol (s : MusicalSymbol) : Char :=
  match s with
  | MusicalSymbol.DoubleFlatSolKey => 'B'
  | MusicalSymbol.ATenorClef => 'A'
  | MusicalSymbol.CAltoClef => 'C'
  | MusicalSymbol.BNaturalSolKey => 'H'

/-- The sequence of symbols in the puzzle --/
def puzzleSequence : List MusicalSymbol := [
  MusicalSymbol.DoubleFlatSolKey,
  MusicalSymbol.ATenorClef,
  MusicalSymbol.CAltoClef,
  MusicalSymbol.BNaturalSolKey
]

/-- The theorem stating that the puzzle sequence spells "BACH" --/
theorem puzzle_spells_bach :
  puzzleSequence.map interpretSymbol = ['B', 'A', 'C', 'H'] := by
  sorry


end NUMINAMATH_CALUDE_puzzle_spells_bach_l3113_311317


namespace NUMINAMATH_CALUDE_flyers_left_to_hand_out_l3113_311385

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed : ℕ) 
  (rose_handed : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed = 120)
  (h3 : rose_handed = 320) : 
  total_flyers - (jack_handed + rose_handed) = 796 :=
by sorry

end NUMINAMATH_CALUDE_flyers_left_to_hand_out_l3113_311385


namespace NUMINAMATH_CALUDE_total_buildable_area_l3113_311375

def num_sections : ℕ := 7
def section_area : ℝ := 9473
def open_space_percent : ℝ := 0.15

theorem total_buildable_area :
  (num_sections : ℝ) * section_area * (1 - open_space_percent) = 56364.35 := by
  sorry

end NUMINAMATH_CALUDE_total_buildable_area_l3113_311375


namespace NUMINAMATH_CALUDE_prime_difference_divisibility_l3113_311349

theorem prime_difference_divisibility (n : ℕ) : 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n ∣ (p - q) := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_divisibility_l3113_311349


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3113_311325

def f (x : ℝ) : ℝ := x^2 + 2

theorem derivative_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3113_311325


namespace NUMINAMATH_CALUDE_dice_rolling_expectation_l3113_311366

/-- The expected value of 6^D after n steps in the dice rolling process -/
def expected_value (n : ℕ) : ℝ :=
  6 + 5 * n

/-- The number of steps in the process -/
def num_steps : ℕ := 2013

theorem dice_rolling_expectation :
  expected_value num_steps = 10071 := by
  sorry

end NUMINAMATH_CALUDE_dice_rolling_expectation_l3113_311366


namespace NUMINAMATH_CALUDE_cut_rectangle_corners_l3113_311331

/-- A shape created by cutting off one corner of a rectangle --/
structure CutRectangle where
  originalCorners : Nat
  cutCorners : Nat
  newCorners : Nat

/-- Properties of a rectangle with one corner cut off --/
def isValidCutRectangle (r : CutRectangle) : Prop :=
  r.originalCorners = 4 ∧
  r.cutCorners = 1 ∧
  r.newCorners = r.originalCorners + r.cutCorners

/-- Theorem: A rectangle with one corner cut off has 5 corners --/
theorem cut_rectangle_corners (r : CutRectangle) (h : isValidCutRectangle r) :
  r.newCorners = 5 := by
  sorry

#check cut_rectangle_corners

end NUMINAMATH_CALUDE_cut_rectangle_corners_l3113_311331


namespace NUMINAMATH_CALUDE_team_a_win_probability_l3113_311335

/-- Probability of Team A winning a non-fifth set -/
def p : ℚ := 2/3

/-- Probability of Team A winning the fifth set -/
def p_fifth : ℚ := 1/2

/-- The probability of Team A winning the volleyball match -/
theorem team_a_win_probability : 
  (p^3) + (3 * p^2 * (1-p) * p) + (6 * p^2 * (1-p)^2 * p_fifth) = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l3113_311335


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l3113_311382

theorem consecutive_product_not_power (n m : ℕ) (h : m ≥ 2) :
  ¬ ∃ a : ℕ, n * (n + 1) = a ^ m :=
sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l3113_311382


namespace NUMINAMATH_CALUDE_arcsin_zero_l3113_311304

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_zero_l3113_311304


namespace NUMINAMATH_CALUDE_unique_number_with_sum_of_largest_divisors_3333_l3113_311330

/-- The largest divisor of a natural number is the number itself -/
def largest_divisor (n : ℕ) : ℕ := n

/-- The second largest divisor of an even natural number is half of the number -/
def second_largest_divisor (n : ℕ) : ℕ := n / 2

/-- The property that the sum of the two largest divisors of n is 3333 -/
def sum_of_largest_divisors_is_3333 (n : ℕ) : Prop :=
  largest_divisor n + second_largest_divisor n = 3333

theorem unique_number_with_sum_of_largest_divisors_3333 :
  ∀ n : ℕ, sum_of_largest_divisors_is_3333 n → n = 2222 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_sum_of_largest_divisors_3333_l3113_311330


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3113_311369

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := 6 * x^2 - 12 * x + 4

-- Define the transformed expression
def transformed (x a h k : ℝ) : ℝ := a * (x - h)^2 + k

-- Theorem statement
theorem quadratic_transformation :
  ∃ (a h k : ℝ), (∀ x, quadratic x = transformed x a h k) ∧ (a + h + k = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3113_311369


namespace NUMINAMATH_CALUDE_equation_solution_l3113_311334

theorem equation_solution (x : ℝ) :
  8.438 * Real.cos (x - π/4) * (1 - 4 * Real.cos (2*x)^2) - 2 * Real.cos (4*x) = 3 →
  ∃ k : ℤ, x = π/4 * (8*k + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3113_311334


namespace NUMINAMATH_CALUDE_additional_weight_needed_l3113_311307

/-- Calculates the additional weight needed to open the cave doors -/
theorem additional_weight_needed 
  (set1_weight : ℝ) 
  (set1_count : ℕ) 
  (set2_weight : ℝ) 
  (set2_count : ℕ) 
  (switch_weight : ℝ) 
  (total_needed : ℝ) 
  (large_rock_kg : ℝ) 
  (kg_to_lbs : ℝ) 
  (h1 : set1_weight = 60) 
  (h2 : set1_count = 3) 
  (h3 : set2_weight = 42) 
  (h4 : set2_count = 5) 
  (h5 : switch_weight = 234) 
  (h6 : total_needed = 712) 
  (h7 : large_rock_kg = 12) 
  (h8 : kg_to_lbs = 2.2) : 
  total_needed - (switch_weight + set1_weight * set1_count + set2_weight * set2_count + large_rock_kg * kg_to_lbs) = 61.6 := by
  sorry

#check additional_weight_needed

end NUMINAMATH_CALUDE_additional_weight_needed_l3113_311307


namespace NUMINAMATH_CALUDE_odd_heads_probability_l3113_311327

/-- The probability of getting heads for the kth coin -/
def p (k : ℕ) : ℚ := 1 / (2 * k + 1)

/-- The probability of getting an odd number of heads when tossing n biased coins -/
def odd_heads_prob (n : ℕ) : ℚ := n / (2 * n + 1)

/-- Theorem stating that the probability of getting an odd number of heads
    when tossing n biased coins is n/(2n+1), where the kth coin has
    probability 1/(2k+1) of falling heads -/
theorem odd_heads_probability (n : ℕ) :
  (∀ k, k ≤ n → p k = 1 / (2 * k + 1)) →
  odd_heads_prob n = n / (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_heads_probability_l3113_311327


namespace NUMINAMATH_CALUDE_obtuse_isosceles_triangle_vertex_angle_l3113_311393

/-- An obtuse isosceles triangle with the given property has a vertex angle of 150° -/
theorem obtuse_isosceles_triangle_vertex_angle 
  (a : ℝ) 
  (θ : ℝ) 
  (h_a_pos : a > 0)
  (h_θ_pos : θ > 0)
  (h_θ_acute : θ < π / 2)
  (h_isosceles : a^2 = (2 * a * Real.cos θ) * (2 * a * Real.sin θ)) :
  π - 2*θ = 5*π/6 := by
sorry

end NUMINAMATH_CALUDE_obtuse_isosceles_triangle_vertex_angle_l3113_311393


namespace NUMINAMATH_CALUDE_prob_two_red_marbles_l3113_311389

/-- The probability of selecting two red marbles without replacement from a bag containing 2 red marbles and 3 green marbles is 1/10. -/
theorem prob_two_red_marbles (red : ℕ) (green : ℕ) (h1 : red = 2) (h2 : green = 3) :
  (red / (red + green)) * ((red - 1) / (red + green - 1)) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_marbles_l3113_311389


namespace NUMINAMATH_CALUDE_student_event_combinations_l3113_311396

theorem student_event_combinations : 
  let num_students : ℕ := 4
  let num_events : ℕ := 3
  num_events ^ num_students = 81 := by sorry

end NUMINAMATH_CALUDE_student_event_combinations_l3113_311396


namespace NUMINAMATH_CALUDE_semicircle_curve_length_l3113_311338

open Real

/-- The length of the curve traced by point D in a semicircle configuration --/
theorem semicircle_curve_length (k : ℝ) (h : k > 0) :
  ∃ (curve_length : ℝ),
    (∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 →
      let C : ℝ × ℝ := (cos (2 * θ), sin (2 * θ))
      let D : ℝ × ℝ := (cos (2 * θ) + k * sin (2 * θ), sin (2 * θ) + k * (1 - cos (2 * θ)))
      (D.1 ^ 2 + (D.2 - k) ^ 2 = 1 + k ^ 2) ∧
      (k * D.1 + D.2 ≥ k)) →
    curve_length = π * sqrt (1 + k ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_curve_length_l3113_311338


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l3113_311365

theorem probability_of_specific_arrangement (n : ℕ) (r : ℕ) : 
  n = 4 → r = 2 → (1 : ℚ) / (n! / r!) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l3113_311365


namespace NUMINAMATH_CALUDE_stack_logs_count_l3113_311361

/-- The number of logs in a stack with arithmetic progression of rows -/
def logsInStack (bottomRow : ℕ) (topRow : ℕ) : ℕ :=
  let numRows := bottomRow - topRow + 1
  numRows * (bottomRow + topRow) / 2

theorem stack_logs_count : logsInStack 15 5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_stack_logs_count_l3113_311361


namespace NUMINAMATH_CALUDE_cost_comparison_l3113_311326

/-- The price of a suit in yuan -/
def suit_price : ℕ := 1000

/-- The price of a tie in yuan -/
def tie_price : ℕ := 200

/-- The number of suits to be purchased -/
def num_suits : ℕ := 20

/-- The discount rate for Option 2 -/
def discount_rate : ℚ := 9/10

/-- The cost calculation for Option 1 -/
def option1_cost (x : ℕ) : ℕ := 
  num_suits * suit_price + (x - num_suits) * tie_price

/-- The cost calculation for Option 2 -/
def option2_cost (x : ℕ) : ℚ := 
  discount_rate * (num_suits * suit_price + x * tie_price)

theorem cost_comparison (x : ℕ) (h : x > num_suits) : 
  option1_cost x = 200 * x + 16000 ∧ 
  option2_cost x = 180 * x + 18000 := by
  sorry

#check cost_comparison

end NUMINAMATH_CALUDE_cost_comparison_l3113_311326


namespace NUMINAMATH_CALUDE_equation_solution_l3113_311387

theorem equation_solution :
  ∃ x : ℝ, (5 * x - 8 * (2 * x + 3) = 4 * (x - 3 * (2 * x - 5)) + 7 * (2 * x - 5)) ∧ x = -9.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3113_311387


namespace NUMINAMATH_CALUDE_v_2008_eq_352_l3113_311343

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ 
| 0 => 1  -- First term
| n + 1 => 
  let group := (Nat.sqrt (8 * (n + 1) + 1) - 1) / 2  -- Determine which group n+1 belongs to
  let groupStart := group * (group + 1) / 2  -- Starting position of the group
  let offset := n + 1 - groupStart  -- Position within the group
  (group + 1) + 3 * ((groupStart - 1) + offset)  -- Calculate the term

/-- The 2008th term of the sequence is 352 -/
theorem v_2008_eq_352 : v 2007 = 352 := by
  sorry

end NUMINAMATH_CALUDE_v_2008_eq_352_l3113_311343


namespace NUMINAMATH_CALUDE_min_modulus_m_for_real_roots_l3113_311300

/-- Given a complex number m such that the equation x^2 + mx + 1 + 2i = 0 has real roots,
    the minimum value of |m| is sqrt(2 + 2sqrt(5)). -/
theorem min_modulus_m_for_real_roots (m : ℂ) : 
  (∃ x : ℝ, x^2 + m * x + (1 : ℂ) + 2*I = 0) → 
  ∀ m' : ℂ, (∃ x : ℝ, x^2 + m' * x + (1 : ℂ) + 2*I = 0) → Complex.abs m ≤ Complex.abs m' → 
  Complex.abs m = Real.sqrt (2 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_m_for_real_roots_l3113_311300


namespace NUMINAMATH_CALUDE_f_inequality_l3113_311391

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 1 → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (x + 1) = f (-x + 1))

theorem f_inequality (f : ℝ → ℝ) (h : f_conditions f) :
  f 5.5 < f 7.8 ∧ f 7.8 < f (-2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3113_311391


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3113_311397

theorem quadratic_equation_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*(a-1)*x + a^2 - 7*a - 4 = 0 ↔ (x = x₁ ∨ x = x₂)) →
  x₁*x₂ - 3*x₁ - 3*x₂ - 2 = 0 →
  (1 + 4/(a^2 - 4)) * (a + 2)/a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l3113_311397


namespace NUMINAMATH_CALUDE_game_outcomes_l3113_311321

/-- The game state -/
inductive GameState
| A (n : ℕ)  -- Player A's turn with current number n
| B (n : ℕ)  -- Player B's turn with current number n

/-- The possible outcomes of the game -/
inductive Outcome
| AWin  -- Player A wins
| BWin  -- Player B wins
| Draw  -- Neither player has a winning strategy

/-- Definition of a winning strategy for a player -/
def has_winning_strategy (player : GameState → Prop) (s : GameState) : Prop :=
  ∃ (strategy : GameState → ℕ), 
    ∀ (game : ℕ → GameState),
      game 0 = s →
      (∀ n, player (game n) → game (n + 1) = GameState.B (strategy (game n))) →
      (∃ m, game m = GameState.A 1990 ∨ game m = GameState.B 1)

/-- The main theorem about the game outcomes -/
theorem game_outcomes (n₀ : ℕ) : 
  (has_winning_strategy (λ s => ∃ n, s = GameState.A n) (GameState.A n₀) ↔ n₀ ≥ 8) ∧
  (has_winning_strategy (λ s => ∃ n, s = GameState.B n) (GameState.A n₀) ↔ n₀ ≤ 5) ∧
  (¬ has_winning_strategy (λ s => ∃ n, s = GameState.A n) (GameState.A n₀) ∧
   ¬ has_winning_strategy (λ s => ∃ n, s = GameState.B n) (GameState.A n₀) ↔ n₀ = 6 ∨ n₀ = 7) :=
sorry

end NUMINAMATH_CALUDE_game_outcomes_l3113_311321


namespace NUMINAMATH_CALUDE_exists_unique_equal_power_point_equal_power_point_is_orthogonal_circle_center_l3113_311362

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The power of a point with respect to a circle -/
def powerOfPoint (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

/-- Theorem: Given three circles, there exists a unique point with equal power to all three circles -/
theorem exists_unique_equal_power_point (c1 c2 c3 : Circle) :
  ∃! p : ℝ × ℝ, powerOfPoint p c1 = powerOfPoint p c2 ∧ powerOfPoint p c2 = powerOfPoint p c3 :=
sorry

/-- Theorem: The point with equal power to three circles is the center of a circle 
    that intersects the three given circles at right angles -/
theorem equal_power_point_is_orthogonal_circle_center 
  (c1 c2 c3 : Circle) (p : ℝ × ℝ) 
  (h : powerOfPoint p c1 = powerOfPoint p c2 ∧ powerOfPoint p c2 = powerOfPoint p c3) :
  ∃ r : ℝ, ∀ i : Fin 3, 
    let c := Circle.mk p r
    let ci := [c1, c2, c3].get i
    ∃ x : ℝ × ℝ, (x.1 - c.center.1)^2 + (x.2 - c.center.2)^2 = c.radius^2 ∧
                 (x.1 - ci.center.1)^2 + (x.2 - ci.center.2)^2 = ci.radius^2 ∧
                 ((x.1 - c.center.1) * (x.1 - ci.center.1) + (x.2 - c.center.2) * (x.2 - ci.center.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_exists_unique_equal_power_point_equal_power_point_is_orthogonal_circle_center_l3113_311362


namespace NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l3113_311320

/-- Given a triangle ABC with sides a, b, c, area S, where a² + b² - c² = 4√3 * S and c = 1,
    the area of its circumcircle is π. -/
theorem circumcircle_area_of_special_triangle (a b c S : ℝ) : 
  a > 0 → b > 0 → c > 0 → S > 0 →
  a^2 + b^2 - c^2 = 4 * Real.sqrt 3 * S →
  c = 1 →
  ∃ (R : ℝ), R > 0 ∧ π * R^2 = π := by sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l3113_311320


namespace NUMINAMATH_CALUDE_prime_power_plus_144_square_l3113_311340

theorem prime_power_plus_144_square (p n m : ℕ) : 
  p.Prime → p > 0 → n > 0 → m > 0 → p^n + 144 = m^2 → 
  (p = 5 ∧ n = 2 ∧ m = 13) ∨ (p = 2 ∧ n = 8 ∧ m = 20) ∨ (p = 3 ∧ n = 4 ∧ m = 15) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_plus_144_square_l3113_311340


namespace NUMINAMATH_CALUDE_simplify_fraction_l3113_311342

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + 2 * Real.sqrt 18) = 5 * Real.sqrt 2 / 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3113_311342


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3113_311306

theorem product_sum_theorem :
  ∀ (a b c d : ℝ),
  (∀ x : ℝ, (5 * x^2 - 3 * x + 7) * (9 - 4 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = -29 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3113_311306


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l3113_311394

/-- The probability of picking 3 red balls from a bag containing 7 red, 9 blue, and 5 green balls -/
theorem probability_three_red_balls (red blue green : ℕ) (total : ℕ) : 
  red = 7 → blue = 9 → green = 5 → total = red + blue + green →
  (red / total) * ((red - 1) / (total - 1)) * ((red - 2) / (total - 2)) = 1 / 38 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l3113_311394


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3113_311379

theorem max_value_of_expression (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 3) :
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 1 ∧
  ∃ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 3 ∧
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3113_311379


namespace NUMINAMATH_CALUDE_max_term_T_l3113_311302

def geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ+ → ℚ :=
  fun n => a₁ * q ^ (n.val - 1)

def sum_geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ+ → ℚ :=
  fun n => a₁ * (1 - q^n.val) / (1 - q)

def T (S : ℕ+ → ℚ) : ℕ+ → ℚ :=
  fun n => S n + 1 / (S n)

theorem max_term_T 
  (a : ℕ+ → ℚ)
  (S : ℕ+ → ℚ)
  (h₁ : a 1 = 3/2)
  (h₂ : ∀ n, S n = sum_geometric_sequence (a 1) (-1/2) n)
  (h₃ : -2*(S 2) + 4*(S 4) = 2*(S 3))
  : ∀ n, T S n ≤ 13/6 ∧ T S 1 = 13/6 :=
sorry

end NUMINAMATH_CALUDE_max_term_T_l3113_311302


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3113_311353

theorem no_positive_integer_solutions :
  ¬∃ (x y z : ℕ+), x^2 + y^2 = 7 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3113_311353


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3113_311388

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2*x^2 - 7 = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3113_311388


namespace NUMINAMATH_CALUDE_unique_triple_l3113_311359

theorem unique_triple : 
  ∃! (A B C : ℕ), A^2 + B - C = 100 ∧ A + B^2 - C = 124 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_triple_l3113_311359


namespace NUMINAMATH_CALUDE_sequence_problem_l3113_311374

def second_difference (a : ℕ → ℤ) : ℕ → ℤ := λ n => a (n + 2) - 2 * a (n + 1) + a n

theorem sequence_problem (a : ℕ → ℤ) 
  (h1 : ∀ n, second_difference a n = 16)
  (h2 : a 63 = 10)
  (h3 : a 89 = 10) :
  a 51 = 3658 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3113_311374


namespace NUMINAMATH_CALUDE_biology_marks_proof_l3113_311308

def english_marks : ℕ := 72
def math_marks : ℕ := 45
def physics_marks : ℕ := 72
def chemistry_marks : ℕ := 77
def average_marks : ℚ := 68.2
def total_subjects : ℕ := 5

theorem biology_marks_proof :
  ∃ (biology_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks ∧
    biology_marks = 75 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_proof_l3113_311308


namespace NUMINAMATH_CALUDE_largest_quotient_is_15_l3113_311380

def S : Set Int := {-30, -4, -2, 2, 4, 10}

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem largest_quotient_is_15 :
  ∀ a b : Int,
    a ∈ S → b ∈ S →
    is_even a → is_even b →
    a < 0 → b > 0 →
    (-a : ℚ) / b ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_is_15_l3113_311380


namespace NUMINAMATH_CALUDE_cosine_arithmetic_sequence_product_l3113_311351

theorem cosine_arithmetic_sequence_product (a : ℕ → ℝ) (S : Set ℝ) (a₀ b₀ : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + 2 * π / 3) →
  S = {x | ∃ n : ℕ, x = Real.cos (a n)} →
  S = {a₀, b₀} →
  a₀ * b₀ = -1/2 :=
sorry

end NUMINAMATH_CALUDE_cosine_arithmetic_sequence_product_l3113_311351


namespace NUMINAMATH_CALUDE_twentieth_is_thursday_l3113_311303

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a month with the given condition -/
structure Month where
  dates : List Date
  threeSundaysOnEvenDates : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.dayOfWeek = DayOfWeek.Sunday ∧ d2.dayOfWeek = DayOfWeek.Sunday ∧ d3.dayOfWeek = DayOfWeek.Sunday ∧
    d1.day % 2 = 0 ∧ d2.day % 2 = 0 ∧ d3.day % 2 = 0

/-- Theorem stating that the 20th is a Thursday in a month with three Sundays on even dates -/
theorem twentieth_is_thursday (m : Month) : 
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 20 ∧ d.dayOfWeek = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_twentieth_is_thursday_l3113_311303


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3113_311384

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3113_311384


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3113_311373

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_mersenne_prime (m : Nat) : Prop :=
  ∃ n : Nat, is_prime n ∧ m = 2^n - 1 ∧ is_prime m

theorem largest_mersenne_prime_under_500 :
  ∀ m : Nat, is_mersenne_prime m → m < 500 → m ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3113_311373


namespace NUMINAMATH_CALUDE_fill_675_cans_in_36_minutes_l3113_311376

/-- A machine that fills cans of paint -/
structure PaintMachine where
  cans_per_batch : ℕ
  minutes_per_batch : ℕ

/-- Calculate the time needed to fill a given number of cans -/
def time_to_fill (machine : PaintMachine) (total_cans : ℕ) : ℕ :=
  (total_cans * machine.minutes_per_batch + machine.cans_per_batch - 1) / machine.cans_per_batch

/-- Theorem stating that it takes 36 minutes to fill 675 cans -/
theorem fill_675_cans_in_36_minutes :
  let machine : PaintMachine := { cans_per_batch := 150, minutes_per_batch := 8 }
  time_to_fill machine 675 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fill_675_cans_in_36_minutes_l3113_311376


namespace NUMINAMATH_CALUDE_students_in_same_group_l3113_311314

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The probability that both students are in the same group -/
def prob_same_group : ℚ := num_groups * (prob_join_group * prob_join_group)

theorem students_in_same_group :
  prob_same_group = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_students_in_same_group_l3113_311314


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3113_311312

/-- A quadratic function f(x) = x^2 + 4x + c, where c is a constant. -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- Theorem stating that for the quadratic function f(x) = x^2 + 4x + c,
    the inequality f(1) > f(0) > f(-2) holds for any constant c. -/
theorem quadratic_inequality (c : ℝ) : f c 1 > f c 0 ∧ f c 0 > f c (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3113_311312


namespace NUMINAMATH_CALUDE_rectangle_to_square_dimension_l3113_311344

/-- Given a rectangle with dimensions 10 and 15, when cut into two congruent hexagons
    and repositioned to form a square, half the length of the square's side is (5√6)/2. -/
theorem rectangle_to_square_dimension (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (square_side : ℝ) (y : ℝ) :
  rectangle_width = 10 →
  rectangle_height = 15 →
  square_side^2 = rectangle_width * rectangle_height →
  y = square_side / 2 →
  y = (5 * Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_dimension_l3113_311344


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3113_311350

/-- An isosceles triangle with side lengths a, b, and c, where two sides are 11 and one side is 5 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_isosceles : (a = b ∧ a = 11 ∧ c = 5) ∨ (a = c ∧ a = 11 ∧ b = 5) ∨ (b = c ∧ b = 11 ∧ a = 5)
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an isosceles triangle with two sides of length 11 and one side of length 5 is 27 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.a + t.b + t.c = 27 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3113_311350


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3113_311333

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3113_311333


namespace NUMINAMATH_CALUDE_wilson_fraction_problem_l3113_311336

theorem wilson_fraction_problem (N F : ℚ) : 
  N = 8 → N - F * N = 16/3 → F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_wilson_fraction_problem_l3113_311336


namespace NUMINAMATH_CALUDE_reseating_twelve_women_l3113_311346

def reseating_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- Consider the empty case as 1
  | 1 => 1
  | 2 => 3
  | n + 3 => reseating_ways (n + 2) + reseating_ways (n + 1) + reseating_ways n

theorem reseating_twelve_women :
  reseating_ways 12 = 1201 := by
  sorry

end NUMINAMATH_CALUDE_reseating_twelve_women_l3113_311346


namespace NUMINAMATH_CALUDE_wire_length_proof_l3113_311315

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  shorter_piece = (2 / 7) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 90 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l3113_311315


namespace NUMINAMATH_CALUDE_percentage_of_male_students_l3113_311377

theorem percentage_of_male_students (male_percentage : ℝ) 
  (h1 : 0 ≤ male_percentage ∧ male_percentage ≤ 100)
  (h2 : 50 = 100 * (1 - (male_percentage / 100 * 0.5 + (100 - male_percentage) / 100 * 0.6)))
  : male_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_students_l3113_311377


namespace NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l3113_311378

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l3113_311378


namespace NUMINAMATH_CALUDE_exist_a_b_l3113_311328

theorem exist_a_b : ∃ (a b : ℝ),
  (a < 0) ∧
  (b = -a) ∧
  (b > 9/4) ∧
  (∀ x : ℝ, x < -1 → a * x > b) ∧
  (∀ y : ℝ, y^2 + 3*y + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_exist_a_b_l3113_311328
