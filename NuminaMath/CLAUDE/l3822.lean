import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3822_382262

/-- Two lines are parallel if and only if their slopes are equal and they are not the same line -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ (m₁, n₁, c₁) ≠ (m₂, n₂, c₂)

/-- The theorem states that a = 3 is a necessary and sufficient condition for the given lines to be parallel -/
theorem parallel_lines_condition (a : ℝ) :
  are_parallel a 2 (3*a) 3 (a-1) (a-7) ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3822_382262


namespace NUMINAMATH_CALUDE_circle_radius_doubled_l3822_382204

theorem circle_radius_doubled (r n : ℝ) : 
  (2 * π * (r + n) = 2 * (2 * π * r)) → r = n :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_doubled_l3822_382204


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l3822_382227

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x = 1) 
  (hy : y^3 - 3*y^2 + 5*y = 5) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l3822_382227


namespace NUMINAMATH_CALUDE_january_book_sales_l3822_382268

/-- Proves that the number of books sold in January is 15, given the sales in February and March,
    and the average sales across all three months. -/
theorem january_book_sales (february_sales march_sales : ℕ) (average_sales : ℚ)
  (h1 : february_sales = 16)
  (h2 : march_sales = 17)
  (h3 : average_sales = 16)
  (h4 : (january_sales + february_sales + march_sales : ℚ) / 3 = average_sales) :
  january_sales = 15 := by
  sorry

end NUMINAMATH_CALUDE_january_book_sales_l3822_382268


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l3822_382299

def calculate_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (coupon : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_coupon := price_after_discount2 - coupon
  let final_price := price_after_coupon * (1 + tax_rate)
  final_price

theorem jacket_price_calculation :
  calculate_final_price 150 0.25 0.10 10 0.10 = 100.38 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l3822_382299


namespace NUMINAMATH_CALUDE_sum_of_z_values_l3822_382225

-- Define the function g
def g (x : ℝ) : ℝ := (4 * x)^2 - (4 * x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, g z₁ = 8 ∧ g z₂ = 8 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/16) :=
sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l3822_382225


namespace NUMINAMATH_CALUDE_collisions_100_balls_l3822_382294

/-- The number of collisions between n identical balls moving along a single dimension,
    where each pair of balls can collide exactly once. -/
def numCollisions (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 100 balls, the number of collisions is 4950. -/
theorem collisions_100_balls :
  numCollisions 100 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_collisions_100_balls_l3822_382294


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3822_382249

/-- Atomic weight of Calcium -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Nitrogen -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Carbon-12 -/
def C12_weight : ℝ := 12.00

/-- Atomic weight of Carbon-13 -/
def C13_weight : ℝ := 13.003

/-- Percentage of Carbon-12 in the compound -/
def C12_percentage : ℝ := 0.95

/-- Percentage of Carbon-13 in the compound -/
def C13_percentage : ℝ := 0.05

/-- Average atomic weight of Carbon in the compound -/
def C_avg_weight : ℝ := C12_percentage * C12_weight + C13_percentage * C13_weight

/-- Number of Calcium atoms in the compound -/
def Ca_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def C_count : ℕ := 1

/-- Molecular weight of the compound -/
def molecular_weight : ℝ :=
  Ca_count * Ca_weight + O_count * O_weight + H_count * H_weight +
  N_count * N_weight + C_count * C_avg_weight

theorem compound_molecular_weight :
  molecular_weight = 156.22615 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3822_382249


namespace NUMINAMATH_CALUDE_ellipse_x_intersection_l3822_382298

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y - 3)^2) + Real.sqrt ((x - 4)^2 + (y - 1)^2) = 10

/-- Theorem stating the other x-axis intersection point of the ellipse -/
theorem ellipse_x_intersection :
  ∃ x : ℝ, x = 1 + Real.sqrt 40 ∧ ellipse x 0 ∧ ellipse 0 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_x_intersection_l3822_382298


namespace NUMINAMATH_CALUDE_fraction_equality_l3822_382259

theorem fraction_equality : ∀ x : ℝ, x ≠ 0 ∧ x^2 + 1 ≠ 0 →
  (x^2 + 5*x - 6) / (x^4 + x^2) = (-6 : ℝ) / x^2 + (0*x + 7) / (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3822_382259


namespace NUMINAMATH_CALUDE_sphere_radius_calculation_l3822_382280

/-- Given a sphere on a horizontal plane, if a vertical stick casts a shadow and the sphere's shadow extends from its base, then we can calculate the radius of the sphere. -/
theorem sphere_radius_calculation (stick_height stick_shadow sphere_shadow : ℝ) 
  (stick_height_pos : stick_height > 0)
  (stick_shadow_pos : stick_shadow > 0)
  (sphere_shadow_pos : sphere_shadow > 0)
  (h_stick : stick_height = 1.5)
  (h_stick_shadow : stick_shadow = 1)
  (h_sphere_shadow : sphere_shadow = 8) :
  ∃ r : ℝ, r > 0 ∧ r / (sphere_shadow - r) = stick_height / stick_shadow ∧ r = 4.8 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_calculation_l3822_382280


namespace NUMINAMATH_CALUDE_tuition_calculation_l3822_382207

/-- Given the total cost and the difference between tuition and room and board,
    calculate the tuition fee. -/
theorem tuition_calculation (total_cost room_and_board tuition : ℕ) : 
  total_cost = tuition + room_and_board ∧ 
  tuition = room_and_board + 704 ∧
  total_cost = 2584 →
  tuition = 1644 := by
  sorry

#check tuition_calculation

end NUMINAMATH_CALUDE_tuition_calculation_l3822_382207


namespace NUMINAMATH_CALUDE_parallelogram_area_36_18_l3822_382212

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 36 cm and height 18 cm is 648 square centimeters -/
theorem parallelogram_area_36_18 : parallelogram_area 36 18 = 648 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_36_18_l3822_382212


namespace NUMINAMATH_CALUDE_hostel_expenditure_l3822_382237

/-- Calculates the new total expenditure of a hostel after accommodating additional students --/
def new_total_expenditure (initial_students : ℕ) (additional_students : ℕ) (average_decrease : ℕ) (total_increase : ℕ) : ℕ :=
  let new_students := initial_students + additional_students
  let original_average := (total_increase + new_students * average_decrease) / (new_students - initial_students)
  new_students * (original_average - average_decrease)

/-- Theorem stating that the new total expenditure is 5400 rupees --/
theorem hostel_expenditure :
  new_total_expenditure 100 20 5 400 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_l3822_382237


namespace NUMINAMATH_CALUDE_drain_rate_calculation_l3822_382233

/-- Represents the filling and draining system of a tank -/
structure TankSystem where
  capacity : ℝ
  fill_rate_A : ℝ
  fill_rate_B : ℝ
  drain_rate_C : ℝ
  cycle_time : ℝ
  total_time : ℝ

/-- Theorem stating the drain rate of pipe C given the system conditions -/
theorem drain_rate_calculation (s : TankSystem)
  (h1 : s.capacity = 950)
  (h2 : s.fill_rate_A = 40)
  (h3 : s.fill_rate_B = 30)
  (h4 : s.cycle_time = 3)
  (h5 : s.total_time = 57)
  (h6 : (s.total_time / s.cycle_time) * (s.fill_rate_A + s.fill_rate_B - s.drain_rate_C) = s.capacity) :
  s.drain_rate_C = 20 := by
  sorry

#check drain_rate_calculation

end NUMINAMATH_CALUDE_drain_rate_calculation_l3822_382233


namespace NUMINAMATH_CALUDE_problem_solution_l3822_382228

theorem problem_solution (x y z : ℕ+) 
  (h1 : x^2 + y^2 + z^2 = 2*(y*z + 1)) 
  (h2 : x + y + z = 4032) : 
  x^2 * y + z = 4031 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3822_382228


namespace NUMINAMATH_CALUDE_power_comparison_l3822_382236

theorem power_comparison : 2^100 < 3^75 := by sorry

end NUMINAMATH_CALUDE_power_comparison_l3822_382236


namespace NUMINAMATH_CALUDE_circle_tangent_line_radius_l3822_382243

/-- Given a circle and a line that are tangent, prove that the radius of the circle is 4. -/
theorem circle_tangent_line_radius (r : ℝ) (h1 : r > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ 3*x - 4*y + 20 = 0) →
  (∀ x y : ℝ, x^2 + y^2 ≤ r^2 → 3*x - 4*y + 20 ≥ 0) →
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ 3*x - 4*y + 20 = 0) →
  r = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_line_radius_l3822_382243


namespace NUMINAMATH_CALUDE_tony_bought_seven_swords_l3822_382296

/-- Represents the purchase of toys by Tony -/
structure ToyPurchase where
  lego_cost : ℕ
  sword_cost : ℕ
  dough_cost : ℕ
  lego_sets : ℕ
  doughs : ℕ
  total_paid : ℕ

/-- Calculates the number of toy swords bought given a ToyPurchase -/
def calculate_swords (purchase : ToyPurchase) : ℕ :=
  let lego_total := purchase.lego_cost * purchase.lego_sets
  let dough_total := purchase.dough_cost * purchase.doughs
  let sword_total := purchase.total_paid - lego_total - dough_total
  sword_total / purchase.sword_cost

/-- Theorem stating that Tony bought 7 toy swords -/
theorem tony_bought_seven_swords : 
  ∀ (purchase : ToyPurchase), 
    purchase.lego_cost = 250 →
    purchase.sword_cost = 120 →
    purchase.dough_cost = 35 →
    purchase.lego_sets = 3 →
    purchase.doughs = 10 →
    purchase.total_paid = 1940 →
    calculate_swords purchase = 7 := by
  sorry

end NUMINAMATH_CALUDE_tony_bought_seven_swords_l3822_382296


namespace NUMINAMATH_CALUDE_at_most_four_greater_than_one_l3822_382257

theorem at_most_four_greater_than_one 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : (Real.sqrt (a * b) - 1) * (Real.sqrt (b * c) - 1) * (Real.sqrt (c * a) - 1) = 1) : 
  ∃ (S : Finset ℝ), S ⊆ {a - b/c, a - c/b, b - a/c, b - c/a, c - a/b, c - b/a} ∧ 
    S.card ≤ 4 ∧ 
    (∀ x ∈ S, x > 1) ∧
    (∀ y ∈ {a - b/c, a - c/b, b - a/c, b - c/a, c - a/b, c - b/a} \ S, y ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_at_most_four_greater_than_one_l3822_382257


namespace NUMINAMATH_CALUDE_equation_equivalence_l3822_382241

theorem equation_equivalence (x : ℝ) : 
  (x^2 + x + 1) * (3*x + 4) * (-7*x + 2) * (2*x - Real.sqrt 5) * (-12*x - 16) = 0 ↔ 
  (3*x + 4 = 0 ∨ -7*x + 2 = 0 ∨ 2*x - Real.sqrt 5 = 0 ∨ -12*x - 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3822_382241


namespace NUMINAMATH_CALUDE_sum_a_b_equals_21_over_8_l3822_382261

/-- Operation ⊕ defined for real numbers -/
def circle_plus (x y : ℝ) : ℝ := x + 2*y + 3

/-- Theorem stating the result of a + b given the conditions -/
theorem sum_a_b_equals_21_over_8 (a b : ℝ) 
  (h : (circle_plus (circle_plus (a^3) (a^2)) a) = (circle_plus (a^3) (circle_plus (a^2) a)) ∧ 
       (circle_plus (circle_plus (a^3) (a^2)) a) = b) : 
  a + b = 21/8 := by sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_21_over_8_l3822_382261


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3822_382251

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 + x₁ - 3 = 0 ∧ x₁ = 1) ∧
                (2 * x₂^2 + x₂ - 3 = 0 ∧ x₂ = -3/2)) ∧
  (∃ y₁ y₂ : ℝ, ((y₁ - 3)^2 = 2 * y₁ * (3 - y₁) ∧ y₁ = 3) ∧
                ((y₂ - 3)^2 = 2 * y₂ * (3 - y₂) ∧ y₂ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3822_382251


namespace NUMINAMATH_CALUDE_equation_solution_l3822_382288

theorem equation_solution (x y : ℝ) : 
  y = 3 * x + 1 →
  4 * y^2 + 2 * y + 5 = 3 * (8 * x^2 + 2 * y + 3) →
  x = (-3 + Real.sqrt 21) / 6 ∨ x = (-3 - Real.sqrt 21) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3822_382288


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_seven_eighths_l3822_382264

/-- Represents a cube with white and black smaller cubes -/
structure ColoredCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

/-- Calculates the fraction of white surface area for a colored cube -/
def white_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- Theorem: The fraction of white surface area for the given cube configuration is 7/8 -/
theorem white_surface_fraction_is_seven_eighths :
  let c : ColoredCube := {
    edge_length := 4,
    total_small_cubes := 64,
    white_cubes := 48,
    black_cubes := 16
  }
  white_surface_fraction c = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_seven_eighths_l3822_382264


namespace NUMINAMATH_CALUDE_travel_theorem_l3822_382250

def travel_problem (total_time : ℝ) (foot_speed : ℝ) (bike_speed : ℝ) (foot_distance : ℝ) : Prop :=
  let foot_time : ℝ := foot_distance / foot_speed
  let bike_time : ℝ := total_time - foot_time
  let bike_distance : ℝ := bike_speed * bike_time
  let total_distance : ℝ := foot_distance + bike_distance
  total_distance = 80

theorem travel_theorem :
  travel_problem 7 8 16 32 := by
  sorry

end NUMINAMATH_CALUDE_travel_theorem_l3822_382250


namespace NUMINAMATH_CALUDE_walking_students_speed_l3822_382256

/-- Two students walking towards each other -/
structure WalkingStudents where
  distance : ℝ
  time : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The conditions of the problem -/
def problem : WalkingStudents where
  distance := 350
  time := 100
  speed1 := 1.9
  speed2 := 1.6  -- The speed we want to prove

theorem walking_students_speed (w : WalkingStudents) 
  (h1 : w.distance = 350)
  (h2 : w.time = 100)
  (h3 : w.speed1 = 1.9)
  (h4 : w.speed2 * w.time + w.speed1 * w.time = w.distance) :
  w.speed2 = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_walking_students_speed_l3822_382256


namespace NUMINAMATH_CALUDE_exactly_two_out_of_four_l3822_382253

def probability_of_success : ℚ := 4/5

def number_of_trials : ℕ := 4

def number_of_successes : ℕ := 2

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exactly_two_out_of_four :
  binomial_probability number_of_trials number_of_successes probability_of_success = 96/625 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_four_l3822_382253


namespace NUMINAMATH_CALUDE_sun_division_l3822_382260

theorem sun_division (x y z total : ℝ) : 
  (∀ (r : ℝ), r > 0 → y = 0.45 * r ∧ z = 0.3 * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 30 paisa
  y = 45 →  -- y's share is Rs. 45
  total = x + y + z →  -- Total is the sum of all shares
  total = 175 :=  -- The total amount is Rs. 175
by sorry

end NUMINAMATH_CALUDE_sun_division_l3822_382260


namespace NUMINAMATH_CALUDE_olympiad_1958_l3822_382222

theorem olympiad_1958 (n : ℤ) : 1155^1958 + 34^1958 ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_1958_l3822_382222


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3822_382209

theorem initial_water_percentage (
  initial_volume : ℝ) 
  (kola_percentage : ℝ)
  (added_sugar : ℝ) 
  (added_water : ℝ) 
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ) :
  initial_volume = 340 →
  kola_percentage = 6 →
  added_sugar = 3.2 →
  added_water = 10 →
  added_kola = 6.8 →
  final_sugar_percentage = 14.111111111111112 →
  ∃ initial_water_percentage : ℝ,
    initial_water_percentage = 80 ∧
    initial_water_percentage + kola_percentage + (100 - initial_water_percentage - kola_percentage) = 100 ∧
    (((100 - initial_water_percentage - kola_percentage) / 100 * initial_volume + added_sugar) / 
      (initial_volume + added_sugar + added_water + added_kola)) * 100 = final_sugar_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l3822_382209


namespace NUMINAMATH_CALUDE_point_d_coordinates_l3822_382283

/-- Given a line segment AB with endpoints A(-3, 2) and B(5, 10), and a point D on AB
    such that AD = 2DB, and the slope of AB is 1, prove that the coordinates of D are (7/3, 22/3). -/
theorem point_d_coordinates :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (5, 10)
  let D : ℝ × ℝ := (x, y)
  ∀ x y : ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B) →  -- D is on segment AB
    (x - (-3))^2 + (y - 2)^2 = 4 * ((5 - x)^2 + (10 - y)^2) →  -- AD = 2DB
    (10 - 2) / (5 - (-3)) = 1 →  -- Slope of AB is 1
    D = (7/3, 22/3) :=
by
  sorry

end NUMINAMATH_CALUDE_point_d_coordinates_l3822_382283


namespace NUMINAMATH_CALUDE_bucket_weight_l3822_382216

theorem bucket_weight (a b : ℝ) : ℝ :=
  let three_fourths_weight := a
  let one_third_weight := b
  let full_weight := (8 / 5) * a - (3 / 5) * b
  full_weight

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l3822_382216


namespace NUMINAMATH_CALUDE_base_twelve_representation_l3822_382210

def is_three_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 2 ≤ n ∧ n < b ^ 3

def has_odd_final_digit (n : ℕ) (b : ℕ) : Prop :=
  n % b % 2 = 1

theorem base_twelve_representation : 
  is_three_digit 125 12 ∧ has_odd_final_digit 125 12 ∧ 
  ∀ b : ℕ, b ≠ 12 → ¬(is_three_digit 125 b ∧ has_odd_final_digit 125 b) :=
sorry

end NUMINAMATH_CALUDE_base_twelve_representation_l3822_382210


namespace NUMINAMATH_CALUDE_sum_fraction_equality_l3822_382273

theorem sum_fraction_equality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ k ∈ ({2, 3, 4, 5, 6} : Set ℕ), 
    (a₁ / (k^2 + 1) + a₂ / (k^2 + 2) + a₃ / (k^2 + 3) + a₄ / (k^2 + 4) + a₅ / (k^2 + 5)) = 1 / k^2) :
  a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 57 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_equality_l3822_382273


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3822_382239

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (3 + Complex.I)

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3822_382239


namespace NUMINAMATH_CALUDE_school_boys_count_l3822_382224

theorem school_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 1443 →
  boys + (boys - diff) = total →
  diff = 141 →
  boys = 792 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l3822_382224


namespace NUMINAMATH_CALUDE_cookie_jar_theorem_l3822_382282

def cookie_jar_problem (initial_amount doris_spent : ℕ) : Prop :=
  let martha_spent := doris_spent / 2
  let total_spent := doris_spent + martha_spent
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 15

theorem cookie_jar_theorem :
  cookie_jar_problem 24 6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_theorem_l3822_382282


namespace NUMINAMATH_CALUDE_age_difference_l3822_382217

/-- Represents the ages of Ramesh, Mahesh, and Suresh -/
structure Ages where
  ramesh : ℕ
  mahesh : ℕ
  suresh : ℕ

/-- The ratio of present ages -/
def presentRatio (a : Ages) : Bool :=
  2 * a.mahesh = 5 * a.ramesh ∧ 5 * a.suresh = 8 * a.mahesh

/-- The ratio of ages after 15 years -/
def futureRatio (a : Ages) : Bool :=
  14 * (a.ramesh + 15) = 9 * (a.mahesh + 15) ∧
  21 * (a.mahesh + 15) = 14 * (a.suresh + 15)

/-- The theorem to be proved -/
theorem age_difference (a : Ages) :
  presentRatio a → futureRatio a → a.suresh - a.mahesh = 45 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3822_382217


namespace NUMINAMATH_CALUDE_smallest_n_divides_l3822_382208

theorem smallest_n_divides (n : ℕ) : n = 90 ↔ 
  (n > 0 ∧ 
   (315^2 - n^2) ∣ (315^3 - n^3) ∧ 
   ∀ m : ℕ, m > 0 ∧ m < n → ¬((315^2 - m^2) ∣ (315^3 - m^3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divides_l3822_382208


namespace NUMINAMATH_CALUDE_james_downhill_speed_l3822_382235

/-- Proves that James' speed on the downhill trail is 5 miles per hour given the problem conditions. -/
theorem james_downhill_speed :
  ∀ (v : ℝ),
    v > 0 →
    (20 : ℝ) / v = (12 : ℝ) / 3 + 1 - 1 →
    v = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_downhill_speed_l3822_382235


namespace NUMINAMATH_CALUDE_expression_evaluation_l3822_382286

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  3 * (2 * x^2 * y - x * y^2) - (4 * x^2 * y + x * y^2) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3822_382286


namespace NUMINAMATH_CALUDE_car_speed_increase_l3822_382266

theorem car_speed_increase (original_speed : ℝ) (supercharge_percent : ℝ) (weight_reduction_increase : ℝ) : 
  original_speed = 150 → 
  supercharge_percent = 30 → 
  weight_reduction_increase = 10 → 
  original_speed * (1 + supercharge_percent / 100) + weight_reduction_increase = 205 :=
by
  sorry

#check car_speed_increase

end NUMINAMATH_CALUDE_car_speed_increase_l3822_382266


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3822_382295

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3*x₂ + 4*x₃ = 100) : 
  ∀ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + 3*y₂ + 4*y₃ = 100 → 
  x₁^2 + x₂^2 + x₃^2 ≤ y₁^2 + y₂^2 + y₃^2 ∧ 
  ∃ z₁ z₂ z₃ : ℝ, z₁ > 0 ∧ z₂ > 0 ∧ z₃ > 0 ∧ z₁ + 3*z₂ + 4*z₃ = 100 ∧ 
  z₁^2 + z₂^2 + z₃^2 = 5000/13 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3822_382295


namespace NUMINAMATH_CALUDE_otimes_calculation_l3822_382245

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b^2

-- State the theorem
theorem otimes_calculation :
  let x := otimes (otimes 2 4) (otimes 1 3)
  let y := otimes 2 (otimes 4 3)
  x - y = 1215 / 512 := by sorry

end NUMINAMATH_CALUDE_otimes_calculation_l3822_382245


namespace NUMINAMATH_CALUDE_f_2023_equals_107_l3822_382229

-- Define the property of the function f
def has_property (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^n → f a + f b = (n^2 + 1 : ℝ)

-- Theorem statement
theorem f_2023_equals_107 (f : ℕ → ℝ) (h : has_property f) : f 2023 = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_107_l3822_382229


namespace NUMINAMATH_CALUDE_roger_shelves_theorem_l3822_382231

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  let remaining_books := total_books - books_taken
  (remaining_books + books_per_shelf - 1) / books_per_shelf

theorem roger_shelves_theorem :
  shelves_needed 24 3 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roger_shelves_theorem_l3822_382231


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3822_382203

def polynomial (x : ℝ) : ℝ := -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = 48 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3822_382203


namespace NUMINAMATH_CALUDE_transport_speed_problem_l3822_382247

/-- Proves that given two transports traveling in opposite directions for 2.71875 hours,
    with one transport traveling at 68 mph, and ending up 348 miles apart,
    the speed of the other transport must be 60 mph. -/
theorem transport_speed_problem (speed_b : ℝ) (time : ℝ) (distance : ℝ) (speed_a : ℝ) : 
  speed_b = 68 →
  time = 2.71875 →
  distance = 348 →
  (speed_a + speed_b) * time = distance →
  speed_a = 60 := by sorry

end NUMINAMATH_CALUDE_transport_speed_problem_l3822_382247


namespace NUMINAMATH_CALUDE_green_to_yellow_ratio_is_two_to_one_l3822_382200

/-- Represents the number of fish of each color in an aquarium -/
structure FishCounts where
  total : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ
  other : ℕ

/-- Calculates the ratio of green fish to yellow fish -/
def greenToYellowRatio (fc : FishCounts) : ℚ :=
  fc.green / fc.yellow

/-- Theorem: The ratio of green fish to yellow fish is 2:1 given the conditions -/
theorem green_to_yellow_ratio_is_two_to_one (fc : FishCounts)
  (h1 : fc.total = 42)
  (h2 : fc.yellow = 12)
  (h3 : fc.blue = fc.yellow / 2)
  (h4 : fc.total = fc.yellow + fc.blue + fc.green + fc.other) :
  greenToYellowRatio fc = 2 := by
  sorry

#eval greenToYellowRatio { total := 42, yellow := 12, blue := 6, green := 24, other := 0 }

end NUMINAMATH_CALUDE_green_to_yellow_ratio_is_two_to_one_l3822_382200


namespace NUMINAMATH_CALUDE_right_triangle_properties_l3822_382293

-- Define a right triangle with hypotenuse 13 and one side 5
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  hypotenuse_length : c = 13
  side_length : a = 5

-- Theorem statement
theorem right_triangle_properties (t : RightTriangle) :
  t.b = 12 ∧
  (1/2 : ℝ) * t.a * t.b = 30 ∧
  t.a + t.b + t.c = 30 ∧
  (∃ θ₁ θ₂ : ℝ, 0 < θ₁ ∧ θ₁ < π/2 ∧ 0 < θ₂ ∧ θ₂ < π/2 ∧ θ₁ + θ₂ = π/2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l3822_382293


namespace NUMINAMATH_CALUDE_toys_per_day_l3822_382274

def toys_per_week : ℕ := 5500
def work_days_per_week : ℕ := 4

theorem toys_per_day (equal_daily_production : True) : 
  toys_per_week / work_days_per_week = 1375 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l3822_382274


namespace NUMINAMATH_CALUDE_acute_triangle_angle_sum_ratio_range_l3822_382219

theorem acute_triangle_angle_sum_ratio_range (A B C : Real) 
  (h_acute : 0 < A ∧ A ≤ B ∧ B ≤ C ∧ C < π/2) 
  (h_triangle : A + B + C = π) : 
  let F := (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C)
  1 + Real.sqrt 2 / 2 < F ∧ F < 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_sum_ratio_range_l3822_382219


namespace NUMINAMATH_CALUDE_min_bananas_theorem_l3822_382201

/-- Represents the number of bananas a monkey takes from the pile -/
structure MonkeyTake where
  amount : ℕ

/-- Represents the final distribution of bananas among the monkeys -/
structure FinalDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of bananas in the pile -/
def totalBananas (t1 t2 t3 : MonkeyTake) : ℕ :=
  t1.amount + t2.amount + t3.amount

/-- Calculates the final distribution of bananas -/
def calculateDistribution (t1 t2 t3 : MonkeyTake) : FinalDistribution :=
  { first := 2 * t1.amount / 3 + t2.amount / 3 + 5 * t3.amount / 12
  , second := t1.amount / 6 + t2.amount / 3 + 5 * t3.amount / 12
  , third := t1.amount / 6 + t2.amount / 3 + t3.amount / 6 }

/-- Checks if the distribution satisfies the 4:3:2 ratio -/
def isValidRatio (d : FinalDistribution) : Prop :=
  3 * d.first = 4 * d.second ∧ 2 * d.second = 3 * d.third

/-- The main theorem stating the minimum number of bananas -/
theorem min_bananas_theorem (t1 t2 t3 : MonkeyTake) :
  (∀ d : FinalDistribution, d = calculateDistribution t1 t2 t3 → isValidRatio d) →
  totalBananas t1 t2 t3 ≥ 558 :=
sorry

end NUMINAMATH_CALUDE_min_bananas_theorem_l3822_382201


namespace NUMINAMATH_CALUDE_parabola_equation_l3822_382265

/-- The equation of a parabola with vertex at the origin and focus at (2, 0) -/
theorem parabola_equation : ∀ x y : ℝ, 
  (∃ p : ℝ, p > 0 ∧ x = p ∧ y = 0) →  -- focus at (p, 0)
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 = (a - 0)^2 + b^2) →  -- definition of parabola
  y^2 = 4 * 2 * x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3822_382265


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3822_382244

theorem arithmetic_sequence_proof :
  ∃ (a b : ℕ), 
    a = 1477 ∧ 
    b = 2089 ∧ 
    a ≤ 2000 ∧ 
    2000 ≤ b ∧ 
    ∃ (d : ℕ), a * (a + 1) - 2 = d ∧ b * (b + 1) - a * (a + 1) = d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3822_382244


namespace NUMINAMATH_CALUDE_meadow_area_is_24_l3822_382277

/-- The area of a meadow that was mowed in two days -/
def meadow_area : ℝ → Prop :=
  fun x => 
    -- Day 1: Half of the meadow plus 3 hectares
    let day1 := x / 2 + 3
    -- Remaining area after day 1
    let remaining := x - day1
    -- Day 2: One-third of the remaining area plus 6 hectares
    let day2 := remaining / 3 + 6
    -- The entire meadow is mowed after two days
    day1 + day2 = x

/-- Theorem: The area of the meadow is 24 hectares -/
theorem meadow_area_is_24 : meadow_area 24 := by
  sorry

#check meadow_area_is_24

end NUMINAMATH_CALUDE_meadow_area_is_24_l3822_382277


namespace NUMINAMATH_CALUDE_greatest_fraction_l3822_382215

theorem greatest_fraction : 
  let f1 := 44444 / 55555
  let f2 := 5555 / 6666
  let f3 := 666 / 777
  let f4 := 77 / 88
  let f5 := 8 / 9
  (f5 > f1) ∧ (f5 > f2) ∧ (f5 > f3) ∧ (f5 > f4) := by
  sorry

end NUMINAMATH_CALUDE_greatest_fraction_l3822_382215


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l3822_382287

theorem grocery_store_inventory (regular_soda diet_soda apples : ℕ) 
  (h1 : regular_soda = 72)
  (h2 : diet_soda = 32)
  (h3 : apples = 78) :
  regular_soda + diet_soda - apples = 26 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l3822_382287


namespace NUMINAMATH_CALUDE_remainder_problem_l3822_382248

theorem remainder_problem (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3822_382248


namespace NUMINAMATH_CALUDE_regression_analysis_relationship_l3822_382211

/-- Represents a statistical relationship between two variables -/
inductive StatisticalRelationship
| Correlation

/-- Represents a method of statistical analysis -/
inductive StatisticalAnalysisMethod
| RegressionAnalysis

/-- The relationship between variables in regression analysis -/
def relationship_in_regression_analysis : StatisticalRelationship := StatisticalRelationship.Correlation

theorem regression_analysis_relationship :
  relationship_in_regression_analysis = StatisticalRelationship.Correlation := by
  sorry

end NUMINAMATH_CALUDE_regression_analysis_relationship_l3822_382211


namespace NUMINAMATH_CALUDE_expected_defectives_in_sample_l3822_382218

/-- Given a population of products with some defectives, calculate the expected number of defectives in a random sample. -/
def expected_defectives (total : ℕ) (defectives : ℕ) (sample_size : ℕ) : ℚ :=
  (sample_size : ℚ) * (defectives : ℚ) / (total : ℚ)

/-- Theorem stating that the expected number of defectives in the given scenario is 10. -/
theorem expected_defectives_in_sample :
  expected_defectives 15000 1000 150 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expected_defectives_in_sample_l3822_382218


namespace NUMINAMATH_CALUDE_find_number_l3822_382238

theorem find_number (A B : ℕ) (h1 : B = 913) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 83) : A = 210 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3822_382238


namespace NUMINAMATH_CALUDE_ratio_equality_l3822_382270

theorem ratio_equality (p q r u v w : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_u : 0 < u) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_pqr : p^2 + q^2 + r^2 = 49)
  (sum_uvw : u^2 + v^2 + w^2 = 64)
  (dot_product : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3822_382270


namespace NUMINAMATH_CALUDE_fraction_equality_l3822_382223

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3822_382223


namespace NUMINAMATH_CALUDE_no_triangle_two_right_angles_l3822_382289

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_is_180 : a + b + c = 180

-- Theorem: No triangle can have two right angles
theorem no_triangle_two_right_angles :
  ∀ t : Triangle, ¬(t.a = 90 ∧ t.b = 90 ∨ t.a = 90 ∧ t.c = 90 ∨ t.b = 90 ∧ t.c = 90) :=
by
  sorry

end NUMINAMATH_CALUDE_no_triangle_two_right_angles_l3822_382289


namespace NUMINAMATH_CALUDE_coupon1_best_at_229_95_l3822_382220

def coupon1_discount (price : ℝ) : ℝ := 0.15 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.2 * (price - 150)

def price_list : List ℝ := [199.95, 229.95, 249.95, 289.95, 319.95]

theorem coupon1_best_at_229_95 :
  let p := 229.95
  (p ≥ 50) ∧
  (p ≥ 150) ∧
  (coupon1_discount p > coupon2_discount p) ∧
  (coupon1_discount p > coupon3_discount p) ∧
  (∀ q ∈ price_list, q < p → 
    coupon1_discount q ≤ coupon2_discount q ∨ 
    coupon1_discount q ≤ coupon3_discount q) :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_at_229_95_l3822_382220


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3822_382291

theorem rectangular_field_width (width length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 240 →
  width = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3822_382291


namespace NUMINAMATH_CALUDE_student_count_proof_l3822_382232

theorem student_count_proof (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 := by
sorry

end NUMINAMATH_CALUDE_student_count_proof_l3822_382232


namespace NUMINAMATH_CALUDE_valid_outfit_count_l3822_382246

/-- Represents the colors available for clothing items -/
inductive Color
  | Tan
  | Black
  | Blue
  | Gray
  | Green
  | White
  | Yellow

/-- Represents a clothing item -/
structure ClothingItem where
  color : Color

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem

def is_valid_outfit (o : Outfit) : Prop :=
  o.shirt.color ≠ o.pants.color ∧ o.hat.color ≠ o.pants.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 8

def pants_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Green]

theorem valid_outfit_count :
  (∃ (valid_outfits : List Outfit),
    (∀ o ∈ valid_outfits, is_valid_outfit o) ∧
    valid_outfits.length = 255) :=
  sorry


end NUMINAMATH_CALUDE_valid_outfit_count_l3822_382246


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3822_382221

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3822_382221


namespace NUMINAMATH_CALUDE_marble_collection_total_l3822_382290

theorem marble_collection_total (r : ℝ) (b : ℝ) (g : ℝ) : 
  r > 0 → 
  r = 1.3 * b → 
  g = 1.5 * r → 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((r + b + g) / r - 3.27) < ε :=
sorry

end NUMINAMATH_CALUDE_marble_collection_total_l3822_382290


namespace NUMINAMATH_CALUDE_decimal_difference_l3822_382214

theorem decimal_difference : 0.650 - (1 / 8 : ℚ) = 0.525 := by sorry

end NUMINAMATH_CALUDE_decimal_difference_l3822_382214


namespace NUMINAMATH_CALUDE_triangle_height_l3822_382271

/-- Given a triangle with angles α, β, γ and side c, mc is the height corresponding to side c -/
theorem triangle_height (α β γ c mc : ℝ) (h_angles : α + β + γ = Real.pi) 
  (h_positive : 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ) :
  mc = (c * Real.sin α * Real.sin β) / Real.sin γ :=
sorry


end NUMINAMATH_CALUDE_triangle_height_l3822_382271


namespace NUMINAMATH_CALUDE_classroom_count_l3822_382272

theorem classroom_count (girls boys : ℕ) (h1 : girls * 4 = boys * 3) (h2 : boys = 28) : 
  girls + boys = 49 := by
sorry

end NUMINAMATH_CALUDE_classroom_count_l3822_382272


namespace NUMINAMATH_CALUDE_triangle_side_less_than_half_perimeter_l3822_382267

theorem triangle_side_less_than_half_perimeter (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_less_than_half_perimeter_l3822_382267


namespace NUMINAMATH_CALUDE_negative_three_inequality_l3822_382213

theorem negative_three_inequality (a b : ℝ) (h : a < b) : -3*a > -3*b := by
  sorry

end NUMINAMATH_CALUDE_negative_three_inequality_l3822_382213


namespace NUMINAMATH_CALUDE_paint_for_solar_system_l3822_382276

/-- Amount of paint available for the solar system given the usage by Mary, Mike, and Lucy --/
theorem paint_for_solar_system 
  (total_paint : ℝ) 
  (mary_paint : ℝ) 
  (mike_extra_paint : ℝ) 
  (lucy_paint : ℝ) 
  (h1 : total_paint = 25) 
  (h2 : mary_paint = 3) 
  (h3 : mike_extra_paint = 2) 
  (h4 : lucy_paint = 4) : 
  total_paint - (mary_paint + (mary_paint + mike_extra_paint) + lucy_paint) = 13 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_solar_system_l3822_382276


namespace NUMINAMATH_CALUDE_special_function_half_l3822_382297

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

/-- The main theorem stating the value of f(1/2) -/
theorem special_function_half (f : ℝ → ℝ) (h : special_function f) : f (1/2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_half_l3822_382297


namespace NUMINAMATH_CALUDE_shopper_fraction_l3822_382269

theorem shopper_fraction (total_shoppers : ℕ) (checkout_shoppers : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : checkout_shoppers = 180) : 
  (total_shoppers - checkout_shoppers : ℚ) / total_shoppers = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shopper_fraction_l3822_382269


namespace NUMINAMATH_CALUDE_sudoku_unique_solution_l3822_382275

def Sudoku := Fin 4 → Fin 4 → Fin 4

def valid_sudoku (s : Sudoku) : Prop :=
  (∀ i j₁ j₂, j₁ ≠ j₂ → s i j₁ ≠ s i j₂) ∧  -- rows
  (∀ i₁ i₂ j, i₁ ≠ i₂ → s i₁ j ≠ s i₂ j) ∧  -- columns
  (∀ b₁ b₂ c₁ c₂, (b₁ ≠ c₁ ∨ b₂ ≠ c₂) →     -- 2x2 subgrids
    s (2*b₁) (2*b₂) ≠ s (2*b₁+c₁) (2*b₂+c₂))

def initial_constraints (s : Sudoku) : Prop :=
  s 0 0 = 0 ∧  -- 3 in top-left (0-indexed)
  s 3 0 = 0 ∧  -- 1 in bottom-left
  s 2 2 = 1 ∧  -- 2 in third row, third column
  s 1 3 = 0    -- 1 in second row, fourth column

theorem sudoku_unique_solution (s : Sudoku) :
  valid_sudoku s ∧ initial_constraints s → s 0 1 = 1 := by sorry

end NUMINAMATH_CALUDE_sudoku_unique_solution_l3822_382275


namespace NUMINAMATH_CALUDE_alice_painted_six_cuboids_l3822_382281

/-- The number of outer faces on a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces Alice painted -/
def total_painted_faces : ℕ := 36

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem alice_painted_six_cuboids :
  num_cuboids = 6 :=
sorry

end NUMINAMATH_CALUDE_alice_painted_six_cuboids_l3822_382281


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3822_382205

theorem linear_equation_solution (a b : ℝ) : 
  (2 : ℝ) * a + (-1 : ℝ) * b = 2 → 2 * a - b - 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3822_382205


namespace NUMINAMATH_CALUDE_graph_shift_l3822_382226

-- Define the functions f and g
def f (x : ℝ) : ℝ := (x + 3)^2 - 1
def g (x : ℝ) : ℝ := (x - 2)^2 + 3

-- State the theorem
theorem graph_shift : ∀ x : ℝ, f x = g (x - 5) + 4 := by sorry

end NUMINAMATH_CALUDE_graph_shift_l3822_382226


namespace NUMINAMATH_CALUDE_bridget_profit_l3822_382242

def total_loaves : ℕ := 60
def morning_price : ℚ := 3
def afternoon_price : ℚ := 2
def late_price : ℚ := 3/2
def production_cost : ℚ := 4/5

def morning_sales : ℕ := total_loaves / 3
def afternoon_sales : ℕ := ((total_loaves - morning_sales) * 3) / 4
def late_sales : ℕ := total_loaves - morning_sales - afternoon_sales

def total_revenue : ℚ := 
  morning_sales * morning_price + 
  afternoon_sales * afternoon_price + 
  late_sales * late_price

def total_cost : ℚ := total_loaves * production_cost

def profit : ℚ := total_revenue - total_cost

theorem bridget_profit : profit = 87 := by sorry

end NUMINAMATH_CALUDE_bridget_profit_l3822_382242


namespace NUMINAMATH_CALUDE_final_amount_proof_l3822_382279

/-- Calculates the final amount after two years of compound interest with different rates each year. -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount1 := initial * (1 + rate1)
  amount1 * (1 + rate2)

/-- Theorem stating that given the specific initial amount and interest rates, 
    the final amount after two years is as calculated. -/
theorem final_amount_proof :
  final_amount 6552 0.04 0.05 = 7154.784 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_proof_l3822_382279


namespace NUMINAMATH_CALUDE_expand_expression_l3822_382284

theorem expand_expression (x y : ℝ) : (3*x - 5) * (4*y + 20) = 12*x*y + 60*x - 20*y - 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3822_382284


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3822_382252

/-- The total number of dogwood trees after planting operations -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating that the total number of trees after planting is 16 -/
theorem dogwood_tree_count :
  total_trees 7 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3822_382252


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_l3822_382255

/-- Given a hyperbola with equation x²/4 - y²/5 = 1, prove that the equation of an ellipse
    with foci at the vertices of the hyperbola and vertices at the foci of the hyperbola
    is x²/9 + y²/5 = 1 -/
theorem ellipse_from_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) →
  ∃ (a b c : ℝ),
    (a^2 = 9 ∧ b^2 = 5 ∧ c^2 = 4) ∧
    (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_l3822_382255


namespace NUMINAMATH_CALUDE_problem_solution_l3822_382230

noncomputable def AC (x : ℝ) : ℝ × ℝ := (Real.cos (x/2) + Real.sin (x/2), Real.sin (x/2))

noncomputable def BC (x : ℝ) : ℝ × ℝ := (Real.sin (x/2) - Real.cos (x/2), 2 * Real.cos (x/2))

noncomputable def f (x : ℝ) : ℝ := (AC x).1 * (BC x).1 + (AC x).2 * (BC x).2

theorem problem_solution :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo Real.pi (3 * Real.pi) ∧ x₂ ∈ Set.Ioo Real.pi (3 * Real.pi) ∧
    f x₁ = Real.sqrt 6 / 2 ∧ f x₂ = Real.sqrt 6 / 2 ∧ x₁ ≠ x₂ →
    x₁ + x₂ = 11 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3822_382230


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l3822_382258

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_a_range 
  (a b c : ℝ) 
  (h1 : f a b c (-2) = 1) 
  (h2 : f a b c 2 = 3) 
  (h3 : 0 < c) 
  (h4 : c < 1) : 
  1/4 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l3822_382258


namespace NUMINAMATH_CALUDE_three_over_x_is_fraction_l3822_382254

/-- A fraction is defined as an expression with a variable in the denominator. -/
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (a : ℚ) (b : ℚ → ℚ), ∀ x, f x = a / (b x) ∧ b x ≠ 0

/-- The function f(x) = 3/x is a fraction. -/
theorem three_over_x_is_fraction :
  is_fraction (λ x : ℚ => 3 / x) :=
sorry

end NUMINAMATH_CALUDE_three_over_x_is_fraction_l3822_382254


namespace NUMINAMATH_CALUDE_girls_in_group_l3822_382263

theorem girls_in_group (n : ℕ) : 
  (4 : ℝ) + n > 0 → -- ensure the total number of students is positive
  (((n + 4) * (n + 3) / 2 - 6) / ((n + 4) * (n + 3) / 2) = 5 / 6) →
  n = 5 := by
  sorry


end NUMINAMATH_CALUDE_girls_in_group_l3822_382263


namespace NUMINAMATH_CALUDE_liar_identification_l3822_382278

def original_number : ℕ := 2014315

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def statement_A (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 8

def statement_B (cards : Finset ℕ) : Prop :=
  ∀ (i j : ℕ), i ∈ cards → j ∈ cards → i ≠ j →
  ¬is_divisible_by (swap_digits original_number i j) 9

def statement_C (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 10

def statement_D (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 11

theorem liar_identification :
  ∃ (cards_A cards_B cards_C cards_D : Finset ℕ),
    cards_A.card ≤ 2 ∧ cards_B.card ≤ 2 ∧ cards_C.card ≤ 2 ∧ cards_D.card ≤ 2 ∧
    cards_A ∪ cards_B ∪ cards_C ∪ cards_D = {0, 1, 2, 3, 4, 5} ∧
    cards_A ∩ cards_B = ∅ ∧ cards_A ∩ cards_C = ∅ ∧ cards_A ∩ cards_D = ∅ ∧
    cards_B ∩ cards_C = ∅ ∧ cards_B ∩ cards_D = ∅ ∧ cards_C ∩ cards_D = ∅ ∧
    statement_A cards_A ∧ statement_B cards_B ∧ ¬statement_C cards_C ∧ statement_D cards_D :=
by sorry

end NUMINAMATH_CALUDE_liar_identification_l3822_382278


namespace NUMINAMATH_CALUDE_students_passed_l3822_382206

def total_students : ℕ := 450

def failed_breakup : ℕ := (5 * total_students) / 12

def remaining_after_breakup : ℕ := total_students - failed_breakup

def no_show : ℕ := (7 * remaining_after_breakup) / 15

def remaining_after_no_show : ℕ := remaining_after_breakup - no_show

def penalized : ℕ := 45

def remaining_after_penalty : ℕ := remaining_after_no_show - penalized

def bonus_but_failed : ℕ := remaining_after_penalty / 8

theorem students_passed :
  total_students - failed_breakup - no_show - penalized - bonus_but_failed = 84 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_l3822_382206


namespace NUMINAMATH_CALUDE_tom_and_michael_have_nine_robots_l3822_382234

/-- The number of car robots Bob has -/
def bob_robots : ℕ := 81

/-- The factor by which Bob's robots outnumber Tom and Michael's combined -/
def factor : ℕ := 9

/-- The number of car robots Tom and Michael have combined -/
def tom_and_michael_robots : ℕ := bob_robots / factor

theorem tom_and_michael_have_nine_robots : tom_and_michael_robots = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_and_michael_have_nine_robots_l3822_382234


namespace NUMINAMATH_CALUDE_class_size_from_mark_error_l3822_382202

theorem class_size_from_mark_error (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 40 → average_increase = 1/2 → 
  (mark_increase : ℚ) / average_increase = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_mark_error_l3822_382202


namespace NUMINAMATH_CALUDE_equal_coin_count_l3822_382285

/-- Represents the value of each coin type in cents -/
def coin_value : Fin 5 → ℕ
  | 0 => 1    -- penny
  | 1 => 5    -- nickel
  | 2 => 10   -- dime
  | 3 => 25   -- quarter
  | 4 => 50   -- half dollar

/-- The theorem statement -/
theorem equal_coin_count (x : ℕ) (h : x * (coin_value 0 + coin_value 1 + coin_value 2 + coin_value 3 + coin_value 4) = 273) :
  5 * x = 15 := by
  sorry

#check equal_coin_count

end NUMINAMATH_CALUDE_equal_coin_count_l3822_382285


namespace NUMINAMATH_CALUDE_jigi_score_l3822_382292

theorem jigi_score (max_score : ℕ) (gibi_percent mike_percent lizzy_percent : ℚ) 
  (average_mark : ℕ) (h1 : max_score = 700) (h2 : gibi_percent = 59/100) 
  (h3 : mike_percent = 99/100) (h4 : lizzy_percent = 67/100) (h5 : average_mark = 490) : 
  (4 * average_mark - (gibi_percent + mike_percent + lizzy_percent) * max_score) / max_score = 55/100 :=
sorry

end NUMINAMATH_CALUDE_jigi_score_l3822_382292


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l3822_382240

/-- The midpoint of a line segment in polar coordinates -/
def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let p₁ : ℝ × ℝ := (5, π/6)
  let p₂ : ℝ × ℝ := (5, -π/6)
  let m : ℝ × ℝ := midpoint_polar p₁.1 p₁.2 p₂.1 p₂.2
  m.1 > 0 ∧ 0 ≤ m.2 ∧ m.2 < 2*π ∧ m = (5*Real.sqrt 3/2, π/6) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l3822_382240
