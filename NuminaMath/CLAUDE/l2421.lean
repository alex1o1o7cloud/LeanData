import Mathlib

namespace NUMINAMATH_CALUDE_largest_among_decimals_l2421_242185

theorem largest_among_decimals :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_among_decimals_l2421_242185


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l2421_242194

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 5 →
    rectangle_height = 12 →
    circle_circumference = π * Real.sqrt (rectangle_width^2 + rectangle_height^2) →
    circle_circumference = 13 * π :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l2421_242194


namespace NUMINAMATH_CALUDE_complex_cube_root_identity_l2421_242143

theorem complex_cube_root_identity (z : ℂ) (h1 : z^3 + 1 = 0) (h2 : z ≠ -1) :
  (z / (z - 1))^2018 + (1 / (z - 1))^2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_identity_l2421_242143


namespace NUMINAMATH_CALUDE_cone_height_l2421_242137

theorem cone_height (r : ℝ) (h : ℝ) :
  r = 1 →
  (2 * Real.pi * r = (2 * Real.pi / 3) * 3) →
  h = Real.sqrt (3^2 - r^2) →
  h = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l2421_242137


namespace NUMINAMATH_CALUDE_stock_certificate_tearing_l2421_242157

theorem stock_certificate_tearing : ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := by
  sorry

end NUMINAMATH_CALUDE_stock_certificate_tearing_l2421_242157


namespace NUMINAMATH_CALUDE_independence_test_relationship_l2421_242145

-- Define the random variable K²
def K_squared : ℝ → ℝ := sorry

-- Define the probability of judging variables as related
def prob_related : ℝ → ℝ := sorry

-- Define the test of independence
def test_of_independence : (ℝ → ℝ) → (ℝ → ℝ) → Prop := sorry

-- Theorem statement
theorem independence_test_relationship :
  ∀ (x y : ℝ), x > y →
  test_of_independence K_squared prob_related →
  prob_related (K_squared x) < prob_related (K_squared y) :=
sorry

end NUMINAMATH_CALUDE_independence_test_relationship_l2421_242145


namespace NUMINAMATH_CALUDE_gecko_hatched_eggs_l2421_242111

/-- Theorem: Number of hatched eggs for a gecko --/
theorem gecko_hatched_eggs (total_eggs : ℕ) (infertile_rate : ℚ) (calcification_rate : ℚ)
  (h_total : total_eggs = 30)
  (h_infertile : infertile_rate = 1/5)
  (h_calcification : calcification_rate = 1/3) :
  (total_eggs : ℚ) * (1 - infertile_rate) * (1 - calcification_rate) = 16 := by
  sorry

end NUMINAMATH_CALUDE_gecko_hatched_eggs_l2421_242111


namespace NUMINAMATH_CALUDE_abs_diff_properties_l2421_242126

-- Define the binary operation ⊕
def abs_diff (x y : ℝ) : ℝ := |x - y|

-- Main theorem
theorem abs_diff_properties :
  -- 1. ⊕ is commutative
  (∀ x y : ℝ, abs_diff x y = abs_diff y x) ∧
  -- 2. Addition distributes over ⊕
  (∀ a b c : ℝ, a + abs_diff b c = abs_diff (a + b) (a + c)) ∧
  -- 3. ⊕ is not associative
  (∃ x y z : ℝ, abs_diff x (abs_diff y z) ≠ abs_diff (abs_diff x y) z) ∧
  -- 4. ⊕ does not have an identity element
  (∀ e : ℝ, ∃ x : ℝ, abs_diff x e ≠ x) ∧
  -- 5. ⊕ does not distribute over addition
  (∃ x y z : ℝ, abs_diff x (y + z) ≠ abs_diff x y + abs_diff x z) :=
by sorry

end NUMINAMATH_CALUDE_abs_diff_properties_l2421_242126


namespace NUMINAMATH_CALUDE_smallest_six_digit_multiple_of_1379_l2421_242178

theorem smallest_six_digit_multiple_of_1379 : 
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 1379 = 0 → n ≥ 100657 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_six_digit_multiple_of_1379_l2421_242178


namespace NUMINAMATH_CALUDE_frog_climb_time_l2421_242188

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℕ
  climb_distance : ℕ
  slip_distance : ℕ
  climb_time : ℚ
  slip_time : ℚ
  intermediate_time : ℕ
  intermediate_distance : ℕ

/-- Theorem stating the time taken for the frog to climb out of the well -/
theorem frog_climb_time (f : FrogClimb)
  (h1 : f.well_depth = 12)
  (h2 : f.climb_distance = 3)
  (h3 : f.slip_distance = 1)
  (h4 : f.slip_time = f.climb_time / 3)
  (h5 : f.intermediate_time = 17)
  (h6 : f.intermediate_distance = f.well_depth - 3)
  (h7 : f.climb_time = 1) :
  ∃ (total_time : ℕ), total_time = 22 ∧ 
  (∃ (cycles : ℕ), 
    cycles * (f.climb_distance - f.slip_distance) + 
    (total_time - cycles * (f.climb_time + f.slip_time)) * f.climb_distance / f.climb_time = f.well_depth) :=
sorry

end NUMINAMATH_CALUDE_frog_climb_time_l2421_242188


namespace NUMINAMATH_CALUDE_function_property_l2421_242179

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x) :
  ∀ x : ℝ, f x + f (-x) = 7 := by
sorry

end NUMINAMATH_CALUDE_function_property_l2421_242179


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2421_242183

/-- A sequence is geometric if the ratio of successive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_3 * a_5 * a_7 = (-√3)^3, prove a_2 * a_8 = 3 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_product : a 3 * a 5 * a 7 = (-Real.sqrt 3)^3) : 
  a 2 * a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2421_242183


namespace NUMINAMATH_CALUDE_board_cut_theorem_l2421_242112

theorem board_cut_theorem (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 69 →
  total_length = shorter_piece + 2 * shorter_piece →
  shorter_piece = 23 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l2421_242112


namespace NUMINAMATH_CALUDE_tim_watched_24_hours_l2421_242165

/-- Calculates the total hours of TV watched given the number of episodes and duration per episode for two shows. -/
def total_hours_watched (short_episodes : ℕ) (short_duration : ℚ) (long_episodes : ℕ) (long_duration : ℚ) : ℚ :=
  short_episodes * short_duration + long_episodes * long_duration

/-- Proves that Tim watched 24 hours of TV given the specified conditions. -/
theorem tim_watched_24_hours :
  let short_episodes : ℕ := 24
  let short_duration : ℚ := 1/2
  let long_episodes : ℕ := 12
  let long_duration : ℚ := 1
  total_hours_watched short_episodes short_duration long_episodes long_duration = 24 := by
  sorry

#eval total_hours_watched 24 (1/2) 12 1

end NUMINAMATH_CALUDE_tim_watched_24_hours_l2421_242165


namespace NUMINAMATH_CALUDE_floor_of_4_7_l2421_242134

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l2421_242134


namespace NUMINAMATH_CALUDE_betty_age_l2421_242191

/-- Given the relationships between Albert's, Mary's, and Betty's ages, prove Betty's age. -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 8) :
  betty = 4 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l2421_242191


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2421_242123

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.cos θ - 6 * Real.sin θ

-- Define the Cartesian equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Theorem statement
theorem polar_to_cartesian :
  ∀ (x y ρ θ : ℝ),
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →  -- Conversion between polar and Cartesian coordinates
  polar_equation ρ θ ↔ cartesian_equation x y :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2421_242123


namespace NUMINAMATH_CALUDE_gmat_test_probabilities_l2421_242102

theorem gmat_test_probabilities
  (p_first : ℝ)
  (p_second : ℝ)
  (p_both : ℝ)
  (h1 : p_first = 0.85)
  (h2 : p_second = 0.80)
  (h3 : p_both = 0.70)
  : 1 - (p_first + p_second - p_both) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_probabilities_l2421_242102


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2421_242167

variables (a b c : ℝ)

theorem simplify_and_sum_exponents :
  ∃ (x y z : ℕ) (w : ℝ),
    (40 * a^7 * b^9 * c^14)^(1/3) = 2 * a^x * b^y * c^z * w^(1/3) ∧
    w = 5 * a * c^2 ∧
    x + y + z = 9 := by sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2421_242167


namespace NUMINAMATH_CALUDE_speed_ratio_l2421_242171

/-- The speed of Person A -/
def speed_A : ℝ := sorry

/-- The speed of Person B -/
def speed_B : ℝ := sorry

/-- The distance covered by Person A in a given time -/
def distance_A : ℝ := sorry

/-- The distance covered by Person B in the same time -/
def distance_B : ℝ := sorry

/-- The time taken for both persons to cover their respective distances -/
def time : ℝ := sorry

theorem speed_ratio :
  (speed_A / speed_B = 3 / 2) ∧
  (distance_A = 3) ∧
  (distance_B = 2) ∧
  (speed_A = distance_A / time) ∧
  (speed_B = distance_B / time) :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_l2421_242171


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2421_242170

/-- Given a triangle with two sides of lengths that are roots of x^2 - 5x + 6 = 0,
    the perimeter l of the triangle satisfies 6 < l < 10 -/
theorem triangle_perimeter_range : ∀ a b c : ℝ,
  (a^2 - 5*a + 6 = 0) →
  (b^2 - 5*b + 6 = 0) →
  (a ≠ b) →
  (a + b > c) →
  (b + c > a) →
  (c + a > b) →
  let l := a + b + c
  6 < l ∧ l < 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2421_242170


namespace NUMINAMATH_CALUDE_inequality_proof_l2421_242199

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2421_242199


namespace NUMINAMATH_CALUDE_liters_to_pints_conversion_l2421_242192

/-- Given that 0.33 liters is approximately 0.7 pints, prove that one liter is approximately 2.1 pints. -/
theorem liters_to_pints_conversion (ε : ℝ) (h_ε : ε > 0) :
  ∃ (δ : ℝ), δ > 0 ∧ 
  ∀ (x y : ℝ), 
    (abs (x - 0.33) < δ ∧ abs (y - 0.7) < δ) → 
    abs ((1 / x * y) - 2.1) < ε :=
sorry

end NUMINAMATH_CALUDE_liters_to_pints_conversion_l2421_242192


namespace NUMINAMATH_CALUDE_remainder_theorem_l2421_242161

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) (hv : v < y) (hxdiv : x = u * y + v) :
  (x + 3 * u * y + 4) % y = (v + 4) % y :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2421_242161


namespace NUMINAMATH_CALUDE_ellipse_parameter_inequality_l2421_242135

/-- An ellipse with equation ax^2 + by^2 = 1 and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b > 0
  foci_on_x_axis : a ≠ b

theorem ellipse_parameter_inequality (e : Ellipse) : 0 < e.a ∧ e.a < e.b := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parameter_inequality_l2421_242135


namespace NUMINAMATH_CALUDE_polynomial_identity_solution_l2421_242169

theorem polynomial_identity_solution :
  ∀ (a b c : ℝ),
    (∀ x : ℝ, x^3 - a*x^2 + b*x - c = (x-a)*(x-b)*(x-c))
    ↔ 
    (a = -1 ∧ b = -1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_solution_l2421_242169


namespace NUMINAMATH_CALUDE_min_value_theorem_l2421_242172

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + 2*n = 1) :
  (1 / (2*m)) + (1 / n) ≥ 9/2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ m₀ + 2*n₀ = 1 ∧ (1 / (2*m₀)) + (1 / n₀) = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2421_242172


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l2421_242117

-- Define the circle radius
def circle_radius : ℝ := 6

-- Define the relationship between rectangle and circle areas
def area_ratio : ℝ := 3

-- Theorem statement
theorem rectangle_longer_side (circle_radius : ℝ) (area_ratio : ℝ) :
  circle_radius = 6 →
  area_ratio = 3 →
  let circle_area := π * circle_radius^2
  let rectangle_area := area_ratio * circle_area
  let shorter_side := 2 * circle_radius
  rectangle_area / shorter_side = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l2421_242117


namespace NUMINAMATH_CALUDE_power_simplification_l2421_242106

theorem power_simplification : 
  ((12^15 / 12^7)^3 * 8^3) / 2^9 = 12^24 := by sorry

end NUMINAMATH_CALUDE_power_simplification_l2421_242106


namespace NUMINAMATH_CALUDE_leon_order_proof_l2421_242132

def toy_organizer_sets : ℕ := 3
def toy_organizer_price : ℚ := 78
def gaming_chair_price : ℚ := 83
def delivery_fee_rate : ℚ := 0.05
def total_payment : ℚ := 420

def gaming_chairs_ordered : ℕ := 2

theorem leon_order_proof :
  ∃ (g : ℕ), 
    (toy_organizer_sets * toy_organizer_price + g * gaming_chair_price) * (1 + delivery_fee_rate) = total_payment ∧
    g = gaming_chairs_ordered :=
by sorry

end NUMINAMATH_CALUDE_leon_order_proof_l2421_242132


namespace NUMINAMATH_CALUDE_two_cars_in_garage_l2421_242115

/-- Represents the number of wheels on various vehicles --/
structure VehicleWheels where
  lawnmower : Nat
  bicycle : Nat
  tricycle : Nat
  unicycle : Nat
  car : Nat

/-- Calculates the total number of wheels for non-car vehicles --/
def nonCarWheels (v : VehicleWheels) (numBicycles : Nat) : Nat :=
  v.lawnmower + numBicycles * v.bicycle + v.tricycle + v.unicycle

/-- Theorem stating that given the conditions in the problem, there are 2 cars in the garage --/
theorem two_cars_in_garage (totalWheels : Nat) (v : VehicleWheels) (numBicycles : Nat) :
  totalWheels = 22 →
  v.lawnmower = 4 →
  v.bicycle = 2 →
  v.tricycle = 3 →
  v.unicycle = 1 →
  v.car = 4 →
  numBicycles = 3 →
  (totalWheels - nonCarWheels v numBicycles) / v.car = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_cars_in_garage_l2421_242115


namespace NUMINAMATH_CALUDE_exponential_property_l2421_242182

theorem exponential_property (a : ℝ) :
  (∀ x > 0, a^x > 1) → a > 1 := by sorry

end NUMINAMATH_CALUDE_exponential_property_l2421_242182


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2421_242181

/-- An arithmetic sequence with 20 terms -/
structure ArithmeticSequence :=
  (a : ℚ)  -- First term
  (d : ℚ)  -- Common difference

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a + (n - 1) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 
  sum_n seq 3 = 15 ∧ 
  sum_n seq 3 - 3 * seq.a - 51 * seq.d = 12 → 
  sum_n seq 20 = 90 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2421_242181


namespace NUMINAMATH_CALUDE_farmer_bean_seedlings_per_row_l2421_242113

/-- Represents the farmer's planting scenario -/
structure FarmPlanting where
  bean_seedlings : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  rows_per_bed : ℕ
  plant_beds : ℕ

/-- Calculates the number of bean seedlings per row -/
def bean_seedlings_per_row (fp : FarmPlanting) : ℕ :=
  fp.bean_seedlings / (fp.plant_beds * fp.rows_per_bed - 
    (fp.pumpkin_seeds / fp.pumpkin_per_row + fp.radishes / fp.radishes_per_row))

/-- Theorem stating that given the farmer's planting scenario, 
    the number of bean seedlings per row is 8 -/
theorem farmer_bean_seedlings_per_row :
  let fp : FarmPlanting := {
    bean_seedlings := 64,
    pumpkin_seeds := 84,
    pumpkin_per_row := 7,
    radishes := 48,
    radishes_per_row := 6,
    rows_per_bed := 2,
    plant_beds := 14
  }
  bean_seedlings_per_row fp = 8 := by
  sorry

end NUMINAMATH_CALUDE_farmer_bean_seedlings_per_row_l2421_242113


namespace NUMINAMATH_CALUDE_ship_passengers_l2421_242109

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 8 : ℚ) + (P / 3 : ℚ) + (P / 6 : ℚ) + 35 = P →
  P = 120 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l2421_242109


namespace NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2421_242114

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2421_242114


namespace NUMINAMATH_CALUDE_y_value_l2421_242129

theorem y_value (y : ℚ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2421_242129


namespace NUMINAMATH_CALUDE_square_difference_65_35_l2421_242147

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l2421_242147


namespace NUMINAMATH_CALUDE_prob_two_sunny_days_l2421_242156

/-- The probability of exactly 2 sunny days in a 5-day period with 75% chance of rain each day -/
theorem prob_two_sunny_days : 
  let n : ℕ := 5  -- Total number of days
  let p : ℚ := 3/4  -- Probability of rain each day
  let k : ℕ := 2  -- Number of sunny days we want
  Nat.choose n k * (1 - p)^k * p^(n - k) = 135/512 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_sunny_days_l2421_242156


namespace NUMINAMATH_CALUDE_coach_cost_l2421_242162

/-- Proves that the cost of the coach before discount is $2500 given the problem conditions -/
theorem coach_cost (sectional_cost other_cost total_paid : ℝ) 
  (h1 : sectional_cost = 3500)
  (h2 : other_cost = 2000)
  (h3 : total_paid = 7200)
  (discount : ℝ) (h4 : discount = 0.1)
  : ∃ (coach_cost : ℝ), 
    coach_cost = 2500 ∧ 
    (1 - discount) * (coach_cost + sectional_cost + other_cost) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_coach_cost_l2421_242162


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l2421_242119

def aquarium (num_stingrays : ℕ) : ℕ → Prop :=
  fun total_fish =>
    ∃ num_sharks : ℕ,
      num_sharks = 2 * num_stingrays ∧
      total_fish = num_sharks + num_stingrays

theorem aquarium_fish_count : aquarium 28 84 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l2421_242119


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2421_242125

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2421_242125


namespace NUMINAMATH_CALUDE_distance_n_n_l2421_242173

/-- The distance function for a point (a,b) on the polygonal path -/
def distance (a b : ℕ) : ℕ := sorry

/-- The theorem stating that the distance of (n,n) is n^2 + n -/
theorem distance_n_n (n : ℕ) : distance n n = n^2 + n := by sorry

end NUMINAMATH_CALUDE_distance_n_n_l2421_242173


namespace NUMINAMATH_CALUDE_point_on_line_m_value_l2421_242152

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

theorem point_on_line_m_value :
  ∀ m : ℝ,
  let P : Point := ⟨3, m⟩
  let M : Point := ⟨2, -1⟩
  let N : Point := ⟨-3, 4⟩
  collinear P M N → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_m_value_l2421_242152


namespace NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l2421_242127

theorem cubic_greater_than_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l2421_242127


namespace NUMINAMATH_CALUDE_food_storage_temperature_l2421_242100

-- Define the temperature range
def temp_center : ℝ := -2
def temp_range : ℝ := 3

-- Define the function to check if a temperature is within the range
def is_within_range (temp : ℝ) : Prop :=
  temp ≥ temp_center - temp_range ∧ temp ≤ temp_center + temp_range

-- State the theorem
theorem food_storage_temperature :
  is_within_range (-1) ∧
  ¬is_within_range 2 ∧
  ¬is_within_range (-6) ∧
  ¬is_within_range 4 :=
sorry

end NUMINAMATH_CALUDE_food_storage_temperature_l2421_242100


namespace NUMINAMATH_CALUDE_correct_statement_l2421_242196

/-- Proposition p: There exists an x₀ ∈ ℝ such that x₀² + x₀ + 1 ≤ 0 -/
def p : Prop := ∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0

/-- Proposition q: The function f(x) = x^(1/3) is an increasing function -/
def q : Prop := ∀ x y : ℝ, x < y → Real.rpow x (1/3) < Real.rpow y (1/3)

/-- The correct statement is (¬p) ∨ q -/
theorem correct_statement : (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_correct_statement_l2421_242196


namespace NUMINAMATH_CALUDE_regular_nonagon_diagonal_relation_l2421_242148

/-- Regular nonagon -/
structure RegularNonagon where
  /-- Length of a side -/
  a : ℝ
  /-- Length of the shortest diagonal -/
  b : ℝ
  /-- Length of the longest diagonal -/
  d : ℝ
  /-- a is positive -/
  a_pos : a > 0

/-- Theorem: In a regular nonagon, d^2 = a^2 + ab + b^2 -/
theorem regular_nonagon_diagonal_relation (N : RegularNonagon) : N.d^2 = N.a^2 + N.a*N.b + N.b^2 := by
  sorry

end NUMINAMATH_CALUDE_regular_nonagon_diagonal_relation_l2421_242148


namespace NUMINAMATH_CALUDE_no_valid_numbers_with_19x_relation_l2421_242118

/-- Checks if a natural number is composed only of digits 2, 3, 4, and 9 -/
def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [2, 3, 4, 9]

/-- The main theorem stating the impossibility of finding two numbers
    with the given properties -/
theorem no_valid_numbers_with_19x_relation :
  ¬∃ (a b : ℕ), is_valid_number a ∧ is_valid_number b ∧ b = 19 * a :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_with_19x_relation_l2421_242118


namespace NUMINAMATH_CALUDE_distinct_collections_count_l2421_242131

/-- Represents the number of occurrences of each letter in "CALCULATIONS" --/
def letterCounts : Fin 26 → ℕ
| 0 => 3  -- A
| 2 => 3  -- C
| 8 => 1  -- I
| 11 => 3 -- L
| 13 => 1 -- N
| 14 => 1 -- O
| 18 => 1 -- S
| 19 => 1 -- T
| 20 => 1 -- U
| _ => 0

/-- Predicate for vowels --/
def isVowel (n : Fin 26) : Bool :=
  n = 0 ∨ n = 4 ∨ n = 8 ∨ n = 14 ∨ n = 20

/-- The number of distinct collections of three vowels and three consonants --/
def distinctCollections : ℕ := sorry

/-- Theorem stating that the number of distinct collections is 126 --/
theorem distinct_collections_count :
  distinctCollections = 126 := by sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l2421_242131


namespace NUMINAMATH_CALUDE_goods_train_length_l2421_242154

/-- The length of a goods train passing a man in an opposite-moving train --/
theorem goods_train_length (man_speed goods_speed : ℝ) (pass_time : ℝ) : 
  man_speed = 64 →
  goods_speed = 20 →
  pass_time = 18 →
  ∃ (length : ℝ), abs (length - (man_speed + goods_speed) * 1000 / 3600 * pass_time) < 1 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_length_l2421_242154


namespace NUMINAMATH_CALUDE_jack_total_miles_driven_l2421_242101

/-- Calculates the total miles driven given the number of years and miles driven per four-month period -/
def total_miles_driven (years : ℕ) (miles_per_period : ℕ) : ℕ :=
  let months : ℕ := years * 12
  let periods : ℕ := months / 4
  periods * miles_per_period

/-- Proves that given 9 years of driving and 37,000 miles driven every four months, the total miles driven is 999,000 -/
theorem jack_total_miles_driven :
  total_miles_driven 9 37000 = 999000 := by
  sorry

#eval total_miles_driven 9 37000

end NUMINAMATH_CALUDE_jack_total_miles_driven_l2421_242101


namespace NUMINAMATH_CALUDE_distance_between_5th_and_30th_red_light_l2421_242144

/-- Represents the color of a light in the sequence -/
inductive LightColor
  | Red
  | Green

/-- Calculates the position of a light in the sequence given its number and color -/
def lightPosition (n : Nat) (color : LightColor) : Nat :=
  match color with
  | LightColor.Red => (n - 1) / 3 * 7 + (n - 1) % 3 + 1
  | LightColor.Green => (n - 1) / 4 * 7 + (n - 1) % 4 + 4

/-- The spacing between lights in inches -/
def lightSpacing : Nat := 8

/-- The number of inches in a foot -/
def inchesPerFoot : Nat := 12

/-- Calculates the distance in feet between two lights given their positions -/
def distanceBetweenLights (pos1 pos2 : Nat) : Nat :=
  ((pos2 - pos1) * lightSpacing) / inchesPerFoot

theorem distance_between_5th_and_30th_red_light :
  distanceBetweenLights (lightPosition 5 LightColor.Red) (lightPosition 30 LightColor.Red) = 41 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_5th_and_30th_red_light_l2421_242144


namespace NUMINAMATH_CALUDE_green_knights_magical_fraction_l2421_242124

/-- Represents the fraction of knights of a certain color who are magical -/
structure MagicalFraction where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Represents the distribution of knights in the kingdom -/
structure KnightDistribution where
  green_fraction : Rat
  yellow_fraction : Rat
  magical_fraction : Rat
  green_magical : MagicalFraction
  yellow_magical : MagicalFraction
  green_fraction_valid : green_fraction = 3 / 8
  yellow_fraction_valid : yellow_fraction = 5 / 8
  fractions_sum_to_one : green_fraction + yellow_fraction = 1
  magical_fraction_valid : magical_fraction = 1 / 5
  green_thrice_yellow : green_magical.numerator * yellow_magical.denominator = 
                        3 * yellow_magical.numerator * green_magical.denominator

theorem green_knights_magical_fraction 
  (k : KnightDistribution) : k.green_magical = MagicalFraction.mk 12 35 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_green_knights_magical_fraction_l2421_242124


namespace NUMINAMATH_CALUDE_unique_solution_aabb_equation_l2421_242163

theorem unique_solution_aabb_equation :
  ∃! (a b n : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1000 * a + 100 * a + 10 * b + b = n^4 - 6 * n^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_aabb_equation_l2421_242163


namespace NUMINAMATH_CALUDE_school_dinner_theatre_attendance_l2421_242151

theorem school_dinner_theatre_attendance
  (child_ticket_price : ℕ)
  (adult_ticket_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (h1 : child_ticket_price = 6)
  (h2 : adult_ticket_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_revenue = 1875) :
  ∃ (child_tickets adult_tickets : ℕ),
    child_tickets + adult_tickets = total_tickets ∧
    child_tickets * child_ticket_price + adult_tickets * adult_ticket_price = total_revenue ∧
    adult_tickets = 175 :=
sorry

end NUMINAMATH_CALUDE_school_dinner_theatre_attendance_l2421_242151


namespace NUMINAMATH_CALUDE_middle_is_four_l2421_242186

/-- Represents a trio of integers -/
structure Trio :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- Checks if a trio satisfies the given conditions -/
def validTrio (t : Trio) : Prop :=
  t.left < t.middle ∧ t.middle < t.right ∧ t.left + t.middle + t.right = 15

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.left = t.left

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.right = t.right

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.middle = t.middle

/-- The main theorem stating that the middle number must be 4 -/
theorem middle_is_four :
  ∀ t : Trio, validTrio t →
    caseyUncertain t → tracyUncertain t → stacyUncertain t →
    t.middle = 4 :=
by sorry

end NUMINAMATH_CALUDE_middle_is_four_l2421_242186


namespace NUMINAMATH_CALUDE_equipment_production_l2421_242198

theorem equipment_production (total : ℕ) (sample_size : ℕ) (sample_a : ℕ) 
  (h_total : total = 4800)
  (h_sample_size : sample_size = 80)
  (h_sample_a : sample_a = 50) :
  total - (total * sample_a / sample_size) = 1800 := by
sorry

end NUMINAMATH_CALUDE_equipment_production_l2421_242198


namespace NUMINAMATH_CALUDE_discount_rate_example_l2421_242121

/-- Given a bag with a marked price and a selling price, calculate the discount rate. -/
def discount_rate (marked_price selling_price : ℚ) : ℚ :=
  (marked_price - selling_price) / marked_price * 100

/-- Theorem: The discount rate for a bag marked at $80 and sold for $68 is 15%. -/
theorem discount_rate_example : discount_rate 80 68 = 15 := by
  sorry

end NUMINAMATH_CALUDE_discount_rate_example_l2421_242121


namespace NUMINAMATH_CALUDE_quadratic_roots_l2421_242138

/-- Given a quadratic function f(x) = ax² - 2ax + c where a ≠ 0,
    if f(3) = 0, then the solutions to f(x) = 0 are x₁ = -1 and x₂ = 3 -/
theorem quadratic_roots (a c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + c
  f 3 = 0 → (∀ x, f x = 0 ↔ x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2421_242138


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2421_242166

/-- The general form equation of a line perpendicular to 2x+y-5=0 and passing through (1,2) is x-2y+3=0 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (∃ (m b : ℝ), y = m * x + b ∧ m * 2 = -1) →  -- perpendicular line condition
  (1 : ℝ) - 2 * (2 : ℝ) + 3 = 0 →              -- point (1,2) satisfies the equation
  x - 2 * y + 3 = 0                            -- the equation we want to prove
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2421_242166


namespace NUMINAMATH_CALUDE_difference_smallest_three_largest_two_l2421_242149

def smallest_three_digit_number : ℕ := 100
def largest_two_digit_number : ℕ := 99

theorem difference_smallest_three_largest_two : 
  smallest_three_digit_number - largest_two_digit_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_smallest_three_largest_two_l2421_242149


namespace NUMINAMATH_CALUDE_quadratic_even_iff_m_eq_neg_two_l2421_242158

/-- A quadratic function f(x) = mx^2 + (m+2)mx + 2 is even if and only if m = -2 -/
theorem quadratic_even_iff_m_eq_neg_two (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (m + 2) * m * x + 2 = m * (-x)^2 + (m + 2) * m * (-x) + 2) ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_m_eq_neg_two_l2421_242158


namespace NUMINAMATH_CALUDE_farm_horses_and_cows_l2421_242146

theorem farm_horses_and_cows (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 3 * initial_cows →
  (initial_horses - 15) * 3 = 5 * (initial_cows + 15) →
  initial_horses - 15 - (initial_cows + 15) = 30 := by
  sorry

end NUMINAMATH_CALUDE_farm_horses_and_cows_l2421_242146


namespace NUMINAMATH_CALUDE_alice_speed_l2421_242187

theorem alice_speed (total_distance : ℝ) (abel_speed : ℝ) (time_difference : ℝ) (alice_delay : ℝ) :
  total_distance = 1000 →
  abel_speed = 50 →
  time_difference = 6 →
  alice_delay = 1 →
  (total_distance / abel_speed + alice_delay) - (total_distance / abel_speed) = time_difference →
  total_distance / ((total_distance / abel_speed) + time_difference) = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_alice_speed_l2421_242187


namespace NUMINAMATH_CALUDE_blue_glass_ball_probability_l2421_242153

/-- The probability of drawing a blue glass ball given that a glass ball is drawn -/
theorem blue_glass_ball_probability :
  let total_balls : ℕ := 5 + 11
  let red_glass_balls : ℕ := 2
  let blue_glass_balls : ℕ := 4
  let total_glass_balls : ℕ := red_glass_balls + blue_glass_balls
  (blue_glass_balls : ℚ) / total_glass_balls = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_blue_glass_ball_probability_l2421_242153


namespace NUMINAMATH_CALUDE_factorization_equality_l2421_242136

theorem factorization_equality (a b : ℝ) : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2421_242136


namespace NUMINAMATH_CALUDE_pure_imaginary_z_implies_a_plus_2i_modulus_l2421_242141

theorem pure_imaginary_z_implies_a_plus_2i_modulus (a : ℝ) : 
  let z : ℂ := (a + 3 * Complex.I) / (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (a + 2 * Complex.I) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_implies_a_plus_2i_modulus_l2421_242141


namespace NUMINAMATH_CALUDE_systematic_sampling_size_l2421_242104

/-- Proves that the sample size for systematic sampling is 6 given the conditions of the problem -/
theorem systematic_sampling_size (total_population : Nat) (n : Nat) 
  (h1 : total_population = 36)
  (h2 : total_population % n = 0)
  (h3 : (total_population - 1) % (n + 1) = 0) : 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_size_l2421_242104


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2421_242164

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)
variable (x y : ℝ)

theorem vector_equation_solution (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_eq : (2*x - y) • a + 4 • b = 5 • a + (x - 2*y) • b) : 
  x + y = 1 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2421_242164


namespace NUMINAMATH_CALUDE_cricket_average_theorem_l2421_242150

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  matches_played : ℕ
  total_runs : ℕ
  
/-- Calculates the batting average -/
def batting_average (stats : CricketStats) : ℚ :=
  stats.total_runs / stats.matches_played

/-- Theorem: If a player has played 5 matches and scoring 69 runs in the next match
    would bring their batting average to 54, then their current batting average is 51 -/
theorem cricket_average_theorem (stats : CricketStats) 
    (h1 : stats.matches_played = 5)
    (h2 : batting_average ⟨stats.matches_played + 1, stats.total_runs + 69⟩ = 54) :
  batting_average stats = 51 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_theorem_l2421_242150


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2421_242142

theorem point_in_fourth_quadrant :
  ∀ x : ℝ, (x^2 + 2 > 0) ∧ (-3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2421_242142


namespace NUMINAMATH_CALUDE_intersection_equals_half_open_interval_l2421_242175

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 < 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- State the theorem
theorem intersection_equals_half_open_interval :
  M_intersect_N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_half_open_interval_l2421_242175


namespace NUMINAMATH_CALUDE_octal_perfect_square_b_is_one_l2421_242140

/-- Represents a digit in base 8 -/
def OctalDigit := { n : Nat // n < 8 }

/-- Converts a number from base 8 to decimal -/
def octalToDecimal (a b c : OctalDigit) : Nat :=
  512 * a.val + 192 + 8 * b.val + c.val

/-- Represents a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

theorem octal_perfect_square_b_is_one
  (a : OctalDigit)
  (h_a : a.val ≠ 0)
  (b : OctalDigit)
  (c : OctalDigit) :
  isPerfectSquare (octalToDecimal a b c) → b.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_octal_perfect_square_b_is_one_l2421_242140


namespace NUMINAMATH_CALUDE_three_numbers_problem_l2421_242103

theorem three_numbers_problem (x y z : ℝ) : 
  x = 0.8 * y ∧ 
  y / z = 0.5 / (9/20) ∧ 
  x + z = y + 70 →
  x = 80 ∧ y = 100 ∧ z = 90 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l2421_242103


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2421_242128

noncomputable def triangle_abc (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a - b) * (Real.sin A + Real.sin B) = c * (Real.sin A - Real.sin C) ∧
  b = 2 ∧
  a = 2 * Real.sqrt 6 / 3

theorem triangle_abc_properties {a b c A B C : ℝ} 
  (h : triangle_abc a b c A B C) : 
  (∃ (R : ℝ), 2 * R = 4 * Real.sqrt 3 / 3) ∧ 
  (∃ (area : ℝ), area = Real.sqrt 3 / 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2421_242128


namespace NUMINAMATH_CALUDE_girls_in_circle_l2421_242168

/-- The number of girls in a circle of children, given specific conditions. -/
def number_of_girls (total : ℕ) (holding_boys_hand : ℕ) (holding_girls_hand : ℕ) : ℕ :=
  (2 * holding_girls_hand + holding_boys_hand - total) / 2

/-- Theorem stating that the number of girls in the circle is 24. -/
theorem girls_in_circle : number_of_girls 40 22 30 = 24 := by
  sorry

#eval number_of_girls 40 22 30

end NUMINAMATH_CALUDE_girls_in_circle_l2421_242168


namespace NUMINAMATH_CALUDE_largest_prime_divisor_check_l2421_242159

theorem largest_prime_divisor_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  Nat.Prime n → ∀ p, Nat.Prime p ∧ p ≤ 31 → ¬(p ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_check_l2421_242159


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2421_242190

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def drawn_balls : ℕ := 5
def red_balls_drawn : ℕ := 2

theorem probability_two_red_balls :
  (Nat.choose red_balls red_balls_drawn * Nat.choose white_balls (drawn_balls - red_balls_drawn)) /
  Nat.choose total_balls drawn_balls = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2421_242190


namespace NUMINAMATH_CALUDE_clock_cost_price_l2421_242133

theorem clock_cost_price (total_clocks : ℕ) (clocks_10_percent : ℕ) (clocks_20_percent : ℕ)
  (price_difference : ℝ) :
  total_clocks = 90 →
  clocks_10_percent = 40 →
  clocks_20_percent = 50 →
  price_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price = 80 ∧
    (clocks_10_percent : ℝ) * cost_price * 1.1 +
    (clocks_20_percent : ℝ) * cost_price * 1.2 -
    (total_clocks : ℝ) * cost_price * 1.15 = price_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l2421_242133


namespace NUMINAMATH_CALUDE_age_of_B_l2421_242116

/-- Given the ages of four people A, B, C, and D, prove that B's age is 27 years. -/
theorem age_of_B (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 28 →
  (a + c) / 2 = 29 →
  (2 * b + 3 * d) / 5 = 27 →
  b = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_of_B_l2421_242116


namespace NUMINAMATH_CALUDE_abc_product_l2421_242122

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 30) 
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (420 : ℚ) / (a * b * c) = 1) : 
  a * b * c = 450 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l2421_242122


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2421_242177

theorem square_difference_of_sum_and_diff (x y : ℕ) 
  (sum_eq : x + y = 70) 
  (diff_eq : x - y = 20) 
  (pos_x : x > 0) 
  (pos_y : y > 0) : 
  x^2 - y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2421_242177


namespace NUMINAMATH_CALUDE_curve_is_ellipse_l2421_242105

open Real

/-- Given that θ is an internal angle of an oblique triangle and 
    F: x²sin²θcos²θ + y²sin²θ = cos²θ is the equation of a curve,
    prove that F represents an ellipse with foci on the x-axis and eccentricity sin θ. -/
theorem curve_is_ellipse (θ : ℝ) (h1 : 0 < θ ∧ θ < π) 
  (h2 : ∀ (x y : ℝ), x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2 → 
    ∃ (a b : ℝ), 0 < b ∧ b < a ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (a^2 - b^2) / a^2 = (sin θ)^2) : 
  ∃ (a b : ℝ), 0 < b ∧ b < a ∧ 
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ 
      x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2) ∧
    (a^2 - b^2) / a^2 = (sin θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_l2421_242105


namespace NUMINAMATH_CALUDE_solve_equation_l2421_242176

/-- Given the equation fp - w = 20000, where f = 10 and w = 10 + 250i, prove that p = 2001 + 25i -/
theorem solve_equation (f w p : ℂ) : 
  f = 10 → w = 10 + 250 * I → f * p - w = 20000 → p = 2001 + 25 * I := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2421_242176


namespace NUMINAMATH_CALUDE_product_pqr_equals_864_l2421_242120

theorem product_pqr_equals_864 (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 36)
  (h3 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 := by
  sorry

end NUMINAMATH_CALUDE_product_pqr_equals_864_l2421_242120


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2421_242189

/-- An isosceles triangle with two angles in the ratio 1:4 has a vertex angle of either 20 or 120 degrees. -/
theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
  a + b + c = 180 →  -- Sum of angles in a triangle
  a = b →  -- Isosceles triangle condition
  (c = a ∧ b = 4 * a) ∨ (a = 4 * c ∧ b = 4 * c) →  -- Ratio condition
  c = 20 ∨ c = 120 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2421_242189


namespace NUMINAMATH_CALUDE_max_roses_for_680_l2421_242193

/-- Represents the pricing structure for roses -/
structure RosePricing where
  individual_price : ℚ
  dozen_price : ℚ
  two_dozen_price : ℚ
  five_dozen_price : ℚ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the maximum number of roses that can be purchased given a budget -/
def max_roses_purchased (pricing : RosePricing) (budget : ℚ) : ℕ :=
  sorry

/-- The specific pricing structure given in the problem -/
def problem_pricing : RosePricing :=
  { individual_price := 9/2,
    dozen_price := 36,
    two_dozen_price := 50,
    five_dozen_price := 110,
    discount_rate := 1/10,
    discount_threshold := 36 }

theorem max_roses_for_680 :
  max_roses_purchased problem_pricing 680 = 364 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l2421_242193


namespace NUMINAMATH_CALUDE_remainder_3_pow_20_mod_5_l2421_242139

theorem remainder_3_pow_20_mod_5 : 3^20 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_20_mod_5_l2421_242139


namespace NUMINAMATH_CALUDE_sequence_ratio_l2421_242110

/-- Given two sequences where (-1, a₁, a₂, 8) form an arithmetic sequence
and (-1, b₁, b₂, b₃, -4) form a geometric sequence,
prove that (a₁ * a₂) / b₂ = -5 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-1 : ℝ) - a₁ = a₁ - a₂) → 
  (a₂ - a₁ = 8 - a₂) → 
  (b₁ / (-1 : ℝ) = b₂ / b₁) → 
  (b₂ / b₁ = b₃ / b₂) → 
  (b₃ / b₂ = (-4 : ℝ) / b₃) → 
  (a₁ * a₂) / b₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2421_242110


namespace NUMINAMATH_CALUDE_resistor_value_l2421_242108

/-- The resistance of a single resistor in a circuit where three identical resistors are initially 
    in series, and then connected in parallel, such that the change in total resistance is 10 Ω. -/
theorem resistor_value (R : ℝ) : 
  (3 * R - R / 3 = 10) → R = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_resistor_value_l2421_242108


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l2421_242180

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Given conditions for the problem -/
structure ProblemConditions (C : Ellipse) where
  eccentricity : ℝ
  focusDistance : ℝ
  h1 : eccentricity = 1/2
  h2 : focusDistance = 2 * Real.sqrt 2

/-- The equation of line l -/
def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 2 * Real.sqrt 3 = 0

/-- Main theorem statement -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (cond : ProblemConditions C)
  (T_y_coord : ℝ)
  (h_T_y : T_y_coord = 6 * Real.sqrt 3) :
  (∀ x y, x^2 / 16 + y^2 / 12 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (∀ x y, line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l2421_242180


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l2421_242160

theorem integral_reciprocal_plus_one : ∫ x in (0:ℝ)..1, 1 / (1 + x) = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l2421_242160


namespace NUMINAMATH_CALUDE_sara_lunch_cost_l2421_242174

/-- The cost of Sara's lunch -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem: Sara's lunch cost $10.46 -/
theorem sara_lunch_cost :
  lunch_cost 5.36 5.10 = 10.46 := by
  sorry

end NUMINAMATH_CALUDE_sara_lunch_cost_l2421_242174


namespace NUMINAMATH_CALUDE_bridge_renovation_problem_l2421_242197

theorem bridge_renovation_problem (bridge_length : ℝ) (efficiency_increase : ℝ) (days_ahead : ℝ) 
  (h1 : bridge_length = 36)
  (h2 : efficiency_increase = 0.5)
  (h3 : days_ahead = 2) :
  ∃ x : ℝ, x = 6 ∧ 
    bridge_length / x = bridge_length / ((1 + efficiency_increase) * x) + days_ahead :=
by sorry

end NUMINAMATH_CALUDE_bridge_renovation_problem_l2421_242197


namespace NUMINAMATH_CALUDE_fourth_term_binomial_expansion_l2421_242195

theorem fourth_term_binomial_expansion 
  (a x : ℝ) (ha : a ≠ 0) (hx : x > 0) :
  let binomial := (2*a/Real.sqrt x - Real.sqrt x/(2*a^2))^8
  let fourth_term := (Nat.choose 8 3) * (2*a/Real.sqrt x)^5 * (-Real.sqrt x/(2*a^2))^3
  fourth_term = -4/(a*x) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_binomial_expansion_l2421_242195


namespace NUMINAMATH_CALUDE_mode_of_shoe_sizes_l2421_242130

def shoe_sizes : List ℝ := [24, 24.5, 25, 25.5, 26]
def sales : List ℕ := [2, 5, 3, 6, 4]

def mode (sizes : List ℝ) (counts : List ℕ) : ℝ :=
  let pairs := List.zip sizes counts
  let max_count := pairs.map Prod.snd |>.maximum?
  match max_count with
  | none => 0  -- Default value if the list is empty
  | some mc => (pairs.filter (fun p => p.2 = mc)).head!.1

theorem mode_of_shoe_sizes :
  mode shoe_sizes sales = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_shoe_sizes_l2421_242130


namespace NUMINAMATH_CALUDE_number_minus_division_equals_l2421_242184

theorem number_minus_division_equals (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 := by
  sorry

end NUMINAMATH_CALUDE_number_minus_division_equals_l2421_242184


namespace NUMINAMATH_CALUDE_special_number_value_l2421_242107

/-- Represents a positive integer with specific properties in different bases -/
def SpecialNumber (n : ℕ+) : Prop :=
  ∃ (X Y : ℕ),
    X < 8 ∧ Y < 9 ∧
    n = 8 * X + Y ∧
    n = 9 * Y + X

/-- The unique value of the special number in base 10 -/
theorem special_number_value :
  ∀ n : ℕ+, SpecialNumber n → n = 71 := by
  sorry

end NUMINAMATH_CALUDE_special_number_value_l2421_242107


namespace NUMINAMATH_CALUDE_square_difference_existence_l2421_242155

theorem square_difference_existence (n : ℤ) : 
  (∃ a b : ℤ, n + a^2 = b^2) ↔ n % 4 ≠ 2 := by sorry

end NUMINAMATH_CALUDE_square_difference_existence_l2421_242155
