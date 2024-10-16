import Mathlib

namespace NUMINAMATH_CALUDE_walter_coins_percentage_l3052_305297

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "half-dollar" => 50
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels half_dollars : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  half_dollars * coin_value "half-dollar"

/-- Converts cents to percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem walter_coins_percentage :
  cents_to_percentage (total_value 3 2 1) = 63 / 100 := by
  sorry

end NUMINAMATH_CALUDE_walter_coins_percentage_l3052_305297


namespace NUMINAMATH_CALUDE_mass_percentage_Al_approx_l3052_305211

-- Define atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_S : ℝ := 32.06
def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_C : ℝ := 12.01
def atomic_mass_O : ℝ := 16.00
def atomic_mass_K : ℝ := 39.10
def atomic_mass_Cl : ℝ := 35.45

-- Define molar masses of compounds
def molar_mass_Al2S3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_S
def molar_mass_CaCO3 : ℝ := atomic_mass_Ca + atomic_mass_C + 3 * atomic_mass_O
def molar_mass_KCl : ℝ := atomic_mass_K + atomic_mass_Cl

-- Define moles of compounds in the mixture
def moles_Al2S3 : ℝ := 2
def moles_CaCO3 : ℝ := 3
def moles_KCl : ℝ := 5

-- Define total mass of the mixture
def total_mass : ℝ := moles_Al2S3 * molar_mass_Al2S3 + moles_CaCO3 * molar_mass_CaCO3 + moles_KCl * molar_mass_KCl

-- Define mass of Al in the mixture
def mass_Al : ℝ := 2 * moles_Al2S3 * atomic_mass_Al

-- Theorem: The mass percentage of Al in the mixture is approximately 11.09%
theorem mass_percentage_Al_approx (ε : ℝ) (h : ε > 0) : 
  ∃ δ : ℝ, δ > 0 ∧ |mass_Al / total_mass * 100 - 11.09| < δ :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_Al_approx_l3052_305211


namespace NUMINAMATH_CALUDE_ordering_abc_l3052_305221

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define a, b, and c
noncomputable def a : ℝ := log2 9 - log2 (Real.sqrt 3)
noncomputable def b : ℝ := 1 + log2 (Real.sqrt 7)
noncomputable def c : ℝ := 1/2 + log2 (Real.sqrt 13)

-- Theorem statement
theorem ordering_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l3052_305221


namespace NUMINAMATH_CALUDE_max_small_squares_in_large_square_l3052_305214

/-- The side length of the large square -/
def large_square_side : ℕ := 8

/-- The side length of the small squares -/
def small_square_side : ℕ := 2

/-- The maximum number of non-overlapping small squares that can fit inside the large square -/
def max_small_squares : ℕ := (large_square_side / small_square_side) ^ 2

theorem max_small_squares_in_large_square :
  max_small_squares = 16 :=
sorry

end NUMINAMATH_CALUDE_max_small_squares_in_large_square_l3052_305214


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l3052_305243

theorem room_breadth_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 18 →
  carpet_width = 0.75 →
  carpet_cost_per_meter = 4.50 →
  total_cost = 810 →
  (total_cost / carpet_cost_per_meter) * carpet_width / room_length = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l3052_305243


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3052_305246

/-- A coloring of positive integers -/
def Coloring := ℕ+ → Fin 2009

/-- Predicate for a valid coloring satisfying the problem conditions -/
def ValidColoring (f : Coloring) : Prop :=
  (∀ c : Fin 2009, Set.Infinite {n : ℕ+ | f n = c}) ∧
  (∀ a b c : ℕ+, ∀ i j k : Fin 2009,
    i ≠ j ∧ j ≠ k ∧ i ≠ k → f a = i ∧ f b = j ∧ f c = k → a * b ≠ c)

/-- Theorem stating the existence of a valid coloring -/
theorem exists_valid_coloring : ∃ f : Coloring, ValidColoring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3052_305246


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l3052_305258

theorem absolute_value_theorem (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) :
  x - q = 3 - 2*q := by sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l3052_305258


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3052_305216

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 * 1 ≤ 45 * n := by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3052_305216


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3052_305296

/-- The compound interest rate that satisfies the given conditions -/
def interest_rate : ℝ := 20

/-- The principal amount (initial deposit) -/
noncomputable def principal : ℝ := 
  3000 / (1 + interest_rate / 100) ^ 3

theorem compound_interest_rate : 
  (principal * (1 + interest_rate / 100) ^ 3 = 3000) ∧ 
  (principal * (1 + interest_rate / 100) ^ 4 = 3600) := by
  sorry

#check compound_interest_rate

end NUMINAMATH_CALUDE_compound_interest_rate_l3052_305296


namespace NUMINAMATH_CALUDE_black_population_west_percentage_l3052_305207

def black_population_ne : ℕ := 6
def black_population_mw : ℕ := 7
def black_population_south : ℕ := 18
def black_population_west : ℕ := 4

def total_black_population : ℕ := black_population_ne + black_population_mw + black_population_south + black_population_west

def percentage_in_west : ℚ := black_population_west / total_black_population

theorem black_population_west_percentage :
  ∃ (p : ℚ), abs (percentage_in_west - p) < 1/100 ∧ p = 11/100 := by
  sorry

end NUMINAMATH_CALUDE_black_population_west_percentage_l3052_305207


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3052_305288

/-- Given a curve in polar coordinates ρ = 4sin θ, prove its equivalence to the rectangular form x² + y² - 4y = 0 -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (ρ = 4 * Real.sin θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  (x^2 + y^2 - 4*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3052_305288


namespace NUMINAMATH_CALUDE_binary_101011_eq_43_l3052_305219

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101011_eq_43 : 
  binary_to_decimal [true, true, false, true, false, true] = 43 := by
sorry

end NUMINAMATH_CALUDE_binary_101011_eq_43_l3052_305219


namespace NUMINAMATH_CALUDE_total_time_cutting_grass_l3052_305235

-- Define the time to cut one lawn in minutes
def time_per_lawn : ℕ := 30

-- Define the number of lawns cut on Saturday
def lawns_saturday : ℕ := 8

-- Define the number of lawns cut on Sunday
def lawns_sunday : ℕ := 8

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_time_cutting_grass :
  (time_per_lawn * (lawns_saturday + lawns_sunday)) / minutes_per_hour = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_time_cutting_grass_l3052_305235


namespace NUMINAMATH_CALUDE_exists_table_with_square_corner_sums_l3052_305213

/-- Represents a 100 x 100 table of natural numbers -/
def Table := Fin 100 → Fin 100 → ℕ

/-- Checks if all numbers in the same row or column are different -/
def all_different (t : Table) : Prop :=
  ∀ i j i' j', i ≠ i' ∨ j ≠ j' → t i j ≠ t i' j'

/-- Checks if the sum of numbers in angle cells of a square submatrix is a square number -/
def corner_sum_is_square (t : Table) : Prop :=
  ∀ i j n, ∃ k : ℕ, 
    t i j + t i (j + n) + t (i + n) j + t (i + n) (j + n) = k * k

/-- The main theorem stating the existence of a table satisfying all conditions -/
theorem exists_table_with_square_corner_sums : 
  ∃ t : Table, all_different t ∧ corner_sum_is_square t := by
  sorry

end NUMINAMATH_CALUDE_exists_table_with_square_corner_sums_l3052_305213


namespace NUMINAMATH_CALUDE_opposite_and_reciprocal_expression_l3052_305242

theorem opposite_and_reciprocal_expression (x y a b : ℝ) 
  (h1 : x = -y) 
  (h2 : a * b = 1) : 
  x + y - 3 / (a * b) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_and_reciprocal_expression_l3052_305242


namespace NUMINAMATH_CALUDE_f_min_at_neg_seven_l3052_305253

/-- The quadratic function we're minimizing -/
def f (x : ℝ) := x^2 + 14*x - 20

/-- The theorem stating that f attains its minimum at x = -7 -/
theorem f_min_at_neg_seven :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_seven_l3052_305253


namespace NUMINAMATH_CALUDE_total_pencils_l3052_305278

theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (desk_added : ℕ) 
  (h1 : drawer = 43)
  (h2 : desk_initial = 19)
  (h3 : desk_added = 16) :
  drawer + desk_initial + desk_added = 78 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l3052_305278


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3052_305250

theorem sqrt_expression_equality : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 20 = 2 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3052_305250


namespace NUMINAMATH_CALUDE_contestant_speaking_orders_l3052_305247

theorem contestant_speaking_orders :
  let total_contestants : ℕ := 6
  let restricted_contestant : ℕ := 1
  let available_positions : ℕ := total_contestants - 2

  available_positions * Nat.factorial (total_contestants - restricted_contestant) = 480 :=
by sorry

end NUMINAMATH_CALUDE_contestant_speaking_orders_l3052_305247


namespace NUMINAMATH_CALUDE_trajectory_and_line_theorem_l3052_305232

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 49/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1/4

-- Define the trajectory of P
def trajectory_P (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = -2

theorem trajectory_and_line_theorem :
  ∃ k : ℝ, k^2 = 2 ∧
  (∀ x y : ℝ, trajectory_P x y →
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
      trajectory_P x₁ y₁ ∧ trajectory_P x₂ y₂ ∧
      dot_product_condition x₁ y₁ x₂ y₂)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_line_theorem_l3052_305232


namespace NUMINAMATH_CALUDE_sheila_work_hours_l3052_305251

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mwf_hours : ℝ  -- Hours worked on Monday, Wednesday, and Friday combined
  tt_hours : ℝ   -- Hours worked on Tuesday and Thursday combined
  hourly_rate : ℝ -- Hourly rate in dollars
  weekly_earnings : ℝ -- Total weekly earnings in dollars

/-- Theorem stating Sheila's work hours on Monday, Wednesday, and Friday --/
theorem sheila_work_hours (s : WorkSchedule) 
  (h1 : s.tt_hours = 12)  -- 6 hours each on Tuesday and Thursday
  (h2 : s.hourly_rate = 14)  -- $14 per hour
  (h3 : s.weekly_earnings = 504)  -- $504 per week
  : s.mwf_hours = 24 := by
  sorry


end NUMINAMATH_CALUDE_sheila_work_hours_l3052_305251


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l3052_305248

theorem choose_four_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l3052_305248


namespace NUMINAMATH_CALUDE_fourth_hour_highest_speed_l3052_305260

def distance_traveled : Fin 7 → ℝ
| 0 => 70
| 1 => 95
| 2 => 85
| 3 => 100
| 4 => 90
| 5 => 85
| 6 => 75

def average_speed (hour : Fin 7) : ℝ := distance_traveled hour

theorem fourth_hour_highest_speed :
  ∀ (hour : Fin 7), average_speed 3 ≥ average_speed hour :=
by sorry

end NUMINAMATH_CALUDE_fourth_hour_highest_speed_l3052_305260


namespace NUMINAMATH_CALUDE_janet_farmland_acreage_l3052_305289

/-- Represents Janet's farm and fertilizer production system -/
structure FarmSystem where
  horses : ℕ
  fertilizer_per_horse : ℕ
  fertilizer_per_acre : ℕ
  acres_spread_per_day : ℕ
  days_to_fertilize : ℕ

/-- Calculates the total acreage of Janet's farmland -/
def total_acreage (farm : FarmSystem) : ℕ :=
  farm.acres_spread_per_day * farm.days_to_fertilize

/-- Theorem: Janet's farmland is 100 acres given the specified conditions -/
theorem janet_farmland_acreage :
  let farm := FarmSystem.mk 80 5 400 4 25
  total_acreage farm = 100 := by
  sorry


end NUMINAMATH_CALUDE_janet_farmland_acreage_l3052_305289


namespace NUMINAMATH_CALUDE_parabola_vertex_l3052_305236

/-- Given a parabola with equation y = -x^2 + ax + b where the solution to y ≤ 0
    is (-∞, -1] ∪ [7, ∞), prove that the vertex of this parabola is (3, 16) -/
theorem parabola_vertex (a b : ℝ) :
  (∀ x, -x^2 + a*x + b ≤ 0 ↔ x ≤ -1 ∨ x ≥ 7) →
  ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = 16 ∧
    ∀ x, -x^2 + a*x + b = -(x - vertex_x)^2 + vertex_y :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3052_305236


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3052_305264

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 5 → 
    b = 12 → 
    c^2 = a^2 + b^2 → 
    c = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3052_305264


namespace NUMINAMATH_CALUDE_total_balls_in_box_l3052_305244

theorem total_balls_in_box (black_balls : ℕ) (white_balls : ℕ) : 
  black_balls = 8 →
  white_balls = 6 * black_balls →
  black_balls + white_balls = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l3052_305244


namespace NUMINAMATH_CALUDE_tangent_points_parameter_l3052_305284

/-- Given a circle and two tangent points, prove that the parameter 'a' has specific values -/
theorem tangent_points_parameter (a : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*x + 2*y - 1 = 0) →  -- Circle equation
  (x₁^2 + y₁^2 - 2*a*x₁ + 2*y₁ - 1 = 0) →  -- M is on the circle
  (x₂^2 + y₂^2 - 2*a*x₂ + 2*y₂ - 1 = 0) →  -- N is on the circle
  ((y₁ - (-1)) / (x₁ - a))^2 = (((-5) - a)^2 + (a - (-1))^2) / ((x₁ - (-5))^2 + (y₁ - a)^2) →  -- M is a tangent point
  ((y₂ - (-1)) / (x₂ - a))^2 = (((-5) - a)^2 + (a - (-1))^2) / ((x₂ - (-5))^2 + (y₂ - a)^2) →  -- N is a tangent point
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0 →  -- Given condition
  a = 3 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_tangent_points_parameter_l3052_305284


namespace NUMINAMATH_CALUDE_peters_vacation_savings_l3052_305225

/-- Peter's vacation savings problem -/
theorem peters_vacation_savings 
  (current_savings : ℕ) 
  (monthly_savings : ℕ) 
  (months_to_wait : ℕ) 
  (h1 : current_savings = 2900)
  (h2 : monthly_savings = 700)
  (h3 : months_to_wait = 3) :
  current_savings + monthly_savings * months_to_wait = 5000 :=
by sorry

end NUMINAMATH_CALUDE_peters_vacation_savings_l3052_305225


namespace NUMINAMATH_CALUDE_square_difference_l3052_305217

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3052_305217


namespace NUMINAMATH_CALUDE_max_peak_consumption_l3052_305277

theorem max_peak_consumption (original_price peak_price off_peak_price total_consumption : ℝ)
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : ∀ x : ℝ, x ≥ 0 ∧ x ≤ total_consumption →
    (x * peak_price + (total_consumption - x) * off_peak_price) ≤ 0.9 * (total_consumption * original_price)) :
  ∃ max_peak : ℝ, max_peak = 118 ∧
    ∀ y : ℝ, y > max_peak →
      (y * peak_price + (total_consumption - y) * off_peak_price) > 0.9 * (total_consumption * original_price) :=
by sorry

end NUMINAMATH_CALUDE_max_peak_consumption_l3052_305277


namespace NUMINAMATH_CALUDE_correct_stool_height_l3052_305292

/-- Calculates the height of a stool needed to reach a light bulb. -/
def stool_height (ceiling_height room_height alice_height alice_reach book_thickness : ℝ) : ℝ :=
  ceiling_height - room_height - (alice_height + alice_reach + book_thickness)

/-- Theorem stating the correct height of the stool needed. -/
theorem correct_stool_height :
  let ceiling_height : ℝ := 300
  let light_bulb_below_ceiling : ℝ := 15
  let alice_height : ℝ := 160
  let alice_reach : ℝ := 50
  let book_thickness : ℝ := 5
  stool_height ceiling_height light_bulb_below_ceiling alice_height alice_reach book_thickness = 70 := by
  sorry

#eval stool_height 300 15 160 50 5

end NUMINAMATH_CALUDE_correct_stool_height_l3052_305292


namespace NUMINAMATH_CALUDE_train_distance_l3052_305280

/-- Proves that a train traveling at a rate of 1 mile per 2 minutes will cover 15 miles in 30 minutes. -/
theorem train_distance (rate : ℚ) (time : ℚ) (distance : ℚ) : 
  rate = 1 / 2 →  -- The train travels 1 mile in 2 minutes
  time = 30 →     -- We want to know the distance traveled in 30 minutes
  distance = rate * time →  -- Distance is calculated as rate times time
  distance = 15 :=  -- The train will travel 15 miles
by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3052_305280


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3052_305255

theorem quadratic_roots_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 2/b)^2 - p*(a + 2/b) + r = 0) →
  ((b + 2/a)^2 - p*(b + 2/a) + r = 0) →
  r = 25/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3052_305255


namespace NUMINAMATH_CALUDE_min_product_tangents_acute_triangle_l3052_305294

theorem min_product_tangents_acute_triangle (α β γ : Real) 
  (h_acute : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α < π/2 ∧ β < π/2 ∧ γ < π/2) 
  (h_sum : α + β + γ = π) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ Real.sqrt 27 ∧ 
  (Real.tan α * Real.tan β * Real.tan γ = Real.sqrt 27 ↔ α = π/3 ∧ β = π/3 ∧ γ = π/3) :=
sorry

end NUMINAMATH_CALUDE_min_product_tangents_acute_triangle_l3052_305294


namespace NUMINAMATH_CALUDE_function_range_l3052_305298

/-- The function f(x) = (x^2 - 2x - 3)(x^2 - 2x - 5) has a range of [-1, +∞) -/
theorem function_range (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 - 2*x - 3) * (x^2 - 2*x - 5)
  ∃ (y : ℝ), y ≥ -1 ∧ ∃ (x : ℝ), f x = y :=
by sorry

end NUMINAMATH_CALUDE_function_range_l3052_305298


namespace NUMINAMATH_CALUDE_p_and_q_true_iff_not_p_or_not_q_false_l3052_305282

theorem p_and_q_true_iff_not_p_or_not_q_false (p q : Prop) :
  (p ∧ q) ↔ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_iff_not_p_or_not_q_false_l3052_305282


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l3052_305208

theorem magical_red_knights_fraction (total : ℕ) (total_pos : 0 < total) :
  let red := (2 : ℚ) / 7 * total
  let blue := total - red
  let magical := (1 : ℚ) / 6 * total
  let red_magical_fraction := magical / red
  let blue_magical_fraction := magical / blue
  red_magical_fraction = 2 * blue_magical_fraction →
  red_magical_fraction = 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l3052_305208


namespace NUMINAMATH_CALUDE_prob_at_least_one_male_l3052_305239

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of students to be chosen -/
def num_chosen : ℕ := 2

/-- The probability of choosing at least one male student -/
theorem prob_at_least_one_male :
  (1 : ℚ) - (Nat.choose num_female num_chosen : ℚ) / (Nat.choose total_students num_chosen : ℚ) = 9/10 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_male_l3052_305239


namespace NUMINAMATH_CALUDE_stratified_sampling_school_l3052_305237

/-- Proves that in a stratified sampling of a school, given the total number of students,
    the number of second-year students, and the number of second-year students selected,
    we can determine the total number of students selected. -/
theorem stratified_sampling_school (total : ℕ) (second_year : ℕ) (selected_second_year : ℕ) 
    (h1 : total = 1800) 
    (h2 : second_year = 600) 
    (h3 : selected_second_year = 21) :
    ∃ n : ℕ, n * second_year = selected_second_year * total ∧ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_school_l3052_305237


namespace NUMINAMATH_CALUDE_pancake_covers_center_l3052_305210

-- Define the pan
def Pan : Type := Unit

-- Define the area of the pan
def pan_area : ℝ := 1

-- Define the pancake
def Pancake : Type := Unit

-- Define the property of the pancake being convex
def is_convex (p : Pancake) : Prop := sorry

-- Define the area of the pancake
def pancake_area (p : Pancake) : ℝ := sorry

-- Define the center of the pan
def pan_center (pan : Pan) : Set ℝ := sorry

-- Define the region covered by the pancake
def pancake_region (p : Pancake) : Set ℝ := sorry

-- The theorem to be proved
theorem pancake_covers_center (pan : Pan) (p : Pancake) :
  is_convex p →
  pancake_area p > 1/2 →
  pan_center pan ⊆ pancake_region p :=
sorry

end NUMINAMATH_CALUDE_pancake_covers_center_l3052_305210


namespace NUMINAMATH_CALUDE_probability_of_six_consecutive_heads_l3052_305263

-- Define a coin flip sequence as a list of booleans (true for heads, false for tails)
def CoinFlipSequence := List Bool

-- Function to check if a sequence has at least n consecutive heads
def hasConsecutiveHeads (n : Nat) (seq : CoinFlipSequence) : Bool :=
  sorry

-- Function to generate all possible coin flip sequences of length n
def allSequences (n : Nat) : List CoinFlipSequence :=
  sorry

-- Count the number of sequences with at least n consecutive heads
def countSequencesWithConsecutiveHeads (n : Nat) (seqs : List CoinFlipSequence) : Nat :=
  sorry

-- Theorem to prove
theorem probability_of_six_consecutive_heads :
  let allSeqs := allSequences 9
  let favorableSeqs := countSequencesWithConsecutiveHeads 6 allSeqs
  (favorableSeqs : ℚ) / (allSeqs.length : ℚ) = 49 / 512 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_six_consecutive_heads_l3052_305263


namespace NUMINAMATH_CALUDE_triangle_theorem_l3052_305299

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition a/(√3 * cos A) = c/sin C --/
def condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

/-- The theorem to be proved --/
theorem triangle_theorem (t : Triangle) (h : condition t) (ha : t.a = 6) :
  t.A = π / 3 ∧ 6 < t.b + t.c ∧ t.b + t.c ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3052_305299


namespace NUMINAMATH_CALUDE_camel_cost_l3052_305224

/-- The cost of animals in a market --/
structure AnimalCosts where
  camel : ℕ
  horse : ℕ
  ox : ℕ
  elephant : ℕ

/-- The conditions of the animal costs problem --/
def animal_costs_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 150000

/-- The theorem stating that under the given conditions, a camel costs 6000 --/
theorem camel_cost (costs : AnimalCosts) : 
  animal_costs_conditions costs → costs.camel = 6000 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l3052_305224


namespace NUMINAMATH_CALUDE_root_expression_equality_l3052_305229

/-- Given a cubic polynomial f(t) with roots p, q, r, and expressions for x, y, z,
    prove that xyz - qrx - rpy - pqz = -674 -/
theorem root_expression_equality (p q r : ℝ) : 
  let f : ℝ → ℝ := fun t ↦ t^3 - 2022*t^2 + 2022*t - 337
  let x := (q-1)*((2022 - q)/(r-1) + (2022 - r)/(p-1))
  let y := (r-1)*((2022 - r)/(p-1) + (2022 - p)/(q-1))
  let z := (p-1)*((2022 - p)/(q-1) + (2022 - q)/(r-1))
  f p = 0 ∧ f q = 0 ∧ f r = 0 →
  x*y*z - q*r*x - r*p*y - p*q*z = -674 := by
sorry

end NUMINAMATH_CALUDE_root_expression_equality_l3052_305229


namespace NUMINAMATH_CALUDE_prop_2_prop_3_l3052_305266

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Theorem for proposition ②
theorem prop_2 : 
  parallel α β → subset m α → line_parallel m β :=
sorry

-- Theorem for proposition ③
theorem prop_3 :
  perpendicular n α → perpendicular n β → perpendicular m α → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_3_l3052_305266


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3052_305271

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi →  -- given area
  A = Real.pi * r^2 →  -- definition of circle area
  d = 2 * r →  -- definition of diameter
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3052_305271


namespace NUMINAMATH_CALUDE_problem_proof_l3052_305259

theorem problem_proof (a b : ℝ) (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) : 
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) := by sorry

end NUMINAMATH_CALUDE_problem_proof_l3052_305259


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3052_305252

theorem geometric_sequence_common_ratio : 
  let a : ℕ → ℝ := fun n => (4 : ℝ) ^ (2 * n + 1)
  ∀ n : ℕ, a (n + 1) / a n = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3052_305252


namespace NUMINAMATH_CALUDE_trisha_walk_distance_l3052_305262

/-- The total distance Trisha walked during her vacation in New York City -/
def total_distance (hotel_to_postcard postcard_to_tshirt tshirt_to_hotel : ℝ) : ℝ :=
  hotel_to_postcard + postcard_to_tshirt + tshirt_to_hotel

/-- Theorem stating that Trisha walked 0.89 miles in total -/
theorem trisha_walk_distance :
  total_distance 0.11 0.11 0.67 = 0.89 := by
  sorry

end NUMINAMATH_CALUDE_trisha_walk_distance_l3052_305262


namespace NUMINAMATH_CALUDE_line_intersection_area_ratio_l3052_305238

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Theorem: Given a line y = c - x where 0 < c < 6, intersecting the y-axis at P
    and the line x = 6 at S, if the ratio of the area of triangle QRS to the area
    of triangle QOP is 4:16, then c = 4 -/
theorem line_intersection_area_ratio (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 6) : 
  let l : Line := { m := -1, b := c }
  let P : Point := { x := 0, y := c }
  let S : Point := { x := 6, y := c - 6 }
  let Q : Point := { x := c, y := 0 }
  let R : Point := { x := 6, y := 0 }
  let O : Point := { x := 0, y := 0 }
  triangleArea Q R S / triangleArea Q O P = 4 / 16 →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_area_ratio_l3052_305238


namespace NUMINAMATH_CALUDE_circle_intersection_and_distance_four_points_distance_l3052_305201

/-- The circle equation parameters -/
structure CircleParams where
  m : ℝ
  h : m < 5

/-- The line equation parameters -/
structure LineParams where
  c : ℝ

/-- The theorem statement -/
theorem circle_intersection_and_distance (p : CircleParams) :
  (∃ M N : ℝ × ℝ, 
    (M.1^2 + M.2^2 - 2*M.1 - 4*M.2 + p.m = 0) ∧
    (N.1^2 + N.2^2 - 2*N.1 - 4*N.2 + p.m = 0) ∧
    (M.1 + 2*M.2 - 4 = 0) ∧
    (N.1 + 2*N.2 - 4 = 0) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = (4*Real.sqrt 5/5)^2)) →
  p.m = 4 :=
sorry

/-- The theorem statement for the second part -/
theorem four_points_distance (p : CircleParams) (l : LineParams) :
  (p.m = 4) →
  (∃ A B C D : ℝ × ℝ,
    (A.1^2 + A.2^2 - 2*A.1 - 4*A.2 + p.m = 0) ∧
    (B.1^2 + B.2^2 - 2*B.1 - 4*B.2 + p.m = 0) ∧
    (C.1^2 + C.2^2 - 2*C.1 - 4*C.2 + p.m = 0) ∧
    (D.1^2 + D.2^2 - 2*D.1 - 4*D.2 + p.m = 0) ∧
    ((A.1 - 2*A.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((B.1 - 2*B.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((C.1 - 2*C.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((D.1 - 2*D.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2)) ↔
  (4 - Real.sqrt 5 < l.c ∧ l.c < 2 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_distance_four_points_distance_l3052_305201


namespace NUMINAMATH_CALUDE_distance_traveled_l3052_305281

/-- Represents the number of footprints per meter for Pogo -/
def pogo_footprints_per_meter : ℚ := 4

/-- Represents the number of footprints per meter for Grimzi -/
def grimzi_footprints_per_meter : ℚ := 1 / 2

/-- Represents the total number of footprints left by both creatures -/
def total_footprints : ℕ := 27000

/-- Theorem stating that the distance traveled by both creatures is 6000 meters -/
theorem distance_traveled :
  ∃ (d : ℚ), d * (pogo_footprints_per_meter + grimzi_footprints_per_meter) = total_footprints ∧ d = 6000 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3052_305281


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3052_305230

theorem simplify_and_evaluate (a : ℚ) (h : a = -2) :
  (2 / (a - 1) - 1 / a) / ((a^2 + a) / (a^2 - 2*a + 1)) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3052_305230


namespace NUMINAMATH_CALUDE_tip_difference_calculation_l3052_305203

/-- Calculates the difference in euro cents between a good tip and a bad tip -/
def tip_difference (initial_bill : ℝ) (bad_tip_percent : ℝ) (good_tip_percent : ℝ) 
  (discount_percent : ℝ) (tax_percent : ℝ) (usd_to_eur : ℝ) : ℝ :=
  let discounted_bill := initial_bill * (1 - discount_percent)
  let final_bill := discounted_bill * (1 + tax_percent)
  let bad_tip := final_bill * bad_tip_percent
  let good_tip := final_bill * good_tip_percent
  let difference_usd := good_tip - bad_tip
  let difference_eur := difference_usd * usd_to_eur
  difference_eur * 100  -- Convert to cents

theorem tip_difference_calculation :
  tip_difference 26 0.05 0.20 0.08 0.07 0.85 = 326.33 := by
  sorry

end NUMINAMATH_CALUDE_tip_difference_calculation_l3052_305203


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l3052_305283

/-- Given a square with side length 2b and a line y = -x/3 intersecting it,
    the ratio of the perimeter of one resulting quadrilateral to b is (14 + √13) / 3 -/
theorem square_intersection_perimeter_ratio (b : ℝ) (b_pos : b > 0) :
  let square_vertices := [(-b, -b), (b, -b), (-b, b), (b, b)]
  let line := fun x => -x / 3
  let intersection_points := [
    (b, line b),
    (-b, line (-b))
  ]
  let quadrilateral_vertices := [
    (-b, -b),
    (b, -b),
    (b, line b),
    (-b, line (-b))
  ]
  let perimeter := 
    (b - line b) + (b - line (-b)) + 2*b + 
    Real.sqrt ((2*b)^2 + (line b - line (-b))^2)
  perimeter / b = (14 + Real.sqrt 13) / 3 :=
by sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l3052_305283


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3052_305241

theorem employed_males_percentage (total_population : ℝ) 
  (employed_percentage : ℝ) (employed_females_percentage : ℝ) :
  employed_percentage = 96 →
  employed_females_percentage = 75 →
  (employed_percentage / 100 * (100 - employed_females_percentage) / 100 * 100) = 24 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3052_305241


namespace NUMINAMATH_CALUDE_circles_tangent_line_parallel_l3052_305240

-- Define the types for points, lines, and circles
variable (Point Line Circle : Type)

-- Define the necessary relations and operations
variable (tangent_circles : Circle → Circle → Prop)
variable (tangent_circle_line : Circle → Line → Point → Prop)
variable (tangent_circles_at : Circle → Circle → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (between : Point → Point → Point → Prop)
variable (intersection : Line → Line → Point)
variable (line_through : Point → Point → Line)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem circles_tangent_line_parallel 
  (Γ Γ₁ Γ₂ : Circle) (l : Line) 
  (A A₁ A₂ B₁ B₂ C D₁ D₂ : Point) :
  tangent_circles Γ Γ₁ →
  tangent_circles Γ Γ₂ →
  tangent_circles Γ₁ Γ₂ →
  tangent_circle_line Γ l A →
  tangent_circle_line Γ₁ l A₁ →
  tangent_circle_line Γ₂ l A₂ →
  tangent_circles_at Γ Γ₁ B₁ →
  tangent_circles_at Γ Γ₂ B₂ →
  tangent_circles_at Γ₁ Γ₂ C →
  between A₁ A A₂ →
  D₁ = intersection (line_through A₁ C) (line_through A₂ B₂) →
  D₂ = intersection (line_through A₂ C) (line_through A₁ B₁) →
  parallel (line_through D₁ D₂) l :=
by sorry

end NUMINAMATH_CALUDE_circles_tangent_line_parallel_l3052_305240


namespace NUMINAMATH_CALUDE_solution_set_of_f_gt_zero_l3052_305287

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Theorem statement
theorem solution_set_of_f_gt_zero
  (h_even : is_even f)
  (h_monotone : is_monotone_increasing_on_nonneg f)
  (h_f_one : f 1 = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_f_gt_zero_l3052_305287


namespace NUMINAMATH_CALUDE_square_sum_from_means_l3052_305267

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 24)
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 156) :
  a^2 + b^2 = 1992 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l3052_305267


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l3052_305228

/-- Represents the maximum distance a car can travel with tire swapping -/
def maxDistanceWithSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + (rearTireLife - frontTireLife) / 2

/-- Theorem stating the maximum distance for the given tire lives -/
theorem max_distance_for_given_tires :
  maxDistanceWithSwap 42000 56000 = 48000 := by
  sorry

#eval maxDistanceWithSwap 42000 56000

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l3052_305228


namespace NUMINAMATH_CALUDE_x_wins_more_probability_l3052_305273

/-- Represents a soccer tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (win_probability : ℚ)

/-- Represents the result of the tournament for two specific teams --/
inductive TournamentResult
  | XWinsMore
  | YWinsMore
  | Tie

/-- The probability of team X finishing with more points than team Y --/
def prob_X_wins_more (t : Tournament) : ℚ :=
  sorry

/-- The main theorem stating the probability of team X finishing with more points than team Y --/
theorem x_wins_more_probability (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.win_probability = 1/2) : 
  prob_X_wins_more t = 1/2 :=
sorry

end NUMINAMATH_CALUDE_x_wins_more_probability_l3052_305273


namespace NUMINAMATH_CALUDE_candy_cost_proof_l3052_305233

def candy_problem (first_candy_weight : ℝ) (second_candy_cost : ℝ) (second_candy_weight : ℝ) (mixture_cost : ℝ) : Prop :=
  ∃ (first_candy_cost : ℝ),
    first_candy_weight * first_candy_cost + second_candy_weight * second_candy_cost =
    (first_candy_weight + second_candy_weight) * mixture_cost ∧
    first_candy_cost = 8

theorem candy_cost_proof :
  candy_problem 25 5 50 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l3052_305233


namespace NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l3052_305274

theorem coefficient_x6_in_expansion : 
  let expansion := (1 + X : Polynomial ℤ)^6 * (1 - X : Polynomial ℤ)^6
  (expansion.coeff 6) = -20 := by sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l3052_305274


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3052_305275

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ) 
  (total_marks : ℕ) 
  (correct_sums : ℕ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 65) 
  (h3 : correct_sums = 25) 
  (h4 : penalty_per_incorrect = 2) :
  (total_marks + penalty_per_incorrect * (total_sums - correct_sums)) / correct_sums = 3 := by
sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3052_305275


namespace NUMINAMATH_CALUDE_Q_space_diagonals_l3052_305286

-- Define the structure of our polyhedron
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

-- Define our specific polyhedron Q
def Q : Polyhedron := {
  vertices := 30,
  edges := 72,
  faces := 38,
  triangular_faces := 20,
  quadrilateral_faces := 18
}

-- Function to calculate the number of space diagonals
def space_diagonals (p : Polyhedron) : ℕ :=
  let total_pairs := p.vertices.choose 2
  let face_diagonals := 2 * p.quadrilateral_faces
  total_pairs - p.edges - face_diagonals

-- Theorem statement
theorem Q_space_diagonals : space_diagonals Q = 327 := by
  sorry


end NUMINAMATH_CALUDE_Q_space_diagonals_l3052_305286


namespace NUMINAMATH_CALUDE_work_multiple_proof_l3052_305222

/-- Represents the time taken to complete a job given the number of workers and the fraction of the job to be completed -/
def time_to_complete (num_workers : ℕ) (job_fraction : ℚ) (base_time : ℕ) : ℚ :=
  (job_fraction * base_time) / num_workers

theorem work_multiple_proof (base_workers : ℕ) (h : base_workers > 0) :
  time_to_complete base_workers 1 12 = 12 →
  time_to_complete (2 * base_workers) (1/2) 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_work_multiple_proof_l3052_305222


namespace NUMINAMATH_CALUDE_fidos_yard_area_fraction_l3052_305256

theorem fidos_yard_area_fraction :
  ∀ (s : ℝ), s > 0 →
  (π * s^2) / (4 * s^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_fidos_yard_area_fraction_l3052_305256


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3052_305272

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 ∧ 
               y^2 + (k^2 - 4)*y + k - 1 = 0 ∧ 
               x = -y) → 
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3052_305272


namespace NUMINAMATH_CALUDE_jerry_added_two_figures_l3052_305254

/-- Represents the number of action figures Jerry added to the shelf. -/
def added_figures : ℕ := sorry

/-- The initial number of books on the shelf. -/
def initial_books : ℕ := 7

/-- The initial number of action figures on the shelf. -/
def initial_figures : ℕ := 3

/-- The difference between the number of books and action figures after adding. -/
def book_figure_difference : ℕ := 2

theorem jerry_added_two_figures : 
  added_figures = 2 ∧ 
  initial_books = (initial_figures + added_figures) + book_figure_difference :=
sorry

end NUMINAMATH_CALUDE_jerry_added_two_figures_l3052_305254


namespace NUMINAMATH_CALUDE_f_max_at_zero_l3052_305209

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := x^4 - 2*x^2 - 5

def f_deriv (x : ℝ) : ℝ := 4*x^3 - 4*x

-- State the theorem
theorem f_max_at_zero :
  (f 0 = -5) →
  (∀ x : ℝ, f_deriv x = 4*x^3 - 4*x) →
  ∀ x : ℝ, f x ≤ f 0 :=
by sorry

end NUMINAMATH_CALUDE_f_max_at_zero_l3052_305209


namespace NUMINAMATH_CALUDE_power_of_same_base_power_of_different_base_l3052_305212

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem 1: Condition for representing a^n as (a^p)^q
theorem power_of_same_base (a n : ℕ) (h : n > 1) :
  (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ a^n = (a^p)^q) ↔ ¬(isPrime n) :=
sorry

-- Theorem 2: Condition for representing a^n as b^m with a different base
theorem power_of_different_base (a n : ℕ) (h : n > 0) :
  (∃ b m : ℕ, b ≠ a ∧ m > 0 ∧ a^n = b^m) ↔
  (∃ k : ℕ, k > 0 ∧ ∃ b : ℕ, b ≠ a ∧ a^n = (b^k)^(n/k)) :=
sorry

end NUMINAMATH_CALUDE_power_of_same_base_power_of_different_base_l3052_305212


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3052_305261

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) = 140 * n → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3052_305261


namespace NUMINAMATH_CALUDE_gathering_gift_equation_l3052_305268

/-- Represents a gathering where gifts are exchanged -/
structure Gathering where
  attendees : ℕ
  gifts_exchanged : ℕ
  gift_exchange_rule : attendees > 0 → gifts_exchanged = attendees * (attendees - 1)

/-- Theorem: In a gathering where each pair of attendees exchanges a different small gift,
    if the total number of gifts exchanged is 56 and the number of attendees is x,
    then x(x-1) = 56 -/
theorem gathering_gift_equation (g : Gathering) (h1 : g.gifts_exchanged = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gathering_gift_equation_l3052_305268


namespace NUMINAMATH_CALUDE_triangle_sine_product_inequality_l3052_305249

theorem triangle_sine_product_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_product_inequality_l3052_305249


namespace NUMINAMATH_CALUDE_star_calculation_l3052_305257

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x^2 + y) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ 4) = -135 -/
theorem star_calculation : star 2 (star 3 4) = -135 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l3052_305257


namespace NUMINAMATH_CALUDE_unique_number_with_special_properties_l3052_305200

def digit_sum (n : ℕ) : ℕ := sorry

def has_1112_digits (n : ℕ) : Prop := sorry

def has_one_1_rest_9 (n : ℕ) : Prop := sorry

def one_at_890th_position (n : ℕ) : Prop := sorry

theorem unique_number_with_special_properties :
  ∃! n : ℕ, has_1112_digits n ∧
            2000 ∣ digit_sum n ∧
            2000 ∣ digit_sum (n + 1) ∧
            has_one_1_rest_9 n ∧
            one_at_890th_position n := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_special_properties_l3052_305200


namespace NUMINAMATH_CALUDE_samples_are_stratified_l3052_305204

/-- Represents a sample of 10 student numbers -/
structure Sample :=
  (numbers : List Nat)
  (h_size : numbers.length = 10)
  (h_range : ∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 270)

/-- Represents the distribution of students across grades -/
structure SchoolDistribution :=
  (total : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)
  (h_total : total = first_grade + second_grade + third_grade)

/-- Checks if a sample can represent stratified sampling for a given school distribution -/
def is_stratified_sampling (s : Sample) (sd : SchoolDistribution) : Prop :=
  ∃ (n1 n2 n3 : Nat),
    n1 + n2 + n3 = 10 ∧
    n1 ≤ sd.first_grade ∧
    n2 ≤ sd.second_grade ∧
    n3 ≤ sd.third_grade ∧
    (∀ n ∈ s.numbers, 
      (n ≤ sd.first_grade) ∨ 
      (sd.first_grade < n ∧ n ≤ sd.first_grade + sd.second_grade) ∨
      (sd.first_grade + sd.second_grade < n))

def sample1 : Sample := {
  numbers := [7, 34, 61, 88, 115, 142, 169, 196, 223, 250],
  h_size := by rfl,
  h_range := sorry
}

def sample3 : Sample := {
  numbers := [11, 38, 65, 92, 119, 146, 173, 200, 227, 254],
  h_size := by rfl,
  h_range := sorry
}

def school : SchoolDistribution := {
  total := 270,
  first_grade := 108,
  second_grade := 81,
  third_grade := 81,
  h_total := by rfl
}

theorem samples_are_stratified : 
  is_stratified_sampling sample1 school ∧ is_stratified_sampling sample3 school :=
sorry

end NUMINAMATH_CALUDE_samples_are_stratified_l3052_305204


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3052_305206

theorem sqrt_equation_solution : 
  ∃! z : ℝ, Real.sqrt (5 + 4 * z) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3052_305206


namespace NUMINAMATH_CALUDE_bert_profit_is_one_l3052_305218

/-- Calculates the profit from a sale given the sale price, markup, and tax rate. -/
def calculate_profit (sale_price markup tax_rate : ℚ) : ℚ :=
  let purchase_price := sale_price - markup
  let tax := sale_price * tax_rate
  sale_price - purchase_price - tax

/-- Proves that the profit is $1 given the specific conditions of Bert's sale. -/
theorem bert_profit_is_one :
  calculate_profit 90 10 (1/10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_bert_profit_is_one_l3052_305218


namespace NUMINAMATH_CALUDE_factoring_expression_l3052_305220

theorem factoring_expression (x : ℝ) :
  (12 * x^6 + 40 * x^4 - 6) - (2 * x^6 - 6 * x^4 - 6) = 2 * x^4 * (5 * x^2 + 23) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3052_305220


namespace NUMINAMATH_CALUDE_debit_card_advantage_l3052_305270

/-- Represents the benefit of using a credit card for N days -/
def credit_card_benefit (N : ℕ) : ℚ :=
  20 * N + 120

/-- Represents the benefit of using a debit card -/
def debit_card_benefit : ℚ := 240

/-- The maximum number of days for which using the debit card is more advantageous -/
def max_days_debit_advantageous : ℕ := 6

theorem debit_card_advantage :
  ∀ N : ℕ, N ≤ max_days_debit_advantageous ↔ debit_card_benefit ≥ credit_card_benefit N :=
by sorry

#check debit_card_advantage

end NUMINAMATH_CALUDE_debit_card_advantage_l3052_305270


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3052_305223

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  (a 2 + a 11 = 3) → (a 5 + a 8 = 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3052_305223


namespace NUMINAMATH_CALUDE_range_of_a_l3052_305202

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q, 
    this theorem states that the range of the parameter a in q is [0, 1/2]. -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ |4*x - 3| ≤ 1) →
  (∀ x, q x ↔ x^2 - (2*a + 1)*x + a^2 + a ≤ 0) →
  (∀ x, p x → q x) →
  (∃ x, q x ∧ ¬p x) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3052_305202


namespace NUMINAMATH_CALUDE_distinct_cuttings_count_l3052_305293

/-- Represents a square grid --/
def Square (n : ℕ) := Fin n → Fin n → Bool

/-- Represents an L-shaped piece (corner) --/
structure LPiece where
  size : ℕ
  position : Fin 4 × Fin 4

/-- Represents a cutting of a 4x4 square --/
structure Cutting where
  lpieces : Fin 3 → LPiece
  small_square : Fin 4 × Fin 4

/-- Checks if two cuttings are distinct (considering rotations and reflections) --/
def is_distinct (c1 c2 : Cutting) : Bool := sorry

/-- Counts the number of distinct ways to cut a 4x4 square --/
def count_distinct_cuttings : ℕ := sorry

/-- The main theorem stating that there are 64 distinct ways to cut the 4x4 square --/
theorem distinct_cuttings_count : count_distinct_cuttings = 64 := by sorry

end NUMINAMATH_CALUDE_distinct_cuttings_count_l3052_305293


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3052_305227

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (3 - x) + Real.sqrt (x - 2) = 2) ↔ (x = 3/4 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3052_305227


namespace NUMINAMATH_CALUDE_data_groups_is_six_l3052_305226

/-- Given a dataset, calculate the number of groups it should be divided into -/
def calculateGroups (maxValue minValue interval : ℕ) : ℕ :=
  let range := maxValue - minValue
  let preliminaryGroups := (range + interval - 1) / interval
  preliminaryGroups

/-- Theorem stating that for the given conditions, the number of groups is 6 -/
theorem data_groups_is_six :
  calculateGroups 36 15 4 = 6 := by
  sorry

#eval calculateGroups 36 15 4

end NUMINAMATH_CALUDE_data_groups_is_six_l3052_305226


namespace NUMINAMATH_CALUDE_evaluate_expression_l3052_305231

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3052_305231


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_equals_two_l3052_305285

noncomputable def f (x : ℝ) : ℝ := ((x + 2)^2 + Real.sin x) / (x^2 + 4)

theorem sum_of_max_and_min_equals_two :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               (M + m = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_equals_two_l3052_305285


namespace NUMINAMATH_CALUDE_range_of_g_l3052_305245

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := f (x^2 + 1)

theorem range_of_g :
  Set.range g = Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l3052_305245


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3052_305205

theorem largest_digit_sum (a b c z : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (100 * a + 10 * b + c = 1000 / z) →  -- 0.abc = 1/z
  (0 < z ∧ z ≤ 12) →  -- 0 < z ≤ 12
  (∃ (x y w : ℕ), x + y + w ≤ 8 ∧ 
    (100 * x + 10 * y + w = 1000 / z) ∧ 
    (x < 10 ∧ y < 10 ∧ w < 10)) →
  a + b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3052_305205


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l3052_305276

/-- The shortest altitude of a right triangle with legs 9 and 12 is 7.2 -/
theorem shortest_altitude_right_triangle :
  let a : ℝ := 9
  let b : ℝ := 12
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let area : ℝ := (1/2) * a * b
  let h : ℝ := (2 * area) / c
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l3052_305276


namespace NUMINAMATH_CALUDE_pizza_dough_scaling_l3052_305295

/-- Calculates the required milk for a scaled pizza dough recipe -/
theorem pizza_dough_scaling (original_flour original_milk new_flour : ℚ) : 
  original_flour > 0 ∧ 
  original_milk > 0 ∧ 
  new_flour > 0 ∧
  original_flour = 400 ∧
  original_milk = 80 ∧
  new_flour = 1200 →
  (new_flour / original_flour) * original_milk = 240 := by
  sorry

end NUMINAMATH_CALUDE_pizza_dough_scaling_l3052_305295


namespace NUMINAMATH_CALUDE_sin_negative_eleven_sixths_pi_l3052_305265

theorem sin_negative_eleven_sixths_pi : Real.sin (-11/6 * Real.pi) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_eleven_sixths_pi_l3052_305265


namespace NUMINAMATH_CALUDE_square_divisibility_l3052_305279

theorem square_divisibility (k : ℕ) (n : ℕ) : 
  (∃ m : ℕ, k ^ 2 = n * m) →  -- k^2 is divisible by n
  (∀ j : ℕ, j < k → ¬(∃ m : ℕ, j ^ 2 = n * m)) →  -- k is the least possible value
  k = 60 →  -- the least possible value of k is 60
  n = 3600 :=  -- the number that k^2 is divisible by is 3600
by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l3052_305279


namespace NUMINAMATH_CALUDE_correct_average_calculation_l3052_305215

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ incorrect_avg = 21 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n : ℚ) * incorrect_avg + (correct_num - wrong_num) = n * 22 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l3052_305215


namespace NUMINAMATH_CALUDE_vector_magnitude_l3052_305290

/-- Given plane vectors a and b, prove that the magnitude of a + 2b is 5√2 -/
theorem vector_magnitude (a b : ℝ × ℝ) (ha : a = (3, 5)) (hb : b = (-2, 1)) :
  ‖a + 2 • b‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3052_305290


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l3052_305269

/-- Represents a repeating decimal with a three-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_subtraction :
  RepeatingDecimal 8 6 4 - RepeatingDecimal 5 7 9 - RepeatingDecimal 1 3 5 = 50 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l3052_305269


namespace NUMINAMATH_CALUDE_intersection_M_N_l3052_305234

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3052_305234


namespace NUMINAMATH_CALUDE_color_stamps_count_l3052_305291

/-- The number of color stamps sold by the postal service -/
def color_stamps : ℕ := 1102609 - 523776

/-- The total number of stamps sold by the postal service -/
def total_stamps : ℕ := 1102609

/-- The number of black-and-white stamps sold by the postal service -/
def bw_stamps : ℕ := 523776

theorem color_stamps_count : color_stamps = 578833 := by
  sorry

end NUMINAMATH_CALUDE_color_stamps_count_l3052_305291
