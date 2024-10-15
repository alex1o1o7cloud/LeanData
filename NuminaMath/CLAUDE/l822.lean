import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_point_slope_l822_82214

/-- The point-slope form of a line with given slope and point. -/
def point_slope_form (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- Theorem: The point-slope form of a line with slope 2 passing through (2, -3) is y + 3 = 2(x - 2). -/
theorem line_equation_point_slope : 
  let k : ℝ := 2
  let x₀ : ℝ := 2
  let y₀ : ℝ := -3
  ∀ x y : ℝ, point_slope_form k x₀ y₀ x y ↔ y + 3 = 2 * (x - 2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_point_slope_l822_82214


namespace NUMINAMATH_CALUDE_employee_count_l822_82261

theorem employee_count (initial_avg : ℝ) (new_avg : ℝ) (manager_salary : ℝ) : 
  initial_avg = 1500 →
  new_avg = 1900 →
  manager_salary = 11500 →
  ∃ n : ℕ, (n : ℝ) * initial_avg + manager_salary = new_avg * ((n : ℝ) + 1) ∧ n = 24 := by
sorry

end NUMINAMATH_CALUDE_employee_count_l822_82261


namespace NUMINAMATH_CALUDE_cosine_equality_l822_82202

theorem cosine_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (745 * π / 180) →
  n = 25 ∨ n = -25 := by
sorry

end NUMINAMATH_CALUDE_cosine_equality_l822_82202


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l822_82207

theorem restaurant_bill_proof (n : ℕ) (extra : ℝ) (total : ℝ) : 
  n = 10 → 
  extra = 3 → 
  (n - 1) * (total / n + extra) = total → 
  total = 270 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l822_82207


namespace NUMINAMATH_CALUDE_balloon_count_l822_82234

theorem balloon_count (my_balloons : ℕ) (friend_balloons : ℕ) 
  (h1 : friend_balloons = 5)
  (h2 : my_balloons - friend_balloons = 2) : 
  my_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_balloon_count_l822_82234


namespace NUMINAMATH_CALUDE_possible_m_values_l822_82212

theorem possible_m_values (x m a b : ℤ) : 
  (∀ x, x^2 + m*x - 14 = (x + a) * (x + b)) → 
  (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
sorry

end NUMINAMATH_CALUDE_possible_m_values_l822_82212


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l822_82203

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | x^2 + (2*m+1)*x + m^2 + m > 0} = {x : ℝ | x > -m ∨ x < -m-1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l822_82203


namespace NUMINAMATH_CALUDE_cheetah_catch_fox_l822_82262

/-- Represents the cheetah's speed in meters per second -/
def cheetah_speed : ℝ := 4

/-- Represents the fox's speed in meters per second -/
def fox_speed : ℝ := 3

/-- Represents the initial distance between the cheetah and the fox in meters -/
def initial_distance : ℝ := 30

/-- Theorem stating that the cheetah will catch the fox after running 120 meters -/
theorem cheetah_catch_fox : 
  cheetah_speed * (initial_distance / (cheetah_speed - fox_speed)) = 120 :=
sorry

end NUMINAMATH_CALUDE_cheetah_catch_fox_l822_82262


namespace NUMINAMATH_CALUDE_max_min_distance_on_sphere_l822_82220

/-- A point on a unit sphere represented by its coordinates -/
def SpherePoint := ℝ × ℝ × ℝ

/-- The distance between two points on a unit sphere -/
def sphereDistance (p q : SpherePoint) : ℝ := sorry

/-- Checks if a point is on the unit sphere -/
def isOnUnitSphere (p : SpherePoint) : Prop := sorry

/-- Represents a configuration of five points on a unit sphere -/
def Configuration := Fin 5 → SpherePoint

/-- The minimum pairwise distance in a configuration -/
def minDistance (c : Configuration) : ℝ := sorry

/-- Checks if a configuration has two points at opposite poles and three equidistant points on the equator -/
def isOptimalConfiguration (c : Configuration) : Prop := sorry

theorem max_min_distance_on_sphere :
  ∀ c : Configuration, (∀ i, isOnUnitSphere (c i)) →
  minDistance c ≤ Real.sqrt 2 ∧
  (minDistance c = Real.sqrt 2 ↔ isOptimalConfiguration c) := by sorry

end NUMINAMATH_CALUDE_max_min_distance_on_sphere_l822_82220


namespace NUMINAMATH_CALUDE_m_minus_n_values_l822_82225

theorem m_minus_n_values (m n : ℤ) 
  (h1 : |m| = 3)
  (h2 : |n| = 5)
  (h3 : m + n > 0) :
  m - n = -2 ∨ m - n = -8 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l822_82225


namespace NUMINAMATH_CALUDE_max_intersections_15_10_l822_82254

/-- The maximum number of intersection points for segments connecting points on x and y axes -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 15 x-axis points and 10 y-axis points -/
theorem max_intersections_15_10 :
  max_intersections 15 10 = 4725 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_15_10_l822_82254


namespace NUMINAMATH_CALUDE_elementary_school_coats_l822_82236

theorem elementary_school_coats 
  (total_coats : ℕ) 
  (high_school_coats : ℕ) 
  (middle_school_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : high_school_coats = 6922)
  (h3 : middle_school_coats = 1825) :
  total_coats - (high_school_coats + middle_school_coats) = 690 := by
  sorry

end NUMINAMATH_CALUDE_elementary_school_coats_l822_82236


namespace NUMINAMATH_CALUDE_derivative_f_at_3pi_4_l822_82272

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x + 1

theorem derivative_f_at_3pi_4 :
  deriv f (3 * Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_3pi_4_l822_82272


namespace NUMINAMATH_CALUDE_matthew_friends_count_l822_82221

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 4

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 32

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 98

/-- The number of crackers each person ate -/
def crackers_per_person : ℕ := 8

theorem matthew_friends_count :
  (initial_crackers / crackers_per_person = num_friends) ∧
  (initial_crackers % crackers_per_person = 0) :=
sorry

end NUMINAMATH_CALUDE_matthew_friends_count_l822_82221


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l822_82213

theorem unique_solution_xyz (x y z : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : x * y - z^2 = 4) :
  x = 2 ∧ y = 2 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l822_82213


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l822_82251

/-- The equation (m-4)x^|m-2| + 2x - 5 = 0 is quadratic if and only if m = 0 -/
theorem quadratic_equation_condition (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 4) * x^(|m - 2|) + 2*x - 5 = a*x^2 + b*x + c) ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l822_82251


namespace NUMINAMATH_CALUDE_segment_length_implies_product_l822_82240

/-- Given that the length of the segment between the points (3a, 2a-5) and (5, 0) is 3√10 units,
    prove that the product of all possible values of a is -40/13. -/
theorem segment_length_implies_product (a : ℝ) : 
  (((3*a - 5)^2 + (2*a - 5)^2) = 90) → 
  (∃ b : ℝ, (a = b ∨ a = -8/13) ∧ a * b = -40/13) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_implies_product_l822_82240


namespace NUMINAMATH_CALUDE_polynomials_equal_sum_of_squares_is_954_l822_82239

/-- The original polynomial expression -/
def original_polynomial (x : ℝ) : ℝ := 5 * (x^3 - 3*x^2 + 4) - 8 * (2*x^4 - x^3 + x)

/-- The fully simplified polynomial -/
def simplified_polynomial (x : ℝ) : ℝ := -16*x^4 - 3*x^3 - 15*x^2 + 8*x + 20

/-- Theorem stating that the original and simplified polynomials are equal -/
theorem polynomials_equal : ∀ x : ℝ, original_polynomial x = simplified_polynomial x := by sorry

/-- The sum of squares of coefficients of the simplified polynomial -/
def sum_of_squares_of_coefficients : ℕ := 16^2 + 3^2 + 15^2 + 8^2 + 20^2

/-- Theorem stating that the sum of squares of coefficients is 954 -/
theorem sum_of_squares_is_954 : sum_of_squares_of_coefficients = 954 := by sorry

end NUMINAMATH_CALUDE_polynomials_equal_sum_of_squares_is_954_l822_82239


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l822_82218

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_3402 :
  largest_perfect_square_factor 3402 = 9 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l822_82218


namespace NUMINAMATH_CALUDE_work_together_time_l822_82205

/-- 
Calculates the time taken to complete a job when two people work together, 
given their individual completion times.
-/
def time_together (time_david time_john : ℚ) : ℚ :=
  1 / (1 / time_david + 1 / time_john)

/-- 
Theorem: If David completes a job in 5 days and John completes the same job in 9 days,
then the time taken to complete the job when they work together is 45/14 days.
-/
theorem work_together_time : time_together 5 9 = 45 / 14 := by
  sorry

end NUMINAMATH_CALUDE_work_together_time_l822_82205


namespace NUMINAMATH_CALUDE_even_decreasing_implies_inequality_l822_82224

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_implies_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_decr : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_implies_inequality_l822_82224


namespace NUMINAMATH_CALUDE_range_of_a_l822_82226

theorem range_of_a : ∀ a : ℝ, 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) →
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l822_82226


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l822_82278

theorem quadratic_roots_relation (m n p : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0)
  (h : ∃ (r₁ r₂ : ℝ), (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧ 
                      (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = n)) :
  n / p = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l822_82278


namespace NUMINAMATH_CALUDE_initial_files_count_l822_82255

theorem initial_files_count (organized_morning : ℕ) (to_organize_afternoon : ℕ) (missing : ℕ) :
  organized_morning = to_organize_afternoon ∧
  to_organize_afternoon = missing ∧
  to_organize_afternoon = 15 →
  2 * organized_morning + to_organize_afternoon + missing = 60 :=
by sorry

end NUMINAMATH_CALUDE_initial_files_count_l822_82255


namespace NUMINAMATH_CALUDE_factorization_proof_l822_82291

theorem factorization_proof (a : ℝ) :
  74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l822_82291


namespace NUMINAMATH_CALUDE_net_income_calculation_l822_82283

def calculate_net_income (spring_lawn : ℝ) (spring_garden : ℝ) (summer_lawn : ℝ) (summer_garden : ℝ)
  (fall_lawn : ℝ) (fall_garden : ℝ) (winter_snow : ℝ)
  (spring_lawn_supplies : ℝ) (spring_garden_supplies : ℝ)
  (summer_lawn_supplies : ℝ) (summer_garden_supplies : ℝ)
  (fall_lawn_supplies : ℝ) (fall_garden_supplies : ℝ)
  (winter_snow_supplies : ℝ)
  (advertising_percent : ℝ) (maintenance_percent : ℝ) : ℝ :=
  let total_earnings := spring_lawn + spring_garden + summer_lawn + summer_garden +
                        fall_lawn + fall_garden + winter_snow
  let total_supplies := spring_lawn_supplies + spring_garden_supplies +
                        summer_lawn_supplies + summer_garden_supplies +
                        fall_lawn_supplies + fall_garden_supplies +
                        winter_snow_supplies
  let total_gardening := spring_garden + summer_garden + fall_garden
  let total_lawn_mowing := spring_lawn + summer_lawn + fall_lawn
  let advertising_expenses := advertising_percent * total_gardening
  let maintenance_expenses := maintenance_percent * total_lawn_mowing
  total_earnings - total_supplies - advertising_expenses - maintenance_expenses

theorem net_income_calculation :
  calculate_net_income 200 150 600 450 300 350 100
                       80 50 150 100 75 75 25
                       0.15 0.10 = 1342.50 := by
  sorry

end NUMINAMATH_CALUDE_net_income_calculation_l822_82283


namespace NUMINAMATH_CALUDE_smallest_arithmetic_mean_of_nine_consecutive_naturals_l822_82200

theorem smallest_arithmetic_mean_of_nine_consecutive_naturals (n : ℕ) : 
  (∀ k : ℕ, k ∈ Finset.range 9 → (n + k) > 0) →
  (((List.range 9).map (λ k => n + k)).prod) % 1111 = 0 →
  (((List.range 9).map (λ k => n + k)).sum / 9 : ℚ) ≥ 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_arithmetic_mean_of_nine_consecutive_naturals_l822_82200


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l822_82290

theorem smallest_number_with_given_remainders : ∃ (n : ℕ), n = 838 ∧ 
  (∃ (a : ℕ), 0 ≤ a ∧ a ≤ 19 ∧ 
    n % 20 = a ∧ 
    n % 21 = a + 1 ∧ 
    n % 22 = 2) ∧ 
  (∀ (m : ℕ), m < n → 
    ¬(∃ (b : ℕ), 0 ≤ b ∧ b ≤ 19 ∧ 
      m % 20 = b ∧ 
      m % 21 = b + 1 ∧ 
      m % 22 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l822_82290


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l822_82298

-- Define a real-valued function
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Define what it means for a function to have extreme values
def has_extreme_value (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for a function to have real roots
def has_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = 0

-- Statement: f'(x) = 0 having real roots is necessary but not sufficient for f(x) having extreme values
theorem necessary_not_sufficient :
  (has_extreme_value f → has_real_roots f') ∧
  ¬(has_real_roots f' → has_extreme_value f) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l822_82298


namespace NUMINAMATH_CALUDE_tea_price_correct_l822_82247

/-- Represents the prices and quantities of tea in two purchases -/
structure TeaPurchases where
  first_quantity_A : ℕ
  first_quantity_B : ℕ
  first_total_cost : ℕ
  second_quantity_A : ℕ
  second_quantity_B : ℕ
  second_total_cost : ℕ
  price_increase : ℚ

/-- The solution to the tea pricing problem -/
def tea_price_solution (tp : TeaPurchases) : ℚ × ℚ :=
  (100, 200)

/-- Theorem stating that the given solution is correct for the specified tea purchases -/
theorem tea_price_correct (tp : TeaPurchases) 
  (h1 : tp.first_quantity_A = 30)
  (h2 : tp.first_quantity_B = 20)
  (h3 : tp.first_total_cost = 7000)
  (h4 : tp.second_quantity_A = 20)
  (h5 : tp.second_quantity_B = 15)
  (h6 : tp.second_total_cost = 6000)
  (h7 : tp.price_increase = 1/5) : 
  let (price_A, price_B) := tea_price_solution tp
  (tp.first_quantity_A : ℚ) * price_A + (tp.first_quantity_B : ℚ) * price_B = tp.first_total_cost ∧
  (tp.second_quantity_A : ℚ) * price_A * (1 + tp.price_increase) + 
  (tp.second_quantity_B : ℚ) * price_B * (1 + tp.price_increase) = tp.second_total_cost :=
by
  sorry

#check tea_price_correct

end NUMINAMATH_CALUDE_tea_price_correct_l822_82247


namespace NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l822_82206

theorem point_inside_circle_implies_a_range (a : ℝ) :
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l822_82206


namespace NUMINAMATH_CALUDE_barbell_cost_increase_l822_82285

theorem barbell_cost_increase (old_cost new_cost : ℝ) (h1 : old_cost = 250) (h2 : new_cost = 325) :
  (new_cost - old_cost) / old_cost * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_barbell_cost_increase_l822_82285


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l822_82210

def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 10
def balls_removed : ℕ := 2

theorem probability_of_white_ball :
  let total_balls := initial_white_balls + initial_black_balls
  let remaining_balls := total_balls - balls_removed
  ∃ (p : ℚ), p = 37/98 ∧ 
    (∀ (w b : ℕ), w + b = remaining_balls → 
      (w : ℚ) / (w + b : ℚ) ≤ p) ∧
    (∃ (w b : ℕ), w + b = remaining_balls ∧ 
      (w : ℚ) / (w + b : ℚ) = p) :=
by sorry


end NUMINAMATH_CALUDE_probability_of_white_ball_l822_82210


namespace NUMINAMATH_CALUDE_hexagon_area_proof_l822_82215

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Calculates the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Checks if lines are parallel -/
def areParallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if y-coordinates are distinct elements of a set -/
def distinctYCoordinates (h : Hexagon) (s : Set ℝ) : Prop := sorry

theorem hexagon_area_proof (h : Hexagon) :
  h.A = ⟨0, 0⟩ →
  h.B = ⟨2 * Real.sqrt 3, 3⟩ →
  h.F = ⟨-7 / 2 * Real.sqrt 3, 5⟩ →
  angle h.F h.A h.B = 150 * π / 180 →
  areParallel h.A h.B h.D h.E →
  areParallel h.B h.C h.E h.F →
  areParallel h.C h.D h.F h.A →
  isEquilateral h →
  distinctYCoordinates h {0, 1, 3, 5, 7, 9} →
  hexagonArea h = 77 / 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_proof_l822_82215


namespace NUMINAMATH_CALUDE_band_second_set_songs_l822_82229

/-- Proves the number of songs played in the second set given the band's repertoire and performance details -/
theorem band_second_set_songs 
  (total_songs : ℕ) 
  (first_set : ℕ) 
  (encore : ℕ) 
  (avg_third_fourth : ℕ) 
  (h1 : total_songs = 30)
  (h2 : first_set = 5)
  (h3 : encore = 2)
  (h4 : avg_third_fourth = 8) :
  ∃ (second_set : ℕ), 
    second_set = 7 ∧ 
    (total_songs - first_set - second_set - encore) / 2 = avg_third_fourth :=
by sorry

end NUMINAMATH_CALUDE_band_second_set_songs_l822_82229


namespace NUMINAMATH_CALUDE_shortest_assembly_time_is_13_l822_82244

/-- Represents the time taken for each step in the assembly process -/
structure AssemblyTimes where
  ac : ℕ -- Time from A to C
  cd : ℕ -- Time from C to D
  be : ℕ -- Time from B to E
  ed : ℕ -- Time from E to D
  df : ℕ -- Time from D to F

/-- Calculates the shortest assembly time given the times for each step -/
def shortestAssemblyTime (times : AssemblyTimes) : ℕ :=
  max (times.ac + times.cd) (times.be + times.ed + times.df)

/-- Theorem stating that for the given assembly times, the shortest assembly time is 13 hours -/
theorem shortest_assembly_time_is_13 :
  let times : AssemblyTimes := {
    ac := 3,
    cd := 4,
    be := 3,
    ed := 4,
    df := 2
  }
  shortestAssemblyTime times = 13 := by
  sorry

end NUMINAMATH_CALUDE_shortest_assembly_time_is_13_l822_82244


namespace NUMINAMATH_CALUDE_may_savings_l822_82219

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l822_82219


namespace NUMINAMATH_CALUDE_digital_earth_prospects_l822_82279

/-- Represents the prospects of digital Earth applications -/
structure DigitalEarthProspects where
  spatialLab : Bool  -- Provides a digital spatial laboratory
  decisionMaking : Bool  -- Government decision-making can fully rely on it
  urbanManagement : Bool  -- Provides a basis for urban management
  predictable : Bool  -- The development is predictable

/-- The correct prospects of digital Earth applications -/
def correctProspects : DigitalEarthProspects :=
  { spatialLab := true
    decisionMaking := false
    urbanManagement := true
    predictable := false }

/-- Theorem stating the correct prospects of digital Earth applications -/
theorem digital_earth_prospects :
  (correctProspects.spatialLab = true) ∧
  (correctProspects.urbanManagement = true) ∧
  (correctProspects.decisionMaking = false) ∧
  (correctProspects.predictable = false) := by
  sorry


end NUMINAMATH_CALUDE_digital_earth_prospects_l822_82279


namespace NUMINAMATH_CALUDE_square_field_area_l822_82275

/-- The area of a square field with side length 25 meters is 625 square meters. -/
theorem square_field_area : 
  let side_length : ℝ := 25
  let area : ℝ := side_length * side_length
  area = 625 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l822_82275


namespace NUMINAMATH_CALUDE_lollipop_sugar_calculation_l822_82287

def chocolate_bars : ℕ := 14
def sugar_per_bar : ℕ := 10
def total_sugar : ℕ := 177

def sugar_in_lollipop : ℕ := total_sugar - (chocolate_bars * sugar_per_bar)

theorem lollipop_sugar_calculation :
  sugar_in_lollipop = 37 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_sugar_calculation_l822_82287


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l822_82260

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem min_sum_of_primes (a b c d n : ℕ) : 
  (∃ k : ℕ, a * 1000 + b * 100 + c * 10 + d = 3 * 3 * 11 * (n + 49)) →
  is_prime a → is_prime b → is_prime c → is_prime d →
  (∀ a' b' c' d' n' : ℕ, 
    (∃ k' : ℕ, a' * 1000 + b' * 100 + c' * 10 + d' = 3 * 3 * 11 * (n' + 49)) →
    is_prime a' → is_prime b' → is_prime c' → is_prime d' →
    a + b + c + d ≤ a' + b' + c' + d') →
  a + b + c + d = 70 := 
sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l822_82260


namespace NUMINAMATH_CALUDE_problem_solution_l822_82267

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 2 - t) 
  (h2 : y = 4 * t + 7) 
  (h3 : x = -3) : 
  y = 27 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l822_82267


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l822_82294

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l822_82294


namespace NUMINAMATH_CALUDE_house_store_transaction_loss_l822_82209

theorem house_store_transaction_loss (house_price store_price : ℝ) : 
  house_price * (1 - 0.2) = 12000 →
  store_price * (1 + 0.2) = 12000 →
  house_price + store_price - 2 * 12000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_house_store_transaction_loss_l822_82209


namespace NUMINAMATH_CALUDE_remainders_inequality_l822_82264

theorem remainders_inequality (X Y M A B s t u : ℕ) : 
  X > Y →
  X % M = A →
  Y % M = B →
  X = Y + 8 →
  (X^2) % M = s →
  (Y^2) % M = t →
  ((A*B)^2) % M = u →
  (s ≠ t ∧ t ≠ u ∧ s ≠ u) :=
by sorry

end NUMINAMATH_CALUDE_remainders_inequality_l822_82264


namespace NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_three_equals_two_l822_82227

theorem sqrt_twelve_div_sqrt_three_equals_two : Real.sqrt 12 / Real.sqrt 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_three_equals_two_l822_82227


namespace NUMINAMATH_CALUDE_production_plan_equation_l822_82252

/-- Represents a factory's production plan -/
structure ProductionPlan where
  original_days : ℕ
  original_parts_per_day : ℕ
  new_days : ℕ
  additional_parts_per_day : ℕ
  extra_parts : ℕ

/-- The equation holds for the given production plan -/
def equation_holds (plan : ProductionPlan) : Prop :=
  plan.original_days * plan.original_parts_per_day = 
  plan.new_days * (plan.original_parts_per_day + plan.additional_parts_per_day) - plan.extra_parts

theorem production_plan_equation (plan : ProductionPlan) 
  (h1 : plan.original_days = 20)
  (h2 : plan.new_days = 15)
  (h3 : plan.additional_parts_per_day = 4)
  (h4 : plan.extra_parts = 10) :
  equation_holds plan := by
  sorry

#check production_plan_equation

end NUMINAMATH_CALUDE_production_plan_equation_l822_82252


namespace NUMINAMATH_CALUDE_nested_root_simplification_l822_82217

theorem nested_root_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x * (x^3)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l822_82217


namespace NUMINAMATH_CALUDE_trig_identity_l822_82286

open Real

theorem trig_identity : 
  sin (150 * π / 180) * cos ((-420) * π / 180) + 
  cos ((-690) * π / 180) * sin (600 * π / 180) + 
  tan (405 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l822_82286


namespace NUMINAMATH_CALUDE_equation_solution_l822_82246

theorem equation_solution : 
  ∃ x : ℝ, (64 + 5 * 12 / (x / 3) = 65) ∧ (x = 180) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l822_82246


namespace NUMINAMATH_CALUDE_rectangle_max_area_l822_82256

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) →  -- perimeter is 40 units
  (l * w ≤ 100) -- area is at most 100 square units
:= by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l822_82256


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l822_82222

/-- The distance between two points on a quadratic curve. -/
theorem distance_on_quadratic_curve (m n p x₁ x₂ : ℝ) :
  let y₁ := m * x₁^2 + n * x₁ + p
  let y₂ := m * x₂^2 + n * x₂ + p
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (x₂ - x₁)^2 * (1 + m^2 * (x₂ + x₁)^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l822_82222


namespace NUMINAMATH_CALUDE_smallest_number_of_groups_l822_82273

theorem smallest_number_of_groups (total_campers : ℕ) (max_group_size : ℕ) : 
  total_campers = 36 → max_group_size = 12 → 
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) ∧
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) → num_groups = 3 :=
by sorry


end NUMINAMATH_CALUDE_smallest_number_of_groups_l822_82273


namespace NUMINAMATH_CALUDE_transistors_in_2010_l822_82201

/-- Moore's law doubling period in years -/
def doubling_period : ℕ := 2

/-- Initial year for the calculation -/
def initial_year : ℕ := 1995

/-- Final year for the calculation -/
def final_year : ℕ := 2010

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 2000000

/-- Calculate the number of transistors based on Moore's law -/
def moores_law_transistors (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / doubling_period)

/-- Theorem stating the number of transistors in 2010 according to Moore's law -/
theorem transistors_in_2010 :
  moores_law_transistors (final_year - initial_year) = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l822_82201


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l822_82289

def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0) (h2 : r ≠ 1) :
  (geometric_sequence a r 1 + geometric_sequence a r 2 + 
   geometric_sequence a r 3 + geometric_sequence a r 4 = 240) →
  (geometric_sequence a r 2 + geometric_sequence a r 4 = 180) →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l822_82289


namespace NUMINAMATH_CALUDE_multiplication_fraction_result_l822_82284

theorem multiplication_fraction_result : 12 * (1 / 17) * 34 = 24 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_result_l822_82284


namespace NUMINAMATH_CALUDE_at_least_five_roots_l822_82235

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the period T
variable (T : ℝ)

-- Assumptions
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x + T) = f x)
variable (h_T_pos : T > 0)

-- Theorem statement
theorem at_least_five_roots :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
     x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
     x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
     x₄ ≠ x₅) ∧
    (x₁ ∈ Set.Icc (-T) T ∧
     x₂ ∈ Set.Icc (-T) T ∧
     x₃ ∈ Set.Icc (-T) T ∧
     x₄ ∈ Set.Icc (-T) T ∧
     x₅ ∈ Set.Icc (-T) T) ∧
    (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_five_roots_l822_82235


namespace NUMINAMATH_CALUDE_a_neg_sufficient_a_neg_not_necessary_a_neg_sufficient_not_necessary_l822_82241

/-- Represents a quadratic equation ax^2 + 2x + 1 = 0 -/
structure QuadraticEquation (a : ℝ) where
  eq : ∀ x, a * x^2 + 2 * x + 1 = 0

/-- Predicate for an equation having at least one negative root -/
def has_negative_root {a : ℝ} (eq : QuadraticEquation a) : Prop :=
  ∃ x, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

/-- Statement that 'a < 0' is a sufficient condition -/
theorem a_neg_sufficient {a : ℝ} (h : a < 0) : 
  ∃ (eq : QuadraticEquation a), has_negative_root eq :=
sorry

/-- Statement that 'a < 0' is not a necessary condition -/
theorem a_neg_not_necessary : 
  ∃ a, ¬(a < 0) ∧ ∃ (eq : QuadraticEquation a), has_negative_root eq :=
sorry

/-- Main theorem stating that 'a < 0' is sufficient but not necessary -/
theorem a_neg_sufficient_not_necessary : 
  (∀ a, a < 0 → ∃ (eq : QuadraticEquation a), has_negative_root eq) ∧
  (∃ a, ¬(a < 0) ∧ ∃ (eq : QuadraticEquation a), has_negative_root eq) :=
sorry

end NUMINAMATH_CALUDE_a_neg_sufficient_a_neg_not_necessary_a_neg_sufficient_not_necessary_l822_82241


namespace NUMINAMATH_CALUDE_line_through_point_l822_82257

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, b * x + (b + 2) * y = b - 1 → x = 3 ∧ y = -5) → b = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_l822_82257


namespace NUMINAMATH_CALUDE_same_solution_equations_l822_82211

theorem same_solution_equations (x b : ℝ) : 
  (2 * x + 7 = 3) ∧ (b * x - 10 = -2) → b = -4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l822_82211


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l822_82263

theorem unique_function_satisfying_condition :
  ∀ f : ℕ → ℕ,
    (f 1 > 0) →
    (∀ m n : ℕ, f (m^2 + 3*n^2) = (f m)^2 + 3*(f n)^2) →
    (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l822_82263


namespace NUMINAMATH_CALUDE_inequality_proof_l822_82280

theorem inequality_proof (x : ℝ) (h : x > 0) : 1/x + 4*x^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l822_82280


namespace NUMINAMATH_CALUDE_percentage_calculation_l822_82233

theorem percentage_calculation (n : ℝ) (h : n = 5600) : 0.15 * (0.30 * (0.50 * n)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l822_82233


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_is_two_sevenths_l822_82237

/-- The probability of two randomly selected diagonals in a nonagon intersecting inside the nonagon -/
def nonagon_diagonal_intersection_probability : ℚ :=
  2 / 7

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of diagonals in a nonagon -/
def nonagon_diagonals : ℕ := nonagon_sides.choose 2 - nonagon_sides

/-- The number of ways to choose two diagonals in a nonagon -/
def diagonal_pairs : ℕ := nonagon_diagonals.choose 2

/-- The number of ways to choose four vertices in a nonagon -/
def four_vertex_selections : ℕ := nonagon_sides.choose 4

theorem nonagon_diagonal_intersection_probability_is_two_sevenths :
  nonagon_diagonal_intersection_probability = four_vertex_selections / diagonal_pairs :=
sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_is_two_sevenths_l822_82237


namespace NUMINAMATH_CALUDE_combined_girls_avg_is_88_l822_82269

/-- Represents a high school with average scores for boys, girls, and combined -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools -/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_boys_avg : ℝ

/-- Calculates the combined average score for girls across two schools -/
def combined_girls_avg (schools : CombinedSchools) : ℝ :=
  sorry

/-- The theorem stating that the combined average score for girls is 88 -/
theorem combined_girls_avg_is_88 (schools : CombinedSchools) 
  (h1 : schools.school1 = { boys_avg := 74, girls_avg := 77, combined_avg := 75 })
  (h2 : schools.school2 = { boys_avg := 83, girls_avg := 94, combined_avg := 90 })
  (h3 : schools.combined_boys_avg = 80) :
  combined_girls_avg schools = 88 := by
  sorry

end NUMINAMATH_CALUDE_combined_girls_avg_is_88_l822_82269


namespace NUMINAMATH_CALUDE_container_emptying_possible_l822_82228

/-- Represents a container with water -/
structure Container where
  water : ℕ

/-- Represents the state of three containers -/
structure ContainerState where
  a : Container
  b : Container
  c : Container

/-- Represents a transfer of water between containers -/
inductive Transfer : ContainerState → ContainerState → Prop where
  | ab (s : ContainerState) : 
      Transfer s ⟨⟨s.a.water + s.b.water⟩, ⟨0⟩, s.c⟩
  | ac (s : ContainerState) : 
      Transfer s ⟨⟨s.a.water + s.c.water⟩, s.b, ⟨0⟩⟩
  | ba (s : ContainerState) : 
      Transfer s ⟨⟨0⟩, ⟨s.a.water + s.b.water⟩, s.c⟩
  | bc (s : ContainerState) : 
      Transfer s ⟨s.a, ⟨s.b.water + s.c.water⟩, ⟨0⟩⟩
  | ca (s : ContainerState) : 
      Transfer s ⟨⟨0⟩, s.b, ⟨s.a.water + s.c.water⟩⟩
  | cb (s : ContainerState) : 
      Transfer s ⟨s.a, ⟨0⟩, ⟨s.b.water + s.c.water⟩⟩

/-- Represents a sequence of transfers -/
def TransferSeq := List (ContainerState → ContainerState)

/-- Applies a sequence of transfers to an initial state -/
def applyTransfers (initial : ContainerState) (seq : TransferSeq) : ContainerState :=
  seq.foldl (fun state transfer => transfer state) initial

/-- Predicate to check if a container is empty -/
def isEmptyContainer (c : Container) : Prop := c.water = 0

/-- Predicate to check if any container in the state is empty -/
def hasEmptyContainer (s : ContainerState) : Prop :=
  isEmptyContainer s.a ∨ isEmptyContainer s.b ∨ isEmptyContainer s.c

/-- The main theorem to prove -/
theorem container_emptying_possible (initial : ContainerState) : 
  ∃ (seq : TransferSeq), hasEmptyContainer (applyTransfers initial seq) := by
  sorry

end NUMINAMATH_CALUDE_container_emptying_possible_l822_82228


namespace NUMINAMATH_CALUDE_cos_equality_with_period_l822_82231

theorem cos_equality_with_period (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → 
  Real.cos (n * π / 180) = Real.cos (845 * π / 180) → 
  n = 125 := by
sorry

end NUMINAMATH_CALUDE_cos_equality_with_period_l822_82231


namespace NUMINAMATH_CALUDE_inequality_proof_l822_82276

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l822_82276


namespace NUMINAMATH_CALUDE_small_planters_needed_l822_82271

/-- Represents the types of seeds --/
inductive SeedType
  | Basil
  | Cilantro
  | Parsley

/-- Represents the types of planters --/
inductive PlanterType
  | Large
  | Medium
  | Small

/-- Represents the planting requirements for each seed type --/
def plantingRequirement (s : SeedType) : Set PlanterType :=
  match s with
  | SeedType.Basil => {PlanterType.Large, PlanterType.Medium}
  | SeedType.Cilantro => {PlanterType.Medium}
  | SeedType.Parsley => {PlanterType.Large, PlanterType.Medium, PlanterType.Small}

/-- The capacity of each planter type --/
def planterCapacity (p : PlanterType) : ℕ :=
  match p with
  | PlanterType.Large => 20
  | PlanterType.Medium => 10
  | PlanterType.Small => 4

/-- The number of each planter type available --/
def planterCount (p : PlanterType) : ℕ :=
  match p with
  | PlanterType.Large => 4
  | PlanterType.Medium => 8
  | PlanterType.Small => 0  -- We're solving for this

/-- The number of seeds for each seed type --/
def seedCount (s : SeedType) : ℕ :=
  match s with
  | SeedType.Basil => 200
  | SeedType.Cilantro => 160
  | SeedType.Parsley => 120

theorem small_planters_needed : 
  ∃ (n : ℕ), 
    n * planterCapacity PlanterType.Small = 
      seedCount SeedType.Parsley + 
      (seedCount SeedType.Cilantro - planterCount PlanterType.Medium * planterCapacity PlanterType.Medium) + 
      (seedCount SeedType.Basil - 
        (planterCount PlanterType.Large * planterCapacity PlanterType.Large + 
         (planterCount PlanterType.Medium - 
          (seedCount SeedType.Cilantro / planterCapacity PlanterType.Medium)) * 
           planterCapacity PlanterType.Medium)) ∧ 
    n = 50 := by
  sorry

end NUMINAMATH_CALUDE_small_planters_needed_l822_82271


namespace NUMINAMATH_CALUDE_imaginary_unit_powers_l822_82232

theorem imaginary_unit_powers (i : ℂ) : i^2 = -1 → i^50 + i^105 = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_powers_l822_82232


namespace NUMINAMATH_CALUDE_number_of_observations_l822_82243

theorem number_of_observations (initial_mean old_value new_value new_mean : ℝ) : 
  initial_mean = 36 →
  old_value = 40 →
  new_value = 25 →
  new_mean = 34.9 →
  ∃ (n : ℕ), (n : ℝ) * initial_mean - old_value + new_value = (n : ℝ) * new_mean ∧ 
              n = 14 :=
by sorry

end NUMINAMATH_CALUDE_number_of_observations_l822_82243


namespace NUMINAMATH_CALUDE_S_inter_T_eq_T_l822_82297

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_inter_T_eq_T_l822_82297


namespace NUMINAMATH_CALUDE_divisibility_condition_l822_82281

theorem divisibility_condition (x y z k : ℤ) :
  (∃ q : ℤ, x^3 + y^3 + z^3 + k*x*y*z = (x + y + z) * q) ↔ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l822_82281


namespace NUMINAMATH_CALUDE_green_triangle_cost_l822_82292

/-- Calculates the cost of greening a right-angled triangle -/
theorem green_triangle_cost 
  (a b c : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a : a = 8) 
  (h_b : b = 15) 
  (h_c : c = 17) 
  (h_cost : cost_per_sqm = 50) : 
  (1/2 * a * b) * cost_per_sqm = 3000 := by
sorry

end NUMINAMATH_CALUDE_green_triangle_cost_l822_82292


namespace NUMINAMATH_CALUDE_bead_mixing_problem_l822_82265

/-- Proves that the total number of boxes is 8 given the conditions of the bead mixing problem. -/
theorem bead_mixing_problem (red_cost yellow_cost mixed_cost : ℚ) 
  (boxes_per_color : ℕ) : 
  red_cost = 13/10 ∧ 
  yellow_cost = 2 ∧ 
  mixed_cost = 43/25 ∧ 
  boxes_per_color = 4 → 
  (red_cost * boxes_per_color + yellow_cost * boxes_per_color) / 
    (2 * boxes_per_color) = mixed_cost ∧
  2 * boxes_per_color = 8 := by
  sorry

end NUMINAMATH_CALUDE_bead_mixing_problem_l822_82265


namespace NUMINAMATH_CALUDE_min_fraction_sum_l822_82250

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_fraction_sum (A B C D : Nat) : 
  A ∈ Digits → B ∈ Digits → C ∈ Digits → D ∈ Digits →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  Nat.Prime B → Nat.Prime D →
  (∀ A' B' C' D' : Nat, 
    A' ∈ Digits → B' ∈ Digits → C' ∈ Digits → D' ∈ Digits →
    A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
    Nat.Prime B' → Nat.Prime D' →
    (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) ≤ (A' : ℚ) / (B' : ℚ) + (C' : ℚ) / (D' : ℚ)) →
  (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l822_82250


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l822_82270

/-- The number of ways to arrange plates around a circular table. -/
def arrange_plates (blue red green orange : ℕ) : ℕ :=
  sorry

/-- The number of valid arrangements of plates. -/
def valid_arrangements : ℕ :=
  arrange_plates 5 3 2 1

/-- Theorem stating the correct number of valid arrangements. -/
theorem valid_arrangements_count : valid_arrangements = 361 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l822_82270


namespace NUMINAMATH_CALUDE_largest_number_l822_82204

def a : ℚ := 883/1000
def b : ℚ := 8839/10000
def c : ℚ := 88/100
def d : ℚ := 839/1000
def e : ℚ := 889/1000

theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l822_82204


namespace NUMINAMATH_CALUDE_unique_n_squared_plus_2n_prime_l822_82230

theorem unique_n_squared_plus_2n_prime :
  ∃! (n : ℕ), n > 0 ∧ Nat.Prime (n^2 + 2*n) :=
sorry

end NUMINAMATH_CALUDE_unique_n_squared_plus_2n_prime_l822_82230


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l822_82238

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

theorem largest_even_digit_multiple_of_5 :
  ∃ (n : ℕ), n = 6880 ∧
  has_only_even_digits n ∧
  n < 8000 ∧
  n % 5 = 0 ∧
  ∀ m : ℕ, has_only_even_digits m → m < 8000 → m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l822_82238


namespace NUMINAMATH_CALUDE_water_flow_problem_l822_82268

/-- The water flow problem -/
theorem water_flow_problem (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : (2 * (30 / x) + 2 * (30 / x) + 4 * (60 / x)) / 2 = 18) -- Total water collected and dumped
  : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_problem_l822_82268


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l822_82258

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the chessboard game -/
structure ChessboardGame :=
  (m : Nat) (n : Nat)

/-- Checks if a position is winning for the current player -/
def isWinningPosition (game : ChessboardGame) (pos : Position) : Prop :=
  pos.x ≠ pos.y

/-- Checks if the first player has a winning strategy -/
def firstPlayerWins (game : ChessboardGame) : Prop :=
  isWinningPosition game ⟨game.m - 1, game.n - 1⟩

/-- The main theorem: The first player wins iff m ≠ n -/
theorem first_player_winning_strategy (game : ChessboardGame) :
  firstPlayerWins game ↔ game.m ≠ game.n :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l822_82258


namespace NUMINAMATH_CALUDE_A_intersect_B_l822_82274

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem A_intersect_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l822_82274


namespace NUMINAMATH_CALUDE_same_solution_equations_l822_82248

theorem same_solution_equations (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 8 = 5 ∧ c * x - 7 = 1) → c = -8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l822_82248


namespace NUMINAMATH_CALUDE_divisibility_theorem_l822_82208

theorem divisibility_theorem (a b c d m : ℤ) 
  (h_odd : Odd m)
  (h_div_sum : m ∣ (a + b + c + d))
  (h_div_sum_squares : m ∣ (a^2 + b^2 + c^2 + d^2)) :
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l822_82208


namespace NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l822_82216

theorem half_plus_seven_equals_seventeen (x : ℝ) : (1/2) * x + 7 = 17 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l822_82216


namespace NUMINAMATH_CALUDE_victory_chain_exists_l822_82288

/-- Represents a chess player in the tournament -/
structure Player :=
  (id : Nat)

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- The chess tournament with 2016 players -/
def Tournament := Fin 2016 → Player

/-- The result of a match between two players -/
def matchResult (p1 p2 : Player) : MatchResult := sorry

/-- Condition: If players A and B tie, then every other player loses to either A or B -/
def tieCondition (t : Tournament) : Prop :=
  ∀ a b : Player, matchResult a b = MatchResult.Draw →
    ∀ c : Player, c ≠ a ∧ c ≠ b →
      matchResult c a = MatchResult.Loss ∨ matchResult c b = MatchResult.Loss

/-- There are at least two draws in the tournament -/
def atLeastTwoDraws (t : Tournament) : Prop :=
  ∃ a b c d : Player, a ≠ b ∧ c ≠ d ∧ matchResult a b = MatchResult.Draw ∧ matchResult c d = MatchResult.Draw

/-- A permutation of players where each player defeats the next -/
def victoryChain (t : Tournament) (p : Fin 2016 → Fin 2016) : Prop :=
  ∀ i : Fin 2015, matchResult (t (p i)) (t (p (i + 1))) = MatchResult.Win

/-- Main theorem: If there are at least two draws and the tie condition holds,
    then there exists a permutation where each player defeats the next -/
theorem victory_chain_exists (t : Tournament)
  (h1 : tieCondition t) (h2 : atLeastTwoDraws t) :
  ∃ p : Fin 2016 → Fin 2016, Function.Bijective p ∧ victoryChain t p := by
  sorry

end NUMINAMATH_CALUDE_victory_chain_exists_l822_82288


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l822_82295

theorem quadratic_form_minimum : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 ∧
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 5 * y₀^2 - 8 * x₀ - 10 * y₀ = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l822_82295


namespace NUMINAMATH_CALUDE_cubic_root_sum_simplification_l822_82282

theorem cubic_root_sum_simplification :
  (((9 : ℝ) / 16 + 25 / 36 + 4 / 9) ^ (1/3 : ℝ)) = (245 : ℝ) ^ (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_simplification_l822_82282


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l822_82253

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 7*x*y - 13*x + 15*y - 37 = 0 ↔ 
    ((x = -2 ∧ y = 11) ∨ (x = -1 ∧ y = 3) ∨ (x = 7 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l822_82253


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l822_82299

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one (m n : ℕ) : x m = y n → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l822_82299


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l822_82277

/-- Triangle ABC with given conditions -/
structure TriangleABC where
  /-- Point B has coordinates (4,4) -/
  B : ℝ × ℝ
  hB : B = (4, 4)
  
  /-- The equation of the angle bisector of ∠A is y = 0 -/
  angle_bisector : ℝ → ℝ
  h_angle_bisector : ∀ x, angle_bisector x = 0
  
  /-- The equation of the altitude from B to AC is x - 2y + 2 = 0 -/
  altitude : ℝ → ℝ
  h_altitude : ∀ x, altitude x = (x + 2) / 2

/-- The coordinates of point C in triangle ABC -/
def point_C (t : TriangleABC) : ℝ × ℝ := (10, -8)

/-- The area of triangle ABC -/
def area (t : TriangleABC) : ℝ := 48

/-- Main theorem: The coordinates of C and the area of triangle ABC are correct -/
theorem triangle_abc_properties (t : TriangleABC) : 
  (point_C t = (10, -8)) ∧ (area t = 48) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l822_82277


namespace NUMINAMATH_CALUDE_inequality_proof_l822_82242

theorem inequality_proof (x a : ℝ) (h : |x - a| < 1) : 
  let f := fun (t : ℝ) => t^2 - 2*t
  |f x - f a| < 2*|a| + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l822_82242


namespace NUMINAMATH_CALUDE_conical_hopper_volume_l822_82266

/-- The volume of a conical hopper with given dimensions -/
theorem conical_hopper_volume :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let height : ℝ := 0.6 * radius
  let volume : ℝ := (1 / 3) * Real.pi * radius^2 * height
  volume = 25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_conical_hopper_volume_l822_82266


namespace NUMINAMATH_CALUDE_distribute_5_2_l822_82245

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_5_2 : distribute 5 2 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_5_2_l822_82245


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l822_82296

theorem parabola_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - 4 * y + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l822_82296


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_55_and_11_l822_82249

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_55_and_11 :
  ∃ (m : ℕ), is_four_digit m ∧
             m % 55 = 0 ∧
             (reverse_digits m) % 55 = 0 ∧
             m % 11 = 0 ∧
             (∀ (n : ℕ), is_four_digit n →
                         n % 55 = 0 →
                         (reverse_digits n) % 55 = 0 →
                         n % 11 = 0 →
                         n ≤ m) ∧
             m = 5445 :=
sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_55_and_11_l822_82249


namespace NUMINAMATH_CALUDE_sum_first_six_multiples_of_twelve_l822_82259

theorem sum_first_six_multiples_of_twelve : 
  (Finset.range 6).sum (fun i => 12 * (i + 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_multiples_of_twelve_l822_82259


namespace NUMINAMATH_CALUDE_workshop_schedule_l822_82293

theorem workshop_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_workshop_schedule_l822_82293


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l822_82223

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2 + x, 9; 4 - x, 5]

theorem matrix_not_invertible (x : ℚ) :
  ¬(IsUnit (matrix x).det) ↔ x = 13/7 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l822_82223
