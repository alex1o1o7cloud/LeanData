import Mathlib

namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3157_315786

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 3 → ℝ := ![3, -2, 4]
  let v₂ : Fin 3 → ℝ := ![2, -1, 5]
  v₁ - 3 • v₂ = ![(-3 : ℝ), 1, -11] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3157_315786


namespace NUMINAMATH_CALUDE_well_volume_l3157_315733

/-- The volume of a circular cylinder with diameter 2 meters and height 14 meters is 14π cubic meters. -/
theorem well_volume :
  let diameter : ℝ := 2
  let height : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  volume = 14 * π :=
by sorry

end NUMINAMATH_CALUDE_well_volume_l3157_315733


namespace NUMINAMATH_CALUDE_optimal_pool_dimensions_l3157_315789

/-- Represents the dimensions and cost of a rectangular pool -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (bottomCost : ℝ)
  (wallCost : ℝ)

/-- Calculates the total cost of the pool -/
def totalCost (p : Pool) : ℝ :=
  p.bottomCost * p.length * p.width + p.wallCost * 2 * (p.length + p.width) * p.depth

/-- Theorem stating the optimal dimensions and minimum cost of the pool -/
theorem optimal_pool_dimensions :
  ∀ p : Pool,
  p.depth = 2 ∧
  p.length * p.width * p.depth = 18 ∧
  p.bottomCost = 200 ∧
  p.wallCost = 150 →
  ∃ (minCost : ℝ),
    minCost = 7200 ∧
    totalCost p ≥ minCost ∧
    (totalCost p = minCost ↔ p.length = 3 ∧ p.width = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_pool_dimensions_l3157_315789


namespace NUMINAMATH_CALUDE_expected_draws_for_given_balls_l3157_315777

/-- Represents the number of balls of each color in the bag -/
structure BallCount where
  red : ℕ
  yellow : ℕ

/-- Calculates the expected number of balls drawn until two different colors are drawn -/
def expectedDraws (balls : BallCount) : ℚ :=
  sorry

/-- The theorem stating that the expected number of draws is 5/2 for the given ball configuration -/
theorem expected_draws_for_given_balls :
  expectedDraws { red := 3, yellow := 2 } = 5/2 := by sorry

end NUMINAMATH_CALUDE_expected_draws_for_given_balls_l3157_315777


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3157_315771

open Real

theorem trigonometric_equation_solution (z : ℝ) :
  cos z ≠ 0 →
  sin z ≠ 0 →
  (5.38 * (1 / (cos z)^4) = 160/9 - (2 * ((cos (2*z) / sin (2*z)) * (cos z / sin z) + 1)) / (sin z)^2) →
  ∃ k : ℤ, z = (π/6) * (3 * k + 1) ∨ z = (π/6) * (3 * k - 1) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3157_315771


namespace NUMINAMATH_CALUDE_translation_proof_l3157_315797

-- Define the original linear function
def original_function (x : ℝ) : ℝ := 3 * x - 1

-- Define the translation
def translation : ℝ := 3

-- Define the resulting function after translation
def translated_function (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem translation_proof :
  ∀ x : ℝ, translated_function x = original_function x + translation :=
by
  sorry

end NUMINAMATH_CALUDE_translation_proof_l3157_315797


namespace NUMINAMATH_CALUDE_sine_symmetry_sum_l3157_315707

open Real

theorem sine_symmetry_sum (α β : ℝ) :
  0 ≤ α ∧ α < π ∧
  0 ≤ β ∧ β < π ∧
  α ≠ β ∧
  sin (2 * α + π / 3) = 1 / 2 ∧
  sin (2 * β + π / 3) = 1 / 2 →
  α + β = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_sine_symmetry_sum_l3157_315707


namespace NUMINAMATH_CALUDE_sugar_salt_difference_l3157_315775

/-- 
Given a recipe that calls for specific amounts of ingredients and Mary's actions,
prove that the difference between the required cups of sugar and salt is 2.
-/
theorem sugar_salt_difference (sugar_required flour_required salt_required flour_added : ℕ) 
  (h1 : sugar_required = 11)
  (h2 : flour_required = 6)
  (h3 : salt_required = 9)
  (h4 : flour_added = 12) :
  sugar_required - salt_required = 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_salt_difference_l3157_315775


namespace NUMINAMATH_CALUDE_library_fiction_percentage_l3157_315746

/-- Proves that given the conditions of the library problem, the percentage of fiction novels in the original collection is approximately 30.66%. -/
theorem library_fiction_percentage 
  (total_volumes : ℕ)
  (transferred_fraction : ℚ)
  (transferred_fiction_fraction : ℚ)
  (remaining_fiction_percentage : ℚ)
  (h_total : total_volumes = 18360)
  (h_transferred : transferred_fraction = 1/3)
  (h_transferred_fiction : transferred_fiction_fraction = 1/5)
  (h_remaining_fiction : remaining_fiction_percentage = 35.99999999999999/100) :
  ∃ (original_fiction_percentage : ℚ), 
    (original_fiction_percentage ≥ 30.65/100) ∧ 
    (original_fiction_percentage ≤ 30.67/100) := by
  sorry

end NUMINAMATH_CALUDE_library_fiction_percentage_l3157_315746


namespace NUMINAMATH_CALUDE_exists_triangle_101_subdivisions_l3157_315713

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a function to check if a triangle can be subdivided into n congruent triangles
def can_subdivide (t : Triangle) (n : ℕ) : Prop :=
  ∃ (m : ℕ), m^2 + 1 = n

-- Theorem statement
theorem exists_triangle_101_subdivisions :
  ∃ (t : Triangle), can_subdivide t 101 := by
sorry

end NUMINAMATH_CALUDE_exists_triangle_101_subdivisions_l3157_315713


namespace NUMINAMATH_CALUDE_burger_cost_is_110_l3157_315785

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 110

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

theorem burger_cost_is_110 :
  (∃ (s : ℕ), 4 * burger_cost + 3 * s = 440 ∧ 3 * burger_cost + 2 * s = 330) →
  burger_cost = 110 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_110_l3157_315785


namespace NUMINAMATH_CALUDE_residue_problem_l3157_315709

theorem residue_problem : (195 * 13 - 25 * 8 + 5) % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_problem_l3157_315709


namespace NUMINAMATH_CALUDE_incorrect_value_calculation_l3157_315752

theorem incorrect_value_calculation (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 30)
  (h2 : initial_mean = 150)
  (h3 : correct_mean = 151)
  (h4 : correct_value = 165) :
  let initial_sum := n * initial_mean
  let correct_sum := n * correct_mean
  let difference := correct_sum - initial_sum
  initial_sum + correct_value - difference = n * correct_mean := by sorry

end NUMINAMATH_CALUDE_incorrect_value_calculation_l3157_315752


namespace NUMINAMATH_CALUDE_group_frequency_l3157_315708

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) : 
  sample_capacity = 20 → group_frequency = 0.25 → 
  (sample_capacity : ℚ) * group_frequency = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_frequency_l3157_315708


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3157_315769

theorem max_value_sqrt_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3157_315769


namespace NUMINAMATH_CALUDE_min_slope_is_three_l3157_315788

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem stating that the minimum slope of tangents is 3
theorem min_slope_is_three :
  ∃ (x : ℝ), ∀ (y : ℝ), f' x ≤ f' y ∧ f' x = 3 :=
sorry

end NUMINAMATH_CALUDE_min_slope_is_three_l3157_315788


namespace NUMINAMATH_CALUDE_vertex_in_third_quadrant_l3157_315723

/-- Definition of the parabola --/
def parabola (x : ℝ) : ℝ := -2 * (x + 3)^2 - 21

/-- Definition of the vertex of the parabola --/
def vertex : ℝ × ℝ := (-3, parabola (-3))

/-- Definition of the third quadrant --/
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- Theorem: The vertex of the parabola is in the third quadrant --/
theorem vertex_in_third_quadrant : in_third_quadrant vertex := by
  sorry

end NUMINAMATH_CALUDE_vertex_in_third_quadrant_l3157_315723


namespace NUMINAMATH_CALUDE_annual_growth_rate_is_30_percent_l3157_315714

-- Define the initial number of users and the number after 2 years
def initial_users : ℝ := 1000000
def users_after_2_years : ℝ := 1690000

-- Define the time period
def years : ℝ := 2

-- Define the growth rate as a function
def growth_rate (x : ℝ) : Prop :=
  initial_users * (1 + x)^years = users_after_2_years

-- Theorem statement
theorem annual_growth_rate_is_30_percent :
  ∃ (x : ℝ), x > 0 ∧ growth_rate x ∧ x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_is_30_percent_l3157_315714


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3157_315748

theorem book_arrangement_theorem :
  let total_books : ℕ := 8
  let advanced_geometry_copies : ℕ := 5
  let essential_number_theory_copies : ℕ := 3
  total_books = advanced_geometry_copies + essential_number_theory_copies →
  (Nat.choose total_books advanced_geometry_copies) = 56 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3157_315748


namespace NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l3157_315764

/-- Proves that 35 million yuan is equal to 3.5 × 10^7 yuan -/
theorem thirty_five_million_scientific_notation :
  ∀ (yuan : ℝ), 
  (35 * 1000000 : ℝ) * yuan = (3.5 * 10^7 : ℝ) * yuan := by
  sorry

#check thirty_five_million_scientific_notation

end NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l3157_315764


namespace NUMINAMATH_CALUDE_bridge_length_proof_l3157_315722

/-- Given a train with length 160 meters, traveling at 45 km/hr, that crosses a bridge in 30 seconds, prove that the length of the bridge is 215 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 215 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l3157_315722


namespace NUMINAMATH_CALUDE_trapezoid_top_width_l3157_315703

/-- Proves that a trapezoid with given dimensions has a top width of 14 meters -/
theorem trapezoid_top_width :
  ∀ (area bottom_width height top_width : ℝ),
    area = 880 →
    bottom_width = 8 →
    height = 80 →
    area = (1 / 2) * (top_width + bottom_width) * height →
    top_width = 14 :=
by
  sorry

#check trapezoid_top_width

end NUMINAMATH_CALUDE_trapezoid_top_width_l3157_315703


namespace NUMINAMATH_CALUDE_min_xy_value_l3157_315767

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 1)⁻¹ + (y + 1)⁻¹ = (1 : ℝ) / 2) : 
  ∀ z, z = x * y → z ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3157_315767


namespace NUMINAMATH_CALUDE_angle_y_value_l3157_315728

-- Define the angles in the problem
def angle_ACB : ℝ := 45
def angle_ABC : ℝ := 90
def angle_CDE : ℝ := 72

-- Define the theorem
theorem angle_y_value : 
  ∀ (angle_BAC angle_ADE angle_AED angle_DEB : ℝ),
  -- Triangle ABC
  angle_BAC + angle_ACB + angle_ABC = 180 →
  -- Angle ADC is a straight angle
  angle_ADE + angle_CDE = 180 →
  -- Triangle AED
  angle_AED + angle_ADE + angle_BAC = 180 →
  -- Angle AEB is a straight angle
  angle_AED + angle_DEB = 180 →
  -- Conclusion
  angle_DEB = 153 :=
by sorry

end NUMINAMATH_CALUDE_angle_y_value_l3157_315728


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3157_315745

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ x < n → 
    (¬∃ (y : ℕ), 5 * x = y^2) ∨ 
    (¬∃ (z : ℕ), 4 * x = z^3)) ∧
  n = 1600 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3157_315745


namespace NUMINAMATH_CALUDE_relationship_abc_l3157_315795

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem relationship_abc : 
  let a := base_to_decimal [1, 2] 16
  let b := base_to_decimal [2, 5] 7
  let c := base_to_decimal [3, 3] 4
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3157_315795


namespace NUMINAMATH_CALUDE_octagon_cannot_cover_floor_l3157_315720

/-- Calculate the interior angle of a regular polygon with n sides -/
def interiorAngle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- Check if a given angle divides 360° evenly -/
def divides360 (angle : ℚ) : Prop :=
  ∃ k : ℕ, k * angle = 360

/-- Theorem: Among equilateral triangles, squares, hexagons, and octagons,
    only the octagon's interior angle does not divide 360° evenly -/
theorem octagon_cannot_cover_floor :
  divides360 (interiorAngle 3) ∧
  divides360 (interiorAngle 4) ∧
  divides360 (interiorAngle 6) ∧
  ¬divides360 (interiorAngle 8) :=
sorry

end NUMINAMATH_CALUDE_octagon_cannot_cover_floor_l3157_315720


namespace NUMINAMATH_CALUDE_candle_burning_theorem_l3157_315712

/-- Represents the state of a burning candle -/
structure BurningCandle where
  burn_rate : ℝ
  remaining : ℝ

/-- Represents the state of three burning candles -/
structure ThreeCandles where
  candle1 : BurningCandle
  candle2 : BurningCandle
  candle3 : BurningCandle

/-- 
Given three candles burning at constant rates, if 2/5 of the second candle
and 3/7 of the third candle remain when the first candle burns out, then
1/21 of the third candle will remain when the second candle burns out.
-/
theorem candle_burning_theorem (candles : ThreeCandles) 
  (h1 : candles.candle1.burn_rate > 0)
  (h2 : candles.candle2.burn_rate > 0)
  (h3 : candles.candle3.burn_rate > 0)
  (h4 : candles.candle2.remaining = 2/5)
  (h5 : candles.candle3.remaining = 3/7) :
  candles.candle3.remaining - (candles.candle2.remaining / candles.candle2.burn_rate) * candles.candle3.burn_rate = 1/21 := by
  sorry

end NUMINAMATH_CALUDE_candle_burning_theorem_l3157_315712


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l3157_315794

-- Define the function f
def f (x : ℤ) : ℤ := x^3 - 3*x^2 + 5*x + 9

-- State the theorem
theorem function_satisfies_conditions : 
  f 3 = 12 ∧ f 4 = 22 ∧ f 5 = 36 ∧ f 6 = 54 ∧ f 7 = 76 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l3157_315794


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l3157_315743

/-- Given vectors a, b, and c in ℝ³, prove they are not coplanar. -/
theorem vectors_not_coplanar :
  let a : ℝ × ℝ × ℝ := (3, 10, 5)
  let b : ℝ × ℝ × ℝ := (-2, -2, -3)
  let c : ℝ × ℝ × ℝ := (2, 4, 3)
  ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l3157_315743


namespace NUMINAMATH_CALUDE_weather_period_calculation_l3157_315781

/-- Represents the weather conditions for a period of time --/
structure WeatherPeriod where
  continuousRainDays : ℕ
  mixedRainDays : ℕ
  clearNights : ℕ
  clearDays : ℕ

/-- Calculates the total number of days and fully clear days in a weather period --/
def calculateDays (w : WeatherPeriod) : ℕ × ℕ :=
  let totalDays := w.continuousRainDays + w.mixedRainDays + (w.clearNights + w.clearDays) / 2
  let fullyClearDays := totalDays - w.continuousRainDays - w.mixedRainDays
  (totalDays, fullyClearDays)

/-- Theorem stating the result for the given weather conditions --/
theorem weather_period_calculation (w : WeatherPeriod) 
  (h1 : w.continuousRainDays = 1)
  (h2 : w.mixedRainDays = 9)
  (h3 : w.clearNights = 6)
  (h4 : w.clearDays = 7) :
  calculateDays w = (12, 2) := by
  sorry

#eval calculateDays ⟨1, 9, 6, 7⟩

end NUMINAMATH_CALUDE_weather_period_calculation_l3157_315781


namespace NUMINAMATH_CALUDE_orange_pill_cost_l3157_315737

/-- The cost of an orange pill given the conditions of Bob's treatment --/
theorem orange_pill_cost : 
  ∀ (duration : ℕ) (total_cost : ℚ) (blue_pill_cost : ℚ),
  duration = 21 →
  total_cost = 735 →
  blue_pill_cost + 3 + blue_pill_cost = total_cost / duration →
  blue_pill_cost + 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_orange_pill_cost_l3157_315737


namespace NUMINAMATH_CALUDE_min_m_value_l3157_315701

theorem min_m_value (x y m : ℝ) 
  (hx : 2 ≤ x ∧ x ≤ 3) 
  (hy : 3 ≤ y ∧ y ≤ 6) 
  (h : ∀ x y, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x*y + y^2 ≥ 0) : 
  m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l3157_315701


namespace NUMINAMATH_CALUDE_circle_radius_five_l3157_315754

/-- The value of c for which the circle x^2 + 8x + y^2 - 2y + c = 0 has radius 5 -/
theorem circle_radius_five (x y : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x^2 + 8*x + y^2 - 2*y + c = 0 ↔ (x + 4)^2 + (y - 1)^2 = 5^2) →
  (∃ c : ℝ, c = -8) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_five_l3157_315754


namespace NUMINAMATH_CALUDE_tangent_line_at_2_and_through_A_l3157_315761

/-- The function f(x) = x³ - 4x² + 5x - 4 -/
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

theorem tangent_line_at_2_and_through_A :
  /- Tangent line at X=2 -/
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ x - y - 4 = 0 ∧ 
    m = f' 2 ∧ -2 = m*2 + b) ∧ 
  /- Tangent lines through A(2,-2) -/
  (∃ (a : ℝ), 
    (∀ x y, y = -2 ↔ f a = f' a * (x - a) + f a ∧ -2 = f' a * (2 - a) + f a) ∨
    (∀ x y, x - y - 4 = 0 ↔ f a = f' a * (x - a) + f a ∧ -2 = f' a * (2 - a) + f a)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_and_through_A_l3157_315761


namespace NUMINAMATH_CALUDE_solve_cassette_problem_l3157_315718

structure AudioVideoCassettes where
  audioCost : ℝ
  videoCost : ℝ
  firstSetAudioCount : ℝ
  secondSetAudioCount : ℝ

def cassetteProblem (c : AudioVideoCassettes) : Prop :=
  c.videoCost = 300 ∧
  c.firstSetAudioCount * c.audioCost + 4 * c.videoCost = 1350 ∧
  7 * c.audioCost + 3 * c.videoCost = 1110 ∧
  c.secondSetAudioCount = 7

theorem solve_cassette_problem :
  ∃ c : AudioVideoCassettes, cassetteProblem c :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cassette_problem_l3157_315718


namespace NUMINAMATH_CALUDE_lock_probability_l3157_315774

/-- Given a set of keys and a subset that can open a lock, 
    calculate the probability of randomly selecting a key that opens the lock -/
def probability_open_lock (total_keys : ℕ) (opening_keys : ℕ) : ℚ :=
  opening_keys / total_keys

/-- Theorem: The probability of opening a lock with 2 out of 5 keys is 2/5 -/
theorem lock_probability : 
  probability_open_lock 5 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lock_probability_l3157_315774


namespace NUMINAMATH_CALUDE_infinitely_many_special_pairs_l3157_315711

theorem infinitely_many_special_pairs :
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ,
    let (a, b) := f n
    (a : ℤ) > 0 ∧ (b : ℤ) > 0 ∧
    (∃ k : ℤ, (a : ℤ) * b + 1 = k * ((a : ℤ) + b)) ∧
    (∃ m : ℤ, (a : ℤ) * b - 1 = m * ((a : ℤ) - b)) ∧
    (b : ℤ) > 1 ∧
    (a : ℤ) > (b : ℤ) * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_special_pairs_l3157_315711


namespace NUMINAMATH_CALUDE_f_zero_gt_f_one_l3157_315738

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isEvenOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, -a ≤ x ∧ x ≤ a → f x = f (-x)

def isMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)

-- State the theorem
theorem f_zero_gt_f_one
  (h_even : isEvenOn f 5)
  (h_mono : isMonotonicOn f 0 5)
  (h_ineq : f (-3) < f (-1)) :
  f 0 > f 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_gt_f_one_l3157_315738


namespace NUMINAMATH_CALUDE_sin_sum_inverse_trig_functions_l3157_315790

theorem sin_sum_inverse_trig_functions :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2) + Real.arccos (3/5)) = 41 * Real.sqrt 5 / 125 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_trig_functions_l3157_315790


namespace NUMINAMATH_CALUDE_total_loaves_is_nine_l3157_315766

/-- The number of bags of bread -/
def num_bags : ℕ := 3

/-- The number of loaves in each bag -/
def loaves_per_bag : ℕ := 3

/-- The total number of loaves of bread -/
def total_loaves : ℕ := num_bags * loaves_per_bag

theorem total_loaves_is_nine : total_loaves = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_is_nine_l3157_315766


namespace NUMINAMATH_CALUDE_sam_distance_sam_drove_220_miles_l3157_315793

/-- Calculates the total distance driven by Sam given Marguerite's speed and Sam's driving conditions. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_initial_time : ℝ) (sam_increased_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let marguerite_speed := marguerite_distance / marguerite_time
  let sam_initial_distance := marguerite_speed * sam_initial_time
  let sam_increased_speed := marguerite_speed * (1 + speed_increase)
  let sam_increased_distance := sam_increased_speed * sam_increased_time
  sam_initial_distance + sam_increased_distance

/-- Proves that Sam drove 220 miles given the problem conditions. -/
theorem sam_drove_220_miles : sam_distance 150 3 2 2 0.2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_sam_drove_220_miles_l3157_315793


namespace NUMINAMATH_CALUDE_exist_distinct_prime_divisors_l3157_315730

theorem exist_distinct_prime_divisors (k n : ℕ+) (h : k > n!) :
  ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧
                     (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
                     (∀ i : Fin n, (p i) ∣ (k + i.val + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exist_distinct_prime_divisors_l3157_315730


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l3157_315773

/-- Profit as a percentage of revenue in the previous year, given:
  1. In 1999, revenues fell by 30% compared to the previous year.
  2. In 1999, profits were 14% of revenues.
  3. Profits in 1999 were 98% of the profits in the previous year. -/
theorem profit_percentage_previous_year (R : ℝ) (P : ℝ) 
  (h1 : 0.7 * R = R - 0.3 * R)  -- Revenue fell by 30%
  (h2 : 0.14 * (0.7 * R) = 0.098 * R)  -- Profits were 14% of revenues in 1999
  (h3 : 0.98 * P = 0.098 * R)  -- Profits in 1999 were 98% of previous year
  : P / R = 0.1 := by
  sorry

#check profit_percentage_previous_year

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l3157_315773


namespace NUMINAMATH_CALUDE_largest_fraction_sum_l3157_315705

theorem largest_fraction_sum (a b c d : ℤ) (ha : a = 3) (hb : b = 4) (hc : c = 6) (hd : d = 7) :
  (max ((a : ℚ) / b) ((a : ℚ) / c) + max ((b : ℚ) / a) ((b : ℚ) / c) + 
   max ((c : ℚ) / a) ((c : ℚ) / b) + max ((d : ℚ) / a) ((d : ℚ) / b)) ≤ 23 / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_l3157_315705


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3157_315702

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (2,k),
    if 2a + b is perpendicular to a, then k = -6. -/
theorem perpendicular_vectors_k_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, k)
  (2 • a + b) • a = 0 → k = -6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3157_315702


namespace NUMINAMATH_CALUDE_hay_consumption_time_l3157_315799

/-- The number of weeks it takes for a group of animals to eat a given amount of hay -/
def time_to_eat_hay (goat_rate sheep_rate cow_rate : ℚ) (num_goats num_sheep num_cows : ℕ) (total_hay : ℚ) : ℚ :=
  total_hay / (goat_rate * num_goats + sheep_rate * num_sheep + cow_rate * num_cows)

/-- Theorem: Given the rates of hay consumption and number of animals, it takes 16 weeks to eat 30 cartloads of hay -/
theorem hay_consumption_time :
  let goat_rate : ℚ := 1 / 6
  let sheep_rate : ℚ := 1 / 8
  let cow_rate : ℚ := 1 / 3
  let num_goats : ℕ := 5
  let num_sheep : ℕ := 3
  let num_cows : ℕ := 2
  let total_hay : ℚ := 30
  time_to_eat_hay goat_rate sheep_rate cow_rate num_goats num_sheep num_cows total_hay = 16 := by
  sorry


end NUMINAMATH_CALUDE_hay_consumption_time_l3157_315799


namespace NUMINAMATH_CALUDE_inequality_proof_l3157_315787

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) :
  x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3157_315787


namespace NUMINAMATH_CALUDE_equation_solutions_l3157_315751

theorem equation_solutions :
  (∃ y₁ y₂ : ℝ, y₁ = 3 + 2 * Real.sqrt 2 ∧ y₂ = 3 - 2 * Real.sqrt 2 ∧
    ∀ y : ℝ, y^2 - 6*y + 1 = 0 ↔ (y = y₁ ∨ y = y₂)) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = 12 ∧
    ∀ x : ℝ, 2*(x-4)^2 = x^2 - 16 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3157_315751


namespace NUMINAMATH_CALUDE_square_side_length_l3157_315758

theorem square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3157_315758


namespace NUMINAMATH_CALUDE_candies_left_after_event_l3157_315757

/-- Calculates the number of candies left after a carousel event --/
theorem candies_left_after_event (
  num_clowns : ℕ
  ) (num_children : ℕ
  ) (initial_supply : ℕ
  ) (candies_per_clown : ℕ
  ) (candies_per_child : ℕ
  ) (candies_as_prizes : ℕ
  ) (h1 : num_clowns = 4
  ) (h2 : num_children = 30
  ) (h3 : initial_supply = 1200
  ) (h4 : candies_per_clown = 10
  ) (h5 : candies_per_child = 15
  ) (h6 : candies_as_prizes = 100
  ) : initial_supply - (num_clowns * candies_per_clown + num_children * candies_per_child + candies_as_prizes) = 610 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_after_event_l3157_315757


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l3157_315726

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_term_of_specific_arithmetic_sequence :
  let a₁ := 3
  let a₂ := 7
  let a₃ := 11
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 5 = 19 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l3157_315726


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3157_315798

theorem simplify_trig_expression :
  let θ : Real := 160 * π / 180  -- Convert 160° to radians
  (θ > π / 2) ∧ (θ < π) →  -- 160° is in the second quadrant
  1 / Real.sqrt (1 + Real.tan θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3157_315798


namespace NUMINAMATH_CALUDE_min_rowers_theorem_l3157_315779

/-- Represents a lyamzik with a weight --/
structure Lyamzik where
  weight : Nat

/-- Represents the boat used for crossing --/
structure Boat where
  maxWeight : Nat

/-- Represents the river crossing scenario --/
structure RiverCrossing where
  lyamziks : List Lyamzik
  boat : Boat
  maxRowsPerLyamzik : Nat

/-- The minimum number of lyamziks required to row --/
def minRowersRequired (rc : RiverCrossing) : Nat :=
  12

theorem min_rowers_theorem (rc : RiverCrossing) 
  (h1 : rc.lyamziks.length = 28)
  (h2 : (rc.lyamziks.filter (fun l => l.weight = 2)).length = 7)
  (h3 : (rc.lyamziks.filter (fun l => l.weight = 3)).length = 7)
  (h4 : (rc.lyamziks.filter (fun l => l.weight = 4)).length = 7)
  (h5 : (rc.lyamziks.filter (fun l => l.weight = 5)).length = 7)
  (h6 : rc.boat.maxWeight = 10)
  (h7 : rc.maxRowsPerLyamzik = 2) :
  minRowersRequired rc ≥ 12 := by
  sorry

#check min_rowers_theorem

end NUMINAMATH_CALUDE_min_rowers_theorem_l3157_315779


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l3157_315727

/-- Represents a sampling task with a population size and sample size -/
structure SamplingTask where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a stratified population with different group sizes -/
structure StratifiedPopulation where
  group_sizes : List ℕ

/-- Enumeration of sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Determines the most appropriate sampling method for a given task -/
def most_appropriate_sampling_method (task : SamplingTask) (stratified_info : Option StratifiedPopulation) : SamplingMethod :=
  sorry

/-- The three sampling tasks from the problem -/
def yogurt_task : SamplingTask := { population_size := 10, sample_size := 3 }
def attendees_task : SamplingTask := { population_size := 1280, sample_size := 32 }
def staff_task : SamplingTask := { population_size := 160, sample_size := 20 }

/-- The stratified population information for the staff task -/
def staff_stratified : StratifiedPopulation := { group_sizes := [120, 16, 24] }

theorem appropriate_sampling_methods :
  most_appropriate_sampling_method yogurt_task none = SamplingMethod.SimpleRandom ∧
  most_appropriate_sampling_method attendees_task none = SamplingMethod.Systematic ∧
  most_appropriate_sampling_method staff_task (some staff_stratified) = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l3157_315727


namespace NUMINAMATH_CALUDE_unit_conversions_l3157_315739

-- Define conversion rates
def kgToGrams : ℚ → ℚ := (· * 1000)
def meterToDecimeter : ℚ → ℚ := (· * 10)

-- Theorem statement
theorem unit_conversions :
  (kgToGrams 4 = 4000) ∧
  (meterToDecimeter 3 - 2 = 28) ∧
  (meterToDecimeter 8 = 80) ∧
  ((1600 : ℚ) - 600 = kgToGrams 1) :=
by sorry

end NUMINAMATH_CALUDE_unit_conversions_l3157_315739


namespace NUMINAMATH_CALUDE_best_of_three_win_probability_l3157_315778

/-- The probability of winning a single game -/
def p : ℚ := 3 / 5

/-- The probability of winning the overall competition in a best-of-three format -/
def win_probability : ℚ :=
  p^2 + 2 * p^2 * (1 - p)

theorem best_of_three_win_probability :
  win_probability = 81 / 125 := by
  sorry

end NUMINAMATH_CALUDE_best_of_three_win_probability_l3157_315778


namespace NUMINAMATH_CALUDE_vector_collinear_opposite_direction_l3157_315791

/-- Two vectors in ℝ² -/
def Vector2D : Type := ℝ × ℝ

/-- Check if two vectors are collinear -/
def collinear (v w : Vector2D) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- Check if two vectors have opposite directions -/
def opposite_directions (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k < 0 ∧ v = (k * w.1, k * w.2)

/-- The main theorem -/
theorem vector_collinear_opposite_direction (m : ℝ) :
  let a : Vector2D := (m, 1)
  let b : Vector2D := (1, m)
  collinear a b → opposite_directions a b → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinear_opposite_direction_l3157_315791


namespace NUMINAMATH_CALUDE_least_n_with_j_geq_10_remainder_M_mod_100_l3157_315704

/-- Sum of digits in base 6 representation -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 10 representation -/
def j (n : ℕ) : ℕ := sorry

/-- The least value of n such that j(n) ≥ 10 -/
def M : ℕ := sorry

theorem least_n_with_j_geq_10 : M = 14 := by sorry

theorem remainder_M_mod_100 : M % 100 = 14 := by sorry

end NUMINAMATH_CALUDE_least_n_with_j_geq_10_remainder_M_mod_100_l3157_315704


namespace NUMINAMATH_CALUDE_relationship_abc_l3157_315765

theorem relationship_abc : 
  2022^0 > 8^2022 * (-0.125)^2023 ∧ 8^2022 * (-0.125)^2023 > 2021 * 2023 - 2022^2 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3157_315765


namespace NUMINAMATH_CALUDE_martinez_family_height_l3157_315732

def chiquitaHeight : ℝ := 5

def mrMartinezHeight : ℝ := chiquitaHeight + 2

def mrsMartinezHeight : ℝ := chiquitaHeight - 1

def sonHeight : ℝ := chiquitaHeight + 3

def combinedFamilyHeight : ℝ := chiquitaHeight + mrMartinezHeight + mrsMartinezHeight + sonHeight

theorem martinez_family_height : combinedFamilyHeight = 24 := by
  sorry

end NUMINAMATH_CALUDE_martinez_family_height_l3157_315732


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3157_315717

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 1) : a / b = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3157_315717


namespace NUMINAMATH_CALUDE_median_salary_is_28000_l3157_315706

/-- Represents a position in the company with its title, number of employees, and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company -/
def companyPositions : List Position := [
  { title := "CEO", count := 1, salary := 150000 },
  { title := "Senior Vice-President", count := 4, salary := 105000 },
  { title := "Manager", count := 15, salary := 80000 },
  { title := "Team Leader", count := 8, salary := 60000 },
  { title := "Office Assistant", count := 39, salary := 28000 }
]

/-- The total number of employees in the company -/
def totalEmployees : Nat := 67

/-- Calculates the median salary of the company -/
def medianSalary (positions : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary of the company is $28,000 -/
theorem median_salary_is_28000 :
  medianSalary companyPositions totalEmployees = 28000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_28000_l3157_315706


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l3157_315742

theorem probability_at_least_one_girl (total_students : Nat) (boys : Nat) (girls : Nat) 
  (selected : Nat) (h1 : total_students = boys + girls) (h2 : total_students = 5) 
  (h3 : boys = 3) (h4 : girls = 2) (h5 : selected = 3) : 
  (Nat.choose total_students selected - Nat.choose boys selected) / 
  Nat.choose total_students selected = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l3157_315742


namespace NUMINAMATH_CALUDE_special_polyhedron_properties_l3157_315710

structure Polyhedron where
  convex : Bool
  flat_faces : Bool
  symmetry_planes : Nat
  vertices : Nat
  edges_per_vertex : Nat
  vertex_types : List (Nat × List Nat)

def special_polyhedron : Polyhedron :=
{
  convex := true,
  flat_faces := true,
  symmetry_planes := 2,
  vertices := 8,
  edges_per_vertex := 3,
  vertex_types := [
    (2, [1, 1, 1]),
    (4, [1, 1, 2]),
    (2, [2, 2, 3])
  ]
}

theorem special_polyhedron_properties (K : Polyhedron) 
  (h : K = special_polyhedron) : 
  ∃ (surface_area volume : ℝ), 
    surface_area = 13.86 ∧ 
    volume = 2.946 :=
sorry

end NUMINAMATH_CALUDE_special_polyhedron_properties_l3157_315710


namespace NUMINAMATH_CALUDE_trains_crossing_time_l3157_315782

/-- Time for trains to cross when moving in the same direction -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 200)
  (h2 : length2 = 150)
  (h3 : speed1 = 40)
  (h4 : speed2 = 46) :
  (length1 + length2) / ((speed2 - speed1) * (5/18)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l3157_315782


namespace NUMINAMATH_CALUDE_min_value_of_function_l3157_315784

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  ∃ (min_y : ℝ), min_y = 3 + 2 * Real.sqrt 3 ∧
  ∀ y : ℝ, y = (x^2 + x + 1) / (x - 1) → y ≥ min_y :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3157_315784


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3157_315756

/-- Proves that the ratio of Rahul's present age to Deepak's present age is 4:3 -/
theorem rahul_deepak_age_ratio :
  let rahul_future_age : ℕ := 26
  let years_to_future : ℕ := 6
  let deepak_present_age : ℕ := 15
  let rahul_present_age : ℕ := rahul_future_age - years_to_future
  (rahul_present_age : ℚ) / deepak_present_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3157_315756


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3157_315740

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, a * c^2 ≤ b * c^2 :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3157_315740


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l3157_315750

/-- Given Melissa's game scoring information, calculate her points per game without bonus -/
theorem melissa_points_per_game (bonus_per_game : ℕ) (total_points : ℕ) (num_games : ℕ) 
  (h1 : bonus_per_game = 82)
  (h2 : total_points = 15089)
  (h3 : num_games = 79) :
  (total_points - bonus_per_game * num_games) / num_games = 109 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l3157_315750


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3157_315772

theorem billion_to_scientific_notation :
  let billion : ℝ := 10^8
  let original_number : ℝ := 4947.66 * billion
  original_number = 4.94766 * 10^11 := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3157_315772


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3157_315729

theorem infinite_series_sum : 
  (∑' n : ℕ, 1 / (n * (n + 3))) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3157_315729


namespace NUMINAMATH_CALUDE_combined_eighth_grade_percentage_l3157_315796

/-- Represents the percentage of 8th grade students in a school -/
structure School :=
  (total_students : ℕ)
  (eighth_grade_percentage : ℚ)

/-- Calculates the total number of 8th grade students in both schools -/
def total_eighth_graders (oakwood pinecrest : School) : ℚ :=
  (oakwood.total_students : ℚ) * oakwood.eighth_grade_percentage / 100 +
  (pinecrest.total_students : ℚ) * pinecrest.eighth_grade_percentage / 100

/-- Calculates the total number of students in both schools -/
def total_students (oakwood pinecrest : School) : ℕ :=
  oakwood.total_students + pinecrest.total_students

/-- Theorem stating that the percentage of 8th graders in both schools combined is 57% -/
theorem combined_eighth_grade_percentage 
  (oakwood : School) 
  (pinecrest : School)
  (h1 : oakwood.total_students = 150)
  (h2 : pinecrest.total_students = 250)
  (h3 : oakwood.eighth_grade_percentage = 60)
  (h4 : pinecrest.eighth_grade_percentage = 55) :
  (total_eighth_graders oakwood pinecrest) / (total_students oakwood pinecrest : ℚ) * 100 = 57 :=
sorry

end NUMINAMATH_CALUDE_combined_eighth_grade_percentage_l3157_315796


namespace NUMINAMATH_CALUDE_remaining_segments_theorem_l3157_315734

/-- Represents the spiral pattern described in the problem -/
def spiral_pattern (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2) + n + 1

/-- The total length of the spiral in centimeters -/
def total_length : ℕ := 400

/-- The number of segments already drawn -/
def segments_drawn : ℕ := 7

/-- Calculates the total number of segments in the spiral -/
def total_segments (n : ℕ) : ℕ := 2 * n + 1

theorem remaining_segments_theorem :
  ∃ n : ℕ, 
    spiral_pattern n = total_length ∧ 
    total_segments n - segments_drawn = 32 :=
sorry

end NUMINAMATH_CALUDE_remaining_segments_theorem_l3157_315734


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3157_315744

def satisfies_inequalities (x : ℤ) : Prop :=
  (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1)

def non_negative_integer_solutions : Set ℤ :=
  {x : ℤ | x ≥ 0 ∧ satisfies_inequalities x}

theorem inequality_system_solution :
  non_negative_integer_solutions = {0, 1, 2, 3} :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3157_315744


namespace NUMINAMATH_CALUDE_laura_weekly_mileage_l3157_315735

-- Define the distances
def school_round_trip : ℕ := 20
def supermarket_extra_distance : ℕ := 10

-- Define the number of trips
def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly mileage
def total_weekly_mileage : ℕ :=
  (school_round_trip * school_trips_per_week) +
  ((school_round_trip / 2 + supermarket_extra_distance) * 2 * supermarket_trips_per_week)

-- Theorem to prove
theorem laura_weekly_mileage :
  total_weekly_mileage = 180 := by
  sorry

end NUMINAMATH_CALUDE_laura_weekly_mileage_l3157_315735


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3157_315776

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  (a > 6 ∨ a < -3) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3157_315776


namespace NUMINAMATH_CALUDE_rubber_bands_total_l3157_315715

theorem rubber_bands_total (harper_bands : ℕ) (brother_difference : ℕ) : 
  harper_bands = 15 → 
  brother_difference = 6 → 
  harper_bands + (harper_bands - brother_difference) = 24 := by
sorry

end NUMINAMATH_CALUDE_rubber_bands_total_l3157_315715


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3157_315741

/-- Represents the side lengths of the squares in the rectangle --/
structure SquareSides where
  a : ℕ
  b : ℕ

/-- Calculates the perimeter of the rectangle given the square sides --/
def rectanglePerimeter (s : SquareSides) : ℕ :=
  2 * ((2 * s.a + 3 * s.b) + (3 * s.a + 4 * s.b))

/-- The theorem stating the smallest possible perimeter --/
theorem smallest_perimeter :
  ∃ (s : SquareSides), 
    (5 * s.a + 2 * s.b = 20 * s.a - 3 * s.b) ∧
    (∀ (t : SquareSides), rectanglePerimeter s ≤ rectanglePerimeter t) ∧
    rectanglePerimeter s = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3157_315741


namespace NUMINAMATH_CALUDE_goldbach_126_max_diff_l3157_315759

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem goldbach_126_max_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p ≠ q ∧ 
    p + q = 126 ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r ≠ s → r + s = 126 → 
      (max r s - min r s) ≤ (max p q - min p q) ∧
    (max p q - min p q) = 100 :=
sorry

end NUMINAMATH_CALUDE_goldbach_126_max_diff_l3157_315759


namespace NUMINAMATH_CALUDE_temp_at_six_km_l3157_315700

/-- The temperature drop per kilometer of altitude increase -/
def temp_drop_per_km : ℝ := 5

/-- The temperature at ground level in Celsius -/
def ground_temp : ℝ := 25

/-- The height in kilometers at which we want to calculate the temperature -/
def target_height : ℝ := 6

/-- Calculates the temperature at a given height -/
def temp_at_height (h : ℝ) : ℝ := ground_temp - temp_drop_per_km * h

/-- Theorem stating that the temperature at 6 km height is -5°C -/
theorem temp_at_six_km : temp_at_height target_height = -5 := by sorry

end NUMINAMATH_CALUDE_temp_at_six_km_l3157_315700


namespace NUMINAMATH_CALUDE_total_slices_equals_twelve_l3157_315725

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := 7

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- The total number of slices of pie served today -/
def total_slices : ℕ := lunch_slices + dinner_slices

theorem total_slices_equals_twelve : total_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_equals_twelve_l3157_315725


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocals_l3157_315736

theorem min_sum_of_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → (2 / a + 2 / b = 1) → a + b ≥ 8 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocals_l3157_315736


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_epsilon_squarish_l3157_315719

/-- A positive integer is ε-squarish if it's the product of two integers a and b
    where 1 < a < b < (1 + ε)a -/
def IsEpsilonSquarish (ε : ℝ) (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a * b ∧ 1 < a ∧ a < b ∧ b < (1 + ε) * a

/-- There exist infinitely many positive integers n such that
    n², n² - 1, n² - 2, n² - 3, n² - 4, and n² - 5 are all ε-squarish -/
theorem infinitely_many_consecutive_epsilon_squarish (ε : ℝ) (hε : ε > 0) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧
    IsEpsilonSquarish ε (n^2) ∧
    IsEpsilonSquarish ε (n^2 - 1) ∧
    IsEpsilonSquarish ε (n^2 - 2) ∧
    IsEpsilonSquarish ε (n^2 - 3) ∧
    IsEpsilonSquarish ε (n^2 - 4) ∧
    IsEpsilonSquarish ε (n^2 - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_epsilon_squarish_l3157_315719


namespace NUMINAMATH_CALUDE_hockey_games_played_l3157_315724

theorem hockey_games_played (layla_goals : ℕ) (kristin_goals_difference : ℕ) (average_goals : ℕ) 
  (h1 : layla_goals = 104)
  (h2 : kristin_goals_difference = 24)
  (h3 : average_goals = 92)
  (h4 : layla_goals - kristin_goals_difference = average_goals * 2) :
  2 = (layla_goals + (layla_goals - kristin_goals_difference)) / average_goals := by
  sorry

end NUMINAMATH_CALUDE_hockey_games_played_l3157_315724


namespace NUMINAMATH_CALUDE_half_red_probability_l3157_315768

def num_balls : ℕ := 8
def num_red : ℕ := 4

theorem half_red_probability :
  let p_red : ℚ := 1 / 2
  let p_event : ℚ := (num_balls.choose num_red : ℚ) * p_red ^ num_balls
  p_event = 35 / 128 := by sorry

end NUMINAMATH_CALUDE_half_red_probability_l3157_315768


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3157_315747

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3157_315747


namespace NUMINAMATH_CALUDE_impossible_coverage_l3157_315749

/-- Represents a rectangular paper strip -/
structure PaperStrip where
  width : ℕ
  length : ℕ

/-- Represents a cube -/
structure Cube where
  sideLength : ℕ

/-- Represents the configuration of paper strips on cube faces -/
def CubeConfiguration := Cube → List PaperStrip

/-- Checks if a configuration covers exactly three faces sharing a vertex -/
def coversThreeFaces (config : CubeConfiguration) (cube : Cube) : Prop :=
  sorry

/-- Checks if strips in a configuration overlap -/
def hasOverlap (config : CubeConfiguration) : Prop :=
  sorry

/-- Checks if a configuration leaves any gaps -/
def hasGaps (config : CubeConfiguration) (cube : Cube) : Prop :=
  sorry

/-- Main theorem: It's impossible to cover three faces of a 4x4x4 cube with 16 1x3 strips -/
theorem impossible_coverage : 
  ∀ (config : CubeConfiguration),
    let cube := Cube.mk 4
    let strips := List.replicate 16 (PaperStrip.mk 1 3)
    (coversThreeFaces config cube) → 
    (¬ hasOverlap config) → 
    (¬ hasGaps config cube) → 
    False :=
  sorry

end NUMINAMATH_CALUDE_impossible_coverage_l3157_315749


namespace NUMINAMATH_CALUDE_problem_solution_l3157_315762

-- Define the function f
def f (x : ℝ) : ℝ := |2 * |x| - 1|

-- Define the solution set A
def A : Set ℝ := {x | f x ≤ 1}

-- Theorem statement
theorem problem_solution :
  (A = {x : ℝ | -1 ≤ x ∧ x ≤ 1}) ∧
  (∀ m n : ℝ, m ∈ A → n ∈ A → |m + n| ≤ m * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3157_315762


namespace NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_l3157_315792

/-- A circle passing through three given points -/
def CircleThroughThreePoints (p1 p2 p3 : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | x^2 + y^2 + D*x + E*y + F = 0}
  where
    D : ℝ := -8
    E : ℝ := 6
    F : ℝ := 0

/-- The circle passes through the given points -/
theorem circle_passes_through_points :
  let C := CircleThroughThreePoints (0, 0) (1, 1) (4, 2)
  (0, 0) ∈ C ∧ (1, 1) ∈ C ∧ (4, 2) ∈ C := by
  sorry

/-- The equation of the circle is x^2 + y^2 - 8x + 6y = 0 -/
theorem circle_equation (x y : ℝ) :
  let C := CircleThroughThreePoints (0, 0) (1, 1) (4, 2)
  (x, y) ∈ C ↔ x^2 + y^2 - 8*x + 6*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_l3157_315792


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l3157_315783

/-- Proves that for a parabola y^2 = 2px, the intersection point of two tangent lines
    has a y-coordinate equal to the average of the y-coordinates of the tangent points. -/
theorem parabola_tangent_intersection
  (p : ℝ) (x₁ y₁ x₂ y₂ x y : ℝ)
  (h_parabola₁ : y₁^2 = 2*p*x₁)
  (h_parabola₂ : y₂^2 = 2*p*x₂)
  (h_tangent₁ : y*y₁ = p*(x + x₁))
  (h_tangent₂ : y*y₂ = p*(x + x₂))
  (h_distinct : y₁ ≠ y₂) :
  y = (y₁ + y₂) / 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l3157_315783


namespace NUMINAMATH_CALUDE_evaluate_expression_l3157_315755

theorem evaluate_expression : 
  3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7 = 6^5 + 3^7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3157_315755


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3157_315770

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 16) : 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3157_315770


namespace NUMINAMATH_CALUDE_sum_of_terms_l3157_315716

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  roots_property : a 2 + a 16 = 6 ∧ a 2 * a 16 = -1

/-- The sum of specific terms in the arithmetic sequence equals 15 -/
theorem sum_of_terms (seq : ArithmeticSequence) : 
  seq.a 5 + seq.a 6 + seq.a 9 + seq.a 12 + seq.a 13 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_l3157_315716


namespace NUMINAMATH_CALUDE_snow_removal_volume_l3157_315753

/-- The volume of snow to be removed from a rectangular driveway -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Proof that the volume of snow to be removed is 67.5 cubic feet -/
theorem snow_removal_volume :
  let length : ℝ := 30
  let width : ℝ := 3
  let depth : ℝ := 0.75
  snow_volume length width depth = 67.5 := by
sorry

end NUMINAMATH_CALUDE_snow_removal_volume_l3157_315753


namespace NUMINAMATH_CALUDE_subtraction_problem_l3157_315721

theorem subtraction_problem (x : ℝ) (h : 40 / x = 5) : 20 - x = 12 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3157_315721


namespace NUMINAMATH_CALUDE_not_power_of_prime_l3157_315731

theorem not_power_of_prime (n : ℕ+) (q : ℕ) (h_prime : Nat.Prime q) :
  ¬∃ k : ℕ, (n : ℝ)^q + ((n - 1 : ℝ) / 2)^2 = (q : ℝ)^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_prime_l3157_315731


namespace NUMINAMATH_CALUDE_two_digit_numbers_from_123_l3157_315763

def digits : Set Nat := {1, 2, 3}

def twoDigitNumber (tens units : Nat) : Nat := 10 * tens + units

def validTwoDigitNumber (n : Nat) : Prop :=
  ∃ (tens units : Nat), tens ∈ digits ∧ units ∈ digits ∧ n = twoDigitNumber tens units

theorem two_digit_numbers_from_123 :
  {n : Nat | validTwoDigitNumber n} = {11, 12, 13, 21, 22, 23, 31, 32, 33} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_from_123_l3157_315763


namespace NUMINAMATH_CALUDE_alex_sandwich_production_l3157_315780

/-- Given that Alex can prepare 18 sandwiches using 3 loaves of bread,
    this theorem proves that he can make 60 sandwiches with 10 loaves of bread. -/
theorem alex_sandwich_production (sandwiches_per_three_loaves : ℕ) 
    (h1 : sandwiches_per_three_loaves = 18) : 
    (sandwiches_per_three_loaves / 3) * 10 = 60 := by
  sorry

#check alex_sandwich_production

end NUMINAMATH_CALUDE_alex_sandwich_production_l3157_315780


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l3157_315760

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let reciprocal (q : ℚ) : ℚ := 1 / q
  reciprocal x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l3157_315760
