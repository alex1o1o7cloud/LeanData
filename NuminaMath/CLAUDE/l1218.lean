import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_x_y_l1218_121829

theorem min_sum_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + x*y - 7 = 0) :
  ∃ (m : ℝ), m = 3 ∧ x + y ≥ m ∧ ∀ (z : ℝ), x + y > z → z < m :=
sorry

end NUMINAMATH_CALUDE_min_sum_x_y_l1218_121829


namespace NUMINAMATH_CALUDE_function_maximum_value_l1218_121808

/-- Given a function f(x) = x / (x^2 + a) where a > 0, 
    if its maximum value on [1, +∞) is √3/3, then a = √3 - 1 -/
theorem function_maximum_value (a : ℝ) : 
  a > 0 → 
  (∀ x : ℝ, x ≥ 1 → x / (x^2 + a) ≤ Real.sqrt 3 / 3) →
  (∃ x : ℝ, x ≥ 1 ∧ x / (x^2 + a) = Real.sqrt 3 / 3) →
  a = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_maximum_value_l1218_121808


namespace NUMINAMATH_CALUDE_area_common_triangles_circle_l1218_121854

/-- The area of the region common to two inscribed equilateral triangles and an inscribed circle in a square -/
theorem area_common_triangles_circle (square_side : ℝ) (triangle_side : ℝ) (circle_radius : ℝ) : ℝ :=
  by
  -- Given conditions
  have h1 : square_side = 4 := by sorry
  have h2 : triangle_side = square_side := by sorry
  have h3 : circle_radius = square_side / 2 := by sorry
  
  -- Approximate area calculation
  have triangle_area : ℝ := by sorry
  have circle_area : ℝ := by sorry
  have overlap_per_triangle : ℝ := by sorry
  have total_overlap : ℝ := by sorry
  
  -- Prove the approximate area is 4π
  sorry

#check area_common_triangles_circle

end NUMINAMATH_CALUDE_area_common_triangles_circle_l1218_121854


namespace NUMINAMATH_CALUDE_melanie_turnips_count_l1218_121823

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The additional number of turnips Melanie grew compared to Benny -/
def melanie_extra_turnips : ℕ := 26

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := benny_turnips + melanie_extra_turnips

theorem melanie_turnips_count : melanie_turnips = 139 := by
  sorry

end NUMINAMATH_CALUDE_melanie_turnips_count_l1218_121823


namespace NUMINAMATH_CALUDE_salary_increase_l1218_121810

/-- Regression equation for monthly salary based on labor productivity -/
def salary_equation (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating that an increase of 1000 yuan in labor productivity
    results in an increase of 80 yuan in salary -/
theorem salary_increase (x : ℝ) :
  salary_equation (x + 1) - salary_equation x = 80 := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_salary_increase_l1218_121810


namespace NUMINAMATH_CALUDE_consecutive_composites_l1218_121856

theorem consecutive_composites
  (a t d r : ℕ+)
  (ha : ¬ Nat.Prime a.val)
  (ht : ¬ Nat.Prime t.val)
  (hd : ¬ Nat.Prime d.val)
  (hr : ¬ Nat.Prime r.val) :
  ∃ k : ℕ, ∀ i : ℕ, i < r → ¬ Nat.Prime (a * t ^ (k + i) + d) :=
sorry

end NUMINAMATH_CALUDE_consecutive_composites_l1218_121856


namespace NUMINAMATH_CALUDE_program_schedule_arrangements_l1218_121890

theorem program_schedule_arrangements (n : ℕ) (h : n = 6) : 
  (n + 1).choose 1 * (n + 2).choose 1 = 56 := by
  sorry

end NUMINAMATH_CALUDE_program_schedule_arrangements_l1218_121890


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1218_121892

-- Define the constants from the problem
def bucket_capacity : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_trips_ratio : ℕ := 3
def jill_trips_ratio : ℕ := 2
def jill_total_trips : ℕ := 30

-- Define the theorem
theorem tank_capacity_proof : 
  let jack_total_trips := (jack_trips_ratio * jill_total_trips) / jill_trips_ratio
  let jill_water := jill_total_trips * jill_buckets_per_trip * bucket_capacity
  let jack_water := jack_total_trips * jack_buckets_per_trip * bucket_capacity
  jill_water + jack_water = 600 := by
  sorry


end NUMINAMATH_CALUDE_tank_capacity_proof_l1218_121892


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1218_121882

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ -1) ∧ 
  (∀ x : ℝ, x = -1 → x^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1218_121882


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1218_121827

theorem linear_equation_solution : 
  ∀ x : ℝ, (x + 1) / 3 = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1218_121827


namespace NUMINAMATH_CALUDE_cube_and_square_root_problem_l1218_121880

theorem cube_and_square_root_problem (a b : ℝ) 
  (h1 : (2*b - 2*a)^(1/3 : ℝ) = -2)
  (h2 : (4*a + 3*b)^(1/2 : ℝ) = 3) :
  a = 3 ∧ b = -1 ∧ (5*a - b)^(1/2 : ℝ) = 4 ∨ (5*a - b)^(1/2 : ℝ) = -4 :=
by sorry

end NUMINAMATH_CALUDE_cube_and_square_root_problem_l1218_121880


namespace NUMINAMATH_CALUDE_estimate_greater_than_exact_l1218_121883

theorem estimate_greater_than_exact 
  (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≤ c) :
  (a' / b') - c' > (a / b) - c :=
sorry

end NUMINAMATH_CALUDE_estimate_greater_than_exact_l1218_121883


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l1218_121893

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 204 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a ≥ 0 ∧ 
    fuel_a ≤ tank_capacity ∧
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol ∧
    fuel_a = 66 :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l1218_121893


namespace NUMINAMATH_CALUDE_mosquito_shadow_speed_l1218_121804

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

end NUMINAMATH_CALUDE_mosquito_shadow_speed_l1218_121804


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1218_121886

theorem complex_fraction_equality : 
  let z₁ : ℂ := 1 - I
  let z₂ : ℂ := 1 + I
  z₁ / (z₂ * I) = -2 * I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1218_121886


namespace NUMINAMATH_CALUDE_students_without_a_count_l1218_121897

/-- Represents the number of students in a school course with various grade distributions. -/
structure CourseData where
  total_students : ℕ
  history_as : ℕ
  math_as : ℕ
  both_as : ℕ
  math_only_a : ℕ
  history_only_attendees : ℕ

/-- Calculates the number of students who did not receive an A in either class. -/
def students_without_a (data : CourseData) : ℕ :=
  data.total_students - (data.history_as + data.math_as - data.both_as)

/-- Theorem stating the number of students who did not receive an A in either class. -/
theorem students_without_a_count (data : CourseData) 
  (h1 : data.total_students = 30)
  (h2 : data.history_only_attendees = 1)
  (h3 : data.history_as = 6)
  (h4 : data.math_as = 15)
  (h5 : data.both_as = 3)
  (h6 : data.math_only_a = 1) :
  students_without_a data = 12 := by
  sorry

#eval students_without_a {
  total_students := 30,
  history_as := 6,
  math_as := 15,
  both_as := 3,
  math_only_a := 1,
  history_only_attendees := 1
}

end NUMINAMATH_CALUDE_students_without_a_count_l1218_121897


namespace NUMINAMATH_CALUDE_meaningful_expression_l1218_121807

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 5)) / x) ↔ (x ≥ -5 ∧ x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1218_121807


namespace NUMINAMATH_CALUDE_enclosed_area_calculation_l1218_121859

/-- The area enclosed by a curve consisting of 9 congruent circular arcs, 
    each of length π/2, whose centers are at the vertices of a regular hexagon 
    with side length 3. -/
def enclosed_area (num_arcs : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) : ℝ :=
  sorry

/-- Theorem stating the enclosed area for the specific problem -/
theorem enclosed_area_calculation : 
  enclosed_area 9 (π/2) 3 = (27 * Real.sqrt 3) / 2 + 9 * π / 8 :=
sorry

end NUMINAMATH_CALUDE_enclosed_area_calculation_l1218_121859


namespace NUMINAMATH_CALUDE_f_composition_inequality_l1218_121839

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 + 2*a*x else 2^x + 1

-- State the theorem
theorem f_composition_inequality (a : ℝ) :
  (f a (f a 1) > 3 * a^2) ↔ (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_inequality_l1218_121839


namespace NUMINAMATH_CALUDE_product_ab_l1218_121891

theorem product_ab (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a * b = 2256 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l1218_121891


namespace NUMINAMATH_CALUDE_solution_concentration_l1218_121824

theorem solution_concentration (x y : ℝ) : 
  (0.45 * x = 0.15 * (x + y + 1)) ∧ 
  (0.30 * y = 0.05 * (x + y + 1)) → 
  x = 2/3 ∧ y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_concentration_l1218_121824


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_when_vertices_on_circle_l1218_121809

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) :=
  (h_pos : 0 < b ∧ b < a)

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- Condition that the left focus, right vertex, and upper and lower vertices lie on the same circle -/
def vertices_on_circle (e : Ellipse a b) : Prop := sorry

/-- Theorem stating that if the vertices lie on the same circle, the eccentricity is (√5 - 1)/2 -/
theorem ellipse_eccentricity_when_vertices_on_circle (e : Ellipse a b) :
  vertices_on_circle e → eccentricity e = (Real.sqrt 5 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_when_vertices_on_circle_l1218_121809


namespace NUMINAMATH_CALUDE_new_pressure_is_two_l1218_121803

/-- Represents the pressure-volume relationship at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ
  constant : ℝ

/-- The pressure-volume relationship is inversely proportional -/
axiom pressure_volume_constant (state : GasState) : state.pressure * state.volume = state.constant

/-- Initial state of the gas -/
def initial_state : GasState :=
  { pressure := 4
    volume := 3
    constant := 4 * 3 }

/-- New state of the gas after transfer -/
def new_state : GasState :=
  { pressure := 2  -- This is what we want to prove
    volume := 6
    constant := initial_state.constant }

/-- Theorem stating that the new pressure is 2 kPa -/
theorem new_pressure_is_two :
  new_state.pressure = 2 := by sorry

end NUMINAMATH_CALUDE_new_pressure_is_two_l1218_121803


namespace NUMINAMATH_CALUDE_exist_three_aliens_common_language_l1218_121862

/-- The number of aliens -/
def num_aliens : ℕ := 3 * Nat.factorial 2005

/-- The number of languages -/
def num_languages : ℕ := 2005

/-- A function representing the language used between two aliens -/
def communication_language : Fin num_aliens → Fin num_aliens → Fin num_languages := sorry

/-- The main theorem -/
theorem exist_three_aliens_common_language :
  ∃ (a b c : Fin num_aliens),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    communication_language a b = communication_language b c ∧
    communication_language b c = communication_language a c :=
by sorry

end NUMINAMATH_CALUDE_exist_three_aliens_common_language_l1218_121862


namespace NUMINAMATH_CALUDE_prob_all_co_captains_l1218_121820

/-- Represents a math team with a certain number of students and co-captains -/
structure MathTeam where
  size : Nat
  coCaptains : Nat

/-- Calculates the probability of selecting all co-captains from a single team -/
def probAllCoCaptains (team : MathTeam) : Rat :=
  1 / (Nat.choose team.size 3)

/-- The set of math teams in the area -/
def mathTeams : List MathTeam := [
  { size := 6, coCaptains := 3 },
  { size := 8, coCaptains := 3 },
  { size := 9, coCaptains := 3 },
  { size := 10, coCaptains := 3 }
]

/-- The main theorem stating the probability of selecting all co-captains -/
theorem prob_all_co_captains : 
  (List.sum (mathTeams.map probAllCoCaptains) / mathTeams.length : Rat) = 53 / 3360 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_co_captains_l1218_121820


namespace NUMINAMATH_CALUDE_factorial_equation_l1218_121845

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation (n : ℕ) : factorial 6 / factorial (6 - n) = 120 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l1218_121845


namespace NUMINAMATH_CALUDE_solve_coin_problem_l1218_121811

def coin_problem (total : ℕ) (coin1 : ℕ) (coin2 : ℕ) : Prop :=
  ∃ (max min : ℕ),
    (∃ (a : ℕ), a * coin1 = total ∧ a = max) ∧
    (∃ (b c : ℕ), b * coin1 + c * coin2 = total ∧ b + c = min) ∧
    max - min = 2

theorem solve_coin_problem :
  coin_problem 45 10 25 := by sorry

end NUMINAMATH_CALUDE_solve_coin_problem_l1218_121811


namespace NUMINAMATH_CALUDE_triangle_reflection_slope_l1218_121881

/-- Triangle DEF with vertices D(3,2), E(5,4), and F(2,6) reflected across y=2x -/
theorem triangle_reflection_slope (D E F D' E' F' : ℝ × ℝ) :
  D = (3, 2) →
  E = (5, 4) →
  F = (2, 6) →
  D' = (1, 3/2) →
  E' = (2, 5/2) →
  F' = (3, 1) →
  (D'.2 - D.2) / (D'.1 - D.1) ≠ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_reflection_slope_l1218_121881


namespace NUMINAMATH_CALUDE_root_in_interval_l1218_121861

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 4 = 0) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l1218_121861


namespace NUMINAMATH_CALUDE_derivative_symmetry_l1218_121838

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem derivative_symmetry (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l1218_121838


namespace NUMINAMATH_CALUDE_expression_equality_l1218_121817

theorem expression_equality (y a : ℝ) (h1 : y > 0) 
  (h2 : (a * y) / 20 + (3 * y) / 10 = 0.6 * y) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1218_121817


namespace NUMINAMATH_CALUDE_floor_product_equation_l1218_121831

theorem floor_product_equation : ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 70 ∧ x = (70 : ℝ) / 8 := by sorry

end NUMINAMATH_CALUDE_floor_product_equation_l1218_121831


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1218_121867

/-- Given an exam with mean score and a score below the mean, calculate the score above the mean -/
theorem exam_score_calculation (mean : ℝ) (below_score : ℝ) (below_sd : ℝ) (above_sd : ℝ)
  (h1 : mean = 76)
  (h2 : below_score = 60)
  (h3 : below_sd = 2)
  (h4 : above_sd = 3)
  (h5 : below_score = mean - below_sd * ((mean - below_score) / below_sd)) :
  mean + above_sd * ((mean - below_score) / below_sd) = 100 := by
sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1218_121867


namespace NUMINAMATH_CALUDE_parabola_equation_l1218_121876

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define points and vectors
variable (F A B C : ℝ × ℝ)  -- Points as pairs of real numbers
variable (AF FB BA BC : ℝ × ℝ)  -- Vectors as pairs of real numbers

-- Define vector operations
def vector_equal (v w : ℝ × ℝ) : Prop := v.1 = w.1 ∧ v.2 = w.2
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem parabola_equation (p : ℝ) :
  parabola p A.1 A.2 →  -- A is on the parabola
  vector_equal AF FB →  -- AF = FB
  dot_product BA BC = 48 →  -- BA · BC = 48
  p = 2 ∧ parabola 2 A.1 A.2  -- The parabola equation is y² = 4x
  := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1218_121876


namespace NUMINAMATH_CALUDE_line_slope_l1218_121851

theorem line_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1218_121851


namespace NUMINAMATH_CALUDE_sum_within_range_l1218_121832

/-- Converts a decimal number to its representation in a given base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in a given base to its decimal value -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Checks if a number is within the valid range -/
def isValidNumber (n : ℕ) : Prop :=
  n ≥ 3577 ∧ n ≤ 3583

/-- Calculates the sum of base conversions -/
def sumOfBaseConversions (n : ℕ) : ℕ :=
  fromBase (toBase n 7) 10 + fromBase (toBase n 8) 10 + fromBase (toBase n 9) 10

/-- Theorem: The sum of base conversions for valid numbers is within 0.5% of 25,000 -/
theorem sum_within_range (n : ℕ) (h : isValidNumber n) :
  (sumOfBaseConversions n : ℝ) > 24875 ∧ (sumOfBaseConversions n : ℝ) < 25125 :=
  sorry

end NUMINAMATH_CALUDE_sum_within_range_l1218_121832


namespace NUMINAMATH_CALUDE_truck_speed_on_dirt_road_l1218_121895

/-- A semi truck travels on two types of roads. This theorem proves the speed on the dirt road. -/
theorem truck_speed_on_dirt_road :
  ∀ (v : ℝ),
  (3 * v) + (2 * (v + 20)) = 200 →
  v = 32 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_on_dirt_road_l1218_121895


namespace NUMINAMATH_CALUDE_always_three_same_color_sum_zero_l1218_121819

-- Define a type for colors
inductive Color
| White
| Black

-- Define a function type for coloring integers
def Coloring := Int → Color

-- Define the property that 2016 and 2017 are different colors
def DifferentColors (c : Coloring) : Prop :=
  c 2016 ≠ c 2017

-- Define the property of three integers having the same color and summing to zero
def ThreeSameColorSumZero (c : Coloring) : Prop :=
  ∃ x y z : Int, (c x = c y ∧ c y = c z) ∧ x + y + z = 0

-- State the theorem
theorem always_three_same_color_sum_zero (c : Coloring) :
  DifferentColors c → ThreeSameColorSumZero c := by
  sorry

end NUMINAMATH_CALUDE_always_three_same_color_sum_zero_l1218_121819


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l1218_121875

theorem subset_implies_a_equals_three (A B : Set ℝ) (a : ℝ) : 
  A = {2, 3} → B = {1, 2, a} → A ⊆ B → a = 3 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l1218_121875


namespace NUMINAMATH_CALUDE_initial_number_proof_l1218_121869

theorem initial_number_proof (x : ℝ) : 
  x + 12.808 - 47.80600000000004 = 3854.002 ↔ x = 3889 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1218_121869


namespace NUMINAMATH_CALUDE_train_crossing_time_l1218_121816

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 56 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1218_121816


namespace NUMINAMATH_CALUDE_peter_age_is_16_l1218_121866

/-- Peter's present age -/
def PeterAge : ℕ := sorry

/-- Jacob's present age -/
def JacobAge : ℕ := sorry

/-- Theorem stating the conditions and the result to prove -/
theorem peter_age_is_16 :
  (JacobAge = PeterAge + 12) ∧
  (PeterAge - 10 = (JacobAge - 10) / 3) →
  PeterAge = 16 := by sorry

end NUMINAMATH_CALUDE_peter_age_is_16_l1218_121866


namespace NUMINAMATH_CALUDE_carltons_outfits_l1218_121849

/-- The number of unique outfit combinations for Carlton -/
def unique_outfit_combinations (button_up_shirts : ℕ) : ℕ :=
  let sweater_vests := 3 * button_up_shirts
  let ties := 2 * sweater_vests
  let shoes := 4 * ties
  let socks := 6 * shoes
  button_up_shirts * sweater_vests * ties * shoes * socks

/-- Theorem stating that Carlton's unique outfit combinations equal 77,760,000 -/
theorem carltons_outfits :
  unique_outfit_combinations 5 = 77760000 := by
  sorry

end NUMINAMATH_CALUDE_carltons_outfits_l1218_121849


namespace NUMINAMATH_CALUDE_smallest_number_is_five_l1218_121801

theorem smallest_number_is_five (x y z : ℕ) 
  (sum_xy : x + y = 20) 
  (sum_xz : x + z = 27) 
  (sum_yz : y + z = 37) : 
  min x (min y z) = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_is_five_l1218_121801


namespace NUMINAMATH_CALUDE_nikola_leaf_price_l1218_121896

/-- The price Nikola charges per leaf -/
def price_per_leaf : ℚ :=
  1 / 100

theorem nikola_leaf_price :
  let num_ants : ℕ := 400
  let food_per_ant : ℚ := 2
  let food_price : ℚ := 1 / 10
  let job_start_price : ℕ := 5
  let num_leaves : ℕ := 6000
  let num_jobs : ℕ := 4
  (↑num_jobs * job_start_price + ↑num_leaves * price_per_leaf : ℚ) =
    ↑num_ants * food_per_ant * food_price :=
by sorry

end NUMINAMATH_CALUDE_nikola_leaf_price_l1218_121896


namespace NUMINAMATH_CALUDE_conditional_probability_animal_longevity_l1218_121825

def prob_birth_to_20 : ℝ := 0.8
def prob_birth_to_25 : ℝ := 0.4

theorem conditional_probability_animal_longevity :
  (prob_birth_to_25 / prob_birth_to_20) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_animal_longevity_l1218_121825


namespace NUMINAMATH_CALUDE_clock_strike_time_l1218_121899

theorem clock_strike_time (strike_three : ℕ) (time_three : ℝ) (strike_six : ℕ) : 
  strike_three = 3 → time_three = 12 → strike_six = 6 → 
  ∃ (time_six : ℝ), time_six = 30 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_time_l1218_121899


namespace NUMINAMATH_CALUDE_stating_bus_students_theorem_l1218_121878

/-- 
Calculates the number of students on a bus after a series of stops where 
students get off and on, given an initial number of students.
-/
def students_after_stops (initial : ℚ) (fraction_off : ℚ) (num_stops : ℕ) (new_students : ℚ) : ℚ :=
  (initial * (1 - fraction_off)^num_stops) + new_students

/-- 
Theorem stating that given 72 initial students, with 1/3 getting off at each of 
the first four stops, and 12 new students boarding at the fifth stop, 
the final number of students is 236/9.
-/
theorem bus_students_theorem : 
  students_after_stops 72 (1/3) 4 12 = 236/9 := by
  sorry

end NUMINAMATH_CALUDE_stating_bus_students_theorem_l1218_121878


namespace NUMINAMATH_CALUDE_always_separable_l1218_121836

/-- Represents a cell in the square -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents a square of size 2n × 2n -/
structure Square (n : Nat) where
  size : Nat := 2 * n

/-- Represents a cut in the square -/
inductive Cut
  | Vertical : Nat → Cut
  | Horizontal : Nat → Cut

/-- Checks if two cells are separated by a cut -/
def separatedByCut (c1 c2 : Cell) (cut : Cut) : Prop :=
  match cut with
  | Cut.Vertical x => (c1.x ≤ x ∧ c2.x > x) ∨ (c1.x > x ∧ c2.x ≤ x)
  | Cut.Horizontal y => (c1.y ≤ y ∧ c2.y > y) ∨ (c1.y > y ∧ c2.y ≤ y)

/-- Main theorem: There always exists a cut that separates any two colored cells -/
theorem always_separable (n : Nat) (c1 c2 : Cell) 
    (h1 : c1.x < 2 * n ∧ c1.y < 2 * n)
    (h2 : c2.x < 2 * n ∧ c2.y < 2 * n)
    (h3 : c1 ≠ c2) :
    ∃ (cut : Cut), separatedByCut c1 c2 cut :=
  sorry


end NUMINAMATH_CALUDE_always_separable_l1218_121836


namespace NUMINAMATH_CALUDE_garage_sale_necklace_cost_l1218_121887

/-- The cost of each necklace in Isabel's garage sale --/
def cost_per_necklace (total_necklaces : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / total_necklaces

/-- Theorem stating that the cost per necklace is $6 --/
theorem garage_sale_necklace_cost :
  cost_per_necklace 6 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_necklace_cost_l1218_121887


namespace NUMINAMATH_CALUDE_min_omega_value_l1218_121863

/-- Given a function f(x) = sin(ω(x - π/4)) where ω > 0, if f(3π/4) = 0, then the minimum value of ω is 2. -/
theorem min_omega_value (ω : ℝ) (h₁ : ω > 0) :
  (fun x => Real.sin (ω * (x - π / 4))) (3 * π / 4) = 0 → ω ≥ 2 ∧ ∀ ω' > 0, (fun x => Real.sin (ω' * (x - π / 4))) (3 * π / 4) = 0 → ω' ≥ ω :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l1218_121863


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1218_121865

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) :
  (∃ x : ℂ, x^3 + a*x^2 + 6*x + b = 0 ∧ x = 1 - 3*I) →
  a = 0 ∧ b = 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1218_121865


namespace NUMINAMATH_CALUDE_f_of_5_equals_92_l1218_121888

/-- Given a function f(x) = 2x^2 + y where f(2) = 50, prove that f(5) = 92 -/
theorem f_of_5_equals_92 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y) 
  (h2 : f 2 = 50) : 
  f 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_92_l1218_121888


namespace NUMINAMATH_CALUDE_parabola_directrix_l1218_121818

/-- The directrix of the parabola y = (x^2 - 4x + 4) / 8 is y = -1/4 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => (x^2 - 4*x + 4) / 8
  ∃ (directrix : ℝ), directrix = -1/4 ∧
    ∀ (x y : ℝ), y = f x → 
      ∃ (focus : ℝ × ℝ), (x - focus.1)^2 + (y - focus.2)^2 = (y - directrix)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1218_121818


namespace NUMINAMATH_CALUDE_max_value_of_f_l1218_121840

noncomputable def f (x : ℝ) := 3 + Real.log x + 4 / Real.log x

theorem max_value_of_f :
  (∀ x : ℝ, 0 < x → x < 1 → f x ≤ -1) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = -1) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1218_121840


namespace NUMINAMATH_CALUDE_solution_x_l1218_121852

-- Define m and n as distinct non-zero real constants
variable (m n : ℝ) (h : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x + m)^2 - 3*(x + n)^2 = m^2 - 3*n^2

-- Theorem statement
theorem solution_x (x : ℝ) : 
  equation m n x → (x = 0 ∨ x = m - 3*n) :=
by sorry

end NUMINAMATH_CALUDE_solution_x_l1218_121852


namespace NUMINAMATH_CALUDE_reciprocals_inversely_proportional_l1218_121822

/-- Two real numbers are inversely proportional if their product is constant --/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

/-- Theorem: If x and y are inversely proportional, then their reciprocals are also inversely proportional --/
theorem reciprocals_inversely_proportional
  (x y : ℝ → ℝ)
  (h : InverselyProportional x y)
  (hx : ∀ t, x t ≠ 0)
  (hy : ∀ t, y t ≠ 0) :
  InverselyProportional (fun t ↦ 1 / x t) (fun t ↦ 1 / y t) :=
by
  sorry

end NUMINAMATH_CALUDE_reciprocals_inversely_proportional_l1218_121822


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1218_121885

-- Define Ω as a nonreal root of z^4 = 1
def Ω : ℂ := Complex.I

-- Define the condition for valid pairs
def isValidPair (a b : ℤ) : Prop := Complex.abs (a • Ω + b) = 2

-- Theorem statement
theorem count_valid_pairs : 
  (∃! (n : ℕ), ∃ (s : Finset (ℤ × ℤ)), s.card = n ∧ 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ isValidPair p.1 p.2) ∧ n = 4) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1218_121885


namespace NUMINAMATH_CALUDE_satellite_sensor_upgrade_fraction_l1218_121812

theorem satellite_sensor_upgrade_fraction :
  ∀ (total_units : ℕ) (non_upgraded_per_unit : ℕ) (total_upgraded : ℕ),
    total_units = 24 →
    non_upgraded_per_unit * 4 = total_upgraded →
    (total_upgraded : ℚ) / (total_upgraded + total_units * non_upgraded_per_unit) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_satellite_sensor_upgrade_fraction_l1218_121812


namespace NUMINAMATH_CALUDE_monotonic_sine_phi_range_l1218_121805

theorem monotonic_sine_phi_range (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = -2 * Real.sin (2 * x + φ)) →
  (|φ| < π) →
  (∀ x ∈ Set.Ioo (π / 5) ((5 / 8) * π), StrictMono f) →
  φ ∈ Set.Ioo (π / 10) (π / 4) := by
sorry

end NUMINAMATH_CALUDE_monotonic_sine_phi_range_l1218_121805


namespace NUMINAMATH_CALUDE_square_root_of_1024_l1218_121834

theorem square_root_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 1024) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l1218_121834


namespace NUMINAMATH_CALUDE_some_number_value_l1218_121847

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * x * 49) : x = 315 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1218_121847


namespace NUMINAMATH_CALUDE_inequality_solution_l1218_121898

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition f'(x) > f(x)
variable (h : ∀ x : ℝ, deriv f x > f x)

-- Define the solution set
def solution_set := {x : ℝ | Real.exp (f (Real.log x)) - x * f 1 < 0}

-- Theorem statement
theorem inequality_solution :
  solution_set f = Ioo 0 (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1218_121898


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1218_121843

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptote : ∀ x, ∃ y, y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1218_121843


namespace NUMINAMATH_CALUDE_part_one_part_two_l1218_121877

/-- The absolute value function -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- Part I: Range of a such that f(x) ≤ 3 for all x in [-1, 3] -/
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, x ∈ [-1, 3] → f a x ≤ 3) ↔ a ∈ Set.Icc 0 2 := by sorry

/-- Part II: Minimum value of a such that f(x-a) + f(x+a) ≥ 1-2a for all x -/
theorem part_two : 
  (∃ a : ℝ, (∀ x : ℝ, f a (x-a) + f a (x+a) ≥ 1-2*a) ∧ 
   (∀ b : ℝ, (∀ x : ℝ, f b (x-b) + f b (x+b) ≥ 1-2*b) → a ≤ b)) ∧
  (let a := (1/4 : ℝ); ∀ x : ℝ, f a (x-a) + f a (x+a) ≥ 1-2*a) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1218_121877


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l1218_121814

theorem mary_baseball_cards (X : ℕ) : 
  X - 8 + 26 + 40 = 84 → X = 26 := by
sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l1218_121814


namespace NUMINAMATH_CALUDE_mixed_fraction_calculation_l1218_121813

theorem mixed_fraction_calculation :
  (13/4 - 13/5 + 21/4 + (-42/5) : ℚ) = -5/2 ∧
  (-(3^2) - (-5 + 3) + 27 / (-3) * (1/3) : ℚ) = -10 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_calculation_l1218_121813


namespace NUMINAMATH_CALUDE_expand_product_l1218_121873

theorem expand_product (x : ℝ) : (x + 3) * (x + 7) = x^2 + 10*x + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1218_121873


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1218_121837

theorem shopkeeper_profit (CP : ℝ) (CP_pos : CP > 0) : 
  let LP := CP * 1.3
  let SP := LP * 0.9
  let profit := SP - CP
  let percent_profit := (profit / CP) * 100
  percent_profit = 17 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1218_121837


namespace NUMINAMATH_CALUDE_existence_of_pair_l1218_121860

theorem existence_of_pair (x : Fin 670 → ℝ)
  (h_positive : ∀ i, 0 < x i)
  (h_less_than_one : ∀ i, x i < 1)
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ i j, i ≠ j ∧ 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_pair_l1218_121860


namespace NUMINAMATH_CALUDE_article_cost_l1218_121855

/-- Represents the cost and selling price of an article -/
structure Article where
  cost : ℝ
  sellingPrice : ℝ

/-- The original article with 25% profit -/
def originalArticle : Article → Prop := fun a => 
  a.sellingPrice = 1.25 * a.cost

/-- The new article with reduced cost and selling price -/
def newArticle : Article → Prop := fun a => 
  (0.8 * a.cost) * 1.3 = a.sellingPrice - 16.8

/-- Theorem stating that the cost of the article is 80 -/
theorem article_cost : ∃ a : Article, originalArticle a ∧ newArticle a ∧ a.cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1218_121855


namespace NUMINAMATH_CALUDE_inequality_proof_l1218_121833

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1218_121833


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_20n_integer_l1218_121879

theorem smallest_n_for_sqrt_20n_integer (n : ℕ) : 
  (∃ k : ℕ, k ^ 2 = 20 * n) → (∀ m : ℕ, m > 0 ∧ m < n → ¬∃ k : ℕ, k ^ 2 = 20 * m) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_20n_integer_l1218_121879


namespace NUMINAMATH_CALUDE_necessary_condition_for_P_l1218_121870

-- Define the set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Define the proposition P(a)
def P (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- Theorem statement
theorem necessary_condition_for_P :
  (∃ a : ℝ, P a) → (∀ a : ℝ, P a → a ≥ 1) ∧ ¬(∀ a : ℝ, a ≥ 1 → P a) := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_P_l1218_121870


namespace NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1218_121857

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem: The center of the given hyperbola is (3, 5) -/
theorem hyperbola_center_is_correct :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    ((x - hyperbola_center.1)^2 / 5 - (y - hyperbola_center.2)^2 / (5/4) = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1218_121857


namespace NUMINAMATH_CALUDE_equation_transformation_l1218_121871

theorem equation_transformation (x y : ℚ) : 
  5 * x - 6 * y = 4 → y = (5/6) * x - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1218_121871


namespace NUMINAMATH_CALUDE_sqrt_a_3a_sqrt_a_l1218_121842

theorem sqrt_a_3a_sqrt_a (a : ℝ) (ha : a > 0) :
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_3a_sqrt_a_l1218_121842


namespace NUMINAMATH_CALUDE_prob_different_topics_correct_l1218_121835

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    out of num_topics is equal to prob_different_topics -/
theorem prob_different_topics_correct :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics := by
  sorry

end NUMINAMATH_CALUDE_prob_different_topics_correct_l1218_121835


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l1218_121826

theorem max_value_of_sum_products (a b c d : ℝ) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  a + b + c + d = 150 →
  a * b + b * c + c * d ≤ 5625 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l1218_121826


namespace NUMINAMATH_CALUDE_part_i_part_ii_l1218_121841

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |3 + a|

-- Part I
theorem part_i : 
  {x : ℝ | f 3 x > 6} = {x : ℝ | x < -4 ∨ x > 2} :=
sorry

-- Part II
theorem part_ii :
  (∃ x : ℝ, g a x = 0) → a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l1218_121841


namespace NUMINAMATH_CALUDE_power_expression_l1218_121864

theorem power_expression (a x y : ℝ) (ha : a > 0) (hx : a^x = 3) (hy : a^y = 5) :
  a^(2*x + y/2) = 9 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_l1218_121864


namespace NUMINAMATH_CALUDE_card_area_reduction_l1218_121806

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_reduction (initial : Rectangle) :
  initial.length = 5 ∧ initial.width = 8 →
  ∃ (reduced : Rectangle),
    (reduced.length = initial.length - 2 ∨ reduced.width = initial.width - 2) ∧
    area reduced = 21 →
  ∃ (other_reduced : Rectangle),
    (other_reduced.length = initial.length - 2 ∨ other_reduced.width = initial.width - 2) ∧
    other_reduced ≠ reduced ∧
    area other_reduced = 24 := by
  sorry

end NUMINAMATH_CALUDE_card_area_reduction_l1218_121806


namespace NUMINAMATH_CALUDE_jiangsu_income_scientific_notation_l1218_121858

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Rounds a real number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original amount in yuan -/
def originalAmount : ℝ := 26341

/-- The number of significant figures required -/
def requiredSigFigs : ℕ := 3

theorem jiangsu_income_scientific_notation :
  toScientificNotation originalAmount requiredSigFigs =
    ScientificNotation.mk 2.63 4 := by sorry

end NUMINAMATH_CALUDE_jiangsu_income_scientific_notation_l1218_121858


namespace NUMINAMATH_CALUDE_trailing_zeros_of_500_power_150_l1218_121830

-- Define 500 as 5 * 10^2
def five_hundred : ℕ := 5 * 10^2

-- Define the exponent
def exponent : ℕ := 150

-- Define the function to count trailing zeros
def trailing_zeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem trailing_zeros_of_500_power_150 :
  trailing_zeros (five_hundred ^ exponent) = 300 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_500_power_150_l1218_121830


namespace NUMINAMATH_CALUDE_clerical_to_total_ratio_l1218_121872

def total_employees : ℕ := 3600

def clerical_ratio (c : ℕ) : Prop :=
  (c / 2 : ℚ) = 0.2 * (total_employees - c / 2 : ℚ)

theorem clerical_to_total_ratio :
  ∃ c : ℕ, clerical_ratio c ∧ c * 3 = total_employees :=
sorry

end NUMINAMATH_CALUDE_clerical_to_total_ratio_l1218_121872


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1218_121889

/-- If x^2 + 6x + k^2 is a perfect square polynomial, then k = ± 3 -/
theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → 
  k = 3 ∨ k = -3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1218_121889


namespace NUMINAMATH_CALUDE_vector_problem_l1218_121802

-- Define the vectors
def OA (k : ℝ) : ℝ × ℝ := (k, 12)
def OB : ℝ × ℝ := (4, 5)
def OC (k : ℝ) : ℝ × ℝ := (-k, 10)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

-- State the theorem
theorem vector_problem (k : ℝ) :
  collinear (OA k) OB (OC k) → k = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l1218_121802


namespace NUMINAMATH_CALUDE_statue_final_weight_l1218_121874

/-- Calculates the final weight of a statue after three weeks of carving. -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_first_week := initial_weight * (1 - 0.3)
  let weight_after_second_week := weight_after_first_week * (1 - 0.3)
  let weight_after_third_week := weight_after_second_week * (1 - 0.15)
  weight_after_third_week

/-- Theorem stating that the final weight of the statue is 124.95 kg. -/
theorem statue_final_weight :
  final_statue_weight 300 = 124.95 := by
  sorry

end NUMINAMATH_CALUDE_statue_final_weight_l1218_121874


namespace NUMINAMATH_CALUDE_max_posters_purchasable_l1218_121815

def initial_amount : ℕ := 20
def book1_price : ℕ := 8
def book2_price : ℕ := 4
def poster_price : ℕ := 4

theorem max_posters_purchasable :
  (initial_amount - book1_price - book2_price) / poster_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_posters_purchasable_l1218_121815


namespace NUMINAMATH_CALUDE_find_divisor_l1218_121894

theorem find_divisor : ∃ (d : ℕ), d = 675 ∧ (9679 - 4) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1218_121894


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l1218_121884

theorem smaller_integer_problem (x y : ℤ) : 
  y = 2 * x → x + y = 96 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l1218_121884


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1218_121853

theorem sum_of_solutions (x : ℝ) : (x + 16 / x = 12) → (∃ y : ℝ, y + 16 / y = 12 ∧ y ≠ x) → x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1218_121853


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l1218_121850

theorem sqrt_difference_inequality (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l1218_121850


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1218_121846

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 - 2*x^5 + 3*x^4 - 4*x^3 + 5*x^2 - 6*x + 12 = 
  (x - 1) * (x^5 - x^4 + 2*x^3 - 2*x^2 + 3*x - 3) + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1218_121846


namespace NUMINAMATH_CALUDE_bucket_filling_time_l1218_121868

theorem bucket_filling_time (total_time : ℝ) (total_fraction : ℝ) (partial_fraction : ℝ) : 
  total_time = 150 → total_fraction = 1 → partial_fraction = 2/3 →
  (partial_fraction * total_time) / total_fraction = 100 := by
sorry

end NUMINAMATH_CALUDE_bucket_filling_time_l1218_121868


namespace NUMINAMATH_CALUDE_estimate_expression_range_l1218_121800

theorem estimate_expression_range : 
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1/5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1/5) < 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_expression_range_l1218_121800


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1218_121844

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 4 * x - 15) =
  x^3 + 2 * x^2 + 2 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1218_121844


namespace NUMINAMATH_CALUDE_single_burger_cost_l1218_121828

/-- Proves that the cost of a single burger is $1.00 given the specified conditions -/
theorem single_burger_cost
  (total_spent : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (double_burger_cost : ℝ)
  (h1 : total_spent = 74.50)
  (h2 : total_hamburgers = 50)
  (h3 : double_burgers = 49)
  (h4 : double_burger_cost = 1.50) :
  total_spent - (double_burgers * double_burger_cost) = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_single_burger_cost_l1218_121828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1218_121848

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 8 = 3 →
  a 1 + a 10 = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1218_121848


namespace NUMINAMATH_CALUDE_downstream_distance_is_35_l1218_121821

-- Define the given constants
def man_speed : ℝ := 5.5
def upstream_distance : ℝ := 20
def swim_time : ℝ := 5

-- Define the theorem
theorem downstream_distance_is_35 :
  let stream_speed := (man_speed - upstream_distance / swim_time) / 2
  let downstream_distance := (man_speed + stream_speed) * swim_time
  downstream_distance = 35 := by sorry

end NUMINAMATH_CALUDE_downstream_distance_is_35_l1218_121821
