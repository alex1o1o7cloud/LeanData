import Mathlib

namespace NUMINAMATH_CALUDE_solid_circles_count_l3043_304393

def circleSequence (n : ℕ) : ℕ := n * (n + 3) / 2 + 1

theorem solid_circles_count (total : ℕ) (h : total = 2019) :
  ∃ n : ℕ, circleSequence n ≤ total ∧ circleSequence (n + 1) > total ∧ n = 62 :=
by sorry

end NUMINAMATH_CALUDE_solid_circles_count_l3043_304393


namespace NUMINAMATH_CALUDE_construct_equilateral_triangle_l3043_304346

/-- A triangle with two 70° angles and one 40° angle -/
structure WoodenTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  two_70 : (angle1 = 70 ∧ angle2 = 70) ∨ (angle1 = 70 ∧ angle3 = 70) ∨ (angle2 = 70 ∧ angle3 = 70)
  one_40 : angle1 = 40 ∨ angle2 = 40 ∨ angle3 = 40

/-- An equilateral triangle has three 60° angles -/
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = 60 ∧ b = 60 ∧ c = 60

/-- The theorem stating that an equilateral triangle can be constructed using only the wooden triangle -/
theorem construct_equilateral_triangle (wt : WoodenTriangle) :
  ∃ a b c : ℝ, is_equilateral_triangle a b c ∧
  (∃ (n : ℕ), n > 0 ∧ a + b + c = n * (wt.angle1 + wt.angle2 + wt.angle3)) :=
sorry

end NUMINAMATH_CALUDE_construct_equilateral_triangle_l3043_304346


namespace NUMINAMATH_CALUDE_four_Y_three_equals_49_l3043_304332

-- Define the new operation Y
def Y (a b : ℝ) : ℝ := (a + b)^2

-- State the theorem
theorem four_Y_three_equals_49 : Y 4 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_49_l3043_304332


namespace NUMINAMATH_CALUDE_largest_angle_120_l3043_304315

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ)

-- Define properties of the triangle
def isObtuse (t : Triangle) : Prop :=
  t.P > 90 ∨ t.Q > 90 ∨ t.R > 90

def isIsosceles (t : Triangle) : Prop :=
  (t.P = t.Q) ∨ (t.Q = t.R) ∨ (t.P = t.R)

def angleP30 (t : Triangle) : Prop :=
  t.P = 30

-- Theorem statement
theorem largest_angle_120 (t : Triangle) 
  (h1 : isObtuse t) 
  (h2 : isIsosceles t) 
  (h3 : angleP30 t) : 
  max t.P (max t.Q t.R) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_120_l3043_304315


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l3043_304334

def power_product : ℕ := 2^2009 * 5^2010 * 7

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_power_product : sum_of_digits power_product = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l3043_304334


namespace NUMINAMATH_CALUDE_f_derivative_at_one_is_zero_g_derivative_formula_l3043_304379

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / x

def g (x : ℝ) : ℝ := f (2 * x)

theorem f_derivative_at_one_is_zero :
  deriv f 1 = 0 := by sorry

theorem g_derivative_formula (x : ℝ) (h : x ≠ 0) :
  deriv g x = (Real.exp (2 * x) * (2 * x - 1)) / (2 * x^2) := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_is_zero_g_derivative_formula_l3043_304379


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3043_304330

/-- 
Given an arithmetic sequence with first term a₁ = k^2 - k + 1 and common difference d = 1,
the sum of the first k + 2 terms is equal to k^3 + 2k^2 + k + 2.
-/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ := k^2 - k + 1
  let d := 1
  let n := k + 2
  let Sn := n * (a₁ + (a₁ + (n - 1) * d)) / 2
  Sn = k^3 + 2*k^2 + k + 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3043_304330


namespace NUMINAMATH_CALUDE_daves_old_cards_l3043_304356

/-- Given Dave's baseball card organization, prove the number of old cards --/
theorem daves_old_cards
  (cards_per_page : ℕ)
  (new_cards : ℕ)
  (pages_used : ℕ)
  (h1 : cards_per_page = 8)
  (h2 : new_cards = 3)
  (h3 : pages_used = 2) :
  pages_used * cards_per_page - new_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_daves_old_cards_l3043_304356


namespace NUMINAMATH_CALUDE_min_distance_between_curve_and_line_l3043_304351

theorem min_distance_between_curve_and_line :
  ∀ (a b c d : ℝ),
  (Real.log b + 1 + a - 3 * b = 0) →
  (2 * d - c + Real.sqrt 5 = 0) →
  (∃ (m : ℝ), ∀ (a' b' c' d' : ℝ),
    (Real.log b' + 1 + a' - 3 * b' = 0) →
    (2 * d' - c' + Real.sqrt 5 = 0) →
    (a - c)^2 + (b - d)^2 ≤ (a' - c')^2 + (b' - d')^2) →
  m = 4/5 := by
sorry

end NUMINAMATH_CALUDE_min_distance_between_curve_and_line_l3043_304351


namespace NUMINAMATH_CALUDE_sector_area_l3043_304342

/-- The area of a circular sector with given radius and arc length. -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h : r > 0) :
  let area := (1 / 2) * r * arc_length
  r = 15 ∧ arc_length = π / 3 → area = 5 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3043_304342


namespace NUMINAMATH_CALUDE_tangent_line_at_one_e_l3043_304300

/-- The tangent line to y = xe^x at (1, e) -/
theorem tangent_line_at_one_e :
  let f (x : ℝ) := x * Real.exp x
  let f' (x : ℝ) := Real.exp x + x * Real.exp x
  let tangent_line (x : ℝ) := 2 * Real.exp 1 * x - Real.exp 1
  f' 1 = 2 * Real.exp 1 ∧
  tangent_line 1 = f 1 ∧
  ∀ x, tangent_line x - f x = f' 1 * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_e_l3043_304300


namespace NUMINAMATH_CALUDE_negation_equivalence_l3043_304318

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x ∈ Set.Icc 1 2 → 2 * x^2 - 3 ≥ 0) ↔
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3043_304318


namespace NUMINAMATH_CALUDE_gears_rotating_when_gear1_rotates_l3043_304374

-- Define the state of a gear (rotating or stopped)
inductive GearState
| rotating
| stopped

-- Define the gearbox with 6 gears
structure Gearbox :=
  (gear1 gear2 gear3 gear4 gear5 gear6 : GearState)

-- Define the conditions of the gearbox operation
def validGearbox (gb : Gearbox) : Prop :=
  -- Condition 1
  (gb.gear1 = GearState.rotating → gb.gear2 = GearState.rotating ∧ gb.gear5 = GearState.stopped) ∧
  -- Condition 2
  ((gb.gear2 = GearState.rotating ∨ gb.gear5 = GearState.rotating) → gb.gear4 = GearState.stopped) ∧
  -- Condition 3
  (gb.gear3 = GearState.rotating ↔ gb.gear4 = GearState.rotating) ∧
  -- Condition 4
  (gb.gear5 = GearState.rotating ∨ gb.gear6 = GearState.rotating)

-- Theorem statement
theorem gears_rotating_when_gear1_rotates (gb : Gearbox) :
  validGearbox gb →
  gb.gear1 = GearState.rotating →
  gb.gear2 = GearState.rotating ∧ gb.gear3 = GearState.rotating ∧ gb.gear6 = GearState.rotating :=
by sorry

end NUMINAMATH_CALUDE_gears_rotating_when_gear1_rotates_l3043_304374


namespace NUMINAMATH_CALUDE_additional_stickers_needed_l3043_304311

def current_stickers : ℕ := 35
def row_size : ℕ := 8

theorem additional_stickers_needed :
  let next_multiple := (current_stickers + row_size - 1) / row_size * row_size
  next_multiple - current_stickers = 5 := by sorry

end NUMINAMATH_CALUDE_additional_stickers_needed_l3043_304311


namespace NUMINAMATH_CALUDE_complex_expansion_l3043_304336

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_expansion : (1 - i) * (1 + 2*i)^2 = 1 + 7*i := by
  sorry

end NUMINAMATH_CALUDE_complex_expansion_l3043_304336


namespace NUMINAMATH_CALUDE_y_divisibility_l3043_304321

def y : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

theorem y_divisibility :
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  (∃ k : ℕ, y = 32 * k) ∧
  (∃ k : ℕ, y = 64 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l3043_304321


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l3043_304399

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to the foci. -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  sum_distances : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse. -/
structure EllipseParameters where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem: For the given ellipse, h + k + a + b = 9 + √7 -/
theorem ellipse_parameter_sum (E : Ellipse) (P : EllipseParameters) :
  E.F₁ = (0, 2) →
  E.F₂ = (6, 2) →
  E.sum_distances = 8 →
  P.h + P.k + P.a + P.b = 9 + Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l3043_304399


namespace NUMINAMATH_CALUDE_exponent_division_l3043_304303

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3043_304303


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3043_304344

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 - x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x - 1

-- Theorem statement
theorem tangent_line_at_origin : 
  ∀ x y : ℝ, (x + y = 0) ↔ (∃ t : ℝ, y = f t ∧ y - f 0 = f' 0 * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3043_304344


namespace NUMINAMATH_CALUDE_relative_error_approximation_l3043_304320

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  let f := fun x => 1 / (1 + x)
  let approx := fun x => 1 - x
  let relative_error := fun x => (f x - approx x) / f x
  relative_error y = y^2 := by
  sorry

end NUMINAMATH_CALUDE_relative_error_approximation_l3043_304320


namespace NUMINAMATH_CALUDE_trig_identity_l3043_304312

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 / Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3043_304312


namespace NUMINAMATH_CALUDE_election_votes_l3043_304327

theorem election_votes (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    loser_votes = (30 * total_votes) / 100 ∧
    winner_votes - loser_votes = 174) →
  total_votes = 435 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3043_304327


namespace NUMINAMATH_CALUDE_jimin_has_greater_sum_l3043_304384

theorem jimin_has_greater_sum : 
  let jungkook_num1 : ℕ := 4
  let jungkook_num2 : ℕ := 4
  let jimin_num1 : ℕ := 3
  let jimin_num2 : ℕ := 6
  jimin_num1 + jimin_num2 > jungkook_num1 + jungkook_num2 :=
by
  sorry

end NUMINAMATH_CALUDE_jimin_has_greater_sum_l3043_304384


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_682_l3043_304316

theorem sin_n_equals_cos_682 (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (682 * π / 180) → n = 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_682_l3043_304316


namespace NUMINAMATH_CALUDE_sugar_problem_l3043_304373

theorem sugar_problem (initial_sugar : ℝ) : 
  (initial_sugar / 4 * 3.5 = 21) → initial_sugar = 24 := by
  sorry

end NUMINAMATH_CALUDE_sugar_problem_l3043_304373


namespace NUMINAMATH_CALUDE_bread_cost_l3043_304368

theorem bread_cost (initial_amount : ℕ) (amount_left : ℕ) (num_bread : ℕ) (num_milk : ℕ) 
  (h1 : initial_amount = 47)
  (h2 : amount_left = 35)
  (h3 : num_bread = 4)
  (h4 : num_milk = 2) :
  (initial_amount - amount_left) / (num_bread + num_milk) = 2 :=
by sorry

end NUMINAMATH_CALUDE_bread_cost_l3043_304368


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3043_304353

/-- Properties of the hyperbola x^2 - y^2 = 2 -/
theorem hyperbola_properties :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 2
  ∃ (a b c : ℝ),
    (∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (2 * a = 2 * Real.sqrt 2) ∧
    (c^2 = a^2 + b^2) ∧
    (c / a = Real.sqrt 2) ∧
    (∀ x y, (y = x ∨ y = -x) → h x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3043_304353


namespace NUMINAMATH_CALUDE_sqrt_difference_sum_abs_l3043_304326

theorem sqrt_difference_sum_abs : 1 + (Real.sqrt 2 - Real.sqrt 3) + |Real.sqrt 2 - Real.sqrt 3| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_sum_abs_l3043_304326


namespace NUMINAMATH_CALUDE_square_root_to_cube_l3043_304369

theorem square_root_to_cube (x : ℝ) : 
  Real.sqrt (2 * x + 4) = 4 → (2 * x + 4)^3 = 4096 := by sorry

end NUMINAMATH_CALUDE_square_root_to_cube_l3043_304369


namespace NUMINAMATH_CALUDE_total_bike_ride_l3043_304352

def morning_ride : ℝ := 2
def evening_ride_factor : ℝ := 5

theorem total_bike_ride : morning_ride + evening_ride_factor * morning_ride = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_bike_ride_l3043_304352


namespace NUMINAMATH_CALUDE_workshop_workers_l3043_304341

/-- The total number of workers in a workshop, given specific salary conditions -/
theorem workshop_workers (average_salary : ℚ) (technician_salary : ℚ) (other_salary : ℚ)
  (h1 : average_salary = 8000)
  (h2 : technician_salary = 18000)
  (h3 : other_salary = 6000) :
  ∃ (total_workers : ℕ), 
    (7 : ℚ) * technician_salary + (total_workers - 7 : ℚ) * other_salary = (total_workers : ℚ) * average_salary ∧
    total_workers = 42 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l3043_304341


namespace NUMINAMATH_CALUDE_bucket_pouring_l3043_304389

theorem bucket_pouring (capacity_a capacity_b : ℚ) : 
  capacity_b = (1 / 2) * capacity_a →
  let initial_sand_a := (1 / 4) * capacity_a
  let initial_sand_b := (3 / 8) * capacity_b
  let final_sand_a := initial_sand_a + initial_sand_b
  final_sand_a = (7 / 16) * capacity_a :=
by sorry

end NUMINAMATH_CALUDE_bucket_pouring_l3043_304389


namespace NUMINAMATH_CALUDE_proposition_p_equivalence_l3043_304391

theorem proposition_p_equivalence (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - m - 1 < 0) ↔ m > -2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_equivalence_l3043_304391


namespace NUMINAMATH_CALUDE_softball_team_ratio_l3043_304313

/-- Represents a co-ed softball team --/
structure Team where
  men : ℕ
  women : ℕ
  total : ℕ
  h1 : women = men + 4
  h2 : men + women = total

/-- The ratio of men to women in a team --/
def menWomenRatio (t : Team) : Rat :=
  t.men / t.women

theorem softball_team_ratio (t : Team) (h : t.total = 14) :
  menWomenRatio t = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l3043_304313


namespace NUMINAMATH_CALUDE_increasing_not_always_unbounded_and_decreasing_not_always_unbounded_l3043_304390

-- Define a constantly increasing function
def constantlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define a constantly decreasing function
def constantlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define a function that is bounded above
def boundedAbove (f : ℝ → ℝ) : Prop :=
  ∃ M, ∀ x, f x ≤ M

-- Define a function that is bounded below
def boundedBelow (f : ℝ → ℝ) : Prop :=
  ∃ m, ∀ x, f x ≥ m

-- Theorem statement
theorem increasing_not_always_unbounded_and_decreasing_not_always_unbounded :
  (∃ f : ℝ → ℝ, constantlyIncreasing f ∧ boundedAbove f) ∧
  (∃ g : ℝ → ℝ, constantlyDecreasing g ∧ boundedBelow g) :=
sorry

end NUMINAMATH_CALUDE_increasing_not_always_unbounded_and_decreasing_not_always_unbounded_l3043_304390


namespace NUMINAMATH_CALUDE_opposite_of_one_sixth_l3043_304301

theorem opposite_of_one_sixth :
  -(1 / 6 : ℚ) = -1 / 6 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_sixth_l3043_304301


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l3043_304331

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one :
  ∀ m n : ℕ, m > 0 ∧ n > 0 → x m ≠ y n :=
by sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l3043_304331


namespace NUMINAMATH_CALUDE_x_plus_z_equals_15_l3043_304394

theorem x_plus_z_equals_15 (x y z : ℝ) 
  (h1 : |x| + x + z = 15) 
  (h2 : x + |y| - y = 8) : 
  x + z = 15 := by
sorry

end NUMINAMATH_CALUDE_x_plus_z_equals_15_l3043_304394


namespace NUMINAMATH_CALUDE_clock_angle_at_seven_l3043_304314

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
def degrees_per_hour : ℕ := full_circle / clock_hours

/-- The hour we're examining -/
def current_hour : ℕ := 7

/-- The smaller angle between the hour hand and 12 o'clock position -/
def smaller_angle : ℕ := min (current_hour * degrees_per_hour) ((clock_hours - current_hour) * degrees_per_hour)

theorem clock_angle_at_seven : smaller_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_seven_l3043_304314


namespace NUMINAMATH_CALUDE_min_yellow_fraction_l3043_304305

/-- Represents a cube with its edge length and number of blue and yellow subcubes. -/
structure Cube where
  edge_length : ℕ
  blue_cubes : ℕ
  yellow_cubes : ℕ

/-- Calculates the minimum yellow surface area for a given cube configuration. -/
def min_yellow_surface_area (c : Cube) : ℕ :=
  sorry

/-- Calculates the total surface area of a cube. -/
def total_surface_area (c : Cube) : ℕ :=
  6 * c.edge_length * c.edge_length

/-- The main theorem stating the minimum fraction of yellow surface area. -/
theorem min_yellow_fraction (c : Cube) 
  (h1 : c.edge_length = 4)
  (h2 : c.blue_cubes = 48)
  (h3 : c.yellow_cubes = 16)
  (h4 : c.blue_cubes + c.yellow_cubes = c.edge_length * c.edge_length * c.edge_length) :
  min_yellow_surface_area c / total_surface_area c = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_min_yellow_fraction_l3043_304305


namespace NUMINAMATH_CALUDE_original_bottle_size_l3043_304322

/-- The amount of wax needed for Kellan's car -/
def car_wax : ℕ := 3

/-- The amount of wax needed for Kellan's SUV -/
def suv_wax : ℕ := 4

/-- The amount of wax spilled before use -/
def spilled_wax : ℕ := 2

/-- The amount of wax left after detailing both vehicles -/
def leftover_wax : ℕ := 2

/-- Theorem stating the original bottle size -/
theorem original_bottle_size : 
  car_wax + suv_wax + spilled_wax + leftover_wax = 11 := by
  sorry

end NUMINAMATH_CALUDE_original_bottle_size_l3043_304322


namespace NUMINAMATH_CALUDE_variance_of_doubled_data_l3043_304363

-- Define a set of data as a list of real numbers
def DataSet := List ℝ

-- Define the standard deviation of a data set
noncomputable def standardDeviation (data : DataSet) : ℝ := sorry

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define a function to double each element in a data set
def doubleData (data : DataSet) : DataSet := data.map (· * 2)

-- Theorem statement
theorem variance_of_doubled_data (data : DataSet) :
  let s := standardDeviation data
  variance (doubleData data) = 4 * (s ^ 2) := by sorry

end NUMINAMATH_CALUDE_variance_of_doubled_data_l3043_304363


namespace NUMINAMATH_CALUDE_fifth_term_value_l3043_304376

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_value
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a 2)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3043_304376


namespace NUMINAMATH_CALUDE_parabola_equation_l3043_304354

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the problem
theorem parabola_equation (p : Parabola) (t : Triangle) (F : Point) :
  -- The vertex of the parabola is at the origin
  p.equation 0 0 ∧
  -- The focus of the parabola is on the x-axis
  F.y = 0 ∧
  -- The three vertices of triangle ABC lie on the parabola
  p.equation t.A.x t.A.y ∧ p.equation t.B.x t.B.y ∧ p.equation t.C.x t.C.y ∧
  -- The centroid of triangle ABC is the focus F of the parabola
  F.x = (t.A.x + t.B.x + t.C.x) / 3 ∧ F.y = (t.A.y + t.B.y + t.C.y) / 3 ∧
  -- The equation of the line where side BC lies is 4x + y - 20 = 0
  4 * t.B.x + t.B.y = 20 ∧ 4 * t.C.x + t.C.y = 20 →
  -- The equation of the parabola is y² = 16x
  ∀ x y, p.equation x y ↔ y^2 = 16*x :=
by sorry


end NUMINAMATH_CALUDE_parabola_equation_l3043_304354


namespace NUMINAMATH_CALUDE_place_value_sum_l3043_304335

/-- Given place values, prove the total number -/
theorem place_value_sum (thousands hundreds tens ones : ℕ) :
  thousands = 6 →
  hundreds = 3 →
  tens = 9 →
  ones = 7 →
  thousands * 1000 + hundreds * 100 + tens * 10 + ones = 6397 := by
  sorry

end NUMINAMATH_CALUDE_place_value_sum_l3043_304335


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3043_304358

theorem right_triangle_third_side (x y : ℝ) :
  (x > 0 ∧ y > 0) →
  (|x^2 - 4| + Real.sqrt (y^2 - 5*y + 6) = 0) →
  ∃ z : ℝ, (z = 2 * Real.sqrt 2 ∨ z = Real.sqrt 13 ∨ z = Real.sqrt 5) ∧
           (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3043_304358


namespace NUMINAMATH_CALUDE_average_income_of_M_and_N_l3043_304325

/-- Given the monthly incomes of three individuals M, N, and O, prove that the average income of M and N is 5050. -/
theorem average_income_of_M_and_N (M N O : ℕ) : 
  (N + O) / 2 = 6250 →
  (M + O) / 2 = 5200 →
  M = 4000 →
  (M + N) / 2 = 5050 := by
sorry

end NUMINAMATH_CALUDE_average_income_of_M_and_N_l3043_304325


namespace NUMINAMATH_CALUDE_line_inclination_through_origin_and_negative_one_l3043_304302

/-- The angle of inclination of a line passing through (0, 0) and (-1, -1) is 45°. -/
theorem line_inclination_through_origin_and_negative_one : ∃ (α : ℝ), 
  (∀ (x y : ℝ), y = x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) →
  α * (π / 180) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_through_origin_and_negative_one_l3043_304302


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l3043_304360

theorem gcd_lcm_problem (a b : ℕ) : 
  a > 0 → b > 0 → Nat.gcd a b = 45 → Nat.lcm a b = 1260 → a = 180 → b = 315 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l3043_304360


namespace NUMINAMATH_CALUDE_empire_state_height_is_443_l3043_304337

/-- The height of the Petronas Towers in meters -/
def petronas_height : ℝ := 452

/-- The height difference between the Empire State Building and the Petronas Towers in meters -/
def height_difference : ℝ := 9

/-- The height of the Empire State Building in meters -/
def empire_state_height : ℝ := petronas_height - height_difference

theorem empire_state_height_is_443 : empire_state_height = 443 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_height_is_443_l3043_304337


namespace NUMINAMATH_CALUDE_world_cup_tickets_equation_l3043_304396

/-- Represents the number of World Cup tickets reserved by a company -/
structure WorldCupTickets where
  groupStage : ℕ
  final : ℕ

/-- Represents the price of World Cup tickets in yuan -/
def ticketPrice : WorldCupTickets → ℕ
  | ⟨x, y⟩ => 2800 * x + 6400 * y

theorem world_cup_tickets_equation (t : WorldCupTickets) :
  (t.groupStage + t.final = 20 ∧ ticketPrice t = 74000) ↔
  (∃ x y : ℕ, x + y = 20 ∧ 2800 * x + 6400 * y = 74000 ∧ t = ⟨x, y⟩) :=
by sorry

end NUMINAMATH_CALUDE_world_cup_tickets_equation_l3043_304396


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l3043_304388

theorem cube_volume_from_diagonal (diagonal : ℝ) (h : diagonal = 6 * Real.sqrt 2) :
  ∃ (side : ℝ), side > 0 ∧ side^3 = 48 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l3043_304388


namespace NUMINAMATH_CALUDE_count_distinct_walls_l3043_304333

/-- The number of distinct walls that can be built with n identical cubes -/
def distinct_walls (n : ℕ+) : ℕ :=
  2^(n.val - 1)

/-- Theorem stating that the number of distinct walls with n cubes is 2^(n-1) -/
theorem count_distinct_walls (n : ℕ+) :
  distinct_walls n = 2^(n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_walls_l3043_304333


namespace NUMINAMATH_CALUDE_zoo_trip_buses_l3043_304372

theorem zoo_trip_buses (total_students : ℕ) (students_per_bus : ℕ) (car_students : ℕ) : 
  total_students = 375 → students_per_bus = 53 → car_students = 4 →
  ((total_students - car_students + students_per_bus - 1) / students_per_bus : ℕ) = 8 := by
sorry

end NUMINAMATH_CALUDE_zoo_trip_buses_l3043_304372


namespace NUMINAMATH_CALUDE_sine_cosine_transformation_l3043_304339

open Real

theorem sine_cosine_transformation (x : ℝ) :
  sin (2 * x) - Real.sqrt 3 * cos (2 * x) = 2 * sin (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_transformation_l3043_304339


namespace NUMINAMATH_CALUDE_abc_bad_theorem_l3043_304317

def is_valid_quadruple (A B C D : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ D ≠ 0 ∧ C ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (100 * A + 10 * B + C) * D = (100 * B + 10 * A + D) * C

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2,1,7,4), (1,2,4,7), (8,1,9,2), (1,8,2,9), (7,2,8,3), (2,7,3,8), (6,3,7,4), (3,6,4,7)}

theorem abc_bad_theorem :
  {q : ℕ × ℕ × ℕ × ℕ | is_valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_abc_bad_theorem_l3043_304317


namespace NUMINAMATH_CALUDE_parallelogram_area_l3043_304319

theorem parallelogram_area (base height : ℝ) (h1 : base = 24) (h2 : height = 10) :
  base * height = 240 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3043_304319


namespace NUMINAMATH_CALUDE_sequence_formulas_l3043_304380

def S (n : ℕ) : ℝ := sorry

def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 2 * S n + 1

def b (n : ℕ) : ℝ := (3 * n - 1) * a n

def T (n : ℕ) : ℝ := sorry

theorem sequence_formulas :
  (∀ n : ℕ, a n = 3^n) ∧
  (∀ n : ℕ, T n = ((3 * n / 2) - 5 / 4) * 3^n + 5 / 4) :=
sorry

end NUMINAMATH_CALUDE_sequence_formulas_l3043_304380


namespace NUMINAMATH_CALUDE_parabola_vertex_l3043_304306

/-- Given a quadratic function f(x) = -2x^2 + cx + d where the solution to f(x) ≤ 0 is [-7/2, ∞),
    the vertex of the parabola defined by f(x) is (-7/2, 0). -/
theorem parabola_vertex (c d : ℝ) :
  let f : ℝ → ℝ := λ x => -2 * x^2 + c * x + d
  (∀ x, f x ≤ 0 ↔ x ∈ Set.Ici (-7/2)) →
  ∃! v : ℝ × ℝ, v.1 = -7/2 ∧ v.2 = 0 ∧ ∀ x, f x ≤ f v.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3043_304306


namespace NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_second_15_l3043_304359

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_first_15 :
  sum_of_squares 15 = 1200 :=
by sorry

-- The condition about the second 15 integers is not directly used in the proof,
-- but we include it as a hypothesis to match the original problem
theorem sum_of_squares_second_15 :
  sum_of_squares 30 - sum_of_squares 15 = 8175 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_second_15_l3043_304359


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3043_304329

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3 - x) ↔ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3043_304329


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_and_gcd_l3043_304345

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_divisibility_and_gcd (m n : ℕ) :
  (m ∣ n → fib m ∣ fib n) ∧ (Nat.gcd (fib m) (fib n) = fib (Nat.gcd m n)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_and_gcd_l3043_304345


namespace NUMINAMATH_CALUDE_seed_germination_problem_l3043_304392

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 total_germination_rate : ℚ) :
  seeds_plot1 = 500 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 1/2 →
  total_germination_rate = 35714285714285715/100000000000000000 →
  (↑(seeds_plot1 * 3/10) + ↑(seeds_plot2 * germination_rate_plot2)) / 
   ↑(seeds_plot1 + seeds_plot2) = total_germination_rate :=
by sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l3043_304392


namespace NUMINAMATH_CALUDE_julio_twice_james_age_l3043_304350

/-- 
Given:
- Julio is currently 36 years old
- James is currently 11 years old

Prove that in 14 years, Julio's age will be twice James's age
-/
theorem julio_twice_james_age (julio_age : ℕ) (james_age : ℕ) (years : ℕ) : 
  julio_age = 36 → james_age = 11 → years = 14 → 
  julio_age + years = 2 * (james_age + years) := by
  sorry

end NUMINAMATH_CALUDE_julio_twice_james_age_l3043_304350


namespace NUMINAMATH_CALUDE_smallest_violet_balls_l3043_304328

theorem smallest_violet_balls (x : ℕ) (y : ℕ) : 
  x > 0 ∧ 
  x % 120 = 0 ∧ 
  x / 10 + x / 8 + x / 3 + (x / 10 + 9) + (x / 8 + 10) + 8 + y = x ∧
  y = x / 60 * 13 - 27 →
  y ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_smallest_violet_balls_l3043_304328


namespace NUMINAMATH_CALUDE_single_digit_between_4_and_9_less_than_6_l3043_304347

theorem single_digit_between_4_and_9_less_than_6 (n : ℕ) 
  (h1 : n ≤ 9)
  (h2 : 4 < n)
  (h3 : n < 9)
  (h4 : n < 6) : 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_single_digit_between_4_and_9_less_than_6_l3043_304347


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_max_l3043_304308

theorem quadratic_function_inequality_max (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≤ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    (∀ a b c : ℝ, b^2 / (a^2 + c^2) ≤ M) ∧
    (∃ a b c : ℝ, b^2 / (a^2 + c^2) = M)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_max_l3043_304308


namespace NUMINAMATH_CALUDE_other_sales_percentage_l3043_304381

/-- The percentage of sales for notebooks -/
def notebooks_sales : ℝ := 25

/-- The percentage of sales for markers -/
def markers_sales : ℝ := 40

/-- The total percentage of all sales -/
def total_sales : ℝ := 100

/-- Theorem: The percentage of sales that were neither notebooks nor markers is 35% -/
theorem other_sales_percentage : 
  total_sales - (notebooks_sales + markers_sales) = 35 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l3043_304381


namespace NUMINAMATH_CALUDE_platform_length_l3043_304365

/-- Given a train of length 300 meters that crosses a platform in 38 seconds
    and a signal pole in 18 seconds, prove that the length of the platform
    is approximately 333.46 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 38)
  (h3 : pole_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 333.46) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3043_304365


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l3043_304349

theorem five_topping_pizzas (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l3043_304349


namespace NUMINAMATH_CALUDE_fraction_simplification_l3043_304324

theorem fraction_simplification (a b c : ℝ) 
  (h1 : a + 2*b + 3*c ≠ 0) 
  (h2 : a^2 + 9*c^2 - 4*b^2 + 6*a*c ≠ 0) 
  (h3 : a - 2*b + 3*c ≠ 0) : 
  (a^2 + 4*b^2 - 9*c^2 + 4*a*b) / (a^2 + 9*c^2 - 4*b^2 + 6*a*c) = (a + 2*b - 3*c) / (a - 2*b + 3*c) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3043_304324


namespace NUMINAMATH_CALUDE_infinitely_many_powers_of_five_with_consecutive_zeros_l3043_304348

theorem infinitely_many_powers_of_five_with_consecutive_zeros : 
  ∀ k : ℕ, ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ m ∈ S, (5^m : ℕ) ≡ 1 [MOD 2^k]) ∧
  (∃ N : ℕ, ∀ n ≥ N, ∃ m ∈ S, 
    (∃ i : ℕ, (5^m : ℕ) / 10^i % 10^1976 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_powers_of_five_with_consecutive_zeros_l3043_304348


namespace NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l3043_304309

theorem hua_luogeng_birthday_factorization (h : 19101112 = 1163 * 16424) :
  Nat.Prime 1163 ∧ ¬ Nat.Prime 16424 := by
  sorry

end NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l3043_304309


namespace NUMINAMATH_CALUDE_english_only_enrollment_l3043_304362

/-- Represents the number of students in different enrollment categories -/
structure EnrollmentCount where
  total : ℕ
  bothSubjects : ℕ
  germanTotal : ℕ

/-- Calculates the number of students enrolled only in English -/
def studentsOnlyEnglish (e : EnrollmentCount) : ℕ :=
  e.total - e.germanTotal

/-- Theorem: Given the enrollment conditions, 28 students are enrolled only in English -/
theorem english_only_enrollment (e : EnrollmentCount) 
  (h1 : e.total = 50)
  (h2 : e.bothSubjects = 12)
  (h3 : e.germanTotal = 22)
  (h4 : e.total ≥ e.germanTotal) : 
  studentsOnlyEnglish e = 28 := by
  sorry

#check english_only_enrollment

end NUMINAMATH_CALUDE_english_only_enrollment_l3043_304362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3043_304355

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Number of terms in an arithmetic sequence -/
def num_terms (a₁ : ℚ) (d : ℚ) (l : ℚ) : ℕ :=
  Nat.floor ((l - a₁) / d + 1)

theorem arithmetic_sequence_ratio :
  let n₁ := num_terms 4 2 40
  let n₂ := num_terms 5 5 75
  let sum₁ := arithmetic_sum 4 2 n₁
  let sum₂ := arithmetic_sum 5 5 n₂
  sum₁ / sum₂ = 209 / 300 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3043_304355


namespace NUMINAMATH_CALUDE_triangle_area_l3043_304375

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a^2 + b^2 - c^2 = 6√3 - 2ab and C = 60°, then the area of triangle ABC is 3/2. -/
theorem triangle_area (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 6 * Real.sqrt 3 - 2*a*b) 
  (h2 : Real.cos (Real.pi / 3) = 1/2) : 
  (1/2) * a * b * Real.sin (Real.pi / 3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3043_304375


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3043_304364

/-- The inequality x^2 + ax - 2 < 0 has solutions within [2, 4] if and only if a ∈ (-∞, -1) -/
theorem inequality_solution_range (a : ℝ) :
  (∃ x ∈ Set.Icc 2 4, x^2 + a*x - 2 < 0) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3043_304364


namespace NUMINAMATH_CALUDE_luke_coin_count_l3043_304377

/-- Represents the number of coins in each pile of quarters --/
def quarter_piles : List Nat := [4, 4, 6, 6, 6, 8]

/-- Represents the number of coins in each pile of dimes --/
def dime_piles : List Nat := [3, 5, 2, 2]

/-- Represents the number of coins in each pile of nickels --/
def nickel_piles : List Nat := [5, 5, 5, 7, 7, 10]

/-- Represents the number of coins in each pile of pennies --/
def penny_piles : List Nat := [12, 8, 20]

/-- Represents the number of coins in each pile of half dollars --/
def half_dollar_piles : List Nat := [2, 4]

/-- The total number of coins Luke has --/
def total_coins : Nat := quarter_piles.sum + dime_piles.sum + nickel_piles.sum + 
                         penny_piles.sum + half_dollar_piles.sum

theorem luke_coin_count : total_coins = 131 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l3043_304377


namespace NUMINAMATH_CALUDE_max_value_product_l3043_304371

theorem max_value_product (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l3043_304371


namespace NUMINAMATH_CALUDE_intersection_theorem_l3043_304340

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x^2 < 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem :
  A_intersect_B = {x | -1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3043_304340


namespace NUMINAMATH_CALUDE_subtraction_result_l3043_304395

theorem subtraction_result : 3.56 - 2.15 = 1.41 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3043_304395


namespace NUMINAMATH_CALUDE_largest_fraction_l3043_304367

theorem largest_fraction : 
  let fractions := [5/12, 7/15, 23/45, 89/178, 199/400]
  ∀ x ∈ fractions, (23/45 : ℚ) ≥ x := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3043_304367


namespace NUMINAMATH_CALUDE_invalid_reasoning_l3043_304397

-- Define the types of reasoning
inductive ReasoningType
  | Analogy
  | Inductive
  | Deductive

-- Define the concept of valid reasoning
def isValidReasoning (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Analogy => true
  | ReasoningType.Inductive => true
  | ReasoningType.Deductive => true

-- Define the reasoning options
def optionA : ReasoningType := ReasoningType.Analogy
def optionB : ReasoningType := ReasoningType.Inductive
def optionC : ReasoningType := ReasoningType.Inductive
def optionD : ReasoningType := ReasoningType.Inductive

-- Theorem to prove
theorem invalid_reasoning :
  isValidReasoning optionA ∧
  isValidReasoning optionB ∧
  ¬(isValidReasoning optionC) ∧
  isValidReasoning optionD :=
by sorry

end NUMINAMATH_CALUDE_invalid_reasoning_l3043_304397


namespace NUMINAMATH_CALUDE_fraction_equality_l3043_304357

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x - 2*y) / (2*x + 3*y) = 1) : 
  (2*x - 5*y) / (5*x + 2*y) = -5/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3043_304357


namespace NUMINAMATH_CALUDE_sqrt_expression_zero_quadratic_equation_solutions_l3043_304383

-- Problem 1
theorem sqrt_expression_zero (a : ℝ) (ha : a > 0) :
  Real.sqrt (8 * a^3) - 4 * a^2 * Real.sqrt (1 / (8 * a)) - 2 * a * Real.sqrt (a / 2) = 0 := by
  sorry

-- Problem 2
theorem quadratic_equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 2 ∧
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_zero_quadratic_equation_solutions_l3043_304383


namespace NUMINAMATH_CALUDE_divisibility_proof_l3043_304361

theorem divisibility_proof (k : ℕ) (p : ℕ) (m : ℕ) 
  (h1 : k > 1) 
  (h2 : p = 6 * k + 1) 
  (h3 : Nat.Prime p) 
  (h4 : m = 2^p - 1) : 
  (127 * m) ∣ (2^(m-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3043_304361


namespace NUMINAMATH_CALUDE_pets_after_one_month_l3043_304343

/-- Calculates the number of pets in an animal shelter after one month --/
theorem pets_after_one_month
  (initial_dogs : ℕ)
  (initial_cats : ℕ)
  (initial_lizards : ℕ)
  (dog_adoption_rate : ℚ)
  (cat_adoption_rate : ℚ)
  (lizard_adoption_rate : ℚ)
  (new_pets : ℕ)
  (h_dogs : initial_dogs = 30)
  (h_cats : initial_cats = 28)
  (h_lizards : initial_lizards = 20)
  (h_dog_rate : dog_adoption_rate = 1/2)
  (h_cat_rate : cat_adoption_rate = 1/4)
  (h_lizard_rate : lizard_adoption_rate = 1/5)
  (h_new_pets : new_pets = 13) :
  ↑initial_dogs + ↑initial_cats + ↑initial_lizards -
  (↑initial_dogs * dog_adoption_rate + ↑initial_cats * cat_adoption_rate + ↑initial_lizards * lizard_adoption_rate) +
  ↑new_pets = 65 := by
  sorry


end NUMINAMATH_CALUDE_pets_after_one_month_l3043_304343


namespace NUMINAMATH_CALUDE_grandfather_grandson_age_ratio_not_six_l3043_304386

theorem grandfather_grandson_age_ratio_not_six : 
  let grandson_age_now : ℕ := 12
  let grandfather_age_now : ℕ := 72
  let grandson_age_three_years_ago : ℕ := grandson_age_now - 3
  let grandfather_age_three_years_ago : ℕ := grandfather_age_now - 3
  ¬ (grandfather_age_three_years_ago = 6 * grandson_age_three_years_ago) :=
by sorry

end NUMINAMATH_CALUDE_grandfather_grandson_age_ratio_not_six_l3043_304386


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l3043_304398

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l3043_304398


namespace NUMINAMATH_CALUDE_base7_312_equals_base4_2310_l3043_304310

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base7_312_equals_base4_2310 :
  base10ToBase4 (base7ToBase10 [2, 1, 3]) = [2, 3, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base7_312_equals_base4_2310_l3043_304310


namespace NUMINAMATH_CALUDE_robert_photos_count_l3043_304370

/-- The number of photos taken by Claire -/
def claire_photos : ℕ := 8

/-- The additional number of photos taken by Robert compared to Claire -/
def robert_extra_photos : ℕ := 16

/-- The number of photos taken by Robert -/
def robert_photos : ℕ := claire_photos + robert_extra_photos

/-- Theorem: Robert has taken 24 photos -/
theorem robert_photos_count : robert_photos = 24 := by
  sorry

end NUMINAMATH_CALUDE_robert_photos_count_l3043_304370


namespace NUMINAMATH_CALUDE_negation_equivalence_not_always_greater_product_quadratic_roots_condition_l3043_304338

-- Statement 1
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
sorry

-- Statement 2
theorem not_always_greater_product :
  ∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d :=
sorry

-- Statement 3
theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a - 3) * x + a = 0 ∧ y^2 + (a - 3) * y + a = 0) →
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_not_always_greater_product_quadratic_roots_condition_l3043_304338


namespace NUMINAMATH_CALUDE_jellybean_removal_l3043_304307

theorem jellybean_removal (initial : ℕ) (added_back : ℕ) (removed_after : ℕ) (final : ℕ) :
  initial = 37 ∧ added_back = 5 ∧ removed_after = 4 ∧ final = 23 →
  ∃ (removed : ℕ), initial - removed + added_back - removed_after = final ∧ removed = 15 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_removal_l3043_304307


namespace NUMINAMATH_CALUDE_living_space_growth_l3043_304382

/-- Represents the growth of average living space per person over two years -/
theorem living_space_growth (x : ℝ) : 
  (10 : ℝ) * (1 + x)^2 = 12.1 ↔ 
  (∃ (initial final : ℝ), 
    initial = 10 ∧ 
    final = 12.1 ∧ 
    final = initial * (1 + x)^2) :=
by sorry

end NUMINAMATH_CALUDE_living_space_growth_l3043_304382


namespace NUMINAMATH_CALUDE_two_roots_iff_a_gt_neg_one_l3043_304323

-- Define the equation as a function of x and a
def f (x a : ℝ) : ℝ := x^2 + 2*x + 2*|x + 1| - a

-- Define the property of having exactly two roots
def has_exactly_two_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x a = 0 ∧ f y a = 0 ∧ ∀ z : ℝ, f z a = 0 → z = x ∨ z = y

-- State the theorem
theorem two_roots_iff_a_gt_neg_one :
  ∀ a : ℝ, has_exactly_two_roots a ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_two_roots_iff_a_gt_neg_one_l3043_304323


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l3043_304385

def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  is_increasing_geometric_sequence a →
  a 1 + a 4 = 9 →
  a 2 * a 3 = 8 →
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l3043_304385


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3043_304366

/-- The line 4x + 3y + k = 0 is tangent to the parabola y² = 16x if and only if k = 9 -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4*x + 3*y + k = 0 → y^2 = 16*x) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3043_304366


namespace NUMINAMATH_CALUDE_wolf_nobel_count_l3043_304304

/-- Represents the number of scientists with different prize combinations -/
structure ScientistCounts where
  total : ℕ
  wolf : ℕ
  nobel : ℕ
  wolfNobel : ℕ

/-- The conditions of the workshop -/
def workshopConditions (s : ScientistCounts) : Prop :=
  s.total = 50 ∧
  s.wolf = 31 ∧
  s.nobel = 29 ∧
  s.total - s.wolf = (s.nobel - s.wolfNobel) + (s.total - s.wolf - (s.nobel - s.wolfNobel)) + 3

/-- The theorem stating that 18 Wolf Prize laureates were also Nobel Prize laureates -/
theorem wolf_nobel_count (s : ScientistCounts) :
  workshopConditions s → s.wolfNobel = 18 := by
  sorry

end NUMINAMATH_CALUDE_wolf_nobel_count_l3043_304304


namespace NUMINAMATH_CALUDE_square_side_length_average_l3043_304378

theorem square_side_length_average (a b c : ℝ) 
  (ha : a = 25) (hb : b = 64) (hc : c = 225) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3043_304378


namespace NUMINAMATH_CALUDE_star_properties_l3043_304387

-- Define the * operation for rational numbers
def star (a b : ℚ) : ℚ := (a + b) - abs (b - a)

-- Theorem statement
theorem star_properties :
  (star (-3) 2 = -6) ∧ (star (star 4 3) (-5) = -10) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l3043_304387
