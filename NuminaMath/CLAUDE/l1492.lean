import Mathlib

namespace NUMINAMATH_CALUDE_garden_yield_mr_green_garden_yield_l1492_149287

/-- Calculates the expected potato yield from a rectangular garden after applying fertilizer -/
theorem garden_yield (length_steps width_steps feet_per_step : ℕ) 
  (original_yield_per_sqft : ℚ) (yield_increase_percent : ℕ) : ℚ :=
  let length_feet := length_steps * feet_per_step
  let width_feet := width_steps * feet_per_step
  let area := length_feet * width_feet
  let original_yield := area * original_yield_per_sqft
  let yield_increase_factor := 1 + yield_increase_percent / 100
  original_yield * yield_increase_factor

/-- Proves that Mr. Green's garden will yield 2227.5 pounds of potatoes after fertilizer -/
theorem mr_green_garden_yield :
  garden_yield 18 25 3 (1/2) 10 = 2227.5 := by
  sorry

end NUMINAMATH_CALUDE_garden_yield_mr_green_garden_yield_l1492_149287


namespace NUMINAMATH_CALUDE_square_sequence_formulas_l1492_149204

/-- The number of squares in the nth figure of the sequence -/
def num_squares (n : ℕ) : ℕ := 2 * n^2 - 2 * n + 1

/-- The first formula: (2n-1)^2 - 4 * (n(n-1)/2) -/
def formula_a (n : ℕ) : ℕ := (2 * n - 1)^2 - 2 * n * (n - 1)

/-- The third formula: 1 + (1 + 2 + ... + (n-1)) * 4 -/
def formula_c (n : ℕ) : ℕ := 1 + 2 * n * (n - 1)

/-- The fourth formula: (n-1)^2 + n^2 -/
def formula_d (n : ℕ) : ℕ := (n - 1)^2 + n^2

theorem square_sequence_formulas (n : ℕ) : 
  n > 0 → num_squares n = formula_a n ∧ num_squares n = formula_c n ∧ num_squares n = formula_d n :=
by sorry

end NUMINAMATH_CALUDE_square_sequence_formulas_l1492_149204


namespace NUMINAMATH_CALUDE_v_2002_equals_4_l1492_149283

-- Define the function g
def g : ℕ → ℕ
| 1 => 2
| 2 => 3
| 3 => 5
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

-- Theorem to prove
theorem v_2002_equals_4 : v 2002 = 4 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_4_l1492_149283


namespace NUMINAMATH_CALUDE_initial_white_lights_equal_total_colored_lights_l1492_149291

/-- The number of white lights Malcolm had initially -/
def initialWhiteLights : ℕ := sorry

/-- The number of red lights Malcolm bought -/
def redLights : ℕ := 12

/-- The number of blue lights Malcolm bought -/
def blueLights : ℕ := 3 * redLights

/-- The number of green lights Malcolm bought -/
def greenLights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy -/
def remainingLights : ℕ := 5

/-- Theorem stating that the initial number of white lights equals the total number of colored lights -/
theorem initial_white_lights_equal_total_colored_lights :
  initialWhiteLights = redLights + blueLights + greenLights + remainingLights := by sorry

end NUMINAMATH_CALUDE_initial_white_lights_equal_total_colored_lights_l1492_149291


namespace NUMINAMATH_CALUDE_polynomial_factor_l1492_149290

/-- The polynomial P(x) = x^3 - 3x^2 + cx - 8 -/
def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + c*x - 8

theorem polynomial_factor (c : ℝ) : 
  (∀ x, P c x = 0 ↔ (x + 2 = 0 ∨ ∃ q, P c x = (x + 2) * q)) → c = -14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1492_149290


namespace NUMINAMATH_CALUDE_ratio_w_y_l1492_149234

-- Define the ratios
def ratio_w_x : ℚ := 5 / 4
def ratio_y_z : ℚ := 7 / 5
def ratio_z_x : ℚ := 1 / 8

-- Theorem statement
theorem ratio_w_y (w x y z : ℚ) 
  (hw : w / x = ratio_w_x)
  (hy : y / z = ratio_y_z)
  (hz : z / x = ratio_z_x) : 
  w / y = 25 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_y_l1492_149234


namespace NUMINAMATH_CALUDE_first_three_decimal_digits_l1492_149253

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → 
  x = (10^n + 1)^(11/7) → 
  ∃ (k : ℕ), x = k + 0.571 + r ∧ 0 ≤ r ∧ r < 0.001 :=
sorry

end NUMINAMATH_CALUDE_first_three_decimal_digits_l1492_149253


namespace NUMINAMATH_CALUDE_incircle_radius_altitude_ratio_l1492_149257

/-- An isosceles right triangle with inscribed circle -/
structure IsoscelesRightTriangle where
  -- Side length of the equal sides
  side : ℝ
  -- Radius of the inscribed circle
  incircle_radius : ℝ
  -- Altitude to the hypotenuse
  altitude : ℝ
  -- The triangle is isosceles and right-angled
  is_isosceles : side = altitude * Real.sqrt 2
  -- Relationship between incircle radius and altitude
  radius_altitude_relation : incircle_radius = altitude * (Real.sqrt 2 - 1)

/-- The ratio of the inscribed circle radius to the altitude in an isosceles right triangle is √2 - 1 -/
theorem incircle_radius_altitude_ratio (t : IsoscelesRightTriangle) :
  t.incircle_radius / t.altitude = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_incircle_radius_altitude_ratio_l1492_149257


namespace NUMINAMATH_CALUDE_torn_sheets_count_l1492_149289

/-- Represents a book with numbered pages -/
structure Book where
  /-- The last page number in the book -/
  lastPage : ℕ

/-- Represents a range of torn out pages -/
structure TornPages where
  /-- The first torn out page number -/
  first : ℕ
  /-- The last torn out page number -/
  last : ℕ

/-- Check if two numbers have the same digits in any order -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Calculate the number of sheets torn out -/
def sheetsTornOut (book : Book) (torn : TornPages) : ℕ :=
  (torn.last - torn.first + 1) / 2

/-- The main theorem -/
theorem torn_sheets_count (book : Book) (torn : TornPages) :
  torn.first = 185 →
  sameDigits torn.first torn.last →
  torn.last % 2 = 0 →
  torn.first < torn.last →
  sheetsTornOut book torn = 167 := by sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l1492_149289


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1492_149218

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b = 4 → x + 2*y ≤ a + 2*b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1492_149218


namespace NUMINAMATH_CALUDE_unique_polynomial_composition_l1492_149219

theorem unique_polynomial_composition (a b c : ℝ) (n : ℕ) (h : a ≠ 0) :
  ∃! Q : Polynomial ℝ,
    (Polynomial.degree Q = n) ∧
    (∀ x : ℝ, Q.eval (a * x^2 + b * x + c) = a * (Q.eval x)^2 + b * (Q.eval x) + c) := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_composition_l1492_149219


namespace NUMINAMATH_CALUDE_depth_is_finite_x_5776_mod_6_x_diff_mod_2016_l1492_149256

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Depth of a number -/
def depth (N : ℕ) : ℕ := sorry

/-- Minimal positive integer with depth n -/
def x (n : ℕ) : ℕ := sorry

theorem depth_is_finite (N : ℕ) : ∃ k : ℕ, S^[k] N < 10 := by sorry

theorem x_5776_mod_6 : x 5776 % 6 = 4 := by sorry

theorem x_diff_mod_2016 : (x 5776 - x 5708) % 2016 = 0 := by sorry

end NUMINAMATH_CALUDE_depth_is_finite_x_5776_mod_6_x_diff_mod_2016_l1492_149256


namespace NUMINAMATH_CALUDE_intersection_circles_sum_l1492_149294

/-- Given two circles intersecting at (2,3) and (m,2), with centers on the line x+y+n=0, prove m+n = -2 -/
theorem intersection_circles_sum (m n : ℝ) : 
  (∃ (c₁ c₂ : ℝ × ℝ), 
    (c₁.1 + c₁.2 + n = 0) ∧ 
    (c₂.1 + c₂.2 + n = 0) ∧
    ((2 - c₁.1)^2 + (3 - c₁.2)^2 = (2 - c₂.1)^2 + (3 - c₂.2)^2) ∧
    ((m - c₁.1)^2 + (2 - c₁.2)^2 = (m - c₂.1)^2 + (2 - c₂.2)^2) ∧
    ((2 - c₁.1)^2 + (3 - c₁.2)^2 = (m - c₁.1)^2 + (2 - c₁.2)^2) ∧
    ((2 - c₂.1)^2 + (3 - c₂.2)^2 = (m - c₂.1)^2 + (2 - c₂.2)^2)) →
  m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_circles_sum_l1492_149294


namespace NUMINAMATH_CALUDE_mixture_ratio_l1492_149262

/-- Given two solutions A and B with different alcohol-water ratios, 
    prove that mixing them in a specific ratio results in a mixture with 60% alcohol. -/
theorem mixture_ratio (V_A V_B : ℝ) : 
  V_A > 0 → V_B > 0 →
  (21 / 25 * V_A + 2 / 5 * V_B) / (V_A + V_B) = 3 / 5 →
  V_A / V_B = 5 / 6 := by
sorry

/-- The ratio of Solution A to Solution B in the mixture -/
def solution_ratio : ℚ := 5 / 6

#check mixture_ratio
#check solution_ratio

end NUMINAMATH_CALUDE_mixture_ratio_l1492_149262


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1492_149247

theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) ↔ m = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1492_149247


namespace NUMINAMATH_CALUDE_bamboo_with_nine_nodes_l1492_149252

/-- Given a geometric sequence of 9 terms, prove that if the product of the first 3 terms is 3
    and the product of the last 3 terms is 9, then the 5th term is √3. -/
theorem bamboo_with_nine_nodes (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  a 1 * a 2 * a 3 = 3 →             -- Product of first 3 terms
  a 7 * a 8 * a 9 = 9 →             -- Product of last 3 terms
  a 5 = Real.sqrt 3 :=              -- 5th term is √3
by sorry

end NUMINAMATH_CALUDE_bamboo_with_nine_nodes_l1492_149252


namespace NUMINAMATH_CALUDE_jeans_business_weekly_hours_l1492_149285

/-- Represents the operating hours of a business for a single day -/
structure DailyHours where
  open_time : Nat
  close_time : Nat

/-- Calculates the number of hours a business is open in a day -/
def hours_open (dh : DailyHours) : Nat :=
  dh.close_time - dh.open_time

/-- Represents the operating hours of a business for a week -/
structure WeeklyHours where
  weekday : DailyHours
  weekend : DailyHours

/-- Calculates the total hours a business is open in a week -/
def total_weekly_hours (wh : WeeklyHours) : Nat :=
  (hours_open wh.weekday * 5) + (hours_open wh.weekend * 2)

/-- Jean's business hours -/
def jeans_business : WeeklyHours :=
  { weekday := { open_time := 16, close_time := 22 }
    weekend := { open_time := 18, close_time := 22 } }

theorem jeans_business_weekly_hours :
  total_weekly_hours jeans_business = 38 := by
  sorry

end NUMINAMATH_CALUDE_jeans_business_weekly_hours_l1492_149285


namespace NUMINAMATH_CALUDE_dvd_cd_ratio_l1492_149245

theorem dvd_cd_ratio (total : ℕ) (dvds : ℕ) (h1 : total = 273) (h2 : dvds = 168) :
  (dvds : ℚ) / (total - dvds : ℚ) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_dvd_cd_ratio_l1492_149245


namespace NUMINAMATH_CALUDE_chord_length_l1492_149275

/-- The length of the chord cut by a line on a circle in polar coordinates -/
theorem chord_length (ρ θ : ℝ) : 
  (ρ * Real.cos θ = 1/2) →  -- Line equation
  (ρ = 2 * Real.cos θ) →    -- Circle equation
  ∃ (chord_length : ℝ), chord_length = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_chord_length_l1492_149275


namespace NUMINAMATH_CALUDE_function_change_proof_l1492_149206

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The initial x value -/
def x_initial : ℝ := 2

/-- The final x value -/
def x_final : ℝ := 2.5

/-- The change in x -/
def delta_x : ℝ := x_final - x_initial

theorem function_change_proof :
  f x_final - f x_initial = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_function_change_proof_l1492_149206


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1492_149281

theorem rectangle_area_increase :
  ∀ (l w : ℝ), l > 0 → w > 0 →
  let new_length := 1.25 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.4375 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1492_149281


namespace NUMINAMATH_CALUDE_reflection_theorem_l1492_149210

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := (p.1, p.2 - 2)
  let p_reflected := (p_translated.2, p_translated.1)
  (p_reflected.1, p_reflected.2 + 2)

/-- The triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {(3, 4), (6, 8), (5, 1)}

theorem reflection_theorem :
  let A : ℝ × ℝ := (3, 4)
  let A' := reflect_x A
  let A'' := reflect_line A'
  A'' = (-6, 5) :=
by
  sorry

end NUMINAMATH_CALUDE_reflection_theorem_l1492_149210


namespace NUMINAMATH_CALUDE_g_of_three_l1492_149229

theorem g_of_three (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3*x - 2) = 4*x + 1) : g 3 = 23/3 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_l1492_149229


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_l1492_149296

theorem expression_nonnegative_iff (x : ℝ) : 
  (3*x - 12*x^2 + 48*x^3) / (27 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_l1492_149296


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1492_149249

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1492_149249


namespace NUMINAMATH_CALUDE_circular_garden_radius_l1492_149286

theorem circular_garden_radius (r : ℝ) (h : r > 0) :
  2 * Real.pi * r = (1 / 4) * Real.pi * r^2 → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l1492_149286


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1492_149284

-- Define the sphere and its properties
def sphere_radius : ℝ := 13
def water_cross_section_radius : ℝ := 12
def submerged_depth : ℝ := 8

-- Theorem statement
theorem sphere_surface_area :
  (sphere_radius ^ 2 = water_cross_section_radius ^ 2 + (sphere_radius - submerged_depth) ^ 2) →
  (4 * π * sphere_radius ^ 2 = 676 * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1492_149284


namespace NUMINAMATH_CALUDE_arithmetic_sum_l1492_149279

theorem arithmetic_sum : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l1492_149279


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1492_149207

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def are_vertical_angles (α β : Angle) : Prop := sorry

/-- Two angles are congruent if they have the same measure. -/
def are_congruent (α β : Angle) : Prop := sorry

/-- If two angles are vertical angles, then these two angles are congruent. -/
theorem vertical_angles_are_congruent (α β : Angle) : 
  are_vertical_angles α β → are_congruent α β := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1492_149207


namespace NUMINAMATH_CALUDE_valid_probability_is_one_fourteenth_l1492_149244

/-- Represents a bead color -/
inductive BeadColor
| Red
| White
| Blue

/-- Represents a configuration of beads -/
def BeadConfiguration := List BeadColor

/-- Checks if a configuration has no adjacent beads of the same color -/
def noAdjacentSameColor (config : BeadConfiguration) : Bool :=
  sorry

/-- Generates all possible bead configurations -/
def allConfigurations : List BeadConfiguration :=
  sorry

/-- Counts the number of valid configurations -/
def countValidConfigurations : Nat :=
  sorry

/-- The total number of possible configurations -/
def totalConfigurations : Nat := 420

/-- The probability of a valid configuration -/
def validProbability : ℚ :=
  sorry

theorem valid_probability_is_one_fourteenth :
  validProbability = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_valid_probability_is_one_fourteenth_l1492_149244


namespace NUMINAMATH_CALUDE_num_non_congruent_triangles_l1492_149241

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The set of points in the 3x3 grid -/
def gridPoints : List Point := [
  ⟨0, 0⟩, ⟨0.5, 0⟩, ⟨1, 0⟩,
  ⟨0, 0.5⟩, ⟨0.5, 0.5⟩, ⟨1, 0.5⟩,
  ⟨0, 1⟩, ⟨0.5, 1⟩, ⟨1, 1⟩
]

/-- Predicate to check if two triangles are congruent -/
def areCongruent (t1 t2 : Triangle) : Prop := sorry

/-- The set of all possible triangles formed from the grid points -/
def allTriangles : List Triangle := sorry

/-- The set of non-congruent triangles -/
def nonCongruentTriangles : List Triangle := sorry

/-- Theorem: The number of non-congruent triangles is 3 -/
theorem num_non_congruent_triangles : 
  nonCongruentTriangles.length = 3 := by sorry

end NUMINAMATH_CALUDE_num_non_congruent_triangles_l1492_149241


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1492_149248

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1492_149248


namespace NUMINAMATH_CALUDE_box_volume_l1492_149266

/-- A rectangular box with specific proportions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * width = 0.5 * (length * height)
  top_one_half_side : length * height = 1.5 * (width * height)
  side_area : width * height = 200

/-- The volume of a box is the product of its length, width, and height -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- Theorem stating that a box with the given proportions has a volume of 3000 -/
theorem box_volume (b : Box) : volume b = 3000 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l1492_149266


namespace NUMINAMATH_CALUDE_log_1458_between_consecutive_integers_l1492_149212

theorem log_1458_between_consecutive_integers (c d : ℤ) : 
  (c : ℝ) < Real.log 1458 / Real.log 10 ∧ 
  Real.log 1458 / Real.log 10 < (d : ℝ) ∧ 
  d = c + 1 → 
  c + d = 7 := by
sorry

end NUMINAMATH_CALUDE_log_1458_between_consecutive_integers_l1492_149212


namespace NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l1492_149228

/-- Converts a binary number (represented as a list of bits) to a decimal number -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to an octal number (represented as a list of digits) -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_101101110_equals_octal_556 :
  decimal_to_octal (binary_to_decimal [0, 1, 1, 1, 0, 1, 1, 0, 1]) = [5, 5, 6] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l1492_149228


namespace NUMINAMATH_CALUDE_bird_count_l1492_149277

theorem bird_count (cardinals bluebirds swallows : ℕ) : 
  cardinals = 3 * bluebirds ∧ 
  swallows = bluebirds / 2 ∧ 
  swallows = 2 → 
  cardinals + bluebirds + swallows = 18 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l1492_149277


namespace NUMINAMATH_CALUDE_solve_system_l1492_149246

theorem solve_system (x y : ℝ) (eq1 : x + y = 15) (eq2 : x - y = 5) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1492_149246


namespace NUMINAMATH_CALUDE_job_completion_time_l1492_149230

-- Define the problem parameters
def initial_workers : ℕ := 6
def initial_days : ℕ := 8
def days_before_joining : ℕ := 3
def additional_workers : ℕ := 4

-- Define the total work as a fraction
def total_work : ℚ := 1

-- Define the work rate of one worker per day
def work_rate_per_worker : ℚ := 1 / (initial_workers * initial_days)

-- Define the work completed in the first 3 days
def work_completed_first_phase : ℚ := initial_workers * work_rate_per_worker * days_before_joining

-- Define the remaining work
def remaining_work : ℚ := total_work - work_completed_first_phase

-- Define the total number of workers after joining
def total_workers : ℕ := initial_workers + additional_workers

-- Define the work rate of all workers after joining
def work_rate_after_joining : ℚ := total_workers * work_rate_per_worker

-- State the theorem
theorem job_completion_time :
  ∃ (remaining_days : ℕ), 
    (days_before_joining : ℚ) + remaining_days = 6 ∧
    remaining_work = work_rate_after_joining * remaining_days :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1492_149230


namespace NUMINAMATH_CALUDE_inequalities_hold_l1492_149227

theorem inequalities_hold (a b : ℝ) (h : a > b) :
  (∀ c : ℝ, c ≠ 0 → a / c^2 > b / c^2) ∧
  (∀ c : ℝ, a * |c| ≥ b * |c|) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1492_149227


namespace NUMINAMATH_CALUDE_simplify_powers_l1492_149243

theorem simplify_powers (a : ℝ) : (a^5 * a^3) * (a^2)^4 = a^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_powers_l1492_149243


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_180_l1492_149251

theorem distinct_prime_factors_of_180 : Nat.card (Nat.factors 180).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_180_l1492_149251


namespace NUMINAMATH_CALUDE_multiply_by_one_seventh_squared_l1492_149226

theorem multiply_by_one_seventh_squared (x : ℝ) : x * (1/7)^2 = 7^3 ↔ x = 16807 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_one_seventh_squared_l1492_149226


namespace NUMINAMATH_CALUDE_original_group_size_l1492_149235

/-- Proves that the original number of men in a group is 22, given the conditions of the problem. -/
theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 20 → absent_men = 2 → final_days = 22 → 
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧ 
    original_men = 22 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l1492_149235


namespace NUMINAMATH_CALUDE_manuscript_fee_calculation_l1492_149203

def tax_rate_1 : ℚ := 14 / 100
def tax_rate_2 : ℚ := 11 / 100
def tax_threshold_1 : ℕ := 800
def tax_threshold_2 : ℕ := 4000
def tax_paid : ℕ := 420

theorem manuscript_fee_calculation (fee : ℕ) : 
  (tax_threshold_1 < fee ∧ fee ≤ tax_threshold_2 ∧ 
   (fee - tax_threshold_1) * tax_rate_1 = tax_paid) → 
  fee = 3800 :=
by sorry

end NUMINAMATH_CALUDE_manuscript_fee_calculation_l1492_149203


namespace NUMINAMATH_CALUDE_seventh_term_is_five_l1492_149271

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a_3_eq_1 : a 3 = 1
  a_11_eq_25 : a 11 = 25

/-- The 7th term of the geometric sequence is 5 -/
theorem seventh_term_is_five (seq : GeometricSequence) : seq.a 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_five_l1492_149271


namespace NUMINAMATH_CALUDE_sheila_saves_for_four_years_l1492_149225

/-- Calculates the number of years Sheila plans to save. -/
def sheilas_savings_years (initial_savings : ℕ) (monthly_savings : ℕ) (family_addition : ℕ) (final_amount : ℕ) : ℕ :=
  ((final_amount - family_addition - initial_savings) / monthly_savings) / 12

/-- Theorem stating that Sheila plans to save for 4 years. -/
theorem sheila_saves_for_four_years :
  sheilas_savings_years 3000 276 7000 23248 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sheila_saves_for_four_years_l1492_149225


namespace NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l1492_149211

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 4 20 = 78 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l1492_149211


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_9_64_l1492_149209

/-- The number of ways to arrange k heads in n + 1 positions without consecutive heads -/
def arrange_heads (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of sequences without two consecutive heads in 10 coin tosses -/
def total_favorable_sequences : ℕ :=
  arrange_heads 10 0 + arrange_heads 9 1 + arrange_heads 8 2 +
  arrange_heads 7 3 + arrange_heads 6 4 + arrange_heads 5 5

/-- The total number of possible outcomes when tossing a coin 10 times -/
def total_outcomes : ℕ := 2^10

/-- The probability of no two consecutive heads in 10 coin tosses -/
def prob_no_consecutive_heads : ℚ := total_favorable_sequences / total_outcomes

theorem prob_no_consecutive_heads_is_9_64 :
  prob_no_consecutive_heads = 9/64 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_9_64_l1492_149209


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1492_149261

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x + 3

-- State the theorem
theorem quadratic_minimum :
  ∃ (x_min y_min : ℝ), x_min = -1 ∧ y_min = 1 ∧
  ∀ x, f x ≥ f x_min := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1492_149261


namespace NUMINAMATH_CALUDE_parabola_perpendicular_point_range_l1492_149258

/-- Given points A, B, C where B and C are on a parabola and AB is perpendicular to BC,
    the y-coordinate of C satisfies y ≤ 0 or y ≥ 4 -/
theorem parabola_perpendicular_point_range 
  (A B C : ℝ × ℝ)
  (h_A : A = (0, 2))
  (h_B : B.1 = B.2^2 - 4)
  (h_C : C.1 = C.2^2 - 4)
  (h_perp : (B.2 - 2) * (C.2 - B.2) = -(B.1 - 0) * (C.1 - B.1)) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_point_range_l1492_149258


namespace NUMINAMATH_CALUDE_shadow_length_sequence_l1492_149217

/-- Represents the position of a person relative to a street lamp -/
inductive Position
  | Before
  | Under
  | After

/-- Represents the length of a shadow -/
inductive ShadowLength
  | Long
  | Short

/-- A street lamp as a fixed light source -/
structure StreetLamp where
  position : ℝ × ℝ  -- (x, y) coordinates
  height : ℝ

/-- A person walking past a street lamp -/
structure Person where
  height : ℝ

/-- Calculates the shadow length based on the person's position relative to the lamp -/
def shadowLength (lamp : StreetLamp) (person : Person) (pos : Position) : ShadowLength :=
  sorry

/-- Theorem stating how the shadow length changes as a person walks under a street lamp -/
theorem shadow_length_sequence (lamp : StreetLamp) (person : Person) :
  shadowLength lamp person Position.Before = ShadowLength.Long ∧
  shadowLength lamp person Position.Under = ShadowLength.Short ∧
  shadowLength lamp person Position.After = ShadowLength.Long :=
sorry

end NUMINAMATH_CALUDE_shadow_length_sequence_l1492_149217


namespace NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_150_percent_l1492_149263

theorem increase_by_percentage (x : ℝ) (p : ℝ) :
  x + x * (p / 100) = x * (1 + p / 100) := by sorry

theorem seventy_five_increased_by_150_percent :
  75 + 75 * (150 / 100) = 187.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_150_percent_l1492_149263


namespace NUMINAMATH_CALUDE_objective_function_range_l1492_149221

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y > 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x + y

-- Theorem statement
theorem objective_function_range :
  ∃ (min max : ℝ), min = 1 ∧ max = 6 ∧
  (∀ x y : ℝ, FeasibleRegion x y →
    min ≤ ObjectiveFunction x y ∧ ObjectiveFunction x y ≤ max) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    FeasibleRegion x1 y1 ∧ FeasibleRegion x2 y2 ∧
    ObjectiveFunction x1 y1 = min ∧ ObjectiveFunction x2 y2 = max) :=
by
  sorry

end NUMINAMATH_CALUDE_objective_function_range_l1492_149221


namespace NUMINAMATH_CALUDE_game_result_l1492_149267

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 5 = 0 then 7
  else if n % 3 = 0 then 3
  else 0

def charlie_rolls : List ℕ := [6, 2, 3, 5]
def dana_rolls : List ℕ := [5, 3, 1, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points charlie_rolls * total_points dana_rolls = 36 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1492_149267


namespace NUMINAMATH_CALUDE_rotation_150_degrees_l1492_149280

-- Define the shapes
inductive Shape
  | Square
  | Triangle
  | Pentagon

-- Define the positions
inductive Position
  | Top
  | Right
  | Bottom

-- Define the circular arrangement
structure CircularArrangement :=
  (top : Shape)
  (right : Shape)
  (bottom : Shape)

-- Define the rotation function
def rotate150 (arr : CircularArrangement) : CircularArrangement :=
  { top := arr.right
  , right := arr.bottom
  , bottom := arr.top }

-- Theorem statement
theorem rotation_150_degrees (initial : CircularArrangement) 
  (h1 : initial.top = Shape.Square)
  (h2 : initial.right = Shape.Triangle)
  (h3 : initial.bottom = Shape.Pentagon) :
  let final := rotate150 initial
  final.top = Shape.Pentagon ∧ 
  final.right = Shape.Square ∧ 
  final.bottom = Shape.Triangle := by
  sorry

end NUMINAMATH_CALUDE_rotation_150_degrees_l1492_149280


namespace NUMINAMATH_CALUDE_greatest_integer_a_for_quadratic_l1492_149240

theorem greatest_integer_a_for_quadratic : 
  ∃ (a : ℤ), a = 6 ∧ 
  (∀ x : ℝ, x^2 + a*x + 9 ≠ -2) ∧
  (∀ b : ℤ, b > a → ∃ x : ℝ, x^2 + b*x + 9 = -2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_a_for_quadratic_l1492_149240


namespace NUMINAMATH_CALUDE_flour_requirement_undetermined_l1492_149222

/-- Represents the recipe requirements and current state of baking --/
structure BakingScenario where
  sugar_required : ℕ
  sugar_added : ℕ
  flour_added : ℕ

/-- Represents the unknown total flour required by the recipe --/
def total_flour_required : ℕ → Prop := fun _ => True

/-- Theorem stating that the total flour required cannot be determined --/
theorem flour_requirement_undetermined (scenario : BakingScenario) 
  (h1 : scenario.sugar_required = 11)
  (h2 : scenario.sugar_added = 10)
  (h3 : scenario.flour_added = 12) :
  ∀ n : ℕ, total_flour_required n :=
by sorry

end NUMINAMATH_CALUDE_flour_requirement_undetermined_l1492_149222


namespace NUMINAMATH_CALUDE_max_distance_between_spheres_max_distance_achieved_l1492_149242

def sphere1_center : ℝ × ℝ × ℝ := (-4, -10, 5)
def sphere1_radius : ℝ := 20

def sphere2_center : ℝ × ℝ × ℝ := (10, 7, -16)
def sphere2_radius : ℝ := 90

theorem max_distance_between_spheres :
  ∀ (p1 p2 : ℝ × ℝ × ℝ),
  ‖p1 - sphere1_center‖ = sphere1_radius →
  ‖p2 - sphere2_center‖ = sphere2_radius →
  ‖p1 - p2‖ ≤ 140.433 :=
by sorry

theorem max_distance_achieved :
  ∃ (p1 p2 : ℝ × ℝ × ℝ),
  ‖p1 - sphere1_center‖ = sphere1_radius ∧
  ‖p2 - sphere2_center‖ = sphere2_radius ∧
  ‖p1 - p2‖ = 140.433 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_spheres_max_distance_achieved_l1492_149242


namespace NUMINAMATH_CALUDE_aaron_earnings_l1492_149236

/-- Represents the work hours for each day of the week -/
structure WorkHours :=
  (monday : Real)
  (tuesday : Real)
  (wednesday : Real)
  (friday : Real)

/-- Calculates the total earnings for the week given work hours and hourly rate -/
def calculateEarnings (hours : WorkHours) (hourlyRate : Real) : Real :=
  (hours.monday + hours.tuesday + hours.wednesday + hours.friday) * hourlyRate

/-- Theorem stating that Aaron's earnings for the week are $38.75 -/
theorem aaron_earnings :
  let hours : WorkHours := {
    monday := 2,
    tuesday := 1.25,
    wednesday := 2.833,
    friday := 0.667
  }
  let hourlyRate : Real := 5
  calculateEarnings hours hourlyRate = 38.75 := by
  sorry

#check aaron_earnings

end NUMINAMATH_CALUDE_aaron_earnings_l1492_149236


namespace NUMINAMATH_CALUDE_equation_solutions_l1492_149292

/-- Definition of matrix expression -/
def matrix_expr (a b c d : ℝ) : ℝ := a * b - c * d

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  matrix_expr (3 * x) (2 * x + 1) 1 (2 * x) = 5

/-- Theorem stating the solutions of the equation -/
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 5/6 ∧ equation x₁ ∧ equation x₂ ∧
  ∀ (x : ℝ), equation x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1492_149292


namespace NUMINAMATH_CALUDE_parabola_equation_l1492_149272

/-- A parabola is a set of points equidistant from a fixed point (focus) and a fixed line (directrix) -/
def Parabola (focus : ℝ × ℝ) (directrix : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p focus = |p.2 - directrix p.1|}

theorem parabola_equation (p : ℝ × ℝ) :
  p ∈ Parabola (0, -1) (fun _ ↦ 1) ↔ p.1^2 = -4 * p.2 := by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l1492_149272


namespace NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l1492_149254

theorem abs_one_point_five_minus_sqrt_two :
  |1.5 - Real.sqrt 2| = 1.5 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l1492_149254


namespace NUMINAMATH_CALUDE_guest_lecturer_fee_l1492_149274

theorem guest_lecturer_fee (B : Nat) (h1 : B < 10) (h2 : (200 + 10 * B + 9) % 13 = 0) : B = 0 := by
  sorry

end NUMINAMATH_CALUDE_guest_lecturer_fee_l1492_149274


namespace NUMINAMATH_CALUDE_sequence_length_l1492_149299

/-- The sequence defined by a(n) = 2 + 5(n-1) for n ≥ 1 -/
def a : ℕ → ℕ := λ n => 2 + 5 * (n - 1)

/-- The last term of the sequence -/
def last_term : ℕ := 57

theorem sequence_length :
  ∃ n : ℕ, n > 0 ∧ a n = last_term ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_sequence_length_l1492_149299


namespace NUMINAMATH_CALUDE_gold_rod_weight_sum_l1492_149216

theorem gold_rod_weight_sum (a : Fin 5 → ℝ) :
  (∀ i j : Fin 5, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
  a 0 = 4 →                                            -- first term is 4
  a 4 = 2 →                                            -- last term is 2
  a 1 + a 3 = 6 :=                                     -- sum of second and fourth terms is 6
by sorry

end NUMINAMATH_CALUDE_gold_rod_weight_sum_l1492_149216


namespace NUMINAMATH_CALUDE_milk_bottle_recycling_l1492_149201

theorem milk_bottle_recycling (marcus_bottles john_bottles : ℕ) 
  (h1 : marcus_bottles = 25) 
  (h2 : john_bottles = 20) : 
  marcus_bottles + john_bottles = 45 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottle_recycling_l1492_149201


namespace NUMINAMATH_CALUDE_rectangle_area_is_72_l1492_149255

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two corner points -/
structure Rectangle where
  topLeft : ℝ × ℝ
  bottomRight : ℝ × ℝ

def circleP : Circle := { center := (0, 3), radius := 3 }
def circleQ : Circle := { center := (3, 3), radius := 3 }
def circleR : Circle := { center := (6, 3), radius := 3 }
def circleS : Circle := { center := (9, 3), radius := 3 }

def rectangleABCD : Rectangle := { topLeft := (0, 6), bottomRight := (12, 0) }

theorem rectangle_area_is_72 
  (h1 : circleP.radius = circleQ.radius ∧ circleP.radius = circleR.radius ∧ circleP.radius = circleS.radius)
  (h2 : circleP.center.2 = circleQ.center.2 ∧ circleP.center.2 = circleR.center.2 ∧ circleP.center.2 = circleS.center.2)
  (h3 : circleP.center.1 + circleP.radius = circleQ.center.1 ∧ 
        circleQ.center.1 + circleQ.radius = circleR.center.1 ∧
        circleR.center.1 + circleR.radius = circleS.center.1)
  (h4 : rectangleABCD.topLeft.1 = circleP.center.1 - circleP.radius ∧
        rectangleABCD.bottomRight.1 = circleS.center.1 + circleS.radius)
  (h5 : rectangleABCD.topLeft.2 = circleP.center.2 + circleP.radius ∧
        rectangleABCD.bottomRight.2 = circleP.center.2 - circleP.radius)
  : (rectangleABCD.bottomRight.1 - rectangleABCD.topLeft.1) * 
    (rectangleABCD.topLeft.2 - rectangleABCD.bottomRight.2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_72_l1492_149255


namespace NUMINAMATH_CALUDE_tangent_to_circumcircle_l1492_149268

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary relations and functions
variable (midpoint : Circle → Point)
variable (intersect : Circle → Circle → Set Point)
variable (line_intersect : Point → Point → Circle → Set Point)
variable (on_line : Point → Point → Point → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (tangent_to : Point → Point → Circle → Prop)

-- State the theorem
theorem tangent_to_circumcircle
  (Γ₁ Γ₂ Γ₃ : Circle)
  (O₁ O₂ A C D S E F : Point) :
  (midpoint Γ₁ = O₁) →
  (midpoint Γ₂ = O₂) →
  (A ∈ intersect Γ₂ (circumcircle O₁ O₂ A)) →
  ({C, D} ⊆ intersect Γ₁ Γ₂) →
  (S ∈ line_intersect A D Γ₁) →
  (on_line C S F) →
  (on_line O₁ O₂ F) →
  (Γ₃ = circumcircle A D E) →
  (E ∈ intersect Γ₁ Γ₃) →
  (E ≠ D) →
  tangent_to O₁ E Γ₃ :=
sorry

end NUMINAMATH_CALUDE_tangent_to_circumcircle_l1492_149268


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1492_149231

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_put_balls_in_boxes 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1492_149231


namespace NUMINAMATH_CALUDE_at_most_one_negative_l1492_149265

theorem at_most_one_negative (a b c : ℝ) (sum_nonneg : a + b + c ≥ 0) (product_nonpos : a * b * c ≤ 0) :
  ¬(((a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_negative_l1492_149265


namespace NUMINAMATH_CALUDE_cos_equality_solution_l1492_149295

theorem cos_equality_solution (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solution_l1492_149295


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1492_149269

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  (a = 3 ∧ b = 2) ∨ (a = 2 ∧ b = 3) →
  (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) →
  c = Real.sqrt 13 ∨ c = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1492_149269


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l1492_149223

theorem determinant_of_cubic_roots (p q : ℝ) (a b c : ℂ) : 
  (a^3 + p*a + q = 0) → 
  (b^3 + p*b + q = 0) → 
  (c^3 + p*c + q = 0) → 
  (Complex.abs a ≠ 0) →
  (Complex.abs b ≠ 0) →
  (Complex.abs c ≠ 0) →
  let matrix := !![2 + a^2, 1, 1; 1, 2 + b^2, 1; 1, 1, 2 + c^2]
  Matrix.det matrix = (2*p^2 : ℂ) - 4*q + q^2 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l1492_149223


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l1492_149237

def satisfies_condition (p : Nat) : Prop :=
  Nat.Prime p ∧
  ∀ q, Nat.Prime q → q < p →
    ∀ k r, p = k * q + r → 0 ≤ r → r < q →
      ∀ a, a > 1 → ¬(a^2 ∣ r)

theorem prime_condition_characterization :
  {p : Nat | satisfies_condition p} = {2, 3, 5, 7, 13} := by sorry

end NUMINAMATH_CALUDE_prime_condition_characterization_l1492_149237


namespace NUMINAMATH_CALUDE_no_prime_solution_l1492_149264

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

theorem no_prime_solution :
  ¬∃ p : Nat, Nat.Prime p ∧
    (base_p_to_decimal [4, 1, 0, 1] p +
     base_p_to_decimal [2, 0, 5] p +
     base_p_to_decimal [7, 1, 2] p +
     base_p_to_decimal [1, 3, 2] p +
     base_p_to_decimal [2, 1] p =
     base_p_to_decimal [4, 5, 2] p +
     base_p_to_decimal [7, 4, 5] p +
     base_p_to_decimal [5, 7, 6] p) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1492_149264


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1492_149202

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 1) :
  (1/x + 1/y) ≥ 5 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1492_149202


namespace NUMINAMATH_CALUDE_marbles_distribution_l1492_149208

/-- Given a total number of marbles and a number of groups, 
    calculates the number of marbles in each group -/
def marbles_per_group (total_marbles : ℕ) (num_groups : ℕ) : ℕ :=
  total_marbles / num_groups

/-- Proves that given 20 marbles and 5 groups, there are 4 marbles in each group -/
theorem marbles_distribution :
  marbles_per_group 20 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l1492_149208


namespace NUMINAMATH_CALUDE_shielas_drawings_l1492_149293

/-- The number of neighbors Shiela has -/
def num_neighbors : ℕ := 6

/-- The number of drawings each neighbor would receive -/
def drawings_per_neighbor : ℕ := 9

/-- The total number of animal drawings Shiela drew -/
def total_drawings : ℕ := num_neighbors * drawings_per_neighbor

theorem shielas_drawings : total_drawings = 54 := by
  sorry

end NUMINAMATH_CALUDE_shielas_drawings_l1492_149293


namespace NUMINAMATH_CALUDE_product_of_sums_powers_specific_product_evaluation_l1492_149288

theorem product_of_sums_powers (a b : ℕ) : 
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) = 
  (1/2 : ℚ) * ((a^16 : ℚ) - (b^16 : ℚ)) :=
by sorry

theorem specific_product_evaluation : 
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21523360 :=
by sorry

end NUMINAMATH_CALUDE_product_of_sums_powers_specific_product_evaluation_l1492_149288


namespace NUMINAMATH_CALUDE_final_amount_is_correct_l1492_149215

/-- Calculates the final amount paid for a shopping trip with specific discounts and promotions -/
def calculate_final_amount (jimmy_shorts : ℕ) (jimmy_short_price : ℚ) 
                           (irene_shirts : ℕ) (irene_shirt_price : ℚ) 
                           (senior_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let jimmy_total := jimmy_shorts * jimmy_short_price
  let irene_total := irene_shirts * irene_shirt_price
  let jimmy_discounted := (jimmy_shorts / 3) * 2 * jimmy_short_price
  let irene_discounted := ((irene_shirts / 3) * 2 + irene_shirts % 3) * irene_shirt_price
  let total_before_discount := jimmy_discounted + irene_discounted
  let discount_amount := total_before_discount * senior_discount
  let total_after_discount := total_before_discount - discount_amount
  let tax_amount := total_after_discount * sales_tax
  total_after_discount + tax_amount

/-- Theorem stating that the final amount paid is $76.55 -/
theorem final_amount_is_correct : 
  calculate_final_amount 3 15 5 17 (1/10) (1/20) = 76.55 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_is_correct_l1492_149215


namespace NUMINAMATH_CALUDE_square_roots_theorem_l1492_149260

theorem square_roots_theorem (a x : ℝ) : 
  x > 0 ∧ (2*a - 1)^2 = x ∧ (-a + 2)^2 = x → x = 9 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l1492_149260


namespace NUMINAMATH_CALUDE_cloth_selling_amount_l1492_149297

/-- Calculates the total selling amount for cloth given the quantity, cost price, and loss per metre. -/
def totalSellingAmount (quantity : ℕ) (costPrice : ℕ) (lossPerMetre : ℕ) : ℕ :=
  quantity * (costPrice - lossPerMetre)

/-- Proves that the total selling amount for 200 metres of cloth with a cost price of 66 and a loss of 6 per metre is 12000. -/
theorem cloth_selling_amount :
  totalSellingAmount 200 66 6 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_amount_l1492_149297


namespace NUMINAMATH_CALUDE_expression_evaluation_l1492_149270

theorem expression_evaluation : 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + 0.40 + (0.5 : ℝ)^2 = 0.9666875 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1492_149270


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l1492_149278

-- Define the equations of the circles
def circle1_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 20
def circle2_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the lines
def line_y0 (x y : ℝ) : Prop := y = 0
def line_2x_y0 (x y : ℝ) : Prop := 2*x + y = 0
def line_tangent (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the points
def point_A : ℝ × ℝ := (1, 4)
def point_B : ℝ × ℝ := (3, 2)
def point_M : ℝ × ℝ := (2, -1)

-- Theorem for the first circle
theorem circle1_properties :
  ∃ (center_x : ℝ),
    (∀ (y : ℝ), circle1_equation center_x y → line_y0 center_x y) ∧
    circle1_equation point_A.1 point_A.2 ∧
    circle1_equation point_B.1 point_B.2 :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  ∃ (center_x center_y : ℝ),
    line_2x_y0 center_x center_y ∧
    (∀ (x y : ℝ), circle2_equation x y → 
      (x = point_M.1 ∧ y = point_M.2) → line_tangent x y) :=
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l1492_149278


namespace NUMINAMATH_CALUDE_magic_square_d_plus_e_l1492_149282

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  sum : ℤ
  row1_sum : 30 + b + 22 = sum
  row2_sum : 19 + c + d = sum
  row3_sum : a + 28 + f = sum
  col1_sum : 30 + 19 + a = sum
  col2_sum : b + c + 28 = sum
  col3_sum : 22 + d + f = sum
  diag1_sum : 30 + c + f = sum
  diag2_sum : a + c + 22 = sum

/-- The sum of d and e in the magic square is 54 -/
theorem magic_square_d_plus_e (ms : MagicSquare) : ms.d + ms.e = 54 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_d_plus_e_l1492_149282


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l1492_149238

/-- Given points A and B, and a point P on the extension of segment AB such that |AP| = 2|PB|, 
    prove that P has specific coordinates. -/
theorem extension_point_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  (∃ t : ℝ, t > 1 ∧ P = A + t • (B - A)) →
  ‖P - A‖ = 2 * ‖P - B‖ →
  P = (6, -9) := by
  sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l1492_149238


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1492_149298

theorem algebraic_expression_value (a b : ℝ) (h : a - b - 2 = 0) :
  a^2 - b^2 - 4*a = -4 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1492_149298


namespace NUMINAMATH_CALUDE_total_rope_inches_is_264_l1492_149205

/-- Represents the length of rope in feet for each week -/
def rope_length : Fin 4 → ℕ
  | 0 => 6  -- Week 1
  | 1 => 2 * rope_length 0  -- Week 2
  | 2 => rope_length 1 - 4  -- Week 3
  | 3 => rope_length 2 / 2  -- Week 4

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the total length of rope in feet at the end of the month -/
def total_rope_length : ℕ :=
  rope_length 0 + rope_length 1 + rope_length 2 - rope_length 3

/-- Theorem stating the total length of rope in inches at the end of the month -/
theorem total_rope_inches_is_264 : feet_to_inches total_rope_length = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_rope_inches_is_264_l1492_149205


namespace NUMINAMATH_CALUDE_volleyball_team_scoring_l1492_149276

/-- Volleyball team scoring problem -/
theorem volleyball_team_scoring 
  (lizzie_score : ℕ) 
  (nathalie_score : ℕ) 
  (aimee_score : ℕ) 
  (team_score : ℕ) 
  (h1 : lizzie_score = 4)
  (h2 : nathalie_score = lizzie_score + 3)
  (h3 : aimee_score = 2 * (lizzie_score + nathalie_score))
  (h4 : team_score = 50) :
  team_score - (lizzie_score + nathalie_score + aimee_score) = 17 := by
  sorry


end NUMINAMATH_CALUDE_volleyball_team_scoring_l1492_149276


namespace NUMINAMATH_CALUDE_kekai_remaining_money_l1492_149233

def shirt_price : ℝ := 1
def shirt_discount : ℝ := 0.2
def pants_price : ℝ := 3
def pants_discount : ℝ := 0.1
def hat_price : ℝ := 2
def hat_discount : ℝ := 0
def shoes_price : ℝ := 10
def shoes_discount : ℝ := 0.15
def parent_contribution : ℝ := 0.35

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def hats_sold : ℕ := 3
def shoes_sold : ℕ := 2

def total_sales (shirt_price pants_price hat_price shoes_price : ℝ)
                (shirt_discount pants_discount hat_discount shoes_discount : ℝ)
                (shirts_sold pants_sold hats_sold shoes_sold : ℕ) : ℝ :=
  (shirt_price * (1 - shirt_discount) * shirts_sold) +
  (pants_price * (1 - pants_discount) * pants_sold) +
  (hat_price * (1 - hat_discount) * hats_sold) +
  (shoes_price * (1 - shoes_discount) * shoes_sold)

def remaining_money (total : ℝ) (contribution : ℝ) : ℝ :=
  total * (1 - contribution)

theorem kekai_remaining_money :
  remaining_money (total_sales shirt_price pants_price hat_price shoes_price
                                shirt_discount pants_discount hat_discount shoes_discount
                                shirts_sold pants_sold hats_sold shoes_sold)
                  parent_contribution = 26.32 := by
  sorry

end NUMINAMATH_CALUDE_kekai_remaining_money_l1492_149233


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1492_149232

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = (x+1)*(x+3)) → m - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1492_149232


namespace NUMINAMATH_CALUDE_hoseok_multiplication_l1492_149213

theorem hoseok_multiplication (x : ℚ) : x / 11 = 2 → 6 * x = 132 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_multiplication_l1492_149213


namespace NUMINAMATH_CALUDE_tara_megan_money_difference_l1492_149250

/-- The problem of determining how much more money Tara has than Megan. -/
theorem tara_megan_money_difference
  (scooter_cost : ℕ)
  (tara_money : ℕ)
  (megan_money : ℕ)
  (h1 : scooter_cost = 26)
  (h2 : tara_money > megan_money)
  (h3 : tara_money + megan_money = scooter_cost)
  (h4 : tara_money = 15) :
  tara_money - megan_money = 4 := by
  sorry

end NUMINAMATH_CALUDE_tara_megan_money_difference_l1492_149250


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l1492_149224

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l1492_149224


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l1492_149273

theorem square_fence_perimeter 
  (num_posts : ℕ) 
  (post_width_inches : ℝ) 
  (gap_between_posts_feet : ℝ) : 
  num_posts = 36 →
  post_width_inches = 6 →
  gap_between_posts_feet = 8 →
  let posts_per_side : ℕ := num_posts / 4
  let gaps_per_side : ℕ := posts_per_side - 1
  let total_gap_length : ℝ := (gaps_per_side : ℝ) * gap_between_posts_feet
  let post_width_feet : ℝ := post_width_inches / 12
  let total_post_width : ℝ := (posts_per_side : ℝ) * post_width_feet
  let side_length : ℝ := total_gap_length + total_post_width
  let perimeter : ℝ := 4 * side_length
  perimeter = 242 := by
sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l1492_149273


namespace NUMINAMATH_CALUDE_two_solutions_exist_l1492_149200

-- Define the function g based on the graph
noncomputable def g : ℝ → ℝ := fun x =>
  if x < -1 then -2 * x
  else if x < 3 then 2 * x + 1
  else -2 * x + 16

-- Define the property we want to prove
def satisfies_equation (x : ℝ) : Prop := g (g x) = 4

-- Theorem statement
theorem two_solutions_exist :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ satisfies_equation x :=
sorry

end NUMINAMATH_CALUDE_two_solutions_exist_l1492_149200


namespace NUMINAMATH_CALUDE_race_outcomes_l1492_149214

/-- The number of contestants in the race -/
def num_contestants : ℕ := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def podium_positions : ℕ := 3

/-- No ties are allowed in the race -/
axiom no_ties : True

/-- The number of different podium outcomes in the race -/
def podium_outcomes : ℕ := num_contestants * (num_contestants - 1) * (num_contestants - 2)

/-- Theorem: The number of different podium outcomes in the race is 120 -/
theorem race_outcomes : podium_outcomes = 120 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l1492_149214


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l1492_149259

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def least_repeating_block_length : ℕ := 6

/-- Theorem stating that the least number of digits in a repeating block of 7/13 is 6 -/
theorem seven_thirteenths_repeating_block_length :
  least_repeating_block_length = 6 := by sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l1492_149259


namespace NUMINAMATH_CALUDE_history_book_cost_l1492_149239

theorem history_book_cost 
  (total_books : ℕ) 
  (math_book_cost : ℕ) 
  (total_price : ℕ) 
  (math_books : ℕ) 
  (h1 : total_books = 80) 
  (h2 : math_book_cost = 4) 
  (h3 : total_price = 368) 
  (h4 : math_books = 32) : 
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 := by
sorry

end NUMINAMATH_CALUDE_history_book_cost_l1492_149239


namespace NUMINAMATH_CALUDE_solution_range_l1492_149220

def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_range (f : ℝ → ℝ) (h_monotone : monotone_increasing f) (h_zero : f 1 = 0) :
  {x : ℝ | f (x^2 + 3*x - 3) < 0} = Set.Ioo (-4) 1 := by sorry

end NUMINAMATH_CALUDE_solution_range_l1492_149220
