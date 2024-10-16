import Mathlib

namespace NUMINAMATH_CALUDE_total_players_on_ground_l2245_224581

def cricket_players : ℕ := 35
def hockey_players : ℕ := 28
def football_players : ℕ := 42
def softball_players : ℕ := 25
def basketball_players : ℕ := 18
def volleyball_players : ℕ := 30

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + 
  softball_players + basketball_players + volleyball_players = 178 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l2245_224581


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2245_224517

theorem solve_quadratic_equation (s t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) :
  s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2245_224517


namespace NUMINAMATH_CALUDE_modulus_of_z_l2245_224533

theorem modulus_of_z (z : ℂ) (h : (z + Complex.I) * Complex.I = -3 + 4 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2245_224533


namespace NUMINAMATH_CALUDE_ecu_distribution_l2245_224597

theorem ecu_distribution (x y z : ℤ) : 
  (x - y - z = 8) ∧ 
  (y - (x - y - z) - z = 8) ∧ 
  (z - (x - y - z) - (y - (x - y - z)) = 8) → 
  x = 13 ∧ y = 7 ∧ z = 4 := by
sorry

end NUMINAMATH_CALUDE_ecu_distribution_l2245_224597


namespace NUMINAMATH_CALUDE_inheritance_calculation_inheritance_value_l2245_224509

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 49655

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The total tax paid in dollars -/
def total_tax_paid : ℝ := 18000

theorem inheritance_calculation :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax_paid := by sorry

theorem inheritance_value :
  inheritance = 49655 := by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_inheritance_value_l2245_224509


namespace NUMINAMATH_CALUDE_snow_probability_l2245_224511

theorem snow_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/4) (h2 : p2 = 1/2) (h3 : p3 = 1/3) :
  1 - (1 - p1)^2 * (1 - p2)^3 * (1 - p3)^2 = 31/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2245_224511


namespace NUMINAMATH_CALUDE_sum_cube_inequality_l2245_224548

theorem sum_cube_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_cube_inequality_l2245_224548


namespace NUMINAMATH_CALUDE_hypotenuse_length_l2245_224549

/-- A right triangle with specific properties -/
structure RightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  longer_leg_property : longer_leg = 3 * shorter_leg - 1
  area_property : (1 / 2) * shorter_leg * longer_leg = 24
  pythagorean_theorem : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The length of the hypotenuse in the specific right triangle is √137 -/
theorem hypotenuse_length (t : RightTriangle) : t.hypotenuse = Real.sqrt 137 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l2245_224549


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l2245_224531

def numerator : ℕ := 30 * 32 * 34 * 36 * 38 * 40
def denominator : ℕ := 2000

theorem units_digit_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n / d) % 10 = 2 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l2245_224531


namespace NUMINAMATH_CALUDE_junior_basketball_league_bad_teams_l2245_224557

/-- Given a total of 18 teams in a junior basketball league, where half are rich,
    and there cannot be 10 teams that are both rich and bad,
    prove that the fraction of bad teams must be less than or equal to 1/2. -/
theorem junior_basketball_league_bad_teams
  (total_teams : ℕ)
  (rich_teams : ℕ)
  (bad_fraction : ℚ)
  (h1 : total_teams = 18)
  (h2 : rich_teams = total_teams / 2)
  (h3 : ¬(bad_fraction * ↑total_teams ≥ 10 ∧ bad_fraction * ↑total_teams ≤ ↑rich_teams)) :
  bad_fraction ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_junior_basketball_league_bad_teams_l2245_224557


namespace NUMINAMATH_CALUDE_max_value_constrained_expression_l2245_224539

theorem max_value_constrained_expression :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 →
  x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constrained_expression_l2245_224539


namespace NUMINAMATH_CALUDE_egg_purchase_cost_l2245_224534

def dozen : ℕ := 12
def egg_price : ℚ := 0.50

theorem egg_purchase_cost (num_dozens : ℕ) : 
  (num_dozens * dozen * egg_price : ℚ) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_egg_purchase_cost_l2245_224534


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2245_224526

theorem smallest_number_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(((m + 7) % 8 = 0) ∧ ((m + 7) % 11 = 0) ∧ ((m + 7) % 24 = 0))) ∧
  ((n + 7) % 8 = 0) ∧ ((n + 7) % 11 = 0) ∧ ((n + 7) % 24 = 0) ∧
  n = 257 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2245_224526


namespace NUMINAMATH_CALUDE_triangle_shape_determination_l2245_224574

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The shape of a triangle is determined by its side lengths and angles -/
def triangle_shape (t : Triangle) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Two angles and the side between them -/
def sas_data (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Ratio of two angle bisectors -/
def angle_bisector_ratio (t : Triangle) : ℝ := sorry

/-- Ratio of circumradius to inradius -/
def radii_ratio (t : Triangle) : ℝ := sorry

/-- Ratio of area to perimeter -/
def area_perimeter_ratio (t : Triangle) : ℝ := sorry

/-- A function is shape-determining if it uniquely determines the triangle's shape -/
def is_shape_determining (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → triangle_shape t1 = triangle_shape t2

theorem triangle_shape_determination :
  is_shape_determining sas_data ∧
  ¬ is_shape_determining angle_bisector_ratio ∧
  is_shape_determining radii_ratio ∧
  is_shape_determining area_perimeter_ratio :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_determination_l2245_224574


namespace NUMINAMATH_CALUDE_min_period_sin_2x_cos_2x_l2245_224553

/-- The minimum positive period of the function y = sin(2x) cos(2x) is π/2 -/
theorem min_period_sin_2x_cos_2x :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x) * Real.cos (2 * x)
  ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_period_sin_2x_cos_2x_l2245_224553


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l2245_224582

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ : ℕ),
  (3 : ℚ) / 5 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧
  b₂ + b₃ + b₄ + b₅ = 4 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l2245_224582


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2245_224505

/-- Given an ellipse with equation x²/(k+8) + y²/9 = 1, foci on the y-axis, 
    and eccentricity 1/2, prove that k = -5/4 -/
theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) →  -- ellipse equation
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, x^2 / (k + 8) + (y - c)^2 / 9 = 1 ∧ 
                              x^2 / (k + 8) + (y + c)^2 / 9 = 1) →  -- foci on y-axis
  (let a := 3; let c := a / 2; c / a = 1 / 2) →  -- eccentricity = 1/2
  k = -5/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2245_224505


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solutions_l2245_224528

theorem quadratic_equation_one_solutions (x : ℝ) :
  x^2 - 6*x - 1 = 0 ↔ x = 3 + Real.sqrt 10 ∨ x = 3 - Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solutions_l2245_224528


namespace NUMINAMATH_CALUDE_fractional_equation_root_l2245_224544

/-- If the equation (3 / (x - 4)) + ((x + m) / (4 - x)) = 1 has a root, then m = -1 -/
theorem fractional_equation_root (x m : ℚ) : 
  (∃ x, (3 / (x - 4)) + ((x + m) / (4 - x)) = 1) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l2245_224544


namespace NUMINAMATH_CALUDE_larger_number_proof_l2245_224579

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1370) (h3 : L = 6 * S + 15) : L = 1641 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2245_224579


namespace NUMINAMATH_CALUDE_brother_birth_year_l2245_224521

/-- Given Karina's birth year, current age, and the fact that she is twice as old as her brother,
    prove her brother's birth year. -/
theorem brother_birth_year
  (karina_birth_year : ℕ)
  (karina_current_age : ℕ)
  (h_karina_birth : karina_birth_year = 1970)
  (h_karina_age : karina_current_age = 40)
  (h_twice_age : karina_current_age = 2 * (karina_current_age / 2)) :
  karina_birth_year + karina_current_age - (karina_current_age / 2) = 1990 := by
  sorry

end NUMINAMATH_CALUDE_brother_birth_year_l2245_224521


namespace NUMINAMATH_CALUDE_quadratic_strictly_increasing_iff_l2245_224555

/-- A function f: ℝ → ℝ is strictly increasing if for all x < y, f(x) < f(y) -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The quadratic function f(x) = ax^2 + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

theorem quadratic_strictly_increasing_iff (a : ℝ) :
  StrictlyIncreasing (f a) ↔ -1/4 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_strictly_increasing_iff_l2245_224555


namespace NUMINAMATH_CALUDE_TI_is_euler_line_l2245_224507

-- Define the basic structures
variable (A B C I T X Y Z : ℝ × ℝ)

-- Define the properties
variable (h1 : is_incenter I A B C)
variable (h2 : is_antigonal_point T I A B C)
variable (h3 : is_antipedal_triangle X Y Z T A B C)

-- Define the Euler line
def euler_line (X Y Z : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the line TI
def line_TI (T I : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem TI_is_euler_line :
  line_TI T I = euler_line X Y Z :=
sorry

end NUMINAMATH_CALUDE_TI_is_euler_line_l2245_224507


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l2245_224583

/-- Given a cubic function f(x) = ax³ + bx + 4 where a and b are non-zero real numbers,
    if f(5) = 10, then f(-5) = -2 -/
theorem cubic_function_symmetry (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := λ x : ℝ => a * x^3 + b * x + 4
  f 5 = 10 → f (-5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l2245_224583


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l2245_224515

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem larger_cuboid_height 
  (small : CuboidDimensions)
  (large_length : ℝ)
  (large_width : ℝ)
  (h_small_dims : small = { length := 5, width := 4, height := 3 })
  (h_large_dims : large_length = 16 ∧ large_width = 10)
  (h_count : 32 * cuboidVolume small = cuboidVolume { length := large_length, width := large_width, height := 12 }) :
  ∃ (large : CuboidDimensions), large.length = large_length ∧ large.width = large_width ∧ large.height = 12 :=
sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l2245_224515


namespace NUMINAMATH_CALUDE_two_evaluations_determine_sequence_l2245_224552

/-- A finite sequence of natural numbers -/
def Sequence := List Nat

/-- Evaluate the polynomial at a given point -/
def evaluatePolynomial (s : Sequence) (β : Nat) : Nat :=
  s.enum.foldl (fun acc (i, a) => acc + a * β ^ i) 0

/-- Theorem stating that two evaluations are sufficient to determine the sequence -/
theorem two_evaluations_determine_sequence (s : Sequence) :
  ∃ β₁ β₂ : Nat, β₁ ≠ β₂ ∧
  ∀ t : Sequence, t.length = s.length →
    evaluatePolynomial s β₁ = evaluatePolynomial t β₁ ∧
    evaluatePolynomial s β₂ = evaluatePolynomial t β₂ →
    s = t :=
  sorry

end NUMINAMATH_CALUDE_two_evaluations_determine_sequence_l2245_224552


namespace NUMINAMATH_CALUDE_slant_asymptote_sum_l2245_224569

/-- Given a rational function y = (3x^2 + 4x - 5) / (x - 4), 
    its slant asymptote is y = 3x + 16, 
    and the sum of the coefficients m and b in y = mx + b is 19 -/
theorem slant_asymptote_sum (x : ℝ) : 
  let y : ℝ → ℝ := λ x => (3*x^2 + 4*x - 5) / (x - 4)
  let asymptote : ℝ → ℝ := λ x => 3*x + 16
  let m : ℝ := 3
  let b : ℝ := 16
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |y x - asymptote x| < ε) ∧ 
  m + b = 19 := by
sorry

end NUMINAMATH_CALUDE_slant_asymptote_sum_l2245_224569


namespace NUMINAMATH_CALUDE_max_b_no_lattice_points_b_max_is_maximum_l2245_224554

/-- Represents a lattice point with integer coordinates -/
structure LatticePoint where
  x : Int
  y : Int

/-- Checks if a given point lies on the line y = mx + 3 -/
def lies_on_line (m : ℚ) (p : LatticePoint) : Prop :=
  p.y = m * p.x + 3

/-- The maximum value of b we want to prove -/
def b_max : ℚ := 76 / 151

theorem max_b_no_lattice_points :
  ∀ m : ℚ, 1/2 < m → m < b_max →
    ∀ x : ℤ, 0 < x → x ≤ 150 →
      ¬∃ p : LatticePoint, p.x = x ∧ lies_on_line m p :=
sorry

theorem b_max_is_maximum :
  ∀ b : ℚ, b > b_max →
    ∃ m : ℚ, 1/2 < m ∧ m < b ∧
      ∃ x : ℤ, 0 < x ∧ x ≤ 150 ∧
        ∃ p : LatticePoint, p.x = x ∧ lies_on_line m p :=
sorry

end NUMINAMATH_CALUDE_max_b_no_lattice_points_b_max_is_maximum_l2245_224554


namespace NUMINAMATH_CALUDE_chris_jogging_time_l2245_224503

/-- Represents the time in minutes -/
def Time := ℝ

/-- Represents the distance in miles -/
def Distance := ℝ

/-- Chris's jogging rate in minutes per mile -/
def chris_rate : ℝ := sorry

/-- Alex's walking rate in minutes per mile -/
def alex_rate : ℝ := sorry

theorem chris_jogging_time 
  (h1 : chris_rate * 4 = 2 * alex_rate * 2)  -- Chris's 4-mile time is twice Alex's 2-mile time
  (h2 : alex_rate * 2 = 40)                  -- Alex's 2-mile time is 40 minutes
  : chris_rate * 6 = 120 :=                  -- Chris's 6-mile time is 120 minutes
sorry

end NUMINAMATH_CALUDE_chris_jogging_time_l2245_224503


namespace NUMINAMATH_CALUDE_larger_number_problem_l2245_224541

theorem larger_number_problem (x y : ℕ) : 
  x + y = 70 ∧ 
  y = 15 ∧ 
  x = 3 * y + 10 → 
  x = 55 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2245_224541


namespace NUMINAMATH_CALUDE_function_positive_l2245_224573

/-- Given a function f: ℝ → ℝ with derivative f', 
    if 2f(x) + xf'(x) > x² for all x ∈ ℝ, 
    then f(x) > 0 for all x ∈ ℝ. -/
theorem function_positive 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * deriv f x > x^2) : 
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_l2245_224573


namespace NUMINAMATH_CALUDE_union_complement_equality_l2245_224568

def U : Set Nat := {1,2,3,4,5,6}
def P : Set Nat := {1,2,4}
def Q : Set Nat := {2,3,4,6}

theorem union_complement_equality : P ∪ (U \ Q) = {1,2,4,5} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l2245_224568


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2245_224525

/-- Given a rectangle with dimensions 12 and 18, prove that the fraction of the rectangle
    that is shaded is 1/12, where the shaded region is 1/3 of a quarter of the rectangle. -/
theorem shaded_fraction_of_rectangle (width : ℕ) (height : ℕ) (shaded_area : ℚ) :
  width = 12 →
  height = 18 →
  shaded_area = (1 / 3) * (1 / 4) * (width * height) →
  shaded_area / (width * height) = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2245_224525


namespace NUMINAMATH_CALUDE_roots_product_theorem_l2245_224529

theorem roots_product_theorem (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) → 
  (b^3 - 15*b^2 + 25*b - 10 = 0) → 
  (c^3 - 15*c^2 + 25*c - 10 = 0) → 
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end NUMINAMATH_CALUDE_roots_product_theorem_l2245_224529


namespace NUMINAMATH_CALUDE_square_area_l2245_224530

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The line y = 7 -/
def line : ℝ := 7

/-- The theorem stating the area of the square -/
theorem square_area : 
  ∃ (x₁ x₂ : ℝ), 
    parabola x₁ = line ∧ 
    parabola x₂ = line ∧ 
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 = 28 :=
sorry

end NUMINAMATH_CALUDE_square_area_l2245_224530


namespace NUMINAMATH_CALUDE_abigail_money_problem_l2245_224514

/-- Proves that Abigail had $11 at the start of the day given the conditions -/
theorem abigail_money_problem :
  ∀ (initial_money : ℕ),
    initial_money - 2 - 6 = 3 →
    initial_money = 11 := by
  sorry

end NUMINAMATH_CALUDE_abigail_money_problem_l2245_224514


namespace NUMINAMATH_CALUDE_sonia_and_joss_moving_l2245_224535

/-- Calculates the time spent filling the car per trip given the total moving time,
    number of trips, and driving time per trip. -/
def time_filling_car_per_trip (total_moving_time : ℕ) (num_trips : ℕ) (driving_time_per_trip : ℕ) : ℕ :=
  let total_minutes := total_moving_time * 60
  let total_driving_time := driving_time_per_trip * num_trips
  let total_filling_time := total_minutes - total_driving_time
  total_filling_time / num_trips

/-- Theorem stating that given the specific conditions of the problem,
    the time spent filling the car per trip is 40 minutes. -/
theorem sonia_and_joss_moving (total_moving_time : ℕ) (num_trips : ℕ) (driving_time_per_trip : ℕ) :
  total_moving_time = 7 →
  num_trips = 6 →
  driving_time_per_trip = 30 →
  time_filling_car_per_trip total_moving_time num_trips driving_time_per_trip = 40 :=
by
  sorry

#eval time_filling_car_per_trip 7 6 30

end NUMINAMATH_CALUDE_sonia_and_joss_moving_l2245_224535


namespace NUMINAMATH_CALUDE_equation_a_graph_l2245_224558

theorem equation_a_graph (x y : ℝ) :
  (x - 2) * (y + 3) = 0 ↔ (x = 2 ∨ y = -3) :=
sorry

end NUMINAMATH_CALUDE_equation_a_graph_l2245_224558


namespace NUMINAMATH_CALUDE_trash_outside_classrooms_l2245_224561

-- Define the number of classrooms
def num_classrooms : Nat := 8

-- Define the total number of trash pieces picked up
def total_trash : Nat := 1576

-- Define the number of trash pieces picked up in each classroom
def classroom_trash : Fin num_classrooms → Nat
  | ⟨0, _⟩ => 124  -- Classroom 1
  | ⟨1, _⟩ => 98   -- Classroom 2
  | ⟨2, _⟩ => 176  -- Classroom 3
  | ⟨3, _⟩ => 212  -- Classroom 4
  | ⟨4, _⟩ => 89   -- Classroom 5
  | ⟨5, _⟩ => 241  -- Classroom 6
  | ⟨6, _⟩ => 121  -- Classroom 7
  | ⟨7, _⟩ => 102  -- Classroom 8
  | ⟨n+8, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 8 n))

-- Theorem to prove
theorem trash_outside_classrooms :
  total_trash - (Finset.sum Finset.univ classroom_trash) = 413 := by
  sorry

end NUMINAMATH_CALUDE_trash_outside_classrooms_l2245_224561


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l2245_224585

/-- A color represented as an enumeration -/
inductive Color
  | Red
  | Green
  | Blue

/-- A grid coloring is a function from grid coordinates to colors -/
def GridColoring := (Fin 4 × Fin 82) → Color

/-- A rectangle is represented by four points in the grid -/
structure Rectangle :=
  (p1 p2 p3 p4 : Fin 4 × Fin 82)

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorRectangle (coloring : GridColoring) (rect : Rectangle) : Prop :=
  coloring rect.p1 = coloring rect.p2 ∧
  coloring rect.p1 = coloring rect.p3 ∧
  coloring rect.p1 = coloring rect.p4

/-- Main theorem: There exists a rectangle with vertices of the same color in any 4x82 grid coloring -/
theorem exists_same_color_rectangle (coloring : GridColoring) :
  ∃ (rect : Rectangle), SameColorRectangle coloring rect := by
  sorry


end NUMINAMATH_CALUDE_exists_same_color_rectangle_l2245_224585


namespace NUMINAMATH_CALUDE_ptolemys_inequality_ptolemys_inequality_equality_l2245_224536

/-- Ptolemy's inequality in the complex plane -/
theorem ptolemys_inequality (a b c d : ℂ) :
  Complex.abs c * Complex.abs (d - b) ≤ 
  Complex.abs d * Complex.abs (c - b) + Complex.abs b * Complex.abs (c - d) :=
sorry

/-- Condition for equality in Ptolemy's inequality -/
def concyclic_or_collinear (a b c d : ℂ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (b - a) * (d - c) = k * (c - b) * (d - a)

/-- Ptolemy's inequality with equality condition -/
theorem ptolemys_inequality_equality (a b c d : ℂ) :
  Complex.abs c * Complex.abs (d - b) = 
  Complex.abs d * Complex.abs (c - b) + Complex.abs b * Complex.abs (c - d) ↔
  concyclic_or_collinear a b c d :=
sorry

end NUMINAMATH_CALUDE_ptolemys_inequality_ptolemys_inequality_equality_l2245_224536


namespace NUMINAMATH_CALUDE_correct_calculation_l2245_224540

theorem correct_calculation (a : ℝ) : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2245_224540


namespace NUMINAMATH_CALUDE_license_plate_count_l2245_224570

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of distinct letters that are repeated -/
def repeated_letters : ℕ := 2

/-- The number of license plate combinations -/
def license_plate_combinations : ℕ := 7776000

theorem license_plate_count :
  (Nat.choose alphabet_size repeated_letters) *
  (alphabet_size - repeated_letters) *
  (Nat.choose letter_positions repeated_letters) *
  (Nat.choose (letter_positions - repeated_letters) repeated_letters) *
  (Nat.factorial digit_positions) = license_plate_combinations :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2245_224570


namespace NUMINAMATH_CALUDE_no_integer_solution_l2245_224594

theorem no_integer_solution : ¬ ∃ (x : ℤ), x^2 * 7 = 2^14 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2245_224594


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l2245_224506

theorem spencer_walk_distance (distance_house_to_library : ℝ) 
                               (distance_library_to_post_office : ℝ) 
                               (distance_post_office_to_home : ℝ) 
                               (h1 : distance_house_to_library = 0.3)
                               (h2 : distance_library_to_post_office = 0.1)
                               (h3 : distance_post_office_to_home = 0.4) :
  distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l2245_224506


namespace NUMINAMATH_CALUDE_intersection_M_N_l2245_224508

def M : Set ℝ := {x | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2245_224508


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2245_224513

def is_valid_number (n : ℕ) : Prop :=
  n % 5 = 184 ∧ n % 6 = 184 ∧ n % 9 = 184 ∧ n % 12 = 184

theorem least_number_with_remainder :
  is_valid_number 364 ∧ ∀ m : ℕ, m < 364 → ¬(is_valid_number m) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2245_224513


namespace NUMINAMATH_CALUDE_books_calculation_initial_books_count_l2245_224560

/-- The number of books initially on the shelf -/
def initial_books : ℕ := sorry

/-- The number of books Marta added to the shelf -/
def books_added : ℕ := 10

/-- The final number of books on the shelf -/
def final_books : ℕ := 48

/-- Theorem stating that the initial number of books plus the added books equals the final number of books -/
theorem books_calculation : initial_books + books_added = final_books := by sorry

/-- Theorem proving that the initial number of books is 38 -/
theorem initial_books_count : initial_books = 38 := by sorry

end NUMINAMATH_CALUDE_books_calculation_initial_books_count_l2245_224560


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2245_224584

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < 2}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | log10 (x^2 + 2*x + 2) < 1} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2245_224584


namespace NUMINAMATH_CALUDE_find_k_l2245_224504

def is_max_solution (k : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 - 4*x + k| + |x - 3| ≤ 5 → x ≤ 3

theorem find_k : ∃! k : ℝ, is_max_solution k ∧ k = 8 := by sorry

end NUMINAMATH_CALUDE_find_k_l2245_224504


namespace NUMINAMATH_CALUDE_nested_sqrt_fraction_l2245_224571

/-- Given a real number x satisfying the equation x = 2 + √3 / x,
    prove that 1 / ((x + 2)(x - 3)) = (√3 + 5) / (-22) -/
theorem nested_sqrt_fraction (x : ℝ) (hx : x = 2 + Real.sqrt 3 / x) :
  1 / ((x + 2) * (x - 3)) = (Real.sqrt 3 + 5) / (-22) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fraction_l2245_224571


namespace NUMINAMATH_CALUDE_rice_distribution_difference_l2245_224576

/-- Given a total amount of rice and the fraction kept by Mr. Llesis,
    calculate how much more rice Mr. Llesis keeps compared to Mr. Everest. -/
def rice_difference (total : ℚ) (llesis_fraction : ℚ) : ℚ :=
  let llesis_amount := total * llesis_fraction
  let everest_amount := total - llesis_amount
  llesis_amount - everest_amount

/-- Theorem stating that given 50 kg of rice, if Mr. Llesis keeps 7/10 of it,
    he will have 20 kg more than Mr. Everest. -/
theorem rice_distribution_difference :
  rice_difference 50 (7/10) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_difference_l2245_224576


namespace NUMINAMATH_CALUDE_soccer_match_total_goals_is_22_l2245_224567

def soccer_match_total_goals : ℕ :=
  let kickers_first := 2
  let kickers_second := 2 * kickers_first
  let spiders_first := kickers_first / 2
  let spiders_second := spiders_first ^ 2
  let kickers_third := 2 * (kickers_first + kickers_second)
  let spiders_third := spiders_second

  let kickers_bonus := 
    (if kickers_first % 3 = 0 then 1 else 0) +
    (if kickers_second % 3 = 0 then 1 else 0) +
    (if kickers_third % 3 = 0 then 1 else 0)

  let spiders_bonus := 
    (if spiders_first % 2 = 0 then 1 else 0) +
    (if spiders_second % 2 = 0 then 1 else 0) +
    (if spiders_third % 2 = 0 then 1 else 0)

  kickers_first + kickers_second + kickers_third + kickers_bonus +
  spiders_first + spiders_second + spiders_third + spiders_bonus

theorem soccer_match_total_goals_is_22 : soccer_match_total_goals = 22 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_total_goals_is_22_l2245_224567


namespace NUMINAMATH_CALUDE_total_cantaloupes_is_65_l2245_224532

/-- The number of cantaloupes grown by Keith -/
def keith_cantaloupes : ℕ := 29

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 16

/-- The number of cantaloupes grown by Jason -/
def jason_cantaloupes : ℕ := 20

/-- The total number of cantaloupes grown by Keith, Fred, and Jason -/
def total_cantaloupes : ℕ := keith_cantaloupes + fred_cantaloupes + jason_cantaloupes

theorem total_cantaloupes_is_65 : total_cantaloupes = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_is_65_l2245_224532


namespace NUMINAMATH_CALUDE_same_side_of_line_l2245_224524

/-- Define a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the line equation --/
def line_equation (p : Point) : ℝ := p.x + p.y - 1

/-- Check if a point is on the positive side of the line --/
def is_positive_side (p : Point) : Prop := line_equation p > 0

/-- The reference point (1,2) --/
def reference_point : Point := ⟨1, 2⟩

/-- The point to be checked (-1,3) --/
def check_point : Point := ⟨-1, 3⟩

/-- Theorem statement --/
theorem same_side_of_line : 
  is_positive_side reference_point → is_positive_side check_point :=
by sorry

end NUMINAMATH_CALUDE_same_side_of_line_l2245_224524


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l2245_224587

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 - Complex.I) * (2 + 3 * Complex.I) ∧ z.re = 5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l2245_224587


namespace NUMINAMATH_CALUDE_equation_solution_l2245_224523

theorem equation_solution : 
  ∃! x : ℝ, x ≠ -1 ∧ x ≠ -(3/2) ∧ x ≠ 1/2 ∧ x ≠ -(1/2) ∧
  (((((2*x+1)/(2*x-1))-1)/(1-((2*x-1)/(2*x+1)))) + 
   ((((2*x+1)/(2*x-1))-2)/(2-((2*x-1)/(2*x+1)))) +
   ((((2*x+1)/(2*x-1))-3)/(3-((2*x-1)/(2*x+1))))) = 0 ∧
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2245_224523


namespace NUMINAMATH_CALUDE_certain_number_problem_l2245_224564

theorem certain_number_problem (p q : ℕ) (x : ℤ) : 
  p > 1 → q > 1 → p + q = 36 → x * (p + 1) = 21 * (q + 1) → x = 245 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2245_224564


namespace NUMINAMATH_CALUDE_girls_in_selection_l2245_224518

theorem girls_in_selection (n : ℕ) : 
  (1 - (Nat.choose 3 3 : ℚ) / (Nat.choose (3 + n) 3 : ℚ) = 34 / 35) → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_selection_l2245_224518


namespace NUMINAMATH_CALUDE_set_a_condition_l2245_224590

theorem set_a_condition (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a > 0}
  1 ∉ A → a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_set_a_condition_l2245_224590


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2245_224545

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (bridge_length : ℝ)
  (train_length : ℝ)
  (lamp_post_time : ℝ)
  (h1 : bridge_length = 200)
  (h2 : train_length = 200)
  (h3 : lamp_post_time = 5)
  : ℝ :=
by
  -- The time taken for the train to cross the bridge is 10 seconds
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2245_224545


namespace NUMINAMATH_CALUDE_balls_in_box_l2245_224516

theorem balls_in_box (initial_balls : ℕ) (balls_taken : ℕ) (balls_left : ℕ) : 
  initial_balls = 10 → balls_taken = 3 → balls_left = initial_balls - balls_taken → balls_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_box_l2245_224516


namespace NUMINAMATH_CALUDE_haley_spent_32_l2245_224562

/-- The amount Haley spent on concert tickets -/
def haley_spent (ticket_price : ℕ) (self_and_friends : ℕ) (extra : ℕ) : ℕ :=
  (self_and_friends + extra) * ticket_price

/-- Proof that Haley spent $32 on concert tickets -/
theorem haley_spent_32 :
  haley_spent 4 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_spent_32_l2245_224562


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2245_224550

theorem no_real_roots_for_nonzero_k :
  ∀ (k : ℝ), k ≠ 0 → ¬∃ (x : ℝ), x^2 + k*x + 2*k^2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2245_224550


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l2245_224589

theorem greatest_four_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≤ 9996 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l2245_224589


namespace NUMINAMATH_CALUDE_constant_triangle_area_l2245_224588

noncomputable section

-- Define the curve C: xy = 1, x > 0
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 1 ∧ p.1 > 0}

-- Define a point P on the curve C
def P (a : ℝ) : ℝ × ℝ := (a, 1/a)

-- Define the tangent line l at point P
def tangent_line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - 1/a = -(1/a^2) * (p.1 - a)}

-- Define points A and B as intersections of tangent line with axes
def A (a : ℝ) : ℝ × ℝ := (0, 2/a)
def B (a : ℝ) : ℝ × ℝ := (2*a, 0)

-- Define the area of triangle OAB
def triangle_area (a : ℝ) : ℝ := (1/2) * (2/a) * (2*a)

-- Theorem statement
theorem constant_triangle_area (a : ℝ) (h : a > 0) :
  P a ∈ C → triangle_area a = 2 := by sorry

end

end NUMINAMATH_CALUDE_constant_triangle_area_l2245_224588


namespace NUMINAMATH_CALUDE_no_natural_solution_l2245_224556

theorem no_natural_solution : ¬∃ (x y : ℕ), x^4 - y^4 = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2245_224556


namespace NUMINAMATH_CALUDE_inequality_proof_l2245_224596

theorem inequality_proof (p : ℝ) (hp : p > 1) :
  ∃ (K_p : ℝ), K_p > 0 ∧
  ∀ (x y : ℝ), (|x|^p + |y|^p = 2) →
  (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2245_224596


namespace NUMINAMATH_CALUDE_typists_problem_l2245_224565

theorem typists_problem (initial_letters : ℕ) (initial_time : ℕ) (new_typists : ℕ) (new_letters : ℕ) (new_time : ℕ) :
  initial_letters = 48 →
  initial_time = 20 →
  new_typists = 30 →
  new_letters = 216 →
  new_time = 60 →
  ∃ x : ℕ, x > 0 ∧ (initial_letters / x : ℚ) * new_typists * (new_time / initial_time : ℚ) = new_letters :=
by
  sorry

end NUMINAMATH_CALUDE_typists_problem_l2245_224565


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l2245_224502

/-- Two parabolas with given properties -/
structure ParabolaPair where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : y₁ = a * x₁^2 + b * x₁ + c  -- Vertex condition for N₁
  h₂ : y₂ = -a * x₂^2 + d * x₂ + e  -- Vertex condition for N₂
  h₃ : 21 = a * 12^2 + b * 12 + c  -- A(12, 21) lies on N₁
  h₄ : 3 = a * 28^2 + b * 28 + c  -- B(28, 3) lies on N₁
  h₅ : 21 = -a * 12^2 + d * 12 + e  -- A(12, 21) lies on N₂
  h₆ : 3 = -a * 28^2 + d * 28 + e  -- B(28, 3) lies on N₂

/-- The main theorem -/
theorem parabola_intersection_ratio (p : ParabolaPair) :
  (p.x₁ + p.x₂) / (p.y₁ + p.y₂) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_ratio_l2245_224502


namespace NUMINAMATH_CALUDE_middle_integer_of_consecutive_evens_l2245_224551

theorem middle_integer_of_consecutive_evens (n : ℕ) : 
  (n > 0) →                                 -- n is positive
  (n < 10) →                                -- n is one-digit
  (n % 2 = 0) →                             -- n is even
  (n + (n - 2) + (n + 2) = (n * (n - 2) * (n + 2)) / 8) → -- sum is one-eighth of product
  (n = 4) :=                                -- middle integer is 4
by sorry

end NUMINAMATH_CALUDE_middle_integer_of_consecutive_evens_l2245_224551


namespace NUMINAMATH_CALUDE_ratio_problem_l2245_224512

theorem ratio_problem (c d : ℚ) : 
  (c / d = 5) → (c = 18 - 7 * d) → (d = 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2245_224512


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2245_224577

theorem investment_interest_rate (total_investment : ℝ) (first_part : ℝ) (first_rate : ℝ) (total_interest : ℝ) : 
  total_investment = 3600 →
  first_part = 1800 →
  first_rate = 3 →
  total_interest = 144 →
  (first_part * first_rate / 100 + (total_investment - first_part) * 5 / 100 = total_interest) :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2245_224577


namespace NUMINAMATH_CALUDE_sup_good_is_ln_2_l2245_224592

/-- A positive real number d is good if there exists an infinite sequence
    a₁, a₂, a₃, ... ∈ (0,d) such that for each n, the points a₁, a₂, ..., aₙ
    partition the interval [0,d] into segments of length at most 1/n each. -/
def IsGood (d : ℝ) : Prop :=
  d > 0 ∧ ∃ a : ℕ → ℝ, ∀ n : ℕ,
    (∀ i : ℕ, i ≤ n → 0 < a i ∧ a i < d) ∧
    (∀ i : ℕ, i ≤ n → ∀ j : ℕ, j ≤ n → i ≠ j → |a i - a j| ≤ 1 / n) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ d → ∃ i : ℕ, i ≤ n ∧ |x - a i| ≤ 1 / n)

/-- The supremum of the set of all good numbers is ln 2. -/
theorem sup_good_is_ln_2 : sSup {d : ℝ | IsGood d} = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_sup_good_is_ln_2_l2245_224592


namespace NUMINAMATH_CALUDE_gcd_of_90_and_252_l2245_224546

theorem gcd_of_90_and_252 : Nat.gcd 90 252 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_252_l2245_224546


namespace NUMINAMATH_CALUDE_inverse_proportion_l2245_224575

/-- Given that x is inversely proportional to y, prove that if x = 4 when y = 2, then x = 4/5 when y = 10 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  x * 10 = k → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2245_224575


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l2245_224586

/-- Represents Susan's candy consumption and purchases over a week -/
structure CandyWeek where
  dailyLimit : ℕ
  tuesdayBought : ℕ
  thursdayBought : ℕ
  fridayBought : ℕ
  leftAtEndOfWeek : ℕ
  totalSpent : ℕ

/-- Calculates the number of candies Susan ate during the week -/
def candiesEaten (week : CandyWeek) : ℕ :=
  week.tuesdayBought + week.thursdayBought + week.fridayBought - week.leftAtEndOfWeek

/-- Theorem stating that Susan ate 6 candies during the week -/
theorem susan_ate_six_candies (week : CandyWeek)
  (h1 : week.dailyLimit = 3)
  (h2 : week.tuesdayBought = 3)
  (h3 : week.thursdayBought = 5)
  (h4 : week.fridayBought = 2)
  (h5 : week.leftAtEndOfWeek = 4)
  (h6 : week.totalSpent = 9) :
  candiesEaten week = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l2245_224586


namespace NUMINAMATH_CALUDE_chocolate_division_l2245_224599

/-- The amount of chocolate Shaina receives when Jordan divides his chocolate -/
theorem chocolate_division (total : ℚ) (keep_fraction : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) : 
  total = 60 / 7 →
  keep_fraction = 1 / 3 →
  num_piles = 5 →
  piles_to_shaina = 2 →
  (1 - keep_fraction) * total * (piles_to_shaina / num_piles) = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l2245_224599


namespace NUMINAMATH_CALUDE_probability_of_third_six_l2245_224578

theorem probability_of_third_six (p_fair : ℝ) (p_biased : ℝ) (p_other : ℝ) : 
  p_fair = 1/6 →
  p_biased = 2/3 →
  p_other = 1/15 →
  (1/6^2 / (1/6^2 + (2/3)^2)) * (1/6) + ((2/3)^2 / (1/6^2 + (2/3)^2)) * (2/3) = 65/102 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_third_six_l2245_224578


namespace NUMINAMATH_CALUDE_distance_point_to_line_l2245_224501

/-- The distance from the point (√2, -√2) to the line x + y = 1 is √2/2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)
  let line (x y : ℝ) : Prop := x + y = 1
  abs (point.1 + point.2 - 1) / Real.sqrt 2 = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l2245_224501


namespace NUMINAMATH_CALUDE_expression_value_at_12_l2245_224542

theorem expression_value_at_12 :
  let y : ℝ := 12
  (y^9 - 27*y^6 + 243*y^3 - 729) / (y^3 - 9) = 5082647079 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_12_l2245_224542


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l2245_224500

/-- The amount of flour Mary has already put in the cake -/
def flour_put_in : ℕ := sorry

/-- The total amount of flour required by the recipe -/
def total_flour_required : ℕ := 12

/-- The amount of flour still needed -/
def flour_still_needed : ℕ := 2

theorem mary_flour_calculation :
  flour_put_in = total_flour_required - flour_still_needed :=
sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l2245_224500


namespace NUMINAMATH_CALUDE_unique_minimum_condition_l2245_224580

/-- The objective function z(x,y) = ax + 2y has its unique minimum at (1,0) for all real x and y
    if and only if a is in the open interval (-4, -2) -/
theorem unique_minimum_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y) ≥ (a * 1 + 2 * 0) ∧ 
   (∀ x' y' : ℝ, (x', y') ≠ (1, 0) → (a * x' + 2 * y') > (a * 1 + 2 * 0)))
  ↔ 
  (-4 < a ∧ a < -2) :=
sorry

end NUMINAMATH_CALUDE_unique_minimum_condition_l2245_224580


namespace NUMINAMATH_CALUDE_line_equivalence_l2245_224527

/-- Definition of the line using dot product equation -/
def line_equation (x y : ℝ) : Prop :=
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0

/-- Slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

/-- The slope and y-intercept of the line -/
def slope_intercept_params : ℝ × ℝ := (2, -10)

theorem line_equivalence :
  ∀ (x y : ℝ),
    line_equation x y ↔ slope_intercept_form (slope_intercept_params.1) (slope_intercept_params.2) x y :=
by sorry

#check line_equivalence

end NUMINAMATH_CALUDE_line_equivalence_l2245_224527


namespace NUMINAMATH_CALUDE_string_cutting_l2245_224520

/-- Proves that cutting off 1/4 of a 2/3 meter long string leaves 50 cm remaining. -/
theorem string_cutting (string_length : ℚ) (h1 : string_length = 2/3) :
  (string_length * 100 - (1/4 * string_length * 100)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_string_cutting_l2245_224520


namespace NUMINAMATH_CALUDE_fruit_bowl_oranges_l2245_224547

theorem fruit_bowl_oranges :
  let bananas : ℕ := 4
  let apples : ℕ := 3 * bananas
  let pears : ℕ := 5
  let total_fruits : ℕ := 30
  let oranges : ℕ := total_fruits - (bananas + apples + pears)
  oranges = 9 :=
by sorry

end NUMINAMATH_CALUDE_fruit_bowl_oranges_l2245_224547


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2245_224543

theorem container_volume_ratio (volume_1 volume_2 : ℚ) : 
  volume_1 > 0 → volume_2 > 0 → 
  (3 / 4 : ℚ) * volume_1 = (5 / 8 : ℚ) * volume_2 → 
  volume_1 / volume_2 = (5 / 6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2245_224543


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2245_224591

/-- A quadratic function of the form y = 3(x - a)² -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := 3 * (x - a)^2

/-- The property that y increases as x increases when x > 2 -/
def increasing_when_x_gt_2 (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → quadratic_function a x₂ > quadratic_function a x₁

theorem quadratic_function_property (a : ℝ) :
  increasing_when_x_gt_2 a → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2245_224591


namespace NUMINAMATH_CALUDE_min_sum_inverse_squares_l2245_224522

/-- Given two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0,
    where a and b are real numbers, ab ≠ 0, and the circles have exactly three common tangent lines,
    prove that the minimum value of 1/a^2 + 1/b^2 is 1. -/
theorem min_sum_inverse_squares (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h_tangent : ∃ (t1 t2 t3 : ℝ × ℝ), 
      t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
      (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∨ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
        ((x - t1.1)^2 + (y - t1.2)^2 = 0 ∨
         (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨
         (x - t3.1)^2 + (y - t3.2)^2 = 0))) :
  (∀ c d : ℝ, c ≠ 0 → d ≠ 0 → 
    (∃ (t1' t2' t3' : ℝ × ℝ), 
      t1' ≠ t2' ∧ t1' ≠ t3' ∧ t2' ≠ t3' ∧
      (∀ (x y : ℝ), (x^2 + y^2 + 2*c*x + c^2 - 4 = 0 ∨ x^2 + y^2 - 4*d*y - 1 + 4*d^2 = 0) →
        ((x - t1'.1)^2 + (y - t1'.2)^2 = 0 ∨
         (x - t2'.1)^2 + (y - t2'.2)^2 = 0 ∨
         (x - t3'.1)^2 + (y - t3'.2)^2 = 0))) →
    1 / c^2 + 1 / d^2 ≥ 1) ∧
  (1 / a^2 + 1 / b^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_min_sum_inverse_squares_l2245_224522


namespace NUMINAMATH_CALUDE_green_team_score_l2245_224566

theorem green_team_score (other_team_score lead : ℕ) (h1 : other_team_score = 68) (h2 : lead = 29) :
  ∃ G : ℕ, other_team_score = G + lead ∧ G = 39 := by
  sorry

end NUMINAMATH_CALUDE_green_team_score_l2245_224566


namespace NUMINAMATH_CALUDE_candidate_votes_l2245_224563

theorem candidate_votes (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 70 / 100 →
  ∃ (valid_votes : ℕ) (candidate_votes : ℕ),
    valid_votes = (1 - invalid_percentage) * total_votes ∧
    candidate_votes = candidate_percentage * valid_votes ∧
    candidate_votes = 333200 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l2245_224563


namespace NUMINAMATH_CALUDE_total_cost_pencils_erasers_l2245_224537

/-- Given the price of a pencil (a) and an eraser (b) in yuan, 
    prove that the total cost of 3 pencils and 7 erasers is 3a + 7b yuan. -/
theorem total_cost_pencils_erasers (a b : ℝ) : 3 * a + 7 * b = 3 * a + 7 * b := by
  sorry

end NUMINAMATH_CALUDE_total_cost_pencils_erasers_l2245_224537


namespace NUMINAMATH_CALUDE_exam_mistakes_l2245_224598

theorem exam_mistakes (total_students : ℕ) (total_mistakes : ℕ) 
  (h1 : total_students = 333) (h2 : total_mistakes = 1000) : 
  ∀ (x y z : ℕ), 
    (x + y + z = total_students) → 
    (4 * y + 6 * z ≤ total_mistakes) → 
    (z ≤ x) :=
by
  sorry

end NUMINAMATH_CALUDE_exam_mistakes_l2245_224598


namespace NUMINAMATH_CALUDE_inequality_proof_l2245_224572

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 3) :
  (a / (1 + b^2 * c)) + (b / (1 + c^2 * a)) + (c / (1 + a^2 * b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2245_224572


namespace NUMINAMATH_CALUDE_number_equation_solution_l2245_224593

theorem number_equation_solution : 
  ∃ x : ℝ, (3034 - (x / 20.04) = 2984) ∧ (x = 1002) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2245_224593


namespace NUMINAMATH_CALUDE_bulb_arrangement_count_l2245_224510

/-- The number of ways to arrange bulbs in a garland with no consecutive white bulbs -/
def bulb_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red) blue * Nat.choose (blue + red + 1) white

/-- Theorem stating the number of arrangements for the given bulb counts -/
theorem bulb_arrangement_count :
  bulb_arrangements 7 6 10 = 1717716 := by
  sorry

end NUMINAMATH_CALUDE_bulb_arrangement_count_l2245_224510


namespace NUMINAMATH_CALUDE_banana_distribution_l2245_224595

/-- Given three people with a total of 200 bananas, where one person has 40 more than another
    and the third person has 40 bananas, prove that the person with the least bananas has 60. -/
theorem banana_distribution (total : ℕ) (difference : ℕ) (donna_bananas : ℕ)
    (h_total : total = 200)
    (h_difference : difference = 40)
    (h_donna : donna_bananas = 40) :
    ∃ (lydia dawn : ℕ),
      lydia + dawn + donna_bananas = total ∧
      dawn = lydia + difference ∧
      lydia = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l2245_224595


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2245_224559

/-- The asymptote equation of the hyperbola x² - y²/3 = -1 is y = ±√3x -/
theorem hyperbola_asymptote :
  let h : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 - y^2/3 + 1
  ∃ (k : ℝ), k = Real.sqrt 3 ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, 
      x^2 + y^2 > M^2 → h (x, y) = 0 → |y - k*x| < ε*|x| ∨ |y + k*x| < ε*|x|) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2245_224559


namespace NUMINAMATH_CALUDE_least_cars_per_day_l2245_224519

/-- Represents a mechanic's work details -/
structure Mechanic where
  rate : ℕ  -- cars per hour
  hours : ℕ  -- total work hours
  lunch_break : ℕ  -- lunch break in hours
  additional_break : ℕ  -- additional break in half-hours

/-- Calculates the number of cars a mechanic can service in a day -/
def cars_serviced (m : Mechanic) : ℕ :=
  m.rate * (m.hours - m.lunch_break - m.additional_break / 2)

/-- The three mechanics at the oil spot -/
def paul : Mechanic := { rate := 2, hours := 8, lunch_break := 1, additional_break := 1 }
def jack : Mechanic := { rate := 3, hours := 6, lunch_break := 1, additional_break := 1 }
def sam : Mechanic := { rate := 4, hours := 5, lunch_break := 1, additional_break := 0 }

/-- Theorem stating the least number of cars the mechanics can finish together per workday -/
theorem least_cars_per_day : 
  cars_serviced paul + cars_serviced jack + cars_serviced sam = 42 := by
  sorry

end NUMINAMATH_CALUDE_least_cars_per_day_l2245_224519


namespace NUMINAMATH_CALUDE_last_number_proof_l2245_224538

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 3 →
  a + d = 13 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2245_224538
