import Mathlib

namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equality_l308_30863

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 9/w = 1 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + 2*y = 19 + 6*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equality_l308_30863


namespace NUMINAMATH_CALUDE_ant_movement_probability_l308_30827

/-- A point in a 3D cubic lattice grid -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The number of adjacent points in a 3D cubic lattice -/
def adjacent_points : ℕ := 6

/-- The number of steps the ant takes -/
def num_steps : ℕ := 4

/-- The probability of moving to a specific adjacent point in one step -/
def step_probability : ℚ := 1 / adjacent_points

/-- 
  Theorem: The probability of an ant moving from point A to point B 
  (directly one floor above A) on a cubic lattice grid in exactly four steps, 
  where each step is to an adjacent point with equal probability, is 1/1296.
-/
theorem ant_movement_probability (A B : Point3D) 
  (h1 : B.x = A.x ∧ B.y = A.y ∧ B.z = A.z + 1) : 
  step_probability ^ num_steps = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_ant_movement_probability_l308_30827


namespace NUMINAMATH_CALUDE_f_2015_5_l308_30879

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_2015_5 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 4)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) :
  f 2015.5 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_5_l308_30879


namespace NUMINAMATH_CALUDE_great_wall_precision_l308_30800

/-- The precision of a number in scientific notation is determined by the place value of its last significant digit. -/
def precision_scientific_notation (mantissa : ℝ) (exponent : ℤ) : ℕ :=
  sorry

/-- The Great Wall's length in scientific notation -/
def great_wall_length : ℝ := 6.7

/-- The exponent in the scientific notation of the Great Wall's length -/
def great_wall_exponent : ℤ := 6

/-- Hundred thousands place value -/
def hundred_thousands : ℕ := 100000

theorem great_wall_precision :
  precision_scientific_notation great_wall_length great_wall_exponent = hundred_thousands :=
sorry

end NUMINAMATH_CALUDE_great_wall_precision_l308_30800


namespace NUMINAMATH_CALUDE_problem_solution_l308_30848

theorem problem_solution :
  ∀ n : ℤ, 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 6557 [ZMOD 7] → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l308_30848


namespace NUMINAMATH_CALUDE_sqrt_nested_square_l308_30891

theorem sqrt_nested_square : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_square_l308_30891


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l308_30831

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l308_30831


namespace NUMINAMATH_CALUDE_common_divisors_9240_10800_l308_30889

theorem common_divisors_9240_10800 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 10800) (Finset.range 10801)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10800_l308_30889


namespace NUMINAMATH_CALUDE_max_sum_given_constraint_l308_30887

theorem max_sum_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + (x+y)^3 + 36*x*y = 3456) : 
  x + y ≤ 12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^3 + y₀^3 + (x₀+y₀)^3 + 36*x₀*y₀ = 3456 ∧ x₀ + y₀ = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraint_l308_30887


namespace NUMINAMATH_CALUDE_coefficient_x_fourth_l308_30828

def binomial_coeff (n k : ℕ) : ℕ := sorry

def binomial_expansion_term (n r : ℕ) (a b : ℚ) : ℚ := sorry

theorem coefficient_x_fourth (n : ℕ) (h : n = 5) :
  ∃ (k : ℕ), binomial_coeff n k * (2 * k - 5) = 10 ∧
             binomial_expansion_term n k 1 1 = binomial_coeff n k * (2 * k - 5) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_fourth_l308_30828


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_decreases_as_base_increases_l308_30838

/-- Given an isosceles triangle with fixed side length and variable base length,
    the height is a decreasing function of the base length. -/
theorem isosceles_triangle_height_decreases_as_base_increases 
  (a : ℝ) (b : ℝ → ℝ) (h : ℝ → ℝ) :
  (∀ x, a > 0 ∧ b x > 0 ∧ h x > 0) →  -- Positive lengths
  (∀ x, a^2 = (h x)^2 + (b x)^2) →   -- Pythagorean theorem
  (∀ x, h x = Real.sqrt (a^2 - (b x)^2)) →  -- Height formula
  (∀ x y, x < y → b x < b y) →  -- b is increasing
  (∀ x y, x < y → h x > h y) :=  -- h is decreasing
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_height_decreases_as_base_increases_l308_30838


namespace NUMINAMATH_CALUDE_unique_divisible_by_1375_l308_30821

theorem unique_divisible_by_1375 : 
  ∃! n : ℕ, 
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = 700000 + 10000 * x + 3600 + 10 * y + 5) ∧ 
    n % 1375 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_1375_l308_30821


namespace NUMINAMATH_CALUDE_ashas_initial_savings_l308_30892

def borrowed_money : ℕ := 20 + 40 + 30
def gift_money : ℕ := 70
def remaining_money : ℕ := 65
def spending_fraction : ℚ := 3 / 4

theorem ashas_initial_savings :
  ∃ (initial_savings : ℕ),
    let total_money := initial_savings + borrowed_money + gift_money
    (total_money : ℚ) * (1 - spending_fraction) = remaining_money ∧
    initial_savings = 100 := by
  sorry

end NUMINAMATH_CALUDE_ashas_initial_savings_l308_30892


namespace NUMINAMATH_CALUDE_roof_weight_capacity_is_500_l308_30852

/-- The number of leaves that fall on Bill's roof each day -/
def leaves_per_day : ℕ := 100

/-- The number of leaves that weigh one pound -/
def leaves_per_pound : ℕ := 1000

/-- The number of days it takes for Bill's roof to collapse -/
def days_to_collapse : ℕ := 5000

/-- The weight Bill's roof can bear in pounds -/
def roof_weight_capacity : ℚ :=
  (leaves_per_day : ℚ) / (leaves_per_pound : ℚ) * days_to_collapse

theorem roof_weight_capacity_is_500 :
  roof_weight_capacity = 500 := by sorry

end NUMINAMATH_CALUDE_roof_weight_capacity_is_500_l308_30852


namespace NUMINAMATH_CALUDE_circumcircle_point_values_l308_30830

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (3, -1)

-- Define the equation of a circle
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the circumcircle of triangle ABC
def circumcircle (x y : ℝ) : Prop :=
  ∃ D E F : ℝ,
    circle_equation x y D E F ∧
    circle_equation A.1 A.2 D E F ∧
    circle_equation B.1 B.2 D E F ∧
    circle_equation C.1 C.2 D E F

-- Theorem statement
theorem circumcircle_point_values :
  ∀ a : ℝ, circumcircle a 2 → a = 2 ∨ a = 6 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_point_values_l308_30830


namespace NUMINAMATH_CALUDE_tan_two_beta_l308_30886

theorem tan_two_beta (α β : Real) 
  (h1 : Real.tan (α + β) = 1) 
  (h2 : Real.tan (α - β) = 7) : 
  Real.tan (2 * β) = -3/4 := by sorry

end NUMINAMATH_CALUDE_tan_two_beta_l308_30886


namespace NUMINAMATH_CALUDE_track_length_is_300_l308_30872

-- Define the track length
def track_length : ℝ := sorry

-- Define Brenda's distance to first meeting
def brenda_first_meeting : ℝ := 120

-- Define Sally's additional distance to second meeting
def sally_additional : ℝ := 180

-- Theorem statement
theorem track_length_is_300 :
  -- Conditions
  (brenda_first_meeting + (track_length - brenda_first_meeting) = track_length) ∧
  (brenda_first_meeting + brenda_first_meeting = 
   track_length - brenda_first_meeting + sally_additional) →
  -- Conclusion
  track_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_track_length_is_300_l308_30872


namespace NUMINAMATH_CALUDE_floor_sum_equals_n_l308_30814

theorem floor_sum_equals_n (n : ℤ) : 
  ⌊n / 2⌋ + ⌊(n + 1) / 2⌋ = n := by sorry

end NUMINAMATH_CALUDE_floor_sum_equals_n_l308_30814


namespace NUMINAMATH_CALUDE_family_reunion_weight_gain_l308_30851

/-- The total weight gain of three family members at a reunion -/
def total_weight_gain (orlando_gain jose_gain fernando_gain : ℝ) : ℝ :=
  orlando_gain + jose_gain + fernando_gain

theorem family_reunion_weight_gain :
  ∃ (orlando_gain jose_gain fernando_gain : ℝ),
    orlando_gain = 5 ∧
    jose_gain = 2 * orlando_gain + 2 ∧
    fernando_gain = (1/2 : ℝ) * jose_gain - 3 ∧
    total_weight_gain orlando_gain jose_gain fernando_gain = 20 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_weight_gain_l308_30851


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l308_30820

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l308_30820


namespace NUMINAMATH_CALUDE_number_added_to_x_l308_30835

theorem number_added_to_x (x : ℝ) (some_number : ℝ) : 
  x + some_number = 2 → x = 1 → some_number = 1 := by
sorry

end NUMINAMATH_CALUDE_number_added_to_x_l308_30835


namespace NUMINAMATH_CALUDE_senior_class_college_attendance_l308_30883

theorem senior_class_college_attendance 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (boys_not_attended_percent : ℚ) 
  (total_attended_percent : ℚ) :
  total_boys = 300 →
  total_girls = 240 →
  boys_not_attended_percent = 30 / 100 →
  total_attended_percent = 70 / 100 →
  (total_girls - (total_attended_percent * (total_boys + total_girls) - 
    (1 - boys_not_attended_percent) * total_boys)) / total_girls = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_senior_class_college_attendance_l308_30883


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_36_l308_30870

/-- The number of ways to assign 4 intern teachers to 3 classes, with at least 1 teacher in each class -/
def allocation_schemes : ℕ :=
  -- We define the number of allocation schemes here
  -- The actual calculation is not provided, as we're only writing the statement
  36

/-- Theorem stating that the number of allocation schemes is 36 -/
theorem allocation_schemes_eq_36 : allocation_schemes = 36 := by
  sorry


end NUMINAMATH_CALUDE_allocation_schemes_eq_36_l308_30870


namespace NUMINAMATH_CALUDE_rain_probability_l308_30816

theorem rain_probability (p_friday p_monday : ℝ) 
  (h1 : p_friday = 0.3)
  (h2 : p_monday = 0.6)
  (h3 : 0 ≤ p_friday ∧ p_friday ≤ 1)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1) :
  1 - (1 - p_friday) * (1 - p_monday) = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l308_30816


namespace NUMINAMATH_CALUDE_intersection_point_on_x_equals_4_l308_30815

/-- An ellipse with center at origin and foci on coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.equation (l : Line) (p : Point) : Prop :=
  p.y = l.k * (p.x - l.m)

/-- The main theorem -/
theorem intersection_point_on_x_equals_4 
  (e : Ellipse)
  (h_passes_through_A : e.equation ⟨-2, 0⟩)
  (h_passes_through_B : e.equation ⟨2, 0⟩)
  (h_passes_through_C : e.equation ⟨1, 3/2⟩)
  (l : Line)
  (h_k_nonzero : l.k ≠ 0)
  (M N : Point)
  (h_M_on_E : e.equation M)
  (h_N_on_E : e.equation N)
  (h_M_on_l : l.equation M)
  (h_N_on_l : l.equation N) :
  ∃ (P : Point), P.x = 4 ∧ 
    (∃ (t : ℝ), P = ⟨4, t * (M.y + 2) + (1 - t) * M.y⟩) ∧
    (∃ (s : ℝ), P = ⟨4, s * (N.y - 2) + (1 - s) * N.y⟩) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_x_equals_4_l308_30815


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l308_30898

theorem fraction_ratio_equality : ∃ x : ℚ, (x / (2/6) = (3/4) / (1/2)) ∧ (x = 2/9) := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l308_30898


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l308_30895

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on the floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let horizontal := (floor.length / tile.length) * (floor.width / tile.width)
  let vertical := (floor.length / tile.width) * (floor.width / tile.length)
  max horizontal vertical

/-- Theorem stating the maximum number of tiles that can be accommodated on the floor -/
theorem max_tiles_on_floor :
  let floor : Dimensions := ⟨390, 150⟩
  let tile : Dimensions := ⟨65, 25⟩
  maxTiles floor tile = 36 := by
  sorry

#eval maxTiles ⟨390, 150⟩ ⟨65, 25⟩

end NUMINAMATH_CALUDE_max_tiles_on_floor_l308_30895


namespace NUMINAMATH_CALUDE_max_r_value_exists_max_r_unique_max_r_l308_30834

open Set Real

/-- The set T parameterized by r -/
def T (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 7)^2 ≤ r^2}

/-- The set S -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ θ : ℝ, cos (2 * θ) + p.1 * cos θ + p.2 ≥ 0}

/-- The main theorem stating the maximum value of r -/
theorem max_r_value (r : ℝ) (h_pos : r > 0) (h_subset : T r ⊆ S) : r ≤ 4 * sqrt 2 := by
  sorry

/-- The existence of the maximum r value -/
theorem exists_max_r : ∃ r : ℝ, r > 0 ∧ T r ⊆ S ∧ ∀ s : ℝ, s > 0 ∧ T s ⊆ S → s ≤ r := by
  sorry

/-- The uniqueness of the maximum r value -/
theorem unique_max_r (r s : ℝ) (hr : r > 0) (hs : s > 0)
    (h_max_r : T r ⊆ S ∧ ∀ t : ℝ, t > 0 ∧ T t ⊆ S → t ≤ r)
    (h_max_s : T s ⊆ S ∧ ∀ t : ℝ, t > 0 ∧ T t ⊆ S → t ≤ s) : r = s := by
  sorry

end NUMINAMATH_CALUDE_max_r_value_exists_max_r_unique_max_r_l308_30834


namespace NUMINAMATH_CALUDE_circle_and_line_proof_l308_30837

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 5 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define point B
def point_B : ℝ × ℝ := (2, -2)

-- Define point D
def point_D : ℝ × ℝ := (-1, -1)

-- Define the line m (both possible equations)
def line_m (x y : ℝ) : Prop := x = -1 ∨ 3*x + 4*y + 7 = 0

theorem circle_and_line_proof :
  ∀ (x y : ℝ),
  (circle_C x y ↔ (x - point_A.1)^2 + (y - point_A.2)^2 = 25 ∧ 
                  (x - point_B.1)^2 + (y - point_B.2)^2 = 25) ∧
  (∃ (cx cy : ℝ), line_l cx cy ∧ circle_C cx cy) ∧
  (∃ (mx my : ℝ), line_m mx my ∧ point_D = (mx, my) ∧
    ∃ (x1 y1 x2 y2 : ℝ),
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      line_m x1 y1 ∧ line_m x2 y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 4 * 21) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_proof_l308_30837


namespace NUMINAMATH_CALUDE_probability_second_white_given_first_white_l308_30819

/-- The probability of drawing a white ball second, given that the first ball drawn is white,
    when there are 5 white balls and 4 black balls initially. -/
theorem probability_second_white_given_first_white :
  let total_balls : ℕ := 9
  let white_balls : ℕ := 5
  let black_balls : ℕ := 4
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_both_white : ℚ := (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))
  prob_both_white / prob_first_white = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_second_white_given_first_white_l308_30819


namespace NUMINAMATH_CALUDE_existence_of_numbers_l308_30817

theorem existence_of_numbers : ∃ n : ℕ, 
  70 ≤ n ∧ n ≤ 80 ∧ 
  Nat.gcd 30 n = 10 ∧ 
  200 < Nat.lcm 30 n ∧ Nat.lcm 30 n < 300 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_numbers_l308_30817


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l308_30854

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n*(x^4 + y^4 + z^4 + w^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m*(x^4 + y^4 + z^4 + w^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l308_30854


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l308_30811

theorem largest_multiple_of_seven_under_hundred : 
  ∃ (n : ℕ), n = 98 ∧ 
  7 ∣ n ∧ 
  n < 100 ∧ 
  ∀ (m : ℕ), 7 ∣ m → m < 100 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l308_30811


namespace NUMINAMATH_CALUDE_angle_around_point_l308_30875

/-- Given a point in a plane with four angles around it, where three of the angles are equal (x°) and the fourth is 140°, prove that x = 220/3. -/
theorem angle_around_point (x : ℚ) : 
  (3 * x + 140 = 360) → x = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l308_30875


namespace NUMINAMATH_CALUDE_derivative_x_exp_x_l308_30896

theorem derivative_x_exp_x (x : ℝ) : deriv (fun x => x * Real.exp x) x = (1 + x) * Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_exp_x_l308_30896


namespace NUMINAMATH_CALUDE_divisors_of_3b_plus_18_l308_30876

theorem divisors_of_3b_plus_18 (a b : ℤ) (h : 4 * b = 10 - 2 * a) :
  (∀ d : ℤ, d ∈ ({1, 2, 3, 6} : Set ℤ) → d ∣ (3 * b + 18)) ∧
  (∃ a b : ℤ, 4 * b = 10 - 2 * a ∧ (¬(4 ∣ (3 * b + 18)) ∨ ¬(5 ∣ (3 * b + 18)) ∨
                                   ¬(7 ∣ (3 * b + 18)) ∨ ¬(8 ∣ (3 * b + 18)))) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_3b_plus_18_l308_30876


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l308_30849

/-- Proves that adding 14.285714285714286 liters of pure alcohol to a 100-liter solution
    results in a 30% alcohol solution if and only if the initial alcohol percentage was 20% -/
theorem alcohol_solution_proof (initial_percentage : ℝ) : 
  (initial_percentage / 100) * 100 + 14.285714285714286 = 0.30 * (100 + 14.285714285714286) ↔ 
  initial_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l308_30849


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l308_30855

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 90 → speed2 = 55 → (speed1 + speed2) / 2 = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l308_30855


namespace NUMINAMATH_CALUDE_dividend_proof_l308_30893

theorem dividend_proof (divisor quotient remainder dividend : ℕ) : 
  divisor = 17 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  dividend = 159 :=
by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l308_30893


namespace NUMINAMATH_CALUDE_cone_base_radius_l308_30825

/-- Given a cone whose lateral surface unfolds into a semicircle with radius 4,
    prove that the radius of the base of the cone is also 4. -/
theorem cone_base_radius (r : ℝ) (h : r = 4) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l308_30825


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l308_30808

theorem not_necessarily_right_triangle 
  (a b c : ℝ) 
  (ha : a^2 = 5) 
  (hb : b^2 = 12) 
  (hc : c^2 = 13) : 
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l308_30808


namespace NUMINAMATH_CALUDE_existence_of_special_multiple_l308_30869

theorem existence_of_special_multiple (n : ℕ+) : 
  ∃ m : ℕ+, (m.val % n.val = 0) ∧ 
             (m.val ≤ n.val^2) ∧ 
             (∃ d : Fin 10, ∀ k : ℕ, (m.val / 10^k % 10) ≠ d.val) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_multiple_l308_30869


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l308_30873

/-- Calculates the time for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (tree_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1100) 
  (h2 : tree_passing_time = 110) 
  (h3 : platform_length = 700) : 
  (train_length + platform_length) / (train_length / tree_passing_time) = 180 :=
sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l308_30873


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l308_30843

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (35^87 + 93^49) % 10 = n :=
  by sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l308_30843


namespace NUMINAMATH_CALUDE_johns_next_birthday_l308_30897

/-- Represents the ages of John, Emily, and Lucas -/
structure Ages where
  john : ℝ
  emily : ℝ
  lucas : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.john = 1.25 * ages.emily ∧
  ages.emily = 0.7 * ages.lucas ∧
  ages.john + ages.emily + ages.lucas = 32

/-- The main theorem -/
theorem johns_next_birthday (ages : Ages) 
  (h : satisfies_conditions ages) : 
  ⌈ages.john⌉ = 11 := by
  sorry


end NUMINAMATH_CALUDE_johns_next_birthday_l308_30897


namespace NUMINAMATH_CALUDE_two_tangent_lines_l308_30801

-- Define the point P
def P : ℝ × ℝ := (-4, 1)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Define the condition for a line to intersect the hyperbola at only one point
def intersects_at_one_point (m : ℝ) : Prop :=
  ∃! (x y : ℝ), hyperbola x y ∧ line_through_P m x y

-- The main theorem
theorem two_tangent_lines :
  ∃! (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
    intersects_at_one_point m₁ ∧ 
    intersects_at_one_point m₂ ∧
    ∀ m, intersects_at_one_point m → m = m₁ ∨ m = m₂ :=
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l308_30801


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l308_30829

/-- The fixed point of the line (2k-1)x-(k+3)y-(k-11)=0 for all real k -/
theorem fixed_point_of_line (k : ℝ) : (2*k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l308_30829


namespace NUMINAMATH_CALUDE_f_min_max_l308_30890

-- Define the function f
def f (x y z : ℝ) : ℝ := x * y + y * z + z * x - 3 * x * y * z

-- State the theorem
theorem f_min_max :
  ∀ x y z : ℝ,
  x ≥ 0 → y ≥ 0 → z ≥ 0 →
  x + y + z = 1 →
  (0 ≤ f x y z) ∧ (f x y z ≤ 1/4) ∧
  (∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ f a b c = 0) ∧
  (∃ d e g : ℝ, d ≥ 0 ∧ e ≥ 0 ∧ g ≥ 0 ∧ d + e + g = 1 ∧ f d e g = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l308_30890


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l308_30840

/-- An arithmetic sequence with first term 3 and ninth term 27 has its thirtieth term equal to 90 -/
theorem arithmetic_sequence_30th_term : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 3 →                            -- first term is 3
  a 9 = 27 →                           -- ninth term is 27
  a 30 = 90 :=                         -- thirtieth term is 90
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l308_30840


namespace NUMINAMATH_CALUDE_johns_allowance_theorem_l308_30813

/-- The fraction of John's remaining allowance spent at the toy store -/
def toy_store_fraction (total_allowance : ℚ) (arcade_fraction : ℚ) (candy_amount : ℚ) : ℚ :=
  let remaining_after_arcade := total_allowance * (1 - arcade_fraction)
  let toy_store_amount := remaining_after_arcade - candy_amount
  toy_store_amount / remaining_after_arcade

/-- Proof that John spent 1/3 of his remaining allowance at the toy store -/
theorem johns_allowance_theorem :
  toy_store_fraction 3.60 (3/5) 0.96 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_theorem_l308_30813


namespace NUMINAMATH_CALUDE_smallest_with_ten_factors_l308_30804

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly ten distinct positive factors -/
def has_ten_factors (n : ℕ+) : Prop := num_factors n = 10

theorem smallest_with_ten_factors :
  ∃ (m : ℕ+), has_ten_factors m ∧ ∀ (k : ℕ+), has_ten_factors k → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_smallest_with_ten_factors_l308_30804


namespace NUMINAMATH_CALUDE_train_journey_time_l308_30806

/-- Calculate the total travel time for a train journey with multiple stops and varying speeds -/
theorem train_journey_time (d1 d2 d3 : ℝ) (v1 v2 v3 : ℝ) (t1 t2 : ℝ) :
  d1 = 30 →
  d2 = 40 →
  d3 = 50 →
  v1 = 60 →
  v2 = 40 →
  v3 = 80 →
  t1 = 10 / 60 →
  t2 = 5 / 60 →
  (d1 / v1 + t1 + d2 / v2 + t2 + d3 / v3) * 60 = 142.5 :=
by sorry

end NUMINAMATH_CALUDE_train_journey_time_l308_30806


namespace NUMINAMATH_CALUDE_number_difference_l308_30832

theorem number_difference (a b : ℕ) : 
  a + b = 26832 → 
  b % 10 = 0 → 
  a = b / 10 + 4 → 
  b - a = 21938 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l308_30832


namespace NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l308_30871

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 20) (h2 : 5 * b = 4 * a) : 
  Nat.lcm a b = 80 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l308_30871


namespace NUMINAMATH_CALUDE_set_equality_proof_l308_30894

theorem set_equality_proof (A B : Set α) (h : A ∩ B = A) : A ∪ B = B := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_l308_30894


namespace NUMINAMATH_CALUDE_different_course_selections_eq_30_l308_30888

/-- The number of ways two people can choose 2 courses each from 4 courses with at least one course different -/
def different_course_selections : ℕ :=
  Nat.choose 4 2 * Nat.choose 2 2 + Nat.choose 4 1 * Nat.choose 3 1 * Nat.choose 2 1

/-- Theorem stating that the number of different course selections is 30 -/
theorem different_course_selections_eq_30 : different_course_selections = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_course_selections_eq_30_l308_30888


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l308_30839

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℤ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def to_rat (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (999 : ℚ)

/-- The repeating decimal 0.428428... -/
def d1 : RepeatingDecimal := ⟨0, 428⟩

/-- The repeating decimal 2.857857... -/
def d2 : RepeatingDecimal := ⟨2, 857⟩

theorem repeating_decimal_fraction :
  (to_rat d1) / (to_rat d2) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l308_30839


namespace NUMINAMATH_CALUDE_corrected_mean_l308_30860

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) :
  n = 50 →
  incorrect_mean = 36 →
  incorrect_value = 21 →
  correct_value = 48 →
  (n : ℝ) * incorrect_mean - incorrect_value + correct_value = 36.54 * n :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l308_30860


namespace NUMINAMATH_CALUDE_A_and_D_independent_l308_30847

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A
def A : Set Ω := {ω | ω.1 = 0}

-- Define event D
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- Theorem statement
theorem A_and_D_independent :
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_and_D_independent_l308_30847


namespace NUMINAMATH_CALUDE_cm_to_m_sq_dm_to_sq_m_min_to_hour_g_to_kg_seven_cm_to_m_thirtyfive_sq_dm_to_sq_m_fortyfive_min_to_hour_twothousandfivehundred_g_to_kg_l308_30842

-- Define conversion rates
def cm_per_m : ℚ := 100
def sq_dm_per_sq_m : ℚ := 100
def min_per_hour : ℚ := 60
def g_per_kg : ℚ := 1000

-- Theorems to prove
theorem cm_to_m (x : ℚ) : x / cm_per_m = x / 100 := by sorry

theorem sq_dm_to_sq_m (x : ℚ) : x / sq_dm_per_sq_m = x / 100 := by sorry

theorem min_to_hour (x : ℚ) : x / min_per_hour = x / 60 := by sorry

theorem g_to_kg (x : ℚ) : x / g_per_kg = x / 1000 := by sorry

-- Specific conversions
theorem seven_cm_to_m : 7 / cm_per_m = 7 / 100 := by sorry

theorem thirtyfive_sq_dm_to_sq_m : 35 / sq_dm_per_sq_m = 7 / 20 := by sorry

theorem fortyfive_min_to_hour : 45 / min_per_hour = 3 / 4 := by sorry

theorem twothousandfivehundred_g_to_kg : 2500 / g_per_kg = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_cm_to_m_sq_dm_to_sq_m_min_to_hour_g_to_kg_seven_cm_to_m_thirtyfive_sq_dm_to_sq_m_fortyfive_min_to_hour_twothousandfivehundred_g_to_kg_l308_30842


namespace NUMINAMATH_CALUDE_morning_snowfall_l308_30882

/-- Given the total snowfall and afternoon snowfall in Yardley, 
    prove that the morning snowfall is the difference between them. -/
theorem morning_snowfall (total : ℝ) (afternoon : ℝ) 
  (h1 : total = 0.625) (h2 : afternoon = 0.5) : 
  total - afternoon = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_morning_snowfall_l308_30882


namespace NUMINAMATH_CALUDE_common_area_rectangle_ellipse_l308_30868

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents an ellipse with semi-major and semi-minor axis lengths -/
structure Ellipse where
  semiMajor : ℝ
  semiMinor : ℝ

/-- Calculates the area of the region common to a rectangle and an ellipse that share the same center -/
def commonArea (r : Rectangle) (e : Ellipse) : ℝ := sorry

theorem common_area_rectangle_ellipse :
  let r := Rectangle.mk 10 4
  let e := Ellipse.mk 3 2
  commonArea r e = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_common_area_rectangle_ellipse_l308_30868


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l308_30812

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 61 ∧ q ≠ 61 ∧ 
   x = 2 * p * q * 61 ∧ 
   ∀ r : ℕ, Prime r → r ∣ x → (r = 2 ∨ r = p ∨ r = q ∨ r = 61)) →
  x = 59048 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l308_30812


namespace NUMINAMATH_CALUDE_kats_training_hours_l308_30823

/-- The number of hours Kat trains per week -/
def total_training_hours (strength_sessions : ℕ) (strength_hours : ℝ) 
  (boxing_sessions : ℕ) (boxing_hours : ℝ) : ℝ :=
  (strength_sessions : ℝ) * strength_hours + (boxing_sessions : ℝ) * boxing_hours

/-- Theorem stating that Kat's total training hours per week is 9 -/
theorem kats_training_hours :
  total_training_hours 3 1 4 1.5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_kats_training_hours_l308_30823


namespace NUMINAMATH_CALUDE_triangle_side_length_l308_30899

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  a = 2 * b →
  b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l308_30899


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l308_30859

theorem diophantine_equation_solutions (a b c : ℤ) :
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 ↔
  ((a = 3 ∧ b = 3 ∧ c = 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 6) ∨
   (a = 2 ∧ b = 4 ∧ c = 4) ∨
   (∃ t : ℤ, a = 1 ∧ b = t ∧ c = -t)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l308_30859


namespace NUMINAMATH_CALUDE_reservoir_refill_rate_l308_30862

theorem reservoir_refill_rate 
  (V : ℝ) (R : ℝ) 
  (h1 : V - 90 * (40000 - R) = 0) 
  (h2 : V - 60 * (32000 - R) = 0) : 
  R = 56000 := by
sorry

end NUMINAMATH_CALUDE_reservoir_refill_rate_l308_30862


namespace NUMINAMATH_CALUDE_correct_number_of_vans_l308_30884

/-- The number of vans taken on a field trip -/
def number_of_vans : ℕ := 2

/-- The total number of people on the field trip -/
def total_people : ℕ := 76

/-- The number of buses taken on the field trip -/
def number_of_buses : ℕ := 3

/-- The number of people each bus can hold -/
def people_per_bus : ℕ := 20

/-- The number of people each van can hold -/
def people_per_van : ℕ := 8

/-- Theorem stating that the number of vans is correct given the conditions -/
theorem correct_number_of_vans : 
  number_of_vans * people_per_van + number_of_buses * people_per_bus = total_people :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_vans_l308_30884


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l308_30844

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 7) :
  Real.cos (π - α) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l308_30844


namespace NUMINAMATH_CALUDE_field_trip_capacity_l308_30865

theorem field_trip_capacity (seats_per_bus : ℕ) (num_buses : ℕ) : 
  let max_students := seats_per_bus * num_buses
  seats_per_bus = 60 → num_buses = 3 → max_students = 180 := by
sorry

end NUMINAMATH_CALUDE_field_trip_capacity_l308_30865


namespace NUMINAMATH_CALUDE_unique_coefficient_exists_l308_30866

theorem unique_coefficient_exists (x y : ℝ) 
  (eq1 : 4 * x + y = 8) 
  (eq2 : 3 * x - 4 * y = 5) : 
  ∃! a : ℝ, a * x - 3 * y = 23 := by
sorry

end NUMINAMATH_CALUDE_unique_coefficient_exists_l308_30866


namespace NUMINAMATH_CALUDE_monicas_first_class_size_l308_30807

/-- Represents the number of students in Monica's classes -/
structure MonicasClasses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- Theorem stating the number of students in Monica's first class -/
theorem monicas_first_class_size (c : MonicasClasses) : c.first = 20 :=
  by
  have h1 : c.second = 25 := by sorry
  have h2 : c.third = 25 := by sorry
  have h3 : c.fourth = c.first / 2 := by sorry
  have h4 : c.fifth = 28 := by sorry
  have h5 : c.sixth = 28 := by sorry
  have h6 : c.first + c.second + c.third + c.fourth + c.fifth + c.sixth = 136 := by sorry
  sorry

#check monicas_first_class_size

end NUMINAMATH_CALUDE_monicas_first_class_size_l308_30807


namespace NUMINAMATH_CALUDE_sequence_monotonicity_l308_30824

/-- A sequence a_n is defined as a_n = -n^2 + tn, where n is a positive natural number and t is a constant real number. The sequence is monotonically decreasing. -/
theorem sequence_monotonicity (t : ℝ) : 
  (∀ n : ℕ+, ∀ m : ℕ+, n < m → (-n^2 + t * n) > (-m^2 + t * m)) → 
  t < 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_monotonicity_l308_30824


namespace NUMINAMATH_CALUDE_daughters_age_in_three_years_l308_30846

/-- Given that 5 years ago, a mother was twice as old as her daughter, and the mother is 41 years old now,
    prove that the daughter will be 26 years old in 3 years. -/
theorem daughters_age_in_three_years 
  (mother_age_now : ℕ) 
  (mother_daughter_age_relation : ℕ → ℕ → Prop) 
  (h1 : mother_age_now = 41)
  (h2 : mother_daughter_age_relation (mother_age_now - 5) ((mother_age_now - 5) / 2)) :
  ((mother_age_now - 5) / 2) + 8 = 26 := by
  sorry

#check daughters_age_in_three_years

end NUMINAMATH_CALUDE_daughters_age_in_three_years_l308_30846


namespace NUMINAMATH_CALUDE_poached_percentage_less_than_sold_l308_30845

def total_pears : ℕ := 42
def sold_pears : ℕ := 20

def canned_pears (poached : ℕ) : ℕ := poached + poached / 5

theorem poached_percentage_less_than_sold :
  ∃ (poached : ℕ),
    poached > 0 ∧
    poached < sold_pears ∧
    total_pears = sold_pears + canned_pears poached + poached ∧
    (sold_pears - poached) * 100 / sold_pears = 50 := by
  sorry

end NUMINAMATH_CALUDE_poached_percentage_less_than_sold_l308_30845


namespace NUMINAMATH_CALUDE_total_cost_european_stamps_50s_60s_l308_30856

-- Define the cost of stamps
def italy_stamp_cost : ℚ := 0.07
def germany_stamp_cost : ℚ := 0.03

-- Define the number of stamps collected
def italy_stamps_50s_60s : ℕ := 9
def germany_stamps_50s_60s : ℕ := 15

-- Theorem statement
theorem total_cost_european_stamps_50s_60s : 
  (italy_stamp_cost * italy_stamps_50s_60s + germany_stamp_cost * germany_stamps_50s_60s : ℚ) = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_european_stamps_50s_60s_l308_30856


namespace NUMINAMATH_CALUDE_tan_two_simplification_l308_30858

theorem tan_two_simplification (x : ℝ) (h : Real.tan x = 2) :
  (2 * Real.sin x + Real.cos x) / (2 * Real.sin x - Real.cos x) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_simplification_l308_30858


namespace NUMINAMATH_CALUDE_quadratic_properties_l308_30810

def f (x : ℝ) := x^2 - 4*x + 6

theorem quadratic_properties :
  (∀ x : ℝ, f x = 2 ↔ x = 2) ∧
  (∀ x y : ℝ, x > 2 ∧ y > x → f y > f x) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m) ∧
  (∀ x : ℝ, f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l308_30810


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l308_30867

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ) (α : ℝ),
    0 < α ∧ α < Real.pi / 2 ∧
    (∃ (r : ℝ),
      Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧
      Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
      Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))) ∧
    (∀ (t' : ℝ) (α' : ℝ),
      0 < α' ∧ α' < Real.pi / 2 →
      (∃ (r' : ℝ),
        Real.arcsin (Real.sin α') * r' = Real.arcsin (Real.sin (3 * α')) ∧
        Real.arcsin (Real.sin (3 * α')) * r' = Real.arcsin (Real.sin (5 * α')) ∧
        Real.arcsin (Real.sin (5 * α')) * r' = Real.arcsin (Real.sin (t' * α'))) →
      t ≤ t') ∧
    t = 27 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l308_30867


namespace NUMINAMATH_CALUDE_no_solution_x4_plus_6_eq_y3_l308_30809

theorem no_solution_x4_plus_6_eq_y3 :
  ∀ (x y : ℤ), (x^4 + 6) % 13 ≠ y^3 % 13 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_x4_plus_6_eq_y3_l308_30809


namespace NUMINAMATH_CALUDE_cost_increase_l308_30857

/-- Cost function -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

theorem cost_increase (t : ℝ) (b₀ b₁ : ℝ) (h : t > 0) :
  cost t b₁ = 16 * cost t b₀ → b₁ = 2 * b₀ := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_l308_30857


namespace NUMINAMATH_CALUDE_sophie_total_spend_l308_30853

def cupcake_quantity : ℕ := 5
def cupcake_price : ℚ := 2

def doughnut_quantity : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_quantity : ℕ := 4
def apple_pie_price : ℚ := 2

def cookie_quantity : ℕ := 15
def cookie_price : ℚ := 0.6

theorem sophie_total_spend :
  (cupcake_quantity : ℚ) * cupcake_price +
  (doughnut_quantity : ℚ) * doughnut_price +
  (apple_pie_quantity : ℚ) * apple_pie_price +
  (cookie_quantity : ℚ) * cookie_price = 33 := by
  sorry

end NUMINAMATH_CALUDE_sophie_total_spend_l308_30853


namespace NUMINAMATH_CALUDE_solution_sum_l308_30836

theorem solution_sum (P q : ℝ) : 
  (2^2 - P*2 + 6 = 0) → (2^2 + 6*2 - q = 0) → P + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l308_30836


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l308_30874

/-- Represents a pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {17, 23, 26, 29, 35}

/-- The area of a CornerCutPentagon is 895 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  895

/-- The area of a CornerCutPentagon is correct -/
theorem corner_cut_pentagon_area_is_correct (p : CornerCutPentagon) :
  corner_cut_pentagon_area p = 895 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l308_30874


namespace NUMINAMATH_CALUDE_snowflake_four_two_l308_30805

-- Define the snowflake operation
def snowflake (a b : ℕ) : ℕ := a * (b - 1) + a * b

-- Theorem statement
theorem snowflake_four_two : snowflake 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_snowflake_four_two_l308_30805


namespace NUMINAMATH_CALUDE_min_jugs_to_fill_container_l308_30880

/-- The capacity of a regular water jug in milliliters -/
def regular_jug_capacity : ℕ := 300

/-- The capacity of a giant water container in milliliters -/
def giant_container_capacity : ℕ := 1800

/-- The minimum number of regular jugs needed to fill a giant container -/
def min_jugs_needed : ℕ := giant_container_capacity / regular_jug_capacity

theorem min_jugs_to_fill_container : min_jugs_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_jugs_to_fill_container_l308_30880


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l308_30803

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 6 ∧
  θ = 5 * π / 3 ∧
  z = 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l308_30803


namespace NUMINAMATH_CALUDE_tadpoles_kept_l308_30818

theorem tadpoles_kept (total : ℕ) (released_percent : ℚ) (kept : ℕ) : 
  total = 180 → 
  released_percent = 75 / 100 → 
  kept = total - (total * released_percent).floor → 
  kept = 45 := by
sorry

end NUMINAMATH_CALUDE_tadpoles_kept_l308_30818


namespace NUMINAMATH_CALUDE_expression_simplification_l308_30841

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l308_30841


namespace NUMINAMATH_CALUDE_alice_second_test_study_time_l308_30850

/-- Represents the relationship between study time and test score -/
def study_score_product (study_time : ℝ) (score : ℝ) : ℝ := study_time * score

/-- Alice's first test data -/
def first_test_time : ℝ := 2
def first_test_score : ℝ := 60

/-- Target average score -/
def target_average : ℝ := 75

/-- Theorem: Alice needs to study 4/3 hours for her second test -/
theorem alice_second_test_study_time :
  ∃ (second_test_time : ℝ),
    second_test_time > 0 ∧
    study_score_product first_test_time first_test_score = study_score_product second_test_time ((target_average * 2) - first_test_score) ∧
    second_test_time = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_alice_second_test_study_time_l308_30850


namespace NUMINAMATH_CALUDE_hostel_expenditure_increase_l308_30833

/-- Calculates the increase in total expenditure for a hostel after accommodating more students. -/
theorem hostel_expenditure_increase
  (initial_students : ℕ)
  (additional_students : ℕ)
  (average_decrease : ℚ)
  (new_total_expenditure : ℚ)
  (h1 : initial_students = 100)
  (h2 : additional_students = 20)
  (h3 : average_decrease = 5)
  (h4 : new_total_expenditure = 5400) :
  let total_students := initial_students + additional_students
  let new_average := new_total_expenditure / total_students
  let original_average := new_average + average_decrease
  let original_total_expenditure := original_average * initial_students
  new_total_expenditure - original_total_expenditure = 400 :=
by sorry

end NUMINAMATH_CALUDE_hostel_expenditure_increase_l308_30833


namespace NUMINAMATH_CALUDE_no_positive_solutions_l308_30861

theorem no_positive_solutions :
  ¬∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^3 + y^3 + z^3 = x + y + z ∧
  x^2 + y^2 + z^2 = x*y*z :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solutions_l308_30861


namespace NUMINAMATH_CALUDE_fib_recurrence_l308_30885

/-- Fibonacci sequence defined as the number of ways to represent n as an ordered sum of ones and twos -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The Fibonacci sequence satisfies the recurrence relation F_n = F_{n-1} + F_{n-2} for n ≥ 2 -/
theorem fib_recurrence (n : ℕ) (h : n ≥ 2) : fib n = fib (n - 1) + fib (n - 2) := by
  sorry

#check fib_recurrence

end NUMINAMATH_CALUDE_fib_recurrence_l308_30885


namespace NUMINAMATH_CALUDE_simplify_expression_l308_30802

theorem simplify_expression (x : ℝ) : 8*x - 3 + 2*x - 7 + 4*x + 15 = 14*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l308_30802


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l308_30878

/-- Represents a square tile pattern -/
structure TilePattern where
  side : ℕ
  black_tiles : ℕ
  white_tiles : ℕ

/-- Creates an extended pattern by adding a border of black tiles -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2,
    black_tiles := p.black_tiles + 4 * p.side + 4,
    white_tiles := p.white_tiles }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern)
  (h1 : p.side * p.side = p.black_tiles + p.white_tiles)
  (h2 : p.black_tiles = 12)
  (h3 : p.white_tiles = 24) :
  let extended := extend_pattern p
  (extended.black_tiles : ℚ) / extended.white_tiles = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l308_30878


namespace NUMINAMATH_CALUDE_shortest_tangent_is_sqrt_449_l308_30822

/-- Circle C₁ with center (8, 3) and radius 7 -/
def C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 8)^2 + (p.2 - 3)^2 = 49}

/-- Circle C₂ with center (-12, -4) and radius 5 -/
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 12)^2 + (p.2 + 4)^2 = 25}

/-- The length of the shortest line segment PQ tangent to C₁ at P and C₂ at Q -/
def shortest_tangent_length (C₁ C₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the shortest tangent length between C₁ and C₂ is √449 -/
theorem shortest_tangent_is_sqrt_449 : 
  shortest_tangent_length C₁ C₂ = Real.sqrt 449 := by sorry

end NUMINAMATH_CALUDE_shortest_tangent_is_sqrt_449_l308_30822


namespace NUMINAMATH_CALUDE_complex_product_real_l308_30864

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := 2 + b * I
  (z₁ * z₂).im = 0 → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l308_30864


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l308_30881

def container_X : ℕ × ℕ := (7, 3)  -- (blue balls, yellow balls)
def container_Y : ℕ × ℕ := (5, 5)
def container_Z : ℕ × ℕ := (8, 2)

def total_balls (c : ℕ × ℕ) : ℕ := c.1 + c.2

def prob_yellow (c : ℕ × ℕ) : ℚ := c.2 / (total_balls c)

def prob_container : ℚ := 1 / 3

theorem yellow_ball_probability :
  prob_container * prob_yellow container_X +
  prob_container * prob_yellow container_Y +
  prob_container * prob_yellow container_Z = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l308_30881


namespace NUMINAMATH_CALUDE_complex_equation_solution_l308_30826

theorem complex_equation_solution (z : ℂ) :
  (1 + 2 * z) / (1 - z) = Complex.I → z = -1/5 + 3/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l308_30826


namespace NUMINAMATH_CALUDE_circumscribing_circle_diameter_l308_30877

/-- The diameter of a circle circumscribing six equal, mutually tangent circles -/
theorem circumscribing_circle_diameter (r : ℝ) (h : r = 4) : 
  let small_circle_radius : ℝ := r
  let small_circles_count : ℕ := 6
  let large_circle_diameter : ℝ := 2 * (2 * small_circle_radius + small_circle_radius)
  large_circle_diameter = 24 := by sorry

end NUMINAMATH_CALUDE_circumscribing_circle_diameter_l308_30877
