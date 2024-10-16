import Mathlib

namespace NUMINAMATH_CALUDE_berry_difference_change_l1869_186998

/-- Represents the number of berries in a box -/
structure Berry where
  count : ℕ

/-- Represents a box of berries -/
inductive Box
  | Red : Berry → Box
  | Blue : Berry → Box

/-- The problem setup -/
structure BerryProblem where
  blue_berry_count : ℕ
  red_berry_count : ℕ
  berry_increase : ℕ
  blue_box_count : ℕ
  red_box_count : ℕ

/-- The theorem to prove -/
theorem berry_difference_change (problem : BerryProblem) 
  (h1 : problem.blue_berry_count = 36)
  (h2 : problem.red_berry_count = problem.blue_berry_count + problem.berry_increase)
  (h3 : problem.berry_increase = 15) :
  problem.red_berry_count - problem.blue_berry_count = 15 := by
  sorry

#check berry_difference_change

end NUMINAMATH_CALUDE_berry_difference_change_l1869_186998


namespace NUMINAMATH_CALUDE_six_points_fifteen_segments_l1869_186985

/-- The number of line segments formed by connecting n distinct points on a circle --/
def lineSegments (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 6 distinct points on a circle, the number of line segments is 15 --/
theorem six_points_fifteen_segments : lineSegments 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_six_points_fifteen_segments_l1869_186985


namespace NUMINAMATH_CALUDE_largest_negative_integer_solution_l1869_186965

theorem largest_negative_integer_solution :
  ∃ (x : ℝ), x = -1 ∧ 
  x < 0 ∧
  |x - 1| > 1 ∧
  (x - 2) / x > 0 ∧
  (x - 2) / x > |x - 1| ∧
  ∀ (y : ℤ), y < 0 → 
    (y < x ∨ ¬(|y - 1| > 1) ∨ ¬((y - 2) / y > 0) ∨ ¬((y - 2) / y > |y - 1|)) :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_solution_l1869_186965


namespace NUMINAMATH_CALUDE_speed_difference_meeting_l1869_186928

/-- The difference in speed between two travelers meeting at a point -/
theorem speed_difference_meeting (distance : ℝ) (time : ℝ) (speed_enrique : ℝ) (speed_jamal : ℝ)
  (h1 : distance = 200)  -- Total distance between Enrique and Jamal
  (h2 : time = 8)        -- Time taken to meet
  (h3 : speed_enrique = 16)  -- Enrique's speed
  (h4 : speed_jamal = 23)    -- Jamal's speed
  (h5 : distance = (speed_enrique + speed_jamal) * time)  -- Distance traveled equals total speed times time
  : speed_jamal - speed_enrique = 7 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_meeting_l1869_186928


namespace NUMINAMATH_CALUDE_triangle_constant_sum_squares_l1869_186956

/-- Given a triangle XYZ where YZ = 10 and the length of median XM is 7,
    the value of XZ^2 + XY^2 is constant. -/
theorem triangle_constant_sum_squares (X Y Z M : ℝ × ℝ) :
  let d (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (d Y Z = 10) →
  (M = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2)) →
  (d X M = 7) →
  ∃ (c : ℝ), ∀ (X' : ℝ × ℝ), d X' M = 7 → (d X' Y)^2 + (d X' Z)^2 = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_constant_sum_squares_l1869_186956


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1869_186954

-- Define the function f(x) = x^4 - 2x^3
def f (x : ℝ) : ℝ := x^4 - 2*x^3

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 4*x^3 - 6*x^2

-- Theorem: The equation of the tangent line to f(x) at x = 1 is y = -2x + 1
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1869_186954


namespace NUMINAMATH_CALUDE_revenue_change_l1869_186915

/-- The effect on revenue receipts given price and demand changes -/
theorem revenue_change 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ) 
  (seasonal_discount : ℝ) 
  (competition_decrease : ℝ) 
  (h1 : price_increase = 0.5) 
  (h2 : volume_decrease = 0.2) 
  (h3 : seasonal_discount = 0.1) 
  (h4 : competition_decrease = 0.05) : 
  let new_price := initial_price * (1 + price_increase) * (1 - seasonal_discount)
  let new_volume := initial_volume * (1 - volume_decrease) * (1 - competition_decrease)
  new_price * new_volume = 1.026 * initial_price * initial_volume := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_l1869_186915


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l1869_186916

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- Theorem: A ball dropped from 120 feet, rebounding 1/3 of its fall distance each time,
    will have traveled 5000/27 feet when it hits the ground for the fifth time. -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 5000/27 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l1869_186916


namespace NUMINAMATH_CALUDE_ribbon_leftover_l1869_186978

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) :
  total_ribbon = 18 ∧ num_gifts = 6 ∧ ribbon_per_gift = 2 →
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
sorry

end NUMINAMATH_CALUDE_ribbon_leftover_l1869_186978


namespace NUMINAMATH_CALUDE_octagon_coloring_count_l1869_186981

/-- The number of disks in the octagonal pattern -/
def num_disks : ℕ := 8

/-- The number of blue disks -/
def num_blue : ℕ := 3

/-- The number of red disks -/
def num_red : ℕ := 3

/-- The number of green disks -/
def num_green : ℕ := 2

/-- The symmetry group of a regular octagon -/
def octagon_symmetry_group_order : ℕ := 16

/-- The number of distinct colorings considering symmetries -/
def distinct_colorings : ℕ := 43

/-- Theorem stating the number of distinct colorings -/
theorem octagon_coloring_count :
  let total_colorings := (Nat.choose num_disks num_blue) * (Nat.choose (num_disks - num_blue) num_red)
  (total_colorings / octagon_symmetry_group_order : ℚ).num = distinct_colorings := by
  sorry

end NUMINAMATH_CALUDE_octagon_coloring_count_l1869_186981


namespace NUMINAMATH_CALUDE_cubic_polynomials_l1869_186929

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 + 10
def B (x e f : ℝ) : ℝ := x^2 + e * x + f

-- Define the alternative form of A
def A_alt (x a b c d : ℝ) : ℝ := a * (x - 1)^3 + b * (x - 1)^2 + c * (x - 1) + d

-- State the theorem
theorem cubic_polynomials (a b c d e f : ℝ) (hf : f ≠ 0) (he : e ≠ 0) :
  (∀ x, A x = A_alt x a b c d) →
  (∀ x, ∃ k₁ k₂ k₃, A x + B x e f = k₁ * x^3 + k₂ * x^2 + k₃ * x + (10 + f)) →
  (a + b + c = 17) ∧
  (∃ x₀, ∀ x, B x e f = 0 ↔ x = x₀) →
  (f = -10 ∧ a + b + c = 17 ∧ e^2 = 4 * f) := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_l1869_186929


namespace NUMINAMATH_CALUDE_no_integer_solution_implies_k_range_l1869_186984

theorem no_integer_solution_implies_k_range (k : ℝ) : 
  (∀ x : ℤ, ¬((k * x - k^2 - 4) * (x - 4) < 0)) → 
  1 ≤ k ∧ k ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_no_integer_solution_implies_k_range_l1869_186984


namespace NUMINAMATH_CALUDE_shaun_age_l1869_186927

/-- Represents the current ages of Kay, Gordon, and Shaun --/
structure Ages where
  kay : ℕ
  gordon : ℕ
  shaun : ℕ

/-- Checks if the given ages satisfy the conditions of the problem --/
def satisfiesConditions (ages : Ages) : Prop :=
  (ages.kay + 4 = 2 * (ages.gordon + 4)) ∧
  (ages.shaun + 8 = 2 * (ages.kay + 8)) ∧
  (ages.shaun + 12 = 3 * (ages.gordon + 12))

/-- Theorem stating that if the ages satisfy the conditions, then Shaun's current age is 48 --/
theorem shaun_age (ages : Ages) :
  satisfiesConditions ages → ages.shaun = 48 := by sorry

end NUMINAMATH_CALUDE_shaun_age_l1869_186927


namespace NUMINAMATH_CALUDE_fraction_equality_l1869_186935

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x - 3*y) / (x + 4*y) = 3) : 
  (x - 4*y) / (4*x + 3*y) = 11/63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1869_186935


namespace NUMINAMATH_CALUDE_part_one_part_two_l1869_186950

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Part 1: Prove that if A = 30°, then a = 5/3
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_A : t.A = 30 * π / 180) :
  t.a = 5/3 := by sorry

-- Part 2: Prove that the maximum area of the triangle is 3
theorem part_two (t : Triangle) (h : triangle_conditions t) :
  (∃ (max_area : ℝ), max_area = 3 ∧ 
    ∀ (t' : Triangle), triangle_conditions t' → 
      1/2 * t'.a * t'.c * Real.sin t'.B ≤ max_area) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1869_186950


namespace NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l1869_186931

-- Define set A
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Theorem for the first part
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

-- Theorem for the second part
theorem union_equals_B (a : ℝ) : A a ∪ B = B ↔ a > 5 ∨ a < -4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l1869_186931


namespace NUMINAMATH_CALUDE_solve_for_a_l1869_186918

theorem solve_for_a (a : ℝ) :
  (∀ x, |2*x - a| + a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1869_186918


namespace NUMINAMATH_CALUDE_joan_missed_games_l1869_186986

/-- The number of baseball games Joan missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Theorem stating that Joan missed 469 games -/
theorem joan_missed_games :
  let total_games : ℕ := 864
  let attended_games : ℕ := 395
  games_missed total_games attended_games = 469 := by
  sorry

end NUMINAMATH_CALUDE_joan_missed_games_l1869_186986


namespace NUMINAMATH_CALUDE_limit_rational_function_l1869_186957

theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 3| ∧ |x - 3| < δ → 
    |((x^6 - 54*x^3 + 729) / (x^3 - 27)) - 0| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l1869_186957


namespace NUMINAMATH_CALUDE_fraction_equality_l1869_186988

theorem fraction_equality (a b c d : ℚ) 
  (h1 : b / a = 1 / 2)
  (h2 : d / c = 1 / 2)
  (h3 : a ≠ c) :
  (2 * b - d) / (2 * a - c) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1869_186988


namespace NUMINAMATH_CALUDE_division_problem_l1869_186905

theorem division_problem (A : ℕ) : A = 1 → 23 = 13 * A + 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1869_186905


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1869_186934

theorem geometric_sequence_middle_term (a : ℝ) : 
  (∃ r : ℝ, 2 * r = a ∧ a * r = 8) → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1869_186934


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1869_186959

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+3) = (8 : ℝ)^(3*x+4) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1869_186959


namespace NUMINAMATH_CALUDE_seating_arrangement_l1869_186924

theorem seating_arrangement (total_people : ℕ) (total_rows : ℕ) 
  (h1 : total_people = 97) 
  (h2 : total_rows = 13) : 
  ∃ (rows_with_8 : ℕ), 
    rows_with_8 * 8 + (total_rows - rows_with_8) * 7 = total_people ∧ 
    rows_with_8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l1869_186924


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l1869_186962

theorem purely_imaginary_z (α : ℝ) :
  let z : ℂ := Complex.mk (Real.sin α) (-(1 - Real.cos α))
  z.re = 0 → ∃ k : ℤ, α = (2 * k + 1) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l1869_186962


namespace NUMINAMATH_CALUDE_car_speed_proof_l1869_186990

/-- Proves that a car's speed is 36 km/h given the conditions of the problem -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v) * 3600 = (1 / 40) * 3600 + 10 → v = 36 :=
by
  sorry

#check car_speed_proof

end NUMINAMATH_CALUDE_car_speed_proof_l1869_186990


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1869_186912

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 9) : Real.tan α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1869_186912


namespace NUMINAMATH_CALUDE_bacteria_growth_l1869_186917

-- Define the division rate of bacteria
def division_rate : ℕ := 10

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 1

-- Define the time passed in minutes
def time_passed : ℕ := 120

-- Define the function to calculate the number of bacteria
def num_bacteria (t : ℕ) : ℕ := 2 ^ (t / division_rate)

-- Theorem to prove
theorem bacteria_growth :
  num_bacteria time_passed = 2^12 :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1869_186917


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1869_186914

/-- A parallelogram with a diagonal and its perpendicular -/
structure Parallelogram where
  diagonal : ℝ
  perpendicular : ℝ

/-- The area of a parallelogram given its diagonal and perpendicular -/
def area (p : Parallelogram) : ℝ := p.diagonal * p.perpendicular

/-- Theorem: A parallelogram with diagonal 30 and perpendicular 20 has area 600 -/
theorem parallelogram_area_example : 
  let p : Parallelogram := { diagonal := 30, perpendicular := 20 }
  area p = 600 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1869_186914


namespace NUMINAMATH_CALUDE_binomial_15_4_l1869_186966

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l1869_186966


namespace NUMINAMATH_CALUDE_factor_expression_l1869_186945

theorem factor_expression (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1869_186945


namespace NUMINAMATH_CALUDE_students_above_90_l1869_186992

/-- Represents a normal distribution of test scores -/
structure ScoreDistribution where
  mean : ℝ
  variance : ℝ
  is_normal : Bool

/-- Represents the class and score information -/
structure ClassScores where
  total_students : ℕ
  distribution : ScoreDistribution
  between_mean_and_plus_10 : ℕ

/-- Theorem stating the number of students scoring above 90 -/
theorem students_above_90 (c : ClassScores) 
  (h1 : c.total_students = 48)
  (h2 : c.distribution.mean = 80)
  (h3 : c.distribution.is_normal = true)
  (h4 : c.between_mean_and_plus_10 = 16) :
  c.total_students / 2 - c.between_mean_and_plus_10 = 8 := by
  sorry


end NUMINAMATH_CALUDE_students_above_90_l1869_186992


namespace NUMINAMATH_CALUDE_max_cone_radius_in_crate_l1869_186944

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Checks if a cone fits upright in a crate -/
def fitsInCrate (cone : Cone) (crate : CrateDimensions) : Prop :=
  cone.height ≤ max crate.length (max crate.width crate.height) ∧
  2 * cone.radius ≤ min crate.length (min crate.width crate.height)

/-- The theorem stating the maximum radius of a cone that fits in the given crate -/
theorem max_cone_radius_in_crate :
  ∃ (maxRadius : ℝ),
    maxRadius = 2.5 ∧
    ∀ (c : Cone),
      fitsInCrate c (CrateDimensions.mk 5 8 12) →
      c.radius ≤ maxRadius :=
sorry

end NUMINAMATH_CALUDE_max_cone_radius_in_crate_l1869_186944


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1869_186999

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

structure Point := (coords : ℝ × ℝ)

def on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p.coords
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def diametrically_opposite (p1 p2 : Point) (c : Circle) : Prop :=
  let (x1, y1) := p1.coords
  let (x2, y2) := p2.coords
  let (cx, cy) := c.center
  (x1 - cx)^2 + (y1 - cy)^2 = c.radius^2 ∧
  (x2 - cx)^2 + (y2 - cy)^2 = c.radius^2 ∧
  (x1 - x2)^2 + (y1 - y2)^2 = 4 * c.radius^2

def angle (p1 p2 p3 : Point) : ℝ := sorry

theorem circle_intersection_theorem (c : Circle) (A B C D M N : Point) :
  on_circle A c ∧ on_circle B c ∧ on_circle C c ∧ on_circle D c →
  (∃ t : ℝ, A.coords = B.coords + t • (C.coords - D.coords)) →
  (∃ s : ℝ, A.coords = D.coords + s • (B.coords - C.coords)) →
  angle B M C = angle C N D ↔
  diametrically_opposite A C c ∨ diametrically_opposite B D c :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1869_186999


namespace NUMINAMATH_CALUDE_two_p_plus_q_l1869_186979

theorem two_p_plus_q (p q : ℚ) (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l1869_186979


namespace NUMINAMATH_CALUDE_smallest_cube_ending_632_l1869_186940

theorem smallest_cube_ending_632 :
  ∃ n : ℕ+, (n : ℤ)^3 ≡ 632 [ZMOD 1000] ∧
  ∀ m : ℕ+, (m : ℤ)^3 ≡ 632 [ZMOD 1000] → n ≤ m ∧ n = 192 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_632_l1869_186940


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l1869_186921

theorem sphere_surface_area_from_cube (a : ℝ) (h : a > 0) :
  ∃ (cube_edge : ℝ) (sphere_radius : ℝ),
    cube_edge > 0 ∧
    sphere_radius > 0 ∧
    (6 * cube_edge ^ 2 = a) ∧
    (cube_edge * Real.sqrt 3 = 2 * sphere_radius) ∧
    (4 * π * sphere_radius ^ 2 = π / 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l1869_186921


namespace NUMINAMATH_CALUDE_total_earnings_proof_l1869_186967

/-- Represents a work day with various attributes -/
structure WorkDay where
  regular_hours : ℝ
  night_shift_hours : ℝ
  overtime_hours : ℝ
  weekend_hours : ℝ
  sales : ℝ

/-- Calculates total earnings for two weeks given work conditions -/
def calculate_total_earnings (
  last_week_hours : ℝ)
  (last_week_rate : ℝ)
  (regular_rate_increase : ℝ)
  (overtime_multiplier : ℝ)
  (weekend_multiplier : ℝ)
  (night_shift_multiplier : ℝ)
  (commission_rate : ℝ)
  (sales_bonus : ℝ)
  (satisfaction_deduction : ℝ)
  (work_week : List WorkDay)
  (total_sales : ℝ)
  (sales_target_reached : Bool)
  (satisfaction_below_threshold : Bool) : ℝ :=
  sorry

/-- Theorem stating that given the problem conditions, total earnings equal $1208.05 -/
theorem total_earnings_proof :
  let last_week_hours : ℝ := 35
  let last_week_rate : ℝ := 10
  let regular_rate_increase : ℝ := 0.5
  let overtime_multiplier : ℝ := 1.5
  let weekend_multiplier : ℝ := 1.7
  let night_shift_multiplier : ℝ := 1.3
  let commission_rate : ℝ := 0.05
  let sales_bonus : ℝ := 50
  let satisfaction_deduction : ℝ := 20
  let work_week : List WorkDay := [
    ⟨8, 3, 0, 0, 200⟩,
    ⟨10, 4, 2, 0, 400⟩,
    ⟨8, 0, 0, 0, 500⟩,
    ⟨9, 3, 1, 0, 300⟩,
    ⟨5, 0, 0, 0, 200⟩,
    ⟨6, 0, 0, 6, 300⟩,
    ⟨4, 2, 0, 4, 100⟩
  ]
  let total_sales : ℝ := 2000
  let sales_target_reached : Bool := true
  let satisfaction_below_threshold : Bool := true
  
  calculate_total_earnings
    last_week_hours
    last_week_rate
    regular_rate_increase
    overtime_multiplier
    weekend_multiplier
    night_shift_multiplier
    commission_rate
    sales_bonus
    satisfaction_deduction
    work_week
    total_sales
    sales_target_reached
    satisfaction_below_threshold = 1208.05 :=
  by sorry

end NUMINAMATH_CALUDE_total_earnings_proof_l1869_186967


namespace NUMINAMATH_CALUDE_last_week_sales_l1869_186949

def chocolate_sales (week1 week2 week3 week4 week5 : ℕ) : Prop :=
  week1 = 75 ∧ week2 = 67 ∧ week3 = 75 ∧ week4 = 70

theorem last_week_sales (week5 : ℕ) :
  chocolate_sales 75 67 75 70 week5 →
  (75 + 67 + 75 + 70 + week5) / 5 = 71 →
  week5 = 68 := by
  sorry

end NUMINAMATH_CALUDE_last_week_sales_l1869_186949


namespace NUMINAMATH_CALUDE_stuart_reward_points_l1869_186963

/-- Represents the reward points earned per $25 spent at the Gauss Store. -/
def reward_points_per_unit : ℕ := 5

/-- Represents the amount Stuart spends at the Gauss Store in dollars. -/
def stuart_spend : ℕ := 200

/-- Represents the dollar amount that earns one unit of reward points. -/
def dollars_per_unit : ℕ := 25

/-- Calculates the number of reward points earned based on the amount spent. -/
def calculate_reward_points (spend : ℕ) : ℕ :=
  (spend / dollars_per_unit) * reward_points_per_unit

/-- Theorem stating that Stuart earns 40 reward points when spending $200 at the Gauss Store. -/
theorem stuart_reward_points : 
  calculate_reward_points stuart_spend = 40 := by
  sorry

end NUMINAMATH_CALUDE_stuart_reward_points_l1869_186963


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l1869_186902

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^4 - 3*x^2 - 4
def g (x : ℝ) : ℝ := -x^4 + 3*x^2 + 2*x

-- State the theorem
theorem polynomial_sum_theorem : 
  ∀ x : ℝ, f x + g x = -4 + 2*x :=
by
  sorry

#check polynomial_sum_theorem

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l1869_186902


namespace NUMINAMATH_CALUDE_work_completion_days_l1869_186982

/-- Proves that given a group of 180 men, where 15 become absent, and the remaining men
    complete the work in 60 days, the original group planned to complete the work in 55 days. -/
theorem work_completion_days (total_men : ℕ) (absent_men : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 180)
  (h2 : absent_men = 15)
  (h3 : actual_days = 60) :
  (total_men * ((total_men - absent_men) * actual_days / total_men : ℚ)).floor = 55 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l1869_186982


namespace NUMINAMATH_CALUDE_product_expansion_l1869_186937

theorem product_expansion (a b c d : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1869_186937


namespace NUMINAMATH_CALUDE_product_equality_l1869_186938

theorem product_equality : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1869_186938


namespace NUMINAMATH_CALUDE_shaded_area_is_12_5_l1869_186901

-- Define the rectangle and its properties
def rectangle_JKLM (J K L M : ℝ × ℝ) : Prop :=
  K.1 = 0 ∧ K.2 = 0 ∧
  L.1 = 5 ∧ L.2 = 0 ∧
  M.1 = 5 ∧ M.2 = 6 ∧
  J.1 = 0 ∧ J.2 = 6

-- Define the additional points I, Q, and N
def point_I (I : ℝ × ℝ) : Prop := I.1 = 0 ∧ I.2 = 5
def point_Q (Q : ℝ × ℝ) : Prop := Q.1 = 5 ∧ Q.2 = 5
def point_N (N : ℝ × ℝ) : Prop := N.1 = 2.5 ∧ N.2 = 3

-- Define the lines JM and LK
def line_JM (J M : ℝ × ℝ) (x y : ℝ) : Prop :=
  y = (6 / 5) * x

def line_LK (L K : ℝ × ℝ) (x y : ℝ) : Prop :=
  y = -(6 / 5) * x + 6

-- Define the areas of trapezoid KQNM and triangle IKN
def area_KQNM (K Q N M : ℝ × ℝ) : ℝ := 11.25
def area_IKN (I K N : ℝ × ℝ) : ℝ := 1.25

-- Theorem statement
theorem shaded_area_is_12_5
  (J K L M I Q N : ℝ × ℝ)
  (h_rect : rectangle_JKLM J K L M)
  (h_I : point_I I)
  (h_Q : point_Q Q)
  (h_N : point_N N)
  (h_JM : line_JM J M N.1 N.2)
  (h_LK : line_LK L K N.1 N.2)
  : area_KQNM K Q N M + area_IKN I K N = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_is_12_5_l1869_186901


namespace NUMINAMATH_CALUDE_smallest_b_factorization_l1869_186936

/-- The smallest positive integer b for which x^2 + bx + 2304 factors into a product of two polynomials with integer coefficients -/
def smallest_factorizable_b : ℕ := 96

/-- Predicate to check if a polynomial factors with integer coefficients -/
def factors_with_integer_coeffs (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (x + p) * (x + q)

theorem smallest_b_factorization :
  (factors_with_integer_coeffs 1 smallest_factorizable_b 2304) ∧
  (∀ b : ℕ, b < smallest_factorizable_b →
    ¬(factors_with_integer_coeffs 1 b 2304)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_factorization_l1869_186936


namespace NUMINAMATH_CALUDE_expression_equality_l1869_186993

theorem expression_equality (a b : ℝ) : -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1869_186993


namespace NUMINAMATH_CALUDE_triangle_abc_right_angle_l1869_186994

theorem triangle_abc_right_angle (A B C : ℝ) (h1 : A = 30) (h2 : B = 60) : C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_right_angle_l1869_186994


namespace NUMINAMATH_CALUDE_least_student_number_l1869_186942

theorem least_student_number (p : ℕ) (q : ℕ) : 
  q % 7 = 0 ∧ 
  q ≥ 1000 ∧ 
  q % (p + 1) = 1 ∧ 
  q % (p + 2) = 1 ∧ 
  q % (p + 3) = 1 ∧ 
  (∀ r : ℕ, r % 7 = 0 ∧ 
            r ≥ 1000 ∧ 
            r % (p + 1) = 1 ∧ 
            r % (p + 2) = 1 ∧ 
            r % (p + 3) = 1 → 
            q ≤ r) → 
  q = 1141 :=
by sorry

end NUMINAMATH_CALUDE_least_student_number_l1869_186942


namespace NUMINAMATH_CALUDE_conference_theorem_l1869_186923

/-- A graph with vertices labeled 1 to n, where edges are colored either red or blue -/
structure ColoredGraph (n : ℕ) where
  edge_color : Fin n → Fin n → Bool

/-- Predicate to check if a subgraph of 4 vertices satisfies the given conditions -/
def valid_subgraph (G : ColoredGraph n) (a b c d : Fin n) : Prop :=
  let edges := [G.edge_color a b, G.edge_color a c, G.edge_color a d, 
                G.edge_color b c, G.edge_color b d, G.edge_color c d]
  let red_count := (edges.filter id).length
  let blue_count := (edges.filter not).length
  (red_count + blue_count) % 2 = 0 ∧ 
  red_count > 0 ∧ 
  (blue_count = 0 ∨ blue_count ≥ red_count)

/-- Theorem statement -/
theorem conference_theorem :
  ∃ (G : ColoredGraph 2017),
    (∀ (a b c d : Fin 2017), valid_subgraph G a b c d) →
    ∃ (S : Finset (Fin 2017)),
      S.card = 673 ∧
      ∀ (x y : Fin 2017), x ∈ S → y ∈ S → x ≠ y → G.edge_color x y = true :=
by sorry

end NUMINAMATH_CALUDE_conference_theorem_l1869_186923


namespace NUMINAMATH_CALUDE_expression_evaluation_l1869_186973

theorem expression_evaluation : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) - (6^2 - 6) = -32 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1869_186973


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l1869_186975

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8)
  (h2 : Real.sin (a - b) = 1/4) :
  Real.tan a / Real.tan b = 7/3 := by
sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l1869_186975


namespace NUMINAMATH_CALUDE_odd_number_set_characterization_l1869_186932

def OddNumberSet : Set ℤ :=
  {x | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2 * k + 1}

theorem odd_number_set_characterization :
  OddNumberSet = {x : ℤ | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2 * k + 1} := by
  sorry

end NUMINAMATH_CALUDE_odd_number_set_characterization_l1869_186932


namespace NUMINAMATH_CALUDE_prob_one_white_two_drawn_correct_expectation_white_three_drawn_correct_l1869_186911

/-- The number of black balls in the bag -/
def num_black : ℕ := 2

/-- The number of white balls in the bag -/
def num_white : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_black + num_white

/-- The probability of drawing exactly one white ball when two balls are randomly drawn -/
def prob_one_white_two_drawn : ℚ := 3 / 5

/-- The mathematical expectation of the number of white balls when three balls are randomly drawn -/
def expectation_white_three_drawn : ℚ := 18 / 10

/-- Theorem stating the probability of drawing exactly one white ball when two balls are randomly drawn -/
theorem prob_one_white_two_drawn_correct :
  prob_one_white_two_drawn = (num_black * num_white : ℚ) / ((total_balls * (total_balls - 1)) / 2) :=
sorry

/-- Theorem stating the mathematical expectation of the number of white balls when three balls are randomly drawn -/
theorem expectation_white_three_drawn_correct :
  expectation_white_three_drawn = 
    (1 * (num_black * num_black * num_white : ℚ) +
     2 * (num_black * num_white * (num_white - 1)) +
     3 * (num_white * (num_white - 1) * (num_white - 2))) /
    ((total_balls * (total_balls - 1) * (total_balls - 2)) / 6) :=
sorry

end NUMINAMATH_CALUDE_prob_one_white_two_drawn_correct_expectation_white_three_drawn_correct_l1869_186911


namespace NUMINAMATH_CALUDE_min_angle_in_special_right_triangle_l1869_186948

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Consecutive Fibonacci numbers -/
def consecutive_fib (a b : ℕ) : Prop :=
  ∃ n : ℕ, fib n = b ∧ fib (n + 1) = a

theorem min_angle_in_special_right_triangle :
  ∀ a b : ℕ,
    a > b →
    consecutive_fib a b →
    a + b = 100 →
    b ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_min_angle_in_special_right_triangle_l1869_186948


namespace NUMINAMATH_CALUDE_restaurant_group_size_restaurant_group_size_proof_l1869_186903

theorem restaurant_group_size (adult_meal_cost : ℕ) (kids_in_group : ℕ) (total_cost : ℕ) : ℕ :=
  let adults_in_group := total_cost / adult_meal_cost
  let total_people := adults_in_group + kids_in_group
  total_people

#check restaurant_group_size 8 2 72 = 11

theorem restaurant_group_size_proof 
  (adult_meal_cost : ℕ) 
  (kids_in_group : ℕ) 
  (total_cost : ℕ) 
  (h1 : adult_meal_cost = 8)
  (h2 : kids_in_group = 2)
  (h3 : total_cost = 72) :
  restaurant_group_size adult_meal_cost kids_in_group total_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_size_restaurant_group_size_proof_l1869_186903


namespace NUMINAMATH_CALUDE_abc_product_l1869_186922

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 168) (h2 : b * (c + a) = 153) (h3 : c * (a + b) = 147) :
  a * b * c = 720 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1869_186922


namespace NUMINAMATH_CALUDE_largest_n_for_inequality_l1869_186953

theorem largest_n_for_inequality : ∃ (n : ℕ), n = 24 ∧ 
  (∀ (a b c d : ℝ), 
    (↑n + 2) * Real.sqrt (a^2 + b^2) + 
    (↑n + 1) * Real.sqrt (a^2 + c^2) + 
    (↑n + 1) * Real.sqrt (a^2 + d^2) ≥ 
    ↑n * (a + b + c + d)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (a b c d : ℝ), 
      (↑m + 2) * Real.sqrt (a^2 + b^2) + 
      (↑m + 1) * Real.sqrt (a^2 + c^2) + 
      (↑m + 1) * Real.sqrt (a^2 + d^2) < 
      ↑m * (a + b + c + d)) :=
by sorry


end NUMINAMATH_CALUDE_largest_n_for_inequality_l1869_186953


namespace NUMINAMATH_CALUDE_p_iff_between_two_and_three_l1869_186907

def p (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

theorem p_iff_between_two_and_three :
  ∀ x : ℝ, p x ↔ 2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_p_iff_between_two_and_three_l1869_186907


namespace NUMINAMATH_CALUDE_hannah_restaurant_bill_hannah_restaurant_bill_proof_l1869_186947

/-- The total amount Hannah spent on the entree and dessert is $23, 
    given that the entree costs $14 and it is $5 more than the dessert. -/
theorem hannah_restaurant_bill : ℕ → ℕ → ℕ → Prop :=
  fun entree_cost dessert_cost total_cost =>
    (entree_cost = 14) →
    (entree_cost = dessert_cost + 5) →
    (total_cost = entree_cost + dessert_cost) →
    (total_cost = 23)

/-- Proof of hannah_restaurant_bill -/
theorem hannah_restaurant_bill_proof : hannah_restaurant_bill 14 9 23 := by
  sorry

end NUMINAMATH_CALUDE_hannah_restaurant_bill_hannah_restaurant_bill_proof_l1869_186947


namespace NUMINAMATH_CALUDE_transaction_gain_per_year_l1869_186926

def principal : ℝ := 5000
def duration : ℕ := 2
def borrow_rate_year1 : ℝ := 0.04
def borrow_rate_year2 : ℝ := 0.06
def lend_rate_year1 : ℝ := 0.05
def lend_rate_year2 : ℝ := 0.07

theorem transaction_gain_per_year : 
  let amount_lend_year1 := principal * (1 + lend_rate_year1)
  let amount_lend_year2 := amount_lend_year1 * (1 + lend_rate_year2)
  let interest_earned := amount_lend_year2 - principal
  let amount_borrow_year1 := principal * (1 + borrow_rate_year1)
  let amount_borrow_year2 := amount_borrow_year1 * (1 + borrow_rate_year2)
  let interest_paid := amount_borrow_year2 - principal
  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / duration
  gain_per_year = 52.75 := by sorry

end NUMINAMATH_CALUDE_transaction_gain_per_year_l1869_186926


namespace NUMINAMATH_CALUDE_speed_relationship_l1869_186968

/-- Represents the speed of travel between two towns -/
structure TravelSpeed where
  xy : ℝ  -- Speed from x to y
  yx : ℝ  -- Speed from y to x
  avg : ℝ  -- Average speed for the whole journey

/-- Theorem stating the relationship between speeds -/
theorem speed_relationship (s : TravelSpeed) (h1 : s.xy = 60) (h2 : s.avg = 40) : s.yx = 30 := by
  sorry

end NUMINAMATH_CALUDE_speed_relationship_l1869_186968


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1869_186972

/-- A geometric sequence with positive integer terms -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = (a n : ℚ) * r

theorem geometric_sequence_second_term
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 0 = 5)
  (h_fourth : a 3 = 480) :
  a 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1869_186972


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1869_186909

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1869_186909


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l1869_186952

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | f x > 1}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (2/3) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l1869_186952


namespace NUMINAMATH_CALUDE_addition_inequality_l1869_186939

theorem addition_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_addition_inequality_l1869_186939


namespace NUMINAMATH_CALUDE_decimal_100_to_binary_l1869_186977

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_100_to_binary :
  decimal_to_binary 100 = [1, 1, 0, 0, 1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_100_to_binary_l1869_186977


namespace NUMINAMATH_CALUDE_y_value_proof_l1869_186995

theorem y_value_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 8)
  (h2 : y + 1/x = 7/12)
  (h3 : x + y = 7) :
  y = 49/103 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1869_186995


namespace NUMINAMATH_CALUDE_stamps_per_page_l1869_186969

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 945) 
  (h2 : book2 = 1260) 
  (h3 : book3 = 1575) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 315 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l1869_186969


namespace NUMINAMATH_CALUDE_remainder_2567139_div_6_l1869_186925

theorem remainder_2567139_div_6 : 2567139 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2567139_div_6_l1869_186925


namespace NUMINAMATH_CALUDE_min_force_to_submerge_cube_l1869_186960

/-- Minimum force required to submerge a cube -/
theorem min_force_to_submerge_cube 
  (cube_volume : Real) 
  (cube_density : Real) 
  (water_density : Real) 
  (gravity : Real) :
  cube_volume = 1e-5 →  -- 10 cm³ = 1e-5 m³
  cube_density = 700 →
  water_density = 1000 →
  gravity = 10 →
  (water_density - cube_density) * cube_volume * gravity = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_min_force_to_submerge_cube_l1869_186960


namespace NUMINAMATH_CALUDE_given_number_scientific_notation_l1869_186964

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number in meters -/
def given_number : ℝ := 0.000000014

/-- The scientific notation representation of the given number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 1.4
    exponent := -8
    coefficient_range := by sorry }

theorem given_number_scientific_notation :
  given_number = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_given_number_scientific_notation_l1869_186964


namespace NUMINAMATH_CALUDE_special_divisor_property_implies_prime_l1869_186910

theorem special_divisor_property_implies_prime (n : ℕ) 
  (h1 : n > 1)
  (h2 : ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) :
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_special_divisor_property_implies_prime_l1869_186910


namespace NUMINAMATH_CALUDE_oyster_consumption_l1869_186971

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- The number of oysters Crabby eats -/
def crabby_oysters : ℕ := 2 * squido_oysters

/-- The total number of oysters eaten by Crabby and Squido -/
def total_oysters : ℕ := squido_oysters + crabby_oysters

theorem oyster_consumption :
  total_oysters = 600 :=
by sorry

end NUMINAMATH_CALUDE_oyster_consumption_l1869_186971


namespace NUMINAMATH_CALUDE_inequality_implication_l1869_186930

theorem inequality_implication (m n : ℝ) (h1 : m > 0) (h2 : n > m) : 1/m - 1/n > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1869_186930


namespace NUMINAMATH_CALUDE_cube_surface_area_doubles_l1869_186933

/-- Theorem: Doubling the edge length of a cube increases its surface area by a factor of 4 -/
theorem cube_surface_area_doubles (a : ℝ) (h : a > 0) :
  (6 * (2 * a)^2) / (6 * a^2) = 4 := by
  sorry

#check cube_surface_area_doubles

end NUMINAMATH_CALUDE_cube_surface_area_doubles_l1869_186933


namespace NUMINAMATH_CALUDE_remainder_eight_pow_six_plus_one_mod_seven_l1869_186943

theorem remainder_eight_pow_six_plus_one_mod_seven :
  (8^6 + 1) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_pow_six_plus_one_mod_seven_l1869_186943


namespace NUMINAMATH_CALUDE_circle_symmetry_l1869_186904

-- Define the line l: x + y = 0
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define circle C: (x-2)^2 + (y-1)^2 = 4
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define circle C': (x+1)^2 + (y+2)^2 = 4
def circle_C' (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 4

-- Function to reflect a point (x, y) across the line l
def reflect_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Theorem stating that C' is symmetric to C with respect to l
theorem circle_symmetry :
  ∀ x y : ℝ, circle_C x y ↔ circle_C' (reflect_point x y).1 (reflect_point x y).2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1869_186904


namespace NUMINAMATH_CALUDE_haris_contribution_haris_contribution_is_9720_l1869_186941

/-- Calculates Hari's contribution to the capital given the investment conditions --/
theorem haris_contribution (praveen_investment : ℕ) (praveen_months : ℕ) (hari_months : ℕ) 
  (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) : ℕ :=
  let total_months := praveen_months
  let hari_contribution := (praveen_investment * praveen_months * profit_ratio_hari) / 
                           (hari_months * profit_ratio_praveen)
  hari_contribution

/-- Proves that Hari's contribution is 9720 given the specific conditions --/
theorem haris_contribution_is_9720 : 
  haris_contribution 3780 12 7 2 3 = 9720 := by
  sorry

end NUMINAMATH_CALUDE_haris_contribution_haris_contribution_is_9720_l1869_186941


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1869_186991

theorem sin_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin (2 * α - π / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1869_186991


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l1869_186997

/-- Given distinct positive integers a and b, where 20a + 17b and 17a + 20b
    are both prime numbers, the minimum sum of these prime numbers is 296. -/
theorem min_sum_of_primes (a b : ℕ+) (h_distinct : a ≠ b)
  (h_prime1 : Nat.Prime (20 * a + 17 * b))
  (h_prime2 : Nat.Prime (17 * a + 20 * b)) :
  (20 * a + 17 * b) + (17 * a + 20 * b) ≥ 296 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l1869_186997


namespace NUMINAMATH_CALUDE_net_population_increase_per_day_l1869_186987

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 5

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 3

/-- Calculates the net population increase per second -/
def net_increase_per_second : ℚ := (birth_rate - death_rate) / 2

/-- Theorem stating the net population increase over one day -/
theorem net_population_increase_per_day :
  (net_increase_per_second * seconds_per_day : ℚ) = 86400 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_per_day_l1869_186987


namespace NUMINAMATH_CALUDE_eight_people_arrangement_l1869_186906

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where 2 specific people are together -/
def arrangementsTwoTogether (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where 2 specific people are not together -/
def arrangementsNotTogether (n : ℕ) : ℕ :=
  totalArrangements n - arrangementsTwoTogether n

theorem eight_people_arrangement :
  arrangementsNotTogether 8 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_arrangement_l1869_186906


namespace NUMINAMATH_CALUDE_trivia_team_size_l1869_186955

theorem trivia_team_size :
  let members_absent : ℝ := 2
  let total_score : ℝ := 6
  let score_per_member : ℝ := 2
  let members_present : ℝ := total_score / score_per_member
  let total_members : ℝ := members_present + members_absent
  total_members = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l1869_186955


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1869_186961

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1869_186961


namespace NUMINAMATH_CALUDE_circus_performers_time_ratio_l1869_186976

theorem circus_performers_time_ratio :
  ∀ (polly_time pulsar_time petra_time : ℕ),
    pulsar_time = 10 →
    ∃ k : ℕ, polly_time = k * pulsar_time →
    petra_time = polly_time / 6 →
    pulsar_time + polly_time + petra_time = 45 →
    polly_time / pulsar_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_circus_performers_time_ratio_l1869_186976


namespace NUMINAMATH_CALUDE_probability_spade_or_diamond_l1869_186974

theorem probability_spade_or_diamond (total_cards : ℕ) (ranks : ℕ) (suits : ℕ) 
  (h1 : total_cards = 52)
  (h2 : ranks = 13)
  (h3 : suits = 4)
  (h4 : total_cards = ranks * suits) :
  (2 : ℚ) * (ranks : ℚ) / (total_cards : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_spade_or_diamond_l1869_186974


namespace NUMINAMATH_CALUDE_vector_representation_l1869_186913

-- Define points A, B, and Q in a 2D plane
variable (A B Q : ℝ × ℝ)

-- Define the ratio condition
def ratio_condition (A B Q : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ Q.1 - A.1 = 7*k ∧ Q.1 - B.1 = 2*k ∧
              Q.2 - A.2 = 7*k ∧ Q.2 - B.2 = 2*k

-- Define vector addition and scalar multiplication
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Theorem statement
theorem vector_representation (A B Q : ℝ × ℝ) 
  (h : ratio_condition A B Q) :
  Q = vec_add (scalar_mul (-2/5) A) B :=
sorry

end NUMINAMATH_CALUDE_vector_representation_l1869_186913


namespace NUMINAMATH_CALUDE_lcm_of_6_and_15_l1869_186980

theorem lcm_of_6_and_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_6_and_15_l1869_186980


namespace NUMINAMATH_CALUDE_book_profit_percentage_l1869_186920

/-- Given a book with cost price $1800, if selling it for $90 more than the initial
    selling price would result in a 15% profit, then the initial profit percentage is 10% -/
theorem book_profit_percentage (cost_price : ℝ) (additional_price : ℝ) 
  (higher_profit_percentage : ℝ) (initial_selling_price : ℝ) :
  cost_price = 1800 →
  additional_price = 90 →
  higher_profit_percentage = 15 →
  initial_selling_price + additional_price = cost_price * (1 + higher_profit_percentage / 100) →
  (initial_selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_percentage_l1869_186920


namespace NUMINAMATH_CALUDE_quadratic_roots_l1869_186996

theorem quadratic_roots :
  ∃ (x₁ x₂ : ℝ), (x₁ = 2 ∧ x₂ = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1869_186996


namespace NUMINAMATH_CALUDE_album_pages_count_l1869_186970

theorem album_pages_count : ∃ (x : ℕ) (y : ℕ), 
  x > 0 ∧ 
  y > 0 ∧ 
  20 * x < y ∧ 
  23 * x > y ∧ 
  21 * x + y = 500 ∧ 
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_album_pages_count_l1869_186970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1869_186908

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  n : ℕ  -- number of terms
  first_sum : ℕ  -- sum of first 4 terms
  last_sum : ℕ  -- sum of last 4 terms
  total_sum : ℕ  -- sum of all terms

/-- The theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.first_sum = 26)
  (h2 : seq.last_sum = 110)
  (h3 : seq.total_sum = 187) :
  seq.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1869_186908


namespace NUMINAMATH_CALUDE_talia_age_in_seven_years_l1869_186946

/-- Proves Talia's age in seven years given the conditions of the problem -/
theorem talia_age_in_seven_years :
  ∀ (talia_age mom_age dad_age : ℕ),
    mom_age = 3 * talia_age →
    dad_age + 3 = mom_age →
    dad_age = 36 →
    talia_age + 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_talia_age_in_seven_years_l1869_186946


namespace NUMINAMATH_CALUDE_division_problem_l1869_186983

theorem division_problem (A : ℕ) (h1 : 26 = A * 8 + 2) : A = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1869_186983


namespace NUMINAMATH_CALUDE_complex_arithmetic_simplification_l1869_186919

theorem complex_arithmetic_simplification :
  ((6 - 3 * Complex.I) - (2 + 4 * Complex.I)) * (2 * Complex.I) = 14 + 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_simplification_l1869_186919


namespace NUMINAMATH_CALUDE_problem_statement_l1869_186900

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2015 + b^2016 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1869_186900


namespace NUMINAMATH_CALUDE_one_white_ball_probability_l1869_186958

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (event total : ℕ) : ℚ := sorry

theorem one_white_ball_probability (bagA_white bagA_red bagB_white bagB_red : ℕ) 
  (h1 : bagA_white = 8)
  (h2 : bagA_red = 4)
  (h3 : bagB_white = 6)
  (h4 : bagB_red = 6) :
  probability 
    (choose bagA_white 1 * choose bagB_red 1 + choose bagA_red 1 * choose bagB_white 1)
    (choose (bagA_white + bagA_red) 1 * choose (bagB_white + bagB_red) 1) =
  probability 
    ((choose 8 1) * (choose 6 1) + (choose 4 1) * (choose 6 1))
    ((choose 12 1) * (choose 12 1)) :=
sorry

end NUMINAMATH_CALUDE_one_white_ball_probability_l1869_186958


namespace NUMINAMATH_CALUDE_expression_value_l1869_186989

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : 
  -2*a - b^2 + 2*a*b = -41 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1869_186989


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1869_186951

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : 
  let x : ℝ := 1
  let y : ℝ := -5
  fourth_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1869_186951
