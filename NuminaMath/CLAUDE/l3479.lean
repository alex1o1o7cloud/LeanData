import Mathlib

namespace NUMINAMATH_CALUDE_probability_not_adjacent_l3479_347976

def total_chairs : ℕ := 10
def broken_chairs : Finset ℕ := {5, 8}
def available_chairs : ℕ := total_chairs - broken_chairs.card

def adjacent_pairs : ℕ := 6

theorem probability_not_adjacent :
  (1 - (adjacent_pairs : ℚ) / (available_chairs.choose 2)) = 11/14 := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l3479_347976


namespace NUMINAMATH_CALUDE_peanut_seed_sprouting_probability_l3479_347901

/-- The probability of exactly k successes in n independent trials,
    where p is the probability of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem peanut_seed_sprouting_probability :
  let n : ℕ := 3  -- total number of seeds
  let k : ℕ := 2  -- number of seeds we want to sprout
  let p : ℝ := 3/5  -- probability of each seed sprouting
  binomial_probability n k p = 54/125 := by
sorry

end NUMINAMATH_CALUDE_peanut_seed_sprouting_probability_l3479_347901


namespace NUMINAMATH_CALUDE_new_person_age_l3479_347939

theorem new_person_age (T : ℕ) : 
  (T / 10 : ℚ) - 3 = ((T - 40 + 10) / 10 : ℚ) → 10 = 10 := by
sorry

end NUMINAMATH_CALUDE_new_person_age_l3479_347939


namespace NUMINAMATH_CALUDE_dihedral_angle_is_120_degrees_l3479_347902

/-- A regular tetrahedron with a circumscribed sphere -/
structure RegularTetrahedronWithSphere where
  /-- The height of the tetrahedron -/
  height : ℝ
  /-- The diameter of the circumscribed sphere -/
  sphere_diameter : ℝ
  /-- The diameter of the sphere is 9 times the height of the tetrahedron -/
  sphere_diameter_relation : sphere_diameter = 9 * height

/-- The dihedral angle between two lateral faces of a regular tetrahedron -/
def dihedral_angle (t : RegularTetrahedronWithSphere) : ℝ :=
  sorry

/-- Theorem: The dihedral angle between two lateral faces of a regular tetrahedron
    with the given sphere relation is 120 degrees -/
theorem dihedral_angle_is_120_degrees (t : RegularTetrahedronWithSphere) :
  dihedral_angle t = 120 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_is_120_degrees_l3479_347902


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l3479_347983

def is_reducible (n : ℕ) : Prop :=
  n > 0 ∧ (n - 17).gcd (7 * n + 5) > 1

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 48 → ¬(is_reducible m)) ∧ is_reducible 48 := by
  sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l3479_347983


namespace NUMINAMATH_CALUDE_box_depth_l3479_347907

theorem box_depth (length width : ℕ) (num_cubes : ℕ) (depth : ℕ) : 
  length = 35 → 
  width = 20 → 
  num_cubes = 56 →
  (∃ (cube_edge : ℕ), 
    cube_edge ∣ length ∧ 
    cube_edge ∣ width ∧ 
    cube_edge ∣ depth ∧
    cube_edge ^ 3 * num_cubes = length * width * depth) →
  depth = 10 := by
  sorry


end NUMINAMATH_CALUDE_box_depth_l3479_347907


namespace NUMINAMATH_CALUDE_leak_empty_time_l3479_347956

/-- Represents the time it takes for a leak to empty a full tank, given the filling times with and without the leak. -/
theorem leak_empty_time (fill_time : ℝ) (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) : 
  fill_time > 0 ∧ fill_time_with_leak > fill_time →
  (1 / fill_time) - (1 / fill_time_with_leak) = 1 / leak_empty_time →
  fill_time = 6 →
  fill_time_with_leak = 9 →
  leak_empty_time = 18 := by
sorry

end NUMINAMATH_CALUDE_leak_empty_time_l3479_347956


namespace NUMINAMATH_CALUDE_russian_doll_purchase_l3479_347950

/-- Given a person's savings for a certain number of items at an original price,
    calculate how many items they can buy when the price drops to a new lower price. -/
theorem russian_doll_purchase (original_price new_price : ℚ) (original_quantity : ℕ) :
  original_price > 0 →
  new_price > 0 →
  new_price < original_price →
  (original_price * original_quantity) / new_price = 20 :=
by
  sorry

#check russian_doll_purchase (4 : ℚ) (3 : ℚ) 15

end NUMINAMATH_CALUDE_russian_doll_purchase_l3479_347950


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l3479_347949

-- Define the number of people in the main committee
def n : ℕ := 8

-- Define the number of people to be selected for each sub-committee
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem three_person_subcommittees_from_eight :
  combination n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l3479_347949


namespace NUMINAMATH_CALUDE_circle_equation_with_center_and_tangent_line_l3479_347909

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ = a * sin(θ - α) + b -/
structure PolarLine where
  a : ℝ
  α : ℝ
  b : ℝ

/-- Represents a circle in polar form ρ = R * sin(θ - β) -/
structure PolarCircle where
  R : ℝ
  β : ℝ

def is_tangent (c : PolarCircle) (l : PolarLine) : Prop :=
  sorry

theorem circle_equation_with_center_and_tangent_line 
  (P : PolarPoint) 
  (l : PolarLine) 
  (h1 : P.r = 2 ∧ P.θ = π/3) 
  (h2 : l.a = 1 ∧ l.α = π/3 ∧ l.b = 2) : 
  ∃ (c : PolarCircle), c.R = 4 ∧ c.β = -π/6 ∧ is_tangent c l :=
sorry

end NUMINAMATH_CALUDE_circle_equation_with_center_and_tangent_line_l3479_347909


namespace NUMINAMATH_CALUDE_total_amount_is_sum_of_shares_l3479_347989

/-- Represents the time in days it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℕ

/-- Represents the share of money received by a person -/
structure Share where
  amount : ℕ

/-- Represents a worker with their individual work time and share -/
structure Worker where
  workTime : WorkTime
  share : Share

/-- Theorem: The total amount received for the work is the sum of individual shares -/
theorem total_amount_is_sum_of_shares 
  (a b c : Worker)
  (h1 : a.workTime.days = 6)
  (h2 : b.workTime.days = 8)
  (h3 : a.share.amount = 300)
  (h4 : b.share.amount = 225)
  (h5 : c.share.amount = 75) :
  a.share.amount + b.share.amount + c.share.amount = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_sum_of_shares_l3479_347989


namespace NUMINAMATH_CALUDE_correct_transformation_l3479_347906

theorem correct_transformation (a b m : ℝ) : a * (m^2 + 1) = b * (m^2 + 1) → a = b := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l3479_347906


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l3479_347995

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (XY : Real)
  (YZ : Real)
  (XZ : Real)

-- Define points P and Q
structure Points (t : Triangle) :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (XP : Real)
  (XQ : Real)

-- Define the area ratio
def areaRatio (t : Triangle) (pts : Points t) : ℚ := sorry

-- State the theorem
theorem area_ratio_theorem (t : Triangle) (pts : Points t) :
  t.XY = 30 →
  t.YZ = 45 →
  t.XZ = 54 →
  pts.XP = 18 →
  pts.XQ = 36 →
  areaRatio t pts = 27 / 50 := by sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l3479_347995


namespace NUMINAMATH_CALUDE_trigonometric_form_of_negative_3i_l3479_347992

theorem trigonometric_form_of_negative_3i :
  ∀ z : ℂ, z = -3 * Complex.I →
  z = 3 * (Complex.cos (3 * Real.pi / 2) + Complex.I * Complex.sin (3 * Real.pi / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_form_of_negative_3i_l3479_347992


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l3479_347972

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Represents the rolling properties of the cone -/
structure ConeRolling (cone : RightCircularCone) where
  rotations : ℕ
  no_slipping : Bool

theorem cone_rolling_ratio (cone : RightCircularCone) (rolling : ConeRolling cone) :
  rolling.rotations = 19 ∧ rolling.no_slipping = true →
  cone.h / cone.r = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l3479_347972


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l3479_347955

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => (4/7) * a (n + 1) + (3/7) * a n

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 17/10| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l3479_347955


namespace NUMINAMATH_CALUDE_square_root_of_8_l3479_347991

-- Define the square root property
def is_square_root (x : ℝ) (y : ℝ) : Prop := x * x = y

-- Theorem statement
theorem square_root_of_8 :
  ∃ (x : ℝ), is_square_root x 8 ∧ x = Real.sqrt 8 ∨ x = -Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_8_l3479_347991


namespace NUMINAMATH_CALUDE_inequality_proof_l3479_347944

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3479_347944


namespace NUMINAMATH_CALUDE_second_stock_percentage_l3479_347977

/-- Prove that the percentage of the second stock is 15% given the investment conditions --/
theorem second_stock_percentage
  (total_investment : ℚ)
  (first_stock_percentage : ℚ)
  (first_stock_face_value : ℚ)
  (second_stock_face_value : ℚ)
  (total_dividend : ℚ)
  (first_stock_investment : ℚ)
  (h1 : total_investment = 12000)
  (h2 : first_stock_percentage = 12 / 100)
  (h3 : first_stock_face_value = 120)
  (h4 : second_stock_face_value = 125)
  (h5 : total_dividend = 1360)
  (h6 : first_stock_investment = 4000.000000000002)
  : (total_dividend - (first_stock_investment / first_stock_face_value * first_stock_percentage)) /
    ((total_investment - first_stock_investment) / second_stock_face_value) = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_stock_percentage_l3479_347977


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l3479_347932

theorem cubic_equation_solutions_mean (x : ℝ) : 
  (x^3 + 5*x^2 - 14*x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, y^3 + 5*y^2 - 14*y = 0) ∧ 
                   (∀ z, z^3 + 5*z^2 - 14*z = 0 → z ∈ s) ∧
                   (Finset.sum s id / s.card = -5/3)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l3479_347932


namespace NUMINAMATH_CALUDE_unique_zero_point_b_range_l3479_347931

-- Define the function f_n
def f_n (n : ℕ) (b c : ℝ) (x : ℝ) : ℝ := x^n + b*x + c

-- Part I
theorem unique_zero_point (n : ℕ) (h : n ≥ 2) :
  ∃! x, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ f_n n 1 (-1) x = 0 :=
sorry

-- Part II
theorem b_range (h : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1 : ℝ) 1 → x₂ ∈ Set.Icc (-1 : ℝ) 1 →
  |f_n 2 b c x₁ - f_n 2 b c x₂| ≤ 4) :
  b ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_point_b_range_l3479_347931


namespace NUMINAMATH_CALUDE_system_solutions_l3479_347905

def solution_set : Set (ℝ × ℝ) :=
  {(-1, 2), (2, -1), (1 + Real.sqrt 2, 1 - Real.sqrt 2), (1 - Real.sqrt 2, 1 + Real.sqrt 2),
   ((-9 + Real.sqrt 57) / 6, (-9 - Real.sqrt 57) / 6), ((-9 - Real.sqrt 57) / 6, (-9 + Real.sqrt 57) / 6)}

theorem system_solutions (x y : ℝ) :
  (x^2 - x*y + y^2 = 7 ∧ x^2*y + x*y^2 = -2) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3479_347905


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3479_347946

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_inequality : (a 5 + a 6 + a 7 + a 8) * (a 6 + a 7 + a 8) < 0) :
  |a 6| > |a 7| := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3479_347946


namespace NUMINAMATH_CALUDE_car_trip_duration_l3479_347973

theorem car_trip_duration :
  ∀ (total_time : ℝ) (second_part_time : ℝ),
    total_time > 0 →
    second_part_time ≥ 0 →
    total_time = 5 + second_part_time →
    (30 * 5 + 42 * second_part_time) / total_time = 34 →
    total_time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l3479_347973


namespace NUMINAMATH_CALUDE_x_varies_as_z_power_l3479_347948

-- Define the relationships between x, y, and z
def varies_as (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ t, f t = k * g t

-- State the theorem
theorem x_varies_as_z_power (x y z : ℝ → ℝ) :
  varies_as x (λ t => (y t)^4) →
  varies_as y (λ t => (z t)^(1/3)) →
  varies_as x (λ t => (z t)^(4/3)) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_z_power_l3479_347948


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l3479_347997

theorem cube_root_of_eight :
  (8 : ℝ) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l3479_347997


namespace NUMINAMATH_CALUDE_f_has_two_zeros_h_range_l3479_347965

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - x
def g (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - x * Real.log x + (1 - a) * x

-- Define the set of a values for g
def A : Set ℝ := Set.Ioo 1 (3 - Real.log 3)

-- Define h(a) as the local minimum of g(x) for a given a
noncomputable def h (a : ℝ) : ℝ := 
  let x₂ := Real.exp a
  2 * x₂ - x₂^2

-- Theorem statements
theorem f_has_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a > 1 := by sorry

theorem h_range :
  Set.range h = Set.Icc (-3) 1 := by sorry

end

end NUMINAMATH_CALUDE_f_has_two_zeros_h_range_l3479_347965


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3479_347982

theorem regular_polygon_sides (n : ℕ) (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 144) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3479_347982


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_x_axis_l3479_347987

/-- A circle with center at (-3, 4) that is tangent to the x-axis has the standard equation (x + 3)^2 + (y - 4)^2 = 16 -/
theorem circle_equation_tangent_to_x_axis (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let is_tangent_to_x_axis := center.2 = 4  -- The y-coordinate of the center is the distance to x-axis
  (x + 3)^2 + (y - 4)^2 = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_tangent_to_x_axis_l3479_347987


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3479_347953

theorem polynomial_factorization : 
  ∀ x : ℝ, x^2 - x + (1/4 : ℝ) = (x - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3479_347953


namespace NUMINAMATH_CALUDE_equation_solution_l3479_347981

theorem equation_solution : 
  ∀ x : ℝ, (1 / (x + 1) + 1 / (x + 2) = 1 / x) ↔ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3479_347981


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3479_347951

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3479_347951


namespace NUMINAMATH_CALUDE_power_sum_sequence_l3479_347942

/-- Given a sequence of sums of powers of a and b, prove that a^10 + b^10 = 123 -/
theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l3479_347942


namespace NUMINAMATH_CALUDE_tangent_slope_exp_at_two_l3479_347903

/-- The slope of the tangent line to y = e^x at x = 2 is e^2 -/
theorem tangent_slope_exp_at_two :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  (deriv f) 2 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_exp_at_two_l3479_347903


namespace NUMINAMATH_CALUDE_total_pencils_l3479_347978

/-- Calculate the total number of pencils Asaf and Alexander have together -/
theorem total_pencils (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) :
  asaf_age + alexander_age = 140 →
  asaf_age = 50 →
  alexander_age - asaf_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3479_347978


namespace NUMINAMATH_CALUDE_even_function_sum_l3479_347929

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_sum (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : f 4 = 3) :
  f 4 + f (-4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l3479_347929


namespace NUMINAMATH_CALUDE_average_not_1380_l3479_347968

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1200]

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem average_not_1380 : average numbers ≠ 1380 := by
  sorry

end NUMINAMATH_CALUDE_average_not_1380_l3479_347968


namespace NUMINAMATH_CALUDE_point_position_l3479_347936

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is to the right of the y-axis -/
def isRightOfYAxis (p : Point) : Prop := p.x > 0

/-- Predicate to check if a point is below the x-axis -/
def isBelowXAxis (p : Point) : Prop := p.y < 0

/-- Theorem stating that if a point is to the right of the y-axis and below the x-axis,
    then its x-coordinate is positive and its y-coordinate is negative -/
theorem point_position (P : Point) 
  (h1 : isRightOfYAxis P) (h2 : isBelowXAxis P) : 
  P.x > 0 ∧ P.y < 0 := by
  sorry

#check point_position

end NUMINAMATH_CALUDE_point_position_l3479_347936


namespace NUMINAMATH_CALUDE_factorization_equality_l3479_347947

theorem factorization_equality (x y : ℝ) : x^2 - 1 + 2*x*y + y^2 = (x+y+1)*(x+y-1) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l3479_347947


namespace NUMINAMATH_CALUDE_percentage_for_sobel_l3479_347966

/-- Represents the percentage of voters who are male -/
def male_percentage : ℝ := 60

/-- Represents the percentage of female voters who voted for Lange -/
def female_for_lange : ℝ := 35

/-- Represents the percentage of male voters who voted for Sobel -/
def male_for_sobel : ℝ := 44

/-- Theorem stating the percentage of total voters who voted for Sobel -/
theorem percentage_for_sobel :
  let female_percentage := 100 - male_percentage
  let female_for_sobel := 100 - female_for_lange
  let total_for_sobel := (male_percentage * male_for_sobel + female_percentage * female_for_sobel) / 100
  total_for_sobel = 52.4 := by sorry

end NUMINAMATH_CALUDE_percentage_for_sobel_l3479_347966


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l3479_347985

/-- Given that N(5,9) is the midpoint of line segment CD and C has coordinates (11,5),
    prove that the sum of the coordinates of point D is 12. -/
theorem sum_coordinates_of_D (C D N : ℝ × ℝ) : 
  C = (11, 5) → 
  N = (5, 9) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l3479_347985


namespace NUMINAMATH_CALUDE_inequality_range_l3479_347940

theorem inequality_range (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x) + (16/y) > a^2 + 24*a) ↔ -25 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3479_347940


namespace NUMINAMATH_CALUDE_function_with_two_zeros_l3479_347911

theorem function_with_two_zeros 
  (f : ℝ → ℝ) 
  (hcont : ContinuousOn f (Set.Icc 1 3))
  (h1 : f 1 * f 2 < 0)
  (h2 : f 2 * f 3 < 0) :
  ∃ (x y : ℝ), x ∈ Set.Ioo 1 3 ∧ y ∈ Set.Ioo 1 3 ∧ x ≠ y ∧ f x = 0 ∧ f y = 0 :=
sorry

end NUMINAMATH_CALUDE_function_with_two_zeros_l3479_347911


namespace NUMINAMATH_CALUDE_product_not_in_set_l3479_347910

def a (n : ℕ) : ℕ := n^2 + n + 1

theorem product_not_in_set : ∃ m k : ℕ, ¬∃ n : ℕ, a m * a k = a n := by
  sorry

end NUMINAMATH_CALUDE_product_not_in_set_l3479_347910


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l3479_347916

theorem fractional_equation_solution_range (m x : ℝ) : 
  (m / (x - 2) + 1 = x / (2 - x)) →  -- Given equation
  (x ≥ 0) →                         -- Non-negative solution
  (x ≠ 2) →                         -- Avoid division by zero
  (m ≤ 2 ∧ m ≠ -2) :=               -- Range of m
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l3479_347916


namespace NUMINAMATH_CALUDE_angle_A_value_l3479_347994

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem angle_A_value (abc : Triangle) (h : abc.b = 2 * abc.a * Real.sin abc.B) :
  abc.A = π/6 ∨ abc.A = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_l3479_347994


namespace NUMINAMATH_CALUDE_sandy_safe_moon_tokens_l3479_347988

theorem sandy_safe_moon_tokens :
  ∀ (T : ℕ),
    (T / 2 = T / 8 + 375000) →
    T = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_sandy_safe_moon_tokens_l3479_347988


namespace NUMINAMATH_CALUDE_odd_function_extension_l3479_347915

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then Real.exp (-x) + 2 * x - 1
  else -Real.exp x + 2 * x + 1

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x < 0, f x = Real.exp (-x) + 2 * x - 1) →
  (∀ x ≥ 0, f x = -Real.exp x + 2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3479_347915


namespace NUMINAMATH_CALUDE_point_outside_circle_l3479_347970

theorem point_outside_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0 → (a - x)^2 + (2 - y)^2 > 0) ↔ 
  (2 < a ∧ a < 9/4) :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3479_347970


namespace NUMINAMATH_CALUDE_not_always_greater_than_original_l3479_347980

theorem not_always_greater_than_original : ¬ (∀ x : ℝ, 1.25 * x > x) := by sorry

end NUMINAMATH_CALUDE_not_always_greater_than_original_l3479_347980


namespace NUMINAMATH_CALUDE_recliner_price_drop_l3479_347922

theorem recliner_price_drop 
  (initial_quantity : ℝ) 
  (initial_price : ℝ) 
  (quantity_increase_ratio : ℝ) 
  (revenue_increase_ratio : ℝ) 
  (h1 : quantity_increase_ratio = 1.60) 
  (h2 : revenue_increase_ratio = 1.2800000000000003) : 
  let new_quantity := initial_quantity * quantity_increase_ratio
  let new_price := initial_price * (revenue_increase_ratio / quantity_increase_ratio)
  new_price / initial_price = 0.80 := by
sorry

end NUMINAMATH_CALUDE_recliner_price_drop_l3479_347922


namespace NUMINAMATH_CALUDE_crease_length_l3479_347930

/-- Given a rectangular piece of paper 8 inches wide, when folded such that one corner
    touches the opposite side and forms an angle θ at the corner where the crease starts,
    the length L of the crease is equal to 8 cos(θ). -/
theorem crease_length (θ : Real) (L : Real) : L = 8 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_crease_length_l3479_347930


namespace NUMINAMATH_CALUDE_two_tangent_or_parallel_lines_l3479_347908

/-- A parabola in the x-y plane defined by y^2 = -8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -8 * p.1}

/-- The point P through which the lines must pass -/
def P : ℝ × ℝ := (-2, -4)

/-- A line that passes through point P and has only one common point with the parabola -/
def TangentOrParallelLine (l : Set (ℝ × ℝ)) : Prop :=
  P ∈ l ∧ (∃! p, p ∈ l ∩ Parabola)

/-- There are exactly two lines that pass through P and have only one common point with the parabola -/
theorem two_tangent_or_parallel_lines : 
  ∃! (l1 l2 : Set (ℝ × ℝ)), l1 ≠ l2 ∧ TangentOrParallelLine l1 ∧ TangentOrParallelLine l2 ∧ 
  (∀ l, TangentOrParallelLine l → l = l1 ∨ l = l2) :=
sorry

end NUMINAMATH_CALUDE_two_tangent_or_parallel_lines_l3479_347908


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l3479_347943

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2/3 ∧ ∀ a b c : ℝ, b^2 / (3 * a^2 + c^2) ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l3479_347943


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3479_347975

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- State the theorem
theorem solution_set_implies_m_value :
  ∃ m : ℝ, (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3479_347975


namespace NUMINAMATH_CALUDE_complement_of_union_l3479_347904

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3479_347904


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3479_347938

/-- The statement "at least one of x and y is greater than 1" is neither a sufficient nor a necessary condition for x^2 + y^2 > 2 -/
theorem not_sufficient_not_necessary (x y : ℝ) : 
  ¬(((x > 1 ∨ y > 1) → x^2 + y^2 > 2) ∧ (x^2 + y^2 > 2 → (x > 1 ∨ y > 1))) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3479_347938


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l3479_347990

/-- The fixed point through which all lines of the form kx-y+1=3k pass -/
def fixed_point : ℝ × ℝ := (3, 1)

/-- The equation of the line parameterized by k -/
def line_equation (k x y : ℝ) : Prop := k*x - y + 1 = 3*k

/-- Theorem stating that the fixed_point is the unique point through which all lines pass -/
theorem fixed_point_theorem :
  ∀ (k : ℝ), line_equation k (fixed_point.1) (fixed_point.2) ∧
  ∀ (x y : ℝ), (∀ (k : ℝ), line_equation k x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l3479_347990


namespace NUMINAMATH_CALUDE_final_area_fraction_l3479_347914

/-- The fraction of area remaining after one iteration -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of iterations -/
def num_iterations : ℕ := 6

/-- The theorem stating the final fraction of area remaining -/
theorem final_area_fraction :
  remaining_fraction ^ num_iterations = 262144 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_final_area_fraction_l3479_347914


namespace NUMINAMATH_CALUDE_circular_pond_area_l3479_347925

/-- Given a circular pond with a diameter of 20 feet and a line from the midpoint 
    of this diameter to the circumference of 18 feet, prove that the area of the 
    pond is 224π square feet. -/
theorem circular_pond_area (diameter : ℝ) (midpoint_to_circle : ℝ) : 
  diameter = 20 → midpoint_to_circle = 18 → 
  ∃ (radius : ℝ), radius^2 * π = 224 * π := by sorry

end NUMINAMATH_CALUDE_circular_pond_area_l3479_347925


namespace NUMINAMATH_CALUDE_distance_before_meeting_is_100_l3479_347963

/-- The distance between two trains one hour before they meet -/
def distance_before_meeting (total_distance : ℝ) (speed_A speed_B : ℝ) (delay : ℝ) : ℝ :=
  let relative_speed := speed_A + speed_B
  let time_to_meet := (total_distance - speed_A * delay) / relative_speed
  relative_speed

/-- Theorem stating the distance between trains one hour before meeting -/
theorem distance_before_meeting_is_100 :
  distance_before_meeting 435 45 55 (40/60) = 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_before_meeting_is_100_l3479_347963


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l3479_347986

theorem quadratic_roots_real_and_equal (a k : ℝ) (ha : a > 0) (hk : k > 0) :
  let discriminant := 36 * k - 72 * a * k
  discriminant = 0 →
  ∃ x : ℝ, ∀ y : ℝ, a * y^2 - 6 * y * Real.sqrt k + 18 * k = 0 ↔ y = x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l3479_347986


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3479_347979

/-- The sum of the infinite series ∑_{k=1}^∞ (k^3 / 3^k) is equal to 6 -/
theorem infinite_series_sum : 
  (∑' k : ℕ+, (k : ℝ)^3 / 3^(k : ℝ)) = 6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3479_347979


namespace NUMINAMATH_CALUDE_cos_2x_plus_sin_pi_half_minus_x_properties_l3479_347984

/-- The function f(x) = cos(2x) + sin(π/2 - x) has both maximum and minimum values and is an even function. -/
theorem cos_2x_plus_sin_pi_half_minus_x_properties :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = Real.cos (2 * x) + Real.sin (Real.pi / 2 - x)) ∧
    (∃ (max min : ℝ), ∀ x, min ≤ f x ∧ f x ≤ max) ∧
    (∀ x, f (-x) = f x) := by
  sorry


end NUMINAMATH_CALUDE_cos_2x_plus_sin_pi_half_minus_x_properties_l3479_347984


namespace NUMINAMATH_CALUDE_irwin_two_point_baskets_l3479_347974

/-- Represents the number of baskets scored for each point value -/
structure BasketCount where
  two_point : ℕ
  five_point : ℕ
  eleven_point : ℕ
  thirteen_point : ℕ

/-- Calculates the product of point values for a given BasketCount -/
def pointValueProduct (b : BasketCount) : ℕ :=
  2^b.two_point * 5^b.five_point * 11^b.eleven_point * 13^b.thirteen_point

theorem irwin_two_point_baskets :
  ∀ b : BasketCount,
    pointValueProduct b = 2420 →
    b.eleven_point = 2 →
    b.two_point = 2 := by
  sorry

end NUMINAMATH_CALUDE_irwin_two_point_baskets_l3479_347974


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l3479_347913

/-- Represents a 9x9 grid filled with numbers 1 to 81 in row-wise order -/
def Grid := Fin 9 → Fin 9 → Fin 81

/-- The value at position (i, j) in the grid -/
def gridValue (i j : Fin 9) : Fin 81 :=
  ⟨i.val * 9 + j.val + 1, by sorry⟩

/-- The sum of the corner values in the grid -/
def cornerSum (g : Grid) : ℕ :=
  (g 0 0).val + (g 0 8).val + (g 8 0).val + (g 8 8).val

/-- Theorem stating that the sum of corner values in the defined grid is 164 -/
theorem corner_sum_is_164 :
  ∃ (g : Grid), cornerSum g = 164 :=
by sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l3479_347913


namespace NUMINAMATH_CALUDE_school_students_count_l3479_347999

/-- Given the number of pencils and erasers ordered, and the number of each item given to each student,
    calculate the number of students in the school. -/
def calculate_students (total_pencils : ℕ) (total_erasers : ℕ) (pencils_per_student : ℕ) (erasers_per_student : ℕ) : ℕ :=
  min (total_pencils / pencils_per_student) (total_erasers / erasers_per_student)

/-- Theorem stating that the number of students in the school is 65. -/
theorem school_students_count : calculate_students 195 65 3 1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l3479_347999


namespace NUMINAMATH_CALUDE_triangle_inequality_l3479_347919

/-- For any triangle with side lengths a, b, and c, 
    3(b+c-a)(c+a-b)(a+b-c) ≤ a²(b+c-a) + b²(c+a-b) + c²(a+b-c) holds. -/
theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3479_347919


namespace NUMINAMATH_CALUDE_sine_cosine_roots_l3479_347923

theorem sine_cosine_roots (θ : Real) (k : Real) 
  (h1 : θ > 0 ∧ θ < 2 * Real.pi)
  (h2 : (Real.sin θ)^2 - k * (Real.sin θ) + k + 1 = 0)
  (h3 : (Real.cos θ)^2 - k * (Real.cos θ) + k + 1 = 0) :
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_l3479_347923


namespace NUMINAMATH_CALUDE_unique_divisor_sum_product_l3479_347924

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

def product_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem unique_divisor_sum_product :
  ∃! P : ℕ, P > 0 ∧ sum_of_divisors P = 2 * P ∧ product_of_divisors P = P ^ 2 ∧ P = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_sum_product_l3479_347924


namespace NUMINAMATH_CALUDE_metal_detector_time_busier_days_is_30_l3479_347920

/-- Represents the time Mark spends on courthouse activities in a week -/
structure CourthouseTime where
  totalWeeklyTime : ℕ
  parkingTime : ℕ
  walkingTime : ℕ
  workDays : ℕ
  lessCrowdedDays : ℕ
  metalDetectorTimeLessCrowded : ℕ

/-- Calculates the time spent on metal detector on busier days -/
def metalDetectorTimeBusierDays (ct : CourthouseTime) : ℕ :=
  let totalParkingWalkingTime := ct.workDays * (ct.parkingTime + ct.walkingTime)
  let totalMetalDetectorTime := ct.totalWeeklyTime - totalParkingWalkingTime
  let metalDetectorTimeLessCrowdedTotal := ct.lessCrowdedDays * ct.metalDetectorTimeLessCrowded
  let metalDetectorTimeBusierTotal := totalMetalDetectorTime - metalDetectorTimeLessCrowdedTotal
  metalDetectorTimeBusierTotal / (ct.workDays - ct.lessCrowdedDays)

theorem metal_detector_time_busier_days_is_30 (ct : CourthouseTime) :
  ct.totalWeeklyTime = 130 ∧
  ct.parkingTime = 5 ∧
  ct.walkingTime = 3 ∧
  ct.workDays = 5 ∧
  ct.lessCrowdedDays = 3 ∧
  ct.metalDetectorTimeLessCrowded = 10 →
  metalDetectorTimeBusierDays ct = 30 := by
  sorry


end NUMINAMATH_CALUDE_metal_detector_time_busier_days_is_30_l3479_347920


namespace NUMINAMATH_CALUDE_assignment_operation_l3479_347928

theorem assignment_operation (A : Int) : A = 15 → -A + 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_assignment_operation_l3479_347928


namespace NUMINAMATH_CALUDE_tangent_line_properties_l3479_347967

/-- Parabola C: x^2 = 4y with focus F(0, 1) -/
structure Parabola where
  C : ℝ → ℝ
  F : ℝ × ℝ
  h : C = fun x ↦ (x^2) / 4
  focus : F = (0, 1)

/-- Line through P(a, -2) forming tangents to C at A(x₁, y₁) and B(x₂, y₂) -/
structure TangentLine (C : Parabola) where
  a : ℝ
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : P = (a, -2)
  h₂ : A.2 = C.C A.1
  h₃ : B.2 = C.C B.1

/-- Circumcenter of triangle PAB -/
def circumcenter (C : Parabola) (L : TangentLine C) : ℝ × ℝ := sorry

/-- Main theorem -/
theorem tangent_line_properties (C : Parabola) (L : TangentLine C) :
  let (x₁, y₁) := L.A
  let (x₂, y₂) := L.B
  let M := circumcenter C L
  (x₁ * x₂ + y₁ * y₂ = -4) ∧
  (∃ r : ℝ, (M.1 - C.F.1)^2 + (M.2 - C.F.2)^2 = r^2 ∧
            (L.P.1 - C.F.1)^2 + (L.P.2 - C.F.2)^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l3479_347967


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3479_347958

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3479_347958


namespace NUMINAMATH_CALUDE_product_real_part_l3479_347952

/-- Given two complex numbers a and b with magnitudes 3 and 5 respectively,
    prove that the positive real part of their product is 6√6. -/
theorem product_real_part (a b : ℂ) (ha : Complex.abs a = 3) (hb : Complex.abs b = 5) :
  (a * b).re = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_product_real_part_l3479_347952


namespace NUMINAMATH_CALUDE_line_segment_parameterization_l3479_347941

theorem line_segment_parameterization (m n p q : ℝ) : 
  (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ 
    1 = m * (-1) + n ∧ 
    -3 = p * (-1) + q ∧
    6 = m * 1 + n ∧ 
    5 = p * 1 + q) →
  m^2 + n^2 + p^2 + q^2 = 99 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_l3479_347941


namespace NUMINAMATH_CALUDE_tylers_meal_combinations_l3479_347993

/-- The number of meat options available --/
def meat_options : ℕ := 4

/-- The number of vegetable options available --/
def vegetable_options : ℕ := 5

/-- The number of dessert options available --/
def dessert_options : ℕ := 5

/-- The number of vegetables Tyler must choose --/
def vegetables_to_choose : ℕ := 3

/-- The number of desserts Tyler must choose --/
def desserts_to_choose : ℕ := 2

/-- The number of unique meal combinations Tyler can choose --/
def unique_meals : ℕ := meat_options * (Nat.choose vegetable_options vegetables_to_choose) * (Nat.choose dessert_options desserts_to_choose)

theorem tylers_meal_combinations :
  unique_meals = 400 :=
sorry

end NUMINAMATH_CALUDE_tylers_meal_combinations_l3479_347993


namespace NUMINAMATH_CALUDE_part_one_part_two_l3479_347917

-- Define the set M
def M (D : Set ℝ) : Set (ℝ → ℝ) :=
  {f | ∀ x y, (x + y) / 2 ∈ D → f ((x + y) / 2) ≥ (f x + f y) / 2 ∧
       (f ((x + y) / 2) = (f x + f y) / 2 ↔ x = y)}

-- Part 1
theorem part_one (f : ℝ → ℝ) (h : f ∈ M (Set.Ioi 0)) :
  f 3 + f 5 ≤ 2 * f 4 := by sorry

-- Part 2
def g : ℝ → ℝ := λ x ↦ -x^2

theorem part_two : g ∈ M Set.univ := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3479_347917


namespace NUMINAMATH_CALUDE_A_subset_B_l3479_347954

def A : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem A_subset_B : A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l3479_347954


namespace NUMINAMATH_CALUDE_parallel_lines_equal_angles_l3479_347969

-- Define the concept of lines
variable (Line : Type)

-- Define the concept of angles between lines
variable (angle : Line → Line → ℝ)

-- Define the concept of parallel lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_lines_equal_angles (a b c : Line) (θ : ℝ) :
  parallel a b → angle a c = θ → angle b c = θ := by sorry

end NUMINAMATH_CALUDE_parallel_lines_equal_angles_l3479_347969


namespace NUMINAMATH_CALUDE_Q_has_35_digits_l3479_347996

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The product of two large numbers -/
def Q : ℕ := 6789432567123456789 * 98765432345678

/-- Theorem stating that Q has 35 digits -/
theorem Q_has_35_digits : num_digits Q = 35 := by sorry

end NUMINAMATH_CALUDE_Q_has_35_digits_l3479_347996


namespace NUMINAMATH_CALUDE_contrapositive_true_l3479_347921

theorem contrapositive_true : 
  (∀ a : ℝ, a > 2 → a^2 > 4) ↔ (∀ a : ℝ, a ≤ 2 → a^2 ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_true_l3479_347921


namespace NUMINAMATH_CALUDE_x_formula_l3479_347959

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (x (n + 1))^2 / (x n + 2 * x (n + 1))

theorem x_formula (n : ℕ) : x n = 1 / (double_factorial (2 * n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_formula_l3479_347959


namespace NUMINAMATH_CALUDE_identical_geometric_sequences_l3479_347937

/-- Two geometric sequences with the same first term -/
def geometric_sequence (a₀ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₀ * q^n

theorem identical_geometric_sequences
  (a₀ : ℝ) (q r : ℝ) :
  (∀ n : ℕ, ∃ s : ℝ, geometric_sequence a₀ q n + geometric_sequence a₀ r n = geometric_sequence (2 * a₀) s n) →
  q = r :=
sorry

end NUMINAMATH_CALUDE_identical_geometric_sequences_l3479_347937


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3479_347998

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3479_347998


namespace NUMINAMATH_CALUDE_walts_age_l3479_347964

theorem walts_age (walt_age music_teacher_age : ℕ) : 
  music_teacher_age = 3 * walt_age →
  (music_teacher_age + 12) = 2 * (walt_age + 12) →
  walt_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_walts_age_l3479_347964


namespace NUMINAMATH_CALUDE_oil_price_reduction_oil_price_reduction_result_l3479_347927

/-- Calculates the percentage reduction in oil price given the conditions -/
theorem oil_price_reduction (additional_oil : ℝ) (total_cost : ℝ) (reduced_price : ℝ) : ℝ :=
  let original_amount := (total_cost / reduced_price) - additional_oil
  let original_price := total_cost / original_amount
  let price_difference := original_price - reduced_price
  (price_difference / original_price) * 100

/-- The percentage reduction in oil price is approximately 24.99% -/
theorem oil_price_reduction_result : 
  ∃ ε > 0, |oil_price_reduction 5 500 25 - 24.99| < ε :=
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_oil_price_reduction_result_l3479_347927


namespace NUMINAMATH_CALUDE_sequence_increasing_k_bound_l3479_347960

theorem sequence_increasing_k_bound (k : ℝ) :
  (∀ n : ℕ+, (2 * n^2 + k * n) < (2 * (n + 1)^2 + k * (n + 1))) →
  k > -6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_k_bound_l3479_347960


namespace NUMINAMATH_CALUDE_delta_five_three_l3479_347926

def delta (a b : ℤ) : ℤ := 4 * a - 6 * b

theorem delta_five_three : delta 5 3 = 2 := by sorry

end NUMINAMATH_CALUDE_delta_five_three_l3479_347926


namespace NUMINAMATH_CALUDE_optimal_bottle_volume_l3479_347971

theorem optimal_bottle_volume (vol1 vol2 vol3 : ℕ) 
  (h1 : vol1 = 4200) (h2 : vol2 = 3220) (h3 : vol3 = 2520) :
  Nat.gcd vol1 (Nat.gcd vol2 vol3) = 140 := by
  sorry

end NUMINAMATH_CALUDE_optimal_bottle_volume_l3479_347971


namespace NUMINAMATH_CALUDE_divisible_by_seven_l3479_347961

theorem divisible_by_seven (n : ℕ) : 7 ∣ (3^(12*n + 1) + 2^(6*n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l3479_347961


namespace NUMINAMATH_CALUDE_political_science_majors_l3479_347918

/-- The number of applicants who majored in political science -/
def P : ℕ := 15

theorem political_science_majors :
  (40 : ℕ) = P + 15 + 10 ∧ 
  (20 : ℕ) = 5 + 15 ∧
  (10 : ℕ) = 40 - (P + 20) :=
by sorry

end NUMINAMATH_CALUDE_political_science_majors_l3479_347918


namespace NUMINAMATH_CALUDE_henrikhs_commute_l3479_347933

def blocks_to_office (x : ℕ) : Prop :=
  let walking_time := 60 * x
  let bicycle_time := 20 * x
  let skateboard_time := 40 * x
  (walking_time = bicycle_time + 480) ∧ 
  (walking_time = skateboard_time + 240)

theorem henrikhs_commute : ∃ (x : ℕ), blocks_to_office x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_henrikhs_commute_l3479_347933


namespace NUMINAMATH_CALUDE_prob_at_least_two_females_l3479_347912

/-- The probability of selecting at least two females when choosing three finalists
    from a group of eight contestants consisting of five females and three males. -/
theorem prob_at_least_two_females (total : ℕ) (females : ℕ) (males : ℕ) (finalists : ℕ) :
  total = 8 →
  females = 5 →
  males = 3 →
  finalists = 3 →
  (Nat.choose females 2 * Nat.choose males 1 + Nat.choose females 3) / Nat.choose total finalists = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_females_l3479_347912


namespace NUMINAMATH_CALUDE_find_number_l3479_347900

theorem find_number : ∃! x : ℝ, ((x / 12 - 32) * 3 - 45) = 159 := by sorry

end NUMINAMATH_CALUDE_find_number_l3479_347900


namespace NUMINAMATH_CALUDE_product_one_sum_at_least_two_l3479_347934

theorem product_one_sum_at_least_two (x : ℝ) (h1 : x > 0) (h2 : x * (1/x) = 1) : x + (1/x) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_at_least_two_l3479_347934


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3479_347962

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m-1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3479_347962


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3479_347957

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 →
  a * b + b * c + c * d + d * a ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3479_347957


namespace NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l3479_347935

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_triangular_array_rows : ∃ N : ℕ, 
  triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l3479_347935


namespace NUMINAMATH_CALUDE_pirate_treasure_chests_l3479_347945

theorem pirate_treasure_chests : ∀ (gold silver bronze chests : ℕ),
  gold = 3500 →
  silver = 500 →
  bronze = 2 * silver →
  (gold + silver + bronze) / 1000 = chests →
  chests * 1000 = gold + silver + bronze →
  chests = 5 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_chests_l3479_347945
