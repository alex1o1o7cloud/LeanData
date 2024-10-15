import Mathlib

namespace NUMINAMATH_CALUDE_sinusoidal_midline_l3096_309625

theorem sinusoidal_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_midline_l3096_309625


namespace NUMINAMATH_CALUDE_min_value_f_exists_min_f_l3096_309661

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Theorem for the minimum value of f(x)
theorem min_value_f (a : ℝ) :
  (a = 1 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 5) ∧
  (a ≤ -1 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 6 + 2*a) ∧
  (-1 < a ∧ a < 0 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 5 - a^2) :=
by sorry

-- Theorem for the existence of x that achieves the minimum
theorem exists_min_f (a : ℝ) :
  (a = 1 → ∃ x ∈ Set.Icc (-1) 0, f a x = 5) ∧
  (a ≤ -1 → ∃ x ∈ Set.Icc (-1) 0, f a x = 6 + 2*a) ∧
  (-1 < a ∧ a < 0 → ∃ x ∈ Set.Icc (-1) 0, f a x = 5 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_exists_min_f_l3096_309661


namespace NUMINAMATH_CALUDE_perimeter_pedal_ratio_l3096_309643

/-- A triangle in a 2D plane -/
structure Triangle where
  -- Define the triangle structure (you may need to adjust this based on your specific needs)
  -- For example, you could define it using three points or side lengths

/-- The pedal triangle of a given triangle -/
def pedal_triangle (t : Triangle) : Triangle :=
  sorry -- Definition of pedal triangle

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  sorry -- Definition of perimeter

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ :=
  sorry -- Definition of circumradius

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ :=
  sorry -- Definition of inradius

/-- Theorem: The ratio of a triangle's perimeter to its pedal triangle's perimeter
    is equal to the ratio of its circumradius to its inradius -/
theorem perimeter_pedal_ratio (t : Triangle) :
  (perimeter t) / (perimeter (pedal_triangle t)) = (circumradius t) / (inradius t) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_pedal_ratio_l3096_309643


namespace NUMINAMATH_CALUDE_smallest_congruent_number_l3096_309660

theorem smallest_congruent_number : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 6 = 1 ∧ 
  n % 7 = 1 ∧ 
  n % 8 = 1 ∧
  (∀ m : ℕ, m > 1 → m % 6 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  n = 169 := by
  sorry

end NUMINAMATH_CALUDE_smallest_congruent_number_l3096_309660


namespace NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l3096_309638

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the surface area of a cube -/
def cubeSurfaceArea (cube : Cube) : ℝ :=
  6 * cube.sideLength^2

/-- Theorem: Removing a cube from the center of a rectangular solid increases surface area -/
theorem surface_area_increase_after_cube_removal 
  (solid : RectangularSolid) 
  (cube : Cube) 
  (h1 : solid.length = 4) 
  (h2 : solid.width = 3) 
  (h3 : solid.height = 5) 
  (h4 : cube.sideLength = 2) :
  surfaceArea solid + cubeSurfaceArea cube = surfaceArea solid + 24 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l3096_309638


namespace NUMINAMATH_CALUDE_sum_max_min_xy_xz_yz_l3096_309693

/-- Given real numbers x, y, and z satisfying 5(x + y + z) = x^2 + y^2 + z^2,
    the sum of the maximum value of xy + xz + yz and 10 times its minimum value is 150. -/
theorem sum_max_min_xy_xz_yz (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  ∃ (N n : ℝ),
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 →
      a * b + a * c + b * c ≤ N ∧ n ≤ a * b + a * c + b * c) ∧
    N + 10 * n = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_max_min_xy_xz_yz_l3096_309693


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l3096_309675

/-- Given 50 observations with an initial mean, if one observation is corrected
    from 23 to 34, and the new mean becomes 36.5, then the initial mean must be 36.28. -/
theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℝ) :
  n = 50 ∧
  corrected_mean = 36.5 ∧
  (n : ℝ) * initial_mean + (34 - 23) = n * corrected_mean →
  initial_mean = 36.28 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l3096_309675


namespace NUMINAMATH_CALUDE_square_area_difference_l3096_309632

theorem square_area_difference (area_B : ℝ) (side_diff : ℝ) : 
  area_B = 81 → 
  side_diff = 4 → 
  let side_B := Real.sqrt area_B
  let side_A := side_B + side_diff
  side_A * side_A = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l3096_309632


namespace NUMINAMATH_CALUDE_problem_statement_l3096_309609

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 2)^2 + 25/((x - 2)^2) = -x + 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3096_309609


namespace NUMINAMATH_CALUDE_intersection_M_N_l3096_309659

def M : Set ℝ := {-4, -3, -2, -1, 0, 1}

def N : Set ℝ := {x : ℝ | x^2 + 3*x < 0}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3096_309659


namespace NUMINAMATH_CALUDE_degree_of_polynomial_power_l3096_309623

/-- The degree of the polynomial (5x^3 + 7)^10 is 30. -/
theorem degree_of_polynomial_power : 
  Polynomial.degree ((5 * X ^ 3 + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_power_l3096_309623


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_is_17_2_l3096_309603

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 
    a^2 + 4 * b^2 + 1 / (a * b) ≤ x^2 + 4 * y^2 + 1 / (x * y) :=
by sorry

theorem min_value_is_17_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_is_17_2_l3096_309603


namespace NUMINAMATH_CALUDE_point_inside_circle_l3096_309607

/-- Given an ellipse and an equation, prove that the point formed by the equation's roots is inside a specific circle -/
theorem point_inside_circle (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → b > 0 → a > b → -- Ellipse conditions
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → x^2 ≤ a^2 ∧ y^2 ≤ b^2) → -- Ellipse equation
  c/a = 1/2 → -- Eccentricity
  x₁^2 + (a*x₁/b)^2 = a^2 → -- x₁ is on the ellipse
  x₂^2 + (a*x₂/b)^2 = a^2 → -- x₂ is on the ellipse
  a*x₁^2 + b*x₁ - c = 0 → -- x₁ is a root of the equation
  a*x₂^2 + b*x₂ - c = 0 → -- x₂ is a root of the equation
  x₁^2 + x₂^2 < 2 -- Point (x₁, x₂) is inside the circle x^2 + y^2 = 2
  := by sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3096_309607


namespace NUMINAMATH_CALUDE_jessicas_balloons_l3096_309619

theorem jessicas_balloons (joan_balloons sally_balloons total_balloons : ℕ) 
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : total_balloons = 16) :
  total_balloons - (joan_balloons + sally_balloons) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_balloons_l3096_309619


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l3096_309633

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a) (a - 2)

/-- Theorem: If z(a) is a pure imaginary number, then a = 0 -/
theorem pure_imaginary_implies_a_zero : 
  ∀ a : ℝ, is_pure_imaginary (z a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l3096_309633


namespace NUMINAMATH_CALUDE_tangent_angle_range_l3096_309616

theorem tangent_angle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  let f : ℝ → ℝ := λ x => Real.log x + x / b
  let θ := Real.arctan (((1 / a) + (1 / b)) : ℝ)
  π / 4 ≤ θ ∧ θ < π / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_angle_range_l3096_309616


namespace NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l3096_309699

theorem sin_pi_12_plus_theta (θ : Real) 
  (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) : 
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l3096_309699


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3096_309662

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : (13 : ℚ) / 20 * total_votes = 39 + (total_votes - 39)) : 
  total_votes = 60 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3096_309662


namespace NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l3096_309618

theorem johnson_family_reunion_ratio :
  ∀ (total_adults : ℕ) (total_children : ℕ),
  total_children = 45 →
  (total_adults / 3 : ℚ) + 10 = total_adults →
  (total_adults : ℚ) / total_children = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l3096_309618


namespace NUMINAMATH_CALUDE_additional_distance_at_faster_speed_l3096_309683

/-- Given a person walking at two different speeds for a fixed distance, 
    calculate the additional distance covered at the faster speed in the same time. -/
theorem additional_distance_at_faster_speed 
  (actual_speed : ℝ) 
  (faster_speed : ℝ) 
  (actual_distance : ℝ) 
  (h1 : actual_speed = 10)
  (h2 : faster_speed = 15)
  (h3 : actual_distance = 30)
  : (faster_speed * (actual_distance / actual_speed)) - actual_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_additional_distance_at_faster_speed_l3096_309683


namespace NUMINAMATH_CALUDE_trouser_original_price_l3096_309605

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 50 ∧ discount_percentage = 0.5 ∧ sale_price = (1 - discount_percentage) * original_price →
  original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_trouser_original_price_l3096_309605


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3096_309671

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9/16) = Real.sqrt 137 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3096_309671


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3096_309695

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3096_309695


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l3096_309600

-- Define a function f: ℝ → ℝ with an inverse
def f : ℝ → ℝ := sorry

-- Assume f is bijective (has an inverse)
axiom f_bijective : Function.Bijective f

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Condition: f(x) + f(-x) = 4 for all x
axiom f_condition (x : ℝ) : f x + f (-x) = 4

-- Theorem to prove
theorem inverse_sum_theorem (x : ℝ) : f_inv (x - 3) + f_inv (7 - x) = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l3096_309600


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3096_309663

/-- Calculates the total number of handshakes in a basketball game scenario --/
def total_handshakes (players_per_team : ℕ) (num_referees : ℕ) (num_coaches : ℕ) : ℕ :=
  let player_handshakes := players_per_team * players_per_team
  let player_referee_handshakes := 2 * players_per_team * num_referees
  let coach_handshakes := num_coaches * (2 * players_per_team + num_referees)
  player_handshakes + player_referee_handshakes + coach_handshakes

/-- Theorem stating that the total number of handshakes in the given scenario is 102 --/
theorem basketball_handshakes :
  total_handshakes 6 3 2 = 102 := by
  sorry

#eval total_handshakes 6 3 2

end NUMINAMATH_CALUDE_basketball_handshakes_l3096_309663


namespace NUMINAMATH_CALUDE_box_weights_l3096_309646

theorem box_weights (a b c : ℝ) 
  (hab : a + b = 122)
  (hbc : b + c = 125)
  (hca : c + a = 127) : 
  a + b + c = 187 := by
  sorry

end NUMINAMATH_CALUDE_box_weights_l3096_309646


namespace NUMINAMATH_CALUDE_composite_function_ratio_l3096_309621

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composite_function_ratio : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_ratio_l3096_309621


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3096_309655

/-- The area of a square with adjacent vertices at (1,3) and (-4,6) is 34 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 34 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3096_309655


namespace NUMINAMATH_CALUDE_original_price_calculation_l3096_309628

def selling_price : ℝ := 1220
def gain_percentage : ℝ := 45.23809523809524

theorem original_price_calculation :
  let original_price := selling_price / (1 + gain_percentage / 100)
  ∃ ε > 0, |original_price - 840| < ε :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3096_309628


namespace NUMINAMATH_CALUDE_power_difference_equals_negative_sixteen_million_l3096_309678

theorem power_difference_equals_negative_sixteen_million : (3^4)^3 - (4^3)^4 = -16245775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_negative_sixteen_million_l3096_309678


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l3096_309674

theorem nesbitt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l3096_309674


namespace NUMINAMATH_CALUDE_mean_cat_weight_l3096_309649

def cat_weights : List ℝ := [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

theorem mean_cat_weight :
  let n : ℕ := cat_weights.length
  let sum : ℝ := cat_weights.sum
  sum / n = 101.64 := by sorry

end NUMINAMATH_CALUDE_mean_cat_weight_l3096_309649


namespace NUMINAMATH_CALUDE_quarter_equals_two_eighths_l3096_309658

theorem quarter_equals_two_eighths : (1 : ℚ) / 4 = 1 / 8 + 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_quarter_equals_two_eighths_l3096_309658


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l3096_309602

theorem a_gt_abs_b_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) := by
sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l3096_309602


namespace NUMINAMATH_CALUDE_train_crossing_time_l3096_309650

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 54 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3096_309650


namespace NUMINAMATH_CALUDE_first_group_size_l3096_309689

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℝ := 28

/-- The number of men in the second group -/
def men_second_group : ℝ := 20

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℝ := 22.4

/-- The work done by a group is inversely proportional to the time taken -/
axiom work_time_inverse_proportion {men days : ℝ} : men * days = (men_second_group * days_second_group)

theorem first_group_size : ∃ (men : ℝ), men * days_first_group = men_second_group * days_second_group ∧ men = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l3096_309689


namespace NUMINAMATH_CALUDE_marias_purse_value_l3096_309630

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of nickels in Maria's purse -/
def num_nickels : ℕ := 2

/-- The number of dimes in Maria's purse -/
def num_dimes : ℕ := 3

/-- The number of quarters in Maria's purse -/
def num_quarters : ℕ := 2

theorem marias_purse_value :
  (num_nickels * nickel_value + num_dimes * dime_value + num_quarters * quarter_value) * 100 / cents_per_dollar = 90 := by
  sorry

end NUMINAMATH_CALUDE_marias_purse_value_l3096_309630


namespace NUMINAMATH_CALUDE_reeboks_sold_count_l3096_309653

def quota : ℕ := 1000
def adidas_price : ℕ := 45
def nike_price : ℕ := 60
def reebok_price : ℕ := 35
def nike_sold : ℕ := 8
def adidas_sold : ℕ := 6
def above_goal : ℕ := 65

theorem reeboks_sold_count :
  ∃ (reebok_sold : ℕ),
    reebok_sold * reebok_price + nike_sold * nike_price + adidas_sold * adidas_price = quota + above_goal ∧
    reebok_sold = 9 := by
  sorry

end NUMINAMATH_CALUDE_reeboks_sold_count_l3096_309653


namespace NUMINAMATH_CALUDE_factorial_calculation_l3096_309680

theorem factorial_calculation : (4 * Nat.factorial 6 + 32 * Nat.factorial 5) / Nat.factorial 7 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l3096_309680


namespace NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l3096_309667

def U : Set Nat := {1,2,3,4,5,6,7,8}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_intersection_equals_specific_set :
  (U \ S) ∩ (U \ T) = {2,4,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l3096_309667


namespace NUMINAMATH_CALUDE_wire_division_proof_l3096_309612

/-- Calculates the number of equal parts a wire can be divided into -/
def wire_parts (total_length : ℕ) (part_length : ℕ) : ℕ :=
  total_length / part_length

/-- Proves that a wire of 64 inches divided into 16-inch parts results in 4 parts -/
theorem wire_division_proof :
  wire_parts 64 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_wire_division_proof_l3096_309612


namespace NUMINAMATH_CALUDE_problem_solution_l3096_309673

theorem problem_solution (A B : ℝ) (h1 : B + A + B = 814.8) (h2 : A = 10 * B) : A - B = 611.1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3096_309673


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3096_309629

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, |x + 5| ≥ m + 2) → m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3096_309629


namespace NUMINAMATH_CALUDE_olympiad_scores_l3096_309624

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (distinct : ∀ i j, i ≠ j → scores i ≠ scores j)
  (sum_condition : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_scores_l3096_309624


namespace NUMINAMATH_CALUDE_composite_divisors_theorem_l3096_309682

/-- A function that returns the set of proper divisors of a natural number -/
def proper_divisors (a : ℕ) : Set ℕ :=
  {d | d ∣ a ∧ 1 < d ∧ d < a}

/-- A function that increases each element of a set by 1 -/
def increase_by_one (S : Set ℕ) : Set ℕ :=
  {x + 1 | x ∈ S}

/-- The main theorem -/
theorem composite_divisors_theorem (n : ℕ) (h_composite : ¬ Prime n) :
  (∃ m : ℕ, increase_by_one (proper_divisors n) = proper_divisors m) ↔ n = 4 ∨ n = 8 := by
  sorry

#check composite_divisors_theorem

end NUMINAMATH_CALUDE_composite_divisors_theorem_l3096_309682


namespace NUMINAMATH_CALUDE_queen_jack_hands_count_l3096_309640

/-- The number of queens in a standard deck --/
def num_queens : ℕ := 4

/-- The number of jacks in a standard deck --/
def num_jacks : ℕ := 4

/-- The total number of queens and jacks --/
def total_queens_jacks : ℕ := num_queens + num_jacks

/-- The number of cards in a hand --/
def hand_size : ℕ := 5

/-- The number of 5-card hands containing only queens and jacks --/
def num_queen_jack_hands : ℕ := Nat.choose total_queens_jacks hand_size

theorem queen_jack_hands_count : num_queen_jack_hands = 56 := by
  sorry

end NUMINAMATH_CALUDE_queen_jack_hands_count_l3096_309640


namespace NUMINAMATH_CALUDE_membership_change_l3096_309681

theorem membership_change (initial_members : ℝ) (h : initial_members > 0) :
  let fall_increase := 0.04
  let spring_decrease := 0.19
  let fall_members := initial_members * (1 + fall_increase)
  let spring_members := fall_members * (1 - spring_decrease)
  let total_change := (spring_members - initial_members) / initial_members
  total_change = -0.1576 := by
sorry

end NUMINAMATH_CALUDE_membership_change_l3096_309681


namespace NUMINAMATH_CALUDE_local_minimum_at_zero_l3096_309651

/-- The function f(x) = (x^2 - 1)^3 + 1 has a local minimum at x = 0 -/
theorem local_minimum_at_zero (f : ℝ → ℝ) (h : f = λ x => (x^2 - 1)^3 + 1) :
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f 0 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_zero_l3096_309651


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3096_309652

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 5
  ∃ x₁ x₂ : ℝ, x₁ = 5/2 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3096_309652


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3096_309669

theorem square_perimeter_ratio (x y : ℝ) (h : x * Real.sqrt 2 = 1.5 * y * Real.sqrt 2) :
  (4 * x) / (4 * y) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3096_309669


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l3096_309611

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : y^2 = 4*x

/-- Theorem: For a parabola y^2 = 4x, if a line passing through its focus intersects 
    the parabola at points A and B, and the distance |AB| = 12, then x₁ + x₂ = 10 -/
theorem parabola_intersection_sum (A B : ParabolaPoint) 
  (focus_line : A.x ≠ B.x → (A.y - B.y) / (A.x - B.x) = (A.y + B.y) / (A.x + B.x - 2))
  (distance : (A.x - B.x)^2 + (A.y - B.y)^2 = 12^2) :
  A.x + B.x = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l3096_309611


namespace NUMINAMATH_CALUDE_equation_solution_l3096_309608

theorem equation_solution (m : ℕ) : 
  ((1^m : ℚ) / (5^m)) * ((1^16 : ℚ) / (4^16)) = 1 / (2 * (10^31)) → m = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3096_309608


namespace NUMINAMATH_CALUDE_exists_config_with_more_than_20_components_l3096_309672

/-- A configuration of diagonals on an 8x8 grid -/
def DiagonalConfiguration := Fin 8 → Fin 8 → Bool

/-- A point on the 8x8 grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Two points are connected if they are in the same cell or adjacent cells with connecting diagonals -/
def connected (config : DiagonalConfiguration) (p1 p2 : GridPoint) : Prop :=
  sorry

/-- A connected component is a maximal set of connected points -/
def ConnectedComponent (config : DiagonalConfiguration) := Set GridPoint

/-- The number of connected components in a configuration -/
def numComponents (config : DiagonalConfiguration) : ℕ :=
  sorry

/-- There exists a configuration with more than 20 connected components -/
theorem exists_config_with_more_than_20_components :
  ∃ (config : DiagonalConfiguration), numComponents config > 20 :=
sorry

end NUMINAMATH_CALUDE_exists_config_with_more_than_20_components_l3096_309672


namespace NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l3096_309620

theorem inequality_solution_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l3096_309620


namespace NUMINAMATH_CALUDE_jessica_rent_last_year_l3096_309698

/-- Calculates Jessica's monthly rent last year given the increase in expenses --/
theorem jessica_rent_last_year (food_cost_last_year car_insurance_last_year : ℕ)
  (rent_increase_percent food_increase_percent : ℚ)
  (car_insurance_multiplier : ℕ)
  (total_yearly_increase : ℕ) :
  food_cost_last_year = 200 →
  car_insurance_last_year = 100 →
  rent_increase_percent = 30 / 100 →
  food_increase_percent = 50 / 100 →
  car_insurance_multiplier = 3 →
  total_yearly_increase = 7200 →
  ∃ (rent_last_year : ℕ),
    rent_last_year = 1000 ∧
    12 * ((1 + rent_increase_percent) * rent_last_year - rent_last_year +
         (1 + food_increase_percent) * food_cost_last_year - food_cost_last_year +
         car_insurance_multiplier * car_insurance_last_year - car_insurance_last_year) =
    total_yearly_increase :=
by sorry

end NUMINAMATH_CALUDE_jessica_rent_last_year_l3096_309698


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3096_309670

theorem quadratic_minimum (x : ℝ) : 
  3 * x^2 - 18 * x + 2023 ≥ 1996 ∧ ∃ y : ℝ, 3 * y^2 - 18 * y + 2023 = 1996 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3096_309670


namespace NUMINAMATH_CALUDE_income_calculation_l3096_309676

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →
  income - expenditure = savings →
  savings = 3800 →
  income = 19000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l3096_309676


namespace NUMINAMATH_CALUDE_matrix_N_property_l3096_309687

variable (N : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_N_property (h1 : N.mulVec ![3, -2] = ![4, 1])
                          (h2 : N.mulVec ![-2, 4] = ![0, 2]) :
  N.mulVec ![7, 0] = ![14, 7] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l3096_309687


namespace NUMINAMATH_CALUDE_twelve_sided_polygon_area_l3096_309665

/-- A 12-sided polygon composed of squares and triangles on a grid --/
structure TwelveSidedPolygon where
  center_square : ℝ  -- Area of the center square
  corner_triangles : ℝ  -- Number of corner triangles
  side_triangles : ℝ  -- Number of effective side triangles
  unit_square_area : ℝ  -- Area of a unit square
  unit_triangle_area : ℝ  -- Area of a unit right triangle

/-- The area of the 12-sided polygon --/
def polygon_area (p : TwelveSidedPolygon) : ℝ :=
  p.center_square * p.unit_square_area +
  p.corner_triangles * p.unit_triangle_area +
  p.side_triangles * p.unit_square_area

/-- Theorem stating that the area of the specific 12-sided polygon is 13 square units --/
theorem twelve_sided_polygon_area :
  ∀ (p : TwelveSidedPolygon),
  p.center_square = 9 ∧
  p.corner_triangles = 4 ∧
  p.side_triangles = 4 ∧
  p.unit_square_area = 1 ∧
  p.unit_triangle_area = 1/2 →
  polygon_area p = 13 := by
  sorry

end NUMINAMATH_CALUDE_twelve_sided_polygon_area_l3096_309665


namespace NUMINAMATH_CALUDE_intersection_with_y_axis_l3096_309601

/-- The intersection point of y = -4x + 2 with the y-axis is (0, 2) -/
theorem intersection_with_y_axis :
  let f (x : ℝ) := -4 * x + 2
  (0, f 0) = (0, 2) := by sorry

end NUMINAMATH_CALUDE_intersection_with_y_axis_l3096_309601


namespace NUMINAMATH_CALUDE_expression_simplification_l3096_309688

theorem expression_simplification (y : ℝ) : 
  4*y + 9*y^2 + 8 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3096_309688


namespace NUMINAMATH_CALUDE_arrangements_count_l3096_309657

/-- The number of people to be arranged -/
def num_people : ℕ := 5

/-- The number of positions available -/
def num_positions : ℕ := 4

/-- Function to calculate the number of arrangements -/
def calculate_arrangements (n_people : ℕ) (n_positions : ℕ) : ℕ :=
  -- Arrangements when A is selected (can't be in position A)
  (n_positions - 1) * (Nat.factorial (n_positions - 1)) +
  -- Arrangements when A is not selected
  (Nat.factorial n_positions)

/-- Theorem stating the number of arrangements -/
theorem arrangements_count :
  calculate_arrangements num_people num_positions = 42 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l3096_309657


namespace NUMINAMATH_CALUDE_larger_number_problem_l3096_309645

theorem larger_number_problem (x y : ℕ) : 
  x * y = 40 → x + y = 13 → max x y = 8 := by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3096_309645


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l3096_309647

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 6/5

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l3096_309647


namespace NUMINAMATH_CALUDE_circle_radius_relation_l3096_309627

theorem circle_radius_relation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (π * x^2 = π * y^2) → (2 * π * x = 20 * π) → y / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_relation_l3096_309627


namespace NUMINAMATH_CALUDE_granola_discounted_price_l3096_309666

/-- Calculates the discounted price per bag of granola given the following conditions:
    - Cost of ingredients per bag
    - Total number of bags made
    - Original selling price per bag
    - Number of bags sold at original price
    - Total net profit -/
def discounted_price (cost_per_bag : ℚ) (total_bags : ℕ) (original_price : ℚ)
                     (bags_sold_full_price : ℕ) (net_profit : ℚ) : ℚ :=
  let total_cost := cost_per_bag * total_bags
  let full_price_revenue := original_price * bags_sold_full_price
  let total_revenue := net_profit + total_cost
  let discounted_revenue := total_revenue - full_price_revenue
  let discounted_bags := total_bags - bags_sold_full_price
  discounted_revenue / discounted_bags

theorem granola_discounted_price :
  discounted_price 3 20 6 15 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_granola_discounted_price_l3096_309666


namespace NUMINAMATH_CALUDE_not_always_possible_to_reduce_box_dimension_counterexample_exists_l3096_309679

/-- Represents a rectangular parallelepiped with dimensions length, width, and height -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a box containing parallelepipeds -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  contents : List Parallelepiped

/-- Predicate to check if a parallelepiped fits in a box -/
def fits_in_box (p : Parallelepiped) (b : Box) : Prop :=
  p.length ≤ b.length ∧ p.width ≤ b.width ∧ p.height ≤ b.height

/-- Predicate to check if all parallelepipeds in a list fit in a box -/
def all_fit_in_box (ps : List Parallelepiped) (b : Box) : Prop :=
  ∀ p ∈ ps, fits_in_box p b

/-- Function to reduce one dimension of each parallelepiped -/
def reduce_parallelepipeds (ps : List Parallelepiped) : List Parallelepiped :=
  ps.map fun p => 
    let reduced_length := p.length * 0.99
    let reduced_width := p.width * 0.99
    let reduced_height := p.height * 0.99
    ⟨reduced_length, reduced_width, reduced_height⟩

/-- Theorem stating that it's not always possible to reduce a box dimension -/
theorem not_always_possible_to_reduce_box_dimension 
  (original_box : Box) 
  (reduced_parallelepipeds : List Parallelepiped) : Prop :=
  ∃ (reduced_box : Box), 
    (reduced_box.length < original_box.length ∨ 
     reduced_box.width < original_box.width ∨ 
     reduced_box.height < original_box.height) ∧
    all_fit_in_box reduced_parallelepipeds reduced_box →
    False

/-- Main theorem -/
theorem counterexample_exists : ∃ (original_box : Box) (original_parallelepipeds : List Parallelepiped),
  all_fit_in_box original_parallelepipeds original_box ∧
  not_always_possible_to_reduce_box_dimension original_box (reduce_parallelepipeds original_parallelepipeds) := by
  sorry

end NUMINAMATH_CALUDE_not_always_possible_to_reduce_box_dimension_counterexample_exists_l3096_309679


namespace NUMINAMATH_CALUDE_original_salary_l3096_309606

def salary_change (x : ℝ) : ℝ := (1 + 0.1) * (1 - 0.05) * x

theorem original_salary : 
  ∃ (x : ℝ), salary_change x = 2090 ∧ x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_original_salary_l3096_309606


namespace NUMINAMATH_CALUDE_weight_problem_l3096_309692

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions --/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →  -- average weight of a, b, and c is 45 kg
  (b + c) / 2 = 44 →      -- average weight of b and c is 44 kg
  b = 33 →                -- weight of b is 33 kg
  (a + b) / 2 = 40        -- average weight of a and b is 40 kg
:= by sorry

end NUMINAMATH_CALUDE_weight_problem_l3096_309692


namespace NUMINAMATH_CALUDE_min_even_integers_l3096_309604

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 28 →
  a + b + c + d = 45 →
  a + b + c + d + e + f = 63 →
  ∃ (n : ℕ), n ≥ 1 ∧ 
    ∀ (m : ℕ), (∃ (evens : Finset ℤ), evens.card = m ∧ 
      (∀ x ∈ evens, x % 2 = 0) ∧ 
      evens ⊆ {a, b, c, d, e, f}) → 
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l3096_309604


namespace NUMINAMATH_CALUDE_inequality_proof_l3096_309654

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3096_309654


namespace NUMINAMATH_CALUDE_sector_max_area_l3096_309684

/-- Given a sector with perimeter 4, its area is maximized when the central angle equals 2 -/
theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 4) :
  let α := l / r
  let area := (1 / 2) * r * l
  (∀ r' l', 2 * r' + l' = 4 → (1 / 2) * r' * l' ≤ area) →
  α = 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l3096_309684


namespace NUMINAMATH_CALUDE_total_flowers_l3096_309697

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) (h1 : num_pots = 544) (h2 : flowers_per_pot = 32) :
  num_pots * flowers_per_pot = 17408 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_l3096_309697


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3096_309691

def binomial (n k : ℕ) : ℕ := (Nat.choose n k)

theorem biased_coin_probability : ∃ (h : ℚ), 
  (0 < h ∧ h < 1) ∧ 
  (binomial 6 2 : ℚ) * h^2 * (1-h)^4 = (binomial 6 3 : ℚ) * h^3 * (1-h)^3 → 
  (binomial 6 4 : ℚ) * h^4 * (1-h)^2 = 19440 / 117649 :=
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3096_309691


namespace NUMINAMATH_CALUDE_distance_from_P_to_y_axis_l3096_309696

def point_to_y_axis_distance (x y : ℝ) : ℝ := |x|

theorem distance_from_P_to_y_axis :
  let P : ℝ × ℝ := (-3, -4)
  point_to_y_axis_distance P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_P_to_y_axis_l3096_309696


namespace NUMINAMATH_CALUDE_ball_bounces_on_table_l3096_309634

/-- Represents a rectangular table -/
structure Table where
  length : ℕ
  width : ℕ

/-- Calculates the number of bounces required for a ball to travel
    from one corner to the opposite corner of a rectangular table,
    moving at a 45° angle and bouncing off sides at 45° -/
def numberOfBounces (t : Table) : ℕ :=
  t.length + t.width - 2

theorem ball_bounces_on_table (t : Table) (h1 : t.length = 5) (h2 : t.width = 2) :
  numberOfBounces t = 5 := by
  sorry

#eval numberOfBounces { length := 5, width := 2 }

end NUMINAMATH_CALUDE_ball_bounces_on_table_l3096_309634


namespace NUMINAMATH_CALUDE_bus_stop_time_l3096_309613

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 64 → speed_with_stops = 48 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l3096_309613


namespace NUMINAMATH_CALUDE_root_sum_inverse_complement_l3096_309622

def cubic_polynomial (x : ℝ) : ℝ := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem root_sum_inverse_complement (a b c : ℝ) : 
  (cubic_polynomial a = 0) → 
  (cubic_polynomial b = 0) → 
  (cubic_polynomial c = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) → 
  (0 < a) → (a < 1) → 
  (0 < b) → (b < 1) → 
  (0 < c) → (c < 1) → 
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_inverse_complement_l3096_309622


namespace NUMINAMATH_CALUDE_grape_to_fruit_ratio_l3096_309636

def red_apples : ℕ := 9
def green_apples : ℕ := 4
def grape_bunches : ℕ := 3
def grapes_per_bunch : ℕ := 15
def yellow_bananas : ℕ := 6
def orange_oranges : ℕ := 2
def kiwis : ℕ := 5
def blueberries : ℕ := 30

def total_grapes : ℕ := grape_bunches * grapes_per_bunch

def total_fruits : ℕ := red_apples + green_apples + total_grapes + yellow_bananas + orange_oranges + kiwis + blueberries

theorem grape_to_fruit_ratio :
  (total_grapes : ℚ) / (total_fruits : ℚ) = 45 / 101 := by
  sorry

end NUMINAMATH_CALUDE_grape_to_fruit_ratio_l3096_309636


namespace NUMINAMATH_CALUDE_q_is_false_l3096_309615

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_is_false_l3096_309615


namespace NUMINAMATH_CALUDE_factor_expression_l3096_309642

theorem factor_expression (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3096_309642


namespace NUMINAMATH_CALUDE_car_trip_distance_theorem_l3096_309639

/-- Represents a segment of a car trip with speed and duration -/
structure TripSegment where
  speed : ℝ  -- Speed in miles per hour
  duration : ℝ  -- Duration in hours

/-- Calculates the distance traveled for a trip segment -/
def distance_traveled (segment : TripSegment) : ℝ :=
  segment.speed * segment.duration

/-- Represents a car trip with multiple segments -/
def CarTrip : Type := List TripSegment

/-- Calculates the total distance traveled for a car trip -/
def total_distance (trip : CarTrip) : ℝ :=
  trip.map distance_traveled |>.sum

theorem car_trip_distance_theorem (trip : CarTrip) : 
  trip = [
    { speed := 65, duration := 3 },
    { speed := 45, duration := 2 },
    { speed := 55, duration := 4 }
  ] → total_distance trip = 505 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_theorem_l3096_309639


namespace NUMINAMATH_CALUDE_circle_center_l3096_309690

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 8*y - 48 = 0

/-- The center of a circle given by its coordinates -/
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (x + h)^2 + (y - k)^2

theorem circle_center :
  is_center (-3) 4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l3096_309690


namespace NUMINAMATH_CALUDE_pet_store_cages_l3096_309631

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3096_309631


namespace NUMINAMATH_CALUDE_smallest_non_phd_count_l3096_309664

/-- The tournament structure -/
structure Tournament where
  total_participants : ℕ
  phd_participants : ℕ
  non_phd_participants : ℕ
  total_points : ℕ
  phd_points : ℕ
  non_phd_points : ℕ

/-- The theorem to prove -/
theorem smallest_non_phd_count (t : Tournament) : 
  199 ≤ t.total_participants ∧ 
  t.total_participants ≤ 229 ∧
  t.total_participants = t.phd_participants + t.non_phd_participants ∧
  t.total_points = t.total_participants * (t.total_participants - 1) / 2 ∧
  t.phd_points = t.phd_participants * (t.phd_participants - 1) / 2 ∧
  t.non_phd_points = t.non_phd_participants * (t.non_phd_participants - 1) / 2 ∧
  2 * (t.phd_points + t.non_phd_points) = t.total_points →
  t.non_phd_participants ≥ 105 ∧ 
  ∃ (t' : Tournament), t'.non_phd_participants = 105 ∧ 
    199 ≤ t'.total_participants ∧ 
    t'.total_participants ≤ 229 ∧
    t'.total_participants = t'.phd_participants + t'.non_phd_participants ∧
    t'.total_points = t'.total_participants * (t'.total_participants - 1) / 2 ∧
    t'.phd_points = t'.phd_participants * (t'.phd_participants - 1) / 2 ∧
    t'.non_phd_points = t'.non_phd_participants * (t'.non_phd_participants - 1) / 2 ∧
    2 * (t'.phd_points + t'.non_phd_points) = t'.total_points :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_phd_count_l3096_309664


namespace NUMINAMATH_CALUDE_expression_factorization_l3096_309626

theorem expression_factorization (x : ℝ) :
  (12 * x^4 + 34 * x^3 + 45 * x - 6) - (3 * x^4 - 7 * x^3 + 8 * x - 6) = x * (9 * x^3 + 41 * x^2 + 37) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3096_309626


namespace NUMINAMATH_CALUDE_two_pencils_length_l3096_309637

/-- The length of a pencil in cubes -/
def PencilLength : ℕ := 12

/-- The total length of two pencils -/
def TotalLength : ℕ := PencilLength + PencilLength

/-- Theorem: The total length of two pencils, each 12 cubes long, is 24 cubes -/
theorem two_pencils_length : TotalLength = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_pencils_length_l3096_309637


namespace NUMINAMATH_CALUDE_total_balloons_is_eighteen_l3096_309644

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := fred_balloons + sam_balloons + mary_balloons

theorem total_balloons_is_eighteen : total_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_is_eighteen_l3096_309644


namespace NUMINAMATH_CALUDE_crayon_count_l3096_309686

theorem crayon_count (num_people : ℕ) (crayons_per_person : ℕ) (h1 : num_people = 3) (h2 : crayons_per_person = 8) : 
  num_people * crayons_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_l3096_309686


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3096_309668

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  parallel (a.1 + (b m).1, a.2 + (b m).2) c → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3096_309668


namespace NUMINAMATH_CALUDE_min_angle_in_special_right_triangle_l3096_309617

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

end NUMINAMATH_CALUDE_min_angle_in_special_right_triangle_l3096_309617


namespace NUMINAMATH_CALUDE_climbing_solution_l3096_309641

/-- Represents the climbing problem with given conditions -/
def ClimbingProblem (v : ℝ) : Prop :=
  let t₁ : ℝ := 14 / 2 + 1  -- Time on first day
  let t₂ : ℝ := 14 / 2 - 1  -- Time on second day
  let v₁ : ℝ := v - 0.5     -- Speed on first day
  let v₂ : ℝ := v           -- Speed on second day
  (v₁ * t₁ + v₂ * t₂ = 52) ∧ (t₁ + t₂ = 14)

/-- The theorem stating the solution to the climbing problem -/
theorem climbing_solution : ∃ v : ℝ, ClimbingProblem v ∧ v = 4 := by
  sorry

end NUMINAMATH_CALUDE_climbing_solution_l3096_309641


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3096_309694

/-- A color type with four possible values -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- A point in the grid -/
structure Point where
  x : Fin 5
  y : Fin 41

/-- A coloring of the grid -/
def Coloring := Point → Color

/-- A rectangle in the grid -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if four points form a valid rectangle with integer side lengths -/
def IsValidRectangle (r : Rectangle) : Prop :=
  (r.p1.x = r.p2.x ∧ r.p3.x = r.p4.x ∧ r.p1.y = r.p3.y ∧ r.p2.y = r.p4.y) ∨
  (r.p1.x = r.p3.x ∧ r.p2.x = r.p4.x ∧ r.p1.y = r.p2.y ∧ r.p3.y = r.p4.y)

/-- Main theorem: There exists a monochromatic rectangle with integer side lengths -/
theorem monochromatic_rectangle_exists (c : Coloring) : 
  ∃ (r : Rectangle), IsValidRectangle r ∧ 
    c r.p1 = c r.p2 ∧ c r.p2 = c r.p3 ∧ c r.p3 = c r.p4 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3096_309694


namespace NUMINAMATH_CALUDE_sin_C_value_max_area_l3096_309656

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 5 ∧ 2 * Real.sin t.A = t.a * Real.cos t.B

-- Theorem 1: If c = 2, then sin C = 2/3
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) (hc : t.c = 2) :
  Real.sin t.C = 2/3 := by
  sorry

-- Theorem 2: Maximum area of triangle ABC
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  ∃ (max_area : ℝ), max_area = (5 * Real.sqrt 5) / 4 ∧
  ∀ (actual_area : ℝ), actual_area = (1/2) * t.a * t.b * Real.sin t.C → actual_area ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_sin_C_value_max_area_l3096_309656


namespace NUMINAMATH_CALUDE_remainder_equality_l3096_309610

theorem remainder_equality (x : ℕ+) (y : ℤ) 
  (h1 : ∃ k : ℤ, 200 = k * x.val + 5) 
  (h2 : ∃ m : ℤ, y = m * x.val + 5) : 
  y % x.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l3096_309610


namespace NUMINAMATH_CALUDE_percent_of_200_l3096_309677

theorem percent_of_200 : (25 / 100) * 200 = 50 := by sorry

end NUMINAMATH_CALUDE_percent_of_200_l3096_309677


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3096_309685

theorem point_in_second_quadrant (θ : Real) (h : π/2 < θ ∧ θ < π) :
  let P := (Real.tan θ, Real.sin θ)
  P.1 < 0 ∧ P.2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3096_309685


namespace NUMINAMATH_CALUDE_princess_daphne_necklaces_l3096_309648

def total_cost : ℕ := 240000
def necklace_cost : ℕ := 40000

def number_of_necklaces : ℕ := 3

theorem princess_daphne_necklaces :
  ∃ (n : ℕ), n * necklace_cost + 3 * necklace_cost = total_cost ∧ n = number_of_necklaces :=
by sorry

end NUMINAMATH_CALUDE_princess_daphne_necklaces_l3096_309648


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3096_309614

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, a > 0 → |a| > 0) ∧
  (∃ a : ℝ, |a| > 0 ∧ ¬(a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3096_309614


namespace NUMINAMATH_CALUDE_intersection_point_d_l3096_309635

theorem intersection_point_d (d : ℝ) : 
  (∀ x y : ℝ, (y = x + d ∧ x = -y + d) → (x = d - 1 ∧ y = d)) → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_d_l3096_309635
