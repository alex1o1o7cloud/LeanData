import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_transverse_axis_range_l927_92728

theorem hyperbola_transverse_axis_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ y : ℝ, y^2 / a^2 - (2*y)^2 / b^2 = 1) →
  b^2 = 1 - a^2 →
  0 < 2*a ∧ 2*a < 2*Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_transverse_axis_range_l927_92728


namespace NUMINAMATH_CALUDE_like_terms_exponents_l927_92763

/-- Given that 3x^(2n-1)y^m and -5x^m y^3 are like terms, prove that m = 3 and n = 2 -/
theorem like_terms_exponents (m n : ℤ) : 
  (∀ x y : ℝ, 3 * x^(2*n - 1) * y^m = -5 * x^m * y^3) → 
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l927_92763


namespace NUMINAMATH_CALUDE_monomial_exponent_equality_l927_92767

/-- Two monomials are of the same type if they have the same exponents for each variable. -/
def same_type_monomial (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

/-- The exponents of a monomial of the form x^a * y^b. -/
def monomial_exponents (a b : ℕ) : ℕ → ℕ
| 0 => a  -- exponent of x
| 1 => b  -- exponent of y
| _ => 0  -- all other variables have exponent 0

theorem monomial_exponent_equality (m : ℕ) :
  same_type_monomial (monomial_exponents (2 * m) 3) (monomial_exponents 6 3) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_equality_l927_92767


namespace NUMINAMATH_CALUDE_smallest_y_for_prime_abs_quadratic_l927_92702

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_quadratic (y : ℤ) : ℕ := Int.natAbs (5 * y^2 - 56 * y + 12)

theorem smallest_y_for_prime_abs_quadratic :
  (∀ y : ℤ, y < 11 → ¬(is_prime (abs_quadratic y))) ∧
  (is_prime (abs_quadratic 11)) :=
sorry

end NUMINAMATH_CALUDE_smallest_y_for_prime_abs_quadratic_l927_92702


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l927_92775

/-- The parabola equation -/
def parabola (a x y : ℝ) : Prop := y = a * x^2 + 5 * x + 2

/-- The line equation -/
def line (x y : ℝ) : Prop := y = -2 * x + 1

/-- The intersection condition -/
def intersect_once (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, parabola a p.1 p.2 ∧ line p.1 p.2

/-- The theorem statement -/
theorem parabola_line_intersection (a : ℝ) :
  intersect_once a ↔ a = 49 / 4 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l927_92775


namespace NUMINAMATH_CALUDE_pyramid_angle_closest_to_40_l927_92747

theorem pyramid_angle_closest_to_40 (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 2017) (h_lateral : lateral_edge = 2000) : 
  let angle := Real.arctan ((base_edge / Real.sqrt 2) / lateral_edge)
  let options := [30, 40, 50, 60]
  (40 : ℝ) ∈ options ∧ 
  ∀ x ∈ options, |angle - 40| ≤ |angle - x| :=
by sorry

end NUMINAMATH_CALUDE_pyramid_angle_closest_to_40_l927_92747


namespace NUMINAMATH_CALUDE_expression_simplification_l927_92726

theorem expression_simplification :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l927_92726


namespace NUMINAMATH_CALUDE_probability_inequality_l927_92744

theorem probability_inequality (p q : ℝ) (m n : ℕ+) 
  (h1 : p ≥ 0) (h2 : q ≥ 0) (h3 : p + q = 1) :
  (1 - p ^ (m : ℝ)) ^ (n : ℝ) + (1 - q ^ (n : ℝ)) ^ (m : ℝ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_inequality_l927_92744


namespace NUMINAMATH_CALUDE_solve_shelves_problem_l927_92741

def shelves_problem (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : Prop :=
  let remaining_books := initial_stock - books_sold
  remaining_books / books_per_shelf = 5

theorem solve_shelves_problem :
  shelves_problem 40 20 4 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_shelves_problem_l927_92741


namespace NUMINAMATH_CALUDE_age_problem_l927_92724

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- A is two years older than B
  b = 2 * c →  -- B is twice as old as C
  a + b + c = 47 →  -- Total age of A, B, and C is 47
  b = 18 :=  -- B's age is 18
by sorry

end NUMINAMATH_CALUDE_age_problem_l927_92724


namespace NUMINAMATH_CALUDE_calculation_one_calculation_two_l927_92733

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem for the first calculation
theorem calculation_one :
  (1 / Real.sqrt 0.04) + (1 / Real.sqrt 27) ^ (1/3) + (Real.sqrt 2 + 1)⁻¹ - 2 ^ (1/2) + (-2) ^ 0 = 8 := by sorry

-- Theorem for the second calculation
theorem calculation_two :
  (2/5) * lg 32 + lg 50 + Real.sqrt ((lg 3)^2 - lg 9 + 1) - lg (2/3) = 3 := by sorry

end NUMINAMATH_CALUDE_calculation_one_calculation_two_l927_92733


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l927_92761

/-- Proves that if the difference between compound interest and simple interest
    on a sum at 10% per annum for 2 years is Rs. 65, then the sum is Rs. 6500. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 65 → P = 6500 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l927_92761


namespace NUMINAMATH_CALUDE_total_shirts_washed_l927_92758

theorem total_shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) 
  (h1 : short_sleeve = 4) (h2 : long_sleeve = 5) : 
  short_sleeve + long_sleeve = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_washed_l927_92758


namespace NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l927_92740

theorem unique_solution_to_diophantine_equation :
  ∃! (a b c : ℕ+), 11^(a:ℕ) + 3^(b:ℕ) = (c:ℕ)^2 ∧ a = 4 ∧ b = 5 ∧ c = 122 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l927_92740


namespace NUMINAMATH_CALUDE_cubic_poly_roots_theorem_l927_92757

/-- A cubic polynomial p(x) = x^3 + cx + d -/
def cubic_poly (c d x : ℝ) : ℝ := x^3 + c*x + d

theorem cubic_poly_roots_theorem (c d : ℝ) :
  ∃ (u v : ℝ),
    (∀ x, cubic_poly c d x = 0 ↔ x = u ∨ x = v ∨ x = -u-v) ∧
    (∀ x, cubic_poly c (d + 360) x = 0 ↔ x = u+3 ∨ x = v-5 ∨ x = -u-v+2) →
    d = -2601 ∨ d = -693 :=
by sorry

end NUMINAMATH_CALUDE_cubic_poly_roots_theorem_l927_92757


namespace NUMINAMATH_CALUDE_common_chord_length_l927_92703

-- Define the circles in polar coordinates
def circle_O1 (ρ θ : ℝ) : Prop := ρ = 2
def circle_O2 (ρ θ : ℝ) : Prop := ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) = 2

-- Define the circles in rectangular coordinates
def rect_O1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def rect_O2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (A B : ℝ × ℝ),
    rect_O1 A.1 A.2 ∧ rect_O1 B.1 B.2 ∧
    rect_O2 A.1 A.2 ∧ rect_O2 B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt (2 * (2 + Real.sqrt 14)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_l927_92703


namespace NUMINAMATH_CALUDE_four_xy_even_l927_92717

theorem four_xy_even (x y : ℕ) (hx : Even x) (hy : Even y) (hxpos : 0 < x) (hypos : 0 < y) : 
  Even (4 * x * y) := by
  sorry

end NUMINAMATH_CALUDE_four_xy_even_l927_92717


namespace NUMINAMATH_CALUDE_r_value_when_n_is_2_l927_92780

theorem r_value_when_n_is_2 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 2^n + 1) 
  (h2 : r = 3^s - s) 
  (h3 : n = 2) : 
  r = 238 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_2_l927_92780


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l927_92793

theorem sum_of_a_and_b (a b : ℝ) : 
  a^2*b^2 + a^2 + b^2 + 1 - 2*a*b = 2*a*b → a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l927_92793


namespace NUMINAMATH_CALUDE_prayer_difference_l927_92794

/-- Represents the number of prayers for a pastor in a week -/
structure WeeklyPrayers where
  weekday : ℕ  -- Number of prayers on a weekday
  sunday : ℕ   -- Number of prayers on Sunday

/-- Calculates the total number of prayers in a week -/
def totalPrayers (wp : WeeklyPrayers) : ℕ :=
  6 * wp.weekday + wp.sunday

/-- Pastor Paul's prayer schedule -/
def paulPrayers : WeeklyPrayers where
  weekday := 20
  sunday := 40

/-- Pastor Bruce's prayer schedule -/
def brucePrayers : WeeklyPrayers where
  weekday := paulPrayers.weekday / 2
  sunday := 2 * paulPrayers.sunday

theorem prayer_difference :
  totalPrayers paulPrayers - totalPrayers brucePrayers = 20 := by
  sorry

end NUMINAMATH_CALUDE_prayer_difference_l927_92794


namespace NUMINAMATH_CALUDE_integral_evaluation_l927_92789

theorem integral_evaluation :
  ∫ x in (1 : ℝ)..2, (x + 1/x + 1/x^2) = 2 + Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_evaluation_l927_92789


namespace NUMINAMATH_CALUDE_sum_yz_zero_percent_of_x_l927_92746

theorem sum_yz_zero_percent_of_x (x y z : ℚ) 
  (h1 : (3/5) * (x - y) = (3/10) * (x + y))
  (h2 : (2/5) * (x + z) = (1/5) * (y + z))
  (h3 : (1/2) * (x - z) = (1/4) * (x + y + z)) :
  y + z = 0 * x :=
by sorry

end NUMINAMATH_CALUDE_sum_yz_zero_percent_of_x_l927_92746


namespace NUMINAMATH_CALUDE_journey_distance_l927_92701

/-- Calculates the total distance of a journey with multiple parts and a detour -/
theorem journey_distance (speed1 speed2 speed3 : ℝ) 
                         (time1 time2 time3 : ℝ) 
                         (detour_distance : ℝ) : 
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 30 →
  time1 = 1.5 →
  time2 = 1 →
  time3 = 2.25 →
  detour_distance = 10 →
  speed1 * time1 + speed2 * time2 + detour_distance + speed3 * time3 = 187.5 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l927_92701


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l927_92716

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l927_92716


namespace NUMINAMATH_CALUDE_salary_increase_after_four_years_l927_92735

theorem salary_increase_after_four_years (annual_raise : ℝ) (h : annual_raise = 0.1) :
  (1 + annual_raise)^4 - 1 > 0.45 := by sorry

end NUMINAMATH_CALUDE_salary_increase_after_four_years_l927_92735


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l927_92770

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 1/2| < 1/2 → x^3 < 1) ∧
  ∃ y : ℝ, y^3 < 1 ∧ |y - 1/2| ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l927_92770


namespace NUMINAMATH_CALUDE_complex_equation_roots_l927_92765

theorem complex_equation_roots : 
  let z₁ : ℂ := 1 + Real.sqrt 6 - (Real.sqrt 6 / 2) * Complex.I
  let z₂ : ℂ := 1 - Real.sqrt 6 + (Real.sqrt 6 / 2) * Complex.I
  (z₁^2 - 2*z₁ = 4 - 3*Complex.I) ∧ (z₂^2 - 2*z₂ = 4 - 3*Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l927_92765


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l927_92749

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 3 = 0 → 
  x₂^2 - 5*x₂ + 3 = 0 → 
  x₁^2 + x₂^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l927_92749


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l927_92732

theorem distance_from_origin_to_point : Real.sqrt ((-12)^2 + 9^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l927_92732


namespace NUMINAMATH_CALUDE_tan_alpha_value_l927_92772

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l927_92772


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l927_92776

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents the quadrilateral ABCD formed by the intersection of a plane with the cube -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculate the area of the quadrilateral ABCD -/
def quadrilateralArea (quad : Quadrilateral) : ℝ := sorry

/-- Main theorem: The area of quadrilateral ABCD is 2√3 -/
theorem area_of_quadrilateral_ABCD :
  let cube := Cube.mk 2
  let A := Point3D.mk 0 0 0
  let C := Point3D.mk 2 2 2
  let B := Point3D.mk (2/3) 2 0
  let D := Point3D.mk 2 (4/3) 2
  let quad := Quadrilateral.mk A B C D
  quadrilateralArea quad = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l927_92776


namespace NUMINAMATH_CALUDE_job_completion_time_l927_92777

theorem job_completion_time (time_a time_b : ℝ) (h1 : time_a = 5) (h2 : time_b = 15) :
  let combined_time := 1 / (1 / time_a + 1 / time_b)
  combined_time = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l927_92777


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l927_92786

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_condition (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : a 1 > 0) :
  (∀ h : a 3 < a 6, a 1 < a 3) ∧
  ¬(∀ h : a 1 < a 3, a 3 < a 6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l927_92786


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l927_92736

theorem unique_solution_factorial_equation :
  ∃! (a b : ℕ), a^2 + 2 = Nat.factorial b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l927_92736


namespace NUMINAMATH_CALUDE_younger_brother_age_l927_92754

/-- Represents the age of Viggo's younger brother -/
def brother_age : ℕ := sorry

/-- Represents Viggo's age -/
def viggo_age : ℕ := sorry

/-- The age difference between Viggo and his brother remains constant -/
axiom age_difference : viggo_age - brother_age = 12

/-- Viggo's age was 10 years more than twice his younger brother's age when his brother was 2 -/
axiom initial_age_relation : viggo_age - brother_age = 2 * 2 + 10 - 2

/-- The sum of their current ages is 32 -/
axiom current_age_sum : brother_age + viggo_age = 32

theorem younger_brother_age : brother_age = 10 := by sorry

end NUMINAMATH_CALUDE_younger_brother_age_l927_92754


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l927_92792

theorem polar_to_cartesian (r : ℝ) (θ : ℝ) (x y : ℝ) :
  r = 2 ∧ θ = 5 * π / 6 →
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  x = -Real.sqrt 3 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l927_92792


namespace NUMINAMATH_CALUDE_power_equality_implies_n_equals_four_l927_92719

theorem power_equality_implies_n_equals_four (n : ℕ) : 4^8 = 16^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_n_equals_four_l927_92719


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l927_92706

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l927_92706


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l927_92762

theorem consecutive_integers_cube_sum : 
  ∃ (a : ℕ), 
    (a > 0) ∧ 
    ((a - 1) * a * (a + 1) * (a + 2) = 12 * (4 * a + 2)) ∧ 
    ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3 = 224) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l927_92762


namespace NUMINAMATH_CALUDE_gcd_228_1995_l927_92710

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l927_92710


namespace NUMINAMATH_CALUDE_baker_cakes_l927_92729

theorem baker_cakes (initial_cakes : ℕ) : 
  (initial_cakes - 75 + 76 = 111) → initial_cakes = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l927_92729


namespace NUMINAMATH_CALUDE_undetermined_sum_l927_92782

/-- Operation # defined for non-negative integers a and b, and positive integer c -/
def sharp (a b c : ℕ) : ℕ := 4 * a^3 + 4 * b^3 + 8 * a^2 * b + c

/-- Operation * defined for non-negative integers a and b, and positive integer d -/
def star (a b d : ℕ) : ℕ := 2 * a^2 - 3 * b^2 + d^3

/-- Theorem stating that the value of (a + b) + 6 cannot be determined -/
theorem undetermined_sum (a b x c d : ℕ) (hc : c > 0) (hd : d > 0) 
  (h1 : sharp a x c = 250) (h2 : star a b d + x = 50) : 
  ∃ (a' b' x' c' d' : ℕ), 
    c' > 0 ∧ d' > 0 ∧
    sharp a' x' c' = 250 ∧ 
    star a' b' d' + x' = 50 ∧
    a + b + 6 ≠ a' + b' + 6 :=
sorry

end NUMINAMATH_CALUDE_undetermined_sum_l927_92782


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l927_92759

theorem tan_sum_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l927_92759


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l927_92745

theorem product_of_sum_of_squares (a b n k : ℝ) :
  let K := a^2 + b^2
  let P := n^2 + k^2
  K * P = (a*n + b*k)^2 + (a*k - b*n)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l927_92745


namespace NUMINAMATH_CALUDE_integer_operation_problem_l927_92799

theorem integer_operation_problem : ∃! x : ℤ, 
  ∃ r : ℤ, 0 ≤ r ∧ r < 7 ∧ ((x - 77) * 8 = 37 * 7 + r) ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_integer_operation_problem_l927_92799


namespace NUMINAMATH_CALUDE_water_capacity_equals_volume_l927_92742

/-- A cylindrical bucket -/
structure CylindricalBucket where
  volume : ℝ
  lateral_area : ℝ
  surface_area : ℝ

/-- The amount of water a cylindrical bucket can hold -/
def water_capacity (bucket : CylindricalBucket) : ℝ := sorry

/-- Theorem: The amount of water a cylindrical bucket can hold is equal to its volume -/
theorem water_capacity_equals_volume (bucket : CylindricalBucket) :
  water_capacity bucket = bucket.volume := sorry

end NUMINAMATH_CALUDE_water_capacity_equals_volume_l927_92742


namespace NUMINAMATH_CALUDE_x_cubed_minus_y_equals_plus_minus_17_l927_92727

theorem x_cubed_minus_y_equals_plus_minus_17 
  (x y : ℝ) 
  (h1 : x^2 = 4) 
  (h2 : |y| = 9) 
  (h3 : x * y < 0) : 
  x^3 - y = 17 ∨ x^3 - y = -17 := by
sorry

end NUMINAMATH_CALUDE_x_cubed_minus_y_equals_plus_minus_17_l927_92727


namespace NUMINAMATH_CALUDE_rabbit_speed_theorem_l927_92713

/-- Given a rabbit's speed, double it, add 4, and double again -/
def rabbit_speed_operation (speed : ℕ) : ℕ :=
  ((speed * 2) + 4) * 2

/-- Theorem stating that the rabbit speed operation on 45 results in 188 -/
theorem rabbit_speed_theorem : rabbit_speed_operation 45 = 188 := by
  sorry

#eval rabbit_speed_operation 45  -- This will evaluate to 188

end NUMINAMATH_CALUDE_rabbit_speed_theorem_l927_92713


namespace NUMINAMATH_CALUDE_binomial_prob_theorem_l927_92721

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The probability mass function of a binomial distribution -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

/-- Theorem: If X ~ B(10,p) with D(X) = 2.4 and P(X=4) > P(X=6), then p = 0.4 -/
theorem binomial_prob_theorem (X : BinomialRV) 
  (h_n : X.n = 10)
  (h_var : variance X = 2.4)
  (h_prob : pmf X 4 > pmf X 6) :
  X.p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_theorem_l927_92721


namespace NUMINAMATH_CALUDE_fred_car_washing_earnings_l927_92714

/-- Fred's earnings from various activities -/
structure FredEarnings where
  total : ℕ
  newspaper : ℕ
  car_washing : ℕ

/-- Theorem stating that Fred's car washing earnings are 74 dollars -/
theorem fred_car_washing_earnings (e : FredEarnings) 
  (h1 : e.total = 90)
  (h2 : e.newspaper = 16)
  (h3 : e.total = e.newspaper + e.car_washing) :
  e.car_washing = 74 := by
  sorry

end NUMINAMATH_CALUDE_fred_car_washing_earnings_l927_92714


namespace NUMINAMATH_CALUDE_floor_plus_self_equation_l927_92766

theorem floor_plus_self_equation (r : ℝ) : ⌊r⌋ + r = 15.4 ↔ r = 7.4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equation_l927_92766


namespace NUMINAMATH_CALUDE_bobs_assorted_candies_l927_92738

/-- The problem of calculating Bob's assorted candies -/
theorem bobs_assorted_candies 
  (total_candies : ℕ) 
  (chewing_gums : ℕ) 
  (chocolate_bars : ℕ) 
  (h1 : total_candies = 50)
  (h2 : chewing_gums = 15)
  (h3 : chocolate_bars = 20) :
  total_candies - (chewing_gums + chocolate_bars) = 15 :=
by sorry

end NUMINAMATH_CALUDE_bobs_assorted_candies_l927_92738


namespace NUMINAMATH_CALUDE_parabola_properties_l927_92705

/-- Parabola passing through a specific point -/
structure Parabola where
  a : ℝ
  passes_through : a * (2 - 3)^2 - 1 = 1

/-- The number of units to move the parabola up for one x-axis intersection -/
def move_up_units (p : Parabola) : ℝ := 1

/-- Theorem stating the properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  p.a = 2 ∧ 
  (∃! x : ℝ, 2 * (x - 3)^2 - 1 + move_up_units p = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l927_92705


namespace NUMINAMATH_CALUDE_two_digit_number_condition_l927_92768

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def satisfies_condition (n : ℕ) : Prop :=
  2 * (tens_digit n + units_digit n) = tens_digit n * units_digit n

theorem two_digit_number_condition :
  ∀ n : ℕ, is_valid_two_digit_number n ∧ satisfies_condition n ↔ n = 36 ∨ n = 44 ∨ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_condition_l927_92768


namespace NUMINAMATH_CALUDE_x_value_l927_92743

theorem x_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l927_92743


namespace NUMINAMATH_CALUDE_crocodile_count_l927_92778

theorem crocodile_count (total : ℕ) (alligators : ℕ) (vipers : ℕ) 
  (h1 : total = 50)
  (h2 : alligators = 23)
  (h3 : vipers = 5)
  (h4 : ∃ crocodiles : ℕ, total = crocodiles + alligators + vipers) :
  ∃ crocodiles : ℕ, crocodiles = 22 ∧ total = crocodiles + alligators + vipers :=
by sorry

end NUMINAMATH_CALUDE_crocodile_count_l927_92778


namespace NUMINAMATH_CALUDE_inequality_solution_range_l927_92798

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (a * Real.cos x - 1) * (a * x^2 - x + 16 * a) < 0) ↔ 
  (a < -1 ∨ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l927_92798


namespace NUMINAMATH_CALUDE_M_inequalities_l927_92774

/-- M_(n,k,h) is the maximum number of h-element subsets of an n-element set X with property P_k(X) -/
def M (n k h : ℕ) : ℕ := sorry

/-- The three inequalities for M_(n,k,h) -/
theorem M_inequalities (n k h : ℕ) (hn : n > 0) (hk : k > 0) (hh : h > 0) (hnkh : n ≥ k) (hkh : k ≥ h) :
  (M n k h ≤ (n / h) * M (n-1) (k-1) (h-1)) ∧
  (M n k h ≥ (n / (n-h)) * M (n-1) k h) ∧
  (M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h) :=
sorry

end NUMINAMATH_CALUDE_M_inequalities_l927_92774


namespace NUMINAMATH_CALUDE_company_workshops_l927_92722

/-- Given a total number of employees and a maximum workshop capacity,
    calculate the minimum number of workshops required. -/
def min_workshops (total_employees : ℕ) (max_capacity : ℕ) : ℕ :=
  (total_employees + max_capacity - 1) / max_capacity

/-- Theorem stating the minimum number of workshops required for the given problem -/
theorem company_workshops :
  let total_employees := 56
  let max_capacity := 15
  min_workshops total_employees max_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_company_workshops_l927_92722


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l927_92787

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 8*x + k = 0 ∧ y^2 + 8*y + k = 0) → k = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l927_92787


namespace NUMINAMATH_CALUDE_unique_solution_l927_92712

/-- Function that calculates the product of digits of a positive integer -/
def product_of_digits (x : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 12 is the only positive integer solution -/
theorem unique_solution :
  ∃! (x : ℕ+), product_of_digits x = x^2 - 10*x - 22 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l927_92712


namespace NUMINAMATH_CALUDE_age_sum_theorem_l927_92730

theorem age_sum_theorem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l927_92730


namespace NUMINAMATH_CALUDE_rectangle_ratio_l927_92700

-- Define the side length of the small squares
def small_square_side : ℝ := sorry

-- Define the side length of the large square
def large_square_side : ℝ := 3 * small_square_side

-- Define the length of the rectangle
def rectangle_length : ℝ := large_square_side

-- Define the width of the rectangle
def rectangle_width : ℝ := small_square_side

-- Theorem stating that the ratio of rectangle's length to width is 3
theorem rectangle_ratio : rectangle_length / rectangle_width = 3 := by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l927_92700


namespace NUMINAMATH_CALUDE_problem_statement_l927_92764

theorem problem_statement (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) 
  (heq : a^2 + 4*b^2 + c^2 - 2*c = 2) : 
  (a + 2*b + c ≤ 4) ∧ 
  (a = 2*b → 1/b + 1/(c-1) ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l927_92764


namespace NUMINAMATH_CALUDE_ms_hatcher_students_l927_92709

def total_students (third_graders : ℕ) : ℕ :=
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  let sixth_graders := (third_graders + fourth_graders) * 3 / 4
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem ms_hatcher_students :
  total_students 20 = 115 := by
  sorry

end NUMINAMATH_CALUDE_ms_hatcher_students_l927_92709


namespace NUMINAMATH_CALUDE_zongzi_theorem_l927_92756

/-- Represents the prices and quantities of zongzi --/
structure ZongziData where
  honey_price : ℝ
  meat_price : ℝ
  honey_quantity : ℕ
  meat_quantity : ℕ
  meat_sold_before : ℕ

/-- Represents the selling prices and profit --/
structure SaleData where
  honey_sell_price : ℝ
  meat_sell_price : ℝ
  meat_price_increase : ℝ
  meat_price_discount : ℝ
  total_profit : ℝ

/-- Main theorem stating the properties of zongzi prices and quantities --/
theorem zongzi_theorem (data : ZongziData) (sale : SaleData) : 
  data.meat_price = data.honey_price + 2.5 ∧ 
  300 / data.meat_price = 2 * (100 / data.honey_price) ∧
  data.honey_quantity = 100 ∧
  data.meat_quantity = 200 ∧
  sale.honey_sell_price = 6 ∧
  sale.meat_sell_price = 10 ∧
  sale.meat_price_increase = 1.1 ∧
  sale.meat_price_discount = 0.9 ∧
  sale.total_profit = 570 →
  data.honey_price = 5 ∧
  data.meat_price = 7.5 ∧
  data.meat_sold_before = 85 := by
  sorry

#check zongzi_theorem

end NUMINAMATH_CALUDE_zongzi_theorem_l927_92756


namespace NUMINAMATH_CALUDE_function_extrema_m_range_l927_92752

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

-- State the theorem
theorem function_extrema_m_range (m : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f m x ≤ f m x_max ∧ f m x_min ≤ f m x) →
  m < -3 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_m_range_l927_92752


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l927_92739

theorem quadratic_equation_root (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x => p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f (-1) = 0) →
  (f (-r * (p - q) / (p * (q - r))) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l927_92739


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l927_92795

theorem alternating_squares_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 242 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l927_92795


namespace NUMINAMATH_CALUDE_present_age_of_b_l927_92785

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → 
  (a = b + 11) → 
  b = 41 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_b_l927_92785


namespace NUMINAMATH_CALUDE_range_of_f_l927_92771

/-- The function f(x) = |x+3| - |x-5| -/
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

/-- The range of f is [-8, 18] -/
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l927_92771


namespace NUMINAMATH_CALUDE_roots_of_unity_real_roots_l927_92725

theorem roots_of_unity_real_roots (n : ℕ) : 
  ¬ (∀ z : ℂ, z^n = 1 → (z.im = 0 → z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_unity_real_roots_l927_92725


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l927_92779

theorem walking_speed_calculation (total_distance : ℝ) (running_speed : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 0.75)
  (h4 : ∃ (walking_time running_time : ℝ), 
    walking_time + running_time = total_time ∧ 
    walking_time * walking_speed = running_time * running_speed ∧
    walking_time * walking_speed = total_distance / 2) :
  walking_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l927_92779


namespace NUMINAMATH_CALUDE_max_value_of_f_l927_92708

/-- The function f(x) = |x| - |x - 3| -/
def f (x : ℝ) : ℝ := |x| - |x - 3|

/-- The maximum value of f(x) is 3 -/
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 3 ∧ ∀ x, f x ≤ M ∧ ∃ y, f y = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l927_92708


namespace NUMINAMATH_CALUDE_range_of_c_l927_92760

-- Define propositions p and q
def p (c : ℝ) : Prop := 2 < 3 * c
def q (c : ℝ) : Prop := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

-- Theorem statement
theorem range_of_c (c : ℝ) 
  (h : (p c ∨ q c) ∨ (p c ∧ q c)) : 
  2/3 < c ∧ c < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l927_92760


namespace NUMINAMATH_CALUDE_mikes_training_time_l927_92737

/-- Proves that Mike trained for 1 hour per day during the first week -/
theorem mikes_training_time (x : ℝ) : 
  (7 * x + 7 * (x + 3) = 35) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_mikes_training_time_l927_92737


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l927_92788

theorem purely_imaginary_z (a : ℝ) :
  let z : ℂ := a^2 - a + a * Complex.I
  (∃ b : ℝ, z = b * Complex.I ∧ b ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l927_92788


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l927_92796

/-- The percentage of motorists who exceed the speed limit -/
def speeding_percentage : ℝ := 25

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 60

/-- The percentage of motorists who receive speeding tickets -/
def ticket_percentage : ℝ := 10

theorem speeding_ticket_percentage :
  ticket_percentage = speeding_percentage * (1 - no_ticket_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l927_92796


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l927_92797

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The set of all possible outcomes when drawing two balls from the bag -/
def allOutcomes : Finset Outcome := sorry

/-- The event "Exactly one black ball" -/
def exactlyOneBlack (outcome : Outcome) : Prop :=
  (outcome.first = Color.Black ∧ outcome.second = Color.Red) ∨
  (outcome.first = Color.Red ∧ outcome.second = Color.Black)

/-- The event "Exactly two black balls" -/
def exactlyTwoBlack (outcome : Outcome) : Prop :=
  outcome.first = Color.Black ∧ outcome.second = Color.Black

theorem mutually_exclusive_not_contradictory :
  (∀ o : Outcome, ¬(exactlyOneBlack o ∧ exactlyTwoBlack o)) ∧
  (∃ o : Outcome, ¬exactlyOneBlack o ∧ ¬exactlyTwoBlack o) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l927_92797


namespace NUMINAMATH_CALUDE_system_solution_l927_92720

/-- Given a system of equations and the condition that a ≠ bc, 
    prove that x = 1, y = 0, and z = 0 are the solutions. -/
theorem system_solution (a b c : ℝ) (h : a ≠ b * c) :
  ∃! (x y z : ℝ), 
    a = (a * x + c * y) / (b * z + 1) ∧
    b = (b * x + y) / (b * z + 1) ∧
    c = (a * z + c) / (b * z + 1) ∧
    x = 1 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l927_92720


namespace NUMINAMATH_CALUDE_negation_equivalence_l927_92791

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 3 ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l927_92791


namespace NUMINAMATH_CALUDE_inequality_properties_l927_92781

theorem inequality_properties (x y : ℝ) (h : x > y) : 
  (x - 3 > y - 3) ∧ 
  (x / 3 > y / 3) ∧ 
  (x + 3 > y + 3) ∧ 
  (-3 * x < -3 * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l927_92781


namespace NUMINAMATH_CALUDE_investment_calculation_l927_92773

theorem investment_calculation (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (total_dividend : ℝ) :
  face_value = 100 →
  premium_rate = 0.2 →
  dividend_rate = 0.07 →
  total_dividend = 840.0000000000001 →
  (total_dividend / (face_value * dividend_rate)) * (face_value * (1 + premium_rate)) = 14400 :=
by sorry

end NUMINAMATH_CALUDE_investment_calculation_l927_92773


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l927_92748

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed given the rowing speeds in still water and upstream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that for a man with given upstream and still water speeds, 
    the downstream speed is 65 kmph -/
theorem downstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 60) 
  (h2 : s.upstream = 55) : 
  downstreamSpeed s = 65 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l927_92748


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l927_92751

theorem magnitude_of_complex_number (z : ℂ) : z = (4 - 2*I) / (1 + I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l927_92751


namespace NUMINAMATH_CALUDE_midpoint_translated_triangle_l927_92723

/-- Given triangle BIG with vertices B(0, 0), I(3, 3), and G(6, 0),
    translated 3 units left and 4 units up to form triangle B'I'G',
    the midpoint of segment B'G' is (0, 4). -/
theorem midpoint_translated_triangle (B I G B' I' G' : ℝ × ℝ) :
  B = (0, 0) →
  I = (3, 3) →
  G = (6, 0) →
  B' = (B.1 - 3, B.2 + 4) →
  I' = (I.1 - 3, I.2 + 4) →
  G' = (G.1 - 3, G.2 + 4) →
  ((B'.1 + G'.1) / 2, (B'.2 + G'.2) / 2) = (0, 4) := by
sorry

end NUMINAMATH_CALUDE_midpoint_translated_triangle_l927_92723


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_one_l927_92755

theorem sin_plus_cos_equals_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x + Real.cos x = 1 → x = 0 ∨ x = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_one_l927_92755


namespace NUMINAMATH_CALUDE_factorization_constant_l927_92718

theorem factorization_constant (c : ℝ) : 
  (∀ x, x^2 - 4*x + c = (x - 1) * (x - 3)) → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorization_constant_l927_92718


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l927_92784

/-- Given a complex number z = i(1-i), prove that it corresponds to a point in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I * (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l927_92784


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l927_92731

/-- The number of different rectangles in a 5x5 grid -/
def num_rectangles (n : ℕ) : ℕ := (n.choose 2) ^ 2

/-- Theorem: The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in a 5x5 square array of dots is 100. -/
theorem rectangles_in_5x5_grid :
  num_rectangles 5 = 100 := by
  sorry

#eval num_rectangles 5  -- Should output 100

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l927_92731


namespace NUMINAMATH_CALUDE_cosine_sum_inequality_l927_92715

theorem cosine_sum_inequality (n : ℕ) (x : ℝ) :
  (Finset.range (n + 1)).sum (fun i => |Real.cos (2^i * x)|) ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_inequality_l927_92715


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l927_92704

-- Define the sets M and N
def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l927_92704


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l927_92750

/-- The area of a rectangle with side lengths 2(x+7) and 2(x+5), 
    minus the area of a rectangle with side lengths 3(2x-3) and 3(x-2), 
    equals -14x^2 + 111x + 86 -/
theorem rectangle_area_difference (x : ℝ) : 
  (2 * (x + 7)) * (2 * (x + 5)) - (3 * (2 * x - 3)) * (3 * (x - 2)) = -14 * x^2 + 111 * x + 86 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l927_92750


namespace NUMINAMATH_CALUDE_franklin_students_count_l927_92753

/-- The number of Valentines Mrs. Franklin already has -/
def valentines_owned : ℝ := 58.0

/-- The number of additional Valentines Mrs. Franklin needs -/
def valentines_needed : ℝ := 16.0

/-- The number of students Mrs. Franklin has -/
def number_of_students : ℝ := valentines_owned + valentines_needed

theorem franklin_students_count : number_of_students = 74.0 := by
  sorry

end NUMINAMATH_CALUDE_franklin_students_count_l927_92753


namespace NUMINAMATH_CALUDE_triangle_inequality_l927_92707

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l927_92707


namespace NUMINAMATH_CALUDE_euler_totient_equation_solution_l927_92711

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_equation_solution :
  ∀ n : ℕ, n > 0 → (n = euler_totient n + 402 ↔ n = 802 ∨ n = 546) := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_equation_solution_l927_92711


namespace NUMINAMATH_CALUDE_min_value_theorem_l927_92769

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  x + 2 / (x + 3) ≥ 2 * Real.sqrt 2 - 3 ∧
  (x + 2 / (x + 3) = 2 * Real.sqrt 2 - 3 ↔ x = -3 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l927_92769


namespace NUMINAMATH_CALUDE_equation_impossible_l927_92734

-- Define the set of digits from 1 to 9
def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the property that all variables are distinct
def AllDistinct (K T U Ch O H : Nat) : Prop :=
  K ≠ T ∧ K ≠ U ∧ K ≠ Ch ∧ K ≠ O ∧ K ≠ H ∧
  T ≠ U ∧ T ≠ Ch ∧ T ≠ O ∧ T ≠ H ∧
  U ≠ Ch ∧ U ≠ O ∧ U ≠ H ∧
  Ch ≠ O ∧ Ch ≠ H ∧
  O ≠ H

theorem equation_impossible :
  ∀ (K T U Ch O H : Nat),
    K ∈ Digits → T ∈ Digits → U ∈ Digits → Ch ∈ Digits → O ∈ Digits → H ∈ Digits →
    AllDistinct K T U Ch O H →
    K * 0 * T ≠ U * Ch * O * H * H * U :=
by sorry

end NUMINAMATH_CALUDE_equation_impossible_l927_92734


namespace NUMINAMATH_CALUDE_rachels_age_l927_92783

/-- Given that Rachel is 4 years older than Leah and the sum of their ages is 34,
    prove that Rachel is 19 years old. -/
theorem rachels_age (rachel_age leah_age : ℕ) 
    (h1 : rachel_age = leah_age + 4)
    (h2 : rachel_age + leah_age = 34) : 
  rachel_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_rachels_age_l927_92783


namespace NUMINAMATH_CALUDE_lcm_is_perfect_square_l927_92790

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) : 
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
sorry

end NUMINAMATH_CALUDE_lcm_is_perfect_square_l927_92790
