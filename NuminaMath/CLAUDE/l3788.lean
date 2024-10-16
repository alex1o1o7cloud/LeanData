import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_and_coefficients_l3788_378878

theorem sum_of_roots_and_coefficients (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  c^2 + a*c + b = 0 →
  d^2 + a*d + b = 0 →
  a^2 + c*a + d = 0 →
  b^2 + c*b + d = 0 →
  a + b + c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_and_coefficients_l3788_378878


namespace NUMINAMATH_CALUDE_G_equals_4F_l3788_378867

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (4*x + x^4)/(1 + 4*x^3)) / (1 - (4*x + x^4)/(1 + 4*x^3)))

theorem G_equals_4F (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1 ∧ 1 + 4*x^3 ≠ 0) : G x = 4 * F x := by
  sorry

end NUMINAMATH_CALUDE_G_equals_4F_l3788_378867


namespace NUMINAMATH_CALUDE_people_joined_line_l3788_378831

theorem people_joined_line (initial : ℕ) (left : ℕ) (current : ℕ) : 
  initial ≥ left → 
  current = (initial - left) + (current - (initial - left)) :=
by sorry

end NUMINAMATH_CALUDE_people_joined_line_l3788_378831


namespace NUMINAMATH_CALUDE_polynomial_identity_l3788_378853

theorem polynomial_identity (P : ℝ → ℝ) 
  (h1 : ∀ x, P (x^3) = (P x)^3) 
  (h2 : P 2 = 2) :
  ∀ x, P x = x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3788_378853


namespace NUMINAMATH_CALUDE_truck_sales_l3788_378802

theorem truck_sales (total_vehicles : ℕ) (car_truck_difference : ℕ) : 
  total_vehicles = 69 → car_truck_difference = 27 → 
  ∃ trucks : ℕ, trucks = 21 ∧ trucks + (trucks + car_truck_difference) = total_vehicles :=
by
  sorry

end NUMINAMATH_CALUDE_truck_sales_l3788_378802


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_constant_difference_l3788_378813

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_constant_difference 
  (f g : ℝ → ℝ) 
  (r : ℝ) 
  (hf_monic : ∃ a, ∀ x, f x = (x - (r + 1)) * (x - (r + 8)) * (x - a))
  (hg_monic : ∃ b, ∀ x, g x = (x - (r + 2)) * (x - (r + 9)) * (x - b))
  (h_diff : ∀ x, f x - g x = r) :
  r = -264/7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_constant_difference_l3788_378813


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3788_378891

theorem inequality_system_solution :
  (∀ x : ℝ, 2 - x ≥ (x - 1) / 3 - 1 ↔ x ≤ 2.5) ∧
  ¬∃ x : ℝ, (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3788_378891


namespace NUMINAMATH_CALUDE_general_term_formula_min_value_S_min_value_n_l3788_378844

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem for the general term
theorem general_term_formula : ∀ n : ℕ, a n = S n - S (n - 1) :=
sorry

-- Theorem for the minimum value of S_n
theorem min_value_S : ∃ n : ℕ, S n = -112 ∧ ∀ m : ℕ, S m ≥ -112 :=
sorry

-- Theorem for the values of n that give the minimum
theorem min_value_n : ∀ n : ℕ, S n = -112 ↔ (n = 7 ∨ n = 8) :=
sorry

end NUMINAMATH_CALUDE_general_term_formula_min_value_S_min_value_n_l3788_378844


namespace NUMINAMATH_CALUDE_complex_number_coordinate_l3788_378830

theorem complex_number_coordinate (i : ℂ) (h : i^2 = -1) :
  (i^2015) / (i - 2) = -1/5 + 2/5 * i := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinate_l3788_378830


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3788_378814

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ m = 2^n - 1 ∧ Prime m

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m ∧ m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3788_378814


namespace NUMINAMATH_CALUDE_z_power_2017_l3788_378864

theorem z_power_2017 (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : 
  z^2017 = Complex.I := by
sorry

end NUMINAMATH_CALUDE_z_power_2017_l3788_378864


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3788_378877

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ = k ∧ x₂^2 - x₂ = k) → k > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3788_378877


namespace NUMINAMATH_CALUDE_monomial_count_l3788_378847

-- Define a structure for algebraic expressions
structure AlgebraicExpr where
  expr : String

-- Define a function to check if an expression is a monomial
def is_monomial (expr : AlgebraicExpr) : Bool :=
  -- Implementation details omitted
  sorry

-- Define the list of algebraic expressions
def expr_list : List AlgebraicExpr := [
  ⟨"-1"⟩,
  ⟨"-2/3a^2"⟩,
  ⟨"1/6x^2y"⟩,
  ⟨"3a+b"⟩,
  ⟨"0"⟩,
  ⟨"(x-1)/2"⟩
]

-- Theorem statement
theorem monomial_count : 
  (expr_list.filter is_monomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l3788_378847


namespace NUMINAMATH_CALUDE_sin_equal_implies_isosceles_exists_isosceles_with_unequal_sines_l3788_378852

/-- A triangle ABC is isosceles if at least two of its sides are equal. -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = b ∨ b = c ∨ a = c

/-- The sine of an angle in a triangle. -/
noncomputable def sinAngle (A B C : ℝ × ℝ) (vertex : ℝ × ℝ) : ℝ :=
  sorry -- Definition of sine for an angle in a triangle

theorem sin_equal_implies_isosceles (A B C : ℝ × ℝ) :
  sinAngle A B C A = sinAngle A B C B → IsIsosceles A B C :=
sorry

theorem exists_isosceles_with_unequal_sines :
  ∃ (A B C : ℝ × ℝ), IsIsosceles A B C ∧ sinAngle A B C A ≠ sinAngle A B C B :=
sorry

end NUMINAMATH_CALUDE_sin_equal_implies_isosceles_exists_isosceles_with_unequal_sines_l3788_378852


namespace NUMINAMATH_CALUDE_largest_integer_m_l3788_378839

theorem largest_integer_m (x y m : ℝ) : 
  x + 2*y = 2*m + 1 →
  2*x + y = m + 2 →
  x - y > 2 →
  ∀ k : ℤ, k > m → k ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_m_l3788_378839


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_half_l3788_378801

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the octagon formed by joining the midpoints
    of a regular octagon is half the area of the original octagon -/
theorem midpoint_octagon_area_half (o : RegularOctagon) :
  area (midpointOctagon o) = (1/2) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_half_l3788_378801


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3788_378816

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  Real.sin B + Real.sin A * (Real.sin C - Real.cos C) = 0 →
  a = 2 →
  c = Real.sqrt 2 →
  a / Real.sin A = c / Real.sin C →
  C = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3788_378816


namespace NUMINAMATH_CALUDE_max_value_squared_sum_l3788_378871

/-- Given a point P(x,y) satisfying certain conditions, 
    the maximum value of x^2 + y^2 is 18. -/
theorem max_value_squared_sum (x y : ℝ) 
  (h1 : x ≥ 1) 
  (h2 : y ≥ x) 
  (h3 : x - 2*y + 3 ≥ 0) : 
  ∃ (max : ℝ), max = 18 ∧ x^2 + y^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_sum_l3788_378871


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3788_378800

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3788_378800


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l3788_378823

/-- Two lines are parallel if their slopes are equal -/
def parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The first line: x + (1+m)y + (m-2) = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + (1+m)*y + (m-2) = 0

/-- The second line: mx + 2y + 8 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := m*x + 2*y + 8 = 0

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel 1 (1+m) m 2 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l3788_378823


namespace NUMINAMATH_CALUDE_juice_drink_cost_l3788_378881

theorem juice_drink_cost (initial_amount : ℕ) (pizza_cost : ℕ) (pizza_quantity : ℕ) 
  (juice_quantity : ℕ) (return_amount : ℕ) : 
  initial_amount = 50 → 
  pizza_cost = 12 → 
  pizza_quantity = 2 → 
  juice_quantity = 2 → 
  return_amount = 22 → 
  (initial_amount - return_amount - pizza_cost * pizza_quantity) / juice_quantity = 2 :=
by sorry

end NUMINAMATH_CALUDE_juice_drink_cost_l3788_378881


namespace NUMINAMATH_CALUDE_regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l3788_378819

/-- The measure of an exterior angle of a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle of a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle_is_45 : regular_octagon_exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l3788_378819


namespace NUMINAMATH_CALUDE_polynomial_equality_l3788_378827

theorem polynomial_equality (x : ℝ) :
  (∃ t c : ℝ, (6*x^2 - 8*x + 9)*(3*x^2 + t*x + 8) = 18*x^4 - 54*x^3 + c*x^2 - 56*x + 72) ↔
  (∃ t c : ℝ, t = -5 ∧ c = 115) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3788_378827


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_eleven_l3788_378893

theorem sum_abcd_equals_negative_eleven (a b c d : ℚ) 
  (h : 2*a + 3 = 2*b + 4 ∧ 2*a + 3 = 2*c + 5 ∧ 2*a + 3 = 2*d + 6 ∧ 2*a + 3 = a + b + c + d + 10) : 
  a + b + c + d = -11 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_eleven_l3788_378893


namespace NUMINAMATH_CALUDE_speed_equivalence_l3788_378855

/-- Conversion factor from m/s to km/h -/
def meters_per_second_to_kmph : ℝ := 3.6

/-- The speed in meters per second -/
def speed_mps : ℝ := 30.0024

/-- The speed in kilometers per hour -/
def speed_kmph : ℝ := 108.00864

/-- Theorem stating that the given speed in km/h is equivalent to the given speed in m/s -/
theorem speed_equivalence : speed_kmph = speed_mps * meters_per_second_to_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l3788_378855


namespace NUMINAMATH_CALUDE_even_function_inequality_l3788_378874

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on a set if f(x) ≤ f(y) whenever x ≤ y in that set -/
def MonoIncOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

/-- The theorem statement -/
theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
    (h_even : IsEven f)
    (h_mono : MonoIncOn f (Set.Ici 0))
    (h_ineq : ∀ x ∈ Set.Icc (1/2) 1, f (a*x + 1) - f (x - 2) ≤ 0) :
  a ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3788_378874


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3788_378868

theorem irreducible_fraction (n : ℤ) : Int.gcd (39*n + 4) (26*n + 3) = 1 := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3788_378868


namespace NUMINAMATH_CALUDE_min_value_theorem_l3788_378815

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 2*a + b = 2) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 2 → 1/a + 2/b ≤ 1/x + 2/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 2 ∧ 1/x + 2/y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3788_378815


namespace NUMINAMATH_CALUDE_cd_length_possibilities_l3788_378876

/-- Represents a tetrahedron ABCD inscribed in a cylinder --/
structure InscribedTetrahedron where
  /-- Length of edge AB --/
  ab : ℝ
  /-- Length of edges AC and CB --/
  ac_cb : ℝ
  /-- Length of edges AD and DB --/
  ad_db : ℝ
  /-- Assertion that the tetrahedron is inscribed in a cylinder with minimal radius --/
  inscribed_minimal : Bool
  /-- Assertion that all vertices lie on the lateral surface of the cylinder --/
  vertices_on_surface : Bool
  /-- Assertion that CD is parallel to the cylinder's axis --/
  cd_parallel_axis : Bool

/-- Theorem stating the possible lengths of CD in the inscribed tetrahedron --/
theorem cd_length_possibilities (t : InscribedTetrahedron) 
  (h1 : t.ab = 2)
  (h2 : t.ac_cb = 6)
  (h3 : t.ad_db = 7)
  (h4 : t.inscribed_minimal)
  (h5 : t.vertices_on_surface)
  (h6 : t.cd_parallel_axis) :
  ∃ (cd : ℝ), (cd = Real.sqrt 47 + Real.sqrt 34) ∨ (cd = |Real.sqrt 47 - Real.sqrt 34|) :=
sorry

end NUMINAMATH_CALUDE_cd_length_possibilities_l3788_378876


namespace NUMINAMATH_CALUDE_unique_natural_number_satisfying_conditions_l3788_378859

theorem unique_natural_number_satisfying_conditions :
  ∃! (x : ℕ), 
    (∃ (k : ℕ), 3 * x + 1 = k^2) ∧ 
    (∃ (t : ℕ), 6 * x - 2 = t^2) ∧ 
    Nat.Prime (6 * x^2 - 1) ∧
    x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_natural_number_satisfying_conditions_l3788_378859


namespace NUMINAMATH_CALUDE_total_gum_pieces_l3788_378863

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (h1 : packages = 9) (h2 : pieces_per_package = 15) :
  packages * pieces_per_package = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l3788_378863


namespace NUMINAMATH_CALUDE_analysis_method_sufficient_conditions_l3788_378886

/-- The analysis method in mathematical proofs -/
structure AnalysisMethod where
  /-- The method starts from the conclusion to be proved -/
  starts_from_conclusion : Bool
  /-- The method progressively searches for conditions -/
  progressive_search : Bool
  /-- The type of conditions the method searches for -/
  condition_type : Type

/-- Definition of sufficient conditions -/
def SufficientCondition : Type := Unit

/-- Theorem: The analysis method searches for sufficient conditions -/
theorem analysis_method_sufficient_conditions (am : AnalysisMethod) :
  am.starts_from_conclusion ∧ am.progressive_search →
  am.condition_type = SufficientCondition := by
  sorry

end NUMINAMATH_CALUDE_analysis_method_sufficient_conditions_l3788_378886


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3788_378836

/-- Given a geometric sequence {a_n} where 2a₁, (3/2)a₂, a₃ form an arithmetic sequence,
    prove that the common ratio of the geometric sequence is either 1 or 2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- {a_n} is a geometric sequence
  (2 * a 1 - (3/2 * a 2) = (3/2 * a 2) - a 3) →  -- 2a₁, (3/2)a₂, a₃ form an arithmetic sequence
  (a 2 / a 1 = 1 ∨ a 2 / a 1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3788_378836


namespace NUMINAMATH_CALUDE_max_handshakes_l3788_378804

theorem max_handshakes (n : ℕ) (h : n = 60) : n * (n - 1) / 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_l3788_378804


namespace NUMINAMATH_CALUDE_product_pricing_equation_l3788_378842

/-- 
Given a product with:
- Marked price of 1375 yuan
- Sold at 80% of the marked price
- Making a profit of 100 yuan
Prove that the equation relating the cost price x to these values is:
1375 * 80% = x + 100
-/
theorem product_pricing_equation (x : ℝ) : 
  1375 * (80 / 100) = x + 100 := by sorry

end NUMINAMATH_CALUDE_product_pricing_equation_l3788_378842


namespace NUMINAMATH_CALUDE_odd_sum_squared_plus_product_not_both_even_l3788_378808

theorem odd_sum_squared_plus_product_not_both_even (p q : ℤ) 
  (h : Odd (p^2 + q^2 + p*q)) : ¬(Even p ∧ Even q) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_squared_plus_product_not_both_even_l3788_378808


namespace NUMINAMATH_CALUDE_no_solutions_to_absolute_value_equation_l3788_378834

theorem no_solutions_to_absolute_value_equation :
  ¬∃ y : ℝ, |y - 2| = |y - 1| + |y - 4| := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_absolute_value_equation_l3788_378834


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3788_378894

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_sum : ∀ n : ℕ, S n = (a 0) * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))) 
  (h_a2 : a 2 = 1/4) 
  (h_S3 : S 3 = 7/8) :
  (a 1 / a 0 = 2) ∨ (a 1 / a 0 = 1/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3788_378894


namespace NUMINAMATH_CALUDE_projection_magnitude_l3788_378824

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (1, 2)

theorem projection_magnitude :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  (dot_product / magnitude_a) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_projection_magnitude_l3788_378824


namespace NUMINAMATH_CALUDE_semicircle_radius_in_trapezoid_l3788_378803

/-- A trapezoid with specific measurements and an inscribed semicircle. -/
structure TrapezoidWithSemicircle where
  -- Define the trapezoid
  AB : ℝ
  CD : ℝ
  side1 : ℝ
  side2 : ℝ
  -- Conditions
  AB_eq : AB = 27
  CD_eq : CD = 45
  side1_eq : side1 = 13
  side2_eq : side2 = 15
  -- Semicircle properties
  semicircle_diameter : ℝ
  semicircle_diameter_eq : semicircle_diameter = AB
  tangential_to_CD : Bool -- represents that the semicircle is tangential to CD

/-- The radius of the semicircle in the trapezoid is 13.5. -/
theorem semicircle_radius_in_trapezoid (t : TrapezoidWithSemicircle) :
  t.semicircle_diameter / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_trapezoid_l3788_378803


namespace NUMINAMATH_CALUDE_bouncy_balls_per_package_l3788_378897

theorem bouncy_balls_per_package (total_packages : Nat) (total_balls : Nat) :
  total_packages = 16 →
  total_balls = 160 →
  ∃ (balls_per_package : Nat), balls_per_package * total_packages = total_balls ∧ balls_per_package = 10 := by
  sorry

end NUMINAMATH_CALUDE_bouncy_balls_per_package_l3788_378897


namespace NUMINAMATH_CALUDE_turban_price_l3788_378857

theorem turban_price (annual_salary : ℝ) (turban_price : ℝ) (work_fraction : ℝ) (partial_payment : ℝ) :
  annual_salary = 90 ∧ 
  work_fraction = 3/4 ∧ 
  work_fraction * (annual_salary + turban_price) = partial_payment + turban_price ∧
  partial_payment = 45 →
  turban_price = 90 := by sorry

end NUMINAMATH_CALUDE_turban_price_l3788_378857


namespace NUMINAMATH_CALUDE_time_to_top_floor_l3788_378837

/-- The number of floors in the building -/
def num_floors : ℕ := 10

/-- The time in seconds to go up to an even-numbered floor -/
def even_floor_time : ℕ := 15

/-- The time in seconds to go up to an odd-numbered floor -/
def odd_floor_time : ℕ := 9

/-- The number of even-numbered floors -/
def num_even_floors : ℕ := num_floors / 2

/-- The number of odd-numbered floors -/
def num_odd_floors : ℕ := (num_floors + 1) / 2

/-- The total time in seconds to reach the top floor -/
def total_time_seconds : ℕ := num_even_floors * even_floor_time + num_odd_floors * odd_floor_time

/-- Conversion factor from seconds to minutes -/
def seconds_per_minute : ℕ := 60

theorem time_to_top_floor :
  total_time_seconds / seconds_per_minute = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_to_top_floor_l3788_378837


namespace NUMINAMATH_CALUDE_linear_functions_through_point_l3788_378850

theorem linear_functions_through_point :
  ∃ (x₀ y₀ : ℝ) (k b : Fin 10 → ℕ),
    (∀ i : Fin 10, 1 ≤ k i ∧ k i ≤ 20 ∧ 1 ≤ b i ∧ b i ≤ 20) ∧
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j ∧ b i ≠ b j) ∧
    (∀ i : Fin 10, y₀ = k i * x₀ + b i) := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_through_point_l3788_378850


namespace NUMINAMATH_CALUDE_boat_breadth_l3788_378870

/-- Given a boat with the following properties:
  - length of 7 meters
  - sinks by 1 cm when a man gets on it
  - the man's mass is 210 kg
  - the density of water is 1000 kg/m³
  - the acceleration due to gravity is 9.81 m/s²
  Prove that the breadth of the boat is 3 meters. -/
theorem boat_breadth (length : ℝ) (sink_depth : ℝ) (man_mass : ℝ) (water_density : ℝ) (gravity : ℝ) :
  length = 7 →
  sink_depth = 0.01 →
  man_mass = 210 →
  water_density = 1000 →
  gravity = 9.81 →
  ∃ (breadth : ℝ), breadth = 3 ∧ man_mass = (length * breadth * sink_depth) * water_density :=
by sorry

end NUMINAMATH_CALUDE_boat_breadth_l3788_378870


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3788_378895

/-- The sum of the infinite series ∑(n=1 to ∞) (2n + 1) / (n(n + 1)(n + 2)) is equal to 1 -/
theorem infinite_series_sum : 
  (∑' n : ℕ+, (2 * n.val + 1 : ℝ) / (n.val * (n.val + 1) * (n.val + 2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3788_378895


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3788_378810

theorem quadratic_form_ratio (j : ℝ) :
  ∃ (c p q : ℝ), 
    (∀ j, 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ 
    q / p = -151 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3788_378810


namespace NUMINAMATH_CALUDE_principal_amount_l3788_378880

/-- Given a principal amount P lent at simple interest rate r,
    prove that P = 710 given the conditions from the problem. -/
theorem principal_amount (P r : ℝ) : 
  (P + P * r * 3 = 920) →
  (P + P * r * 9 = 1340) →
  P = 710 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l3788_378880


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l3788_378896

-- Define the piecewise function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≠ 3 then 2 / |x - 3| else a

-- Define the function y
noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

-- Theorem statement
theorem three_zeros_implies_a_equals_four (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    y x₁ a = 0 ∧ y x₂ a = 0 ∧ y x₃ a = 0) →
  (∀ x : ℝ, y x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  a = 4 :=
by sorry


end NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l3788_378896


namespace NUMINAMATH_CALUDE_recycling_point_calculation_l3788_378805

/-- The number of pounds needed to recycle to earn one point -/
def pounds_per_point (zoe_pounds : ℕ) (friends_pounds : ℕ) (total_points : ℕ) : ℚ :=
  (zoe_pounds + friends_pounds : ℚ) / total_points

theorem recycling_point_calculation :
  pounds_per_point 25 23 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_recycling_point_calculation_l3788_378805


namespace NUMINAMATH_CALUDE_triangle_inequality_constant_l3788_378875

theorem triangle_inequality_constant (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2) / c^2 > 1/2 ∧ ∀ N : ℝ, (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' > c' → b' + c' > a' → c' + a' > b' → (a'^2 + b'^2) / c'^2 > N) → N ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_constant_l3788_378875


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l3788_378835

theorem sqrt_fourth_power_eq_256 (y : ℝ) : (Real.sqrt y) ^ 4 = 256 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l3788_378835


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3788_378806

/-- The coefficient of x^2 in the expansion of (3x^3 + 5x^2 - 4x + 1)(2x^2 - 9x + 3) -/
def coefficient_x_squared : ℤ := 51

/-- The first polynomial in the product -/
def poly1 (x : ℚ) : ℚ := 3 * x^3 + 5 * x^2 - 4 * x + 1

/-- The second polynomial in the product -/
def poly2 (x : ℚ) : ℚ := 2 * x^2 - 9 * x + 3

/-- Theorem stating that the coefficient of x^2 in the expansion of (poly1 * poly2) is equal to coefficient_x_squared -/
theorem coefficient_x_squared_expansion :
  ∃ (a b c d e : ℚ), (poly1 * poly2) = (λ x => a * x^4 + b * x^3 + coefficient_x_squared * x^2 + d * x + e) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3788_378806


namespace NUMINAMATH_CALUDE_haley_candy_count_l3788_378888

/-- The number of candy pieces Haley has at the end -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem stating that Haley's final candy count is 35 -/
theorem haley_candy_count :
  final_candy_count 33 17 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_haley_candy_count_l3788_378888


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3788_378833

def M : Set ℝ := {-1, 1, 2}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3788_378833


namespace NUMINAMATH_CALUDE_chocolates_remaining_day5_l3788_378809

/-- Calculates the number of chocolates remaining after 4 days of consumption -/
def chocolates_remaining (initial : ℕ) (day1 : ℕ) : ℕ :=
  let day2 := 2 * day1 - 3
  let day3 := day1 - 2
  let day4 := day3 - 1
  initial - (day1 + day2 + day3 + day4)

/-- Theorem stating that given the initial conditions, 12 chocolates remain on Day 5 -/
theorem chocolates_remaining_day5 :
  chocolates_remaining 24 4 = 12 := by
  sorry

#eval chocolates_remaining 24 4

end NUMINAMATH_CALUDE_chocolates_remaining_day5_l3788_378809


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3788_378889

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ (∀ y : ℝ, y^2 + 2*y + k = 0 → y = x)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3788_378889


namespace NUMINAMATH_CALUDE_pen_retailer_profit_percentage_specific_pen_retailer_profit_l3788_378873

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_retailer_profit_percentage 
  (num_pens : ℕ) 
  (price_per_36_pens : ℝ) 
  (discount_percent : ℝ) : ℝ :=
let cost_per_pen := price_per_36_pens / 36
let total_cost := num_pens * cost_per_pen
let selling_price_per_pen := price_per_36_pens / 36 * (1 - discount_percent / 100)
let total_selling_price := num_pens * selling_price_per_pen
let profit := total_selling_price - total_cost
let profit_percentage := (profit / total_cost) * 100
profit_percentage

/-- The profit percentage for a retailer buying 120 pens at the price of 36 pens 
    and selling with a 1% discount is 230% -/
theorem specific_pen_retailer_profit :
  pen_retailer_profit_percentage 120 36 1 = 230 := by
  sorry

end NUMINAMATH_CALUDE_pen_retailer_profit_percentage_specific_pen_retailer_profit_l3788_378873


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3788_378854

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem quadratic_function_properties :
  (f 0 = 0) ∧
  (∀ x, f (x + 2) = f (x + 1) + 2*x + 1) ∧
  (∀ m, (∀ x ∈ Set.Icc (-1) m, f x ∈ Set.Icc (-1) 3) → m ∈ Set.Icc 1 3) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3788_378854


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3788_378840

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 10) / (factorial 5)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3788_378840


namespace NUMINAMATH_CALUDE_van_transport_l3788_378826

theorem van_transport (students_per_van : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : students_per_van = 28)
  (h2 : num_boys = 60)
  (h3 : num_girls = 80) :
  (num_boys + num_girls) / students_per_van = 5 := by
  sorry

#check van_transport

end NUMINAMATH_CALUDE_van_transport_l3788_378826


namespace NUMINAMATH_CALUDE_remaining_investment_rate_l3788_378882

-- Define the investment amounts and rates
def total_investment : ℝ := 12000
def investment1_amount : ℝ := 5000
def investment1_rate : ℝ := 0.03
def investment2_amount : ℝ := 4000
def investment2_rate : ℝ := 0.045
def desired_income : ℝ := 600

-- Define the remaining investment amount
def remaining_investment : ℝ := total_investment - (investment1_amount + investment2_amount)

-- Define the income from the first two investments
def known_income : ℝ := investment1_amount * investment1_rate + investment2_amount * investment2_rate

-- Define the required income from the remaining investment
def required_income : ℝ := desired_income - known_income

-- Theorem to prove
theorem remaining_investment_rate : 
  (required_income / remaining_investment) = 0.09 := by sorry

end NUMINAMATH_CALUDE_remaining_investment_rate_l3788_378882


namespace NUMINAMATH_CALUDE_enrollment_increase_l3788_378869

/-- Theorem: Enrollment Increase Calculation

Given:
- Enrollment at the beginning of 1992 was 20% greater than at the beginning of 1991
- Enrollment at the beginning of 1993 was 26% greater than at the beginning of 1991

Prove:
The percent increase in enrollment from the beginning of 1992 to the beginning of 1993 is 5%
-/
theorem enrollment_increase (e : ℝ) : 
  let e_1992 := 1.20 * e
  let e_1993 := 1.26 * e
  (e_1993 - e_1992) / e_1992 * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l3788_378869


namespace NUMINAMATH_CALUDE_existence_of_positive_reals_l3788_378828

theorem existence_of_positive_reals : ∃ (x y z : ℝ), 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^4 + y^4 + z^4 = 13 ∧
  x^3*y^3*z + y^3*z^3*x + z^3*x^3*y = 6*Real.sqrt 3 ∧
  x^3*y*z + y^3*z*x + z^3*x*y = 5*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_positive_reals_l3788_378828


namespace NUMINAMATH_CALUDE_complement_of_union_relative_to_U_l3788_378879

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union_relative_to_U :
  (U \ (A ∪ B)) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_relative_to_U_l3788_378879


namespace NUMINAMATH_CALUDE_final_price_is_66_percent_l3788_378861

/-- The percentage of the suggested retail price paid after discounts and tax -/
def final_price_percentage (initial_discount : ℝ) (clearance_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  (1 - initial_discount) * (1 - clearance_discount) * (1 + sales_tax)

/-- Theorem stating that the final price is 66% of the suggested retail price -/
theorem final_price_is_66_percent :
  final_price_percentage 0.2 0.25 0.1 = 0.66 := by
  sorry

#eval final_price_percentage 0.2 0.25 0.1

end NUMINAMATH_CALUDE_final_price_is_66_percent_l3788_378861


namespace NUMINAMATH_CALUDE_quadrilateral_prism_edges_and_vertices_l3788_378899

/-- A prism with a quadrilateral base -/
structure QuadrilateralPrism :=
  (lateral_faces : ℕ)
  (lateral_faces_eq : lateral_faces = 4)

/-- The number of edges in a quadrilateral prism -/
def num_edges (p : QuadrilateralPrism) : ℕ := 12

/-- The number of vertices in a quadrilateral prism -/
def num_vertices (p : QuadrilateralPrism) : ℕ := 8

/-- Theorem stating that a quadrilateral prism has 12 edges and 8 vertices -/
theorem quadrilateral_prism_edges_and_vertices (p : QuadrilateralPrism) :
  num_edges p = 12 ∧ num_vertices p = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_prism_edges_and_vertices_l3788_378899


namespace NUMINAMATH_CALUDE_inequality_and_equality_l3788_378856

theorem inequality_and_equality (x : ℝ) (h : x > 0) : 
  (x + 1/x ≥ 2) ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_l3788_378856


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3788_378822

-- Define the line equation
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 25

-- Theorem stating that any real slope m results in an intersection
theorem line_intersects_circle (m : ℝ) :
  ∃ x : ℝ, circle_equation x (line_equation m x) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3788_378822


namespace NUMINAMATH_CALUDE_array_sum_property_l3788_378846

/-- Represents the sum of an infinite array with specific properties -/
def array_sum (p : ℕ) : ℚ :=
  (2 * p^2) / ((2 * p - 1) * (p - 1))

/-- Theorem stating the property of the sum for p = 1004 -/
theorem array_sum_property :
  let p := 1004
  let S := array_sum p
  let a := 2 * p^2
  let b := (2 * p - 1) * (p - 1)
  (a + b) % p = 1 := by sorry

end NUMINAMATH_CALUDE_array_sum_property_l3788_378846


namespace NUMINAMATH_CALUDE_original_number_before_increase_l3788_378890

theorem original_number_before_increase (final_number : ℝ) (increase_percentage : ℝ) (original_number : ℝ) : 
  final_number = 90 ∧ 
  increase_percentage = 50 ∧ 
  final_number = original_number * (1 + increase_percentage / 100) → 
  original_number = 60 := by
sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l3788_378890


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l3788_378858

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l3788_378858


namespace NUMINAMATH_CALUDE_cylinder_volume_l3788_378872

/-- Given a cylinder with base radius 3 and lateral surface area 12π, its volume is 18π. -/
theorem cylinder_volume (r h : ℝ) : r = 3 ∧ 2 * π * r * h = 12 * π → π * r^2 * h = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3788_378872


namespace NUMINAMATH_CALUDE_camp_food_consumption_l3788_378898

/-- Represents the amount of food eaten by dogs and puppies in a day -/
def total_food_eaten (num_puppies num_dogs : ℕ) 
                     (dog_meal_frequency puppy_meal_frequency : ℕ) 
                     (dog_meal_amount : ℚ) 
                     (dog_puppy_food_ratio : ℚ) : ℚ :=
  let dog_daily_food := dog_meal_amount * dog_meal_frequency
  let puppy_meal_amount := dog_meal_amount / dog_puppy_food_ratio
  let puppy_daily_food := puppy_meal_amount * puppy_meal_frequency
  (num_dogs : ℚ) * dog_daily_food + (num_puppies : ℚ) * puppy_daily_food

/-- Theorem stating the total food eaten by dogs and puppies in a day -/
theorem camp_food_consumption : 
  total_food_eaten 6 5 2 8 6 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_camp_food_consumption_l3788_378898


namespace NUMINAMATH_CALUDE_imaginary_part_implies_a_value_l3788_378812

theorem imaginary_part_implies_a_value (a : ℝ) :
  (Complex.im ((1 - a * Complex.I) / (1 + Complex.I)) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_implies_a_value_l3788_378812


namespace NUMINAMATH_CALUDE_yellow_two_days_ago_count_l3788_378841

/-- Represents the count of dandelions for a specific day -/
structure DandelionCount where
  yellow : ℕ
  white : ℕ

/-- Represents the dandelion lifecycle and counts for three consecutive days -/
structure DandelionMeadow where
  twoDaysAgo : DandelionCount
  yesterday : DandelionCount
  today : DandelionCount

/-- Theorem stating the relationship between yellow dandelions two days ago and white dandelions on subsequent days -/
theorem yellow_two_days_ago_count (meadow : DandelionMeadow) 
  (h1 : meadow.yesterday.yellow = 20)
  (h2 : meadow.yesterday.white = 14)
  (h3 : meadow.today.yellow = 15)
  (h4 : meadow.today.white = 11) :
  meadow.twoDaysAgo.yellow = meadow.yesterday.white + meadow.today.white :=
sorry

end NUMINAMATH_CALUDE_yellow_two_days_ago_count_l3788_378841


namespace NUMINAMATH_CALUDE_lee_lawn_mowing_earnings_l3788_378832

/-- Lee's lawn mowing earnings problem -/
theorem lee_lawn_mowing_earnings :
  ∀ (charge_per_lawn : ℕ) (lawns_mowed : ℕ) (tip_amount : ℕ) (num_tippers : ℕ),
    charge_per_lawn = 33 →
    lawns_mowed = 16 →
    tip_amount = 10 →
    num_tippers = 3 →
    charge_per_lawn * lawns_mowed + tip_amount * num_tippers = 558 :=
by
  sorry


end NUMINAMATH_CALUDE_lee_lawn_mowing_earnings_l3788_378832


namespace NUMINAMATH_CALUDE_second_number_proof_l3788_378860

theorem second_number_proof (a b c : ℚ) : 
  a + b + c = 98 ∧ 
  a / b = 2 / 3 ∧ 
  b / c = 5 / 8 → 
  b = 30 :=
by sorry

end NUMINAMATH_CALUDE_second_number_proof_l3788_378860


namespace NUMINAMATH_CALUDE_parallel_non_existent_slopes_intersect_one_non_existent_slope_line_equation_through_two_points_l3788_378887

-- Define a straight line in a coordinate plane
structure Line where
  slope : Option ℝ
  point : ℝ × ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

-- Theorem 1: If the slopes of two lines do not exist, then the two lines are parallel
theorem parallel_non_existent_slopes (l1 l2 : Line) :
  l1.slope = none ∧ l2.slope = none → parallel l1 l2 := by sorry

-- Theorem 2: If one of two lines has a non-existent slope and the other has a slope, 
-- then the two lines intersect
theorem intersect_one_non_existent_slope (l1 l2 : Line) :
  (l1.slope = none ∧ l2.slope ≠ none) ∨ (l1.slope ≠ none ∧ l2.slope = none) 
  → intersect l1 l2 := by sorry

-- Theorem 3: The equation of the line passing through any two different points 
-- P₁(x₁, y₁), P₂(x₂, y₂) is (x₂-x₁)(y-y₁)=(y₂-y₁)(x-x₁)
theorem line_equation_through_two_points (P1 P2 : ℝ × ℝ) (x y : ℝ) :
  P1 ≠ P2 → (P2.1 - P1.1) * (y - P1.2) = (P2.2 - P1.2) * (x - P1.1) := by sorry

end NUMINAMATH_CALUDE_parallel_non_existent_slopes_intersect_one_non_existent_slope_line_equation_through_two_points_l3788_378887


namespace NUMINAMATH_CALUDE_expression_simplification_l3788_378845

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x^2 + 1) / (x^2 - 1) - (x - 2) / (x - 1) / ((x - 2) / x) = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3788_378845


namespace NUMINAMATH_CALUDE_two_face_cards_probability_l3788_378862

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of face cards in a standard deck
def face_cards : ℕ := 12

-- Define the probability of selecting two face cards
def prob_two_face_cards : ℚ := 22 / 442

-- Theorem statement
theorem two_face_cards_probability :
  (face_cards / total_cards) * ((face_cards - 1) / (total_cards - 1)) = prob_two_face_cards := by
  sorry

end NUMINAMATH_CALUDE_two_face_cards_probability_l3788_378862


namespace NUMINAMATH_CALUDE_extended_square_counts_l3788_378811

/-- Represents a square configuration with extended sides -/
structure ExtendedSquare where
  /-- Side length of the small square -/
  a : ℝ
  /-- Area of the shaded triangle -/
  S : ℝ
  /-- Condition that S is a quarter of the area of the small square -/
  h_S : S = a^2 / 4

/-- Count of triangles with area 2S in the extended square configuration -/
def count_triangles_2S (sq : ExtendedSquare) : ℕ := 20

/-- Count of squares with area 8S in the extended square configuration -/
def count_squares_8S (sq : ExtendedSquare) : ℕ := 1

/-- Main theorem stating the counts of specific triangles and squares -/
theorem extended_square_counts (sq : ExtendedSquare) :
  count_triangles_2S sq = 20 ∧ count_squares_8S sq = 1 := by
  sorry

end NUMINAMATH_CALUDE_extended_square_counts_l3788_378811


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3788_378866

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- The number of members chosen from the host team -/
def host_members : ℕ := 3

/-- The number of teams that select 2 members -/
def teams_select_two : ℕ := 3

/-- The number of teams that select 3 members (excluding the host) -/
def teams_select_three : ℕ := 1

/-- The total number of ways to form a tournament committee -/
def total_committees : ℕ := 229105500

theorem tournament_committee_count :
  (num_teams) *
  (Nat.choose team_size host_members) *
  (Nat.choose (num_teams - 1) teams_select_three) *
  (Nat.choose team_size host_members) *
  (Nat.choose team_size 2 ^ teams_select_two) = total_committees := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3788_378866


namespace NUMINAMATH_CALUDE_lawnmower_value_drop_l3788_378829

/-- Calculates the final value of a lawnmower after three successive value drops -/
theorem lawnmower_value_drop (initial_value : ℝ) (drop1 drop2 drop3 : ℝ) :
  initial_value = 100 →
  drop1 = 0.25 →
  drop2 = 0.20 →
  drop3 = 0.15 →
  initial_value * (1 - drop1) * (1 - drop2) * (1 - drop3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_value_drop_l3788_378829


namespace NUMINAMATH_CALUDE_negative_three_squared_plus_negative_two_cubed_l3788_378848

theorem negative_three_squared_plus_negative_two_cubed : -3^2 + (-2)^3 = -17 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_plus_negative_two_cubed_l3788_378848


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3788_378851

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^(m^2 - 2) - 3*x + 1 = a*x^2 + b*x + c) → 
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3788_378851


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l3788_378820

theorem car_speed_time_relation (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ original_time = 12 ∧ new_speed = 60 →
  (distance / new_speed) / original_time = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_time_relation_l3788_378820


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_7_l3788_378883

theorem sum_of_sixth_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_7_l3788_378883


namespace NUMINAMATH_CALUDE_human_family_members_l3788_378849

/-- Represents the number of feet for each type of animal and the alien pet. -/
structure AnimalFeet where
  birds : ℕ
  dogs : ℕ
  cats : ℕ
  alien : ℕ

/-- Represents the number of heads for each type of animal and the alien pet. -/
structure AnimalHeads where
  birds : ℕ
  dogs : ℕ
  cats : ℕ
  alien : ℕ

/-- Calculates the total number of feet for all animals and the alien pet. -/
def totalAnimalFeet (af : AnimalFeet) : ℕ :=
  af.birds + af.dogs + af.cats + af.alien

/-- Calculates the total number of heads for all animals and the alien pet. -/
def totalAnimalHeads (ah : AnimalHeads) : ℕ :=
  ah.birds + ah.dogs + ah.cats + ah.alien

/-- Theorem stating the number of human family members. -/
theorem human_family_members :
  ∃ (h : ℕ),
    let af : AnimalFeet := ⟨7, 13, 74, 6⟩
    let ah : AnimalHeads := ⟨4, 3, 18, 1⟩
    totalAnimalFeet af + 2 * h = totalAnimalHeads ah + h + 108 ∧ h = 34 := by
  sorry

end NUMINAMATH_CALUDE_human_family_members_l3788_378849


namespace NUMINAMATH_CALUDE_rectangle_hexagon_pqr_sum_l3788_378807

/-- A hexagon formed by three rectangles intersecting three straight lines -/
structure RectangleHexagon where
  -- External angles at S, T, U
  s : ℝ
  t : ℝ
  u : ℝ
  -- External angles at P, Q, R
  p : ℝ
  q : ℝ
  r : ℝ
  -- Conditions
  angle_s : s = 55
  angle_t : t = 60
  angle_u : u = 65
  sum_external : p + q + r + s + t + u = 360

/-- The sum of external angles at P, Q, and R in the RectangleHexagon is 180° -/
theorem rectangle_hexagon_pqr_sum (h : RectangleHexagon) : h.p + h.q + h.r = 180 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_hexagon_pqr_sum_l3788_378807


namespace NUMINAMATH_CALUDE_complement_of_union_l3788_378825

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem complement_of_union : U \ (A ∪ B) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3788_378825


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l3788_378884

/-- Given an ellipse that passes through a point (a, b), prove the minimum value of m + n -/
theorem ellipse_min_sum (a b m n : ℝ) : 
  m > 0 → n > 0 → m > n → a ≠ 0 → b ≠ 0 → abs a ≠ abs b →
  (a^2 / m^2) + (b^2 / n^2) = 1 →
  ∀ m' n', m' > 0 → n' > 0 → m' > n' → (a^2 / m'^2) + (b^2 / n'^2) = 1 →
  m + n ≤ m' + n' →
  m + n = (a^(2/3) + b^(2/3))^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l3788_378884


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3788_378821

theorem right_triangle_sides (x Δ : ℝ) (hx : x > 0) (hΔ : Δ > 0) :
  (x + 2*Δ)^2 = x^2 + (x + Δ)^2 ↔ x = (Δ*(-1 + 2*Real.sqrt 7))/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3788_378821


namespace NUMINAMATH_CALUDE_supervisors_average_salary_l3788_378865

/-- Given the following conditions in a factory:
  1. The average monthly salary of laborers and supervisors combined is 1250.
  2. There are 6 supervisors.
  3. There are 42 laborers.
  4. The average monthly salary of the laborers is 950.
  Prove that the average monthly salary of the supervisors is 3350. -/
theorem supervisors_average_salary
  (total_average : ℚ)
  (num_supervisors : ℕ)
  (num_laborers : ℕ)
  (laborers_average : ℚ)
  (h1 : total_average = 1250)
  (h2 : num_supervisors = 6)
  (h3 : num_laborers = 42)
  (h4 : laborers_average = 950) :
  (total_average * (num_supervisors + num_laborers) - laborers_average * num_laborers) / num_supervisors = 3350 := by
  sorry


end NUMINAMATH_CALUDE_supervisors_average_salary_l3788_378865


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l3788_378843

theorem interval_length_implies_difference (c d : ℝ) :
  (∀ x : ℝ, c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) →
  (∀ x : ℝ, c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d ↔ (c + 2) / 3 ≤ x ∧ x ≤ (d + 2) / 3) →
  ((d + 2) / 3 - (c + 2) / 3 = 15) →
  d - c = 45 := by
  sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l3788_378843


namespace NUMINAMATH_CALUDE_pulled_pork_sandwiches_l3788_378818

def total_sauce : ℚ := 5
def burger_sauce : ℚ := 1/4
def sandwich_sauce : ℚ := 1/6
def num_burgers : ℕ := 8

theorem pulled_pork_sandwiches :
  ∃ (n : ℕ), n * sandwich_sauce + num_burgers * burger_sauce = total_sauce ∧ n = 18 :=
by sorry

end NUMINAMATH_CALUDE_pulled_pork_sandwiches_l3788_378818


namespace NUMINAMATH_CALUDE_negation_of_exactly_one_intersection_negation_of_if_3_or_4_then_equation_l3788_378838

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the proposition for exactly one intersection point
def exactly_one_intersection (a b c : ℝ) : Prop :=
  ∃! x, f a b c x = 0

-- Define the proposition for no or at least two intersection points
def no_or_at_least_two_intersections (a b c : ℝ) : Prop :=
  (∀ x, f a b c x ≠ 0) ∨ (∃ x y, x ≠ y ∧ f a b c x = 0 ∧ f a b c y = 0)

-- Theorem for the negation of the first proposition
theorem negation_of_exactly_one_intersection (a b c : ℝ) :
  ¬(exactly_one_intersection a b c) ↔ no_or_at_least_two_intersections a b c :=
sorry

-- Define the proposition for the second statement
def if_3_or_4_then_equation : Prop :=
  (3^2 - 7*3 + 12 = 0) ∧ (4^2 - 7*4 + 12 = 0)

-- Theorem for the negation of the second proposition
theorem negation_of_if_3_or_4_then_equation :
  ¬if_3_or_4_then_equation ↔ (3^2 - 7*3 + 12 ≠ 0) ∨ (4^2 - 7*4 + 12 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_exactly_one_intersection_negation_of_if_3_or_4_then_equation_l3788_378838


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l3788_378885

/-- Calculates the total distance walked given the number of blocks and the length of each block -/
def total_distance (blocks_east blocks_north block_length : ℚ) : ℚ :=
  (blocks_east + blocks_north) * block_length

theorem arthur_walk_distance :
  let blocks_east : ℚ := 8
  let blocks_north : ℚ := 15
  let block_length : ℚ := 1/4
  total_distance blocks_east blocks_north block_length = 5.75 := by sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l3788_378885


namespace NUMINAMATH_CALUDE_clock_hands_angle_l3788_378892

-- Define the initial speeds of the hands
def initial_hour_hand_speed : ℝ := 0.5
def initial_minute_hand_speed : ℝ := 6

-- Define the swapped speeds
def swapped_hour_hand_speed : ℝ := initial_minute_hand_speed
def swapped_minute_hand_speed : ℝ := initial_hour_hand_speed

-- Define the starting position (3 PM)
def starting_hour_position : ℝ := 90
def starting_minute_position : ℝ := 0

-- Define the target position (4 o'clock)
def target_hour_position : ℝ := 120

-- Theorem statement
theorem clock_hands_angle :
  let time_to_target := (target_hour_position - starting_hour_position) / swapped_hour_hand_speed
  let final_minute_position := starting_minute_position + swapped_minute_hand_speed * time_to_target
  let angle := target_hour_position - final_minute_position
  min angle (360 - angle) = 117.5 := by sorry

end NUMINAMATH_CALUDE_clock_hands_angle_l3788_378892


namespace NUMINAMATH_CALUDE_opposite_sign_power_l3788_378817

theorem opposite_sign_power (x y : ℝ) : 
  (|x + 3| + (y - 2)^2 = 0) → x^y = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_power_l3788_378817
