import Mathlib

namespace NUMINAMATH_CALUDE_sandbox_area_l1569_156986

/-- The area of a rectangle with length 312 cm and width 146 cm is 45552 square centimeters. -/
theorem sandbox_area :
  let length : ℕ := 312
  let width : ℕ := 146
  length * width = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_l1569_156986


namespace NUMINAMATH_CALUDE_base_conversion_proof_l1569_156923

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

theorem base_conversion_proof :
  let base_5_101 := to_base_10 [1, 0, 1] 5
  let base_7_1234 := to_base_10 [4, 3, 2, 1] 7
  let base_9_3456 := to_base_10 [6, 5, 4, 3] 9
  2468 / base_5_101 * base_7_1234 - base_9_3456 = 41708 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_proof_l1569_156923


namespace NUMINAMATH_CALUDE_least_common_multiple_addition_l1569_156935

theorem least_common_multiple_addition (a b c d : ℕ) (n m : ℕ) : 
  (∀ k : ℕ, k < m → ¬(a ∣ (n + k) ∧ b ∣ (n + k) ∧ c ∣ (n + k) ∧ d ∣ (n + k))) →
  (a ∣ (n + m) ∧ b ∣ (n + m) ∧ c ∣ (n + m) ∧ d ∣ (n + m)) →
  m = 7 ∧ n = 857 ∧ a = 24 ∧ b = 32 ∧ c = 36 ∧ d = 54 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_addition_l1569_156935


namespace NUMINAMATH_CALUDE_blood_pressure_analysis_l1569_156984

def systolic_pressure : List ℝ := [151, 148, 140, 139, 140, 136, 140]
def diastolic_pressure : List ℝ := [90, 92, 88, 88, 90, 80, 88]

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem blood_pressure_analysis :
  (median systolic_pressure = 140) ∧
  (mode diastolic_pressure = 88) ∧
  (mean systolic_pressure = 142) ∧
  (variance diastolic_pressure = 88 / 7) :=
by sorry

end NUMINAMATH_CALUDE_blood_pressure_analysis_l1569_156984


namespace NUMINAMATH_CALUDE_pam_has_1200_apples_l1569_156903

/-- The number of apples in each of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- The number of Gerald's bags equivalent to one of Pam's bags -/
def gerald_to_pam_ratio : ℕ := 3

/-- The number of bags Pam has -/
def pams_bag_count : ℕ := 10

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := pams_bag_count * (gerald_to_pam_ratio * geralds_bag_count)

theorem pam_has_1200_apples : pams_total_apples = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_1200_apples_l1569_156903


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l1569_156938

/-- Calculates the final number of dimes Melanie has -/
def final_dimes (initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

/-- Theorem: The final number of dimes is correct given the problem conditions -/
theorem melanie_dimes_count : final_dimes 8 7 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l1569_156938


namespace NUMINAMATH_CALUDE_inequality_proof_l1569_156962

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y) / (y + z) + (y^2 * z) / (z + x) + (z^2 * x) / (x + y) ≥ (1/2) * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1569_156962


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_22_l1569_156922

theorem smallest_k_for_64_power_gt_4_power_22 : 
  ∃ k : ℕ, (∀ m : ℕ, 64^m > 4^22 → k ≤ m) ∧ 64^k > 4^22 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_22_l1569_156922


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1569_156937

theorem complex_fraction_evaluation : 
  (2 + 2)^2 / 2^2 * (3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3 * (6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6 = 108 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1569_156937


namespace NUMINAMATH_CALUDE_missing_number_proof_l1569_156975

theorem missing_number_proof (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1569_156975


namespace NUMINAMATH_CALUDE_problem_solution_l1569_156989

theorem problem_solution (a b m n k : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : k^2 = 2) : 
  2011*a + 2012*b + m*n*a + k^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1569_156989


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1569_156919

/-- Given a hyperbola with equation x²/a² - y²/(4a-2) = 1 and eccentricity √3, prove that a = 1 -/
theorem hyperbola_eccentricity (a : ℝ) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / (4*a - 2) = 1) →
  (∃ b : ℝ, b^2 = 4*a - 2 ∧ b^2 / a^2 = 2) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1569_156919


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1569_156952

theorem complex_expression_simplification :
  (7 + 4 * Real.sqrt 3) * (2 - Real.sqrt 3)^2 + (2 + Real.sqrt 3) * (2 - Real.sqrt 3) - Real.sqrt 3 = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1569_156952


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l1569_156955

/-- Triangle with vertices A(0,1), B(-2,0), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ := (0, 1)
  B : ℝ × ℝ := (-2, 0)
  C : ℝ × ℝ := (2, 0)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Altitude from A to AC -/
def altitude (t : Triangle) : LineEquation :=
  { a := 2, b := -1, c := 1 }

/-- Median from A to BC -/
def median (t : Triangle) : LineEquation :=
  { a := 1, b := 0, c := 0 }

theorem triangle_altitude_and_median (t : Triangle) :
  (altitude t = { a := 2, b := -1, c := 1 }) ∧
  (median t = { a := 1, b := 0, c := 0 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l1569_156955


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1569_156966

theorem vector_equation_solution :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, -2]
  ∀ m n : ℝ, (m • a + n • b = ![9, -8]) → (m - n = -3) := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1569_156966


namespace NUMINAMATH_CALUDE_cone_section_area_l1569_156907

-- Define the cone structure
structure Cone where
  -- Axial section is an isosceles right triangle
  axial_section_isosceles_right : Bool
  -- Hypotenuse of axial section
  hypotenuse : ℝ
  -- Angle between section and base
  α : ℝ

-- Define the theorem
theorem cone_section_area (c : Cone) 
  (h1 : c.axial_section_isosceles_right = true) 
  (h2 : c.hypotenuse = 2) 
  (h3 : 0 < c.α ∧ c.α < π / 2) : 
  ∃ (area : ℝ), area = (Real.sqrt 2 / 2) * (1 / (Real.cos c.α)^2) :=
sorry

end NUMINAMATH_CALUDE_cone_section_area_l1569_156907


namespace NUMINAMATH_CALUDE_simplify_expression_l1569_156942

theorem simplify_expression : 
  (((81 : ℝ) ^ (1/4 : ℝ)) + (Real.sqrt (8 + 3/4)))^2 = (71 + 12 * Real.sqrt 35) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1569_156942


namespace NUMINAMATH_CALUDE_polyhedron_inequalities_l1569_156958

/-- A simply connected polyhedron -/
structure SimplyConnectedPolyhedron where
  B : ℕ  -- number of vertices
  P : ℕ  -- number of edges
  G : ℕ  -- number of faces
  euler : B - P + G = 2  -- Euler's formula
  edge_face : P ≥ 3 * G / 2  -- each face has at least 3 edges, each edge is shared by 2 faces
  edge_vertex : P ≥ 3 * B / 2  -- each vertex is connected to at least 3 edges

/-- Theorem stating the inequalities for a simply connected polyhedron -/
theorem polyhedron_inequalities (poly : SimplyConnectedPolyhedron) :
  (3 / 2 : ℝ) ≤ (poly.P : ℝ) / poly.B ∧ (poly.P : ℝ) / poly.B < 3 ∧
  (3 / 2 : ℝ) ≤ (poly.P : ℝ) / poly.G ∧ (poly.P : ℝ) / poly.G < 3 :=
by sorry

end NUMINAMATH_CALUDE_polyhedron_inequalities_l1569_156958


namespace NUMINAMATH_CALUDE_m_range_l1569_156927

def p (m : ℝ) : Prop := ∀ x > 0, m^2 + 2*m - 1 ≤ x + 1/x

def q (m : ℝ) : Prop := ∀ x₁ x₂, x₁ < x₂ → (5 - m^2)^x₁ < (5 - m^2)^x₂

theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) :
  (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1569_156927


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1569_156916

theorem subtraction_of_fractions : (5 : ℚ) / 6 - (1 : ℚ) / 3 = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1569_156916


namespace NUMINAMATH_CALUDE_complex_multiplication_l1569_156997

def i : ℂ := Complex.I

theorem complex_multiplication :
  (6 - 3 * i) * (-7 + 2 * i) = -36 + 33 * i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1569_156997


namespace NUMINAMATH_CALUDE_cube_edge_sum_l1569_156959

/-- Given a cube with surface area 486 square centimeters, 
    prove that the sum of the lengths of all its edges is 108 centimeters. -/
theorem cube_edge_sum (surface_area : ℝ) (h : surface_area = 486) : 
  ∃ (edge_length : ℝ), 
    surface_area = 6 * edge_length^2 ∧ 
    12 * edge_length = 108 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l1569_156959


namespace NUMINAMATH_CALUDE_mean_of_other_two_l1569_156944

def numbers : List ℤ := [2179, 2231, 2307, 2375, 2419, 2433]

def sum_of_all : ℤ := numbers.sum

def mean_of_four : ℤ := 2323

def sum_of_four : ℤ := 4 * mean_of_four

theorem mean_of_other_two (h : sum_of_four = 4 * mean_of_four) :
  (sum_of_all - sum_of_four) / 2 = 2321 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_other_two_l1569_156944


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1569_156911

theorem arithmetic_sequence_length 
  (a : ℤ) (an : ℤ) (d : ℤ) (n : ℕ) 
  (h1 : a = -50) 
  (h2 : an = 74) 
  (h3 : d = 6) 
  (h4 : an = a + (n - 1) * d) : n = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1569_156911


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1569_156961

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDiff : ℤ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1) * seq.commonDiff

theorem arithmetic_sequence_ninth_term
  (seq : ArithmeticSequence)
  (h3 : seq.nthTerm 3 = 5)
  (h6 : seq.nthTerm 6 = 17) :
  seq.nthTerm 9 = 29 := by
  sorry

#check arithmetic_sequence_ninth_term

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1569_156961


namespace NUMINAMATH_CALUDE_encryption_game_team_sizes_l1569_156915

theorem encryption_game_team_sizes :
  ∀ (num_two num_three num_four num_five : ℕ),
    -- Total number of players
    168 = 2 * num_two + 3 * num_three + 4 * num_four + 5 * num_five →
    -- Total number of teams
    50 = num_two + num_three + num_four + num_five →
    -- Number of three-player teams
    num_three = 20 →
    -- At least one five-player team
    num_five > 0 →
    -- Four is the most common team size
    num_four ≥ num_two ∧ num_four > num_three ∧ num_four > num_five →
    -- Conclusion
    num_two = 7 ∧ num_four = 21 ∧ num_five = 2 := by
  sorry

end NUMINAMATH_CALUDE_encryption_game_team_sizes_l1569_156915


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1569_156936

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

theorem greatest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    p ∣ (factorial 15 + factorial 18) ∧ 
    ∀ (q : ℕ), is_prime q → q ∣ (factorial 15 + factorial 18) → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1569_156936


namespace NUMINAMATH_CALUDE_intersection_points_l1569_156957

theorem intersection_points (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.2 = a * p.1 + a ∧ p.2 = p.1 ∧ p.2 = 2 - 2 * a * p.1)) ↔ 
  (a = 1/2 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l1569_156957


namespace NUMINAMATH_CALUDE_cos_sin_18_equality_l1569_156905

theorem cos_sin_18_equality :
  let cos_18 : ℝ := (Real.sqrt 5 + 1) / 4
  let sin_18 : ℝ := (Real.sqrt 5 - 1) / 4
  4 * cos_18^2 - 1 = 1 / (4 * sin_18^2) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_18_equality_l1569_156905


namespace NUMINAMATH_CALUDE_strawberry_area_l1569_156983

/-- Given a garden with the following properties:
  * The total area is 64 square feet
  * Half of the garden is for fruits
  * A quarter of the fruit section is for strawberries
  Prove that the area for strawberries is 8 square feet. -/
theorem strawberry_area (garden_area : ℝ) (fruit_ratio : ℝ) (strawberry_ratio : ℝ) : 
  garden_area = 64 → 
  fruit_ratio = 1/2 → 
  strawberry_ratio = 1/4 → 
  garden_area * fruit_ratio * strawberry_ratio = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_area_l1569_156983


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l1569_156947

-- Define the hyperbola equation
def hyperbola_equation (x y m n : ℝ) : Prop :=
  x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the distance between foci
def foci_distance : ℝ := 4

-- Theorem statement
theorem hyperbola_n_range (x y m n : ℝ) :
  hyperbola_equation x y m n ∧ 
  (∃ (a b : ℝ), (a - b)^2 = foci_distance^2) →
  -1 < n ∧ n < 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l1569_156947


namespace NUMINAMATH_CALUDE_max_fraction_sum_l1569_156992

theorem max_fraction_sum (x y : ℝ) :
  (Real.sqrt 3 * x - y + Real.sqrt 3 ≥ 0) →
  (Real.sqrt 3 * x + y - Real.sqrt 3 ≤ 0) →
  (y ≥ 0) →
  (∀ x' y' : ℝ, (Real.sqrt 3 * x' - y' + Real.sqrt 3 ≥ 0) →
                (Real.sqrt 3 * x' + y' - Real.sqrt 3 ≤ 0) →
                (y' ≥ 0) →
                ((y' + 1) / (x' + 3) ≤ (y + 1) / (x + 3))) →
  x + y = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l1569_156992


namespace NUMINAMATH_CALUDE_optimal_hospital_location_l1569_156948

/-- Given three points A, B, and C in a plane, with AB = AC = 13 and BC = 10,
    prove that the point P(0, 4) on the perpendicular bisector of BC
    minimizes the sum of squares of distances PA^2 + PB^2 + PC^2 -/
theorem optimal_hospital_location (A B C P : ℝ × ℝ) :
  A = (0, 12) →
  B = (-5, 0) →
  C = (5, 0) →
  P.1 = 0 →
  (∀ y : ℝ, (0, y).1^2 + (0, y).2^2 + (-5 - 0)^2 + (0 - y)^2 + (5 - 0)^2 + (0 - y)^2 ≥
             (0, 4).1^2 + (0, 4).2^2 + (-5 - 0)^2 + (0 - 4)^2 + (5 - 0)^2 + (0 - 4)^2) →
  P = (0, 4) :=
by sorry

end NUMINAMATH_CALUDE_optimal_hospital_location_l1569_156948


namespace NUMINAMATH_CALUDE_danny_found_seven_caps_l1569_156943

/-- The number of bottle caps Danny found at the park -/
def bottleCapsFound (initialCaps currentCaps : ℕ) : ℕ :=
  currentCaps - initialCaps

/-- Proof that Danny found 7 bottle caps at the park -/
theorem danny_found_seven_caps : bottleCapsFound 25 32 = 7 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_seven_caps_l1569_156943


namespace NUMINAMATH_CALUDE_quadrilateral_cyclic_l1569_156949

-- Define the points
variable (A B C D P O B' D' X : EuclideanPlane)

-- Define the conditions
def is_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

def is_intersection (P : EuclideanPlane) (AB CD : Set EuclideanPlane) : Prop := sorry

def is_perpendicular_bisector_intersection (O : EuclideanPlane) (AB CD : Set EuclideanPlane) : Prop := sorry

def not_on_line (O : EuclideanPlane) (AB : Set EuclideanPlane) : Prop := sorry

def is_reflection (B' : EuclideanPlane) (B : EuclideanPlane) (OP : Set EuclideanPlane) : Prop := sorry

def meet_on_line (AB' CD' OP : Set EuclideanPlane) : Prop := sorry

def is_cyclic (A B C D : EuclideanPlane) : Prop := sorry

-- State the theorem
theorem quadrilateral_cyclic 
  (h1 : is_quadrilateral A B C D)
  (h2 : is_intersection P {A, B} {C, D})
  (h3 : is_perpendicular_bisector_intersection O {A, B} {C, D})
  (h4 : not_on_line O {A, B})
  (h5 : not_on_line O {C, D})
  (h6 : is_reflection B' B {O, P})
  (h7 : is_reflection D' D {O, P})
  (h8 : meet_on_line {A, B'} {C, D'} {O, P}) :
  is_cyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_cyclic_l1569_156949


namespace NUMINAMATH_CALUDE_inequalities_satisfaction_l1569_156913

theorem inequalities_satisfaction (a b c x y z : ℝ) 
  (hx : |x| < |a|) (hy : |y| < |b|) (hz : |z| < |c|) : 
  (|x*y| + |y*z| + |z*x| < |a*b| + |b*c| + |c*a|) ∧ 
  (x^2 + z^2 < a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfaction_l1569_156913


namespace NUMINAMATH_CALUDE_water_price_solution_l1569_156946

/-- Represents the problem of calculating water price per gallon -/
def water_price_problem (gallons_per_inch : ℝ) (monday_rain : ℝ) (tuesday_rain : ℝ) (total_revenue : ℝ) : Prop :=
  let total_gallons := gallons_per_inch * (monday_rain + tuesday_rain)
  let price_per_gallon := total_revenue / total_gallons
  price_per_gallon = 1.20

/-- The main theorem stating the solution to the water pricing problem -/
theorem water_price_solution :
  water_price_problem 15 4 3 126 := by
  sorry

#check water_price_solution

end NUMINAMATH_CALUDE_water_price_solution_l1569_156946


namespace NUMINAMATH_CALUDE_complex_real_condition_l1569_156969

theorem complex_real_condition (z m : ℂ) : z = (1 + Complex.I) * (1 + m * Complex.I) ∧ z.im = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1569_156969


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l1569_156967

def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_values (a : ℝ) :
  N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l1569_156967


namespace NUMINAMATH_CALUDE_negation_of_existence_rational_sqrt_two_l1569_156921

theorem negation_of_existence_rational_sqrt_two :
  (¬ ∃ (x : ℚ), x^2 - 2 = 0) ↔ (∀ (x : ℚ), x^2 - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_rational_sqrt_two_l1569_156921


namespace NUMINAMATH_CALUDE_power_equality_l1569_156963

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^50 = 4^x → x = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1569_156963


namespace NUMINAMATH_CALUDE_area_equality_l1569_156939

/-- Given a function g defined on {a, b, c}, prove that the area of the triangle
    formed by y = 3g(3x) is equal to the area of the triangle formed by y = g(x) -/
theorem area_equality (g : ℝ → ℝ) (a b c : ℝ) (area : ℝ) 
    (h1 : Set.range g = {g a, g b, g c})
    (h2 : area = 50)
    (h3 : area = abs ((b - a) * (g c - g a) - (c - a) * (g b - g a)) / 2) :
  abs ((b/3 - a/3) * (3 * g c - 3 * g a) - (c/3 - a/3) * (3 * g b - 3 * g a)) / 2 = area := by
  sorry

end NUMINAMATH_CALUDE_area_equality_l1569_156939


namespace NUMINAMATH_CALUDE_centroid_trajectory_l1569_156920

/-- The trajectory of the centroid of a triangle ABC, where A and B are fixed points
    and C moves on a hyperbola. -/
theorem centroid_trajectory
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (x y : ℝ)        -- Coordinates of the centroid
  (h1 : A = (0, 0))
  (h2 : B = (6, 0))
  (h3 : (C.1^2 / 16) - (C.2^2 / 9) = 1)  -- C moves on the hyperbola
  (h4 : x = (A.1 + B.1 + C.1) / 3)       -- Centroid x-coordinate
  (h5 : y = (A.2 + B.2 + C.2) / 3)       -- Centroid y-coordinate
  (h6 : y ≠ 0) :
  9 * (x - 2)^2 / 16 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_centroid_trajectory_l1569_156920


namespace NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l1569_156998

theorem smallest_k_for_no_real_roots : 
  ∃ (k : ℤ), k = 1 ∧ 
  (∀ (x : ℝ), (k + 1 : ℝ) * x^2 - (6 * k + 2 : ℝ) * x + (3 * k + 2 : ℝ) ≠ 0) ∧
  (∀ (j : ℤ), j < k → ∃ (x : ℝ), (j + 1 : ℝ) * x^2 - (6 * j + 2 : ℝ) * x + (3 * j + 2 : ℝ) = 0) :=
by sorry

#check smallest_k_for_no_real_roots

end NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l1569_156998


namespace NUMINAMATH_CALUDE_solution_p_proportion_l1569_156956

/-- Represents a solution mixture with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- The final mixture of solutions P and Q -/
structure Mixture where
  p : ℝ
  q : ℝ
  sum_to_one : p + q = 1

/-- Given two solutions and their mixture, prove that the proportion of Solution P is 0.4 -/
theorem solution_p_proportion
  (P : Solution)
  (Q : Solution)
  (M : Mixture)
  (h_P : P.carbonated_water = 0.8)
  (h_Q : Q.carbonated_water = 0.55)
  (h_M : P.carbonated_water * M.p + Q.carbonated_water * M.q = 0.65) :
  M.p = 0.4 := by
sorry

end NUMINAMATH_CALUDE_solution_p_proportion_l1569_156956


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l1569_156978

def digits : Finset Nat := {3, 0, 2, 5, 7}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

theorem smallest_three_digit_number :
  ∀ n, is_valid_number n → n ≥ 203 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l1569_156978


namespace NUMINAMATH_CALUDE_cookie_calorie_consumption_l1569_156925

/-- Represents the number of calories in a single cookie of each type -/
structure CookieCalories where
  caramel : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Represents the number of cookies selected of each type -/
structure SelectedCookies where
  caramel : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Calculates the total calories consumed based on the number of cookies selected and their calorie content -/
def totalCalories (calories : CookieCalories) (selected : SelectedCookies) : ℕ :=
  calories.caramel * selected.caramel +
  calories.chocolate_chip * selected.chocolate_chip +
  calories.peanut_butter * selected.peanut_butter

/-- Proves that selecting 5 caramel, 3 chocolate chip, and 2 peanut butter cookies results in consuming 204 calories -/
theorem cookie_calorie_consumption :
  let calories := CookieCalories.mk 18 22 24
  let selected := SelectedCookies.mk 5 3 2
  totalCalories calories selected = 204 := by
  sorry

end NUMINAMATH_CALUDE_cookie_calorie_consumption_l1569_156925


namespace NUMINAMATH_CALUDE_prob_defective_bulb_selection_l1569_156917

/-- Given a box of electric bulbs, this function calculates the probability of
    selecting at least one defective bulb when choosing two bulbs at random. -/
def prob_at_least_one_defective (total : ℕ) (defective : ℕ) : ℚ :=
  1 - (total - defective : ℚ) / total * ((total - defective - 1) : ℚ) / (total - 1)

/-- Theorem stating that for a box with 24 bulbs, 4 of which are defective,
    the probability of choosing at least one defective bulb when randomly
    selecting two bulbs is equal to 43/138. -/
theorem prob_defective_bulb_selection :
  prob_at_least_one_defective 24 4 = 43 / 138 := by
  sorry

end NUMINAMATH_CALUDE_prob_defective_bulb_selection_l1569_156917


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l1569_156930

theorem sum_of_multiples_is_even (a b : ℤ) (ha : 4 ∣ a) (hb : 6 ∣ b) : Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l1569_156930


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1569_156999

/-- 
Given a geometric sequence where the fifth term is 64 and the sixth term is 128,
prove that the first term of the sequence is 4.
-/
theorem geometric_sequence_first_term (a b c d : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    b = a * r ∧ 
    c = b * r ∧ 
    d = c * r ∧ 
    64 = d * r ∧ 
    128 = 64 * r) → 
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1569_156999


namespace NUMINAMATH_CALUDE_outfit_combinations_l1569_156951

/-- The number of possible outfits given the number of shirts, ties, and pants -/
def number_of_outfits (shirts ties pants : ℕ) : ℕ := shirts * ties * pants

/-- Theorem: Given 8 shirts, 6 ties, and 4 pairs of pants, the number of possible outfits is 192 -/
theorem outfit_combinations : number_of_outfits 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1569_156951


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_times_14_l1569_156928

def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, 2; 0, 4, 1; 0, 0, 2]

theorem B_power_15_minus_3_times_14 :
  B^15 - 3 • (B^14) = !![0, 3, 1; 0, 4, 1; 0, 0, -2] := by
  sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_times_14_l1569_156928


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1569_156979

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = 7/5 * 90 →
  -- One angle is 20° larger than the other
  b = a + 20 →
  -- All angles are non-negative
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 73°
  max a (max b c) = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1569_156979


namespace NUMINAMATH_CALUDE_cards_distribution_l1569_156902

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) 
  (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l1569_156902


namespace NUMINAMATH_CALUDE_sin_cos_product_l1569_156929

theorem sin_cos_product (α : Real) (h : Real.sin α + Real.cos α = Real.sqrt 2) : 
  Real.sin α * Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l1569_156929


namespace NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l1569_156981

/-- Represents months as integers from 1 to 12 -/
def Month := Fin 12

/-- Convert a number of months to a Month value -/
def monthsToMonth (n : ℕ) : Month :=
  ⟨(n - 1) % 12 + 1, by sorry⟩

/-- January represented as a Month -/
def january : Month := ⟨1, by sorry⟩

/-- December represented as a Month -/
def december : Month := ⟨12, by sorry⟩

/-- The number of months between replacements -/
def replacementInterval : ℕ := 7

/-- The number of the replacement we're interested in -/
def targetReplacement : ℕ := 18

theorem eighteenth_replacement_in_december :
  monthsToMonth (replacementInterval * (targetReplacement - 1) + 1) = december := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l1569_156981


namespace NUMINAMATH_CALUDE_black_midwest_percentage_is_31_l1569_156965

/-- Represents the population data for different ethnic groups in different regions --/
structure PopulationData :=
  (ne_white : ℕ) (mw_white : ℕ) (south_white : ℕ) (west_white : ℕ)
  (ne_black : ℕ) (mw_black : ℕ) (south_black : ℕ) (west_black : ℕ)
  (ne_asian : ℕ) (mw_asian : ℕ) (south_asian : ℕ) (west_asian : ℕ)
  (ne_hispanic : ℕ) (mw_hispanic : ℕ) (south_hispanic : ℕ) (west_hispanic : ℕ)

/-- Calculates the percentage of Black population in the Midwest --/
def black_midwest_percentage (data : PopulationData) : ℚ :=
  let total_black := data.ne_black + data.mw_black + data.south_black + data.west_black
  (data.mw_black : ℚ) / total_black * 100

/-- Rounds a rational number to the nearest integer --/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- The main theorem stating that the rounded percentage of Black population in the Midwest is 31% --/
theorem black_midwest_percentage_is_31 (data : PopulationData) 
  (h : data = { ne_white := 45, mw_white := 55, south_white := 60, west_white := 40,
                ne_black := 6, mw_black := 12, south_black := 18, west_black := 3,
                ne_asian := 2, mw_asian := 2, south_asian := 2, west_asian := 5,
                ne_hispanic := 2, mw_hispanic := 3, south_hispanic := 4, west_hispanic := 6 }) :
  round_to_nearest (black_midwest_percentage data) = 31 := by
  sorry

end NUMINAMATH_CALUDE_black_midwest_percentage_is_31_l1569_156965


namespace NUMINAMATH_CALUDE_chenny_cups_bought_l1569_156993

def plate_cost : ℝ := 2
def spoon_cost : ℝ := 1.5
def fork_cost : ℝ := 1.25
def cup_cost : ℝ := 3
def num_plates : ℕ := 9

def total_spoons_forks_cost : ℝ := 13.5
def total_plates_cups_cost : ℝ := 25.5

theorem chenny_cups_bought :
  ∃ (num_spoons num_forks num_cups : ℕ),
    num_spoons = num_forks ∧
    num_spoons * spoon_cost + num_forks * fork_cost = total_spoons_forks_cost ∧
    num_plates * plate_cost + num_cups * cup_cost = total_plates_cups_cost ∧
    num_cups = 2 :=
by sorry

end NUMINAMATH_CALUDE_chenny_cups_bought_l1569_156993


namespace NUMINAMATH_CALUDE_D_l1569_156985

def D' : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => D' (n + 2) + D' (n + 1) + D' n

theorem D'_parity_2024_2025_2026 :
  Even (D' 2024) ∧ Odd (D' 2025) ∧ Odd (D' 2026) :=
by
  sorry

end NUMINAMATH_CALUDE_D_l1569_156985


namespace NUMINAMATH_CALUDE_inequality_proof_l1569_156976

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1569_156976


namespace NUMINAMATH_CALUDE_one_quarter_between_thirds_l1569_156909

theorem one_quarter_between_thirds (x : ℚ) : 
  (x = 1/3 + 1/4 * (2/3 - 1/3)) → x = 5/12 := by
sorry

end NUMINAMATH_CALUDE_one_quarter_between_thirds_l1569_156909


namespace NUMINAMATH_CALUDE_female_student_count_l1569_156933

theorem female_student_count (total_students : ℕ) (selection_ways : ℕ) :
  total_students = 8 →
  selection_ways = 30 →
  (∃ (male_students : ℕ) (female_students : ℕ),
    male_students + female_students = total_students ∧
    (male_students.choose 2) * female_students = selection_ways ∧
    (female_students = 2 ∨ female_students = 3)) :=
by sorry

end NUMINAMATH_CALUDE_female_student_count_l1569_156933


namespace NUMINAMATH_CALUDE_steve_coins_value_l1569_156912

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the total value of coins given the number of nickels and dimes -/
def total_value (nickels dimes : ℕ) : ℕ :=
  nickels * nickel_value + dimes * dime_value

/-- Proves that given 2 nickels and 4 more dimes than nickels, the total value is 70 cents -/
theorem steve_coins_value : 
  ∀ (nickels : ℕ), nickels = 2 → total_value nickels (nickels + 4) = 70 := by
  sorry

end NUMINAMATH_CALUDE_steve_coins_value_l1569_156912


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1569_156968

/-- Represents the number of employees in each job category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  midLevel : ℕ
  junior : ℕ

/-- Represents the number of sampled employees in each job category -/
structure SampleCount where
  senior : ℕ
  midLevel : ℕ
  junior : ℕ

/-- Checks if the sample counts are correct for stratified sampling -/
def isCorrectStratifiedSample (ec : EmployeeCount) (sc : SampleCount) (sampleSize : ℕ) : Prop :=
  sc.senior = ec.senior * sampleSize / ec.total ∧
  sc.midLevel = ec.midLevel * sampleSize / ec.total ∧
  sc.junior = ec.junior * sampleSize / ec.total

theorem correct_stratified_sample :
  let ec : EmployeeCount := ⟨450, 45, 135, 270⟩
  let sc : SampleCount := ⟨3, 9, 18⟩
  isCorrectStratifiedSample ec sc 30 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1569_156968


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1569_156960

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 11*p - 3 = 0 →
  q^3 - 8*q^2 + 11*q - 3 = 0 →
  r^3 - 8*r^2 + 11*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 32/15 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1569_156960


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1569_156900

/-- The number of puppies available in the pet store -/
def num_puppies : ℕ := 10

/-- The number of kittens available in the pet store -/
def num_kittens : ℕ := 6

/-- The number of hamsters available in the pet store -/
def num_hamsters : ℕ := 8

/-- The total number of ways Alice, Bob, and Charlie can buy pets and leave the store satisfied -/
def total_ways : ℕ := 960

/-- Theorem stating that the number of ways Alice, Bob, and Charlie can buy pets
    and leave the store satisfied is equal to total_ways -/
theorem pet_store_combinations :
  (num_puppies * num_kittens * num_hamsters) +
  (num_kittens * num_puppies * num_hamsters) = total_ways :=
by sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1569_156900


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l1569_156994

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := sorry

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := 53

/-- The difference between the frog's jump and the grasshopper's jump in inches -/
def frog_grasshopper_diff : ℕ := 17

theorem grasshopper_jump_distance :
  grasshopper_jump = frog_jump - frog_grasshopper_diff :=
by sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l1569_156994


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1569_156914

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1569_156914


namespace NUMINAMATH_CALUDE_project_time_difference_l1569_156932

/-- Represents the working times of three people on a project -/
structure ProjectTime where
  t1 : ℕ  -- Time of person 1
  t2 : ℕ  -- Time of person 2
  t3 : ℕ  -- Time of person 3

/-- The proposition that the working times are in the ratio 1:2:3 -/
def ratio_correct (pt : ProjectTime) : Prop :=
  2 * pt.t1 = pt.t2 ∧ 3 * pt.t1 = pt.t3

/-- The total project time is 120 hours -/
def total_time_correct (pt : ProjectTime) : Prop :=
  pt.t1 + pt.t2 + pt.t3 = 120

/-- The main theorem stating the difference between longest and shortest working times -/
theorem project_time_difference (pt : ProjectTime) 
  (h1 : ratio_correct pt) (h2 : total_time_correct pt) : 
  pt.t3 - pt.t1 = 40 := by
  sorry


end NUMINAMATH_CALUDE_project_time_difference_l1569_156932


namespace NUMINAMATH_CALUDE_g_difference_l1569_156988

/-- A linear function with a constant difference of 4 between consecutive integers -/
def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
  (∀ h : ℝ, g (h + 1) - g h = 4)

/-- The difference between g(3) and g(7) is -16 -/
theorem g_difference (g : ℝ → ℝ) (hg : g_property g) : g 3 - g 7 = -16 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1569_156988


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1569_156910

theorem purely_imaginary_complex_number (i : ℂ) (a : ℝ) : 
  i * i = -1 →
  (∃ (b : ℝ), (1 + a * i) / (2 - i) = b * i) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1569_156910


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1569_156970

-- Problem 1
theorem problem_1 (x : ℝ) : 4 * (x + 1)^2 - (2 * x + 5) * (2 * x - 5) = 8 * x + 29 := by
  sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) : 
  (4 * a * b)^2 * (-1/4 * a^4 * b^3 * c^2) / (-4 * a^3 * b^2 * c^2) = a^3 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1569_156970


namespace NUMINAMATH_CALUDE_juliet_age_l1569_156996

theorem juliet_age (maggie ralph juliet : ℕ) 
  (h1 : juliet = maggie + 3)
  (h2 : juliet = ralph - 2)
  (h3 : maggie + ralph = 19) :
  juliet = 10 := by
  sorry

end NUMINAMATH_CALUDE_juliet_age_l1569_156996


namespace NUMINAMATH_CALUDE_number_of_students_l1569_156971

theorem number_of_students (student_avg : ℝ) (teacher_age : ℝ) (new_avg : ℝ) :
  student_avg = 26 →
  teacher_age = 52 →
  new_avg = 27 →
  ∃ n : ℕ, (n : ℝ) * student_avg + teacher_age = (n + 1) * new_avg ∧ n = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1569_156971


namespace NUMINAMATH_CALUDE_median_length_l1569_156945

/-- Triangle ABC with given side lengths and median BM --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  BM : ℝ
  h_AB : AB = 5
  h_BC : BC = 12
  h_AC : AC = 13
  h_BM : ∃ m : ℝ, BM = m * Real.sqrt 2

/-- The value of m in the equation BM = m√2 is 13/2 --/
theorem median_length (t : Triangle) : ∃ m : ℝ, t.BM = m * Real.sqrt 2 ∧ m = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_median_length_l1569_156945


namespace NUMINAMATH_CALUDE_triangle_inequality_l1569_156901

theorem triangle_inequality (a b c n : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) (h_n : 1 ≤ n) :
  let s := (a + b + c) / 2
  (a^n / (b + c) + b^n / (c + a) + c^n / (a + b)) ≥ (2/3)^(n-2) * s^(n-1) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1569_156901


namespace NUMINAMATH_CALUDE_platform_length_l1569_156973

/-- Given a train and platform with specific properties, prove the platform length --/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 36) :
  ∃ platform_length : ℝ,
    platform_length = 25 ∧
    (train_length + platform_length) / time_cross_platform = train_length / time_cross_pole :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1569_156973


namespace NUMINAMATH_CALUDE_max_good_quadratics_less_than_500_l1569_156982

/-- A good quadratic trinomial has distinct coefficients and two distinct real roots -/
def is_good_quadratic (a b c : ℕ+) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b.val : ℝ)^2 > 4 * (a.val : ℝ) * (c.val : ℝ)

/-- The set of 10 positive integers from which coefficients are chosen -/
def coefficient_set : Finset ℕ+ :=
  sorry

/-- The set of all good quadratic trinomials formed from the coefficient set -/
def good_quadratics : Finset (ℕ+ × ℕ+ × ℕ+) :=
  sorry

theorem max_good_quadratics_less_than_500 :
  Finset.card good_quadratics < 500 :=
sorry

end NUMINAMATH_CALUDE_max_good_quadratics_less_than_500_l1569_156982


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_equals_sqrt_3_l1569_156950

theorem sqrt_12_minus_sqrt_3_equals_sqrt_3 : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_equals_sqrt_3_l1569_156950


namespace NUMINAMATH_CALUDE_number_of_divisors_of_90_l1569_156953

theorem number_of_divisors_of_90 : Finset.card (Nat.divisors 90) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_90_l1569_156953


namespace NUMINAMATH_CALUDE_divisibility_pairs_l1569_156980

theorem divisibility_pairs : 
  {p : ℕ × ℕ | p.1 ∣ (2^(Nat.totient p.2) + 1) ∧ p.2 ∣ (2^(Nat.totient p.1) + 1)} = 
  {(1, 1), (1, 3), (3, 1)} := by
sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l1569_156980


namespace NUMINAMATH_CALUDE_fixed_points_bound_l1569_156926

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n

/-- Evaluate a polynomial at a point -/
def evalPoly (p : IntPolynomial n) (x : ℤ) : ℤ := sorry

/-- Compose a polynomial with itself k times -/
def composeK (p : IntPolynomial n) (k : ℕ) : IntPolynomial n := sorry

/-- The number of integer fixed points of a polynomial -/
def numIntFixedPoints (p : IntPolynomial n) : ℕ := sorry

/-- Main theorem: The number of integer fixed points of Q is at most n -/
theorem fixed_points_bound (n k : ℕ) (p : IntPolynomial n) 
  (h1 : n > 1) (h2 : k > 0) : 
  numIntFixedPoints (composeK p k) ≤ n := by sorry

end NUMINAMATH_CALUDE_fixed_points_bound_l1569_156926


namespace NUMINAMATH_CALUDE_target_hit_probability_l1569_156934

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.5) (h2 : p2 = 0.7) :
  1 - (1 - p1) * (1 - p2) = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1569_156934


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1569_156977

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1569_156977


namespace NUMINAMATH_CALUDE_triplet_satisfies_conditions_l1569_156964

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Checks if three numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r

theorem triplet_satisfies_conditions : 
  isPrime 17 ∧ isPrime 23 ∧ isPrime 31 ∧
  17 < 23 ∧ 23 < 31 ∧ 31 < 100 ∧
  isGeometricSequence 18 24 32 :=
by sorry

end NUMINAMATH_CALUDE_triplet_satisfies_conditions_l1569_156964


namespace NUMINAMATH_CALUDE_prime_sequence_existence_l1569_156954

theorem prime_sequence_existence (k : ℕ) (hk : k > 1) :
  ∃ (p : ℕ) (a : ℕ → ℕ),
    Prime p ∧
    (∀ n m, n < m → a n < a m) ∧
    (∀ n, n > 1 → Prime (p + k * a n)) := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_existence_l1569_156954


namespace NUMINAMATH_CALUDE_expression_simplification_l1569_156991

theorem expression_simplification (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a*b + b*c + c*a + 3)⁻¹ * ((a*b)⁻¹ + (b*c)⁻¹ + (c*a)⁻¹ + 3) = (a*b*c)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1569_156991


namespace NUMINAMATH_CALUDE_isosceles_triangle_property_l1569_156940

/-- Represents a triangle with vertices A, B, C and incentre I -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  I : ℝ × ℝ

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The squared distance between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ := sorry

/-- Check if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  distance t.A t.B = distance t.A t.C

/-- The distance from a point to a line defined by two points -/
def distanceToLine (p : ℝ × ℝ) (q r : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In an isosceles triangle ABC with incentre I, 
    if AB = AC, AI = 3, and the distance from I to BC is 2, then BC² = 80 -/
theorem isosceles_triangle_property (t : Triangle) :
  isIsosceles t →
  distance t.A t.I = 3 →
  distanceToLine t.I t.B t.C = 2 →
  distanceSquared t.B t.C = 80 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_property_l1569_156940


namespace NUMINAMATH_CALUDE_cone_volume_lateral_area_l1569_156995

/-- The volume of a cone in terms of its lateral surface area and the distance from the center of the base to the slant height. -/
theorem cone_volume_lateral_area (S r : ℝ) (h1 : S > 0) (h2 : r > 0) : ∃ V : ℝ, V = (1/3) * S * r ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_lateral_area_l1569_156995


namespace NUMINAMATH_CALUDE_inverse_proportion_change_l1569_156908

theorem inverse_proportion_change (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = c) :
  let a' := 1.2 * a
  let b' := 80
  a' * b' = c →
  b = 96 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_change_l1569_156908


namespace NUMINAMATH_CALUDE_product_a4b4_l1569_156906

theorem product_a4b4 (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by sorry

end NUMINAMATH_CALUDE_product_a4b4_l1569_156906


namespace NUMINAMATH_CALUDE_point_y_coordinate_l1569_156972

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the distance from a point to the focus
def distance_to_focus (x y : ℝ) : ℝ := 4

-- Define the y-coordinate of the directrix
def directrix_y : ℝ := -1

-- Theorem statement
theorem point_y_coordinate (x y : ℝ) :
  is_on_parabola x y →
  distance_to_focus x y = 4 →
  y = 3 := by sorry

end NUMINAMATH_CALUDE_point_y_coordinate_l1569_156972


namespace NUMINAMATH_CALUDE_commercial_reduction_l1569_156904

def original_length : ℝ := 30
def reduction_percentage : ℝ := 0.30

theorem commercial_reduction :
  original_length * (1 - reduction_percentage) = 21 := by
  sorry

end NUMINAMATH_CALUDE_commercial_reduction_l1569_156904


namespace NUMINAMATH_CALUDE_number_of_teachers_l1569_156924

/-- Represents the number of students at King Middle School -/
def total_students : ℕ := 1200

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 5

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 4

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 30

/-- Represents the number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem stating that the number of teachers at King Middle School is 50 -/
theorem number_of_teachers : 
  (total_students * classes_per_student) / students_per_class / classes_per_teacher = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teachers_l1569_156924


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l1569_156990

/-- Define the box operation for integers a, b, and c -/
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

/-- Theorem stating that box(2, -2, 3) = 69/4 -/
theorem box_2_neg2_3 : box 2 (-2) 3 = 69/4 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l1569_156990


namespace NUMINAMATH_CALUDE_tangent_slope_parabola_l1569_156974

/-- The slope of the tangent line to y = (1/5)x^2 at (2, 4/5) is 4/5 -/
theorem tangent_slope_parabola :
  let f (x : ℝ) := (1/5) * x^2
  let a : ℝ := 2
  let slope := (deriv f) a
  slope = 4/5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_parabola_l1569_156974


namespace NUMINAMATH_CALUDE_paul_prediction_accuracy_l1569_156941

/-- Represents a team in the FIFA World Cup -/
inductive Team
| Ghana
| Bolivia
| Argentina
| France

/-- The probability of a team winning the tournament -/
def winProbability (t : Team) : ℚ :=
  match t with
  | Team.Ghana => 1/2
  | Team.Bolivia => 1/6
  | Team.Argentina => 1/6
  | Team.France => 1/6

/-- The probability of Paul correctly predicting the winner -/
def paulCorrectProbability : ℚ :=
  (winProbability Team.Ghana)^2 +
  (winProbability Team.Bolivia)^2 +
  (winProbability Team.Argentina)^2 +
  (winProbability Team.France)^2

theorem paul_prediction_accuracy :
  paulCorrectProbability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_paul_prediction_accuracy_l1569_156941


namespace NUMINAMATH_CALUDE_pencil_distribution_l1569_156987

/-- The number of ways to distribute n identical objects among k people,
    where each person gets at least one object. -/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 identical pencils among 3 friends,
    where each friend gets at least one pencil. -/
theorem pencil_distribution : distributionWays 6 3 = 10 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1569_156987


namespace NUMINAMATH_CALUDE_relationship_xyz_l1569_156931

theorem relationship_xyz (x y z : ℝ) 
  (hx : x = Real.log π) 
  (hy : y = Real.log 2 / Real.log 5)
  (hz : z = Real.exp (-1/2)) :
  y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l1569_156931


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1569_156918

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (1 - 4 * x) = 5 → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1569_156918
