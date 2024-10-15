import Mathlib

namespace NUMINAMATH_CALUDE_power_of_product_l4076_407669

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4076_407669


namespace NUMINAMATH_CALUDE_units_digit_of_n_l4076_407696

/-- Given two natural numbers m and n, where mn = 17^6 and m has a units digit of 8,
    prove that the units digit of n is 2. -/
theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 17^6) (h2 : m % 10 = 8) : n % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l4076_407696


namespace NUMINAMATH_CALUDE_problem_solution_l4076_407659

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l4076_407659


namespace NUMINAMATH_CALUDE_solution_proof_l4076_407661

-- Part 1: System of equations
def satisfies_system (x y : ℝ) : Prop :=
  2 * x - y = 5 ∧ 3 * x + 4 * y = 2

-- Part 2: System of inequalities
def satisfies_inequalities (x : ℝ) : Prop :=
  -2 * x < 6 ∧ 3 * (x - 2) ≤ x - 4

-- Part 3: Integer solutions
def is_integer_solution (x : ℤ) : Prop :=
  -3 < (x : ℝ) ∧ (x : ℝ) ≤ 1

theorem solution_proof :
  -- Part 1
  satisfies_system 2 (-1) ∧
  -- Part 2
  (∀ x : ℝ, satisfies_inequalities x ↔ -3 < x ∧ x ≤ 1) ∧
  -- Part 3
  (∀ x : ℤ, is_integer_solution x ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_proof_l4076_407661


namespace NUMINAMATH_CALUDE_sum_cannot_have_all_odd_digits_l4076_407662

/-- A digit is a natural number between 0 and 9. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- A sequence of 1001 digits. -/
def DigitSequence : Type := Fin 1001 → Digit

/-- The first number formed by the digit sequence. -/
def firstNumber (a : DigitSequence) : ℕ := sorry

/-- The second number formed by the reversed digit sequence. -/
def secondNumber (a : DigitSequence) : ℕ := sorry

/-- A number has all odd digits if each of its digits is odd. -/
def hasAllOddDigits (n : ℕ) : Prop := sorry

theorem sum_cannot_have_all_odd_digits (a : DigitSequence) :
  ¬(hasAllOddDigits (firstNumber a + secondNumber a)) :=
sorry

end NUMINAMATH_CALUDE_sum_cannot_have_all_odd_digits_l4076_407662


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4076_407649

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y t : ℝ) : Prop := x - 3*y + t = 0

-- Define point M
def point_M (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the asymptotes
def asymptotes (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k*x ∨ y = -k*x}

-- Theorem statement
theorem hyperbola_asymptotes 
  (a b t : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (ht : t ≠ 0) :
  ∃ (A B : ℝ × ℝ),
    (∀ x y, hyperbola a b x y → line x y t → (x, y) ∈ asymptotes (1/2)) ∧
    (A ∈ asymptotes (1/2) ∧ B ∈ asymptotes (1/2)) ∧
    (line A.1 A.2 t ∧ line B.1 B.2 t) ∧
    (dist (point_M t) A = dist (point_M t) B) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4076_407649


namespace NUMINAMATH_CALUDE_amount_calculation_l4076_407678

theorem amount_calculation (a b : ℝ) 
  (h1 : a + b = 1210)
  (h2 : (1/3) * a = (1/4) * b) : 
  b = 4840 / 7 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l4076_407678


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l4076_407695

theorem average_of_a_and_b (a b c : ℝ) : 
  (4 + 6 + 8 + 12 + a + b + c) / 7 = 20 →
  a + b + c = 3 * ((4 + 6 + 8) / 3) →
  (a + b) / 2 = (18 - c) / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l4076_407695


namespace NUMINAMATH_CALUDE_square_sum_nonzero_implies_nonzero_element_l4076_407622

theorem square_sum_nonzero_implies_nonzero_element (a b : ℝ) : 
  a^2 + b^2 ≠ 0 → (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_implies_nonzero_element_l4076_407622


namespace NUMINAMATH_CALUDE_inner_hexagon_area_l4076_407691

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℝ
  area : ℝ

/-- Represents the configuration of triangles in the problem --/
structure TriangleConfiguration where
  largeTriangle : EquilateralTriangle
  smallTriangles : List EquilateralTriangle
  innerHexagonArea : ℝ

/-- The given configuration satisfies the problem conditions --/
def satisfiesProblemConditions (config : TriangleConfiguration) : Prop :=
  config.smallTriangles.length = 6 ∧
  config.smallTriangles.map (λ t => t.area) = [1, 1, 9, 9, 16, 16]

/-- The theorem to be proved --/
theorem inner_hexagon_area 
  (config : TriangleConfiguration) 
  (h : satisfiesProblemConditions config) : 
  config.innerHexagonArea = 38 := by
  sorry

end NUMINAMATH_CALUDE_inner_hexagon_area_l4076_407691


namespace NUMINAMATH_CALUDE_park_bench_spaces_l4076_407650

/-- Calculates the number of available spaces on benches in a park. -/
def availableSpaces (numBenches : ℕ) (capacityPerBench : ℕ) (peopleSitting : ℕ) : ℕ :=
  numBenches * capacityPerBench - peopleSitting

/-- Theorem stating that there are 120 available spaces on the benches. -/
theorem park_bench_spaces :
  availableSpaces 50 4 80 = 120 := by
  sorry

end NUMINAMATH_CALUDE_park_bench_spaces_l4076_407650


namespace NUMINAMATH_CALUDE_hyperbola_properties_l4076_407693

/-- An equilateral hyperbola with foci on the x-axis passing through (4, -2) -/
def equilateralHyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 12

theorem hyperbola_properties :
  -- The hyperbola is equilateral
  ∀ (x y : ℝ), equilateralHyperbola x y → x^2 - y^2 = 12 ∧
  -- The foci are on the x-axis (implied by the equation form)
  -- The hyperbola passes through the point (4, -2)
  equilateralHyperbola 4 (-2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l4076_407693


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4076_407601

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4076_407601


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l4076_407676

theorem min_reciprocal_sum (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1/x + 1/y + 1/z ≥ 3 ∧ ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 ∧ 1/a + 1/b + 1/c = 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l4076_407676


namespace NUMINAMATH_CALUDE_minimum_score_for_eligibility_l4076_407665

def minimum_score (q1 q2 q3 : ℚ) (target_average : ℚ) : ℚ :=
  4 * target_average - (q1 + q2 + q3)

theorem minimum_score_for_eligibility 
  (q1 q2 q3 : ℚ) 
  (target_average : ℚ) 
  (h1 : q1 = 80) 
  (h2 : q2 = 85) 
  (h3 : q3 = 78) 
  (h4 : target_average = 85) :
  minimum_score q1 q2 q3 target_average = 97 := by
sorry

end NUMINAMATH_CALUDE_minimum_score_for_eligibility_l4076_407665


namespace NUMINAMATH_CALUDE_sixth_edge_possibilities_l4076_407699

/-- Represents the edge lengths of a tetrahedron -/
structure TetrahedronEdges :=
  (a b c d e f : ℕ)

/-- Checks if three lengths satisfy the triangle inequality -/
def satisfiesTriangleInequality (x y z : ℕ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Checks if all faces of a tetrahedron satisfy the triangle inequality -/
def validTetrahedron (t : TetrahedronEdges) : Prop :=
  satisfiesTriangleInequality t.a t.b t.c ∧
  satisfiesTriangleInequality t.a t.d t.e ∧
  satisfiesTriangleInequality t.b t.d t.f ∧
  satisfiesTriangleInequality t.c t.e t.f

/-- The main theorem stating that there are exactly 6 possible lengths for the sixth edge -/
theorem sixth_edge_possibilities :
  ∃! (s : Finset ℕ),
    s.card = 6 ∧
    (∀ x, x ∈ s ↔ ∃ t : TetrahedronEdges,
      t.a = 14 ∧ t.b = 20 ∧ t.c = 40 ∧ t.d = 52 ∧ t.e = 70 ∧ t.f = x ∧
      validTetrahedron t) :=
by sorry


end NUMINAMATH_CALUDE_sixth_edge_possibilities_l4076_407699


namespace NUMINAMATH_CALUDE_shoeing_time_for_48_blacksmiths_60_horses_l4076_407687

/-- The minimum time required for a group of blacksmiths to shoe a group of horses -/
def minimum_shoeing_time (num_blacksmiths : ℕ) (num_horses : ℕ) (time_per_horseshoe : ℕ) : ℕ :=
  let total_horseshoes := num_horses * 4
  let total_time := total_horseshoes * time_per_horseshoe
  total_time / num_blacksmiths

theorem shoeing_time_for_48_blacksmiths_60_horses : 
  minimum_shoeing_time 48 60 5 = 25 := by
  sorry

#eval minimum_shoeing_time 48 60 5

end NUMINAMATH_CALUDE_shoeing_time_for_48_blacksmiths_60_horses_l4076_407687


namespace NUMINAMATH_CALUDE_mason_savings_l4076_407619

theorem mason_savings (savings : ℝ) (total_books : ℕ) (book_price : ℝ) 
  (h1 : savings > 0) 
  (h2 : total_books > 0) 
  (h3 : book_price > 0) 
  (h4 : (1/4) * savings = (2/5) * total_books * book_price) : 
  savings - total_books * book_price = (3/8) * savings := by
sorry

end NUMINAMATH_CALUDE_mason_savings_l4076_407619


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_eccentricity_is_four_l4076_407681

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity : ℝ → ℝ → ℝ → Prop :=
  fun a b e =>
    -- Hyperbola equation
    (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 →
    -- Parabola equation
    ∃ x₀, ∀ y, y^2 = 16 * x₀ →
    -- Right focus of hyperbola coincides with focus of parabola
    4 = (a^2 + b^2).sqrt →
    -- Eccentricity definition
    e = (a^2 + b^2).sqrt / a →
    -- Prove eccentricity is 4
    e = 4)

/-- The main theorem stating the eccentricity is 4 -/
theorem eccentricity_is_four :
  ∃ a b e, hyperbola_eccentricity a b e :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_eccentricity_is_four_l4076_407681


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l4076_407653

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a * x^2 - 2*x + 2 > 0) →
  a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l4076_407653


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l4076_407620

theorem negative_integer_equation_solution :
  ∃ (N : ℤ), N < 0 ∧ 3 * N^2 + N = 15 → N = -3 :=
by sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l4076_407620


namespace NUMINAMATH_CALUDE_employee_pay_l4076_407615

/-- Proves that employee y is paid 268.18 per week given the conditions -/
theorem employee_pay (x y : ℝ) (h1 : x + y = 590) (h2 : x = 1.2 * y) : y = 268.18 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l4076_407615


namespace NUMINAMATH_CALUDE_middle_circle_radius_l4076_407610

/-- A sequence of five circles tangent to two parallel lines and to each other -/
structure CircleSequence where
  radii : Fin 5 → ℝ
  tangent_to_lines : Bool
  sequentially_tangent : Bool

/-- The property that the radii form a geometric sequence -/
def is_geometric_sequence (cs : CircleSequence) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, cs.radii i.succ = cs.radii i * r

theorem middle_circle_radius 
  (cs : CircleSequence)
  (h_tangent : cs.tangent_to_lines = true)
  (h_seq_tangent : cs.sequentially_tangent = true)
  (h_geometric : is_geometric_sequence cs)
  (h_smallest : cs.radii 0 = 8)
  (h_largest : cs.radii 4 = 18) :
  cs.radii 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l4076_407610


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l4076_407682

theorem half_abs_diff_squares_20_15 : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l4076_407682


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l4076_407611

theorem cost_increase_percentage (initial_cost final_cost : ℝ) 
  (h1 : initial_cost = 75)
  (h2 : final_cost = 72)
  (h3 : ∃ x : ℝ, final_cost = (initial_cost + (x / 100) * initial_cost) * 0.8) :
  ∃ x : ℝ, x = 20 ∧ final_cost = (initial_cost + (x / 100) * initial_cost) * 0.8 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l4076_407611


namespace NUMINAMATH_CALUDE_age_problem_l4076_407634

theorem age_problem (x : ℝ) : (1/2) * (8 * (x + 8) - 8 * (x - 8)) = x ↔ x = 64 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l4076_407634


namespace NUMINAMATH_CALUDE_triangle_area_perimeter_inequality_triangle_area_perimeter_equality_l4076_407664

/-- Represents a triangle with area and perimeter -/
structure Triangle where
  area : ℝ
  perimeter : ℝ

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  sorry -- Definition of equilateral triangle

theorem triangle_area_perimeter_inequality (t : Triangle) :
  36 * t.area ≤ t.perimeter^2 * Real.sqrt 3 :=
sorry

theorem triangle_area_perimeter_equality (t : Triangle) :
  36 * t.area = t.perimeter^2 * Real.sqrt 3 ↔ IsEquilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_area_perimeter_inequality_triangle_area_perimeter_equality_l4076_407664


namespace NUMINAMATH_CALUDE_wire_length_ratio_l4076_407635

-- Define the given constants
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length : ℕ := 8
def roark_wire_length : ℕ := 2

-- Define Bonnie's prism
def bonnie_prism_volume : ℕ := (bonnie_wire_length / 2) ^ 3

-- Define Roark's unit prism
def roark_unit_prism_volume : ℕ := roark_wire_length ^ 3

-- Define the number of Roark's prisms
def roark_prism_count : ℕ := bonnie_prism_volume / roark_unit_prism_volume

-- Define the total wire lengths
def bonnie_total_wire : ℕ := bonnie_wire_pieces * bonnie_wire_length
def roark_total_wire : ℕ := roark_prism_count * (12 * roark_wire_length)

-- Theorem to prove
theorem wire_length_ratio :
  bonnie_total_wire / roark_total_wire = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l4076_407635


namespace NUMINAMATH_CALUDE_kara_water_consumption_l4076_407692

/-- The amount of water Kara drinks with each medication dose in ounces -/
def water_per_dose : ℕ := 4

/-- The number of times Kara takes her medication per day -/
def doses_per_day : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of times Kara forgot to take her medication in the second week -/
def forgotten_doses : ℕ := 2

/-- The total amount of water Kara drank with her medication over two weeks -/
def total_water : ℕ := 
  (water_per_dose * doses_per_day * days_per_week) + 
  (water_per_dose * (doses_per_day * days_per_week - forgotten_doses))

theorem kara_water_consumption : total_water = 160 := by
  sorry

end NUMINAMATH_CALUDE_kara_water_consumption_l4076_407692


namespace NUMINAMATH_CALUDE_max_log_sum_l4076_407613

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∃ (max : ℝ), max = Real.log 4 ∧ ∀ z w : ℝ, z > 0 → w > 0 → z + w = 4 → Real.log z + Real.log w ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l4076_407613


namespace NUMINAMATH_CALUDE_circular_track_length_l4076_407609

theorem circular_track_length :
  ∀ (track_length : ℝ) (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 →
    sally_speed > 0 →
    track_length / 2 - 120 = 120 * sally_speed / brenda_speed →
    track_length / 2 + 40 = (track_length / 2 - 80) * sally_speed / brenda_speed →
    track_length = 480 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_length_l4076_407609


namespace NUMINAMATH_CALUDE_path_width_l4076_407688

theorem path_width (R r : ℝ) (h1 : R > r) (h2 : 2 * π * R - 2 * π * r = 15 * π) : R - r = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_path_width_l4076_407688


namespace NUMINAMATH_CALUDE_prob_through_c_is_three_sevenths_l4076_407686

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points -/
def numPaths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of passing through a point when moving from start to finish -/
def probThroughPoint (start mid finish : Point) : Rat :=
  (numPaths start mid * numPaths mid finish : Rat) / numPaths start finish

theorem prob_through_c_is_three_sevenths : 
  let a := Point.mk 0 0
  let b := Point.mk 4 4
  let c := Point.mk 3 2
  probThroughPoint a c b = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_through_c_is_three_sevenths_l4076_407686


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l4076_407657

/-- A natural number that ends in 5 zeros and has exactly 42 divisors -/
def SpecialNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^5 * k ∧ (Nat.divisors n).card = 42

/-- The theorem stating that there are exactly two distinct natural numbers
    that satisfy the SpecialNumber property, and their sum is 700000 -/
theorem sum_of_special_numbers :
  ∃! (a b : ℕ), a < b ∧ SpecialNumber a ∧ SpecialNumber b ∧ a + b = 700000 := by
  sorry

#check sum_of_special_numbers

end NUMINAMATH_CALUDE_sum_of_special_numbers_l4076_407657


namespace NUMINAMATH_CALUDE_fraction_scaling_l4076_407673

theorem fraction_scaling (x y : ℝ) : 
  (5*x - 5*(5*y)) / ((5*x)^2 + (5*y)^2) = (1/5) * ((x - 5*y) / (x^2 + y^2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_scaling_l4076_407673


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l4076_407652

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (16 : ℚ) / 100 * total_students = (25 : ℚ) / 100 * camping_students)
  (h2 : (75 : ℚ) / 100 * camping_students + (16 : ℚ) / 100 * total_students = camping_students)
  (camping_students : ℕ) :
  (camping_students : ℚ) / total_students = 64 / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_camping_trip_percentage_l4076_407652


namespace NUMINAMATH_CALUDE_line_x_intercept_l4076_407679

/-- A straight line passing through two points (2, -3) and (6, 5) has an x-intercept of 7/2 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ),
  (f 2 = -3) →
  (f 6 = 5) →
  (∀ x y : ℝ, f y - f x = (y - x) * ((5 - (-3)) / (6 - 2))) →
  (∃ x : ℝ, f x = 0 ∧ x = 7/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l4076_407679


namespace NUMINAMATH_CALUDE_room_tiling_l4076_407685

-- Define the room dimensions in centimeters
def room_length : ℕ := 544
def room_width : ℕ := 374

-- Define the function to calculate the least number of square tiles
def least_number_of_tiles (length width : ℕ) : ℕ :=
  let tile_size := Nat.gcd length width
  (length / tile_size) * (width / tile_size)

-- Theorem statement
theorem room_tiling :
  least_number_of_tiles room_length room_width = 176 := by
  sorry

end NUMINAMATH_CALUDE_room_tiling_l4076_407685


namespace NUMINAMATH_CALUDE_triangle_side_count_l4076_407614

theorem triangle_side_count : ∃! n : ℕ, n = (Finset.filter (fun x => x > 3 ∧ x < 11) (Finset.range 11)).card := by sorry

end NUMINAMATH_CALUDE_triangle_side_count_l4076_407614


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l4076_407690

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * (Real.exp x + x + 2)

theorem tangent_line_and_inequality 
  (a b : ℝ) 
  (h1 : f a b 0 = 0) 
  (h2 : (deriv (f a b)) 0 = 6) :
  (a = 2 ∧ b = 0) ∧ 
  ∀ x > 0, f 2 0 x > 2 * Real.log x + 2 * x + 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l4076_407690


namespace NUMINAMATH_CALUDE_worker_y_defective_rate_l4076_407621

-- Define the fractions and percentages
def worker_x_fraction : ℝ := 1 - 0.1666666666666668
def worker_y_fraction : ℝ := 0.1666666666666668
def worker_x_defective_rate : ℝ := 0.005
def total_defective_rate : ℝ := 0.0055

-- Theorem statement
theorem worker_y_defective_rate :
  ∃ (y_rate : ℝ),
    y_rate = 0.008 ∧
    total_defective_rate = worker_x_fraction * worker_x_defective_rate + worker_y_fraction * y_rate :=
by sorry

end NUMINAMATH_CALUDE_worker_y_defective_rate_l4076_407621


namespace NUMINAMATH_CALUDE_square_difference_given_linear_equations_l4076_407697

theorem square_difference_given_linear_equations (x y : ℝ) :
  (3 * x + 2 * y = 30) → (4 * x + 2 * y = 34) → x^2 - y^2 = -65 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_linear_equations_l4076_407697


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l4076_407617

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l4076_407617


namespace NUMINAMATH_CALUDE_cost_calculation_l4076_407639

/-- The cost of items given their quantities and price ratios -/
def cost_of_items (pen_price pencil_price eraser_price : ℚ) : ℚ :=
  4 * pen_price + 6 * pencil_price + 2 * eraser_price

/-- The cost of a dozen pens and half a dozen erasers -/
def cost_of_dozen_pens_and_half_dozen_erasers (pen_price eraser_price : ℚ) : ℚ :=
  12 * pen_price + 6 * eraser_price

theorem cost_calculation :
  ∀ (x : ℚ),
    cost_of_items (4*x) (2*x) x = 360 →
    cost_of_dozen_pens_and_half_dozen_erasers (4*x) x = 648 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_calculation_l4076_407639


namespace NUMINAMATH_CALUDE_cranberries_left_l4076_407624

/-- The number of cranberries left in a bog after harvesting and elk consumption -/
theorem cranberries_left (total : ℕ) (harvest_percent : ℚ) (elk_eaten : ℕ) 
  (h1 : total = 60000)
  (h2 : harvest_percent = 40 / 100)
  (h3 : elk_eaten = 20000) :
  total - (total * harvest_percent).floor - elk_eaten = 16000 := by
  sorry

#check cranberries_left

end NUMINAMATH_CALUDE_cranberries_left_l4076_407624


namespace NUMINAMATH_CALUDE_investment_interest_rate_l4076_407606

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (investment_at_r : ℝ) 
  (total_interest : ℝ) 
  (known_rate : ℝ) :
  total_investment = 10000 →
  investment_at_r = 7200 →
  known_rate = 0.09 →
  total_interest = 684 →
  ∃ r : ℝ, 
    r * investment_at_r + known_rate * (total_investment - investment_at_r) = total_interest ∧
    r = 0.06 :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l4076_407606


namespace NUMINAMATH_CALUDE_corn_acreage_l4076_407674

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l4076_407674


namespace NUMINAMATH_CALUDE_chord_length_polar_l4076_407658

/-- The length of the chord intercepted by a line on a circle in polar coordinates -/
theorem chord_length_polar (r : ℝ) (h : r > 0) :
  let line := {θ : ℝ | r * (Real.sin θ + Real.cos θ) = 2 * Real.sqrt 2}
  let circle := {ρ : ℝ | ρ = 2 * Real.sqrt 2}
  let chord_length := 2 * Real.sqrt ((2 * Real.sqrt 2)^2 - (2 * Real.sqrt 2 / Real.sqrt 2)^2)
  chord_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l4076_407658


namespace NUMINAMATH_CALUDE_problem_statement_l4076_407670

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 + a*b = c) 
  (h2 : a*b + b^2 = c + 5) : 
  (2*c + 5 ≥ 0) ∧ 
  (a^2 - b^2 = -5) ∧ 
  (a ≠ b ∧ a ≠ -b) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4076_407670


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_one_l4076_407628

open Complex

theorem modulus_of_z_equals_one (z : ℂ) (h : z * (1 + I) = 1 - I) : abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_one_l4076_407628


namespace NUMINAMATH_CALUDE_misha_phone_number_l4076_407605

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

def is_consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0))

theorem misha_phone_number :
  ∃! n : ℕ,
    n ≥ 1000000 ∧ n < 10000000 ∧
    is_palindrome (n / 100) ∧
    is_consecutive (n % 10) ((n / 10) % 10) ((n / 100) % 10) ∧
    (n / 10000) % 9 = 0 ∧
    ∃ i : ℕ, i < 5 → (n / (10^i)) % 1000 = 111 ∧
    (is_prime ((n / 100) % 100) ∨ is_prime (n % 100)) ∧
    n = 7111765 :=
  sorry

end NUMINAMATH_CALUDE_misha_phone_number_l4076_407605


namespace NUMINAMATH_CALUDE_gcd_problem_l4076_407677

theorem gcd_problem :
  ∃! n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.gcd n 15 = 5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l4076_407677


namespace NUMINAMATH_CALUDE_carnations_in_third_bouquet_l4076_407641

theorem carnations_in_third_bouquet 
  (total_bouquets : ℕ)
  (first_bouquet : ℕ)
  (second_bouquet : ℕ)
  (average_carnations : ℕ)
  (h1 : total_bouquets = 3)
  (h2 : first_bouquet = 9)
  (h3 : second_bouquet = 14)
  (h4 : average_carnations = 12) :
  average_carnations * total_bouquets - (first_bouquet + second_bouquet) = 13 :=
by
  sorry

#check carnations_in_third_bouquet

end NUMINAMATH_CALUDE_carnations_in_third_bouquet_l4076_407641


namespace NUMINAMATH_CALUDE_line_equation_l4076_407607

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define when a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem line_equation (l : Line2D) :
  (pointOnLine ⟨0, 0⟩ l) →
  (perpendicular l ⟨1, -1, -3⟩) →
  l = ⟨1, 1, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l4076_407607


namespace NUMINAMATH_CALUDE_marlas_driving_time_l4076_407656

/-- The time Marla spends driving one way to her son's school -/
def driving_time : ℕ := sorry

/-- The total time Marla spends on the errand -/
def total_time : ℕ := 110

/-- The time Marla spends at parent-teacher night -/
def parent_teacher_time : ℕ := 70

/-- Theorem stating that the driving time is 20 minutes -/
theorem marlas_driving_time : driving_time = 20 :=
by
  have h1 : total_time = driving_time + parent_teacher_time + driving_time :=
    sorry
  sorry

end NUMINAMATH_CALUDE_marlas_driving_time_l4076_407656


namespace NUMINAMATH_CALUDE_inverse_proportion_function_l4076_407667

/-- Given that y is inversely proportional to x and y = 1 when x = 2,
    prove that the function expression of y with respect to x is y = 2/x. -/
theorem inverse_proportion_function (x : ℝ) (y : ℝ → ℝ) (k : ℝ) :
  (∀ x ≠ 0, y x = k / x) →  -- y is inversely proportional to x
  y 2 = 1 →                 -- when x = 2, y = 1
  ∀ x ≠ 0, y x = 2 / x :=   -- the function expression is y = 2/x
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_function_l4076_407667


namespace NUMINAMATH_CALUDE_rain_probability_l4076_407638

theorem rain_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l4076_407638


namespace NUMINAMATH_CALUDE_carnation_fraction_l4076_407660

/-- Represents a flower bouquet with pink and red roses and carnations -/
structure Bouquet where
  pink_roses : ℚ
  red_roses : ℚ
  pink_carnations : ℚ
  red_carnations : ℚ

/-- The fraction of carnations in the bouquet is 7/10 -/
theorem carnation_fraction (b : Bouquet) : 
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations = 1 →
  b.pink_roses + b.pink_carnations = 6/10 →
  b.pink_roses = 1/3 * (b.pink_roses + b.pink_carnations) →
  b.red_carnations = 3/4 * (b.red_roses + b.red_carnations) →
  b.pink_carnations + b.red_carnations = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_carnation_fraction_l4076_407660


namespace NUMINAMATH_CALUDE_square_of_1009_l4076_407646

theorem square_of_1009 : 1009 ^ 2 = 1018081 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1009_l4076_407646


namespace NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l4076_407655

open Real

noncomputable def f (x : ℝ) : ℝ := tan (x - π / 4)

theorem tan_strictly_increasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Ioo (k * π - π / 4) (k * π + 3 * π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l4076_407655


namespace NUMINAMATH_CALUDE_log_inequality_equivalence_l4076_407680

-- Define the logarithm with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_inequality_equivalence :
  ∀ x : ℝ, log_one_third (2*x - 1) > 1 ↔ 1/2 < x ∧ x < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_equivalence_l4076_407680


namespace NUMINAMATH_CALUDE_widget_production_difference_l4076_407671

/-- Given David's widget production scenario, this theorem proves the difference
    in production between two consecutive days. -/
theorem widget_production_difference
  (t : ℕ) -- Number of hours worked on the first day
  (w : ℕ) -- Number of widgets produced per hour on the first day
  (h1 : w = 2 * t^2) -- Relation between w and t
  : w * t - (w + 3) * (t - 3) = 6 * t^2 - 3 * t + 9 :=
by sorry

end NUMINAMATH_CALUDE_widget_production_difference_l4076_407671


namespace NUMINAMATH_CALUDE_sequence_property_l4076_407654

theorem sequence_property (n : ℕ) (x : ℕ → ℚ) (h_n : n ≥ 7) 
  (h_def : ∀ k > 1, x k = 1 / (1 - x (k-1)))
  (h_x2 : x 2 = 5) : 
  x 7 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l4076_407654


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l4076_407632

theorem min_value_expression (x : ℝ) (h : x > 0) : x^2 + 8*x + 64/x^3 ≥ 28 := by
  sorry

theorem equality_condition : ∃ x > 0, x^2 + 8*x + 64/x^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l4076_407632


namespace NUMINAMATH_CALUDE_rectangle_properties_l4076_407683

structure Quadrilateral where
  isRectangle : Bool
  diagonalsEqual : Bool
  diagonalsBisect : Bool

theorem rectangle_properties (q : Quadrilateral) :
  (q.isRectangle → q.diagonalsEqual ∧ q.diagonalsBisect) ∧
  (q.diagonalsEqual ∧ q.diagonalsBisect → q.isRectangle) ∧
  (¬q.isRectangle → ¬q.diagonalsEqual ∨ ¬q.diagonalsBisect) ∧
  (¬q.diagonalsEqual ∨ ¬q.diagonalsBisect → ¬q.isRectangle) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_properties_l4076_407683


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l4076_407689

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance c,
    prove that if the point symmetric to the focus with respect to y = (b/c)x lies on the ellipse,
    then the eccentricity is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let e := c / a
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let focus := (c, 0)
  let symmetry_line := fun (x : ℝ) ↦ (b / c) * x
  let Q := (
    let m := (c^3 - c*b^2) / a^2
    let n := 2*b*c^2 / a^2
    (m, n)
  )
  (ellipse Q.1 Q.2) → e = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l4076_407689


namespace NUMINAMATH_CALUDE_total_chocolates_in_large_box_l4076_407629

/-- Represents the number of small boxes in the large box -/
def num_small_boxes : ℕ := 19

/-- Represents the number of chocolate bars in each small box -/
def chocolates_per_small_box : ℕ := 25

/-- Theorem stating that the total number of chocolate bars in the large box is 475 -/
theorem total_chocolates_in_large_box : 
  num_small_boxes * chocolates_per_small_box = 475 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolates_in_large_box_l4076_407629


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_with_opposite_numbers_l4076_407684

theorem solution_to_linear_equation_with_opposite_numbers :
  ∃ (x y : ℝ), 2 * x + 3 * y - 4 = 0 ∧ x = -y ∧ x = -4 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_with_opposite_numbers_l4076_407684


namespace NUMINAMATH_CALUDE_fixed_point_values_l4076_407602

/-- A function has exactly one fixed point if and only if
    the equation f(x) = x has exactly one solution. -/
def has_exactly_one_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = x

/-- The quadratic function f(x) = ax² + (2a-3)x + 1 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + (2*a - 3) * x + 1

/-- The set of values for a such that f has exactly one fixed point -/
def A : Set ℝ := {a | has_exactly_one_fixed_point (f a)}

theorem fixed_point_values :
  A = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_fixed_point_values_l4076_407602


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4076_407612

/-- A geometric sequence with a positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 6 = 8 * a 4 →
  a 2 = 2 →
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l4076_407612


namespace NUMINAMATH_CALUDE_distance_to_y_axis_angle_bisector_and_line_l4076_407623

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2*m)

-- Part 1
theorem distance_to_y_axis (m : ℝ) :
  abs (2 - m) = 2 → (M m = (2, 1) ∨ M m = (-2, 9)) :=
sorry

-- Part 2
theorem angle_bisector_and_line (m k b : ℝ) :
  (2 - m = 1 + 2*m) →  -- M lies on angle bisector
  ((2 - m) = k*(2 - m) + b) →  -- Line passes through M
  (0 = k*0 + b) →  -- Line passes through (0,5)
  (5 = k*0 + b) →
  (k = -2 ∧ b = 5) :=  -- Line equation is y = -2x + 5
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_angle_bisector_and_line_l4076_407623


namespace NUMINAMATH_CALUDE_cotton_candy_to_candy_bar_ratio_l4076_407630

/-- The price of candy bars, caramel, and cotton candy -/
structure CandyPrices where
  caramel : ℝ
  candy_bar : ℝ
  cotton_candy : ℝ

/-- The conditions of the candy pricing problem -/
def candy_pricing_conditions (p : CandyPrices) : Prop :=
  p.candy_bar = 2 * p.caramel ∧
  p.caramel = 3 ∧
  6 * p.candy_bar + 3 * p.caramel + p.cotton_candy = 57

/-- The theorem stating the ratio of cotton candy price to 4 candy bars -/
theorem cotton_candy_to_candy_bar_ratio (p : CandyPrices) 
  (h : candy_pricing_conditions p) : 
  p.cotton_candy / (4 * p.candy_bar) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cotton_candy_to_candy_bar_ratio_l4076_407630


namespace NUMINAMATH_CALUDE_zigzag_angle_l4076_407651

theorem zigzag_angle (ACB FEG DCE DEC : Real) (h1 : ACB = 80)
  (h2 : FEG = 64) (h3 : DCE + 80 + 14 = 180) (h4 : DEC + 64 + 33 = 180) :
  180 - DCE - DEC = 11 := by
  sorry

end NUMINAMATH_CALUDE_zigzag_angle_l4076_407651


namespace NUMINAMATH_CALUDE_percentage_green_shirts_l4076_407668

theorem percentage_green_shirts (total_students : ℕ) (blue_percent red_percent : ℚ) (other_students : ℕ) :
  total_students = 600 →
  blue_percent = 45/100 →
  red_percent = 23/100 →
  other_students = 102 →
  (total_students - (blue_percent * total_students + red_percent * total_students + other_students)) / total_students = 15/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_green_shirts_l4076_407668


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4076_407666

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 6 → 
    b = 8 → 
    c^2 = a^2 + b^2 → 
    c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4076_407666


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4076_407643

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 + 2^3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4076_407643


namespace NUMINAMATH_CALUDE_journey_speed_proof_l4076_407626

/-- Proves that given a journey of 120 miles in 90 minutes, where the average speed
    for the first 30 minutes was 70 mph and for the second 30 minutes was 75 mph,
    the average speed for the last 30 minutes must be 95 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
    (speed1 : ℝ) (speed2 : ℝ) (time_segment : ℝ) :
    total_distance = 120 →
    total_time = 1.5 →
    speed1 = 70 →
    speed2 = 75 →
    time_segment = 0.5 →
    speed1 * time_segment + speed2 * time_segment + 
    ((total_distance - (speed1 * time_segment + speed2 * time_segment)) / time_segment) = 
    total_distance / total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l4076_407626


namespace NUMINAMATH_CALUDE_canvas_area_l4076_407637

/-- The area of a rectangular canvas inside a decorative border -/
theorem canvas_area (outer_width outer_height border_width : ℝ) : 
  outer_width = 100 →
  outer_height = 140 →
  border_width = 15 →
  (outer_width - 2 * border_width) * (outer_height - 2 * border_width) = 7700 := by
sorry

end NUMINAMATH_CALUDE_canvas_area_l4076_407637


namespace NUMINAMATH_CALUDE_R_squared_eq_one_when_no_error_l4076_407647

/-- A structure representing a set of observations in a linear regression model. -/
structure LinearRegressionData (n : ℕ) where
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  e : Fin n → ℝ

/-- The coefficient of determination (R-squared) for a linear regression model. -/
def R_squared (data : LinearRegressionData n) : ℝ :=
  sorry

/-- Theorem stating that if all error terms are zero, then R-squared equals 1. -/
theorem R_squared_eq_one_when_no_error (n : ℕ) (data : LinearRegressionData n)
  (h1 : ∀ i, data.y i = data.b * data.x i + data.a + data.e i)
  (h2 : ∀ i, data.e i = 0) :
  R_squared data = 1 :=
sorry

end NUMINAMATH_CALUDE_R_squared_eq_one_when_no_error_l4076_407647


namespace NUMINAMATH_CALUDE_floor_of_4_7_l4076_407603

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l4076_407603


namespace NUMINAMATH_CALUDE_plot_length_is_60_l4076_407618

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMetre : ℝ
  totalFencingCost : ℝ

/-- The length of the plot is 20 metres more than its breadth. -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 20

/-- The cost of fencing the plot at the given rate equals the total fencing cost. -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMetre * (2 * plot.length + 2 * plot.breadth) = plot.totalFencingCost

/-- The theorem stating that under the given conditions, the length of the plot is 60 metres. -/
theorem plot_length_is_60 (plot : RectangularPlot)
    (h1 : lengthCondition plot)
    (h2 : fencingCostCondition plot)
    (h3 : plot.fencingCostPerMetre = 26.5)
    (h4 : plot.totalFencingCost = 5300) :
    plot.length = 60 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_60_l4076_407618


namespace NUMINAMATH_CALUDE_total_paintable_area_is_876_l4076_407633

/-- The number of bedrooms in Isabella's house -/
def num_bedrooms : ℕ := 3

/-- The length of each bedroom in feet -/
def bedroom_length : ℕ := 12

/-- The width of each bedroom in feet -/
def bedroom_width : ℕ := 10

/-- The height of each bedroom in feet -/
def bedroom_height : ℕ := 8

/-- The area occupied by doorways and windows in each bedroom in square feet -/
def unpaintable_area : ℕ := 60

/-- The total area of walls to be painted in all bedrooms -/
def total_paintable_area : ℕ :=
  num_bedrooms * (
    2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area
  )

/-- Theorem stating that the total area to be painted is 876 square feet -/
theorem total_paintable_area_is_876 : total_paintable_area = 876 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_876_l4076_407633


namespace NUMINAMATH_CALUDE_simplify_fraction_l4076_407644

theorem simplify_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(3/2) * b^(5/2)) / (a*b)^(1/2) = a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4076_407644


namespace NUMINAMATH_CALUDE_triangle_area_equalities_l4076_407675

theorem triangle_area_equalities (S r R A B C : ℝ) 
  (h_positive : S > 0 ∧ r > 0 ∧ R > 0)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_area : S = r * R * (Real.sin A + Real.sin B + Real.sin C)) :
  S = r * R * (Real.sin A + Real.sin B + Real.sin C) ∧
  S = 4 * r * R * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) ∧
  S = (R^2 / 2) * (Real.sin (2*A) + Real.sin (2*B) + Real.sin (2*C)) ∧
  S = 2 * R^2 * Real.sin A * Real.sin B * Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_equalities_l4076_407675


namespace NUMINAMATH_CALUDE_rectangle_width_is_five_l4076_407642

/-- A rectangle with specific properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_longer : width = length + 2
  perimeter : length * 2 + width * 2 = 16

/-- The width of the rectangle is 5 -/
theorem rectangle_width_is_five (r : Rectangle) : r.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_is_five_l4076_407642


namespace NUMINAMATH_CALUDE_inequalities_comparison_l4076_407698

theorem inequalities_comparison (a b : ℝ) : 
  (∃ a b : ℝ, a + b < 2) ∧ 
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b) ∧ 
  (∀ a b : ℝ, a*b ≤ ((a + b)/2)^2) ∧ 
  (∀ a b : ℝ, |a| + |b| ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_comparison_l4076_407698


namespace NUMINAMATH_CALUDE_fred_book_purchase_l4076_407636

theorem fred_book_purchase (initial_amount remaining_amount cost_per_book : ℕ) 
  (h1 : initial_amount = 236)
  (h2 : remaining_amount = 14)
  (h3 : cost_per_book = 37) :
  (initial_amount - remaining_amount) / cost_per_book = 6 := by
  sorry

end NUMINAMATH_CALUDE_fred_book_purchase_l4076_407636


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l4076_407631

theorem ticket_price_possibilities (x : ℕ) : 
  (∃ n m : ℕ, n * x = 72 ∧ m * x = 108 ∧ Even x) ↔ 
  x ∈ ({2, 4, 6, 12, 18, 36} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l4076_407631


namespace NUMINAMATH_CALUDE_max_distance_sparkling_points_l4076_407625

theorem max_distance_sparkling_points :
  ∀ (a₁ b₁ a₂ b₂ : ℝ),
    a₁^2 + b₁^2 = 1 →
    a₂^2 + b₂^2 = 1 →
    ∀ (d : ℝ),
      d = Real.sqrt ((a₂ - a₁)^2 + (b₂ - b₁)^2) →
      d ≤ 2 ∧ ∃ (a₁' b₁' a₂' b₂' : ℝ),
        a₁'^2 + b₁'^2 = 1 ∧
        a₂'^2 + b₂'^2 = 1 ∧
        Real.sqrt ((a₂' - a₁')^2 + (b₂' - b₁')^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_sparkling_points_l4076_407625


namespace NUMINAMATH_CALUDE_soda_difference_l4076_407645

/-- The number of liters of soda in each bottle -/
def liters_per_bottle : ℕ := 2

/-- The number of orange soda bottles Julio has -/
def julio_orange : ℕ := 4

/-- The number of grape soda bottles Julio has -/
def julio_grape : ℕ := 7

/-- The number of orange soda bottles Mateo has -/
def mateo_orange : ℕ := 1

/-- The number of grape soda bottles Mateo has -/
def mateo_grape : ℕ := 3

/-- The difference in total liters of soda between Julio and Mateo -/
theorem soda_difference : 
  (julio_orange + julio_grape) * liters_per_bottle - 
  (mateo_orange + mateo_grape) * liters_per_bottle = 14 := by
sorry

end NUMINAMATH_CALUDE_soda_difference_l4076_407645


namespace NUMINAMATH_CALUDE_sum_powers_l4076_407616

theorem sum_powers (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_powers_l4076_407616


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l4076_407648

theorem lcm_gcd_product (a b : ℕ) (ha : a = 240) (hb : b = 360) :
  Nat.lcm a b * Nat.gcd a b = 17280 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l4076_407648


namespace NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l4076_407640

/-- Proves that a train with given length, crossing its own length in a certain time,
    takes the calculated time to cross a platform of given length. -/
theorem train_crossing_time (train_length platform_length cross_own_length_time : ℝ) 
    (train_length_pos : 0 < train_length)
    (platform_length_pos : 0 < platform_length)
    (cross_own_length_time_pos : 0 < cross_own_length_time) :
  let train_speed := train_length / cross_own_length_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 45 :=
by
  sorry

/-- Specific instance of the train crossing problem -/
theorem specific_train_crossing_time :
  let train_length := 300
  let platform_length := 450
  let cross_own_length_time := 18
  let train_speed := train_length / cross_own_length_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l4076_407640


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_dividing_powers_l4076_407600

theorem infinitely_many_pairs_dividing_powers (d : ℤ) 
  (h1 : d > 1) (h2 : d % 4 = 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (a + b : ℤ) ∣ (a^b + b^a : ℤ) := by
  sorry

#check infinitely_many_pairs_dividing_powers

end NUMINAMATH_CALUDE_infinitely_many_pairs_dividing_powers_l4076_407600


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l4076_407663

theorem three_digit_number_problem (a b c d e f : ℕ) :
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) ∧
  (100 ≤ 100 * d + 10 * e + f) ∧ (100 * d + 10 * e + f < 1000) ∧
  (a = b + 1) ∧ (b = c + 2) ∧
  ((100 * a + 10 * b + c) * 3 + 4 = 100 * d + 10 * e + f) →
  100 * d + 10 * e + f = 964 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_number_problem_l4076_407663


namespace NUMINAMATH_CALUDE_correct_answer_after_resolving_errors_l4076_407694

theorem correct_answer_after_resolving_errors 
  (incorrect_divisor : ℝ)
  (correct_divisor : ℝ)
  (incorrect_answer : ℝ)
  (subtracted_value : ℝ)
  (should_add_value : ℝ)
  (h1 : incorrect_divisor = 63.5)
  (h2 : correct_divisor = 36.2)
  (h3 : incorrect_answer = 24)
  (h4 : subtracted_value = 12)
  (h5 : should_add_value = 8) :
  ∃ (correct_answer : ℝ), abs (correct_answer - 42.98) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_correct_answer_after_resolving_errors_l4076_407694


namespace NUMINAMATH_CALUDE_childrens_buffet_price_l4076_407604

def adult_price : ℚ := 30
def senior_discount : ℚ := 1/10
def num_adults : ℕ := 2
def num_seniors : ℕ := 2
def num_children : ℕ := 3
def total_spent : ℚ := 159

theorem childrens_buffet_price :
  ∃ (child_price : ℚ),
    child_price * num_children +
    adult_price * num_adults +
    adult_price * (1 - senior_discount) * num_seniors = total_spent ∧
    child_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_childrens_buffet_price_l4076_407604


namespace NUMINAMATH_CALUDE_apartment_units_per_floor_l4076_407627

theorem apartment_units_per_floor (total_units : ℕ) (first_floor_units : ℕ) (num_buildings : ℕ) (num_floors : ℕ) :
  total_units = 34 →
  first_floor_units = 2 →
  num_buildings = 2 →
  num_floors = 4 →
  ∃ (other_floor_units : ℕ),
    total_units = num_buildings * (first_floor_units + (num_floors - 1) * other_floor_units) ∧
    other_floor_units = 5 :=
by sorry

end NUMINAMATH_CALUDE_apartment_units_per_floor_l4076_407627


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4076_407672

theorem x_plus_y_value (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 2) (h3 : x * y < 0) :
  x + y = 1 ∨ x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4076_407672


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l4076_407608

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3)

theorem monotonic_decreasing_interval_of_f :
  ∀ x₁ x₂, x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ > f x₂ :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l4076_407608
