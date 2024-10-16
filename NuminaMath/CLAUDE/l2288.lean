import Mathlib

namespace NUMINAMATH_CALUDE_swimmers_speed_l2288_228850

/-- The speed of a swimmer in still water, given:
  1. The speed of the water current is 2 km/h.
  2. The swimmer takes 1.5 hours to swim 3 km against the current. -/
theorem swimmers_speed (current_speed : ℝ) (swim_time : ℝ) (swim_distance : ℝ) 
  (h1 : current_speed = 2)
  (h2 : swim_time = 1.5)
  (h3 : swim_distance = 3)
  (h4 : swim_distance = (swimmer_speed - current_speed) * swim_time) :
  swimmer_speed = 4 :=
by
  sorry

#check swimmers_speed

end NUMINAMATH_CALUDE_swimmers_speed_l2288_228850


namespace NUMINAMATH_CALUDE_range_of_m_l2288_228855

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2288_228855


namespace NUMINAMATH_CALUDE_xyz_sum_eq_32_l2288_228803

theorem xyz_sum_eq_32 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 48)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 64) :
  x*y + y*z + x*z = 32 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_eq_32_l2288_228803


namespace NUMINAMATH_CALUDE_measure_one_kg_l2288_228852

theorem measure_one_kg (n : ℕ) (h : ¬ 3 ∣ n) : 
  ∃ (k : ℕ), n - 3 * k = 1 ∨ n - 3 * k = 2 :=
sorry

end NUMINAMATH_CALUDE_measure_one_kg_l2288_228852


namespace NUMINAMATH_CALUDE_determinant_scaling_l2288_228884

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 7 →
  Matrix.det ![![3*a, 3*b], ![3*c, 3*d]] = 63 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l2288_228884


namespace NUMINAMATH_CALUDE_function_characterization_l2288_228863

/-- A function is strictly increasing if for all x < y, f(x) < f(y) -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- g is the composition inverse of f if f(g(x)) = x and g(f(x)) = x for all real x -/
def CompositionInverse (f g : ℝ → ℝ) : Prop :=
  (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x)

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : StrictlyIncreasing f)
  (h2 : ∃ g : ℝ → ℝ, CompositionInverse f g ∧ ∀ x, f x + g x = 2 * x) :
  ∃ c : ℝ, ∀ x, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2288_228863


namespace NUMINAMATH_CALUDE_dollar_square_sum_l2288_228872

/-- Custom operation ▩ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating that (x + y)² ▩ (y² + x²) = 4x²y² -/
theorem dollar_square_sum (x y : ℝ) : dollar ((x + y)^2) (y^2 + x^2) = 4 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_dollar_square_sum_l2288_228872


namespace NUMINAMATH_CALUDE_find_divisor_l2288_228826

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 689)
  (h2 : quotient = 19)
  (h3 : remainder = 5)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) :
  dividend / quotient = 36 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2288_228826


namespace NUMINAMATH_CALUDE_ten_machines_four_minutes_l2288_228843

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute := (420 * machines) / 6
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines produce 2800 bottles in 4 minutes -/
theorem ten_machines_four_minutes :
  bottles_produced 10 4 = 2800 := by
  sorry

#eval bottles_produced 10 4

end NUMINAMATH_CALUDE_ten_machines_four_minutes_l2288_228843


namespace NUMINAMATH_CALUDE_tom_jogging_distance_l2288_228808

/-- Tom's jogging rate in miles per minute -/
def jogging_rate : ℚ := 1 / 15

/-- Time Tom jogs in minutes -/
def jogging_time : ℚ := 45

/-- Distance Tom jogs in miles -/
def jogging_distance : ℚ := jogging_rate * jogging_time

theorem tom_jogging_distance :
  jogging_distance = 3 := by sorry

end NUMINAMATH_CALUDE_tom_jogging_distance_l2288_228808


namespace NUMINAMATH_CALUDE_distance_between_points_on_lines_l2288_228880

/-- The distance between two points on different lines with a given midpoint. -/
theorem distance_between_points_on_lines (xP yP xQ yQ : ℝ) :
  -- P is on the line 6y = 17x
  6 * yP = 17 * xP →
  -- Q is on the line 8y = 5x
  8 * yQ = 5 * xQ →
  -- (10, 5) is the midpoint of PQ
  (xP + xQ) / 2 = 10 →
  (yP + yQ) / 2 = 5 →
  -- The distance formula
  let distance := Real.sqrt ((xP - xQ)^2 + (yP - yQ)^2)
  -- The distance is equal to some real value (which we don't specify)
  ∃ (d : ℝ), distance = d :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_lines_l2288_228880


namespace NUMINAMATH_CALUDE_finite_painted_blocks_l2288_228816

theorem finite_painted_blocks : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    ∀ m n r : ℕ, 
      m * n * r = 2 * (m - 2) * (n - 2) * (r - 2) → 
      (m, n, r) ∈ S := by
sorry

end NUMINAMATH_CALUDE_finite_painted_blocks_l2288_228816


namespace NUMINAMATH_CALUDE_table_area_is_128_l2288_228889

/-- A rectangular table with one side against a wall -/
structure RectangularTable where
  -- Length of the side opposite the wall
  opposite_side : ℝ
  -- Length of each of the other two free sides
  other_side : ℝ
  -- The side opposite the wall is twice the length of each of the other two free sides
  opposite_twice_other : opposite_side = 2 * other_side
  -- The total length of the table's free sides is 32 feet
  total_free_sides : opposite_side + 2 * other_side = 32

/-- The area of the rectangular table is 128 square feet -/
theorem table_area_is_128 (table : RectangularTable) : table.opposite_side * table.other_side = 128 := by
  sorry

end NUMINAMATH_CALUDE_table_area_is_128_l2288_228889


namespace NUMINAMATH_CALUDE_exact_three_correct_deliveries_probability_l2288_228815

def num_packages : ℕ := 5

def num_correct_deliveries : ℕ := 3

def total_permutations : ℕ := num_packages.factorial

def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

def num_ways_correct_deliveries : ℕ := choose num_packages num_correct_deliveries

def num_derangements_remaining : ℕ := 1

theorem exact_three_correct_deliveries_probability :
  (num_ways_correct_deliveries * num_derangements_remaining : ℚ) / total_permutations = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_correct_deliveries_probability_l2288_228815


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_subset_implies_a_range_l2288_228820

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | -a < x ∧ x < a + 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | 4 - x < a}

-- Part 1
theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = Set.Ioo 2 3 → a = 2 := by sorry

-- Part 2
theorem subset_implies_a_range (a : ℝ) :
  A a ⊆ (Set.univ \ B a) → a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_subset_implies_a_range_l2288_228820


namespace NUMINAMATH_CALUDE_prime_sum_gcd_ratio_composite_sum_gcd_ratio_l2288_228881

-- Part 1
theorem prime_sum_gcd_ratio (n : ℕ) (hn : Nat.Prime (2 * n - 1)) :
  ∀ (a : Fin n → ℕ), Function.Injective a →
  ∃ i j : Fin n, (a i + a j : ℚ) / Nat.gcd (a i) (a j) ≥ 2 * n - 1 := by sorry

-- Part 2
theorem composite_sum_gcd_ratio (n : ℕ) (hn : ¬Nat.Prime (2 * n - 1)) (hn2 : 2 * n - 1 > 1) :
  ∃ (a : Fin n → ℕ), Function.Injective a ∧
  ∀ i j : Fin n, (a i + a j : ℚ) / Nat.gcd (a i) (a j) < 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_prime_sum_gcd_ratio_composite_sum_gcd_ratio_l2288_228881


namespace NUMINAMATH_CALUDE_initial_men_count_l2288_228878

/-- Represents the work completion scenario in a garment industry -/
structure WorkScenario where
  men : ℕ
  hours_per_day : ℕ
  days : ℕ

/-- Calculates the total man-hours for a given work scenario -/
def total_man_hours (scenario : WorkScenario) : ℕ :=
  scenario.men * scenario.hours_per_day * scenario.days

/-- The initial work scenario -/
def initial_scenario (initial_men : ℕ) : WorkScenario :=
  { men := initial_men, hours_per_day := 8, days := 10 }

/-- The second work scenario -/
def second_scenario : WorkScenario :=
  { men := 8, hours_per_day := 15, days := 8 }

/-- Theorem stating that the initial number of men is 12 -/
theorem initial_men_count : ∃ (initial_men : ℕ), 
  initial_men = 12 ∧ 
  total_man_hours (initial_scenario initial_men) = total_man_hours second_scenario :=
sorry

end NUMINAMATH_CALUDE_initial_men_count_l2288_228878


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l2288_228817

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x) + (0.15 * (x - 0.25 * x)) = 15000 → x = 41379 :=
by sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l2288_228817


namespace NUMINAMATH_CALUDE_fibonacci_m_digit_count_fibonacci_5n_plus_2_digits_l2288_228898

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

theorem fibonacci_m_digit_count (m : ℕ) (h : m ≥ 2) :
  ∃ k : ℕ, fib k ≥ 10^(m-1) ∧ fib (k+3) < 10^m ∧ fib (k+4) ≥ 10^m :=
sorry

theorem fibonacci_5n_plus_2_digits (n : ℕ) :
  fib (5*n + 2) ≥ 10^n :=
sorry

end NUMINAMATH_CALUDE_fibonacci_m_digit_count_fibonacci_5n_plus_2_digits_l2288_228898


namespace NUMINAMATH_CALUDE_factor_expression_l2288_228825

theorem factor_expression (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2288_228825


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l2288_228856

theorem binomial_expansion_terms (x a : ℚ) (n : ℕ) :
  (Nat.choose n 3 : ℚ) * x^(n - 3) * a^3 = 330 ∧
  (Nat.choose n 4 : ℚ) * x^(n - 4) * a^4 = 792 ∧
  (Nat.choose n 5 : ℚ) * x^(n - 5) * a^5 = 1716 →
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l2288_228856


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l2288_228806

-- Define the set S
def S (m l : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ l}

-- Define the property that x^2 ∈ S for all x ∈ S
def closed_under_square (m l : ℝ) :=
  ∀ x, x ∈ S m l → x^2 ∈ S m l

-- Theorem 1
theorem theorem_1 (m l : ℝ) (h : closed_under_square m l) :
  m = 1 → S m l = {1} := by sorry

-- Theorem 2
theorem theorem_2 (m l : ℝ) (h : closed_under_square m l) :
  m = -1/2 → 1/4 ≤ l ∧ l ≤ 1 := by sorry

-- Theorem 3
theorem theorem_3 (m l : ℝ) (h : closed_under_square m l) :
  l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l2288_228806


namespace NUMINAMATH_CALUDE_circle_center_locus_l2288_228882

/-- Given a circle passing through A(0,a) with a chord of length 2a on the x-axis,
    prove that the locus of its center C(x,y) satisfies x^2 = 2ay -/
theorem circle_center_locus (a : ℝ) (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (x^2 + (y - a)^2 = r^2) ∧  -- Circle passes through A(0,a)
    (y^2 + a^2 = r^2))         -- Chord on x-axis has length 2a
  → x^2 = 2*a*y := by sorry

end NUMINAMATH_CALUDE_circle_center_locus_l2288_228882


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2288_228854

/-- Given a quadratic function y = ax² + bx + c, if the points (2, y₁) and (-2, y₂) lie on the curve
    and y₁ - y₂ = 4, then b = 1 -/
theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 4 + b * 2 + c →
  y₂ = a * 4 - b * 2 + c →
  y₁ - y₂ = 4 →
  b = 1 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_coefficient_l2288_228854


namespace NUMINAMATH_CALUDE_sine_matrix_det_zero_l2288_228858

/-- The determinant of a 3x3 matrix with sine entries is zero -/
theorem sine_matrix_det_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  Matrix.det A = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_matrix_det_zero_l2288_228858


namespace NUMINAMATH_CALUDE_optimal_box_height_l2288_228887

/-- Represents the dimensions of an open-top rectangular box with a square base -/
structure BoxDimensions where
  side : ℝ
  height : ℝ

/-- The volume of the box -/
def volume (b : BoxDimensions) : ℝ := b.side^2 * b.height

/-- The surface area of the box -/
def surfaceArea (b : BoxDimensions) : ℝ := b.side^2 + 4 * b.side * b.height

/-- The constraint that the volume must be 4 -/
def volumeConstraint (b : BoxDimensions) : Prop := volume b = 4

theorem optimal_box_height :
  ∃ (b : BoxDimensions), volumeConstraint b ∧
    (∀ (b' : BoxDimensions), volumeConstraint b' → surfaceArea b ≤ surfaceArea b') ∧
    b.height = 1 := by
  sorry

end NUMINAMATH_CALUDE_optimal_box_height_l2288_228887


namespace NUMINAMATH_CALUDE_person_A_savings_l2288_228811

/-- The amount of money saved by person A -/
def savings_A : ℕ := sorry

/-- The amount of money saved by person B -/
def savings_B : ℕ := sorry

/-- The amount of money saved by person C -/
def savings_C : ℕ := sorry

/-- Person A and B together have saved 640 yuan -/
axiom AB_savings : savings_A + savings_B = 640

/-- Person B and C together have saved 600 yuan -/
axiom BC_savings : savings_B + savings_C = 600

/-- Person A and C together have saved 440 yuan -/
axiom AC_savings : savings_A + savings_C = 440

/-- Theorem: Given the conditions, person A has saved 240 yuan -/
theorem person_A_savings : savings_A = 240 :=
  sorry

end NUMINAMATH_CALUDE_person_A_savings_l2288_228811


namespace NUMINAMATH_CALUDE_remainder_properties_l2288_228835

theorem remainder_properties (a b n : ℤ) (hn : n ≠ 0) :
  (((a + b) % n = ((a % n + b % n) % n)) ∧
   ((a - b) % n = ((a % n - b % n) % n)) ∧
   ((a * b) % n = ((a % n * b % n) % n))) := by
  sorry

end NUMINAMATH_CALUDE_remainder_properties_l2288_228835


namespace NUMINAMATH_CALUDE_hyperbola_axis_relation_l2288_228821

-- Define the hyperbola equation
def hyperbola_equation (x y b : ℝ) : Prop := x^2 - y^2 / b^2 = 1

-- Define the length of the conjugate axis
def conjugate_axis_length (b : ℝ) : ℝ := 2 * b

-- Define the length of the transverse axis
def transverse_axis_length : ℝ := 2

-- State the theorem
theorem hyperbola_axis_relation (b : ℝ) :
  b > 0 →
  hyperbola_equation x y b →
  conjugate_axis_length b = 2 * transverse_axis_length →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_axis_relation_l2288_228821


namespace NUMINAMATH_CALUDE_spaceship_speed_halving_l2288_228895

/-- The number of additional people that cause the spaceship's speed to be halved -/
def additional_people : ℕ := sorry

/-- The speed of the spaceship given the number of people on board -/
def speed (people : ℕ) : ℝ := sorry

/-- Theorem: The number of additional people that cause the spaceship's speed to be halved is 100 -/
theorem spaceship_speed_halving :
  (speed 200 = 500) →
  (speed 400 = 125) →
  (∀ n : ℕ, speed (n + additional_people) = (speed n) / 2) →
  additional_people = 100 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_speed_halving_l2288_228895


namespace NUMINAMATH_CALUDE_class_average_mark_l2288_228801

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_average : ℝ) (remaining_average : ℝ) : 
  total_students = 13 →
  excluded_students = 5 →
  excluded_average = 40 →
  remaining_average = 92 →
  (total_students : ℝ) * (total_students * (remaining_average : ℝ) - excluded_students * excluded_average) / 
    (total_students * (total_students - excluded_students)) = 72 := by
  sorry


end NUMINAMATH_CALUDE_class_average_mark_l2288_228801


namespace NUMINAMATH_CALUDE_validSquaresCount_l2288_228831

/-- Represents a square on the checkerboard -/
structure Square :=
  (x : Nat) -- x-coordinate of the top-left corner
  (y : Nat) -- y-coordinate of the top-left corner
  (size : Nat) -- side length of the square

/-- Defines the 10x10 checkerboard -/
def checkerboard : Nat := 10

/-- Checks if a square contains at least 8 black squares -/
def hasAtLeast8BlackSquares (s : Square) : Bool :=
  -- Implementation details omitted
  sorry

/-- Counts the number of valid squares on the checkerboard -/
def countValidSquares : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that there are exactly 115 valid squares -/
theorem validSquaresCount : countValidSquares = 115 := by
  sorry

end NUMINAMATH_CALUDE_validSquaresCount_l2288_228831


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2288_228873

theorem proposition_equivalence (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x > m) ↔ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2288_228873


namespace NUMINAMATH_CALUDE_f_min_max_l2288_228892

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

end NUMINAMATH_CALUDE_f_min_max_l2288_228892


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l2288_228877

theorem modulo_residue_problem :
  (312 + 6 * 51 + 8 * 187 + 5 * 34) % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l2288_228877


namespace NUMINAMATH_CALUDE_shaded_perimeter_is_32_l2288_228814

/-- Given two squares ABCD and BEFG sharing vertex B, with E on BC, G on AB,
    this structure represents the configuration described in the problem. -/
structure SquareConfiguration where
  -- Side length of square ABCD
  x : ℝ
  -- Assertion that E is on BC and G is on AB
  h_e_on_bc : True
  h_g_on_ab : True
  -- Length of CG is 9
  h_cg_length : 9 = 9
  -- Area of shaded region (ABCD - BEFG) is 47
  h_shaded_area : x^2 - (81 - x^2) = 47

/-- Theorem stating that under the given conditions, 
    the perimeter of the shaded region (which is the same as ABCD) is 32. -/
theorem shaded_perimeter_is_32 (config : SquareConfiguration) :
  4 * config.x = 32 := by
  sorry

#check shaded_perimeter_is_32

end NUMINAMATH_CALUDE_shaded_perimeter_is_32_l2288_228814


namespace NUMINAMATH_CALUDE_range_of_a_l2288_228844

-- Define the inequality function
def f (x a : ℝ) : ℝ := x^2 + (2-a)*x + 4-2*a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ≥ 2, f x a > 0) ↔ a < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2288_228844


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2288_228848

theorem multiplication_table_odd_fraction :
  let factors := Finset.range 16
  let table := factors.product factors
  let odd_product (a b : ℕ) := Odd (a * b)
  (table.filter (fun (a, b) => odd_product a b)).card / table.card = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2288_228848


namespace NUMINAMATH_CALUDE_initial_classes_l2288_228839

theorem initial_classes (initial_classes : ℕ) : 
  (20 * initial_classes : ℕ) + (20 * 5 : ℕ) = 400 → initial_classes = 15 :=
by sorry

end NUMINAMATH_CALUDE_initial_classes_l2288_228839


namespace NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l2288_228838

theorem smallest_a_for_nonprime_polynomial : ∃ (a : ℕ), a > 0 ∧
  (∀ (x : ℤ), ¬ Prime (x^4 + a^3)) ∧
  (∀ (b : ℕ), b > 0 ∧ b < a → ∃ (x : ℤ), Prime (x^4 + b^3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l2288_228838


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2288_228875

theorem isosceles_triangle (A B C : Real) (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2288_228875


namespace NUMINAMATH_CALUDE_grass_field_width_l2288_228899

/-- Proves that given a rectangular grass field with length 75 m, surrounded by a 2.5 m wide path,
    if the cost of constructing the path is Rs. 6750 at Rs. 10 per sq m,
    then the width of the grass field is 55 m. -/
theorem grass_field_width (field_length : ℝ) (path_width : ℝ) (path_cost : ℝ) (cost_per_sqm : ℝ) :
  field_length = 75 →
  path_width = 2.5 →
  path_cost = 6750 →
  cost_per_sqm = 10 →
  ∃ (field_width : ℝ),
    field_width = 55 ∧
    path_cost = cost_per_sqm * (
      (field_length + 2 * path_width) * (field_width + 2 * path_width) -
      field_length * field_width
    ) := by
  sorry

end NUMINAMATH_CALUDE_grass_field_width_l2288_228899


namespace NUMINAMATH_CALUDE_knights_count_l2288_228837

/-- Represents the statement made by the i-th person on the island -/
def statement (i : ℕ) (num_knights : ℕ) : Prop :=
  num_knights ∣ i

/-- Represents whether a person at position i is telling the truth -/
def is_truthful (i : ℕ) (num_knights : ℕ) : Prop :=
  statement i num_knights

/-- The total number of inhabitants on the island -/
def total_inhabitants : ℕ := 100

/-- Theorem stating that the only possible numbers of knights are 0 and 10 -/
theorem knights_count : 
  ∃ (num_knights : ℕ), 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ total_inhabitants → 
      (is_truthful i num_knights ↔ i % num_knights = 0)) ∧
    (num_knights = 0 ∨ num_knights = 10) :=
sorry

end NUMINAMATH_CALUDE_knights_count_l2288_228837


namespace NUMINAMATH_CALUDE_common_divisors_9240_10800_l2288_228891

theorem common_divisors_9240_10800 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 10800) (Finset.range 10801)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10800_l2288_228891


namespace NUMINAMATH_CALUDE_horse_distribution_exists_l2288_228840

/-- Represents the distribution of horses to sons -/
structure Distribution (b₁ b₂ b₃ : ℕ) :=
  (x₁₁ x₁₂ x₁₃ : ℕ)
  (x₂₁ x₂₂ x₂₃ : ℕ)
  (x₃₁ x₃₂ x₃₃ : ℕ)
  (sum_eq_b₁ : x₁₁ + x₂₁ + x₃₁ = b₁)
  (sum_eq_b₂ : x₁₂ + x₂₂ + x₃₂ = b₂)
  (sum_eq_b₃ : x₁₃ + x₂₃ + x₃₃ = b₃)

/-- Represents the value matrix for horses -/
def ValueMatrix := Matrix (Fin 3) (Fin 3) ℚ

/-- The theorem statement -/
theorem horse_distribution_exists :
  ∃ n : ℕ, ∀ b₁ b₂ b₃ : ℕ, ∀ A : ValueMatrix,
    (∀ i j : Fin 3, i ≠ j → A i i > A i j) →
    min b₁ (min b₂ b₃) > n →
    ∃ d : Distribution b₁ b₂ b₃,
      (A 0 0 * d.x₁₁ + A 0 1 * d.x₁₂ + A 0 2 * d.x₁₃ > A 0 0 * d.x₂₁ + A 0 1 * d.x₂₂ + A 0 2 * d.x₂₃) ∧
      (A 0 0 * d.x₁₁ + A 0 1 * d.x₁₂ + A 0 2 * d.x₁₃ > A 0 0 * d.x₃₁ + A 0 1 * d.x₃₂ + A 0 2 * d.x₃₃) ∧
      (A 1 0 * d.x₂₁ + A 1 1 * d.x₂₂ + A 1 2 * d.x₂₃ > A 1 0 * d.x₁₁ + A 1 1 * d.x₁₂ + A 1 2 * d.x₁₃) ∧
      (A 1 0 * d.x₂₁ + A 1 1 * d.x₂₂ + A 1 2 * d.x₂₃ > A 1 0 * d.x₃₁ + A 1 1 * d.x₃₂ + A 1 2 * d.x₃₃) ∧
      (A 2 0 * d.x₃₁ + A 2 1 * d.x₃₂ + A 2 2 * d.x₃₃ > A 2 0 * d.x₁₁ + A 2 1 * d.x₁₂ + A 2 2 * d.x₁₃) ∧
      (A 2 0 * d.x₃₁ + A 2 1 * d.x₃₂ + A 2 2 * d.x₃₃ > A 2 0 * d.x₂₁ + A 2 1 * d.x₂₂ + A 2 2 * d.x₂₃) :=
sorry

end NUMINAMATH_CALUDE_horse_distribution_exists_l2288_228840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2288_228868

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)) →  -- arithmetic sequence condition
  a 3 + a 6 + a 9 = 3 * Real.pi / 4 →               -- given condition
  Real.cos (a 2 + a 10 + Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2288_228868


namespace NUMINAMATH_CALUDE_first_book_pictures_correct_l2288_228885

/-- The number of pictures in the first coloring book -/
def pictures_in_first_book : ℕ := 23

/-- The number of pictures in the second coloring book -/
def pictures_in_second_book : ℕ := 32

/-- The total number of pictures in both coloring books -/
def total_pictures : ℕ := 55

/-- Theorem stating that the number of pictures in the first coloring book is correct -/
theorem first_book_pictures_correct :
  pictures_in_first_book + pictures_in_second_book = total_pictures :=
by sorry

end NUMINAMATH_CALUDE_first_book_pictures_correct_l2288_228885


namespace NUMINAMATH_CALUDE_max_min_values_part1_unique_b_part2_l2288_228871

noncomputable section

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem max_min_values_part1 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x ≤ max) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = max) ∧
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, min ≤ f (-1) 3 x) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = min) ∧
    max = 2 ∧
    min = Real.log 2 + 5/4 :=
sorry

theorem unique_b_part2 :
  ∃! b : ℝ,
    b > 0 ∧
    (∀ x ∈ Set.Ioo 0 (Real.exp 1),
      f 0 b x ≥ 3) ∧
    (∃ x ∈ Set.Ioo 0 (Real.exp 1),
      f 0 b x = 3) ∧
    b = Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_part1_unique_b_part2_l2288_228871


namespace NUMINAMATH_CALUDE_percent_y_of_x_l2288_228802

theorem percent_y_of_x (x y : ℝ) (h : 0.5 * (x - y) = 0.3 * (x + y)) : y / x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l2288_228802


namespace NUMINAMATH_CALUDE_x_four_coefficient_range_l2288_228830

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^4 in the expansion of (1+x+mx^2)^10
def coefficient (m : ℝ) : ℝ := binomial 10 4 + binomial 10 2 * m^2 + binomial 10 1 * binomial 9 2 * m

-- State the theorem
theorem x_four_coefficient_range :
  {m : ℝ | coefficient m > -330} = {m : ℝ | m < -6 ∨ m > -2} := by sorry

end NUMINAMATH_CALUDE_x_four_coefficient_range_l2288_228830


namespace NUMINAMATH_CALUDE_faster_train_speed_l2288_228849

/-- The speed of the faster train given the conditions of the problem -/
theorem faster_train_speed 
  (slower_speed : ℝ) 
  (slower_length : ℝ) 
  (faster_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : slower_speed = 90) 
  (h2 : slower_length = 1.10) 
  (h3 : faster_length = 0.90) 
  (h4 : crossing_time = 24 / 3600) : 
  ∃ faster_speed : ℝ, 
    faster_speed = 210 ∧ 
    (slower_length + faster_length) / crossing_time = faster_speed + slower_speed :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2288_228849


namespace NUMINAMATH_CALUDE_robot_purchase_strategy_l2288_228870

/-- The problem of finding optimal robot purchase strategy -/
theorem robot_purchase_strategy 
  (price_difference : ℕ) 
  (cost_A cost_B : ℕ) 
  (total_units : ℕ) 
  (discount_rate : ℚ) : 
  price_difference = 200 →
  cost_A = 2000 →
  cost_B = 1200 →
  total_units = 40 →
  discount_rate = 1/5 →
  ∃ (price_A price_B units_A units_B min_cost : ℕ),
    -- Unit prices
    price_A = 500 ∧ 
    price_B = 300 ∧ 
    price_A = price_B + price_difference ∧
    cost_A * price_B = cost_B * price_A ∧
    -- Optimal purchase strategy
    units_A = 10 ∧
    units_B = 30 ∧
    units_A + units_B = total_units ∧
    units_B ≤ 3 * units_A ∧
    min_cost = 11200 ∧
    min_cost = (price_A * units_A + price_B * units_B) * (1 - discount_rate) ∧
    ∀ (other_A other_B : ℕ), 
      other_A + other_B = total_units →
      other_B ≤ 3 * other_A →
      min_cost ≤ (price_A * other_A + price_B * other_B) * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_robot_purchase_strategy_l2288_228870


namespace NUMINAMATH_CALUDE_inverse_composition_equals_two_l2288_228893

-- Define the function f
def f : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 2
| 4 => 5
| 5 => 1

-- Assume f has an inverse
axiom f_has_inverse : Function.Bijective f

-- Define f⁻¹ using the inverse of f
noncomputable def f_inv : Fin 5 → Fin 5 := Function.invFun f

-- State the theorem
theorem inverse_composition_equals_two :
  f_inv (f_inv (f_inv 3)) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_two_l2288_228893


namespace NUMINAMATH_CALUDE_container_capacity_l2288_228823

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 54 = 0.75 * C) : C = 120 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2288_228823


namespace NUMINAMATH_CALUDE_find_special_numbers_l2288_228896

theorem find_special_numbers : ∃ (x y : ℕ), 
  x + y = 2013 ∧ 
  y = 5 * ((x / 100) + 1) ∧ 
  x ≥ y ∧ 
  x = 1913 := by
  sorry

end NUMINAMATH_CALUDE_find_special_numbers_l2288_228896


namespace NUMINAMATH_CALUDE_x_value_proof_l2288_228822

theorem x_value_proof (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2288_228822


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2288_228836

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2288_228836


namespace NUMINAMATH_CALUDE_number_grid_solution_l2288_228827

theorem number_grid_solution : 
  ∃ (a b c d : ℕ) (s : Finset ℕ),
    s = {1, 2, 3, 4, 5, 6, 7, 8} ∧
    a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b = c ∧
    c / b = d ∧
    a = d :=
by sorry

end NUMINAMATH_CALUDE_number_grid_solution_l2288_228827


namespace NUMINAMATH_CALUDE_equation_solutions_l2288_228834

theorem equation_solutions (x y n : ℕ+) : 
  (((x : ℝ)^2 + (y : ℝ)^2)^(n : ℝ) = ((x * y : ℝ)^2016)) ↔ 
  n ∈ ({1344, 1728, 1792, 1920, 1984} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2288_228834


namespace NUMINAMATH_CALUDE_series_sum_l2288_228813

/-- r is the positive real solution to x³ - ¼x - 1 = 0 -/
def r : ℝ := sorry

/-- T is the sum of the infinite series r + 2r⁴ + 3r⁷ + 4r¹⁰ + ... -/
noncomputable def T : ℝ := sorry

/-- The equation that r satisfies -/
axiom r_eq : r^3 - (1/4)*r - 1 = 0

/-- The main theorem: T equals 4 / (1 + 4/r) -/
theorem series_sum : T = 4 / (1 + 4/r) := by sorry

end NUMINAMATH_CALUDE_series_sum_l2288_228813


namespace NUMINAMATH_CALUDE_game_probability_difference_l2288_228866

def p_heads : ℚ := 3/4
def p_tails : ℚ := 1/4

def p_win_game_c : ℚ := p_heads^4 + p_tails^4

def p_win_game_d : ℚ := p_heads^4 * p_tails + p_tails^4 * p_heads

theorem game_probability_difference :
  p_win_game_c - p_win_game_d = 61/256 := by sorry

end NUMINAMATH_CALUDE_game_probability_difference_l2288_228866


namespace NUMINAMATH_CALUDE_total_adjusted_income_equals_1219_72_l2288_228883

def initial_investment : ℝ := 6800
def stock_allocation : ℝ := 0.6
def bond_allocation : ℝ := 0.3
def cash_allocation : ℝ := 0.1
def inflation_rate : ℝ := 0.02

def cash_interest_rates : Fin 3 → ℝ
| 0 => 0.01
| 1 => 0.02
| 2 => 0.03

def stock_gains : Fin 3 → ℝ
| 0 => 0.08
| 1 => 0.04
| 2 => 0.10

def bond_returns : Fin 3 → ℝ
| 0 => 0.05
| 1 => 0.06
| 2 => 0.04

def adjusted_annual_income (i : Fin 3) : ℝ :=
  let stock_income := initial_investment * stock_allocation * stock_gains i
  let bond_income := initial_investment * bond_allocation * bond_returns i
  let cash_income := initial_investment * cash_allocation * cash_interest_rates i
  let total_income := stock_income + bond_income + cash_income
  total_income * (1 - inflation_rate)

theorem total_adjusted_income_equals_1219_72 :
  (adjusted_annual_income 0) + (adjusted_annual_income 1) + (adjusted_annual_income 2) = 1219.72 :=
sorry

end NUMINAMATH_CALUDE_total_adjusted_income_equals_1219_72_l2288_228883


namespace NUMINAMATH_CALUDE_range_of_k_l2288_228867

/-- Given a function f(x) = √(x+1) + k and an interval [a, b] where the range of f(x) on [a, b] is [a+1, b+1], prove that the range of k is (-1/4, 0]. -/
theorem range_of_k (a b : ℝ) (k : ℝ) (h_le : a ≤ b) :
  (∀ y ∈ Set.Icc (a + 1) (b + 1), ∃ x ∈ Set.Icc a b, Real.sqrt (x + 1) + k = y) →
  k ∈ Set.Ioo (-1/4) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l2288_228867


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2288_228842

theorem smallest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / (a + b + 2 * c) + b * c / (b + c + 2 * a) + c * a / (c + a + 2 * b) ≤ (1/4) * (a + b + c)) ∧
  ∀ k : ℝ, k > 0 → k < 1/4 →
    ∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
      a' * b' / (a' + b' + 2 * c') + b' * c' / (b' + c' + 2 * a') + c' * a' / (c' + a' + 2 * b') > k * (a' + b' + c') :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2288_228842


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l2288_228818

theorem triangle_trig_identity (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) (h1 : a = 4) (h2 : b = 7) (h3 : c = 5) :
  let α := Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))
  (Real.sin (α/2))^6 + (Real.cos (α/2))^6 = 7/25 := by sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l2288_228818


namespace NUMINAMATH_CALUDE_phone_not_answered_probability_l2288_228861

theorem phone_not_answered_probability
  (p1 : ℝ) (p2 : ℝ) (p3 : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.25)
  (h3 : p3 = 0.45) :
  1 - p1 - p2 - p3 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_phone_not_answered_probability_l2288_228861


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2288_228805

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2288_228805


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2288_228853

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 8) - (2 * x^3 + x^2 + 3 * x - 15) = x^3 + 3 * x^2 - 8 * x + 23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2288_228853


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2288_228804

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The positive difference between two terms of an arithmetic sequence -/
def positiveDifference (a₁ : ℤ) (d : ℤ) (m n : ℕ) : ℕ :=
  (arithmeticSequenceTerm a₁ d m - arithmeticSequenceTerm a₁ d n).natAbs

theorem arithmetic_sequence_difference :
  positiveDifference (-8) 8 1020 1000 = 160 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2288_228804


namespace NUMINAMATH_CALUDE_common_difference_is_three_l2288_228886

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem common_difference_is_three
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 2 + a 3 = 9)
  (h_sum2 : a 4 + a 5 = 21) :
  ∃ d, d = 3 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_three_l2288_228886


namespace NUMINAMATH_CALUDE_problem_solution_l2288_228809

theorem problem_solution (x : ℝ) 
  (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (6 * x) * Real.sqrt (5 * x) * Real.sqrt (10 * x) = 20) : 
  x = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2288_228809


namespace NUMINAMATH_CALUDE_M_characterization_inequality_in_M_l2288_228897

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 4}

-- Theorem 1: Characterization of set M
theorem M_characterization : M = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a^2 + 2*a - 3) * (b^2 + 2*b - 3) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_M_characterization_inequality_in_M_l2288_228897


namespace NUMINAMATH_CALUDE_train_speed_proof_l2288_228841

/-- Proves that a train crossing a bridge has a speed of approximately 36 kmph given specific conditions. -/
theorem train_speed_proof (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 150)
  (h3 : time_to_cross = 28.997680185585153) : 
  ∃ (speed : ℝ), abs (speed - 36) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l2288_228841


namespace NUMINAMATH_CALUDE_pen_package_size_l2288_228810

def is_proper_factor (n m : ℕ) : Prop := n ∣ m ∧ n ≠ 1 ∧ n ≠ m

theorem pen_package_size (pen_package_size : ℕ) 
  (h1 : pen_package_size > 0)
  (h2 : ∃ (num_packages : ℕ), num_packages * pen_package_size = 60) :
  is_proper_factor pen_package_size 60 := by
sorry

end NUMINAMATH_CALUDE_pen_package_size_l2288_228810


namespace NUMINAMATH_CALUDE_music_stand_cost_l2288_228800

/-- The cost of Jason's music stand, given his total spending and the costs of other items. -/
theorem music_stand_cost (total_spent flute_cost book_cost : ℚ) 
  (h1 : total_spent = 158.35)
  (h2 : flute_cost = 142.46)
  (h3 : book_cost = 7) :
  total_spent - (flute_cost + book_cost) = 8.89 := by
  sorry

end NUMINAMATH_CALUDE_music_stand_cost_l2288_228800


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l2288_228862

theorem two_digit_number_theorem :
  ∀ n : ℕ,
  (n ≥ 80) →
  (n < 100) →
  (n % 10 = n / 10 - 1) →
  (n = 87 ∨ n = 98) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l2288_228862


namespace NUMINAMATH_CALUDE_pages_needed_is_twelve_l2288_228846

/-- Calculates the number of pages needed to organize sports cards -/
def pages_needed (new_baseball old_baseball new_basketball old_basketball new_football old_football cards_per_page : ℕ) : ℕ :=
  let total_baseball := new_baseball + old_baseball
  let total_basketball := new_basketball + old_basketball
  let total_football := new_football + old_football
  let baseball_pages := (total_baseball + cards_per_page - 1) / cards_per_page
  let basketball_pages := (total_basketball + cards_per_page - 1) / cards_per_page
  let football_pages := (total_football + cards_per_page - 1) / cards_per_page
  baseball_pages + basketball_pages + football_pages

/-- Theorem stating that the number of pages needed is 12 -/
theorem pages_needed_is_twelve :
  pages_needed 3 9 4 6 7 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pages_needed_is_twelve_l2288_228846


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2288_228824

theorem function_satisfies_equation (x : ℝ) :
  let y : ℝ → ℝ := λ x => x * Real.sqrt (1 - x^2)
  let y' : ℝ → ℝ := λ x => Real.sqrt (1 - x^2) - x^2 / Real.sqrt (1 - x^2)
  y x * y' x = x - 2 * x^3 := by sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2288_228824


namespace NUMINAMATH_CALUDE_no_equal_functions_l2288_228874

def f₁ (x : ℤ) : ℤ := x * (x - 2007)
def f₂ (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f₁₀₀₄ (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equal_functions :
  ∀ x : ℤ, 0 ≤ x ∧ x ≤ 2007 →
    (f₁ x ≠ f₂ x) ∧ (f₁ x ≠ f₁₀₀₄ x) ∧ (f₂ x ≠ f₁₀₀₄ x) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_functions_l2288_228874


namespace NUMINAMATH_CALUDE_simplify_expression_l2288_228869

theorem simplify_expression (x y : ℝ) : 8*x + 5*y + 3 - 2*x + 9*y + 15 = 6*x + 14*y + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2288_228869


namespace NUMINAMATH_CALUDE_largest_n_is_correct_l2288_228859

/-- Represents the coefficients of the quadratic expression 6x^2 + nx + 48 -/
structure QuadraticCoeffs where
  n : ℤ

/-- Represents the coefficients of the linear factors (2x + A)(3x + B) -/
structure LinearFactors where
  A : ℤ
  B : ℤ

/-- Checks if the given linear factors produce the quadratic expression -/
def is_valid_factorization (q : QuadraticCoeffs) (f : LinearFactors) : Prop :=
  (2 * f.B + 3 * f.A = q.n) ∧ (f.A * f.B = 48)

/-- The largest value of n for which the quadratic can be factored -/
def largest_n : ℤ := 99

theorem largest_n_is_correct : 
  (∀ q : QuadraticCoeffs, ∃ f : LinearFactors, is_valid_factorization q f → q.n ≤ largest_n) ∧
  (∃ q : QuadraticCoeffs, ∃ f : LinearFactors, is_valid_factorization q f ∧ q.n = largest_n) :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_correct_l2288_228859


namespace NUMINAMATH_CALUDE_table_tennis_probabilities_l2288_228860

/-- Represents the probability of player A winning a serve -/
def p_win : ℝ := 0.6

/-- Probability that player A scores i points in two consecutive serves -/
def p_score (i : Fin 3) : ℝ :=
  match i with
  | 0 => (1 - p_win)^2
  | 1 => 2 * p_win * (1 - p_win)
  | 2 => p_win^2

/-- Theorem stating the probabilities of specific score situations in a table tennis game -/
theorem table_tennis_probabilities :
  let p_b_leads := p_score 0 * p_win + p_score 1 * (1 - p_win)
  let p_a_leads := p_score 1 * p_score 2 + p_score 2 * p_score 1 + p_score 2 * p_score 2
  (p_b_leads = 0.352) ∧ (p_a_leads = 0.3072) := by
  sorry


end NUMINAMATH_CALUDE_table_tennis_probabilities_l2288_228860


namespace NUMINAMATH_CALUDE_reuleaux_triangle_fits_all_holes_l2288_228876

-- Define a Reuleaux Triangle
structure ReuleauxTriangle where
  -- Add necessary properties of a Reuleaux Triangle
  constant_width : ℝ

-- Define the types of holes
inductive HoleType
  | Triangular
  | Square
  | Circular

-- Define a function to check if a shape fits into a hole
def fits_into (shape : ReuleauxTriangle) (hole : HoleType) : Prop :=
  match hole with
  | HoleType.Triangular => true -- Assume it fits into triangular hole
  | HoleType.Square => true     -- Assume it fits into square hole
  | HoleType.Circular => true   -- Assume it fits into circular hole

-- Theorem statement
theorem reuleaux_triangle_fits_all_holes (r : ReuleauxTriangle) :
  (∀ (h : HoleType), fits_into r h) :=
sorry

end NUMINAMATH_CALUDE_reuleaux_triangle_fits_all_holes_l2288_228876


namespace NUMINAMATH_CALUDE_finite_steps_33_disks_infinite_steps_32_disks_l2288_228832

/-- Represents a board with disks -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (disks : Nat)

/-- Represents a move on the board -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Represents the state of the game after some number of steps -/
structure GameState :=
  (board : Board)
  (step : Nat)

/-- Predicate to check if a game state is valid -/
def isValid (state : GameState) : Prop :=
  state.board.disks ≤ state.board.rows * state.board.cols

/-- Predicate to check if a move is valid given the previous move -/
def isValidMove (prevMove : Option Move) (currMove : Move) : Prop :=
  match prevMove with
  | none => true
  | some Move.Up => currMove = Move.Left ∨ currMove = Move.Right
  | some Move.Down => currMove = Move.Left ∨ currMove = Move.Right
  | some Move.Left => currMove = Move.Up ∨ currMove = Move.Down
  | some Move.Right => currMove = Move.Up ∨ currMove = Move.Down

/-- Theorem: With 33 disks on a 5x9 board, only finitely many steps are possible -/
theorem finite_steps_33_disks (board : Board) (h : board.rows = 5 ∧ board.cols = 9 ∧ board.disks = 33) :
  ∃ n : Nat, ∀ state : GameState, state.board = board → state.step > n → ¬isValid state :=
sorry

/-- Theorem: With 32 disks on a 5x9 board, infinitely many steps are possible -/
theorem infinite_steps_32_disks (board : Board) (h : board.rows = 5 ∧ board.cols = 9 ∧ board.disks = 32) :
  ∀ n : Nat, ∃ state : GameState, state.board = board ∧ state.step = n ∧ isValid state :=
sorry

end NUMINAMATH_CALUDE_finite_steps_33_disks_infinite_steps_32_disks_l2288_228832


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2288_228879

/-- The area of the region covered by two identical squares overlapping to form a regular octagon
    but not covered by a circle, given the circle's radius and π value. -/
theorem shaded_area_theorem (R : ℝ) (π : ℝ) (h1 : R = 60) (h2 : π = 3.14) :
  let total_square_area := 2 * R * R
  let circle_area := π * R * R
  total_square_area - circle_area = 3096 := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2288_228879


namespace NUMINAMATH_CALUDE_add_twice_eq_thrice_l2288_228829

theorem add_twice_eq_thrice (a : ℝ) : a + 2 * a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_add_twice_eq_thrice_l2288_228829


namespace NUMINAMATH_CALUDE_sam_initial_money_l2288_228812

/-- The amount of money Sam had initially -/
def initial_money (num_books : ℕ) (cost_per_book : ℕ) (money_left : ℕ) : ℕ :=
  num_books * cost_per_book + money_left

/-- Theorem stating that Sam's initial money was 79 dollars -/
theorem sam_initial_money :
  initial_money 9 7 16 = 79 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_money_l2288_228812


namespace NUMINAMATH_CALUDE_choose_service_providers_and_accessories_l2288_228865

def total_individuals : ℕ := 4
def total_service_providers : ℕ := 25
def total_accessories : ℕ := 5

def ways_to_choose : ℕ := (total_service_providers - 0) *
                           (total_service_providers - 1) *
                           (total_service_providers - 2) *
                           (total_service_providers - 3) *
                           (total_accessories - 0) *
                           (total_accessories - 1) *
                           (total_accessories - 2) *
                           (total_accessories - 3)

theorem choose_service_providers_and_accessories :
  ways_to_choose = 36432000 :=
sorry

end NUMINAMATH_CALUDE_choose_service_providers_and_accessories_l2288_228865


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2288_228888

theorem square_to_rectangle_ratio : 
  ∀ (square_side : ℝ) (rectangle_base rectangle_height : ℝ),
  square_side = 3 →
  rectangle_base * rectangle_height = square_side^2 →
  rectangle_base = (square_side^2 + (square_side/2)^2).sqrt →
  (rectangle_height / rectangle_base) = 4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2288_228888


namespace NUMINAMATH_CALUDE_part_one_part_two_l2288_228819

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part I
theorem part_one :
  let m : ℝ := -1
  {x : ℝ | f x m ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem part_two :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (3/4 : ℝ) 2, f x m ≤ |2*x + 1|) →
  -11/4 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2288_228819


namespace NUMINAMATH_CALUDE_cubic_max_value_l2288_228845

/-- Given a cubic function with a known maximum value, prove the constant term --/
theorem cubic_max_value (m : ℝ) : 
  (∃ (x : ℝ), ∀ (t : ℝ), -t^3 + 3*t^2 + m ≤ -x^3 + 3*x^2 + m) ∧
  (∃ (x : ℝ), -x^3 + 3*x^2 + m = 10) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_max_value_l2288_228845


namespace NUMINAMATH_CALUDE_rogers_dimes_l2288_228828

/-- The number of dimes Roger initially collected -/
def initial_dimes : ℕ := 15

/-- The number of pennies Roger collected -/
def pennies : ℕ := 42

/-- The number of nickels Roger collected -/
def nickels : ℕ := 36

/-- The number of coins Roger had left after donating -/
def coins_left : ℕ := 27

/-- The number of coins Roger donated -/
def coins_donated : ℕ := 66

theorem rogers_dimes :
  initial_dimes = 15 ∧
  pennies + nickels + initial_dimes = coins_left + coins_donated :=
by sorry

end NUMINAMATH_CALUDE_rogers_dimes_l2288_228828


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2288_228864

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eccentricity : Real.sqrt (1 + b^2 / a^2) = Real.sqrt 6 / 2) :
  let asymptote (x : ℝ) := Real.sqrt 2 / 2 * x
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → 
    (y = asymptote x ∨ y = -asymptote x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2288_228864


namespace NUMINAMATH_CALUDE_present_age_of_B_l2288_228851

/-- Given two people A and B, proves that B's current age is 70 years -/
theorem present_age_of_B (A B : ℕ) : 
  (A + 20 = 2 * (B - 20)) →  -- In 20 years, A will be twice as old as B was 20 years ago
  (A = B + 10) →             -- A is now 10 years older than B
  B = 70 :=                  -- B's current age is 70 years
by
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_l2288_228851


namespace NUMINAMATH_CALUDE_craftsman_production_l2288_228894

/-- The number of parts manufactured by a master craftsman during a shift -/
def total_parts : ℕ := by sorry

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 1.5

theorem craftsman_production :
  let initial_rate := first_hour_parts
  let new_rate := initial_rate + rate_increase
  let remaining_parts := total_parts - first_hour_parts
  (remaining_parts : ℚ) / initial_rate - (remaining_parts : ℚ) / new_rate = time_saved →
  total_parts = 210 := by sorry

end NUMINAMATH_CALUDE_craftsman_production_l2288_228894


namespace NUMINAMATH_CALUDE_A_equals_B_l2288_228857

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4*a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4*b^2 + 4*b + 2}

theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l2288_228857


namespace NUMINAMATH_CALUDE_function_with_finitely_many_discontinuities_doesnt_satisfy_condition1_l2288_228847

-- Define the function type
def RealFunction (a b : ℝ) := ℝ → ℝ

-- Define the property of having finitely many discontinuities
def HasFinitelyManyDiscontinuities (f : RealFunction a b) : Prop := sorry

-- Define condition (1) (we don't know what it is exactly, so we'll leave it abstract)
def SatisfiesCondition1 (f : RealFunction a b) : Prop := sorry

-- The main theorem
theorem function_with_finitely_many_discontinuities_doesnt_satisfy_condition1 
  {a b : ℝ} (f : RealFunction a b) 
  (h_finite : HasFinitelyManyDiscontinuities f) : 
  ¬(SatisfiesCondition1 f) := by
  sorry


end NUMINAMATH_CALUDE_function_with_finitely_many_discontinuities_doesnt_satisfy_condition1_l2288_228847


namespace NUMINAMATH_CALUDE_hockey_handshakes_l2288_228890

theorem hockey_handshakes (team_size : Nat) (num_teams : Nat) (num_referees : Nat) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 → 
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 := by
  sorry

end NUMINAMATH_CALUDE_hockey_handshakes_l2288_228890


namespace NUMINAMATH_CALUDE_choir_arrangement_theorem_l2288_228833

theorem choir_arrangement_theorem :
  ∃ (m : ℕ), 
    (∃ (k : ℕ), m = k^2 + 6) ∧ 
    (∃ (n : ℕ), m = n * (n + 6)) ∧
    (∀ (x : ℕ), 
      ((∃ (y : ℕ), x = y^2 + 6) ∧ 
       (∃ (z : ℕ), x = z * (z + 6))) → 
      x ≤ m) ∧
    m = 294 := by
  sorry

end NUMINAMATH_CALUDE_choir_arrangement_theorem_l2288_228833


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_sqrt_3_l2288_228807

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 2*x*y - 1 = 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*a*b - 1 = 0 → 2*x + y ≤ 2*a + b :=
by
  sorry

theorem min_value_is_sqrt_3 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 2*x*y - 1 = 0) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*a*b - 1 = 0 ∧ 2*a + b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_sqrt_3_l2288_228807
