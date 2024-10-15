import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3459_345983

/-- Given a geometric sequence {a_n} where a_2 + a_3 = 1 and a_3 + a_4 = -2,
    prove that a_5 + a_6 + a_7 = 24 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence
  a 2 + a 3 = 1 →                           -- a_2 + a_3 = 1
  a 3 + a 4 = -2 →                          -- a_3 + a_4 = -2
  a 5 + a 6 + a 7 = 24 :=                   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3459_345983


namespace NUMINAMATH_CALUDE_min_mutual_greetings_school_l3459_345981

/-- Represents a school with students and their greetings. -/
structure School :=
  (num_students : Nat)
  (greetings_per_student : Nat)
  (h_students : num_students = 400)
  (h_greetings : greetings_per_student = 200)

/-- The minimum number of pairs of students who have mutually greeted each other. -/
def min_mutual_greetings (s : School) : Nat :=
  s.greetings_per_student * s.num_students - Nat.choose s.num_students 2

/-- Theorem stating the minimum number of mutual greetings in the given school. -/
theorem min_mutual_greetings_school :
    ∀ s : School, min_mutual_greetings s = 200 :=
  sorry

end NUMINAMATH_CALUDE_min_mutual_greetings_school_l3459_345981


namespace NUMINAMATH_CALUDE_sqrt_equality_l3459_345960

theorem sqrt_equality : ∃ (a b : ℕ+), a < b ∧ Real.sqrt (1 + Real.sqrt (45 + 18 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l3459_345960


namespace NUMINAMATH_CALUDE_topsoil_cost_l3459_345989

/-- The cost of premium topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil to be purchased -/
def cubic_yards_to_purchase : ℝ := 7

/-- Theorem: The total cost of purchasing 7 cubic yards of premium topsoil is 1512 dollars -/
theorem topsoil_cost : 
  cost_per_cubic_foot * cubic_feet_per_cubic_yard * cubic_yards_to_purchase = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3459_345989


namespace NUMINAMATH_CALUDE_no_prime_interior_angles_l3459_345993

def interior_angle (n : ℕ) : ℚ := (180 * (n - 2)) / n

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_interior_angles :
  ∀ n : ℕ, 10 ≤ n → n < 20 →
    ¬(∃ k : ℕ, interior_angle n = k ∧ is_prime k) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_interior_angles_l3459_345993


namespace NUMINAMATH_CALUDE_max_notebooks_purchase_l3459_345936

theorem max_notebooks_purchase (total_items : ℕ) (notebook_cost pencil_case_cost max_cost : ℚ) :
  total_items = 10 →
  notebook_cost = 12 →
  pencil_case_cost = 7 →
  max_cost = 100 →
  (∀ x : ℕ, x ≤ total_items →
    x * notebook_cost + (total_items - x) * pencil_case_cost ≤ max_cost →
    x ≤ 6) ∧
  ∃ x : ℕ, x = 6 ∧ x ≤ total_items ∧
    x * notebook_cost + (total_items - x) * pencil_case_cost ≤ max_cost :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchase_l3459_345936


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3459_345941

theorem equation_one_solutions (x : ℝ) : (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3459_345941


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3459_345924

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x + 1 > 0) ↔ -2 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3459_345924


namespace NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l3459_345971

theorem sum_of_divisors_900_prime_factors : 
  let n := 900
  let sum_of_divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id
  (Nat.factors sum_of_divisors).toFinset.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l3459_345971


namespace NUMINAMATH_CALUDE_remainder_problem_l3459_345961

theorem remainder_problem (x : ℤ) (h : (x + 13) % 41 = 18) : x % 82 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3459_345961


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l3459_345984

-- Define a function to get the units digit in base 8
def units_digit_base8 (n : Nat) : Nat :=
  n % 8

-- Define the addition operation in base 8
def add_base8 (a b : Nat) : Nat :=
  (a + b) % 8

-- Theorem statement
theorem units_digit_sum_base8 :
  units_digit_base8 (add_base8 65 75) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l3459_345984


namespace NUMINAMATH_CALUDE_area_of_region_T_l3459_345906

/-- Represents a rhombus PQRS -/
structure Rhombus where
  side_length : ℝ
  angle_Q : ℝ

/-- Represents the region T inside the rhombus -/
def region_T (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a region -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem area_of_region_T (r : Rhombus) :
  r.side_length = 4 ∧ r.angle_Q = 150 * π / 180 →
  area (region_T r) = 8 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_area_of_region_T_l3459_345906


namespace NUMINAMATH_CALUDE_luna_makes_seven_per_hour_l3459_345909

/-- The number of milkshakes Augustus can make per hour -/
def augustus_rate : ℕ := 3

/-- The number of hours Augustus and Luna work -/
def work_hours : ℕ := 8

/-- The total number of milkshakes made by Augustus and Luna -/
def total_milkshakes : ℕ := 80

/-- The number of milkshakes Luna can make per hour -/
def luna_rate : ℕ := (total_milkshakes - augustus_rate * work_hours) / work_hours

theorem luna_makes_seven_per_hour : luna_rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_luna_makes_seven_per_hour_l3459_345909


namespace NUMINAMATH_CALUDE_solution_proof_l3459_345932

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)

noncomputable def g (x : ℝ) : ℝ := 4 * (f⁻¹ x)

theorem solution_proof : ∃ x : ℝ, g x = 20 ∧ x = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_proof_l3459_345932


namespace NUMINAMATH_CALUDE_eighth_group_frequency_l3459_345905

theorem eighth_group_frequency 
  (f1 f2 f3 f4 : ℝ) 
  (f5_to_7 : ℝ) 
  (h1 : f1 = 0.15)
  (h2 : f2 = 0.17)
  (h3 : f3 = 0.11)
  (h4 : f4 = 0.13)
  (h5 : f5_to_7 = 0.32)
  (h6 : ∀ f : ℝ, f ≥ 0 → f ≤ 1) -- Assumption: all frequencies are between 0 and 1
  (h7 : f1 + f2 + f3 + f4 + f5_to_7 + (1 - (f1 + f2 + f3 + f4 + f5_to_7)) = 1) -- Sum of all frequencies is 1
  : 1 - (f1 + f2 + f3 + f4 + f5_to_7) = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_frequency_l3459_345905


namespace NUMINAMATH_CALUDE_one_fourth_of_eight_x_plus_two_l3459_345900

theorem one_fourth_of_eight_x_plus_two (x : ℝ) : (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_eight_x_plus_two_l3459_345900


namespace NUMINAMATH_CALUDE_min_value_theorem_l3459_345975

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3459_345975


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3459_345926

theorem remainder_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 3 * k + 2) ∧ 
  (∃ m : ℕ, k = 4 * m + 3) → 
  n % 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3459_345926


namespace NUMINAMATH_CALUDE_triangle_area_l3459_345982

/-- Given a triangle with perimeter 28 cm, inradius 2.5 cm, one angle of 75 degrees,
    and side lengths in the ratio 3:4:5, prove that its area is 35 cm². -/
theorem triangle_area (p : ℝ) (r : ℝ) (angle : ℝ) (a b c : ℝ)
  (h_perimeter : p = 28)
  (h_inradius : r = 2.5)
  (h_angle : angle = 75)
  (h_ratio : ∃ k : ℝ, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_sides : a + b + c = p) :
  r * p / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3459_345982


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l3459_345990

theorem simplify_cube_roots (h1 : 343 = 7^3) (h2 : 125 = 5^3) :
  (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l3459_345990


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3459_345998

theorem quadratic_roots_relation (u v : ℝ) (m n : ℝ) : 
  (3 * u^2 + 4 * u + 5 = 0) →
  (3 * v^2 + 4 * v + 5 = 0) →
  ((u^2 + 1)^2 + m * (u^2 + 1) + n = 0) →
  ((v^2 + 1)^2 + m * (v^2 + 1) + n = 0) →
  m = -4/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3459_345998


namespace NUMINAMATH_CALUDE_unique_remainder_mod_10_l3459_345959

theorem unique_remainder_mod_10 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := by sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_10_l3459_345959


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3459_345955

def solution_set : Set ℝ := Set.union (Set.Icc (-1/2) 1) (Set.Ico 1 3)

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 5) / ((x - 1)^2) ≥ 2 ∧ x ≠ 1} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3459_345955


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_and_cubes_infinitely_many_coprime_sums_l3459_345944

-- Define a function to check if a number is the sum of two squares
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n ∧ a > 0 ∧ b > 0

-- Define a function to check if a number is the sum of two cubes
def isSumOfTwoCubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^3 + b^3 = n ∧ a > 0 ∧ b > 0

-- Define a function to check if two numbers are coprime
def areCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem smallest_sum_of_squares_and_cubes :
  (∀ n : ℕ, n > 2 ∧ n < 65 → ¬(isSumOfTwoSquares n ∧ isSumOfTwoCubes n)) ∧
  (isSumOfTwoSquares 65 ∧ isSumOfTwoCubes 65) :=
sorry

theorem infinitely_many_coprime_sums :
  ∀ k : ℕ, ∃ n : ℕ,
    (∃ a b : ℕ, n = a^2 + b^2 ∧ areCoprime a b) ∧
    (∃ c d : ℕ, n = c^3 + d^3 ∧ areCoprime c d) ∧
    n > k :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_and_cubes_infinitely_many_coprime_sums_l3459_345944


namespace NUMINAMATH_CALUDE_circle_ratio_l3459_345929

theorem circle_ratio (r R c d : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : 0 < c) (h4 : c < d) :
  (π * R^2) = (c / d) * (π * R^2 - π * r^2) →
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_l3459_345929


namespace NUMINAMATH_CALUDE_problem_statement_l3459_345922

theorem problem_statement : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3459_345922


namespace NUMINAMATH_CALUDE_prob_end_multiple_3_l3459_345973

/-- The number of cards --/
def num_cards : ℕ := 15

/-- The probability of moving left on the spinner --/
def prob_left : ℚ := 1/4

/-- The probability of moving right on the spinner --/
def prob_right : ℚ := 3/4

/-- The probability of starting at a multiple of 3 --/
def prob_start_multiple_3 : ℚ := 1/3

/-- The probability of starting one more than a multiple of 3 --/
def prob_start_one_more : ℚ := 4/15

/-- The probability of starting one less than a multiple of 3 --/
def prob_start_one_less : ℚ := 1/3

/-- The probability of ending at a multiple of 3 after two spins --/
theorem prob_end_multiple_3 : 
  prob_start_multiple_3 * prob_left * prob_left +
  prob_start_one_more * prob_right * prob_right +
  prob_start_one_less * prob_left * prob_left = 7/30 := by sorry

end NUMINAMATH_CALUDE_prob_end_multiple_3_l3459_345973


namespace NUMINAMATH_CALUDE_slope_of_line_l3459_345968

/-- The slope of a line given by the equation x/4 + y/3 = 1 is -3/4 -/
theorem slope_of_line (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l3459_345968


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3459_345939

theorem quadratic_root_difference (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0 ∧ (x₁ - x₂)^2 = 20) → 
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3459_345939


namespace NUMINAMATH_CALUDE_x_value_proof_l3459_345991

theorem x_value_proof (x y : ℝ) : 
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3459_345991


namespace NUMINAMATH_CALUDE_grid_and_unshaded_area_sum_l3459_345912

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)
  (square_size : ℝ)

/-- Represents an unshaded square -/
structure UnshadedSquare :=
  (side_length : ℝ)

/-- Calculates the total area of a grid -/
def grid_area (g : Grid) : ℝ :=
  (g.size * g.square_size) ^ 2

/-- Calculates the area of an unshaded square -/
def unshaded_square_area (u : UnshadedSquare) : ℝ :=
  u.side_length ^ 2

/-- The main theorem to prove -/
theorem grid_and_unshaded_area_sum :
  let g : Grid := { size := 6, square_size := 3 }
  let u : UnshadedSquare := { side_length := 1.5 }
  let num_unshaded : ℕ := 5
  grid_area g + (num_unshaded * unshaded_square_area u) = 335.25 := by
  sorry


end NUMINAMATH_CALUDE_grid_and_unshaded_area_sum_l3459_345912


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3459_345966

theorem yellow_balls_count (total : Nat) (white green red purple : Nat) (prob_not_red_purple : Real) :
  total = 60 →
  white = 22 →
  green = 18 →
  red = 15 →
  purple = 3 →
  prob_not_red_purple = 0.7 →
  ∃ yellow : Nat, yellow = 2 ∧ 
    total = white + green + yellow + red + purple ∧
    (white + green + yellow : Real) / total = prob_not_red_purple :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3459_345966


namespace NUMINAMATH_CALUDE_school_classes_count_l3459_345962

/-- Represents a school with classes -/
structure School where
  total_students : ℕ
  largest_class : ℕ
  class_difference : ℕ

/-- Calculates the number of classes in a school -/
def number_of_classes (s : School) : ℕ :=
  sorry

/-- Theorem stating that for a school with 120 students, largest class of 28, 
    and class difference of 2, the number of classes is 5 -/
theorem school_classes_count (s : School) 
  (h1 : s.total_students = 120) 
  (h2 : s.largest_class = 28) 
  (h3 : s.class_difference = 2) : 
  number_of_classes s = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_classes_count_l3459_345962


namespace NUMINAMATH_CALUDE_distance_to_larger_cross_section_l3459_345996

/-- A right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Area of the smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of the larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- The distance from the apex to the larger cross section -/
def distance_to_larger (p : OctagonalPyramid) : ℝ := sorry

theorem distance_to_larger_cross_section
  (p : OctagonalPyramid)
  (h1 : p.area_small = 400 * Real.sqrt 2)
  (h2 : p.area_large = 900 * Real.sqrt 2)
  (h3 : p.distance_between = 10) :
  distance_to_larger p = 30 := by sorry

end NUMINAMATH_CALUDE_distance_to_larger_cross_section_l3459_345996


namespace NUMINAMATH_CALUDE_multiply_213_by_16_l3459_345956

theorem multiply_213_by_16 (h : 213 * 1.6 = 340.8) : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_multiply_213_by_16_l3459_345956


namespace NUMINAMATH_CALUDE_bulb_switch_problem_l3459_345980

theorem bulb_switch_problem :
  let n : Nat := 11
  let target_state : Fin n → Bool := fun i => i.val + 1 == n
  let valid_state (state : Fin n → Bool) :=
    ∃ (k : Nat), k < 2^n ∧ state = fun i => (k.digits 2).get? i.val == some 1
  { count : Nat // ∀ state, valid_state state ∧ state = target_state → count = 2^(n-1) } :=
by
  sorry

#check bulb_switch_problem

end NUMINAMATH_CALUDE_bulb_switch_problem_l3459_345980


namespace NUMINAMATH_CALUDE_not_p_and_not_q_implies_not_p_and_not_q_l3459_345914

theorem not_p_and_not_q_implies_not_p_and_not_q (p q : Prop) :
  (¬p ∧ ¬q) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_implies_not_p_and_not_q_l3459_345914


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l3459_345934

theorem roots_of_quadratic (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 12) :
  x^2 - 10*x - 11 = 0 ∧ y^2 - 10*y - 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l3459_345934


namespace NUMINAMATH_CALUDE_constant_z_is_plane_l3459_345957

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying z = c in cylindrical coordinates -/
def ConstantZSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.z = c}

/-- Definition of a plane in cylindrical coordinates -/
def IsPlane (S : Set CylindricalPoint) : Prop :=
  ∃ (a b c d : ℝ), c ≠ 0 ∧ ∀ p ∈ S, a * p.r * (Real.cos p.θ) + b * p.r * (Real.sin p.θ) + c * p.z = d

theorem constant_z_is_plane (c : ℝ) : IsPlane (ConstantZSet c) := by
  sorry

end NUMINAMATH_CALUDE_constant_z_is_plane_l3459_345957


namespace NUMINAMATH_CALUDE_lunch_break_duration_l3459_345907

/-- Represents the painting scenario with Paul and his assistants --/
structure PaintingScenario where
  paul_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

/-- Checks if the given scenario satisfies all conditions --/
def satisfies_conditions (s : PaintingScenario) : Prop :=
  -- Monday's condition
  (8 - s.lunch_break) * (s.paul_rate + s.assistants_rate) = 0.6 ∧
  -- Tuesday's condition
  (6 - s.lunch_break) * s.assistants_rate = 0.3 ∧
  -- Wednesday's condition
  (4 - s.lunch_break) * s.paul_rate = 0.1

/-- Theorem stating that the lunch break duration is 60 minutes --/
theorem lunch_break_duration :
  ∃ (s : PaintingScenario), s.lunch_break = 1 ∧ satisfies_conditions s :=
sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l3459_345907


namespace NUMINAMATH_CALUDE_problem_statement_l3459_345995

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 + b^2 ≥ 1/2) ∧ (a*b ≤ 1/4) ∧ (1/a + 1/b > 4) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3459_345995


namespace NUMINAMATH_CALUDE_cubic_polynomial_unique_solution_l3459_345952

def cubic_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a + b * x + c * x^2 + d * x^3

theorem cubic_polynomial_unique_solution 
  (P : ℝ → ℝ) 
  (h_cubic : cubic_polynomial P) 
  (h_neg_one : P (-1) = 2)
  (h_zero : P 0 = 3)
  (h_one : P 1 = 1)
  (h_two : P 2 = 15) :
  ∀ x, P x = 3 + x - 2 * x^2 - x^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_unique_solution_l3459_345952


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3459_345928

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 3) * (Real.sqrt 4 / Real.sqrt 5) *
  (Real.sqrt 6 / Real.sqrt 7) * (Real.sqrt 8 / Real.sqrt 9) =
  16 * Real.sqrt 105 / 315 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3459_345928


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l3459_345979

theorem product_of_cubic_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 168 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l3459_345979


namespace NUMINAMATH_CALUDE_max_revenue_l3459_345954

/-- The revenue function for the bookstore --/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- Theorem stating the maximum revenue and the price at which it occurs --/
theorem max_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  revenue p = 140.625 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue p :=
by
  sorry

end NUMINAMATH_CALUDE_max_revenue_l3459_345954


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l3459_345938

-- Define the number of red and green shoes
def num_red_shoes : ℕ := 7
def num_green_shoes : ℕ := 3

-- Define the total number of shoes
def total_shoes : ℕ := num_red_shoes + num_green_shoes

-- Define the number of shoes to be drawn
def shoes_drawn : ℕ := 2

-- Define the probability of drawing two red shoes
def prob_two_red_shoes : ℚ := 7 / 15

-- Theorem statement
theorem probability_two_red_shoes :
  (Nat.choose num_red_shoes shoes_drawn : ℚ) / (Nat.choose total_shoes shoes_drawn : ℚ) = prob_two_red_shoes :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l3459_345938


namespace NUMINAMATH_CALUDE_morning_evening_email_difference_l3459_345935

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- Theorem stating the difference between morning and evening emails -/
theorem morning_evening_email_difference : 
  morning_emails - evening_emails = 2 := by sorry

end NUMINAMATH_CALUDE_morning_evening_email_difference_l3459_345935


namespace NUMINAMATH_CALUDE_quadruple_solutions_l3459_345992

theorem quadruple_solutions : 
  ∀ (a b c d : ℕ+), 
    (a * b + 2 * a - b = 58 ∧ 
     b * c + 4 * b + 2 * c = 300 ∧ 
     c * d - 6 * c + 4 * d = 101) → 
    ((a = 3 ∧ b = 26 ∧ c = 7 ∧ d = 13) ∨ 
     (a = 15 ∧ b = 2 ∧ c = 73 ∧ d = 7)) := by
  sorry

end NUMINAMATH_CALUDE_quadruple_solutions_l3459_345992


namespace NUMINAMATH_CALUDE_right_triangle_max_expression_l3459_345921

/-- For a right triangle with legs a and b, and hypotenuse c, 
    the expression (a^2 + b^2 + ab) / c^2 is maximized at 1.5 -/
theorem right_triangle_max_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_triangle : a^2 + b^2 = c^2) :
  (a^2 + b^2 + a*b) / c^2 ≤ (3/2) ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    a'^2 + b'^2 = c'^2 ∧ (a'^2 + b'^2 + a'*b') / c'^2 = (3/2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_expression_l3459_345921


namespace NUMINAMATH_CALUDE_marble_redistribution_l3459_345930

/-- Represents the number of marbles each person has -/
structure Marbles :=
  (dilan : ℕ)
  (martha : ℕ)
  (phillip : ℕ)
  (veronica : ℕ)

/-- The theorem statement -/
theorem marble_redistribution (initial : Marbles) (final : Marbles) :
  initial.dilan = 14 →
  initial.martha = 20 →
  initial.veronica = 7 →
  final.dilan = 15 →
  final.martha = 15 →
  final.phillip = 15 →
  final.veronica = 15 →
  initial.dilan + initial.martha + initial.phillip + initial.veronica =
  final.dilan + final.martha + final.phillip + final.veronica →
  initial.phillip = 19 := by
  sorry

end NUMINAMATH_CALUDE_marble_redistribution_l3459_345930


namespace NUMINAMATH_CALUDE_equation_solution_l3459_345908

theorem equation_solution :
  ∃ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + n / (n + 2) = 6 - 3 / (n + 2)) ∧ (n = -6/5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3459_345908


namespace NUMINAMATH_CALUDE_new_years_day_in_big_month_l3459_345937

-- Define the set of months
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

-- Define the set of holidays
inductive Holiday
| NewYearsDay
| ChildrensDay
| TeachersDay

-- Define a function to get the month of a holiday
def holiday_month (h : Holiday) : Month :=
  match h with
  | Holiday.NewYearsDay => Month.January
  | Holiday.ChildrensDay => Month.June
  | Holiday.TeachersDay => Month.September

-- Define the set of big months
def is_big_month (m : Month) : Prop :=
  m = Month.January ∨ m = Month.March ∨ m = Month.May ∨
  m = Month.July ∨ m = Month.August ∨ m = Month.October ∨
  m = Month.December

-- Theorem: New Year's Day falls in a big month
theorem new_years_day_in_big_month :
  is_big_month (holiday_month Holiday.NewYearsDay) :=
by sorry

end NUMINAMATH_CALUDE_new_years_day_in_big_month_l3459_345937


namespace NUMINAMATH_CALUDE_doubled_average_l3459_345904

theorem doubled_average (n : ℕ) (initial_avg : ℝ) (h1 : n = 12) (h2 : initial_avg = 36) :
  let total := n * initial_avg
  let new_total := 2 * total
  let new_avg := new_total / n
  new_avg = 72 := by sorry

end NUMINAMATH_CALUDE_doubled_average_l3459_345904


namespace NUMINAMATH_CALUDE_acrobats_count_correct_l3459_345913

/-- Represents the number of acrobats in the parade. -/
def num_acrobats : ℕ := 4

/-- Represents the number of elephants in the parade. -/
def num_elephants : ℕ := 8

/-- Represents the number of horses in the parade. -/
def num_horses : ℕ := 8

/-- The total number of legs in the parade. -/
def total_legs : ℕ := 72

/-- The total number of heads in the parade. -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  num_acrobats * 2 + num_elephants * 4 + num_horses * 4 = total_legs ∧
  num_acrobats + num_elephants + num_horses = total_heads :=
by sorry

end NUMINAMATH_CALUDE_acrobats_count_correct_l3459_345913


namespace NUMINAMATH_CALUDE_continuous_function_composition_eq_power_l3459_345986

/-- A continuous function satisfying f(f(x)) = kx^9 exists if and only if k ≥ 0 -/
theorem continuous_function_composition_eq_power (k : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = k * x^9) ↔ k ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_continuous_function_composition_eq_power_l3459_345986


namespace NUMINAMATH_CALUDE_rosas_phone_book_calling_l3459_345942

/-- Rosa's phone book calling problem -/
theorem rosas_phone_book_calling (pages_last_week pages_total : ℝ) 
  (h1 : pages_last_week = 10.2)
  (h2 : pages_total = 18.8) :
  pages_total - pages_last_week = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_rosas_phone_book_calling_l3459_345942


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3459_345903

/-- The number of chess players in the tournament. -/
def n : ℕ := 21

/-- The score of the winner. -/
def winner_score (n : ℕ) : ℚ := 3/4 * (n - 1)

/-- The total score of all games in the tournament. -/
def total_score (n : ℕ) : ℚ := 1/2 * n * (n - 1)

/-- The main theorem stating the conditions and the result of the chess tournament. -/
theorem chess_tournament_players :
  (∀ (m : ℕ), m > 1 →
    (winner_score m = 1/13 * (total_score m - winner_score m)) →
    m = n) ∧
  n > 1 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3459_345903


namespace NUMINAMATH_CALUDE_jordana_age_proof_l3459_345958

/-- Jennifer's current age -/
def jennifer_current_age : ℕ := 30 - 10

/-- Jennifer's age in 10 years -/
def jennifer_future_age : ℕ := 30

/-- Jordana's age in 10 years -/
def jordana_future_age : ℕ := 3 * jennifer_future_age

/-- Jordana's current age -/
def jordana_current_age : ℕ := jordana_future_age - 10

theorem jordana_age_proof : jordana_current_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_jordana_age_proof_l3459_345958


namespace NUMINAMATH_CALUDE_flush_probability_l3459_345972

/-- Represents the number of players in the card game -/
def num_players : ℕ := 4

/-- Represents the total number of cards in the deck -/
def total_cards : ℕ := 20

/-- Represents the number of cards per suit -/
def cards_per_suit : ℕ := 5

/-- Represents the number of cards dealt to each player -/
def cards_per_player : ℕ := 5

/-- Calculates the probability of at least one player having a flush after card exchange -/
def probability_of_flush : ℚ := 8 / 969

/-- Theorem stating the probability of at least one player having a flush after card exchange -/
theorem flush_probability : 
  probability_of_flush = 8 / 969 :=
sorry

end NUMINAMATH_CALUDE_flush_probability_l3459_345972


namespace NUMINAMATH_CALUDE_joes_speed_to_petes_speed_ratio_l3459_345917

/-- Prove that the ratio of Joe's speed to Pete's speed is 2:1 -/
theorem joes_speed_to_petes_speed_ratio (
  time : ℝ)
  (total_distance : ℝ)
  (joes_speed : ℝ)
  (h1 : time = 40)
  (h2 : total_distance = 16)
  (h3 : joes_speed = 0.266666666667)
  : joes_speed / ((total_distance - joes_speed * time) / time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_joes_speed_to_petes_speed_ratio_l3459_345917


namespace NUMINAMATH_CALUDE_modulus_of_z_l3459_345949

def i : ℂ := Complex.I

def z : ℂ := (1 + i) * (1 + 2*i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3459_345949


namespace NUMINAMATH_CALUDE_mrs_hilt_initial_money_l3459_345901

/-- Mrs. Hilt's shopping problem -/
theorem mrs_hilt_initial_money :
  ∀ (initial_money toy_truck_cost pencil_case_cost money_left : ℕ),
  toy_truck_cost = 3 →
  pencil_case_cost = 2 →
  money_left = 5 →
  initial_money = toy_truck_cost + pencil_case_cost + money_left →
  initial_money = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_mrs_hilt_initial_money_l3459_345901


namespace NUMINAMATH_CALUDE_cosine_irrationality_l3459_345965

theorem cosine_irrationality (n : ℕ) (h : n ≥ 2) : Irrational (Real.cos (π / 2^n)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_irrationality_l3459_345965


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3459_345987

theorem quadratic_factorization (x : ℝ) : 15 * x^2 + 10 * x - 20 = 5 * (x - 1) * (3 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3459_345987


namespace NUMINAMATH_CALUDE_student_count_last_year_l3459_345911

theorem student_count_last_year 
  (increase_rate : Real) 
  (current_count : Nat) 
  (h1 : increase_rate = 0.2) 
  (h2 : current_count = 960) : 
  ∃ (last_year_count : Nat), 
    (last_year_count : Real) * (1 + increase_rate) = current_count ∧ 
    last_year_count = 800 := by
  sorry

end NUMINAMATH_CALUDE_student_count_last_year_l3459_345911


namespace NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l3459_345946

def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

def uses_digits_once (n : ℕ) (digits : List ℕ) : Prop :=
  let digit_list := n.digits 10
  digit_list.length = digits.length ∧ 
  ∀ d, d ∈ digit_list ↔ d ∈ digits

theorem greatest_five_digit_multiple_of_6 :
  let digits : List ℕ := [4, 5, 7, 8, 9]
  ∀ n : ℕ, 
    n ≤ 99999 ∧ 
    10000 ≤ n ∧
    is_multiple_of_6 n ∧ 
    uses_digits_once n digits →
    n ≤ 97548 :=
by sorry

end NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l3459_345946


namespace NUMINAMATH_CALUDE_car_speed_increase_l3459_345999

/-- Calculates the final speed of a car after modifications -/
def final_speed (original_speed : ℝ) (supercharge_percentage : ℝ) (weight_cut_increase : ℝ) : ℝ :=
  original_speed * (1 + supercharge_percentage) + weight_cut_increase

/-- Theorem stating that the final speed is 205 mph given the specified conditions -/
theorem car_speed_increase :
  final_speed 150 0.3 10 = 205 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_increase_l3459_345999


namespace NUMINAMATH_CALUDE_donnys_remaining_money_l3459_345953

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℕ) (kite_cost : ℕ) (frisbee_cost : ℕ) : ℕ :=
  initial - (kite_cost + frisbee_cost)

/-- Theorem: Donny's remaining money after purchases -/
theorem donnys_remaining_money :
  remaining_money 78 8 9 = 61 := by
  sorry

end NUMINAMATH_CALUDE_donnys_remaining_money_l3459_345953


namespace NUMINAMATH_CALUDE_ruth_math_class_time_l3459_345951

/-- Represents Ruth's school schedule and math class time --/
structure RuthSchedule where
  hours_per_day : ℝ
  days_per_week : ℝ
  math_class_percentage : ℝ

/-- Calculates the number of hours Ruth spends in math class per week --/
def math_class_hours (schedule : RuthSchedule) : ℝ :=
  schedule.hours_per_day * schedule.days_per_week * schedule.math_class_percentage

/-- Theorem stating that Ruth spends 10 hours per week in math class --/
theorem ruth_math_class_time :
  ∃ (schedule : RuthSchedule),
    schedule.hours_per_day = 8 ∧
    schedule.days_per_week = 5 ∧
    schedule.math_class_percentage = 0.25 ∧
    math_class_hours schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_ruth_math_class_time_l3459_345951


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l3459_345940

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure CubeWithTunnel where
  sideLength : ℝ
  pointI : Point3D
  pointJ : Point3D
  pointK : Point3D

/-- Calculates the total surface area of a cube with a drilled tunnel -/
def totalSurfaceArea (cube : CubeWithTunnel) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific cube with tunnel -/
theorem specific_cube_surface_area :
  let cube : CubeWithTunnel := {
    sideLength := 10,
    pointI := { x := 3, y := 0, z := 0 },
    pointJ := { x := 0, y := 3, z := 0 },
    pointK := { x := 0, y := 0, z := 3 }
  }
  totalSurfaceArea cube = 630 := by sorry

end NUMINAMATH_CALUDE_specific_cube_surface_area_l3459_345940


namespace NUMINAMATH_CALUDE_percent_relation_l3459_345994

theorem percent_relation (a b : ℝ) (h : a = 2 * b) : 4 * b = 2 * a := by sorry

end NUMINAMATH_CALUDE_percent_relation_l3459_345994


namespace NUMINAMATH_CALUDE_emma_skateboard_time_l3459_345985

/-- The time taken for Emma to skateboard along a looping path on a highway --/
theorem emma_skateboard_time : ∀ (highway_length highway_width emma_speed : ℝ),
  highway_length = 2 * 5280 →
  highway_width = 50 →
  emma_speed = 4 →
  ∃ (time : ℝ), time = π / 2 ∧ time * emma_speed = 2 * π :=
by
  sorry

end NUMINAMATH_CALUDE_emma_skateboard_time_l3459_345985


namespace NUMINAMATH_CALUDE_factorial_divisibility_l3459_345925

theorem factorial_divisibility :
  ¬(57 ∣ Nat.factorial 18) ∧ (57 ∣ Nat.factorial 19) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l3459_345925


namespace NUMINAMATH_CALUDE_sue_chewing_gums_l3459_345988

theorem sue_chewing_gums (mary_gums sam_gums total_gums : ℕ) (sue_gums : ℕ) :
  mary_gums = 5 →
  sam_gums = 10 →
  total_gums = 30 →
  total_gums = mary_gums + sam_gums + sue_gums →
  sue_gums = 15 := by
  sorry

end NUMINAMATH_CALUDE_sue_chewing_gums_l3459_345988


namespace NUMINAMATH_CALUDE_photo_exhibition_total_l3459_345915

/-- Represents the number of photographs in various categories -/
structure PhotoExhibition where
  octavia_total : ℕ  -- Total photos taken by Octavia
  jack_octavia : ℕ   -- Photos taken by Octavia and framed by Jack
  jack_others : ℕ    -- Photos taken by others and framed by Jack

/-- Theorem stating the total number of photos either framed by Jack or taken by Octavia -/
theorem photo_exhibition_total (e : PhotoExhibition) 
  (h1 : e.octavia_total = 36)
  (h2 : e.jack_octavia = 24)
  (h3 : e.jack_others = 12) : 
  e.octavia_total + e.jack_others = 48 := by
  sorry


end NUMINAMATH_CALUDE_photo_exhibition_total_l3459_345915


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3459_345902

theorem arithmetic_sequence_terms (a₁ aₙ : ℤ) (d : ℤ) (n : ℕ) : 
  a₁ = 1 ∧ aₙ = -89 ∧ d = -2 ∧ aₙ = a₁ + (n - 1) * d → n = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3459_345902


namespace NUMINAMATH_CALUDE_jenny_change_calculation_l3459_345969

/-- Calculate Jenny's change after her purchase -/
theorem jenny_change_calculation :
  let printing_discount : Float := 0.05
  let gift_card_balance : Float := 8.00
  let single_sided_cost : Float := 0.10
  let double_sided_cost : Float := 0.17
  let total_copies : Nat := 7
  let pages_per_essay : Nat := 25
  let single_sided_copies : Nat := 5
  let double_sided_copies : Nat := total_copies - single_sided_copies
  let pen_cost : Float := 1.50
  let pen_count : Nat := 7
  let sales_tax : Float := 0.10
  let cash_payment : Float := 2 * 20.00

  let single_sided_total : Float := single_sided_cost * (single_sided_copies.toFloat * pages_per_essay.toFloat)
  let double_sided_total : Float := double_sided_cost * (double_sided_copies.toFloat * pages_per_essay.toFloat)
  let printing_total : Float := single_sided_total + double_sided_total
  let printing_discounted : Float := printing_total * (1 - printing_discount)
  let pens_total : Float := pen_cost * pen_count.toFloat
  let pens_with_tax : Float := pens_total * (1 + sales_tax)
  let total_cost : Float := printing_discounted + pens_with_tax
  let remaining_cost : Float := total_cost - gift_card_balance
  let change : Float := cash_payment - remaining_cost

  change = 16.50 := by sorry

end NUMINAMATH_CALUDE_jenny_change_calculation_l3459_345969


namespace NUMINAMATH_CALUDE_percentage_product_theorem_l3459_345967

theorem percentage_product_theorem :
  let p1 : ℝ := 40
  let p2 : ℝ := 35
  let p3 : ℝ := 60
  let p4 : ℝ := 70
  let result : ℝ := p1 * p2 * p3 * p4 / 1000000 * 100
  result = 5.88 := by
sorry

end NUMINAMATH_CALUDE_percentage_product_theorem_l3459_345967


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_sqrt_152_is_solution_sqrt_152_is_smallest_solution_l3459_345948

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 23) → x ≥ Real.sqrt 152 :=
by sorry

theorem sqrt_152_is_solution :
  ⌊(Real.sqrt 152)^2⌋ - ⌊Real.sqrt 152⌋^2 = 23 :=
by sorry

theorem sqrt_152_is_smallest_solution :
  ∀ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 23) → x ≥ Real.sqrt 152 ∧
  ⌊(Real.sqrt 152)^2⌋ - ⌊Real.sqrt 152⌋^2 = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_sqrt_152_is_solution_sqrt_152_is_smallest_solution_l3459_345948


namespace NUMINAMATH_CALUDE_max_shapes_8x14_l3459_345978

/-- The number of grid points in an m × n rectangle --/
def gridPoints (m n : ℕ) : ℕ := (m + 1) * (n + 1)

/-- The number of grid points covered by each shape --/
def pointsPerShape : ℕ := 8

/-- The maximum number of shapes that can be placed in the grid --/
def maxShapes (m n : ℕ) : ℕ := (gridPoints m n) / pointsPerShape

theorem max_shapes_8x14 :
  maxShapes 8 14 = 16 := by sorry

end NUMINAMATH_CALUDE_max_shapes_8x14_l3459_345978


namespace NUMINAMATH_CALUDE_log_ratio_squared_l3459_345964

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (hprod : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l3459_345964


namespace NUMINAMATH_CALUDE_local_maximum_value_l3459_345927

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x - 1

theorem local_maximum_value (x : ℝ) :
  ∃ (a : ℝ), (∀ (y : ℝ), ∃ (ε : ℝ), ε > 0 ∧ ∀ (z : ℝ), |z - a| < ε → f z ≤ f a) ∧
  f a = -23/27 :=
sorry

end NUMINAMATH_CALUDE_local_maximum_value_l3459_345927


namespace NUMINAMATH_CALUDE_planting_methods_eq_120_l3459_345970

/-- The number of ways to select and plant vegetables -/
def plantingMethods (totalVarieties : ℕ) (selectedVarieties : ℕ) (plots : ℕ) : ℕ :=
  Nat.choose totalVarieties selectedVarieties * Nat.factorial plots

/-- Theorem stating the number of planting methods for the given scenario -/
theorem planting_methods_eq_120 :
  plantingMethods 5 4 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_planting_methods_eq_120_l3459_345970


namespace NUMINAMATH_CALUDE_continuity_at_two_l3459_345923

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x^2 - 4)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_continuity_at_two_l3459_345923


namespace NUMINAMATH_CALUDE_trapezoid_base_lengths_l3459_345933

theorem trapezoid_base_lengths (h : ℝ) (leg1 leg2 larger_base : ℝ) :
  h = 12 ∧ leg1 = 20 ∧ leg2 = 15 ∧ larger_base = 42 →
  ∃ (smaller_base : ℝ), (smaller_base = 17 ∨ smaller_base = 35) ∧
  (∃ (x y : ℝ), x^2 + h^2 = leg1^2 ∧ y^2 + h^2 = leg2^2 ∧
  (larger_base = x + y + smaller_base ∨ larger_base = x - y + smaller_base)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_base_lengths_l3459_345933


namespace NUMINAMATH_CALUDE_gage_received_fraction_l3459_345963

-- Define the initial numbers of cubes
def grady_red : ℕ := 20
def grady_blue : ℕ := 15
def gage_initial_red : ℕ := 10
def gage_initial_blue : ℕ := 12

-- Define the fraction of blue cubes Gage received
def blue_fraction : ℚ := 1/3

-- Define the total number of cubes Gage has after receiving some from Grady
def gage_total : ℕ := 35

-- Define the fraction of red cubes Gage received as a rational number
def red_fraction : ℚ := 2/5

-- Theorem statement
theorem gage_received_fraction :
  (gage_initial_red : ℚ) + red_fraction * grady_red + 
  (gage_initial_blue : ℚ) + blue_fraction * grady_blue = gage_total :=
sorry

end NUMINAMATH_CALUDE_gage_received_fraction_l3459_345963


namespace NUMINAMATH_CALUDE_remainder_eleven_pow_2023_mod_8_l3459_345918

theorem remainder_eleven_pow_2023_mod_8 : 11^2023 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eleven_pow_2023_mod_8_l3459_345918


namespace NUMINAMATH_CALUDE_cos_difference_formula_l3459_345910

theorem cos_difference_formula (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.cos (α - β) = -59/72 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_formula_l3459_345910


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3459_345950

theorem roots_sum_of_squares (p q : ℝ) : 
  (p^2 - 5*p + 6 = 0) → (q^2 - 5*q + 6 = 0) → p ≠ q → p^2 + q^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3459_345950


namespace NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l3459_345931

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_values_and_monotonicity_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ x y, x < y → f 2 1 x > f 2 1 y) ∧
  (∀ k, (∀ x ≥ 1, f 2 1 (k * 3^x) + f 2 1 (3^x - 9^x + 2) > 0) ↔ k < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l3459_345931


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l3459_345977

def running_shoes_original_price : ℝ := 80
def casual_shoes_original_price : ℝ := 60
def running_shoes_discount : ℝ := 0.25
def casual_shoes_discount : ℝ := 0.40
def sales_tax_rate : ℝ := 0.08
def num_running_shoes : ℕ := 2
def num_casual_shoes : ℕ := 3

def total_cost : ℝ :=
  let running_shoes_discounted_price := running_shoes_original_price * (1 - running_shoes_discount)
  let casual_shoes_discounted_price := casual_shoes_original_price * (1 - casual_shoes_discount)
  let subtotal := num_running_shoes * running_shoes_discounted_price + num_casual_shoes * casual_shoes_discounted_price
  subtotal * (1 + sales_tax_rate)

theorem total_cost_is_correct : total_cost = 246.24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l3459_345977


namespace NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3459_345947

/-- The fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def required_fraction (total_members : ℕ) (min_lists : ℚ) : ℚ :=
  min_lists / total_members

/-- Proof that the required fraction is 0.25 given the specific conditions -/
theorem movie_of_the_year_fraction :
  let total_members : ℕ := 775
  let min_lists : ℚ := 193.75
  required_fraction total_members min_lists = 0.25 := by
sorry

end NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3459_345947


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3459_345997

noncomputable section

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the foci
def LeftFocus (a c : ℝ) : ℝ × ℝ := (-c, 0)
def RightFocus (a c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the point P on the right branch of the hyperbola
def P (a b : ℝ) : ℝ × ℝ := sorry

-- Define the perpendicular bisector of PF₁
def PerpendicularBisectorPF₁ (a b c : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the distance from origin to line PF₁
def DistanceOriginToPF₁ (a b c : ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  (P a b ∈ Hyperbola a b) →
  (RightFocus a c ∈ PerpendicularBisectorPF₁ a b c) →
  (DistanceOriginToPF₁ a b c = a) →
  (c / a = 5 / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3459_345997


namespace NUMINAMATH_CALUDE_parentheses_removal_l3459_345920

theorem parentheses_removal (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l3459_345920


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3459_345919

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3459_345919


namespace NUMINAMATH_CALUDE_integer_list_mean_mode_l3459_345976

theorem integer_list_mean_mode (x : ℕ) : 
  x ≤ 120 →
  x > 0 →
  let list := [45, 76, 110, x, x]
  (list.sum / list.length : ℚ) = 2 * x →
  x = 29 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_mode_l3459_345976


namespace NUMINAMATH_CALUDE_hash_difference_seven_four_l3459_345916

-- Define the # operation
def hash (x y : ℤ) : ℤ := 2*x*y - 3*x - y

-- Theorem statement
theorem hash_difference_seven_four : hash 7 4 - hash 4 7 = -6 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_seven_four_l3459_345916


namespace NUMINAMATH_CALUDE_f_max_value_l3459_345945

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.cos x

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3459_345945


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l3459_345974

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 71)
  (h2 : x^2 * y + x * y^2 = 880) : 
  x^2 + y^2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l3459_345974


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3459_345943

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 + 2x - 4y + 1 = 0
    with a chord length of 4, prove that the minimum value of 1/a + 1/b is 3/2 + √2 -/
theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), a * x - b * y + 2 = 0 ∧ x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    a * x₁ - b * y₁ + 2 = 0 ∧ x₁^2 + y₁^2 + 2*x₁ - 4*y₁ + 1 = 0 ∧
    a * x₂ - b * y₂ + 2 = 0 ∧ x₂^2 + y₂^2 + 2*x₂ - 4*y₂ + 1 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  (1 / a + 1 / b) ≥ 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3459_345943
