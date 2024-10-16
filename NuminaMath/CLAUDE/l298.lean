import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l298_29837

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = k is sqrt(2) -/
theorem hyperbola_eccentricity (k : ℝ) (h : k > 0) :
  let e := Real.sqrt (1 + (Real.sqrt k / Real.sqrt k)^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l298_29837


namespace NUMINAMATH_CALUDE_jennas_tanning_schedule_l298_29860

/-- Jenna's tanning schedule problem -/
theorem jennas_tanning_schedule :
  ∀ (x : ℝ),
  (x ≥ 0) →  -- Non-negative tanning time
  (4 * x + 80 ≤ 200) →  -- Total tanning time constraint
  (x = 30) :=  -- Prove that x is 30 minutes
by
  sorry

end NUMINAMATH_CALUDE_jennas_tanning_schedule_l298_29860


namespace NUMINAMATH_CALUDE_no_special_quadrilateral_l298_29888

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def side_lengths_different (q : Quadrilateral) : Prop := sorry

def angles_different (q : Quadrilateral) : Prop := sorry

-- Define functions to get side lengths and angles
def side_length (q : Quadrilateral) (side : Fin 4) : ℝ := sorry

def angle (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

-- Define predicates for greatest and smallest
def is_greatest_side (q : Quadrilateral) (side : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ side → side_length q side > side_length q other

def is_smallest_side (q : Quadrilateral) (side : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ side → side_length q side < side_length q other

def is_greatest_angle (q : Quadrilateral) (vertex : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ vertex → angle q vertex > angle q other

def is_smallest_angle (q : Quadrilateral) (vertex : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ vertex → angle q vertex < angle q other

-- Define the main theorem
theorem no_special_quadrilateral :
  ¬ ∃ (q : Quadrilateral) (s g a : Fin 4),
    is_convex q ∧
    side_lengths_different q ∧
    angles_different q ∧
    is_smallest_side q s ∧
    is_greatest_side q g ∧
    is_greatest_angle q a ∧
    is_smallest_angle q ((a + 2) % 4) ∧
    (a + 1) % 4 ≠ s ∧
    (a + 3) % 4 ≠ s ∧
    ((a + 2) % 4 + 1) % 4 ≠ g ∧
    ((a + 2) % 4 + 3) % 4 ≠ g :=
sorry

end NUMINAMATH_CALUDE_no_special_quadrilateral_l298_29888


namespace NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l298_29881

theorem students_neither_football_nor_cricket 
  (total : ℕ) 
  (football : ℕ) 
  (cricket : ℕ) 
  (both : ℕ) 
  (h1 : total = 450) 
  (h2 : football = 325) 
  (h3 : cricket = 175) 
  (h4 : both = 100) : 
  total - (football + cricket - both) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l298_29881


namespace NUMINAMATH_CALUDE_equation_solutions_l298_29871

theorem equation_solutions :
  (∃ x : ℝ, 0.4 * x = -1.2 * x + 1.6 ∧ x = 1) ∧
  (∃ y : ℝ, (1/3) * (y + 2) = 1 - (1/6) * (2 * y - 1) ∧ y = 3/4) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l298_29871


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l298_29889

theorem binomial_floor_divisibility (n p : ℕ) (h1 : n ≥ p) (h2 : Nat.Prime (50 * p)) : 
  p ∣ (Nat.choose n p - n / p) :=
by sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l298_29889


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l298_29893

/-- Represents the rowing scenario in a river with current --/
structure RowingScenario where
  stream_rate : ℝ
  rowing_speed : ℝ
  time_ratio : ℝ

/-- Checks if the rowing scenario satisfies the given conditions --/
def is_valid_scenario (s : RowingScenario) : Prop :=
  s.stream_rate = 18 ∧ 
  s.time_ratio = 3 ∧
  (1 / (s.rowing_speed - s.stream_rate)) = s.time_ratio * (1 / (s.rowing_speed + s.stream_rate))

/-- Theorem stating that the man's rowing speed in still water is 36 kmph --/
theorem mans_rowing_speed (s : RowingScenario) : 
  is_valid_scenario s → s.rowing_speed = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_mans_rowing_speed_l298_29893


namespace NUMINAMATH_CALUDE_rose_bush_count_l298_29885

theorem rose_bush_count (initial_bushes planted_bushes : ℕ) :
  initial_bushes = 2 → planted_bushes = 4 →
  initial_bushes + planted_bushes = 6 := by
  sorry

end NUMINAMATH_CALUDE_rose_bush_count_l298_29885


namespace NUMINAMATH_CALUDE_square_side_length_from_rectangle_l298_29840

/-- The side length of a square with an area 7 times larger than a rectangle with length 400 feet and width 300 feet is approximately 916.515 feet. -/
theorem square_side_length_from_rectangle (ε : ℝ) (h : ε > 0) : ∃ (s : ℝ), 
  abs (s - Real.sqrt (7 * 400 * 300)) < ε ∧ 
  s^2 = 7 * 400 * 300 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_from_rectangle_l298_29840


namespace NUMINAMATH_CALUDE_not_consecutive_numbers_l298_29824

theorem not_consecutive_numbers (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ¬∃ (k : ℕ), ({2023 + a - b, 2023 + b - c, 2023 + c - a} : Finset ℕ) = {k - 1, k, k + 1} :=
by sorry

end NUMINAMATH_CALUDE_not_consecutive_numbers_l298_29824


namespace NUMINAMATH_CALUDE_min_abs_sum_l298_29898

theorem min_abs_sum (x : ℝ) : 
  ∃ (l : ℝ), l = 45 ∧ ∀ y : ℝ, |y - 2| + |y - 47| ≥ l :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l298_29898


namespace NUMINAMATH_CALUDE_largest_non_representable_l298_29846

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k

def is_representable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_representable : 
  (∀ n > 187, is_representable n) ∧ ¬is_representable 187 := by sorry

end NUMINAMATH_CALUDE_largest_non_representable_l298_29846


namespace NUMINAMATH_CALUDE_art_class_problem_l298_29870

theorem art_class_problem (total_students : ℕ) (total_artworks : ℕ) 
  (first_half_artworks_per_student : ℕ) :
  total_students = 10 →
  total_artworks = 35 →
  first_half_artworks_per_student = 3 →
  ∃ (second_half_artworks_per_student : ℕ),
    (total_students / 2 * first_half_artworks_per_student) + 
    (total_students / 2 * second_half_artworks_per_student) = total_artworks ∧
    second_half_artworks_per_student = 4 :=
by sorry

end NUMINAMATH_CALUDE_art_class_problem_l298_29870


namespace NUMINAMATH_CALUDE_existence_of_single_root_quadratic_l298_29807

/-- Given a quadratic polynomial with leading coefficient 1 and exactly one root,
    there exists a point (p, q) such that x^2 + px + q also has exactly one root. -/
theorem existence_of_single_root_quadratic 
  (b c : ℝ) 
  (h1 : b^2 - 4*c = 0) : 
  ∃ p q : ℝ, p^2 - 4*q = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_single_root_quadratic_l298_29807


namespace NUMINAMATH_CALUDE_binary_110101_is_53_l298_29829

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101 -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_is_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_is_53_l298_29829


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l298_29862

theorem divisibility_implies_equality (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : a * b ∣ (a^2 + b^2)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l298_29862


namespace NUMINAMATH_CALUDE_stadium_fee_difference_l298_29827

def stadium_capacity : ℕ := 2000
def entry_fee : ℕ := 20

theorem stadium_fee_difference :
  let full_capacity := stadium_capacity
  let partial_capacity := (3 * stadium_capacity) / 4
  let full_fees := full_capacity * entry_fee
  let partial_fees := partial_capacity * entry_fee
  full_fees - partial_fees = 10000 := by
sorry

end NUMINAMATH_CALUDE_stadium_fee_difference_l298_29827


namespace NUMINAMATH_CALUDE_vector_expression_inequality_l298_29865

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given arbitrary points P, A, B, Q in a real vector space V, 
    the expression PA + AB - BQ is not always equal to PQ. -/
theorem vector_expression_inequality (P A B Q : V) :
  ¬ (∀ (P A B Q : V), (A - P) + (B - A) - (Q - B) = Q - P) :=
sorry

end NUMINAMATH_CALUDE_vector_expression_inequality_l298_29865


namespace NUMINAMATH_CALUDE_median_salary_is_clerk_salary_l298_29878

/-- Represents a position in the company with its title, number of employees, and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company --/
def positions : List Position := [
  ⟨"CEO", 1, 135000⟩,
  ⟨"Senior Vice-President", 4, 95000⟩,
  ⟨"Manager", 12, 80000⟩,
  ⟨"Assistant Manager", 8, 55000⟩,
  ⟨"Clerk", 38, 25000⟩
]

/-- The total number of employees in the company --/
def totalEmployees : Nat := 63

/-- Calculates the median salary of the company --/
def medianSalary (pos : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary is equal to the Clerk's salary --/
theorem median_salary_is_clerk_salary :
  medianSalary positions totalEmployees = 25000 := by sorry

end NUMINAMATH_CALUDE_median_salary_is_clerk_salary_l298_29878


namespace NUMINAMATH_CALUDE_fermat_point_sum_l298_29815

theorem fermat_point_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 + x*y = 1)
  (h2 : y^2 + z^2 + y*z = 2)
  (h3 : z^2 + x^2 + z*x = 3) :
  x + y + z = Real.sqrt (3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_fermat_point_sum_l298_29815


namespace NUMINAMATH_CALUDE_clothing_combinations_l298_29843

theorem clothing_combinations (hoodies sweatshirts jeans slacks : ℕ) 
  (h_hoodies : hoodies = 5)
  (h_sweatshirts : sweatshirts = 4)
  (h_jeans : jeans = 3)
  (h_slacks : slacks = 5) :
  (hoodies + sweatshirts) * (jeans + slacks) = 72 := by
  sorry

end NUMINAMATH_CALUDE_clothing_combinations_l298_29843


namespace NUMINAMATH_CALUDE_travel_time_l298_29855

/-- Given a person's travel rate, calculate the time to travel a certain distance -/
theorem travel_time (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) :
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  (distance_to_bernard / distance_to_julia) * time_to_julia = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_travel_time_l298_29855


namespace NUMINAMATH_CALUDE_solve_system_l298_29805

theorem solve_system (u v : ℚ) 
  (eq1 : 4 * u - 5 * v = 23)
  (eq2 : 2 * u + 4 * v = -8) :
  u + v = -1 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l298_29805


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l298_29847

/-- The distance between the foci of a hyperbola given by the equation 3x^2 - 18x - 2y^2 - 4y = 48 -/
theorem hyperbola_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, 3 * x^2 - 18 * x - 2 * y^2 - 4 * y = 48) →
    (a^2 = 53 / 3) →
    (b^2 = 53 / 6) →
    (c^2 = a^2 + b^2) →
    (2 * c = 2 * Real.sqrt (53 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l298_29847


namespace NUMINAMATH_CALUDE_pencils_per_row_l298_29899

theorem pencils_per_row (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) 
  (h1 : packs = 28) 
  (h2 : pencils_per_pack = 24) 
  (h3 : rows = 42) :
  (packs * pencils_per_pack) / rows = 16 := by
  sorry

#check pencils_per_row

end NUMINAMATH_CALUDE_pencils_per_row_l298_29899


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l298_29850

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the points
def P₁ : ℝ × ℝ := (-1, f (-1))
def P₂ : ℝ × ℝ := (2, f 2)
def P₃ : ℝ × ℝ := (5, f 5)

-- Theorem statement
theorem quadratic_points_ordering :
  P₂.2 > P₁.2 ∧ P₁.2 > P₃.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l298_29850


namespace NUMINAMATH_CALUDE_expression_equals_one_l298_29869

theorem expression_equals_one (b : ℝ) (hb : b ≠ 0) :
  ∀ x : ℝ, x ≠ b ∧ x ≠ -b →
    (b / (b - x) - x / (b + x)) / (b / (b + x) + x / (b - x)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l298_29869


namespace NUMINAMATH_CALUDE_floor_plus_square_equals_72_l298_29867

theorem floor_plus_square_equals_72 : 
  ∃! (x : ℝ), x > 0 ∧ ⌊x⌋ + x^2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_square_equals_72_l298_29867


namespace NUMINAMATH_CALUDE_complex_modulus_example_l298_29864

theorem complex_modulus_example : ∃ (z : ℂ), z = 4 + 3*I ∧ Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l298_29864


namespace NUMINAMATH_CALUDE_range_of_a_l298_29856

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - 1 ≥ a^2 ∧ x - 4 < 2*a) → 
  -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l298_29856


namespace NUMINAMATH_CALUDE_cuboid_diagonal_count_l298_29852

/-- The number of unit cubes a diagonal passes through in a cuboid -/
def diagonalCubeCount (length width height : ℕ) : ℕ :=
  length + width + height - 2

/-- Theorem: The number of unit cubes a diagonal passes through in a 77 × 81 × 100 cuboid is 256 -/
theorem cuboid_diagonal_count :
  diagonalCubeCount 77 81 100 = 256 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_count_l298_29852


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l298_29814

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (x, -1)
  let b : ℝ × ℝ := (4, 2)
  parallel a b → x = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l298_29814


namespace NUMINAMATH_CALUDE_exists_non_increasing_exponential_l298_29877

theorem exists_non_increasing_exponential : 
  ∃ (a : ℝ), a > 0 ∧ ¬(∀ x y : ℝ, x < y → (a^(-x) : ℝ) < a^(-y)) :=
sorry

end NUMINAMATH_CALUDE_exists_non_increasing_exponential_l298_29877


namespace NUMINAMATH_CALUDE_A_is_half_of_B_l298_29838

def A : ℕ → ℕ
| 0 => 0
| (n + 1) => A n + (n + 1) * (2023 - n)

def B : ℕ → ℕ
| 0 => 0
| (n + 1) => B n + (n + 1) * (2024 - n)

theorem A_is_half_of_B : A 2022 = (B 2022) / 2 := by
  sorry

end NUMINAMATH_CALUDE_A_is_half_of_B_l298_29838


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l298_29880

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of unique ice cream flavors -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l298_29880


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l298_29842

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ p = 2^n - 1 ∧ is_prime p

theorem largest_mersenne_prime_under_1000 :
  (∀ p : ℕ, p < 1000 → is_mersenne_prime p → p ≤ 127) ∧
  is_mersenne_prime 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l298_29842


namespace NUMINAMATH_CALUDE_seventh_row_cans_l298_29859

/-- Represents a triangular display of cans -/
structure CanDisplay where
  rows : Nat
  firstRowCans : Nat
  increment : Nat
  totalCans : Nat

/-- Calculates the number of cans in a specific row -/
def cansInRow (d : CanDisplay) (row : Nat) : Nat :=
  d.firstRowCans + (row - 1) * d.increment

/-- Calculates the total number of cans in the display -/
def totalCans (d : CanDisplay) : Nat :=
  (d.rows * (2 * d.firstRowCans + (d.rows - 1) * d.increment)) / 2

/-- The main theorem -/
theorem seventh_row_cans (d : CanDisplay) :
  d.rows = 9 ∧ d.increment = 3 ∧ d.totalCans < 120 → cansInRow d 7 = 19 := by
  sorry

#eval cansInRow { rows := 9, firstRowCans := 1, increment := 3, totalCans := 119 } 7

end NUMINAMATH_CALUDE_seventh_row_cans_l298_29859


namespace NUMINAMATH_CALUDE_equivalent_systems_intersection_l298_29895

-- Define the type for a linear equation
def LinearEquation := ℝ → ℝ → ℝ

-- Define a system of two linear equations
structure LinearSystem :=
  (eq1 eq2 : LinearEquation)

-- Define the solution set of a linear system
def SolutionSet (sys : LinearSystem) := {p : ℝ × ℝ | sys.eq1 p.1 p.2 = 0 ∧ sys.eq2 p.1 p.2 = 0}

-- Define equivalence of two linear systems
def EquivalentSystems (sys1 sys2 : LinearSystem) :=
  SolutionSet sys1 = SolutionSet sys2

-- Define the intersection points of two lines
def IntersectionPoints (eq1 eq2 : LinearEquation) :=
  {p : ℝ × ℝ | eq1 p.1 p.2 = 0 ∧ eq2 p.1 p.2 = 0}

-- Theorem statement
theorem equivalent_systems_intersection
  (sys1 sys2 : LinearSystem)
  (h : EquivalentSystems sys1 sys2) :
  IntersectionPoints sys1.eq1 sys1.eq2 = IntersectionPoints sys2.eq1 sys2.eq2 := by
  sorry


end NUMINAMATH_CALUDE_equivalent_systems_intersection_l298_29895


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l298_29891

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l298_29891


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l298_29892

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l298_29892


namespace NUMINAMATH_CALUDE_f_sqrt5_minus1_eq_neg_half_l298_29897

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

theorem f_sqrt5_minus1_eq_neg_half
  (f : ℝ → ℝ)
  (h1 : is_monotone_increasing f)
  (h2 : ∀ x > 0, f x * f (f x + 1 / x) = 1) :
  f (Real.sqrt 5 - 1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_sqrt5_minus1_eq_neg_half_l298_29897


namespace NUMINAMATH_CALUDE_shortest_path_in_room_l298_29802

theorem shortest_path_in_room (a b h : ℝ) 
  (ha : a = 7) (hb : b = 8) (hh : h = 4) : 
  let diagonal := Real.sqrt (a^2 + b^2 + h^2)
  let floor_path := Real.sqrt ((a^2 + b^2) + h^2)
  diagonal ≥ floor_path ∧ floor_path = Real.sqrt 265 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_in_room_l298_29802


namespace NUMINAMATH_CALUDE_peaches_picked_l298_29809

theorem peaches_picked (initial_peaches final_peaches : ℕ) 
  (h1 : initial_peaches = 34)
  (h2 : final_peaches = 86) :
  final_peaches - initial_peaches = 52 := by
  sorry

end NUMINAMATH_CALUDE_peaches_picked_l298_29809


namespace NUMINAMATH_CALUDE_smallest_beneficial_discount_l298_29872

theorem smallest_beneficial_discount : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (1 - m / 100) > (1 - 20 / 100) * (1 - 20 / 100) ∨
    (1 - m / 100) > (1 - 10 / 100) * (1 - 15 / 100) ∨
    (1 - m / 100) > (1 - 8 / 100) * (1 - 8 / 100) * (1 - 8 / 100)) ∧
  (1 - n / 100) ≤ (1 - 20 / 100) * (1 - 20 / 100) ∧
  (1 - n / 100) ≤ (1 - 10 / 100) * (1 - 15 / 100) ∧
  (1 - n / 100) ≤ (1 - 8 / 100) * (1 - 8 / 100) * (1 - 8 / 100) ∧
  n = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_beneficial_discount_l298_29872


namespace NUMINAMATH_CALUDE_compound_interest_10_years_l298_29825

/-- Calculates the total amount of principal and interest after a given number of years
    with compound interest. -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating that the total amount after 10 years of compound interest
    is equal to the initial principal multiplied by (1 + rate) raised to the power of 10. -/
theorem compound_interest_10_years
  (a : ℝ) -- initial deposit
  (r : ℝ) -- annual interest rate
  (h1 : a > 0) -- assumption that initial deposit is positive
  (h2 : r > 0) -- assumption that interest rate is positive
  : compoundInterest a r 10 = a * (1 + r)^10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_10_years_l298_29825


namespace NUMINAMATH_CALUDE_concert_attendance_l298_29811

/-- Represents the number of adults attending the concert. -/
def num_adults : ℕ := sorry

/-- Represents the number of children attending the concert. -/
def num_children : ℕ := sorry

/-- The cost of an adult ticket in dollars. -/
def adult_ticket_cost : ℕ := 7

/-- The cost of a child ticket in dollars. -/
def child_ticket_cost : ℕ := 3

/-- The total revenue from ticket sales in dollars. -/
def total_revenue : ℕ := 6000

theorem concert_attendance :
  (num_children = 3 * num_adults) ∧
  (num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_revenue) →
  (num_adults + num_children = 1500) := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l298_29811


namespace NUMINAMATH_CALUDE_right_trapezoid_diagonals_bases_squares_diff_l298_29823

/-- A right trapezoid with given properties -/
structure RightTrapezoid where
  b₁ : ℝ  -- length of smaller base BC
  b₂ : ℝ  -- length of larger base AD
  h : ℝ   -- height (length of legs AB and CD)
  h_pos : h > 0
  b₁_pos : b₁ > 0
  b₂_pos : b₂ > 0
  b₁_lt_b₂ : b₁ < b₂

/-- The theorem stating that the difference of squares of diagonals equals
    the difference of squares of bases in a right trapezoid -/
theorem right_trapezoid_diagonals_bases_squares_diff
  (t : RightTrapezoid) :
  (t.h^2 + t.b₂^2) - (t.h^2 + t.b₁^2) = t.b₂^2 - t.b₁^2 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_diagonals_bases_squares_diff_l298_29823


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l298_29876

theorem necessary_not_sufficient_condition (x : ℝ) : 
  (x < 4 → x < 0) ∧ ¬(x < 0 → x < 4) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l298_29876


namespace NUMINAMATH_CALUDE_original_selling_price_with_loss_l298_29854

-- Define the selling price with 10% gain
def selling_price_with_gain : ℝ := 660

-- Define the gain percentage
def gain_percentage : ℝ := 0.1

-- Define the loss percentage
def loss_percentage : ℝ := 0.1

-- Theorem to prove
theorem original_selling_price_with_loss :
  let cost_price := selling_price_with_gain / (1 + gain_percentage)
  let selling_price_with_loss := cost_price * (1 - loss_percentage)
  selling_price_with_loss = 540 := by sorry

end NUMINAMATH_CALUDE_original_selling_price_with_loss_l298_29854


namespace NUMINAMATH_CALUDE_root_line_tangent_to_discriminant_parabola_l298_29861

/-- The discriminant parabola in the Opq plane -/
def discriminant_parabola (p q : ℝ) : Prop := p^2 - 4*q = 0

/-- The root line for a given real number a in the Opq plane -/
def root_line (a p q : ℝ) : Prop := a^2 + a*p + q = 0

/-- A line is tangent to the discriminant parabola -/
def is_tangent_line (p q : ℝ → ℝ) : Prop :=
  ∃ (x : ℝ), discriminant_parabola (p x) (q x) ∧
    ∀ (y : ℝ), y ≠ x → ¬discriminant_parabola (p y) (q y)

theorem root_line_tangent_to_discriminant_parabola :
  (∀ a : ℝ, ∃ p q : ℝ → ℝ, is_tangent_line p q ∧ ∀ x : ℝ, root_line a (p x) (q x)) ∧
  (∀ p q : ℝ → ℝ, is_tangent_line p q → ∃ a : ℝ, ∀ x : ℝ, root_line a (p x) (q x)) :=
sorry

end NUMINAMATH_CALUDE_root_line_tangent_to_discriminant_parabola_l298_29861


namespace NUMINAMATH_CALUDE_line_points_k_value_l298_29874

/-- Given a line containing the points (-1, 6), (6, k), and (20, 3), prove that k = 5 -/
theorem line_points_k_value :
  ∀ k : ℝ,
  (∃ m b : ℝ,
    (m * (-1) + b = 6) ∧
    (m * 6 + b = k) ∧
    (m * 20 + b = 3)) →
  k = 5 :=
by sorry

end NUMINAMATH_CALUDE_line_points_k_value_l298_29874


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l298_29882

theorem recurring_decimal_to_fraction : (6 / 10 : ℚ) + (23 / 99 : ℚ) = 412 / 495 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l298_29882


namespace NUMINAMATH_CALUDE_sqrt_15_bounds_l298_29813

theorem sqrt_15_bounds : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_15_bounds_l298_29813


namespace NUMINAMATH_CALUDE_correct_statements_l298_29851

theorem correct_statements :
  (∀ a : ℝ, ¬(- a < 0) → a ≤ 0) ∧
  (∀ a : ℝ, |-(a^2)| = (-a)^2) ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a / |a| + b / |b| = 0 → a * b / |a * b| = -1) ∧
  (∀ a b : ℝ, |a| = -b → |b| = b → a = b) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l298_29851


namespace NUMINAMATH_CALUDE_unique_zero_implies_m_equals_one_l298_29819

/-- A quadratic function with coefficient 1 for x^2, 2 for x, and m as the constant term -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := 4 - 4*m

theorem unique_zero_implies_m_equals_one (m : ℝ) :
  (∃! x, quadratic m x = 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_m_equals_one_l298_29819


namespace NUMINAMATH_CALUDE_system_solution_l298_29803

theorem system_solution :
  let f (x y z : ℝ) := x^2 = 2 * Real.sqrt (y^2 + 1) ∧
                       y^2 = 2 * Real.sqrt (z^2 - 1) - 2 ∧
                       z^2 = 4 * Real.sqrt (x^2 + 2) - 6
  (∀ x y z : ℝ, f x y z ↔ 
    ((x = Real.sqrt 2 ∧ y = 0 ∧ z = Real.sqrt 2) ∨
     (x = Real.sqrt 2 ∧ y = 0 ∧ z = -Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = 0 ∧ z = Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = 0 ∧ z = -Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l298_29803


namespace NUMINAMATH_CALUDE_min_value_theorem_l298_29835

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  4 * x + 1 / x^6 ≥ 5 ∧ ∃ y : ℝ, y > 0 ∧ 4 * y + 1 / y^6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l298_29835


namespace NUMINAMATH_CALUDE_fair_selection_condition_l298_29822

/-- Fairness condition for ball selection --/
def is_fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

/-- The probability of selecting same color balls --/
def prob_same_color (b c : ℕ) : ℚ :=
  (b * (b - 1) + c * (c - 1)) / ((b + c) * (b + c - 1))

/-- The probability of selecting different color balls --/
def prob_diff_color (b c : ℕ) : ℚ :=
  (2 * b * c) / ((b + c) * (b + c - 1))

/-- Theorem stating the fairness condition for ball selection --/
theorem fair_selection_condition (b c : ℕ) :
  prob_same_color b c = prob_diff_color b c ↔ is_fair_selection b c :=
sorry

end NUMINAMATH_CALUDE_fair_selection_condition_l298_29822


namespace NUMINAMATH_CALUDE_cross_section_perimeter_bounds_l298_29812

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A triangular cross-section through a vertex of a regular tetrahedron -/
structure TriangularCrossSection (a : ℝ) (t : RegularTetrahedron a) where
  perimeter : ℝ

/-- The perimeter of any triangular cross-section through a vertex of a regular tetrahedron
    with edge length a satisfies 2a < P ≤ 3a -/
theorem cross_section_perimeter_bounds (a : ℝ) (t : RegularTetrahedron a) 
  (s : TriangularCrossSection a t) : 2 * a < s.perimeter ∧ s.perimeter ≤ 3 * a := by
  sorry


end NUMINAMATH_CALUDE_cross_section_perimeter_bounds_l298_29812


namespace NUMINAMATH_CALUDE_fraction_value_l298_29883

theorem fraction_value : (2 * 3 + 4) / (2 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l298_29883


namespace NUMINAMATH_CALUDE_tile_arrangements_l298_29818

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (orange purple blue red : ℕ) : ℕ :=
  Nat.factorial (orange + purple + blue + red) /
  (Nat.factorial orange * Nat.factorial purple * Nat.factorial blue * Nat.factorial red)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 orange, 1 purple, 3 blue, and 2 red tiles is 1680 -/
theorem tile_arrangements :
  num_arrangements 2 1 3 2 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l298_29818


namespace NUMINAMATH_CALUDE_factorization_equality_l298_29849

theorem factorization_equality (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l298_29849


namespace NUMINAMATH_CALUDE_stating_num_elective_ways_l298_29821

/-- Represents the number of elective courses -/
def num_courses : ℕ := 4

/-- Represents the number of academic years -/
def num_years : ℕ := 3

/-- Represents the maximum number of courses a student can take per year -/
def max_courses_per_year : ℕ := 3

/-- 
Calculates the number of ways to distribute distinct courses over years
-/
def distribute_courses : ℕ := sorry

/-- 
Theorem stating that the number of ways to distribute the courses is 78
-/
theorem num_elective_ways : distribute_courses = 78 := by sorry

end NUMINAMATH_CALUDE_stating_num_elective_ways_l298_29821


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l298_29844

/-- Represents a conical flask with an inscribed sphere -/
structure ConicalFlask where
  base_radius : ℝ
  height : ℝ
  liquid_height : ℝ
  sphere_radius : ℝ

/-- Checks if the sphere is properly inscribed in the flask -/
def is_properly_inscribed (flask : ConicalFlask) : Prop :=
  flask.sphere_radius > 0 ∧
  flask.sphere_radius ≤ flask.base_radius ∧
  flask.sphere_radius + flask.liquid_height ≤ flask.height

/-- The main theorem about the inscribed sphere's radius -/
theorem inscribed_sphere_radius 
  (flask : ConicalFlask)
  (h_base : flask.base_radius = 15)
  (h_height : flask.height = 30)
  (h_liquid : flask.liquid_height = 10)
  (h_inscribed : is_properly_inscribed flask) :
  flask.sphere_radius = 10 :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l298_29844


namespace NUMINAMATH_CALUDE_fibonacci_13th_term_l298_29816

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_13th_term : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_13th_term_l298_29816


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l298_29890

/-- The swimming speed of a person in still water, given their performance against a current. -/
theorem swimming_speed_in_still_water (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 4)
  (h2 : distance = 8)
  (h3 : time = 2)
  (h4 : (swimming_speed - water_speed) * time = distance) :
  swimming_speed = 8 :=
sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l298_29890


namespace NUMINAMATH_CALUDE_rice_division_l298_29884

theorem rice_division (total_pounds : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_pounds = 35 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_pounds * ounces_per_pound) / num_containers = 70 :=
by sorry

end NUMINAMATH_CALUDE_rice_division_l298_29884


namespace NUMINAMATH_CALUDE_a_value_theorem_l298_29839

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x + 1

theorem a_value_theorem (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((a * Real.log x) + a) x) →
  HasDerivAt (f a) 2 1 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_theorem_l298_29839


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l298_29806

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def num_cubes_passed (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating that the number of unit cubes an internal diagonal passes through
    in a 200 × 300 × 450 rectangular solid is 700 -/
theorem internal_diagonal_cubes :
  num_cubes_passed 200 300 450 = 700 := by sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l298_29806


namespace NUMINAMATH_CALUDE_line_equation_proof_l298_29810

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2*x - 3

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem line_equation_proof :
  ∀ (A B : ℝ × ℝ),
  parabola_C A.1 A.2 →
  parabola_C B.1 B.2 →
  (A.1 + B.1) / 2 = midpoint_AB.1 →
  (A.2 + B.2) / 2 = midpoint_AB.2 →
  line_l A.1 A.2 ∧ line_l B.1 B.2 :=
by sorry


end NUMINAMATH_CALUDE_line_equation_proof_l298_29810


namespace NUMINAMATH_CALUDE_exactly_one_negative_l298_29866

theorem exactly_one_negative 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hx₃ : x₃ ≠ 0) 
  (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0) (hy₃ : y₃ ≠ 0) 
  (v₁ : ℝ) (hv₁ : v₁ = x₁ + y₁)
  (v₂ : ℝ) (hv₂ : v₂ = x₂ + y₂)
  (v₃ : ℝ) (hv₃ : v₃ = x₃ + y₃)
  (h_prod : x₁ * x₂ * x₃ = -(y₁ * y₂ * y₃))
  (h_sum_squares : x₁^2 + x₂^2 + x₃^2 = y₁^2 + y₂^2 + y₃^2)
  (h_triangle : v₁ + v₂ ≥ v₃ ∧ v₂ + v₃ ≥ v₁ ∧ v₃ + v₁ ≥ v₂)
  (h_triangle_squares : v₁^2 + v₂^2 ≥ v₃^2 ∧ v₂^2 + v₃^2 ≥ v₁^2 ∧ v₃^2 + v₁^2 ≥ v₂^2) :
  (x₁ < 0 ∨ x₂ < 0 ∨ x₃ < 0 ∨ y₁ < 0 ∨ y₂ < 0 ∨ y₃ < 0) ∧
  ¬(x₁ < 0 ∧ x₂ < 0) ∧ ¬(x₁ < 0 ∧ x₃ < 0) ∧ ¬(x₂ < 0 ∧ x₃ < 0) ∧
  ¬(y₁ < 0 ∧ y₂ < 0) ∧ ¬(y₁ < 0 ∧ y₃ < 0) ∧ ¬(y₂ < 0 ∧ y₃ < 0) ∧
  ¬(x₁ < 0 ∧ y₁ < 0) ∧ ¬(x₁ < 0 ∧ y₂ < 0) ∧ ¬(x₁ < 0 ∧ y₃ < 0) ∧
  ¬(x₂ < 0 ∧ y₁ < 0) ∧ ¬(x₂ < 0 ∧ y₂ < 0) ∧ ¬(x₂ < 0 ∧ y₃ < 0) ∧
  ¬(x₃ < 0 ∧ y₁ < 0) ∧ ¬(x₃ < 0 ∧ y₂ < 0) ∧ ¬(x₃ < 0 ∧ y₃ < 0) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_negative_l298_29866


namespace NUMINAMATH_CALUDE_correct_operations_l298_29841

theorem correct_operations (a b : ℝ) : 
  (2 * a * (3 * b) = 6 * a * b) ∧ ((-a^3)^2 = a^6) := by
  sorry

end NUMINAMATH_CALUDE_correct_operations_l298_29841


namespace NUMINAMATH_CALUDE_area_triangle_parallel_lines_circle_l298_29848

/-- Given two parallel lines with distance x between them, where one line is tangent 
    to a circle of radius R at point A and the other line intersects the circle at 
    points B and C, the area S of triangle ABC is equal to x √(2Rx - x²). -/
theorem area_triangle_parallel_lines_circle (R x : ℝ) (h : 0 < R ∧ 0 < x ∧ x < 2*R) :
  ∃ (S : ℝ), S = x * Real.sqrt (2 * R * x - x^2) := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_parallel_lines_circle_l298_29848


namespace NUMINAMATH_CALUDE_second_tap_empty_time_l298_29830

-- Define the filling time of the first tap
def fill_time : ℝ := 3

-- Define the simultaneous filling time when both taps are open
def simultaneous_fill_time : ℝ := 4.2857142857142865

-- Define the emptying time of the second tap
def empty_time : ℝ := 10

-- Theorem statement
theorem second_tap_empty_time :
  (1 / fill_time - 1 / empty_time = 1 / simultaneous_fill_time) ∧
  empty_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_tap_empty_time_l298_29830


namespace NUMINAMATH_CALUDE_octal_563_equals_base12_261_l298_29887

-- Define a function to convert from octal to decimal
def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from decimal to base 12
def decimal_to_base12 (n : ℕ) : ℕ :=
  (n / 144) * 100 + ((n / 12) % 12) * 10 + (n % 12)

-- Theorem statement
theorem octal_563_equals_base12_261 :
  decimal_to_base12 (octal_to_decimal 563) = 261 :=
sorry

end NUMINAMATH_CALUDE_octal_563_equals_base12_261_l298_29887


namespace NUMINAMATH_CALUDE_borrowing_schemes_l298_29828

theorem borrowing_schemes (n : ℕ) (m : ℕ) :
  n = 5 →  -- number of students
  m = 4 →  -- number of novels
  (∃ (schemes : ℕ), schemes = 60) :=
by
  intros hn hm
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_borrowing_schemes_l298_29828


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l298_29800

/-- The probability of an individual being selected in systematic sampling -/
theorem systematic_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 1003)
  (h2 : sample_size = 50) :
  (sample_size : ℚ) / population_size = 50 / 1003 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l298_29800


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_l298_29886

/-- Given a geometric sequence {a_n} satisfying certain conditions, 
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tan (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- {a_n} is a geometric sequence
  a 2 * a 3 * a 4 = -a 7^2 →                 -- a_2 * a_3 * a_4 = -a_7^2
  a 2 * a 3 * a 4 = -64 →                    -- a_2 * a_3 * a_4 = -64
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_l298_29886


namespace NUMINAMATH_CALUDE_town_population_l298_29832

theorem town_population (present_population : ℝ) 
  (growth_rate : ℝ) (future_population : ℝ) : 
  growth_rate = 0.1 → 
  future_population = present_population * (1 + growth_rate) → 
  future_population = 220 → 
  present_population = 200 := by
sorry

end NUMINAMATH_CALUDE_town_population_l298_29832


namespace NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l298_29853

theorem smallest_rectangle_containing_circle (r : ℝ) (h : r = 6) :
  (2 * r) * (2 * r) = 144 := by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l298_29853


namespace NUMINAMATH_CALUDE_arthur_reading_challenge_l298_29820

/-- Arthur's summer reading challenge -/
theorem arthur_reading_challenge 
  (total_goal : ℕ) 
  (book1_pages : ℕ) 
  (book1_read_percent : ℚ) 
  (book2_pages : ℕ) 
  (book2_read_fraction : ℚ) 
  (h1 : total_goal = 800)
  (h2 : book1_pages = 500)
  (h3 : book1_read_percent = 80 / 100)
  (h4 : book2_pages = 1000)
  (h5 : book2_read_fraction = 1 / 5)
  : ℕ := by
  sorry

#check arthur_reading_challenge

end NUMINAMATH_CALUDE_arthur_reading_challenge_l298_29820


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l298_29873

theorem quadratic_real_solutions (m : ℝ) : 
  (∀ x : ℝ, x^2 + x + m = 0 → ∃ y : ℝ, y^2 + y + m = 0) ∧ 
  (∃ n : ℝ, n ≥ 1/4 ∧ ∃ z : ℝ, z^2 + z + n = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l298_29873


namespace NUMINAMATH_CALUDE_first_rope_length_l298_29834

/-- Represents the lengths of ropes Tony found -/
structure Ropes where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Calculates the total length of ropes before tying -/
def total_length (r : Ropes) : ℝ :=
  r.first + r.second + r.third + r.fourth + r.fifth

/-- Calculates the length lost due to knots -/
def knot_loss (num_ropes : ℕ) (loss_per_knot : ℝ) : ℝ :=
  (num_ropes - 1 : ℝ) * loss_per_knot

/-- Theorem stating that given the conditions, the first rope Tony found is 20 feet long -/
theorem first_rope_length
  (r : Ropes)
  (h1 : r.second = 2)
  (h2 : r.third = 2)
  (h3 : r.fourth = 2)
  (h4 : r.fifth = 7)
  (h5 : total_length r - knot_loss 5 1.2 = 35) :
  r.first = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_rope_length_l298_29834


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l298_29894

theorem problem_1 (m : ℤ) (h : m = -3) : 4 * (m + 1)^2 - (2*m + 5) * (2*m - 5) = 5 := by sorry

theorem problem_2 (x : ℚ) (h : x = 2) : (x^2 - 1) / (x^2 + 2*x) / ((x - 1) / x) = 3/4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l298_29894


namespace NUMINAMATH_CALUDE_hidden_sea_portion_l298_29845

/-- Represents the composition of the landscape visible from an airplane window -/
structure Landscape where
  cloud : ℚ  -- Fraction of landscape covered by cloud
  island : ℚ  -- Fraction of landscape occupied by island
  sea : ℚ    -- Fraction of landscape occupied by sea

/-- The conditions of the landscape as described in the problem -/
def airplane_view : Landscape where
  cloud := 1/2
  island := 1/3
  sea := 2/3

theorem hidden_sea_portion (L : Landscape) 
  (h1 : L.cloud = 1/2)
  (h2 : L.island = 1/3)
  (h3 : L.cloud + L.island + L.sea = 1) :
  L.cloud * L.sea = 5/12 := by
  sorry

#check hidden_sea_portion

end NUMINAMATH_CALUDE_hidden_sea_portion_l298_29845


namespace NUMINAMATH_CALUDE_bug_flower_consumption_l298_29858

theorem bug_flower_consumption (total_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) :
  total_bugs = 3 →
  total_flowers = 6 →
  total_flowers = total_bugs * flowers_per_bug →
  flowers_per_bug = 2 := by
  sorry

end NUMINAMATH_CALUDE_bug_flower_consumption_l298_29858


namespace NUMINAMATH_CALUDE_move_right_two_units_l298_29826

/-- Moving a point 2 units to the right in a Cartesian coordinate system -/
theorem move_right_two_units (initial_x initial_y : ℝ) :
  let initial_point := (initial_x, initial_y)
  let final_point := (initial_x + 2, initial_y)
  initial_point = (1, 1) → final_point = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_move_right_two_units_l298_29826


namespace NUMINAMATH_CALUDE_discount_age_limit_l298_29804

/-- Represents the age limit for the discount at an amusement park. -/
def age_limit : ℕ := 10

/-- Represents the regular ticket cost. -/
def regular_ticket_cost : ℕ := 109

/-- Represents the discount amount for children. -/
def child_discount : ℕ := 5

/-- Represents the number of adults in the family. -/
def num_adults : ℕ := 2

/-- Represents the number of children in the family. -/
def num_children : ℕ := 2

/-- Represents the ages of the children in the family. -/
def children_ages : List ℕ := [6, 10]

/-- Represents the amount paid by the family. -/
def amount_paid : ℕ := 500

/-- Represents the change received by the family. -/
def change_received : ℕ := 74

/-- Theorem stating that the age limit for the discount is 10 years old. -/
theorem discount_age_limit : 
  (∀ (age : ℕ), age ∈ children_ages → age ≤ age_limit) ∧
  (amount_paid - change_received = 
    num_adults * regular_ticket_cost + 
    num_children * (regular_ticket_cost - child_discount)) →
  age_limit = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_age_limit_l298_29804


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l298_29831

/-- Given a line L1 with equation 3x - 4y + 6 = 0 and a point P(4, -1),
    the line L2 passing through P and perpendicular to L1 has equation 4x + 3y - 13 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 4 * y + 6 = 0
  let P : ℝ × ℝ := (4, -1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y - 13 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(4/3) * (x - P.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → 
    ((x₂ - x₁) * (3/4) + (y₂ - y₁) * (-1) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l298_29831


namespace NUMINAMATH_CALUDE_perfect_square_sum_l298_29836

theorem perfect_square_sum (a b c d : ℤ) (h : a + b + c + d = 0) :
  2 * (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = (a^2 + b^2 + c^2 + d^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l298_29836


namespace NUMINAMATH_CALUDE_three_face_painted_count_l298_29833

/-- Represents a cuboid made of small cubes -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the state of the cuboid after modifications -/
structure ModifiedCuboid extends Cuboid where
  removed_cubes : ℕ
  surface_painted : Bool

/-- Counts the number of small cubes with three painted faces -/
def count_three_face_painted (c : ModifiedCuboid) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem three_face_painted_count 
  (c : ModifiedCuboid) 
  (h1 : c.length = 12 ∧ c.width = 3 ∧ c.height = 6)
  (h2 : c.removed_cubes = 3)
  (h3 : c.surface_painted = true) :
  count_three_face_painted c = 8 :=
sorry

end NUMINAMATH_CALUDE_three_face_painted_count_l298_29833


namespace NUMINAMATH_CALUDE_multiplier_value_l298_29875

def f (x : ℝ) : ℝ := 3 * x - 5

theorem multiplier_value (x : ℝ) (h : x = 3) :
  ∃ m : ℝ, m * f x - 10 = f (x - 2) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l298_29875


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l298_29863

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 20) :
  1 / x + 1 / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l298_29863


namespace NUMINAMATH_CALUDE_inequality_proof_l298_29817

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l298_29817


namespace NUMINAMATH_CALUDE_line_equation_proof_l298_29868

/-- Given a line described by the vector equation (3, -4) · ((x, y) - (-2, 8)) = 0,
    prove that its slope-intercept form y = mx + b has m = 3/4 and b = 19/2. -/
theorem line_equation_proof :
  let vector_eq := fun (x y : ℝ) => 3 * (x + 2) + (-4) * (y - 8) = 0
  ∃ m b : ℝ, (∀ x y : ℝ, vector_eq x y ↔ y = m * x + b) ∧ m = 3/4 ∧ b = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l298_29868


namespace NUMINAMATH_CALUDE_trig_identity_l298_29879

theorem trig_identity : 
  Real.sin (47 * π / 180) * Real.sin (103 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (77 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l298_29879


namespace NUMINAMATH_CALUDE_current_speed_l298_29896

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 10) :
  ∃ (current_speed : ℝ), current_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l298_29896


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l298_29808

/-- Given a circle C' with equation x^2 - 4y - 15 = -y^2 + 12x + 27,
    prove that its center (p, q) and radius s satisfy p + q + s = 8 + √82 -/
theorem circle_center_radius_sum (x y p q s : ℝ) : 
  (∀ x y, x^2 - 4*y - 15 = -y^2 + 12*x + 27) →
  (∀ x y, (x - p)^2 + (y - q)^2 = s^2) →
  p + q + s = 8 + Real.sqrt 82 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l298_29808


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l298_29801

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∃ (x : ℤ), x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 102 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l298_29801


namespace NUMINAMATH_CALUDE_solve_for_t_l298_29857

theorem solve_for_t (a b d x y t : ℕ) 
  (h1 : a + b = x)
  (h2 : x + d = t)
  (h3 : t + a = y)
  (h4 : b + d + y = 16)
  (ha : a > 0)
  (hb : b > 0)
  (hd : d > 0)
  (hx : x > 0)
  (hy : y > 0)
  (ht : t > 0) :
  t = 8 := by
sorry


end NUMINAMATH_CALUDE_solve_for_t_l298_29857
