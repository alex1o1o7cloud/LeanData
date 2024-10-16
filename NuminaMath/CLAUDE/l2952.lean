import Mathlib

namespace NUMINAMATH_CALUDE_peters_to_amandas_flower_ratio_l2952_295221

theorem peters_to_amandas_flower_ratio : 
  ∀ (amanda_flowers peter_flowers peter_flowers_after : ℕ),
    amanda_flowers = 20 →
    peter_flowers = peter_flowers_after + 15 →
    peter_flowers_after = 45 →
    peter_flowers = 3 * amanda_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_peters_to_amandas_flower_ratio_l2952_295221


namespace NUMINAMATH_CALUDE_max_value_under_constraint_l2952_295209

/-- The objective function to be maximized -/
def f (x y : ℝ) : ℝ := 8 * x^2 + 9 * x * y + 18 * y^2 + 2 * x + 3 * y

/-- The constraint function -/
def g (x y : ℝ) : ℝ := 4 * x^2 + 9 * y^2 - 8

/-- Theorem stating that the maximum value of f subject to the constraint g = 0 is 26 -/
theorem max_value_under_constraint : 
  ∃ (x y : ℝ), g x y = 0 ∧ f x y = 26 ∧ ∀ (x' y' : ℝ), g x' y' = 0 → f x' y' ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_max_value_under_constraint_l2952_295209


namespace NUMINAMATH_CALUDE_g_difference_l2952_295247

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define the g function
def g (n : ℕ) : ℚ :=
  (sigma n : ℚ) / n

-- Theorem statement
theorem g_difference : g 432 - g 216 = 5 / 54 := by sorry

end NUMINAMATH_CALUDE_g_difference_l2952_295247


namespace NUMINAMATH_CALUDE_vector_decomposition_l2952_295213

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![-1, 7, 0]
def p : Fin 3 → ℝ := ![0, 3, 1]
def q : Fin 3 → ℝ := ![1, -1, 2]
def r : Fin 3 → ℝ := ![2, -1, 0]

/-- Theorem stating the decomposition of x in terms of p and q -/
theorem vector_decomposition : x = 2 • p - q := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2952_295213


namespace NUMINAMATH_CALUDE_zero_point_condition_l2952_295253

theorem zero_point_condition (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, 3 * a * x + 1 - 2 * a = 0) ↔ 
  (a < -1 ∨ a > 1/5) := by sorry

end NUMINAMATH_CALUDE_zero_point_condition_l2952_295253


namespace NUMINAMATH_CALUDE_oplus_calculation_l2952_295292

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a * b + a + b + 1

-- State the theorem
theorem oplus_calculation : oplus (-3) (oplus 4 2) = -32 := by
  sorry

end NUMINAMATH_CALUDE_oplus_calculation_l2952_295292


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l2952_295258

/-- Represents the sum of the first k terms of a geometric sequence -/
noncomputable def S (a₁ q : ℝ) (k : ℕ) : ℝ :=
  if q = 1 then k * a₁ else a₁ * (1 - q^k) / (1 - q)

theorem geometric_sequence_sum_problem
  (a₁ q : ℝ)
  (h_pos : ∀ n : ℕ, 0 < a₁ * q^n)
  (h_Sn : S a₁ q n = 2)
  (h_S3n : S a₁ q (3*n) = 14) :
  S a₁ q (4*n) = 30 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l2952_295258


namespace NUMINAMATH_CALUDE_decimal_expansion_non_periodic_length_l2952_295211

/-- The length of the non-periodic part of the decimal expansion of 1/n -/
def nonPeriodicLength (n : ℕ) : ℕ :=
  max (Nat.factorization n 2) (Nat.factorization n 5)

/-- Theorem stating that for any natural number n > 1, the length of the non-periodic part
    of the decimal expansion of 1/n is equal to max[v₂(n), v₅(n)] -/
theorem decimal_expansion_non_periodic_length (n : ℕ) (h : n > 1) :
  nonPeriodicLength n = max (Nat.factorization n 2) (Nat.factorization n 5) := by
  sorry

#check decimal_expansion_non_periodic_length

end NUMINAMATH_CALUDE_decimal_expansion_non_periodic_length_l2952_295211


namespace NUMINAMATH_CALUDE_foreign_language_teachers_l2952_295219

/-- The number of teachers who do not teach English, Japanese, or French -/
theorem foreign_language_teachers (total : ℕ) (english : ℕ) (japanese : ℕ) (french : ℕ)
  (eng_jap : ℕ) (eng_fre : ℕ) (jap_fre : ℕ) (all_three : ℕ) :
  total = 120 →
  english = 50 →
  japanese = 45 →
  french = 40 →
  eng_jap = 15 →
  eng_fre = 10 →
  jap_fre = 8 →
  all_three = 4 →
  total - (english + japanese + french - eng_jap - eng_fre - jap_fre + all_three) = 14 :=
by sorry

end NUMINAMATH_CALUDE_foreign_language_teachers_l2952_295219


namespace NUMINAMATH_CALUDE_vector_b_determination_l2952_295204

def vector_a : ℝ × ℝ := (4, 3)

theorem vector_b_determination (b : ℝ × ℝ) 
  (h1 : (b.1 * vector_a.1 + b.2 * vector_a.2) / Real.sqrt (vector_a.1^2 + vector_a.2^2) = 4)
  (h2 : b.1 = 2) :
  b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_determination_l2952_295204


namespace NUMINAMATH_CALUDE_largest_coin_distribution_exists_largest_distribution_l2952_295223

theorem largest_coin_distribution (n : ℕ) : n < 150 → n % 15 = 3 → n ≤ 138 := by
  sorry

theorem exists_largest_distribution : ∃ n : ℕ, n < 150 ∧ n % 15 = 3 ∧ n = 138 := by
  sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_exists_largest_distribution_l2952_295223


namespace NUMINAMATH_CALUDE_nonzero_digits_count_l2952_295256

def original_fraction : ℚ := 120 / (2^5 * 5^10)

def decimal_result : ℝ := (original_fraction : ℝ) - 0.000001

def count_nonzero_digits (x : ℝ) : ℕ :=
  sorry -- Implementation of counting non-zero digits after decimal point

theorem nonzero_digits_count :
  count_nonzero_digits decimal_result = 3 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_digits_count_l2952_295256


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2952_295239

theorem prime_equation_solution (p q r : ℕ) (A : ℕ) 
  (h_prime_p : Nat.Prime p) 
  (h_prime_q : Nat.Prime q) 
  (h_prime_r : Nat.Prime r) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h_equation : 2*p*q*r + 50*p*q = 7*p*q*r + 55*p*r ∧ 
                7*p*q*r + 55*p*r = 8*p*q*r + 12*q*r ∧
                8*p*q*r + 12*q*r = A)
  (h_positive : A > 0) : 
  A = 1980 := by
sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2952_295239


namespace NUMINAMATH_CALUDE_foldPointSetArea_l2952_295286

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of a right triangle ABC with AB = 45, AC = 90 -/
def rightTriangle : Triangle :=
  { A := { x := 0, y := 0 }
  , B := { x := 45, y := 0 }
  , C := { x := 0, y := 90 }
  }

/-- A point P is a fold point if creases formed when A, B, and C are folded onto P do not intersect inside the triangle -/
def isFoldPoint (P : Point) (T : Triangle) : Prop := sorry

/-- The set of all fold points for a given triangle -/
def foldPointSet (T : Triangle) : Set Point :=
  {P | isFoldPoint P T}

/-- The area of a set of points -/
def areaOfSet (S : Set Point) : ℝ := sorry

/-- Theorem: The area of the fold point set for the right triangle is 506.25π - 607.5√3 -/
theorem foldPointSetArea :
  areaOfSet (foldPointSet rightTriangle) = 506.25 * Real.pi - 607.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_foldPointSetArea_l2952_295286


namespace NUMINAMATH_CALUDE_union_of_intervals_l2952_295297

open Set

theorem union_of_intervals (A B : Set ℝ) :
  A = {x : ℝ | -1 < x ∧ x < 4} →
  B = {x : ℝ | 2 < x ∧ x < 5} →
  A ∪ B = {x : ℝ | -1 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_intervals_l2952_295297


namespace NUMINAMATH_CALUDE_games_in_division_l2952_295285

/-- Represents a baseball league with the given conditions -/
structure BaseballLeague where
  P : ℕ  -- Number of games played against each team in own division
  Q : ℕ  -- Number of games played against each team in other divisions
  p_gt_3q : P > 3 * Q
  q_gt_3 : Q > 3
  total_games : 2 * P + 6 * Q = 78

/-- Theorem stating that each team plays 54 games within its own division -/
theorem games_in_division (league : BaseballLeague) : 2 * league.P = 54 := by
  sorry

end NUMINAMATH_CALUDE_games_in_division_l2952_295285


namespace NUMINAMATH_CALUDE_ice_cream_sales_l2952_295257

theorem ice_cream_sales (chocolate : ℕ) (mango : ℕ) 
  (h1 : chocolate = 50) 
  (h2 : mango = 54) : 
  chocolate + mango - (chocolate * 3 / 5 + mango * 2 / 3) = 38 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l2952_295257


namespace NUMINAMATH_CALUDE_triangle_problem_l2952_295273

noncomputable section

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector m in the problem -/
def m (t : Triangle) : ℝ × ℝ := (Real.cos t.A, Real.sin t.A)

/-- Vector n in the problem -/
def n (t : Triangle) : ℝ × ℝ := (Real.cos t.A, -Real.sin t.A)

/-- Dot product of vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
    (h_acute : 0 < t.A ∧ t.A < π / 2)
    (h_dot : dot_product (m t) (n t) = 1 / 2)
    (h_a : t.a = Real.sqrt 5) :
  t.A = π / 6 ∧ 
  Real.arccos (dot_product (m t) (n t) / (Real.sqrt ((m t).1^2 + (m t).2^2) * Real.sqrt ((n t).1^2 + (n t).2^2))) = π / 3 ∧
  (let max_area := (10 + 5 * Real.sqrt 3) / 4
   ∀ b c, t.b = b → t.c = c → 1 / 2 * b * c * Real.sin t.A ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2952_295273


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_P_x_eq_x_l2952_295228

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Property: for any integers a and b, b - a divides P(b) - P(a) -/
def IntegerCoefficientProperty (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, (b - a) ∣ (P b - P a)

theorem no_integer_solutions_for_P_x_eq_x
  (P : IntPolynomial)
  (h_int_coeff : IntegerCoefficientProperty P)
  (h_P_3 : P 3 = 4)
  (h_P_4 : P 4 = 3) :
  ¬∃ x : ℤ, P x = x :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_P_x_eq_x_l2952_295228


namespace NUMINAMATH_CALUDE_shoes_sold_l2952_295274

theorem shoes_sold (large medium small left : ℕ) 
  (h_large : large = 22)
  (h_medium : medium = 50)
  (h_small : small = 24)
  (h_left : left = 13) :
  large + medium + small - left = 83 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_l2952_295274


namespace NUMINAMATH_CALUDE_dam_building_time_with_reduced_workers_l2952_295248

/-- The time taken to build a dam given the number of workers and their work rate -/
def build_time (workers : ℕ) (rate : ℚ) : ℚ :=
  1 / (workers * rate)

/-- The work rate of a single worker -/
def worker_rate (initial_workers : ℕ) (initial_time : ℚ) : ℚ :=
  1 / (initial_workers * initial_time)

theorem dam_building_time_with_reduced_workers 
  (initial_workers : ℕ) 
  (initial_time : ℚ) 
  (new_workers : ℕ) : 
  initial_workers = 60 → 
  initial_time = 5 → 
  new_workers = 40 → 
  build_time new_workers (worker_rate initial_workers initial_time) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_dam_building_time_with_reduced_workers_l2952_295248


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2952_295283

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (5 * y + 15) = 15 → y = 42 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2952_295283


namespace NUMINAMATH_CALUDE_max_fourth_root_sum_max_fourth_root_sum_achievable_l2952_295269

theorem max_fourth_root_sum (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a + b + c + d = 1) : 
  (abcd : ℝ)^(1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ)^(1/4) ≤ 1 :=
by sorry

theorem max_fourth_root_sum_achievable : 
  ∃ (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a + b + c + d = 1 ∧ 
    (abcd : ℝ)^(1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ)^(1/4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_root_sum_max_fourth_root_sum_achievable_l2952_295269


namespace NUMINAMATH_CALUDE_divisor_count_equality_l2952_295227

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem: For all positive integers n and k, there exists a positive integer s
    such that the number of positive divisors of sn equals the number of positive divisors of sk
    if and only if n does not divide k and k does not divide n -/
theorem divisor_count_equality (n k : ℕ+) :
  (∃ s : ℕ+, num_divisors (s * n) = num_divisors (s * k)) ↔ (¬(n ∣ k) ∧ ¬(k ∣ n)) :=
sorry

end NUMINAMATH_CALUDE_divisor_count_equality_l2952_295227


namespace NUMINAMATH_CALUDE_average_steps_needed_l2952_295282

def goal : ℕ := 10000
def days : ℕ := 9
def remaining_days : ℕ := 3

def steps_walked : List ℕ := [10200, 10400, 9400, 9100, 8300, 9200, 8900, 9500]

def total_goal : ℕ := goal * days

def steps_walked_so_far : ℕ := steps_walked.sum

def remaining_steps : ℕ := total_goal - steps_walked_so_far

theorem average_steps_needed (h : steps_walked.length = days - remaining_days) :
  remaining_steps / remaining_days = 5000 := by
  sorry

end NUMINAMATH_CALUDE_average_steps_needed_l2952_295282


namespace NUMINAMATH_CALUDE_ellipse_area_theorem_l2952_295201

/-- Represents an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  a_gt_b : a > b
  b_pos : b > 0
  vertex_y : b = 1
  eccentricity : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2

/-- Represents a line passing through the right focus of the ellipse -/
structure FocusLine (e : Ellipse) where
  k : ℝ  -- Slope of the line

/-- Represents two points on the ellipse intersected by the focus line -/
structure IntersectionPoints (e : Ellipse) (l : FocusLine e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_ellipse_A : A.1^2 / e.a^2 + A.2^2 / e.b^2 = 1
  on_ellipse_B : B.1^2 / e.a^2 + B.2^2 / e.b^2 = 1
  on_line_A : A.2 = l.k * (A.1 - Real.sqrt (e.a^2 - e.b^2))
  on_line_B : B.2 = l.k * (B.1 - Real.sqrt (e.a^2 - e.b^2))
  perpendicular : A.1 * B.1 + A.2 * B.2 = 0  -- OA ⊥ OB condition

/-- Main theorem statement -/
theorem ellipse_area_theorem (e : Ellipse) (l : FocusLine e) (p : IntersectionPoints e l) :
  e.a^2 = 2 ∧ 
  (abs (p.A.1 - p.B.1) * abs (p.A.2 - p.B.2) / 2 = 2 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_theorem_l2952_295201


namespace NUMINAMATH_CALUDE_cubes_fill_box_completely_l2952_295203

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along a dimension -/
def cubesAlongDimension (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (d : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  (cubesAlongDimension d.length cubeSize) *
  (cubesAlongDimension d.width cubeSize) *
  (cubesAlongDimension d.height cubeSize)

/-- Calculates the volume occupied by the cubes -/
def cubesVolume (d : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  totalCubes d cubeSize * (cubeSize ^ 3)

/-- Theorem: The volume occupied by 4-inch cubes in the given box is 100% of the box's volume -/
theorem cubes_fill_box_completely (d : BoxDimensions) (h1 : d.length = 16) (h2 : d.width = 12) (h3 : d.height = 8) :
  cubesVolume d 4 = boxVolume d := by
  sorry

#eval cubesVolume ⟨16, 12, 8⟩ 4
#eval boxVolume ⟨16, 12, 8⟩

end NUMINAMATH_CALUDE_cubes_fill_box_completely_l2952_295203


namespace NUMINAMATH_CALUDE_milburg_population_l2952_295210

/-- The total population of Milburg is the sum of grown-ups and children. -/
theorem milburg_population :
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l2952_295210


namespace NUMINAMATH_CALUDE_petrol_expense_percentage_l2952_295220

/-- Represents the problem of calculating the percentage of income spent on petrol --/
theorem petrol_expense_percentage
  (total_income : ℝ)
  (petrol_expense : ℝ)
  (rent_expense : ℝ)
  (rent_percentage : ℝ)
  (h1 : petrol_expense = 300)
  (h2 : rent_expense = 210)
  (h3 : rent_percentage = 30)
  (h4 : rent_expense = (rent_percentage / 100) * (total_income - petrol_expense)) :
  (petrol_expense / total_income) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_petrol_expense_percentage_l2952_295220


namespace NUMINAMATH_CALUDE_thousand_factorization_sum_l2952_295289

/-- Checks if a positive integer contains zero in its decimal representation -/
def containsZero (n : Nat) : Bool :=
  n.repr.contains '0'

/-- Theorem stating the existence of two positive integers satisfying the given conditions -/
theorem thousand_factorization_sum :
  ∃ (a b : Nat), a * b = 1000 ∧ ¬containsZero a ∧ ¬containsZero b ∧ a + b = 133 := by
  sorry

end NUMINAMATH_CALUDE_thousand_factorization_sum_l2952_295289


namespace NUMINAMATH_CALUDE_school_pupils_count_l2952_295233

theorem school_pupils_count (girls boys total : ℕ) : 
  girls = 692 → 
  girls = boys + 458 → 
  total = girls + boys → 
  total = 926 := by
sorry

end NUMINAMATH_CALUDE_school_pupils_count_l2952_295233


namespace NUMINAMATH_CALUDE_evaluate_expression_l2952_295276

theorem evaluate_expression : 3000 * (3000^1500 + 3000^1500) = 2 * 3000^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2952_295276


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_l2952_295262

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (b g : ℕ) (hb : b = 8) (hg : g = 12) :
  let total := b + g
  let prob_bg := (b : ℚ) / total * (g : ℚ) / (total - 1)
  let prob_pair := 2 * prob_bg
  let num_pairs := total - 1
  num_pairs * prob_pair = 912 / 95 := by sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_l2952_295262


namespace NUMINAMATH_CALUDE_prism_volume_l2952_295229

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) 
  (h₁ : a * b = 18) 
  (h₂ : b * c = 12) 
  (h₃ : a * c = 8) : 
  a * b * c = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2952_295229


namespace NUMINAMATH_CALUDE_sum_abs_coefficients_f6_l2952_295230

def polynomial_sequence : ℕ → (ℝ → ℝ) 
  | 0 => λ x => 1
  | n + 1 => λ x => (x^2 - 1) * (polynomial_sequence n x) - 2*x

def sum_abs_coefficients (f : ℝ → ℝ) : ℝ := sorry

theorem sum_abs_coefficients_f6 : 
  sum_abs_coefficients (polynomial_sequence 6) = 190 := by sorry

end NUMINAMATH_CALUDE_sum_abs_coefficients_f6_l2952_295230


namespace NUMINAMATH_CALUDE_circle_travel_in_triangle_l2952_295259

/-- The distance traveled by the center of a circle rolling inside a triangle -/
def circle_travel_distance (a b c r : ℝ) : ℝ :=
  (a - 2 * r) + (b - 2 * r) + (c - 2 * r)

/-- Theorem: The distance traveled by the center of a circle with radius 2
    rolling inside a 6-8-10 triangle is 8 -/
theorem circle_travel_in_triangle :
  circle_travel_distance 6 8 10 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_travel_in_triangle_l2952_295259


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l2952_295225

theorem perfect_square_quadratic (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 18 * x + 9 = (r * x + s)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l2952_295225


namespace NUMINAMATH_CALUDE_division_simplification_l2952_295264

theorem division_simplification (x y : ℝ) (h : x ≠ 0) :
  6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2952_295264


namespace NUMINAMATH_CALUDE_dichromate_molecular_weight_l2952_295255

/-- Given that the molecular weight of 9 moles of Dichromate is 2664 g/mol,
    prove that the molecular weight of one mole of Dichromate is 296 g/mol. -/
theorem dichromate_molecular_weight :
  let mw_9_moles : ℝ := 2664 -- molecular weight of 9 moles in g/mol
  let num_moles : ℝ := 9 -- number of moles
  mw_9_moles / num_moles = 296 := by sorry

end NUMINAMATH_CALUDE_dichromate_molecular_weight_l2952_295255


namespace NUMINAMATH_CALUDE_chess_club_officers_l2952_295206

def choose_officers (n : ℕ) (k : ℕ) (special_pair : ℕ) : ℕ :=
  (n - special_pair).choose k * (n - special_pair - k + 1).choose (k - 1) +
  special_pair * (special_pair - 1) * (n - special_pair)

theorem chess_club_officers :
  choose_officers 24 3 2 = 9372 :=
sorry

end NUMINAMATH_CALUDE_chess_club_officers_l2952_295206


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l2952_295251

/-- Given a cube with side length 6, the volume of the tetrahedron formed by any vertex
    and the three vertices connected to that vertex by edges of the cube is 36. -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (tetrahedron_volume : ℝ) :
  cube_side_length = 6 →
  tetrahedron_volume = (1 / 3) * (1 / 2 * cube_side_length * cube_side_length) * cube_side_length →
  tetrahedron_volume = 36 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l2952_295251


namespace NUMINAMATH_CALUDE_thursday_return_count_l2952_295216

/-- Calculates the number of books brought back on Thursday given the initial
    number of books, books taken out on Tuesday and Friday, and the final
    number of books in the library. -/
def books_brought_back (initial : ℕ) (taken_tuesday : ℕ) (taken_friday : ℕ) (final : ℕ) : ℕ :=
  initial - taken_tuesday + taken_friday - final

theorem thursday_return_count :
  books_brought_back 235 227 35 29 = 56 := by
  sorry

end NUMINAMATH_CALUDE_thursday_return_count_l2952_295216


namespace NUMINAMATH_CALUDE_article_cost_price_l2952_295298

theorem article_cost_price (marked_price : ℝ) (cost_price : ℝ) : 
  marked_price = 112.5 →
  0.95 * marked_price = 1.25 * cost_price →
  cost_price = 85.5 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l2952_295298


namespace NUMINAMATH_CALUDE_fishing_theorem_l2952_295208

/-- The total number of fish caught by Leo and Agrey -/
def total_fish (leo_fish : ℕ) (agrey_fish : ℕ) : ℕ :=
  leo_fish + agrey_fish

/-- Theorem: Given Leo caught 40 fish and Agrey caught 20 more fish than Leo,
    the total number of fish they caught together is 100. -/
theorem fishing_theorem :
  let leo_fish : ℕ := 40
  let agrey_fish : ℕ := leo_fish + 20
  total_fish leo_fish agrey_fish = 100 := by
sorry

end NUMINAMATH_CALUDE_fishing_theorem_l2952_295208


namespace NUMINAMATH_CALUDE_distance_AB_is_600_l2952_295279

/-- The distance between city A and city B -/
def distance_AB : ℝ := 600

/-- The time taken by Eddy to travel from A to B -/
def time_Eddy : ℝ := 3

/-- The time taken by Freddy to travel from A to C -/
def time_Freddy : ℝ := 4

/-- The distance between city A and city C -/
def distance_AC : ℝ := 460

/-- The ratio of Eddy's average speed to Freddy's average speed -/
def speed_ratio : ℝ := 1.7391304347826086

theorem distance_AB_is_600 :
  distance_AB = (speed_ratio * distance_AC * time_Eddy) / time_Freddy :=
sorry

end NUMINAMATH_CALUDE_distance_AB_is_600_l2952_295279


namespace NUMINAMATH_CALUDE_runner_stops_at_d_l2952_295242

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
  | A : Quarter
  | B : Quarter
  | C : Quarter
  | D : Quarter

/-- Represents a point on the circular track -/
structure TrackPoint where
  position : ℝ  -- position in feet from the start point
  quarter : Quarter

/-- The circular track -/
structure Track where
  circumference : ℝ
  start_point : TrackPoint

/-- Calculates the final position after running a given distance -/
def final_position (track : Track) (distance : ℝ) : TrackPoint :=
  sorry

theorem runner_stops_at_d (track : Track) (distance : ℝ) :
  track.circumference = 100 →
  distance = 10000 →
  track.start_point.quarter = Quarter.A →
  (final_position track distance).quarter = Quarter.D :=
sorry

end NUMINAMATH_CALUDE_runner_stops_at_d_l2952_295242


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_range_l2952_295215

/-- A point P(x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P as a function of a -/
def x_coord (a : ℝ) : ℝ := 2 * a + 1

/-- The y-coordinate of point P as a function of a -/
def y_coord (a : ℝ) : ℝ := 1 - a

/-- Theorem: If P(2a+1, 1-a) is in the second quadrant, then a < -1/2 -/
theorem point_in_second_quadrant_implies_a_range (a : ℝ) :
  in_second_quadrant (x_coord a) (y_coord a) → a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_range_l2952_295215


namespace NUMINAMATH_CALUDE_expression_evaluation_l2952_295294

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  ((a + 3) / (a + 1)) * ((b - 2) / (b - 3)) * ((c + 9) / (c + 7)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2952_295294


namespace NUMINAMATH_CALUDE_min_value_theorem_l2952_295214

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 2 / (y + 3) = 1 / 4) :
  2 * x + 3 * y ≥ 16 * Real.sqrt 3 - 16 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 3) + 2 / (y₀ + 3) = 1 / 4 ∧
    2 * x₀ + 3 * y₀ = 16 * Real.sqrt 3 - 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2952_295214


namespace NUMINAMATH_CALUDE_door_width_calculation_l2952_295284

/-- Calculates the width of a door given room dimensions and whitewashing costs -/
theorem door_width_calculation (room_length room_width room_height : ℝ)
  (door_height : ℝ) (window_width window_height : ℝ) (num_windows : ℕ)
  (cost_per_sqft total_cost : ℝ) : 
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_height = 6 ∧ window_width = 4 ∧ window_height = 3 ∧
  num_windows = 3 ∧ cost_per_sqft = 9 ∧ total_cost = 8154 →
  ∃ (door_width : ℝ),
    (2 * (room_length * room_height + room_width * room_height) - 
     (door_height * door_width + num_windows * window_width * window_height)) * cost_per_sqft = total_cost ∧
    door_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_door_width_calculation_l2952_295284


namespace NUMINAMATH_CALUDE_geometric_progression_p_l2952_295245

theorem geometric_progression_p (p : ℝ) : 
  p > 0 ∧ 
  (3 * Real.sqrt p) ^ 2 = (-p - 8) * (p - 7) ↔ 
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_p_l2952_295245


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l2952_295212

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2*x - 4 ≥ x - 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l2952_295212


namespace NUMINAMATH_CALUDE_slope_condition_implies_coefficient_bound_l2952_295260

/-- Given two distinct points on a linear function, if the slope between them is negative, then the coefficient of x in the function is less than 1. -/
theorem slope_condition_implies_coefficient_bound
  (x₁ x₂ y₁ y₂ a : ℝ)
  (h_distinct : x₁ ≠ x₂)
  (h_on_graph₁ : y₁ = (a - 1) * x₁ + 1)
  (h_on_graph₂ : y₂ = (a - 1) * x₂ + 1)
  (h_slope_neg : (y₁ - y₂) / (x₁ - x₂) < 0) :
  a < 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_condition_implies_coefficient_bound_l2952_295260


namespace NUMINAMATH_CALUDE_jills_shopping_trip_tax_percentage_l2952_295296

/-- Represents the spending and tax information for a shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ

/-- Calculates the total tax as a percentage of the total amount spent (excluding taxes) -/
def totalTaxPercentage (trip : ShoppingTrip) : ℝ :=
  (trip.clothing_percent * trip.clothing_tax_rate +
   trip.food_percent * trip.food_tax_rate +
   trip.other_percent * trip.other_tax_rate) * 100

/-- Theorem stating that the total tax percentage for Jill's shopping trip is 4.40% -/
theorem jills_shopping_trip_tax_percentage :
  let trip : ShoppingTrip := {
    clothing_percent := 0.50,
    food_percent := 0.20,
    other_percent := 0.30,
    clothing_tax_rate := 0.04,
    food_tax_rate := 0,
    other_tax_rate := 0.08
  }
  totalTaxPercentage trip = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_jills_shopping_trip_tax_percentage_l2952_295296


namespace NUMINAMATH_CALUDE_circle_properties_l2952_295277

/-- Represents a circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Calculates the center of a circle given its equation -/
def circle_center (c : Circle) : ℝ × ℝ := sorry

/-- Calculates the length of the shortest chord passing through a given point -/
def shortest_chord_length (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

/-- The main theorem about the circle and its properties -/
theorem circle_properties :
  let c : Circle := { equation := fun x y => x^2 + y^2 - 6*x - 8*y = 0 }
  let p : ℝ × ℝ := (3, 5)
  circle_center c = (3, 4) ∧
  shortest_chord_length c p = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l2952_295277


namespace NUMINAMATH_CALUDE_initial_articles_sold_l2952_295226

/-- The number of articles sold to gain 20% when the total selling price is $60 -/
def articles_sold_gain (n : ℕ) : Prop :=
  ∃ (cp : ℚ), 1.2 * cp * n = 60

/-- The number of articles that should be sold to incur a loss of 20% when the total selling price is $60 -/
def articles_sold_loss : ℚ := 29.99999625000047

/-- The proposition that the initial number of articles sold is correct -/
def correct_initial_articles (n : ℕ) : Prop :=
  articles_sold_gain n ∧
  ∃ (cp : ℚ), 0.8 * cp * articles_sold_loss = 60 ∧
              cp * articles_sold_loss = 75 ∧
              cp * n = 50

theorem initial_articles_sold :
  ∃ (n : ℕ), correct_initial_articles n ∧ n = 20 := by sorry

end NUMINAMATH_CALUDE_initial_articles_sold_l2952_295226


namespace NUMINAMATH_CALUDE_emily_candy_duration_l2952_295281

/-- The number of days Emily's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Proof that Emily's candy will last for 2 days -/
theorem emily_candy_duration :
  candy_duration 5 13 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_duration_l2952_295281


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2952_295278

theorem complex_product_theorem :
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 - i
  let z₂ : ℂ := 2 + i
  z₁ * z₂ = 3 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2952_295278


namespace NUMINAMATH_CALUDE_simplify_expression_l2952_295263

theorem simplify_expression :
  ∀ w x : ℝ, 3*w + 6*w + 9*w + 12*w + 15*w - 2*x - 4*x - 6*x - 8*x - 10*x + 24 = 45*w - 30*x + 24 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2952_295263


namespace NUMINAMATH_CALUDE_helmet_cost_l2952_295218

theorem helmet_cost (total_cost bicycle_cost helmet_cost : ℝ) : 
  total_cost = 240 →
  bicycle_cost = 5 * helmet_cost →
  total_cost = bicycle_cost + helmet_cost →
  helmet_cost = 40 := by
sorry

end NUMINAMATH_CALUDE_helmet_cost_l2952_295218


namespace NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l2952_295272

theorem smallest_common_multiple_9_6 : ∃ n : ℕ+, (∀ m : ℕ+, 9 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 9 ∣ n ∧ 6 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l2952_295272


namespace NUMINAMATH_CALUDE_pen_pencil_cost_difference_l2952_295207

/-- The cost difference between a pen and a pencil -/
def cost_difference (pen_cost pencil_cost : ℝ) : ℝ := pen_cost - pencil_cost

/-- The total cost of a pen and a pencil -/
def total_cost (pen_cost pencil_cost : ℝ) : ℝ := pen_cost + pencil_cost

theorem pen_pencil_cost_difference :
  ∀ (pen_cost : ℝ),
    pencil_cost = 2 →
    total_cost pen_cost pencil_cost = 13 →
    pen_cost > pencil_cost →
    cost_difference pen_cost pencil_cost = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_difference_l2952_295207


namespace NUMINAMATH_CALUDE_nadines_pebbles_l2952_295275

/-- The number of pebbles Nadine has -/
def total_pebbles (white red blue green : ℕ) : ℕ := white + red + blue + green

/-- Theorem stating the total number of pebbles Nadine has -/
theorem nadines_pebbles :
  ∀ (white red blue green : ℕ),
  white = 20 →
  red = white / 2 →
  blue = red / 3 →
  green = blue + 5 →
  total_pebbles white red blue green = 41 := by
sorry

#eval total_pebbles 20 10 3 8

end NUMINAMATH_CALUDE_nadines_pebbles_l2952_295275


namespace NUMINAMATH_CALUDE_olympic_medal_theorem_l2952_295202

/-- Represents the number of ways to award medals in the Olympic 100-meter finals -/
def olympic_medal_ways (total_sprinters : ℕ) (british_sprinters : ℕ) (medals : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_theorem :
  let total_sprinters := 10
  let british_sprinters := 4
  let medals := 3
  olympic_medal_ways total_sprinters british_sprinters medals = 912 :=
by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_theorem_l2952_295202


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2952_295222

theorem rectangle_max_area (d : ℝ) (h : d > 0) :
  ∀ (w h : ℝ), w > 0 → h > 0 → w^2 + h^2 = d^2 →
  w * h ≤ (d^2) / 2 ∧ (w * h = (d^2) / 2 ↔ w = h) :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2952_295222


namespace NUMINAMATH_CALUDE_meal_combinations_l2952_295290

/-- Number of meat options -/
def meatOptions : ℕ := 4

/-- Number of vegetable options -/
def vegOptions : ℕ := 5

/-- Number of dessert options -/
def dessertOptions : ℕ := 5

/-- Number of meat choices -/
def meatChoices : ℕ := 2

/-- Number of vegetable choices -/
def vegChoices : ℕ := 3

/-- Number of dessert choices -/
def dessertChoices : ℕ := 1

/-- The total number of meal combinations -/
theorem meal_combinations : 
  (meatOptions.choose meatChoices) * (vegOptions.choose vegChoices) * dessertOptions = 300 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l2952_295290


namespace NUMINAMATH_CALUDE_staff_dress_price_l2952_295291

/-- The final price of a dress for staff members after discounts -/
theorem staff_dress_price (d : ℝ) : 
  let initial_discount : ℝ := 0.65
  let staff_discount : ℝ := 0.60
  let price_after_initial_discount : ℝ := d * (1 - initial_discount)
  let final_price : ℝ := price_after_initial_discount * (1 - staff_discount)
  final_price = d * 0.14 := by
sorry

end NUMINAMATH_CALUDE_staff_dress_price_l2952_295291


namespace NUMINAMATH_CALUDE_palace_number_puzzle_l2952_295267

theorem palace_number_puzzle :
  ∀ (x : ℕ),
    x < 15 →
    (15 - x) + (15 + x) = 30 →
    (15 + x) - (15 - x) = 2 * x →
    2 * x * 30 = 780 →
    x = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_palace_number_puzzle_l2952_295267


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2952_295280

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 3 + a 4 + a 5 + a 6 + a 7 = 15)
  (h_sum2 : a 9 + a 10 + a 11 = 39) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2952_295280


namespace NUMINAMATH_CALUDE_root_ratio_equality_l2952_295288

theorem root_ratio_equality (a : ℝ) (h_pos : a > 0) : 
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
   x₁^3 + 1 = a*x₁ ∧ x₂^3 + 1 = a*x₂ ∧
   x₂ / x₁ = 2018 ∧
   (∀ x : ℝ, x^3 + 1 = a*x → x = x₁ ∨ x = x₂ ∨ x ≤ 0)) →
  (∃ y₁ y₂ : ℝ, 0 < y₁ ∧ y₁ < y₂ ∧ 
   y₁^3 + 1 = a*y₁^2 ∧ y₂^3 + 1 = a*y₂^2 ∧
   y₂ / y₁ = 2018 ∧
   (∀ y : ℝ, y^3 + 1 = a*y^2 → y = y₁ ∨ y = y₂ ∨ y ≤ 0)) := by
sorry

end NUMINAMATH_CALUDE_root_ratio_equality_l2952_295288


namespace NUMINAMATH_CALUDE_xyz_stock_price_evolution_l2952_295241

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

theorem xyz_stock_price_evolution :
  stock_price_evolution 120 1 0.3 = 168 := by
  sorry

end NUMINAMATH_CALUDE_xyz_stock_price_evolution_l2952_295241


namespace NUMINAMATH_CALUDE_dog_food_consumption_l2952_295200

theorem dog_food_consumption (total_food : ℝ) (num_dogs : ℕ) (h1 : total_food = 0.25) (h2 : num_dogs = 2) :
  let food_per_dog := total_food / num_dogs
  food_per_dog = 0.125 := by
sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l2952_295200


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_23_l2952_295231

theorem modular_inverse_of_3_mod_23 : ∃ x : ℕ, x ≤ 22 ∧ (3 * x) % 23 = 1 :=
by
  use 8
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_23_l2952_295231


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2952_295237

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2952_295237


namespace NUMINAMATH_CALUDE_problem_C_most_suitable_for_systematic_sampling_l2952_295246

/-- Represents a sampling problem with population size and sample size -/
structure SamplingProblem where
  population_size : ℕ
  sample_size : ℕ

/-- Defines the suitability of a sampling method for a given problem -/
def systematic_sampling_suitability (problem : SamplingProblem) : ℕ :=
  if problem.population_size ≥ 1000 ∧ problem.sample_size ≥ 100 then 3
  else if problem.population_size < 100 ∨ problem.sample_size < 20 then 1
  else 2

/-- The sampling problems given in the question -/
def problem_A : SamplingProblem := ⟨48, 8⟩
def problem_B : SamplingProblem := ⟨210, 21⟩
def problem_C : SamplingProblem := ⟨1200, 100⟩
def problem_D : SamplingProblem := ⟨1200, 10⟩

/-- Theorem stating that problem C is most suitable for systematic sampling -/
theorem problem_C_most_suitable_for_systematic_sampling :
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_A ∧
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_B ∧
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_D :=
sorry

end NUMINAMATH_CALUDE_problem_C_most_suitable_for_systematic_sampling_l2952_295246


namespace NUMINAMATH_CALUDE_problem_statement_l2952_295243

theorem problem_statement : 3 * 3^4 - 9^35 / 9^33 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2952_295243


namespace NUMINAMATH_CALUDE_period_of_cos_3x_l2952_295235

theorem period_of_cos_3x :
  let f : ℝ → ℝ := λ x => Real.cos (3 * x)
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x :=
by
  sorry

end NUMINAMATH_CALUDE_period_of_cos_3x_l2952_295235


namespace NUMINAMATH_CALUDE_xyz_inequality_l2952_295232

theorem xyz_inequality (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) ≤ Real.sqrt (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2952_295232


namespace NUMINAMATH_CALUDE_hayden_pants_ironing_time_l2952_295287

/-- Represents the ironing routine of Hayden --/
structure IroningRoutine where
  shirt_time : ℕ  -- Time spent ironing shirt per day (in minutes)
  days_per_week : ℕ  -- Number of days Hayden irons per week
  total_time : ℕ  -- Total time spent ironing over 4 weeks (in minutes)

/-- Calculates the time spent ironing pants per day --/
def pants_ironing_time (routine : IroningRoutine) : ℕ :=
  let total_per_week := routine.total_time / 4
  let shirt_per_week := routine.shirt_time * routine.days_per_week
  let pants_per_week := total_per_week - shirt_per_week
  pants_per_week / routine.days_per_week

/-- Theorem stating that Hayden spends 3 minutes ironing his pants each day --/
theorem hayden_pants_ironing_time :
  pants_ironing_time ⟨5, 5, 160⟩ = 3 := by
  sorry


end NUMINAMATH_CALUDE_hayden_pants_ironing_time_l2952_295287


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2952_295266

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = (a + 1) / (b + 1) → x + y = 2 * c →
  min x y = (2 * c * (a + 1)) / (a + b + 2) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2952_295266


namespace NUMINAMATH_CALUDE_equation_solution_l2952_295217

theorem equation_solution : ∃ y : ℝ, (32 : ℝ) ^ (3 * y) = 8 ^ (2 * y + 1) ∧ y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2952_295217


namespace NUMINAMATH_CALUDE_nancy_bottle_caps_l2952_295295

theorem nancy_bottle_caps (initial : ℕ) : initial + 88 = 179 → initial = 91 := by
  sorry

end NUMINAMATH_CALUDE_nancy_bottle_caps_l2952_295295


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_800_l2952_295270

theorem greatest_multiple_of_5_and_7_under_800 : 
  ∀ n : ℕ, n % 5 = 0 ∧ n % 7 = 0 ∧ n < 800 → n ≤ 770 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_800_l2952_295270


namespace NUMINAMATH_CALUDE_connections_in_specific_system_l2952_295250

/-- A communication system with a fixed number of subscribers. -/
structure CommunicationSystem where
  /-- The number of subscribers in the system. -/
  num_subscribers : ℕ
  /-- The number of connections each subscriber has. -/
  connections_per_subscriber : ℕ
  /-- The total number of connections is even. -/
  even_total_connections : Even (num_subscribers * connections_per_subscriber)

/-- Theorem stating the properties of the number of connections in a specific communication system. -/
theorem connections_in_specific_system
  (sys : CommunicationSystem)
  (h_subscribers : sys.num_subscribers = 2001)
  : Even sys.connections_per_subscriber ∧
    0 ≤ sys.connections_per_subscriber ∧
    sys.connections_per_subscriber ≤ 2000 := by
  sorry

#check connections_in_specific_system

end NUMINAMATH_CALUDE_connections_in_specific_system_l2952_295250


namespace NUMINAMATH_CALUDE_overall_profit_calculation_john_profit_is_50_l2952_295236

/-- Calculates the overall profit from selling two items with given costs and profit/loss percentages -/
theorem overall_profit_calculation 
  (grinder_cost mobile_cost : ℕ) 
  (grinder_loss_percent mobile_profit_percent : ℚ) : ℕ :=
  let grinder_selling_price := grinder_cost - (grinder_cost * grinder_loss_percent).floor
  let mobile_selling_price := mobile_cost + (mobile_cost * mobile_profit_percent).ceil
  let total_selling_price := grinder_selling_price + mobile_selling_price
  let total_cost := grinder_cost + mobile_cost
  (total_selling_price - total_cost).toNat

/-- Proves that given the specific costs and percentages, the overall profit is 50 -/
theorem john_profit_is_50 : 
  overall_profit_calculation 15000 8000 (5/100) (10/100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_calculation_john_profit_is_50_l2952_295236


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_3_512_minus_1_l2952_295299

theorem largest_power_of_two_dividing_3_512_minus_1 :
  (∃ (n : ℕ), 2^n ∣ (3^512 - 1) ∧ ∀ (m : ℕ), 2^m ∣ (3^512 - 1) → m ≤ n) ∧
  (∀ (n : ℕ), (2^n ∣ (3^512 - 1) ∧ ∀ (m : ℕ), 2^m ∣ (3^512 - 1) → m ≤ n) → n = 11) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_3_512_minus_1_l2952_295299


namespace NUMINAMATH_CALUDE_parabola_p_value_l2952_295238

/-- Given a parabola y^2 = 2px with directrix x = -2, prove that p = 4 -/
theorem parabola_p_value (y x p : ℝ) : 
  (y^2 = 2*p*x) → -- Parabola equation
  (-p/2 = -2) →   -- Directrix equation (transformed)
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_p_value_l2952_295238


namespace NUMINAMATH_CALUDE_no_three_distinct_squares_sum_to_100_l2952_295293

/-- A function that checks if a natural number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The proposition that there are no three distinct positive perfect squares that sum to 100 --/
theorem no_three_distinct_squares_sum_to_100 : 
  ¬ ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectSquare c ∧
    a + b + c = 100 :=
sorry

end NUMINAMATH_CALUDE_no_three_distinct_squares_sum_to_100_l2952_295293


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2952_295265

theorem geometric_sequence_second_term (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, r > 0 ∧ b = 15 * r ∧ 45/4 = b * r) → b = 15 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2952_295265


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l2952_295240

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 500) (h2 : cat_owners = 75) : 
  (cat_owners : ℝ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l2952_295240


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l2952_295205

theorem integer_less_than_sqrt_23 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l2952_295205


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2952_295224

theorem expression_simplification_and_evaluation (x : ℝ) 
  (hx_neq_neg1 : x ≠ -1) (hx_neq_0 : x ≠ 0) (hx_neq_1 : x ≠ 1) :
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = 1 / (x + 1) ∧
  (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2952_295224


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l2952_295268

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l2952_295268


namespace NUMINAMATH_CALUDE_marcella_shoes_theorem_l2952_295261

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem stating that with 50 initial pairs and 15 individual shoes lost,
    the maximum number of complete pairs remaining is 35. -/
theorem marcella_shoes_theorem :
  max_pairs_remaining 50 15 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_theorem_l2952_295261


namespace NUMINAMATH_CALUDE_floor_plus_x_equals_20_5_l2952_295249

theorem floor_plus_x_equals_20_5 :
  ∃! x : ℝ, ⌊x⌋ + x = 20.5 ∧ x = 10.5 := by
sorry

end NUMINAMATH_CALUDE_floor_plus_x_equals_20_5_l2952_295249


namespace NUMINAMATH_CALUDE_gym_membership_ratio_l2952_295271

theorem gym_membership_ratio (f m : ℕ) (h1 : f > 0) (h2 : m > 0) : 
  (35 : ℝ) * f + 20 * m = 25 * (f + m) → f / m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_gym_membership_ratio_l2952_295271


namespace NUMINAMATH_CALUDE_min_remainders_consecutive_numbers_l2952_295234

theorem min_remainders_consecutive_numbers : ∃ (x a r : ℕ), 
  (100 ≤ x) ∧ (x < 1000) ∧
  (10 ≤ a) ∧ (a < 100) ∧
  (r < a) ∧ (a + r ≥ 100) ∧
  (∀ i : Fin 4, (x + i) % (a + i) = r) :=
by sorry

end NUMINAMATH_CALUDE_min_remainders_consecutive_numbers_l2952_295234


namespace NUMINAMATH_CALUDE_centerIsSeven_l2952_295252

-- Define the type for our 3x3 array
def Array3x3 := Fin 3 → Fin 3 → Fin 9

-- Define what it means for two positions to be adjacent
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

-- Define the property of consecutive numbers
def consecutive (n m : Fin 9) : Prop :=
  n.val + 1 = m.val ∨ m.val + 1 = n.val

-- Define the property that consecutive numbers are in adjacent squares
def consecutiveAdjacent (arr : Array3x3) : Prop :=
  ∀ i j k l, consecutive (arr i j) (arr k l) → adjacent (i, j) (k, l)

-- Define the property that corner numbers sum to 20
def cornerSum20 (arr : Array3x3) : Prop :=
  (arr 0 0).val + (arr 0 2).val + (arr 2 0).val + (arr 2 2).val = 20

-- Define the property that all numbers from 1 to 9 are used
def allNumbersUsed (arr : Array3x3) : Prop :=
  ∀ n : Fin 9, ∃ i j, arr i j = n

-- The main theorem
theorem centerIsSeven (arr : Array3x3) 
  (h1 : consecutiveAdjacent arr) 
  (h2 : cornerSum20 arr) 
  (h3 : allNumbersUsed arr) : 
  arr 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_centerIsSeven_l2952_295252


namespace NUMINAMATH_CALUDE_largest_initial_number_l2952_295254

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 + a + b + c + d + e = 200 ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l2952_295254


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2952_295244

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2952_295244
