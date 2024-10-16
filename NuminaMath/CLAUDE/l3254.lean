import Mathlib

namespace NUMINAMATH_CALUDE_clown_mobile_count_l3254_325467

theorem clown_mobile_count (num_mobiles : ℕ) (clowns_per_mobile : ℕ) 
  (h1 : num_mobiles = 5) 
  (h2 : clowns_per_mobile = 28) : 
  num_mobiles * clowns_per_mobile = 140 := by
sorry

end NUMINAMATH_CALUDE_clown_mobile_count_l3254_325467


namespace NUMINAMATH_CALUDE_unique_function_solution_l3254_325446

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1) ↔ 
  (∀ x : ℝ, f x = 1 - x^2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3254_325446


namespace NUMINAMATH_CALUDE_integral_abs_x_squared_minus_x_l3254_325486

theorem integral_abs_x_squared_minus_x : ∫ x in (-1)..1, |x^2 - x| = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_abs_x_squared_minus_x_l3254_325486


namespace NUMINAMATH_CALUDE_initial_caterpillars_l3254_325480

theorem initial_caterpillars (initial : ℕ) (added : ℕ) (left : ℕ) (remaining : ℕ) : 
  added = 4 → left = 8 → remaining = 10 → initial + added - left = remaining → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_caterpillars_l3254_325480


namespace NUMINAMATH_CALUDE_largest_number_l3254_325448

theorem largest_number (a b c d : ℝ) : 
  a + (b + c + d) / 3 = 92 →
  b + (a + c + d) / 3 = 86 →
  c + (a + b + d) / 3 = 80 →
  d + (a + b + c) / 3 = 90 →
  max a (max b (max c d)) = 51 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3254_325448


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3254_325443

theorem logarithm_expression_equality : 
  (Real.log 160 / Real.log 4) / (Real.log 4 / Real.log 80) - 
  (Real.log 40 / Real.log 4) / (Real.log 4 / Real.log 10) = 
  4.25 + (3/2) * (Real.log 5 / Real.log 4) := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3254_325443


namespace NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_l3254_325424

/-- Given two parallel lines in a 2D plane, this theorem states that 
    the minimum distance from the midpoint of any line segment 
    connecting points on these lines to the origin is 3√2. -/
theorem min_distance_midpoint_to_origin 
  (l₁ l₂ : Set (ℝ × ℝ)) 
  (h₁ : l₁ = {(x, y) | x + y = 7})
  (h₂ : l₂ = {(x, y) | x + y = 5})
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A ∈ l₁) (hB : B ∈ l₂) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧ 
    ∀ (A' : ℝ × ℝ) (B' : ℝ × ℝ), A' ∈ l₁ → B' ∈ l₂ → 
      let M' := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
      d ≤ Real.sqrt (M'.1^2 + M'.2^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_l3254_325424


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3254_325408

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℝ) 
                                  (set2_count : ℕ) (set2_mean : ℝ) :
  set1_count = 7 →
  set2_count = 8 →
  set1_mean = 15 →
  set2_mean = 27 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℝ) = 21.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3254_325408


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3254_325430

theorem alcohol_mixture_percentage :
  ∀ x : ℝ,
  (x / 100) * 8 + (12 / 100) * 2 = (22.4 / 100) * 10 →
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3254_325430


namespace NUMINAMATH_CALUDE_expression_evaluation_l3254_325410

theorem expression_evaluation : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3254_325410


namespace NUMINAMATH_CALUDE_can_tile_4x7_with_4x1_l3254_325458

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tetromino with width and height -/
structure Tetromino where
  width : ℕ
  height : ℕ

/-- Checks if a rectangle can be tiled with a given tetromino -/
def can_tile (r : Rectangle) (t : Tetromino) : Prop :=
  ∃ (n : ℕ), n * (t.width * t.height) = r.width * r.height

/-- The 4x7 rectangle -/
def rectangle_4x7 : Rectangle :=
  { width := 4, height := 7 }

/-- The 4x1 tetromino -/
def tetromino_4x1 : Tetromino :=
  { width := 4, height := 1 }

/-- Theorem stating that the 4x7 rectangle can be tiled with 4x1 tetrominos -/
theorem can_tile_4x7_with_4x1 : can_tile rectangle_4x7 tetromino_4x1 :=
  sorry

end NUMINAMATH_CALUDE_can_tile_4x7_with_4x1_l3254_325458


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3254_325411

/-- Given a line equation in vector form, prove its slope-intercept form -/
theorem line_vector_to_slope_intercept 
  (x y : ℝ) : 
  (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 5) = 0 ↔ y = 2 * x - 13 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3254_325411


namespace NUMINAMATH_CALUDE_exists_respectful_quadratic_with_zero_at_neg_one_l3254_325427

/-- A respectful quadratic polynomial. -/
structure RespectfulQuadratic where
  a : ℝ
  b : ℝ

/-- The polynomial function for a respectful quadratic. -/
def q (p : RespectfulQuadratic) (x : ℝ) : ℝ :=
  x^2 + p.a * x + p.b

/-- The condition that q(q(x)) = 0 has exactly four real roots. -/
def hasFourRoots (p : RespectfulQuadratic) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    ∀ (x : ℝ), q p (q p x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄

/-- The main theorem stating the existence of a respectful quadratic polynomial
    satisfying the required conditions. -/
theorem exists_respectful_quadratic_with_zero_at_neg_one :
  ∃ (p : RespectfulQuadratic), hasFourRoots p ∧ q p (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_respectful_quadratic_with_zero_at_neg_one_l3254_325427


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3254_325471

/-- The asymptotes of the hyperbola -/
def asymptote1 (x : ℝ) : ℝ := 3 * x + 6
def asymptote2 (x : ℝ) : ℝ := -3 * x + 4

/-- The point through which the hyperbola passes -/
def point : ℝ × ℝ := (1, 10)

/-- The standard form of the hyperbola equation -/
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem hyperbola_sum (h k a b : ℝ) :
  (∀ x, asymptote1 x = asymptote2 x → x = -1/3 ∧ asymptote1 x = 5) →
  hyperbola_equation point.1 point.2 h k a b →
  a > 0 ∧ b > 0 →
  a + h = 8/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3254_325471


namespace NUMINAMATH_CALUDE_range_of_m_l3254_325441

-- Define the conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧  -- p is sufficient for q
  (∃ x, q x m ∧ ¬p x) ∧ -- p is not necessary for q
  (m > 0) →             -- given condition
  m ≥ 9 :=               -- conclusion
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3254_325441


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_c_coords_l3254_325433

/-- An isosceles right triangle in 2D space -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2
  isRight : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- Theorem: The coordinates of C in the given isosceles right triangle -/
theorem isosceles_right_triangle_c_coords :
  ∀ t : IsoscelesRightTriangle,
  t.A = (1, 0) → t.B = (3, 1) →
  t.C = (2, 3) ∨ t.C = (4, -1) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_c_coords_l3254_325433


namespace NUMINAMATH_CALUDE_unique_base6_divisible_by_13_l3254_325416

/-- Converts a base-6 number of the form 2dd3₆ to base 10 --/
def base6_to_base10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6 + 3

/-- Checks if a number is divisible by 13 --/
def divisible_by_13 (n : Nat) : Prop :=
  n % 13 = 0

/-- Theorem stating that 2553₆ is divisible by 13 and is the only number of the form 2dd3₆ with this property --/
theorem unique_base6_divisible_by_13 :
  divisible_by_13 (base6_to_base10 5) ∧
  ∀ d : Nat, d < 6 → d ≠ 5 → ¬(divisible_by_13 (base6_to_base10 d)) :=
by sorry

end NUMINAMATH_CALUDE_unique_base6_divisible_by_13_l3254_325416


namespace NUMINAMATH_CALUDE_calcium_chloride_formation_l3254_325440

/-- Represents a chemical reaction --/
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String
  ratio1 : ℚ
  ratio2 : ℚ

/-- Calculates the moles of product formed in a chemical reaction --/
def calculate_product_moles (r : Reaction) (moles_reactant1 : ℚ) (moles_reactant2 : ℚ) : ℚ :=
  min (moles_reactant1 / r.ratio1) (moles_reactant2 / r.ratio2)

/-- Theorem: Given 4 moles of HCl and 2 moles of CaCO3, 2 moles of CaCl2 are formed --/
theorem calcium_chloride_formation :
  let r : Reaction := {
    reactant1 := "CaCO3",
    reactant2 := "HCl",
    product1 := "CaCl2",
    product2 := "CO2",
    product3 := "H2O",
    ratio1 := 1,
    ratio2 := 2
  }
  calculate_product_moles r 2 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calcium_chloride_formation_l3254_325440


namespace NUMINAMATH_CALUDE_problem_statement_l3254_325462

theorem problem_statement : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3254_325462


namespace NUMINAMATH_CALUDE_exist_sequence_l3254_325421

/-- Sum of digits of a positive integer -/
def S (m : ℕ+) : ℕ := sorry

/-- Product of digits of a positive integer -/
def P (m : ℕ+) : ℕ := sorry

/-- For any positive integer n, there exist positive integers a₁, a₂, ..., aₙ
    satisfying the required conditions -/
theorem exist_sequence (n : ℕ+) : 
  ∃ (a : Fin n → ℕ+), 
    (∀ i j : Fin n, i < j → S (a i) < S (a j)) ∧ 
    (∀ i : Fin n, S (a i) = P (a ((i + 1) % n))) := by
  sorry

end NUMINAMATH_CALUDE_exist_sequence_l3254_325421


namespace NUMINAMATH_CALUDE_calculation_proof_l3254_325419

theorem calculation_proof : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3254_325419


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l3254_325482

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  abs (a - b) = 45 :=  -- positive difference is 45°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l3254_325482


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l3254_325413

/-- A circle with center on the y-axis, radius 1, passing through (1, 2) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  point_on_circle : passes_through = (1, 2)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) :
  ∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l3254_325413


namespace NUMINAMATH_CALUDE_quadratic_roots_count_l3254_325404

/-- The number of real roots of the quadratic function y = x^2 + x - 1 is 2 -/
theorem quadratic_roots_count : 
  let f : ℝ → ℝ := fun x ↦ x^2 + x - 1
  (∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0) ∧ 
  (∀ (x y z : ℝ), f x = 0 → f y = 0 → f z = 0 → x = y ∨ x = z ∨ y = z) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_count_l3254_325404


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3254_325406

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) / (a * b) = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y) / (x * y) = 1 → a + 2 * b ≤ x + 2 * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3254_325406


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3254_325474

theorem quadratic_rewrite (b : ℝ) (h1 : b < 0) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 1/4 = (x + m)^2 + 1/6) →
  b = -1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3254_325474


namespace NUMINAMATH_CALUDE_complex_calculation_l3254_325456

theorem complex_calculation : (26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3254_325456


namespace NUMINAMATH_CALUDE_car_expense_difference_l3254_325499

/-- Calculates the difference between Alberto's and Samara's car expenses -/
theorem car_expense_difference : 
  let alberto_engine : ℚ := 2457
  let alberto_transmission : ℚ := 374
  let alberto_tires : ℚ := 520
  let alberto_battery : ℚ := 129
  let alberto_exhaust : ℚ := 799
  let alberto_exhaust_discount : ℚ := 0.05
  let alberto_loyalty_discount : ℚ := 0.07
  let samara_oil : ℚ := 25
  let samara_tires : ℚ := 467
  let samara_detailing : ℚ := 79
  let samara_brake_pads : ℚ := 175
  let samara_paint : ℚ := 599
  let samara_stereo : ℚ := 225
  let samara_sales_tax : ℚ := 0.06

  let alberto_total := alberto_engine + alberto_transmission + alberto_tires + alberto_battery + 
                       (alberto_exhaust * (1 - alberto_exhaust_discount))
  let alberto_final := alberto_total * (1 - alberto_loyalty_discount)
  
  let samara_total := samara_oil + samara_tires + samara_detailing + samara_brake_pads + 
                      samara_paint + samara_stereo
  let samara_final := samara_total * (1 + samara_sales_tax)

  alberto_final - samara_final = 2278.12 := by sorry

end NUMINAMATH_CALUDE_car_expense_difference_l3254_325499


namespace NUMINAMATH_CALUDE_second_stop_off_count_l3254_325484

/-- Represents the number of passengers on the bus after each stop -/
def passengers : List ℕ := [0, 7, 0, 11]

/-- Represents the number of people getting on at each stop -/
def people_on : List ℕ := [7, 5, 4]

/-- Represents the number of people getting off at each stop -/
def people_off : List ℕ := [0, 0, 2]

/-- The unknown number of people who got off at the second stop -/
def x : ℕ := sorry

theorem second_stop_off_count :
  x = 3 ∧
  passengers[3] = passengers[1] + people_on[1] - x + people_on[2] - people_off[2] :=
by sorry

end NUMINAMATH_CALUDE_second_stop_off_count_l3254_325484


namespace NUMINAMATH_CALUDE_parallel_lines_transitive_line_plane_parallelism_l3254_325497

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "not subset of" relation between a line and a plane
variable (not_subset : Line → Plane → Prop)

-- Axioms for non-coincidence
variable (a b c : Line)
variable (α β : Plane)
axiom lines_non_coincident : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom planes_non_coincident : α ≠ β

-- Theorem 1: Transitivity of parallel lines
theorem parallel_lines_transitive :
  parallel_lines a c → parallel_lines b c → parallel_lines a b :=
sorry

-- Theorem 2: Parallelism between line and plane
theorem line_plane_parallelism :
  not_subset a α → parallel_line_plane b α → parallel_lines a b → parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_transitive_line_plane_parallelism_l3254_325497


namespace NUMINAMATH_CALUDE_f_monotonic_implies_a_range_l3254_325478

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x)

-- Theorem statement
theorem f_monotonic_implies_a_range :
  ∀ a : ℝ, monotonic_on (f a) 2 4 → a ≤ 3 ∨ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonic_implies_a_range_l3254_325478


namespace NUMINAMATH_CALUDE_number_greater_than_three_l3254_325425

theorem number_greater_than_three (x : ℝ) : 7 * x - 15 > 2 * x → x > 3 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_three_l3254_325425


namespace NUMINAMATH_CALUDE_parallel_lines_a_values_l3254_325464

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 10 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ a x y → l₂ a x y

-- Theorem statement
theorem parallel_lines_a_values :
  ∀ a : ℝ, parallel a → (a = -1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_values_l3254_325464


namespace NUMINAMATH_CALUDE_paul_bought_101_books_l3254_325493

/-- Calculates the number of books bought given initial and final book counts -/
def books_bought (initial_count final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Proves that Paul bought 101 books -/
theorem paul_bought_101_books (initial_count final_count : ℕ) 
  (h1 : initial_count = 50)
  (h2 : final_count = 151) :
  books_bought initial_count final_count = 101 := by
  sorry

end NUMINAMATH_CALUDE_paul_bought_101_books_l3254_325493


namespace NUMINAMATH_CALUDE_unique_number_between_cubes_l3254_325409

theorem unique_number_between_cubes : ∃! (n : ℕ), 
  n > 0 ∧ 
  24 ∣ n ∧ 
  (9 : ℝ) < n^(1/3) ∧ 
  n^(1/3) < (9.1 : ℝ) ∧ 
  n = 744 := by sorry

end NUMINAMATH_CALUDE_unique_number_between_cubes_l3254_325409


namespace NUMINAMATH_CALUDE_village_population_l3254_325496

theorem village_population (P : ℝ) : 
  (1 - 0.1) * (1 - 0.25) * P - 
  0.05 * 0.05 * P + 
  0.04 * (1 - 0.1) * P - 
  (0.01 - 0.008) * P = 
  4725 → P = 6731 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l3254_325496


namespace NUMINAMATH_CALUDE_pythagorean_difference_zero_l3254_325407

theorem pythagorean_difference_zero : 
  Real.sqrt (5^2 - 4^2 - 3^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_difference_zero_l3254_325407


namespace NUMINAMATH_CALUDE_pet_store_cats_l3254_325455

theorem pet_store_cats (white_cats black_cats total_cats : ℕ) 
  (h1 : white_cats = 2)
  (h2 : black_cats = 10)
  (h3 : total_cats = 15)
  : total_cats - (white_cats + black_cats) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_l3254_325455


namespace NUMINAMATH_CALUDE_determinant_property_l3254_325434

theorem determinant_property (p q r s : ℝ) 
  (h : Matrix.det !![p, q; r, s] = 3) : 
  Matrix.det !![2*p, 2*p + 5*q; 2*r, 2*r + 5*s] = 30 := by
  sorry

end NUMINAMATH_CALUDE_determinant_property_l3254_325434


namespace NUMINAMATH_CALUDE_consecutive_even_sum_representation_l3254_325422

theorem consecutive_even_sum_representation (n k : ℕ) (hn : n > 2) (hk : k > 2) :
  ∃ m : ℕ, n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_representation_l3254_325422


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l3254_325463

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_k_for_prime_roots : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 78 ∧ 
    p * q = k ∧ 
    p^2 - 78*p + k = 0 ∧ 
    q^2 - 78*q + k = 0 ∧
    k = 146 :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l3254_325463


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l3254_325432

theorem greatest_x_with_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 12 18) = 180) → x ≤ 180 ∧ ∃ y : ℕ, y = 180 ∧ Nat.lcm y (Nat.lcm 12 18) = 180 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l3254_325432


namespace NUMINAMATH_CALUDE_salt_calculation_l3254_325495

/-- Calculates the amount of salt obtained from seawater evaporation -/
def salt_from_seawater (volume : ℝ) (salt_percentage : ℝ) : ℝ :=
  volume * salt_percentage * 1000

/-- Proves that 2 liters of seawater with 20% salt content yields 400 ml of salt -/
theorem salt_calculation :
  salt_from_seawater 2 0.20 = 400 := by
  sorry

end NUMINAMATH_CALUDE_salt_calculation_l3254_325495


namespace NUMINAMATH_CALUDE_quadratic_vertex_and_extremum_l3254_325452

/-- Given a quadratic equation y = -x^2 + cx + d with roots -5 and 3,
    prove that its vertex is (4, 1) and it represents a maximum point. -/
theorem quadratic_vertex_and_extremum (c d : ℝ) :
  (∀ x, -x^2 + c*x + d = 0 ↔ x = -5 ∨ x = 3) →
  (∃! p : ℝ × ℝ, p.1 = 4 ∧ p.2 = 1 ∧ 
    (∀ x, -x^2 + c*x + d ≤ -p.1^2 + c*p.1 + d)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_and_extremum_l3254_325452


namespace NUMINAMATH_CALUDE_erasers_lost_l3254_325437

def initial_erasers : ℕ := 95
def final_erasers : ℕ := 53

theorem erasers_lost : initial_erasers - final_erasers = 42 := by
  sorry

end NUMINAMATH_CALUDE_erasers_lost_l3254_325437


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_similarity_theorem_l3254_325494

-- Define cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define similarity of quadrilaterals
def are_similar_quadrilaterals (A B C D A' B' C' D' : Point) : Prop := sorry

-- Define area of a triangle
def area_triangle (A B C : Point) : ℝ := sorry

-- Define distance between two points
def distance (A B : Point) : ℝ := sorry

theorem cyclic_quadrilateral_similarity_theorem 
  (A B C D A' B' C' D' : Point) 
  (h1 : is_cyclic_quadrilateral A B C D) 
  (h2 : is_cyclic_quadrilateral A' B' C' D')
  (h3 : are_similar_quadrilaterals A B C D A' B' C' D') :
  (distance A A')^2 * area_triangle B C D + (distance C C')^2 * area_triangle A B D = 
  (distance B B')^2 * area_triangle A C D + (distance D D')^2 * area_triangle A B C := by
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_similarity_theorem_l3254_325494


namespace NUMINAMATH_CALUDE_table_football_points_l3254_325489

/-- The total points scored by four friends in table football games -/
def total_points (darius matt marius sofia : ℕ) : ℕ :=
  darius + matt + marius + sofia

/-- Theorem stating the total points scored by the four friends -/
theorem table_football_points : ∃ (darius matt marius sofia : ℕ),
  darius = 10 ∧
  marius = darius + 3 ∧
  matt = darius + 5 ∧
  sofia = 2 * matt ∧
  total_points darius matt marius sofia = 68 := by
  sorry


end NUMINAMATH_CALUDE_table_football_points_l3254_325489


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_cosine_l3254_325429

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem smallest_positive_period_of_cosine 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (P : ℝ × ℝ) 
  (h_center_symmetry : ∀ x, f ω (2 * P.1 - x) = f ω x) 
  (h_min_distance : ∀ y, abs (P.2 - y) ≥ π) :
  ∃ T > 0, (∀ x, f ω (x + T) = f ω x) ∧ 
  (∀ S, S > 0 → (∀ x, f ω (x + S) = f ω x) → S ≥ T) ∧ 
  T = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_cosine_l3254_325429


namespace NUMINAMATH_CALUDE_parabola_translation_l3254_325435

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 3x² -/
def original_parabola : Parabola := ⟨3, 0, 0⟩

/-- The vertical translation amount -/
def translation : ℝ := -2

/-- Translates a parabola vertically by a given amount -/
def translate_vertically (p : Parabola) (t : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + t⟩

/-- The resulting parabola after translation -/
def resulting_parabola : Parabola :=
  translate_vertically original_parabola translation

theorem parabola_translation :
  resulting_parabola = ⟨3, 0, -2⟩ := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3254_325435


namespace NUMINAMATH_CALUDE_carries_money_l3254_325461

/-- The amount of money Carrie spent on the sweater -/
def sweater_cost : ℕ := 24

/-- The amount of money Carrie spent on the T-shirt -/
def tshirt_cost : ℕ := 6

/-- The amount of money Carrie spent on the shoes -/
def shoes_cost : ℕ := 11

/-- The amount of money Carrie has left after shopping -/
def money_left : ℕ := 50

/-- The total amount of money Carrie's mom gave her -/
def total_money : ℕ := sweater_cost + tshirt_cost + shoes_cost + money_left

theorem carries_money : total_money = 91 := by sorry

end NUMINAMATH_CALUDE_carries_money_l3254_325461


namespace NUMINAMATH_CALUDE_all_ones_satisfy_l3254_325472

def satisfies_inequalities (a : Fin 100 → ℝ) : Prop :=
  ∀ i : Fin 100, a i - 4 * a (i.succ) + 3 * a (i.succ.succ) ≥ 0

theorem all_ones_satisfy (a : Fin 100 → ℝ) 
  (h : satisfies_inequalities a) (h1 : a 0 = 1) : 
  ∀ i : Fin 100, a i = 1 :=
sorry

end NUMINAMATH_CALUDE_all_ones_satisfy_l3254_325472


namespace NUMINAMATH_CALUDE_cube_volume_and_surface_area_l3254_325498

/-- Given a cube with body diagonal length 6√3, prove its volume and surface area are both 216 -/
theorem cube_volume_and_surface_area (s : ℝ) (h : s > 0) :
  s^2 + s^2 + s^2 = (6 * Real.sqrt 3)^2 →
  s^3 = 216 ∧ 6 * s^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_and_surface_area_l3254_325498


namespace NUMINAMATH_CALUDE_cody_tickets_l3254_325447

/-- The number of tickets Cody spent on a beanie -/
def beanie_cost : ℕ := 25

/-- The number of additional tickets Cody won later -/
def additional_tickets : ℕ := 6

/-- The number of tickets Cody has now -/
def current_tickets : ℕ := 30

/-- The initial number of tickets Cody won -/
def initial_tickets : ℕ := 49

theorem cody_tickets : 
  initial_tickets = beanie_cost + (current_tickets - additional_tickets) :=
by sorry

end NUMINAMATH_CALUDE_cody_tickets_l3254_325447


namespace NUMINAMATH_CALUDE_suresh_completion_time_l3254_325490

/-- Proves that Suresh can complete the job alone in 15 hours given the problem conditions -/
theorem suresh_completion_time :
  ∀ (S : ℝ),
  (S > 0) →  -- Suresh's completion time is positive
  (9 / S + 10 / 25 = 1) →  -- Combined work of Suresh and Ashutosh equals the whole job
  (S = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_suresh_completion_time_l3254_325490


namespace NUMINAMATH_CALUDE_correct_group_capacity_l3254_325488

/-- The capacity of each group in a systematic sampling -/
def group_capacity (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  (total_students - (total_students % sample_size)) / sample_size

/-- Theorem stating the correct group capacity for the given problem -/
theorem correct_group_capacity :
  group_capacity 5008 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_group_capacity_l3254_325488


namespace NUMINAMATH_CALUDE_max_volume_cone_l3254_325466

/-- Given a right-angled triangle with hypotenuse c, the triangle that forms a cone
    with maximum volume when rotated around one of its legs has the following properties: -/
theorem max_volume_cone (c : ℝ) (h : c > 0) :
  ∃ (x y : ℝ),
    -- The triangle is right-angled
    x^2 + y^2 = c^2 ∧
    -- x and y are positive
    x > 0 ∧ y > 0 ∧
    -- y is the optimal radius of the cone's base
    y = c * Real.sqrt (2/3) ∧
    -- x is the optimal height of the cone
    x = c / Real.sqrt 3 ∧
    -- The volume formed by this triangle is maximum
    ∀ (x' y' : ℝ), x'^2 + y'^2 = c^2 → x' > 0 → y' > 0 →
      (1/3) * π * y'^2 * x' ≤ (1/3) * π * y^2 * x ∧
    -- The maximum volume is (2 * π * √3 * c^3) / 27
    (1/3) * π * y^2 * x = (2 * π * Real.sqrt 3 * c^3) / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_cone_l3254_325466


namespace NUMINAMATH_CALUDE_linear_function_characterization_l3254_325405

/-- A function f: ℚ → ℚ satisfies the arithmetic progression property if
    f(x) + f(t) = f(y) + f(z) for all rational x < y < z < t in arithmetic progression -/
def ArithmeticProgressionProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ), x < y ∧ y < z ∧ z < t ∧ (y - x = z - y) ∧ (z - y = t - z) →
    f x + f t = f y + f z

/-- The main theorem: if f satisfies the arithmetic progression property,
    then f is a linear function -/
theorem linear_function_characterization (f : ℚ → ℚ) 
  (h : ArithmeticProgressionProperty f) :
  ∃ (c b : ℚ), ∀ (q : ℚ), f q = c * q + b :=
sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l3254_325405


namespace NUMINAMATH_CALUDE_binomial_20_18_l3254_325470

theorem binomial_20_18 : Nat.choose 20 18 = 190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_18_l3254_325470


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l3254_325423

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- Predicate to check if a number has exactly 3 digits in base 7 -/
def has_three_digits_base_7 (n : ℕ) : Prop :=
  7^2 ≤ n ∧ n < 7^3

theorem largest_three_digit_square_base_7 :
  M = 18 ∧
  has_three_digits_base_7 (M^2) ∧
  ∀ n : ℕ, n > M → ¬has_three_digits_base_7 (n^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l3254_325423


namespace NUMINAMATH_CALUDE_english_class_grouping_l3254_325483

/-- The maximum number of groups that can be formed with equal composition -/
def maxGroups (boys girls : ℕ) : ℕ := Nat.gcd boys girls

/-- The problem statement -/
theorem english_class_grouping (boys girls : ℕ) 
  (h_boys : boys = 10) 
  (h_girls : girls = 15) : 
  maxGroups boys girls = 5 := by
  sorry

end NUMINAMATH_CALUDE_english_class_grouping_l3254_325483


namespace NUMINAMATH_CALUDE_rainbow_preschool_full_day_students_l3254_325491

theorem rainbow_preschool_full_day_students 
  (total_students : ℕ) 
  (half_day_percentage : ℚ) 
  (h1 : total_students = 80)
  (h2 : half_day_percentage = 1/4) : 
  (1 - half_day_percentage) * total_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_rainbow_preschool_full_day_students_l3254_325491


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3254_325438

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 + 2 * Complex.I) = 4/5 + (2/5) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3254_325438


namespace NUMINAMATH_CALUDE_div_problem_l3254_325442

theorem div_problem (a b c : ℚ) (h1 : a / b = 5) (h2 : b / c = 2/5) : c / a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_div_problem_l3254_325442


namespace NUMINAMATH_CALUDE_married_couples_with_2_to_4_children_l3254_325473

/-- The fraction of married couples with 2 to 4 children in a population with given characteristics -/
theorem married_couples_with_2_to_4_children (total_population : ℕ) 
  (married_couple_percentage : ℚ) (one_child : ℚ) (two_children : ℚ) 
  (three_children : ℚ) (four_children : ℚ) (five_children : ℚ) :
  total_population = 10000 →
  married_couple_percentage = 1/5 →
  one_child = 1/5 →
  two_children = 1/4 →
  three_children = 3/20 →
  four_children = 1/6 →
  five_children = 1/10 →
  two_children + three_children + four_children = 17/30 := by
  sorry


end NUMINAMATH_CALUDE_married_couples_with_2_to_4_children_l3254_325473


namespace NUMINAMATH_CALUDE_sum_gcf_lcm_18_30_45_l3254_325431

theorem sum_gcf_lcm_18_30_45 : 
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  A + B = 93 := by
sorry

end NUMINAMATH_CALUDE_sum_gcf_lcm_18_30_45_l3254_325431


namespace NUMINAMATH_CALUDE_pet_ownership_l3254_325460

theorem pet_ownership (total_students : ℕ) 
  (dog_owners cat_owners bird_owners fish_only_owners no_pet_owners : ℕ) : 
  total_students = 40 →
  dog_owners = (40 * 5) / 8 →
  cat_owners = 40 / 2 →
  bird_owners = 40 / 4 →
  fish_only_owners = 8 →
  no_pet_owners = 6 →
  ∃ (all_pet_owners : ℕ), all_pet_owners = 6 ∧
    all_pet_owners + fish_only_owners + no_pet_owners ≤ total_students :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_l3254_325460


namespace NUMINAMATH_CALUDE_tourist_group_room_capacity_l3254_325459

/-- Given a tourist group and room arrangements, calculate the capacity of small rooms -/
theorem tourist_group_room_capacity
  (total_people : ℕ)
  (large_room_capacity : ℕ)
  (large_rooms_rented : ℕ)
  (h1 : total_people = 26)
  (h2 : large_room_capacity = 3)
  (h3 : large_rooms_rented = 8)
  : ∃ (small_room_capacity : ℕ),
    small_room_capacity > 0 ∧
    small_room_capacity * (total_people - large_room_capacity * large_rooms_rented) = total_people - large_room_capacity * large_rooms_rented ∧
    small_room_capacity = 2 :=
by sorry

end NUMINAMATH_CALUDE_tourist_group_room_capacity_l3254_325459


namespace NUMINAMATH_CALUDE_triangle_properties_l3254_325453

-- Define the triangle ABC
def Triangle (A B C : Real) (a b c : Real) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Given conditions
axiom triangle_condition {A B C a b c : Real} (h : Triangle A B C a b c) :
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

axiom c_value {A B C a b c : Real} (h : Triangle A B C a b c) :
  c = Real.sqrt 7

axiom area_value {A B C a b c : Real} (h : Triangle A B C a b c) :
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2

-- Theorem to prove
theorem triangle_properties {A B C a b c : Real} (h : Triangle A B C a b c) :
  C = Real.pi / 3 ∧ a + b + c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3254_325453


namespace NUMINAMATH_CALUDE_chess_club_female_fraction_l3254_325468

/-- Represents the chess club membership data --/
structure ChessClub where
  last_year_males : ℕ
  last_year_females : ℕ
  male_increase_rate : ℚ
  female_increase_rate : ℚ
  total_increase_rate : ℚ

/-- Calculates the fraction of female participants this year --/
def female_fraction (club : ChessClub) : ℚ :=
  let this_year_males : ℚ := club.last_year_males * (1 + club.male_increase_rate)
  let this_year_females : ℚ := club.last_year_females * (1 + club.female_increase_rate)
  this_year_females / (this_year_males + this_year_females)

/-- Theorem statement for the chess club problem --/
theorem chess_club_female_fraction :
  let club : ChessClub := {
    last_year_males := 30,
    last_year_females := 15,
    male_increase_rate := 1/10,
    female_increase_rate := 1/4,
    total_increase_rate := 3/20
  }
  female_fraction club = 19/52 := by
  sorry


end NUMINAMATH_CALUDE_chess_club_female_fraction_l3254_325468


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3254_325400

-- Define sets A and B
def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | x ≤ 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3254_325400


namespace NUMINAMATH_CALUDE_solve_flour_problem_l3254_325465

def flour_problem (total_flour sugar flour_to_add flour_already_in : ℕ) : Prop :=
  total_flour = 10 ∧
  sugar = 2 ∧
  flour_to_add = sugar + 1 ∧
  flour_already_in + flour_to_add = total_flour

theorem solve_flour_problem :
  ∃ (flour_already_in : ℕ), flour_problem 10 2 3 flour_already_in ∧ flour_already_in = 7 :=
by sorry

end NUMINAMATH_CALUDE_solve_flour_problem_l3254_325465


namespace NUMINAMATH_CALUDE_inequality_proof_l3254_325477

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) : 
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3254_325477


namespace NUMINAMATH_CALUDE_intersection_point_product_range_l3254_325439

theorem intersection_point_product_range (k x₀ y₀ : ℝ) :
  x₀ + y₀ = 2 * k - 1 →
  x₀^2 + y₀^2 = k^2 + 2 * k - 3 →
  (11 - 6 * Real.sqrt 2) / 4 ≤ x₀ * y₀ ∧ x₀ * y₀ ≤ (11 + 6 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_product_range_l3254_325439


namespace NUMINAMATH_CALUDE_jennifer_discount_is_28_l3254_325420

/-- Calculates the discount for whole milk based on the number of cans purchased -/
def whole_milk_discount (cans : ℕ) : ℕ := (cans / 10) * 4

/-- Calculates the discount for almond milk based on the number of cans purchased -/
def almond_milk_discount (cans : ℕ) : ℕ := 
  ((cans / 7) * 3) + ((cans % 7) / 3)

/-- Represents Jennifer's milk purchase and calculates her total discount -/
def jennifer_discount : ℕ :=
  let initial_whole_milk := 40
  let mark_whole_milk := 30
  let mark_skim_milk := 15
  let additional_almond_milk := (mark_whole_milk / 3) * 2
  let additional_whole_milk := (mark_skim_milk / 5) * 4
  let total_whole_milk := initial_whole_milk + additional_whole_milk
  let total_almond_milk := additional_almond_milk
  whole_milk_discount total_whole_milk + almond_milk_discount total_almond_milk

theorem jennifer_discount_is_28 : jennifer_discount = 28 := by
  sorry

#eval jennifer_discount

end NUMINAMATH_CALUDE_jennifer_discount_is_28_l3254_325420


namespace NUMINAMATH_CALUDE_bell_pepper_pieces_l3254_325426

/-- The number of bell peppers Tamia has -/
def num_peppers : ℕ := 5

/-- The number of large slices each pepper is cut into -/
def slices_per_pepper : ℕ := 20

/-- The number of smaller pieces each selected large slice is cut into -/
def pieces_per_slice : ℕ := 3

/-- The total number of bell pepper pieces Tamia will have -/
def total_pieces : ℕ := 
  let total_slices := num_peppers * slices_per_pepper
  let slices_to_cut := total_slices / 2
  let smaller_pieces := slices_to_cut * pieces_per_slice
  let remaining_slices := total_slices - slices_to_cut
  smaller_pieces + remaining_slices

theorem bell_pepper_pieces : total_pieces = 200 := by
  sorry

end NUMINAMATH_CALUDE_bell_pepper_pieces_l3254_325426


namespace NUMINAMATH_CALUDE_expression_evaluation_l3254_325485

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - y^2 ≠ 0) :
  (y^2 - 1/x) / (x - y^2) = (x*y^2 - 1) / (x^2 - x*y^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3254_325485


namespace NUMINAMATH_CALUDE_pages_revised_once_is_30_l3254_325428

/-- Represents the typing service problem --/
structure TypingService where
  totalPages : ℕ
  pagesRevisedTwice : ℕ
  totalCost : ℕ
  firstTypingRate : ℕ
  revisionRate : ℕ

/-- Calculates the number of pages revised once --/
def pagesRevisedOnce (ts : TypingService) : ℕ :=
  ((ts.totalCost - ts.firstTypingRate * ts.totalPages - 
    ts.revisionRate * ts.pagesRevisedTwice * 2) / ts.revisionRate)

/-- Theorem stating the number of pages revised once --/
theorem pages_revised_once_is_30 (ts : TypingService) 
  (h1 : ts.totalPages = 100)
  (h2 : ts.pagesRevisedTwice = 20)
  (h3 : ts.totalCost = 1350)
  (h4 : ts.firstTypingRate = 10)
  (h5 : ts.revisionRate = 5) :
  pagesRevisedOnce ts = 30 := by
  sorry

#eval pagesRevisedOnce {
  totalPages := 100,
  pagesRevisedTwice := 20,
  totalCost := 1350,
  firstTypingRate := 10,
  revisionRate := 5
}

end NUMINAMATH_CALUDE_pages_revised_once_is_30_l3254_325428


namespace NUMINAMATH_CALUDE_y1_greater_y2_l3254_325457

/-- A linear function passing through the first, second, and fourth quadrants -/
structure QuadrantCrossingLine where
  m : ℝ
  n : ℝ
  first_quadrant : ∃ x > 0, m * x + n > 0
  second_quadrant : ∃ x < 0, m * x + n > 0
  fourth_quadrant : ∃ x > 0, m * x + n < 0

/-- Theorem: For a linear function y = mx + n passing through the first, second, and fourth quadrants,
    if (1, y₁) and (3, y₂) are points on the graph, then y₁ > y₂ -/
theorem y1_greater_y2 (line : QuadrantCrossingLine) (y₁ y₂ : ℝ)
    (point1 : line.m * 1 + line.n = y₁)
    (point2 : line.m * 3 + line.n = y₂) :
    y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_y2_l3254_325457


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3254_325451

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (2 + Complex.I ^ 3)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3254_325451


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3254_325469

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}

-- Define the set B
def B : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3254_325469


namespace NUMINAMATH_CALUDE_sum_lower_bound_l3254_325450

noncomputable def f (x : ℝ) := Real.log x + x^2 + x

theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
  (h : f x₁ + f x₂ + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l3254_325450


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3254_325401

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3254_325401


namespace NUMINAMATH_CALUDE_percentage_red_shirts_l3254_325414

theorem percentage_red_shirts (total_students : ℕ) 
  (blue_percentage green_percentage : ℚ) (other_count : ℕ) :
  total_students = 800 →
  blue_percentage = 45/100 →
  green_percentage = 15/100 →
  other_count = 136 →
  (blue_percentage + green_percentage + (other_count : ℚ)/total_students + 
    (total_students - (blue_percentage * total_students + green_percentage * total_students + other_count))/total_students) = 1 →
  (total_students - (blue_percentage * total_students + green_percentage * total_students + other_count))/total_students = 23/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_red_shirts_l3254_325414


namespace NUMINAMATH_CALUDE_negative_two_exponent_division_l3254_325476

theorem negative_two_exponent_division : 
  (-2: ℤ) ^ 2014 / (-2 : ℤ) ^ 2013 = -2 := by sorry

end NUMINAMATH_CALUDE_negative_two_exponent_division_l3254_325476


namespace NUMINAMATH_CALUDE_range_of_a_l3254_325449

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

-- Define set A
def A (a : ℝ) : Set ℝ := {x | f a x = x}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | f a (f a x) = x}

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : A a = B a) (h2 : (A a).Nonempty) :
  a ∈ Set.Icc (-1/4 : ℝ) (3/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3254_325449


namespace NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l3254_325444

theorem polynomial_roots_in_arithmetic_progression (j k : ℝ) : 
  (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    (∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r) ∧
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 900 = (x - a)*(x - b)*(x - c)*(x - d))) →
  j = -900 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l3254_325444


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l3254_325417

theorem system_of_equations_sum (x y : ℝ) :
  3 * x + 2 * y = 10 →
  2 * x + 3 * y = 5 →
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l3254_325417


namespace NUMINAMATH_CALUDE_lucy_fish_goal_l3254_325479

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := 212

/-- The number of additional fish Lucy needs to buy -/
def additional_fish : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := current_fish + additional_fish

theorem lucy_fish_goal : total_fish = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_goal_l3254_325479


namespace NUMINAMATH_CALUDE_apple_count_difference_l3254_325487

/-- The number of green apples initially in the store -/
def initial_green_apples : ℕ := 32

/-- The number of additional red apples compared to green apples initially -/
def red_apple_surplus : ℕ := 200

/-- The number of green apples delivered by the truck -/
def delivered_green_apples : ℕ := 340

/-- The final difference between green and red apples -/
def final_green_red_difference : ℤ := 140

theorem apple_count_difference :
  (initial_green_apples + delivered_green_apples : ℤ) - 
  (initial_green_apples + red_apple_surplus) = 
  final_green_red_difference :=
by sorry

end NUMINAMATH_CALUDE_apple_count_difference_l3254_325487


namespace NUMINAMATH_CALUDE_congruent_side_length_l3254_325475

-- Define the triangle
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ
  side : ℝ

-- Define our specific triangle
def ourTriangle : IsoscelesTriangle where
  base := 24
  area := 60
  side := 13

-- Theorem statement
theorem congruent_side_length (t : IsoscelesTriangle) 
  (h1 : t.base = 24) 
  (h2 : t.area = 60) : 
  t.side = 13 := by
  sorry

#check congruent_side_length

end NUMINAMATH_CALUDE_congruent_side_length_l3254_325475


namespace NUMINAMATH_CALUDE_orange_weight_equivalence_l3254_325412

-- Define the weight relationship between oranges and apples
def orange_apple_ratio : ℚ := 6 / 9

-- Define the weight relationship between oranges and pears
def orange_pear_ratio : ℚ := 4 / 10

-- Define the number of oranges Jimmy has
def jimmy_oranges : ℕ := 36

-- Theorem statement
theorem orange_weight_equivalence :
  ∃ (apples pears : ℕ),
    (apples : ℚ) = jimmy_oranges * orange_apple_ratio ∧
    (pears : ℚ) = jimmy_oranges * orange_pear_ratio ∧
    apples = 24 ∧
    pears = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_weight_equivalence_l3254_325412


namespace NUMINAMATH_CALUDE_increasing_quadratic_l3254_325481

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

-- State the theorem
theorem increasing_quadratic :
  ∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > x₁ → f x₂ > f x₁ :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_l3254_325481


namespace NUMINAMATH_CALUDE_purely_imaginary_m_value_l3254_325415

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_m_value (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - m - 2) (m + 1)
  is_purely_imaginary z → m = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_m_value_l3254_325415


namespace NUMINAMATH_CALUDE_midpoint_square_area_l3254_325418

/-- Given a square with area 100, prove that a smaller square formed by 
    connecting the midpoints of the sides of the larger square has an area of 25. -/
theorem midpoint_square_area (large_square : Real × Real → Real × Real) 
  (h_area : (large_square (1, 1) - large_square (0, 0)).1 ^ 2 = 100) :
  let small_square := fun (t : Real × Real) => 
    ((large_square (t.1, t.2) + large_square (t.1 + 1, t.2 + 1)) : Real × Real) / 2
  (small_square (1, 1) - small_square (0, 0)).1 ^ 2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_square_area_l3254_325418


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_l3254_325402

theorem quadrilateral_angle_sum (x y z w : ℝ) : 
  x = 36 → y = 44 → z = 52 → x + w + y + z = 180 → w = 48 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_l3254_325402


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_difference_l3254_325436

theorem crazy_silly_school_series_difference : 
  let num_books : ℕ := 15
  let num_movies : ℕ := 14
  num_books - num_movies = 1 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_difference_l3254_325436


namespace NUMINAMATH_CALUDE_kanul_original_amount_l3254_325454

theorem kanul_original_amount 
  (raw_materials : ℝ)
  (machinery : ℝ)
  (marketing : ℝ)
  (percentage_spent : ℝ)
  (h1 : raw_materials = 35000)
  (h2 : machinery = 40000)
  (h3 : marketing = 15000)
  (h4 : percentage_spent = 0.25)
  (h5 : (raw_materials + machinery + marketing) = percentage_spent * (raw_materials + machinery + marketing) / percentage_spent) :
  (raw_materials + machinery + marketing) / percentage_spent = 360000 := by
  sorry

end NUMINAMATH_CALUDE_kanul_original_amount_l3254_325454


namespace NUMINAMATH_CALUDE_johns_father_age_multiple_l3254_325403

/-- 
Given John's age, the sum of John and his father's ages, and the relationship between
John's father's age and John's age, this theorem proves the multiple of John's age
that represents his father's age without the additional 32 years.
-/
theorem johns_father_age_multiple 
  (john_age : ℕ)
  (sum_ages : ℕ)
  (father_age_relation : ℕ → ℕ)
  (h1 : john_age = 15)
  (h2 : sum_ages = 77)
  (h3 : father_age_relation m = m * john_age + 32)
  (h4 : sum_ages = john_age + father_age_relation m) :
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_father_age_multiple_l3254_325403


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l3254_325492

/-- A line with slope 3/4 passing through (-12, -39) intersects the x-axis at x = 40 -/
theorem line_intersection_x_axis :
  ∀ (f : ℝ → ℝ),
  (∀ x y, f y - f x = (3/4) * (y - x)) →  -- Slope condition
  f (-12) = -39 →                         -- Point condition
  ∃ x, f x = 0 ∧ x = 40 :=                -- Intersection with x-axis
by
  sorry


end NUMINAMATH_CALUDE_line_intersection_x_axis_l3254_325492


namespace NUMINAMATH_CALUDE_certain_number_problem_l3254_325445

theorem certain_number_problem (x : ℝ) : ((7 * (x + 10)) / 5) - 5 = 44 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3254_325445
