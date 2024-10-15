import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l1992_199258

noncomputable def f (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def g (x : ℝ) : ℝ := f x - abs x

theorem f_properties :
  (∀ y ∈ Set.range f, 0 < y ∧ y < 4) ∧
  (∀ x, f x + f (-x) = 4) ∧
  (∃! a b, a < b ∧ g a = 0 ∧ g b = 0 ∧ ∀ x, x ≠ a ∧ x ≠ b → g x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1992_199258


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1992_199210

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 42 →  -- The total of the ages of a, b, and c is 42
  b = 16 →  -- b is 16 years old
  b = 2 * c  -- The ratio of b's age to c's age is 2:1
  := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1992_199210


namespace NUMINAMATH_CALUDE_be_length_l1992_199288

-- Define the parallelogram and points
structure Parallelogram :=
  (A B C D E F G : ℝ × ℝ)

-- Define the conditions
def is_valid_configuration (p : Parallelogram) : Prop :=
  let ⟨A, B, C, D, E, F, G⟩ := p
  -- F is on the extension of AD
  ∃ t : ℝ, t > 1 ∧ F = A + t • (D - A) ∧
  -- ABCD is a parallelogram
  B - A = C - D ∧ D - A = C - B ∧
  -- BF intersects AC at E
  ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = A + s • (C - A) ∧
  -- BF intersects DC at G
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ G = D + r • (C - D) ∧
  -- EF = 15
  ‖F - E‖ = 15 ∧
  -- GF = 20
  ‖F - G‖ = 20

-- The theorem to prove
theorem be_length (p : Parallelogram) (h : is_valid_configuration p) : 
  ‖p.B - p.E‖ = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_be_length_l1992_199288


namespace NUMINAMATH_CALUDE_supermarket_difference_l1992_199224

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 41

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- There are more supermarkets in the US than in Canada -/
axiom more_in_us : us_supermarkets > canada_supermarkets

theorem supermarket_difference : us_supermarkets - canada_supermarkets = 22 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_difference_l1992_199224


namespace NUMINAMATH_CALUDE_min_triangles_proof_l1992_199294

/-- Represents an 8x8 square with one corner cell removed -/
structure ModifiedSquare where
  side_length : ℕ
  removed_cell_area : ℕ
  total_area : ℕ

/-- Represents a triangulation of the modified square -/
structure Triangulation where
  num_triangles : ℕ
  triangle_area : ℝ

/-- The minimum number of equal-area triangles that can divide the modified square -/
def min_triangles : ℕ := 18

theorem min_triangles_proof (s : ModifiedSquare) (t : Triangulation) :
  s.side_length = 8 ∧ 
  s.removed_cell_area = 1 ∧ 
  s.total_area = s.side_length * s.side_length - s.removed_cell_area ∧
  t.triangle_area = s.total_area / t.num_triangles ∧
  t.triangle_area ≤ 3.5 →
  t.num_triangles ≥ min_triangles :=
sorry

end NUMINAMATH_CALUDE_min_triangles_proof_l1992_199294


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_B_to_B_l1992_199215

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) :
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * abs y := by sorry

/-- The specific case for point B(1, 4) --/
theorem distance_B_to_B'_is_8 :
  Real.sqrt ((1 - 1)^2 + ((-4) - 4)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_B_to_B_l1992_199215


namespace NUMINAMATH_CALUDE_smallest_sum_is_28_l1992_199221

/-- Converts a number from base 6 to base 10 --/
def base6To10 (x y z : Nat) : Nat :=
  36 * x + 6 * y + z

/-- Converts a number from base b to base 10 --/
def baseBTo10 (b : Nat) : Nat :=
  3 * b + 3

/-- Represents the conditions of the problem --/
def validConfiguration (x y z b : Nat) : Prop :=
  x ≤ 5 ∧ y ≤ 5 ∧ z ≤ 5 ∧ b > 6 ∧ base6To10 x y z = baseBTo10 b

theorem smallest_sum_is_28 :
  ∃ x y z b, validConfiguration x y z b ∧
  ∀ x' y' z' b', validConfiguration x' y' z' b' →
    x + y + z + b ≤ x' + y' + z' + b' ∧
    x + y + z + b = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_28_l1992_199221


namespace NUMINAMATH_CALUDE_vasya_floor_l1992_199293

theorem vasya_floor (steps_per_floor : ℕ) (petya_steps : ℕ) (vasya_steps : ℕ) : 
  steps_per_floor * 2 = petya_steps → 
  vasya_steps = steps_per_floor * 4 → 
  5 = vasya_steps / steps_per_floor + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_vasya_floor_l1992_199293


namespace NUMINAMATH_CALUDE_base4_sum_equals_2133_l1992_199279

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Converts a base 4 number to its decimal representation --/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal number to its base 4 representation --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Adds two base 4 numbers --/
def base4_add (a b : Base4) : Base4 := sorry

theorem base4_sum_equals_2133 :
  let a := to_base4 2
  let b := to_base4 (4 + 3)
  let c := to_base4 (16 + 12 + 2)
  let d := to_base4 (256 + 192 + 0)
  base4_add (base4_add (base4_add a b) c) d = to_base4 (512 + 48 + 12 + 3) := by
  sorry

end NUMINAMATH_CALUDE_base4_sum_equals_2133_l1992_199279


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l1992_199250

/-- The amount of coffee Carina has, given the following conditions:
  * She has coffee divided into 5- and 10-ounce packages
  * She has 2 more 5-ounce packages than 10-ounce packages
  * She has 5 10-ounce packages
-/
theorem carina_coffee_amount :
  let num_10oz_packages : ℕ := 5
  let num_5oz_packages : ℕ := num_10oz_packages + 2
  let total_ounces : ℕ := num_10oz_packages * 10 + num_5oz_packages * 5
  total_ounces = 85 := by sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l1992_199250


namespace NUMINAMATH_CALUDE_fourth_score_calculation_l1992_199298

def score1 : ℝ := 65
def score2 : ℝ := 67
def score3 : ℝ := 76
def average_score : ℝ := 76.6
def num_subjects : ℕ := 4

theorem fourth_score_calculation :
  ∃ (score4 : ℝ),
    (score1 + score2 + score3 + score4) / num_subjects = average_score ∧
    score4 = 98.4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_score_calculation_l1992_199298


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l1992_199207

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem
theorem M_intersect_N_eq_open_interval : M ∩ N = {x | -2 < x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l1992_199207


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1992_199284

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence where a₂ + 2a₆ + a₁₀ = 120, a₃ + a₉ = 60. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h_sum : a 2 + 2 * a 6 + a 10 = 120) : a 3 + a 9 = 60 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1992_199284


namespace NUMINAMATH_CALUDE_least_difference_consecutive_primes_l1992_199248

theorem least_difference_consecutive_primes (x y z p : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧  -- x, y, and z are prime numbers
  x < y ∧ y < z ∧  -- x < y < z
  y - x > 5 ∧  -- y - x > 5
  Even x ∧  -- x is an even integer
  Odd y ∧ Odd z ∧  -- y and z are odd integers
  (∃ k : ℕ, y^2 + x^2 = k * p) ∧  -- (y^2 + x^2) is divisible by a specific prime p
  Prime p →  -- p is prime
  (∃ s : ℕ, s = z - x ∧ ∀ t : ℕ, t = z - x → s ≤ t) ∧ s = 11  -- The least possible value s of z - x is 11
  := by sorry

end NUMINAMATH_CALUDE_least_difference_consecutive_primes_l1992_199248


namespace NUMINAMATH_CALUDE_function_inequality_l1992_199206

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_pos : ∀ x, f x > 0)
  (h_ineq : ∀ x, f x < x * deriv f x) :
  2 * f 1 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1992_199206


namespace NUMINAMATH_CALUDE_digits_of_2_pow_100_l1992_199246

theorem digits_of_2_pow_100 (N : ℕ) :
  (N = (Nat.digits 10 (2^100)).length) → 29 ≤ N ∧ N ≤ 34 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_100_l1992_199246


namespace NUMINAMATH_CALUDE_john_earnings_160_l1992_199290

/-- Calculates John's weekly streaming earnings --/
def johnWeeklyEarnings (daysOff : ℕ) (hoursPerDay : ℕ) (ratePerHour : ℕ) : ℕ :=
  let daysStreaming := 7 - daysOff
  let hoursPerWeek := daysStreaming * hoursPerDay
  hoursPerWeek * ratePerHour

/-- Theorem: John's weekly earnings are $160 --/
theorem john_earnings_160 :
  johnWeeklyEarnings 3 4 10 = 160 := by
  sorry

#eval johnWeeklyEarnings 3 4 10

end NUMINAMATH_CALUDE_john_earnings_160_l1992_199290


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1992_199254

-- Define the set A
def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 3}

-- Define the set B
def B : Set ℤ := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ Bᶜ = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1992_199254


namespace NUMINAMATH_CALUDE_p_fourth_minus_one_divisible_by_ten_l1992_199280

theorem p_fourth_minus_one_divisible_by_ten (p : ℕ) (hp : Prime p) (hp_not_two : p ≠ 2) (hp_not_five : p ≠ 5) :
  10 ∣ (p^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_p_fourth_minus_one_divisible_by_ten_l1992_199280


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l1992_199238

/-- Represents a cube with square holes cut through each face. -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a cube with square holes, including inside surfaces. -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length ^ 2
  let hole_area := 6 * cube.hole_side_length ^ 2
  let new_exposed_area := 6 * 4 * cube.hole_side_length ^ 2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 4 and hole side length 2 has a total surface area of 168. -/
theorem cube_with_holes_surface_area :
  let cube := CubeWithHoles.mk 4 2
  total_surface_area cube = 168 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l1992_199238


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_radius_relation_l1992_199277

/-- Given a tetrahedron with congruent triangular faces, an inscribed sphere,
    and a triangular face with known angles and circumradius,
    prove the relationship between the inscribed sphere radius and the face properties. -/
theorem tetrahedron_sphere_radius_relation
  (r : ℝ) -- radius of inscribed sphere
  (R : ℝ) -- radius of circumscribed circle of a face
  (α β γ : ℝ) -- angles of a triangular face
  (h_positive_r : r > 0)
  (h_positive_R : R > 0)
  (h_angle_sum : α + β + γ = π)
  (h_positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_congruent_faces : True) -- placeholder for the condition of congruent faces
  : r = R * Real.sqrt (Real.cos α * Real.cos β * Real.cos γ) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_radius_relation_l1992_199277


namespace NUMINAMATH_CALUDE_set_operations_l1992_199253

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 3}

-- Define the theorem
theorem set_operations :
  (A ∪ B = Set.univ) ∧
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x < 2 ∨ x ≥ 3}) := by
  sorry


end NUMINAMATH_CALUDE_set_operations_l1992_199253


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l1992_199274

/-- Given a quadratic polynomial 6x^2 + 36x + 150, prove that when rewritten in the form a(x+b)^2+c, 
    where a, b, and c are constants, a + b + c = 105 -/
theorem quadratic_rewrite_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (∀ x, 6*x^2 + 36*x + 150 = a*(x+b)^2 + c) ∧ (a + b + c = 105) := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l1992_199274


namespace NUMINAMATH_CALUDE_decimal_2015_is_octal_3737_l1992_199251

/-- Converts a natural number from decimal to octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid octal number -/
def is_valid_octal (l : List ℕ) : Prop :=
  l.all (· < 8) ∧ l ≠ []

theorem decimal_2015_is_octal_3737 :
  decimal_to_octal 2015 = [3, 7, 3, 7] ∧ is_valid_octal [3, 7, 3, 7] := by
  sorry

#eval decimal_to_octal 2015

end NUMINAMATH_CALUDE_decimal_2015_is_octal_3737_l1992_199251


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1992_199266

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1992_199266


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1992_199282

theorem point_in_second_quadrant :
  let x : ℝ := Real.sin (2014 * π / 180)
  let y : ℝ := Real.tan (2014 * π / 180)
  x < 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1992_199282


namespace NUMINAMATH_CALUDE_quadratic_function_and_triangle_area_l1992_199230

def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_function_and_triangle_area 
  (a b c : ℝ) 
  (h_opens_upward : a > 0)
  (h_not_origin : QuadraticFunction a b c 0 ≠ 0)
  (h_vertex : QuadraticFunction a b c 1 = -2)
  (x₁ x₂ : ℝ) 
  (h_roots : QuadraticFunction a b c x₁ = 0 ∧ QuadraticFunction a b c x₂ = 0)
  (h_y_intercept : (QuadraticFunction a b c 0)^2 = |x₁ * x₂|) :
  ((a = 1 ∧ b = -2 ∧ c = -1 ∧ (x₁ - x₂)^2 / 4 = 2) ∨
   (a = 1 + Real.sqrt 2 ∧ b = -(2 + 2 * Real.sqrt 2) ∧ c = Real.sqrt 2 - 1 ∧
    (x₁ - x₂)^2 / 4 = 2 * (Real.sqrt 2 - 1))) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_and_triangle_area_l1992_199230


namespace NUMINAMATH_CALUDE_inequality_proof_l1992_199220

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  a^3 + b^3 ≥ a^2*b + a*b^2 ∧ (1/a - 1)*(1/b - 1)*(1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1992_199220


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1992_199219

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 196713
  let d := 7
  let x := 6
  x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1992_199219


namespace NUMINAMATH_CALUDE_cubic_difference_equals_2011_l1992_199209

theorem cubic_difference_equals_2011 (x y : ℕ+) (h : x.val^2 - y.val^2 = 53) :
  x.val^3 - y.val^3 - 2 * (x.val + y.val) + 10 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_equals_2011_l1992_199209


namespace NUMINAMATH_CALUDE_alice_pears_l1992_199200

/-- The number of pears Alice sold -/
def sold : ℕ := sorry

/-- The number of pears Alice poached -/
def poached : ℕ := sorry

/-- The number of pears Alice canned -/
def canned : ℕ := sorry

/-- The total number of pears -/
def total : ℕ := 42

theorem alice_pears :
  (canned = poached + poached / 5) ∧
  (poached = sold / 2) ∧
  (sold + poached + canned = total) →
  sold = 20 := by sorry

end NUMINAMATH_CALUDE_alice_pears_l1992_199200


namespace NUMINAMATH_CALUDE_zero_smallest_natural_l1992_199299

theorem zero_smallest_natural : ∀ n : ℕ, 0 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_zero_smallest_natural_l1992_199299


namespace NUMINAMATH_CALUDE_exists_coplanar_even_sum_l1992_199232

-- Define a cube as a set of 8 integers (representing the labels on vertices)
def Cube := Fin 8 → ℤ

-- Define a function to check if a set of four vertices is coplanar
def isCoplanar (v1 v2 v3 v4 : Fin 8) : Prop := sorry

-- Define a function to check if the sum of four integers is even
def sumIsEven (a b c d : ℤ) : Prop :=
  (a + b + c + d) % 2 = 0

-- Theorem statement
theorem exists_coplanar_even_sum (cube : Cube) :
  ∃ (v1 v2 v3 v4 : Fin 8), isCoplanar v1 v2 v3 v4 ∧ sumIsEven (cube v1) (cube v2) (cube v3) (cube v4) := by
  sorry

end NUMINAMATH_CALUDE_exists_coplanar_even_sum_l1992_199232


namespace NUMINAMATH_CALUDE_function_properties_l1992_199217

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 10

theorem function_properties (m : ℝ) (h1 : m > 1) (h2 : f m m = 1) :
  ∃ (g : ℝ → ℝ),
    (∀ x, g x = x^2 - 6*x + 10) ∧
    (∀ x ∈ Set.Icc 3 5, g x ≤ 5) ∧
    (∀ x ∈ Set.Icc 3 5, g x ≥ 1) ∧
    (∃ x ∈ Set.Icc 3 5, g x = 5) ∧
    (∃ x ∈ Set.Icc 3 5, g x = 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1992_199217


namespace NUMINAMATH_CALUDE_loan_division_l1992_199243

/-- Given a total sum of 2730 divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is 1680. -/
theorem loan_division (total : ℝ) (part1 part2 : ℝ) : 
  total = 2730 →
  part1 + part2 = total →
  (part1 * 3 * 8) / 100 = (part2 * 5 * 3) / 100 →
  part2 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_loan_division_l1992_199243


namespace NUMINAMATH_CALUDE_kiwi_juice_blend_percentage_l1992_199267

/-- The amount of juice (in ounces) that can be extracted from one kiwi -/
def kiwi_juice : ℚ := 6 / 4

/-- The amount of juice (in ounces) that can be extracted from one apple -/
def apple_juice : ℚ := 10 / 3

/-- The number of kiwis used in the blend -/
def kiwis_in_blend : ℕ := 5

/-- The number of apples used in the blend -/
def apples_in_blend : ℕ := 4

/-- The percentage of kiwi juice in the blend -/
def kiwi_juice_percentage : ℚ := 
  (kiwi_juice * kiwis_in_blend) / 
  (kiwi_juice * kiwis_in_blend + apple_juice * apples_in_blend) * 100

theorem kiwi_juice_blend_percentage :
  kiwi_juice_percentage = 36 := by sorry

end NUMINAMATH_CALUDE_kiwi_juice_blend_percentage_l1992_199267


namespace NUMINAMATH_CALUDE_trig_identity_l1992_199281

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1992_199281


namespace NUMINAMATH_CALUDE_triangle_existence_l1992_199296

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2*(a^2*b^2 + a^2*c^2 + b^2*c^2)) :
  ∃ (α β γ : ℝ), α + β + γ = π ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_l1992_199296


namespace NUMINAMATH_CALUDE_prob_different_colors_7_5_l1992_199262

/-- The probability of drawing two chips of different colors from a bag with replacement -/
def prob_different_colors (red_chips green_chips : ℕ) : ℚ :=
  let total_chips := red_chips + green_chips
  let prob_red := red_chips / total_chips
  let prob_green := green_chips / total_chips
  2 * (prob_red * prob_green)

/-- Theorem stating that the probability of drawing two chips of different colors
    from a bag with 7 red chips and 5 green chips, with replacement, is 35/72 -/
theorem prob_different_colors_7_5 :
  prob_different_colors 7 5 = 35 / 72 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_7_5_l1992_199262


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1992_199202

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 * x - 5) ↔ x ≥ 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1992_199202


namespace NUMINAMATH_CALUDE_smallest_sum_squared_pythagorean_triple_l1992_199236

theorem smallest_sum_squared_pythagorean_triple (p q r : ℤ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) : 
  ∃ (p' q' r' : ℤ), p'^2 + q'^2 = r'^2 ∧ (p' + q' + r')^2 = 4 ∧ 
  ∀ (a b c : ℤ), a^2 + b^2 = c^2 → (a + b + c)^2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_squared_pythagorean_triple_l1992_199236


namespace NUMINAMATH_CALUDE_horner_method_correctness_l1992_199263

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℝ) : ℝ := 
  let v0 := 1
  let v1 := v0 * x + 0
  let v2 := v1 * x + 2
  let v3 := v2 * x + 3
  let v4 := v3 * x + 1
  v4 * x + 1

theorem horner_method_correctness : f 3 = horner_eval 3 := by sorry

end NUMINAMATH_CALUDE_horner_method_correctness_l1992_199263


namespace NUMINAMATH_CALUDE_min_adults_in_park_l1992_199271

theorem min_adults_in_park (total_people : ℕ) (total_amount : ℚ) 
  (adult_price youth_price child_price : ℚ) :
  total_people = 100 →
  total_amount = 100 →
  adult_price = 3 →
  youth_price = 2 →
  child_price = (3 : ℚ) / 10 →
  ∃ (adults youths children : ℕ),
    adults + youths + children = total_people ∧
    adult_price * adults + youth_price * youths + child_price * children = total_amount ∧
    adults = 2 ∧
    ∀ (a y c : ℕ),
      a + y + c = total_people →
      adult_price * a + youth_price * y + child_price * c = total_amount →
      a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_adults_in_park_l1992_199271


namespace NUMINAMATH_CALUDE_box_volume_increase_l1992_199270

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 3456, surface area is 1368, and sum of edges is 192,
    prove that increasing each dimension by 2 results in a volume of 5024 -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 3456)
  (hs : 2 * (l * w + w * h + h * l) = 1368)
  (he : 4 * (l + w + h) = 192) :
  (l + 2) * (w + 2) * (h + 2) = 5024 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1992_199270


namespace NUMINAMATH_CALUDE_solve_for_q_l1992_199297

theorem solve_for_q : ∀ (k h q : ℚ),
  (3/4 : ℚ) = k/48 ∧ 
  (3/4 : ℚ) = (h+k)/60 ∧ 
  (3/4 : ℚ) = (q-h)/80 →
  q = 69 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l1992_199297


namespace NUMINAMATH_CALUDE_solution_to_equation_l1992_199259

theorem solution_to_equation (x : ℝ) :
  Real.sqrt (4 * x^2 + 4 * x + 1) - Real.sqrt (4 * x^2 - 12 * x + 9) = 4 →
  x ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1992_199259


namespace NUMINAMATH_CALUDE_simplify_expression_combine_like_terms_l1992_199264

-- Define variables
variable (a b : ℝ)

-- Theorem 1
theorem simplify_expression :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b :=
by sorry

-- Theorem 2
theorem combine_like_terms :
  3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_combine_like_terms_l1992_199264


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1992_199276

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔ 
  (a ≤ -2 ∨ a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1992_199276


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1992_199237

/-- Given a geometric sequence with positive terms and common ratio not equal to 1,
    prove that the sum of the first and fourth terms is greater than
    the sum of the second and third terms. -/
theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (h1 : ∀ n, a n > 0)  -- All terms are positive
  (h2 : q ≠ 1)  -- Common ratio is not 1
  (h3 : ∀ n, a (n + 1) = a n * q)  -- Definition of geometric sequence
  : a 1 + a 4 > a 2 + a 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1992_199237


namespace NUMINAMATH_CALUDE_garden_walkway_area_l1992_199245

/-- Calculates the total area of walkways in a garden with given specifications -/
def walkway_area (rows : ℕ) (columns : ℕ) (bed_length : ℕ) (bed_width : ℕ) (walkway_width : ℕ) : ℕ :=
  let total_width := columns * bed_length + (columns + 1) * walkway_width
  let total_height := rows * bed_width + (rows + 1) * walkway_width
  let total_area := total_width * total_height
  let bed_area := rows * columns * bed_length * bed_width
  total_area - bed_area

theorem garden_walkway_area :
  walkway_area 4 3 8 3 2 = 416 :=
by sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l1992_199245


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l1992_199239

/-- The number of hours Annie spends on extracurriculars before midterms -/
def extracurricular_hours : ℕ :=
  let chess_hours : ℕ := 2
  let drama_hours : ℕ := 8
  let glee_hours : ℕ := 3
  let weekly_hours : ℕ := chess_hours + drama_hours + glee_hours
  let semester_weeks : ℕ := 12
  let midterm_weeks : ℕ := semester_weeks / 2
  let sick_weeks : ℕ := 2
  let active_weeks : ℕ := midterm_weeks - sick_weeks
  weekly_hours * active_weeks

/-- Theorem stating that Annie spends 52 hours on extracurriculars before midterms -/
theorem annie_extracurricular_hours : extracurricular_hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l1992_199239


namespace NUMINAMATH_CALUDE_problem_solution_l1992_199261

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  b / (a + b) + c / (b + c) + a / (c + a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1992_199261


namespace NUMINAMATH_CALUDE_function_value_implies_b_equals_negative_one_l1992_199278

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - b else 2^x

theorem function_value_implies_b_equals_negative_one (b : ℝ) :
  (f b (f b (1/2)) = 4) → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_implies_b_equals_negative_one_l1992_199278


namespace NUMINAMATH_CALUDE_simplify_t_l1992_199223

theorem simplify_t (t : ℝ) : t = 1 / (3 - Real.rpow 3 (1/3)) → t = (3 + Real.rpow 3 (1/3)) / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_t_l1992_199223


namespace NUMINAMATH_CALUDE_arithmetic_subsequence_l1992_199268

theorem arithmetic_subsequence (a : ℕ → ℝ) (d : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  ∃ c : ℝ, ∀ k : ℕ+, a (3 * k - 1) = c + (k - 1) * (3 * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_subsequence_l1992_199268


namespace NUMINAMATH_CALUDE_apple_distribution_l1992_199233

theorem apple_distribution (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) : 
  total_apples = 9 → num_friends = 3 → total_apples / num_friends = apples_per_friend → apples_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1992_199233


namespace NUMINAMATH_CALUDE_depth_difference_is_four_l1992_199235

/-- The depth of Mark's pond in feet -/
def marks_pond_depth : ℕ := 19

/-- The depth of Peter's pond in feet -/
def peters_pond_depth : ℕ := 5

/-- The difference between Mark's pond depth and 3 times Peter's pond depth -/
def depth_difference : ℕ := marks_pond_depth - 3 * peters_pond_depth

theorem depth_difference_is_four :
  depth_difference = 4 :=
by sorry

end NUMINAMATH_CALUDE_depth_difference_is_four_l1992_199235


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1992_199229

theorem complex_fraction_simplification (z : ℂ) :
  z = 1 - I → 2 / z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1992_199229


namespace NUMINAMATH_CALUDE_divided_proportion_problem_l1992_199225

theorem divided_proportion_problem (total : ℚ) (a b c : ℚ) (h1 : total = 782) 
  (h2 : a = 1/2) (h3 : b = 1/3) (h4 : c = 3/4) : 
  (total * a) / (a + b + c) = 247 := by
  sorry

end NUMINAMATH_CALUDE_divided_proportion_problem_l1992_199225


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l1992_199255

theorem at_least_two_equations_have_solution (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y : ℝ), (
    ((x - a) * (x - b) = x - c ∧ (y - b) * (y - c) = y - a) ∨
    ((x - a) * (x - b) = x - c ∧ (y - c) * (y - a) = y - b) ∨
    ((x - b) * (x - c) = x - a ∧ (y - c) * (y - a) = y - b)
  ) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l1992_199255


namespace NUMINAMATH_CALUDE_negation_of_implication_l1992_199218

theorem negation_of_implication (a : ℝ) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1992_199218


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1992_199234

theorem smallest_prime_divisor_of_sum (p : Nat) 
  (h1 : Prime 7) (h2 : Prime 11) : 
  (p.Prime ∧ p ∣ (7^13 + 11^15) ∧ ∀ q, q.Prime → q ∣ (7^13 + 11^15) → p ≤ q) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1992_199234


namespace NUMINAMATH_CALUDE_vector_magnitude_condition_l1992_199286

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_condition (a b : V) :
  ¬(∀ a b : V, ‖a‖ = ‖b‖ → ‖a + b‖ = ‖a - b‖) ∧
  ¬(∀ a b : V, ‖a + b‖ = ‖a - b‖ → ‖a‖ = ‖b‖) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_condition_l1992_199286


namespace NUMINAMATH_CALUDE_solve_sock_problem_l1992_199275

def sock_problem (lisa_initial : ℕ) (sandra : ℕ) (total : ℕ) : Prop :=
  let cousin := sandra / 5
  let before_mom := lisa_initial + sandra + cousin
  ∃ (mom : ℕ), before_mom + mom = total

theorem solve_sock_problem :
  sock_problem 12 20 80 → ∃ (mom : ℕ), mom = 44 := by
  sorry

end NUMINAMATH_CALUDE_solve_sock_problem_l1992_199275


namespace NUMINAMATH_CALUDE_abs_neg_five_equals_five_l1992_199216

theorem abs_neg_five_equals_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_equals_five_l1992_199216


namespace NUMINAMATH_CALUDE_smallest_multiples_of_17_l1992_199222

theorem smallest_multiples_of_17 :
  (∃ n : ℕ, n * 17 = 34 ∧ ∀ m : ℕ, m * 17 ≥ 10 ∧ m * 17 < 100 → m * 17 ≥ 34) ∧
  (∃ n : ℕ, n * 17 = 1003 ∧ ∀ m : ℕ, m * 17 ≥ 1000 ∧ m * 17 < 10000 → m * 17 ≥ 1003) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_of_17_l1992_199222


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1992_199211

theorem quadratic_inequality (x : ℝ) : x^2 - x - 12 < 0 ↔ -3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1992_199211


namespace NUMINAMATH_CALUDE_centers_form_rectangle_l1992_199295

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : Prop := ∃ (C C1 C2 C3 C4 : Circle),
  -- C has radius 2
  C.radius = 2 ∧
  -- C1 and C2 have radius 1
  C1.radius = 1 ∧ C2.radius = 1 ∧
  -- C1 and C2 touch at the center of C
  C1.center = C.center + (1, 0) ∧ C2.center = C.center + (-1, 0) ∧
  -- C3 is inside C and touches C, C1, and C2
  (∃ x : ℝ, C3.radius = x ∧
    dist C3.center C.center = 2 - x ∧
    dist C3.center C1.center = 1 + x ∧
    dist C3.center C2.center = 1 + x) ∧
  -- C4 is inside C and touches C, C1, and C3
  (∃ y : ℝ, C4.radius = y ∧
    dist C4.center C.center = 2 - y ∧
    dist C4.center C1.center = 1 + y ∧
    dist C4.center C3.center = C3.radius + y)

-- Define what it means for four points to form a rectangle
def form_rectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := dist p1 p2
  let d23 := dist p2 p3
  let d34 := dist p3 p4
  let d41 := dist p4 p1
  let d13 := dist p1 p3
  let d24 := dist p2 p4
  d12 = d34 ∧ d23 = d41 ∧ d13 = d24

-- Theorem statement
theorem centers_form_rectangle :
  problem_setup →
  ∃ (C C1 C3 C4 : Circle),
    form_rectangle C.center C1.center C3.center C4.center :=
sorry

end NUMINAMATH_CALUDE_centers_form_rectangle_l1992_199295


namespace NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_11_l1992_199283

theorem sum_six_smallest_multiples_of_11 : 
  (Finset.range 6).sum (fun i => 11 * (i + 1)) = 231 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_11_l1992_199283


namespace NUMINAMATH_CALUDE_greater_number_sum_difference_l1992_199265

theorem greater_number_sum_difference (x y : ℝ) 
  (sum_eq : x + y = 22) 
  (diff_eq : x - y = 4) : 
  max x y = 13 := by
sorry

end NUMINAMATH_CALUDE_greater_number_sum_difference_l1992_199265


namespace NUMINAMATH_CALUDE_min_decimal_digits_fraction_l1992_199227

theorem min_decimal_digits_fraction (n : ℕ) (d : ℕ) (h : n = 987654321 ∧ d = 2^30 * 5^5) :
  (∃ k : ℕ, k = 30 ∧ 
    ∀ m : ℕ, m < k → ∃ r : ℚ, r ≠ 0 ∧ (n : ℚ) / d * 10^m - ((n : ℚ) / d * 10^m).floor ≠ 0) ∧
    (∃ q : ℚ, (n : ℚ) / d = q ∧ (q * 10^30).floor / 10^30 = q) :=
sorry

end NUMINAMATH_CALUDE_min_decimal_digits_fraction_l1992_199227


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l1992_199247

/-- Given two people moving in opposite directions, this theorem proves
    the speed of one person given the speed of the other and their final distance. -/
theorem opposite_direction_speed 
  (pooja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : pooja_speed = 3) 
  (h2 : time = 4) 
  (h3 : final_distance = 32) : 
  ∃ (roja_speed : ℝ), roja_speed = 5 ∧ final_distance = (roja_speed + pooja_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l1992_199247


namespace NUMINAMATH_CALUDE_colin_speed_proof_l1992_199269

theorem colin_speed_proof (bruce_speed tony_speed brandon_speed colin_speed : ℝ) : 
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = (1 / 3) * tony_speed →
  colin_speed = 4 →
  ∃ (multiple : ℝ), colin_speed = multiple * brandon_speed ∧ colin_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_colin_speed_proof_l1992_199269


namespace NUMINAMATH_CALUDE_point_set_characterization_l1992_199285

theorem point_set_characterization (x y : ℝ) : 
  (∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 → t^2 + y*t + x ≥ 0) ↔ 
  (y ∈ Set.Icc (-2) 2 → x ≥ y^2/4) ∧ 
  (y < -2 → x ≥ -y - 1) ∧ 
  (y > 2 → x ≥ y - 1) := by
sorry

end NUMINAMATH_CALUDE_point_set_characterization_l1992_199285


namespace NUMINAMATH_CALUDE_boat_downstream_speed_l1992_199226

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a boat given its speed in still water and upstream -/
def downstreamSpeed (b : BoatSpeed) : ℝ :=
  2 * b.stillWater - b.upstream

/-- Theorem stating that a boat with 11 km/hr speed in still water and 7 km/hr upstream 
    will have a downstream speed of 15 km/hr -/
theorem boat_downstream_speed :
  let b : BoatSpeed := { stillWater := 11, upstream := 7 }
  downstreamSpeed b = 15 := by sorry

end NUMINAMATH_CALUDE_boat_downstream_speed_l1992_199226


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1992_199242

theorem min_value_of_expression (x y : ℝ) : (x * y - 2)^2 + (x + y - 1)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1992_199242


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l1992_199231

theorem real_solutions_quadratic (x y : ℝ) : 
  (3 * y^2 + 6 * x * y + 2 * x + 4 = 0) → 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ 
  (x ≤ -2/3 ∨ x ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l1992_199231


namespace NUMINAMATH_CALUDE_parabola_ratio_l1992_199287

-- Define the parabola R
def Parabola (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = a * p.1^2}

-- Define the vertex and focus of a parabola
def vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def focus (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the locus of midpoints
def midpointLocus (R : Set (ℝ × ℝ)) (W : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem parabola_ratio 
  (R : Set (ℝ × ℝ)) 
  (h_R : ∃ a : ℝ, R = Parabola a) 
  (W₁ : ℝ × ℝ) 
  (G₁ : ℝ × ℝ) 
  (h_W₁ : W₁ = vertex R) 
  (h_G₁ : G₁ = focus R) 
  (S : Set (ℝ × ℝ)) 
  (h_S : S = midpointLocus R W₁) 
  (W₂ : ℝ × ℝ) 
  (G₂ : ℝ × ℝ) 
  (h_W₂ : W₂ = vertex S) 
  (h_G₂ : G₂ = focus S) : 
  ‖G₁ - G₂‖ / ‖W₁ - W₂‖ = 1/4 := by sorry


end NUMINAMATH_CALUDE_parabola_ratio_l1992_199287


namespace NUMINAMATH_CALUDE_carrie_tomatoes_l1992_199212

/-- The number of tomatoes Carrie harvested -/
def tomatoes : ℕ := sorry

/-- The number of carrots Carrie harvested -/
def carrots : ℕ := 350

/-- The price of a tomato in dollars -/
def tomato_price : ℚ := 1

/-- The price of a carrot in dollars -/
def carrot_price : ℚ := 3/2

/-- The total revenue from selling all tomatoes and carrots in dollars -/
def total_revenue : ℚ := 725

theorem carrie_tomatoes : 
  tomatoes = 200 :=
sorry

end NUMINAMATH_CALUDE_carrie_tomatoes_l1992_199212


namespace NUMINAMATH_CALUDE_lucas_class_size_l1992_199260

theorem lucas_class_size : ∃! x : ℕ, 
  70 < x ∧ x < 120 ∧ 
  x % 6 = 4 ∧ 
  x % 5 = 2 ∧ 
  x % 7 = 3 ∧
  x = 148 := by
  sorry

end NUMINAMATH_CALUDE_lucas_class_size_l1992_199260


namespace NUMINAMATH_CALUDE_volume_between_concentric_spheres_l1992_199249

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 8
  let V₁ := (4 / 3) * π * r₁^3
  let V₂ := (4 / 3) * π * r₂^3
  V₂ - V₁ = (1792 / 3) * π := by sorry

end NUMINAMATH_CALUDE_volume_between_concentric_spheres_l1992_199249


namespace NUMINAMATH_CALUDE_bear_climbing_problem_l1992_199204

/-- Represents the mountain climbing problem with two bears -/
structure MountainClimb where
  S : ℝ  -- Total distance from base to summit in meters
  VA : ℝ  -- Bear A's ascending speed
  VB : ℝ  -- Bear B's ascending speed
  meetingTime : ℝ  -- Time when bears meet (in hours)
  meetingDistance : ℝ  -- Distance from summit where bears meet (in meters)

/-- The theorem statement for the mountain climbing problem -/
theorem bear_climbing_problem (m : MountainClimb) : 
  m.VA > m.VB ∧  -- Bear A is faster than Bear B
  m.meetingTime = 2 ∧  -- Bears meet after 2 hours
  m.meetingDistance = 1600 ∧  -- Bears meet 1600 meters from summit
  m.S - 1600 = 2 * m.meetingTime * (m.VA + m.VB) ∧  -- Meeting condition
  (m.S + 800) / (m.S - 1600) = 5 / 4 →  -- Condition when Bear B reaches summit
  (m.S / m.VA + m.S / (2 * m.VA)) = 14 / 5  -- Total time for Bear A
  := by sorry

end NUMINAMATH_CALUDE_bear_climbing_problem_l1992_199204


namespace NUMINAMATH_CALUDE_age_difference_l1992_199208

/-- Proves the age difference between a man and his son --/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 22 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1992_199208


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_l1992_199244

theorem complex_magnitude_equals_five (t : ℝ) (ht : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 5 → t = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_l1992_199244


namespace NUMINAMATH_CALUDE_fraction_zero_at_minus_one_denominator_nonzero_at_minus_one_largest_x_for_zero_fraction_l1992_199256

theorem fraction_zero_at_minus_one (x : ℝ) :
  (x + 1) / (9 * x^2 - 74 * x + 9) = 0 ↔ x = -1 :=
by
  sorry

theorem denominator_nonzero_at_minus_one :
  9 * (-1)^2 - 74 * (-1) + 9 ≠ 0 :=
by
  sorry

theorem largest_x_for_zero_fraction :
  ∀ y > -1, (y + 1) / (9 * y^2 - 74 * y + 9) ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_at_minus_one_denominator_nonzero_at_minus_one_largest_x_for_zero_fraction_l1992_199256


namespace NUMINAMATH_CALUDE_f_value_at_5_l1992_199203

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

-- State the theorem
theorem f_value_at_5 (a b c m : ℝ) :
  f a b c (-5) = m → f a b c 5 = -m + 4 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_5_l1992_199203


namespace NUMINAMATH_CALUDE_grid_property_l1992_199228

/-- Represents a 3x3 grid -/
structure Grid :=
  (cells : Matrix (Fin 3) (Fin 3) ℤ)

/-- Represents an operation on the grid -/
inductive Operation
  | add_adjacent : Fin 3 → Fin 3 → Fin 3 → Fin 3 → Operation
  | subtract_adjacent : Fin 3 → Fin 3 → Fin 3 → Fin 3 → Operation

/-- Applies an operation to a grid -/
def apply_operation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- The sum of all cells in a grid -/
def grid_sum (g : Grid) : ℤ :=
  sorry

/-- The difference between shaded and non-shaded cells -/
def shaded_difference (g : Grid) (shaded : Set (Fin 3 × Fin 3)) : ℤ :=
  sorry

/-- Theorem stating the property of the grid after operations -/
theorem grid_property (initial : Grid) (final : Grid) (ops : List Operation) 
    (shaded : Set (Fin 3 × Fin 3)) :
  (∀ op ∈ ops, grid_sum (apply_operation initial op) = grid_sum initial) →
  (∀ op ∈ ops, shaded_difference (apply_operation initial op) shaded = shaded_difference initial shaded) →
  (∃ A : ℤ, final.cells 0 0 = A ∧ 4 * 2010 + A - 4 * 2010 = 5) →
  final.cells 0 0 = 5 :=
by sorry

end NUMINAMATH_CALUDE_grid_property_l1992_199228


namespace NUMINAMATH_CALUDE_similar_triangle_leg_length_l1992_199240

theorem similar_triangle_leg_length 
  (a b c d : ℝ) 
  (h1 : a^2 + 24^2 = 25^2) 
  (h2 : b^2 + c^2 = d^2) 
  (h3 : d / 25 = 100 / 25) 
  (h4 : b / a = d / 25) 
  (h5 : c / 24 = d / 25) : 
  c = 28 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_leg_length_l1992_199240


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1992_199205

theorem arithmetic_series_sum : ∀ (a₁ aₙ : ℤ) (d : ℤ) (n : ℕ),
  a₁ = -41 →
  aₙ = 1 →
  d = 2 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℤ) * (a₁ + aₙ) / 2 = -440 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l1992_199205


namespace NUMINAMATH_CALUDE_granola_initial_price_l1992_199213

/-- Proves that the initial selling price per bag was $6.00 --/
theorem granola_initial_price (ingredient_cost : ℝ) (total_bags : ℕ) 
  (full_price_sold : ℕ) (discount_price : ℝ) (net_profit : ℝ) :
  ingredient_cost = 3 →
  total_bags = 20 →
  full_price_sold = 15 →
  discount_price = 4 →
  net_profit = 50 →
  ∃ (initial_price : ℝ), 
    initial_price * full_price_sold + discount_price * (total_bags - full_price_sold) - 
    ingredient_cost * total_bags = net_profit ∧
    initial_price = 6 := by
  sorry

#check granola_initial_price

end NUMINAMATH_CALUDE_granola_initial_price_l1992_199213


namespace NUMINAMATH_CALUDE_kevin_has_eight_toads_l1992_199241

/-- The number of worms each toad eats daily -/
def worms_per_toad : ℕ := 3

/-- The time in minutes it takes Kevin to find one worm -/
def minutes_per_worm : ℕ := 15

/-- The total time in hours Kevin spends finding worms -/
def total_hours : ℕ := 6

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

/-- Calculates the number of toads Kevin has -/
def number_of_toads : ℕ :=
  (hours_to_minutes total_hours) / minutes_per_worm / worms_per_toad

theorem kevin_has_eight_toads : number_of_toads = 8 := by
  sorry

end NUMINAMATH_CALUDE_kevin_has_eight_toads_l1992_199241


namespace NUMINAMATH_CALUDE_triangle_max_area_l1992_199289

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a * Real.cos C - c / 2 = b →
  a = 2 * Real.sqrt 3 →
  (∃ (S : ℝ), S = (1 / 2) * b * c * Real.sin A ∧
    ∀ (S' : ℝ), S' = (1 / 2) * b * c * Real.sin A → S' ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1992_199289


namespace NUMINAMATH_CALUDE_total_erasers_l1992_199257

/-- Given an initial number of erasers and a number of erasers added, 
    the total number of erasers is equal to the sum of the initial number and the added number. -/
theorem total_erasers (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_erasers_l1992_199257


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1992_199252

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  Prime p ∧ p ∣ (2^12 + 3^10 + 7^15) → p = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1992_199252


namespace NUMINAMATH_CALUDE_darcy_remaining_clothes_l1992_199272

def remaining_clothes_to_fold (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts + total_shorts) - (folded_shirts + folded_shorts)

theorem darcy_remaining_clothes : 
  remaining_clothes_to_fold 20 8 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_darcy_remaining_clothes_l1992_199272


namespace NUMINAMATH_CALUDE_island_population_l1992_199214

theorem island_population (centipedes humans sheep : ℕ) : 
  centipedes = 100 →
  centipedes = 2 * humans →
  sheep = humans / 2 →
  sheep + humans = 75 := by
sorry

end NUMINAMATH_CALUDE_island_population_l1992_199214


namespace NUMINAMATH_CALUDE_distribute_equals_choose_l1992_199291

/-- The number of ways to distribute n indistinguishable objects into k distinct groups,
    with each group receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribute_equals_choose :
  distribute 10 7 = choose 9 6 := by sorry

end NUMINAMATH_CALUDE_distribute_equals_choose_l1992_199291


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1992_199273

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 9 ∣ n ∧
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ 3 ∣ m ∧ 4 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  n = 108 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1992_199273


namespace NUMINAMATH_CALUDE_ellipse_equation_l1992_199201

/-- Given a circle and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
     ((x = 0 ∧ y = b) ∨ (x = 0 ∧ y = -b) ∨ 
      (y = 0 ∧ x^2 = a^2 - b^2) ∨ (y = 0 ∧ x^2 = a^2 - b^2)))) →
  a^2 = 8 ∧ b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1992_199201


namespace NUMINAMATH_CALUDE_chris_savings_l1992_199292

/-- Chris's savings problem -/
theorem chris_savings (total : ℕ) (grandmother : ℕ) (parents : ℕ) (aunt_uncle : ℕ) 
  (h1 : total = 279)
  (h2 : grandmother = 25)
  (h3 : parents = 75)
  (h4 : aunt_uncle = 20) :
  total - (grandmother + parents + aunt_uncle) = 159 := by
  sorry

end NUMINAMATH_CALUDE_chris_savings_l1992_199292
