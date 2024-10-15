import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1627_162787

-- Define the inverse proportionality relation
def inversely_proportional (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, a x * b x = k

theorem inverse_proportion_ratio 
  (a b : ℝ → ℝ) (a₁ a₂ b₁ b₂ : ℝ) :
  inversely_proportional a b →
  a₁ ≠ 0 → a₂ ≠ 0 → b₁ ≠ 0 → b₂ ≠ 0 →
  a₁ / a₂ = 3 / 4 →
  b₁ - b₂ = 5 →
  b₁ / b₂ = 4 / 3 := by
    sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1627_162787


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1627_162753

theorem constant_ratio_problem (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) (k : ℝ) :
  (3 * x₁ - 4) / (y₁ + 7) = k →
  (3 * x₂ - 4) / (y₂ + 7) = k →
  x₁ = 3 →
  y₁ = 5 →
  y₂ = 20 →
  x₂ = 5.0833 := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1627_162753


namespace NUMINAMATH_CALUDE_no_integer_solution_for_2007_l1627_162786

theorem no_integer_solution_for_2007 :
  ¬ ∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_2007_l1627_162786


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l1627_162722

/-- The greatest exponent x such that 3^x divides 22! is 9 -/
theorem greatest_power_of_three_in_factorial : 
  (∃ x : ℕ, x = 9 ∧ 
    (∀ y : ℕ, 3^y ∣ Nat.factorial 22 → y ≤ x) ∧
    (3^x ∣ Nat.factorial 22)) := by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l1627_162722


namespace NUMINAMATH_CALUDE_intersection_count_504_220_l1627_162706

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A line segment from (0,0) to (a,b) -/
structure LineSegment where
  a : ℤ
  b : ℤ

/-- Count of intersections with squares and circles -/
structure IntersectionCount where
  squares : ℕ
  circles : ℕ

/-- Function to count intersections of a line segment with squares and circles -/
def countIntersections (l : LineSegment) : IntersectionCount :=
  sorry

theorem intersection_count_504_220 :
  let l : LineSegment := ⟨504, 220⟩
  let count : IntersectionCount := countIntersections l
  count.squares + count.circles = 255 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_504_220_l1627_162706


namespace NUMINAMATH_CALUDE_cos_2theta_value_l1627_162768

theorem cos_2theta_value (θ : ℝ) (h : Real.tan (θ + π/4) = (1/2) * Real.tan θ - 7/2) : 
  Real.cos (2 * θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l1627_162768


namespace NUMINAMATH_CALUDE_average_sales_per_month_l1627_162794

def sales_data : List ℕ := [100, 60, 40, 120]

theorem average_sales_per_month :
  (List.sum sales_data) / (List.length sales_data) = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_per_month_l1627_162794


namespace NUMINAMATH_CALUDE_factory_employee_count_l1627_162737

/-- Given a factory with three workshops and stratified sampling information, 
    prove the total number of employees. -/
theorem factory_employee_count 
  (x : ℕ) -- number of employees in Workshop A
  (y : ℕ) -- number of employees in Workshop C
  (h1 : x + 300 + y = 900) -- total employees
  (h2 : 20 + 15 + 10 = 45) -- stratified sample
  : x + 300 + y = 900 := by
  sorry

#check factory_employee_count

end NUMINAMATH_CALUDE_factory_employee_count_l1627_162737


namespace NUMINAMATH_CALUDE_finite_solutions_l1627_162748

def f (z : ℂ) : ℂ := z^2 + Complex.I * z + 1

theorem finite_solutions :
  ∃ (S : Finset ℂ), ∀ z : ℂ,
    Complex.im z > 0 ∧
    (∃ (a b : ℤ), f z = ↑a + ↑b * Complex.I ∧ abs a ≤ 15 ∧ abs b ≤ 15) →
    z ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_l1627_162748


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l1627_162710

/-- The focus of a parabola y = ax^2 + k is at (0, k - 1/(4a)) when a ≠ 0 -/
theorem parabola_focus (a k : ℝ) (ha : a ≠ 0) :
  let f : ℝ × ℝ := (0, k - 1 / (4 * a))
  ∀ x y : ℝ, y = a * x^2 + k → (x - f.1)^2 + (y - f.2)^2 = (y - k + 1 / (4 * a))^2 / (4 * a^2) :=
sorry

/-- The focus of the parabola y = -2x^2 + 4 is at (0, 33/8) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, 33/8)
  ∀ x y : ℝ, y = -2 * x^2 + 4 → (x - f.1)^2 + (y - f.2)^2 = (y - 4 + 1/8)^2 / 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l1627_162710


namespace NUMINAMATH_CALUDE_parabola_same_side_l1627_162729

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by a quadratic function -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two points are on the same side of a parabola -/
def sameSide (p : Parabola) (A B : Point) : Prop :=
  (p.a * A.x^2 + p.b * A.x + p.c - A.y) * (p.a * B.x^2 + p.b * B.x + p.c - B.y) > 0

/-- The main theorem to prove -/
theorem parabola_same_side :
  let A : Point := ⟨-1, -1⟩
  let B : Point := ⟨0, 2⟩
  let p1 : Parabola := ⟨2, 4, 0⟩
  let p2 : Parabola := ⟨-1, 2, -1⟩
  let p3 : Parabola := ⟨-1, 0, 3⟩
  let p4 : Parabola := ⟨1/2, -1, -3/2⟩
  let p5 : Parabola := ⟨-1, -4, -3⟩
  sameSide p1 A B ∧ sameSide p2 A B ∧ sameSide p3 A B ∧
  ¬(sameSide p4 A B) ∧ ¬(sameSide p5 A B) :=
by sorry


end NUMINAMATH_CALUDE_parabola_same_side_l1627_162729


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l1627_162714

-- Define the types for lines and planes
variable (L : Type*) [AddCommGroup L] [Module ℝ L]
variable (P : Type*) [AddCommGroup P] [Module ℝ P]

-- Define the perpendicular and parallel relations
variable (perpendicular : L → P → Prop)
variable (parallel : L → L → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel
  (a b : L) (M : P)
  (h1 : perpendicular a M)
  (h2 : perpendicular b M) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l1627_162714


namespace NUMINAMATH_CALUDE_intersection_A_B_l1627_162728

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1627_162728


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l1627_162757

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l1627_162757


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l1627_162799

def g (x : ℝ) : ℝ := 12 * x^4 - 8 * x^2 + 1

theorem smallest_root_of_g :
  let r := Real.sqrt (1/6)
  (g r = 0) ∧ (∀ x : ℝ, g x = 0 → x ≥ 0 → x ≥ r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l1627_162799


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1627_162781

theorem smallest_integer_with_remainders : ∃ n : ℕ,
  n > 1 ∧
  n % 13 = 2 ∧
  n % 7 = 2 ∧
  n % 3 = 2 ∧
  ∀ m : ℕ, m > 1 → m % 13 = 2 → m % 7 = 2 → m % 3 = 2 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1627_162781


namespace NUMINAMATH_CALUDE_range_of_positive_values_l1627_162761

def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem range_of_positive_values 
  (f : ℝ → ℝ) 
  (odd : OddFunction f)
  (incr_neg : ∀ x y, x < y ∧ y ≤ 0 → f x < f y)
  (f_neg_one_zero : f (-1) = 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_positive_values_l1627_162761


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1627_162792

def point_A : ℝ × ℝ := (-3, 2)

def move_right (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 + units, p.2)

def move_down (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - units)

def point_B : ℝ × ℝ :=
  move_down (move_right point_A 1) 2

theorem point_B_coordinates :
  point_B = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l1627_162792


namespace NUMINAMATH_CALUDE_daily_profit_function_l1627_162709

/-- The daily profit function for a product with given cost and sales quantity relation -/
theorem daily_profit_function (x : ℝ) : 
  let cost : ℝ := 8
  let sales_quantity : ℝ → ℝ := λ price => -price + 30
  let profit : ℝ → ℝ := λ price => (price - cost) * (sales_quantity price)
  profit x = -x^2 + 38*x - 240 := by
sorry

end NUMINAMATH_CALUDE_daily_profit_function_l1627_162709


namespace NUMINAMATH_CALUDE_power_two_vs_square_l1627_162775

theorem power_two_vs_square (n : ℕ+) :
  (n = 2 ∨ n = 4 → 2^(n:ℕ) = n^2) ∧
  (n = 3 → 2^(n:ℕ) < n^2) ∧
  (n = 1 ∨ n > 4 → 2^(n:ℕ) > n^2) := by
  sorry

end NUMINAMATH_CALUDE_power_two_vs_square_l1627_162775


namespace NUMINAMATH_CALUDE_largest_non_expressible_l1627_162703

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ a b, a > 0 ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_expressible : 
  (∀ n > 188, is_expressible n) ∧ ¬is_expressible 188 :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l1627_162703


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l1627_162772

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_coefficient_sum (p : Parabola) :
  p.x_coord (-4) = 5 →
  p.x_coord (-2) = 3 →
  p.a + p.b + p.c = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l1627_162772


namespace NUMINAMATH_CALUDE_area_difference_value_l1627_162730

/-- A square with side length 2 units -/
def square_side : ℝ := 2

/-- Right-angled isosceles triangle with legs of length 2 -/
def large_triangle_leg : ℝ := 2

/-- Right-angled isosceles triangle with legs of length 1 -/
def small_triangle_leg : ℝ := 1

/-- The region R formed by the union of the square and all triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- The smallest convex polygon S containing R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of the region inside S but outside R -/
def area_difference : ℝ := sorry

/-- Theorem stating the area difference between S and R -/
theorem area_difference_value : area_difference = (27 * Real.sqrt 3 - 28) / 2 := by sorry

end NUMINAMATH_CALUDE_area_difference_value_l1627_162730


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l1627_162720

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_sock_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem stating the number of ways to choose a pair of socks of different colors
    given the specific quantities of each color -/
theorem sock_pair_combinations :
  different_color_sock_pairs 5 3 3 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l1627_162720


namespace NUMINAMATH_CALUDE_stating_pyramid_levels_for_1023_toothpicks_l1627_162739

/-- Represents the number of toothpicks in a pyramid level. -/
def toothpicks_in_level (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of toothpicks used up to a given level. -/
def total_toothpicks (n : ℕ) : ℕ := 2^n - 1

/-- 
Theorem stating that a pyramid with 1023 toothpicks has 10 levels,
where each level doubles the number of toothpicks from the previous level.
-/
theorem pyramid_levels_for_1023_toothpicks : 
  ∃ n : ℕ, n = 10 ∧ total_toothpicks n = 1023 := by
  sorry


end NUMINAMATH_CALUDE_stating_pyramid_levels_for_1023_toothpicks_l1627_162739


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1627_162736

theorem imaginary_part_of_complex_fraction :
  Complex.im ((1 + 2*Complex.I) / (1 + Complex.I)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1627_162736


namespace NUMINAMATH_CALUDE_area_of_square_e_l1627_162778

/-- Given a rectangle composed of squares a, b, c, d, and e, prove the area of square e. -/
theorem area_of_square_e (a b c d e : ℝ) : 
  a + b + c = 30 →
  a + b = 22 →
  2 * c + e = 22 →
  e^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_e_l1627_162778


namespace NUMINAMATH_CALUDE_arts_students_count_l1627_162791

/-- Represents the number of arts students in the college -/
def arts_students : ℕ := sorry

/-- Represents the number of local arts students -/
def local_arts_students : ℕ := sorry

/-- Represents the number of local science students -/
def local_science_students : ℕ := 25

/-- Represents the number of local commerce students -/
def local_commerce_students : ℕ := 102

/-- Represents the total number of local students -/
def total_local_students : ℕ := 327

/-- Theorem stating that the number of arts students is 400 -/
theorem arts_students_count :
  (local_arts_students = arts_students / 2) ∧
  (local_arts_students + local_science_students + local_commerce_students = total_local_students) →
  arts_students = 400 :=
by sorry

end NUMINAMATH_CALUDE_arts_students_count_l1627_162791


namespace NUMINAMATH_CALUDE_planar_graph_properties_l1627_162715

structure PlanarGraph where
  s : ℕ  -- number of vertices
  a : ℕ  -- number of edges
  f : ℕ  -- number of faces

def no_triangular_faces (G : PlanarGraph) : Prop :=
  -- This is a placeholder for the condition that no face is a triangle
  True

theorem planar_graph_properties (G : PlanarGraph) :
  (G.s - G.a + G.f = 2) ∧
  (G.a ≤ 3 * G.s - 6) ∧
  (no_triangular_faces G → G.a ≤ 2 * G.s - 4) := by
  sorry

end NUMINAMATH_CALUDE_planar_graph_properties_l1627_162715


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1627_162765

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {x | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1627_162765


namespace NUMINAMATH_CALUDE_sin_plus_tan_special_angle_l1627_162777

/-- 
If the terminal side of angle α passes through point (4,-3), 
then sin α + tan α = -27/20 
-/
theorem sin_plus_tan_special_angle (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) → 
  Real.sin α + Real.tan α = -27/20 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_tan_special_angle_l1627_162777


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l1627_162764

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + m * p.1 + 2}

def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_m_range_l1627_162764


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l1627_162751

theorem cosine_sine_identity (α : Real) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6)^2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l1627_162751


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1627_162716

theorem no_solution_quadratic_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1627_162716


namespace NUMINAMATH_CALUDE_g_50_not_18_l1627_162782

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The smallest positive divisor of n -/
def smallest_divisor (n : ℕ+) : ℕ+ := sorry

/-- g₁ function as defined in the problem -/
def g₁ (n : ℕ+) : ℕ := (num_divisors n) * (smallest_divisor n).val

/-- General gⱼ function for j ≥ 1 -/
def g (j : ℕ) (n : ℕ+) : ℕ :=
  match j with
  | 0 => n.val
  | 1 => g₁ n
  | j+1 => g₁ ⟨g j n, sorry⟩

/-- Main theorem: For all positive integers n ≤ 100, g₅₀(n) ≠ 18 -/
theorem g_50_not_18 : ∀ n : ℕ+, n.val ≤ 100 → g 50 n ≠ 18 := by sorry

end NUMINAMATH_CALUDE_g_50_not_18_l1627_162782


namespace NUMINAMATH_CALUDE_marcel_corn_count_l1627_162741

/-- The number of ears of corn Marcel bought -/
def marcel_corn : ℕ := sorry

/-- The number of ears of corn Dale bought -/
def dale_corn : ℕ := sorry

/-- The number of potatoes Dale bought -/
def dale_potatoes : ℕ := 8

/-- The number of potatoes Marcel bought -/
def marcel_potatoes : ℕ := 4

/-- The total number of vegetables bought -/
def total_vegetables : ℕ := 27

theorem marcel_corn_count :
  (dale_corn = marcel_corn / 2) →
  (dale_potatoes = 8) →
  (marcel_potatoes = 4) →
  (marcel_corn + dale_corn + dale_potatoes + marcel_potatoes = total_vegetables) →
  marcel_corn = 10 := by sorry

end NUMINAMATH_CALUDE_marcel_corn_count_l1627_162741


namespace NUMINAMATH_CALUDE_one_right_intersection_implies_negative_n_l1627_162789

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a quadratic function has one intersection point with the x-axis to the right of the y-axis -/
def hasOneRightIntersection (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f.a * x^2 + f.b * x + f.c = 0 ∧
  ∀ y : ℝ, y ≠ x → f.a * y^2 + f.b * y + f.c ≠ 0

/-- Theorem: If a quadratic function y = x^2 + 3x + n has one intersection point
    with the x-axis to the right of the y-axis, then n < 0 -/
theorem one_right_intersection_implies_negative_n :
  ∀ n : ℝ, hasOneRightIntersection ⟨1, 3, n⟩ → n < 0 := by
  sorry


end NUMINAMATH_CALUDE_one_right_intersection_implies_negative_n_l1627_162789


namespace NUMINAMATH_CALUDE_zeros_in_fraction_l1627_162760

def count_leading_zeros (n : ℚ) : ℕ :=
  sorry

theorem zeros_in_fraction : count_leading_zeros (1 / (2^7 * 5^9)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_fraction_l1627_162760


namespace NUMINAMATH_CALUDE_fraction_difference_equality_l1627_162732

theorem fraction_difference_equality (x y : ℝ) : 
  let P := x^2 + y^2
  let Q := x - y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_equality_l1627_162732


namespace NUMINAMATH_CALUDE_cricket_game_run_rate_l1627_162798

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.target - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem statement for the cricket game scenario -/
theorem cricket_game_run_rate 
  (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.2)
  (h4 : game.target = 282) :
  requiredRunRate game = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_run_rate_l1627_162798


namespace NUMINAMATH_CALUDE_yuri_total_puppies_l1627_162725

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

def total_puppies : ℕ := puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4

theorem yuri_total_puppies : total_puppies = 74 := by
  sorry

end NUMINAMATH_CALUDE_yuri_total_puppies_l1627_162725


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1627_162788

theorem product_of_three_numbers (a b c m : ℝ) 
  (sum_eq : a + b + c = 195)
  (m_eq_8a : m = 8 * a)
  (m_eq_b_minus_10 : m = b - 10)
  (m_eq_c_plus_10 : m = c + 10)
  (a_smallest : a < b ∧ a < c) :
  a * b * c = 95922 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1627_162788


namespace NUMINAMATH_CALUDE_area_gray_region_l1627_162767

/-- The area of the gray region between two concentric circles -/
theorem area_gray_region (r : ℝ) (h1 : r > 0) (h2 : 2 * r = r + 3) : 
  π * (2 * r)^2 - π * r^2 = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_area_gray_region_l1627_162767


namespace NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l1627_162797

/-- The Kwik-e-Tax Center problem -/
theorem kwik_e_tax_center_problem 
  (federal_charge : ℕ) 
  (state_charge : ℕ) 
  (quarterly_charge : ℕ)
  (federal_sold : ℕ) 
  (state_sold : ℕ) 
  (total_revenue : ℕ)
  (h1 : federal_charge = 50)
  (h2 : state_charge = 30)
  (h3 : quarterly_charge = 80)
  (h4 : federal_sold = 60)
  (h5 : state_sold = 20)
  (h6 : total_revenue = 4400) :
  ∃ (quarterly_sold : ℕ), 
    federal_charge * federal_sold + 
    state_charge * state_sold + 
    quarterly_charge * quarterly_sold = total_revenue ∧ 
    quarterly_sold = 10 := by
  sorry

end NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l1627_162797


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1627_162759

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1627_162759


namespace NUMINAMATH_CALUDE_power_six_mod_five_remainder_six_power_23_mod_five_l1627_162700

theorem power_six_mod_five (n : ℕ) : 6^n ≡ 1 [ZMOD 5] := by sorry

theorem remainder_six_power_23_mod_five : 6^23 ≡ 1 [ZMOD 5] := by sorry

end NUMINAMATH_CALUDE_power_six_mod_five_remainder_six_power_23_mod_five_l1627_162700


namespace NUMINAMATH_CALUDE_solution_set_f_positive_l1627_162747

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem solution_set_f_positive
    (f : ℝ → ℝ)
    (h_even : EvenFunction f)
    (h_nonneg : ∀ x ≥ 0, f x = 2^x - 4) :
    {x : ℝ | f x > 0} = Set.Ioi 2 ∪ Set.Iio (-2) :=
  sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_l1627_162747


namespace NUMINAMATH_CALUDE_exactly_one_number_satisfies_condition_l1627_162763

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 700 ∧ n = 7 * sum_of_digits n

theorem exactly_one_number_satisfies_condition : 
  ∃! n : ℕ, satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_exactly_one_number_satisfies_condition_l1627_162763


namespace NUMINAMATH_CALUDE_shorterToLongerBaseRatio_l1627_162749

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the shorter base -/
  s : ℝ
  /-- Length of the longer base -/
  t : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The length of the longer base is equal to the length of its diagonals -/
  longerBaseEqualsDiagonal : True
  /-- The length of the shorter base is equal to the height -/
  shorterBaseEqualsHeight : True

/-- The ratio of the shorter base to the longer base is 3/5 -/
theorem shorterToLongerBaseRatio (trap : IsoscelesTrapezoid) : 
  trap.s / trap.t = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shorterToLongerBaseRatio_l1627_162749


namespace NUMINAMATH_CALUDE_cosine_sine_square_difference_l1627_162779

theorem cosine_sine_square_difference (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos α ^ 2 - Real.sin α ^ 2 = -(Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_square_difference_l1627_162779


namespace NUMINAMATH_CALUDE_point_location_l1627_162744

theorem point_location (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 5 - 2 * x) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end NUMINAMATH_CALUDE_point_location_l1627_162744


namespace NUMINAMATH_CALUDE_discount_difference_l1627_162701

def initial_amount : ℝ := 12000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def scheme1 (amount : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount amount 0.25) 0.15) 0.10

def scheme2 (amount : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount amount 0.30) 0.10) 0.05

theorem discount_difference :
  scheme1 initial_amount - scheme2 initial_amount = 297 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1627_162701


namespace NUMINAMATH_CALUDE_kyle_caught_14_fish_l1627_162793

/-- The number of fish Kyle caught given the conditions of the problem -/
def kyles_fish (total : ℕ) (carlas : ℕ) : ℕ :=
  (total - carlas) / 2

/-- Theorem stating that Kyle caught 14 fish under the given conditions -/
theorem kyle_caught_14_fish (total : ℕ) (carlas : ℕ) 
  (h1 : total = 36) 
  (h2 : carlas = 8) : 
  kyles_fish total carlas = 14 := by
  sorry

#eval kyles_fish 36 8

end NUMINAMATH_CALUDE_kyle_caught_14_fish_l1627_162793


namespace NUMINAMATH_CALUDE_select_two_from_six_l1627_162718

theorem select_two_from_six (n : ℕ) (k : ℕ) : n = 6 → k = 2 → Nat.choose n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_six_l1627_162718


namespace NUMINAMATH_CALUDE_abc_inequality_l1627_162738

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1627_162738


namespace NUMINAMATH_CALUDE_bench_press_changes_l1627_162721

/-- Calculates the final bench press weight after a series of changes -/
def final_bench_press (initial_weight : ℝ) : ℝ :=
  let after_injury := initial_weight * (1 - 0.8)
  let after_recovery := after_injury * (1 + 0.6)
  let after_setback := after_recovery * (1 - 0.2)
  let final_weight := after_setback * 3
  final_weight

/-- Theorem stating that the final bench press weight is 384 pounds -/
theorem bench_press_changes (initial_weight : ℝ) 
  (h : initial_weight = 500) : 
  final_bench_press initial_weight = 384 := by
  sorry

#eval final_bench_press 500

end NUMINAMATH_CALUDE_bench_press_changes_l1627_162721


namespace NUMINAMATH_CALUDE_difference_of_fractions_l1627_162723

theorem difference_of_fractions : 
  (3 - 390 / 5) - (4 - 210 / 7) = -49 := by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l1627_162723


namespace NUMINAMATH_CALUDE_hypotenuse_of_45_45_90_triangle_l1627_162717

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  angle_opposite_leg1 : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem hypotenuse_of_45_45_90_triangle 
  (triangle : RightTriangle) 
  (h1 : triangle.leg1 = 12)
  (h2 : triangle.angle_opposite_leg1 = 45) :
  triangle.hypotenuse = 12 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_of_45_45_90_triangle_l1627_162717


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1627_162705

/-- Theorem: For the quadratic equation x^2 + x - 2 = m, when m > 0, the equation has two distinct real roots. -/
theorem quadratic_equation_roots (m : ℝ) (h : m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ - 2 = m ∧ x₂^2 + x₂ - 2 = m :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1627_162705


namespace NUMINAMATH_CALUDE_point_reflection_fourth_to_second_l1627_162785

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If A(a,b) is in the fourth quadrant, then B(-b,-a) is in the second quadrant -/
theorem point_reflection_fourth_to_second (a b : ℝ) :
  is_in_fourth_quadrant (Point.mk a b) →
  is_in_second_quadrant (Point.mk (-b) (-a)) := by
  sorry


end NUMINAMATH_CALUDE_point_reflection_fourth_to_second_l1627_162785


namespace NUMINAMATH_CALUDE_fatima_phone_probability_l1627_162771

def first_three_digits : List ℕ := [295, 296, 299]
def base_last_four : List ℕ := [1, 6, 7]

def possible_numbers : ℕ := sorry

theorem fatima_phone_probability :
  (1 : ℚ) / possible_numbers = (1 : ℚ) / 72 := by sorry

end NUMINAMATH_CALUDE_fatima_phone_probability_l1627_162771


namespace NUMINAMATH_CALUDE_equation_of_line_l_equations_of_line_m_l1627_162702

-- Define the slope of line l
def slope_l : ℚ := -3/4

-- Define the equation of the line that point P is on
def line_p (k : ℚ) (x y : ℝ) : Prop := k * x - y + 2 * k + 5 = 0

-- Define point P
def point_p : ℝ × ℝ := (-2, 5)

-- Define the distance from point P to line m
def distance_p_to_m : ℝ := 3

-- Theorem for the equation of line l
theorem equation_of_line_l :
  ∃ (A B C : ℝ), A * point_p.1 + B * point_p.2 + C = 0 ∧
  B ≠ 0 ∧ -A/B = slope_l ∧
  ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ y = slope_l * x + (point_p.2 - slope_l * point_p.1) :=
sorry

-- Theorem for the equations of line m
theorem equations_of_line_m :
  ∃ (b₁ b₂ : ℝ), 
    (∀ (x y : ℝ), y = slope_l * x + b₁ ↔ 
      distance_p_to_m = |slope_l * point_p.1 - point_p.2 + b₁| / Real.sqrt (slope_l^2 + 1)) ∧
    (∀ (x y : ℝ), y = slope_l * x + b₂ ↔ 
      distance_p_to_m = |slope_l * point_p.1 - point_p.2 + b₂| / Real.sqrt (slope_l^2 + 1)) ∧
    b₁ ≠ b₂ :=
sorry

end NUMINAMATH_CALUDE_equation_of_line_l_equations_of_line_m_l1627_162702


namespace NUMINAMATH_CALUDE_max_product_bound_l1627_162795

/-- A three-digit number without zeros -/
structure ThreeDigitNoZero where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNoZero) : ℕ :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of reciprocals of digits -/
def sum_reciprocals (n : ThreeDigitNoZero) : ℚ :=
  1 / n.a + 1 / n.b + 1 / n.c

/-- The product of the number and the sum of reciprocals of its digits -/
def product (n : ThreeDigitNoZero) : ℚ :=
  (value n : ℚ) * sum_reciprocals n

theorem max_product_bound :
  ∀ n : ThreeDigitNoZero, product n ≤ 1923.222 := by
  sorry

end NUMINAMATH_CALUDE_max_product_bound_l1627_162795


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1627_162707

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove the properties of the triangle given specific conditions. -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  B = π / 3 →
  3 * b * Real.sin A = 2 * c * Real.sin B →
  c = 3 ∧
  b = Real.sqrt 7 ∧
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1627_162707


namespace NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l1627_162776

theorem smallest_positive_solution_tan_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 ∧ Real.tan (3 * y) - Real.tan (2 * y) = 1 / Real.cos (2 * y) → x ≤ y) ∧
  Real.tan (3 * x) - Real.tan (2 * x) = 1 / Real.cos (2 * x) ∧
  x = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l1627_162776


namespace NUMINAMATH_CALUDE_five_three_number_properties_l1627_162740

/-- Definition of a "five-three number" -/
def is_five_three_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ d ≥ 0 ∧ d ≤ 9 ∧
    a = c + 5 ∧ b = d + 3

/-- Definition of M(A) -/
def M (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + c + 2 * (b + d)

/-- Definition of N(A) -/
def N (n : ℕ) : ℤ :=
  (n / 100 % 10) - 3

theorem five_three_number_properties :
  (∃ (max min : ℕ),
    is_five_three_number max ∧
    is_five_three_number min ∧
    (∀ n, is_five_three_number n → n ≤ max ∧ n ≥ min) ∧
    max - min = 4646) ∧
  (∃ A : ℕ,
    is_five_three_number A ∧
    (M A) % (N A) = 0 ∧
    A = 5401) :=
sorry

end NUMINAMATH_CALUDE_five_three_number_properties_l1627_162740


namespace NUMINAMATH_CALUDE_max_value_of_f_l1627_162742

open Real

noncomputable def f (x : ℝ) : ℝ := x / (exp x)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x ∈ Set.Icc 0 2, f x ≤ f c) ∧
  f c = 1 / exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1627_162742


namespace NUMINAMATH_CALUDE_min_cooking_time_is_15_l1627_162766

/-- Represents the duration of each cooking step in minutes -/
structure CookingSteps :=
  (washPot : ℕ)
  (washVegetables : ℕ)
  (prepareNoodles : ℕ)
  (boilWater : ℕ)
  (cookNoodles : ℕ)

/-- Calculates the minimum cooking time given the cooking steps -/
def minCookingTime (steps : CookingSteps) : ℕ :=
  max steps.boilWater (steps.washPot + steps.washVegetables + steps.prepareNoodles + steps.cookNoodles)

/-- Theorem stating that the minimum cooking time for the given steps is 15 minutes -/
theorem min_cooking_time_is_15 (steps : CookingSteps) 
  (h1 : steps.washPot = 2)
  (h2 : steps.washVegetables = 6)
  (h3 : steps.prepareNoodles = 2)
  (h4 : steps.boilWater = 10)
  (h5 : steps.cookNoodles = 3) :
  minCookingTime steps = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_cooking_time_is_15_l1627_162766


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1627_162790

theorem system_of_inequalities_solution (x : ℝ) :
  (x - 1 ≤ x / 2 ∧ x + 2 > 3 * (x - 2)) ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1627_162790


namespace NUMINAMATH_CALUDE_graph_decomposition_l1627_162733

/-- The graph of the equation (x^2 - 1)(x+y) = y^2(x+y) -/
def GraphEquation (x y : ℝ) : Prop :=
  (x^2 - 1) * (x + y) = y^2 * (x + y)

/-- The line y = -x -/
def Line (x y : ℝ) : Prop :=
  y = -x

/-- The hyperbola (x+y)(x-y) = 1 -/
def Hyperbola (x y : ℝ) : Prop :=
  (x + y) * (x - y) = 1

theorem graph_decomposition :
  ∀ x y : ℝ, GraphEquation x y ↔ (Line x y ∨ Hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_graph_decomposition_l1627_162733


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1627_162735

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1627_162735


namespace NUMINAMATH_CALUDE_savings_calculation_l1627_162796

-- Define the income and ratio
def income : ℕ := 15000
def income_ratio : ℕ := 15
def expenditure_ratio : ℕ := 8

-- Define the function to calculate savings
def calculate_savings (inc : ℕ) (inc_ratio : ℕ) (exp_ratio : ℕ) : ℕ :=
  inc - (inc * exp_ratio) / inc_ratio

-- Theorem to prove
theorem savings_calculation :
  calculate_savings income income_ratio expenditure_ratio = 7000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1627_162796


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l1627_162784

theorem sum_of_specific_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l1627_162784


namespace NUMINAMATH_CALUDE_scientific_notation_of_11930000_l1627_162719

/-- Proves that 11,930,000 is equal to 1.193 × 10^7 in scientific notation -/
theorem scientific_notation_of_11930000 : 
  11930000 = 1.193 * (10 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_11930000_l1627_162719


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l1627_162762

/-- Calculates the percentage of loss given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Proves that the loss percentage for a cycle with cost price 1400 and selling price 1190 is 15%. -/
theorem cycle_loss_percentage :
  let cost_price : ℚ := 1400
  let selling_price : ℚ := 1190
  loss_percentage cost_price selling_price = 15 := by
sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l1627_162762


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1627_162754

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 9

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

/-- The theorem stating that the given equation represents the circle with the specified properties -/
theorem circle_tangent_to_line :
  ∀ x y : ℝ,
  circle_equation x y ↔
  (∃ r : ℝ, r > 0 ∧
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2 ∧
    r = |3 * circle_center.1 - 4 * circle_center.2 + 5| / 5) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1627_162754


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l1627_162704

theorem ice_cream_distribution (nieces : ℚ) (total_sandwiches : ℕ) :
  nieces = 11 ∧ total_sandwiches = 1573 →
  (total_sandwiches : ℚ) / nieces = 143 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l1627_162704


namespace NUMINAMATH_CALUDE_roses_cut_l1627_162774

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) :
  final_roses - initial_roses = 11 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1627_162774


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1627_162712

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 3) = -15 + k * x) ↔ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1627_162712


namespace NUMINAMATH_CALUDE_tenth_digit_of_expression_l1627_162708

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def tenthDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenthDigit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_digit_of_expression_l1627_162708


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1627_162756

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (2*x - 1/(2*x))^6
  ∃ (a b c d e f g : ℝ), expansion = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + (-20) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1627_162756


namespace NUMINAMATH_CALUDE_octal_2016_to_binary_l1627_162783

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary --/
def decimal_to_binary (decimal : ℕ) : List ℕ := sorry

/-- Converts an octal number to binary --/
def octal_to_binary (octal : ℕ) : List ℕ :=
  decimal_to_binary (octal_to_decimal octal)

theorem octal_2016_to_binary :
  octal_to_binary 2016 = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_octal_2016_to_binary_l1627_162783


namespace NUMINAMATH_CALUDE_system_solution_l1627_162731

theorem system_solution : ∃! (x y : ℝ), 
  (x^2 * y + x * y^2 + 3*x + 3*y + 24 = 0) ∧ 
  (x^3 * y - x * y^3 + 3*x^2 - 3*y^2 - 48 = 0) ∧
  (x = -3) ∧ (y = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1627_162731


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_part1_simplify_and_evaluate_part2_l1627_162780

-- Part 1
theorem simplify_and_evaluate_part1 :
  ∀ a : ℝ, 3*a*(a^2 - 2*a + 1) - 2*a^2*(a - 3) = a^3 + 3*a ∧
  3*2*(2^2 - 2*2 + 1) - 2*2^2*(2 - 3) = 14 :=
sorry

-- Part 2
theorem simplify_and_evaluate_part2 :
  ∀ x : ℝ, (x - 4)*(x - 2) - (x - 1)*(x + 3) = -8*x + 11 ∧
  ((-5/2) - 4)*((-5/2) - 2) - ((-5/2) - 1)*((-5/2) + 3) = 31 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_part1_simplify_and_evaluate_part2_l1627_162780


namespace NUMINAMATH_CALUDE_line_through_midpoint_l1627_162745

-- Define the points and lines
def P : ℝ × ℝ := (1, 2)
def L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y + 2 = 0
def L2 : ℝ → ℝ → Prop := λ x y => x - 2 * y + 1 = 0

-- Define the property of A and B being on L1 and L2 respectively
def A_on_L1 (A : ℝ × ℝ) : Prop := L1 A.1 A.2
def B_on_L2 (B : ℝ × ℝ) : Prop := L2 B.1 B.2

-- Define the midpoint property
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- State the theorem
theorem line_through_midpoint :
  ∀ (A B : ℝ × ℝ),
    A_on_L1 A →
    B_on_L2 B →
    is_midpoint P A B →
    ∀ (x y : ℝ),
      (∃ (t : ℝ), x = P.1 + t * (A.1 - P.1) ∧ y = P.2 + t * (A.2 - P.2)) →
      line_equation x y :=
sorry

end NUMINAMATH_CALUDE_line_through_midpoint_l1627_162745


namespace NUMINAMATH_CALUDE_hundredth_term_is_9999_l1627_162711

/-- The nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ := n^2 - 1

/-- Theorem: The 100th term of the sequence is 9999 -/
theorem hundredth_term_is_9999 : sequenceTerm 100 = 9999 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_is_9999_l1627_162711


namespace NUMINAMATH_CALUDE_solution_set_eq_neg_reals_l1627_162750

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Axiom: f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Given conditions
axiom condition1 : ∀ x, f x + f' x < 1
axiom condition2 : f 0 = 2016

-- Define the solution set
def solution_set : Set ℝ := {x | Real.exp x * f x - Real.exp x > 2015}

-- Theorem statement
theorem solution_set_eq_neg_reals : solution_set f = Set.Iio 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_eq_neg_reals_l1627_162750


namespace NUMINAMATH_CALUDE_range_of_m_l1627_162713

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - m*x + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + m > 0

-- Define the main theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m) → 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1627_162713


namespace NUMINAMATH_CALUDE_pq_passes_through_centroid_l1627_162724

-- Define the points
variable (A B C D E F P Q : ℝ × ℝ)

-- Define the properties of the triangle and points
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def is_altitude_foot (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

def is_centroid (E A C D : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + C.1 + D.1) / 3 ∧ E.2 = (A.2 + C.2 + D.2) / 3

def is_perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def equal_distance (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

def line_passes_through_point (P Q X : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (X.1 - P.1) = (Q.1 - P.1) * (X.2 - P.2)

def centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

-- State the theorem
theorem pq_passes_through_centroid
  (h1 : is_right_triangle A B C)
  (h2 : is_altitude_foot A B C D)
  (h3 : is_centroid E A C D)
  (h4 : is_centroid F B C D)
  (h5 : is_perpendicular C E P)
  (h6 : equal_distance C P A)
  (h7 : is_perpendicular C F Q)
  (h8 : equal_distance C Q B) :
  ∃ G, centroid G A B C ∧ line_passes_through_point P Q G :=
sorry

end NUMINAMATH_CALUDE_pq_passes_through_centroid_l1627_162724


namespace NUMINAMATH_CALUDE_xyz_value_l1627_162734

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 280 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1627_162734


namespace NUMINAMATH_CALUDE_modular_congruence_l1627_162743

theorem modular_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (102 * n) % 103 = 74 % 103 → n % 103 = 29 % 103 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_l1627_162743


namespace NUMINAMATH_CALUDE_f_is_direct_proportion_l1627_162770

def f (x : ℝ) : ℝ := 3 * x

theorem f_is_direct_proportion : 
  (∀ x : ℝ, f x = 3 * x) ∧ 
  (f 0 = 0) ∧ 
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f x / x = f y / y) := by
  sorry

end NUMINAMATH_CALUDE_f_is_direct_proportion_l1627_162770


namespace NUMINAMATH_CALUDE_kelvins_classes_l1627_162752

theorem kelvins_classes (grant_vacations kelvin_classes : ℕ) 
  (h1 : grant_vacations = 4 * kelvin_classes)
  (h2 : grant_vacations + kelvin_classes = 450) :
  kelvin_classes = 90 := by
  sorry

end NUMINAMATH_CALUDE_kelvins_classes_l1627_162752


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1627_162773

theorem no_solution_for_sock_problem : ¬ ∃ (m n : ℕ), 
  m + n = 2009 ∧ 
  (m^2 - m + n^2 - n : ℚ) / (2009 * 2008) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1627_162773


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l1627_162746

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) :
  ∃ k : ℕ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l1627_162746


namespace NUMINAMATH_CALUDE_crayons_distribution_l1627_162755

theorem crayons_distribution (total_crayons : ℝ) (x : ℝ) : 
  total_crayons = 210 →
  x / total_crayons = 1 / 30 →
  30 * x = 0.7 * total_crayons →
  x = 4.9 := by
sorry

end NUMINAMATH_CALUDE_crayons_distribution_l1627_162755


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1627_162726

/-- The quadratic equation x^2 + 2x + m + 1 = 0 has two distinct real roots if and only if m < 0 -/
theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m + 1 = 0 ∧ y^2 + 2*y + m + 1 = 0) ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1627_162726


namespace NUMINAMATH_CALUDE_open_cells_are_perfect_squares_l1627_162758

/-- Represents whether a cell is open (true) or closed (false) -/
def CellState := Bool

/-- The state of a cell after the jailer's procedure -/
def final_cell_state (n : ℕ) : CellState :=
  sorry

/-- A number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- The main theorem: a cell remains open iff its number is a perfect square -/
theorem open_cells_are_perfect_squares (n : ℕ) :
  final_cell_state n = true ↔ is_perfect_square n :=
  sorry

end NUMINAMATH_CALUDE_open_cells_are_perfect_squares_l1627_162758


namespace NUMINAMATH_CALUDE_complete_square_equation_l1627_162769

theorem complete_square_equation : ∃ (a b c : ℤ), 
  (a > 0) ∧ 
  (∀ x : ℝ, 64 * x^2 + 80 * x - 81 = 0 ↔ (a * x + b)^2 = c) ∧
  (a = 8 ∧ b = 5 ∧ c = 106) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equation_l1627_162769


namespace NUMINAMATH_CALUDE_cube_root_of_110592_l1627_162727

theorem cube_root_of_110592 :
  ∃! (x : ℕ), x^3 = 110592 ∧ x > 0 :=
by
  use 48
  constructor
  · simp
  · intro y hy
    sorry

#eval 48^3  -- This will output 110592

end NUMINAMATH_CALUDE_cube_root_of_110592_l1627_162727
