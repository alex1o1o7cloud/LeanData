import Mathlib

namespace NUMINAMATH_CALUDE_parabola_properties_l1841_184115

theorem parabola_properties (a b c m : ℝ) (ha : a ≠ 0) (hm : -2 < m ∧ m < -1)
  (h_downward : a < 0)
  (h_root1 : a * 1^2 + b * 1 + c = 0)
  (h_root2 : a * m^2 + b * m + c = 0) :
  abc > 0 ∧ a - b + c > 0 ∧ a * (m + 1) - b + c > 0 := by
  sorry

#check parabola_properties

end NUMINAMATH_CALUDE_parabola_properties_l1841_184115


namespace NUMINAMATH_CALUDE_total_students_sum_l1841_184164

/-- The number of students in Varsity school -/
def varsity : ℕ := 1300

/-- The number of students in Northwest school -/
def northwest : ℕ := 1400

/-- The number of students in Central school -/
def central : ℕ := 1800

/-- The number of students in Greenbriar school -/
def greenbriar : ℕ := 1650

/-- The total number of students across all schools -/
def total_students : ℕ := varsity + northwest + central + greenbriar

theorem total_students_sum :
  total_students = 6150 := by sorry

end NUMINAMATH_CALUDE_total_students_sum_l1841_184164


namespace NUMINAMATH_CALUDE_unique_solution_symmetric_difference_l1841_184139

variable {U : Type*} -- Universe set

def symmetric_difference (A B : Set U) : Set U := (A \ B) ∪ (B \ A)

theorem unique_solution_symmetric_difference
  (A B X : Set U)
  (h1 : X ∩ (A ∪ B) = X)
  (h2 : A ∩ (B ∪ X) = A)
  (h3 : B ∩ (A ∪ X) = B)
  (h4 : X ∩ A ∩ B = ∅) :
  X = symmetric_difference A B ∧ 
  (∀ Y : Set U, Y ∩ (A ∪ B) = Y → A ∩ (B ∪ Y) = A → B ∩ (A ∪ Y) = B → Y ∩ A ∩ B = ∅ → Y = X) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_symmetric_difference_l1841_184139


namespace NUMINAMATH_CALUDE_triangle_side_a_value_l1841_184159

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_side_a_value (A B C : Real) (a b c : Real) : 
  -- Given conditions
  (Real.tan A = 2 * Real.tan B) →
  (b = Real.sqrt 2) →
  -- Assuming the area is at its maximum (we can't directly express this in Lean without additional setup)
  -- Conclusion
  (a = Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_a_value_l1841_184159


namespace NUMINAMATH_CALUDE_num_regions_convex_ngon_l1841_184183

/-- A convex n-gon is a polygon with n sides where all interior angles are less than 180 degrees. -/
structure ConvexNGon (n : ℕ) where
  -- Add necessary fields here
  n_ge_3 : n ≥ 3

/-- The number of regions formed by the diagonals of a convex n-gon where no three diagonals intersect at a single interior point. -/
def num_regions (n : ℕ) : ℕ := Nat.choose n 4 + Nat.choose (n - 1) 2

/-- Theorem stating that the number of regions formed by the diagonals of a convex n-gon
    where no three diagonals intersect at a single interior point is C_n^4 + C_{n-1}^2. -/
theorem num_regions_convex_ngon (n : ℕ) (polygon : ConvexNGon n) :
  num_regions n = Nat.choose n 4 + Nat.choose (n - 1) 2 := by
  sorry

#check num_regions_convex_ngon

end NUMINAMATH_CALUDE_num_regions_convex_ngon_l1841_184183


namespace NUMINAMATH_CALUDE_find_q_l1841_184168

theorem find_q (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 2/b)^2 - p*(a + 2/b) + q = 0) → 
  ((b + 2/a)^2 - p*(b + 2/a) + q = 0) → 
  q = 25/3 :=
by sorry

end NUMINAMATH_CALUDE_find_q_l1841_184168


namespace NUMINAMATH_CALUDE_min_slope_is_three_l1841_184104

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem stating that the minimum slope of tangents is 3
theorem min_slope_is_three :
  ∃ (x : ℝ), ∀ (y : ℝ), f' x ≤ f' y ∧ f' x = 3 :=
sorry

end NUMINAMATH_CALUDE_min_slope_is_three_l1841_184104


namespace NUMINAMATH_CALUDE_unique_c_value_l1841_184158

/-- A polynomial has exactly one real root if and only if its discriminant is zero -/
def has_one_real_root (b c : ℝ) : Prop :=
  b ^ 2 = 4 * c

/-- The product of all possible values of c satisfying the conditions -/
def product_of_c_values (b c : ℝ) : ℝ :=
  -- This is a placeholder; the actual computation would be more complex
  1

theorem unique_c_value (b c : ℝ) 
  (h1 : has_one_real_root b c)
  (h2 : b = c^2 + 1) :
  product_of_c_values b c = 1 := by
  sorry

#check unique_c_value

end NUMINAMATH_CALUDE_unique_c_value_l1841_184158


namespace NUMINAMATH_CALUDE_sum_of_digits_Q_is_six_l1841_184150

-- Define R_k as a function that takes k and returns the integer with k ones in base 10
def R (k : ℕ) : ℕ := (10^k - 1) / 9

-- Define Q as R_30 / R_5
def Q : ℕ := R 30 / R 5

-- Function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of Q is 6
theorem sum_of_digits_Q_is_six : sum_of_digits Q = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_Q_is_six_l1841_184150


namespace NUMINAMATH_CALUDE_square_is_quadratic_l1841_184191

/-- A quadratic function is of the form y = ax² + bx + c, where a, b, and c are constants, and a ≠ 0 -/
def IsQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic function -/
theorem square_is_quadratic : IsQuadraticFunction (λ x => x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_is_quadratic_l1841_184191


namespace NUMINAMATH_CALUDE_parabola_properties_l1841_184125

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

-- Theorem statement
theorem parabola_properties (a : ℝ) (h : a ≠ 0) :
  -- 1. Axis of symmetry
  (∀ x : ℝ, parabola a x = parabola a (2 - x)) ∧
  -- 2. Vertex on x-axis after shifting
  ((∃ x : ℝ, parabola a x - 3 * |a| = 0 ∧
             ∀ y : ℝ, parabola a y - 3 * |a| ≥ 0) ↔ (a = 3/4 ∨ a = -3/2)) ∧
  -- 3. Range of a for given points
  (∀ y₁ y₂ : ℝ, y₁ > y₂ → parabola a a = y₁ → parabola a 2 = y₂ → a > 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1841_184125


namespace NUMINAMATH_CALUDE_school_store_sale_l1841_184154

/-- The number of pencils sold in a school store sale -/
def pencils_sold (first_two : ℕ) (next_six : ℕ) (last_two : ℕ) : ℕ :=
  2 * first_two + 6 * next_six + 2 * last_two

/-- Theorem: Given the conditions of the pencil sale, 24 pencils were sold -/
theorem school_store_sale : pencils_sold 2 3 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_store_sale_l1841_184154


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l1841_184114

/-- Represents a modified cube with smaller cubes removed from corners -/
structure ModifiedCube where
  initialSideLength : ℕ
  removedCubeSideLength : ℕ
  numCornersRemoved : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  12 + 6 * cube.numCornersRemoved

/-- Theorem stating that a cube of side length 5 with 1x1 cubes removed from 4 corners has 36 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := {
    initialSideLength := 5,
    removedCubeSideLength := 1,
    numCornersRemoved := 4
  }
  edgeCount cube = 36 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l1841_184114


namespace NUMINAMATH_CALUDE_seventh_group_selection_l1841_184120

/-- Represents a systematic sampling method for a class of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  group_size : ℕ
  third_group_selection : ℕ

/-- Calculates the number drawn from a specific group in a systematic sampling method. -/
def number_drawn (s : SystematicSampling) (group : ℕ) : ℕ :=
  (group - 1) * s.group_size + (s.third_group_selection - ((3 - 1) * s.group_size))

/-- Theorem stating that if the number drawn from the third group is 13,
    then the number drawn from the seventh group is 33. -/
theorem seventh_group_selection
  (s : SystematicSampling)
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 10)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.third_group_selection = 13) :
  number_drawn s 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_seventh_group_selection_l1841_184120


namespace NUMINAMATH_CALUDE_angle_ratio_3_4_5_not_right_triangle_l1841_184152

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (sum_angles : A + B + C = Real.pi)
  (side_angle_correspondence : True)  -- This is a placeholder for the side-angle correspondence

/-- A right triangle is a triangle with one right angle (π/2) -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

/-- The condition that angle ratios are 3:4:5 -/
def angle_ratio_3_4_5 (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 3 * k ∧ t.B = 4 * k ∧ t.C = 5 * k

/-- Theorem: The condition ∠A:∠B:∠C = 3:4:5 cannot determine △ABC to be a right triangle -/
theorem angle_ratio_3_4_5_not_right_triangle :
  ∃ (t : Triangle), angle_ratio_3_4_5 t ∧ ¬(is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_3_4_5_not_right_triangle_l1841_184152


namespace NUMINAMATH_CALUDE_simplify_G_l1841_184197

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((2 * x + x^2) / (1 + 2 * x))

theorem simplify_G (x : ℝ) (h : x ≠ -1/2 ∧ x ≠ 1) : 
  G x = 2 * Real.log (1 + 2 * x) - F x :=
by sorry

end NUMINAMATH_CALUDE_simplify_G_l1841_184197


namespace NUMINAMATH_CALUDE_order_of_abc_l1841_184194

theorem order_of_abc (a b c : ℝ) (ha : a = 2^(1/10)) (hb : b = (1/2)^(4/5)) (hc : c = (1/2)^(1/2)) :
  a > c ∧ c > b :=
sorry

end NUMINAMATH_CALUDE_order_of_abc_l1841_184194


namespace NUMINAMATH_CALUDE_novosibirsk_divisible_by_three_l1841_184166

/-- Represents a mapping from letters to digits -/
def LetterToDigitMap := Char → Nat

/-- Checks if a mapping is valid for the word "NOVOSIBIRSK" -/
def isValidMapping (m : LetterToDigitMap) : Prop :=
  m 'N' ≠ m 'O' ∧ m 'N' ≠ m 'V' ∧ m 'N' ≠ m 'S' ∧ m 'N' ≠ m 'I' ∧ m 'N' ≠ m 'B' ∧ m 'N' ≠ m 'R' ∧ m 'N' ≠ m 'K' ∧
  m 'O' ≠ m 'V' ∧ m 'O' ≠ m 'S' ∧ m 'O' ≠ m 'I' ∧ m 'O' ≠ m 'B' ∧ m 'O' ≠ m 'R' ∧ m 'O' ≠ m 'K' ∧
  m 'V' ≠ m 'S' ∧ m 'V' ≠ m 'I' ∧ m 'V' ≠ m 'B' ∧ m 'V' ≠ m 'R' ∧ m 'V' ≠ m 'K' ∧
  m 'S' ≠ m 'I' ∧ m 'S' ≠ m 'B' ∧ m 'S' ≠ m 'R' ∧ m 'S' ≠ m 'K' ∧
  m 'I' ≠ m 'B' ∧ m 'I' ≠ m 'R' ∧ m 'I' ≠ m 'K' ∧
  m 'B' ≠ m 'R' ∧ m 'B' ≠ m 'K' ∧
  m 'R' ≠ m 'K'

/-- Calculates the sum of digits for "NOVOSIBIRSK" using the given mapping -/
def sumOfDigits (m : LetterToDigitMap) : Nat :=
  m 'N' + m 'O' + m 'V' + m 'O' + m 'S' + m 'I' + m 'B' + m 'I' + m 'R' + m 'S' + m 'K'

/-- Theorem: There exists a valid mapping for "NOVOSIBIRSK" that results in a number divisible by 3 -/
theorem novosibirsk_divisible_by_three : ∃ (m : LetterToDigitMap), isValidMapping m ∧ sumOfDigits m % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_novosibirsk_divisible_by_three_l1841_184166


namespace NUMINAMATH_CALUDE_square_division_rectangle_perimeter_l1841_184106

/-- Given a square with perimeter 120 units divided into four congruent rectangles,
    the perimeter of one of these rectangles is 90 units. -/
theorem square_division_rectangle_perimeter :
  ∀ (s : ℝ),
  s > 0 →
  4 * s = 120 →
  2 * (s + s / 2) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_square_division_rectangle_perimeter_l1841_184106


namespace NUMINAMATH_CALUDE_unique_k_solution_l1841_184100

def f (n : ℤ) : ℤ := 
  if n % 2 = 1 then n + 3 else n / 2

theorem unique_k_solution : 
  ∃! k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 27 ∧ k = 105 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l1841_184100


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1841_184196

/-- Represents the length of a wire segment between two marks -/
def segment_length : ℕ := 3

/-- Represents the total length of the wire -/
def wire_length : ℕ := 78

/-- Represents the total number of segments in the wire -/
def total_segments : ℕ := wire_length / segment_length

/-- Represents the perimeter of the rectangle in terms of segments -/
def perimeter_segments : ℕ := total_segments / 2

/-- Calculates the area of a rectangle given its length and width in segments -/
def rectangle_area (length width : ℕ) : ℕ :=
  (length * segment_length) * (width * segment_length)

/-- Theorem stating that the maximum area of the rectangle is 378 square centimeters -/
theorem max_rectangle_area :
  (∃ length width : ℕ,
    length + width = perimeter_segments ∧
    rectangle_area length width = 378 ∧
    ∀ l w : ℕ, l + w = perimeter_segments → rectangle_area l w ≤ 378) :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1841_184196


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l1841_184122

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel 
  (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l1841_184122


namespace NUMINAMATH_CALUDE_existence_of_six_snakes_l1841_184165

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A snake is a polyline with 5 segments connecting 6 points -/
structure Snake where
  points : Fin 6 → Point
  is_valid : Bool

/-- Check if two snakes are different -/
def are_different_snakes (s1 s2 : Snake) : Bool :=
  sorry

/-- Check if a snake satisfies the angle condition -/
def satisfies_angle_condition (s : Snake) : Bool :=
  sorry

/-- Check if a snake satisfies the half-plane condition -/
def satisfies_half_plane_condition (s : Snake) : Bool :=
  sorry

/-- The main theorem stating that a configuration of 6 points exists
    that can form 6 different valid snakes -/
theorem existence_of_six_snakes :
  ∃ (points : Fin 6 → Point),
    ∃ (snakes : Fin 6 → Snake),
      (∀ i : Fin 6, (snakes i).points = points) ∧
      (∀ i : Fin 6, (snakes i).is_valid) ∧
      (∀ i j : Fin 6, i ≠ j → are_different_snakes (snakes i) (snakes j)) ∧
      (∀ i : Fin 6, satisfies_angle_condition (snakes i)) ∧
      (∀ i : Fin 6, satisfies_half_plane_condition (snakes i)) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_six_snakes_l1841_184165


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocals_l1841_184161

theorem min_sum_of_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → (2 / a + 2 / b = 1) → a + b ≥ 8 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocals_l1841_184161


namespace NUMINAMATH_CALUDE_smallest_stairs_count_l1841_184156

theorem smallest_stairs_count : ∃ (n : ℕ), n > 15 ∧ n % 6 = 4 ∧ n % 7 = 3 ∧ ∀ (m : ℕ), m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_stairs_count_l1841_184156


namespace NUMINAMATH_CALUDE_percent_lost_is_twenty_l1841_184192

/-- Represents the number of games in each category -/
structure GameStats where
  won : ℕ
  lost : ℕ
  tied : ℕ

/-- Calculates the percentage of games lost -/
def percentLost (stats : GameStats) : ℚ :=
  stats.lost / (stats.won + stats.lost + stats.tied) * 100

/-- Theorem stating that for a team with a 7:3 win-to-loss ratio and 5 tied games,
    the percentage of games lost is 20% -/
theorem percent_lost_is_twenty {x : ℕ} (stats : GameStats)
    (h1 : stats.won = 7 * x)
    (h2 : stats.lost = 3 * x)
    (h3 : stats.tied = 5) :
  percentLost stats = 20 := by
  sorry

#eval percentLost ⟨7, 3, 5⟩

end NUMINAMATH_CALUDE_percent_lost_is_twenty_l1841_184192


namespace NUMINAMATH_CALUDE_equation_holds_for_all_y_l1841_184190

theorem equation_holds_for_all_y (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) ↔ x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_y_l1841_184190


namespace NUMINAMATH_CALUDE_monotone_special_function_characterization_l1841_184182

/-- A monotone function on real numbers satisfying f(x) + 2x = f(f(x)) -/
def MonotoneSpecialFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ ∀ x, f x + 2 * x = f (f x)

/-- The theorem stating that a MonotoneSpecialFunction must be either f(x) = -x or f(x) = 2x -/
theorem monotone_special_function_characterization (f : ℝ → ℝ) 
  (hf : MonotoneSpecialFunction f) : 
  (∀ x, f x = -x) ∨ (∀ x, f x = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_monotone_special_function_characterization_l1841_184182


namespace NUMINAMATH_CALUDE_product_greater_than_sum_plus_two_l1841_184180

theorem product_greater_than_sum_plus_two 
  (a b c : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hab : a * b > a + b) 
  (hbc : b * c > b + c) 
  (hac : a * c > a + c) : 
  a * b * c > a + b + c + 2 := by
sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_plus_two_l1841_184180


namespace NUMINAMATH_CALUDE_z_axis_symmetry_of_M_l1841_184134

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The z-axis symmetry operation on a 3D point -/
def zAxisSymmetry (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

/-- The original point M -/
def M : Point3D :=
  { x := 3, y := -4, z := 5 }

/-- The expected symmetric point -/
def SymmetricPoint : Point3D :=
  { x := -3, y := 4, z := 5 }

theorem z_axis_symmetry_of_M :
  zAxisSymmetry M = SymmetricPoint := by sorry

end NUMINAMATH_CALUDE_z_axis_symmetry_of_M_l1841_184134


namespace NUMINAMATH_CALUDE_residue_negative_1234_mod_32_l1841_184155

theorem residue_negative_1234_mod_32 : Int.mod (-1234) 32 = 14 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_1234_mod_32_l1841_184155


namespace NUMINAMATH_CALUDE_solve_salt_merchant_problem_l1841_184144

def salt_merchant_problem (initial_purchase : ℝ) (profit1 : ℝ) (profit2 : ℝ) : Prop :=
  let revenue1 := initial_purchase + profit1
  let profit_rate := profit2 / revenue1
  profit_rate * initial_purchase = profit1 ∧ profit1 = 100 ∧ profit2 = 120

theorem solve_salt_merchant_problem :
  ∃ (initial_purchase : ℝ), salt_merchant_problem initial_purchase 100 120 ∧ initial_purchase = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_salt_merchant_problem_l1841_184144


namespace NUMINAMATH_CALUDE_laura_weekly_mileage_l1841_184160

-- Define the distances
def school_round_trip : ℕ := 20
def supermarket_extra_distance : ℕ := 10

-- Define the number of trips
def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly mileage
def total_weekly_mileage : ℕ :=
  (school_round_trip * school_trips_per_week) +
  ((school_round_trip / 2 + supermarket_extra_distance) * 2 * supermarket_trips_per_week)

-- Theorem to prove
theorem laura_weekly_mileage :
  total_weekly_mileage = 180 := by
  sorry

end NUMINAMATH_CALUDE_laura_weekly_mileage_l1841_184160


namespace NUMINAMATH_CALUDE_trigonometric_inequality_and_supremum_l1841_184176

theorem trigonometric_inequality_and_supremum 
  (x y z : ℝ) (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (Real.sin x)^m * (Real.cos y)^n + 
  (Real.sin y)^m * (Real.cos z)^n + 
  (Real.sin z)^m * (Real.cos x)^n ≤ 1 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), 
    (Real.sin x₀)^m * (Real.cos y₀)^n + 
    (Real.sin y₀)^m * (Real.cos z₀)^n + 
    (Real.sin z₀)^m * (Real.cos x₀)^n = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_and_supremum_l1841_184176


namespace NUMINAMATH_CALUDE_angle4_value_l1841_184109

-- Define the angles
def angle1 : ℝ := 50
def angle2 : ℝ := 110
def angle3 : ℝ := 35
def angle4 : ℝ := 35
def angle5 : ℝ := 60
def angle6 : ℝ := 70

-- State the theorem
theorem angle4_value :
  angle1 + angle2 = 180 ∧
  angle3 = angle4 ∧
  angle1 = 50 ∧
  angle5 = 60 ∧
  angle1 + angle5 + angle6 = 180 ∧
  angle2 + angle6 = 180 ∧
  angle3 + angle4 = 180 - angle2 →
  angle4 = 35 := by sorry

end NUMINAMATH_CALUDE_angle4_value_l1841_184109


namespace NUMINAMATH_CALUDE_intersection_range_l1841_184145

-- Define the endpoints of the line segment
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (2, 3)

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

-- Define the condition for intersection
def intersects (k : ℝ) : Prop :=
  ∃ x y, x ≥ min A.1 B.1 ∧ x ≤ max A.1 B.1 ∧
         y ≥ min A.2 B.2 ∧ y ≤ max A.2 B.2 ∧
         y = line_equation k x

-- Theorem statement
theorem intersection_range :
  {k : ℝ | intersects k} = {k : ℝ | 1 ≤ k ∧ k ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1841_184145


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l1841_184178

theorem least_addition_for_divisibility (n m : ℕ) : 
  ∃ x : ℕ, x ≤ m - 1 ∧ (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ x : ℕ, x ≤ 22 ∧ (1054 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1054 + y) % 23 ≠ 0 ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l1841_184178


namespace NUMINAMATH_CALUDE_intersection_A_B_l1841_184138

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_A_B : A ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1841_184138


namespace NUMINAMATH_CALUDE_range_of_a_l1841_184135

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * (a - 1) * x - 4

-- Define the solution set of the inequality
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≥ 0}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, solution_set a = ∅) ↔ (∀ a : ℝ, -3 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1841_184135


namespace NUMINAMATH_CALUDE_rational_number_ordering_l1841_184140

theorem rational_number_ordering : -3^2 < -(1/3) ∧ -(1/3) < (-3)^2 ∧ (-3)^2 = |-3^2| := by
  sorry

end NUMINAMATH_CALUDE_rational_number_ordering_l1841_184140


namespace NUMINAMATH_CALUDE_pizza_slice_angle_l1841_184186

theorem pizza_slice_angle (p : ℝ) (h1 : p > 0) (h2 : p < 1) (h3 : p = 1/8) :
  let angle := p * 360
  angle = 45 := by sorry

end NUMINAMATH_CALUDE_pizza_slice_angle_l1841_184186


namespace NUMINAMATH_CALUDE_initial_capacity_proof_l1841_184195

/-- The daily processing capacity of each machine before modernization. -/
def initial_capacity : ℕ := 1215

/-- The number of machines before modernization. -/
def initial_machines : ℕ := 32

/-- The daily processing capacity of each machine after modernization. -/
def new_capacity : ℕ := 1280

/-- The number of machines after modernization. -/
def new_machines : ℕ := initial_machines + 3

/-- The total daily processing before modernization. -/
def total_before : ℕ := 38880

/-- The total daily processing after modernization. -/
def total_after : ℕ := 44800

theorem initial_capacity_proof :
  initial_capacity * initial_machines = total_before ∧
  new_capacity * new_machines = total_after ∧
  initial_capacity < new_capacity :=
by sorry

end NUMINAMATH_CALUDE_initial_capacity_proof_l1841_184195


namespace NUMINAMATH_CALUDE_gcd_of_360_and_504_l1841_184143

theorem gcd_of_360_and_504 : Nat.gcd 360 504 = 72 := by sorry

end NUMINAMATH_CALUDE_gcd_of_360_and_504_l1841_184143


namespace NUMINAMATH_CALUDE_unique_arrangement_l1841_184151

/-- Represents the three types of people in the problem -/
inductive PersonType
  | TruthTeller
  | Liar
  | Diplomat

/-- Represents the three positions -/
inductive Position
  | Left
  | Middle
  | Right

/-- A person's statement about another person's type -/
structure Statement where
  speaker : Position
  subject : Position
  claimedType : PersonType

/-- The arrangement of people -/
structure Arrangement where
  left : PersonType
  middle : PersonType
  right : PersonType

def isConsistent (arr : Arrangement) (statements : List Statement) : Prop :=
  ∀ s ∈ statements,
    (s.speaker = Position.Left ∧ arr.left = PersonType.TruthTeller) ∨
    (s.speaker = Position.Left ∧ arr.left = PersonType.Diplomat) ∨
    (s.speaker = Position.Middle ∧ arr.middle = PersonType.Liar) ∨
    (s.speaker = Position.Right ∧ arr.right = PersonType.TruthTeller) →
      ((s.subject = Position.Middle ∧ s.claimedType = arr.middle) ∨
       (s.subject = Position.Right ∧ s.claimedType = arr.right))

def problemStatements : List Statement :=
  [ ⟨Position.Left, Position.Middle, PersonType.TruthTeller⟩,
    ⟨Position.Middle, Position.Middle, PersonType.Diplomat⟩,
    ⟨Position.Right, Position.Middle, PersonType.Liar⟩ ]

theorem unique_arrangement :
  ∃! arr : Arrangement,
    arr.left = PersonType.Diplomat ∧
    arr.middle = PersonType.Liar ∧
    arr.right = PersonType.TruthTeller ∧
    isConsistent arr problemStatements :=
  sorry

end NUMINAMATH_CALUDE_unique_arrangement_l1841_184151


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1841_184153

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (50 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 10 * q * Real.sqrt (15 * q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1841_184153


namespace NUMINAMATH_CALUDE_baker_cake_difference_l1841_184149

/-- Given the initial number of cakes, number of cakes sold, and number of cakes bought,
    prove that the difference between cakes bought and sold is 63. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
  (h1 : initial = 13)
  (h2 : sold = 91)
  (h3 : bought = 154) :
  bought - sold = 63 := by
  sorry

end NUMINAMATH_CALUDE_baker_cake_difference_l1841_184149


namespace NUMINAMATH_CALUDE_second_derivative_zero_not_implies_extreme_point_l1841_184121

open Real

-- Define the function f(x) = x^3
def f (x : ℝ) := x^3

-- Define what it means for a point to be an extreme point
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ x, |x - x₀| < 1 → f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem second_derivative_zero_not_implies_extreme_point :
  ∃ x₀ : ℝ, (deriv (deriv f)) x₀ = 0 ∧ ¬(is_extreme_point f x₀) := by
  sorry


end NUMINAMATH_CALUDE_second_derivative_zero_not_implies_extreme_point_l1841_184121


namespace NUMINAMATH_CALUDE_combined_tower_height_l1841_184124

/-- The height of Grace's tower in inches -/
def grace_height : ℕ := 40

/-- The ratio of Grace's tower height to Clyde's tower height -/
def grace_to_clyde_ratio : ℕ := 8

/-- The ratio of Sarah's tower height to Clyde's tower height -/
def sarah_to_clyde_ratio : ℕ := 2

/-- Theorem stating the combined height of all three towers -/
theorem combined_tower_height : 
  grace_height + (grace_height / grace_to_clyde_ratio) * (1 + sarah_to_clyde_ratio) = 55 := by
  sorry

end NUMINAMATH_CALUDE_combined_tower_height_l1841_184124


namespace NUMINAMATH_CALUDE_red_blood_cell_surface_area_calculation_l1841_184147

/-- The sum of the surface areas of all red blood cells in a normal adult body. -/
def red_blood_cell_surface_area (body_surface_area : ℝ) : ℝ :=
  2000 * body_surface_area

/-- Theorem: The sum of the surface areas of all red blood cells in an adult body
    with a body surface area of 1800 cm² is 3.6 × 10⁶ cm². -/
theorem red_blood_cell_surface_area_calculation :
  red_blood_cell_surface_area 1800 = 3.6 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_red_blood_cell_surface_area_calculation_l1841_184147


namespace NUMINAMATH_CALUDE_polynomial_roots_condition_l1841_184128

open Real

/-- The polynomial in question -/
def polynomial (q x : ℝ) : ℝ := x^4 + 2*q*x^3 + 3*x^2 + 2*q*x + 2

/-- Predicate for a number being a root of the polynomial -/
def is_root (q x : ℝ) : Prop := polynomial q x = 0

/-- Theorem stating the condition for the polynomial to have at least two distinct negative real roots with product 2 -/
theorem polynomial_roots_condition (q : ℝ) : 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x * y = 2 ∧ is_root q x ∧ is_root q y) ↔ q < -7 * sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_condition_l1841_184128


namespace NUMINAMATH_CALUDE_wang_elevator_journey_l1841_184132

def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.2

theorem wang_elevator_journey :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map Int.natAbs floor_movements) * floor_height * electricity_per_meter = 33.6) := by
  sorry

end NUMINAMATH_CALUDE_wang_elevator_journey_l1841_184132


namespace NUMINAMATH_CALUDE_problem_solution_l1841_184185

theorem problem_solution (x : ℂ) (h : x + 1/x = -1) : x^1994 + 1/x^1994 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1841_184185


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1841_184193

theorem polynomial_division_theorem (x : ℝ) :
  let dividend := 12 * x^3 + 20 * x^2 - 7 * x + 4
  let divisor := 3 * x + 4
  let quotient := 4 * x^2 + (4/3) * x - 37/9
  let remainder := 74/9
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1841_184193


namespace NUMINAMATH_CALUDE_least_number_divisor_l1841_184170

theorem least_number_divisor (n : ℕ) (h1 : n % 5 = 3) (h2 : n % 67 = 3) (h3 : n % 8 = 3)
  (h4 : ∀ m : ℕ, m < n → (m % 5 = 3 ∧ m % 67 = 3 ∧ m % 8 = 3) → False)
  (h5 : n = 1683) :
  3 = Nat.gcd n (n - 3) :=
sorry

end NUMINAMATH_CALUDE_least_number_divisor_l1841_184170


namespace NUMINAMATH_CALUDE_max_elevation_is_550_l1841_184111

/-- The elevation function for a vertically projected particle -/
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

/-- The time at which the maximum elevation occurs -/
def max_time : ℝ := 5

theorem max_elevation_is_550 :
  ∃ (t : ℝ), ∀ (t' : ℝ), elevation t ≥ elevation t' ∧ elevation t = 550 :=
sorry

end NUMINAMATH_CALUDE_max_elevation_is_550_l1841_184111


namespace NUMINAMATH_CALUDE_exists_integers_for_n_squared_and_cubed_l1841_184129

theorem exists_integers_for_n_squared_and_cubed (n : ℕ) : 
  (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
sorry

end NUMINAMATH_CALUDE_exists_integers_for_n_squared_and_cubed_l1841_184129


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1841_184179

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^3 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1841_184179


namespace NUMINAMATH_CALUDE_annual_pension_formula_l1841_184189

/-- Represents an employee's pension calculation -/
structure PensionCalculation where
  x : ℝ  -- Years of service
  c : ℝ  -- Additional years scenario 1
  d : ℝ  -- Additional years scenario 2
  r : ℝ  -- Pension increase for scenario 1
  s : ℝ  -- Pension increase for scenario 2
  h1 : c ≠ d  -- Assumption that c and d are different

/-- The pension is proportional to years of service squared -/
def pension_proportional (p : PensionCalculation) (k : ℝ) : Prop :=
  ∃ (base_pension : ℝ), base_pension = k * p.x^2

/-- The pension increase after c more years of service -/
def pension_increase_c (p : PensionCalculation) (k : ℝ) : Prop :=
  k * (p.x + p.c)^2 - k * p.x^2 = p.r

/-- The pension increase after d more years of service -/
def pension_increase_d (p : PensionCalculation) (k : ℝ) : Prop :=
  k * (p.x + p.d)^2 - k * p.x^2 = p.s

/-- The theorem stating the formula for the annual pension -/
theorem annual_pension_formula (p : PensionCalculation) :
  ∃ (k : ℝ), 
    pension_proportional p k ∧ 
    pension_increase_c p k ∧ 
    pension_increase_d p k → 
    k = (p.s - p.r) / (2 * p.x * (p.d - p.c) + p.d^2 - p.c^2) :=
by sorry

end NUMINAMATH_CALUDE_annual_pension_formula_l1841_184189


namespace NUMINAMATH_CALUDE_only_C_is_perfect_square_l1841_184199

-- Define the expressions
def expr_A : ℕ := 3^3 * 4^5 * 7^7
def expr_B : ℕ := 3^4 * 4^4 * 7^5
def expr_C : ℕ := 3^6 * 4^3 * 7^6
def expr_D : ℕ := 3^5 * 4^6 * 7^4
def expr_E : ℕ := 3^4 * 4^6 * 7^6

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

-- Theorem statement
theorem only_C_is_perfect_square :
  is_perfect_square expr_C ∧
  ¬is_perfect_square expr_A ∧
  ¬is_perfect_square expr_B ∧
  ¬is_perfect_square expr_D ∧
  ¬is_perfect_square expr_E :=
sorry

end NUMINAMATH_CALUDE_only_C_is_perfect_square_l1841_184199


namespace NUMINAMATH_CALUDE_michael_purchase_l1841_184141

/-- The amount Michael paid for his purchases after a discount -/
def amountPaid (suitCost shoesCost discount : ℕ) : ℕ :=
  suitCost + shoesCost - discount

/-- Theorem stating the correct amount Michael paid -/
theorem michael_purchase : amountPaid 430 190 100 = 520 := by
  sorry

end NUMINAMATH_CALUDE_michael_purchase_l1841_184141


namespace NUMINAMATH_CALUDE_marble_difference_l1841_184177

theorem marble_difference (pink orange purple : ℕ) : 
  pink = 13 →
  orange < pink →
  purple = 4 * orange →
  pink + orange + purple = 33 →
  pink - orange = 9 := by
sorry

end NUMINAMATH_CALUDE_marble_difference_l1841_184177


namespace NUMINAMATH_CALUDE_john_remaining_cards_l1841_184148

def cards_per_deck : ℕ := 52
def half_full_decks : ℕ := 3
def full_decks : ℕ := 3
def discarded_cards : ℕ := 34

theorem john_remaining_cards : 
  cards_per_deck * full_decks + (cards_per_deck / 2) * half_full_decks - discarded_cards = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_remaining_cards_l1841_184148


namespace NUMINAMATH_CALUDE_bee_count_l1841_184101

theorem bee_count (flowers : ℕ) (bees : ℕ) : 
  flowers = 5 → bees = flowers - 2 → bees = 3 := by sorry

end NUMINAMATH_CALUDE_bee_count_l1841_184101


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1841_184172

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1841_184172


namespace NUMINAMATH_CALUDE_nuts_left_l1841_184181

theorem nuts_left (total : ℕ) (eaten_fraction : ℚ) (left : ℕ) : 
  total = 30 → eaten_fraction = 5/6 → left = total - (eaten_fraction * total) → left = 5 := by
  sorry

end NUMINAMATH_CALUDE_nuts_left_l1841_184181


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1841_184102

theorem right_triangle_leg_length 
  (north_distance : ℝ) 
  (hypotenuse : ℝ) 
  (h1 : north_distance = 10)
  (h2 : hypotenuse = 14.142135623730951) : 
  ∃ west_distance : ℝ, 
    west_distance ^ 2 + north_distance ^ 2 = hypotenuse ^ 2 ∧ 
    west_distance = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1841_184102


namespace NUMINAMATH_CALUDE_mass_of_man_is_60kg_l1841_184116

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg under the given conditions. -/
theorem mass_of_man_is_60kg :
  mass_of_man 3 2 0.01 1000 = 60 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_60kg_l1841_184116


namespace NUMINAMATH_CALUDE_same_color_probability_l1841_184187

/-- The probability of drawing two balls of the same color from a bag with replacement -/
theorem same_color_probability (total : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h_total : total = blue + yellow)
  (h_blue : blue = 5)
  (h_yellow : yellow = 5) :
  (blue / total) * (blue / total) + (yellow / total) * (yellow / total) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1841_184187


namespace NUMINAMATH_CALUDE_area_of_intersection_l1841_184198

/-- Given two overlapping rectangles ABNF and CMKD, prove the area of their intersection MNFK --/
theorem area_of_intersection (BN KD : ℝ) (area_ABMK area_CDFN : ℝ) :
  BN = 8 →
  KD = 9 →
  area_ABMK = 25 →
  area_CDFN = 32 →
  ∃ (AB CD : ℝ),
    AB * BN - area_ABMK = CD * KD - area_CDFN ∧
    AB * BN - area_ABMK = 31 :=
by sorry

end NUMINAMATH_CALUDE_area_of_intersection_l1841_184198


namespace NUMINAMATH_CALUDE_max_sum_of_digits_is_24_l1841_184108

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour format digital watch display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, sumOfDigitsTime24 t ≤ maxSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_is_24_l1841_184108


namespace NUMINAMATH_CALUDE_yogurt_production_cost_l1841_184107

/-- The price of fruit per kilogram that satisfies the yogurt production constraints -/
def fruit_price : ℝ := 2

/-- The cost of milk per liter -/
def milk_cost : ℝ := 1.5

/-- The number of liters of milk needed for one batch of yogurt -/
def milk_per_batch : ℝ := 10

/-- The number of kilograms of fruit needed for one batch of yogurt -/
def fruit_per_batch : ℝ := 3

/-- The cost to produce three batches of yogurt -/
def cost_three_batches : ℝ := 63

theorem yogurt_production_cost :
  fruit_price * fruit_per_batch * 3 + milk_cost * milk_per_batch * 3 = cost_three_batches :=
sorry

end NUMINAMATH_CALUDE_yogurt_production_cost_l1841_184107


namespace NUMINAMATH_CALUDE_manuscript_solution_l1841_184113

/-- Represents the problem of determining the number of pages in a manuscript. -/
def ManuscriptProblem (copies : ℕ) (printCost : ℚ) (bindCost : ℚ) (totalCost : ℚ) : Prop :=
  ∃ (pages : ℕ),
    (copies : ℚ) * printCost * (pages : ℚ) + (copies : ℚ) * bindCost = totalCost ∧
    pages = 400

/-- The solution to the manuscript problem. -/
theorem manuscript_solution :
  ManuscriptProblem 10 (5/100) 5 250 := by
  sorry

#check manuscript_solution

end NUMINAMATH_CALUDE_manuscript_solution_l1841_184113


namespace NUMINAMATH_CALUDE_quadratic_roots_differ_by_two_l1841_184130

/-- For a quadratic equation ax^2 + bx + c = 0 where a ≠ 0, 
    if the roots of the equation differ by 2, then c = (b^2 / (4a)) - a -/
theorem quadratic_roots_differ_by_two (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x - y = 2 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  c = (b^2 / (4 * a)) - a := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_differ_by_two_l1841_184130


namespace NUMINAMATH_CALUDE_octagon_trapezoid_area_l1841_184162

/-- The area of a trapezoid formed by four consecutive vertices of a regular octagon --/
theorem octagon_trapezoid_area (side_length : ℝ) (h : side_length = 6) :
  let diagonal_ratio : ℝ := Real.sqrt (4 + 2 * Real.sqrt 2)
  let height : ℝ := side_length * diagonal_ratio * (Real.sqrt (2 - Real.sqrt 2) / 2)
  let area : ℝ := side_length * height
  area = 18 * Real.sqrt (16 - 4 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_octagon_trapezoid_area_l1841_184162


namespace NUMINAMATH_CALUDE_monogram_count_l1841_184126

def alphabet_size : ℕ := 26

theorem monogram_count : (alphabet_size.choose 2) = 325 := by sorry

end NUMINAMATH_CALUDE_monogram_count_l1841_184126


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_23_l1841_184163

theorem modular_inverse_of_3_mod_23 : ∃ x : ℕ, x ≤ 22 ∧ (3 * x) % 23 = 1 :=
by
  use 8
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_23_l1841_184163


namespace NUMINAMATH_CALUDE_trajectory_equation_of_midpoints_l1841_184146

/-- Given three real numbers forming an arithmetic sequence and equations of a line and parabola,
    prove the trajectory equation of the midpoints of the intercepted chords. -/
theorem trajectory_equation_of_midpoints
  (a b c : ℝ)
  (h_arithmetic : c = 2*b - a) -- arithmetic sequence condition
  (h_line : ∀ x y, b*x + a*y + c = 0 → (x : ℝ) = x ∧ (y : ℝ) = y) -- line equation
  (h_parabola : ∀ x y, y^2 = -1/2*x → (x : ℝ) = x ∧ (y : ℝ) = y) -- parabola equation
  : ∃ (x y : ℝ), x + 1 = -(2*y - 1)^2 ∧ y ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_of_midpoints_l1841_184146


namespace NUMINAMATH_CALUDE_tangent_segment_difference_l1841_184171

/-- A quadrilateral inscribed in a circle with an inscribed circle --/
structure CyclicTangentialQuadrilateral where
  -- Sides of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Condition: quadrilateral is inscribed in a circle
  is_cyclic : True
  -- Condition: quadrilateral has an inscribed circle
  has_incircle : True

/-- Theorem about the difference of segments on a side --/
theorem tangent_segment_difference
  (q : CyclicTangentialQuadrilateral)
  (h1 : q.a = 80)
  (h2 : q.b = 100)
  (h3 : q.c = 120)
  (h4 : q.d = 140)
  (x y : ℝ)
  (h5 : x + y = q.c)
  : |x - y| = 80 := by
  sorry

end NUMINAMATH_CALUDE_tangent_segment_difference_l1841_184171


namespace NUMINAMATH_CALUDE_cost_of_bread_and_drinks_l1841_184118

/-- The cost of buying bread and drinks -/
theorem cost_of_bread_and_drinks 
  (a b : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0) : 
  a + 2 * b = (1 : ℝ) * a + (2 : ℝ) * b := by sorry

end NUMINAMATH_CALUDE_cost_of_bread_and_drinks_l1841_184118


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1841_184174

/-- Given a cylinder with base area 4π and a lateral surface that unfolds into a square,
    prove that its lateral surface area is 16π. -/
theorem cylinder_lateral_surface_area (r h : ℝ) : 
  (π * r^2 = 4 * π) →  -- base area condition
  (2 * π * r = h) →    -- lateral surface unfolds into a square condition
  (2 * π * r * h = 16 * π) := by 
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1841_184174


namespace NUMINAMATH_CALUDE_complex_magnitude_calculation_l1841_184133

theorem complex_magnitude_calculation : 
  Complex.abs (6 - 3 * Complex.I) * Complex.abs (6 + 3 * Complex.I) - 2 * Complex.abs (5 - Complex.I) = 45 - 2 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_calculation_l1841_184133


namespace NUMINAMATH_CALUDE_expression_evaluation_l1841_184103

theorem expression_evaluation (a b c : ℝ) 
  (h : a / (45 - a) + b / (85 - b) + c / (75 - c) = 9) :
  9 / (45 - a) + 17 / (85 - b) + 15 / (75 - c) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1841_184103


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1841_184173

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 9th term is 38, the 10th term is 41. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (h : ArithmeticSequence a) 
    (h4 : a 4 = 23) (h9 : a 9 = 38) : a 10 = 41 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1841_184173


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_rhombus_area_when_m_neg_seven_l1841_184117

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Theorem 1: The equation always has real roots
theorem quadratic_has_real_roots (m : ℝ) :
  ∃ x : ℝ, quadratic_equation x m = 0 :=
sorry

-- Theorem 2: Area of rhombus when m = -7
theorem rhombus_area_when_m_neg_seven :
  let m : ℝ := -7
  let root1 : ℝ := (9 + Real.sqrt 25) / 4
  let root2 : ℝ := (9 - Real.sqrt 25) / 4
  quadratic_equation root1 m = 0 ∧
  quadratic_equation root2 m = 0 →
  (1 / 2) * root1 * root2 = 7 / 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_rhombus_area_when_m_neg_seven_l1841_184117


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_values_l1841_184184

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

-- State the theorem
theorem sum_of_four_consecutive_values (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period : has_period_two_property f) : 
  f 2008 + f 2009 + f 2010 + f 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_values_l1841_184184


namespace NUMINAMATH_CALUDE_download_calculation_l1841_184137

/-- Calculates the number of songs that can be downloaded given internet speed, song size, and time. -/
def songs_downloaded (internet_speed : ℕ) (song_size : ℕ) (time_minutes : ℕ) : ℕ :=
  (internet_speed * 60 * time_minutes) / song_size

/-- Theorem stating that with given conditions, 7200 songs can be downloaded. -/
theorem download_calculation :
  let internet_speed : ℕ := 20  -- MBps
  let song_size : ℕ := 5        -- MB
  let time_minutes : ℕ := 30    -- half an hour
  songs_downloaded internet_speed song_size time_minutes = 7200 := by
sorry

end NUMINAMATH_CALUDE_download_calculation_l1841_184137


namespace NUMINAMATH_CALUDE_similar_triangles_sequence_l1841_184119

/-- Given a sequence of six similar right triangles with vertex A, where AB = 24 and AC = 54,
    prove that the length of AD (hypotenuse of the third triangle) is 36. -/
theorem similar_triangles_sequence (a b x c d : ℝ) : 
  (24 : ℝ) / a = a / b ∧ 
  a / b = b / x ∧ 
  b / x = x / c ∧ 
  x / c = c / d ∧ 
  c / d = d / 54 → 
  x = 36 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_sequence_l1841_184119


namespace NUMINAMATH_CALUDE_triathlon_bicycle_speed_triathlon_solution_l1841_184136

theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_speed swim_distance : ℝ) 
  (run_speed run_distance : ℝ) 
  (bike_distance : ℝ) : ℝ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem triathlon_solution :
  triathlon_bicycle_speed 3 1 0.5 8 4 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bicycle_speed_triathlon_solution_l1841_184136


namespace NUMINAMATH_CALUDE_integer_solution_l1841_184142

theorem integer_solution (x : ℤ) : 
  x + 15 ≥ 16 ∧ -3*x ≥ -15 → x ∈ ({1, 2, 3, 4, 5} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_integer_solution_l1841_184142


namespace NUMINAMATH_CALUDE_circle_radius_condition_l1841_184188

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4*x + y^2 + 8*y + c = 0 ↔ (x + 2)^2 + (y + 4)^2 = 25) → 
  c = -5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l1841_184188


namespace NUMINAMATH_CALUDE_one_in_M_l1841_184157

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_one_in_M_l1841_184157


namespace NUMINAMATH_CALUDE_stock_percentage_l1841_184112

/-- Calculate the percentage of a stock given income, stock price, and total investment. -/
theorem stock_percentage (income : ℚ) (stock_price : ℚ) (total_investment : ℚ) :
  income = 450 →
  stock_price = 108 →
  total_investment = 4860 →
  (income / total_investment) * 100 = (450 : ℚ) / 4860 * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_l1841_184112


namespace NUMINAMATH_CALUDE_rollo_guinea_pigs_l1841_184169

/-- The amount of food eaten by the first guinea pig -/
def first_guinea_pig_food : ℕ := 2

/-- The amount of food eaten by the second guinea pig -/
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food

/-- The amount of food eaten by the third guinea pig -/
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

/-- The total amount of food needed to feed all guinea pigs -/
def total_food_needed : ℕ := 13

/-- The number of guinea pigs Rollo has -/
def number_of_guinea_pigs : ℕ := 3

theorem rollo_guinea_pigs :
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = total_food_needed ∧
  number_of_guinea_pigs = 3 := by
  sorry

end NUMINAMATH_CALUDE_rollo_guinea_pigs_l1841_184169


namespace NUMINAMATH_CALUDE_people_per_cubic_yard_l1841_184131

theorem people_per_cubic_yard (people_per_yard : ℕ) : 
  (9000 * people_per_yard - 6400 * people_per_yard = 208000) → 
  people_per_yard = 80 := by
sorry

end NUMINAMATH_CALUDE_people_per_cubic_yard_l1841_184131


namespace NUMINAMATH_CALUDE_first_number_is_55_l1841_184123

def problem (x : ℝ) : Prop :=
  let known_numbers : List ℝ := [48, 507, 2, 684, 42]
  let all_numbers : List ℝ := x :: known_numbers
  (List.sum all_numbers) / 6 = 223

theorem first_number_is_55 : 
  ∃ (x : ℝ), problem x ∧ x = 55 :=
sorry

end NUMINAMATH_CALUDE_first_number_is_55_l1841_184123


namespace NUMINAMATH_CALUDE_optimal_pool_dimensions_l1841_184105

/-- Represents the dimensions and cost of a rectangular pool -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (bottomCost : ℝ)
  (wallCost : ℝ)

/-- Calculates the total cost of the pool -/
def totalCost (p : Pool) : ℝ :=
  p.bottomCost * p.length * p.width + p.wallCost * 2 * (p.length + p.width) * p.depth

/-- Theorem stating the optimal dimensions and minimum cost of the pool -/
theorem optimal_pool_dimensions :
  ∀ p : Pool,
  p.depth = 2 ∧
  p.length * p.width * p.depth = 18 ∧
  p.bottomCost = 200 ∧
  p.wallCost = 150 →
  ∃ (minCost : ℝ),
    minCost = 7200 ∧
    totalCost p ≥ minCost ∧
    (totalCost p = minCost ↔ p.length = 3 ∧ p.width = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_pool_dimensions_l1841_184105


namespace NUMINAMATH_CALUDE_equation_solutions_l1841_184110

theorem equation_solutions : 
  (∃ (S₁ : Set ℝ), S₁ = {x : ℝ | x * (x + 2) = 2 * x + 4} ∧ S₁ = {-2, 2}) ∧
  (∃ (S₂ : Set ℝ), S₂ = {x : ℝ | 3 * x^2 - x - 2 = 0} ∧ S₂ = {1, -2/3}) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1841_184110


namespace NUMINAMATH_CALUDE_students_in_range_estimate_l1841_184175

/-- Represents a normal distribution of scores -/
structure ScoreDistribution where
  mean : ℝ
  stdDev : ℝ
  isNormal : Bool

/-- Represents the student population and their score distribution -/
structure StudentPopulation where
  totalStudents : ℕ
  scoreDistribution : ScoreDistribution

/-- Calculates the number of students within a given score range -/
def studentsInRange (pop : StudentPopulation) (lowerBound upperBound : ℝ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem students_in_range_estimate 
  (pop : StudentPopulation) 
  (h1 : pop.totalStudents = 3000) 
  (h2 : pop.scoreDistribution.isNormal = true) : 
  ∃ (ε : ℕ), ε ≤ 10 ∧ 
  (studentsInRange pop 70 80 = 408 + ε ∨ studentsInRange pop 70 80 = 408 - ε) :=
sorry

end NUMINAMATH_CALUDE_students_in_range_estimate_l1841_184175


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1841_184127

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m ≥ 120 ∧ (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) →
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1841_184127


namespace NUMINAMATH_CALUDE_enumeration_pattern_correct_l1841_184167

/-- Represents the number in a square of the enumerated grid -/
def square_number (m n : ℕ) : ℕ := Nat.choose (m + n - 1) 2 + n

/-- The enumeration pattern for the squared paper -/
def enumeration_pattern : ℕ → ℕ → ℕ := square_number

theorem enumeration_pattern_correct :
  ∀ (m n : ℕ), enumeration_pattern m n = square_number m n :=
by sorry

end NUMINAMATH_CALUDE_enumeration_pattern_correct_l1841_184167
