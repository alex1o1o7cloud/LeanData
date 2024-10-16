import Mathlib

namespace NUMINAMATH_CALUDE_no_810_triple_l2229_222969

/-- Converts a list of digits in base 8 to a natural number -/
def fromBase8 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a list of digits in base 10 to a natural number -/
def fromBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 10 * acc + d) 0

/-- Checks if a number is an 8-10 triple -/
def is810Triple (n : Nat) : Prop :=
  n > 0 ∧ ∃ digits : List Nat, 
    (∀ d ∈ digits, d < 8) ∧
    fromBase8 digits = n ∧
    fromBase10 digits = 3 * n

theorem no_810_triple : ¬∃ n : Nat, is810Triple n := by
  sorry

end NUMINAMATH_CALUDE_no_810_triple_l2229_222969


namespace NUMINAMATH_CALUDE_triangle_height_l2229_222933

theorem triangle_height (a b : ℝ) (α : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_angle : 0 < α ∧ α < π) :
  let c := Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos α))
  let h := (a * b * Real.sin α) / c
  0 < h ∧ h < a ∧ h < b ∧ h * c = a * b * Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_l2229_222933


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2229_222971

theorem arithmetic_sequence_property (n : ℕ+) : 
  (∀ S : Finset ℕ, S ⊆ Finset.range 1989 → S.card = n → 
    ∃ (a d : ℕ) (H : Finset ℕ), H ⊆ S ∧ H.card = 29 ∧ 
    ∀ k, k ∈ H → ∃ i, 0 ≤ i ∧ i < 29 ∧ k = a + i * d) → 
  n > 1788 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2229_222971


namespace NUMINAMATH_CALUDE_original_rate_l2229_222920

/-- Given a reduction of 'a' yuan followed by a 20% reduction resulting in a final rate of 'b' yuan per minute, 
    the original rate was a + 1.25b yuan per minute. -/
theorem original_rate (a b : ℝ) : 
  (∃ x : ℝ, 0.8 * (x - a) = b) → 
  (∃ x : ℝ, x = a + 1.25 * b ∧ 0.8 * (x - a) = b) :=
by sorry

end NUMINAMATH_CALUDE_original_rate_l2229_222920


namespace NUMINAMATH_CALUDE_law_of_sines_iff_equilateral_l2229_222999

/-- In a triangle ABC, the law of sines condition is equivalent to the triangle being equilateral -/
theorem law_of_sines_iff_equilateral (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin B = b / Real.sin C ∧ b / Real.sin C = c / Real.sin A) ↔
  (a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_law_of_sines_iff_equilateral_l2229_222999


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2229_222921

/-- Given that x and y are inversely proportional, x + y = 32, and x - y = 8, 
    prove that when x = 4, y = 60. -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k) 
  (h2 : x + y = 32) (h3 : x - y = 8) : x = 4 → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2229_222921


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2229_222972

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ x y : ℝ, y = m * x ∧ x^2 + y^2 - 4*x + 2 = 0 ∧
   ∀ x' y' : ℝ, y' = m * x' → x'^2 + y'^2 - 4*x' + 2 ≥ 0) →
  m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2229_222972


namespace NUMINAMATH_CALUDE_smallest_positive_angle_exists_l2229_222989

theorem smallest_positive_angle_exists : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ 
  (∀ φ : ℝ, φ > 0 ∧ φ < 360 ∧ 
    Real.cos (φ * Real.pi / 180) = 
      Real.sin (45 * Real.pi / 180) + 
      Real.cos (30 * Real.pi / 180) - 
      Real.sin (18 * Real.pi / 180) - 
      Real.cos (12 * Real.pi / 180) → 
    θ ≤ φ) ∧
  Real.cos (θ * Real.pi / 180) = 
    Real.sin (45 * Real.pi / 180) + 
    Real.cos (30 * Real.pi / 180) - 
    Real.sin (18 * Real.pi / 180) - 
    Real.cos (12 * Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_exists_l2229_222989


namespace NUMINAMATH_CALUDE_lives_difference_l2229_222924

/-- The number of lives of animals in a fictional world --/
def lives_problem (cat_lives dog_lives mouse_lives : ℕ) : Prop :=
  cat_lives = 9 ∧
  dog_lives = cat_lives - 3 ∧
  mouse_lives = 13 ∧
  mouse_lives - dog_lives = 7

theorem lives_difference :
  ∃ (cat_lives dog_lives mouse_lives : ℕ),
    lives_problem cat_lives dog_lives mouse_lives :=
by
  sorry

end NUMINAMATH_CALUDE_lives_difference_l2229_222924


namespace NUMINAMATH_CALUDE_sector_max_area_l2229_222977

/-- Given a sector with circumference 30, its area is maximized when the radius is 15/2 and the central angle is 2. -/
theorem sector_max_area (R α : ℝ) : 
  R + R + (α * R) = 30 →  -- circumference condition
  (∀ R' α' : ℝ, R' + R' + (α' * R') = 30 → 
    (1/2) * α * R^2 ≥ (1/2) * α' * R'^2) →  -- area is maximized
  R = 15/2 ∧ α = 2 := by
sorry


end NUMINAMATH_CALUDE_sector_max_area_l2229_222977


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2229_222900

-- Part 1
def positive_integer_solutions (x : ℕ) : Prop :=
  4 * (x + 2) < 18 + 2 * x

theorem solution_set_part1 :
  {x : ℕ | positive_integer_solutions x} = {1, 2, 3, 4} :=
sorry

-- Part 2
def inequality_system (x : ℝ) : Prop :=
  5 * x + 2 ≥ 4 * x + 1 ∧ (x + 1) / 4 > (x - 3) / 2 + 1

theorem solution_set_part2 :
  {x : ℝ | inequality_system x} = {x : ℝ | -1 ≤ x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2229_222900


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2229_222966

def is_divisible_by_one_of (F : ℤ → ℤ) (divisors : List ℤ) : Prop :=
  ∀ n : ℤ, ∃ a ∈ divisors, (F n) % a = 0

theorem polynomial_divisibility
  (F : ℤ → ℤ)
  (divisors : List ℤ)
  (h_polynomial : ∀ x y : ℤ, (F x - F y) % (x - y) = 0)
  (h_divisible : is_divisible_by_one_of F divisors) :
  ∃ a ∈ divisors, ∀ n : ℤ, (F n) % a = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2229_222966


namespace NUMINAMATH_CALUDE_base3_21021_equals_196_l2229_222988

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_21021_equals_196 :
  base3_to_base10 [2, 1, 0, 2, 1] = 196 := by
  sorry

end NUMINAMATH_CALUDE_base3_21021_equals_196_l2229_222988


namespace NUMINAMATH_CALUDE_lovely_class_size_l2229_222918

/-- Proves that the number of students in Mrs. Lovely's class is 29 given the jelly bean distribution conditions. -/
theorem lovely_class_size :
  ∀ (g : ℕ),
  let b := g + 3
  let total_jelly_beans := 420
  let remaining_jelly_beans := 18
  let distributed_jelly_beans := total_jelly_beans - remaining_jelly_beans
  g * g + b * b = distributed_jelly_beans →
  g + b = 29 := by
  sorry

end NUMINAMATH_CALUDE_lovely_class_size_l2229_222918


namespace NUMINAMATH_CALUDE_inequality_proof_l2229_222929

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1/3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2229_222929


namespace NUMINAMATH_CALUDE_work_fraction_left_l2229_222946

theorem work_fraction_left (p q : ℕ) (h1 : p = 15) (h2 : q = 20) : 
  1 - 4 * (1 / p.cast + 1 / q.cast) = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_left_l2229_222946


namespace NUMINAMATH_CALUDE_smallest_sum_is_84_l2229_222995

/-- Represents a rectangular prism made of dice -/
structure DicePrism where
  length : Nat
  width : Nat
  height : Nat
  total_dice : Nat
  dice_opposite_sum : Nat

/-- Calculates the smallest possible sum of visible values on the prism faces -/
def smallest_visible_sum (prism : DicePrism) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum for the given prism configuration -/
theorem smallest_sum_is_84 (prism : DicePrism) 
  (h1 : prism.length = 4)
  (h2 : prism.width = 3)
  (h3 : prism.height = 2)
  (h4 : prism.total_dice = 24)
  (h5 : prism.dice_opposite_sum = 7) :
  smallest_visible_sum prism = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_84_l2229_222995


namespace NUMINAMATH_CALUDE_product_of_sum_of_roots_l2229_222908

theorem product_of_sum_of_roots (x : ℝ) :
  (Real.sqrt (5 + x) + Real.sqrt (25 - x) = 8) →
  (5 + x) * (25 - x) = 289 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_roots_l2229_222908


namespace NUMINAMATH_CALUDE_work_completion_time_l2229_222991

/-- Given two people working together can complete a task in 10 days,
    and one person (Prakash) can complete the task in 30 days,
    prove that the other person can complete the task alone in 15 days. -/
theorem work_completion_time (prakash_time : ℕ) (joint_time : ℕ) (x : ℕ) :
  prakash_time = 30 →
  joint_time = 10 →
  (1 : ℚ) / x + (1 : ℚ) / prakash_time = (1 : ℚ) / joint_time →
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2229_222991


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2229_222980

/-- A dodecahedron is a 3-dimensional figure with 20 vertices -/
structure Dodecahedron where
  vertices : Finset ℕ
  vertex_count : vertices.card = 20

/-- Each vertex in a dodecahedron is connected to 3 other vertices by edges -/
def connected_vertices (d : Dodecahedron) (v : ℕ) : Finset ℕ :=
  sorry

axiom connected_vertices_count (d : Dodecahedron) (v : ℕ) (h : v ∈ d.vertices) :
  (connected_vertices d v).card = 3

/-- An interior diagonal is a segment connecting two vertices which do not share an edge -/
def interior_diagonals (d : Dodecahedron) : Finset (ℕ × ℕ) :=
  sorry

/-- The main theorem: a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  (interior_diagonals d).card = 160 :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2229_222980


namespace NUMINAMATH_CALUDE_tokyo_tech_1956_entrance_exam_l2229_222993

theorem tokyo_tech_1956_entrance_exam
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha1 : a < 1) (hb1 : b < 1) (hc1 : c < 1) :
  a + b + c - a * b * c < 2 :=
sorry

end NUMINAMATH_CALUDE_tokyo_tech_1956_entrance_exam_l2229_222993


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2229_222931

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (3 * x₁^2 - 4*(3*a-2)*x₁ + a^2 + 2*a = 0) ∧ 
    (3 * x₂^2 - 4*(3*a-2)*x₂ + a^2 + 2*a = 0) ∧ 
    (x₁ < a ∧ a < x₂)) 
  ↔ 
  (a < 0 ∨ a > 5/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2229_222931


namespace NUMINAMATH_CALUDE_four_values_with_2001_l2229_222911

/-- Represents a sequence where each term after the first two is defined by the previous two terms. -/
def SpecialSequence (x : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => 2000
  | (n + 2) => SpecialSequence x n * SpecialSequence x (n + 1) - 1

/-- The set of positive real numbers x such that 2001 appears in the special sequence starting with x. -/
def SequencesWith2001 : Set ℝ :=
  {x : ℝ | x > 0 ∧ ∃ n : ℕ, SpecialSequence x n = 2001}

theorem four_values_with_2001 :
  ∃ (S : Finset ℝ), S.card = 4 ∧ (∀ x ∈ SequencesWith2001, x ∈ S) ∧ (∀ x ∈ S, x ∈ SequencesWith2001) :=
sorry

end NUMINAMATH_CALUDE_four_values_with_2001_l2229_222911


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2229_222975

theorem complex_equation_solution (a b : ℝ) (z : ℂ) : 
  z = Complex.mk a b → z + Complex.I = (2 - Complex.I) / (1 + 2 * Complex.I) → b = -2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2229_222975


namespace NUMINAMATH_CALUDE_supermarket_spending_l2229_222907

theorem supermarket_spending (total : ℚ) : 
  (3/7 : ℚ) * total + (2/5 : ℚ) * total + (1/4 : ℚ) * total + (1/14 : ℚ) * total + 12 = total →
  total = 80 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2229_222907


namespace NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l2229_222957

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 1)^2
def parabola2 (x y : ℝ) : Prop := x - 6 = (y + 1)^2

-- Define the intersection points
def intersection_points : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem sum_of_intersection_coordinates :
  intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l2229_222957


namespace NUMINAMATH_CALUDE_milk_cisterns_l2229_222942

theorem milk_cisterns (x y z : ℝ) (h1 : x + y + z = 780) 
  (h2 : (3/4) * x = (4/5) * y) (h3 : (3/4) * x = (4/7) * z) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  x = 240 ∧ y = 225 ∧ z = 315 := by
  sorry

end NUMINAMATH_CALUDE_milk_cisterns_l2229_222942


namespace NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l2229_222916

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l2229_222916


namespace NUMINAMATH_CALUDE_favorite_fruit_strawberries_l2229_222992

theorem favorite_fruit_strawberries (total : ℕ) (oranges pears apples bananas grapes : ℕ)
  (h_total : total = 900)
  (h_oranges : oranges = 130)
  (h_pears : pears = 210)
  (h_apples : apples = 275)
  (h_bananas : bananas = 93)
  (h_grapes : grapes = 119) :
  total - (oranges + pears + apples + bananas + grapes) = 73 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_strawberries_l2229_222992


namespace NUMINAMATH_CALUDE_prime_product_sum_proper_fractions_l2229_222915

/-- Sum of proper fractions with denominator k -/
def sum_proper_fractions (k : ℕ) : ℚ :=
  (k - 1) / 2

theorem prime_product_sum_proper_fractions : 
  ∀ m n : ℕ, 
  m.Prime → n.Prime → m < n → 
  (sum_proper_fractions m) * (sum_proper_fractions n) = 5 → 
  m = 3 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_sum_proper_fractions_l2229_222915


namespace NUMINAMATH_CALUDE_rhombus_area_l2229_222986

/-- The area of a rhombus with vertices at (0, 3.5), (7, 0), (0, -3.5), and (-7, 0) is 49 square units. -/
theorem rhombus_area : 
  let v1 : ℝ × ℝ := (0, 3.5)
  let v2 : ℝ × ℝ := (7, 0)
  let v3 : ℝ × ℝ := (0, -3.5)
  let v4 : ℝ × ℝ := (-7, 0)
  let d1 : ℝ := v1.2 - v3.2  -- Vertical diagonal
  let d2 : ℝ := v2.1 - v4.1  -- Horizontal diagonal
  let area : ℝ := (d1 * d2) / 2
  area = 49 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2229_222986


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_sufficient_not_necessary_l2229_222985

/-- A hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 --/
structure Hyperbola (a b : ℝ) : Type :=
  (hap : a > 0)
  (hbp : b > 0)

/-- The asymptotes of a hyperbola --/
def asymptotes (h : Hyperbola a b) (x : ℝ) : Set ℝ :=
  {y | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The theorem stating that the hyperbola equation is a sufficient but not necessary condition for its asymptotes --/
theorem hyperbola_asymptotes_sufficient_not_necessary (a b : ℝ) :
  (∃ (h : Hyperbola a b), ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → y ∈ asymptotes h x) ∧
  (∃ a' b' : ℝ, ∃ (h : Hyperbola a' b'), ∀ x y : ℝ, y ∈ asymptotes h x ∧ (x^2 / a'^2) - (y^2 / b'^2) ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_sufficient_not_necessary_l2229_222985


namespace NUMINAMATH_CALUDE_common_point_implies_c_equals_d_l2229_222987

/-- Given three linear functions with a common point, prove that c = d -/
theorem common_point_implies_c_equals_d 
  (a b c d : ℝ) 
  (h_neq : a ≠ b) 
  (h_common : ∃ (x y : ℝ), 
    y = a * x + a ∧ 
    y = b * x + b ∧ 
    y = c * x + d) : 
  c = d := by
sorry


end NUMINAMATH_CALUDE_common_point_implies_c_equals_d_l2229_222987


namespace NUMINAMATH_CALUDE_mushroom_picking_theorem_l2229_222903

/-- Calculates the total number of mushrooms picked over a three-day trip --/
def total_mushrooms (day1_revenue : ℕ) (day2_picked : ℕ) (price_per_mushroom : ℕ) : ℕ :=
  let day1_picked := day1_revenue / price_per_mushroom
  let day3_picked := 2 * day2_picked
  day1_picked + day2_picked + day3_picked

/-- The total number of mushrooms picked over three days is 65 --/
theorem mushroom_picking_theorem :
  total_mushrooms 58 12 2 = 65 := by
  sorry

#eval total_mushrooms 58 12 2

end NUMINAMATH_CALUDE_mushroom_picking_theorem_l2229_222903


namespace NUMINAMATH_CALUDE_domain_of_g_l2229_222984

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the function g(x) = f(x²)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l2229_222984


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_and_p_and_q_true_l2229_222947

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 15 > 0}
def B : Set ℝ := {x | x - 6 < 0}

-- Define propositions p and q
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Theorem for the first part
theorem range_when_p_true :
  {m : ℝ | p m} = {x | x < -3 ∨ x > 5} :=
sorry

-- Theorem for the second part
theorem range_when_p_or_q_and_p_and_q_true :
  {m : ℝ | (p m ∨ q m) ∧ (p m ∧ q m)} = {x | x < -3} :=
sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_and_p_and_q_true_l2229_222947


namespace NUMINAMATH_CALUDE_x_value_equality_l2229_222976

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem x_value_equality : ∃ x : ℝ, F x 3 2 = F x 2 5 ∧ x = 21/19 := by
  sorry

end NUMINAMATH_CALUDE_x_value_equality_l2229_222976


namespace NUMINAMATH_CALUDE_cost_780_candies_l2229_222994

/-- The cost of buying a given number of chocolate candies -/
def chocolateCost (candies : ℕ) : ℚ :=
  let boxSize := 30
  let boxCost := 8
  let discountThreshold := 500
  let discountRate := 0.1
  let boxes := (candies + boxSize - 1) / boxSize  -- Ceiling division
  let totalCost := boxes * boxCost
  if candies > discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

/-- Theorem: The cost of buying 780 chocolate candies is $187.2 -/
theorem cost_780_candies :
  chocolateCost 780 = 187.2 := by
  sorry

end NUMINAMATH_CALUDE_cost_780_candies_l2229_222994


namespace NUMINAMATH_CALUDE_triangle_side_length_l2229_222981

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- a, b, c form an arithmetic sequence
  (B = π / 6) →  -- Angle B = 30° (in radians)
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- Area of triangle ABC = 3/2
  -- Conclusion
  b = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2229_222981


namespace NUMINAMATH_CALUDE_equation_equality_l2229_222973

theorem equation_equality (a b : ℝ) : 1 - a^2 + 2*a*b - b^2 = 1 - (a^2 - 2*a*b + b^2) := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2229_222973


namespace NUMINAMATH_CALUDE_point_properties_l2229_222967

def P : ℝ × ℝ := (4, -2)

theorem point_properties :
  -- Distance from P to y-axis
  (abs P.1 = 4) ∧
  -- Reflection of P across x-axis
  (P.1, -P.2) = (4, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_properties_l2229_222967


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2229_222901

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 6 = 0 ∧ 
   x₂^2 + a*x₂ + 6 = 0 ∧ 
   x₁ - 72/(25*x₂^3) = x₂ - 72/(25*x₁^3)) → 
  (a = 9 ∨ a = -9) := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_condition_l2229_222901


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2229_222935

theorem solve_linear_equation :
  ∃ y : ℝ, (7 * y - 10 = 4 * y + 5) ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2229_222935


namespace NUMINAMATH_CALUDE_gas_refill_amount_l2229_222904

/-- Calculates the amount of gas needed to refill a car's tank --/
theorem gas_refill_amount (initial_gas tank_capacity store_trip doctor_trip : ℝ) 
  (h1 : initial_gas = 10)
  (h2 : tank_capacity = 12)
  (h3 : store_trip = 6)
  (h4 : doctor_trip = 2) :
  tank_capacity - (initial_gas - store_trip - doctor_trip) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gas_refill_amount_l2229_222904


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2229_222952

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  sphere_radius : ℝ
  is_tangent : Bool

/-- The theorem stating the radius of the sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone 
  (cone : TruncatedConeWithSphere) 
  (h1 : cone.bottom_radius = 24) 
  (h2 : cone.top_radius = 6) 
  (h3 : cone.is_tangent = true) : 
  cone.sphere_radius = 12 := by
  sorry

#check sphere_radius_in_truncated_cone

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2229_222952


namespace NUMINAMATH_CALUDE_ken_kept_pencils_l2229_222909

def pencil_distribution (total : ℕ) (manny nilo carlos tina rina : ℕ) : Prop :=
  total = 200 ∧
  manny = 20 ∧
  nilo = manny + 10 ∧
  carlos = nilo + 5 ∧
  tina = carlos + 15 ∧
  rina = tina + 5

theorem ken_kept_pencils (total manny nilo carlos tina rina : ℕ) :
  pencil_distribution total manny nilo carlos tina rina →
  total - (manny + nilo + carlos + tina + rina) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ken_kept_pencils_l2229_222909


namespace NUMINAMATH_CALUDE_washer_dryer_cost_l2229_222925

theorem washer_dryer_cost (dryer_cost washer_cost total_cost : ℕ) : 
  dryer_cost = 150 →
  washer_cost = 3 * dryer_cost →
  total_cost = washer_cost + dryer_cost →
  total_cost = 600 := by
sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_l2229_222925


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2229_222997

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | Real.sqrt (a^2 - 2*x^2) > x + a} = {x : ℝ | (Real.sqrt 2 / 2) * a ≤ x ∧ x ≤ -(Real.sqrt 2 / 2) * a} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2229_222997


namespace NUMINAMATH_CALUDE_horner_method_v3_l2229_222974

def f (x : ℝ) : ℝ := 3*x^5 - 2*x^4 + 2*x^3 - 4*x^2 - 7

def horner_v3 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (x : ℝ) : ℝ :=
  ((((a * x + b) * x + c) * x + d) * x + e) * x + f

theorem horner_method_v3 : 
  horner_v3 3 (-2) 2 (-4) 0 (-7) 2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2229_222974


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l2229_222928

theorem meaningful_fraction_range :
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l2229_222928


namespace NUMINAMATH_CALUDE_tiffany_bag_difference_l2229_222956

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 8

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 7

/-- The difference between the number of bags on Monday and the next day -/
def bag_difference : ℕ := monday_bags - next_day_bags

theorem tiffany_bag_difference : bag_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bag_difference_l2229_222956


namespace NUMINAMATH_CALUDE_peach_baskets_l2229_222927

theorem peach_baskets (red_per_basket : ℕ) (total_red : ℕ) (h1 : red_per_basket = 16) (h2 : total_red = 96) :
  total_red / red_per_basket = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_baskets_l2229_222927


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2229_222970

/-- Given a principal amount and an interest rate, if the simple interest for 2 years
    is $600 and the compound interest for 2 years is $612, then the interest rate is 104%. -/
theorem interest_rate_calculation (P R : ℝ) : 
  P * R * 2 / 100 = 600 →
  P * ((1 + R / 100)^2 - 1) = 612 →
  R = 104 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2229_222970


namespace NUMINAMATH_CALUDE_calculate_flat_fee_l2229_222979

/-- Calculates the flat fee for shipping given the total cost, cost per pound, and weight. -/
theorem calculate_flat_fee (C : ℝ) (cost_per_pound : ℝ) (weight : ℝ) (h1 : C = 9) (h2 : cost_per_pound = 0.8) (h3 : weight = 5) :
  ∃ F : ℝ, C = F + cost_per_pound * weight ∧ F = 5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_flat_fee_l2229_222979


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l2229_222917

theorem quadratic_form_k_value :
  ∀ (a h k : ℝ),
  (∀ x, x^2 - 7*x + 1 = a*(x - h)^2 + k) →
  k = -45/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l2229_222917


namespace NUMINAMATH_CALUDE_problem_solution_l2229_222926

theorem problem_solution (m n c : Int) (hm : m = -4) (hn : n = -5) (hc : c = -7) :
  m - n - c = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2229_222926


namespace NUMINAMATH_CALUDE_expression_result_l2229_222930

theorem expression_result : (3.242 * 12) / 100 = 0.38904 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l2229_222930


namespace NUMINAMATH_CALUDE_sams_juice_consumption_l2229_222958

theorem sams_juice_consumption (total_juice : ℚ) (sams_portion : ℚ) : 
  total_juice = 3/7 → sams_portion = 4/5 → sams_portion * total_juice = 12/35 := by
  sorry

end NUMINAMATH_CALUDE_sams_juice_consumption_l2229_222958


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2229_222996

theorem imaginary_part_of_complex_number (z : ℂ) : z = 1 - 2*I → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2229_222996


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2229_222955

theorem system_of_equations_solution (x y m : ℝ) : 
  x + 2*y = 5*m →
  x - 2*y = 9*m →
  3*x + 2*y = 19 →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2229_222955


namespace NUMINAMATH_CALUDE_order_of_fractions_l2229_222950

theorem order_of_fractions (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) := by
  sorry

end NUMINAMATH_CALUDE_order_of_fractions_l2229_222950


namespace NUMINAMATH_CALUDE_three_digit_palindrome_gcf_and_divisibility_l2229_222943

/-- Represents a three-digit palindrome -/
def ThreeDigitPalindrome : Type := { n : ℕ // ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 102 * a + 10 * b }

/-- The set of all three-digit palindromes -/
def AllThreeDigitPalindromes : Set ℕ :=
  { n | ∃ (p : ThreeDigitPalindrome), n = p.val }

theorem three_digit_palindrome_gcf_and_divisibility :
  (∃ (g : ℕ), g > 0 ∧ 
    (∀ n ∈ AllThreeDigitPalindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ AllThreeDigitPalindromes, d ∣ n) → d ∣ g)) ∧
  (∀ n ∈ AllThreeDigitPalindromes, 3 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_palindrome_gcf_and_divisibility_l2229_222943


namespace NUMINAMATH_CALUDE_pi_half_irrational_l2229_222939

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l2229_222939


namespace NUMINAMATH_CALUDE_max_a_for_increasing_cubic_l2229_222923

/-- Given that f(x) = x^3 - ax is increasing on [1, +∞), 
    the maximum value of the real number a is 3. -/
theorem max_a_for_increasing_cubic (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) →
  (∀ x : ℝ, f x = x^3 - a*x) →
  a ≤ 3 ∧ ∀ b > 3, ∃ x ≥ 1, ∃ y > x, f x ≥ f y := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_increasing_cubic_l2229_222923


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2229_222910

theorem other_root_of_quadratic (p : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + p * x = 9) ∧ 
  (7 * (-3)^2 + p * (-3) = 9) → 
  7 * (3/7)^2 + p * (3/7) = 9 :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2229_222910


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2229_222962

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α < 0) : 
  ∃ (x y : Real), x > 0 ∧ y < 0 ∧ Real.cos α = x ∧ Real.sin α = y :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2229_222962


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l2229_222913

theorem unique_solution_inequality (x : ℝ) : 
  x > 0 → x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l2229_222913


namespace NUMINAMATH_CALUDE_sum_divisibility_l2229_222990

theorem sum_divisibility : 
  let y := 72 + 144 + 216 + 288 + 576 + 720 + 4608
  (∃ k : ℤ, y = 6 * k) ∧ 
  (∃ k : ℤ, y = 12 * k) ∧ 
  (∃ k : ℤ, y = 24 * k) ∧ 
  ¬(∃ k : ℤ, y = 48 * k) := by
  sorry

end NUMINAMATH_CALUDE_sum_divisibility_l2229_222990


namespace NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_proof_l2229_222965

theorem multiply_subtract_distribute (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem computation_proof : 72 * 808 - 22 * 808 = 40400 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_proof_l2229_222965


namespace NUMINAMATH_CALUDE_equation_solutions_l2229_222919

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6) ∧
  (∀ x : ℝ, (1/2) * (x - 1)^3 = -4 ↔ x = -1) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2229_222919


namespace NUMINAMATH_CALUDE_unique_pairs_theorem_l2229_222948

theorem unique_pairs_theorem (x y : ℕ) : 
  x ≥ 2 → y ≥ 2 → 
  (3 * x) % y = 1 → 
  (3 * y) % x = 1 → 
  (x * y) % 3 = 1 → 
  ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) := by
  sorry

#check unique_pairs_theorem

end NUMINAMATH_CALUDE_unique_pairs_theorem_l2229_222948


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l2229_222934

theorem roots_of_polynomials (α : ℝ) : 
  α^2 = 2*α + 2 → α^5 = 44*α + 32 := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_l2229_222934


namespace NUMINAMATH_CALUDE_first_digit_of_5_to_n_l2229_222938

theorem first_digit_of_5_to_n (n : ℕ) : 
  (∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) → 
  (∃ m : ℕ, 10^m ≤ 5^n ∧ 5^n < 2 * 10^m) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_of_5_to_n_l2229_222938


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2229_222936

/-- A rectangle with perimeter 60 meters and area 221 square meters has dimensions 17 meters and 13 meters. -/
theorem rectangle_dimensions (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 60) (h_area : l * w = 221) :
  (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2229_222936


namespace NUMINAMATH_CALUDE_inscribed_polygon_sides_l2229_222944

theorem inscribed_polygon_sides (r : ℝ) (n : ℕ) (a : ℝ) : 
  r = 1 → 
  a = 2 * Real.sin (π / n) → 
  1 < a → 
  a < Real.sqrt 2 → 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_polygon_sides_l2229_222944


namespace NUMINAMATH_CALUDE_towel_purchase_cost_is_correct_l2229_222912

/-- Calculates the total cost of Bailey's towel purchase --/
def towel_purchase_cost : ℝ :=
  let guest_price := 40
  let master_price := 50
  let hand_price := 30
  let kitchen_price := 20
  
  let guest_discount := 0.15
  let master_discount := 0.20
  let hand_discount := 0.15
  let kitchen_discount := 0.10
  
  let sales_tax := 0.08
  
  let guest_discounted := guest_price * (1 - guest_discount)
  let master_discounted := master_price * (1 - master_discount)
  let hand_discounted := hand_price * (1 - hand_discount)
  let kitchen_discounted := kitchen_price * (1 - kitchen_discount)
  
  let total_before_tax := 
    2 * guest_discounted + 
    4 * master_discounted + 
    3 * hand_discounted + 
    5 * kitchen_discounted
  
  total_before_tax * (1 + sales_tax)

/-- Theorem stating that the total cost of Bailey's towel purchase is $426.06 --/
theorem towel_purchase_cost_is_correct : 
  towel_purchase_cost = 426.06 := by sorry

end NUMINAMATH_CALUDE_towel_purchase_cost_is_correct_l2229_222912


namespace NUMINAMATH_CALUDE_three_times_relationship_l2229_222905

theorem three_times_relationship (M₁ M₂ M₃ M₄ : ℝ) 
  (h₁ : M₁ = 2.02e-6)
  (h₂ : M₂ = 0.0000202)
  (h₃ : M₃ = 0.00000202)
  (h₄ : M₄ = 6.06e-5) :
  (M₄ = 3 * M₂ ∧ 
   M₄ ≠ 3 * M₁ ∧ 
   M₄ ≠ 3 * M₃ ∧ 
   M₃ ≠ 3 * M₁ ∧ 
   M₃ ≠ 3 * M₂ ∧ 
   M₂ ≠ 3 * M₁) :=
by sorry

end NUMINAMATH_CALUDE_three_times_relationship_l2229_222905


namespace NUMINAMATH_CALUDE_point_coordinates_l2229_222906

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (M : Point) 
  (h1 : fourth_quadrant M)
  (h2 : distance_to_x_axis M = 3)
  (h3 : distance_to_y_axis M = 4) :
  M.x = 4 ∧ M.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2229_222906


namespace NUMINAMATH_CALUDE_periodic_function_value_l2229_222961

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value :
  ∀ (f : ℝ → ℝ),
  is_periodic f 4 →
  (∀ x ∈ Set.Icc 0 4, f x = x) →
  f 7.6 = 3.6 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2229_222961


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l2229_222998

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l2229_222998


namespace NUMINAMATH_CALUDE_compare_sizes_l2229_222954

theorem compare_sizes (a b : ℝ) (ha : a = 0.2^(1/2)) (hb : b = 0.5^(1/5)) :
  0 < a ∧ a < b ∧ b < 1 := by sorry

end NUMINAMATH_CALUDE_compare_sizes_l2229_222954


namespace NUMINAMATH_CALUDE_sum_b_c_is_48_l2229_222932

/-- An arithmetic sequence with six terms -/
structure ArithmeticSequence :=
  (a₁ : ℝ)
  (a₂ : ℝ)
  (a₃ : ℝ)
  (a₄ : ℝ)
  (a₅ : ℝ)
  (a₆ : ℝ)
  (is_arithmetic : ∃ d : ℝ, a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d ∧ a₅ - a₄ = d ∧ a₆ - a₅ = d)

/-- The sum of the third and fifth terms in the specific arithmetic sequence -/
def sum_b_c (seq : ArithmeticSequence) : ℝ := seq.a₃ + seq.a₅

/-- Theorem stating that for the given arithmetic sequence, the sum of b and c is 48 -/
theorem sum_b_c_is_48 (seq : ArithmeticSequence) 
  (h₁ : seq.a₁ = 3)
  (h₂ : seq.a₂ = 10)
  (h₃ : seq.a₄ = 24)
  (h₄ : seq.a₆ = 38) :
  sum_b_c seq = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_c_is_48_l2229_222932


namespace NUMINAMATH_CALUDE_radical_simplification_l2229_222983

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (8 * q^3) = 6 * q^3 * Real.sqrt (10 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2229_222983


namespace NUMINAMATH_CALUDE_no_lcm_83_l2229_222914

theorem no_lcm_83 (a b c : ℕ) : 
  a = 23 → b = 46 → Nat.lcm a (Nat.lcm b c) = 83 → False :=
by
  sorry

#check no_lcm_83

end NUMINAMATH_CALUDE_no_lcm_83_l2229_222914


namespace NUMINAMATH_CALUDE_oarsmen_count_l2229_222922

/-- The number of oarsmen in the boat -/
def n : ℕ := sorry

/-- The total weight of the oarsmen before replacement -/
def W : ℝ := sorry

/-- The average weight increase after replacement -/
def weight_increase : ℝ := 2

/-- The weight of the replaced crew member -/
def old_weight : ℝ := 40

/-- The weight of the new crew member -/
def new_weight : ℝ := 80

/-- Theorem stating that the number of oarsmen is 20 -/
theorem oarsmen_count : n = 20 := by
  have h1 : (W + new_weight - old_weight) / n = W / n + weight_increase := by sorry
  sorry

end NUMINAMATH_CALUDE_oarsmen_count_l2229_222922


namespace NUMINAMATH_CALUDE_vector_problem_l2229_222964

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

/-- Acute angle between vectors -/
def is_acute_angle (v w : Fin 2 → ℝ) : Prop := 
  dot_product v w > 0 ∧ ¬ ∃ (k : ℝ), v = fun i => k * (w i)

/-- Orthogonality of vectors -/
def is_orthogonal (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

theorem vector_problem (x : ℝ) :
  (is_acute_angle a (b x) ↔ x > -2 ∧ x ≠ 1/2) ∧
  (is_orthogonal (fun i => a i + 2 * (b x i)) (fun i => 2 * a i - b x i) ↔ x = 7/2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2229_222964


namespace NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l2229_222960

/-- A convex quadrilateral with an interior point -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  wq : ℝ
  xq : ℝ
  yq : ℝ
  zq : ℝ
  convex : Bool
  interior : Bool

/-- The perimeter of a quadrilateral -/
def perimeter (quad : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific quadrilateral is 230 + 10√41 -/
theorem specific_quadrilateral_perimeter (quad : ConvexQuadrilateral) 
  (h_area : quad.area = 2500)
  (h_wq : quad.wq = 30)
  (h_xq : quad.xq = 40)
  (h_yq : quad.yq = 50)
  (h_zq : quad.zq = 60)
  (h_convex : quad.convex = true)
  (h_interior : quad.interior = true) :
  perimeter quad = 230 + 10 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l2229_222960


namespace NUMINAMATH_CALUDE_continued_fraction_convergents_l2229_222963

/-- Continued fraction convergents -/
theorem continued_fraction_convergents
  (P Q : ℕ → ℤ)  -- Sequences of numerators and denominators
  (a : ℕ → ℕ)    -- Sequence of continued fraction coefficients
  (h1 : ∀ k, k ≥ 2 → P k = a k * P (k-1) + P (k-2))
  (h2 : ∀ k, k ≥ 2 → Q k = a k * Q (k-1) + Q (k-2))
  (h3 : ∀ k, a k > 0) :
  (∀ k, k ≥ 2 → P k * Q (k-2) - P (k-2) * Q k = (-1)^k * a k) ∧
  (∀ k, k ≥ 1 → (P k : ℚ) / Q k - (P (k-1) : ℚ) / Q (k-1) = (-1)^(k+1) / (Q k * Q (k-1))) ∧
  (∀ n, n ≥ 1 → ∀ k, 1 ≤ k → k < n → Q k < Q (k+1)) ∧
  (∀ n, n ≥ 0 → (P 0 : ℚ) / Q 0 < (P 2 : ℚ) / Q 2 ∧ 
    (P n : ℚ) / Q n < (P (n+1) : ℚ) / Q (n+1) ∧
    (P (n+2) : ℚ) / Q (n+2) < (P (n+1) : ℚ) / Q (n+1)) ∧
  (∀ k l, k ≥ 0 → l ≥ 0 → (P (2*k) : ℚ) / Q (2*k) < (P (2*l+1) : ℚ) / Q (2*l+1)) :=
by sorry

end NUMINAMATH_CALUDE_continued_fraction_convergents_l2229_222963


namespace NUMINAMATH_CALUDE_regular_polygon_140_deg_interior_angle_l2229_222941

/-- A regular polygon with interior angles of 140 degrees has 9 sides -/
theorem regular_polygon_140_deg_interior_angle (n : ℕ) : 
  (n ≥ 3) → 
  (∀ θ : ℝ, θ = 140 → (180 * (n - 2) : ℝ) = n * θ) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_140_deg_interior_angle_l2229_222941


namespace NUMINAMATH_CALUDE_count_numbers_with_4_or_6_eq_1105_l2229_222949

-- Define the range of numbers we're considering
def range_end : Nat := 2401

-- Define a function to check if a number in base 8 contains 4 or 6
def contains_4_or_6 (n : Nat) : Bool :=
  sorry

-- Define the count of numbers containing 4 or 6
def count_numbers_with_4_or_6 : Nat :=
  (List.range range_end).filter contains_4_or_6 |>.length

-- Theorem to prove
theorem count_numbers_with_4_or_6_eq_1105 :
  count_numbers_with_4_or_6 = 1105 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_4_or_6_eq_1105_l2229_222949


namespace NUMINAMATH_CALUDE_balance_theorem_l2229_222978

-- Define the weights of balls in terms of blue balls
def green_weight : ℚ := 2
def yellow_weight : ℚ := 5/2
def white_weight : ℚ := 3/2

-- Define the balance conditions
axiom green_balance : 3 * green_weight = 6
axiom yellow_balance : 2 * yellow_weight = 5
axiom white_balance : 6 = 4 * white_weight

-- Theorem to prove
theorem balance_theorem : 
  4 * green_weight + 2 * yellow_weight + 2 * white_weight = 16 := by
  sorry


end NUMINAMATH_CALUDE_balance_theorem_l2229_222978


namespace NUMINAMATH_CALUDE_exhibition_arrangement_l2229_222937

/-- The number of display stands -/
def n : ℕ := 9

/-- The number of exhibits -/
def k : ℕ := 3

/-- The number of ways to arrange k distinct objects in n positions,
    where the objects cannot be placed at the ends or adjacent to each other -/
def arrangement_count (n k : ℕ) : ℕ :=
  if n < 2 * k + 1 then 0
  else (n - k - 1).choose k * k.factorial

theorem exhibition_arrangement :
  arrangement_count n k = 60 := by sorry

end NUMINAMATH_CALUDE_exhibition_arrangement_l2229_222937


namespace NUMINAMATH_CALUDE_characteristic_equation_of_A_l2229_222940

def A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 3, 1, 2; 2, 3, 1]

theorem characteristic_equation_of_A :
  ∃ (p q r : ℝ), A^3 + p • A^2 + q • A + r • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧
  p = -3 ∧ q = -9 ∧ r = -2 := by
  sorry

end NUMINAMATH_CALUDE_characteristic_equation_of_A_l2229_222940


namespace NUMINAMATH_CALUDE_broken_line_circle_cover_l2229_222982

/-- A closed broken line on a plane -/
structure ClosedBrokenLine :=
  (points : Set (ℝ × ℝ))
  (is_closed : sorry)
  (length : ℝ)

/-- Theorem: Any closed broken line of length 1 on a plane can be covered by a circle of radius 1/4 -/
theorem broken_line_circle_cover (L : ClosedBrokenLine) (h : L.length = 1) :
  ∃ (center : ℝ × ℝ), ∀ p ∈ L.points, dist p center ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_broken_line_circle_cover_l2229_222982


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2229_222951

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 3 + Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2229_222951


namespace NUMINAMATH_CALUDE_sector_to_cone_base_radius_l2229_222968

/-- Given a sector with central angle 120° and radius 9 cm, when formed into a cone,
    the radius of the base circle is 3 cm. -/
theorem sector_to_cone_base_radius (θ : ℝ) (R : ℝ) (r : ℝ) : 
  θ = 120 → R = 9 → r = (θ / 360) * R → r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_to_cone_base_radius_l2229_222968


namespace NUMINAMATH_CALUDE_solve_marbles_problem_l2229_222902

def marbles_problem (initial : ℕ) (gifted : ℕ) (final : ℕ) : Prop :=
  ∃ (lost : ℕ), initial - lost - gifted = final

theorem solve_marbles_problem :
  marbles_problem 85 25 43 → (∃ (lost : ℕ), lost = 17) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_marbles_problem_l2229_222902


namespace NUMINAMATH_CALUDE_shape_arrangement_possible_l2229_222945

-- Define a type for geometric shapes
structure Shape :=
  (area : ℕ)

-- Define a type for arrangements of shapes
structure Arrangement :=
  (shapes : List Shape)
  (width : ℕ)
  (height : ℕ)

-- Define the properties of the desired arrangements
def is_square_with_cutout (arr : Arrangement) : Prop :=
  arr.width = 9 ∧ arr.height = 9 ∧
  ∃ (center : Shape), center ∈ arr.shapes ∧ center.area = 9

def is_rectangle (arr : Arrangement) : Prop :=
  arr.width = 9 ∧ arr.height = 12

-- Define the given set of shapes
def given_shapes : List Shape := sorry

-- State the theorem
theorem shape_arrangement_possible :
  ∃ (arr1 arr2 : Arrangement),
    (∀ s ∈ arr1.shapes, s ∈ given_shapes) ∧
    (∀ s ∈ arr2.shapes, s ∈ given_shapes) ∧
    is_square_with_cutout arr1 ∧
    is_rectangle arr2 :=
  sorry

end NUMINAMATH_CALUDE_shape_arrangement_possible_l2229_222945


namespace NUMINAMATH_CALUDE_circle_area_difference_l2229_222959

/-- The difference between the areas of two circles with given circumferences -/
theorem circle_area_difference (c₁ c₂ : ℝ) (h₁ : c₁ = 660) (h₂ : c₂ = 704) :
  ∃ (diff : ℝ), abs (diff - ((c₂^2 - c₁^2) / (4 * Real.pi))) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2229_222959


namespace NUMINAMATH_CALUDE_smallest_sum_is_3257_l2229_222953

def Digits : Finset ℕ := {3, 7, 2, 9, 5}

def is_valid_pair (a b : ℕ) : Prop :=
  (a ≥ 1000 ∧ a < 10000) ∧ 
  (b ≥ 100 ∧ b < 1000) ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 5)

def sum_of_pair (a b : ℕ) : ℕ := a + b

theorem smallest_sum_is_3257 :
  ∀ a b : ℕ, is_valid_pair a b → sum_of_pair a b ≥ 3257 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_3257_l2229_222953
