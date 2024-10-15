import Mathlib

namespace NUMINAMATH_CALUDE_angle_phi_value_l2958_295899

theorem angle_phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : Real.sqrt 3 * Real.sin (20 * Real.pi / 180) = Real.cos φ - Real.sin φ) : 
  φ = 25 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_phi_value_l2958_295899


namespace NUMINAMATH_CALUDE_inequality_proof_l2958_295837

theorem inequality_proof (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_one : a + b + c + d = 1) :
  a * b * c + b * c * d + c * a * d + d * a * b ≤ 1 / 27 + (176 / 27) * a * b * c * d :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2958_295837


namespace NUMINAMATH_CALUDE_unique_prime_tower_l2958_295815

def tower_of_twos (p : ℕ) : ℕ :=
  match p with
  | 0 => 1
  | n + 1 => 2^(tower_of_twos n)

def is_prime_tower (p : ℕ) : Prop :=
  Nat.Prime (tower_of_twos p + 9)

theorem unique_prime_tower : ∀ p : ℕ, is_prime_tower p ↔ p = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_tower_l2958_295815


namespace NUMINAMATH_CALUDE_f_equality_l2958_295832

noncomputable def f (x : ℝ) : ℝ := Real.arctan ((2 * x) / (1 - x^2))

theorem f_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : 3 - 4 * x^2 ≠ 0) :
  f ((x - 4 * x^3) / (3 - 4 * x^2)) = f x :=
by sorry

end NUMINAMATH_CALUDE_f_equality_l2958_295832


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2958_295856

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 22) : 
  x^2 - y^2 = -120 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2958_295856


namespace NUMINAMATH_CALUDE_not_equivalent_expression_l2958_295870

theorem not_equivalent_expression (x : ℝ) : 
  (3 * (x + 2) = 3 * x + 6) ∧
  ((-9 * x - 18) / (-3) = 3 * x + 6) ∧
  ((1/3) * (9 * x + 18) = 3 * x + 6) ∧
  ((1/3) * (3 * x) + (2/3) * 9 ≠ 3 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_not_equivalent_expression_l2958_295870


namespace NUMINAMATH_CALUDE_stamp_difference_l2958_295834

theorem stamp_difference (p q : ℕ) (h1 : p * 4 = q * 7) 
  (h2 : (p - 8) * 5 = (q + 8) * 6) : p - q = 8 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l2958_295834


namespace NUMINAMATH_CALUDE_square_root_inequalities_l2958_295893

theorem square_root_inequalities : 
  (∃ (x y : ℝ), x = Real.sqrt 7 ∧ y = Real.sqrt 3 ∧ x + y ≠ Real.sqrt 10) ∧
  (Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15) ∧
  (Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2) ∧
  ((-Real.sqrt 3)^2 = 3) := by
  sorry


end NUMINAMATH_CALUDE_square_root_inequalities_l2958_295893


namespace NUMINAMATH_CALUDE_tan_domain_theorem_l2958_295876

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - π / 4)

def domain_set : Set ℝ := ⋃ k : ℤ, Ioo ((k : ℝ) * π / 2 - π / 8) ((k : ℝ) * π / 2 + 3 * π / 8)

theorem tan_domain_theorem :
  {x : ℝ | ∃ y, f x = y} = domain_set :=
sorry

end NUMINAMATH_CALUDE_tan_domain_theorem_l2958_295876


namespace NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l2958_295853

theorem product_greater_than_sum_minus_one {a₁ a₂ : ℝ} 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ > a₁ + a₂ - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l2958_295853


namespace NUMINAMATH_CALUDE_actual_tissue_diameter_l2958_295818

/-- The actual diameter of a circular tissue given its magnification and magnified image diameter -/
theorem actual_tissue_diameter 
  (magnification : ℝ) 
  (magnified_diameter : ℝ) 
  (h_magnification : magnification = 1000) 
  (h_magnified_diameter : magnified_diameter = 1) : 
  magnified_diameter / magnification = 0.001 := by
  sorry

end NUMINAMATH_CALUDE_actual_tissue_diameter_l2958_295818


namespace NUMINAMATH_CALUDE_weight_of_new_person_new_person_weight_l2958_295895

theorem weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  let new_weight := replaced_weight + initial_count * average_increase
  new_weight

/-- Given a group of 8 people where one person weighing 65 kg is replaced by a new person, 
    and the average weight of the group increases by 3 kg, 
    the weight of the new person is 89 kg. -/
theorem new_person_weight : weight_of_new_person 8 3 65 = 89 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_new_person_weight_l2958_295895


namespace NUMINAMATH_CALUDE_third_vertex_y_coord_value_l2958_295806

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle with two vertices given -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point
  is_equilateral : True  -- This is a placeholder for the equilateral property
  third_vertex_in_first_quadrant : v3.x > 0 ∧ v3.y > 0

/-- The y-coordinate of the third vertex of an equilateral triangle -/
def third_vertex_y_coord (t : EquilateralTriangle) : ℝ :=
  t.v3.y

/-- The theorem stating the y-coordinate of the third vertex -/
theorem third_vertex_y_coord_value (t : EquilateralTriangle) 
    (h1 : t.v1 = ⟨2, 3⟩) 
    (h2 : t.v2 = ⟨10, 3⟩) : 
  third_vertex_y_coord t = 3 + 4 * Real.sqrt 3 := by
  sorry

#check third_vertex_y_coord_value

end NUMINAMATH_CALUDE_third_vertex_y_coord_value_l2958_295806


namespace NUMINAMATH_CALUDE_seashells_sum_l2958_295865

/-- The number of seashells Mary found -/
def mary_shells : ℕ := 18

/-- The number of seashells Jessica found -/
def jessica_shells : ℕ := 41

/-- The total number of seashells found by Mary and Jessica -/
def total_shells : ℕ := mary_shells + jessica_shells

theorem seashells_sum : total_shells = 59 := by
  sorry

end NUMINAMATH_CALUDE_seashells_sum_l2958_295865


namespace NUMINAMATH_CALUDE_smallest_stairs_fifty_three_satisfies_stairs_solution_l2958_295844

theorem smallest_stairs (n : ℕ) : 
  (n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4) → n ≥ 53 :=
by sorry

theorem fifty_three_satisfies : 
  53 > 20 ∧ 53 % 6 = 5 ∧ 53 % 7 = 4 :=
by sorry

theorem stairs_solution : 
  ∃ (n : ℕ), n = 53 ∧ n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4 ∧
  ∀ (m : ℕ), (m > 20 ∧ m % 6 = 5 ∧ m % 7 = 4) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_fifty_three_satisfies_stairs_solution_l2958_295844


namespace NUMINAMATH_CALUDE_square_equation_solution_l2958_295824

theorem square_equation_solution (x y : ℕ) : 
  x^2 = y^2 + 7*y + 6 → x = 6 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2958_295824


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2958_295872

theorem arithmetic_geometric_sequence (x : ℝ) : 
  (∃ y : ℝ, 
    -- y is between 3 and x
    3 < y ∧ y < x ∧
    -- arithmetic sequence condition
    (y - 3 = x - y) ∧
    -- geometric sequence condition after subtracting 6 from the middle term
    ((y - 6) / 3 = x / (y - 6))) →
  (x = 3 ∨ x = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2958_295872


namespace NUMINAMATH_CALUDE_production_line_b_units_l2958_295811

/-- 
Given a factory with three production lines A, B, and C, prove that production line B 
produced 1000 units under the following conditions:
1. The total number of units produced is 3000
2. The number of units sampled from each production line (a, b, c) form an arithmetic sequence
3. The sum of a, b, and c equals the total number of units produced
-/
theorem production_line_b_units (a b c : ℕ) : 
  (a + b + c = 3000) → 
  (2 * b = a + c) → 
  b = 1000 := by
sorry

end NUMINAMATH_CALUDE_production_line_b_units_l2958_295811


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l2958_295829

def point_symmetry_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_y_axis :
  let P : ℝ × ℝ := (2, 3)
  point_symmetry_y_axis P = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l2958_295829


namespace NUMINAMATH_CALUDE_best_strategy_is_red_l2958_295849

/-- Represents the color of a disk side -/
inductive Color
| Red
| Blue

/-- Represents a disk with two sides -/
structure Disk where
  side1 : Color
  side2 : Color

/-- The set of all disks in the hat -/
def diskSet : Finset Disk := sorry

/-- The total number of disks -/
def totalDisks : ℕ := 10

/-- The number of disks with both sides red -/
def redDisks : ℕ := 3

/-- The number of disks with both sides blue -/
def blueDisks : ℕ := 2

/-- The number of disks with one side red and one side blue -/
def mixedDisks : ℕ := 5

/-- The probability of observing a red side -/
def probRedSide : ℚ := 11 / 20

/-- The probability of observing a blue side -/
def probBlueSide : ℚ := 9 / 20

/-- The probability that the other side is red, given that a red side is observed -/
def probRedGivenRed : ℚ := 6 / 11

/-- The probability that the other side is red, given that a blue side is observed -/
def probRedGivenBlue : ℚ := 5 / 9

theorem best_strategy_is_red :
  probRedGivenRed > 1 / 2 ∧ probRedGivenBlue > 1 / 2 := by sorry

end NUMINAMATH_CALUDE_best_strategy_is_red_l2958_295849


namespace NUMINAMATH_CALUDE_pam_has_1200_apples_l2958_295833

/-- The number of apples Pam has in total -/
def pams_total_apples (pams_bags : ℕ) (geralds_apples_per_bag : ℕ) : ℕ :=
  pams_bags * (3 * geralds_apples_per_bag)

/-- Theorem stating that Pam has 1200 apples given the conditions -/
theorem pam_has_1200_apples :
  pams_total_apples 10 40 = 1200 := by
  sorry

#eval pams_total_apples 10 40

end NUMINAMATH_CALUDE_pam_has_1200_apples_l2958_295833


namespace NUMINAMATH_CALUDE_sams_book_count_l2958_295846

/-- The number of books Sam bought at the school's book fair -/
def total_books (adventure_books mystery_books crime_books : ℝ) : ℝ :=
  adventure_books + mystery_books + crime_books

/-- Theorem stating the total number of books Sam bought -/
theorem sams_book_count :
  total_books 13 17 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sams_book_count_l2958_295846


namespace NUMINAMATH_CALUDE_beast_of_war_runtime_l2958_295841

/-- The running time of Millennium in hours -/
def millennium_runtime : ℝ := 2

/-- The difference in minutes between Millennium and Alpha Epsilon runtimes -/
def alpha_epsilon_diff : ℝ := 30

/-- The difference in minutes between Beast of War and Alpha Epsilon runtimes -/
def beast_of_war_diff : ℝ := 10

/-- Conversion factor from hours to minutes -/
def hours_to_minutes : ℝ := 60

/-- Theorem stating the runtime of Beast of War: Armoured Command -/
theorem beast_of_war_runtime : 
  millennium_runtime * hours_to_minutes - alpha_epsilon_diff + beast_of_war_diff = 100 := by
sorry

end NUMINAMATH_CALUDE_beast_of_war_runtime_l2958_295841


namespace NUMINAMATH_CALUDE_football_lineup_count_l2958_295873

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose a starting lineup for the given football team -/
theorem football_lineup_count :
  choose_lineup 15 5 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_count_l2958_295873


namespace NUMINAMATH_CALUDE_coin_equation_solution_l2958_295831

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of quarters on the left side of the equation -/
def left_quarters : ℕ := 30

/-- The number of dimes on the left side of the equation -/
def left_dimes : ℕ := 20

/-- The number of quarters on the right side of the equation -/
def right_quarters : ℕ := 5

theorem coin_equation_solution :
  ∃ n : ℕ, 
    left_quarters * quarter_value + left_dimes * dime_value = 
    right_quarters * quarter_value + n * dime_value ∧
    n = 83 := by
  sorry

end NUMINAMATH_CALUDE_coin_equation_solution_l2958_295831


namespace NUMINAMATH_CALUDE_expansion_equality_l2958_295879

-- Define the left-hand side of the equation
def lhs (x : ℝ) : ℝ := (5 * x^2 + 3 * x - 7) * 4 * x^3

-- Define the right-hand side of the equation
def rhs (x : ℝ) : ℝ := 20 * x^5 + 12 * x^4 - 28 * x^3

-- State the theorem
theorem expansion_equality : ∀ x : ℝ, lhs x = rhs x := by sorry

end NUMINAMATH_CALUDE_expansion_equality_l2958_295879


namespace NUMINAMATH_CALUDE_sequence_inequality_l2958_295852

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

/-- The main theorem -/
theorem sequence_inequality (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (heq : a 11 = b 10) : 
  a 13 + a 9 ≤ b 14 + b 6 :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2958_295852


namespace NUMINAMATH_CALUDE_phone_plan_fee_proof_l2958_295828

/-- The monthly fee for the first plan -/
def first_plan_fee : ℝ := 22

/-- The per-minute rate for the first plan -/
def first_plan_rate : ℝ := 0.13

/-- The per-minute rate for the second plan -/
def second_plan_rate : ℝ := 0.18

/-- The number of minutes at which the plans cost the same -/
def equal_cost_minutes : ℝ := 280

/-- The monthly fee for the second plan -/
def second_plan_fee : ℝ := 8

theorem phone_plan_fee_proof :
  first_plan_fee + first_plan_rate * equal_cost_minutes =
  second_plan_fee + second_plan_rate * equal_cost_minutes :=
by sorry

end NUMINAMATH_CALUDE_phone_plan_fee_proof_l2958_295828


namespace NUMINAMATH_CALUDE_supermarket_flour_import_l2958_295858

theorem supermarket_flour_import (long_grain : ℚ) (glutinous : ℚ) (flour : ℚ) : 
  long_grain = 9/20 →
  glutinous = 7/20 →
  flour = long_grain + glutinous - 3/20 →
  flour = 13/20 := by
sorry

end NUMINAMATH_CALUDE_supermarket_flour_import_l2958_295858


namespace NUMINAMATH_CALUDE_inequality_proof_l2958_295886

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2958_295886


namespace NUMINAMATH_CALUDE_pure_imaginary_m_l2958_295855

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_m (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 1) (m^2 + 2*m - 3)
  is_pure_imaginary z → m = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_l2958_295855


namespace NUMINAMATH_CALUDE_equation_may_not_hold_l2958_295874

theorem equation_may_not_hold (a b c : ℝ) : 
  a = b → ¬(∀ c, a / c = b / c) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_may_not_hold_l2958_295874


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2958_295840

theorem inscribed_circle_radius (AB AC BC : ℝ) (h_AB : AB = 8) (h_AC : AC = 10) (h_BC : BC = 12) :
  let s := (AB + AC + BC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  area / s = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2958_295840


namespace NUMINAMATH_CALUDE_complex_distance_bounds_l2958_295889

theorem complex_distance_bounds (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  (∃ (w : ℂ), Complex.abs (z - 2 - 2*I) = 3 ∧ 
    ∀ (u : ℂ), Complex.abs (u + 2 - 2*I) = 1 → Complex.abs (u - 2 - 2*I) ≥ 3) ∧
  (∃ (w : ℂ), Complex.abs (z - 2 - 2*I) = 5 ∧ 
    ∀ (u : ℂ), Complex.abs (u + 2 - 2*I) = 1 → Complex.abs (u - 2 - 2*I) ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_complex_distance_bounds_l2958_295889


namespace NUMINAMATH_CALUDE_distance_apex_to_circumsphere_center_l2958_295823

/-- Represents a rectangular pyramid with a frustum -/
structure RectangularPyramidWithFrustum where
  /-- Length of the rectangle base -/
  baseLength : ℝ
  /-- Width of the rectangle base -/
  baseWidth : ℝ
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Ratio of the volume of the smaller pyramid to the whole pyramid -/
  volumeRatio : ℝ

/-- Theorem stating the distance between the apex and the center of the frustum's circumsphere -/
theorem distance_apex_to_circumsphere_center
  (p : RectangularPyramidWithFrustum)
  (h1 : p.baseLength = 15)
  (h2 : p.baseWidth = 20)
  (h3 : p.pyramidHeight = 30)
  (h4 : p.volumeRatio = 1/9) :
  let xt := p.pyramidHeight - (1 - p.volumeRatio^(1/3)) * p.pyramidHeight +
            (p.baseLength^2 + p.baseWidth^2) / (18 * p.pyramidHeight)
  xt = 425/9 := by
  sorry

#check distance_apex_to_circumsphere_center

end NUMINAMATH_CALUDE_distance_apex_to_circumsphere_center_l2958_295823


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l2958_295859

/-- Proposition p: For all x ∈ [1, 2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: The equation x^2 + 2ax + 2 - a = 0 has real roots -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The proposition "¬p ∨ ¬q" is false -/
def not_p_or_not_q_is_false (a : ℝ) : Prop :=
  ¬(¬(prop_p a) ∨ ¬(prop_q a))

/-- The range of the real number a is a ≤ -2 or a = 1 -/
def range_of_a (a : ℝ) : Prop :=
  a ≤ -2 ∨ a = 1

theorem range_of_a_theorem (a : ℝ) :
  prop_p a ∧ prop_q a ∧ not_p_or_not_q_is_false a → range_of_a a :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l2958_295859


namespace NUMINAMATH_CALUDE_expand_expression_l2958_295864

theorem expand_expression (x y z : ℝ) : 
  (2*x - 3) * (4*y + 5 - 2*z) = 8*x*y + 10*x - 4*x*z - 12*y + 6*z - 15 := by
sorry

end NUMINAMATH_CALUDE_expand_expression_l2958_295864


namespace NUMINAMATH_CALUDE_average_weight_problem_l2958_295860

theorem average_weight_problem (A B C : ℝ) : 
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2958_295860


namespace NUMINAMATH_CALUDE_isosceles_triangle_k_values_l2958_295835

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  isIsosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ (side1 = side3 ∧ side2 ≠ side1) ∨ (side2 = side3 ∧ side1 ≠ side2)

-- Define the quadratic equation
def quadraticRoots (k : ℝ) : Set ℝ :=
  {x : ℝ | x^2 - 4*x + k = 0}

-- Theorem statement
theorem isosceles_triangle_k_values :
  ∀ (t : IsoscelesTriangle) (k : ℝ),
    (t.side1 = 3 ∨ t.side2 = 3 ∨ t.side3 = 3) →
    (∃ (x y : ℝ), x ∈ quadraticRoots k ∧ y ∈ quadraticRoots k ∧ 
      ((t.side1 = x ∧ t.side2 = y) ∨ (t.side1 = x ∧ t.side3 = y) ∨ (t.side2 = x ∧ t.side3 = y))) →
    (k = 3 ∨ k = 4) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_k_values_l2958_295835


namespace NUMINAMATH_CALUDE_quadratic_roots_l2958_295863

theorem quadratic_roots (x : ℝ) : x^2 - 3*x + 2 = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2958_295863


namespace NUMINAMATH_CALUDE_negation_of_implication_l2958_295882

theorem negation_of_implication (a b : ℝ) :
  ¬(a = 0 → a * b = 0) ↔ (a ≠ 0 → a * b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2958_295882


namespace NUMINAMATH_CALUDE_absent_students_percentage_l2958_295845

theorem absent_students_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (boys_absent_fraction : ℚ) (girls_absent_fraction : ℚ) :
  total_students = 120 →
  boys = 70 →
  girls = 50 →
  boys_absent_fraction = 1 / 7 →
  girls_absent_fraction = 1 / 5 →
  (↑(boys_absent_fraction * boys + girls_absent_fraction * girls) : ℚ) / total_students = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_absent_students_percentage_l2958_295845


namespace NUMINAMATH_CALUDE_last_digit_difference_l2958_295825

theorem last_digit_difference (p q : ℕ) : 
  p > q → 
  p % 10 ≠ 0 → 
  q % 10 ≠ 0 → 
  ∃ k : ℕ, p * q = 10^k → 
  (p - q) % 10 ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_difference_l2958_295825


namespace NUMINAMATH_CALUDE_min_value_of_a_l2958_295817

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

theorem min_value_of_a (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, f a x ≥ 2) → a ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2958_295817


namespace NUMINAMATH_CALUDE_john_test_scores_l2958_295890

theorem john_test_scores (total_tests : ℕ) (target_percentage : ℚ) 
  (tests_taken : ℕ) (tests_at_target : ℕ) : 
  total_tests = 60 →
  target_percentage = 85 / 100 →
  tests_taken = 40 →
  tests_at_target = 28 →
  (total_tests - tests_taken : ℕ) - 
    (↑total_tests * target_percentage - tests_at_target : ℚ).floor = 0 :=
by sorry

end NUMINAMATH_CALUDE_john_test_scores_l2958_295890


namespace NUMINAMATH_CALUDE_triangle_height_l2958_295888

theorem triangle_height (a b c : ℝ) (h_sum : a + c = 11) 
  (h_angle : Real.cos (π / 3) = 1 / 2) 
  (h_radius : (a * b * Real.sin (π / 3)) / (a + b + c) = 2 / Real.sqrt 3) 
  (h_longer : a > c) : 
  c * Real.sin (π / 3) = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l2958_295888


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2958_295826

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  P : ℝ × ℝ
  h_on_ellipse : (P.1^2 / E.a^2) + (P.2^2 / E.b^2) = 1

/-- The foci of the ellipse -/
def foci (E : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (E : Ellipse) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity (E : Ellipse) (P : PointOnEllipse E) :
  let (F₁, F₂) := foci E
  dot_product (P.P.1 - F₁.1, P.P.2 - F₁.2) (P.P.1 - F₂.1, P.P.2 - F₂.2) = 0 →
  Real.tan (angle (P.P.1 - F₁.1, P.P.2 - F₁.2) (F₂.1 - F₁.1, F₂.2 - F₁.2)) = 1/2 →
  eccentricity E = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2958_295826


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l2958_295827

theorem negation_of_existence_inequality (p : Prop) :
  (p ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) →
  (¬p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l2958_295827


namespace NUMINAMATH_CALUDE_expression_simplification_l2958_295897

theorem expression_simplification (a : ℝ) (h : a = 3) : 
  (a^2 / (a + 1)) - (1 / (a + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2958_295897


namespace NUMINAMATH_CALUDE_shirt_cost_to_marked_price_ratio_l2958_295843

/-- Given a shop with shirts on sale, this theorem proves the ratio of cost to marked price. -/
theorem shirt_cost_to_marked_price_ratio :
  ∀ (marked_price : ℝ), marked_price > 0 →
  let discount_rate : ℝ := 0.25
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 0.60
  let cost_price : ℝ := selling_price * cost_rate
  cost_price / marked_price = 0.45 := by
sorry


end NUMINAMATH_CALUDE_shirt_cost_to_marked_price_ratio_l2958_295843


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l2958_295801

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l2958_295801


namespace NUMINAMATH_CALUDE_phone_plan_ratio_l2958_295804

/-- Given Mandy's phone data plan details, prove the ratio of promotional rate to normal rate -/
theorem phone_plan_ratio : 
  ∀ (normal_rate promotional_rate : ℚ),
  normal_rate = 30 →
  promotional_rate + 2 * normal_rate + (normal_rate + 15) + 2 * normal_rate = 175 →
  promotional_rate / normal_rate = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_phone_plan_ratio_l2958_295804


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2958_295848

theorem inequality_system_solution_set :
  ∀ x : ℝ, (3/2 * x + 5 ≤ -1 ∧ x + 3 < 0) ↔ x ≤ -4 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2958_295848


namespace NUMINAMATH_CALUDE_ice_skating_skiing_probability_l2958_295854

theorem ice_skating_skiing_probability (P_ice_skating P_skiing P_either : ℝ)
  (h1 : P_ice_skating = 0.6)
  (h2 : P_skiing = 0.5)
  (h3 : P_either = 0.7)
  (h4 : 0 ≤ P_ice_skating ∧ P_ice_skating ≤ 1)
  (h5 : 0 ≤ P_skiing ∧ P_skiing ≤ 1)
  (h6 : 0 ≤ P_either ∧ P_either ≤ 1) :
  (P_ice_skating + P_skiing - P_either) / P_skiing = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_ice_skating_skiing_probability_l2958_295854


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l2958_295871

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - abs x - 6

-- Theorem stating that f has exactly two zeros
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l2958_295871


namespace NUMINAMATH_CALUDE_set_equality_implies_a_squared_minus_b_zero_l2958_295875

theorem set_equality_implies_a_squared_minus_b_zero (a b : ℝ) (h : a ≠ 0) :
  ({1, a + b, a} : Set ℝ) = {0, b / a, b} →
  a^2 - b = 0 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_squared_minus_b_zero_l2958_295875


namespace NUMINAMATH_CALUDE_third_month_sale_proof_l2958_295892

/-- Calculates the sale in the third month given the sales for other months and the average --/
def third_month_sale (first_month : ℕ) (second_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  6 * average - (first_month + second_month + fourth_month + fifth_month + sixth_month)

theorem third_month_sale_proof :
  third_month_sale 5266 5744 6122 6588 4916 5750 = 5864 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_proof_l2958_295892


namespace NUMINAMATH_CALUDE_final_score_for_five_hours_l2958_295898

/-- Represents a student's test performance -/
structure TestPerformance where
  maxPoints : ℝ
  preparationTime : ℝ
  score : ℝ
  effortBonus : ℝ

/-- Calculates the final score given a TestPerformance -/
def finalScore (tp : TestPerformance) : ℝ :=
  tp.score * (1 + tp.effortBonus)

/-- Theorem stating the final score for 5 hours of preparation -/
theorem final_score_for_five_hours 
  (tp : TestPerformance)
  (h1 : tp.maxPoints = 150)
  (h2 : tp.preparationTime = 5)
  (h3 : tp.effortBonus = 0.1)
  (h4 : ∃ (t : TestPerformance), t.preparationTime = 2 ∧ t.score = 90 ∧ 
        tp.score / tp.preparationTime = t.score / t.preparationTime) :
  finalScore tp = 247.5 := by
sorry


end NUMINAMATH_CALUDE_final_score_for_five_hours_l2958_295898


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2958_295821

theorem polynomial_multiplication_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^9 + 5*y^7 + 2*y^5) = 
  15*y^13 - 10*y^12 + 9*y^10 - 6*y^9 + 15*y^8 - 10*y^7 + 6*y^6 - 4*y^5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2958_295821


namespace NUMINAMATH_CALUDE_hash_difference_l2958_295803

-- Define the # operation
def hash (x y : ℝ) : ℝ := x * y - 3 * x

-- State the theorem
theorem hash_difference : hash 8 3 - hash 3 8 = -15 := by sorry

end NUMINAMATH_CALUDE_hash_difference_l2958_295803


namespace NUMINAMATH_CALUDE_min_sum_with_real_roots_l2958_295881

theorem min_sum_with_real_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  a + b ≥ Real.rpow 1728 (1/3) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (∃ x : ℝ, x^2 + a₀*x + 3*b₀ = 0) ∧
    (∃ x : ℝ, x^2 + 2*b₀*x + a₀ = 0) ∧
    a₀ + b₀ = Real.rpow 1728 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_real_roots_l2958_295881


namespace NUMINAMATH_CALUDE_parabola_shift_l2958_295830

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The shifted parabola function -/
def g (x : ℝ) : ℝ := (x + 1)^2 - 4*(x + 1) + 3 + 2

/-- Theorem stating that the shifted parabola is equivalent to x^2 - 2x + 2 -/
theorem parabola_shift :
  ∀ x : ℝ, g x = x^2 - 2*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2958_295830


namespace NUMINAMATH_CALUDE_escalator_speed_l2958_295878

theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 180 →
  person_speed = 3 →
  time_taken = 10 →
  ∃ (escalator_speed : ℝ),
    escalator_speed = 15 ∧
    (person_speed + escalator_speed) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_l2958_295878


namespace NUMINAMATH_CALUDE_josies_initial_money_l2958_295802

/-- The amount of money Josie's mom gave her initially --/
def initial_money (milk_price bread_price detergent_price banana_price_per_pound : ℚ)
  (milk_discount detergent_discount : ℚ) (banana_pounds leftover_money : ℚ) : ℚ :=
  (milk_price * (1 - milk_discount) + bread_price + 
   (detergent_price - detergent_discount) + 
   (banana_price_per_pound * banana_pounds) + leftover_money)

/-- Theorem stating that Josie's mom gave her $20.00 initially --/
theorem josies_initial_money :
  initial_money 4 3.5 10.25 0.75 0.5 1.25 2 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_josies_initial_money_l2958_295802


namespace NUMINAMATH_CALUDE_walking_speed_problem_l2958_295842

/-- Proves that given the conditions of the problem, A's walking speed is 10 kmph -/
theorem walking_speed_problem (v : ℝ) : 
  v > 0 → -- A's walking speed is positive
  v * (200 / v) = 20 * (200 / v - 10) → -- Distance equation
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l2958_295842


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_l2958_295805

/-- The total amount Mike spent on car parts -/
def total_spent : ℝ := 224.87

/-- The amount Mike spent on speakers -/
def speakers_cost : ℝ := 118.54

/-- The amount Mike spent on new tires -/
def tires_cost : ℝ := 106.33

/-- Theorem stating that the total amount spent is the sum of speakers and tires costs -/
theorem total_spent_equals_sum : total_spent = speakers_cost + tires_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_l2958_295805


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l2958_295867

/-- The number of boys in the lineup -/
def num_boys : ℕ := 4

/-- The number of girls in the lineup -/
def num_girls : ℕ := 3

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of arrangements when Boy A must stand at either end -/
def arrangements_boy_a_at_end : ℕ := 1440

/-- The number of arrangements when Girl B cannot stand to the left of Girl C -/
def arrangements_girl_b_not_left_of_c : ℕ := 2520

/-- The number of arrangements when Girl B does not stand at either end, and Girl C does not stand in the middle -/
def arrangements_girl_b_not_end_c_not_middle : ℕ := 3120

theorem photo_lineup_arrangements :
  (arrangements_boy_a_at_end = 1440) ∧
  (arrangements_girl_b_not_left_of_c = 2520) ∧
  (arrangements_girl_b_not_end_c_not_middle = 3120) := by
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l2958_295867


namespace NUMINAMATH_CALUDE_translation_of_quadratic_l2958_295812

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := -2 * x^2

/-- The translated quadratic function -/
def f (x : ℝ) : ℝ := -2 * x^2 - 12 * x - 16

/-- The vertex of the original function g -/
def vertex_g : ℝ × ℝ := (0, 0)

/-- The vertex of the translated function f -/
def vertex_f : ℝ × ℝ := (-3, 2)

/-- Theorem stating that f is the translation of g -/
theorem translation_of_quadratic :
  ∀ x : ℝ, f x = g (x + 3) + 2 :=
sorry

end NUMINAMATH_CALUDE_translation_of_quadratic_l2958_295812


namespace NUMINAMATH_CALUDE_largest_odd_in_sum_not_exceeding_200_l2958_295813

/-- The sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The nth odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2*n - 1

theorem largest_odd_in_sum_not_exceeding_200 :
  ∃ n : ℕ, sumOddNumbers n ≤ 200 ∧ 
           sumOddNumbers (n + 1) > 200 ∧ 
           nthOddNumber n = 27 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_in_sum_not_exceeding_200_l2958_295813


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2958_295809

theorem unique_solution_for_equation : ∃! p n k : ℕ+, 
  Nat.Prime p ∧ 
  k > 1 ∧ 
  (3 : ℕ)^(p : ℕ) + (4 : ℕ)^(p : ℕ) = (n : ℕ)^(k : ℕ) ∧ 
  p = 2 ∧ n = 5 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2958_295809


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l2958_295894

theorem cylinder_radius_problem (rounds1 rounds2 : ℕ) (radius2 : ℝ) (radius1 : ℝ) :
  rounds1 = 70 →
  rounds2 = 49 →
  radius2 = 20 →
  rounds1 * (2 * Real.pi * radius1) = rounds2 * (2 * Real.pi * radius2) →
  radius1 = 14 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l2958_295894


namespace NUMINAMATH_CALUDE_ratio_problem_l2958_295808

theorem ratio_problem (a b c : ℚ) (h1 : b/a = 4) (h2 : c/b = 5) : 
  (a + 2*b) / (3*b + c) = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2958_295808


namespace NUMINAMATH_CALUDE_point_coordinates_l2958_295839

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    SecondQuadrant p →
    DistToXAxis p = 3 →
    DistToYAxis p = 7 →
    p.x = -7 ∧ p.y = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l2958_295839


namespace NUMINAMATH_CALUDE_bill_meets_dexter_at_12_50_l2958_295810

/-- Represents a person or dog in the problem -/
structure Participant where
  speed : ℝ
  startTime : ℝ

/-- Calculates the time when Bill meets Dexter -/
def meetingTime (anna bill dexter : Participant) : ℝ :=
  sorry

/-- Theorem stating that Bill meets Dexter at 12:50 pm -/
theorem bill_meets_dexter_at_12_50 :
  let anna : Participant := { speed := 4, startTime := 0 }
  let bill : Participant := { speed := 3, startTime := 0 }
  let dexter : Participant := { speed := 6, startTime := 0.25 }
  meetingTime anna bill dexter = 0.8333333333 := by
  sorry

end NUMINAMATH_CALUDE_bill_meets_dexter_at_12_50_l2958_295810


namespace NUMINAMATH_CALUDE_number_division_theorem_l2958_295868

theorem number_division_theorem (x : ℚ) : 
  x / 6 = 1 / 10 → x / (3 / 25) = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_division_theorem_l2958_295868


namespace NUMINAMATH_CALUDE_binomial_expansion_result_l2958_295883

theorem binomial_expansion_result (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_result_l2958_295883


namespace NUMINAMATH_CALUDE_other_intersection_point_l2958_295836

def f (x k : ℝ) : ℝ := 3 * (x - 4)^2 + k

theorem other_intersection_point (k : ℝ) :
  f 2 k = 0 → f 6 k = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_intersection_point_l2958_295836


namespace NUMINAMATH_CALUDE_stationary_tank_radius_l2958_295877

theorem stationary_tank_radius 
  (h : Real) 
  (r : Real) 
  (h_truck : Real) 
  (r_truck : Real) 
  (h_drop : Real) :
  h = 25 → 
  r_truck = 4 → 
  h_truck = 10 → 
  h_drop = 0.016 → 
  π * r^2 * h_drop = π * r_truck^2 * h_truck → 
  r = 100 := by
sorry

end NUMINAMATH_CALUDE_stationary_tank_radius_l2958_295877


namespace NUMINAMATH_CALUDE_total_lives_after_joining_l2958_295896

theorem total_lives_after_joining (initial_players : Nat) (joined_players : Nat) (lives_per_player : Nat) : 
  initial_players = 8 → joined_players = 2 → lives_per_player = 6 → 
  (initial_players + joined_players) * lives_per_player = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_after_joining_l2958_295896


namespace NUMINAMATH_CALUDE_system_solution_equivalence_l2958_295880

-- Define the system of linear inequalities
def system (x : ℝ) : Prop := (x - 2 > 1) ∧ (x < 4)

-- Define the solution set
def solution_set : Set ℝ := {x | 3 < x ∧ x < 4}

-- Theorem statement
theorem system_solution_equivalence :
  {x : ℝ | system x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_equivalence_l2958_295880


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2958_295891

theorem polynomial_evaluation : 
  let a : ℤ := 2999
  let b : ℤ := 3000
  b^3 - a*b^2 - a^2*b + a^3 = b + a := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2958_295891


namespace NUMINAMATH_CALUDE_fruits_given_to_jane_l2958_295884

def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def initial_apples : ℕ := 21
def fruits_left : ℕ := 15

def total_initial_fruits : ℕ := initial_plums + initial_guavas + initial_apples

theorem fruits_given_to_jane : 
  total_initial_fruits - fruits_left = 40 := by sorry

end NUMINAMATH_CALUDE_fruits_given_to_jane_l2958_295884


namespace NUMINAMATH_CALUDE_prime_quadratic_residue_equivalence_l2958_295807

theorem prime_quadratic_residue_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ α : ℕ+, p ∣ α * (α - 1) + 3) ↔ (∃ β : ℕ+, p ∣ β * (β - 1) + 25) := by
  sorry

end NUMINAMATH_CALUDE_prime_quadratic_residue_equivalence_l2958_295807


namespace NUMINAMATH_CALUDE_modulus_v_is_five_l2958_295838

/-- Given two complex numbers u and v, prove that |v| = 5 when uv = 15 - 20i and |u| = 5 -/
theorem modulus_v_is_five (u v : ℂ) (h1 : u * v = 15 - 20 * I) (h2 : Complex.abs u = 5) : 
  Complex.abs v = 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_v_is_five_l2958_295838


namespace NUMINAMATH_CALUDE_monkey_banana_distribution_l2958_295866

/-- Calculates the number of bananas each monkey receives when a family of monkeys divides a collection of bananas equally -/
def bananas_per_monkey (num_monkeys : ℕ) (num_piles_type1 : ℕ) (hands_per_pile_type1 : ℕ) (bananas_per_hand_type1 : ℕ)
                       (num_piles_type2 : ℕ) (hands_per_pile_type2 : ℕ) (bananas_per_hand_type2 : ℕ) : ℕ :=
  let total_bananas := num_piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
                       num_piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2
  total_bananas / num_monkeys

/-- Theorem stating that under the given conditions, each monkey receives 99 bananas -/
theorem monkey_banana_distribution :
  bananas_per_monkey 12 6 9 14 4 12 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_monkey_banana_distribution_l2958_295866


namespace NUMINAMATH_CALUDE_spice_combinations_l2958_295862

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem spice_combinations : choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_spice_combinations_l2958_295862


namespace NUMINAMATH_CALUDE_radius_q3_is_one_point_five_l2958_295857

/-- A triangle with an inscribed circle and two additional tangent circles -/
structure TripleCircleTriangle where
  /-- Side length AB of the triangle -/
  ab : ℝ
  /-- Side length BC of the triangle -/
  bc : ℝ
  /-- Side length AC of the triangle -/
  ac : ℝ
  /-- Radius of the inscribed circle Q1 -/
  r1 : ℝ
  /-- Radius of circle Q2, tangent to Q1 and sides AB and BC -/
  r2 : ℝ
  /-- Radius of circle Q3, tangent to Q2 and sides AB and BC -/
  r3 : ℝ
  /-- AB equals BC -/
  ab_eq_bc : ab = bc
  /-- AB equals 80 -/
  ab_eq_80 : ab = 80
  /-- AC equals 96 -/
  ac_eq_96 : ac = 96
  /-- Q1 is inscribed in the triangle -/
  q1_inscribed : r1 = (ab + bc + ac) / 2 - ab
  /-- Q2 is tangent to Q1 and sides AB and BC -/
  q2_tangent : r2 = r1 / 4
  /-- Q3 is tangent to Q2 and sides AB and BC -/
  q3_tangent : r3 = r2 / 4

/-- The radius of Q3 is 1.5 in the given triangle configuration -/
theorem radius_q3_is_one_point_five (t : TripleCircleTriangle) : t.r3 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_radius_q3_is_one_point_five_l2958_295857


namespace NUMINAMATH_CALUDE_triangle_count_theorem_l2958_295816

/-- Represents a rectangle divided into columns and rows with diagonal lines -/
structure DividedRectangle where
  columns : Nat
  rows : Nat

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (rect : DividedRectangle) : Nat :=
  let smallest_triangles := rect.columns * rect.rows * 2
  let small_isosceles := rect.columns + rect.rows * 2
  let medium_right := (rect.columns / 2) * rect.rows * 2
  let large_isosceles := rect.columns / 2
  smallest_triangles + small_isosceles + medium_right + large_isosceles

/-- The main theorem stating the number of triangles in the specific rectangle -/
theorem triangle_count_theorem (rect : DividedRectangle) 
    (h_columns : rect.columns = 8) 
    (h_rows : rect.rows = 2) : 
  count_triangles rect = 76 := by
  sorry

#eval count_triangles ⟨8, 2⟩

end NUMINAMATH_CALUDE_triangle_count_theorem_l2958_295816


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2958_295851

/-- Given a hyperbola and a circle satisfying certain conditions, prove that the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  ((c - a)^2 = c^2 / 16) →  -- Circle passes through right focus F(c, 0)
  (∃ k : ℝ, ∀ x y : ℝ, 
    (x - a)^2 + y^2 = c^2 / 16 →  -- Circle equation
    (y = k * x ∨ y = -k * x) →  -- Asymptote equations
    ∃ m : ℝ, m * k = -1 ∧ 
      ∃ x₀ y₀ : ℝ, (x₀ - a)^2 + y₀^2 = c^2 / 16 ∧ 
        y₀ - 0 = m * (x₀ - c)) →  -- Tangent line perpendicular to asymptote
  c / a = 2  -- Eccentricity is 2
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2958_295851


namespace NUMINAMATH_CALUDE_desmond_bought_240_toys_l2958_295800

/-- The number of toys Mr. Desmond bought for his elder son -/
def elder_son_toys : ℕ := 60

/-- The number of toys Mr. Desmond bought for his younger son -/
def younger_son_toys : ℕ := 3 * elder_son_toys

/-- The total number of toys Mr. Desmond bought -/
def total_toys : ℕ := elder_son_toys + younger_son_toys

theorem desmond_bought_240_toys : total_toys = 240 := by
  sorry

end NUMINAMATH_CALUDE_desmond_bought_240_toys_l2958_295800


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l2958_295822

theorem absolute_sum_zero_implies_sum (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l2958_295822


namespace NUMINAMATH_CALUDE_last_hour_probability_l2958_295885

/-- The number of attractions available -/
def num_attractions : ℕ := 6

/-- The number of attractions each person chooses -/
def num_chosen : ℕ := 4

/-- The probability of two people being at the same attraction during their last hour -/
def same_attraction_probability : ℚ := 1 / 6

theorem last_hour_probability :
  (num_attractions : ℚ) / (num_attractions * num_attractions) = same_attraction_probability :=
sorry

end NUMINAMATH_CALUDE_last_hour_probability_l2958_295885


namespace NUMINAMATH_CALUDE_inequalities_given_m_gt_neg_one_l2958_295850

theorem inequalities_given_m_gt_neg_one (m : ℝ) (h : m > -1) :
  (4*m > -4) ∧ (-5*m < -5) ∧ (m+1 > 0) ∧ (1-m < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_m_gt_neg_one_l2958_295850


namespace NUMINAMATH_CALUDE_initial_players_count_video_game_players_l2958_295861

theorem initial_players_count (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let remaining_players := total_lives / lives_per_player
  remaining_players + players_quit

theorem video_game_players : initial_players_count 7 8 24 = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_players_count_video_game_players_l2958_295861


namespace NUMINAMATH_CALUDE_hexagon_arrangement_count_l2958_295819

/-- Represents a valid arrangement of digits on a regular hexagon with center -/
structure HexagonArrangement where
  vertices : Fin 6 → Fin 7
  center : Fin 7
  all_different : ∀ i j : Fin 6, i ≠ j → vertices i ≠ vertices j
  center_different : ∀ i : Fin 6, center ≠ vertices i
  sum_equal : ∀ i : Fin 3, 
    (vertices i).val + center.val + (vertices (i + 3)).val = 
    (vertices (i + 1)).val + center.val + (vertices (i + 4)).val

/-- The number of valid hexagon arrangements -/
def count_arrangements : ℕ := sorry

/-- Theorem stating the correct number of arrangements -/
theorem hexagon_arrangement_count : count_arrangements = 144 := by sorry

end NUMINAMATH_CALUDE_hexagon_arrangement_count_l2958_295819


namespace NUMINAMATH_CALUDE_school_supplies_cost_l2958_295869

/-- The cost of all pencils and pens given their individual prices and quantities -/
def total_cost (pencil_price pen_price : ℚ) (num_pencils num_pens : ℕ) : ℚ :=
  pencil_price * num_pencils + pen_price * num_pens

/-- Theorem stating the total cost of 38 pencils at $2.50 each and 56 pens at $3.50 each is $291.00 -/
theorem school_supplies_cost :
  total_cost (5/2) (7/2) 38 56 = 291 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l2958_295869


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2958_295887

theorem cube_root_simplification :
  (40^3 + 50^3 + 60^3 : ℝ)^(1/3) = 10 * 405^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2958_295887


namespace NUMINAMATH_CALUDE_position_of_2000_l2958_295847

/-- Represents the column number (1 to 5) in the table -/
inductive Column
| one
| two
| three
| four
| five

/-- Represents a position in the table -/
structure Position where
  row : Nat
  column : Column

/-- Function to determine the position of a given even number in the table -/
def positionOfEvenNumber (n : Nat) : Position :=
  sorry

/-- The arrangement of positive even numbers follows the pattern described in the problem -/
axiom arrangement_pattern : ∀ n : Nat, n % 2 = 0 → n > 0 → 
  (positionOfEvenNumber n).column = Column.one ↔ n % 8 = 0

/-- Theorem stating that 2000 is in Row 250, Column 1 -/
theorem position_of_2000 : positionOfEvenNumber 2000 = { row := 250, column := Column.one } :=
  sorry

end NUMINAMATH_CALUDE_position_of_2000_l2958_295847


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l2958_295820

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (10 * dozen) + (dozen / 2)

/-- The total number of students in the class -/
def total_students : ℕ := 48

/-- The number of teachers -/
def teachers : ℕ := 2

/-- The number of teacher's aids -/
def teacher_aids : ℕ := 2

/-- The number of absent students -/
def absent_students : ℕ := 6

/-- The number of students on a field trip -/
def field_trip_students : ℕ := 8

/-- The number of people present in the class -/
def people_present : ℕ := total_students - absent_students - field_trip_students + teachers + teacher_aids

/-- The number of cupcakes left after distribution -/
def cupcakes_left : ℕ := cupcakes_brought - people_present

theorem cupcakes_remaining :
  cupcakes_left = 85 :=
sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l2958_295820


namespace NUMINAMATH_CALUDE_seven_mondays_in_45_days_l2958_295814

/-- The number of Mondays in the first 45 days of a year that starts on a Monday. -/
def mondaysIn45Days (yearStartsOnMonday : Bool) : ℕ :=
  if yearStartsOnMonday then 7 else 0

/-- Theorem stating that if a year starts on a Monday, there are 7 Mondays in the first 45 days. -/
theorem seven_mondays_in_45_days (yearStartsOnMonday : Bool) 
  (h : yearStartsOnMonday = true) : mondaysIn45Days yearStartsOnMonday = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_mondays_in_45_days_l2958_295814
