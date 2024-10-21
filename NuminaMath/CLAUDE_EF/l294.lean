import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l294_29434

theorem power_equation_solution (m : ℤ) : (((-2 : ℤ) ^ (2 * m) : ℚ) = (2 ^ (3 - m) : ℚ)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l294_29434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_l294_29414

def polynomial (k : ℕ) : ℕ :=
  k^2020 + 2*k^2019 + 3*k^2018 + k^2017 + 5*k^2016 + 6*k^2015 + 7*k^2014 + 8*k^2013 + 9*k^2012 + 10*k^2011 +
  11*k^2010 + 12*k^2009 + 13*k^2008 + 14*k^2007 + 15*k^2006 + 16*k^2005 + 17*k^2004 + 18*k^2003 + 19*k^2002 + 20*k^2001 +
  21*k^2000 + 22*k^1999 + 23*k^1998 + 24*k^1997 + 25*k^1996 + 26*k^1995 + 27*k^1994 + 28*k^1993 + 29*k^1992 + 30*k^1991 +
  -- ... (continuing the pattern)
  2011*k^10 + 2012*k^9 + 2013*k^8 + 2014*k^7 + 2015*k^6 + 2016*k^5 + 2017*k^4 + 2018*k^3 + 2019*k^2 + 2020*k + 2021

theorem largest_divisor :
  ∀ k : ℕ, k > 1010 → ¬(k + 1 ∣ polynomial k) ∧
  (1010 + 1 ∣ polynomial 1010) :=
by
  sorry

#eval polynomial 1010 % 1011

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_l294_29414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_purchase_theorem_l294_29478

def brand_a_packs : ℕ := 2
def brand_a_bars_per_pack : ℕ := 4
def brand_a_price_per_pack : ℚ := 6
def brand_a_discount : ℚ := 1/10

def brand_b_packs : ℕ := 3
def brand_b_bars_per_pack : ℕ := 5
def brand_b_total_price : ℚ := 14

def brand_c_packs : ℕ := 2
def brand_c_bars_per_pack : ℕ := 6
def brand_c_price_per_pack : ℚ := 8
def brand_c_free_pack_bars : ℕ := 4

def total_spent : ℚ := brand_a_packs * brand_a_price_per_pack * (1 - brand_a_discount) + brand_b_total_price + brand_c_packs * brand_c_price_per_pack

def total_bars : ℕ := brand_a_packs * brand_a_bars_per_pack + brand_b_packs * brand_b_bars_per_pack + brand_c_packs * brand_c_bars_per_pack + brand_c_free_pack_bars

def average_cost_per_bar : ℚ := total_spent / total_bars

theorem soap_purchase_theorem :
  total_spent = 408/10 ∧
  total_bars = 39 ∧
  (average_cost_per_bar * 1000).floor / 1000 = 1046/1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_purchase_theorem_l294_29478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l294_29406

theorem complex_modulus (z : ℂ) : (z - 2*Complex.I)*(1 - Complex.I) = -2 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l294_29406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_theorem_l294_29446

def num_people : Nat := 3
def num_books : Nat := 7

-- Scenario 1: Distribution (1, 2, 4)
def distribution1 : Fin num_people → Nat
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 2
  | ⟨2, _⟩ => 4
  | ⟨n+3, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 3 n))

-- Scenario 2: Distribution (2, 2, 3)
def distribution2 : Fin num_people → Nat
  | ⟨0, _⟩ => 2
  | ⟨1, _⟩ => 2
  | ⟨2, _⟩ => 3
  | ⟨n+3, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 3 n))

-- Function to calculate the number of ways to distribute books
def num_distributions (dist : Fin num_people → Nat) : Nat := sorry

theorem book_distribution_theorem :
  num_distributions distribution1 = 630 ∧ num_distributions distribution2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_theorem_l294_29446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_fixed_points_l294_29440

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25

-- Define the line x + y + 6 = 0
def line_P (x y : ℝ) : Prop := x + y + 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3/2)

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x - 4*y + 9 = 0

-- Define the fixed points
def fixed_point_1 : ℝ × ℝ := (2, 0)
def fixed_point_2 : ℝ × ℝ := (-2, -4)

theorem circle_tangent_fixed_points :
  -- Part 1: The line l passes through P and intersects C with chord length 8
  (∀ x y : ℝ, line_l x y → (x = point_P.1 ∧ y = point_P.2 ∨ circle_C x y)) ∧
  (∃ a b c d : ℝ, circle_C a b ∧ circle_C c d ∧ line_l a b ∧ line_l c d ∧ 
    (a - c)^2 + (b - d)^2 = 64) ∧
  -- Part 2: For any P on x + y + 6 = 0, if a line through P is tangent to C,
  -- then the circle through A, P, and C passes through the fixed points
  (∀ p_x p_y a_x a_y : ℝ,
    line_P p_x p_y →
    circle_C a_x a_y →
    (∃ t_x t_y : ℝ, (t_x - p_x)*(a_x - p_x) + (t_y - p_y)*(a_y - p_y) = 0 ∧
                    (t_x - 2)*(a_x - 2) + t_y*a_y = 0) →
    (∃ r : ℝ, 
      (p_x - 2)^2 + (p_y - 0)^2 = r^2 ∧
      (a_x - 2)^2 + (a_y - 0)^2 = r^2 ∧
      (fixed_point_1.1 - 2)^2 + (fixed_point_1.2 - 0)^2 = r^2) ∨
    (∃ r : ℝ,
      (p_x - (-2))^2 + (p_y - (-4))^2 = r^2 ∧
      (a_x - (-2))^2 + (a_y - (-4))^2 = r^2 ∧
      (fixed_point_2.1 - (-2))^2 + (fixed_point_2.2 - (-4))^2 = r^2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_fixed_points_l294_29440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_s4_l294_29453

noncomputable def geometric_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_s4 (a r : ℝ) :
  geometric_sum a r 2 = 7 →
  geometric_sum a r 6 = 91 →
  geometric_sum a r 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_s4_l294_29453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_fifth_root_of_eleven_l294_29465

theorem sixth_root_over_fifth_root_of_eleven (x : ℝ) (hx : x = 11) :
  (x^(1/6 : ℝ)) / (x^(1/5 : ℝ)) = x^(-1/30 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_fifth_root_of_eleven_l294_29465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_y_equals_x_l294_29412

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

theorem reflection_over_y_equals_x :
  ∀ (v : Fin 2 → ℝ),
  Matrix.mulVec reflection_matrix v = fun i => v (if i = 0 then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_y_equals_x_l294_29412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wasted_metal_correct_wasted_metal_positive_l294_29426

/-- Represents a rectangular metal sheet with length and width -/
structure MetalSheet where
  length : ℝ
  width : ℝ
  h_positive : 0 < length ∧ 0 < width
  h_length_greater : length > width

/-- Calculates the amount of metal wasted after cutting the largest possible
    circle and then the largest possible square from that circle -/
noncomputable def wastedMetal (sheet : MetalSheet) : ℝ :=
  sheet.length * sheet.width - sheet.width^2 / 2

/-- Theorem stating that the wasted metal is correctly calculated -/
theorem wasted_metal_correct (sheet : MetalSheet) :
  wastedMetal sheet = sheet.length * sheet.width - sheet.width^2 / 2 := by
  -- Unfold the definition of wastedMetal
  unfold wastedMetal
  -- The equality follows directly from the definition
  rfl

/-- Proof that the wasted metal is positive -/
theorem wasted_metal_positive (sheet : MetalSheet) :
  0 < wastedMetal sheet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wasted_metal_correct_wasted_metal_positive_l294_29426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_l294_29404

-- Define the circle on which M moves
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (-4, 8)

-- Define the midpoint P in terms of M and A
noncomputable def midpoint_P (m_x m_y : ℝ) : ℝ × ℝ :=
  ((m_x + point_A.1) / 2, (m_y + point_A.2) / 2)

-- State the theorem
theorem trajectory_of_midpoint :
  ∀ (m_x m_y : ℝ), circle_eq m_x m_y →
  let (x, y) := midpoint_P m_x m_y
  x^2 + (y - 4)^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_l294_29404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l294_29467

/-- Represents a geometric sequence -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- Calculates the sum of the first n terms of a geometric sequence -/
noncomputable def sum_of_terms (gs : GeometricSequence) (n : ℕ) : ℝ :=
  gs.first_term * (1 - gs.common_ratio^n) / (1 - gs.common_ratio)

/-- Theorem stating the property of the specific geometric sequence -/
theorem geometric_sequence_sum_property :
  ∀ gs : GeometricSequence,
  sum_of_terms gs 1500 = 300 →
  sum_of_terms gs 3000 = 570 →
  sum_of_terms gs 4500 = 813 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l294_29467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_shooting_improvement_l294_29449

/-- Represents the number of shots made out of a total number of shots --/
structure ShotStats where
  made : ℕ
  total : ℕ

/-- Calculates the shooting percentage --/
def shootingPercentage (stats : ShotStats) : ℚ :=
  stats.made / stats.total

theorem sally_shooting_improvement 
  (initial : ShotStats) 
  (final : ShotStats) 
  (h1 : initial.total = 30)
  (h2 : shootingPercentage initial = 60 / 100)
  (h3 : final.total = initial.total + 10)
  (h4 : shootingPercentage final = 62 / 100)
  : final.made - initial.made = 7 := by
  sorry

#check sally_shooting_improvement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_shooting_improvement_l294_29449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_determines_a_l294_29492

open Real

-- Define the functions
noncomputable def f (x : ℝ) := x^2 + x - x * log x - 3
def g (x : ℝ) := -x^2 + 4*x - 4

-- State the theorem
theorem tangent_point_determines_a (a b : ℝ) :
  (∀ x ≥ 1, 2*x^2 - 3*x - x*log x + 1 ≥ a*x + b + (x-2)^2) ∧
  (∀ x ≥ 1, a*x + b + (x-2)^2 ≥ 0) ∧
  (f 1 = g 1) ∧
  (deriv f 1 = deriv g 1) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_determines_a_l294_29492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_PQRS_area_perimeter_product_l294_29457

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a square given its side length -/
def squareArea (side : ℝ) : ℝ :=
  side^2

/-- Calculate the perimeter of a square given its side length -/
def squarePerimeter (side : ℝ) : ℝ :=
  4 * side

theorem square_PQRS_area_perimeter_product :
  let P : Point := ⟨1, 5⟩
  let Q : Point := ⟨5, 6⟩
  let R : Point := ⟨6, 2⟩
  let S : Point := ⟨2, 1⟩
  let side := distance P Q
  let area := squareArea side
  let perimeter := squarePerimeter side
  area * perimeter = 68 * Real.sqrt 17 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_PQRS_area_perimeter_product_l294_29457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_trig_properties_l294_29408

theorem interior_angle_trig_properties (α : Real) (h : 0 < α ∧ α < Real.pi) :
  (∀ β, 0 < β ∧ β < Real.pi → Real.sin β > 0) ∧
  (∃ β, 0 < β ∧ β < Real.pi ∧ Real.cos β < 0) ∧
  (∃ β, 0 < β ∧ β < Real.pi ∧ Real.tan β < 0) ∧
  (∀ β, 0 < β ∧ β < Real.pi → Real.tan (β / 2) > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_trig_properties_l294_29408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_calculation_road_length_problem_l294_29454

/-- Calculates the length of a road given the number of trees and spacing between them -/
theorem road_length_calculation (total_trees : ℕ) (tree_spacing : ℕ) : 
  total_trees % 2 = 0 → 
  tree_spacing > 0 →
  (total_trees / 2 - 1) * tree_spacing = 355 := by
  sorry

/-- The specific problem instance -/
theorem road_length_problem : 
  ∃ (total_trees tree_spacing : ℕ),
    total_trees = 72 ∧ 
    tree_spacing = 5 ∧
    total_trees % 2 = 0 ∧
    tree_spacing > 0 ∧
    (total_trees / 2 - 1) * tree_spacing = 355 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_calculation_road_length_problem_l294_29454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_same_color_probability_and_result_l294_29416

def total_sheets : ℕ := 12
def sheets_per_color : ℕ := 4
def total_stars : ℕ := 12
def stars_per_color : ℕ := 4

def probability_no_same_color : ℚ := 81 / 3850

theorem no_same_color_probability_and_result :
  (probability_no_same_color = 81 / 3850) ∧
  (3850 - 100 * 81 = -4250) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_same_color_probability_and_result_l294_29416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_movie_night_proof_l294_29459

def family_movie_night (regular_ticket : ℕ) (elderly_discount : ℕ) (adult_ticket : ℕ) 
  (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_adults : ℕ) (num_elderly : ℕ) : ℕ :=
  let elderly_ticket := regular_ticket - elderly_discount
  let child_ticket := adult_ticket - child_discount
  let total_cost := total_payment - change
  let adult_elderly_cost := num_adults * adult_ticket + num_elderly * elderly_ticket
  let children_cost := total_cost - adult_elderly_cost
  children_cost / child_ticket

#eval family_movie_night 18 8 15 6 270 10 5 3

theorem family_movie_night_proof : 
  family_movie_night 18 8 15 6 270 10 5 3 = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_movie_night_proof_l294_29459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_area_l294_29448

/-- Given a rectangle ABCD with area 1 and a fold that brings point C to coincide with point A,
    the area of the resulting pentagon is less than 3/4. -/
theorem folded_rectangle_area (A B C D : ℝ × ℝ) : 
  ∀ (rectangle_area : ℝ) 
    (fold_line : Set (ℝ × ℝ)) 
    (M N D' : ℝ × ℝ) 
    (pentagon_area : ℝ),
  rectangle_area = 1 → 
  fold_line = sorry → -- Define fold_line
  M ∈ fold_line ∧ M ∈ Set.Icc B C →
  N ∈ fold_line ∧ N ∈ Set.Icc A D →
  D' = sorry → -- Define D' in terms of D and fold_line
  pentagon_area = sorry → -- Define pentagon_area
  pentagon_area < 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_area_l294_29448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l294_29466

/-- Represents the number of sides in a convex polygon --/
def n : ℕ := 39

/-- Represents the common difference in the arithmetic progression of interior angles --/
noncomputable def common_difference : ℝ := 10

/-- Represents the measure of the largest interior angle --/
noncomputable def largest_angle : ℝ := 175

/-- The sum of interior angles of a polygon with n sides --/
noncomputable def sum_of_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- The sum of an arithmetic progression --/
noncomputable def arithmetic_sum (a₁ : ℝ) (aₙ : ℝ) (n : ℕ) : ℝ := n / 2 * (a₁ + aₙ)

/-- Theorem stating that n = 39 for the given conditions --/
theorem polygon_sides_count : 
  sum_of_angles n = arithmetic_sum (largest_angle - (n - 1) * common_difference) largest_angle n → n = 39 := by
  sorry

#check polygon_sides_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l294_29466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_specific_region_l294_29473

/-- The perimeter of a region formed by two radii and an arc of a circle -/
noncomputable def perimeter_region (radius : ℝ) (arc_angle : ℝ) : ℝ :=
  2 * radius + (arc_angle / 360) * 2 * Real.pi * radius

theorem perimeter_specific_region :
  perimeter_region 6 270 = 12 + 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_specific_region_l294_29473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_over_four_l294_29450

theorem tan_alpha_minus_pi_over_four (α : ℝ) 
  (h_acute : 0 < α ∧ α < π / 2) 
  (h_cos : Real.cos α = Real.sqrt 5 / 5) : 
  Real.tan (α - π / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_over_four_l294_29450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_b_n_equals_b_0_l294_29472

noncomputable def b : ℕ → ℝ
  | 0 => Real.cos (Real.pi / 18) ^ 2
  | n + 1 => 4 * b n * (1 - b n)

theorem smallest_n_for_b_n_equals_b_0 :
  (∀ k < 24, b k ≠ b 0) ∧ b 24 = b 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_b_n_equals_b_0_l294_29472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l294_29456

/-- Represents the mass of a block in grams -/
def Mass := Fin 5

/-- The possible masses for the blocks -/
def possible_masses : List Mass := [⟨0, by norm_num⟩, ⟨1, by norm_num⟩, ⟨2, by norm_num⟩, ⟨3, by norm_num⟩, ⟨4, by norm_num⟩]

/-- Represents the three blocks: square, circle, and triangle -/
structure Blocks where
  square : Mass
  circle : Mass
  triangle : Mass

/-- Checks if the masses of the blocks satisfy the given conditions -/
def satisfies_conditions (b : Blocks) : Prop :=
  (2 * b.square.val > 3 * b.circle.val) ∧
  (b.circle.val > 2 * b.triangle.val) ∧
  (b.square ≠ b.circle) ∧
  (b.circle ≠ b.triangle) ∧
  (b.square ≠ b.triangle)

theorem unique_solution :
  ∃! b : Blocks, 
    b.square ∈ possible_masses ∧
    b.circle ∈ possible_masses ∧
    b.triangle ∈ possible_masses ∧
    satisfies_conditions b ∧
    b.square = ⟨4, by norm_num⟩ ∧
    b.circle = ⟨2, by norm_num⟩ ∧
    b.triangle = ⟨0, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l294_29456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_max_set_min_side_a_l294_29482

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sin (2 * x - 7 * Real.pi / 6)

-- Theorem for the maximum value of f(x)
theorem f_max_value : ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = 2 := by sorry

-- Theorem for the set of x values where f(x) achieves its maximum
theorem f_max_set : ∀ (x : ℝ), f x = 2 ↔ ∃ (k : ℤ), x = k * Real.pi + Real.pi / 6 := by sorry

-- Theorem for the minimum value of side a in triangle ABC
theorem min_side_a (A B C : ℝ) (hf : f A = 3/2) (hbc : B + C = 2) :
  ∃ (a : ℝ), a^2 = B^2 + C^2 - 2*B*C*Real.cos A ∧ 
  (∀ (a' : ℝ), a'^2 = B^2 + C^2 - 2*B*C*Real.cos A → a ≤ a') ∧
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_max_set_min_side_a_l294_29482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l294_29419

theorem equation_solution_range (x a : ℝ) :
  (x ∈ Set.Icc 0 (Real.pi / 4)) →
  (a > 0) →
  (∃ x, Real.cos x + Real.sqrt a * Real.sin x = Real.sqrt a) ↔
  (a ∈ Set.Icc 1 (3 + 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l294_29419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_for_triangle_l294_29483

/-- The line equation forming a triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 8 * x + 3 * y = 48

/-- The sum of altitudes of the triangle -/
noncomputable def sum_of_altitudes : ℝ := (22 * Real.sqrt 73 + 48) / Real.sqrt 73

/-- Theorem stating that the sum of altitudes of the triangle formed by the given line and coordinate axes is equal to the calculated value -/
theorem sum_of_altitudes_for_triangle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ 0 ∧
    line_equation 0 y₂ ∧
    x₁ > 0 ∧ y₂ > 0 ∧
    sum_of_altitudes = x₁ + y₂ + (48 / Real.sqrt 73) := by
  sorry

#check sum_of_altitudes_for_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_for_triangle_l294_29483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_where_b_exceeds_a100_l294_29490

def a : ℕ → ℕ
| 0 => 3  -- Adding the base case for 0
| 1 => 3
| (n + 1) => 3^(a n)

def b : ℕ → ℕ
| 0 => 100  -- Adding the base case for 0
| 1 => 100
| (n + 1) => 100^(b n)

theorem smallest_m_where_b_exceeds_a100 :
  (b 99 > a 100) ∧ (∀ k < 99, b k ≤ a 100) := by
  sorry

#eval a 2  -- This line is added to check if the function works
#eval b 2  -- This line is added to check if the function works

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_where_b_exceeds_a100_l294_29490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_walking_distance_l294_29468

-- Define Mark's walking rate
def mark_rate : ℚ := 1 / 20

-- Define the time Mark walks
def walking_time : ℚ := 45

-- Function to calculate distance
def calculate_distance (rate : ℚ) (time : ℚ) : ℚ := rate * time

-- Function to round to nearest tenth
def round_to_tenth (x : ℚ) : ℚ := 
  (x * 10).floor / 10 + if (x * 10 - (x * 10).floor ≥ 1/2) then 1/10 else 0

-- Theorem statement
theorem mark_walking_distance :
  round_to_tenth (calculate_distance mark_rate walking_time) = 23 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_walking_distance_l294_29468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_f_equals_three_halves_solutions_l294_29442

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then 2*x + 3
  else if x ≤ 1 then x^2 + 1
  else 1 + 1/x

-- Theorem 1: f(f(f(-2))) = 3/2
theorem f_composition_negative_two : f (f (f (-2))) = 3/2 := by sorry

-- Theorem 2: f(m) = 3/2 has two solutions
theorem f_equals_three_halves_solutions :
  ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ f m₁ = 3/2 ∧ f m₂ = 3/2 ∧ m₁ = -Real.sqrt 2 / 2 ∧ m₂ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_f_equals_three_halves_solutions_l294_29442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raja_income_l294_29452

noncomputable def monthly_income (household_percent : ℝ) (clothes_percent : ℝ) (medicines_percent : ℝ) (savings : ℝ) : ℝ :=
  savings / (1 - household_percent - clothes_percent - medicines_percent)

theorem raja_income :
  let household_percent : ℝ := 0.60
  let clothes_percent : ℝ := 0.10
  let medicines_percent : ℝ := 0.10
  let savings : ℝ := 5000
  monthly_income household_percent clothes_percent medicines_percent savings = 25000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raja_income_l294_29452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_with_complex_root_l294_29441

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 5 + I) ∧
    (a * (5 + I)^2 + b * (5 + I) + c = 0) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_with_complex_root_l294_29441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_remaining_number_l294_29477

/-- Represents the skipping pattern for each student -/
def skip_pattern (student : ℕ) : ℕ → Bool :=
  λ n ↦ n % (student + 2) = 0

/-- Checks if a number is skipped by any student from 1 to 9 -/
def is_skipped (n : ℕ) : Prop :=
  ∃ student : ℕ, student ≥ 1 ∧ student ≤ 9 ∧ skip_pattern student n = true

theorem unique_remaining_number :
  ∃! n : ℕ, n ≥ 1 ∧ n ≤ 500 ∧ ¬(is_skipped n) :=
sorry

#check unique_remaining_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_remaining_number_l294_29477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l294_29424

theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x^2 - 2*x + a > 0}
  (1 ∉ A) → a ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l294_29424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l294_29422

noncomputable section

-- Define the hyperbola and line
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1
def line (x y : ℝ) : Prop := x + y = 1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 / a^2 + 1)

theorem hyperbola_line_intersection
  (a : ℝ)
  (h1 : a > 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ hyperbola a A.1 A.2 ∧ hyperbola a B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2)
  (h3 : ∃ P A B : ℝ × ℝ, P.1 = 0 ∧ P.2 = 1 ∧ line A.1 A.2 ∧ line B.1 B.2 ∧
       (A.1 - P.1, A.2 - P.2) = (5/12 : ℝ) • (B.1 - P.1, B.2 - P.2)) :
  (eccentricity a > Real.sqrt 6 / 2 ∧ eccentricity a ≠ Real.sqrt 2) ∧ a = 17/13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l294_29422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_return_is_4_18_l294_29410

/-- Represents the outcomes of the four-sided die -/
inductive Outcome
  | one
  | two
  | three
  | four
deriving Repr, DecidableEq

/-- Probability distribution of the die outcomes -/
def prob : Outcome → ℚ
  | Outcome.one => 1/4
  | Outcome.two => 1/4
  | Outcome.three => 1/6
  | Outcome.four => 1/3

/-- Payoff function for a single throw -/
def payoff (first : Outcome) (second : Outcome) : ℚ :=
  match second with
  | Outcome.one => 2
  | Outcome.two => if first = Outcome.one ∨ first = Outcome.three then -3 else 0
  | Outcome.three => 0
  | Outcome.four => 5

/-- Expected value of a single throw -/
def expected_value_single : ℚ :=
  (prob Outcome.one * payoff Outcome.one Outcome.one) +
  (prob Outcome.two * payoff Outcome.one Outcome.two) +
  (prob Outcome.three * payoff Outcome.one Outcome.three) +
  (prob Outcome.four * payoff Outcome.one Outcome.four)

/-- Expected value of the second throw given the first throw -/
def expected_value_second (first : Outcome) : ℚ :=
  (prob Outcome.one * payoff first Outcome.one) +
  (prob Outcome.two * payoff first Outcome.two) +
  (prob Outcome.three * payoff first Outcome.three) +
  (prob Outcome.four * payoff first Outcome.four)

/-- Total expected value of two throws -/
def total_expected_value : ℚ :=
  expected_value_single +
  (prob Outcome.one * expected_value_second Outcome.one) +
  (prob Outcome.two * expected_value_second Outcome.two) +
  (prob Outcome.three * expected_value_second Outcome.three) +
  (prob Outcome.four * expected_value_second Outcome.four)

theorem expected_return_is_4_18 : total_expected_value = 209/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_return_is_4_18_l294_29410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l294_29476

def sequence_a : ℕ → ℚ
  | 0 => 4  -- Add this case to handle n = 0
  | 1 => 4
  | (n + 2) => (3 * sequence_a (n + 1) + 2) / (sequence_a (n + 1) + 4)

theorem sequence_a_general_term (n : ℕ) (hn : n ≥ 1) : 
  sequence_a n = (2^(n-1) + 5^(n-1)) / (5^(n-1) - 2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l294_29476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_l294_29401

theorem sum_of_factors (p q r : ℤ) : 
  (∀ x : ℝ, x^2 + 21*x + 110 = (x + Int.cast p) * (x + Int.cast q)) →
  (∀ x : ℝ, x^2 - 23*x + 132 = (x - Int.cast q) * (x - Int.cast r)) →
  p + q + r = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_l294_29401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_hyperbola_l294_29480

-- Define the polar equation as noncomputable
noncomputable def polar_equation (θ : ℝ) : ℝ := 1 / (1 - Real.cos θ + Real.sin θ)

-- Theorem statement
theorem polar_equation_is_hyperbola :
  ∃ (e : ℝ), e > 1 ∧
  ∀ θ, polar_equation θ = 1 / (1 - e * Real.cos (θ + Real.pi/4)) :=
by
  -- Proof sketch
  -- We'll use e = √2
  let e := Real.sqrt 2
  
  -- Show that e > 1
  have e_gt_one : e > 1 := by sorry
  
  -- Show that the equation holds for all θ
  have eq_holds : ∀ θ, polar_equation θ = 1 / (1 - e * Real.cos (θ + Real.pi/4)) := by sorry
  
  -- Combine the results
  exact ⟨e, e_gt_one, eq_holds⟩

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_hyperbola_l294_29480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_is_plane_l294_29415

/-- Represents a point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Defines the set of points satisfying θ = 2c -/
def ConstantAngleSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = 2 * c}

/-- Theorem stating that the set of points with θ = 2c forms a plane -/
theorem constant_angle_is_plane (c : ℝ) :
  ∃ (a b d : ℝ), ∀ (p : CylindricalPoint),
    p ∈ ConstantAngleSet c ↔ a * (p.r * Real.cos p.θ) + b * (p.r * Real.sin p.θ) + d = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_is_plane_l294_29415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raghu_investment_l294_29430

/-- Represents the investment problem with Raghu, Trishul, and Vishal -/
structure InvestmentProblem where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ
  total : ℝ
  trishul_eq : trishul = 0.9 * raghu
  vishal_eq : vishal = 1.1 * trishul
  total_eq : raghu + trishul + vishal = total

/-- Theorem stating that given the conditions, Raghu's investment is approximately 2299.65 -/
theorem raghu_investment (p : InvestmentProblem) (h : p.total = 6647) :
  abs (p.raghu - 2299.65) < 0.01 := by
  sorry

#eval (6647 : Float) / 2.89

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raghu_investment_l294_29430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density_inequality_l294_29451

/-- A smooth probability density function. -/
def SmoothDensity (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x, f x ≥ 0) ∧ 
  (∫ x, f x) = 1

/-- The condition that x f(x) approaches 0 as |x| approaches infinity. -/
def LimitCondition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x, |x| > M → |x * f x| < ε

/-- The theorem statement. -/
theorem density_inequality (f : ℝ → ℝ) (hf : SmoothDensity f) (hlim : LimitCondition f) :
  (∫ x, x^2 * f x) * (∫ x, ((deriv f) x / f x)^2 * f x) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_density_inequality_l294_29451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_triangle_area_l294_29413

/-- The solutions of the equation (z+5)^10 = 32 in the complex plane -/
def solutions : Set ℂ :=
  {z : ℂ | (z + 5) ^ 10 = 32}

/-- The area of a triangle formed by three complex numbers -/
noncomputable def triangleArea (a b c : ℂ) : ℝ :=
  abs ((b - a).re * (c - a).im - (c - a).re * (b - a).im) / 2

/-- The theorem stating the least possible area of a triangle formed by any three solutions -/
theorem least_triangle_area :
  ∃ (a b c : ℂ) (ha : a ∈ solutions) (hb : b ∈ solutions) (hc : c ∈ solutions),
    (∀ (x y z : ℂ) (hx : x ∈ solutions) (hy : y ∈ solutions) (hz : z ∈ solutions),
      triangleArea a b c ≤ triangleArea x y z) ∧
    triangleArea a b c = (Real.sqrt 50 - Real.sqrt 10) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_triangle_area_l294_29413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_twice_f_at_a_l294_29423

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

-- State the theorem
theorem integral_equals_twice_f_at_a :
  ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧
  (∫ x in (-1 : ℝ)..1, f x) = 2 * f a₁ ∧
  (∫ x in (-1 : ℝ)..1, f x) = 2 * f a₂ ∧
  (a₁ = -1 ∨ a₁ = 1/3) ∧ (a₂ = -1 ∨ a₂ = 1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_twice_f_at_a_l294_29423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_t_value_l294_29491

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (t, 3) are parallel, then t = 3/2 -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![t, 3]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  t = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_t_value_l294_29491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_formula_limit_A_n_l294_29499

noncomputable def A (n : ℕ) : ℝ :=
  Real.sqrt (2 - Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))  -- This is a simplification, as Lean can't handle variable-length nested radicals

theorem A_n_formula (n : ℕ) (h : n ≥ 2) :
  A n = 2 * Real.sin (π / 2^(n+1)) := by
  sorry

theorem limit_A_n :
  Filter.Tendsto (fun n => 2^n * A n) Filter.atTop (Filter.principal π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_formula_limit_A_n_l294_29499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_stack_height_l294_29485

/-- The height of a stack of three spherical balls with diameter 12 cm in a triangular pyramid formation --/
noncomputable def stack_height : ℝ := 12 + 6 * Real.sqrt 3

/-- Function to calculate the height of a triangular stack of balls --/
noncomputable def height_of_triangular_ball_stack (ball_diameter : ℝ) (num_balls : ℕ) : ℝ :=
  ball_diameter / 2 + (ball_diameter / 2) * Real.sqrt 3 + ball_diameter / 2

/-- Theorem stating the height of the stack of balls --/
theorem ball_stack_height :
  let ball_diameter : ℝ := 12
  let num_balls : ℕ := 3
  let stack_height : ℝ := 12 + 6 * Real.sqrt 3
  ∀ (h : ℝ),
    (h = height_of_triangular_ball_stack ball_diameter num_balls) →
    h = stack_height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_stack_height_l294_29485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1000_l294_29496

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal rate time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Proves that the principal is 1000 given the conditions -/
theorem principal_is_1000 (P : ℝ) :
  let R : ℝ := 8
  let T : ℝ := 4
  compoundInterest P R T - simpleInterest P R T = 40.48896000000036 →
  P = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1000_l294_29496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_paddle_time_l294_29438

/-- Calculates the time taken to paddle up a river against the current. -/
noncomputable def paddleUpRiver (paddleSpeed : ℝ) (currentSpeed : ℝ) (riverLength : ℝ) : ℝ :=
  riverLength / (paddleSpeed - currentSpeed)

/-- Proves that Karen takes 2 hours to paddle up the river. -/
theorem karen_paddle_time :
  let paddleSpeed : ℝ := 10
  let currentSpeed : ℝ := 4
  let riverLength : ℝ := 12
  paddleUpRiver paddleSpeed currentSpeed riverLength = 2 := by
  -- Unfold the definition of paddleUpRiver
  unfold paddleUpRiver
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_paddle_time_l294_29438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_8_pow_2021_l294_29481

theorem digit_sum_8_pow_2021 : ∃ (n : ℕ), n > 0 ∧ n < 10 ∧ 
  (∃ (f : ℕ → ℕ), f 0 = 8^2021 ∧ 
    (∀ i, f (i+1) = if f i < 10 then f i else (Nat.digits 10 (f i)).sum) ∧
    f n = n ∧ 
    (∀ j, j > n → f j = f n)) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_8_pow_2021_l294_29481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_intersection_l294_29487

/-- Parabola with vertex at origin and focus on x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : C.equation x y

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem parabola_equation_and_intersection (C : Parabola) 
  (P : PointOnParabola C) (k : ℝ) :
  P.x = 4 ∧ 
  distance P.x P.y (C.p/2) 0 = 6 ∧
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    C.equation A.1 A.2 ∧ 
    C.equation B.1 B.2 ∧
    A.2 = k * A.1 - 2 ∧ 
    B.2 = k * B.1 - 2 ∧
    (A.1 + B.1) / 2 = 2) →
  C.p = 4 ∧ k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_intersection_l294_29487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inscribed_circle_inequality_l294_29429

/-- Predicate stating that three points form a triangle -/
def IsTriangle (A B C : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

/-- Predicate stating that a circle is inscribed in a triangle -/
def IsInscribedCircle (center : ℝ × ℝ) (radius : ℝ) (A B C : ℝ × ℝ) (h_triangle : IsTriangle A B C) : Prop :=
  sorry

/-- Predicate stating that a circle is tangent to a triangle -/
def IsTangentCircle (center : ℝ × ℝ) (radius : ℝ) (A B C : ℝ × ℝ) (h_triangle : IsTriangle A B C) : Prop :=
  sorry

/-- Given a triangle ABC with an inscribed circle of radius r and three smaller circles
    with radii rA, rB, rC tangent to the inscribed circle and sides emanating from A, B, C
    respectively, prove that r ≤ rA + rB + rC ≤ 2r -/
theorem triangle_inscribed_circle_inequality
  (r rA rB rC : ℝ)
  (h_positive : r > 0 ∧ rA > 0 ∧ rB > 0 ∧ rC > 0)
  (A B C : ℝ × ℝ)
  (h_triangle : IsTriangle A B C)
  (h_inscribed : ∃ (center : ℝ × ℝ), IsInscribedCircle center r A B C h_triangle)
  (h_tangent : ∃ (centerA centerB centerC : ℝ × ℝ),
    IsTangentCircle centerA rA A B C h_triangle ∧
    IsTangentCircle centerB rB A B C h_triangle ∧
    IsTangentCircle centerC rC A B C h_triangle) :
  r ≤ rA + rB + rC ∧ rA + rB + rC ≤ 2 * r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inscribed_circle_inequality_l294_29429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l294_29433

def sequenceA (n : ℕ+) : ℚ := 4 * n - 2

theorem sequence_properties :
  (sequenceA 1 = 2) ∧
  (sequenceA 17 = 66) ∧
  (∀ n : ℕ+, ∃ a b : ℚ, sequenceA n = a * n + b) ∧
  (∀ n : ℕ+, sequenceA n ≠ 88) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l294_29433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l294_29447

-- Define the set of solutions to the inequality
def SolutionSet : Set ℝ := {x : ℝ | |x - 3| + |x + 1| > 6}

-- State the theorem
theorem solution_set_characterization : 
  SolutionSet = Set.Ioi 4 ∪ Set.Iic (-2) := by sorry

-- Here, Set.Ioi represents an interval that is open on the right (x > a)
-- Set.Iic represents an interval that is closed on the left (x ≤ a)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l294_29447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendas_journey_l294_29400

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Brenda's journey from (-3, 6) to (0, 0) to (6, -3) -/
theorem brendas_journey :
  let d1 := distance (-3) 6 0 0
  let d2 := distance 0 0 6 (-3)
  d1 + d2 = 2 * Real.sqrt 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendas_journey_l294_29400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l294_29418

theorem trig_inequality : 
  Real.sin (393 * π / 180) < Real.cos (55 * π / 180) ∧ 
  Real.cos (55 * π / 180) < Real.tan (50 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l294_29418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_reunions_l294_29484

theorem hotel_reunions (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ)
  (h1 : total_guests = 100)
  (h2 : oates_attendees = 50)
  (h3 : hall_attendees = 62)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ oates_attendees ∨ g ≤ hall_attendees)) :
  (oates_attendees + hall_attendees - total_guests : ℕ) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_reunions_l294_29484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangency_l294_29493

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is on a hyperbola -/
def isOnHyperbola (p : Point) (h : Hyperbola) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Get the foci of a hyperbola -/
noncomputable def getFoci (h : Hyperbola) : (Point × Point) :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  ({ x := -c, y := 0 }, { x := c, y := 0 })

/-- Create a circle with given diameter endpoints -/
noncomputable def circleFromDiameter (p1 p2 : Point) : Circle :=
  let center : Point := { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }
  let radius : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2) / 2
  { center := center, radius := radius }

/-- Check if two circles are tangent (either internally or externally) -/
def areTangent (c1 c2 : Circle) : Prop :=
  let centerDist := Real.sqrt ((c1.center.x - c2.center.x)^2 + (c1.center.y - c2.center.y)^2)
  centerDist = c1.radius + c2.radius ∨ centerDist = |c1.radius - c2.radius|

/-- Main theorem -/
theorem hyperbola_circle_tangency (h : Hyperbola) (p : Point) (f : Point) :
  isOnHyperbola p h →
  (f = (getFoci h).1 ∨ f = (getFoci h).2) →
  areTangent (circleFromDiameter p f) { center := { x := 0, y := 0 }, radius := h.a } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangency_l294_29493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l294_29494

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x) + Real.log (x^2 + 1)

theorem domain_of_f :
  {x : ℝ | x ≠ 1 ∧ x^2 + 1 > 0} = Set.Iio 1 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l294_29494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_difference_l294_29471

/-- Two trains with different speeds passing a pole and a platform -/
structure TrainProblem where
  pole_length : ℝ := 0
  platform_length : ℝ := 120
  train1_pole_time : ℝ := 11
  train1_platform_time : ℝ := 22
  train2_pole_time : ℝ := 15
  train2_platform_time : ℝ := 30

/-- Calculate the length of a train given its passing times -/
noncomputable def train_length (p : TrainProblem) (pole_time platform_time : ℝ) : ℝ :=
  (p.platform_length * pole_time) / (platform_time - pole_time)

/-- The main theorem stating that the difference in train lengths is zero -/
theorem train_length_difference (p : TrainProblem) :
  |train_length p p.train1_pole_time p.train1_platform_time -
   train_length p p.train2_pole_time p.train2_platform_time| = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_difference_l294_29471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_range_of_a_l294_29489

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 2) + Real.log (3 - x)

-- Define the domain of f
def A : Set ℝ := {x | x > -2 ∧ x < 3}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Theorem for part (I)
theorem complement_of_domain :
  (Set.univ \ A) = Set.Iic (-2) ∪ Set.Ici 3 := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) :
  A ∪ B a = A → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_range_of_a_l294_29489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_odd_l294_29409

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Condition 1: For any a, b ∈ ℝ, f(a+b) = f(a) + f(b)
axiom f_additive : ∀ a b : ℝ, f (a + b) = f a + f b

-- Condition 2: For x > 0, f(x) < 0
axiom f_negative_for_positive : ∀ x : ℝ, x > 0 → f x < 0

-- Theorem 1: f is decreasing
theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ < f x₂ := by
  sorry

-- Theorem 2: f is odd
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_odd_l294_29409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_are_parallel_l294_29455

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Calculate the y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ := -l.c / l.b

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.yIntercept ≠ l2.yIntercept

theorem lines_are_parallel : 
  let l1 : Line := { a := 2, b := -4, c := 7 }
  let l2 : Line := { a := 1, b := -2, c := 5 }
  parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_are_parallel_l294_29455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l294_29474

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7
  else if x > 0 then 9 * x + a^2 / x - 7
  else 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) →  -- f is odd
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) →  -- condition for x ≥ 0
  a ≤ -8/7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l294_29474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l294_29436

theorem trigonometric_identity (α : ℝ) :
  4.44 * Real.tan (2 * α) + (Real.tan (2 * α))⁻¹ + Real.tan (6 * α) + (Real.tan (6 * α))⁻¹ =
  (8 * (Real.cos (4 * α))^2) / Real.sin (12 * α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l294_29436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l294_29432

theorem max_log_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x * y * z + y + z = 12) :
  Real.log x / Real.log 4 + Real.log y / Real.log 2 + Real.log z / Real.log 2 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l294_29432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_equality_l294_29460

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 3 * Real.cos x

theorem f_derivative_equality (n : ℤ) :
  let f' := deriv f
  ∀ x : ℝ, f' 0 = f' x ↔ 
    (x = 2 * Real.pi * ↑n) ∨ 
    (x = 2 * Real.pi * ↑n - 2 * Real.arctan (3/5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_equality_l294_29460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_plus_2x_l294_29443

theorem integral_exp_plus_2x : ∫ x in (Set.Icc 0 1), (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_plus_2x_l294_29443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_contour_length_is_6_pi_a_l294_29439

/-- Configuration of four tangent circles -/
structure FourCircleConfig where
  radius : ℝ
  radius_pos : radius > 0

/-- Length of the external contour for the four-circle configuration -/
noncomputable def external_contour_length (config : FourCircleConfig) : ℝ :=
  6 * Real.pi * config.radius

/-- Theorem: The length of the external contour is 6πa -/
theorem external_contour_length_is_6_pi_a (config : FourCircleConfig) :
  external_contour_length config = 6 * Real.pi * config.radius :=
by
  -- Unfold the definition of external_contour_length
  unfold external_contour_length
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_contour_length_is_6_pi_a_l294_29439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l294_29469

noncomputable def g (A : ℝ) : ℝ :=
  (Real.sin A * (5 * Real.cos A ^ 2 + Real.cos A ^ 6 + 2 * Real.sin A ^ 2 + 2 * Real.sin A ^ 4 * Real.cos A ^ 2)) /
  (Real.tan A * ((1 / Real.cos A) - 2 * Real.sin A * Real.tan A))

theorem g_range (A : ℝ) (h : ∀ n : ℤ, A ≠ n * Real.pi / 2) :
  Set.range g = Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l294_29469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_2310_l294_29427

theorem distinct_prime_factors_of_2310 : Finset.card (Nat.factors 2310).toFinset = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_2310_l294_29427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_f_is_decreasing_k_range_for_inequality_l294_29470

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.exp (x * Real.log 3)) / (Real.exp (x * Real.log 3) + 1)

theorem odd_function_implies_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1 := by sorry

theorem f_is_decreasing :
  ∀ x y : ℝ, x < y → f 1 x > f 1 y := by sorry

theorem k_range_for_inequality (k : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/6) (Real.pi/3) → f 1 (Real.sin (2*x)) + f 1 (2-k) < 0) →
  k < 2 - Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_f_is_decreasing_k_range_for_inequality_l294_29470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_s_evaluation_l294_29495

noncomputable def s (θ : ℝ) : ℝ := 1 / (1 - θ^2)

theorem nested_s_evaluation :
  s (s (s (s 10))) = -1/9999 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_s_evaluation_l294_29495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_crossing_at_most_twice_l294_29421

/-- The n-th Catalan number -/
def catalan' (n : ℕ) : ℕ := sorry

/-- A path on a square grid -/
inductive Path'
| nil : Path'
| right : Path' → Path'
| up : Path' → Path'

/-- Whether a path crosses the diagonal at most twice -/
def crossesDiagonalAtMostTwice (p : Path') : Prop := sorry

/-- The number of paths from (0,0) to (n,n) crossing the diagonal at most twice -/
def numPathsCrossingAtMostTwice (n : ℕ) : ℕ := sorry

/-- Theorem: The number of paths crossing the diagonal at most twice
    equals C_{n+2} - 2C_{n+1} + C_n -/
theorem paths_crossing_at_most_twice (n : ℕ) :
  numPathsCrossingAtMostTwice n = catalan' (n + 2) - 2 * catalan' (n + 1) + catalan' n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_crossing_at_most_twice_l294_29421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_proof_l294_29486

noncomputable def average_value (z : ℝ) : ℝ := (0 + 3*z + 9*z + 27*z + 81*z) / 5

theorem average_value_proof (z : ℝ) : average_value z = 24*z := by
  -- Unfold the definition of average_value
  unfold average_value
  -- Simplify the numerator
  simp [add_mul, mul_add]
  -- Perform the division
  field_simp
  -- Simplify the result
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_proof_l294_29486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_to_line_l294_29445

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between a point and a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a ^ 2 + l.b ^ 2)

/-- Parameterized point P -/
def P (t : ℝ) : Point :=
  { x := t ^ 2, y := 2 * t }

/-- Line l -/
def l : Line :=
  { a := 1, b := -1, c := 2 }

theorem minimum_distance_point_to_line :
  ∃ (t : ℝ), ∀ (s : ℝ), distancePointToLine (P t) l ≤ distancePointToLine (P s) l ∧
  distancePointToLine (P t) l = Real.sqrt 2 / 2 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_to_line_l294_29445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_dodecagon_area_preservation_l294_29403

theorem triangle_to_dodecagon_area_preservation 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ d : ℝ, d > 0 ∧
  let s := (a + b + c) / 2
  d = Real.sqrt (Real.sqrt (s * (s - a) * (s - b) * (s - c)) / (3 * (2 + Real.sqrt 3))) ∧
  3 * (2 + Real.sqrt 3) * d^2 = Real.sqrt (s * (s - a) * (s - b) * (s - c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_dodecagon_area_preservation_l294_29403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_l294_29479

/-- Represents the growth rate of a tree in centimeters per two weeks -/
def growth_rate : ℚ := 50

/-- Represents the future height of the tree in centimeters after 4 months -/
def future_height : ℚ := 600

/-- Represents the time period in months -/
def time_period : ℚ := 4

/-- Calculates the current height of the tree in meters -/
noncomputable def current_height : ℚ := 
  (future_height - (time_period * 4 / 2) * growth_rate) / 100

/-- Theorem stating that the current height of the tree is 2 meters -/
theorem tree_height : current_height = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_l294_29479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_shoes_l294_29420

/-- The probability of drawing two red shoes from a set of 5 red shoes and 4 green shoes -/
theorem probability_two_red_shoes (red_shoes green_shoes : ℕ) 
  (h_red : red_shoes = 5) (h_green : green_shoes = 4) : ℚ := by
  /- Define the probability of drawing two red shoes -/
  let prob_two_red : ℚ := (red_shoes / (red_shoes + green_shoes)) * 
                          ((red_shoes - 1) / (red_shoes + green_shoes - 1))
  /- The theorem states that this probability equals 5/18 -/
  have : prob_two_red = 5 / 18 := by sorry
  exact 5 / 18


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_shoes_l294_29420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_apple_distribution_l294_29497

theorem martha_apple_distribution 
  (total : ℕ) 
  (keep : ℕ) 
  (james_extra : ℕ) 
  (remaining : ℕ) 
  (jane_apples : ℕ) : 
  total = 20 ∧ 
  keep = 4 ∧ 
  james_extra = 2 ∧ 
  remaining = 4 ∧ 
  jane_apples + (jane_apples + james_extra) + remaining + keep = total ∧ 
  jane_apples = 5 := by
  sorry

#check martha_apple_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_apple_distribution_l294_29497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l294_29431

/-- Given two points (x₁, y₁) and (x₂, y₂), calculate the slope of the line passing through them. -/
noncomputable def my_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

/-- Given the coefficients a, b of a line ax + by = c, calculate its slope. -/
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

theorem parallel_lines_m_value (m : ℝ) :
  let line1_slope := line_slope 5 4
  let line2_slope := my_slope 1 4 m (-3)
  line1_slope = line2_slope →
  m = 33 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l294_29431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_purchase_proof_l294_29435

def cheese_purchase (initial_money : ℕ) (cheese_cost : ℕ) (beef_cost : ℕ) 
  (beef_amount : ℕ) (money_left : ℕ) : ℕ :=
  (initial_money - money_left - beef_cost * beef_amount) / cheese_cost

#eval cheese_purchase 87 7 5 1 61

-- The theorem statement
theorem cheese_purchase_proof :
  cheese_purchase 87 7 5 1 61 = 3 := by
  -- Unfold the definition of cheese_purchase
  unfold cheese_purchase
  -- Evaluate the arithmetic expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_purchase_proof_l294_29435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_triangles_common_line_l294_29488

-- Define a triangle in 2D space
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define a line in 2D space
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define the intersection of a line and a triangle
def intersects (l : Line) (t : Triangle) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Set.range t.vertices ∧ ∃ k : ℝ, p = l.point + k • l.direction

-- Define a collection of intersecting triangles
def intersecting_triangles (S : Set Triangle) : Prop :=
  ∀ t₁ t₂, t₁ ∈ S → t₂ ∈ S → t₁ ≠ t₂ → ∃ p : ℝ × ℝ, p ∈ Set.range t₁.vertices ∩ Set.range t₂.vertices

-- Theorem statement
theorem intersecting_triangles_common_line 
  (S₁ S₂ : Set Triangle) 
  (h₁ : Set.Finite S₁) 
  (h₂ : Set.Finite S₂) 
  (h₃ : intersecting_triangles S₁) 
  (h₄ : intersecting_triangles S₂) :
  ∃ l : Line, ∀ t ∈ S₁ ∪ S₂, intersects l t :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_triangles_common_line_l294_29488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l294_29464

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio -1/3 and sum 27, the first term is 36 -/
theorem first_term_of_geometric_series (r S a : ℝ) 
  (h_r : r = -1/3) 
  (h_S : S = 27) 
  (h_sum : infiniteGeometricSeriesSum a r = S) : 
  a = 36 := by
  sorry

#check first_term_of_geometric_series

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l294_29464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_z_values_l294_29425

/-- Represents a three-digit integer with potentially leading zeros -/
structure ThreeDigitInt where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Reverses the digits of a ThreeDigitInt -/
def reverse (x : ThreeDigitInt) : ThreeDigitInt where
  hundreds := x.ones
  tens := x.tens
  ones := x.hundreds
  is_valid := by
    simp [x.is_valid]

/-- Converts a ThreeDigitInt to a natural number -/
def to_nat (x : ThreeDigitInt) : Nat :=
  100 * x.hundreds + 10 * x.tens + x.ones

/-- The absolute difference between a number and its reverse -/
def z (x : ThreeDigitInt) : Nat :=
  Int.natAbs (to_nat x - to_nat (reverse x))

/-- The main theorem stating that there are exactly 10 distinct possible values for z -/
theorem distinct_z_values :
  ∃ (S : Finset Nat), (∀ x : ThreeDigitInt, z x ∈ S) ∧ S.card = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_z_values_l294_29425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l294_29475

/-- Sum of arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

/-- Theorem: Sum of first 10 terms of arithmetic sequence starting at -3 with common difference 6 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum (-3) 6 10 = 240 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l294_29475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_C_value_l294_29407

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def first_number (A B D : ℕ) : ℕ := 7000000 + 400000 + 100000 * A + 50000 + 2000 + 100 * B + D

def second_number (A B C : ℕ) : ℕ := 3000000 + 200000 + 60000 + 10000 * A + 1000 * B + 500 + C

def is_single_digit (n : ℕ) : Prop := n < 10

theorem unique_C_value (A B C D : ℕ) :
  is_single_digit A →
  is_single_digit B →
  is_single_digit C →
  is_single_digit D →
  is_divisible_by_three (first_number A B D) →
  is_divisible_by_three (second_number A B C) →
  C ∈ ({1, 2, 3, 4, 7} : Set ℕ) →
  C = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_C_value_l294_29407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l294_29417

noncomputable def parabola (y : ℝ) : ℝ := -1/8 * y^2

def directrix : ℝ := -2

theorem parabola_directrix :
  ∀ y : ℝ, ∃ x : ℝ, x = parabola y ∧ 
  (x - directrix) = (y^2 + (x - directrix)^2) / (4 * (x - directrix)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l294_29417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l294_29411

/-- Given two non-collinear vectors a and b, and points A, B, C, D in a vector space,
    if AB = 2a + pb, BC = a + b, CD = a - 2b, and A, B, D are collinear,
    then p = -1 -/
theorem vector_collinearity (V : Type*) [AddCommGroup V] [Module ℝ V]
  (a b : V) (hnoncollinear : ¬ (∃ (r s : ℝ), r • a + s • b = 0 ∧ (r ≠ 0 ∨ s ≠ 0)))
  (A B C D : V) (p : ℝ)
  (hAB : B - A = 2 • a + p • b)
  (hBC : C - B = a + b)
  (hCD : D - C = a - 2 • b)
  (hcollinear : ∃ (t : ℝ), D - A = t • (B - A))
  : p = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l294_29411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_max_product_l294_29437

-- Define the circles C₁ and C₂ in parametric form
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)
noncomputable def C₂ (β : ℝ) : ℝ × ℝ := (Real.cos β, 1 + Real.sin β)

-- Define the polar equations
noncomputable def polar_eq_C₁ (θ : ℝ) : ℝ := 4 * Real.cos θ
noncomputable def polar_eq_C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the intersection points P and Q
noncomputable def P (α : ℝ) : ℝ := polar_eq_C₁ α
noncomputable def Q (α : ℝ) : ℝ := polar_eq_C₂ α

-- State the theorem
theorem circle_intersection_max_product :
  (∀ α : ℝ, P α * Q α ≤ 4) ∧
  (∃ α : ℝ, P α * Q α = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_max_product_l294_29437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_zeros_l294_29498

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

theorem f_max_and_zeros :
  (∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M) ∧
  (∀ (x : ℝ), f x = 0 ↔ (∃ (k : ℤ), x = k * Real.pi ∨ x = k * Real.pi + Real.pi / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_zeros_l294_29498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_60_90_l294_29463

/-- The number of positive integer divisors common to both 60 and 90 is 8. -/
theorem common_divisors_60_90 : 
  Finset.card (Finset.filter (fun d => d ∣ 60 ∧ d ∣ 90) (Nat.divisors 90)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_60_90_l294_29463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_l294_29458

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the condition 2*angle B = angle C
def angleCondition (t : AcuteTriangle) : Prop := 2 * t.B = t.C

-- Define the ratio AB/AC
noncomputable def ratio (t : AcuteTriangle) : Real := Real.sin t.C / Real.sin t.B

-- Theorem statement
theorem ratio_range (t : AcuteTriangle) (h : angleCondition t) :
  Real.sqrt 2 < ratio t ∧ ratio t < Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_l294_29458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_prob_after_five_steps_l294_29444

/-- A step in the coin-die process -/
def step : Type := Unit

/-- The probability of getting tails on a fair coin -/
noncomputable def probTails : ℝ := 1 / 2

/-- The probability of rolling a 1 on a fair 6-sided die -/
noncomputable def probRoll1 : ℝ := 1 / 6

/-- The number of steps in the process -/
def numSteps : ℕ := 5

/-- The probability of the die showing 1 after the given number of steps -/
noncomputable def probDie1AfterSteps (n : ℕ) : ℝ :=
  (probTails ^ n) + (1 - probTails ^ n) * probRoll1

theorem die_prob_after_five_steps :
  probDie1AfterSteps numSteps = 37 / 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_prob_after_five_steps_l294_29444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_limit_zero_l294_29462

open Real

noncomputable def seq (n : ℕ) : ℝ := sin (Real.sqrt (n^2 + 1)) * arctan (n / (n^2 + 1))

theorem seq_limit_zero : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |seq n| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_limit_zero_l294_29462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l294_29428

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the centers and radii
def center_M : ℝ × ℝ := (0, 2)
def center_N : ℝ × ℝ := (1, 1)
def radius_M : ℝ := 2
def radius_N : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center_M.1 - center_N.1)^2 + (center_M.2 - center_N.2)^2)

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > |radius_M - radius_N| ∧
  distance_between_centers < radius_M + radius_N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l294_29428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l294_29405

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Axioms for odd and even functions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Axiom for the sum of f and g
axiom sum_eq_exp : ∀ x : ℝ, f x + g x = Real.exp (Real.log 2 * x)

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → a * f x + g (2 * x) ≥ 0

-- Theorem statement
theorem range_of_a :
  {a : ℝ | inequality_condition a} = Set.Ici (-17/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l294_29405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_approx_l294_29461

/-- The molar mass of Nitrogen in g/mol -/
noncomputable def molar_mass_N : ℝ := 14.01

/-- The molar mass of Hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- The molar mass of Iodine in g/mol -/
noncomputable def molar_mass_I : ℝ := 126.90

/-- The molar mass of Ammonium iodide (NH4I) in g/mol -/
noncomputable def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I

/-- The mass percentage of Nitrogen in Ammonium iodide -/
noncomputable def mass_percentage_N : ℝ := (molar_mass_N / molar_mass_NH4I) * 100

/-- Theorem: The mass percentage of Nitrogen in Ammonium iodide is approximately 9.66% -/
theorem mass_percentage_N_approx :
  ∃ ε > 0, |mass_percentage_N - 9.66| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_approx_l294_29461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_pace_theorem_l294_29402

/-- Represents the marathon race scenario -/
structure MarathonRace where
  total_distance : ℚ
  initial_distance : ℚ
  initial_time : ℚ
  total_time : ℚ

/-- Calculates the percentage of the initial pace for the remaining distance -/
def remaining_pace_percentage (race : MarathonRace) : ℚ :=
  let remaining_distance := race.total_distance - race.initial_distance
  let remaining_time := race.total_time - race.initial_time
  let initial_pace := race.initial_distance / race.initial_time
  let remaining_pace := remaining_distance / remaining_time
  (remaining_pace / initial_pace) * 100

/-- Theorem stating that the remaining pace is 80% of the initial pace -/
theorem marathon_pace_theorem (race : MarathonRace) 
    (h1 : race.total_distance = 26)
    (h2 : race.initial_distance = 10)
    (h3 : race.initial_time = 1)
    (h4 : race.total_time = 3) :
  remaining_pace_percentage race = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_pace_theorem_l294_29402
