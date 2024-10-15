import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_sum_l3262_326280

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are equal. -/
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_about_y_axis (a, 5) (2, b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l3262_326280


namespace NUMINAMATH_CALUDE_team_selection_ways_eq_8400_l3262_326219

/-- The number of ways to select a team of 4 boys from 8 boys and 3 girls from 10 girls -/
def team_selection_ways : ℕ :=
  (Nat.choose 8 4) * (Nat.choose 10 3)

/-- Theorem stating that the number of ways to select the team is 8400 -/
theorem team_selection_ways_eq_8400 : team_selection_ways = 8400 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_eq_8400_l3262_326219


namespace NUMINAMATH_CALUDE_ring_width_equals_disk_radius_l3262_326208

/-- A flat ring formed by two concentric circles with seven equal touching disks inserted -/
structure FlatRing where
  R₁ : ℝ  -- Radius of the outer circle
  R₂ : ℝ  -- Radius of the inner circle
  r : ℝ   -- Radius of each disk
  h₁ : R₁ > R₂  -- Outer radius is greater than inner radius
  h₂ : R₂ = 3 * r  -- Inner radius is 3 times the disk radius
  h₃ : 7 * π * r^2 = π * (R₁^2 - R₂^2)  -- Area of ring equals sum of disk areas

/-- The width of the ring is equal to the radius of one disk -/
theorem ring_width_equals_disk_radius (ring : FlatRing) : ring.R₁ - ring.R₂ = ring.r := by
  sorry


end NUMINAMATH_CALUDE_ring_width_equals_disk_radius_l3262_326208


namespace NUMINAMATH_CALUDE_parabola_max_value_l3262_326274

theorem parabola_max_value (x : ℝ) : 
  ∃ (max : ℝ), max = 6 ∧ ∀ y : ℝ, y = -3 * x^2 + 6 → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_parabola_max_value_l3262_326274


namespace NUMINAMATH_CALUDE_log_equation_proof_l3262_326276

theorem log_equation_proof (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → (Real.log 125 / Real.log 2 = m * y) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_proof_l3262_326276


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3262_326235

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x^2 ≤ 4 }

theorem set_intersection_theorem :
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x ≤ 2 } := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3262_326235


namespace NUMINAMATH_CALUDE_quadruplet_babies_l3262_326257

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1250)
  (h_twins_quintuplets : ∃ t p : ℕ, t = 4 * p)
  (h_triplets_quadruplets : ∃ r q : ℕ, r = 2 * q)
  (h_quadruplets_quintuplets : ∃ q p : ℕ, q = 2 * p)
  (h_sum : ∃ t r q p : ℕ, 2 * t + 3 * r + 4 * q + 5 * p = total_babies) :
  ∃ q : ℕ, 4 * q = 303 :=
by sorry

end NUMINAMATH_CALUDE_quadruplet_babies_l3262_326257


namespace NUMINAMATH_CALUDE_remainder_problem_l3262_326228

theorem remainder_problem (x y : ℤ) 
  (hx : x % 72 = 65) 
  (hy : y % 54 = 22) : 
  (x - y) % 18 = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3262_326228


namespace NUMINAMATH_CALUDE_pond_to_field_ratio_l3262_326239

/-- Represents a rectangular field with a square pond inside -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ
  length_double_width : field_length = 2 * field_width
  field_length_16 : field_length = 16
  pond_side_8 : pond_side = 8

/-- The ratio of the pond area to the field area is 1:2 -/
theorem pond_to_field_ratio (f : FieldWithPond) : 
  (f.pond_side ^ 2) / (f.field_length * f.field_width) = 1 / 2 := by
  sorry

#check pond_to_field_ratio

end NUMINAMATH_CALUDE_pond_to_field_ratio_l3262_326239


namespace NUMINAMATH_CALUDE_det_positive_for_special_matrix_l3262_326220

open Matrix

theorem det_positive_for_special_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A + Aᵀ = 1) : 
  0 < det A := by
  sorry

end NUMINAMATH_CALUDE_det_positive_for_special_matrix_l3262_326220


namespace NUMINAMATH_CALUDE_ellipse_axis_sum_l3262_326213

/-- Proves that for an ellipse with given conditions, a + b = 40 -/
theorem ellipse_axis_sum (M N a b : ℝ) : 
  M > 0 → 
  N > 0 → 
  M = π * a * b → 
  N = π * (a + b) → 
  M / N = 10 → 
  a = b → 
  a + b = 40 := by
sorry

end NUMINAMATH_CALUDE_ellipse_axis_sum_l3262_326213


namespace NUMINAMATH_CALUDE_opposite_sign_implies_y_power_x_25_l3262_326207

theorem opposite_sign_implies_y_power_x_25 (x y : ℝ) : 
  (((x - 2)^2 > 0 ∧ |5 + y| < 0) ∨ ((x - 2)^2 < 0 ∧ |5 + y| > 0)) → y^x = 25 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_implies_y_power_x_25_l3262_326207


namespace NUMINAMATH_CALUDE_translation_theorem_l3262_326252

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translate_right (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_theorem :
  ∀ (A : Point),
    translate_right A 2 = Point.mk 3 2 →
    A = Point.mk 1 2 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3262_326252


namespace NUMINAMATH_CALUDE_all_symmetry_statements_correct_l3262_326271

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry with respect to y-axis
def symmetric_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define symmetry with respect to x-axis
def symmetric_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f x

-- Define symmetry with respect to origin
def symmetric_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define symmetry with respect to vertical line x = a
def symmetric_vertical_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem all_symmetry_statements_correct (f : ℝ → ℝ) : 
  (symmetric_y_axis f) ∧ 
  (symmetric_x_axis f) ∧ 
  (symmetric_origin f) ∧ 
  (∀ a : ℝ, symmetric_vertical_line f a → 
    ∃ g : ℝ → ℝ, ∀ x, g (x - a) = f x) :=
by sorry

end NUMINAMATH_CALUDE_all_symmetry_statements_correct_l3262_326271


namespace NUMINAMATH_CALUDE_range_of_m_l3262_326201

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → m ≤ x^2

def q (m l : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + l > 0

-- Theorem statement
theorem range_of_m (m l : ℝ) (h : p m ∧ q m l) : m ∈ Set.Ioo (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3262_326201


namespace NUMINAMATH_CALUDE_unique_single_solution_quadratic_l3262_326291

theorem unique_single_solution_quadratic :
  ∃! (p : ℝ), p ≠ 0 ∧ (∃! x : ℝ, p * x^2 - 12 * x + 4 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_single_solution_quadratic_l3262_326291


namespace NUMINAMATH_CALUDE_base_two_representation_123_l3262_326265

theorem base_two_representation_123 :
  ∃ (a b c d e f g : Nat),
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 ∧
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 :=
by sorry

end NUMINAMATH_CALUDE_base_two_representation_123_l3262_326265


namespace NUMINAMATH_CALUDE_two_distinct_solutions_l3262_326247

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ := x^3 - 2*a*x^2 - 3*a*x + a^2 - 2

/-- Theorem stating the condition for the cubic equation to have exactly two distinct real solutions -/
theorem two_distinct_solutions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    cubic_equation a x = 0 ∧ 
    cubic_equation a y = 0 ∧ 
    ∀ z : ℝ, cubic_equation a z = 0 → z = x ∨ z = y) ↔ 
  a > 15/8 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_solutions_l3262_326247


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l3262_326202

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of a point being on the ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Definition of the line l -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Definition of a point being on the line l -/
def on_line_l (k m : ℝ) (p : ℝ × ℝ) : Prop := line_l k m p.1 p.2

/-- Definition of the right vertex of the ellipse C -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Definition of the circle with diameter AB passing through a point -/
def circle_AB_passes_through (A B p : ℝ × ℝ) : Prop :=
  (p.1 - A.1) * (p.1 - B.1) + (p.2 - A.2) * (p.2 - B.2) = 0

/-- The main theorem -/
theorem ellipse_intersection_fixed_point :
  ∀ (k m : ℝ) (A B : ℝ × ℝ),
    on_ellipse_C A ∧ on_ellipse_C B ∧
    on_line_l k m A ∧ on_line_l k m B ∧
    A ≠ right_vertex ∧ B ≠ right_vertex ∧
    circle_AB_passes_through A B right_vertex →
    on_line_l k m (1/2, 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l3262_326202


namespace NUMINAMATH_CALUDE_concentric_circles_chords_l3262_326294

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the number of chords needed to complete
    a full circle is 3. -/
theorem concentric_circles_chords (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chords_l3262_326294


namespace NUMINAMATH_CALUDE_adams_pants_l3262_326267

/-- The number of pairs of pants Adam initially took out -/
def P : ℕ := 31

/-- The number of jumpers Adam took out -/
def jumpers : ℕ := 4

/-- The number of pajama sets Adam took out -/
def pajama_sets : ℕ := 4

/-- The number of t-shirts Adam took out -/
def tshirts : ℕ := 20

/-- The number of friends who donate the same amount as Adam -/
def friends : ℕ := 3

/-- The total number of articles of clothing being donated -/
def total_donated : ℕ := 126

theorem adams_pants :
  P = 31 ∧
  (4 * (P + jumpers + 2 * pajama_sets + tshirts) / 2 = total_donated) :=
sorry

end NUMINAMATH_CALUDE_adams_pants_l3262_326267


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l3262_326244

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) :
  is_geometric_sequence a →
  (∀ k, a k > 0) →
  a 3 = a 2 + 2 * a 1 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l3262_326244


namespace NUMINAMATH_CALUDE_star_value_zero_l3262_326230

-- Define the star operation
def star (a b c : ℤ) : ℤ := (a + b + c)^2

-- Theorem statement
theorem star_value_zero : star 3 (-5) 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_value_zero_l3262_326230


namespace NUMINAMATH_CALUDE_optimal_cutting_l3262_326214

/-- Represents a rectangular piece of cardboard -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Represents the problem of cutting small rectangles from a large rectangle -/
structure CuttingProblem :=
  (large : Rectangle)
  (small : Rectangle)

/-- Calculates the maximum number of small rectangles that can be cut from a large rectangle -/
def maxPieces (p : CuttingProblem) : ℕ :=
  sorry

theorem optimal_cutting (p : CuttingProblem) 
  (h1 : p.large = ⟨17, 22⟩) 
  (h2 : p.small = ⟨3, 5⟩) : 
  maxPieces p = 24 :=
sorry

end NUMINAMATH_CALUDE_optimal_cutting_l3262_326214


namespace NUMINAMATH_CALUDE_frank_hamburger_sales_l3262_326282

/-- The number of additional hamburgers Frank needs to sell to reach his target revenue -/
def additional_hamburgers (target_revenue : ℕ) (price_per_hamburger : ℕ) (initial_sales : ℕ) : ℕ :=
  (target_revenue - price_per_hamburger * initial_sales) / price_per_hamburger

theorem frank_hamburger_sales : additional_hamburgers 50 5 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_hamburger_sales_l3262_326282


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3262_326204

/-- The vertex coordinates of a parabola in the form y = -(x + h)^2 + k are (h, k) -/
theorem parabola_vertex_coordinates (h k : ℝ) :
  let f : ℝ → ℝ := λ x => -(x + h)^2 + k
  (∀ x, f x = -(x + h)^2 + k) →
  (h, k) = Prod.mk (- h) k :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3262_326204


namespace NUMINAMATH_CALUDE_only_vegetarian_count_l3262_326243

/-- Represents the number of people in a family with different eating habits -/
structure FamilyEatingHabits where
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ
  total_veg : ℕ

/-- Theorem stating the number of people who eat only vegetarian -/
theorem only_vegetarian_count (f : FamilyEatingHabits) 
  (h1 : f.only_non_veg = 6)
  (h2 : f.both_veg_and_non_veg = 9)
  (h3 : f.total_veg = 20) :
  f.total_veg - f.both_veg_and_non_veg = 11 := by
  sorry

end NUMINAMATH_CALUDE_only_vegetarian_count_l3262_326243


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l3262_326279

-- Define set A as the solution set of -x^2 - 2x + 8 = 0
def A : Set ℝ := {x | -x^2 - 2*x + 8 = 0}

-- Define set B as the solution set of ax - 1 ≤ 0
def B (a : ℝ) : Set ℝ := {x | a*x - 1 ≤ 0}

-- Theorem 1: When a = 1, A ∩ B = {-4}
theorem intersection_when_a_is_one : A ∩ B 1 = {-4} := by sorry

-- Theorem 2: A ⊆ B if and only if -1/4 ≤ a ≤ 1/2
theorem subset_condition : 
  ∀ a : ℝ, A ⊆ B a ↔ -1/4 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l3262_326279


namespace NUMINAMATH_CALUDE_tulip_count_after_addition_tulip_count_is_24_l3262_326248

/-- Given a garden with tulips and sunflowers, prove the number of tulips after an addition of sunflowers. -/
theorem tulip_count_after_addition 
  (initial_ratio : Rat) 
  (initial_sunflowers : Nat) 
  (added_sunflowers : Nat) : Nat :=
  let final_sunflowers := initial_sunflowers + added_sunflowers
  let tulip_ratio := 3
  let sunflower_ratio := 7
  (tulip_ratio * final_sunflowers) / sunflower_ratio

#check tulip_count_after_addition (3/7) 42 14 = 24

/-- Prove that the result is indeed 24 -/
theorem tulip_count_is_24 : 
  tulip_count_after_addition (3/7) 42 14 = 24 := by
  sorry


end NUMINAMATH_CALUDE_tulip_count_after_addition_tulip_count_is_24_l3262_326248


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l3262_326298

theorem forgotten_angle_measure (n : ℕ) (h : n > 2) :
  (n - 1) * 180 - 2017 = 143 :=
sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l3262_326298


namespace NUMINAMATH_CALUDE_bakery_revenue_l3262_326216

/-- Calculates the total revenue from selling pumpkin and custard pies --/
def total_revenue (pumpkin_slices_per_pie : ℕ) (custard_slices_per_pie : ℕ) 
                  (pumpkin_price_per_slice : ℕ) (custard_price_per_slice : ℕ) 
                  (pumpkin_pies_sold : ℕ) (custard_pies_sold : ℕ) : ℕ :=
  (pumpkin_slices_per_pie * pumpkin_pies_sold * pumpkin_price_per_slice) +
  (custard_slices_per_pie * custard_pies_sold * custard_price_per_slice)

theorem bakery_revenue : 
  total_revenue 8 6 5 6 4 5 = 340 := by
  sorry

end NUMINAMATH_CALUDE_bakery_revenue_l3262_326216


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3262_326245

/-- The circle equation passing through points A(4, 1), B(6, -3), and C(-3, 0) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y - 15 = 0

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (4, 1)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (6, -3)

/-- Point C coordinates -/
def point_C : ℝ × ℝ := (-3, 0)

theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l3262_326245


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l3262_326254

theorem reciprocal_sum_equals_one : 
  1/2 + 1/3 + 1/12 + 1/18 + 1/72 + 1/108 + 1/216 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l3262_326254


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l3262_326217

theorem sum_of_reciprocal_roots (γ δ : ℝ) : 
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ 
   6 * c^2 + 5 * c + 7 = 0 ∧ 
   6 * d^2 + 5 * d + 7 = 0 ∧ 
   γ = 1 / c ∧ 
   δ = 1 / d) → 
  γ + δ = -5 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l3262_326217


namespace NUMINAMATH_CALUDE_orange_distribution_l3262_326224

theorem orange_distribution (x : ℚ) : 
  (x/2 + 1/2) + (1/2 * (x/2 - 1/2) + 1/2) + (1/2 * (x/4 - 3/4) + 1/2) = x → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l3262_326224


namespace NUMINAMATH_CALUDE_special_numbers_are_one_and_nine_l3262_326200

/-- The number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- The set of natural numbers that are equal to the square of their divisor count -/
def special_numbers : Set ℕ := {n : ℕ | n = (divisor_count n)^2}

/-- Theorem stating that the set of special numbers is equal to {1, 9} -/
theorem special_numbers_are_one_and_nine : special_numbers = {1, 9} := by sorry

end NUMINAMATH_CALUDE_special_numbers_are_one_and_nine_l3262_326200


namespace NUMINAMATH_CALUDE_M_less_than_N_l3262_326250

theorem M_less_than_N (x y : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y := by
  sorry

end NUMINAMATH_CALUDE_M_less_than_N_l3262_326250


namespace NUMINAMATH_CALUDE_expression_evaluation_l3262_326222

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (6 * x^2 * y * (-2 * x * y + y^3)) / (x * y^2) = -36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3262_326222


namespace NUMINAMATH_CALUDE_first_year_exceeding_2_million_is_correct_l3262_326288

/-- The year when the R&D investment first exceeds 2 million yuan -/
def first_year_exceeding_2_million : ℕ := 2020

/-- The initial R&D investment in 2016 (in millions of yuan) -/
def initial_investment : ℝ := 1.3

/-- The annual increase rate of R&D investment -/
def annual_increase_rate : ℝ := 0.12

/-- The target R&D investment (in millions of yuan) -/
def target_investment : ℝ := 2.0

/-- Function to calculate the R&D investment for a given year -/
def investment_for_year (year : ℕ) : ℝ :=
  initial_investment * (1 + annual_increase_rate) ^ (year - 2016)

theorem first_year_exceeding_2_million_is_correct :
  (∀ y : ℕ, y < first_year_exceeding_2_million → investment_for_year y ≤ target_investment) ∧
  investment_for_year first_year_exceeding_2_million > target_investment :=
by sorry

end NUMINAMATH_CALUDE_first_year_exceeding_2_million_is_correct_l3262_326288


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_no_zero_digit_l3262_326278

theorem exists_number_divisible_by_5_pow_1000_no_zero_digit :
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ k : ℕ, n / 10^k % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_no_zero_digit_l3262_326278


namespace NUMINAMATH_CALUDE_chess_draw_probability_l3262_326212

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.4) 
  (h_not_lose : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l3262_326212


namespace NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l3262_326227

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| - 2

-- State the theorem
theorem min_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (x₀ = 4 ∧ f x₀ = -2) :=
sorry

end NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l3262_326227


namespace NUMINAMATH_CALUDE_museum_visit_orders_l3262_326223

-- Define the number of museums
def n : ℕ := 5

-- Define the factorial function
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

-- Theorem: The number of permutations of n distinct objects is n!
theorem museum_visit_orders : factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_orders_l3262_326223


namespace NUMINAMATH_CALUDE_cucumber_salad_problem_l3262_326270

theorem cucumber_salad_problem (total : ℕ) (ratio : ℕ) : 
  total = 280 → ratio = 3 → ∃ (cucumbers : ℕ), cucumbers * (ratio + 1) = total ∧ cucumbers = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_cucumber_salad_problem_l3262_326270


namespace NUMINAMATH_CALUDE_original_magazine_cost_l3262_326263

/-- The original cost of a magazine can be determined from the number of magazines, 
    selling price, and total profit. -/
theorem original_magazine_cost 
  (num_magazines : ℕ) 
  (selling_price : ℚ) 
  (total_profit : ℚ) : 
  num_magazines = 10 → 
  selling_price = 7/2 → 
  total_profit = 5 → 
  (num_magazines : ℚ) * selling_price - total_profit = 30 ∧ 
  ((num_magazines : ℚ) * selling_price - total_profit) / num_magazines = 3 :=
by sorry

end NUMINAMATH_CALUDE_original_magazine_cost_l3262_326263


namespace NUMINAMATH_CALUDE_seminar_scheduling_l3262_326260

theorem seminar_scheduling (n : ℕ) (h : n = 5) : 
  (n! / 2 : ℕ) = 60 :=
sorry

end NUMINAMATH_CALUDE_seminar_scheduling_l3262_326260


namespace NUMINAMATH_CALUDE_curve_k_range_l3262_326262

theorem curve_k_range (a : ℝ) (k : ℝ) : 
  ((-a)^2 - a*(-a) + 2*a + k = 0) → k ≤ (1/2 : ℝ) ∧ ∀ (ε : ℝ), ε > 0 → ∃ (k' : ℝ), k' < -ε ∧ ∃ (a' : ℝ), ((-a')^2 - a'*(-a') + 2*a' + k' = 0) :=
by sorry

end NUMINAMATH_CALUDE_curve_k_range_l3262_326262


namespace NUMINAMATH_CALUDE_train_speed_proof_l3262_326290

theorem train_speed_proof (train_length bridge_length crossing_time : Real) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 170)
  (h3 : crossing_time = 16.7986561075114) : 
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmph := speed_ms * 3.6
  ⌊speed_kmph⌋ = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l3262_326290


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3262_326232

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6

def alternating_arrangement (m h : ℕ) : Prop :=
  m > 1 ∧ h > 0 ∧ m = h + 1

theorem book_arrangement_count :
  alternating_arrangement num_math_books num_history_books →
  (num_math_books * (num_math_books - 1) * (num_history_books.factorial / (num_history_books - (num_math_books - 1)).factorial)) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3262_326232


namespace NUMINAMATH_CALUDE_cone_surface_area_l3262_326258

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r l θ : Real) (h1 : l = 1) (h2 : θ = π / 2) :
  let lateral_area := π * r * l
  let base_area := π * r^2
  lateral_area = l^2 * θ / 2 →
  lateral_area + base_area = 5 * π / 16 := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3262_326258


namespace NUMINAMATH_CALUDE_cubic_product_theorem_l3262_326211

theorem cubic_product_theorem : 
  (2^3 - 1) / (2^3 + 1) * 
  (3^3 - 1) / (3^3 + 1) * 
  (4^3 - 1) / (4^3 + 1) * 
  (5^3 - 1) / (5^3 + 1) * 
  (6^3 - 1) / (6^3 + 1) * 
  (7^3 - 1) / (7^3 + 1) = 19 / 56 := by
  sorry

end NUMINAMATH_CALUDE_cubic_product_theorem_l3262_326211


namespace NUMINAMATH_CALUDE_lucas_future_age_l3262_326273

def age_problem (gladys_age billy_age lucas_age : ℕ) : Prop :=
  (gladys_age = 30) ∧
  (gladys_age = 3 * billy_age) ∧
  (gladys_age = 2 * (billy_age + lucas_age))

theorem lucas_future_age 
  (gladys_age billy_age lucas_age : ℕ) 
  (h : age_problem gladys_age billy_age lucas_age) : 
  lucas_age + 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucas_future_age_l3262_326273


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_l3262_326283

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_l3262_326283


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_negative_one_l3262_326237

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem sum_of_max_min_is_negative_one :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max + min = -1 := by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_negative_one_l3262_326237


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_6_l3262_326209

theorem closest_integer_to_sqrt_6 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 6| ≤ |m - Real.sqrt 6| ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_6_l3262_326209


namespace NUMINAMATH_CALUDE_find_k_l3262_326246

theorem find_k : ∃ k : ℝ, (64 / k = 4) ∧ (k = 16) := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3262_326246


namespace NUMINAMATH_CALUDE_perfect_square_expression_l3262_326285

theorem perfect_square_expression (x y z : ℤ) :
  9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3*x*y*z) =
  ((x + y + z)^2 - 6*(x*y + y*z + z*x))^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l3262_326285


namespace NUMINAMATH_CALUDE_f_zero_range_l3262_326206

/-- The function f(x) = x^3 + 2x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x - a

/-- The theorem stating that if f(x) has exactly one zero in (1, 2), then a is in (3, 12) -/
theorem f_zero_range (a : ℝ) : 
  (∃! x, x ∈ (Set.Ioo 1 2) ∧ f a x = 0) → a ∈ Set.Ioo 3 12 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_range_l3262_326206


namespace NUMINAMATH_CALUDE_f_is_power_function_l3262_326295

-- Define what a power function is
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the function we want to prove is a power function
def f (x : ℝ) : ℝ := x ^ (1/2)

-- Theorem statement
theorem f_is_power_function : is_power_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_power_function_l3262_326295


namespace NUMINAMATH_CALUDE_sum_of_mobile_keypad_numbers_l3262_326264

def mobile_keypad : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem sum_of_mobile_keypad_numbers : 
  mobile_keypad.sum = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_mobile_keypad_numbers_l3262_326264


namespace NUMINAMATH_CALUDE_books_division_l3262_326261

theorem books_division (total_books : ℕ) (divisions : ℕ) (final_category_size : ℕ) : 
  total_books = 400 → divisions = 4 → final_category_size = total_books / (2^divisions) → 
  final_category_size = 25 := by
sorry

end NUMINAMATH_CALUDE_books_division_l3262_326261


namespace NUMINAMATH_CALUDE_four_item_match_probability_correct_match_probability_theorem_l3262_326253

/-- The probability of correctly matching n distinct items to n distinct positions when guessing randomly. -/
def correct_match_probability (n : ℕ) : ℚ :=
  1 / n.factorial

/-- Theorem: For 4 items, the probability of a correct random match is 1/24. -/
theorem four_item_match_probability :
  correct_match_probability 4 = 1 / 24 := by
  sorry

/-- Theorem: The probability of correctly matching n distinct items to n distinct positions
    when guessing randomly is 1/n!. -/
theorem correct_match_probability_theorem (n : ℕ) :
  correct_match_probability n = 1 / n.factorial := by
  sorry

end NUMINAMATH_CALUDE_four_item_match_probability_correct_match_probability_theorem_l3262_326253


namespace NUMINAMATH_CALUDE_mom_bought_71_packages_l3262_326226

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The total number of t-shirts Mom has -/
def total_shirts : ℕ := 426

/-- The number of packages Mom bought -/
def packages_bought : ℕ := total_shirts / shirts_per_package

theorem mom_bought_71_packages : packages_bought = 71 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_71_packages_l3262_326226


namespace NUMINAMATH_CALUDE_mike_bought_21_books_l3262_326281

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that Mike bought 21 books at the yard sale -/
theorem mike_bought_21_books :
  books_bought 35 56 = 21 := by
  sorry

end NUMINAMATH_CALUDE_mike_bought_21_books_l3262_326281


namespace NUMINAMATH_CALUDE_correct_conic_propositions_l3262_326251

/-- Represents a proposition about conic sections -/
inductive ConicProposition
| Prop1
| Prop2
| Prop3
| Prop4
| Prop5

/-- Determines if a given proposition is correct -/
def is_correct (prop : ConicProposition) : Bool :=
  match prop with
  | ConicProposition.Prop1 => true
  | ConicProposition.Prop2 => false
  | ConicProposition.Prop3 => false
  | ConicProposition.Prop4 => true
  | ConicProposition.Prop5 => false

/-- The theorem to be proved -/
theorem correct_conic_propositions :
  (List.filter is_correct [ConicProposition.Prop1, ConicProposition.Prop2, 
                           ConicProposition.Prop3, ConicProposition.Prop4, 
                           ConicProposition.Prop5]).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_conic_propositions_l3262_326251


namespace NUMINAMATH_CALUDE_lemonade_stand_revenue_calculation_l3262_326299

/-- Calculates the gross revenue of a lemonade stand given total profit, babysitting income, and lemonade stand expenses. -/
def lemonade_stand_revenue (total_profit babysitting_income lemonade_expenses : ℤ) : ℤ :=
  (total_profit - babysitting_income) + lemonade_expenses

theorem lemonade_stand_revenue_calculation :
  lemonade_stand_revenue 44 31 34 = 47 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_revenue_calculation_l3262_326299


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l3262_326215

def monthly_salary : ℚ := 4166.67
def initial_savings_rate : ℚ := 0.20
def new_savings : ℚ := 500

def initial_savings : ℚ := monthly_salary * initial_savings_rate
def original_expenses : ℚ := monthly_salary - initial_savings
def increase_in_expenses : ℚ := initial_savings - new_savings
def percentage_increase : ℚ := (increase_in_expenses / original_expenses) * 100

theorem expense_increase_percentage :
  percentage_increase = 10 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l3262_326215


namespace NUMINAMATH_CALUDE_lily_bouquet_cost_l3262_326240

/-- The cost of a bouquet is directly proportional to the number of lilies it contains. -/
def DirectlyProportional (cost : ℝ → ℝ) (lilies : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, cost x = k * lilies x

theorem lily_bouquet_cost 
  (cost : ℝ → ℝ) 
  (lilies : ℝ → ℝ) 
  (h_prop : DirectlyProportional cost lilies)
  (h_18 : cost 18 = 30)
  (h_pos : ∀ x, lilies x > 0) :
  cost 27 = 45 := by
sorry

end NUMINAMATH_CALUDE_lily_bouquet_cost_l3262_326240


namespace NUMINAMATH_CALUDE_min_value_fraction_l3262_326268

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^3 / ((x + y)^3 * (y + z)^3) ≥ 27/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3262_326268


namespace NUMINAMATH_CALUDE_basis_transformation_l3262_326236

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem basis_transformation (OA OB OC : V) 
  (h : LinearIndependent ℝ ![OA, OB, OC]) 
  (h_span : Submodule.span ℝ {OA, OB, OC} = ⊤) :
  LinearIndependent ℝ ![OA + OB, OA - OB, OC] ∧ 
  Submodule.span ℝ {OA + OB, OA - OB, OC} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_basis_transformation_l3262_326236


namespace NUMINAMATH_CALUDE_marley_louis_orange_ratio_l3262_326210

theorem marley_louis_orange_ratio :
  let louis_oranges : ℕ := 5
  let samantha_apples : ℕ := 7
  let marley_apples : ℕ := 3 * samantha_apples
  let marley_total_fruits : ℕ := 31
  let marley_oranges : ℕ := marley_total_fruits - marley_apples
  (marley_oranges : ℚ) / louis_oranges = 2 := by sorry

end NUMINAMATH_CALUDE_marley_louis_orange_ratio_l3262_326210


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l3262_326297

theorem arcsin_equation_solution (x : ℝ) : 
  Real.arcsin (3 * x) - Real.arcsin x = π / 6 → 
  x = 1 / Real.sqrt (40 - 12 * Real.sqrt 3) ∨ 
  x = -1 / Real.sqrt (40 - 12 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l3262_326297


namespace NUMINAMATH_CALUDE_cos_plus_sin_range_l3262_326229

/-- 
Given a point P(x,1) where x ≥ 1 on the terminal side of angle θ in the Cartesian coordinate system,
the sum of cosine and sine of θ is strictly greater than 1 and less than or equal to √2.
-/
theorem cos_plus_sin_range (x : ℝ) (θ : ℝ) (h1 : x ≥ 1) 
  (h2 : x = Real.cos θ * Real.sqrt (x^2 + 1)) 
  (h3 : 1 = Real.sin θ * Real.sqrt (x^2 + 1)) : 
  1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cos_plus_sin_range_l3262_326229


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3262_326292

theorem molecular_weight_calculation (total_weight : ℝ) (number_of_moles : ℝ) 
  (h1 : total_weight = 2376)
  (h2 : number_of_moles = 8) : 
  total_weight / number_of_moles = 297 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3262_326292


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3262_326231

def f (a : ℤ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + 1

theorem quadratic_inequality_solution_set 
  (a : ℤ) 
  (h1 : ∃! x : ℝ, -2 < x ∧ x < -1 ∧ f a x = 0) :
  {x : ℝ | f a x > 1} = {x : ℝ | -1 < x ∧ x < 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3262_326231


namespace NUMINAMATH_CALUDE_pauls_books_count_paul_has_151_books_l3262_326255

/-- Calculates the total number of books Paul has after buying new ones -/
def total_books (initial_books new_books : ℕ) : ℕ :=
  initial_books + new_books

/-- Theorem: Paul's total books equal the sum of initial and new books -/
theorem pauls_books_count (initial_books new_books : ℕ) :
  total_books initial_books new_books = initial_books + new_books :=
by sorry

/-- Theorem: Paul now has 151 books -/
theorem paul_has_151_books :
  total_books 50 101 = 151 :=
by sorry

end NUMINAMATH_CALUDE_pauls_books_count_paul_has_151_books_l3262_326255


namespace NUMINAMATH_CALUDE_abc_sum_mod_7_l3262_326284

theorem abc_sum_mod_7 (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 2 ∧ 
  (4 * c) % 7 = 3 ∧ 
  (7 * b) % 7 = (4 + b) % 7 → 
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_7_l3262_326284


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l3262_326286

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 8)^2 = 49

-- Define a function that counts the number of distinct real solutions
def count_solutions : ℕ := sorry

-- Theorem statement
theorem equation_has_four_solutions : count_solutions = 4 := by sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l3262_326286


namespace NUMINAMATH_CALUDE_toys_in_box_time_l3262_326233

/-- The time required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (net_increase_per_minute : ℕ) : ℕ :=
  ((total_toys - net_increase_per_minute) / net_increase_per_minute) + 1

/-- Theorem: It takes 15 minutes to put 45 toys in the box with a net increase of 3 toys per minute -/
theorem toys_in_box_time : time_to_put_toys_in_box 45 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_toys_in_box_time_l3262_326233


namespace NUMINAMATH_CALUDE_incorrect_inequality_implication_l3262_326296

theorem incorrect_inequality_implication : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_implication_l3262_326296


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3262_326289

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≥ 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- Define the complement of A and B with respect to ℝ
def C_UA : Set ℝ := {x : ℝ | x ∉ A}
def C_UB : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem complement_intersection_theorem :
  (C_UA ∩ C_UB) = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3262_326289


namespace NUMINAMATH_CALUDE_evaluate_power_l3262_326234

theorem evaluate_power : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_evaluate_power_l3262_326234


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l3262_326275

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l3262_326275


namespace NUMINAMATH_CALUDE_max_value_of_a_l3262_326277

theorem max_value_of_a : 
  (∃ a : ℝ, ∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∀ a : ℝ, ∃ x : ℝ, x^2 - 2*x - 3 > 0 ∧ x ≥ a) →
  (∀ b : ℝ, (∀ x : ℝ, x < b → x^2 - 2*x - 3 > 0) → b ≤ -1) ∧
  (∀ x : ℝ, x < -1 → x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3262_326277


namespace NUMINAMATH_CALUDE_unique_prime_in_form_l3262_326266

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def number_form (A : ℕ) : ℕ := 305200 + A

theorem unique_prime_in_form :
  ∃! A : ℕ, A < 10 ∧ is_prime (number_form A) ∧ number_form A = 305201 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_form_l3262_326266


namespace NUMINAMATH_CALUDE_problem_solution_l3262_326272

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3262_326272


namespace NUMINAMATH_CALUDE_square_sum_from_system_l3262_326203

theorem square_sum_from_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (14400 - 4056) / 169 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_system_l3262_326203


namespace NUMINAMATH_CALUDE_melany_money_theorem_l3262_326256

/-- The amount of money Melany initially had to fence a square field --/
def melany_initial_money (field_size : ℕ) (wire_cost_per_foot : ℕ) (unfenced_length : ℕ) : ℕ :=
  (field_size - unfenced_length) * wire_cost_per_foot

/-- Theorem stating that Melany's initial money was $120,000 --/
theorem melany_money_theorem (field_size : ℕ) (wire_cost_per_foot : ℕ) (unfenced_length : ℕ) 
  (h1 : field_size = 5000)
  (h2 : wire_cost_per_foot = 30)
  (h3 : unfenced_length = 1000) :
  melany_initial_money field_size wire_cost_per_foot unfenced_length = 120000 := by
  sorry

end NUMINAMATH_CALUDE_melany_money_theorem_l3262_326256


namespace NUMINAMATH_CALUDE_quadratic_always_has_two_roots_find_m_value_l3262_326205

/-- Given quadratic equation x^2 - (2m+1)x + m - 2 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 - (2*m+1)*x + m - 2 = 0

theorem quadratic_always_has_two_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

theorem find_m_value :
  ∃ m : ℝ, m = 6/5 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation m x₁ ∧ 
    quadratic_equation m x₂ ∧
    x₁ + x₂ + 3*x₁*x₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_has_two_roots_find_m_value_l3262_326205


namespace NUMINAMATH_CALUDE_water_and_bottle_weights_l3262_326287

/-- The weight of one cup of water in grams -/
def cup_weight : ℝ := 80

/-- The weight of one empty bottle in grams -/
def bottle_weight : ℝ := 200

/-- The total weight of 3 cups of water and 1 empty bottle in grams -/
def weight_3cups_1bottle : ℝ := 440

/-- The total weight of 5 cups of water and 1 empty bottle in grams -/
def weight_5cups_1bottle : ℝ := 600

theorem water_and_bottle_weights :
  (3 * cup_weight + bottle_weight = weight_3cups_1bottle) ∧
  (5 * cup_weight + bottle_weight = weight_5cups_1bottle) := by
  sorry

end NUMINAMATH_CALUDE_water_and_bottle_weights_l3262_326287


namespace NUMINAMATH_CALUDE_units_digit_problem_l3262_326241

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_problem : units_digit (8 * 19 * 1978 - 8^3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3262_326241


namespace NUMINAMATH_CALUDE_orthocenter_from_circumcenter_l3262_326293

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a point is the circumcenter of a triangle -/
def isCircumcenter (O : Point3D) (A B C : Point3D) : Prop := sorry

/-- Checks if a point is the orthocenter of a triangle -/
def isOrthocenter (H : Point3D) (A B C : Point3D) : Prop := sorry

/-- Checks if a sphere is inscribed in a tetrahedron -/
def isInscribed (s : Sphere) (t : Tetrahedron) : Prop := sorry

/-- Checks if a sphere touches a plane at a point -/
def touchesPlaneAt (s : Sphere) (p : Point3D) : Prop := sorry

/-- Checks if a sphere touches the planes of the other faces of a tetrahedron externally -/
def touchesOtherFacesExternally (s : Sphere) (t : Tetrahedron) : Prop := sorry

theorem orthocenter_from_circumcenter 
  (t : Tetrahedron) 
  (s1 s2 : Sphere) 
  (H O : Point3D) :
  isInscribed s1 t →
  touchesPlaneAt s1 H →
  touchesPlaneAt s2 O →
  touchesOtherFacesExternally s2 t →
  isCircumcenter O t.A t.B t.C →
  isOrthocenter H t.A t.B t.C := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_from_circumcenter_l3262_326293


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3262_326221

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3262_326221


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3262_326218

theorem two_numbers_with_specific_means : ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1 / a + 1 / b) = 2 ∧
  a = (5 + Real.sqrt 5) / 2 ∧ 
  b = (5 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3262_326218


namespace NUMINAMATH_CALUDE_carpenter_square_problem_l3262_326249

theorem carpenter_square_problem (s : ℝ) :
  (s^2 - 4 * (0.09 * s^2) = 256) → s = 20 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_square_problem_l3262_326249


namespace NUMINAMATH_CALUDE_equation_proof_l3262_326225

theorem equation_proof : (100 - 6) * 7 - 52 + 8 + 9 = 623 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3262_326225


namespace NUMINAMATH_CALUDE_total_is_700_l3262_326242

/-- The number of magazines Marie sold -/
def magazines : ℕ := 425

/-- The number of newspapers Marie sold -/
def newspapers : ℕ := 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := magazines + newspapers

/-- Proof that the total number of reading materials sold is 700 -/
theorem total_is_700 : total_reading_materials = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_is_700_l3262_326242


namespace NUMINAMATH_CALUDE_fraction_equality_l3262_326269

/-- Given that (Bx-13)/(x^2-7x+10) = A/(x-2) + 5/(x-5) for all x ≠ 2 and x ≠ 5,
    prove that A = 3/5, B = 28/5, and A + B = 31/5 -/
theorem fraction_equality (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 2 → x ≠ 5 → (B * x - 13) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) →
  A = 3/5 ∧ B = 28/5 ∧ A + B = 31/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3262_326269


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3262_326259

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1 : ℚ) * d)

theorem first_term_of_arithmetic_sequence :
  ∃ (a d : ℚ),
    sum_arithmetic_sequence a d 30 = 300 ∧
    sum_arithmetic_sequence (arithmetic_sequence a d 31) d 40 = 2200 ∧
    a = -121 / 14 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3262_326259


namespace NUMINAMATH_CALUDE_factorial_236_trailing_zeros_l3262_326238

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 236! has 57 trailing zeros -/
theorem factorial_236_trailing_zeros :
  trailingZeros 236 = 57 := by sorry

end NUMINAMATH_CALUDE_factorial_236_trailing_zeros_l3262_326238
