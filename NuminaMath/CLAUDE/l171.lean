import Mathlib

namespace abc_inequality_l171_17172

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a * b * c) / (a + b + c) ≤ 4 / 3 := by
  sorry

end abc_inequality_l171_17172


namespace daughters_age_l171_17178

theorem daughters_age (father_age : ℕ) (daughter_age : ℕ) : 
  father_age = 40 → 
  father_age = 4 * daughter_age → 
  father_age + 20 = 2 * (daughter_age + 20) → 
  daughter_age = 10 := by
sorry

end daughters_age_l171_17178


namespace cube_isosceles_right_probability_l171_17114

/-- A cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- A triangle formed by 3 vertices of a cube -/
structure CubeTriangle :=
  (v1 v2 v3 : Fin 8)
  (distinct : v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3)

/-- An isosceles right triangle on a cube face -/
def IsIsoscelesRight (t : CubeTriangle) : Prop :=
  sorry

/-- The number of isosceles right triangles that can be formed on a cube -/
def numIsoscelesRight : ℕ := 24

/-- The total number of ways to select 3 vertices from 8 -/
def totalTriangles : ℕ := 56

/-- The probability of forming an isosceles right triangle -/
def probabilityIsoscelesRight : ℚ := 3/7

theorem cube_isosceles_right_probability :
  (numIsoscelesRight : ℚ) / totalTriangles = probabilityIsoscelesRight :=
sorry

end cube_isosceles_right_probability_l171_17114


namespace colored_copies_correct_l171_17136

/-- The number of colored copies Sandy made, given that:
  * Colored copies cost 10 cents each
  * White copies cost 5 cents each
  * Sandy made 400 copies in total
  * The total bill was $22.50 -/
def colored_copies : ℕ :=
  let colored_cost : ℚ := 10 / 100  -- 10 cents in dollars
  let white_cost : ℚ := 5 / 100     -- 5 cents in dollars
  let total_copies : ℕ := 400
  let total_bill : ℚ := 45 / 2      -- $22.50 as a rational number
  50  -- The actual value to be proven

theorem colored_copies_correct :
  let colored_cost : ℚ := 10 / 100
  let white_cost : ℚ := 5 / 100
  let total_copies : ℕ := 400
  let total_bill : ℚ := 45 / 2
  ∃ (white_copies : ℕ),
    colored_copies + white_copies = total_copies ∧
    colored_cost * colored_copies + white_cost * white_copies = total_bill :=
by sorry

end colored_copies_correct_l171_17136


namespace fixed_point_parabola_l171_17135

theorem fixed_point_parabola (d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 5 * x^2 + d * x + 3 * d
  f (-3) = 45 := by sorry

end fixed_point_parabola_l171_17135


namespace father_picked_22_8_pounds_l171_17177

/-- Represents the amount of strawberries picked by each person in pounds -/
structure StrawberryPicking where
  marco : ℝ
  sister : ℝ
  father : ℝ

/-- Converts kilograms to pounds -/
def kg_to_pounds (kg : ℝ) : ℝ := kg * 2.2

/-- Calculates the amount of strawberries picked by each person -/
def strawberry_picking : StrawberryPicking :=
  let marco_pounds := 1 + kg_to_pounds 3
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  { marco := marco_pounds,
    sister := sister_pounds,
    father := father_pounds }

/-- Theorem stating that the father picked 22.8 pounds of strawberries -/
theorem father_picked_22_8_pounds :
  strawberry_picking.father = 22.8 := by
  sorry

end father_picked_22_8_pounds_l171_17177


namespace perpendicular_equivalence_l171_17115

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_equivalence
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_non_coincident : m ≠ n)
  (h_m_perp_α : perp m α)
  (h_m_perp_β : perp m β) :
  perp n α ↔ perp n β :=
sorry

end perpendicular_equivalence_l171_17115


namespace fair_coin_four_flips_at_least_two_tails_l171_17131

/-- The probability of getting exactly k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting at least 2 but not more than 4 tails in 4 flips of a fair coin -/
theorem fair_coin_four_flips_at_least_two_tails : 
  (binomial_probability 4 2 0.5 + binomial_probability 4 3 0.5 + binomial_probability 4 4 0.5) = 0.6875 := by
  sorry

end fair_coin_four_flips_at_least_two_tails_l171_17131


namespace subset_implies_a_range_l171_17145

/-- The solution set of x^2 - 5x + 4 < 0 is a subset of x^2 - (a+5)x + 5a < 0 -/
def subset_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 5*x + 4 < 0 → x^2 - (a+5)*x + 5*a < 0

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≤ 1

/-- Theorem stating the relationship between the subset condition and the range of a -/
theorem subset_implies_a_range :
  ∀ a : ℝ, subset_condition a → a_range a :=
sorry

end subset_implies_a_range_l171_17145


namespace surface_area_order_l171_17108

/-- Represents the types of geometric solids -/
inductive Solid
  | Tetrahedron
  | Cube
  | Octahedron
  | Sphere
  | Cylinder
  | Cone

/-- Computes the surface area of a solid given its volume -/
noncomputable def surfaceArea (s : Solid) (v : ℝ) : ℝ :=
  match s with
  | Solid.Tetrahedron => (216 * Real.sqrt 3) ^ (1/3) * v ^ (2/3)
  | Solid.Cube => 6 * v ^ (2/3)
  | Solid.Octahedron => (108 * Real.sqrt 3) ^ (1/3) * v ^ (2/3)
  | Solid.Sphere => (36 * Real.pi) ^ (1/3) * v ^ (2/3)
  | Solid.Cylinder => (54 * Real.pi) ^ (1/3) * v ^ (2/3)
  | Solid.Cone => (81 * Real.pi) ^ (1/3) * v ^ (2/3)

/-- Theorem stating the order of surface areas for equal volume solids -/
theorem surface_area_order (v : ℝ) (h : v > 0) :
  surfaceArea Solid.Sphere v < surfaceArea Solid.Cylinder v ∧
  surfaceArea Solid.Cylinder v < surfaceArea Solid.Octahedron v ∧
  surfaceArea Solid.Octahedron v < surfaceArea Solid.Cube v ∧
  surfaceArea Solid.Cube v < surfaceArea Solid.Cone v ∧
  surfaceArea Solid.Cone v < surfaceArea Solid.Tetrahedron v :=
by
  sorry

end surface_area_order_l171_17108


namespace hyperbola_x_axis_m_range_l171_17184

/-- Represents the equation of a conic section -/
structure ConicSection where
  m : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x^2 / m + y^2 / (m - 4) = 1

/-- Represents a hyperbola with foci on the x-axis -/
class HyperbolaXAxis extends ConicSection

/-- The range of m for a hyperbola with foci on the x-axis -/
def is_valid_m (m : ℝ) : Prop := 0 < m ∧ m < 4

/-- Theorem stating the condition for m to represent a hyperbola with foci on the x-axis -/
theorem hyperbola_x_axis_m_range (h : HyperbolaXAxis) :
  is_valid_m h.m :=
sorry

end hyperbola_x_axis_m_range_l171_17184


namespace valid_numbers_l171_17146

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 1 ∧ n % 5 = 3

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {13, 28, 43, 58, 73, 88} :=
by sorry

end valid_numbers_l171_17146


namespace eldoras_purchase_cost_is_55_40_l171_17171

/-- The cost of Eldora's purchase of paper clips and index cards -/
def eldoras_purchase_cost (index_card_price : ℝ) : ℝ :=
  15 * 1.85 + 7 * index_card_price

/-- The cost of Finn's purchase of paper clips and index cards -/
def finns_purchase_cost (index_card_price : ℝ) : ℝ :=
  12 * 1.85 + 10 * index_card_price

/-- Theorem stating the cost of Eldora's purchase -/
theorem eldoras_purchase_cost_is_55_40 :
  ∃ (index_card_price : ℝ),
    finns_purchase_cost index_card_price = 61.70 ∧
    eldoras_purchase_cost index_card_price = 55.40 := by
  sorry

end eldoras_purchase_cost_is_55_40_l171_17171


namespace candy_remaining_l171_17187

theorem candy_remaining (initial_candy : ℕ) (people : ℕ) (eaten_per_person : ℕ) 
  (h1 : initial_candy = 68) 
  (h2 : people = 2) 
  (h3 : eaten_per_person = 4) : 
  initial_candy - (people * eaten_per_person) = 60 := by
  sorry

end candy_remaining_l171_17187


namespace local_monotonicity_not_implies_global_l171_17116

/-- A function that satisfies the local monotonicity condition but is not globally monotonic -/
def exists_locally_monotonic_not_globally : Prop :=
  ∃ (f : ℝ → ℝ), 
    (∀ a : ℝ, ∃ b : ℝ, b > a ∧ (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y ∨ f x ≥ f y)) ∧
    ¬(∀ x y : ℝ, x < y → f x ≤ f y ∨ f x ≥ f y)

theorem local_monotonicity_not_implies_global : exists_locally_monotonic_not_globally :=
sorry

end local_monotonicity_not_implies_global_l171_17116


namespace circle_land_theorem_l171_17138

/-- Represents a digit with its associated number of circles in Circle Land notation -/
structure CircleLandDigit where
  digit : Nat
  circles : Nat

/-- Calculates the value of a CircleLandDigit in the Circle Land number system -/
def circleValue (d : CircleLandDigit) : Nat :=
  d.digit * (10 ^ d.circles)

/-- Represents a number in Circle Land notation as a list of CircleLandDigits -/
def CircleLandNumber := List CircleLandDigit

/-- Calculates the value of a CircleLandNumber -/
def circleLandValue (n : CircleLandNumber) : Nat :=
  n.foldl (fun acc d => acc + circleValue d) 0

/-- The Circle Land representation of the number in the problem -/
def problemNumber : CircleLandNumber :=
  [⟨3, 4⟩, ⟨1, 2⟩, ⟨5, 0⟩]

theorem circle_land_theorem :
  circleLandValue problemNumber = 30105 := by sorry

end circle_land_theorem_l171_17138


namespace range_of_a_l171_17102

noncomputable def f (x : ℝ) : ℝ := 2 * x + (Real.exp x)⁻¹ - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) :
  a ≤ -1 ∨ a ≥ 1/2 :=
sorry

end range_of_a_l171_17102


namespace height_estimate_theorem_l171_17107

/-- Represents the regression line for estimating height from foot length -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Represents the sample statistics -/
structure SampleStats where
  mean_x : ℝ
  mean_y : ℝ

/-- Calculates the estimated height given a foot length and regression line -/
def estimate_height (x : ℝ) (line : RegressionLine) : ℝ :=
  line.slope * x + line.intercept

/-- Theorem stating that given the sample statistics and slope, 
    the estimated height for a foot length of 24 cm is 166 cm -/
theorem height_estimate_theorem 
  (stats : SampleStats) 
  (given_slope : ℝ) 
  (h_mean_x : stats.mean_x = 22.5) 
  (h_mean_y : stats.mean_y = 160) 
  (h_slope : given_slope = 4) :
  let line := RegressionLine.mk given_slope (stats.mean_y - given_slope * stats.mean_x)
  estimate_height 24 line = 166 := by
  sorry

#check height_estimate_theorem

end height_estimate_theorem_l171_17107


namespace exists_function_with_property_l171_17142

def apply_n_times (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (apply_n_times f n)

theorem exists_function_with_property : 
  ∃ (f : ℝ → ℝ), 
    (∀ x : ℝ, x ≥ 0 → f x ≥ 0) ∧ 
    (∀ x : ℝ, x ≥ 0 → apply_n_times f 45 x = 1 + x + 2 * Real.sqrt x) :=
  sorry

end exists_function_with_property_l171_17142


namespace bijection_and_size_equivalence_l171_17151

/-- Represents an integer grid -/
def IntegerGrid := ℤ → ℤ → ℤ

/-- Represents a plane partition -/
def PlanePartition := ℕ → ℕ → ℕ

/-- The size of a plane partition -/
def size (pp : PlanePartition) : ℕ := sorry

/-- The bijection between integer grids and plane partitions -/
def grid_to_partition (g : IntegerGrid) : PlanePartition := sorry

/-- The inverse bijection from plane partitions to integer grids -/
def partition_to_grid (pp : PlanePartition) : IntegerGrid := sorry

/-- The sum of integers in a grid, counting k times for k-th highest diagonal -/
def weighted_sum (g : IntegerGrid) : ℤ := sorry

theorem bijection_and_size_equivalence :
  ∃ (f : IntegerGrid → PlanePartition) (g : PlanePartition → IntegerGrid),
    (∀ grid, g (f grid) = grid) ∧
    (∀ partition, f (g partition) = partition) ∧
    (∀ grid, size (f grid) = weighted_sum grid) := by
  sorry

end bijection_and_size_equivalence_l171_17151


namespace paintings_per_room_l171_17111

theorem paintings_per_room (total_paintings : ℕ) (num_rooms : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : num_rooms = 4) 
  (h3 : total_paintings % num_rooms = 0) : 
  total_paintings / num_rooms = 8 := by
sorry

end paintings_per_room_l171_17111


namespace tangent_line_at_zero_l171_17176

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) + 1

theorem tangent_line_at_zero : 
  let p : ℝ × ℝ := (0, f 0)
  let m : ℝ := -((deriv f) 0)
  ∀ x y : ℝ, (y - p.2 = m * (x - p.1)) ↔ (x + y - 2 = 0) :=
by sorry

end tangent_line_at_zero_l171_17176


namespace tree_height_difference_l171_17180

def maple_height : ℚ := 10 + 3/4
def pine_height : ℚ := 12 + 7/8

theorem tree_height_difference :
  pine_height - maple_height = 2 + 1/8 := by sorry

end tree_height_difference_l171_17180


namespace average_of_solutions_is_zero_l171_17139

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (s₁ s₂ : ℝ), s₁ ∈ solutions ∧ s₂ ∈ solutions ∧ s₁ ≠ s₂ ∧
    (s₁ + s₂) / 2 = 0 ∧
    ∀ (s : ℝ), s ∈ solutions → s = s₁ ∨ s = s₂ :=
by sorry

end average_of_solutions_is_zero_l171_17139


namespace square_units_digit_nine_l171_17157

theorem square_units_digit_nine (n : ℕ) : n ≤ 9 → (n^2 % 10 = 9 ↔ n = 3 ∨ n = 7) := by
  sorry

end square_units_digit_nine_l171_17157


namespace inequality_proof_l171_17179

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : x - Real.sqrt x ≤ y - 1/4 ∧ y - 1/4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1/4 ∧ x - 1/4 ≤ y + Real.sqrt y := by
  sorry

end inequality_proof_l171_17179


namespace geometric_sequence_sum_l171_17148

/-- Given a geometric sequence {aₙ} with a₁ > 0 and a₂a₄ + 2a₃a₅ + a₄a₆ = 36, prove that a₃ + a₅ = 6 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
    (h_pos : a 1 > 0) (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
    a 3 + a 5 = 6 := by
  sorry

end geometric_sequence_sum_l171_17148


namespace chord_length_line_ellipse_intersection_l171_17130

/-- The length of the chord formed by the intersection of a line and an ellipse -/
theorem chord_length_line_ellipse_intersection :
  let line : ℝ → ℝ × ℝ := λ t ↦ (1 + t, -2 + t)
  let ellipse : ℝ × ℝ → Prop := λ p ↦ p.1^2 + 2*p.2^2 = 8
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ t₁, line t₁ = A) ∧ 
    (∃ t₂, line t₂ = B) ∧
    ellipse A ∧ 
    ellipse B ∧
    dist A B = 4 * Real.sqrt 3 / 3 :=
by sorry


end chord_length_line_ellipse_intersection_l171_17130


namespace six_pairs_l171_17198

/-- The number of distinct pairs of integers (x, y) satisfying the conditions -/
def num_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    0 < p.1 ∧ p.1 < p.2 ∧ p.1 * p.2 = 2025
  ) (Finset.product (Finset.range 2026) (Finset.range 2026))).card

/-- Theorem stating that there are exactly 6 pairs satisfying the conditions -/
theorem six_pairs : num_pairs = 6 := by
  sorry

end six_pairs_l171_17198


namespace language_course_enrollment_l171_17110

theorem language_course_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
  total = 150 →
  french = 58 →
  german = 40 →
  spanish = 35 →
  french_german = 20 →
  french_spanish = 15 →
  german_spanish = 10 →
  all_three = 5 →
  total - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 62 :=
by sorry

end language_course_enrollment_l171_17110


namespace abs_eq_self_not_negative_l171_17185

theorem abs_eq_self_not_negative (x : ℝ) : |x| = x → x ≥ 0 := by
  sorry

end abs_eq_self_not_negative_l171_17185


namespace isosceles_right_triangle_l171_17154

open Real

theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  log a - log c = log (sin B) →
  log (sin B) = -log (sqrt 2) →
  B < π / 2 →
  a = b ∧ C = π / 2 :=
by sorry

end isosceles_right_triangle_l171_17154


namespace inequality_proof_l171_17105

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end inequality_proof_l171_17105


namespace x_value_l171_17168

theorem x_value : ∃ x : ℝ, (49 / 49 = x ^ 4) → x = 1 := by
  sorry

end x_value_l171_17168


namespace area_outside_smaller_squares_l171_17165

theorem area_outside_smaller_squares (larger_side : ℝ) (smaller_side : ℝ) : 
  larger_side = 10 → 
  smaller_side = 4 → 
  larger_side^2 - 2 * smaller_side^2 = 68 := by
sorry

end area_outside_smaller_squares_l171_17165


namespace sum_of_53_odd_numbers_l171_17106

theorem sum_of_53_odd_numbers : 
  (Finset.range 53).sum (fun n => 2 * n + 1) = 2809 := by
  sorry

end sum_of_53_odd_numbers_l171_17106


namespace inequality_of_positive_reals_l171_17159

theorem inequality_of_positive_reals (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  (a * b) / (a + b) + (c * d) / (c + d) + (e * f) / (e + f) ≤ 
  ((a + c + e) * (b + d + f)) / (a + b + c + d + e + f) := by
  sorry

end inequality_of_positive_reals_l171_17159


namespace f_min_value_f_attains_min_l171_17181

def f (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem f_min_value : ∀ x : ℝ, f x ≥ 2 := by sorry

theorem f_attains_min : ∃ x : ℝ, f x = 2 := by sorry

end f_min_value_f_attains_min_l171_17181


namespace uphill_distance_l171_17160

/-- Proves that given specific conditions, the uphill distance traveled by a car is 100 km. -/
theorem uphill_distance (uphill_speed downhill_speed downhill_distance average_speed : ℝ) 
  (h1 : uphill_speed = 30)
  (h2 : downhill_speed = 60)
  (h3 : downhill_distance = 50)
  (h4 : average_speed = 36) : 
  ∃ uphill_distance : ℝ, 
    uphill_distance = 100 ∧ 
    average_speed = (uphill_distance + downhill_distance) / (uphill_distance / uphill_speed + downhill_distance / downhill_speed) := by
  sorry

end uphill_distance_l171_17160


namespace factorial_equation_solutions_l171_17162

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 2^x + 5^y + 63 = z.factorial → 
    ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) := by
  sorry

end factorial_equation_solutions_l171_17162


namespace coefficient_x3y5_in_expansion_l171_17152

theorem coefficient_x3y5_in_expansion (x y : ℝ) :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * x^k * y^(8-k)) =
  56 * x^3 * y^5 + (Finset.range 9).sum (fun k => if k ≠ 3 then Nat.choose 8 k * x^k * y^(8-k) else 0) :=
by sorry

end coefficient_x3y5_in_expansion_l171_17152


namespace johns_brother_age_l171_17132

theorem johns_brother_age :
  ∀ (john_age brother_age : ℕ),
  john_age = 6 * brother_age - 4 →
  john_age + brother_age = 10 →
  brother_age = 2 := by
sorry

end johns_brother_age_l171_17132


namespace sequence_third_term_l171_17123

theorem sequence_third_term (a : ℕ → ℕ) (h : ∀ n, a n = n^2 + n) : a 3 = 12 := by
  sorry

end sequence_third_term_l171_17123


namespace min_additional_coins_for_alex_l171_17140

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := friends * (friends + 1) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins_for_alex : 
  min_additional_coins 15 90 = 30 := by
  sorry

end min_additional_coins_for_alex_l171_17140


namespace reciprocal_of_negative_two_l171_17195

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end reciprocal_of_negative_two_l171_17195


namespace remainder_3_100_mod_7_l171_17190

theorem remainder_3_100_mod_7 : 3^100 % 7 = 4 := by
  sorry

end remainder_3_100_mod_7_l171_17190


namespace polynomial_division_remainder_l171_17112

theorem polynomial_division_remainder :
  ∃ Q : Polynomial ℝ, (X : Polynomial ℝ)^5 - 3 * X^3 + 4 * X + 5 = 
  (X - 3)^2 * Q + (261 * X - 643) := by sorry

end polynomial_division_remainder_l171_17112


namespace junior_score_theorem_l171_17169

theorem junior_score_theorem (n : ℝ) (h : n > 0) :
  let junior_ratio : ℝ := 0.2
  let senior_ratio : ℝ := 0.8
  let total_average : ℝ := 80
  let senior_average : ℝ := 78
  let junior_count : ℝ := junior_ratio * n
  let senior_count : ℝ := senior_ratio * n
  let total_score : ℝ := total_average * n
  let senior_total_score : ℝ := senior_average * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 88 :=
by sorry

end junior_score_theorem_l171_17169


namespace minimum_point_of_translated_absolute_value_function_l171_17155

def f (x : ℝ) := |x + 2| - 6

theorem minimum_point_of_translated_absolute_value_function :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = -2 ∧ f x₀ = -6 :=
sorry

end minimum_point_of_translated_absolute_value_function_l171_17155


namespace quadratic_equation_with_given_roots_l171_17127

theorem quadratic_equation_with_given_roots :
  ∀ (a b c : ℝ), a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = -2 ∨ x = 3) →
  a * x^2 + b * x + c = x^2 - x - 6 := by
  sorry

end quadratic_equation_with_given_roots_l171_17127


namespace election_percentage_l171_17158

theorem election_percentage (total_votes : ℝ) (candidate_votes : ℝ) 
  (h1 : candidate_votes > 0)
  (h2 : total_votes > candidate_votes)
  (h3 : candidate_votes + (1/3) * (total_votes - candidate_votes) = (1/2) * total_votes) :
  candidate_votes / total_votes = 1/4 := by
sorry

end election_percentage_l171_17158


namespace solve_equation_l171_17137

theorem solve_equation (n : ℚ) : (1 / (2 * n)) + (1 / (4 * n)) = 3 / 12 → n = 3 := by
  sorry

end solve_equation_l171_17137


namespace min_value_x_plus_y_l171_17199

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = x * y) :
  x + y ≥ 9 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + y₀ = x₀ * y₀ ∧ x₀ + y₀ = 9 :=
by sorry

end min_value_x_plus_y_l171_17199


namespace unique_five_digit_numbers_l171_17128

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Checks if a number starts with a specific digit -/
def starts_with (n : FiveDigitNumber) (d : ℕ) : Prop :=
  n.val / 10000 = d

/-- Moves the first digit of a number to the last position -/
def move_first_to_last (n : FiveDigitNumber) : ℕ :=
  (n.val % 10000) * 10 + (n.val / 10000)

/-- The main theorem stating the unique solution to the problem -/
theorem unique_five_digit_numbers :
  ∃! (n₁ n₂ : FiveDigitNumber),
    starts_with n₁ 2 ∧
    starts_with n₂ 4 ∧
    move_first_to_last n₁ = n₁.val + n₂.val ∧
    move_first_to_last n₂ = n₁.val - n₂.val ∧
    n₁.val = 26829 ∧
    n₂.val = 41463 := by
  sorry


end unique_five_digit_numbers_l171_17128


namespace floor_a4_div_a3_l171_17134

def a (k : ℕ) : ℕ := Nat.choose 100 (k + 1)

theorem floor_a4_div_a3 : ⌊(a 4 : ℚ) / (a 3 : ℚ)⌋ = 19 := by sorry

end floor_a4_div_a3_l171_17134


namespace grape_rate_calculation_l171_17144

theorem grape_rate_calculation (grape_weight : ℕ) (mango_weight : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grape_weight = 8 →
  mango_weight = 9 →
  mango_rate = 55 →
  total_paid = 1055 →
  ∃ (grape_rate : ℕ), grape_rate * grape_weight + mango_rate * mango_weight = total_paid ∧ grape_rate = 70 :=
by
  sorry

end grape_rate_calculation_l171_17144


namespace sum_of_powers_of_i_equals_zero_l171_17124

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end sum_of_powers_of_i_equals_zero_l171_17124


namespace wickets_before_last_match_value_l171_17166

/-- The number of wickets taken by a bowler before his last match -/
def wickets_before_last_match (initial_average : ℚ) (wickets_last_match : ℕ) 
  (runs_last_match : ℕ) (average_decrease : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of wickets taken before the last match -/
theorem wickets_before_last_match_value :
  wickets_before_last_match 12.4 3 26 0.4 = 25 := by
  sorry

end wickets_before_last_match_value_l171_17166


namespace powers_of_two_difference_divisible_by_1987_l171_17141

theorem powers_of_two_difference_divisible_by_1987 :
  ∃ a b : ℕ, 0 ≤ a ∧ a < b ∧ b ≤ 1987 ∧ (2^b - 2^a) % 1987 = 0 := by
  sorry

end powers_of_two_difference_divisible_by_1987_l171_17141


namespace completing_square_quadratic_l171_17113

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 4*x - 11 = 0) ↔ ((x - 2)^2 = 15) :=
by sorry

end completing_square_quadratic_l171_17113


namespace sufficient_not_necessary_l171_17101

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b, a > 0 → b > 0 → a * b < 3 → 1 / a + 4 / b > 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 1 / a + 4 / b > 2 ∧ a * b ≥ 3) :=
by sorry

end sufficient_not_necessary_l171_17101


namespace ellipse_chord_slope_l171_17147

/-- Given an ellipse with equation x²/16 + y²/9 = 1, 
    the slope of any chord with midpoint (1,2) is -9/32 -/
theorem ellipse_chord_slope :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 1) →
  ((y₁ + y₂) / 2 = 2) →
  (y₂ - y₁) / (x₂ - x₁) = -9/32 :=
by sorry

end ellipse_chord_slope_l171_17147


namespace certain_number_proof_l171_17193

theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, 0.016 * x = 0.03408 ∧ x = 2.13 := by
  sorry

end certain_number_proof_l171_17193


namespace school_ball_dance_l171_17174

theorem school_ball_dance (b g : ℕ) : 
  (∀ n : ℕ, n ≤ b → n + 2 ≤ g) →  -- Each boy dances with at least 3 girls
  (b + 2 = g) →                   -- The last boy dances with all girls
  b = g - 2 := by
sorry

end school_ball_dance_l171_17174


namespace solution_distribution_l171_17175

def test_tube_volumes : List ℝ := [7, 4, 5, 4, 6, 8, 7, 3, 9, 6]
def num_beakers : ℕ := 5

theorem solution_distribution (volumes : List ℝ) (num_beakers : ℕ) 
  (h1 : volumes = test_tube_volumes) 
  (h2 : num_beakers = 5) : 
  (volumes.sum / num_beakers : ℝ) = 11.8 := by
  sorry

#check solution_distribution

end solution_distribution_l171_17175


namespace hayley_meatballs_l171_17163

/-- The number of meatballs Hayley has left after Kirsten stole some -/
def meatballs_left (initial : ℕ) (stolen : ℕ) : ℕ :=
  initial - stolen

/-- Theorem stating that Hayley has 11 meatballs left -/
theorem hayley_meatballs : meatballs_left 25 14 = 11 := by
  sorry

end hayley_meatballs_l171_17163


namespace movies_watched_undetermined_l171_17118

/-- Represents the "Crazy Silly School" series -/
structure CrazySillySchool where
  total_movies : ℕ
  total_books : ℕ
  books_read : ℕ
  movie_book_difference : ℕ

/-- The conditions of the problem -/
def series : CrazySillySchool :=
  { total_movies := 17
  , total_books := 11
  , books_read := 13
  , movie_book_difference := 6 }

/-- Predicate to check if the number of movies watched can be determined -/
def can_determine_movies_watched (s : CrazySillySchool) : Prop :=
  ∃! n : ℕ, n ≤ s.total_movies

/-- Theorem stating that it's impossible to determine the number of movies watched -/
theorem movies_watched_undetermined (s : CrazySillySchool) 
  (h1 : s.total_movies = s.total_books + s.movie_book_difference)
  (h2 : s.books_read ≤ s.total_books) :
  ¬(can_determine_movies_watched s) :=
sorry

end movies_watched_undetermined_l171_17118


namespace symmetric_circle_equation_l171_17173

/-- The standard equation of a circle symmetric to x^2 + y^2 = 1 with respect to x + y = 1 -/
theorem symmetric_circle_equation :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}
  let symmetric_circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}
  symmetric_circle = {p | ∃q ∈ C, p.1 + q.1 = 1 ∧ p.2 + q.2 = 1} := by
  sorry

end symmetric_circle_equation_l171_17173


namespace only_two_random_events_l171_17191

-- Define the universe of events
inductive Event : Type
| real_number_multiplication : Event
| draw_odd_numbered_ball : Event
| win_lottery : Event
| number_inequality : Event

-- Define a predicate for random events
def is_random_event : Event → Prop
| Event.real_number_multiplication => False
| Event.draw_odd_numbered_ball => True
| Event.win_lottery => True
| Event.number_inequality => False

-- Theorem statement
theorem only_two_random_events :
  (∀ e : Event, is_random_event e ↔ (e = Event.draw_odd_numbered_ball ∨ e = Event.win_lottery)) :=
by sorry

end only_two_random_events_l171_17191


namespace complex_number_in_second_quadrant_l171_17120

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + Complex.I
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l171_17120


namespace cosine_equality_l171_17119

theorem cosine_equality (a : ℝ) (h : Real.sin (π/3 + a) = 5/12) : 
  Real.cos (π/6 - a) = 5/12 := by sorry

end cosine_equality_l171_17119


namespace bank_coin_count_l171_17170

/-- The total number of coins turned in by a customer at a bank -/
def total_coins (dimes nickels quarters : ℕ) : ℕ :=
  dimes + nickels + quarters

/-- Theorem stating that the total number of coins is 11 given the specific quantities -/
theorem bank_coin_count : total_coins 2 2 7 = 11 := by
  sorry

end bank_coin_count_l171_17170


namespace probability_three_heads_in_eight_tosses_l171_17129

-- Define a fair coin toss
def fair_coin_toss : Type := Bool

-- Define the number of tosses
def num_tosses : Nat := 8

-- Define the number of heads we're looking for
def target_heads : Nat := 3

-- Define the probability of getting exactly 'target_heads' in 'num_tosses'
def probability_exact_heads : ℚ :=
  (Nat.choose num_tosses target_heads : ℚ) / (2 ^ num_tosses : ℚ)

-- Theorem statement
theorem probability_three_heads_in_eight_tosses :
  probability_exact_heads = 7 / 32 := by
  sorry

end probability_three_heads_in_eight_tosses_l171_17129


namespace min_soldiers_to_add_l171_17121

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  ∃ x : ℕ, x = 82 ∧ 
    (∀ y : ℕ, y < x → ¬((N + y) % 7 = 0 ∧ (N + y) % 12 = 0)) ∧
    (N + x) % 7 = 0 ∧ (N + x) % 12 = 0 := by
  sorry

end min_soldiers_to_add_l171_17121


namespace sum_of_x_and_y_is_two_l171_17100

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 := by
  sorry

end sum_of_x_and_y_is_two_l171_17100


namespace range_of_f_l171_17125

-- Define the function f
def f (x : ℝ) : ℝ := x + |x - 2|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 2 := by sorry

end range_of_f_l171_17125


namespace min_value_of_a_l171_17104

def matrixOp (a b c d : ℝ) : ℝ := a * d - b * c

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, matrixOp (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≥ -1/2 ∧ ∀ b, b < -1/2 → ∃ x, matrixOp (x - 1) (b - 2) (b + 1) x < 1 :=
by sorry

end min_value_of_a_l171_17104


namespace puppies_per_dog_l171_17164

theorem puppies_per_dog (num_dogs : ℕ) (total_puppies : ℕ) : 
  num_dogs = 15 → total_puppies = 75 → total_puppies / num_dogs = 5 := by
  sorry

end puppies_per_dog_l171_17164


namespace person_B_lap_time_l171_17153

/-- The time it takes for person B to complete a lap on a circular track -/
def time_B_lap : ℝ :=
  let time_A_lap : ℝ := 80  -- 1 minute and 20 seconds in seconds
  let meeting_interval : ℝ := 30
  48  -- The time we want to prove

theorem person_B_lap_time :
  let time_A_lap : ℝ := 80  -- 1 minute and 20 seconds in seconds
  let meeting_interval : ℝ := 30
  (1 / time_B_lap + 1 / time_A_lap) * meeting_interval = 1 ∧
  time_B_lap > 0 :=
by sorry

end person_B_lap_time_l171_17153


namespace camping_bowls_l171_17196

theorem camping_bowls (total_bowls : ℕ) (rice_per_person : ℚ) (dish_per_person : ℚ) (soup_per_person : ℚ) :
  total_bowls = 55 ∧ 
  rice_per_person = 1 ∧ 
  dish_per_person = 1/2 ∧ 
  soup_per_person = 1/3 →
  (total_bowls : ℚ) / (rice_per_person + dish_per_person + soup_per_person) = 30 := by
sorry

end camping_bowls_l171_17196


namespace cube_layer_removal_l171_17189

/-- Calculates the number of smaller cubes remaining inside a cube after removing layers to form a hollow cuboid --/
def remaining_cubes (original_size : Nat) (hollow_size : Nat) : Nat :=
  hollow_size^3 - (hollow_size - 2)^3

/-- Theorem stating that for a 12x12x12 cube with a 10x10x10 hollow cuboid, 488 smaller cubes remain --/
theorem cube_layer_removal :
  remaining_cubes 12 10 = 488 := by
  sorry

end cube_layer_removal_l171_17189


namespace positive_root_iff_p_in_set_l171_17188

-- Define the polynomial equation
def f (p x : ℝ) : ℝ := x^4 + 4*p*x^3 + x^2 + 4*p*x + 4

-- Define the set of p values
def P : Set ℝ := {p | p < -Real.sqrt 2 / 2 ∨ p > Real.sqrt 2 / 2}

-- Theorem statement
theorem positive_root_iff_p_in_set (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f p x = 0) ↔ p ∈ P :=
sorry

end positive_root_iff_p_in_set_l171_17188


namespace new_average_age_with_teacher_l171_17103

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℕ) 
  (h1 : num_students = 20) 
  (h2 : student_avg_age = 15) 
  (h3 : teacher_age = 36) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 16 := by
  sorry

end new_average_age_with_teacher_l171_17103


namespace negation_of_existence_quadratic_inequality_l171_17122

theorem negation_of_existence_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end negation_of_existence_quadratic_inequality_l171_17122


namespace cookie_difference_l171_17126

/-- Given that Alyssa has 129 cookies and Aiyanna has 140 cookies, 
    prove that Aiyanna has 11 more cookies than Alyssa. -/
theorem cookie_difference (alyssa_cookies : ℕ) (aiyanna_cookies : ℕ) 
    (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
    aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end cookie_difference_l171_17126


namespace family_strawberry_picking_l171_17150

/-- The total weight of strawberries picked by a family -/
theorem family_strawberry_picking (marco_weight dad_weight mom_weight sister_weight : ℕ) 
  (h1 : marco_weight = 8)
  (h2 : dad_weight = 32)
  (h3 : mom_weight = 22)
  (h4 : sister_weight = 14) :
  marco_weight + dad_weight + mom_weight + sister_weight = 76 := by
  sorry

#check family_strawberry_picking

end family_strawberry_picking_l171_17150


namespace equation_equivalence_l171_17182

theorem equation_equivalence (a b c : ℕ) 
  (ha : 0 < a ∧ a < 12) 
  (hb : 0 < b ∧ b < 12) 
  (hc : 0 < c ∧ c < 12) : 
  ((12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) ↔ (b + c = 12) :=
by sorry

end equation_equivalence_l171_17182


namespace jose_ducks_count_l171_17117

/-- Given that Jose has 28 chickens and 46 fowls in total, prove that he has 18 ducks. -/
theorem jose_ducks_count (chickens : ℕ) (total_fowls : ℕ) (ducks : ℕ) 
    (h1 : chickens = 28) 
    (h2 : total_fowls = 46) 
    (h3 : total_fowls = chickens + ducks) : 
  ducks = 18 := by
  sorry

end jose_ducks_count_l171_17117


namespace class_average_weight_l171_17197

theorem class_average_weight (students_A students_B : ℕ) (avg_weight_A avg_weight_B : ℝ) :
  students_A = 36 →
  students_B = 44 →
  avg_weight_A = 40 →
  avg_weight_B = 35 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B : ℝ) = 37.25 := by
  sorry

end class_average_weight_l171_17197


namespace parallel_line_point_slope_form_l171_17186

/-- Given points A, B, and C in the plane, this theorem states that the line passing through A
    and parallel to BC has the specified point-slope form. -/
theorem parallel_line_point_slope_form 
  (A B C : ℝ × ℝ) 
  (hA : A = (4, 6)) 
  (hB : B = (-3, -1)) 
  (hC : C = (5, -5)) : 
  ∃ (m : ℝ), m = -1/2 ∧ 
  ∀ (x y : ℝ), (y - 6 = m * (x - 4) ↔ 
    (∃ (t : ℝ), (x, y) = A + t • (C - B) ∧ (x, y) ≠ A)) :=
sorry

end parallel_line_point_slope_form_l171_17186


namespace temperature_peak_l171_17156

theorem temperature_peak (t : ℝ) : 
  (∀ s : ℝ, -s^2 + 10*s + 60 = 80 → s ≤ 5 + Real.sqrt 5) ∧ 
  (-((5 + Real.sqrt 5)^2) + 10*(5 + Real.sqrt 5) + 60 = 80) := by
sorry

end temperature_peak_l171_17156


namespace atMostTwoInPlaceFive_l171_17149

/-- The number of ways to arrange n people in n seats. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of seating arrangements where at most two people
    are in their numbered seats, given n people and n seats. -/
def atMostTwoInPlace (n : ℕ) : ℕ :=
  totalArrangements n - choose n 3 * totalArrangements (n - 3) - 1

theorem atMostTwoInPlaceFive :
  atMostTwoInPlace 5 = 109 := by sorry

end atMostTwoInPlaceFive_l171_17149


namespace chord_length_l171_17183

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length (x y : ℝ) : 
  let circle := (x - 1)^2 + y^2 = 4
  let line := x + y + 1 = 0
  let chord_length := Real.sqrt (8 : ℝ)
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    ((A.1 - 1)^2 + A.2^2 = 4) ∧ (A.1 + A.2 + 1 = 0) ∧
    ((B.1 - 1)^2 + B.2^2 = 4) ∧ (B.1 + B.2 + 1 = 0)) →
  ∃ A B : ℝ × ℝ, 
    ((A.1 - 1)^2 + A.2^2 = 4) ∧ (A.1 + A.2 + 1 = 0) ∧
    ((B.1 - 1)^2 + B.2^2 = 4) ∧ (B.1 + B.2 + 1 = 0) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length :=
by sorry

end chord_length_l171_17183


namespace divisors_not_div_by_3_eq_6_l171_17143

/-- The number of positive divisors of 180 that are not divisible by 3 -/
def divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 180 ∧ ¬(3 ∣ d)) (Finset.range 181)).card

/-- Theorem stating that the number of positive divisors of 180 not divisible by 3 is 6 -/
theorem divisors_not_div_by_3_eq_6 : divisors_not_div_by_3 = 6 := by
  sorry

end divisors_not_div_by_3_eq_6_l171_17143


namespace sin_2x_value_l171_17194

theorem sin_2x_value (x : ℝ) (h : Real.cos (x - π/4) = 4/5) : Real.sin (2*x) = 7/25 := by
  sorry

end sin_2x_value_l171_17194


namespace sequence_problem_l171_17133

/-- Given two sequences {a_n} and {b_n}, where:
    1) a_1 = 1
    2) {b_n} is a geometric sequence
    3) For all n, b_n = a_(n+1) / a_n
    4) b_10 * b_11 = 2016^(1/10)
    Prove that a_21 = 2016 -/
theorem sequence_problem (a b : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n)
  (h3 : ∀ n : ℕ, b n = a (n + 1) / a n)
  (h4 : b 10 * b 11 = 2016^(1/10)) :
  a 21 = 2016 := by
  sorry

end sequence_problem_l171_17133


namespace no_triangle_with_special_angles_l171_17161

theorem no_triangle_with_special_angles : 
  ¬ ∃ (α β γ : Real), 
    α + β + γ = Real.pi ∧ 
    ((3 * Real.cos α - 2) * (14 * Real.sin α ^ 2 + Real.sin (2 * α) - 12) = 0) ∧
    ((3 * Real.cos β - 2) * (14 * Real.sin β ^ 2 + Real.sin (2 * β) - 12) = 0) ∧
    ((3 * Real.cos γ - 2) * (14 * Real.sin γ ^ 2 + Real.sin (2 * γ) - 12) = 0) :=
by sorry

end no_triangle_with_special_angles_l171_17161


namespace base_10_to_base_2_l171_17192

theorem base_10_to_base_2 (n : Nat) (h : n = 123) :
  ∃ (a b c d e f g : Nat),
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 ∧
    n = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_10_to_base_2_l171_17192


namespace johns_earnings_l171_17109

/-- John's earnings over two weeks --/
theorem johns_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 20 →
  hours_week2 = 30 →
  extra_earnings = 102.75 →
  let hourly_wage := extra_earnings / (hours_week2 - hours_week1)
  let total_earnings := (hours_week1 + hours_week2) * hourly_wage
  total_earnings = 513.75 := by
  sorry

end johns_earnings_l171_17109


namespace cards_lost_l171_17167

theorem cards_lost (initial_cards remaining_cards : ℕ) : 
  initial_cards = 88 → remaining_cards = 18 → initial_cards - remaining_cards = 70 := by
  sorry

end cards_lost_l171_17167
