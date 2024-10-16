import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_multiple_l2681_268168

theorem reciprocal_multiple (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 8) (h3 : x + 8 = k * (1 / x)) : k = 128 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_multiple_l2681_268168


namespace NUMINAMATH_CALUDE_sum_lent_problem_l2681_268157

/-- Proves that given the conditions of the problem, the sum lent is 450 Rs. -/
theorem sum_lent_problem (P : ℝ) : 
  (P * 0.04 * 8 = P - 306) → P = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_problem_l2681_268157


namespace NUMINAMATH_CALUDE_f_difference_at_five_l2681_268145

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x + 3

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l2681_268145


namespace NUMINAMATH_CALUDE_papi_calot_plants_l2681_268173

/-- The number of plants Papi Calot needs to buy -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Theorem stating the total number of plants Papi Calot needs to buy -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_l2681_268173


namespace NUMINAMATH_CALUDE_range_of_m_l2681_268189

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 5) > 0}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x < m + 1}

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (B m ⊆ (Set.univ \ A)) → (-2 ≤ m ∧ m ≤ 4) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_range_of_m_l2681_268189


namespace NUMINAMATH_CALUDE_tangency_condition_iff_tangent_l2681_268129

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := y = 2 * x + 1

/-- The ellipse equation -/
def ellipse_eq (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The tangency condition -/
def tangency_condition (a b : ℝ) : Prop := 4 * a^2 + b^2 = 1

/-- Theorem stating that the tangency condition is necessary and sufficient -/
theorem tangency_condition_iff_tangent (a b : ℝ) :
  tangency_condition a b ↔ 
  ∃! x y : ℝ, line_eq x y ∧ ellipse_eq x y a b :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_iff_tangent_l2681_268129


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l2681_268181

/-- A parallelogram with consecutive side lengths 12, 5y-3, 3x+2, and 9 has x+y equal to 86/15 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (3*x + 2 = 12) → (5*y - 3 = 9) → x + y = 86/15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l2681_268181


namespace NUMINAMATH_CALUDE_part1_selection_count_part2_selection_count_l2681_268195

def num_male : ℕ := 4
def num_female : ℕ := 5
def total_selected : ℕ := 4

def combinations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem part1_selection_count : 
  combinations num_male 2 * combinations num_female 2 = 60 := by sorry

theorem part2_selection_count :
  let total_selections := combinations num_male 1 * combinations num_female 3 +
                          combinations num_male 2 * combinations num_female 2 +
                          combinations num_male 3 * combinations num_female 1
  let invalid_selections := combinations (num_male - 1) 2 +
                            combinations (num_female - 1) 1 * combinations (num_male - 1) 1 +
                            combinations (num_female - 1) 2
  total_selections - invalid_selections = 99 := by sorry

end NUMINAMATH_CALUDE_part1_selection_count_part2_selection_count_l2681_268195


namespace NUMINAMATH_CALUDE_triangle_base_length_l2681_268127

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = (base * height) / 2 → area = 12 → height = 6 → base = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2681_268127


namespace NUMINAMATH_CALUDE_petri_dishes_count_l2681_268167

/-- The number of petri dishes in the biology lab -/
def num_petri_dishes : ℕ :=
  10800

/-- The total number of germs in the lab -/
def total_germs : ℕ :=
  5400000

/-- The number of germs in a single dish -/
def germs_per_dish : ℕ :=
  500

/-- Theorem stating that the number of petri dishes is correct -/
theorem petri_dishes_count :
  num_petri_dishes = total_germs / germs_per_dish :=
by sorry

end NUMINAMATH_CALUDE_petri_dishes_count_l2681_268167


namespace NUMINAMATH_CALUDE_equal_areas_of_same_side_lengths_l2681_268118

/-- A polygon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  sides : Fin n → ℝ
  inscribed : Bool

/-- The area of an inscribed polygon -/
noncomputable def area (p : InscribedPolygon n) : ℝ := sorry

/-- Two polygons have the same set of side lengths -/
def same_side_lengths (p1 p2 : InscribedPolygon n) : Prop :=
  ∃ (σ : Equiv (Fin n) (Fin n)), ∀ i, p1.sides i = p2.sides (σ i)

theorem equal_areas_of_same_side_lengths (n : ℕ) (p1 p2 : InscribedPolygon n) 
  (h1 : p1.inscribed) (h2 : p2.inscribed) (h3 : same_side_lengths p1 p2) : 
  area p1 = area p2 := by sorry

end NUMINAMATH_CALUDE_equal_areas_of_same_side_lengths_l2681_268118


namespace NUMINAMATH_CALUDE_spatial_relationships_l2681_268110

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the perpendicular relationship between lines
variable (perpendicular_line : Line → Line → Prop)

theorem spatial_relationships 
  (m n : Line) (α β γ : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (perpendicular m α ∧ parallel_line_plane n α → perpendicular_line m n) ∧
  (parallel_plane α β ∧ parallel_plane β γ ∧ perpendicular m α → perpendicular m γ) :=
sorry

end NUMINAMATH_CALUDE_spatial_relationships_l2681_268110


namespace NUMINAMATH_CALUDE_pauls_supplies_l2681_268144

/-- Given Paul's initial and final crayon counts, and initial eraser count,
    prove the difference between remaining erasers and crayons. -/
theorem pauls_supplies (initial_crayons : ℕ) (initial_erasers : ℕ) (final_crayons : ℕ)
    (h1 : initial_crayons = 601)
    (h2 : initial_erasers = 406)
    (h3 : final_crayons = 336) :
    initial_erasers - final_crayons = 70 := by
  sorry

end NUMINAMATH_CALUDE_pauls_supplies_l2681_268144


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2681_268131

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 4 - y^2 = 1}
  let a : ℝ := 2  -- semi-major axis
  let b : ℝ := 1  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- focal distance
  let e : ℝ := c / a  -- eccentricity
  e = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2681_268131


namespace NUMINAMATH_CALUDE_power_of_seven_mod_ten_thousand_l2681_268139

theorem power_of_seven_mod_ten_thousand :
  7^2045 % 10000 = 6807 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_ten_thousand_l2681_268139


namespace NUMINAMATH_CALUDE_min_f_correct_a_range_condition_l2681_268182

noncomputable section

def f (a x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

def min_f (a : ℝ) : ℝ :=
  if a ≤ 1 then 1 - a
  else if a < Real.exp 1 then a - (a + 1) * Real.log a - 1
  else Real.exp 1 - (a + 1) - a / Real.exp 1

theorem min_f_correct (a : ℝ) :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ min_f a := by sorry

theorem a_range_condition (a : ℝ) :
  a < 1 →
  (∃ x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2),
    ∀ x₂ ∈ Set.Icc (-2) 0, f a x₁ < g x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a := by sorry

end

end NUMINAMATH_CALUDE_min_f_correct_a_range_condition_l2681_268182


namespace NUMINAMATH_CALUDE_no_pascal_row_with_four_distinct_elements_l2681_268197

theorem no_pascal_row_with_four_distinct_elements : 
  ¬ ∃ (n : ℕ) (k m : ℕ) (a b c d : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (b = Nat.choose n k) ∧
    (d = Nat.choose n m) ∧
    (a = b / 2) ∧
    (c = d / 2) :=
by sorry

end NUMINAMATH_CALUDE_no_pascal_row_with_four_distinct_elements_l2681_268197


namespace NUMINAMATH_CALUDE_abc_area_is_sqrt3_over_12_l2681_268122

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the points M, P, and O
def M (t : Triangle) : ℝ × ℝ := sorry
def P (t : Triangle) : ℝ × ℝ := sorry
def O (t : Triangle) : ℝ × ℝ := sorry

-- Define the similarity of triangles BOM and AOP
def triangles_similar (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist (t.B) (O t) / dist (t.A) (O t) = k ∧
    dist (O t) (M t) / dist (O t) (P t) = k ∧
    dist (t.B) (M t) / dist (t.A) (P t) = k

-- Define the condition BO = (1 + √3) OP
def bo_op_relation (t : Triangle) : Prop :=
  dist (t.B) (O t) = (1 + Real.sqrt 3) * dist (O t) (P t)

-- Define the condition BC = 1
def bc_length (t : Triangle) : Prop :=
  dist (t.B) (t.C) = 1

-- Define the area of the triangle
def triangle_area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem abc_area_is_sqrt3_over_12 (t : Triangle) 
  (h1 : triangles_similar t) 
  (h2 : bo_op_relation t) 
  (h3 : bc_length t) : 
  triangle_area t = Real.sqrt 3 / 12 := by sorry

end NUMINAMATH_CALUDE_abc_area_is_sqrt3_over_12_l2681_268122


namespace NUMINAMATH_CALUDE_joe_age_difference_l2681_268160

theorem joe_age_difference (joe_age : ℕ) (james_age : ℕ) : joe_age = 22 → 2 * (joe_age + 8) = 3 * (james_age + 8) → joe_age - james_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_joe_age_difference_l2681_268160


namespace NUMINAMATH_CALUDE_system_solution_l2681_268132

/-- Given a system of two linear equations in two variables,
    prove that the solution satisfies both equations. -/
theorem system_solution (x y : ℚ) : 
  x = 14 ∧ y = 29/5 →
  -x + 5*y = 15 ∧ 4*x - 10*y = -2 := by sorry

end NUMINAMATH_CALUDE_system_solution_l2681_268132


namespace NUMINAMATH_CALUDE_triangle_inequality_third_side_length_l2681_268149

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (a < b + c ∧ b < c + a ∧ c < a + b) :=
sorry

theorem third_side_length (x : ℝ) : 
  x > 0 → 
  5 > 0 → 
  8 > 0 → 
  (5 + 8 > x ∧ 8 + x > 5 ∧ x + 5 > 8) → 
  (3 < x ∧ x < 13) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_side_length_l2681_268149


namespace NUMINAMATH_CALUDE_simplify_expression_l2681_268134

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) :
  (1 - a / (a + 1)) / ((a^2 - a) / (a^2 - 1)) = 1 / a :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2681_268134


namespace NUMINAMATH_CALUDE_mass_of_copper_sulfate_pentahydrate_l2681_268148

-- Define the constants
def volume : ℝ := 0.5 -- in L
def concentration : ℝ := 1 -- in mol/L
def molar_mass : ℝ := 250 -- in g/mol

-- Theorem statement
theorem mass_of_copper_sulfate_pentahydrate (volume concentration molar_mass : ℝ) : 
  volume * concentration * molar_mass = 125 := by
  sorry

#check mass_of_copper_sulfate_pentahydrate

end NUMINAMATH_CALUDE_mass_of_copper_sulfate_pentahydrate_l2681_268148


namespace NUMINAMATH_CALUDE_circle_equation_specific_l2681_268143

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (1, -1) and radius √3 is (x-1)² + (y+1)² = 3 -/
theorem circle_equation_specific :
  let h : ℝ := 1
  let k : ℝ := -1
  let r : ℝ := Real.sqrt 3
  ∀ x y : ℝ, circle_equation h k r x y ↔ (x - 1)^2 + (y + 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_specific_l2681_268143


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2681_268199

def p (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 - 12 * x + 7
def d (x : ℝ) : ℝ := 2 * x + 3
def q (x : ℝ) : ℝ := x^2 - 4 * x + 2
def r (x : ℝ) : ℝ := -4 * x + 1

theorem polynomial_division_remainder :
  ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2681_268199


namespace NUMINAMATH_CALUDE_inequality_proof_l2681_268178

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^1999 + b^2000 ≥ a^2000 + b^2001) : 
  a^2000 + b^2000 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2681_268178


namespace NUMINAMATH_CALUDE_clock_rings_count_l2681_268175

/-- Represents the number of times a clock rings in a day -/
def clock_rings (ring_interval : ℕ) (start_hour : ℕ) (day_length : ℕ) : ℕ :=
  (day_length - start_hour) / ring_interval + 1

/-- Theorem stating that a clock ringing every 3 hours starting at 1 A.M. will ring 8 times in a day -/
theorem clock_rings_count : clock_rings 3 1 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_count_l2681_268175


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l2681_268184

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l2681_268184


namespace NUMINAMATH_CALUDE_correct_ages_l2681_268138

/-- Represents the ages of family members -/
structure FamilyAges where
  man : ℕ
  son : ℕ
  sibling : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.man = ages.son + 30) ∧
  (ages.man + 2 = 2 * (ages.son + 2)) ∧
  (ages.sibling + 2 = (ages.son + 2) / 2)

/-- Theorem stating that the ages 58, 28, and 13 satisfy the conditions -/
theorem correct_ages : 
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ ages.son = 28 ∧ ages.sibling = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_correct_ages_l2681_268138


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2681_268187

/-- The equation of a circle symmetric to another circle with respect to a line. -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = 7 ∧ x₀ + y₀ = 4) → 
  (x^2 + y^2 = 7) := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2681_268187


namespace NUMINAMATH_CALUDE_tree_planting_equation_l2681_268102

/-- Represents the relationship between the number of people planting trees and the total number of seedlings. -/
theorem tree_planting_equation (x : ℤ) (total_seedlings : ℤ) : 
  (5 * x + 3 = total_seedlings) ∧ (6 * x = total_seedlings + 4) →
  5 * x + 3 = 6 * x - 4 := by
  sorry

#check tree_planting_equation

end NUMINAMATH_CALUDE_tree_planting_equation_l2681_268102


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l2681_268104

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed in a given interval -/
def changeObservationWindow (cycle : TrafficLightCycle) (interval : ℕ) : ℕ :=
  3 * interval  -- There are 3 color changes in a cycle

/-- Calculates the probability of observing a color change during a given interval -/
def probabilityOfChange (cycle : TrafficLightCycle) (interval : ℕ) : ℚ :=
  (changeObservationWindow cycle interval : ℚ) / (cycleDuration cycle : ℚ)

theorem traffic_light_change_probability :
  ∀ (cycle : TrafficLightCycle),
    cycle.green = 45 →
    cycle.yellow = 5 →
    cycle.red = 40 →
    probabilityOfChange cycle 4 = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l2681_268104


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l2681_268119

def largest_number_divisible_by_88 : ℕ := 9944

theorem largest_number_divisible_by_88_has_4_digits :
  (largest_number_divisible_by_88 ≥ 1000) ∧ (largest_number_divisible_by_88 < 10000) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l2681_268119


namespace NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l2681_268190

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_intersection (a : ℝ) :
  (∀ x : ℝ, a ≥ 1/3 → Monotone (f a)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = a + 1 ∧ f a x = y ∧ f' a x * (-x) + y = 0) ∧
  (∃ x y : ℝ, x = -1 ∧ y = -a - 1 ∧ f a x = y ∧ f' a x * (-x) + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l2681_268190


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2681_268137

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + m*x₁ - 3 = 0) ∧ 
  (x₂^2 + m*x₂ - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2681_268137


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_cube_l2681_268130

theorem imaginary_part_of_complex_cube (i : ℂ) : i ^ 2 = -1 → Complex.im ((i⁻¹ - i) ^ 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_cube_l2681_268130


namespace NUMINAMATH_CALUDE_reciprocal_roots_iff_m_eq_p_l2681_268150

/-- A quadratic equation with coefficients p, q, and m -/
structure QuadraticEquation where
  p : ℝ
  q : ℝ
  m : ℝ

/-- The roots of a quadratic equation are reciprocals -/
def has_reciprocal_roots (eq : QuadraticEquation) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ eq.p * r^2 + eq.q * r + eq.m = 0 ∧ eq.p * (1/r)^2 + eq.q * (1/r) + eq.m = 0

/-- Theorem: The roots of px^2 + qx + m = 0 are reciprocals iff m = p -/
theorem reciprocal_roots_iff_m_eq_p (eq : QuadraticEquation) :
  has_reciprocal_roots eq ↔ eq.m = eq.p :=
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_iff_m_eq_p_l2681_268150


namespace NUMINAMATH_CALUDE_largest_value_l2681_268193

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 4 ∧ x + 3 = z + 2 ∧ x + 3 = w - 1) :
  y = max x (max y (max z w)) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l2681_268193


namespace NUMINAMATH_CALUDE_equation_3x_eq_4y_is_linear_l2681_268185

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The equation 3x = 4y is a linear equation in two variables. -/
theorem equation_3x_eq_4y_is_linear :
  IsLinearEquationInTwoVariables (fun x y => 3 * x - 4 * y) :=
sorry

end NUMINAMATH_CALUDE_equation_3x_eq_4y_is_linear_l2681_268185


namespace NUMINAMATH_CALUDE_restaurant_bill_problem_l2681_268155

theorem restaurant_bill_problem (kate_bill : ℝ) (bob_discount : ℝ) (kate_discount : ℝ) (total_after_discount : ℝ) :
  kate_bill = 25 →
  bob_discount = 0.05 →
  kate_discount = 0.02 →
  total_after_discount = 53 →
  ∃ bob_bill : ℝ, 
    bob_bill * (1 - bob_discount) + kate_bill * (1 - kate_discount) = total_after_discount ∧
    bob_bill = 30 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_problem_l2681_268155


namespace NUMINAMATH_CALUDE_gcd_of_specific_squares_l2681_268117

theorem gcd_of_specific_squares : Nat.gcd (131^2 + 243^2 + 357^2) (130^2 + 242^2 + 358^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_squares_l2681_268117


namespace NUMINAMATH_CALUDE_hawk_crow_percentage_l2681_268108

theorem hawk_crow_percentage (num_crows : ℕ) (total_birds : ℕ) (percentage : ℚ) : 
  num_crows = 30 →
  total_birds = 78 →
  total_birds = num_crows + (num_crows * (1 + percentage / 100)) →
  percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_hawk_crow_percentage_l2681_268108


namespace NUMINAMATH_CALUDE_new_year_markup_l2681_268161

theorem new_year_markup (initial_markup : ℝ) (discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.20 →
  discount = 0.07 →
  final_profit = 0.395 →
  ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - discount) = 1 + final_profit ∧
    new_year_markup = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_new_year_markup_l2681_268161


namespace NUMINAMATH_CALUDE_greatest_n_for_given_conditions_l2681_268124

theorem greatest_n_for_given_conditions (x : ℤ) (N : ℝ) : 
  (N * 10^x < 210000 ∧ x ≤ 4) → 
  ∃ (max_N : ℤ), max_N = 20 ∧ ∀ (m : ℤ), (m : ℝ) * 10^4 < 210000 → m ≤ max_N :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_given_conditions_l2681_268124


namespace NUMINAMATH_CALUDE_evaluate_expression_l2681_268196

theorem evaluate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (4 * y^2) * (1 / (2*y)^3) = 9 * y^2 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2681_268196


namespace NUMINAMATH_CALUDE_prob_draw_3_equals_expected_l2681_268198

-- Define the defect rate
def defect_rate : ℝ := 0.03

-- Define the probability of drawing exactly 3 products
def prob_draw_3 (p : ℝ) : ℝ := p^2 * (1 - p) + p^3

-- Theorem statement
theorem prob_draw_3_equals_expected : 
  prob_draw_3 defect_rate = defect_rate^2 * (1 - defect_rate) + defect_rate^3 :=
by sorry

end NUMINAMATH_CALUDE_prob_draw_3_equals_expected_l2681_268198


namespace NUMINAMATH_CALUDE_total_balloons_l2681_268165

/-- Given an initial number of balloons and an additional number of balloons,
    the total number of balloons is equal to their sum. -/
theorem total_balloons (initial additional : ℕ) :
  initial + additional = (initial + additional) := by sorry

end NUMINAMATH_CALUDE_total_balloons_l2681_268165


namespace NUMINAMATH_CALUDE_min_value_z_l2681_268114

/-- The minimum value of z = x - y given the specified constraints -/
theorem min_value_z (x y : ℝ) (h1 : x + y - 2 ≥ 0) (h2 : x ≤ 4) (h3 : y ≤ 5) :
  ∀ (x' y' : ℝ), x' + y' - 2 ≥ 0 → x' ≤ 4 → y' ≤ 5 → x - y ≤ x' - y' :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l2681_268114


namespace NUMINAMATH_CALUDE_spadesuit_calculation_l2681_268174

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spadesuit_calculation :
  (spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_calculation_l2681_268174


namespace NUMINAMATH_CALUDE_diamond_2_3_4_eq_zero_l2681_268103

/-- Definition of the diamond operation for real numbers -/
def diamond (a b c : ℝ) : ℝ := (b + 1)^2 - 4 * (a - 1) * c

/-- Theorem stating that diamond(2, 3, 4) equals 0 -/
theorem diamond_2_3_4_eq_zero : diamond 2 3 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_diamond_2_3_4_eq_zero_l2681_268103


namespace NUMINAMATH_CALUDE_unique_digit_solution_l2681_268111

theorem unique_digit_solution :
  ∃! (E U L S R T : ℕ),
    (E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (U ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (L ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (S ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (R ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (T ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    E ≠ U ∧ E ≠ L ∧ E ≠ S ∧ E ≠ R ∧ E ≠ T ∧
    U ≠ L ∧ U ≠ S ∧ U ≠ R ∧ U ≠ T ∧
    L ≠ S ∧ L ≠ R ∧ L ≠ T ∧
    S ≠ R ∧ S ≠ T ∧
    R ≠ T ∧
    E + U + L = 6 ∧
    S + R + U + T = 18 ∧
    U * T = 15 ∧
    S * L = 8 ∧
    E = 1 ∧ U = 3 ∧ L = 2 ∧ S = 4 ∧ R = 6 ∧ T = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l2681_268111


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l2681_268133

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Add any necessary properties for a convex polygon

/-- The number of diagonals in a convex polygon that skip exactly one vertex -/
def diagonals_skipping_one_vertex (n : ℕ) : ℕ := 2 * n

/-- Theorem: In a convex 25-sided polygon, there are 50 diagonals that skip exactly one vertex -/
theorem diagonals_25_sided_polygon :
  diagonals_skipping_one_vertex 25 = 50 := by
  sorry

#eval diagonals_skipping_one_vertex 25

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l2681_268133


namespace NUMINAMATH_CALUDE_partition_theorem_l2681_268159

theorem partition_theorem (a m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n < 1/m) :
  let x := a * (n - 1) / (m * n - 1)
  let first_partition := (m * x, a - m * x)
  let second_partition := (x, n * (a - m * x))
  first_partition.1 + first_partition.2 = a ∧
  second_partition.1 + second_partition.2 = a ∧
  first_partition.1 = m * second_partition.1 ∧
  second_partition.2 = n * first_partition.2 :=
by sorry

end NUMINAMATH_CALUDE_partition_theorem_l2681_268159


namespace NUMINAMATH_CALUDE_meals_given_away_l2681_268172

theorem meals_given_away (initial_meals : ℕ) (additional_meals : ℕ) (meals_left : ℕ) : 
  initial_meals = 113 → additional_meals = 50 → meals_left = 78 → 
  initial_meals + additional_meals - meals_left = 85 := by
  sorry

end NUMINAMATH_CALUDE_meals_given_away_l2681_268172


namespace NUMINAMATH_CALUDE_train_length_l2681_268171

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 179.99999999999997) (h2 : time = 3) :
  speed * time = 540 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2681_268171


namespace NUMINAMATH_CALUDE_nested_custom_op_equals_two_l2681_268170

/-- Custom operation defined as (a + b) / c -/
def customOp (a b c : ℚ) : ℚ := (a + b) / c

/-- Nested application of customOp -/
def nestedCustomOp (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℚ) : ℚ :=
  customOp (customOp a₁ b₁ c₁) (customOp a₂ b₂ c₂) (customOp a₃ b₃ c₃)

/-- Theorem stating that the nested custom operation equals 2 -/
theorem nested_custom_op_equals_two :
  nestedCustomOp 120 60 180 4 2 6 20 10 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_custom_op_equals_two_l2681_268170


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l2681_268152

/-- Given sets A and B, prove the range of a when A ∩ B = B -/
theorem intersection_equality_implies_a_range
  (A : Set ℝ)
  (B : Set ℝ)
  (a : ℝ)
  (h_A : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (h_B : B = {x : ℝ | a < x ∧ x < a + 1})
  (h_int : A ∩ B = B) :
  -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l2681_268152


namespace NUMINAMATH_CALUDE_four_of_a_kind_probability_l2681_268140

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of cards drawn
def cardsDrawn : ℕ := 6

-- Define the number of different card values (ranks)
def cardValues : ℕ := 13

-- Define the number of cards of each value
def cardsPerValue : ℕ := 4

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem four_of_a_kind_probability :
  (cardValues * binomial (standardDeck - cardsPerValue) (cardsDrawn - cardsPerValue)) /
  (binomial standardDeck cardsDrawn) = 3 / 4165 :=
sorry

end NUMINAMATH_CALUDE_four_of_a_kind_probability_l2681_268140


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_2d_l2681_268191

theorem cauchy_schwarz_inequality_2d (a₁ a₂ b₁ b₂ : ℝ) :
  a₁ * b₁ + a₂ * b₂ ≤ Real.sqrt (a₁^2 + a₂^2) * Real.sqrt (b₁^2 + b₂^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_2d_l2681_268191


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l2681_268106

theorem gcd_polynomial_and_multiple (b : ℤ) (h : ∃ k : ℤ, b = 342 * k) :
  Nat.gcd (Int.natAbs (5*b^3 + b^2 + 8*b + 38)) (Int.natAbs b) = 38 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l2681_268106


namespace NUMINAMATH_CALUDE_brown_paint_amount_l2681_268176

def total_paint : ℕ := 69
def white_paint : ℕ := 20
def green_paint : ℕ := 15

theorem brown_paint_amount :
  total_paint - (white_paint + green_paint) = 34 := by
  sorry

end NUMINAMATH_CALUDE_brown_paint_amount_l2681_268176


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_l2681_268107

/-- The length of one side of the largest equilateral triangle created from a 78 cm string -/
def triangle_side_length : ℝ := 26

/-- The total length of the string used to create the triangle -/
def string_length : ℝ := 78

/-- Theorem stating that the length of one side of the largest equilateral triangle
    created from a 78 cm string is 26 cm -/
theorem equilateral_triangle_side (s : ℝ) :
  s = triangle_side_length ↔ s * 3 = string_length :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_l2681_268107


namespace NUMINAMATH_CALUDE_factorization_y_squared_minus_one_l2681_268135

/-- A factorization is valid if the expanded form equals the factored form -/
def IsValidFactorization (expanded factored : ℝ → ℝ) : Prop :=
  ∀ x, expanded x = factored x

/-- A factorization is from left to right if it's in the form of factors multiplied together -/
def IsFactorizationLeftToRight (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), ∀ x, f x = g x * h x

theorem factorization_y_squared_minus_one :
  IsValidFactorization (fun y => y^2 - 1) (fun y => (y + 1) * (y - 1)) ∧
  IsFactorizationLeftToRight (fun y => (y + 1) * (y - 1)) ∧
  ¬IsValidFactorization (fun x => x * (a - b)) (fun x => a*x - b*x) ∧
  ¬IsValidFactorization (fun x => x^2 - 2*x) (fun x => x * (x - 2/x)) ∧
  ¬IsFactorizationLeftToRight (fun x => x * (a + b) + c) :=
by sorry

end NUMINAMATH_CALUDE_factorization_y_squared_minus_one_l2681_268135


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l2681_268147

def third_smallest_prime : ℕ := sorry

theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l2681_268147


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2681_268162

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 2) (a 4) →
  a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2681_268162


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l2681_268183

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 25 / 100 →
  germination_rate2 = 30 / 100 →
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100 = 27 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l2681_268183


namespace NUMINAMATH_CALUDE_art_marks_calculation_l2681_268151

theorem art_marks_calculation (geography : ℕ) (history_government : ℕ) (computer_science : ℕ) (modern_literature : ℕ) (average : ℚ) :
  geography = 56 →
  history_government = 60 →
  computer_science = 85 →
  modern_literature = 80 →
  average = 70.6 →
  ∃ (art : ℕ), (geography + history_government + art + computer_science + modern_literature : ℚ) / 5 = average ∧ art = 72 :=
by
  sorry

#check art_marks_calculation

end NUMINAMATH_CALUDE_art_marks_calculation_l2681_268151


namespace NUMINAMATH_CALUDE_sum_of_squares_l2681_268125

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 86) : x^2 + y^2 = 404 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2681_268125


namespace NUMINAMATH_CALUDE_function_approximation_by_additive_l2681_268120

/-- Given a function f: ℝ → ℝ satisfying |f(x+y) - f(x) - f(y)| ≤ 1 for all x, y ∈ ℝ,
    there exists a function g: ℝ → ℝ such that |f(x) - g(x)| ≤ 1 and
    g(x+y) = g(x) + g(y) for all x, y ∈ ℝ. -/
theorem function_approximation_by_additive (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ 
               (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end NUMINAMATH_CALUDE_function_approximation_by_additive_l2681_268120


namespace NUMINAMATH_CALUDE_smallest_common_multiple_proof_l2681_268154

/-- The smallest number divisible by 3, 15, and 9 -/
def smallest_common_multiple : ℕ := 45

/-- Gabe's group size -/
def gabe_group : ℕ := 3

/-- Steven's group size -/
def steven_group : ℕ := 15

/-- Maya's group size -/
def maya_group : ℕ := 9

theorem smallest_common_multiple_proof :
  (smallest_common_multiple % gabe_group = 0) ∧
  (smallest_common_multiple % steven_group = 0) ∧
  (smallest_common_multiple % maya_group = 0) ∧
  (∀ n : ℕ, n < smallest_common_multiple →
    ¬((n % gabe_group = 0) ∧ (n % steven_group = 0) ∧ (n % maya_group = 0))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_proof_l2681_268154


namespace NUMINAMATH_CALUDE_equation_solution_l2681_268179

theorem equation_solution : ∃ x : ℝ, (x / 2 - 1 = 3) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2681_268179


namespace NUMINAMATH_CALUDE_function_equality_l2681_268192

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom cond1 : ∀ x : ℝ, f x ≤ x
axiom cond2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y

-- State the theorem
theorem function_equality : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2681_268192


namespace NUMINAMATH_CALUDE_sequence_matches_first_five_terms_general_term_formula_l2681_268101

/-- The sequence a_n defined by the given first five terms and the general formula -/
def a : ℕ → ℕ := λ n => n^2 + 5

/-- The theorem stating that the sequence matches the given first five terms -/
theorem sequence_matches_first_five_terms :
  a 1 = 6 ∧ a 2 = 9 ∧ a 3 = 14 ∧ a 4 = 21 ∧ a 5 = 30 := by sorry

/-- The main theorem proving that a_n is the general term formula for the sequence -/
theorem general_term_formula (n : ℕ) (h : n > 0) : a n = n^2 + 5 := by sorry

end NUMINAMATH_CALUDE_sequence_matches_first_five_terms_general_term_formula_l2681_268101


namespace NUMINAMATH_CALUDE_f_difference_at_five_l2681_268141

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

-- Theorem statement
theorem f_difference_at_five : f 5 - f (-5) = 800 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l2681_268141


namespace NUMINAMATH_CALUDE_least_side_of_right_triangle_l2681_268109

theorem least_side_of_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a = 8 → b = 15 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_least_side_of_right_triangle_l2681_268109


namespace NUMINAMATH_CALUDE_children_off_bus_l2681_268123

theorem children_off_bus (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 22 → got_on = 40 → final = 2 → initial + got_on - final = 60 := by
  sorry

end NUMINAMATH_CALUDE_children_off_bus_l2681_268123


namespace NUMINAMATH_CALUDE_hex_palindrome_probability_l2681_268164

/-- Represents a hexadecimal digit (0-15) -/
def HexDigit := Fin 16

/-- Represents a 6-digit hexadecimal palindrome -/
structure HexPalindrome where
  a : HexDigit
  b : HexDigit
  c : HexDigit
  value : ℕ := 1048592 * a.val + 65792 * b.val + 4096 * c.val

/-- Predicate to check if a natural number is a hexadecimal palindrome -/
def isHexPalindrome (n : ℕ) : Prop := sorry

/-- The total number of 6-digit hexadecimal palindromes -/
def totalPalindromes : ℕ := 3840

/-- The number of 6-digit hexadecimal palindromes that, when divided by 17, 
    result in another hexadecimal palindrome -/
def validPalindromes : ℕ := sorry

theorem hex_palindrome_probability : 
  (validPalindromes : ℚ) / totalPalindromes = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_hex_palindrome_probability_l2681_268164


namespace NUMINAMATH_CALUDE_simplify_expression_l2681_268177

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2681_268177


namespace NUMINAMATH_CALUDE_function_inequality_and_sum_inequality_l2681_268169

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

-- Define the theorem
theorem function_inequality_and_sum_inequality :
  (∀ x m : ℝ, f x ≥ |m + 1|) →
  (∃ M : ℝ, M = 4 ∧
    (∀ m : ℝ, (∀ x : ℝ, f x ≥ |m + 1|) → m ≤ M) ∧
    (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + c = M →
      1 / (a + b) + 1 / (b + c) ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_sum_inequality_l2681_268169


namespace NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_9_and_9_1_l2681_268105

theorem unique_number_divisible_by_24_with_cube_root_between_9_and_9_1 :
  ∃! (n : ℕ), n > 0 ∧ 24 ∣ n ∧ 9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_9_and_9_1_l2681_268105


namespace NUMINAMATH_CALUDE_friday_temp_is_35_l2681_268128

/-- Represents the temperature for a given day -/
structure DayTemp where
  temp : ℕ
  isOdd : Odd temp

/-- Represents the temperatures for Monday to Friday -/
structure WeekTemp where
  monday : DayTemp
  tuesday : DayTemp
  wednesday : DayTemp
  thursday : DayTemp
  friday : DayTemp

def WeekTemp.avgMondayToThursday (w : WeekTemp) : ℚ :=
  (w.monday.temp + w.tuesday.temp + w.wednesday.temp + w.thursday.temp) / 4

def WeekTemp.avgTuesdayToFriday (w : WeekTemp) : ℚ :=
  (w.tuesday.temp + w.wednesday.temp + w.thursday.temp + w.friday.temp) / 4

theorem friday_temp_is_35 (w : WeekTemp) :
  w.avgMondayToThursday = 48 →
  w.avgTuesdayToFriday = 46 →
  w.monday.temp = 43 →
  w.friday.temp = 35 := by sorry

end NUMINAMATH_CALUDE_friday_temp_is_35_l2681_268128


namespace NUMINAMATH_CALUDE_daisy_sales_proof_l2681_268116

/-- The number of daisies sold on the first day -/
def first_day_sales : ℕ := 45

/-- The number of daisies sold on the second day -/
def second_day_sales : ℕ := first_day_sales + 20

/-- The number of daisies sold on the third day -/
def third_day_sales : ℕ := 2 * second_day_sales - 10

/-- The number of daisies sold on the fourth day -/
def fourth_day_sales : ℕ := 120

/-- The total number of daisies sold over 4 days -/
def total_sales : ℕ := 350

theorem daisy_sales_proof :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = total_sales :=
by sorry

end NUMINAMATH_CALUDE_daisy_sales_proof_l2681_268116


namespace NUMINAMATH_CALUDE_product_of_roots_eq_one_l2681_268158

theorem product_of_roots_eq_one :
  let f : ℝ → ℝ := λ x => x^(Real.log x / Real.log 5) - 25
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ * r₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_eq_one_l2681_268158


namespace NUMINAMATH_CALUDE_rectangle_area_l2681_268194

/-- Proves that a rectangle with perimeter 126 and difference between sides 37 has an area of 650 -/
theorem rectangle_area (l w : ℝ) : 
  (2 * (l + w) = 126) → 
  (l - w = 37) → 
  (l * w = 650) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2681_268194


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l2681_268156

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ
  is_isosceles : True
  base1_longer : base1 > base2

-- Define the rotation of the trapezoid
def rotate_trapezoid (t : IsoscelesTrapezoid) : Solid :=
  sorry

-- Define the components of a solid
inductive SolidComponent
  | Cylinder
  | Cone
  | Frustum

-- Define a solid as a collection of components
def Solid := List SolidComponent

-- Theorem statement
theorem isosceles_trapezoid_rotation 
  (t : IsoscelesTrapezoid) : 
  rotate_trapezoid t = [SolidComponent.Cylinder, SolidComponent.Cone, SolidComponent.Cone] :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l2681_268156


namespace NUMINAMATH_CALUDE_nancy_shoe_count_l2681_268163

def shoe_count (boots slippers heels : ℕ) : ℕ :=
  2 * (boots + slippers + heels)

theorem nancy_shoe_count :
  ∀ (boots slippers heels : ℕ),
    boots = 6 →
    slippers = boots + 9 →
    heels = 3 * (boots + slippers) →
    shoe_count boots slippers heels = 168 := by
  sorry

end NUMINAMATH_CALUDE_nancy_shoe_count_l2681_268163


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l2681_268136

def A (n : ℕ) : ℕ := n * (n - 1)

theorem permutation_equation_solution :
  ∃ (x : ℕ), 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l2681_268136


namespace NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l2681_268186

theorem factorization_of_2a2_minus_8b2 (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l2681_268186


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l2681_268115

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l2681_268115


namespace NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l2681_268146

theorem least_x_squared_divisible_by_240 :
  ∀ x : ℕ, x > 0 → x^2 % 240 = 0 → x ≥ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l2681_268146


namespace NUMINAMATH_CALUDE_subset_M_l2681_268126

def M : Set ℝ := {x | x + 1 > 0}

theorem subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_M_l2681_268126


namespace NUMINAMATH_CALUDE_farmer_apples_final_apple_count_l2681_268100

theorem farmer_apples (initial : ℝ) (given_away : ℝ) (harvested : ℝ) :
  initial - given_away + harvested = initial + harvested - given_away :=
by sorry

theorem final_apple_count (initial : ℝ) (given_away : ℝ) (harvested : ℝ) :
  initial = 5708 → given_away = 2347.5 → harvested = 1526.75 →
  initial - given_away + harvested = 4887.25 :=
by sorry

end NUMINAMATH_CALUDE_farmer_apples_final_apple_count_l2681_268100


namespace NUMINAMATH_CALUDE_f_properties_l2681_268180

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2)*x

theorem f_properties :
  let f := f
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
    (∃ min_val : ℝ, a = 1 → (∀ y > 0, f 1 y ≥ f 1 2) ∧ f 1 2 = -2 * Real.log 2) ∧
    (
      (a ≥ 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ > f a x₂) ∧
                (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (-2 < a ∧ a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂) ∧
                        (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ > f a x₂) ∧
                        (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (a = -2 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (a < -2 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂) ∧
                (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ > f a x₂) ∧
                (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))
    ) := by sorry

end NUMINAMATH_CALUDE_f_properties_l2681_268180


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l2681_268166

theorem fraction_sum_difference : (7 : ℚ) / 12 + 8 / 15 - 2 / 5 = 43 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l2681_268166


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2681_268153

theorem trigonometric_identities :
  (∀ n : ℤ,
    (Real.sin (4 * Real.pi / 3) * Real.cos (25 * Real.pi / 6) * Real.tan (5 * Real.pi / 4) = -3/4) ∧
    (Real.sin ((2 * n + 1) * Real.pi - 2 * Real.pi / 3) = Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2681_268153


namespace NUMINAMATH_CALUDE_problem_solution_l2681_268121

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = -6) : 
  y = 33 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2681_268121


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l2681_268112

/-- Represents the number of students in grade 10 -/
def grade_10_students : ℕ := sorry

/-- Represents the number of students in grade 11 -/
def grade_11_students : ℕ := grade_10_students + 300

/-- Represents the number of students in grade 12 -/
def grade_12_students : ℕ := 2 * grade_10_students

/-- The total number of students in all three grades -/
def total_students : ℕ := 3500

/-- The sampling ratio -/
def sampling_ratio : ℚ := 1 / 100

/-- Theorem stating the number of grade 10 students to be sampled -/
theorem grade_10_sample_size : 
  grade_10_students + grade_11_students + grade_12_students = total_students →
  (↑grade_10_students * sampling_ratio).floor = 8 := by
  sorry

end NUMINAMATH_CALUDE_grade_10_sample_size_l2681_268112


namespace NUMINAMATH_CALUDE_calculate_savings_l2681_268188

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
theorem calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income_ratio > 0 ∧ expenditure_ratio > 0 ∧ income = 21000 ∧ income_ratio = 3 ∧ expenditure_ratio = 2 →
  income - (income * expenditure_ratio / income_ratio) = 7000 := by
sorry

end NUMINAMATH_CALUDE_calculate_savings_l2681_268188


namespace NUMINAMATH_CALUDE_triangle_area_l2681_268113

theorem triangle_area (a b c : ℝ) (A : ℝ) : 
  b = 3 → 
  a - c = 2 → 
  A = 2 * Real.pi / 3 → 
  (1 / 2) * b * c * Real.sin A = 15 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2681_268113


namespace NUMINAMATH_CALUDE_function_identity_l2681_268142

theorem function_identity (f : ℕ → ℕ) (h : ∀ n, f (n + 1) > f (f n)) : ∀ n, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2681_268142
