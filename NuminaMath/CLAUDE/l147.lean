import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l147_14788

theorem quadratic_equation_solution (b : ℝ) : 
  (2 * (-5)^2 + b * (-5) - 20 = 0) → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l147_14788


namespace NUMINAMATH_CALUDE_quadratic_minimum_l147_14741

/-- 
Given a quadratic function y = 3x^2 + px + q,
if the minimum value of y is 4,
then q = p^2/12 + 4
-/
theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, 3 * x^2 + p * x + q ≥ 4) ∧ 
  (∃ x, 3 * x^2 + p * x + q = 4) →
  q = p^2 / 12 + 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l147_14741


namespace NUMINAMATH_CALUDE_equation_satisfied_l147_14797

theorem equation_satisfied (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l147_14797


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l147_14796

/-- Given a line y = -3x + 5 and two points (-6, y₁) and (3, y₂) on this line,
    prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -3 * (-6) + 5) →  -- Point (-6, y₁) lies on the line
  (y₂ = -3 * 3 + 5) →     -- Point (3, y₂) lies on the line
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l147_14796


namespace NUMINAMATH_CALUDE_kim_integer_problem_l147_14766

theorem kim_integer_problem (x y : ℤ) : 
  3 * x + 2 * y = 145 → (x = 35 ∨ y = 35) → (x = 20 ∨ y = 20) :=
by sorry

end NUMINAMATH_CALUDE_kim_integer_problem_l147_14766


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l147_14705

/-- Given a parabola y^2 = 2px (p > 0) with directrix x = -p/2, 
    if the directrix is tangent to the circle (x - 3)^2 + y^2 = 16, then p = 2 -/
theorem parabola_directrix_tangent_circle (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 → (x - 3)^2 + y^2 = 16) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l147_14705


namespace NUMINAMATH_CALUDE_empty_container_mass_l147_14765

/-- The mass of an empty container, given its mass when filled with kerosene and water, and the densities of kerosene and water. -/
theorem empty_container_mass
  (mass_with_kerosene : ℝ)
  (mass_with_water : ℝ)
  (density_water : ℝ)
  (density_kerosene : ℝ)
  (h1 : mass_with_kerosene = 20)
  (h2 : mass_with_water = 24)
  (h3 : density_water = 1000)
  (h4 : density_kerosene = 800) :
  ∃ (empty_mass : ℝ), empty_mass = 4 ∧
  mass_with_kerosene = empty_mass + density_kerosene * ((mass_with_water - mass_with_kerosene) / (density_water - density_kerosene)) ∧
  mass_with_water = empty_mass + density_water * ((mass_with_water - mass_with_kerosene) / (density_water - density_kerosene)) :=
by
  sorry


end NUMINAMATH_CALUDE_empty_container_mass_l147_14765


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l147_14712

theorem consecutive_integer_product_divisibility (j : ℤ) : 
  let m := j * (j + 1) * (j + 2) * (j + 3)
  (∃ k : ℤ, m = 11 * k) →
  (∃ k : ℤ, m = 12 * k) ∧
  (∃ k : ℤ, m = 33 * k) ∧
  (∃ k : ℤ, m = 44 * k) ∧
  (∃ k : ℤ, m = 66 * k) ∧
  ¬(∀ j : ℤ, ∃ k : ℤ, m = 24 * k) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l147_14712


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l147_14729

theorem field_length_width_ratio :
  ∀ (w : ℝ),
    w > 0 →
    24 > 0 →
    ∃ (k : ℕ), 24 = k * w →
    36 = (1/8) * (24 * w) →
    24 / w = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l147_14729


namespace NUMINAMATH_CALUDE_test_maximum_marks_l147_14757

theorem test_maximum_marks (passing_threshold : Real) (student_score : Nat) (failing_margin : Nat) 
  (h1 : passing_threshold = 0.60)
  (h2 : student_score = 80)
  (h3 : failing_margin = 40) :
  (student_score + failing_margin) / passing_threshold = 200 := by
sorry

end NUMINAMATH_CALUDE_test_maximum_marks_l147_14757


namespace NUMINAMATH_CALUDE_rectangle_area_l147_14703

/-- The area of a rectangle with perimeter equal to a triangle with sides 7.3, 9.4, and 11.3,
    and length twice its width, is 392/9 square centimeters. -/
theorem rectangle_area (triangle_side1 triangle_side2 triangle_side3 : ℝ)
  (rectangle_width rectangle_length : ℝ) :
  triangle_side1 = 7.3 →
  triangle_side2 = 9.4 →
  triangle_side3 = 11.3 →
  2 * (rectangle_length + rectangle_width) = triangle_side1 + triangle_side2 + triangle_side3 →
  rectangle_length = 2 * rectangle_width →
  rectangle_length * rectangle_width = 392 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l147_14703


namespace NUMINAMATH_CALUDE_profit_per_meter_l147_14759

/-- Calculate the profit per meter of cloth -/
theorem profit_per_meter (meters_sold : ℕ) (selling_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_sold = 45 →
  selling_price = 4500 →
  cost_price_per_meter = 88 →
  (selling_price - meters_sold * cost_price_per_meter) / meters_sold = 12 :=
by sorry

end NUMINAMATH_CALUDE_profit_per_meter_l147_14759


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l147_14722

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 38 →
  a 4 + a 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l147_14722


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l147_14734

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 64) (h2 : sum = 2^12) :
  (sum / n : ℚ) = 64 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l147_14734


namespace NUMINAMATH_CALUDE_particular_propositions_count_l147_14782

/-- A proposition is particular if it contains quantifiers like "some", "exists", or "some of". -/
def is_particular_proposition (p : Prop) : Prop := sorry

/-- The first proposition: Some triangles are isosceles triangles. -/
def prop1 : Prop := sorry

/-- The second proposition: There exists an integer x such that x^2 - 2x - 3 = 0. -/
def prop2 : Prop := sorry

/-- The third proposition: There exists a triangle whose sum of interior angles is 170°. -/
def prop3 : Prop := sorry

/-- The fourth proposition: Rectangles are parallelograms. -/
def prop4 : Prop := sorry

/-- The list of all given propositions. -/
def propositions : List Prop := [prop1, prop2, prop3, prop4]

/-- Count the number of particular propositions in a list. -/
def count_particular_propositions (props : List Prop) : Nat := sorry

theorem particular_propositions_count :
  count_particular_propositions propositions = 3 := by sorry

end NUMINAMATH_CALUDE_particular_propositions_count_l147_14782


namespace NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_translation_then_symmetry_is_symmetry_symmetry_then_translation_is_symmetry_l147_14744

-- Define central symmetry
def central_symmetry (O : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Define parallel translation
def parallel_translation (a : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + a.1, P.2 + a.2)

-- Theorem 1: Composition of two central symmetries is a parallel translation
theorem composition_central_symmetries_is_translation 
  (O₁ O₂ : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ a : ℝ × ℝ, central_symmetry O₂ (central_symmetry O₁ P) = parallel_translation a P :=
sorry

-- Theorem 2a: Composition of translation and central symmetry is a central symmetry
theorem translation_then_symmetry_is_symmetry 
  (a : ℝ × ℝ) (O : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ O' : ℝ × ℝ, central_symmetry O (parallel_translation a P) = central_symmetry O' P :=
sorry

-- Theorem 2b: Composition of central symmetry and translation is a central symmetry
theorem symmetry_then_translation_is_symmetry 
  (O : ℝ × ℝ) (a : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ O' : ℝ × ℝ, parallel_translation a (central_symmetry O P) = central_symmetry O' P :=
sorry

end NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_translation_then_symmetry_is_symmetry_symmetry_then_translation_is_symmetry_l147_14744


namespace NUMINAMATH_CALUDE_megan_initial_bottles_l147_14724

/-- The number of water bottles Megan had initially -/
def initial_bottles : ℕ := sorry

/-- The number of water bottles Megan drank -/
def bottles_drank : ℕ := 3

/-- The number of water bottles Megan had left -/
def bottles_left : ℕ := 14

theorem megan_initial_bottles : 
  initial_bottles = bottles_left + bottles_drank :=
by sorry

end NUMINAMATH_CALUDE_megan_initial_bottles_l147_14724


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l147_14723

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l147_14723


namespace NUMINAMATH_CALUDE_cubic_polynomial_determinant_l147_14795

/-- Given a cubic polynomial x^3 + sx^2 + px + q with roots a, b, and c,
    the determinant of the matrix [[s + a, 1, 1], [1, s + b, 1], [1, 1, s + c]]
    is equal to s^3 + sp - q - 2s - 2(p - s) -/
theorem cubic_polynomial_determinant (s p q a b c : ℝ) : 
  a^3 + s*a^2 + p*a + q = 0 →
  b^3 + s*b^2 + p*b + q = 0 →
  c^3 + s*c^2 + p*c + q = 0 →
  Matrix.det ![![s + a, 1, 1], ![1, s + b, 1], ![1, 1, s + c]] = s^3 + s*p - q - 2*s - 2*(p - s) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_determinant_l147_14795


namespace NUMINAMATH_CALUDE_max_min_y_over_x_l147_14764

theorem max_min_y_over_x :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 3 →
  (∀ (z w : ℝ), (z - 2)^2 + w^2 = 3 → w / z ≤ Real.sqrt 3) ∧
  (∀ (z w : ℝ), (z - 2)^2 + w^2 = 3 → w / z ≥ -Real.sqrt 3) ∧
  (∃ (x₁ y₁ : ℝ), (x₁ - 2)^2 + y₁^2 = 3 ∧ y₁ / x₁ = Real.sqrt 3) ∧
  (∃ (x₂ y₂ : ℝ), (x₂ - 2)^2 + y₂^2 = 3 ∧ y₂ / x₂ = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_y_over_x_l147_14764


namespace NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l147_14768

/-- A tic-tac-toe board is a 3x3 grid. -/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- A winning position is a line (row, column, or diagonal) on the board. -/
def WinningPosition : Type := List (Fin 3 × Fin 3)

/-- The set of all winning positions on a tic-tac-toe board. -/
def allWinningPositions : List WinningPosition :=
  -- 3 horizontal lines
  [[(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)]] ++
  -- 3 vertical lines
  [[(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)]] ++
  -- 2 diagonal lines
  [[(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]]

/-- The number of ways to arrange 3 noughts on a 3x3 board. -/
def totalArrangements : ℕ := 84

/-- The probability of three noughts being in a winning position. -/
def winningProbability : ℚ := 2 / 21

theorem tic_tac_toe_winning_probability :
  (List.length allWinningPositions : ℚ) / totalArrangements = winningProbability := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l147_14768


namespace NUMINAMATH_CALUDE_wipes_per_pack_l147_14733

theorem wipes_per_pack (wipes_per_day : ℕ) (days : ℕ) (num_packs : ℕ) : 
  wipes_per_day = 2 → days = 360 → num_packs = 6 → 
  (wipes_per_day * days) / num_packs = 120 := by
  sorry

end NUMINAMATH_CALUDE_wipes_per_pack_l147_14733


namespace NUMINAMATH_CALUDE_inequality_condition_l147_14776

theorem inequality_condition (m : ℝ) : m ≠ 0 →
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) ↔ m < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l147_14776


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l147_14772

theorem quadratic_equation_roots (a : ℝ) : 
  (3 : ℝ)^2 + a * 3 - 2 * a = 0 → 
  ∃ b : ℝ, b^2 + a * b - 2 * a = 0 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l147_14772


namespace NUMINAMATH_CALUDE_family_weight_theorem_l147_14727

/-- Represents the weights of a family with three generations -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- The total weight of the family -/
def FamilyWeights.total (w : FamilyWeights) : ℝ :=
  w.mother + w.daughter + w.grandchild

/-- The conditions given in the problem -/
def WeightConditions (w : FamilyWeights) : Prop :=
  w.daughter + w.grandchild = 60 ∧
  w.grandchild = (1/5) * w.mother ∧
  w.daughter = 50

/-- Theorem stating that given the conditions, the total weight is 110 kg -/
theorem family_weight_theorem (w : FamilyWeights) (h : WeightConditions w) :
  w.total = 110 := by
  sorry


end NUMINAMATH_CALUDE_family_weight_theorem_l147_14727


namespace NUMINAMATH_CALUDE_max_area_rectangle_l147_14787

/-- The maximum area of a rectangle with perimeter P is P²/16 -/
theorem max_area_rectangle (P : ℝ) (h : P > 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + 2*y = P ∧ 
  x*y = P^2/16 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a + 2*b = P → a*b ≤ P^2/16 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l147_14787


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l147_14725

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := y - x = 0

/-- The original line -/
def original_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line_of_symmetry -/
theorem symmetry_of_lines :
  ∀ (x y x' y' : ℝ),
    original_line x y →
    line_of_symmetry ((x + x') / 2) ((y + y') / 2) →
    symmetric_line x' y' :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l147_14725


namespace NUMINAMATH_CALUDE_quarterback_passes_l147_14738

theorem quarterback_passes (total : ℕ) (left : ℕ) (right : ℕ) (center : ℕ) 
  (h1 : total = 50)
  (h2 : right = 2 * left)
  (h3 : center = left + 2)
  (h4 : total = left + right + center) :
  left = 12 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l147_14738


namespace NUMINAMATH_CALUDE_A_intersect_B_l147_14781

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x < 3}

theorem A_intersect_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l147_14781


namespace NUMINAMATH_CALUDE_no_real_solutions_to_equation_l147_14779

theorem no_real_solutions_to_equation :
  ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 4 ∧ (3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_to_equation_l147_14779


namespace NUMINAMATH_CALUDE_largest_number_l147_14704

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 995/1000) 
  (hb : b = 9995/10000) 
  (hc : c = 99/100) 
  (hd : d = 999/1000) 
  (he : e = 9959/10000) : 
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l147_14704


namespace NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l147_14708

theorem cube_volume_equals_surface_area (s : ℝ) (h : s > 0) :
  s^3 = 6 * s^2 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l147_14708


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l147_14784

/-- The complex number z = (2-i)/(1-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l147_14784


namespace NUMINAMATH_CALUDE_right_triangle_medians_semiperimeter_l147_14770

theorem right_triangle_medians_semiperimeter (a b : ℝ) (h1 : a = 6) (h2 : b = 4) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let m1 := c / 2
  let m2 := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  m1 + m2 = s := by sorry

end NUMINAMATH_CALUDE_right_triangle_medians_semiperimeter_l147_14770


namespace NUMINAMATH_CALUDE_circle_area_increase_l147_14700

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := r * 2.5
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l147_14700


namespace NUMINAMATH_CALUDE_sector_arc_length_l147_14740

theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = 120) (h2 : r = 2) :
  let arc_length := θ / 360 * (2 * Real.pi * r)
  arc_length = 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l147_14740


namespace NUMINAMATH_CALUDE_thirteenth_fib_is_610_l147_14767

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The 13th Fibonacci number is 610 -/
theorem thirteenth_fib_is_610 : fib 13 = 610 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_fib_is_610_l147_14767


namespace NUMINAMATH_CALUDE_correct_new_upstream_time_l147_14773

/-- Represents the boat's journey on a river with varying current conditions -/
structure RiverJourney where
  downstream_time : ℝ  -- Time from A to C downstream
  upstream_time : ℝ    -- Time from C to A upstream
  new_downstream_time : ℝ  -- Time from A to C with uniform current
  boat_speed : ℝ        -- Boat's own speed (constant)

/-- Calculates the upstream time under new conditions -/
def new_upstream_time (journey : RiverJourney) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct upstream time under new conditions -/
theorem correct_new_upstream_time (journey : RiverJourney) 
  (h1 : journey.downstream_time = 6)
  (h2 : journey.upstream_time = 7)
  (h3 : journey.new_downstream_time = 5.5)
  (h4 : journey.boat_speed > 0) :
  new_upstream_time journey = 7.7 := by
  sorry

end NUMINAMATH_CALUDE_correct_new_upstream_time_l147_14773


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l147_14747

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 24 = (x + n)^2 + 16) → 
  b = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l147_14747


namespace NUMINAMATH_CALUDE_q_div_p_eq_90_l147_14745

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers * Nat.choose cards_per_number cards_drawn) / Nat.choose total_cards cards_drawn

/-- The probability of drawing three cards with one number and one card with a different number -/
def q : ℚ := (distinct_numbers * Nat.choose cards_per_number 3 * (distinct_numbers - 1) * Nat.choose cards_per_number 1) / Nat.choose total_cards cards_drawn

/-- The main theorem stating that q/p = 90 -/
theorem q_div_p_eq_90 : q / p = 90 := by
  sorry

end NUMINAMATH_CALUDE_q_div_p_eq_90_l147_14745


namespace NUMINAMATH_CALUDE_middle_integer_of_three_consecutive_l147_14751

/-- Given three consecutive integers whose sum is 360, the middle integer is 120. -/
theorem middle_integer_of_three_consecutive (n : ℤ) : 
  (n - 1) + n + (n + 1) = 360 → n = 120 := by
  sorry

end NUMINAMATH_CALUDE_middle_integer_of_three_consecutive_l147_14751


namespace NUMINAMATH_CALUDE_intersection_of_lines_l147_14706

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (1/5, 2/5)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 1 = 7 * x

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l147_14706


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l147_14780

/-- The polynomial p(x) that satisfies the given conditions -/
def p (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that p(x) satisfies the required conditions -/
theorem p_satisfies_conditions :
  (p 3 = 10) ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_conditions_l147_14780


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l147_14771

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -4 ∧ c = -12 → abs (r₁ - r₂) = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l147_14771


namespace NUMINAMATH_CALUDE_missing_number_is_eight_l147_14791

-- Define the structure of the pyramid
def Pyramid (a b c d e : ℕ) : Prop :=
  b * c = d ∧ c * a = e ∧ d * e = 3360

-- Theorem statement
theorem missing_number_is_eight :
  ∃ (x : ℕ), Pyramid 8 6 7 42 x ∧ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_missing_number_is_eight_l147_14791


namespace NUMINAMATH_CALUDE_kelly_carrots_l147_14730

/-- The number of carrots Kelly pulled out from the first bed -/
def carrots_first_bed (total_carrots second_bed third_bed : ℕ) : ℕ :=
  total_carrots - second_bed - third_bed

/-- Theorem stating the number of carrots Kelly pulled out from the first bed -/
theorem kelly_carrots :
  carrots_first_bed (39 * 6) 101 78 = 55 := by
  sorry

end NUMINAMATH_CALUDE_kelly_carrots_l147_14730


namespace NUMINAMATH_CALUDE_inverse_matrices_solution_l147_14720

def matrix1 (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![5, -9; a, 12]
def matrix2 (b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![12, b; 3, 5]

theorem inverse_matrices_solution (a b : ℝ) :
  (matrix1 a) * (matrix2 b) = 1 → a = -3 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_solution_l147_14720


namespace NUMINAMATH_CALUDE_divisor_proof_l147_14718

theorem divisor_proof : ∃ x : ℝ, (26.3 * 12 * 20) / x + 125 = 2229 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisor_proof_l147_14718


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l147_14753

theorem trigonometric_inequality (x : ℝ) : 
  (9.276 * Real.sin (2 * x) * Real.sin (3 * x) - Real.cos (2 * x) * Real.cos (3 * x) > Real.sin (10 * x)) ↔ 
  (∃ n : ℤ, ((-Real.pi / 10 + 2 * Real.pi * n / 5 < x ∧ x < -Real.pi / 30 + 2 * Real.pi * n) ∨ 
             (Real.pi / 10 + 2 * Real.pi * n / 5 < x ∧ x < 7 * Real.pi / 30 + 2 * Real.pi * n))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l147_14753


namespace NUMINAMATH_CALUDE_root_in_interval_l147_14737

theorem root_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) (-3 : ℝ) ∧ x^3 + 3*x^2 - x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l147_14737


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l147_14769

theorem line_intersects_parabola_vertex (b : ℝ) : 
  (∃! (x y : ℝ), y = x + b ∧ y = x^2 + 2*b^2 ∧ x = 0) ↔ (b = 0 ∨ b = 1/2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l147_14769


namespace NUMINAMATH_CALUDE_farm_has_eleven_goats_l147_14716

/-- Represents the number of animals on a farm -/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Defines the properties of the farm in the problem -/
def ProblemFarm (f : Farm) : Prop :=
  (f.pigs = 2 * f.cows) ∧ 
  (f.cows = f.goats + 4) ∧ 
  (f.goats + f.cows + f.pigs = 56)

/-- Theorem stating that a farm satisfying the problem conditions has 11 goats -/
theorem farm_has_eleven_goats (f : Farm) (h : ProblemFarm f) : f.goats = 11 := by
  sorry


end NUMINAMATH_CALUDE_farm_has_eleven_goats_l147_14716


namespace NUMINAMATH_CALUDE_total_distance_walked_l147_14719

-- Define constants for conversion
def feet_per_mile : ℕ := 5280
def feet_per_yard : ℕ := 3

-- Define the distances walked by each person
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

-- Theorem statement
theorem total_distance_walked :
  lionel_miles * feet_per_mile + esther_yards * feet_per_yard + niklaus_feet = 24332 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_walked_l147_14719


namespace NUMINAMATH_CALUDE_curve_point_when_a_is_one_curve_passes_through_fixed_point_l147_14793

-- Define the curve equation
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0

-- Theorem for case a = 1
theorem curve_point_when_a_is_one :
  ∀ x y : ℝ, curve x y 1 ↔ x = 1 ∧ y = 1 :=
sorry

-- Theorem for case a ≠ 1
theorem curve_passes_through_fixed_point :
  ∀ a : ℝ, a ≠ 1 → curve 1 1 a :=
sorry

end NUMINAMATH_CALUDE_curve_point_when_a_is_one_curve_passes_through_fixed_point_l147_14793


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l147_14785

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + (1/4 : ℝ) ≤ 0) ↔ (0 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l147_14785


namespace NUMINAMATH_CALUDE_one_number_is_zero_l147_14739

/-- Represents a card with a number -/
structure Card where
  value : ℤ

/-- Represents the deck of 30 cards -/
def Deck : Type := Fin 30 → Card

/-- The property that for any 5 cards, there exist another 5 cards such that their sum is zero -/
def has_zero_sum_property (deck : Deck) : Prop :=
  ∀ (s : Finset (Fin 30)) (hs : s.card = 5),
    ∃ (t : Finset (Fin 30)) (ht : t.card = 5) (hd : Disjoint s t),
      (s.sum (λ i => (deck i).value) + t.sum (λ i => (deck i).value) = 0)

/-- The theorem to be proved -/
theorem one_number_is_zero
  (deck : Deck)
  (ha : ∃ a : ℤ, (Finset.filter (λ i => (deck i).value = a) (Finset.univ : Finset (Fin 30))).card = 10)
  (hb : ∃ b : ℤ, (Finset.filter (λ i => (deck i).value = b) (Finset.univ : Finset (Fin 30))).card = 10)
  (hc : ∃ c : ℤ, (Finset.filter (λ i => (deck i).value = c) (Finset.univ : Finset (Fin 30))).card = 10)
  (hdiff : ∀ x y, x ≠ y → (Finset.filter (λ i => (deck i).value = x) (Finset.univ : Finset (Fin 30))).card = 10 →
                         (Finset.filter (λ i => (deck i).value = y) (Finset.univ : Finset (Fin 30))).card = 10 → x ≠ y)
  (hzero_sum : has_zero_sum_property deck) :
  ∃ x : ℤ, x = 0 ∧ (Finset.filter (λ i => (deck i).value = x) (Finset.univ : Finset (Fin 30))).card = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_one_number_is_zero_l147_14739


namespace NUMINAMATH_CALUDE_max_value_problem_l147_14789

theorem max_value_problem (x y : ℝ) (h : y^2 + x - 2 = 0) :
  ∃ (M : ℝ), M = 7 ∧ ∀ (x' y' : ℝ), y'^2 + x' - 2 = 0 → y'^2 - x'^2 + x' + 5 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l147_14789


namespace NUMINAMATH_CALUDE_product_of_numbers_l147_14758

theorem product_of_numbers (x y : ℝ) : x + y = 50 ∧ x - y = 6 → x * y = 616 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l147_14758


namespace NUMINAMATH_CALUDE_impossible_division_l147_14715

/-- Represents a chess-like board with alternating colors -/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents an L-shaped piece on the board -/
structure LPiece :=
  (x : Fin 8) (y : Fin 8)

/-- Checks if an L-piece is valid (within bounds and not in the cut-out corner) -/
def isValidPiece (b : Board) (p : LPiece) : Prop :=
  p.x < 6 ∧ p.y < 6 ∧ ¬(p.x = 0 ∧ p.y = 0)

/-- Counts the number of squares of each color covered by an L-piece -/
def colorCount (b : Board) (p : LPiece) : Nat × Nat :=
  let trueCount := (b p.x p.y).toNat + (b p.x (p.y + 1)).toNat + (b (p.x + 1) p.y).toNat + (b (p.x + 1) (p.y + 1)).toNat
  (trueCount, 4 - trueCount)

/-- The main theorem stating that it's impossible to divide the board as required -/
theorem impossible_division (b : Board) : ¬ ∃ (pieces : List LPiece),
  pieces.length = 15 ∧ 
  (∀ p ∈ pieces, isValidPiece b p) ∧
  (∀ p ∈ pieces, (colorCount b p).1 = 3 ∨ (colorCount b p).2 = 3) ∧
  (pieces.map (λ p => (colorCount b p).1)).sum = 30 :=
sorry

end NUMINAMATH_CALUDE_impossible_division_l147_14715


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l147_14752

theorem simplified_fraction_sum (c d : ℕ+) : 
  (c : ℚ) / d = 0.375 ∧ 
  ∀ (a b : ℕ+), (a : ℚ) / b = 0.375 → c ≤ a ∧ d ≤ b → 
  c + d = 11 := by sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l147_14752


namespace NUMINAMATH_CALUDE_ceiling_equation_solution_l147_14756

theorem ceiling_equation_solution :
  ∃! x : ℝ, ⌈x⌉ * x + 15 = 210 :=
by
  -- The unique solution is 195/14
  use 195/14
  sorry

end NUMINAMATH_CALUDE_ceiling_equation_solution_l147_14756


namespace NUMINAMATH_CALUDE_new_men_average_age_greater_than_22_l147_14702

theorem new_men_average_age_greater_than_22 
  (A : ℝ) -- Age of the third man who is not replaced
  (B C : ℝ) -- Ages of the two new men
  (h1 : (A + B + C) / 3 > (A + 21 + 23) / 3) -- Average age increases after replacement
  : (B + C) / 2 > 22 := by
sorry

end NUMINAMATH_CALUDE_new_men_average_age_greater_than_22_l147_14702


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l147_14761

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 3*a + 2) (a - 2)).im ≠ 0 ∧ 
  (Complex.mk (a^2 - 3*a + 2) (a - 2)).re = 0 → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l147_14761


namespace NUMINAMATH_CALUDE_both_selected_probability_l147_14707

theorem both_selected_probability (ram_prob ravi_prob : ℚ) 
  (h1 : ram_prob = 2/7) 
  (h2 : ravi_prob = 1/5) : 
  ram_prob * ravi_prob = 2/35 := by
sorry

end NUMINAMATH_CALUDE_both_selected_probability_l147_14707


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l147_14710

theorem greatest_integer_radius (A : ℝ) (h : A < 80 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ r ≤ 8 ∧ ∀ (s : ℕ), s * s * Real.pi = A → s ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l147_14710


namespace NUMINAMATH_CALUDE_minimize_F_l147_14794

/-- The optimization problem -/
def OptimizationProblem (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧
  -2 * x₁ + x₂ + x₃ = 2 ∧
  x₁ - 2 * x₂ + x₄ = 2 ∧
  x₁ + x₂ + x₅ = 5

/-- The objective function -/
def F (x₁ x₂ : ℝ) : ℝ := x₂ - x₁

/-- The theorem stating the minimum value of F and the point where it's achieved -/
theorem minimize_F :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ),
    OptimizationProblem x₁ x₂ x₃ x₄ x₅ ∧
    F x₁ x₂ = -3 ∧
    x₁ = 4 ∧ x₂ = 1 ∧ x₃ = 9 ∧ x₄ = 0 ∧ x₅ = 0 ∧
    ∀ (y₁ y₂ y₃ y₄ y₅ : ℝ), OptimizationProblem y₁ y₂ y₃ y₄ y₅ → F y₁ y₂ ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_minimize_F_l147_14794


namespace NUMINAMATH_CALUDE_investment_problem_l147_14792

/-- Given two investors P and Q, where the profit is divided in the ratio 3:5
    and P invested 12000, prove that Q invested 20000. -/
theorem investment_problem (P Q : ℕ) (profit_ratio : ℚ) (P_investment : ℕ) :
  profit_ratio = 3 / 5 →
  P_investment = 12000 →
  Q = 20000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l147_14792


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l147_14750

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l147_14750


namespace NUMINAMATH_CALUDE_cubic_inequality_l147_14762

theorem cubic_inequality (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l147_14762


namespace NUMINAMATH_CALUDE_next_month_has_five_wednesdays_l147_14735

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its number of days and starting day -/
structure Month where
  days : Nat
  startDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Returns the next month given the current month -/
def nextMonth (m : Month) : Month :=
  sorry

/-- Theorem: If a month has 5 Saturdays, 5 Sundays, 4 Mondays, and 4 Fridays,
    then the following month will have 5 Wednesdays -/
theorem next_month_has_five_wednesdays (m : Month) :
  countDaysInMonth m DayOfWeek.Saturday = 5 →
  countDaysInMonth m DayOfWeek.Sunday = 5 →
  countDaysInMonth m DayOfWeek.Monday = 4 →
  countDaysInMonth m DayOfWeek.Friday = 4 →
  countDaysInMonth (nextMonth m) DayOfWeek.Wednesday = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_next_month_has_five_wednesdays_l147_14735


namespace NUMINAMATH_CALUDE_combined_share_specific_case_l147_14742

/-- Represents the share distribution problem -/
def ShareDistribution (total : ℚ) (ratio : List ℚ) : Prop :=
  total > 0 ∧ ratio.length = 5 ∧ ∀ r ∈ ratio, r > 0

/-- Calculates the combined share of two specific parts in the ratio -/
def CombinedShare (total : ℚ) (ratio : List ℚ) (index1 index2 : ℕ) : ℚ :=
  let sum_ratio := ratio.sum
  let part_value := total / sum_ratio
  part_value * (ratio[index1]! + ratio[index2]!)

theorem combined_share_specific_case :
  ∀ (total : ℚ) (ratio : List ℚ),
    ShareDistribution total ratio →
    ratio = [2, 4, 3, 1, 5] →
    total = 12000 →
    CombinedShare total ratio 3 4 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_combined_share_specific_case_l147_14742


namespace NUMINAMATH_CALUDE_z_coordinates_l147_14777

def z : ℂ := Complex.I * (2 - Complex.I)

theorem z_coordinates : z = Complex.ofReal 1 + Complex.I * Complex.ofReal 2 := by sorry

end NUMINAMATH_CALUDE_z_coordinates_l147_14777


namespace NUMINAMATH_CALUDE_inequality_proof_l147_14721

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a)^2 ≥ (3 / 2) * ((a + b) / c + (b + c) / a + (c + a) / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l147_14721


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l147_14748

theorem empty_solution_set_implies_a_range 
  (h : ∀ x : ℝ, ¬(|x + 3| + |x - 1| < a^2 - 3*a)) : 
  a ∈ Set.Icc (-1 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l147_14748


namespace NUMINAMATH_CALUDE_line_equation_from_conditions_l147_14763

/-- Vector in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Line in R² -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in R² -/
structure Point2D where
  x : ℝ
  y : ℝ

def vector_add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

def vector_scale (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

def is_perpendicular (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.a + v.y * l.b = 0

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_from_conditions 
  (a b : Vector2D)
  (A : Point2D)
  (l : Line2D)
  (h1 : a = ⟨6, 2⟩)
  (h2 : b = ⟨-4, 1/2⟩)
  (h3 : A = ⟨3, -1⟩)
  (h4 : is_perpendicular (vector_add a (vector_scale 2 b)) l)
  (h5 : point_on_line A l) :
  l = ⟨2, -3, -9⟩ :=
sorry

end NUMINAMATH_CALUDE_line_equation_from_conditions_l147_14763


namespace NUMINAMATH_CALUDE_octal_multiplication_53_26_l147_14728

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Multiplies two octal numbers --/
def octal_multiply (a b : ℕ) : ℕ :=
  decimal_to_octal (octal_to_decimal a * octal_to_decimal b)

theorem octal_multiplication_53_26 :
  octal_multiply 53 26 = 1662 := by sorry

end NUMINAMATH_CALUDE_octal_multiplication_53_26_l147_14728


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l147_14760

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | n + 1 => S a n + |a (n + 1) - 4|

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 = 8 * a 1) →
  arithmetic_sequence (λ n => match n with
                              | 1 => a 1
                              | 2 => a 2 + 1
                              | 3 => a 3
                              | _ => 0) →
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ n : ℕ, S a n = if n = 1 then 2 else 2^(n+1) - 4*n + 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l147_14760


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l147_14714

theorem chocolate_bar_count (milk_chocolate dark_chocolate white_chocolate : ℕ) 
  (h1 : milk_chocolate = 25)
  (h2 : dark_chocolate = 25)
  (h3 : white_chocolate = 25)
  (h4 : ∃ (total : ℕ), total > 0 ∧ 
    milk_chocolate = total / 4 ∧ 
    dark_chocolate = total / 4 ∧ 
    white_chocolate = total / 4) :
  ∃ (almond_chocolate : ℕ), almond_chocolate = 25 :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l147_14714


namespace NUMINAMATH_CALUDE_A_annual_income_is_537600_l147_14731

/-- The monthly income of person C in rupees -/
def C_monthly_income : ℕ := 16000

/-- The monthly income of person B in rupees -/
def B_monthly_income : ℕ := C_monthly_income + (C_monthly_income * 12 / 100)

/-- The monthly income of person A in rupees -/
def A_monthly_income : ℕ := B_monthly_income * 5 / 2

/-- The annual income of person A in rupees -/
def A_annual_income : ℕ := A_monthly_income * 12

/-- Theorem stating that A's annual income is 537600 rupees -/
theorem A_annual_income_is_537600 : A_annual_income = 537600 := by
  sorry

end NUMINAMATH_CALUDE_A_annual_income_is_537600_l147_14731


namespace NUMINAMATH_CALUDE_complex_colinear_l147_14736

/-- Two non-zero complex numbers lie on the same straight line if and only if their cross product is zero -/
theorem complex_colinear (a₁ b₁ a₂ b₂ : ℝ) (h₁ : a₁ + b₁ * I ≠ 0) (h₂ : a₂ + b₂ * I ≠ 0) :
  (∃ (t : ℝ), (a₂, b₂) = t • (a₁, b₁)) ↔ a₁ * b₂ = a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_complex_colinear_l147_14736


namespace NUMINAMATH_CALUDE_max_correct_answers_l147_14799

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 25 →
  correct_points = 6 →
  incorrect_points = -3 →
  total_score = 60 →
  (∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score) →
  (∀ (correct : ℕ),
    (∃ (incorrect unanswered : ℕ),
      correct + incorrect + unanswered = total_questions ∧
      correct * correct_points + incorrect * incorrect_points = total_score) →
    correct ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l147_14799


namespace NUMINAMATH_CALUDE_equation_solution_l147_14783

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3 ↔ x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l147_14783


namespace NUMINAMATH_CALUDE_area_of_circles_with_inscribed_rhombus_l147_14701

/-- A rhombus inscribed in the intersection of two equal circles -/
structure InscribedRhombus where
  /-- The length of one diagonal of the rhombus -/
  diagonal1 : ℝ
  /-- The length of the other diagonal of the rhombus -/
  diagonal2 : ℝ
  /-- The radius of each circle -/
  radius : ℝ
  /-- The diagonal1 is positive -/
  diagonal1_pos : 0 < diagonal1
  /-- The diagonal2 is positive -/
  diagonal2_pos : 0 < diagonal2
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The rhombus is inscribed in the intersection of the circles -/
  inscribed : (diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2 = radius ^ 2

/-- The theorem stating the relationship between the diagonals and the area of the circles -/
theorem area_of_circles_with_inscribed_rhombus 
  (r : InscribedRhombus) 
  (h1 : r.diagonal1 = 6) 
  (h2 : r.diagonal2 = 12) : 
  π * r.radius ^ 2 = (225 / 4) * π := by
sorry

end NUMINAMATH_CALUDE_area_of_circles_with_inscribed_rhombus_l147_14701


namespace NUMINAMATH_CALUDE_average_goals_l147_14774

theorem average_goals (layla_goals : ℕ) (kristin_difference : ℕ) (num_games : ℕ) :
  layla_goals = 104 →
  kristin_difference = 24 →
  num_games = 4 →
  (layla_goals + (layla_goals - kristin_difference)) / num_games = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_average_goals_l147_14774


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l147_14775

/-- A line with slope 3 passing through (-2, 4) has m + b = 13 when written as y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 3 → 
  4 = m * (-2) + b → 
  m + b = 13 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l147_14775


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero_l147_14713

theorem sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero :
  Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0 = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero_l147_14713


namespace NUMINAMATH_CALUDE_mutual_fund_range_increase_l147_14717

theorem mutual_fund_range_increase (n : ℕ) (increase_percent : ℝ) (new_range : ℝ) 
  (h1 : n = 100)
  (h2 : increase_percent = 0.15)
  (h3 : new_range = 11500) : 
  ∃ (old_range : ℝ), old_range * (1 + increase_percent) = new_range ∧ old_range = 10000 := by
  sorry

end NUMINAMATH_CALUDE_mutual_fund_range_increase_l147_14717


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l147_14746

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start → k < start + count → ¬(is_prime k)

theorem smallest_prime_after_seven_nonprimes :
  (∃ start : ℕ, consecutive_nonprimes start 7) ∧
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ ∃ start : ℕ, consecutive_nonprimes start 7 ∧ start + 7 ≤ p)) ∧
  is_prime 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l147_14746


namespace NUMINAMATH_CALUDE_bank_deposit_duration_l147_14711

theorem bank_deposit_duration (initial_deposit : ℝ) (interest_rate : ℝ) (final_amount : ℝ) :
  initial_deposit = 5600 →
  interest_rate = 0.07 →
  final_amount = 6384 →
  (final_amount - initial_deposit) / (interest_rate * initial_deposit) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_duration_l147_14711


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l147_14778

/-- Given points A, B, X, and Y on a Cartesian plane, prove that if AB is parallel to XY, then k = -8 -/
theorem parallel_segments_k_value (k : ℝ) : 
  let A : ℝ × ℝ := (-6, 2)
  let B : ℝ × ℝ := (2, -6)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (18, k)
  let slope (p q : ℝ × ℝ) := (q.2 - p.2) / (q.1 - p.1)
  slope A B = slope X Y → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l147_14778


namespace NUMINAMATH_CALUDE_arrangements_count_l147_14755

/-- The number of workers available for the production process -/
def total_workers : Nat := 6

/-- The number of steps in the production process -/
def total_steps : Nat := 4

/-- The set of workers who can oversee the first step -/
def first_step_workers : Finset Char := {'A', 'B'}

/-- The set of workers who can oversee the fourth step -/
def fourth_step_workers : Finset Char := {'A', 'C'}

/-- The function that calculates the number of arrangements -/
def count_arrangements : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of different arrangements is 36 -/
theorem arrangements_count : count_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l147_14755


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l147_14732

/-- The total distance Terrell hiked over two days -/
theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l147_14732


namespace NUMINAMATH_CALUDE_intersection_points_l147_14743

-- Define the curves
def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5/2
def curve2 (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2/4 = 1
def curve4 (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + y = Real.sqrt 5

-- Define a function to check if a curve has only one intersection point with the line
def hasOnlyOneIntersection (curve : (ℝ → ℝ → Prop)) : Prop :=
  ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line p.1 p.2

-- State the theorem
theorem intersection_points :
  hasOnlyOneIntersection curve1 ∧
  hasOnlyOneIntersection curve3 ∧
  hasOnlyOneIntersection curve4 ∧
  ¬hasOnlyOneIntersection curve2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_l147_14743


namespace NUMINAMATH_CALUDE_math_team_selection_count_l147_14790

theorem math_team_selection_count :
  let total_boys : ℕ := 7
  let total_girls : ℕ := 10
  let boys_needed : ℕ := 2
  let girls_needed : ℕ := 3
  (Nat.choose total_boys boys_needed) * (Nat.choose total_girls girls_needed) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_math_team_selection_count_l147_14790


namespace NUMINAMATH_CALUDE_birthday_money_ratio_l147_14749

theorem birthday_money_ratio : 
  let aunt_money : ℚ := 75
  let grandfather_money : ℚ := 150
  let bank_money : ℚ := 45
  let total_money := aunt_money + grandfather_money
  (bank_money / total_money) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_ratio_l147_14749


namespace NUMINAMATH_CALUDE_food_lasts_14_days_l147_14798

/-- Represents the amount of food each dog consumes per meal in grams -/
def dog_food_per_meal : List ℕ := [250, 350, 450, 550, 300, 400]

/-- Number of meals per day -/
def meals_per_day : ℕ := 3

/-- Weight of each sack in kilograms -/
def sack_weight_kg : ℕ := 50

/-- Number of sacks -/
def num_sacks : ℕ := 2

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

theorem food_lasts_14_days :
  let total_food_per_meal := dog_food_per_meal.sum
  let daily_consumption := total_food_per_meal * meals_per_day
  let total_food := num_sacks * sack_weight_kg * kg_to_g
  (total_food / daily_consumption : ℕ) = 14 := by sorry

end NUMINAMATH_CALUDE_food_lasts_14_days_l147_14798


namespace NUMINAMATH_CALUDE_cos_zeros_range_l147_14786

theorem cos_zeros_range (ω : ℝ) (h_pos : ω > 0) : 
  (∃ (z₁ z₂ z₃ : ℝ), z₁ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₂ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₃ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
                      Real.cos (ω * z₁ - Real.pi / 6) = 0 ∧
                      Real.cos (ω * z₂ - Real.pi / 6) = 0 ∧
                      Real.cos (ω * z₃ - Real.pi / 6) = 0 ∧
                      (∀ z ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi), 
                        Real.cos (ω * z - Real.pi / 6) = 0 → z = z₁ ∨ z = z₂ ∨ z = z₃)) →
  11 / 6 ≤ ω ∧ ω < 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cos_zeros_range_l147_14786


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l147_14726

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circles
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem triangle_area_theorem (ABC : Triangle) (circle1 circle2 : Circle) 
  (L K M N : ℝ × ℝ) :
  -- Given conditions
  circle1.radius = 1/18 →
  circle2.radius = 2/9 →
  (ABC.A.1 - L.1)^2 + (ABC.A.2 - L.2)^2 = (1/9)^2 →
  (ABC.C.1 - M.1)^2 + (ABC.C.2 - M.2)^2 = (1/6)^2 →
  -- Circle1 touches AB at L and AC at K
  ((ABC.A.1 - L.1)^2 + (ABC.A.2 - L.2)^2 = circle1.radius^2 ∧
   (ABC.B.1 - L.1)^2 + (ABC.B.2 - L.2)^2 = circle1.radius^2) →
  ((ABC.A.1 - K.1)^2 + (ABC.A.2 - K.2)^2 = circle1.radius^2 ∧
   (ABC.C.1 - K.1)^2 + (ABC.C.2 - K.2)^2 = circle1.radius^2) →
  -- Circle2 touches AC at N and BC at M
  ((ABC.A.1 - N.1)^2 + (ABC.A.2 - N.2)^2 = circle2.radius^2 ∧
   (ABC.C.1 - N.1)^2 + (ABC.C.2 - N.2)^2 = circle2.radius^2) →
  ((ABC.B.1 - M.1)^2 + (ABC.B.2 - M.2)^2 = circle2.radius^2 ∧
   (ABC.C.1 - M.1)^2 + (ABC.C.2 - M.2)^2 = circle2.radius^2) →
  -- Circles touch each other
  (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2 →
  -- Conclusion: Area of triangle ABC is 15/11
  abs ((ABC.B.1 - ABC.A.1) * (ABC.C.2 - ABC.A.2) - (ABC.C.1 - ABC.A.1) * (ABC.B.2 - ABC.A.2)) / 2 = 15/11 :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_theorem_l147_14726


namespace NUMINAMATH_CALUDE_roots_theorem_l147_14754

theorem roots_theorem :
  (∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2) ∧
  (∃ x y : ℝ, x^2 = 9 ∧ y^2 = 9 ∧ x = 3 ∧ y = -3) ∧
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_roots_theorem_l147_14754


namespace NUMINAMATH_CALUDE_y_value_at_x_4_l147_14709

/-- Given a function y = k * x^(1/4) where y = 3√2 when x = 81, 
    prove that y = 2 when x = 4 -/
theorem y_value_at_x_4 (k : ℝ) :
  (∀ x : ℝ, x > 0 → k * x^(1/4) = 3 * Real.sqrt 2 ↔ x = 81) →
  k * 4^(1/4) = 2 :=
by sorry

end NUMINAMATH_CALUDE_y_value_at_x_4_l147_14709
