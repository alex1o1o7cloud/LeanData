import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l1722_172236

theorem remainder_theorem (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1722_172236


namespace NUMINAMATH_CALUDE_triangle_radius_ratio_l1722_172206

/-- Given a triangle with area S, circumradius R, and inradius r, 
    such that S^2 = 2R^2 + 8Rr + 3r^2, prove that R/r = 2 or R/r ≥ √2 + 1 -/
theorem triangle_radius_ratio (S R r : ℝ) (h : S^2 = 2*R^2 + 8*R*r + 3*r^2) :
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_radius_ratio_l1722_172206


namespace NUMINAMATH_CALUDE_problem_solution_l1722_172257

theorem problem_solution (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  m^2 + (a + b) / 2 + (-x * y)^2023 = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1722_172257


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l1722_172295

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l1722_172295


namespace NUMINAMATH_CALUDE_a_profit_share_l1722_172219

-- Define the total investment and profit
def total_investment : ℕ := 90000
def total_profit : ℕ := 8640

-- Define the relationships between investments
def investment_relations (a b c : ℕ) : Prop :=
  a = b + 6000 ∧ b + 3000 = c ∧ a + b + c = total_investment

-- Define the profit sharing ratio
def profit_ratio (a b c : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ a / k = 11 ∧ b / k = 9 ∧ c / k = 10

-- Theorem statement
theorem a_profit_share (a b c : ℕ) :
  investment_relations a b c →
  profit_ratio a b c →
  (11 : ℚ) / 30 * total_profit = 3168 :=
by sorry

end NUMINAMATH_CALUDE_a_profit_share_l1722_172219


namespace NUMINAMATH_CALUDE_biscuit_dimensions_l1722_172242

theorem biscuit_dimensions (sheet_side : ℝ) (num_biscuits : ℕ) (biscuit_side : ℝ) : 
  sheet_side = 12 →
  num_biscuits = 16 →
  (sheet_side * sheet_side) = (biscuit_side * biscuit_side * num_biscuits) →
  biscuit_side = 3 := by
sorry

end NUMINAMATH_CALUDE_biscuit_dimensions_l1722_172242


namespace NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l1722_172218

-- Define the property of being in the fourth quadrant
def is_fourth_quadrant (θ : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + 3 * Real.pi / 2 ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi

-- Define the property of being in the second quadrant
def is_second_quadrant (θ : Real) : Prop :=
  ∃ k : Int, k * Real.pi + Real.pi / 2 ≤ θ ∧ θ ≤ k * Real.pi + Real.pi

-- State the theorem
theorem half_angle_in_second_quadrant (θ : Real) 
  (h1 : is_fourth_quadrant θ) 
  (h2 : |Real.cos (θ/2)| = -Real.cos (θ/2)) : 
  is_second_quadrant (θ/2) :=
sorry

end NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l1722_172218


namespace NUMINAMATH_CALUDE_total_balloons_l1722_172211

def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def mary_balloons : ℕ := 7

theorem total_balloons : fred_balloons + sam_balloons + mary_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l1722_172211


namespace NUMINAMATH_CALUDE_correct_proof_by_contradiction_components_l1722_172204

/-- Represents the components used in a proof by contradiction --/
inductive ProofByContradictionComponent
  | assumption
  | originalConditions
  | axiomTheoremsDefinitions
  | originalConclusion

/-- Defines the set of components used in a proof by contradiction --/
def proofByContradictionComponents : Set ProofByContradictionComponent :=
  {ProofByContradictionComponent.assumption,
   ProofByContradictionComponent.originalConditions,
   ProofByContradictionComponent.axiomTheoremsDefinitions}

/-- Theorem stating the correct components used in a proof by contradiction --/
theorem correct_proof_by_contradiction_components :
  proofByContradictionComponents =
    {ProofByContradictionComponent.assumption,
     ProofByContradictionComponent.originalConditions,
     ProofByContradictionComponent.axiomTheoremsDefinitions} :=
by
  sorry


end NUMINAMATH_CALUDE_correct_proof_by_contradiction_components_l1722_172204


namespace NUMINAMATH_CALUDE_prime_factor_puzzle_l1722_172229

theorem prime_factor_puzzle (a b c d w x y z : ℕ) : 
  w.Prime → x.Prime → y.Prime → z.Prime →
  w < x → x < y → y < z →
  (w^a) * (x^b) * (y^c) * (z^d) = 660 →
  (a + b) - (c + d) = 1 →
  b = 1 := by sorry

end NUMINAMATH_CALUDE_prime_factor_puzzle_l1722_172229


namespace NUMINAMATH_CALUDE_range_of_a_l1722_172277

theorem range_of_a (a : ℝ) 
  (h : ∀ x ∈ Set.Icc 3 4, x^2 - 3 > a*x - a) : 
  a < 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1722_172277


namespace NUMINAMATH_CALUDE_train_length_l1722_172260

/-- The length of a train given its crossing time, bridge length, and speed -/
theorem train_length (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 26.997840172786177 →
  bridge_length = 170 →
  train_speed_kmph = 36 →
  ∃ (train_length : ℝ), abs (train_length - 99.978) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1722_172260


namespace NUMINAMATH_CALUDE_supermarket_spending_l1722_172243

theorem supermarket_spending (total : ℚ) :
  (1 / 4 : ℚ) * total +
  (1 / 3 : ℚ) * total +
  (1 / 6 : ℚ) * total +
  6 = total →
  total = 24 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1722_172243


namespace NUMINAMATH_CALUDE_carol_blocks_l1722_172254

/-- Given that Carol starts with 42 blocks and loses 25 blocks, 
    prove that she ends with 17 blocks. -/
theorem carol_blocks : 
  let initial_blocks : ℕ := 42
  let lost_blocks : ℕ := 25
  initial_blocks - lost_blocks = 17 := by
  sorry

end NUMINAMATH_CALUDE_carol_blocks_l1722_172254


namespace NUMINAMATH_CALUDE_powers_of_two_sum_theorem_l1722_172282

/-- A sequence of powers of 2 -/
def PowersOfTwoSequence := List ℕ

/-- The sum of a sequence of powers of 2 -/
def sumPowersOfTwo (seq : PowersOfTwoSequence) : ℕ :=
  seq.foldl (λ sum power => sum + 2^power) 0

/-- The target sum we're aiming for -/
def targetSum : ℚ := (2^97 + 1) / (2^5 + 1)

/-- A proposition stating that a sequence of powers of 2 sums to the target sum -/
def sumsToTarget (seq : PowersOfTwoSequence) : Prop :=
  (sumPowersOfTwo seq : ℚ) = targetSum

/-- The main theorem: there exists a unique sequence of 10 powers of 2 that sums to the target -/
theorem powers_of_two_sum_theorem :
  ∃! (seq : PowersOfTwoSequence), seq.length = 10 ∧ sumsToTarget seq :=
sorry

end NUMINAMATH_CALUDE_powers_of_two_sum_theorem_l1722_172282


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1722_172296

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = 15 ∧ h_large / h_small = Real.sqrt area_ratio :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1722_172296


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l1722_172269

-- Define the propositions p and q
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Define the relationship between ¬p and q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, ¬(p x) → q x) ∧ ¬(∀ x, q x → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l1722_172269


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1722_172210

theorem two_digit_number_property (n : ℕ) : 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 + n % 10 = 3) →
  (n / 2 : ℚ) - (n / 4 : ℚ) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1722_172210


namespace NUMINAMATH_CALUDE_polynomial_sum_l1722_172298

/-- Two distinct polynomials with real coefficients -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

/-- The theorem statement -/
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ x, f a b x = 0 ∧ x = -c/2) →  -- x-coordinate of vertex of g is root of f
  (∃ x, g c d x = 0 ∧ x = -a/2) →  -- x-coordinate of vertex of f is root of g
  (∀ x, f a b x ≥ -144) →          -- minimum value of f is -144
  (∀ x, g c d x ≥ -144) →          -- minimum value of g is -144
  (∃ x, f a b x = -144) →          -- f achieves its minimum
  (∃ x, g c d x = -144) →          -- g achieves its minimum
  f a b 150 = -200 →               -- f(150) = -200
  g c d 150 = -200 →               -- g(150) = -200
  a + c = -300 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1722_172298


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1722_172268

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 2 + a 3 + a 4 = 15)
  (h_geom : is_geometric_sequence (λ n => 
    match n with
    | 1 => a 1 + 2
    | 2 => a 3 + 4
    | 3 => a 6 + 16
    | _ => 0
  )) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1722_172268


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1722_172264

-- Define the propositions
def p (x : ℝ) : Prop := -2 < x ∧ x < 0
def q (x : ℝ) : Prop := |x| < 2

-- Theorem statement
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1722_172264


namespace NUMINAMATH_CALUDE_min_sum_squares_l1722_172202

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : y₁ > 0) (pos₂ : y₂ > 0) (pos₃ : y₃ > 0)
  (sum_constraint : y₁ + 3*y₂ + 4*y₃ = 72) :
  ∃ (min : ℝ), min = 2592/13 ∧ 
  ∀ (z₁ z₂ z₃ : ℝ), z₁ > 0 → z₂ > 0 → z₃ > 0 → 
  z₁ + 3*z₂ + 4*z₃ = 72 → 
  z₁^2 + z₂^2 + z₃^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1722_172202


namespace NUMINAMATH_CALUDE_find_other_number_l1722_172258

theorem find_other_number (a b : ℕ) (ha : a = 36) 
  (hhcf : Nat.gcd a b = 20) (hlcm : Nat.lcm a b = 396) : b = 220 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l1722_172258


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l1722_172200

/-- Represents the cost calculation for a pizza order with special pricing --/
def pizza_order_cost (small_price medium_price large_price topping_price : ℚ)
  (triple_cheese_count triple_cheese_toppings : ℕ)
  (meat_lovers_count meat_lovers_toppings : ℕ)
  (veggie_delight_count veggie_delight_toppings : ℕ) : ℚ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * large_price + 
                            triple_cheese_count * triple_cheese_toppings * topping_price
  let meat_lovers_cost := ((meat_lovers_count + 1) / 3) * 2 * medium_price + 
                          meat_lovers_count * meat_lovers_toppings * topping_price
  let veggie_delight_cost := ((veggie_delight_count + 1) / 3) * 2 * small_price + 
                             veggie_delight_count * veggie_delight_toppings * topping_price
  triple_cheese_cost + meat_lovers_cost + veggie_delight_cost

/-- Theorem stating that the given pizza order costs $169 --/
theorem pizza_order_theorem : 
  pizza_order_cost 5 8 10 (5/2) 6 2 4 3 10 1 = 169 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l1722_172200


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1722_172253

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a^3 + t.b^3 + t.c^3) / (t.a + t.b + t.c) = t.c^2

def condition2 (t : Triangle) : Prop :=
  Real.sin t.α * Real.sin t.β = (Real.sin t.γ)^2

-- Define the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) 
  (h2 : condition2 t) : 
  t.a = t.b ∧ t.b = t.c := by
  sorry


end NUMINAMATH_CALUDE_triangle_is_equilateral_l1722_172253


namespace NUMINAMATH_CALUDE_library_meeting_problem_l1722_172255

theorem library_meeting_problem (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_prime : ∀ (p : ℕ), Prime p → ¬(p^2 ∣ z))
  (h_n : (x : ℝ) - y * Real.sqrt z = 120 - 60 * Real.sqrt 2)
  (h_prob : (14400 - (120 - (x - y * Real.sqrt z))^2) / 14400 = 1/2) :
  x + y + z = 182 := by
sorry

end NUMINAMATH_CALUDE_library_meeting_problem_l1722_172255


namespace NUMINAMATH_CALUDE_parallelogram_vertex_D_l1722_172216

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if ABCD forms a parallelogram -/
def isParallelogram (A B C D : Point3D) : Prop :=
  (B.x - A.x) + (D.x - C.x) = 0 ∧
  (B.y - A.y) + (D.y - C.y) = 0 ∧
  (B.z - A.z) + (D.z - C.z) = 0

theorem parallelogram_vertex_D :
  let A : Point3D := ⟨2, 0, 3⟩
  let B : Point3D := ⟨0, 3, -5⟩
  let C : Point3D := ⟨0, 0, 3⟩
  let D : Point3D := ⟨2, -3, 11⟩
  isParallelogram A B C D := by sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_D_l1722_172216


namespace NUMINAMATH_CALUDE_complex_power_sum_l1722_172266

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1/z^100 = -2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1722_172266


namespace NUMINAMATH_CALUDE_chord_dot_product_l1722_172222

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the focus
def chord_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t, -2*t) ∧ B = (1 + t, 2*t)

-- Theorem statement
theorem chord_dot_product (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → chord_through_focus A B →
  A.1 * B.1 + A.2 * B.2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_chord_dot_product_l1722_172222


namespace NUMINAMATH_CALUDE_second_athlete_high_jump_l1722_172286

def athlete1_long_jump : ℝ := 26
def athlete1_triple_jump : ℝ := 30
def athlete1_high_jump : ℝ := 7

def athlete2_long_jump : ℝ := 24
def athlete2_triple_jump : ℝ := 34

def winner_average_jump : ℝ := 22

def number_of_jumps : ℕ := 3

theorem second_athlete_high_jump :
  let athlete1_total := athlete1_long_jump + athlete1_triple_jump + athlete1_high_jump
  let athlete1_average := athlete1_total / number_of_jumps
  let athlete2_total_before_high := athlete2_long_jump + athlete2_triple_jump
  let winner_total := winner_average_jump * number_of_jumps
  athlete1_average < winner_average_jump →
  winner_total - athlete2_total_before_high = 8 := by
sorry

end NUMINAMATH_CALUDE_second_athlete_high_jump_l1722_172286


namespace NUMINAMATH_CALUDE_proposition_truth_values_l1722_172212

open Real

theorem proposition_truth_values :
  ∃ (p q : Prop),
  (∀ x, 0 < x → x < π / 2 → (p ↔ sin x > x)) ∧
  (∀ x, 0 < x → x < π / 2 → (q ↔ tan x > x)) ∧
  (¬(p ∧ q)) ∧
  (p ∨ q) ∧
  (¬(p ∨ ¬q)) ∧
  ((¬p) ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l1722_172212


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_union_B_intersect_C_nonempty_iff_l1722_172233

open Set

-- Define the sets A, B, and C
def A : Set ℝ := Ioc 2 3
def B : Set ℝ := Ioo 1 3
def C (m : ℝ) : Set ℝ := Ici m

-- Statement for part (1)
theorem complement_A_intersect_B : (Aᶜ ∩ B) = Ico 1 2 := by sorry

-- Statement for part (2)
theorem A_union_B_intersect_C_nonempty_iff (m : ℝ) :
  ((A ∪ B) ∩ C m).Nonempty ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_union_B_intersect_C_nonempty_iff_l1722_172233


namespace NUMINAMATH_CALUDE_segment_length_l1722_172249

/-- Given a line segment AB with points P, Q, and R, prove that AB has length 567 -/
theorem segment_length (A B P Q R : Real) : 
  (P - A) / (B - P) = 3 / 4 →  -- P divides AB in ratio 3:4
  (Q - A) / (B - Q) = 4 / 5 →  -- Q divides AB in ratio 4:5
  (R - P) / (Q - R) = 1 / 2 →  -- R divides PQ in ratio 1:2
  R - P = 3 →                  -- Length of PR is 3 units
  B - A = 567 := by            -- Length of AB is 567 units
  sorry


end NUMINAMATH_CALUDE_segment_length_l1722_172249


namespace NUMINAMATH_CALUDE_distance_from_two_equals_three_l1722_172220

theorem distance_from_two_equals_three (x : ℝ) : 
  |x - 2| = 3 ↔ x = 5 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_distance_from_two_equals_three_l1722_172220


namespace NUMINAMATH_CALUDE_lauryn_company_men_count_l1722_172246

theorem lauryn_company_men_count :
  ∀ (men women : ℕ),
    men + women = 180 →
    men = women - 20 →
    men = 80 := by
  sorry

end NUMINAMATH_CALUDE_lauryn_company_men_count_l1722_172246


namespace NUMINAMATH_CALUDE_legs_product_ge_parallel_sides_product_l1722_172247

/-- A trapezoid with perpendicular diagonals -/
structure PerpDiagonalTrapezoid where
  -- Parallel sides
  a : ℝ
  c : ℝ
  -- Legs
  b : ℝ
  d : ℝ
  -- All sides are positive
  a_pos : 0 < a
  c_pos : 0 < c
  b_pos : 0 < b
  d_pos : 0 < d
  -- Diagonals are perpendicular (using the property from the solution)
  perp_diag : b^2 + d^2 = a^2 + c^2

/-- 
  The product of the legs is at least as large as 
  the product of the parallel sides in a trapezoid 
  with perpendicular diagonals
-/
theorem legs_product_ge_parallel_sides_product (t : PerpDiagonalTrapezoid) : 
  t.b * t.d ≥ t.a * t.c := by
  sorry

end NUMINAMATH_CALUDE_legs_product_ge_parallel_sides_product_l1722_172247


namespace NUMINAMATH_CALUDE_no_real_roots_l1722_172299

theorem no_real_roots : 
  ¬∃ x : ℝ, (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l1722_172299


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1722_172208

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|2 * x₁ - 3| = 15) ∧ 
  (|2 * x₂ - 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 15) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1722_172208


namespace NUMINAMATH_CALUDE_infinitely_many_special_even_numbers_l1722_172232

theorem infinitely_many_special_even_numbers :
  ∃ (n : ℕ → ℕ), 
    (∀ k, Even (n k)) ∧ 
    (∀ k, n k < n (k + 1)) ∧
    (∀ k, (n k) ∣ (2^(n k) + 2)) ∧
    (∀ k, (n k - 1) ∣ (2^(n k) + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_even_numbers_l1722_172232


namespace NUMINAMATH_CALUDE_equidistant_function_property_l1722_172215

/-- Given a function g(z) = (c+di)z where c and d are real numbers,
    if g(z) is equidistant from z and the origin for all complex z,
    and |c+di| = 5, then d^2 = 99/4 -/
theorem equidistant_function_property (c d : ℝ) :
  (∀ z : ℂ, ‖(c + d * Complex.I) * z - z‖ = ‖(c + d * Complex.I) * z‖) →
  Complex.abs (c + d * Complex.I) = 5 →
  d^2 = 99/4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l1722_172215


namespace NUMINAMATH_CALUDE_opposite_number_l1722_172230

theorem opposite_number (a : ℤ) : (∀ b : ℤ, a + b = 0 → b = -2022) → a = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_l1722_172230


namespace NUMINAMATH_CALUDE_brand_z_percentage_approx_l1722_172290

/-- Represents the capacity of the fuel tank -/
def tank_capacity : ℚ := 12

/-- Represents the amount of brand Z gasoline after the final filling -/
def final_brand_z : ℚ := 10

/-- Represents the total amount of gasoline after the final filling -/
def final_total : ℚ := 12

/-- Calculates the percentage of a part relative to the whole -/
def percentage (part whole : ℚ) : ℚ := (part / whole) * 100

/-- Theorem stating that the percentage of brand Z gasoline is approximately 83.33% -/
theorem brand_z_percentage_approx : 
  abs (percentage final_brand_z final_total - 83.33) < 0.01 := by
  sorry

#eval percentage final_brand_z final_total

end NUMINAMATH_CALUDE_brand_z_percentage_approx_l1722_172290


namespace NUMINAMATH_CALUDE_tshirt_sale_revenue_l1722_172245

/-- Calculates the money made per minute during a t-shirt sale -/
def money_per_minute (total_shirts : ℕ) (sale_duration : ℕ) (black_price white_price : ℚ) : ℚ :=
  let black_shirts := total_shirts / 2
  let white_shirts := total_shirts / 2
  let total_revenue := (black_shirts : ℚ) * black_price + (white_shirts : ℚ) * white_price
  total_revenue / (sale_duration : ℚ)

/-- Proves that the money made per minute during the specific t-shirt sale is $220 -/
theorem tshirt_sale_revenue : money_per_minute 200 25 30 25 = 220 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sale_revenue_l1722_172245


namespace NUMINAMATH_CALUDE_square_position_2010_l1722_172240

/-- Represents the positions of the square's vertices -/
inductive SquarePosition
| ABCD
| CABD
| DACB
| BCAD
| ADCB
| CBDA
| BADC
| CDAB

/-- Applies the transformation sequence to a given position -/
def applyTransformation (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CABD
  | SquarePosition.CABD => SquarePosition.DACB
  | SquarePosition.DACB => SquarePosition.BCAD
  | SquarePosition.BCAD => SquarePosition.ADCB
  | SquarePosition.ADCB => SquarePosition.CBDA
  | SquarePosition.CBDA => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.ABCD

/-- Returns the position after n transformations -/
def nthPosition (n : Nat) : SquarePosition :=
  match n % 8 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.CABD
  | 2 => SquarePosition.DACB
  | 3 => SquarePosition.BCAD
  | 4 => SquarePosition.ADCB
  | 5 => SquarePosition.CBDA
  | 6 => SquarePosition.BADC
  | 7 => SquarePosition.CDAB
  | _ => SquarePosition.ABCD  -- This case should never occur due to % 8

theorem square_position_2010 :
  nthPosition 2010 = SquarePosition.CABD := by
  sorry

end NUMINAMATH_CALUDE_square_position_2010_l1722_172240


namespace NUMINAMATH_CALUDE_probability_third_smallest_is_five_l1722_172203

def set_size : ℕ := 15
def selection_size : ℕ := 8
def target_number : ℕ := 5
def target_position : ℕ := 3

theorem probability_third_smallest_is_five :
  let total_combinations := Nat.choose set_size selection_size
  let favorable_combinations := 
    (Nat.choose (set_size - target_number) (selection_size - target_position)) *
    (Nat.choose (target_number - 1) (target_position - 1))
  (favorable_combinations : ℚ) / total_combinations = 4 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_smallest_is_five_l1722_172203


namespace NUMINAMATH_CALUDE_no_integer_points_between_l1722_172280

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- The line passing through points A(2, 3) and B(50, 500) -/
def line (p : IntPoint) : Prop :=
  (p.y - 3) * 48 = 497 * (p.x - 2)

/-- A point is strictly between A and B if its x-coordinate is between 2 and 50 exclusively -/
def strictly_between (p : IntPoint) : Prop :=
  2 < p.x ∧ p.x < 50

theorem no_integer_points_between : 
  ¬ ∃ p : IntPoint, line p ∧ strictly_between p :=
sorry

end NUMINAMATH_CALUDE_no_integer_points_between_l1722_172280


namespace NUMINAMATH_CALUDE_equation_to_general_form_l1722_172256

theorem equation_to_general_form :
  ∀ x : ℝ, (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_to_general_form_l1722_172256


namespace NUMINAMATH_CALUDE_pet_food_discount_l1722_172272

theorem pet_food_discount (msrp : ℝ) (regular_discount : ℝ) (final_price : ℝ) (additional_discount : ℝ) : 
  msrp = 40 →
  regular_discount = 0.3 →
  final_price = 22.4 →
  additional_discount = (msrp * (1 - regular_discount) - final_price) / (msrp * (1 - regular_discount)) →
  additional_discount = 0.2 := by
sorry

end NUMINAMATH_CALUDE_pet_food_discount_l1722_172272


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l1722_172231

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- The last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set of the sequence -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l1722_172231


namespace NUMINAMATH_CALUDE_ice_cream_volume_l1722_172251

/-- The volume of ice cream in a cone with a spherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let sphere_volume := (4 / 3) * π * r^3
  h = 12 ∧ r = 3 → cone_volume + sphere_volume = 72 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l1722_172251


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1722_172279

/-- Proves that a rectangle with perimeter 34 and length 5 more than width has width 6 and length 11 -/
theorem rectangle_dimensions :
  ∀ (w l : ℕ), 
    (2 * w + 2 * l = 34) →  -- Perimeter is 34
    (l = w + 5) →           -- Length is 5 more than width
    (w = 6 ∧ l = 11) :=     -- Width is 6 and length is 11
by
  sorry

#check rectangle_dimensions

end NUMINAMATH_CALUDE_rectangle_dimensions_l1722_172279


namespace NUMINAMATH_CALUDE_mary_dog_walking_earnings_l1722_172263

/-- 
Given:
- Mary earns $20 washing cars each month
- Mary earns D dollars walking dogs each month
- Mary saves half of her total earnings each month
- It takes Mary 5 months to save $150

Prove that D = $40
-/
theorem mary_dog_walking_earnings (D : ℝ) : 
  (5 : ℝ) * ((20 + D) / 2) = 150 → D = 40 := by sorry

end NUMINAMATH_CALUDE_mary_dog_walking_earnings_l1722_172263


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1722_172241

theorem regular_polygon_sides (n : ℕ) (h : n > 0) : 
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1722_172241


namespace NUMINAMATH_CALUDE_min_period_cosine_l1722_172289

/-- The minimum positive period of the cosine function Y = 3cos(2/5x - π/6) is 5π. -/
theorem min_period_cosine (x : ℝ) : 
  let Y : ℝ → ℝ := λ x => 3 * Real.cos ((2/5) * x - π/6)
  ∃ (T : ℝ), T > 0 ∧ (∀ t, Y (t + T) = Y t) ∧ (∀ S, S > 0 ∧ (∀ t, Y (t + S) = Y t) → T ≤ S) ∧ T = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_min_period_cosine_l1722_172289


namespace NUMINAMATH_CALUDE_ruth_total_score_l1722_172274

-- Define the given conditions
def dean_total_points : ℕ := 252
def dean_games : ℕ := 28
def games_difference : ℕ := 10
def average_difference : ℚ := 1/2

-- Define Ruth's games
def ruth_games : ℕ := dean_games - games_difference

-- Define Dean's average
def dean_average : ℚ := dean_total_points / dean_games

-- Define Ruth's average
def ruth_average : ℚ := dean_average + average_difference

-- Theorem to prove
theorem ruth_total_score : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ruth_total_score_l1722_172274


namespace NUMINAMATH_CALUDE_cycle_price_proof_l1722_172226

theorem cycle_price_proof (sale_price : ℝ) (gain_percentage : ℝ) 
  (h1 : sale_price = 1440)
  (h2 : gain_percentage = 60) : 
  ∃ original_price : ℝ, 
    original_price = 900 ∧ 
    sale_price = original_price + (gain_percentage / 100) * original_price :=
by
  sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l1722_172226


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1722_172294

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (a x : ℝ) : ℝ := 2 * |x - a|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x - g 2 x ≤ x - 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

-- Part 2
theorem range_of_a :
  (∀ m > 1, ∃ x₀ : ℝ, f x₀ + g a x₀ ≤ (m^2 + m + 4) / (m - 1)) →
  a ∈ Set.Icc (-2 * Real.sqrt 6 - 2) (2 * Real.sqrt 6 + 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1722_172294


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1722_172292

theorem right_triangle_hypotenuse : 
  ∀ (a : ℝ), a > 0 → a^2 = 8^2 + 15^2 → a = 17 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1722_172292


namespace NUMINAMATH_CALUDE_least_b_with_conditions_l1722_172252

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least positive integer with a given number of factors -/
def least_with_factors (k : ℕ) : ℕ+ := sorry

theorem least_b_with_conditions (a b : ℕ+) 
  (ha : num_factors a = 4)
  (hb : num_factors b = 2 * (num_factors a))
  (hdiv : b.val % a.val = 0) :
  b ≥ 60 ∧ ∃ (a₀ b₀ : ℕ+), 
    num_factors a₀ = 4 ∧ 
    num_factors b₀ = 2 * (num_factors a₀) ∧ 
    b₀.val % a₀.val = 0 ∧ 
    b₀ = 60 := by sorry

end NUMINAMATH_CALUDE_least_b_with_conditions_l1722_172252


namespace NUMINAMATH_CALUDE_wall_length_is_800_l1722_172201

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

theorem wall_length_is_800 (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 25)
    (h2 : brick.width = 11.25)
    (h3 : brick.height = 6)
    (h4 : wall.width = 600)
    (h5 : wall.height = 22.5)
    (h6 : brickVolume brick * 6400 = wallVolume wall) :
    wall.length = 800 := by
  sorry

#check wall_length_is_800

end NUMINAMATH_CALUDE_wall_length_is_800_l1722_172201


namespace NUMINAMATH_CALUDE_dog_eaten_cost_l1722_172225

-- Define the ingredients and their costs
def flour_cost : ℝ := 3.20
def sugar_cost : ℝ := 2.10
def butter_cost : ℝ := 5.50
def egg_cost : ℝ := 0.45
def baking_soda_cost : ℝ := 0.60
def baking_powder_cost : ℝ := 1.30
def salt_cost : ℝ := 0.35
def vanilla_extract_cost : ℝ := 1.75
def milk_cost : ℝ := 1.40
def vegetable_oil_cost : ℝ := 2.10

-- Define the quantities of ingredients
def flour_qty : ℝ := 2.5
def sugar_qty : ℝ := 1.5
def butter_qty : ℝ := 0.75
def egg_qty : ℝ := 4
def baking_soda_qty : ℝ := 1
def baking_powder_qty : ℝ := 1
def salt_qty : ℝ := 1
def vanilla_extract_qty : ℝ := 1
def milk_qty : ℝ := 1.25
def vegetable_oil_qty : ℝ := 0.75

-- Define other constants
def sales_tax_rate : ℝ := 0.07
def total_slices : ℕ := 12
def mother_eaten_slices : ℕ := 4

-- Theorem to prove
theorem dog_eaten_cost (total_cost : ℝ) (cost_with_tax : ℝ) (cost_per_slice : ℝ) :
  total_cost = flour_cost * flour_qty + sugar_cost * sugar_qty + butter_cost * butter_qty +
               egg_cost * egg_qty + baking_soda_cost * baking_soda_qty + 
               baking_powder_cost * baking_powder_qty + salt_cost * salt_qty +
               vanilla_extract_cost * vanilla_extract_qty + milk_cost * milk_qty +
               vegetable_oil_cost * vegetable_oil_qty →
  cost_with_tax = total_cost * (1 + sales_tax_rate) →
  cost_per_slice = cost_with_tax / total_slices →
  cost_per_slice * (total_slices - mother_eaten_slices) = 17.44 :=
by sorry

end NUMINAMATH_CALUDE_dog_eaten_cost_l1722_172225


namespace NUMINAMATH_CALUDE_not_perfect_square_l1722_172285

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), 2 * 13^n + 5 * 7^n + 26 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1722_172285


namespace NUMINAMATH_CALUDE_parking_lot_cars_remaining_l1722_172228

theorem parking_lot_cars_remaining (initial_cars : ℕ) 
  (first_group_left : ℕ) (second_group_left : ℕ) : 
  initial_cars = 24 → first_group_left = 8 → second_group_left = 6 →
  initial_cars - first_group_left - second_group_left = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_remaining_l1722_172228


namespace NUMINAMATH_CALUDE_adjacent_empty_seats_l1722_172275

theorem adjacent_empty_seats (n : ℕ) (k : ℕ) : n = 6 → k = 3 →
  (number_of_arrangements : ℕ) →
  (number_of_arrangements = 
    -- Case 1: Two adjacent empty seats at the ends
    (2 * (Nat.choose 3 1) * (Nat.choose 3 2)) +
    -- Case 2: Two adjacent empty seats not at the ends
    (3 * (Nat.choose 3 2) * (Nat.choose 2 1))) →
  number_of_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_adjacent_empty_seats_l1722_172275


namespace NUMINAMATH_CALUDE_supermarket_spending_l1722_172265

theorem supermarket_spending (total : ℝ) (category1 : ℝ) (category2 : ℝ) (category3 : ℝ) (category4 : ℝ) :
  total = 120 →
  category1 = (1 / 2) * total →
  category2 = (1 / 10) * total →
  category3 = 8 →
  category1 + category2 + category3 + category4 = total →
  category4 / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1722_172265


namespace NUMINAMATH_CALUDE_triangle_properties_l1722_172205

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (∀ a' b' : ℝ, a' + b' + c ≤ 6) ∧
  (2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C →
   1/2 * a * b * Real.sin C = 2 * Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1722_172205


namespace NUMINAMATH_CALUDE_smaller_cuboid_height_l1722_172283

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- The original large cuboid -/
def original : Cuboid := { length := 18, width := 15, height := 2 }

/-- The smaller cuboid with unknown height -/
def smaller (h : ℝ) : Cuboid := { length := 5, width := 6, height := h }

/-- The number of smaller cuboids that can be formed -/
def num_smaller : ℕ := 6

/-- Theorem: The height of each smaller cuboid is 3 meters -/
theorem smaller_cuboid_height :
  ∃ h : ℝ, volume original = num_smaller * volume (smaller h) ∧ h = 3 := by
  sorry


end NUMINAMATH_CALUDE_smaller_cuboid_height_l1722_172283


namespace NUMINAMATH_CALUDE_power_76_mod_7_l1722_172235

theorem power_76_mod_7 (n : ℕ) (h : Odd n) : 76^n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_76_mod_7_l1722_172235


namespace NUMINAMATH_CALUDE_square_root_problem_l1722_172209

theorem square_root_problem (c d : ℕ) (h : 241 * c + 214 = d^2) : d = 334 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1722_172209


namespace NUMINAMATH_CALUDE_election_total_votes_l1722_172238

/-- Represents the total number of votes in the election -/
def total_votes : ℕ := sorry

/-- Represents the vote percentage for Candidate A -/
def candidate_a_percentage : ℚ := 30 / 100

/-- Represents the vote percentage for Candidate B -/
def candidate_b_percentage : ℚ := 25 / 100

/-- Represents the vote difference between Candidate A and Candidate B -/
def vote_difference_a_b : ℕ := 1800

theorem election_total_votes : 
  (candidate_a_percentage - candidate_b_percentage) * total_votes = vote_difference_a_b ∧ 
  total_votes = 36000 := by sorry

end NUMINAMATH_CALUDE_election_total_votes_l1722_172238


namespace NUMINAMATH_CALUDE_triangle_side_length_l1722_172273

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 2 →
  b = Real.sqrt 3 - 1 →
  C = π / 6 →
  c^2 = 5 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1722_172273


namespace NUMINAMATH_CALUDE_percentage_increase_l1722_172224

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 600 → final = 660 → (final - initial) / initial * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1722_172224


namespace NUMINAMATH_CALUDE_ant_count_approximation_l1722_172214

/-- Represents the dimensions and ant densities of a park -/
structure ParkInfo where
  width : ℝ
  length : ℝ
  mainDensity : ℝ
  squareSide : ℝ
  squareDensity : ℝ

/-- Calculates the total number of ants in the park -/
def totalAnts (park : ParkInfo) : ℝ :=
  let totalArea := park.width * park.length
  let squareArea := park.squareSide * park.squareSide
  let mainArea := totalArea - squareArea
  let mainAnts := mainArea * 144 * park.mainDensity  -- Convert to square inches
  let squareAnts := squareArea * park.squareDensity
  mainAnts + squareAnts

/-- The park information as given in the problem -/
def givenPark : ParkInfo := {
  width := 250
  length := 350
  mainDensity := 4
  squareSide := 50
  squareDensity := 6
}

/-- Theorem stating that the total number of ants is approximately 50 million -/
theorem ant_count_approximation :
  abs (totalAnts givenPark - 50000000) ≤ 1000000 := by
  sorry

end NUMINAMATH_CALUDE_ant_count_approximation_l1722_172214


namespace NUMINAMATH_CALUDE_gold_quarter_value_ratio_is_80_l1722_172271

/-- Represents the ratio of melted gold value to face value for gold quarters -/
def gold_quarter_value_ratio : ℚ :=
  let quarter_weight : ℚ := 1 / 5
  let melted_gold_value_per_ounce : ℚ := 100
  let quarter_face_value : ℚ := 1 / 4
  let quarters_per_ounce : ℚ := 1 / quarter_weight
  let melted_value_per_quarter : ℚ := melted_gold_value_per_ounce * quarter_weight
  melted_value_per_quarter / quarter_face_value

/-- Theorem stating that the gold quarter value ratio is 80 -/
theorem gold_quarter_value_ratio_is_80 : gold_quarter_value_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_gold_quarter_value_ratio_is_80_l1722_172271


namespace NUMINAMATH_CALUDE_equation_solutions_l1722_172288

theorem equation_solutions :
  let f : ℝ → ℝ := fun x ↦ x * (x - 3)^2 * (5 - x)
  {x : ℝ | f x = 0} = {0, 3, 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1722_172288


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l1722_172291

def v : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (a : ℕ) : 
  (∀ k : ℕ, k ≤ 30 → k > 0 → v % 3^k = 0) → 
  (∀ m : ℕ, m > a → ¬(v % 3^m = 0)) → 
  a = 14 := by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l1722_172291


namespace NUMINAMATH_CALUDE_parabola_properties_l1722_172227

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem parabola_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1722_172227


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l1722_172248

theorem remainder_of_sum_of_powers (n : ℕ) : (9^24 + 12^37) % 23 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l1722_172248


namespace NUMINAMATH_CALUDE_cylindrical_eight_queens_impossible_l1722_172270

/-- Represents a position on the cylindrical chessboard -/
structure Position :=
  (x : Fin 8) -- column
  (y : Fin 8) -- row

/-- Checks if two positions are attacking each other on the cylindrical chessboard -/
def isAttacking (p1 p2 : Position) : Prop :=
  p1.x = p2.x ∨ 
  p1.y = p2.y ∨ 
  (p1.x.val - p2.x.val) % 8 = (p1.y.val - p2.y.val) % 8 ∨
  (p1.x.val - p2.x.val) % 8 = (p2.y.val - p1.y.val) % 8

/-- A configuration of 8 queens on the cylindrical chessboard -/
def QueenConfiguration := Fin 8 → Position

/-- Theorem: It's impossible to place 8 queens on a cylindrical chessboard without attacks -/
theorem cylindrical_eight_queens_impossible :
  ∀ (config : QueenConfiguration), 
    ∃ (i j : Fin 8), i ≠ j ∧ isAttacking (config i) (config j) := by
  sorry


end NUMINAMATH_CALUDE_cylindrical_eight_queens_impossible_l1722_172270


namespace NUMINAMATH_CALUDE_triangle_properties_l1722_172223

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove properties about angle A and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 - a^2 + b*c = 0 →
  Real.sin C = Real.sqrt 2 / 2 →
  a = Real.sqrt 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  A = 2 * Real.pi / 3 ∧
  (1/2 * a * c * Real.sin B) = (3 - Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1722_172223


namespace NUMINAMATH_CALUDE_cubic_polynomial_with_rational_roots_l1722_172293

def P (x : ℚ) : ℚ := x^3 + x^2 - x - 1

theorem cubic_polynomial_with_rational_roots :
  ∃ (r₁ r₂ r₃ : ℚ), 
    (∀ x, P x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ 
    (r₁ ≠ r₂ ∨ r₁ ≠ r₃ ∨ r₂ ≠ r₃) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_with_rational_roots_l1722_172293


namespace NUMINAMATH_CALUDE_cloth_selling_price_l1722_172239

/-- Calculates the total selling price of cloth given the quantity, cost price, and loss per metre -/
def total_selling_price (quantity : ℕ) (cost_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  quantity * (cost_price - loss_per_metre)

/-- Theorem stating that the total selling price of 200 metres of cloth
    with a cost price of Rs. 72 per metre and a loss of Rs. 12 per metre
    is Rs. 12,000 -/
theorem cloth_selling_price :
  total_selling_price 200 72 12 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l1722_172239


namespace NUMINAMATH_CALUDE_sequence_properties_l1722_172278

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_property (n : ℕ) : S n + n = 2 * sequence_a n

def sequence_b (n : ℕ) : ℝ := n * sequence_a n + n

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n - 1) ∧
  (∀ n : ℕ, n ≥ 11 → (T n - 2) / n > 2018) ∧
  (∀ n : ℕ, n < 11 → (T n - 2) / n ≤ 2018) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1722_172278


namespace NUMINAMATH_CALUDE_buttons_needed_for_shirts_l1722_172261

theorem buttons_needed_for_shirts 
  (shirts_per_kid : ℕ)
  (num_kids : ℕ)
  (buttons_per_shirt : ℕ)
  (h1 : shirts_per_kid = 3)
  (h2 : num_kids = 3)
  (h3 : buttons_per_shirt = 7) :
  shirts_per_kid * num_kids * buttons_per_shirt = 63 :=
by sorry

end NUMINAMATH_CALUDE_buttons_needed_for_shirts_l1722_172261


namespace NUMINAMATH_CALUDE_base_conversion_185_to_113_l1722_172297

/-- Converts a base 13 number to base 10 --/
def base13ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 13^2 + tens * 13^1 + ones * 13^0

/-- Checks if a number is a valid base 13 digit --/
def isValidBase13Digit (d : Nat) : Prop :=
  d < 13

theorem base_conversion_185_to_113 :
  (∀ d, isValidBase13Digit d → d < 13) →
  base13ToBase10 1 1 3 = 185 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_185_to_113_l1722_172297


namespace NUMINAMATH_CALUDE_middle_trapezoid_radius_l1722_172250

/-- Given a trapezoid divided into three similar trapezoids by lines parallel to the bases,
    each with an inscribed circle, this theorem proves that the radius of the middle circle
    is the geometric mean of the radii of the other two circles. -/
theorem middle_trapezoid_radius (R r x : ℝ) 
  (h_positive : R > 0 ∧ r > 0 ∧ x > 0) 
  (h_similar : r / x = x / R) : 
  x = Real.sqrt (r * R) := by
sorry

end NUMINAMATH_CALUDE_middle_trapezoid_radius_l1722_172250


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1722_172213

/-- Given vectors a, b, and c in ℝ², prove that if (a + k * c) is parallel to (2 * b - a), then k = -16/13 -/
theorem parallel_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (3, 2)) 
    (hb : b = (-1, 2)) 
    (hc : c = (4, 1)) 
    (h_parallel : ∃ (t : ℝ), t • (a.1 + k * c.1, a.2 + k * c.2) = (2 * b.1 - a.1, 2 * b.2 - a.2)) :
  k = -16/13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1722_172213


namespace NUMINAMATH_CALUDE_ab_nonpositive_l1722_172244

theorem ab_nonpositive (a b : ℝ) : (∀ x, (2*a + b)*x - 1 ≠ 0) → a*b ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l1722_172244


namespace NUMINAMATH_CALUDE_min_value_theorem_l1722_172234

-- Define a positive term geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  is_positive_geometric_sequence a →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  a 6 = a 5 + 2 * a 4 →
  (∀ k l : ℕ, 1 / k + 4 / l ≥ 3 / 2) ∧
  (∃ k l : ℕ, 1 / k + 4 / l = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1722_172234


namespace NUMINAMATH_CALUDE_stick_ratio_proof_l1722_172259

/-- Prove that the ratio of the uncovered portion of Pat's stick to Sarah's stick is 1/2 -/
theorem stick_ratio_proof (pat_stick : ℕ) (pat_covered : ℕ) (jane_stick : ℕ) (sarah_stick : ℕ) : 
  pat_stick = 30 →
  pat_covered = 7 →
  jane_stick = 22 →
  sarah_stick = jane_stick + 24 →
  (pat_stick - pat_covered : ℚ) / sarah_stick = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_stick_ratio_proof_l1722_172259


namespace NUMINAMATH_CALUDE_derivative_roots_in_triangle_l1722_172287

/-- A polynomial of degree three with complex roots -/
def cubic_polynomial (a b c : ℂ) (x : ℂ) : ℂ :=
  (x - a) * (x - b) * (x - c)

/-- The derivative of the cubic polynomial -/
def cubic_derivative (a b c : ℂ) (x : ℂ) : ℂ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

/-- The triangle formed by the roots of the cubic polynomial -/
def root_triangle (a b c : ℂ) : Set ℂ :=
  {z : ℂ | ∃ (t₁ t₂ t₃ : ℝ), t₁ + t₂ + t₃ = 1 ∧ t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ z = t₁ * a + t₂ * b + t₃ * c}

/-- Theorem stating that the roots of the derivative lie inside the triangle formed by the roots of the original polynomial -/
theorem derivative_roots_in_triangle (a b c : ℂ) :
  ∀ z : ℂ, cubic_derivative a b c z = 0 → z ∈ root_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_derivative_roots_in_triangle_l1722_172287


namespace NUMINAMATH_CALUDE_power_sum_equation_l1722_172217

theorem power_sum_equation (p : ℕ) (a : ℤ) (n : ℕ) :
  Nat.Prime p → (2^p : ℤ) + 3^p = a^n → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equation_l1722_172217


namespace NUMINAMATH_CALUDE_fifi_closet_total_hangers_l1722_172276

/-- The number of colored hangers in Fifi's closet -/
def total_hangers (pink green blue yellow : ℕ) : ℕ := pink + green + blue + yellow

/-- The conditions of Fifi's closet hangers -/
def fifi_closet_conditions (pink green blue yellow : ℕ) : Prop :=
  pink = 7 ∧ green = 4 ∧ blue = green - 1 ∧ yellow = blue - 1

/-- Theorem: The total number of colored hangers in Fifi's closet is 16 -/
theorem fifi_closet_total_hangers :
  ∃ (pink green blue yellow : ℕ),
    fifi_closet_conditions pink green blue yellow ∧
    total_hangers pink green blue yellow = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifi_closet_total_hangers_l1722_172276


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_1021_l1722_172267

theorem modular_inverse_17_mod_1021 (p : Nat) (prime_p : Nat.Prime p) (h : p = 1021) :
  (17 * 961) % p = 1 :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_1021_l1722_172267


namespace NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l1722_172262

theorem z_greater_than_w_by_50_percent 
  (w x y z : ℝ) 
  (hw : w = 0.6 * x) 
  (hx : x = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
sorry

end NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l1722_172262


namespace NUMINAMATH_CALUDE_abs_f_decreasing_on_4_6_l1722_172207

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem abs_f_decreasing_on_4_6 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_sym : is_symmetric_about f 2)
  (h_inc : is_increasing_on f (-2) 0)
  (h_nonneg : f (-2) ≥ 0) :
  ∀ x₁ x₂, 4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 6 → |f x₁| > |f x₂| :=
sorry

end NUMINAMATH_CALUDE_abs_f_decreasing_on_4_6_l1722_172207


namespace NUMINAMATH_CALUDE_some_number_value_l1722_172281

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 49 * 45 * 25) : n = 21 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1722_172281


namespace NUMINAMATH_CALUDE_age_difference_constant_l1722_172284

theorem age_difference_constant (seokjin_initial_age mother_initial_age years_passed : ℕ) :
  mother_initial_age - seokjin_initial_age = 
  (mother_initial_age + years_passed) - (seokjin_initial_age + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_age_difference_constant_l1722_172284


namespace NUMINAMATH_CALUDE_quadratic_function_ratio_l1722_172237

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_at_neg_two : ∀ x : ℝ, a * x^2 + b * x + c ≤ a^2
  max_value : a * (-2)^2 + b * (-2) + c = a^2
  passes_through_point : a * (-1)^2 + b * (-1) + c = 6

/-- Theorem stating that (a + c) / b = 1/2 for the given quadratic function -/
theorem quadratic_function_ratio (f : QuadraticFunction) : (f.a + f.c) / f.b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_ratio_l1722_172237


namespace NUMINAMATH_CALUDE_initial_players_l1722_172221

theorem initial_players (initial_players : ℕ) : 
  (∀ (players : ℕ), 
    (players = initial_players + 2) →
    (7 * players = 63)) →
  initial_players = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_players_l1722_172221
