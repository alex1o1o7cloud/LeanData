import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2327_232721

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P Q F : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F = (c, 0) →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →
  (Q.1 - c/3)^2 + Q.2^2 = b^2/9 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 * ((Q.1 - F.1)^2 + (Q.2 - F.2)^2) →
  (P.1 - F.1) * (Q.1 - c/3) + (P.2 - F.2) * Q.2 = 0 →
  c^2 = a^2 + b^2 →
  c / a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2327_232721


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_l2327_232722

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define function f
def f : A → ℝ := fun x ↦ 2 - x

-- Define the range of f as B
def B : Set ℝ := Set.range f

-- Theorem statement
theorem intersection_complement_A_and_B :
  (Set.univ \ A) ∩ B = Set.Ioo 1 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_l2327_232722


namespace NUMINAMATH_CALUDE_length_of_diagonal_l2327_232709

/-- Given two triangles AOC and BOD sharing a vertex O, with specified side lengths,
    prove that the length of AC is √1036/7 -/
theorem length_of_diagonal (AO BO CO DO BD : ℝ) (x : ℝ) 
    (h1 : AO = 3)
    (h2 : CO = 5)
    (h3 : BO = 7)
    (h4 : DO = 6)
    (h5 : BD = 11)
    (h6 : x = Real.sqrt (AO^2 + CO^2 - 2*AO*CO*(BO^2 + DO^2 - BD^2)/(2*BO*DO))) :
  x = Real.sqrt 1036 / 7 := by
  sorry

end NUMINAMATH_CALUDE_length_of_diagonal_l2327_232709


namespace NUMINAMATH_CALUDE_sqrt_of_square_of_negative_l2327_232767

theorem sqrt_of_square_of_negative : ∀ (x : ℝ), x < 0 → Real.sqrt (x^2) = -x := by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_of_negative_l2327_232767


namespace NUMINAMATH_CALUDE_red_balls_unchanged_l2327_232744

/-- A box containing colored balls -/
structure Box where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Remove one blue ball from the box -/
def removeOneBlueBall (b : Box) : Box :=
  { red := b.red, blue := b.blue - 1, yellow := b.yellow }

theorem red_balls_unchanged (initial : Box) (h : initial.blue ≥ 1) :
  (removeOneBlueBall initial).red = initial.red :=
by sorry

end NUMINAMATH_CALUDE_red_balls_unchanged_l2327_232744


namespace NUMINAMATH_CALUDE_julia_age_proof_l2327_232727

/-- Julia's age in years -/
def julia_age : ℚ := 20 / 7

/-- Julia's mother's age in years -/
def mother_age : ℚ := 15 * julia_age

theorem julia_age_proof :
  (mother_age - julia_age = 40) ∧ (mother_age = 15 * julia_age) →
  julia_age = 20 / 7 := by
sorry

end NUMINAMATH_CALUDE_julia_age_proof_l2327_232727


namespace NUMINAMATH_CALUDE_condition_one_condition_two_l2327_232704

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for condition (1)
theorem condition_one :
  ∃! a : ℝ, A a ∩ B = A a ∪ B := by sorry

-- Theorem for condition (2)
theorem condition_two :
  ∃! a : ℝ, (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) := by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_l2327_232704


namespace NUMINAMATH_CALUDE_equation_with_added_constant_l2327_232791

theorem equation_with_added_constant (y : ℝ) (n : ℝ) :
  y^4 - 20*y + 1 = 22 ∧ n = 3 →
  y^4 - 20*y + (1 + n) = 22 + n :=
by sorry

end NUMINAMATH_CALUDE_equation_with_added_constant_l2327_232791


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l2327_232750

/-- Triangle ABC with inscribed square PQRS -/
structure InscribedSquareTriangle where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of BC -/
  bc : ℝ
  /-- Side length of CA -/
  ca : ℝ
  /-- Point P lies on BC -/
  p_on_bc : Bool
  /-- Point R lies on BC -/
  r_on_bc : Bool
  /-- Point Q lies on CA -/
  q_on_ca : Bool
  /-- Point S lies on AB -/
  s_on_ab : Bool

/-- The side length of the inscribed square PQRS -/
def squareSideLength (t : InscribedSquareTriangle) : ℝ := sorry

/-- Theorem: The side length of the inscribed square is 42 -/
theorem inscribed_square_side_length 
  (t : InscribedSquareTriangle) 
  (h1 : t.ab = 13) 
  (h2 : t.bc = 14) 
  (h3 : t.ca = 15) 
  (h4 : t.p_on_bc = true) 
  (h5 : t.r_on_bc = true) 
  (h6 : t.q_on_ca = true) 
  (h7 : t.s_on_ab = true) : 
  squareSideLength t = 42 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l2327_232750


namespace NUMINAMATH_CALUDE_parallel_linear_functions_touch_theorem_l2327_232710

/-- Two linear functions that are parallel but not parallel to the coordinate axes -/
structure ParallelLinearFunctions where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The condition that (f(x))^2 touches 20g(x) -/
def touches_condition_1 (f : ParallelLinearFunctions) : Prop :=
  ∃! x : ℝ, (f.a * x + f.b)^2 = 20 * (f.a * x + f.c)

/-- The condition that (g(x))^2 touches f(x)/A -/
def touches_condition_2 (f : ParallelLinearFunctions) (A : ℝ) : Prop :=
  ∃! x : ℝ, (f.a * x + f.c)^2 = (f.a * x + f.b) / A

/-- The main theorem -/
theorem parallel_linear_functions_touch_theorem (f : ParallelLinearFunctions) :
  touches_condition_1 f → (touches_condition_2 f A ↔ A = -1/20) :=
sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_touch_theorem_l2327_232710


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l2327_232778

/-- Converts a three-digit number in base 9 to base 10 --/
def base9_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The age of the alien spaceship --/
def spaceship_age : Nat := 362

theorem spaceship_age_conversion :
  base9_to_base10 3 6 2 = 299 :=
by sorry

end NUMINAMATH_CALUDE_spaceship_age_conversion_l2327_232778


namespace NUMINAMATH_CALUDE_quadratic_root_inequality_l2327_232779

theorem quadratic_root_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) 
  (h3 : a * 1^2 + b * 1 + c = 0) : -2 ≤ c / a ∧ c / a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_inequality_l2327_232779


namespace NUMINAMATH_CALUDE_alpha_plus_beta_value_l2327_232713

theorem alpha_plus_beta_value (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/4))
  (h2 : β ∈ Set.Ioo 0 (π/4))
  (h3 : α.sin * (3*π/2 + α).cos - (π/2 + α).sin * α.cos = -3/5)
  (h4 : 3 * β.sin = (2*α + β).sin) :
  α + β = π/4 := by sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_value_l2327_232713


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2327_232786

theorem consecutive_integers_average (a : ℕ) (c : ℕ) (h1 : c = 3 * a + 3) : 
  (c + (c + 1) + (c + 2)) / 3 = 3 * a + 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2327_232786


namespace NUMINAMATH_CALUDE_dylans_mother_hotdogs_l2327_232766

theorem dylans_mother_hotdogs (helens_hotdogs : ℕ) (total_hotdogs : ℕ) 
  (h1 : helens_hotdogs = 101)
  (h2 : total_hotdogs = 480) :
  total_hotdogs - helens_hotdogs = 379 := by
  sorry

end NUMINAMATH_CALUDE_dylans_mother_hotdogs_l2327_232766


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2327_232740

theorem complex_number_theorem (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^4 + z^4) = 56 := by
sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2327_232740


namespace NUMINAMATH_CALUDE_matrix_operation_result_l2327_232795

theorem matrix_operation_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; 0, -3]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![6, 0; -1, 8]
  A - B + C = !![12, -7; 1, 16] := by
sorry

end NUMINAMATH_CALUDE_matrix_operation_result_l2327_232795


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2327_232738

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 3) / (x - 2) - 1 = 3 / x
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation 1
theorem equation1_solution :
  (∃! x : ℝ, equation1 x) ∧ equation1 (3/2) :=
sorry

-- Theorem for equation 2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2327_232738


namespace NUMINAMATH_CALUDE_simplify_expression_l2327_232702

theorem simplify_expression (y : ℝ) : 5*y + 6*y + 7*y + 2 = 18*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2327_232702


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2327_232768

theorem arithmetic_sequence_ninth_term
  (a : ℝ) (d : ℝ) -- first term and common difference
  (h1 : a + 2 * d = 23) -- third term is 23
  (h2 : a + 5 * d = 29) -- sixth term is 29
  : a + 8 * d = 35 := -- ninth term is 35
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2327_232768


namespace NUMINAMATH_CALUDE_true_propositions_l2327_232708

-- Define the propositions
def p₁ : Prop := ∀ a b : ℝ, a < b → a^2 < b^2
def p₂ : Prop := ∀ x : ℝ, x > 0 → Real.sin x < x
def p₃ : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f x / f (-x) = -1) ↔ (∀ x : ℝ, f (-x) = -f x)
def p₄ : Prop := ∀ a : ℕ → ℝ, (∀ n : ℕ, a (n+1) = a n * (a 2 / a 1)) →
  ((a 1 > a 2 ∧ a 2 > a 3) ↔ (∀ n : ℕ, a (n+1) < a n))

-- Theorem stating which propositions are true
theorem true_propositions :
  ¬p₁ ∧ p₂ ∧ ¬p₃ ∧ p₄ :=
sorry

end NUMINAMATH_CALUDE_true_propositions_l2327_232708


namespace NUMINAMATH_CALUDE_one_fourth_of_8_point_8_l2327_232700

theorem one_fourth_of_8_point_8 : 
  (8.8 / 4 : ℚ) = 11 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_point_8_l2327_232700


namespace NUMINAMATH_CALUDE_x_value_proof_l2327_232777

theorem x_value_proof (x : ℝ) (h : 9 / x^2 = x / 25) : x = (225 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2327_232777


namespace NUMINAMATH_CALUDE_range_of_a_l2327_232765

def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) ↔ a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2327_232765


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_simplify_expression3_l2327_232734

-- Expression 1
theorem simplify_expression1 (a b x : ℝ) (h : b ≠ 0) :
  (12 * a^3 * x^4 + 2 * a^2 * x^5) / (18 * a * b^2 * x + 3 * b^2 * x^2) = 
  (2 * a^2 * x^3) / (3 * b^2) :=
sorry

-- Expression 2
theorem simplify_expression2 (x : ℝ) (h : x ≠ -2) :
  (4 - 2*x + x^2) / (x + 2) - x - 2 = -6*x / (x + 2) :=
sorry

-- Expression 3
theorem simplify_expression3 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  1 / ((a-b)*(a-c)) + 1 / ((b-a)*(b-c)) + 1 / ((c-a)*(c-b)) = 0 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_simplify_expression3_l2327_232734


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2327_232728

theorem divisibility_equivalence (x y : ℤ) :
  (2 * x + 3 * y) % 7 = 0 ↔ (5 * x + 4 * y) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2327_232728


namespace NUMINAMATH_CALUDE_fruit_theorem_l2327_232717

def fruit_problem (apples pears plums cherries : ℕ) : Prop :=
  apples = 180 ∧
  apples = 3 * plums ∧
  pears = 2 * plums ∧
  cherries = 4 * apples ∧
  251 = apples - (13 * apples / 15) +
        plums - (5 * plums / 6) +
        pears - (3 * pears / 4) +
        cherries - (37 * cherries / 50)

theorem fruit_theorem :
  ∃ (apples pears plums cherries : ℕ),
    fruit_problem apples pears plums cherries := by
  sorry

end NUMINAMATH_CALUDE_fruit_theorem_l2327_232717


namespace NUMINAMATH_CALUDE_toy_piles_l2327_232776

theorem toy_piles (total : ℕ) (small : ℕ) (large : ℕ) : 
  total = 120 → 
  large = 2 * small → 
  total = small + large → 
  large = 80 := by
sorry

end NUMINAMATH_CALUDE_toy_piles_l2327_232776


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l2327_232733

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (7, 0)
def circle2_radius : ℝ := 1

-- Define the tangent line (implicitly)
def tangent_line : Set (ℝ × ℝ) := sorry

-- Condition that the tangent line touches both circles in the first quadrant
axiom tangent_touches_circles :
  ∃ (p q : ℝ × ℝ),
    p.1 > 0 ∧ p.2 > 0 ∧
    q.1 > 0 ∧ q.2 > 0 ∧
    p ∈ tangent_line ∧
    q ∈ tangent_line ∧
    (p.1 - circle1_center.1)^2 + (p.2 - circle1_center.2)^2 = circle1_radius^2 ∧
    (q.1 - circle2_center.1)^2 + (q.2 - circle2_center.2)^2 = circle2_radius^2

-- Theorem statement
theorem tangent_y_intercept :
  ∃ (y : ℝ), y = 9 ∧ (0, y) ∈ tangent_line :=
sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l2327_232733


namespace NUMINAMATH_CALUDE_solution_count_l2327_232752

/-- The number of distinct ordered pairs of non-negative integers (a, b) that sum to 50 -/
def count_solutions : ℕ := 51

/-- Predicate for valid solutions -/
def is_valid_solution (a b : ℕ) : Prop := a + b = 50

theorem solution_count :
  (∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_solution p.1 p.2) ∧ 
    s.card = count_solutions) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l2327_232752


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2327_232743

/-- Given a cube with surface area 150 square inches, its volume is 125 cubic inches -/
theorem cube_volume_from_surface_area :
  ∀ (edge_length : ℝ),
  (6 * edge_length^2 = 150) →
  edge_length^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2327_232743


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2327_232757

/-- The line x - ky + 1 = 0 (k ∈ ℝ) always intersects the circle x^2 + y^2 + 4x - 2y + 2 = 0 -/
theorem line_intersects_circle (k : ℝ) : 
  ∃ (x y : ℝ), 
    (x - k*y + 1 = 0) ∧ 
    (x^2 + y^2 + 4*x - 2*y + 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_line_intersects_circle_l2327_232757


namespace NUMINAMATH_CALUDE_vote_increase_l2327_232724

/-- Represents the voting scenario for a bill --/
structure VotingScenario where
  total_members : ℕ
  initial_for : ℕ
  initial_against : ℕ
  revote_for : ℕ
  revote_against : ℕ

/-- Conditions for the voting scenario --/
def voting_conditions (v : VotingScenario) : Prop :=
  v.total_members = 500 ∧
  v.initial_for + v.initial_against = v.total_members ∧
  v.initial_against > v.initial_for ∧
  v.revote_for + v.revote_against = v.total_members ∧
  v.revote_for = (10 * v.initial_against) / 9 ∧
  (v.revote_for - v.revote_against) = 3 * (v.initial_against - v.initial_for)

/-- Theorem stating the increase in votes for the bill --/
theorem vote_increase (v : VotingScenario) (h : voting_conditions v) :
  v.revote_for - v.initial_for = 59 :=
sorry

end NUMINAMATH_CALUDE_vote_increase_l2327_232724


namespace NUMINAMATH_CALUDE_parabola_intersection_l2327_232746

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 9 * x - 15
  let g (x : ℝ) := x^2 - 5 * x + 7
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ 
    (x = 1 + 2 * Real.sqrt 3 ∧ y = 19 - 6 * Real.sqrt 3) ∨
    (x = 1 - 2 * Real.sqrt 3 ∧ y = 19 + 6 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2327_232746


namespace NUMINAMATH_CALUDE_proportion_reciprocal_outer_terms_l2327_232789

theorem proportion_reciprocal_outer_terms (a b c d : ℚ) : 
  (a / b = c / d) →  -- proportion
  (b * c = 1) →      -- middle terms are reciprocals
  (a = 7 / 9) →      -- one outer term is 7/9
  (d = 9 / 7) :=     -- other outer term is 9/7
by
  sorry


end NUMINAMATH_CALUDE_proportion_reciprocal_outer_terms_l2327_232789


namespace NUMINAMATH_CALUDE_baker_cakes_l2327_232769

theorem baker_cakes (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (h1 : pastries_made = 169)
  (h2 : cakes_sold = pastries_sold + 11)
  (h3 : cakes_sold = 158)
  (h4 : pastries_sold = 147) :
  pastries_made + 11 = 180 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l2327_232769


namespace NUMINAMATH_CALUDE_probability_at_least_nine_correct_l2327_232756

theorem probability_at_least_nine_correct (n : ℕ) (p : ℝ) : 
  n = 10 → 
  p = 1/4 → 
  let P := (n.choose 9) * p^9 * (1-p)^1 + (n.choose 10) * p^10
  ∃ ε > 0, abs (P - 3e-5) < ε := by sorry

end NUMINAMATH_CALUDE_probability_at_least_nine_correct_l2327_232756


namespace NUMINAMATH_CALUDE_expression_evaluation_l2327_232796

/-- Evaluates the given expression for x = 1.5 and y = -2 -/
theorem expression_evaluation :
  let x : ℝ := 1.5
  let y : ℝ := -2
  let expr := (1.2 * x^3 + 4 * y) * (0.86)^3 - (0.1)^3 / (0.86)^2 + 0.086 + (0.1)^2 * (2 * x^2 - 3 * y^2)
  ∃ ε > 0, |expr + 2.5027737774| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2327_232796


namespace NUMINAMATH_CALUDE_smallest_value_of_x_plus_yz_l2327_232726

theorem smallest_value_of_x_plus_yz (x y z : ℕ+) (h : x * y + z = 160) :
  ∃ (a b c : ℕ+), a * b + c = 160 ∧ a + b * c = 64 ∧ ∀ (p q r : ℕ+), p * q + r = 160 → p + q * r ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_x_plus_yz_l2327_232726


namespace NUMINAMATH_CALUDE_books_for_girls_l2327_232714

theorem books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ) : 
  num_girls = 15 → 
  num_boys = 10 → 
  total_books = 375 → 
  (num_girls * (total_books / (num_girls + num_boys))) = 225 := by
sorry

end NUMINAMATH_CALUDE_books_for_girls_l2327_232714


namespace NUMINAMATH_CALUDE_system_solution_l2327_232712

theorem system_solution (a : ℝ) (x y z : ℝ) :
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = a^2) →
  (x^3 + y^3 + z^3 = a^3) →
  ((x = a ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = a ∧ z = 0) ∨
   (x = 0 ∧ y = 0 ∧ z = a)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2327_232712


namespace NUMINAMATH_CALUDE_negation_of_forall_not_equal_l2327_232705

theorem negation_of_forall_not_equal (x : ℝ) :
  (¬ ∀ x > 0, Real.log x ≠ x - 1) ↔ (∃ x > 0, Real.log x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_not_equal_l2327_232705


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2327_232725

-- Problem 1
theorem problem_1 : (-8) + 10 - 2 + (-1) = -1 := by sorry

-- Problem 2
theorem problem_2 : 12 - 7 * (-4) + 8 / (-2) = 36 := by sorry

-- Problem 3
theorem problem_3 : (1/2 + 1/3 - 1/6) / (-1/18) = -12 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 + 0.5) * (1/3) / (-4)^2 = -33/32 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2327_232725


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l2327_232759

def monthly_salary : ℝ := 6250
def initial_savings_rate : ℝ := 0.20
def final_savings : ℝ := 250

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - final_savings
  let percentage_increase := (expense_increase / initial_expenses) * 100
  percentage_increase = 20 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l2327_232759


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2327_232715

theorem min_value_x_plus_2y (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : 1 / (x + 1) + 2 / y = 1) : 
  ∀ z, x + 2 * y ≤ z → 8 ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2327_232715


namespace NUMINAMATH_CALUDE_third_score_proof_l2327_232763

/-- Given three scores with an average of 122, where two scores are 118 and 125, prove the third score is 123. -/
theorem third_score_proof (average : ℝ) (score1 score2 : ℝ) (h_average : average = 122) 
  (h_score1 : score1 = 118) (h_score2 : score2 = 125) : 
  3 * average - (score1 + score2) = 123 := by
  sorry

end NUMINAMATH_CALUDE_third_score_proof_l2327_232763


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l2327_232739

def choose (n k : ℕ) : ℕ := Nat.choose n k

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ring_arrangements_count : 
  let total_rings : ℕ := 10
  let arranged_rings : ℕ := 6
  let fingers : ℕ := 4
  choose total_rings arranged_rings * factorial arranged_rings * choose (arranged_rings + fingers - 1) (fingers - 1) = 9130560 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l2327_232739


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourths_l2327_232737

theorem cos_seven_pi_fourths : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourths_l2327_232737


namespace NUMINAMATH_CALUDE_jimmy_bread_packs_l2327_232792

/-- The number of packs of bread needed for a given number of sandwiches -/
def bread_packs_needed (num_sandwiches : ℕ) (slices_per_sandwich : ℕ) (slices_per_pack : ℕ) (initial_slices : ℕ) : ℕ :=
  ((num_sandwiches * slices_per_sandwich - initial_slices) + slices_per_pack - 1) / slices_per_pack

/-- Theorem: Jimmy needs 4 packs of bread for his picnic -/
theorem jimmy_bread_packs : bread_packs_needed 8 2 4 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_bread_packs_l2327_232792


namespace NUMINAMATH_CALUDE_train_length_l2327_232747

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ (length : ℝ), abs (length - 250.05) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2327_232747


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l2327_232732

def digits : List Nat := [9, 3, 1, 2, 6, 4]

def max_occurrences : Nat := 2

def largest_number : Nat := 99664332211

def smallest_number : Nat := 1122334699

theorem difference_largest_smallest :
  largest_number - smallest_number = 98541997512 :=
by sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l2327_232732


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2327_232770

/-- The line equation y = kx + 1 - 2k -/
def line (k x : ℝ) : ℝ := k * x + 1 - 2 * k

/-- The ellipse equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The point P(2,1) is inside the ellipse -/
def point_inside_ellipse : Prop := 2^2 / 9 + 1^2 / 4 < 1

theorem line_intersects_ellipse :
  ∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2327_232770


namespace NUMINAMATH_CALUDE_share_decrease_proof_l2327_232742

theorem share_decrease_proof (total : ℕ) (c_share : ℕ) (b_decrease : ℕ) (c_decrease : ℕ) 
  (h_total : total = 1010)
  (h_c_share : c_share = 495)
  (h_b_decrease : b_decrease = 10)
  (h_c_decrease : c_decrease = 15) :
  ∃ (a_share b_share : ℕ) (x : ℕ),
    a_share + b_share + c_share = total ∧
    (a_share - x) / 3 = (b_share - b_decrease) / 2 ∧
    (a_share - x) / 3 = (c_share - c_decrease) / 5 ∧
    x = 25 := by
  sorry

end NUMINAMATH_CALUDE_share_decrease_proof_l2327_232742


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2327_232730

theorem expand_and_simplify (x : ℝ) : 2 * (x + 3) * (x + 8) = 2 * x^2 + 22 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2327_232730


namespace NUMINAMATH_CALUDE_equation_equivalence_implies_uvw_product_l2327_232782

theorem equation_equivalence_implies_uvw_product (a b x y : ℝ) (u v w : ℤ) :
  (a^10 * x * y - a^9 * y - a^8 * x = a^6 * (b^5 - 1)) →
  ((a^u * x - a^v) * (a^w * y - a^3) = a^6 * b^5) →
  u * v * w = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_implies_uvw_product_l2327_232782


namespace NUMINAMATH_CALUDE_shopping_mall_goods_problem_l2327_232772

/-- Shopping mall goods problem -/
theorem shopping_mall_goods_problem 
  (total_cost_A : ℝ) 
  (total_cost_B : ℝ) 
  (cost_diff : ℝ) 
  (selling_price_A : ℝ) 
  (selling_price_B : ℝ) 
  (discount_rate : ℝ) 
  (min_profit : ℝ)
  (h1 : total_cost_A = 2000)
  (h2 : total_cost_B = 2400)
  (h3 : cost_diff = 8)
  (h4 : selling_price_A = 60)
  (h5 : selling_price_B = 88)
  (h6 : discount_rate = 0.3)
  (h7 : min_profit = 2460)
  : ∃ (cost_price_A cost_price_B : ℝ) (min_units_A : ℕ),
    cost_price_A = 40 ∧ 
    cost_price_B = 48 ∧ 
    min_units_A = 20 ∧
    (total_cost_A / cost_price_A = total_cost_B / cost_price_B) ∧
    (selling_price_A - cost_price_A) * min_units_A + 
    (selling_price_A * (1 - discount_rate) - cost_price_A) * (total_cost_A / cost_price_A - min_units_A) + 
    (selling_price_B - cost_price_B) * (total_cost_B / cost_price_B) ≥ min_profit :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_goods_problem_l2327_232772


namespace NUMINAMATH_CALUDE_paint_remaining_l2327_232773

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  let remaining_after_day1 := initial_paint - (1/4 * initial_paint)
  let remaining_after_day2 := remaining_after_day1 - (1/2 * remaining_after_day1)
  remaining_after_day2 = 3/8 * initial_paint := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l2327_232773


namespace NUMINAMATH_CALUDE_nehas_mother_twice_age_l2327_232790

/-- Represents the age difference between Neha's mother and Neha when the mother will be twice Neha's age -/
def AgeDifference (n : ℕ) : Prop :=
  ∃ (neha_age : ℕ),
    -- Neha's mother's current age is 60
    60 = neha_age + n ∧
    -- 12 years ago, Neha's mother was 4 times Neha's age
    (60 - 12) = 4 * (neha_age - 12) ∧
    -- In n years, Neha's mother will be twice as old as Neha
    (60 + n) = 2 * (neha_age + n)

/-- The number of years until Neha's mother is twice as old as Neha is 12 -/
theorem nehas_mother_twice_age : AgeDifference 12 := by
  sorry

end NUMINAMATH_CALUDE_nehas_mother_twice_age_l2327_232790


namespace NUMINAMATH_CALUDE_contradiction_proof_l2327_232723

theorem contradiction_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l2327_232723


namespace NUMINAMATH_CALUDE_range_of_m_l2327_232729

/-- Given the equation (m+3)/(x-1) = 1 where x is a positive number, 
    prove that the range of m is m > -4 and m ≠ -3 -/
theorem range_of_m (m : ℝ) (x : ℝ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) :
  m > -4 ∧ m ≠ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2327_232729


namespace NUMINAMATH_CALUDE_complex_expression_equals_four_l2327_232780

theorem complex_expression_equals_four :
  (1/2)⁻¹ - Real.sqrt 3 * Real.tan (30 * π / 180) + (π - 2023)^0 + |-2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_four_l2327_232780


namespace NUMINAMATH_CALUDE_fraction_simplification_l2327_232735

theorem fraction_simplification :
  (12 : ℚ) / 11 * 15 / 28 * 44 / 45 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2327_232735


namespace NUMINAMATH_CALUDE_minimum_students_l2327_232784

theorem minimum_students (b g : ℕ) : 
  (3 * b = 4 * g) →  -- Equal number of boys and girls passed
  (∃ k : ℕ, b = 4 * k ∧ g = 3 * k) →  -- b and g are integers
  (b + g ≥ 7) ∧ (∀ m n : ℕ, (3 * m = 4 * n) → (m + n < 7 → m = 0 ∨ n = 0)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_students_l2327_232784


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2327_232719

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 13 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2327_232719


namespace NUMINAMATH_CALUDE_set_equality_l2327_232749

open Set

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x ≤ 3}

theorem set_equality : (Mᶜ ∩ (M ∩ N)) = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2327_232749


namespace NUMINAMATH_CALUDE_daily_allowance_calculation_l2327_232758

/-- Proves that if a person saves half of their daily allowance for 6 days
    and a quarter of their daily allowance for 1 day, and the total saved is $39,
    then their daily allowance is $12. -/
theorem daily_allowance_calculation (allowance : ℚ) : 
  (6 * (allowance / 2) + 1 * (allowance / 4) = 39) → allowance = 12 := by
  sorry

end NUMINAMATH_CALUDE_daily_allowance_calculation_l2327_232758


namespace NUMINAMATH_CALUDE_third_measurement_is_integer_meters_l2327_232787

def tape_length : ℕ := 100
def length1 : ℕ := 600
def length2 : ℕ := 500

theorem third_measurement_is_integer_meters :
  ∃ (k : ℕ), ∀ (third_length : ℕ),
    (tape_length ∣ length1) ∧
    (tape_length ∣ length2) ∧
    (tape_length ∣ third_length) →
    ∃ (n : ℕ), third_length = n * 100 := by
  sorry

end NUMINAMATH_CALUDE_third_measurement_is_integer_meters_l2327_232787


namespace NUMINAMATH_CALUDE_inscribed_triangle_condition_l2327_232760

/-- A rectangle with side lengths a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- An equilateral triangle inscribed in a rectangle such that one vertex is at A
    and the other two vertices lie on sides BC and CD respectively. -/
structure InscribedTriangle (rect : Rectangle) where
  vertex_on_BC : ℝ
  vertex_on_CD : ℝ
  vertex_on_BC_in_range : 0 ≤ vertex_on_BC ∧ vertex_on_BC ≤ rect.b
  vertex_on_CD_in_range : 0 ≤ vertex_on_CD ∧ vertex_on_CD ≤ rect.a
  is_equilateral : True  -- We assume this condition is met

/-- The theorem stating the condition for inscribing an equilateral triangle in a rectangle. -/
theorem inscribed_triangle_condition (rect : Rectangle) :
  (∃ t : InscribedTriangle rect, True) ↔ 
  (Real.sqrt 3 / 2 ≤ rect.a / rect.b ∧ rect.a / rect.b ≤ 2 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_condition_l2327_232760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2327_232741

theorem arithmetic_sequence_squares (k : ℤ) : k = 1612 →
  ∃ (a d : ℤ), 
    (25 + k = (a - d)^2) ∧ 
    (289 + k = a^2) ∧ 
    (529 + k = (a + d)^2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2327_232741


namespace NUMINAMATH_CALUDE_olivia_hourly_rate_l2327_232755

/-- Olivia's hourly rate given her work hours and total earnings --/
theorem olivia_hourly_rate (monday_hours wednesday_hours friday_hours total_earnings : ℕ) :
  monday_hours = 4 →
  wednesday_hours = 3 →
  friday_hours = 6 →
  total_earnings = 117 →
  (total_earnings : ℚ) / (monday_hours + wednesday_hours + friday_hours : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_hourly_rate_l2327_232755


namespace NUMINAMATH_CALUDE_parabola_vertex_l2327_232703

/-- The parabola defined by the equation y = 3(x-1)^2 + 2 has vertex at (1, 2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3*(x-1)^2 + 2 → (1, 2) = (x, y) := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2327_232703


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2327_232771

theorem quadratic_equation_roots (k : ℝ) (θ : ℝ) : 
  (∃ x y : ℝ, x = Real.sin θ ∧ y = Real.cos θ ∧ 
    8 * x^2 + 6 * k * x + 2 * k + 1 = 0 ∧
    8 * y^2 + 6 * k * y + 2 * k + 1 = 0) →
  k = -10/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2327_232771


namespace NUMINAMATH_CALUDE_mr_langsley_arrival_time_l2327_232753

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define addition operation for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hour * 60 + t1.minute + t2.hour * 60 + t2.minute
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- Define the problem parameters
def pickup_time : Time := { hour := 6, minute := 0 }
def time_to_first_station : Time := { hour := 0, minute := 40 }
def time_from_first_station_to_work : Time := { hour := 2, minute := 20 }

-- Theorem to prove
theorem mr_langsley_arrival_time :
  (pickup_time.add time_to_first_station).add time_from_first_station_to_work = { hour := 9, minute := 0 } := by
  sorry


end NUMINAMATH_CALUDE_mr_langsley_arrival_time_l2327_232753


namespace NUMINAMATH_CALUDE_cos_20_cos_10_minus_sin_160_sin_10_l2327_232781

theorem cos_20_cos_10_minus_sin_160_sin_10 :
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_cos_10_minus_sin_160_sin_10_l2327_232781


namespace NUMINAMATH_CALUDE_plate_cup_cost_theorem_l2327_232731

/-- The cost of plates and cups -/
structure CostStructure where
  plate_cost : ℝ
  cup_cost : ℝ

/-- Given that 20 plates and 40 cups cost $1.50 -/
def small_order_cost (c : CostStructure) : Prop :=
  20 * c.plate_cost + 40 * c.cup_cost = 1.50

/-- The cost of 100 plates and 200 cups -/
def large_order_cost (c : CostStructure) : ℝ :=
  100 * c.plate_cost + 200 * c.cup_cost

/-- Theorem stating that if 20 plates and 40 cups cost $1.50, 
    then 100 plates and 200 cups cost $7.50 -/
theorem plate_cup_cost_theorem (c : CostStructure) :
  small_order_cost c → large_order_cost c = 7.50 := by
  sorry

end NUMINAMATH_CALUDE_plate_cup_cost_theorem_l2327_232731


namespace NUMINAMATH_CALUDE_range_of_a_p_or_q_range_of_a_p_or_q_not_p_and_q_l2327_232718

def p (a : ℝ) : Prop := 
  (a > 3 ∨ (1 < a ∧ a < 2))

def q (a : ℝ) : Prop := 
  (2 < a ∧ a < 4)

theorem range_of_a_p_or_q (a : ℝ) : 
  p a ∨ q a → a ∈ Set.union (Set.Ioo 1 2) (Set.Ioi 2) := by
  sorry

theorem range_of_a_p_or_q_not_p_and_q (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → 
  a ∈ Set.union (Set.union (Set.Ioo 1 2) (Set.Ico 2 3)) (Set.Ici 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_p_or_q_range_of_a_p_or_q_not_p_and_q_l2327_232718


namespace NUMINAMATH_CALUDE_exists_zero_of_f_n_l2327_232716

/-- The function f(x) = x^2 + 2017x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2017*x + 1

/-- n-fold composition of f -/
def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f (f_n n x)

/-- For any positive integer n, there exists a real x such that f_n(x) = 0 -/
theorem exists_zero_of_f_n (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, f_n n x = 0 := by
  sorry


end NUMINAMATH_CALUDE_exists_zero_of_f_n_l2327_232716


namespace NUMINAMATH_CALUDE_baker_cakes_remaining_l2327_232701

theorem baker_cakes_remaining (initial_cakes bought_cakes sold_cakes : ℕ) 
  (h1 : initial_cakes = 173)
  (h2 : bought_cakes = 103)
  (h3 : sold_cakes = 86) :
  initial_cakes + bought_cakes - sold_cakes = 190 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_remaining_l2327_232701


namespace NUMINAMATH_CALUDE_train_length_problem_l2327_232745

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 46 km/hr and the slower train at 36 km/hr,
    if the faster train passes the slower train in 72 seconds,
    then the length of each train is 100 meters. -/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 72 →
  (faster_speed - slower_speed) * passing_time * (5 / 18) = 2 * train_length →
  train_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l2327_232745


namespace NUMINAMATH_CALUDE_sand_bags_problem_l2327_232707

/-- Given that each bag has a capacity of 65 pounds and 12 bags are needed,
    prove that the total pounds of sand is 780. -/
theorem sand_bags_problem (bag_capacity : ℕ) (num_bags : ℕ) 
    (h1 : bag_capacity = 65) (h2 : num_bags = 12) : 
    bag_capacity * num_bags = 780 := by
  sorry

end NUMINAMATH_CALUDE_sand_bags_problem_l2327_232707


namespace NUMINAMATH_CALUDE_problem_statement_l2327_232799

theorem problem_statement (x y : ℝ) (h : (x + 2*y)^3 + x^3 + 2*x + 2*y = 0) : 
  x + y - 1 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2327_232799


namespace NUMINAMATH_CALUDE_finite_triples_satisfying_equation_l2327_232788

theorem finite_triples_satisfying_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), Finite S ∧
  ∀ (a b c : ℕ), (a * b * c = 2009 * (a + b + c) ∧ a > 0 ∧ b > 0 ∧ c > 0) →
  (a, b, c) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_triples_satisfying_equation_l2327_232788


namespace NUMINAMATH_CALUDE_dandelion_seed_percentage_l2327_232706

/-- Represents the number of sunflowers Carla has -/
def num_sunflowers : ℕ := 6

/-- Represents the number of dandelions Carla has -/
def num_dandelions : ℕ := 8

/-- Represents the number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- Represents the number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- Calculates the total number of seeds from sunflowers -/
def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower

/-- Calculates the total number of seeds from dandelions -/
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion

/-- Calculates the total number of seeds -/
def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds

/-- Theorem: The percentage of seeds from dandelions is 64% -/
theorem dandelion_seed_percentage : 
  (total_dandelion_seeds : ℚ) / (total_seeds : ℚ) * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_percentage_l2327_232706


namespace NUMINAMATH_CALUDE_point_b_value_l2327_232748

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_value (a b : Point) (h1 : a.value = -2) (h2 : distance a b = 4) : b.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l2327_232748


namespace NUMINAMATH_CALUDE_florist_roses_count_l2327_232762

/-- Calculates the total number of roses after picking two batches -/
def total_roses (initial : Float) (batch1 : Float) (batch2 : Float) : Float :=
  initial + batch1 + batch2

/-- Theorem stating that given the specific numbers from the problem, 
    the total number of roses is 72.0 -/
theorem florist_roses_count : 
  total_roses 37.0 16.0 19.0 = 72.0 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_count_l2327_232762


namespace NUMINAMATH_CALUDE_line_direction_vector_l2327_232793

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

/-- The point on the line at t = 0 -/
def initial_point (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := 
    (4 + 3 * t / Real.sqrt 34, 2 + 5 * t / Real.sqrt 34)
  let y (x : ℝ) : ℝ := (5 * x - 7) / 3
  ∀ (x : ℝ), x ≥ 4 → 
    let point := (x, y x)
    let dist := Real.sqrt ((x - 4)^2 + (y x - 2)^2)
    point = initial_point line + dist • direction_vector line ∧
    direction_vector line = (3 / Real.sqrt 34, 5 / Real.sqrt 34) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2327_232793


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l2327_232764

/-- The number of ways to distribute candies into boxes. -/
def distribute_candy (candies boxes : ℕ) : ℕ := sorry

/-- The number of ways to distribute candies into boxes with no adjacent empty boxes. -/
def distribute_candy_no_adjacent_empty (candies boxes : ℕ) : ℕ := sorry

/-- Theorem: There are 34 ways to distribute 10 pieces of candy into 5 boxes
    such that no two adjacent boxes are empty. -/
theorem candy_distribution_theorem :
  distribute_candy_no_adjacent_empty 10 5 = 34 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l2327_232764


namespace NUMINAMATH_CALUDE_nth_prime_47_l2327_232774

def is_nth_prime (n : ℕ) (p : ℕ) : Prop :=
  p.Prime ∧ (Finset.filter Nat.Prime (Finset.range p)).card = n

theorem nth_prime_47 (n : ℕ) :
  is_nth_prime n 47 → n = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_nth_prime_47_l2327_232774


namespace NUMINAMATH_CALUDE_triangle_segment_calculation_l2327_232736

/-- Given a triangle ABC with point D on AB and point E on AD, prove that FC has a specific value. -/
theorem triangle_segment_calculation (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB AD ED : ℝ) (h3 : AB = (1/5) * AD) (h4 : ED = (2/3) * AD) : 
  ∃ (FC : ℝ), FC = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_calculation_l2327_232736


namespace NUMINAMATH_CALUDE_leftover_coin_value_l2327_232794

/-- Represents the number of coins in a complete roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of coins in dollars --/
def coinValue (quarters dimes : Nat) : Rat :=
  (quarters * 25 + dimes * 10) / 100

theorem leftover_coin_value
  (charles marta : CoinCount)
  (roll_size : RollSize)
  (h1 : charles.quarters = 57)
  (h2 : charles.dimes = 216)
  (h3 : marta.quarters = 88)
  (h4 : marta.dimes = 193)
  (h5 : roll_size.quarters = 50)
  (h6 : roll_size.dimes = 40) :
  let total_quarters := charles.quarters + marta.quarters
  let total_dimes := charles.dimes + marta.dimes
  let leftover_quarters := total_quarters % roll_size.quarters
  let leftover_dimes := total_dimes % roll_size.dimes
  coinValue leftover_quarters leftover_dimes = 1215 / 100 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l2327_232794


namespace NUMINAMATH_CALUDE_letter_count_cycle_exists_l2327_232775

/-- Represents the number of letters in the Russian word for a number -/
def russianWordLength (n : ℕ) : ℕ := sorry

/-- Generates the sequence of letter counts -/
def letterCountSequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => russianWordLength (letterCountSequence start n)

/-- Checks if a sequence has entered a cycle -/
def hasCycle (seq : ℕ → ℕ) (start : ℕ) (length : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i, seq (k + i) = seq (k + i + length)

theorem letter_count_cycle_exists (start : ℕ) :
  ∃ k length : ℕ, hasCycle (letterCountSequence start) k length :=
sorry

end NUMINAMATH_CALUDE_letter_count_cycle_exists_l2327_232775


namespace NUMINAMATH_CALUDE_cubic_inequality_l2327_232754

theorem cubic_inequality (x : ℝ) (h : x ≥ 1000000) :
  x^3 + x + 1 ≤ x^4 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2327_232754


namespace NUMINAMATH_CALUDE_robin_bobbin_chickens_l2327_232783

def chickens_eaten_sept_1 (chickens_sept_2 chickens_total_sept_15 : ℕ) : ℕ :=
  let avg_daily_consumption := chickens_total_sept_15 / 15
  let chickens_sept_1_and_2 := 2 * avg_daily_consumption
  chickens_sept_1_and_2 - chickens_sept_2

theorem robin_bobbin_chickens :
  chickens_eaten_sept_1 12 32 = 52 :=
sorry

end NUMINAMATH_CALUDE_robin_bobbin_chickens_l2327_232783


namespace NUMINAMATH_CALUDE_xy_squared_l2327_232785

theorem xy_squared (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + x + y = 7) : 
  x^2 * y^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_l2327_232785


namespace NUMINAMATH_CALUDE_ratio_equality_l2327_232797

theorem ratio_equality (a b : ℝ) (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  a^2 / 5 = b^3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2327_232797


namespace NUMINAMATH_CALUDE_f_min_at_three_l2327_232711

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3 -/
theorem f_min_at_three : ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l2327_232711


namespace NUMINAMATH_CALUDE_hyperbola_foci_on_x_axis_l2327_232798

/-- A curve C defined by mx^2 + (2-m)y^2 = 1 is a hyperbola with foci on the x-axis if and only if m ∈ (2, +∞) -/
theorem hyperbola_foci_on_x_axis (m : ℝ) :
  (∀ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, x^2 / (a^2 + c^2) + y^2 / a^2 = 1) →
  m > 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_on_x_axis_l2327_232798


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l2327_232751

-- First equation
theorem solve_equation_1 : 
  ∃ x : ℚ, 15 - (7 - 5 * x) = 2 * x + (5 - 3 * x) ↔ x = -1/2 := by sorry

-- Second equation
theorem solve_equation_2 : 
  ∃ x : ℚ, (x - 3) / 2 - (2 * x - 3) / 5 = 1 ↔ x = 19 := by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l2327_232751


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2327_232761

/-- Given a pentagon ABCDE where:
  - ΔABE, ΔBCE, and ΔCDE are right-angled triangles
  - ∠AEB = 45°
  - ∠BEC = 60°
  - ∠CED = 45°
  - AE = 40
Prove that the perimeter of pentagon ABCDE is 140 + (40√3)/3 -/
theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let angle (p q r : ℝ × ℝ) := Real.arccos ((p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)) / 
    (((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt * ((r.1 - q.1)^2 + (r.2 - q.2)^2).sqrt)
  let dist (p q : ℝ × ℝ) := ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  let perimeter := dist A B + dist B C + dist C D + dist D E + dist E A
  angle A E B = π/4 ∧ 
  angle B E C = π/3 ∧ 
  angle C E D = π/4 ∧
  angle B A E = π/2 ∧
  angle C B E = π/2 ∧
  angle D C E = π/2 ∧
  dist A E = 40 →
  perimeter = 140 + 40 * Real.sqrt 3 / 3 := by
sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l2327_232761


namespace NUMINAMATH_CALUDE_meaningful_expression_l2327_232720

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x / Real.sqrt (4 - x)) ↔ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2327_232720
