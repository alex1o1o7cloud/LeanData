import Mathlib

namespace NUMINAMATH_CALUDE_expected_original_positions_l44_4463

/-- The number of balls arranged in a circle -/
def numBalls : ℕ := 7

/-- The number of independent random transpositions -/
def numTranspositions : ℕ := 3

/-- The probability of a ball being in its original position after the transpositions -/
def probOriginalPosition : ℚ := 127 / 343

/-- The expected number of balls in their original positions after the transpositions -/
def expectedOriginalPositions : ℚ := numBalls * probOriginalPosition

theorem expected_original_positions :
  expectedOriginalPositions = 889 / 343 := by sorry

end NUMINAMATH_CALUDE_expected_original_positions_l44_4463


namespace NUMINAMATH_CALUDE_lost_money_proof_l44_4497

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  (initial_amount - spent_amount) - remaining_amount

theorem lost_money_proof (initial_amount spent_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  money_lost initial_amount spent_amount remaining_amount = 6 := by
  sorry

end NUMINAMATH_CALUDE_lost_money_proof_l44_4497


namespace NUMINAMATH_CALUDE_exists_counterexample_for_statement_C_l44_4474

-- Define the heart operation
def heart (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem exists_counterexample_for_statement_C :
  ∃ (x y z : ℝ), (heart (heart x y) (heart y z)) ≠ |x - z| :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_for_statement_C_l44_4474


namespace NUMINAMATH_CALUDE_peters_pumpkin_profit_l44_4460

/-- Represents the total amount of money collected from selling pumpkins -/
def total_money (jumbo_price regular_price : ℝ) (total_pumpkins regular_pumpkins : ℕ) : ℝ :=
  regular_price * regular_pumpkins + jumbo_price * (total_pumpkins - regular_pumpkins)

/-- Theorem stating that Peter's total money collected is $395.00 -/
theorem peters_pumpkin_profit :
  total_money 9 4 80 65 = 395 := by
  sorry

end NUMINAMATH_CALUDE_peters_pumpkin_profit_l44_4460


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l44_4465

theorem recurring_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x = 3 + 7 / 9 * (1 / 10^n)) → x = 34 / 9 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l44_4465


namespace NUMINAMATH_CALUDE_lucille_weeding_ratio_l44_4452

def weed_value : ℕ := 6
def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def total_grass_weeds : ℕ := 32
def soda_cost : ℕ := 99
def remaining_money : ℕ := 147

theorem lucille_weeding_ratio :
  let total_earned := remaining_money + soda_cost
  let flower_veg_earnings := (flower_bed_weeds + vegetable_patch_weeds) * weed_value
  let grass_earnings := total_earned - flower_veg_earnings
  let grass_weeds_pulled := grass_earnings / weed_value
  (grass_weeds_pulled : ℚ) / total_grass_weeds = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_lucille_weeding_ratio_l44_4452


namespace NUMINAMATH_CALUDE_candy_game_solution_l44_4446

theorem candy_game_solution (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3) :
  ∃ correct_answers : ℕ, 
    correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty ∧ 
    correct_answers = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_game_solution_l44_4446


namespace NUMINAMATH_CALUDE_pure_imaginary_product_second_quadrant_product_and_magnitude_l44_4469

-- Define the complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := a + 2 * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

-- Part I
theorem pure_imaginary_product (a : ℝ) : 
  (z₁ a * z₂).re = 0 → a = -8/3 :=
sorry

-- Part II
theorem second_quadrant_product_and_magnitude (a : ℝ) :
  (z₁ a * z₂).re < 0 ∧ (z₁ a * z₂).im > 0 ∧ Complex.abs (z₁ a) ≤ 4 →
  -2 * Real.sqrt 3 ≤ a ∧ a < -8/3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_second_quadrant_product_and_magnitude_l44_4469


namespace NUMINAMATH_CALUDE_no_point_satisfies_conditions_l44_4479

-- Define a triangle as a structure with three points
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is inside a triangle
def isInside (T : Triangle) (D : ℝ × ℝ) : Prop :=
  sorry

-- Define a function to get the shortest side of a triangle
def shortestSide (T : Triangle) : ℝ :=
  sorry

-- Main theorem
theorem no_point_satisfies_conditions (ABC : Triangle) :
  ¬ ∃ D : ℝ × ℝ,
    isInside ABC D ∧
    shortestSide (Triangle.mk ABC.B ABC.C D) = 1 ∧
    shortestSide (Triangle.mk ABC.A ABC.C D) = 2 ∧
    shortestSide (Triangle.mk ABC.A ABC.B D) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_no_point_satisfies_conditions_l44_4479


namespace NUMINAMATH_CALUDE_remaining_child_meal_capacity_l44_4433

/-- Represents the meal capacity and consumption for a trekking group -/
structure TrekkingMeal where
  total_adults : ℕ
  adults_fed : ℕ
  adult_meal_capacity : ℕ
  child_meal_capacity : ℕ
  remaining_child_capacity : ℕ

/-- Theorem stating that given the conditions of the trekking meal,
    the number of children that can be catered with the remaining food is 36 -/
theorem remaining_child_meal_capacity
  (meal : TrekkingMeal)
  (h1 : meal.total_adults = 55)
  (h2 : meal.adult_meal_capacity = 70)
  (h3 : meal.child_meal_capacity = 90)
  (h4 : meal.adults_fed = 42)
  (h5 : meal.remaining_child_capacity = 36) :
  meal.remaining_child_capacity = 36 := by
  sorry


end NUMINAMATH_CALUDE_remaining_child_meal_capacity_l44_4433


namespace NUMINAMATH_CALUDE_equal_part_implies_a_eq_neg_two_l44_4499

/-- A complex number is an "equal part complex number" if its real and imaginary parts are equal -/
def is_equal_part (z : ℂ) : Prop := z.re = z.im

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.I * (2 + a * Complex.I)

/-- Theorem: If z(a) is an equal part complex number, then a = -2 -/
theorem equal_part_implies_a_eq_neg_two (a : ℝ) :
  is_equal_part (z a) → a = -2 := by sorry

end NUMINAMATH_CALUDE_equal_part_implies_a_eq_neg_two_l44_4499


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l44_4445

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (7 + Real.sqrt (49 - 48)) / 2
  let r₂ := (7 - Real.sqrt (49 - 48)) / 2
  r₁ + r₂ = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l44_4445


namespace NUMINAMATH_CALUDE_P_equals_F_l44_4423

-- Define the sets P and F
def P : Set ℝ := {y | ∃ x, y = x^2 + 1}
def F : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem P_equals_F : P = F := by sorry

end NUMINAMATH_CALUDE_P_equals_F_l44_4423


namespace NUMINAMATH_CALUDE_seashells_given_l44_4407

theorem seashells_given (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 62) 
  (h2 : remaining_seashells = 13) : 
  initial_seashells - remaining_seashells = 49 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_l44_4407


namespace NUMINAMATH_CALUDE_arctan_sum_two_five_l44_4425

theorem arctan_sum_two_five : Real.arctan (2/5) + Real.arctan (5/2) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_two_five_l44_4425


namespace NUMINAMATH_CALUDE_smallest_cube_square_l44_4428

theorem smallest_cube_square (x : ℕ) (M : ℤ) : x = 11025 ↔ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(∃ N : ℤ, 2520 * y = N^3 ∧ ∃ z : ℕ, y = z^2)) ∧ 
  (∃ N : ℤ, 2520 * x = N^3) ∧ 
  (∃ z : ℕ, x = z^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_square_l44_4428


namespace NUMINAMATH_CALUDE_lecture_scheduling_l44_4461

theorem lecture_scheduling (n : ℕ) (h : n = 7) :
  (n.factorial / 2) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lecture_scheduling_l44_4461


namespace NUMINAMATH_CALUDE_years_until_26_l44_4434

/-- Kiril's current age -/
def current_age : ℕ := sorry

/-- Kiril's target age -/
def target_age : ℕ := 26

/-- Condition that current age is a multiple of 5 -/
axiom current_age_multiple_of_5 : ∃ k : ℕ, current_age = 5 * k

/-- Condition that last year's age was a multiple of 7 -/
axiom last_year_age_multiple_of_7 : ∃ m : ℕ, current_age - 1 = 7 * m

/-- Theorem stating the number of years until Kiril is 26 -/
theorem years_until_26 : target_age - current_age = 11 := by sorry

end NUMINAMATH_CALUDE_years_until_26_l44_4434


namespace NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_l44_4440

/-- A quadrilateral is defined as a polygon with four sides -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- Parallel sides in a quadrilateral -/
def parallel_sides (q : Quadrilateral) (side1 side2 : Fin 4) : Prop :=
  -- Definition of parallel sides omitted for brevity
  sorry

/-- A parallelogram is a quadrilateral with both pairs of opposite sides parallel -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  parallel_sides q 0 2 ∧ parallel_sides q 1 3

/-- Theorem: If both pairs of opposite sides of a quadrilateral are parallel, then it is a parallelogram -/
theorem parallel_sides_implies_parallelogram (q : Quadrilateral) :
  (parallel_sides q 0 2 ∧ parallel_sides q 1 3) → is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_l44_4440


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l44_4438

theorem gcd_lcm_sum : Nat.gcd 28 63 + Nat.lcm 18 24 = 79 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l44_4438


namespace NUMINAMATH_CALUDE_painting_frame_ratio_l44_4488

theorem painting_frame_ratio {x l : ℝ} (h_positive : x > 0 ∧ l > 0) 
  (h_area_equality : (x + 2*l) * ((3/2)*x + 2*l) = 2 * (x * (3/2)*x)) :
  (x + 2*l) / ((3/2)*x + 2*l) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_painting_frame_ratio_l44_4488


namespace NUMINAMATH_CALUDE_quadratic_value_theorem_l44_4487

theorem quadratic_value_theorem (x : ℝ) (h : x^2 + 4*x - 2 = 0) :
  3*x^2 + 12*x - 23 = -17 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_theorem_l44_4487


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l44_4471

theorem muffin_banana_cost_ratio :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  4 * m + 3 * b > 0 →
  2 * (4 * m + 3 * b) = 2 * m + 16 * b →
  m / b = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l44_4471


namespace NUMINAMATH_CALUDE_negation_zero_collinear_with_any_l44_4477

open Set

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def IsCollinear (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

theorem negation_zero_collinear_with_any :
  (¬ ∀ (v : V), IsCollinear (0 : V) v) ↔ ∃ (v : V), ¬ IsCollinear (0 : V) v :=
sorry

end NUMINAMATH_CALUDE_negation_zero_collinear_with_any_l44_4477


namespace NUMINAMATH_CALUDE_square_transformation_l44_4441

-- Define the square in the xy-plane
def square_vertices : List (ℝ × ℝ) := [(0, 0), (1, 0), (1, 1), (0, 1)]

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x^2 - y^2, 2*x*y)

-- Define the transformed square
def transformed_square : List (ℝ × ℝ) := square_vertices.map transform

-- Define the expected shape in the uv-plane
def expected_shape (u v : ℝ) : Prop :=
  (u = 0 ∧ 0 ≤ v ∧ v ≤ 1) ∨  -- Line segment from (0,0) to (1,0)
  (u = 1 - v^2/4 ∧ 0 ≤ v ∧ v ≤ 2) ∨  -- Parabola segment
  (u = v^2/4 - 1 ∧ 0 ≤ v ∧ v ≤ 2) ∨  -- Parabola segment
  (v = 0 ∧ -1 ≤ u ∧ u ≤ 0)  -- Line segment from (-1,0) to (0,0)

theorem square_transformation :
  ∀ (u v : ℝ), (∃ (x y : ℝ), (x, y) ∈ square_vertices ∧ transform (x, y) = (u, v)) ↔ expected_shape u v := by
  sorry

end NUMINAMATH_CALUDE_square_transformation_l44_4441


namespace NUMINAMATH_CALUDE_inclined_prism_volume_l44_4422

/-- The volume of an inclined prism with a parallelogram base and inclined lateral edge. -/
theorem inclined_prism_volume 
  (base_side1 base_side2 lateral_edge : ℝ) 
  (base_angle lateral_angle : ℝ) : 
  base_side1 = 3 →
  base_side2 = 6 →
  lateral_edge = 4 →
  base_angle = Real.pi / 4 →
  lateral_angle = Real.pi / 6 →
  (base_side1 * base_side2 * Real.sin base_angle) * (lateral_edge * Real.sin lateral_angle) = 18 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inclined_prism_volume_l44_4422


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l44_4439

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

theorem cosine_of_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 2 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l44_4439


namespace NUMINAMATH_CALUDE_square_sum_triples_l44_4492

theorem square_sum_triples :
  ∀ a b c : ℝ,
  (a = (b + c)^2 ∧ b = (a + c)^2 ∧ c = (a + b)^2) →
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/4 ∧ b = 1/4 ∧ c = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_triples_l44_4492


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l44_4406

theorem quadratic_minimum_value :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + 5
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l44_4406


namespace NUMINAMATH_CALUDE_cube_sum_l44_4410

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The sum of faces, edges, and vertices in a cube is 26 -/
theorem cube_sum : cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l44_4410


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l44_4450

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  divided_squares : ℕ
  divided_rectangles : ℕ
  shaded_column : ℕ

/-- The fraction of the quilt block that is shaded -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  q.shaded_column / q.total_squares

/-- Theorem stating that the shaded fraction is 1/3 for the given quilt block configuration -/
theorem quilt_shaded_fraction :
  ∀ q : QuiltBlock,
    q.total_squares = 9 ∧
    q.divided_squares = 3 ∧
    q.divided_rectangles = 3 ∧
    q.shaded_column = 1 →
    shaded_fraction q = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_quilt_shaded_fraction_l44_4450


namespace NUMINAMATH_CALUDE_exists_three_numbers_sum_geq_54_l44_4414

theorem exists_three_numbers_sum_geq_54 
  (S : Finset ℕ) 
  (distinct : S.card = 10) 
  (sum_gt_144 : S.sum id > 144) : 
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c ≥ 54 :=
by sorry

end NUMINAMATH_CALUDE_exists_three_numbers_sum_geq_54_l44_4414


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l44_4458

/-- Given a point P with coordinates (-2, 3), prove that the coordinates of the point symmetric to the origin with respect to P are (2, -3). -/
theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let symmetric_point := (-P.1, -P.2)
  symmetric_point = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l44_4458


namespace NUMINAMATH_CALUDE_equation_solutions_l44_4468

-- Define the equation
def equation (x a : ℝ) : Prop :=
  ((1 - x^2)^2 + 2*a^2 + 5*a)^7 - ((3*a + 2)*(1 - x^2) + 3)^7 = 5 - 2*a - (3*a + 2)*x^2 - 2*a^2 - (1 - x^2)^2

-- Define the interval
def in_interval (x : ℝ) : Prop :=
  -Real.sqrt 6 / 2 ≤ x ∧ x ≤ Real.sqrt 2

-- Define the solution sets
def solution_set_1 (a : ℝ) : Set ℝ :=
  {x | x = Real.sqrt (2 - 2*a) ∨ x = -Real.sqrt (2 - 2*a)}

def solution_set_2 (a : ℝ) : Set ℝ :=
  {x | x = Real.sqrt (-a - 2) ∨ x = -Real.sqrt (-a - 2)}

-- The main theorem
theorem equation_solutions :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ in_interval x₁ ∧ in_interval x₂ ∧ equation x₁ a ∧ equation x₂ a) ↔
  ((0.25 ≤ a ∧ a < 1 ∧ ∀ x ∈ solution_set_1 a, in_interval x ∧ equation x a) ∨
   (-3.5 ≤ a ∧ a < -2 ∧ ∀ x ∈ solution_set_2 a, in_interval x ∧ equation x a)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l44_4468


namespace NUMINAMATH_CALUDE_reciprocal_problem_l44_4483

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 150 * (1 / x) = 240 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l44_4483


namespace NUMINAMATH_CALUDE_solution_characterization_l44_4411

theorem solution_characterization (x y : ℤ) :
  x^2 - y^4 = 2009 ↔ (x = 45 ∧ y = 2) ∨ (x = 45 ∧ y = -2) ∨ (x = -45 ∧ y = 2) ∨ (x = -45 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l44_4411


namespace NUMINAMATH_CALUDE_board_numbers_divisibility_l44_4485

theorem board_numbers_divisibility (X Y N A B : ℤ) 
  (sum_eq : X + Y = N) 
  (tanya_div : (A * X + B * Y) % N = 0) : 
  (B * X + A * Y) % N = 0 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_divisibility_l44_4485


namespace NUMINAMATH_CALUDE_square_independence_of_p_l44_4431

theorem square_independence_of_p (m n p k : ℕ) : 
  m > 0 → n > 0 → p.Prime → p > m → 
  m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2 → 
  ∃ f : ℕ → ℕ, ∀ q : ℕ, q.Prime → q > m → 
    m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = (f q)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_independence_of_p_l44_4431


namespace NUMINAMATH_CALUDE_alpha_beta_composition_l44_4449

theorem alpha_beta_composition (α β : ℝ → ℝ) (h_α : ∀ x, α x = 4 * x + 9) (h_β : ∀ x, β x = 7 * x + 6) :
  (∃ x, (α ∘ β) x = 4) ↔ (∃ x, x = -29/28) :=
by sorry

end NUMINAMATH_CALUDE_alpha_beta_composition_l44_4449


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l44_4437

/-- A circle with center (0, 0) and radius r > 0 is tangent to the line 3x - 4y + 20 = 0 if and only if r = 4 -/
theorem circle_tangent_to_line (r : ℝ) (hr : r > 0) :
  (∀ x y : ℝ, x^2 + y^2 = r^2 ↔ (3*x - 4*y + 20 = 0 → x^2 + y^2 ≥ r^2) ∧ 
  (∃ x y : ℝ, 3*x - 4*y + 20 = 0 ∧ x^2 + y^2 = r^2)) ↔ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l44_4437


namespace NUMINAMATH_CALUDE_alex_makes_100_dresses_l44_4408

/-- Given the initial amount of silk, silk given to friends, and silk required per dress,
    calculate the number of dresses Alex can make. -/
def dresses_alex_can_make (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (silk_per_dress : ℕ) : ℕ :=
  (initial_silk - friends * silk_per_friend) / silk_per_dress

/-- Prove that Alex can make 100 dresses given the conditions. -/
theorem alex_makes_100_dresses :
  dresses_alex_can_make 600 5 20 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_alex_makes_100_dresses_l44_4408


namespace NUMINAMATH_CALUDE_percentage_ratio_proof_l44_4413

theorem percentage_ratio_proof (P Q M N R : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.3 * P)
  (hN : N = 0.6 * P)
  (hR : R = 0.2 * P)
  (hP : P ≠ 0) : (M + R) / N = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_ratio_proof_l44_4413


namespace NUMINAMATH_CALUDE_smallest_positive_a_quartic_polynomial_l44_4495

theorem smallest_positive_a_quartic_polynomial (a b : ℝ) : 
  (∀ x : ℝ, x^4 - a*x^3 + b*x^2 - a*x + a = 0 → x > 0) →
  (∀ c : ℝ, c > 0 → (∃ d : ℝ, ∀ x : ℝ, x^4 - c*x^3 + d*x^2 - c*x + c = 0 → x > 0) → c ≥ a) →
  b = 6 * (4^(1/3))^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_quartic_polynomial_l44_4495


namespace NUMINAMATH_CALUDE_inspector_meter_count_l44_4470

/-- Given an inspector who rejects 10% of meters as defective and finds 20 meters to be defective,
    the total number of meters examined is 200. -/
theorem inspector_meter_count (reject_rate : ℝ) (defective_count : ℕ) (total_count : ℕ) : 
  reject_rate = 0.1 →
  defective_count = 20 →
  (reject_rate : ℝ) * total_count = defective_count →
  total_count = 200 := by
  sorry

end NUMINAMATH_CALUDE_inspector_meter_count_l44_4470


namespace NUMINAMATH_CALUDE_sum_of_combinations_specific_combination_l44_4444

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Statement for the first part
theorem sum_of_combinations : C 5 0 + C 6 5 + C 7 5 + C 8 5 + C 9 5 + C 10 5 = 462 := by sorry

-- Statement for the second part
theorem specific_combination (m : ℕ) :
  (1 / C 5 m : ℚ) - (1 / C 6 m : ℚ) = (7 : ℚ) / (10 * C 7 m) → C 8 m = 28 := by sorry

end NUMINAMATH_CALUDE_sum_of_combinations_specific_combination_l44_4444


namespace NUMINAMATH_CALUDE_exp_addition_property_l44_4448

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by
  sorry

end NUMINAMATH_CALUDE_exp_addition_property_l44_4448


namespace NUMINAMATH_CALUDE_equation_c_is_quadratic_l44_4401

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (4x-3)(3x+1)=0 -/
def f (x : ℝ) : ℝ := (4*x - 3) * (3*x + 1)

/-- Theorem: The equation (4x-3)(3x+1)=0 is a quadratic equation -/
theorem equation_c_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_c_is_quadratic_l44_4401


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_1_hundreds_l44_4403

def digits : List Nat := [1, 5, 6, 9]

def isValidNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 = 1) ∧
  (∀ d, d ∈ digits → (n / 10 % 10 = d ∨ n % 10 = d))

theorem largest_three_digit_number_with_1_hundreds :
  ∀ n : Nat, isValidNumber n → n ≤ 196 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_1_hundreds_l44_4403


namespace NUMINAMATH_CALUDE_correct_calculation_l44_4432

theorem correct_calculation (x y : ℝ) : -2 * x^2 * y - 3 * y * x^2 = -5 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l44_4432


namespace NUMINAMATH_CALUDE_fred_basketball_games_l44_4443

/-- The number of basketball games Fred went to last year -/
def last_year_games : ℕ := 36

/-- The difference in games between last year and this year -/
def game_difference : ℕ := 11

/-- The number of basketball games Fred went to this year -/
def this_year_games : ℕ := last_year_games - game_difference

theorem fred_basketball_games : this_year_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_basketball_games_l44_4443


namespace NUMINAMATH_CALUDE_train_speed_without_stoppages_l44_4429

/-- The average speed of a train without stoppages, given certain conditions. -/
theorem train_speed_without_stoppages (distance : ℝ) (time_with_stops : ℝ) (time_without_stops : ℝ)
  (h1 : time_without_stops = time_with_stops / 2)
  (h2 : distance / time_with_stops = 125) :
  distance / time_without_stops = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_without_stoppages_l44_4429


namespace NUMINAMATH_CALUDE_map_scale_conversion_l44_4400

/-- Given a scale where 1 inch represents 500 feet, and a path measuring 6.5 inches on a map,
    the actual length of the path in feet is 3250. -/
theorem map_scale_conversion (scale : ℝ) (map_length : ℝ) (actual_length : ℝ) : 
  scale = 500 → map_length = 6.5 → actual_length = scale * map_length → actual_length = 3250 :=
by sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l44_4400


namespace NUMINAMATH_CALUDE_probability_heart_then_club_is_13_204_l44_4427

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Suits in a deck -/
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

/-- A card in the deck -/
structure Card :=
  (number : Fin 13)
  (suit : Suit)

/-- The probability of drawing a heart first and a club second from a standard deck -/
def probability_heart_then_club (d : Deck) : ℚ :=
  13 / 204

/-- Theorem: The probability of drawing a heart first and a club second from a standard deck is 13/204 -/
theorem probability_heart_then_club_is_13_204 (d : Deck) :
  probability_heart_then_club d = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_club_is_13_204_l44_4427


namespace NUMINAMATH_CALUDE_exists_nonperiodic_with_repeating_subsequence_l44_4404

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: For any index k, there exists a t such that the sequence repeats at multiples of t -/
def HasRepeatingSubsequence (a : Sequence) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, ∀ n : ℕ, a k = a (k + n * t)

/-- Property: A sequence is periodic -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T : ℕ, ∀ k : ℕ, a k = a (k + T)

/-- Theorem: There exists a sequence that has repeating subsequences but is not periodic -/
theorem exists_nonperiodic_with_repeating_subsequence :
  ∃ a : Sequence, HasRepeatingSubsequence a ∧ ¬IsPeriodic a :=
sorry

end NUMINAMATH_CALUDE_exists_nonperiodic_with_repeating_subsequence_l44_4404


namespace NUMINAMATH_CALUDE_find_k_l44_4476

theorem find_k (x y k : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -3) 
  (h3 : 2 * x^2 + k * x * y = 4) : 
  k = 2/3 := by
sorry

end NUMINAMATH_CALUDE_find_k_l44_4476


namespace NUMINAMATH_CALUDE_jeff_performance_time_per_point_l44_4416

/-- Represents a tennis player's performance -/
structure TennisPerformance where
  playTime : ℕ  -- play time in hours
  pointsPerMatch : ℕ  -- points needed to win a match
  gamesWon : ℕ  -- number of games won

/-- Calculates the time it takes to score a point in minutes -/
def timePerPoint (perf : TennisPerformance) : ℚ :=
  (perf.playTime * 60) / (perf.pointsPerMatch * perf.gamesWon)

/-- Theorem stating that for the given performance, it takes 5 minutes to score a point -/
theorem jeff_performance_time_per_point :
  let jeff : TennisPerformance := ⟨2, 8, 3⟩
  timePerPoint jeff = 5 := by sorry

end NUMINAMATH_CALUDE_jeff_performance_time_per_point_l44_4416


namespace NUMINAMATH_CALUDE_max_acute_angles_eq_three_l44_4418

/-- A convex polygon with n sides, where n ≥ 3 --/
structure ConvexPolygon where
  n : ℕ
  n_ge_three : n ≥ 3

/-- The maximum number of acute angles in a convex polygon --/
def max_acute_angles (p : ConvexPolygon) : ℕ := 3

/-- Theorem: The maximum number of acute angles in a convex polygon is 3 --/
theorem max_acute_angles_eq_three (p : ConvexPolygon) :
  max_acute_angles p = 3 := by sorry

end NUMINAMATH_CALUDE_max_acute_angles_eq_three_l44_4418


namespace NUMINAMATH_CALUDE_function_value_proof_l44_4415

theorem function_value_proof : 
  ∀ f : ℝ → ℝ, 
  (∀ x, f x = (x - 3) * (x + 4)) → 
  f 29 = 170 → 
  ∃ x, f x = 170 ∧ x = 13 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l44_4415


namespace NUMINAMATH_CALUDE_angle_relations_l44_4442

theorem angle_relations (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2) 
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 4 / 3)
  (h_sin_diff : Real.sin (α - β) = -(Real.sqrt 5) / 5) :
  Real.cos (2 * α) = -7 / 25 ∧ 
  Real.tan (α + β) = -41 / 38 := by
sorry

end NUMINAMATH_CALUDE_angle_relations_l44_4442


namespace NUMINAMATH_CALUDE_exists_removable_column_l44_4498

-- Define a type for the table
def Table (n : ℕ) := Fin n → Fin n → ℕ

-- Define a property that all rows are distinct
def all_rows_distinct (n : ℕ) (t : Table n) : Prop :=
  ∀ i j, i ≠ j → ∃ k, t i k ≠ t j k

-- Define a property that rows remain distinct after removing a column
def rows_distinct_after_removal (n : ℕ) (t : Table n) (c : Fin n) : Prop :=
  ∀ i j, i ≠ j → ∃ k, k ≠ c → t i k ≠ t j k

-- Theorem statement
theorem exists_removable_column (n : ℕ) (t : Table n) 
  (h : all_rows_distinct n t) : 
  ∃ c : Fin n, rows_distinct_after_removal n t c :=
sorry

end NUMINAMATH_CALUDE_exists_removable_column_l44_4498


namespace NUMINAMATH_CALUDE_range_of_s_l44_4496

-- Define the set of composite positive integers
def CompositePositiveIntegers : Set ℕ := {n : ℕ | n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function s
def s (n : ℕ) : ℕ := sorry

-- State the theorem
theorem range_of_s :
  (∀ n ∈ CompositePositiveIntegers, s n > 7) ∧
  (∀ k > 7, ∃ n ∈ CompositePositiveIntegers, s n = k) :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l44_4496


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l44_4489

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (A : Point) (B : Point) (C : Point)

/-- The perpendicular distance from a point to a line -/
def perpendicularDistance (P : Point) (A : Point) (B : Point) : ℝ := sorry

theorem equilateral_triangle_side_length 
  (ABC : EquilateralTriangle) (P : Point) :
  perpendicularDistance P ABC.A ABC.B = 2 →
  perpendicularDistance P ABC.B ABC.C = 4 →
  perpendicularDistance P ABC.C ABC.A = 6 →
  ∃ (side : ℝ), side = 8 * Real.sqrt 3 ∧ 
    (perpendicularDistance ABC.A ABC.B ABC.C = side ∧
     perpendicularDistance ABC.B ABC.C ABC.A = side ∧
     perpendicularDistance ABC.C ABC.A ABC.B = side) :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l44_4489


namespace NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l44_4472

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 42*x + 336 = -48

-- Define the roots of the equation
noncomputable def root1 : ℝ := -24
noncomputable def root2 : ℝ := -16

-- Theorem statement
theorem nonnegative_difference_of_roots : 
  (∀ x, quadratic_equation x ↔ x = root1 ∨ x = root2) →
  |root1 - root2| = 8 := by sorry

end NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l44_4472


namespace NUMINAMATH_CALUDE_percentage_comparison_l44_4454

theorem percentage_comparison : 
  (0.85 * 250 - 0.75 * 180) < 0.90 * 320 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l44_4454


namespace NUMINAMATH_CALUDE_diagonal_cubes_180_270_360_l44_4481

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_on_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- Theorem: The internal diagonal of a 180 × 270 × 360 rectangular solid passes through 540 cubes -/
theorem diagonal_cubes_180_270_360 :
  cubes_on_diagonal 180 270 360 = 540 := by sorry

end NUMINAMATH_CALUDE_diagonal_cubes_180_270_360_l44_4481


namespace NUMINAMATH_CALUDE_edge_coloring_theorem_l44_4462

/-- A complete graph on n vertices -/
def CompleteGraph (n : ℕ) := Unit

/-- A coloring of edges with k colors -/
def Coloring (G : CompleteGraph 10) (k : ℕ) := Unit

/-- Predicate: Any subset of m vertices contains edges of all k colors -/
def AllColorsInSubset (G : CompleteGraph 10) (c : Coloring G k) (m k : ℕ) : Prop := sorry

theorem edge_coloring_theorem (G : CompleteGraph 10) :
  (∃ c : Coloring G 5, AllColorsInSubset G c 5 5) ∧
  (¬ ∃ c : Coloring G 4, AllColorsInSubset G c 4 4) := by sorry

end NUMINAMATH_CALUDE_edge_coloring_theorem_l44_4462


namespace NUMINAMATH_CALUDE_desired_outcome_probability_l44_4412

/-- Represents a die with a fixed number of sides --/
structure Die :=
  (sides : Nat)
  (values : Fin sides → Nat)

/-- Carla's die always shows 7 --/
def carla_die : Die :=
  { sides := 6,
    values := λ _ => 7 }

/-- Derek's die has numbers from 2 to 7 --/
def derek_die : Die :=
  { sides := 6,
    values := λ i => i.val + 2 }

/-- Emily's die has four faces showing 3 and two faces showing 8 --/
def emily_die : Die :=
  { sides := 6,
    values := λ i => if i.val < 4 then 3 else 8 }

/-- The probability of the desired outcome --/
def probability : Rat :=
  8 / 27

/-- Theorem stating the probability of the desired outcome --/
theorem desired_outcome_probability :
  (∀ (c : Fin carla_die.sides) (d : Fin derek_die.sides) (e : Fin emily_die.sides),
    (carla_die.values c > derek_die.values d ∧
     carla_die.values c > emily_die.values e ∧
     derek_die.values d + emily_die.values e < 10) →
    probability = 8 / 27) :=
by
  sorry

end NUMINAMATH_CALUDE_desired_outcome_probability_l44_4412


namespace NUMINAMATH_CALUDE_quadratic_positive_combination_l44_4453

/-- A quadratic polynomial -/
def QuadraticPolynomial := ℝ → ℝ

/-- Predicate to check if a function is negative on an interval -/
def NegativeOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f x < 0

/-- Predicate to check if two intervals are non-overlapping -/
def NonOverlappingIntervals (a b c d : ℝ) : Prop :=
  b < c ∨ d < a

/-- Main theorem statement -/
theorem quadratic_positive_combination
  (f g : QuadraticPolynomial)
  (a b c d : ℝ)
  (hf : NegativeOnInterval f a b)
  (hg : NegativeOnInterval g c d)
  (h_non_overlap : NonOverlappingIntervals a b c d) :
  ∃ (α β : ℝ), α > 0 ∧ β > 0 ∧ ∀ x, α * f x + β * g x > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_positive_combination_l44_4453


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l44_4451

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Condition for f to have both maximum and minimum -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0

theorem cubic_function_extrema (a : ℝ) :
  has_max_and_min a → a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l44_4451


namespace NUMINAMATH_CALUDE_double_inequality_abc_l44_4455

theorem double_inequality_abc (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a*b - b*c - c*a ∧ 
  a + b + c - a*b - b*c - c*a ≤ (1 + a^2 + b^2 + c^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_abc_l44_4455


namespace NUMINAMATH_CALUDE_log_equation_solution_l44_4421

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (5 * x) / Real.log 3 → x = (5 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l44_4421


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l44_4456

theorem quadratic_equation_solutions :
  (∀ x, x^2 - 9 = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x, x^2 + 2*x - 1 = 0 ↔ x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l44_4456


namespace NUMINAMATH_CALUDE_caden_coin_value_l44_4473

/-- Represents the number of coins of each type Caden has -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (counts : CoinCounts) : ℚ :=
  (counts.pennies : ℚ) * (1 : ℚ) / 100 +
  (counts.nickels : ℚ) * (5 : ℚ) / 100 +
  (counts.dimes : ℚ) * (10 : ℚ) / 100 +
  (counts.quarters : ℚ) * (25 : ℚ) / 100

/-- Theorem stating that Caden has $8.00 in coins -/
theorem caden_coin_value :
  ∃ (counts : CoinCounts),
    counts.pennies = 120 ∧
    counts.nickels = counts.pennies / 3 ∧
    counts.dimes = counts.nickels / 5 ∧
    counts.quarters = counts.dimes * 2 ∧
    totalValue counts = 8 := by
  sorry


end NUMINAMATH_CALUDE_caden_coin_value_l44_4473


namespace NUMINAMATH_CALUDE_bakery_muffins_l44_4482

/-- The number of muffins in each box -/
def muffins_per_box : ℕ := 5

/-- The number of available boxes -/
def available_boxes : ℕ := 10

/-- The number of additional boxes needed -/
def additional_boxes_needed : ℕ := 9

/-- The total number of muffins made by the bakery -/
def total_muffins : ℕ := muffins_per_box * (available_boxes + additional_boxes_needed)

theorem bakery_muffins : total_muffins = 95 := by
  sorry

end NUMINAMATH_CALUDE_bakery_muffins_l44_4482


namespace NUMINAMATH_CALUDE_constant_term_when_sum_is_64_l44_4435

-- Define the sum of binomial coefficients
def sum_binomial_coeffs (n : ℕ) : ℕ := 2^n

-- Define the constant term in the expansion
def constant_term (n : ℕ) : ℤ :=
  (-1)^(n/2) * (n.choose (n/2))

-- Theorem statement
theorem constant_term_when_sum_is_64 :
  ∃ n : ℕ, sum_binomial_coeffs n = 64 ∧ constant_term n = 15 :=
sorry

end NUMINAMATH_CALUDE_constant_term_when_sum_is_64_l44_4435


namespace NUMINAMATH_CALUDE_ellipse_regions_l44_4494

/-- 
Given n ellipses in a plane where:
- Any two ellipses intersect at exactly two points
- No three ellipses intersect at the same point

The number of regions these ellipses divide the plane into is n(n-1) + 2.
-/
theorem ellipse_regions (n : ℕ) : ℕ := by
  sorry

#check ellipse_regions

end NUMINAMATH_CALUDE_ellipse_regions_l44_4494


namespace NUMINAMATH_CALUDE_cs_consecutive_probability_l44_4426

/-- The number of people sitting at the table -/
def total_people : ℕ := 12

/-- The number of computer scientists -/
def num_cs : ℕ := 5

/-- The number of chemistry majors -/
def num_chem : ℕ := 4

/-- The number of history majors -/
def num_hist : ℕ := 3

/-- The probability of all computer scientists sitting consecutively -/
def prob_cs_consecutive : ℚ := 1 / 66

theorem cs_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let consecutive_arrangements := Nat.factorial (num_cs) * Nat.factorial (total_people - num_cs)
  (consecutive_arrangements : ℚ) / total_arrangements = prob_cs_consecutive :=
sorry

end NUMINAMATH_CALUDE_cs_consecutive_probability_l44_4426


namespace NUMINAMATH_CALUDE_words_per_page_larger_type_l44_4490

/-- Given an article with a total of 48,000 words printed on 21 pages,
    where 17 pages use smaller type with 2,400 words each,
    prove that the remaining pages in larger type contain 1,800 words each. -/
theorem words_per_page_larger_type :
  ∀ (total_words total_pages smaller_type_pages words_per_page_smaller : ℕ),
    total_words = 48000 →
    total_pages = 21 →
    smaller_type_pages = 17 →
    words_per_page_smaller = 2400 →
    (total_pages - smaller_type_pages) * 
      ((total_words - smaller_type_pages * words_per_page_smaller) / (total_pages - smaller_type_pages)) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_larger_type_l44_4490


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l44_4436

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l44_4436


namespace NUMINAMATH_CALUDE_arrangement_satisfies_condition_l44_4493

def arrangement : List ℕ := [3, 1, 4, 1, 3, 0, 2, 4, 2, 0]

def count_between (list : List ℕ) (n : ℕ) : ℕ :=
  match list.indexOf? n, list.reverse.indexOf? n with
  | some i, some j => list.length - i - j - 2
  | _, _ => 0

def satisfies_condition (list : List ℕ) : Prop :=
  ∀ n ∈ list, count_between list n = n

theorem arrangement_satisfies_condition : 
  satisfies_condition arrangement :=
sorry

end NUMINAMATH_CALUDE_arrangement_satisfies_condition_l44_4493


namespace NUMINAMATH_CALUDE_range_of_7a_minus_5b_l44_4447

theorem range_of_7a_minus_5b (a b : ℝ) 
  (h1 : 5 ≤ a - b ∧ a - b ≤ 27) 
  (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  ∃ (x : ℝ), 36 ≤ 7*a - 5*b ∧ 7*a - 5*b ≤ 192 ∧
  (∀ (y : ℝ), 36 ≤ y ∧ y ≤ 192 → ∃ (a' b' : ℝ), 
    (5 ≤ a' - b' ∧ a' - b' ≤ 27) ∧ 
    (6 ≤ a' + b' ∧ a' + b' ≤ 30) ∧ 
    y = 7*a' - 5*b') :=
by sorry

end NUMINAMATH_CALUDE_range_of_7a_minus_5b_l44_4447


namespace NUMINAMATH_CALUDE_shifted_sine_equivalence_shift_amount_l44_4405

/-- Proves that the given function is equivalent to a shifted sine function -/
theorem shifted_sine_equivalence (x : ℝ) : 
  (1/2 : ℝ) * Real.sin (4*x) - (Real.sqrt 3 / 2) * Real.cos (4*x) = Real.sin (4*x - π/3) :=
by sorry

/-- Proves that the shift is π/12 units to the right -/
theorem shift_amount : 
  ∃ (k : ℝ), ∀ (x : ℝ), Real.sin (4*x - π/3) = Real.sin (4*(x - k)) ∧ k = π/12 :=
by sorry

end NUMINAMATH_CALUDE_shifted_sine_equivalence_shift_amount_l44_4405


namespace NUMINAMATH_CALUDE_empty_set_implies_a_zero_l44_4459

theorem empty_set_implies_a_zero (a : ℝ) : (∀ x : ℝ, ax + 2 ≠ 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_a_zero_l44_4459


namespace NUMINAMATH_CALUDE_triangular_number_difference_l44_4475

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 30th and 28th triangular numbers is 59 -/
theorem triangular_number_difference : triangular_number 30 - triangular_number 28 = 59 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l44_4475


namespace NUMINAMATH_CALUDE_simplify_expression_l44_4430

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 + 2*b^2) - 2*b^2 + 5 = 9*b^4 + 6*b^3 - 2*b^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l44_4430


namespace NUMINAMATH_CALUDE_solution_to_system_l44_4466

theorem solution_to_system (x y z : ℝ) : 
  (3 * (x^2 + y^2 + z^2) = 1 ∧ 
   x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3) → 
  ((x = 0 ∧ y = 0 ∧ z = 1/Real.sqrt 3) ∨ 
   (x = 0 ∧ y = 0 ∧ z = -1/Real.sqrt 3) ∨ 
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
   (x = 1/3 ∧ y = 1/3 ∧ z = -1/3) ∨ 
   (x = 1/3 ∧ y = -1/3 ∧ z = 1/3) ∨ 
   (x = 1/3 ∧ y = -1/3 ∧ z = -1/3) ∨ 
   (x = -1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
   (x = -1/3 ∧ y = 1/3 ∧ z = -1/3) ∨ 
   (x = -1/3 ∧ y = -1/3 ∧ z = 1/3) ∨ 
   (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l44_4466


namespace NUMINAMATH_CALUDE_range_of_a_l44_4417

theorem range_of_a (x y a : ℝ) : 
  (77 * a = x + y) →
  (Real.sqrt (abs a) = Real.sqrt (x * y)) →
  (a ≤ -4 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l44_4417


namespace NUMINAMATH_CALUDE_line_parameterization_l44_4409

/-- Given a line y = 5x - 7 parameterized by [x, y] = [p, 3] + t[3, q], 
    prove that p = 2 and q = 15 -/
theorem line_parameterization (x y p q t : ℝ) : 
  (y = 5*x - 7) ∧ 
  (∃ t, x = p + 3*t ∧ y = 3 + q*t) →
  p = 2 ∧ q = 15 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l44_4409


namespace NUMINAMATH_CALUDE_x_value_l44_4480

theorem x_value : ∃ x : ℝ, x * 0.65 = 552.50 * 0.20 ∧ x = 170 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l44_4480


namespace NUMINAMATH_CALUDE_brown_dog_weight_l44_4486

/-- The weight of the brown dog -/
def brown_weight : ℝ := sorry

/-- The weight of the black dog -/
def black_weight : ℝ := brown_weight + 1

/-- The weight of the white dog -/
def white_weight : ℝ := 2 * brown_weight

/-- The weight of the grey dog -/
def grey_weight : ℝ := black_weight - 2

/-- The average weight of all dogs -/
def average_weight : ℝ := 5

theorem brown_dog_weight :
  (brown_weight + black_weight + white_weight + grey_weight) / 4 = average_weight →
  brown_weight = 4 := by sorry

end NUMINAMATH_CALUDE_brown_dog_weight_l44_4486


namespace NUMINAMATH_CALUDE_jack_walking_distance_l44_4478

/-- Calculates the distance walked given the time in hours and minutes and the walking rate in miles per hour -/
def distance_walked (hours : ℕ) (minutes : ℕ) (rate : ℚ) : ℚ :=
  rate * (hours + minutes / 60)

/-- Proves that walking for 1 hour and 15 minutes at a rate of 7.2 miles per hour results in a distance of 9 miles -/
theorem jack_walking_distance :
  distance_walked 1 15 (7.2 : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jack_walking_distance_l44_4478


namespace NUMINAMATH_CALUDE_village_cats_l44_4491

theorem village_cats (total : ℕ) (spotted : ℕ) (fluffy_spotted : ℕ) : 
  spotted = total / 3 →
  fluffy_spotted = spotted / 4 →
  fluffy_spotted = 10 →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_village_cats_l44_4491


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l44_4484

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ / a₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l44_4484


namespace NUMINAMATH_CALUDE_triangle_inequality_specific_l44_4457

/-- Triangle inequality theorem for a specific triangle --/
theorem triangle_inequality_specific (a b c : ℝ) (ha : a = 5) (hb : b = 8) (hc : c = 6) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_specific_l44_4457


namespace NUMINAMATH_CALUDE_port_distance_l44_4464

/-- The distance between two ports given travel times and current speed -/
theorem port_distance (downstream_time upstream_time current_speed : ℝ) 
  (h_downstream : downstream_time = 3)
  (h_upstream : upstream_time = 4)
  (h_current : current_speed = 5) : 
  ∃ (distance boat_speed : ℝ),
    distance = downstream_time * (boat_speed + current_speed) ∧
    distance = upstream_time * (boat_speed - current_speed) ∧
    distance = 120 := by
  sorry

end NUMINAMATH_CALUDE_port_distance_l44_4464


namespace NUMINAMATH_CALUDE_collinear_with_a_l44_4402

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

/-- Given vector a = (1, 2), prove that (k, 2k) is collinear with a for any non-zero real k -/
theorem collinear_with_a (k : ℝ) (hk : k ≠ 0) : 
  collinear (1, 2) (k, 2*k) := by
sorry

end NUMINAMATH_CALUDE_collinear_with_a_l44_4402


namespace NUMINAMATH_CALUDE_soccer_team_penalty_kicks_l44_4419

/-- Calculates the total number of penalty kicks in a soccer team training exercise. -/
def total_penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - goalies) * goalies

/-- Theorem: In a soccer team with 24 players, including 4 goalies, 
    where each player shoots once at each goalie, the total number of penalty kicks is 92. -/
theorem soccer_team_penalty_kicks :
  total_penalty_kicks 24 4 = 92 := by
  sorry


end NUMINAMATH_CALUDE_soccer_team_penalty_kicks_l44_4419


namespace NUMINAMATH_CALUDE_zero_descriptions_l44_4467

theorem zero_descriptions (x : ℝ) :
  (x = 0) ↔ 
  (∀ (y : ℝ), x ≤ y ∧ x ≥ y → y = x) ∧ 
  (∀ (y : ℝ), x + y = y) ∧
  (∀ (y : ℝ), x * y = x) :=
sorry

end NUMINAMATH_CALUDE_zero_descriptions_l44_4467


namespace NUMINAMATH_CALUDE_smallest_base_for_145_l44_4424

theorem smallest_base_for_145 :
  ∃ (b : ℕ), b = 12 ∧ 
  (∀ (n : ℕ), n^2 ≤ 145 ∧ 145 < n^3 → b ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_145_l44_4424


namespace NUMINAMATH_CALUDE_clock_movement_theorem_l44_4420

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ

/-- Represents a clock -/
structure Clock where
  startTime : Time
  degreeMoved : ℝ
  degreesPer12Hours : ℝ

/-- Calculates the ending time given a clock -/
def endingTime (c : Clock) : Time :=
  sorry

/-- The theorem to prove -/
theorem clock_movement_theorem (c : Clock) : 
  c.startTime = ⟨12, 0⟩ →
  c.degreeMoved = 74.99999999999999 →
  c.degreesPer12Hours = 360 →
  endingTime c = ⟨14, 30⟩ :=
sorry

end NUMINAMATH_CALUDE_clock_movement_theorem_l44_4420
