import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l857_85720

-- Define the hyperbola C
def hyperbola_C : Set (ℝ × ℝ) := sorry

-- Define the foci of the hyperbola
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point P on both the hyperbola and the parabola
def P : ℝ × ℝ := sorry

-- Define the eccentricity of a hyperbola
def eccentricity (h : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity :
  P ∈ hyperbola_C ∧ 
  parabola P.1 P.2 ∧
  dot_product (vector_add (vector_sub P F₂) (vector_sub F₁ F₂)) 
              (vector_sub (vector_sub P F₂) (vector_sub F₁ F₂)) = 0 →
  eccentricity hyperbola_C = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l857_85720


namespace NUMINAMATH_CALUDE_three_digit_number_sum_l857_85783

theorem three_digit_number_sum (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a ≠ 0 →
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 3194 →
  100 * a + 10 * b + c = 358 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_sum_l857_85783


namespace NUMINAMATH_CALUDE_constant_term_expansion_l857_85708

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the kth term in the expansion of (√x + 2/x²)^n -/
def coeff (n k : ℕ) : ℚ := binomial n k * 2^k

theorem constant_term_expansion (n : ℕ) :
  (coeff n 4 / coeff n 2 = 56 / 3) →
  (binomial n 2 * 2^2 = 180) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l857_85708


namespace NUMINAMATH_CALUDE_data_comparison_l857_85746

def set1 (x₁ x₂ x₃ x₄ x₅ : ℝ) := [x₁, x₂, x₃, x₄, x₅]
def set2 (x₁ x₂ x₃ x₄ x₅ : ℝ) := [2*x₁+3, 2*x₂+3, 2*x₃+3, 2*x₄+3, 2*x₅+3]

def standardDeviation (xs : List ℝ) : ℝ := sorry
def median (xs : List ℝ) : ℝ := sorry
def mean (xs : List ℝ) : ℝ := sorry

theorem data_comparison (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (standardDeviation (set2 x₁ x₂ x₃ x₄ x₅) ≠ standardDeviation (set1 x₁ x₂ x₃ x₄ x₅)) ∧
  (median (set2 x₁ x₂ x₃ x₄ x₅) ≠ median (set1 x₁ x₂ x₃ x₄ x₅)) ∧
  (mean (set2 x₁ x₂ x₃ x₄ x₅) ≠ mean (set1 x₁ x₂ x₃ x₄ x₅)) := by
  sorry

end NUMINAMATH_CALUDE_data_comparison_l857_85746


namespace NUMINAMATH_CALUDE_final_quiz_score_for_a_l857_85780

def number_of_quizzes : ℕ := 4
def average_score : ℚ := 92 / 100
def required_average : ℚ := 90 / 100

theorem final_quiz_score_for_a (final_score : ℚ) :
  (number_of_quizzes * average_score + final_score) / (number_of_quizzes + 1) ≥ required_average →
  final_score ≥ 82 / 100 :=
by sorry

end NUMINAMATH_CALUDE_final_quiz_score_for_a_l857_85780


namespace NUMINAMATH_CALUDE_equation_solution_l857_85712

theorem equation_solution : ∀ x : ℚ, 
  (Real.sqrt (6 * x) / Real.sqrt (4 * (x - 1)) = 3) → x = 24 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l857_85712


namespace NUMINAMATH_CALUDE_sales_and_profit_formula_optimal_price_reduction_no_solution_for_higher_profit_l857_85760

-- Define constants
def initial_cost : ℝ := 80
def initial_price : ℝ := 120
def initial_sales : ℝ := 20
def sales_increase_rate : ℝ := 2

-- Define functions
def daily_sales_increase (x : ℝ) : ℝ := sales_increase_rate * x
def profit_per_piece (x : ℝ) : ℝ := initial_price - initial_cost - x

-- Theorem 1
theorem sales_and_profit_formula (x : ℝ) :
  (daily_sales_increase x = 2 * x) ∧ (profit_per_piece x = 40 - x) := by sorry

-- Theorem 2
theorem optimal_price_reduction :
  ∃ x : ℝ, (profit_per_piece x) * (initial_sales + daily_sales_increase x) = 1200 ∧ x = 20 := by sorry

-- Theorem 3
theorem no_solution_for_higher_profit :
  ¬∃ y : ℝ, (profit_per_piece y) * (initial_sales + daily_sales_increase y) = 1800 := by sorry

end NUMINAMATH_CALUDE_sales_and_profit_formula_optimal_price_reduction_no_solution_for_higher_profit_l857_85760


namespace NUMINAMATH_CALUDE_roots_sum_bound_l857_85796

theorem roots_sum_bound (v w : ℂ) : 
  v ≠ w → 
  v^2021 = 1 → 
  w^2021 = 1 → 
  Complex.abs (v + w) < Real.sqrt (2 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_bound_l857_85796


namespace NUMINAMATH_CALUDE_chloe_score_l857_85773

/-- Calculates the total score in Chloe's video game. -/
def total_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : ℕ :=
  points_per_treasure * (treasures_level1 + treasures_level2)

/-- Proves that Chloe's total score is 81 points given the specified conditions. -/
theorem chloe_score :
  total_score 9 6 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_chloe_score_l857_85773


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_sum_l857_85768

-- Define the parallelogram and its properties
structure Parallelogram :=
  (area : ℝ)
  (pq_length : ℝ)
  (rs_length : ℝ)

-- Define the diagonal representation
structure DiagonalRepresentation :=
  (m : ℕ)
  (n : ℕ)
  (p : ℕ)

-- Define the theorem
theorem parallelogram_diagonal_sum 
  (ABCD : Parallelogram) 
  (h_area : ABCD.area = 24)
  (h_pq : ABCD.pq_length = 8)
  (h_rs : ABCD.rs_length = 10)
  (d_rep : DiagonalRepresentation)
  (h_prime : ∀ (q : ℕ), Prime q → ¬(q^2 ∣ d_rep.p))
  (h_diagonal : ∃ (d : ℝ), d^2 = d_rep.m + d_rep.n * Real.sqrt d_rep.p) :
  d_rep.m + d_rep.n + d_rep.p = 50 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_sum_l857_85768


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l857_85798

theorem arithmetic_calculations :
  (1405 - (816 + 487) = 102) ∧
  (3450 - 107 * 13 = 2059) ∧
  (48306 / (311 - 145) = 291) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l857_85798


namespace NUMINAMATH_CALUDE_triangle_area_part1_triangle_side_part2_l857_85714

-- Part 1
theorem triangle_area_part1 (A B C : ℝ) (a b c : ℝ) :
  A = π/6 → C = π/4 → a = 2 →
  (1/2) * a * b * Real.sin C = 1 + Real.sqrt 3 :=
sorry

-- Part 2
theorem triangle_side_part2 (A B C : ℝ) (a b c : ℝ) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 → b = 2 → C = π/3 →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_part1_triangle_side_part2_l857_85714


namespace NUMINAMATH_CALUDE_max_volume_sphere_in_cube_l857_85786

theorem max_volume_sphere_in_cube (edge_length : Real) (h : edge_length = 1) :
  let sphere_volume := (4 / 3) * Real.pi * (edge_length / 2)^3
  sphere_volume = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_sphere_in_cube_l857_85786


namespace NUMINAMATH_CALUDE_square_side_length_l857_85736

theorem square_side_length (s : ℝ) (S : ℝ) : 
  s = 5 →
  S = s + 5 →
  S^2 = 4 * s^2 →
  S = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l857_85736


namespace NUMINAMATH_CALUDE_lollipop_distribution_theorem_l857_85788

/-- Given a number of lollipops and kids, calculate the minimum number of additional
    lollipops needed for equal distribution -/
def min_additional_lollipops (total_lollipops : ℕ) (num_kids : ℕ) : ℕ :=
  let lollipops_per_kid := (total_lollipops + num_kids - 1) / num_kids
  lollipops_per_kid * num_kids - total_lollipops

/-- Theorem stating that for 650 lollipops and 42 kids, 
    the minimum number of additional lollipops needed is 22 -/
theorem lollipop_distribution_theorem :
  min_additional_lollipops 650 42 = 22 := by
  sorry


end NUMINAMATH_CALUDE_lollipop_distribution_theorem_l857_85788


namespace NUMINAMATH_CALUDE_function_value_proof_l857_85715

theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f ((1/2) * x - 1) = 2 * x - 5) →
  f a = 6 →
  a = 7/4 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l857_85715


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l857_85775

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 12 → num_groups = 6 → caps_per_group = total_caps / num_groups → caps_per_group = 2 := by
  sorry

#check bottle_cap_distribution

end NUMINAMATH_CALUDE_bottle_cap_distribution_l857_85775


namespace NUMINAMATH_CALUDE_inverse_relationship_R_squared_residuals_l857_85716

/-- Represents the coefficient of determination in regression analysis -/
def R_squared : ℝ := sorry

/-- Represents the sum of squares of residuals in regression analysis -/
def sum_of_squares_residuals : ℝ := sorry

/-- States that there is an inverse relationship between R² and the sum of squares of residuals -/
theorem inverse_relationship_R_squared_residuals :
  ∀ (R₁ R₂ : ℝ) (SSR₁ SSR₂ : ℝ),
    R₁ < R₂ → SSR₁ > SSR₂ :=
by sorry

end NUMINAMATH_CALUDE_inverse_relationship_R_squared_residuals_l857_85716


namespace NUMINAMATH_CALUDE_inequality_solution_l857_85734

theorem inequality_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x < -6 ∨ |x - 30| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 74 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l857_85734


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l857_85757

theorem solution_implies_k_value (k x y : ℚ) : 
  x = 3 ∧ y = 2 ∧ k * x + 3 * y = 1 → k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l857_85757


namespace NUMINAMATH_CALUDE_contrapositive_example_l857_85766

theorem contrapositive_example (a b : ℝ) :
  (¬(a + 1 > b) → ¬(a > b)) ↔ ((a + 1 ≤ b) → (a ≤ b)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l857_85766


namespace NUMINAMATH_CALUDE_best_fit_line_slope_l857_85758

/-- Represents a temperature measurement at a specific time -/
structure Measurement where
  time : ℝ
  temp : ℝ

/-- Given three equally spaced time measurements with corresponding temperatures,
    the slope of the best-fit line is (T₃ - T₁) / (t₃ - t₁) -/
theorem best_fit_line_slope (m₁ m₂ m₃ : Measurement) (h : ℝ) 
    (h1 : m₂.time = m₁.time + h)
    (h2 : m₃.time = m₁.time + 2 * h) :
  (m₃.temp - m₁.temp) / (m₃.time - m₁.time) =
    ((m₁.time - (m₁.time + h)) * (m₁.temp - (m₁.temp + m₂.temp + m₃.temp) / 3) +
     (m₂.time - (m₁.time + h)) * (m₂.temp - (m₁.temp + m₂.temp + m₃.temp) / 3) +
     (m₃.time - (m₁.time + h)) * (m₃.temp - (m₁.temp + m₂.temp + m₃.temp) / 3)) /
    ((m₁.time - (m₁.time + h))^2 + (m₂.time - (m₁.time + h))^2 + (m₃.time - (m₁.time + h))^2) :=
by sorry

end NUMINAMATH_CALUDE_best_fit_line_slope_l857_85758


namespace NUMINAMATH_CALUDE_functional_equation_solution_l857_85778

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) : 
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) → 
  ((∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l857_85778


namespace NUMINAMATH_CALUDE_students_not_picked_l857_85774

theorem students_not_picked (total : ℕ) (groups : ℕ) (per_group : ℕ) (h1 : total = 64) (h2 : groups = 4) (h3 : per_group = 7) : 
  total - (groups * per_group) = 36 := by
sorry

end NUMINAMATH_CALUDE_students_not_picked_l857_85774


namespace NUMINAMATH_CALUDE_frog_jump_probability_l857_85728

/-- Represents a position on the grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the grid -/
def Grid := {p : Position // p.x ≤ 5 ∧ p.y ≤ 5}

/-- The blocked cell -/
def blockedCell : Position := ⟨3, 3⟩

/-- Check if a position is on the grid boundary -/
def isOnBoundary (p : Position) : Bool :=
  p.x = 0 ∨ p.x = 5 ∨ p.y = 0 ∨ p.y = 5

/-- Check if a position is on a vertical side of the grid -/
def isOnVerticalSide (p : Position) : Bool :=
  p.x = 0 ∨ p.x = 5

/-- Probability of ending on a vertical side starting from a given position -/
noncomputable def probabilityVerticalSide (p : Position) : Real :=
  sorry

/-- Theorem: The probability of ending on a vertical side starting from (2,2) is 5/8 -/
theorem frog_jump_probability :
  probabilityVerticalSide ⟨2, 2⟩ = 5/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l857_85728


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l857_85791

/-- Represents the age ratio between a man and his son after two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  let man_age := son_age + age_difference
  (man_age + 2) / (son_age + 2)

/-- Theorem stating the age ratio between a man and his son after two years -/
theorem man_son_age_ratio :
  age_ratio 22 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l857_85791


namespace NUMINAMATH_CALUDE_height_range_l857_85762

def heights : List ℕ := [153, 167, 148, 170, 154, 166, 149, 159, 167, 153]

theorem height_range :
  (List.maximum heights).map (λ max =>
    (List.minimum heights).map (λ min =>
      max - min
    )
  ) = some 22 := by
  sorry

end NUMINAMATH_CALUDE_height_range_l857_85762


namespace NUMINAMATH_CALUDE_B_proper_subset_A_l857_85723

-- Define sets A and B
def A : Set ℝ := {x | x > (1/2)}
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem B_proper_subset_A : B ⊂ A := by sorry

end NUMINAMATH_CALUDE_B_proper_subset_A_l857_85723


namespace NUMINAMATH_CALUDE_tangent_line_equation_circle_radius_l857_85765

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*x + a = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 8*x - 15*y + 43 = 0

-- Define the dot product of vectors OA and OB
def dot_product_OA_OB (a : ℝ) : ℝ := -6

-- Theorem for part 1
theorem tangent_line_equation :
  ∀ x y : ℝ,
  circle_M (-8) x y →
  (tangent_line_1 x ∨ tangent_line_2 x y) →
  (x - (point_P.1))^2 + (y - (point_P.2))^2 = 
  (x - 1)^2 + y^2 :=
sorry

-- Theorem for part 2
theorem circle_radius :
  ∀ a : ℝ,
  dot_product_OA_OB a = -6 →
  ∃ r : ℝ, r^2 = 7 ∧
  ∀ x y : ℝ, circle_M a x y → (x - 1)^2 + y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_circle_radius_l857_85765


namespace NUMINAMATH_CALUDE_triangle_properties_l857_85790

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) 
  (h2 : t.a = Real.sqrt 3)
  (h3 : Real.cos t.C = Real.sqrt 3 / 3) :
  (t.A = π / 3) ∧ (t.c = 2 * Real.sqrt 6 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l857_85790


namespace NUMINAMATH_CALUDE_chess_tournament_games_l857_85769

/-- The number of games played in a chess tournament with n participants,
    where each participant plays exactly one game with each other participant. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 22 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 231. -/
theorem chess_tournament_games :
  tournament_games 22 = 231 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l857_85769


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l857_85726

theorem distance_between_circle_centers (a b c : ℝ) (ha : a = 7) (hb : b = 8) (hc : c = 9) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * A)
  let r := A / s
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let O₁O₂ := Real.sqrt (R^2 + 2 * R * r * cos_C + r^2)
  ∃ ε > 0, abs (O₁O₂ - 5.75) < ε :=
sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l857_85726


namespace NUMINAMATH_CALUDE_eulers_formula_l857_85779

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Euler's formula for convex polyhedra states that the number of faces minus the number of edges plus the number of vertices equals two. -/
theorem eulers_formula (p : ConvexPolyhedron) : p.faces - p.edges + p.vertices = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l857_85779


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l857_85787

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (16 / x) + (108 / y) + x * y ≥ 36 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ x y, (16 / x) + (108 / y) + x * y = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l857_85787


namespace NUMINAMATH_CALUDE_carols_weight_l857_85733

/-- Given two people's weights satisfying certain conditions, prove Carol's weight. -/
theorem carols_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 220)
  (h2 : alice_weight + 2 * carol_weight = 280) : 
  carol_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_carols_weight_l857_85733


namespace NUMINAMATH_CALUDE_system_solution_l857_85735

theorem system_solution (x y b : ℚ) : 
  5 * x - 2 * y = b →
  3 * x + 4 * y = 3 * b →
  y = 3 →
  b = 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l857_85735


namespace NUMINAMATH_CALUDE_harry_uses_whole_bag_l857_85725

/-- The number of batches of cookies -/
def num_batches : ℕ := 3

/-- The number of chocolate chips per cookie -/
def chips_per_cookie : ℕ := 9

/-- The number of chips in a bag -/
def chips_per_bag : ℕ := 81

/-- The number of cookies in a batch -/
def cookies_per_batch : ℕ := 3

/-- The portion of the bag used for making the dough -/
def portion_used : ℚ := (num_batches * cookies_per_batch * chips_per_cookie) / chips_per_bag

theorem harry_uses_whole_bag : portion_used = 1 := by
  sorry

end NUMINAMATH_CALUDE_harry_uses_whole_bag_l857_85725


namespace NUMINAMATH_CALUDE_min_sum_distances_l857_85743

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [OrderedAddCommGroup α]

def points (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α) : Prop :=
  P₁ < P₂ ∧ P₂ < P₃ ∧ P₃ < P₄ ∧ P₄ < P₅ ∧ P₅ < P₆ ∧ P₆ < P₇

def distance (x y : α) : α := abs (x - y)

def sum_distances (P : α) (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α) : α :=
  distance P P₁ + distance P P₂ + distance P P₃ + distance P P₄ +
  distance P P₅ + distance P P₆ + distance P P₇

theorem min_sum_distances
  (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α)
  (h : points P₁ P₂ P₃ P₄ P₅ P₆ P₇) :
  ∀ P, sum_distances P P₁ P₂ P₃ P₄ P₅ P₆ P₇ ≥ sum_distances P₄ P₁ P₂ P₃ P₄ P₅ P₆ P₇ ∧
  (sum_distances P P₁ P₂ P₃ P₄ P₅ P₆ P₇ = sum_distances P₄ P₁ P₂ P₃ P₄ P₅ P₆ P₇ ↔ P = P₄) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_l857_85743


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l857_85754

theorem x_range_for_inequality (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2*x - 1 > m*(x^2 - 1)) ↔ 
  ((Real.sqrt 7 - 1) / 2 < x ∧ x < (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l857_85754


namespace NUMINAMATH_CALUDE_fraction_problem_l857_85792

theorem fraction_problem (a b : ℚ) : 
  b / (a - 2) = 3 / 4 →
  b / (a + 9) = 5 / 7 →
  b / a = 165 / 222 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l857_85792


namespace NUMINAMATH_CALUDE_b_range_l857_85772

noncomputable section

def y (a x : ℝ) : ℝ := a^x

def f (a b x : ℝ) : ℝ := a^x + (Real.log x) / (Real.log a) + b

theorem b_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, y a x ≤ 6 - y a x) →
  (∃ x ∈ Set.Ioo 1 2, f a b x = 0) →
  -5 < b ∧ b < -2 :=
sorry

end NUMINAMATH_CALUDE_b_range_l857_85772


namespace NUMINAMATH_CALUDE_max_friends_is_m_l857_85761

/-- Represents a compartment of passengers -/
structure Compartment where
  passengers : Type
  friendship : passengers → passengers → Prop
  m : ℕ
  h_m : m ≥ 3
  h_symmetric : ∀ a b, friendship a b ↔ friendship b a
  h_irreflexive : ∀ a, ¬friendship a a
  h_unique_common_friend : ∀ (S : Finset passengers), S.card = m → 
    ∃! f, ∀ s ∈ S, friendship f s

/-- The maximum number of friends any passenger can have is m -/
theorem max_friends_is_m (C : Compartment) : 
  ∃ (max_friends : ℕ), max_friends = C.m ∧ 
    ∀ p : C.passengers, ∃ (friends : Finset C.passengers), 
      (∀ f ∈ friends, C.friendship p f) ∧ 
      friends.card ≤ max_friends :=
sorry

end NUMINAMATH_CALUDE_max_friends_is_m_l857_85761


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l857_85704

theorem intersection_point_of_function_and_inverse (b a : ℤ) : 
  let f : ℝ → ℝ := λ x ↦ -2 * x + b
  let f_inv : ℝ → ℝ := Function.invFun f
  (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x) ∧ f 2 = a ∧ f_inv 2 = a
  → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l857_85704


namespace NUMINAMATH_CALUDE_points_on_line_relationship_l857_85752

/-- Given two points A(-2, y₁) and B(1, y₂) on the line y = -2x + 3, prove that y₁ > y₂ -/
theorem points_on_line_relationship (y₁ y₂ : ℝ) : 
  ((-2 : ℝ), y₁) ∈ {(x, y) | y = -2*x + 3} → 
  ((1 : ℝ), y₂) ∈ {(x, y) | y = -2*x + 3} → 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_relationship_l857_85752


namespace NUMINAMATH_CALUDE_binary_string_power_of_two_sum_l857_85797

/-- A binary string is represented as a list of booleans, where true represents 1 and false represents 0. -/
def BinaryString := List Bool

/-- Count the number of ones in a binary string. -/
def countOnes (s : BinaryString) : Nat :=
  s.filter id |>.length

/-- Represents a way of inserting plus signs into a binary string. 
    true means "insert a plus sign after this digit", false means "don't insert". -/
def PlusInsertion := List Bool

/-- Compute the sum of a binary string with plus signs inserted according to a PlusInsertion. -/
def computeSum (s : BinaryString) (insertion : PlusInsertion) : Nat :=
  sorry  -- Implementation details omitted for brevity

/-- Check if a number is a power of two. -/
def isPowerOfTwo (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

/-- The main theorem statement. -/
theorem binary_string_power_of_two_sum 
  (s : BinaryString) 
  (h : countOnes s ≥ 2017) : 
  ∃ insertion : PlusInsertion, isPowerOfTwo (computeSum s insertion) := by
  sorry


end NUMINAMATH_CALUDE_binary_string_power_of_two_sum_l857_85797


namespace NUMINAMATH_CALUDE_range_of_a_l857_85724

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l857_85724


namespace NUMINAMATH_CALUDE_hyperbola_equation_l857_85703

/-- Definition of a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  intersection_line : ℝ → ℝ
  midpoint_x : ℝ

/-- Theorem stating the equation of the hyperbola with given properties -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focus : h.focus = (Real.sqrt 7, 0))
  (h_line : h.intersection_line = fun x ↦ x - 1)
  (h_midpoint : h.midpoint_x = -2/3) :
  ∃ (x y : ℝ), x^2/2 - y^2/5 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l857_85703


namespace NUMINAMATH_CALUDE_vector_arrangements_l857_85789

-- Define a structure for a vector in 2D space
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define a function to check if two vectors are parallel
def areParallel (v1 v2 : Vector2D) : Prop :=
  ∃ (k : ℝ), v1.x = k * v2.x ∧ v1.y = k * v2.y

-- Define a function to check if a quadrilateral is non-convex
def isNonConvex (v1 v2 v3 v4 : Vector2D) : Prop :=
  sorry -- Definition of non-convex quadrilateral

-- Define a function to check if a four-segment broken line is self-intersecting
def isSelfIntersecting (v1 v2 v3 v4 : Vector2D) : Prop :=
  sorry -- Definition of self-intersecting broken line

theorem vector_arrangements (v1 v2 v3 v4 : Vector2D) :
  (¬ areParallel v1 v2 ∧ ¬ areParallel v1 v3 ∧ ¬ areParallel v1 v4 ∧
   ¬ areParallel v2 v3 ∧ ¬ areParallel v2 v4 ∧ ¬ areParallel v3 v4) →
  (v1.x + v2.x + v3.x + v4.x = 0 ∧ v1.y + v2.y + v3.y + v4.y = 0) →
  (∃ (a b c d : Vector2D), isNonConvex a b c d) ∧
  (∃ (a b c d : Vector2D), isSelfIntersecting a b c d) :=
by
  sorry


end NUMINAMATH_CALUDE_vector_arrangements_l857_85789


namespace NUMINAMATH_CALUDE_all_cells_equal_l857_85737

/-- Represents a 10x10 board with integer values -/
def Board := Fin 10 → Fin 10 → ℤ

/-- Predicate to check if a board satisfies the given conditions -/
def satisfies_conditions (b : Board) : Prop :=
  ∃ d : ℤ,
    (∀ i : Fin 10, b i i = d) ∧
    (∀ i j : Fin 10, b i j ≤ d)

/-- Theorem stating that if a board satisfies the conditions, all cells are equal -/
theorem all_cells_equal (b : Board) (h : satisfies_conditions b) :
    ∃ d : ℤ, ∀ i j : Fin 10, b i j = d := by
  sorry


end NUMINAMATH_CALUDE_all_cells_equal_l857_85737


namespace NUMINAMATH_CALUDE_f_properties_l857_85777

open Real

noncomputable def f (x : ℝ) : ℝ := x / (exp x - 1)

theorem f_properties :
  (∀ x > 0, ∀ y > x, f y < f x) ∧
  (∀ a > 2, ∃ x₀ > 0, f x₀ < a / (exp x₀ + 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l857_85777


namespace NUMINAMATH_CALUDE_worker_arrival_time_l857_85739

theorem worker_arrival_time (S : ℝ) (D : ℝ) (h1 : D = S * 36) (h2 : S > 0) :
  D / (3/4 * S) - 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_worker_arrival_time_l857_85739


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l857_85706

/-- Represents the scenario of a motorboat and kayak traveling on a river -/
structure RiverTravel where
  r : ℝ  -- Speed of the river current (and kayak's speed)
  m : ℝ  -- Speed of the motorboat relative to the river
  t : ℝ  -- Time for motorboat to travel from X to Y
  total_time : ℝ  -- Total time until motorboat meets kayak

/-- The theorem representing the problem -/
theorem motorboat_travel_time (rt : RiverTravel) : 
  rt.m = rt.r ∧ rt.total_time = 8 → rt.t = 4 := by
  sorry

#check motorboat_travel_time

end NUMINAMATH_CALUDE_motorboat_travel_time_l857_85706


namespace NUMINAMATH_CALUDE_rect_to_polar_8_8_l857_85776

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_8_8 :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 8 * Real.sqrt 2 ∧ θ = π / 4 ∧
  8 = r * Real.cos θ ∧ 8 = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rect_to_polar_8_8_l857_85776


namespace NUMINAMATH_CALUDE_lizette_minerva_stamp_difference_l857_85732

theorem lizette_minerva_stamp_difference :
  let lizette_stamps : ℕ := 813
  let minerva_stamps : ℕ := 688
  lizette_stamps - minerva_stamps = 125 := by
sorry

end NUMINAMATH_CALUDE_lizette_minerva_stamp_difference_l857_85732


namespace NUMINAMATH_CALUDE_max_value_quadratic_l857_85767

theorem max_value_quadratic :
  (∃ (p : ℝ), -3 * p^2 + 24 * p + 5 = 53) ∧
  (∀ (p : ℝ), -3 * p^2 + 24 * p + 5 ≤ 53) := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l857_85767


namespace NUMINAMATH_CALUDE_slower_train_speed_theorem_l857_85718

/-- The speed of the faster train in km/h -/
def faster_train_speed : ℝ := 120

/-- The length of the first train in meters -/
def train_length_1 : ℝ := 500

/-- The length of the second train in meters -/
def train_length_2 : ℝ := 700

/-- The time taken for the trains to cross each other in seconds -/
def crossing_time : ℝ := 19.6347928529354

/-- The speed of the slower train in km/h -/
def slower_train_speed : ℝ := 100

theorem slower_train_speed_theorem :
  let total_length := train_length_1 + train_length_2
  let relative_speed := (slower_train_speed + faster_train_speed) * (1000 / 3600)
  total_length = relative_speed * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_theorem_l857_85718


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l857_85729

/-- The total number of balls -/
def total_balls : ℕ := 6

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of black balls -/
def black_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing two balls of the same color -/
def prob_same_color : ℚ := 2/5

/-- The probability of drawing two balls of different colors -/
def prob_diff_color : ℚ := 3/5

theorem ball_drawing_probabilities :
  (prob_same_color + prob_diff_color = 1) ∧
  (prob_same_color = 2/5) ∧
  (prob_diff_color = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l857_85729


namespace NUMINAMATH_CALUDE_constant_value_l857_85799

theorem constant_value : ∀ (x : ℝ) (c : ℝ),
  (5 * x + c = 10 * x - 22) →
  (x = 5) →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l857_85799


namespace NUMINAMATH_CALUDE_conference_handshakes_l857_85700

/-- The number of handshakes in a conference with multiple companies --/
def num_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that the number of handshakes for the given scenario is 75 --/
theorem conference_handshakes :
  num_handshakes 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l857_85700


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l857_85741

/-- Represents the average number of days it takes for one cow to eat one bag of husk -/
def average_days_per_bag (num_cows : ℕ) (total_bags : ℕ) (total_days : ℕ) : ℚ :=
  (num_cows * total_days : ℚ) / total_bags

/-- Proves that given 30 cows consuming 50 bags of husk in 20 days, 
    the average number of days it takes for one cow to eat one bag of husk is 12 days -/
theorem dairy_farm_husk_consumption :
  average_days_per_bag 30 50 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l857_85741


namespace NUMINAMATH_CALUDE_quadrilateral_cosine_sum_l857_85710

theorem quadrilateral_cosine_sum (α β γ δ : Real) :
  (α + β + γ + δ = 2 * Real.pi) →
  (Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) →
  (α + β = Real.pi) ∨ (α + γ = Real.pi) ∨ (α + δ = Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_cosine_sum_l857_85710


namespace NUMINAMATH_CALUDE_attendees_with_all_items_l857_85753

def venue_capacity : ℕ := 5400
def tshirt_interval : ℕ := 90
def cap_interval : ℕ := 45
def wristband_interval : ℕ := 60

theorem attendees_with_all_items :
  (venue_capacity / (Nat.lcm tshirt_interval (Nat.lcm cap_interval wristband_interval))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_attendees_with_all_items_l857_85753


namespace NUMINAMATH_CALUDE_right_angled_triangle_m_values_l857_85781

/-- Given three lines that form a right-angled triangle, prove the possible values of m -/
theorem right_angled_triangle_m_values :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), 3*x + 2*y + 6 = 0 ∧ 2*x - 3*m^2*y + 18 = 0 ∧ 2*m*x - 3*y + 12 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (3*x₁ + 2*y₁ + 6 = 0 ∧ 2*x₁ - 3*m^2*y₁ + 18 = 0) ∧
    (3*x₂ + 2*y₂ + 6 = 0 ∧ 2*m*x₂ - 3*y₂ + 12 = 0) ∧
    ((3*2 + 2*(-3*m^2) = 0) ∨ (3*(2*m) + 2*(-3) = 0) ∨ (2*(-3*m^2) + (-3)*(2*m) = 0))) →
  m = 0 ∨ m = -1 ∨ m = -4/9 :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_m_values_l857_85781


namespace NUMINAMATH_CALUDE_impossible_cover_all_endings_l857_85771

theorem impossible_cover_all_endings (a : Fin 14 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ¬(∀ d : Fin 100, ∃ k l : Fin 14, (a k + a l) % 100 = d) := by
  sorry

end NUMINAMATH_CALUDE_impossible_cover_all_endings_l857_85771


namespace NUMINAMATH_CALUDE_fraction_sum_l857_85785

theorem fraction_sum : (3 : ℚ) / 5 + 2 / 15 = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l857_85785


namespace NUMINAMATH_CALUDE_external_tangent_points_theorem_l857_85744

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def intersect (c1 c2 : Circle) : Prop := sorry

def touches (c1 c2 : Circle) (p : Point) : Prop := sorry

def on_line (p : Point) (l : Line) : Prop := sorry

def passes_through (l : Line) (p : Point) : Prop := sorry

-- Main theorem
theorem external_tangent_points_theorem 
  (C C' : Circle) (X Y : Point) 
  (h1 : intersect C C') 
  (h2 : on_line X (Line.mk 0 1 0)) 
  (h3 : on_line Y (Line.mk 0 1 0)) :
  ∃ (T1 T2 T3 T4 : Point),
    ∀ (P Q R S : Point) (third_circle : Circle),
      touches C third_circle P →
      touches C' third_circle Q →
      on_line R (Line.mk 0 1 0) →
      on_line S (Line.mk 0 1 0) →
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_points_theorem_l857_85744


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l857_85751

def product_of_evens (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    Finset.prod (Finset.range ((x / 2) + 1)) (fun i => 2 * i)
  else
    Finset.prod (Finset.range (x / 2)) (fun i => 2 * i)

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem greatest_prime_factor_of_sum : 
  greatest_prime_factor (product_of_evens 26 + product_of_evens 24) = 23 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l857_85751


namespace NUMINAMATH_CALUDE_bob_garden_area_l857_85749

/-- Calculates the area of a garden given property dimensions and garden proportions. -/
def garden_area (property_width property_length : ℝ) (garden_width_ratio garden_length_ratio : ℝ) : ℝ :=
  (property_width * garden_width_ratio) * (property_length * garden_length_ratio)

/-- Theorem stating that Bob's garden area is 28125 square feet. -/
theorem bob_garden_area :
  garden_area 1000 2250 (1/8) (1/10) = 28125 := by
  sorry

end NUMINAMATH_CALUDE_bob_garden_area_l857_85749


namespace NUMINAMATH_CALUDE_infinitely_many_square_sum_averages_l857_85701

theorem infinitely_many_square_sum_averages :
  ∀ k : ℕ, ∃ n > k, ∃ m : ℕ, ((n + 1) * (2 * n + 1)) / 6 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_square_sum_averages_l857_85701


namespace NUMINAMATH_CALUDE_carlos_blocks_given_l857_85795

/-- The number of blocks Carlos gave to Rachel -/
def blocks_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem carlos_blocks_given :
  blocks_given 58 37 = 21 := by
  sorry

end NUMINAMATH_CALUDE_carlos_blocks_given_l857_85795


namespace NUMINAMATH_CALUDE_inequality_condition_l857_85722

theorem inequality_condition (t : ℝ) : (t + 1) * (1 - |t|) > 0 ↔ t < 1 ∧ t ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l857_85722


namespace NUMINAMATH_CALUDE_robins_total_distance_l857_85713

/-- The total distance Robin walks given his journey to the city center -/
theorem robins_total_distance (distance_to_center : ℕ) (initial_distance : ℕ) : 
  distance_to_center = 500 → initial_distance = 200 → 
  initial_distance + initial_distance + distance_to_center = 900 := by
sorry

end NUMINAMATH_CALUDE_robins_total_distance_l857_85713


namespace NUMINAMATH_CALUDE_octal_536_to_base7_l857_85750

def octal_to_decimal (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 6 * 8^0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  [1, 0, 1, 0]

theorem octal_536_to_base7 :
  decimal_to_base7 (octal_to_decimal 536) = [1, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_octal_536_to_base7_l857_85750


namespace NUMINAMATH_CALUDE_power_equation_solution_l857_85747

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 17 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l857_85747


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_120_l857_85755

/-- Given a sequence of consecutive natural numbers with sum 120, 
    the largest number in the sequence is 26 -/
theorem largest_consecutive_sum_120 (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : (n : ℝ) * (2 * a + n - 1) / 2 = 120) :
  a + n - 1 ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_120_l857_85755


namespace NUMINAMATH_CALUDE_expansion_proofs_l857_85738

theorem expansion_proofs (x a b c : ℝ) : 
  (3*(x+1)*(x-1) - (3*x+2)*(2-3*x) = 12*x^2 - 7) ∧ 
  ((a+2*b+3*c)*(a+2*b-3*c) = a^2 + 4*a*b + 4*b^2 - 9*c^2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_proofs_l857_85738


namespace NUMINAMATH_CALUDE_systematic_sampling_tenth_group_l857_85782

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstDraw : ℕ) (n : ℕ) : ℕ :=
  firstDraw + (totalStudents / sampleSize) * (n - 1)

/-- Theorem: In a systematic sampling of 1000 students into 100 groups,
    if the number drawn from the first group is 6,
    then the number drawn from the tenth group is 96. -/
theorem systematic_sampling_tenth_group :
  systematicSample 1000 100 6 10 = 96 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_tenth_group_l857_85782


namespace NUMINAMATH_CALUDE_tour_group_size_l857_85719

/-- Represents the number of people in a tour group -/
structure TourGroup where
  adults : ℕ
  children : ℕ

/-- Calculates the total cost for a tour group -/
def totalCost (g : TourGroup) : ℕ :=
  8 * g.adults + 3 * g.children

/-- Theorem: Given the ticket prices and total spent, the only possible numbers of people in the tour group are 8 or 13 -/
theorem tour_group_size :
  ∀ g : TourGroup, totalCost g = 44 → g.adults + g.children = 8 ∨ g.adults + g.children = 13 :=
by
  sorry

#check tour_group_size

end NUMINAMATH_CALUDE_tour_group_size_l857_85719


namespace NUMINAMATH_CALUDE_polynomial_factorization_l857_85745

theorem polynomial_factorization (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b)^2 * (b - c) * (c - a) * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l857_85745


namespace NUMINAMATH_CALUDE_tylers_and_brothers_age_sum_l857_85709

theorem tylers_and_brothers_age_sum : 
  ∀ (tyler_age brother_age : ℕ),
    tyler_age = 7 →
    brother_age = tyler_age + 3 →
    tyler_age + brother_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_tylers_and_brothers_age_sum_l857_85709


namespace NUMINAMATH_CALUDE_jewelry_pattern_purple_beads_jewelry_pattern_purple_beads_proof_l857_85731

theorem jewelry_pattern_purple_beads : ℕ → Prop :=
  fun purple_beads =>
    let green_beads : ℕ := 3
    let red_beads : ℕ := 2 * green_beads
    let pattern_total : ℕ := green_beads + purple_beads + red_beads
    let bracelet_repeats : ℕ := 3
    let necklace_repeats : ℕ := 5
    let bracelet_beads : ℕ := bracelet_repeats * pattern_total
    let necklace_beads : ℕ := necklace_repeats * pattern_total
    let total_beads : ℕ := 742
    let num_bracelets : ℕ := 1
    let num_necklaces : ℕ := 10
    num_bracelets * bracelet_beads + num_necklaces * necklace_beads = total_beads →
    purple_beads = 5

-- Proof
theorem jewelry_pattern_purple_beads_proof : jewelry_pattern_purple_beads 5 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_pattern_purple_beads_jewelry_pattern_purple_beads_proof_l857_85731


namespace NUMINAMATH_CALUDE_three_times_value_interval_examples_l857_85756

/-- A function has a k-times value interval if there exists a closed interval [a,b]
    such that the function is monotonic on [a,b] and its range on [a,b] is [ka,kb] --/
def has_k_times_value_interval (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
  (∀ y, f a ≤ y ∧ y ≤ f b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y) ∧
  f a = k * a ∧ f b = k * b

theorem three_times_value_interval_examples :
  (has_k_times_value_interval (fun x => 1 / x) 3) ∧
  (has_k_times_value_interval (fun x => x ^ 2) 3) := by
  sorry

end NUMINAMATH_CALUDE_three_times_value_interval_examples_l857_85756


namespace NUMINAMATH_CALUDE_points_on_decreasing_line_l857_85721

theorem points_on_decreasing_line (a₁ a₂ b₁ b₂ : ℝ) :
  a₁ ≠ a₂ →
  b₁ = -3 * a₁ + 4 →
  b₂ = -3 * a₂ + 4 →
  (a₁ - a₂) * (b₁ - b₂) < 0 :=
by sorry

end NUMINAMATH_CALUDE_points_on_decreasing_line_l857_85721


namespace NUMINAMATH_CALUDE_probability_factor_less_than_5_l857_85740

def factors_of_90 : Finset ℕ := sorry

theorem probability_factor_less_than_5 : 
  (Finset.filter (λ x => x < 5) factors_of_90).card / factors_of_90.card = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_5_l857_85740


namespace NUMINAMATH_CALUDE_board_length_l857_85711

-- Define the lengths of the two pieces
def shorter_piece : ℝ := 2
def longer_piece : ℝ := 2 * shorter_piece

-- Define the total length of the board
def total_length : ℝ := shorter_piece + longer_piece

-- Theorem to prove
theorem board_length : total_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_board_length_l857_85711


namespace NUMINAMATH_CALUDE_equation_solution_l857_85742

theorem equation_solution (y : ℝ) : (30 : ℝ) / 50 = Real.sqrt (y / 50) → y = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l857_85742


namespace NUMINAMATH_CALUDE_remaining_students_l857_85727

/-- Given a group of students divided into 3 groups of 8, with 2 students leaving early,
    prove that 22 students remain. -/
theorem remaining_students (initial_groups : Nat) (students_per_group : Nat) (students_left : Nat) :
  initial_groups = 3 →
  students_per_group = 8 →
  students_left = 2 →
  initial_groups * students_per_group - students_left = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_l857_85727


namespace NUMINAMATH_CALUDE_bridge_length_proof_l857_85748

/-- 
Given a train with length 120 meters crossing a bridge in 55 seconds at a speed of 39.27272727272727 m/s,
prove that the length of the bridge is 2040 meters.
-/
theorem bridge_length_proof (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 120 →
  crossing_time = 55 →
  train_speed = 39.27272727272727 →
  train_speed * crossing_time - train_length = 2040 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l857_85748


namespace NUMINAMATH_CALUDE_tim_kittens_l857_85764

theorem tim_kittens (initial : ℕ) (given_away : ℕ) (received : ℕ) :
  initial = 6 →
  given_away = 3 →
  received = 9 →
  initial - given_away + received = 12 := by
sorry

end NUMINAMATH_CALUDE_tim_kittens_l857_85764


namespace NUMINAMATH_CALUDE_count_divisors_5940_mult_6_l857_85770

/-- The number of positive divisors of 5940 that are multiples of 6 -/
def divisors_5940_mult_6 : ℕ := 24

/-- 5940 expressed as a product of prime factors -/
def factorization_5940 : ℕ := 2^2 * 3^3 * 5 * 11

theorem count_divisors_5940_mult_6 :
  (∀ d : ℕ, d > 0 ∧ d ∣ factorization_5940 ∧ 6 ∣ d) →
  (∃! n : ℕ, n = divisors_5940_mult_6) :=
sorry

end NUMINAMATH_CALUDE_count_divisors_5940_mult_6_l857_85770


namespace NUMINAMATH_CALUDE_inequality_proof_l857_85707

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l857_85707


namespace NUMINAMATH_CALUDE_subset_condition_implies_m_range_l857_85759

theorem subset_condition_implies_m_range (m : ℝ) : 
  (∀ x, -1 < x ∧ x < 2 → -1 < x ∧ x < m + 1) ∧ 
  (∃ y, -1 < y ∧ y < m + 1 ∧ ¬(-1 < y ∧ y < 2)) → 
  m > 1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_implies_m_range_l857_85759


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l857_85793

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 3 ∧ x ≠ 2 ∧ (1 - x) / (x - 2) = m / (2 - x) - 2) → 
  m < 6 ∧ m ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l857_85793


namespace NUMINAMATH_CALUDE_july_birth_percentage_l857_85794

def total_athletes : ℕ := 120
def july_athletes : ℕ := 18

def percentage_born_in_july : ℚ := july_athletes / total_athletes * 100

theorem july_birth_percentage :
  percentage_born_in_july = 15 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l857_85794


namespace NUMINAMATH_CALUDE_simplify_fraction_l857_85717

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (75 * b^3) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l857_85717


namespace NUMINAMATH_CALUDE_apples_joan_can_buy_l857_85763

def total_budget : ℕ := 60
def hummus_containers : ℕ := 2
def hummus_price : ℕ := 5
def chicken_price : ℕ := 20
def bacon_price : ℕ := 10
def vegetables_price : ℕ := 10
def apple_price : ℕ := 2

theorem apples_joan_can_buy :
  (total_budget - (hummus_containers * hummus_price + chicken_price + bacon_price + vegetables_price)) / apple_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_apples_joan_can_buy_l857_85763


namespace NUMINAMATH_CALUDE_wedding_bouquets_l857_85784

/-- Represents the number of flowers of each type --/
structure FlowerCount where
  roses : ℕ
  lilies : ℕ
  tulips : ℕ
  sunflowers : ℕ

/-- Represents the requirements for a single bouquet --/
def BouquetRequirement : FlowerCount :=
  { roses := 2, lilies := 1, tulips := 3, sunflowers := 1 }

/-- Calculates the number of complete bouquets that can be made --/
def completeBouquets (available : FlowerCount) : ℕ :=
  min (available.roses / BouquetRequirement.roses)
    (min (available.lilies / BouquetRequirement.lilies)
      (min (available.tulips / BouquetRequirement.tulips)
        (available.sunflowers / BouquetRequirement.sunflowers)))

theorem wedding_bouquets :
  let initial : FlowerCount := { roses := 48, lilies := 40, tulips := 76, sunflowers := 34 }
  let wilted : FlowerCount := { roses := 24, lilies := 10, tulips := 14, sunflowers := 7 }
  let remaining : FlowerCount := {
    roses := initial.roses - wilted.roses,
    lilies := initial.lilies - wilted.lilies,
    tulips := initial.tulips - wilted.tulips,
    sunflowers := initial.sunflowers - wilted.sunflowers
  }
  completeBouquets remaining = 12 := by sorry

end NUMINAMATH_CALUDE_wedding_bouquets_l857_85784


namespace NUMINAMATH_CALUDE_shaded_area_proof_l857_85705

theorem shaded_area_proof (rectangle_length rectangle_width : ℝ)
  (triangle_a_leg1 triangle_a_leg2 : ℝ)
  (triangle_b_leg1 triangle_b_leg2 : ℝ)
  (h1 : rectangle_length = 14)
  (h2 : rectangle_width = 7)
  (h3 : triangle_a_leg1 = 8)
  (h4 : triangle_a_leg2 = 5)
  (h5 : triangle_b_leg1 = 6)
  (h6 : triangle_b_leg2 = 2) :
  rectangle_length * rectangle_width - 3 * ((1/2 * triangle_a_leg1 * triangle_a_leg2) + (1/2 * triangle_b_leg1 * triangle_b_leg2)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l857_85705


namespace NUMINAMATH_CALUDE_park_bushes_count_l857_85730

def park_bushes (initial_orchids initial_roses initial_tulips added_orchids removed_roses : ℕ) : ℕ × ℕ × ℕ :=
  let final_orchids := initial_orchids + added_orchids
  let final_roses := initial_roses - removed_roses
  let final_tulips := initial_tulips * 2
  (final_orchids, final_roses, final_tulips)

theorem park_bushes_count : park_bushes 2 5 3 4 1 = (6, 4, 6) := by sorry

end NUMINAMATH_CALUDE_park_bushes_count_l857_85730


namespace NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l857_85702

theorem cinnamon_swirls_distribution (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → num_people = 3 → total_pieces = num_people * pieces_per_person → pieces_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l857_85702
