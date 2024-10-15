import Mathlib

namespace NUMINAMATH_CALUDE_potato_peeling_theorem_l609_60947

def potato_peeling_problem (julie_rate ted_rate combined_time : ℝ) : Prop :=
  let julie_part := combined_time * julie_rate
  let ted_part := combined_time * ted_rate
  let remaining_part := 1 - (julie_part + ted_part)
  remaining_part / julie_rate = 1

theorem potato_peeling_theorem :
  potato_peeling_problem (1/10) (1/8) 4 := by
  sorry

end NUMINAMATH_CALUDE_potato_peeling_theorem_l609_60947


namespace NUMINAMATH_CALUDE_bus_stop_walk_time_l609_60992

theorem bus_stop_walk_time (usual_time : ℝ) (usual_speed : ℝ) : 
  usual_speed > 0 →
  (2 / 3 * usual_speed) * (usual_time + 15) = usual_speed * usual_time →
  usual_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walk_time_l609_60992


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequence_problem_l609_60956

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Define the b_n sequence
def b_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := geometric_sequence a₁ q n + 2*n

-- Define the sum of the first n terms of b_n
def T_n (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  Finset.sum (Finset.range n) (λ i => b_sequence a₁ q (i+1))

theorem geometric_and_arithmetic_sequence_problem :
  ∀ a₁ q : ℝ,
  (a₁ > 0) →
  (q > 1) →
  (a₁ * (a₁*q) * (a₁*q^2) = 8) →
  (2*((a₁*q)+2) = (a₁+1) + ((a₁*q^2)+2)) →
  (a₁ = 1 ∧ q = 2) ∧
  (∀ n : ℕ, n > 0 → T_n 1 2 n = 2^n + n^2 + n - 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequence_problem_l609_60956


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l609_60933

theorem tic_tac_toe_tie_probability (max_win_prob zoe_win_prob : ℚ) :
  max_win_prob = 4/9 →
  zoe_win_prob = 5/12 →
  1 - (max_win_prob + zoe_win_prob) = 5/36 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l609_60933


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l609_60940

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  ((1.3 * l) * (1.2 * w) - l * w) / (l * w) = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l609_60940


namespace NUMINAMATH_CALUDE_a_range_theorem_l609_60946

-- Define the line equation
def line_equation (x y a : ℝ) : ℝ := x + y - a

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (line_equation 1 1 a) * (line_equation 2 (-1) a) < 0

-- Theorem statement
theorem a_range_theorem :
  ∀ a : ℝ, opposite_sides a ↔ a ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_a_range_theorem_l609_60946


namespace NUMINAMATH_CALUDE_apple_banana_cost_l609_60930

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 3 * a + 4 * b

/-- Theorem stating that the total cost of buying 3 kg of apples at 'a' yuan/kg
    and 4 kg of bananas at 'b' yuan/kg is (3a + 4b) yuan -/
theorem apple_banana_cost (a b : ℝ) :
  total_cost a b = 3 * a + 4 * b := by
  sorry

end NUMINAMATH_CALUDE_apple_banana_cost_l609_60930


namespace NUMINAMATH_CALUDE_inequality_solution_set_l609_60994

theorem inequality_solution_set (x : ℝ) :
  (3 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 8) ↔ (8/3 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l609_60994


namespace NUMINAMATH_CALUDE_principal_amount_l609_60934

/-- Proves that given the conditions of the problem, the principal amount must be 600 --/
theorem principal_amount (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 300 →
  P = 600 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l609_60934


namespace NUMINAMATH_CALUDE_property_satisfied_l609_60921

theorem property_satisfied (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_property_satisfied_l609_60921


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l609_60915

def manuscript_typing_cost (total_pages : ℕ) (initial_cost : ℕ) (revision_cost : ℕ) 
  (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  (total_pages * initial_cost) + 
  (pages_revised_once * revision_cost) + 
  (pages_revised_twice * revision_cost * 2)

theorem manuscript_cost_theorem :
  manuscript_typing_cost 100 5 4 30 20 = 780 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l609_60915


namespace NUMINAMATH_CALUDE_pure_imaginary_z_l609_60953

theorem pure_imaginary_z (a : ℝ) : 
  (∃ (b : ℝ), (1 - a * Complex.I) / (1 + a * Complex.I) = Complex.I * b) → 
  (a = 1 ∨ a = -1) := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_l609_60953


namespace NUMINAMATH_CALUDE_min_value_theorem_l609_60911

theorem min_value_theorem (a b c d : ℝ) (sum_constraint : a + b + c + d = 8) :
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d + c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c) ≥ 112 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l609_60911


namespace NUMINAMATH_CALUDE_impossible_to_tile_with_sphinx_l609_60957

/-- Represents a sphinx tile -/
structure SphinxTile :=
  (upward_triangles : Nat)
  (downward_triangles : Nat)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (side_length : Nat)

/-- Defines the properties of a sphinx tile -/
def is_valid_sphinx_tile (tile : SphinxTile) : Prop :=
  tile.upward_triangles + tile.downward_triangles = 6 ∧
  (tile.upward_triangles = 4 ∧ tile.downward_triangles = 2) ∨
  (tile.upward_triangles = 2 ∧ tile.downward_triangles = 4)

/-- Calculates the number of unit triangles in an equilateral triangle -/
def num_unit_triangles (triangle : EquilateralTriangle) : Nat :=
  triangle.side_length * (triangle.side_length + 1)

/-- Calculates the number of upward-pointing unit triangles -/
def num_upward_triangles (triangle : EquilateralTriangle) : Nat :=
  (triangle.side_length * (triangle.side_length - 1)) / 2

/-- Calculates the number of downward-pointing unit triangles -/
def num_downward_triangles (triangle : EquilateralTriangle) : Nat :=
  (triangle.side_length * (triangle.side_length + 1)) / 2

/-- Theorem stating the impossibility of tiling the triangle with sphinx tiles -/
theorem impossible_to_tile_with_sphinx (triangle : EquilateralTriangle) 
  (h1 : triangle.side_length = 6) : 
  ¬ ∃ (tiling : List SphinxTile), 
    (∀ tile ∈ tiling, is_valid_sphinx_tile tile) ∧ 
    (List.sum (tiling.map (λ tile => tile.upward_triangles)) = num_upward_triangles triangle) ∧
    (List.sum (tiling.map (λ tile => tile.downward_triangles)) = num_downward_triangles triangle) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_tile_with_sphinx_l609_60957


namespace NUMINAMATH_CALUDE_power_of_power_l609_60975

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l609_60975


namespace NUMINAMATH_CALUDE_remainder_theorem_l609_60908

theorem remainder_theorem (x : Int) (h : x % 285 = 31) :
  (x % 17 = 14) ∧ (x % 23 = 8) ∧ (x % 19 = 12) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l609_60908


namespace NUMINAMATH_CALUDE_extreme_value_implies_ab_eq_neg_three_l609_60935

/-- A function f(x) = ax³ + bx has an extreme value at x = 1/a -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => a * x^3 + b * x
  ∃ (h : ℝ), h = (1 : ℝ) / a ∧ (deriv f) h = 0

/-- If f(x) = ax³ + bx has an extreme value at x = 1/a, then ab = -3 -/
theorem extreme_value_implies_ab_eq_neg_three (a b : ℝ) (h : a ≠ 0) :
  has_extreme_value a b → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_ab_eq_neg_three_l609_60935


namespace NUMINAMATH_CALUDE_quadratic_inequality_equiv_interval_l609_60964

theorem quadratic_inequality_equiv_interval (x : ℝ) :
  x^2 - 8*x + 15 < 0 ↔ 3 < x ∧ x < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equiv_interval_l609_60964


namespace NUMINAMATH_CALUDE_sequence_fifth_b_l609_60950

/-- Given a sequence {aₙ}, where 2aₙ and aₙ₊₁ are the roots of x² - 3x + bₙ = 0,
    and a₁ = 2, prove that b₅ = -1054 -/
theorem sequence_fifth_b (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, (2 * a n) * (a (n + 1)) = b n) → 
  (∀ n, 2 * a n + a (n + 1) = 3) → 
  a 1 = 2 → 
  b 5 = -1054 :=
by sorry

end NUMINAMATH_CALUDE_sequence_fifth_b_l609_60950


namespace NUMINAMATH_CALUDE_concentric_circles_properties_l609_60945

/-- Two concentric circles with a width of 15 feet between them -/
structure ConcentricCircles where
  inner_diameter : ℝ
  width : ℝ
  width_is_15 : width = 15

theorem concentric_circles_properties (c : ConcentricCircles) :
  let outer_diameter := c.inner_diameter + 2 * c.width
  (π * outer_diameter - π * c.inner_diameter = 30 * π) ∧
  (π * (15 * c.inner_diameter + 225) = 
   π * ((outer_diameter / 2)^2 - (c.inner_diameter / 2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_properties_l609_60945


namespace NUMINAMATH_CALUDE_abc_product_l609_60949

theorem abc_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h1 : a + 1 = b + 2) (h2 : b + 2 = c + 3) :
  a * b * c = c * (c + 1) * (c + 2) := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l609_60949


namespace NUMINAMATH_CALUDE_sequence_b_is_geometric_progression_l609_60907

def sequence_a (a : ℝ) (n : ℕ) : ℝ := 
  if n = 1 then a else 3 * (4 ^ (n - 1)) + 2 * (a - 4) * (3 ^ (n - 2))

def sum_S (a : ℝ) (n : ℕ) : ℝ := 
  (4 ^ n) + (a - 4) * (3 ^ (n - 1))

def sequence_b (a : ℝ) (n : ℕ) : ℝ := 
  sum_S a n - (4 ^ n)

theorem sequence_b_is_geometric_progression (a : ℝ) (h : a ≠ 4) :
  ∀ n : ℕ, n ≥ 1 → sequence_b a (n + 1) = 3 * sequence_b a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_b_is_geometric_progression_l609_60907


namespace NUMINAMATH_CALUDE_cubic_sum_equals_linear_sum_l609_60974

theorem cubic_sum_equals_linear_sum (k : ℝ) : 
  (∀ r s : ℝ, 3 * r^2 + 6 * r + k = 0 ∧ 3 * s^2 + 6 * s + k = 0 → r^3 + s^3 = r + s) ↔ 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_linear_sum_l609_60974


namespace NUMINAMATH_CALUDE_f_expression_m_values_l609_60962

/-- A quadratic function satisfying certain properties -/
def f (x : ℝ) : ℝ := sorry

/-- The properties of the quadratic function -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x - 1
axiom f_zero : f 0 = 3

/-- The expression of f(x) -/
theorem f_expression (x : ℝ) : f x = x^2 - 2*x + 3 := sorry

/-- The function y in terms of x and m -/
def y (x m : ℝ) : ℝ := f (Real.log x / Real.log 3 + m)

/-- The set of x values -/
def X : Set ℝ := Set.Icc (1/3) 3

/-- The theorem about the values of m -/
theorem m_values :
  ∀ m : ℝ, (∀ x ∈ X, y x m ≥ 3) ∧ (∃ x ∈ X, y x m = 3) →
  m = -1 ∨ m = 3 := sorry

end NUMINAMATH_CALUDE_f_expression_m_values_l609_60962


namespace NUMINAMATH_CALUDE_max_value_abc_l609_60903

theorem max_value_abc (a b c : ℝ) (h : a + 3 * b + c = 5) :
  ∃ (max : ℝ), max = 25/3 ∧ ∀ (x y z : ℝ), x + 3 * y + z = 5 → x * y + x * z + y * z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l609_60903


namespace NUMINAMATH_CALUDE_lcm_problem_l609_60925

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l609_60925


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l609_60922

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.05

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The moon's speed in kilometers per hour -/
theorem moon_speed_km_per_hour :
  moon_speed_km_per_sec * seconds_per_hour = 3780 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l609_60922


namespace NUMINAMATH_CALUDE_complement_intersect_theorem_l609_60973

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 5}

theorem complement_intersect_theorem :
  (U \ B) ∩ A = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersect_theorem_l609_60973


namespace NUMINAMATH_CALUDE_increase_by_percentage_l609_60986

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 110 → percentage = 50 → final = initial * (1 + percentage / 100) → final = 165 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l609_60986


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l609_60920

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32 = 3 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2 = -7 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l609_60920


namespace NUMINAMATH_CALUDE_population_after_four_years_l609_60965

def population_after_n_years (initial_population : ℕ) (new_people : ℕ) (people_leaving : ℕ) (years : ℕ) : ℕ :=
  let population_after_changes := initial_population + new_people - people_leaving
  (population_after_changes / 2^years : ℕ)

theorem population_after_four_years :
  population_after_n_years 780 100 400 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_population_after_four_years_l609_60965


namespace NUMINAMATH_CALUDE_first_store_unload_percentage_l609_60978

def initial_load : ℝ := 50000
def second_store_percentage : ℝ := 0.20
def final_load : ℝ := 36000

theorem first_store_unload_percentage :
  ∃ x : ℝ, 
    x ≥ 0 ∧ x ≤ 1 ∧
    (1 - x) * initial_load * (1 - second_store_percentage) = final_load ∧
    x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_first_store_unload_percentage_l609_60978


namespace NUMINAMATH_CALUDE_police_hat_multiple_l609_60977

/-- Proves that the multiple of Fire Chief Simpson's hats that Policeman O'Brien had before he lost one is 2 -/
theorem police_hat_multiple :
  let simpson_hats : ℕ := 15
  let obrien_current_hats : ℕ := 34
  let obrien_previous_hats : ℕ := obrien_current_hats + 1
  ∃ x : ℕ, x * simpson_hats + 5 = obrien_previous_hats ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_police_hat_multiple_l609_60977


namespace NUMINAMATH_CALUDE_black_square_area_proof_l609_60966

/-- The edge length of the cube in feet -/
def cube_edge : ℝ := 12

/-- The total area covered by yellow paint in square feet -/
def yellow_area : ℝ := 432

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The area of the black square on each face of the cube in square feet -/
def black_square_area : ℝ := 72

theorem black_square_area_proof :
  let total_surface_area := num_faces * cube_edge ^ 2
  let yellow_area_per_face := yellow_area / num_faces
  black_square_area = cube_edge ^ 2 - yellow_area_per_face := by
  sorry

end NUMINAMATH_CALUDE_black_square_area_proof_l609_60966


namespace NUMINAMATH_CALUDE_quadratic_product_is_square_l609_60996

/-- Given quadratic trinomials f and g satisfying the inequality condition,
    prove that their product is the square of some quadratic trinomial. -/
theorem quadratic_product_is_square
  (f g : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h : ℝ, ∀ x, g x = d * x^2 + e * x + h)
  (h_ineq : ∀ x, (deriv f x) * (deriv g x) ≥ |f x| + |g x|) :
  ∃ (k : ℝ) (p : ℝ → ℝ),
    (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) ∧
    (∀ x, f x * g x = k * (p x)^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_product_is_square_l609_60996


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l609_60931

/-- The locus of points satisfying the given conditions is one branch of a hyperbola -/
theorem moving_circle_trajectory (M : ℝ × ℝ) :
  (∃ (x y : ℝ), M = (x, y) ∧ x > 0) →
  (Real.sqrt (M.1^2 + M.2^2) - Real.sqrt ((M.1 - 3)^2 + M.2^2) = 2) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ M.1^2 / a^2 - M.2^2 / b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l609_60931


namespace NUMINAMATH_CALUDE_dance_group_average_age_l609_60993

theorem dance_group_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_female_age : ℝ) 
  (avg_male_age : ℝ) 
  (h1 : num_females = 12)
  (h2 : avg_female_age = 25)
  (h3 : num_males = 18)
  (h4 : avg_male_age = 40)
  (h5 : num_females + num_males = 30) : 
  (num_females * avg_female_age + num_males * avg_male_age) / (num_females + num_males) = 34 := by
  sorry

end NUMINAMATH_CALUDE_dance_group_average_age_l609_60993


namespace NUMINAMATH_CALUDE_power_sum_zero_l609_60972

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l609_60972


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l609_60951

theorem no_solution_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |3 - x| ≥ 2*a + 1) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l609_60951


namespace NUMINAMATH_CALUDE_roots_of_equation_l609_60936

/-- The polynomial equation whose roots we want to find -/
def f (x : ℝ) : ℝ := (x^3 - 4*x^2 - x + 4)*(x-3)*(x+2)

/-- The set of roots we claim to be correct -/
def root_set : Set ℝ := {-2, -1, 1, 3, 4}

/-- Theorem stating that the roots of the equation are exactly the elements of root_set -/
theorem roots_of_equation : 
  ∀ x : ℝ, f x = 0 ↔ x ∈ root_set :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l609_60936


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l609_60958

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (x - 3) / x ≥ 0 ↔ x < 0 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l609_60958


namespace NUMINAMATH_CALUDE_simplify_expression_l609_60941

theorem simplify_expression (a b : ℝ) : 
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (-1/2 * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l609_60941


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l609_60961

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l609_60961


namespace NUMINAMATH_CALUDE_mike_spent_500_on_self_l609_60928

def total_rose_bushes : ℕ := 6
def price_per_rose_bush : ℕ := 75
def rose_bushes_for_friend : ℕ := 2
def tiger_tooth_aloes : ℕ := 2
def price_per_aloe : ℕ := 100

def money_spent_on_self : ℕ :=
  (total_rose_bushes - rose_bushes_for_friend) * price_per_rose_bush +
  tiger_tooth_aloes * price_per_aloe

theorem mike_spent_500_on_self :
  money_spent_on_self = 500 := by
  sorry

end NUMINAMATH_CALUDE_mike_spent_500_on_self_l609_60928


namespace NUMINAMATH_CALUDE_parade_formation_l609_60970

theorem parade_formation (total : Nat) (red_flower : Nat) (red_balloon : Nat) (yellow_green : Nat)
  (h1 : total = 100)
  (h2 : red_flower = 42)
  (h3 : red_balloon = 63)
  (h4 : yellow_green = 28) :
  total - red_balloon - yellow_green + red_flower = 33 := by
  sorry

end NUMINAMATH_CALUDE_parade_formation_l609_60970


namespace NUMINAMATH_CALUDE_fruit_seller_problem_l609_60997

/-- Represents the number of apples whose selling price equals the total gain -/
def reference_apples : ℕ := 50

/-- Represents the gain percent as a rational number -/
def gain_percent : ℚ := 100 / 3

/-- Calculates the number of apples sold given the reference apples and gain percent -/
def apples_sold (reference : ℕ) (gain : ℚ) : ℕ := sorry

theorem fruit_seller_problem :
  apples_sold reference_apples gain_percent = 200 := by sorry

end NUMINAMATH_CALUDE_fruit_seller_problem_l609_60997


namespace NUMINAMATH_CALUDE_roots_of_equation_l609_60942

theorem roots_of_equation (x y : ℝ) (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) :
  x^2 - 10*x + 19.24 = 0 ∧ y^2 - 10*y + 19.24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l609_60942


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l609_60927

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l609_60927


namespace NUMINAMATH_CALUDE_subtraction_multiplication_fractions_l609_60924

theorem subtraction_multiplication_fractions :
  (5 / 12 - 1 / 6) * (3 / 4) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_fractions_l609_60924


namespace NUMINAMATH_CALUDE_consecutive_sum_prime_iff_n_one_or_two_l609_60963

/-- The sum of n consecutive natural numbers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

/-- A natural number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem consecutive_sum_prime_iff_n_one_or_two :
  ∀ n : ℕ, (∃ k : ℕ, isPrime (consecutiveSum n k)) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_prime_iff_n_one_or_two_l609_60963


namespace NUMINAMATH_CALUDE_triangles_in_pentagon_l609_60932

/-- The number of triangles formed when all diagonals are drawn in a pentagon -/
def num_triangles_in_pentagon : ℕ := 35

/-- Theorem stating that the number of triangles in a fully connected pentagon is 35 -/
theorem triangles_in_pentagon :
  num_triangles_in_pentagon = 35 := by
  sorry

#check triangles_in_pentagon

end NUMINAMATH_CALUDE_triangles_in_pentagon_l609_60932


namespace NUMINAMATH_CALUDE_skipping_odometer_conversion_l609_60976

/-- Represents an odometer that skips digits 3 and 4 --/
def SkippingOdometer := Nat → Nat

/-- Converts a regular number to its representation on the skipping odometer --/
def toSkippingOdometer : Nat → Nat :=
  sorry

/-- Converts a number from the skipping odometer to its actual value --/
def fromSkippingOdometer : Nat → Nat :=
  sorry

theorem skipping_odometer_conversion :
  ∃ (odo : SkippingOdometer),
    (toSkippingOdometer 1029 = 002006) ∧
    (fromSkippingOdometer 002006 = 1029) := by
  sorry

end NUMINAMATH_CALUDE_skipping_odometer_conversion_l609_60976


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l609_60904

theorem trig_product_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l609_60904


namespace NUMINAMATH_CALUDE_new_oarsman_weight_l609_60968

theorem new_oarsman_weight (n : ℕ) (old_weight average_increase : ℝ) :
  n = 10 ∧ old_weight = 53 ∧ average_increase = 1.8 →
  ∃ new_weight : ℝ,
    new_weight = old_weight + n * average_increase ∧
    new_weight = 71 := by
  sorry

end NUMINAMATH_CALUDE_new_oarsman_weight_l609_60968


namespace NUMINAMATH_CALUDE_partition_positive_integers_l609_60948

def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  y - x = z - y ∧ x < y ∧ y < z

def has_infinite_arithmetic_subsequence (S : Set ℕ) : Prop :=
  ∃ (a d : ℕ), d ≠ 0 ∧ ∀ n : ℕ, (a + n * d) ∈ S

theorem partition_positive_integers :
  ∃ (A B : Set ℕ),
    (A ∪ B = {n : ℕ | n > 0}) ∧
    (A ∩ B = ∅) ∧
    (∀ x y z : ℕ, x ∈ A → y ∈ A → z ∈ A → x ≠ y → y ≠ z → x ≠ z →
      ¬is_arithmetic_sequence x y z) ∧
    ¬has_infinite_arithmetic_subsequence B :=
by sorry

end NUMINAMATH_CALUDE_partition_positive_integers_l609_60948


namespace NUMINAMATH_CALUDE_pta_fundraiser_remaining_money_l609_60989

theorem pta_fundraiser_remaining_money (initial_amount : ℝ) : 
  initial_amount = 400 → 
  (initial_amount - initial_amount / 4) / 2 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_pta_fundraiser_remaining_money_l609_60989


namespace NUMINAMATH_CALUDE_min_value_a_l609_60926

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1 / (x^2 + 1)) ≤ (a / x)) → 
  a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l609_60926


namespace NUMINAMATH_CALUDE_hair_cut_calculation_l609_60923

/-- Given the total amount of hair cut and the amount cut on the first day,
    calculate the amount cut on the second day. -/
theorem hair_cut_calculation (total : ℝ) (first_day : ℝ) (h1 : total = 0.88) (h2 : first_day = 0.38) :
  total - first_day = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_calculation_l609_60923


namespace NUMINAMATH_CALUDE_r_exceeds_s_by_two_l609_60913

theorem r_exceeds_s_by_two (x y r s : ℝ) : 
  3 * x + 2 * y = 16 →
  5 * x + 3 * y = 26 →
  r = x →
  s = y →
  r - s = 2 := by
sorry

end NUMINAMATH_CALUDE_r_exceeds_s_by_two_l609_60913


namespace NUMINAMATH_CALUDE_emily_egg_collection_l609_60983

theorem emily_egg_collection (baskets : ℕ) (eggs_per_basket : ℕ) 
  (h1 : baskets = 303) (h2 : eggs_per_basket = 28) : 
  baskets * eggs_per_basket = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l609_60983


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l609_60939

theorem similar_triangles_problem (A₁ A₂ : ℕ) (k : ℕ) (s : ℝ) :
  A₁ > A₂ →
  A₁ - A₂ = 18 →
  A₁ = k^2 * A₂ →
  s = 3 →
  (∃ (a b c : ℝ), A₂ = (a * b) / 2 ∧ c^2 = a^2 + b^2 ∧ s = c) →
  (∃ (a' b' c' : ℝ), A₁ = (a' * b') / 2 ∧ c'^2 = a'^2 + b'^2 ∧ 6 = c') :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l609_60939


namespace NUMINAMATH_CALUDE_cole_trip_time_l609_60905

/-- Proves that given a round trip where the outbound journey is at 75 km/h,
    the return journey is at 105 km/h, and the total trip time is 4 hours,
    the time taken for the outbound journey is 140 minutes. -/
theorem cole_trip_time (distance : ℝ) :
  distance / 75 + distance / 105 = 4 →
  distance / 75 * 60 = 140 := by
sorry

end NUMINAMATH_CALUDE_cole_trip_time_l609_60905


namespace NUMINAMATH_CALUDE_product_remainder_l609_60900

theorem product_remainder (a b m : ℕ) (ha : a % m = 7) (hb : b % m = 1) (hm : m = 8) :
  (a * b) % m = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l609_60900


namespace NUMINAMATH_CALUDE_company_sales_royalties_l609_60982

/-- A company's sales and royalties problem -/
theorem company_sales_royalties
  (initial_sales : ℝ)
  (initial_royalties : ℝ)
  (next_royalties : ℝ)
  (royalty_ratio_decrease : ℝ)
  (h1 : initial_sales = 10000000)
  (h2 : initial_royalties = 2000000)
  (h3 : next_royalties = 8000000)
  (h4 : royalty_ratio_decrease = 0.6)
  : ∃ (next_sales : ℝ), next_sales = 100000000 ∧
    next_royalties / next_sales = (initial_royalties / initial_sales) * (1 - royalty_ratio_decrease) :=
by sorry

end NUMINAMATH_CALUDE_company_sales_royalties_l609_60982


namespace NUMINAMATH_CALUDE_max_remainder_is_456_l609_60954

/-- The maximum number on the board initially -/
def max_initial : ℕ := 2012

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of operations performed -/
def num_operations : ℕ := max_initial - 1

/-- The final number N after all operations -/
def final_number : ℕ := sum_to_n max_initial * 2^num_operations

theorem max_remainder_is_456 : final_number % 1000 = 456 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_is_456_l609_60954


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l609_60991

def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l609_60991


namespace NUMINAMATH_CALUDE_carla_laundry_rate_l609_60929

/-- Given a total number of laundry pieces and available hours, 
    calculate the number of pieces to be cleaned per hour. -/
def piecesPerHour (totalPieces : ℕ) (availableHours : ℕ) : ℕ :=
  totalPieces / availableHours

theorem carla_laundry_rate :
  piecesPerHour 80 4 = 20 := by
  sorry


end NUMINAMATH_CALUDE_carla_laundry_rate_l609_60929


namespace NUMINAMATH_CALUDE_puzzle_solution_l609_60984

theorem puzzle_solution (a b : ℕ) 
  (sum_eq : a + b = 24581)
  (b_div_12 : ∃ k : ℕ, b = 12 * k)
  (a_times_10 : a * 10 = b) :
  b - a = 20801 := by
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l609_60984


namespace NUMINAMATH_CALUDE_pentomino_tiling_l609_60987

/-- A pentomino is a shape that covers exactly 5 squares. -/
def Pentomino : Type := Unit

/-- A rectangle of size 5 × m. -/
structure Rectangle (m : ℕ) :=
  (width : Fin 5)
  (height : Fin m)

/-- Predicate to determine if a rectangle can be tiled by a pentomino. -/
def IsTileable (m : ℕ) : Prop := sorry

theorem pentomino_tiling (m : ℕ) : 
  IsTileable m ↔ Even m := by sorry

end NUMINAMATH_CALUDE_pentomino_tiling_l609_60987


namespace NUMINAMATH_CALUDE_cos_330_degrees_l609_60999

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l609_60999


namespace NUMINAMATH_CALUDE_john_mary_probability_l609_60906

-- Define the set of people
inductive Person : Type
| John : Person
| Mary : Person
| Alice : Person
| Bob : Person
| Clara : Person

-- Define the seating arrangement
structure Seating :=
(long_side1 : Person × Person)
(long_side2 : Person × Person)
(short_side1 : Person)
(short_side2 : Person)

-- Define a function to check if John and Mary are seated together on a longer side
def john_and_mary_together (s : Seating) : Prop :=
  (s.long_side1 = (Person.John, Person.Mary) ∨ s.long_side1 = (Person.Mary, Person.John)) ∨
  (s.long_side2 = (Person.John, Person.Mary) ∨ s.long_side2 = (Person.Mary, Person.John))

-- Define the set of all possible seating arrangements
def all_seatings : Set Seating := sorry

-- Define the probability measure on the set of all seating arrangements
def prob : Set Seating → ℝ := sorry

-- The main theorem
theorem john_mary_probability :
  prob {s ∈ all_seatings | john_and_mary_together s} = 1/4 := by sorry

end NUMINAMATH_CALUDE_john_mary_probability_l609_60906


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l609_60912

/-- The repeating decimal 0.̅5̅6̅ is equal to the fraction 56/99 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ = 0.56) ∧ x = 56/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l609_60912


namespace NUMINAMATH_CALUDE_box_volume_conversion_l609_60916

/-- Converts cubic feet to cubic yards -/
def cubic_feet_to_cubic_yards (cubic_feet : ℚ) : ℚ :=
  cubic_feet / 27

theorem box_volume_conversion :
  let box_volume_cubic_feet : ℚ := 200
  let box_volume_cubic_yards : ℚ := cubic_feet_to_cubic_yards box_volume_cubic_feet
  box_volume_cubic_yards = 200 / 27 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_conversion_l609_60916


namespace NUMINAMATH_CALUDE_range_of_Z_l609_60969

theorem range_of_Z (a b : ℝ) (h : a^2 + 3*a*b + 9*b^2 = 4) :
  ∃ (z : ℝ), z = a^2 + 9*b^2 ∧ 8/3 ≤ z ∧ z ≤ 8 ∧
  (∀ (w : ℝ), w = a^2 + 9*b^2 → 8/3 ≤ w ∧ w ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_Z_l609_60969


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l609_60985

theorem half_angle_quadrant (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) →
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l609_60985


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l609_60937

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l609_60937


namespace NUMINAMATH_CALUDE_calorie_calculation_l609_60914

/-- Represents the daily calorie allowance for a certain age group -/
def average_daily_allowance : ℕ := 2000

/-- The number of calories to reduce daily to hypothetically live to 100 years -/
def calorie_reduction : ℕ := 500

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The allowed weekly calorie intake for the age group -/
def allowed_weekly_intake : ℕ := 10500

theorem calorie_calculation :
  (average_daily_allowance - calorie_reduction) * days_in_week = allowed_weekly_intake := by
  sorry

end NUMINAMATH_CALUDE_calorie_calculation_l609_60914


namespace NUMINAMATH_CALUDE_orange_slices_problem_l609_60901

/-- The number of additional slices needed to fill the last container -/
def additional_slices_needed (total_slices : ℕ) (container_capacity : ℕ) : ℕ :=
  container_capacity - (total_slices % container_capacity)

/-- Theorem stating that given 329 slices and a container capacity of 4,
    3 additional slices are needed to fill the last container -/
theorem orange_slices_problem :
  additional_slices_needed 329 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_slices_problem_l609_60901


namespace NUMINAMATH_CALUDE_equation_solutions_count_l609_60909

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 7)^2 = 49) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l609_60909


namespace NUMINAMATH_CALUDE_inscribed_squares_perimeter_ratio_l609_60918

theorem inscribed_squares_perimeter_ratio :
  let r : ℝ := 5
  let s₁ : ℝ := Real.sqrt ((2 * r^2) / 5)  -- side length of square in semicircle
  let s₂ : ℝ := r * Real.sqrt 2           -- side length of square in circle
  (4 * s₁) / (4 * s₂) = Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_perimeter_ratio_l609_60918


namespace NUMINAMATH_CALUDE_two_real_roots_iff_nonneg_discriminant_quadratic_always_two_real_roots_l609_60980

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has two real roots if and only if its discriminant is non-negative -/
theorem two_real_roots_iff_nonneg_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x y : ℝ, a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 ∧ x ≠ y ↔ discriminant a b c ≥ 0 :=
sorry

theorem quadratic_always_two_real_roots (k : ℝ) :
  discriminant 1 (-(k+4)) (4*k) ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_two_real_roots_iff_nonneg_discriminant_quadratic_always_two_real_roots_l609_60980


namespace NUMINAMATH_CALUDE_equation_solution_l609_60960

theorem equation_solution : ∃ a : ℝ, -6 * a^2 = 3 * (4 * a + 2) ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l609_60960


namespace NUMINAMATH_CALUDE_s_five_value_l609_60979

theorem s_five_value (x : ℝ) (h : x + 1/x = 4) : x^5 + 1/x^5 = 724 := by
  sorry

end NUMINAMATH_CALUDE_s_five_value_l609_60979


namespace NUMINAMATH_CALUDE_number_is_composite_l609_60902

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the number formed by the given sequence of digits -/
def formNumber (digits : List Digit) : ℕ :=
  sorry

/-- Checks if a natural number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The main theorem to be proved -/
theorem number_is_composite (digits : List Digit) :
  isComposite (formNumber digits) :=
sorry

end NUMINAMATH_CALUDE_number_is_composite_l609_60902


namespace NUMINAMATH_CALUDE_ilya_incorrect_l609_60917

theorem ilya_incorrect : ¬∃ (s t : ℝ), s + t = s * t ∧ s + t = s / t := by
  sorry

end NUMINAMATH_CALUDE_ilya_incorrect_l609_60917


namespace NUMINAMATH_CALUDE_class_election_is_survey_conduction_l609_60938

/-- Represents the steps in a survey process -/
inductive SurveyStep
  | DetermineObject
  | SelectMethod
  | Conduct
  | DrawConclusions

/-- Represents a voting process in a class election -/
structure ClassElection where
  students : Set Student
  candidates : Set Candidate
  ballot_box : Set Vote

/-- Definition of conducting a survey -/
def conducSurvey (process : ClassElection) : SurveyStep :=
  SurveyStep.Conduct

theorem class_election_is_survey_conduction (election : ClassElection) :
  conducSurvey election = SurveyStep.Conduct := by
  sorry

#check class_election_is_survey_conduction

end NUMINAMATH_CALUDE_class_election_is_survey_conduction_l609_60938


namespace NUMINAMATH_CALUDE_article_cost_price_l609_60981

theorem article_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (price_increase : ℝ) 
  (h1 : loss_percentage = 25)
  (h2 : gain_percentage = 15)
  (h3 : price_increase = 500) : 
  ∃ (cost_price : ℝ), 
    cost_price * (1 - loss_percentage / 100) + price_increase = cost_price * (1 + gain_percentage / 100) ∧ 
    cost_price = 1250 := by
  sorry

#check article_cost_price

end NUMINAMATH_CALUDE_article_cost_price_l609_60981


namespace NUMINAMATH_CALUDE_square_equation_solution_l609_60910

theorem square_equation_solution : ∃ (M : ℕ), M > 0 ∧ 33^2 * 66^2 = 15^2 * M^2 ∧ M = 726 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l609_60910


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l609_60943

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum of the first n terms
  first_term : a 1 = -7
  third_sum : s 3 = -15

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 9) ∧
  (∀ n : ℕ, seq.s n = (n - 4)^2 - 16) ∧
  (∀ n : ℕ, seq.s n ≥ -16) ∧
  seq.s 4 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l609_60943


namespace NUMINAMATH_CALUDE_sum_of_bases_equality_l609_60959

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equality : 
  base13ToBase10 372 + base14ToBase10 (4 * 14^2 + C * 14 + 5) = 1557 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equality_l609_60959


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l609_60944

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 3 : ℂ) + (2 / 5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 3 : ℂ) - (2 / 5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l609_60944


namespace NUMINAMATH_CALUDE_sum_of_operation_l609_60955

def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {1, 2}

def operation (A B : Finset ℕ) : Finset ℕ :=
  Finset.image (λ (x : ℕ × ℕ) => x.1 * x.2) (A.product B)

theorem sum_of_operation :
  (operation A B).sum id = 31 := by sorry

end NUMINAMATH_CALUDE_sum_of_operation_l609_60955


namespace NUMINAMATH_CALUDE_lyndees_chicken_pieces_l609_60990

/-- Given the total number of chicken pieces, the number of friends, and the number of pieces each friend ate,
    calculate the number of pieces Lyndee ate. -/
theorem lyndees_chicken_pieces (total_pieces friends_pieces friends : ℕ) : 
  total_pieces - (friends_pieces * friends) = total_pieces - (friends_pieces * friends) := by
  sorry

#check lyndees_chicken_pieces 11 2 5

end NUMINAMATH_CALUDE_lyndees_chicken_pieces_l609_60990


namespace NUMINAMATH_CALUDE_existence_of_x_with_abs_f_ge_2_l609_60988

theorem existence_of_x_with_abs_f_ge_2 (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc 1 9, |a * x₀ + b + 9 / x₀| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_with_abs_f_ge_2_l609_60988


namespace NUMINAMATH_CALUDE_competition_probabilities_l609_60995

-- Define the possible grades
inductive Grade : Type
  | Qualified
  | Good
  | Excellent

-- Define the probabilities for each participant
def probA : Grade → ℝ
  | Grade.Qualified => 0.6
  | Grade.Good => 0.3
  | Grade.Excellent => 0.1

def probB : Grade → ℝ
  | Grade.Qualified => 0.4
  | Grade.Good => 0.4
  | Grade.Excellent => 0.2

-- Define a function to check if one grade is higher than another
def isHigher : Grade → Grade → Bool
  | Grade.Excellent, Grade.Excellent => false
  | Grade.Excellent, _ => true
  | Grade.Good, Grade.Excellent => false
  | Grade.Good, _ => true
  | Grade.Qualified, Grade.Qualified => false
  | Grade.Qualified, _ => false

-- Define the probability that A's grade is higher than B's in one round
def probAHigherThanB : ℝ := 0.2

-- Define the probability that A's grade is higher than B's in at least two out of three rounds
def probAHigherThanBTwiceInThree : ℝ := 0.104

theorem competition_probabilities :
  (probAHigherThanB = 0.2) ∧
  (probAHigherThanBTwiceInThree = 0.104) := by
  sorry


end NUMINAMATH_CALUDE_competition_probabilities_l609_60995


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l609_60919

def complex_number (z : ℂ) : Prop :=
  z = (3 + Complex.I) / (1 - Complex.I)

theorem z_in_first_quadrant (z : ℂ) (h : complex_number z) :
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l609_60919


namespace NUMINAMATH_CALUDE_jaron_snickers_needed_l609_60967

/-- The number of Snickers bars Jaron needs to sell to win the Nintendo Switch -/
def snickers_needed (total_points_needed : ℕ) (bunnies_sold : ℕ) (points_per_bunny : ℕ) (points_per_snickers : ℕ) : ℕ :=
  ((total_points_needed - bunnies_sold * points_per_bunny) + points_per_snickers - 1) / points_per_snickers

theorem jaron_snickers_needed :
  snickers_needed 2000 8 100 25 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jaron_snickers_needed_l609_60967


namespace NUMINAMATH_CALUDE_circle_equation_simplified_fixed_point_satisfies_line_main_theorem_l609_60998

/-- The fixed point P through which all lines pass -/
def P : ℝ × ℝ := (2, -1)

/-- The radius of the circle -/
def r : ℝ := 2

/-- The line equation that passes through P for all a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop :=
  (1 - a) * x + y + 2 * a - 1 = 0

/-- The circle equation with center P and radius r -/
def circle_equation (x y : ℝ) : Prop :=
  (x - P.1)^2 + (y - P.2)^2 = r^2

theorem circle_equation_simplified :
  ∀ x y : ℝ, circle_equation x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

theorem fixed_point_satisfies_line :
  ∀ a : ℝ, line_equation a P.1 P.2 :=
by sorry

theorem main_theorem :
  ∀ x y : ℝ, circle_equation x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_simplified_fixed_point_satisfies_line_main_theorem_l609_60998


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l609_60971

/-- Given real numbers x, y, and z satisfying the equation (z-x)^2 - 4(x-y)(y-z) = 0,
    prove that 2y = x + z, which implies that x, y, and z form an arithmetic sequence. -/
theorem arithmetic_sequence (x y z : ℝ) (h : (z - x)^2 - 4*(x - y)*(y - z) = 0) :
  2*y = x + z := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l609_60971


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_invariant_l609_60952

theorem consecutive_numbers_product_invariant :
  ∃ (a : ℕ), 
    let original := [a, a+1, a+2, a+3, a+4, a+5, a+6]
    ∃ (modified : List ℕ),
      (∀ i, i ∈ original → ∃ j, j ∈ modified ∧ (j = i - 1 ∨ j = i ∨ j = i + 1)) ∧
      (modified.length = 7) ∧
      (original.prod = modified.prod) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_invariant_l609_60952
