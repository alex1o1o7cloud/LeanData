import Mathlib

namespace NUMINAMATH_CALUDE_parabolic_archway_height_l2695_269593

/-- Represents a parabolic function of the form f(x) = ax² + 20 -/
def parabolic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 20

theorem parabolic_archway_height :
  ∃ a : ℝ, 
    (parabolic_function a 25 = 0) ∧ 
    (parabolic_function a 0 = 20) ∧
    (parabolic_function a 10 = 16.8) := by
  sorry

end NUMINAMATH_CALUDE_parabolic_archway_height_l2695_269593


namespace NUMINAMATH_CALUDE_rectangle_fourth_vertex_l2695_269505

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a rectangle by its four vertices
structure Rectangle where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define the property of being a rectangle
def isRectangle (rect : Rectangle) : Prop :=
  -- Add properties that define a rectangle, such as perpendicular sides and equal diagonals
  sorry

-- Theorem statement
theorem rectangle_fourth_vertex 
  (rect : Rectangle)
  (h1 : isRectangle rect)
  (h2 : rect.A = ⟨1, 1⟩)
  (h3 : rect.B = ⟨3, 1⟩)
  (h4 : rect.C = ⟨3, 5⟩) :
  rect.D = ⟨1, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_rectangle_fourth_vertex_l2695_269505


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2695_269536

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) ≥ (x*y + y*z + z*x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2695_269536


namespace NUMINAMATH_CALUDE_trig_identity_l2695_269518

theorem trig_identity (x : Real) (h : Real.sin x - 2 * Real.cos x = 0) :
  2 * Real.sin x ^ 2 + Real.cos x ^ 2 + 1 = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2695_269518


namespace NUMINAMATH_CALUDE_pet_store_profit_l2695_269576

/-- The profit calculation for a pet store reselling geckos --/
theorem pet_store_profit (brandon_price : ℕ) (pet_store_markup : ℕ → ℕ) : 
  brandon_price = 100 → 
  (∀ x, pet_store_markup x = 3 * x + 5) →
  pet_store_markup brandon_price - brandon_price = 205 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_profit_l2695_269576


namespace NUMINAMATH_CALUDE_price_change_percentage_l2695_269539

theorem price_change_percentage (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.84 * P → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_change_percentage_l2695_269539


namespace NUMINAMATH_CALUDE_turkey_weight_ratio_l2695_269562

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The total amount spent on all turkeys in dollars -/
def total_spent : ℝ := 66

/-- The number of turkeys bought -/
def num_turkeys : ℕ := 3

theorem turkey_weight_ratio :
  let total_weight := total_spent / cost_per_kg
  let third_turkey_weight := total_weight - (first_turkey_weight + second_turkey_weight)
  third_turkey_weight / second_turkey_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_turkey_weight_ratio_l2695_269562


namespace NUMINAMATH_CALUDE_vector_orthogonality_l2695_269566

def a (k : ℝ) : ℝ × ℝ := (k, 3)
def b : ℝ × ℝ := (1, 4)
def c : ℝ × ℝ := (2, 1)

theorem vector_orthogonality (k : ℝ) :
  (2 • a k - 3 • b) • c = 0 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l2695_269566


namespace NUMINAMATH_CALUDE_parabola_focus_parameter_l2695_269525

/-- Given a parabola with equation x^2 = 2py and focus at (0, 2), prove that p = 4 -/
theorem parabola_focus_parameter : ∀ p : ℝ, 
  (∀ x y : ℝ, x^2 = 2*p*y) →  -- parabola equation
  (0, 2) = (0, p/2) →        -- focus coordinates
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_parameter_l2695_269525


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2695_269589

theorem smallest_winning_number : ∃ (N : ℕ), 
  (0 ≤ N ∧ N ≤ 1999) ∧ 
  (∃ (k : ℕ), 1900 ≤ 2 * N + 100 * k ∧ 2 * N + 100 * k ≤ 1999) ∧
  (∀ (M : ℕ), M < N → ¬∃ (j : ℕ), 1900 ≤ 2 * M + 100 * j ∧ 2 * M + 100 * j ≤ 1999) ∧
  N = 800 := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2695_269589


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l2695_269565

def M : Set (ℝ → ℝ) :=
  {P | ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d ∧ ∀ x ∈ Set.Icc (-1) 1, |P x| ≤ 1}

theorem polynomial_coefficient_bound :
  ∃ k : ℝ, k = 4 ∧ (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| ≤ k) ∧
  ∀ k' : ℝ, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| ≤ k') → k' ≥ k :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l2695_269565


namespace NUMINAMATH_CALUDE_range_of_3a_plus_2b_l2695_269530

theorem range_of_3a_plus_2b (a b : ℝ) (h : a^2 + b^2 = 4) :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) ∧ 
  x = 3*a + 2*b ∧ 
  ∀ (y : ℝ), y = 3*a + 2*b → y ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_plus_2b_l2695_269530


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l2695_269575

/-- Given two lines that intersect at (3,6), prove that the sum of their y-intercepts is 6 -/
theorem intersection_y_intercept_sum (c d : ℝ) : 
  (3 = (1/3) * 6 + c) →   -- First line passes through (3,6)
  (6 = (1/3) * 3 + d) →   -- Second line passes through (3,6)
  c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l2695_269575


namespace NUMINAMATH_CALUDE_digital_root_prime_probability_l2695_269549

/-- The digital root of a positive integer -/
def digitalRoot (n : ℕ+) : ℕ :=
  if n.val % 9 = 0 then 9 else n.val % 9

/-- Whether a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The count of numbers with prime digital roots in the first n positive integers -/
def countPrimeDigitalRoots (n : ℕ) : ℕ := sorry

theorem digital_root_prime_probability :
  (countPrimeDigitalRoots 1000 : ℚ) / 1000 = 444 / 1000 := by sorry

end NUMINAMATH_CALUDE_digital_root_prime_probability_l2695_269549


namespace NUMINAMATH_CALUDE_game_C_more_likely_than_D_l2695_269548

/-- Probability of getting heads when tossing the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails when tossing the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game C -/
def p_win_C : ℚ := p_heads^4

/-- Probability of winning Game D -/
def p_win_D : ℚ := p_heads^5 + p_heads^3 * p_tails^2 + p_tails^3 * p_heads^2 + p_tails^5

theorem game_C_more_likely_than_D : p_win_C - p_win_D = 11/256 := by
  sorry

end NUMINAMATH_CALUDE_game_C_more_likely_than_D_l2695_269548


namespace NUMINAMATH_CALUDE_inequality_proof_l2695_269588

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 0 < a / b ∧ a / b < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2695_269588


namespace NUMINAMATH_CALUDE_number_problem_l2695_269512

theorem number_problem (x : ℚ) : (3 / 4 : ℚ) * x = x - 19 → x = 76 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2695_269512


namespace NUMINAMATH_CALUDE_triangle_configuration_theorem_l2695_269535

/-- A configuration of wire triangles in space. -/
structure TriangleConfiguration where
  /-- The number of wire triangles. -/
  k : ℕ
  /-- The number of triangles converging at each vertex. -/
  p : ℕ
  /-- Each pair of triangles has exactly one common vertex. -/
  one_common_vertex : True
  /-- At each vertex, the same number p of triangles converge. -/
  p_triangles_at_vertex : True

/-- The theorem stating the possible configurations of wire triangles. -/
theorem triangle_configuration_theorem (config : TriangleConfiguration) :
  (config.k = 1 ∧ config.p = 1) ∨ (config.k = 4 ∧ config.p = 2) ∨ (config.k = 7 ∧ config.p = 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_configuration_theorem_l2695_269535


namespace NUMINAMATH_CALUDE_pencil_case_problem_l2695_269594

theorem pencil_case_problem (total : ℕ) (difference : ℕ) (erasers : ℕ) : 
  total = 240 →
  difference = 2 →
  erasers + (erasers - difference) = total →
  erasers = 121 := by
  sorry

end NUMINAMATH_CALUDE_pencil_case_problem_l2695_269594


namespace NUMINAMATH_CALUDE_infinite_geometric_sequence_formula_l2695_269592

/-- An infinite geometric sequence satisfying given conditions -/
def InfiniteGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∑' n, a n) = 3 ∧ (∑' n, (a n)^2) = 9/2

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2 * (1/3)^(n-1)

/-- Theorem stating that the general formula holds for the given infinite geometric sequence -/
theorem infinite_geometric_sequence_formula 
    (a : ℕ → ℝ) (h : InfiniteGeometricSequence a) : GeneralFormula a := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_sequence_formula_l2695_269592


namespace NUMINAMATH_CALUDE_no_common_terms_l2695_269557

-- Define the sequences a_n and b_n
def a_n (a : ℝ) (n : ℕ) : ℝ := a * n + 2
def b_n (b : ℝ) (n : ℕ) : ℝ := b * n + 1

-- Theorem statement
theorem no_common_terms (a b : ℝ) (h : a > b) :
  ∀ n : ℕ, a_n a n ≠ b_n b n := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_l2695_269557


namespace NUMINAMATH_CALUDE_second_derivative_y_l2695_269516

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_y (x : ℝ) :
  (deriv (deriv y)) x = 2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x^2) / (1 + Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_y_l2695_269516


namespace NUMINAMATH_CALUDE_parallelogram_centers_coincide_l2695_269567

-- Define a parallelogram
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

-- Define a point on a line segment
def PointOnSegment (V : Type*) [AddCommGroup V] [Module ℝ V] (A B P : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the center of a parallelogram
def CenterOfParallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] (p : Parallelogram V) : V :=
  (1/2) • (p.A + p.C)

-- State the theorem
theorem parallelogram_centers_coincide
  (V : Type*) [AddCommGroup V] [Module ℝ V]
  (p₁ p₂ : Parallelogram V)
  (h₁ : PointOnSegment V p₁.A p₁.B p₂.A)
  (h₂ : PointOnSegment V p₁.B p₁.C p₂.B)
  (h₃ : PointOnSegment V p₁.C p₁.D p₂.C)
  (h₄ : PointOnSegment V p₁.D p₁.A p₂.D) :
  CenterOfParallelogram V p₁ = CenterOfParallelogram V p₂ :=
sorry

end NUMINAMATH_CALUDE_parallelogram_centers_coincide_l2695_269567


namespace NUMINAMATH_CALUDE_odd_plus_even_combination_l2695_269504

theorem odd_plus_even_combination (p q : ℤ) 
  (h_p : ∃ k, p = 2 * k + 1) 
  (h_q : ∃ m, q = 2 * m) : 
  ∃ n, 3 * p + 2 * q = 2 * n + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_plus_even_combination_l2695_269504


namespace NUMINAMATH_CALUDE_find_divisor_l2695_269581

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 52 →
  quotient = 16 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2695_269581


namespace NUMINAMATH_CALUDE_ratio_sum_squares_l2695_269538

theorem ratio_sum_squares (a b c : ℝ) : 
  b = 2 * a ∧ c = 3 * a ∧ a^2 + b^2 + c^2 = 2016 → a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_l2695_269538


namespace NUMINAMATH_CALUDE_simplify_fraction_l2695_269587

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) :
  ((a + 1) / (a - 1) + 1) / (2 * a / (a^2 - 1)) = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2695_269587


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2695_269584

theorem negative_fraction_comparison : -2/3 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2695_269584


namespace NUMINAMATH_CALUDE_cards_distribution_l2695_269585

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  num_people - (total_cards % num_people) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2695_269585


namespace NUMINAMATH_CALUDE_city_population_l2695_269510

theorem city_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 85 / 100 →
  partial_population = 85000 →
  (percentage * total_population : ℚ) = partial_population →
  total_population = 100000 := by
sorry

end NUMINAMATH_CALUDE_city_population_l2695_269510


namespace NUMINAMATH_CALUDE_third_term_coefficient_a_plus_b_10_l2695_269546

def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem third_term_coefficient_a_plus_b_10 :
  binomial_coefficient 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_third_term_coefficient_a_plus_b_10_l2695_269546


namespace NUMINAMATH_CALUDE_average_weight_a_b_l2695_269528

-- Define the weights of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (b + c) / 2 = 45)
variable (h3 : b = 35)

-- Theorem statement
theorem average_weight_a_b : (a + b) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_a_b_l2695_269528


namespace NUMINAMATH_CALUDE_solve_for_m_l2695_269520

-- Define the equation
def is_quadratic (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, (m + 1) * x^(|m| + 1) + 6 * m * x - 2 = a * x^2 + b * x + c

-- Theorem statement
theorem solve_for_m :
  ∀ m : ℝ, is_quadratic m ∧ m + 1 ≠ 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2695_269520


namespace NUMINAMATH_CALUDE_equation_solution_l2695_269507

theorem equation_solution :
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2695_269507


namespace NUMINAMATH_CALUDE_neighbor_oranges_correct_l2695_269500

/-- The number of kilograms of oranges added for the neighbor -/
def neighbor_oranges : ℕ := 25

/-- The initial purchase of oranges in kilograms -/
def initial_purchase : ℕ := 10

/-- The total quantity of oranges bought over three weeks in kilograms -/
def total_quantity : ℕ := 75

/-- The quantity of oranges bought in each of the next two weeks -/
def next_weeks_purchase : ℕ := 2 * initial_purchase

theorem neighbor_oranges_correct :
  (initial_purchase + neighbor_oranges) + next_weeks_purchase + next_weeks_purchase = total_quantity :=
by sorry

end NUMINAMATH_CALUDE_neighbor_oranges_correct_l2695_269500


namespace NUMINAMATH_CALUDE_total_players_is_51_l2695_269568

/-- The number of cricket players -/
def cricket_players : ℕ := 10

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 16

/-- The number of softball players -/
def softball_players : ℕ := 13

/-- Theorem stating that the total number of players is 51 -/
theorem total_players_is_51 :
  cricket_players + hockey_players + football_players + softball_players = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_players_is_51_l2695_269568


namespace NUMINAMATH_CALUDE_sequence_sum_l2695_269563

theorem sequence_sum (A B C D E F G H I J : ℤ) : 
  D = 7 ∧ 
  A + B + C = 24 ∧ 
  B + C + D = 24 ∧ 
  C + D + E = 24 ∧ 
  D + E + F = 24 ∧ 
  E + F + G = 24 ∧ 
  F + G + H = 24 ∧ 
  G + H + I = 24 ∧ 
  H + I + J = 24 → 
  A + J = 105 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l2695_269563


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_l2695_269506

theorem angle_C_in_triangle (A B C : ℝ) (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
    (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) : C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_in_triangle_l2695_269506


namespace NUMINAMATH_CALUDE_mens_wages_proof_l2695_269556

-- Define the number of men, women, and boys
def num_men : ℕ := 5
def num_boys : ℕ := 8

-- Define the total earnings
def total_earnings : ℚ := 90

-- Define the relationship between men, women, and boys
axiom men_women_equality : ∃ w : ℕ, num_men = w
axiom women_boys_equality : ∃ w : ℕ, w = num_boys

-- Define the theorem
theorem mens_wages_proof :
  ∃ (wage_man wage_woman wage_boy : ℚ),
    wage_man > 0 ∧ wage_woman > 0 ∧ wage_boy > 0 ∧
    num_men * wage_man + num_boys * wage_boy + num_boys * wage_woman = total_earnings ∧
    num_men * wage_man = 30 :=
sorry

end NUMINAMATH_CALUDE_mens_wages_proof_l2695_269556


namespace NUMINAMATH_CALUDE_gcd_540_180_minus_2_l2695_269591

theorem gcd_540_180_minus_2 : Int.gcd 540 180 - 2 = 178 := by
  sorry

end NUMINAMATH_CALUDE_gcd_540_180_minus_2_l2695_269591


namespace NUMINAMATH_CALUDE_parabola_vertex_l2695_269533

/-- The parabola defined by y = (x-1)^2 + 2 has its vertex at (1,2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x - 1)^2 + 2 → (1, 2) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2695_269533


namespace NUMINAMATH_CALUDE_valid_paths_count_l2695_269559

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Represents a vertical line segment -/
structure VerticalSegment where
  x : Nat
  y_start : Nat
  y_end : Nat

/-- Definition of the grid and forbidden segments -/
def grid_height : Nat := 5
def grid_width : Nat := 8
def forbidden_segment1 : VerticalSegment := { x := 3, y_start := 1, y_end := 3 }
def forbidden_segment2 : VerticalSegment := { x := 4, y_start := 2, y_end := 5 }

/-- Function to calculate the number of valid paths -/
def count_valid_paths (height width : Nat) (forbidden1 forbidden2 : VerticalSegment) : Nat :=
  sorry

/-- Theorem stating the number of valid paths -/
theorem valid_paths_count :
  count_valid_paths grid_height grid_width forbidden_segment1 forbidden_segment2 = 838 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l2695_269559


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l2695_269572

/-- A triangle in the diagram -/
structure Triangle where
  label : String

/-- The set of all triangles in the diagram -/
def all_triangles : Finset Triangle := sorry

/-- The set of shaded triangles -/
def shaded_triangles : Finset Triangle := sorry

/-- Each triangle has the same probability of being selected -/
axiom equal_probability : ∀ t : Triangle, t ∈ all_triangles → 
  (Finset.card {t} : ℚ) / (Finset.card all_triangles : ℚ) = 1 / (Finset.card all_triangles : ℚ)

theorem probability_of_shaded_triangle :
  (Finset.card shaded_triangles : ℚ) / (Finset.card all_triangles : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l2695_269572


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_expression_tight_l2695_269560

theorem min_value_expression (a b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (ha : a ≠ 0) :
  ((2*a + b)^2 + (b - c)^2 + (c - 2*a)^2) / b^2 ≥ 4/3 :=
sorry

theorem min_value_expression_tight (a b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (ha : a ≠ 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ((2*a + b)^2 + (b - c)^2 + (c - 2*a)^2) / b^2 < 4/3 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_expression_tight_l2695_269560


namespace NUMINAMATH_CALUDE_time_after_2023_hours_l2695_269522

def hours_later (current_time : Nat) (hours_passed : Nat) : Nat :=
  (current_time + hours_passed) % 12

theorem time_after_2023_hours :
  let current_time := 9
  let hours_passed := 2023
  hours_later current_time hours_passed = 8 := by
sorry

end NUMINAMATH_CALUDE_time_after_2023_hours_l2695_269522


namespace NUMINAMATH_CALUDE_reconstruct_quadrilateral_l2695_269595

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A convex quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The projection of a point onto a line segment -/
def projectPointOntoSegment (p : Point) (a : Point) (b : Point) : Point :=
  sorry

/-- Theorem: Given four points that are projections of the diagonal intersection
    onto the sides of a convex quadrilateral, we can reconstruct the quadrilateral -/
theorem reconstruct_quadrilateral 
  (M N K L : Point) 
  (h : ∃ (q : Quadrilateral), 
    M = projectPointOntoSegment (diagonalIntersection q) q.A q.B ∧
    N = projectPointOntoSegment (diagonalIntersection q) q.B q.C ∧
    K = projectPointOntoSegment (diagonalIntersection q) q.C q.D ∧
    L = projectPointOntoSegment (diagonalIntersection q) q.D q.A) :
  ∃! (q : Quadrilateral), 
    M = projectPointOntoSegment (diagonalIntersection q) q.A q.B ∧
    N = projectPointOntoSegment (diagonalIntersection q) q.B q.C ∧
    K = projectPointOntoSegment (diagonalIntersection q) q.C q.D ∧
    L = projectPointOntoSegment (diagonalIntersection q) q.D q.A :=
  sorry

end NUMINAMATH_CALUDE_reconstruct_quadrilateral_l2695_269595


namespace NUMINAMATH_CALUDE_max_change_percentage_l2695_269503

theorem max_change_percentage (initial_yes initial_no final_yes final_no fixed_mindset_ratio : ℚ)
  (h1 : initial_yes + initial_no = 1)
  (h2 : final_yes + final_no = 1)
  (h3 : initial_yes = 2/5)
  (h4 : initial_no = 3/5)
  (h5 : final_yes = 4/5)
  (h6 : final_no = 1/5)
  (h7 : fixed_mindset_ratio = 1/5) :
  let fixed_mindset := fixed_mindset_ratio * initial_no
  let max_change := final_yes - initial_yes
  max_change ≤ initial_no - fixed_mindset ∧ max_change = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_max_change_percentage_l2695_269503


namespace NUMINAMATH_CALUDE_tea_in_milk_equals_milk_in_tea_l2695_269513

/-- Represents the contents of a cup --/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure CupState where
  tea_cup : Cup
  milk_cup : Cup

/-- Initial state of the cups --/
def initial_state : CupState :=
  { tea_cup := { tea := 5, milk := 0 },
    milk_cup := { tea := 0, milk := 5 } }

/-- State after transferring milk to tea cup --/
def after_milk_transfer (state : CupState) : CupState :=
  { tea_cup := { tea := state.tea_cup.tea, milk := state.tea_cup.milk + 1 },
    milk_cup := { tea := state.milk_cup.tea, milk := state.milk_cup.milk - 1 } }

/-- State after transferring mixture back to milk cup --/
def after_mixture_transfer (state : CupState) : CupState :=
  let total_in_tea_cup := state.tea_cup.tea + state.tea_cup.milk
  let tea_fraction := state.tea_cup.tea / total_in_tea_cup
  let milk_fraction := state.tea_cup.milk / total_in_tea_cup
  { tea_cup := { tea := state.tea_cup.tea - tea_fraction, 
                 milk := state.tea_cup.milk - milk_fraction },
    milk_cup := { tea := state.milk_cup.tea + tea_fraction, 
                  milk := state.milk_cup.milk + milk_fraction } }

/-- Final state after both transfers --/
def final_state : CupState :=
  after_mixture_transfer (after_milk_transfer initial_state)

theorem tea_in_milk_equals_milk_in_tea :
  final_state.milk_cup.tea = final_state.tea_cup.milk := by
  sorry

end NUMINAMATH_CALUDE_tea_in_milk_equals_milk_in_tea_l2695_269513


namespace NUMINAMATH_CALUDE_raspberry_pies_l2695_269519

theorem raspberry_pies (total_pies : ℝ) (peach_ratio strawberry_ratio raspberry_ratio : ℝ) :
  total_pies = 36 ∧
  peach_ratio = 2 ∧
  strawberry_ratio = 5 ∧
  raspberry_ratio = 3 →
  (raspberry_ratio / (peach_ratio + strawberry_ratio + raspberry_ratio)) * total_pies = 10.8 :=
by sorry

end NUMINAMATH_CALUDE_raspberry_pies_l2695_269519


namespace NUMINAMATH_CALUDE_candy_problem_solution_l2695_269577

def candy_problem (packets : ℕ) (candies_per_packet : ℕ) (weekdays : ℕ) (weekend_days : ℕ) 
  (weekday_consumption : ℕ) (weekend_consumption : ℕ) : ℕ := 
  (packets * candies_per_packet) / 
  (weekdays * weekday_consumption + weekend_days * weekend_consumption)

theorem candy_problem_solution : 
  candy_problem 2 18 5 2 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_solution_l2695_269577


namespace NUMINAMATH_CALUDE_mike_score_l2695_269545

theorem mike_score (max_score : ℕ) (passing_percentage : ℚ) (shortfall : ℕ) (actual_score : ℕ) : 
  max_score = 780 → 
  passing_percentage = 30 / 100 → 
  shortfall = 22 → 
  actual_score = (max_score * passing_percentage).floor - shortfall → 
  actual_score = 212 := by
sorry

end NUMINAMATH_CALUDE_mike_score_l2695_269545


namespace NUMINAMATH_CALUDE_fish_total_weight_l2695_269586

/-- The weight of a fish given specific conditions about its parts -/
def fish_weight (tail_weight head_weight body_weight : ℝ) : Prop :=
  tail_weight = 4 ∧
  head_weight = tail_weight + (body_weight / 2) ∧
  body_weight = head_weight + tail_weight

theorem fish_total_weight :
  ∀ (tail_weight head_weight body_weight : ℝ),
    fish_weight tail_weight head_weight body_weight →
    tail_weight + head_weight + body_weight = 32 :=
by sorry

end NUMINAMATH_CALUDE_fish_total_weight_l2695_269586


namespace NUMINAMATH_CALUDE_problem_statement_l2695_269597

theorem problem_statement (a b : ℝ) 
  (h1 : a / b + b / a = 5 / 2)
  (h2 : a - b = 3 / 2) :
  (a^2 + 2*a*b + b^2 + 2*a^2*b + 2*a*b^2 + a^2*b^2 = 0) ∨ 
  (a^2 + 2*a*b + b^2 + 2*a^2*b + 2*a*b^2 + a^2*b^2 = 81) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2695_269597


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2695_269517

theorem ratio_sum_problem (x y z b : ℚ) : 
  x / y = 4 / 5 →
  y / z = 5 / 6 →
  y = 15 * b - 5 →
  x + y + z = 90 →
  b = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2695_269517


namespace NUMINAMATH_CALUDE_power_comparisons_l2695_269553

theorem power_comparisons :
  (3^40 > 4^30 ∧ 4^30 > 5^20) ∧
  (16^31 > 8^41 ∧ 8^41 > 4^61) ∧
  (∀ a b : ℝ, a > 1 → b > 1 → a^5 = 2 → b^7 = 3 → a < b) := by
  sorry


end NUMINAMATH_CALUDE_power_comparisons_l2695_269553


namespace NUMINAMATH_CALUDE_equal_integers_from_divisor_properties_l2695_269582

/-- The product of all divisors of a natural number -/
noncomputable def productOfDivisors (n : ℕ) : ℕ := sorry

/-- The number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ := sorry

theorem equal_integers_from_divisor_properties (m n s : ℕ) 
  (h_m_ge_n : m ≥ n) 
  (h_s_pos : s > 0) 
  (h_product : productOfDivisors (s * m) = productOfDivisors (s * n))
  (h_number : numberOfDivisors (s * m) = numberOfDivisors (s * n)) : 
  m = n :=
sorry

end NUMINAMATH_CALUDE_equal_integers_from_divisor_properties_l2695_269582


namespace NUMINAMATH_CALUDE_correct_grass_bundle_equations_l2695_269551

/-- Represents the number of roots in grass bundles -/
structure GrassBundles where
  high_quality : ℕ  -- number of roots in one bundle of high-quality grass
  low_quality : ℕ   -- number of roots in one bundle of low-quality grass

/-- Represents the relationships between high-quality and low-quality grass bundles -/
def grass_bundle_relations (g : GrassBundles) : Prop :=
  (5 * g.high_quality - 11 = 7 * g.low_quality) ∧
  (7 * g.high_quality - 25 = 5 * g.low_quality)

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem correct_grass_bundle_equations (g : GrassBundles) :
  grass_bundle_relations g ↔
  (5 * g.high_quality - 11 = 7 * g.low_quality) ∧
  (7 * g.high_quality - 25 = 5 * g.low_quality) :=
by sorry

end NUMINAMATH_CALUDE_correct_grass_bundle_equations_l2695_269551


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2695_269511

theorem inequality_equivalence (x : ℝ) (h : x ≠ 4) :
  (x^2 - 16) / (x - 4) ≤ 0 ↔ x ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2695_269511


namespace NUMINAMATH_CALUDE_bookstore_inventory_l2695_269529

/-- The number of books acquired by the bookstore. -/
def total_books : ℕ := 1000

/-- The number of books sold on the first day. -/
def first_day_sales : ℕ := total_books / 2

/-- The number of books sold on the second day. -/
def second_day_sales : ℕ := first_day_sales / 2 + first_day_sales + 50

/-- The number of books remaining after both days of sales. -/
def remaining_books : ℕ := 200

/-- Theorem stating that the total number of books is 1000, given the sales conditions. -/
theorem bookstore_inventory :
  total_books = 1000 ∧
  first_day_sales = total_books / 2 ∧
  second_day_sales = first_day_sales / 2 + first_day_sales + 50 ∧
  remaining_books = 200 ∧
  total_books = first_day_sales + second_day_sales + remaining_books :=
by sorry

end NUMINAMATH_CALUDE_bookstore_inventory_l2695_269529


namespace NUMINAMATH_CALUDE_length_width_difference_approx_l2695_269508

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  length_gt_width : length > width
  area_eq : area = length * width

/-- The difference between length and width of a rectangular field -/
def length_width_difference (field : RectangularField) : ℝ :=
  field.length - field.width

theorem length_width_difference_approx 
  (field : RectangularField) 
  (h_area : field.area = 171) 
  (h_length : field.length = 19.13) : 
  ∃ ε > 0, |length_width_difference field - 10.19| < ε :=
sorry

end NUMINAMATH_CALUDE_length_width_difference_approx_l2695_269508


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_5_range_of_m_for_f_geq_7_l2695_269552

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

-- Theorem for part I
theorem solution_set_when_m_eq_5 :
  {x : ℝ | f x 5 ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_7 :
  {m : ℝ | ∀ x, f x m ≥ 7} = {m : ℝ | m ≤ -13 ∨ 1 ≤ m} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_5_range_of_m_for_f_geq_7_l2695_269552


namespace NUMINAMATH_CALUDE_triangles_in_3x4_grid_l2695_269537

/-- Represents a rectangular grid with diagonals --/
structure RectangularGridWithDiagonals where
  rows : Nat
  columns : Nat

/-- Calculates the number of triangles in a rectangular grid with diagonals --/
def count_triangles (grid : RectangularGridWithDiagonals) : Nat :=
  let basic_triangles := 2 * grid.rows * grid.columns
  let row_triangles := grid.rows * (grid.columns - 1) * grid.columns / 2
  let diagonal_triangles := 2
  basic_triangles + row_triangles + diagonal_triangles

/-- Theorem: The number of triangles in a 3x4 grid with diagonals is 44 --/
theorem triangles_in_3x4_grid :
  count_triangles ⟨3, 4⟩ = 44 := by
  sorry

#eval count_triangles ⟨3, 4⟩

end NUMINAMATH_CALUDE_triangles_in_3x4_grid_l2695_269537


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l2695_269596

theorem geometric_mean_of_4_and_16 : 
  ∃ x : ℝ, x > 0 ∧ x^2 = 4 * 16 ∧ (x = 8 ∨ x = -8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l2695_269596


namespace NUMINAMATH_CALUDE_virginia_adrienne_teaching_difference_l2695_269598

theorem virginia_adrienne_teaching_difference :
  ∀ (V A D : ℕ),
  V + A + D = 102 →
  D = 43 →
  V = D - 9 →
  V > A →
  V - A = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_virginia_adrienne_teaching_difference_l2695_269598


namespace NUMINAMATH_CALUDE_exam_score_difference_l2695_269583

theorem exam_score_difference (score_65 score_75 score_85 score_95 : ℝ)
  (percent_65 percent_75 percent_85 percent_95 : ℝ)
  (h1 : score_65 = 65)
  (h2 : score_75 = 75)
  (h3 : score_85 = 85)
  (h4 : score_95 = 95)
  (h5 : percent_65 = 0.15)
  (h6 : percent_75 = 0.40)
  (h7 : percent_85 = 0.20)
  (h8 : percent_95 = 0.25)
  (h9 : percent_65 + percent_75 + percent_85 + percent_95 = 1) :
  let mean := score_65 * percent_65 + score_75 * percent_75 + score_85 * percent_85 + score_95 * percent_95
  let median := score_75
  mean - median = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_difference_l2695_269583


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l2695_269543

theorem phoenix_airport_on_time_rate (late_flights : ℕ) (on_time_flights : ℕ) (additional_on_time_flights : ℕ) :
  late_flights = 1 →
  on_time_flights = 3 →
  additional_on_time_flights = 2 →
  (on_time_flights + additional_on_time_flights : ℚ) / (late_flights + on_time_flights + additional_on_time_flights) > 83.33 / 100 := by
sorry

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l2695_269543


namespace NUMINAMATH_CALUDE_min_circle_property_l2695_269579

/-- Definition of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + y = -1

/-- Definition of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Definition of the circle with minimal area -/
def minCircle (x y : ℝ) : Prop := x^2 + y^2 + (6/5)*x + (3/5)*y + 1 = 0

/-- Theorem stating that the minCircle passes through the intersection points of circle1 and circle2 and has the minimum area -/
theorem min_circle_property :
  ∀ (x y : ℝ), 
    (circle1 x y ∧ circle2 x y → minCircle x y) ∧
    (∀ (a b c : ℝ), (∀ (u v : ℝ), circle1 u v ∧ circle2 u v → x^2 + y^2 + 2*a*x + 2*b*y + c = 0) →
      (x^2 + y^2 + (6/5)*x + (3/5)*y + 1)^2 ≤ (x^2 + y^2 + 2*a*x + 2*b*y + c)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_circle_property_l2695_269579


namespace NUMINAMATH_CALUDE_girls_in_college_l2695_269569

theorem girls_in_college (total : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) (girls : ℕ) :
  total = 440 →
  ratio_boys = 6 →
  ratio_girls = 5 →
  ratio_boys * girls = ratio_girls * (total - girls) →
  girls = 200 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l2695_269569


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2695_269599

/-- A rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  /-- Width of the left rectangles -/
  a : ℝ
  /-- Height of the top rectangles -/
  b : ℝ
  /-- Width of the right rectangles -/
  c : ℝ
  /-- Height of the bottom rectangles -/
  d : ℝ
  /-- Area of the top left rectangle is 6 -/
  top_left_area : a * b = 6
  /-- Area of the top right rectangle is 15 -/
  top_right_area : b * c = 15
  /-- Area of the bottom right rectangle is 25 -/
  bottom_right_area : c * d = 25

/-- The area of the fourth (shaded) rectangle in a DividedRectangle is 10 -/
theorem fourth_rectangle_area (r : DividedRectangle) : r.a * r.d = 10 := by
  sorry


end NUMINAMATH_CALUDE_fourth_rectangle_area_l2695_269599


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2695_269502

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1 ∧
    ∀ k l : ℕ, k > 0 → l > 0 → (a k * a l).sqrt = 4 * a 1 →
      1 / m + 4 / n ≤ 1 / k + 4 / l) ∧
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1 ∧
    1 / m + 4 / n = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2695_269502


namespace NUMINAMATH_CALUDE_dog_cat_food_weight_difference_l2695_269561

theorem dog_cat_food_weight_difference :
  -- Define the constants from the problem
  let cat_food_bags : ℕ := 2
  let cat_food_weight_per_bag : ℕ := 3 -- in pounds
  let dog_food_bags : ℕ := 2
  let ounces_per_pound : ℕ := 16
  let total_pet_food_ounces : ℕ := 256

  -- Calculate total cat food weight in ounces
  let total_cat_food_ounces : ℕ := cat_food_bags * cat_food_weight_per_bag * ounces_per_pound
  
  -- Calculate total dog food weight in ounces
  let total_dog_food_ounces : ℕ := total_pet_food_ounces - total_cat_food_ounces
  
  -- Calculate weight per bag of dog food in ounces
  let dog_food_weight_per_bag_ounces : ℕ := total_dog_food_ounces / dog_food_bags
  
  -- Calculate weight per bag of cat food in ounces
  let cat_food_weight_per_bag_ounces : ℕ := cat_food_weight_per_bag * ounces_per_pound
  
  -- Calculate the difference in weight between dog and cat food bags in ounces
  let weight_difference_ounces : ℕ := dog_food_weight_per_bag_ounces - cat_food_weight_per_bag_ounces
  
  -- Convert the weight difference to pounds
  weight_difference_ounces / ounces_per_pound = 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_cat_food_weight_difference_l2695_269561


namespace NUMINAMATH_CALUDE_odot_inequality_implies_a_range_l2695_269555

-- Define the operation ⊙
def odot (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem odot_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_odot_inequality_implies_a_range_l2695_269555


namespace NUMINAMATH_CALUDE_area_divisibility_l2695_269580

/-- A point with integer coordinates -/
structure IntegerPoint where
  x : ℤ
  y : ℤ

/-- A convex polygon with vertices on a circle -/
structure ConvexPolygonOnCircle where
  vertices : List IntegerPoint
  is_convex : sorry
  on_circle : sorry

/-- The statement of the theorem -/
theorem area_divisibility
  (P : ConvexPolygonOnCircle)
  (n : ℕ)
  (n_odd : Odd n)
  (side_length_squared_div : ∃ (side_length : ℕ), (side_length ^ 2) % n = 0) :
  ∃ (area : ℕ), (2 * area) % n = 0 := by
  sorry


end NUMINAMATH_CALUDE_area_divisibility_l2695_269580


namespace NUMINAMATH_CALUDE_sequence_inequality_l2695_269509

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n ≥ 0)
  (h2 : ∀ n : ℕ, a n + a (2*n) ≥ 3*n)
  (h3 : ∀ n : ℕ, a (n+1) + n ≤ 2 * Real.sqrt (a n * (n+1))) :
  ∀ n : ℕ, a n ≥ n := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2695_269509


namespace NUMINAMATH_CALUDE_stimulus_check_amount_l2695_269521

theorem stimulus_check_amount : ∃ T : ℚ, 
  (27 / 125 : ℚ) * T = 432 ∧ T = 2000 := by
  sorry

end NUMINAMATH_CALUDE_stimulus_check_amount_l2695_269521


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l2695_269524

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let f := fun x => (Real.cos x)^2
  let g := Real.tan
  Matrix.det !![f A, g A, 1; f B, g B, 1; f C, g C, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l2695_269524


namespace NUMINAMATH_CALUDE_total_oreos_count_l2695_269574

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := 11

/-- The number of Oreos James has -/
def james_oreos : ℕ := 2 * jordan_oreos + 3

/-- The total number of Oreos -/
def total_oreos : ℕ := jordan_oreos + james_oreos

theorem total_oreos_count : total_oreos = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_oreos_count_l2695_269574


namespace NUMINAMATH_CALUDE_sum_squares_inequality_sum_squares_equality_l2695_269540

theorem sum_squares_inequality (a b c : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) 
  (h_sum_cubes : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := by
  sorry

-- Equality case
theorem sum_squares_equality (a b c : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) 
  (h_sum_cubes : a^3 + b^3 + c^3 = 1) :
  (a + b + c + a^2 + b^2 + c^2 = 4) ↔ 
  ((a = 1 ∧ b = 1 ∧ c = -1) ∨ 
   (a = 1 ∧ b = -1 ∧ c = 1) ∨ 
   (a = -1 ∧ b = 1 ∧ c = 1)) := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_sum_squares_equality_l2695_269540


namespace NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l2695_269573

/-- Given a point P(-3, 5), its symmetrical point P' with respect to the x-axis has coordinates (-3, -5). -/
theorem symmetry_wrt_x_axis :
  let P : ℝ × ℝ := (-3, 5)
  let P' : ℝ × ℝ := (-3, -5)
  (P'.1 = P.1) ∧ (P'.2 = -P.2) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l2695_269573


namespace NUMINAMATH_CALUDE_probability_white_or_red_l2695_269550

def total_balls : ℕ := 8 + 9 + 3
def white_balls : ℕ := 8
def black_balls : ℕ := 9
def red_balls : ℕ := 3

theorem probability_white_or_red :
  (white_balls + red_balls : ℚ) / total_balls = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_or_red_l2695_269550


namespace NUMINAMATH_CALUDE_beach_probability_l2695_269571

/-- Given a beach scenario where:
  * 75 people are wearing sunglasses
  * 60 people are wearing hats
  * The probability of wearing sunglasses given wearing a hat is 1/3
  This theorem proves that the probability of wearing a hat given wearing sunglasses is 4/15. -/
theorem beach_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_sunglasses_given_hat : ℚ) :
  total_sunglasses = 75 →
  total_hats = 60 →
  prob_sunglasses_given_hat = 1/3 →
  (total_hats * prob_sunglasses_given_hat : ℚ) / total_sunglasses = 4/15 :=
by sorry

end NUMINAMATH_CALUDE_beach_probability_l2695_269571


namespace NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_joker_l2695_269527

theorem probability_at_least_one_diamond_or_joker :
  let total_cards : ℕ := 60
  let diamond_cards : ℕ := 15
  let joker_cards : ℕ := 6
  let favorable_cards : ℕ := diamond_cards + joker_cards
  let prob_not_favorable : ℚ := (total_cards - favorable_cards) / total_cards
  let prob_neither_favorable : ℚ := prob_not_favorable * prob_not_favorable
  prob_neither_favorable = 169 / 400 →
  1 - prob_neither_favorable = 231 / 400 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_joker_l2695_269527


namespace NUMINAMATH_CALUDE_choir_third_group_members_l2695_269514

theorem choir_third_group_members (total_members : ℕ) (group1_members : ℕ) (group2_members : ℕ) 
  (h1 : total_members = 70)
  (h2 : group1_members = 25)
  (h3 : group2_members = 30) :
  total_members - (group1_members + group2_members) = 15 := by
sorry

end NUMINAMATH_CALUDE_choir_third_group_members_l2695_269514


namespace NUMINAMATH_CALUDE_base3_11111_is_121_l2695_269532

def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base3_11111_is_121 :
  base3_to_decimal [1, 1, 1, 1, 1] = 121 := by
  sorry

end NUMINAMATH_CALUDE_base3_11111_is_121_l2695_269532


namespace NUMINAMATH_CALUDE_farm_cows_l2695_269554

theorem farm_cows (milk_per_6_cows : ℝ) (total_milk : ℝ) (weeks : ℕ) :
  milk_per_6_cows = 108 →
  total_milk = 2160 →
  weeks = 5 →
  (total_milk / (milk_per_6_cows / 6) / weeks : ℝ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_farm_cows_l2695_269554


namespace NUMINAMATH_CALUDE_exterior_angles_hexagon_pentagon_l2695_269523

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (n : ℕ) : ℝ := 360

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- A pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

theorem exterior_angles_hexagon_pentagon : 
  sum_exterior_angles hexagon_sides = sum_exterior_angles pentagon_sides := by
  sorry

end NUMINAMATH_CALUDE_exterior_angles_hexagon_pentagon_l2695_269523


namespace NUMINAMATH_CALUDE_jakes_comic_books_l2695_269578

/-- Jake's comic book problem -/
theorem jakes_comic_books (jake_books : ℕ) (total_books : ℕ) (brother_books : ℕ) : 
  jake_books = 36 →
  total_books = 87 →
  brother_books > jake_books →
  total_books = jake_books + brother_books →
  brother_books - jake_books = 15 := by
sorry

end NUMINAMATH_CALUDE_jakes_comic_books_l2695_269578


namespace NUMINAMATH_CALUDE_total_sugar_calculation_l2695_269531

def chocolate_bars : ℕ := 14
def sugar_per_bar : ℕ := 10
def lollipop_sugar : ℕ := 37

theorem total_sugar_calculation :
  chocolate_bars * sugar_per_bar + lollipop_sugar = 177 := by
  sorry

end NUMINAMATH_CALUDE_total_sugar_calculation_l2695_269531


namespace NUMINAMATH_CALUDE_equation_2x_minus_1_is_linear_l2695_269515

/-- A linear equation in one variable is of the form ax + b = 0, where a ≠ 0 and x is the variable. --/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 1 = 0 --/
def f (x : ℝ) : ℝ := 2 * x - 1

theorem equation_2x_minus_1_is_linear : is_linear_equation_one_var f := by
  sorry


end NUMINAMATH_CALUDE_equation_2x_minus_1_is_linear_l2695_269515


namespace NUMINAMATH_CALUDE_two_questions_determine_number_l2695_269564

theorem two_questions_determine_number : 
  ∃ (q₁ q₂ : ℕ → ℕ → ℕ), 
    (∀ m : ℕ, m ≥ 2 → q₁ m ≥ 2) ∧ 
    (∀ m : ℕ, m ≥ 2 → q₂ m ≥ 2) ∧ 
    (∀ V : ℕ, 1 ≤ V ∧ V ≤ 100 → 
      ∀ V' : ℕ, 1 ≤ V' ∧ V' ≤ 100 → 
        (V / q₁ V = V' / q₁ V' ∧ V / q₂ V = V' / q₂ V') → V = V') :=
sorry

end NUMINAMATH_CALUDE_two_questions_determine_number_l2695_269564


namespace NUMINAMATH_CALUDE_fuel_cost_per_liter_l2695_269542

-- Define constants
def service_cost : ℝ := 2.10
def mini_vans : ℕ := 3
def trucks : ℕ := 2
def total_cost : ℝ := 299.1
def mini_van_tank : ℝ := 65
def truck_tank_multiplier : ℝ := 2.2  -- 120% bigger means 2.2 times the size

-- Define functions
def total_service_cost : ℝ := service_cost * (mini_vans + trucks)
def truck_tank : ℝ := mini_van_tank * truck_tank_multiplier
def total_fuel_volume : ℝ := mini_vans * mini_van_tank + trucks * truck_tank
def fuel_cost : ℝ := total_cost - total_service_cost

-- Theorem to prove
theorem fuel_cost_per_liter : fuel_cost / total_fuel_volume = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_per_liter_l2695_269542


namespace NUMINAMATH_CALUDE_jerry_showers_l2695_269544

/-- Represents the water usage scenario for Jerry in July --/
structure WaterUsage where
  totalAllowance : ℕ
  drinkingCooking : ℕ
  showerUsage : ℕ
  poolLength : ℕ
  poolWidth : ℕ
  poolHeight : ℕ
  gallonToCubicFoot : ℕ

/-- Calculates the number of showers Jerry can take in July --/
def calculateShowers (w : WaterUsage) : ℕ :=
  let poolVolume := w.poolLength * w.poolWidth * w.poolHeight
  let remainingWater := w.totalAllowance - w.drinkingCooking - poolVolume
  remainingWater / w.showerUsage

/-- Theorem stating that Jerry can take 15 showers in July --/
theorem jerry_showers (w : WaterUsage) 
  (h1 : w.totalAllowance = 1000)
  (h2 : w.drinkingCooking = 100)
  (h3 : w.showerUsage = 20)
  (h4 : w.poolLength = 10)
  (h5 : w.poolWidth = 10)
  (h6 : w.poolHeight = 6)
  (h7 : w.gallonToCubicFoot = 1) :
  calculateShowers w = 15 := by
  sorry

end NUMINAMATH_CALUDE_jerry_showers_l2695_269544


namespace NUMINAMATH_CALUDE_problem_statement_l2695_269570

theorem problem_statement (x y : ℝ) : 
  y = (3/4) * x →
  x^y = y^x →
  x + y = 448/81 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2695_269570


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2695_269526

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 → p_black = 0.5 → p_red + p_black + p_white = 1 → p_white = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2695_269526


namespace NUMINAMATH_CALUDE_difference_of_sixes_in_7669_l2695_269501

/-- Given a natural number n, returns the digit at the i-th place (0-indexed from right) -/
def digit_at_place (n : ℕ) (i : ℕ) : ℕ := 
  (n / (10 ^ i)) % 10

/-- Given a natural number n, returns the place value of the digit at the i-th place -/
def place_value (n : ℕ) (i : ℕ) : ℕ := 
  digit_at_place n i * (10 ^ i)

theorem difference_of_sixes_in_7669 : 
  place_value 7669 2 - place_value 7669 1 = 540 := by sorry

end NUMINAMATH_CALUDE_difference_of_sixes_in_7669_l2695_269501


namespace NUMINAMATH_CALUDE_davids_physics_marks_l2695_269534

/-- Calculates the marks in Physics given marks in other subjects and the average --/
def physics_marks (english : ℕ) (mathematics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + chemistry + biology)

/-- Theorem: Given David's marks and average, his Physics marks are 82 --/
theorem davids_physics_marks :
  physics_marks 86 89 87 81 85 = 82 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l2695_269534


namespace NUMINAMATH_CALUDE_greatest_angle_in_triangle_l2695_269547

theorem greatest_angle_in_triangle (a b c : ℝ) (h : (b / (c - a)) - (a / (b + c)) = 1) :
  ∃ (A B C : ℝ), 
    A + B + C = 180 ∧ 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C ∧
    max A (max B C) = 120 := by
  sorry

end NUMINAMATH_CALUDE_greatest_angle_in_triangle_l2695_269547


namespace NUMINAMATH_CALUDE_pencil_length_l2695_269590

theorem pencil_length : ∀ (L : ℝ), 
  (L > 0) →                          -- Ensure positive length
  ((1/8) * L + (1/2) * (7/8) * L + 7/2 = L) →  -- Parts sum to total
  (L = 8) :=                         -- Total length is 8
by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l2695_269590


namespace NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l2695_269558

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The land area of the Earth in km² -/
def earthLandArea : ℝ := 149000000

/-- The number of significant figures to retain -/
def sigFiguresRequired : ℕ := 3

theorem earth_land_area_scientific_notation :
  toScientificNotation earthLandArea sigFiguresRequired =
    ScientificNotation.mk 1.49 8 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l2695_269558


namespace NUMINAMATH_CALUDE_min_sum_distances_squared_l2695_269541

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the center of the ellipse -/
def center : ℝ × ℝ := (0, 0)

/-- Definition of the left focus of the ellipse -/
def left_focus : ℝ × ℝ := (-1, 0)

/-- Square of the distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The minimum value of |OP|^2 + |PF|^2 is 2 -/
theorem min_sum_distances_squared :
  ∀ (x y : ℝ), is_on_ellipse x y →
  ∃ (min : ℝ), min = 2 ∧
  ∀ (p : ℝ × ℝ), is_on_ellipse p.1 p.2 →
  distance_squared center p + distance_squared p left_focus ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_squared_l2695_269541
