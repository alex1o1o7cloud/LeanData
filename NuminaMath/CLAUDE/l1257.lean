import Mathlib

namespace NUMINAMATH_CALUDE_initial_deposit_proof_l1257_125745

/-- Represents the initial deposit amount in dollars -/
def initial_deposit : ℝ := 500

/-- Represents the interest earned in the first year in dollars -/
def first_year_interest : ℝ := 100

/-- Represents the balance at the end of the first year in dollars -/
def first_year_balance : ℝ := 600

/-- Represents the percentage increase in the second year -/
def second_year_increase_rate : ℝ := 0.1

/-- Represents the total percentage increase over two years -/
def total_increase_rate : ℝ := 0.32

theorem initial_deposit_proof :
  initial_deposit + first_year_interest = first_year_balance ∧
  first_year_balance * (1 + second_year_increase_rate) = initial_deposit * (1 + total_increase_rate) := by
  sorry

#check initial_deposit_proof

end NUMINAMATH_CALUDE_initial_deposit_proof_l1257_125745


namespace NUMINAMATH_CALUDE_extreme_value_and_slope_l1257_125737

/-- A function f with an extreme value at x = 1 -/
def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (x : ℝ) (a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_and_slope (a b : ℝ) :
  f 1 a b = 10 ∧ f' 1 a b = 0 → f' 2 a b = 17 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_slope_l1257_125737


namespace NUMINAMATH_CALUDE_classroom_addition_problem_l1257_125735

theorem classroom_addition_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 6) (h3 : x * y = 45) : 
  x = 11 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_classroom_addition_problem_l1257_125735


namespace NUMINAMATH_CALUDE_negation_equivalence_l1257_125752

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Teenager : U → Prop)
variable (Responsible : U → Prop)

-- State the theorem
theorem negation_equivalence :
  (∃ x, Teenager x ∧ ¬Responsible x) ↔ ¬(∀ x, Teenager x → Responsible x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1257_125752


namespace NUMINAMATH_CALUDE_f_max_value_l1257_125791

/-- The quadratic function f(x) = -3x^2 + 18x - 4 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 4

/-- The maximum value of f(x) is 77 -/
theorem f_max_value : ∃ (M : ℝ), M = 77 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1257_125791


namespace NUMINAMATH_CALUDE_range_of_m_l1257_125704

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 5
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, p x ∧ ¬(q x m)) →
  (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1257_125704


namespace NUMINAMATH_CALUDE_equation_solutions_l1257_125769

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3) ∧
  (∀ x : ℝ, 5*x + 2 = 3*x^2 ↔ x = -1/3 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1257_125769


namespace NUMINAMATH_CALUDE_exists_close_points_on_graphs_l1257_125727

open Real

/-- The function f(x) = x^4 -/
def f (x : ℝ) : ℝ := x^4

/-- The function g(x) = x^4 + x^2 + x + 1 -/
def g (x : ℝ) : ℝ := x^4 + x^2 + x + 1

/-- Theorem stating the existence of points A and B on the graphs of f and g with distance < 1/100 -/
theorem exists_close_points_on_graphs :
  ∃ (u v : ℝ), |u - v| < 1/100 ∧ f v = g u := by sorry

end NUMINAMATH_CALUDE_exists_close_points_on_graphs_l1257_125727


namespace NUMINAMATH_CALUDE_gcd_39_91_l1257_125760

theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_39_91_l1257_125760


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1257_125730

/-- A sequence where each term is 1/3 of the previous term -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1 / 3) * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h : geometric_sequence a) (h1 : a 4 + a 5 = 4) : 
  a 2 + a 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1257_125730


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1257_125719

/-- Given two real numbers with sum S, prove that adding 5 to each and then tripling results in a sum of 3S + 30 -/
theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry


end NUMINAMATH_CALUDE_final_sum_after_operations_l1257_125719


namespace NUMINAMATH_CALUDE_inequality_solution_l1257_125732

theorem inequality_solution :
  ∀ x : ℕ, 1 + x ≥ 2 * x - 1 ↔ x ∈ ({0, 1, 2} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1257_125732


namespace NUMINAMATH_CALUDE_problem_distribution_l1257_125746

theorem problem_distribution (n m : ℕ) (h1 : n = 7) (h2 : m = 5) :
  (Nat.choose n m) * (m ^ (n - m)) = 525 := by
  sorry

end NUMINAMATH_CALUDE_problem_distribution_l1257_125746


namespace NUMINAMATH_CALUDE_square_product_inequality_l1257_125779

theorem square_product_inequality (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_inequality_l1257_125779


namespace NUMINAMATH_CALUDE_sum_independence_and_value_l1257_125762

theorem sum_independence_and_value (a : ℝ) (h : a ≥ -3/4) :
  let s := (((a + 1) / 2 + (a + 3) / 6 * Real.sqrt ((4 * a + 3) / 3)) ^ (1/3 : ℝ) : ℝ)
  let t := (((a + 1) / 2 - (a + 3) / 6 * Real.sqrt ((4 * a + 3) / 3)) ^ (1/3 : ℝ) : ℝ)
  s + t = 1 := by sorry

end NUMINAMATH_CALUDE_sum_independence_and_value_l1257_125762


namespace NUMINAMATH_CALUDE_evaluate_expression_l1257_125702

theorem evaluate_expression (x y : ℚ) (hx : x = 4/8) (hy : y = 5/6) :
  (8*x + 6*y) / (72*x*y) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1257_125702


namespace NUMINAMATH_CALUDE_puzzle_pieces_count_l1257_125759

theorem puzzle_pieces_count (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (num_500_piece_puzzles : ℕ) (num_unknown_piece_puzzles : ℕ) :
  pieces_per_hour = 100 →
  hours_per_day = 7 →
  days = 7 →
  num_500_piece_puzzles = 5 →
  num_unknown_piece_puzzles = 8 →
  (pieces_per_hour * hours_per_day * days - num_500_piece_puzzles * 500) / num_unknown_piece_puzzles = 300 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_count_l1257_125759


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1257_125715

/-- Given a hyperbola and a circle, if the chord cut by one of the asymptotes of the hyperbola
    from the circle has a length of 2, then the length of the real axis of the hyperbola is 2. -/
theorem hyperbola_real_axis_length (a : ℝ) : 
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / 3 = 1
  let circle := fun (x y : ℝ) ↦ (x - 2)^2 + y^2 = 4
  let asymptote := fun (x : ℝ) ↦ (Real.sqrt 3 / a) * x
  ∃ (x₁ x₂ : ℝ), 
    (circle x₁ (asymptote x₁) ∧ circle x₂ (asymptote x₂)) ∧ 
    ((x₁ - x₂)^2 + (asymptote x₁ - asymptote x₂)^2 = 4) →
  2 * a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1257_125715


namespace NUMINAMATH_CALUDE_function_lower_bound_l1257_125734

/-- Given a function f(x) = x^2 - (a+1)x + a, where a is a real number,
    if f(x) ≥ -1 for all x > 1, then a ≤ 3 -/
theorem function_lower_bound (a : ℝ) :
  (∀ x > 1, x^2 - (a + 1)*x + a ≥ -1) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l1257_125734


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1257_125767

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 4 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1257_125767


namespace NUMINAMATH_CALUDE_max_total_points_l1257_125714

/-- Represents the types of buckets in the ring toss game -/
inductive Bucket
| Red
| Green
| Blue

/-- Represents the game state -/
structure GameState where
  money : ℕ
  points : ℕ
  rings_per_play : ℕ
  red_points : ℕ
  green_points : ℕ
  blue_points : ℕ
  blue_success_rate : ℚ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ
  blue_buckets_hit : ℕ

/-- Calculates the maximum points achievable in one play -/
def max_points_per_play (gs : GameState) : ℕ :=
  gs.rings_per_play * max gs.red_points (max gs.green_points gs.blue_points)

/-- Calculates the total points from already hit buckets -/
def current_points (gs : GameState) : ℕ :=
  gs.red_buckets_hit * gs.red_points +
  gs.green_buckets_hit * gs.green_points +
  gs.blue_buckets_hit * gs.blue_points

/-- Theorem: The maximum total points Tiffany can achieve in three games is 43 -/
theorem max_total_points (gs : GameState)
  (h1 : gs.money = 3)
  (h2 : gs.rings_per_play = 5)
  (h3 : gs.red_points = 2)
  (h4 : gs.green_points = 3)
  (h5 : gs.blue_points = 5)
  (h6 : gs.blue_success_rate = 1/10)
  (h7 : gs.red_buckets_hit = 4)
  (h8 : gs.green_buckets_hit = 5)
  (h9 : gs.blue_buckets_hit = 1) :
  current_points gs + max_points_per_play gs = 43 :=
by sorry

end NUMINAMATH_CALUDE_max_total_points_l1257_125714


namespace NUMINAMATH_CALUDE_angle_sum_undetermined_l1257_125749

/-- Two angles are consecutive interior angles -/
def consecutive_interior (α β : ℝ) : Prop := sorry

/-- The statement that the sum of two angles equals 180 degrees cannot be proven or disproven -/
def undetermined_sum (α β : ℝ) : Prop :=
  ¬(∀ (h : consecutive_interior α β), α + β = 180) ∧
  ¬(∀ (h : consecutive_interior α β), α + β ≠ 180)

theorem angle_sum_undetermined (α β : ℝ) :
  consecutive_interior α β → α = 78 → undetermined_sum α β :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_undetermined_l1257_125749


namespace NUMINAMATH_CALUDE_min_value_d_l1257_125712

def a (n : ℕ+) : ℚ := 1000 / n
def b (n k : ℕ+) : ℚ := 2000 / (k * n)
def c (n k : ℕ+) : ℚ := 1500 / (200 - n - k * n)
def d (n k : ℕ+) : ℚ := max (a n) (max (b n k) (c n k))

theorem min_value_d (n k : ℕ+) (h : n + k * n < 200) :
  ∃ (n₀ k₀ : ℕ+), d n₀ k₀ = 250 / 11 ∧ ∀ (n' k' : ℕ+), n' + k' * n' < 200 → d n' k' ≥ 250 / 11 :=
sorry

end NUMINAMATH_CALUDE_min_value_d_l1257_125712


namespace NUMINAMATH_CALUDE_ellipse_properties_l1257_125770

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (3/2)^2 / a^2 + 6 / b^2 = 1  -- Point M (3/2, √6) lies on the ellipse
  h4 : 2 * (a^2 - b^2).sqrt = 2     -- Focal length is 2

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  e.a = 3 ∧ e.b^2 = 8

/-- The trajectory equation of point E -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3

theorem ellipse_properties (e : Ellipse) :
  standard_equation e ∧ ∀ x y, trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1257_125770


namespace NUMINAMATH_CALUDE_arcsin_one_half_l1257_125718

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l1257_125718


namespace NUMINAMATH_CALUDE_gcd_of_squares_l1257_125785

theorem gcd_of_squares : Nat.gcd (114^2 + 226^2 + 338^2) (113^2 + 225^2 + 339^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l1257_125785


namespace NUMINAMATH_CALUDE_intersection_difference_is_zero_l1257_125725

noncomputable def f (x : ℝ) : ℝ := 2 - x^3 + x^4
noncomputable def g (x : ℝ) : ℝ := 1 + 2*x^3 + x^4

theorem intersection_difference_is_zero :
  ∀ x y : ℝ, f x = g x → f y = g y → |f x - g y| = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_difference_is_zero_l1257_125725


namespace NUMINAMATH_CALUDE_multiple_of_number_l1257_125799

theorem multiple_of_number : ∃ m : ℕ, m < 4 ∧ 7 * 5 - 15 > m * 5 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_number_l1257_125799


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1257_125729

theorem unique_solution_quadratic (q : ℝ) : 
  (q ≠ 0 ∧ ∀ x : ℝ, (q * x^2 - 18 * x + 8 = 0 → (∀ y : ℝ, q * y^2 - 18 * y + 8 = 0 → x = y))) ↔ 
  q = 81/8 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1257_125729


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l1257_125740

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Given two similar triangles satisfying certain conditions, 
    prove that the corresponding side of the larger triangle is 12 feet -/
theorem similar_triangles_problem 
  (small large : Triangle)
  (area_diff : large.area - small.area = 72)
  (area_ratio : ∃ k : ℕ, large.area / small.area = k^2)
  (small_area_int : ∃ n : ℕ, small.area = n)
  (small_side : small.side = 6)
  : large.side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l1257_125740


namespace NUMINAMATH_CALUDE_tan_ratio_equals_two_l1257_125773

theorem tan_ratio_equals_two (α β γ : ℝ) 
  (h : Real.sin (2 * (α + γ)) = 3 * Real.sin (2 * β)) : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_equals_two_l1257_125773


namespace NUMINAMATH_CALUDE_equal_chords_implies_tangential_l1257_125787

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Add necessary fields

/-- A circle -/
structure Circle where
  -- Add necessary fields

/-- Represents the property that a circle intersects each side of a quadrilateral at two points forming equal chords -/
def has_equal_chords_intersection (q : ConvexQuadrilateral) (c : Circle) : Prop :=
  sorry

/-- A quadrilateral is tangential if it has an inscribed circle -/
def is_tangential (q : ConvexQuadrilateral) : Prop :=
  ∃ c : Circle, sorry -- c is inscribed in q

/-- If a convex quadrilateral has the property that a circle intersects each of its sides 
    at two points forming equal chords, then the quadrilateral is tangential -/
theorem equal_chords_implies_tangential (q : ConvexQuadrilateral) (c : Circle) :
  has_equal_chords_intersection q c → is_tangential q :=
by
  sorry

end NUMINAMATH_CALUDE_equal_chords_implies_tangential_l1257_125787


namespace NUMINAMATH_CALUDE_solve_for_k_l1257_125754

theorem solve_for_k : ∃ k : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 4 → k * x + y = 3) → 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l1257_125754


namespace NUMINAMATH_CALUDE_matrix_power_four_l1257_125701

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A^4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1257_125701


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l1257_125781

/-- The probability of A hitting the enemy plane -/
def prob_A_hit : ℝ := 0.6

/-- The probability of B hitting the enemy plane -/
def prob_B_hit : ℝ := 0.5

/-- The probability that the enemy plane is hit by at least one of A or B -/
def prob_plane_hit : ℝ := 1 - (1 - prob_A_hit) * (1 - prob_B_hit)

theorem enemy_plane_hit_probability :
  prob_plane_hit = 0.8 :=
sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l1257_125781


namespace NUMINAMATH_CALUDE_parabola_equation_theorem_l1257_125750

/-- Define a parabola by its focus and directrix -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

/-- The equation of a parabola in general form -/
def parabola_equation (a b c d e f : ℤ) (x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0

/-- The given parabola -/
def given_parabola : Parabola :=
  { focus := (4, 4),
    directrix := λ x y => 4 * x + 8 * y - 32 }

/-- Theorem stating the equation of the given parabola -/
theorem parabola_equation_theorem :
  ∃ (a b c d e f : ℤ),
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | parabola_equation a b c d e f p.1 p.2} ↔ 
      (x - given_parabola.focus.1)^2 + (y - given_parabola.focus.2)^2 = 
      (given_parabola.directrix x y)^2 / (4^2 + 8^2)) ∧
    a > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1 ∧
    a = 16 ∧ b = -64 ∧ c = 64 ∧ d = -128 ∧ e = -256 ∧ f = 768 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_theorem_l1257_125750


namespace NUMINAMATH_CALUDE_factorization_equality_l1257_125761

theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1257_125761


namespace NUMINAMATH_CALUDE_log_equation_solution_l1257_125716

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1257_125716


namespace NUMINAMATH_CALUDE_twelfth_prime_l1257_125788

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem twelfth_prime :
  (nth_prime 7 = 17) → (nth_prime 12 = 37) := by sorry

end NUMINAMATH_CALUDE_twelfth_prime_l1257_125788


namespace NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l1257_125765

/-- A quadratic function f(x) = ax² + bx + c is even if and only if b = 0 -/
theorem quadratic_even_iff_b_zero (a b c : ℝ) :
  (∀ x, (a * x^2 + b * x + c) = (a * (-x)^2 + b * (-x) + c)) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l1257_125765


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1257_125711

/-- Given a principal amount and a time period of 10 years, 
    if the simple interest is 7/5 of the principal, 
    then the annual interest rate is 14%. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  (P * 14 * 10) / 100 = (7 / 5) * P := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1257_125711


namespace NUMINAMATH_CALUDE_decimal_as_fraction_l1257_125789

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.73864864864

/-- The denominator of the target fraction -/
def denominator : ℕ := 999900

/-- The theorem stating that our decimal equals the target fraction -/
theorem decimal_as_fraction : decimal = 737910 / denominator := by sorry

end NUMINAMATH_CALUDE_decimal_as_fraction_l1257_125789


namespace NUMINAMATH_CALUDE_sack_lunch_cost_l1257_125755

/-- The cost of each sack lunch for a field trip -/
theorem sack_lunch_cost (num_children : ℕ) (num_chaperones : ℕ) (num_teachers : ℕ) (num_additional : ℕ) (total_cost : ℚ) : 
  num_children = 35 →
  num_chaperones = 5 →
  num_teachers = 1 →
  num_additional = 3 →
  total_cost = 308 →
  total_cost / (num_children + num_chaperones + num_teachers + num_additional) = 7 := by
sorry

end NUMINAMATH_CALUDE_sack_lunch_cost_l1257_125755


namespace NUMINAMATH_CALUDE_project_hours_ratio_l1257_125764

/-- Represents the hours worked by each person on the project -/
structure ProjectHours where
  least : ℕ
  hardest : ℕ
  third : ℕ

/-- Checks if the given ProjectHours satisfies the problem conditions -/
def isValidProjectHours (hours : ProjectHours) : Prop :=
  hours.least + hours.hardest + hours.third = 90 ∧
  hours.hardest = hours.least + 20

/-- Theorem stating the ratio of hours worked -/
theorem project_hours_ratio :
  ∃ (hours : ProjectHours),
    isValidProjectHours hours ∧
    hours.least = 25 ∧
    hours.hardest = 45 ∧
    hours.third = 20 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_ratio_l1257_125764


namespace NUMINAMATH_CALUDE_modulus_one_minus_i_to_eight_l1257_125741

theorem modulus_one_minus_i_to_eight : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_modulus_one_minus_i_to_eight_l1257_125741


namespace NUMINAMATH_CALUDE_work_completion_time_l1257_125757

theorem work_completion_time (a_time b_time initial_days : ℝ) 
  (ha : a_time = 12)
  (hb : b_time = 6)
  (hi : initial_days = 3) :
  let a_rate := 1 / a_time
  let b_rate := 1 / b_time
  let initial_work := a_rate * initial_days
  let remaining_work := 1 - initial_work
  let combined_rate := a_rate + b_rate
  (remaining_work / combined_rate) = 3 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1257_125757


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1257_125795

theorem ferris_wheel_capacity 
  (total_people : ℕ) 
  (total_seats : ℕ) 
  (h1 : total_people = 16) 
  (h2 : total_seats = 4) 
  : total_people / total_seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1257_125795


namespace NUMINAMATH_CALUDE_ratio_equality_solution_l1257_125783

theorem ratio_equality_solution (x : ℝ) : (0.75 / x = 5 / 9) → x = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_solution_l1257_125783


namespace NUMINAMATH_CALUDE_elevator_theorem_l1257_125717

/-- Represents the elevator system described in the problem -/
structure ElevatorSystem where
  /-- The probability of moving up on the nth press is current_floor / (n-1) -/
  move_up_prob : (current_floor : ℕ) → (n : ℕ) → ℚ
  move_up_prob_def : ∀ (current_floor n : ℕ), move_up_prob current_floor n = current_floor / (n - 1)

/-- The expected number of pairs of consecutive presses that both move up -/
def expected_consecutive_up_pairs (system : ElevatorSystem) (start_press end_press : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem elevator_theorem (system : ElevatorSystem) :
  expected_consecutive_up_pairs system 3 100 = 97 / 3 := by sorry

end NUMINAMATH_CALUDE_elevator_theorem_l1257_125717


namespace NUMINAMATH_CALUDE_quadratic_inequalities_solutions_l1257_125720

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x ≤ -5 ∨ x ≥ 2}
def solution_set2 : Set ℝ := {x | (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2}

-- State the theorem
theorem quadratic_inequalities_solutions :
  (∀ x : ℝ, x^2 + 3*x - 10 ≥ 0 ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, x^2 - 3*x - 2 ≤ 0 ↔ x ∈ solution_set2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_solutions_l1257_125720


namespace NUMINAMATH_CALUDE_jade_handled_81_transactions_l1257_125775

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90

def anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10

def cal_transactions : ℕ := anthony_transactions * 2 / 3

def jade_transactions : ℕ := cal_transactions + 15

-- Theorem to prove
theorem jade_handled_81_transactions : jade_transactions = 81 := by
  sorry

end NUMINAMATH_CALUDE_jade_handled_81_transactions_l1257_125775


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l1257_125797

theorem solution_implies_k_value (k : ℝ) : 
  (k * (-3 + 4) - 2 * k - (-3) = 5) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l1257_125797


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l1257_125724

theorem right_triangle_max_ratio :
  ∀ (x y z : ℝ), 
    x > 0 → y > 0 → z > 0 →
    x^2 + y^2 = z^2 →
    (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a^2 + b^2 = c^2 → (a + 2*b) / c ≤ (x + 2*y) / z) →
    (x + 2*y) / z = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l1257_125724


namespace NUMINAMATH_CALUDE_blue_balls_unchanged_l1257_125705

/-- The number of blue balls remains unchanged when red balls are removed from a box -/
theorem blue_balls_unchanged (initial_blue : ℕ) (initial_red : ℕ) (removed_red : ℕ) :
  initial_blue = initial_blue :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_unchanged_l1257_125705


namespace NUMINAMATH_CALUDE_new_person_weight_l1257_125738

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3 →
  replaced_weight = 70 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1257_125738


namespace NUMINAMATH_CALUDE_birds_in_tree_l1257_125790

theorem birds_in_tree (initial_birds final_birds : ℕ) (h1 : initial_birds = 29) (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l1257_125790


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1257_125751

theorem min_value_of_expression (x y k : ℝ) : 
  (x * y + k)^2 + (x - y)^2 ≥ 0 ∧ 
  ∃ (x y k : ℝ), (x * y + k)^2 + (x - y)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1257_125751


namespace NUMINAMATH_CALUDE_paper_towel_pricing_l1257_125747

theorem paper_towel_pricing (case_price : ℝ) (savings_percent : ℝ) (rolls_per_case : ℕ) :
  case_price = 9 →
  savings_percent = 25 →
  rolls_per_case = 12 →
  let individual_price := case_price * (1 + savings_percent / 100) / rolls_per_case
  individual_price = 0.9375 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_pricing_l1257_125747


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1257_125721

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 6*x + 4 = 0) ∧
  (∃ x : ℝ, (3*x - 1)^2 - 4*x^2 = 0) ∧
  (∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) ∧
  (∀ x : ℝ, (3*x - 1)^2 - 4*x^2 = 0 ↔ x = 1/5 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1257_125721


namespace NUMINAMATH_CALUDE_square_sum_101_99_l1257_125777

theorem square_sum_101_99 : 101 * 101 + 99 * 99 = 20200 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_101_99_l1257_125777


namespace NUMINAMATH_CALUDE_bathroom_visits_time_calculation_l1257_125723

/-- Given that it takes 20 minutes for 8 bathroom visits, 
    prove that 6 visits will take 15 minutes. -/
theorem bathroom_visits_time_calculation 
  (total_time : ℝ) 
  (total_visits : ℕ) 
  (target_visits : ℕ) 
  (h1 : total_time = 20) 
  (h2 : total_visits = 8) 
  (h3 : target_visits = 6) : 
  (total_time / total_visits) * target_visits = 15 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_visits_time_calculation_l1257_125723


namespace NUMINAMATH_CALUDE_log_product_sqrt_equals_sqrt_two_l1257_125763

theorem log_product_sqrt_equals_sqrt_two : 
  Real.sqrt (Real.log 8 / Real.log 4 * Real.log 16 / Real.log 8) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_sqrt_equals_sqrt_two_l1257_125763


namespace NUMINAMATH_CALUDE_distribute_four_students_three_companies_l1257_125748

/-- The number of ways to distribute students among companies -/
def distribute_students (num_students : ℕ) (num_companies : ℕ) : ℕ :=
  3^4 - 3 * 2^4 + 3

/-- Theorem stating the correct number of ways to distribute 4 students among 3 companies -/
theorem distribute_four_students_three_companies :
  distribute_students 4 3 = 36 := by
  sorry

#eval distribute_students 4 3

end NUMINAMATH_CALUDE_distribute_four_students_three_companies_l1257_125748


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1257_125743

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 40, 360, and 125 -/
def product : ℕ := 40 * 360 * 125

theorem product_trailing_zeros :
  trailingZeros product = 5 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1257_125743


namespace NUMINAMATH_CALUDE_pyramid_cross_sections_l1257_125772

/-- Theorem about cross-sectional areas in a pyramid --/
theorem pyramid_cross_sections
  (S : ℝ) -- Base area of the pyramid
  (S₁ S₂ S₃ : ℝ) -- Cross-sectional areas
  (h₁ : S₁ = S / 4) -- S₁ bisects lateral edges
  (h₂ : S₂ = S / 2) -- S₂ bisects lateral surface area
  (h₃ : S₃ = S / (4 ^ (1/3))) -- S₃ bisects volume
  : S₁ < S₂ ∧ S₂ < S₃ := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cross_sections_l1257_125772


namespace NUMINAMATH_CALUDE_quadratic_sum_l1257_125776

theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 3 = a * (x - h)^2 + k) → 
  a + h + k = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1257_125776


namespace NUMINAMATH_CALUDE_monster_family_kids_l1257_125766

/-- The number of kids in the monster family -/
def num_kids : ℕ := 3

/-- The number of eyes the mom has -/
def mom_eyes : ℕ := 1

/-- The number of eyes the dad has -/
def dad_eyes : ℕ := 3

/-- The number of eyes each kid has -/
def kid_eyes : ℕ := 4

/-- The total number of eyes in the family -/
def total_eyes : ℕ := 16

theorem monster_family_kids :
  mom_eyes + dad_eyes + num_kids * kid_eyes = total_eyes :=
by sorry

end NUMINAMATH_CALUDE_monster_family_kids_l1257_125766


namespace NUMINAMATH_CALUDE_four_drivers_sufficient_l1257_125726

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Represents a driver -/
inductive Driver
| A | B | C | D

/-- Represents a trip -/
structure Trip where
  driver : Driver
  departure : Time
  arrival : Time

def one_way_duration : Time := ⟨2, 40, sorry⟩
def round_trip_duration : Time := ⟨5, 20, sorry⟩
def min_rest_duration : Time := ⟨1, 0, sorry⟩

def driver_A_return : Time := ⟨12, 40, sorry⟩
def driver_D_departure : Time := ⟨13, 5, sorry⟩
def driver_B_return : Time := ⟨16, 0, sorry⟩
def driver_A_fifth_departure : Time := ⟨16, 10, sorry⟩
def driver_B_sixth_departure : Time := ⟨17, 30, sorry⟩
def alexey_return : Time := ⟨21, 30, sorry⟩

def is_valid_schedule (trips : List Trip) : Prop :=
  sorry

theorem four_drivers_sufficient :
  ∃ (trips : List Trip),
    trips.length ≥ 6 ∧
    is_valid_schedule trips ∧
    (∃ (last_trip : Trip),
      last_trip ∈ trips ∧
      last_trip.driver = Driver.A ∧
      last_trip.departure = driver_A_fifth_departure ∧
      last_trip.arrival = alexey_return) ∧
    (∀ (trip : Trip), trip ∈ trips → trip.driver ∈ [Driver.A, Driver.B, Driver.C, Driver.D]) :=
  sorry

end NUMINAMATH_CALUDE_four_drivers_sufficient_l1257_125726


namespace NUMINAMATH_CALUDE_root_implies_difference_of_fourth_powers_l1257_125731

theorem root_implies_difference_of_fourth_powers (a b : ℝ) :
  (∃ x, x^2 - 4*a^2*b^2*x = 4 ∧ x = (a^2 + b^2)^2) →
  (a^4 - b^4 = 2 ∨ a^4 - b^4 = -2) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_difference_of_fourth_powers_l1257_125731


namespace NUMINAMATH_CALUDE_min_k_value_l1257_125784

/-- Given a line and a circle in a Cartesian coordinate system,
    prove that the minimum value of k satisfying the conditions is -√3 -/
theorem min_k_value (k : ℝ) : 
  (∃ P : ℝ × ℝ, P.2 = k * (P.1 - 3 * Real.sqrt 3)) →
  (∃ Q : ℝ × ℝ, Q.1^2 + (Q.2 - 1)^2 = 1) →
  (∃ P Q : ℝ × ℝ, P = (3 * Q.1, 3 * Q.2)) →
  -Real.sqrt 3 ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l1257_125784


namespace NUMINAMATH_CALUDE_omega_value_l1257_125793

/-- Given a function f(x) = sin(ωx) + cos(ωx) where ω > 0 and x ∈ ℝ,
    if f(x) is monotonically increasing on (-ω, ω) and
    the graph of y = f(x) is symmetric with respect to x = ω,
    then ω = √π / 2 -/
theorem omega_value (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) + Real.cos (ω * x)
  (∀ x ∈ Set.Ioo (-ω) ω, Monotone f) →
  (∀ x : ℝ, f (ω + x) = f (ω - x)) →
  ω = Real.sqrt π / 2 := by
  sorry

end NUMINAMATH_CALUDE_omega_value_l1257_125793


namespace NUMINAMATH_CALUDE_triangle_altitude_length_l1257_125733

theorem triangle_altitude_length (r : ℝ) (h : r > 0) : 
  let square_side : ℝ := 4 * r
  let square_area : ℝ := square_side ^ 2
  let diagonal_length : ℝ := square_side * Real.sqrt 2
  let triangle_area : ℝ := 2 * square_area
  let altitude : ℝ := 2 * triangle_area / diagonal_length
  altitude = 8 * r * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_length_l1257_125733


namespace NUMINAMATH_CALUDE_probability_is_9_128_l1257_125782

/-- Four points chosen uniformly at random on a circle -/
def random_points_on_circle : Type := Fin 4 → ℝ × ℝ

/-- The circle's center -/
def circle_center : ℝ × ℝ := (0, 0)

/-- Checks if three points form an obtuse triangle -/
def is_obtuse_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- The probability of no two points forming an obtuse triangle with the center -/
def probability_no_obtuse_triangle (points : random_points_on_circle) : ℝ := sorry

/-- Main theorem: The probability is 9/128 -/
theorem probability_is_9_128 :
  ∀ points : random_points_on_circle,
  probability_no_obtuse_triangle points = 9 / 128 := by sorry

end NUMINAMATH_CALUDE_probability_is_9_128_l1257_125782


namespace NUMINAMATH_CALUDE_cos_18_deg_root_l1257_125768

theorem cos_18_deg_root : ∃ (p : ℝ → ℝ), (∀ x, p x = 16 * x^4 - 20 * x^2 + 5) ∧ p (Real.cos (18 * Real.pi / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_deg_root_l1257_125768


namespace NUMINAMATH_CALUDE_afternoon_shells_l1257_125792

/-- Given that Lino picked up 292 shells in the morning and a total of 616 shells,
    prove that he picked up 324 shells in the afternoon. -/
theorem afternoon_shells (morning_shells : ℕ) (total_shells : ℕ) (h1 : morning_shells = 292) (h2 : total_shells = 616) :
  total_shells - morning_shells = 324 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_shells_l1257_125792


namespace NUMINAMATH_CALUDE_test_completion_ways_l1257_125774

/-- The number of questions in the test -/
def num_questions : ℕ := 8

/-- The number of answer choices for each question -/
def num_choices : ℕ := 7

/-- The total number of options for each question (including unanswered) -/
def total_options : ℕ := num_choices + 1

/-- The theorem stating the total number of ways to complete the test -/
theorem test_completion_ways :
  (total_options : ℕ) ^ num_questions = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_test_completion_ways_l1257_125774


namespace NUMINAMATH_CALUDE_train_length_l1257_125778

/-- Given a train traveling at 45 kmph that passes a 140 m long bridge in 40 seconds,
    prove that the length of the train is 360 m. -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (time : ℝ) (train_length : ℝ) : 
  speed = 45 → bridge_length = 140 → time = 40 → 
  train_length = (speed * 1000 / 3600 * time) - bridge_length → 
  train_length = 360 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1257_125778


namespace NUMINAMATH_CALUDE_jennifer_apples_l1257_125728

def initial_apples : ℕ := 7
def hours : ℕ := 3
def multiply_factor : ℕ := 3
def additional_apples : ℕ := 74

def apples_after_tripling (start : ℕ) (hours : ℕ) (factor : ℕ) : ℕ :=
  start * (factor ^ hours)

theorem jennifer_apples : 
  apples_after_tripling initial_apples hours multiply_factor + additional_apples = 263 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_apples_l1257_125728


namespace NUMINAMATH_CALUDE_radio_cost_price_l1257_125756

/-- Proves that the cost price of a radio is 2400 given the selling price and loss percentage --/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 2100 → loss_percentage = 12.5 → 
  ∃ (cost_price : ℝ), cost_price = 2400 ∧ selling_price = cost_price * (1 - loss_percentage / 100) :=
by
  sorry

#check radio_cost_price

end NUMINAMATH_CALUDE_radio_cost_price_l1257_125756


namespace NUMINAMATH_CALUDE_evaluate_expression_l1257_125700

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) : y * (y - 2 * x + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1257_125700


namespace NUMINAMATH_CALUDE_product_inequality_l1257_125798

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1257_125798


namespace NUMINAMATH_CALUDE_sqrt_nested_equality_l1257_125713

theorem sqrt_nested_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt x)) = (x^7)^(1/8) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_equality_l1257_125713


namespace NUMINAMATH_CALUDE_square_division_perimeter_l1257_125780

theorem square_division_perimeter 
  (original_perimeter : ℝ) 
  (h_original_perimeter : original_perimeter = 200) : 
  ∃ (smaller_square_perimeter : ℝ), 
    smaller_square_perimeter = 100 ∧
    ∃ (original_side : ℝ), 
      4 * original_side = original_perimeter ∧
      ∃ (rectangle_width rectangle_height : ℝ),
        rectangle_width = original_side ∧
        rectangle_height = original_side / 2 ∧
        smaller_square_perimeter = 4 * rectangle_height :=
by sorry

end NUMINAMATH_CALUDE_square_division_perimeter_l1257_125780


namespace NUMINAMATH_CALUDE_product_equality_proof_l1257_125744

theorem product_equality_proof : ∃! X : ℕ, 865 * 48 = X * 240 ∧ X = 173 := by sorry

end NUMINAMATH_CALUDE_product_equality_proof_l1257_125744


namespace NUMINAMATH_CALUDE_triangle_345_is_acute_l1257_125796

/-- A triangle with sides 3, 4, and 4.5 is acute. -/
theorem triangle_345_is_acute : 
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 4.5 → 
  (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_345_is_acute_l1257_125796


namespace NUMINAMATH_CALUDE_cube_sum_equals_negative_eighteen_l1257_125739

theorem cube_sum_equals_negative_eighteen
  (a b c : ℝ)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_equals_negative_eighteen_l1257_125739


namespace NUMINAMATH_CALUDE_savings_calculation_l1257_125710

/-- Given a person's income and the ratio of income to expenditure, calculate their savings. -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given an income of 15000 and an income to expenditure ratio of 5:4, the savings are 3000. -/
theorem savings_calculation :
  calculate_savings 15000 5 4 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1257_125710


namespace NUMINAMATH_CALUDE_feet_in_garden_l1257_125758

/-- The number of feet in the garden --/
def total_feet (num_dogs num_ducks num_cats num_birds num_insects : ℕ) : ℕ :=
  num_dogs * 4 + num_ducks * 2 + num_cats * 4 + num_birds * 2 + num_insects * 6

/-- Theorem stating that the total number of feet in the garden is 118 --/
theorem feet_in_garden : total_feet 6 2 4 7 10 = 118 := by
  sorry

end NUMINAMATH_CALUDE_feet_in_garden_l1257_125758


namespace NUMINAMATH_CALUDE_greatest_number_with_gcd_l1257_125786

theorem greatest_number_with_gcd (X : ℕ) : 
  X ≤ 840 ∧ 
  7 ∣ X ∧ 
  Nat.gcd X 91 = 7 ∧ 
  Nat.gcd X 840 = 7 →
  X = 840 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_gcd_l1257_125786


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1257_125794

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x + 1 < 4 ∧ 1 - 3*x ≥ -5

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x ≤ 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1257_125794


namespace NUMINAMATH_CALUDE_expression_equality_l1257_125709

theorem expression_equality : (1 + 0.25) / (2 * (3/4) - 0.75) + (3 * 0.5) / (1.5 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1257_125709


namespace NUMINAMATH_CALUDE_solve_complex_equation_l1257_125703

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1+i)Z = 2
def equation (Z : ℂ) : Prop := (1 + i) * Z = 2

-- Theorem statement
theorem solve_complex_equation :
  ∀ Z : ℂ, equation Z → Z = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l1257_125703


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l1257_125707

theorem complex_number_imaginary_part (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 1 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l1257_125707


namespace NUMINAMATH_CALUDE_flowerbed_perimeter_l1257_125736

/-- A rectangular flowerbed with given dimensions --/
structure Flowerbed where
  width : ℝ
  length : ℝ

/-- The perimeter of a rectangular flowerbed --/
def perimeter (f : Flowerbed) : ℝ := 2 * (f.length + f.width)

/-- Theorem: The perimeter of the specific flowerbed is 22 meters --/
theorem flowerbed_perimeter :
  ∃ (f : Flowerbed), f.width = 4 ∧ f.length = 2 * f.width - 1 ∧ perimeter f = 22 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_perimeter_l1257_125736


namespace NUMINAMATH_CALUDE_kath_group_cost_l1257_125708

/-- Calculates the total cost for a group watching a movie with early showing discount -/
def total_cost (standard_price : ℕ) (discount : ℕ) (group_size : ℕ) : ℕ :=
  (standard_price - discount) * group_size

/-- Theorem: The total cost for Kath's group is $30 -/
theorem kath_group_cost :
  let standard_price : ℕ := 8
  let early_discount : ℕ := 3
  let group_size : ℕ := 6
  total_cost standard_price early_discount group_size = 30 := by
  sorry

end NUMINAMATH_CALUDE_kath_group_cost_l1257_125708


namespace NUMINAMATH_CALUDE_expression_simplification_l1257_125742

theorem expression_simplification (a : ℝ) (h : a = 2023) :
  (a^2 - 6*a + 9) / (a^2 - 2*a) / (1 - 1/(a - 2)) = 2020 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1257_125742


namespace NUMINAMATH_CALUDE_equation_solution_l1257_125771

theorem equation_solution : 
  let n : ℝ := 73.0434782609
  0.07 * n + 0.12 * (30 + n) + 0.04 * n = 20.4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1257_125771


namespace NUMINAMATH_CALUDE_inverse_g_75_l1257_125722

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 - 6

-- State the theorem
theorem inverse_g_75 : g⁻¹ 75 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_75_l1257_125722


namespace NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l1257_125753

/-- The probability of success for each event -/
def p : ℝ := 0.7

/-- The probability of at least one success in two independent events -/
def prob_at_least_one (p : ℝ) : ℝ := 1 - (1 - p) * (1 - p)

/-- Theorem stating that the probability of at least one success is 0.91 -/
theorem prob_at_least_one_is_correct : prob_at_least_one p = 0.91 := by
  sorry

#eval prob_at_least_one p

end NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l1257_125753


namespace NUMINAMATH_CALUDE_a_power_sum_l1257_125706

theorem a_power_sum (a x : ℝ) (ha : a > 0) (hx : a^(x/2) + a^(-x/2) = 5) : 
  a^x + a^(-x) = 23 := by
sorry

end NUMINAMATH_CALUDE_a_power_sum_l1257_125706
