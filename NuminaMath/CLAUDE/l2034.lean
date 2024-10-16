import Mathlib

namespace NUMINAMATH_CALUDE_child_ticket_cost_l2034_203454

/-- Proves that the cost of a child ticket is $5 given the theater conditions --/
theorem child_ticket_cost (total_seats : ℕ) (adult_price : ℕ) (child_tickets : ℕ) (total_revenue : ℕ) :
  total_seats = 80 →
  adult_price = 12 →
  child_tickets = 63 →
  total_revenue = 519 →
  ∃ (child_price : ℕ), 
    child_price = 5 ∧
    total_revenue = (total_seats - child_tickets) * adult_price + child_tickets * child_price :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2034_203454


namespace NUMINAMATH_CALUDE_ball_count_theorem_l2034_203484

/-- Represents the count of balls of each color in a jar. -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball count satisfies the 4:3:2 ratio. -/
def satisfiesRatio (bc : BallCount) : Prop :=
  3 * bc.white = 4 * bc.red ∧ 2 * bc.white = 4 * bc.blue

theorem ball_count_theorem (bc : BallCount) 
    (h_ratio : satisfiesRatio bc) (h_white : bc.white = 20) : 
    bc.red = 15 ∧ bc.blue = 10 := by
  sorry

#check ball_count_theorem

end NUMINAMATH_CALUDE_ball_count_theorem_l2034_203484


namespace NUMINAMATH_CALUDE_fraction_simplification_l2034_203471

theorem fraction_simplification : (5 * (8 + 2)) / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2034_203471


namespace NUMINAMATH_CALUDE_square_equals_self_l2034_203412

theorem square_equals_self (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_square_equals_self_l2034_203412


namespace NUMINAMATH_CALUDE_sin_pi_six_l2034_203477

theorem sin_pi_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_six_l2034_203477


namespace NUMINAMATH_CALUDE_train_length_l2034_203446

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 7 → speed_kmh * (1000 / 3600) * time_s = 350 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2034_203446


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2034_203403

/-- The equation of the tangent line to y = x^3 + 2x at (1, 3) is 5x - y - 2 = 0 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := x^3 + 2*x
  let P : ℝ × ℝ := (1, 3)
  ∃ (m b : ℝ), 
    (∀ x y, y = m*x + b ↔ m*x - y + b = 0) ∧ 
    (f P.1 = P.2) ∧
    (∀ x, x ≠ P.1 → (f x - P.2) / (x - P.1) ≠ m) ∧
    m*P.1 - P.2 + b = 0 ∧
    m = 5 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2034_203403


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l2034_203422

def a (n : ℕ+) : ℤ := n^2 - 3*n - 4

theorem a_4_equals_zero : a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l2034_203422


namespace NUMINAMATH_CALUDE_simulation_needed_for_exact_probability_l2034_203440

structure Player where
  money : Nat

structure GameState where
  players : List Player

def initial_state : GameState :=
  { players := [{ money := 2 }, { money := 2 }, { money := 2 }] }

def can_give_money (p : Player) : Bool :=
  p.money > 1

def ring_bell (state : GameState) : GameState :=
  sorry

def is_final_state (state : GameState) : Bool :=
  state.players.all (fun p => p.money = 2)

noncomputable def probability_of_final_state (num_rings : Nat) : ℝ :=
  sorry

theorem simulation_needed_for_exact_probability :
  ∀ (analytical_function : Nat → ℝ),
    ∃ (ε : ℝ), ε > 0 ∧
      |probability_of_final_state 2019 - analytical_function 2019| > ε :=
by sorry

end NUMINAMATH_CALUDE_simulation_needed_for_exact_probability_l2034_203440


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2034_203425

theorem sin_2theta_value (θ : ℝ) (h : Real.cos (π/4 - θ) = 1/2) : 
  Real.sin (2*θ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2034_203425


namespace NUMINAMATH_CALUDE_eight_pencils_l2034_203404

/-- Represents Sam's pen and pencil collection -/
structure SamsCollection where
  pencils : ℕ
  blue_pens : ℕ
  black_pens : ℕ
  red_pens : ℕ

/-- The conditions of Sam's collection -/
def valid_collection (c : SamsCollection) : Prop :=
  c.black_pens = c.blue_pens + 10 ∧
  c.blue_pens = 2 * c.pencils ∧
  c.red_pens = c.pencils - 2 ∧
  c.black_pens + c.blue_pens + c.red_pens = 48

/-- Theorem stating that in a valid collection, there are 8 pencils -/
theorem eight_pencils (c : SamsCollection) (h : valid_collection c) : c.pencils = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_pencils_l2034_203404


namespace NUMINAMATH_CALUDE_triangle_angles_from_exterior_ratio_l2034_203492

/-- Proves that a triangle with exterior angles in the ratio 12:13:15 has interior angles of 45°, 63°, and 72° -/
theorem triangle_angles_from_exterior_ratio :
  ∀ (E₁ E₂ E₃ : ℝ),
  E₁ > 0 ∧ E₂ > 0 ∧ E₃ > 0 →
  E₁ / 12 = E₂ / 13 ∧ E₂ / 13 = E₃ / 15 →
  E₁ + E₂ + E₃ = 360 →
  ∃ (I₁ I₂ I₃ : ℝ),
    I₁ = 180 - E₁ ∧
    I₂ = 180 - E₂ ∧
    I₃ = 180 - E₃ ∧
    I₁ + I₂ + I₃ = 180 ∧
    I₁ = 45 ∧ I₂ = 63 ∧ I₃ = 72 :=
by sorry


end NUMINAMATH_CALUDE_triangle_angles_from_exterior_ratio_l2034_203492


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2034_203413

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 - 3*x^11 + 2*x^9 - x^6) =
  15*x^13 - 19*x^12 - 6*x^11 + 6*x^10 - 4*x^9 - 3*x^7 + 2*x^6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2034_203413


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l2034_203442

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧ 
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧ 
  (x₁ ≠ x₂) ∧ 
  (|x₁ - x₂| = 14) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l2034_203442


namespace NUMINAMATH_CALUDE_triangle_problem_l2034_203453

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > c →
  b = 3 →
  (a * c : ℝ) * (1 / 3 : ℝ) = 2 →  -- This represents BA · BC = 2 and cos B = 1/3
  a^2 + c^2 = b^2 + 2 * (a * c : ℝ) * (1 / 3 : ℝ) →  -- Law of cosines
  (a = 3 ∧ c = 2) ∧ 
  Real.cos (B - C) = 23 / 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2034_203453


namespace NUMINAMATH_CALUDE_product_upper_bound_l2034_203421

theorem product_upper_bound (x y z t : ℝ) 
  (h_order : x ≤ y ∧ y ≤ z ∧ z ≤ t) 
  (h_sum : x*y + x*z + x*t + y*z + y*t + z*t = 1) : 
  x*t < 1/3 ∧ ∀ C, (∀ a b c d, a ≤ b ∧ b ≤ c ∧ c ≤ d → 
    a*b + a*c + a*d + b*c + b*d + c*d = 1 → a*d < C) → 1/3 ≤ C :=
by sorry

end NUMINAMATH_CALUDE_product_upper_bound_l2034_203421


namespace NUMINAMATH_CALUDE_inequality_range_l2034_203491

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2034_203491


namespace NUMINAMATH_CALUDE_ellipse_parabola_line_equations_l2034_203473

/-- Given an ellipse and a parabola with specific properties, prove the equations of both curves and a line. -/
theorem ellipse_parabola_line_equations :
  ∀ (a b c p : ℝ) (F A B P Q D : ℝ × ℝ),
  a > 0 → b > 0 → a > b →
  c / a = 1 / 2 →
  A.1 - F.1 = a →
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 →
  ∀ (x y : ℝ), y^2 = 2 * p * x →
  A.1 - F.1 = 1 / 2 →
  P.1 = Q.1 ∧ P.2 = -Q.2 →
  B ≠ A →
  D.2 = 0 →
  abs ((A.1 - P.1) * (D.2 - P.2) - (A.2 - P.2) * (D.1 - P.1)) / 2 = Real.sqrt 6 / 2 →
  ((∀ (x y : ℝ), x^2 + 4 * y^2 / 3 = 1) ∧
   (∀ (x y : ℝ), y^2 = 4 * x) ∧
   ((3 * P.1 + Real.sqrt 6 * P.2 - 3 = 0) ∨ (3 * P.1 - Real.sqrt 6 * P.2 - 3 = 0))) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_line_equations_l2034_203473


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eleven_l2034_203481

/-- The repeating decimal 0.4567 as a rational number -/
def repeating_decimal : ℚ := 4567 / 9999

theorem product_of_repeating_decimal_and_eleven :
  11 * repeating_decimal = 50237 / 9999 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eleven_l2034_203481


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2034_203432

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem f_derivative_at_one : 
  (deriv f) 1 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2034_203432


namespace NUMINAMATH_CALUDE_single_interval_condition_l2034_203451

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Condition for single interval solution of [x]^2 + k[x] + l = 0 -/
theorem single_interval_condition (k l : ℤ) : 
  (∃ (a b : ℝ), ∀ x, (floor x)^2 + k * (floor x) + l = 0 ↔ a ≤ x ∧ x < b) ↔ 
  l = floor ((k^2 : ℝ) / 4) :=
sorry

end NUMINAMATH_CALUDE_single_interval_condition_l2034_203451


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_60_l2034_203456

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product_of_60 (a b : ℕ) :
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 60 →
  is_factor b 60 →
  ¬ is_factor (a * b) 60 →
  ∀ c d : ℕ, c ≠ d → c > 0 → d > 0 → is_factor c 60 → is_factor d 60 → ¬ is_factor (c * d) 60 → a * b ≤ c * d :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_60_l2034_203456


namespace NUMINAMATH_CALUDE_solve_equation_l2034_203449

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l2034_203449


namespace NUMINAMATH_CALUDE_garden_ratio_l2034_203483

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Theorem: For a rectangular garden with area 675 sq meters and width 15 meters, 
    the ratio of length to width is 3:1 -/
theorem garden_ratio (g : RectangularGarden) 
    (h1 : g.area = 675)
    (h2 : g.width = 15)
    (h3 : g.area = g.length * g.width) :
  g.length / g.width = 3 := by
  sorry

#check garden_ratio

end NUMINAMATH_CALUDE_garden_ratio_l2034_203483


namespace NUMINAMATH_CALUDE_marbles_problem_l2034_203496

theorem marbles_problem (fabian kyle miles : ℕ) : 
  fabian = 36 ∧ 
  fabian = 4 * kyle ∧ 
  fabian = 9 * miles → 
  kyle + miles = 13 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l2034_203496


namespace NUMINAMATH_CALUDE_danny_wrappers_found_l2034_203405

/-- Represents the number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- Represents the number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- Represents the difference between bottle caps and wrappers found -/
def difference : ℕ := 4

theorem danny_wrappers_found :
  bottle_caps_found = wrappers_found + difference →
  wrappers_found = 46 := by
  sorry

end NUMINAMATH_CALUDE_danny_wrappers_found_l2034_203405


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2034_203420

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (x + 2) * (x^2 - 5*a*x + 1) = x^3 + (-9*a)*x + 2) → a = 2/5 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2034_203420


namespace NUMINAMATH_CALUDE_money_distribution_l2034_203447

/-- Given three people A, B, and C with money amounts a, b, and c respectively,
    if their total amount is 500, B and C together have 310, and C has 10,
    then A and C together have 200. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → b + c = 310 → c = 10 → a + c = 200 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l2034_203447


namespace NUMINAMATH_CALUDE_det_A_eq_48_l2034_203463

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]

theorem det_A_eq_48 : Matrix.det A = 48 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_48_l2034_203463


namespace NUMINAMATH_CALUDE_burger_lovers_l2034_203408

theorem burger_lovers (total : ℕ) (fries : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 25 → 
  fries = 15 → 
  both = 6 → 
  neither = 6 → 
  ∃ burgers : ℕ, burgers = 10 ∧ 
    total = fries + burgers - both + neither :=
by sorry

end NUMINAMATH_CALUDE_burger_lovers_l2034_203408


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2034_203433

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x^2 - 6*x + 8 = (x - r₁) * (x - r₂)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2034_203433


namespace NUMINAMATH_CALUDE_probability_equals_three_elevenths_l2034_203465

/-- A quadruple of non-negative integers satisfying 2p + q + r + s = 4 -/
def ValidQuadruple : Type := 
  { quad : Fin 4 → ℕ // 2 * quad 0 + quad 1 + quad 2 + quad 3 = 4 }

/-- The set of all valid quadruples -/
def AllQuadruples : Finset ValidQuadruple := sorry

/-- The set of quadruples satisfying p + q + r + s = 3 -/
def SatisfyingQuadruples : Finset ValidQuadruple :=
  AllQuadruples.filter (fun quad => quad.val 0 + quad.val 1 + quad.val 2 + quad.val 3 = 3)

theorem probability_equals_three_elevenths :
  Nat.card SatisfyingQuadruples / Nat.card AllQuadruples = 3 / 11 := by sorry

end NUMINAMATH_CALUDE_probability_equals_three_elevenths_l2034_203465


namespace NUMINAMATH_CALUDE_simplification_fraction_l2034_203415

theorem simplification_fraction (k : ℝ) :
  ∃ (c d : ℤ), (6 * k + 12 + 3) / 3 = c * k + d ∧ (c : ℚ) / d = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplification_fraction_l2034_203415


namespace NUMINAMATH_CALUDE_square_sum_difference_l2034_203476

theorem square_sum_difference (n : ℕ) : 
  (2*n + 1)^2 - (2*n - 1)^2 + (2*n - 1)^2 - (2*n - 3)^2 + (2*n - 3)^2 - (2*n - 5)^2 + 
  (2*n - 5)^2 - (2*n - 7)^2 + (2*n - 7)^2 - (2*n - 9)^2 + (2*n - 9)^2 - (2*n - 11)^2 = 288 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_difference_l2034_203476


namespace NUMINAMATH_CALUDE_expected_value_special_coin_l2034_203469

/-- The expected value of winnings for a special coin flip -/
theorem expected_value_special_coin : 
  let p_heads : ℚ := 2 / 5
  let p_tails : ℚ := 3 / 5
  let win_heads : ℚ := 4
  let lose_tails : ℚ := 3
  p_heads * win_heads - p_tails * lose_tails = -1 / 5 := by
sorry

end NUMINAMATH_CALUDE_expected_value_special_coin_l2034_203469


namespace NUMINAMATH_CALUDE_intersection_M_N_l2034_203460

def M : Set ℝ := {0, 1, 2}

def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2034_203460


namespace NUMINAMATH_CALUDE_total_students_l2034_203457

/-- Proves that in a college with a boy-to-girl ratio of 8:5 and 400 girls, the total number of students is 1040 -/
theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 400) : boys + girls = 1040 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l2034_203457


namespace NUMINAMATH_CALUDE_perpendicular_distance_is_four_l2034_203452

/-- A rectangular parallelepiped with vertices H, E, F, and G -/
structure Parallelepiped where
  H : ℝ × ℝ × ℝ
  E : ℝ × ℝ × ℝ
  F : ℝ × ℝ × ℝ
  G : ℝ × ℝ × ℝ

/-- The specific parallelepiped described in the problem -/
def specificParallelepiped : Parallelepiped :=
  { H := (0, 0, 0)
    E := (5, 0, 0)
    F := (0, 6, 0)
    G := (0, 0, 4) }

/-- The perpendicular distance from a point to a plane -/
noncomputable def perpendicularDistance (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The plane containing points E, F, and G -/
def planEFG (p : Parallelepiped) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The theorem to be proved -/
theorem perpendicular_distance_is_four :
  perpendicularDistance specificParallelepiped.H (planEFG specificParallelepiped) = 4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_distance_is_four_l2034_203452


namespace NUMINAMATH_CALUDE_max_circles_in_square_l2034_203468

/-- The maximum number of non-overlapping circles with radius 2 cm
    that can fit inside a square with side length 8 cm -/
def max_circles : ℕ := 4

/-- The side length of the square in cm -/
def square_side : ℝ := 8

/-- The radius of each circle in cm -/
def circle_radius : ℝ := 2

theorem max_circles_in_square :
  ∀ n : ℕ,
  (n : ℝ) * (2 * circle_radius) ≤ square_side →
  (n : ℝ) * (2 * circle_radius) > square_side - 2 * circle_radius →
  n * n = max_circles :=
by sorry

end NUMINAMATH_CALUDE_max_circles_in_square_l2034_203468


namespace NUMINAMATH_CALUDE_parabola_directrix_l2034_203417

/-- Given a parabola y = 3x^2 - 6x + 2, its directrix is y = -13/12 -/
theorem parabola_directrix (x y : ℝ) :
  y = 3 * x^2 - 6 * x + 2 →
  ∃ (k : ℝ), k = -13/12 ∧ k = y - 3 * (x - 1)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2034_203417


namespace NUMINAMATH_CALUDE_range_symmetric_range_b_decreasing_l2034_203459

-- Define the function f
def f (a b x : ℝ) : ℝ := -2 * x^2 + a * x + b

-- Theorem for part (1)
theorem range_symmetric (a b : ℝ) :
  f a b 2 = -3 →
  (∀ x : ℝ, f a b (1 + x) = f a b (1 - x)) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 3 → f a b x ∈ Set.Icc (-19) (-1)) :=
sorry

-- Theorem for part (2)
theorem range_b_decreasing (a b : ℝ) :
  f a b 2 = -3 →
  (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f a b x ≥ f a b y) →
  b ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_range_symmetric_range_b_decreasing_l2034_203459


namespace NUMINAMATH_CALUDE_all_hungarian_teams_face_foreign_l2034_203427

-- Define the total number of teams
def total_teams : ℕ := 8

-- Define the number of Hungarian teams
def hungarian_teams : ℕ := 3

-- Define the number of foreign teams
def foreign_teams : ℕ := total_teams - hungarian_teams

-- Define the probability of all Hungarian teams facing foreign opponents
def prob_all_hungarian_foreign : ℚ := 4/7

-- Theorem statement
theorem all_hungarian_teams_face_foreign :
  (foreign_teams.choose hungarian_teams * hungarian_teams.factorial) / 
  (total_teams.choose 2 * (total_teams / 2).factorial) = prob_all_hungarian_foreign := by
  sorry

end NUMINAMATH_CALUDE_all_hungarian_teams_face_foreign_l2034_203427


namespace NUMINAMATH_CALUDE_farm_count_solution_l2034_203439

/-- Represents the count of animals in a farm -/
structure FarmCount where
  hens : ℕ
  cows : ℕ

/-- Checks if the given farm count satisfies the conditions -/
def isValidFarmCount (f : FarmCount) : Prop :=
  f.hens + f.cows = 46 ∧ 2 * f.hens + 4 * f.cows = 140

/-- Theorem stating that the farm with 22 hens satisfies the conditions -/
theorem farm_count_solution :
  ∃ (f : FarmCount), isValidFarmCount f ∧ f.hens = 22 := by
  sorry

#check farm_count_solution

end NUMINAMATH_CALUDE_farm_count_solution_l2034_203439


namespace NUMINAMATH_CALUDE_cricket_ratio_l2034_203445

/-- Represents the number of crickets Spike hunts in the morning -/
def morning_crickets : ℕ := 5

/-- Represents the total number of crickets Spike hunts per day -/
def total_crickets : ℕ := 20

/-- Represents the number of crickets Spike hunts in the afternoon and evening -/
def afternoon_evening_crickets : ℕ := total_crickets - morning_crickets

/-- The theorem stating the ratio of crickets hunted in the afternoon and evening to morning -/
theorem cricket_ratio : 
  afternoon_evening_crickets / morning_crickets = 3 :=
sorry

end NUMINAMATH_CALUDE_cricket_ratio_l2034_203445


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_existence_of_m_outside_interval_l2034_203487

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 - m*x + 1 > 0) ↔ m < 2 :=
by sorry

theorem existence_of_m_outside_interval :
  ∃ m : ℝ, (m ≤ -2 ∨ m ≥ 2) ∧ (∀ x : ℝ, x > 1 → x^2 - m*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_existence_of_m_outside_interval_l2034_203487


namespace NUMINAMATH_CALUDE_bruce_triple_age_in_six_years_l2034_203486

/-- The number of years it will take for Bruce to be three times as old as his son -/
def years_until_triple_age (bruce_age : ℕ) (son_age : ℕ) : ℕ :=
  let x : ℕ := (bruce_age - 3 * son_age) / 2
  x

/-- Theorem stating that it will take 6 years for Bruce to be three times as old as his son -/
theorem bruce_triple_age_in_six_years :
  years_until_triple_age 36 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bruce_triple_age_in_six_years_l2034_203486


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_2109_l2034_203479

theorem no_two_digit_factors_of_2109 : 
  ¬∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 2109 :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_2109_l2034_203479


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2034_203461

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 7/11)
  (h_seventeenth : a 17 = 5/6) :
  a 9 = 97/132 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2034_203461


namespace NUMINAMATH_CALUDE_watch_selling_price_l2034_203444

/-- Calculates the selling price of an item given its cost price and profit percentage. -/
def selling_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (cost_price * profit_percentage) / 100

/-- Proves that for a watch with a cost price of 90 rupees, 
    if the profit percentage is equal to the cost price, 
    then the selling price is 180 rupees. -/
theorem watch_selling_price : 
  let cost_price : ℕ := 90
  let profit_percentage : ℕ := 100
  selling_price cost_price profit_percentage = 180 := by
sorry


end NUMINAMATH_CALUDE_watch_selling_price_l2034_203444


namespace NUMINAMATH_CALUDE_intersection_empty_union_equals_A_l2034_203423

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*a*x + 3*a^2 = 0}

-- Theorem for part (1)
theorem intersection_empty (a : ℝ) : 
  A ∩ B a = ∅ ↔ a ≤ -3 ∨ a ≥ 4 :=
sorry

-- Theorem for part (2)
theorem union_equals_A (a : ℝ) :
  A ∪ B a = A ↔ -1 < a ∧ a < 4/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_union_equals_A_l2034_203423


namespace NUMINAMATH_CALUDE_solution_set_is_positive_reals_l2034_203428

open Set
open Function
open Real

noncomputable section

variables {f : ℝ → ℝ} (hf : Differentiable ℝ f)

def condition_1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + deriv f x > 1

def condition_2 (f : ℝ → ℝ) : Prop :=
  f 0 = 2018

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | Real.exp x * f x - Real.exp x > 2017}

theorem solution_set_is_positive_reals
  (h1 : condition_1 f) (h2 : condition_2 f) :
  solution_set f = Ioi 0 :=
sorry

end

end NUMINAMATH_CALUDE_solution_set_is_positive_reals_l2034_203428


namespace NUMINAMATH_CALUDE_otimes_four_two_l2034_203480

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

-- Theorem statement
theorem otimes_four_two : otimes 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_two_l2034_203480


namespace NUMINAMATH_CALUDE_initial_breads_l2034_203466

/-- The number of thieves -/
def num_thieves : ℕ := 8

/-- The number of breads remaining after all thieves -/
def remaining_breads : ℕ := 5

/-- The function representing how many breads remain after each thief -/
def breads_after_thief (n : ℕ) (b : ℚ) : ℚ :=
  if n = 0 then b else (1/2) * (breads_after_thief (n-1) b) - (1/2)

/-- The theorem stating the initial number of breads -/
theorem initial_breads :
  ∃ (b : ℚ), breads_after_thief num_thieves b = remaining_breads ∧ b = 1535 := by
  sorry

end NUMINAMATH_CALUDE_initial_breads_l2034_203466


namespace NUMINAMATH_CALUDE_tunnel_volume_calculation_l2034_203409

/-- The volume of a tunnel with a trapezoidal cross-section -/
def tunnel_volume (top_width bottom_width cross_section_area length : ℝ) : ℝ :=
  cross_section_area * length

/-- Theorem stating the volume of the tunnel under given conditions -/
theorem tunnel_volume_calculation
  (top_width : ℝ)
  (bottom_width : ℝ)
  (cross_section_area : ℝ)
  (length : ℝ)
  (h_top : top_width = 15)
  (h_bottom : bottom_width = 5)
  (h_area : cross_section_area = 400)
  (h_length : length = 300) :
  tunnel_volume top_width bottom_width cross_section_area length = 120000 := by
  sorry


end NUMINAMATH_CALUDE_tunnel_volume_calculation_l2034_203409


namespace NUMINAMATH_CALUDE_orange_juice_orders_l2034_203416

/-- Proves that the number of members who ordered orange juice is 12 --/
theorem orange_juice_orders (total_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : ∃ lemon_orders : ℕ, lemon_orders = (2 : ℕ) * total_members / (5 : ℕ))
  (h3 : ∃ remaining : ℕ, remaining = total_members - (2 : ℕ) * total_members / (5 : ℕ))
  (h4 : ∃ mango_orders : ℕ, mango_orders = remaining / (3 : ℕ))
  (h5 : ∃ orange_orders : ℕ, orange_orders = total_members - ((2 : ℕ) * total_members / (5 : ℕ) + remaining / (3 : ℕ))) :
  orange_orders = 12 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_orders_l2034_203416


namespace NUMINAMATH_CALUDE_kth_level_associated_point_coordinates_l2034_203467

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the k-th level associated point -/
def kth_level_associated_point (A : Point) (k : ℝ) : Point :=
  { x := k * A.x + A.y,
    y := A.x + k * A.y }

/-- Theorem: The k-th level associated point B of A(x,y) has coordinates (kx+y, x+ky) -/
theorem kth_level_associated_point_coordinates (A : Point) (k : ℝ) (h : k ≠ 0) :
  let B := kth_level_associated_point A k
  B.x = k * A.x + A.y ∧ B.y = A.x + k * A.y :=
by sorry

end NUMINAMATH_CALUDE_kth_level_associated_point_coordinates_l2034_203467


namespace NUMINAMATH_CALUDE_b_95_mod_49_l2034_203485

def b (n : ℕ) : ℕ := 5^n + 7^n

theorem b_95_mod_49 : b 95 ≡ 42 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l2034_203485


namespace NUMINAMATH_CALUDE_set_operations_l2034_203419

def A : Set ℕ := {6, 8, 10, 12}
def B : Set ℕ := {1, 6, 8}

theorem set_operations :
  (A ∪ B = {1, 6, 8, 10, 12}) ∧
  (𝒫(A ∩ B) = {∅, {6}, {8}, {6, 8}}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2034_203419


namespace NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l2034_203430

/-- The length of the real axis of a hyperbola given by the equation 2x^2 - y^2 = 8 -/
def real_axis_length : ℝ := 4

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

theorem real_axis_length_of_hyperbola :
  ∀ x y : ℝ, hyperbola_equation x y → real_axis_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l2034_203430


namespace NUMINAMATH_CALUDE_fan_sales_theorem_l2034_203407

/-- Represents the sales data for a week -/
structure WeeklySales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the fan models and their properties -/
structure FanModels where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceA : ℕ
  sellingPriceB : ℕ

def totalUnits : ℕ := 40
def maxBudget : ℕ := 7650

def weekOneSales : WeeklySales := {
  modelA := 3
  modelB := 5
  revenue := 2150
}

def weekTwoSales : WeeklySales := {
  modelA := 4
  modelB := 10
  revenue := 3700
}

def fanModels : FanModels := {
  purchasePriceA := 210
  purchasePriceB := 180
  sellingPriceA := 300
  sellingPriceB := 250
}

theorem fan_sales_theorem (w1 : WeeklySales) (w2 : WeeklySales) (f : FanModels) :
  w1 = weekOneSales ∧ w2 = weekTwoSales ∧ f.purchasePriceA = 210 ∧ f.purchasePriceB = 180 →
  (f.sellingPriceA * w1.modelA + f.sellingPriceB * w1.modelB = w1.revenue ∧
   f.sellingPriceA * w2.modelA + f.sellingPriceB * w2.modelB = w2.revenue) →
  f.sellingPriceA = 300 ∧ f.sellingPriceB = 250 ∧
  (∀ a : ℕ, a ≤ totalUnits →
    f.purchasePriceA * a + f.purchasePriceB * (totalUnits - a) ≤ maxBudget →
    (f.sellingPriceA - f.purchasePriceA) * a + (f.sellingPriceB - f.purchasePriceB) * (totalUnits - a) ≤ 3100) ∧
  ∃ a : ℕ, a ≤ totalUnits ∧
    f.purchasePriceA * a + f.purchasePriceB * (totalUnits - a) ≤ maxBudget ∧
    (f.sellingPriceA - f.purchasePriceA) * a + (f.sellingPriceB - f.purchasePriceB) * (totalUnits - a) = 3100 :=
by sorry

end NUMINAMATH_CALUDE_fan_sales_theorem_l2034_203407


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2034_203474

theorem largest_multiple_of_15_under_500 : ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  (∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2034_203474


namespace NUMINAMATH_CALUDE_mother_twice_age_2040_l2034_203401

/-- The year when Tina's mother's age is twice Tina's age -/
def year_mother_twice_age (tina_birth_year : ℕ) (tina_age_2010 : ℕ) (mother_age_multiplier_2010 : ℕ) : ℕ :=
  tina_birth_year + (mother_age_multiplier_2010 - 2) * tina_age_2010

theorem mother_twice_age_2040 :
  year_mother_twice_age 2000 10 5 = 2040 := by
  sorry

#eval year_mother_twice_age 2000 10 5

end NUMINAMATH_CALUDE_mother_twice_age_2040_l2034_203401


namespace NUMINAMATH_CALUDE_infinitely_many_k_with_Q_3k_geq_Q_3k1_l2034_203450

-- Define Q(n) as the sum of the decimal digits of n
def Q (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem infinitely_many_k_with_Q_3k_geq_Q_3k1 :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ Q (3^k) ≥ Q (3^(k+1)) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_k_with_Q_3k_geq_Q_3k1_l2034_203450


namespace NUMINAMATH_CALUDE_problem_solution_l2034_203499

-- Define the propositions
def p : Prop := ∀ x > 0, 3^x > 1
def q : Prop := ∀ a, a < -2 → (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
                    ¬(∀ a, (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) → a < -2)

-- Theorem statement
theorem problem_solution :
  (¬p ↔ ∃ x > 0, 3^x ≤ 1) ∧
  ¬p ∧
  q :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2034_203499


namespace NUMINAMATH_CALUDE_isosceles_triangle_two_two_one_l2034_203435

/-- Checks if three numbers can form a triangle based on the triangle inequality theorem -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if three numbers can form an isosceles triangle -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  is_triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

/-- Theorem: The set of side lengths (2, 2, 1) forms an isosceles triangle -/
theorem isosceles_triangle_two_two_one :
  is_isosceles_triangle 2 2 1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_two_two_one_l2034_203435


namespace NUMINAMATH_CALUDE_division_subtraction_problem_l2034_203464

theorem division_subtraction_problem (x : ℝ) : 
  (848 / x) - 100 = 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_problem_l2034_203464


namespace NUMINAMATH_CALUDE_first_half_speed_l2034_203434

/-- Given a trip with the following properties:
  - Total distance is 50 km
  - First half (25 km) is traveled at speed v km/h
  - Second half (25 km) is traveled at 30 km/h
  - Average speed of the entire trip is 40 km/h
  Then the speed v of the first half of the trip is 100/3 km/h. -/
theorem first_half_speed (v : ℝ) : 
  v > 0 → -- Ensure v is positive
  (25 / v + 25 / 30) * 40 = 50 → -- Average speed equation
  v = 100 / 3 := by
sorry


end NUMINAMATH_CALUDE_first_half_speed_l2034_203434


namespace NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l2034_203495

theorem numbers_less_than_reciprocals : ∃ (S : Set ℝ), 
  S = {-1/2, -3, 3, 1/2, 0} ∧ 
  (∀ x ∈ S, x ≠ 0 → (x < 1/x ↔ (x = -3 ∨ x = 1/2))) := by
  sorry

end NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l2034_203495


namespace NUMINAMATH_CALUDE_probability_select_one_from_each_probability_select_one_from_each_name_l2034_203455

/-- The probability of selecting one element from each of three equal-sized sets
    when drawing three elements without replacement from their union. -/
theorem probability_select_one_from_each (n : ℕ) : 
  n > 0 → (6 : ℚ) * (n : ℚ) * n * n / ((3 * n) * (3 * n - 1) * (3 * n - 2)) = 125 / 455 := by
  sorry

/-- The specific case for the problem where each set has 5 elements. -/
theorem probability_select_one_from_each_name : 
  (6 : ℚ) * 5 * 5 * 5 / (15 * 14 * 13) = 125 / 455 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_one_from_each_probability_select_one_from_each_name_l2034_203455


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2034_203426

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 30 and a_4 + a_5 = 120, 
    then a_7 + a_8 = 480. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence
  a 1 + a 2 = 30 →                          -- a_1 + a_2 = 30
  a 4 + a 5 = 120 →                         -- a_4 + a_5 = 120
  a 7 + a 8 = 480 :=                        -- a_7 + a_8 = 480
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2034_203426


namespace NUMINAMATH_CALUDE_red_yellow_black_shirts_l2034_203478

theorem red_yellow_black_shirts (total : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : total = 420) (h2 : blue = 85) (h3 : green = 157) :
  total - (blue + green) = 178 := by
  sorry

end NUMINAMATH_CALUDE_red_yellow_black_shirts_l2034_203478


namespace NUMINAMATH_CALUDE_prob_a_prob_b_prob_c_prob_d_prob_e_chess_probabilities_l2034_203472

/-- The total number of chess pieces -/
def total_pieces : ℕ := 32

/-- The number of pieces of each color -/
def pieces_per_color : ℕ := total_pieces / 2

/-- The number of pawns of each color -/
def pawns_per_color : ℕ := 8

/-- The number of bishops of each color -/
def bishops_per_color : ℕ := 2

/-- The number of rooks of each color -/
def rooks_per_color : ℕ := 2

/-- The number of knights of each color -/
def knights_per_color : ℕ := 2

/-- The number of kings of each color -/
def kings_per_color : ℕ := 1

/-- The number of queens of each color -/
def queens_per_color : ℕ := 1

/-- The probability of drawing 2 dark pieces or 2 pieces of different colors -/
theorem prob_a : ℚ :=
  47 / 62

/-- The probability of drawing 1 bishop and 1 pawn or 2 pieces of different colors -/
theorem prob_b : ℚ :=
  18 / 31

/-- The probability of drawing 2 different-colored rooks or 2 pieces of the same color but different sizes -/
theorem prob_c : ℚ :=
  91 / 248

/-- The probability of drawing 1 king and one knight of the same color, or two pieces of the same color -/
theorem prob_d : ℚ :=
  15 / 31

/-- The probability of drawing 2 pieces of the same size or 2 pieces of the same color -/
theorem prob_e : ℚ :=
  159 / 248

/-- The main theorem combining all probabilities -/
theorem chess_probabilities :
  (prob_a = 47 / 62) ∧
  (prob_b = 18 / 31) ∧
  (prob_c = 91 / 248) ∧
  (prob_d = 15 / 31) ∧
  (prob_e = 159 / 248) :=
by sorry

end NUMINAMATH_CALUDE_prob_a_prob_b_prob_c_prob_d_prob_e_chess_probabilities_l2034_203472


namespace NUMINAMATH_CALUDE_prob_purple_second_l2034_203488

-- Define the bags
def bag_A : Nat × Nat := (5, 5)  -- (red, green)
def bag_B : Nat × Nat := (8, 2)  -- (purple, orange)
def bag_C : Nat × Nat := (3, 7)  -- (purple, orange)

-- Define the probability of drawing a red marble from Bag A
def prob_red_A : Rat := bag_A.1 / (bag_A.1 + bag_A.2)

-- Define the probability of drawing a green marble from Bag A
def prob_green_A : Rat := bag_A.2 / (bag_A.1 + bag_A.2)

-- Define the probability of drawing a purple marble from Bag B
def prob_purple_B : Rat := bag_B.1 / (bag_B.1 + bag_B.2)

-- Define the probability of drawing a purple marble from Bag C
def prob_purple_C : Rat := bag_C.1 / (bag_C.1 + bag_C.2)

-- Theorem: The probability of drawing a purple marble as the second marble is 11/20
theorem prob_purple_second : 
  prob_red_A * prob_purple_B + prob_green_A * prob_purple_C = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_purple_second_l2034_203488


namespace NUMINAMATH_CALUDE_valid_new_usage_exists_l2034_203470

/-- Represents the time spent on an app --/
structure AppTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the usage data for four apps --/
structure AppUsage where
  app1 : AppTime
  app2 : AppTime
  app3 : AppTime
  app4 : AppTime

/-- Checks if the new usage data is consistent with halving two app times --/
def is_valid_new_usage (old_usage new_usage : AppUsage) : Prop :=
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app3.time = old_usage.app3.time / 2 ∧
   new_usage.app2.time = old_usage.app2.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app2.time = old_usage.app2.time / 2 ∧
   new_usage.app3.time = old_usage.app3.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app2.time = old_usage.app2.time ∧ new_usage.app3.time = old_usage.app3.time) ∨
  (new_usage.app2.time = old_usage.app2.time / 2 ∧ new_usage.app3.time = old_usage.app3.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app2.time = old_usage.app2.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app3.time = old_usage.app3.time) ∨
  (new_usage.app3.time = old_usage.app3.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app2.time = old_usage.app2.time)

theorem valid_new_usage_exists (old_usage : AppUsage) :
  ∃ new_usage : AppUsage, is_valid_new_usage old_usage new_usage :=
sorry

end NUMINAMATH_CALUDE_valid_new_usage_exists_l2034_203470


namespace NUMINAMATH_CALUDE_total_rubber_bands_l2034_203406

def harper_rubber_bands : ℕ := 100
def brother_difference : ℕ := 56
def sister_difference : ℕ := 47

theorem total_rubber_bands :
  harper_rubber_bands +
  (harper_rubber_bands - brother_difference) +
  (harper_rubber_bands - brother_difference + sister_difference) = 235 := by
  sorry

end NUMINAMATH_CALUDE_total_rubber_bands_l2034_203406


namespace NUMINAMATH_CALUDE_janice_homework_time_l2034_203424

/-- Represents the time (in minutes) it takes Janice to complete various tasks before watching a movie -/
structure JanicesTasks where
  total_time : ℝ
  homework_time : ℝ
  cleaning_time : ℝ
  dog_walking_time : ℝ
  trash_time : ℝ
  remaining_time : ℝ

/-- The theorem stating that Janice's homework time is 30 minutes given the conditions -/
theorem janice_homework_time (tasks : JanicesTasks) :
  tasks.total_time = 120 ∧
  tasks.cleaning_time = tasks.homework_time / 2 ∧
  tasks.dog_walking_time = tasks.homework_time + 5 ∧
  tasks.trash_time = tasks.homework_time / 6 ∧
  tasks.remaining_time = 35 ∧
  tasks.total_time = tasks.homework_time + tasks.cleaning_time + tasks.dog_walking_time + tasks.trash_time + tasks.remaining_time
  →
  tasks.homework_time = 30 :=
by sorry

end NUMINAMATH_CALUDE_janice_homework_time_l2034_203424


namespace NUMINAMATH_CALUDE_complex_multiplication_l2034_203482

theorem complex_multiplication (i : ℂ) :
  i * i = -1 →
  (-1 + i) * (2 - i) = -1 + 3 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2034_203482


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l2034_203498

/-- Given a sphere with circumference 18π inches cut into six congruent wedges,
    prove that the volume of one wedge is 162π cubic inches. -/
theorem volume_of_sphere_wedge :
  ∀ (r : ℝ), 
    r > 0 →
    2 * Real.pi * r = 18 * Real.pi →
    (4 / 3 * Real.pi * r^3) / 6 = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l2034_203498


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2034_203411

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 8 * x + 3 < 0) ↔ (1/3 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2034_203411


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2034_203489

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) :
  ∃ k : ℕ, a = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2034_203489


namespace NUMINAMATH_CALUDE_target_hit_probability_l2034_203414

def probability_hit : ℚ := 1 / 2

def total_shots : ℕ := 6

def successful_hits : ℕ := 3

def consecutive_hits : ℕ := 2

theorem target_hit_probability :
  (probability_hit ^ successful_hits) *
  ((1 - probability_hit) ^ (total_shots - successful_hits)) *
  (3 * (Nat.factorial 2 * Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2))) =
  3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2034_203414


namespace NUMINAMATH_CALUDE_exists_special_quadrilateral_l2034_203436

/-- Represents a quadrilateral with its properties -/
structure Quadrilateral where
  sides : Fin 4 → ℕ
  diagonals : Fin 2 → ℕ
  area : ℕ
  radius : ℕ

/-- Predicate to check if the quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if the side lengths are pairwise distinct -/
def hasPairwiseDistinctSides (q : Quadrilateral) : Prop :=
  ∀ i j, i ≠ j → q.sides i ≠ q.sides j

/-- Theorem stating the existence of a quadrilateral with the required properties -/
theorem exists_special_quadrilateral :
  ∃ q : Quadrilateral,
    isCyclic q ∧
    hasPairwiseDistinctSides q :=
  sorry

end NUMINAMATH_CALUDE_exists_special_quadrilateral_l2034_203436


namespace NUMINAMATH_CALUDE_complex_radical_expression_simplification_l2034_203441

theorem complex_radical_expression_simplification :
  3 * Real.sqrt (1/3) + Real.sqrt 2 * (Real.sqrt 3 - Real.sqrt 6) - Real.sqrt 12 / Real.sqrt 2 = - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_radical_expression_simplification_l2034_203441


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l2034_203418

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l2034_203418


namespace NUMINAMATH_CALUDE_min_value_when_a_2_a_values_for_max_3_l2034_203494

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- Part 1: Minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (min : ℝ), min = -1 ∧ ∀ x ∈ Set.Icc 0 3, f 2 x ≥ min :=
sorry

-- Part 2: Values of a for maximum 3 in [0, 1]
theorem a_values_for_max_3 :
  (∃ (max : ℝ), max = 3 ∧ ∀ x ∈ Set.Icc 0 1, f a x ≤ max) →
  (a = -2 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_2_a_values_for_max_3_l2034_203494


namespace NUMINAMATH_CALUDE_y1_value_l2034_203402

theorem y1_value (y1 y2 y3 : ℝ) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1) 
  (h2 : (1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1/9) : 
  y1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_y1_value_l2034_203402


namespace NUMINAMATH_CALUDE_wall_building_time_l2034_203490

/-- Given that 8 persons can build a 140m wall in 8 days, this theorem calculates
    the number of days it takes 30 persons to build a similar 100m wall. -/
theorem wall_building_time (persons1 persons2 : ℕ) (length1 length2 : ℝ) (days1 : ℝ) : 
  persons1 = 8 →
  persons2 = 30 →
  length1 = 140 →
  length2 = 100 →
  days1 = 8 →
  ∃ days2 : ℝ, days2 = (persons1 * days1 * length2) / (persons2 * length1) :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l2034_203490


namespace NUMINAMATH_CALUDE_t_perimeter_is_14_l2034_203410

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a T-shaped figure formed by two rectangles -/
def t_perimeter (top : Rectangle) (bottom : Rectangle) : ℝ :=
  let exposed_top := top.width
  let exposed_sides := (top.width - bottom.width) + 2 * bottom.height
  let exposed_bottom := bottom.width
  exposed_top + exposed_sides + exposed_bottom

/-- Theorem stating that the perimeter of the T-shaped figure is 14 inches -/
theorem t_perimeter_is_14 :
  let top := Rectangle.mk 6 1
  let bottom := Rectangle.mk 3 4
  t_perimeter top bottom = 14 := by
  sorry

end NUMINAMATH_CALUDE_t_perimeter_is_14_l2034_203410


namespace NUMINAMATH_CALUDE_min_abs_z_l2034_203462

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z - 8 * Complex.I) = 18) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 16) + Complex.abs (w - 8 * Complex.I) = 18 ∧ Complex.abs w = 64 / 9 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l2034_203462


namespace NUMINAMATH_CALUDE_basketball_tryouts_l2034_203448

/-- Given the number of girls and boys trying out for a basketball team,
    and the number of students called back, calculate the number of
    students who didn't make the cut. -/
theorem basketball_tryouts (girls boys called_back : ℕ) : 
  girls = 39 → boys = 4 → called_back = 26 → 
  girls + boys - called_back = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l2034_203448


namespace NUMINAMATH_CALUDE_max_xy_value_l2034_203429

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 4*y = 12) :
  xy ≤ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ = 12 ∧ x₀*y₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l2034_203429


namespace NUMINAMATH_CALUDE_max_S_at_7_or_8_l2034_203475

/-- Represents the sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ :=
  5 * n - (5 / 14) * n * (n - 1)

/-- The maximum value of S occurs when n is 7 or 8 -/
theorem max_S_at_7_or_8 :
  ∀ k : ℕ, (S k ≤ S 7 ∧ S k ≤ S 8) ∧
  (S 7 = S 8 ∨ (∀ m : ℕ, m ≠ 7 → m ≠ 8 → S m < max (S 7) (S 8))) := by
  sorry

end NUMINAMATH_CALUDE_max_S_at_7_or_8_l2034_203475


namespace NUMINAMATH_CALUDE_cody_game_count_l2034_203400

def final_game_count (initial_games : ℕ) (games_to_jake : ℕ) (games_to_sarah : ℕ) (new_games : ℕ) : ℕ :=
  initial_games - (games_to_jake + games_to_sarah) + new_games

theorem cody_game_count :
  final_game_count 9 4 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cody_game_count_l2034_203400


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2034_203443

def original_set_size : ℕ := 60
def original_mean : ℚ := 42
def discarded_numbers : List ℚ := [50, 60, 70]

theorem arithmetic_mean_after_removal :
  let original_sum : ℚ := original_mean * original_set_size
  let remaining_sum : ℚ := original_sum - (discarded_numbers.sum)
  let remaining_set_size : ℕ := original_set_size - discarded_numbers.length
  (remaining_sum / remaining_set_size : ℚ) = 41 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2034_203443


namespace NUMINAMATH_CALUDE_simplify_expression_l2034_203493

theorem simplify_expression : 
  (2 * 10^12) / (4 * 10^5 - 1 * 10^4) = 5.1282 * 10^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2034_203493


namespace NUMINAMATH_CALUDE_function_minimum_implies_a_range_l2034_203431

theorem function_minimum_implies_a_range :
  ∀ (a : ℝ),
  (∀ (x : ℝ), (a * (Real.cos x)^2 - 3) * Real.sin x ≥ -3) →
  (∃ (x : ℝ), (a * (Real.cos x)^2 - 3) * Real.sin x = -3) →
  a ∈ Set.Icc (-3/2) 12 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_implies_a_range_l2034_203431


namespace NUMINAMATH_CALUDE_skittles_distribution_l2034_203437

theorem skittles_distribution (total_skittles : ℕ) (num_friends : ℕ) (skittles_per_friend : ℕ) : 
  total_skittles = 40 → num_friends = 5 → skittles_per_friend = total_skittles / num_friends → skittles_per_friend = 8 := by
  sorry

end NUMINAMATH_CALUDE_skittles_distribution_l2034_203437


namespace NUMINAMATH_CALUDE_valid_combinations_for_elixir_l2034_203438

/-- Represents the number of different magical roots. -/
def num_roots : ℕ := 4

/-- Represents the number of different mystical minerals. -/
def num_minerals : ℕ := 6

/-- Represents the number of minerals incompatible with one root. -/
def minerals_incompatible_with_one_root : ℕ := 2

/-- Represents the number of roots incompatible with one mineral. -/
def roots_incompatible_with_one_mineral : ℕ := 2

/-- Represents the total number of incompatible combinations. -/
def total_incompatible_combinations : ℕ :=
  minerals_incompatible_with_one_root + roots_incompatible_with_one_mineral

/-- Theorem stating the number of valid combinations for the wizard's elixir. -/
theorem valid_combinations_for_elixir :
  num_roots * num_minerals - total_incompatible_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_valid_combinations_for_elixir_l2034_203438


namespace NUMINAMATH_CALUDE_number_division_problem_l2034_203458

theorem number_division_problem (x : ℝ) : (x / 5 = 70 + x / 6) → x = 2100 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2034_203458


namespace NUMINAMATH_CALUDE_hash_computation_l2034_203497

def hash (a b : ℤ) : ℤ := a * b - a - 3

theorem hash_computation : hash (hash 2 0) (hash 1 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hash_computation_l2034_203497
