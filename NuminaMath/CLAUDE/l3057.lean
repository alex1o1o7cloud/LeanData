import Mathlib

namespace NUMINAMATH_CALUDE_series_sum_l3057_305740

def series_term (n : ℕ) : ℚ := (2^n : ℚ) / ((3^(3^n) : ℚ) + 1)

theorem series_sum : ∑' n, series_term n = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_l3057_305740


namespace NUMINAMATH_CALUDE_determine_set_B_l3057_305724

def U : Set Nat := {2, 4, 6, 8, 10}

theorem determine_set_B (A B : Set Nat) 
  (h1 : (A ∪ B)ᶜ = {8, 10})
  (h2 : A ∩ (U \ B) = {2}) :
  B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_determine_set_B_l3057_305724


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3057_305741

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that ax^2 + bx + c = (px + q)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 4 → k = 4 ∨ k = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3057_305741


namespace NUMINAMATH_CALUDE_ratio_theorem_l3057_305706

theorem ratio_theorem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_l3057_305706


namespace NUMINAMATH_CALUDE_inverse_function_condition_l3057_305784

noncomputable def g (a b c d x : ℝ) : ℝ := (2*a*x + b) / (2*c*x - d)

theorem inverse_function_condition (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : ∀ x, x ∈ {x | 2*c*x - d ≠ 0} → g a b c d (g a b c d x) = x) : 
  2*a - d = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_condition_l3057_305784


namespace NUMINAMATH_CALUDE_unattainable_y_value_l3057_305707

theorem unattainable_y_value (x : ℝ) (h : x ≠ -5/4) :
  ∀ y : ℝ, y = (2 - 3*x) / (4*x + 5) → y ≠ -3/4 := by
sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l3057_305707


namespace NUMINAMATH_CALUDE_loop_statement_efficiency_l3057_305733

/-- Enum representing different types of algorithm statements -/
inductive AlgorithmStatement
  | InputOutput
  | Assignment
  | Conditional
  | Loop

/-- Definition of a program's capability to handle large computational problems -/
def CanHandleLargeProblems (statements : List AlgorithmStatement) : Prop :=
  statements.length > 0

/-- Definition of the primary reason for efficient handling of large problems -/
def PrimaryReasonForEfficiency (statement : AlgorithmStatement) (statements : List AlgorithmStatement) : Prop :=
  CanHandleLargeProblems statements ∧ statement ∈ statements

theorem loop_statement_efficiency :
  ∀ (statements : List AlgorithmStatement),
    CanHandleLargeProblems statements →
    AlgorithmStatement.InputOutput ∈ statements →
    AlgorithmStatement.Assignment ∈ statements →
    AlgorithmStatement.Conditional ∈ statements →
    AlgorithmStatement.Loop ∈ statements →
    PrimaryReasonForEfficiency AlgorithmStatement.Loop statements :=
by
  sorry

#check loop_statement_efficiency

end NUMINAMATH_CALUDE_loop_statement_efficiency_l3057_305733


namespace NUMINAMATH_CALUDE_same_suit_bottom_probability_l3057_305760

def deck_size : Nat := 6
def black_cards : Nat := 3
def red_cards : Nat := 3

theorem same_suit_bottom_probability :
  let total_arrangements := Nat.factorial deck_size
  let favorable_outcomes := 2 * (Nat.factorial black_cards * Nat.factorial red_cards)
  (favorable_outcomes : ℚ) / total_arrangements = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_same_suit_bottom_probability_l3057_305760


namespace NUMINAMATH_CALUDE_line_mb_value_l3057_305730

/-- A line passing through (-1, -3) and intersecting the y-axis at y = -1 has mb = 2 -/
theorem line_mb_value (m b : ℝ) : 
  (∀ x y, y = m * x + b → (x = -1 ∧ y = -3) ∨ (x = 0 ∧ y = -1)) → 
  m * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_value_l3057_305730


namespace NUMINAMATH_CALUDE_product_four_consecutive_integers_l3057_305780

theorem product_four_consecutive_integers (a : ℤ) : 
  a^2 = 1000 * 1001 * 1002 * 1003 + 1 → a = 1002001 := by
  sorry

end NUMINAMATH_CALUDE_product_four_consecutive_integers_l3057_305780


namespace NUMINAMATH_CALUDE_spinsters_and_cats_l3057_305781

theorem spinsters_and_cats (spinsters : ℕ) (cats : ℕ) : 
  spinsters = 18 → 
  spinsters * 9 = cats * 2 → 
  cats - spinsters = 63 := by
sorry

end NUMINAMATH_CALUDE_spinsters_and_cats_l3057_305781


namespace NUMINAMATH_CALUDE_four_tellers_coins_l3057_305773

/-- Calculates the total number of coins for a given number of bank tellers -/
def totalCoins (numTellers : ℕ) (rollsPerTeller : ℕ) (coinsPerRoll : ℕ) : ℕ :=
  numTellers * rollsPerTeller * coinsPerRoll

/-- Theorem: Four bank tellers have 1000 coins in total -/
theorem four_tellers_coins :
  totalCoins 4 10 25 = 1000 := by
  sorry

#eval totalCoins 4 10 25  -- Should output 1000

end NUMINAMATH_CALUDE_four_tellers_coins_l3057_305773


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3057_305754

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 10 / Real.sqrt 11) = 4 * Real.sqrt 66 / 33 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3057_305754


namespace NUMINAMATH_CALUDE_telephone_pole_height_l3057_305713

-- Define the problem parameters
def base_height : Real := 1
def cable_ground_distance : Real := 5
def leah_distance : Real := 4
def leah_height : Real := 1.8

-- Define the theorem
theorem telephone_pole_height :
  let total_ground_distance : Real := cable_ground_distance + base_height
  let remaining_distance : Real := total_ground_distance - leah_distance
  let pole_height : Real := (leah_height * total_ground_distance) / remaining_distance
  pole_height = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_telephone_pole_height_l3057_305713


namespace NUMINAMATH_CALUDE_original_ratio_proof_l3057_305777

theorem original_ratio_proof (x y : ℕ+) (h1 : y = 24) (h2 : (x + 6 : ℚ) / y = 1 / 2) : 
  (x : ℚ) / y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_proof_l3057_305777


namespace NUMINAMATH_CALUDE_inequality_proof_l3057_305722

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hx_le_1 : x ≤ 1) :
  x * y + y + 2 * z ≥ 4 * Real.sqrt (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3057_305722


namespace NUMINAMATH_CALUDE_trajectory_equation_l3057_305744

/-- The trajectory of a point P(x,y) satisfying a specific condition with respect to fixed points M and N -/
theorem trajectory_equation (x y : ℝ) : 
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  let P : ℝ × ℝ := (x, y)
  let MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
  let MP : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)
  let NP : ℝ × ℝ := (P.1 - N.1, P.2 - N.2)
  ‖MN‖ * ‖MP‖ + MN.1 * NP.1 + MN.2 * NP.2 = 0 →
  y^2 = -8*x := by
sorry


end NUMINAMATH_CALUDE_trajectory_equation_l3057_305744


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3057_305782

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3057_305782


namespace NUMINAMATH_CALUDE_bookshop_unsold_percentage_l3057_305771

def initial_stock : ℕ := 1200
def sales : List ℕ := [75, 50, 64, 78, 135]

def books_sold (sales : List ℕ) : ℕ := sales.sum

def books_not_sold (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

def percentage_not_sold (initial : ℕ) (not_sold : ℕ) : ℚ :=
  (not_sold : ℚ) / (initial : ℚ) * 100

theorem bookshop_unsold_percentage :
  let sold := books_sold sales
  let not_sold := books_not_sold initial_stock sold
  percentage_not_sold initial_stock not_sold = 66.5 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_unsold_percentage_l3057_305771


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3057_305765

theorem simplify_trigonometric_expression (θ : Real) 
  (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) : 
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3057_305765


namespace NUMINAMATH_CALUDE_perp_line_plane_from_conditions_l3057_305753

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Axiom: If a line is perpendicular to two planes, those planes are parallel
axiom perp_two_planes_parallel (n : Line) (α β : Plane) :
  perp_line_plane n α → perp_line_plane n β → α = β

-- Axiom: If a line is perpendicular to one of two parallel planes, it's perpendicular to the other
axiom perp_parallel_planes (m : Line) (α β : Plane) :
  α = β → perp_line_plane m α → perp_line_plane m β

-- Theorem to prove
theorem perp_line_plane_from_conditions (n m : Line) (α β : Plane) :
  perp_line_plane n α →
  perp_line_plane n β →
  perp_line_plane m α →
  perp_line_plane m β :=
by sorry

end NUMINAMATH_CALUDE_perp_line_plane_from_conditions_l3057_305753


namespace NUMINAMATH_CALUDE_base8_digit_product_7890_l3057_305775

/-- Given a natural number n, returns the list of its digits in base 8 --/
def toBase8Digits (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product_7890 :
  listProduct (toBase8Digits 7890) = 336 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_7890_l3057_305775


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3057_305708

/-- Given a point P and a line L, this theorem proves that the line
    perpendicular to L passing through P has the correct equation. -/
theorem perpendicular_line_through_point
  (P : ℝ × ℝ)  -- Point P
  (L : ℝ → ℝ → Prop)  -- Line L
  (h_L : L = fun x y ↦ x - 2 * y + 3 = 0)  -- Equation of line L
  (h_P : P = (-1, 3))  -- Coordinates of point P
  : (fun x y ↦ 2 * x + y - 1 = 0) P.1 P.2 ∧  -- The line passes through P
    (∀ x₁ y₁ x₂ y₂, L x₁ y₁ → L x₂ y₂ →
      (x₂ - x₁) * 2 + (y₂ - y₁) * 1 = 0)  -- The lines are perpendicular
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3057_305708


namespace NUMINAMATH_CALUDE_marys_nickels_l3057_305723

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Mary's total nickels is the sum of her initial nickels and received nickels -/
theorem marys_nickels (initial : ℕ) (received : ℕ) :
  total_nickels initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_marys_nickels_l3057_305723


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l3057_305776

def vector_angle (u v : ℝ × ℝ) : ℝ := sorry

theorem vectors_perpendicular : 
  let u : ℝ × ℝ := (3, -4)
  let v : ℝ × ℝ := (4, 3)
  vector_angle u v = 90 := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l3057_305776


namespace NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l3057_305700

theorem min_value_sum_of_distances (x : ℝ) :
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2) ≥ Real.sqrt 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2) = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l3057_305700


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3057_305791

theorem floor_plus_self_unique_solution (r : ℝ) : 
  (⌊r⌋ : ℝ) + r = 16.5 ↔ r = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3057_305791


namespace NUMINAMATH_CALUDE_jims_paycheck_l3057_305736

def gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

def retirement_deduction : ℝ := gross_pay * retirement_rate

def net_pay : ℝ := gross_pay - retirement_deduction - tax_deduction

theorem jims_paycheck : net_pay = 740 := by
  sorry

end NUMINAMATH_CALUDE_jims_paycheck_l3057_305736


namespace NUMINAMATH_CALUDE_vasya_always_wins_l3057_305762

/-- Represents the state of the game with the number of piles -/
structure GameState :=
  (piles : ℕ)

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Defines a single move in the game -/
def move (state : GameState) : GameState :=
  { piles := state.piles + 2 }

/-- Determines if a given state is a winning state for the current player -/
def is_winning_state (state : GameState) : Prop :=
  ∃ (n : ℕ), state.piles = 2 * n + 1

/-- The main theorem stating that Vasya (second player) always wins -/
theorem vasya_always_wins :
  ∀ (initial_state : GameState),
  initial_state.piles = 3 →
  is_winning_state (move initial_state) :=
sorry

end NUMINAMATH_CALUDE_vasya_always_wins_l3057_305762


namespace NUMINAMATH_CALUDE_average_problem_l3057_305774

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of four numbers
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem average_problem :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3057_305774


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l3057_305742

theorem sum_of_squares_bound 
  (a b c d x y z t : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (ht : t ≥ 1) 
  (hsum : a + b + c + d + x + y + z + t = 8) : 
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l3057_305742


namespace NUMINAMATH_CALUDE_simplify_expression_l3057_305793

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 64) - Real.sqrt (9 + 1/4))^2 = 69/4 - 2 * Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3057_305793


namespace NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l3057_305787

/-- Definition of the ellipse E -/
def is_ellipse (E : Set (ℝ × ℝ)) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ x y, (x, y) ∈ E ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- E passes through the points (2, √2) and (√6, 1) -/
def passes_through_points (E : Set (ℝ × ℝ)) : Prop :=
  (2, Real.sqrt 2) ∈ E ∧ (Real.sqrt 6, 1) ∈ E

/-- Definition of perpendicular vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem ellipse_and_circle_theorem (E : Set (ℝ × ℝ)) (a b : ℝ) 
  (h_ellipse : is_ellipse E a b) (h_points : passes_through_points E) :
  (∃ r : ℝ, r > 0 ∧
    (∀ x y, (x, y) ∈ E ↔ x^2 / 8 + y^2 / 4 = 1) ∧
    (∀ k m : ℝ,
      (∃ A B : ℝ × ℝ,
        A ∈ E ∧ B ∈ E ∧
        A.2 = k * A.1 + m ∧
        B.2 = k * B.1 + m ∧
        perpendicular A B ∧
        A.1^2 + A.2^2 = r^2 ∧
        B.1^2 + B.2^2 = r^2) ↔
      k^2 + 1 = (8 / 3) / m^2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l3057_305787


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l3057_305752

theorem absolute_value_not_positive (x : ℚ) : |4*x + 6| ≤ 0 ↔ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l3057_305752


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3057_305743

theorem solution_set_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3057_305743


namespace NUMINAMATH_CALUDE_calculator_cost_ratio_l3057_305792

theorem calculator_cost_ratio :
  ∀ (basic scientific graphing : ℝ),
  basic = 8 →
  graphing = 3 * scientific →
  100 - (basic + scientific + graphing) = 28 →
  scientific / basic = 2 := by
sorry

end NUMINAMATH_CALUDE_calculator_cost_ratio_l3057_305792


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l3057_305786

/-- The minimum number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem min_triangles_to_cover (small_side large_side : ℝ) : 
  small_side = 2 →
  large_side = 16 →
  (large_side / small_side) ^ 2 = 64 := by
  sorry

#check min_triangles_to_cover

end NUMINAMATH_CALUDE_min_triangles_to_cover_l3057_305786


namespace NUMINAMATH_CALUDE_find_B_value_l3057_305714

theorem find_B_value (A B : Nat) (h1 : A ≤ 9) (h2 : B ≤ 9) 
  (h3 : 32 + A * 100 + 70 + B = 705) : B = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_B_value_l3057_305714


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_5_11_l3057_305779

theorem smallest_five_digit_divisible_by_2_5_11 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 11 ∣ n → 10010 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_5_11_l3057_305779


namespace NUMINAMATH_CALUDE_specific_polygon_properties_l3057_305796

/-- Represents a regular polygon with given properties -/
structure RegularPolygon where
  total_angle_sum : ℝ
  known_angle : ℝ
  num_sides : ℕ
  remaining_angle : ℝ

/-- Theorem about a specific regular polygon -/
theorem specific_polygon_properties :
  let p := RegularPolygon.mk 3420 160 21 163
  p.num_sides = 21 ∧
  p.remaining_angle = 163 ∧
  p.total_angle_sum = 180 * (p.num_sides - 2) ∧
  p.total_angle_sum = p.known_angle + (p.num_sides - 1) * p.remaining_angle :=
by sorry

end NUMINAMATH_CALUDE_specific_polygon_properties_l3057_305796


namespace NUMINAMATH_CALUDE_green_pill_cost_l3057_305795

theorem green_pill_cost (weeks : ℕ) (daily_green : ℕ) (daily_pink : ℕ) 
  (green_pink_diff : ℚ) (total_cost : ℚ) :
  weeks = 3 →
  daily_green = 1 →
  daily_pink = 1 →
  green_pink_diff = 3 →
  total_cost = 819 →
  ∃ (green_cost : ℚ), 
    green_cost = 21 ∧ 
    (weeks * 7 * (green_cost + (green_cost - green_pink_diff))) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_l3057_305795


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l3057_305758

/-- The minimum distance between a point on the line y = 8/15 * x - 10 and a point on the parabola y = x^2 is 2234/255 -/
theorem min_distance_line_parabola :
  let line := {p : ℝ × ℝ | p.2 = 8/15 * p.1 - 10}
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  ∃ d : ℝ, d = 2234/255 ∧
    ∀ p₁ ∈ line, ∀ p₂ ∈ parabola,
      Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l3057_305758


namespace NUMINAMATH_CALUDE_mary_work_hours_l3057_305734

/-- Mary's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, and Friday (each)
  hours_tt : ℕ   -- Hours worked on Tuesday and Thursday (combined)
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Weekly earnings in dollars

/-- Theorem stating Mary's work hours on Tuesday and Thursday --/
theorem mary_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_mwf = 9)
  (h2 : schedule.hourly_rate = 11)
  (h3 : schedule.weekly_earnings = 407)
  (h4 : schedule.weekly_earnings = 
        schedule.hourly_rate * (3 * schedule.hours_mwf + schedule.hours_tt)) :
  schedule.hours_tt = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_work_hours_l3057_305734


namespace NUMINAMATH_CALUDE_simplify_fraction_l3057_305737

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3057_305737


namespace NUMINAMATH_CALUDE_contains_quadrilateral_l3057_305726

/-- A plane graph with n vertices and m edges, where no three points are collinear -/
structure PlaneGraph where
  n : ℕ
  m : ℕ
  no_collinear_triple : True  -- Placeholder for the condition that no three points are collinear

/-- Theorem: If m > (1/4)n(1 + √(4n - 3)) in a plane graph, then it contains a quadrilateral -/
theorem contains_quadrilateral (G : PlaneGraph) :
  G.m > (1/4 : ℝ) * G.n * (1 + Real.sqrt (4 * G.n - 3)) →
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    (∃ (e1 e2 e3 e4 : Set ℕ), 
      e1 = {a, b} ∧ e2 = {b, c} ∧ e3 = {c, d} ∧ e4 = {d, a}) :=
by sorry

end NUMINAMATH_CALUDE_contains_quadrilateral_l3057_305726


namespace NUMINAMATH_CALUDE_identical_second_differences_imply_arithmetic_progression_l3057_305763

/-- Second difference of a function -/
def secondDifference (g : ℕ → ℝ) (n : ℕ) : ℝ :=
  g (n + 2) - 2 * g (n + 1) + g n

/-- A sequence is an arithmetic progression if its second difference is zero -/
def isArithmeticProgression (g : ℕ → ℝ) : Prop :=
  ∀ n, secondDifference g n = 0

theorem identical_second_differences_imply_arithmetic_progression
  (f φ : ℕ → ℝ)
  (h : ∀ n, secondDifference f n = secondDifference φ n) :
  isArithmeticProgression (fun n ↦ f n - φ n) :=
by sorry

end NUMINAMATH_CALUDE_identical_second_differences_imply_arithmetic_progression_l3057_305763


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l3057_305798

theorem smallest_positive_integer_satisfying_congruences : ∃! b : ℕ+, 
  (b : ℤ) % 3 = 2 ∧ 
  (b : ℤ) % 4 = 3 ∧ 
  (b : ℤ) % 5 = 4 ∧ 
  (b : ℤ) % 7 = 6 ∧ 
  ∀ c : ℕ+, 
    ((c : ℤ) % 3 = 2 ∧ 
     (c : ℤ) % 4 = 3 ∧ 
     (c : ℤ) % 5 = 4 ∧ 
     (c : ℤ) % 7 = 6) → 
    b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l3057_305798


namespace NUMINAMATH_CALUDE_inequality_iff_in_solution_set_l3057_305718

/-- The solution set for the inequality 1/(x(x+2)) - 1/((x+2)(x+3)) < 1/4 -/
def solution_set : Set ℝ :=
  { x | x < -3 ∨ (-2 < x ∧ x < 0) ∨ 1 < x }

/-- The inequality function -/
def inequality (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 2) * (x + 3)) < 1 / 4

theorem inequality_iff_in_solution_set :
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_in_solution_set_l3057_305718


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3057_305725

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ) * Complex.I = -1 →
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = b * Complex.I) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3057_305725


namespace NUMINAMATH_CALUDE_pet_ownership_l3057_305719

theorem pet_ownership (S : Finset Nat) (D C B : Finset Nat) : 
  S.card = 60 ∧
  (∀ s ∈ S, s ∈ D ∪ C ∪ B) ∧
  D.card = 35 ∧
  C.card = 45 ∧
  B.card = 10 ∧
  (∀ b ∈ B, b ∈ D ∪ C) →
  ((D ∩ C) \ B).card = 10 := by
sorry

end NUMINAMATH_CALUDE_pet_ownership_l3057_305719


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3057_305789

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := x^2 - 16*x + 15

/-- The transformed quadratic function -/
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

theorem quadratic_transformation :
  ∃ b c : ℝ, (∀ x : ℝ, f x = g x b c) ∧ b + c = -57 := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3057_305789


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3057_305799

theorem least_positive_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 2 = 0) ∧ 
  (a % 5 = 1) ∧ 
  (a % 4 = 2) ∧ 
  (∀ (b : ℕ), b > 0 ∧ b % 2 = 0 ∧ b % 5 = 1 ∧ b % 4 = 2 → a ≤ b) ∧
  (a = 6) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3057_305799


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3057_305750

theorem complementary_angles_difference (x : ℝ) (h1 : 4 * x + x = 90) (h2 : x > 0) : |4 * x - x| = 54 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3057_305750


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_one_l3057_305757

theorem fraction_meaningful_iff_not_neg_one (a : ℝ) :
  (∃ (x : ℝ), x = 2 / (a + 1)) ↔ a ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_one_l3057_305757


namespace NUMINAMATH_CALUDE_investment_problem_l3057_305728

/-- Investment problem -/
theorem investment_problem (a b total_profit a_profit c : ℚ) 
  (ha : a = 6300)
  (hb : b = 4200)
  (htotal : total_profit = 13600)
  (ha_profit : a_profit = 4080)
  (h_ratio : a / (a + b + c) = a_profit / total_profit) :
  c = 10500 := by
  sorry


end NUMINAMATH_CALUDE_investment_problem_l3057_305728


namespace NUMINAMATH_CALUDE_teacher_student_meeting_l3057_305720

/-- Represents the teacher-student meeting scenario -/
structure MeetingScenario where
  total_participants : ℕ
  first_teacher_students : ℕ
  teachers : ℕ
  students : ℕ

/-- Checks if the given scenario satisfies the meeting conditions -/
def is_valid_scenario (m : MeetingScenario) : Prop :=
  m.total_participants = m.teachers + m.students ∧
  m.first_teacher_students = m.students - m.teachers + 1 ∧
  m.teachers > 0 ∧
  m.students > 0

/-- The theorem stating the correct number of teachers and students -/
theorem teacher_student_meeting :
  ∃ (m : MeetingScenario), is_valid_scenario m ∧ m.teachers = 8 ∧ m.students = 23 :=
sorry

end NUMINAMATH_CALUDE_teacher_student_meeting_l3057_305720


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3057_305712

theorem quadratic_unique_solution (b d : ℤ) : 
  (∃! x : ℝ, b * x^2 + 24 * x + d = 0) →
  b + d = 41 →
  b < d →
  b = 9 ∧ d = 32 := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3057_305712


namespace NUMINAMATH_CALUDE_coplanar_condition_l3057_305769

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (O A B C M : V)

theorem coplanar_condition (h : A - M + (B - M) + (C - M) = 0) : 
  ∃ (a b c : ℝ), a + b + c = 1 ∧ M - O = a • (A - O) + b • (B - O) + c • (C - O) := by
  sorry

end NUMINAMATH_CALUDE_coplanar_condition_l3057_305769


namespace NUMINAMATH_CALUDE_simplify_fraction_l3057_305764

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2*x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3057_305764


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l3057_305756

theorem existence_of_counterexample : ∃ m n : ℝ, m > n ∧ m^2 ≤ n^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l3057_305756


namespace NUMINAMATH_CALUDE_lucy_money_problem_l3057_305767

theorem lucy_money_problem (initial_money : ℚ) : 
  (initial_money * (2/3) * (3/4) = 15) → initial_money = 30 := by
  sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l3057_305767


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3057_305711

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < 1 ∨ x > 3}) :
  c < 0 ∧
  a + 2*b + 4*c < 0 ∧
  {x : ℝ | c*x + a < 0} = {x : ℝ | x > -1/3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3057_305711


namespace NUMINAMATH_CALUDE_equation_solutions_l3057_305768

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(77, 14, 1), (14, 77, 1), (70, 35, 1), (35, 70, 1), (8, 4, 0), (4, 8, 0)}

def satisfies_equation (x y z : ℕ) : Prop :=
  x^2 + y^2 = 3 * 2016^z + 77

theorem equation_solutions :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3057_305768


namespace NUMINAMATH_CALUDE_function_properties_l3057_305790

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_odd (fun x ↦ f (x + 1/2)))
  (h2 : ∀ x, f (2 - 3*x) = f (3*x)) :
  f (-1/2) = 0 ∧ 
  is_even (fun x ↦ f (x + 2)) ∧ 
  is_odd (fun x ↦ f (x - 1/2)) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3057_305790


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3057_305751

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  let billion : ℝ := 1000000000
  let gdp : ℝ := 53100 * billion
  toScientificNotation gdp = ScientificNotation.mk 5.31 12 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3057_305751


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l3057_305761

/-- The time when a ball hits the ground given its height equation -/
theorem ball_hitting_ground_time : ∃ t : ℚ, t > 0 ∧ -4.9 * t^2 + 4 * t + 6 = 0 ∧ t = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l3057_305761


namespace NUMINAMATH_CALUDE_smallest_d_value_l3057_305727

theorem smallest_d_value (c d : ℕ+) (h1 : c.val - d.val = 8) 
  (h2 : Nat.gcd ((c.val^3 + d.val^3) / (c.val + d.val)) (c.val * d.val) = 16) : 
  d.val ≥ 4 ∧ ∃ (c' d' : ℕ+), d'.val = 4 ∧ c'.val - d'.val = 8 ∧ 
    Nat.gcd ((c'.val^3 + d'.val^3) / (c'.val + d'.val)) (c'.val * d'.val) = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_value_l3057_305727


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3057_305731

theorem book_arrangement_theorem :
  let total_books : ℕ := 10
  let spanish_books : ℕ := 4
  let french_books : ℕ := 3
  let german_books : ℕ := 3
  let number_of_units : ℕ := 2 + german_books

  spanish_books + french_books + german_books = total_books →
  (number_of_units.factorial * spanish_books.factorial * french_books.factorial : ℕ) = 17280 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3057_305731


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3057_305735

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_perpendicular :
  perpendicular (3 * i.1 - j.1, 3 * i.2 - j.2) (i.1 + 3 * j.1, i.2 + 3 * j.2) := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3057_305735


namespace NUMINAMATH_CALUDE_smallest_n_for_jason_win_l3057_305770

/-- Represents the game board -/
structure GameBoard :=
  (width : Nat)
  (length : Nat)

/-- Represents a block that can be placed on the game board -/
structure Block :=
  (width : Nat)
  (length : Nat)

/-- Represents a player in the game -/
inductive Player
  | Jason
  | Jared

/-- Defines the game rules and conditions -/
def GameRules (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) :=
  board.width = 3 ∧
  board.length = 300 ∧
  jasonBlock.width = 2 ∧
  jasonBlock.length = 100 ∧
  jaredBlock.width = 2 ∧
  jaredBlock.length > 3

/-- Determines if a player can win given the game rules and block sizes -/
def CanWin (player : Player) (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) : Prop :=
  sorry

/-- The main theorem stating that 51 is the smallest n for Jason to guarantee a win -/
theorem smallest_n_for_jason_win (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) :
  GameRules board jasonBlock jaredBlock →
  (∀ n : Nat, n > 3 → n < 51 → ¬CanWin Player.Jason board jasonBlock {width := 2, length := n}) ∧
  CanWin Player.Jason board jasonBlock {width := 2, length := 51} :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_jason_win_l3057_305770


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3057_305759

-- Define the circle C₁
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 19 = 0

-- Define the line l
def line_l (x y a : ℝ) : Prop :=
  x + 2*y - a = 0

-- Theorem statement
theorem circle_symmetry_line (a : ℝ) :
  (∃ (x y : ℝ), circle_C1 x y ∧ line_l x y a) →
  (∀ (x y : ℝ), circle_C1 x y → 
    ∃ (x' y' : ℝ), circle_C1 x' y' ∧ 
      ((x + x')/2, (y + y')/2) ∈ {(x, y) | line_l x y a}) →
  a = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3057_305759


namespace NUMINAMATH_CALUDE_sophie_rearrangement_time_l3057_305702

def name_length : ℕ := 6
def rearrangements_per_minute : ℕ := 18

theorem sophie_rearrangement_time :
  (name_length.factorial / rearrangements_per_minute : ℚ) / 60 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sophie_rearrangement_time_l3057_305702


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l3057_305746

/-- Represents the number of trees of each type -/
structure TreeCounts where
  apricot : ℕ
  peach : ℕ
  cherry : ℕ

/-- Calculates the ratio of trees given the counts -/
def tree_ratio (counts : TreeCounts) : ℕ × ℕ × ℕ :=
  let gcd := Nat.gcd (Nat.gcd counts.apricot counts.peach) counts.cherry
  (counts.apricot / gcd, counts.peach / gcd, counts.cherry / gcd)

theorem tree_planting_ratio :
  ∀ (yard_size : ℕ) (space_per_tree : ℕ),
    yard_size = 2000 →
    space_per_tree = 10 →
    ∃ (counts : TreeCounts),
      counts.apricot = 58 ∧
      counts.peach = 3 * counts.apricot ∧
      counts.cherry = 5 * counts.peach ∧
      tree_ratio counts = (1, 3, 15) :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_ratio_l3057_305746


namespace NUMINAMATH_CALUDE_standard_deviation_of_scores_l3057_305704

def scores : List ℝ := [10, 10, 10, 9, 10, 8, 8, 10, 10, 8]

theorem standard_deviation_of_scores :
  let n : ℕ := scores.length
  let mean : ℝ := (scores.sum) / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  Real.sqrt variance = 0.9 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_scores_l3057_305704


namespace NUMINAMATH_CALUDE_optimal_triangle_count_l3057_305732

/-- A configuration of points in space --/
structure PointConfiguration where
  total_points : Nat
  num_groups : Nat
  group_sizes : Fin num_groups → Nat
  non_collinear : Bool
  different_sizes : ∀ i j, i ≠ j → group_sizes i ≠ group_sizes j

/-- The number of triangles formed by selecting one point from each of three different groups --/
def num_triangles (config : PointConfiguration) : Nat :=
  sorry

/-- The optimal configuration for maximizing the number of triangles --/
def optimal_config : PointConfiguration where
  total_points := 1989
  num_groups := 30
  group_sizes := fun i => 
    if i.val < 6 then 51 + i.val
    else if i.val = 6 then 58
    else 59 + i.val - 7
  non_collinear := true
  different_sizes := sorry

theorem optimal_triangle_count (config : PointConfiguration) :
  config.total_points = 1989 →
  config.num_groups = 30 →
  config.non_collinear = true →
  num_triangles config ≤ num_triangles optimal_config :=
sorry

end NUMINAMATH_CALUDE_optimal_triangle_count_l3057_305732


namespace NUMINAMATH_CALUDE_tyrone_eric_marbles_l3057_305797

/-- Proves that Tyrone gave 10 marbles to Eric -/
theorem tyrone_eric_marbles : ∀ x : ℕ,
  (100 : ℕ) - x = 3 * ((20 : ℕ) + x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tyrone_eric_marbles_l3057_305797


namespace NUMINAMATH_CALUDE_discriminant_of_polynomial_l3057_305772

/-- The discriminant of a quadratic polynomial ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 5x^2 + (5 + 1/5)x + 1/5 -/
def polynomial (x : ℚ) : ℚ := 5*x^2 + (5 + 1/5)*x + 1/5

theorem discriminant_of_polynomial :
  discriminant 5 (5 + 1/5) (1/5) = 576/25 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_polynomial_l3057_305772


namespace NUMINAMATH_CALUDE_min_distance_point_is_diagonal_intersection_l3057_305709

/-- Given a quadrilateral ABCD in a plane, the point that minimizes the sum of
    distances to all vertices is the intersection of its diagonals. -/
theorem min_distance_point_is_diagonal_intersection
  (A B C D : EuclideanSpace ℝ (Fin 2)) :
  ∃ O : EuclideanSpace ℝ (Fin 2),
    (∀ P : EuclideanSpace ℝ (Fin 2),
      dist O A + dist O B + dist O C + dist O D ≤
      dist P A + dist P B + dist P C + dist P D) ∧
    (∃ t s : ℝ, O = (1 - t) • A + t • C ∧ O = (1 - s) • B + s • D) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_point_is_diagonal_intersection_l3057_305709


namespace NUMINAMATH_CALUDE_at_least_one_genuine_certain_l3057_305721

def total_products : ℕ := 12
def genuine_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem at_least_one_genuine_certain :
  (1 : ℚ) = 1 - (defective_products.choose selected_products : ℚ) / (total_products.choose selected_products : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_genuine_certain_l3057_305721


namespace NUMINAMATH_CALUDE_union_and_intersection_of_rational_and_irrational_l3057_305748

-- Define A as the set of rational numbers
def A : Set ℝ := {x : ℝ | ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q}

-- Define B as the set of irrational numbers
def B : Set ℝ := {x : ℝ | x ∉ A}

theorem union_and_intersection_of_rational_and_irrational :
  (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) := by
  sorry

end NUMINAMATH_CALUDE_union_and_intersection_of_rational_and_irrational_l3057_305748


namespace NUMINAMATH_CALUDE_dot_product_range_l3057_305745

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (angle_BAC : Real)
  (length_AB : Real)
  (length_AC : Real)

-- Define a point D on side BC
def PointOnBC (triangle : Triangle) := 
  {D : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • triangle.B + t • triangle.C}

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_range (triangle : Triangle) 
  (h1 : triangle.angle_BAC = 2*π/3)  -- 120° in radians
  (h2 : triangle.length_AB = 2)
  (h3 : triangle.length_AC = 1) :
  ∀ D ∈ PointOnBC triangle, 
    -5 ≤ dot_product (D - triangle.A) (triangle.C - triangle.B) ∧ 
    dot_product (D - triangle.A) (triangle.C - triangle.B) ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_dot_product_range_l3057_305745


namespace NUMINAMATH_CALUDE_product_quotient_puzzle_l3057_305705

theorem product_quotient_puzzle :
  ∃ (x y t : ℕ+),
    100 ≤ (x * y : ℕ) ∧ (x * y : ℕ) ≤ 999 ∧
    x * y = t^3 ∧
    (x : ℚ) / y = t^2 ∧
    x = 243 ∧ y = 3 :=
by sorry

end NUMINAMATH_CALUDE_product_quotient_puzzle_l3057_305705


namespace NUMINAMATH_CALUDE_first_child_share_l3057_305785

/-- Given a total amount and relationships between three children's shares, 
    calculate the first child's share. -/
theorem first_child_share (total : ℚ) (n : ℕ) 
  (h_total : total = 378)
  (h_n : n = 3)
  (h_relation : ∃ (a b c : ℚ), a + b + c = total ∧ 12 * a = 8 * b ∧ 8 * b = 6 * c) :
  ∃ (a : ℚ), a = 84 ∧ ∃ (b c : ℚ), a + b + c = total ∧ 12 * a = 8 * b ∧ 8 * b = 6 * c :=
by sorry

end NUMINAMATH_CALUDE_first_child_share_l3057_305785


namespace NUMINAMATH_CALUDE_y_equation_proof_l3057_305788

theorem y_equation_proof (y : ℝ) (h : y + 1/y = 3) : y^6 - 8*y^3 + 4*y = 20*y - 5 := by
  sorry

end NUMINAMATH_CALUDE_y_equation_proof_l3057_305788


namespace NUMINAMATH_CALUDE_no_equal_group_division_l3057_305747

theorem no_equal_group_division (k : ℕ) : 
  ¬ ∃ (g1 g2 : List ℕ), 
    (∀ n, n ∈ g1 ∪ g2 ↔ 1 ≤ n ∧ n ≤ k) ∧ 
    (∀ n, n ∈ g1 → n ∉ g2) ∧
    (∀ n, n ∈ g2 → n ∉ g1) ∧
    (g1.foldl (λ acc x => acc * 10 + x) 0 = g2.foldl (λ acc x => acc * 10 + x) 0) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_group_division_l3057_305747


namespace NUMINAMATH_CALUDE_sally_sock_order_l3057_305710

/-- The ratio of black socks to blue socks in Sally's original order -/
def sock_ratio : ℚ := 5

theorem sally_sock_order :
  ∀ (x : ℝ) (b : ℕ),
  x > 0 →  -- Price of black socks is positive
  b > 0 →  -- Number of blue socks is positive
  (5 * x + 3 * b * x) * 2 = b * x + 15 * x →  -- Doubled bill condition
  sock_ratio = 5 := by
sorry

end NUMINAMATH_CALUDE_sally_sock_order_l3057_305710


namespace NUMINAMATH_CALUDE_num_special_words_is_35280_l3057_305738

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 21

/-- The number of six-letter words that begin and end with the same vowel,
    alternate between vowels and consonants, and start with a vowel -/
def num_special_words : ℕ := num_vowels * num_consonants * (num_vowels - 1) * num_consonants * (num_vowels - 1)

/-- Theorem stating that the number of special words is 35280 -/
theorem num_special_words_is_35280 : num_special_words = 35280 := by sorry

end NUMINAMATH_CALUDE_num_special_words_is_35280_l3057_305738


namespace NUMINAMATH_CALUDE_sum_of_fractions_bound_l3057_305701

theorem sum_of_fractions_bound (x y z : ℝ) (h : |x*y*z| = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_bound_l3057_305701


namespace NUMINAMATH_CALUDE_initial_liquid_A_amount_l3057_305717

/-- Proves that the initial amount of liquid A in a can is 36.75 litres given the specified conditions -/
theorem initial_liquid_A_amount
  (initial_ratio_A : ℚ)
  (initial_ratio_B : ℚ)
  (drawn_off_amount : ℚ)
  (new_ratio_A : ℚ)
  (new_ratio_B : ℚ)
  (h1 : initial_ratio_A = 7)
  (h2 : initial_ratio_B = 5)
  (h3 : drawn_off_amount = 18)
  (h4 : new_ratio_A = 7)
  (h5 : new_ratio_B = 9) :
  ∃ (initial_A : ℚ),
    initial_A = 36.75 ∧
    (initial_A / (initial_A * initial_ratio_B / initial_ratio_A) = initial_ratio_A / initial_ratio_B) ∧
    ((initial_A - drawn_off_amount * initial_ratio_A / (initial_ratio_A + initial_ratio_B)) /
     (initial_A * initial_ratio_B / initial_ratio_A - drawn_off_amount * initial_ratio_B / (initial_ratio_A + initial_ratio_B) + drawn_off_amount) =
     new_ratio_A / new_ratio_B) :=
by sorry

end NUMINAMATH_CALUDE_initial_liquid_A_amount_l3057_305717


namespace NUMINAMATH_CALUDE_problem_solution_l3057_305715

theorem problem_solution :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (max_value : ℝ), (a + 3*b + 3/a + 4/b = 18) → max_value = 9 + 3*Real.sqrt 6 ∧ a + 3*b ≤ max_value) ∧
  (a > b → ∃ (min_value : ℝ), min_value = 32 ∧ a^2 + 64 / (b*(a-b)) ≥ min_value) ∧
  (∃ (min_value : ℝ), (1/(a+1) + 1/(b+2) = 1/3) → min_value = 14 + 6*Real.sqrt 6 ∧ a*b + a + b ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3057_305715


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l3057_305766

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 5x - 3 and y = (3c)x + 1 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * c) * x + 1) ↔ c = 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l3057_305766


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3057_305778

theorem inequality_system_solution_set :
  ∀ x : ℝ, (3 * x - 1 ≥ x + 1 ∧ x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3057_305778


namespace NUMINAMATH_CALUDE_locus_all_importance_l3057_305794

/-- Definition of a locus --/
def Locus (P : Type*) (condition : P → Prop) : Set P :=
  {p : P | condition p}

/-- Property of comprehensiveness --/
def Comprehensive (S : Set P) (condition : P → Prop) : Prop :=
  ∀ p, condition p → p ∈ S

/-- Property of exclusivity --/
def Exclusive (S : Set P) (condition : P → Prop) : Prop :=
  ∀ p, p ∈ S → condition p

/-- Theorem: The definition of locus ensures both comprehensiveness and exclusivity --/
theorem locus_all_importance {P : Type*} (condition : P → Prop) :
  let L := Locus P condition
  Comprehensive L condition ∧ Exclusive L condition := by
  sorry

end NUMINAMATH_CALUDE_locus_all_importance_l3057_305794


namespace NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l3057_305755

theorem x_power_six_plus_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^6 + 1/x^6 = 322 := by
  sorry

end NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l3057_305755


namespace NUMINAMATH_CALUDE_total_pizza_slices_l3057_305716

theorem total_pizza_slices : 
  let number_of_pizzas : ℕ := 17
  let slices_per_pizza : ℕ := 4
  number_of_pizzas * slices_per_pizza = 68 := by
sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l3057_305716


namespace NUMINAMATH_CALUDE_factorization_equality_l3057_305749

theorem factorization_equality (a b : ℝ) :
  2*a*b^2 - 6*a^2*b^2 + 4*a^3*b^2 = 2*a*b^2*(2*a - 1)*(a - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l3057_305749


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3057_305703

theorem profit_percent_calculation (selling_price cost_price : ℝ) :
  cost_price = 0.25 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3057_305703


namespace NUMINAMATH_CALUDE_min_area_k_sum_l3057_305739

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The theorem stating the sum of k values that minimize the triangle area -/
theorem min_area_k_sum :
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨10, 20⟩
  let p3 (k : ℤ) : Point := ⟨7, k⟩
  let minArea := fun (k : ℤ) ↦ triangleArea p1 p2 (p3 k)
  ∃ (k1 k2 : ℤ),
    (∀ (k : ℤ), minArea k ≥ minArea k1 ∧ minArea k ≥ minArea k2) ∧
    k1 + k2 = 29 :=
sorry


end NUMINAMATH_CALUDE_min_area_k_sum_l3057_305739


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3057_305783

theorem inequality_system_solution (x : ℝ) : 
  (2 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 20) →
  (1 / (x - 2) > 1 / 5) →
  (x ∈ Set.Ioo 2 3) ∨ (x ∈ Set.Ioo 4 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3057_305783


namespace NUMINAMATH_CALUDE_pages_revised_twice_l3057_305729

/-- Represents the manuscript typing scenario -/
structure ManuscriptTyping where
  totalPages : Nat
  revisedOnce : Nat
  revisedTwice : Nat
  firstTypingCost : Nat
  revisionCost : Nat
  totalCost : Nat

/-- Calculates the total cost of typing and revising a manuscript -/
def calculateTotalCost (m : ManuscriptTyping) : Nat :=
  m.firstTypingCost * m.totalPages + 
  m.revisionCost * m.revisedOnce + 
  2 * m.revisionCost * m.revisedTwice

/-- Theorem stating that given the specified conditions, 30 pages were revised twice -/
theorem pages_revised_twice (m : ManuscriptTyping) 
  (h1 : m.totalPages = 100)
  (h2 : m.revisedOnce = 20)
  (h3 : m.firstTypingCost = 10)
  (h4 : m.revisionCost = 5)
  (h5 : m.totalCost = 1400)
  (h6 : calculateTotalCost m = m.totalCost) :
  m.revisedTwice = 30 := by
  sorry

end NUMINAMATH_CALUDE_pages_revised_twice_l3057_305729
