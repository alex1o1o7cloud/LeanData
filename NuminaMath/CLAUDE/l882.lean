import Mathlib

namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l882_88210

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l882_88210


namespace NUMINAMATH_CALUDE_seven_sum_to_8000_l882_88288

theorem seven_sum_to_8000 : ∃! (count : ℕ), count = 111 ∧ 
  (∀ n : ℕ, (∃ (a b c : ℕ), 7 * a + 77 * b + 7777 * c = 8000 ∧ n = a + 2 * b + 4 * c) ↔ n ∈ Finset.range count) :=
sorry

end NUMINAMATH_CALUDE_seven_sum_to_8000_l882_88288


namespace NUMINAMATH_CALUDE_intersection_equals_T_l882_88290

noncomputable def S : Set ℝ := {y | ∃ x : ℝ, y = x^3}
noncomputable def T : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

theorem intersection_equals_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_intersection_equals_T_l882_88290


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l882_88295

theorem quadratic_always_positive (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l882_88295


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l882_88237

-- Define the ellipse C
def ellipse_C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define point P
structure Point_P where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C a b x y
  on_parabola : parabola_E x y
  first_quadrant : x > 0 ∧ y > 0

-- Define the tangent line PF₁
def tangent_line (P : Point_P) (x y : ℝ) : Prop :=
  y = (P.y / (P.x + 1)) * (x + 1)

theorem ellipse_major_axis_length
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (P : Point_P)
  (h3 : tangent_line P P.x P.y) :
  2 * a = 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l882_88237


namespace NUMINAMATH_CALUDE_prime_product_divisible_by_seven_l882_88228

theorem prime_product_divisible_by_seven (C D : ℕ+) 
  (hC : Nat.Prime C)
  (hD : Nat.Prime D)
  (hCmD : Nat.Prime (C - D))
  (hCpD : Nat.Prime (C + D)) :
  7 ∣ (C - D) * C * D * (C + D) := by
sorry

end NUMINAMATH_CALUDE_prime_product_divisible_by_seven_l882_88228


namespace NUMINAMATH_CALUDE_sphere_always_circular_cross_section_l882_88255

-- Define the basic geometric shapes
inductive GeometricShape
  | Cone
  | Sphere
  | Cylinder
  | Prism

-- Define a predicate for having a circular cross section
def hasCircularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_always_circular_cross_section :
  ∀ (shape : GeometricShape),
    hasCircularCrossSection shape ↔ shape = GeometricShape.Sphere := by
  sorry

end NUMINAMATH_CALUDE_sphere_always_circular_cross_section_l882_88255


namespace NUMINAMATH_CALUDE_circles_internally_tangent_with_common_tangent_l882_88299

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12*y + 4 = 0

-- Define the common tangent line
def common_tangent (x y : ℝ) : Prop := 3*x + 4*y = 0

-- Theorem statement
theorem circles_internally_tangent_with_common_tangent :
  ∃ (x₀ y₀ : ℝ),
    (circle_M x₀ y₀ ∧ circle_N x₀ y₀) ∧  -- Circles are internally tangent
    (∀ x y, circle_M x y ∧ circle_N x y → x = x₀ ∧ y = y₀) ∧  -- Only one intersection point
    (∀ x y, common_tangent x y →  -- Common tangent is tangent to both circles
      (∃ ε > 0, ∀ δ ∈ Set.Ioo (-ε) ε,
        ¬(circle_M (x + δ) (y - 3*δ/4) ∧ circle_N (x + δ) (y - 3*δ/4)))) :=
by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_with_common_tangent_l882_88299


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l882_88219

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

def is_geometric_progression (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b / a = c / b

theorem smallest_third_term_of_geometric_progression :
  ∀ a b c : ℝ,
  a = 5 →
  is_arithmetic_progression a b c →
  is_geometric_progression 5 (b + 3) (c + 12) →
  ∃ d : ℝ, d ≥ 0 ∧ is_geometric_progression 5 (b + 3) d ∧
    ∀ e : ℝ, e ≥ 0 → is_geometric_progression 5 (b + 3) e → d ≤ e :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l882_88219


namespace NUMINAMATH_CALUDE_alex_gumballs_problem_l882_88265

theorem alex_gumballs_problem : ∃ n : ℕ, 
  n ≥ 50 ∧ 
  n % 7 = 5 ∧ 
  ∀ m : ℕ, (m ≥ 50 ∧ m % 7 = 5) → m ≥ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_alex_gumballs_problem_l882_88265


namespace NUMINAMATH_CALUDE_power_equality_l882_88293

theorem power_equality (n b : ℝ) (h1 : n = 2^(1/4)) (h2 : n^b = 8) : b = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l882_88293


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l882_88248

/-- Represents the molecular weight of a compound in g/mol -/
def molecular_weight : ℝ := 816

/-- Represents the number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- Theorem stating that the molecular weight remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  ∀ n : ℝ, n > 0 → molecular_weight = 816 := by
  sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l882_88248


namespace NUMINAMATH_CALUDE_delivery_problem_l882_88208

theorem delivery_problem (total_bottles : ℕ) (cider_bottles : ℕ) (beer_bottles : ℕ) 
  (h1 : total_bottles = 180)
  (h2 : cider_bottles = 40)
  (h3 : beer_bottles = 80)
  (h4 : cider_bottles + beer_bottles < total_bottles) :
  (cider_bottles / 2) + (beer_bottles / 2) + ((total_bottles - cider_bottles - beer_bottles) / 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_delivery_problem_l882_88208


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l882_88272

/-- The number of players on the team -/
def total_players : ℕ := 15

/-- The size of the starting lineup -/
def lineup_size : ℕ := 6

/-- The number of players guaranteed to be in the starting lineup -/
def guaranteed_players : ℕ := 3

/-- The number of remaining players to choose from -/
def remaining_players : ℕ := total_players - guaranteed_players

/-- The number of additional players needed to complete the lineup -/
def players_to_choose : ℕ := lineup_size - guaranteed_players

theorem starting_lineup_combinations :
  Nat.choose remaining_players players_to_choose = 220 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l882_88272


namespace NUMINAMATH_CALUDE_first_class_occupancy_is_three_l882_88240

/-- Represents the seating configuration and occupancy of an airplane -/
structure Airplane where
  first_class_capacity : ℕ
  business_class_capacity : ℕ
  economy_class_capacity : ℕ
  economy_occupancy : ℕ
  business_occupancy : ℕ
  first_class_occupancy : ℕ

/-- Theorem stating the number of people in first class -/
theorem first_class_occupancy_is_three (plane : Airplane) : plane.first_class_occupancy = 3 :=
  by
  have h1 : plane.first_class_capacity = 10 := by sorry
  have h2 : plane.business_class_capacity = 30 := by sorry
  have h3 : plane.economy_class_capacity = 50 := by sorry
  have h4 : plane.economy_occupancy = plane.economy_class_capacity / 2 := by sorry
  have h5 : plane.first_class_occupancy + plane.business_occupancy = plane.economy_occupancy := by sorry
  have h6 : plane.business_class_capacity - plane.business_occupancy = 8 := by sorry
  sorry

#check first_class_occupancy_is_three

end NUMINAMATH_CALUDE_first_class_occupancy_is_three_l882_88240


namespace NUMINAMATH_CALUDE_complex_equation_solution_l882_88227

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l882_88227


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l882_88262

/-- The function f(x) = 4x^2 - kx - 8 is monotonic on [5, 8] iff k ∈ (-∞, 40] ∪ [64, +∞) -/
theorem monotonic_quadratic_function (k : ℝ) :
  (∀ x ∈ Set.Icc 5 8, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ 
  (k ≤ 40 ∨ k ≥ 64) := by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l882_88262


namespace NUMINAMATH_CALUDE_luke_trivia_rounds_l882_88220

/-- Given that Luke scored a total of 300 points in a trivia game, 
    gained the same number of points each round, and scored 60 points per round, 
    prove that he played 5 rounds. -/
theorem luke_trivia_rounds (total_points : ℕ) (points_per_round : ℕ) (rounds : ℕ) : 
  total_points = 300 ∧ 
  points_per_round = 60 ∧ 
  total_points = points_per_round * rounds → 
  rounds = 5 := by
sorry

end NUMINAMATH_CALUDE_luke_trivia_rounds_l882_88220


namespace NUMINAMATH_CALUDE_joans_kittens_l882_88214

theorem joans_kittens (initial_kittens : ℕ) (given_away : ℕ) (remaining : ℕ) 
  (h1 : given_away = 2) 
  (h2 : remaining = 6) 
  (h3 : initial_kittens = remaining + given_away) : initial_kittens = 8 :=
by sorry

end NUMINAMATH_CALUDE_joans_kittens_l882_88214


namespace NUMINAMATH_CALUDE_saline_mixture_concentration_l882_88268

/-- Proves that mixing 3.6L of 1% saline and 1.4L of 9% saline results in 5L of 3.24% saline -/
theorem saline_mixture_concentration :
  let vol_1_percent : ℝ := 3.6
  let vol_9_percent : ℝ := 1.4
  let total_volume : ℝ := 5
  let concentration_1_percent : ℝ := 0.01
  let concentration_9_percent : ℝ := 0.09
  let resulting_concentration : ℝ := (vol_1_percent * concentration_1_percent + 
                                      vol_9_percent * concentration_9_percent) / total_volume
  resulting_concentration = 0.0324 := by
sorry

#eval (3.6 * 0.01 + 1.4 * 0.09) / 5

end NUMINAMATH_CALUDE_saline_mixture_concentration_l882_88268


namespace NUMINAMATH_CALUDE_water_removal_for_concentration_l882_88202

/-- 
Proves that the amount of water removed to concentrate a 40% acidic liquid to 60% acidic liquid 
is 5 liters, given that the final volume is 5 liters less than the initial volume.
-/
theorem water_removal_for_concentration (initial_volume : ℝ) : 
  initial_volume > 0 →
  let initial_concentration : ℝ := 0.4
  let final_concentration : ℝ := 0.6
  let volume_decrease : ℝ := 5
  let final_volume : ℝ := initial_volume - volume_decrease
  let water_removed : ℝ := volume_decrease
  initial_concentration * initial_volume = final_concentration * final_volume →
  water_removed = 5 := by
sorry

end NUMINAMATH_CALUDE_water_removal_for_concentration_l882_88202


namespace NUMINAMATH_CALUDE_lab_expense_ratio_l882_88205

/-- Given a laboratory budget and expenses, prove the ratio of test tube cost to flask cost -/
theorem lab_expense_ratio (total_budget flask_cost remaining : ℚ) : 
  total_budget = 325 →
  flask_cost = 150 →
  remaining = 25 →
  ∃ (test_tube_cost : ℚ),
    total_budget = flask_cost + test_tube_cost + (test_tube_cost / 2) + remaining →
    test_tube_cost / flask_cost = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_lab_expense_ratio_l882_88205


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l882_88200

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of BM, where M is the point where the circle touches AB -/
  bm : ℝ
  /-- The length of the top side CD -/
  cd : ℝ

/-- The area of a trapezoid with an inscribed circle -/
def trapezoidArea (t : InscribedCircleTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific trapezoid is 108 -/
theorem specific_trapezoid_area :
  ∃ t : InscribedCircleTrapezoid, t.r = 4 ∧ t.bm = 16 ∧ t.cd = 3 ∧ trapezoidArea t = 108 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l882_88200


namespace NUMINAMATH_CALUDE_second_player_wins_l882_88211

/-- Represents the state of the game -/
structure GameState where
  pile1 : Nat
  pile2 : Nat

/-- Represents a move in the game -/
structure Move where
  pile : Nat -- 1 or 2
  balls : Nat

/-- Defines a valid move in the game -/
def validMove (state : GameState) (move : Move) : Prop :=
  (move.pile = 1 ∧ move.balls ≤ state.pile1) ∨
  (move.pile = 2 ∧ move.balls ≤ state.pile2)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  if move.pile = 1 then
    { pile1 := state.pile1 - move.balls, pile2 := state.pile2 }
  else
    { pile1 := state.pile1, pile2 := state.pile2 - move.balls }

/-- Defines the winning condition -/
def isWinningState (state : GameState) : Prop :=
  state.pile1 = 0 ∧ state.pile2 = 0

/-- Defines a winning strategy for the second player -/
def secondPlayerWinningStrategy (initialState : GameState) : Prop :=
  ∀ (move : Move), 
    validMove initialState move → 
    ∃ (response : Move), 
      validMove (applyMove initialState move) response ∧
      (applyMove (applyMove initialState move) response).pile1 = 
      (applyMove (applyMove initialState move) response).pile2

/-- Theorem: The second player has a winning strategy in the two-pile game -/
theorem second_player_wins (initialState : GameState) 
  (h1 : initialState.pile1 = 30) (h2 : initialState.pile2 = 30) : 
  secondPlayerWinningStrategy initialState :=
sorry


end NUMINAMATH_CALUDE_second_player_wins_l882_88211


namespace NUMINAMATH_CALUDE_nine_fourth_cubed_eq_three_to_nine_l882_88241

theorem nine_fourth_cubed_eq_three_to_nine :
  9^4 + 9^4 + 9^4 = 3^9 := by sorry

end NUMINAMATH_CALUDE_nine_fourth_cubed_eq_three_to_nine_l882_88241


namespace NUMINAMATH_CALUDE_intersection_properties_l882_88233

/-- A line passing through point (1,1) with an angle of inclination π/4 -/
def line_l : Set (ℝ × ℝ) :=
  {p | p.2 - 1 = p.1 - 1}

/-- A parabola defined by y² = x + 1 -/
def parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = p.1 + 1}

/-- The intersection points of the line and parabola -/
def intersection_points : Set (ℝ × ℝ) :=
  line_l ∩ parabola

/-- Point P -/
def P : ℝ × ℝ := (1, 1)

/-- Theorem stating the properties of the intersection -/
theorem intersection_properties :
  ∃ (A B : ℝ × ℝ) (M : ℝ × ℝ),
    A ∈ intersection_points ∧
    B ∈ intersection_points ∧
    A ≠ B ∧
    (‖A - P‖ * ‖B - P‖ = Real.sqrt 10) ∧
    (‖A - B‖ = Real.sqrt 10) ∧
    (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
    (M = (1/2, 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_intersection_properties_l882_88233


namespace NUMINAMATH_CALUDE_expression_evaluation_l882_88257

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 
  2 * x * y * (x^2 * y^2) + 8 / (x * y) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l882_88257


namespace NUMINAMATH_CALUDE_seats_per_row_is_eight_l882_88229

/-- The number of people that fit in a row on an airplane, given the specified conditions. -/
def seats_per_row : ℕ := by
  sorry

theorem seats_per_row_is_eight :
  let total_rows : ℕ := 12
  let occupancy_rate : ℚ := 3/4
  let unoccupied_seats : ℕ := 24
  seats_per_row = 8 := by
  sorry

end NUMINAMATH_CALUDE_seats_per_row_is_eight_l882_88229


namespace NUMINAMATH_CALUDE_new_rectangle_area_comparison_l882_88275

theorem new_rectangle_area_comparison (a : ℝ) (h : a > 0) :
  let original_diagonal := Real.sqrt (4 * a^2 + 9 * a^2)
  let new_base := original_diagonal + 3 * a
  let new_height := 9 * a - (1/2) * original_diagonal
  let new_area := new_base * new_height
  let original_area := 2 * a * 3 * a
  new_area > 2 * original_area := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_comparison_l882_88275


namespace NUMINAMATH_CALUDE_range_of_m_for_perpendicular_vectors_l882_88246

/-- Given two points M(-1,0) and N(1,0), and a line 3x - 4y + m = 0,
    if there exists a point P on the line such that PM · PN = 0,
    then the range of values for m is [-5, 5]. -/
theorem range_of_m_for_perpendicular_vectors :
  let M : ℝ × ℝ := (-1, 0)
  let N : ℝ × ℝ := (1, 0)
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | 3 * x - 4 * y + m = 0}
  ∀ m : ℝ, (∃ P ∈ line m, (P.1 - M.1) * (N.1 - P.1) + (P.2 - M.2) * (N.2 - P.2) = 0) ↔ 
    m ∈ Set.Icc (-5 : ℝ) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_perpendicular_vectors_l882_88246


namespace NUMINAMATH_CALUDE_ravi_selection_probability_l882_88212

theorem ravi_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 4/7)
  (h2 : p_both = 0.11428571428571428) :
  p_both / p_ram = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ravi_selection_probability_l882_88212


namespace NUMINAMATH_CALUDE_employee_assignment_l882_88251

/-- The number of ways to assign employees to workshops -/
def assign_employees (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  (n.choose (k - 1)) * (k.factorial)

/-- Theorem: Assigning 5 employees to 3 workshops with constraints -/
theorem employee_assignment :
  let total_employees : ℕ := 5
  let workshops : ℕ := 3
  let effective_employees : ℕ := total_employees - 1  -- Considering A and B as one entity
  assign_employees effective_employees workshops workshops = 36 := by
  sorry


end NUMINAMATH_CALUDE_employee_assignment_l882_88251


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l882_88250

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℝ
  day : ℕ

/-- Represents the volume of orangeade made on a given day -/
structure OrangeadeVolume where
  volume : ℝ
  day : ℕ

/-- Represents the revenue from selling orangeade on a given day -/
def revenue (p : OrangeadePrice) (v : OrangeadeVolume) : ℝ :=
  p.price * v.volume

theorem orangeade_price_day2 
  (juice : ℝ) -- Amount of orange juice used (same for both days)
  (v1 : OrangeadeVolume) -- Volume of orangeade on day 1
  (v2 : OrangeadeVolume) -- Volume of orangeade on day 2
  (p1 : OrangeadePrice) -- Price of orangeade on day 1
  (p2 : OrangeadePrice) -- Price of orangeade on day 2
  (h1 : v1.volume = 2 * juice) -- Volume on day 1 is twice the amount of juice
  (h2 : v2.volume = 3 * juice) -- Volume on day 2 is thrice the amount of juice
  (h3 : v1.day = 1 ∧ v2.day = 2) -- Volumes correspond to days 1 and 2
  (h4 : p1.day = 1 ∧ p2.day = 2) -- Prices correspond to days 1 and 2
  (h5 : p1.price = 0.6) -- Price on day 1 is $0.60
  (h6 : revenue p1 v1 = revenue p2 v2) -- Revenue is the same for both days
  : p2.price = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l882_88250


namespace NUMINAMATH_CALUDE_home_learning_percentage_l882_88217

-- Define the percentage of students present in school
def students_present : ℝ := 30

-- Define the theorem
theorem home_learning_percentage :
  let total_percentage : ℝ := 100
  let non_home_learning : ℝ := 2 * students_present
  let home_learning : ℝ := total_percentage - non_home_learning
  home_learning = 40 := by sorry

end NUMINAMATH_CALUDE_home_learning_percentage_l882_88217


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l882_88269

/-- Given a rhombus with one diagonal of 65 meters and an area of 1950 square meters,
    prove that the length of the other diagonal is 60 meters. -/
theorem rhombus_diagonal (d₁ : ℝ) (area : ℝ) (d₂ : ℝ) : 
  d₁ = 65 → area = 1950 → area = (d₁ * d₂) / 2 → d₂ = 60 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l882_88269


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l882_88216

theorem cubic_equation_solution :
  ∃! x : ℝ, (x^3 - 5*x^2 + 5*x - 1) + (x - 1) = 0 :=
by
  -- The unique solution is x = 2
  use 2
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l882_88216


namespace NUMINAMATH_CALUDE_black_squares_in_56th_row_l882_88282

/-- Represents the number of squares in a row of the geometric pattern -/
def squares_in_row (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- Represents the number of black squares in a row of the geometric pattern -/
def black_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

theorem black_squares_in_56th_row :
  black_squares_in_row 56 = 56 := by sorry

end NUMINAMATH_CALUDE_black_squares_in_56th_row_l882_88282


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l882_88201

/-- A function g satisfying the given property -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → g x + g (1 / x^2) = x^2

/-- The domain of g is the set of all non-zero real numbers -/
theorem largest_domain_of_g :
  ∃ g : ℝ → ℝ, g_property g ∧
  ∀ S : Set ℝ, (∃ h : ℝ → ℝ, g_property h ∧ ∀ x ∈ S, h x ≠ 0) →
  S ⊆ {x : ℝ | x ≠ 0} :=
sorry

end NUMINAMATH_CALUDE_largest_domain_of_g_l882_88201


namespace NUMINAMATH_CALUDE_car_meeting_distance_l882_88224

/-- Two cars moving towards each other -/
structure CarMeeting where
  speedA : ℝ
  speedB : ℝ
  meetingTime : ℝ
  speedRatio : ℝ

/-- The initial distance between two cars given their speeds and meeting time -/
def initialDistance (m : CarMeeting) : ℝ :=
  (m.speedA + m.speedB) * m.meetingTime

theorem car_meeting_distance (m : CarMeeting) 
  (h1 : m.speedRatio = 5 / 6)
  (h2 : m.speedB = 90)
  (h3 : m.meetingTime = 32 / 60) 
  (h4 : m.speedA = m.speedB * m.speedRatio) :
  initialDistance m = 88 := by
  sorry

end NUMINAMATH_CALUDE_car_meeting_distance_l882_88224


namespace NUMINAMATH_CALUDE_box_filling_cubes_l882_88286

/-- Given a box with dimensions 49 inches long, 42 inches wide, and 14 inches deep,
    the smallest number of identical cubes that can completely fill the box without
    leaving any space is 84. -/
theorem box_filling_cubes : ∀ (length width depth : ℕ),
  length = 49 → width = 42 → depth = 14 →
  ∃ (cube_side : ℕ), cube_side > 0 ∧
    length % cube_side = 0 ∧
    width % cube_side = 0 ∧
    depth % cube_side = 0 ∧
    (length / cube_side) * (width / cube_side) * (depth / cube_side) = 84 :=
by sorry

end NUMINAMATH_CALUDE_box_filling_cubes_l882_88286


namespace NUMINAMATH_CALUDE_plot_perimeter_l882_88209

/-- A rectangular plot with specific dimensions and fencing costs -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  length_eq : length = width + 10
  cost_eq : totalFencingCost = fencingCostPerMeter * (2 * (length + width))

/-- The perimeter of the rectangular plot is 340 meters -/
theorem plot_perimeter (plot : RectangularPlot) (h : plot.fencingCostPerMeter = 6.5 ∧ plot.totalFencingCost = 2210) :
  2 * (plot.length + plot.width) = 340 := by
  sorry


end NUMINAMATH_CALUDE_plot_perimeter_l882_88209


namespace NUMINAMATH_CALUDE_jeremy_overall_accuracy_l882_88270

theorem jeremy_overall_accuracy 
  (individual_portion : Real) 
  (collaborative_portion : Real)
  (terry_individual_accuracy : Real)
  (terry_overall_accuracy : Real)
  (jeremy_individual_accuracy : Real)
  (h1 : individual_portion = 0.6)
  (h2 : collaborative_portion = 0.4)
  (h3 : individual_portion + collaborative_portion = 1)
  (h4 : terry_individual_accuracy = 0.75)
  (h5 : terry_overall_accuracy = 0.85)
  (h6 : jeremy_individual_accuracy = 0.8) :
  jeremy_individual_accuracy * individual_portion + 
  (terry_overall_accuracy - terry_individual_accuracy * individual_portion) = 0.88 :=
sorry

end NUMINAMATH_CALUDE_jeremy_overall_accuracy_l882_88270


namespace NUMINAMATH_CALUDE_eddie_number_l882_88203

theorem eddie_number (n : ℕ) (m : ℕ) (h1 : n ≥ 40) (h2 : n % 5 = 0) (h3 : n % m = 0) :
  (∀ k : ℕ, k ≥ 40 ∧ k % 5 = 0 ∧ ∃ j : ℕ, k % j = 0 → k ≥ n) →
  n = 40 ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_eddie_number_l882_88203


namespace NUMINAMATH_CALUDE_fraction_product_l882_88267

theorem fraction_product : (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l882_88267


namespace NUMINAMATH_CALUDE_smaller_number_problem_l882_88221

theorem smaller_number_problem (x y : ℝ) : 
  y = 2 * x - 3 →   -- One number is 3 less than twice the other
  x + y = 39 →      -- Their sum is 39
  x = 14            -- The smaller number is 14
  := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l882_88221


namespace NUMINAMATH_CALUDE_intersection_condition_l882_88284

/-- The line y = k(x+1) intersects the circle (x-1)² + y² = 1 -/
def intersects (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * (x + 1) ∧ (x - 1)^2 + y^2 = 1

/-- The condition k > -√3/3 is neither sufficient nor necessary for intersection -/
theorem intersection_condition (k : ℝ) :
  ¬(k > -Real.sqrt 3 / 3 → intersects k) ∧
  ¬(intersects k → k > -Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l882_88284


namespace NUMINAMATH_CALUDE_ryan_chinese_hours_l882_88238

def hours_english : ℕ := 2
def hours_spanish : ℕ := 4

theorem ryan_chinese_hours :
  ∀ hours_chinese : ℕ,
  hours_chinese = hours_spanish + 1 →
  hours_chinese = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_hours_l882_88238


namespace NUMINAMATH_CALUDE_first_month_sale_l882_88276

def sales_4_months : List Int := [6927, 6855, 7230, 6562]
def average_6_months : Int := 6500
def sale_6th_month : Int := 4691
def num_months : Int := 6

theorem first_month_sale (sales_4_months : List Int) 
                         (average_6_months : Int) 
                         (sale_6th_month : Int) 
                         (num_months : Int) : 
  sales_4_months = [6927, 6855, 7230, 6562] →
  average_6_months = 6500 →
  sale_6th_month = 4691 →
  num_months = 6 →
  (List.sum sales_4_months + sale_6th_month + 6735) / num_months = average_6_months :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l882_88276


namespace NUMINAMATH_CALUDE_max_batteries_produced_l882_88225

/-- Represents the production capacity of a robot type -/
structure RobotCapacity where
  time_per_battery : ℕ
  num_robots : ℕ

/-- Calculates the number of batteries a robot type can produce in a given time -/
def batteries_produced (capacity : RobotCapacity) (total_time : ℕ) : ℕ :=
  (total_time / capacity.time_per_battery) * capacity.num_robots

/-- Theorem: The maximum number of batteries produced is limited by the lowest production capacity -/
theorem max_batteries_produced 
  (robot_a : RobotCapacity) 
  (robot_b : RobotCapacity) 
  (robot_c : RobotCapacity) 
  (total_time : ℕ) :
  robot_a.time_per_battery = 6 →
  robot_b.time_per_battery = 9 →
  robot_c.time_per_battery = 3 →
  robot_a.num_robots = 8 →
  robot_b.num_robots = 6 →
  robot_c.num_robots = 6 →
  total_time = 300 →
  min (batteries_produced robot_a total_time) 
      (min (batteries_produced robot_b total_time) 
           (batteries_produced robot_c total_time)) = 198 :=
by sorry

end NUMINAMATH_CALUDE_max_batteries_produced_l882_88225


namespace NUMINAMATH_CALUDE_raduzhny_residents_l882_88231

/-- The number of villages in Solar Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe village -/
def znoynoe_population : ℕ := 1000

/-- The amount by which Znoynoe's population exceeds the average -/
def excess_population : ℕ := 90

/-- The total population of all villages in Solar Valley -/
def total_population : ℕ := znoynoe_population + (num_villages - 1) * (znoynoe_population - excess_population)

/-- The population of Raduzhny village -/
def raduzhny_population : ℕ := znoynoe_population - excess_population

theorem raduzhny_residents : raduzhny_population = 900 := by
  sorry

end NUMINAMATH_CALUDE_raduzhny_residents_l882_88231


namespace NUMINAMATH_CALUDE_gold_cost_calculation_l882_88245

/-- The cost of Gary and Anna's combined gold -/
def combined_gold_cost (gary_grams : ℝ) (gary_price : ℝ) (anna_grams : ℝ) (anna_price : ℝ) : ℝ :=
  gary_grams * gary_price + anna_grams * anna_price

/-- Theorem stating the combined cost of Gary and Anna's gold -/
theorem gold_cost_calculation :
  combined_gold_cost 30 15 50 20 = 1450 := by
  sorry

end NUMINAMATH_CALUDE_gold_cost_calculation_l882_88245


namespace NUMINAMATH_CALUDE_points_on_same_line_l882_88242

def point_A : ℝ × ℝ := (-1, 0.5)
def point_B : ℝ × ℝ := (3, -3.5)
def point_C : ℝ × ℝ := (7, -7.5)

def collinear (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem points_on_same_line : collinear point_A point_B point_C := by
  sorry

end NUMINAMATH_CALUDE_points_on_same_line_l882_88242


namespace NUMINAMATH_CALUDE_speed_limit_proof_l882_88252

/-- Prove that the speed limit is 50 mph given the conditions of Natasha's travel --/
theorem speed_limit_proof (natasha_speed : ℝ) (speed_limit : ℝ) (time : ℝ) (distance : ℝ) :
  natasha_speed = speed_limit + 10 →
  time = 1 →
  distance = 60 →
  natasha_speed = distance / time →
  speed_limit = 50 := by
sorry

end NUMINAMATH_CALUDE_speed_limit_proof_l882_88252


namespace NUMINAMATH_CALUDE_medical_team_selection_l882_88273

theorem medical_team_selection (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 6) (h2 : female_doctors = 5) :
  Nat.choose male_doctors 2 * Nat.choose female_doctors 1 = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l882_88273


namespace NUMINAMATH_CALUDE_xy_equals_three_l882_88243

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l882_88243


namespace NUMINAMATH_CALUDE_profit_division_l882_88280

/-- Given a profit divided between X and Y in the ratio 1/2 : 1/3, where the difference
    between their profit shares is 200, prove that the total profit amount is 1000. -/
theorem profit_division (profit_x profit_y : ℝ) 
    (h_ratio : profit_x / profit_y = (1 : ℝ) / 2 / ((1 : ℝ) / 3))
    (h_diff : profit_x - profit_y = 200) :
    profit_x + profit_y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_profit_division_l882_88280


namespace NUMINAMATH_CALUDE_equilateral_triangle_line_theorem_l882_88247

/-- Given an equilateral triangle ABC with side length a and a line A₁B₁ passing through its center O,
    cutting segments x and y from sides AC and BC respectively, prove that 3xy - 2a(x + y) + a² = 0 --/
theorem equilateral_triangle_line_theorem
  (a x y : ℝ)
  (h_positive : a > 0)
  (h_x_positive : x > 0)
  (h_y_positive : y > 0)
  (h_x_bound : x < a)
  (h_y_bound : y < a) :
  3 * x * y - 2 * a * (x + y) + a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_line_theorem_l882_88247


namespace NUMINAMATH_CALUDE_democrat_ratio_l882_88249

theorem democrat_ratio (total_participants male_participants female_participants female_democrats : ℕ)
  (h1 : total_participants = 840)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : 3 * (female_democrats + male_participants / 4) = total_participants)
  (h4 : female_democrats = 140) :
  2 * female_democrats = female_participants :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l882_88249


namespace NUMINAMATH_CALUDE_tangent_line_value_l882_88261

/-- A line passing through point P(1, 2) is tangent to the circle x^2 + y^2 = 4 
    and perpendicular to the line ax - y + 1 = 0. The value of a is -3/4. -/
theorem tangent_line_value (a : ℝ) : 
  (∃ (m : ℝ), 
    -- Line equation: y - 2 = m(x - 1)
    (∀ x y : ℝ, y - 2 = m * (x - 1) → 
      -- Point P(1, 2) satisfies the line equation
      (1 : ℝ) - 1 = 0 ∧ 2 - 2 = 0 ∧
      -- Line is tangent to the circle
      ((x - 0)^2 + (y - 0)^2 = 4 → 
        (y - 2 = m * (x - 1) → x^2 + y^2 ≥ 4)) ∧
      -- Line is perpendicular to ax - y + 1 = 0
      m * a = -1)) →
  a = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_value_l882_88261


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l882_88226

-- Define the condition "1 < x < 2"
def condition (x : ℝ) : Prop := 1 < x ∧ x < 2

-- Define the statement "x < 2"
def statement (x : ℝ) : Prop := x < 2

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2"
theorem sufficient_not_necessary :
  (∀ x : ℝ, condition x → statement x) ∧
  (∃ x : ℝ, statement x ∧ ¬condition x) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l882_88226


namespace NUMINAMATH_CALUDE_dinner_bill_calculation_l882_88289

/-- Calculate the total cost of a food item including sales tax and service charge -/
def itemTotalCost (basePrice : ℚ) (salesTaxRate : ℚ) (serviceChargeRate : ℚ) : ℚ :=
  basePrice * (1 + salesTaxRate + serviceChargeRate)

/-- Calculate the total bill for the family's dinner -/
def totalBill : ℚ :=
  itemTotalCost 20 (7/100) (8/100) +
  itemTotalCost 15 (17/200) (1/10) +
  itemTotalCost 10 (3/50) (3/25)

/-- Theorem stating that the total bill is equal to $52.58 -/
theorem dinner_bill_calculation : 
  ∃ (n : ℕ), (n : ℚ) / 100 = totalBill ∧ n = 5258 :=
sorry

end NUMINAMATH_CALUDE_dinner_bill_calculation_l882_88289


namespace NUMINAMATH_CALUDE_domain_equivalence_l882_88271

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2^x)
def domain_f_exp : Set ℝ := Set.Ioo 1 2

-- Define the domain of f(√(x^2 - 1))
def domain_f_sqrt : Set ℝ := Set.union (Set.Ioo (-Real.sqrt 17) (-Real.sqrt 5)) (Set.Ioo (Real.sqrt 5) (Real.sqrt 17))

-- Theorem statement
theorem domain_equivalence (h : ∀ x ∈ domain_f_exp, f (2^x) = f (2^x)) :
  ∀ x ∈ domain_f_sqrt, f (Real.sqrt (x^2 - 1)) = f (Real.sqrt (x^2 - 1)) :=
sorry

end NUMINAMATH_CALUDE_domain_equivalence_l882_88271


namespace NUMINAMATH_CALUDE_impossible_to_use_all_components_l882_88253

theorem impossible_to_use_all_components (p q r : ℤ) : 
  ¬ ∃ (x y z : ℤ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
    (2 * x + y = 2 * p + q + 1) ∧ 
    (y + z = q + r) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_components_l882_88253


namespace NUMINAMATH_CALUDE_garden_ratio_l882_88292

theorem garden_ratio (area width length : ℝ) : 
  area = 507 →
  width = 13 →
  area = length * width →
  length / width = 3 := by
sorry

end NUMINAMATH_CALUDE_garden_ratio_l882_88292


namespace NUMINAMATH_CALUDE_crayon_count_l882_88260

theorem crayon_count (blue : ℕ) (red : ℕ) : 
  blue = 3 → red = 4 * blue → blue + red = 15 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_l882_88260


namespace NUMINAMATH_CALUDE_blue_balls_in_box_l882_88204

theorem blue_balls_in_box (purple_balls yellow_balls min_tries : ℕ) 
  (h1 : purple_balls = 7)
  (h2 : yellow_balls = 11)
  (h3 : min_tries = 19) :
  ∃! blue_balls : ℕ, 
    blue_balls > 0 ∧ 
    purple_balls + yellow_balls + blue_balls = min_tries :=
by
  sorry

end NUMINAMATH_CALUDE_blue_balls_in_box_l882_88204


namespace NUMINAMATH_CALUDE_quadratic_root_range_l882_88232

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x > 1 ∧ y < -1 ∧ 
   x^2 + (a^2 + 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 + 1)*y + a - 2 = 0) →
  -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l882_88232


namespace NUMINAMATH_CALUDE_smallest_n_with_specific_decimal_periods_l882_88285

/-- A function that checks if a fraction has a repeating decimal representation with a given period -/
def hasRepeatingDecimalPeriod (numerator : ℕ) (denominator : ℕ) (period : ℕ) : Prop :=
  ∃ k : ℕ, (10^period - 1) * numerator = k * denominator

/-- The smallest positive integer n less than 1000 such that 1/n has a repeating decimal
    representation with a period of 5 and 1/(n+7) has a repeating decimal representation
    with a period of 4 is 266 -/
theorem smallest_n_with_specific_decimal_periods : 
  ∃ n : ℕ, n > 0 ∧ n < 1000 ∧
    hasRepeatingDecimalPeriod 1 n 5 ∧
    hasRepeatingDecimalPeriod 1 (n + 7) 4 ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(hasRepeatingDecimalPeriod 1 m 5 ∧ hasRepeatingDecimalPeriod 1 (m + 7) 4)) ∧
    n = 266 :=
by
  sorry


end NUMINAMATH_CALUDE_smallest_n_with_specific_decimal_periods_l882_88285


namespace NUMINAMATH_CALUDE_circle_area_decrease_l882_88279

/-- Given a circle with initial area 25π, prove that a 10% decrease in diameter results in a 19% decrease in area -/
theorem circle_area_decrease (π : ℝ) (h_π : π > 0) : 
  let initial_area : ℝ := 25 * π
  let initial_radius : ℝ := (initial_area / π).sqrt
  let new_radius : ℝ := initial_radius * 0.9
  let new_area : ℝ := π * new_radius^2
  (initial_area - new_area) / initial_area = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l882_88279


namespace NUMINAMATH_CALUDE_winning_strategy_exists_l882_88239

/-- Represents the three jars --/
inductive Jar
  | one
  | two
  | three

/-- Represents the three players --/
inductive Player
  | W
  | R
  | P

/-- The state of the game, tracking the number of nuts in each jar --/
structure GameState where
  jar1 : Nat
  jar2 : Nat
  jar3 : Nat

/-- Defines valid moves for each player --/
def validMove (p : Player) (j : Jar) : Prop :=
  match p, j with
  | Player.W, Jar.one => True
  | Player.W, Jar.two => True
  | Player.R, Jar.two => True
  | Player.R, Jar.three => True
  | Player.P, Jar.one => True
  | Player.P, Jar.three => True
  | _, _ => False

/-- Defines a winning state (any jar contains exactly 1999 nuts) --/
def isWinningState (s : GameState) : Prop :=
  s.jar1 = 1999 ∨ s.jar2 = 1999 ∨ s.jar3 = 1999

/-- Defines a strategy for W and P --/
def Strategy := GameState → Player → Jar

/-- Theorem: There exists a strategy for W and P that forces R to lose --/
theorem winning_strategy_exists :
  ∃ (strat : Strategy),
    ∀ (initial_state : GameState),
      ∀ (moves : Nat → Player → Jar),
        (∀ (n : Nat) (p : Player), validMove p (moves n p)) →
        ∃ (n : Nat),
          let final_state := -- state after n moves
            sorry -- Implementation of game progression
          isWinningState final_state ∧ moves n Player.R = (moves n Player.R) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_exists_l882_88239


namespace NUMINAMATH_CALUDE_initial_boarders_l882_88259

/-- Proves that the initial number of boarders was 150 given the conditions of the problem -/
theorem initial_boarders (B D : ℕ) (h1 : B * 12 = D * 5) 
  (h2 : (B + 30) * 2 = D * 1) : B = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_boarders_l882_88259


namespace NUMINAMATH_CALUDE_cartons_per_box_l882_88256

/-- Given information about gum packaging and distribution, prove the number of cartons per box -/
theorem cartons_per_box 
  (packs_per_carton : ℕ) 
  (sticks_per_pack : ℕ)
  (total_sticks : ℕ)
  (total_boxes : ℕ)
  (h1 : packs_per_carton = 5)
  (h2 : sticks_per_pack = 3)
  (h3 : total_sticks = 480)
  (h4 : total_boxes = 8)
  : (total_sticks / total_boxes) / (packs_per_carton * sticks_per_pack) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cartons_per_box_l882_88256


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l882_88277

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 1) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l882_88277


namespace NUMINAMATH_CALUDE_solution_distribution_l882_88274

theorem solution_distribution (num_tubes : ℕ) (num_beakers : ℕ) (beaker_volume : ℚ) :
  num_tubes = 6 →
  num_beakers = 3 →
  beaker_volume = 14 →
  (num_beakers * beaker_volume) / num_tubes = 7 :=
by sorry

end NUMINAMATH_CALUDE_solution_distribution_l882_88274


namespace NUMINAMATH_CALUDE_expression_simplification_l882_88213

theorem expression_simplification (p : ℝ) :
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l882_88213


namespace NUMINAMATH_CALUDE_shoes_outside_library_l882_88234

/-- The number of people in the group -/
def num_people : ℕ := 10

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of shoes kept outside the library -/
def total_shoes : ℕ := num_people * shoes_per_person

theorem shoes_outside_library :
  total_shoes = 20 :=
by sorry

end NUMINAMATH_CALUDE_shoes_outside_library_l882_88234


namespace NUMINAMATH_CALUDE_remainder_sum_l882_88223

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 53) (hd : d % 40 = 29) :
  (c + d) % 20 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l882_88223


namespace NUMINAMATH_CALUDE_boys_without_laptops_l882_88222

theorem boys_without_laptops (total_boys : ℕ) (total_laptops : ℕ) (girls_with_laptops : ℕ) : 
  total_boys = 20 → 
  total_laptops = 25 → 
  girls_with_laptops = 16 → 
  total_boys - (total_laptops - girls_with_laptops) = 11 := by
sorry

end NUMINAMATH_CALUDE_boys_without_laptops_l882_88222


namespace NUMINAMATH_CALUDE_equal_sum_sequence_property_l882_88264

def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = a (n + 1) + a (n + 2)

theorem equal_sum_sequence_property (a : ℕ → ℝ) (h : is_equal_sum_sequence a) :
  (∀ k m : ℕ, Odd k → Odd m → a k = a m) ∧
  (∀ k m : ℕ, Even k → Even m → a k = a m) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_property_l882_88264


namespace NUMINAMATH_CALUDE_count_odd_two_digit_integers_l882_88281

/-- A function that returns true if a natural number is odd, false otherwise -/
def is_odd (n : ℕ) : Bool :=
  n % 2 = 1

/-- The set of odd digits (1, 3, 5, 7, 9) -/
def odd_digits : Finset ℕ :=
  {1, 3, 5, 7, 9}

/-- A function that returns true if a natural number is a two-digit integer, false otherwise -/
def is_two_digit (n : ℕ) : Bool :=
  10 ≤ n ∧ n ≤ 99

/-- The set of two-digit integers where both digits are odd -/
def odd_two_digit_integers : Finset ℕ :=
  Finset.filter (fun n => is_two_digit n ∧ is_odd (n / 10) ∧ is_odd (n % 10)) (Finset.range 100)

theorem count_odd_two_digit_integers : 
  Finset.card odd_two_digit_integers = 25 :=
sorry

end NUMINAMATH_CALUDE_count_odd_two_digit_integers_l882_88281


namespace NUMINAMATH_CALUDE_fuel_station_total_cost_l882_88294

/-- Calculates the total cost for filling up mini-vans and trucks at a fuel station -/
theorem fuel_station_total_cost
  (service_cost : ℝ)
  (fuel_cost_per_liter : ℝ)
  (num_minivans : ℕ)
  (num_trucks : ℕ)
  (minivan_tank_capacity : ℝ)
  (truck_tank_capacity_factor : ℝ)
  (h1 : service_cost = 2.20)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : num_minivans = 4)
  (h4 : num_trucks = 2)
  (h5 : minivan_tank_capacity = 65)
  (h6 : truck_tank_capacity_factor = 2.20) :
  let truck_tank_capacity := minivan_tank_capacity * truck_tank_capacity_factor
  let total_service_cost := service_cost * (num_minivans + num_trucks)
  let total_fuel_cost_minivans := num_minivans * minivan_tank_capacity * fuel_cost_per_liter
  let total_fuel_cost_trucks := num_trucks * truck_tank_capacity * fuel_cost_per_liter
  let total_cost := total_service_cost + total_fuel_cost_minivans + total_fuel_cost_trucks
  total_cost = 395.40 := by
  sorry

#eval 2.20 * (4 + 2) + 4 * 65 * 0.70 + 2 * (65 * 2.20) * 0.70

end NUMINAMATH_CALUDE_fuel_station_total_cost_l882_88294


namespace NUMINAMATH_CALUDE_alyssa_toy_spending_l882_88236

/-- The amount Alyssa spent on a football -/
def football_cost : ℚ := 571/100

/-- The amount Alyssa spent on marbles -/
def marbles_cost : ℚ := 659/100

/-- The total amount Alyssa spent on toys -/
def total_cost : ℚ := football_cost + marbles_cost

theorem alyssa_toy_spending :
  total_cost = 1230/100 := by sorry

end NUMINAMATH_CALUDE_alyssa_toy_spending_l882_88236


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l882_88283

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l882_88283


namespace NUMINAMATH_CALUDE_orange_difference_l882_88258

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := 44

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := 25

/-- The difference between the number of sacks of ripe and unripe oranges harvested per day -/
theorem orange_difference : ripe_oranges - unripe_oranges = 19 := by
  sorry

end NUMINAMATH_CALUDE_orange_difference_l882_88258


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_150_l882_88207

theorem largest_whole_number_less_than_150 :
  ∀ x : ℕ, x ≤ 24 ↔ 6 * x + 3 < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_150_l882_88207


namespace NUMINAMATH_CALUDE_fair_prize_distribution_l882_88298

/-- Represents the probability of winning a single game -/
def win_probability : ℚ := 1 / 2

/-- Represents the total prize money in yuan -/
def total_prize : ℕ := 12000

/-- Represents the score of the leading player -/
def leading_score : ℕ := 2

/-- Represents the score of the trailing player -/
def trailing_score : ℕ := 1

/-- Represents the number of games needed to win the series -/
def games_to_win : ℕ := 3

/-- Calculates the fair prize for the leading player -/
def fair_prize (p : ℚ) (total : ℕ) : ℚ := p * total

/-- Theorem stating the fair prize distribution for the leading player -/
theorem fair_prize_distribution :
  fair_prize ((win_probability + win_probability * win_probability) : ℚ) total_prize =
  (3 / 4 : ℚ) * total_prize :=
sorry

end NUMINAMATH_CALUDE_fair_prize_distribution_l882_88298


namespace NUMINAMATH_CALUDE_zoo_admission_solution_l882_88287

/-- Represents the zoo admission problem for two classes -/
structure ZooAdmission where
  ticket_price : ℕ
  class_a_students : ℕ
  class_b_students : ℕ

/-- Calculates the number of free tickets given the total number of students -/
def free_tickets (n : ℕ) : ℕ :=
  n / 5

/-- Calculates the total cost for a class given the number of students and ticket price -/
def class_cost (students : ℕ) (price : ℕ) : ℕ :=
  (students - free_tickets students) * price

/-- The main theorem representing the zoo admission problem -/
theorem zoo_admission_solution :
  ∃ (za : ZooAdmission),
    za.ticket_price > 0 ∧
    class_cost za.class_a_students za.ticket_price = 1995 ∧
    class_cost (za.class_a_students + za.class_b_students) za.ticket_price = 4410 ∧
    za.class_a_students = 23 ∧
    za.class_b_students = 29 ∧
    za.ticket_price = 105 :=
  sorry

end NUMINAMATH_CALUDE_zoo_admission_solution_l882_88287


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_to_prism_volume_l882_88296

theorem cube_surface_area_equal_to_prism_volume (a b c : ℝ) (h1 : a = 10) (h2 : b = 3) (h3 : c = 30) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 6 * 900 ^ (2/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_to_prism_volume_l882_88296


namespace NUMINAMATH_CALUDE_pencil_costs_two_l882_88291

def pencil_cost : ℝ → Prop := λ x =>
  ∃ (pen_cost : ℝ),
    pen_cost = x + 9 ∧
    x + pen_cost = 13

theorem pencil_costs_two : pencil_cost 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_costs_two_l882_88291


namespace NUMINAMATH_CALUDE_equation_roots_existence_and_bounds_l882_88206

theorem equation_roots_existence_and_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x₁ x₂ : ℝ, 
    (1 / x₁ + 1 / (x₁ - a) + 1 / (x₁ + b) = 0) ∧
    (1 / x₂ + 1 / (x₂ - a) + 1 / (x₂ + b) = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_existence_and_bounds_l882_88206


namespace NUMINAMATH_CALUDE_solve_linear_equation_l882_88244

theorem solve_linear_equation (y : ℚ) (h : -3*y - 8 = 5*y + 4) : y = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l882_88244


namespace NUMINAMATH_CALUDE_trig_equation_solution_l882_88278

theorem trig_equation_solution (x : ℝ) : 
  1 - Real.cos (6 * x) = Real.tan (3 * x) ↔ 
  (∃ k : ℤ, x = k * Real.pi / 3) ∨ 
  (∃ k : ℤ, x = Real.pi / 12 * (4 * k + 1)) := by
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l882_88278


namespace NUMINAMATH_CALUDE_interest_difference_l882_88263

/-- Calculate the loss when using simple interest instead of compound interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) : 
  principal = 2500 →
  rate = 0.04 →
  time = 2 →
  principal * (1 + rate) ^ time - principal - (principal * rate * time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l882_88263


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l882_88235

/-- Given a quadrilateral EFGH where ∠E = 2∠F = 3∠G = 6∠H, prove that ∠E = 180° -/
theorem angle_measure_in_special_quadrilateral (E F G H : ℝ) : 
  E + F + G + H = 360 → -- sum of angles in a quadrilateral
  E = 2 * F →           -- given condition
  E = 3 * G →           -- given condition
  E = 6 * H →           -- given condition
  E = 180 :=             -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l882_88235


namespace NUMINAMATH_CALUDE_perfect_seventh_power_l882_88218

theorem perfect_seventh_power (x y z : ℕ+) (h : ∃ (n : ℕ+), x^3 * y^5 * z^6 = n^7) :
  ∃ (m : ℕ+), x^5 * y^6 * z^3 = m^7 := by sorry

end NUMINAMATH_CALUDE_perfect_seventh_power_l882_88218


namespace NUMINAMATH_CALUDE_zayne_bracelet_count_l882_88230

/-- Calculates the number of bracelets Zayne started with given the sales conditions -/
def bracelets_count (single_price : ℕ) (pair_price : ℕ) (single_sales : ℕ) (total_revenue : ℕ) : ℕ :=
  let single_revenue := single_price * single_sales
  let pair_revenue := total_revenue - single_revenue
  let pair_sales := pair_revenue / pair_price
  single_sales + 2 * pair_sales

/-- Theorem stating that Zayne started with 30 bracelets given the sales conditions -/
theorem zayne_bracelet_count :
  bracelets_count 5 8 (60 / 5) 132 = 30 := by
  sorry

end NUMINAMATH_CALUDE_zayne_bracelet_count_l882_88230


namespace NUMINAMATH_CALUDE_total_reams_is_five_l882_88297

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := 3

/-- The total number of reams of paper bought -/
def total_reams : ℕ := reams_for_haley + reams_for_sister

theorem total_reams_is_five : total_reams = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_reams_is_five_l882_88297


namespace NUMINAMATH_CALUDE_school_gender_ratio_l882_88254

theorem school_gender_ratio (boys girls : ℕ) : 
  (boys : ℚ) / girls = 7.5 / 15.4 →
  girls = boys + 174 →
  boys = 165 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l882_88254


namespace NUMINAMATH_CALUDE_sin_300_degrees_l882_88266

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l882_88266


namespace NUMINAMATH_CALUDE_intersection_point_lines_parallel_line_equation_y_intercept_4_equation_l882_88215

-- Define the lines and point M
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0
def l (x y : ℝ) : Prop := 2 * x + 4 * y - 5 = 0

-- M is the intersection point of l₁ and l₂
def M : ℝ × ℝ := (-1, 2)

-- Define the equations of the lines we want to prove
def line_parallel (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def line_y_intercept_4 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem intersection_point_lines (x y : ℝ) :
  l₁ x y ∧ l₂ x y ↔ (x, y) = M :=
sorry

theorem parallel_line_equation :
  ∀ x y : ℝ, (x, y) = M → line_parallel x y ∧ ∃ k : ℝ, ∀ x y : ℝ, line_parallel x y ↔ l (x + k) (y + k) :=
sorry

theorem y_intercept_4_equation :
  ∀ x y : ℝ, (x, y) = M → line_y_intercept_4 x y ∧ line_y_intercept_4 0 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_lines_parallel_line_equation_y_intercept_4_equation_l882_88215
