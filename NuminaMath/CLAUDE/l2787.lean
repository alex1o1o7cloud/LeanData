import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_slope_l2787_278748

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x * (-2)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * x + a * (-2)

-- Theorem statement
theorem tangent_line_slope (a : ℝ) :
  f' a 1 = -2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2787_278748


namespace NUMINAMATH_CALUDE_expression_value_l2787_278788

theorem expression_value : 
  let a : ℤ := 2025
  let b : ℤ := a + 1
  let k : ℤ := 1
  (a^3 - 2*k*a^2*b + 3*k*a*b^2 - b^3 + k) / (a*b) = 2025 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2787_278788


namespace NUMINAMATH_CALUDE_notched_circle_distance_l2787_278774

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 72}

def B : ℝ × ℝ := (1, -4)
def A : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (7, -4)

-- State the theorem
theorem notched_circle_distance :
  B ∈ Circle ∧
  A ∈ Circle ∧
  C ∈ Circle ∧
  A.1 = B.1 ∧
  A.2 - B.2 = 8 ∧
  C.1 - B.1 = 6 ∧
  C.2 = B.2 ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  B.1^2 + B.2^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_notched_circle_distance_l2787_278774


namespace NUMINAMATH_CALUDE_machine_a_time_proof_l2787_278752

/-- The time it takes for Machine A to finish the job alone -/
def machine_a_time : ℝ := 4

/-- The time it takes for Machine B to finish the job alone -/
def machine_b_time : ℝ := 12

/-- The time it takes for Machine C to finish the job alone -/
def machine_c_time : ℝ := 6

/-- The time it takes for all machines to finish the job together -/
def combined_time : ℝ := 2

theorem machine_a_time_proof :
  (1 / machine_a_time + 1 / machine_b_time + 1 / machine_c_time) * combined_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_machine_a_time_proof_l2787_278752


namespace NUMINAMATH_CALUDE_number_problem_l2787_278728

theorem number_problem (x : ℝ) : 0.7 * x - 40 = 30 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2787_278728


namespace NUMINAMATH_CALUDE_playground_count_l2787_278787

theorem playground_count (x : ℤ) : 
  let known_numbers : List ℤ := [12, 1, 12, 7, 3, 8]
  let all_numbers : List ℤ := x :: known_numbers
  (all_numbers.sum / all_numbers.length : ℚ) = 7 → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_playground_count_l2787_278787


namespace NUMINAMATH_CALUDE_x950x_divisible_by_36_l2787_278730

def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def form_x950x (x : ℕ) : ℕ :=
  x * 10000 + 9500 + x

theorem x950x_divisible_by_36 :
  ∃ (x : ℕ), 
    x < 10 ∧ 
    is_five_digit_number (form_x950x x) ∧ 
    (form_x950x x) % 36 = 0 ↔ 
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x950x_divisible_by_36_l2787_278730


namespace NUMINAMATH_CALUDE_terms_before_negative_three_l2787_278772

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem terms_before_negative_three (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 105 ∧ d = -6 →
  (∀ k < n, arithmetic_sequence a₁ d k > -3) ∧
  arithmetic_sequence a₁ d n = -3 →
  n - 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_negative_three_l2787_278772


namespace NUMINAMATH_CALUDE_student_composition_l2787_278722

/-- The number of ways to select participants from a group of students -/
def selectionWays (males females : ℕ) : ℕ :=
  males * (males - 1) * females

theorem student_composition :
  ∃ (males females : ℕ),
    males + females = 8 ∧
    selectionWays males females = 90 →
    males = 3 ∧ females = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_composition_l2787_278722


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2787_278750

/-- The trajectory of the midpoint of a line segment connecting a point on a parabola and a fixed point -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (P : ℝ × ℝ), 
    P.2 = 2 * P.1^2 + 1 ∧ 
    P.1 = 2 * x ∧ 
    P.2 = 2 * y + 1) →
  y = 4 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2787_278750


namespace NUMINAMATH_CALUDE_tree_height_problem_l2787_278711

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 24 →  -- One tree is 24 feet taller than the other
  h₂ / h₁ = 2 / 3 →  -- The heights are in the ratio 2:3
  h₁ = 72 :=  -- The height of the taller tree is 72 feet
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l2787_278711


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l2787_278703

theorem multiple_with_binary_digits (n : ℕ+) : ∃ m : ℕ,
  (n : ℕ) ∣ m ∧
  (Nat.digits 2 m).length ≤ n ∧
  ∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l2787_278703


namespace NUMINAMATH_CALUDE_max_x_value_l2787_278751

theorem max_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + (5 * x - 20) / (4 * x - 5) = 18 → x ≤ 50 / 29 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l2787_278751


namespace NUMINAMATH_CALUDE_inequality_proof_l2787_278773

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 + c^2 = 14) : 
  a^5 + (1/8)*b^5 + (1/27)*c^5 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2787_278773


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l2787_278725

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hand of the watch -/
structure WatchHand where
  axis : Point
  angularVelocity : ℝ
  initialAngle : ℝ

/-- Represents the watch configuration -/
structure Watch where
  secondHand : WatchHand
  stopwatchHand : WatchHand

/-- The locus of intersection points between extended watch hands -/
def intersectionLocus (w : Watch) (t : ℝ) : Point :=
  sorry

/-- Theorem stating that the intersection locus forms a circle -/
theorem intersection_locus_is_circle (w : Watch) : 
  ∃ (center : Point) (radius : ℝ), 
    ∀ t, ∃ θ, intersectionLocus w t = Point.mk (center.x + radius * Real.cos θ) (center.y + radius * Real.sin θ) :=
  sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l2787_278725


namespace NUMINAMATH_CALUDE_two_true_propositions_l2787_278795

theorem two_true_propositions :
  let P : ℝ → Prop := λ x => x > -3
  let Q : ℝ → Prop := λ x => x > -6
  let original := ∀ x, P x → Q x
  let converse := ∀ x, Q x → P x
  let inverse := ∀ x, ¬(P x) → ¬(Q x)
  let contrapositive := ∀ x, ¬(Q x) → ¬(P x)
  (original ∧ contrapositive ∧ ¬converse ∧ ¬inverse) ∨
  (original ∧ contrapositive ∧ converse ∧ ¬inverse) ∨
  (original ∧ contrapositive ∧ ¬converse ∧ inverse) :=
by
  sorry


end NUMINAMATH_CALUDE_two_true_propositions_l2787_278795


namespace NUMINAMATH_CALUDE_rectangle_color_theorem_l2787_278753

/-- A cell in the rectangle can be either white or black -/
inductive CellColor
  | White
  | Black

/-- The rectangle is represented as a 3 × 7 matrix of cell colors -/
def Rectangle := Matrix (Fin 3) (Fin 7) CellColor

/-- A point in the rectangle, represented by its row and column -/
structure Point where
  row : Fin 3
  col : Fin 7

/-- Check if four points form a rectangle parallel to the sides of the original rectangle -/
def isParallelRectangle (p1 p2 p3 p4 : Point) : Prop :=
  (p1.row = p2.row ∧ p3.row = p4.row ∧ p1.col = p3.col ∧ p2.col = p4.col) ∨
  (p1.row = p3.row ∧ p2.row = p4.row ∧ p1.col = p2.col ∧ p3.col = p4.col)

/-- Check if all four points have the same color in the given rectangle -/
def sameColor (rect : Rectangle) (p1 p2 p3 p4 : Point) : Prop :=
  rect p1.row p1.col = rect p2.row p2.col ∧
  rect p2.row p2.col = rect p3.row p3.col ∧
  rect p3.row p3.col = rect p4.row p4.col

theorem rectangle_color_theorem (rect : Rectangle) :
  ∃ p1 p2 p3 p4 : Point,
    isParallelRectangle p1 p2 p3 p4 ∧
    sameColor rect p1 p2 p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_color_theorem_l2787_278753


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l2787_278764

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l2787_278764


namespace NUMINAMATH_CALUDE_probability_different_colors_is_two_thirds_l2787_278789

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def balls_drawn : ℕ := 2

def total_ways : ℕ := Nat.choose total_balls balls_drawn
def different_color_ways : ℕ := red_balls * white_balls

def probability_different_colors : ℚ := different_color_ways / total_ways

theorem probability_different_colors_is_two_thirds :
  probability_different_colors = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_two_thirds_l2787_278789


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2787_278742

theorem coin_flip_probability (n m k l : ℕ) (h1 : n = 11) (h2 : m = 5) (h3 : k = 7) (h4 : l = 3) :
  let p := (1 : ℚ) / 2
  let total_success_prob := (n.choose k : ℚ) * p^k * (1 - p)^(n - k)
  let monday_success_prob := (m.choose l : ℚ) * p^l * (1 - p)^(m - l)
  let tuesday_success_prob := ((n - m).choose (k - l) : ℚ) * p^(k - l) * (1 - p)^(n - m - (k - l))
  (monday_success_prob * tuesday_success_prob) / total_success_prob = 5 / 11 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2787_278742


namespace NUMINAMATH_CALUDE_percentage_problem_l2787_278790

theorem percentage_problem : 
  ∃ (P : ℝ), (0.1 * 30 + P * 50 = 10.5) ∧ (P = 0.15) := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2787_278790


namespace NUMINAMATH_CALUDE_canoe_rowing_probability_l2787_278704

/-- The probability of rowing a canoe given certain conditions on oar functionality and weather -/
theorem canoe_rowing_probability :
  let p_left_works : ℚ := 3/5  -- Probability left oar works
  let p_right_works : ℚ := 3/5  -- Probability right oar works
  let p_weather : ℚ := 1/4  -- Probability of adverse weather
  let p_oar_works_in_weather (p : ℚ) : ℚ := 1 - 2 * (1 - p)  -- Probability oar works in adverse weather
  
  let p_both_work_no_weather : ℚ := p_left_works * p_right_works
  let p_both_work_weather : ℚ := p_oar_works_in_weather p_left_works * p_oar_works_in_weather p_right_works
  
  let p_row : ℚ := p_both_work_no_weather * (1 - p_weather) + p_both_work_weather * p_weather

  p_row = 7/25 := by sorry

end NUMINAMATH_CALUDE_canoe_rowing_probability_l2787_278704


namespace NUMINAMATH_CALUDE_equivalent_equations_product_l2787_278755

/-- Given that the equation a^8xy - 2a^7y - 3a^6x = 2a^5(b^5 - 2) is equivalent to 
    (a^m*x - 2a^n)(a^p*y - 3a^3) = 2a^5*b^5 for some integers m, n, and p, 
    prove that m*n*p = 60 -/
theorem equivalent_equations_product (a b x y : ℝ) (m n p : ℤ) 
  (h1 : a^8*x*y - 2*a^7*y - 3*a^6*x = 2*a^5*(b^5 - 2))
  (h2 : (a^m*x - 2*a^n)*(a^p*y - 3*a^3) = 2*a^5*b^5) :
  m * n * p = 60 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_equations_product_l2787_278755


namespace NUMINAMATH_CALUDE_all_integers_are_cute_l2787_278708

/-- An integer is cute if it can be written as a^2 + b^3 + c^3 + d^5 for some integers a, b, c, and d. -/
def IsCute (n : ℤ) : Prop :=
  ∃ a b c d : ℤ, n = a^2 + b^3 + c^3 + d^5

/-- All integers are cute. -/
theorem all_integers_are_cute : ∀ n : ℤ, IsCute n := by
  sorry


end NUMINAMATH_CALUDE_all_integers_are_cute_l2787_278708


namespace NUMINAMATH_CALUDE_root_range_l2787_278791

theorem root_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ ∈ Set.Icc (k - 1) (k + 1) ∧ 
    x₂ ∈ Set.Icc (k - 1) (k + 1) ∧
    Real.sqrt 2 * |x₁ - k| = k * Real.sqrt x₁ ∧
    Real.sqrt 2 * |x₂ - k| = k * Real.sqrt x₂) 
  ↔ 
  (0 < k ∧ k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_root_range_l2787_278791


namespace NUMINAMATH_CALUDE_bons_winning_probability_l2787_278785

/-- The probability of rolling a six on a six-sided die. -/
def probSix : ℚ := 1/6

/-- The probability of not rolling a six on a six-sided die. -/
def probNotSix : ℚ := 1 - probSix

/-- The probability that B. Bons wins the game. -/
noncomputable def probBonsWins : ℚ :=
  (probNotSix * probSix) / (1 - probNotSix * probNotSix)

theorem bons_winning_probability :
  probBonsWins = 5/11 := by sorry

end NUMINAMATH_CALUDE_bons_winning_probability_l2787_278785


namespace NUMINAMATH_CALUDE_sin_sum_upper_bound_l2787_278729

theorem sin_sum_upper_bound (x y z : ℝ) (hx : x ∈ Set.Icc 0 Real.pi) 
  (hy : y ∈ Set.Icc 0 Real.pi) (hz : z ∈ Set.Icc 0 Real.pi) : 
  Real.sin (x - y) + Real.sin (y - z) + Real.sin (z - x) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_upper_bound_l2787_278729


namespace NUMINAMATH_CALUDE_laundry_drying_time_l2787_278731

theorem laundry_drying_time 
  (num_loads : ℕ) 
  (wash_time_per_load : ℕ) 
  (total_laundry_time : ℕ) 
  (h1 : num_loads = 2) 
  (h2 : wash_time_per_load = 45) 
  (h3 : total_laundry_time = 165) : 
  total_laundry_time - (num_loads * wash_time_per_load) = 75 := by
sorry

end NUMINAMATH_CALUDE_laundry_drying_time_l2787_278731


namespace NUMINAMATH_CALUDE_log_base_three_squared_l2787_278757

theorem log_base_three_squared (m : ℝ) (b : ℝ) (h : 3^m = b) :
  Real.log b / Real.log (3^2) = m / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_base_three_squared_l2787_278757


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2787_278744

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * a^3 + 9 * b^3 + 32 * c^3 + 1 / (4 * a * b * c) ≥ 6 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * a^3 + 9 * b^3 + 32 * c^3 + 1 / (4 * a * b * c) = 6 ↔
  a = (1 : ℝ) / (6 : ℝ)^(1/3) ∧ b = (1 : ℝ) / (9 : ℝ)^(1/3) ∧ c = (1 : ℝ) / (32 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2787_278744


namespace NUMINAMATH_CALUDE_function_always_positive_l2787_278769

theorem function_always_positive (k : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (k - 2) * x + 2 * |k| - 1 > 0) ↔ k > 5/4 := by
  sorry

end NUMINAMATH_CALUDE_function_always_positive_l2787_278769


namespace NUMINAMATH_CALUDE_expression_simplification_l2787_278766

theorem expression_simplification (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  3 * m^2 * n + 2 * (2 * m * n^2 - 3 * m^2 * n) - 3 * (m * n^2 - m^2 * n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2787_278766


namespace NUMINAMATH_CALUDE_cars_ratio_after_days_l2787_278702

/-- Represents the number of days after which Station A will have 7 times as many cars as Station B -/
def days_to_reach_ratio : ℕ :=
  let initial_a : ℕ := 192
  let initial_b : ℕ := 48
  let daily_a_to_b : ℕ := 21
  let daily_b_to_a : ℕ := 24
  6

/-- Theorem stating that after the calculated number of days, 
    Station A will have 7 times as many cars as Station B -/
theorem cars_ratio_after_days :
  let initial_a : ℕ := 192
  let initial_b : ℕ := 48
  let daily_a_to_b : ℕ := 21
  let daily_b_to_a : ℕ := 24
  let days := days_to_reach_ratio
  let final_a := initial_a + days * (daily_b_to_a - daily_a_to_b)
  let final_b := initial_b + days * (daily_a_to_b - daily_b_to_a)
  final_a = 7 * final_b :=
by
  sorry

end NUMINAMATH_CALUDE_cars_ratio_after_days_l2787_278702


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l2787_278758

theorem bernoulli_inequality (x : ℝ) (m : ℕ+) (h : x > -1) :
  (1 + x)^(m : ℕ) ≥ 1 + m * x :=
by sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l2787_278758


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l2787_278743

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (2/3) * (a n) + 1/3) →
  (∀ n : ℕ, a n = (-2)^(n-1)) :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l2787_278743


namespace NUMINAMATH_CALUDE_garden_flowers_count_l2787_278723

/-- Represents a rectangular garden with a rose planted in it. -/
structure Garden where
  columns : ℕ
  rows : ℕ
  rose_col_left : ℕ
  rose_col_right : ℕ
  rose_row_front : ℕ
  rose_row_back : ℕ

/-- The total number of flowers in the garden. -/
def total_flowers (g : Garden) : ℕ := g.columns * g.rows

/-- Theorem stating the total number of flowers in the specific garden configuration. -/
theorem garden_flowers_count :
  ∀ g : Garden,
  g.rose_col_left = 9 →
  g.rose_col_right = 13 →
  g.rose_row_front = 7 →
  g.rose_row_back = 16 →
  g.columns = g.rose_col_left + g.rose_col_right - 1 →
  g.rows = g.rose_row_front + g.rose_row_back - 1 →
  total_flowers g = 462 := by
  sorry

#check garden_flowers_count

end NUMINAMATH_CALUDE_garden_flowers_count_l2787_278723


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2787_278784

-- Define the sum of angles around a point
def sum_of_angles : ℝ := 360

-- Define the four angles as functions of x
def angle1 (x : ℝ) : ℝ := 5 * x
def angle2 (x : ℝ) : ℝ := 4 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem angle_sum_theorem :
  ∃ x : ℝ, angle1 x + angle2 x + angle3 x + angle4 x = sum_of_angles ∧ x = 30 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2787_278784


namespace NUMINAMATH_CALUDE_root_range_implies_a_range_l2787_278720

theorem root_range_implies_a_range (a : ℝ) :
  (∃ α β : ℝ, 5 * α^2 - 7 * α - a = 0 ∧
              5 * β^2 - 7 * β - a = 0 ∧
              -1 < α ∧ α < 0 ∧
              1 < β ∧ β < 2) →
  (0 < a ∧ a < 6) := by
sorry

end NUMINAMATH_CALUDE_root_range_implies_a_range_l2787_278720


namespace NUMINAMATH_CALUDE_circle_radius_l2787_278746

theorem circle_radius (P Q : ℝ) (h : P / Q = 40 / Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ P = Real.pi * r^2 ∧ Q = 2 * Real.pi * r ∧ r = 80 / Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l2787_278746


namespace NUMINAMATH_CALUDE_solve_equation_l2787_278701

theorem solve_equation : ∃ x : ℝ, 10 * x - (2 * 1.5 / 0.3) = 50 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2787_278701


namespace NUMINAMATH_CALUDE_odd_implies_abs_symmetric_abs_symmetric_not_sufficient_for_odd_l2787_278737

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The graph of |f(x)| is symmetric about the y-axis if |f(-x)| = |f(x)| for all x ∈ ℝ -/
def IsAbsSymmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem odd_implies_abs_symmetric (f : ℝ → ℝ) :
  IsOdd f → IsAbsSymmetric f :=
sorry

theorem abs_symmetric_not_sufficient_for_odd :
  ∃ f : ℝ → ℝ, IsAbsSymmetric f ∧ ¬IsOdd f :=
sorry

end NUMINAMATH_CALUDE_odd_implies_abs_symmetric_abs_symmetric_not_sufficient_for_odd_l2787_278737


namespace NUMINAMATH_CALUDE_linear_function_m_value_l2787_278749

/-- Given a linear function y = (m^2 + 2m)x + m^2 + m - 1 + (2m - 3), prove that m = 1 -/
theorem linear_function_m_value (m : ℝ) : 
  (∃ k b, ∀ x, (m^2 + 2*m)*x + (m^2 + m - 1 + (2*m - 3)) = k*x + b) → 
  (m^2 + 2*m ≠ 0) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l2787_278749


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l2787_278716

theorem simplify_and_ratio (k : ℝ) : ∃ (a b : ℝ), 
  (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a / b = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l2787_278716


namespace NUMINAMATH_CALUDE_not_necessarily_congruent_with_two_sides_one_angle_l2787_278724

/-- Triangle represented by three points in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Predicate for triangle congruence -/
def IsCongruent (t1 t2 : Triangle) : Prop :=
  sorry

/-- Predicate for two sides and one angle being equal -/
def HasTwoSidesOneAngleEqual (t1 t2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that triangles with two corresponding sides and one corresponding angle equal
    are not necessarily congruent -/
theorem not_necessarily_congruent_with_two_sides_one_angle :
  ∃ t1 t2 : Triangle, HasTwoSidesOneAngleEqual t1 t2 ∧ ¬IsCongruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_congruent_with_two_sides_one_angle_l2787_278724


namespace NUMINAMATH_CALUDE_zero_success_probability_l2787_278756

/-- The probability of 0 successes in 7 Bernoulli trials with success probability 2/7 -/
def prob_zero_success (n : ℕ) (p : ℚ) : ℚ :=
  (1 - p) ^ n

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The probability of success in a single trial -/
def success_prob : ℚ := 2/7

theorem zero_success_probability :
  prob_zero_success num_trials success_prob = (5/7) ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_zero_success_probability_l2787_278756


namespace NUMINAMATH_CALUDE_emilys_coin_collection_value_l2787_278777

/-- Proves that given the conditions of Emily's coin collection, the total value is $128 -/
theorem emilys_coin_collection_value :
  ∀ (total_coins : ℕ) 
    (first_type_count : ℕ) 
    (first_type_total_value : ℝ) 
    (second_type_count : ℕ),
  total_coins = 20 →
  first_type_count = 8 →
  first_type_total_value = 32 →
  second_type_count = total_coins - first_type_count →
  (second_type_count * (first_type_total_value / first_type_count) * 2 + first_type_total_value = 128) :=
by
  sorry


end NUMINAMATH_CALUDE_emilys_coin_collection_value_l2787_278777


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2787_278707

theorem system_of_equations_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + x*y + y^2 = 108)
  (h_eq2 : y^2 + y*z + z^2 = 49)
  (h_eq3 : z^2 + x*z + x^2 = 157) :
  x*y + y*z + x*z = 104 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2787_278707


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2787_278727

theorem remainder_sum_mod_seven : 
  (2 * (4561 + 4562 + 4563 + 4564 + 4565)) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2787_278727


namespace NUMINAMATH_CALUDE_train_speed_l2787_278798

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 300 → 
  time = 18 → 
  speed = (length / time) * 3.6 → 
  speed = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_l2787_278798


namespace NUMINAMATH_CALUDE_infinite_monotone_subsequence_l2787_278747

-- Define an infinite sequence of distinct real numbers
def InfiniteSequence := ℕ → ℝ

-- Define the property that all elements in the sequence are distinct
def AllDistinct (seq : InfiniteSequence) : Prop :=
  ∀ i j : ℕ, i ≠ j → seq i ≠ seq j

-- Define a strictly increasing subsequence
def StrictlyIncreasing (subseq : ℕ → ℕ) (seq : InfiniteSequence) : Prop :=
  ∀ i j : ℕ, i < j → seq (subseq i) < seq (subseq j)

-- Define a strictly decreasing subsequence
def StrictlyDecreasing (subseq : ℕ → ℕ) (seq : InfiniteSequence) : Prop :=
  ∀ i j : ℕ, i < j → seq (subseq i) > seq (subseq j)

-- The main theorem
theorem infinite_monotone_subsequence
  (seq : InfiniteSequence) (h : AllDistinct seq) :
  (∃ subseq : ℕ → ℕ, StrictlyIncreasing subseq seq) ∨
  (∃ subseq : ℕ → ℕ, StrictlyDecreasing subseq seq) :=
sorry

end NUMINAMATH_CALUDE_infinite_monotone_subsequence_l2787_278747


namespace NUMINAMATH_CALUDE_fifteen_point_five_minutes_in_hours_l2787_278721

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ :=
  minutes * (1 / 60)

theorem fifteen_point_five_minutes_in_hours : 
  minutes_to_hours 15.5 = 930 / 3600 := by
sorry

end NUMINAMATH_CALUDE_fifteen_point_five_minutes_in_hours_l2787_278721


namespace NUMINAMATH_CALUDE_max_value_of_derived_function_l2787_278771

/-- Given a function f(x) = a * sin(x) + b with max value 1 and min value -7,
    prove that the max value of b * sin²(x) - a * cos²(x) is either 4 or -3 -/
theorem max_value_of_derived_function 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * Real.sin x + b)
  (h_max : ∀ x, f x ≤ 1)
  (h_min : ∀ x, f x ≥ -7)
  : (∃ x, b * Real.sin x ^ 2 - a * Real.cos x ^ 2 = 4) ∨ 
    (∃ x, b * Real.sin x ^ 2 - a * Real.cos x ^ 2 = -3) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_derived_function_l2787_278771


namespace NUMINAMATH_CALUDE_initial_distance_is_50_l2787_278715

/-- The initial distance between two people walking towards each other -/
def initial_distance (speed : ℝ) (distance_walked : ℝ) : ℝ :=
  2 * distance_walked

/-- Theorem: The initial distance between Fred and Sam is 50 miles -/
theorem initial_distance_is_50 (fred_speed sam_speed : ℝ) (sam_distance : ℝ) :
  fred_speed = 5 →
  sam_speed = 5 →
  sam_distance = 25 →
  initial_distance sam_speed sam_distance = 50 :=
by
  sorry

#check initial_distance_is_50

end NUMINAMATH_CALUDE_initial_distance_is_50_l2787_278715


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_unique_l2787_278786

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_unique
  (q : ℝ → ℝ)
  (h_monic : ∃ a b c : ℝ, q = MonicCubicPolynomial a b c)
  (h_root : q (2 - 3*I) = 0)
  (h_value : q 1 = 26) :
  q = MonicCubicPolynomial (-2.4) 6.6 20.8 := by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_unique_l2787_278786


namespace NUMINAMATH_CALUDE_third_week_cases_new_york_coronavirus_cases_l2787_278733

/-- Proves the number of new coronavirus cases in the third week --/
theorem third_week_cases (first_week : ℕ) (total_cases : ℕ) : ℕ :=
  let second_week := first_week / 2
  let first_two_weeks := first_week + second_week
  total_cases - first_two_weeks

/-- The main theorem that proves the number of new cases in the third week is 2000 --/
theorem new_york_coronavirus_cases : third_week_cases 5000 9500 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_third_week_cases_new_york_coronavirus_cases_l2787_278733


namespace NUMINAMATH_CALUDE_circle_area_three_fourths_l2787_278799

/-- Given a circle where three times the reciprocal of its circumference 
    equals its diameter, prove that its area is 3/4 -/
theorem circle_area_three_fourths (r : ℝ) (h : 3 * (1 / (2 * π * r)) = 2 * r) : 
  π * r^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_three_fourths_l2787_278799


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2787_278762

/-- Given two lines m and n that intersect at (1, 6), where
    m has equation y = 4x + 2 and n has equation y = kx + 3,
    prove that k = 3. -/
theorem intersection_of_lines (k : ℝ) : 
  (∀ x y : ℝ, y = 4*x + 2 → (x = 1 ∧ y = 6)) →  -- line m passes through (1, 6)
  (∀ x y : ℝ, y = k*x + 3 → (x = 1 ∧ y = 6)) →  -- line n passes through (1, 6)
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2787_278762


namespace NUMINAMATH_CALUDE_imaginary_part_of_x_l2787_278740

theorem imaginary_part_of_x (x : ℂ) (h : (3 + 4*I)*x = Complex.abs (4 + 3*I)) : 
  x.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_x_l2787_278740


namespace NUMINAMATH_CALUDE_polynomial_value_constraint_l2787_278778

theorem polynomial_value_constraint (a b c : ℤ) : 
  (b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) → 
  (b + c + a ≠ 2009) := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_constraint_l2787_278778


namespace NUMINAMATH_CALUDE_calen_excess_pencils_l2787_278768

/-- The number of pencils each person has -/
structure PencilCount where
  calen : ℕ
  caleb : ℕ
  candy : ℕ

/-- The conditions of the problem -/
def pencil_problem (p : PencilCount) : Prop :=
  p.calen > p.caleb ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.candy = 9 ∧
  p.calen - 10 = 10

/-- The theorem to prove -/
theorem calen_excess_pencils (p : PencilCount) :
  pencil_problem p → p.calen - p.caleb = 5 := by
  sorry

end NUMINAMATH_CALUDE_calen_excess_pencils_l2787_278768


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2787_278796

theorem infinite_series_sum : 
  (∑' n : ℕ, n / (5 ^ n : ℝ)) = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2787_278796


namespace NUMINAMATH_CALUDE_domain_of_sqrt_plus_fraction_l2787_278726

theorem domain_of_sqrt_plus_fraction (x : ℝ) :
  (x + 3 ≥ 0 ∧ x + 2 ≠ 0) ↔ (x ≥ -3 ∧ x ≠ -2) := by sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_plus_fraction_l2787_278726


namespace NUMINAMATH_CALUDE_train_speed_l2787_278770

/-- The speed of a train passing a platform -/
theorem train_speed (train_length platform_length : ℝ) (time : ℝ) 
  (h1 : train_length = 50)
  (h2 : platform_length = 100)
  (h3 : time = 10) :
  (train_length + platform_length) / time = 15 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2787_278770


namespace NUMINAMATH_CALUDE_internet_price_difference_l2787_278719

/-- Represents the internet service with speed and price -/
structure InternetService where
  speed : ℕ  -- in Mbps
  price : ℕ  -- in dollars

/-- The problem setup -/
def internetProblem : Prop :=
  ∃ (current twentyMbps thirtyMbps : InternetService),
    -- Current service
    current.speed = 10 ∧ current.price = 20 ∧
    -- 30 Mbps service
    thirtyMbps.speed = 30 ∧ thirtyMbps.price = 2 * current.price ∧
    -- 20 Mbps service
    twentyMbps.speed = 20 ∧ twentyMbps.price > current.price ∧
    -- Yearly savings
    (thirtyMbps.price - twentyMbps.price) * 12 = 120 ∧
    -- The statement to prove
    twentyMbps.price = current.price + 10

theorem internet_price_difference :
  internetProblem :=
sorry

end NUMINAMATH_CALUDE_internet_price_difference_l2787_278719


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l2787_278700

theorem sqrt_two_squared : (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l2787_278700


namespace NUMINAMATH_CALUDE_bellas_dancer_friends_l2787_278776

theorem bellas_dancer_friends (total_roses : ℕ) (parent_roses : ℕ) (roses_per_friend : ℕ) 
  (h1 : total_roses = 44)
  (h2 : parent_roses = 2 * 12)
  (h3 : roses_per_friend = 2) :
  (total_roses - parent_roses) / roses_per_friend = 10 := by
  sorry

end NUMINAMATH_CALUDE_bellas_dancer_friends_l2787_278776


namespace NUMINAMATH_CALUDE_find_M_l2787_278793

theorem find_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end NUMINAMATH_CALUDE_find_M_l2787_278793


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2787_278718

theorem trigonometric_identity : 
  (Real.sin (20 * π / 180) / Real.cos (20 * π / 180)) + 
  (Real.sin (40 * π / 180) / Real.cos (40 * π / 180)) + 
  Real.tan (60 * π / 180) * Real.tan (20 * π / 180) * Real.tan (40 * π / 180) = 
  Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2787_278718


namespace NUMINAMATH_CALUDE_sum_sequence_existence_l2787_278714

theorem sum_sequence_existence (n : ℕ) (h : n ≤ 2^1000000) :
  ∃ (k : ℕ) (x : ℕ → ℕ),
    x 0 = 1 ∧
    k ≤ 1100000 ∧
    x k = n ∧
    ∀ i ∈ Finset.range (k + 1), i ≠ 0 →
      ∃ r s, r ≤ s ∧ s < i ∧ x i = x r + x s :=
by sorry

end NUMINAMATH_CALUDE_sum_sequence_existence_l2787_278714


namespace NUMINAMATH_CALUDE_complement_of_A_l2787_278734

theorem complement_of_A (U A : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} → 
  A = {3, 4, 5} → 
  Aᶜ = {1, 2, 6} := by
sorry

end NUMINAMATH_CALUDE_complement_of_A_l2787_278734


namespace NUMINAMATH_CALUDE_choose_two_from_three_l2787_278709

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n.choose k = 3 :=
  by
  -- Assume n = 3 and k = 2
  have h1 : n = 3 := by sorry
  have h2 : k = 2 := by sorry
  
  -- Define the number of interest groups
  let num_groups : ℕ := 3
  
  -- Define the number of groups to choose
  let groups_to_choose : ℕ := 2
  
  -- Assert that n and k match our problem
  have h3 : n = num_groups := by rw [h1]
  have h4 : k = groups_to_choose := by rw [h2]
  
  -- Prove that choosing 2 from 3 equals 3
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l2787_278709


namespace NUMINAMATH_CALUDE_second_cat_weight_l2787_278759

theorem second_cat_weight (total_weight first_weight third_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : first_weight = 2)
  (h3 : third_weight = 4) :
  total_weight - first_weight - third_weight = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_cat_weight_l2787_278759


namespace NUMINAMATH_CALUDE_ellipse_focal_length_specific_ellipse_focal_length_l2787_278782

/-- The focal length of an ellipse with equation x²/a² + y²/b² = 1 is 2c, where c² = a² - b² -/
theorem ellipse_focal_length (a b : ℝ) (h : 0 < b ∧ b < a) :
  let c := Real.sqrt (a^2 - b^2)
  let focal_length := 2 * c
  focal_length = 2 → a^2 = 2 ∧ b^2 = 1 := by sorry

/-- The focal length of the ellipse x²/2 + y² = 1 is 2 -/
theorem specific_ellipse_focal_length :
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 - b^2)
  let focal_length := 2 * c
  focal_length = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_specific_ellipse_focal_length_l2787_278782


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l2787_278739

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def prob_sun : ℝ := 0.3

/-- The probability of 3 inches of rain --/
def prob_rain_3 : ℝ := 0.4

/-- The probability of 7 inches of rain --/
def prob_rain_7 : ℝ := 0.3

/-- The amount of rain in inches for the sunny scenario --/
def rain_sun : ℝ := 0

/-- The amount of rain in inches for the 3-inch rain scenario --/
def rain_3 : ℝ := 3

/-- The amount of rain in inches for the 7-inch rain scenario --/
def rain_7 : ℝ := 7

/-- The expected value of rainfall for a single day --/
def expected_daily_rainfall : ℝ :=
  prob_sun * rain_sun + prob_rain_3 * rain_3 + prob_rain_7 * rain_7

theorem expected_weekly_rainfall :
  days * expected_daily_rainfall = 23.1 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l2787_278739


namespace NUMINAMATH_CALUDE_benjie_margo_age_difference_l2787_278767

/-- The age difference between Benjie and Margo -/
def ageDifference (benjieAge : ℕ) (margoFutureAge : ℕ) (yearsTillMargoFutureAge : ℕ) : ℕ :=
  benjieAge - (margoFutureAge - yearsTillMargoFutureAge)

/-- Theorem stating the age difference between Benjie and Margo -/
theorem benjie_margo_age_difference :
  ageDifference 6 4 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_benjie_margo_age_difference_l2787_278767


namespace NUMINAMATH_CALUDE_tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k_l2787_278779

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating that the equation [tan x] = 2 cos^2 x is satisfied if and only if x = π/4 + 2kπ, where k is an integer -/
theorem tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k (x : ℝ) :
  floor (Real.tan x) = (2 : ℝ) * (Real.cos x)^2 ↔ ∃ k : ℤ, x = π/4 + 2*k*π :=
sorry

end NUMINAMATH_CALUDE_tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k_l2787_278779


namespace NUMINAMATH_CALUDE_jose_investment_is_4500_l2787_278797

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given shop investment scenario --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that Jose's investment is 4500 given the problem conditions --/
theorem jose_investment_is_4500 (s : ShopInvestment) 
  (h1 : s.tom_investment = 3000)
  (h2 : s.jose_join_delay = 2)
  (h3 : s.total_profit = 5400)
  (h4 : s.jose_profit = 3000) :
  calculate_jose_investment s = 4500 := by
  sorry

#check jose_investment_is_4500

end NUMINAMATH_CALUDE_jose_investment_is_4500_l2787_278797


namespace NUMINAMATH_CALUDE_halloween_jelly_beans_l2787_278765

/-- Given the conditions of a Halloween jelly bean distribution, 
    prove that the total number of children at the celebration is 40. -/
theorem halloween_jelly_beans 
  (initial_jelly_beans : ℕ)
  (remaining_jelly_beans : ℕ)
  (allowed_percentage : ℚ)
  (jelly_beans_per_child : ℕ)
  (h1 : initial_jelly_beans = 100)
  (h2 : remaining_jelly_beans = 36)
  (h3 : allowed_percentage = 4/5)
  (h4 : jelly_beans_per_child = 2) :
  (initial_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child / allowed_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_halloween_jelly_beans_l2787_278765


namespace NUMINAMATH_CALUDE_max_t_value_max_t_is_negative_one_l2787_278706

open Real

noncomputable def f (x : ℝ) : ℝ := log x / (x + 1)

theorem max_t_value (t : ℝ) :
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t / x > log x / (x - 1)) →
  t ≤ -1 :=
by sorry

theorem max_t_is_negative_one :
  ∃ t : ℝ, t = -1 ∧
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t / x > log x / (x - 1)) ∧
  (∀ t' : ℝ, (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t' / x > log x / (x - 1)) → t' ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_max_t_is_negative_one_l2787_278706


namespace NUMINAMATH_CALUDE_remainder_theorem_l2787_278761

theorem remainder_theorem (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2787_278761


namespace NUMINAMATH_CALUDE_laran_weekly_profit_l2787_278741

/-- Calculates the profit for Laran's poster business over a 5-day school week --/
def calculate_profit (
  total_posters_per_day : ℕ)
  (large_posters_per_day : ℕ)
  (large_poster_price : ℚ)
  (large_poster_tax_rate : ℚ)
  (large_poster_cost : ℚ)
  (small_poster_price : ℚ)
  (small_poster_tax_rate : ℚ)
  (small_poster_cost : ℚ)
  (fixed_weekly_expense : ℚ)
  (days_per_week : ℕ) : ℚ :=
  sorry

/-- Theorem stating that Laran's weekly profit is $98.50 --/
theorem laran_weekly_profit :
  calculate_profit 5 2 10 (1/10) 5 6 (3/20) 3 20 5 = 197/2 :=
  sorry

end NUMINAMATH_CALUDE_laran_weekly_profit_l2787_278741


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l2787_278792

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 5 ∧ x ≠ -3 → 
    1 / (x^3 - 7*x^2 + 11*x + 45) = A / (x - 5) + B / (x + 3) + C / (x + 3)^2) →
  B = -1 / 64 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l2787_278792


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2787_278794

theorem least_integer_greater_than_sqrt_500 : 
  (∀ n : ℕ, n ≤ 22 → n ^ 2 ≤ 500) ∧ 23 ^ 2 > 500 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2787_278794


namespace NUMINAMATH_CALUDE_box_ratio_l2787_278712

/-- Represents the number of cardboards of each type -/
structure Cardboards where
  square : ℕ
  rectangular : ℕ

/-- Represents the number of boxes of each type -/
structure Boxes where
  vertical : ℕ
  horizontal : ℕ

/-- Represents the number of cardboards used for each type of box -/
structure BoxRequirements where
  vertical_square : ℕ
  vertical_rectangular : ℕ
  horizontal_square : ℕ
  horizontal_rectangular : ℕ

/-- The main theorem stating the ratio of vertical to horizontal boxes -/
theorem box_ratio 
  (c : Cardboards) 
  (b : Boxes) 
  (r : BoxRequirements) 
  (h1 : c.rectangular = 2 * c.square)  -- Ratio of cardboards is 1:2
  (h2 : r.vertical_square * b.vertical + r.horizontal_square * b.horizontal = c.square)  -- All square cardboards are used
  (h3 : r.vertical_rectangular * b.vertical + r.horizontal_rectangular * b.horizontal = c.rectangular)  -- All rectangular cardboards are used
  : b.vertical = b.horizontal / 2 := by
  sorry

end NUMINAMATH_CALUDE_box_ratio_l2787_278712


namespace NUMINAMATH_CALUDE_portias_high_school_students_portias_high_school_students_proof_l2787_278754

theorem portias_high_school_students : ℕ → ℕ → Prop :=
  fun (portia_students lara_students : ℕ) =>
    (portia_students = 3 * lara_students) →
    (portia_students + lara_students = 2600) →
    (portia_students = 1950)

-- Proof
theorem portias_high_school_students_proof : 
  ∃ (portia_students lara_students : ℕ), 
    portias_high_school_students portia_students lara_students :=
by
  sorry

end NUMINAMATH_CALUDE_portias_high_school_students_portias_high_school_students_proof_l2787_278754


namespace NUMINAMATH_CALUDE_certain_number_problem_l2787_278710

theorem certain_number_problem (x : ℝ) (y : ℝ) (h1 : x = 3) 
  (h2 : (x + 1) / (x + y) = (x + y) / (x + 13)) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2787_278710


namespace NUMINAMATH_CALUDE_intersection_unique_l2787_278705

/-- The line is defined by (x-2)/1 = (y-3)/1 = (z-4)/2 -/
def line (x y z : ℝ) : Prop :=
  (x - 2) = (y - 3) ∧ (x - 2) = (z - 4) / 2

/-- The plane is defined by 2X + Y + Z = 0 -/
def plane (x y z : ℝ) : Prop :=
  2 * x + y + z = 0

/-- The point of intersection -/
def intersection_point : ℝ × ℝ × ℝ := (-0.2, 0.8, -0.4)

theorem intersection_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 ∧ p = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_unique_l2787_278705


namespace NUMINAMATH_CALUDE_geometric_sum_value_l2787_278713

/-- Sum of a geometric series with 15 terms, first term 4/5, and common ratio 4/5 -/
def geometricSum : ℚ :=
  let a : ℚ := 4/5
  let r : ℚ := 4/5
  let n : ℕ := 15
  a * (1 - r^n) / (1 - r)

/-- The sum of the geometric series is equal to 117775277204/30517578125 -/
theorem geometric_sum_value : geometricSum = 117775277204/30517578125 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_value_l2787_278713


namespace NUMINAMATH_CALUDE_red_yellow_flowers_l2787_278738

theorem red_yellow_flowers (total : ℕ) (yellow_white : ℕ) (red_white : ℕ) (red_excess : ℕ) :
  total = 44 →
  yellow_white = 13 →
  red_white = 14 →
  red_excess = 4 →
  ∃ (red_yellow : ℕ), red_yellow = 17 ∧
    total = yellow_white + red_white + red_yellow ∧
    red_white + red_yellow = yellow_white + red_white + red_excess :=
by sorry

end NUMINAMATH_CALUDE_red_yellow_flowers_l2787_278738


namespace NUMINAMATH_CALUDE_fake_coin_strategy_exists_find_fake_coin_correct_l2787_278763

/-- Represents a strategy to find a fake coin among 2^(2^k) coins using dogs -/
structure FakeCoinStrategy (k : ℕ) :=
  (num_tests : ℕ)
  (find_fake_coin : Unit → ℕ)

/-- Theorem stating the existence of a strategy to find the fake coin -/
theorem fake_coin_strategy_exists (k : ℕ) :
  ∃ (strategy : FakeCoinStrategy k),
    strategy.num_tests ≤ 2^k + k + 2 ∧
    strategy.find_fake_coin () < 2^(2^k) :=
by sorry

/-- Function to perform a test with selected coins and a dog -/
def perform_test (selected_coins : Finset ℕ) (dog : ℕ) : Bool :=
sorry

/-- Function to select a dog for testing -/
def select_dog : ℕ :=
sorry

/-- Function to implement the strategy and find the fake coin -/
def find_fake_coin (k : ℕ) : ℕ :=
sorry

/-- Theorem proving the correctness of the find_fake_coin function -/
theorem find_fake_coin_correct (k : ℕ) :
  ∃ (num_tests : ℕ),
    num_tests ≤ 2^k + k + 2 ∧
    find_fake_coin k < 2^(2^k) :=
by sorry

end NUMINAMATH_CALUDE_fake_coin_strategy_exists_find_fake_coin_correct_l2787_278763


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2787_278735

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → abs a > abs b) ∧
  (∃ a b : ℝ, abs a > abs b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2787_278735


namespace NUMINAMATH_CALUDE_parabola_tangent_min_area_l2787_278736

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

/-- The point M -/
def M (y₀ : ℝ) : ℝ × ℝ := (-1, y₀)

/-- The area of triangle MAB -/
noncomputable def triangleArea (p : ℝ) (y₀ : ℝ) : ℝ :=
  2 * Real.sqrt (y₀^2 + 2*p)

/-- The main theorem -/
theorem parabola_tangent_min_area (p : ℝ) :
  p > 0 →
  (∀ y₀ : ℝ, triangleArea p y₀ ≥ 4) →
  (∃ y₀ : ℝ, triangleArea p y₀ = 4) →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_tangent_min_area_l2787_278736


namespace NUMINAMATH_CALUDE_certain_number_is_four_l2787_278732

theorem certain_number_is_four (k : ℝ) (certain_number : ℝ) 
  (h1 : 64 / k = certain_number) 
  (h2 : k = 16) : 
  certain_number = 4 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_four_l2787_278732


namespace NUMINAMATH_CALUDE_absolute_value_and_opposite_l2787_278775

theorem absolute_value_and_opposite :
  (|-2/5| = 2/5) ∧ (-(2023 : ℤ) = -2023) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_opposite_l2787_278775


namespace NUMINAMATH_CALUDE_f_minimum_and_range_l2787_278717

/-- The function f(x) = |2x+1| + |2x-1| -/
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

theorem f_minimum_and_range :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), |2*a + b| + |a| - 1/2 * |a + b| * f x ≥ 0) →
    x ∈ Set.Icc (-1/2) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_and_range_l2787_278717


namespace NUMINAMATH_CALUDE_problem_solution_l2787_278783

theorem problem_solution : ∃ x : ℕ, x = 13 ∧ (4 * x) / 8 = 6 ∧ (4 * x) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2787_278783


namespace NUMINAMATH_CALUDE_sixth_grade_count_l2787_278781

/-- The number of students in the sixth grade -/
def sixth_grade_students : ℕ := 108

/-- The total number of students in fifth and sixth grades -/
def total_students : ℕ := 200

/-- The number of fifth grade students who went to the celebration -/
def fifth_grade_celebration : ℕ := 11

/-- The percentage of sixth grade students who went to the celebration -/
def sixth_grade_celebration_percent : ℚ := 1/4

theorem sixth_grade_count : 
  sixth_grade_students = 108 ∧
  total_students = 200 ∧
  fifth_grade_celebration = 11 ∧
  sixth_grade_celebration_percent = 1/4 ∧
  (total_students - sixth_grade_students - fifth_grade_celebration) = 
  (sixth_grade_students * (1 - sixth_grade_celebration_percent)) :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_count_l2787_278781


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2787_278760

theorem unknown_number_proof (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 192)
  (h_hcf : Nat.gcd a b = 16)
  (h_a : a = 64) :
  b = 48 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2787_278760


namespace NUMINAMATH_CALUDE_avg_percent_grades_5_6_midville_easton_l2787_278780

/-- Represents a school with its total number of students and percentages for each grade --/
structure School where
  total_students : ℕ
  grade_k_percent : ℚ
  grade_1_percent : ℚ
  grade_2_percent : ℚ
  grade_3_percent : ℚ
  grade_4_percent : ℚ
  grade_5_percent : ℚ
  grade_6_percent : ℚ

def midville : School := {
  total_students := 150,
  grade_k_percent := 18/100,
  grade_1_percent := 14/100,
  grade_2_percent := 15/100,
  grade_3_percent := 12/100,
  grade_4_percent := 16/100,
  grade_5_percent := 12/100,
  grade_6_percent := 13/100
}

def easton : School := {
  total_students := 250,
  grade_k_percent := 10/100,
  grade_1_percent := 14/100,
  grade_2_percent := 17/100,
  grade_3_percent := 18/100,
  grade_4_percent := 13/100,
  grade_5_percent := 15/100,
  grade_6_percent := 13/100
}

/-- Calculates the average percentage of students in grades 5 and 6 for two schools combined --/
def avg_percent_grades_5_6 (s1 s2 : School) : ℚ :=
  let total_students := s1.total_students + s2.total_students
  let students_5_6 := s1.total_students * (s1.grade_5_percent + s1.grade_6_percent) +
                      s2.total_students * (s2.grade_5_percent + s2.grade_6_percent)
  students_5_6 / total_students

theorem avg_percent_grades_5_6_midville_easton :
  avg_percent_grades_5_6 midville easton = 2725/10000 := by
  sorry

end NUMINAMATH_CALUDE_avg_percent_grades_5_6_midville_easton_l2787_278780


namespace NUMINAMATH_CALUDE_inequality_proof_l2787_278745

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2787_278745
