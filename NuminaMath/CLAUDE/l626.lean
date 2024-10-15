import Mathlib

namespace NUMINAMATH_CALUDE_cos_75_deg_l626_62667

/-- Proves that cos 75° = (√6 - √2) / 4 using cos 60° and cos 15° -/
theorem cos_75_deg (cos_60_deg : Real) (cos_15_deg : Real) :
  cos_60_deg = 1 / 2 →
  cos_15_deg = (Real.sqrt 6 + Real.sqrt 2) / 4 →
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_deg_l626_62667


namespace NUMINAMATH_CALUDE_revenue_change_with_price_increase_and_quantity_decrease_l626_62693

/-- Theorem: Effect on revenue when price increases and quantity decreases -/
theorem revenue_change_with_price_increase_and_quantity_decrease 
  (P Q : ℝ) (P_new Q_new R_new : ℝ) :
  P_new = P * (1 + 0.30) →
  Q_new = Q * (1 - 0.20) →
  R_new = P_new * Q_new →
  R_new = P * Q * 1.04 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_with_price_increase_and_quantity_decrease_l626_62693


namespace NUMINAMATH_CALUDE_total_gold_spent_l626_62670

-- Define the quantities and prices
def gary_grams : ℝ := 30
def gary_price : ℝ := 15
def anna_grams : ℝ := 50
def anna_price : ℝ := 20
def lisa_grams : ℝ := 40
def lisa_price : ℝ := 15
def john_grams : ℝ := 60
def john_price : ℝ := 18

-- Define conversion rates
def euro_to_dollar : ℝ := 1.1
def pound_to_dollar : ℝ := 1.3

-- Define the total spent function
def total_spent : ℝ := 
  gary_grams * gary_price + 
  anna_grams * anna_price + 
  lisa_grams * lisa_price * euro_to_dollar + 
  john_grams * john_price * pound_to_dollar

-- Theorem statement
theorem total_gold_spent : total_spent = 3514 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_spent_l626_62670


namespace NUMINAMATH_CALUDE_square_of_105_l626_62628

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l626_62628


namespace NUMINAMATH_CALUDE_jennifer_spending_l626_62650

theorem jennifer_spending (initial_amount : ℚ) : 
  (initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 16) → 
  initial_amount = 120 := by
sorry

end NUMINAMATH_CALUDE_jennifer_spending_l626_62650


namespace NUMINAMATH_CALUDE_special_functions_identity_l626_62689

/-- Non-constant, differentiable functions satisfying certain conditions -/
class SpecialFunctions (f g : ℝ → ℝ) where
  non_constant_f : ∃ x y, f x ≠ f y
  non_constant_g : ∃ x y, g x ≠ g y
  differentiable_f : Differentiable ℝ f
  differentiable_g : Differentiable ℝ g
  condition1 : ∀ x y, f (x + y) = f x * f y - g x * g y
  condition2 : ∀ x y, g (x + y) = f x * g y + g x * f y
  condition3 : deriv f 0 = 0

/-- Theorem stating that f(x)^2 + g(x)^2 = 1 for all x ∈ ℝ -/
theorem special_functions_identity {f g : ℝ → ℝ} [SpecialFunctions f g] :
  ∀ x, f x ^ 2 + g x ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_functions_identity_l626_62689


namespace NUMINAMATH_CALUDE_hyperbola_and_condition_implies_m_range_l626_62616

/-- Represents a hyperbola equation -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 3) + y^2 / (m - 4) = 1

/-- Condition for all real x -/
def condition_for_all_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + m + 3 ≥ 0

/-- The range of m -/
def m_range (m : ℝ) : Prop :=
  -2 ≤ m ∧ m < 4

theorem hyperbola_and_condition_implies_m_range :
  ∀ m : ℝ, is_hyperbola m ∧ condition_for_all_x m → m_range m :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_condition_implies_m_range_l626_62616


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l626_62686

/-- The number of volunteers --/
def n : ℕ := 6

/-- The number of exhibition areas --/
def m : ℕ := 4

/-- The number of areas that require one person --/
def single_person_areas : ℕ := 2

/-- The number of areas that require two people --/
def double_person_areas : ℕ := 2

/-- The number of specific volunteers that cannot be together --/
def restricted_volunteers : ℕ := 2

/-- The total number of arrangements without restrictions --/
def total_arrangements : ℕ := 180

/-- The number of arrangements where the restricted volunteers are together --/
def restricted_arrangements : ℕ := 24

theorem volunteer_arrangement_count :
  (n = 6) →
  (m = 4) →
  (single_person_areas = 2) →
  (double_person_areas = 2) →
  (restricted_volunteers = 2) →
  (total_arrangements = 180) →
  (restricted_arrangements = 24) →
  (total_arrangements - restricted_arrangements = 156) := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l626_62686


namespace NUMINAMATH_CALUDE_four_layer_grid_triangles_l626_62688

/-- Calculates the total number of triangles in a triangular grid with a given number of layers. -/
def triangles_in_grid (layers : ℕ) : ℕ :=
  let small_triangles := (layers * (layers + 1)) / 2
  let medium_triangles := if layers ≥ 3 then (layers - 2) * (layers - 1) / 2 else 0
  let large_triangles := 1
  small_triangles + medium_triangles + large_triangles

/-- Theorem stating that a triangular grid with 4 layers contains 21 triangles. -/
theorem four_layer_grid_triangles :
  triangles_in_grid 4 = 21 :=
by sorry

end NUMINAMATH_CALUDE_four_layer_grid_triangles_l626_62688


namespace NUMINAMATH_CALUDE_fraction_equality_l626_62649

theorem fraction_equality (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l626_62649


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l626_62646

/-- Proves that adding 24 ounces of pure gold to a 16-ounce alloy that is 50% gold
    will result in an alloy that is 80% gold. -/
theorem gold_alloy_composition (initial_weight : ℝ) (initial_purity : ℝ) 
    (added_gold : ℝ) (final_purity : ℝ) : 
  initial_weight = 16 →
  initial_purity = 0.5 →
  added_gold = 24 →
  final_purity = 0.8 →
  (initial_weight * initial_purity + added_gold) / (initial_weight + added_gold) = final_purity :=
by
  sorry

#check gold_alloy_composition

end NUMINAMATH_CALUDE_gold_alloy_composition_l626_62646


namespace NUMINAMATH_CALUDE_john_yasmin_children_ratio_l626_62665

/-- The number of children John has -/
def john_children : ℕ := sorry

/-- The number of children Yasmin has -/
def yasmin_children : ℕ := 2

/-- The total number of grandchildren Gabriel has -/
def gabriel_grandchildren : ℕ := 6

/-- The ratio of John's children to Yasmin's children -/
def children_ratio : ℚ := john_children / yasmin_children

theorem john_yasmin_children_ratio :
  (john_children + yasmin_children = gabriel_grandchildren) →
  children_ratio = 2 := by
sorry

end NUMINAMATH_CALUDE_john_yasmin_children_ratio_l626_62665


namespace NUMINAMATH_CALUDE_fourth_guard_distance_theorem_l626_62634

/-- Represents a rectangular classified area with guards -/
structure ClassifiedArea where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  guard_count : ℕ
  three_guards_distance : ℝ

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (area : ClassifiedArea) : ℝ :=
  area.perimeter - area.three_guards_distance

/-- Theorem stating the distance run by the fourth guard -/
theorem fourth_guard_distance_theorem (area : ClassifiedArea) 
  (h1 : area.length = 200)
  (h2 : area.width = 300)
  (h3 : area.perimeter = 2 * (area.length + area.width))
  (h4 : area.guard_count = 4)
  (h5 : area.three_guards_distance = 850)
  : fourth_guard_distance area = 150 := by
  sorry

end NUMINAMATH_CALUDE_fourth_guard_distance_theorem_l626_62634


namespace NUMINAMATH_CALUDE_f_one_values_l626_62636

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y
axiom not_constant_zero : ∃ (x : ℝ), f x ≠ 0
axiom exists_a : ∃ (a : ℝ), a ≠ 0 ∧ f a = 2

-- Theorem statement
theorem f_one_values : (f 1 = Real.sqrt 2) ∨ (f 1 = -Real.sqrt 2) :=
sorry

end

end NUMINAMATH_CALUDE_f_one_values_l626_62636


namespace NUMINAMATH_CALUDE_intersection_of_lines_l626_62612

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (3, -2, 6) → 
  B = (13, -12, 11) → 
  C = (1, 5, -3) → 
  D = (3, -1, 9) → 
  ∃ t s : ℝ, 
    (3 + 10*t, -2 - 10*t, 6 + 5*t) = 
    (1 + 2*s, 5 - 6*s, -3 + 12*s) ∧
    (3 + 10*t, -2 - 10*t, 6 + 5*t) = (7.5, -6.5, 8.25) := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_intersection_of_lines_l626_62612


namespace NUMINAMATH_CALUDE_binomial_15_5_l626_62617

theorem binomial_15_5 : Nat.choose 15 5 = 3003 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_5_l626_62617


namespace NUMINAMATH_CALUDE_power_sum_l626_62664

theorem power_sum (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l626_62664


namespace NUMINAMATH_CALUDE_logic_propositions_l626_62684

-- Define the propositions
def corresponding_angles_equal (l₁ l₂ : Line) : Prop := sorry
def lines_parallel (l₁ l₂ : Line) : Prop := sorry

-- Define the sine function and angle measure
def sin : ℝ → ℝ := sorry
def degree : ℝ → ℝ := sorry

-- Define the theorem
theorem logic_propositions :
  -- 1. Contrapositive
  (∀ l₁ l₂ : Line, (corresponding_angles_equal l₁ l₂ → lines_parallel l₁ l₂) ↔ 
    (¬lines_parallel l₁ l₂ → ¬corresponding_angles_equal l₁ l₂)) ∧
  -- 2. Necessary but not sufficient condition
  (∀ α : ℝ, sin α = 1/2 → degree α = 30) ∧
  (∃ β : ℝ, sin β = 1/2 ∧ degree β ≠ 30) ∧
  -- 3. Falsity of conjunction
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  -- 4. Negation of existence
  (¬(∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_logic_propositions_l626_62684


namespace NUMINAMATH_CALUDE_inequality_proof_l626_62630

theorem inequality_proof (x y : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l626_62630


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l626_62662

/-- Circle C₁ in the Cartesian plane -/
def C₁ (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - (4*m+6)*y - 4 = 0

/-- Circle C₂ in the Cartesian plane -/
def C₂ (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = (x + 2)^2 + (y - 3)^2

/-- The theorem stating the value of m for the given conditions -/
theorem circle_intersection_theorem (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  C₁ m x₁ y₁ ∧ C₁ m x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
  x₁^2 - x₂^2 = y₂^2 - y₁^2 →
  m = -6 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l626_62662


namespace NUMINAMATH_CALUDE_field_trip_seats_l626_62699

theorem field_trip_seats (students : ℕ) (buses : ℕ) (seats_per_bus : ℕ) 
  (h1 : students = 28) 
  (h2 : buses = 4) 
  (h3 : students = buses * seats_per_bus) : 
  seats_per_bus = 7 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_seats_l626_62699


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l626_62661

/-- Given a bag with 5 balls where the probability of drawing 2 white balls out of 2 draws is 3/10,
    prove that the probability of getting at least 1 white ball in 2 draws is 9/10. -/
theorem prob_at_least_one_white (total_balls : ℕ) (prob_two_white : ℚ) :
  total_balls = 5 →
  prob_two_white = 3 / 10 →
  (∃ white_balls : ℕ, white_balls ≤ total_balls ∧
    prob_two_white = (white_balls.choose 2 : ℚ) / (total_balls.choose 2 : ℚ)) →
  (1 : ℚ) - ((total_balls - white_balls).choose 2 : ℚ) / (total_balls.choose 2 : ℚ) = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l626_62661


namespace NUMINAMATH_CALUDE_grape_price_l626_62637

theorem grape_price (G : ℝ) : 
  (8 * G + 11 * 55 = 1165) → G = 70 := by sorry

end NUMINAMATH_CALUDE_grape_price_l626_62637


namespace NUMINAMATH_CALUDE_quadratic_roots_l626_62629

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop :=
  (a - 1) * x^2 + 2 * x + a - 1 = 0

theorem quadratic_roots :
  -- Part 1
  (∃ a : ℝ, quadratic_equation a 2 ∧ 
    ∃ x : ℝ, x ≠ 2 ∧ quadratic_equation a x) →
  (quadratic_equation (1/5) 2 ∧ quadratic_equation (1/5) (1/2)) ∧
  -- Part 2
  (∃ x : ℝ, quadratic_equation 1 x ↔ x = 0) ∧
  (∃ x : ℝ, quadratic_equation 2 x ↔ x = -1) ∧
  (∃ x : ℝ, quadratic_equation 0 x ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l626_62629


namespace NUMINAMATH_CALUDE_divisibility_by_ten_l626_62645

theorem divisibility_by_ten (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a * b * c) % 10 = 0 ∧
  (a * b * d) % 10 = 0 ∧
  (a * b * e) % 10 = 0 ∧
  (a * c * d) % 10 = 0 ∧
  (a * c * e) % 10 = 0 ∧
  (a * d * e) % 10 = 0 ∧
  (b * c * d) % 10 = 0 ∧
  (b * c * e) % 10 = 0 ∧
  (b * d * e) % 10 = 0 ∧
  (c * d * e) % 10 = 0 →
  a % 10 = 0 ∨ b % 10 = 0 ∨ c % 10 = 0 ∨ d % 10 = 0 ∨ e % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_ten_l626_62645


namespace NUMINAMATH_CALUDE_expand_expression_l626_62609

theorem expand_expression (y : ℝ) : 5 * (4 * y^3 - 3 * y^2 + y - 7) = 20 * y^3 - 15 * y^2 + 5 * y - 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l626_62609


namespace NUMINAMATH_CALUDE_special_number_in_list_l626_62677

theorem special_number_in_list (numbers : List ℝ) (n : ℝ) : 
  numbers.length = 21 ∧ 
  n ∈ numbers ∧
  n = 4 * ((numbers.sum - n) / 20) →
  n = (1 / 6) * numbers.sum :=
by sorry

end NUMINAMATH_CALUDE_special_number_in_list_l626_62677


namespace NUMINAMATH_CALUDE_chessboard_polygon_tasteful_tiling_l626_62673

-- Define a chessboard polygon
def ChessboardPolygon : Type := sorry

-- Define a domino
def Domino : Type := sorry

-- Define a tiling
def Tiling (p : ChessboardPolygon) : Type := sorry

-- Define a tasteful tiling
def TastefulTiling (p : ChessboardPolygon) : Type := sorry

-- Define the property of being tileable by dominoes
def IsTileable (p : ChessboardPolygon) : Prop := sorry

-- Theorem statement
theorem chessboard_polygon_tasteful_tiling 
  (p : ChessboardPolygon) (h : IsTileable p) :
  (∃ t : TastefulTiling p, True) ∧ 
  (∀ t1 t2 : TastefulTiling p, t1 = t2) :=
sorry

end NUMINAMATH_CALUDE_chessboard_polygon_tasteful_tiling_l626_62673


namespace NUMINAMATH_CALUDE_power_sum_equals_39_l626_62681

theorem power_sum_equals_39 : 
  (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 3 + 2^1 + 2^2 + 2^3 + 2^4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_39_l626_62681


namespace NUMINAMATH_CALUDE_stock_price_decrease_l626_62643

theorem stock_price_decrease (P : ℝ) (X : ℝ) : 
  P > 0 →
  1.20 * P * (1 - X) * 1.35 = 1.215 * P →
  X = 0.25 := by
sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l626_62643


namespace NUMINAMATH_CALUDE_no_binomial_arithmetic_progression_l626_62648

theorem no_binomial_arithmetic_progression :
  ∀ (n k : ℕ+), k ≤ n →
    ¬∃ (d : ℚ), 
      (Nat.choose n (k + 1) : ℚ) - (Nat.choose n k : ℚ) = d ∧
      (Nat.choose n (k + 2) : ℚ) - (Nat.choose n (k + 1) : ℚ) = d ∧
      (Nat.choose n (k + 3) : ℚ) - (Nat.choose n (k + 2) : ℚ) = d :=
by sorry

end NUMINAMATH_CALUDE_no_binomial_arithmetic_progression_l626_62648


namespace NUMINAMATH_CALUDE_max_value_of_b_l626_62605

theorem max_value_of_b (a b c : ℝ) : 
  (∃ q : ℝ, a = b / q ∧ c = b * q) →  -- geometric sequence condition
  (b + 2 = (a + 6 + c + 1) / 2) →     -- arithmetic sequence condition
  b ≤ 3/4 :=                          -- maximum value of b
by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l626_62605


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l626_62671

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part (1)
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem for part (2)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l626_62671


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l626_62640

/-- The line equation 5y - 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y - 3 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (-5, 0)

theorem intersection_point_on_line_and_x_axis :
  line_equation intersection_point.1 intersection_point.2 ∧
  on_x_axis intersection_point.1 intersection_point.2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l626_62640


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l626_62656

theorem geometric_arithmetic_progression_problem (a b c : ℝ) :
  (∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ 12 = a * q^2) ∧  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ 9 = a + 2 * d) →        -- Arithmetic progression condition
  ((a = -9 ∧ b = -6 ∧ c = 12) ∨ (a = 15 ∧ b = 12 ∧ c = 9)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l626_62656


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l626_62678

def P : Set ℕ := {x : ℕ | x * (x - 3) ≤ 0}
def Q : Set ℕ := {x : ℕ | x ≥ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l626_62678


namespace NUMINAMATH_CALUDE_associates_hired_to_change_ratio_l626_62658

/-- The number of additional associates hired to change the ratio -/
def additional_associates (initial_ratio_partners initial_ratio_associates new_ratio_partners new_ratio_associates current_partners : ℕ) : ℕ :=
  let initial_associates := (initial_ratio_associates * current_partners) / initial_ratio_partners
  let total_new_associates := (new_ratio_associates * current_partners) / new_ratio_partners
  total_new_associates - initial_associates

/-- Theorem stating that 50 additional associates were hired to change the ratio -/
theorem associates_hired_to_change_ratio :
  additional_associates 2 63 1 34 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_associates_hired_to_change_ratio_l626_62658


namespace NUMINAMATH_CALUDE_height_comparison_l626_62606

theorem height_comparison (a b : ℝ) (h : a = b * 0.6) :
  (b - a) / a = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l626_62606


namespace NUMINAMATH_CALUDE_gino_bears_count_l626_62666

theorem gino_bears_count (total : ℕ) (brown : ℕ) (white : ℕ) (black : ℕ) : 
  total = 66 → brown = 15 → white = 24 → total = brown + white + black → black = 27 := by
sorry

end NUMINAMATH_CALUDE_gino_bears_count_l626_62666


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l626_62635

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z : ℝ), (x^2 + 2*y^2 + z^2)^2 ≤ n*(x^4 + 3*y^4 + z^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x^2 + 2*y^2 + z^2)^2 > m*(x^4 + 3*y^4 + z^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l626_62635


namespace NUMINAMATH_CALUDE_range_of_f_when_a_is_1_a_values_when_f_min_is_3_l626_62672

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Theorem 1: Range of f when a = 1
theorem range_of_f_when_a_is_1 :
  ∀ y ∈ Set.Icc 0 9, ∃ x ∈ Set.Icc 0 2, f 1 x = y ∧
  ∀ x ∈ Set.Icc 0 2, 0 ≤ f 1 x ∧ f 1 x ≤ 9 :=
sorry

-- Theorem 2: Values of a when f has minimum value 3
theorem a_values_when_f_min_is_3 :
  (∃ x ∈ Set.Icc 0 2, f a x = 3 ∧ ∀ y ∈ Set.Icc 0 2, f a y ≥ 3) →
  (a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_when_a_is_1_a_values_when_f_min_is_3_l626_62672


namespace NUMINAMATH_CALUDE_video_votes_l626_62632

theorem video_votes (score : ℕ) (like_percent : ℚ) (dislike_percent : ℚ) (neutral_percent : ℚ) :
  score = 180 →
  like_percent = 60 / 100 →
  dislike_percent = 20 / 100 →
  neutral_percent = 20 / 100 →
  like_percent + dislike_percent + neutral_percent = 1 →
  ∃ (total_votes : ℕ), 
    (↑score : ℚ) = (like_percent - dislike_percent) * ↑total_votes ∧
    total_votes = 450 := by
  sorry

end NUMINAMATH_CALUDE_video_votes_l626_62632


namespace NUMINAMATH_CALUDE_second_integer_problem_l626_62694

theorem second_integer_problem (x y : ℕ+) (hx : x = 3) (h : x * y + x = 33) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_integer_problem_l626_62694


namespace NUMINAMATH_CALUDE_number_of_points_l626_62623

theorem number_of_points (initial_sum : ℝ) (shift : ℝ) (final_sum : ℝ) : 
  initial_sum = -1.5 → 
  shift = -2 → 
  final_sum = -15.5 → 
  (final_sum - initial_sum) / shift = 7 := by
sorry

end NUMINAMATH_CALUDE_number_of_points_l626_62623


namespace NUMINAMATH_CALUDE_typists_problem_l626_62696

theorem typists_problem (initial_letters : ℕ) (initial_time : ℕ) (new_typists : ℕ) (new_letters : ℕ) (new_time : ℕ) :
  initial_letters = 48 →
  initial_time = 20 →
  new_typists = 30 →
  new_letters = 216 →
  new_time = 60 →
  ∃ x : ℕ, x > 0 ∧ (initial_letters / x : ℚ) * new_typists * (new_time / initial_time : ℚ) = new_letters :=
by
  sorry

end NUMINAMATH_CALUDE_typists_problem_l626_62696


namespace NUMINAMATH_CALUDE_lcm_of_36_and_100_l626_62638

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_100_l626_62638


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_12_825_l626_62655

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α
  pos : 0 < side

/-- Represents the configuration of three squares aligned on their bottom edges -/
structure SquareConfiguration (α : Type*) [LinearOrderedField α] where
  small : Square α
  medium : Square α
  large : Square α
  alignment : small.side + medium.side + large.side > 0

/-- Calculates the area of the quadrilateral formed in the square configuration -/
noncomputable def quadrilateralArea {α : Type*} [LinearOrderedField α] (config : SquareConfiguration α) : α :=
  sorry

/-- Theorem stating that the area of the quadrilateral in the given configuration is 12.825 -/
theorem quadrilateral_area_is_12_825 :
  let config : SquareConfiguration ℝ := {
    small := { side := 3, pos := by norm_num },
    medium := { side := 5, pos := by norm_num },
    large := { side := 7, pos := by norm_num },
    alignment := by norm_num
  }
  quadrilateralArea config = 12.825 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_12_825_l626_62655


namespace NUMINAMATH_CALUDE_line_equivalence_l626_62659

/-- Definition of the line using dot product equation -/
def line_equation (x y : ℝ) : Prop :=
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0

/-- Slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

/-- The slope and y-intercept of the line -/
def slope_intercept_params : ℝ × ℝ := (2, -10)

theorem line_equivalence :
  ∀ (x y : ℝ),
    line_equation x y ↔ slope_intercept_form (slope_intercept_params.1) (slope_intercept_params.2) x y :=
by sorry

#check line_equivalence

end NUMINAMATH_CALUDE_line_equivalence_l626_62659


namespace NUMINAMATH_CALUDE_solution_set_l626_62691

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom increasing_f : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0
axiom odd_shifted : ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

-- State the theorem
theorem solution_set (x : ℝ) : f (1 - x) > 0 ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l626_62691


namespace NUMINAMATH_CALUDE_coefficient_is_nine_l626_62620

/-- The coefficient of x^2 in the expansion of (1+x)^10 - (1-x)^9 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 10 2) - (Nat.choose 9 2)

/-- Theorem stating that the coefficient of x^2 in the expansion of (1+x)^10 - (1-x)^9 is 9 -/
theorem coefficient_is_nine : coefficient_x_squared = 9 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_is_nine_l626_62620


namespace NUMINAMATH_CALUDE_range_of_a_l626_62651

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ -3 ∨ x ≥ 0}
def Q (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Define the conditions
def condition_not_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def condition_not_q (x a : ℝ) : Prop := x > a

-- Define the relationship between q and p
def q_sufficient_not_necessary_for_p (a : ℝ) : Prop :=
  Q a ⊂ P ∧ Q a ≠ P

-- The main theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, condition_not_p x → condition_not_q x a) →
  q_sufficient_not_necessary_for_p a →
  a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l626_62651


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l626_62698

theorem cubic_roots_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  s^3 - 15*s^2 + 25*s - 10 = 0 →
  t^3 - 15*t^2 + 25*t - 10 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 175/11 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l626_62698


namespace NUMINAMATH_CALUDE_problem_solution_l626_62600

theorem problem_solution (a b c d : ℕ) 
  (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  a * 1000 + b * 100 + c * 10 + d = 1949 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l626_62600


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l626_62682

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l626_62682


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l626_62603

theorem sum_of_four_integers (a b c d : ℕ+) : 
  (a > 1) → (b > 1) → (c > 1) → (d > 1) →
  (a * b * c * d = 1000000) →
  (Nat.gcd a.val b.val = 1) → (Nat.gcd a.val c.val = 1) → (Nat.gcd a.val d.val = 1) →
  (Nat.gcd b.val c.val = 1) → (Nat.gcd b.val d.val = 1) →
  (Nat.gcd c.val d.val = 1) →
  (a + b + c + d = 15698) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l626_62603


namespace NUMINAMATH_CALUDE_profit_calculation_l626_62675

/-- Represents the profit distribution in a partnership --/
structure ProfitDistribution where
  mary_investment : ℝ
  mike_investment : ℝ
  total_profit : ℝ
  effort_share : ℝ
  investment_share : ℝ
  mary_extra : ℝ

/-- Theorem stating the profit calculation based on given conditions --/
theorem profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.mary_investment = 600)
  (h2 : pd.mike_investment = 400)
  (h3 : pd.effort_share = 1/3)
  (h4 : pd.investment_share = 2/3)
  (h5 : pd.mary_extra = 1000)
  (h6 : pd.effort_share + pd.investment_share = 1) :
  pd.total_profit = 15000 := by
  sorry

#check profit_calculation

end NUMINAMATH_CALUDE_profit_calculation_l626_62675


namespace NUMINAMATH_CALUDE_age_sum_theorem_l626_62663

theorem age_sum_theorem (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 2050 → 
  a + b + c = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l626_62663


namespace NUMINAMATH_CALUDE_lemonade_juice_requirement_l626_62607

/-- The amount of lemon juice required for a lemonade mixture -/
def lemon_juice_required (total_volume : ℚ) (water_parts : ℕ) (juice_parts : ℕ) : ℚ :=
  (total_volume * juice_parts) / (water_parts + juice_parts)

/-- Conversion from gallons to quarts -/
def gallons_to_quarts (gallons : ℚ) : ℚ := 4 * gallons

theorem lemonade_juice_requirement :
  let total_volume := (3 : ℚ) / 2  -- 1.5 gallons
  let water_parts := 5
  let juice_parts := 3
  lemon_juice_required (gallons_to_quarts total_volume) water_parts juice_parts = (9 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_juice_requirement_l626_62607


namespace NUMINAMATH_CALUDE_jerrys_debt_problem_jerrys_total_debt_l626_62601

/-- Jerry's debt payment problem -/
theorem jerrys_debt_problem (payment_two_months_ago : ℕ) 
                            (payment_increase : ℕ) 
                            (remaining_debt : ℕ) : ℕ :=
  let payment_last_month := payment_two_months_ago + payment_increase
  let total_paid := payment_two_months_ago + payment_last_month
  let total_debt := total_paid + remaining_debt
  total_debt

/-- Proof of Jerry's total debt -/
theorem jerrys_total_debt : jerrys_debt_problem 12 3 23 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_debt_problem_jerrys_total_debt_l626_62601


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l626_62631

/-- Given a parabola y² = -2x with focus F, and a point A(x₀, y₀) on the parabola,
    if |AF| = 3/2, then x₀ = -1 -/
theorem parabola_focus_distance (x₀ y₀ : ℝ) :
  y₀^2 = -2*x₀ →  -- A is on the parabola
  ∃ F : ℝ × ℝ, (F.1 = 1/2 ∧ F.2 = 0) →  -- Focus coordinates
  (x₀ - F.1)^2 + (y₀ - F.2)^2 = (3/2)^2 →  -- |AF| = 3/2
  x₀ = -1 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l626_62631


namespace NUMINAMATH_CALUDE_first_customer_boxes_l626_62608

def cookie_problem (x : ℚ) : Prop :=
  let second_customer := 4 * x
  let third_customer := second_customer / 2
  let fourth_customer := 3 * third_customer
  let final_customer := 10
  let total_sold := x + second_customer + third_customer + fourth_customer + final_customer
  let goal := 150
  let left_to_sell := 75
  total_sold + left_to_sell = goal

theorem first_customer_boxes : ∃ x : ℚ, cookie_problem x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_customer_boxes_l626_62608


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solutions_l626_62660

theorem quadratic_equation_one_solutions (x : ℝ) :
  x^2 - 6*x - 1 = 0 ↔ x = 3 + Real.sqrt 10 ∨ x = 3 - Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solutions_l626_62660


namespace NUMINAMATH_CALUDE_average_of_remaining_checks_l626_62653

def travelers_checks_problem (x y z : ℕ) : Prop :=
  x + y = 30 ∧ 
  50 * x + z * y = 1800 ∧ 
  x ≥ 24 ∧
  z > 0

theorem average_of_remaining_checks (x y z : ℕ) 
  (h : travelers_checks_problem x y z) : 
  (1800 - 50 * 24) / (30 - 24) = 100 :=
sorry

end NUMINAMATH_CALUDE_average_of_remaining_checks_l626_62653


namespace NUMINAMATH_CALUDE_customized_packaging_combinations_l626_62644

def wrapping_papers : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

theorem customized_packaging_combinations : 
  wrapping_papers * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end NUMINAMATH_CALUDE_customized_packaging_combinations_l626_62644


namespace NUMINAMATH_CALUDE_unique_solution_l626_62604

theorem unique_solution : ∃! (n : ℕ+), 
  Real.sin (π / (3 * n.val : ℝ)) + Real.cos (π / (3 * n.val : ℝ)) = Real.sqrt (2 * n.val : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l626_62604


namespace NUMINAMATH_CALUDE_prove_first_divisor_l626_62647

def least_number : ℕ := 1394

def first_divisor : ℕ := 6

theorem prove_first_divisor :
  (least_number % first_divisor = 14) ∧
  (2535 % first_divisor = 1929) ∧
  (40 % first_divisor = 34) :=
by sorry

end NUMINAMATH_CALUDE_prove_first_divisor_l626_62647


namespace NUMINAMATH_CALUDE_books_sold_on_friday_l626_62619

theorem books_sold_on_friday (initial_stock : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (not_sold : ℕ)
  (h1 : initial_stock = 800)
  (h2 : monday = 60)
  (h3 : tuesday = 10)
  (h4 : wednesday = 20)
  (h5 : thursday = 44)
  (h6 : not_sold = 600) :
  initial_stock - not_sold - (monday + tuesday + wednesday + thursday) = 66 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_on_friday_l626_62619


namespace NUMINAMATH_CALUDE_container_volume_comparison_l626_62674

theorem container_volume_comparison (x y : ℝ) (h : x ≠ y) :
  x^3 + y^3 > x^2*y + x*y^2 := by
  sorry

#check container_volume_comparison

end NUMINAMATH_CALUDE_container_volume_comparison_l626_62674


namespace NUMINAMATH_CALUDE_train_crossing_tree_time_l626_62685

/-- Given a train and a platform with specific properties, calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_pass_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : platform_length = 400)
  (h3 : time_pass_platform = 160) :
  (train_length / ((train_length + platform_length) / time_pass_platform)) = 120 := by
  sorry

#check train_crossing_tree_time

end NUMINAMATH_CALUDE_train_crossing_tree_time_l626_62685


namespace NUMINAMATH_CALUDE_sequence_general_term_l626_62621

/-- Given a sequence {a_n} with the sum of the first n terms S_n satisfying
    S_n + a_n = (n-1) / (n(n+1)) for n = 1, 2, ..., 
    prove that the general term a_n = 1/(2^n) - 1/(n(n+1)). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n + a n = (n - 1) / (n * (n + 1))) →
  (∀ n : ℕ, n ≥ 1 → a n = 1 / (2^n) - 1 / (n * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l626_62621


namespace NUMINAMATH_CALUDE_philatelist_stamps_problem_l626_62695

theorem philatelist_stamps_problem :
  ∃! x : ℕ,
    x % 3 = 1 ∧
    x % 5 = 3 ∧
    x % 7 = 5 ∧
    150 < x ∧
    x ≤ 300 ∧
    x = 208 := by
  sorry

end NUMINAMATH_CALUDE_philatelist_stamps_problem_l626_62695


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l626_62625

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  third_term : a 3 = 2
  product_46 : a 4 * a 6 = 16

/-- The main theorem -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence) :
  (seq.a 9 - seq.a 11) / (seq.a 5 - seq.a 7) = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l626_62625


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l626_62668

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l626_62668


namespace NUMINAMATH_CALUDE_min_angle_B_in_special_triangle_l626_62679

open Real

theorem min_angle_B_in_special_triangle (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  ∃ k : ℝ, tan A + k = (1 + sqrt 2) * tan B ∧ (1 + sqrt 2) * tan B + k = tan C →
  π / 4 ≤ B :=
by sorry

end NUMINAMATH_CALUDE_min_angle_B_in_special_triangle_l626_62679


namespace NUMINAMATH_CALUDE_function_periodicity_l626_62641

/-- Given a > 0 and a function f satisfying the specified condition, 
    prove that f is periodic with period 2a -/
theorem function_periodicity 
  (a : ℝ) 
  (ha : a > 0) 
  (f : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) : 
  ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, f (x + b) = f x :=
sorry

end NUMINAMATH_CALUDE_function_periodicity_l626_62641


namespace NUMINAMATH_CALUDE_smallest_norwegian_number_l626_62690

/-- A number is Norwegian if it has three distinct positive divisors whose sum is equal to 2022. -/
def IsNorwegian (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧
    d₁ + d₂ + d₃ = 2022

/-- 1344 is the smallest Norwegian number. -/
theorem smallest_norwegian_number : 
  IsNorwegian 1344 ∧ ∀ m : ℕ, m < 1344 → ¬IsNorwegian m :=
by sorry

end NUMINAMATH_CALUDE_smallest_norwegian_number_l626_62690


namespace NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l626_62633

theorem three_numbers_in_unit_interval (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l626_62633


namespace NUMINAMATH_CALUDE_expression_not_constant_l626_62692

theorem expression_not_constant (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  ¬ ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 3 → x ≠ -2 → 
    (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - 
    (x^2 + 4 * x + 4) / ((x - 3) * (x + 2)) = c :=
by sorry

end NUMINAMATH_CALUDE_expression_not_constant_l626_62692


namespace NUMINAMATH_CALUDE_triangle_area_l626_62654

/-- Given a triangle ABC where sin A = 3/5 and the dot product of vectors AB and AC is 8,
    prove that the area of the triangle is 3. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let sinA : ℝ := 3/5
  let dotProduct : ℝ := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  dotProduct = 8 →
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
         Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * sinA = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l626_62654


namespace NUMINAMATH_CALUDE_nicoles_clothes_theorem_l626_62652

/-- Calculates the total number of clothing pieces Nicole ends up with --/
def nicoles_total_clothes (nicole_start : ℕ) : ℕ :=
  let first_sister := nicole_start / 3
  let second_sister := nicole_start + 5
  let third_sister := 2 * first_sister
  let youngest_four_total := nicole_start + first_sister + second_sister + third_sister
  let oldest_sister := (youngest_four_total / 4 * 3 + (youngest_four_total % 4) / 2 + 1) / 2
  nicole_start + first_sister + second_sister + third_sister + oldest_sister

theorem nicoles_clothes_theorem :
  nicoles_total_clothes 15 = 69 := by
  sorry

end NUMINAMATH_CALUDE_nicoles_clothes_theorem_l626_62652


namespace NUMINAMATH_CALUDE_points_on_line_procedure_l626_62613

theorem points_on_line_procedure (x : ℕ) : ∃ x, 9 * x - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_procedure_l626_62613


namespace NUMINAMATH_CALUDE_a_range_l626_62687

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a*x^2 - x + a)

-- Define the theorem
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (a ≥ 1/2 ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_a_range_l626_62687


namespace NUMINAMATH_CALUDE_cone_base_diameter_l626_62626

/-- A cone with surface area 3π and lateral surface that unfolds into a semicircle -/
structure Cone where
  /-- The radius of the base of the cone -/
  radius : ℝ
  /-- The slant height of the cone -/
  slant_height : ℝ
  /-- The lateral surface unfolds into a semicircle -/
  lateral_surface_semicircle : slant_height = 2 * radius
  /-- The surface area of the cone is 3π -/
  surface_area : π * radius^2 + π * radius * slant_height = 3 * π

/-- The diameter of the base of the cone is 2 -/
theorem cone_base_diameter (c : Cone) : 2 * c.radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l626_62626


namespace NUMINAMATH_CALUDE_fraction_inequality_l626_62676

theorem fraction_inequality (a b : ℝ) (h : b < a ∧ a < 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l626_62676


namespace NUMINAMATH_CALUDE_cookie_making_time_l626_62602

/-- Given the total time to make cookies, baking time, and icing hardening times,
    prove that the time for making dough and cooling cookies is 45 minutes. -/
theorem cookie_making_time (total_time baking_time white_icing_time chocolate_icing_time : ℕ)
  (h1 : total_time = 120)
  (h2 : baking_time = 15)
  (h3 : white_icing_time = 30)
  (h4 : chocolate_icing_time = 30) :
  total_time - (baking_time + white_icing_time + chocolate_icing_time) = 45 := by
  sorry

#check cookie_making_time

end NUMINAMATH_CALUDE_cookie_making_time_l626_62602


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_not_neg_6_4_l626_62680

/-- Two lines in 3D space --/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew --/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (b : ℝ), l1.point.2.2 = b ∧ 
  ¬∃ (t u : ℝ), 
    (l1.point.1 + t * l1.direction.1 = l2.point.1 + u * l2.direction.1) ∧
    (l1.point.2.1 + t * l1.direction.2.1 = l2.point.2.1 + u * l2.direction.2.1) ∧
    (b + t * l1.direction.2.2 = l2.point.2.2 + u * l2.direction.2.2)

theorem lines_skew_iff_b_not_neg_6_4 :
  ∀ (b : ℝ), are_skew 
    (Line3D.mk (2, 3, b) (3, 4, 5)) 
    (Line3D.mk (5, 2, 1) (6, 3, 2))
  ↔ b ≠ -6.4 := by sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_not_neg_6_4_l626_62680


namespace NUMINAMATH_CALUDE_fraction_equality_l626_62683

theorem fraction_equality : (36 + 12) / (6 - 3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l626_62683


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l626_62614

/-- Represents a small die in the 4x4x4 cube -/
structure SmallDie where
  /-- The value on each face of the die -/
  faces : Fin 6 → ℕ
  /-- The property that opposite sides sum to 7 -/
  opposite_sum_seven : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the 4x4x4 cube made of small dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → SmallDie

/-- Calculates the sum of visible values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ := sorry

/-- Theorem stating the smallest possible sum of visible values -/
theorem smallest_visible_sum (cube : LargeCube) :
  visible_sum cube ≥ 144 ∧ ∃ (optimal_cube : LargeCube), visible_sum optimal_cube = 144 := by sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l626_62614


namespace NUMINAMATH_CALUDE_oil_barrels_problem_l626_62610

theorem oil_barrels_problem (a b : ℝ) : 
  a > 0 ∧ b > 0 →  -- Initial amounts are positive
  (2/3 * a + 1/5 * (b + 1/3 * a) = 24) ∧  -- Amount in A after transfers
  ((b + 1/3 * a) * 4/5 = 24) →  -- Amount in B after transfers
  a - b = 6 := by sorry

end NUMINAMATH_CALUDE_oil_barrels_problem_l626_62610


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l626_62627

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l626_62627


namespace NUMINAMATH_CALUDE_logarithmic_equation_proof_l626_62639

theorem logarithmic_equation_proof : 2 * (Real.log 10 / Real.log 5) + Real.log (1/4) / Real.log 5 + 2^(Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_proof_l626_62639


namespace NUMINAMATH_CALUDE_probability_theorem_l626_62611

def is_valid (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧
  1 ≤ b ∧ b ≤ 60 ∧
  1 ≤ c ∧ c ≤ 60 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def satisfies_condition (a b c : ℕ) : Prop :=
  ∃ m : ℕ, (a * b * c + a + b + c) = 6 * m - 2

def total_combinations : ℕ := Nat.choose 60 3

def valid_combinations : ℕ := 14620

theorem probability_theorem :
  (valid_combinations : ℚ) / total_combinations = 2437 / 5707 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l626_62611


namespace NUMINAMATH_CALUDE_workshop_efficiency_l626_62642

theorem workshop_efficiency (x : ℝ) : 
  (1500 / x - 1500 / (2.5 * x) = 18) → x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_efficiency_l626_62642


namespace NUMINAMATH_CALUDE_total_watermelon_slices_l626_62669

-- Define the number of watermelons and slices for each person
def danny_watermelons : ℕ := 3
def danny_slices_per_melon : ℕ := 10

def sister_watermelons : ℕ := 1
def sister_slices_per_melon : ℕ := 15

def cousin_watermelons : ℕ := 2
def cousin_slices_per_melon : ℕ := 8

def aunt_watermelons : ℕ := 4
def aunt_slices_per_melon : ℕ := 12

def grandfather_watermelons : ℕ := 1
def grandfather_slices_per_melon : ℕ := 6

-- Define the total number of slices
def total_slices : ℕ := 
  danny_watermelons * danny_slices_per_melon +
  sister_watermelons * sister_slices_per_melon +
  cousin_watermelons * cousin_slices_per_melon +
  aunt_watermelons * aunt_slices_per_melon +
  grandfather_watermelons * grandfather_slices_per_melon

-- Theorem statement
theorem total_watermelon_slices : total_slices = 115 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelon_slices_l626_62669


namespace NUMINAMATH_CALUDE_ivans_bird_feeder_feeds_21_l626_62615

/-- Calculates the number of birds fed weekly by a bird feeder --/
def birds_fed_weekly (feeder_capacity : ℝ) (birds_per_cup : ℝ) (stolen_amount : ℝ) : ℝ :=
  (feeder_capacity - stolen_amount) * birds_per_cup

/-- Theorem: Ivan's bird feeder feeds 21 birds weekly --/
theorem ivans_bird_feeder_feeds_21 :
  birds_fed_weekly 2 14 0.5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ivans_bird_feeder_feeds_21_l626_62615


namespace NUMINAMATH_CALUDE_green_team_score_l626_62697

theorem green_team_score (other_team_score lead : ℕ) (h1 : other_team_score = 68) (h2 : lead = 29) :
  ∃ G : ℕ, other_team_score = G + lead ∧ G = 39 := by
  sorry

end NUMINAMATH_CALUDE_green_team_score_l626_62697


namespace NUMINAMATH_CALUDE_rectangle_area_is_twelve_l626_62622

/-- Represents a rectangle with given properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ
  perimeter : ℝ
  length_eq : length = 3 * width
  perimeter_eq : perimeter = 2 * (length + width)
  diagonal_eq : diagonal^2 = width^2 + length^2

/-- The area of a rectangle with specific properties is 12 -/
theorem rectangle_area_is_twelve (rect : Rectangle) (h : rect.perimeter = 16) : 
  rect.width * rect.length = 12 := by
  sorry

#check rectangle_area_is_twelve

end NUMINAMATH_CALUDE_rectangle_area_is_twelve_l626_62622


namespace NUMINAMATH_CALUDE_quadratic_expression_l626_62657

/-- A quadratic function passing through the point (3, 10) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that f passes through (3, 10) -/
def passes_through (a b c : ℝ) : Prop := f a b c 3 = 10

theorem quadratic_expression (a b c : ℝ) (h : passes_through a b c) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_l626_62657


namespace NUMINAMATH_CALUDE_solution_set_transformation_l626_62618

theorem solution_set_transformation (k a b c : ℝ) :
  (∀ x, (x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 2 3) ↔ 
    (k * x / (a * x - 1) + (b * x - 1) / (c * x - 1) < 0)) →
  (∀ x, (x ∈ Set.Ioo (-1/2 : ℝ) (-1/3) ∪ Set.Ioo (1/2) 1) ↔ 
    (k / (x + a) + (x + b) / (x + c) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_transformation_l626_62618


namespace NUMINAMATH_CALUDE_broken_flagpole_tip_height_l626_62624

/-- Represents a broken flagpole -/
structure BrokenFlagpole where
  initial_height : ℝ
  break_height : ℝ
  folds_in_half : Bool

/-- Calculates the height of the tip of a broken flagpole from the ground -/
def tip_height (f : BrokenFlagpole) : ℝ :=
  if f.folds_in_half then f.break_height else f.initial_height

/-- Theorem stating that the height of the tip of a broken flagpole is equal to the break height -/
theorem broken_flagpole_tip_height 
  (f : BrokenFlagpole) 
  (h1 : f.initial_height = 12)
  (h2 : f.break_height = 7)
  (h3 : f.folds_in_half = true) :
  tip_height f = 7 := by
  sorry

end NUMINAMATH_CALUDE_broken_flagpole_tip_height_l626_62624
