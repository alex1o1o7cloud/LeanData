import Mathlib

namespace NUMINAMATH_CALUDE_plates_with_parents_is_eight_l3071_307132

/-- The number of plates used when Matt's parents join them -/
def plates_with_parents (total_plates : ℕ) (days_per_week : ℕ) (days_with_son : ℕ) (plates_per_person_with_son : ℕ) : ℕ :=
  (total_plates - days_with_son * plates_per_person_with_son * 2) / (days_per_week - days_with_son)

/-- Proof that the number of plates used when Matt's parents join them is 8 -/
theorem plates_with_parents_is_eight :
  plates_with_parents 38 7 3 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_plates_with_parents_is_eight_l3071_307132


namespace NUMINAMATH_CALUDE_calculate_expression_factorize_polynomial_l3071_307123

-- Part 1
theorem calculate_expression : (1 / 3)⁻¹ - Real.sqrt 16 + (-2016)^0 = 0 := by sorry

-- Part 2
theorem factorize_polynomial (x : ℝ) : 3 * x^2 - 6 * x + 3 = 3 * (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_factorize_polynomial_l3071_307123


namespace NUMINAMATH_CALUDE_chord_diameter_ratio_l3071_307174

/-- Given two concentric circles with radii R/2 and R, prove that if a chord of the larger circle
    is divided into three equal parts by the smaller circle, then the ratio of this chord to
    the diameter of the larger circle is 3√6/8. -/
theorem chord_diameter_ratio (R : ℝ) (h : R > 0) :
  ∃ (chord : ℝ), 
    (∃ (a : ℝ), chord = 3 * a ∧ 
      (∃ (x : ℝ), x^2 = 2 * a^2 ∧ x = R/2)) →
    chord / (2 * R) = 3 * Real.sqrt 6 / 8 := by
  sorry

end NUMINAMATH_CALUDE_chord_diameter_ratio_l3071_307174


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3071_307129

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 702 ∧ ∀ x, 3 * x^2 - 18 * x + 729 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3071_307129


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l3071_307185

/-- The line equation mx + y - m - 1 = 0 passes through the point (1, 1) for all real m -/
theorem fixed_point_of_line (m : ℝ) : m * 1 + 1 - m - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l3071_307185


namespace NUMINAMATH_CALUDE_beth_score_l3071_307144

/-- The score of a basketball game between two teams -/
structure BasketballScore where
  team1_player1 : ℕ  -- Beth's score
  team1_player2 : ℕ  -- Jan's score
  team2_player1 : ℕ  -- Judy's score
  team2_player2 : ℕ  -- Angel's score

/-- The conditions of the basketball game -/
def game_conditions (score : BasketballScore) : Prop :=
  score.team1_player2 = 10 ∧
  score.team2_player1 = 8 ∧
  score.team2_player2 = 11 ∧
  score.team1_player1 + score.team1_player2 = score.team2_player1 + score.team2_player2 + 3

/-- Theorem: Given the game conditions, Beth scored 12 points -/
theorem beth_score (score : BasketballScore) 
  (h : game_conditions score) : score.team1_player1 = 12 := by
  sorry


end NUMINAMATH_CALUDE_beth_score_l3071_307144


namespace NUMINAMATH_CALUDE_time_to_paint_remaining_rooms_l3071_307193

/-- Given a painting job with the following conditions:
  - There are 10 rooms in total to be painted
  - Each room takes 8 hours to paint
  - 8 rooms have already been painted
This theorem proves that it will take 16 hours to paint the remaining rooms. -/
theorem time_to_paint_remaining_rooms :
  let total_rooms : ℕ := 10
  let painted_rooms : ℕ := 8
  let time_per_room : ℕ := 8
  let remaining_rooms := total_rooms - painted_rooms
  let time_for_remaining := remaining_rooms * time_per_room
  time_for_remaining = 16 := by sorry

end NUMINAMATH_CALUDE_time_to_paint_remaining_rooms_l3071_307193


namespace NUMINAMATH_CALUDE_person_height_calculation_l3071_307131

/-- The height of a person used to determine the depth of water -/
def personHeight : ℝ := 6

/-- The depth of the water in feet -/
def waterDepth : ℝ := 60

/-- The relationship between the water depth and the person's height -/
def depthRelation : Prop := waterDepth = 10 * personHeight

theorem person_height_calculation : 
  depthRelation → personHeight = 6 := by sorry

end NUMINAMATH_CALUDE_person_height_calculation_l3071_307131


namespace NUMINAMATH_CALUDE_shape_C_has_two_lines_of_symmetry_l3071_307183

-- Define a type for shapes
inductive Shape : Type
  | A
  | B
  | C
  | D

-- Define a function to count lines of symmetry
def linesOfSymmetry : Shape → ℕ
  | Shape.A => 4
  | Shape.B => 0
  | Shape.C => 2
  | Shape.D => 1

-- Theorem statement
theorem shape_C_has_two_lines_of_symmetry :
  linesOfSymmetry Shape.C = 2 ∧
  ∀ s : Shape, s ≠ Shape.C → linesOfSymmetry s ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_shape_C_has_two_lines_of_symmetry_l3071_307183


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3071_307172

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 2 * x + 5 * y = 20) 
  (eq2 : 5 * x + 2 * y = 26) : 
  20 * x^2 + 60 * x * y + 50 * y^2 = 59600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3071_307172


namespace NUMINAMATH_CALUDE_sallys_earnings_l3071_307198

theorem sallys_earnings (first_month_earnings : ℝ) : 
  first_month_earnings + (first_month_earnings * 1.1) = 2100 → 
  first_month_earnings = 1000 := by
sorry

end NUMINAMATH_CALUDE_sallys_earnings_l3071_307198


namespace NUMINAMATH_CALUDE_base_sum_theorem_l3071_307171

/-- Represents a repeating decimal in a given base -/
def repeating_decimal (numerator denominator base : ℕ) : ℚ :=
  (numerator : ℚ) / ((base ^ 2 - 1) : ℚ)

theorem base_sum_theorem :
  ∃! (B₁ B₂ : ℕ), 
    B₁ > 1 ∧ B₂ > 1 ∧
    repeating_decimal 45 99 B₁ = repeating_decimal 3 9 B₂ ∧
    repeating_decimal 54 99 B₁ = repeating_decimal 6 9 B₂ ∧
    B₁ + B₂ = 20 := by sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l3071_307171


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l3071_307175

/-- Custom operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 - y

/-- Theorem stating that h ⊗ (h ⊗ h) = h for any real h -/
theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l3071_307175


namespace NUMINAMATH_CALUDE_vertex_below_x_axis_iff_k_less_than_4_l3071_307182

/-- A quadratic function of the form y = x^2 - 4x + k -/
def quadratic_function (x k : ℝ) : ℝ := x^2 - 4*x + k

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x : ℝ := 2

/-- The y-coordinate of the vertex of the quadratic function -/
def vertex_y (k : ℝ) : ℝ := quadratic_function vertex_x k

/-- The vertex is below the x-axis if its y-coordinate is negative -/
def vertex_below_x_axis (k : ℝ) : Prop := vertex_y k < 0

theorem vertex_below_x_axis_iff_k_less_than_4 :
  ∀ k : ℝ, vertex_below_x_axis k ↔ k < 4 := by sorry

end NUMINAMATH_CALUDE_vertex_below_x_axis_iff_k_less_than_4_l3071_307182


namespace NUMINAMATH_CALUDE_min_workers_theorem_l3071_307188

/-- Represents the company's profit scenario -/
structure CompanyProfit where
  maintenance_cost : ℕ
  worker_wage : ℕ
  production_rate : ℕ
  gadget_price : ℚ
  workday_length : ℕ

/-- Calculates the minimum number of workers required for profit -/
def min_workers_for_profit (c : CompanyProfit) : ℕ :=
  Nat.succ (Nat.ceil ((c.maintenance_cost : ℚ) / 
    (c.production_rate * c.workday_length * c.gadget_price - c.worker_wage * c.workday_length)))

/-- Theorem stating the minimum number of workers required for profit -/
theorem min_workers_theorem (c : CompanyProfit) 
  (h1 : c.maintenance_cost = 800)
  (h2 : c.worker_wage = 20)
  (h3 : c.production_rate = 6)
  (h4 : c.gadget_price = 9/2)
  (h5 : c.workday_length = 9) :
  min_workers_for_profit c = 13 := by
  sorry

#eval min_workers_for_profit { 
  maintenance_cost := 800, 
  worker_wage := 20, 
  production_rate := 6, 
  gadget_price := 9/2, 
  workday_length := 9 
}

end NUMINAMATH_CALUDE_min_workers_theorem_l3071_307188


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3071_307139

/-- If the solution set of the inequality -1/2x^2 + 2x > mx is {x | 0 < x < 2}, then m = 1 -/
theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (-1/2 * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3071_307139


namespace NUMINAMATH_CALUDE_power_equality_l3071_307178

theorem power_equality (m : ℕ) : 9^4 = 3^m → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3071_307178


namespace NUMINAMATH_CALUDE_expression_value_l3071_307157

theorem expression_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x + y) :
  x / y + y / x = 2 + 1 / (y^2 + y) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3071_307157


namespace NUMINAMATH_CALUDE_book_cost_proof_l3071_307124

/-- The original cost of a book before discount -/
def original_cost : ℝ := sorry

/-- The number of books bought -/
def num_books : ℕ := 10

/-- The discount per book -/
def discount_per_book : ℝ := 0.5

/-- The total amount paid -/
def total_paid : ℝ := 45

theorem book_cost_proof :
  original_cost = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_book_cost_proof_l3071_307124


namespace NUMINAMATH_CALUDE_quadratic_sum_l3071_307151

/-- A quadratic function with vertex at (-2, 5) and specific points -/
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (∀ x, g d e f x = d * (x + 2)^2 + 5) →  -- vertex form
  g d e f 0 = -1 →                       -- g(0) = -1
  g d e f 1 = -4 →                       -- g(1) = -4
  d + e + 3 * f = 14 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3071_307151


namespace NUMINAMATH_CALUDE_pet_groomer_problem_l3071_307158

theorem pet_groomer_problem (total_animals : ℕ) (cats : ℕ) (selected : ℕ) (prob : ℚ) :
  total_animals = 7 →
  cats = 2 →
  selected = 4 →
  prob = 2/7 →
  (Nat.choose cats cats * Nat.choose (total_animals - cats) (selected - cats)) / Nat.choose total_animals selected = prob →
  total_animals - cats = 5 := by
sorry

end NUMINAMATH_CALUDE_pet_groomer_problem_l3071_307158


namespace NUMINAMATH_CALUDE_bowling_balls_count_l3071_307112

theorem bowling_balls_count (red : ℕ) (green : ℕ) : 
  green = red + 6 →
  red + green = 66 →
  red = 30 := by
sorry

end NUMINAMATH_CALUDE_bowling_balls_count_l3071_307112


namespace NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l3071_307125

theorem fourth_power_nested_sqrt : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l3071_307125


namespace NUMINAMATH_CALUDE_both_sports_fans_l3071_307150

/-- Represents the number of students who like basketball -/
def basketball_fans : ℕ := 7

/-- Represents the number of students who like cricket -/
def cricket_fans : ℕ := 8

/-- Represents the number of students who like either basketball or cricket or both -/
def total_fans : ℕ := 10

/-- Theorem stating that the number of students who like both basketball and cricket is 5 -/
theorem both_sports_fans : 
  basketball_fans + cricket_fans - total_fans = 5 := by sorry

end NUMINAMATH_CALUDE_both_sports_fans_l3071_307150


namespace NUMINAMATH_CALUDE_outfits_count_l3071_307115

/-- The number of outfits with different colored shirts and hats -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
  (pants : ℕ) (red_hats green_hats blue_hats : ℕ) : ℕ :=
  (red_shirts * (green_hats + blue_hats) * pants) +
  (green_shirts * (red_hats + blue_hats) * pants) +
  (blue_shirts * (red_hats + green_hats) * pants)

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfits_count : 
  num_outfits 6 4 5 7 9 7 6 = 1526 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3071_307115


namespace NUMINAMATH_CALUDE_simplify_expression_l3071_307168

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x - 2) = 7*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3071_307168


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l3071_307104

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := ![x, 2, 0]
def b (x : ℝ) : Fin 3 → ℝ := ![3, 2 - x, x^2]

-- Define the dot product of two vectors
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define the condition for an obtuse angle
def is_obtuse_angle (v w : Fin 3 → ℝ) : Prop :=
  dot_product v w < 0

-- State the theorem
theorem obtuse_angle_range (x : ℝ) :
  is_obtuse_angle (a x) (b x) → x < -4 :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l3071_307104


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l3071_307140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + (1/2) * a * x^2 - x

theorem f_monotonicity_and_range :
  (∀ x > -1, ∀ y ∈ (Set.Ioo (-1 : ℝ) (-1/2) ∪ Set.Ioi 0), x < y → f 2 x < f 2 y) ∧
  (∀ x > -1, ∀ y ∈ Set.Ioo (-1/2 : ℝ) 0, x < y → f 2 x > f 2 y) ∧
  (∀ a : ℝ, (∀ x > 0, f a x ≥ a * x - x) ↔ 0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l3071_307140


namespace NUMINAMATH_CALUDE_water_requirement_proof_l3071_307107

/-- The number of households in the village -/
def num_households : ℕ := 10

/-- The total amount of water available in litres -/
def total_water : ℕ := 2000

/-- The number of months the water lasts -/
def num_months : ℕ := 10

/-- The number of litres of water required per household per month -/
def water_per_household_per_month : ℚ :=
  total_water / (num_households * num_months)

theorem water_requirement_proof :
  water_per_household_per_month = 20 := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_proof_l3071_307107


namespace NUMINAMATH_CALUDE_union_A_complement_B_l3071_307181

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Theorem statement
theorem union_A_complement_B : 
  A ∪ (U \ B) = Iic 1 ∪ Ioi 2 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l3071_307181


namespace NUMINAMATH_CALUDE_triangle_inequality_l3071_307166

theorem triangle_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3071_307166


namespace NUMINAMATH_CALUDE_ron_sold_twelve_tickets_l3071_307122

/-- Represents the ticket sales problem with Ron and Kathy --/
structure TicketSales where
  ron_price : ℝ
  kathy_price : ℝ
  total_tickets : ℕ
  total_income : ℝ

/-- Theorem stating that Ron sold 12 tickets given the problem conditions --/
theorem ron_sold_twelve_tickets (ts : TicketSales) 
  (h1 : ts.ron_price = 2)
  (h2 : ts.kathy_price = 4.5)
  (h3 : ts.total_tickets = 20)
  (h4 : ts.total_income = 60) : 
  ∃ (ron_tickets : ℕ) (kathy_tickets : ℕ), 
    ron_tickets + kathy_tickets = ts.total_tickets ∧ 
    ron_tickets * ts.ron_price + kathy_tickets * ts.kathy_price = ts.total_income ∧
    ron_tickets = 12 := by
  sorry

end NUMINAMATH_CALUDE_ron_sold_twelve_tickets_l3071_307122


namespace NUMINAMATH_CALUDE_fraction_ordering_l3071_307113

theorem fraction_ordering : 
  (5 : ℚ) / 19 < 7 / 21 ∧ 7 / 21 < 9 / 23 := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3071_307113


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l3071_307148

/-- Given that x and y are real numbers satisfying 3x² + 4y² = 48,
    the maximum value of √(x² + y² - 4x + 4) + √(x² + y² - 2x + 4y + 5) is 8 + √13 -/
theorem max_value_sum_of_roots (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 48) :
  (∃ (m : ℝ), ∀ (a b : ℝ), 3 * a^2 + 4 * b^2 = 48 →
    Real.sqrt (a^2 + b^2 - 4*a + 4) + Real.sqrt (a^2 + b^2 - 2*a + 4*b + 5) ≤ m) ∧
  (Real.sqrt (x^2 + y^2 - 4*x + 4) + Real.sqrt (x^2 + y^2 - 2*x + 4*y + 5) ≤ 8 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l3071_307148


namespace NUMINAMATH_CALUDE_gcd_b_n_b_n_plus_one_is_one_l3071_307143

def b (n : ℕ) : ℚ := (15^n - 1) / 14

theorem gcd_b_n_b_n_plus_one_is_one (n : ℕ) : 
  Nat.gcd (Nat.floor (b n)) (Nat.floor (b (n + 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_b_n_b_n_plus_one_is_one_l3071_307143


namespace NUMINAMATH_CALUDE_complex_power_sum_l3071_307120

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1/(z^100) = -2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3071_307120


namespace NUMINAMATH_CALUDE_two_digit_sum_problem_l3071_307173

/-- Given two-digit numbers ab and cd, and a three-digit number jjj,
    where a, b, c, and d are distinct positive integers,
    c = 9, and ab + cd = jjj, prove that cd = 98. -/
theorem two_digit_sum_problem (ab cd jjj : ℕ) (a b c d : ℕ) : 
  (10 ≤ ab) ∧ (ab < 100) →  -- ab is a two-digit number
  (10 ≤ cd) ∧ (cd < 100) →  -- cd is a two-digit number
  (100 ≤ jjj) ∧ (jjj < 1000) →  -- jjj is a three-digit number
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- a, b, c, d are distinct
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →  -- a, b, c, d are positive
  c = 9 →  -- given condition
  ab + cd = jjj →  -- sum equation
  cd = 98 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_problem_l3071_307173


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3071_307197

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : a * b + a * c + b * c = 7) 
  (h3 : a * b * c = -10) : 
  a^3 + b^3 + c^3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3071_307197


namespace NUMINAMATH_CALUDE_waynes_age_l3071_307194

theorem waynes_age (birth_year_julia : ℕ) (current_year : ℕ) : 
  birth_year_julia = 1979 → current_year = 2021 →
  ∃ (age_wayne age_peter age_julia : ℕ),
    age_julia = current_year - birth_year_julia ∧
    age_peter = age_julia - 2 ∧
    age_wayne = age_peter - 3 ∧
    age_wayne = 37 :=
by sorry

end NUMINAMATH_CALUDE_waynes_age_l3071_307194


namespace NUMINAMATH_CALUDE_task_completion_time_l3071_307128

theorem task_completion_time (a b c : ℝ) 
  (h1 : 1/a + 1/b = 1/2)
  (h2 : 1/b + 1/c = 1/4)
  (h3 : 1/c + 1/a = 5/12) :
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_task_completion_time_l3071_307128


namespace NUMINAMATH_CALUDE_paving_cost_example_l3071_307167

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with length 5.5 m and width 3.75 m
    at a rate of $600 per square metre is $12,375. -/
theorem paving_cost_example : paving_cost 5.5 3.75 600 = 12375 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_example_l3071_307167


namespace NUMINAMATH_CALUDE_hat_price_reduction_l3071_307114

theorem hat_price_reduction (original_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  original_price = 12 ∧ first_reduction = 0.2 ∧ second_reduction = 0.25 →
  original_price * (1 - first_reduction) * (1 - second_reduction) = 7.2 := by
sorry

end NUMINAMATH_CALUDE_hat_price_reduction_l3071_307114


namespace NUMINAMATH_CALUDE_marble_fraction_l3071_307109

theorem marble_fraction (total : ℝ) (h : total > 0) : 
  let initial_blue := (2/3) * total
  let initial_red := (1/3) * total
  let new_blue := 3 * initial_blue
  let new_total := new_blue + initial_red
  initial_red / new_total = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_marble_fraction_l3071_307109


namespace NUMINAMATH_CALUDE_complex_fraction_eval_l3071_307135

theorem complex_fraction_eval (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 + c*d + d^2 = 0) : 
  (c^12 + d^12) / (c^3 + d^3)^4 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_eval_l3071_307135


namespace NUMINAMATH_CALUDE_tickets_bought_l3071_307110

theorem tickets_bought (ticket_cost : ℕ) (total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 := by
sorry

end NUMINAMATH_CALUDE_tickets_bought_l3071_307110


namespace NUMINAMATH_CALUDE_simplify_fraction_l3071_307163

theorem simplify_fraction : (150 : ℚ) / 450 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3071_307163


namespace NUMINAMATH_CALUDE_average_weight_problem_l3071_307186

theorem average_weight_problem (total_boys : ℕ) (group1_boys : ℕ) (group2_boys : ℕ) 
  (group2_avg_weight : ℚ) (total_avg_weight : ℚ) :
  total_boys = group1_boys + group2_boys →
  total_boys = 24 →
  group1_boys = 16 →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.55 →
  (group1_boys * (50.25 : ℚ) + group2_boys * group2_avg_weight) / total_boys = total_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3071_307186


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l3071_307117

theorem sqrt_2x_minus_1_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 1) → x ≥ (1 / 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l3071_307117


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3071_307153

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := (a * (1 - r^n)) / (1 - r)
  let a := 1 / 5
  let r := -1 / 5
  let n := 5
  series_sum = 521 / 3125 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3071_307153


namespace NUMINAMATH_CALUDE_board_number_generation_l3071_307196

theorem board_number_generation (target : ℕ := 2020) : ∃ a b : ℕ, 20 * a + 21 * b = target := by
  sorry

end NUMINAMATH_CALUDE_board_number_generation_l3071_307196


namespace NUMINAMATH_CALUDE_sin_cos_difference_74_14_l3071_307106

theorem sin_cos_difference_74_14 :
  Real.sin (74 * π / 180) * Real.cos (14 * π / 180) -
  Real.cos (74 * π / 180) * Real.sin (14 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_74_14_l3071_307106


namespace NUMINAMATH_CALUDE_cube_of_neg_cube_l3071_307133

theorem cube_of_neg_cube (x : ℝ) : (-x^3)^3 = -x^9 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_neg_cube_l3071_307133


namespace NUMINAMATH_CALUDE_point_not_on_line_l3071_307105

/-- Given m > 2 and mb > 0, prove that (0, -2023) cannot lie on y = mx + b -/
theorem point_not_on_line (m b : ℝ) (hm : m > 2) (hmb : m * b > 0) :
  ¬ (∃ (x y : ℝ), x = 0 ∧ y = -2023 ∧ y = m * x + b) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l3071_307105


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3071_307142

theorem min_value_of_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (2/x) + (1/y) ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ (2/x₀) + (1/y₀) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3071_307142


namespace NUMINAMATH_CALUDE_car_profit_percent_l3071_307152

/-- Calculates the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℕ) 
  (mechanical_repairs : ℕ) 
  (bodywork : ℕ) 
  (interior_refurbishment : ℕ) 
  (taxes_and_fees : ℕ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 48000)
  (h2 : mechanical_repairs = 6000)
  (h3 : bodywork = 4000)
  (h4 : interior_refurbishment = 3000)
  (h5 : taxes_and_fees = 2000)
  (h6 : selling_price = 72900) :
  ∃ (profit_percent : ℚ), 
    abs (profit_percent - 15.71) < 0.01 ∧ 
    profit_percent = (selling_price - (purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees)) / 
                     (purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees) * 100 := by
  sorry


end NUMINAMATH_CALUDE_car_profit_percent_l3071_307152


namespace NUMINAMATH_CALUDE_fencing_cost_l3071_307156

/-- Calculate the total cost of fencing a rectangular plot -/
theorem fencing_cost (length breadth perimeter cost_per_metre : ℝ) : 
  length = 200 →
  length = breadth + 20 →
  cost_per_metre = 26.5 →
  perimeter = 2 * (length + breadth) →
  perimeter * cost_per_metre = 20140 := by
  sorry

#check fencing_cost

end NUMINAMATH_CALUDE_fencing_cost_l3071_307156


namespace NUMINAMATH_CALUDE_vector_magnitude_l3071_307169

theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (Real.cos (5 * π / 180), Real.sin (5 * π / 180)) →
  b = (Real.cos (65 * π / 180), Real.sin (65 * π / 180)) →
  Real.sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3071_307169


namespace NUMINAMATH_CALUDE_max_product_with_sum_and_diff_l3071_307159

/-- Given two real numbers with a difference of 4 and a sum of 35, 
    their product is maximized when the numbers are 19.5 and 15.5 -/
theorem max_product_with_sum_and_diff (x y : ℝ) : 
  x - y = 4 → x + y = 35 → x * y ≤ 19.5 * 15.5 :=
by sorry

end NUMINAMATH_CALUDE_max_product_with_sum_and_diff_l3071_307159


namespace NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_3x_eq_5x_l3071_307119

theorem smallest_positive_solution_sqrt_3x_eq_5x :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x ∧
  ∀ (y : ℝ), y > 0 → Real.sqrt (3 * y) = 5 * y → x ≤ y ∧
  x = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_3x_eq_5x_l3071_307119


namespace NUMINAMATH_CALUDE_log_product_l3071_307176

theorem log_product (x y : ℝ) (h1 : Real.log (x / 2) = 0.5) (h2 : Real.log (y / 5) = 0.1) :
  Real.log (x * y) = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_log_product_l3071_307176


namespace NUMINAMATH_CALUDE_unique_sum_of_four_smallest_divisor_squares_l3071_307118

def is_sum_of_four_smallest_divisor_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ d ∣ n ∧
    (∀ x : ℕ, x ∣ n → x ≤ d) ∧
    n = a^2 + b^2 + c^2 + d^2

theorem unique_sum_of_four_smallest_divisor_squares : 
  ∀ n : ℕ, is_sum_of_four_smallest_divisor_squares n ↔ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_of_four_smallest_divisor_squares_l3071_307118


namespace NUMINAMATH_CALUDE_max_a_fourth_quadrant_l3071_307111

theorem max_a_fourth_quadrant (a : ℤ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  (z.re > 0 ∧ z.im < 0) → a ≤ 3 ∧ ∃ (a : ℤ), a = 3 ∧ 
    let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
    (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_max_a_fourth_quadrant_l3071_307111


namespace NUMINAMATH_CALUDE_power_function_properties_l3071_307191

-- Define the power function f(x) = x^α
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 3 / Real.log 9)

-- Theorem statement
theorem power_function_properties :
  -- The function passes through (9,3)
  f 9 = 3 ∧
  -- f(x) is increasing on its domain
  (∀ x y, x < y → x > 0 → y > 0 → f x < f y) ∧
  -- When x ≥ 4, f(x) ≥ 2
  (∀ x, x ≥ 4 → f x ≥ 2) ∧
  -- When x₂ > x₁ > 0, (f(x₁) + f(x₂))/2 < f((x₁ + x₂)/2)
  (∀ x₁ x₂, x₂ > x₁ → x₁ > 0 → (f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_power_function_properties_l3071_307191


namespace NUMINAMATH_CALUDE_max_moves_less_than_500000_l3071_307136

/-- Represents the maximum number of moves for a given number of cards. -/
def max_moves (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the maximum number of moves for 1000 cards is less than 500,000. -/
theorem max_moves_less_than_500000 :
  max_moves 1000 < 500000 := by
  sorry

#eval max_moves 1000  -- This will evaluate to 499500

end NUMINAMATH_CALUDE_max_moves_less_than_500000_l3071_307136


namespace NUMINAMATH_CALUDE_symmetric_circle_l3071_307141

/-- The equation of a circle symmetric to x^2 + y^2 = 4 with respect to the line x + y - 1 = 0 -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x + y - 1 = 0 → (x-1)^2 + (y-1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l3071_307141


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l3071_307195

theorem existence_of_special_numbers :
  ∃ (S : Finset ℕ), Finset.card S = 100 ∧
  ∀ (a b c d e : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
  (a * b * c * d * e) % (a + b + c + d + e) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l3071_307195


namespace NUMINAMATH_CALUDE_sock_cost_is_three_l3071_307137

/-- The cost of a uniform given the cost of socks -/
def uniform_cost (sock_cost : ℚ) : ℚ :=
  20 + 2 * 20 + (2 * 20) / 5 + sock_cost

/-- The total cost of 5 uniforms given the cost of socks -/
def total_cost (sock_cost : ℚ) : ℚ :=
  5 * uniform_cost sock_cost

theorem sock_cost_is_three :
  ∃ (sock_cost : ℚ), total_cost sock_cost = 355 ∧ sock_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_sock_cost_is_three_l3071_307137


namespace NUMINAMATH_CALUDE_first_discount_calculation_l3071_307199

theorem first_discount_calculation (original_price final_price second_discount : ℝ) 
  (h1 : original_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5)
  : ∃ first_discount : ℝ, 
    first_discount = 20 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_calculation_l3071_307199


namespace NUMINAMATH_CALUDE_debby_bottles_per_day_l3071_307154

/-- The number of bottles Debby bought -/
def total_bottles : ℕ := 8066

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The number of bottles Debby drank per day -/
def bottles_per_day : ℕ := total_bottles / days_lasted

theorem debby_bottles_per_day :
  bottles_per_day = 109 := by sorry

end NUMINAMATH_CALUDE_debby_bottles_per_day_l3071_307154


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l3071_307116

theorem negative_fractions_comparison : -2/3 < -1/2 := by sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l3071_307116


namespace NUMINAMATH_CALUDE_firm_partners_count_l3071_307127

theorem firm_partners_count (partners associates : ℕ) : 
  partners / associates = 2 / 63 →
  partners / (associates + 45) = 1 / 34 →
  partners = 18 := by
sorry

end NUMINAMATH_CALUDE_firm_partners_count_l3071_307127


namespace NUMINAMATH_CALUDE_find_a_and_b_l3071_307189

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = {x | 0 < x ∧ x ≤ 2}) ∧
    (A ∪ B a b = {x | x > -2}) ∧
    a = -1 ∧
    b = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l3071_307189


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3071_307149

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 3| > 1} = Set.Iio 1 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3071_307149


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l3071_307184

theorem nesbitt_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l3071_307184


namespace NUMINAMATH_CALUDE_unique_solution_l3071_307192

/-- The functional equation that f must satisfy for all real x and y -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)

/-- The theorem stating that the only function satisfying the equation is f(x) = x + 2 -/
theorem unique_solution (f : ℝ → ℝ) (h : functional_equation f) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3071_307192


namespace NUMINAMATH_CALUDE_simplify_expression_l3071_307161

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3071_307161


namespace NUMINAMATH_CALUDE_f_composition_neg_two_l3071_307108

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- State the theorem
theorem f_composition_neg_two : f (f (-2)) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_composition_neg_two_l3071_307108


namespace NUMINAMATH_CALUDE_probability_all_white_is_correct_l3071_307190

def total_balls : ℕ := 18
def white_balls : ℕ := 8
def black_balls : ℕ := 10
def drawn_balls : ℕ := 7

def probability_all_white : ℚ :=
  (Nat.choose white_balls drawn_balls) / (Nat.choose total_balls drawn_balls)

theorem probability_all_white_is_correct :
  probability_all_white = 1 / 3980 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_white_is_correct_l3071_307190


namespace NUMINAMATH_CALUDE_blue_eyed_students_l3071_307101

theorem blue_eyed_students (total_students : ℕ) (blond_blue : ℕ) (neither : ℕ) :
  total_students = 30 →
  blond_blue = 6 →
  neither = 3 →
  ∃ (blue_eyes : ℕ),
    blue_eyes = 11 ∧
    2 * blue_eyes + (blue_eyes - blond_blue) + neither = total_students :=
by sorry

end NUMINAMATH_CALUDE_blue_eyed_students_l3071_307101


namespace NUMINAMATH_CALUDE_undefined_fraction_min_x_l3071_307177

theorem undefined_fraction_min_x : 
  let f (x : ℝ) := (x - 3) / (6 * x^2 - 37 * x + 6)
  ∀ y < 1/6, ∃ ε > 0, ∀ x ∈ Set.Ioo (y - ε) (y + ε), f x ≠ 0⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_undefined_fraction_min_x_l3071_307177


namespace NUMINAMATH_CALUDE_smallest_difference_PR_QR_l3071_307170

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  PQ : ℕ
  QR : ℕ
  PR : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.PQ + t.QR > t.PR ∧ t.PQ + t.PR > t.QR ∧ t.QR + t.PR > t.PQ

/-- Represents the conditions of the problem -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.PQ + t.QR + t.PR = 2023 ∧
  t.PQ ≤ t.QR ∧ t.QR < t.PR ∧
  is_valid_triangle t

/-- The main theorem stating the smallest possible difference between PR and QR -/
theorem smallest_difference_PR_QR :
  ∃ (t : Triangle), satisfies_conditions t ∧
  ∀ (t' : Triangle), satisfies_conditions t' → t.PR - t.QR ≤ t'.PR - t'.QR ∧
  t.PR - t.QR = 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_PR_QR_l3071_307170


namespace NUMINAMATH_CALUDE_system_solution_unique_l3071_307102

theorem system_solution_unique :
  ∃! (x y : ℝ), 
    3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6 ∧
    x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7 ∧
    x = 1/2 ∧ y = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3071_307102


namespace NUMINAMATH_CALUDE_inscribed_circle_tangency_angles_l3071_307165

/-- A rhombus with an inscribed circle -/
structure RhombusWithInscribedCircle where
  /-- The measure of the acute angle of the rhombus in degrees -/
  acute_angle : ℝ
  /-- The assumption that the acute angle is 37 degrees -/
  acute_angle_is_37 : acute_angle = 37

/-- The angles formed by the points of tangency on the inscribed circle -/
def tangency_angles (r : RhombusWithInscribedCircle) : List ℝ :=
  [180 - r.acute_angle, r.acute_angle, 180 - r.acute_angle, r.acute_angle]

/-- Theorem stating the angles formed by the points of tangency -/
theorem inscribed_circle_tangency_angles (r : RhombusWithInscribedCircle) :
  tangency_angles r = [143, 37, 143, 37] := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangency_angles_l3071_307165


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l3071_307138

theorem number_of_divisors_180 : Finset.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l3071_307138


namespace NUMINAMATH_CALUDE_count_valid_m_l3071_307134

theorem count_valid_m : ∃! (s : Finset ℕ), 
  (∀ m ∈ s, m > 0 ∧ (2520 : ℤ) % (m^2 - 2) = 0) ∧
  (∀ m : ℕ, m > 0 ∧ (2520 : ℤ) % (m^2 - 2) = 0 → m ∈ s) ∧
  s.card = 5 := by sorry

end NUMINAMATH_CALUDE_count_valid_m_l3071_307134


namespace NUMINAMATH_CALUDE_costume_cost_theorem_l3071_307147

/-- Calculates the total cost of materials for a costume --/
def costume_cost (skirt_length : ℝ) (skirt_width : ℝ) (num_skirts : ℕ) 
                 (skirt_cost_per_sqft : ℝ) (bodice_shirt_area : ℝ) 
                 (bodice_sleeve_area : ℝ) (bodice_cost_per_sqft : ℝ)
                 (bonnet_length : ℝ) (bonnet_width : ℝ) (bonnet_cost_per_sqft : ℝ)
                 (shoe_cover_length : ℝ) (shoe_cover_width : ℝ) 
                 (num_shoe_covers : ℕ) (shoe_cover_cost_per_sqft : ℝ) : ℝ :=
  let skirt_total_area := skirt_length * skirt_width * num_skirts
  let skirt_cost := skirt_total_area * skirt_cost_per_sqft
  let bodice_total_area := bodice_shirt_area + 2 * bodice_sleeve_area
  let bodice_cost := bodice_total_area * bodice_cost_per_sqft
  let bonnet_area := bonnet_length * bonnet_width
  let bonnet_cost := bonnet_area * bonnet_cost_per_sqft
  let shoe_cover_total_area := shoe_cover_length * shoe_cover_width * num_shoe_covers
  let shoe_cover_cost := shoe_cover_total_area * shoe_cover_cost_per_sqft
  skirt_cost + bodice_cost + bonnet_cost + shoe_cover_cost

/-- The total cost of materials for the costume is $479.63 --/
theorem costume_cost_theorem : 
  costume_cost 12 4 3 3 2 5 2.5 2.5 1.5 1.5 1 1.5 2 4 = 479.63 := by
  sorry

end NUMINAMATH_CALUDE_costume_cost_theorem_l3071_307147


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l3071_307100

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

/-- A number is a perfect cube if it's equal to some integer cubed. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 3

/-- The smallest positive integer n such that 3n is a perfect square and 2n is a perfect cube is 108. -/
theorem smallest_n_square_cube : (
  ∀ n : ℕ, 
  n > 0 ∧ 
  IsPerfectSquare (3 * n) ∧ 
  IsPerfectCube (2 * n) → 
  n ≥ 108
) ∧ 
IsPerfectSquare (3 * 108) ∧ 
IsPerfectCube (2 * 108) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l3071_307100


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3071_307145

def toy_car_price : ℕ := 11
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7

def total_spent : ℕ := 2 * toy_car_price + scarf_price + beanie_price

theorem initial_money_calculation :
  total_spent + remaining_money = 53 := by sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3071_307145


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3071_307180

theorem geometric_sequence_product (b : ℕ → ℝ) (q : ℝ) :
  (∀ n, b (n + 1) = q * b n) →
  ∀ n, (b n * b (n + 1) * b (n + 2)) * q^3 = (b (n + 1) * b (n + 2) * b (n + 3)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3071_307180


namespace NUMINAMATH_CALUDE_permutation_and_combination_problem_l3071_307126

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem permutation_and_combination_problem :
  ∃ (x : ℕ), x > 0 ∧ 7 * A 6 x = 20 * A 7 (x - 1) ∧ 
  x = 3 ∧
  C 20 (20 - x) + C (17 + x) (x - 1) = 1330 :=
sorry

end NUMINAMATH_CALUDE_permutation_and_combination_problem_l3071_307126


namespace NUMINAMATH_CALUDE_sisters_height_l3071_307187

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Converts inches to feet and remaining inches -/
def inches_to_feet_and_inches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_valid : inches < 12

theorem sisters_height 
  (sunflower_height_feet : ℕ)
  (height_difference_inches : ℕ)
  (h_sunflower : sunflower_height_feet = 6)
  (h_difference : height_difference_inches = 21) :
  let sunflower_height_inches := feet_to_inches sunflower_height_feet
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let (sister_feet, sister_inches) := inches_to_feet_and_inches sister_height_inches
  Height.mk sister_feet sister_inches (by sorry) = Height.mk 4 3 (by sorry) :=
by sorry

end NUMINAMATH_CALUDE_sisters_height_l3071_307187


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3071_307162

theorem polynomial_division_quotient :
  let dividend := 5 * X^5 - 3 * X^4 + 6 * X^3 - 8 * X^2 + 9 * X - 4
  let divisor := 4 * X^2 + 5 * X + 3
  let quotient := 5/4 * X^3 - 47/16 * X^2 + 257/64 * X - 1547/256
  dividend = divisor * quotient + (dividend % divisor) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3071_307162


namespace NUMINAMATH_CALUDE_simplify_fraction_l3071_307130

theorem simplify_fraction : 
  1 / (1 / ((1/2)^0) + 1 / ((1/2)^1) + 1 / ((1/2)^2) + 1 / ((1/2)^3) + 1 / ((1/2)^4)) = 1 / 31 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3071_307130


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l3071_307179

theorem square_difference_of_integers (x y : ℕ) 
  (h1 : x + y = 40) 
  (h2 : x - y = 14) : 
  x^2 - y^2 = 560 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l3071_307179


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3071_307155

/-- 
A rectangular plot has an area that is 20 times its breadth,
and its length is 10 meters more than its breadth.
This theorem proves that the breadth of such a plot is 10 meters.
-/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
  area = 20 * breadth →
  length = breadth + 10 →
  area = length * breadth →
  breadth = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3071_307155


namespace NUMINAMATH_CALUDE_interest_rate_first_part_l3071_307121

/-- Given a total amount of 3200, divided into two parts where the first part is 800
    and the second part is at 5% interest rate, and the total annual interest is 144,
    prove that the interest rate of the first part is 3%. -/
theorem interest_rate_first_part (total : ℕ) (first_part : ℕ) (second_part : ℕ) 
  (second_rate : ℚ) (total_interest : ℕ) :
  total = 3200 →
  first_part = 800 →
  second_part = total - first_part →
  second_rate = 5 / 100 →
  total_interest = 144 →
  ∃ (first_rate : ℚ), 
    first_rate * first_part / 100 + second_rate * second_part = total_interest ∧
    first_rate = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_first_part_l3071_307121


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3071_307146

theorem modulus_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.abs ((3 - 4 * i) / i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3071_307146


namespace NUMINAMATH_CALUDE_cantaloupes_left_l3071_307164

/-- Represents the number of melons and their prices --/
structure MelonSales where
  cantaloupe_price : ℕ
  honeydew_price : ℕ
  initial_cantaloupes : ℕ
  initial_honeydews : ℕ
  dropped_cantaloupes : ℕ
  rotten_honeydews : ℕ
  remaining_honeydews : ℕ
  total_revenue : ℕ

/-- Theorem stating the number of cantaloupes left at the end of the day --/
theorem cantaloupes_left (s : MelonSales)
    (h1 : s.cantaloupe_price = 2)
    (h2 : s.honeydew_price = 3)
    (h3 : s.initial_cantaloupes = 30)
    (h4 : s.initial_honeydews = 27)
    (h5 : s.dropped_cantaloupes = 2)
    (h6 : s.rotten_honeydews = 3)
    (h7 : s.remaining_honeydews = 9)
    (h8 : s.total_revenue = 85) :
    s.initial_cantaloupes - s.dropped_cantaloupes -
    ((s.total_revenue - (s.honeydew_price * (s.initial_honeydews - s.rotten_honeydews - s.remaining_honeydews))) / s.cantaloupe_price) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_left_l3071_307164


namespace NUMINAMATH_CALUDE_grazing_problem_solution_l3071_307103

/-- Represents the grazing scenario with oxen and rent -/
structure GrazingScenario where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℚ
  c_rent : ℚ

/-- Calculates the total oxen-months for a given scenario -/
def total_oxen_months (s : GrazingScenario) : ℕ :=
  s.a_oxen * s.a_months + s.b_oxen * s.b_months + s.c_oxen * s.c_months

/-- Theorem stating the solution to the grazing problem -/
theorem grazing_problem_solution (s : GrazingScenario) 
  (h1 : s.a_oxen = 10)
  (h2 : s.a_months = 7)
  (h3 : s.b_oxen = 12)
  (h4 : s.c_oxen = 15)
  (h5 : s.c_months = 3)
  (h6 : s.total_rent = 245)
  (h7 : s.c_rent = 62.99999999999999)
  : s.b_months = 5 := by
  sorry


end NUMINAMATH_CALUDE_grazing_problem_solution_l3071_307103


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3071_307160

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The difference between the probability of 4 heads in 5 flips and 5 heads in 5 flips -/
def prob_difference : ℚ :=
  prob_k_heads 5 4 - prob_k_heads 5 5

theorem coin_flip_probability : prob_difference = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3071_307160
