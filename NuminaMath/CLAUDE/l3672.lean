import Mathlib

namespace NUMINAMATH_CALUDE_select_questions_theorem_l3672_367255

/-- The number of ways to select 3 questions from a set of questions with the given conditions -/
def select_questions (multiple_choice : ℕ) (fill_in_blank : ℕ) (open_ended : ℕ) : ℕ :=
  let total_questions := multiple_choice + fill_in_blank + open_ended
  let one_each := Nat.choose multiple_choice 1 * Nat.choose fill_in_blank 1 * Nat.choose open_ended 1
  let two_multiple_one_open := Nat.choose multiple_choice 2 * Nat.choose open_ended 1
  let one_multiple_two_open := Nat.choose multiple_choice 1 * Nat.choose open_ended 2
  one_each + two_multiple_one_open + one_multiple_two_open

theorem select_questions_theorem :
  select_questions 12 4 6 = 864 := by
  sorry

end NUMINAMATH_CALUDE_select_questions_theorem_l3672_367255


namespace NUMINAMATH_CALUDE_no_solution_equation_one_unique_solution_equation_two_l3672_367232

-- Problem 1
theorem no_solution_equation_one (x : ℝ) : 
  (x ≠ 2) → (1 / (x - 2) ≠ (1 - x) / (2 - x) - 3) :=
by sorry

-- Problem 2
theorem unique_solution_equation_two :
  ∃! x : ℝ, (x ≠ 1) ∧ (x^2 ≠ 1) ∧ (x / (x - 1) - (2*x - 1) / (x^2 - 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_one_unique_solution_equation_two_l3672_367232


namespace NUMINAMATH_CALUDE_complete_factorization_l3672_367207

theorem complete_factorization (x : ℝ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3*x^2 + 9) * (x^3 - 3) * (x^3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l3672_367207


namespace NUMINAMATH_CALUDE_unique_fish_count_l3672_367204

/-- The number of unique fish owned by four friends given specific conditions -/
theorem unique_fish_count :
  let micah_fish : ℕ := 7
  let kenneth_fish : ℕ := 3 * micah_fish
  let matthias_fish : ℕ := kenneth_fish - 15
  let gabrielle_fish : ℕ := 2 * (micah_fish + kenneth_fish + matthias_fish)
  let micah_matthias_shared : ℕ := 4
  let kenneth_gabrielle_shared : ℕ := 6
  (micah_fish + kenneth_fish + matthias_fish + gabrielle_fish) - 
  (micah_matthias_shared + kenneth_gabrielle_shared) = 92 :=
by sorry

end NUMINAMATH_CALUDE_unique_fish_count_l3672_367204


namespace NUMINAMATH_CALUDE_mathematics_players_count_l3672_367239

-- Define the set of all players
def TotalPlayers : ℕ := 30

-- Define the set of players taking physics
def PhysicsPlayers : ℕ := 15

-- Define the set of players taking both physics and mathematics
def BothSubjectsPlayers : ℕ := 7

-- Define the set of players taking mathematics
def MathematicsPlayers : ℕ := TotalPlayers - (PhysicsPlayers - BothSubjectsPlayers)

-- Theorem statement
theorem mathematics_players_count : MathematicsPlayers = 22 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_players_count_l3672_367239


namespace NUMINAMATH_CALUDE_prob_vertical_side_from_start_l3672_367293

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of jumping in each direction -/
def jump_prob : Fin 4 → ℝ
| 0 => 0.3  -- up
| 1 => 0.3  -- down
| 2 => 0.2  -- left
| 3 => 0.2  -- right

/-- The dimensions of the grid -/
def grid_size : ℕ := 6

/-- The starting point of the frog -/
def start : Point := ⟨2, 3⟩

/-- Predicate to check if a point is on the vertical side of the grid -/
def on_vertical_side (p : Point) : Prop :=
  p.x = 0 ∨ p.x = grid_size

/-- The probability of reaching a vertical side first from a given point -/
noncomputable def prob_vertical_side (p : Point) : ℝ := sorry

/-- The main theorem: probability of reaching a vertical side first from the starting point -/
theorem prob_vertical_side_from_start :
  prob_vertical_side start = 5/8 := by sorry

end NUMINAMATH_CALUDE_prob_vertical_side_from_start_l3672_367293


namespace NUMINAMATH_CALUDE_inequality_solution_and_minimum_l3672_367200

-- Define the solution set
def solution_set (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- Define the inequality
def inequality (x m n : ℝ) : Prop := |x - m| ≤ n

-- Define the constraint on a and b
def constraint (a b m n : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a + b = m / a + n / b

theorem inequality_solution_and_minimum (m n : ℝ) :
  (∀ x, inequality x m n ↔ solution_set x) →
  (m = 2 ∧ n = 2) ∧
  (∀ a b, constraint a b m n → a + b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b, constraint a b m n ∧ a + b = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_and_minimum_l3672_367200


namespace NUMINAMATH_CALUDE_math_books_count_l3672_367270

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 80 ∧ 
  math_cost = 4 ∧ 
  history_cost = 5 ∧ 
  total_price = 368 →
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 32 :=
by sorry

end NUMINAMATH_CALUDE_math_books_count_l3672_367270


namespace NUMINAMATH_CALUDE_bike_fundraising_days_l3672_367221

/-- The number of days required to raise money for a bike by selling bracelets -/
def days_to_raise_money (bike_cost : ℕ) (bracelet_price : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  bike_cost / (bracelet_price * bracelets_per_day)

/-- Theorem: Given the specific costs and sales plan, it takes 14 days to raise money for the bike -/
theorem bike_fundraising_days :
  days_to_raise_money 112 1 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bike_fundraising_days_l3672_367221


namespace NUMINAMATH_CALUDE_model_a_better_fit_l3672_367274

/-- Represents a regression model --/
structure RegressionModel where
  rsquare : ℝ
  (rsquare_nonneg : 0 ≤ rsquare)
  (rsquare_le_one : rsquare ≤ 1)

/-- Defines when one model has a better fit than another --/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.rsquare > model2.rsquare

/-- Theorem stating that model A has a better fit than model B --/
theorem model_a_better_fit (model_a model_b : RegressionModel)
  (ha : model_a.rsquare = 0.98)
  (hb : model_b.rsquare = 0.80) :
  better_fit model_a model_b :=
sorry

end NUMINAMATH_CALUDE_model_a_better_fit_l3672_367274


namespace NUMINAMATH_CALUDE_find_m_value_l3672_367224

/-- Given two functions f and g, prove that m = -7 when f(5) - g(5) = 55 -/
theorem find_m_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + 3 * x + 7
  let g : ℝ → ℝ := λ x => 2 * x^2 - m * x + 1
  (f 5 - g 5 = 55) → m = -7 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l3672_367224


namespace NUMINAMATH_CALUDE_triangle_inequality_l3672_367231

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  (a * b) / (s - c) + (b * c) / (s - a) + (c * a) / (s - b) ≥ 4 * s := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3672_367231


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l3672_367261

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the two possible standard equations for the parabola
def parabola_eq1 (x y : ℝ) : Prop := y^2 = -12 * x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -6 * y

-- Theorem statement
theorem parabola_focus_theorem :
  ∀ (x y : ℝ), focus_line x y →
  (parabola_eq1 x y ∨ parabola_eq2 x y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l3672_367261


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l3672_367287

/-- The distance from the center of the circle (x+4)^2 + (y-3)^2 = 9 to the line 4x + 3y - 1 = 0 is 8/5 -/
theorem distance_circle_center_to_line : 
  let circle := fun (x y : ℝ) => (x + 4)^2 + (y - 3)^2 = 9
  let line := fun (x y : ℝ) => 4*x + 3*y - 1 = 0
  let center := (-4, 3)
  abs (4 * center.1 + 3 * center.2 - 1) / Real.sqrt (4^2 + 3^2) = 8/5 := by
sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l3672_367287


namespace NUMINAMATH_CALUDE_no_real_roots_l3672_367238

theorem no_real_roots : ¬∃ (x : ℝ), x^2 - 4*x + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3672_367238


namespace NUMINAMATH_CALUDE_first_week_sales_l3672_367228

/-- Represents the sales of chips in a convenience store over a month -/
structure ChipSales where
  total : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  fourth_week : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem first_week_sales (s : ChipSales) :
  s.total = 100 ∧
  s.second_week = 3 * s.first_week ∧
  s.third_week = 20 ∧
  s.fourth_week = 20 ∧
  s.total = s.first_week + s.second_week + s.third_week + s.fourth_week →
  s.first_week = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_week_sales_l3672_367228


namespace NUMINAMATH_CALUDE_max_value_product_sum_l3672_367208

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 →
    A * M * C + A * M + M * C + C * A + A + M + C ≥
    a * m * c + a * m + m * c + c * a + a + m + c) →
  A * M * C + A * M + M * C + C * A + A + M + C = 215 :=
sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l3672_367208


namespace NUMINAMATH_CALUDE_solve_for_k_l3672_367296

theorem solve_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l3672_367296


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3672_367252

theorem sum_remainder_mod_seven : (5283 + 5284 + 5285 + 5286 + 5287) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3672_367252


namespace NUMINAMATH_CALUDE_warehouse_analysis_l3672_367210

/-- Represents the daily changes in goods, where positive values indicate goods entering
    and negative values indicate goods leaving the warehouse -/
def daily_changes : List Int := [31, -31, -16, 34, -38, -20]

/-- The final amount of goods in the warehouse after 6 days -/
def final_amount : Int := 430

/-- The fee for loading or unloading one ton of goods -/
def fee_per_ton : Int := 5

theorem warehouse_analysis :
  let net_change := daily_changes.sum
  let initial_amount := final_amount - net_change
  let total_fees := (daily_changes.map abs).sum * fee_per_ton
  (net_change < 0) ∧
  (initial_amount = 470) ∧
  (total_fees = 850) := by sorry

end NUMINAMATH_CALUDE_warehouse_analysis_l3672_367210


namespace NUMINAMATH_CALUDE_log_equation_solution_l3672_367223

theorem log_equation_solution (a x : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (h : (Real.log x) / (Real.log (a^3)) + (Real.log a) / (Real.log (x^2)) = 2) :
  x = a^(3 + (5 * Real.sqrt 3) / 2) ∨ x = a^(3 - (5 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3672_367223


namespace NUMINAMATH_CALUDE_power_sum_sequence_l3672_367259

theorem power_sum_sequence (a b x y : ℝ) 
  (eq1 : a*x + b*y = 3)
  (eq2 : a*x^2 + b*y^2 = 7)
  (eq3 : a*x^3 + b*y^3 = 16)
  (eq4 : a*x^4 + b*y^4 = 42) :
  a*x^5 + b*y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l3672_367259


namespace NUMINAMATH_CALUDE_bob_distance_from_start_l3672_367248

-- Define the regular pentagon
def regularPentagon (sideLength : ℝ) : Set (ℝ × ℝ) :=
  sorry

-- Define Bob's position after walking a certain distance
def bobPosition (distance : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem bob_distance_from_start :
  let pentagon := regularPentagon 3
  let finalPosition := bobPosition 7
  let distance := Real.sqrt ((finalPosition.1)^2 + (finalPosition.2)^2)
  distance = Real.sqrt 6.731 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_from_start_l3672_367248


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3672_367233

theorem min_distance_to_line (x y : ℝ) :
  (3 * x + y = 10) → (x^2 + y^2 ≥ 10) := by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3672_367233


namespace NUMINAMATH_CALUDE_range_of_a_l3672_367247

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a^2 + a)*x + a^3 > 0 ↔ x < a^2 ∨ x > a) →
  0 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3672_367247


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3672_367283

theorem pie_eating_contest (student1_session1 student1_session2 student2_session1 student2_session2 : ℚ)
  (h1 : student1_session1 = 7/8)
  (h2 : student1_session2 = 3/4)
  (h3 : student2_session1 = 5/6)
  (h4 : student2_session2 = 2/3) :
  (student1_session1 + student1_session2) - (student2_session1 + student2_session2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3672_367283


namespace NUMINAMATH_CALUDE_inequality_properties_l3672_367226

theorem inequality_properties (m n : ℝ) :
  (∀ a : ℝ, a ≠ 0 → m * a^2 < n * a^2 → m < n) ∧
  (m < n → n < 0 → n / m < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3672_367226


namespace NUMINAMATH_CALUDE_number_wall_solution_l3672_367280

/-- Represents a block in the Number Wall --/
structure Block where
  value : ℕ

/-- Represents the Number Wall --/
structure NumberWall where
  n : Block
  block1 : Block
  block2 : Block
  block3 : Block
  block4 : Block
  top : Block

/-- The sum of two adjacent blocks equals the block above them --/
def sum_rule (b1 b2 b_above : Block) : Prop :=
  b1.value + b2.value = b_above.value

/-- The Number Wall satisfies all given conditions --/
def valid_wall (w : NumberWall) : Prop :=
  w.block1.value = 4 ∧
  w.block2.value = 8 ∧
  w.block3.value = 7 ∧
  w.block4.value = 15 ∧
  w.top.value = 46 ∧
  sum_rule w.n w.block1 { value := w.n.value + 4 } ∧
  sum_rule { value := w.n.value + 4 } w.block2 w.block4 ∧
  sum_rule w.block4 w.block3 { value := 27 } ∧
  sum_rule { value := w.n.value + 16 } { value := 27 } w.top

theorem number_wall_solution (w : NumberWall) (h : valid_wall w) : w.n.value = 3 := by
  sorry


end NUMINAMATH_CALUDE_number_wall_solution_l3672_367280


namespace NUMINAMATH_CALUDE_prob_at_least_one_three_l3672_367242

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := sides * sides

/-- The number of outcomes where neither die shows a 3 -/
def neither_three : ℕ := (sides - 1) * (sides - 1)

/-- The number of outcomes where at least one die shows a 3 -/
def at_least_one_three : ℕ := total_outcomes - neither_three

/-- The probability of getting at least one 3 when rolling two 8-sided dice -/
theorem prob_at_least_one_three : 
  (at_least_one_three : ℚ) / total_outcomes = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_three_l3672_367242


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l3672_367299

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.im (i / (1 + i)) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l3672_367299


namespace NUMINAMATH_CALUDE_zero_natural_number_ambiguity_l3672_367220

-- Define a type for natural number conventions
inductive NatConvention where
  | withZero    : NatConvention
  | withoutZero : NatConvention

-- Define a function that checks if 0 is a natural number based on the convention
def isZeroNatural (conv : NatConvention) : Prop :=
  match conv with
  | NatConvention.withZero    => True
  | NatConvention.withoutZero => False

-- Theorem statement
theorem zero_natural_number_ambiguity :
  ∃ (conv : NatConvention), isZeroNatural conv :=
sorry


end NUMINAMATH_CALUDE_zero_natural_number_ambiguity_l3672_367220


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l3672_367218

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9 * y
  | 2 => 27 * y^2
  | 3 => 81 * y^3
  | n + 4 => geometric_sequence y 3 * (3 * y)^(n + 1)

/-- The fifth term of the geometric sequence is 243y^4 -/
theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 243 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l3672_367218


namespace NUMINAMATH_CALUDE_triangle_problem_l3672_367282

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.sin C = b * Real.sin A ∧ b * Real.sin C = c * Real.sin B ∧ c * Real.sin A = a * Real.sin B →
  (a * (Real.sin C - Real.sin A)) / (Real.sin C + Real.sin B) = c - b →
  Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4 →
  B = π / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3672_367282


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3672_367291

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the line
def line (x : ℝ) : Prop := x = -1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (4, 0)

-- Define points A and B
variable (A B : ℝ × ℝ)

-- State the theorem
theorem parabola_intersection_length :
  parabola B.1 B.2 →
  line A.1 →
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (1 - t) • focus + t • A) →
  (A - focus) = 5 • (B - focus) →
  ‖A - B‖ = 28 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3672_367291


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l3672_367202

/-- A conic section in the xy-plane -/
structure ConicSection where
  equation : ℝ → ℝ → Prop

/-- A hyperbola in the xy-plane -/
structure Hyperbola extends ConicSection

/-- The specific conic section given by the equation x^2 + my^2 = 1 -/
def specific_conic (m : ℝ) : ConicSection where
  equation := fun x y => x^2 + m*y^2 = 1

theorem hyperbola_iff_m_negative (m : ℝ) :
  ∃ (h : Hyperbola), h.equation = (specific_conic m).equation ↔ m < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l3672_367202


namespace NUMINAMATH_CALUDE_evaluate_expression_l3672_367212

theorem evaluate_expression : 6 - 8 * (9 - 2^3 + 12/3) * 5 = -194 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3672_367212


namespace NUMINAMATH_CALUDE_complement_of_union_MN_l3672_367216

def I : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {2, 3, 5}

theorem complement_of_union_MN :
  (M ∪ N)ᶜ = {4} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_MN_l3672_367216


namespace NUMINAMATH_CALUDE_jogger_speed_l3672_367273

/-- Proves that the jogger's speed is 9 kmph given the conditions of the problem -/
theorem jogger_speed (train_length : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : train_speed = 45)
  (h3 : initial_distance = 240)
  (h4 : passing_time = 36)
  : ∃ (jogger_speed : ℝ), jogger_speed = 9 ∧ 
    (train_speed - jogger_speed) * passing_time * (5/18) = initial_distance + train_length :=
by
  sorry

#check jogger_speed

end NUMINAMATH_CALUDE_jogger_speed_l3672_367273


namespace NUMINAMATH_CALUDE_f_second_derivative_positive_l3672_367214

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

def f_domain : Set ℝ := {x : ℝ | x > 0}

noncomputable def f'' (x : ℝ) : ℝ := 2 + 4 / x^2

theorem f_second_derivative_positive :
  {x ∈ f_domain | f'' x > 0} = f_domain :=
sorry

end NUMINAMATH_CALUDE_f_second_derivative_positive_l3672_367214


namespace NUMINAMATH_CALUDE_triangle_angles_l3672_367276

/-- Given a triangle with sides 5, 5, and √17 - √5, prove that its angles are θ, φ, φ, where
    θ = arccos((14 + √85) / 25) and φ = (180° - θ) / 2 -/
theorem triangle_angles (a b c : ℝ) (θ φ : ℝ) : 
  a = 5 → b = 5 → c = Real.sqrt 17 - Real.sqrt 5 →
  θ = Real.arccos ((14 + Real.sqrt 85) / 25) →
  φ = (π - θ) / 2 →
  ∃ (α β γ : ℝ), 
    (α = θ ∧ β = φ ∧ γ = φ) ∧
    (α + β + γ = π) ∧
    (Real.cos α = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
    (Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (Real.cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l3672_367276


namespace NUMINAMATH_CALUDE_robot_constraint_l3672_367272

-- Define the robot's path as a parabola
def robot_path (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through P(-1, 0) with slope k
def line_through_P (k x y : ℝ) : Prop := y = k*(x + 1)

-- Define the condition that the line does not intersect the robot's path
def no_intersection (k : ℝ) : Prop :=
  ∀ x y : ℝ, robot_path x y ∧ line_through_P k x y → False

-- Theorem statement
theorem robot_constraint (k : ℝ) :
  no_intersection k ↔ k < -Real.sqrt 2 ∨ k > Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_robot_constraint_l3672_367272


namespace NUMINAMATH_CALUDE_sqrt_x_plus_5_equals_3_l3672_367275

theorem sqrt_x_plus_5_equals_3 (x : ℝ) : 
  Real.sqrt (x + 5) = 3 → (x + 5)^2 = 81 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_5_equals_3_l3672_367275


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3672_367289

theorem sum_of_four_consecutive_even_integers (n : ℤ) : 
  (∃ k : ℤ, n = 4*k + 12 ∧ k % 2 = 0) ↔ n ∈ ({56, 80, 124, 200} : Set ℤ) := by
  sorry

#check sum_of_four_consecutive_even_integers 34
#check sum_of_four_consecutive_even_integers 56
#check sum_of_four_consecutive_even_integers 80
#check sum_of_four_consecutive_even_integers 124
#check sum_of_four_consecutive_even_integers 200

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3672_367289


namespace NUMINAMATH_CALUDE_area_of_region_R_l3672_367258

/-- Represents a rhombus ABCD -/
structure Rhombus where
  sideLength : ℝ
  angleB : ℝ

/-- Represents the region R inside the rhombus -/
def RegionR (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
noncomputable def areaR (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of region R in the specific rhombus -/
theorem area_of_region_R : 
  let r : Rhombus := { sideLength := 3, angleB := 150 * π / 180 }
  ∃ ε > 0, |areaR r - 0.873| < ε :=
sorry

end NUMINAMATH_CALUDE_area_of_region_R_l3672_367258


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l3672_367286

theorem smallest_staircase_steps (n : ℕ) : 
  (n > 15) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l3672_367286


namespace NUMINAMATH_CALUDE_cos_2alpha_problem_l3672_367285

theorem cos_2alpha_problem (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 10 / 5) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_problem_l3672_367285


namespace NUMINAMATH_CALUDE_milk_packet_price_problem_l3672_367227

/-- Given 5 packets of milk with an average price of 20 cents, if 2 packets are returned
    and the average price of the remaining 3 packets is 12 cents, then the average price
    of the 2 returned packets is 32 cents. -/
theorem milk_packet_price_problem (total_packets : Nat) (remaining_packets : Nat) 
    (initial_avg_price : ℚ) (remaining_avg_price : ℚ) :
  total_packets = 5 →
  remaining_packets = 3 →
  initial_avg_price = 20 →
  remaining_avg_price = 12 →
  let returned_packets := total_packets - remaining_packets
  let total_cost := total_packets * initial_avg_price
  let remaining_cost := remaining_packets * remaining_avg_price
  let returned_cost := total_cost - remaining_cost
  (returned_cost / returned_packets : ℚ) = 32 := by
sorry

end NUMINAMATH_CALUDE_milk_packet_price_problem_l3672_367227


namespace NUMINAMATH_CALUDE_unique_amazing_rectangle_l3672_367251

/-- An amazing rectangle is a rectangle where the area is equal to three times its perimeter,
    one side is double the other, and both sides are positive integers. -/
structure AmazingRectangle where
  width : ℕ+
  length : ℕ+
  is_double : length = 2 * width
  is_amazing : width * length = 3 * (2 * (width + length))

/-- Theorem stating that there exists only one amazing rectangle and its area is 162. -/
theorem unique_amazing_rectangle :
  (∃! r : AmazingRectangle, True) ∧
  (∀ r : AmazingRectangle, r.width * r.length = 162) := by
  sorry


end NUMINAMATH_CALUDE_unique_amazing_rectangle_l3672_367251


namespace NUMINAMATH_CALUDE_a_is_geometric_sequence_l3672_367244

/-- A linear function f(x) = bx + 1 where b is a constant not equal to 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := b * x + 1

/-- A recursive function g(n) defined as:
    g(0) = 1
    g(n) = f(g(n-1)) for n ≥ 1 -/
def g (b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => f b (g b n)

/-- The sequence a_n defined as a_n = g(n) - g(n-1) for n ∈ ℕ* -/
def a (b : ℝ) (n : ℕ) : ℝ := g b (n + 1) - g b n

/-- Theorem: The sequence {a_n} is a geometric sequence -/
theorem a_is_geometric_sequence (b : ℝ) (h : b ≠ 1) :
  ∃ r : ℝ, ∀ n : ℕ, a b (n + 1) = r * a b n :=
sorry

end NUMINAMATH_CALUDE_a_is_geometric_sequence_l3672_367244


namespace NUMINAMATH_CALUDE_jerry_has_49_feathers_l3672_367284

/-- The number of feathers Jerry has left after his adventure -/
def jerrys_remaining_feathers : ℕ :=
  let hawk_feathers : ℕ := 6
  let eagle_feathers : ℕ := 17 * hawk_feathers
  let total_feathers : ℕ := hawk_feathers + eagle_feathers
  let feathers_after_giving : ℕ := total_feathers - 10
  (feathers_after_giving / 2 : ℕ)

/-- Theorem stating that Jerry has 49 feathers left -/
theorem jerry_has_49_feathers : jerrys_remaining_feathers = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_49_feathers_l3672_367284


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l3672_367262

theorem triangle_is_equilateral (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 + c^2 - a^2 = b*c)
  (h2 : 2 * Real.cos B * Real.sin C = Real.sin A)
  (h3 : A + B + C = π)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h6 : A < π ∧ B < π ∧ C < π) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l3672_367262


namespace NUMINAMATH_CALUDE_max_cone_radius_in_crate_l3672_367237

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Checks if a cone fits upright in a crate -/
def fitsInCrate (cone : Cone) (crate : CrateDimensions) : Prop :=
  cone.height ≤ max crate.length (max crate.width crate.height) ∧
  2 * cone.radius ≤ min crate.length (min crate.width crate.height)

/-- The theorem stating the maximum radius of a cone that fits in the given crate -/
theorem max_cone_radius_in_crate :
  ∃ (maxRadius : ℝ),
    maxRadius = 2.5 ∧
    ∀ (c : Cone),
      fitsInCrate c (CrateDimensions.mk 5 8 12) →
      c.radius ≤ maxRadius :=
sorry

end NUMINAMATH_CALUDE_max_cone_radius_in_crate_l3672_367237


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3672_367271

theorem consecutive_odd_integers_sum (x : ℤ) :
  (∃ y z : ℤ, y = x + 2 ∧ z = x + 4 ∧ x + z = 150) →
  x + (x + 2) + (x + 4) = 225 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3672_367271


namespace NUMINAMATH_CALUDE_modulo_congruence_l3672_367243

theorem modulo_congruence : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4792 - 242 [ZMOD 8] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_l3672_367243


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l3672_367288

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 56) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l3672_367288


namespace NUMINAMATH_CALUDE_min_z_value_l3672_367257

theorem min_z_value (x y z : ℤ) (sum_eq : x + y + z = 100) (ineq : x < y ∧ y < 2*z) : 
  ∀ w : ℤ, (∃ a b : ℤ, a + b + w = 100 ∧ a < b ∧ b < 2*w) → w ≥ 21 := by
  sorry

#check min_z_value

end NUMINAMATH_CALUDE_min_z_value_l3672_367257


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l3672_367213

/-- Given a square with side length s_s and an equilateral triangle with side length s_t,
    if their perimeters are equal, then the ratio of s_t to s_s is 4/3. -/
theorem square_triangle_perimeter_ratio (s_s s_t : ℝ) (h : s_s > 0) (h' : s_t > 0) :
  4 * s_s = 3 * s_t → s_t / s_s = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l3672_367213


namespace NUMINAMATH_CALUDE_find_divisor_l3672_367281

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (h1 : dividend = 1254) (h2 : quotient = 209) 
  (h3 : dividend % (dividend / quotient) = 0) : dividend / quotient = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3672_367281


namespace NUMINAMATH_CALUDE_unique_valid_square_l3672_367215

/-- A perfect square less than 100 with ones digit 5, 6, or 7 -/
def ValidSquare (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k^2 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 6 ∨ n % 10 = 7)

/-- There is exactly one perfect square less than 100 with ones digit 5, 6, or 7 -/
theorem unique_valid_square : ∃! (n : ℕ), ValidSquare n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_square_l3672_367215


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3672_367298

/-- Triangle ABC with vertices A(-4,0), B(0,2), and C(2,-2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation :=
  (D : ℝ)
  (E : ℝ)
  (F : ℝ)

/-- Function to check if a point satisfies a line equation -/
def satisfiesLineEquation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Function to check if a point satisfies a circle equation -/
def satisfiesCircleEquation (p : ℝ × ℝ) (eq : CircleEquation) : Prop :=
  p.1^2 + p.2^2 + eq.D * p.1 + eq.E * p.2 + eq.F = 0

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.A = (-4, 0))
  (h2 : t.B = (0, 2))
  (h3 : t.C = (2, -2)) :
  ∃ (medianAB : LineEquation) (circumcircle : CircleEquation),
    -- Median equation
    (medianAB = ⟨3, 4, -2⟩) ∧ 
    -- Circumcircle equation
    (circumcircle = ⟨2, 2, -8⟩) ∧
    -- Verify that C and the midpoint of AB satisfy the median equation
    (satisfiesLineEquation t.C medianAB) ∧
    (satisfiesLineEquation ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2) medianAB) ∧
    -- Verify that all vertices satisfy the circumcircle equation
    (satisfiesCircleEquation t.A circumcircle) ∧
    (satisfiesCircleEquation t.B circumcircle) ∧
    (satisfiesCircleEquation t.C circumcircle) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3672_367298


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_non_obtuse_triangle_l3672_367253

/-- For any non-obtuse triangle with angles α, β, and γ, the sum of the sines of these angles 
is greater than the sum of the cosines of these angles. -/
theorem sine_cosine_inequality_non_obtuse_triangle (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi)
  (h_non_obtuse : α ≤ Real.pi/2 ∧ β ≤ Real.pi/2 ∧ γ ≤ Real.pi/2) :
  Real.sin α + Real.sin β + Real.sin γ > Real.cos α + Real.cos β + Real.cos γ :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_non_obtuse_triangle_l3672_367253


namespace NUMINAMATH_CALUDE_square_less_than_power_of_three_l3672_367245

theorem square_less_than_power_of_three (n : ℕ) (h : n ≥ 3) : (n + 1)^2 < 3^n := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_power_of_three_l3672_367245


namespace NUMINAMATH_CALUDE_one_third_1206_percent_of_134_l3672_367240

theorem one_third_1206_percent_of_134 : 
  (1206 / 3) / 134 * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_one_third_1206_percent_of_134_l3672_367240


namespace NUMINAMATH_CALUDE_consecutive_integers_base_sum_l3672_367219

/-- Given two consecutive positive integers X and Y, 
    if 241 in base X plus 52 in base Y equals 194 in base (X+Y), 
    then X + Y equals 15 -/
theorem consecutive_integers_base_sum (X Y : ℕ) : 
  X > 0 ∧ Y > 0 ∧ Y = X + 1 →
  (2 * X^2 + 4 * X + 1) + (5 * Y + 2) = ((X + Y)^2 + 9 * (X + Y) + 4) →
  X + Y = 15 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_sum_l3672_367219


namespace NUMINAMATH_CALUDE_benzoic_acid_weight_l3672_367265

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound -/
def molecularWeight (formula : MolecularFormula) (weights : AtomicWeights) : ℝ :=
  formula.carbon * weights.carbon +
  formula.hydrogen * weights.hydrogen +
  formula.oxygen * weights.oxygen

/-- Theorem: The molecular weight of 4 moles of Benzoic acid is 488.472 grams -/
theorem benzoic_acid_weight :
  let benzoicAcid : MolecularFormula := { carbon := 7, hydrogen := 6, oxygen := 2 }
  let atomicWeights : AtomicWeights := { carbon := 12.01, hydrogen := 1.008, oxygen := 16.00 }
  (4 : ℝ) * molecularWeight benzoicAcid atomicWeights = 488.472 := by
  sorry


end NUMINAMATH_CALUDE_benzoic_acid_weight_l3672_367265


namespace NUMINAMATH_CALUDE_m_plus_n_values_l3672_367267

theorem m_plus_n_values (m n : ℤ) (hm : m = 3) (hn : |n| = 1) :
  m + n = 4 ∨ m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_values_l3672_367267


namespace NUMINAMATH_CALUDE_profit_with_discount_theorem_l3672_367278

/-- Calculates the profit percentage with discount given the discount rate and profit percentage without discount -/
def profit_percentage_with_discount (discount_rate : ℝ) (profit_no_discount : ℝ) : ℝ :=
  ((1 - discount_rate) * (1 + profit_no_discount) - 1) * 100

/-- Theorem stating that given a 5% discount and 28% profit without discount, the profit percentage with discount is 21.6% -/
theorem profit_with_discount_theorem :
  profit_percentage_with_discount 0.05 0.28 = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_profit_with_discount_theorem_l3672_367278


namespace NUMINAMATH_CALUDE_words_exceeded_proof_l3672_367234

def word_limit : ℕ := 1000
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650

theorem words_exceeded_proof :
  (saturday_words + sunday_words) - word_limit = 100 := by
  sorry

end NUMINAMATH_CALUDE_words_exceeded_proof_l3672_367234


namespace NUMINAMATH_CALUDE_polygon_division_theorem_l3672_367256

/-- A polygon is a closed planar figure with straight sides -/
structure Polygon where
  sides : ℕ
  is_closed : Bool
  is_planar : Bool

/-- Represents a division of a polygon into shapes -/
structure PolygonDivision (P : Polygon) (n : ℕ) (shape : Type) where
  num_divisions : ℕ
  is_valid : Bool

/-- Given a polygon that can be divided into 100 rectangles but not 99,
    it cannot be divided into 100 triangles -/
theorem polygon_division_theorem (P : Polygon) 
  (h1 : ∃ (d : PolygonDivision P 100 Rectangle), d.is_valid)
  (h2 : ¬ ∃ (d : PolygonDivision P 99 Rectangle), d.is_valid) :
  ¬ ∃ (d : PolygonDivision P 100 Triangle), d.is_valid :=
by sorry

end NUMINAMATH_CALUDE_polygon_division_theorem_l3672_367256


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3672_367249

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P Q F : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F = (c, 0) →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →
  (Q.1 - c/3)^2 + Q.2^2 = b^2/9 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 * ((Q.1 - F.1)^2 + (Q.2 - F.2)^2) →
  (P.1 - F.1) * (Q.1 - c/3) + (P.2 - F.2) * Q.2 = 0 →
  c^2 = a^2 + b^2 →
  c / a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3672_367249


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3672_367225

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 11) = 10 → x = 89 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3672_367225


namespace NUMINAMATH_CALUDE_oldest_child_daily_cheese_is_two_l3672_367203

/-- The number of string cheeses Kelly's oldest child wants every day. -/
def oldest_child_daily_cheese : ℕ := 
  let days_per_week : ℕ := 5
  let total_weeks : ℕ := 4
  let cheeses_per_package : ℕ := 30
  let packages_needed : ℕ := 2
  let youngest_child_daily_cheese : ℕ := 1
  let total_days : ℕ := days_per_week * total_weeks
  let total_cheeses : ℕ := packages_needed * cheeses_per_package
  let youngest_total_cheeses : ℕ := youngest_child_daily_cheese * total_days
  let oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses
  oldest_total_cheeses / total_days

theorem oldest_child_daily_cheese_is_two : oldest_child_daily_cheese = 2 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_daily_cheese_is_two_l3672_367203


namespace NUMINAMATH_CALUDE_m_minus_n_values_l3672_367236

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 5)
  (hn : |n| = 7)
  (hmn_neg : m + n < 0) :
  m - n = 12 ∨ m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l3672_367236


namespace NUMINAMATH_CALUDE_january_oil_bill_l3672_367235

theorem january_oil_bill (feb_bill jan_bill : ℚ) : 
  (feb_bill / jan_bill = 3 / 2) → 
  ((feb_bill + 10) / jan_bill = 5 / 3) → 
  jan_bill = 60 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l3672_367235


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3672_367254

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ 2520) :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3672_367254


namespace NUMINAMATH_CALUDE_rats_meet_on_day_10_l3672_367230

/-- The thickness of the wall in feet -/
def wall_thickness : ℕ := 1000

/-- The initial drilling speed of both rats in feet per day -/
def initial_speed : ℕ := 1

/-- The function representing the total distance drilled by both rats after n days -/
def total_distance (n : ℕ) : ℚ :=
  (2^n - 1) + 2 * (1 - (1/2)^n)

/-- The theorem stating that the rats meet on the 10th day -/
theorem rats_meet_on_day_10 :
  total_distance 9 < wall_thickness ∧ total_distance 10 ≥ wall_thickness :=
sorry

end NUMINAMATH_CALUDE_rats_meet_on_day_10_l3672_367230


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_range_l3672_367250

theorem sin_cos_equation_solution_range :
  let f : ℝ → ℝ → ℝ := λ x a => Real.sin x ^ 2 + 2 * Real.cos x + a
  ∀ a : ℝ, (∃ x : ℝ, f x a = 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_range_l3672_367250


namespace NUMINAMATH_CALUDE_smallest_x_value_l3672_367266

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 90 = x * (x + 15)) → x ≥ -15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3672_367266


namespace NUMINAMATH_CALUDE_line_equation_represents_line_l3672_367264

/-- A line in the 2D plane defined by the equation y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The set of points (x, y) satisfying a linear equation -/
def LinePoints (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = l.m * p.1 + l.b}

theorem line_equation_represents_line :
  ∃ (l : Line), l.m = 2 ∧ l.b = 1 ∧
  LinePoints l = {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} :=
by sorry

end NUMINAMATH_CALUDE_line_equation_represents_line_l3672_367264


namespace NUMINAMATH_CALUDE_f_inequality_iff_a_range_l3672_367292

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - (a + 1) * x + (1/2) * x^2

theorem f_inequality_iff_a_range (a : ℝ) :
  (a > 0 ∧ ∀ x > 1, f a x ≥ x^a - Real.exp x + (1/2) * x^2 - a * x) ↔ 0 < a ∧ a ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_iff_a_range_l3672_367292


namespace NUMINAMATH_CALUDE_nba_player_age_distribution_l3672_367209

theorem nba_player_age_distribution (total_players : ℕ) 
  (h1 : total_players = 1000)
  (h2 : (2 : ℚ) / 5 * total_players = (players_25_to_35 : ℕ))
  (h3 : (3 : ℚ) / 8 * total_players = (players_over_35 : ℕ)) :
  total_players - (players_25_to_35 + players_over_35) = 225 :=
by sorry

end NUMINAMATH_CALUDE_nba_player_age_distribution_l3672_367209


namespace NUMINAMATH_CALUDE_remainder_theorem_l3672_367211

theorem remainder_theorem (N : ℤ) : 
  (N % 779 = 47) → (N % 19 = 9) := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3672_367211


namespace NUMINAMATH_CALUDE_lukas_points_l3672_367269

/-- Given a basketball player's average points per game and a number of games,
    calculates the total points scored. -/
def total_points (avg_points : ℕ) (num_games : ℕ) : ℕ :=
  avg_points * num_games

/-- Proves that a player averaging 12 points per game scores 60 points in 5 games. -/
theorem lukas_points : total_points 12 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lukas_points_l3672_367269


namespace NUMINAMATH_CALUDE_equation_solutions_l3672_367295

theorem equation_solutions :
  (∀ x : ℝ, 12 * (x - 1)^2 = 3 ↔ x = 3/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, (x + 1)^3 = 0.125 ↔ x = -0.5) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3672_367295


namespace NUMINAMATH_CALUDE_sum_plus_even_count_equals_1811_l3672_367201

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_plus_even_count_equals_1811 :
  sum_of_integers 10 60 + count_even_integers 10 60 = 1811 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_even_count_equals_1811_l3672_367201


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l3672_367268

theorem abs_inequality_equivalence :
  ∀ x : ℝ, |5 - 2*x| < 3 ↔ 1 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l3672_367268


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3672_367290

theorem opposite_of_negative_three : 
  (∃ x : ℤ, -3 + x = 0) → (∃ x : ℤ, -3 + x = 0 ∧ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3672_367290


namespace NUMINAMATH_CALUDE_f_lower_bound_g_min_max_l3672_367263

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x^2 + Real.log x

def g (x : ℝ) : ℝ := x^2 - 2 * Real.log x

-- State the theorems
theorem f_lower_bound (x : ℝ) (hx : x > 0) : f x ≥ (x^3 + x - 1) / x := by sorry

theorem g_min_max :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g x ≥ 1) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g x = 1) ∧
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g x ≤ 4 - 2 * Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g x = 4 - 2 * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_g_min_max_l3672_367263


namespace NUMINAMATH_CALUDE_parabola_focus_distance_range_l3672_367260

theorem parabola_focus_distance_range :
  ∀ (A : ℝ × ℝ) (θ : ℝ),
    let F : ℝ × ℝ := (1/4, 0)
    let y : ℝ → ℝ := λ x => Real.sqrt x
    let l : ℝ → ℝ := λ x => Real.tan θ * (x - F.1) + F.2
    A.2 = y A.1 ∧  -- A is on the parabola
    A.2 > 0 ∧  -- A is above x-axis
    l A.1 = A.2 ∧  -- A is on line l
    θ ≥ π/4 →
    ∃ (FA : ℝ), FA > 1/4 ∧ FA ≤ 1 + Real.sqrt 2 / 2 ∧
      FA = Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_range_l3672_367260


namespace NUMINAMATH_CALUDE_candidate_marks_l3672_367277

theorem candidate_marks (max_marks : ℝ) (pass_percentage : ℝ) (fail_margin : ℕ) 
  (h1 : max_marks = 152.38)
  (h2 : pass_percentage = 0.42)
  (h3 : fail_margin = 22) : 
  ∃ (secured_marks : ℕ), secured_marks = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_marks_l3672_367277


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3672_367217

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℝ) 
  (set2_count : ℕ) (set2_mean : ℝ) : 
  set1_count = 5 → 
  set1_mean = 16 → 
  set2_count = 8 → 
  set2_mean = 21 → 
  let total_count := set1_count + set2_count
  let combined_mean := (set1_count * set1_mean + set2_count * set2_mean) / total_count
  combined_mean = 19.08 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3672_367217


namespace NUMINAMATH_CALUDE_exponential_inequality_l3672_367294

theorem exponential_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) :
  a^(Real.sqrt a) > a^(a^a) ∧ a^(a^a) > a := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3672_367294


namespace NUMINAMATH_CALUDE_algorithmC_is_best_l3672_367206

-- Define the durations of each task
def washAndBrush : ℕ := 5
def cleanKettle : ℕ := 2
def boilWater : ℕ := 8
def makeNoodles : ℕ := 3
def eat : ℕ := 10
def listenRadio : ℕ := 8

-- Define the algorithms
def algorithmA : ℕ := washAndBrush + cleanKettle + boilWater + makeNoodles + eat + listenRadio
def algorithmB : ℕ := cleanKettle + max boilWater washAndBrush + makeNoodles + eat + listenRadio
def algorithmC : ℕ := cleanKettle + max boilWater washAndBrush + makeNoodles + max eat listenRadio
def algorithmD : ℕ := max eat listenRadio + makeNoodles + max boilWater washAndBrush + cleanKettle

-- Theorem stating that algorithm C takes the least time
theorem algorithmC_is_best : 
  algorithmC ≤ algorithmA ∧ 
  algorithmC ≤ algorithmB ∧ 
  algorithmC ≤ algorithmD :=
sorry

end NUMINAMATH_CALUDE_algorithmC_is_best_l3672_367206


namespace NUMINAMATH_CALUDE_students_left_unassigned_l3672_367279

/-- The number of students left unassigned to groups in a school with specific classroom distributions -/
theorem students_left_unassigned (total_students : ℕ) (num_classrooms : ℕ) 
  (classroom_A : ℕ) (classroom_B : ℕ) (classroom_C : ℕ) (classroom_D : ℕ) 
  (num_groups : ℕ) : 
  total_students = 128 →
  num_classrooms = 4 →
  classroom_A = 37 →
  classroom_B = 31 →
  classroom_C = 25 →
  classroom_D = 35 →
  num_groups = 9 →
  classroom_A + classroom_B + classroom_C + classroom_D = total_students →
  total_students - (num_groups * (total_students / num_groups)) = 2 := by
  sorry

#eval 128 - (9 * (128 / 9))  -- This should evaluate to 2

end NUMINAMATH_CALUDE_students_left_unassigned_l3672_367279


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l3672_367246

theorem simplify_algebraic_expression (a b : ℝ) (h : b ≠ 0) :
  (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l3672_367246


namespace NUMINAMATH_CALUDE_negation_of_not_all_zero_l3672_367241

theorem negation_of_not_all_zero (a b c : ℝ) :
  ¬(¬(a = 0 ∧ b = 0 ∧ c = 0)) ↔ (a = 0 ∧ b = 0 ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_not_all_zero_l3672_367241


namespace NUMINAMATH_CALUDE_right_triangle_circle_chord_length_l3672_367205

/-- Represents a triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: In a right triangle ABC with hypotenuse AB = 10, AC = 8, and BC = 6,
    if a circle P passes through C and is tangent to AB at its midpoint,
    then the length of the chord QR (where Q and R are the intersections of
    the circle with AC and BC respectively) is equal to 9.6. -/
theorem right_triangle_circle_chord_length
  (abc : Triangle)
  (p : Circle)
  (h1 : abc.a = 10 ∧ abc.b = 8 ∧ abc.c = 6)
  (h2 : abc.a^2 = abc.b^2 + abc.c^2)
  (h3 : p.center = (5, p.radius))
  (h4 : p.radius = abc.b * abc.c / abc.a) :
  2 * p.radius = 9.6 := by sorry

end NUMINAMATH_CALUDE_right_triangle_circle_chord_length_l3672_367205


namespace NUMINAMATH_CALUDE_caitlin_age_l3672_367222

theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) : 
  anna_age = 48 →
  brianna_age = anna_age / 2 →
  caitlin_age = brianna_age - 7 →
  caitlin_age = 17 := by
sorry

end NUMINAMATH_CALUDE_caitlin_age_l3672_367222


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3672_367297

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y₁ ≠ 0 ∧ y₂ ≠ 0) 
  (h_inverse : ∃ c : ℝ, c ≠ 0 ∧ x₁ * y₁ = c ∧ x₂ * y₂ = c) 
  (h_ratio : x₁ / x₂ = 3 / 5) : 
  y₁ / y₂ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3672_367297


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3672_367229

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a < b ∧ b < 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3672_367229
