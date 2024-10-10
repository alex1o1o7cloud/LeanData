import Mathlib

namespace divisibility_of_sum_of_squares_l65_6594

theorem divisibility_of_sum_of_squares (p k a b : ℤ) : 
  Prime p → 
  p = 4*k + 3 → 
  p ∣ (a^2 + b^2) → 
  p ∣ a ∧ p ∣ b := by
  sorry

end divisibility_of_sum_of_squares_l65_6594


namespace f_properties_l65_6581

/-- Definition of an odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a^x - 1 else 1 - a^(-x)

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  OddFunction (f a) ∧ 
  (f a 2 + f a (-2) = 0) ∧
  (∀ x, f a x = if x ≥ 0 then a^x - 1 else 1 - a^(-x)) := by
  sorry

end f_properties_l65_6581


namespace bike_cost_theorem_l65_6541

def apple_price : ℚ := 1.25
def apples_sold : ℕ := 20
def repair_ratio : ℚ := 1/4
def remaining_ratio : ℚ := 1/5

def total_earnings : ℚ := apple_price * apples_sold

theorem bike_cost_theorem (h1 : total_earnings = apple_price * apples_sold)
                          (h2 : repair_ratio * (total_earnings * (1 - remaining_ratio)) = total_earnings * (1 - remaining_ratio)) :
  (total_earnings * (1 - remaining_ratio)) / repair_ratio = 80 := by sorry

end bike_cost_theorem_l65_6541


namespace maria_bottles_l65_6576

/-- The number of bottles Maria has at the end, given her initial number of bottles,
    the number she drinks, and the number she buys. -/
def final_bottles (initial : ℕ) (drunk : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Theorem stating that Maria ends up with 51 bottles given the problem conditions -/
theorem maria_bottles : final_bottles 14 8 45 = 51 := by
  sorry

end maria_bottles_l65_6576


namespace final_position_l65_6532

/-- Represents the position of the letter F -/
inductive Position
  | PositiveX_PositiveY
  | NegativeX_NegativeY
  | PositiveX_NegativeY
  | NegativeX_PositiveY
  | PositiveXPlusY
  | NegativeXPlusY
  | PositiveXMinusY
  | NegativeXMinusY

/-- Represents the transformations -/
inductive Transformation
  | RotateClockwise (angle : ℝ)
  | ReflectXAxis
  | RotateAroundOrigin (angle : ℝ)

/-- Initial position of F after 90° clockwise rotation -/
def initialPosition : Position := Position.PositiveX_NegativeY

/-- Sequence of transformations -/
def transformations : List Transformation := [
  Transformation.RotateClockwise 45,
  Transformation.ReflectXAxis,
  Transformation.RotateAroundOrigin 180
]

/-- Applies a single transformation to a position -/
def applyTransformation (p : Position) (t : Transformation) : Position :=
  sorry

/-- Applies a sequence of transformations to a position -/
def applyTransformations (p : Position) (ts : List Transformation) : Position :=
  sorry

/-- The final position theorem -/
theorem final_position :
  applyTransformations initialPosition transformations = Position.NegativeXPlusY :=
  sorry

end final_position_l65_6532


namespace cubic_equation_solutions_l65_6506

theorem cubic_equation_solutions :
  (¬ ∃ (x y : ℕ), x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) := by
  sorry

end cubic_equation_solutions_l65_6506


namespace margin_formula_l65_6564

theorem margin_formula (n : ℝ) (C S M : ℝ) 
  (h1 : n > 0) 
  (h2 : M = (2/n) * C) 
  (h3 : S - M = C) : 
  M = (2/(n+2)) * S := 
by sorry

end margin_formula_l65_6564


namespace fraction_to_decimal_l65_6545

theorem fraction_to_decimal : (53 : ℚ) / (2^2 * 5^3) = 0.106 := by sorry

end fraction_to_decimal_l65_6545


namespace existence_of_solution_l65_6505

theorem existence_of_solution :
  ∃ t : ℝ, Real.exp (1 - 2*t) = 3 * Real.sin (2*t - 2) + Real.cos (2*t) := by
  sorry

end existence_of_solution_l65_6505


namespace january_first_is_tuesday_l65_6592

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Main theorem: If January has 31 days, and there are exactly four Fridays and four Mondays, then January 1st is a Tuesday -/
theorem january_first_is_tuesday (jan : Month) :
  jan.days = 31 →
  countDayInMonth jan DayOfWeek.Friday = 4 →
  countDayInMonth jan DayOfWeek.Monday = 4 →
  jan.firstDay = DayOfWeek.Tuesday :=
sorry

end january_first_is_tuesday_l65_6592


namespace geometric_sequence_b6_l65_6563

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The theorem statement -/
theorem geometric_sequence_b6 (b : ℕ → ℝ) :
  geometric_sequence b → b 3 * b 9 = 9 → b 6 = 3 ∨ b 6 = -3 := by
  sorry

end geometric_sequence_b6_l65_6563


namespace martha_started_with_three_cards_l65_6543

/-- The number of cards Martha started with -/
def initial_cards : ℕ := sorry

/-- The number of cards Martha received from Emily -/
def cards_from_emily : ℕ := 76

/-- The total number of cards Martha ended up with -/
def total_cards : ℕ := 79

/-- Theorem stating that Martha started with 3 cards -/
theorem martha_started_with_three_cards : 
  initial_cards = 3 :=
by
  sorry

end martha_started_with_three_cards_l65_6543


namespace emily_sixth_score_l65_6561

def emily_scores : List ℕ := [91, 94, 86, 88, 101]

theorem emily_sixth_score (target_mean : ℕ := 94) (sixth_score : ℕ := 104) :
  let all_scores := emily_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℚ) = target_mean := by
  sorry

end emily_sixth_score_l65_6561


namespace blank_value_l65_6567

theorem blank_value : (6 : ℝ) / Real.sqrt 18 = Real.sqrt 2 := by
  sorry

end blank_value_l65_6567


namespace relationship_abc_l65_6584

theorem relationship_abc : 
  let a : ℝ := 2^(1/2)
  let b : ℝ := 3^(1/3)
  let c : ℝ := Real.log 2
  b > a ∧ a > c := by sorry

end relationship_abc_l65_6584


namespace expression_evaluation_l65_6557

theorem expression_evaluation : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := by
  sorry

end expression_evaluation_l65_6557


namespace carson_roller_coaster_rides_l65_6526

/-- Represents the carnival problem with given wait times and ride frequencies. -/
def carnival_problem (total_time roller_coaster_wait tilt_a_whirl_wait giant_slide_wait : ℕ)
  (tilt_a_whirl_rides giant_slide_rides : ℕ) : Prop :=
  ∃ (roller_coaster_rides : ℕ),
    roller_coaster_rides * roller_coaster_wait +
    tilt_a_whirl_rides * tilt_a_whirl_wait +
    giant_slide_rides * giant_slide_wait = total_time

/-- Theorem stating that Carson rides the roller coaster 4 times. -/
theorem carson_roller_coaster_rides :
  carnival_problem (4 * 60) 30 60 15 1 4 →
  ∃ (roller_coaster_rides : ℕ), roller_coaster_rides = 4 := by
  sorry


end carson_roller_coaster_rides_l65_6526


namespace apple_count_difference_l65_6595

theorem apple_count_difference (initial_green : ℕ) (delivered_green : ℕ) (final_difference : ℕ) : 
  initial_green = 32 →
  delivered_green = 340 →
  initial_green + delivered_green = initial_green + final_difference + 140 →
  ∃ (initial_red : ℕ), initial_red - initial_green = 200 :=
by sorry

end apple_count_difference_l65_6595


namespace odd_7x_plus_4_l65_6577

theorem odd_7x_plus_4 (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) := by
  sorry

end odd_7x_plus_4_l65_6577


namespace consecutive_odd_integers_l65_6521

theorem consecutive_odd_integers (x : ℤ) : 
  (x % 2 = 1) →                           -- x is odd
  ((x + 2) % 2 = 1) →                     -- x + 2 is odd
  ((x + 4) % 2 = 1) →                     -- x + 4 is odd
  ((x + 2) + (x + 4) = x + 17) →          -- sum of last two equals first plus 17
  (x + 4 = 15) :=                         -- third integer is 15
by sorry

end consecutive_odd_integers_l65_6521


namespace max_intersected_edges_l65_6525

/-- A regular p-gonal prism -/
structure RegularPrism (p : ℕ) :=
  (p_pos : p > 0)

/-- A plane that does not pass through the vertices of the prism -/
structure NonVertexPlane (p : ℕ) (prism : RegularPrism p) :=

/-- The number of edges of a regular p-gonal prism intersected by a plane -/
def intersected_edges (p : ℕ) (prism : RegularPrism p) (plane : NonVertexPlane p prism) : ℕ :=
  sorry

/-- The maximum number of edges that can be intersected is 3p -/
theorem max_intersected_edges (p : ℕ) (prism : RegularPrism p) :
  ∃ (plane : NonVertexPlane p prism), intersected_edges p prism plane = 3 * p ∧
  ∀ (other_plane : NonVertexPlane p prism), intersected_edges p prism other_plane ≤ 3 * p :=
sorry

end max_intersected_edges_l65_6525


namespace function_equation_solution_l65_6531

theorem function_equation_solution (a b : ℚ) :
  ∃ (f : ℚ → ℚ), (∀ x y : ℚ, f (x + a + f y) = f (x + b) + y) →
  ∃ A : ℚ, ∀ x : ℚ, f x = A * x + (a - b) / 2 := by
sorry

end function_equation_solution_l65_6531


namespace exactly_two_b_values_l65_6535

theorem exactly_two_b_values : 
  ∃! (s : Finset ℤ), 
    (∀ b ∈ s, ∃! (t : Finset ℤ), 
      (∀ x ∈ t, x^2 + b*x + 6 ≤ 0) ∧ 
      (∀ x ∉ t, x^2 + b*x + 6 > 0) ∧ 
      t.card = 3) ∧ 
    s.card = 2 := by
  sorry

end exactly_two_b_values_l65_6535


namespace seven_rows_of_ten_for_79_people_l65_6580

/-- Represents a seating arrangement with rows of either 9 or 10 people -/
structure SeatingArrangement where
  rows_of_9 : ℕ
  rows_of_10 : ℕ

/-- The total number of people in a seating arrangement -/
def total_people (s : SeatingArrangement) : ℕ :=
  9 * s.rows_of_9 + 10 * s.rows_of_10

/-- Theorem stating that for 79 people, there are 7 rows of 10 people -/
theorem seven_rows_of_ten_for_79_people :
  ∃ (s : SeatingArrangement), total_people s = 79 ∧ s.rows_of_10 = 7 := by
  sorry

end seven_rows_of_ten_for_79_people_l65_6580


namespace bagel_count_is_three_l65_6591

/-- Represents the number of items bought at each price point -/
structure PurchaseCount where
  sixtyCount : ℕ
  eightyCount : ℕ
  hundredCount : ℕ

/-- Calculates the total cost in cents for a given purchase count -/
def totalCost (p : PurchaseCount) : ℕ :=
  60 * p.sixtyCount + 80 * p.eightyCount + 100 * p.hundredCount

/-- Theorem stating that under the given conditions, the number of 80-cent items is 3 -/
theorem bagel_count_is_three :
  ∃ (p : PurchaseCount),
    p.sixtyCount + p.eightyCount + p.hundredCount = 5 ∧
    totalCost p = 400 ∧
    p.eightyCount = 3 :=
by
  sorry

end bagel_count_is_three_l65_6591


namespace min_value_of_expression_l65_6551

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  x + 4 / (x + 1) ≥ 3 ∧ ∃ y > 0, y + 4 / (y + 1) = 3 := by
  sorry

end min_value_of_expression_l65_6551


namespace sum_right_angles_rectangle_square_l65_6513

-- Define a rectangle
def Rectangle := Nat

-- Define a square
def Square := Nat

-- Define the number of right angles in a rectangle
def right_angles_rectangle (r : Rectangle) : Nat := 4

-- Define the number of right angles in a square
def right_angles_square (s : Square) : Nat := 4

-- Theorem: The sum of right angles in a rectangle and a square is 8
theorem sum_right_angles_rectangle_square (r : Rectangle) (s : Square) :
  right_angles_rectangle r + right_angles_square s = 8 := by
  sorry

end sum_right_angles_rectangle_square_l65_6513


namespace inequality_proof_l65_6587

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq : a + b + c = 3) : 
  (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) + 
  (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) + 
  (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤ 3 := by
sorry

end inequality_proof_l65_6587


namespace hyperbola_larger_y_focus_l65_6572

def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 10)^2 / 3^2 = 1

def is_focus (x y : ℝ) : Prop :=
  (x - 5)^2 + (y - 10)^2 = 58

def larger_y_focus (x y : ℝ) : Prop :=
  is_focus x y ∧ y > 10

theorem hyperbola_larger_y_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ larger_y_focus x y ∧ x = 5 ∧ y = 10 + Real.sqrt 58 :=
sorry

end hyperbola_larger_y_focus_l65_6572


namespace distance_home_to_school_l65_6582

/-- The distance between home and school given the travel conditions --/
theorem distance_home_to_school :
  ∀ (D T : ℝ),
  (3 * (T + 7/60) = D) →
  (6 * (T - 8/60) = D) →
  D = 1.5 := by
  sorry

end distance_home_to_school_l65_6582


namespace operation_simplification_l65_6538

theorem operation_simplification (x : ℚ) : 
  ((3 * x + 6) - 5 * x + 10) / 5 = -2/5 * x + 16/5 := by
  sorry

end operation_simplification_l65_6538


namespace tangent_points_sum_constant_l65_6524

/-- Parabola defined by x^2 = 4y -/
def Parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P with coordinates (a, -2) -/
def PointP (a : ℝ) : ℝ × ℝ := (a, -2)

/-- Tangent point on the parabola -/
def TangentPoint (x y : ℝ) : Prop := Parabola x y

/-- The theorem stating that for any point P(a, -2) and two tangent points A(x₁, y₁) and B(x₂, y₂) 
    on the parabola x^2 = 4y, the sum x₁x₂ + y₁y₂ is always equal to -4 -/
theorem tangent_points_sum_constant 
  (a x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : TangentPoint x₁ y₁) 
  (h₂ : TangentPoint x₂ y₂) : 
  x₁ * x₂ + y₁ * y₂ = -4 := by
  sorry

end tangent_points_sum_constant_l65_6524


namespace total_shells_eq_195_l65_6574

/-- The number of shells David has -/
def david_shells : ℕ := 15

/-- The number of shells Mia has -/
def mia_shells : ℕ := 4 * david_shells

/-- The number of shells Ava has -/
def ava_shells : ℕ := mia_shells + 20

/-- The number of shells Alice has -/
def alice_shells : ℕ := ava_shells / 2

/-- The total number of shells -/
def total_shells : ℕ := david_shells + mia_shells + ava_shells + alice_shells

theorem total_shells_eq_195 : total_shells = 195 := by
  sorry

end total_shells_eq_195_l65_6574


namespace neg_p_sufficient_not_necessary_for_neg_q_l65_6548

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) :=
sorry

end neg_p_sufficient_not_necessary_for_neg_q_l65_6548


namespace inverse_proportion_comparison_l65_6559

theorem inverse_proportion_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k > 0) 
  (h2 : y₁ = k / (-2)) 
  (h3 : y₂ = k / (-1)) : 
  y₁ > y₂ := by
  sorry

end inverse_proportion_comparison_l65_6559


namespace perpendicular_vectors_l65_6516

theorem perpendicular_vectors (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![4, -2]
  (∀ i, i < 2 → a i * b i = 0) → m = 2 := by
sorry

end perpendicular_vectors_l65_6516


namespace ratio_to_eleven_l65_6510

theorem ratio_to_eleven : ∃ x : ℚ, (5 : ℚ) / 1 = x / 11 ∧ x = 55 := by
  sorry

end ratio_to_eleven_l65_6510


namespace arcade_tickets_l65_6530

theorem arcade_tickets (initial_tickets yoyo_cost : ℝ) 
  (h1 : initial_tickets = 48.5)
  (h2 : yoyo_cost = 11.7) : 
  initial_tickets - (initial_tickets - yoyo_cost) = yoyo_cost := by
sorry

end arcade_tickets_l65_6530


namespace sugar_mixture_profit_l65_6500

/-- 
Proves that mixing 41.724 kg of sugar costing Rs. 9 per kg with 21.276 kg of sugar costing Rs. 7 per kg 
results in a 10% gain when selling the mixture at Rs. 9.24 per kg, given that the total weight of the mixture is 63 kg.
-/
theorem sugar_mixture_profit (
  total_weight : ℝ) 
  (sugar_a_cost sugar_b_cost selling_price : ℝ)
  (sugar_a_weight sugar_b_weight : ℝ) :
  total_weight = 63 →
  sugar_a_cost = 9 →
  sugar_b_cost = 7 →
  selling_price = 9.24 →
  sugar_a_weight = 41.724 →
  sugar_b_weight = 21.276 →
  sugar_a_weight + sugar_b_weight = total_weight →
  let total_cost := sugar_a_cost * sugar_a_weight + sugar_b_cost * sugar_b_weight
  let total_revenue := selling_price * total_weight
  total_revenue = 1.1 * total_cost :=
by sorry

end sugar_mixture_profit_l65_6500


namespace quadratic_inequalities_intersection_l65_6514

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

theorem quadratic_inequalities_intersection (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = A ∩ B) →
  a + b = -3 := by
  sorry

end quadratic_inequalities_intersection_l65_6514


namespace evening_temp_calculation_l65_6534

/-- Given a noon temperature and a temperature decrease, calculate the evening temperature. -/
def evening_temperature (noon_temp : Int) (decrease : Int) : Int :=
  noon_temp - decrease

/-- Theorem: If the noon temperature is 1℃ and it decreases by 3℃, then the evening temperature is -2℃. -/
theorem evening_temp_calculation :
  evening_temperature 1 3 = -2 := by
  sorry

end evening_temp_calculation_l65_6534


namespace trajectory_of_symmetric_point_l65_6536

/-- The trajectory of point P, symmetric to a point Q on the curve y = x^2 - 2 with respect to point A(1, 0) -/
theorem trajectory_of_symmetric_point :
  let A : ℝ × ℝ := (1, 0)
  let C : ℝ → ℝ := fun x => x^2 - 2
  ∀ Q : ℝ × ℝ, (Q.2 = C Q.1) →
  ∀ P : ℝ × ℝ, (Q.1 = 2 - P.1 ∧ Q.2 = -P.2) →
  P.2 = -P.1^2 + 4*P.1 - 2 :=
by sorry

end trajectory_of_symmetric_point_l65_6536


namespace prom_ticket_cost_l65_6502

def total_cost : ℝ := 836
def dinner_cost : ℝ := 120
def tip_percentage : ℝ := 0.30
def limo_cost_per_hour : ℝ := 80
def limo_rental_duration : ℝ := 6
def number_of_tickets : ℝ := 2

theorem prom_ticket_cost :
  let tip_cost := dinner_cost * tip_percentage
  let limo_total_cost := limo_cost_per_hour * limo_rental_duration
  let total_cost_without_tickets := dinner_cost + tip_cost + limo_total_cost
  let ticket_total_cost := total_cost - total_cost_without_tickets
  let ticket_cost := ticket_total_cost / number_of_tickets
  ticket_cost = 100 := by sorry

end prom_ticket_cost_l65_6502


namespace problem_solution_l65_6520

theorem problem_solution : (120 / (6 / 3)) * 2 = 120 := by
  sorry

end problem_solution_l65_6520


namespace max_value_of_expression_l65_6599

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c + a + c = b) :
  ∃ (p : ℝ), p = 2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1) ∧
  p ≤ 10/3 ∧ 
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' * b' * c' + a' + c' = b' ∧
    2 / (a'^2 + 1) - 2 / (b'^2 + 1) + 3 / (c'^2 + 1) = 10/3 :=
by sorry

end max_value_of_expression_l65_6599


namespace arcsin_cos_eq_x_div_3_solutions_l65_6593

theorem arcsin_cos_eq_x_div_3_solutions (x : Real) :
  -3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  (Real.arcsin (Real.cos x) = x / 3 ↔ (x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8)) :=
by sorry

end arcsin_cos_eq_x_div_3_solutions_l65_6593


namespace arithmetic_sequence_common_difference_l65_6555

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) :
  seq.d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l65_6555


namespace greatest_prime_factor_of_341_l65_6550

theorem greatest_prime_factor_of_341 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 341 ∧ p = 19 ∧ ∀ (q : ℕ), q.Prime → q ∣ 341 → q ≤ p :=
by sorry

end greatest_prime_factor_of_341_l65_6550


namespace geometric_sequence_solution_l65_6590

/-- A geometric sequence with three consecutive terms x, 2x+2, and 3x+3 has x = -4 -/
theorem geometric_sequence_solution (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → x = -4 :=
by
  sorry


end geometric_sequence_solution_l65_6590


namespace value_of_a_minus_b_l65_6503

theorem value_of_a_minus_b (a b : ℝ) 
  (eq1 : 2010 * a + 2014 * b = 2018)
  (eq2 : 2012 * a + 2016 * b = 2020) : 
  a - b = -3 := by
  sorry

end value_of_a_minus_b_l65_6503


namespace fridays_in_non_leap_year_starting_saturday_l65_6519

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year where
  isLeapYear : Bool
  firstDayOfYear : DayOfWeek

/-- Counts the number of occurrences of a specific day in a year -/
def countDaysInYear (y : Year) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: In a non-leap year where January 1st is a Saturday, there are 52 Fridays -/
theorem fridays_in_non_leap_year_starting_saturday (y : Year) 
  (h1 : y.isLeapYear = false) 
  (h2 : y.firstDayOfYear = DayOfWeek.Saturday) : 
  countDaysInYear y DayOfWeek.Friday = 52 :=
by sorry

end fridays_in_non_leap_year_starting_saturday_l65_6519


namespace cubic_one_real_solution_l65_6527

/-- The cubic equation 4x^3 + 9x^2 + kx + 4 = 0 has exactly one real solution if and only if k = 6.75 -/
theorem cubic_one_real_solution (k : ℝ) : 
  (∃! x : ℝ, 4 * x^3 + 9 * x^2 + k * x + 4 = 0) ↔ k = 27/4 := by
sorry

end cubic_one_real_solution_l65_6527


namespace square_perimeter_ratio_l65_6598

theorem square_perimeter_ratio (s S : ℝ) (hs : s > 0) (hS : S > 0) : 
  S * Real.sqrt 2 = 7 * (s * Real.sqrt 2) → 
  (4 * S) / (4 * s) = 7 := by sorry

end square_perimeter_ratio_l65_6598


namespace square_plus_abs_zero_implies_both_zero_l65_6540

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : 
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_plus_abs_zero_implies_both_zero_l65_6540


namespace ab_plus_cd_equals_45_l65_6537

theorem ab_plus_cd_equals_45 (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = 14) :
  a * b + c * d = 45 := by
sorry

end ab_plus_cd_equals_45_l65_6537


namespace unique_line_through_sqrt3_and_rationals_l65_6512

-- Define a point in R²
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through (√3, 0)
structure Line where
  slope : ℝ

def isRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

def linePassesThroughRationalPoints (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ isRational p1.x ∧ isRational p1.y ∧ 
                     isRational p2.x ∧ isRational p2.y ∧
                     p1.y = l.slope * (p1.x - Real.sqrt 3) ∧
                     p2.y = l.slope * (p2.x - Real.sqrt 3)

theorem unique_line_through_sqrt3_and_rationals :
  ∃! (l : Line), linePassesThroughRationalPoints l :=
sorry

end unique_line_through_sqrt3_and_rationals_l65_6512


namespace largest_b_value_l65_6533

theorem largest_b_value (b : ℝ) (h : (3*b + 6)*(b - 2) = 9*b) : b ≤ 4 :=
by sorry

end largest_b_value_l65_6533


namespace smallest_b_value_l65_6509

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 10) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 25) :
  ∀ b' : ℕ+, b'.val < b.val → 
    ¬(∃ a' : ℕ+, a'.val - b'.val = 10 ∧ 
      Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val * b'.val) = 25) :=
by sorry

end smallest_b_value_l65_6509


namespace inequality_properties_l65_6588

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := a * (x - 1) * (x - 3) + 2 > 0

def solution_set (x₁ x₂ : ℝ) : Set ℝ :=
  {x | x < x₁ ∨ x > x₂}

-- State the theorem
theorem inequality_properties
  (a x₁ x₂ : ℝ)
  (h_solution : ∀ x, inequality a x ↔ x ∈ solution_set x₁ x₂)
  (h_order : x₁ < x₂) :
  (x₁ + x₂ = 4) ∧
  (3 < x₁ * x₂ ∧ x₁ * x₂ < 4) ∧
  (∀ x, (3*a + 2) * x^2 - 4*a*x + a < 0 ↔ 1/x₂ < x ∧ x < 1/x₁) :=
by sorry

end inequality_properties_l65_6588


namespace min_cooking_time_is_12_l65_6589

/-- Represents the time taken for each step in the cooking process -/
structure CookingSteps where
  step1 : ℕ  -- Wash pot and fill with water
  step2 : ℕ  -- Wash vegetables
  step3 : ℕ  -- Prepare noodles and seasonings
  step4 : ℕ  -- Boil water
  step5 : ℕ  -- Cook noodles and vegetables

/-- Calculates the minimum cooking time given the cooking steps -/
def minCookingTime (steps : CookingSteps) : ℕ :=
  steps.step1 + max steps.step4 (steps.step2 + steps.step3) + steps.step5

/-- Theorem stating that the minimum cooking time is 12 minutes -/
theorem min_cooking_time_is_12 (steps : CookingSteps) 
  (h1 : steps.step1 = 2)
  (h2 : steps.step2 = 3)
  (h3 : steps.step3 = 2)
  (h4 : steps.step4 = 7)
  (h5 : steps.step5 = 3) :
  minCookingTime steps = 12 := by
  sorry


end min_cooking_time_is_12_l65_6589


namespace jebbs_take_home_pay_l65_6558

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Theorem stating that given a total pay of 650 and a tax rate of 10%, the take-home pay is 585 -/
theorem jebbs_take_home_pay :
  takeHomePay 650 0.1 = 585 := by
  sorry

end jebbs_take_home_pay_l65_6558


namespace square_side_lengths_l65_6568

theorem square_side_lengths (a b : ℝ) (h1 : a > b) (h2 : a - b = 2) (h3 : a^2 - b^2 = 40) : 
  a = 11 ∧ b = 9 := by
sorry

end square_side_lengths_l65_6568


namespace imaginary_part_of_complex_fraction_l65_6552

theorem imaginary_part_of_complex_fraction :
  Complex.im ((4 - 5 * Complex.I) / Complex.I) = -4 := by
  sorry

end imaginary_part_of_complex_fraction_l65_6552


namespace angle_inequality_l65_6556

theorem angle_inequality : 
  let a := (2 * Real.tan (22.5 * π / 180)) / (1 - Real.tan (22.5 * π / 180) ^ 2)
  let b := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  c < b ∧ b < a := by sorry

end angle_inequality_l65_6556


namespace f_minus_six_equals_minus_one_l65_6517

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_minus_six_equals_minus_one 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 6)
  (h3 : ∀ x ∈ Set.Icc (-3) 3, f x = (x + 1) * (x - a)) :
  f (-6) = -1 := by
sorry

end f_minus_six_equals_minus_one_l65_6517


namespace square_area_from_vertices_l65_6560

/-- The area of a square with adjacent vertices at (1,5) and (4,-2) is 58 -/
theorem square_area_from_vertices : 
  let x1 : ℝ := 1
  let y1 : ℝ := 5
  let x2 : ℝ := 4
  let y2 : ℝ := -2
  let side_length : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let area : ℝ := side_length^2
  area = 58 := by sorry

end square_area_from_vertices_l65_6560


namespace quantity_cost_relation_l65_6554

theorem quantity_cost_relation (Q : ℝ) (h1 : Q * 20 = 1) (h2 : 3.5 * Q * 28 = 1) :
  20 / 8 = 2.5 := by sorry

end quantity_cost_relation_l65_6554


namespace polynomial_uniqueness_l65_6571

def is_valid_polynomial (P : ℝ → ℝ) (n : ℕ) : Prop :=
  (∀ k : ℕ, k ≤ n → P (2 * k) = 0) ∧
  (∀ k : ℕ, k < n → P (2 * k + 1) = 2) ∧
  (P (2 * n + 1) = -6)

theorem polynomial_uniqueness (P : ℝ → ℝ) (n : ℕ) :
  is_valid_polynomial P n →
  (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) →
  (n = 1 ∧ ∀ x, P x = -2 * x^2 + 4 * x) :=
sorry

end polynomial_uniqueness_l65_6571


namespace trash_cans_on_streets_l65_6542

theorem trash_cans_on_streets (street_cans back_cans : ℕ) : 
  back_cans = 2 * street_cans → 
  street_cans + back_cans = 42 → 
  street_cans = 14 := by
sorry

end trash_cans_on_streets_l65_6542


namespace even_sum_squares_half_l65_6544

theorem even_sum_squares_half (n x y : ℤ) (h : 2 * n = x^2 + y^2) :
  n = ((x + y) / 2)^2 + ((x - y) / 2)^2 := by
  sorry

end even_sum_squares_half_l65_6544


namespace expression_comparison_l65_6586

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ¬(∀ x y : ℝ, x > 0 → y > 0 → x ≠ y →
    ((x + 1/x) * (y + 1/y) > (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 ∧
     (x + 1/x) * (y + 1/y) > ((x + y)/2 + 2/(x + y))^2) ∨
    ((Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 > (x + 1/x) * (y + 1/y) ∧
     (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 > ((x + y)/2 + 2/(x + y))^2) ∨
    (((x + y)/2 + 2/(x + y))^2 > (x + 1/x) * (y + 1/y) ∧
     ((x + y)/2 + 2/(x + y))^2 > (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2)) :=
by sorry

end expression_comparison_l65_6586


namespace x_minus_y_equals_one_l65_6578

theorem x_minus_y_equals_one (x y : ℝ) 
  (h1 : x^2 + y^2 = 25) 
  (h2 : x + y = 7) 
  (h3 : x > y) : 
  x - y = 1 := by
sorry

end x_minus_y_equals_one_l65_6578


namespace polynomial_remainder_l65_6515

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 7, where the remainder when divided by x - 2 is 21,
    the remainder when divided by x + 2 is 21 - 2F -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x ↦ D * x^4 + E * x^2 + F * x + 7
  (q 2 = 21) → 
  ∃ r : ℝ, ∀ x : ℝ, ∃ k : ℝ, q x = (x + 2) * k + r ∧ r = 21 - 2 * F :=
by sorry

end polynomial_remainder_l65_6515


namespace celebration_attendees_l65_6597

theorem celebration_attendees (men : ℕ) (women : ℕ) : 
  men = 15 →
  men * 4 = women * 3 →
  women = 20 :=
by sorry

end celebration_attendees_l65_6597


namespace number_problem_l65_6507

theorem number_problem : ∃ n : ℤ, n - 44 = 15 ∧ n = 59 := by
  sorry

end number_problem_l65_6507


namespace investment_rate_problem_l65_6504

/-- Proves that given the specified conditions, the unknown interest rate is 5% -/
theorem investment_rate_problem (total : ℝ) (first_part : ℝ) (first_rate : ℝ) (total_interest : ℝ)
  (h1 : total = 4000)
  (h2 : first_part = 2800)
  (h3 : first_rate = 3)
  (h4 : total_interest = 144)
  (h5 : first_part * (first_rate / 100) + (total - first_part) * (unknown_rate / 100) = total_interest) :
  unknown_rate = 5 := by
  sorry

end investment_rate_problem_l65_6504


namespace expansion_terms_count_l65_6562

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_in_expansion (n m : ℕ) : ℕ := n * m

/-- The first factor (a+b+c+d) has 4 terms -/
def first_factor_terms : ℕ := 4

/-- The second factor (e+f+g+h+i) has 5 terms -/
def second_factor_terms : ℕ := 5

theorem expansion_terms_count :
  num_terms_in_expansion first_factor_terms second_factor_terms = 20 := by
  sorry

end expansion_terms_count_l65_6562


namespace solution_set_quadratic_inequality_l65_6523

theorem solution_set_quadratic_inequality :
  {x : ℝ | 4 - x^2 < 0} = Set.Ioi 2 ∪ Set.Iio (-2) :=
by sorry

end solution_set_quadratic_inequality_l65_6523


namespace remainder_theorem_l65_6528

-- Define the polynomial and its divisor
def p (x : ℝ) : ℝ := 3*x^7 + 2*x^5 - 5*x^3 + x^2 - 9
def d (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the remainder
def r (x : ℝ) : ℝ := 14*x - 16

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end remainder_theorem_l65_6528


namespace trailing_zeros_1_to_20_l65_6575

/-- The number of factors of 5 in n! -/
def count_factors_of_5 (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The number of trailing zeros in the product of factorials from 1 to n -/
def trailing_zeros_factorial_product (n : ℕ) : ℕ :=
  count_factors_of_5 n

theorem trailing_zeros_1_to_20 :
  trailing_zeros_factorial_product 20 = 8 ∧
  trailing_zeros_factorial_product 20 % 100 = 8 := by
  sorry

end trailing_zeros_1_to_20_l65_6575


namespace mathematician_meeting_theorem_l65_6518

theorem mathematician_meeting_theorem (n p q r : ℕ) (h1 : n = p - q * Real.sqrt r) 
  (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : ∀ (prime : ℕ), Prime prime → ¬(prime^2 ∣ r)) 
  (h4 : ((120 - n : ℝ) / 120)^2 = 1/2) : p + q + r = 182 := by
sorry

end mathematician_meeting_theorem_l65_6518


namespace johns_score_increase_l65_6573

/-- Given John's four test scores, prove that the difference between
    the average of all four scores and the average of the first three scores is 0.92. -/
theorem johns_score_increase (score1 score2 score3 score4 : ℚ) 
    (h1 : score1 = 92)
    (h2 : score2 = 89)
    (h3 : score3 = 93)
    (h4 : score4 = 95) :
    (score1 + score2 + score3 + score4) / 4 - (score1 + score2 + score3) / 3 = 92 / 100 := by
  sorry

end johns_score_increase_l65_6573


namespace geometric_sequence_problem_l65_6508

/-- Given a geometric sequence {a_n} where a_1 and a_5 are the positive roots of x^2 - 10x + 16 = 0, prove that a_3 = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (a 1 * a 1 - 10 * a 1 + 16 = 0) →  -- a_1 is a root of x^2 - 10x + 16 = 0
  (a 5 * a 5 - 10 * a 5 + 16 = 0) →  -- a_5 is a root of x^2 - 10x + 16 = 0
  (0 < a 1) →  -- a_1 is positive
  (0 < a 5) →  -- a_5 is positive
  a 3 = 4 := by
sorry

end geometric_sequence_problem_l65_6508


namespace line_passes_through_parabola_vertex_l65_6579

/-- The number of values of a for which the line y = ax + a passes through the vertex of the parabola y = x^2 + ax -/
theorem line_passes_through_parabola_vertex : 
  ∃! (s : Finset ℝ), (∀ a ∈ s, ∃ x y : ℝ, 
    (y = a*x + a) ∧ 
    (y = x^2 + a*x) ∧ 
    (∀ x' y' : ℝ, y' = x'^2 + a*x' → y' ≥ y)) ∧ 
  Finset.card s = 2 := by
sorry

end line_passes_through_parabola_vertex_l65_6579


namespace adams_trivia_score_l65_6570

/-- Adam's trivia game score calculation -/
theorem adams_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
    first_half = 8 →
    second_half = 2 →
    points_per_question = 8 →
    (first_half + second_half) * points_per_question = 80 :=
by
  sorry

end adams_trivia_score_l65_6570


namespace right_triangle_hypotenuse_l65_6565

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 20 → 
    b = 21 → 
    c^2 = a^2 + b^2 → 
    c = 29 :=
by
  sorry

end right_triangle_hypotenuse_l65_6565


namespace basketball_game_score_theorem_l65_6583

/-- Represents a team's scores for each quarter -/
structure TeamScores :=
  (q1 : ℕ) (q2 : ℕ) (q3 : ℕ) (q4 : ℕ)

/-- Checks if a sequence of four numbers is an arithmetic sequence -/
def isArithmeticSequence (s : TeamScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3 ∧ s.q2 > s.q1

/-- Checks if a sequence of four numbers is a geometric sequence -/
def isGeometricSequence (s : TeamScores) : Prop :=
  s.q2 / s.q1 = s.q3 / s.q2 ∧ s.q3 / s.q2 = s.q4 / s.q3 ∧ s.q2 > s.q1

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_score_theorem 
  (eagles lions : TeamScores) : 
  isArithmeticSequence eagles →
  isGeometricSequence lions →
  eagles.q1 = lions.q1 + 2 →
  eagles.q1 + eagles.q2 + eagles.q3 = lions.q1 + lions.q2 + lions.q3 →
  totalScore eagles ≤ 100 →
  totalScore lions ≤ 100 →
  totalScore eagles + totalScore lions = 144 := by
  sorry

#check basketball_game_score_theorem

end basketball_game_score_theorem_l65_6583


namespace emily_necklaces_l65_6566

def beads_per_necklace : ℕ := 8
def total_beads : ℕ := 16

theorem emily_necklaces :
  total_beads / beads_per_necklace = 2 := by
  sorry

end emily_necklaces_l65_6566


namespace same_solutions_implies_a_equals_four_l65_6553

theorem same_solutions_implies_a_equals_four :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 - a = 0 ↔ 3*x^4 - 48 = 0) →
  a = 4 := by
sorry

end same_solutions_implies_a_equals_four_l65_6553


namespace max_questions_l65_6546

/-- Represents a contestant's answers to n questions -/
def Answers (n : ℕ) := Fin n → Bool

/-- The number of contestants -/
def num_contestants : ℕ := 8

/-- Condition: For any pair of questions, exactly two contestants answered each combination -/
def valid_distribution (n : ℕ) (answers : Fin num_contestants → Answers n) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = true ∧ answers k j = true) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = false ∧ answers k j = false) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = true ∧ answers k j = false) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = false ∧ answers k j = true)

/-- The maximum number of questions satisfying the conditions -/
theorem max_questions :
  ∀ n : ℕ, (∃ answers : Fin num_contestants → Answers n, valid_distribution n answers) →
    n ≤ 7 :=
sorry

end max_questions_l65_6546


namespace union_of_sets_l65_6596

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4}
  A ∪ B = {1, 2, 4} := by
sorry

end union_of_sets_l65_6596


namespace function_property_l65_6585

variable (f : ℝ → ℝ)
variable (p q : ℝ)

theorem function_property
  (h1 : ∀ a b, f (a * b) = f a + f b)
  (h2 : f 2 = p)
  (h3 : f 3 = q) :
  f 12 = 2 * p + q :=
by sorry

end function_property_l65_6585


namespace purely_imaginary_condition_fourth_quadrant_condition_l65_6529

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6)

theorem purely_imaginary_condition (a : ℝ) : 
  (z a).re = 0 ∧ (z a).im ≠ 0 → a = 1 := by sorry

theorem fourth_quadrant_condition (a : ℝ) :
  (z a).re > 0 ∧ (z a).im < 0 → -1 < a ∧ a < 1 := by sorry

end purely_imaginary_condition_fourth_quadrant_condition_l65_6529


namespace least_number_of_marbles_l65_6522

def is_divisible_by_all (n : ℕ) : Prop :=
  ∀ i ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ), n % i = 0

theorem least_number_of_marbles :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_all n ∧ ∀ m, 0 < m ∧ m < n → ¬is_divisible_by_all m :=
by
  use 840
  sorry

end least_number_of_marbles_l65_6522


namespace complex_number_in_first_quadrant_l65_6549

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.mk a b) = (1 : ℂ) / (1 - Complex.I) := by
  sorry

end complex_number_in_first_quadrant_l65_6549


namespace fifth_friend_contribution_l65_6501

def friend_contribution (total : ℝ) (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = total ∧
  a = (1/2) * (b + c + d + e) ∧
  b = (1/3) * (a + c + d + e) ∧
  c = (1/4) * (a + b + d + e) ∧
  d = (1/5) * (a + b + c + e)

theorem fifth_friend_contribution :
  ∃ a b c d : ℝ, friend_contribution 120 a b c d 52.55 := by
  sorry

end fifth_friend_contribution_l65_6501


namespace college_student_count_l65_6511

/-- Given a college with a ratio of boys to girls of 8:5 and 160 girls, 
    the total number of students is 416. -/
theorem college_student_count 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8 ∧ ratio_girls = 5)
  (h_girls : num_girls = 160) : 
  (ratio_boys * num_girls / ratio_girls + num_girls : ℕ) = 416 := by
  sorry

#check college_student_count

end college_student_count_l65_6511


namespace arithmetic_sequence_fifth_term_l65_6569

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term is 123/40 -/
theorem arithmetic_sequence_fifth_term
  (x y : ℚ)
  (h1 : (x + y) - (x - y) = (x - y) - (x * y))
  (h2 : (x - y) - (x * y) = (x * y) - (x / y))
  (h3 : y ≠ 0)
  : (x / y) + ((x / y) - (x * y)) = 123/40 := by
  sorry

end arithmetic_sequence_fifth_term_l65_6569


namespace djibo_sister_age_djibo_sister_age_is_28_l65_6539

theorem djibo_sister_age (djibo_current_age : ℕ) (sum_five_years_ago : ℕ) : ℕ :=
  let djibo_age_five_years_ago := djibo_current_age - 5
  let sister_age_five_years_ago := sum_five_years_ago - djibo_age_five_years_ago
  sister_age_five_years_ago + 5

/-- Given Djibo's current age and the sum of his and his sister's ages five years ago,
    prove that Djibo's sister's current age is 28. -/
theorem djibo_sister_age_is_28 :
  djibo_sister_age 17 35 = 28 := by
  sorry

end djibo_sister_age_djibo_sister_age_is_28_l65_6539


namespace intersection_line_circle_l65_6547

theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 - A.2 + 3 = 0 ∧ (A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧
    (a * B.1 - B.2 + 3 = 0 ∧ (B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  a = 0 := by
sorry

end intersection_line_circle_l65_6547
