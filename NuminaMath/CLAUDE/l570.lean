import Mathlib

namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l570_57022

theorem fourth_root_equation_solution :
  ∃! x : ℚ, (62 - 3*x)^(1/4) + (38 + 3*x)^(1/4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l570_57022


namespace NUMINAMATH_CALUDE_tim_and_tina_same_age_l570_57002

def tim_age_condition (x : ℕ) : Prop := x + 2 = 2 * (x - 2)

def tina_age_condition (y : ℕ) : Prop := y + 3 = 3 * (y - 3)

theorem tim_and_tina_same_age :
  ∃ (x y : ℕ), tim_age_condition x ∧ tina_age_condition y ∧ x = y :=
by
  sorry

end NUMINAMATH_CALUDE_tim_and_tina_same_age_l570_57002


namespace NUMINAMATH_CALUDE_unique_solution_mn_l570_57085

theorem unique_solution_mn : ∃! (m n : ℕ+), 18 * m * n = 63 - 9 * m - 3 * n ∧ m = 7 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l570_57085


namespace NUMINAMATH_CALUDE_horseshoe_profit_800_sets_l570_57078

/-- Calculates the profit for horseshoe manufacturing given the specified conditions --/
def horseshoe_profit (
  initial_outlay : ℕ)
  (cost_first_300 : ℕ)
  (cost_beyond_300 : ℕ)
  (price_first_400 : ℕ)
  (price_beyond_400 : ℕ)
  (total_sets : ℕ) : ℕ :=
  let manufacturing_cost := initial_outlay +
    (min total_sets 300) * cost_first_300 +
    (max (total_sets - 300) 0) * cost_beyond_300
  let revenue := (min total_sets 400) * price_first_400 +
    (max (total_sets - 400) 0) * price_beyond_400
  revenue - manufacturing_cost

theorem horseshoe_profit_800_sets :
  horseshoe_profit 10000 20 15 50 45 800 = 14500 := by
  sorry

end NUMINAMATH_CALUDE_horseshoe_profit_800_sets_l570_57078


namespace NUMINAMATH_CALUDE_terminating_decimal_expansion_of_7_625_l570_57064

theorem terminating_decimal_expansion_of_7_625 :
  (7 : ℚ) / 625 = (112 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_terminating_decimal_expansion_of_7_625_l570_57064


namespace NUMINAMATH_CALUDE_sandy_clothes_spending_l570_57075

theorem sandy_clothes_spending (shorts_cost shirt_cost jacket_cost total_cost : ℝ) 
  (h1 : shorts_cost = 13.99)
  (h2 : shirt_cost = 12.14)
  (h3 : jacket_cost = 7.43)
  (h4 : total_cost = shorts_cost + shirt_cost + jacket_cost) :
  total_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_spending_l570_57075


namespace NUMINAMATH_CALUDE_circle_equation_l570_57012

/-- A circle passing through points A(0, -6) and B(1, -5) with center C on the line x-y+1=0 
    has the standard equation (x + 3)^2 + (y + 2)^2 = 25 -/
theorem circle_equation (C : ℝ × ℝ) : 
  (C.1 - C.2 + 1 = 0) →  -- C lies on the line x-y+1=0
  ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 = ((1 : ℝ) - C.1)^2 + ((-5 : ℝ) - C.2)^2 →  -- C is equidistant from A and B
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l570_57012


namespace NUMINAMATH_CALUDE_inverse_sum_mod_thirteen_l570_57034

theorem inverse_sum_mod_thirteen : 
  (((3⁻¹ : ZMod 13) + (4⁻¹ : ZMod 13) + (5⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_thirteen_l570_57034


namespace NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l570_57032

def abcd_plus_dcba (a : ℕ) : ℕ := 4201 * a + 12606

def number_set : Set ℕ := {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ n = abcd_plus_dcba a}

theorem gcd_of_abcd_plus_dcba :
  ∃ g : ℕ, g > 0 ∧ (∀ n ∈ number_set, g ∣ n) ∧
  (∀ d : ℕ, d > 0 → (∀ n ∈ number_set, d ∣ n) → d ≤ g) ∧
  g = 4201 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l570_57032


namespace NUMINAMATH_CALUDE_unique_condition_implies_sum_l570_57017

-- Define the set of possible values
def S : Set ℕ := {1, 2, 5}

-- Define the conditions
def condition1 (a b c : ℕ) : Prop := a ≠ 5
def condition2 (a b c : ℕ) : Prop := b = 5
def condition3 (a b c : ℕ) : Prop := c ≠ 2

-- Main theorem
theorem unique_condition_implies_sum (a b c : ℕ) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  (condition1 a b c ∨ condition2 a b c ∨ condition3 a b c) →
  (¬condition1 a b c ∨ ¬condition2 a b c) →
  (¬condition1 a b c ∨ ¬condition3 a b c) →
  (¬condition2 a b c ∨ ¬condition3 a b c) →
  100 * a + 10 * b + c = 521 :=
by sorry

end NUMINAMATH_CALUDE_unique_condition_implies_sum_l570_57017


namespace NUMINAMATH_CALUDE_inequality_solution_l570_57031

theorem inequality_solution (x : ℝ) : 
  (Real.sqrt (x^3 - 18*x - 5) + 2) * abs (x^3 - 4*x^2 - 5*x + 18) ≤ 0 ↔ x = 1 - Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l570_57031


namespace NUMINAMATH_CALUDE_correct_selection_methods_l570_57073

def total_people : ℕ := 16
def people_per_class : ℕ := 4
def num_classes : ℕ := 4
def people_to_select : ℕ := 3

def selection_methods : ℕ := sorry

theorem correct_selection_methods :
  selection_methods = 472 := by sorry

end NUMINAMATH_CALUDE_correct_selection_methods_l570_57073


namespace NUMINAMATH_CALUDE_table_price_is_56_l570_57000

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of some chairs and 2 tables -/
axiom price_ratio : ∃ x : ℝ, 2 * chair_price + table_price = 0.6 * (x * chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $64 -/
axiom total_price : chair_price + table_price = 64

/-- Theorem stating that the price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by sorry

end NUMINAMATH_CALUDE_table_price_is_56_l570_57000


namespace NUMINAMATH_CALUDE_lucas_payment_l570_57069

/-- Calculates the payment for window cleaning based on given conditions --/
def calculate_payment (stories : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
  (deduction_per_2_days : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := stories * windows_per_floor
  let base_payment := total_windows * payment_per_window
  let time_deductions := (days_taken / 2) * deduction_per_2_days
  base_payment - time_deductions

/-- Theorem stating that Lucas' father will pay him $33 --/
theorem lucas_payment :
  calculate_payment 4 5 2 1 14 = 33 := by
  sorry

end NUMINAMATH_CALUDE_lucas_payment_l570_57069


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l570_57070

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through (-3,0) and (4,0),
    prove that the solutions to a(x-1)^2 + c = b - bx are -2 and 5. -/
theorem quadratic_equation_solutions 
  (a b c : ℝ) 
  (h1 : a * (-3)^2 + b * (-3) + c = 0)  -- f(-3) = 0
  (h2 : a * 4^2 + b * 4 + c = 0)        -- f(4) = 0
  : ∃ (x1 x2 : ℝ), x1 = -2 ∧ x2 = 5 ∧ 
    ∀ (x : ℝ), a * (x - 1)^2 + c = b - b * x ↔ x = x1 ∨ x = x2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l570_57070


namespace NUMINAMATH_CALUDE_marie_profit_l570_57094

def total_loaves : ℕ := 60
def cost_per_loaf : ℚ := 1
def morning_price : ℚ := 3
def afternoon_discount : ℚ := 0.25
def donated_loaves : ℕ := 5

def morning_sales : ℕ := total_loaves / 3
def remaining_after_morning : ℕ := total_loaves - morning_sales
def afternoon_sales : ℕ := remaining_after_morning / 2
def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_sales
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

def afternoon_price : ℚ := morning_price * (1 - afternoon_discount)

def total_revenue : ℚ := morning_sales * morning_price + afternoon_sales * afternoon_price
def total_cost : ℚ := total_loaves * cost_per_loaf
def profit : ℚ := total_revenue - total_cost

theorem marie_profit : profit = 45 := by
  sorry

end NUMINAMATH_CALUDE_marie_profit_l570_57094


namespace NUMINAMATH_CALUDE_johns_weekly_earnings_l570_57047

/-- Calculates John's total earnings per week from crab fishing --/
theorem johns_weekly_earnings :
  let monday_baskets : ℕ := 3
  let thursday_baskets : ℕ := 4
  let small_crabs_per_basket : ℕ := 4
  let large_crabs_per_basket : ℕ := 5
  let small_crab_price : ℕ := 3
  let large_crab_price : ℕ := 5

  let monday_crabs : ℕ := monday_baskets * small_crabs_per_basket
  let thursday_crabs : ℕ := thursday_baskets * large_crabs_per_basket

  let monday_earnings : ℕ := monday_crabs * small_crab_price
  let thursday_earnings : ℕ := thursday_crabs * large_crab_price

  let total_earnings : ℕ := monday_earnings + thursday_earnings

  total_earnings = 136 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_earnings_l570_57047


namespace NUMINAMATH_CALUDE_rabbits_ate_three_watermelons_l570_57010

/-- The number of watermelons eaten by rabbits -/
def watermelons_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem: Given that Sam initially grew 4 watermelons and now has 1 left,
    prove that rabbits ate 3 watermelons -/
theorem rabbits_ate_three_watermelons :
  watermelons_eaten 4 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_ate_three_watermelons_l570_57010


namespace NUMINAMATH_CALUDE_car_downhill_speed_l570_57063

/-- Proves that given specific conditions about a car's journey, the downhill speed is 60 km/hr -/
theorem car_downhill_speed 
  (uphill_speed : ℝ) 
  (uphill_distance : ℝ) 
  (downhill_distance : ℝ) 
  (average_speed : ℝ) 
  (h1 : uphill_speed = 30) 
  (h2 : uphill_distance = 100) 
  (h3 : downhill_distance = 50) 
  (h4 : average_speed = 36) : 
  ∃ downhill_speed : ℝ, 
    downhill_speed = 60 ∧ 
    average_speed = (uphill_distance + downhill_distance) / 
      (uphill_distance / uphill_speed + downhill_distance / downhill_speed) := by
  sorry

#check car_downhill_speed

end NUMINAMATH_CALUDE_car_downhill_speed_l570_57063


namespace NUMINAMATH_CALUDE_decimal_to_binary_l570_57071

theorem decimal_to_binary (n : Nat) : n = 23 → ToString.toString n = "10111" := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_l570_57071


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l570_57092

theorem square_area_equal_perimeter_triangle (a b c : ℝ) (square_side : ℝ) : 
  a = 5.8 ∧ b = 7.5 ∧ c = 10.7 →
  4 * square_side = a + b + c →
  square_side ^ 2 = 36 := by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l570_57092


namespace NUMINAMATH_CALUDE_reflection_composition_l570_57080

/-- Two lines in the xy-plane that intersect at the origin -/
structure IntersectingLines where
  ℓ₁ : Set (ℝ × ℝ)
  ℓ₂ : Set (ℝ × ℝ)
  intersect_origin : (0, 0) ∈ ℓ₁ ∩ ℓ₂

/-- A point in the xy-plane -/
def Point := ℝ × ℝ

/-- Reflection of a point over a line -/
def reflect (p : Point) (ℓ : Set Point) : Point := sorry

theorem reflection_composition 
  (lines : IntersectingLines)
  (Q : Point)
  (h₁ : Q = (-2, 3))
  (h₂ : lines.ℓ₁ = {(x, y) | 3 * x - y = 0})
  (h₃ : reflect (reflect Q lines.ℓ₁) lines.ℓ₂ = (5, -2)) :
  lines.ℓ₂ = {(x, y) | x + 4 * y = 0} := by
  sorry

end NUMINAMATH_CALUDE_reflection_composition_l570_57080


namespace NUMINAMATH_CALUDE_mean_weight_of_participants_l570_57081

/-- Represents a stem and leaf plot entry -/
structure StemLeafEntry :=
  (stem : ℕ)
  (leaves : List ℕ)

/-- Calculates the sum of weights from a stem and leaf entry -/
def sumWeights (entry : StemLeafEntry) : ℕ :=
  entry.leaves.sum + entry.stem * 100 * entry.leaves.length

/-- Calculates the number of participants from a stem and leaf entry -/
def countParticipants (entry : StemLeafEntry) : ℕ :=
  entry.leaves.length

theorem mean_weight_of_participants (data : List StemLeafEntry) 
  (h1 : data = [
    ⟨12, [3, 5]⟩, 
    ⟨13, [0, 2, 3, 5, 7, 8]⟩, 
    ⟨14, [1, 5, 5, 9, 9]⟩, 
    ⟨15, [0, 2, 3, 5, 8]⟩, 
    ⟨16, [4, 7, 7, 9]⟩
  ]) : 
  (data.map sumWeights).sum / (data.map countParticipants).sum = 3217 / 22 := by
  sorry

end NUMINAMATH_CALUDE_mean_weight_of_participants_l570_57081


namespace NUMINAMATH_CALUDE_complex_division_proof_l570_57023

theorem complex_division_proof : ∀ (i : ℂ), i^2 = -1 → (1 : ℂ) / (1 + i) = (1 - i) / 2 := by sorry

end NUMINAMATH_CALUDE_complex_division_proof_l570_57023


namespace NUMINAMATH_CALUDE_orange_harvest_calculation_l570_57090

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def harvest_days : ℕ := 49

/-- The total number of sacks harvested after the given number of days -/
def total_sacks : ℕ := sacks_per_day * harvest_days

theorem orange_harvest_calculation :
  total_sacks = 1862 := by sorry

end NUMINAMATH_CALUDE_orange_harvest_calculation_l570_57090


namespace NUMINAMATH_CALUDE_four_teacher_proctoring_l570_57056

/-- Represents the number of teachers and classes -/
def n : ℕ := 4

/-- The number of ways to arrange n teachers to proctor n classes, where no teacher proctors their own class -/
def derangement (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of ways to arrange 4 teachers to proctor 4 classes, where no teacher proctors their own class, is equal to 9 -/
theorem four_teacher_proctoring : derangement n = 9 := by sorry

end NUMINAMATH_CALUDE_four_teacher_proctoring_l570_57056


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l570_57042

theorem complex_fraction_simplification :
  (7 : ℂ) + 15 * Complex.I / ((3 : ℂ) - 4 * Complex.I) = -39 / 25 + (73 / 25 : ℝ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l570_57042


namespace NUMINAMATH_CALUDE_prob_white_then_red_is_four_fifteenths_l570_57024

/-- Represents the number of red marbles in the bag -/
def red_marbles : ℕ := 4

/-- Represents the number of white marbles in the bag -/
def white_marbles : ℕ := 6

/-- Represents the total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles

/-- The probability of drawing a white marble first and a red marble second -/
def prob_white_then_red : ℚ :=
  (white_marbles : ℚ) / total_marbles * red_marbles / (total_marbles - 1)

theorem prob_white_then_red_is_four_fifteenths :
  prob_white_then_red = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_then_red_is_four_fifteenths_l570_57024


namespace NUMINAMATH_CALUDE_front_view_correct_l570_57026

def ColumnHeights := List Nat

def frontView (columns : List ColumnHeights) : List Nat :=
  columns.map (List.foldl max 0)

theorem front_view_correct (columns : List ColumnHeights) :
  frontView columns = [3, 4, 5, 2] :=
by
  -- The proof would go here
  sorry

#eval frontView [[3, 2], [1, 4, 2], [5], [2, 1]]

end NUMINAMATH_CALUDE_front_view_correct_l570_57026


namespace NUMINAMATH_CALUDE_xoxox_probability_l570_57048

def total_tiles : ℕ := 5
def x_tiles : ℕ := 3
def o_tiles : ℕ := 2

theorem xoxox_probability :
  (x_tiles : ℚ) / total_tiles *
  (o_tiles : ℚ) / (total_tiles - 1) *
  ((x_tiles - 1) : ℚ) / (total_tiles - 2) *
  ((o_tiles - 1) : ℚ) / (total_tiles - 3) *
  ((x_tiles - 2) : ℚ) / (total_tiles - 4) = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_xoxox_probability_l570_57048


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l570_57098

theorem quadratic_root_implies_a_value (x a : ℝ) : 
  x = 1 → x^2 + a*x - 2 = 0 → a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l570_57098


namespace NUMINAMATH_CALUDE_cranberry_juice_cost_per_ounce_l570_57029

/-- The cost per ounce of a can of cranberry juice -/
def cost_per_ounce (total_cost : ℚ) (volume : ℚ) : ℚ :=
  total_cost / volume

/-- Theorem: The cost per ounce of a 12-ounce can of cranberry juice selling for 84 cents is 7 cents -/
theorem cranberry_juice_cost_per_ounce :
  cost_per_ounce 84 12 = 7 := by
  sorry

#eval cost_per_ounce 84 12

end NUMINAMATH_CALUDE_cranberry_juice_cost_per_ounce_l570_57029


namespace NUMINAMATH_CALUDE_complex_equation_solution_l570_57052

theorem complex_equation_solution (a : ℝ) (z : ℂ) 
  (h1 : a ≥ 0) 
  (h2 : z * Complex.abs z + a * z + Complex.I = 0) : 
  z = Complex.I * ((a - Real.sqrt (a^2 + 4)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l570_57052


namespace NUMINAMATH_CALUDE_stock_price_fluctuation_l570_57072

theorem stock_price_fluctuation (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.4
  let decrease_factor := 1 - 0.2857
  decrease_factor * increased_price = original_price := by
  sorry

end NUMINAMATH_CALUDE_stock_price_fluctuation_l570_57072


namespace NUMINAMATH_CALUDE_line_parallel_perp_plane_implies_perp_line_l570_57097

/-- In three-dimensional space -/
structure Space :=
  (points : Type*)
  (vectors : Type*)

/-- A line in space -/
structure Line (S : Space) :=
  (point : S.points)
  (direction : S.vectors)

/-- A plane in space -/
structure Plane (S : Space) :=
  (point : S.points)
  (normal : S.vectors)

/-- Parallel relation between lines -/
def parallel (S : Space) (a b : Line S) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def perp_line_plane (S : Space) (l : Line S) (α : Plane S) : Prop := sorry

/-- Perpendicular relation between lines -/
def perp_line_line (S : Space) (l1 l2 : Line S) : Prop := sorry

/-- The main theorem -/
theorem line_parallel_perp_plane_implies_perp_line 
  (S : Space) (a b l : Line S) (α : Plane S) :
  parallel S a b → perp_line_plane S l α → perp_line_line S l b := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_perp_plane_implies_perp_line_l570_57097


namespace NUMINAMATH_CALUDE_joy_tape_leftover_l570_57027

/-- Calculates the amount of tape left over after wrapping a rectangular field once. -/
def tape_left_over (total_tape : ℕ) (length width : ℕ) : ℕ :=
  total_tape - 2 * (length + width)

/-- Theorem stating that wrapping a 60x20 field with 250 feet of tape leaves 90 feet. -/
theorem joy_tape_leftover :
  tape_left_over 250 60 20 = 90 := by
  sorry

end NUMINAMATH_CALUDE_joy_tape_leftover_l570_57027


namespace NUMINAMATH_CALUDE_merchant_problem_l570_57040

theorem merchant_problem (n : ℕ) (C : ℕ) : 
  (8 * n = C + 3) → 
  (7 * n = C - 4) → 
  (n = 7 ∧ C = 53) := by
  sorry

end NUMINAMATH_CALUDE_merchant_problem_l570_57040


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l570_57074

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 7

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 500

/-- Al's rest days in his cycle -/
def al_rest_days : Finset ℕ := {6, 7}

/-- Barb's rest day in her cycle -/
def barb_rest_day : ℕ := 5

/-- The number of days both Al and Barb rest in the same 35-day period -/
def coinciding_rest_days_per_cycle : ℕ := 1

theorem coinciding_rest_days_count : 
  (total_days / (al_cycle * barb_cycle)) * coinciding_rest_days_per_cycle = 14 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l570_57074


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l570_57041

theorem right_triangle_third_side : ∀ a b c : ℝ,
  (a^2 - 9*a + 20 = 0) →
  (b^2 - 9*b + 20 = 0) →
  (a ≠ b) →
  (a^2 + b^2 = c^2) →
  (c = 3 ∨ c = Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l570_57041


namespace NUMINAMATH_CALUDE_range_of_a_circles_intersect_l570_57049

noncomputable section

-- Define the circles and line
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def circle_D (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x = 0
def line (x y a : ℝ) : Prop := x + y - a = 0

-- Define the inequality condition
def inequality_condition (x y m : ℝ) : Prop :=
  x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x y, circle_C x y ∧ line x y a) →
  2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2 :=
sorry

-- Theorem for the intersection of circles
theorem circles_intersect (m : ℝ) :
  (∀ x y, circle_C x y → inequality_condition x y m) →
  ∃ x y, circle_C x y ∧ circle_D x y m :=
sorry

end NUMINAMATH_CALUDE_range_of_a_circles_intersect_l570_57049


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l570_57051

theorem multiplication_equation_solution : ∃ x : ℚ, 9 * x = 36 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l570_57051


namespace NUMINAMATH_CALUDE_divisibility_by_900_l570_57089

theorem divisibility_by_900 (n : ℕ) : ∃ k : ℤ, 6^(2*(n+1)) - 2^(n+3) * 3^(n+2) + 36 = 900 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_900_l570_57089


namespace NUMINAMATH_CALUDE_ones_and_seven_primality_l570_57028

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def ones_and_seven (n : ℕ) : ℕ :=
  if n = 1 then 7
  else (10^(n-1) - 1) / 9 + 7 * 10^((n-1) / 2)

theorem ones_and_seven_primality (n : ℕ) :
  is_prime (ones_and_seven n) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_ones_and_seven_primality_l570_57028


namespace NUMINAMATH_CALUDE_distance_sum_property_l570_57060

/-- Linear mapping between two line segments -/
structure LinearSegmentMap (AB A'B' : ℝ) where
  scale : ℝ
  map_points : ℝ → ℝ
  map_property : ∀ x, map_points x = scale * x

/-- Representation of a point on a line segment -/
structure SegmentPoint (total_length : ℝ) where
  position : ℝ
  valid_position : 0 ≤ position ∧ position ≤ total_length

theorem distance_sum_property 
  (AB A'B' : ℝ) 
  (h_AB_pos : AB > 0)
  (h_A'B'_pos : A'B' > 0)
  (h_linear_map : LinearSegmentMap AB A'B')
  (D : SegmentPoint AB)
  (D' : SegmentPoint A'B')
  (h_D_midpoint : D.position = AB / 2)
  (h_D'_third : D'.position = A'B' / 3)
  (P : SegmentPoint AB)
  (P' : SegmentPoint A'B')
  (h_P'_mapped : P'.position = h_linear_map.map_points P.position)
  (h_AB_length : AB = 3)
  (h_A'B'_length : A'B' = 6)
  (a : ℝ)
  (h_x_eq_a : |P.position - D.position| = a) :
  |P.position - D.position| + |P'.position - D'.position| = 3 * a :=
sorry

end NUMINAMATH_CALUDE_distance_sum_property_l570_57060


namespace NUMINAMATH_CALUDE_highlighter_difference_l570_57066

theorem highlighter_difference (total pink blue yellow : ℕ) : 
  total = 40 →
  yellow = 7 →
  blue = pink + 5 →
  total = yellow + pink + blue →
  pink - yellow = 7 := by
sorry

end NUMINAMATH_CALUDE_highlighter_difference_l570_57066


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l570_57050

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l570_57050


namespace NUMINAMATH_CALUDE_total_toys_cost_is_20_74_l570_57039

/-- The amount spent on toy cars -/
def toy_cars_cost : ℚ := 14.88

/-- The amount spent on toy trucks -/
def toy_trucks_cost : ℚ := 5.86

/-- The total amount spent on toys -/
def total_toys_cost : ℚ := toy_cars_cost + toy_trucks_cost

/-- Theorem stating that the total amount spent on toys is $20.74 -/
theorem total_toys_cost_is_20_74 : total_toys_cost = 20.74 := by sorry

end NUMINAMATH_CALUDE_total_toys_cost_is_20_74_l570_57039


namespace NUMINAMATH_CALUDE_abc_mod_five_l570_57013

theorem abc_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 →
  (a + 2*b + 3*c) % 5 = 0 →
  (2*a + 3*b + c) % 5 = 2 →
  (3*a + b + 2*c) % 5 = 3 →
  (a*b*c) % 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_abc_mod_five_l570_57013


namespace NUMINAMATH_CALUDE_jacks_water_bottles_l570_57018

/-- Represents the problem of determining how many bottles of water Jack initially bought. -/
theorem jacks_water_bottles :
  ∀ (initial_bottles : ℕ),
    (100 : ℚ) - (2 : ℚ) * (initial_bottles : ℚ) - (2 : ℚ) * (2 : ℚ) * (initial_bottles : ℚ) - (5 : ℚ) = (71 : ℚ) →
    initial_bottles = 4 := by
  sorry

end NUMINAMATH_CALUDE_jacks_water_bottles_l570_57018


namespace NUMINAMATH_CALUDE_ratio_transformation_l570_57087

theorem ratio_transformation (x y : ℝ) (h : x / y = 7 / 3) : 
  (x + y) / (x - y) = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_transformation_l570_57087


namespace NUMINAMATH_CALUDE_onesDigit_73_pow_355_l570_57096

-- Define a function to get the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem onesDigit_73_pow_355 : onesDigit (73^355) = 7 := by
  sorry

end NUMINAMATH_CALUDE_onesDigit_73_pow_355_l570_57096


namespace NUMINAMATH_CALUDE_prob_first_second_win_eq_three_tenths_l570_57088

/-- Represents a lottery with winning and non-winning tickets -/
structure Lottery where
  total_tickets : ℕ
  winning_tickets : ℕ
  people : ℕ
  h_winning_le_total : winning_tickets ≤ total_tickets
  h_people_le_total : people ≤ total_tickets

/-- The probability of drawing a winning ticket -/
def prob_win (L : Lottery) : ℚ :=
  L.winning_tickets / L.total_tickets

/-- The probability of both the first and second person drawing a winning ticket -/
def prob_first_second_win (L : Lottery) : ℚ :=
  (L.winning_tickets / L.total_tickets) * ((L.winning_tickets - 1) / (L.total_tickets - 1))

/-- Theorem stating the probability of both first and second person drawing a winning ticket -/
theorem prob_first_second_win_eq_three_tenths (L : Lottery) 
    (h_total : L.total_tickets = 5)
    (h_winning : L.winning_tickets = 3)
    (h_people : L.people = 5) :
    prob_first_second_win L = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_prob_first_second_win_eq_three_tenths_l570_57088


namespace NUMINAMATH_CALUDE_trucks_lifted_trucks_lifted_proof_l570_57076

/-- Proof that 3 trucks are being lifted -/
theorem trucks_lifted : ℕ :=
  let people_per_car : ℕ := 5
  let people_per_truck : ℕ := 2 * people_per_car
  let cars_lifted : ℕ := 6
  let total_lifted : ℕ := cars_lifted + 3
  3

theorem trucks_lifted_proof (people_per_car : ℕ) (cars_lifted : ℕ) 
  (h1 : people_per_car = 5)
  (h2 : cars_lifted = 6) :
  trucks_lifted = 3 := by
  sorry

end NUMINAMATH_CALUDE_trucks_lifted_trucks_lifted_proof_l570_57076


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l570_57036

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l570_57036


namespace NUMINAMATH_CALUDE_sqrt_320_simplification_l570_57004

theorem sqrt_320_simplification : Real.sqrt 320 = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_320_simplification_l570_57004


namespace NUMINAMATH_CALUDE_range_of_a_l570_57055

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |x - 4| + |x - 3| < a) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l570_57055


namespace NUMINAMATH_CALUDE_arcsin_of_negative_one_l570_57043

theorem arcsin_of_negative_one :
  Real.arcsin (-1) = -π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_negative_one_l570_57043


namespace NUMINAMATH_CALUDE_product_equals_two_thirds_l570_57082

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 1 + (a n - 1)^2

-- Define the infinite product of a_n
def infiniteProduct : ℚ := sorry

-- Theorem statement
theorem product_equals_two_thirds : infiniteProduct = 2/3 := by sorry

end NUMINAMATH_CALUDE_product_equals_two_thirds_l570_57082


namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l570_57091

/-- The difference in miles biked between Alberto and Bjorn after four hours -/
theorem alberto_bjorn_distance_difference :
  let alberto_distance : ℕ := 60
  let bjorn_distance : ℕ := 45
  alberto_distance - bjorn_distance = 15 := by
sorry

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l570_57091


namespace NUMINAMATH_CALUDE_triangle_side_length_l570_57007

theorem triangle_side_length 
  (AB : ℝ) 
  (time_AB time_BC_CA : ℝ) 
  (h1 : AB = 1992)
  (h2 : time_AB = 24)
  (h3 : time_BC_CA = 166)
  : ∃ (BC : ℝ), BC = 6745 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l570_57007


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l570_57053

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_vol : a^3 / b^3 = 27 / 8) : 
  a / b = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l570_57053


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l570_57077

theorem reciprocal_roots_quadratic (k : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
   ∀ x : ℝ, 5.2 * x * x + 14.3 * x + k = 0 ↔ (x = r₁ ∨ x = r₂)) →
  k = 5.2 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l570_57077


namespace NUMINAMATH_CALUDE_f_t_ratio_is_power_of_two_l570_57005

/-- Define f_t(n) as the number of odd C_k^t for 1 ≤ k ≤ n -/
def f_t (t n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem f_t_ratio_is_power_of_two (t : ℕ) (h : ℕ) :
  t > 0 → ∃ r : ℕ, ∀ n : ℕ, n = 2^h → (f_t t n : ℚ) / n = 1 / (2^r) := by
  sorry

end NUMINAMATH_CALUDE_f_t_ratio_is_power_of_two_l570_57005


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l570_57030

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l570_57030


namespace NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l570_57006

theorem five_integers_sum_20_product_420 : 
  ∃ (a b c d e : ℕ+), 
    (a.val + b.val + c.val + d.val + e.val = 20) ∧ 
    (a.val * b.val * c.val * d.val * e.val = 420) := by
  sorry

end NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l570_57006


namespace NUMINAMATH_CALUDE_complete_graph_inequality_l570_57033

theorem complete_graph_inequality (n k : ℕ) (N_k N_k_plus_1 : ℕ) 
  (h1 : 2 ≤ k) (h2 : k < n) (h3 : N_k > 0) (h4 : N_k_plus_1 > 0) :
  (N_k_plus_1 : ℚ) / N_k ≥ (1 : ℚ) / (k^2 - 1) * (k^2 * N_k / N_k_plus_1 - n) := by
  sorry

end NUMINAMATH_CALUDE_complete_graph_inequality_l570_57033


namespace NUMINAMATH_CALUDE_trajectory_of_M_center_l570_57020

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the properties of circle M
def M_externally_tangent_C₁ (M : ℝ × ℝ) : Prop :=
  ∃ r > 0, ∀ x y, C₁ x y → (x - M.1)^2 + (y - M.2)^2 = (r + 1)^2

def M_internally_tangent_C₂ (M : ℝ × ℝ) : Prop :=
  ∃ r > 0, ∀ x y, C₂ x y → (x - M.1)^2 + (y - M.2)^2 = (5 - r)^2

-- Theorem statement
theorem trajectory_of_M_center :
  ∀ M : ℝ × ℝ,
  M_externally_tangent_C₁ M →
  M_internally_tangent_C₂ M →
  M.1^2 / 9 + M.2^2 / 8 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_center_l570_57020


namespace NUMINAMATH_CALUDE_no_solution_exists_l570_57008

theorem no_solution_exists : ¬∃ (x y z : ℕ+), x^(x.val) + y^(y.val) = 9^(z.val) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l570_57008


namespace NUMINAMATH_CALUDE_thursday_monday_difference_l570_57067

/-- Represents the number of bonnets made on each day of the week --/
structure BonnetProduction where
  monday : ℕ
  tuesday_wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the difference between Thursday and Monday bonnet production --/
theorem thursday_monday_difference (bp : BonnetProduction) : 
  bp.monday = 10 →
  bp.tuesday_wednesday = 2 * bp.monday →
  bp.friday = bp.thursday - 5 →
  (bp.monday + bp.tuesday_wednesday + bp.thursday + bp.friday) / 5 = 11 →
  bp.thursday - bp.monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_thursday_monday_difference_l570_57067


namespace NUMINAMATH_CALUDE_width_covered_formula_l570_57001

/-- The width covered by n asbestos tiles -/
def width_covered (n : ℕ+) : ℝ :=
  let tile_width : ℝ := 60
  let overlap : ℝ := 10
  (n : ℝ) * (tile_width - overlap) + overlap

/-- Theorem: The width covered by n asbestos tiles is (50n + 10) cm -/
theorem width_covered_formula (n : ℕ+) :
  width_covered n = 50 * (n : ℝ) + 10 := by
  sorry

end NUMINAMATH_CALUDE_width_covered_formula_l570_57001


namespace NUMINAMATH_CALUDE_maria_journey_distance_l570_57009

/-- A journey with two stops and a final leg -/
structure Journey where
  total_distance : ℝ
  first_stop : ℝ
  second_stop : ℝ
  final_leg : ℝ

/-- The conditions of Maria's journey -/
def maria_journey (j : Journey) : Prop :=
  j.first_stop = j.total_distance / 2 ∧
  j.second_stop = (j.total_distance - j.first_stop) / 4 ∧
  j.final_leg = 135 ∧
  j.total_distance = j.first_stop + j.second_stop + j.final_leg

/-- Theorem stating that Maria's journey has a total distance of 360 miles -/
theorem maria_journey_distance :
  ∃ j : Journey, maria_journey j ∧ j.total_distance = 360 :=
sorry

end NUMINAMATH_CALUDE_maria_journey_distance_l570_57009


namespace NUMINAMATH_CALUDE_sequence_general_term_l570_57065

theorem sequence_general_term 
  (a : ℕ+ → ℝ) 
  (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = 3 * n.val ^ 2 - 2 * n.val) :
  ∀ n : ℕ+, a n = 6 * n.val - 5 :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l570_57065


namespace NUMINAMATH_CALUDE_mans_rate_l570_57044

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 22)
  (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l570_57044


namespace NUMINAMATH_CALUDE_younger_person_age_l570_57045

/-- Given two people with an age difference of 20 years, where 15 years ago the elder was twice as old as the younger, prove that the younger person's current age is 35 years. -/
theorem younger_person_age (younger elder : ℕ) : 
  elder - younger = 20 →
  elder - 15 = 2 * (younger - 15) →
  younger = 35 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_l570_57045


namespace NUMINAMATH_CALUDE_min_people_for_hundred_chairs_is_minimum_people_l570_57011

/-- The number of chairs in the circle -/
def num_chairs : ℕ := 100

/-- A function that calculates the minimum number of people needed -/
def min_people (chairs : ℕ) : ℕ :=
  (chairs + 2) / 3

/-- The theorem stating the minimum number of people for 100 chairs -/
theorem min_people_for_hundred_chairs :
  min_people num_chairs = 34 := by
  sorry

/-- The theorem proving that this is indeed the minimum -/
theorem is_minimum_people (n : ℕ) :
  n < min_people num_chairs →
  ∃ (m : ℕ), m > 2 ∧ m < num_chairs ∧
  ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧
  (m + i) % num_chairs = (m + j) % num_chairs := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_hundred_chairs_is_minimum_people_l570_57011


namespace NUMINAMATH_CALUDE_maria_stationery_cost_l570_57054

/-- The cost of Maria's stationery purchase -/
def stationery_cost (pencil_cost : ℝ) (pen_cost : ℝ) : Prop :=
  pencil_cost = 8 ∧ 
  pen_cost = pencil_cost / 2 ∧
  pencil_cost + pen_cost = 12

/-- Theorem: Maria paid $12 for both the pen and the pencil -/
theorem maria_stationery_cost : 
  ∃ (pencil_cost pen_cost : ℝ), stationery_cost pencil_cost pen_cost :=
by
  sorry

end NUMINAMATH_CALUDE_maria_stationery_cost_l570_57054


namespace NUMINAMATH_CALUDE_negation_of_proposition_l570_57057

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ 
  (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l570_57057


namespace NUMINAMATH_CALUDE_cuboid_area_example_l570_57058

/-- The surface area of a cuboid -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 12, breadth 6, and height 10 is 504 -/
theorem cuboid_area_example : cuboid_surface_area 12 6 10 = 504 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l570_57058


namespace NUMINAMATH_CALUDE_total_games_played_l570_57014

/-- Proves that a team with the given win percentages played 175 games in total -/
theorem total_games_played (first_100_win_rate : ℝ) (remaining_win_rate : ℝ) (total_win_rate : ℝ) 
  (h1 : first_100_win_rate = 0.85)
  (h2 : remaining_win_rate = 0.5)
  (h3 : total_win_rate = 0.7) :
  ∃ (total_games : ℕ), 
    total_games = 175 ∧ 
    (first_100_win_rate * 100 + remaining_win_rate * (total_games - 100 : ℝ)) / total_games = total_win_rate :=
by sorry

end NUMINAMATH_CALUDE_total_games_played_l570_57014


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l570_57086

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The actual counts of balls in the box -/
def initialCounts : BallCounts :=
  { red := 35, green := 27, yellow := 22, blue := 18, white := 15, black := 12 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : Nat := 20

/-- Theorem stating the minimum number of balls to draw to guarantee the target count -/
theorem min_balls_to_draw (counts : BallCounts) (target : Nat) :
  counts = initialCounts → target = targetCount →
  (∃ (n : Nat), n = 103 ∧
    (∀ (m : Nat), m < n →
      ¬∃ (color : Nat), color ≥ target ∧
        (color ≤ counts.red ∨ color ≤ counts.green ∨ color ≤ counts.yellow ∨
         color ≤ counts.blue ∨ color ≤ counts.white ∨ color ≤ counts.black)) ∧
    (∃ (color : Nat), color ≥ target ∧
      (color ≤ counts.red ∨ color ≤ counts.green ∨ color ≤ counts.yellow ∨
       color ≤ counts.blue ∨ color ≤ counts.white ∨ color ≤ counts.black))) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_l570_57086


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l570_57059

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_35 :
  ∃ (n : ℕ), is_four_digit n ∧ n % 35 = 0 ∧ ∀ (m : ℕ), is_four_digit m ∧ m % 35 = 0 → n ≤ m :=
by
  use 1050
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l570_57059


namespace NUMINAMATH_CALUDE_michael_chicken_count_l570_57038

/-- Calculates the number of chickens after a given number of years -/
def chickenCount (initialCount : ℕ) (annualIncrease : ℕ) (years : ℕ) : ℕ :=
  initialCount + annualIncrease * years

/-- Theorem stating that Michael will have 1900 chickens after 9 years -/
theorem michael_chicken_count :
  chickenCount 550 150 9 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_michael_chicken_count_l570_57038


namespace NUMINAMATH_CALUDE_eldest_child_age_l570_57093

/-- Represents the ages of three grandchildren -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.middle = ages.youngest + 3 ∧
  ages.eldest = 3 * ages.youngest ∧
  ages.eldest = ages.youngest + ages.middle + 2

/-- The theorem stating that the eldest child's age is 15 years -/
theorem eldest_child_age (ages : GrandchildrenAges) :
  satisfiesConditions ages → ages.eldest = 15 := by
  sorry


end NUMINAMATH_CALUDE_eldest_child_age_l570_57093


namespace NUMINAMATH_CALUDE_star_seven_three_l570_57025

def star (a b : ℝ) : ℝ := 2*a + 5*b - a*b + 2

theorem star_seven_three : star 7 3 = 10 := by sorry

end NUMINAMATH_CALUDE_star_seven_three_l570_57025


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l570_57099

theorem mark_and_carolyn_money_sum : 
  let mark_money : ℚ := 3 / 4
  let carolyn_money : ℚ := 3 / 10
  mark_money + carolyn_money = 21 / 20 := by
sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l570_57099


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l570_57083

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 134) (h2 : b = 98) (h3 : c = 120) (h4 : d = 110) (h5 : e = 96) :
  720 - (a + b + c + d + e) = 162 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l570_57083


namespace NUMINAMATH_CALUDE_only_rectangle_area_certain_l570_57016

-- Define the events
inductive Event
  | WaterFreeze : Event
  | ExamScore : Event
  | CoinToss : Event
  | RectangleArea : Event

-- Define a function to check if an event is certain
def isCertainEvent : Event → Prop
  | Event.WaterFreeze => False
  | Event.ExamScore => False
  | Event.CoinToss => False
  | Event.RectangleArea => True

-- Theorem statement
theorem only_rectangle_area_certain :
  ∀ e : Event, isCertainEvent e ↔ e = Event.RectangleArea :=
by sorry

end NUMINAMATH_CALUDE_only_rectangle_area_certain_l570_57016


namespace NUMINAMATH_CALUDE_root_implies_h_value_l570_57061

theorem root_implies_h_value (h : ℝ) :
  (3 : ℝ)^3 - 2*h*3 + 15 = 0 → h = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l570_57061


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l570_57035

theorem power_fraction_simplification :
  (6^5 * 3^5) / 18^4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l570_57035


namespace NUMINAMATH_CALUDE_trigonometric_identities_l570_57079

theorem trigonometric_identities :
  (∀ x : Real, (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2) ∧
  (∀ x : Real, (3 - Real.sin (70 * π / 180)) / (2 - Real.cos (10 * π / 180)^2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l570_57079


namespace NUMINAMATH_CALUDE_total_rabbits_l570_57021

theorem total_rabbits (initial additional : ℕ) : 
  initial + additional = (initial + additional) :=
by sorry

end NUMINAMATH_CALUDE_total_rabbits_l570_57021


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l570_57019

theorem sum_of_absolute_coefficients (x a a₁ a₂ a₃ a₄ : ℝ) 
  (h : (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4)
  (ha : a > 0)
  (ha₂ : a₂ > 0)
  (ha₄ : a₄ > 0)
  (ha₁ : a₁ < 0)
  (ha₃ : a₃ < 0) :
  |a| + |a₁| + |a₂| + |a₃| + |a₄| = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l570_57019


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l570_57037

/-- Calculates the molecular weight of a compound given its atomic composition and atomic weights -/
def molecular_weight (num_C num_H num_O num_N num_S : ℕ) 
                     (weight_C weight_H weight_O weight_N weight_S : ℝ) : ℝ :=
  (num_C : ℝ) * weight_C + 
  (num_H : ℝ) * weight_H + 
  (num_O : ℝ) * weight_O + 
  (num_N : ℝ) * weight_N + 
  (num_S : ℝ) * weight_S

/-- The molecular weight of the given compound is approximately 134.184 g/mol -/
theorem compound_molecular_weight : 
  ∀ (ε : ℝ), ε > 0 → 
  |molecular_weight 4 8 2 1 1 12.01 1.008 16.00 14.01 32.07 - 134.184| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l570_57037


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l570_57084

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) : 
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -122 / 121 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l570_57084


namespace NUMINAMATH_CALUDE_rhombus_side_length_l570_57015

/-- A rhombus with one diagonal of length 20 and area 480 has sides of length 26 -/
theorem rhombus_side_length (d1 d2 area side : ℝ) : 
  d1 = 20 →
  area = 480 →
  area = d1 * d2 / 2 →
  side * side = (d1/2)^2 + (d2/2)^2 →
  side = 26 := by
sorry


end NUMINAMATH_CALUDE_rhombus_side_length_l570_57015


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l570_57068

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l570_57068


namespace NUMINAMATH_CALUDE_inequality_of_three_nonnegative_reals_l570_57062

theorem inequality_of_three_nonnegative_reals (a b c : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  |c * a - a * b| + |a * b - b * c| + |b * c - c * a| ≤ 
  |b^2 - c^2| + |c^2 - a^2| + |a^2 - b^2| := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_three_nonnegative_reals_l570_57062


namespace NUMINAMATH_CALUDE_chosen_number_proof_l570_57095

theorem chosen_number_proof (x : ℝ) : (x / 2) - 100 = 4 → x = 208 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l570_57095


namespace NUMINAMATH_CALUDE_circle_equation_l570_57003

/-- The circle passing through points A(-1, 1) and B(-2, -2), with center C lying on the line x+y-1=0, has the standard equation (x - 3)² + (y + 2)² = 25 -/
theorem circle_equation : 
  ∀ (C : ℝ × ℝ) (r : ℝ),
  (C.1 + C.2 - 1 = 0) →  -- Center C lies on the line x+y-1=0
  ((-1 - C.1)^2 + (1 - C.2)^2 = r^2) →  -- Circle passes through A(-1, 1)
  ((-2 - C.1)^2 + (-2 - C.2)^2 = r^2) →  -- Circle passes through B(-2, -2)
  ∀ (x y : ℝ), 
  ((x - 3)^2 + (y + 2)^2 = 25) ↔ ((x - C.1)^2 + (y - C.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l570_57003


namespace NUMINAMATH_CALUDE_vertex_angle_is_160_degrees_l570_57046

/-- An isosceles triangle with specific properties -/
structure SpecialIsoscelesTriangle where
  -- The length of each equal side
  a : ℝ
  -- The base of the triangle
  b : ℝ
  -- The height of the triangle
  h : ℝ
  -- The vertex angle in radians
  θ : ℝ
  -- The triangle is isosceles
  isIsosceles : b = 2 * a * Real.cos θ
  -- The square of the length of each equal side is three times the product of the base and the height
  sideSquareProperty : a^2 = 3 * b * h
  -- The triangle is obtuse
  isObtuse : θ > Real.pi / 2

/-- The theorem stating that the vertex angle of the special isosceles triangle is 160 degrees -/
theorem vertex_angle_is_160_degrees (t : SpecialIsoscelesTriangle) : 
  t.θ = 160 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_is_160_degrees_l570_57046
