import Mathlib

namespace orange_ribbons_l3382_338215

theorem orange_ribbons (total : ℚ) (black : ℕ) : 
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + black = total →
  black = 40 →
  (1/6 : ℚ) * total = 80/3 := by
sorry

end orange_ribbons_l3382_338215


namespace sum_base4_equals_l3382_338230

/-- Convert a base 4 number to its decimal representation -/
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Convert a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Addition of two base 4 numbers -/
def addBase4 (a b : List Nat) : List Nat :=
  decimalToBase4 (base4ToDecimal a + base4ToDecimal b)

theorem sum_base4_equals : 
  addBase4 (addBase4 [3, 0, 2] [2, 1, 1]) [0, 3, 3] = [0, 1, 1, 3, 1] := by
  sorry


end sum_base4_equals_l3382_338230


namespace total_training_hours_endurance_training_hours_l3382_338268

/-- Represents the training schedule for a goalkeeper --/
structure GoalkeeperSchedule where
  diving_catching : ℝ
  strength_conditioning : ℝ
  goalkeeper_specific : ℝ
  footwork : ℝ
  reaction_time : ℝ
  aerial_ball : ℝ
  shot_stopping : ℝ
  defensive_communication : ℝ
  game_simulation : ℝ
  endurance : ℝ

/-- Calculates the total training hours per week --/
def weekly_hours (s : GoalkeeperSchedule) : ℝ :=
  s.diving_catching + s.strength_conditioning + s.goalkeeper_specific +
  s.footwork + s.reaction_time + s.aerial_ball + s.shot_stopping +
  s.defensive_communication + s.game_simulation + s.endurance

/-- Mike's weekly training schedule --/
def mike_schedule : GoalkeeperSchedule :=
  { diving_catching := 2
  , strength_conditioning := 4
  , goalkeeper_specific := 2
  , footwork := 2
  , reaction_time := 1
  , aerial_ball := 3.5
  , shot_stopping := 1.5
  , defensive_communication := 1.5
  , game_simulation := 3
  , endurance := 3
  }

/-- The number of weeks Mike will train --/
def training_weeks : ℕ := 3

/-- Theorem: Mike's total training hours over 3 weeks is 70.5 --/
theorem total_training_hours :
  (weekly_hours mike_schedule) * training_weeks = 70.5 := by sorry

/-- Theorem: Mike's endurance training hours over 3 weeks is 9 --/
theorem endurance_training_hours :
  mike_schedule.endurance * training_weeks = 9 := by sorry

end total_training_hours_endurance_training_hours_l3382_338268


namespace more_girls_than_boys_l3382_338276

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 49 →
  3 * girls = 4 * boys →
  total_students = boys + girls →
  girls - boys = 7 := by
sorry

end more_girls_than_boys_l3382_338276


namespace m_equals_eight_m_uniqueness_l3382_338220

/-- The value of m for which the given conditions are satisfied -/
def find_m : ℝ → Prop := λ m =>
  m ≠ 0 ∧
  ∃ A B : ℝ × ℝ,
    -- Circle equation
    (A.1 + 1)^2 + A.2^2 = 4 ∧
    (B.1 + 1)^2 + B.2^2 = 4 ∧
    -- Points A and B are on the directrix of the parabola
    A.1 = -m/4 ∧
    B.1 = -m/4 ∧
    -- Distance between A and B
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 ∧
    -- Parabola equation (not directly used, but implied by the directrix)
    ∀ x y, y^2 = m*x → x ≥ -m/4

/-- Theorem stating that m = 8 satisfies the given conditions -/
theorem m_equals_eight : find_m 8 := by sorry

/-- Theorem stating that 8 is the only value of m that satisfies the given conditions -/
theorem m_uniqueness : ∀ m, find_m m → m = 8 := by sorry

end m_equals_eight_m_uniqueness_l3382_338220


namespace base7_addition_subtraction_l3382_338299

-- Define a function to convert base 7 to decimal
def base7ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Define a function to convert decimal to base 7
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

-- Define the numbers in base 7
def n1 : List Nat := [0, 0, 0, 1]  -- 1000₇
def n2 : List Nat := [6, 6, 6]     -- 666₇
def n3 : List Nat := [4, 3, 2, 1]  -- 1234₇

-- State the theorem
theorem base7_addition_subtraction :
  decimalToBase7 (base7ToDecimal n1 + base7ToDecimal n2 - base7ToDecimal n3) = [4, 5, 2] := by
  sorry

end base7_addition_subtraction_l3382_338299


namespace stating_largest_valid_m_l3382_338218

/-- 
Given a positive integer m, checks if m! can be expressed as the product 
of m - 4 consecutive positive integers.
-/
def is_valid (m : ℕ+) : Prop :=
  ∃ a : ℕ, m.val.factorial = (Finset.range (m - 4)).prod (λ i => i + a + 1)

/-- 
Theorem stating that 1 is the largest positive integer m such that m! 
can be expressed as the product of m - 4 consecutive positive integers.
-/
theorem largest_valid_m : 
  is_valid 1 ∧ ∀ m : ℕ+, m > 1 → ¬is_valid m :=
sorry

end stating_largest_valid_m_l3382_338218


namespace fred_movie_change_l3382_338292

theorem fred_movie_change 
  (ticket_price : ℚ)
  (num_tickets : ℕ)
  (borrowed_movie_price : ℚ)
  (paid_amount : ℚ)
  (h1 : ticket_price = 592/100)
  (h2 : num_tickets = 2)
  (h3 : borrowed_movie_price = 679/100)
  (h4 : paid_amount = 20) :
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price) = 137/100 := by
  sorry

end fred_movie_change_l3382_338292


namespace even_function_sum_l3382_338213

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + 1

-- Define the property of being an even function
def is_even_function (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a (a + 1) → g x = g (-x)

-- Theorem statement
theorem even_function_sum (a b : ℝ) :
  is_even_function (f a b) a → a + a^b = 1/2 := by
  sorry

end even_function_sum_l3382_338213


namespace or_not_implies_other_l3382_338228

theorem or_not_implies_other (p q : Prop) : (p ∨ q) → ¬p → q := by sorry

end or_not_implies_other_l3382_338228


namespace circle_radius_from_intersecting_chords_l3382_338267

theorem circle_radius_from_intersecting_chords (a b d : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) :
  ∃ (r : ℝ),
    (r = (a/d) * Real.sqrt (a^2 + b^2 - 2*b * Real.sqrt (a^2 - d^2))) ∨
    (r = (a/d) * Real.sqrt (a^2 + b^2 + 2*b * Real.sqrt (a^2 - d^2))) ∨
    (a = d ∧ r = Real.sqrt (a^2 + b^2)) :=
by sorry

end circle_radius_from_intersecting_chords_l3382_338267


namespace total_questions_submitted_l3382_338222

/-- Given the ratio of questions submitted by Rajat, Vikas, and Abhishek,
    and the number of questions submitted by Vikas, calculate the total
    number of questions submitted. -/
theorem total_questions_submitted
  (ratio_rajat : ℕ)
  (ratio_vikas : ℕ)
  (ratio_abhishek : ℕ)
  (vikas_questions : ℕ)
  (h_ratio : ratio_rajat = 7 ∧ ratio_vikas = 3 ∧ ratio_abhishek = 2)
  (h_vikas : vikas_questions = 6) :
  ratio_rajat * vikas_questions / ratio_vikas +
  vikas_questions +
  ratio_abhishek * vikas_questions / ratio_vikas = 24 :=
by sorry

end total_questions_submitted_l3382_338222


namespace x_plus_y_values_l3382_338274

theorem x_plus_y_values (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 := by
  sorry

end x_plus_y_values_l3382_338274


namespace even_increasing_neg_sum_positive_l3382_338251

/-- An even function that is increasing on the negative real line -/
def EvenIncreasingNeg (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- Theorem statement -/
theorem even_increasing_neg_sum_positive
  (f : ℝ → ℝ) (hf : EvenIncreasingNeg f) (x₁ x₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ < 0) (hf_x : f x₁ < f x₂) :
  x₁ + x₂ > 0 :=
sorry

end even_increasing_neg_sum_positive_l3382_338251


namespace spinning_tops_theorem_l3382_338294

/-- The number of spinning tops obtained from gift boxes --/
def spinning_tops_count (red_price yellow_price : ℕ) (red_tops yellow_tops : ℕ) 
  (total_spent total_boxes : ℕ) : ℕ :=
  let red_boxes := (total_spent - yellow_price * total_boxes) / (red_price - yellow_price)
  let yellow_boxes := total_boxes - red_boxes
  red_boxes * red_tops + yellow_boxes * yellow_tops

/-- Theorem stating the number of spinning tops obtained --/
theorem spinning_tops_theorem : 
  spinning_tops_count 5 9 3 5 600 72 = 336 := by
  sorry

#eval spinning_tops_count 5 9 3 5 600 72

end spinning_tops_theorem_l3382_338294


namespace cryptarithm_solution_is_unique_l3382_338225

/-- Represents a cryptarithm solution -/
structure CryptarithmSolution where
  F : Nat
  R : Nat
  Y : Nat
  H : Nat
  A : Nat
  M : Nat
  digit_constraint : F < 10 ∧ R < 10 ∧ Y < 10 ∧ H < 10 ∧ A < 10 ∧ M < 10
  unique_digits : F ≠ R ∧ F ≠ Y ∧ F ≠ H ∧ F ≠ A ∧ F ≠ M ∧
                  R ≠ Y ∧ R ≠ H ∧ R ≠ A ∧ R ≠ M ∧
                  Y ≠ H ∧ Y ≠ A ∧ Y ≠ M ∧
                  H ≠ A ∧ H ≠ M ∧
                  A ≠ M
  equation_holds : 7 * (100000 * F + 10000 * R + 1000 * Y + 100 * H + 10 * A + M) =
                   6 * (100000 * H + 10000 * A + 1000 * M + 100 * F + 10 * R + Y)

theorem cryptarithm_solution_is_unique : 
  ∀ (sol : CryptarithmSolution), 
    100 * sol.F + 10 * sol.R + sol.Y = 461 ∧ 
    100 * sol.H + 10 * sol.A + sol.M = 538 := by
  sorry

end cryptarithm_solution_is_unique_l3382_338225


namespace tangent_line_and_inequality_and_minimum_value_l3382_338286

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_and_inequality_and_minimum_value :
  -- 1. The tangent line to f(x) at x = 1 is y = x - 1
  (∀ x, (f x - f 1) = (x - 1) * (Real.log 1 + 1)) ∧
  -- 2. f(x) ≥ x - 1 for all x > 0
  (∀ x > 0, f x ≥ x - 1) ∧
  -- 3. The minimum value of a such that f(x) ≥ ax² + 2/a for all x > 0 and a ≠ 0 is -e³
  (∀ a ≠ 0, (∀ x > 0, f x ≥ a * x^2 + 2/a) ↔ a ≥ -Real.exp 3) :=
by sorry

end tangent_line_and_inequality_and_minimum_value_l3382_338286


namespace f_500_equals_39_l3382_338259

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ) : Prop :=
  (∀ x y : ℕ+, f (x * y) = f x + f y) ∧ 
  (f 10 = 14) ∧ 
  (f 40 = 20)

/-- Theorem stating the result for f(500) -/
theorem f_500_equals_39 (f : ℕ+ → ℕ) (h : special_function f) : f 500 = 39 := by
  sorry

end f_500_equals_39_l3382_338259


namespace lcm_of_150_and_456_l3382_338249

theorem lcm_of_150_and_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end lcm_of_150_and_456_l3382_338249


namespace correct_subtraction_l3382_338272

theorem correct_subtraction (x : ℤ) (h1 : x - 32 = 25) (h2 : 23 ≠ 32) : x - 23 = 34 := by
  sorry

end correct_subtraction_l3382_338272


namespace prob_at_least_one_three_l3382_338242

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of outcomes where neither die shows the target number -/
def non_target_outcomes : ℕ := (num_sides - 1) * (num_sides - 1)

/-- The probability of at least one die showing the target number -/
def prob_at_least_one_target : ℚ := (total_outcomes - non_target_outcomes) / total_outcomes

theorem prob_at_least_one_three :
  prob_at_least_one_target = 15 / 64 :=
sorry

end prob_at_least_one_three_l3382_338242


namespace tangent_line_equations_l3382_338287

/-- The curve to which the line is tangent -/
def f (x : ℝ) : ℝ := x^2 * (x + 1)

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * x

/-- A line that passes through (3/5, 0) and is tangent to f at point t -/
def tangent_line (t : ℝ) (x : ℝ) : ℝ :=
  f' t * (x - t) + f t

/-- The point (3/5, 0) lies on the tangent line -/
def point_condition (t : ℝ) : Prop :=
  tangent_line t (3/5) = 0

/-- The possible equations for the tangent line -/
def possible_equations (x : ℝ) : Prop :=
  (∃ t, point_condition t ∧ tangent_line t x = 0) ∨
  (∃ t, point_condition t ∧ tangent_line t x = -3/2 * x + 9/125) ∨
  (∃ t, point_condition t ∧ tangent_line t x = 5 * x - 3)

theorem tangent_line_equations :
  ∀ x, possible_equations x :=
sorry

end tangent_line_equations_l3382_338287


namespace digital_earth_functions_l3382_338207

/-- Represents the Digital Earth system -/
structure DigitalEarth where
  -- Define properties of Digital Earth
  is_huge : Bool
  is_precise : Bool
  is_digital_representation : Bool
  is_information_repository : Bool

/-- Functions that Digital Earth can perform -/
inductive DigitalEarthFunction
  | JointResearch
  | GlobalEducation
  | CrimeTracking
  | SustainableDevelopment

/-- Theorem stating that Digital Earth supports all four functions -/
theorem digital_earth_functions (de : DigitalEarth) : 
  (de.is_huge ∧ de.is_precise ∧ de.is_digital_representation ∧ de.is_information_repository) →
  (∀ f : DigitalEarthFunction, f ∈ [DigitalEarthFunction.JointResearch, 
                                    DigitalEarthFunction.GlobalEducation, 
                                    DigitalEarthFunction.CrimeTracking, 
                                    DigitalEarthFunction.SustainableDevelopment]) :=
by
  sorry


end digital_earth_functions_l3382_338207


namespace modulus_of_complex_quotient_l3382_338280

theorem modulus_of_complex_quotient :
  let z : ℂ := (1 - Complex.I) / (3 + 4 * Complex.I)
  Complex.abs z = Real.sqrt 2 / 5 := by
  sorry

end modulus_of_complex_quotient_l3382_338280


namespace square_area_difference_l3382_338271

theorem square_area_difference (area_A : ℝ) (side_diff : ℝ) : 
  area_A = 25 → side_diff = 4 → 
  let side_A := Real.sqrt area_A
  let side_B := side_A + side_diff
  side_B ^ 2 = 81 := by
sorry

end square_area_difference_l3382_338271


namespace point_on_line_l3382_338279

/-- Given a line passing through point M(0, 1) with slope -1,
    prove that any point P(3, m) on this line satisfies m = -2 -/
theorem point_on_line (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P.1 = 3 ∧ P.2 = m ∧ 
   (m - 1) / (3 - 0) = -1) → 
  m = -2 :=
by sorry

end point_on_line_l3382_338279


namespace initial_profit_percentage_l3382_338295

/-- Proves that the initial profit percentage is 5% given the conditions of the problem -/
theorem initial_profit_percentage (cost_price selling_price : ℝ) : 
  cost_price = 1000 →
  (0.95 * cost_price) * 1.1 = selling_price - 5 →
  (selling_price - cost_price) / cost_price = 0.05 := by
  sorry

end initial_profit_percentage_l3382_338295


namespace new_books_count_l3382_338255

def adventure_books : ℕ := 13
def mystery_books : ℕ := 17
def used_books : ℕ := 15

def total_books : ℕ := adventure_books + mystery_books

theorem new_books_count : total_books - used_books = 15 := by
  sorry

end new_books_count_l3382_338255


namespace vector_operation_l3382_338261

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 1)) :
  2 • a - b = (-1, 3) := by sorry

end vector_operation_l3382_338261


namespace sophomore_mean_is_94_l3382_338216

/-- Represents the number of students and their scores in a math competition -/
structure MathCompetition where
  total_students : ℕ
  overall_mean : ℝ
  sophomores : ℕ
  juniors : ℕ
  sophomore_mean : ℝ
  junior_mean : ℝ

/-- The math competition satisfies the given conditions -/
def satisfies_conditions (mc : MathCompetition) : Prop :=
  mc.total_students = 150 ∧
  mc.overall_mean = 85 ∧
  mc.juniors = mc.sophomores - (mc.sophomores / 5) ∧
  mc.sophomore_mean = mc.junior_mean * 1.25

/-- Theorem stating that under the given conditions, the sophomore mean score is 94 -/
theorem sophomore_mean_is_94 (mc : MathCompetition) 
  (h : satisfies_conditions mc) : mc.sophomore_mean = 94 := by
  sorry

#check sophomore_mean_is_94

end sophomore_mean_is_94_l3382_338216


namespace square_area_ratio_l3382_338209

theorem square_area_ratio (R : ℝ) (R_pos : R > 0) : 
  let x := Real.sqrt ((4 / 5) * R^2)
  let y := R * Real.sqrt 2
  x^2 / y^2 = 2 / 5 := by sorry

end square_area_ratio_l3382_338209


namespace pizza_consumption_order_l3382_338239

/-- Represents the amount of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : Rat
  beth : Rat
  cyril : Rat
  dan : Rat
  eliza : Rat

/-- Checks if a list of rationals is in decreasing order -/
def isDecreasing (l : List Rat) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≥ l[j]!

/-- The main theorem stating the correct order of pizza consumption -/
theorem pizza_consumption_order (p : PizzaConsumption) 
  (h1 : p.alex = 1/6)
  (h2 : p.beth = 1/4)
  (h3 : p.cyril = 1/3)
  (h4 : p.dan = 0)
  (h5 : p.eliza = 1 - (p.alex + p.beth + p.cyril + p.dan)) :
  isDecreasing [p.cyril, p.beth, p.eliza, p.alex, p.dan] := by
  sorry

end pizza_consumption_order_l3382_338239


namespace sqrt_inequality_l3382_338250

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (x^2 - 3*x + 2) > x + 5 ↔ x < -23/13 := by sorry

end sqrt_inequality_l3382_338250


namespace y_axis_symmetry_l3382_338283

/-- Given a point P(2, 1), its symmetric point P' with respect to the y-axis has coordinates (-2, 1) -/
theorem y_axis_symmetry :
  let P : ℝ × ℝ := (2, 1)
  let P' : ℝ × ℝ := (-P.1, P.2)
  P' = (-2, 1) := by sorry

end y_axis_symmetry_l3382_338283


namespace writing_ways_equals_notebooks_l3382_338258

/-- The number of ways to start writing given a ratio of pens to notebooks and their quantities -/
def ways_to_start_writing (pen_ratio : ℕ) (notebook_ratio : ℕ) (num_pens : ℕ) (num_notebooks : ℕ) : ℕ :=
  min num_pens num_notebooks

/-- Theorem: Given the ratio of pens to notebooks is 5:4, with 50 pens and 40 notebooks,
    the number of ways to start writing is equal to the number of notebooks -/
theorem writing_ways_equals_notebooks :
  ways_to_start_writing 5 4 50 40 = 40 := by
  sorry

end writing_ways_equals_notebooks_l3382_338258


namespace power_two_greater_than_square_l3382_338253

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end power_two_greater_than_square_l3382_338253


namespace total_books_is_54_l3382_338212

/-- The total number of books Darla, Katie, and Gary have is 54 -/
theorem total_books_is_54 (darla_books : ℕ) (katie_books : ℕ) (gary_books : ℕ)
  (h1 : darla_books = 6)
  (h2 : katie_books = darla_books / 2)
  (h3 : gary_books = 5 * (darla_books + katie_books)) :
  darla_books + katie_books + gary_books = 54 := by
  sorry

end total_books_is_54_l3382_338212


namespace isabelle_concert_savings_l3382_338297

/-- The number of weeks Isabelle needs to work to afford concert tickets for herself and her brothers -/
def weeks_to_work (isabelle_ticket_cost : ℕ) (brother_ticket_cost : ℕ) (isabelle_savings : ℕ) (brothers_savings : ℕ) (weekly_pay : ℕ) : ℕ :=
  let total_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
  let total_savings := isabelle_savings + brothers_savings
  let remaining_cost := total_cost - total_savings
  remaining_cost / weekly_pay

theorem isabelle_concert_savings : weeks_to_work 20 10 5 5 3 = 10 := by
  sorry

end isabelle_concert_savings_l3382_338297


namespace equation_solutions_l3382_338265

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, (1/4) * (2*x + 3)^3 = 16 ↔ x = 1/2) :=
by sorry

end equation_solutions_l3382_338265


namespace jackson_souvenir_collection_l3382_338277

/-- Proves that given the conditions in Jackson's souvenir collection, 
    the number of starfish per spiral shell is 2. -/
theorem jackson_souvenir_collection 
  (hermit_crabs : ℕ) 
  (shells_per_crab : ℕ) 
  (total_souvenirs : ℕ) 
  (h1 : hermit_crabs = 45)
  (h2 : shells_per_crab = 3)
  (h3 : total_souvenirs = 450) :
  (total_souvenirs - hermit_crabs - hermit_crabs * shells_per_crab) / (hermit_crabs * shells_per_crab) = 2 :=
by sorry

end jackson_souvenir_collection_l3382_338277


namespace rectangle_formation_ways_l3382_338229

/-- The number of ways to choose 2 items from a set of 5 -/
def choose_2_from_5 : ℕ := 10

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The number of lines needed to form a rectangle -/
def lines_for_rectangle : ℕ := 4

/-- Theorem: The number of ways to choose 4 lines (2 horizontal and 2 vertical) 
    from 5 horizontal and 5 vertical lines to form a rectangle is 100 -/
theorem rectangle_formation_ways : 
  (choose_2_from_5 * choose_2_from_5 = 100) ∧ 
  (num_horizontal_lines = 5) ∧ 
  (num_vertical_lines = 5) ∧ 
  (lines_for_rectangle = 4) := by
  sorry

end rectangle_formation_ways_l3382_338229


namespace angle_c_is_right_angle_l3382_338200

theorem angle_c_is_right_angle (A B C : ℝ) (a b c : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (a > 0) → (b > 0) → (c > 0) →
  (A + B + C = π) →
  (a / Real.sin B + b / Real.sin A = 2 * c) →
  C = π / 2 := by
sorry

end angle_c_is_right_angle_l3382_338200


namespace elevator_time_l3382_338298

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 9

/-- Represents the number of steps per floor -/
def steps_per_floor : ℕ := 30

/-- Represents the number of steps Jake descends per second -/
def jake_steps_per_second : ℕ := 3

/-- Represents the time difference (in seconds) between Jake and the elevator reaching the ground floor -/
def time_difference : ℕ := 30

/-- Calculates the total number of steps Jake needs to descend -/
def total_steps : ℕ := (num_floors - 1) * steps_per_floor

/-- Calculates the time (in seconds) it takes Jake to reach the ground floor -/
def jake_time : ℕ := total_steps / jake_steps_per_second

/-- Theorem stating that the elevator takes 50 seconds to reach the ground level -/
theorem elevator_time : jake_time - time_difference = 50 := by sorry

end elevator_time_l3382_338298


namespace unique_x_l3382_338290

theorem unique_x : ∃! x : ℕ, 
  (∃ k : ℕ, x = 12 * k) ∧ 
  x^2 > 200 ∧ 
  x < 30 :=
by
  -- The proof would go here
  sorry

end unique_x_l3382_338290


namespace wrapping_paper_area_l3382_338210

/-- The area of a square sheet of wrapping paper required to wrap a rectangular box -/
theorem wrapping_paper_area (l w h : ℝ) (h_positive : l > 0 ∧ w > 0 ∧ h > 0) 
  (h_different : h ≠ l ∧ h ≠ w) : 
  let side_length := l + w
  (side_length ^ 2 : ℝ) = (l + w) ^ 2 := by
  sorry

end wrapping_paper_area_l3382_338210


namespace percentage_failed_both_subjects_l3382_338232

theorem percentage_failed_both_subjects 
  (failed_hindi : Real) 
  (failed_english : Real) 
  (passed_both : Real) 
  (h1 : failed_hindi = 32) 
  (h2 : failed_english = 56) 
  (h3 : passed_both = 24) : 
  Real := by
  sorry

end percentage_failed_both_subjects_l3382_338232


namespace zhuge_liang_army_count_l3382_338257

theorem zhuge_liang_army_count : 
  let n := 8
  let sum := n + n^2 + n^3 + n^4 + n^5 + n^6
  sum = (1 / 7) * (n^7 - n) := by
  sorry

end zhuge_liang_army_count_l3382_338257


namespace find_a_l3382_338206

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem find_a : ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3 ↔ x ∈ solution_set a) → a = 2 := by
  sorry

end find_a_l3382_338206


namespace square_difference_given_sum_and_product_l3382_338278

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 := by
  sorry

end square_difference_given_sum_and_product_l3382_338278


namespace consecutive_letters_probability_l3382_338246

/-- The number of cards in the deck -/
def n : ℕ := 5

/-- The number of cards to draw -/
def k : ℕ := 2

/-- The number of ways to choose k cards from n cards -/
def total_outcomes : ℕ := n.choose k

/-- The number of pairs of consecutive letters -/
def favorable_outcomes : ℕ := n - 1

/-- The probability of drawing 2 cards with consecutive letters -/
def probability : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of drawing 2 cards with consecutive letters is 2/5 -/
theorem consecutive_letters_probability :
  probability = 2 / 5 := by sorry

end consecutive_letters_probability_l3382_338246


namespace sqrt_product_equality_l3382_338243

theorem sqrt_product_equality (x : ℝ) : 
  Real.sqrt (x * (x - 6)) = Real.sqrt x * Real.sqrt (x - 6) → x ≥ 6 := by
  sorry

end sqrt_product_equality_l3382_338243


namespace isosceles_base_length_l3382_338238

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of one of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : y + 2*x = 20
  /-- x is positive and less than 10 -/
  xBound : 0 < x ∧ x < 10
  /-- y is positive -/
  yPositive : y > 0

/-- The base length of an isosceles triangle with perimeter 20 is 20 - 2x, where 5 < x < 10 -/
theorem isosceles_base_length (t : IsoscelesTriangle) : 
  t.y = 20 - 2*t.x ∧ 5 < t.x ∧ t.x < 10 := by
  sorry


end isosceles_base_length_l3382_338238


namespace prob_b_is_point_four_l3382_338284

/-- Given two events a and b, prove that the probability of b is 0.4 -/
theorem prob_b_is_point_four (a b : Set α) (p : Set α → ℝ) 
  (h1 : p a = 2/5)
  (h2 : p (a ∩ b) = 0.16000000000000003)
  (h3 : p (a ∩ b) = p a * p b) : 
  p b = 0.4 := by
  sorry

end prob_b_is_point_four_l3382_338284


namespace bethany_saw_80_paintings_l3382_338231

/-- The number of paintings Bethany saw at the museum -/
structure MuseumVisit where
  portraits : ℕ
  stillLifes : ℕ

/-- Bethany's visit to the museum satisfies the given conditions -/
def bethanysVisit : MuseumVisit where
  portraits := 16
  stillLifes := 4 * 16

/-- The total number of paintings Bethany saw -/
def totalPaintings (visit : MuseumVisit) : ℕ :=
  visit.portraits + visit.stillLifes

/-- Theorem stating that Bethany saw 80 paintings in total -/
theorem bethany_saw_80_paintings :
  totalPaintings bethanysVisit = 80 := by
  sorry

end bethany_saw_80_paintings_l3382_338231


namespace division_remainder_l3382_338254

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 23 →
  divisor = 4 →
  quotient = 5 →
  dividend = divisor * quotient + remainder →
  remainder = 3 := by
sorry

end division_remainder_l3382_338254


namespace function_inequality_l3382_338245

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem function_inequality (h1 : IsEven f) (h2 : MonoDecreasing (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end function_inequality_l3382_338245


namespace thirty_percent_more_than_hundred_l3382_338266

theorem thirty_percent_more_than_hundred (x : ℝ) : x + (1/4) * x = 130 → x = 104 := by
  sorry

end thirty_percent_more_than_hundred_l3382_338266


namespace not_both_nonstandard_l3382_338219

def IntegerFunction (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def NonStandard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_nonstandard (G : ℤ → ℤ) (h : IntegerFunction G) :
  ¬(NonStandard G 267 ∧ NonStandard G 269) := by
  sorry

end not_both_nonstandard_l3382_338219


namespace tangent_line_to_exp_curve_l3382_338244

/-- The value of k for which the line y = kx is tangent to the curve y = e^x -/
theorem tangent_line_to_exp_curve (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = Real.exp x₀ ∧ k = Real.exp x₀) → k = Real.exp 1 :=
by sorry

end tangent_line_to_exp_curve_l3382_338244


namespace domain_transformation_l3382_338205

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc 0 4

-- Define the domain of f(x²)
def domain_f_squared : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem domain_transformation (hf : ∀ x ∈ domain_f, f x ≠ 0) :
  {x : ℝ | f (x^2) ≠ 0} = domain_f_squared := by sorry

end domain_transformation_l3382_338205


namespace derivative_of_f_l3382_338252

-- Define the function
def f (x : ℝ) : ℝ := (5 * x - 3) ^ 3

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 15 * (5 * x - 3) ^ 2 := by sorry

end derivative_of_f_l3382_338252


namespace total_shingles_needed_l3382_338201

/-- Represents the dimensions of a rectangular roof side -/
structure RoofSide where
  length : ℕ
  width : ℕ

/-- Represents a roof with two identical slanted sides and shingle requirement -/
structure Roof where
  side : RoofSide
  shingles_per_sqft : ℕ

/-- Calculates the number of shingles needed for a roof -/
def shingles_needed (roof : Roof) : ℕ :=
  2 * roof.side.length * roof.side.width * roof.shingles_per_sqft

/-- The three roofs in the problem -/
def roof_A : Roof := { side := { length := 20, width := 40 }, shingles_per_sqft := 8 }
def roof_B : Roof := { side := { length := 25, width := 35 }, shingles_per_sqft := 10 }
def roof_C : Roof := { side := { length := 30, width := 30 }, shingles_per_sqft := 12 }

/-- Theorem stating the total number of shingles needed for all three roofs -/
theorem total_shingles_needed :
  shingles_needed roof_A + shingles_needed roof_B + shingles_needed roof_C = 51900 := by
  sorry

end total_shingles_needed_l3382_338201


namespace parallel_linear_functions_min_value_l3382_338227

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_min (h : ℝ → ℝ) : ℝ := sorry

theorem parallel_linear_functions_min_value 
  (funcs : ParallelLinearFunctions) 
  (h_min : quadratic_min (λ x => (funcs.f x)^2 + 2 * funcs.g x) = 5) :
  quadratic_min (λ x => (funcs.g x)^2 + 2 * funcs.f x) = -7 := by
  sorry

end parallel_linear_functions_min_value_l3382_338227


namespace at_least_100_triangles_l3382_338264

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersection : Bool

/-- Calculates the number of triangular regions formed by the lines -/
def num_triangular_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem stating that for 300 lines with given conditions, there are at least 100 triangular regions -/
theorem at_least_100_triangles (config : LineConfiguration) 
  (h1 : config.num_lines = 300)
  (h2 : config.no_parallel = true)
  (h3 : config.no_triple_intersection = true) :
  num_triangular_regions config ≥ 100 := by
  sorry

end at_least_100_triangles_l3382_338264


namespace normal_distribution_probability_l3382_338240

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for the normal distribution -/
noncomputable def prob (X : NormalDistribution) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability (X : NormalDistribution) 
  (h1 : X.μ = 4)
  (h2 : X.σ = 1)
  (h3 : prob X (X.μ - 2*X.σ) (X.μ + 2*X.σ) = 0.9544)
  (h4 : prob X (X.μ - X.σ) (X.μ + X.σ) = 0.6826) :
  prob X 5 6 = 0.1359 := by
  sorry

end normal_distribution_probability_l3382_338240


namespace range_of_p_or_q_range_of_a_intersection_l3382_338237

-- Define sets A, B, and C
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) : Prop := x ∈ B

-- Theorem 1: The set of x satisfying p ∨ q is equal to [-2, 5)
theorem range_of_p_or_q : {x : ℝ | p x ∨ q x} = Set.Ico (-2) 5 := by sorry

-- Theorem 2: The set of a satisfying A ∩ C = C is equal to (-∞, -4] ∪ [-1, 1/2]
theorem range_of_a_intersection : 
  {a : ℝ | A ∩ C a = C a} = Set.Iic (-4) ∪ Set.Icc (-1) (1/2) := by sorry

end range_of_p_or_q_range_of_a_intersection_l3382_338237


namespace lcm_from_hcf_and_product_l3382_338296

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 9 → a * b = 1800 → Nat.lcm a b = 200 := by
  sorry

end lcm_from_hcf_and_product_l3382_338296


namespace unique_square_number_l3382_338202

theorem unique_square_number : ∃! x : ℕ, 
  x > 39 ∧ x < 80 ∧ 
  ∃ y : ℕ, x = y * y ∧ 
  ∃ z : ℕ, x = 4 * z :=
by
  sorry

end unique_square_number_l3382_338202


namespace min_m_plus_n_l3382_338247

theorem min_m_plus_n (m n : ℕ+) (h : 90 * m = n^3) : 
  ∃ (m' n' : ℕ+), 90 * m' = n'^3 ∧ m' + n' ≤ m + n ∧ m' + n' = 120 :=
sorry

end min_m_plus_n_l3382_338247


namespace circle_equation_l3382_338204

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

/-- Tangent line -/
def tangent_line (y : ℝ) : Prop :=
  y = 0

/-- Possible equations of the sought circle -/
def sought_circle (x y : ℝ) : Prop :=
  ((x - 2 - 2*Real.sqrt 10)^2 + (y - 4)^2 = 16) ∨
  ((x - 2 + 2*Real.sqrt 10)^2 + (y - 4)^2 = 16) ∨
  ((x - 2 - 2*Real.sqrt 6)^2 + (y + 4)^2 = 16) ∨
  ((x - 2 + 2*Real.sqrt 6)^2 + (y + 4)^2 = 16)

/-- Theorem stating the properties of the sought circle -/
theorem circle_equation :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = 16) ∧
    (∃ (x y : ℝ), given_circle x y ∧ (x - a)^2 + (y - b)^2 = 36) ∧
    (∃ y : ℝ, tangent_line y ∧ (a - a)^2 + (y - b)^2 = 16) →
    sought_circle a b :=
by sorry

end circle_equation_l3382_338204


namespace linear_transformation_uniqueness_l3382_338214

theorem linear_transformation_uniqueness (z₁ z₂ w₁ w₂ : ℂ) 
  (h₁ : z₁ ≠ z₂) (h₂ : w₁ ≠ w₂) :
  ∃! (a b : ℂ), (a * z₁ + b = w₁) ∧ (a * z₂ + b = w₂) := by
  sorry

end linear_transformation_uniqueness_l3382_338214


namespace min_value_of_expression_l3382_338275

/-- Given vectors a = (x, 2) and b = (1, y) where x > 0, y > 0, and a ⋅ b = 1,
    the minimum value of 1/x + 2/y is 35/6 -/
theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_dot_product : x * 1 + 2 * y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' * 1 + 2 * y' = 1 → 1 / x + 2 / y ≤ 1 / x' + 2 / y') ∧
  1 / x + 2 / y = 35 / 6 := by
sorry

end min_value_of_expression_l3382_338275


namespace toy_distribution_ratio_l3382_338241

theorem toy_distribution_ratio (total_toys : ℕ) (num_friends : ℕ) 
  (h1 : total_toys = 118) (h2 : num_friends = 4) :
  (total_toys / num_friends) / total_toys = 1 / 4 := by
  sorry

end toy_distribution_ratio_l3382_338241


namespace complex_exp_13pi_over_2_l3382_338269

theorem complex_exp_13pi_over_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end complex_exp_13pi_over_2_l3382_338269


namespace orange_ribbons_l3382_338248

theorem orange_ribbons (total : ℚ) 
  (yellow_frac : ℚ) (purple_frac : ℚ) (orange_frac : ℚ) (black_count : ℕ) :
  yellow_frac = 1/3 →
  purple_frac = 1/4 →
  orange_frac = 1/6 →
  black_count = 40 →
  (1 - yellow_frac - purple_frac - orange_frac) * total = black_count →
  orange_frac * total = 80/3 := by
sorry

end orange_ribbons_l3382_338248


namespace geometric_sequence_common_ratio_l3382_338226

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ (n : ℕ), a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_sum : a 1 * a 5 + 2 * a 3 * a 7 + a 5 * a 9 = 16)
  (h_mean : (a 5 + a 9) / 2 = 4) :
  ∃ (q : ℝ), q > 0 ∧ (∀ (n : ℕ), a (n + 1) = q * a n) ∧ q = Real.sqrt 2 :=
sorry

end geometric_sequence_common_ratio_l3382_338226


namespace lines_are_parallel_l3382_338256

/-- Two lines in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem lines_are_parallel (l1 l2 : Line) 
  (h1 : l1 = { slope := 2, intercept := 1 })
  (h2 : l2 = { slope := 2, intercept := 5 }) : 
  parallel l1 l2 := by sorry

end lines_are_parallel_l3382_338256


namespace smallest_four_digit_divisible_by_53_l3382_338208

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l3382_338208


namespace four_correct_propositions_l3382_338217

theorem four_correct_propositions : 
  (2 ≤ 3) ∧ 
  (∀ m : ℝ, m ≥ 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧ 
  (∀ x y : ℝ, x^2 = y^2 → |x| = |y|) ∧ 
  (∀ a b c : ℝ, a > b ↔ a + c > b + c) := by
  sorry

end four_correct_propositions_l3382_338217


namespace sin_pi_half_plus_alpha_l3382_338291

/-- Given a point P(-4, 3) on the terminal side of angle α, prove that sin(π/2 + α) = -4/5 -/
theorem sin_pi_half_plus_alpha (α : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  Real.sin (π / 2 + α) = -4 / 5 := by
  sorry

end sin_pi_half_plus_alpha_l3382_338291


namespace no_perfect_squares_l3382_338260

def sequence_x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * sequence_x (n + 1) - sequence_x n

theorem no_perfect_squares (n : ℕ) (h : n ≥ 1) :
  ¬ ∃ m : ℤ, sequence_x n = m * m := by
  sorry

end no_perfect_squares_l3382_338260


namespace average_study_time_difference_l3382_338263

def daily_differences : List Int := [15, -5, 25, -10, 5, 20, -15]

def days_in_week : Nat := 7

theorem average_study_time_difference :
  (daily_differences.sum : ℚ) / days_in_week = 5 := by sorry

end average_study_time_difference_l3382_338263


namespace regular_polygon_sides_l3382_338224

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  (n ≥ 3) → 
  (interior_angle = 144) → 
  (interior_angle = (n - 2) * 180 / n) →
  n = 10 := by
sorry

end regular_polygon_sides_l3382_338224


namespace equation_solution_l3382_338221

theorem equation_solution : ∃ x : ℚ, (-2*x + 3 - 2*x + 3 = 3*x - 6) ∧ (x = 12/7) := by
  sorry

end equation_solution_l3382_338221


namespace cube_root_function_l3382_338285

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 125, 
    prove that y = 8√3/5 when x = 8 -/
theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → (k * x^(1/3) = 4 * Real.sqrt 3 ↔ x = 125)) → 
  k * 8^(1/3) = 8 * Real.sqrt 3 / 5 := by
  sorry

end cube_root_function_l3382_338285


namespace merchant_scale_problem_merchant_loss_l3382_338262

theorem merchant_scale_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  m / n + n / m > 2 :=
sorry

theorem merchant_loss (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  let x := n / m
  let y := m / n
  x + y > 2 :=
sorry

end merchant_scale_problem_merchant_loss_l3382_338262


namespace log_216_equals_3_log_36_l3382_338235

theorem log_216_equals_3_log_36 : Real.log 216 = 3 * Real.log 36 := by
  sorry

end log_216_equals_3_log_36_l3382_338235


namespace hyperbola_triangle_perimeter_l3382_338203

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a point on the hyperbola
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h1 : on_hyperbola A) 
  (h2 : on_hyperbola B) 
  (h3 : A.1 < 0 ∧ B.1 < 0)  -- A and B are on the left branch
  (h4 : ∃ t : ℝ, A.1 = (1 - t) * left_focus.1 + t * B.1 ∧ 
               A.2 = (1 - t) * left_focus.2 + t * B.2)  -- AB passes through left focus
  (h5 : distance A B = 5) :
  distance A right_focus + distance B right_focus + distance A B = 26 :=
sorry

end hyperbola_triangle_perimeter_l3382_338203


namespace power_of_power_at_three_l3382_338273

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end power_of_power_at_three_l3382_338273


namespace anna_rearrangement_time_l3382_338236

def name : String := "Anna"
def letters : ℕ := 4
def repetitions : List ℕ := [2, 2]  -- 'A' repeated twice, 'N' repeated twice
def rearrangements_per_minute : ℕ := 8

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_rearrangements : ℕ :=
  factorial letters / (factorial repetitions[0]! * factorial repetitions[1]!)

def time_in_minutes : ℚ :=
  total_rearrangements / rearrangements_per_minute

theorem anna_rearrangement_time :
  time_in_minutes / 60 = 0.0125 := by sorry

end anna_rearrangement_time_l3382_338236


namespace quadratic_always_greater_than_ten_l3382_338233

theorem quadratic_always_greater_than_ten (k : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + k > 10) ↔ k > 11 := by sorry

end quadratic_always_greater_than_ten_l3382_338233


namespace relay_team_permutations_l3382_338289

theorem relay_team_permutations :
  (Finset.range 4).card.factorial = 24 := by
  sorry

end relay_team_permutations_l3382_338289


namespace geometric_arithmetic_sequence_problem_l3382_338211

theorem geometric_arithmetic_sequence_problem :
  ∃ (a b c : ℝ) (d : ℝ),
    a + b + c = 114 ∧
    b^2 = a * c ∧
    b ≠ a ∧
    b = a + 3 * d ∧
    c = a + 24 * d ∧
    a = 2 ∧
    b = 14 ∧
    c = 98 := by
  sorry

end geometric_arithmetic_sequence_problem_l3382_338211


namespace complex_equality_l3382_338281

theorem complex_equality (x y : ℝ) (i : ℂ) (h : i * i = -1) :
  (x + y * i : ℂ) = 1 / i → x + y = -1 := by
  sorry

end complex_equality_l3382_338281


namespace rocks_theorem_l3382_338288

def rocks_problem (initial_rocks : ℕ) (eaten_fraction : ℚ) (retrieved_rocks : ℕ) : Prop :=
  let remaining_after_eating := initial_rocks - (initial_rocks * eaten_fraction).floor
  let final_rocks := remaining_after_eating + retrieved_rocks
  initial_rocks = 10 ∧ eaten_fraction = 1/2 ∧ retrieved_rocks = 2 → final_rocks = 7

theorem rocks_theorem : rocks_problem 10 (1/2) 2 := by
  sorry

end rocks_theorem_l3382_338288


namespace max_value_on_interval_l3382_338223

/-- The function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The closed interval [0, 3] -/
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ I ∧ f c = 6 ∧ ∀ x ∈ I, f x ≤ 6 :=
sorry

end max_value_on_interval_l3382_338223


namespace cube_equation_solution_l3382_338282

theorem cube_equation_solution : ∃! x : ℝ, (x - 3)^3 = 27 ∧ x = 6 := by sorry

end cube_equation_solution_l3382_338282


namespace omitted_number_proof_l3382_338293

/-- Sequence of even numbers starting from 2 -/
def evenSeq (n : ℕ) : ℕ := 2 * n

/-- Sum of even numbers from 2 to 2n -/
def evenSum (n : ℕ) : ℕ := n * (n + 1)

/-- The incorrect sum obtained -/
def incorrectSum : ℕ := 2014

/-- The omitted number -/
def omittedNumber : ℕ := 56

theorem omitted_number_proof :
  ∃ n : ℕ, evenSum n - incorrectSum = omittedNumber ∧
  evenSeq (n + 1) = omittedNumber :=
sorry

end omitted_number_proof_l3382_338293


namespace leilas_savings_l3382_338270

theorem leilas_savings (savings : ℚ) : 
  (3 / 4 : ℚ) * savings + 20 = savings → savings = 80 :=
by sorry

end leilas_savings_l3382_338270


namespace sqrt_a_plus_b_is_three_l3382_338234

theorem sqrt_a_plus_b_is_three (a b : ℝ) 
  (h1 : 2*a - 1 = 9) 
  (h2 : 3*a + 2*b + 4 = 27) : 
  Real.sqrt (a + b) = 3 := by
  sorry

end sqrt_a_plus_b_is_three_l3382_338234
