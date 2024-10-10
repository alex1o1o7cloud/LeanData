import Mathlib

namespace ball_probabilities_l1061_106114

/-- Represents the box of balls -/
structure BallBox where
  red_balls : ℕ
  white_balls : ℕ

/-- The probability of drawing exactly one red ball and one white ball without replacement -/
def prob_one_red_one_white (box : BallBox) : ℚ :=
  let total := box.red_balls + box.white_balls
  (box.red_balls : ℚ) / total * (box.white_balls : ℚ) / (total - 1) +
  (box.white_balls : ℚ) / total * (box.red_balls : ℚ) / (total - 1)

/-- The probability of getting at least one red ball in three draws with replacement -/
def prob_at_least_one_red (box : BallBox) : ℚ :=
  let p_red := (box.red_balls : ℚ) / (box.red_balls + box.white_balls)
  1 - (1 - p_red) ^ 3

theorem ball_probabilities (box : BallBox) (h1 : box.red_balls = 2) (h2 : box.white_balls = 4) :
  prob_one_red_one_white box = 8/15 ∧ prob_at_least_one_red box = 19/27 := by
  sorry


end ball_probabilities_l1061_106114


namespace y_value_l1061_106137

-- Define the property for y
def satisfies_condition (y : ℝ) : Prop :=
  y = (1 / y) * (-y) - 3

-- Theorem statement
theorem y_value : ∃ y : ℝ, satisfies_condition y ∧ y = -4 := by
  sorry

end y_value_l1061_106137


namespace solution_sum_l1061_106163

-- Define the solution set
def SolutionSet : Set ℝ := Set.union (Set.Iio 1) (Set.Ioi 4)

-- Define the theorem
theorem solution_sum (a b : ℝ) 
  (h : ∀ x, x ∈ SolutionSet ↔ (x - a) / (x - b) > 0) : 
  a + b = 5 := by sorry

end solution_sum_l1061_106163


namespace complex_fraction_sum_l1061_106131

theorem complex_fraction_sum : (1 - 2*Complex.I) / (1 + Complex.I) + (1 + 2*Complex.I) / (1 - Complex.I) = -1 := by
  sorry

end complex_fraction_sum_l1061_106131


namespace job_completion_time_l1061_106177

/-- The time it takes for three workers to complete a job together, given their individual efficiencies -/
theorem job_completion_time 
  (sakshi_time : ℝ) 
  (tanya_efficiency : ℝ) 
  (rahul_efficiency : ℝ) 
  (h1 : sakshi_time = 5) 
  (h2 : tanya_efficiency = 1.25) 
  (h3 : rahul_efficiency = 0.6) : 
  (1 / (1 / sakshi_time + tanya_efficiency / sakshi_time + rahul_efficiency / sakshi_time)) = 100 / 57 := by
  sorry

end job_completion_time_l1061_106177


namespace intersection_when_a_is_one_b_proper_subset_of_a_iff_a_in_range_l1061_106174

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part (1)
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem b_proper_subset_of_a_iff_a_in_range (a : ℝ) :
  B a ⊂ A ↔ (0 ≤ a ∧ a ≤ 1) ∨ a > 2 := by sorry

end intersection_when_a_is_one_b_proper_subset_of_a_iff_a_in_range_l1061_106174


namespace real_part_of_z_l1061_106153

theorem real_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = -1) : 
  Complex.re z = -1/2 := by
  sorry

end real_part_of_z_l1061_106153


namespace smallest_k_for_64k_gt_4_20_l1061_106123

theorem smallest_k_for_64k_gt_4_20 : ∃ k : ℕ, k = 7 ∧ 64^k > 4^20 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^20 := by
  sorry

end smallest_k_for_64k_gt_4_20_l1061_106123


namespace unique_solution_l1061_106151

theorem unique_solution (x y z : ℝ) 
  (h1 : x + y^2 + z^3 = 3)
  (h2 : y + z^2 + x^3 = 3)
  (h3 : z + x^2 + y^3 = 3)
  (px : x > 0)
  (py : y > 0)
  (pz : z > 0) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_solution_l1061_106151


namespace equal_distribution_of_cards_l1061_106111

theorem equal_distribution_of_cards (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 455) (h2 : num_friends = 5) :
  total_cards / num_friends = 91 :=
by sorry

end equal_distribution_of_cards_l1061_106111


namespace max_expression_value_l1061_106193

/-- Represents the count of integers equal to each value from 1 to 2003 -/
def IntegerCounts := Fin 2003 → ℕ

/-- The sum of all integers is 2003 -/
def SumConstraint (counts : IntegerCounts) : Prop :=
  (Finset.range 2003).sum (fun i => (i + 1) * counts i) = 2003

/-- The expression to be maximized -/
def ExpressionToMaximize (counts : IntegerCounts) : ℕ :=
  (Finset.range 2002).sum (fun i => i * counts (i + 1))

/-- There are at least two integers in the set -/
def AtLeastTwoIntegers (counts : IntegerCounts) : Prop :=
  (Finset.range 2003).sum (fun i => counts i) ≥ 2

theorem max_expression_value (counts : IntegerCounts) 
  (h1 : SumConstraint counts) (h2 : AtLeastTwoIntegers counts) :
  ExpressionToMaximize counts ≤ 2001 := by
  sorry

end max_expression_value_l1061_106193


namespace B_grazed_five_months_l1061_106149

/-- Represents the number of months B grazed his cows -/
def B_months : ℕ := sorry

/-- Total rent of the field in rupees -/
def total_rent : ℕ := 3250

/-- A's share of rent in rupees -/
def A_rent : ℕ := 720

/-- Number of cows grazed by each milkman -/
def cows : Fin 4 → ℕ
| 0 => 24  -- A
| 1 => 10  -- B
| 2 => 35  -- C
| 3 => 21  -- D

/-- Number of months each milkman grazed their cows -/
def months : Fin 4 → ℕ
| 0 => 3         -- A
| 1 => B_months  -- B
| 2 => 4         -- C
| 3 => 3         -- D

/-- Total cow-months for all milkmen -/
def total_cow_months : ℕ := 
  (cows 0 * months 0) + (cows 1 * months 1) + (cows 2 * months 2) + (cows 3 * months 3)

theorem B_grazed_five_months : B_months = 5 := by
  sorry

end B_grazed_five_months_l1061_106149


namespace scoring_ratio_is_two_to_one_l1061_106181

/-- Represents the scoring system for a test -/
structure TestScoring where
  totalQuestions : ℕ
  correctAnswers : ℕ
  score : ℕ
  scoringRatio : ℚ

/-- Calculates the score based on correct answers, incorrect answers, and the scoring ratio -/
def calculateScore (correct : ℕ) (incorrect : ℕ) (ratio : ℚ) : ℚ :=
  correct - ratio * incorrect

/-- Theorem stating that the scoring ratio is 2:1 for the given test conditions -/
theorem scoring_ratio_is_two_to_one (t : TestScoring)
    (h1 : t.totalQuestions = 100)
    (h2 : t.correctAnswers = 91)
    (h3 : t.score = 73)
    (h4 : calculateScore t.correctAnswers (t.totalQuestions - t.correctAnswers) t.scoringRatio = t.score) :
    t.scoringRatio = 2 := by
  sorry


end scoring_ratio_is_two_to_one_l1061_106181


namespace counterexample_exists_l1061_106122

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n % 27 = 0) ∧ 
  (n % 27 ≠ 0) ∧ 
  (n = 81 ∨ n = 999 ∨ n = 9918 ∨ n = 18) := by
  sorry

end counterexample_exists_l1061_106122


namespace toothpicks_250th_stage_l1061_106109

/-- The nth term of an arithmetic sequence -/
def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  arithmeticSequence 4 3 n

theorem toothpicks_250th_stage :
  toothpicks 250 = 751 := by sorry

end toothpicks_250th_stage_l1061_106109


namespace fish_sales_revenue_l1061_106161

theorem fish_sales_revenue : 
  let first_week_quantity : ℕ := 50
  let first_week_price : ℚ := 10
  let second_week_quantity_multiplier : ℕ := 3
  let second_week_discount_percentage : ℚ := 25 / 100

  let first_week_revenue := first_week_quantity * first_week_price
  let second_week_quantity := first_week_quantity * second_week_quantity_multiplier
  let second_week_price := first_week_price * (1 - second_week_discount_percentage)
  let second_week_revenue := second_week_quantity * second_week_price
  let total_revenue := first_week_revenue + second_week_revenue

  total_revenue = 1625 := by
sorry

end fish_sales_revenue_l1061_106161


namespace quadratic_form_sum_l1061_106160

theorem quadratic_form_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 171) := by
  sorry

end quadratic_form_sum_l1061_106160


namespace stock_price_after_two_years_stock_price_calculation_l1061_106133

theorem stock_price_after_two_years 
  (initial_price : ℝ) 
  (first_year_increase : ℝ) 
  (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price

theorem stock_price_calculation : 
  stock_price_after_two_years 120 1 0.3 = 168 := by
  sorry

end stock_price_after_two_years_stock_price_calculation_l1061_106133


namespace weight_of_b_l1061_106120

-- Define the variables
variable (wa wb wc : ℝ)
variable (ha hb hc : ℝ)

-- Define the conditions
def average_weight_abc : Prop := (wa + wb + wc) / 3 = 45
def average_weight_ab : Prop := (wa + wb) / 2 = 40
def average_weight_bc : Prop := (wb + wc) / 2 = 43
def average_height_ac : Prop := (ha + hc) / 2 = 155

-- Define the quadratic relationship
def weight_height_relation (w h : ℝ) : Prop := w = 2 * h^2 + 3 * h - 5

-- Theorem statement
theorem weight_of_b (hwabc : average_weight_abc wa wb wc)
                    (hwab : average_weight_ab wa wb)
                    (hwbc : average_weight_bc wb wc)
                    (hhac : average_height_ac ha hc)
                    (hwa : weight_height_relation wa ha)
                    (hwb : weight_height_relation wb hb)
                    (hwc : weight_height_relation wc hc) :
  wb = 31 := by sorry

end weight_of_b_l1061_106120


namespace oliver_bill_denomination_l1061_106138

/-- The denomination of Oliver's unknown bills -/
def x : ℕ := sorry

/-- Oliver's total money -/
def oliver_money : ℕ := 10 * x + 3 * 5

/-- William's total money -/
def william_money : ℕ := 15 * 10 + 4 * 5

theorem oliver_bill_denomination :
  (oliver_money = william_money + 45) → x = 20 := by sorry

end oliver_bill_denomination_l1061_106138


namespace chocos_remainder_l1061_106117

theorem chocos_remainder (n : ℕ) (h : n % 11 = 5) : (4 * n) % 11 = 9 := by
  sorry

end chocos_remainder_l1061_106117


namespace count_negative_numbers_l1061_106185

theorem count_negative_numbers : ∃ (negative_count : ℕ), 
  negative_count = 2 ∧ 
  negative_count = (if (-1 : ℚ)^2007 < 0 then 1 else 0) + 
                   (if (|(-1 : ℚ)|^3 : ℚ) < 0 then 1 else 0) + 
                   (if (-1 : ℚ)^18 > 0 then 1 else 0) + 
                   (if (18 : ℚ) < 0 then 1 else 0) := by
  sorry

end count_negative_numbers_l1061_106185


namespace supplier_payment_proof_l1061_106178

/-- Calculates the amount paid to a supplier given initial funds, received payment, expenses, and final amount -/
def amount_paid_to_supplier (initial_funds : ℤ) (received_payment : ℤ) (expenses : ℤ) (final_amount : ℤ) : ℤ :=
  initial_funds + received_payment - expenses - final_amount

/-- Proves that the amount paid to the supplier is 600 given the problem conditions -/
theorem supplier_payment_proof (initial_funds : ℤ) (received_payment : ℤ) (expenses : ℤ) (final_amount : ℤ)
  (h1 : initial_funds = 2000)
  (h2 : received_payment = 800)
  (h3 : expenses = 1200)
  (h4 : final_amount = 1000) :
  amount_paid_to_supplier initial_funds received_payment expenses final_amount = 600 := by
  sorry

#eval amount_paid_to_supplier 2000 800 1200 1000

end supplier_payment_proof_l1061_106178


namespace probability_both_selected_l1061_106157

theorem probability_both_selected (p_ram p_ravi : ℚ) 
  (h1 : p_ram = 6/7) (h2 : p_ravi = 1/5) : 
  p_ram * p_ravi = 6/35 := by
  sorry

end probability_both_selected_l1061_106157


namespace sum_of_specific_numbers_l1061_106169

theorem sum_of_specific_numbers : 3 + 33 + 333 + 33.3 = 402.3 := by
  sorry

end sum_of_specific_numbers_l1061_106169


namespace cylinder_surface_area_l1061_106124

/-- The surface area of a cylinder with base radius 2 and lateral surface length
    equal to the diameter of the base is 24π. -/
theorem cylinder_surface_area : 
  let r : ℝ := 2
  let l : ℝ := 2 * r
  let surface_area : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  surface_area = 24 * Real.pi :=
by sorry

end cylinder_surface_area_l1061_106124


namespace paul_school_supplies_l1061_106142

theorem paul_school_supplies : 
  let initial_regular_erasers : ℕ := 307
  let initial_jumbo_erasers : ℕ := 150
  let initial_standard_crayons : ℕ := 317
  let initial_jumbo_crayons : ℕ := 300
  let lost_regular_erasers : ℕ := 52
  let used_standard_crayons : ℕ := 123
  let used_jumbo_crayons : ℕ := 198

  let remaining_regular_erasers : ℕ := initial_regular_erasers - lost_regular_erasers
  let remaining_jumbo_erasers : ℕ := initial_jumbo_erasers
  let remaining_standard_crayons : ℕ := initial_standard_crayons - used_standard_crayons
  let remaining_jumbo_crayons : ℕ := initial_jumbo_crayons - used_jumbo_crayons

  let total_remaining_erasers : ℕ := remaining_regular_erasers + remaining_jumbo_erasers
  let total_remaining_crayons : ℕ := remaining_standard_crayons + remaining_jumbo_crayons

  (total_remaining_crayons : ℤ) - (total_remaining_erasers : ℤ) = -109
  := by sorry

end paul_school_supplies_l1061_106142


namespace equation_equivalence_l1061_106119

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0

-- Define the equivalent quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  9 * x^2 - 26 * x - 12 = 0

-- Theorem stating the equivalence of the two equations
theorem equation_equivalence :
  ∀ x : ℝ, original_equation x ↔ quadratic_equation x :=
by sorry

end equation_equivalence_l1061_106119


namespace remaining_balance_is_correct_l1061_106144

-- Define the problem parameters
def initial_balance : ℚ := 100
def daily_spending : ℚ := 8
def exchange_fee_rate : ℚ := 0.03
def days_in_week : ℕ := 7
def flat_fee : ℚ := 2
def bill_denomination : ℚ := 5

-- Define the function to calculate the remaining balance
def calculate_remaining_balance : ℚ := 
  let total_daily_spend := daily_spending * (1 + exchange_fee_rate)
  let weekly_spend := total_daily_spend * days_in_week
  let balance_after_spending := initial_balance - weekly_spend
  let balance_after_fee := balance_after_spending - flat_fee
  let bills_taken := (balance_after_fee / bill_denomination).floor * bill_denomination
  balance_after_fee - bills_taken

-- Theorem statement
theorem remaining_balance_is_correct : 
  calculate_remaining_balance = 0.32 := by sorry

end remaining_balance_is_correct_l1061_106144


namespace sequence_conditions_l1061_106183

theorem sequence_conditions (a : ℝ) : 
  let a₁ : ℝ := 1
  let a₂ : ℝ := 1
  let a₃ : ℝ := 1
  let a₄ : ℝ := a
  let a₅ : ℝ := a
  (a₁ = a₂ * a₃) ∧ 
  (a₂ = a₁ * a₃) ∧ 
  (a₃ = a₁ * a₂) ∧ 
  (a₄ = a₁ * a₅) ∧ 
  (a₅ = a₁ * a₄) := by
sorry

end sequence_conditions_l1061_106183


namespace min_value_sqrt_reciprocal_l1061_106118

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) :
  2 * Real.sqrt (2 * x) + 4 / x ≥ 6 ∧
  (2 * Real.sqrt (2 * x) + 4 / x = 6 ↔ x = 2) :=
by sorry

end min_value_sqrt_reciprocal_l1061_106118


namespace expression_value_l1061_106179

theorem expression_value (a b : ℝ) (h : a + 3*b = 0) : 
  a^3 + 3*a^2*b - 2*a - 6*b - 5 = -5 := by
sorry

end expression_value_l1061_106179


namespace x_value_l1061_106121

theorem x_value : ∃ x : ℝ, (0.4 * x = (1/3) * x + 110) ∧ (x = 1650) := by
  sorry

end x_value_l1061_106121


namespace ellipse_major_axis_length_l1061_106195

/-- Represents an ellipse with equation x²/(2m) + y²/m = 1, where m > 0 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / (2*m) + y^2 / m = 1
  m_pos : m > 0

/-- Represents a point on the ellipse -/
structure EllipsePoint (m : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / (2*m) + y^2 / m = 1

/-- The theorem stating that if an ellipse with equation x²/(2m) + y²/m = 1 (m > 0)
    is intersected by the line x = √m at two points with distance 2 between them,
    then the length of the major axis of the ellipse is 4 -/
theorem ellipse_major_axis_length 
  (m : ℝ) 
  (e : Ellipse m) 
  (A B : EllipsePoint m) 
  (h1 : A.x = Real.sqrt m) 
  (h2 : B.x = Real.sqrt m) 
  (h3 : (A.y - B.y)^2 = 4) : 
  ∃ (a : ℝ), a = 2 ∧ 2*a = 4 := by sorry

end ellipse_major_axis_length_l1061_106195


namespace color_one_third_square_l1061_106146

theorem color_one_third_square (n : ℕ) (k : ℕ) : n = 18 → k = 6 → Nat.choose n k = 18564 := by
  sorry

end color_one_third_square_l1061_106146


namespace hyperbola_eccentricity_l1061_106140

/-- A hyperbola with foci F₁ and F₂, and a point P on the hyperbola. -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points in ℝ² -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (angle_condition : angle h.F₁ h.P h.F₂ = π / 3)
  (distance_condition : distance h.P h.F₁ = 3 * distance h.P h.F₂) :
  eccentricity h = Real.sqrt 7 / 2 := by sorry

end hyperbola_eccentricity_l1061_106140


namespace hyperbola_circle_tangency_l1061_106154

/-- Given a hyperbola and a circle, if one asymptote of the hyperbola is tangent to the circle,
    then the ratio of the hyperbola's parameters is 3/4 -/
theorem hyperbola_circle_tangency (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ∧ 
   (∃ (t : ℝ), y = (a/b) * x + t) ∧
   (x - 2)^2 + (y - 1)^2 = 1) →
  b / a = 3 / 4 := by sorry

end hyperbola_circle_tangency_l1061_106154


namespace rotation_equivalence_l1061_106171

theorem rotation_equivalence (y : ℝ) : 
  (330 : ℝ) = (360 - y) → y < 360 → y = 30 := by sorry

end rotation_equivalence_l1061_106171


namespace ellipse_eccentricity_l1061_106164

/-- Given an ellipse with semi-major axis a and semi-minor axis b, where a > b > 0,
    and foci F₁ and F₂, a line passing through F₁ intersects the ellipse at points A and B.
    If AB ⟂ AF₂ and |AB| = |AF₂|, then the eccentricity of the ellipse is √6 - √3. -/
theorem ellipse_eccentricity (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) :
  a > b ∧ b > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ({F₁, F₂, A, B} : Set (ℝ × ℝ))) →
  (A.1 - B.1) * (A.1 - F₂.1) + (A.2 - B.2) * (A.2 - F₂.2) = 0 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - F₂.1)^2 + (A.2 - F₂.2)^2 →
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  e = Real.sqrt 6 - Real.sqrt 3 := by
sorry

end ellipse_eccentricity_l1061_106164


namespace sum_of_digits_power_of_nine_l1061_106108

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of 9^n is greater than 9 for all n ≥ 3 -/
theorem sum_of_digits_power_of_nine (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 := by
  sorry

end sum_of_digits_power_of_nine_l1061_106108


namespace smallest_natural_with_remainders_l1061_106194

theorem smallest_natural_with_remainders : ∃ N : ℕ,
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  (∀ M : ℕ, M < N →
    ¬((M % 9 = 8) ∧
      (M % 8 = 7) ∧
      (M % 7 = 6) ∧
      (M % 6 = 5) ∧
      (M % 5 = 4) ∧
      (M % 4 = 3) ∧
      (M % 3 = 2) ∧
      (M % 2 = 1))) ∧
  N = 2519 :=
by sorry

end smallest_natural_with_remainders_l1061_106194


namespace quadratic_expression_value_l1061_106182

theorem quadratic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x - 5 = 0) :
  4 * x^2 + 6 * x + 9 = 19 := by
  sorry

end quadratic_expression_value_l1061_106182


namespace financial_equation_balance_l1061_106187

theorem financial_equation_balance (f w p : ℂ) : 
  f = 10 → w = -10 + 250 * I → f * p - w = 8000 → p = 799 + 25 * I := by
  sorry

end financial_equation_balance_l1061_106187


namespace tangent_line_y_intercept_l1061_106130

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of a line tangent to two specific circles -/
def yIntercept (line : TangentLine) : ℝ :=
  sorry

/-- The main theorem stating the y-intercept of the tangent line -/
theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (2, 0), radius := 2 }
  let c2 : Circle := { center := (5, 0), radius := 1 }
  ∀ (line : TangentLine),
    line.circle1 = c1 →
    line.circle2 = c2 →
    line.tangentPoint1.1 > 2 →
    line.tangentPoint1.2 > 0 →
    line.tangentPoint2.1 > 5 →
    line.tangentPoint2.2 > 0 →
    yIntercept line = 2 * Real.sqrt 2 :=
  sorry

end tangent_line_y_intercept_l1061_106130


namespace difference_of_squares_representation_l1061_106110

theorem difference_of_squares_representation (n : ℕ) : 
  n = 2^4035 → 
  (∃ (count : ℕ), count = 2018 ∧ 
    (∃ (S : Finset (ℕ × ℕ)), 
      S.card = count ∧
      ∀ (pair : ℕ × ℕ), pair ∈ S ↔ 
        (∃ (a b : ℕ), pair = (a, b) ∧ n = a^2 - b^2))) :=
by sorry

end difference_of_squares_representation_l1061_106110


namespace triangle_angle_measure_l1061_106184

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a² + b² - c²), then the measure of angle C is π/3. -/
theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := Real.sqrt 3 / 4 * (a^2 + b^2 - c^2)
  S = 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) →
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry


end triangle_angle_measure_l1061_106184


namespace orange_slices_theorem_l1061_106167

/-- Represents the number of slices each animal received -/
structure OrangeSlices where
  siskin : ℕ
  hedgehog : ℕ
  beaver : ℕ

/-- Conditions for the orange slices distribution -/
def valid_distribution (slices : OrangeSlices) : Prop :=
  slices.hedgehog = 2 * slices.siskin ∧
  slices.beaver = 5 * slices.siskin ∧
  slices.beaver = slices.siskin + 8

/-- The total number of slices in the orange -/
def total_slices (slices : OrangeSlices) : ℕ :=
  slices.siskin + slices.hedgehog + slices.beaver

/-- Theorem stating that the total number of slices is 16 -/
theorem orange_slices_theorem :
  ∃ (slices : OrangeSlices), valid_distribution slices ∧ total_slices slices = 16 :=
sorry

end orange_slices_theorem_l1061_106167


namespace max_min_product_l1061_106188

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 20) (prod_sum_eq : x*y + y*z + z*x = 78) :
  ∃ (M : ℝ), M = min (x*y) (min (y*z) (z*x)) ∧ M ≤ 400/9 ∧
  ∀ (M' : ℝ), (∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x' + y' + z' = 20 ∧ x'*y' + y'*z' + z'*x' = 78 ∧
    M' = min (x'*y') (min (y'*z') (z'*x'))) → M' ≤ 400/9 :=
by sorry

end max_min_product_l1061_106188


namespace bicycle_trip_time_l1061_106158

/-- Proves that the time taken to go forth is 1 hour given the conditions of the bicycle problem -/
theorem bicycle_trip_time (speed_forth speed_back : ℝ) (time_diff : ℝ) 
  (h1 : speed_forth = 15)
  (h2 : speed_back = 10)
  (h3 : time_diff = 0.5)
  : ∃ (time_forth : ℝ), 
    speed_forth * time_forth = speed_back * (time_forth + time_diff) ∧ 
    time_forth = 1 := by
  sorry

end bicycle_trip_time_l1061_106158


namespace inequality_proof_l1061_106180

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l1061_106180


namespace intersection_in_second_quadrant_l1061_106168

/-- The intersection point of two lines is in the second quadrant if and only if k is in the open interval (0, 1/2) -/
theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k ∧ x < 0 ∧ y > 0) ↔ 0 < k ∧ k < 1/2 := by
  sorry

end intersection_in_second_quadrant_l1061_106168


namespace quadruple_equation_solutions_l1061_106116

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 1, 1, 2), (0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 0, 0)}

theorem quadruple_equation_solutions :
  ∀ x y z n : ℕ, is_solution x y z n ↔ (x, y, z, n) ∈ solution_set :=
by sorry

end quadruple_equation_solutions_l1061_106116


namespace food_distributor_comparison_l1061_106152

theorem food_distributor_comparison (p₁ p₂ : ℝ) 
  (h₁ : 0 < p₁) (h₂ : 0 < p₂) (h₃ : p₁ < p₂) :
  (2 * p₁ * p₂) / (p₁ + p₂) < (p₁ + p₂) / 2 := by
  sorry

end food_distributor_comparison_l1061_106152


namespace parking_ticket_multiple_l1061_106100

theorem parking_ticket_multiple (total_tickets : ℕ) (alan_tickets : ℕ) (marcy_tickets : ℕ) (m : ℕ) :
  total_tickets = 150 →
  alan_tickets = 26 →
  marcy_tickets = m * alan_tickets - 6 →
  total_tickets = alan_tickets + marcy_tickets →
  m = 5 := by
sorry

end parking_ticket_multiple_l1061_106100


namespace seventeen_is_possible_result_l1061_106132

def expression (op1 op2 op3 : ℕ → ℕ → ℕ) : ℕ :=
  op1 7 (op2 2 (op3 5 8))

def is_valid_operation (op : ℕ → ℕ → ℕ) : Prop :=
  (op = (·+·)) ∨ (op = (·-·)) ∨ (op = (·*·))

theorem seventeen_is_possible_result :
  ∃ (op1 op2 op3 : ℕ → ℕ → ℕ),
    is_valid_operation op1 ∧
    is_valid_operation op2 ∧
    is_valid_operation op3 ∧
    op1 ≠ op2 ∧ op2 ≠ op3 ∧ op1 ≠ op3 ∧
    expression op1 op2 op3 = 17 :=
by
  sorry

#check seventeen_is_possible_result

end seventeen_is_possible_result_l1061_106132


namespace absolute_value_simplification_l1061_106196

theorem absolute_value_simplification : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end absolute_value_simplification_l1061_106196


namespace stream_speed_l1061_106102

/-- Given a man's downstream and upstream speeds, calculate the speed of the stream. -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 15)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3.5 := by
  sorry

end stream_speed_l1061_106102


namespace arithmetic_sequence_formula_l1061_106145

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 1 = 1 →                                         -- first term
  a 2 = 5 →                                         -- second term
  a 3 = 9 →                                         -- third term
  ∀ n, a n = 4 * n - 3 :=                           -- general formula
by sorry

end arithmetic_sequence_formula_l1061_106145


namespace traffic_light_combinations_l1061_106103

/-- The number of different signals that can be transmitted by k traffic lights -/
def total_signals (k : ℕ) : ℕ := 3^k

/-- Theorem: Given k traffic lights, each capable of transmitting 3 different signals,
    the total number of unique signal combinations is 3^k -/
theorem traffic_light_combinations (k : ℕ) :
  total_signals k = 3^k := by
  sorry

end traffic_light_combinations_l1061_106103


namespace parallel_not_coincident_condition_l1061_106176

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- Two lines are coincident if they are parallel and have the same y-intercept -/
def coincident (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop := 
  parallel a₁ b₁ a₂ b₂ ∧ (c₁ * b₂ = c₂ * b₁)

/-- The necessary and sufficient condition for the given lines to be parallel and not coincident -/
theorem parallel_not_coincident_condition : 
  ∀ a : ℝ, (parallel a 2 3 (a-1) ∧ 
            ¬coincident a 2 (-3*a) 3 (a-1) (7-a)) ↔ 
           (a = 3) := by sorry

end parallel_not_coincident_condition_l1061_106176


namespace roots_of_unity_quadratic_count_l1061_106172

/-- A complex number z is a root of unity if there exists a positive integer n such that z^n = 1 -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ z^n = 1

/-- The quadratic equation z^2 + az - 1 = 0 for some integer a -/
def quadratic_equation (z : ℂ) : Prop :=
  ∃ (a : ℤ), z^2 + a*z - 1 = 0

/-- The number of roots of unity that are also roots of the quadratic equation is exactly two -/
theorem roots_of_unity_quadratic_count :
  ∃! (S : Finset ℂ), (∀ z ∈ S, is_root_of_unity z ∧ quadratic_equation z) ∧ S.card = 2 :=
sorry

end roots_of_unity_quadratic_count_l1061_106172


namespace angle_A_is_pi_over_three_b_plus_c_range_l1061_106197

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  t.c * (t.a * Real.cos t.B - t.b / 2) = t.a^2 - t.b^2

-- Theorem for part I
theorem angle_A_is_pi_over_three (t : Triangle) 
  (h : satisfies_condition t) : t.A = π / 3 := by
  sorry

-- Theorem for part II
theorem b_plus_c_range (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.a = Real.sqrt 3) : 
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end angle_A_is_pi_over_three_b_plus_c_range_l1061_106197


namespace point_inside_circle_implies_a_range_l1061_106134

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 = 4

-- Define what it means for a point to be inside the circle
def point_inside_circle (x y a : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 < 4

-- Theorem statement
theorem point_inside_circle_implies_a_range :
  ∀ a : ℝ, point_inside_circle (-1) (-1) a → -1 < a ∧ a < 1 :=
by
  sorry


end point_inside_circle_implies_a_range_l1061_106134


namespace two_numbers_difference_l1061_106159

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 50)
  (triple_minus_quadruple : 3 * y - 4 * x = 10)
  (y_geq_x : y ≥ x) :
  |y - x| = 10 := by
  sorry

end two_numbers_difference_l1061_106159


namespace average_students_count_l1061_106148

theorem average_students_count (total : ℕ) (honor average poor : ℕ)
  (first_yes second_yes third_yes : ℕ) :
  total = 30 →
  total = honor + average + poor →
  first_yes = 19 →
  second_yes = 12 →
  third_yes = 9 →
  first_yes = honor + average / 2 →
  second_yes = average →
  third_yes = poor + average / 2 →
  average = 12 := by
  sorry

end average_students_count_l1061_106148


namespace octagon_area_l1061_106198

/-- The area of a regular octagon inscribed in a circle with area 400π -/
theorem octagon_area (circle_area : ℝ) (h : circle_area = 400 * Real.pi) :
  let r := (circle_area / Real.pi).sqrt
  let triangle_area := (1 / 2) * r^2 * Real.sin (Real.pi / 4)
  8 * triangle_area = 800 * Real.sqrt 2 := by
  sorry

end octagon_area_l1061_106198


namespace error_percentage_calculation_l1061_106101

theorem error_percentage_calculation (x : ℝ) (h : x > 0) :
  let correct_result := x + 5
  let erroneous_result := x - 5
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = (10 / (x + 5)) * 100 := by sorry

end error_percentage_calculation_l1061_106101


namespace num_polygons_twelve_points_l1061_106135

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The minimum number of sides for the polygons we're considering -/
def min_sides : ℕ := 4

/-- The number of distinct convex polygons with 4 or more sides 
    that can be drawn using some or all of n points marked on a circle -/
def num_polygons (n : ℕ) (min_sides : ℕ) : ℕ :=
  2^n - (n.choose 0 + n.choose 1 + n.choose 2 + n.choose 3)

theorem num_polygons_twelve_points : 
  num_polygons n min_sides = 3797 := by
  sorry

end num_polygons_twelve_points_l1061_106135


namespace ab_value_l1061_106165

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5/8) : a * b = Real.sqrt 3 / 4 := by
  sorry

end ab_value_l1061_106165


namespace sufficient_not_necessary_l1061_106128

theorem sufficient_not_necessary : 
  (∀ X Y : ℝ, X > 2 ∧ Y > 3 → X + Y > 5 ∧ X * Y > 6) ∧ 
  (∃ X Y : ℝ, X + Y > 5 ∧ X * Y > 6 ∧ ¬(X > 2 ∧ Y > 3)) :=
by sorry

end sufficient_not_necessary_l1061_106128


namespace num_divisors_36_eq_9_l1061_106199

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ :=
  (Finset.filter (· ∣ 36) (Finset.range 37)).card

/-- Theorem stating that the number of positive divisors of 36 is 9 -/
theorem num_divisors_36_eq_9 : num_divisors_36 = 9 := by
  sorry

end num_divisors_36_eq_9_l1061_106199


namespace fixed_point_exponential_function_l1061_106129

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by sorry

end fixed_point_exponential_function_l1061_106129


namespace reinforcement_size_problem_l1061_106112

/-- Given a garrison with initial men, initial provision days, days before reinforcement,
    and remaining days after reinforcement, calculate the size of the reinforcement. -/
def reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) 
                       (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_men * initial_days
  let remaining_provisions := initial_men * (initial_days - days_before_reinforcement)
  let total_men_after := remaining_provisions / remaining_days
  total_men_after - initial_men

/-- The size of the reinforcement for the given problem is 1300. -/
theorem reinforcement_size_problem : 
  reinforcement_size 2000 54 21 20 = 1300 := by
  sorry

end reinforcement_size_problem_l1061_106112


namespace last_digit_of_35_power_last_digit_of_35_to_large_power_l1061_106127

theorem last_digit_of_35_power (n : ℕ) : 35^n ≡ 5 [MOD 10] := by sorry

theorem last_digit_of_35_to_large_power :
  35^(18 * (13^33)) ≡ 5 [MOD 10] := by sorry

end last_digit_of_35_power_last_digit_of_35_to_large_power_l1061_106127


namespace lenkas_numbers_l1061_106105

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def both_digits_even (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 2 = 0 ∧ (n / 10) % 2 = 0

def both_digits_odd (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 2 = 1 ∧ (n / 10) % 2 = 1

def sum_has_even_odd_digits (n : ℕ) : Prop :=
  is_two_digit n ∧ (n / 10) % 2 = 0 ∧ n % 2 = 1

theorem lenkas_numbers :
  ∀ a b : ℕ,
    both_digits_even a →
    both_digits_odd b →
    sum_has_even_odd_digits (a + b) →
    a % 3 = 0 →
    b % 3 = 0 →
    (a % 10 = 9 ∨ b % 10 = 9 ∨ (a + b) % 10 = 9) →
    ((a = 24 ∧ b = 39) ∨ (a = 42 ∧ b = 39) ∨ (a = 48 ∧ b = 39)) :=
by sorry

end lenkas_numbers_l1061_106105


namespace triangle_base_difference_l1061_106143

theorem triangle_base_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let b_A := (0.99 * b * h) / (0.9 * h)
  let h_A := 0.9 * h
  b_A = 1.1 * b := by sorry

end triangle_base_difference_l1061_106143


namespace arithmetic_sequence_fourth_term_l1061_106189

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_odd_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 10)
  (h_even_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 20) :
  a 4 = 0 := by
sorry

end arithmetic_sequence_fourth_term_l1061_106189


namespace multiplication_puzzle_l1061_106125

theorem multiplication_puzzle (a b : ℕ) : 
  a < 10 → b < 10 → (20 + a) * (10 * b + 3) = 989 → a + b = 7 := by
  sorry

end multiplication_puzzle_l1061_106125


namespace complex_division_l1061_106192

/-- Given complex numbers z₁ and z₂ corresponding to points (2, -1) and (0, -1) in the complex plane,
    prove that z₁ / z₂ = 1 + 2i -/
theorem complex_division (z₁ z₂ : ℂ) (h₁ : z₁ = 2 - I) (h₂ : z₂ = -I) : z₁ / z₂ = 1 + 2*I := by
  sorry

end complex_division_l1061_106192


namespace marbles_lost_l1061_106156

theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 16 → current = 9 → lost = initial - current → lost = 7 := by
  sorry

end marbles_lost_l1061_106156


namespace journey_time_is_41_hours_l1061_106141

-- Define the flight and layover times
def flight_NO_ATL : ℝ := 2
def layover_ATL : ℝ := 4
def flight_ATL_CHI : ℝ := 5
def layover_CHI : ℝ := 3
def flight_CHI_NY : ℝ := 3
def layover_NY : ℝ := 16
def flight_NY_SF : ℝ := 24

-- Define the total time from New Orleans to New York
def time_NO_NY : ℝ := flight_NO_ATL + layover_ATL + flight_ATL_CHI + layover_CHI + flight_CHI_NY

-- Define the total journey time
def total_journey_time : ℝ := time_NO_NY + layover_NY + flight_NY_SF

-- Theorem to prove
theorem journey_time_is_41_hours : total_journey_time = 41 := by
  sorry

end journey_time_is_41_hours_l1061_106141


namespace total_holiday_savings_l1061_106115

def holiday_savings (sam_savings victory_savings : ℕ) : ℕ :=
  sam_savings + victory_savings

theorem total_holiday_savings : 
  ∀ (sam_savings victory_savings : ℕ),
    sam_savings = 1000 →
    victory_savings = sam_savings - 100 →
    holiday_savings sam_savings victory_savings = 1900 :=
by
  sorry

end total_holiday_savings_l1061_106115


namespace minimum_requirement_proof_l1061_106126

/-- The minimum pound requirement for purchasing peanuts -/
def minimum_requirement : ℕ := 15

/-- The cost of peanuts per pound in dollars -/
def cost_per_pound : ℕ := 3

/-- The amount spent by the customer in dollars -/
def amount_spent : ℕ := 105

/-- The number of pounds purchased over the minimum requirement -/
def extra_pounds : ℕ := 20

/-- Theorem stating that the minimum requirement is correct given the conditions -/
theorem minimum_requirement_proof :
  cost_per_pound * (minimum_requirement + extra_pounds) = amount_spent :=
by sorry

end minimum_requirement_proof_l1061_106126


namespace parabola_vertex_on_x_axis_l1061_106173

/-- A parabola with equation y = -x^2 + 2x + m has its vertex on the x-axis if and only if m = -1 -/
theorem parabola_vertex_on_x_axis (m : ℝ) : 
  (∃ x, -x^2 + 2*x + m = 0 ∧ ∀ y, y = -x^2 + 2*x + m → y ≤ 0) ↔ m = -1 := by
sorry

end parabola_vertex_on_x_axis_l1061_106173


namespace x_completion_time_l1061_106104

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℝ
  is_positive : days > 0

/-- Represents a worker who can complete a work in a given time -/
structure Worker where
  time_to_complete : WorkTime

/-- The work scenario -/
structure WorkScenario where
  x : Worker
  y : Worker
  x_partial_work : WorkTime
  y_completion_after_x : WorkTime
  y_solo_completion : WorkTime
  work_continuity : x_partial_work.days + y_completion_after_x.days = y_solo_completion.days

/-- The theorem stating that x takes 40 days to complete the work -/
theorem x_completion_time (scenario : WorkScenario) 
  (h1 : scenario.x_partial_work.days = 8)
  (h2 : scenario.y_completion_after_x.days = 16)
  (h3 : scenario.y_solo_completion.days = 20) :
  scenario.x.time_to_complete.days = 40 := by
  sorry


end x_completion_time_l1061_106104


namespace triangle_problem_l1061_106150

theorem triangle_problem (a b c A B C : Real) (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3) (h3 : (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = π/3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := by sorry

end triangle_problem_l1061_106150


namespace special_rectangle_area_length_width_ratio_width_is_diameter_l1061_106136

/-- A rectangle with an inscribed circle of radius 10 and a circumscribed circle -/
structure Rectangle where
  width : ℝ
  length : ℝ
  inscribed_circle_radius : ℝ
  has_circumscribed_circle : Prop

/-- The properties of our specific rectangle -/
def special_rectangle : Rectangle where
  width := 20
  length := 60
  inscribed_circle_radius := 10
  has_circumscribed_circle := true

/-- The theorem stating that the area of the special rectangle is 1200 -/
theorem special_rectangle_area :
  special_rectangle.length * special_rectangle.width = 1200 := by
  sorry

/-- The ratio of length to width is 3:1 -/
theorem length_width_ratio :
  special_rectangle.length = 3 * special_rectangle.width := by
  sorry

/-- The width is twice the radius of the inscribed circle -/
theorem width_is_diameter :
  special_rectangle.width = 2 * special_rectangle.inscribed_circle_radius := by
  sorry

end special_rectangle_area_length_width_ratio_width_is_diameter_l1061_106136


namespace pentagon_reflection_rotation_l1061_106170

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = dist (vertices j) (vertices ((j + 1) % 5))
  is_pentagon : ∀ i : Fin 5, vertices i ≠ vertices ((i + 1) % 5)

/-- Reflection of a point over a line through the center of the pentagon -/
def reflect (p : RegularPentagon) (line : ℝ × ℝ → Prop) : RegularPentagon :=
  sorry

/-- Rotation of a pentagon by an angle about its center -/
def rotate (p : RegularPentagon) (angle : ℝ) : RegularPentagon :=
  sorry

/-- The center of a regular pentagon -/
def center (p : RegularPentagon) : ℝ × ℝ :=
  sorry

theorem pentagon_reflection_rotation (p : RegularPentagon) (line : ℝ × ℝ → Prop) :
  rotate (reflect p line) (144 * π / 180) = rotate p (144 * π / 180) :=
sorry

end pentagon_reflection_rotation_l1061_106170


namespace symmetry_implies_phi_value_l1061_106191

/-- Given a function f(x) = sin(x) + √3 * cos(x), prove that if y = f(x + φ) is symmetric about x = 0, then φ = π/6 -/
theorem symmetry_implies_phi_value (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin x + Real.sqrt 3 * Real.cos x) →
  (∀ x, f (x + φ) = f (-x + φ)) →
  φ = Real.pi / 6 := by
  sorry

end symmetry_implies_phi_value_l1061_106191


namespace cube_sum_of_symmetric_relations_l1061_106147

theorem cube_sum_of_symmetric_relations (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a * b + a * c + b * c = 2)
  (h3 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 1 := by
  sorry

end cube_sum_of_symmetric_relations_l1061_106147


namespace simplify_expression_l1061_106107

theorem simplify_expression (x y : ℝ) : -x + y - 2*x - 3*y = -3*x - 2*y := by
  sorry

end simplify_expression_l1061_106107


namespace tuesday_poodles_count_l1061_106106

/-- Represents the number of hours Charlotte can walk dogs on a weekday -/
def weekday_hours : ℕ := 8

/-- Represents the number of hours Charlotte can walk dogs on a weekend day -/
def weekend_hours : ℕ := 4

/-- Represents the number of weekdays in a week -/
def weekdays : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- Represents the time it takes to walk a poodle -/
def poodle_time : ℕ := 2

/-- Represents the time it takes to walk a Chihuahua -/
def chihuahua_time : ℕ := 1

/-- Represents the time it takes to walk a Labrador -/
def labrador_time : ℕ := 3

/-- Represents the time it takes to walk a Golden Retriever -/
def golden_retriever_time : ℕ := 4

/-- Represents the number of poodles walked on Monday -/
def monday_poodles : ℕ := 4

/-- Represents the number of Chihuahuas walked on Monday and Tuesday -/
def monday_tuesday_chihuahuas : ℕ := 2

/-- Represents the number of Golden Retrievers walked on Monday -/
def monday_golden_retrievers : ℕ := 1

/-- Represents the number of Labradors walked on Wednesday -/
def wednesday_labradors : ℕ := 4

/-- Represents the number of Golden Retrievers walked on Tuesday -/
def tuesday_golden_retrievers : ℕ := 1

theorem tuesday_poodles_count :
  ∃ (tuesday_poodles : ℕ),
    tuesday_poodles = 1 ∧
    weekday_hours * weekdays + weekend_hours * weekend_days ≥
      (monday_poodles * poodle_time +
       monday_tuesday_chihuahuas * chihuahua_time +
       monday_golden_retrievers * golden_retriever_time) +
      (tuesday_poodles * poodle_time +
       monday_tuesday_chihuahuas * chihuahua_time +
       tuesday_golden_retrievers * golden_retriever_time) +
      (wednesday_labradors * labrador_time) :=
by sorry

end tuesday_poodles_count_l1061_106106


namespace parallelogram_sticks_l1061_106113

/-- A parallelogram formed by four sticks -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  is_parallelogram : side1 = side3 ∧ side2 = side4

/-- The theorem stating that if four sticks of lengths 5, 5, 7, and a can form a parallelogram, then a = 7 -/
theorem parallelogram_sticks (a : ℝ) :
  (∃ p : Parallelogram, p.side1 = 5 ∧ p.side2 = 5 ∧ p.side3 = 7 ∧ p.side4 = a) →
  a = 7 := by
  sorry

end parallelogram_sticks_l1061_106113


namespace ellipse_hyperbola_same_foci_l1061_106190

/-- The value of m for which an ellipse and hyperbola with given equations have the same foci -/
theorem ellipse_hyperbola_same_foci (m : ℝ) : m > 0 →
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1) →
  (∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1) →
  (∀ c : ℝ, c^2 = 4 - m^2 ↔ c^2 = m^2 + 2) →
  m = 1 := by
  sorry

end ellipse_hyperbola_same_foci_l1061_106190


namespace concrete_cost_theorem_l1061_106155

/-- Calculates the cost of concrete for home foundations -/
theorem concrete_cost_theorem 
  (num_homes : ℕ) 
  (length width height : ℝ) 
  (density : ℝ) 
  (cost_per_pound : ℝ) : 
  num_homes * length * width * height * density * cost_per_pound = 45000 :=
by
  sorry

#check concrete_cost_theorem 3 100 100 0.5 150 0.02

end concrete_cost_theorem_l1061_106155


namespace facebook_group_members_l1061_106175

/-- Proves that the original number of members in a Facebook group was 150 -/
theorem facebook_group_members : 
  ∀ (original_members removed_members remaining_messages_per_week messages_per_member_per_day : ℕ),
  removed_members = 20 →
  messages_per_member_per_day = 50 →
  remaining_messages_per_week = 45500 →
  original_members = 
    (remaining_messages_per_week / (messages_per_member_per_day * 7)) + removed_members →
  original_members = 150 := by
sorry

end facebook_group_members_l1061_106175


namespace complex_and_imaginary_solution_l1061_106162

-- Define z as a complex number
variable (z : ℂ)

-- Define the conditions
def condition1 : Prop := (z + Complex.I).im = 0
def condition2 : Prop := (z / (1 - Complex.I)).im = 0

-- Define m as a purely imaginary number
def m : ℂ → ℂ := fun c => Complex.I * c

-- Define the equation with real roots
def has_real_roots (z m : ℂ) : Prop :=
  ∃ x : ℝ, x^2 + x * (1 + z) - (3 * m - 1) * Complex.I = 0

-- State the theorem
theorem complex_and_imaginary_solution :
  condition1 z → condition2 z → has_real_roots z (m 1) →
  z = 1 - Complex.I ∧ m 1 = -Complex.I :=
sorry

end complex_and_imaginary_solution_l1061_106162


namespace tree_shadow_length_l1061_106166

/-- Given a person and a tree casting shadows, this theorem calculates the length of the tree's shadow. -/
theorem tree_shadow_length 
  (person_height : ℝ) 
  (person_shadow : ℝ) 
  (tree_height : ℝ) 
  (h1 : person_height = 1.5)
  (h2 : person_shadow = 0.5)
  (h3 : tree_height = 30) :
  ∃ (tree_shadow : ℝ), tree_shadow = 10 ∧ 
    person_height / person_shadow = tree_height / tree_shadow :=
by sorry

end tree_shadow_length_l1061_106166


namespace program_output_l1061_106139

theorem program_output (A : ℕ) (h : A = 1) : (((A * 2) * 3) * 4) * 5 = 120 := by
  sorry

end program_output_l1061_106139


namespace hotel_loss_calculation_l1061_106186

/-- Calculates the loss incurred by a hotel given its operations expenses and the fraction of expenses covered by client payments. -/
def hotel_loss (expenses : ℝ) (payment_fraction : ℝ) : ℝ :=
  expenses - (payment_fraction * expenses)

/-- Theorem stating that a hotel with $100 in expenses and client payments covering 3/4 of expenses incurs a $25 loss. -/
theorem hotel_loss_calculation :
  hotel_loss 100 (3/4) = 25 := by
  sorry

end hotel_loss_calculation_l1061_106186
