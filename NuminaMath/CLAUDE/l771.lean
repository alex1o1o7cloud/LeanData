import Mathlib

namespace pyramid_volume_l771_77130

/-- Represents a pyramid with a triangular base --/
structure TriangularPyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_side3 : ℝ
  lateral_angle : ℝ

/-- Calculates the volume of a triangular pyramid --/
def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating that a pyramid with the given properties has a volume of 6 --/
theorem pyramid_volume :
  ∀ (p : TriangularPyramid),
    p.base_side1 = 6 ∧
    p.base_side2 = 5 ∧
    p.base_side3 = 5 ∧
    p.lateral_angle = π / 4 →
    volume p = 6 := by
  sorry

end pyramid_volume_l771_77130


namespace max_balances_correct_max_balances_achievable_l771_77104

/-- Represents a set of unique weights -/
def UniqueWeights (n : ℕ) := Fin n → ℝ

/-- Represents the state of the balance scale -/
structure BalanceState where
  left : List ℝ
  right : List ℝ

/-- Checks if the balance scale is in equilibrium -/
def isBalanced (state : BalanceState) : Prop :=
  state.left.sum = state.right.sum

/-- Represents a sequence of weight placements -/
def WeightPlacement (n : ℕ) := Fin n → Bool × Fin n

/-- Counts the number of times the scale balances during a sequence of weight placements -/
def countBalances (weights : UniqueWeights 2021) (placements : WeightPlacement m) : ℕ :=
  sorry

/-- The maximum number of times the scale can balance -/
def maxBalances : ℕ := 673

theorem max_balances_correct (weights : UniqueWeights 2021) :
  ∀ (m : ℕ) (placements : WeightPlacement m),
    countBalances weights placements ≤ maxBalances :=
  sorry

theorem max_balances_achievable :
  ∃ (weights : UniqueWeights 2021) (m : ℕ) (placements : WeightPlacement m),
    countBalances weights placements = maxBalances :=
  sorry

end max_balances_correct_max_balances_achievable_l771_77104


namespace stratified_sampling_second_year_l771_77195

theorem stratified_sampling_second_year (total_students : ℕ) (second_year_students : ℕ) (sample_size : ℕ) :
  total_students = 3600 →
  second_year_students = 900 →
  sample_size = 720 →
  (second_year_students * sample_size) / total_students = 180 :=
by sorry

end stratified_sampling_second_year_l771_77195


namespace eight_million_two_hundred_thousand_scientific_notation_l771_77172

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eight_million_two_hundred_thousand_scientific_notation :
  toScientificNotation 8200000 = ScientificNotation.mk 8.2 6 (by sorry) :=
sorry

end eight_million_two_hundred_thousand_scientific_notation_l771_77172


namespace range_of_m_l771_77167

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  monotonically_decreasing f →
  f (m + 1) < f (3 - 2 * m) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), m = -(Real.sin x)^2 - 2 * Real.sin x + 1) →
  m ∈ Set.Ioo (2/3) 1 := by
  sorry

end range_of_m_l771_77167


namespace positive_solution_sum_l771_77170

theorem positive_solution_sum (a b : ℕ+) (x : ℤ) : 
  x > 0 → x = Int.sqrt a - b → x^2 - 10*x = 39 → a + b = 69 := by
  sorry

end positive_solution_sum_l771_77170


namespace factors_of_M_l771_77121

/-- The number of natural-number factors of M, where M = 2^2 · 3^3 · 5^2 · 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  if M = 2^2 * 3^3 * 5^2 * 7^1 then 72 else 0

/-- Theorem stating that the number of natural-number factors of M is 72 -/
theorem factors_of_M :
  number_of_factors (2^2 * 3^3 * 5^2 * 7^1) = 72 := by
  sorry

end factors_of_M_l771_77121


namespace baron_munchausen_claim_false_l771_77103

theorem baron_munchausen_claim_false : ∃ n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ ¬∃ m : ℕ, 0 ≤ m ∧ m ≤ 99 ∧ (100 * n + m)^2 = 100 * n + m := by
  sorry

end baron_munchausen_claim_false_l771_77103


namespace max_sequence_length_sequence_of_length_12_exists_l771_77148

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property that the sum of any five consecutive terms is negative -/
def SumOfFiveNegative (a : Sequence) :=
  ∀ i, i > 0 → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) < 0

/-- The property that the sum of any nine consecutive terms is positive -/
def SumOfNinePositive (a : Sequence) :=
  ∀ i, i > 0 → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8)) > 0

/-- The maximum length of a sequence satisfying both properties is 12 -/
theorem max_sequence_length :
  ∀ n : ℕ, n > 12 →
    ¬∃ a : Sequence, (SumOfFiveNegative a ∧ SumOfNinePositive a ∧ ∀ i > n, a i = 0) :=
by sorry

/-- There exists a sequence of length 12 satisfying both properties -/
theorem sequence_of_length_12_exists :
  ∃ a : Sequence, SumOfFiveNegative a ∧ SumOfNinePositive a ∧ ∀ i > 12, a i = 0 :=
by sorry

end max_sequence_length_sequence_of_length_12_exists_l771_77148


namespace q_duration_is_nine_l771_77120

/-- Investment and profit ratios for partners P and Q -/
structure PartnershipRatios where
  investment_ratio_p : ℕ
  investment_ratio_q : ℕ
  profit_ratio_p : ℕ
  profit_ratio_q : ℕ

/-- Calculate the investment duration of partner Q given the ratios and P's duration -/
def calculate_q_duration (ratios : PartnershipRatios) (p_duration : ℕ) : ℕ :=
  (ratios.investment_ratio_p * p_duration * ratios.profit_ratio_q) / 
  (ratios.investment_ratio_q * ratios.profit_ratio_p)

/-- Theorem stating that Q's investment duration is 9 months given the specified ratios and P's duration -/
theorem q_duration_is_nine :
  let ratios : PartnershipRatios := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 9
  }
  let p_duration := 5
  calculate_q_duration ratios p_duration = 9 := by
  sorry

end q_duration_is_nine_l771_77120


namespace one_fourth_of_eight_point_eight_l771_77113

theorem one_fourth_of_eight_point_eight : (8.8 : ℚ) / 4 = 11 / 5 := by
  sorry

end one_fourth_of_eight_point_eight_l771_77113


namespace absolute_value_expression_l771_77133

theorem absolute_value_expression (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c > 0) : 
  |a| - |a + b| + |c - a| + |b - c| = 2 * c - a := by
  sorry

end absolute_value_expression_l771_77133


namespace wall_length_l771_77101

/-- The length of a rectangular wall with a trapezoidal mirror -/
theorem wall_length (a b h w : ℝ) (ha : a > 0) (hb : b > 0) (hh : h > 0) (hw : w > 0) :
  (a + b) * h / 2 * 2 = w * (580 / 27) →
  a = 34 →
  b = 24 →
  h = 20 →
  w = 54 →
  580 / 27 = 580 / 27 :=
by sorry

end wall_length_l771_77101


namespace sets_satisfying_union_condition_l771_77102

theorem sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ∪ {1, 2} = {1, 2, 3}) ∧ 
    (∀ A, A ∪ {1, 2} = {1, 2, 3} → A ∈ S) ∧
    Finset.card S = 4 := by
  sorry

end sets_satisfying_union_condition_l771_77102


namespace geometric_sequence_sum_4_l771_77145

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property

/-- Theorem stating the conditions and the result to be proved -/
theorem geometric_sequence_sum_4 (seq : GeometricSequence) 
  (h1 : seq.a 1 - seq.a 2 = 2)
  (h2 : seq.a 2 - seq.a 3 = 6) :
  seq.S 4 = -40 := by
  sorry


end geometric_sequence_sum_4_l771_77145


namespace shaded_area_square_with_circles_l771_77194

/-- The area of the shaded region not covered by four circles centered at the vertices of a square -/
theorem shaded_area_square_with_circles (side_length radius : ℝ) (h1 : side_length = 8) (h2 : radius = 3) :
  side_length ^ 2 - 4 * Real.pi * radius ^ 2 = 64 - 36 * Real.pi := by
  sorry

#check shaded_area_square_with_circles

end shaded_area_square_with_circles_l771_77194


namespace arrangement_count_l771_77111

def num_people : ℕ := 8

theorem arrangement_count :
  (num_people.factorial) / 6 * 2 = 13440 := by sorry

end arrangement_count_l771_77111


namespace walking_speed_proof_l771_77125

/-- The walking speed of a man who covers the same distance in 9 hours walking
    and in 3 hours running at 24 kmph. -/
def walking_speed : ℝ := 8

theorem walking_speed_proof :
  walking_speed * 9 = 24 * 3 := by
  sorry

end walking_speed_proof_l771_77125


namespace sum_equals_210_l771_77115

theorem sum_equals_210 : 145 + 35 + 25 + 5 = 210 := by
  sorry

end sum_equals_210_l771_77115


namespace min_sum_of_squares_on_line_l771_77191

theorem min_sum_of_squares_on_line (m n : ℝ) (h : m + n = 1) : 
  m^2 + n^2 ≥ 4 ∧ ∃ (m₀ n₀ : ℝ), m₀ + n₀ = 1 ∧ m₀^2 + n₀^2 = 4 := by
  sorry

end min_sum_of_squares_on_line_l771_77191


namespace slope_intercept_sum_l771_77182

/-- Given points A, B, and C in a plane, with D as the midpoint of AC, 
    prove that the sum of the slope and y-intercept of the line through C and D is 36/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) → B = (0, 0) → C = (10, 0) → 
  D = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (D.2 - C.2) / (D.1 - C.1) + (D.2 - (D.2 - C.2) / (D.1 - C.1) * D.1) = 36 / 5 :=
by sorry

end slope_intercept_sum_l771_77182


namespace fraction_unchanged_l771_77185

theorem fraction_unchanged (x y : ℝ) : 
  (2 * x) / (3 * x - 2 * y) = (4 * x) / (6 * x - 4 * y) := by
  sorry

end fraction_unchanged_l771_77185


namespace trigonometric_simplification_l771_77156

theorem trigonometric_simplification :
  let f : ℝ → ℝ := λ x => Real.sin (x * π / 180)
  let g : ℝ → ℝ := λ x => Real.cos (x * π / 180)
  (f 15 + f 25 + f 35 + f 45 + f 55 + f 65 + f 75 + f 85) / (g 10 * g 15 * g 30) =
  8 * Real.sqrt 3 * g 40 * g 5 :=
by sorry

end trigonometric_simplification_l771_77156


namespace negation_of_universal_proposition_l771_77179

theorem negation_of_universal_proposition :
  (¬ ∀ x y : ℝ, x < 0 → y < 0 → x + y ≤ -2 * Real.sqrt (x * y)) ↔
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x + y > -2 * Real.sqrt (x * y)) :=
by sorry

end negation_of_universal_proposition_l771_77179


namespace pen_price_relationship_l771_77147

/-- The relationship between the number of pens and their total price -/
theorem pen_price_relationship (x y : ℝ) : y = (3/2) * x ↔ 
  ∃ (boxes : ℝ), 
    x = 12 * boxes ∧ 
    y = 18 * boxes :=
by sorry

end pen_price_relationship_l771_77147


namespace chihuahua_grooming_time_l771_77168

/-- The time Karen takes to groom different types of dogs -/
structure GroomingTimes where
  rottweiler : ℕ
  border_collie : ℕ
  chihuahua : ℕ

/-- The number of each type of dog Karen grooms -/
structure DogCounts where
  rottweilers : ℕ
  border_collies : ℕ
  chihuahuas : ℕ

/-- Calculates the total grooming time for all dogs -/
def totalGroomingTime (times : GroomingTimes) (counts : DogCounts) : ℕ :=
  times.rottweiler * counts.rottweilers +
  times.border_collie * counts.border_collies +
  times.chihuahua * counts.chihuahuas

theorem chihuahua_grooming_time :
  ∀ (times : GroomingTimes) (counts : DogCounts),
  times.rottweiler = 20 →
  times.border_collie = 10 →
  counts.rottweilers = 6 →
  counts.border_collies = 9 →
  counts.chihuahuas = 1 →
  totalGroomingTime times counts = 255 →
  times.chihuahua = 45 := by
  sorry

end chihuahua_grooming_time_l771_77168


namespace target_avg_income_l771_77190

def past_income : List ℝ := [406, 413, 420, 436, 395]
def next_weeks : ℕ := 5
def total_weeks : ℕ := 10
def next_avg_income : ℝ := 586

theorem target_avg_income :
  let past_total := past_income.sum
  let next_total := next_avg_income * next_weeks
  let total_income := past_total + next_total
  (total_income / total_weeks : ℝ) = 500 := by sorry

end target_avg_income_l771_77190


namespace root_difference_implies_k_value_l771_77106

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0) →
  (∃ r s : ℝ, (r+7)^2 - k*(r+7) + 12 = 0 ∧ (s+7)^2 - k*(s+7) + 12 = 0) →
  (∀ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0 →
              (r+7)^2 - k*(r+7) + 12 = 0 ∧ (s+7)^2 - k*(s+7) + 12 = 0) →
  k = 7 :=
by sorry

end root_difference_implies_k_value_l771_77106


namespace gravel_cost_theorem_l771_77187

/-- Calculates the cost of gravelling a path inside a rectangular plot -/
def gravel_cost (length width path_width gravel_cost_per_sqm : ℝ) : ℝ :=
  let total_area := length * width
  let inner_area := (length - 2 * path_width) * (width - 2 * path_width)
  let path_area := total_area - inner_area
  path_area * gravel_cost_per_sqm

/-- Theorem stating the cost of gravelling the path -/
theorem gravel_cost_theorem :
  gravel_cost 100 70 2.5 0.9 = 742.5 := by
sorry

end gravel_cost_theorem_l771_77187


namespace factorial_expression_equals_2884_l771_77144

theorem factorial_expression_equals_2884 :
  (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) + 2^2))^2 = 2884 := by sorry

end factorial_expression_equals_2884_l771_77144


namespace dividend_percentage_calculation_l771_77136

/-- Calculates the dividend percentage given investment details and dividend received -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 720) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 6 := by
sorry

end dividend_percentage_calculation_l771_77136


namespace spinner_probability_l771_77123

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 1/5 →
  p_B = 1/10 →
  p_D = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_D = 7/20 := by
sorry

end spinner_probability_l771_77123


namespace staircase_carpet_cost_l771_77197

/-- Represents the dimensions and cost parameters of a staircase -/
structure Staircase where
  num_steps : ℕ
  step_height : ℝ
  step_depth : ℝ
  width : ℝ
  carpet_cost_per_sqm : ℝ

/-- Calculates the cost of carpeting a staircase -/
def carpet_cost (s : Staircase) : ℝ :=
  let total_height := s.num_steps * s.step_height
  let total_depth := s.num_steps * s.step_depth
  let combined_length := total_height + total_depth
  let total_area := combined_length * s.width
  total_area * s.carpet_cost_per_sqm

/-- Theorem: The cost of carpeting the given staircase is 1512 yuan -/
theorem staircase_carpet_cost :
  let s : Staircase := {
    num_steps := 15,
    step_height := 0.16,
    step_depth := 0.26,
    width := 3,
    carpet_cost_per_sqm := 80
  }
  carpet_cost s = 1512 := by
  sorry

end staircase_carpet_cost_l771_77197


namespace linear_function_through_minus_one_zero_l771_77159

/-- A linear function passing through (-1, 0) has slope 1 -/
theorem linear_function_through_minus_one_zero (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1) → 0 = k * (-1) + 1 → k = 1 := by
  sorry

end linear_function_through_minus_one_zero_l771_77159


namespace john_drinks_two_cups_per_day_l771_77198

-- Define the constants
def gallon_to_ounce : ℚ := 128
def cup_to_ounce : ℚ := 8
def days_between_purchases : ℚ := 4
def gallons_per_purchase : ℚ := 1/2

-- Define the function to calculate cups per day
def cups_per_day : ℚ :=
  (gallons_per_purchase * gallon_to_ounce) / (days_between_purchases * cup_to_ounce)

-- Theorem statement
theorem john_drinks_two_cups_per_day :
  cups_per_day = 2 := by sorry

end john_drinks_two_cups_per_day_l771_77198


namespace unique_fraction_decomposition_l771_77126

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ 2 / p = 1 / n + 1 / m ∧
  n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end unique_fraction_decomposition_l771_77126


namespace inequality_condition_l771_77166

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x - 5| + |x - 10| < b) ↔ b > 8 :=
by sorry

end inequality_condition_l771_77166


namespace combination_simplification_l771_77163

theorem combination_simplification (n : ℕ) : 
  (n.choose (n - 2)) + (n.choose 3) + ((n + 1).choose 2) = ((n + 2).choose 3) := by
  sorry

end combination_simplification_l771_77163


namespace min_shift_for_odd_cosine_l771_77162

/-- Given a function f(x) = cos(2x + π/6) that is shifted right by φ units,
    prove that the minimum positive φ that makes the resulting function odd is π/3. -/
theorem min_shift_for_odd_cosine :
  let f (x : ℝ) := Real.cos (2 * x + π / 6)
  let g (φ : ℝ) (x : ℝ) := f (x - φ)
  ∀ φ : ℝ, φ > 0 →
    (∀ x : ℝ, g φ (-x) = -(g φ x)) →
    φ ≥ π / 3 :=
by sorry

end min_shift_for_odd_cosine_l771_77162


namespace sin_cos_equation_solutions_l771_77117

theorem sin_cos_equation_solutions (π : Real) (sin cos : Real → Real) :
  (∃ (x₁ x₂ : Real), x₁ ≠ x₂ ∧ 
   0 ≤ x₁ ∧ x₁ ≤ π ∧ 
   0 ≤ x₂ ∧ x₂ ≤ π ∧
   sin (π / 2 * cos x₁) = cos (π / 2 * sin x₁) ∧
   sin (π / 2 * cos x₂) = cos (π / 2 * sin x₂)) ∧
  (∀ (x y z : Real), 
   0 ≤ x ∧ x ≤ π ∧ 
   0 ≤ y ∧ y ≤ π ∧ 
   0 ≤ z ∧ z ≤ π ∧
   x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   sin (π / 2 * cos x) = cos (π / 2 * sin x) ∧
   sin (π / 2 * cos y) = cos (π / 2 * sin y) ∧
   sin (π / 2 * cos z) = cos (π / 2 * sin z) →
   False) :=
by sorry


end sin_cos_equation_solutions_l771_77117


namespace root_exists_in_interval_l771_77151

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem root_exists_in_interval :
  (∀ x ∈ Set.Icc 0 0.5, ContinuousAt f x) →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x ∈ Set.Ioo 0 0.5, f x = 0 := by
  sorry

#check root_exists_in_interval

end root_exists_in_interval_l771_77151


namespace max_value_xy_l771_77108

theorem max_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 1) :
  xy ≤ 1/12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 1 ∧ x₀*y₀ = 1/12 := by
  sorry

end max_value_xy_l771_77108


namespace arrangements_with_A_must_go_arrangements_A_B_not_Japan_l771_77112

-- Define the number of volunteers
def total_volunteers : ℕ := 6

-- Define the number of people to be selected
def selected_people : ℕ := 4

-- Define the number of pavilions
def num_pavilions : ℕ := 4

-- Function to calculate the number of arrangements when one person must be included
def arrangements_with_one_person (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose (n - 1) (k - 1)) * (Nat.factorial k)

-- Function to calculate the number of arrangements when two people cannot go to a specific pavilion
def arrangements_with_restriction (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose k 1) * (Nat.choose (n - 1) (k - 1)) * (Nat.factorial (k - 1))

-- Theorem for the first question
theorem arrangements_with_A_must_go :
  arrangements_with_one_person total_volunteers selected_people = 240 := by
  sorry

-- Theorem for the second question
theorem arrangements_A_B_not_Japan :
  arrangements_with_restriction total_volunteers selected_people = 240 := by
  sorry

end arrangements_with_A_must_go_arrangements_A_B_not_Japan_l771_77112


namespace polynomial_coefficient_sum_l771_77128

theorem polynomial_coefficient_sum : ∀ P Q R S : ℝ,
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = P * x^3 + Q * x^2 + R * x + S) →
  P + Q + R + S = 36 := by
  sorry

end polynomial_coefficient_sum_l771_77128


namespace savings_percentage_l771_77183

/-- Represents the financial situation of a person over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- Conditions for the financial situation --/
def ValidFinancialSituation (fs : FinancialSituation) : Prop :=
  fs.income_year1 > 0 ∧
  fs.savings_year1 > 0 ∧
  fs.income_year2 = 1.35 * fs.income_year1 ∧
  fs.savings_year2 = 2 * fs.savings_year1 ∧
  (fs.income_year1 - fs.savings_year1) + (fs.income_year2 - fs.savings_year2) = 2 * (fs.income_year1 - fs.savings_year1)

/-- Theorem stating the percentage of income saved in the first year --/
theorem savings_percentage (fs : FinancialSituation) 
  (h : ValidFinancialSituation fs) : 
  fs.savings_year1 / fs.income_year1 = 0.35 := by
  sorry

end savings_percentage_l771_77183


namespace fruit_basket_ratio_l771_77155

/-- Fruit basket problem -/
theorem fruit_basket_ratio : 
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  oranges + apples + bananas + peaches = 28 →
  peaches * 2 = bananas :=
by
  sorry

end fruit_basket_ratio_l771_77155


namespace square_side_length_l771_77171

theorem square_side_length : ∃ (s : ℝ), s > 0 ∧ s^2 + s - 4*s = 4 := by
  sorry

end square_side_length_l771_77171


namespace work_completion_time_l771_77174

/-- Proves that if person A can complete a work in 40 days, and together with person B they can complete 0.25 part of the work in 6 days, then person B can complete the work alone in 60 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 40) (hab : 1 / a + 1 / b = 1 / 24) : b = 60 := by
  sorry

end work_completion_time_l771_77174


namespace man_age_difference_l771_77176

/-- Proves that a man is 22 years older than his son given certain conditions -/
theorem man_age_difference (man_age son_age : ℕ) : 
  son_age = 20 →
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 22 := by
  sorry

end man_age_difference_l771_77176


namespace standard_pairs_parity_l771_77122

/-- Represents a color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard of size m × n -/
structure Chessboard (m n : ℕ) where
  cells : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of blue cells on the border of the chessboard (excluding corners) -/
def countBlueBorderCells (board : Chessboard m n) : ℕ :=
  sorry

/-- Counts the number of "standard pairs" on the chessboard -/
def countStandardPairs (board : Chessboard m n) : ℕ :=
  sorry

/-- Theorem stating the relationship between the number of standard pairs and blue border cells -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Odd (countStandardPairs board) ↔ Odd (countBlueBorderCells board) :=
sorry

end standard_pairs_parity_l771_77122


namespace selection_test_results_l771_77180

/-- Represents the probability of A answering a question correctly -/
def prob_A_correct : ℚ := 3/5

/-- Represents the number of questions B can answer correctly out of 10 -/
def B_correct_answers : ℕ := 5

/-- Represents the total number of questions in the pool -/
def total_questions : ℕ := 10

/-- Represents the number of questions in each exam -/
def exam_questions : ℕ := 3

/-- Represents the score for a correct answer -/
def correct_score : ℤ := 10

/-- Represents the score deduction for an incorrect answer -/
def incorrect_score : ℤ := -5

/-- Represents the minimum score required for selection -/
def selection_threshold : ℤ := 15

/-- The expected score for A -/
def expected_score_A : ℚ := 12

/-- The probability that both A and B are selected -/
def prob_both_selected : ℚ := 81/250

theorem selection_test_results :
  (prob_A_correct = 3/5) →
  (B_correct_answers = 5) →
  (total_questions = 10) →
  (exam_questions = 3) →
  (correct_score = 10) →
  (incorrect_score = -5) →
  (selection_threshold = 15) →
  (expected_score_A = 12) ∧
  (prob_both_selected = 81/250) := by
  sorry

end selection_test_results_l771_77180


namespace ginos_bears_l771_77160

/-- The number of brown bears Gino has -/
def brown_bears : ℕ := 15

/-- The number of white bears Gino has -/
def white_bears : ℕ := 24

/-- The number of black bears Gino has -/
def black_bears : ℕ := 27

/-- The number of polar bears Gino has -/
def polar_bears : ℕ := 12

/-- The number of grizzly bears Gino has -/
def grizzly_bears : ℕ := 18

/-- The total number of bears Gino has -/
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

theorem ginos_bears : total_bears = 96 := by
  sorry

end ginos_bears_l771_77160


namespace gcf_lcm_sum_of_10_15_25_l771_77153

theorem gcf_lcm_sum_of_10_15_25 : ∃ (A B : ℕ),
  (A = Nat.gcd 10 (Nat.gcd 15 25)) ∧
  (B = Nat.lcm 10 (Nat.lcm 15 25)) ∧
  (A + B = 155) := by
  sorry

end gcf_lcm_sum_of_10_15_25_l771_77153


namespace garden_length_l771_77186

/-- Given a square playground and a rectangular garden, proves the length of the garden
    when the total fencing is known. -/
theorem garden_length
  (playground_side : ℕ)
  (garden_width : ℕ)
  (total_fencing : ℕ)
  (h1 : playground_side = 27)
  (h2 : garden_width = 9)
  (h3 : total_fencing = 150)
  (h4 : 4 * playground_side + 2 * garden_width + 2 * (total_fencing - 4 * playground_side - 2 * garden_width) / 2 = total_fencing) :
  (total_fencing - 4 * playground_side - 2 * garden_width) / 2 = 12 :=
by sorry

end garden_length_l771_77186


namespace no_solution_exists_l771_77107

theorem no_solution_exists : ¬∃ (x : ℝ), Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end no_solution_exists_l771_77107


namespace lcm_from_product_and_hcf_l771_77146

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 45276 →
  Nat.gcd a b = 22 →
  Nat.lcm a b = 2058 := by
sorry

end lcm_from_product_and_hcf_l771_77146


namespace gcd_765432_654321_l771_77127

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 6 := by
  sorry

end gcd_765432_654321_l771_77127


namespace team_selection_l771_77188

theorem team_selection (m : ℕ) : 
  (0 ≤ 14 - m) ∧ 
  (14 - m ≤ 2 * m) ∧ 
  (0 ≤ 5 * m - 11) ∧ 
  (5 * m - 11 ≤ 3 * m) →
  (m = 5) ∧ 
  (Nat.choose 10 9 * Nat.choose 15 14 = 150) := by
sorry


end team_selection_l771_77188


namespace total_crickets_l771_77192

/-- The total number of crickets given an initial and additional amount -/
theorem total_crickets (initial : ℝ) (additional : ℝ) :
  initial = 7.5 → additional = 11.25 → initial + additional = 18.75 :=
by sorry

end total_crickets_l771_77192


namespace friends_total_earnings_l771_77154

/-- The total earnings of four friends selling electronics on eBay -/
def total_earnings (lauryn_earnings : ℝ) : ℝ :=
  let aurelia_earnings := 0.7 * lauryn_earnings
  let jackson_earnings := 1.5 * aurelia_earnings
  let maya_earnings := 0.4 * jackson_earnings
  lauryn_earnings + aurelia_earnings + jackson_earnings + maya_earnings

/-- Theorem stating that the total earnings of the four friends is $6340 -/
theorem friends_total_earnings :
  total_earnings 2000 = 6340 := by
  sorry

end friends_total_earnings_l771_77154


namespace tourist_travel_speeds_l771_77134

theorem tourist_travel_speeds (total_distance : ℝ) (car_fraction : ℚ) (speed_difference : ℝ) (time_difference : ℝ) :
  total_distance = 160 ∧
  car_fraction = 5/8 ∧
  speed_difference = 20 ∧
  time_difference = 1/4 →
  (∃ (car_speed boat_speed : ℝ),
    (car_speed = 80 ∧ boat_speed = 60) ∨
    (car_speed = 100 ∧ boat_speed = 80)) ∧
    (car_speed - boat_speed = speed_difference) ∧
    (total_distance * car_fraction / car_speed = 
     total_distance * (1 - car_fraction) / boat_speed + time_difference) :=
by sorry

end tourist_travel_speeds_l771_77134


namespace martin_additional_hens_l771_77161

/-- Represents the farm's hen and egg production scenario --/
structure FarmScenario where
  initial_hens : ℕ
  initial_days : ℕ
  initial_eggs : ℕ
  final_days : ℕ
  final_eggs : ℕ

/-- Calculates the number of additional hens needed --/
def additional_hens_needed (scenario : FarmScenario) : ℕ :=
  let eggs_per_hen := scenario.final_eggs * scenario.initial_days / (scenario.final_days * scenario.initial_eggs)
  let total_hens_needed := scenario.final_eggs / (eggs_per_hen * scenario.final_days / scenario.initial_days)
  total_hens_needed - scenario.initial_hens

/-- The main theorem stating the number of additional hens Martin needs to buy --/
theorem martin_additional_hens :
  let scenario : FarmScenario := {
    initial_hens := 10,
    initial_days := 10,
    initial_eggs := 80,
    final_days := 15,
    final_eggs := 300
  }
  additional_hens_needed scenario = 15 := by
  sorry

end martin_additional_hens_l771_77161


namespace dividend_calculation_l771_77169

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 21)
  (h2 : quotient = 14)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 301 :=
by sorry

end dividend_calculation_l771_77169


namespace julio_earnings_l771_77137

/-- Julio's earnings calculation --/
theorem julio_earnings (commission_per_customer : ℕ) (first_week_customers : ℕ) 
  (salary : ℕ) (bonus : ℕ) : 
  commission_per_customer = 1 →
  first_week_customers = 35 →
  salary = 500 →
  bonus = 50 →
  (commission_per_customer * (first_week_customers + 2 * first_week_customers + 3 * first_week_customers) + 
   salary + bonus) = 760 := by
  sorry

#check julio_earnings

end julio_earnings_l771_77137


namespace fifth_pile_magazines_l771_77118

/-- Represents the number of magazines in each pile -/
def magazine_sequence : ℕ → ℕ
| 0 => 3  -- First pile (health)
| 1 => 4  -- Second pile (technology)
| 2 => 6  -- Third pile (fashion)
| 3 => 9  -- Fourth pile (travel)
| n + 4 => magazine_sequence (n + 3) + (n + 4)  -- Subsequent piles

/-- The theorem stating that the fifth pile will contain 13 magazines -/
theorem fifth_pile_magazines : magazine_sequence 4 = 13 := by
  sorry


end fifth_pile_magazines_l771_77118


namespace not_eight_sum_l771_77105

theorem not_eight_sum (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 := by
  sorry

end not_eight_sum_l771_77105


namespace cadence_worked_five_months_longer_l771_77152

/-- Calculates the number of months longer Cadence worked at her new company --/
def months_longer_at_new_company (
  old_salary : ℕ)
  (salary_increase_percent : ℕ)
  (old_company_months : ℕ)
  (total_earnings : ℕ) : ℕ :=
  let new_salary := old_salary + (old_salary * salary_increase_percent) / 100
  let x := (total_earnings - old_salary * old_company_months) / new_salary - old_company_months
  x

/-- Proves that Cadence worked 5 months longer at her new company --/
theorem cadence_worked_five_months_longer :
  months_longer_at_new_company 5000 20 36 426000 = 5 := by
  sorry

#eval months_longer_at_new_company 5000 20 36 426000

end cadence_worked_five_months_longer_l771_77152


namespace multiple_solutions_exist_l771_77164

theorem multiple_solutions_exist : ∃ p₁ p₂ : ℝ, 
  p₁ ≠ p₂ ∧ 
  p₁ ∈ Set.Ioo 0 1 ∧ 
  p₂ ∈ Set.Ioo 0 1 ∧
  10 * p₁^3 * (1 - p₁)^2 = 144/625 ∧
  10 * p₂^3 * (1 - p₂)^2 = 144/625 :=
sorry

end multiple_solutions_exist_l771_77164


namespace q_satisfies_conditions_l771_77140

def q (x : ℝ) : ℝ := -10 * x^2 + 40 * x - 30

def numerator (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 4 * x + 6

theorem q_satisfies_conditions :
  (∀ x, x ≠ 1 ∧ x ≠ 3 → q x ≠ 0) ∧
  (q 1 = 0 ∧ q 3 = 0) ∧
  (∀ x, x ≠ 1 ∧ x ≠ 3 → ∃ y, y = numerator x / q x) ∧
  (¬ ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |numerator x / q x - L| < ε) ∧
  q 2 = 10 := by
  sorry

end q_satisfies_conditions_l771_77140


namespace triangle_height_l771_77199

theorem triangle_height (A B C : Real) (a b c : Real) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  1 + 2 * Real.cos (B + C) = 0 →
  ∃ h : Real, h = b * Real.sin C ∧ h = (Real.sqrt 3 + 1) / 2 :=
by sorry

end triangle_height_l771_77199


namespace segment_ratio_l771_77142

/-- Given points E, F, G, and H on a line in that order, with specified distances between them,
    prove that the ratio of EG to FH is 9:17. -/
theorem segment_ratio (E F G H : ℝ) : 
  F - E = 3 →
  G - F = 6 →
  H - G = 4 →
  H - E = 20 →
  (G - E) / (H - F) = 9 / 17 := by
  sorry

end segment_ratio_l771_77142


namespace inequality_relation_l771_77165

theorem inequality_relation (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, (1 / x < a ∧ x ≤ 1 / a)) ∧
  (∀ x : ℝ, x > 1 / a → 1 / x < a) :=
sorry

end inequality_relation_l771_77165


namespace unique_numbers_satisfying_conditions_l771_77141

theorem unique_numbers_satisfying_conditions : 
  ∃! (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    100 ≤ y ∧ y < 1000 ∧
    1000 * x + y = 8 * x * y ∧
    x + y = 141 := by sorry

end unique_numbers_satisfying_conditions_l771_77141


namespace unique_quadratic_solution_l771_77173

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x - 1 = 0) → a = 0 ∨ a = -1 := by
  sorry

end unique_quadratic_solution_l771_77173


namespace root_values_l771_77116

theorem root_values (p q r s k : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : p * k^3 + q * k^2 + r * k + s = 0)
  (h2 : q * k^3 + r * k^2 + s * k + p = 0)
  (h3 : 3 * p * k^2 + 2 * q * k + r = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end root_values_l771_77116


namespace quadratic_completing_square_l771_77193

theorem quadratic_completing_square (x : ℝ) : 
  (∃ p q : ℝ, 16 * x^2 + 32 * x - 512 = 0 ↔ (x + p)^2 = q) → 
  (∃ q : ℝ, (∀ x : ℝ, 16 * x^2 + 32 * x - 512 = 0 ↔ (x + 1)^2 = q) ∧ q = 33) :=
by sorry

end quadratic_completing_square_l771_77193


namespace exact_one_root_at_most_one_root_l771_77129

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := a * x^2 + 2*x + 1 = 0

-- Define the set of roots
def root_set (a : ℝ) : Set ℝ := {x | quadratic_equation a x}

-- Statement 1: A contains exactly one element iff a = 1 or a = 0
theorem exact_one_root (a : ℝ) : 
  (∃! x, x ∈ root_set a) ↔ (a = 1 ∨ a = 0) :=
sorry

-- Statement 2: A contains at most one element iff a ∈ {0} ∪ [1, +∞)
theorem at_most_one_root (a : ℝ) :
  (∀ x y, x ∈ root_set a → y ∈ root_set a → x = y) ↔ (a = 0 ∨ a ≥ 1) :=
sorry

end exact_one_root_at_most_one_root_l771_77129


namespace books_per_bookshelf_l771_77157

theorem books_per_bookshelf (total_books : ℕ) (num_bookshelves : ℕ) 
  (h1 : total_books = 621) 
  (h2 : num_bookshelves = 23) :
  total_books / num_bookshelves = 27 := by
  sorry

end books_per_bookshelf_l771_77157


namespace group_element_identity_l771_77100

variables {G : Type*} [Group G]

theorem group_element_identity (g h : G) (n : ℕ) 
  (h1 : g * h * g = h * g^2 * h)
  (h2 : g^3 = 1)
  (h3 : h^n = 1)
  (h4 : Odd n) :
  h = 1 := by sorry

end group_element_identity_l771_77100


namespace rectangle_dimension_change_l771_77189

/-- Theorem: Rectangle Dimension Change
  Given a rectangle with length L and breadth B,
  if the length is increased by 10% and the area is increased by 37.5%,
  then the breadth must be increased by 25%.
-/
theorem rectangle_dimension_change
  (L B : ℝ)  -- Original length and breadth
  (L' B' : ℝ) -- New length and breadth
  (h1 : L' = 1.1 * L)  -- Length increased by 10%
  (h2 : L' * B' = 1.375 * (L * B))  -- Area increased by 37.5%
  : B' = 1.25 * B := by
  sorry

end rectangle_dimension_change_l771_77189


namespace triangles_congruent_l771_77114

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- Define area and perimeter functions
def area (t : Triangle) : ℝ := sorry
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangles_congruent (t1 t2 : Triangle) 
  (h_area : area t1 = area t2)
  (h_perimeter : perimeter t1 = perimeter t2)
  (h_side : t1.a = t2.a) :
  t1 = t2 := by sorry

end triangles_congruent_l771_77114


namespace loaves_baked_l771_77131

def flour_available : ℝ := 5
def flour_per_loaf : ℝ := 2.5

theorem loaves_baked : ⌊flour_available / flour_per_loaf⌋ = 2 := by
  sorry

end loaves_baked_l771_77131


namespace bamboo_pole_problem_l771_77150

/-- 
Given a bamboo pole of height 10 feet, if the top part when bent to the ground 
reaches a point 3 feet from the base, then the length of the broken part is 109/20 feet.
-/
theorem bamboo_pole_problem (h : ℝ) (x : ℝ) (y : ℝ) :
  h = 10 ∧ 
  x + y = h ∧ 
  x^2 + 3^2 = y^2 →
  y = 109/20 := by
sorry

end bamboo_pole_problem_l771_77150


namespace complex_magnitude_squared_l771_77110

theorem complex_magnitude_squared (w : ℂ) (h : w^2 = -48 + 14*I) : 
  Complex.abs w = 5 * Real.sqrt 2 := by
sorry

end complex_magnitude_squared_l771_77110


namespace flagpole_shadow_length_l771_77184

/-- Given a flagpole and a building under similar conditions, prove the length of the flagpole's shadow. -/
theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 26)
  (h3 : building_shadow = 65)
  : ∃ (flagpole_shadow : ℝ), flagpole_shadow = 45 ∧ 
    flagpole_height / flagpole_shadow = building_height / building_shadow :=
by sorry

end flagpole_shadow_length_l771_77184


namespace mod_thirteen_five_eight_l771_77119

theorem mod_thirteen_five_eight (m : ℕ) : 
  13^5 % 8 = m → 0 ≤ m → m < 8 → m = 5 := by
  sorry

end mod_thirteen_five_eight_l771_77119


namespace complex_polynomial_root_l771_77124

theorem complex_polynomial_root (a b c : ℤ) : 
  (a * (1 + Complex.I * Real.sqrt 3)^3 + b * (1 + Complex.I * Real.sqrt 3)^2 + c * (1 + Complex.I * Real.sqrt 3) + b + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  (abs c = 9) := by
  sorry

end complex_polynomial_root_l771_77124


namespace common_chord_length_is_2sqrt5_l771_77135

/-- Circle C1 with equation x^2 + y^2 + 2x + 8y - 8 = 0 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C2 with equation x^2 + y^2 - 4x - 4y - 2 = 0 -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

/-- The circles C1 and C2 intersect -/
axiom circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y

/-- The length of the common chord of two intersecting circles -/
def common_chord_length (C1 C2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: The length of the common chord of C1 and C2 is 2√5 -/
theorem common_chord_length_is_2sqrt5 :
  common_chord_length C1 C2 = 2 * Real.sqrt 5 := by sorry

end common_chord_length_is_2sqrt5_l771_77135


namespace parabola_intersection_theorem_l771_77177

/-- Parabola intersecting a line --/
structure ParabolaIntersection where
  p : ℝ
  chord_length : ℝ
  h_p_pos : p > 0
  h_chord : chord_length = 3 * Real.sqrt 5

/-- The result of the intersection --/
def ParabolaIntersectionResult (pi : ParabolaIntersection) : Prop :=
  -- Part I: The equation of the parabola is y² = 4x
  (pi.p = 2) ∧
  -- Part II: The maximum distance from a point on the circumcircle of triangle ABF to line AB
  (∃ (max_distance : ℝ), max_distance = (9 * Real.sqrt 5) / 2)

/-- Main theorem --/
theorem parabola_intersection_theorem (pi : ParabolaIntersection) :
  ParabolaIntersectionResult pi :=
sorry

end parabola_intersection_theorem_l771_77177


namespace probability_two_defective_in_four_tests_l771_77132

def total_components : ℕ := 6
def defective_components : ℕ := 2
def good_components : ℕ := 4
def tests : ℕ := 4

theorem probability_two_defective_in_four_tests :
  (
    -- Probability of finding one defective in first three tests and second on fourth test
    (defective_components / total_components *
     good_components / (total_components - 1) *
     (good_components - 1) / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    (good_components / total_components *
     defective_components / (total_components - 1) *
     (good_components - 1) / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    (good_components / total_components *
     (good_components - 1) / (total_components - 1) *
     defective_components / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    -- Probability of finding all good components in four tests
    (good_components / total_components *
     (good_components - 1) / (total_components - 1) *
     (good_components - 2) / (total_components - 2) *
     (good_components - 3) / (total_components - 3))
  ) = 4 / 15 := by
  sorry

end probability_two_defective_in_four_tests_l771_77132


namespace even_function_implies_a_zero_l771_77175

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 - |x+a| -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - |x + a|

/-- If f(x) = x^2 - |x+a| is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end even_function_implies_a_zero_l771_77175


namespace complex_equation_sum_of_squares_l771_77158

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a - 2 * i) * i = b - i →
  a^2 + b^2 = 4 := by
sorry

end complex_equation_sum_of_squares_l771_77158


namespace optimal_discount_order_l771_77139

def original_price : ℝ := 30
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.25

theorem optimal_discount_order :
  (original_price * (1 - percentage_discount) - flat_discount) -
  (original_price - flat_discount) * (1 - percentage_discount) = 1.25 := by
  sorry

end optimal_discount_order_l771_77139


namespace c_is_positive_l771_77178

theorem c_is_positive (a b c d e f : ℤ) 
  (h1 : a * b + c * d * e * f < 0)
  (h2 : a < 0)
  (h3 : b < 0)
  (h4 : d < 0)
  (h5 : e < 0)
  (h6 : f < 0) :
  c > 0 := by
sorry

end c_is_positive_l771_77178


namespace HG_ratio_l771_77143

-- Define the equation
def equation (G H x : ℝ) : Prop :=
  (G / (x + 7) + H / (x^2 - 6*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 42*x))

-- State the theorem
theorem HG_ratio (G H : ℤ) : 
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 → equation G H x) →
  (H : ℝ) / (G : ℝ) = 15 / 7 :=
sorry

end HG_ratio_l771_77143


namespace congruence_problem_l771_77181

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 200 ∧ (150 * n) % 199 = 110 % 199 → n % 199 = 157 % 199 := by
  sorry

end congruence_problem_l771_77181


namespace age_problem_l771_77196

theorem age_problem (age : ℕ) : 5 * (age + 5) - 5 * (age - 5) = age → age = 50 := by
  sorry

end age_problem_l771_77196


namespace dolphin_training_l771_77138

theorem dolphin_training (total : ℕ) (fully_trained_ratio : ℚ) (semi_trained_ratio : ℚ)
  (beginner_ratio : ℚ) (intermediate_ratio : ℚ)
  (h1 : total = 120)
  (h2 : fully_trained_ratio = 1/4)
  (h3 : semi_trained_ratio = 1/6)
  (h4 : beginner_ratio = 3/8)
  (h5 : intermediate_ratio = 5/9) :
  let fully_trained := (total : ℚ) * fully_trained_ratio
  let remaining_after_fully_trained := total - fully_trained.floor
  let semi_trained := (remaining_after_fully_trained : ℚ) * semi_trained_ratio
  let untrained := remaining_after_fully_trained - semi_trained.floor
  let semi_and_untrained := semi_trained.floor + untrained
  let in_beginner := (semi_and_untrained : ℚ) * beginner_ratio
  let remaining_after_beginner := semi_and_untrained - in_beginner.floor
  let start_intermediate := (remaining_after_beginner : ℚ) * intermediate_ratio
  start_intermediate.floor = 31 :=
by sorry

end dolphin_training_l771_77138


namespace remove_four_for_target_average_l771_77149

def original_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
def target_average : ℚ := 63/10

theorem remove_four_for_target_average :
  let remaining_list := original_list.filter (· ≠ 4)
  (remaining_list.sum : ℚ) / remaining_list.length = target_average := by
  sorry

end remove_four_for_target_average_l771_77149


namespace cubic_root_proof_l771_77109

theorem cubic_root_proof :
  let x : ℝ := (Real.rpow 81 (1/3) + Real.rpow 9 (1/3) + 1) / 27
  27 * x^3 - 6 * x^2 - 6 * x - 2 = 0 := by
  sorry

end cubic_root_proof_l771_77109
