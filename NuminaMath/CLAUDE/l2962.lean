import Mathlib

namespace first_term_of_constant_ratio_l2962_296291

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℚ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (c : ℚ) :
  d = 5 →
  (∀ n : ℕ, n > 0 → 
    (arithmetic_sum a d (2*n)) / (arithmetic_sum a d n) = c) →
  a = 5/2 :=
sorry

end first_term_of_constant_ratio_l2962_296291


namespace overtaking_distance_l2962_296205

/-- Represents a vehicle with a given length -/
structure Vehicle where
  length : ℝ

/-- Represents the overtaking scenario on a highway -/
structure OvertakingScenario where
  sedan : Vehicle
  truck : Vehicle

/-- The additional distance traveled by the sedan during overtaking -/
def additionalDistance (scenario : OvertakingScenario) : ℝ :=
  scenario.sedan.length + scenario.truck.length

theorem overtaking_distance (scenario : OvertakingScenario) :
  additionalDistance scenario = scenario.sedan.length + scenario.truck.length := by
  sorry

end overtaking_distance_l2962_296205


namespace tetrahedron_face_area_relation_l2962_296225

/-- Theorem about the relationship between face areas, edges, and angles in a tetrahedron -/
theorem tetrahedron_face_area_relation 
  (S₁ S₂ a b : ℝ) (α φ : ℝ) 
  (h_S₁ : S₁ > 0) (h_S₂ : S₂ > 0) 
  (h_a : a > 0) (h_b : b > 0)
  (h_α : 0 < α ∧ α < π) (h_φ : 0 < φ ∧ φ < π) :
  S₁^2 + S₂^2 - 2*S₁*S₂*(Real.cos α) = (a*b*(Real.sin φ) / 4)^2 := by
  sorry

end tetrahedron_face_area_relation_l2962_296225


namespace average_of_three_l2962_296215

theorem average_of_three (total : ℝ) (avg_all : ℝ) (avg_two : ℝ) :
  total = 5 →
  avg_all = 10 →
  avg_two = 19 →
  (total * avg_all - 2 * avg_two) / 3 = 4 := by
sorry

end average_of_three_l2962_296215


namespace second_serving_is_ten_l2962_296231

/-- Represents the number of maggots in various scenarios --/
structure MaggotCounts where
  total : ℕ
  firstServing : ℕ
  firstEaten : ℕ
  secondEaten : ℕ

/-- Calculates the number of maggots in the second serving --/
def secondServing (counts : MaggotCounts) : ℕ :=
  counts.total - counts.firstServing

/-- Theorem stating that the second serving contains 10 maggots --/
theorem second_serving_is_ten (counts : MaggotCounts)
  (h1 : counts.total = 20)
  (h2 : counts.firstServing = 10)
  (h3 : counts.firstEaten = 1)
  (h4 : counts.secondEaten = 3) :
  secondServing counts = 10 := by
  sorry

#eval secondServing { total := 20, firstServing := 10, firstEaten := 1, secondEaten := 3 }

end second_serving_is_ten_l2962_296231


namespace can_capacity_proof_l2962_296294

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 36

/-- The amount of milk added in liters -/
def milkAdded : ℝ := 8

theorem can_capacity_proof (initial : CanContents) (final : CanContents) :
  -- Initial ratio of milk to water is 4:3
  initial.milk / initial.water = 4 / 3 →
  -- Final contents after adding milk
  final.milk = initial.milk + milkAdded ∧
  final.water = initial.water →
  -- Can is full after adding milk
  final.milk + final.water = canCapacity →
  -- Final ratio of milk to water is 2:1
  final.milk / final.water = 2 / 1 →
  -- Prove that the capacity of the can is 36 liters
  canCapacity = 36 := by
  sorry


end can_capacity_proof_l2962_296294


namespace at_least_four_same_prob_l2962_296212

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a specific value on a single die -/
def single_prob : ℚ := 1 / sides

/-- The probability that all five dice show the same number -/
def all_same_prob : ℚ := single_prob ^ (num_dice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def four_same_prob : ℚ := num_dice * (single_prob ^ (num_dice - 2)) * ((sides - 1) / sides)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_prob : all_same_prob + four_same_prob = 13 / 648 := by
  sorry

end at_least_four_same_prob_l2962_296212


namespace ellipse_equation_with_given_parameters_l2962_296241

/-- Standard equation of an ellipse with foci on coordinate axes -/
def standard_ellipse_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}

/-- Theorem: Standard equation of ellipse with given parameters -/
theorem ellipse_equation_with_given_parameters :
  ∀ (a c : ℝ),
  a^2 = 13 →
  c^2 = 12 →
  ∃ (b : ℝ),
  b^2 = 1 ∧
  (standard_ellipse_equation 13 1 = standard_ellipse_equation 13 1 ∨
   standard_ellipse_equation 1 13 = standard_ellipse_equation 1 13) :=
by sorry

end ellipse_equation_with_given_parameters_l2962_296241


namespace gcd_456_357_l2962_296239

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  -- The proof would go here
  sorry

end gcd_456_357_l2962_296239


namespace derivative_y_l2962_296229

noncomputable def y (x : ℝ) : ℝ := Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

theorem derivative_y (x : ℝ) (h : 2 * x + 3 > 0) :
  deriv y x = (4 * Real.sqrt (x^2 + 3 * x + 2)) / (2 * x + 3) := by sorry

end derivative_y_l2962_296229


namespace same_terminal_side_l2962_296273

theorem same_terminal_side (θ : ℝ) : 
  ∃ k : ℤ, θ + 360 * k = 330 → θ = -30 := by sorry

end same_terminal_side_l2962_296273


namespace set_operations_l2962_296298

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 3, 5, 7}
def B : Set Nat := {3, 5}

theorem set_operations :
  (A ∪ B = {1, 3, 5, 7}) ∧
  ((U \ A) ∪ B = {2, 3, 4, 5, 6}) := by
  sorry

end set_operations_l2962_296298


namespace statue_weight_l2962_296227

-- Define the initial weight and cutting percentages
def initial_weight : ℝ := 250
def first_cut : ℝ := 0.30
def second_cut : ℝ := 0.20
def third_cut : ℝ := 0.25

-- Define the final weight calculation
def final_weight : ℝ :=
  initial_weight * (1 - first_cut) * (1 - second_cut) * (1 - third_cut)

-- Theorem statement
theorem statue_weight :
  final_weight = 105 := by sorry

end statue_weight_l2962_296227


namespace sum_of_rotated_digits_l2962_296286

theorem sum_of_rotated_digits : 
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end sum_of_rotated_digits_l2962_296286


namespace midpoint_triangle_area_ratio_l2962_296243

/-- Given a triangle with area T, the triangle formed by joining the midpoints of its sides has area M = T/4 -/
theorem midpoint_triangle_area_ratio (T : ℝ) (h : T > 0) : 
  ∃ M : ℝ, M = T / 4 ∧ M > 0 := by
sorry

end midpoint_triangle_area_ratio_l2962_296243


namespace parallelogram_base_length_l2962_296299

/-- Proves that a parallelogram with area 288 sq m and altitude twice the base has a base length of 12 m -/
theorem parallelogram_base_length : 
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  altitude = 2 * base →
  base * altitude = 288 →
  base = 12 := by
sorry

end parallelogram_base_length_l2962_296299


namespace principle_countable_noun_meaning_l2962_296290

/-- Define a type for English words -/
def EnglishWord : Type := String

/-- Define a type for word meanings -/
def WordMeaning : Type := String

/-- Function to get the meaning of a word when used as a countable noun -/
def countableNounMeaning (word : EnglishWord) : WordMeaning :=
  sorry

/-- Theorem stating that "principle" as a countable noun means "principle, criterion" -/
theorem principle_countable_noun_meaning :
  countableNounMeaning "principle" = "principle, criterion" :=
sorry

end principle_countable_noun_meaning_l2962_296290


namespace unique_z_value_l2962_296270

theorem unique_z_value : ∃! z : ℝ,
  (∃ x : ℤ, x = ⌊z⌋ ∧ 3 * x^2 + 19 * x - 84 = 0) ∧
  (∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ y = z - ⌊z⌋ ∧ 4 * y^2 - 14 * y + 6 = 0) ∧
  z = -11 := by
sorry

end unique_z_value_l2962_296270


namespace all_students_accounted_for_no_unsatisfactory_grades_l2962_296211

theorem all_students_accounted_for (top_marks : ℚ) (average_marks : ℚ) (good_marks : ℚ)
  (h1 : top_marks = 1 / 6)
  (h2 : average_marks = 1 / 3)
  (h3 : good_marks = 1 / 2) :
  top_marks + average_marks + good_marks = 1 :=
by
  sorry

theorem no_unsatisfactory_grades (total_fraction : ℚ)
  (h : total_fraction = 1) :
  1 - total_fraction = 0 :=
by
  sorry

end all_students_accounted_for_no_unsatisfactory_grades_l2962_296211


namespace concentric_circles_radii_difference_l2962_296222

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r is radius of smaller circle, R is radius of larger circle
  (h_positive : r > 0) -- r is positive
  (h_area_ratio : R^2 / r^2 = 16 / 3) -- area ratio condition
  : R - r = r * (4 * Real.sqrt 3 - 3) / 3 := by
  sorry

end concentric_circles_radii_difference_l2962_296222


namespace library_growth_rate_l2962_296271

theorem library_growth_rate (initial_collection : ℝ) (final_collection : ℝ) (years : ℝ) :
  initial_collection = 100000 →
  final_collection = 144000 →
  years = 2 →
  let growth_rate := ((final_collection / initial_collection) ^ (1 / years)) - 1
  growth_rate = 0.2 := by
sorry

end library_growth_rate_l2962_296271


namespace integral_problem_1_l2962_296223

theorem integral_problem_1 (x : ℝ) (h : x > 0) :
  (deriv (fun x => 4 * (x^(1/2)/2 - x^(1/4) + Real.log (1 + x^(1/4)))) x) = 1 / (x^(1/2) + x^(1/4)) :=
sorry

end integral_problem_1_l2962_296223


namespace sum_of_ages_in_future_l2962_296208

-- Define Will's age 3 years ago
def will_age_3_years_ago : ℕ := 4

-- Define the current year (relative to the problem's frame)
def current_year : ℕ := 3

-- Define the future year we're interested in
def future_year : ℕ := 5

-- Define Will's current age
def will_current_age : ℕ := will_age_3_years_ago + current_year

-- Define Diane's current age
def diane_current_age : ℕ := 2 * will_current_age

-- Define Will's future age
def will_future_age : ℕ := will_current_age + future_year

-- Define Diane's future age
def diane_future_age : ℕ := diane_current_age + future_year

-- Theorem to prove
theorem sum_of_ages_in_future : will_future_age + diane_future_age = 31 := by
  sorry

end sum_of_ages_in_future_l2962_296208


namespace sum_of_squares_from_means_l2962_296221

theorem sum_of_squares_from_means (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := by
  sorry

end sum_of_squares_from_means_l2962_296221


namespace no_rational_solution_l2962_296207

theorem no_rational_solution : ¬∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (1 : ℚ) / (x - y)^2 + (1 : ℚ) / (y - z)^2 + (1 : ℚ) / (z - x)^2 = 2014 := by
  sorry

end no_rational_solution_l2962_296207


namespace double_age_in_years_until_double_l2962_296214

/-- The number of years until I'm twice my brother's age -/
def years_until_double : ℕ := 10

/-- My current age -/
def my_current_age : ℕ := 20

/-- My brother's current age -/
def brothers_current_age : ℕ := my_current_age - years_until_double

theorem double_age_in_years_until_double :
  (my_current_age + years_until_double) = 2 * (brothers_current_age + years_until_double) ∧
  (my_current_age + years_until_double) + (brothers_current_age + years_until_double) = 45 :=
by sorry

end double_age_in_years_until_double_l2962_296214


namespace equation_solution_l2962_296210

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end equation_solution_l2962_296210


namespace square_difference_emily_calculation_l2962_296289

theorem square_difference (a b : ℕ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry

theorem emily_calculation : 50^2 - 46^2 = 384 := by sorry

end square_difference_emily_calculation_l2962_296289


namespace percentage_saved_approx_l2962_296238

/-- Represents the discount information for each day of the sale -/
structure DayDiscount where
  minQuantity : Nat
  discountedQuantity : Nat
  discountedPrice : Nat

/-- Calculates the savings for a given day's discount -/
def calculateSavings (discount : DayDiscount) : Nat :=
  discount.minQuantity - discount.discountedPrice

/-- Calculates the total savings and original price for all days -/
def calculateTotals (discounts : List DayDiscount) : (Nat × Nat) :=
  let savings := discounts.map calculateSavings |>.sum
  let originalPrice := discounts.map (fun d => d.minQuantity) |>.sum
  (savings, originalPrice)

/-- The discounts for each day of the five-day sale -/
def saleDays : List DayDiscount := [
  { minQuantity := 11, discountedQuantity := 12, discountedPrice := 4 },
  { minQuantity := 15, discountedQuantity := 15, discountedPrice := 5 },
  { minQuantity := 18, discountedQuantity := 18, discountedPrice := 6 },
  { minQuantity := 21, discountedQuantity := 25, discountedPrice := 8 },
  { minQuantity := 26, discountedQuantity := 30, discountedPrice := 10 }
]

/-- Theorem stating that the percentage saved is approximately 63.74% -/
theorem percentage_saved_approx (ε : ℝ) (h : ε > 0) :
  let (savings, originalPrice) := calculateTotals saleDays
  let percentageSaved := (savings : ℝ) / (originalPrice : ℝ) * 100
  |percentageSaved - 63.74| < ε :=
sorry

end percentage_saved_approx_l2962_296238


namespace savings_for_engagement_ring_l2962_296295

/-- Calculates the monthly savings required to accumulate two months' salary in a given time period. -/
def monthly_savings (annual_salary : ℚ) (months_to_save : ℕ) : ℚ :=
  (2 * annual_salary) / (12 * months_to_save)

/-- Proves that given an annual salary of $60,000 and 10 months to save,
    the amount to save per month to accumulate two months' salary is $1,000. -/
theorem savings_for_engagement_ring :
  monthly_savings 60000 10 = 1000 := by
  sorry

end savings_for_engagement_ring_l2962_296295


namespace negative_product_implies_odd_negatives_l2962_296251

theorem negative_product_implies_odd_negatives (a b c : ℝ) : 
  a * b * c < 0 → (a < 0 ∧ b < 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0) ∨ 
                   (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a > 0 ∧ b > 0 ∧ c < 0) := by
  sorry

end negative_product_implies_odd_negatives_l2962_296251


namespace square_between_prime_sums_l2962_296203

/-- Sum of the first n prime numbers -/
def S (n : ℕ) : ℕ := sorry

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem square_between_prime_sums :
  ∀ n : ℕ, n > 0 → ∃ k : ℕ, S n < k^2 ∧ k^2 < S (n + 1) :=
sorry

end square_between_prime_sums_l2962_296203


namespace ceiling_floor_sum_l2962_296254

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l2962_296254


namespace cows_gifted_is_eight_l2962_296218

/-- Calculates the number of cows given as a gift -/
def cows_gifted (initial : ℕ) (died : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (total : ℕ) : ℕ :=
  total - (initial - died - sold + increased + bought)

/-- Theorem stating that the number of cows gifted is 8 -/
theorem cows_gifted_is_eight :
  cows_gifted 39 25 6 24 43 83 = 8 := by
  sorry

end cows_gifted_is_eight_l2962_296218


namespace gcd_100_450_l2962_296213

theorem gcd_100_450 : Nat.gcd 100 450 = 50 := by
  sorry

end gcd_100_450_l2962_296213


namespace cost_price_per_metre_l2962_296230

/-- Given the total cloth length, total selling price, and loss per metre, 
    calculate the cost price for one metre of cloth. -/
theorem cost_price_per_metre 
  (total_length : ℕ) 
  (total_selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (h1 : total_length = 200)
  (h2 : total_selling_price = 12000)
  (h3 : loss_per_metre = 12) : 
  (total_selling_price + total_length * loss_per_metre) / total_length = 72 := by
  sorry

#check cost_price_per_metre

end cost_price_per_metre_l2962_296230


namespace custom_operation_equality_l2962_296232

-- Define the custom operation
def delta (a b : ℝ) : ℝ := a^3 - 2*b

-- State the theorem
theorem custom_operation_equality :
  let x := delta 6 8
  let y := delta 2 7
  delta (5^x) (2^y) = (5^200)^3 - 1/32 := by sorry

end custom_operation_equality_l2962_296232


namespace inscribed_circle_radius_rhombus_l2962_296217

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let r := (d1 * d2) / (4 * a)
  r = 60 / 13 := by
  sorry

end inscribed_circle_radius_rhombus_l2962_296217


namespace complex_equation_l2962_296252

theorem complex_equation (z : ℂ) (h : z = 1 + I) : z^2 + 2 / z = 1 + I := by
  sorry

end complex_equation_l2962_296252


namespace probability_of_white_ball_l2962_296264

theorem probability_of_white_ball (p_red_or_white p_yellow_or_white : ℝ) 
  (h1 : p_red_or_white = 0.65)
  (h2 : p_yellow_or_white = 0.6) :
  1 - (1 - p_yellow_or_white) - (1 - p_red_or_white) = 0.25 := by
  sorry

end probability_of_white_ball_l2962_296264


namespace coin_weighing_possible_l2962_296280

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents a 2-pan scale that can compare two sets of coins -/
def Scale : (List Coin) → (List Coin) → WeighResult := sorry

/-- The main theorem to be proven -/
theorem coin_weighing_possible (k : ℕ) : 
  ∀ (coins : List Coin), 
    coins.length = 2^k → 
    (∀ a b c : Coin, a ∈ coins → b ∈ coins → c ∈ coins → 
      (a.weight ≠ b.weight ∧ b.weight ≠ c.weight) → a.weight = c.weight) →
    ∃ (measurements : List (List Coin × List Coin)),
      measurements.length ≤ k ∧
      (∀ m ∈ measurements, m.1.length + m.2.length ≤ coins.length) ∧
      (∃ (heavy light : Coin), heavy ∈ coins ∧ light ∈ coins ∧ heavy.weight > light.weight) ∨
      (∀ c1 c2 : Coin, c1 ∈ coins → c2 ∈ coins → c1.weight = c2.weight) := by
  sorry

end coin_weighing_possible_l2962_296280


namespace relative_errors_equal_l2962_296209

theorem relative_errors_equal (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 20)
  (h2 : length2 = 150)
  (h3 : error1 = 0.04)
  (h4 : error2 = 0.3) :
  error1 / length1 = error2 / length2 := by
  sorry

end relative_errors_equal_l2962_296209


namespace student_count_is_35_l2962_296275

/-- The number of different Roman numerals -/
def num_roman_numerals : ℕ := 7

/-- The number of sketches for each Roman numeral -/
def sketches_per_numeral : ℕ := 5

/-- The total number of students in the class -/
def num_students : ℕ := num_roman_numerals * sketches_per_numeral

theorem student_count_is_35 : num_students = 35 := by
  sorry

end student_count_is_35_l2962_296275


namespace point_3_0_on_line_point_0_4_on_line_is_line_equation_main_theorem_l2962_296258

/-- The line passing through points (3, 0) and (0, 4) -/
def line_equation (x y : ℝ) : Prop := 4*x + 3*y - 12 = 0

/-- Point (3, 0) lies on the line -/
theorem point_3_0_on_line : line_equation 3 0 := by sorry

/-- Point (0, 4) lies on the line -/
theorem point_0_4_on_line : line_equation 0 4 := by sorry

/-- The equation represents a line -/
theorem is_line_equation : ∃ (m b : ℝ), ∀ (x y : ℝ), line_equation x y ↔ y = m*x + b := by sorry

/-- Main theorem: The given equation represents the unique line passing through (3, 0) and (0, 4) -/
theorem main_theorem : 
  ∀ (f : ℝ → ℝ → Prop), 
  (f 3 0 ∧ f 0 4 ∧ (∃ (m b : ℝ), ∀ (x y : ℝ), f x y ↔ y = m*x + b)) → 
  (∀ (x y : ℝ), f x y ↔ line_equation x y) := by sorry

end point_3_0_on_line_point_0_4_on_line_is_line_equation_main_theorem_l2962_296258


namespace part_one_part_two_l2962_296278

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 5

-- Part 1
theorem part_one (x : ℝ) : 
  p x 1 → q x → x ∈ Set.Ioo 2 4 :=
sorry

-- Part 2
theorem part_two (a : ℝ) : 
  a > 0 → 
  (Set.Ioo 2 5 ⊂ {x | p x a}) → 
  ({x | p x a} ≠ Set.Ioo 2 5) → 
  a ∈ Set.Ioc (5/4) 2 :=
sorry

end part_one_part_two_l2962_296278


namespace condition_type_l2962_296233

theorem condition_type (A B : Prop) 
  (h1 : ¬B → ¬A) 
  (h2 : ¬(¬A → ¬B)) : 
  (A → B) ∧ ¬(B → A) := by sorry

end condition_type_l2962_296233


namespace triangle_y_value_l2962_296237

-- Define the triangle
structure AcuteTriangle where
  a : ℝ
  y : ℝ
  area_small : ℝ

-- Define the properties of the triangle
def triangle_properties (t : AcuteTriangle) : Prop :=
  t.a > 0 ∧ t.y > 0 ∧
  6 > 0 ∧ 4 > 0 ∧
  t.area_small = 12 ∧
  (6 * (6 + t.y) = t.y * (10 + t.a)) ∧
  (1/2 * 10 * (24 / t.y) = 12)

-- Theorem statement
theorem triangle_y_value (t : AcuteTriangle) 
  (h : triangle_properties t) : t.y = 10 := by
  sorry

end triangle_y_value_l2962_296237


namespace ratio_equality_l2962_296296

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end ratio_equality_l2962_296296


namespace dark_lord_squads_l2962_296228

/-- The number of squads needed to transport swords --/
def num_squads (total_weight : ℕ) (orcs_per_squad : ℕ) (weight_per_orc : ℕ) : ℕ :=
  total_weight / (orcs_per_squad * weight_per_orc)

/-- Proof that 10 squads are needed for the given conditions --/
theorem dark_lord_squads :
  num_squads 1200 8 15 = 10 := by
  sorry

end dark_lord_squads_l2962_296228


namespace ellipse_sum_property_l2962_296201

/-- Represents an ellipse with its properties -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length
  θ : ℝ  -- rotation angle in radians

/-- Theorem: For a specific ellipse, the sum of its center coordinates and axis lengths is 11 -/
theorem ellipse_sum_property : 
  ∀ (e : Ellipse), 
  e.h = -2 ∧ e.k = 3 ∧ e.a = 6 ∧ e.b = 4 ∧ e.θ = π/4 → 
  e.h + e.k + e.a + e.b = 11 := by
sorry

end ellipse_sum_property_l2962_296201


namespace scientific_notation_eleven_million_l2962_296279

theorem scientific_notation_eleven_million :
  (11000000 : ℝ) = 1.1 * (10 ^ 7) := by
  sorry

end scientific_notation_eleven_million_l2962_296279


namespace expression_simplification_l2962_296249

def simplify_expression (a b : ℤ) : ℤ :=
  -2 * (10 * a^2 + 2 * a * b + 3 * b^2) + 3 * (5 * a^2 - 4 * a * b)

theorem expression_simplification :
  simplify_expression 1 (-2) = 3 := by
  sorry

end expression_simplification_l2962_296249


namespace car_profit_percentage_l2962_296245

theorem car_profit_percentage (P : ℝ) (P_positive : P > 0) : 
  let discount_rate : ℝ := 0.20
  let increase_rate : ℝ := 0.70
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - P
  let profit_percentage : ℝ := (profit / P) * 100
  profit_percentage = 36 := by sorry

end car_profit_percentage_l2962_296245


namespace columbia_arrangements_l2962_296265

def columbia_letters : Nat := 9
def repeated_i : Nat := 2
def repeated_u : Nat := 2

theorem columbia_arrangements :
  (columbia_letters.factorial) / (repeated_i.factorial * repeated_u.factorial) = 90720 := by
  sorry

end columbia_arrangements_l2962_296265


namespace cos_30_minus_cos_60_l2962_296293

theorem cos_30_minus_cos_60 : Real.cos (π / 6) - Real.cos (π / 3) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end cos_30_minus_cos_60_l2962_296293


namespace paint_project_cost_l2962_296236

/-- Calculates the total cost of paint and primer for a house painting project. -/
def total_cost (rooms : ℕ) (primer_cost : ℚ) (primer_discount : ℚ) (paint_cost : ℚ) : ℚ :=
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := rooms * discounted_primer_cost
  let total_paint_cost := rooms * paint_cost
  total_primer_cost + total_paint_cost

/-- Proves that the total cost for paint and primer is $245.00 under given conditions. -/
theorem paint_project_cost :
  total_cost 5 30 (1/5) 25 = 245 :=
by sorry

end paint_project_cost_l2962_296236


namespace no_lines_satisfy_conditions_l2962_296268

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def line_through_point (a b : ℝ) : Prop :=
  6 / a + 5 / b = 1

def satisfies_conditions (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ is_prime a ∧ a + b < 20 ∧ line_through_point a b

theorem no_lines_satisfy_conditions :
  ¬ ∃ a b : ℕ, satisfies_conditions a b :=
sorry

end no_lines_satisfy_conditions_l2962_296268


namespace polynomial_divisibility_l2962_296266

theorem polynomial_divisibility (a b : ℚ) : 
  (∀ x : ℚ, (x^2 - x - 2) ∣ (a * x^4 + b * x^2 + 1)) ↔ 
  (a = 1/4 ∧ b = -5/4) :=
by sorry

end polynomial_divisibility_l2962_296266


namespace fifth_root_of_unity_l2962_296263

theorem fifth_root_of_unity (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * u^4 + q * u^3 + r * u^2 + s * u + t = 0)
  (h2 : q * u^4 + r * u^3 + s * u^2 + t * u + p = 0) :
  u^5 = 1 :=
sorry

end fifth_root_of_unity_l2962_296263


namespace book_arrangements_eq_120960_l2962_296242

/-- The number of ways to arrange 4 different math books and 5 different history books on a bookshelf,
    with a math book at both ends and exactly one math book in the middle -/
def book_arrangements : ℕ :=
  let math_books := 4
  let history_books := 5
  let end_arrangements := math_books * (math_books - 1)
  let middle_math_book := math_books - 2
  let remaining_books := (math_books - 3) + history_books
  end_arrangements * middle_math_book * Nat.factorial remaining_books

theorem book_arrangements_eq_120960 : book_arrangements = 120960 := by
  sorry

end book_arrangements_eq_120960_l2962_296242


namespace inequality_proof_l2962_296234

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 
  3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end inequality_proof_l2962_296234


namespace coffee_price_increase_l2962_296253

/-- Calculates the percentage increase in coffee price given the original conditions and savings. -/
theorem coffee_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (new_quantity : ℕ) 
  (daily_savings : ℝ) 
  (h1 : original_price = 2)
  (h2 : original_quantity = 4)
  (h3 : new_quantity = 2)
  (h4 : daily_savings = 2) : 
  (((original_price * original_quantity - daily_savings) / new_quantity - original_price) / original_price) * 100 = 50 := by
  sorry

#check coffee_price_increase

end coffee_price_increase_l2962_296253


namespace trig_identity_l2962_296240

theorem trig_identity (α : ℝ) : 
  (Real.sin α + Real.sin (3 * α) - Real.sin (5 * α)) / 
  (Real.cos α - Real.cos (3 * α) - Real.cos (5 * α)) = Real.tan α :=
sorry

end trig_identity_l2962_296240


namespace cosine_sum_theorem_l2962_296285

theorem cosine_sum_theorem : 
  Real.cos 0 ^ 4 + Real.cos (π / 6) ^ 4 + Real.cos (π / 3) ^ 4 + Real.cos (π / 2) ^ 4 + 
  Real.cos (2 * π / 3) ^ 4 + Real.cos (5 * π / 6) ^ 4 + Real.cos π ^ 4 = 13 / 4 := by
  sorry

end cosine_sum_theorem_l2962_296285


namespace no_integer_solution_l2962_296247

theorem no_integer_solution :
  ¬ ∃ (a b x y : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧
    a * x - b * y = 16 ∧
    a * y + b * x = 1 :=
by sorry

end no_integer_solution_l2962_296247


namespace max_m_condition_l2962_296269

theorem max_m_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ m : ℝ, 4 / a + 1 / b ≥ m / (a + 4 * b)) →
  (∃ m_max : ℝ, ∀ m : ℝ, m ≤ m_max ∧ (m = m_max ↔ b / a = 1 / 4)) :=
by sorry

end max_m_condition_l2962_296269


namespace ellipse_dot_product_range_l2962_296267

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the left focus and left vertex
def F₁ : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (-2, 0)

-- Define the dot product of PF₁ and PA
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1) * (x + 2) + y^2

-- Theorem statement
theorem ellipse_dot_product_range :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → 0 ≤ dot_product P ∧ dot_product P ≤ 12 :=
sorry

end ellipse_dot_product_range_l2962_296267


namespace unique_valid_cube_configuration_l2962_296256

-- Define a cube face
inductive Face
| White
| Gray
| Mixed

-- Define a cube
structure Cube :=
(front back left right top bottom : Face)

-- Define the conditions
def oppositeFacesValid (c : Cube) : Prop :=
  (c.front = Face.White → c.back = Face.Gray) ∧
  (c.left = Face.White → c.right = Face.Gray) ∧
  (c.top = Face.White → c.bottom = Face.Gray)

def adjacentFacesValid (c : Cube) : Prop :=
  (c.front = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed ∧ 
                          c.left ≠ Face.Mixed ∧ c.right ≠ Face.Mixed) ∧
  (c.back = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed ∧ 
                         c.left ≠ Face.Mixed ∧ c.right ≠ Face.Mixed) ∧
  (c.left = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed) ∧
  (c.right = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed)

-- Theorem stating the uniqueness of the valid cube configuration
theorem unique_valid_cube_configuration :
  ∃! c : Cube, oppositeFacesValid c ∧ adjacentFacesValid c :=
sorry

end unique_valid_cube_configuration_l2962_296256


namespace range_of_m_l2962_296219

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem range_of_m : 
  (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) → 
  (∀ m : ℝ, (1 < m ∧ m ≤ 2) ∨ m ≥ 3 ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
sorry

end range_of_m_l2962_296219


namespace set_difference_equals_open_interval_l2962_296274

/-- The set A of real numbers x such that |4x - 1| > 9 -/
def A : Set ℝ := {x | |4*x - 1| > 9}

/-- The set B of non-negative real numbers -/
def B : Set ℝ := {x | x ≥ 0}

/-- The open interval (5/2, +∞) -/
def openInterval : Set ℝ := {x | x > 5/2}

/-- Theorem stating that the set difference A - B is equal to the open interval (5/2, +∞) -/
theorem set_difference_equals_open_interval : A \ B = openInterval := by
  sorry

end set_difference_equals_open_interval_l2962_296274


namespace intersection_M_N_l2962_296284

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end intersection_M_N_l2962_296284


namespace always_defined_division_by_two_l2962_296216

theorem always_defined_division_by_two (a : ℝ) : ∃ (x : ℝ), x = a / 2 := by
  sorry

end always_defined_division_by_two_l2962_296216


namespace largest_m_for_factorization_l2962_296235

theorem largest_m_for_factorization : 
  ∀ m : ℤ, (∃ a b c d : ℤ, 5 * x^2 + m * x + 120 = (a * x + b) * (c * x + d)) → m ≤ 601 :=
by sorry

end largest_m_for_factorization_l2962_296235


namespace min_half_tiles_for_29_l2962_296226

/-- Represents a tiling of a square area -/
structure Tiling where
  size : ℕ  -- The size of the square area in unit squares
  whole_tiles : ℕ  -- Number of whole tiles used
  half_tiles : ℕ  -- Number of tiles cut in half

/-- Checks if a tiling is valid for the given area -/
def is_valid_tiling (t : Tiling) : Prop :=
  t.whole_tiles + t.half_tiles / 2 = t.size

/-- Theorem: The minimum number of tiles to be cut in half for a 29-unit square area is 12 -/
theorem min_half_tiles_for_29 :
  ∀ t : Tiling, t.size = 29 → is_valid_tiling t →
  t.half_tiles ≥ 12 ∧ ∃ t' : Tiling, t'.size = 29 ∧ is_valid_tiling t' ∧ t'.half_tiles = 12 :=
by sorry

#check min_half_tiles_for_29

end min_half_tiles_for_29_l2962_296226


namespace dogs_can_prevent_wolf_escape_l2962_296257

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square plot -/
structure Square where
  side : ℝ
  center : Point

/-- Represents an animal (wolf or dog) -/
structure Animal where
  position : Point
  speed : ℝ

/-- Represents the game state -/
structure GameState where
  square : Square
  wolf : Animal
  dogs : List Animal

/-- Checks if a point is inside or on the boundary of a square -/
def isInsideSquare (s : Square) (p : Point) : Prop :=
  abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2

/-- Checks if a point is on the boundary of a square -/
def isOnSquareBoundary (s : Square) (p : Point) : Prop :=
  (abs (p.x - s.center.x) = s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2) ∨
  (abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) = s.side / 2)

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_wolf_escape (g : GameState) 
  (h1 : g.wolf.position = g.square.center) 
  (h2 : ∀ d ∈ g.dogs, isOnSquareBoundary g.square d.position)
  (h3 : ∀ d ∈ g.dogs, d.speed = 1.5 * g.wolf.speed)
  (h4 : g.dogs.length = 4) :
  ∀ t : ℝ, ∃ strategy : ℝ → List Point, 
    (∀ p ∈ strategy t, isOnSquareBoundary g.square p) ∧ 
    isInsideSquare g.square (g.wolf.position) :=
sorry

end dogs_can_prevent_wolf_escape_l2962_296257


namespace lila_tulips_l2962_296276

/-- Calculates the number of tulips after maintaining the ratio --/
def final_tulips (initial_orchids : ℕ) (added_orchids : ℕ) (tulip_ratio : ℕ) (orchid_ratio : ℕ) : ℕ :=
  let final_orchids := initial_orchids + added_orchids
  let groups := final_orchids / orchid_ratio
  tulip_ratio * groups

/-- Proves that Lila will have 21 tulips after maintaining the ratio --/
theorem lila_tulips : 
  final_tulips 16 12 3 4 = 21 := by
  sorry

#eval final_tulips 16 12 3 4

end lila_tulips_l2962_296276


namespace b_72_mod_50_l2962_296259

/-- The sequence b_n defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The theorem stating that b_72 is congruent to 2 modulo 50 -/
theorem b_72_mod_50 : b 72 ≡ 2 [ZMOD 50] := by
  sorry

end b_72_mod_50_l2962_296259


namespace matt_card_trade_profit_l2962_296277

def matt_card_value : ℕ := 6
def jane_card1_value : ℕ := 2
def jane_card2_value : ℕ := 9
def matt_cards_traded : ℕ := 2
def jane_cards1_received : ℕ := 3
def jane_cards2_received : ℕ := 1
def profit : ℕ := 3

theorem matt_card_trade_profit :
  (jane_cards1_received * jane_card1_value + jane_cards2_received * jane_card2_value) -
  (matt_cards_traded * matt_card_value) = profit := by
  sorry

end matt_card_trade_profit_l2962_296277


namespace three_quadrilaterals_with_circumcenter_l2962_296202

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A point is equidistant from all vertices of a quadrilateral. -/
def has_circumcenter (q : Quadrilateral) : Prop :=
  ∃ (c : ℝ × ℝ), ∀ (i : Fin 4), dist c (q.vertices i) = dist c (q.vertices 0)

/-- A kite is a quadrilateral with two pairs of adjacent congruent sides. -/
def is_kite (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral has exactly two right angles. -/
def has_two_right_angles (q : Quadrilateral) : Prop := sorry

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- A rhombus is a quadrilateral with all sides equal. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- An equilateral trapezoid is a trapezoid with the non-parallel sides equal. -/
def is_equilateral_trapezoid (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral can be inscribed in a circle. -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- The main theorem stating that exactly 3 types of the given quadrilaterals have a circumcenter. -/
theorem three_quadrilaterals_with_circumcenter : 
  ∃ (a b c : Quadrilateral),
    (is_kite a ∧ has_two_right_angles a ∧ has_circumcenter a) ∧
    (is_square b ∧ has_circumcenter b) ∧
    (is_equilateral_trapezoid c ∧ is_cyclic c ∧ has_circumcenter c) ∧
    (∀ (d : Quadrilateral), 
      (is_rhombus d ∧ ¬is_square d) → ¬has_circumcenter d) ∧
    (∀ (e : Quadrilateral),
      has_circumcenter e → 
      (e = a ∨ e = b ∨ e = c)) :=
sorry

end three_quadrilaterals_with_circumcenter_l2962_296202


namespace student_uniform_cost_l2962_296261

/-- Calculates the total cost for a student's uniforms including discounts, fees, and taxes -/
def uniform_cost (num_uniforms : ℕ) (pants_cost : ℚ) (socks_cost : ℚ) (shoes_cost : ℚ) 
  (uniform_fee : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let jacket_cost := 3 * shirt_cost
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost + jacket_cost + shoes_cost
  let subtotal := num_uniforms * uniform_cost * (1 - discount_rate) + uniform_fee
  subtotal * (1 + tax_rate)

/-- The total cost for a student buying 5 uniforms is $1117.77 -/
theorem student_uniform_cost : 
  uniform_cost 5 20 3 40 15 (10/100) (6/100) = 1117.77 := by
  sorry

end student_uniform_cost_l2962_296261


namespace sum_342_78_base5_l2962_296220

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list representing a number in base 5 to a natural number in base 10 -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_342_78_base5 :
  addBase5 (toBase5 342) (toBase5 78) = [3, 1, 4, 0] :=
sorry

end sum_342_78_base5_l2962_296220


namespace percent_decrease_proof_l2962_296262

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 := by
  sorry

end percent_decrease_proof_l2962_296262


namespace range_of_a_l2962_296248

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def proposition_p (a : ℝ) : Prop :=
  is_monotonically_increasing (fun x => a^x)

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ¬(¬proposition_p a ∧ ¬proposition_q a))
  (h3 : proposition_p a ∨ proposition_q a) :
  a ∈ Set.Ioo 0 1 ∪ Set.Ici 4 := by
  sorry

end range_of_a_l2962_296248


namespace contrapositive_square_inequality_l2962_296260

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x > y) → ¬(x^2 > y^2)) ↔ (x ≤ y → x^2 ≤ y^2) := by
  sorry

end contrapositive_square_inequality_l2962_296260


namespace fourth_root_81_times_cube_root_27_times_sqrt_16_l2962_296292

theorem fourth_root_81_times_cube_root_27_times_sqrt_16 : 
  (81 : ℝ) ^ (1/4 : ℝ) * (27 : ℝ) ^ (1/3 : ℝ) * (16 : ℝ) ^ (1/2 : ℝ) = 36 := by
  sorry

end fourth_root_81_times_cube_root_27_times_sqrt_16_l2962_296292


namespace exists_fraction_with_99th_digit_4_l2962_296244

/-- Represents a decimal expansion as a sequence of digits -/
def DecimalExpansion := ℕ → Fin 10

/-- Returns the nth digit after the decimal point in the decimal expansion of a rational number -/
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : Fin 10 := sorry

/-- The decimal expansion of 3/11 -/
def threeElevenths : DecimalExpansion := 
  fun n => if n % 2 = 0 then 2 else 7

theorem exists_fraction_with_99th_digit_4 : 
  ∃ q : ℚ, nthDigitAfterDecimal (q + 3/11) 99 = 4 := by sorry

end exists_fraction_with_99th_digit_4_l2962_296244


namespace nonadjacent_arrangements_correct_nonadjacent_arrangements_simplified_l2962_296250

/-- The number of circular arrangements of n people where two specific people are not adjacent -/
def nonadjacent_arrangements (n : ℕ) : ℕ :=
  (n - 3) * (n - 2).factorial

/-- Theorem stating the number of circular arrangements of n people (n ≥ 3) 
    where two specific people are not adjacent -/
theorem nonadjacent_arrangements_correct (n : ℕ) (h : n ≥ 3) :
  nonadjacent_arrangements n = (n - 1).factorial - 2 * (n - 2).factorial :=
by
  sorry

/-- Corollary: The number of arrangements where two specific people are not adjacent
    is equal to (n-3)(n-2)! -/
theorem nonadjacent_arrangements_simplified (n : ℕ) (h : n ≥ 3) :
  nonadjacent_arrangements n = (n - 3) * (n - 2).factorial :=
by
  sorry

end nonadjacent_arrangements_correct_nonadjacent_arrangements_simplified_l2962_296250


namespace intersection_A_B_l2962_296206

-- Define set A
def A : Set ℝ := {x : ℝ | x * (x - 4) < 0}

-- Define set B
def B : Set ℝ := {0, 1, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l2962_296206


namespace quadratic_roots_sum_squares_l2962_296204

/-- Given that x₁ and x₂ are real roots of a quadratic equation, this theorem proves
    properties about y = x₁² + x₂² as a function of m. -/
theorem quadratic_roots_sum_squares (m : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - 2*(m-1)*x₁ + m + 1 = 0 →
  x₂^2 - 2*(m-1)*x₂ + m + 1 = 0 →
  let y := x₁^2 + x₂^2
  -- 1. y as a function of m
  y = 4*m^2 - 10*m + 2 ∧
  -- 2. Minimum value of y
  (∃ (m₀ : ℝ), y = 6 ∧ ∀ (m' : ℝ), y ≥ 6) ∧
  -- 3. y ≥ 6 for all valid m
  y ≥ 6 := by
sorry

end quadratic_roots_sum_squares_l2962_296204


namespace odd_power_sum_is_prime_power_l2962_296282

theorem odd_power_sum_is_prime_power (n p x y k : ℕ) :
  Odd n →
  n > 1 →
  Prime p →
  Odd p →
  x^n + y^n = p^k →
  ∃ m : ℕ, n = p^m :=
by sorry

end odd_power_sum_is_prime_power_l2962_296282


namespace factorization_equality_l2962_296297

theorem factorization_equality (x : ℝ) : 8*x - 2*x^2 = 2*x*(4 - x) := by
  sorry

end factorization_equality_l2962_296297


namespace group_dinner_cost_l2962_296272

/-- Calculate the total cost for a group dinner including service charge -/
theorem group_dinner_cost (num_people : ℕ) (meal_cost drink_cost dessert_cost : ℚ) 
  (service_charge_rate : ℚ) : 
  num_people = 5 →
  meal_cost = 12 →
  drink_cost = 3 →
  dessert_cost = 5 →
  service_charge_rate = 1/10 →
  (num_people * (meal_cost + drink_cost + dessert_cost)) * (1 + service_charge_rate) = 110 := by
  sorry

end group_dinner_cost_l2962_296272


namespace xyz_equality_l2962_296288

theorem xyz_equality (x y z : ℝ) (h : x * y * z = x + y + z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by sorry

end xyz_equality_l2962_296288


namespace ratio_sum_squares_to_sum_l2962_296283

theorem ratio_sum_squares_to_sum (a b c : ℝ) : 
  (b = 2 * a) → 
  (c = 4 * a) → 
  (a^2 + b^2 + c^2 = 1701) → 
  (a + b + c = 63) := by
sorry

end ratio_sum_squares_to_sum_l2962_296283


namespace julies_savings_l2962_296255

/-- Represents the initial savings amount in each account -/
def P : ℝ := sorry

/-- Represents the annual interest rate (as a decimal) -/
def r : ℝ := sorry

/-- Theorem stating that given the conditions, Julie's initial total savings was $1000 -/
theorem julies_savings : 
  (P * r * 2 = 100) →  -- Simple interest earned after 2 years
  (P * ((1 + r)^2 - 1) = 105) →  -- Compound interest earned after 2 years
  (2 * P = 1000) :=  -- Total initial savings
by sorry

end julies_savings_l2962_296255


namespace race_probability_l2962_296281

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) :
  total_cars = 15 →
  prob_x = 1 / 4 →
  prob_y = 1 / 8 →
  prob_z = 1 / 12 →
  (prob_x + prob_y + prob_z : ℚ) = 11 / 24 :=
by sorry

end race_probability_l2962_296281


namespace complex_number_in_first_quadrant_l2962_296246

/-- The complex number z = 2i / (1 + i) has both positive real and imaginary parts -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 * Complex.I) / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im := by sorry

end complex_number_in_first_quadrant_l2962_296246


namespace latticePoindsInsideTriangleABO_l2962_296200

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0, 30)
def B : ℤ × ℤ := (20, 10)
def O : ℤ × ℤ := (0, 0)

-- Define a function to calculate the area of a triangle
def triangleArea (p1 p2 p3 : ℤ × ℤ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Define Pick's theorem
def picksTheorem (S : ℚ) (N L : ℤ) : Prop :=
  S = N + L / 2 - 1

-- State the theorem
theorem latticePoindsInsideTriangleABO :
  ∃ (N L : ℤ),
    picksTheorem (triangleArea A B O) N L ∧
    L = 60 ∧
    N = 271 :=
  sorry

end latticePoindsInsideTriangleABO_l2962_296200


namespace short_trees_after_planting_l2962_296224

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem stating that the total number of short trees after planting is 98 -/
theorem short_trees_after_planting :
  total_short_trees 41 57 = 98 := by
  sorry

end short_trees_after_planting_l2962_296224


namespace inverse_composition_equals_neg_sixteen_ninths_l2962_296287

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x - 7) / 3

-- Theorem statement
theorem inverse_composition_equals_neg_sixteen_ninths :
  g_inv (g_inv 12) = -16/9 :=
by sorry

end inverse_composition_equals_neg_sixteen_ninths_l2962_296287
