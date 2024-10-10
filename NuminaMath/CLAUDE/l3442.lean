import Mathlib

namespace reach_probability_l3442_344297

-- Define the type for a point in the coordinate plane
structure Point where
  x : Int
  y : Int

-- Define the type for a step direction
inductive Direction
  | Left
  | Right
  | Up
  | Down

-- Define the function to calculate the probability
def probability_reach_target (start : Point) (target : Point) (max_steps : Nat) : Rat :=
  sorry

-- Theorem statement
theorem reach_probability :
  probability_reach_target ⟨0, 0⟩ ⟨2, 3⟩ 7 = 179 / 8192 := by sorry

end reach_probability_l3442_344297


namespace range_of_m_l3442_344253

/-- Represents the condition for proposition p -/
def is_hyperbola_y_axis (m : ℝ) : Prop :=
  (2 - m < 0) ∧ (m - 1 > 0)

/-- Represents the condition for proposition q -/
def has_no_real_roots (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) 
  (h_p_or_q : is_hyperbola_y_axis m ∨ has_no_real_roots m)
  (h_not_q : ¬has_no_real_roots m) :
  m ≥ 3 :=
sorry

end range_of_m_l3442_344253


namespace ariella_savings_after_two_years_l3442_344214

/-- Calculates the final amount in a savings account after simple interest is applied. -/
def final_amount (initial_amount : ℝ) (interest_rate : ℝ) (years : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate * years)

/-- Proves that Ariella will have $720 after two years given the problem conditions. -/
theorem ariella_savings_after_two_years 
  (daniella_amount : ℝ)
  (ariella_excess : ℝ)
  (interest_rate : ℝ)
  (years : ℝ)
  (h1 : daniella_amount = 400)
  (h2 : ariella_excess = 200)
  (h3 : interest_rate = 0.1)
  (h4 : years = 2) :
  final_amount (daniella_amount + ariella_excess) interest_rate years = 720 :=
by
  sorry

#check ariella_savings_after_two_years

end ariella_savings_after_two_years_l3442_344214


namespace villa_tournament_correct_l3442_344261

/-- A tournament where each player plays with a fixed number of other players. -/
structure Tournament where
  num_players : ℕ
  games_per_player : ℕ
  total_games : ℕ

/-- The specific tournament described in the problem. -/
def villa_tournament : Tournament :=
  { num_players := 6,
    games_per_player := 4,
    total_games := 10 }

/-- Theorem stating that the total number of games in the Villa tournament is correct. -/
theorem villa_tournament_correct :
  villa_tournament.total_games = (villa_tournament.num_players * villa_tournament.games_per_player) / 2 :=
by sorry

end villa_tournament_correct_l3442_344261


namespace mean_temperature_l3442_344247

def temperatures : List ℝ := [82, 83, 78, 86, 88, 90, 88]

theorem mean_temperature : 
  (List.sum temperatures) / temperatures.length = 84.5714 := by
  sorry

end mean_temperature_l3442_344247


namespace square_area_from_diagonal_l3442_344218

/-- The area of a square with a diagonal of 10 meters is 50 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  let s := d / Real.sqrt 2
  s ^ 2 = 50 := by sorry

end square_area_from_diagonal_l3442_344218


namespace parabola_directrix_l3442_344203

/-- Given a parabola with equation y = 2x^2, its directrix equation is y = -1/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = 2 * x^2) → (∃ (k : ℝ), k = -1/8 ∧ k = y) :=
by sorry

end parabola_directrix_l3442_344203


namespace no_prime_sum_53_l3442_344215

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem statement
theorem no_prime_sum_53 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 := by
  sorry

end no_prime_sum_53_l3442_344215


namespace rectangle_count_l3442_344287

theorem rectangle_count (a : ℝ) (ha : a > 0) : 
  ∃! (x y : ℝ), x < 2*a ∧ y < 2*a ∧ 
  2*(x + y) = 2*((2*a + 3*a) * (2/3)) ∧ 
  x*y = (2*a * 3*a) * (2/9) := by
  sorry

end rectangle_count_l3442_344287


namespace sin_cos_symmetry_l3442_344242

open Real

theorem sin_cos_symmetry :
  ∃ (k : ℤ), (∀ x : ℝ, sin (2 * x - π / 6) = sin (π / 2 - (2 * x - π / 6))) ∧
             (∀ x : ℝ, cos (x - π / 3) = cos (π - (x - π / 3))) ∧
  ¬ ∃ (c : ℝ), (∀ x : ℝ, sin (2 * (x + c) - π / 6) = -sin (2 * (x - c) - π / 6)) ∧
                (∀ x : ℝ, cos ((x + c) - π / 3) = cos ((x - c) - π / 3)) :=
by sorry

end sin_cos_symmetry_l3442_344242


namespace stock_price_is_102_l3442_344263

/-- Given an income, dividend rate, and investment amount, calculate the price of a stock. -/
def stock_price (income : ℚ) (dividend_rate : ℚ) (investment : ℚ) : ℚ :=
  let face_value := income / dividend_rate
  (investment / face_value) * 100

/-- Theorem stating that given the specific conditions, the stock price is 102. -/
theorem stock_price_is_102 :
  stock_price 900 (20 / 100) 4590 = 102 := by
  sorry

#eval stock_price 900 (20 / 100) 4590

end stock_price_is_102_l3442_344263


namespace product_of_four_integers_l3442_344227

theorem product_of_four_integers (P Q R S : ℕ+) : 
  (P : ℚ) + (Q : ℚ) + (R : ℚ) + (S : ℚ) = 50 →
  (P : ℚ) + 4 = (Q : ℚ) - 4 ∧ 
  (P : ℚ) + 4 = (R : ℚ) * 3 ∧ 
  (P : ℚ) + 4 = (S : ℚ) / 3 →
  (P : ℚ) * (Q : ℚ) * (R : ℚ) * (S : ℚ) = (43 * 107 * 75 * 225) / 1536 := by
  sorry

#check product_of_four_integers

end product_of_four_integers_l3442_344227


namespace min_value_f_l3442_344258

/-- The function f(x) = 12x - x³ -/
def f (x : ℝ) : ℝ := 12 * x - x^3

/-- The theorem stating that the minimum value of f(x) on [-3, 3] is -16 -/
theorem min_value_f : 
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-3) 3 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-3) 3 → f x ≥ f x₀) ∧
  f x₀ = -16 := by
  sorry


end min_value_f_l3442_344258


namespace line_x_intercept_l3442_344288

/-- Given a line passing through points (10, 3) and (-4, -4), 
    prove that its x-intercept is 4 -/
theorem line_x_intercept : 
  let p1 : ℝ × ℝ := (10, 3)
  let p2 : ℝ × ℝ := (-4, -4)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  (0 : ℝ) = m * 4 + b :=
by sorry

end line_x_intercept_l3442_344288


namespace optimal_purchase_plan_l3442_344237

/-- Represents the daily carrying capacity and cost of robots --/
structure Robot where
  capacity : ℕ  -- daily carrying capacity in tons
  cost : ℕ      -- cost in yuan

/-- Represents the purchase plan for robots --/
structure PurchasePlan where
  typeA : ℕ  -- number of type A robots
  typeB : ℕ  -- number of type B robots

/-- Calculates the total daily carrying capacity for a given purchase plan --/
def totalCapacity (a b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.typeA * a.capacity + plan.typeB * b.capacity

/-- Calculates the total cost for a given purchase plan --/
def totalCost (a b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.typeA * a.cost + plan.typeB * b.cost

/-- Theorem stating the optimal purchase plan --/
theorem optimal_purchase_plan (a b : Robot) :
  a.capacity = b.capacity + 20 →
  3 * a.capacity + 2 * b.capacity = 460 →
  a.cost = 30000 →
  b.cost = 20000 →
  (∀ plan : PurchasePlan, plan.typeA + plan.typeB = 20 →
    totalCapacity a b plan ≥ 1820 →
    totalCost a b plan ≥ 510000) ∧
  (∃ plan : PurchasePlan, plan.typeA = 11 ∧ plan.typeB = 9 ∧
    totalCapacity a b plan ≥ 1820 ∧
    totalCost a b plan = 510000) :=
by sorry

end optimal_purchase_plan_l3442_344237


namespace geometric_sequence_308th_term_l3442_344224

theorem geometric_sequence_308th_term
  (a₁ : ℝ)
  (a₂ : ℝ)
  (h₁ : a₁ = 10)
  (h₂ : a₂ = -10) :
  let r := a₂ / a₁
  let aₙ := a₁ * r^(308 - 1)
  aₙ = -10 :=
by sorry

end geometric_sequence_308th_term_l3442_344224


namespace reciprocal_and_absolute_value_l3442_344233

theorem reciprocal_and_absolute_value :
  (1 / (- (-2))) = 1/2 ∧ 
  {x : ℝ | |x| = 5} = {-5, 5} := by
  sorry

end reciprocal_and_absolute_value_l3442_344233


namespace cheryl_material_problem_l3442_344207

theorem cheryl_material_problem (x : ℚ) : 
  (x + 2/3 : ℚ) - 8/18 = 2/3 → x = 4/9 := by sorry

end cheryl_material_problem_l3442_344207


namespace absolute_difference_of_roots_l3442_344264

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := x^2 - (k+3)*x + k

-- Define the roots of the quadratic equation
def roots (k : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem absolute_difference_of_roots (k : ℝ) :
  let (r₁, r₂) := roots k
  |r₁ - r₂| = Real.sqrt (k^2 + 2*k + 9) := by sorry

end absolute_difference_of_roots_l3442_344264


namespace shortest_rope_length_l3442_344201

theorem shortest_rope_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (ratio : b = (5/4) * a ∧ c = (6/4) * a) (sum_condition : a + c = b + 100) : 
  a = 80 := by
sorry

end shortest_rope_length_l3442_344201


namespace evaluate_expression_l3442_344277

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) : y * (2 * y - x) = 24 := by
  sorry

end evaluate_expression_l3442_344277


namespace rectangle_width_is_five_l3442_344219

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Theorem: A rectangle with length 6 and perimeter 22 has width 5 --/
theorem rectangle_width_is_five :
  ∀ r : Rectangle, r.length = 6 → perimeter r = 22 → r.width = 5 := by
  sorry

end rectangle_width_is_five_l3442_344219


namespace fireworks_per_letter_l3442_344252

/-- The number of fireworks needed to display a digit --/
def fireworks_per_digit : ℕ := 6

/-- The number of digits in the year display --/
def year_digits : ℕ := 4

/-- The number of letters in "HAPPY NEW YEAR" --/
def phrase_letters : ℕ := 12

/-- The number of additional boxes of fireworks --/
def additional_boxes : ℕ := 50

/-- The number of fireworks in each box --/
def fireworks_per_box : ℕ := 8

/-- The total number of fireworks lit during the display --/
def total_fireworks : ℕ := 484

/-- Theorem: The number of fireworks needed to display a letter is 5 --/
theorem fireworks_per_letter :
  ∃ (x : ℕ), 
    x * phrase_letters + 
    fireworks_per_digit * year_digits + 
    additional_boxes * fireworks_per_box = 
    total_fireworks ∧ x = 5 := by
  sorry

end fireworks_per_letter_l3442_344252


namespace acacia_arrangement_probability_l3442_344295

/-- The number of fir trees -/
def num_fir : ℕ := 4

/-- The number of pine trees -/
def num_pine : ℕ := 5

/-- The number of acacia trees -/
def num_acacia : ℕ := 6

/-- The total number of trees -/
def total_trees : ℕ := num_fir + num_pine + num_acacia

/-- The probability of no two acacia trees being next to each other -/
def prob_no_adjacent_acacia : ℚ := 84 / 159

theorem acacia_arrangement_probability :
  let total_arrangements := Nat.choose total_trees num_acacia
  let valid_arrangements := Nat.choose (num_fir + num_pine + 1) num_acacia * Nat.choose (num_fir + num_pine) num_fir
  (valid_arrangements : ℚ) / total_arrangements = prob_no_adjacent_acacia := by
  sorry

end acacia_arrangement_probability_l3442_344295


namespace hexagon_diagonals_l3442_344299

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end hexagon_diagonals_l3442_344299


namespace larger_number_proof_l3442_344289

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1515) (h3 : L = 16 * S + 15) : L = 1617 := by
  sorry

end larger_number_proof_l3442_344289


namespace y1_greater_than_y2_l3442_344217

/-- A linear function f(x) = -3x + 1 -/
def f (x : ℝ) : ℝ := -3 * x + 1

theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : f 2 = y1) 
  (h2 : f 3 = y2) : 
  y1 > y2 := by
  sorry

end y1_greater_than_y2_l3442_344217


namespace decimal_places_product_specific_case_l3442_344293

/-- Given two real numbers a and b, this function returns the number of decimal places in their product. -/
def decimal_places_in_product (a b : ℝ) : ℕ :=
  sorry

/-- This function returns the number of decimal places in a real number. -/
def count_decimal_places (x : ℝ) : ℕ :=
  sorry

theorem decimal_places_product (a b : ℝ) :
  decimal_places_in_product a b = count_decimal_places a + count_decimal_places b :=
sorry

theorem specific_case : 
  decimal_places_in_product 0.38 0.26 = 4 :=
sorry

end decimal_places_product_specific_case_l3442_344293


namespace intersection_and_complement_union_condition_implies_m_range_l3442_344290

-- Define the sets
def U : Set ℝ := {x | 1 < x ∧ x < 7}
def A1 : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B1 : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

def A2 : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
def B2 (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- Theorem for the first part
theorem intersection_and_complement :
  (A1 ∩ B1 = {x | 3 ≤ x ∧ x < 5}) ∧
  (U \ A1 = {x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7)}) :=
sorry

-- Theorem for the second part
theorem union_condition_implies_m_range :
  ∀ m, (A2 ∪ B2 m = A2) → m ≤ 4 :=
sorry

end intersection_and_complement_union_condition_implies_m_range_l3442_344290


namespace simplify_fraction_l3442_344225

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l3442_344225


namespace hyperbola_satisfies_conditions_l3442_344273

/-- A hyperbola is defined by its equation and properties -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  passes_through : ℝ × ℝ
  asymptotes : ℝ → ℝ → Prop

/-- The given hyperbola with equation x²/2 - y² = 1 -/
def given_hyperbola : Hyperbola where
  equation := fun x y => x^2 / 2 - y^2 = 1
  passes_through := (2, -2)
  asymptotes := fun x y => x^2 / 2 - y^2 = 0

/-- The hyperbola we need to prove -/
def our_hyperbola : Hyperbola where
  equation := fun x y => y^2 / 2 - x^2 / 4 = 1
  passes_through := (2, -2)
  asymptotes := fun x y => x^2 / 2 - y^2 = 0

/-- Theorem stating that our_hyperbola satisfies the required conditions -/
theorem hyperbola_satisfies_conditions :
  (our_hyperbola.equation our_hyperbola.passes_through.1 our_hyperbola.passes_through.2) ∧
  (∀ x y, our_hyperbola.asymptotes x y ↔ given_hyperbola.asymptotes x y) :=
sorry

end hyperbola_satisfies_conditions_l3442_344273


namespace lighter_ball_problem_l3442_344254

/-- Represents the maximum number of balls that can be checked in a given number of weighings -/
def max_balls (weighings : ℕ) : ℕ := 3^weighings

/-- The problem statement -/
theorem lighter_ball_problem (n : ℕ) :
  (∀ m : ℕ, m > n → max_balls 5 < m) →
  (∃ strategy : Unit, true) →  -- placeholder for the existence of a strategy
  n ≤ max_balls 5 :=
sorry

end lighter_ball_problem_l3442_344254


namespace wall_width_calculation_l3442_344200

/-- Given a wall with specific proportions and volume, calculate its width -/
theorem wall_width_calculation (w h l : ℝ) (V : ℝ) (h_def : h = 6 * w) (l_def : l = 7 * h^2) (V_def : V = w * h * l) (V_val : V = 86436) :
  w = (86436 / 1512) ^ (1/4) := by
sorry

end wall_width_calculation_l3442_344200


namespace smallest_valid_fourth_number_l3442_344204

def is_valid_fourth_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (6 + 8 + 2 + 4 + 8 + 5 + (n / 10) + (n % 10)) * 4 = 68 + 24 + 85 + n

theorem smallest_valid_fourth_number :
  ∀ m : ℕ, m ≥ 10 ∧ m < 57 → ¬(is_valid_fourth_number m) ∧ is_valid_fourth_number 57 := by
  sorry

end smallest_valid_fourth_number_l3442_344204


namespace min_rectangle_side_l3442_344240

/-- Given a rectangle with one side of length 1, divided into four smaller rectangles
    by two perpendicular lines, where three of the smaller rectangles have areas of
    at least 1 and the fourth has an area of at least 2, the minimum length of the
    other side of the original rectangle is 3 + 2√2. -/
theorem min_rectangle_side (a b c d : ℝ) : 
  a + b = 1 →
  a * c ≥ 1 →
  a * d ≥ 1 →
  b * c ≥ 1 →
  b * d ≥ 2 →
  c + d ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_rectangle_side_l3442_344240


namespace bianca_birthday_money_l3442_344280

/-- The amount of money Bianca received for her birthday -/
def birthday_money (num_friends : ℕ) (amount_per_friend : ℕ) : ℕ :=
  num_friends * amount_per_friend

/-- Theorem: Bianca received 120 dollars for her birthday -/
theorem bianca_birthday_money :
  birthday_money 8 15 = 120 := by
  sorry

end bianca_birthday_money_l3442_344280


namespace max_volume_difference_l3442_344262

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The measured dimensions of the box -/
def measuredDimensions : BoxDimensions :=
  { length := 150, width := 150, height := 225 }

/-- The maximum error in each measurement -/
def maxError : ℝ := 1

/-- Theorem: The maximum possible difference between the actual capacity
    and the computed capacity of the box is 90726 cubic centimeters -/
theorem max_volume_difference :
  ∃ (actualDimensions : BoxDimensions),
    actualDimensions.length ≤ measuredDimensions.length + maxError ∧
    actualDimensions.length ≥ measuredDimensions.length - maxError ∧
    actualDimensions.width ≤ measuredDimensions.width + maxError ∧
    actualDimensions.width ≥ measuredDimensions.width - maxError ∧
    actualDimensions.height ≤ measuredDimensions.height + maxError ∧
    actualDimensions.height ≥ measuredDimensions.height - maxError ∧
    (boxVolume actualDimensions - boxVolume measuredDimensions) ≤ 90726 ∧
    ∀ (d : BoxDimensions),
      d.length ≤ measuredDimensions.length + maxError →
      d.length ≥ measuredDimensions.length - maxError →
      d.width ≤ measuredDimensions.width + maxError →
      d.width ≥ measuredDimensions.width - maxError →
      d.height ≤ measuredDimensions.height + maxError →
      d.height ≥ measuredDimensions.height - maxError →
      (boxVolume d - boxVolume measuredDimensions) ≤ 90726 :=
by sorry

end max_volume_difference_l3442_344262


namespace circle_parabola_intersection_l3442_344249

/-- The number of intersection points between a circle and a parabola -/
def intersection_count (b : ℝ) : ℕ :=
  sorry

/-- The curves x^2 + y^2 = b^2 and y = x^2 - b + 1 intersect at exactly 4 points
    if and only if b > 2 -/
theorem circle_parabola_intersection (b : ℝ) :
  intersection_count b = 4 ↔ b > 2 := by
  sorry

end circle_parabola_intersection_l3442_344249


namespace coeff_4th_term_of_1_minus_2x_to_15_l3442_344238

/-- The coefficient of the 4th term in the expansion of (1-2x)^15 -/
def coeff_4th_term : ℤ := -3640

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_4th_term_of_1_minus_2x_to_15 :
  coeff_4th_term = (-2)^3 * (binomial 15 3) := by sorry

end coeff_4th_term_of_1_minus_2x_to_15_l3442_344238


namespace largest_quantity_l3442_344202

theorem largest_quantity : 
  (2008 / 2007 + 2008 / 2009 : ℚ) > (2009 / 2008 + 2009 / 2010 : ℚ) ∧ 
  (2009 / 2008 + 2009 / 2010 : ℚ) > (2008 / 2009 + 2010 / 2009 : ℚ) := by
  sorry

end largest_quantity_l3442_344202


namespace min_set_size_with_mean_constraints_l3442_344271

theorem min_set_size_with_mean_constraints (n : ℕ) (S : Finset ℕ) : 
  n > 0 ∧ 
  S.card = n ∧ 
  (∃ m L P : ℕ, 
    L ∈ S ∧ 
    P ∈ S ∧ 
    (∀ x ∈ S, x ≤ L ∧ x ≥ P) ∧
    (S.sum id) / n = m ∧
    m = (2 * L) / 5 ∧ 
    m = (7 * P) / 4) →
  n ≥ 5 :=
by sorry

end min_set_size_with_mean_constraints_l3442_344271


namespace square_difference_equality_l3442_344265

theorem square_difference_equality : (43 + 15)^2 - (43^2 + 15^2) = 1290 := by
  sorry

end square_difference_equality_l3442_344265


namespace point_d_and_k_value_l3442_344282

/-- Given four points in a plane, prove the coordinates of D and the value of k. -/
theorem point_d_and_k_value 
  (A B C D : ℝ × ℝ)
  (hA : A = (1, 3))
  (hB : B = (2, -2))
  (hC : C = (4, 1))
  (h_AB_CD : B - A = D - C)
  (a b : ℝ × ℝ)
  (ha : a = B - A)
  (hb : b = C - B)
  (h_parallel : ∃ (t : ℝ), t ≠ 0 ∧ t • (k • a - b) = a + 3 • b) :
  D = (5, -4) ∧ k = -1/3 := by sorry

end point_d_and_k_value_l3442_344282


namespace mityas_age_l3442_344278

/-- Represents the ages of Mitya and Shura -/
structure Ages where
  mitya : ℕ
  shura : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.mitya = ages.shura + 11) ∧
  (ages.shura = 2 * (ages.shura - (ages.mitya - ages.shura)))

/-- The theorem stating Mitya's age -/
theorem mityas_age :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.mitya = 33 :=
sorry

end mityas_age_l3442_344278


namespace complex_equation_sum_l3442_344246

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2*i) * (2 + a*i) = b - 2*i → a + b = 8 :=
by sorry

end complex_equation_sum_l3442_344246


namespace investment_interest_proof_l3442_344228

/-- Calculates the total interest earned on an investment -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Proves that the total interest earned on $2,000 invested at 5% annually for 5 years is $552.56 -/
theorem investment_interest_proof :
  let principal := 2000
  let rate := 0.05
  let years := 5
  ∃ ε > 0, abs (totalInterestEarned principal rate years - 552.56) < ε :=
by sorry

end investment_interest_proof_l3442_344228


namespace arithmetic_equality_l3442_344211

theorem arithmetic_equality : 19 * 17 + 29 * 17 + 48 * 25 = 2016 := by
  sorry

end arithmetic_equality_l3442_344211


namespace fruit_combinations_l3442_344231

theorem fruit_combinations (n r : ℕ) (h1 : n = 5) (h2 : r = 2) :
  (n + r - 1).choose r = 15 := by
sorry

end fruit_combinations_l3442_344231


namespace sequence_sum_property_l3442_344209

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = 2 * a n - n) :
  2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) = 30 / 31 :=
by sorry

end sequence_sum_property_l3442_344209


namespace total_covered_area_l3442_344256

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular strip -/
def Strip.area (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of overlap between two strips -/
def overlap_area (width : ℝ) (overlap_length : ℝ) : ℝ := width * overlap_length

/-- Theorem: The total area covered by three intersecting strips -/
theorem total_covered_area (s : Strip) (overlap_length : ℝ) : 
  s.length = 12 → s.width = 2 → overlap_length = 2 →
  3 * s.area - 3 * overlap_area s.width overlap_length = 60 := by
  sorry

end total_covered_area_l3442_344256


namespace contrapositive_theorem_negation_theorem_l3442_344230

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem contrapositive_theorem (a b : ℝ) :
  (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M) :=
sorry

theorem negation_theorem :
  (∃ x : ℝ, x^2 - x - 1 > 0) ↔ ¬(∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end contrapositive_theorem_negation_theorem_l3442_344230


namespace infinitely_many_n_squared_plus_one_divides_factorial_l3442_344298

/-- The set of positive integers n for which n^2 + 1 divides n! is infinite -/
theorem infinitely_many_n_squared_plus_one_divides_factorial :
  Set.Infinite {n : ℕ+ | (n^2 + 1) ∣ n!} := by sorry

end infinitely_many_n_squared_plus_one_divides_factorial_l3442_344298


namespace nail_polish_theorem_l3442_344268

def nail_polish_problem (kim heidi karen : ℕ) : Prop :=
  kim = 25 ∧
  heidi = kim + 8 ∧
  karen = kim - 6 ∧
  heidi + karen = 52

theorem nail_polish_theorem :
  ∃ (kim heidi karen : ℕ), nail_polish_problem kim heidi karen := by
  sorry

end nail_polish_theorem_l3442_344268


namespace final_payment_calculation_final_payment_is_861_90_l3442_344244

/-- Calculates the final payment amount for a product purchase given specific deposit and discount conditions --/
theorem final_payment_calculation (total_cost : ℝ) (first_deposit : ℝ) (second_deposit : ℝ) 
  (promotional_discount_rate : ℝ) (interest_rate : ℝ) : ℝ :=
  let remaining_balance_before_discount := total_cost - (first_deposit + second_deposit)
  let promotional_discount := total_cost * promotional_discount_rate
  let remaining_balance_after_discount := remaining_balance_before_discount - promotional_discount
  let interest := remaining_balance_after_discount * interest_rate
  remaining_balance_after_discount + interest

/-- Proves that the final payment amount is $861.90 given the specific conditions of the problem --/
theorem final_payment_is_861_90 : 
  let total_cost := 1300
  let first_deposit := 130
  let second_deposit := 260
  let promotional_discount_rate := 0.05
  let interest_rate := 0.02
  (final_payment_calculation total_cost first_deposit second_deposit promotional_discount_rate interest_rate) = 861.90 := by
  sorry

end final_payment_calculation_final_payment_is_861_90_l3442_344244


namespace valid_seating_arrangements_count_l3442_344234

/-- Represents a seating arrangement for two people -/
structure SeatingArrangement :=
  (front : Fin 4 → Bool)
  (back : Fin 5 → Bool)

/-- Checks if a seating arrangement is valid (two people not adjacent) -/
def is_valid (s : SeatingArrangement) : Bool :=
  sorry

/-- Counts the number of valid seating arrangements -/
def count_valid_arrangements : Nat :=
  sorry

/-- Theorem stating that the number of valid seating arrangements is 58 -/
theorem valid_seating_arrangements_count :
  count_valid_arrangements = 58 := by sorry

end valid_seating_arrangements_count_l3442_344234


namespace sales_profit_equation_max_profit_selling_price_range_l3442_344284

-- Define the cost to produce each item
def production_cost : ℝ := 50

-- Define the daily sales volume as a function of price
def sales_volume (x : ℝ) : ℝ := 50 + 5 * (100 - x)

-- Define the daily sales profit function
def sales_profit (x : ℝ) : ℝ := (x - production_cost) * sales_volume x

-- Theorem 1: The daily sales profit function
theorem sales_profit_equation (x : ℝ) :
  sales_profit x = -5 * x^2 + 800 * x - 27500 := by sorry

-- Theorem 2: The maximum daily sales profit
theorem max_profit :
  ∃ (x : ℝ), x = 80 ∧ sales_profit x = 4500 ∧
  ∀ (y : ℝ), 50 ≤ y ∧ y ≤ 100 → sales_profit y ≤ sales_profit x := by sorry

-- Theorem 3: The range of selling prices satisfying the conditions
theorem selling_price_range :
  ∀ (x : ℝ), (sales_profit x ≥ 4000 ∧ production_cost * sales_volume x ≤ 7000) ↔
  (82 ≤ x ∧ x ≤ 90) := by sorry

end sales_profit_equation_max_profit_selling_price_range_l3442_344284


namespace M_intersect_N_eq_N_l3442_344208

def M : Set Int := {-1, 0, 1}

def N : Set Int := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end M_intersect_N_eq_N_l3442_344208


namespace power_sum_difference_l3442_344210

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) = 58929 := by
  sorry

end power_sum_difference_l3442_344210


namespace treatment_volume_is_120_ml_l3442_344285

/-- Calculates the total volume of treatment received from a saline drip. -/
def total_treatment_volume (drops_per_minute : ℕ) (treatment_hours : ℕ) (ml_per_100_drops : ℕ) : ℕ :=
  let minutes_per_hour : ℕ := 60
  let drops_per_100 : ℕ := 100
  let total_minutes : ℕ := treatment_hours * minutes_per_hour
  let total_drops : ℕ := drops_per_minute * total_minutes
  (total_drops * ml_per_100_drops) / drops_per_100

/-- The theorem stating that the total treatment volume is 120 ml under given conditions. -/
theorem treatment_volume_is_120_ml :
  total_treatment_volume 20 2 5 = 120 :=
by
  sorry

#eval total_treatment_volume 20 2 5

end treatment_volume_is_120_ml_l3442_344285


namespace probability_divisor_of_12_on_8_sided_die_l3442_344291

def is_divisor_of_12 (n : ℕ) : Prop := 12 % n = 0

def die_sides : ℕ := 8

def favorable_outcomes : Finset ℕ := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12_on_8_sided_die :
  (favorable_outcomes.card : ℚ) / die_sides = 5 / 8 :=
sorry

end probability_divisor_of_12_on_8_sided_die_l3442_344291


namespace solution_exists_l3442_344241

def f (x : ℝ) := x^3 + x - 3

theorem solution_exists : ∃ c ∈ Set.Icc 1 2, f c = 0 := by
  sorry

end solution_exists_l3442_344241


namespace compute_expression_l3442_344236

theorem compute_expression : 3 * 3^4 - 9^27 / 9^25 = 162 := by sorry

end compute_expression_l3442_344236


namespace raines_change_l3442_344245

/-- Calculates the change Raine receives after purchasing items from a gift shop --/
theorem raines_change (bracelet_price necklace_price mug_price : ℕ)
  (bracelet_count necklace_count mug_count : ℕ)
  (payment : ℕ)
  (h1 : bracelet_price = 15)
  (h2 : necklace_price = 10)
  (h3 : mug_price = 20)
  (h4 : bracelet_count = 3)
  (h5 : necklace_count = 2)
  (h6 : mug_count = 1)
  (h7 : payment = 100) :
  payment - (bracelet_price * bracelet_count + necklace_price * necklace_count + mug_price * mug_count) = 15 := by
  sorry

#check raines_change

end raines_change_l3442_344245


namespace opposite_sides_inequality_l3442_344212

/-- Given two points P and A on opposite sides of a line, prove that P satisfies a specific inequality --/
theorem opposite_sides_inequality (x y : ℝ) :
  (3*x + 2*y - 8) * (3*1 + 2*2 - 8) < 0 →
  3*x + 2*y > 8 := by
  sorry

end opposite_sides_inequality_l3442_344212


namespace train_length_l3442_344243

/-- The length of a train given specific conditions -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 230 →
  passing_time = 35 →
  (train_speed - jogger_speed) * passing_time + initial_distance = 580 :=
by sorry

end train_length_l3442_344243


namespace total_amount_is_fifteen_l3442_344266

/-- Represents the share of each person in Rupees -/
structure Share where
  w : ℚ
  x : ℚ
  y : ℚ

/-- The total amount of the sum -/
def total_amount (s : Share) : ℚ :=
  s.w + s.x + s.y

/-- The condition that for each rupee w gets, x gets 30 paisa and y gets 20 paisa -/
def share_ratio (s : Share) : Prop :=
  s.x = (3/10) * s.w ∧ s.y = (1/5) * s.w

/-- The theorem stating that if w's share is 10 rupees and the share ratio is maintained,
    then the total amount is 15 rupees -/
theorem total_amount_is_fifteen (s : Share) 
    (h1 : s.w = 10)
    (h2 : share_ratio s) : 
    total_amount s = 15 := by
  sorry


end total_amount_is_fifteen_l3442_344266


namespace equilateral_triangles_are_similar_l3442_344213

/-- An equilateral triangle is a triangle with all sides equal -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Similarity of two equilateral triangles -/
def are_similar (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.side = k * t1.side

/-- Theorem: Any two equilateral triangles are similar -/
theorem equilateral_triangles_are_similar (t1 t2 : EquilateralTriangle) :
  are_similar t1 t2 := by
  sorry


end equilateral_triangles_are_similar_l3442_344213


namespace basketball_conference_games_l3442_344269

/-- The number of divisions in the basketball conference -/
def num_divisions : ℕ := 3

/-- The number of teams in each division -/
def teams_per_division : ℕ := 4

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams from other divisions -/
def inter_division_games : ℕ := 2

/-- The total number of scheduled games in the basketball conference -/
def total_games : ℕ := 150

theorem basketball_conference_games :
  (num_divisions * (teams_per_division.choose 2) * intra_division_games) +
  (num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division * inter_division_games / 2) = total_games :=
by sorry

end basketball_conference_games_l3442_344269


namespace may_day_travel_scientific_notation_l3442_344276

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem may_day_travel_scientific_notation :
  scientific_notation (56.99 * 1000000) = (5.699, 7) :=
sorry

end may_day_travel_scientific_notation_l3442_344276


namespace system_solution_range_l3442_344250

theorem system_solution_range (a x y : ℝ) : 
  (5 * x + 2 * y = 11 * a + 18) →
  (2 * x - 3 * y = 12 * a - 8) →
  (x > 0) →
  (y > 0) →
  (-2/3 < a ∧ a < 2) :=
by sorry

end system_solution_range_l3442_344250


namespace task_completion_rate_l3442_344267

/-- Given two people A and B who can complete a task in x and y days respectively,
    this theorem proves that together they can complete a fraction of 1/x + 1/y of the task in one day. -/
theorem task_completion_rate (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 : ℝ) / x + (1 : ℝ) / y = (x + y) / (x * y) := by
  sorry


end task_completion_rate_l3442_344267


namespace polynomial_divisibility_l3442_344232

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + A*x^2 + B = 0) → 
  A + B = 2 := by
sorry

end polynomial_divisibility_l3442_344232


namespace tan_roots_and_angle_sum_cosine_product_l3442_344272

theorem tan_roots_and_angle_sum_cosine_product
  (α β : Real)
  (h1 : ∀ x, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ↔ x = Real.tan α ∨ x = Real.tan β)
  (h2 : α ∈ Set.Ioo (-π/2) (π/2))
  (h3 : β ∈ Set.Ioo (-π/2) (π/2)) :
  (α + β = -2*π/3) ∧ (Real.cos α * Real.cos β = 1/6) := by
sorry

end tan_roots_and_angle_sum_cosine_product_l3442_344272


namespace binomial_expansion_positive_integer_powers_l3442_344296

theorem binomial_expansion_positive_integer_powers (x : ℝ) : 
  (Finset.filter (fun r : ℕ => (10 - 3*r) / 2 > 0 ∧ (10 - 3*r) % 2 = 0) (Finset.range 11)).card = 2 :=
sorry

end binomial_expansion_positive_integer_powers_l3442_344296


namespace stratified_sampling_l3442_344286

theorem stratified_sampling (total_employees : ℕ) (total_sample : ℕ) (dept_employees : ℕ) :
  total_employees = 240 →
  total_sample = 20 →
  dept_employees = 60 →
  (dept_employees * total_sample) / total_employees = 5 := by
  sorry

end stratified_sampling_l3442_344286


namespace sqrt_inequality_l3442_344275

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) := by
  sorry

end sqrt_inequality_l3442_344275


namespace candy_boxes_total_l3442_344283

theorem candy_boxes_total (x y z : ℕ) : 
  x = y / 2 → 
  x + z = 24 → 
  y + z = 34 → 
  x + y + z = 44 :=
by
  sorry

end candy_boxes_total_l3442_344283


namespace arithmetic_sequence_sum_l3442_344248

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_list (L : List ℕ) : ℕ :=
  L.foldl (· + ·) 0

theorem arithmetic_sequence_sum : 
  2 * (sum_list (arithmetic_sequence 102 2 10)) = 2220 := by sorry

end arithmetic_sequence_sum_l3442_344248


namespace grace_lee_calculation_difference_l3442_344294

theorem grace_lee_calculation_difference : 
  (12 - (3 * 4 - 2)) - (12 - 3 * 4 - 2) = -32 := by
  sorry

end grace_lee_calculation_difference_l3442_344294


namespace product_range_difference_l3442_344216

theorem product_range_difference (f g : ℝ → ℝ) :
  (∀ x, -3 ≤ f x ∧ f x ≤ 9) →
  (∀ x, -1 ≤ g x ∧ g x ≤ 6) →
  (∃ a b, ∀ x, f x * g x ≤ a ∧ b ≤ f x * g x ∧ a - b = 72) :=
by sorry

end product_range_difference_l3442_344216


namespace inverse_matrices_sum_l3442_344239

open Matrix

theorem inverse_matrices_sum (x y z w p q r s : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x, 2, y; 3, 4, 5; z, 6, w]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![-7, p, -13; q, -15, r; 3, s, 6]
  A * B = 1 → x + y + z + w + p + q + r + s = -5.5 := by
sorry

end inverse_matrices_sum_l3442_344239


namespace sin_product_zero_l3442_344259

theorem sin_product_zero : Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (60 * π / 180) * Real.sin (84 * π / 180) = 0 := by
  sorry

end sin_product_zero_l3442_344259


namespace inscribed_rectangles_area_sum_l3442_344223

-- Define a rectangle
structure Rectangle where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

-- Define an inscribed rectangle
structure InscribedRectangle (R : Rectangle) where
  x : ℝ
  y : ℝ
  h_x_bounds : 0 ≤ x ∧ x ≤ R.a
  h_y_bounds : 0 ≤ y ∧ y ≤ R.b

-- Define the area of a rectangle
def area (R : Rectangle) : ℝ := R.a * R.b

-- Define the area of an inscribed rectangle
def inscribed_area (R : Rectangle) (IR : InscribedRectangle R) : ℝ :=
  IR.x * IR.y + (R.a - IR.x) * (R.b - IR.y)

-- Theorem statement
theorem inscribed_rectangles_area_sum (R : Rectangle) 
  (IR1 IR2 : InscribedRectangle R) (h : IR1.x = IR2.x) :
  inscribed_area R IR1 + inscribed_area R IR2 = area R := by
  sorry

end inscribed_rectangles_area_sum_l3442_344223


namespace expression_satisfies_equation_l3442_344279

theorem expression_satisfies_equation (x : ℝ) (E : ℝ → ℝ) : 
  x = 4 → (7 * E x = 21) → E = fun y ↦ y - 1 := by
  sorry

end expression_satisfies_equation_l3442_344279


namespace arcsin_equation_solution_l3442_344274

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = Real.sqrt 102 / 51 ∧ 
    Real.arcsin x + Real.arcsin (3 * x) = π / 4 ∧
    -1 < x ∧ x < 1 ∧ -1 < 3 * x ∧ 3 * x < 1 :=
by sorry

end arcsin_equation_solution_l3442_344274


namespace polynomial_simplification_l3442_344270

/-- The given polynomial is equal to its simplified form for all x. -/
theorem polynomial_simplification :
  ∀ x : ℝ, 3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 4*x^3 =
            -2*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end polynomial_simplification_l3442_344270


namespace unique_right_triangle_18_l3442_344292

/-- Represents a triple of positive integers (a, b, c) that form a right triangle with perimeter 18. -/
structure RightTriangle18 where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_triangle : a^2 + b^2 = c^2
  perimeter_18 : a + b + c = 18

/-- There exists exactly one right triangle with integer side lengths and perimeter 18. -/
theorem unique_right_triangle_18 : ∃! t : RightTriangle18, True := by sorry

end unique_right_triangle_18_l3442_344292


namespace derivative_at_zero_does_not_exist_l3442_344281

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else Real.sin x * Real.cos (5 / x)

-- State the theorem
theorem derivative_at_zero_does_not_exist :
  ¬ ∃ (L : ℝ), HasDerivAt f L 0 := by sorry

end derivative_at_zero_does_not_exist_l3442_344281


namespace product_equals_zero_l3442_344251

theorem product_equals_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := by
  sorry

end product_equals_zero_l3442_344251


namespace problem_proof_l3442_344257

def problem (aunt_gift : ℝ) : Prop :=
  let jade_initial : ℝ := 38
  let julia_initial : ℝ := jade_initial / 2
  let jade_final : ℝ := jade_initial + aunt_gift
  let julia_final : ℝ := julia_initial + aunt_gift
  let total : ℝ := jade_final + julia_final
  total = 57 + 2 * aunt_gift

theorem problem_proof (aunt_gift : ℝ) : problem aunt_gift :=
  sorry

end problem_proof_l3442_344257


namespace max_gcd_sum_1729_l3442_344222

theorem max_gcd_sum_1729 :
  ∃ (x y : ℕ+), x + y = 1729 ∧ 
  ∀ (a b : ℕ+), a + b = 1729 → Nat.gcd x y ≥ Nat.gcd a b ∧
  Nat.gcd x y = 247 :=
sorry

end max_gcd_sum_1729_l3442_344222


namespace polygon_interior_exterior_angles_equal_l3442_344229

theorem polygon_interior_exterior_angles_equal (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 6 := by
  sorry

end polygon_interior_exterior_angles_equal_l3442_344229


namespace f_minimum_l3442_344220

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 - 2*x*y + 6*y^2 - 14*x - 6*y + 72

/-- Theorem stating that f attains its minimum at (15/2, 1/2) -/
theorem f_minimum : 
  ∀ (x y : ℝ), f x y ≥ f (15/2) (1/2) := by sorry

end f_minimum_l3442_344220


namespace not_necessarily_linear_l3442_344255

open Set MeasureTheory

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- Define the Minkowski sum of graphs
def minkowskiSumGraphs (f g : RealFunction) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x + y, f x + g y)}

-- State the theorem
theorem not_necessarily_linear :
  ∃ (f g : RealFunction),
    Continuous f ∧
    (volume (minkowskiSumGraphs f g) = 0) ∧
    ¬∃ (a b : ℝ), ∀ x, f x = a * x + b :=
by sorry

end not_necessarily_linear_l3442_344255


namespace jim_sara_equal_savings_l3442_344205

/-- The number of weeks in the saving period -/
def weeks : ℕ := 820

/-- Sara's initial savings in dollars -/
def sara_initial : ℕ := 4100

/-- Sara's weekly savings in dollars -/
def sara_weekly : ℕ := 10

/-- Jim's weekly savings in dollars -/
def jim_weekly : ℕ := 15

/-- Total savings after the given period -/
def total_savings (initial weekly : ℕ) : ℕ :=
  initial + weekly * weeks

theorem jim_sara_equal_savings :
  total_savings 0 jim_weekly = total_savings sara_initial sara_weekly := by
  sorry

end jim_sara_equal_savings_l3442_344205


namespace probability_below_8_l3442_344206

theorem probability_below_8 (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) :
  1 - (p_10 + p_9 + p_8) = 0.29 := by
  sorry

end probability_below_8_l3442_344206


namespace first_divisor_l3442_344260

theorem first_divisor (k : ℕ) (h1 : k > 0) (h2 : k % 5 = 2) (h3 : k % 6 = 5) (h4 : k % 7 = 3) (h5 : k < 42) :
  min 5 (min 6 7) = 5 :=
by sorry

end first_divisor_l3442_344260


namespace increasing_function_range_l3442_344221

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (6/5 < a ∧ a < 6) :=
sorry

end increasing_function_range_l3442_344221


namespace negation_of_existential_proposition_l3442_344235

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℤ, x^2 + 2*x - 1 < 0) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) := by
  sorry

end negation_of_existential_proposition_l3442_344235


namespace arithmetic_sequence_middle_term_l3442_344226

theorem arithmetic_sequence_middle_term : ∀ (a b c : ℤ),
  (a = 2^2 ∧ c = 2^4 ∧ b - a = c - b) → b = 10 :=
by sorry

end arithmetic_sequence_middle_term_l3442_344226
