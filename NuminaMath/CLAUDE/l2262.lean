import Mathlib

namespace cubes_remaining_after_removal_l2262_226228

/-- Represents a cube arrangement --/
structure CubeArrangement where
  width : Nat
  height : Nat
  depth : Nat

/-- Calculates the total number of cubes in an arrangement --/
def totalCubes (arrangement : CubeArrangement) : Nat :=
  arrangement.width * arrangement.height * arrangement.depth

/-- Represents the number of vertical columns removed from the front --/
def removedColumns : Nat := 6

/-- Represents the height of each removed column --/
def removedColumnHeight : Nat := 3

/-- Calculates the number of remaining cubes after removal --/
def remainingCubes (arrangement : CubeArrangement) : Nat :=
  totalCubes arrangement - (removedColumns * removedColumnHeight)

/-- The theorem to be proved --/
theorem cubes_remaining_after_removal :
  let arrangement : CubeArrangement := { width := 4, height := 4, depth := 4 }
  remainingCubes arrangement = 46 := by
  sorry

end cubes_remaining_after_removal_l2262_226228


namespace bottle_caps_eaten_l2262_226259

theorem bottle_caps_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 34 → remaining = 26 → eaten = initial - remaining → eaten = 8 := by
sorry

end bottle_caps_eaten_l2262_226259


namespace multiply_subtract_distribute_computation_proof_l2262_226209

theorem multiply_subtract_distribute (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem computation_proof : 72 * 808 - 22 * 808 = 40400 := by sorry

end multiply_subtract_distribute_computation_proof_l2262_226209


namespace polynomial_simplification_l2262_226287

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 6) = s^2 - 4 * s + 3 := by
  sorry

end polynomial_simplification_l2262_226287


namespace event_probability_l2262_226225

-- Define the probability of the event occurring in a single trial
def p : ℝ := sorry

-- Define the probability of the event not occurring in a single trial
def q : ℝ := 1 - p

-- Define the number of trials
def n : ℕ := 3

-- State the theorem
theorem event_probability :
  (1 - q^n = 0.973) → p = 0.7 := by
  sorry

end event_probability_l2262_226225


namespace total_hours_played_l2262_226298

def football_minutes : ℕ := 60
def basketball_minutes : ℕ := 30

def total_minutes : ℕ := football_minutes + basketball_minutes

def minutes_per_hour : ℕ := 60

theorem total_hours_played :
  (total_minutes : ℚ) / minutes_per_hour = 1.5 := by
  sorry

end total_hours_played_l2262_226298


namespace remainder_of_large_number_l2262_226286

theorem remainder_of_large_number (N : ℕ) (h : N = 109876543210) :
  N % 180 = 10 := by
  sorry

end remainder_of_large_number_l2262_226286


namespace common_point_implies_c_equals_d_l2262_226201

/-- Given three linear functions with a common point, prove that c = d -/
theorem common_point_implies_c_equals_d 
  (a b c d : ℝ) 
  (h_neq : a ≠ b) 
  (h_common : ∃ (x y : ℝ), 
    y = a * x + a ∧ 
    y = b * x + b ∧ 
    y = c * x + d) : 
  c = d := by
sorry


end common_point_implies_c_equals_d_l2262_226201


namespace add_9876_seconds_to_2_45_pm_l2262_226293

/-- Represents a time of day in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Converts seconds to a Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let remainingSeconds := totalSeconds % 3600
  let minutes := remainingSeconds / 60
  let seconds := remainingSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds two Time structures -/
def addTime (t1 t2 : Time) : Time :=
  let totalSeconds := t1.hours * 3600 + t1.minutes * 60 + t1.seconds +
                      t2.hours * 3600 + t2.minutes * 60 + t2.seconds
  secondsToTime totalSeconds

/-- The main theorem to prove -/
theorem add_9876_seconds_to_2_45_pm (startTime : Time) 
  (h1 : startTime.hours = 14) 
  (h2 : startTime.minutes = 45) 
  (h3 : startTime.seconds = 0) : 
  addTime startTime (secondsToTime 9876) = { hours := 17, minutes := 29, seconds := 36 } := by
  sorry

end add_9876_seconds_to_2_45_pm_l2262_226293


namespace solve_linear_equation_l2262_226292

theorem solve_linear_equation :
  ∃ y : ℝ, (7 * y - 10 = 4 * y + 5) ∧ y = 5 := by
  sorry

end solve_linear_equation_l2262_226292


namespace lizzies_group_difference_l2262_226246

theorem lizzies_group_difference (total : ℕ) (lizzies_group : ℕ) : 
  total = 91 → lizzies_group = 54 → lizzies_group > (total - lizzies_group) → 
  lizzies_group - (total - lizzies_group) = 17 := by
sorry

end lizzies_group_difference_l2262_226246


namespace one_root_implies_a_range_l2262_226247

-- Define the function f(x) = 2x³ - 3x² + a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- Define the property of having only one root in [-2, 2]
def has_one_root_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc (-2) 2 ∧ f a x = 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 1 28

-- State the theorem
theorem one_root_implies_a_range :
  ∀ a : ℝ, has_one_root_in_interval a → a_range a := by
  sorry

end one_root_implies_a_range_l2262_226247


namespace tiffany_bag_difference_l2262_226255

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 8

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 7

/-- The difference between the number of bags on Monday and the next day -/
def bag_difference : ℕ := monday_bags - next_day_bags

theorem tiffany_bag_difference : bag_difference = 1 := by
  sorry

end tiffany_bag_difference_l2262_226255


namespace arithmetic_sequence_formula_l2262_226289

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a (n + 1) < a n) →  -- decreasing sequence
  a 2 * a 4 * a 6 = 45 →
  a 2 + a 4 + a 6 = 15 →
  ∃ d : ℝ, d < 0 ∧ ∀ n, a n = a 1 + (n - 1) * d ∧ a n = -2 * n + 13 :=
by sorry

end arithmetic_sequence_formula_l2262_226289


namespace smallest_with_ten_factors_l2262_226233

/-- A function that returns the number of distinct positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly ten distinct positive factors -/
def has_ten_factors (n : ℕ) : Prop := num_factors n = 10

theorem smallest_with_ten_factors :
  ∃ (n : ℕ), has_ten_factors n ∧ ∀ m : ℕ, m < n → ¬(has_ten_factors m) :=
sorry

end smallest_with_ten_factors_l2262_226233


namespace system_no_solution_l2262_226216

-- Define the system of equations
def system (n : ℝ) (x y z : ℝ) : Prop :=
  2*n*x + 3*y = 2 ∧ 3*n*y + 4*z = 3 ∧ 4*x + 2*n*z = 4

-- Theorem statement
theorem system_no_solution (n : ℝ) :
  (∀ x y z : ℝ, ¬ system n x y z) ↔ n = -Real.rpow 4 (1/3) :=
sorry

end system_no_solution_l2262_226216


namespace calculate_flat_fee_l2262_226202

/-- Calculates the flat fee for shipping given the total cost, cost per pound, and weight. -/
theorem calculate_flat_fee (C : ℝ) (cost_per_pound : ℝ) (weight : ℝ) (h1 : C = 9) (h2 : cost_per_pound = 0.8) (h3 : weight = 5) :
  ∃ F : ℝ, C = F + cost_per_pound * weight ∧ F = 5 := by
  sorry

end calculate_flat_fee_l2262_226202


namespace histogram_frequency_l2262_226236

theorem histogram_frequency (m : ℕ) (S1 S2 S3 : ℚ) :
  m ≥ 3 →
  S1 + S2 + S3 = (1 : ℚ) / 4 * (1 - (S1 + S2 + S3)) →
  S2 - S1 = S3 - S2 →
  S1 = (1 : ℚ) / 20 →
  (120 : ℚ) * S3 = 10 := by
  sorry

end histogram_frequency_l2262_226236


namespace escalator_length_l2262_226235

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time : ℝ),
    escalator_speed = 10 →
    person_speed = 4 →
    time = 8 →
    (escalator_speed + person_speed) * time = 112 := by
  sorry

end escalator_length_l2262_226235


namespace investment_percentage_l2262_226281

/-- Given two investors with equal initial investments, where one investor's value quadruples and ends up with $1900 more than the other, prove that the other investor's final value is 20% of their initial investment. -/
theorem investment_percentage (initial_investment : ℝ) (jackson_final : ℝ) (brandon_final : ℝ) :
  initial_investment > 0 →
  jackson_final = 4 * initial_investment →
  jackson_final - brandon_final = 1900 →
  brandon_final / initial_investment = 0.2 := by
sorry

end investment_percentage_l2262_226281


namespace total_toy_cost_l2262_226280

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_toy_cost : football_cost + marbles_cost = 12.30 := by
  sorry

end total_toy_cost_l2262_226280


namespace prob_same_number_four_dice_l2262_226269

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of getting the same number on all dice -/
def prob_same_number : ℚ := 1 / (standard_die_sides ^ (num_dice - 1))

/-- Theorem: The probability of getting the same number on all four standard six-sided dice is 1/216 -/
theorem prob_same_number_four_dice : 
  prob_same_number = 1 / 216 := by sorry

end prob_same_number_four_dice_l2262_226269


namespace village_population_equality_l2262_226250

/-- The rate of population increase for Village Y -/
def rate_Y : ℕ := sorry

/-- The initial population of Village X -/
def pop_X : ℕ := 76000

/-- The initial population of Village Y -/
def pop_Y : ℕ := 42000

/-- The rate of population decrease for Village X -/
def rate_X : ℕ := 1200

/-- The number of years after which the populations will be equal -/
def years : ℕ := 17

theorem village_population_equality :
  pop_X - (rate_X * years) = pop_Y + (rate_Y * years) ∧ rate_Y = 800 := by sorry

end village_population_equality_l2262_226250


namespace square_dissection_l2262_226219

/-- Given two squares with side lengths a and b respectively, prove that:
    1. The square with side a can be dissected into 3 identical squares.
    2. The square with side b can be dissected into 7 identical squares. -/
theorem square_dissection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (sa sb : ℝ),
    sa = a / Real.sqrt 3 ∧
    sb = b / Real.sqrt 7 ∧
    3 * sa ^ 2 = a ^ 2 ∧
    7 * sb ^ 2 = b ^ 2 :=
by sorry

end square_dissection_l2262_226219


namespace derivative_of_y_l2262_226243

open Real

noncomputable def y (x : ℝ) : ℝ := cos (2*x - 1) + 1 / (x^2)

theorem derivative_of_y :
  deriv y = λ x => -2 * sin (2*x - 1) - 2 / (x^3) :=
by sorry

end derivative_of_y_l2262_226243


namespace purse_value_is_107_percent_of_dollar_l2262_226238

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  quarters * coin_value "quarter"

theorem purse_value_is_107_percent_of_dollar (pennies nickels dimes quarters : ℕ) 
  (h_pennies : pennies = 2)
  (h_nickels : nickels = 3)
  (h_dimes : dimes = 4)
  (h_quarters : quarters = 2) :
  (total_value pennies nickels dimes quarters : ℚ) / 100 = 107 / 100 := by
  sorry

end purse_value_is_107_percent_of_dollar_l2262_226238


namespace range_of_a_for_false_proposition_l2262_226207

theorem range_of_a_for_false_proposition :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 ≤ 0) ↔ -2 < a ∧ a < 2 := by
  sorry

end range_of_a_for_false_proposition_l2262_226207


namespace fourth_person_height_l2262_226262

/-- Proves that the height of the 4th person is 85 inches given the conditions of the problem -/
theorem fourth_person_height :
  ∀ (h₁ h₂ h₃ h₄ : ℝ),
    h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
    h₂ - h₁ = 2 →  -- Difference between 1st and 2nd person
    h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd person
    h₄ - h₃ = 6 →  -- Difference between 3rd and 4th person
    (h₁ + h₂ + h₃ + h₄) / 4 = 79 →  -- Average height
    h₄ = 85 := by
  sorry

end fourth_person_height_l2262_226262


namespace equation_is_parabola_l2262_226254

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines the equation |y - 3| = √((x+4)² + (y-1)²) -/
def equation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Defines a parabola as a set of points satisfying a quadratic equation -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
    ∀ p ∈ S, a * p.x^2 + b * p.x * p.y + c * p.y^2 + d * p.x + e * p.y = 0

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  isParabola {p : Point2D | equation p} :=
sorry

end equation_is_parabola_l2262_226254


namespace water_jars_count_l2262_226204

/-- Given 35 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars,
    the total number of water-filled jars is 60. -/
theorem water_jars_count (total_volume : ℚ) (jar_sizes : Fin 3 → ℚ) :
  total_volume = 35 →
  jar_sizes 0 = 1/4 →
  jar_sizes 1 = 1/2 →
  jar_sizes 2 = 1 →
  (∃ (x : ℕ), total_volume = x * (jar_sizes 0 + jar_sizes 1 + jar_sizes 2)) →
  (∃ (x : ℕ), 3 * x = 60) :=
by sorry

end water_jars_count_l2262_226204


namespace ellipse_line_intersection_ratio_l2262_226213

/-- An ellipse intersecting with a line -/
structure EllipseLineIntersection where
  /-- Coefficient of x^2 in the ellipse equation -/
  m : ℝ
  /-- Coefficient of y^2 in the ellipse equation -/
  n : ℝ
  /-- x-coordinate of point M -/
  x₁ : ℝ
  /-- y-coordinate of point M -/
  y₁ : ℝ
  /-- x-coordinate of point N -/
  x₂ : ℝ
  /-- y-coordinate of point N -/
  y₂ : ℝ
  /-- Ellipse equation for point M -/
  ellipse_eq_m : m * x₁^2 + n * y₁^2 = 1
  /-- Ellipse equation for point N -/
  ellipse_eq_n : m * x₂^2 + n * y₂^2 = 1
  /-- Line equation for point M -/
  line_eq_m : x₁ + y₁ = 1
  /-- Line equation for point N -/
  line_eq_n : x₂ + y₂ = 1
  /-- Slope of OP, where P is the midpoint of MN -/
  slope_op : (y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2 / 2

/-- Theorem: If the slope of OP is √2/2, then m/n = √2/2 -/
theorem ellipse_line_intersection_ratio (e : EllipseLineIntersection) : e.m / e.n = Real.sqrt 2 / 2 := by
  sorry

end ellipse_line_intersection_ratio_l2262_226213


namespace sum_of_intersection_coordinates_l2262_226256

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 1)^2
def parabola2 (x y : ℝ) : Prop := x - 6 = (y + 1)^2

-- Define the intersection points
def intersection_points : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem sum_of_intersection_coordinates :
  intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry

end sum_of_intersection_coordinates_l2262_226256


namespace gondor_repair_earnings_l2262_226290

/-- Gondor's repair earnings problem -/
theorem gondor_repair_earnings (phone_repair_price laptop_repair_price : ℕ)
  (monday_phones wednesday_laptops thursday_laptops : ℕ)
  (total_earnings : ℕ) :
  phone_repair_price = 10 →
  laptop_repair_price = 20 →
  monday_phones = 3 →
  wednesday_laptops = 2 →
  thursday_laptops = 4 →
  total_earnings = 200 →
  ∃ tuesday_phones : ℕ,
    total_earnings = 
      phone_repair_price * (monday_phones + tuesday_phones) +
      laptop_repair_price * (wednesday_laptops + thursday_laptops) ∧
    tuesday_phones = 5 :=
by sorry

end gondor_repair_earnings_l2262_226290


namespace sean_julie_sum_ratio_l2262_226224

/-- The sum of even integers from 2 to 600, inclusive -/
def sean_sum : ℕ := 2 * (300 * 301) / 2

/-- The sum of integers from 1 to 300, inclusive -/
def julie_sum : ℕ := (300 * 301) / 2

/-- Theorem stating that Sean's sum divided by Julie's sum equals 2 -/
theorem sean_julie_sum_ratio :
  (sean_sum : ℚ) / (julie_sum : ℚ) = 2 := by sorry

end sean_julie_sum_ratio_l2262_226224


namespace smallest_n_value_l2262_226271

theorem smallest_n_value (o y m : ℕ+) (n : ℕ+) : 
  (10 * o = 16 * y) ∧ (16 * y = 18 * m) ∧ (18 * m = 18 * n) →
  n ≥ 40 ∧ ∃ (o' y' m' : ℕ+), 10 * o' = 16 * y' ∧ 16 * y' = 18 * m' ∧ 18 * m' = 18 * 40 :=
by sorry

end smallest_n_value_l2262_226271


namespace complex_fraction_simplification_l2262_226252

theorem complex_fraction_simplification :
  (7 + 8 * Complex.I) / (3 - 4 * Complex.I) = -11/25 + 52/25 * Complex.I := by
  sorry

end complex_fraction_simplification_l2262_226252


namespace rain_probability_tel_aviv_l2262_226240

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv :
  let n : ℕ := 6
  let k : ℕ := 4
  let p : ℝ := 0.5
  binomial_probability n k p = 0.234375 := by
sorry

end rain_probability_tel_aviv_l2262_226240


namespace expression_has_four_terms_l2262_226220

/-- The expression with the asterisk replaced by a monomial -/
def expression (x : ℝ) : ℝ := (x^4 - 3)^2 + (x^3 + 3*x)^2

/-- The result after expanding and combining like terms -/
def expanded_result (x : ℝ) : ℝ := x^8 + x^6 + 9*x^2 + 9

/-- Theorem stating that the expanded result has exactly four terms -/
theorem expression_has_four_terms :
  ∃ (a b c d : ℝ → ℝ),
    (∀ x, expanded_result x = a x + b x + c x + d x) ∧
    (∀ x, a x ≠ 0 ∧ b x ≠ 0 ∧ c x ≠ 0 ∧ d x ≠ 0) ∧
    (∀ x, expression x = expanded_result x) :=
sorry

end expression_has_four_terms_l2262_226220


namespace final_turtle_count_l2262_226264

def turtle_statues : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => turtle_statues 0 * 4  -- Second year: quadrupled
| 2 => turtle_statues 1 + 12 - 3  -- Third year: added 12, removed 3
| 3 => turtle_statues 2 + 2 * 3  -- Fourth year: added twice the number broken in year 3
| _ => 0  -- We only care about the first 4 years

theorem final_turtle_count : turtle_statues 3 = 31 := by
  sorry

#eval turtle_statues 3

end final_turtle_count_l2262_226264


namespace three_card_draw_different_colors_l2262_226261

def total_cards : ℕ := 16
def cards_per_color : ℕ := 4
def num_colors : ℕ := 4
def cards_drawn : ℕ := 3

theorem three_card_draw_different_colors : 
  (Nat.choose total_cards cards_drawn) - (num_colors * Nat.choose cards_per_color cards_drawn) = 544 := by
  sorry

end three_card_draw_different_colors_l2262_226261


namespace rectangle_width_decrease_l2262_226208

theorem rectangle_width_decrease (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let new_length := 1.4 * L
  let new_width := W / 1.4
  (new_length * new_width = L * W) ∧
  (2 * new_length + 2 * new_width = 2 * L + 2 * W) →
  (W - new_width) / W = 2 / 7 := by
sorry

end rectangle_width_decrease_l2262_226208


namespace burger_combinations_l2262_226251

/-- The number of different toppings available. -/
def num_toppings : ℕ := 10

/-- The number of choices for meat patties. -/
def patty_choices : ℕ := 4

/-- Theorem stating the total number of burger combinations. -/
theorem burger_combinations :
  (2 ^ num_toppings) * patty_choices = 4096 :=
sorry

end burger_combinations_l2262_226251


namespace negation_existence_statement_l2262_226277

open Real

theorem negation_existence_statement (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end negation_existence_statement_l2262_226277


namespace three_times_relationship_l2262_226278

theorem three_times_relationship (M₁ M₂ M₃ M₄ : ℝ) 
  (h₁ : M₁ = 2.02e-6)
  (h₂ : M₂ = 0.0000202)
  (h₃ : M₃ = 0.00000202)
  (h₄ : M₄ = 6.06e-5) :
  (M₄ = 3 * M₂ ∧ 
   M₄ ≠ 3 * M₁ ∧ 
   M₄ ≠ 3 * M₃ ∧ 
   M₃ ≠ 3 * M₁ ∧ 
   M₃ ≠ 3 * M₂ ∧ 
   M₂ ≠ 3 * M₁) :=
by sorry

end three_times_relationship_l2262_226278


namespace divide_and_add_l2262_226249

theorem divide_and_add (x : ℝ) : x = 72 → (x / 6) + 5 = 17 := by
  sorry

end divide_and_add_l2262_226249


namespace no_solution_exists_l2262_226285

theorem no_solution_exists : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2) := by
  sorry

end no_solution_exists_l2262_226285


namespace parabola_coeff_sum_l2262_226297

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, 2), and passing through (1, 6) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c) ∧
  (2 = a * (-3)^2 + b * (-3) + c) ∧
  (6 = a * 1^2 + b * 1 + c)

/-- The sum of coefficients a, b, and c equals 6 -/
theorem parabola_coeff_sum (a b c : ℝ) (h : Parabola a b c) : a + b + c = 6 := by
  sorry

end parabola_coeff_sum_l2262_226297


namespace square_root_sum_equals_ten_l2262_226241

theorem square_root_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end square_root_sum_equals_ten_l2262_226241


namespace captain_selection_theorem_l2262_226222

/-- The number of ways to choose 4 captains from a team of 12 people,
    where two of the captains must be from a subset of 4 specific players. -/
def captain_selection_ways (total_team : ℕ) (total_captains : ℕ) (specific_subset : ℕ) (required_from_subset : ℕ) : ℕ :=
  (Nat.choose specific_subset required_from_subset) * 
  (Nat.choose (total_team - specific_subset) (total_captains - required_from_subset))

/-- Theorem stating that the number of ways to choose 4 captains from a team of 12 people,
    where two of the captains must be from a subset of 4 specific players, is 168. -/
theorem captain_selection_theorem : 
  captain_selection_ways 12 4 4 2 = 168 := by
  sorry

end captain_selection_theorem_l2262_226222


namespace net_effect_theorem_l2262_226218

/-- Calculates the net effect on sale given price reduction, sale increase, tax, and discount -/
def net_effect_on_sale (price_reduction : ℝ) (sale_increase : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  let new_price_factor := 1 - price_reduction
  let new_quantity_factor := 1 + sale_increase
  let after_tax_factor := 1 + tax
  let after_discount_factor := 1 - discount
  new_price_factor * new_quantity_factor * after_tax_factor * after_discount_factor

/-- Theorem stating the net effect on sale given specific conditions -/
theorem net_effect_theorem :
  net_effect_on_sale 0.60 1.50 0.10 0.05 = 1.045 := by
  sorry

end net_effect_theorem_l2262_226218


namespace min_value_hyperbola_ellipse_foci_l2262_226206

/-- The minimum value of (4/m + 1/n) given the conditions of the problem -/
theorem min_value_hyperbola_ellipse_foci (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, x^2/m - y^2/n = 1) → 
  (∃ x y : ℝ, x^2/5 + y^2/2 = 1) → 
  (∀ x y : ℝ, x^2/m - y^2/n = 1 ↔ x^2/5 + y^2/2 = 1) → 
  (4/m + 1/n ≥ 3 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 4/m₀ + 1/n₀ = 3) :=
sorry

end min_value_hyperbola_ellipse_foci_l2262_226206


namespace smoothies_from_fifteen_bananas_l2262_226244

/-- The number of smoothies Caroline can make from a given number of bananas. -/
def smoothies_from_bananas (bananas : ℕ) : ℕ :=
  (9 * bananas) / 3

/-- Theorem stating that Caroline can make 45 smoothies from 15 bananas. -/
theorem smoothies_from_fifteen_bananas :
  smoothies_from_bananas 15 = 45 := by
  sorry

#eval smoothies_from_bananas 15

end smoothies_from_fifteen_bananas_l2262_226244


namespace characteristic_equation_of_A_l2262_226275

def A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 3, 1, 2; 2, 3, 1]

theorem characteristic_equation_of_A :
  ∃ (p q r : ℝ), A^3 + p • A^2 + q • A + r • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧
  p = -3 ∧ q = -9 ∧ r = -2 := by
  sorry

end characteristic_equation_of_A_l2262_226275


namespace dodecahedron_interior_diagonals_l2262_226203

/-- A dodecahedron is a 3-dimensional figure with 20 vertices -/
structure Dodecahedron where
  vertices : Finset ℕ
  vertex_count : vertices.card = 20

/-- Each vertex in a dodecahedron is connected to 3 other vertices by edges -/
def connected_vertices (d : Dodecahedron) (v : ℕ) : Finset ℕ :=
  sorry

axiom connected_vertices_count (d : Dodecahedron) (v : ℕ) (h : v ∈ d.vertices) :
  (connected_vertices d v).card = 3

/-- An interior diagonal is a segment connecting two vertices which do not share an edge -/
def interior_diagonals (d : Dodecahedron) : Finset (ℕ × ℕ) :=
  sorry

/-- The main theorem: a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  (interior_diagonals d).card = 160 :=
sorry

end dodecahedron_interior_diagonals_l2262_226203


namespace problem_solution_l2262_226237

theorem problem_solution : ∃ x : ℚ, (x + x / 4 = 80 - 80 / 4) ∧ x = 48 := by
  sorry

end problem_solution_l2262_226237


namespace imaginary_part_of_z_l2262_226299

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * Complex.I) * z = 5) : 
  Complex.im z = -2 := by sorry

end imaginary_part_of_z_l2262_226299


namespace a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight_l2262_226284

theorem a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight :
  ∀ a : ℝ, a = 1 / (Real.sqrt 5 - 2) → a^4 - 4*a^3 - 4*a + 7 = 8 := by
sorry

end a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight_l2262_226284


namespace sin_equality_iff_side_equality_l2262_226253

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (angle_sum : A + B + C = π)

-- State the theorem
theorem sin_equality_iff_side_equality (t : Triangle) : 
  Real.sin t.A = Real.sin t.B ↔ t.a = t.b :=
sorry

end sin_equality_iff_side_equality_l2262_226253


namespace rhombus_area_l2262_226200

/-- The area of a rhombus with vertices at (0, 3.5), (7, 0), (0, -3.5), and (-7, 0) is 49 square units. -/
theorem rhombus_area : 
  let v1 : ℝ × ℝ := (0, 3.5)
  let v2 : ℝ × ℝ := (7, 0)
  let v3 : ℝ × ℝ := (0, -3.5)
  let v4 : ℝ × ℝ := (-7, 0)
  let d1 : ℝ := v1.2 - v3.2  -- Vertical diagonal
  let d2 : ℝ := v2.1 - v4.1  -- Horizontal diagonal
  let area : ℝ := (d1 * d2) / 2
  area = 49 := by sorry

end rhombus_area_l2262_226200


namespace leftover_coins_value_l2262_226260

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 60
def michael_quarters : ℕ := 94
def michael_dimes : ℕ := 184
def sara_quarters : ℕ := 137
def sara_dimes : ℕ := 312

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

def total_quarters : ℕ := michael_quarters + sara_quarters
def total_dimes : ℕ := michael_dimes + sara_dimes

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll

def leftover_value : ℚ := 
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value

theorem leftover_coins_value : leftover_value = 6.85 := by
  sorry

end leftover_coins_value_l2262_226260


namespace stripe_area_on_cylinder_l2262_226266

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylinder (diameter : Real) (stripe_width : Real) (revolutions : Real) :
  diameter = 40 ∧ stripe_width = 4 ∧ revolutions = 3 →
  stripe_width * revolutions * π * diameter = 480 * π := by
  sorry

end stripe_area_on_cylinder_l2262_226266


namespace triangle_angle_solution_l2262_226294

-- Define the angles in degrees
def angle_PQR : ℝ := 90
def angle_PQS (x : ℝ) : ℝ := 3 * x
def angle_SQR (y : ℝ) : ℝ := y

-- State the theorem
theorem triangle_angle_solution :
  ∃ (x y : ℝ),
    angle_PQS x + angle_SQR y = angle_PQR ∧
    x = 18 ∧
    y = 36 := by
  sorry

end triangle_angle_solution_l2262_226294


namespace harry_potter_book_price_l2262_226239

theorem harry_potter_book_price : 
  ∀ (wang_money li_money book_price : ℕ),
  wang_money + 6 = 2 * book_price →
  li_money + 31 = 2 * book_price →
  wang_money + li_money = 3 * book_price →
  book_price = 37 := by
sorry

end harry_potter_book_price_l2262_226239


namespace expression_simplification_l2262_226211

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹) = (a * b)⁻¹ := by
  sorry

end expression_simplification_l2262_226211


namespace toms_dimes_calculation_l2262_226223

/-- Calculates the final number of dimes Tom has after receiving and spending some. -/
def final_dimes (initial : ℕ) (received : ℕ) (spent : ℕ) : ℕ :=
  initial + received - spent

/-- Proves that Tom's final number of dimes is correct given the initial amount, 
    the amount received, and the amount spent. -/
theorem toms_dimes_calculation (initial : ℕ) (received : ℕ) (spent : ℕ) :
  final_dimes initial received spent = initial + received - spent :=
by sorry

end toms_dimes_calculation_l2262_226223


namespace subset_iff_elements_l2262_226272

theorem subset_iff_elements (A B : Set α) : A ⊆ B ↔ ∀ x, x ∈ A → x ∈ B := by sorry

end subset_iff_elements_l2262_226272


namespace discount_sales_increase_l2262_226214

/-- Calculates the percent increase in gross income given a discount and increase in sales volume -/
theorem discount_sales_increase (discount : ℝ) (sales_increase : ℝ) : 
  discount = 0.1 → sales_increase = 0.3 → 
  (1 - discount) * (1 + sales_increase) - 1 = 0.17 := by
  sorry

#check discount_sales_increase

end discount_sales_increase_l2262_226214


namespace regular_polygon_140_deg_interior_angle_l2262_226276

/-- A regular polygon with interior angles of 140 degrees has 9 sides -/
theorem regular_polygon_140_deg_interior_angle (n : ℕ) : 
  (n ≥ 3) → 
  (∀ θ : ℝ, θ = 140 → (180 * (n - 2) : ℝ) = n * θ) → 
  n = 9 := by
sorry

end regular_polygon_140_deg_interior_angle_l2262_226276


namespace function_properties_l2262_226279

def FunctionProperties (f : ℝ → ℝ) : Prop :=
  (∀ x y, f x + f y = f (x + y)) ∧ (∀ x, x > 0 → f x < 0)

theorem function_properties (f : ℝ → ℝ) (h : FunctionProperties f) :
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y) := by
  sorry

end function_properties_l2262_226279


namespace z₁z₂_value_a_value_when_sum_real_l2262_226248

-- Define complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := 2 + a * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

-- Theorem 1: When a = 1, z₁z₂ = 10 - 5i
theorem z₁z₂_value : z₁ 1 * z₂ = 10 - 5 * Complex.I := by sorry

-- Theorem 2: When z₁ + z₂ is a real number, a = 4
theorem a_value_when_sum_real : 
  ∃ (a : ℝ), (z₁ a + z₂).im = 0 → a = 4 := by sorry

end z₁z₂_value_a_value_when_sum_real_l2262_226248


namespace factory_sample_theorem_l2262_226268

/-- Given a factory with total products, a sample size, and products from one workshop,
    calculate the number of products drawn from this workshop in a stratified sampling. -/
def stratifiedSampleSize (totalProducts sampleSize workshopProducts : ℕ) : ℕ :=
  (workshopProducts * sampleSize) / totalProducts

/-- Theorem stating that for the given values, the stratified sample size is 16. -/
theorem factory_sample_theorem :
  stratifiedSampleSize 2048 128 256 = 16 := by
  sorry

end factory_sample_theorem_l2262_226268


namespace prize_distribution_and_cost_l2262_226257

/-- Represents the prize distribution and cost calculation for a school event --/
theorem prize_distribution_and_cost 
  (x : ℕ) -- number of first prize items
  (h1 : x + (3*x - 2) + (52 - 4*x) = 50) -- total prizes constraint
  (h2 : x > 0) -- ensure positive number of first prize items
  (h3 : 3*x - 2 ≥ 0) -- ensure non-negative number of second prize items
  : 
  (20*x + 14*(3*x - 2) + 8*(52 - 4*x) = 30*x + 388) ∧ 
  (3*x - 2 = 22 → 20*x + 14*(3*x - 2) + 8*(52 - 4*x) = 628)
  := by sorry


end prize_distribution_and_cost_l2262_226257


namespace sum_of_xyz_l2262_226234

theorem sum_of_xyz (a b : ℝ) (x y z : ℕ+) : 
  a^2 = 9/14 ∧ 
  b^2 = (3 + Real.sqrt 7)^2 / 14 ∧ 
  a < 0 ∧ 
  b > 0 ∧ 
  (a + b)^3 = (x : ℝ) * Real.sqrt y / z →
  x + y + z = 7 := by
  sorry

end sum_of_xyz_l2262_226234


namespace max_cos_sum_l2262_226215

theorem max_cos_sum (x y : ℝ) 
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (max : ℝ), max = 1 + (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ 
    ∀ (x' y' : ℝ), 
      Real.sin y' + Real.sin x' + Real.cos (3 * x') = 0 →
      Real.sin (2 * y') - Real.sin (2 * x') = Real.cos (4 * x') + Real.cos (2 * x') →
      Real.cos y' + Real.cos x' ≤ max :=
by sorry

end max_cos_sum_l2262_226215


namespace cistern_fill_time_l2262_226282

-- Define the fill rates of pipes
def fill_rate_p : ℚ := 1 / 10
def fill_rate_q : ℚ := 1 / 15
def drain_rate_r : ℚ := 1 / 30

-- Define the time pipes p and q are open together
def initial_time : ℚ := 2

-- Define the function to calculate the remaining time to fill the cistern
def remaining_fill_time (fill_rate_p fill_rate_q drain_rate_r initial_time : ℚ) : ℚ :=
  let initial_fill := (fill_rate_p + fill_rate_q) * initial_time
  let remaining_volume := 1 - initial_fill
  let net_fill_rate := fill_rate_q - drain_rate_r
  remaining_volume / net_fill_rate

-- Theorem statement
theorem cistern_fill_time :
  remaining_fill_time fill_rate_p fill_rate_q drain_rate_r initial_time = 20 := by
  sorry

end cistern_fill_time_l2262_226282


namespace sum_real_imag_parts_3_minus_4i_l2262_226265

theorem sum_real_imag_parts_3_minus_4i :
  let z : ℂ := 3 - 4*I
  (z.re + z.im : ℝ) = -1 := by sorry

end sum_real_imag_parts_3_minus_4i_l2262_226265


namespace rationalize_denominator_l2262_226296

-- Define the original expression
def original_expr := (4 : ℚ) / (3 * (7 : ℚ)^(1/3))

-- Define the rationalized expression
def rationalized_expr := (4 * (49 : ℚ)^(1/3)) / 21

-- Define the property that 49 is not divisible by the cube of any prime
def not_cube_divisible (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^3 ∣ n)

-- Theorem statement
theorem rationalize_denominator :
  original_expr = rationalized_expr ∧ not_cube_divisible 49 := by sorry

end rationalize_denominator_l2262_226296


namespace factor_x_10_minus_1024_l2262_226270

theorem factor_x_10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2*x^3 + 4*x^2 + 8*x + 16) := by
  sorry

end factor_x_10_minus_1024_l2262_226270


namespace largest_non_sum_30_and_composite_l2262_226229

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

/-- The property we want to prove for 211 -/
def IsLargestNonSum30AndComposite (m : ℕ) : Prop :=
  (∀ n > m, ∃ k c : ℕ, n = 30 * k + c ∧ k > 0 ∧ IsComposite c) ∧
  (¬∃ k c : ℕ, m = 30 * k + c ∧ k > 0 ∧ IsComposite c)

/-- The main theorem -/
theorem largest_non_sum_30_and_composite :
  IsLargestNonSum30AndComposite 211 :=
sorry

end largest_non_sum_30_and_composite_l2262_226229


namespace mathematical_induction_l2262_226288

theorem mathematical_induction (P : ℕ → Prop) (base : ℕ) 
  (base_case : P base)
  (inductive_step : ∀ k : ℕ, k ≥ base → P k → P (k + 1)) :
  ∀ n : ℕ, n ≥ base → P n :=
by
  sorry


end mathematical_induction_l2262_226288


namespace equation_represents_two_lines_and_point_l2262_226245

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0

-- Define the point (1, -2)
def point : ℝ × ℝ := (1, -2)

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem equation_represents_two_lines_and_point :
  ∀ x y : ℝ, equation x y ↔ (x = point.1 ∧ y = point.2) ∨ line1 x y ∨ line2 x y :=
sorry

end equation_represents_two_lines_and_point_l2262_226245


namespace root_shift_polynomial_l2262_226226

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 4*a^2 + 6*a - 8 = 0) ∧ 
  (b^3 - 4*b^2 + 6*b - 8 = 0) ∧ 
  (c^3 - 4*c^2 + 6*c - 8 = 0) →
  ((a + 3)^3 - 13*(a + 3)^2 + 57*(a + 3) - 89 = 0) ∧
  ((b + 3)^3 - 13*(b + 3)^2 + 57*(b + 3) - 89 = 0) ∧
  ((c + 3)^3 - 13*(c + 3)^2 + 57*(c + 3) - 89 = 0) :=
by sorry

end root_shift_polynomial_l2262_226226


namespace last_three_digits_of_1973_power_46_l2262_226212

theorem last_three_digits_of_1973_power_46 :
  1973^46 % 1000 = 689 := by
sorry

end last_three_digits_of_1973_power_46_l2262_226212


namespace special_function_properties_l2262_226263

/-- A function satisfying certain properties -/
structure SpecialFunction where
  g : ℝ → ℝ
  pos : ∀ x, g x > 0
  mult : ∀ a b, g a * g b = g (a * b)

/-- Properties of the special function -/
theorem special_function_properties (f : SpecialFunction) :
  (f.g 1 = 1) ∧
  (∀ a ≠ 0, f.g (a⁻¹) = (f.g a)⁻¹) ∧
  (∀ a, f.g (a^2) = f.g a * f.g a) := by
  sorry

end special_function_properties_l2262_226263


namespace first_digit_base8_is_3_l2262_226273

/-- The base 3 representation of y -/
def y_base3 : List Nat := [2, 1, 2, 0, 2, 1, 2]

/-- Convert a list of digits in base b to a natural number -/
def to_nat (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (λ d acc => d + b * acc) 0

/-- The value of y in base 10 -/
def y : Nat := to_nat y_base3 3

/-- Get the first digit of a number in base b -/
def first_digit (n : Nat) (b : Nat) : Nat :=
  n / (b ^ ((Nat.log b n) - 1))

theorem first_digit_base8_is_3 : first_digit y 8 = 3 := by
  sorry

end first_digit_base8_is_3_l2262_226273


namespace triangle_inequality_l2262_226230

/-- Given points P, Q, R, S on a line with PQ = a, PR = b, PS = c,
    if PQ and RS can be rotated to form a non-degenerate triangle,
    then a < c/2 and b < a + c/2 -/
theorem triangle_inequality (a b c : ℝ) 
  (h_order : 0 < a ∧ a < b ∧ b < c)
  (h_triangle : 2*b > c ∧ c > a ∧ c > b - a) :
  a < c/2 ∧ b < a + c/2 := by
sorry

end triangle_inequality_l2262_226230


namespace roots_of_polynomials_l2262_226291

theorem roots_of_polynomials (α : ℝ) : 
  α^2 = 2*α + 2 → α^5 = 44*α + 32 := by sorry

end roots_of_polynomials_l2262_226291


namespace inequalities_hold_l2262_226242

theorem inequalities_hold (a b c x y z : ℝ) 
  (hx : x ≤ a) (hy : y ≤ b) (hz : z ≤ c) : 
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x * y * z ≤ a * b * c) := by
  sorry

end inequalities_hold_l2262_226242


namespace sequence_non_negative_l2262_226258

/-- Sequence a_n defined recursively -/
def a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n + 1, a₀ => 2 * a n a₀ - n^2

/-- Theorem stating the condition for non-negativity of the sequence -/
theorem sequence_non_negative (a₀ : ℝ) :
  (∀ n : ℕ, a n a₀ ≥ 0) ↔ a₀ ≥ 3 := by
  sorry

end sequence_non_negative_l2262_226258


namespace smallest_integer_in_ratio_l2262_226274

theorem smallest_integer_in_ratio (a b c : ℕ+) : 
  (a : ℝ) / b = 2 / 3 → 
  (a : ℝ) / c = 2 / 5 → 
  (b : ℝ) / c = 3 / 5 → 
  a + b + c = 90 → 
  a = 18 := by
sorry

end smallest_integer_in_ratio_l2262_226274


namespace book_price_increase_l2262_226283

theorem book_price_increase (original_price : ℝ) : 
  original_price > 0 →
  original_price * 1.5 = 450 →
  original_price = 300 := by
sorry

end book_price_increase_l2262_226283


namespace min_angle_line_equation_l2262_226267

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The angle between two points and a center point -/
def angle (center : ℝ × ℝ) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a line intersects a circle at two points -/
def intersectsCircle (l : Line) (c : Circle) : Prop := sorry

/-- Check if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem min_angle_line_equation 
  (c : Circle)
  (m : ℝ × ℝ)
  (l : Line) :
  c.center = (3, 4) →
  c.radius = 5 →
  m = (1, 2) →
  pointOnLine m l →
  intersectsCircle l c →
  (∀ l' : Line, intersectsCircle l' c → angle c.center (1, 2) (3, 4) ≤ angle c.center (1, 2) (3, 4)) →
  l.slope = -1 ∧ l.yIntercept = 3 :=
sorry

end min_angle_line_equation_l2262_226267


namespace tan_alpha_minus_beta_l2262_226217

theorem tan_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.tan (α + π / 5) = 2) 
  (h2 : Real.tan (β - 4 * π / 5) = -3) : 
  Real.tan (α - β) = -1 := by
sorry

end tan_alpha_minus_beta_l2262_226217


namespace garden_area_l2262_226221

theorem garden_area (width : ℝ) (length : ℝ) :
  length = 3 * width + 30 →
  2 * (length + width) = 800 →
  width * length = 28443.75 := by
sorry

end garden_area_l2262_226221


namespace soda_price_calculation_soda_price_proof_l2262_226295

/-- Given the cost of sandwiches and total cost, calculate the price of each soda -/
theorem soda_price_calculation (sandwich_price : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) (total_cost : ℚ) : ℚ :=
  let sandwich_total := sandwich_price * num_sandwiches
  let soda_total := total_cost - sandwich_total
  soda_total / num_sodas

/-- Prove that the price of each soda is $1.87 given the problem conditions -/
theorem soda_price_proof :
  soda_price_calculation 2.49 2 4 12.46 = 1.87 := by
  sorry

end soda_price_calculation_soda_price_proof_l2262_226295


namespace cell_growth_proof_l2262_226227

def geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

theorem cell_growth_proof :
  let initial_cells : ℕ := 3
  let growth_factor : ℕ := 2
  let num_terms : ℕ := 5
  geometric_sequence initial_cells growth_factor num_terms = 48 := by
  sorry

end cell_growth_proof_l2262_226227


namespace min_value_fraction_l2262_226232

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_fraction_l2262_226232


namespace square_plus_reciprocal_square_l2262_226231

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 3) : a^2 + 1/a^2 = 7 := by
  sorry

end square_plus_reciprocal_square_l2262_226231


namespace min_value_theorem_l2262_226210

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  4 * x^2 + 1 / x^3 ≥ 5 ∧
  (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) := by
sorry

end min_value_theorem_l2262_226210


namespace largest_four_digit_prime_product_l2262_226205

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_four_digit_prime_product :
  ∀ (n x y z : ℕ),
    n = x * y * z * (10 * x + y) →
    x < 20 ∧ y < 20 ∧ z < 20 →
    is_prime x ∧ is_prime y ∧ is_prime z →
    is_prime (10 * x + y) →
    x ≠ y ∧ x ≠ z ∧ y ≠ z →
    n ≤ 25058 :=
by sorry

end largest_four_digit_prime_product_l2262_226205
