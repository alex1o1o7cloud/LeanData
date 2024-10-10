import Mathlib

namespace mabel_marbles_l1120_112081

/-- Given information about marbles of Amanda, Katrina, and Mabel -/
def marble_problem (amanda katrina mabel : ℕ) : Prop :=
  (amanda + 12 = 2 * katrina) ∧
  (mabel = 5 * katrina) ∧
  (mabel = amanda + 63)

/-- Theorem stating that under the given conditions, Mabel has 85 marbles -/
theorem mabel_marbles :
  ∀ amanda katrina mabel : ℕ,
  marble_problem amanda katrina mabel →
  mabel = 85 := by
  sorry

end mabel_marbles_l1120_112081


namespace parabola_c_value_l1120_112088

/-- A parabola with equation y = x^2 + bx + c passes through points (2,3) and (5,6) -/
def parabola_through_points (b c : ℝ) : Prop :=
  3 = 2^2 + 2*b + c ∧ 6 = 5^2 + 5*b + c

/-- The theorem stating that c = -13 for the given parabola -/
theorem parabola_c_value : ∃ b : ℝ, parabola_through_points b (-13) := by
  sorry

end parabola_c_value_l1120_112088


namespace ellipse_eccentricity_l1120_112084

/-- The eccentricity of an ellipse with equation x^2 + y^2/4 = 1 is √3/2 -/
theorem ellipse_eccentricity : 
  let e : ℝ := (Real.sqrt 3) / 2
  ∀ x y : ℝ, x^2 + y^2/4 = 1 → e = (Real.sqrt (4 - 1)) / 2 :=
by sorry

end ellipse_eccentricity_l1120_112084


namespace equation_solution_l1120_112079

theorem equation_solution : 
  ∃ x : ℝ, 
    (2.5 * ((3.6 * x * 2.50) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002) ∧ 
    (abs (x - 0.48) < 0.00000000000001) := by
  sorry

end equation_solution_l1120_112079


namespace carl_reaches_goal_in_53_days_l1120_112091

/-- Represents Carl's earnings and candy bar goal --/
structure CarlsEarnings where
  candy_bar_cost : ℚ
  weekly_trash_pay : ℚ
  biweekly_dog_pay : ℚ
  aunt_payment : ℚ
  candy_bar_goal : ℕ

/-- Calculates the number of days needed for Carl to reach his candy bar goal --/
def days_to_reach_goal (e : CarlsEarnings) : ℕ :=
  sorry

/-- Theorem stating that given Carl's specific earnings and goal, it takes 53 days to reach the goal --/
theorem carl_reaches_goal_in_53_days :
  let e : CarlsEarnings := {
    candy_bar_cost := 1/2,
    weekly_trash_pay := 3/4,
    biweekly_dog_pay := 5/4,
    aunt_payment := 5,
    candy_bar_goal := 30
  }
  days_to_reach_goal e = 53 := by
  sorry

end carl_reaches_goal_in_53_days_l1120_112091


namespace parabola_hyperbola_focus_coincidence_l1120_112000

/-- The value of p for which the focus of the parabola y² = 2px (p > 0) 
    coincides with the right focus of the hyperbola x² - y² = 2 -/
theorem parabola_hyperbola_focus_coincidence (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 - y^2 = 2 ∧ x = p/2 ∧ x = 2) → 
  p = 4 :=
by sorry

end parabola_hyperbola_focus_coincidence_l1120_112000


namespace imaginary_part_of_complex_expression_l1120_112092

theorem imaginary_part_of_complex_expression :
  let z : ℂ := 1 - I
  (z^2 + 2/z).im = -1 := by sorry

end imaginary_part_of_complex_expression_l1120_112092


namespace equation_solution_l1120_112076

theorem equation_solution (x : ℝ) : 
  (24 : ℝ) / 36 = Real.sqrt (x / 36) → x = 16 := by
  sorry

end equation_solution_l1120_112076


namespace playerB_is_best_choice_l1120_112018

-- Define a structure for a player
structure Player where
  name : String
  average : Float
  variance : Float

-- Define the players
def playerA : Player := { name := "A", average := 9.2, variance := 3.6 }
def playerB : Player := { name := "B", average := 9.5, variance := 3.6 }
def playerC : Player := { name := "C", average := 9.5, variance := 7.4 }
def playerD : Player := { name := "D", average := 9.2, variance := 8.1 }

def players : List Player := [playerA, playerB, playerC, playerD]

-- Function to determine if a player is the best choice
def isBestChoice (p : Player) (players : List Player) : Prop :=
  (∀ q ∈ players, p.average ≥ q.average) ∧
  (∀ q ∈ players, p.variance ≤ q.variance) ∧
  (∃ q ∈ players, p.average > q.average ∨ p.variance < q.variance)

-- Theorem stating that playerB is the best choice
theorem playerB_is_best_choice : isBestChoice playerB players := by
  sorry

end playerB_is_best_choice_l1120_112018


namespace glove_sequences_l1120_112028

/-- Represents the number of hands. -/
def num_hands : ℕ := 2

/-- Represents the number of layers of gloves. -/
def num_layers : ℕ := 2

/-- Represents whether the inner gloves are identical. -/
def inner_gloves_identical : Prop := True

/-- Represents whether the outer gloves are distinct for left and right hands. -/
def outer_gloves_distinct : Prop := True

/-- The number of different sequences for wearing the gloves. -/
def num_sequences : ℕ := 6

/-- Theorem stating that the number of different sequences for wearing the gloves is 6. -/
theorem glove_sequences :
  num_hands = 2 ∧ 
  num_layers = 2 ∧ 
  inner_gloves_identical ∧ 
  outer_gloves_distinct →
  num_sequences = 6 :=
by sorry


end glove_sequences_l1120_112028


namespace painter_workdays_l1120_112033

theorem painter_workdays (job_size : ℝ) (rate : ℝ) (h : job_size = 6 * 1.5 * rate) :
  job_size = 4 * 2.25 * rate := by
sorry

end painter_workdays_l1120_112033


namespace percentage_calculation_l1120_112072

theorem percentage_calculation (x : ℝ) : 
  (x / 100) * (25 / 100 * 1600) = 20 → x = 5 := by
  sorry

end percentage_calculation_l1120_112072


namespace wire_cutting_l1120_112059

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 21 →
  ratio = 2 / 5 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 6 := by
sorry

end wire_cutting_l1120_112059


namespace platform_length_l1120_112034

/-- The length of a train platform given crossing times and lengths -/
theorem platform_length 
  (train_length : ℝ) 
  (first_time : ℝ) 
  (second_time : ℝ) 
  (second_platform : ℝ) 
  (h1 : train_length = 190) 
  (h2 : first_time = 15) 
  (h3 : second_time = 20) 
  (h4 : second_platform = 250) : 
  ∃ (first_platform : ℝ), 
    first_platform = 140 ∧ 
    (train_length + first_platform) / first_time = 
    (train_length + second_platform) / second_time :=
by sorry

end platform_length_l1120_112034


namespace negation_equivalence_l1120_112042

theorem negation_equivalence :
  (¬ ∃ x₀ > 2, x₀^3 - 2*x₀^2 < 0) ↔ (∀ x > 2, x^3 - 2*x^2 ≥ 0) := by
  sorry

end negation_equivalence_l1120_112042


namespace base_b_not_divisible_by_five_l1120_112093

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 8, 10} : Set ℤ) →
  (b^2 * (3*b - 2) % 5 ≠ 0 ↔ b = 8) := by
  sorry

end base_b_not_divisible_by_five_l1120_112093


namespace power_sum_inequality_l1120_112047

theorem power_sum_inequality (a b c : ℝ) (m : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) := by
  sorry

end power_sum_inequality_l1120_112047


namespace chord_line_equation_l1120_112055

/-- Given a circle with equation x^2 + y^2 = 10 and a chord with midpoint P(1, 1),
    the equation of the line containing this chord is x + y - 2 = 0 -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 + y^2 = 10) →
  (∃ (t : ℝ), x = 1 + t ∧ y = 1 - t) →
  (x + y - 2 = 0) :=
by sorry

end chord_line_equation_l1120_112055


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1120_112050

/-- The quadratic function f(x) = x^2 - ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a

/-- The discriminant of f(x) -/
def discriminant (a : ℝ) : ℝ := a^2 - 4*a

/-- f(x) has two distinct zeros -/
def has_two_distinct_zeros (a : ℝ) : Prop := discriminant a > 0

/-- Condition "a > 4" is sufficient for f(x) to have two distinct zeros -/
theorem sufficient_condition (a : ℝ) (h : a > 4) : has_two_distinct_zeros a := by
  sorry

/-- Condition "a > 4" is not necessary for f(x) to have two distinct zeros -/
theorem not_necessary_condition : ∃ a : ℝ, a ≤ 4 ∧ has_two_distinct_zeros a := by
  sorry

/-- "a > 4" is a sufficient but not necessary condition for f(x) to have two distinct zeros -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a > 4 → has_two_distinct_zeros a) ∧
  (∃ a : ℝ, a ≤ 4 ∧ has_two_distinct_zeros a) := by
  sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1120_112050


namespace intersection_sum_l1120_112009

/-- Given two lines y = 2x + c and y = 4x + d intersecting at (3, 12), prove that c + d = 6 -/
theorem intersection_sum (c d : ℝ) : 
  (2 * 3 + c = 12) → (4 * 3 + d = 12) → c + d = 6 := by
  sorry

end intersection_sum_l1120_112009


namespace probability_of_winning_l1120_112015

/-- The probability of player A winning a match in a game with 2n rounds -/
def P (n : ℕ) : ℚ :=
  1/2 * (1 - (Nat.choose (2*n) n : ℚ) / (2^(2*n)))

/-- Theorem stating the probability of player A winning the match -/
theorem probability_of_winning (n : ℕ) (h : n > 0) : 
  P n = 1/2 * (1 - (Nat.choose (2*n) n : ℚ) / (2^(2*n))) :=
by sorry

end probability_of_winning_l1120_112015


namespace bertha_initial_balls_l1120_112002

def tennis_balls (initial_balls : ℕ) : Prop :=
  let worn_out := 20 / 10
  let lost := 20 / 5
  let bought := 20 / 4 * 3
  let final_balls := initial_balls - 1 - worn_out - lost + bought
  final_balls = 10

theorem bertha_initial_balls : 
  ∃ (x : ℕ), tennis_balls x ∧ x = 2 :=
sorry

end bertha_initial_balls_l1120_112002


namespace optimal_solution_is_valid_and_minimal_l1120_112056

/-- Represents a nail in the painting hanging problem -/
inductive Nail
| a₁ : Nail
| a₂ : Nail
| a₃ : Nail
| a₄ : Nail

/-- Represents a sequence of nails and their inverses -/
inductive NailSequence
| empty : NailSequence
| cons : Nail → NailSequence → NailSequence
| inv : Nail → NailSequence → NailSequence

/-- Counts the number of symbols in a nail sequence -/
def symbolCount : NailSequence → Nat
| NailSequence.empty => 0
| NailSequence.cons _ s => 1 + symbolCount s
| NailSequence.inv _ s => 1 + symbolCount s

/-- Checks if a nail sequence falls when a given nail is removed -/
def fallsWhenRemoved (s : NailSequence) (n : Nail) : Prop := sorry

/-- Represents the optimal solution [[a₁, a₂], [a₃, a₄]] -/
def optimalSolution : NailSequence := sorry

/-- Theorem: The optimal solution is valid and minimal -/
theorem optimal_solution_is_valid_and_minimal :
  (∀ n : Nail, fallsWhenRemoved optimalSolution n) ∧
  (∀ s : NailSequence, (∀ n : Nail, fallsWhenRemoved s n) → symbolCount optimalSolution ≤ symbolCount s) := by
  sorry

end optimal_solution_is_valid_and_minimal_l1120_112056


namespace pipe_sale_result_l1120_112054

theorem pipe_sale_result : 
  ∀ (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ),
    price = 1.20 →
    profit_percent = 20 →
    loss_percent = 20 →
    let profit_pipe_cost := price / (1 + profit_percent / 100)
    let loss_pipe_cost := price / (1 - loss_percent / 100)
    let total_cost := profit_pipe_cost + loss_pipe_cost
    let total_revenue := 2 * price
    total_revenue - total_cost = -0.10 := by
  sorry

end pipe_sale_result_l1120_112054


namespace smallest_gcd_yz_l1120_112099

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x.val y.val = 224) (h2 : Nat.gcd x.val z.val = 546) :
  ∃ (y' z' : ℕ+), Nat.gcd y'.val z'.val = 14 ∧ 
  (∀ (a b : ℕ+), Nat.gcd x.val a.val = 224 → Nat.gcd x.val b.val = 546 → 
    Nat.gcd a.val b.val ≥ 14) :=
sorry

end smallest_gcd_yz_l1120_112099


namespace cube_volume_from_space_diagonal_l1120_112032

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 216 := by
  sorry

end cube_volume_from_space_diagonal_l1120_112032


namespace count_two_digit_primes_ending_in_3_l1120_112021

def two_digit_primes_ending_in_3 : List Nat := [13, 23, 33, 43, 53, 63, 73, 83, 93]

theorem count_two_digit_primes_ending_in_3 : 
  (two_digit_primes_ending_in_3.filter Nat.Prime).length = 6 := by
  sorry

end count_two_digit_primes_ending_in_3_l1120_112021


namespace square_root_statements_l1120_112077

theorem square_root_statements :
  (Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10) ∧
  (Real.sqrt 2 + Real.sqrt 5 ≠ Real.sqrt 7) ∧
  (Real.sqrt 18 / Real.sqrt 2 = 3) ∧
  (Real.sqrt 12 = 2 * Real.sqrt 3) := by
  sorry

end square_root_statements_l1120_112077


namespace line_slope_135_degrees_l1120_112094

/-- The slope of a line in degrees -/
def Slope : Type := ℝ

/-- The equation of a line in the form mx + y + c = 0 -/
structure Line where
  m : ℝ
  c : ℝ

/-- The tangent of an angle in degrees -/
noncomputable def tan_degrees (θ : ℝ) : ℝ := sorry

theorem line_slope_135_degrees (l : Line) (h : l.c = 2) : 
  (tan_degrees 135 = -l.m) → l.m = 1 := by sorry

end line_slope_135_degrees_l1120_112094


namespace distinct_numbers_probability_l1120_112012

def num_sides : ℕ := 5
def num_dice : ℕ := 5

theorem distinct_numbers_probability :
  (Nat.factorial num_dice : ℚ) / (num_sides ^ num_dice : ℚ) = 120 / 3125 := by
  sorry

end distinct_numbers_probability_l1120_112012


namespace folded_rectangle_BC_l1120_112068

/-- A rectangle ABCD with the following properties:
  - AB = 10
  - AD is folded onto AB, creating crease AE
  - Triangle AED is folded along DE
  - AE intersects BC at point F
  - Area of triangle ABF is 2 -/
structure FoldedRectangle where
  AB : ℝ
  BC : ℝ
  area_ABF : ℝ
  AB_eq_10 : AB = 10
  area_ABF_eq_2 : area_ABF = 2

/-- Theorem: In a FoldedRectangle, BC = 5.2 -/
theorem folded_rectangle_BC (r : FoldedRectangle) : r.BC = 5.2 := by
  sorry

end folded_rectangle_BC_l1120_112068


namespace min_snack_cost_l1120_112045

/-- Calculates the minimum number of packs/bags needed given the number of items per pack/bag and the total number of items required -/
def min_packs_needed (items_per_pack : ℕ) (total_items_needed : ℕ) : ℕ :=
  (total_items_needed + items_per_pack - 1) / items_per_pack

/-- Represents the problem of buying snacks for soccer players -/
def snack_problem (num_players : ℕ) (juice_per_pack : ℕ) (juice_pack_cost : ℚ) 
                  (apples_per_bag : ℕ) (apple_bag_cost : ℚ) : ℚ :=
  let juice_packs := min_packs_needed juice_per_pack num_players
  let apple_bags := min_packs_needed apples_per_bag num_players
  juice_packs * juice_pack_cost + apple_bags * apple_bag_cost

/-- The theorem stating the minimum amount Danny spends -/
theorem min_snack_cost : 
  snack_problem 17 3 2 5 4 = 28 :=
sorry

end min_snack_cost_l1120_112045


namespace system_solution_l1120_112082

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, 4*x - 3*y = k ∧ 2*x + 3*y = 5 ∧ x = y) → k = 1 := by
  sorry

end system_solution_l1120_112082


namespace gain_percent_calculation_l1120_112085

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 15 * S →
  (S - C) / C * 100 = 233.33 := by
sorry

end gain_percent_calculation_l1120_112085


namespace lending_interest_rate_l1120_112078

/-- Proves that the lending interest rate is 6% given the specified conditions --/
theorem lending_interest_rate (borrowed_amount : ℕ) (borrowing_period : ℕ) 
  (borrowing_rate : ℚ) (gain_per_year : ℕ) (lending_rate : ℚ) : 
  borrowed_amount = 6000 →
  borrowing_period = 2 →
  borrowing_rate = 4 / 100 →
  gain_per_year = 120 →
  (borrowed_amount * borrowing_rate * borrowing_period + 
   borrowing_period * gain_per_year) / (borrowed_amount * borrowing_period) * 100 = lending_rate →
  lending_rate = 6 / 100 := by
sorry


end lending_interest_rate_l1120_112078


namespace dividend_calculation_l1120_112074

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 65)
  (h_divisor : divisor = 24)
  (h_remainder : remainder = 5) :
  (divisor * quotient) + remainder = 1565 := by
sorry

end dividend_calculation_l1120_112074


namespace min_value_fraction_l1120_112039

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = 1) :
  (x + y) / (x * y) ≥ 9 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 1 ∧ (x + y) / (x * y) = 9 :=
by sorry

end min_value_fraction_l1120_112039


namespace x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two_l1120_112058

theorem x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two 
  (x : ℝ) (h : x + 1/x = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := by
sorry

end x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two_l1120_112058


namespace points_in_quadrant_I_l1120_112001

theorem points_in_quadrant_I (x y : ℝ) : y > 3*x ∧ y > 5 - 2*x → x > 0 ∧ y > 0 := by
  sorry

end points_in_quadrant_I_l1120_112001


namespace floor_calculation_l1120_112035

/-- The floor of (2011^3 / (2009 * 2010)) - (2009^3 / (2010 * 2011)) is 8 -/
theorem floor_calculation : 
  ⌊(2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011)⌋ = 8 := by
  sorry

end floor_calculation_l1120_112035


namespace perfect_cube_pair_solution_l1120_112060

theorem perfect_cube_pair_solution : ∀ a b : ℕ+,
  (∃ k : ℕ+, (a ^ 3 + 6 * a * b + 1 : ℕ) = k ^ 3) →
  (∃ m : ℕ+, (b ^ 3 + 6 * a * b + 1 : ℕ) = m ^ 3) →
  a = 1 ∧ b = 1 := by
sorry

end perfect_cube_pair_solution_l1120_112060


namespace prob_at_least_two_equals_result_l1120_112080

-- Define the probabilities for each person hitting the target
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.6

-- Define the probability of at least two people hitting the target
def prob_at_least_two : ℝ :=
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) -
  (prob_A * (1 - prob_B) * (1 - prob_C) +
   (1 - prob_A) * prob_B * (1 - prob_C) +
   (1 - prob_A) * (1 - prob_B) * prob_C)

-- Theorem statement
theorem prob_at_least_two_equals_result :
  prob_at_least_two = 0.832 := by sorry

end prob_at_least_two_equals_result_l1120_112080


namespace greatest_k_inequality_l1120_112044

theorem greatest_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 > b*c) :
  (a^2 - b*c)^2 ≥ 4*(b^2 - c*a)*(c^2 - a*b) := by
  sorry

end greatest_k_inequality_l1120_112044


namespace sample_size_is_100_l1120_112016

/-- Represents a city with its total sales and number of cars selected for investigation. -/
structure City where
  name : String
  totalSales : Nat
  selected : Nat

/-- Represents the sampling data for the car manufacturer's investigation. -/
def samplingData : List City :=
  [{ name := "A", totalSales := 420, selected := 30 },
   { name := "B", totalSales := 280, selected := 20 },
   { name := "C", totalSales := 700, selected := 50 }]

/-- Checks if the sampling is proportional to the total sales. -/
def isProportionalSampling (data : List City) : Prop :=
  ∀ i j, i ∈ data → j ∈ data → 
    i.totalSales * j.selected = j.totalSales * i.selected

/-- The total sample size is the sum of all selected cars. -/
def totalSampleSize (data : List City) : Nat :=
  (data.map (·.selected)).sum

/-- Theorem stating that the total sample size is 100 given the conditions. -/
theorem sample_size_is_100 (h : isProportionalSampling samplingData) :
  totalSampleSize samplingData = 100 := by
  sorry

end sample_size_is_100_l1120_112016


namespace price_change_theorem_l1120_112052

theorem price_change_theorem (initial_price : ℝ) (h : initial_price > 0) :
  let egg_price_new := initial_price * (1 - 0.02)
  let apple_price_new := initial_price * (1 + 0.10)
  let total_price_old := 2 * initial_price
  let total_price_new := egg_price_new + apple_price_new
  let price_increase := total_price_new - total_price_old
  let percentage_increase := price_increase / total_price_old * 100
  percentage_increase = 4 := by sorry

end price_change_theorem_l1120_112052


namespace problem_solution_l1120_112041

theorem problem_solution (x : ℝ) 
  (h : x * Real.sqrt (x^2 - 1) + 1 / (x + Real.sqrt (x^2 - 1)) = 21) :
  x^2 * Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 2 * x^2 :=
by sorry

end problem_solution_l1120_112041


namespace completing_square_quadratic_l1120_112046

theorem completing_square_quadratic : 
  ∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ (x - 3/4)^2 = 17/16 := by
  sorry

end completing_square_quadratic_l1120_112046


namespace cone_lateral_surface_area_l1120_112025

/-- The lateral surface area of a cone with base radius 3 and slant height 5 is 15π. -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 3 → l = 5 → π * r * l = 15 * π := by
  sorry

end cone_lateral_surface_area_l1120_112025


namespace triangle_translation_l1120_112069

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem triangle_translation (a b : ℝ) :
  let A : Point := { x := -1, y := 2 }
  let B : Point := { x := 1, y := -1 }
  let C : Point := { x := 2, y := 1 }
  let A' : Point := { x := -3, y := a }
  let B' : Point := { x := b, y := 3 }
  let t : Translation := { dx := A'.x - A.x, dy := B'.y - B.y }
  applyTranslation C t = { x := 0, y := 5 } := by
  sorry

end triangle_translation_l1120_112069


namespace product_of_x_and_y_l1120_112011

theorem product_of_x_and_y (x y : ℝ) : 
  3 * x + 4 * y = 60 → 6 * x - 4 * y = 12 → x * y = 72 := by
  sorry

end product_of_x_and_y_l1120_112011


namespace grass_withering_is_certain_event_l1120_112020

/-- An event that occurs regularly and predictably every year -/
structure AnnualEvent where
  occurs_yearly : Bool
  predictable : Bool

/-- Definition of a certain event in probability theory -/
def CertainEvent (e : AnnualEvent) : Prop :=
  e.occurs_yearly ∧ e.predictable

/-- The withering of grass on a plain as described in the poem -/
def grass_withering : AnnualEvent :=
  { occurs_yearly := true
  , predictable := true }

/-- Theorem stating that the grass withering is a certain event -/
theorem grass_withering_is_certain_event : CertainEvent grass_withering := by
  sorry

end grass_withering_is_certain_event_l1120_112020


namespace equation_d_is_quadratic_l1120_112003

-- Define a quadratic equation
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation from Option D
def equation_d (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

-- Theorem stating that equation_d is a quadratic equation
theorem equation_d_is_quadratic : is_quadratic_equation equation_d :=
sorry

end equation_d_is_quadratic_l1120_112003


namespace friend_meeting_distance_l1120_112014

theorem friend_meeting_distance (trail_length : ℝ) (rate_difference : ℝ) : 
  trail_length = 36 → rate_difference = 0.25 → 
  let faster_friend_distance : ℝ := trail_length * (1 + rate_difference) / (2 + rate_difference)
  faster_friend_distance = 20 := by
sorry

end friend_meeting_distance_l1120_112014


namespace value_of_expression_l1120_112053

theorem value_of_expression (x : ℝ) (h : x^2 - 3*x - 12 = 0) : 3*x^2 - 9*x + 5 = 41 := by
  sorry

end value_of_expression_l1120_112053


namespace euro_calculation_l1120_112086

-- Define the € operation
def euro (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem euro_calculation : euro 7 (euro 4 5) = 560 := by
  sorry

end euro_calculation_l1120_112086


namespace triangle_side_inequality_l1120_112049

theorem triangle_side_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) < 2 := by
  sorry

end triangle_side_inequality_l1120_112049


namespace bobby_shoes_count_l1120_112017

theorem bobby_shoes_count (bonny_shoes becky_shoes bobby_shoes : ℕ) : 
  bonny_shoes = 13 →
  bonny_shoes = 2 * becky_shoes - 5 →
  bobby_shoes = 3 * becky_shoes →
  bobby_shoes = 27 := by
sorry

end bobby_shoes_count_l1120_112017


namespace order_of_expressions_l1120_112043

theorem order_of_expressions :
  let x : ℝ := Real.exp (-1/2)
  let y : ℝ := (Real.log 2) / (Real.log 5)
  let z : ℝ := Real.log 3
  y < x ∧ x < z := by sorry

end order_of_expressions_l1120_112043


namespace school_attendance_l1120_112030

/-- Calculates the number of years a student attends school given the cost per semester,
    number of semesters per year, and total cost. -/
def years_of_school (cost_per_semester : ℕ) (semesters_per_year : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (cost_per_semester * semesters_per_year)

/-- Theorem stating that given the specific costs and duration, the student attends 13 years of school. -/
theorem school_attendance : years_of_school 20000 2 520000 = 13 := by
  sorry

end school_attendance_l1120_112030


namespace initial_water_amount_l1120_112066

/-- The amount of water initially in the bucket, in gallons. -/
def initial_water : ℝ := sorry

/-- The amount of water remaining in the bucket, in gallons. -/
def remaining_water : ℝ := 0.5

/-- The amount of water that leaked out of the bucket, in gallons. -/
def leaked_water : ℝ := 0.25

/-- Theorem stating that the initial amount of water is equal to 0.75 gallon. -/
theorem initial_water_amount : initial_water = 0.75 := by sorry

end initial_water_amount_l1120_112066


namespace sandbox_area_l1120_112019

-- Define the sandbox dimensions
def sandbox_length : ℝ := 312
def sandbox_width : ℝ := 146

-- State the theorem
theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end sandbox_area_l1120_112019


namespace tennis_ball_ratio_l1120_112007

/-- The number of tennis balls originally ordered -/
def total_balls : ℕ := 114

/-- The number of extra yellow balls sent by mistake -/
def extra_yellow : ℕ := 50

/-- The number of white balls received -/
def white_balls : ℕ := total_balls / 2

/-- The number of yellow balls received -/
def yellow_balls : ℕ := total_balls / 2 + extra_yellow

/-- The ratio of white balls to yellow balls after the error -/
def ball_ratio : ℚ := white_balls / yellow_balls

theorem tennis_ball_ratio : ball_ratio = 57 / 107 := by sorry

end tennis_ball_ratio_l1120_112007


namespace parabola_tangent_circle_problem_l1120_112013

-- Define the parabola T₀: y = x²
def T₀ (x : ℝ) : ℝ := x^2

-- Define point P
def P : ℝ × ℝ := (1, -1)

-- Define the tangent line passing through P and intersecting T₀
def tangent_line (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ 
  T₀ x₁ = (x₁ - 1) * 2 * x₁ + (-1) ∧
  T₀ x₂ = (x₂ - 1) * 2 * x₂ + (-1)

-- Define circle E with center at P and tangent to line MN
def circle_E (r : ℝ) : Prop :=
  r = (4 : ℝ) / Real.sqrt 5

-- Define chords AC and BD passing through origin and perpendicular in circle E
def chords_ABCD (d₁ d₂ : ℝ) : Prop :=
  d₁^2 + d₂^2 = 2

-- Main theorem
theorem parabola_tangent_circle_problem :
  ∃ (x₁ x₂ r d₁ d₂ : ℝ),
    tangent_line x₁ x₂ ∧
    circle_E r ∧
    chords_ABCD d₁ d₂ ∧
    x₁ = 1 - Real.sqrt 2 ∧
    x₂ = 1 + Real.sqrt 2 ∧
    r^2 * Real.pi = (16 : ℝ) / 5 ∧
    2 * r^2 - (d₁^2 + d₂^2) ≤ (22 : ℝ) / 5 :=
sorry

end parabola_tangent_circle_problem_l1120_112013


namespace stair_climbing_time_l1120_112023

theorem stair_climbing_time (n : ℕ) (a d : ℝ) (h : n = 7 ∧ a = 25 ∧ d = 10) :
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 385 :=
by sorry

end stair_climbing_time_l1120_112023


namespace imaginary_part_of_2_plus_i_l1120_112029

theorem imaginary_part_of_2_plus_i : Complex.im (2 + Complex.I) = 1 := by
  sorry

end imaginary_part_of_2_plus_i_l1120_112029


namespace minimum_groups_l1120_112006

/-- A function that determines if a number belongs to the set G_k -/
def in_G_k (n : ℕ) (k : ℕ) : Prop :=
  n % 6 = k ∧ 1 ≤ n ∧ n ≤ 600

/-- A function that checks if two numbers can be in the same group -/
def can_be_in_same_group (a b : ℕ) : Prop :=
  (a + b) % 6 = 0

/-- A valid grouping of numbers -/
def valid_grouping (groups : List (List ℕ)) : Prop :=
  (∀ group ∈ groups, ∀ a ∈ group, ∀ b ∈ group, a ≠ b → can_be_in_same_group a b) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 600 → ∃ group ∈ groups, n ∈ group)

theorem minimum_groups :
  ∃ (groups : List (List ℕ)), valid_grouping groups ∧
    (∀ (other_groups : List (List ℕ)), valid_grouping other_groups →
      groups.length ≤ other_groups.length) ∧
    groups.length = 202 := by
  sorry

end minimum_groups_l1120_112006


namespace winning_pair_probability_l1120_112022

/-- Represents the color of a card -/
inductive Color
| Blue
| Purple

/-- Represents the letter on a card -/
inductive Letter
| A | B | C | D | E | F

/-- Represents a card with a color and a letter -/
structure Card where
  color : Color
  letter : Letter

/-- The deck of cards -/
def deck : List Card := sorry

/-- Checks if two cards form a winning pair -/
def is_winning_pair (c1 c2 : Card) : Bool := sorry

/-- Calculates the probability of drawing a winning pair -/
def probability_winning_pair : ℚ := sorry

/-- Theorem stating the probability of drawing a winning pair -/
theorem winning_pair_probability : 
  probability_winning_pair = 29 / 45 := by sorry

end winning_pair_probability_l1120_112022


namespace min_value_of_arithmetic_geometric_seq_l1120_112064

/-- A positive arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = a k * q

theorem min_value_of_arithmetic_geometric_seq
  (a : ℕ → ℝ)
  (h_seq : ArithGeomSeq a)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 3 / 2) := by
  sorry

end min_value_of_arithmetic_geometric_seq_l1120_112064


namespace problem_solution_l1120_112051

theorem problem_solution (x y z w : ℝ) 
  (h1 : x * w > 0)
  (h2 : y * z > 0)
  (h3 : 1 / x + 1 / w = 20)
  (h4 : 1 / y + 1 / z = 25)
  (h5 : 1 / (x * w) = 6)
  (h6 : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 := by
  sorry

end problem_solution_l1120_112051


namespace complex_magnitude_l1120_112057

theorem complex_magnitude (z : ℂ) : z = -2 - I → Complex.abs (z + I) = 2 := by
  sorry

end complex_magnitude_l1120_112057


namespace find_A_l1120_112090

theorem find_A : ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ A * 100 + 30 + B - 41 = 591 ∧ A = 6 := by
  sorry

end find_A_l1120_112090


namespace boys_playing_neither_l1120_112075

/-- Given a group of boys with information about their sports participation,
    calculate the number of boys who play neither basketball nor football. -/
theorem boys_playing_neither (total : ℕ) (basketball : ℕ) (football : ℕ) (both : ℕ)
    (h_total : total = 22)
    (h_basketball : basketball = 13)
    (h_football : football = 15)
    (h_both : both = 18) :
    total - (basketball + football - both) = 12 := by
  sorry

end boys_playing_neither_l1120_112075


namespace triangle_midpoints_x_sum_l1120_112027

theorem triangle_midpoints_x_sum (p q r : ℝ) : 
  p + q + r = 15 → 
  (p + q) / 2 + (q + r) / 2 + (r + p) / 2 = 15 := by
  sorry

end triangle_midpoints_x_sum_l1120_112027


namespace scale_division_l1120_112031

/-- Represents a length in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ

/-- Converts a Length to total inches -/
def Length.to_inches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- Converts total inches to a Length -/
def inches_to_length (total_inches : ℕ) : Length :=
  { feet := total_inches / 12, inches := total_inches % 12 }

theorem scale_division (scale : Length) (parts : ℕ) 
    (h1 : scale.feet = 6 ∧ scale.inches = 8) 
    (h2 : parts = 4) : 
  inches_to_length (scale.to_inches / parts) = { feet := 1, inches := 8 } := by
sorry

end scale_division_l1120_112031


namespace stirling_bounds_l1120_112038

-- Define e as the limit of (1 + 1/n)^n as n approaches infinity
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem stirling_bounds (n : ℕ) (h : n > 6) :
  (n / e : ℝ)^n < n! ∧ (n! : ℝ) < n * (n / e)^n :=
sorry

end stirling_bounds_l1120_112038


namespace sum_reciprocals_inequality_l1120_112004

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

end sum_reciprocals_inequality_l1120_112004


namespace cloth_cost_price_theorem_l1120_112061

/-- Calculates the cost price per meter of cloth given the total meters sold,
    the total selling price, and the profit per meter. -/
def costPricePerMeter (totalMeters : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ) : ℕ :=
  (sellingPrice - profitPerMeter * totalMeters) / totalMeters

/-- Proves that given the specified conditions, the cost price per meter of cloth is 95 Rs. -/
theorem cloth_cost_price_theorem (totalMeters sellingPrice profitPerMeter : ℕ)
    (h1 : totalMeters = 85)
    (h2 : sellingPrice = 8925)
    (h3 : profitPerMeter = 10) :
    costPricePerMeter totalMeters sellingPrice profitPerMeter = 95 := by
  sorry

#eval costPricePerMeter 85 8925 10

end cloth_cost_price_theorem_l1120_112061


namespace greatest_three_digit_number_l1120_112040

theorem greatest_three_digit_number : ∃ (n : ℕ), 
  n = 970 ∧ 
  n < 1000 ∧ 
  n ≥ 100 ∧ 
  ∃ (k : ℕ), n = 8 * k + 2 ∧ 
  ∃ (m : ℕ), n = 7 * m + 4 ∧ 
  ∀ (x : ℕ), x < 1000 ∧ x ≥ 100 ∧ (∃ (a : ℕ), x = 8 * a + 2) ∧ (∃ (b : ℕ), x = 7 * b + 4) → x ≤ n :=
by sorry

end greatest_three_digit_number_l1120_112040


namespace circle_equation_proof_l1120_112024

theorem circle_equation_proof (x y : ℝ) : 
  let center := (1, -2)
  let radius := Real.sqrt 2
  let circle_eq := (x - 1)^2 + (y + 2)^2 = 2
  let center_line_eq := -2 * center.1 = center.2
  let tangent_line_eq := x + y = 1
  let tangent_point := (2, -1)
  (
    center_line_eq ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ circle_eq ∧
    (center.1 - tangent_point.1)^2 + (center.2 - tangent_point.2)^2 = radius^2 ∧
    (tangent_point.1 + tangent_point.2 = 1)
  ) := by sorry

end circle_equation_proof_l1120_112024


namespace students_facing_teacher_l1120_112010

theorem students_facing_teacher (n : ℕ) (h : n = 50) : 
  n - (n / 3 + n / 7 - n / 21) = 31 :=
sorry

end students_facing_teacher_l1120_112010


namespace right_triangles_AF_length_l1120_112083

theorem right_triangles_AF_length 
  (AB DE CD EF BC : ℝ)
  (h1 : AB = 12)
  (h2 : DE = 12)
  (h3 : CD = 8)
  (h4 : EF = 8)
  (h5 : BC = 5)
  (h6 : AB^2 + BC^2 = AC^2)  -- ABC is a right triangle
  (h7 : AC^2 + CD^2 = AD^2)  -- ACD is a right triangle
  (h8 : AD^2 + DE^2 = AE^2)  -- ADE is a right triangle
  (h9 : AE^2 + EF^2 = AF^2)  -- AEF is a right triangle
  : AF = 21 := by
    sorry

end right_triangles_AF_length_l1120_112083


namespace fixed_fee_is_7_42_l1120_112095

/-- Represents the monthly bill structure for an online service provider -/
structure Bill where
  fixed_fee : ℝ
  connect_time_charge : ℝ
  data_usage_charge_per_gb : ℝ

/-- The December bill without data usage -/
def december_bill (b : Bill) : ℝ :=
  b.fixed_fee + b.connect_time_charge

/-- The January bill with 3 GB data usage -/
def january_bill (b : Bill) : ℝ :=
  b.fixed_fee + b.connect_time_charge + 3 * b.data_usage_charge_per_gb

/-- Theorem stating that the fixed monthly fee is $7.42 -/
theorem fixed_fee_is_7_42 (b : Bill) : b.fixed_fee = 7.42 :=
  by
  have h1 : december_bill b = 18.50 := by sorry
  have h2 : january_bill b = 23.45 := by sorry
  have h3 : january_bill b - december_bill b = 3 * b.data_usage_charge_per_gb := by sorry
  sorry

end fixed_fee_is_7_42_l1120_112095


namespace jason_music_store_spending_l1120_112062

/-- The cost of Jason's flute -/
def flute_cost : ℚ := 142.46

/-- The cost of Jason's music tool -/
def music_tool_cost : ℚ := 8.89

/-- The cost of Jason's song book -/
def song_book_cost : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_cost + music_tool_cost + song_book_cost

theorem jason_music_store_spending :
  total_spent = 158.35 := by sorry

end jason_music_store_spending_l1120_112062


namespace tangent_line_to_logarithmic_curve_l1120_112089

theorem tangent_line_to_logarithmic_curve (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * x = 1 + Real.log x ∧ 
    ∀ y : ℝ, y > 0 → a * y ≤ 1 + Real.log y) → 
  a = 1 := by
sorry

end tangent_line_to_logarithmic_curve_l1120_112089


namespace last_letter_151st_permutation_l1120_112026

-- Define the word and its permutations
def word : String := "JOKING"
def num_permutations : Nat := 720

-- Define the dictionary order function (not implemented, just declared)
def dictionary_order (s1 s2 : String) : Bool :=
  sorry

-- Define a function to get the nth permutation in dictionary order
def nth_permutation (n : Nat) : String :=
  sorry

-- Define a function to get the last letter of a string
def last_letter (s : String) : Char :=
  sorry

-- The theorem to prove
theorem last_letter_151st_permutation :
  last_letter (nth_permutation 151) = 'O' :=
sorry

end last_letter_151st_permutation_l1120_112026


namespace quadratic_problem_l1120_112065

def quadratic_function (a b x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 + b

theorem quadratic_problem (a b : ℤ) (h1 : a ≠ 0) (h2 : a > 0) 
  (h3 : 4 < a + |b| ∧ a + |b| < 9) 
  (h4 : quadratic_function a b 1 = 3) :
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), quadratic_function a b (x - y) = quadratic_function a b (x + y)) ∧
  (a = 2 ∧ b = 6) ∧
  (∃ (t : ℝ), (t = 1/2 ∨ t = 5/2) ∧
    ∀ (x : ℝ), t ≤ x ∧ x ≤ t + 1 → quadratic_function a b x ≥ 3/2 ∧
    ∃ (x₀ : ℝ), t ≤ x₀ ∧ x₀ ≤ t + 1 ∧ quadratic_function a b x₀ = 3/2) :=
by sorry

end quadratic_problem_l1120_112065


namespace prob_rolling_six_is_five_thirty_sixths_l1120_112098

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to roll a sum of 6 with two dice -/
def waysToRollSix : ℕ := 5

/-- The probability of rolling a sum of 6 with two fair dice -/
def probRollingSix : ℚ := waysToRollSix / totalOutcomes

theorem prob_rolling_six_is_five_thirty_sixths :
  probRollingSix = 5 / 36 := by
  sorry

end prob_rolling_six_is_five_thirty_sixths_l1120_112098


namespace temperature_at_night_l1120_112008

/-- Given the temperature changes throughout a day, prove the final temperature at night. -/
theorem temperature_at_night 
  (noon_temp : ℤ) 
  (afternoon_temp : ℤ) 
  (temp_drop : ℤ) 
  (h1 : noon_temp = 5)
  (h2 : afternoon_temp = 7)
  (h3 : temp_drop = 9) : 
  afternoon_temp - temp_drop = -2 := by
  sorry

end temperature_at_night_l1120_112008


namespace greatest_integer_satisfying_inequality_l1120_112036

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℕ), x > 0 ∧ (x^4 : ℚ) / (x^2 : ℚ) < 12 ∧
  ∀ (y : ℕ), y > x → (y^4 : ℚ) / (y^2 : ℚ) ≥ 12 :=
by
  -- The proof goes here
  sorry

end greatest_integer_satisfying_inequality_l1120_112036


namespace bus_seating_capacity_bus_total_capacity_l1120_112063

/-- The number of people that can sit in a bus given the seating arrangement --/
theorem bus_seating_capacity 
  (left_seats : ℕ) 
  (right_seats_difference : ℕ) 
  (people_per_seat : ℕ) 
  (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seats_difference
  let left_capacity := left_seats * people_per_seat
  let right_capacity := right_seats * people_per_seat
  left_capacity + right_capacity + back_seat_capacity

/-- The total number of people that can sit in the bus is 90 --/
theorem bus_total_capacity : 
  bus_seating_capacity 15 3 3 9 = 90 := by
  sorry

end bus_seating_capacity_bus_total_capacity_l1120_112063


namespace projection_matrix_values_l1120_112071

/-- A 2x2 matrix is a projection matrix if and only if Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The specific form of our matrix Q -/
def Q (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 20/49; c, 29/49]

theorem projection_matrix_values :
  ∀ a c : ℚ, is_projection_matrix (Q a c) → a = 20/49 ∧ c = 29/49 := by
  sorry

end projection_matrix_values_l1120_112071


namespace fraction_meaningful_l1120_112037

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 2 / x) ↔ x ≠ 0 := by sorry

end fraction_meaningful_l1120_112037


namespace functional_equation_solution_l1120_112070

-- Define the property that the function f must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x^2) :=
by sorry

end functional_equation_solution_l1120_112070


namespace perpendicular_line_theorem_l1120_112005

structure Plane where
  -- Define a plane structure

structure Point where
  -- Define a point structure

structure Line where
  -- Define a line structure

-- Define perpendicularity between planes
def perpendicular (α β : Plane) : Prop := sorry

-- Define a point lying in a plane
def lies_in (A : Point) (α : Plane) : Prop := sorry

-- Define a line passing through a point
def passes_through (l : Line) (A : Point) : Prop := sorry

-- Define a line perpendicular to a plane
def perpendicular_to_plane (l : Line) (β : Plane) : Prop := sorry

-- Define a line lying in a plane
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

theorem perpendicular_line_theorem (α β : Plane) (A : Point) :
  perpendicular α β →
  lies_in A α →
  ∃! l : Line, passes_through l A ∧ perpendicular_to_plane l β ∧ line_in_plane l α :=
sorry

end perpendicular_line_theorem_l1120_112005


namespace team_a_championship_probability_l1120_112067

-- Define the game state
structure GameState where
  team_a_wins_needed : ℕ
  team_b_wins_needed : ℕ

-- Define the probability of Team A winning
def prob_team_a_wins (state : GameState) : ℚ :=
  if state.team_a_wins_needed = 0 then 1
  else if state.team_b_wins_needed = 0 then 0
  else sorry

-- Theorem statement
theorem team_a_championship_probability :
  let initial_state : GameState := ⟨1, 2⟩
  prob_team_a_wins initial_state = 3/4 :=
sorry

end team_a_championship_probability_l1120_112067


namespace inequality_condition_l1120_112096

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ a ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ) := by
  sorry

end inequality_condition_l1120_112096


namespace photograph_perimeter_l1120_112087

theorem photograph_perimeter (w h m : ℝ) 
  (area_with_2inch_border : (w + 4) * (h + 4) = m)
  (area_with_4inch_border : (w + 8) * (h + 8) = m + 94)
  : 2 * (w + h) = 23 := by
  sorry

end photograph_perimeter_l1120_112087


namespace camel_cost_calculation_l1120_112073

/-- The cost of a camel in rupees -/
def camel_cost : ℝ := 4184.62

/-- The cost of a horse in rupees -/
def horse_cost : ℝ := 1743.59

/-- The cost of an ox in rupees -/
def ox_cost : ℝ := 11333.33

/-- The cost of an elephant in rupees -/
def elephant_cost : ℝ := 17000

theorem camel_cost_calculation :
  (10 * camel_cost = 24 * horse_cost) ∧
  (26 * horse_cost = 4 * ox_cost) ∧
  (6 * ox_cost = 4 * elephant_cost) ∧
  (10 * elephant_cost = 170000) →
  camel_cost = 4184.62 := by
sorry

#eval camel_cost

end camel_cost_calculation_l1120_112073


namespace new_vessel_capacity_l1120_112048

/-- Given two vessels with different alcohol concentrations, prove the capacity of a new vessel -/
theorem new_vessel_capacity
  (vessel1_capacity : ℝ) (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ) (new_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 0.3)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 0.45)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 0.33) :
  (vessel1_capacity * vessel1_alcohol_percentage + vessel2_capacity * vessel2_alcohol_percentage) / new_concentration = 10 := by
sorry

end new_vessel_capacity_l1120_112048


namespace geometric_sequence_common_ratio_l1120_112097

/-- A geometric sequence with the given first four terms has a common ratio of -3/2 -/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℚ), 
    a 0 = 32 ∧ a 1 = -48 ∧ a 2 = 72 ∧ a 3 = -108 →
    ∃ (r : ℚ), r = -3/2 ∧ ∀ n, a (n + 1) = r * a n :=
by sorry

end geometric_sequence_common_ratio_l1120_112097
