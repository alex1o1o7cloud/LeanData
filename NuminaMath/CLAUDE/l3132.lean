import Mathlib

namespace division_problem_l3132_313244

theorem division_problem (x y : ℤ) (hx : x > 0) : 
  (∃ q : ℤ, x = 11 * y + 4 ∧ q * 11 + 4 = x) →
  (∃ q : ℤ, 2 * x = 6 * (3 * y) + 1 ∧ q * 6 + 1 = 2 * x) →
  7 * y - x = 3 := by
  sorry

end division_problem_l3132_313244


namespace park_area_l3132_313250

/-- The area of a rectangular park with a given length-to-breadth ratio and perimeter -/
theorem park_area (length breadth perimeter : ℝ) : 
  length > 0 →
  breadth > 0 →
  length / breadth = 1 / 3 →
  perimeter = 2 * (length + breadth) →
  length * breadth = 30000 := by
  sorry

#check park_area

end park_area_l3132_313250


namespace max_blocks_in_box_l3132_313239

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

def box : Dimensions := ⟨3, 4, 3⟩
def block : Dimensions := ⟨3, 1, 1⟩

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := volume box / volume block

theorem max_blocks_in_box : max_blocks = 12 := by sorry

end max_blocks_in_box_l3132_313239


namespace probability_all_sweet_is_one_sixth_l3132_313281

def total_oranges : ℕ := 10
def sweet_oranges : ℕ := 6
def picked_oranges : ℕ := 3

def probability_all_sweet : ℚ :=
  (sweet_oranges.choose picked_oranges) / (total_oranges.choose picked_oranges)

theorem probability_all_sweet_is_one_sixth :
  probability_all_sweet = 1 / 6 := by sorry

end probability_all_sweet_is_one_sixth_l3132_313281


namespace length_width_ratio_l3132_313257

-- Define the rectangle
def rectangle (width : ℝ) (length : ℝ) : Prop :=
  width > 0 ∧ length > 0

-- Define the area of the rectangle
def area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

-- Theorem statement
theorem length_width_ratio (width : ℝ) (length : ℝ) :
  rectangle width length →
  width = 6 →
  area width length = 108 →
  length / width = 3 := by
  sorry


end length_width_ratio_l3132_313257


namespace sqrt_five_squared_times_seven_sixth_power_l3132_313255

theorem sqrt_five_squared_times_seven_sixth_power : 
  Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_five_squared_times_seven_sixth_power_l3132_313255


namespace range_of_a_l3132_313204

theorem range_of_a (a : ℝ) : 
  a < 9 * a^3 - 11 * a ∧ 9 * a^3 - 11 * a < |a| → 
  a ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-Real.sqrt 10 / 3) := by
  sorry

end range_of_a_l3132_313204


namespace magnitude_a_minus_2b_l3132_313237

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-1, 4)  -- Derived from a + b = (0, 2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := vector_a
  let b : ℝ × ℝ := vector_b
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 109 := by
  sorry

end magnitude_a_minus_2b_l3132_313237


namespace function_shape_is_graph_l3132_313283

/-- A function from real numbers to real numbers -/
def RealFunction := ℝ → ℝ

/-- A point in the Cartesian coordinate system -/
def CartesianPoint := ℝ × ℝ

/-- The set of all points representing a function in the Cartesian coordinate system -/
def FunctionPoints (f : RealFunction) : Set CartesianPoint :=
  {p : CartesianPoint | ∃ x : ℝ, p = (x, f x)}

/-- The graph of a function is the set of all points representing that function -/
def Graph (f : RealFunction) : Set CartesianPoint := FunctionPoints f

/-- Theorem: The shape formed by all points plotted in the Cartesian coordinate system 
    that represent a function is called the graph of the function -/
theorem function_shape_is_graph (f : RealFunction) : 
  FunctionPoints f = Graph f := by sorry

end function_shape_is_graph_l3132_313283


namespace distance_maximized_at_neg_one_l3132_313274

/-- The point P -/
def P : ℝ × ℝ := (3, 2)

/-- The point Q -/
def Q : ℝ × ℝ := (2, 1)

/-- The line equation: mx - y + 1 - 2m = 0 -/
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - 2 * m = 0

/-- The line passes through point Q for all m -/
axiom line_through_Q (m : ℝ) : line_equation m Q.1 Q.2

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem distance_maximized_at_neg_one :
  ∃ (max_dist : ℝ), ∀ (m : ℝ),
    distance_to_line P m ≤ max_dist ∧
    distance_to_line P (-1) = max_dist :=
  sorry

end distance_maximized_at_neg_one_l3132_313274


namespace y_derivative_l3132_313211

-- Define the function
noncomputable def y (x : ℝ) : ℝ := 
  -(Real.sinh x) / (2 * (Real.cosh x)^2) + (3/2) * Real.arcsin (Real.tanh x)

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = Real.cosh (2*x) / (Real.cosh x)^3 := by sorry

end y_derivative_l3132_313211


namespace pure_imaginary_magnitude_l3132_313280

theorem pure_imaginary_magnitude (a : ℝ) : 
  (((a - 2 * Complex.I) / (1 + Complex.I)).re = 0) → 
  Complex.abs (1 + a * Complex.I) = Real.sqrt 5 := by
sorry

end pure_imaginary_magnitude_l3132_313280


namespace largest_n_for_sin_cos_inequality_l3132_313258

open Real

theorem largest_n_for_sin_cos_inequality :
  ∃ (n : ℕ), n = 3 ∧
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → sin x ^ n + cos x ^ n > 1 / 2) ∧
  ¬(∀ x : ℝ, 0 < x ∧ x < π / 2 → sin x ^ (n + 1) + cos x ^ (n + 1) > 1 / 2) :=
by sorry

end largest_n_for_sin_cos_inequality_l3132_313258


namespace steak_cost_solution_l3132_313247

/-- The cost of a steak given the conditions of the problem -/
def steak_cost : ℝ → Prop := λ s =>
  let drink_cost : ℝ := 5
  let tip_paid : ℝ := 8
  let tip_percentage : ℝ := 0.2
  let tip_coverage : ℝ := 0.8
  let total_meal_cost : ℝ := 2 * s + 2 * drink_cost
  tip_paid = tip_coverage * tip_percentage * total_meal_cost ∧ s = 20

theorem steak_cost_solution :
  ∃ s : ℝ, steak_cost s :=
sorry

end steak_cost_solution_l3132_313247


namespace tenth_term_of_arithmetic_sequence_l3132_313216

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  a 10 = 18 := by
sorry

end tenth_term_of_arithmetic_sequence_l3132_313216


namespace simplest_form_expression_l3132_313223

theorem simplest_form_expression (x y a : ℝ) (h : x ≠ 2) : 
  (∀ k : ℝ, k ≠ 0 → (1 : ℝ) / (x - 2) ≠ k * (1 : ℝ) / (x - 2)) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ (x^2 * y) / (2 * x) = k * (x * y) / 2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ (2 * a) / 8 = k * a / 4) :=
sorry

end simplest_form_expression_l3132_313223


namespace negation_of_square_sum_nonnegative_l3132_313285

theorem negation_of_square_sum_nonnegative :
  (¬ ∀ x y : ℝ, x^2 + y^2 ≥ 0) ↔ (∃ x y : ℝ, x^2 + y^2 < 0) :=
by sorry

end negation_of_square_sum_nonnegative_l3132_313285


namespace p_sufficient_but_not_necessary_for_r_l3132_313210

-- Define the propositions
variable (p q r : Prop)

-- Define what it means for a condition to be sufficient but not necessary
def sufficient_but_not_necessary (a b : Prop) : Prop :=
  (a → b) ∧ ¬(b → a)

-- Define what it means for a condition to be necessary but not sufficient
def necessary_but_not_sufficient (a b : Prop) : Prop :=
  (b → a) ∧ ¬(a → b)

-- State the theorem
theorem p_sufficient_but_not_necessary_for_r
  (h1 : sufficient_but_not_necessary p q)
  (h2 : necessary_but_not_sufficient r q) :
  sufficient_but_not_necessary p r :=
sorry

end p_sufficient_but_not_necessary_for_r_l3132_313210


namespace y_minus_3x_equals_7_l3132_313298

theorem y_minus_3x_equals_7 (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 := by
  sorry

end y_minus_3x_equals_7_l3132_313298


namespace smallest_marble_count_l3132_313265

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Calculates the probability of selecting a specific combination of marbles -/
def probability (mc : MarbleCount) (red white blue green : ℕ) : ℚ :=
  (mc.red.choose red * mc.white.choose white * mc.blue.choose blue * mc.green.choose green : ℚ) /
  (total_marbles mc).choose 5

/-- Checks if all specified probabilities are equal -/
def probabilities_equal (mc : MarbleCount) : Prop :=
  probability mc 5 0 0 0 = probability mc 3 2 0 0 ∧
  probability mc 3 2 0 0 = probability mc 1 2 2 0 ∧
  probability mc 1 2 2 0 = probability mc 2 1 1 1

/-- The theorem stating that the smallest number of marbles satisfying the conditions is 24 -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), probabilities_equal mc ∧ total_marbles mc = 24 ∧
  (∀ (mc' : MarbleCount), probabilities_equal mc' → total_marbles mc' ≥ 24) :=
sorry

end smallest_marble_count_l3132_313265


namespace john_total_distance_l3132_313249

-- Define the driving segments
def segment1_speed : ℝ := 55
def segment1_time : ℝ := 2.5
def segment2_speed : ℝ := 65
def segment2_time : ℝ := 3.25
def segment3_speed : ℝ := 50
def segment3_time : ℝ := 4

-- Define the total distance function
def total_distance : ℝ :=
  segment1_speed * segment1_time +
  segment2_speed * segment2_time +
  segment3_speed * segment3_time

-- Theorem statement
theorem john_total_distance :
  total_distance = 548.75 := by
  sorry

end john_total_distance_l3132_313249


namespace base6_210_equals_base4_1032_l3132_313243

-- Define a function to convert a base 6 number to base 10
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

-- Define a function to convert a base 10 number to base 4
def base10ToBase4 (n : ℕ) : ℕ :=
  (n / 64) * 1000 + ((n / 16) % 4) * 100 + ((n / 4) % 4) * 10 + (n % 4)

-- Theorem statement
theorem base6_210_equals_base4_1032 :
  base10ToBase4 (base6ToBase10 210) = 1032 :=
sorry

end base6_210_equals_base4_1032_l3132_313243


namespace min_buses_second_group_l3132_313291

theorem min_buses_second_group 
  (total_students : ℕ) 
  (bus_capacity : ℕ) 
  (max_buses_first_group : ℕ) 
  (min_buses_second_group : ℕ) : 
  total_students = 550 → 
  bus_capacity = 45 → 
  max_buses_first_group = 8 → 
  min_buses_second_group = 5 → 
  (max_buses_first_group * bus_capacity + min_buses_second_group * bus_capacity ≥ total_students) ∧
  ((min_buses_second_group - 1) * bus_capacity < total_students - max_buses_first_group * bus_capacity) :=
by
  sorry

#check min_buses_second_group

end min_buses_second_group_l3132_313291


namespace girls_in_class_l3132_313215

theorem girls_in_class (total_students : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total_students = 20) (h2 : girl_ratio = 2) (h3 : boy_ratio = 3) : 
  (girl_ratio * total_students) / (girl_ratio + boy_ratio) = 8 := by
sorry

end girls_in_class_l3132_313215


namespace power_division_l3132_313253

theorem power_division (n : ℕ) : n = 3^4053 → n / 3^2 = 3^4051 := by
  sorry

end power_division_l3132_313253


namespace xy_equals_one_l3132_313251

theorem xy_equals_one (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 25) (h4 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 := by
  sorry

end xy_equals_one_l3132_313251


namespace a_less_than_reciprocal_relationship_l3132_313275

theorem a_less_than_reciprocal_relationship (a : ℝ) :
  (a < -1 → a < 1/a) ∧ ¬(a < 1/a → a < -1) :=
by sorry

end a_less_than_reciprocal_relationship_l3132_313275


namespace smaller_cuboid_width_l3132_313222

/-- Proves that the width of smaller cuboids is 6 meters given the dimensions of the original cuboid,
    the length and height of smaller cuboids, and the number of smaller cuboids. -/
theorem smaller_cuboid_width
  (original_length : ℝ)
  (original_width : ℝ)
  (original_height : ℝ)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : original_length = 18)
  (h2 : original_width = 15)
  (h3 : original_height = 2)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small_cuboids = 6) :
  ∃ (small_width : ℝ), small_width = 6 ∧
    original_length * original_width * original_height =
    num_small_cuboids * small_length * small_width * small_height :=
by sorry

end smaller_cuboid_width_l3132_313222


namespace rogers_money_l3132_313267

/-- Roger's money calculation -/
theorem rogers_money (initial_amount spent_amount received_amount : ℕ) :
  initial_amount = 45 →
  spent_amount = 20 →
  received_amount = 46 →
  initial_amount - spent_amount + received_amount = 71 := by
  sorry

end rogers_money_l3132_313267


namespace ladybugs_with_spots_count_l3132_313230

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := 67082

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := total_ladybugs - ladybugs_without_spots

theorem ladybugs_with_spots_count : ladybugs_with_spots = 12170 := by
  sorry

end ladybugs_with_spots_count_l3132_313230


namespace prob_non_expired_single_draw_prob_expired_two_draws_l3132_313238

/-- Represents the total number of bottles --/
def total_bottles : ℕ := 6

/-- Represents the number of expired bottles --/
def expired_bottles : ℕ := 2

/-- Represents the number of non-expired bottles --/
def non_expired_bottles : ℕ := total_bottles - expired_bottles

/-- Theorem for the probability of drawing a non-expired bottle in a single draw --/
theorem prob_non_expired_single_draw : 
  (non_expired_bottles : ℚ) / total_bottles = 2 / 3 := by sorry

/-- Theorem for the probability of drawing at least one expired bottle in two draws --/
theorem prob_expired_two_draws : 
  1 - (non_expired_bottles * (non_expired_bottles - 1) : ℚ) / (total_bottles * (total_bottles - 1)) = 3 / 5 := by sorry

end prob_non_expired_single_draw_prob_expired_two_draws_l3132_313238


namespace notebook_buyers_difference_l3132_313260

theorem notebook_buyers_difference (notebook_cost : ℕ) 
  (fifth_grade_total : ℕ) (fourth_grade_total : ℕ) 
  (fourth_grade_count : ℕ) :
  notebook_cost > 0 ∧ 
  notebook_cost * 100 ∣ fifth_grade_total ∧ 
  notebook_cost * 100 ∣ fourth_grade_total ∧
  fifth_grade_total = 210 ∧
  fourth_grade_total = 252 ∧
  fourth_grade_count = 28 ∧
  fourth_grade_count ≥ fourth_grade_total / (notebook_cost * 100) →
  (fourth_grade_total / (notebook_cost * 100)) - 
  (fifth_grade_total / (notebook_cost * 100)) = 2 :=
sorry

end notebook_buyers_difference_l3132_313260


namespace product_of_square_roots_l3132_313226

theorem product_of_square_roots (x y z : ℝ) (hx : x = 75) (hy : y = 48) (hz : z = 3) :
  Real.sqrt x * Real.sqrt y * Real.sqrt z = 60 * Real.sqrt 3 := by
  sorry

end product_of_square_roots_l3132_313226


namespace smallest_divisible_by_18_and_25_l3132_313262

theorem smallest_divisible_by_18_and_25 : Nat.lcm 18 25 = 450 := by
  sorry

end smallest_divisible_by_18_and_25_l3132_313262


namespace social_practice_problem_l3132_313233

/-- Represents the number of students -/
def num_students : ℕ := sorry

/-- Represents the number of 35-seat buses needed to exactly fit all students -/
def num_35_seat_buses : ℕ := sorry

/-- Represents the number of 55-seat buses needed -/
def num_55_seat_buses : ℕ := sorry

/-- Cost of renting a 35-seat bus -/
def cost_35_seat : ℕ := 320

/-- Cost of renting a 55-seat bus -/
def cost_55_seat : ℕ := 400

/-- Total number of buses to rent -/
def total_buses : ℕ := 4

/-- Maximum budget for bus rental -/
def max_budget : ℕ := 1500

/-- Theorem stating the conditions and the result to be proven -/
theorem social_practice_problem :
  num_students = 35 * num_35_seat_buses ∧
  num_students = 55 * num_55_seat_buses - 45 ∧
  num_55_seat_buses = num_35_seat_buses - 1 ∧
  num_students = 175 ∧
  ∃ (x y : ℕ), x + y = total_buses ∧
               x * cost_35_seat + y * cost_55_seat ≤ max_budget ∧
               x * cost_35_seat + y * cost_55_seat = 1440 :=
by sorry

end social_practice_problem_l3132_313233


namespace james_steak_purchase_l3132_313271

/-- Represents the buy one get one free deal -/
def buyOneGetOneFree (x : ℝ) : ℝ := 2 * x

/-- Represents the price per pound in dollars -/
def pricePerPound : ℝ := 15

/-- Represents the total amount James paid in dollars -/
def totalPaid : ℝ := 150

/-- Theorem stating that James bought 20 pounds of steaks -/
theorem james_steak_purchase :
  ∃ (x : ℝ), x > 0 ∧ x * pricePerPound = totalPaid ∧ buyOneGetOneFree x = 20 :=
by
  sorry


end james_steak_purchase_l3132_313271


namespace shirt_profit_theorem_l3132_313268

/-- Represents the daily profit function for a shirt department -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

theorem shirt_profit_theorem 
  (initial_sales : ℕ) 
  (initial_profit : ℝ) 
  (h_initial_sales : initial_sales = 30)
  (h_initial_profit : initial_profit = 40) :
  (∃ (x : ℝ), daily_profit initial_sales initial_profit x = 1200) ∧
  (∀ (y : ℝ), daily_profit initial_sales initial_profit y ≠ 1600) :=
sorry

#check shirt_profit_theorem

end shirt_profit_theorem_l3132_313268


namespace all_dice_same_number_probability_l3132_313200

/-- The probability of a single die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same number -/
def all_same_prob : ℚ := (single_die_prob) ^ (num_dice - 1)

theorem all_dice_same_number_probability :
  all_same_prob = 1 / 216 := by
  sorry

end all_dice_same_number_probability_l3132_313200


namespace least_integer_with_conditions_l3132_313207

/-- The number of 8's in the solution -/
def num_eights : ℕ := 93

/-- The number of 9's in the solution -/
def num_nines : ℕ := 140

/-- The sum of digits in the solution -/
def digit_sum : ℕ := 2011

/-- Constructs the integer from the given number of 8's and 9's -/
def construct_number (n_eights n_nines : ℕ) : ℕ := sorry

/-- Checks if a number is a power of 6 -/
def is_power_of_six (n : ℕ) : Prop := sorry

/-- Calculates the sum of digits of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the product of digits of a number -/
def product_of_digits (n : ℕ) : ℕ := sorry

theorem least_integer_with_conditions :
  let n := construct_number num_eights num_nines
  ∀ m : ℕ, m < n →
    (sum_of_digits m = digit_sum ∧ is_power_of_six (product_of_digits m)) →
    False :=
by sorry

end least_integer_with_conditions_l3132_313207


namespace inscribed_hexagon_diagonal_sum_l3132_313246

/-- A hexagon inscribed in a circle with five sides of length 90 and one side of length 36 -/
structure InscribedHexagon where
  /-- The length of five sides of the hexagon -/
  regularSideLength : ℝ
  /-- The length of the sixth side of the hexagon -/
  irregularSideLength : ℝ
  /-- The hexagon is inscribed in a circle -/
  inscribed : Bool
  /-- Five sides have the same length -/
  fiveSidesEqual : regularSideLength = 90
  /-- The sixth side has a different length -/
  sixthSideDifferent : irregularSideLength = 36
  /-- The hexagon is actually inscribed in a circle -/
  isInscribed : inscribed = true

/-- The sum of the lengths of the three diagonals drawn from one vertex of the hexagon -/
def diagonalSum (h : InscribedHexagon) : ℝ := 428.4

/-- Theorem: The sum of the lengths of the three diagonals drawn from one vertex
    of the inscribed hexagon with the given properties is 428.4 -/
theorem inscribed_hexagon_diagonal_sum (h : InscribedHexagon) :
  diagonalSum h = 428.4 := by sorry

end inscribed_hexagon_diagonal_sum_l3132_313246


namespace balloon_distribution_l3132_313229

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 400) 
  (h2 : num_friends = 10) : 
  (total_balloons / num_friends) - ((total_balloons / num_friends) * 3 / 5) = 16 := by
  sorry

end balloon_distribution_l3132_313229


namespace sum_of_fractions_l3132_313259

theorem sum_of_fractions : (1 : ℚ) / 3 + (1 : ℚ) / 4 = 7 / 12 := by
  sorry

end sum_of_fractions_l3132_313259


namespace f_maximum_l3132_313254

/-- The quadratic function f(x) = -3x^2 + 9x + 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 5

/-- The value of x that maximizes f(x) -/
def x_max : ℝ := 1.5

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
sorry

end f_maximum_l3132_313254


namespace sin_arctan_reciprocal_square_l3132_313294

theorem sin_arctan_reciprocal_square (x : ℝ) (h_pos : x > 0) (h_eq : Real.sin (Real.arctan x) = 1 / x) : x^2 = 1 := by
  sorry

end sin_arctan_reciprocal_square_l3132_313294


namespace total_pies_l3132_313205

theorem total_pies (percent_with_forks : ℝ) (pies_without_forks : ℕ) : 
  percent_with_forks = 0.68 →
  pies_without_forks = 640 →
  ∃ (total_pies : ℕ), 
    (1 - percent_with_forks) * (total_pies : ℝ) = pies_without_forks ∧
    total_pies = 2000 :=
by sorry

end total_pies_l3132_313205


namespace expression_evaluation_l3132_313292

theorem expression_evaluation : 
  let cos_45 : ℝ := Real.sqrt 2 / 2
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (cos_45 - 3)) = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 := by
  sorry

end expression_evaluation_l3132_313292


namespace gcd_bound_from_lcm_l3132_313220

theorem gcd_bound_from_lcm (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) →
  Nat.gcd a b < 1000 := by sorry

end gcd_bound_from_lcm_l3132_313220


namespace equation_solution_l3132_313201

theorem equation_solution : ∃ x : ℝ, 61 + 5 * 12 / (180 / x) = 62 ∧ x = 3 := by
  sorry

end equation_solution_l3132_313201


namespace haley_garden_problem_l3132_313277

def seeds_in_big_garden (total_seeds small_gardens seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - small_gardens * seeds_per_small_garden

theorem haley_garden_problem (total_seeds small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : small_gardens = 7)
  (h3 : seeds_per_small_garden = 3) :
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 := by
  sorry

end haley_garden_problem_l3132_313277


namespace expression_simplification_and_evaluation_l3132_313287

theorem expression_simplification_and_evaluation :
  ∀ x y : ℤ, x = -1 ∧ y = 2 →
  (x * y + (3 * x * y - 4 * x^2) - 2 * (x * y - 2 * x^2)) = 2 * x * y ∧
  2 * x * y = -4 :=
by
  sorry

end expression_simplification_and_evaluation_l3132_313287


namespace line_length_difference_l3132_313278

/-- Conversion rate from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Length of the white line in inches -/
def white_line_inch : ℝ := 7.666666666666667

/-- Length of the blue line in inches -/
def blue_line_inch : ℝ := 3.3333333333333335

/-- Converts a length from inches to centimeters -/
def to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

/-- The difference in length between the white and blue lines in centimeters -/
theorem line_length_difference : 
  to_cm white_line_inch - to_cm blue_line_inch = 11.005555555555553 := by
  sorry

end line_length_difference_l3132_313278


namespace game_draw_probability_l3132_313270

theorem game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3) 
  (h_not_lose : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end game_draw_probability_l3132_313270


namespace arithmetic_sequence_sum_l3132_313264

/-- A sequence satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, 2 * a n = a (n - 1) + a (n + 1)) ∧
  (a 1 + a 3 + a 5 = 9) ∧
  (a 3 + a 5 + a 7 = 15)

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 3 + a 4 + a 5 = 12 := by
  sorry

end arithmetic_sequence_sum_l3132_313264


namespace arithmetic_sequence_properties_l3132_313218

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  first_term : a 1 = 2
  third_sum : S 3 = 12

/-- The main theorem combining both parts of the problem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n) ∧
  (∃ k : ℕ, k > 0 ∧ (seq.a 3) * (seq.a (k + 1)) = (seq.S k)^2 ∧ k = 2) := by
  sorry


end arithmetic_sequence_properties_l3132_313218


namespace isosceles_base_length_l3132_313231

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An equilateral triangle is a triangle where all sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- An isosceles triangle is a triangle where at least two sides are equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The perimeter of a triangle is the sum of its side lengths -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Given an equilateral triangle with perimeter 45 and an isosceles triangle with perimeter 40,
    where at least one side of the isosceles triangle is equal to the side of the equilateral triangle,
    prove that the base of the isosceles triangle is 10 units -/
theorem isosceles_base_length
  (equilateral : Triangle)
  (isosceles : Triangle)
  (h_equilateral : equilateral.isEquilateral)
  (h_isosceles : isosceles.isIsosceles)
  (h_equilateral_perimeter : equilateral.perimeter = 45)
  (h_isosceles_perimeter : isosceles.perimeter = 40)
  (h_shared_side : isosceles.a = equilateral.a ∨ isosceles.b = equilateral.a ∨ isosceles.c = equilateral.a) :
  isosceles.c = 10 ∨ isosceles.b = 10 ∨ isosceles.a = 10 :=
by sorry

end isosceles_base_length_l3132_313231


namespace twins_age_product_difference_l3132_313284

theorem twins_age_product_difference (current_age : ℕ) (h : current_age = 8) : 
  (current_age + 1) * (current_age + 1) - current_age * current_age = 17 := by
  sorry

end twins_age_product_difference_l3132_313284


namespace count_even_factors_l3132_313228

def n : ℕ := 2^3 * 3^2 * 5

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 18 :=
sorry

end count_even_factors_l3132_313228


namespace largest_number_l3132_313234

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a rational number -/
def toRational (r : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 5.14322 -/
def a : ℚ := 5.14322

/-- The number 5.143̅2 -/
def b : RepeatingDecimal := ⟨5, [1, 4, 3], [2]⟩

/-- The number 5.14̅32 -/
def c : RepeatingDecimal := ⟨5, [1, 4], [3, 2]⟩

/-- The number 5.1̅432 -/
def d : RepeatingDecimal := ⟨5, [1], [4, 3, 2]⟩

/-- The number 5.̅4321 -/
def e : RepeatingDecimal := ⟨5, [], [4, 3, 2, 1]⟩

theorem largest_number : 
  toRational d > a ∧ 
  toRational d > toRational b ∧ 
  toRational d > toRational c ∧ 
  toRational d > toRational e :=
sorry

end largest_number_l3132_313234


namespace expression_value_l3132_313242

theorem expression_value : (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 := by
  sorry

end expression_value_l3132_313242


namespace smallest_common_multiple_of_5_and_8_l3132_313263

theorem smallest_common_multiple_of_5_and_8 : 
  ∃ (n : ℕ), n > 0 ∧ Even n ∧ 5 ∣ n ∧ 8 ∣ n ∧ ∀ (m : ℕ), m > 0 → Even m → 5 ∣ m → 8 ∣ m → n ≤ m :=
by sorry

end smallest_common_multiple_of_5_and_8_l3132_313263


namespace imaginary_part_of_z_l3132_313235

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)/z = I) : z.im = -1 := by
  sorry

end imaginary_part_of_z_l3132_313235


namespace line_parameterization_l3132_313286

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = (2/3) * x + 5

-- Define the parameterization
def parameterization (x y s l t : ℝ) : Prop :=
  (x = -3 + t * l) ∧ (y = s - 6 * t)

-- Theorem statement
theorem line_parameterization (s l : ℝ) :
  (∀ x y t : ℝ, line_equation x y ↔ parameterization x y s l t) →
  s = 3 ∧ l = -9 := by
  sorry

end line_parameterization_l3132_313286


namespace sin_cos_sum_equals_negative_one_l3132_313209

theorem sin_cos_sum_equals_negative_one : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end sin_cos_sum_equals_negative_one_l3132_313209


namespace subtraction_of_decimals_l3132_313261

theorem subtraction_of_decimals : (25.50 : ℝ) - 3.245 = 22.255 := by
  sorry

end subtraction_of_decimals_l3132_313261


namespace triangle_side_length_l3132_313296

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 3 →
  b = Real.sqrt 6 →
  B = π / 4 →
  c = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 ∨ c = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 :=
by sorry

end triangle_side_length_l3132_313296


namespace salt_trade_initial_investment_l3132_313256

/-- Represents the merchant's salt trading scenario -/
structure SaltTrade where
  initial_investment : ℕ  -- Initial investment in rubles
  first_profit : ℕ        -- Profit from first sale in rubles
  second_profit : ℕ       -- Profit from second sale in rubles

/-- Theorem stating the initial investment in the salt trade scenario -/
theorem salt_trade_initial_investment (trade : SaltTrade) 
  (h1 : trade.first_profit = 100)
  (h2 : trade.second_profit = 120)
  (h3 : (trade.initial_investment + trade.first_profit + trade.second_profit) = 
        (trade.initial_investment + trade.first_profit) * 
        (trade.initial_investment + trade.first_profit) / trade.initial_investment) :
  trade.initial_investment = 500 := by
  sorry

end salt_trade_initial_investment_l3132_313256


namespace inequality_solution_sum_l3132_313289

/-- Given an inequality ax^2 - 3x + 2 > 0 with solution set {x | x < 1 or x > b}, prove a + b = 3 -/
theorem inequality_solution_sum (a b : ℝ) : 
  (∀ x, ax^2 - 3*x + 2 > 0 ↔ (x < 1 ∨ x > b)) → a + b = 3 := by
  sorry

end inequality_solution_sum_l3132_313289


namespace paper_length_calculation_l3132_313236

/-- The length of a rectangular sheet of paper satisfying specific area conditions -/
theorem paper_length_calculation (L : ℝ) : 
  (2 * 11 * L = 2 * 9.5 * 11 + 100) → L = 14 := by
  sorry

end paper_length_calculation_l3132_313236


namespace graph_is_pair_of_straight_lines_l3132_313252

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- Theorem stating that the graph of x^2 - 9y^2 = 0 is a pair of straight lines -/
theorem graph_is_pair_of_straight_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end graph_is_pair_of_straight_lines_l3132_313252


namespace exponential_inequality_l3132_313248

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end exponential_inequality_l3132_313248


namespace correct_calculation_l3132_313227

theorem correct_calculation : (-4) * (-3) * (-5) = -60 := by
  sorry

end correct_calculation_l3132_313227


namespace cycling_trip_tailwind_time_l3132_313241

theorem cycling_trip_tailwind_time 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_with_tailwind : ℝ) 
  (speed_against_wind : ℝ) 
  (h1 : total_distance = 150) 
  (h2 : total_time = 12) 
  (h3 : speed_with_tailwind = 15) 
  (h4 : speed_against_wind = 10) : 
  ∃ (time_with_tailwind : ℝ), 
    time_with_tailwind = 6 ∧ 
    speed_with_tailwind * time_with_tailwind + 
    speed_against_wind * (total_time - time_with_tailwind) = total_distance := by
  sorry

end cycling_trip_tailwind_time_l3132_313241


namespace power_five_mod_hundred_l3132_313293

theorem power_five_mod_hundred : 5^2023 % 100 = 25 := by
  sorry

end power_five_mod_hundred_l3132_313293


namespace journey_average_speed_l3132_313232

/-- Calculates the average speed of a journey with two segments -/
def average_speed (speed1 : ℝ) (time1_fraction : ℝ) (speed2 : ℝ) (time2_fraction : ℝ) : ℝ :=
  speed1 * time1_fraction + speed2 * time2_fraction

theorem journey_average_speed :
  let speed1 := 10
  let speed2 := 50
  let time1_fraction := 0.25
  let time2_fraction := 0.75
  average_speed speed1 time1_fraction speed2 time2_fraction = 40 := by
sorry

end journey_average_speed_l3132_313232


namespace intersection_M_N_l3132_313212

def M : Set ℝ := {x | (x - 1)^2 < 4}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_M_N_l3132_313212


namespace rhombus_count_in_triangle_l3132_313288

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  smallTrianglesCount : ℕ

/-- Counts the number of rhombuses in a given equilateral triangle -/
def countRhombuses (triangle : EquilateralTriangle) (rhombusSize : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem rhombus_count_in_triangle :
  let largeTriangle := EquilateralTriangle.mk 10
  let rhombusType := Rhombus.mk 8
  countRhombuses largeTriangle rhombusType.smallTrianglesCount = 84 := by
  sorry

end rhombus_count_in_triangle_l3132_313288


namespace floor_sqrt_150_l3132_313225

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by
  sorry

end floor_sqrt_150_l3132_313225


namespace ken_kept_pencils_l3132_313276

def pencil_distribution (total : ℕ) (manny : ℕ) : Prop :=
  let nilo := 2 * manny
  let carlos := nilo / 2
  let tina := carlos + 10
  let rina := tina - 20
  let given_away := manny + nilo + carlos + tina + rina
  total - given_away = 100

theorem ken_kept_pencils :
  pencil_distribution 250 25 := by sorry

end ken_kept_pencils_l3132_313276


namespace hot_dogs_remainder_l3132_313203

theorem hot_dogs_remainder :
  25197625 % 4 = 1 := by sorry

end hot_dogs_remainder_l3132_313203


namespace car_distance_covered_l3132_313290

/-- Prove that a car traveling at 195 km/h for 3 1/5 hours covers a distance of 624 km. -/
theorem car_distance_covered (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 195 → time = 3 + 1 / 5 → distance = speed * time → distance = 624 := by
sorry

end car_distance_covered_l3132_313290


namespace sum_of_y_values_l3132_313266

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 →
  ∃ (y1 y2 : ℝ), y = y1 ∨ y = y2 ∧ y1 + y2 = 5/2 :=
by sorry

end sum_of_y_values_l3132_313266


namespace find_h_l3132_313273

-- Define the two quadratic functions
def f (h j x : ℝ) : ℝ := 4 * (x - h)^2 + j
def g (h k x : ℝ) : ℝ := 3 * (x - h)^2 + k

-- State the theorem
theorem find_h : 
  ∃ (h j k : ℝ),
    (f h j 0 = 2024) ∧ 
    (g h k 0 = 2025) ∧
    (∃ (x₁ x₂ y₁ y₂ : ℤ), x₁ > 0 ∧ x₂ > 0 ∧ y₁ > 0 ∧ y₂ > 0 ∧ 
      f h j (x₁ : ℝ) = 0 ∧ f h j (x₂ : ℝ) = 0 ∧
      g h k (y₁ : ℝ) = 0 ∧ g h k (y₂ : ℝ) = 0) →
    h = 22.5 := by
  sorry


end find_h_l3132_313273


namespace units_digit_of_a_l3132_313240

theorem units_digit_of_a (a : ℕ) : a = 2003^2004 - 2004^2003 → a % 10 = 7 := by
  sorry

end units_digit_of_a_l3132_313240


namespace pie_weight_theorem_l3132_313213

theorem pie_weight_theorem (total_weight : ℝ) (fridge_weight : ℝ) : 
  (5 / 6 : ℝ) * total_weight = fridge_weight → 
  (1 / 6 : ℝ) * total_weight = 240 :=
by
  sorry

#check pie_weight_theorem 1440 1200

end pie_weight_theorem_l3132_313213


namespace triangle_sine_inequality_l3132_313202

theorem triangle_sine_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) (h7 : A + B + C = π) :
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 ∧
  (Real.sin A * Real.sin B * Real.sin C = 3 * Real.sqrt 3 / 8 ↔ A = π/3 ∧ B = π/3 ∧ C = π/3) :=
by sorry

end triangle_sine_inequality_l3132_313202


namespace max_value_inequality_l3132_313279

theorem max_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^3 * (b + c)^3) ≤ 4 / 9 := by
  sorry

end max_value_inequality_l3132_313279


namespace existence_of_special_integers_l3132_313295

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  ¬(7 ∣ (a.val * b.val * (a.val + b.val))) ∧ 
  (7^7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
sorry

end existence_of_special_integers_l3132_313295


namespace right_of_symmetry_decreasing_l3132_313208

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x - 1)^2

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := 1

-- Theorem statement
theorem right_of_symmetry_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ > axis_of_symmetry → x₂ > x₁ → f x₂ < f x₁ := by
  sorry

end right_of_symmetry_decreasing_l3132_313208


namespace strawberry_yogurt_probability_l3132_313219

def prob_strawberry_yogurt (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem strawberry_yogurt_probability :
  let n₁ := 3
  let n₂ := 3
  let p₁ := (1 : ℚ) / 2
  let p₂ := (3 : ℚ) / 4
  let total_days := n₁ + n₂
  let success_days := 4
  (total_days.choose success_days : ℚ) *
    (prob_strawberry_yogurt n₁ 2 p₁ * prob_strawberry_yogurt n₂ 2 p₂ +
     prob_strawberry_yogurt n₁ 3 p₁ * prob_strawberry_yogurt n₂ 1 p₂) =
  1485 / 64 := by
  sorry

end strawberry_yogurt_probability_l3132_313219


namespace min_value_and_max_value_l3132_313297

theorem min_value_and_max_value :
  (∀ x : ℝ, x > 1 → (x + 1 / (x - 1)) ≥ 3) ∧
  (∃ x : ℝ, x > 1 ∧ (x + 1 / (x - 1)) = 3) ∧
  (∀ x : ℝ, 0 < x ∧ x < 10 → Real.sqrt (x * (10 - x)) ≤ 5) ∧
  (∃ x : ℝ, 0 < x ∧ x < 10 ∧ Real.sqrt (x * (10 - x)) = 5) :=
by sorry

end min_value_and_max_value_l3132_313297


namespace geometric_sequence_and_log_function_l3132_313206

/-- Given real numbers a, b, c, and d forming a geometric sequence,
    and for the function y = ln x - x, when x = b reaches its maximum value at c,
    then ad = -1 -/
theorem geometric_sequence_and_log_function
  (a b c d : ℝ)
  (h_geometric : b / a = c / b ∧ c / b = d / c)
  (h_max : b > 0 ∧ c > 0 ∧ (∀ x > 0, Real.log x - x ≤ Real.log c - c) ∧ Real.log b - b = Real.log c - c) :
  a * d = -1 :=
sorry

end geometric_sequence_and_log_function_l3132_313206


namespace isosceles_triangle_angle_measure_l3132_313245

/-- Proves that in an isosceles triangle where one angle is 40% larger than a right angle,
    the measure of one of the two smallest angles is 27°. -/
theorem isosceles_triangle_angle_measure :
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One angle is 40% larger than a right angle (90°)
  c = 90 + 0.4 * 90 →
  -- One of the two smallest angles measures 27°
  a = 27 :=
by
  sorry

end isosceles_triangle_angle_measure_l3132_313245


namespace expression_simplification_l3132_313269

theorem expression_simplification (x : ℝ) (h : x = 5) :
  (2 / (x^2 - 2*x) - (x - 6) / (x^2 - 4*x + 4) / ((x - 6) / (x - 2))) = -1/5 :=
by sorry

end expression_simplification_l3132_313269


namespace unique_digit_solution_l3132_313214

theorem unique_digit_solution :
  ∃! (a b c d e f g h i j : ℕ),
    (a ∈ Finset.range 10) ∧
    (b ∈ Finset.range 10) ∧
    (c ∈ Finset.range 10) ∧
    (d ∈ Finset.range 10) ∧
    (e ∈ Finset.range 10) ∧
    (f ∈ Finset.range 10) ∧
    (g ∈ Finset.range 10) ∧
    (h ∈ Finset.range 10) ∧
    (i ∈ Finset.range 10) ∧
    (j ∈ Finset.range 10) ∧
    ({a, b, c, d, e, f, g, h, i, j} : Finset ℕ).card = 10 ∧
    20 * (a - 8) = 20 ∧
    b / 2 + 17 = 20 ∧
    c * d - 4 = 20 ∧
    (e + 8) / 12 = f ∧
    4 * g + h = 20 ∧
    20 * (i - j) = 100 :=
by sorry

end unique_digit_solution_l3132_313214


namespace yellow_better_for_fine_gift_l3132_313217

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow

/-- Represents the contents of the bag -/
structure Bag :=
  (red : Nat)
  (yellow : Nat)

/-- Calculates the probability of drawing two balls of the same color -/
def probSameColor (b : Bag) : Rat :=
  let total := b.red + b.yellow
  let sameRed := (b.red * (b.red - 1)) / 2
  let sameYellow := (b.yellow * (b.yellow - 1)) / 2
  (sameRed + sameYellow) / ((total * (total - 1)) / 2)

/-- The initial bag configuration -/
def initialBag : Bag := ⟨1, 3⟩

/-- Theorem: Adding a yellow ball gives a higher probability of drawing two balls of the same color -/
theorem yellow_better_for_fine_gift :
  probSameColor ⟨initialBag.red, initialBag.yellow + 1⟩ > 
  probSameColor ⟨initialBag.red + 1, initialBag.yellow⟩ :=
sorry

end yellow_better_for_fine_gift_l3132_313217


namespace unique_modular_solution_l3132_313282

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [ZMOD 11] ∧ n = 5 := by
  sorry

end unique_modular_solution_l3132_313282


namespace dave_outer_space_books_l3132_313224

/-- The number of books about outer space Dave bought -/
def outer_space_books : ℕ := 6

/-- The number of books about animals Dave bought -/
def animal_books : ℕ := 8

/-- The number of books about trains Dave bought -/
def train_books : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 6

/-- The total amount Dave spent on books in dollars -/
def total_spent : ℕ := 102

theorem dave_outer_space_books :
  outer_space_books = (total_spent - book_cost * (animal_books + train_books)) / book_cost :=
by sorry

end dave_outer_space_books_l3132_313224


namespace square_diff_product_plus_square_l3132_313272

theorem square_diff_product_plus_square (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a * b = 2) : 
  a^2 - a*b + b^2 = 27 := by
  sorry

end square_diff_product_plus_square_l3132_313272


namespace complement_A_inter_B_l3132_313221

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}
def A : Set (Fin 5) := {2, 3, 4}
def B : Set (Fin 5) := {0, 1, 4}

theorem complement_A_inter_B :
  (Aᶜ ∩ B : Set (Fin 5)) = {0, 1} := by sorry

end complement_A_inter_B_l3132_313221


namespace article_cost_changes_l3132_313299

theorem article_cost_changes (initial_cost : ℝ) : 
  initial_cost = 75 →
  (initial_cost * (1 + 0.2) * (1 - 0.2) * (1 + 0.3) * (1 - 0.25)) = 70.2 := by
  sorry

end article_cost_changes_l3132_313299
