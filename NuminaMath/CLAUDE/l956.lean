import Mathlib

namespace spoiled_apple_probability_l956_95642

/-- The probability of selecting a spoiled apple from a basket -/
def prob_spoiled_apple (total : ℕ) (spoiled : ℕ) (selected : ℕ) : ℚ :=
  (selected : ℚ) / total

/-- The number of ways to choose k items from n items -/
def combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem spoiled_apple_probability :
  let total := 7
  let spoiled := 1
  let selected := 2
  prob_spoiled_apple total spoiled selected = 2 / 7 := by
  sorry

end spoiled_apple_probability_l956_95642


namespace cube_sum_odd_numbers_l956_95699

theorem cube_sum_odd_numbers (m : ℕ) : 
  (∃ k : ℕ, k ≥ m^2 - m + 1 ∧ k ≤ m^2 + m - 1 ∧ k = 2015) → m = 45 := by
  sorry

end cube_sum_odd_numbers_l956_95699


namespace scout_troop_profit_l956_95678

-- Define the number of candy bars
def num_bars : ℕ := 1500

-- Define the buying price
def buy_price : ℚ := 3 / 4

-- Define the selling price
def sell_price : ℚ := 2 / 3

-- Calculate the total cost
def total_cost : ℚ := num_bars * buy_price

-- Calculate the total revenue
def total_revenue : ℚ := num_bars * sell_price

-- Calculate the profit
def profit : ℚ := total_revenue - total_cost

-- Theorem to prove
theorem scout_troop_profit :
  profit = -125 := by sorry

end scout_troop_profit_l956_95678


namespace parking_lot_wheels_l956_95608

theorem parking_lot_wheels (num_cars num_bikes : ℕ) (wheels_per_car wheels_per_bike : ℕ) :
  num_cars = 14 →
  num_bikes = 5 →
  wheels_per_car = 4 →
  wheels_per_bike = 2 →
  num_cars * wheels_per_car + num_bikes * wheels_per_bike = 66 := by
  sorry

end parking_lot_wheels_l956_95608


namespace centipede_sock_shoe_orders_l956_95619

/-- Represents the number of legs of the centipede -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the number of valid orders to put on socks and shoes -/
def valid_orders : ℕ := Nat.factorial total_items / (2^num_legs)

/-- Theorem stating the number of valid orders for the centipede to put on socks and shoes -/
theorem centipede_sock_shoe_orders :
  valid_orders = Nat.factorial total_items / (2^num_legs) :=
by sorry

end centipede_sock_shoe_orders_l956_95619


namespace sandy_work_hours_l956_95676

theorem sandy_work_hours (total_hours : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 45 → 
  num_days = 5 → 
  total_hours = num_days * hours_per_day → 
  hours_per_day = 9 := by
  sorry

end sandy_work_hours_l956_95676


namespace min_sum_of_squares_for_sum_16_l956_95640

theorem min_sum_of_squares_for_sum_16 :
  ∀ a b c : ℕ+,
  a + b + c = 16 →
  a^2 + b^2 + c^2 ≥ 86 :=
by sorry

end min_sum_of_squares_for_sum_16_l956_95640


namespace probability_is_four_ninths_l956_95632

/-- A cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes the original cube is cut into -/
  total_cubes : ℕ
  /-- The number of smaller cubes with exactly two faces painted -/
  two_faced_cubes : ℕ
  /-- The total number of smaller cubes is 27 -/
  total_is_27 : total_cubes = 27
  /-- The number of two-faced cubes is 12 -/
  two_faced_is_12 : two_faced_cubes = 12

/-- The probability of selecting a small cube with exactly two faces painted -/
def probability_two_faced (c : CutCube) : ℚ :=
  c.two_faced_cubes / c.total_cubes

theorem probability_is_four_ninths (c : CutCube) :
  probability_two_faced c = 4/9 := by
  sorry

end probability_is_four_ninths_l956_95632


namespace snowball_difference_l956_95685

def charlie_snowballs : ℕ := 50
def lucy_snowballs : ℕ := 19

theorem snowball_difference : charlie_snowballs - lucy_snowballs = 31 := by
  sorry

end snowball_difference_l956_95685


namespace h_function_proof_l956_95636

theorem h_function_proof (x : ℝ) (h : ℝ → ℝ) : 
  (12 * x^4 + 4 * x^3 - 2 * x + 3 + h x = 6 * x^3 + 8 * x^2 - 10 * x + 6) →
  (h x = -12 * x^4 + 2 * x^3 + 8 * x^2 - 8 * x + 3) :=
by
  sorry

end h_function_proof_l956_95636


namespace first_part_to_total_ratio_l956_95677

theorem first_part_to_total_ratio : 
  ∃ (n : ℕ), (246.95 : ℝ) / 782 = (4939 : ℝ) / (15640 : ℝ) := by
  sorry

end first_part_to_total_ratio_l956_95677


namespace n_has_nine_digits_l956_95690

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^4 is a perfect cube -/
axiom n_fourth_cube : ∃ k : ℕ, n^4 = k^3

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^4) ∧ (∃ k : ℕ, m^4 = k^3))

/-- The number of digits in n -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 9 digits -/
theorem n_has_nine_digits : num_digits n = 9 := by sorry

end n_has_nine_digits_l956_95690


namespace min_sum_of_primes_for_99_consecutive_sum_l956_95688

/-- The sum of 99 consecutive natural numbers -/
def sum_99_consecutive (x : ℕ) : ℕ := 99 * x

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem min_sum_of_primes_for_99_consecutive_sum :
  ∃ (a b c d : ℕ), 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    (∃ x : ℕ, sum_99_consecutive x = a * b * c * d) ∧
    (∀ a' b' c' d' : ℕ, 
      is_prime a' ∧ is_prime b' ∧ is_prime c' ∧ is_prime d' ∧
      (∃ x : ℕ, sum_99_consecutive x = a' * b' * c' * d') →
      a + b + c + d ≤ a' + b' + c' + d') ∧
    a + b + c + d = 70 :=
by sorry

end min_sum_of_primes_for_99_consecutive_sum_l956_95688


namespace max_value_of_f_l956_95614

def f (x : ℝ) : ℝ := -x^2 + 6*x - 10

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = -1 :=
sorry

end max_value_of_f_l956_95614


namespace circle_tangent_to_y_axis_equation_l956_95666

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle is tangent to the y-axis --/
def is_tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- The equation of a circle --/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_tangent_to_y_axis_equation 
  (c : Circle) 
  (h1 : c.center = (1, 2)) 
  (h2 : is_tangent_to_y_axis c) : 
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 2)^2 = 1 :=
sorry

end circle_tangent_to_y_axis_equation_l956_95666


namespace parabola_focus_coordinates_l956_95623

/-- Given a parabola with equation y^2 = -4x, its focus has coordinates (-1, 0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = -4*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (-1, 0) ∧ ∀ (p : ℝ × ℝ), p ∈ parabola → ‖p - f‖ = ‖p - (p.1, 0)‖ := by
  sorry


end parabola_focus_coordinates_l956_95623


namespace target_probability_l956_95670

def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/2
def num_shots : ℕ := 4

theorem target_probability : 
  let prob_A_2 := (num_shots.choose 2) * prob_A^2 * (1 - prob_A)^2
  let prob_B_3 := (num_shots.choose 3) * prob_B^3 * (1 - prob_B)
  prob_A_2 * prob_B_3 = 2/27 := by sorry

end target_probability_l956_95670


namespace eighteen_percent_of_x_is_ninety_l956_95683

theorem eighteen_percent_of_x_is_ninety (x : ℝ) : (18 / 100) * x = 90 → x = 500 := by
  sorry

end eighteen_percent_of_x_is_ninety_l956_95683


namespace quadratic_roots_property_l956_95682

theorem quadratic_roots_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ * x₂ = 2 →
  (x₁^2 + 2) * (x₂^2 + 2) = 32 := by
  sorry

end quadratic_roots_property_l956_95682


namespace power_five_fifteen_div_power_twentyfive_six_l956_95655

theorem power_five_fifteen_div_power_twentyfive_six :
  5^15 / 25^6 = 125 := by
sorry

end power_five_fifteen_div_power_twentyfive_six_l956_95655


namespace probability_end_multiple_of_three_is_31_90_l956_95606

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def probability_end_multiple_of_three : ℚ :=
  let total_cards := 10
  let prob_left := 1 / 3
  let prob_right := 2 / 3
  let prob_start_multiple_3 := 3 / 10
  let prob_start_one_more := 4 / 10
  let prob_start_one_less := 3 / 10
  let prob_end_multiple_3_from_multiple_3 := prob_left * prob_right + prob_right * prob_left
  let prob_end_multiple_3_from_one_more := prob_right * prob_right
  let prob_end_multiple_3_from_one_less := prob_left * prob_left
  prob_start_multiple_3 * prob_end_multiple_3_from_multiple_3 +
  prob_start_one_more * prob_end_multiple_3_from_one_more +
  prob_start_one_less * prob_end_multiple_3_from_one_less

theorem probability_end_multiple_of_three_is_31_90 :
  probability_end_multiple_of_three = 31 / 90 := by
  sorry

end probability_end_multiple_of_three_is_31_90_l956_95606


namespace b_equals_two_l956_95698

theorem b_equals_two (x y z a b : ℝ) 
  (eq1 : x + y = 2)
  (eq2 : x * y - z^2 = a)
  (eq3 : b = x + y + z) :
  b = 2 := by
sorry

end b_equals_two_l956_95698


namespace final_acid_concentration_l956_95679

/-- Calculates the final acid concentration after removing water from an acidic solution -/
theorem final_acid_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_removed : ℝ)
  (h1 : initial_volume = 12)
  (h2 : initial_concentration = 0.4)
  (h3 : water_removed = 4)
  : (initial_volume * initial_concentration) / (initial_volume - water_removed) = 0.6 := by
  sorry

#check final_acid_concentration

end final_acid_concentration_l956_95679


namespace bowling_ball_weight_l956_95684

theorem bowling_ball_weight (canoe_weight : ℝ) (h1 : canoe_weight = 35) :
  let total_canoe_weight := 2 * canoe_weight
  let bowling_ball_weight := total_canoe_weight / 9
  bowling_ball_weight = 70 / 9 := by
  sorry

end bowling_ball_weight_l956_95684


namespace rectangle_circle_tangent_l956_95609

theorem rectangle_circle_tangent (r : ℝ) (w l : ℝ) : 
  r = 6 →  -- Circle radius is 6 cm
  w = 2 * r →  -- Width of rectangle is diameter of circle
  l * w = 3 * (π * r^2) →  -- Area of rectangle is 3 times area of circle
  l = 9 * π :=  -- Length of longer side is 9π cm
by
  sorry

end rectangle_circle_tangent_l956_95609


namespace dot_product_sum_and_a_l956_95611

/-- Given vectors a and b in ℝ², prove that the dot product of (a + b) and a equals 1. -/
theorem dot_product_sum_and_a (a b : ℝ × ℝ) (h1 : a = (1/2, Real.sqrt 3/2)) 
    (h2 : b = (-Real.sqrt 3/2, 1/2)) : 
    (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 1 := by
  sorry

end dot_product_sum_and_a_l956_95611


namespace integral_2x_over_half_pi_l956_95631

theorem integral_2x_over_half_pi : ∫ x in (0)..(π/2), 2*x = π^2 / 4 := by
  sorry

end integral_2x_over_half_pi_l956_95631


namespace average_length_of_strings_l956_95665

def string1_length : ℝ := 2
def string2_length : ℝ := 6
def num_strings : ℕ := 2

theorem average_length_of_strings :
  (string1_length + string2_length) / num_strings = 4 := by
  sorry

end average_length_of_strings_l956_95665


namespace largest_term_binomial_expansion_l956_95662

theorem largest_term_binomial_expansion (k : ℕ) :
  k ≠ 64 →
  Nat.choose 100 64 * (Real.sqrt 3) ^ 64 > Nat.choose 100 k * (Real.sqrt 3) ^ k :=
sorry

end largest_term_binomial_expansion_l956_95662


namespace min_value_expression_equality_condition_l956_95696

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  let y := x * Real.sqrt 2
  (((x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) = 2 * Real.sqrt 2 :=
by sorry

end min_value_expression_equality_condition_l956_95696


namespace complement_of_A_in_U_l956_95633

universe u

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {4} := by sorry

end complement_of_A_in_U_l956_95633


namespace diagonals_in_polygon_l956_95612

/-- The number of diagonals in a convex k-sided polygon. -/
def num_diagonals (k : ℕ) : ℕ := k * (k - 3) / 2

/-- Theorem stating that the number of diagonals in a convex k-sided polygon
    (where k > 3) is equal to k(k-3)/2. -/
theorem diagonals_in_polygon (k : ℕ) (h : k > 3) :
  num_diagonals k = k * (k - 3) / 2 :=
by sorry

end diagonals_in_polygon_l956_95612


namespace f_convex_when_a_negative_a_range_when_f_bounded_l956_95675

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Define convexity condition
def is_convex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2

-- Theorem 1: f is convex when a < 0
theorem f_convex_when_a_negative (a : ℝ) (h : a < 0) : is_convex (f a) := by
  sorry

-- Theorem 2: Range of a when |f(x)| ≤ 1 for x ∈ [0, 1]
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1) → -2 ≤ a ∧ a < 0 := by
  sorry

end f_convex_when_a_negative_a_range_when_f_bounded_l956_95675


namespace prob_jill_draws_spade_l956_95659

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing a spade from a standard deck -/
def ProbSpade : ℚ := NumSpades / StandardDeck

/-- Probability of not drawing a spade from a standard deck -/
def ProbNotSpade : ℚ := 1 - ProbSpade

/-- Probability that Jill draws a spade in a single round -/
def ProbJillSpadeInRound : ℚ := ProbNotSpade * ProbSpade

/-- Probability that neither Jack nor Jill draws a spade in a round -/
def ProbNoSpadeInRound : ℚ := ProbNotSpade * ProbNotSpade

theorem prob_jill_draws_spade :
  (ProbJillSpadeInRound / (1 - ProbNoSpadeInRound)) = 3 / 7 :=
sorry

end prob_jill_draws_spade_l956_95659


namespace find_A_l956_95613

theorem find_A : ∃ A : ℕ, A = 23 ∧ A / 8 = 2 ∧ A % 8 = 7 := by
  sorry

end find_A_l956_95613


namespace drums_filled_per_day_l956_95647

/-- Given the total number of drums filled and the number of days, 
    calculate the number of drums filled per day -/
def drums_per_day (total_drums : ℕ) (num_days : ℕ) : ℕ :=
  total_drums / num_days

/-- Theorem stating that given 6264 drums filled in 58 days, 
    the number of drums filled per day is 108 -/
theorem drums_filled_per_day : 
  drums_per_day 6264 58 = 108 := by
  sorry

#eval drums_per_day 6264 58

end drums_filled_per_day_l956_95647


namespace table_height_is_33_l956_95634

/-- Represents a block of wood -/
structure Block where
  length : ℝ
  width : ℝ

/-- Represents a table -/
structure Table where
  height : ℝ

/-- Represents the configuration of blocks on the table -/
inductive Configuration
| A
| B

/-- Calculates the total visible length for a given configuration -/
def totalVisibleLength (block : Block) (table : Table) (config : Configuration) : ℝ :=
  match config with
  | Configuration.A => block.length + table.height - block.width
  | Configuration.B => block.width + table.height - block.length

/-- Theorem stating that under the given conditions, the table's height is 33 inches -/
theorem table_height_is_33 (block : Block) (table : Table) :
  totalVisibleLength block table Configuration.A = 36 →
  totalVisibleLength block table Configuration.B = 30 →
  table.height = 33 := by
  sorry

#check table_height_is_33

end table_height_is_33_l956_95634


namespace jeans_price_markup_l956_95689

theorem jeans_price_markup (cost : ℝ) (h : cost > 0) :
  let retailer_price := cost * 1.4
  let customer_price := retailer_price * 1.1
  (customer_price - cost) / cost = 0.54 := by
sorry

end jeans_price_markup_l956_95689


namespace opposite_of_negative_2023_l956_95603

theorem opposite_of_negative_2023 :
  (∀ x : ℤ, x + (-2023) = 0 → x = 2023) :=
by sorry

end opposite_of_negative_2023_l956_95603


namespace only_negative_three_smaller_than_negative_two_l956_95648

theorem only_negative_three_smaller_than_negative_two :
  (0 > -2) ∧ (-1 > -2) ∧ (-3 < -2) ∧ (1 > -2) :=
by sorry

end only_negative_three_smaller_than_negative_two_l956_95648


namespace expression_calculation_l956_95668

theorem expression_calculation : 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + 0.086 + (0.1 : ℝ)^2 = 0.730704 := by
  sorry

end expression_calculation_l956_95668


namespace smallest_circle_area_l956_95669

theorem smallest_circle_area (p1 p2 : ℝ × ℝ) (h : p1 = (-3, -2) ∧ p2 = (2, 4)) :
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let r := d / 2
  let A := π * r^2
  A = (61 * π) / 4 := by
  sorry

end smallest_circle_area_l956_95669


namespace ticket_price_reduction_l956_95627

theorem ticket_price_reduction (x : ℝ) (y : ℝ) (h1 : x > 0) : 
  (4/3 * x * (50 - y) = 5/4 * x * 50) → y = 25/2 := by
  sorry

end ticket_price_reduction_l956_95627


namespace distance_between_A_and_C_l956_95644

-- Define a type for points on a line
structure Point := (x : ℝ)

-- Define a function to calculate distance between two points
def distance (p q : Point) : ℝ := |p.x - q.x|

-- State the theorem
theorem distance_between_A_and_C 
  (A B C : Point) 
  (on_same_line : ∃ (k : ℝ), B.x = k * A.x + (1 - k) * C.x)
  (AB_distance : distance A B = 5)
  (BC_distance : distance B C = 4) :
  distance A C = 1 ∨ distance A C = 9 := by
sorry


end distance_between_A_and_C_l956_95644


namespace inequality_implies_k_bound_l956_95622

theorem inequality_implies_k_bound (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (2*x + y)) → 
  k ≥ Real.sqrt 6 / 2 := by
sorry

end inequality_implies_k_bound_l956_95622


namespace infinite_nested_radical_sqrt_3_l956_95694

theorem infinite_nested_radical_sqrt_3 :
  ∃! (x : ℝ), x > 0 ∧ x = Real.sqrt (3 - x) :=
by
  -- The unique positive solution is (-1 + √13) / 2
  have solution : ℝ := (-1 + Real.sqrt 13) / 2
  
  -- Proof goes here
  sorry

end infinite_nested_radical_sqrt_3_l956_95694


namespace symmetric_points_sum_power_l956_95626

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_about_x_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = x2 ∧ y1 = -y2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_x_axis (a - 1) 5 2 (b - 1) →
  (a + b) ^ 2005 = -1 := by
  sorry

end symmetric_points_sum_power_l956_95626


namespace function_properties_l956_95630

open Real

-- Define the function and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the interval
variable (a b : ℝ)

-- State the theorem
theorem function_properties
  (hf : Continuous f)
  (hf' : Continuous f')
  (hderiv : ∀ x, HasDerivAt f (f' x) x)
  (hab : a < b)
  (hf'a : f' a > 0)
  (hf'b : f' b < 0) :
  (∃ x₀ ∈ Set.Icc a b, f x₀ > f b) ∧
  (∃ x₀ ∈ Set.Icc a b, f a - f b > f' x₀ * (a - b)) ∧
  ¬(∀ x₀ ∈ Set.Icc a b, f x₀ = 0 → False) ∧
  ¬(∀ x₀ ∈ Set.Icc a b, f x₀ > f a → False) := by
  sorry

end function_properties_l956_95630


namespace brick_height_l956_95692

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Proves that the height of a brick with given dimensions and surface area is 3 cm -/
theorem brick_height (l w sa : ℝ) (hl : l = 10) (hw : w = 4) (hsa : sa = 164) :
  ∃ h : ℝ, h = 3 ∧ surface_area l w h = sa :=
by sorry

end brick_height_l956_95692


namespace math_club_trips_l956_95646

/-- Represents a math club with field trips -/
structure MathClub where
  total_students : ℕ
  students_per_trip : ℕ
  (total_students_pos : total_students > 0)
  (students_per_trip_pos : students_per_trip > 0)
  (students_per_trip_le_total : students_per_trip ≤ total_students)

/-- The minimum number of trips for one student to meet all others -/
def min_trips_for_one (club : MathClub) : ℕ :=
  (club.total_students - 1 + club.students_per_trip - 2) / (club.students_per_trip - 1)

/-- The minimum number of trips for all pairs to meet -/
def min_trips_for_all_pairs (club : MathClub) : ℕ :=
  (club.total_students * (club.total_students - 1)) / (club.students_per_trip * (club.students_per_trip - 1))

theorem math_club_trips (club : MathClub) 
  (h1 : club.total_students = 12) 
  (h2 : club.students_per_trip = 6) : 
  min_trips_for_one club = 3 ∧ min_trips_for_all_pairs club = 6 := by
  sorry

#eval min_trips_for_one ⟨12, 6, by norm_num, by norm_num, by norm_num⟩
#eval min_trips_for_all_pairs ⟨12, 6, by norm_num, by norm_num, by norm_num⟩

end math_club_trips_l956_95646


namespace sequence_ratio_l956_95610

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that the ratio of the difference of two terms in the arithmetic sequence 
    to a term in the geometric sequence is 1/2. -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-2 : ℝ) - a₁ = a₁ - a₂) ∧ (a₂ - (-8 : ℝ) = a₁ - a₂) ∧  -- Arithmetic sequence condition
  (b₁ / (-2 : ℝ) = b₂ / b₁) ∧ (b₂ / b₁ = b₃ / b₂) ∧ (b₃ / b₂ = (-8 : ℝ) / b₃) →  -- Geometric sequence condition
  (a₂ - a₁) / b₂ = (1 : ℝ) / 2 := by sorry

end sequence_ratio_l956_95610


namespace bus_ride_net_change_l956_95604

/-- Represents the number of children on a bus and the changes at each stop -/
structure BusRide where
  initial : Int
  first_stop_off : Int
  first_stop_on : Int
  second_stop_off : Int
  final : Int

/-- Calculates the difference between total children who got off and got on -/
def net_change (ride : BusRide) : Int :=
  ride.first_stop_off + ride.second_stop_off - 
  (ride.first_stop_on + (ride.final - (ride.initial - ride.first_stop_off + ride.first_stop_on - ride.second_stop_off)))

/-- Theorem stating the net change in children for the given bus ride -/
theorem bus_ride_net_change :
  let ride : BusRide := {
    initial := 36,
    first_stop_off := 45,
    first_stop_on := 25,
    second_stop_off := 68,
    final := 12
  }
  net_change ride = 24 := by sorry

end bus_ride_net_change_l956_95604


namespace john_balloons_l956_95649

/-- The number of balloons John bought -/
def num_balloons : ℕ := sorry

/-- The volume of air each balloon holds in liters -/
def air_per_balloon : ℕ := 10

/-- The volume of gas in each tank in liters -/
def gas_per_tank : ℕ := 500

/-- The number of tanks John needs to fill all balloons -/
def num_tanks : ℕ := 20

theorem john_balloons :
  num_balloons = 1000 :=
by sorry

end john_balloons_l956_95649


namespace interior_angle_regular_octagon_l956_95653

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ) (sum_interior_angles : ℝ) (interior_angle : ℝ),
  n = 8 →
  sum_interior_angles = (n - 2) * 180 →
  interior_angle = sum_interior_angles / n →
  interior_angle = 135 := by
sorry

end interior_angle_regular_octagon_l956_95653


namespace fraction_sum_approximation_l956_95639

theorem fraction_sum_approximation : 
  let sum := (2007 : ℚ) / 2999 + 8001 / 5998 + 2001 / 3999 + 4013 / 7997 + 10007 / 15999 + 2803 / 11998
  5.99 < sum ∧ sum < 6.01 := by
  sorry

end fraction_sum_approximation_l956_95639


namespace theater_ticket_difference_l956_95643

theorem theater_ticket_difference :
  ∀ (orchestra_price balcony_price : ℕ) 
    (total_tickets total_cost : ℕ) 
    (orchestra_tickets balcony_tickets : ℕ),
  orchestra_price = 12 →
  balcony_price = 8 →
  total_tickets = 360 →
  total_cost = 3320 →
  orchestra_tickets + balcony_tickets = total_tickets →
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost →
  balcony_tickets - orchestra_tickets = 140 :=
by
  sorry

end theater_ticket_difference_l956_95643


namespace class_ratio_theorem_l956_95672

theorem class_ratio_theorem (boys girls : ℕ) (h : boys * 7 = girls * 8) :
  -- 1. The number of girls is 7/8 of the number of boys
  (girls : ℚ) / boys = 7 / 8 ∧
  -- 2. The number of boys accounts for 8/15 of the total number of students
  (boys : ℚ) / (boys + girls) = 8 / 15 ∧
  -- 3. The number of girls accounts for 7/15 of the total number of students
  (girls : ℚ) / (boys + girls) = 7 / 15 ∧
  -- 4. If there are 45 students in total, there are 24 boys
  (boys + girls = 45 → boys = 24) :=
by sorry

end class_ratio_theorem_l956_95672


namespace no_prime_covering_triples_l956_95660

/-- A polynomial is prime-covering if for every prime p, there exists an integer n for which p divides P(n) -/
def IsPrimeCovering (P : ℤ → ℤ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ∃ n : ℤ, (p : ℤ) ∣ P n

/-- The polynomial P(x) = (x^2 - a)(x^2 - b)(x^2 - c) -/
def P (a b c : ℤ) (x : ℤ) : ℤ :=
  (x^2 - a) * (x^2 - b) * (x^2 - c)

theorem no_prime_covering_triples :
  ¬ ∃ a b c : ℤ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25 ∧ IsPrimeCovering (P a b c) := by
  sorry

end no_prime_covering_triples_l956_95660


namespace gcf_72_108_l956_95656

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l956_95656


namespace jessicas_purchases_total_cost_l956_95664

/-- The total cost of Jessica's purchases is $21.95, given that she spent $10.22 on a cat toy and $11.73 on a cage. -/
theorem jessicas_purchases_total_cost : 
  let cat_toy_cost : ℚ := 10.22
  let cage_cost : ℚ := 11.73
  cat_toy_cost + cage_cost = 21.95 := by sorry

end jessicas_purchases_total_cost_l956_95664


namespace students_in_both_teams_l956_95620

theorem students_in_both_teams (total : ℕ) (baseball : ℕ) (hockey : ℕ) 
  (h1 : total = 36) 
  (h2 : baseball = 25) 
  (h3 : hockey = 19) : 
  baseball + hockey - total = 8 := by
  sorry

end students_in_both_teams_l956_95620


namespace class_size_proof_l956_95680

def class_composition (total : ℕ) (girls : ℕ) (boys : ℕ) : Prop :=
  girls + boys = total ∧ girls = (60 * total) / 100

def absent_composition (total : ℕ) (girls : ℕ) (boys : ℕ) : Prop :=
  (girls - 1) = (625 * (total - 3)) / 1000

theorem class_size_proof (total : ℕ) (girls : ℕ) (boys : ℕ) :
  class_composition total girls boys ∧ 
  absent_composition total girls boys →
  girls = 21 ∧ boys = 14 :=
by sorry

end class_size_proof_l956_95680


namespace find_number_l956_95651

theorem find_number : ∃ x : ℤ, x - 27 = 49 ∧ x = 76 := by
  sorry

end find_number_l956_95651


namespace jellybean_count_l956_95691

theorem jellybean_count (steve matt matilda katy : ℕ) : 
  steve = 84 →
  matt = 10 * steve →
  matilda = matt / 2 →
  katy = 3 * matilda →
  katy = matt / 2 →
  katy = 1260 := by
sorry

end jellybean_count_l956_95691


namespace divisibility_property_l956_95624

theorem divisibility_property (y : ℕ) (h : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) :=
by sorry

end divisibility_property_l956_95624


namespace less_than_minus_l956_95654

theorem less_than_minus (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b := by
  sorry

end less_than_minus_l956_95654


namespace pizza_eaters_fraction_l956_95605

theorem pizza_eaters_fraction (total_people : ℕ) (total_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
  (h1 : total_people = 15)
  (h2 : total_pizza = 50)
  (h3 : pieces_per_person = 4)
  (h4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (pieces_per_person * total_people) = 3 / 5 := by
sorry

end pizza_eaters_fraction_l956_95605


namespace solution_set_abs_inequality_l956_95602

theorem solution_set_abs_inequality (x : ℝ) :
  (Set.Icc 1 3 : Set ℝ) = {x | |2 - x| ≤ 1} :=
by sorry

end solution_set_abs_inequality_l956_95602


namespace min_rectangles_cover_square_l956_95650

/-- The smallest number of 3-by-4 non-overlapping rectangles needed to cover a square region -/
def min_rectangles : ℕ := 16

/-- The width of each rectangle -/
def rectangle_width : ℕ := 4

/-- The height of each rectangle -/
def rectangle_height : ℕ := 3

/-- The side length of the square region -/
def square_side : ℕ := 12

theorem min_rectangles_cover_square :
  (min_rectangles * rectangle_width * rectangle_height = square_side * square_side) ∧
  (square_side % rectangle_height = 0) ∧
  (∀ n : ℕ, n < min_rectangles →
    n * rectangle_width * rectangle_height < square_side * square_side) := by
  sorry

#check min_rectangles_cover_square

end min_rectangles_cover_square_l956_95650


namespace average_physics_chemistry_l956_95638

/-- Given the scores in three subjects, prove the average of two subjects --/
theorem average_physics_chemistry 
  (total_average : ℝ) 
  (physics_math_average : ℝ) 
  (physics_score : ℝ) 
  (h1 : total_average = 60) 
  (h2 : physics_math_average = 90) 
  (h3 : physics_score = 140) : 
  (physics_score + (3 * total_average - physics_score - (2 * physics_math_average - physics_score))) / 2 = 70 := by
  sorry

end average_physics_chemistry_l956_95638


namespace right_triangle_perimeter_l956_95618

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 :=
by sorry

end right_triangle_perimeter_l956_95618


namespace ellipse_equation_l956_95645

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c = 2 ∧ c^2 = a^2 - b^2) →  -- Right focus coincides with parabola focus
  (a / 2 = c) →  -- Eccentricity is 1/2
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 16 + y^2 / 12 = 1) :=
by sorry

end ellipse_equation_l956_95645


namespace sum_inequality_l956_95616

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 3) : 
  1 / (x^5 - x^2 + 3) + 1 / (y^5 - y^2 + 3) + 1 / (z^5 - z^2 + 3) ≤ 1 := by
  sorry

end sum_inequality_l956_95616


namespace store_fruit_cost_l956_95673

/-- The cost of fruit in a store -/
structure FruitCost where
  banana_to_apple : ℚ  -- Ratio of banana cost to apple cost
  apple_to_orange : ℚ  -- Ratio of apple cost to orange cost

/-- Given the cost ratios, calculate how many oranges cost the same as a given number of bananas -/
def bananas_to_oranges (cost : FruitCost) (num_bananas : ℕ) : ℚ :=
  (num_bananas : ℚ) * cost.apple_to_orange * cost.banana_to_apple

theorem store_fruit_cost (cost : FruitCost) 
  (h1 : cost.banana_to_apple = 3 / 4)
  (h2 : cost.apple_to_orange = 5 / 7) :
  bananas_to_oranges cost 28 = 15 := by
  sorry

end store_fruit_cost_l956_95673


namespace brians_breath_holding_factor_l956_95667

/-- Given Brian's breath-holding practice over three weeks, prove the factor of increase after the first week. -/
theorem brians_breath_holding_factor
  (initial_time : ℝ)
  (final_time : ℝ)
  (h_initial : initial_time = 10)
  (h_final : final_time = 60)
  (F : ℝ)
  (h_week2 : F * initial_time * 2 = F * initial_time * 2)
  (h_week3 : F * initial_time * 2 * 1.5 = final_time) :
  F = 2 := by
  sorry

end brians_breath_holding_factor_l956_95667


namespace triangle_properties_l956_95615

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  c * Real.cos B = (2 * a - b) * Real.cos C →
  c = 2 →
  a + b + c = 2 * Real.sqrt 3 + 2 →
  -- Triangle validity conditions
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  -- Theorem statements
  C = Real.pi / 3 ∧
  (1/2) * a * b * Real.sin C = (2 * Real.sqrt 3) / 3 := by
sorry

end triangle_properties_l956_95615


namespace nancy_work_hours_nancy_specific_case_l956_95686

/-- Given Nancy's earnings and work hours, calculate the number of hours needed to earn a target amount -/
theorem nancy_work_hours (earnings : ℝ) (work_hours : ℝ) (target_amount : ℝ) :
  earnings > 0 ∧ work_hours > 0 ∧ target_amount > 0 →
  let hourly_rate := earnings / work_hours
  (target_amount / hourly_rate) = (target_amount * work_hours) / earnings :=
by sorry

/-- Nancy's specific work scenario -/
theorem nancy_specific_case :
  let earnings := 28
  let work_hours := 4
  let target_amount := 70
  let hours_needed := (target_amount * work_hours) / earnings
  hours_needed = 10 :=
by sorry

end nancy_work_hours_nancy_specific_case_l956_95686


namespace unique_solution_sum_l956_95674

theorem unique_solution_sum (x y : ℝ) : 
  (|x - 5| = |y - 11|) →
  (|x - 11| = 2*|y - 5|) →
  (x + y = 16) →
  (x + y = 16) :=
by
  sorry

#check unique_solution_sum

end unique_solution_sum_l956_95674


namespace intercept_sum_modulo_13_l956_95617

theorem intercept_sum_modulo_13 : ∃ (x₀ y₀ : ℕ), 
  x₀ < 13 ∧ y₀ < 13 ∧ 
  (4 * x₀ ≡ 1 [MOD 13]) ∧ 
  (3 * y₀ ≡ 12 [MOD 13]) ∧ 
  x₀ + y₀ = 14 := by
  sorry

end intercept_sum_modulo_13_l956_95617


namespace log_seven_eighteen_l956_95621

theorem log_seven_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) := by
  sorry

end log_seven_eighteen_l956_95621


namespace count_checkered_rectangles_l956_95600

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of blue cells -/
def blue_cells : ℕ := 36

/-- The number of red cells -/
def red_cells : ℕ := 4

/-- The number of rectangles containing each blue cell -/
def rectangles_per_blue : ℕ := 4

/-- The number of rectangles containing each red cell -/
def rectangles_per_red : ℕ := 8

/-- The total number of checkered rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue + red_cells * rectangles_per_red

theorem count_checkered_rectangles : total_rectangles = 176 := by
  sorry

end count_checkered_rectangles_l956_95600


namespace geometric_sequence_a6_l956_95628

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_prod : a 4 * a 10 = 16) : 
  a 6 = 2 := by
sorry

end geometric_sequence_a6_l956_95628


namespace volume_of_specific_tetrahedron_l956_95652

/-- Triangle DEF with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Tetrahedron ODEF -/
structure Tetrahedron where
  O : Point3D
  D : Point3D
  E : Point3D
  F : Point3D

def origin : Point3D := ⟨0, 0, 0⟩

/-- Volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: Volume of tetrahedron ODEF is 110/3 -/
theorem volume_of_specific_tetrahedron (tri : Triangle) (t : Tetrahedron) :
  tri.a = 8 ∧ tri.b = 10 ∧ tri.c = 12 ∧
  t.O = origin ∧
  t.D.y = 0 ∧ t.D.z = 0 ∧
  t.E.x = 0 ∧ t.E.z = 0 ∧
  t.F.x = 0 ∧ t.F.y = 0 →
  tetrahedronVolume t = 110 / 3 := by
  sorry

end volume_of_specific_tetrahedron_l956_95652


namespace cookies_per_bag_l956_95657

theorem cookies_per_bag 
  (chocolate_chip : ℕ) 
  (oatmeal : ℕ) 
  (baggies : ℕ) 
  (h1 : chocolate_chip = 2) 
  (h2 : oatmeal = 16) 
  (h3 : baggies = 6) 
  : (chocolate_chip + oatmeal) / baggies = 3 := by
  sorry

end cookies_per_bag_l956_95657


namespace quadratic_two_distinct_roots_l956_95693

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 3*x₁ + k = 0 ∧ x₂^2 + 3*x₂ + k = 0) ↔ k < 9/4 :=
by sorry

end quadratic_two_distinct_roots_l956_95693


namespace composite_function_equality_l956_95601

theorem composite_function_equality (a : ℚ) : 
  let f (x : ℚ) := x / 5 + 4
  let g (x : ℚ) := 5 * x - 3
  f (g a) = 7 → a = 18 / 5 := by
sorry

end composite_function_equality_l956_95601


namespace smallest_p_for_multiple_of_ten_l956_95681

theorem smallest_p_for_multiple_of_ten (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) :
  ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ ∀ q : ℕ, 0 < q → (n + q) % 10 = 0 → p ≤ q :=
by sorry

end smallest_p_for_multiple_of_ten_l956_95681


namespace phone_bill_increase_l956_95671

theorem phone_bill_increase (usual_bill : ℝ) (increase_rate : ℝ) (months : ℕ) : 
  usual_bill = 50 → 
  increase_rate = 0.1 → 
  months = 12 → 
  (usual_bill + usual_bill * increase_rate) * months = 660 := by
sorry

end phone_bill_increase_l956_95671


namespace smallest_value_x_l956_95695

/-- Given a system of linear equations, prove that x is the smallest value -/
theorem smallest_value_x (x y z : ℝ) 
  (eq1 : 3 * x - y = 20)
  (eq2 : 2 * z = 3 * y)
  (eq3 : x + y + z = 48) :
  x < y ∧ x < z :=
by sorry

end smallest_value_x_l956_95695


namespace distance_from_origin_l956_95635

theorem distance_from_origin (z : ℂ) (h : (3 - 4*Complex.I)*z = Complex.abs (4 + 3*Complex.I)) : 
  Complex.abs z = 1 := by sorry

end distance_from_origin_l956_95635


namespace golden_silk_button_optimal_price_reduction_l956_95641

/-- Represents the problem of finding the optimal price reduction for Golden Silk Button --/
theorem golden_silk_button_optimal_price_reduction 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (sales_increase_rate : ℝ) 
  (target_profit : ℝ) 
  (price_reduction : ℝ) : 
  initial_cost = 24 → 
  initial_price = 40 → 
  initial_sales = 20 → 
  sales_increase_rate = 2 → 
  target_profit = 330 → 
  price_reduction = 5 → 
  (initial_price - price_reduction - initial_cost) * (initial_sales + sales_increase_rate * price_reduction) = target_profit :=
by sorry

end golden_silk_button_optimal_price_reduction_l956_95641


namespace sharon_supplies_theorem_l956_95661

def angela_pots : ℕ → ℕ := λ p => p

def angela_plates : ℕ → ℕ := λ p => 3 * p + 6

def angela_cutlery : ℕ → ℕ := λ p => (3 * p + 6) / 2

def sharon_pots : ℕ → ℕ := λ p => p / 2

def sharon_plates : ℕ → ℕ := λ p => 3 * (3 * p + 6) - 20

def sharon_cutlery : ℕ → ℕ := λ p => 3 * p + 6

def sharon_total_supplies : ℕ → ℕ := λ p => 
  sharon_pots p + sharon_plates p + sharon_cutlery p

theorem sharon_supplies_theorem (p : ℕ) : 
  p = 20 → sharon_total_supplies p = 254 := by
  sorry

end sharon_supplies_theorem_l956_95661


namespace expression_evaluation_l956_95629

theorem expression_evaluation : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end expression_evaluation_l956_95629


namespace equality_implies_two_equal_l956_95663

theorem equality_implies_two_equal (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2/y + y^2/z + z^2/x = x^2/z + z^2/y + y^2/x) :
  x = y ∨ x = z ∨ y = z := by
  sorry

end equality_implies_two_equal_l956_95663


namespace solution_difference_l956_95658

-- Define the function f
def f (c₁ c₂ c₃ : ℕ) (x : ℝ) : ℝ :=
  (x^2 - 6*x + c₁) * (x^2 - 6*x + c₂) * (x^2 - 6*x + c₃)

-- Define the set M
def M (c₁ c₂ c₃ : ℕ) : Set ℕ :=
  {x : ℕ | f c₁ c₂ c₃ x = 0}

-- State the theorem
theorem solution_difference (c₁ c₂ c₃ : ℕ) :
  (c₁ ≥ c₂) → (c₂ ≥ c₃) →
  (∃ x₁ x₂ x₃ x₄ x₅ : ℕ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
                         x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                         x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
                         x₄ ≠ x₅ ∧
                         M c₁ c₂ c₃ = {x₁, x₂, x₃, x₄, x₅}) →
  c₁ - c₃ = 4 :=
by sorry

end solution_difference_l956_95658


namespace banana_groups_l956_95607

theorem banana_groups (total_bananas : ℕ) (bananas_per_group : ℕ) 
  (h1 : total_bananas = 203) 
  (h2 : bananas_per_group = 29) : 
  (total_bananas / bananas_per_group : ℕ) = 7 := by
  sorry

end banana_groups_l956_95607


namespace reflection_squared_is_identity_l956_95697

-- Define a reflection matrix over a non-zero vector
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

-- Theorem: The square of a reflection matrix is the identity matrix
theorem reflection_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = !![1, 0; 0, 1] :=
sorry

end reflection_squared_is_identity_l956_95697


namespace debby_initial_bottles_l956_95687

/-- The number of water bottles Debby bought initially -/
def initial_bottles : ℕ := sorry

/-- The number of bottles Debby drinks per day -/
def bottles_per_day : ℕ := 15

/-- The number of days Debby drank water -/
def days_drinking : ℕ := 11

/-- The number of bottles Debby has left -/
def bottles_left : ℕ := 99

/-- Theorem stating that Debby bought 264 water bottles initially -/
theorem debby_initial_bottles : initial_bottles = 264 := by
  sorry

end debby_initial_bottles_l956_95687


namespace binomial_expansion_example_l956_95637

theorem binomial_expansion_example : 16^3 + 3*(16^2)*2 + 3*16*(2^2) + 2^3 = (16 + 2)^3 := by
  sorry

end binomial_expansion_example_l956_95637


namespace fahrenheit_to_celsius_l956_95625

theorem fahrenheit_to_celsius (F C : ℝ) : 
  F = 95 → F = (9/5) * C + 32 → C = 35 := by
  sorry

end fahrenheit_to_celsius_l956_95625
