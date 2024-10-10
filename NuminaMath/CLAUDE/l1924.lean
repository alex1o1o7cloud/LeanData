import Mathlib

namespace tournament_winning_group_exists_l1924_192426

/-- A directed graph representing a tournament. -/
def Tournament (n : ℕ) := Fin n → Fin n → Prop

/-- The property that player i wins against player j. -/
def Wins (t : Tournament n) (i j : Fin n) : Prop := t i j

/-- An ordered group of four players satisfying the winning condition. -/
def WinningGroup (t : Tournament n) (a₁ a₂ a₃ a₄ : Fin n) : Prop :=
  Wins t a₁ a₂ ∧ Wins t a₁ a₃ ∧ Wins t a₁ a₄ ∧
  Wins t a₂ a₃ ∧ Wins t a₂ a₄ ∧
  Wins t a₃ a₄

/-- The main theorem: For n = 8, every tournament has a winning group,
    and this property does not hold for n < 8. -/
theorem tournament_winning_group_exists :
  (∀ (t : Tournament 8), ∃ a₁ a₂ a₃ a₄, WinningGroup t a₁ a₂ a₃ a₄) ∧
  (∀ n < 8, ∃ (t : Tournament n), ∀ a₁ a₂ a₃ a₄, ¬WinningGroup t a₁ a₂ a₃ a₄) :=
sorry

end tournament_winning_group_exists_l1924_192426


namespace decimal_difference_l1924_192418

-- Define the repeating decimal 0.3̄6
def repeating_decimal : ℚ := 4 / 11

-- Define the terminating decimal 0.36
def terminating_decimal : ℚ := 36 / 100

-- Theorem statement
theorem decimal_difference : 
  repeating_decimal - terminating_decimal = 4 / 1100 := by
  sorry

end decimal_difference_l1924_192418


namespace hexagon_and_circle_construction_l1924_192496

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Constructs a hexagon from three non-adjacent vertex projections -/
def construct_hexagon (p1 p2 p3 : Point) : Hexagon :=
  sorry

/-- Constructs an inscribed circle for a given hexagon -/
def construct_inscribed_circle (h : Hexagon) : Circle :=
  sorry

theorem hexagon_and_circle_construction 
  (p1 p2 p3 : Point) 
  (h_not_collinear : ¬ are_collinear p1 p2 p3) :
  ∃ (hex : Hexagon) (circ : Circle), 
    hex = construct_hexagon p1 p2 p3 ∧ 
    circ = construct_inscribed_circle hex :=
  by sorry

end hexagon_and_circle_construction_l1924_192496


namespace watermelons_left_after_sales_l1924_192400

def initial_watermelons : ℕ := 10 * 12

def yesterday_sale_percentage : ℚ := 40 / 100

def today_sale_fraction : ℚ := 1 / 4

def tomorrow_sale_multiplier : ℚ := 3 / 2

def discount_threshold : ℕ := 10

theorem watermelons_left_after_sales : 
  let yesterday_sale := initial_watermelons * yesterday_sale_percentage
  let after_yesterday := initial_watermelons - yesterday_sale
  let today_sale := after_yesterday * today_sale_fraction
  let after_today := after_yesterday - today_sale
  let tomorrow_sale := today_sale * tomorrow_sale_multiplier
  after_today - tomorrow_sale = 27 := by sorry

end watermelons_left_after_sales_l1924_192400


namespace simplify_expression_l1924_192467

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by
  sorry

end simplify_expression_l1924_192467


namespace solution_of_linear_equation_l1924_192478

theorem solution_of_linear_equation (a : ℝ) : 
  (∃ x y : ℝ, x = 3 ∧ y = 2 ∧ a * x + 2 * y = 1) → a = -1 := by
  sorry

end solution_of_linear_equation_l1924_192478


namespace geometric_sequence_product_l1924_192462

/-- Given real numbers x, y, and z, if -1, x, y, z, -3 form a geometric sequence,
    then the product of x and z equals 3. -/
theorem geometric_sequence_product (x y z : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) →
  x * z = 3 := by
  sorry

end geometric_sequence_product_l1924_192462


namespace ben_owes_rachel_l1924_192444

theorem ben_owes_rachel (rate : ℚ) (lawns_mowed : ℚ) 
  (h1 : rate = 13 / 3) 
  (h2 : lawns_mowed = 8 / 5) : 
  rate * lawns_mowed = 104 / 15 := by
  sorry

end ben_owes_rachel_l1924_192444


namespace blue_red_face_ratio_l1924_192453

theorem blue_red_face_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2) / (6 * n^2) = 12 := by
  sorry

end blue_red_face_ratio_l1924_192453


namespace airport_gate_probability_l1924_192485

/-- The number of gates in the airport --/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet --/
def gate_distance : ℕ := 90

/-- The maximum walking distance in feet --/
def max_distance : ℕ := 450

/-- The probability of selecting two gates within the maximum distance --/
def probability : ℚ := 10 / 21

theorem airport_gate_probability :
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs := (num_gates - max_distance / gate_distance) * (max_distance / gate_distance)
    + 2 * (max_distance / gate_distance * (max_distance / gate_distance + 1) / 2)
  (valid_pairs : ℚ) / total_pairs = probability := by sorry

end airport_gate_probability_l1924_192485


namespace quadratic_no_real_roots_l1924_192474

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 1 ≠ 0 := by
sorry

end quadratic_no_real_roots_l1924_192474


namespace triangle_inequality_l1924_192442

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a * b + 1) / (a^2 + c * a + 1) + 
  (b * c + 1) / (b^2 + a * b + 1) + 
  (c * a + 1) / (c^2 + b * c + 1) > 3/2 := by
sorry

end triangle_inequality_l1924_192442


namespace clown_mobile_count_l1924_192468

theorem clown_mobile_count (num_mobiles : ℕ) (clowns_per_mobile : ℕ) 
  (h1 : num_mobiles = 5) 
  (h2 : clowns_per_mobile = 28) : 
  num_mobiles * clowns_per_mobile = 140 := by
sorry

end clown_mobile_count_l1924_192468


namespace smallest_multiple_of_112_l1924_192482

theorem smallest_multiple_of_112 (n : ℕ) : (n * 14 % 112 = 0 ∧ n > 0) → n ≥ 8 := by
  sorry

end smallest_multiple_of_112_l1924_192482


namespace function_value_at_negative_one_l1924_192494

/-- Given a function f(x) = a*tan³(x) - b*sin(3x) + cx + 7 where f(1) = 14, 
    prove that f(-1) = 0 -/
theorem function_value_at_negative_one 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * (Real.tan x)^3 - b * Real.sin (3 * x) + c * x + 7)
  (h2 : f 1 = 14) : 
  f (-1) = 0 := by
  sorry

end function_value_at_negative_one_l1924_192494


namespace odd_minus_odd_is_even_l1924_192460

theorem odd_minus_odd_is_even (a b : ℤ) (ha : Odd a) (hb : Odd b) : Even (a - b) := by
  sorry

end odd_minus_odd_is_even_l1924_192460


namespace recurring_decimal_product_l1924_192455

theorem recurring_decimal_product : 
  ∃ (s : ℚ), (s = 456 / 999) ∧ (7 * s = 355 / 111) := by sorry

end recurring_decimal_product_l1924_192455


namespace perfect_square_problem_l1924_192439

theorem perfect_square_problem :
  (∃ (x : ℕ), 7^2040 = x^2) ∧
  (∀ (x : ℕ), 8^2041 ≠ x^2) ∧
  (∃ (x : ℕ), 9^2042 = x^2) ∧
  (∃ (x : ℕ), 10^2043 = x^2) ∧
  (∃ (x : ℕ), 11^2044 = x^2) :=
by sorry

end perfect_square_problem_l1924_192439


namespace remainder_27_power_27_plus_27_mod_28_l1924_192440

theorem remainder_27_power_27_plus_27_mod_28 :
  (27^27 + 27) % 28 = 26 := by
  sorry

end remainder_27_power_27_plus_27_mod_28_l1924_192440


namespace line_slope_l1924_192456

/-- The slope of a line given by the equation (x/2) + (y/3) = 1 is -3/2 -/
theorem line_slope : 
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ (x / 2 + y / 3 = 1)
  ∃ m b : ℝ, (∀ x y, line_eq x y ↔ y = m * x + b) ∧ m = -3/2 :=
by sorry

end line_slope_l1924_192456


namespace sequence_remainder_l1924_192492

theorem sequence_remainder (n : ℕ) : (7 * n + 4) % 7 = 4 := by
  sorry

end sequence_remainder_l1924_192492


namespace abs_frac_inequality_l1924_192405

theorem abs_frac_inequality (x : ℝ) : 
  |((x - 3) / x)| > ((x - 3) / x) ↔ 0 < x ∧ x < 3 :=
by sorry

end abs_frac_inequality_l1924_192405


namespace stating_prob_three_students_same_group_l1924_192495

/-- Represents the total number of students -/
def total_students : ℕ := 800

/-- Represents the number of lunch groups -/
def num_groups : ℕ := 4

/-- Represents the size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- Represents the probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- 
Theorem stating that the probability of three specific students 
being assigned to the same lunch group is 1/16
-/
theorem prob_three_students_same_group : 
  (prob_assigned_to_group * prob_assigned_to_group : ℚ) = 1 / 16 := by
  sorry

#check prob_three_students_same_group

end stating_prob_three_students_same_group_l1924_192495


namespace probability_above_parabola_l1924_192432

def is_single_digit_positive (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def point_above_parabola (a b : ℕ) : Prop :=
  is_single_digit_positive a ∧ is_single_digit_positive b ∧ b > a * a + b * a

def total_combinations : ℕ := 81

def valid_combinations : ℕ := 7

theorem probability_above_parabola :
  (valid_combinations : ℚ) / total_combinations = 7 / 81 := by sorry

end probability_above_parabola_l1924_192432


namespace car_expense_difference_l1924_192473

/-- Calculates the difference between Alberto's and Samara's car expenses -/
theorem car_expense_difference : 
  let alberto_engine : ℚ := 2457
  let alberto_transmission : ℚ := 374
  let alberto_tires : ℚ := 520
  let alberto_battery : ℚ := 129
  let alberto_exhaust : ℚ := 799
  let alberto_exhaust_discount : ℚ := 0.05
  let alberto_loyalty_discount : ℚ := 0.07
  let samara_oil : ℚ := 25
  let samara_tires : ℚ := 467
  let samara_detailing : ℚ := 79
  let samara_brake_pads : ℚ := 175
  let samara_paint : ℚ := 599
  let samara_stereo : ℚ := 225
  let samara_sales_tax : ℚ := 0.06

  let alberto_total := alberto_engine + alberto_transmission + alberto_tires + alberto_battery + 
                       (alberto_exhaust * (1 - alberto_exhaust_discount))
  let alberto_final := alberto_total * (1 - alberto_loyalty_discount)
  
  let samara_total := samara_oil + samara_tires + samara_detailing + samara_brake_pads + 
                      samara_paint + samara_stereo
  let samara_final := samara_total * (1 + samara_sales_tax)

  alberto_final - samara_final = 2278.12 := by sorry

end car_expense_difference_l1924_192473


namespace f_neg_l1924_192411

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 + 2*x - 3

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + 2*x + 3 := by sorry

end f_neg_l1924_192411


namespace eight_friends_receive_necklace_l1924_192489

/-- The number of friends receiving a candy necklace -/
def friends_receiving_necklace (pieces_per_necklace : ℕ) (pieces_per_block : ℕ) (blocks_used : ℕ) : ℕ :=
  (blocks_used * pieces_per_block) / pieces_per_necklace - 1

/-- Theorem: Given the conditions, prove that 8 friends receive a candy necklace -/
theorem eight_friends_receive_necklace :
  friends_receiving_necklace 10 30 3 = 8 := by
  sorry

end eight_friends_receive_necklace_l1924_192489


namespace intersection_of_A_and_B_l1924_192421

def A : Set ℝ := {x : ℝ | -4 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -4 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_A_and_B_l1924_192421


namespace maggie_yellow_packs_l1924_192429

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs (red_packs green_packs : ℕ) (balls_per_pack : ℕ) (total_balls : ℕ) : ℕ :=
  (total_balls - (red_packs + green_packs) * balls_per_pack) / balls_per_pack

/-- Theorem stating that Maggie bought 8 packs of yellow bouncy balls -/
theorem maggie_yellow_packs : yellow_packs 4 4 10 160 = 8 := by
  sorry

end maggie_yellow_packs_l1924_192429


namespace line_y_coordinate_at_x_10_l1924_192407

/-- Given a line passing through points (4, 0) and (-8, -6), 
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 :
  let m : ℚ := (0 - (-6)) / (4 - (-8))  -- Slope of the line
  let b : ℚ := 0 - m * 4                -- y-intercept of the line
  m * 10 + b = 3 := by sorry

end line_y_coordinate_at_x_10_l1924_192407


namespace tom_catches_sixteen_trout_l1924_192483

/-- The number of trout Melanie catches -/
def melanie_trout : ℕ := 8

/-- Tom catches twice as many trout as Melanie -/
def tom_multiplier : ℕ := 2

/-- The number of trout Tom catches -/
def tom_trout : ℕ := tom_multiplier * melanie_trout

theorem tom_catches_sixteen_trout : tom_trout = 16 := by
  sorry

end tom_catches_sixteen_trout_l1924_192483


namespace new_years_numbers_evenness_l1924_192497

theorem new_years_numbers_evenness (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) :
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 :=
sorry

end new_years_numbers_evenness_l1924_192497


namespace east_northwest_angle_l1924_192448

/-- A circle with ten equally spaced rays -/
structure TenRayCircle where
  rays : Fin 10 → ℝ
  north_ray : rays 0 = 0
  equally_spaced : ∀ i : Fin 10, rays i = (i : ℝ) * 36

/-- The angle between two rays in a TenRayCircle -/
def angle_between (c : TenRayCircle) (i j : Fin 10) : ℝ :=
  ((j - i : ℤ) % 10 : ℤ) * 36

theorem east_northwest_angle (c : TenRayCircle) :
  min (angle_between c 3 8) (angle_between c 8 3) = 144 :=
sorry

end east_northwest_angle_l1924_192448


namespace expansion_coefficients_l1924_192480

theorem expansion_coefficients (n : ℕ) : 
  (2^(2*n) = 2^n + 240) → 
  (∃ k, k = (Nat.choose 8 4) ∧ k = 70) ∧ 
  (∃ m, m = (2^4) ∧ m = 16) := by
  sorry

end expansion_coefficients_l1924_192480


namespace product_selection_probability_l1924_192446

/-- Given a set of products with some being first-class, this function calculates
    the probability that one of two randomly selected products is not first-class,
    given that one of them is first-class. -/
def conditional_probability (total : ℕ) (first_class : ℕ) : ℚ :=
  let not_first_class := total - first_class
  let total_combinations := (total.choose 2 : ℚ)
  let one_not_first_class := (first_class * not_first_class : ℚ)
  one_not_first_class / total_combinations

/-- The theorem states that for 8 total products with 6 being first-class,
    the conditional probability of selecting one non-first-class product
    given that one first-class product is selected is 12/13. -/
theorem product_selection_probability :
  conditional_probability 8 6 = 12 / 13 := by
  sorry

end product_selection_probability_l1924_192446


namespace sqrt_equation_solution_l1924_192451

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 16) = 12 → x = 128 := by
  sorry

end sqrt_equation_solution_l1924_192451


namespace mixed_fraction_power_product_l1924_192484

theorem mixed_fraction_power_product :
  (1 + 2/3)^4 * (-3/5)^5 = -3/5 := by sorry

end mixed_fraction_power_product_l1924_192484


namespace price_increase_x_l1924_192454

/-- The annual price increase of commodity x -/
def annual_increase_x : ℚ := 30 / 100

/-- The annual price increase of commodity y -/
def annual_increase_y : ℚ := 20 / 100

/-- The price of commodity x in 2001 -/
def price_x_2001 : ℚ := 420 / 100

/-- The price of commodity y in 2001 -/
def price_y_2001 : ℚ := 440 / 100

/-- The number of years between 2001 and 2012 -/
def years : ℕ := 11

/-- The difference in price between commodities x and y in 2012 -/
def price_difference_2012 : ℚ := 90 / 100

theorem price_increase_x : 
  annual_increase_x * years + price_x_2001 = 
  annual_increase_y * years + price_y_2001 + price_difference_2012 :=
sorry

end price_increase_x_l1924_192454


namespace cupcakes_per_package_l1924_192491

theorem cupcakes_per_package 
  (initial_cupcakes : ℕ) 
  (eaten_cupcakes : ℕ) 
  (total_packages : ℕ) 
  (h1 : initial_cupcakes = 39) 
  (h2 : eaten_cupcakes = 21) 
  (h3 : total_packages = 6) 
  (h4 : eaten_cupcakes < initial_cupcakes) : 
  (initial_cupcakes - eaten_cupcakes) / total_packages = 3 :=
by
  sorry

end cupcakes_per_package_l1924_192491


namespace seventh_term_of_arithmetic_sequence_l1924_192498

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The seventh term of an arithmetic sequence is the average of its third and eleventh terms. -/
theorem seventh_term_of_arithmetic_sequence
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_third_term : a 3 = 2 / 11)
  (h_eleventh_term : a 11 = 5 / 6) :
  a 7 = 67 / 132 := by
sorry

end seventh_term_of_arithmetic_sequence_l1924_192498


namespace purely_imaginary_product_l1924_192414

theorem purely_imaginary_product (x : ℝ) : 
  (Complex.I * (x^4 + 2*x^3 + x^2 + 2*x) = (x + Complex.I) * ((x^2 + 1) + Complex.I) * ((x + 2) + Complex.I)) ↔ 
  (x = 0 ∨ x = -1) := by sorry

end purely_imaginary_product_l1924_192414


namespace f_is_quadratic_l1924_192441

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 5y = 5y² -/
def f (y : ℝ) : ℝ := 5 * y^2 - 5 * y

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l1924_192441


namespace reassignment_count_l1924_192450

/-- The number of people and jobs -/
def n : ℕ := 5

/-- The number of ways to reassign n jobs to n people such that at least 2 people change jobs -/
def reassignments (n : ℕ) : ℕ := n.factorial - 1

/-- Theorem: The number of ways to reassign 5 jobs to 5 people, 
    such that at least 2 people change jobs from their initial assignment, is 5! - 1 -/
theorem reassignment_count : reassignments n = 119 := by
  sorry

end reassignment_count_l1924_192450


namespace complex_modulus_problem_l1924_192427

theorem complex_modulus_problem (z : ℂ) : (1 + Complex.I) * z = (1 - Complex.I)^2 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l1924_192427


namespace sum_of_squares_l1924_192403

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : w * x = a^2) (h2 : w * y = b^2) (h3 : w * z = c^2) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := by
sorry

end sum_of_squares_l1924_192403


namespace consecutive_integers_sum_l1924_192447

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) = 1680 → (n - 1) + n + (n + 1) + (n + 2) = 26 := by
  sorry

end consecutive_integers_sum_l1924_192447


namespace intersection_of_A_and_B_l1924_192423

def A : Set ℤ := {-2, 0, 1}
def B : Set ℤ := {x | x^2 > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by
  sorry

end intersection_of_A_and_B_l1924_192423


namespace time_per_check_is_two_minutes_l1924_192401

/-- The time per check for lice checks at an elementary school -/
def time_per_check : ℕ :=
  let kindergarteners : ℕ := 26
  let first_graders : ℕ := 19
  let second_graders : ℕ := 20
  let third_graders : ℕ := 25
  let total_students : ℕ := kindergarteners + first_graders + second_graders + third_graders
  let total_time_hours : ℕ := 3
  let total_time_minutes : ℕ := total_time_hours * 60
  total_time_minutes / total_students

/-- Theorem stating that the time per check is 2 minutes -/
theorem time_per_check_is_two_minutes : time_per_check = 2 := by
  sorry

end time_per_check_is_two_minutes_l1924_192401


namespace cubic_root_h_value_l1924_192425

theorem cubic_root_h_value (h : ℝ) : (3 : ℝ)^3 + h * 3 + 14 = 0 → h = -41/3 := by
  sorry

end cubic_root_h_value_l1924_192425


namespace polynomial_remainder_theorem_remainder_equals_897_l1924_192477

def f (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 2*x^6 - 9*x^4 + 3*x^3 - 7

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ (q : ℝ → ℝ), f = fun x ↦ (x - a) * q x + f a := by sorry

theorem remainder_equals_897 :
  ∃ (q : ℝ → ℝ), f = fun x ↦ (3*x - 6) * q x + 897 := by
  have h : ∃ (q : ℝ → ℝ), f = fun x ↦ (x - 2) * q x + f 2 := polynomial_remainder_theorem f 2
  sorry

end polynomial_remainder_theorem_remainder_equals_897_l1924_192477


namespace log_one_fourth_sixteen_l1924_192481

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fourth_sixteen : log (1/4) 16 = -2 := by sorry

end log_one_fourth_sixteen_l1924_192481


namespace projection_vector_is_correct_l1924_192493

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line l -/
def line_l : ParametricLine :=
  { x := λ t => 2 + 3*t,
    y := λ t => 3 + 2*t }

/-- The line m -/
def line_m : ParametricLine :=
  { x := λ s => 4 + 2*s,
    y := λ s => 5 + 3*s }

/-- Direction vector of line l -/
def dir_l : Vector2D :=
  { x := 3,
    y := 2 }

/-- Direction vector of line m -/
def dir_m : Vector2D :=
  { x := 2,
    y := 3 }

/-- The vector perpendicular to the direction of line m -/
def perp_m : Vector2D :=
  { x := 3,
    y := -2 }

/-- The theorem to prove -/
theorem projection_vector_is_correct :
  ∃ (k : ℝ),
    let v : Vector2D := { x := k * perp_m.x, y := k * perp_m.y }
    v.x + v.y = 3 ∧
    v.x = 9 ∧
    v.y = -6 := by
  sorry

end projection_vector_is_correct_l1924_192493


namespace two_numbers_problem_l1924_192459

theorem two_numbers_problem :
  ∃ (x y : ℤ),
    x + y = 44 ∧
    y < 0 ∧
    (x - y) * 100 = y * y ∧
    x = 264 ∧
    y = -220 := by
  sorry

end two_numbers_problem_l1924_192459


namespace dodecagon_area_l1924_192472

/-- Given a square with side length a, prove that the area of a regular dodecagon
    constructed outside the square, where the upper bases of trapezoids on each side
    of the square and their lateral sides form the dodecagon, is equal to (3*a^2)/2. -/
theorem dodecagon_area (a : ℝ) (a_pos : a > 0) :
  let square_side := a
  let dodecagon_area := (3 * a^2) / 2
  dodecagon_area = (3 * square_side^2) / 2 :=
by sorry

end dodecagon_area_l1924_192472


namespace nancy_chips_l1924_192463

/-- Nancy's tortilla chip distribution problem -/
theorem nancy_chips (initial : ℕ) (brother sister : ℕ) (kept : ℕ) : 
  initial = 22 → brother = 7 → sister = 5 → kept = initial - (brother + sister) → kept = 10 := by
  sorry

end nancy_chips_l1924_192463


namespace right_triangle_area_l1924_192479

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 26) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 120 :=
by sorry

end right_triangle_area_l1924_192479


namespace total_distance_walked_l1924_192471

-- Define the walking rate in miles per hour
def walking_rate : ℝ := 4

-- Define the total time in hours
def total_time : ℝ := 2

-- Define the break time in hours
def break_time : ℝ := 0.5

-- Define the effective walking time
def effective_walking_time : ℝ := total_time - break_time

-- Theorem to prove
theorem total_distance_walked :
  walking_rate * effective_walking_time = 6 := by
  sorry

end total_distance_walked_l1924_192471


namespace chess_club_female_fraction_l1924_192469

/-- Represents the chess club membership data --/
structure ChessClub where
  last_year_males : ℕ
  last_year_females : ℕ
  male_increase_rate : ℚ
  female_increase_rate : ℚ
  total_increase_rate : ℚ

/-- Calculates the fraction of female participants this year --/
def female_fraction (club : ChessClub) : ℚ :=
  let this_year_males : ℚ := club.last_year_males * (1 + club.male_increase_rate)
  let this_year_females : ℚ := club.last_year_females * (1 + club.female_increase_rate)
  this_year_females / (this_year_males + this_year_females)

/-- Theorem statement for the chess club problem --/
theorem chess_club_female_fraction :
  let club : ChessClub := {
    last_year_males := 30,
    last_year_females := 15,
    male_increase_rate := 1/10,
    female_increase_rate := 1/4,
    total_increase_rate := 3/20
  }
  female_fraction club = 19/52 := by
  sorry


end chess_club_female_fraction_l1924_192469


namespace chord_distance_half_arc_l1924_192428

/-- Given a circle with radius R and a chord at distance d from the center,
    the distance of the chord corresponding to an arc half as long is √(R(R+d)/2). -/
theorem chord_distance_half_arc (R d : ℝ) (h₁ : R > 0) (h₂ : 0 ≤ d) (h₃ : d < R) :
  let distance_half_arc := Real.sqrt (R * (R + d) / 2)
  distance_half_arc > 0 ∧ distance_half_arc < R :=
by sorry

end chord_distance_half_arc_l1924_192428


namespace shark_stingray_ratio_l1924_192413

theorem shark_stingray_ratio :
  ∀ (total_fish sharks stingrays : ℕ),
    total_fish = 84 →
    stingrays = 28 →
    sharks + stingrays = total_fish →
    sharks / stingrays = 2 :=
by
  sorry

end shark_stingray_ratio_l1924_192413


namespace existence_of_integer_combination_l1924_192486

theorem existence_of_integer_combination (a b c : ℝ) 
  (hab : ∃ (q : ℚ), a * b = q)
  (hbc : ∃ (q : ℚ), b * c = q)
  (hca : ∃ (q : ℚ), c * a = q) :
  ∃ (x y z : ℤ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
sorry

end existence_of_integer_combination_l1924_192486


namespace sixteen_to_power_divided_by_eight_l1924_192487

theorem sixteen_to_power_divided_by_eight (n : ℕ) : n = 16^1024 → n / 8 = 2^4093 := by
  sorry

end sixteen_to_power_divided_by_eight_l1924_192487


namespace equity_investment_l1924_192406

def total_investment : ℝ := 250000

theorem equity_investment (debt : ℝ) 
  (h1 : debt + 3 * debt = total_investment) : 
  3 * debt = 187500 := by
  sorry

#check equity_investment

end equity_investment_l1924_192406


namespace distinguishable_triangles_l1924_192424

def num_colors : ℕ := 8

def corner_configurations : ℕ := 
  num_colors + num_colors * (num_colors - 1) + (num_colors.choose 3)

def center_configurations : ℕ := num_colors * (num_colors - 1)

theorem distinguishable_triangles : 
  corner_configurations * center_configurations = 6720 := by sorry

end distinguishable_triangles_l1924_192424


namespace gold_copper_alloy_density_l1924_192445

/-- The density of gold relative to water -/
def gold_density : ℝ := 10

/-- The density of copper relative to water -/
def copper_density : ℝ := 5

/-- The desired density of the alloy relative to water -/
def alloy_density : ℝ := 9

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 4

theorem gold_copper_alloy_density :
  ∀ (g c : ℝ),
  g > 0 → c > 0 →
  g / c = gold_copper_ratio →
  (gold_density * g + copper_density * c) / (g + c) = alloy_density :=
by sorry

end gold_copper_alloy_density_l1924_192445


namespace unique_solution_l1924_192415

/-- Represents a 3-digit number AAA where A is a single digit -/
def three_digit_AAA (A : ℕ) : ℕ := 100 * A + 10 * A + A

/-- Represents a 6-digit number AAABBB where A and B are single digits -/
def six_digit_AAABBB (A B : ℕ) : ℕ := 1000 * (three_digit_AAA A) + 100 * B + 10 * B + B

/-- Proves that the only solution to AAA × AAA + AAA = AAABBB is A = 9 and B = 0 -/
theorem unique_solution : 
  ∀ A B : ℕ, 
  A ≠ 0 → 
  A < 10 → 
  B < 10 → 
  (three_digit_AAA A) * (three_digit_AAA A) + (three_digit_AAA A) = six_digit_AAABBB A B → 
  A = 9 ∧ B = 0 := by
sorry

end unique_solution_l1924_192415


namespace population_growth_rate_l1924_192436

/-- Proves that given an initial population of 1200, a 25% increase in the first year,
    and a final population of 1950 after two years, the percentage increase in the second year is 30%. -/
theorem population_growth_rate (initial_population : ℕ) (first_year_increase : ℚ) 
  (final_population : ℕ) (second_year_increase : ℚ) : 
  initial_population = 1200 →
  first_year_increase = 25 / 100 →
  final_population = 1950 →
  (initial_population * (1 + first_year_increase) * (1 + second_year_increase) = final_population) →
  second_year_increase = 30 / 100 := by
  sorry

end population_growth_rate_l1924_192436


namespace sum_of_powers_of_i_l1924_192416

/-- The sum of complex numbers 1 + i + i² + ... + i¹⁰ equals i -/
theorem sum_of_powers_of_i : 
  (Finset.range 11).sum (fun k => (Complex.I : ℂ) ^ k) = Complex.I :=
by sorry

end sum_of_powers_of_i_l1924_192416


namespace age_difference_l1924_192466

/-- Proves that the age difference between a man and his son is 24 years -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 22 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 24 := by
  sorry

end age_difference_l1924_192466


namespace ladybugs_per_leaf_l1924_192458

theorem ladybugs_per_leaf (total_leaves : ℕ) (total_ladybugs : ℕ) (h1 : total_leaves = 84) (h2 : total_ladybugs = 11676) :
  total_ladybugs / total_leaves = 139 :=
by sorry

end ladybugs_per_leaf_l1924_192458


namespace greatest_square_power_of_three_under_200_l1924_192461

theorem greatest_square_power_of_three_under_200 : ∃ n : ℕ, 
  n < 200 ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  (∃ k : ℕ, n = 3^k) ∧
  (∀ x : ℕ, x < 200 → (∃ y : ℕ, x = y^2) → (∃ z : ℕ, x = 3^z) → x ≤ n) ∧
  n = 81 :=
by sorry

end greatest_square_power_of_three_under_200_l1924_192461


namespace cars_return_to_start_l1924_192433

/-- Represents a car on a circular race track -/
structure Car where
  position : ℝ  -- Position on the track (0 ≤ position < track_length)
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race -/
structure RaceState where
  track_length : ℝ
  cars : Vector Car n
  time : ℝ

/-- The race system evolves over time -/
def evolve_race (initial_state : RaceState) (t : ℝ) : RaceState :=
  sorry

/-- Predicate to check if all cars are at their initial positions -/
def all_cars_at_initial_positions (initial_state : RaceState) (current_state : RaceState) : Prop :=
  sorry

/-- Main theorem: There exists a time when all cars return to their initial positions -/
theorem cars_return_to_start {n : ℕ} (initial_state : RaceState) :
  ∃ t : ℝ, all_cars_at_initial_positions initial_state (evolve_race initial_state t) :=
  sorry

end cars_return_to_start_l1924_192433


namespace arithmetic_sequence_property_l1924_192464

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end arithmetic_sequence_property_l1924_192464


namespace hunter_frog_count_l1924_192422

/-- The number of frogs Hunter saw in the pond -/
def total_frogs (lily_pad_frogs log_frogs baby_frogs : ℕ) : ℕ :=
  lily_pad_frogs + log_frogs + baby_frogs

/-- Two dozen -/
def two_dozen : ℕ := 2 * 12

theorem hunter_frog_count :
  total_frogs 5 3 two_dozen = 32 := by
  sorry

end hunter_frog_count_l1924_192422


namespace π_approximation_relation_l1924_192488

/-- Approximate value of π obtained with an n-sided inscribed regular polygon -/
noncomputable def π_n (n : ℕ) : ℝ := sorry

/-- Theorem stating the relationship between π_2n and π_n -/
theorem π_approximation_relation (n : ℕ) :
  π_n (2 * n) = π_n n / Real.cos (π / n) := by sorry

end π_approximation_relation_l1924_192488


namespace lens_discount_l1924_192437

def old_camera_price : ℝ := 4000
def lens_original_price : ℝ := 400
def total_paid : ℝ := 5400
def price_increase_percentage : ℝ := 0.30

theorem lens_discount (new_camera_price : ℝ) (lens_paid : ℝ) 
  (h1 : new_camera_price = old_camera_price * (1 + price_increase_percentage))
  (h2 : total_paid = new_camera_price + lens_paid) :
  lens_original_price - lens_paid = 200 := by
sorry

end lens_discount_l1924_192437


namespace oplus_problem_l1924_192408

def oplus (a b : ℚ) : ℚ := a^3 / b

theorem oplus_problem : 
  let x := oplus (oplus 2 4) 6
  let y := oplus 2 (oplus 4 6)
  x - y = 7/12 := by sorry

end oplus_problem_l1924_192408


namespace johns_piggy_bank_l1924_192417

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 →
  quarters = 22 :=
by sorry

end johns_piggy_bank_l1924_192417


namespace calculation_proof_l1924_192419

theorem calculation_proof : (0.08 / 0.002) * 0.5 = 20 := by
  sorry

end calculation_proof_l1924_192419


namespace inverse_function_symmetry_l1924_192410

def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

theorem inverse_function_symmetry 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h₁ : Function.Bijective f) 
  (h₂ : Function.RightInverse g f) 
  (h₃ : Function.LeftInverse g f)
  (h₄ : symmetric_about f (0, 1)) : 
  ∀ a : ℝ, g a + g (2 - a) = 0 := by
sorry

end inverse_function_symmetry_l1924_192410


namespace correct_monthly_repayment_l1924_192452

/-- Calculates the monthly repayment amount for a loan -/
def calculate_monthly_repayment (loan_amount : ℝ) (monthly_interest_rate : ℝ) (loan_term_months : ℕ) : ℝ :=
  sorry

/-- Theorem stating the correct monthly repayment amount -/
theorem correct_monthly_repayment :
  let loan_amount : ℝ := 500000
  let monthly_interest_rate : ℝ := 0.005
  let loan_term_months : ℕ := 360
  abs (calculate_monthly_repayment loan_amount monthly_interest_rate loan_term_months - 2997.75) < 0.01 := by
  sorry

end correct_monthly_repayment_l1924_192452


namespace shelf_capacity_l1924_192476

/-- The number of CDs each rack can hold -/
def cds_per_rack : ℕ := 8

/-- The total number of CDs the shelf can hold -/
def total_cds : ℕ := 32

/-- The number of racks the shelf can hold -/
def num_racks : ℕ := total_cds / cds_per_rack

theorem shelf_capacity : num_racks = 4 := by
  sorry

end shelf_capacity_l1924_192476


namespace tesseract_parallel_edges_l1924_192431

/-- A tesseract is a four-dimensional hypercube -/
structure Tesseract where
  dim : Nat
  edges : Nat

/-- The number of pairs of parallel edges in a tesseract -/
def parallel_edge_pairs (t : Tesseract) : Nat :=
  sorry

/-- Theorem: A tesseract with 32 edges has 36 pairs of parallel edges -/
theorem tesseract_parallel_edges (t : Tesseract) (h1 : t.dim = 4) (h2 : t.edges = 32) :
  parallel_edge_pairs t = 36 := by
  sorry

end tesseract_parallel_edges_l1924_192431


namespace line_disjoint_from_circle_l1924_192449

/-- Given a point M(a,b) inside the unit circle, prove that the line ax + by = 1 is disjoint from the circle -/
theorem line_disjoint_from_circle (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y ≠ 1 :=
by sorry

end line_disjoint_from_circle_l1924_192449


namespace library_visitors_l1924_192457

theorem library_visitors (sunday_avg : ℕ) (month_days : ℕ) (month_avg : ℕ) :
  sunday_avg = 500 →
  month_days = 30 →
  month_avg = 200 →
  let sundays := (month_days + 6) / 7
  let other_days := month_days - sundays
  let other_avg := (month_days * month_avg - sundays * sunday_avg) / other_days
  other_avg = 140 := by
sorry

#eval (30 + 6) / 7  -- Should output 5, representing the number of Sundays

end library_visitors_l1924_192457


namespace unique_solution_circle_equation_l1924_192499

theorem unique_solution_circle_equation :
  ∃! (x y : ℝ), (x - 5)^2 + (y - 6)^2 + (x - y)^2 = 1/3 ∧
  x = 16/3 ∧ y = 17/3 := by
  sorry

end unique_solution_circle_equation_l1924_192499


namespace arithmetic_sequence_property_l1924_192475

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 2 + a 3 = 6) : 
  3 * a 4 + a 6 = 12 := by
  sorry

end arithmetic_sequence_property_l1924_192475


namespace no_three_similar_piles_l1924_192435

theorem no_three_similar_piles (x : ℝ) (hx : x > 0) :
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    a ≤ b ∧ b ≤ c ∧
    c ≤ Real.sqrt 2 * b ∧
    b ≤ Real.sqrt 2 * a :=
by
  sorry

end no_three_similar_piles_l1924_192435


namespace prob_at_least_one_defective_l1924_192402

/-- The probability of selecting at least one defective bulb when randomly choosing two bulbs from a box containing 22 bulbs, of which 4 are defective. -/
theorem prob_at_least_one_defective (total : Nat) (defective : Nat) (h1 : total = 22) (h2 : defective = 4) :
  (1 : ℚ) - (total - defective) * (total - defective - 1) / (total * (total - 1)) = 26 / 77 := by
  sorry

end prob_at_least_one_defective_l1924_192402


namespace disprove_propositions_l1924_192470

open Set

/-- Definition of an M point -/
def is_M_point (f : ℝ → ℝ) (c : ℝ) (a b : ℝ) : Prop :=
  ∃ I : Set ℝ, IsOpen I ∧ c ∈ I ∩ Icc a b ∧
  ∀ x ∈ I ∩ Icc a b, x ≠ c → f x < f c

/-- Main theorem stating the existence of a function that disproves both propositions -/
theorem disprove_propositions : ∃ f : ℝ → ℝ,
  (∃ a b x₀ : ℝ, x₀ ∈ Icc a b ∧ 
    (∀ x ∈ Icc a b, f x ≤ f x₀) ∧ 
    ¬is_M_point f x₀ a b) ∧
  (∀ a b : ℝ, a < b → is_M_point f b a b) ∧
  ¬StrictMono f :=
sorry

end disprove_propositions_l1924_192470


namespace shooting_match_sequences_l1924_192430

/-- Represents the number of targets in each column --/
structure TargetArrangement where
  columnA : Nat
  columnB : Nat
  columnC : Nat

/-- Calculates the number of valid sequences for breaking targets --/
def validSequences (arrangement : TargetArrangement) : Nat :=
  (Nat.factorial 4 / Nat.factorial 1 / Nat.factorial 3) *
  (Nat.factorial 6 / Nat.factorial 3 / Nat.factorial 3)

/-- Theorem statement for the shooting match problem --/
theorem shooting_match_sequences (arrangement : TargetArrangement)
  (h1 : arrangement.columnA = 4)
  (h2 : arrangement.columnB = 3)
  (h3 : arrangement.columnC = 3) :
  validSequences arrangement = 80 := by
  sorry

end shooting_match_sequences_l1924_192430


namespace hyperbola_m_range_l1924_192409

-- Define the equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 1) + y^2 / (m + 2) = 1 ∧ (m + 1) * (m + 2) < 0

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m → -2 < m ∧ m < -1 := by
  sorry

end hyperbola_m_range_l1924_192409


namespace lattice_points_count_l1924_192420

/-- A triangular lattice -/
structure TriangularLattice where
  /-- The distance between adjacent points is 1 -/
  adjacent_distance : ℝ
  adjacent_distance_eq : adjacent_distance = 1

/-- An equilateral triangle on a triangular lattice -/
structure EquilateralTriangle (L : ℝ) where
  /-- The side length of the triangle -/
  side_length : ℝ
  side_length_eq : side_length = L
  /-- The triangle has no lattice points on its sides -/
  no_lattice_points_on_sides : Prop

/-- The number of lattice points inside an equilateral triangle -/
def lattice_points_inside (L : ℝ) (triangle : EquilateralTriangle L) : ℕ :=
  sorry

theorem lattice_points_count (L : ℝ) (triangle : EquilateralTriangle L) :
  lattice_points_inside L triangle = (L^2 - 1) / 2 :=
sorry

end lattice_points_count_l1924_192420


namespace problem_solution_l1924_192490

theorem problem_solution (X Y : ℝ) : 
  (18 / 100 * X = 54 / 100 * 1200) → 
  (X = 4 * Y) → 
  (X = 3600 ∧ Y = 900) := by
sorry

end problem_solution_l1924_192490


namespace trigonometric_expression_evaluation_l1924_192438

theorem trigonometric_expression_evaluation :
  (Real.cos (40 * π / 180) + Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) /
  (Real.sin (70 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end trigonometric_expression_evaluation_l1924_192438


namespace total_oranges_picked_l1924_192465

theorem total_oranges_picked (del_per_day : ℕ) (del_days : ℕ) (juan_oranges : ℕ) :
  del_per_day = 23 →
  del_days = 2 →
  juan_oranges = 61 →
  del_per_day * del_days + juan_oranges = 107 :=
by
  sorry

end total_oranges_picked_l1924_192465


namespace james_toy_cost_l1924_192434

/-- Calculates the cost per toy given the total number of toys, percentage sold, selling price, and profit. -/
def cost_per_toy (total_toys : ℕ) (percent_sold : ℚ) (selling_price : ℚ) (profit : ℚ) : ℚ :=
  let sold_toys : ℚ := total_toys * percent_sold
  let revenue : ℚ := sold_toys * selling_price
  let cost : ℚ := revenue - profit
  cost / sold_toys

/-- Proves that the cost per toy is $25 given the problem conditions. -/
theorem james_toy_cost :
  cost_per_toy 200 (80 / 100) 30 800 = 25 := by
  sorry

end james_toy_cost_l1924_192434


namespace fib_units_digit_periodic_fib_15_value_units_digit_of_fib_fib_15_l1924_192443

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_units_digit_periodic (n : ℕ) : fib n % 10 = fib (n % 60) % 10 := by sorry

theorem fib_15_value : fib 15 = 610 := by sorry

theorem units_digit_of_fib_fib_15 : fib (fib 15) % 10 = 5 := by sorry

end fib_units_digit_periodic_fib_15_value_units_digit_of_fib_fib_15_l1924_192443


namespace necessary_condition_equality_l1924_192412

theorem necessary_condition_equality (a b c : ℝ) (h : c ≠ 0) :
  a = b → a * c = b * c :=
by sorry

end necessary_condition_equality_l1924_192412


namespace equation_3y_plus_1_eq_6_is_linear_l1924_192404

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 3y + 1 = 6 is a linear equation -/
theorem equation_3y_plus_1_eq_6_is_linear :
  is_linear_equation (λ y => 3 * y + 1) :=
by
  sorry

#check equation_3y_plus_1_eq_6_is_linear

end equation_3y_plus_1_eq_6_is_linear_l1924_192404
