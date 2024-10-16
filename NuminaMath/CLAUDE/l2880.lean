import Mathlib

namespace NUMINAMATH_CALUDE_power_of_five_equality_l2880_288038

theorem power_of_five_equality (k : ℕ) : 5^k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_equality_l2880_288038


namespace NUMINAMATH_CALUDE_only_tiger_and_leopard_can_participate_l2880_288023

-- Define the animals
inductive Animal : Type
| Lion : Animal
| Tiger : Animal
| Leopard : Animal
| Elephant : Animal

-- Define a function to represent selection
def isSelected : Animal → Prop := sorry

-- Define the conditions
def conditions (isSelected : Animal → Prop) : Prop :=
  (isSelected Animal.Lion → isSelected Animal.Tiger) ∧
  (¬isSelected Animal.Leopard → ¬isSelected Animal.Tiger) ∧
  (isSelected Animal.Leopard → ¬isSelected Animal.Elephant) ∧
  (∃ (a b : Animal), a ≠ b ∧ isSelected a ∧ isSelected b ∧
    ∀ (c : Animal), c ≠ a ∧ c ≠ b → ¬isSelected c)

-- Theorem statement
theorem only_tiger_and_leopard_can_participate :
  ∀ (isSelected : Animal → Prop),
    conditions isSelected →
    isSelected Animal.Tiger ∧ isSelected Animal.Leopard ∧
    ¬isSelected Animal.Lion ∧ ¬isSelected Animal.Elephant :=
sorry

end NUMINAMATH_CALUDE_only_tiger_and_leopard_can_participate_l2880_288023


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2880_288086

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 35

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 28

theorem bowling_ball_weight_proof :
  (5 * bowling_ball_weight = 4 * kayak_weight) →
  bowling_ball_weight = 28 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2880_288086


namespace NUMINAMATH_CALUDE_triangle_equal_area_division_l2880_288069

theorem triangle_equal_area_division :
  let triangle := [(0, 0), (1, 1), (9, 1)]
  let total_area := 4
  let dividing_line := 3
  let left_area := (1/2) * dividing_line * (dividing_line/9)
  let right_area := (1/2) * (1 - dividing_line/9) * (9 - dividing_line)
  left_area = right_area ∧ left_area = total_area/2 := by sorry

end NUMINAMATH_CALUDE_triangle_equal_area_division_l2880_288069


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2880_288029

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 3 = 0) → 
  (x₂^2 + x₂ - 3 = 0) → 
  x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2880_288029


namespace NUMINAMATH_CALUDE_local_minimum_condition_l2880_288019

/-- The function f(x) = x^3 + (x-a)^2 has a local minimum at x = 2 if and only if a = 8 -/
theorem local_minimum_condition (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), 
    x^3 + (x - a)^2 ≥ 2^3 + (2 - a)^2) ↔ a = 8 := by
  sorry


end NUMINAMATH_CALUDE_local_minimum_condition_l2880_288019


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2880_288070

theorem gcd_lcm_sum : Nat.gcd 42 63 + Nat.lcm 48 18 = 165 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2880_288070


namespace NUMINAMATH_CALUDE_line_through_point_l2880_288051

theorem line_through_point (k : ℝ) : (2 * k * 3 - 1 = 5) ↔ (k = 1) := by sorry

end NUMINAMATH_CALUDE_line_through_point_l2880_288051


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2880_288095

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) (wrong_answer_penalty : ℕ) :
  total_questions = 80 →
  correct_answers = 40 →
  total_marks = 120 →
  wrong_answer_penalty = 1 →
  ∃ (marks_per_correct : ℕ),
    marks_per_correct * correct_answers - wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2880_288095


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l2880_288049

theorem smallest_n_for_candy (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 25 * m % 10 = 0 ∧ 25 * m % 16 = 0 ∧ 25 * m % 18 = 0) →
  (25 * n % 10 = 0 ∧ 25 * n % 16 = 0 ∧ 25 * n % 18 = 0) →
  n ≥ 29 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l2880_288049


namespace NUMINAMATH_CALUDE_missing_figure_proof_l2880_288087

theorem missing_figure_proof (x : ℝ) (h : (0.50 / 100) * x = 0.12) : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l2880_288087


namespace NUMINAMATH_CALUDE_cistern_filling_problem_l2880_288099

/-- The time taken for pipe A to fill the cistern -/
def time_A : ℝ := 16

/-- The time taken for pipe B to empty the cistern -/
def time_B : ℝ := 20

/-- The time taken to fill the cistern when both pipes are open -/
def time_both : ℝ := 80

/-- Theorem stating that the given times satisfy the cistern filling problem -/
theorem cistern_filling_problem :
  1 / time_A - 1 / time_B = 1 / time_both := by sorry

end NUMINAMATH_CALUDE_cistern_filling_problem_l2880_288099


namespace NUMINAMATH_CALUDE_haley_concert_spending_l2880_288055

def ticket_price : ℕ := 4
def tickets_for_self_and_friends : ℕ := 3
def extra_tickets : ℕ := 5

theorem haley_concert_spending :
  (tickets_for_self_and_friends + extra_tickets) * ticket_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_concert_spending_l2880_288055


namespace NUMINAMATH_CALUDE_equilateral_triangle_tiling_l2880_288006

theorem equilateral_triangle_tiling (large_side : ℝ) (small_side : ℝ) : 
  large_side = 15 →
  small_side = 3 →
  (large_side^2 / small_side^2 : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_tiling_l2880_288006


namespace NUMINAMATH_CALUDE_largest_integer_less_than_95_with_remainder_5_mod_7_l2880_288053

theorem largest_integer_less_than_95_with_remainder_5_mod_7 :
  ∃ n : ℤ, n < 95 ∧ n % 7 = 5 ∧ ∀ m : ℤ, m < 95 ∧ m % 7 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_95_with_remainder_5_mod_7_l2880_288053


namespace NUMINAMATH_CALUDE_range_of_x_for_meaningful_sqrt_l2880_288052

theorem range_of_x_for_meaningful_sqrt (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3*x - 2) → x ≥ 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_for_meaningful_sqrt_l2880_288052


namespace NUMINAMATH_CALUDE_discount_calculation_l2880_288008

/-- Given an article with a cost price of 100 units, if the selling price is marked 12% above 
    the cost price and the trader suffers a loss of 1% at the time of selling, 
    then the discount allowed is 13 units. -/
theorem discount_calculation (cost_price : ℝ) (marked_price : ℝ) (selling_price : ℝ) : 
  cost_price = 100 →
  marked_price = cost_price * 1.12 →
  selling_price = cost_price * 0.99 →
  marked_price - selling_price = 13 :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l2880_288008


namespace NUMINAMATH_CALUDE_unique_number_with_equal_sums_l2880_288035

def ends_with_9876 (n : ℕ) : Prop :=
  n % 10000 = 9876

def masha_sum (n : ℕ) : ℕ :=
  (n / 1000) * 10 + n % 1000

def misha_sum (n : ℕ) : ℕ :=
  (n / 10000) + n % 10000

theorem unique_number_with_equal_sums :
  ∃! n : ℕ, n > 9999 ∧ ends_with_9876 n ∧ masha_sum n = misha_sum n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_equal_sums_l2880_288035


namespace NUMINAMATH_CALUDE_min_value_expression_l2880_288065

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ((x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2)) ≥ 4 * Real.sqrt 5 + 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2880_288065


namespace NUMINAMATH_CALUDE_arrangement_counts_l2880_288031

/-- The number of boys in the row -/
def num_boys : ℕ := 4

/-- The number of girls in the row -/
def num_girls : ℕ := 2

/-- The total number of students in the row -/
def total_students : ℕ := num_boys + num_girls

/-- The number of arrangements where Boy A does not stand at the head or the tail of the row -/
def arrangements_A_not_ends : ℕ := 480

/-- The number of arrangements where the two girls must stand next to each other -/
def arrangements_girls_together : ℕ := 240

/-- The number of arrangements where Students A, B, and C are not next to each other -/
def arrangements_ABC_not_adjacent : ℕ := 144

/-- The number of arrangements where A does not stand at the head, and B does not stand at the tail -/
def arrangements_A_not_head_B_not_tail : ℕ := 504

theorem arrangement_counts :
  arrangements_A_not_ends = 480 ∧
  arrangements_girls_together = 240 ∧
  arrangements_ABC_not_adjacent = 144 ∧
  arrangements_A_not_head_B_not_tail = 504 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l2880_288031


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_sq_gt_a_l2880_288047

theorem a_gt_one_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > a) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_sq_gt_a_l2880_288047


namespace NUMINAMATH_CALUDE_oranges_remaining_l2880_288089

def initial_oranges : ℕ := 60
def percentage_taken : ℚ := 45 / 100

theorem oranges_remaining : 
  initial_oranges - (percentage_taken * initial_oranges).floor = 33 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_l2880_288089


namespace NUMINAMATH_CALUDE_population_exceeds_target_in_2075_l2880_288014

/-- The initial population of Nisos in the year 2000 -/
def initial_population : ℕ := 500

/-- The year when the population count starts -/
def start_year : ℕ := 2000

/-- The number of years it takes for the population to triple -/
def tripling_period : ℕ := 25

/-- The target population we want to exceed -/
def target_population : ℕ := 9000

/-- Calculate the population after a given number of tripling periods -/
def population_after (periods : ℕ) : ℕ :=
  initial_population * (3 ^ periods)

/-- Calculate the year after a given number of tripling periods -/
def year_after (periods : ℕ) : ℕ :=
  start_year + tripling_period * periods

/-- The theorem to be proved -/
theorem population_exceeds_target_in_2075 :
  ∃ n : ℕ, year_after n = 2075 ∧ 
    population_after n > target_population ∧
    population_after (n - 1) ≤ target_population :=
by
  sorry


end NUMINAMATH_CALUDE_population_exceeds_target_in_2075_l2880_288014


namespace NUMINAMATH_CALUDE_faye_crayons_l2880_288032

/-- The number of rows of crayons and pencils -/
def num_rows : ℕ := 16

/-- The number of crayons in each row -/
def crayons_per_row : ℕ := 6

/-- The total number of crayons -/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem faye_crayons : total_crayons = 96 := by
  sorry

end NUMINAMATH_CALUDE_faye_crayons_l2880_288032


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2880_288027

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2880_288027


namespace NUMINAMATH_CALUDE_cows_sold_l2880_288042

/-- The number of cows sold by a man last year, given the following conditions:
  * He initially had 39 cows
  * 25 cows died last year
  * The number of cows increased by 24 this year
  * He bought 43 more cows
  * His friend gave him 8 cows as a gift
  * He now has 83 cows -/
theorem cows_sold (initial : ℕ) (died : ℕ) (increased : ℕ) (bought : ℕ) (gifted : ℕ) (current : ℕ)
  (h_initial : initial = 39)
  (h_died : died = 25)
  (h_increased : increased = 24)
  (h_bought : bought = 43)
  (h_gifted : gifted = 8)
  (h_current : current = 83)
  (h_equation : current = initial - died - (initial - died - increased - bought - gifted)) :
  initial - died - increased - bought - gifted = 6 := by
  sorry

end NUMINAMATH_CALUDE_cows_sold_l2880_288042


namespace NUMINAMATH_CALUDE_factor_expression_l2880_288026

theorem factor_expression (y : ℝ) : 84 * y^13 + 210 * y^26 = 42 * y^13 * (2 + 5 * y^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2880_288026


namespace NUMINAMATH_CALUDE_net_profit_calculation_l2880_288000

/-- Given the purchase price, overhead percentage, and markup, calculate the net profit --/
def calculate_net_profit (purchase_price overhead_percentage markup : ℝ) : ℝ :=
  let overhead := purchase_price * overhead_percentage
  markup - overhead

/-- Theorem stating that given the specified conditions, the net profit is $27.60 --/
theorem net_profit_calculation :
  let purchase_price : ℝ := 48
  let overhead_percentage : ℝ := 0.05
  let markup : ℝ := 30
  calculate_net_profit purchase_price overhead_percentage markup = 27.60 := by
  sorry

#eval calculate_net_profit 48 0.05 30

end NUMINAMATH_CALUDE_net_profit_calculation_l2880_288000


namespace NUMINAMATH_CALUDE_store_purchase_exists_l2880_288012

theorem store_purchase_exists :
  ∃ (P L E : ℕ), 0.45 * (P : ℝ) + 0.35 * (L : ℝ) + 0.30 * (E : ℝ) = 7.80 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_exists_l2880_288012


namespace NUMINAMATH_CALUDE_correct_operation_result_l2880_288039

theorem correct_operation_result (x : ℝ) : 
  ((x / 8) ^ 2 = 49) → ((x * 8) * 2 = 896) := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_result_l2880_288039


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2880_288079

/-- The polynomial function we're investigating -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 3*x^2 + 2*x - 6

/-- Theorem stating that -1 and 3 are the only real roots of the polynomial -/
theorem real_roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2880_288079


namespace NUMINAMATH_CALUDE_circle_radius_through_ROV_l2880_288043

-- Define the pentagon LOVER
structure Pentagon :=
  (L O V E R : ℝ × ℝ)

-- Define properties of the pentagon
def is_convex (p : Pentagon) : Prop := sorry

def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_radius_through_ROV (LOVER : Pentagon) :
  is_convex LOVER →
  is_rectangle LOVER.L LOVER.O LOVER.V LOVER.E →
  distance LOVER.O LOVER.V = 20 →
  distance LOVER.L LOVER.O = 23 →
  distance LOVER.V LOVER.E = 23 →
  distance LOVER.R LOVER.E = 23 →
  distance LOVER.R LOVER.L = 23 →
  ∃ (center : ℝ × ℝ), 
    distance center LOVER.R = 23 ∧
    distance center LOVER.O = 23 ∧
    distance center LOVER.V = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_radius_through_ROV_l2880_288043


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2880_288024

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p ≠ q ∧
    (p : ℤ) * (q : ℤ) = k ∧ 
    (p : ℤ) + (q : ℤ) = 58 :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2880_288024


namespace NUMINAMATH_CALUDE_marco_run_time_l2880_288084

-- Define the track and run parameters
def total_laps : ℕ := 6
def track_length : ℝ := 450
def first_segment : ℝ := 150
def second_segment : ℝ := 300
def speed_first : ℝ := 5
def speed_second : ℝ := 4

-- Define the theorem
theorem marco_run_time :
  let time_first := first_segment / speed_first
  let time_second := second_segment / speed_second
  let time_per_lap := time_first + time_second
  let total_time := total_laps * time_per_lap
  total_time = 630 := by sorry

end NUMINAMATH_CALUDE_marco_run_time_l2880_288084


namespace NUMINAMATH_CALUDE_sum_of_variables_l2880_288046

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 18 - 4*x)
  (eq2 : x + z = 22 - 4*y)
  (eq3 : x + y = 15 - 4*z) :
  3*x + 3*y + 3*z = 55/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l2880_288046


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2880_288034

theorem sphere_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 675 * Real.pi → 
  V_S = 0.2 * V_L → 
  V_L = (4/3) * Real.pi * r_L^3 → 
  V_S = (4/3) * Real.pi * r_S^3 → 
  r_S / r_L = 1 / Real.rpow 5 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2880_288034


namespace NUMINAMATH_CALUDE_min_abs_diff_bound_l2880_288068

theorem min_abs_diff_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  min (min (|a - b|) (|b - c|)) (|c - a|) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_diff_bound_l2880_288068


namespace NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l2880_288097

theorem factorial_squared_greater_than_power (n : ℕ) (h : n > 2) :
  (Nat.factorial n)^2 > n^n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l2880_288097


namespace NUMINAMATH_CALUDE_zoe_mp3_songs_l2880_288096

theorem zoe_mp3_songs (initial_songs : ℕ) (deleted_songs : ℕ) (added_songs : ℕ) :
  initial_songs = 6 →
  deleted_songs = 3 →
  added_songs = 20 →
  initial_songs - deleted_songs + added_songs = 23 :=
by sorry

end NUMINAMATH_CALUDE_zoe_mp3_songs_l2880_288096


namespace NUMINAMATH_CALUDE_x_investment_value_l2880_288098

/-- Represents the investment and profit scenario of a business partnership --/
structure BusinessPartnership where
  x_investment : ℕ  -- X's investment
  y_investment : ℕ  -- Y's investment
  z_investment : ℕ  -- Z's investment
  total_profit : ℕ  -- Total profit
  z_profit : ℕ      -- Z's share of the profit
  x_months : ℕ      -- Months X and Y were in business before Z joined
  z_months : ℕ      -- Months Z was in business

/-- The main theorem stating that X's investment was 35700 given the conditions --/
theorem x_investment_value (bp : BusinessPartnership) : 
  bp.y_investment = 42000 ∧ 
  bp.z_investment = 48000 ∧ 
  bp.total_profit = 14300 ∧ 
  bp.z_profit = 4160 ∧
  bp.x_months = 12 ∧
  bp.z_months = 8 →
  bp.x_investment = 35700 := by
  sorry

#check x_investment_value

end NUMINAMATH_CALUDE_x_investment_value_l2880_288098


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2880_288011

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 3) :
  (1/a + 1/b + 1/c) ≥ 3 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 1/x + 1/y + 1/z = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2880_288011


namespace NUMINAMATH_CALUDE_machine_value_depletion_rate_l2880_288004

theorem machine_value_depletion_rate 
  (present_value : ℝ) 
  (value_after_2_years : ℝ) 
  (depletion_rate : ℝ) : 
  present_value = 1100 → 
  value_after_2_years = 891 → 
  value_after_2_years = present_value * (1 - depletion_rate)^2 → 
  depletion_rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_machine_value_depletion_rate_l2880_288004


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l2880_288061

-- Define the function f(x) = 2x
def f (x : ℝ) : ℝ := 2 * x

-- Theorem stating that f is both odd and increasing
theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l2880_288061


namespace NUMINAMATH_CALUDE_cake_recipe_flour_amount_l2880_288022

/-- The total number of cups of flour in Mary's cake recipe -/
def total_flour : ℕ := 9

/-- The total number of cups of sugar in the recipe -/
def total_sugar : ℕ := 11

/-- The number of cups of flour already added -/
def flour_added : ℕ := 4

/-- The difference between remaining sugar and remaining flour to be added -/
def sugar_flour_diff : ℕ := 6

theorem cake_recipe_flour_amount :
  total_flour = 9 ∧
  total_sugar = 11 ∧
  flour_added = 4 ∧
  sugar_flour_diff = 6 →
  total_flour = 9 :=
by sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_amount_l2880_288022


namespace NUMINAMATH_CALUDE_parallel_transitive_l2880_288066

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_l2880_288066


namespace NUMINAMATH_CALUDE_primitive_points_polynomial_theorem_l2880_288090

/-- A primitive point is an ordered pair of integers with greatest common divisor 1. -/
def PrimitivePoint : Type := { p : ℤ × ℤ // Int.gcd p.1 p.2 = 1 }

/-- The theorem statement -/
theorem primitive_points_polynomial_theorem (S : Finset PrimitivePoint) :
  ∃ (n : ℕ+) (a : Fin (n + 1) → ℤ),
    ∀ (p : PrimitivePoint), p ∈ S →
      (Finset.range (n + 1)).sum (fun i => a i * p.val.1^(n - i) * p.val.2^i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_primitive_points_polynomial_theorem_l2880_288090


namespace NUMINAMATH_CALUDE_sum_abcd_equals_neg_ten_thirds_l2880_288064

theorem sum_abcd_equals_neg_ten_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 6) : 
  a + b + c + d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_neg_ten_thirds_l2880_288064


namespace NUMINAMATH_CALUDE_coffee_maker_capacity_l2880_288030

/-- Represents a cylindrical coffee maker -/
structure CoffeeMaker where
  capacity : ℝ
  remaining : ℝ
  emptyPercentage : ℝ

/-- Theorem: A coffee maker with 30 cups remaining when 75% empty has a total capacity of 120 cups -/
theorem coffee_maker_capacity (cm : CoffeeMaker) 
  (h1 : cm.remaining = 30)
  (h2 : cm.emptyPercentage = 0.75)
  : cm.capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_capacity_l2880_288030


namespace NUMINAMATH_CALUDE_fraction_simplification_l2880_288015

theorem fraction_simplification : (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2880_288015


namespace NUMINAMATH_CALUDE_gcd_with_30_is_6_l2880_288037

theorem gcd_with_30_is_6 : ∃ n : ℕ, 70 < n ∧ n < 80 ∧ Nat.gcd n 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_with_30_is_6_l2880_288037


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2880_288025

theorem triangle_angle_value (A B C : Real) : 
  -- A, B, and C are internal angles of a triangle
  A + B + C = π → 
  0 < A → 0 < B → 0 < C →
  -- Given equation
  Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2 + Real.sin A * Real.sin B →
  -- Conclusion
  C = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2880_288025


namespace NUMINAMATH_CALUDE_five_digit_integers_with_leading_three_count_l2880_288077

/-- The count of five-digit positive integers with the ten-thousands digit 3 -/
def count_five_digit_integers_with_leading_three : ℕ :=
  10000

/-- Theorem stating that the count of five-digit positive integers 
    with the ten-thousands digit 3 is 10000 -/
theorem five_digit_integers_with_leading_three_count :
  count_five_digit_integers_with_leading_three = 10000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_integers_with_leading_three_count_l2880_288077


namespace NUMINAMATH_CALUDE_range_of_a_l2880_288017

theorem range_of_a (a : ℝ) :
  (((∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∨ 
    (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0)) ∧
   ¬((∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∧ 
     (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0))) →
  (a > 1 ∨ (-2 < a ∧ a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2880_288017


namespace NUMINAMATH_CALUDE_min_liking_both_mozart_and_bach_l2880_288083

theorem min_liking_both_mozart_and_bach
  (total : ℕ)
  (like_mozart : ℕ)
  (like_bach : ℕ)
  (h_total : total = 200)
  (h_mozart : like_mozart = 160)
  (h_bach : like_bach = 150) :
  like_mozart + like_bach - total ≥ 110 :=
by sorry

end NUMINAMATH_CALUDE_min_liking_both_mozart_and_bach_l2880_288083


namespace NUMINAMATH_CALUDE_inequality_proof_l2880_288078

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) ≥ x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2880_288078


namespace NUMINAMATH_CALUDE_proposition_relationship_l2880_288067

theorem proposition_relationship :
  (∀ a : ℝ, 0 < a ∧ a < 1 → ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧ ¬(0 < a ∧ a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l2880_288067


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2880_288062

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) + (n + 1) = 118 → n = 59 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2880_288062


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2880_288028

theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = -3) →
  (b + c) / a = -13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2880_288028


namespace NUMINAMATH_CALUDE_intersection_equals_A_l2880_288045

def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_equals_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_intersection_equals_A_l2880_288045


namespace NUMINAMATH_CALUDE_comic_books_bought_correct_comic_books_bought_l2880_288003

theorem comic_books_bought (initial : ℕ) (current : ℕ) : ℕ :=
  let sold := initial / 2
  let remaining := initial - sold
  let bought := current - remaining
  bought

theorem correct_comic_books_bought :
  comic_books_bought 22 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_bought_correct_comic_books_bought_l2880_288003


namespace NUMINAMATH_CALUDE_positive_addition_positive_multiplication_positive_division_positive_exponentiation_positive_root_extraction_l2880_288071

-- Define positive real numbers
def PositiveReal := {x : ℝ | x > 0}

-- Theorem for addition
theorem positive_addition (a b : PositiveReal) : (↑a + ↑b : ℝ) > 0 := by sorry

-- Theorem for multiplication
theorem positive_multiplication (a b : PositiveReal) : (↑a * ↑b : ℝ) > 0 := by sorry

-- Theorem for division
theorem positive_division (a b : PositiveReal) : (↑a / ↑b : ℝ) > 0 := by sorry

-- Theorem for exponentiation
theorem positive_exponentiation (a : PositiveReal) (n : ℝ) : (↑a ^ n : ℝ) > 0 := by sorry

-- Theorem for root extraction
theorem positive_root_extraction (a : PositiveReal) (n : PositiveReal) : 
  ∃ (x : ℝ), x > 0 ∧ x ^ (↑n : ℝ) = ↑a := by sorry

end NUMINAMATH_CALUDE_positive_addition_positive_multiplication_positive_division_positive_exponentiation_positive_root_extraction_l2880_288071


namespace NUMINAMATH_CALUDE_inverse_sixteen_mod_97_l2880_288033

theorem inverse_sixteen_mod_97 (h : (8⁻¹ : ZMod 97) = 85) : (16⁻¹ : ZMod 97) = 47 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sixteen_mod_97_l2880_288033


namespace NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2880_288074

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles of an octagon is 1080° -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2880_288074


namespace NUMINAMATH_CALUDE_power_equation_solution_l2880_288040

theorem power_equation_solution (a : ℝ) (k : ℝ) (h1 : a ≠ 0) : 
  (a^10 / (a^k)^4 = a^2) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2880_288040


namespace NUMINAMATH_CALUDE_unique_a_sqrt_2_l2880_288036

-- Define the set of options
def options : Set ℝ := {Real.sqrt (2/3), Real.sqrt 3, Real.sqrt 8, Real.sqrt 12}

-- Define the property of being expressible as a * √2
def is_a_sqrt_2 (x : ℝ) : Prop := ∃ (a : ℚ), x = a * Real.sqrt 2

-- Theorem statement
theorem unique_a_sqrt_2 : ∃! (x : ℝ), x ∈ options ∧ is_a_sqrt_2 x :=
sorry

end NUMINAMATH_CALUDE_unique_a_sqrt_2_l2880_288036


namespace NUMINAMATH_CALUDE_system_solution_l2880_288058

theorem system_solution (x y : ℝ) 
  (eq1 : x^2 - 4 * Real.sqrt (3*x - 2) + 6 = y)
  (eq2 : y^2 - 4 * Real.sqrt (3*y - 2) + 6 = x)
  (domain_x : 3*x - 2 ≥ 0)
  (domain_y : 3*y - 2 ≥ 0) :
  x = 2 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2880_288058


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l2880_288018

theorem one_thirds_in_nine_halves : (9 : ℚ) / 2 / (1 / 3) = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l2880_288018


namespace NUMINAMATH_CALUDE_saturn_diameter_times_ten_l2880_288081

/-- The diameter of Saturn in kilometers -/
def saturn_diameter : ℝ := 1.2 * 10^5

/-- Theorem stating the correct multiplication of Saturn's diameter by 10 -/
theorem saturn_diameter_times_ten :
  saturn_diameter * 10 = 1.2 * 10^6 := by
  sorry

end NUMINAMATH_CALUDE_saturn_diameter_times_ten_l2880_288081


namespace NUMINAMATH_CALUDE_magnitude_of_parallel_vector_difference_l2880_288054

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of their difference is 2√5. -/
theorem magnitude_of_parallel_vector_difference :
  ∀ (x : ℝ), 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 6]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →  -- Parallel condition
  ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_parallel_vector_difference_l2880_288054


namespace NUMINAMATH_CALUDE_unique_a_value_l2880_288092

def A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem unique_a_value : ∃! a : ℝ, 3 ∈ A a ∧ a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2880_288092


namespace NUMINAMATH_CALUDE_inequality_range_l2880_288044

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2880_288044


namespace NUMINAMATH_CALUDE_fraction_simplification_l2880_288073

theorem fraction_simplification : (2020 : ℚ) / (20 * 20) = 5.05 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2880_288073


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l2880_288085

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 220 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes,
    with each box containing at least one ball. -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 220 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l2880_288085


namespace NUMINAMATH_CALUDE_incorrect_equation_l2880_288007

theorem incorrect_equation (a b : ℤ) : 
  (-a + b = -1) → (a + b = 5) → (4*a + b = 14) → (2*a + b ≠ 7) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_l2880_288007


namespace NUMINAMATH_CALUDE_circumradius_eq_one_l2880_288021

/-- Three unit circles passing through a common point -/
structure ThreeIntersectingCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  center3 : ℝ × ℝ
  commonPoint : ℝ × ℝ
  radius : ℝ
  radius_eq_one : radius = 1
  passes_through_common : 
    dist center1 commonPoint = radius ∧
    dist center2 commonPoint = radius ∧
    dist center3 commonPoint = radius

/-- The three intersection points forming triangle ABC -/
def intersectionPoints (c : ThreeIntersectingCircles) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The circumcenter of triangle ABC -/
def circumcenter (points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- The circumradius of triangle ABC -/
def circumradius (c : ThreeIntersectingCircles) : ℝ :=
  let points := intersectionPoints c
  dist (circumcenter points) points.1

/-- Theorem: The circumradius of triangle ABC is equal to 1 -/
theorem circumradius_eq_one (c : ThreeIntersectingCircles) :
  circumradius c = 1 :=
sorry

end NUMINAMATH_CALUDE_circumradius_eq_one_l2880_288021


namespace NUMINAMATH_CALUDE_max_product_of_functions_l2880_288060

/-- Given two real-valued functions f and g with specified ranges,
    prove that the maximum value of their product is 14 -/
theorem max_product_of_functions (f g : ℝ → ℝ)
  (hf : Set.range f = Set.Icc (-7) 4)
  (hg : Set.range g = Set.Icc 0 2) :
  ∃ x : ℝ, f x * g x = 14 ∧ ∀ y : ℝ, f y * g y ≤ 14 := by
  sorry


end NUMINAMATH_CALUDE_max_product_of_functions_l2880_288060


namespace NUMINAMATH_CALUDE_vector_collinearity_l2880_288093

def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem vector_collinearity (x : ℝ) :
  (∃ (k : ℝ), a - b x = k • b x) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2880_288093


namespace NUMINAMATH_CALUDE_expression_value_l2880_288001

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 - z^2 + 3*x*y = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2880_288001


namespace NUMINAMATH_CALUDE_no_common_root_l2880_288088

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x₀ : ℝ, (x₀^2 + b*x₀ + c = 0) ∧ (x₀^2 + a*x₀ + d = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_root_l2880_288088


namespace NUMINAMATH_CALUDE_direct_proportion_problem_l2880_288059

/-- A direct proportion function -/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

theorem direct_proportion_problem (f : ℝ → ℝ) 
  (h1 : DirectProportion f) 
  (h2 : f (-2) = 4) : 
  f 3 = -6 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_problem_l2880_288059


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2880_288016

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x^(1/4) = 12 / (7 - x^(1/4))) ↔ (x = 81 ∨ x = 256) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2880_288016


namespace NUMINAMATH_CALUDE_probability_is_one_eighth_l2880_288094

/-- A standard die with 8 sides -/
def StandardDie : Finset ℕ := Finset.range 8

/-- The set of all possible outcomes when rolling the die twice -/
def AllOutcomes : Finset (ℕ × ℕ) := StandardDie.product StandardDie

/-- The set of favorable outcomes (pairs that differ by 3) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  AllOutcomes.filter (fun p => (p.1 + 3 = p.2) ∨ (p.2 + 3 = p.1))

/-- The probability of rolling two integers that differ by 3 -/
def probability : ℚ := (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_is_one_eighth :
  probability = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_eighth_l2880_288094


namespace NUMINAMATH_CALUDE_expression_factorization_l2880_288082

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l2880_288082


namespace NUMINAMATH_CALUDE_total_portfolios_l2880_288075

theorem total_portfolios (num_students : ℕ) (portfolios_per_student : ℕ) 
  (h1 : num_students = 15)
  (h2 : portfolios_per_student = 8) :
  num_students * portfolios_per_student = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_portfolios_l2880_288075


namespace NUMINAMATH_CALUDE_searchlight_probability_l2880_288002

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 2

/-- The time in seconds for which the man needs to stay in the dark -/
def dark_time : ℝ := 5

/-- The number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

theorem searchlight_probability :
  let time_per_revolution := seconds_per_minute / revolutions_per_minute
  (dark_time / time_per_revolution : ℝ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_searchlight_probability_l2880_288002


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2880_288048

/-- A circle with center on the y-axis, radius 1, and tangent to y = 2 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  tangent_to_y_2 : ∃ (x : ℝ), (center.1 - x)^2 + (center.2 - 2)^2 = radius^2

/-- The equation of a TangentCircle is x^2 + (y-3)^2 = 1 or x^2 + (y-1)^2 = 1 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∃ (y₀ : ℝ), y₀ = 1 ∨ y₀ = 3 ∧ ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2} ↔ x^2 + (y - y₀)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2880_288048


namespace NUMINAMATH_CALUDE_car_original_price_l2880_288050

/-- Proves the original price of a car given repair cost, selling price, and profit percentage -/
theorem car_original_price (repair_cost selling_price : ℝ) (profit_percentage : ℝ) :
  repair_cost = 12000 →
  selling_price = 80000 →
  profit_percentage = 40.35 →
  ∃ (original_price : ℝ),
    (selling_price - (original_price + repair_cost)) / original_price * 100 = profit_percentage ∧
    abs (original_price - 48425.44) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_original_price_l2880_288050


namespace NUMINAMATH_CALUDE_smallest_class_size_l2880_288072

/-- Represents a class of students and their test scores. -/
structure TestClass where
  n : ℕ              -- Total number of students
  scores : Fin n → ℕ -- Scores of each student

/-- Conditions for the test class. -/
def validTestClass (c : TestClass) : Prop :=
  (∀ i, c.scores i ≥ 70 ∧ c.scores i ≤ 120) ∧
  (∃ s : Finset (Fin c.n), s.card = 7 ∧ ∀ i ∈ s, c.scores i = 120) ∧
  (Finset.sum (Finset.univ : Finset (Fin c.n)) c.scores / c.n = 85)

/-- The theorem stating the smallest possible number of students. -/
theorem smallest_class_size :
  ∀ c : TestClass, validTestClass c → c.n ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2880_288072


namespace NUMINAMATH_CALUDE_andrew_payment_l2880_288041

/-- Calculate the total amount Andrew paid to the shopkeeper for grapes and mangoes. -/
theorem andrew_payment (grape_quantity : ℕ) (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) :
  grape_quantity = 11 →
  grape_price = 98 →
  mango_quantity = 7 →
  mango_price = 50 →
  grape_quantity * grape_price + mango_quantity * mango_price = 1428 :=
by
  sorry

#check andrew_payment

end NUMINAMATH_CALUDE_andrew_payment_l2880_288041


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2880_288080

theorem complex_equation_solution (z : ℂ) (h : z * (2 - Complex.I) = 5 * Complex.I) :
  z = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2880_288080


namespace NUMINAMATH_CALUDE_concert_songs_count_l2880_288076

theorem concert_songs_count (total_duration intermission_duration regular_song_duration special_song_duration : ℕ) 
  (h1 : total_duration = 80)
  (h2 : intermission_duration = 10)
  (h3 : regular_song_duration = 5)
  (h4 : special_song_duration = 10) :
  (total_duration - intermission_duration - special_song_duration) / regular_song_duration + 1 = 13 := by
  sorry

#check concert_songs_count

end NUMINAMATH_CALUDE_concert_songs_count_l2880_288076


namespace NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_max_k_no_intersection_l2880_288010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

theorem tangent_parallel_condition (a : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, f a x = f a 1 + k * (x - 1)) ↔ a = Real.exp 1 := by sorry

theorem extreme_values (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > 0 → ∃ x : ℝ, x = Real.log a ∧ ∀ y : ℝ, y ≠ x → f a x < f a y) := by sorry

theorem max_k_no_intersection :
  ∃ k : ℝ, k = 1 ∧
    (∀ k' : ℝ, (∀ x : ℝ, f 1 x ≠ k' * x - 1) → k' ≤ k) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_max_k_no_intersection_l2880_288010


namespace NUMINAMATH_CALUDE_grapes_in_robs_bowl_l2880_288005

theorem grapes_in_robs_bowl (rob_grapes : ℕ) 
  (allie_grapes : ℕ) (allyn_grapes : ℕ) : 
  (allie_grapes = rob_grapes + 2) → 
  (allyn_grapes = allie_grapes + 4) → 
  (rob_grapes + allie_grapes + allyn_grapes = 83) → 
  rob_grapes = 25 := by
sorry

end NUMINAMATH_CALUDE_grapes_in_robs_bowl_l2880_288005


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l2880_288009

/-- Represents the number of participants in a soccer league for two consecutive years -/
structure LeagueParticipation where
  malesLastYear : ℕ
  femalesLastYear : ℕ
  malesThisYear : ℕ
  femalesThisYear : ℕ

/-- Calculates the fraction of female participants this year -/
def femaleFraction (lp : LeagueParticipation) : Rat :=
  lp.femalesThisYear / (lp.malesThisYear + lp.femalesThisYear)

theorem soccer_league_female_fraction 
  (lp : LeagueParticipation)
  (male_increase : lp.malesThisYear = (110 * lp.malesLastYear) / 100)
  (female_increase : lp.femalesThisYear = (125 * lp.femalesLastYear) / 100)
  (total_increase : lp.malesThisYear + lp.femalesThisYear = 
    (115 * (lp.malesLastYear + lp.femalesLastYear)) / 100)
  (males_last_year : lp.malesLastYear = 30)
  : femaleFraction lp = 19 / 52 := by
  sorry

#check soccer_league_female_fraction

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l2880_288009


namespace NUMINAMATH_CALUDE_proposition_is_false_l2880_288063

theorem proposition_is_false : ¬(∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l2880_288063


namespace NUMINAMATH_CALUDE_solution_set_is_correct_l2880_288056

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The solution set of the inequality floor(x)^2 - 5*floor(x) + 6 ≤ 0 -/
def solution_set : Set ℝ :=
  {x : ℝ | (floor x)^2 - 5*(floor x) + 6 ≤ 0}

/-- Theorem stating that the solution set is [2,4) -/
theorem solution_set_is_correct : solution_set = Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_correct_l2880_288056


namespace NUMINAMATH_CALUDE_complex_power_220_36_l2880_288091

theorem complex_power_220_36 : (Complex.exp (220 * π / 180 * I))^36 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_220_36_l2880_288091


namespace NUMINAMATH_CALUDE_church_members_count_l2880_288013

theorem church_members_count :
  ∀ (total adults children : ℕ),
  adults = (40 * total) / 100 →
  children = total - adults →
  children = adults + 24 →
  total = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_church_members_count_l2880_288013


namespace NUMINAMATH_CALUDE_distance_between_intersections_l2880_288020

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ

-- Define the ray
def ray (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∧ x ≥ 0

-- Define the intersection points
def intersectionC₁ (x y : ℝ) : Prop := C₁ x y ∧ ray x y
def intersectionC₂ (x y : ℝ) : Prop := C₂ x y ∧ ray x y

-- Theorem statement
theorem distance_between_intersections :
  ∃ (A B : ℝ × ℝ),
    intersectionC₁ A.1 A.2 ∧
    intersectionC₂ B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersections_l2880_288020


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2880_288057

theorem smallest_n_congruence : ∃ n : ℕ+, (∀ m : ℕ+, 813 * m ≡ 1224 * m [ZMOD 30] → n ≤ m) ∧ 813 * n ≡ 1224 * n [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2880_288057
