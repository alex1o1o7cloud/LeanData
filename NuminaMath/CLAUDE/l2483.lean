import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l2483_248380

theorem circle_equation_from_diameter (P Q : ℝ × ℝ) : 
  P = (4, 0) → Q = (0, 2) → 
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 5 ↔ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      x = 4 * (1 - t) + 0 * t ∧ 
      y = 0 * (1 - t) + 2 * t ∧
      (x - 4)^2 + (y - 0)^2 = (0 - 4)^2 + (2 - 0)^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l2483_248380


namespace NUMINAMATH_CALUDE_cone_height_l2483_248388

/-- The height of a cone with base area π and slant height 2 is √3 -/
theorem cone_height (base_area : Real) (slant_height : Real) :
  base_area = Real.pi → slant_height = 2 → ∃ (height : Real), height = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l2483_248388


namespace NUMINAMATH_CALUDE_stones_sent_away_l2483_248319

theorem stones_sent_away (original_stones kept_stones : ℕ) 
  (h1 : original_stones = 78) 
  (h2 : kept_stones = 15) : 
  original_stones - kept_stones = 63 := by
  sorry

end NUMINAMATH_CALUDE_stones_sent_away_l2483_248319


namespace NUMINAMATH_CALUDE_mr_gates_classes_l2483_248373

/-- Proves that given the conditions in the problem, Mr. Gates has 4 classes --/
theorem mr_gates_classes : 
  ∀ (buns_per_package : ℕ) 
    (packages_bought : ℕ) 
    (students_per_class : ℕ) 
    (buns_per_student : ℕ),
  buns_per_package = 8 →
  packages_bought = 30 →
  students_per_class = 30 →
  buns_per_student = 2 →
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 :=
by sorry

end NUMINAMATH_CALUDE_mr_gates_classes_l2483_248373


namespace NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l2483_248340

-- Define the property of a function satisfying the given condition
def SatisfiesCondition (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

-- Define the set X_v
def X_v (f : ℤ → ℤ) (v : ℤ) : Set ℤ :=
  {x : ℤ | f x = v}

-- Define what it means for an integer to be rare under f
def IsRare (f : ℤ → ℤ) (v : ℤ) : Prop :=
  (X_v f v).Nonempty ∧ (X_v f v).Finite

-- Theorem statement
theorem rare_integer_existence_and_uniqueness :
  (∃ f : ℤ → ℤ, SatisfiesCondition f ∧ ∃ v : ℤ, IsRare f v) ∧
  (∀ f : ℤ → ℤ, SatisfiesCondition f → ∀ v w : ℤ, IsRare f v → IsRare f w → v = w) :=
sorry

end NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l2483_248340


namespace NUMINAMATH_CALUDE_apple_profit_percentage_l2483_248335

/-- Calculates the total profit percentage for a stock of apples -/
theorem apple_profit_percentage 
  (total_stock : ℝ)
  (first_portion : ℝ)
  (second_portion : ℝ)
  (profit_rate : ℝ)
  (h1 : total_stock = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : profit_rate = 0.3)
  (h5 : first_portion + second_portion = 1) :
  let total_sp := (first_portion * total_stock * (1 + profit_rate)) + 
                  (second_portion * total_stock * (1 + profit_rate))
  let total_profit := total_sp - total_stock
  let total_profit_percentage := (total_profit / total_stock) * 100
  total_profit_percentage = 30 := by
sorry


end NUMINAMATH_CALUDE_apple_profit_percentage_l2483_248335


namespace NUMINAMATH_CALUDE_line_segment_slope_l2483_248322

theorem line_segment_slope (m n p : ℝ) : 
  (m = 4 * n + 5) → 
  (m + 2 = 4 * (n + p) + 5) → 
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_slope_l2483_248322


namespace NUMINAMATH_CALUDE_caffeine_content_proof_l2483_248351

/-- The amount of caffeine in one energy drink -/
def caffeine_per_drink : ℕ := sorry

/-- The maximum safe amount of caffeine per day -/
def max_safe_caffeine : ℕ := 500

/-- The number of energy drinks Brandy consumes -/
def num_drinks : ℕ := 4

/-- The additional amount of caffeine Brandy can safely consume after drinking the energy drinks -/
def additional_safe_caffeine : ℕ := 20

theorem caffeine_content_proof :
  caffeine_per_drink * num_drinks + additional_safe_caffeine = max_safe_caffeine ∧
  caffeine_per_drink = 120 := by sorry

end NUMINAMATH_CALUDE_caffeine_content_proof_l2483_248351


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2483_248321

/-- Represents the capacity of a tank with a leak and two inlet pipes. -/
def tank_capacity : Real :=
  let leak_rate : Real := tank_capacity / 6
  let pipe_a_rate : Real := 3.5 * 60
  let pipe_b_rate : Real := 4.5 * 60
  let net_rate_both_pipes : Real := pipe_a_rate + pipe_b_rate - leak_rate
  let net_rate_pipe_a : Real := pipe_a_rate - leak_rate
  tank_capacity

/-- Theorem stating the capacity of the tank under given conditions. -/
theorem tank_capacity_proof :
  let leak_rate : Real := tank_capacity / 6
  let pipe_a_rate : Real := 3.5 * 60
  let pipe_b_rate : Real := 4.5 * 60
  let net_rate_both_pipes : Real := pipe_a_rate + pipe_b_rate - leak_rate
  let net_rate_pipe_a : Real := pipe_a_rate - leak_rate
  (net_rate_pipe_a * 1 + net_rate_both_pipes * 7 - leak_rate * 8 = 0) →
  tank_capacity = 1338.75 := by
  sorry

#eval tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_proof_l2483_248321


namespace NUMINAMATH_CALUDE_john_change_proof_l2483_248391

/-- Calculates the change received when buying oranges -/
def calculate_change (num_oranges : ℕ) (cost_per_orange_cents : ℕ) (paid_dollars : ℕ) : ℚ :=
  paid_dollars - (num_oranges * cost_per_orange_cents) / 100

theorem john_change_proof :
  calculate_change 4 75 10 = 7 := by
  sorry

#eval calculate_change 4 75 10

end NUMINAMATH_CALUDE_john_change_proof_l2483_248391


namespace NUMINAMATH_CALUDE_pages_of_maps_skipped_l2483_248364

theorem pages_of_maps_skipped (total_pages read_pages pages_left : ℕ) 
  (h1 : total_pages = 372)
  (h2 : read_pages = 125)
  (h3 : pages_left = 231) :
  total_pages - (read_pages + pages_left) = 16 := by
  sorry

end NUMINAMATH_CALUDE_pages_of_maps_skipped_l2483_248364


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2483_248307

theorem unique_positive_solution (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^y = z ∧ y^z = x ∧ z^x = y →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2483_248307


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2483_248310

theorem factorial_divisibility (a : ℕ) : 
  (a.factorial + (a + 2).factorial) ∣ (a + 4).factorial ↔ a = 0 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2483_248310


namespace NUMINAMATH_CALUDE_largest_two_digit_product_l2483_248313

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_single_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

theorem largest_two_digit_product :
  ∃ (n x : ℕ), 
    is_two_digit n ∧
    is_single_digit x ∧
    n = x * (10 * x + 2 * x) ∧
    ∀ (m y : ℕ), 
      is_two_digit m → 
      is_single_digit y → 
      m = y * (10 * y + 2 * y) → 
      m ≤ n ∧
    n = 48 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_product_l2483_248313


namespace NUMINAMATH_CALUDE_smiths_children_ages_l2483_248352

def is_divisible (n m : ℕ) : Prop := m ∣ n

theorem smiths_children_ages (children_ages : Finset ℕ) : 
  children_ages.card = 7 ∧ 
  (∀ a ∈ children_ages, 2 ≤ a ∧ a ≤ 11) ∧
  (∃ x, 2 ≤ x ∧ x ≤ 11 ∧ x ∉ children_ages) ∧
  5 ∈ children_ages ∧
  (∀ a ∈ children_ages, is_divisible 3339 a) ∧
  39 ∉ children_ages →
  6 ∉ children_ages :=
by sorry

end NUMINAMATH_CALUDE_smiths_children_ages_l2483_248352


namespace NUMINAMATH_CALUDE_ball_probabilities_l2483_248349

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 12

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing a yellow ball -/
def prob_yellow : ℚ := 0

/-- Represents the probability of drawing at least one red ball out of 9 balls drawn at once -/
def prob_at_least_one_red : ℚ := 1

/-- Represents the probability of drawing a red ball at random -/
def prob_red : ℚ := 3 / 5

/-- Represents the number of red balls removed and white balls added -/
def x : ℕ := 8

theorem ball_probabilities :
  (prob_yellow = 0) ∧
  (prob_at_least_one_red = 1) ∧
  (prob_red = 3 / 5) ∧
  (((initial_white_balls + x : ℚ) / total_balls) = 4 / 5 → x = 8) :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2483_248349


namespace NUMINAMATH_CALUDE_g_12_equals_191_l2483_248318

def g (n : ℕ) : ℕ := n^2 + 2*n + 23

theorem g_12_equals_191 : g 12 = 191 := by
  sorry

end NUMINAMATH_CALUDE_g_12_equals_191_l2483_248318


namespace NUMINAMATH_CALUDE_balloons_left_l2483_248316

theorem balloons_left (round_bags : ℕ) (round_per_bag : ℕ) (long_bags : ℕ) (long_per_bag : ℕ) (burst : ℕ) : 
  round_bags = 5 → 
  round_per_bag = 20 → 
  long_bags = 4 → 
  long_per_bag = 30 → 
  burst = 5 → 
  round_bags * round_per_bag + long_bags * long_per_bag - burst = 215 := by
  sorry

end NUMINAMATH_CALUDE_balloons_left_l2483_248316


namespace NUMINAMATH_CALUDE_x_plus_y_equals_one_l2483_248317

theorem x_plus_y_equals_one (x y : ℝ) 
  (h1 : 2021 * x + 2025 * y = 2029)
  (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_one_l2483_248317


namespace NUMINAMATH_CALUDE_bianca_candy_eaten_l2483_248347

theorem bianca_candy_eaten (total : ℕ) (piles : ℕ) (per_pile : ℕ) 
  (h1 : total = 78)
  (h2 : piles = 6)
  (h3 : per_pile = 8) :
  total - (piles * per_pile) = 30 := by
  sorry

end NUMINAMATH_CALUDE_bianca_candy_eaten_l2483_248347


namespace NUMINAMATH_CALUDE_sample_size_equals_sampled_students_l2483_248348

/-- Represents a survey conducted on eighth-grade students -/
structure Survey where
  sampled_students : ℕ

/-- The sample size of a survey is equal to the number of sampled students -/
theorem sample_size_equals_sampled_students (s : Survey) : s.sampled_students = 1500 → s.sampled_students = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_equals_sampled_students_l2483_248348


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_union_equals_A_iff_l2483_248394

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: When a = 0, A ∩ B = {x | -1 < x < 5}
theorem intersection_when_a_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: A ∪ B = A if and only if a ∈ (0, 1] ∪ [6, +∞)
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_union_equals_A_iff_l2483_248394


namespace NUMINAMATH_CALUDE_hockey_games_per_month_l2483_248314

/-- 
Given a hockey season with the following properties:
- The season lasts for 14 months
- There are 182 hockey games in the season

This theorem proves that the number of hockey games played each month is 13.
-/
theorem hockey_games_per_month (season_length : ℕ) (total_games : ℕ) 
  (h1 : season_length = 14) (h2 : total_games = 182) :
  total_games / season_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_hockey_games_per_month_l2483_248314


namespace NUMINAMATH_CALUDE_plot_length_is_57_l2483_248353

/-- A rectangular plot with specific fencing cost and length-breadth relationship -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_relation : length = breadth + 14
  fencing_cost_equation : total_fencing_cost = fencing_cost_per_meter * (2 * length + 2 * breadth)

/-- The length of the rectangular plot is 57 meters -/
theorem plot_length_is_57 (plot : RectangularPlot) 
    (h1 : plot.fencing_cost_per_meter = 26.5)
    (h2 : plot.total_fencing_cost = 5300) : 
  plot.length = 57 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_57_l2483_248353


namespace NUMINAMATH_CALUDE_trivia_team_size_l2483_248395

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  absent_members = 2 →
  points_per_member = 6 →
  total_points = 18 →
  ∃ initial_members : ℕ, initial_members = 5 ∧ 
    points_per_member * (initial_members - absent_members) = total_points :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_size_l2483_248395


namespace NUMINAMATH_CALUDE_lcm_gcd_product_equals_number_product_l2483_248389

theorem lcm_gcd_product_equals_number_product : 
  let a := 24
  let b := 36
  Nat.lcm a b * Nat.gcd a b = a * b :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_equals_number_product_l2483_248389


namespace NUMINAMATH_CALUDE_min_value_implies_b_range_l2483_248358

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 6*b*x + 3*b

-- Define the derivative of f
def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*b

-- State the theorem
theorem min_value_implies_b_range (b : ℝ) :
  (∃ x ∈ (Set.Ioo 0 1), ∀ y ∈ (Set.Ioo 0 1), f b x ≤ f b y) →
  b ∈ (Set.Ioo 0 (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_b_range_l2483_248358


namespace NUMINAMATH_CALUDE_polynomial_symmetry_representation_l2483_248370

theorem polynomial_symmetry_representation (p : ℝ → ℝ) (a : ℝ) 
  (h_symmetry : ∀ x, p x = p (a - x)) :
  ∃ h : ℝ → ℝ, ∀ x, p x = h ((x - a / 2) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_representation_l2483_248370


namespace NUMINAMATH_CALUDE_division_simplification_l2483_248303

theorem division_simplification (x y : ℝ) : -6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2483_248303


namespace NUMINAMATH_CALUDE_paula_candy_distribution_l2483_248332

def minimum_candies (initial_candies : ℕ) (num_friends : ℕ) : ℕ :=
  let total := initial_candies + (num_friends - initial_candies % num_friends) % num_friends
  total

theorem paula_candy_distribution (initial_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : num_friends = 10) :
  minimum_candies initial_candies num_friends = 30 ∧
  minimum_candies initial_candies num_friends / num_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_paula_candy_distribution_l2483_248332


namespace NUMINAMATH_CALUDE_correct_num_arrangements_l2483_248339

/-- The number of arrangements of 3 girls and 6 boys in a row, 
    with boys at both ends and no two girls adjacent. -/
def num_arrangements : ℕ := 43200

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of boys -/
def num_boys : ℕ := 6

/-- Theorem stating that the number of arrangements satisfying the given conditions is 43200 -/
theorem correct_num_arrangements : 
  (num_girls = 3 ∧ num_boys = 6) → num_arrangements = 43200 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_arrangements_l2483_248339


namespace NUMINAMATH_CALUDE_sum_due_theorem_l2483_248315

/-- Calculates the sum due given the true discount, interest rate, and time period. -/
def sum_due (true_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  let present_value := (true_discount * 100) / (interest_rate * time)
  present_value + true_discount

/-- Proves that the sum due is 568 given the specified conditions. -/
theorem sum_due_theorem :
  sum_due 168 14 3 = 568 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_theorem_l2483_248315


namespace NUMINAMATH_CALUDE_expression_value_l2483_248385

theorem expression_value : 
  (2024^3 - 2 * 2024^2 * 2025 + 3 * 2024 * 2025^2 - 2025^3 + 4) / (2024 * 2025) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2483_248385


namespace NUMINAMATH_CALUDE_angle_A_measure_l2483_248302

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Define a triangle ABC with sides a, b, c opposite to angles A, B, C
  true  -- Placeholder, as we don't need to specify all triangle properties

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) :
  triangle_ABC A B C a b c →
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l2483_248302


namespace NUMINAMATH_CALUDE_remaining_cookies_l2483_248372

theorem remaining_cookies (white_initial : ℕ) (black_initial : ℕ) : 
  white_initial = 80 →
  black_initial = white_initial + 50 →
  (white_initial - (3 * white_initial / 4)) + (black_initial / 2) = 85 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cookies_l2483_248372


namespace NUMINAMATH_CALUDE_hotel_air_conditioning_l2483_248343

theorem hotel_air_conditioning (total_rooms : ℝ) (total_rooms_pos : 0 < total_rooms) : 
  let rented_rooms := (3/4 : ℝ) * total_rooms
  let air_conditioned_rooms := (3/5 : ℝ) * total_rooms
  let rented_air_conditioned := (2/3 : ℝ) * air_conditioned_rooms
  let not_rented_rooms := total_rooms - rented_rooms
  let not_rented_air_conditioned := air_conditioned_rooms - rented_air_conditioned
  (not_rented_air_conditioned / not_rented_rooms) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_hotel_air_conditioning_l2483_248343


namespace NUMINAMATH_CALUDE_negative_inequality_l2483_248360

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l2483_248360


namespace NUMINAMATH_CALUDE_inequality_condition_l2483_248329

theorem inequality_condition :
  (∀ a b c d : ℝ, (a > b ∧ c > d) → (a + c > b + d)) ∧
  (∃ a b c d : ℝ, (a + c > b + d) ∧ ¬(a > b ∧ c > d)) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l2483_248329


namespace NUMINAMATH_CALUDE_taco_truck_profit_l2483_248308

/-- Calculates the profit for a taco truck given the specified conditions -/
theorem taco_truck_profit
  (total_beef : ℝ)
  (beef_per_taco : ℝ)
  (selling_price : ℝ)
  (cost_per_taco : ℝ)
  (h1 : total_beef = 100)
  (h2 : beef_per_taco = 0.25)
  (h3 : selling_price = 2)
  (h4 : cost_per_taco = 1.5) :
  (total_beef / beef_per_taco) * (selling_price - cost_per_taco) = 200 :=
by sorry

end NUMINAMATH_CALUDE_taco_truck_profit_l2483_248308


namespace NUMINAMATH_CALUDE_vector_sum_problem_l2483_248355

theorem vector_sum_problem :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![-2, 4]
  (v1 + 3 • v2) = ![-1, 9] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_problem_l2483_248355


namespace NUMINAMATH_CALUDE_sin_double_angle_given_sin_pi_fourth_minus_x_l2483_248336

theorem sin_double_angle_given_sin_pi_fourth_minus_x
  (x : ℝ) (h : Real.sin (π/4 - x) = 3/5) :
  Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_given_sin_pi_fourth_minus_x_l2483_248336


namespace NUMINAMATH_CALUDE_alfonso_helmet_weeks_l2483_248330

/-- Calculates the number of weeks Alfonso needs to work to buy a helmet -/
def weeks_to_buy_helmet (daily_earnings : ℚ) (days_per_week : ℕ) (helmet_cost : ℚ) (savings : ℚ) : ℚ :=
  (helmet_cost - savings) / (daily_earnings * days_per_week)

/-- Proves that Alfonso needs 10 weeks to buy the helmet -/
theorem alfonso_helmet_weeks : 
  let daily_earnings : ℚ := 6
  let days_per_week : ℕ := 5
  let helmet_cost : ℚ := 340
  let savings : ℚ := 40
  weeks_to_buy_helmet daily_earnings days_per_week helmet_cost savings = 10 := by
sorry

#eval weeks_to_buy_helmet 6 5 340 40

end NUMINAMATH_CALUDE_alfonso_helmet_weeks_l2483_248330


namespace NUMINAMATH_CALUDE_printing_speed_proof_l2483_248366

/-- Mike's initial printing speed in pamphlets per hour -/
def initial_speed : ℕ := 600

/-- Total number of pamphlets printed -/
def total_pamphlets : ℕ := 9400

/-- Mike's initial printing time in hours -/
def mike_initial_time : ℕ := 9

/-- Mike's reduced speed printing time in hours -/
def mike_reduced_time : ℕ := 2

/-- Leo's printing time in hours -/
def leo_time : ℕ := 3

theorem printing_speed_proof :
  initial_speed * mike_initial_time + 
  (initial_speed / 3) * mike_reduced_time + 
  (2 * initial_speed) * leo_time = total_pamphlets :=
by sorry

end NUMINAMATH_CALUDE_printing_speed_proof_l2483_248366


namespace NUMINAMATH_CALUDE_polynomial_product_constraint_l2483_248328

theorem polynomial_product_constraint (a b : ℝ) : 
  (∀ x, (a * x + b) * (2 * x + 1) = 2 * a * x^2 + b) ∧ b = 6 → a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_constraint_l2483_248328


namespace NUMINAMATH_CALUDE_annika_hiking_rate_l2483_248309

/-- Annika's hiking problem -/
theorem annika_hiking_rate (initial_distance : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  initial_distance = 2.75 →
  total_distance = 3.5 →
  total_time = 51 →
  (total_time / (2 * (total_distance - initial_distance))) = 34 :=
by
  sorry

#check annika_hiking_rate

end NUMINAMATH_CALUDE_annika_hiking_rate_l2483_248309


namespace NUMINAMATH_CALUDE_spring_mows_count_l2483_248357

def total_mows : ℕ := 11
def summer_mows : ℕ := 5

theorem spring_mows_count : total_mows - summer_mows = 6 := by
  sorry

end NUMINAMATH_CALUDE_spring_mows_count_l2483_248357


namespace NUMINAMATH_CALUDE_fence_posts_count_l2483_248338

/-- Represents the fence setup for a rectangular field -/
structure FenceSetup where
  wallLength : ℕ
  rectLength : ℕ
  rectWidth : ℕ
  postSpacing : ℕ
  gateWidth : ℕ

/-- Calculates the number of posts required for the fence setup -/
def calculatePosts (setup : FenceSetup) : ℕ :=
  sorry

/-- Theorem stating that the specific fence setup requires 19 posts -/
theorem fence_posts_count :
  let setup : FenceSetup := {
    wallLength := 120,
    rectLength := 80,
    rectWidth := 50,
    postSpacing := 10,
    gateWidth := 20
  }
  calculatePosts setup = 19 := by
  sorry

end NUMINAMATH_CALUDE_fence_posts_count_l2483_248338


namespace NUMINAMATH_CALUDE_total_marble_weight_l2483_248363

def marble_weights : List Float := [
  0.3333333333333333,
  0.3333333333333333,
  0.08333333333333333,
  0.21666666666666667,
  0.4583333333333333,
  0.12777777777777778
]

theorem total_marble_weight :
  marble_weights.sum = 1.5527777777777777 := by sorry

end NUMINAMATH_CALUDE_total_marble_weight_l2483_248363


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2483_248301

theorem triangle_equilateral (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * a * b * c →
  Real.sin A = 2 * Real.sin B * Real.cos C →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a = b ∧ b = c ∧ A = B ∧ B = C ∧ A = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l2483_248301


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_logarithmic_inequality_l2483_248381

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) :=
by sorry

theorem negation_of_logarithmic_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x + x - 1 ≤ 0) ↔
  (∀ x : ℝ, x > 0 → Real.log x + x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_logarithmic_inequality_l2483_248381


namespace NUMINAMATH_CALUDE_is_centre_of_hyperbola_l2483_248396

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0

/-- The centre of the hyperbola -/
def hyperbola_centre : ℝ × ℝ := (1, 2)

/-- Theorem stating that the given point is the centre of the hyperbola -/
theorem is_centre_of_hyperbola :
  let (h, k) := hyperbola_centre
  ∀ (a b : ℝ), hyperbola_equation (h + a) (k + b) ↔ hyperbola_equation (h - a) (k - b) :=
by sorry

end NUMINAMATH_CALUDE_is_centre_of_hyperbola_l2483_248396


namespace NUMINAMATH_CALUDE_vote_intersection_l2483_248382

theorem vote_intersection (U A B : Finset Int) (h1 : U.card = 300) 
  (h2 : A.card = 230) (h3 : B.card = 190) (h4 : (U \ A).card + (U \ B).card - U.card = 40) :
  (A ∩ B).card = 160 := by
  sorry

end NUMINAMATH_CALUDE_vote_intersection_l2483_248382


namespace NUMINAMATH_CALUDE_total_squares_blocked_5x6_grid_l2483_248327

/-- Represents a partially blocked grid -/
structure PartialGrid :=
  (rows : Nat)
  (cols : Nat)
  (squares_1x1 : Nat)
  (squares_2x2 : Nat)
  (squares_3x3 : Nat)
  (squares_4x4 : Nat)

/-- Calculates the total number of squares in a partially blocked grid -/
def total_squares (grid : PartialGrid) : Nat :=
  grid.squares_1x1 + grid.squares_2x2 + grid.squares_3x3 + grid.squares_4x4

/-- Theorem: The total number of squares in the given partially blocked 5x6 grid is 57 -/
theorem total_squares_blocked_5x6_grid :
  ∃ (grid : PartialGrid),
    grid.rows = 5 ∧
    grid.cols = 6 ∧
    grid.squares_1x1 = 30 ∧
    grid.squares_2x2 = 18 ∧
    grid.squares_3x3 = 7 ∧
    grid.squares_4x4 = 2 ∧
    total_squares grid = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_total_squares_blocked_5x6_grid_l2483_248327


namespace NUMINAMATH_CALUDE_connie_grandmother_birth_year_l2483_248379

/-- Calculates the birth year of Connie's grandmother given the birth years of her siblings and the gap condition. -/
def grandmotherBirthYear (brotherBirthYear sisterBirthYear : ℕ) : ℕ :=
  let siblingGap := sisterBirthYear - brotherBirthYear
  sisterBirthYear - 2 * siblingGap

/-- Proves that Connie's grandmother was born in 1928 given the known conditions. -/
theorem connie_grandmother_birth_year :
  grandmotherBirthYear 1932 1936 = 1928 := by
  sorry

#eval grandmotherBirthYear 1932 1936

end NUMINAMATH_CALUDE_connie_grandmother_birth_year_l2483_248379


namespace NUMINAMATH_CALUDE_lollipop_collection_time_l2483_248398

theorem lollipop_collection_time (total_sticks : ℕ) (visits_per_week : ℕ) (completion_percentage : ℚ) : 
  total_sticks = 400 →
  visits_per_week = 3 →
  completion_percentage = 3/5 →
  (total_sticks * completion_percentage / visits_per_week : ℚ) = 80 := by
sorry

end NUMINAMATH_CALUDE_lollipop_collection_time_l2483_248398


namespace NUMINAMATH_CALUDE_polynomial_B_value_l2483_248325

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36

theorem polynomial_B_value :
  ∀ (A B C D : ℤ),
  (∀ r : ℤ, polynomial r A B C D = 0 → r > 0) →
  (∃ r1 r2 r3 r4 r5 r6 : ℤ, 
    ∀ z : ℤ, polynomial z A B C D = (z - r1) * (z - r2) * (z - r3) * (z - r4) * (z - r5) * (z - r6)) →
  B = -136 := by
sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l2483_248325


namespace NUMINAMATH_CALUDE_aloks_order_l2483_248311

/-- Given Alok's order and payment information, prove the number of mixed vegetable plates ordered -/
theorem aloks_order (chapati_count : ℕ) (rice_count : ℕ) (icecream_count : ℕ) 
  (chapati_cost : ℕ) (rice_cost : ℕ) (vegetable_cost : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  rice_count = 5 →
  icecream_count = 6 →
  chapati_cost = 6 →
  rice_cost = 45 →
  vegetable_cost = 70 →
  total_paid = 1015 →
  ∃ (vegetable_count : ℕ), 
    total_paid = chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost + 
      (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) ∧
    vegetable_count = 9 :=
by sorry

end NUMINAMATH_CALUDE_aloks_order_l2483_248311


namespace NUMINAMATH_CALUDE_amount_paid_is_correct_l2483_248333

/-- Calculates the amount paid to Jerry after discount --/
def amount_paid_after_discount (
  painting_hours : ℕ)
  (painting_rate : ℚ)
  (mowing_hours : ℕ)
  (mowing_rate : ℚ)
  (plumbing_hours : ℕ)
  (plumbing_rate : ℚ)
  (counter_time_multiplier : ℕ)
  (discount_rate : ℚ) : ℚ :=
  let painting_cost := painting_hours * painting_rate
  let mowing_cost := mowing_hours * mowing_rate
  let plumbing_cost := plumbing_hours * plumbing_rate
  let total_cost := painting_cost + mowing_cost + plumbing_cost
  let discount := total_cost * discount_rate
  total_cost - discount

/-- Theorem stating that Miss Stevie paid $226.80 after the discount --/
theorem amount_paid_is_correct : 
  amount_paid_after_discount 8 15 6 10 4 18 3 (1/10) = 226.8 := by
  sorry

end NUMINAMATH_CALUDE_amount_paid_is_correct_l2483_248333


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_and_average_l2483_248342

theorem consecutive_integers_sum_and_average (n : ℤ) :
  let consecutive_integers := [n+1, n+2, n+3, n+4, n+5, n+6]
  (consecutive_integers.sum = 6*n + 21) ∧ 
  (consecutive_integers.sum / 6 : ℚ) = n + (21 : ℚ) / 6 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_and_average_l2483_248342


namespace NUMINAMATH_CALUDE_cubic_not_prime_l2483_248350

theorem cubic_not_prime (n : ℕ+) : ¬ Nat.Prime (n.val^3 - 7*n.val^2 + 16*n.val - 12) := by
  sorry

end NUMINAMATH_CALUDE_cubic_not_prime_l2483_248350


namespace NUMINAMATH_CALUDE_union_of_A_and_B_intersection_of_complements_l2483_248331

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def B : Set ℝ := {x | |2*x + 1| ≤ 1}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

-- Define the intersection of complements of A and B
def ACompIntBComp : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statements
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

theorem intersection_of_complements : Aᶜ ∩ Bᶜ = ACompIntBComp := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_intersection_of_complements_l2483_248331


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2483_248362

/-- Given a quadratic equation x^2 = 5x - 1, prove that its coefficients are 1, -5, and 1 -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 = 5*x - 1 ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -5 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2483_248362


namespace NUMINAMATH_CALUDE_abcd_sum_l2483_248383

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -2)
  (eq3 : a + c + d = 5)
  (eq4 : b + c + d = 4) :
  a * b + c * d = 26 / 9 := by
sorry

end NUMINAMATH_CALUDE_abcd_sum_l2483_248383


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2483_248341

theorem quadratic_no_real_roots (m : ℝ) :
  (∀ x : ℝ, x^2 - x + m ≠ 0) → m > 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2483_248341


namespace NUMINAMATH_CALUDE_intersection_slope_l2483_248356

/-- Given two lines p and q that intersect at (4, 11), 
    where p has equation y = 2x + 3 and q has equation y = mx + 1,
    prove that m = 2.5 -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 3 → y = m*x + 1 → x = 4 ∧ y = 11) →
  m = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l2483_248356


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2483_248320

def U : Set Int := {-1, -2, -3, -4, 0}
def A : Set Int := {-1, -2, 0}
def B : Set Int := {-3, -4, 0}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {-3, -4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2483_248320


namespace NUMINAMATH_CALUDE_existence_of_decreasing_lcm_sequence_l2483_248365

theorem existence_of_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_decreasing_lcm_sequence_l2483_248365


namespace NUMINAMATH_CALUDE_triangle_side_length_l2483_248378

theorem triangle_side_length (a b c : ℝ) (area : ℝ) : 
  a = 1 → b = Real.sqrt 7 → area = Real.sqrt 3 / 2 → 
  (c = 2 ∨ c = 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2483_248378


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2483_248345

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2483_248345


namespace NUMINAMATH_CALUDE_exists_quadratic_function_l2483_248399

/-- A quadratic function that fits the given points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem exists_quadratic_function : 
  ∃ (a b c : ℝ), 
    quadratic_function a b c 1 = 1 ∧
    quadratic_function a b c 2 = 4 ∧
    quadratic_function a b c 4 = 16 ∧
    quadratic_function a b c 5 = 25 ∧
    quadratic_function a b c 7 = 49 ∧
    quadratic_function a b c 8 = 64 ∧
    quadratic_function a b c 10 = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_quadratic_function_l2483_248399


namespace NUMINAMATH_CALUDE_breakfast_egg_scramble_time_l2483_248376

/-- Calculates the time to scramble each egg given the breakfast preparation parameters. -/
def time_to_scramble_egg (num_sausages : ℕ) (num_eggs : ℕ) (time_per_sausage : ℕ) (total_time : ℕ) : ℕ :=
  let time_for_sausages := num_sausages * time_per_sausage
  let time_for_eggs := total_time - time_for_sausages
  time_for_eggs / num_eggs

/-- Proves that the time to scramble each egg is 4 minutes given the specific breakfast parameters. -/
theorem breakfast_egg_scramble_time :
  time_to_scramble_egg 3 6 5 39 = 4 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_egg_scramble_time_l2483_248376


namespace NUMINAMATH_CALUDE_sweater_price_proof_l2483_248397

/-- Price of a T-shirt in dollars -/
def t_shirt_price : ℝ := 8

/-- Price of a jacket before discount in dollars -/
def jacket_price : ℝ := 80

/-- Discount rate for jackets -/
def jacket_discount : ℝ := 0.1

/-- Sales tax rate -/
def sales_tax : ℝ := 0.05

/-- Number of T-shirts purchased -/
def num_tshirts : ℕ := 6

/-- Number of sweaters purchased -/
def num_sweaters : ℕ := 4

/-- Number of jackets purchased -/
def num_jackets : ℕ := 5

/-- Total cost including tax in dollars -/
def total_cost : ℝ := 504

/-- Price of a sweater in dollars -/
def sweater_price : ℝ := 18

theorem sweater_price_proof :
  (num_tshirts * t_shirt_price +
   num_sweaters * sweater_price +
   num_jackets * jacket_price * (1 - jacket_discount)) *
  (1 + sales_tax) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_proof_l2483_248397


namespace NUMINAMATH_CALUDE_remainder_sum_product_l2483_248384

theorem remainder_sum_product (X Y Z E S T U s t q : ℕ) 
  (hX : X > Y) (hY : Y > Z)
  (hS : X % E = S) (hT : Y % E = T) (hU : Z % E = U)
  (hs : (X * Y * Z) % E = s) (ht : (S * T * U) % E = t)
  (hq : (X * Y * Z + S * T * U) % E = q) :
  q = (2 * s) % E :=
sorry

end NUMINAMATH_CALUDE_remainder_sum_product_l2483_248384


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2483_248312

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y = 1) :
  (1/x + 1/y) ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2483_248312


namespace NUMINAMATH_CALUDE_break_even_price_per_lot_l2483_248377

/-- Given a land purchase scenario, calculate the break-even price per lot -/
theorem break_even_price_per_lot (acres : ℕ) (price_per_acre : ℕ) (num_lots : ℕ) :
  acres = 4 →
  price_per_acre = 1863 →
  num_lots = 9 →
  (acres * price_per_acre) / num_lots = 828 := by
  sorry

end NUMINAMATH_CALUDE_break_even_price_per_lot_l2483_248377


namespace NUMINAMATH_CALUDE_teacher_volunteers_count_l2483_248375

/-- Calculates the number of teacher volunteers for a school Christmas play. -/
def teacher_volunteers (total_needed : ℕ) (math_classes : ℕ) (students_per_class : ℕ) (more_needed : ℕ) : ℕ :=
  total_needed - (math_classes * students_per_class) - more_needed

/-- Theorem stating that the number of teacher volunteers is 13. -/
theorem teacher_volunteers_count : teacher_volunteers 50 6 5 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_teacher_volunteers_count_l2483_248375


namespace NUMINAMATH_CALUDE_pencil_price_l2483_248344

theorem pencil_price (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) (pen_price : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 510 →
  pen_price = 12 →
  (total_cost - num_pens * pen_price) / num_pencils = 2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_price_l2483_248344


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2483_248361

theorem arithmetic_proof : (139 + 27) * 2 + (23 + 11) = 366 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2483_248361


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2483_248359

/-- The complex equation whose roots define the points on the ellipse -/
def complex_equation (z : ℂ) : Prop :=
  (z + 1) * (z^2 + 6*z + 10) * (z^2 + 8*z + 18) = 0

/-- The set of solutions to the complex equation -/
def solution_set : Set ℂ :=
  {z : ℂ | complex_equation z}

/-- The condition that the solutions are in the form x_k + y_k*i with x_k and y_k real -/
axiom solutions_form : ∀ z ∈ solution_set, ∃ (x y : ℝ), z = x + y * Complex.I

/-- The unique ellipse passing through the points defined by the solutions -/
axiom exists_unique_ellipse : ∃! E : Set (ℝ × ℝ), 
  ∀ z ∈ solution_set, (z.re, z.im) ∈ E

/-- The eccentricity of the ellipse -/
def eccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the eccentricity of the ellipse is √(3/4) -/
theorem ellipse_eccentricity : 
  ∀ E : Set (ℝ × ℝ), (∀ z ∈ solution_set, (z.re, z.im) ∈ E) → 
    eccentricity E = Real.sqrt (3/4) := 
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2483_248359


namespace NUMINAMATH_CALUDE_tangent_equation_solutions_l2483_248323

open Real

theorem tangent_equation_solutions (t : ℝ) :
  cos t ≠ 0 →
  (tan t = (sin t ^ 2 + sin (2 * t) - 1) / (cos t ^ 2 - sin (2 * t) + 1)) ↔
  (∃ k : ℤ, t = π / 4 + π * k) ∨
  (∃ n : ℤ, t = arctan ((1 - Real.sqrt 5) / 2) + π * n) ∨
  (∃ l : ℤ, t = arctan ((1 + Real.sqrt 5) / 2) + π * l) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solutions_l2483_248323


namespace NUMINAMATH_CALUDE_coin_game_probability_l2483_248334

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor

/-- The number of rounds in the game -/
def numRounds : Nat := 5

/-- The initial number of coins each player has -/
def initialCoins : Nat := 5

/-- The number of coins transferred when green and red balls are drawn -/
def coinTransfer : Nat := 2

/-- The total number of balls in the urn -/
def totalBalls : Nat := 5

/-- The number of green balls in the urn -/
def greenBalls : Nat := 1

/-- The number of red balls in the urn -/
def redBalls : Nat := 1

/-- The number of blue balls in the urn -/
def blueBalls : Nat := 3

/-- Represents the state of the game after each round -/
structure GameState :=
  (coins : Player → Nat)

/-- The probability of a specific pair (green/red) occurring in one round -/
def pairProbability : ℚ := 1 / 20

/-- 
Theorem: The probability that each player has exactly 5 coins after 5 rounds is 1/3,200,000
-/
theorem coin_game_probability : 
  ∀ (finalState : GameState),
    (∀ p : Player, finalState.coins p = initialCoins) →
    (pairProbability ^ numRounds : ℚ) = 1 / 3200000 := by
  sorry


end NUMINAMATH_CALUDE_coin_game_probability_l2483_248334


namespace NUMINAMATH_CALUDE_max_distance_to_line_l2483_248337

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ × ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The distance between two points in ℝ² -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- A line in ℝ² represented by its equation -/
def line (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => a * p.1 + b * p.2 + c = 0

theorem max_distance_to_line :
  let l1 := line 1 1 (-1)
  let l2 := line 1 (-2) (-4)
  let p := intersection_point l1 l2
  ∀ k : ℝ,
    let l3 := line k (-1) (1 + 2*k)
    ∀ q : ℝ × ℝ,
      l3 q →
      distance p q ≤ 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l2483_248337


namespace NUMINAMATH_CALUDE_new_person_weight_specific_new_person_weight_l2483_248392

/-- Given a group of people, calculate the weight of a new person who causes the average weight to increase when replacing another person. -/
theorem new_person_weight (initial_size : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  let total_increase := initial_size * avg_increase
  replaced_weight + total_increase

/-- Prove that for the given conditions, the weight of the new person is 61.3 kg. -/
theorem specific_new_person_weight :
  new_person_weight 12 1.3 45.7 = 61.3 := by sorry

end NUMINAMATH_CALUDE_new_person_weight_specific_new_person_weight_l2483_248392


namespace NUMINAMATH_CALUDE_eric_jogging_time_l2483_248326

/-- Proves the time Eric spent jogging given his running time and return trip time -/
theorem eric_jogging_time 
  (total_time_to_park : ℕ) 
  (running_time : ℕ) 
  (jogging_time : ℕ) 
  (return_trip_time : ℕ) 
  (h1 : total_time_to_park = running_time + jogging_time)
  (h2 : running_time = 20)
  (h3 : return_trip_time = 90)
  (h4 : return_trip_time = 3 * total_time_to_park) :
  jogging_time = 10 := by
sorry

end NUMINAMATH_CALUDE_eric_jogging_time_l2483_248326


namespace NUMINAMATH_CALUDE_total_books_calculation_l2483_248387

theorem total_books_calculation (darryl_books : ℕ) (lamont_books : ℕ) (loris_books : ℕ) (danielle_books : ℕ) : 
  darryl_books = 20 →
  lamont_books = 2 * darryl_books →
  loris_books + 3 = lamont_books →
  danielle_books = lamont_books + darryl_books + 10 →
  darryl_books + lamont_books + loris_books + danielle_books = 167 := by
sorry

end NUMINAMATH_CALUDE_total_books_calculation_l2483_248387


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2483_248369

/-- Given an ellipse and a parabola with a common point, this theorem proves the range of parameter a. -/
theorem ellipse_parabola_intersection_range :
  ∀ (a x y : ℝ),
  (x^2 + 4*(y - a)^2 = 4) →  -- Ellipse equation
  (x^2 = 2*y) →              -- Parabola equation
  (-1 ≤ a ∧ a ≤ 17/8) :=     -- Range of a
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2483_248369


namespace NUMINAMATH_CALUDE_charles_city_population_l2483_248300

theorem charles_city_population (C G : ℕ) : 
  G + 119666 = C → 
  C + G = 845640 → 
  C = 482653 := by
sorry

end NUMINAMATH_CALUDE_charles_city_population_l2483_248300


namespace NUMINAMATH_CALUDE_odd_function_property_l2483_248393

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = x * (1 + x)) : 
  ∀ x > 0, f x = x * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l2483_248393


namespace NUMINAMATH_CALUDE_no_extremum_implies_a_nonnegative_l2483_248371

/-- A function that has no extremum on ℝ -/
def NoExtremum (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → f y ≠ f x ∨ (f y < f x ∧ f y > f x)

/-- The main theorem -/
theorem no_extremum_implies_a_nonnegative (a : ℝ) :
  NoExtremum (fun x => Real.exp x + a * x) → a ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_no_extremum_implies_a_nonnegative_l2483_248371


namespace NUMINAMATH_CALUDE_borrowed_sum_calculation_l2483_248346

/-- Proves that given a sum of money borrowed at 6% per annum simple interest, 
    if the interest after 6 years is Rs. 672 less than the borrowed sum, 
    then the borrowed sum is Rs. 1050. -/
theorem borrowed_sum_calculation (P : ℝ) : 
  (P * 0.06 * 6 = P - 672) → P = 1050 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sum_calculation_l2483_248346


namespace NUMINAMATH_CALUDE_range_of_m_l2483_248324

def A (m : ℝ) : Set ℝ := {x | x^2 + Real.sqrt m * x + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ Set.univ = ∅) → (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2483_248324


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_length_l2483_248368

/-- Given a rectangle, square, and equilateral triangle with the same perimeter,
    if the square's side length is 9 cm, the rectangle's shorter side is 6 cm. -/
theorem rectangle_shorter_side_length
  (rectangle : Real × Real)
  (square : Real)
  (equilateral_triangle : Real)
  (h1 : 2 * (rectangle.1 + rectangle.2) = 4 * square)
  (h2 : 2 * (rectangle.1 + rectangle.2) = 3 * equilateral_triangle)
  (h3 : square = 9) :
  min rectangle.1 rectangle.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_length_l2483_248368


namespace NUMINAMATH_CALUDE_team_ate_63_slices_l2483_248354

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  extraLarge : Nat
  large : Nat
  medium : Nat

/-- Represents the number of pizzas of each size -/
structure PizzaCounts where
  extraLarge : Nat
  large : Nat
  medium : Nat

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (counts : PizzaCounts) : Nat :=
  slices.extraLarge * counts.extraLarge +
  slices.large * counts.large +
  slices.medium * counts.medium

/-- Theorem stating that the team ate 63 slices of pizza -/
theorem team_ate_63_slices 
  (slices : PizzaSlices)
  (counts : PizzaCounts)
  (h1 : slices.extraLarge = 16)
  (h2 : slices.large = 12)
  (h3 : slices.medium = 8)
  (h4 : counts.extraLarge = 3)
  (h5 : counts.large = 2)
  (h6 : counts.medium = 1)
  (h7 : totalSlices slices counts - 17 = 63) :
  63 = totalSlices slices counts - 17 := by
  sorry

#eval totalSlices ⟨16, 12, 8⟩ ⟨3, 2, 1⟩ - 17

end NUMINAMATH_CALUDE_team_ate_63_slices_l2483_248354


namespace NUMINAMATH_CALUDE_unique_solution_l2483_248304

def equation (x y : ℤ) : Prop := 3 * x + y = 10

theorem unique_solution :
  (equation 2 4) ∧
  ¬(equation 1 6) ∧
  ¬(equation (-2) 12) ∧
  ¬(equation (-1) 11) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2483_248304


namespace NUMINAMATH_CALUDE_problem_solution_l2483_248390

theorem problem_solution (m n : ℝ) 
  (hm : 3 * m^2 + 5 * m - 3 = 0) 
  (hn : 3 * n^2 - 5 * n - 3 = 0) 
  (hmn : m * n ≠ 1) : 
  1 / n^2 + m / n - (5/3) * m = 25/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2483_248390


namespace NUMINAMATH_CALUDE_anthony_ate_two_bananas_l2483_248374

/-- The number of bananas Anthony bought -/
def initial_bananas : ℕ := 12

/-- The number of bananas Anthony has left -/
def remaining_bananas : ℕ := 10

/-- The number of bananas Anthony ate -/
def eaten_bananas : ℕ := initial_bananas - remaining_bananas

theorem anthony_ate_two_bananas : eaten_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_ate_two_bananas_l2483_248374


namespace NUMINAMATH_CALUDE_area_inequality_l2483_248305

/-- Triangle type -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Point on a line segment -/
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- Area of a triangle -/
noncomputable def TriangleArea (T : Triangle) : ℝ :=
  abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2)) / 2

/-- Theorem statement -/
theorem area_inequality (ABC : Triangle) (X Y Z : ℝ × ℝ) 
  (hX : PointOnSegment X ABC.B ABC.C)
  (hY : PointOnSegment Y ABC.C ABC.A)
  (hZ : PointOnSegment Z ABC.A ABC.B)
  (hBX : dist ABC.B X ≤ dist X ABC.C)
  (hCY : dist ABC.C Y ≤ dist Y ABC.A)
  (hAZ : dist ABC.A Z ≤ dist Z ABC.B) :
  4 * TriangleArea ⟨X, Y, Z⟩ ≥ TriangleArea ABC :=
sorry

end NUMINAMATH_CALUDE_area_inequality_l2483_248305


namespace NUMINAMATH_CALUDE_line_sum_m_b_l2483_248386

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) with slope m and y-intercept b -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ
  m : ℚ
  b : ℚ
  eq₁ : y₁ = m * x₁ + b
  eq₂ : y₂ = m * x₂ + b

/-- Theorem: For a line passing through (2, -1) and (5, 3), m + b = -7/3 -/
theorem line_sum_m_b :
  ∀ l : Line,
    l.x₁ = 2 ∧ l.y₁ = -1 ∧ l.x₂ = 5 ∧ l.y₂ = 3 →
    l.m + l.b = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_line_sum_m_b_l2483_248386


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l2483_248367

theorem choose_three_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l2483_248367


namespace NUMINAMATH_CALUDE_symmetric_points_imply_fourth_quadrant_l2483_248306

/-- Given two points A and B symmetric with respect to the y-axis, 
    prove that point C lies in the fourth quadrant. -/
theorem symmetric_points_imply_fourth_quadrant 
  (a b : ℝ) 
  (h_symmetric : (a - 2, 3) = (-(-1), b + 5)) : 
  (a > 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_imply_fourth_quadrant_l2483_248306
